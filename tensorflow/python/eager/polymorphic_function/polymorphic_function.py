# [MODULE DOC]
# Implements the user-facing tf.function machinery: the Function class (a polymorphic function
# that holds a cache of concrete traced graph functions and chooses one per call based on the
# argument types and values), the public function entry point used as a decorator or a plain
# wrapper, and helper classes supporting variable creation inside traced functions.
# Key pieces: a variable subtype that does not lift its initializer out of the function (used for
# the first trace so initializers can later be lifted or run conditionally), a detector that warns
# when a function is retraced too often, an optional XLA control-flow context wrapper, and a
# helper that rebinds a Function to an instance so decorated methods work.
# Import-time side effects: reads the environment setting controlling whether functions are
# compiled with XLA by default, creates a usage counter metric, and creates one module-level
# manager object that tracks retracing statistics for all functions. The public function is
# registered in the exported API under its public name, and the Function class under an internal
# API name.
# Part of the eager polymorphic_function package; delegates actual tracing and calling to the
# tracing_compilation module.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# pylint: disable=unidiomatic-typecheck
"""API for defining graph functions with some additional eager semantics.

tf.function utilizes varying configurations of tracing compilation to allow
initializing `tf.Variable`s with subgraphs of the function. For example:

```python
class M(tf.Module):
  def __init__(self):
    self.v_opinit = None
    self.v_arginit = None

  @tf.function
  def __call__(self, x):
    # Variables are only created on the first call to the function. This is a
    # common pattern in layer libraries.
    if self.v_opinit is None:
      # self.v_opinit will outlive the function call, but `tf.ones` is traced as
      # part of the function body before the `tf.Variable` object is
      # created. This subgraph is easy to lift out of the function.
      self.v_opinit = tf.Variable(tf.ones([]))

      # If arguments feed into variable initialization, it can be very tricky to
      # disentangle from the rest of the function. We don't attempt it.
      self.v_arginit = tf.Variable(tf.ones(tf.shape(x)) * tf.constant(2.))
    return self.v_opinit + self.v_arginit + x
```

These patterns using tracing compilation directly throw an error asking
the user to put the variable's initializer in a lambda. With tf.function they
work with eager semantics either by lifting the subgraph out of the function and
using it to initialize the variable, or by initializing variables on the first
call to the function (if they weren't already initialized by something else,
e.g. a checkpoint API). The latter requires tf.conds, and is not well supported
by TF-XLA, so we only do it when necessary.

Since these patterns are relatively common in layer libraries, we expose the
wrapper in this file as `tf.function`. The defun concept in quarantine.py is a
legacy internal API.

In order to support these variable initialization patterns, tf.function defines
a variable subtype (UnliftedInitializerVariable) which collects the input
subgraph. This type of variable replaces the regular variable type on the first
tf.function trace. To exclude initializers from the function body (the `tf.ones`
ops above and associated assignment operations), tf.function traces a second
time if it sees variables on the first call.
"""

import dataclasses
import functools
import os
import threading
import types as types_lib
import warnings
import weakref

from google.protobuf import text_format as _text_format
from google.protobuf.message import DecodeError
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.function.capture import capture_container
from tensorflow.core.function.polymorphism import function_cache
from tensorflow.python.distribute.parallel_device import parallel_device
from tensorflow.python.eager import context
from tensorflow.python.eager import lift_to_graph
from tensorflow.python.eager import monitoring
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import autograph_util
from tensorflow.python.eager.polymorphic_function import compiler_ir
from tensorflow.python.eager.polymorphic_function import eager_function_run
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.eager.polymorphic_function import tf_method_target
from tensorflow.python.eager.polymorphic_function import tracing_compilation
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import traceback_utils
from tensorflow.python.util.tf_export import tf_export

FREQUENT_TRACING_WARNING_MAX_CALL_HISTORY = 10
FREQUENT_TRACING_WARNING_THRESHOLD = 5
FREQUENT_TRACING_WARNING_MAX_WARNING_PER_DETECTOR = 2

_tf_function_counter = monitoring.Counter(
    "/tensorflow/core/tf_function_counter",
    "Counter for the number of tf.functions created when Eager execution is "
    "enabled.",
    # jit_compile is "0" or "1".
    "jit_compile")


class _FrequentTracingDetector(object):
  # [CLASS DOC]
  # HELPER: Supports Function in warning users about excessive retracing.
  # Keeps a sliding window of recent calls of one function, grouped by tracing events (each entry
  # is the number of calls since a tracing call, starting at one for the tracing call itself).
  # Holds the window list, a running call count and the number of warnings already emitted.
  # Emits at most a fixed number of warnings per detector.
  """Class keeping track of how many recent calls triggered tracing."""

  __slots__ = ["_calls_per_tracings", "_call_count", "_total_warning_count"]

  def __init__(self):
    # [FUNCTION DOC]
    # HELPER: Constructor of the retracing detector.
    # Takes no inputs. Starts with an empty call window, zero warnings emitted and zero calls
    # counted. Returns nothing.
    self._calls_per_tracings = []
    self._total_warning_count = 0
    self._call_count = 0

  def called_with_tracing(self, function_name, omit_warning):
    # [FUNCTION DOC]
    # HELPER: Records a tracing call for the retracing warning logic.
    # Inputs: function_name is the python function (or its name) shown in the warning text;
    # omit_warning is a boolean that suppresses the warning when true.
    # Appends a new window entry, then drops old entries from the front while the recorded call
    # count minus the oldest entry's count exceeds the history limit. If warnings are not
    # omitted, the per-detector warning limit is not reached and the window holds at least the
    # threshold number of tracing events, logs a warning about excessive retracing and
    # increments the warning counter. Returns nothing; mutates internal state and logs.
    """Updates the list of most recent calls' tracing information.

    Warns the user when recent calls caused retracing too often.

    Args:
      function_name: the python function being traced.
      omit_warning: If 'True', this call will not warn the user even if
        retracing happens too often.
    """
    self._call_count += 1
    self._calls_per_tracings.append(1)

    while self._calls_per_tracings:
      if (self._call_count - self._calls_per_tracings[0] >
          FREQUENT_TRACING_WARNING_MAX_CALL_HISTORY):
        self._call_count -= self._calls_per_tracings.pop(0)
      else:
        break

    if (omit_warning or self._total_warning_count >=
        FREQUENT_TRACING_WARNING_MAX_WARNING_PER_DETECTOR):
      return
    if len(self._calls_per_tracings) >= FREQUENT_TRACING_WARNING_THRESHOLD:
      self._total_warning_count += 1
      logging.warning(
          "{} out of the last {} calls to {} triggered tf.function "
          "retracing. Tracing is expensive and the excessive number of "
          "tracings could be due to (1) creating @tf.function repeatedly in "
          "a loop, (2) passing tensors with different shapes, (3) passing "
          "Python objects instead of tensors. For (1), please define your "
          "@tf.function outside of the loop. For (2), @tf.function has "
          "reduce_retracing=True option that can avoid unnecessary "
          "retracing. For (3), please refer to "
          "https://www.tensorflow.org/guide/function#controlling_retracing"
          " and https://www.tensorflow.org/api_docs/python/tf/function for "
          " more details.".format(
              len(self._calls_per_tracings), self._call_count, function_name))

  def called_without_tracing(self):
    # [FUNCTION DOC]
    # HELPER: Records a non-tracing (cache hit) call for the retracing warning logic.
    # Takes no inputs and returns nothing. If no window entry exists yet (for example when the
    # first call used an already loaded concrete function), creates an entry with zero, then
    # increments the latest entry and the total call count.
    # We don't count tracing when users load a concrete function directly or
    # call get_concrete_function, so the first call can be not a tracing call.
    if not self._calls_per_tracings:
      self._calls_per_tracings = [0]
    self._calls_per_tracings[-1] += 1
    self._call_count += 1


class _FrequentTracingDetectorManager(object):
  # [CLASS DOC]
  # HELPER: Registry of retracing detectors used by Function.
  # Holds a weak-keyed mapping from a key object (typically a function's code object) to its
  # detector, plus a lock guarding the mapping. Detectors are created lazily on first use and
  # disappear when the key is garbage collected. One module-level instance is shared by all
  # functions. All public methods are thread-safe.
  """Class for the management of all _FrequentTracingDetector objects."""

  __slots__ = ["_detectors", "_lock"]

  def __init__(self):
    # [FUNCTION DOC]
    # HELPER: Constructor of the detector registry.
    # Takes no inputs; creates an empty weak-keyed mapping of detectors and a lock. Returns nothing.
    self._detectors = weakref.WeakKeyDictionary()  # GUARDED_BY(self._lock)
    self._lock = threading.Lock()

  def _get_detector(self, key):
    # [FUNCTION DOC]
    # HELPER: Looks up or lazily creates the detector for a key.
    # Input: key, a weakly referenceable object identifying a function. Returns the detector
    # stored for it, creating and storing a new one if absent. Callers must hold the lock;
    # this method does not take it itself.
    if key not in self._detectors:
      self._detectors[key] = _FrequentTracingDetector()
    return self._detectors[key]

  def called_without_tracing(self, key):
    # [FUNCTION DOC]
    # HELPER: Thread-safe forwarding of a non-tracing call record to the key's detector.
    # Input: key identifying the function. Under the lock, gets or creates the detector and
    # tells it a call happened without tracing. Returns nothing.
    with self._lock:
      detector = self._get_detector(key)
      detector.called_without_tracing()

  def called_with_tracing(self, key, function_name, omit_warning):
    # [FUNCTION DOC]
    # HELPER: Thread-safe forwarding of a tracing call record to the key's detector.
    # Inputs: key identifying the function; function_name used in any warning text;
    # omit_warning, a boolean that suppresses warnings. Under the lock, gets or creates the
    # detector and records the tracing call (which may log a retracing warning). Returns nothing.
    with self._lock:
      detector = self._get_detector(key)
      detector.called_with_tracing(function_name, omit_warning)


_frequent_tracing_detector_manager = _FrequentTracingDetectorManager()


class UnliftedInitializerVariable(resource_variable_ops.UninitializedVariable):
  # [CLASS DOC]
  # Resource variable created during the first trace of a tf.function whose initializer is
  # evaluated inside the function graph instead of being lifted out. It subclasses the
  # uninitialized resource variable, creating the handle without a value, and then either lifts
  # the initial value into the outer graph (legacy graph mode), assigns it immediately (eager),
  # or records it in a caller-supplied list and adds a conditional assignment that runs only if
  # the variable is not yet initialized. Instances are made by the variable creator scope that
  # Function installs while tracing; direct use outside a function falls back to a regular
  # resource variable. Graph-only (non-function) graphs are not supported.
  """Variable which does not lift its initializer out of function context.

  Instances of this variable, when created, build a graph which runs their
  initializer inside a tf.cond(is_initialized) block.

  This can only be created during tracing compilation called from
  (eventually) eager mode. That is, non-function-building graphs are not
  supported.
  """

  def __init__(
      self,
      initial_value=None,
      trainable=None,
      caching_device=None,
      name=None,
      dtype=None,
      constraint=None,
      add_initializers_to=None,
      synchronization=None,
      aggregation=None,
      shape=None,
      **unused_kwargs,
  ):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring says it raises a runtime error when called outside a function
    # and mentions a validate-shape option; the code never raises that error (outside a function
    # it silently delegates to the regular resource variable constructor, ignoring the
    # synchronization, aggregation, shape and add_initializers_to arguments) and has no
    # validate-shape parameter.
    # Creates the variable. Inputs: initial_value (tensor-convertible value or a no-argument
    # callable producing it; required), trainable, caching_device, name, dtype, constraint
    # (optional callable), add_initializers_to (optional list that receives pairs of variable
    # and initial value when the initializer must be lifted later), synchronization,
    # aggregation, shape (defaults to the initial value's shape) and extra keyword arguments
    # forwarded to the parent. Raises a value error if initial_value is missing or constraint is
    # not callable. Side effects: builds ops in the current graph (initializer under a name
    # scope, conditional assign op), may append to add_initializers_to, may capture the handle.
    """Creates a variable.

    Args:
      initial_value: A `Tensor`, or Python object convertible to a `Tensor`,
        which is the initial value for the Variable. The initial value must have
        a shape specified unless `validate_shape` is set to False. Can also be a
        callable with no argument that returns the initial value when called.
        (Note that initializer functions from init_ops.py must first be bound to
        a shape before being used here.)
      trainable: If `True`, GradientTapes automatically watch uses of this
        Variable.
      caching_device: Optional device string or function describing where the
        Variable should be cached for reading.  Defaults to the Variable's
        device.  If not `None`, caches on another device.  Typical use is to
        cache on the device where the Ops using the Variable reside, to
        deduplicate copying through `Switch` and other conditional statements.
      name: Optional name for the variable. Defaults to `'Variable'` and gets
        uniquified automatically.
      dtype: If set, initial_value will be converted to the given type. If None,
        either the datatype will be kept (if initial_value is a Tensor) or
        float32 will be used (if it is a Python object convertible to a Tensor).
      constraint: An optional projection function to be applied to the variable
        after being updated by an `Optimizer` (e.g. used to implement norm
        constraints or value constraints for layer weights). The function must
        take as input the unprojected Tensor representing the value of the
        variable and return the Tensor for the projected value (which must have
        the same shape). Constraints are not safe to use when doing asynchronous
        distributed training.
      add_initializers_to: if not None and not in legacy graph mode, the
        initializer tensor will be added to this map in addition to adding the
        assignment to the function.
      synchronization: Indicates when a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableSynchronization`. By default the synchronization is set to
        `AUTO` and the current `DistributionStrategy` chooses when to
        synchronize.
      aggregation: Indicates how a distributed variable will be aggregated.
        Accepted values are constants defined in the class
        `tf.VariableAggregation`.
      shape: (optional) The shape of this variable. If None, the shape of
        `initial_value` will be used. When setting this argument to
        `tf.TensorShape(None)` (representing an unspecified shape), the variable
        can be assigned with values of different shapes.

    Raises:
      ValueError: If the initial value is not specified, or does not have a
        shape and `validate_shape` is `True`.
      RuntimeError: If called outside of a function definition.
    """
    with ops.init_scope():
      self._in_graph_mode = not context.executing_eagerly()
    if not ops.inside_function():
      # If we've been init_scope()d out of the function definition nothing to do
      # here; we can't really do the capturing or conditional logic.
      resource_variable_ops.ResourceVariable.__init__(
          self, initial_value=initial_value, trainable=trainable,
          caching_device=caching_device, name=name, dtype=dtype,
          constraint=constraint)
      return
    if initial_value is None:
      raise ValueError("`initial_value` must be a Tensor or a Python "
                       "object convertible to a Tensor. Got None.")
    init_from_fn = callable(initial_value)

    if constraint is not None and not callable(constraint):
      raise ValueError(f"`constraint` with type {type(constraint)} must be a "
                       "callable.")

    with ops.name_scope(name, "Variable", []
                        if init_from_fn else [initial_value]) as scope_name:
      with ops.name_scope("Initializer"):
        if init_from_fn:
          initial_value = initial_value()
        if isinstance(initial_value, trackable.CheckpointInitialValue):
          self._maybe_initialize_trackable()
          self._update_uid = initial_value.checkpoint_position.restore_uid
          initial_value = initial_value.wrapped_value

        initial_value = ops.convert_to_tensor(initial_value,
                                              name="initial_value", dtype=dtype)
      assert initial_value is not None

      # Don't use `shape or initial_value.shape` since TensorShape has
      # overridden `__bool__`.
      if shape is None:
        shape = initial_value.shape

    # Use the constructor for UninitializedVariable to start. Outside the name
    # scope so we don't double up the prefix.
    super().__init__(
        trainable=trainable,
        caching_device=caching_device,
        name=name,
        shape=shape,
        dtype=initial_value.dtype,
        constraint=constraint,
        synchronization=synchronization,
        aggregation=aggregation,
        extra_handle_data=initial_value,
        **unused_kwargs)

    with ops.name_scope(scope_name):
      if self._in_graph_mode:
        with ops.init_scope():
          outer_graph = ops.get_default_graph()
        func_graph = ops.get_default_graph()
        function_placeholders = (
            func_graph.inputs + func_graph.internal_captures)
        placeholder_ops = set(
            [tensor.op for tensor in function_placeholders])
        lifted_initializer = lift_to_graph.lift_to_graph(
            [initial_value], outer_graph,
            disallowed_placeholders=placeholder_ops)[initial_value]
        with ops.init_scope():
          self._initial_value = lifted_initializer
          with ops.name_scope("IsInitialized"):
            self._is_initialized_op = (
                resource_variable_ops.var_is_initialized_op(self._handle))
          if initial_value is not None:
            with ops.name_scope("Assign") as n, ops.colocate_with(self._handle):
              self._initializer_op = resource_variable_ops.assign_variable_op(
                  self._handle, lifted_initializer, name=n)
      elif context.executing_eagerly():
        # In this case, both current scope and init scope are eager.
        # Assign_variable_op will be executed immediately. So we don't need to
        # add it to "add_initializers_to" to lift it out.
        with ops.name_scope("Assign") as n, ops.colocate_with(self._handle):
          resource_variable_ops.assign_variable_op(
              self._handle, initial_value, name=n)
      else:
        # Init scope is eager but current scope is graph. We will lift out this
        # variable by addint it into "add_initializers_to".
        if add_initializers_to is not None:
          add_initializers_to.append((self, initial_value))

        def assign_fn():
          # [FUNCTION DOC]
          # HELPER: Branch of the initialization conditional that runs when the variable is not
          # yet initialized. Assigns the initial value to the variable's handle (colocated with
          # it) and returns a constant one tensor so both branches return matching structures.
          with ops.name_scope("Assign") as n, ops.colocate_with(self._handle):
            resource_variable_ops.assign_variable_op(
                self._handle,
                initial_value,
                name=n)
            # Returning values to keep tf.cond happy.
          return ops.convert_to_tensor(1)
        def not_assign_fn():
          # [FUNCTION DOC]
          # HELPER: Branch of the initialization conditional that runs when the variable is
          # already initialized. Does nothing and returns a constant zero tensor.
          return ops.convert_to_tensor(0)
        # Note: this cond is always guaranteed to run because we're inside
        # tracing compilation which will insert automatic control dependencies.
        # It will only execute assign_fn if lifting failed.
        graph = ops.get_default_graph()

        # Capture the handle ahead of time in order to avoid querying the shape
        # of the handle which helps async execution performance
        graph.capture(self._handle, shape=())
        cond.cond(
            resource_variable_ops.var_is_initialized_op(self._handle),
            not_assign_fn, assign_fn)


JIT_COMPILE_FUNCTIONS = (
    os.getenv("TF_FUNCTION_JIT_COMPILE_DEFAULT", "false").lower()
    in ("true", "1"))


def _evaluate_var_is_initialized(variables):
  # [FUNCTION DOC]
  # HELPER: Used by Function to decide which lifted variable initializers still need to run.
  # Input: variables, a list of resource variables. Runs outside any function (init scope),
  # builds an is-initialized op per variable, stacks them and reads the values to the host in
  # one transfer. If the device cannot copy to host, falls back to per-variable reads; for
  # variables on a parallel device it unpacks the per-replica values and requires them to be
  # either all initialized or all uninitialized. Returns a sequence of booleans, one per
  # variable. Raises a not-implemented error when only some replicas are initialized.
  """Compute booleans indicating whether each variable is initialized."""
  with ops.init_scope():
    var_is_initialized = []
    for v in variables:
      var_is_initialized.append(
          resource_variable_ops.var_is_initialized_op(v.handle))
    try:
      # Stack all the var_is_initialized values into one tensor and interpret
      # the numpy value. This will reduce the number of RPCs between client and
      # worker in the remote case.
      return array_ops_stack.stack(var_is_initialized).numpy()
    except errors.UnimplementedError:
      # Some devices do not support implicit copy-off to host. Fall back to
      # variable-by-variable processing.
      for index, v in enumerate(variables):
        try:
          numpy_value = var_is_initialized[index].numpy()
        except errors.UnimplementedError:
          # This is a variable on a parallel device; we'll extract its value on
          # each replica and assert that they're identical.
          components = parallel_device.unpack(var_is_initialized[index])
          with ops.device(None):
            components = array_ops_stack.stack(components)
            all_initialized = math_ops.reduce_all(components).numpy()
            any_initialized = math_ops.reduce_any(components).numpy()
          if all_initialized != any_initialized:
            raise NotImplementedError(
                f"Some but not all components of a parallel variable {v!r} "
                "were initialized between their creation in a tf.function and "
                "the function's trace having completed. This is not "
                "supported; consider initializing either all or none of the "
                "components, or moving initialization out of the function.")
          numpy_value = all_initialized
        var_is_initialized[index] = numpy_value
  return var_is_initialized


class OptionalXlaContext:
  # [CLASS DOC]
  # Context manager that enters an XLA control-flow context only when requested and when the
  # current graph is not already inside an XLA context. Holds the context object or nothing.
  # Used by Function when calling or tracing with jit compilation enabled.
  """Wrapper for XLA context optionally applied under a context manager."""

  def __init__(self, is_compiled):
    # [FUNCTION DOC]
    # Input: is_compiled, truthy when XLA compilation is requested. Creates an XLA control-flow
    # context only if compilation is requested and neither the default graph nor its parents are
    # already in an XLA context; otherwise stores nothing. Returns nothing.
    wrap = is_compiled and not control_flow_util.GraphOrParentsInXlaContext( \
              ops.get_default_graph())
    self.xla_context = control_flow_ops.XLAControlFlowContext() \
        if wrap else None

  def __enter__(self):
    # [FUNCTION DOC]
    # Enters the stored XLA context if one exists; otherwise does nothing. Returns nothing
    # (so a with-statement target would be empty).
    if self.xla_context:
      self.xla_context.Enter()

  def __exit__(self, t, value, traceback):
    # [FUNCTION DOC]
    # Inputs: the exception type, value and traceback (ignored). Exits the stored XLA context if
    # one exists. Returns nothing, so exceptions are never suppressed.
    if self.xla_context:
      self.xla_context.Exit()


@tf_export("__internal__.function.Function", v1=[])
class Function(core.PolymorphicFunction, trackable.Trackable):
  # [CLASS DOC]
  # The object returned by the tf.function decorator. Wraps a python callable and lazily traces
  # it into concrete graph functions keyed by argument types, caching them in a function cache.
  # State: the wrapped python function, its function type and default values, function cache and
  # captures, attributes for generated function definitions, autograph and jit-compile settings,
  # a reentrant lock, and two tracing configurations: one allowing variable creation (used for
  # the first call) and one forbidding it (used afterwards). Variables created on the first call
  # are tracked by weak reference. Also supports descriptor binding for methods, pickling
  # (dropping the lock and caches), and the trackable protocol for saving concrete functions.
  # Implements the polymorphic function interface: calling, getting concrete functions and
  # compiler IR.
  """A `tf.types.experimental.PolymorphicFunction` created by `tf.function`.

  Currently, individual methods/attributes under this class are not guaranteed
  by the TF API contract, and are subject to future changes.

  (Previously also known as `tf.types.experimental.GenericFunction`)
  """

  def __init__(self,
               python_function,
               name,
               input_signature=None,
               autograph=True,
               jit_compile=None,
               reduce_retracing=False,
               experimental_implements=None,
               experimental_autograph_options=None,
               experimental_attributes=None,):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring says a value error is raised when the input signature is given
    # and the function has keyword arguments; this constructor does not check that (the function
    # type builder does). The only error raised here is for an unsupported attribute, and its
    # message is missing a closing quote mark.
    # Inputs: python_function (callable to wrap), name (string), input_signature (optional
    # list of tensor specs), autograph (bool, default true), jit_compile (optional bool),
    # reduce_retracing (bool), experimental_implements (optional string or text-format
    # attribute list), experimental_autograph_options (optional features), experimental_attributes
    # (optional mapping merged into attributes). Builds the function type, cache and captures,
    # validates attribute names against the allow-list (raising a value error otherwise), and
    # marks the tf.function API gauge as used. Returns nothing.
    """Initializes a `Function`.

    Args:
      python_function: the function to be wrapped.
      name: the name given to it.
      input_signature: See the documentation for `tf.function`.
      autograph: See the documentation for `tf.function`.
      jit_compile: See the documentation for `tf.function`.
      reduce_retracing: See the documentation for `tf.function`.
      experimental_implements: See the documentation for `tf.function`.
      experimental_autograph_options: See the documentation for `tf.function`.
      experimental_attributes: See the documentation for `tf.function`.

    Raises:
      ValueError: if `input_signature` is not None and the `python_function`'s
        argspec has keyword arguments.
    """
    self._lock = threading.RLock()
    self._python_function = python_function
    self._function_type, self._default_values = (
        function_type_utils.make_function_type(python_function, input_signature)
    )
    self._function_cache = function_cache.FunctionCache()
    self._function_captures = capture_container.FunctionCaptures()

    self._attributes = {}
    if experimental_implements is not None:
      self._attributes = self._create_implements_attribute(
          experimental_implements
      )

    if experimental_attributes is not None:
      self._attributes.update(experimental_attributes)

    for attribute in self._attributes:
      if attribute not in attributes_lib.POLYMORPHIC_FUNCTION_ALLOWLIST:
        raise ValueError(
            f"`{attribute} is not supported by tf.function as an attribute."
        )

    self._is_pure = (
        self._attributes and attributes_lib.IMPLEMENTS in self._attributes
    )
    # If `True`, the function uses the rendezvous of the parent. This is only
    # needed to support code where raw send/recv operations are inserted and
    # when functions are run in graph mode where they may not be inlined.
    self._shared_rendezvous = None
    self._autograph = autograph
    self._experimental_autograph_options = experimental_autograph_options
    self._reduce_retracing = reduce_retracing
    self._jit_compile = jit_compile
    self._created_variables = None  # GUARDED_BY(self._lock)
    self._variable_creation_config = None  # GUARDED_BY(self._lock)
    self._no_variable_creation_config = None  # GUARDED_BY(self._lock)
    self._descriptor_cache = weakref.WeakKeyDictionary()
    self._name = name
    self._key_for_call_stats = self._get_key_for_call_stats()
    self._omit_frequent_tracing_warning = False
    ops._tf_function_api_gauge.get_cell().set(True)  # pylint: disable=protected-access

  @property
  def name(self):
    # [FUNCTION DOC]
    # Read-only property returning the name given to this function.
    return self._name

  def __getstate__(self):
    # [FUNCTION DOC]
    # Pickling support. Returns a copy of the instance state without the lock, descriptor cache
    # and call-stats key, which cannot be pickled. Does not modify the instance.
    """Custom pickling, to omit unpickleable objects."""
    result = self.__dict__.copy()
    del result["_lock"]
    del result["_descriptor_cache"]
    del result["_key_for_call_stats"]
    return result

  def __setstate__(self, state):
    # [FUNCTION DOC]
    # Unpickling support. Input: state, the mapping produced by the state getter. Replaces the
    # instance state with it and recreates the lock, an empty weak descriptor cache and the
    # call-stats key. Returns nothing.
    """Restore from pickled state."""
    self.__dict__ = state
    self._lock = threading.RLock()
    self._descriptor_cache = weakref.WeakKeyDictionary()
    self._key_for_call_stats = self._get_key_for_call_stats()

  def _get_key_for_call_stats(self):
    # [FUNCTION DOC]
    # HELPER: Computes the key under which retracing statistics are tracked.
    # Unwraps the python function through any chain of wrapped-function links, then through a
    # bound-method link, and returns its code object if it has one, otherwise the original
    # python function. Using the code object keeps statistics shared across re-created wrappers
    # of the same source function. Takes no inputs; no side effects.
    """Returns key instance to track call stats and retracings.

    The key instance a best-effort to preserve global consistency.
    """
    target_function = self._python_function
    # `__wrapped__` is a conventional Python attribute that a higher-order
    # function keeps its original function's instance.  We also directly use
    # this attribute for dealing with a class method.  See
    # `bound_method_wrapper` in `function.py`.  If we don't use `__wrapped__`,
    # all class methods will return the same `bound_method_wrapper` instance
    # from this function.
    while hasattr(target_function, "__wrapped__"):
      target_function = target_function.__wrapped__

    if hasattr(target_function, "__func__"):
      target_function = target_function.__func__

    if hasattr(target_function, "__code__"):
      return target_function.__code__

    return self._python_function

  def _generate_scoped_tracing_options(self, scope, scope_type):
    # [FUNCTION DOC]
    # HELPER: Builds tracing options whose traced function runs inside a variable creator scope.
    # Inputs: scope, a variable creator callback to install at reduced priority while tracing;
    # scope_type, whether the trace permits or forbids variable creation. Wraps the python
    # function so each invocation enters the scope (and an optional XLA context when jit
    # compilation is on), holding only a weak reference to the wrapper to avoid a reference
    # cycle. Returns the tracing options for the wrapped function.
    """Creates TracingOptions for variable creator scopes."""

    weak_wrapped_fn = None
    compile_with_xla = self._jit_compile

    def wrapped_fn(*args, **kwds):
      # [FUNCTION DOC]
      # HELPER: Traced stand-in for the python function. Takes the same positional and keyword
      # arguments, registers the captured variable creator scope on the default graph with
      # priority 50, optionally enters an XLA context, calls the (possibly autograph-converted)
      # wrapped function and returns its result.
      """Wraps `self._python_function` in a variable creator scope."""
      # We register a variable creator with reduced priority. If an outer
      # variable creator is just modifying keyword arguments to the variable
      # constructor, this will work harmoniously. Since the `scope` registered
      # here actually creates the variable, it taking priority would otherwise
      # ignore the outer creator.
      #
      # If an outer variable creator calls the variable constructor manually,
      # for example creating a MirroredVariable, then they won't call our
      # creator. This means we won't be able to trace the initialization graph,
      # and so variable initializers can't depend on function arguments. This is
      # better than the alternative, tracing the initialization graph but giving
      # the user a variable type they didn't want.
      default_graph = ops.get_default_graph()
      with default_graph._variable_creator_scope(scope, priority=50):  # pylint: disable=protected-access
        # __wrapped__ allows AutoGraph to swap in a converted function. We give
        # the function a weak reference to itself to avoid a reference cycle.
        with OptionalXlaContext(compile_with_xla):
          out = weak_wrapped_fn().__wrapped__(*args, **kwds)
        return out

    weak_wrapped_fn = weakref.ref(wrapped_fn)

    return self._generate_tracing_options(tf_decorator.make_decorator(
        self._python_function,
        wrapped_fn), scope_type)

  def _create_implements_attribute(self, implements_arg):
    # [FUNCTION DOC]
    # HELPER: Converts the experimental-implements argument into an attribute mapping.
    # KNOWN ISSUE: If implements_arg is not a string, the result is an empty mapping and the
    # argument is silently ignored; no error is raised.
    # Input: implements_arg, a string that is either a plain function name or a text-format
    # attribute-list message. Tries to parse it as the message and stores it as a function
    # attribute value; if parsing fails, stores the raw string. Returns a mapping with at most
    # one entry keyed by the implements attribute name.
    """Creates the attribute value corresponding to attribute_lib.IMPLEMENTS."""
    attributes = {}
    if isinstance(implements_arg, str):
      # First check if the attribute_lib.IMPLEMENTS is specified as a
      # NameAttrList. This is used when apart from the function name being
      # implemented, a list of attributes is also being specified.
      # The attributes are specified as key-value pairs in the NameAttrList
      # of the corresponding AttrValue. The function name will be in the
      # 'name' field of the NameAttrList. Else, it is just a string
      # corresponding to the function name.
      try:
        attr_value = attr_value_pb2.AttrValue()
        nameattrlist = attr_value_pb2.NameAttrList()
        _text_format.Merge(implements_arg, nameattrlist)
        attr_value.func.CopyFrom(nameattrlist)
        attributes[attributes_lib.IMPLEMENTS] = attr_value
      except (_text_format.ParseError, DecodeError):
        attributes[attributes_lib.IMPLEMENTS] = implements_arg
    return attributes

  def _generate_tracing_options(self, fn, scope_type):
    # [FUNCTION DOC]
    # HELPER: Creates the tracing options object for tracing a given callable.
    # Inputs: fn, the callable to trace; scope_type, variable-creation scope kind. Copies the
    # function attributes, adds the shared-rendezvous attribute if set, adds the XLA-compile
    # attribute (and no-inline when compiling) if jit_compile was specified, and converts fn
    # with autograph when enabled. Returns options carrying the function type, defaults, cache,
    # captures, lock, autograph and retracing settings. Does not mutate stored attributes.
    """Return a TracingOptions catered to the input function."""
    attributes = self._attributes.copy()

    share = self._shared_rendezvous
    if share is not None:
      attributes[attributes_lib.SHARED_RENDEZVOUS] = share

    if self._jit_compile is not None:
      attributes[attributes_lib.XLA_COMPILE] = bool(self._jit_compile)
      if self._jit_compile:
        attributes[attributes_lib.NO_INLINE] = True

    if self._autograph:
      fn = autograph_util.py_func_from_autograph(
          fn, self._experimental_autograph_options)

    return tracing_compilation.TracingOptions(
        fn,
        self._name,
        polymorphic_type=self._function_type,
        default_values=self._default_values,
        scope_type=scope_type,
        attributes=attributes,
        autograph=self._autograph,
        reduce_retracing=self._reduce_retracing,
        autograph_options=self._experimental_autograph_options,
        function_cache=self._function_cache,
        function_captures=self._function_captures,
        lock=self._lock,
    )

  def _initialize(self, args, kwds, add_initializers_to=None):
    # [FUNCTION DOC]
    # HELPER: First-call setup of the two tracing configurations.
    # Inputs: args and kwds, the positional and keyword arguments of the first call;
    # add_initializers_to, optional list that receives pairs of created variable and its initial
    # value. Installs a variable-creating configuration (making unlifted initializer variables
    # and remembering them by weak reference, unless a variable opts out of lifting), traces the
    # function once with it and stores the concrete function, then installs a second
    # configuration that raises a value error on any variable creation. Returns nothing; sets
    # several instance fields and may append to the supplied list.
    """Initializes, on the first call.

    Creates two `Function`s, one that will allow creation of variables
    and one that won't.

    Additionally runs a trace for the `Function` that allows creation
    of variables.

    Args:
      args: Arguments to the underlying python callable.
      kwds: Keyword arguments to the python callable.
      add_initializers_to: Where to collect variable initializers, if not None.
    """
    created_variables = []

    def variable_capturing_scope(next_creator, **kwds):
      # [FUNCTION DOC]
      # HELPER: Variable creator callback for the first trace. Inputs: next_creator, the next
      # creator in the chain, and the variable constructor keyword arguments. If the keyword for
      # variable lifting is present and false, defers to next_creator. Otherwise builds an
      # unlifted initializer variable (passing the initializer collection list), records a weak
      # reference to it in the created variables list and returns the variable.
      """Creates UnliftedInitializerVariables and saves references to them."""
      enable_variable_lifting = kwds.get("experimental_enable_variable_lifting")
      if enable_variable_lifting is None:
        enable_variable_lifting = True
      if not enable_variable_lifting:
        return next_creator(**kwds)
      v = UnliftedInitializerVariable(
          add_initializers_to=add_initializers_to, **kwds
      )
      created_variables.append(weakref.ref(v))
      return v

    self._created_variables = created_variables
    self._variable_creation_config = self._generate_scoped_tracing_options(
        variable_capturing_scope,
        tracing_compilation.ScopeType.VARIABLE_CREATION,
    )
    # Force the definition of the function for these arguments
    self._concrete_variable_creation_fn = tracing_compilation.trace_function(
        args, kwds, self._variable_creation_config
    )

    def invalid_creator_scope(*unused_args, **unused_kwds):
      # [FUNCTION DOC]
      # HELPER: Variable creator callback for non-first traces. Ignores its arguments and always
      # raises a value error explaining that variables may only be created once on the first call.
      """Disables variable creation."""
      raise ValueError(
          "tf.function only supports singleton tf.Variables created on the "
          "first call. Make sure the tf.Variable is only created once or "
          "created outside tf.function. See "
          "https://www.tensorflow.org/guide/function#creating_tfvariables "
          "for more information.")

    self._no_variable_creation_config = self._generate_scoped_tracing_options(
        invalid_creator_scope,
        tracing_compilation.ScopeType.NO_VARIABLE_CREATION,
    )

  def _clone(self, python_function):
    # [FUNCTION DOC]
    # HELPER: Creates a fresh Function with the same settings.
    # Input: python_function, the callable to wrap, or nothing to reuse the current one. Copies
    # name, input signature, autograph, jit-compile, retracing and attribute settings, and the
    # shared-rendezvous flag if set. Returns the new, untraced Function; the cache is not shared.
    """Clone the function with different python function."""
    f = Function(
        python_function=(self._python_function
                         if python_function is None else python_function),
        name=self._name,
        input_signature=self.input_signature,
        autograph=self._autograph,
        jit_compile=self._jit_compile,
        reduce_retracing=self._reduce_retracing,
        experimental_attributes=self._attributes,
        experimental_autograph_options=self._experimental_autograph_options)

    if self._shared_rendezvous:
      f._shared_rendezvous = self._shared_rendezvous  # pylint: disable=protected-access

    return f

  def _decorate(self, decorator):
    # [FUNCTION DOC]
    # HELPER: Lets modules push a decorator inside the function (for example to add a name scope).
    # Input: decorator, a callable taking the python function and returning a replacement.
    # Raises a value error if the function has already been traced. Replaces the stored python
    # function with the decorated one and recomputes the function type and default values.
    # Returns nothing.
    """Allows the captured Python function to be decorated in place.

    This method is only safe to call when the Function has not been called by a
    user. It makes sense to use this method to push a decorator into the
    function rather than wrapping the function in the decorator.

    We use this in tf.Module to allow user annotated `tf.functions` to remain as
    `Function` objects but still automatically enter the Module name_scope
    when they are evaluated like all other methods.

    Args:
      decorator: A callable accepting a single argument which is the function
        to decorate and returning a callable result.

    Raises:
      ValueError: If the function has been called a ValueError is raised.
    """
    if (
        self._variable_creation_config is not None
        or self._no_variable_creation_config is not None
    ):
      raise ValueError(
          "Functions cannot be decorated after they have been traced."
      )

    self._python_function = decorator(self._python_function)
    self._function_type, self._default_values = (
        function_type_utils.make_function_type(
            self._python_function, self.input_signature
        )
    )

  # TODO: Remove this private method after updating all its uses
  # A good moment to do this could be when the experimental label is removed
  def _get_tracing_count(self):
    # [FUNCTION DOC]
    # HELPER: Private alias kept for existing callers; returns the same number as the
    # experimental tracing-count method.
    return self.experimental_get_tracing_count()

  def experimental_get_tracing_count(self):
    # [FUNCTION DOC]
    # Returns the number of concrete functions currently in the function cache, which equals
    # the number of traces performed (retraces add entries). Takes no inputs; read-only.
    """Returns the number of times the function has been traced.

    For more information on when a function is traced and when it is
    traced multiple times see https://www.tensorflow.org/guide/function.
    Example:

    >>> @tf.function
    ... def double(a):
    ...   return a + a
    >>> double(tf.constant(1))
    >>> double(tf.constant(2))
    >>> double.experimental_get_tracing_count()
    1
    >>> double(tf.constant("a"))
    >>> double.experimental_get_tracing_count()
    2


    The first time experimental_get_tracing_count is called
    it returns 1, as the function is traced the first
    time it is called, and the second time the same graph is used
    since we're calling it with a parameter of the same type.

    The second time experimental_get_tracing_count is called
    it returns 2, as we called double with a
    different argument type, and so it was traced again.

    """
    return len(self._function_cache)

  @property
  def _run_functions_eagerly(self):
    # [FUNCTION DOC]
    # HELPER: Read-only property returning the global flag that makes tf.functions execute
    # their python code eagerly instead of tracing.
    return eager_function_run.RUN_FUNCTIONS_EAGERLY

  @traceback_utils.filter_traceback
  def __call__(self, *args, **kwds):
    # [FUNCTION DOC]
    # Calls the function with the given positional and keyword arguments and returns the result
    # of the selected concrete function. If run-functions-eagerly is on, just runs the python
    # function inside a profiler trace. Otherwise, on the very first call it increments the
    # usage counter labeled by whether XLA compilation applies, then delegates to the internal
    # call inside a profiler trace and optional XLA context, records tracing metadata, and, when
    # executing eagerly, reports to the retracing detector whether this call traced. Exceptions
    # from tracing or execution propagate; traceback frames are filtered.
    # Implements PolymorphicFunction.__call__.
    if self._run_functions_eagerly:
      with trace.Trace(self._name, tf_function_call="eager"):
        return self._python_function(*args, **kwds)

    # Only count the statistics the first time, before initialization took
    # place.
    if self._created_variables is None:
      compiled = bool(self._jit_compile and
                      not control_flow_util.GraphOrParentsInXlaContext(
                          ops.get_default_graph()))
      # For nested functions, increment the counter only when a function with
      # jit_compile=True is called within a function with jit_compile=False. We
      # count this special case to correctly record that both jit_compile=True
      # and jit_compile=False is being used for parts of the outer function.
      if ops.executing_eagerly_outside_functions() and (
          context.executing_eagerly() or compiled):
        # Labels must be strings in Python, so we convert 'compiled' to a string
        _tf_function_counter.get_cell(str(int(compiled))).increase_by(1)

    tracing_count = self.experimental_get_tracing_count()
    with trace.Trace(self._name) as tm:
      # TODO(cheshire): Do not duplicate the XLAControlFlowContext annotation.
      compiler = "xla" if self._jit_compile else "nonXla"

      with OptionalXlaContext(self._jit_compile):
        result = self._call(*args, **kwds)

      new_tracing_count = self.experimental_get_tracing_count()
      without_tracing = (tracing_count == new_tracing_count)
      execution_mode = "notTraced" if without_tracing else "traced"
      tm.set_metadata(tf_function_call=execution_mode + "-" + compiler,
                      tracing_count=new_tracing_count)

    if context.executing_eagerly():
      if without_tracing:
        _frequent_tracing_detector_manager.called_without_tracing(
            self._key_for_call_stats)
      else:
        _frequent_tracing_detector_manager.called_with_tracing(
            self._key_for_call_stats, self._python_function,
            self._omit_frequent_tracing_warning)

    return result

  def _call(self, *args, **kwds):
    # [FUNCTION DOC]
    # HELPER: Core dispatch of a call; implements variable initialization on the first call.
    # KNOWN ISSUE: The lock is acquired at the start without a protecting try block, so if
    # canonicalizing the inputs raises, the lock is never released.
    # Inputs: positional and keyword arguments of the call. Takes the lock and canonicalizes the
    # inputs. If variables were created already, releases the lock and calls the no-variable-
    # creation configuration. If only the creation configuration exists, calls it and raises a
    # value error if that call created variables. On the very first call, initializes (tracing
    # with variable creation), releases the lock, then: if variables were created, tries to lift
    # their initializers and run them eagerly, then calls the no-creation configuration; if
    # lifting is impossible, builds a conditional wrapper that initializes only when needed (this
    # raises an unimplemented error under XLA); if no variables were created, runs the first
    # concrete function directly. Returns the call results.
    """Calls the graph function."""
    self._lock.acquire()
    bound_args = function_type_utils.canonicalize_function_inputs(
        args,
        kwds,
        self._function_type,
        self._default_values,
        self._is_pure,
    )
    args, kwds = bound_args.args, bound_args.kwargs
    if self._created_variables:
      # Release the lock early so that multiple threads can perform the call
      # in parallel.
      self._lock.release()
      # In this case we have created variables on the first call, so we run the
      # defunned version which is guaranteed to never create variables.
      return tracing_compilation.call_function(
          args, kwds, self._no_variable_creation_config
      )
    elif self._variable_creation_config is not None:
      # Release the lock early so that multiple threads can perform the call
      # in parallel.
      self._lock.release()
      # In this case we have not created variables on the first call. So we can
      # run the first trace but we should fail if variables are created.
      results = tracing_compilation.call_function(
          args, kwds, self._variable_creation_config
      )
      if self._created_variables:
        raise ValueError("Creating variables on a non-first call to a function"
                         " decorated with tf.function.")
      return results

    try:
      # This is the first call of __call__, so we have to initialize.
      initializers = []
      self._initialize(args, kwds, add_initializers_to=initializers)
    finally:
      # At this point we know that the initialization is complete (or less
      # interestingly an exception was raised) so we no longer need a lock.
      self._lock.release()

    if self._created_variables:
      try:
        # Attempt to initialize variables eagerly and without conds by lifting
        # out initialization graphs. This is the only initialization strategy
        # compatible with XLA at the moment.
        self._initialize_uninitialized_variables(initializers)
      except lift_to_graph.UnliftableError:
        pass  # Fall through to cond-based initialization.
      else:
        # Lifting succeeded, so variables are initialized and we can run the
        # no_variable_creation function.
        return tracing_compilation.call_function(
            args, kwds, self._no_variable_creation_config
        )
    else:
      bound_args = self._concrete_variable_creation_fn.function_type.bind(
          *args, **kwds
      )
      # If we did not create any variables the trace we have is good enough.
      filtered_flat_args = (
          self._concrete_variable_creation_fn.function_type.unpack_inputs(
              bound_args
          )
      )
      return self._concrete_variable_creation_fn._call_flat(  # pylint: disable=protected-access
          filtered_flat_args,
          self._concrete_variable_creation_fn.captured_inputs,
      )

    def fn_with_cond(inner_args, inner_kwds):
      # [FUNCTION DOC]
      # HELPER: Fallback body used when initializers cannot be lifted. Inputs: the canonical
      # positional arguments and keyword arguments. Computes whether all created variables are
      # initialized and conditionally runs the no-variable-creation function (if so) or the
      # variable-creating concrete function (otherwise, which performs initialization). Returns
      # the result of whichever branch runs.
      """Conditionally runs initialization if it's needed."""
      condition = True
      for v, _ in initializers:
        condition = math_ops.logical_and(
            condition, resource_variable_ops.var_is_initialized_op(
                v.handle))
      # We want to call no_variable_creation if possible because it avoids
      # recomputing potentially expensive initializers.
      return cond.cond(
          condition,
          lambda: tracing_compilation.call_function(  # pylint: disable=g-long-lambda
              inner_args, inner_kwds, self._no_variable_creation_config
          ),
          lambda: self._concrete_variable_creation_fn(  # pylint: disable=g-long-lambda
              *inner_args, **inner_kwds
          ),
      )

    # We've created variables and are unable to lift the initialization graphs,
    # so we fall back to initializing with conds while running the function.
    # TODO(b/216870587) Note that this path is not currently supported for XLA.
    if self._jit_compile:
      raise errors.UnimplementedError(
          None, None,
          "We failed to lift variable creations out of this tf.function, "
          "so this tf.function cannot be run on XLA. A possible workaround is "
          "to move variable creation outside of the XLA compiled function.")
    canon_args, canon_kwds = bound_args.args, bound_args.kwargs
    options = tracing_compilation.TracingOptions(fn_with_cond, "fn_with_cond")
    return tracing_compilation.call_function(
        (canon_args, canon_kwds), {}, options
    )

  def experimental_get_compiler_ir(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Returns a callable that produces the compiler intermediate representation of the function
    # for the given example arguments (or tensor specs). Raises a value error if jit compilation
    # was not enabled, or if the inputs mix tensor specs with real values. Initializes the
    # runtime context and gets (tracing if needed) the concrete function. When all inputs are
    # tensor specs it returns the generator built from the concrete function directly;
    # otherwise it returns a local generator taking stage, device name and platform name.
    # Implements PolymorphicFunction.experimental_get_compiler_ir
    context.ensure_initialized()
    if not self._jit_compile:
      raise ValueError("Compiler IR can only be returned for functions marked "
                       "with 'jit_compile=True'")

    is_tensor_spec = lambda x: isinstance(x, tensor_spec.TensorSpec)

    def _check_inputs(args, kwargs):
      # [FUNCTION DOC]
      # HELPER: Validates example inputs for compiler IR retrieval. Inputs: positional list and
      # keyword mapping. Empty input is accepted; a mixture of tensor specs and other values
      # raises a value error. Returns nothing.
      all_inputs = list(args) + list(kwargs.values())
      # Empty input is okay.
      if not all_inputs:
        return
      if any(map(is_tensor_spec, all_inputs)) and any(
          map(lambda x: not is_tensor_spec(x), all_inputs)
      ):
        raise ValueError(
            "experimental_get_compiler_ir supports either "
            "(1) all inputs are TensorSpec  or "
            "(2) all inputs are tf.Tensor/python variables"
        )

    _check_inputs(args, kwargs)
    if (
        len(args) + len(kwargs.values()) > 0
        and all(map(is_tensor_spec, args))
        and all(map(is_tensor_spec, kwargs.values()))
    ):
      # For the case inputs are not empty and input types are all tf.TensorSpec
      concrete_fn = self.get_concrete_function(*args, **kwargs)
      return compiler_ir.from_concrete_function(concrete_fn)

    concrete_fn = self.get_concrete_function(*args, **kwargs)
    fn_name = concrete_fn.name

    # pylint: disable=protected-access
    bound_args = function_type_utils.canonicalize_function_inputs(
        args, kwargs, concrete_fn.function_type
    )
    filtered_flat_args = concrete_fn.function_type.unpack_inputs(bound_args)

    def compiler_ir_generator(
        stage="hlo", device_name=None, platform_name=None
    ):
      # [FUNCTION DOC]
      # Returns the compiler IR for the captured concrete function. Inputs: stage (which export
      # stage, default high-level optimizer form), device_name (deprecated, optional device
      # string) and platform_name (optional platform). Raises a value error if both device and
      # platform names are given; warns when device_name is used. Asks the runtime context for the
      # IR. Returns raw bytes for the serialized stages, otherwise text decoded from UTF-8.
      """Gets the compiler IR bytes.

      Args:
        stage: The exported stage for the given function.
        device_name: The name of the device with the form as
          "/job:localhost/replica:0/task:0/device:CPU:0", "/device:TPU:0" etc.
          When this is used, actual device is used for getting the compiler IR.
        platform_name: The name of the platform, e.g. "TPU". See the comment in
          `get_compiler_ir` in `context.py`.

      Returns:
        The compiler IR bytes.
      """
      if device_name is not None:
        if platform_name is not None:
          raise ValueError(
              "device_name and platform_name cannot be provided at the same"
              " time."
          )
        warnings.warn("device_name is being deprecated. Use platform_name.")
      device_name = compiler_ir.maybe_get_device_name(device_name)
      res_bytes = context.context().get_compiler_ir(
          device_name=device_name,
          platform_name=platform_name,
          function_name=fn_name,
          flat_args=list(filtered_flat_args),
          captured_inputs=concrete_fn.captured_inputs,
          stage=stage,
      )
      if stage in (
          # Ordered by IrExportStage enum order
          "stablehlo_serialized",
          "hlo_serialized",
          "optimized_hlo_serialized",
          "optimized_hlo_proto_serialized",
      ):
        return res_bytes
      else:
        return res_bytes.decode("utf-8")

    return compiler_ir_generator

  @property
  def python_function(self):
    # [FUNCTION DOC]
    # Read-only property returning the wrapped python callable.
    """The python function wrapped in this tf.function."""
    return self._python_function

  @property
  def input_signature(self):
    # [FUNCTION DOC]
    # Read-only property: the user-specified input signature derived from the function type, or
    # nothing if none was specified.
    return function_type_utils.to_input_signature(self._function_type)

  @property
  def function_spec(self):
    # [FUNCTION DOC]
    # Read-only property building a function-spec object from the function type, default values,
    # a false flag for the class-method indicator, the name and the jit-compile setting.
    return function_type_utils.FunctionSpec(
        self._function_type,
        self._default_values,
        False,
        self._name,
        self._jit_compile,
    )

  @property
  def function_type(self):
    # [FUNCTION DOC]
    # Read-only property returning the function type describing the wrapped function's parameters.
    return self._function_type

  def pretty_printed_concrete_signatures(self, verbose=True):
    # [FUNCTION DOC]
    # Input: verbose, boolean (default true) choosing detailed signatures separated by blank
    # lines versus brief ones separated by single newlines. Returns a string joining the printed
    # signature of every concrete function (tracing the declared input signature first if one
    # exists).
    joiner = "\n\n" if verbose else "\n"
    return joiner.join([
        c.pretty_printed_signature(verbose=verbose)
        for c in self._list_all_concrete_functions()
    ])

  def _initialize_uninitialized_variables(self, initializers):
    # [FUNCTION DOC]
    # HELPER: Runs lifted variable initializers eagerly for variables not yet initialized.
    # Input: initializers, a list of pairs of variable and initial-value tensor (may be empty,
    # in which case nothing happens). Checks which variables are already initialized, then traces
    # and calls a small function (without autograph) that lifts the initial-value subgraphs and
    # assigns them to the uninitialized variables. Raises an unliftable error if the initializers
    # depend on function inputs. Returns the result of that call.
    """Make and call a `ConcreteFunction` which initializes variables."""

    if not initializers:
      return

    var_is_initialized = _evaluate_var_is_initialized(
        [v for v, _ in initializers])

    def initialize_variables():
      # [FUNCTION DOC]
      # HELPER: Traced body that lifts initial values of the not-yet-initialized variables into
      # the current graph and assigns them without reading back. Takes no inputs; returns nothing.
      op_map = object_identity.ObjectIdentityDictionary()

      inits = []
      for (v, init), is_initialized in zip(initializers, var_is_initialized):
        with ops.init_scope():
          if is_initialized:
            continue
        inits.append(init)

      if inits:
        op_map = lift_to_graph.lift_to_graph(
            inits, ops.get_default_graph(), op_map=op_map)
      for (v, init), is_initialized in zip(initializers, var_is_initialized):
        with ops.init_scope():
          if is_initialized:
            continue
        v.assign(op_map[init], read_value=False)

    with ops.init_scope():
      # Note: using tracing compilation here avoids an infinite recursion.
      # Most of the code in this function runs eagerly with init_scope, where
      # autograph is not necessary.
      options = tracing_compilation.TracingOptions(
          initialize_variables, "initialize_variables", autograph=False
      )
      return tracing_compilation.call_function(tracing_options=options)

  def get_initialization_function(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Returns a concrete function that, when run, initializes this function's variables.
    # Inputs: example positional and keyword arguments for the first trace. Raises a runtime
    # error if the function was already used. Under the lock, traces the function once with
    # variable creation and collects initializers, then traces and returns a function that
    # assigns each lifted initial value to its variable. Running it overwrites current values.
    """Returns a `ConcreteFunction` which initializes this function's variables.

    Requires that this function hasn't been accessed yet through either calling
    it or calling get_concrete_function. Fails if we cannot build an initializer
    function which does not depend on the concrete values of the inputs to this
    function.

    Note that running this function will overwrite any values currently assigned
    to variables, for example restores from a checkpoint.

    Args:
      *args: arguments to the underlying python callable.
      **kwargs: keyword arguments to the python callable.

    Returns:
      A `ConcreteFunction` object which initializes the variables of this
      function.

    Raises:
      RuntimeError: if called after the variables have been initialized.
    """
    with self._lock:
      if self._variable_creation_config is not None:
        raise RuntimeError(
            "get_initialization_function cannot be called after the function "
            "has been used")
      # Here we trace the function, collect the initializers, and attempt to
      # extract them and run them eagerly. Fail only if we cannot do so.
      initializers = []
      self._initialize(args, kwargs, add_initializers_to=initializers)

    def initialize_variables():
      # [FUNCTION DOC]
      # HELPER: Traced body that lifts each collected initial value into the current graph and
      # assigns it to its variable unconditionally. Takes no inputs; returns nothing.
      for v, init in initializers:
        v.assign(
            lift_to_graph.lift_to_graph([init], ops.get_default_graph())[init],
            read_value=False)

    # Note: using tracing compilation here avoids an infinite recursion.
    options = tracing_compilation.TracingOptions(
        initialize_variables, "initialize_variables"
    )
    return tracing_compilation.trace_function(tracing_options=options)

  def _list_all_concrete_functions(self):
    # [FUNCTION DOC]
    # HELPER: Returns all cached concrete functions. If an input signature was declared, first
    # ensures that its concrete function has been traced. Returns the cache's values.
    """Returns all concrete functions."""
    if self.input_signature is not None:
      self.get_concrete_function()
    return self._function_cache.values()

  def _list_all_concrete_functions_for_serialization(self):
    # [FUNCTION DOC]
    # HELPER: Collects concrete functions to save. Gathers distinct argument signatures (the
    # declared input signature, or the structured signatures of cached functions, skipping and
    # logging those with unknown arguments and deduplicating structurally equal ones), then
    # re-creates a concrete function per signature so stale cache keys are retraced. Returns a
    # list of concrete functions; may trace.
    """Returns all concrete functions for serialization.

    Returns:
      A list of instances of `ConcreteFunction`.
    """
    seen_signatures = []
    if self.input_signature is not None:
      seen_signatures.append((self.input_signature, {}))
    else:
      concrete_functions = self._list_all_concrete_functions()
      for concrete_function in concrete_functions:
        signature = concrete_function.structured_input_signature
        flattened = nest.flatten(signature)
        if any(
            isinstance(arg, func_graph_module.UnknownArgument)
            for arg in flattened):
          logging.info("Unsupported signature for serialization: %s.",
                       signature)
          continue
        equal_to_signature = functools.partial(
            function_type_utils.is_same_structure, signature, check_values=True)
        if not any(equal_to_signature(s) for s in seen_signatures):
          seen_signatures.append(signature)

    # Re-create concrete functions for these signatures. Re-creating ensures
    # that if the cache key has changed, the function will be traced again.
    concrete_functions = []
    for args, kwargs in seen_signatures:
      concrete_functions.append(self.get_concrete_function(*args, **kwargs))
    return concrete_functions

  def _trackable_children(self, save_type="checkpoint", **kwargs):
    # [FUNCTION DOC]
    # Trackable protocol hook. Input: save_type (default checkpoint) and ignored extra keyword
    # arguments. Returns an empty mapping for checkpoints, otherwise a mapping from trace names
    # (numbered) to the concrete functions that should be serialized.
    """For implementing `Trackable`."""
    if save_type == "checkpoint":
      return {}
    return {f"trace_{n}": fn for n, fn in
            enumerate(self._list_all_concrete_functions_for_serialization())}

  def _deserialization_dependencies(self, children):
    # [FUNCTION DOC]
    # Trackable protocol hook. Input: children, the restored child objects. Returns them
    # unchanged, declaring all children as dependencies to load first.
    """Returns concrete functions which must be loaded before this object."""
    return children

  def _get_concrete_function_garbage_collected(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Returns a concrete function specialized to the given example arguments whose graph is
    # bound to the function's lifetime. Under the lock, initializes on first use and runs lifted
    # initializers. Then traces with the no-variable-creation configuration (if variables were
    # created) or the creation configuration (raising a value error if it creates variables on a
    # non-first call), with the graph bound to the returned function. May trace new functions.
    """Returns a `ConcreteFunction` specialized to inputs and execution context.

    Unlike `get_concrete_function(...)`, the graph will be deleted when the
    returned function is deleted.  It's useful to avoid creating a reference
    cycle when you know for sure that the graph will be no longer used without
    the returned function.

    Args:
      *args: inputs to specialize on.
      **kwargs: inputs to specialize on.

    Returns:
      A TensorFlow function which takes exactly one `tf.Tensor` per argument.

    Raises:
      ValueError: if this object has not yet been called on concrete values.
    """
    with self._lock:
      if self._variable_creation_config is None:
        initializers = []
        self._initialize(args, kwargs, add_initializers_to=initializers)
        self._initialize_uninitialized_variables(initializers)

    if self._created_variables:
      # In this case we have created variables on the first call, so we run the
      # version which is guaranteed to never create variables.
      return tracing_compilation.trace_function(
          args,
          kwargs,
          dataclasses.replace(
              self._no_variable_creation_config, bind_graph_to_function=True
          ),
      )
    elif self._variable_creation_config is not None:
      # In this case we have not created variables on the first call. So we can
      # run the first trace but we should fail if variables are created.
      concrete = tracing_compilation.trace_function(
          args,
          kwargs,
          dataclasses.replace(
              self._variable_creation_config, bind_graph_to_function=True
          ),
      )
      if self._created_variables:
        raise ValueError("Creating variables on a non-first call to a function"
                         " decorated with tf.function.")
      return concrete

  def get_concrete_function(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Returns a concrete function specialized to the example arguments (values or tensor
    # specs), tracing if necessary. Obtains the garbage-collected variant and then releases its
    # garbage collector so the graph stays alive in the cache. Errors propagate from tracing.
    # Implements PolymorphicFunction.get_concrete_function.
    concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
    concrete._garbage_collector.release()  # pylint: disable=protected-access
    return concrete

  def __tf_tracing_type__(self, _):
    # [FUNCTION DOC]
    # Tracing-type protocol hook. The context argument is ignored. Returns a weak-reference
    # trace type for this function, so passing a Function as an argument does not keep it alive.
    return trace_type.Weakref(weakref.ref(self))

  def __get__(self, instance, owner):
    # [FUNCTION DOC]
    # Descriptor hook enabling decorated methods. Inputs: instance (object accessed through, or
    # nothing for class access) and owner (ignored). For composite tensor instances with a type
    # spec, returns a bound method treating the instance as a normal argument. With no instance,
    # returns this function. Otherwise creates (once per instance, cached in a weak mapping) a
    # per-instance function with the instance bound weakly, and returns it.
    """Makes it possible to decorate instance methods."""
    del owner
    # `instance` here is the instance that this `Function` was accessed through
    # e.g., for
    #
    #   class Foo:
    #
    #     @tf.function
    #     def bar(self):
    #       ...
    #
    #   foo = Foo()
    #   foo.bar()  # `foo.bar` is a `Function` instance
    #
    # then `instance` will be `foo` (and `owner` will be `Foo`).  For composite
    # tensors, we can just treat `instance` as a normal parameter.  But for
    # other types, we create a new instance of `Function` here to allow
    # different instances each to create variables once, thereby allowing
    # methods to be decorated with tf.function. Keeps a cache to avoid retracing
    # the function every time the descriptor is accessed.
    # TODO(mdan): Identify types which can just be parameters more generically.
    #
    # The check for instance._type_spec=None is used because certain classes
    # (including subclasses of tf.linalg.LinearOperator) are subclasses of
    # CompositeTensor but do not actually implement the required APIs.
    # TODO(b/199278478): Fix those classes, then remove the check for
    # `instance._type_spec is not None`.
    if (isinstance(instance, composite_tensor.CompositeTensor) and
        instance._type_spec is not None):  # pylint: disable=protected-access
      return types_lib.MethodType(self, instance)
    if instance not in self._descriptor_cache:
      if instance is None:
        return self
      # TODO(mdan): If the CompositeTensor path works, do the same here.
      # It's unclear whether we need the tf-decorator, or could just call
      # MethodType(self.clone(), instance)
      self._descriptor_cache[instance] = (
          class_method_to_instance_method(self, instance))
    return self._descriptor_cache[instance]


@tf_export("function")
@deprecation.deprecated_args(None,
                             "experimental_compile is deprecated, use "
                             "jit_compile instead", "experimental_compile")
@deprecation.deprecated_args(None,
                             "experimental_relax_shapes is deprecated, use "
                             "reduce_retracing instead",
                             "experimental_relax_shapes")
@deprecation.deprecated_args(None,
                             "experimental_follow_type_hints is deprecated",
                             "experimental_follow_type_hints")
def function(
    func=None,
    input_signature=None,
    autograph=True,
    jit_compile=None,
    reduce_retracing=False,
    experimental_implements=None,
    experimental_autograph_options=None,
    experimental_attributes=None,
    experimental_relax_shapes=None,
    experimental_compile=None,
    experimental_follow_type_hints=None  # pylint: disable=unused-argument
) -> core.PolymorphicFunction:
  # [FUNCTION DOC]
  # KNOWN ISSUE: The docstring says a value error is raised when XLA support is unavailable;
  # this function never raises that itself (it only stores the setting). Also the follow-type-
  # hints argument is accepted but ignored.
  # Public tf.function entry point. Inputs: func (callable, or nothing to get a decorator),
  # input_signature, autograph, jit_compile (defaults to the environment-based global default
  # when unset), reduce_retracing, experimental_implements, experimental_autograph_options,
  # experimental_attributes, and deprecated aliases (relax-shapes forces retracing reduction;
  # compile aliases jit_compile via a deprecated-argument lookup). Returns a wrapped function
  # object if func is given, otherwise a decorator that does so. Side effect: none until the
  # wrapper is called, except constructing the wrapper.
  """Compiles a function into a callable TensorFlow graph.

  `tf.function` constructs a `tf.types.experimental.PolymorphicFunction` that
  executes a TensorFlow graph (`tf.Graph`) created by trace-compiling the
  TensorFlow operations in `func`. More information on the topic can be found
  in [Introduction to Graphs and tf.function]
  (https://www.tensorflow.org/guide/intro_to_graphs).

  See [Better Performance with tf.function]
  (https://www.tensorflow.org/guide/function) for tips on performance and
  known limitations.

  Example usage:

  >>> @tf.function
  ... def f(x, y):
  ...   return x ** 2 + y
  >>> x = tf.constant([2, 3])
  >>> y = tf.constant([3, -2])
  >>> f(x, y)
  <tf.Tensor: ... numpy=array([7, 7], ...)>

  The trace-compilation allows non-TensorFlow operations to execute, but under
  special conditions. In general, only TensorFlow operations are guaranteed to
  run and create fresh results whenever the `PolymorphicFunction` is called.

  ## Features

  `func` may use data-dependent Python control flow statements, including `if`,
  `for`, `while` `break`, `continue` and `return`:

  >>> @tf.function
  ... def f(x):
  ...   if tf.reduce_sum(x) > 0:
  ...     return x * x
  ...   else:
  ...     return -x // 2
  >>> f(tf.constant(-2))
  <tf.Tensor: ... numpy=1>

  `func`'s closure may include `tf.Tensor` and `tf.Variable` objects:

  >>> @tf.function
  ... def f():
  ...   return x ** 2 + y
  >>> x = tf.constant([-2, -3])
  >>> y = tf.Variable([3, -2])
  >>> f()
  <tf.Tensor: ... numpy=array([7, 7], ...)>

  `func` may also use ops with side effects, such as `tf.print`, `tf.Variable`
  and others:

  >>> v = tf.Variable(1)
  >>> @tf.function
  ... def f(x):
  ...   for i in tf.range(x):
  ...     v.assign_add(i)
  >>> f(3)
  >>> v
  <tf.Variable ... numpy=4>

  Important: Any Python side-effects (appending to a list, printing with
  `print`, etc) will only happen once, when `func` is traced. To have
  side-effects executed into your `tf.function` they need to be written
  as TF ops:

  >>> l = []
  >>> @tf.function
  ... def f(x):
  ...   for i in x:
  ...     l.append(i + 1)    # Caution! Will only happen once when tracing
  >>> f(tf.constant([1, 2, 3]))
  >>> l
  [<tf.Tensor ...>]

  Instead, use TensorFlow collections like `tf.TensorArray`:

  >>> @tf.function
  ... def f(x):
  ...   ta = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
  ...   for i in range(len(x)):
  ...     ta = ta.write(i, x[i] + 1)
  ...   return ta.stack()
  >>> f(tf.constant([1, 2, 3]))
  <tf.Tensor: ..., numpy=array([2, 3, 4], ...)>

  ## `tf.function` creates polymorphic callables

  Internally, `tf.types.experimental.PolymorphicFunction` may contain multiple
  `tf.types.experimental.ConcreteFunction`s, each specialized to arguments with
  different data types or shapes, since TensorFlow can perform more
  optimizations on graphs of specific shapes, dtypes and values of constant
  arguments. `tf.function` treats any pure Python values as opaque objects (best
  thought of as compile-time constants), and builds a separate `tf.Graph` for
  each set of Python arguments that it encounters.
  For more information, see the
  [tf.function
  guide](https://www.tensorflow.org/guide/function#rules_of_tracing)

  Executing a `PolymorphicFunction` will select and execute the appropriate
  `ConcreteFunction` based on the argument types and values.

  To obtain an individual `ConcreteFunction`, use the
  `PolymorphicFunction.get_concrete_function` method. It can be called with the
  same arguments as `func` and returns a
  `tf.types.experimental.ConcreteFunction`. `ConcreteFunction`s are backed by a
  single `tf.Graph`:

  >>> @tf.function
  ... def f(x):
  ...   return x + 1
  >>> isinstance(f.get_concrete_function(1).graph, tf.Graph)
  True

  `ConcreteFunction`s can be executed just like `PolymorphicFunction`s, but
  their
  input is restricted to the types to which they're specialized.

  ## Retracing

  `ConcreteFunctions` are built (traced) on the fly, as the
  `PolymorphicFunction` is
  called with new TensorFlow types or shapes, or with new Python values as
  arguments. When `PolymorphicFunction` builds a new trace, it is said that
  `func`
  is retraced. Retracing is a frequent performance concern for `tf.function` as
  it can be considerably slower than executing a graph that's already been
  traced. It is ideal to minimize the amount of retracing in your code.

  Caution: Passing python scalars or lists as arguments to `tf.function` will
  usually retrace. To avoid this, pass numeric arguments as Tensors whenever
  possible:

  >>> @tf.function
  ... def f(x):
  ...   return tf.abs(x)
  >>> f1 = f.get_concrete_function(1)
  >>> f2 = f.get_concrete_function(2)  # Slow - compiles new graph
  >>> f1 is f2
  False
  >>> f1 = f.get_concrete_function(tf.constant(1))
  >>> f2 = f.get_concrete_function(tf.constant(2))  # Fast - reuses f1
  >>> f1 is f2
  True

  Python numerical arguments should only be used when they take few distinct
  values, such as hyperparameters like the number of layers in a neural network.

  ## Input signatures

  For Tensor arguments, `PolymorphicFunction`creates a new `ConcreteFunction`
  for
  every unique set of input shapes and datatypes. The example below creates two
  separate `ConcreteFunction`s, each specialized to a different shape:

  >>> @tf.function
  ... def f(x):
  ...   return x + 1
  >>> vector = tf.constant([1.0, 1.0])
  >>> matrix = tf.constant([[3.0]])
  >>> f.get_concrete_function(vector) is f.get_concrete_function(matrix)
  False

  An "input signature" can be optionally provided to `tf.function` to control
  this process. The input signature specifies the shape and type of each
  Tensor argument to the function using a `tf.TensorSpec` object. More general
  shapes can be used. This ensures only one `ConcreteFunction` is created, and
  restricts the `PolymorphicFunction` to the specified shapes and types. It is
  an effective way to limit retracing when Tensors have dynamic shapes.

  >>> @tf.function(
  ...     input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
  ... def f(x):
  ...   return x + 1
  >>> vector = tf.constant([1.0, 1.0])
  >>> matrix = tf.constant([[3.0]])
  >>> f.get_concrete_function(vector) is f.get_concrete_function(matrix)
  True

  ## Variables may only be created once

  `tf.function` only allows creating new `tf.Variable` objects when it is called
  for the first time:

  >>> class MyModule(tf.Module):
  ...   def __init__(self):
  ...     self.v = None
  ...
  ...   @tf.function
  ...   def __call__(self, x):
  ...     if self.v is None:
  ...       self.v = tf.Variable(tf.ones_like(x))
  ...     return self.v * x

  In general, it is recommended to create `tf.Variable`s outside of
  `tf.function`.
  In simple cases, persisting state across `tf.function` boundaries may be
  implemented using a pure functional style in which state is represented by
  `tf.Tensor`s passed as arguments and returned as return values.

  Contrast the two styles below:

  >>> state = tf.Variable(1)
  >>> @tf.function
  ... def f(x):
  ...   state.assign_add(x)
  >>> f(tf.constant(2))  # Non-pure functional style
  >>> state
  <tf.Variable ... numpy=3>

  >>> state = tf.constant(1)
  >>> @tf.function
  ... def f(state, x):
  ...   state += x
  ...   return state
  >>> state = f(state, tf.constant(2))  # Pure functional style
  >>> state
  <tf.Tensor: ... numpy=3>

  ## Python operations execute only once per trace

  `func` may contain TensorFlow operations mixed with pure Python operations.
  However, when the function is executed, only the TensorFlow operations will
  run. The Python operations run only once, at trace time. If TensorFlow
  operations depend on results from Python operations, those results will be
  frozen into the graph.

  >>> @tf.function
  ... def f(a, b):
  ...   print('this runs at trace time; a is', a, 'and b is', b)
  ...   return b
  >>> f(1, tf.constant(1))
  this runs at trace time; a is 1 and b is Tensor("...", shape=(), dtype=int32)
  <tf.Tensor: shape=(), dtype=int32, numpy=1>

  >>> f(1, tf.constant(2))
  <tf.Tensor: shape=(), dtype=int32, numpy=2>

  >>> f(2, tf.constant(1))
  this runs at trace time; a is 2 and b is Tensor("...", shape=(), dtype=int32)
  <tf.Tensor: shape=(), dtype=int32, numpy=1>

  >>> f(2, tf.constant(2))
  <tf.Tensor: shape=(), dtype=int32, numpy=2>

  Args:
    func: The function to be compiled. If `func` is None, `tf.function` returns
      a decorator that can be invoked with a single argument - `func`. In other
      words, `tf.function(input_signature=...)(func)` is equivalent to
      `tf.function(func, input_signature=...)`. The former can be used as
      decorator.
    input_signature: A possibly nested sequence of `tf.TensorSpec` objects
      specifying the shapes and dtypes of the Tensors that will be supplied to
      this function. If `None`, a separate function is instantiated for each
      inferred input signature.  If input_signature is specified, every input to
      `func` must be a `Tensor`, and `func` cannot accept `**kwargs`.
    autograph: Whether autograph should be applied on `func` before tracing a
      graph. Data-dependent Python control flow statements require
      `autograph=True`. For more information, see the [tf.function and AutoGraph
      guide](
      https://www.tensorflow.org/guide/function#autograph_transformations).
    jit_compile: If `True`, compiles the function using
      [XLA](https://tensorflow.org/xla). XLA performs compiler optimizations,
      such as fusion, and attempts to emit more efficient code. This may
      drastically improve the performance. If set to `True`, the whole function
      needs to be compilable by XLA, or an `errors.InvalidArgumentError` is
      thrown. If `None` (default), compiles the function with XLA when running
      on TPU and goes through the regular function execution path when running
      on other devices. If `False`, executes the function without XLA
      compilation.  Set this value to `False` when directly running a
      multi-device function on TPUs (e.g. two TPU cores, one TPU core and its
      host CPU). Not all functions are compilable, see a list of [sharp
      corners](https://tensorflow.org/xla/known_issues).
    reduce_retracing: When True, `tf.function` attempts to reduce the amount of
      retracing, for example by using more generic shapes. This can be
      controlled for user objects by customizing their associated
      `tf.types.experimental.TraceType`.
    experimental_implements: If provided, contains a name of a "known" function
      this implements. For example "mycompany.my_recurrent_cell". This is stored
      as an attribute in inference function, which can then be detected when
      processing serialized function. See [standardizing composite
      ops](https://github.com/tensorflow/community/blob/master/rfcs/20190610-standardizing-composite_ops.md)
      # pylint: disable=line-too-long for details.  For an example of utilizing
      this attribute see this
      [example](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/lite/transforms/prepare_composite_functions_tf.cc)
      The code above automatically detects and substitutes function that
      implements "embedded_matmul" and allows TFLite to substitute its own
      implementations. For instance, a tensorflow user can use this attribute to
      mark that their function also implements `embedded_matmul` (perhaps more
      efficiently!) by specifying it using this parameter:
      `@tf.function(experimental_implements="embedded_matmul")` This can either
      be specified as just the string name of the function or a NameAttrList
      corresponding to a list of key-value attributes associated with the
      function name. The name of the function will be in the 'name' field of the
      NameAttrList. To define a formal TF op for this function implements, try
      the experimental [composite
      TF](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/mlir/tfr)
      project.
    experimental_autograph_options: Optional tuple of
      `tf.autograph.experimental.Feature` values.
    experimental_attributes: Optional dictionary of attributes to include in the
      generated FunctionDefs.
    experimental_relax_shapes: Deprecated. Use `reduce_retracing` instead.
    experimental_compile: Deprecated alias to 'jit_compile'.
    experimental_follow_type_hints: Deprecated. Please use input_signature or
      reduce_retracing instead.

  Returns:
     If `func` is not None, returns a
     `tf.types.experimental.PolymorphicFunction`.
     If `func` is None, returns a decorator that, when invoked with a single
     `func` argument, returns a `tf.types.experimental.PolymorphicFunction`.

  Raises:
     `ValueError` when attempting to use `jit_compile=True`, but XLA support is
     not available.
  """
  if jit_compile is None and JIT_COMPILE_FUNCTIONS:
    jit_compile = True

  # TODO(b/224808187): Remove after renaming usages.
  if experimental_relax_shapes:
    reduce_retracing = True

  def decorated(inner_function):
    # [FUNCTION DOC]
    # HELPER: Decorator produced by the public entry point. Input: inner_function, the callable
    # to wrap. Uses its name (or a default name if it has none), builds a Function with the
    # captured options and returns it wrapped with a decorator-preserving wrapper so that
    # inspection sees the original function.
    try:
      name = inner_function.__name__
    except AttributeError:
      name = "function"
    return tf_decorator.make_decorator(
        inner_function,
        decorator_name="tf.function",
        decorator_func=Function(
            inner_function,
            name,
            input_signature=input_signature,
            autograph=autograph,
            experimental_autograph_options=experimental_autograph_options,
            reduce_retracing=reduce_retracing,

            # TODO(b/171825496): Update once `experimental_compile` is removed
            # entirely in favor of 'jit_compile'.
            jit_compile=deprecation.deprecated_argument_lookup(
                "jit_compile",
                jit_compile,
                "experimental_compile",
                experimental_compile),
            experimental_implements=experimental_implements,
            experimental_attributes=experimental_attributes))

  # This code path is for the `foo = tf.function(foo, ...)` use case
  if func is not None:
    return decorated(func)

  # This code path is for the
  #
  # @tf.function(...)
  # def foo(...):
  #    ...
  #
  # use case, which is equivalent to `foo = tf.function(...)(foo)`
  return decorated


def class_method_to_instance_method(original_function, instance):
  # [FUNCTION DOC]
  # HELPER: Used by the descriptor hook of Function to bind a function to an instance.
  # Inputs: original_function, a polymorphic function wrapping a method; instance, the object
  # the method was accessed through. Builds a bound method whose target holds only a weak
  # reference to the instance, creates a new Function of the same class with copied settings
  # around a wrapper that calls the original python function with the instance, and returns it
  # wrapped by a decorator-preserving wrapper. Asserts the original has the expected attributes.
  """Constructs a new `Function` with `self` bound."""
  weak_instance = weakref.ref(instance)

  # Note: while we could bind to a weakref proxy instead, that causes the
  # bound method to be unhashable.
  bound_method = types_lib.MethodType(
      original_function.python_function,
      tf_method_target.TfMethodTarget(weak_instance,
                                      original_function.python_function))

  # original_function is expected to be PolymorphicFunction
  assert hasattr(original_function, "_name")
  assert hasattr(original_function, "_autograph")
  assert hasattr(original_function, "_function_type")
  assert hasattr(original_function, "python_function")

  weak_bound_method_wrapper = None

  def bound_method_wrapper(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: Callable placed inside the per-instance function. Takes the call's arguments. If
    # the wrapped function was not replaced by an autograph-converted one, calls the original
    # python function with the weakly held instance first; otherwise calls the replacement with
    # the arguments as given. Returns the call result.
    """Wraps either a dummy MethodType or a converted AutoGraph function."""
    # __wrapped__ allows AutoGraph to swap in a converted function.
    strong_bound_method_wrapper = weak_bound_method_wrapper()
    wrapped_fn = strong_bound_method_wrapper.__wrapped__

    if wrapped_fn is strong_bound_method_wrapper.__original_wrapped__:
      # If __wrapped__ was not replaced, then call original_function.
      # TODO(mdan): For better consistency, use the wrapper's call().
      wrapped_fn = original_function.python_function
      return wrapped_fn(weak_instance(), *args, **kwargs)

    # If __wrapped__ was replaced, then it is always an unbound function.
    # However, the replacer is still responsible for attaching self properly.
    # TODO(mdan): Is it possible to do it here instead?
    return wrapped_fn(*args, **kwargs)

  weak_bound_method_wrapper = weakref.ref(bound_method_wrapper)

  # pylint: disable=protected-access
  # We make a dummy MethodType object to generate the correct bound method
  # signature. The actual call is to a function with a weak reference to
  # `instance`.
  instance_func = type(original_function)(
      tf_decorator.make_decorator(bound_method, bound_method_wrapper),
      name=original_function._name,
      autograph=original_function._autograph,
      input_signature=original_function.input_signature,
      reduce_retracing=original_function._reduce_retracing,
      jit_compile=original_function._jit_compile,
      experimental_attributes=original_function._attributes)
  # pylint: enable=protected-access

  # We wrap the bound method with tf_decorator so inspection works correctly
  wrapped_instance_func = tf_decorator.make_decorator(bound_method,
                                                      instance_func)
  return wrapped_instance_func
