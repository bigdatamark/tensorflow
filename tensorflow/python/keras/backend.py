# [MODULE DOC]
# Keras backend API for the legacy (in-tree) Keras copy: a large collection of thin, mostly stateless
# helper functions that wrap TensorFlow ops with the Keras conventions (variables carrying Keras
# metadata, symbolic placeholders, channels-first/last handling, learning phase, name uids).
# It exposes: global-state helpers (graph and session management, learning phase flag, unique name
# counters, variable tracking), tensor creation (variable, constant, placeholder, zeros, ones, random
# variables), shape and element-wise math ops, reductions, normalization, convolution and pooling
# wrappers, losses (crossentropy variants), CTC helpers, random sampling, an RNN loop driver (rnn),
# a graph-mode callable (GraphExecutionFunction and function), and a thread-aware cache class
# (ContextValueCache). Configuration functions (epsilon, floatx, image data format) are re-exported
# from the backend_config module for compatibility.
# Import-time side effects: creates thread-local holders for the Keras graph, scratch graph and
# session; creates module level registries (name uid maps per graph, observed names, learning phase,
# variables and optimizers caches); patches the documentation strings of get_value and set_value;
# reads the Keras configuration file (keras.json under the Keras home directory, which defaults to
# a hidden folder in the user home, or the location in the KERAS_HOME environment variable) and
# applies floatx, epsilon and image data format from it, asserting they are valid; and, if the
# file or folder is missing, tries to create the folder and write a default configuration file
# (silently ignoring permission errors). This is a core backend module used by layers, models,
# losses and metrics rather than a helper module of its own.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
# pylint: disable=protected-access
# pylint: disable=redefined-outer-name
# pylint: disable=redefined-builtin
# pylint: disable=g-classes-have-attributes
"""Keras backend API."""

import collections
import itertools
import json
import os
import sys
import threading
import warnings
import weakref

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python import tf2
from tensorflow.python.client import session as session_module
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.eager import context
from tensorflow.python.eager.context import get_config
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device_spec
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend_config
from tensorflow.python.keras.distribute import distribute_coordinator_utils as dc
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import object_identity
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients as gradients_module
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import map_fn as map_fn_lib
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables as variables_module
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import moving_averages
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import numpy_compat
from tensorflow.tools.docs import doc_controls


py_all = all
py_sum = sum
py_any = any

# INTERNAL UTILS

# The internal graph maintained by Keras and used by the symbolic Keras APIs
# while executing eagerly (such as the functional API for model-building).
# This is thread-local to allow building separate models in different threads
# concurrently, but comes at the cost of not being able to build one model
# across threads.
_GRAPH = threading.local()

# A graph which is used for constructing functions in eager mode.
_CURRENT_SCRATCH_GRAPH = threading.local()

# This is a thread local object that will hold the default internal TF session
# used by Keras. It can be set manually via `set_session(sess)`.
_SESSION = threading.local()


# A global dictionary mapping graph objects to an index of counters used
# for various layer/optimizer names in each graph.
# Allows to give unique autogenerated names to layers, in a graph-specific way.
PER_GRAPH_OBJECT_NAME_UIDS = weakref.WeakKeyDictionary()


# A global set tracking what object names have been seen so far.
# Optionally used as an avoid-list when generating names
OBSERVED_NAMES = set()


# _DUMMY_EAGER_GRAPH.key is used as a key in _GRAPH_LEARNING_PHASES.
# We keep a separate reference to it to make sure it does not get removed from
# _GRAPH_LEARNING_PHASES.
# _DummyEagerGraph inherits from threading.local to make its `key` attribute
# thread local. This is needed to make set_learning_phase affect only the
# current thread during eager execution (see b/123096885 for more details).
class _DummyEagerGraph(threading.local):
  # [CLASS DOC]
  # HELPER: internal support class for the learning phase bookkeeping in eager mode.
  # Thread-local holder whose per-thread attribute named key is a unique, weak-referenceable object used
  # as the dictionary key for the learning phase in eager execution (standing in for a graph), plus a
  # per-thread flag saying whether the learning phase was explicitly set in this thread. It subclasses
  # the standard thread-local type so each thread gets its own key and flag; a single module-level
  # instance is created right after the class definition.
  """_DummyEagerGraph provides a thread local `key` attribute.

  We can't use threading.local directly, i.e. without subclassing, because
  gevent monkey patches threading.local and its version does not support
  weak references.
  """

  class _WeakReferencableClass:
    # [CLASS DOC]
    # HELPER: internal marker class used only to create unique dictionary keys.
    # Empty class whose instances support weak references and compare by identity, so every instance is
    # a distinct key in the learning phase mapping. Holds no state and defines no behavior.
    """This dummy class is needed for two reasons.

    - We need something that supports weak references. Basic types like string
    and ints don't.
    - We need something whose hash and equality are based on object identity
    to make sure they are treated as different keys to _GRAPH_LEARNING_PHASES.

    An empty Python class satisfies both of these requirements.
    """
    pass

  def __init__(self):
    # [FUNCTION DOC]
    # HELPER: constructor of the thread-local eager graph stand-in.
    # Runs once for every thread that touches the instance (thread-local semantics). Calls the parent
    # constructor, creates a fresh unique key object for this thread, and initializes the
    # learning-phase-is-set flag to false. Takes no inputs and returns nothing.
    # Constructors for classes subclassing threading.local run once
    # per thread accessing something in the class. Thus, each thread will
    # get a different key.
    super(_DummyEagerGraph, self).__init__()
    self.key = _DummyEagerGraph._WeakReferencableClass()
    self.learning_phase_is_set = False


_DUMMY_EAGER_GRAPH = _DummyEagerGraph()

# This boolean flag can be set to True to leave variable initialization
# up to the user.
# Change its value via `manual_variable_initialization(value)`.
_MANUAL_VAR_INIT = False

# This list holds the available devices.
# It is populated when `_get_available_gpus()` is called for the first time.
# We assume our devices don't change henceforth.
_LOCAL_DEVICES = None

# The below functions are kept accessible from backend for compatibility.
epsilon = backend_config.epsilon
floatx = backend_config.floatx
image_data_format = backend_config.image_data_format
set_epsilon = backend_config.set_epsilon
set_floatx = backend_config.set_floatx
set_image_data_format = backend_config.set_image_data_format


@doc_controls.do_not_generate_docs
def backend():
  # [FUNCTION DOC]
  # Returns the name of the current backend, kept only for API compatibility with multi-backend Keras.
  # No inputs; always returns the string naming TensorFlow. No side effects. Excluded from generated
  # API documentation by a decorator.
  """Publicly accessible method for determining the current backend.

  Only exists for API compatibility with multi-backend Keras.

  Returns:
      The string "tensorflow".
  """
  return 'tensorflow'


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def cast_to_floatx(x):
  # [FUNCTION DOC]
  # Casts an array-like or tensor to the default Keras floating point type (the configured floatx).
  # Input x: a numpy array, list, tensor, variable or sparse tensor. If x is a TensorFlow tensor,
  # variable or sparse tensor, a TensorFlow cast op is applied and a tensor is returned; otherwise the
  # value is converted to a numpy array of the floatx type. Returns the converted object. No side
  # effects. Supports TensorFlow API dispatch.
  """Cast a Numpy array to the default Keras float type.

  Args:
      x: Numpy array or TensorFlow tensor.

  Returns:
      The same array (Numpy array if `x` was a Numpy array, or TensorFlow tensor
      if `x` was a tensor), cast to its new type.

  Example:

  >>> tf.keras.backend.floatx()
  'float32'
  >>> arr = np.array([1.0, 2.0], dtype='float64')
  >>> arr.dtype
  dtype('float64')
  >>> new_arr = cast_to_floatx(arr)
  >>> new_arr
  array([1.,  2.], dtype=float32)
  >>> new_arr.dtype
  dtype('float32')

  """
  if isinstance(x, (tensor_lib.Tensor,
                    variables_module.Variable,
                    sparse_tensor.SparseTensor)):
    return math_ops.cast(x, dtype=floatx())
  return numpy_compat.np_asarray(x, dtype=floatx())


def get_uid(prefix=''):
  # [FUNCTION DOC]
  # Returns an increasing integer id associated with a string prefix within the current graph.
  # Input prefix: string to count (defaults to the empty string). Looks up (creating if needed) the
  # per-graph counter mapping for the graph returned by the Keras graph accessor, increments the
  # counter of the prefix and returns the new value (first call returns one). Side effect: mutates the
  # global per-graph name uid registry.
  """Associates a string prefix with an integer counter in a TensorFlow graph.

  Args:
    prefix: String prefix to index.

  Returns:
    Unique integer ID.

  Example:

  >>> get_uid('dense')
  1
  >>> get_uid('dense')
  2

  """
  graph = get_graph()
  if graph not in PER_GRAPH_OBJECT_NAME_UIDS:
    PER_GRAPH_OBJECT_NAME_UIDS[graph] = collections.defaultdict(int)
  layer_name_uids = PER_GRAPH_OBJECT_NAME_UIDS[graph]
  layer_name_uids[prefix] += 1
  return layer_name_uids[prefix]


def reset_uids():
  # [FUNCTION DOC]
  # Clears the global per-graph name counters and the set of observed object names, so that
  # automatically generated layer names restart from scratch. No inputs, returns nothing. Mutates
  # module level state only.
  """Resets graph identifiers.
  """

  PER_GRAPH_OBJECT_NAME_UIDS.clear()
  OBSERVED_NAMES.clear()


def clear_session():
  # [FUNCTION DOC]
  # Resets all global state that Keras keeps (name counters, Keras graph, session, learning phase
  # registry, tracked variables and optimizers). No inputs, returns nothing.
  # Steps: drops the thread-local Keras graph, resets the default TensorFlow graph, clears the uid
  # maps, drops the cached session, then builds a new Keras graph and inside it clears the learning
  # phase registry (re-creating a default entry for the new graph) and removes the graph's entries from
  # the variable and optimizer caches; the dummy eager learning phase flag is turned off. If eager
  # execution is active it also clears the eager kernel cache. Side effects are the global mutations
  # and the replacement of the thread's Keras graph.
  """Resets all state generated by Keras.

  Keras manages a global state, which it uses to implement the Functional
  model-building API and to uniquify autogenerated layer names.

  If you are creating many models in a loop, this global state will consume
  an increasing amount of memory over time, and you may want to clear it.
  Calling `clear_session()` releases the global state: this helps avoid clutter
  from old models and layers, especially when memory is limited.

  Example 1: calling `clear_session()` when creating models in a loop

  ```python
  for _ in range(100):
    # Without `clear_session()`, each iteration of this loop will
    # slightly increase the size of the global state managed by Keras
    model = tf.keras.Sequential([tf.keras.layers.Dense(10) for _ in range(10)])

  for _ in range(100):
    # With `clear_session()` called at the beginning,
    # Keras starts with a blank state at each iteration
    # and memory consumption is constant over time.
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential([tf.keras.layers.Dense(10) for _ in range(10)])
  ```

  Example 2: resetting the layer name generation counter

  >>> import tensorflow as tf
  >>> layers = [tf.keras.layers.Dense(10) for _ in range(10)]
  >>> new_layer = tf.keras.layers.Dense(10)
  >>> print(new_layer.name)
  dense_10
  >>> tf.keras.backend.set_learning_phase(1)
  >>> print(tf.keras.backend.learning_phase())
  1
  >>> tf.keras.backend.clear_session()
  >>> new_layer = tf.keras.layers.Dense(10)
  >>> print(new_layer.name)
  dense
  """
  global _SESSION
  global _GRAPH_LEARNING_PHASES  # pylint: disable=global-variable-not-assigned
  global _GRAPH_VARIABLES  # pylint: disable=global-variable-not-assigned
  global _GRAPH_TF_OPTIMIZERS  # pylint: disable=global-variable-not-assigned
  global _GRAPH
  _GRAPH.graph = None
  ops.reset_default_graph()
  reset_uids()
  _SESSION.session = None
  graph = get_graph()
  with graph.as_default():
    _DUMMY_EAGER_GRAPH.learning_phase_is_set = False
    _GRAPH_LEARNING_PHASES.clear()
    # Create the learning phase placeholder in graph using the default factory.
    _GRAPH_LEARNING_PHASES.setdefault(graph)
    _GRAPH_VARIABLES.pop(graph, None)
    _GRAPH_TF_OPTIMIZERS.pop(graph, None)
  if context.executing_eagerly():
    # Clear pending nodes in eager executors, kernel caches and step_containers.
    context.context().clear_kernel_cache()


@doc_controls.do_not_generate_docs
def manual_variable_initialization(value):
  # [FUNCTION DOC]
  # Sets the module-wide flag controlling whether variables are initialized automatically when a session
  # is obtained (false, the default) or left for the user to initialize (true).
  # Input value: boolean. Returns nothing; only mutates the global flag.
  """Sets the manual variable initialization flag.

  This boolean flag determines whether
  variables should be initialized
  as they are instantiated (default), or if
  the user should handle the initialization
  (e.g. via `tf.compat.v1.initialize_all_variables()`).

  Args:
      value: Python boolean.
  """
  global _MANUAL_VAR_INIT
  _MANUAL_VAR_INIT = value


@doc_controls.do_not_generate_docs
def learning_phase():
  # [FUNCTION DOC]
  # Returns the learning phase flag (training versus inference) used as a default by Keras layers.
  # No inputs. If the current default graph is the Keras internal graph, returns the symbolic learning
  # phase of that graph; otherwise enters an initialization scope and returns the learning phase
  # registered for the current context (a Python integer in eager mode when it was set, or a
  # placeholder tensor with default value false in graph mode). Before returning, marks the current
  # function graph (if any) as unsaveable when the phase is a placeholder, so that exporting a
  # SavedModel raises a clear error. Side effects: may create the default learning phase entry in the
  # registry and mark graphs unsaveable.
  """Returns the learning phase flag.

  The learning phase flag is a bool tensor (0 = test, 1 = train)
  to be passed as input to any Keras function
  that uses a different behavior at train time and test time.

  Returns:
      Learning phase (scalar integer tensor or Python integer).
  """
  graph = ops.get_default_graph()
  if graph is getattr(_GRAPH, 'graph', None):
    # Don't enter an init_scope for the learning phase if eager execution
    # is enabled but we're inside the Keras workspace graph.
    learning_phase = symbolic_learning_phase()
  else:
    with ops.init_scope():
      # We always check & set the learning phase inside the init_scope,
      # otherwise the wrong default_graph will be used to look up the learning
      # phase inside of functions & defuns.
      #
      # This is because functions & defuns (both in graph & in eager mode)
      # will always execute non-eagerly using a function-specific default
      # subgraph.
      learning_phase = _GRAPH_LEARNING_PHASES[None]
  _mark_func_graph_as_unsaveable(graph, learning_phase)
  return learning_phase


def global_learning_phase_is_set():
  # [FUNCTION DOC]
  # Returns whether the learning phase was explicitly set in the current thread (the flag stored on the
  # dummy eager graph object). No inputs, no side effects.
  return _DUMMY_EAGER_GRAPH.learning_phase_is_set


def _mark_func_graph_as_unsaveable(graph, learning_phase):
  # [FUNCTION DOC]
  # HELPER: used by learning_phase to protect SavedModel export.
  # Inputs: graph (a regular graph or function graph) and learning_phase (placeholder or integer).
  # If the graph is a function-building graph and the learning phase is a placeholder, marks the graph
  # as unsaveable with a message telling the user to set the learning phase to a constant before saving.
  # Returns nothing; mutates the graph.
  """Mark func graph as unsaveable due to use of symbolic keras learning phase.

  Functions that capture the symbolic learning phase cannot be exported to
  SavedModel. Mark the funcgraph as unsaveable, so that an error will be raised
  if it is exported.

  Args:
    graph: Graph or FuncGraph object.
    learning_phase: Learning phase placeholder or int defined in the graph.
  """
  if graph.building_function and is_placeholder(learning_phase):
    graph.mark_as_unsaveable(
        'The keras learning phase placeholder was used inside a function. '
        'Exporting placeholders is not supported when saving out a SavedModel. '
        'Please call `tf.keras.backend.set_learning_phase(0)` in the function '
        'to set the learning phase to a constant value.')


def symbolic_learning_phase():
  # [FUNCTION DOC]
  # Returns the symbolic learning phase entry of the Keras graph (the current graph in graph mode, the
  # internal Keras graph in eager mode), creating it through the registry default factory if missing.
  # No inputs. Side effect: may insert a default entry into the global learning phase registry.
  graph = get_graph()
  with graph.as_default():
    return _GRAPH_LEARNING_PHASES[graph]


def _default_learning_phase():
  # [FUNCTION DOC]
  # HELPER: default factory of the learning phase registry.
  # Returns the integer zero when executing eagerly; in graph mode returns a scalar boolean placeholder
  # with default value false, created under an empty name scope and named for the Keras learning phase.
  # No inputs, no other side effects.
  if context.executing_eagerly():
    return 0
  else:
    with name_scope(''):
      return array_ops.placeholder_with_default(
          False, shape=(), name='keras_learning_phase')


@doc_controls.do_not_generate_docs
def set_learning_phase(value):
  # [FUNCTION DOC]
  # Deprecated public API that sets the learning phase to a fixed value for the current context.
  # Input value: integer zero (test) or one (train). Emits a deprecation warning and then delegates to
  # the internal non-warning setter, which raises a value error for any other value. Returns nothing.
  # Side effects: warning, and mutation of the global learning phase registry and thread flag.
  """Sets the learning phase to a fixed value.

  The backend learning phase affects any code that calls
  `backend.learning_phase()`
  In particular, all Keras built-in layers use the learning phase as the default
  for the `training` arg to `Layer.__call__`.

  User-written layers and models can achieve the same behavior with code that
  looks like:

  ```python
    def call(self, inputs, training=None):
      if training is None:
        training = backend.learning_phase()
  ```

  Args:
      value: Learning phase value, either 0 or 1 (integers).
             0 = test, 1 = train

  Raises:
      ValueError: if `value` is neither `0` nor `1`.
  """
  warnings.warn('`tf.keras.backend.set_learning_phase` is deprecated and '
                'will be removed after 2020-10-11. To update it, simply '
                'pass a True/False value to the `training` argument of the '
                '`__call__` method of your layer or model.')
  deprecated_internal_set_learning_phase(value)


def deprecated_internal_set_learning_phase(value):
  # [FUNCTION DOC]
  # HELPER: warning-free implementation behind set_learning_phase, also used by SavedModel saving.
  # Input value: integer zero or one; any other value raises a value error. Within an initialization
  # scope, when executing eagerly it marks the thread's learning phase as set and records the value
  # for the eager dummy key; in all cases it records the value for the current Keras graph. Returns
  # nothing; mutates the global learning phase registry.
  """A deprecated internal implementation of set_learning_phase.

  This method is an internal-only version of `set_learning_phase` that
  does not raise a deprecation error. It is required because
  saved_model needs to keep working with user code that uses the deprecated
  learning phase methods until those APIs are fully removed from the public API.

  Specifically SavedModel saving needs to make sure the learning phase is 0
  during tracing even if users overwrote it to a different value.

  But, we don't want to raise deprecation warnings for users when savedmodel
  sets learning phase just for compatibility with code that relied on
  explicitly setting the learning phase for other values.

  Args:
      value: Learning phase value, either 0 or 1 (integers). 0 = test, 1 = train

  Raises:
      ValueError: if `value` is neither `0` nor `1`.
  """
  global _GRAPH_LEARNING_PHASES  # pylint: disable=global-variable-not-assigned
  if value not in {0, 1}:
    raise ValueError('Expected learning phase to be 0 or 1.')
  with ops.init_scope():
    if context.executing_eagerly():
      # In an eager context, the learning phase values applies to both the eager
      # context and the internal Keras graph.
      _DUMMY_EAGER_GRAPH.learning_phase_is_set = True
      _GRAPH_LEARNING_PHASES[_DUMMY_EAGER_GRAPH.key] = value
    _GRAPH_LEARNING_PHASES[get_graph()] = value


@tf_contextlib.contextmanager
@doc_controls.do_not_generate_docs
def learning_phase_scope(value):
  # [FUNCTION DOC]
  # Deprecated context manager that temporarily sets the learning phase and restores it on exit.
  # Input value: zero or one (a value error comes from the internal scope otherwise). Emits a
  # deprecation warning, then enters the internal scope and yields nothing; the try/finally around the
  # yield does nothing. Side effects: temporary mutation of the learning phase registry.
  """Provides a scope within which the learning phase is equal to `value`.

  The learning phase gets restored to its original value upon exiting the scope.

  Args:
     value: Learning phase value, either 0 or 1 (integers).
            0 = test, 1 = train

  Yields:
    None.

  Raises:
     ValueError: if `value` is neither `0` nor `1`.
  """
  warnings.warn('`tf.keras.backend.learning_phase_scope` is deprecated and '
                'will be removed after 2020-10-11. To update it, simply '
                'pass a True/False value to the `training` argument of the '
                '`__call__` method of your layer or model.')
  with deprecated_internal_learning_phase_scope(value):
    try:
      yield
    finally:
      pass


@tf_contextlib.contextmanager
def deprecated_internal_learning_phase_scope(value):
  # [FUNCTION DOC]
  # HELPER: warning-free context manager behind learning_phase_scope, used by SavedModel saving.
  # Input value: zero or one, otherwise a value error is raised before anything changes. Remembers the
  # previous eager and graph learning phase entries and the thread's set flag, sets the new value, yields
  # nothing, and in the cleanup restores or deletes the entries and flag so the state is exactly as it
  # was before. Side effects: temporarily mutates the global registry.
  """An internal-only version of `learning_phase_scope`.

  Unlike the public method, this method does not raise a deprecation warning.
  This is needed because saved model saving needs to set learning phase
  to maintain compatibility
  with code that sets/gets the learning phase, but saved model
  saving itself shouldn't raise a deprecation warning.

  We can get rid of this method and its usages when the public API is
  removed.

  Args:
     value: Learning phase value, either 0 or 1 (integers). 0 = test, 1 = train

  Yields:
    None.

  Raises:
     ValueError: if `value` is neither `0` nor `1`.
  """
  global _GRAPH_LEARNING_PHASES  # pylint: disable=global-variable-not-assigned
  if value not in {0, 1}:
    raise ValueError('Expected learning phase to be 0 or 1.')

  with ops.init_scope():
    if context.executing_eagerly():
      previous_eager_value = _GRAPH_LEARNING_PHASES.get(
          _DUMMY_EAGER_GRAPH.key, None)
    previous_graph_value = _GRAPH_LEARNING_PHASES.get(get_graph(), None)

  learning_phase_previously_set = _DUMMY_EAGER_GRAPH.learning_phase_is_set
  try:
    deprecated_internal_set_learning_phase(value)
    yield
  finally:
    # Restore learning phase to initial value.
    if not learning_phase_previously_set:
      _DUMMY_EAGER_GRAPH.learning_phase_is_set = False
    with ops.init_scope():
      if context.executing_eagerly():
        if previous_eager_value is not None:
          _GRAPH_LEARNING_PHASES[_DUMMY_EAGER_GRAPH.key] = previous_eager_value
        elif _DUMMY_EAGER_GRAPH.key in _GRAPH_LEARNING_PHASES:
          del _GRAPH_LEARNING_PHASES[_DUMMY_EAGER_GRAPH.key]

      graph = get_graph()
      if previous_graph_value is not None:
        _GRAPH_LEARNING_PHASES[graph] = previous_graph_value
      elif graph in _GRAPH_LEARNING_PHASES:
        del _GRAPH_LEARNING_PHASES[graph]


@tf_contextlib.contextmanager
def eager_learning_phase_scope(value):
  # [FUNCTION DOC]
  # HELPER: internal context manager used for eager and function execution.
  # KNOWN ISSUE: the docstring says a value error is raised for values other than zero or one, but the
  # code only uses assertions, so invalid values raise an assertion error (and nothing at all when
  # assertions are disabled). It also asserts that eager execution is active outside functions.
  # Input value: zero or one. Remembers the current learning phase if one was explicitly set, writes the
  # value for the eager dummy key, yields nothing, and in the cleanup restores the previous value or
  # deletes the key. Side effects: temporary mutation of the learning phase registry.
  """Internal scope that sets the learning phase in eager / tf.function only.

  Args:
      value: Learning phase value, either 0 or 1 (integers).
             0 = test, 1 = train

  Yields:
    None.

  Raises:
     ValueError: if `value` is neither `0` nor `1`.
  """
  global _GRAPH_LEARNING_PHASES  # pylint: disable=global-variable-not-assigned
  assert value in {0, 1}
  assert ops.executing_eagerly_outside_functions()
  global_learning_phase_was_set = global_learning_phase_is_set()
  if global_learning_phase_was_set:
    previous_value = learning_phase()
  try:
    _GRAPH_LEARNING_PHASES[_DUMMY_EAGER_GRAPH.key] = value
    yield
  finally:
    # Restore learning phase to initial value or unset.
    if global_learning_phase_was_set:
      _GRAPH_LEARNING_PHASES[_DUMMY_EAGER_GRAPH.key] = previous_value
    else:
      del _GRAPH_LEARNING_PHASES[_DUMMY_EAGER_GRAPH.key]


def _as_graph_element(obj):
  # [FUNCTION DOC]
  # HELPER: used by the graph selection and callable creation code.
  # Input obj: any object. If it exposes a callable conversion hook for graph elements, returns the
  # result of calling it; otherwise returns nothing (null value). No side effects.
  """Convert `obj` to a graph element if possible, otherwise return `None`.

  Args:
    obj: Object to convert.

  Returns:
    The result of `obj._as_graph_element()` if that method is available;
        otherwise `None`.
  """
  conv_fn = getattr(obj, '_as_graph_element', None)
  if conv_fn and callable(conv_fn):
    return conv_fn()
  return None


def _assert_same_graph(original_item, item):
  # [FUNCTION DOC]
  # HELPER: consistency check used by the graph selection logic.
  # Inputs: original_item and item, any objects that may carry a graph attribute. If both have a graph
  # and the graphs differ, raises a value error that names the items and graphs; otherwise returns
  # nothing.
  """Fail if the 2 items are from different graphs.

  Args:
    original_item: Original item to check against.
    item: Item to check.

  Raises:
    ValueError: if graphs do not match.
  """
  original_graph = getattr(original_item, 'graph', None)
  graph = getattr(item, 'graph', None)
  if original_graph and graph and original_graph is not graph:
    raise ValueError(
        '%s must be from the same graph as %s (graphs are %s and %s).' %
        (item, original_item, graph, original_graph))


def _current_graph(op_input_list, graph=None):
  # [FUNCTION DOC]
  # HELPER: picks the graph used to build ops or sessions from a list of inputs.
  # KNOWN ISSUE: the docstring says a type error is raised when the input list is not a list or tuple,
  # but the code converts any iterable (including generators) into a tuple and only raises a type error
  # when the explicit graph argument is not a graph.
  # Inputs: op_input_list (iterable of tensors, operations or convertible objects) and optional graph.
  # If the default graph is building a function it is returned immediately. Otherwise validates that the
  # explicit graph is a graph, then scans the inputs: graph elements are taken from plain tensors,
  # operations and composite tensors (not subclasses of tensor) or through the conversion hook; the
  # first one found defines the graph when none was given and all others must share it (value error on
  # mismatch). Returns the chosen graph or, failing that, the default graph.
  """Returns the appropriate graph to use for the given inputs.

  This library method provides a consistent algorithm for choosing the graph
  in which an Operation should be constructed:

  1. If the default graph is being used to construct a function, we
     use the default graph.
  2. If the "graph" is specified explicitly, we validate that all of the inputs
     in "op_input_list" are compatible with that graph.
  3. Otherwise, we attempt to select a graph from the first Operation-
     or Tensor-valued input in "op_input_list", and validate that all other
     such inputs are in the same graph.
  4. If the graph was not specified and it could not be inferred from
     "op_input_list", we attempt to use the default graph.

  Args:
    op_input_list: A list of inputs to an operation, which may include `Tensor`,
      `Operation`, and other objects that may be converted to a graph element.
    graph: (Optional) The explicit graph to use.

  Raises:
    TypeError: If op_input_list is not a list or tuple, or if graph is not a
      Graph.
    ValueError: If a graph is explicitly passed and not all inputs are from it,
      or if the inputs are from multiple graphs, or we could not find a graph
      and there was no default graph.

  Returns:
    The appropriate graph to use for the given inputs.

  """
  current_default_graph = ops.get_default_graph()
  if current_default_graph.building_function:
    return current_default_graph

  op_input_list = tuple(op_input_list)  # Handle generators correctly
  if graph and not isinstance(graph, ops.Graph):
    raise TypeError('Input graph needs to be a Graph: %s' % (graph,))

  # 1. We validate that all of the inputs are from the same graph. This is
  #    either the supplied graph parameter, or the first one selected from one
  #    the graph-element-valued inputs. In the latter case, we hold onto
  #    that input in original_graph_element so we can provide a more
  #    informative error if a mismatch is found.
  original_graph_element = None
  for op_input in op_input_list:
    # Determine if this is a valid graph_element.
    # TODO(josh11b): Note that we exclude subclasses of Tensor. Need to clean this
    # up.
    if (isinstance(op_input, (
        ops.Operation, tensor_lib.Tensor, composite_tensor.CompositeTensor)) and
        ((not isinstance(op_input, tensor_lib.Tensor))
         or type(op_input) == tensor_lib.Tensor)):  # pylint: disable=unidiomatic-typecheck
      graph_element = op_input
    else:
      graph_element = _as_graph_element(op_input)

    if graph_element is not None:
      if not graph:
        original_graph_element = graph_element
        graph = getattr(graph_element, 'graph', None)
      elif original_graph_element is not None:
        _assert_same_graph(original_graph_element, graph_element)
      elif graph_element.graph is not graph:
        raise ValueError('%s is not from the passed-in graph.' % graph_element)

  # 2. If all else fails, we use the default graph, which is always there.
  return graph or current_default_graph


def _get_session(op_input_list=()):
  # [FUNCTION DOC]
  # HELPER: returns the session for the current thread without initializing variables.
  # Input op_input_list: optional sequence of tensors/ops used to determine the graph. Returns the
  # default session when one exists. Otherwise raises a runtime error when called inside a graph
  # function, and creates (and caches in the thread-local holder) a new session if none is cached or
  # the cached session belongs to another graph; under a distribution strategy it asks the strategy
  # helper to configure and create the session, else builds one from the default session config.
  # Side effect: may create and store a session.
  """Returns the session object for the current thread."""
  global _SESSION
  default_session = ops.get_default_session()
  if default_session is not None:
    session = default_session
  else:
    if ops.inside_function():
      raise RuntimeError('Cannot get session inside Tensorflow graph function.')
    # If we don't have a session, or that session does not match the current
    # graph, create and cache a new session.
    if (getattr(_SESSION, 'session', None) is None or
        _SESSION.session.graph is not _current_graph(op_input_list)):
      # If we are creating the Session inside a tf.distribute.Strategy scope,
      # we ask the strategy for the right session options to use.
      if distribute_lib.has_strategy():
        configure_and_create_distributed_session(
            distribute_lib.get_strategy())
      else:
        _SESSION.session = session_module.Session(
            config=get_default_session_config())
    session = _SESSION.session
  return session


def get_session(op_input_list=()):
  # [FUNCTION DOC]
  # Returns the TensorFlow session to be used by the backend, creating the global Keras session if
  # necessary. Input op_input_list: optional sequence of tensors or ops used to determine the current
  # graph (default graph otherwise). Unless manual variable initialization is on, initializes any
  # tracked variables that have not been initialized yet inside the session's graph. Returns the
  # session. Side effects: session creation, variable initialization.
  """Returns the TF session to be used by the backend.

  If a default TensorFlow session is available, we will return it.

  Else, we will return the global Keras session assuming it matches
  the current graph.

  If no global Keras session exists at this point:
  we will create a new global session.

  Note that you can manually set the global session
  via `K.set_session(sess)`.

  Args:
      op_input_list: An option sequence of tensors or ops, which will be used
        to determine the current graph. Otherwise the default graph will be
        used.

  Returns:
      A TensorFlow session.
  """
  session = _get_session(op_input_list)
  if not _MANUAL_VAR_INIT:
    with session.graph.as_default():
      _initialize_variables(session)
  return session


def get_graph():
  # [FUNCTION DOC]
  # Returns the graph in which Keras builds symbolic objects. When executing eagerly, returns the
  # thread-local internal function graph used for functional model construction, creating it (named for
  # Keras) on first use; otherwise returns the current default graph. No inputs. Side effect: may create
  # and cache the internal graph.
  if context.executing_eagerly():
    global _GRAPH
    if not getattr(_GRAPH, 'graph', None):
      _GRAPH.graph = func_graph.FuncGraph('keras_graph')
    return _GRAPH.graph
  else:
    return ops.get_default_graph()


@tf_contextlib.contextmanager
def _scratch_graph(graph=None):
  # [FUNCTION DOC]
  # HELPER: context manager supporting function creation in eager mode and distribution strategies.
  # Input graph: optional graph to use as the scratch graph. If a scratch graph is already active it is
  # yielded (a value error is raised if a different graph was also passed). Otherwise uses the given
  # graph or creates a new function graph for scratch use, stores it in the thread-local holder, yields
  # it, and clears the holder afterwards (even on error). Side effect: temporary thread-local state.
  """Retrieve a shared and temporary func graph.

  The eager execution path lifts a subgraph from the keras global graph into
  a scratch graph in order to create a function. DistributionStrategies, in
  turn, constructs multiple functions as well as a final combined function. In
  order for that logic to work correctly, all of the functions need to be
  created on the same scratch FuncGraph.

  Args:
    graph: A graph to be used as the current scratch graph. If not set then
      a scratch graph will either be retrieved or created:

  Yields:
    The current scratch graph.
  """
  global _CURRENT_SCRATCH_GRAPH
  scratch_graph = getattr(_CURRENT_SCRATCH_GRAPH, 'graph', None)
  # If scratch graph and `graph` are both configured, they must match.
  if (scratch_graph is not None and graph is not None and
      scratch_graph is not graph):
    raise ValueError('Multiple scratch graphs specified.')

  if scratch_graph:
    yield scratch_graph
    return

  graph = graph or func_graph.FuncGraph('keras_scratch_graph')
  try:
    _CURRENT_SCRATCH_GRAPH.graph = graph
    yield graph
  finally:
    _CURRENT_SCRATCH_GRAPH.graph = None


def set_session(session):
  # [FUNCTION DOC]
  # Sets the global (thread-local) session used by Keras to the given session object. Input session: a
  # TensorFlow session. Returns nothing.
  """Sets the global TensorFlow session.

  Args:
      session: A TF Session.
  """
  global _SESSION
  _SESSION.session = session


def get_default_session_config():
  # [FUNCTION DOC]
  # Builds the default session configuration. Logs a warning if the OMP_NUM_THREADS environment
  # variable is set (since it is no longer used), obtains the current configuration proto from the eager
  # context, turns on soft device placement in it and returns it. No inputs. Side effects: logging and
  # mutation of the configuration object it obtained.
  if os.environ.get('OMP_NUM_THREADS'):
    logging.warning(
        'OMP_NUM_THREADS is no longer used by the default Keras config. '
        'To configure the number of threads, use tf.config.threading APIs.')

  config = get_config()
  config.allow_soft_placement = True

  return config


def get_default_graph_uid_map():
  # [FUNCTION DOC]
  # Returns the name-to-counter mapping for the current default graph, creating and registering an empty
  # integer counter mapping when the graph has none yet. No inputs. Side effect: may add an entry to the
  # global per-graph registry.
  graph = ops.get_default_graph()
  name_uid_map = PER_GRAPH_OBJECT_NAME_UIDS.get(graph, None)
  if name_uid_map is None:
    name_uid_map = collections.defaultdict(int)
    PER_GRAPH_OBJECT_NAME_UIDS[graph] = name_uid_map
  return name_uid_map


# DEVICE MANIPULATION


class _TfDeviceCaptureOp:
  # [CLASS DOC]
  # HELPER: stand-in operation object used to read the device scope that is currently active.
  # Holds a single attribute named device. Instances are passed to the graph's device-function
  # application routine, which calls the setter methods with the device chosen by the active device
  # scopes.
  """Class for capturing the TF device scope."""

  def __init__(self):
    # [FUNCTION DOC]
    # HELPER: constructor of the device capture stand-in. Initializes the stored device to nothing
    # (null value). No inputs; returns nothing.
    self.device = None

  def _set_device(self, device):
    # [FUNCTION DOC]
    # HELPER: callback invoked by the graph when it applies device functions.
    # Input device: either a device specification object or a string. A specification object is
    # converted to its string form; the result is stored as the captured device. Returns nothing.
    """This method captures TF's explicit device scope setting."""
    if isinstance(device, device_spec.DeviceSpecV2):
      device = device.to_string()
    self.device = device

  def _set_device_from_string(self, device_str):
    # [FUNCTION DOC]
    # HELPER: callback variant of the device setter for string input. Input device_str: a device string,
    # stored as the captured device. Returns nothing.
    self.device = device_str


def _get_current_tf_device():
  # [FUNCTION DOC]
  # HELPER: used by the NCHW-support check.
  # KNOWN ISSUE: the docstring says it returns a string, or nothing when the device scope is not
  # explicitly set. The code always returns a device specification object (parsed from the captured
  # device string, which may be empty or null) and never returns nothing itself.
  # Creates a capture operation, asks the Keras graph to apply its device functions to it, and parses
  # the captured device with the version-2 or version-1 device specification parser depending on
  # whether TF2 behavior is enabled. No inputs.
  """Return explicit device of current context, otherwise returns `None`.

  Returns:
      If the current device scope is explicitly set, it returns a string with
      the device (`CPU` or `GPU`). If the scope is not explicitly set, it will
      return `None`.
  """
  graph = get_graph()
  op = _TfDeviceCaptureOp()
  graph._apply_device_functions(op)
  if tf2.enabled():
    return device_spec.DeviceSpecV2.from_string(op.device)
  else:
    return device_spec.DeviceSpecV1.from_string(op.device)


def _is_current_explicit_device(device_type):
  # [FUNCTION DOC]
  # HELPER: used to decide whether NCHW operations are supported.
  # Input device_type: string CPU or GPU (case-insensitive); any other value raises a value error.
  # Returns true when the current device scope specifies a device of that type, false otherwise.
  """Check if the current device is explicitly set on the device type specified.

  Args:
      device_type: A string containing `GPU` or `CPU` (case-insensitive).

  Returns:
      A boolean indicating if the current device scope is explicitly set on the
      device type.

  Raises:
      ValueError: If the `device_type` string indicates an unsupported device.
  """
  device_type = device_type.upper()
  if device_type not in ['CPU', 'GPU']:
    raise ValueError('`device_type` should be either "CPU" or "GPU".')
  device = _get_current_tf_device()
  return device is not None and device.device_type == device_type.upper()


def _get_available_gpus():
  # [FUNCTION DOC]
  # HELPER: used by the NCHW-support check.
  # No inputs. Returns a list of names of logical GPU devices. When eager execution is active outside
  # functions this asks the configuration API directly; otherwise it lazily lists devices of the Keras
  # session once, caches them in a module level variable (assuming devices never change) and filters
  # for GPUs. Side effect: may create a session and fill the device cache.
  """Get a list of available GPU devices (formatted as strings).

  Returns:
      A list of available GPU devices.
  """
  if ops.executing_eagerly_outside_functions():
    # Returns names of devices directly.
    return [d.name for d in config.list_logical_devices('GPU')]

  global _LOCAL_DEVICES
  if _LOCAL_DEVICES is None:
    _LOCAL_DEVICES = get_session().list_devices()
  return [x.name for x in _LOCAL_DEVICES if x.device_type == 'GPU']


def _has_nchw_support():
  # [FUNCTION DOC]
  # HELPER: decides whether channels-first (NCHW) ops can run in the current scope.
  # No inputs. Returns true only if the scope is not explicitly placed on the CPU and at least one GPU
  # is available (ops then soft-place on the GPU). Used by convolution, pooling and normalization code.
  """Check whether the current scope supports NCHW ops.

  TensorFlow does not support NCHW on CPU. Therefore we check if we are not
  explicitly put on
  CPU, and have GPUs available. In this case there will be soft-placing on the
  GPU device.

  Returns:
      bool: if the current scope device placement would support nchw
  """
  explicitly_on_cpu = _is_current_explicit_device('CPU')
  gpus_available = bool(_get_available_gpus())
  return not explicitly_on_cpu and gpus_available


# VARIABLE MANIPULATION


def _constant_to_tensor(x, dtype):
  # [FUNCTION DOC]
  # HELPER: fast conversion of Python numbers, numpy arrays or lists to a constant tensor.
  # Inputs: x (value to convert) and dtype (target type). Returns a constant tensor of that type;
  # handles fewer cases than the general conversion helper.
  """Convert the input `x` to a tensor of type `dtype`.

  This is slightly faster than the _to_tensor function, at the cost of
  handling fewer cases.

  Args:
      x: An object to be converted (numpy arrays, floats, ints and lists of
        them).
      dtype: The destination type.

  Returns:
      A tensor.
  """
  return constant_op.constant(x, dtype=dtype)


def _to_tensor(x, dtype):
  # [FUNCTION DOC]
  # HELPER: general conversion of values to a tensor with a given type.
  # Inputs: x (numpy array, list, tensor) and dtype (target type). Returns the tensor produced by the
  # dispatching tensor conversion routine.
  """Convert the input `x` to a tensor of type `dtype`.

  Args:
      x: An object to be converted (numpy array, list, tensors).
      dtype: The destination type.

  Returns:
      A tensor.
  """
  return tensor_conversion.convert_to_tensor_v2_with_dispatch(x, dtype=dtype)


@doc_controls.do_not_generate_docs
def is_sparse(tensor):
  # [FUNCTION DOC]
  # Returns whether the given tensor is a sparse tensor. Input tensor: any tensor-like object. Uses its
  # type specification when it has one (true if that is a sparse tensor spec), otherwise checks the
  # object's class. Returns a boolean. No side effects.
  """Returns whether a tensor is a sparse tensor.

  Args:
      tensor: A tensor instance.

  Returns:
      A boolean.

  Example:


  >>> a = tf.keras.backend.placeholder((2, 2), sparse=False)
  >>> print(tf.keras.backend.is_sparse(a))
  False
  >>> b = tf.keras.backend.placeholder((2, 2), sparse=True)
  >>> print(tf.keras.backend.is_sparse(b))
  True

  """
  spec = getattr(tensor, '_type_spec', None)
  if spec is not None:
    return isinstance(spec, sparse_tensor.SparseTensorSpec)
  return isinstance(tensor, sparse_tensor.SparseTensor)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def to_dense(tensor):
  # [FUNCTION DOC]
  # Converts a sparse tensor to a dense tensor and returns it; any other input is returned unchanged.
  # Input tensor: potentially sparse tensor. Supports TensorFlow API dispatch.
  """Converts a sparse tensor into a dense tensor and returns it.

  Args:
      tensor: A tensor instance (potentially sparse).

  Returns:
      A dense tensor.

  Examples:


  >>> b = tf.keras.backend.placeholder((2, 2), sparse=True)
  >>> print(tf.keras.backend.is_sparse(b))
  True
  >>> c = tf.keras.backend.to_dense(b)
  >>> print(tf.keras.backend.is_sparse(c))
  False

  """
  if is_sparse(tensor):
    return sparse_ops.sparse_tensor_to_dense(tensor)
  else:
    return tensor


@doc_controls.do_not_generate_docs
def name_scope(name):
  # [FUNCTION DOC]
  # Context manager that pushes a TensorFlow name scope so ops created inside get the given prefix.
  # Input name: prefix string. Returns the version-2 name scope context manager. Side effect on entry:
  # changes the active name scope.
  """A context manager for use when defining a Python op.

  This context manager pushes a name scope, which will make the name of all
  operations added within it have a prefix.

  For example, to define a new Python op called `my_op`:


  def my_op(a):
    with tf.name_scope("MyOp") as scope:
      a = tf.convert_to_tensor(a, name="a")
      # Define some computation that uses `a`.
      return foo_op(..., name=scope)


  When executed, the Tensor `a` will have the name `MyOp/a`.

  Args:
    name: The prefix to use on all names created within the name scope.

  Returns:
    Name scope context manager.
  """
  return ops.name_scope_v2(name)


@doc_controls.do_not_generate_docs
def variable(value, dtype=None, name=None, constraint=None):
  # [FUNCTION DOC]
  # Creates a variable with Keras metadata from an initial value and returns it.
  # Inputs: value (numpy array, tensor-like or scipy sparse matrix), dtype (defaults to floatx), name
  # (optional), constraint (optional projection function applied after optimizer updates). A value that
  # has a conversion hook to coordinate format (a scipy sparse matrix) is turned into a sparse tensor
  # carrying a recorded Keras shape and returned directly without being tracked. Otherwise builds a
  # TensorFlow variable, records the Keras shape attribute (from the numpy shape or by querying the
  # integer shape), registers it for initialization tracking (graph mode only) and returns it.
  # Side effects: variable creation and registration in the global variable cache.
  """Instantiates a variable and returns it.

  Args:
      value: Numpy array, initial value of the tensor.
      dtype: Tensor type.
      name: Optional name string for the tensor.
      constraint: Optional projection function to be
          applied to the variable after an optimizer update.

  Returns:
      A variable instance (with Keras metadata included).

  Examples:

  >>> val = np.array([[1, 2], [3, 4]])
  >>> kvar = tf.keras.backend.variable(value=val, dtype='float64',
  ...                                  name='example_var')
  >>> tf.keras.backend.dtype(kvar)
  'float64'
  >>> print(kvar)
  <tf.Variable 'example_var:...' shape=(2, 2) dtype=float64, numpy=
    array([[1., 2.],
           [3., 4.]])>

  """
  if dtype is None:
    dtype = floatx()
  if hasattr(value, 'tocoo'):
    sparse_coo = value.tocoo()
    indices = np.concatenate((np.expand_dims(sparse_coo.row, 1), np.expand_dims(
        sparse_coo.col, 1)), 1)
    v = sparse_tensor.SparseTensor(
        indices=indices, values=sparse_coo.data, dense_shape=sparse_coo.shape)
    v._keras_shape = sparse_coo.shape
    return v
  v = variables_module.Variable(
      value,
      dtype=dtypes_module.as_dtype(dtype),
      name=name,
      constraint=constraint)
  if isinstance(value, np.ndarray):
    v._keras_shape = value.shape
  elif hasattr(value, 'shape'):
    v._keras_shape = int_shape(value)
  track_variable(v)
  return v


def track_tf_optimizer(tf_optimizer):
  # [FUNCTION DOC]
  # Registers a TensorFlow optimizer so its variables get initialized with the session. Input
  # tf_optimizer: optimizer wrapper object. Does nothing when executing eagerly; otherwise adds it to
  # the global per-graph optimizer cache. Returns nothing.
  """Tracks the given TF optimizer for initialization of its variables."""
  if context.executing_eagerly():
    return
  optimizers = _GRAPH_TF_OPTIMIZERS[None]
  optimizers.add(tf_optimizer)


def track_variable(v):
  # [FUNCTION DOC]
  # Registers a variable for lazy initialization. Input v: a variable. Does nothing in eager mode;
  # otherwise adds it to the set of tracked variables for its own graph (or the Keras graph when it has
  # none). Returns nothing; mutates the global registry.
  """Tracks the given variable for initialization."""
  if context.executing_eagerly():
    return
  graph = v.graph if hasattr(v, 'graph') else get_graph()
  _GRAPH_VARIABLES[graph].add(v)


def observe_object_name(name):
  # [FUNCTION DOC]
  # Records a name so the unique name generator can avoid it when asked to. Input name: string. Adds it
  # to the global observed-name set; returns nothing.
  """Observe a name and make sure it won't be used by `unique_object_name`."""
  OBSERVED_NAMES.add(name)


def unique_object_name(name,
                       name_uid_map=None,
                       avoid_names=None,
                       namespace='',
                       zero_based=False,
                       avoid_observed_names=False):
  # [FUNCTION DOC]
  # Makes an object name unique within a graph (or arbitrary namespace) by appending a counter.
  # Inputs: name (base string); name_uid_map (optional counter mapping, default the current graph's);
  # avoid_names (optional set or mapping of names that must not be produced; defaults to the observed
  # names if avoid_observed_names is true, else empty); namespace (string scoping the counters, empty
  # by default); zero_based (when true the first name has no suffix and later ones are numbered from
  # one; when false numbering starts at one); avoid_observed_names (boolean). Loops, incrementing the
  # counter for the (namespace, name) pair, until the proposed name is not in the avoid set. Returns the
  # unique string. Side effect: mutates the counter mapping.
  """Makes a object name (or arbitrary string) unique within a TensorFlow graph.

  Args:
    name: String name to make unique.
    name_uid_map: An optional defaultdict(int) to use when creating unique
      names. If None (default), uses a per-Graph dictionary.
    avoid_names: An optional set or dict with names which should not be used. If
      None (default), don't avoid any names unless `avoid_observed_names` is
      True.
    namespace: Gets a name which is unique within the (graph, namespace). Layers
      which are not Networks use a blank namespace and so get graph-global
      names.
    zero_based: If True, name sequences start with no suffix (e.g. "dense",
      "dense_1"). If False, naming is one-based ("dense_1", "dense_2").
    avoid_observed_names: If True, avoid any names that have been observed by
      `backend.observe_object_name`.

  Returns:
    Unique string name.

  Example:


  unique_object_name('dense')  # dense_1
  unique_object_name('dense')  # dense_2

  """
  if name_uid_map is None:
    name_uid_map = get_default_graph_uid_map()
  if avoid_names is None:
    if avoid_observed_names:
      avoid_names = OBSERVED_NAMES
    else:
      avoid_names = set()
  proposed_name = None
  while proposed_name is None or proposed_name in avoid_names:
    name_key = (namespace, name)
    if zero_based:
      number = name_uid_map[name_key]
      if number:
        proposed_name = name + '_' + str(number)
      else:
        proposed_name = name
      name_uid_map[name_key] += 1
    else:
      name_uid_map[name_key] += 1
      proposed_name = name + '_' + str(name_uid_map[name_key])
  return proposed_name


def _get_variables(graph=None):
  # [FUNCTION DOC]
  # HELPER: collects the variables that may need initialization for a graph.
  # Input graph: graph key (or nothing for the current context); asserts graph mode. Returns the set of
  # tracked variables, after adding variables of every tracked TF optimizer of that graph (so the set is
  # mutated).
  """Returns variables corresponding to the given graph for initialization."""
  assert not context.executing_eagerly()
  variables = _GRAPH_VARIABLES[graph]
  for opt in _GRAPH_TF_OPTIMIZERS[graph]:
    variables.update(opt.optimizer.variables())
  return variables


def _initialize_variables(session):
  # [FUNCTION DOC]
  # HELPER: initializes uninitialized variables on the fly for a session.
  # Input session: a session. Takes the tracked variables of the Keras graph, considers those not yet
  # marked as initialized, asks the session which are actually uninitialized, marks every candidate as
  # initialized and runs the initializer for those that were uninitialized and have an initializer.
  # Returns nothing. Side effects: session runs and marking attributes on variables.
  """Utility to initialize uninitialized variables on the fly."""
  variables = _get_variables(get_graph())
  candidate_vars = []
  for v in variables:
    if not getattr(v, '_keras_initialized', False):
      candidate_vars.append(v)
  if candidate_vars:
    # This step is expensive, so we only run it on variables not already
    # marked as initialized.
    is_initialized = session.run(
        [variable_v1.is_variable_initialized(v) for v in candidate_vars])
    # TODO(kathywu): Some metric variables loaded from SavedModel are never
    # actually used, and do not have an initializer.
    should_be_initialized = [
        (not is_initialized[n]) and v.initializer is not None
        for n, v in enumerate(candidate_vars)]
    uninitialized_vars = []
    for flag, v in zip(should_be_initialized, candidate_vars):
      if flag:
        uninitialized_vars.append(v)
      v._keras_initialized = True
    if uninitialized_vars:
      session.run(variables_module.variables_initializer(uninitialized_vars))


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def constant(value, dtype=None, shape=None, name=None):
  # [FUNCTION DOC]
  # Creates a constant tensor. Inputs: value (constant value or list), dtype (defaults to floatx),
  # shape (optional dimensions), name (optional). Returns the constant tensor. Supports API dispatch.
  """Creates a constant tensor.

  Args:
      value: A constant value (or list)
      dtype: The type of the elements of the resulting tensor.
      shape: Optional dimensions of resulting tensor.
      name: Optional name for the tensor.

  Returns:
      A Constant Tensor.
  """
  if dtype is None:
    dtype = floatx()

  return constant_op.constant(value, dtype=dtype, shape=shape, name=name)


def is_keras_tensor(x):
  # [FUNCTION DOC]
  # Returns whether x is a Keras tensor (an output of a layer or an input object).
  # Input x: must be a tensor, variable, sparse or ragged tensor or Keras symbolic tensor; any other
  # type raises a value error saying a symbolic tensor was expected. When eager execution is active
  # outside functions, true only for the symbolic Keras tensor class; otherwise true when the object has
  # Keras history metadata. Returns a boolean.
  """Returns whether `x` is a Keras tensor.

  A "Keras tensor" is a tensor that was returned by a Keras layer,
  (`Layer` class) or by `Input`.

  Args:
      x: A candidate tensor.

  Returns:
      A boolean: Whether the argument is a Keras tensor.

  Raises:
      ValueError: In case `x` is not a symbolic tensor.

  Examples:

  >>> np_var = np.array([1, 2])
  >>> # A numpy array is not a symbolic tensor.
  >>> tf.keras.backend.is_keras_tensor(np_var)
  Traceback (most recent call last):
  ...
  ValueError: Unexpectedly found an instance of type `<class 'numpy.ndarray'>`.
  Expected a symbolic tensor instance.
  >>> keras_var = tf.keras.backend.variable(np_var)
  >>> # A variable created with the keras backend is not a Keras tensor.
  >>> tf.keras.backend.is_keras_tensor(keras_var)
  False
  >>> keras_placeholder = tf.keras.backend.placeholder(shape=(2, 4, 5))
  >>> # A placeholder is a Keras tensor.
  >>> tf.keras.backend.is_keras_tensor(keras_placeholder)
  True
  >>> keras_input = tf.keras.layers.Input([10])
  >>> # An Input is a Keras tensor.
  >>> tf.keras.backend.is_keras_tensor(keras_input)
  True
  >>> keras_layer_output = tf.keras.layers.Dense(10)(keras_input)
  >>> # Any Keras layer output is a Keras tensor.
  >>> tf.keras.backend.is_keras_tensor(keras_layer_output)
  True

  """
  if not isinstance(x,
                    (tensor_lib.Tensor, variables_module.Variable,
                     sparse_tensor.SparseTensor, ragged_tensor.RaggedTensor,
                     keras_tensor.KerasTensor)):
    raise ValueError('Unexpectedly found an instance of type `' + str(type(x)) +
                     '`. Expected a symbolic tensor instance.')
  if ops.executing_eagerly_outside_functions():
    return isinstance(x, keras_tensor.KerasTensor)
  return hasattr(x, '_keras_history')


@doc_controls.do_not_generate_docs
def placeholder(shape=None,
                ndim=None,
                dtype=None,
                sparse=False,
                name=None,
                ragged=False):
  # [FUNCTION DOC]
  # Creates a placeholder (symbolic input) and returns it.
  # Inputs: shape (tuple possibly with null entries), ndim (number of axes, used only if no shape is
  # given), dtype (defaults to floatx), sparse (boolean), name (optional), ragged (boolean; null shape
  # entries mean ragged dimensions). Raises a value error when both sparse and ragged are requested.
  # When eager execution is active outside functions it builds a type specification (sparse, ragged
  # with the last null dimension as ragged rank, or dense) and returns a Keras symbolic tensor made
  # from it. Otherwise, inside the Keras graph, builds a sparse, ragged (one placeholder per component)
  # or dense graph placeholder. If executing eagerly at the top level it additionally wraps the result
  # in an Input layer output and flags it as a backend placeholder so it can be used with function.
  """Instantiates a placeholder tensor and returns it.

  Args:
      shape: Shape of the placeholder
          (integer tuple, may include `None` entries).
      ndim: Number of axes of the tensor.
          At least one of {`shape`, `ndim`} must be specified.
          If both are specified, `shape` is used.
      dtype: Placeholder type.
      sparse: Boolean, whether the placeholder should have a sparse type.
      name: Optional name string for the placeholder.
      ragged: Boolean, whether the placeholder should have a ragged type.
          In this case, values of 'None' in the 'shape' argument represent
          ragged dimensions. For more information about RaggedTensors, see this
          [guide](https://www.tensorflow.org/guide/ragged_tensors).

  Raises:
      ValueError: If called with sparse = True and ragged = True.

  Returns:
      Tensor instance (with Keras metadata included).

  Examples:


  >>> input_ph = tf.keras.backend.placeholder(shape=(2, 4, 5))
  >>> input_ph
  <KerasTensor: shape=(2, 4, 5) dtype=float32 (created by layer ...)>

  """
  if sparse and ragged:
    raise ValueError(
        'Cannot set both sparse and ragged to True when creating a placeholder.'
    )
  if dtype is None:
    dtype = floatx()
  if not shape:
    if ndim:
      shape = (None,) * ndim
  if ops.executing_eagerly_outside_functions():
    if sparse:
      spec = sparse_tensor.SparseTensorSpec(
          shape=shape, dtype=dtype)
    elif ragged:
      ragged_rank = 0
      for i in range(1, len(shape)):
        # Hacky because could be tensorshape or tuple maybe?
        # Or just tensorshape?
        if shape[i] is None or (
            hasattr(shape[i], 'value') and
            shape[i].value is None):
          ragged_rank = i
      spec = ragged_tensor.RaggedTensorSpec(
          shape=shape, dtype=dtype, ragged_rank=ragged_rank)
    else:
      spec = tensor_lib.TensorSpec(
          shape=shape, dtype=dtype, name=name)
    x = keras_tensor.keras_tensor_from_type_spec(spec, name=name)
  else:
    with get_graph().as_default():
      if sparse:
        x = array_ops.sparse_placeholder(dtype, shape=shape, name=name)
      elif ragged:
        ragged_rank = 0
        for i in range(1, len(shape)):
          if shape[i] is None:
            ragged_rank = i
        type_spec = ragged_tensor.RaggedTensorSpec(
            shape=shape, dtype=dtype, ragged_rank=ragged_rank)
        def tensor_spec_to_placeholder(tensorspec):
          # [FUNCTION DOC]
          # HELPER: local mapper used while building ragged placeholders in graph mode.
          # Input tensorspec: type specification of one component. Returns a graph placeholder with the
          # spec's dtype and shape.
          return array_ops.placeholder(tensorspec.dtype, tensorspec.shape)
        x = nest.map_structure(tensor_spec_to_placeholder, type_spec,
                               expand_composites=True)
      else:
        x = array_ops.placeholder(dtype, shape=shape, name=name)

  if context.executing_eagerly():
    # Add keras_history connectivity information to the placeholder
    # when the placeholder is built in a top-level eager context
    # (intended to be used with keras.backend.function)
    from tensorflow.python.keras.engine import input_layer  # pylint: disable=g-import-not-at-top
    x = input_layer.Input(tensor=x)
    x._is_backend_placeholder = True

  return x


def is_placeholder(x):
  # [FUNCTION DOC]
  # Returns whether x is a placeholder. Input x: candidate object. With eager execution outside
  # functions, true if the object carries the backend-placeholder flag. Otherwise for extension types
  # (composite tensors) true if any flattened component is a placeholder; for ordinary tensors true if
  # the producing op type is a placeholder. Any missing attribute is treated as false. Returns boolean.
  """Returns whether `x` is a placeholder.

  Args:
      x: A candidate placeholder.

  Returns:
      Boolean.
  """
  try:
    if ops.executing_eagerly_outside_functions():
      return hasattr(x, '_is_backend_placeholder')
    from tensorflow.python.keras.utils import tf_utils  # pylint: disable=g-import-not-at-top
    if tf_utils.is_extension_type(x):
      flat_components = nest.flatten(x, expand_composites=True)
      return py_any(is_placeholder(c) for c in flat_components)
    else:
      return x.op.type == 'Placeholder'
  except AttributeError:
    return False


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def shape(x):
  # [FUNCTION DOC]
  # Returns the symbolic (dynamic) shape of a tensor or variable as a tensor. Input x: tensor or
  # variable. Supports API dispatch.
  """Returns the symbolic shape of a tensor or variable.

  Args:
      x: A tensor or variable.

  Returns:
      A symbolic shape (which is itself a tensor).

  Examples:

  >>> val = np.array([[1, 2], [3, 4]])
  >>> kvar = tf.keras.backend.variable(value=val)
  >>> tf.keras.backend.shape(kvar)
  <tf.Tensor: shape=(2,), dtype=int32, numpy=array([2, 2], dtype=int32)>
  >>> input = tf.keras.backend.placeholder(shape=(2, 4, 5))
  >>> tf.keras.backend.shape(input)
  <KerasTensor: shape=(3,) dtype=int32 inferred_value=[2, 4, 5] ...>

  """
  return array_ops.shape(x)


@doc_controls.do_not_generate_docs
def int_shape(x):
  # [FUNCTION DOC]
  # Returns the static shape of a tensor or variable as a tuple of integers or null entries. Input x:
  # tensor or variable. Uses the shape attribute, converting a tensor shape to a tuple; returns nothing
  # (null value) if the shape cannot be converted because of a value error (unknown rank).
  """Returns the shape of tensor or variable as a tuple of int or None entries.

  Args:
      x: Tensor or variable.

  Returns:
      A tuple of integers (or None entries).

  Examples:

  >>> input = tf.keras.backend.placeholder(shape=(2, 4, 5))
  >>> tf.keras.backend.int_shape(input)
  (2, 4, 5)
  >>> val = np.array([[1, 2], [3, 4]])
  >>> kvar = tf.keras.backend.variable(value=val)
  >>> tf.keras.backend.int_shape(kvar)
  (2, 2)

  """
  try:
    shape = x.shape
    if not isinstance(shape, tuple):
      shape = tuple(shape.as_list())
    return shape
  except ValueError:
    return None


@doc_controls.do_not_generate_docs
def ndim(x):
  # [FUNCTION DOC]
  # Returns the number of axes (static rank) of a tensor or variable as an integer, or nothing when the
  # rank is unknown. Input x: tensor or variable.
  """Returns the number of axes in a tensor, as an integer.

  Args:
      x: Tensor or variable.

  Returns:
      Integer (scalar), number of axes.

  Examples:


  >>> input = tf.keras.backend.placeholder(shape=(2, 4, 5))
  >>> val = np.array([[1, 2], [3, 4]])
  >>> kvar = tf.keras.backend.variable(value=val)
  >>> tf.keras.backend.ndim(input)
  3
  >>> tf.keras.backend.ndim(kvar)
  2

  """
  return x.shape.rank


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def dtype(x):
  # [FUNCTION DOC]
  # Returns the name of the base data type of a tensor or variable as a string (for example the float
  # names). Input x: tensor or variable. Supports API dispatch.
  """Returns the dtype of a Keras tensor or variable, as a string.

  Args:
      x: Tensor or variable.

  Returns:
      String, dtype of `x`.

  Examples:

  >>> tf.keras.backend.dtype(tf.keras.backend.placeholder(shape=(2,4,5)))
  'float32'
  >>> tf.keras.backend.dtype(tf.keras.backend.placeholder(shape=(2,4,5),
  ...                                                     dtype='float32'))
  'float32'
  >>> tf.keras.backend.dtype(tf.keras.backend.placeholder(shape=(2,4,5),
  ...                                                     dtype='float64'))
  'float64'
  >>> kvar = tf.keras.backend.variable(np.array([[1, 2], [3, 4]]))
  >>> tf.keras.backend.dtype(kvar)
  'float32'
  >>> kvar = tf.keras.backend.variable(np.array([[1, 2], [3, 4]]),
  ...                                  dtype='float32')
  >>> tf.keras.backend.dtype(kvar)
  'float32'

  """
  return x.dtype.base_dtype.name


@doc_controls.do_not_generate_docs
def dtype_numpy(x):
  # [FUNCTION DOC]
  # Returns the numpy data type corresponding to the dtype of a tensor or variable. Input x: tensor or
  # variable.
  """Returns the numpy dtype of a Keras tensor or variable.

  Args:
      x: Tensor or variable.

  Returns:
      numpy.dtype, dtype of `x`.
  """
  return dtypes_module.as_dtype(x.dtype).as_numpy_dtype


@doc_controls.do_not_generate_docs
def eval(x):
  # [FUNCTION DOC]
  # Evaluates a variable (or tensor) and returns its value as a numpy array. Input x: variable; sparse
  # tensors are first made dense. Delegates to the get-value helper. Side effects: may run a session in
  # graph mode.
  """Evaluates the value of a variable.

  Args:
      x: A variable.

  Returns:
      A Numpy array.

  Examples:

  >>> kvar = tf.keras.backend.variable(np.array([[1, 2], [3, 4]]),
  ...                                  dtype='float32')
  >>> tf.keras.backend.eval(kvar)
  array([[1.,  2.],
         [3.,  4.]], dtype=float32)

  """
  return get_value(to_dense(x))


@doc_controls.do_not_generate_docs
def zeros(shape, dtype=None, name=None):
  # [FUNCTION DOC]
  # Creates an all-zeros variable. Inputs: shape (tuple or list of integers, possibly symbolic), dtype
  # (defaults to floatx), name (optional). Done inside an initialization scope (lifted out of any
  # function). Returns a Keras variable when every dimension is known and non-zero (the check treats
  # any zero-sized dimension as not fully defined); otherwise returns the plain tensor.
  """Instantiates an all-zeros variable and returns it.

  Args:
      shape: Tuple or list of integers, shape of returned Keras variable
      dtype: data type of returned Keras variable
      name: name of returned Keras variable

  Returns:
      A variable (including Keras metadata), filled with `0.0`.
      Note that if `shape` was symbolic, we cannot return a variable,
      and will return a dynamically-shaped tensor instead.

  Example:

  >>> kvar = tf.keras.backend.zeros((3,4))
  >>> tf.keras.backend.eval(kvar)
  array([[0.,  0.,  0.,  0.],
         [0.,  0.,  0.,  0.],
         [0.,  0.,  0.,  0.]], dtype=float32)
  >>> A = tf.constant([1,2,3])
  >>> kvar2 = tf.keras.backend.zeros(A.shape) # [0., 0., 0.]
  >>> tf.keras.backend.eval(kvar2)
  array([0., 0., 0.], dtype=float32)
  >>> kvar3 = tf.keras.backend.zeros(A.shape,dtype=tf.int32)
  >>> tf.keras.backend.eval(kvar3)
  array([0, 0, 0], dtype=int32)
  >>> kvar4 = tf.keras.backend.zeros([2,3])
  >>> tf.keras.backend.eval(kvar4)
  array([[0., 0., 0.],
         [0., 0., 0.]], dtype=float32)

  """
  with ops.init_scope():
    if dtype is None:
      dtype = floatx()
    tf_dtype = dtypes_module.as_dtype(dtype)
    v = array_ops.zeros(shape=shape, dtype=tf_dtype, name=name)
    if py_all(v.shape.as_list()):
      return variable(v, dtype=dtype, name=name)
    return v


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def ones(shape, dtype=None, name=None):
  # [FUNCTION DOC]
  # Creates an all-ones variable. Inputs: shape, dtype (defaults to floatx), name (optional). Done in an
  # initialization scope. Returns a Keras variable if all static dimensions are known and non-zero,
  # otherwise a plain dynamically shaped tensor. Supports API dispatch.
  """Instantiates an all-ones variable and returns it.

  Args:
      shape: Tuple of integers, shape of returned Keras variable.
      dtype: String, data type of returned Keras variable.
      name: String, name of returned Keras variable.

  Returns:
      A Keras variable, filled with `1.0`.
      Note that if `shape` was symbolic, we cannot return a variable,
      and will return a dynamically-shaped tensor instead.

  Example:


  >>> kvar = tf.keras.backend.ones((3,4))
  >>> tf.keras.backend.eval(kvar)
  array([[1.,  1.,  1.,  1.],
         [1.,  1.,  1.,  1.],
         [1.,  1.,  1.,  1.]], dtype=float32)

  """
  with ops.init_scope():
    if dtype is None:
      dtype = floatx()
    tf_dtype = dtypes_module.as_dtype(dtype)
    v = array_ops.ones(shape=shape, dtype=tf_dtype, name=name)
    if py_all(v.shape.as_list()):
      return variable(v, dtype=dtype, name=name)
    return v


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def eye(size, dtype=None, name=None):
  # [FUNCTION DOC]
  # Creates an identity matrix variable. Inputs: size (number of rows and columns), dtype (defaults to
  # floatx), name (optional). Returns a Keras variable holding the identity matrix.
  """Instantiate an identity matrix and returns it.

  Args:
      size: Integer, number of rows/columns.
      dtype: String, data type of returned Keras variable.
      name: String, name of returned Keras variable.

  Returns:
      A Keras variable, an identity matrix.

  Example:


  >>> kvar = tf.keras.backend.eye(3)
  >>> tf.keras.backend.eval(kvar)
  array([[1.,  0.,  0.],
         [0.,  1.,  0.],
         [0.,  0.,  1.]], dtype=float32)


  """
  if dtype is None:
    dtype = floatx()
  tf_dtype = dtypes_module.as_dtype(dtype)
  return variable(linalg_ops.eye(size, dtype=tf_dtype), dtype, name)


@doc_controls.do_not_generate_docs
def zeros_like(x, dtype=None, name=None):
  # [FUNCTION DOC]
  # Returns a tensor of zeros with the same shape as another tensor. Inputs: x (variable or tensor),
  # dtype (optional; defaults to the dtype of x), name (optional). Note it returns a plain tensor
  # (not a variable).
  """Instantiates an all-zeros variable of the same shape as another tensor.

  Args:
      x: Keras variable or Keras tensor.
      dtype: dtype of returned Keras variable.
             `None` uses the dtype of `x`.
      name: name for the variable to create.

  Returns:
      A Keras variable with the shape of `x` filled with zeros.

  Example:

  ```python
  from tensorflow.keras import backend as K
  kvar = K.variable(np.random.random((2,3)))
  kvar_zeros = K.zeros_like(kvar)
  K.eval(kvar_zeros)
  # array([[ 0.,  0.,  0.], [ 0.,  0.,  0.]], dtype=float32)
  ```
  """
  return array_ops.zeros_like(x, dtype=dtype, name=name)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def ones_like(x, dtype=None, name=None):
  # [FUNCTION DOC]
  # Returns a tensor of ones with the same shape as another tensor. Inputs: x, dtype (optional;
  # defaults to that of x), name (optional). Supports API dispatch.
  """Instantiates an all-ones variable of the same shape as another tensor.

  Args:
      x: Keras variable or tensor.
      dtype: String, dtype of returned Keras variable.
           None uses the dtype of x.
      name: String, name for the variable to create.

  Returns:
      A Keras variable with the shape of x filled with ones.

  Example:

  >>> kvar = tf.keras.backend.variable(np.random.random((2,3)))
  >>> kvar_ones = tf.keras.backend.ones_like(kvar)
  >>> tf.keras.backend.eval(kvar_ones)
  array([[1.,  1.,  1.],
         [1.,  1.,  1.]], dtype=float32)

  """
  return array_ops.ones_like(x, dtype=dtype, name=name)


def identity(x, name=None):
  # [FUNCTION DOC]
  # Returns a tensor with the same shape, type and content as the input. Inputs: x (tensor), name
  # (optional name for the op).
  """Returns a tensor with the same content as the input tensor.

  Args:
      x: The input tensor.
      name: String, name for the variable to create.

  Returns:
      A tensor of the same shape, type and content.
  """
  return array_ops.identity(x, name=name)


@doc_controls.do_not_generate_docs
def random_uniform_variable(shape, low, high, dtype=None, name=None, seed=None):
  # [FUNCTION DOC]
  # Creates a variable filled with samples from a uniform distribution. Inputs: shape, low and high
  # (interval bounds), dtype (defaults to floatx), name (optional), seed (optional integer; when absent
  # one is drawn from the numpy random generator so results depend on numpy's seed). Returns a Keras
  # variable. Side effect: consumes numpy randomness when no seed is given.
  """Instantiates a variable with values drawn from a uniform distribution.

  Args:
      shape: Tuple of integers, shape of returned Keras variable.
      low: Float, lower boundary of the output interval.
      high: Float, upper boundary of the output interval.
      dtype: String, dtype of returned Keras variable.
      name: String, name of returned Keras variable.
      seed: Integer, random seed.

  Returns:
      A Keras variable, filled with drawn samples.

  Example:

  >>> kvar = tf.keras.backend.random_uniform_variable(shape=(2,3),
  ... low=0.0, high=1.0)
  >>> kvar
  <tf.Variable 'Variable:0' shape=(2, 3) dtype=float32, numpy=...,
  dtype=float32)>
  """
  if dtype is None:
    dtype = floatx()
  tf_dtype = dtypes_module.as_dtype(dtype)
  if seed is None:
    # ensure that randomness is conditioned by the Numpy RNG
    seed = np.random.randint(10e8)
  value = init_ops.random_uniform_initializer(
      low, high, dtype=tf_dtype, seed=seed)(shape)
  return variable(value, dtype=dtype, name=name)


@doc_controls.do_not_generate_docs
def random_normal_variable(shape, mean, scale, dtype=None, name=None,
                           seed=None):
  # [FUNCTION DOC]
  # Creates a variable filled with samples from a normal distribution. Inputs: shape, mean, scale
  # (standard deviation), dtype (defaults to floatx), name (optional), seed (optional; drawn from numpy
  # randomness when absent). Returns a Keras variable.
  """Instantiates a variable with values drawn from a normal distribution.

  Args:
      shape: Tuple of integers, shape of returned Keras variable.
      mean: Float, mean of the normal distribution.
      scale: Float, standard deviation of the normal distribution.
      dtype: String, dtype of returned Keras variable.
      name: String, name of returned Keras variable.
      seed: Integer, random seed.

  Returns:
      A Keras variable, filled with drawn samples.

  Example:

  >>> kvar = tf.keras.backend.random_normal_variable(shape=(2,3),
  ... mean=0.0, scale=1.0)
  >>> kvar
  <tf.Variable 'Variable:0' shape=(2, 3) dtype=float32, numpy=...,
  dtype=float32)>
  """
  if dtype is None:
    dtype = floatx()
  tf_dtype = dtypes_module.as_dtype(dtype)
  if seed is None:
    # ensure that randomness is conditioned by the Numpy RNG
    seed = np.random.randint(10e8)
  value = init_ops.random_normal_initializer(
      mean, scale, dtype=tf_dtype, seed=seed)(shape)
  return variable(value, dtype=dtype, name=name)


@doc_controls.do_not_generate_docs
def count_params(x):
  # [FUNCTION DOC]
  # Returns the static number of elements in a variable or tensor, computed as the product of its
  # static shape dimensions. Input x: variable or tensor with fully defined shape.
  """Returns the static number of elements in a variable or tensor.

  Args:
      x: Variable or tensor.

  Returns:
      Integer, the number of scalars in `x`.

  Example:

  >>> kvar = tf.keras.backend.zeros((2,3))
  >>> tf.keras.backend.count_params(kvar)
  6
  >>> tf.keras.backend.eval(kvar)
  array([[0.,  0.,  0.],
         [0.,  0.,  0.]], dtype=float32)

  """
  return np.prod(x.shape.as_list())


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def cast(x, dtype):
  # [FUNCTION DOC]
  # Casts a tensor or variable to another data type and returns a tensor. Inputs: x, dtype (string or
  # type). Supports API dispatch.
  """Casts a tensor to a different dtype and returns it.

  You can cast a Keras variable but it still returns a Keras tensor.

  Args:
      x: Keras tensor (or variable).
      dtype: String, either (`'float16'`, `'float32'`, or `'float64'`).

  Returns:
      Keras tensor with dtype `dtype`.

  Examples:
      Cast a float32 variable to a float64 tensor

  >>> input = tf.keras.backend.ones(shape=(1,3))
  >>> print(input)
  <tf.Variable 'Variable:0' shape=(1, 3) dtype=float32,
  numpy=array([[1., 1., 1.]], dtype=float32)>
  >>> cast_input = tf.keras.backend.cast(input, dtype='float64')
  >>> print(cast_input)
  tf.Tensor([[1. 1. 1.]], shape=(1, 3), dtype=float64)

  """
  return math_ops.cast(x, dtype)


# UPDATES OPS


@doc_controls.do_not_generate_docs
def update(x, new_x):
  # [FUNCTION DOC]
  # Assigns a new value to a variable. Inputs: x (variable) and new_x (new value). Returns the result of
  # the assign operation. Has no docstring.
  return state_ops.assign(x, new_x)


@doc_controls.do_not_generate_docs
def update_add(x, increment):
  # [FUNCTION DOC]
  # Adds an increment to a variable in place and returns the updated variable. Inputs: x (variable),
  # increment (tensor of the same shape).
  """Update the value of `x` by adding `increment`.

  Args:
      x: A Variable.
      increment: A tensor of same shape as `x`.

  Returns:
      The variable `x` updated.
  """
  return state_ops.assign_add(x, increment)


@doc_controls.do_not_generate_docs
def update_sub(x, decrement):
  # [FUNCTION DOC]
  # Subtracts a decrement from a variable in place and returns the updated variable. Inputs: x
  # (variable), decrement (tensor of the same shape).
  """Update the value of `x` by subtracting `decrement`.

  Args:
      x: A Variable.
      decrement: A tensor of same shape as `x`.

  Returns:
      The variable `x` updated.
  """
  return state_ops.assign_sub(x, decrement)


@doc_controls.do_not_generate_docs
def moving_average_update(x, value, momentum):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring describes a plain exponential moving average, which is what the TF2
  # branch does. When TF2 behavior is disabled the code calls the legacy moving average assign helper
  # with zero debiasing, which gives different numerical results from the documented formula.
  # Updates a moving average variable. Inputs: x (variable holding the average), value (new tensor with
  # the same shape), momentum (decay). In TF2 mode casts momentum and value to the dtype of x and
  # assigns the weighted combination; otherwise delegates to the legacy helper. Returns the updated
  # variable (or its update result). Side effect: modifies x.
  """Compute the exponential moving average of a value.

  The moving average 'x' is updated with 'value' following:

  ```
  x = x * momentum + value * (1 - momentum)
  ```

  For example:

  >>> x = tf.Variable(0.0)
  >>> momentum=0.9
  >>> moving_average_update(x, value = 2.0, momentum=momentum).numpy()
  >>> x.numpy()
  0.2

  The result will be biased towards the initial value of the variable.

  If the variable was initialized to zero, you can divide by
  `1 - momentum ** num_updates` to debias it (Section 3 of
  [Kingma et al., 2015](https://arxiv.org/abs/1412.6980)):

  >>> num_updates = 1.0
  >>> x_zdb = x/(1 - momentum**num_updates)
  >>> x_zdb.numpy()
  2.0

  Args:
      x: A Variable, the moving average.
      value: A tensor with the same shape as `x`, the new value to be
        averaged in.
      momentum: The moving average momentum.

  Returns:
      The updated variable.
  """
  if tf2.enabled():
    momentum = math_ops.cast(momentum, x.dtype)
    value = math_ops.cast(value, x.dtype)
    return x.assign(x * momentum + value * (1 - momentum))
  else:
    return moving_averages.assign_moving_average(
        x, value, momentum, zero_debias=True)


# LINEAR ALGEBRA


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def dot(x, y):
  # [FUNCTION DOC]
  # Multiplies two tensors, like numpy dot. Inputs: x and y (tensors or variables, x may be sparse).
  # When either has more than two axes it reshapes x to a matrix, moves the second-to-last axis of y to
  # the front and flattens the rest, multiplies, and reshapes the result to the concatenation of the
  # leading dimensions of x and the remaining dimensions of y (static sizes used where known, dynamic
  # otherwise). Otherwise uses sparse-dense multiplication for sparse x or a standard matrix multiply.
  # Returns the product tensor. Supports API dispatch.
  """Multiplies 2 tensors (and/or variables) and returns a tensor.

  This operation corresponds to `numpy.dot(a, b, out=None)`.

  Args:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A tensor, dot product of `x` and `y`.

  Examples:

  If inputs `x` and `y` are 2-D arrays, then it is equivalent to `tf.matmul`.
  >>> x = tf.keras.backend.placeholder(shape=(2, 3))
  >>> y = tf.keras.backend.placeholder(shape=(3, 4))
  >>> xy = tf.keras.backend.dot(x, y)
  >>> xy
  <KerasTensor: shape=(2, 4) dtype=float32 ...>

  >>> x = tf.keras.backend.placeholder(shape=(32, 28, 3))
  >>> y = tf.keras.backend.placeholder(shape=(3, 4))
  >>> xy = tf.keras.backend.dot(x, y)
  >>> xy
  <KerasTensor: shape=(32, 28, 4) dtype=float32 ...>

  If `x` is an N-D array and `y` is an M-D array (where M>=2), it is a sum
  product over the last axis of `x` and the second-to-last axis of `y`.
  >>> x = tf.keras.backend.random_uniform_variable(shape=(2, 3), low=0, high=1)
  >>> y = tf.keras.backend.ones((4, 3, 5))
  >>> xy = tf.keras.backend.dot(x, y)
  >>> tf.keras.backend.int_shape(xy)
  (2, 4, 5)
  """
  if ndim(x) is not None and (ndim(x) > 2 or ndim(y) > 2):
    x_shape = []
    for i, s in zip(int_shape(x), array_ops_stack.unstack(array_ops.shape(x))):
      if i is not None:
        x_shape.append(i)
      else:
        x_shape.append(s)
    x_shape = tuple(x_shape)
    y_shape = []
    for i, s in zip(int_shape(y), array_ops_stack.unstack(array_ops.shape(y))):
      if i is not None:
        y_shape.append(i)
      else:
        y_shape.append(s)
    y_shape = tuple(y_shape)
    y_permute_dim = list(range(ndim(y)))
    y_permute_dim = [y_permute_dim.pop(-2)] + y_permute_dim
    xt = array_ops.reshape(x, [-1, x_shape[-1]])
    yt = array_ops.reshape(
        array_ops.transpose(y, perm=y_permute_dim), [y_shape[-2], -1])
    return array_ops.reshape(
        math_ops.matmul(xt, yt), x_shape[:-1] + y_shape[:-2] + y_shape[-1:])
  if is_sparse(x):
    out = sparse_ops.sparse_tensor_dense_matmul(x, y)
  else:
    out = math_ops.matmul(x, y)
  return out


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def batch_dot(x, y, axes=None):
  # [FUNCTION DOC]
  # Batchwise dot product of two tensors that share a leading batch dimension.
  # Inputs: x and y (rank at least two) and axes (an integer, a pair of integers, or nothing; default
  # picks the last axis of x and, for y, the last axis when rank two else the second to last).
  # Raises value errors for rank below two, different static batch sizes, nested axes, an axis of zero
  # (including a negative one that resolves to zero) and differing static sizes along the contracted
  # axes. Rank-two inputs are expanded to rank three, the contracted axes are transposed to the last
  # axis of x and axis one of y, higher-rank inputs are squashed to rank three, a batched matrix
  # multiply is done, the result is reshaped back to the expected output shape, and the added unit
  # dimension is removed for original rank-two inputs. Returns the result tensor.
  """Batchwise dot product.

  `batch_dot` is used to compute dot product of `x` and `y` when
  `x` and `y` are data in batch, i.e. in a shape of
  `(batch_size, :)`.
  `batch_dot` results in a tensor or variable with less dimensions
  than the input. If the number of dimensions is reduced to 1,
  we use `expand_dims` to make sure that ndim is at least 2.

  Args:
    x: Keras tensor or variable with `ndim >= 2`.
    y: Keras tensor or variable with `ndim >= 2`.
    axes: Tuple or list of integers with target dimensions, or single integer.
      The sizes of `x.shape[axes[0]]` and `y.shape[axes[1]]` should be equal.

  Returns:
    A tensor with shape equal to the concatenation of `x`'s shape
    (less the dimension that was summed over) and `y`'s shape
    (less the batch dimension and the dimension that was summed over).
    If the final rank is 1, we reshape it to `(batch_size, 1)`.

  Examples:

  >>> x_batch = tf.keras.backend.ones(shape=(32, 20, 1))
  >>> y_batch = tf.keras.backend.ones(shape=(32, 30, 20))
  >>> xy_batch_dot = tf.keras.backend.batch_dot(x_batch, y_batch, axes=(1, 2))
  >>> tf.keras.backend.int_shape(xy_batch_dot)
  (32, 1, 30)

  Shape inference:
    Let `x`'s shape be `(100, 20)` and `y`'s shape be `(100, 30, 20)`.
    If `axes` is (1, 2), to find the output shape of resultant tensor,
        loop through each dimension in `x`'s shape and `y`'s shape:
    * `x.shape[0]` : 100 : append to output shape
    * `x.shape[1]` : 20 : do not append to output shape,
        dimension 1 of `x` has been summed over. (`dot_axes[0]` = 1)
    * `y.shape[0]` : 100 : do not append to output shape,
        always ignore first dimension of `y`
    * `y.shape[1]` : 30 : append to output shape
    * `y.shape[2]` : 20 : do not append to output shape,
        dimension 2 of `y` has been summed over. (`dot_axes[1]` = 2)
    `output_shape` = `(100, 30)`
  """
  x_shape = int_shape(x)
  y_shape = int_shape(y)

  x_ndim = len(x_shape)
  y_ndim = len(y_shape)

  if x_ndim < 2 or y_ndim < 2:
    raise ValueError('Cannot do batch_dot on inputs '
                     'with rank < 2. '
                     'Received inputs with shapes ' +
                     str(x_shape) + ' and ' +
                     str(y_shape) + '.')

  x_batch_size = x_shape[0]
  y_batch_size = y_shape[0]

  if x_batch_size is not None and y_batch_size is not None:
    if x_batch_size != y_batch_size:
      raise ValueError('Cannot do batch_dot on inputs '
                       'with different batch sizes. '
                       'Received inputs with shapes ' +
                       str(x_shape) + ' and ' +
                       str(y_shape) + '.')
  if isinstance(axes, int):
    axes = [axes, axes]

  if axes is None:
    if y_ndim == 2:
      axes = [x_ndim - 1, y_ndim - 1]
    else:
      axes = [x_ndim - 1, y_ndim - 2]

  if py_any(isinstance(a, (list, tuple)) for a in axes):
    raise ValueError('Multiple target dimensions are not supported. ' +
                     'Expected: None, int, (int, int), ' +
                     'Provided: ' + str(axes))

  # if tuple, convert to list.
  axes = list(axes)

  # convert negative indices.
  if axes[0] < 0:
    axes[0] += x_ndim
  if axes[1] < 0:
    axes[1] += y_ndim

  # sanity checks
  if 0 in axes:
    raise ValueError('Cannot perform batch_dot over axis 0. '
                     'If your inputs are not batched, '
                     'add a dummy batch dimension to your '
                     'inputs using K.expand_dims(x, 0)')
  a0, a1 = axes
  d1 = x_shape[a0]
  d2 = y_shape[a1]

  if d1 is not None and d2 is not None and d1 != d2:
    raise ValueError('Cannot do batch_dot on inputs with shapes ' +
                     str(x_shape) + ' and ' + str(y_shape) +
                     ' with axes=' + str(axes) + '. x.shape[%d] != '
                     'y.shape[%d] (%d != %d).' % (axes[0], axes[1], d1, d2))

  # backup ndims. Need them later.
  orig_x_ndim = x_ndim
  orig_y_ndim = y_ndim

  # if rank is 2, expand to 3.
  if x_ndim == 2:
    x = array_ops.expand_dims(x, 1)
    a0 += 1
    x_ndim += 1
  if y_ndim == 2:
    y = array_ops.expand_dims(y, 2)
    y_ndim += 1

  # bring x's dimension to be reduced to last axis.
  if a0 != x_ndim - 1:
    pattern = list(range(x_ndim))
    for i in range(a0, x_ndim - 1):
      pattern[i] = pattern[i + 1]
    pattern[-1] = a0
    x = array_ops.transpose(x, pattern)

  # bring y's dimension to be reduced to axis 1.
  if a1 != 1:
    pattern = list(range(y_ndim))
    for i in range(a1, 1, -1):
      pattern[i] = pattern[i - 1]
    pattern[1] = a1
    y = array_ops.transpose(y, pattern)

  # normalize both inputs to rank 3.
  if x_ndim > 3:
    # squash middle dimensions of x.
    x_shape = shape(x)
    x_mid_dims = x_shape[1:-1]
    x_squashed_shape = array_ops_stack.stack(
        [x_shape[0], -1, x_shape[-1]])
    x = array_ops.reshape(x, x_squashed_shape)
    x_squashed = True
  else:
    x_squashed = False

  if y_ndim > 3:
    # squash trailing dimensions of y.
    y_shape = shape(y)
    y_trail_dims = y_shape[2:]
    y_squashed_shape = array_ops_stack.stack(
        [y_shape[0], y_shape[1], -1])
    y = array_ops.reshape(y, y_squashed_shape)
    y_squashed = True
  else:
    y_squashed = False

  result = math_ops.matmul(x, y)

  # if inputs were squashed, we have to reshape the matmul output.
  output_shape = array_ops.shape(result)
  do_reshape = False

  if x_squashed:
    output_shape = array_ops.concat(
        [output_shape[:1],
         x_mid_dims,
         output_shape[-1:]], 0)
    do_reshape = True

  if y_squashed:
    output_shape = array_ops.concat([output_shape[:-1], y_trail_dims], 0)
    do_reshape = True

  if do_reshape:
    result = array_ops.reshape(result, output_shape)

  # if the inputs were originally rank 2, we remove the added 1 dim.
  if orig_x_ndim == 2:
    result = array_ops.squeeze(result, 1)
  elif orig_y_ndim == 2:
    result = array_ops.squeeze(result, -1)

  return result


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def transpose(x):
  # [FUNCTION DOC]
  # Transposes a tensor (reverses its axes) and returns it. Input x: tensor or variable.
  """Transposes a tensor and returns it.

  Args:
      x: Tensor or variable.

  Returns:
      A tensor.

  Examples:

  >>> var = tf.keras.backend.variable([[1, 2, 3], [4, 5, 6]])
  >>> tf.keras.backend.eval(var)
  array([[1.,  2.,  3.],
         [4.,  5.,  6.]], dtype=float32)
  >>> var_transposed = tf.keras.backend.transpose(var)
  >>> tf.keras.backend.eval(var_transposed)
  array([[1.,  4.],
         [2.,  5.],
         [3.,  6.]], dtype=float32)
  >>> input = tf.keras.backend.placeholder((2, 3))
  >>> input
  <KerasTensor: shape=(2, 3) dtype=float32 ...>
  >>> input_transposed = tf.keras.backend.transpose(input)
  >>> input_transposed
  <KerasTensor: shape=(3, 2) dtype=float32 ...>
  """
  return array_ops.transpose(x)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def gather(reference, indices):
  # [FUNCTION DOC]
  # Retrieves slices of a tensor along its first axis. Inputs: reference (tensor) and indices (integer
  # tensor or list). Returns a tensor of the same type as reference.
  """Retrieves the elements of indices `indices` in the tensor `reference`.

  Args:
      reference: A tensor.
      indices: An integer tensor of indices.

  Returns:
      A tensor of same type as `reference`.

  Examples:

  >>> var = tf.keras.backend.variable([[1, 2, 3], [4, 5, 6]])
  >>> tf.keras.backend.eval(var)
  array([[1., 2., 3.],
         [4., 5., 6.]], dtype=float32)
  >>> var_gathered = tf.keras.backend.gather(var, [0])
  >>> tf.keras.backend.eval(var_gathered)
  array([[1., 2., 3.]], dtype=float32)
  >>> var_gathered = tf.keras.backend.gather(var, [1])
  >>> tf.keras.backend.eval(var_gathered)
  array([[4., 5., 6.]], dtype=float32)
  >>> var_gathered = tf.keras.backend.gather(var, [0,1,0])
  >>> tf.keras.backend.eval(var_gathered)
  array([[1., 2., 3.],
         [4., 5., 6.],
         [1., 2., 3.]], dtype=float32)
  """
  return array_ops.gather(reference, indices)


# ELEMENT-WISE OPERATIONS


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def max(x, axis=None, keepdims=False):
  # [FUNCTION DOC]
  # Maximum value of a tensor along the given axis. Inputs: x, axis (integer or nothing for all axes),
  # keepdims (whether reduced axes stay with length one). Returns the reduced tensor.
  """Maximum value in a tensor.

  Args:
      x: A tensor or variable.
      axis: An integer, the axis to find maximum values.
      keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced
          by 1. If `keepdims` is `True`,
          the reduced dimension is retained with length 1.

  Returns:
      A tensor with maximum values of `x`.
  """
  return math_ops.reduce_max(x, axis, keepdims)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def min(x, axis=None, keepdims=False):
  # [FUNCTION DOC]
  # Minimum value of a tensor along the given axis. Inputs: x, axis (integer or nothing for all axes),
  # keepdims (keep reduced axes with length one). Returns the reduced tensor.
  """Minimum value in a tensor.

  Args:
      x: A tensor or variable.
      axis: An integer, the axis to find minimum values.
      keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced
          by 1. If `keepdims` is `True`,
          the reduced dimension is retained with length 1.

  Returns:
      A tensor with minimum values of `x`.
  """
  return math_ops.reduce_min(x, axis, keepdims)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def sum(x, axis=None, keepdims=False):
  # [FUNCTION DOC]
  # Sum of the values in a tensor along the given axis. Inputs: x, axis (integer or nothing for all
  # axes), keepdims (keep reduced axes with length one). Returns the reduced tensor.
  """Sum of the values in a tensor, alongside the specified axis.

  Args:
      x: A tensor or variable.
      axis: An integer, the axis to sum over.
      keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced
          by 1. If `keepdims` is `True`,
          the reduced dimension is retained with length 1.

  Returns:
      A tensor with sum of `x`.
  """
  return math_ops.reduce_sum(x, axis, keepdims)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def prod(x, axis=None, keepdims=False):
  # [FUNCTION DOC]
  # Product of the values in a tensor along the given axis. Inputs: x, axis (integer or nothing for all
  # axes), keepdims. Returns the reduced tensor.
  """Multiplies the values in a tensor, alongside the specified axis.

  Args:
      x: A tensor or variable.
      axis: An integer, the axis to compute the product.
      keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced
          by 1. If `keepdims` is `True`,
          the reduced dimension is retained with length 1.

  Returns:
      A tensor with the product of elements of `x`.
  """
  return math_ops.reduce_prod(x, axis, keepdims)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def cumsum(x, axis=0):
  # [FUNCTION DOC]
  # Cumulative sum along an axis. Inputs: x (tensor) and axis (integer, default zero). Returns a tensor
  # of the same shape.
  """Cumulative sum of the values in a tensor, alongside the specified axis.

  Args:
      x: A tensor or variable.
      axis: An integer, the axis to compute the sum.

  Returns:
      A tensor of the cumulative sum of values of `x` along `axis`.
  """
  return math_ops.cumsum(x, axis=axis)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def cumprod(x, axis=0):
  # [FUNCTION DOC]
  # Cumulative product along an axis. Inputs: x (tensor) and axis (integer, default zero). Returns a
  # tensor of the same shape.
  """Cumulative product of the values in a tensor, alongside the specified axis.

  Args:
      x: A tensor or variable.
      axis: An integer, the axis to compute the product.

  Returns:
      A tensor of the cumulative product of values of `x` along `axis`.
  """
  return math_ops.cumprod(x, axis=axis)


@doc_controls.do_not_generate_docs
def var(x, axis=None, keepdims=False):
  # [FUNCTION DOC]
  # Variance of a tensor along the given axis. Inputs: x (boolean tensors are first cast to floatx),
  # axis (integer or nothing for all), keepdims. Returns the variance tensor.
  """Variance of a tensor, alongside the specified axis.

  Args:
      x: A tensor or variable.
      axis: An integer, the axis to compute the variance.
      keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced
          by 1. If `keepdims` is `True`,
          the reduced dimension is retained with length 1.

  Returns:
      A tensor with the variance of elements of `x`.
  """
  if x.dtype.base_dtype == dtypes_module.bool:
    x = math_ops.cast(x, floatx())
  return math_ops.reduce_variance(x, axis=axis, keepdims=keepdims)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def std(x, axis=None, keepdims=False):
  # [FUNCTION DOC]
  # Standard deviation of a tensor along the given axis. Inputs: x (boolean input is cast to floatx),
  # axis (nothing reduces all axes), keepdims. Returns the standard deviation tensor with the dtype of
  # the (possibly cast) input.
  """Standard deviation of a tensor, alongside the specified axis.

  It is an alias to `tf.math.reduce_std`.

  Args:
      x: A tensor or variable. It should have numerical dtypes. Boolean type
        inputs will be converted to float.
      axis: An integer, the axis to compute the standard deviation. If `None`
        (the default), reduces all dimensions. Must be in the range
        `[-rank(x), rank(x))`.
      keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced
          by 1. If `keepdims` is `True`, the reduced dimension is retained with
          length 1.

  Returns:
      A tensor with the standard deviation of elements of `x` with same dtype.
      Boolean type input will be converted to float.
  """
  if x.dtype.base_dtype == dtypes_module.bool:
    x = math_ops.cast(x, floatx())
  return math_ops.reduce_std(x, axis=axis, keepdims=keepdims)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def mean(x, axis=None, keepdims=False):
  # [FUNCTION DOC]
  # Mean of a tensor along the given axis. Inputs: x (boolean input is cast to floatx), axis (integer,
  # list of integers or nothing), keepdims. Returns the mean tensor.
  """Mean of a tensor, alongside the specified axis.

  Args:
      x: A tensor or variable.
      axis: A list of integer. Axes to compute the mean.
      keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced
          by 1 for each entry in `axis`. If `keepdims` is `True`,
          the reduced dimensions are retained with length 1.

  Returns:
      A tensor with the mean of elements of `x`.
  """
  if x.dtype.base_dtype == dtypes_module.bool:
    x = math_ops.cast(x, floatx())
  return math_ops.reduce_mean(x, axis, keepdims)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def any(x, axis=None, keepdims=False):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says the result is an unsigned 8-bit tensor of zeros and ones, but the
  # code returns a boolean tensor.
  # Logical OR reduction. Inputs: x (cast to boolean), axis, keepdims. Returns the reduced boolean
  # tensor.
  """Bitwise reduction (logical OR).

  Args:
      x: Tensor or variable.
      axis: axis along which to perform the reduction.
      keepdims: whether the drop or broadcast the reduction axes.

  Returns:
      A uint8 tensor (0s and 1s).
  """
  x = math_ops.cast(x, dtypes_module.bool)
  return math_ops.reduce_any(x, axis, keepdims)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def all(x, axis=None, keepdims=False):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says the result is an unsigned 8-bit tensor of zeros and ones, but the
  # code returns a boolean tensor.
  # Logical AND reduction. Inputs: x (cast to boolean), axis, keepdims. Returns the reduced boolean
  # tensor.
  """Bitwise reduction (logical AND).

  Args:
      x: Tensor or variable.
      axis: axis along which to perform the reduction.
      keepdims: whether the drop or broadcast the reduction axes.

  Returns:
      A uint8 tensor (0s and 1s).
  """
  x = math_ops.cast(x, dtypes_module.bool)
  return math_ops.reduce_all(x, axis, keepdims)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def argmax(x, axis=-1):
  # [FUNCTION DOC]
  # Index of the maximum value along an axis. Inputs: x, axis (default last axis). Returns an integer
  # tensor.
  """Returns the index of the maximum value along an axis.

  Args:
      x: Tensor or variable.
      axis: axis along which to perform the reduction.

  Returns:
      A tensor.
  """
  return math_ops.argmax(x, axis)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def argmin(x, axis=-1):
  # [FUNCTION DOC]
  # Index of the minimum value along an axis. Inputs: x, axis (default last axis). Returns an integer
  # tensor.
  """Returns the index of the minimum value along an axis.

  Args:
      x: Tensor or variable.
      axis: axis along which to perform the reduction.

  Returns:
      A tensor.
  """
  return math_ops.argmin(x, axis)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def square(x):
  # [FUNCTION DOC]
  # Element-wise square of a tensor. Input x. Returns a tensor.
  """Element-wise square.

  Args:
      x: Tensor or variable.

  Returns:
      A tensor.
  """
  return math_ops.square(x)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def abs(x):
  # [FUNCTION DOC]
  # Element-wise absolute value of a tensor. Input x. Returns a tensor.
  """Element-wise absolute value.

  Args:
      x: Tensor or variable.

  Returns:
      A tensor.
  """
  return math_ops.abs(x)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def sqrt(x):
  # [FUNCTION DOC]
  # Element-wise square root with negative values first clipped to zero (so the result is never not-a-
  # number for negative input). Input x. Returns a tensor.
  """Element-wise square root.

     This function clips negative tensor values to 0 before computing the
     square root.

  Args:
      x: Tensor or variable.

  Returns:
      A tensor.
  """
  zero = _constant_to_tensor(0., x.dtype.base_dtype)
  x = math_ops.maximum(x, zero)
  return math_ops.sqrt(x)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def exp(x):
  # [FUNCTION DOC]
  # Element-wise exponential. Input x. Returns a tensor.
  """Element-wise exponential.

  Args:
      x: Tensor or variable.

  Returns:
      A tensor.
  """
  return math_ops.exp(x)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def log(x):
  # [FUNCTION DOC]
  # Element-wise natural logarithm. Input x. Returns a tensor.
  """Element-wise log.

  Args:
      x: Tensor or variable.

  Returns:
      A tensor.
  """
  return math_ops.log(x)


def logsumexp(x, axis=None, keepdims=False):
  # [FUNCTION DOC]
  # Numerically stable logarithm of the sum of exponentials along an axis. Inputs: x, axis (integer or
  # nothing), keepdims. Returns the reduced tensor. (Not wrapped with API dispatch, unlike its
  # neighbors.)
  """Computes log(sum(exp(elements across dimensions of a tensor))).

  This function is more numerically stable than log(sum(exp(x))).
  It avoids overflows caused by taking the exp of large inputs and
  underflows caused by taking the log of small inputs.

  Args:
      x: A tensor or variable.
      axis: An integer, the axis to reduce over.
      keepdims: A boolean, whether to keep the dimensions or not.
          If `keepdims` is `False`, the rank of the tensor is reduced
          by 1. If `keepdims` is `True`, the reduced dimension is
          retained with length 1.

  Returns:
      The reduced tensor.
  """
  return math_ops.reduce_logsumexp(x, axis, keepdims)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def round(x):
  # [FUNCTION DOC]
  # Element-wise rounding to the nearest integer, ties to even. Input x. Returns a tensor.
  """Element-wise rounding to the closest integer.

  In case of tie, the rounding mode used is "half to even".

  Args:
      x: Tensor or variable.

  Returns:
      A tensor.
  """
  return math_ops.round(x)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def sign(x):
  # [FUNCTION DOC]
  # Element-wise sign (negative one, zero, one). Input x. Returns a tensor.
  """Element-wise sign.

  Args:
      x: Tensor or variable.

  Returns:
      A tensor.
  """
  return math_ops.sign(x)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def pow(x, a):
  # [FUNCTION DOC]
  # Element-wise exponentiation. Inputs: x (tensor) and a (exponent, an integer or tensor). Returns a
  # tensor.
  """Element-wise exponentiation.

  Args:
      x: Tensor or variable.
      a: Python integer.

  Returns:
      A tensor.
  """
  return math_ops.pow(x, a)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def clip(x, min_value, max_value):
  # [FUNCTION DOC]
  # Element-wise clipping of values to a range. Inputs: x, min_value and max_value (numbers or tensors;
  # nothing means unbounded on that side). If both bounds are plain numbers and the maximum is below
  # the minimum, the maximum is raised to the minimum. Returns the clipped tensor.
  """Element-wise value clipping.

  Args:
      x: Tensor or variable.
      min_value: Python float, integer, or tensor.
      max_value: Python float, integer, or tensor.

  Returns:
      A tensor.
  """
  if (isinstance(min_value, (int, float)) and
      isinstance(max_value, (int, float))):
    if max_value < min_value:
      max_value = min_value
  if min_value is None:
    min_value = -np.inf
  if max_value is None:
    max_value = np.inf
  return clip_ops.clip_by_value(x, min_value, max_value)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def equal(x, y):
  # [FUNCTION DOC]
  # Element-wise equality of two tensors. Inputs x and y. Returns a boolean tensor.
  """Element-wise equality between two tensors.

  Args:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A bool tensor.
  """
  return math_ops.equal(x, y)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def not_equal(x, y):
  # [FUNCTION DOC]
  # Element-wise inequality of two tensors. Inputs x and y. Returns a boolean tensor.
  """Element-wise inequality between two tensors.

  Args:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A bool tensor.
  """
  return math_ops.not_equal(x, y)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def greater(x, y):
  # [FUNCTION DOC]
  # Element-wise greater-than comparison. Inputs x and y. Returns a boolean tensor.
  """Element-wise truth value of (x > y).

  Args:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A bool tensor.
  """
  return math_ops.greater(x, y)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def greater_equal(x, y):
  # [FUNCTION DOC]
  # Element-wise greater-or-equal comparison. Inputs x and y. Returns a boolean tensor.
  """Element-wise truth value of (x >= y).

  Args:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A bool tensor.
  """
  return math_ops.greater_equal(x, y)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def less(x, y):
  # [FUNCTION DOC]
  # Element-wise less-than comparison. Inputs x and y. Returns a boolean tensor.
  """Element-wise truth value of (x < y).

  Args:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A bool tensor.
  """
  return math_ops.less(x, y)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def less_equal(x, y):
  # [FUNCTION DOC]
  # Element-wise less-or-equal comparison. Inputs x and y. Returns a boolean tensor.
  """Element-wise truth value of (x <= y).

  Args:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A bool tensor.
  """
  return math_ops.less_equal(x, y)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def maximum(x, y):
  # [FUNCTION DOC]
  # Element-wise maximum of two tensors. Inputs x and y. Returns a tensor.
  """Element-wise maximum of two tensors.

  Args:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A tensor with the element wise maximum value(s) of `x` and `y`.

  Examples:

  >>> x = tf.Variable([[1, 2], [3, 4]])
  >>> y = tf.Variable([[2, 1], [0, -1]])
  >>> m = tf.keras.backend.maximum(x, y)
  >>> m
  <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
  array([[2, 2],
         [3, 4]], dtype=int32)>
  """
  return math_ops.maximum(x, y)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def minimum(x, y):
  # [FUNCTION DOC]
  # Element-wise minimum of two tensors. Inputs x and y. Returns a tensor.
  """Element-wise minimum of two tensors.

  Args:
      x: Tensor or variable.
      y: Tensor or variable.

  Returns:
      A tensor.
  """
  return math_ops.minimum(x, y)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def sin(x):
  # [FUNCTION DOC]
  # Element-wise sine. Input x. Returns a tensor.
  """Computes sin of x element-wise.

  Args:
      x: Tensor or variable.

  Returns:
      A tensor.
  """
  return math_ops.sin(x)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def cos(x):
  # [FUNCTION DOC]
  # Element-wise cosine. Input x. Returns a tensor.
  """Computes cos of x element-wise.

  Args:
      x: Tensor or variable.

  Returns:
      A tensor.
  """
  return math_ops.cos(x)


def _regular_normalize_batch_in_training(x,
                                         gamma,
                                         beta,
                                         reduction_axes,
                                         epsilon=1e-3):
  # [FUNCTION DOC]
  # HELPER: non-fused branch of normalize_batch_in_training, used when all axes but the last are reduced.
  # Inputs: x, gamma (scale), beta (offset), reduction_axes (iterable of axes), epsilon. Computes moments
  # over the reduction axes and applies standard batch normalization. Returns a tuple of the normalized
  # tensor, mean and variance.
  """Non-fused version of `normalize_batch_in_training`.

  Args:
      x: Input tensor or variable.
      gamma: Tensor by which to scale the input.
      beta: Tensor with which to center the input.
      reduction_axes: iterable of integers,
          axes over which to normalize.
      epsilon: Fuzz factor.

  Returns:
      A tuple length of 3, `(normalized_tensor, mean, variance)`.
  """
  mean, var = nn.moments(x, reduction_axes, None, None, False)
  normed = nn.batch_normalization(x, mean, var, beta, gamma, epsilon)
  return normed, mean, var


def _broadcast_normalize_batch_in_training(x,
                                           gamma,
                                           beta,
                                           reduction_axes,
                                           epsilon=1e-3):
  # [FUNCTION DOC]
  # HELPER: non-fused branch of normalize_batch_in_training for arbitrary reduction axes.
  # Inputs as for the regular variant. Computes moments, reshapes mean, variance and (when given) gamma
  # and beta to a shape with one in every reduced axis and the dynamic size elsewhere so they
  # broadcast, then applies batch normalization. Returns a tuple of the normalized tensor, mean and
  # variance.
  """Non-fused, broadcast version of `normalize_batch_in_training`.

  Args:
      x: Input tensor or variable.
      gamma: Tensor by which to scale the input.
      beta: Tensor with which to center the input.
      reduction_axes: iterable of integers,
          axes over which to normalize.
      epsilon: Fuzz factor.

  Returns:
      A tuple length of 3, `(normalized_tensor, mean, variance)`.
  """
  mean, var = nn.moments(x, reduction_axes, None, None, False)
  target_shape = []
  for axis in range(ndim(x)):
    if axis in reduction_axes:
      target_shape.append(1)
    else:
      target_shape.append(array_ops.shape(x)[axis])
  target_shape = array_ops_stack.stack(target_shape)

  broadcast_mean = array_ops.reshape(mean, target_shape)
  broadcast_var = array_ops.reshape(var, target_shape)
  if gamma is None:
    broadcast_gamma = None
  else:
    broadcast_gamma = array_ops.reshape(gamma, target_shape)
  if beta is None:
    broadcast_beta = None
  else:
    broadcast_beta = array_ops.reshape(beta, target_shape)

  normed = nn.batch_normalization(x, broadcast_mean, broadcast_var,
                                  broadcast_beta, broadcast_gamma, epsilon)
  return normed, mean, var


def _fused_normalize_batch_in_training(x,
                                       gamma,
                                       beta,
                                       reduction_axes,
                                       epsilon=1e-3):
  # [FUNCTION DOC]
  # HELPER: fused branch of normalize_batch_in_training for four-dimensional inputs.
  # Inputs: x, gamma, beta, reduction_axes, epsilon. Reduction axes zero-one-two mean channels-last
  # (normalization axis three), otherwise channels-first (axis one). Missing gamma or beta are replaced
  # by constants of ones or zeros sized to the channel dimension. Returns whatever the fused batch norm
  # op returns (normalized tensor, mean, variance).
  """Fused version of `normalize_batch_in_training`.

  Args:
      x: Input tensor or variable.
      gamma: Tensor by which to scale the input.
      beta: Tensor with which to center the input.
      reduction_axes: iterable of integers,
          axes over which to normalize.
      epsilon: Fuzz factor.

  Returns:
      A tuple length of 3, `(normalized_tensor, mean, variance)`.
  """
  if list(reduction_axes) == [0, 1, 2]:
    normalization_axis = 3
    tf_data_format = 'NHWC'
  else:
    normalization_axis = 1
    tf_data_format = 'NCHW'

  if gamma is None:
    gamma = constant_op.constant(
        1.0, dtype=x.dtype, shape=[x.shape[normalization_axis]])
  if beta is None:
    beta = constant_op.constant(
        0.0, dtype=x.dtype, shape=[x.shape[normalization_axis]])

  return nn.fused_batch_norm(
      x, gamma, beta, epsilon=epsilon, data_format=tf_data_format)


@doc_controls.do_not_generate_docs
def normalize_batch_in_training(x, gamma, beta, reduction_axes, epsilon=1e-3):
  # [FUNCTION DOC]
  # Computes batch mean and variance and normalizes the batch with them.
  # Inputs: x, gamma (scale), beta (offset), reduction_axes (iterable of axes), epsilon. Chooses the
  # implementation: for rank-four input with reduction axes zero-one-two or zero-two-three it uses the
  # fused version (but the broadcast version when channels-first reduction is requested and NCHW is not
  # supported); otherwise uses the regular version when the reduced axes are all axes but the last, else
  # the broadcast version. Returns a tuple of normalized tensor, mean and variance.
  """Computes mean and std for batch then apply batch_normalization on batch.

  Args:
      x: Input tensor or variable.
      gamma: Tensor by which to scale the input.
      beta: Tensor with which to center the input.
      reduction_axes: iterable of integers,
          axes over which to normalize.
      epsilon: Fuzz factor.

  Returns:
      A tuple length of 3, `(normalized_tensor, mean, variance)`.
  """
  if ndim(x) == 4 and list(reduction_axes) in [[0, 1, 2], [0, 2, 3]]:
    if not _has_nchw_support() and list(reduction_axes) == [0, 2, 3]:
      return _broadcast_normalize_batch_in_training(
          x, gamma, beta, reduction_axes, epsilon=epsilon)
    return _fused_normalize_batch_in_training(
        x, gamma, beta, reduction_axes, epsilon=epsilon)
  else:
    if sorted(reduction_axes) == list(range(ndim(x)))[:-1]:
      return _regular_normalize_batch_in_training(
          x, gamma, beta, reduction_axes, epsilon=epsilon)
    else:
      return _broadcast_normalize_batch_in_training(
          x, gamma, beta, reduction_axes, epsilon=epsilon)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def batch_normalization(x, mean, var, beta, gamma, axis=-1, epsilon=1e-3):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring formula divides by the square root of the variance plus epsilon, but the
  # underlying ops add epsilon inside the square root (variance plus epsilon, then root). Also, for
  # rank-four input whose axis is not one of the recognized channel axes the data format becomes
  # nothing yet the fused op is still called with it, and the reshaping of mean, variance, beta and
  # gamma happens only for NHWC or supported NCHW.
  # Applies batch normalization using given statistics. Inputs: x, mean, var, beta (offset or nothing),
  # gamma (scale or nothing), axis (channel axis, default last), epsilon. For rank-four input it uses the
  # fused op in inference mode with the matching data format, flattening broadcast parameters and
  # filling missing beta/gamma with zeros/ones; otherwise uses the generic normalization op. Returns the
  # normalized tensor.
  """Applies batch normalization on x given mean, var, beta and gamma.

  I.e. returns:
  `output = (x - mean) / (sqrt(var) + epsilon) * gamma + beta`

  Args:
      x: Input tensor or variable.
      mean: Mean of batch.
      var: Variance of batch.
      beta: Tensor with which to center the input.
      gamma: Tensor by which to scale the input.
      axis: Integer, the axis that should be normalized.
          (typically the features axis).
      epsilon: Fuzz factor.

  Returns:
      A tensor.
  """
  if ndim(x) == 4:
    # The CPU implementation of `fused_batch_norm` only supports NHWC
    if axis == 1 or axis == -3:
      tf_data_format = 'NCHW'
    elif axis == 3 or axis == -1:
      tf_data_format = 'NHWC'
    else:
      tf_data_format = None

    if (tf_data_format == 'NHWC' or
        tf_data_format == 'NCHW' and _has_nchw_support()):
      # The mean / var / beta / gamma tensors may be broadcasted
      # so they may have extra axes of size 1, which should be squeezed.
      if ndim(mean) > 1:
        mean = array_ops.reshape(mean, [-1])
      if ndim(var) > 1:
        var = array_ops.reshape(var, [-1])
      if beta is None:
        beta = zeros_like(mean)
      elif ndim(beta) > 1:
        beta = array_ops.reshape(beta, [-1])
      if gamma is None:
        gamma = ones_like(mean)
      elif ndim(gamma) > 1:
        gamma = array_ops.reshape(gamma, [-1])
    y, _, _ = nn.fused_batch_norm(
        x,
        gamma,
        beta,
        epsilon=epsilon,
        mean=mean,
        variance=var,
        data_format=tf_data_format,
        is_training=False
    )
    return y
  return nn.batch_normalization(x, mean, var, beta, gamma, epsilon)


# SHAPE OPERATIONS


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def concatenate(tensors, axis=-1):
  # [FUNCTION DOC]
  # Concatenates a list of tensors along an axis. Inputs: tensors (list) and axis (default last; a
  # negative axis is wrapped using the rank of the first tensor, or becomes zero if the rank is
  # zero/unknown). Uses sparse concatenation if all inputs are sparse, plain concatenation if all are
  # ragged, otherwise converts everything to dense first. Returns the combined tensor.
  """Concatenates a list of tensors alongside the specified axis.

  Args:
      tensors: list of tensors to concatenate.
      axis: concatenation axis.

  Returns:
      A tensor.

  Example:

      >>> a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
      >>> b = tf.constant([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
      >>> tf.keras.backend.concatenate((a, b), axis=-1)
      <tf.Tensor: shape=(3, 6), dtype=int32, numpy=
      array([[ 1,  2,  3, 10, 20, 30],
             [ 4,  5,  6, 40, 50, 60],
             [ 7,  8,  9, 70, 80, 90]], dtype=int32)>

  """
  if axis < 0:
    rank = ndim(tensors[0])
    if rank:
      axis %= rank
    else:
      axis = 0

  if py_all(is_sparse(x) for x in tensors):
    return sparse_ops.sparse_concat(axis, tensors)
  elif py_all(isinstance(x, ragged_tensor.RaggedTensor) for x in tensors):
    return array_ops.concat(tensors, axis)
  else:
    return array_ops.concat([to_dense(x) for x in tensors], axis)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def reshape(x, shape):
  # [FUNCTION DOC]
  # Reshapes a tensor. Inputs: x and shape (target shape tuple). Returns the reshaped tensor.
  """Reshapes a tensor to the specified shape.

  Args:
      x: Tensor or variable.
      shape: Target shape tuple.

  Returns:
      A tensor.

  Example:

    >>> a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    >>> a
    <tf.Tensor: shape=(4, 3), dtype=int32, numpy=
    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 7,  8,  9],
           [10, 11, 12]], dtype=int32)>
    >>> tf.keras.backend.reshape(a, shape=(2, 6))
    <tf.Tensor: shape=(2, 6), dtype=int32, numpy=
    array([[ 1,  2,  3,  4,  5,  6],
           [ 7,  8,  9, 10, 11, 12]], dtype=int32)>

  """
  return array_ops.reshape(x, shape)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def permute_dimensions(x, pattern):
  # [FUNCTION DOC]
  # Permutes the axes of a tensor. Inputs: x and pattern (tuple of axis indices). Returns the transposed
  # tensor.
  """Permutes axes in a tensor.

  Args:
      x: Tensor or variable.
      pattern: A tuple of
          dimension indices, e.g. `(0, 2, 1)`.

  Returns:
      A tensor.

  Example:

    >>> a = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    >>> a
    <tf.Tensor: shape=(4, 3), dtype=int32, numpy=
    array([[ 1,  2,  3],
           [ 4,  5,  6],
           [ 7,  8,  9],
           [10, 11, 12]], dtype=int32)>
    >>> tf.keras.backend.permute_dimensions(a, pattern=(1, 0))
    <tf.Tensor: shape=(3, 4), dtype=int32, numpy=
    array([[ 1,  4,  7, 10],
           [ 2,  5,  8, 11],
           [ 3,  6,  9, 12]], dtype=int32)>

  """
  return array_ops.transpose(x, perm=pattern)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def resize_images(x, height_factor, width_factor, data_format,
                  interpolation='nearest'):
  # [FUNCTION DOC]
  # Resizes the spatial dimensions of a four-dimensional image batch by integer factors.
  # Inputs: x, height_factor, width_factor (positive integers), data_format (channels_first or
  # channels_last; otherwise a value error), interpolation (nearest or bilinear; otherwise a value
  # error). Computes the new size from the static or dynamic shape, converts channels-first input to
  # channels-last, resizes, and converts back. Returns the resized tensor.
  """Resizes the images contained in a 4D tensor.

  Args:
      x: Tensor or variable to resize.
      height_factor: Positive integer.
      width_factor: Positive integer.
      data_format: One of `"channels_first"`, `"channels_last"`.
      interpolation: A string, one of `nearest` or `bilinear`.

  Returns:
      A tensor.

  Raises:
      ValueError: in case of incorrect value for
        `data_format` or `interpolation`.
  """
  if data_format == 'channels_first':
    rows, cols = 2, 3
  elif data_format == 'channels_last':
    rows, cols = 1, 2
  else:
    raise ValueError('Invalid `data_format` argument: %s' % (data_format,))

  new_shape = x.shape[rows:cols + 1]
  if new_shape.is_fully_defined():
    new_shape = constant_op.constant(new_shape.as_list(), dtype='int32')
  else:
    new_shape = array_ops.shape_v2(x)[rows:cols + 1]
  new_shape *= constant_op.constant(
      np.array([height_factor, width_factor], dtype='int32'))

  if data_format == 'channels_first':
    x = permute_dimensions(x, [0, 2, 3, 1])
  if interpolation == 'nearest':
    x = image_ops.resize_images_v2(
        x, new_shape, method=image_ops.ResizeMethod.NEAREST_NEIGHBOR)
  elif interpolation == 'bilinear':
    x = image_ops.resize_images_v2(x, new_shape,
                                   method=image_ops.ResizeMethod.BILINEAR)
  else:
    raise ValueError('interpolation should be one '
                     'of "nearest" or "bilinear".')
  if data_format == 'channels_first':
    x = permute_dimensions(x, [0, 3, 1, 2])

  return x


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def resize_volumes(x, depth_factor, height_factor, width_factor, data_format):
  # [FUNCTION DOC]
  # Resizes a five-dimensional volume by repeating elements along the depth, height and width axes.
  # Inputs: x, depth_factor, height_factor, width_factor (positive integers), data_format (otherwise a
  # value error). Returns the repeated tensor.
  """Resizes the volume contained in a 5D tensor.

  Args:
      x: Tensor or variable to resize.
      depth_factor: Positive integer.
      height_factor: Positive integer.
      width_factor: Positive integer.
      data_format: One of `"channels_first"`, `"channels_last"`.

  Returns:
      A tensor.

  Raises:
      ValueError: if `data_format` is neither
          `channels_last` or `channels_first`.
  """
  if data_format == 'channels_first':
    output = repeat_elements(x, depth_factor, axis=2)
    output = repeat_elements(output, height_factor, axis=3)
    output = repeat_elements(output, width_factor, axis=4)
    return output
  elif data_format == 'channels_last':
    output = repeat_elements(x, depth_factor, axis=1)
    output = repeat_elements(output, height_factor, axis=2)
    output = repeat_elements(output, width_factor, axis=3)
    return output
  else:
    raise ValueError('Invalid data_format: ' + str(data_format))


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def repeat_elements(x, rep, axis):
  # [FUNCTION DOC]
  # Repeats each element along an axis, like numpy repeat. Inputs: x, rep (integer repeat count), axis.
  # If the size along the axis is statically known, splits into slices, repeats each and concatenates.
  # Otherwise uses an auxiliary axis with tile and reshape to support dynamic shapes, then restores the
  # static shape and records a Keras shape attribute on the result (mutating the result tensor).
  # Returns the repeated tensor.
  """Repeats the elements of a tensor along an axis, like `np.repeat`.

  If `x` has shape `(s1, s2, s3)` and `axis` is `1`, the output
  will have shape `(s1, s2 * rep, s3)`.

  Args:
      x: Tensor or variable.
      rep: Python integer, number of times to repeat.
      axis: Axis along which to repeat.

  Returns:
      A tensor.

  Example:

      >>> b = tf.constant([1, 2, 3])
      >>> tf.keras.backend.repeat_elements(b, rep=2, axis=0)
      <tf.Tensor: shape=(6,), dtype=int32,
          numpy=array([1, 1, 2, 2, 3, 3], dtype=int32)>

  """
  x_shape = x.shape.as_list()
  # For static axis
  if x_shape[axis] is not None:
    # slices along the repeat axis
    splits = array_ops.split(value=x,
                             num_or_size_splits=x_shape[axis],
                             axis=axis)
    # repeat each slice the given number of reps
    x_rep = [s for s in splits for _ in range(rep)]
    return concatenate(x_rep, axis)

  # Here we use tf.tile to mimic behavior of np.repeat so that
  # we can handle dynamic shapes (that include None).
  # To do that, we need an auxiliary axis to repeat elements along
  # it and then merge them along the desired axis.

  # Repeating
  auxiliary_axis = axis + 1
  x_shape = array_ops.shape(x)
  x_rep = array_ops.expand_dims(x, axis=auxiliary_axis)
  reps = np.ones(len(x.shape) + 1)
  reps[auxiliary_axis] = rep
  x_rep = array_ops.tile(x_rep, reps)

  # Merging
  reps = np.delete(reps, auxiliary_axis)
  reps[axis] = rep
  reps = array_ops.constant(reps, dtype='int32')
  x_shape *= reps
  x_rep = array_ops.reshape(x_rep, x_shape)

  # Fix shape representation
  x_shape = x.shape.as_list()
  x_rep.set_shape(x_shape)
  x_rep._keras_shape = tuple(x_shape)
  return x_rep


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def repeat(x, n):
  # [FUNCTION DOC]
  # Repeats a two-dimensional tensor of shape samples by dimension n times along a new middle axis,
  # producing samples by n by dimension. Inputs: x and n (integer). Asserts that the rank is two (an
  # assertion error otherwise, not a value error). Returns the tiled tensor.
  """Repeats a 2D tensor.

  if `x` has shape (samples, dim) and `n` is `2`,
  the output will have shape `(samples, 2, dim)`.

  Args:
      x: Tensor or variable.
      n: Python integer, number of times to repeat.

  Returns:
      A tensor.

  Example:

      >>> b = tf.constant([[1, 2], [3, 4]])
      >>> b
      <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
      array([[1, 2],
             [3, 4]], dtype=int32)>
      >>> tf.keras.backend.repeat(b, n=2)
      <tf.Tensor: shape=(2, 2, 2), dtype=int32, numpy=
      array([[[1, 2],
              [1, 2]],
             [[3, 4],
              [3, 4]]], dtype=int32)>

  """
  assert ndim(x) == 2
  x = array_ops.expand_dims(x, 1)
  pattern = array_ops_stack.stack([1, n, 1])
  return array_ops.tile(x, pattern)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def arange(start, stop=None, step=1, dtype='int32'):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says dtype must be an integer type, but any type is accepted (the
  # example uses a float step) and the result is simply cast to that type when it is not int32.
  # Creates a one-dimensional sequence. Inputs: start, stop (optional; when absent start acts as the
  # stop and a negative start gives an empty sequence), step (default one), dtype (default int32).
  # Returns the tensor of values.
  """Creates a 1D tensor containing a sequence of integers.

  The function arguments use the same convention as
  Theano's arange: if only one argument is provided,
  it is in fact the "stop" argument and "start" is 0.

  The default type of the returned tensor is `'int32'` to
  match TensorFlow's default.

  Args:
      start: Start value.
      stop: Stop value.
      step: Difference between two successive values.
      dtype: Integer dtype to use.

  Returns:
      An integer tensor.

  Example:

      >>> tf.keras.backend.arange(start=0, stop=10, step=1.5)
      <tf.Tensor: shape=(7,), dtype=float32,
          numpy=array([0. , 1.5, 3. , 4.5, 6. , 7.5, 9. ], dtype=float32)>



  """
  # Match the behavior of numpy and Theano by returning an empty sequence.
  if stop is None and start < 0:
    start = 0
  result = math_ops.range(start, limit=stop, delta=step, name='arange')
  if dtype != 'int32':
    result = cast(result, dtype)
  return result


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def tile(x, n):
  # [FUNCTION DOC]
  # Tiles a tensor. Inputs: x and n (list of repeat counts, one per dimension; a bare integer is wrapped
  # into a one-element list). Returns the tiled tensor.
  """Creates a tensor by tiling `x` by `n`.

  Args:
      x: A tensor or variable
      n: A list of integer. The length must be the same as the number of
          dimensions in `x`.

  Returns:
      A tiled tensor.
  """
  if isinstance(n, int):
    n = [n]
  return array_ops.tile(x, n)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def flatten(x):
  # [FUNCTION DOC]
  # Flattens a tensor into one dimension. Input x. Returns the one-dimensional tensor.
  """Flatten a tensor.

  Args:
      x: A tensor or variable.

  Returns:
      A tensor, reshaped into 1-D

  Example:

      >>> b = tf.constant([[1, 2], [3, 4]])
      >>> b
      <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
      array([[1, 2],
             [3, 4]], dtype=int32)>
      >>> tf.keras.backend.flatten(b)
      <tf.Tensor: shape=(4,), dtype=int32,
          numpy=array([1, 2, 3, 4], dtype=int32)>

  """
  return array_ops.reshape(x, [-1])


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def batch_flatten(x):
  # [FUNCTION DOC]
  # Turns a tensor of rank n into a two-dimensional tensor keeping the first dimension, flattening each
  # sample. Input x. Uses the dynamic shape to compute the flattened size. Returns the tensor.
  """Turn a nD tensor into a 2D tensor with same 0th dimension.

  In other words, it flattens each data samples of a batch.

  Args:
      x: A tensor or variable.

  Returns:
      A tensor.

  Examples:
    Flattening a 3D tensor to 2D by collapsing the last dimension.

  >>> x_batch = tf.keras.backend.ones(shape=(2, 3, 4, 5))
  >>> x_batch_flatten = batch_flatten(x_batch)
  >>> tf.keras.backend.int_shape(x_batch_flatten)
  (2, 60)

  """
  x = array_ops.reshape(x, array_ops_stack.stack([-1, prod(shape(x)[1:])]))
  return x


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def expand_dims(x, axis=-1):
  # [FUNCTION DOC]
  # Adds a size-one dimension at the given axis. Inputs: x and axis (default last). Returns the tensor.
  """Adds a 1-sized dimension at index "axis".

  Args:
      x: A tensor or variable.
      axis: Position where to add a new axis.

  Returns:
      A tensor with expanded dimensions.
  """
  return array_ops.expand_dims(x, axis)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def squeeze(x, axis):
  # [FUNCTION DOC]
  # Removes a size-one dimension at the given axis. Inputs: x and axis (integer). Returns the tensor.
  """Removes a 1-dimension from the tensor at index "axis".

  Args:
      x: A tensor or variable.
      axis: Axis to drop.

  Returns:
      A tensor with the same data as `x` but reduced dimensions.
  """
  return array_ops.squeeze(x, [axis])


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def temporal_padding(x, padding=(1, 1)):
  # [FUNCTION DOC]
  # Pads the middle dimension of a three-dimensional tensor with zeros. Inputs: x and padding (pair of
  # integers: zeros to add at the start and end of dimension one; asserts the pair has length two).
  # Returns the padded tensor.
  """Pads the middle dimension of a 3D tensor.

  Args:
      x: Tensor or variable.
      padding: Tuple of 2 integers, how many zeros to
          add at the start and end of dim 1.

  Returns:
      A padded 3D tensor.
  """
  assert len(padding) == 2
  pattern = [[0, 0], [padding[0], padding[1]], [0, 0]]
  return array_ops.pad(x, pattern)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def spatial_2d_padding(x, padding=((1, 1), (1, 1)), data_format=None):
  # [FUNCTION DOC]
  # Pads the two spatial dimensions of a four-dimensional tensor with zeros. Inputs: x, padding (pair of
  # pairs, asserted), data_format (channels_first or channels_last, defaults to the configured image
  # data format; otherwise a value error). Returns the padded tensor.
  """Pads the 2nd and 3rd dimensions of a 4D tensor.

  Args:
      x: Tensor or variable.
      padding: Tuple of 2 tuples, padding pattern.
      data_format: One of `channels_last` or `channels_first`.

  Returns:
      A padded 4D tensor.

  Raises:
      ValueError: if `data_format` is neither
          `channels_last` or `channels_first`.
  """
  assert len(padding) == 2
  assert len(padding[0]) == 2
  assert len(padding[1]) == 2
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))

  if data_format == 'channels_first':
    pattern = [[0, 0], [0, 0], list(padding[0]), list(padding[1])]
  else:
    pattern = [[0, 0], list(padding[0]), list(padding[1]), [0, 0]]
  return array_ops.pad(x, pattern)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1)), data_format=None):
  # [FUNCTION DOC]
  # Pads the three spatial dimensions of a five-dimensional tensor with zeros. Inputs: x, padding (three
  # pairs, asserted), data_format (defaults to the configured format; invalid value gives a value
  # error). Returns the padded tensor.
  """Pads 5D tensor with zeros along the depth, height, width dimensions.

  Pads these dimensions with respectively
  "padding[0]", "padding[1]" and "padding[2]" zeros left and right.

  For 'channels_last' data_format,
  the 2nd, 3rd and 4th dimension will be padded.
  For 'channels_first' data_format,
  the 3rd, 4th and 5th dimension will be padded.

  Args:
      x: Tensor or variable.
      padding: Tuple of 3 tuples, padding pattern.
      data_format: One of `channels_last` or `channels_first`.

  Returns:
      A padded 5D tensor.

  Raises:
      ValueError: if `data_format` is neither
          `channels_last` or `channels_first`.

  """
  assert len(padding) == 3
  assert len(padding[0]) == 2
  assert len(padding[1]) == 2
  assert len(padding[2]) == 2
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))

  if data_format == 'channels_first':
    pattern = [[0, 0], [0, 0], [padding[0][0], padding[0][1]],
               [padding[1][0], padding[1][1]], [padding[2][0], padding[2][1]]]
  else:
    pattern = [[0, 0], [padding[0][0], padding[0][1]],
               [padding[1][0], padding[1][1]], [padding[2][0],
                                                padding[2][1]], [0, 0]]
  return array_ops.pad(x, pattern)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def stack(x, axis=0):
  # [FUNCTION DOC]
  # Stacks a list of rank R tensors into one rank R plus one tensor. Inputs: x (list of tensors) and
  # axis (default zero). Returns the stacked tensor.
  """Stacks a list of rank `R` tensors into a rank `R+1` tensor.

  Args:
      x: List of tensors.
      axis: Axis along which to perform stacking.

  Returns:
      A tensor.

  Example:

      >>> a = tf.constant([[1, 2],[3, 4]])
      >>> b = tf.constant([[10, 20],[30, 40]])
      >>> tf.keras.backend.stack((a, b))
      <tf.Tensor: shape=(2, 2, 2), dtype=int32, numpy=
      array([[[ 1,  2],
              [ 3,  4]],
             [[10, 20],
              [30, 40]]], dtype=int32)>

  """
  return array_ops_stack.stack(x, axis=axis)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def one_hot(indices, num_classes):
  # [FUNCTION DOC]
  # Computes the one-hot encoding of an integer tensor. Inputs: indices (integer tensor) and
  # num_classes (depth). Adds a trailing axis of that size. Returns the encoded tensor. (The docstring
  # lists the return section twice.)
  """Computes the one-hot representation of an integer tensor.

  Args:
      indices: nD integer tensor of shape
          `(batch_size, dim1, dim2, ... dim(n-1))`
      num_classes: Integer, number of classes to consider.

  Returns:
      (n + 1)D one hot representation of the input
      with shape `(batch_size, dim1, dim2, ... dim(n-1), num_classes)`

  Returns:
      The one-hot tensor.
  """
  return array_ops.one_hot(indices, depth=num_classes, axis=-1)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def reverse(x, axes):
  # [FUNCTION DOC]
  # Reverses a tensor along the given axes. Inputs: x and axes (integer or iterable of integers; a bare
  # integer is wrapped in a list). Returns the reversed tensor.
  """Reverse a tensor along the specified axes.

  Args:
      x: Tensor to reverse.
      axes: Integer or iterable of integers.
          Axes to reverse.

  Returns:
      A tensor.
  """
  if isinstance(axes, int):
    axes = [axes]
  return array_ops.reverse(x, axes)


# VALUE MANIPULATION
_VALUE_SET_CODE_STRING = """
  >>> K = tf.keras.backend  # Common keras convention
  >>> v = K.variable(1.)

  >>> # reassign
  >>> K.set_value(v, 2.)
  >>> print(K.get_value(v))
  2.0

  >>> # increment
  >>> K.set_value(v, K.get_value(v) + 1)
  >>> print(K.get_value(v))
  3.0

  Variable semantics in TensorFlow 2 are eager execution friendly. The above
  code is roughly equivalent to:

  >>> v = tf.Variable(1.)

  >>> v.assign(2.)
  >>> print(v.numpy())
  2.0

  >>> v.assign_add(1.)
  >>> print(v.numpy())
  3.0"""[3:]  # Prune first newline and indent to match the docstring template.


@doc_controls.do_not_generate_docs
def get_value(x):
  # [FUNCTION DOC]
  # Returns the value of a variable as a numpy array. Input x: variable or tensor; inputs that are not
  # TensorFlow types are returned unchanged. Uses the eager numpy conversion when executing eagerly,
  # for eager tensors, for variables created eagerly but evaluated from a graph (inside eager mode) and
  # when eager execution is on outside functions (inside an initialization scope); otherwise evaluates
  # in the variable's graph using the Keras session. The docstring contains a placeholder that is filled
  # in at import time with an example string. Side effect: may run a session.
  """Returns the value of a variable.

  `backend.get_value` is the complement of `backend.set_value`, and provides
  a generic interface for reading from variables while abstracting away the
  differences between TensorFlow 1.x and 2.x semantics.

  {snippet}

  Args:
      x: input variable.

  Returns:
      A Numpy array.
  """
  if not tensor_util.is_tf_type(x):
    return x
  if context.executing_eagerly() or isinstance(x, ops.EagerTensor):
    return x.numpy()
  if not getattr(x, '_in_graph_mode', True):
    # This is a variable which was created in an eager context, but is being
    # evaluated from a Graph.
    with context.eager_mode():
      return x.numpy()

  if ops.executing_eagerly_outside_functions():
    # This method of evaluating works inside the Keras FuncGraph.
    with ops.init_scope():
      return x.numpy()

  with x.graph.as_default():
    return x.eval(session=get_session((x,)))


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def batch_get_value(tensors):
  # [FUNCTION DOC]
  # Returns the values of several tensors or variables. Input tensors: list of tensors/variables.
  # Executes eagerly by converting each to numpy; raises a runtime error inside a graph function;
  # otherwise runs them all in one session call (an empty list returns an empty list). Returns a list
  # of numpy arrays.
  """Returns the value of more than one tensor variable.

  Args:
      tensors: list of ops to run.

  Returns:
      A list of Numpy arrays.

  Raises:
      RuntimeError: If this method is called inside defun.
  """
  if context.executing_eagerly():
    return [x.numpy() for x in tensors]
  elif ops.inside_function():  # pylint: disable=protected-access
    raise RuntimeError('Cannot get value inside Tensorflow graph function.')
  if tensors:
    return get_session(tensors).run(tensors)
  else:
    return []


@doc_controls.do_not_generate_docs
def set_value(x, value):
  # [FUNCTION DOC]
  # Sets the value of a variable from a numpy array. Inputs: x (variable) and value (array-like of the
  # same shape; converted to the variable's numpy dtype). With eager execution outside functions it
  # assigns directly. In graph mode it builds (once, cached as attributes on the variable) a placeholder
  # with unknown size in each dimension and an assign op, then runs it in the Keras session feeding the
  # value. Returns nothing. Side effects: mutates the variable and may add attributes to it.
  """Sets the value of a variable, from a Numpy array.

  `backend.set_value` is the complement of `backend.get_value`, and provides
  a generic interface for assigning to variables while abstracting away the
  differences between TensorFlow 1.x and 2.x semantics.

  {snippet}

  Args:
      x: Variable to set to a new value.
      value: Value to set the tensor to, as a Numpy array
          (of the same shape).
  """
  value = numpy_compat.np_asarray(value, dtype=dtype_numpy(x))
  if ops.executing_eagerly_outside_functions():
    x.assign(value)
  else:
    with get_graph().as_default():
      tf_dtype = dtypes_module.as_dtype(x.dtype.name.split('_')[0])
      if hasattr(x, '_assign_placeholder'):
        assign_placeholder = x._assign_placeholder
        assign_op = x._assign_op
      else:
        # In order to support assigning weights to resizable variables in
        # Keras, we make a placeholder with the correct number of dimensions
        # but with None in each dimension. This way, we can assign weights
        # of any size (as long as they have the correct dimensionality).
        placeholder_shape = tensor_shape.TensorShape([None] * value.ndim)
        assign_placeholder = array_ops.placeholder(
            tf_dtype, shape=placeholder_shape)
        assign_op = x.assign(assign_placeholder)
        x._assign_placeholder = assign_placeholder
        x._assign_op = assign_op
      get_session().run(assign_op, feed_dict={assign_placeholder: value})


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def batch_set_value(tuples):
  # [FUNCTION DOC]
  # Sets many variables at once. Input tuples: list of pairs of variable and numpy value. In eager mode
  # or inside a function, assigns each directly. Otherwise, in the Keras graph, creates or reuses
  # placeholder and assign ops stored on each variable and runs all of them in one session call with
  # the feed values (does nothing if the list is empty). Returns nothing.
  """Sets the values of many tensor variables at once.

  Args:
      tuples: a list of tuples `(tensor, value)`.
          `value` should be a Numpy array.
  """
  if context.executing_eagerly() or ops.inside_function():
    for x, value in tuples:
      x.assign(numpy_compat.np_asarray(value, dtype=dtype_numpy(x)))
  else:
    with get_graph().as_default():
      if tuples:
        assign_ops = []
        feed_dict = {}
        for x, value in tuples:
          value = numpy_compat.np_asarray(value, dtype=dtype_numpy(x))
          tf_dtype = dtypes_module.as_dtype(x.dtype.name.split('_')[0])
          if hasattr(x, '_assign_placeholder'):
            assign_placeholder = x._assign_placeholder
            assign_op = x._assign_op
          else:
            # In order to support assigning weights to resizable variables in
            # Keras, we make a placeholder with the correct number of dimensions
            # but with None in each dimension. This way, we can assign weights
            # of any size (as long as they have the correct dimensionality).
            placeholder_shape = tensor_shape.TensorShape([None] * value.ndim)
            assign_placeholder = array_ops.placeholder(
                tf_dtype, shape=placeholder_shape)
            assign_op = x.assign(assign_placeholder)
            x._assign_placeholder = assign_placeholder
            x._assign_op = assign_op
          assign_ops.append(assign_op)
          feed_dict[assign_placeholder] = value
        get_session().run(assign_ops, feed_dict=feed_dict)


if get_value.__doc__ is not None:
  get_value.__doc__ = get_value.__doc__.format(snippet=_VALUE_SET_CODE_STRING)
if set_value.__doc__ is not None:
  set_value.__doc__ = set_value.__doc__.format(snippet=_VALUE_SET_CODE_STRING)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def print_tensor(x, message='', summarize=3):
  # [FUNCTION DOC]
  # Prints a message and the tensor value when evaluated, returning an identical tensor.
  # Inputs: x (tensor), message (string prefix, default empty), summarize (how many leading and trailing
  # elements per dimension to print, default three). For graph tensors builds a print op writing to
  # standard output in the Keras graph and returns an identity of x that depends on it (so the returned
  # tensor must be used); otherwise prints immediately and returns x unchanged. Side effect: output to
  # standard output.
  """Prints `message` and the tensor value when evaluated.

  Note that `print_tensor` returns a new tensor identical to `x`
  which should be used in the following code. Otherwise the
  print operation is not taken into account during evaluation.

  Example:

  >>> x = tf.constant([[1.0, 2.0], [3.0, 4.0]])
  >>> _ = tf.keras.backend.print_tensor(x)
  [[1 2]
   [3 4]]

  Args:
      x: Tensor to print.
      message: Message to print jointly with the tensor.
      summarize: The first and last `summarize` elements within each dimension
          are recursively printed per Tensor. If None, then the first 3 and last
          3 elements of each dimension are printed for each tensor. If set to
          -1, it will print all elements of every tensor.

  Returns:
      The same tensor `x`, unchanged.
  """
  if isinstance(x, tensor_lib.Tensor) and hasattr(x, 'graph'):
    with get_graph().as_default():
      op = logging_ops.print_v2(
          message, x, output_stream=sys.stdout, summarize=summarize)
      with ops.control_dependencies([op]):
        return array_ops.identity(x)
  else:
    logging_ops.print_v2(
        message, x, output_stream=sys.stdout, summarize=summarize)
    return x

# GRAPH MANIPULATION


class GraphExecutionFunction:
  # [CLASS DOC]
  # Callable object that runs a graph-mode computation through a session callable, used by the Keras
  # function helper in graph (non-eager) mode and by legacy training loops.
  # Holds: flattened input placeholders, output tensors (variables cast to tensors), a grouped updates
  # operation (run after the first output), a name, an extra feed dictionary, additional fetches wrapped
  # in identity ops, run options and metadata, a mapping of fetch callbacks (functions that receive the
  # fetched value), and a cached session callable together with the feed and fetch signature it was
  # built for. Instances are created with the inputs, outputs, optional updates and session keyword
  # arguments, and are called with input values; the underlying callable is rebuilt whenever feeds,
  # fetches or the session change.
  """Runs a computation graph.

  It's possible to pass arguments to `tf.Session.run()` via `session_kwargs`.
  In particular additional operations via `fetches` argument and additional
  tensor substitutions via `feed_dict` arguments. Note that given
  substitutions are merged with substitutions from `inputs`. Even though
  `feed_dict` is passed once in the constructor (called in `model.compile()`)
  we can modify the values in the dictionary. Through this feed_dict we can
  provide additional substitutions besides Keras inputs.

  Args:
      inputs: Feed placeholders to the computation graph.
      outputs: Output tensors to fetch.
      updates: Additional update ops to be run at function call.
      name: A name to help users identify what this function does.
      session_kwargs: Arguments to `tf.Session.run()`:
                      `fetches`, `feed_dict`, `options`, `run_metadata`.
  """

  def __init__(self, inputs, outputs, updates=None, name=None,
               **session_kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: inputs (placeholders, possibly nested), outputs (tensors, possibly nested),
    # updates (optional list or tuple of assignment pairs or ready ops; a type error is raised if it is
    # not a list or tuple), name (optional label), and extra keyword session arguments of which only
    # feed dictionary, fetches, options and run metadata are recognized; any other key raises a value
    # error. Flattens structures, converts variable outputs to tensors, groups the update operations so
    # they run after the first output, wraps fetches (a single fetch is put in a list) with identity ops,
    # creates the empty fetch callbacks mapping and clears the cached callable state. Returns nothing.
    updates = updates or []
    if not isinstance(updates, (list, tuple)):
      raise TypeError('`updates` in a Keras backend function '
                      'should be a list or tuple.')

    self._inputs_structure = inputs
    self.inputs = nest.flatten(inputs, expand_composites=True)
    self._outputs_structure = outputs
    self.outputs = cast_variables_to_tensor(
        nest.flatten(outputs, expand_composites=True))
    # TODO(b/127668432): Consider using autograph to generate these
    # dependencies in call.
    # Index 0 = total loss or model output for `predict`.
    with ops.control_dependencies([self.outputs[0]]):
      updates_ops = []
      for update in updates:
        if isinstance(update, tuple):
          p, new_p = update
          updates_ops.append(state_ops.assign(p, new_p))
        else:
          # assumed already an op
          updates_ops.append(update)
      self.updates_op = control_flow_ops.group(*updates_ops)
    self.name = name
    # additional tensor substitutions
    self.feed_dict = session_kwargs.pop('feed_dict', None)
    # additional operations
    self.fetches = session_kwargs.pop('fetches', [])
    if not isinstance(self.fetches, list):
      self.fetches = [self.fetches]
    self.run_options = session_kwargs.pop('options', None)
    self.run_metadata = session_kwargs.pop('run_metadata', None)
    # The main use case of `fetches` being passed to a model is the ability
    # to run custom updates
    # This requires us to wrap fetches in `identity` ops.
    self.fetches = [array_ops.identity(x) for x in self.fetches]
    self.session_kwargs = session_kwargs
    # This mapping keeps track of the function that should receive the
    # output from a fetch in `fetches`: { fetch: function(fetch_output) }
    # A Callback can use this to register a function with access to the
    # output values for a fetch it added.
    self.fetch_callbacks = {}

    if session_kwargs:
      raise ValueError('Some keys in session_kwargs are not supported at this '
                       'time: %s' % (session_kwargs.keys(),))

    self._callable_fn = None
    self._feed_arrays = None
    self._feed_symbols = None
    self._symbol_vals = None
    self._fetches = None
    self._session = None

  def _make_callable(self, feed_arrays, feed_symbols, symbol_vals, session):
    # [FUNCTION DOC]
    # HELPER: builds and caches the session callable used by calls of this object.
    # Inputs: feed_arrays (placeholders fed with numpy arrays), feed_symbols (placeholders fed with
    # symbolic tensors), symbol_vals (the symbolic tensors to connect to them; cast to the placeholder
    # dtype if needed) and session. Builds callable options listing feeds (including the extra feed
    # dictionary keys in sorted order), tensor connections, fetches (outputs then extra fetches), target
    # (the updates op) and run options, creates the callable in the session and stores it together with
    # the arguments that identify it. Returns nothing.
    """Generates a callable that runs the graph.

    Args:
      feed_arrays: List of input tensors to be fed Numpy arrays at runtime.
      feed_symbols: List of input tensors to be fed symbolic tensors at runtime.
      symbol_vals: List of symbolic tensors to be fed to `feed_symbols`.
      session: Session to use to generate the callable.

    Returns:
      Function that runs the graph according to the above options.
    """
    # Prepare callable options.
    callable_opts = config_pb2.CallableOptions()
    # Handle external-data feed.
    for x in feed_arrays:
      callable_opts.feed.append(x.name)
    if self.feed_dict:
      for key in sorted(self.feed_dict.keys()):
        callable_opts.feed.append(key.name)
    # Handle symbolic feed.
    for x, y in zip(feed_symbols, symbol_vals):
      connection = callable_opts.tensor_connection.add()
      if x.dtype != y.dtype:
        y = math_ops.cast(y, dtype=x.dtype)
      from_tensor = _as_graph_element(y)
      if from_tensor is None:
        from_tensor = y
      connection.from_tensor = from_tensor.name  # Data tensor
      connection.to_tensor = x.name  # Placeholder
    # Handle fetches.
    for x in self.outputs + self.fetches:
      callable_opts.fetch.append(x.name)
    # Handle updates.
    callable_opts.target.append(self.updates_op.name)
    # Handle run_options.
    if self.run_options:
      callable_opts.run_options.CopyFrom(self.run_options)
    # Create callable.
    callable_fn = session._make_callable_from_options(callable_opts)
    # Cache parameters corresponding to the generated callable, so that
    # we can detect future mismatches and refresh the callable.
    self._callable_fn = callable_fn
    self._feed_arrays = feed_arrays
    self._feed_symbols = feed_symbols
    self._symbol_vals = symbol_vals
    self._fetches = list(self.fetches)
    self._session = session

  def _call_fetch_callbacks(self, fetches_output):
    # [FUNCTION DOC]
    # HELPER: dispatches fetched values to registered callbacks. Input fetches_output: list of values for
    # the extra fetches. For every fetch that has a registered callback, calls it with its value.
    # Returns nothing.
    for fetch, output in zip(self._fetches, fetches_output):
      if fetch in self.fetch_callbacks:
        self.fetch_callbacks[fetch](output)

  def _eval_if_composite(self, tensor):
    # [FUNCTION DOC]
    # HELPER: used on the structured output of a call. Input tensor: an output element. If it is a
    # composite (extension) tensor such as a sparse tensor, evaluates it in the session so that its value
    # type is returned; otherwise returns it unchanged.
    """Helper method which evaluates any CompositeTensors passed to it."""
    # We need to evaluate any composite tensor objects that have been
    # reconstructed in 'pack_sequence_as', since otherwise they'll be output as
    # actual CompositeTensor objects instead of the value(s) contained in the
    # CompositeTensors. E.g., if output_structure contains a SparseTensor, then
    # this ensures that we return its value as a SparseTensorValue rather than
    # a SparseTensor.
    from tensorflow.python.keras.utils import tf_utils  # pylint: disable=g-import-not-at-top
    if tf_utils.is_extension_type(tensor):
      return self._session.run(tensor)
    else:
      return tensor

  def __call__(self, inputs):
    # [FUNCTION DOC]
    # Runs the graph function. Input inputs: values matching the input structure (numpy arrays or
    # symbolic tensors; entries that are nothing are skipped). Obtains the session, sorts values into
    # numpy feeds (cast to each placeholder's dtype) and symbolic feeds, appends values of the extra feed
    # dictionary, rebuilds the cached callable when feeds, fetches or session changed, runs it, calls the
    # fetch callbacks, packs the first outputs into the original output structure and evaluates any
    # composite tensors in it. Returns the structure of output values. The slice used to select fetch
    # results is the whole list when there are no extra fetches, which is harmless because the callback
    # loop is then empty.
    inputs = nest.flatten(inputs, expand_composites=True)

    session = get_session(inputs)
    feed_arrays = []
    array_vals = []
    feed_symbols = []
    symbol_vals = []
    for tensor, value in zip(self.inputs, inputs):
      if value is None:
        continue

      if tensor_util.is_tf_type(value):
        # Case: feeding symbolic tensor.
        feed_symbols.append(tensor)
        symbol_vals.append(value)
      else:
        # Case: feeding Numpy array.
        feed_arrays.append(tensor)
        # We need to do array conversion and type casting at this level, since
        # `callable_fn` only supports exact matches.
        tensor_type = dtypes_module.as_dtype(tensor.dtype)
        array_vals.append(numpy_compat.np_asarray(
            value, dtype=tensor_type.as_numpy_dtype))

    if self.feed_dict:
      for key in sorted(self.feed_dict.keys()):
        array_vals.append(
            numpy_compat.np_asarray(
                self.feed_dict[key], dtype=key.dtype.as_numpy_dtype))

    # Refresh callable if anything has changed.
    if (self._callable_fn is None or feed_arrays != self._feed_arrays or
        symbol_vals != self._symbol_vals or
        feed_symbols != self._feed_symbols or self.fetches != self._fetches or
        session != self._session):
      self._make_callable(feed_arrays, feed_symbols, symbol_vals, session)

    fetched = self._callable_fn(*array_vals,
                                run_metadata=self.run_metadata)
    self._call_fetch_callbacks(fetched[-len(self._fetches):])
    output_structure = nest.pack_sequence_as(
        self._outputs_structure,
        fetched[:len(self.outputs)],
        expand_composites=True)
    # We need to evaluate any composite tensor objects that have been
    # reconstructed in 'pack_sequence_as', since otherwise they'll be output as
    # actual CompositeTensor objects instead of the value(s) contained in the
    # CompositeTensors. E.g., if output_structure contains a SparseTensor, then
    # this ensures that we return its value as a SparseTensorValue rather than
    # a SparseTensor.
    return nest.map_structure(self._eval_if_composite, output_structure)


@doc_controls.do_not_generate_docs
def function(inputs, outputs, updates=None, name=None, **kwargs):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it returns output values as numpy arrays, but it actually returns a
  # callable object (a plain function in eager mode, a graph execution function otherwise) that must be
  # called with input values. The documented value error "if in eager execution" only applies when
  # session keyword arguments or updates are passed.
  # Instantiates a Keras function. Inputs: inputs (placeholders), outputs (tensors), updates (optional
  # update ops), name, and extra keyword arguments forwarded to the session run. With eager execution
  # outside functions it rejects keyword arguments and updates (value errors), builds a functional
  # model from inputs and outputs, and returns a function that applies the model and converts results
  # to numpy or Python values (wrapping a single-output list result). Otherwise it validates that every
  # keyword is an argument of the session run method (or inputs, outputs, updates, name), raising a
  # value error otherwise, and returns a graph execution function.
  """Instantiates a Keras function.

  Args:
      inputs: List of placeholder tensors.
      outputs: List of output tensors.
      updates: List of update ops.
      name: String, name of function.
      **kwargs: Passed to `tf.Session.run`.

  Returns:
      Output values as Numpy arrays.

  Raises:
      ValueError: if invalid kwargs are passed in or if in eager execution.
  """
  if ops.executing_eagerly_outside_functions():
    if kwargs:
      raise ValueError('Session keyword arguments are not supported during '
                       'eager execution. You passed: %s' % (kwargs,))
    if updates:
      raise ValueError('`updates` argument is not supported during '
                       'eager execution. You passed: %s' % (updates,))
    from tensorflow.python.keras import models  # pylint: disable=g-import-not-at-top
    from tensorflow.python.keras.utils import tf_utils  # pylint: disable=g-import-not-at-top
    model = models.Model(inputs=inputs, outputs=outputs)

    wrap_outputs = isinstance(outputs, list) and len(outputs) == 1
    def func(model_inputs):
      # [FUNCTION DOC]
      # HELPER: the callable returned by function in eager mode. Input model_inputs: values for the model.
      # Calls the model, wraps a single output in a list when the original outputs were a one-element list,
      # and converts the result to numpy or Python values. Returns that converted result.
      outs = model(model_inputs)
      if wrap_outputs:
        outs = [outs]
      return tf_utils.sync_to_numpy_or_python_type(outs)

    return func

  if kwargs:
    for key in kwargs:
      if (key not in tf_inspect.getfullargspec(session_module.Session.run)[0]
          and key not in ['inputs', 'outputs', 'updates', 'name']):
        msg = ('Invalid argument "%s" passed to K.function with TensorFlow '
               'backend') % key
        raise ValueError(msg)
  return GraphExecutionFunction(
      inputs, outputs, updates=updates, name=name, **kwargs)


@doc_controls.do_not_generate_docs
def gradients(loss, variables):
  # [FUNCTION DOC]
  # Returns the gradients of a loss with respect to variables. Inputs: loss (scalar tensor) and
  # variables (list). Uses graph gradients with colocation of gradients with their ops. Returns the
  # list of gradient tensors (graph mode only).
  """Returns the gradients of `loss` w.r.t. `variables`.

  Args:
      loss: Scalar tensor to minimize.
      variables: List of variables.

  Returns:
      A gradients tensor.
  """
  return gradients_module.gradients(
      loss, variables, colocate_gradients_with_ops=True)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def stop_gradient(variables):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says a list of tensors is returned for a list input, but for a list or
  # tuple the code returns a lazy map iterator, not a list; it can be consumed only once.
  # Blocks gradient flow through tensors. Input variables: a tensor, or a list or tuple of tensors.
  # Returns the stopped tensor, or the lazy iterator of stopped tensors for sequences.
  """Returns `variables` but with zero gradient w.r.t. every other variable.

  Args:
      variables: Tensor or list of tensors to consider constant with respect
        to any other variable.


  Returns:
      A single tensor or a list of tensors (depending on the passed argument)
      that has no gradient with respect to any other variable.
  """
  if isinstance(variables, (list, tuple)):
    return map(array_ops.stop_gradient, variables)
  return array_ops.stop_gradient(variables)


# CONTROL FLOW


@dispatch.add_dispatch_support
def rnn(step_function,
        inputs,
        initial_states,
        go_backwards=False,
        mask=None,
        constants=None,
        unroll=False,
        input_length=None,
        time_major=False,
        zero_output_for_mask=False):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says a value error is raised when a mask is given but no states are
  # provided; the code has no such check. Also, when building under an XLA context the maximum number
  # of iterations is computed from input_length, which fails if input_length was left as nothing.
  # Iterates a step function over the time dimension of (possibly nested) input tensors.
  # Inputs: step_function (callable taking the input slice at one step and the list of states plus
  # constants, returning an output and new states), inputs (tensors of shape samples by time by
  # features, or nested structure of them; at least rank three otherwise a value error from the rank
  # check), initial_states (tensor or nested structure of initial states), go_backwards (iterate in
  # reverse), mask (optional boolean-like tensor, samples by time, masked steps keep the previous
  # output and states), constants (extra values passed to every step), unroll (use a Python loop,
  # requires a static number of steps otherwise a value error), input_length (integer or tensor of
  # valid lengths, used for masking when no mask is given), time_major (inputs already time first;
  # otherwise batch and time axes are swapped at the start and end), zero_output_for_mask (masked
  # outputs are zeros instead of the previous output).
  # Unrolled mode steps through unstacked time slices; the non-unrolled mode uses tensor arrays and a
  # symbolic while loop with 32 parallel iterations and swap memory (maximum iterations set only under
  # XLA). Static time and batch shapes are restored on the outputs. Returns a tuple of the last output,
  # all outputs stacked over time, and the final states. Side effects: builds graph ops; the step
  # function is traced once on the first step to learn the output structure.
  """Iterates over the time dimension of a tensor.

  Args:
      step_function: RNN step function.
          Args;
              input; Tensor with shape `(samples, ...)` (no time dimension),
                  representing input for the batch of samples at a certain
                  time step.
              states; List of tensors.
          Returns;
              output; Tensor with shape `(samples, output_dim)`
                  (no time dimension).
              new_states; List of tensors, same length and shapes
                  as 'states'. The first state in the list must be the
                  output tensor at the previous timestep.
      inputs: Tensor of temporal data of shape `(samples, time, ...)`
          (at least 3D), or nested tensors, and each of which has shape
          `(samples, time, ...)`.
      initial_states: Tensor with shape `(samples, state_size)`
          (no time dimension), containing the initial values for the states used
          in the step function. In the case that state_size is in a nested
          shape, the shape of initial_states will also follow the nested
          structure.
      go_backwards: Boolean. If True, do the iteration over the time
          dimension in reverse order and return the reversed sequence.
      mask: Binary tensor with shape `(samples, time, 1)`,
          with a zero for every element that is masked.
      constants: List of constant values passed at each step.
      unroll: Whether to unroll the RNN or to use a symbolic `while_loop`.
      input_length: An integer or a 1-D Tensor, depending on whether
          the time dimension is fixed-length or not. In case of variable length
          input, it is used for masking in case there's no mask specified.
      time_major: Boolean. If true, the inputs and outputs will be in shape
          `(timesteps, batch, ...)`, whereas in the False case, it will be
          `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
          efficient because it avoids transposes at the beginning and end of the
          RNN calculation. However, most TensorFlow data is batch-major, so by
          default this function accepts input and emits output in batch-major
          form.
      zero_output_for_mask: Boolean. If True, the output for masked timestep
          will be zeros, whereas in the False case, output from previous
          timestep is returned.

  Returns:
      A tuple, `(last_output, outputs, new_states)`.
          last_output: the latest output of the rnn, of shape `(samples, ...)`
          outputs: tensor with shape `(samples, time, ...)` where each
              entry `outputs[s, t]` is the output of the step function
              at time `t` for sample `s`.
          new_states: list of tensors, latest states returned by
              the step function, of shape `(samples, ...)`.

  Raises:
      ValueError: if input dimension is less than 3.
      ValueError: if `unroll` is `True` but input timestep is not a fixed
      number.
      ValueError: if `mask` is provided (not `None`) but states is not provided
          (`len(states)` == 0).
  """

  def swap_batch_timestep(input_t):
    # [FUNCTION DOC]
    # HELPER: local function of rnn. Input input_t: a tensor of rank at least two. Returns it with the
    # first two axes (batch and time) exchanged.
    # Swap the batch and timestep dim for the incoming tensor.
    axes = list(range(len(input_t.shape)))
    axes[0], axes[1] = 1, 0
    return array_ops.transpose(input_t, axes)

  if not time_major:
    inputs = nest.map_structure(swap_batch_timestep, inputs)

  flatted_inputs = nest.flatten(inputs)
  time_steps = flatted_inputs[0].shape[0]
  batch = flatted_inputs[0].shape[1]
  time_steps_t = array_ops.shape(flatted_inputs[0])[0]

  for input_ in flatted_inputs:
    input_.shape.with_rank_at_least(3)

  if mask is not None:
    if mask.dtype != dtypes_module.bool:
      mask = math_ops.cast(mask, dtypes_module.bool)
    if len(mask.shape) == 2:
      mask = expand_dims(mask)
    if not time_major:
      mask = swap_batch_timestep(mask)

  if constants is None:
    constants = []

  # tf.where needs its condition tensor to be the same shape as its two
  # result tensors, but in our case the condition (mask) tensor is
  # (nsamples, 1), and inputs are (nsamples, ndimensions) or even more.
  # So we need to broadcast the mask to match the shape of inputs.
  # That's what the tile call does, it just repeats the mask along its
  # second dimension n times.
  def _expand_mask(mask_t, input_t, fixed_dim=1):
    # [FUNCTION DOC]
    # HELPER: local function of rnn that broadcasts a mask to the shape of an input or output.
    # Inputs: mask_t (mask tensor), input_t (tensor to match), fixed_dim (number of leading dimensions to
    # keep as in the mask, default one). Raises value errors if either argument is a nested structure.
    # Appends trailing unit axes to the mask until the ranks match, then tiles it over the remaining
    # dimensions of the target. Returns the tiled mask.
    if nest.is_nested(mask_t):
      raise ValueError('mask_t is expected to be tensor, but got %s' % mask_t)
    if nest.is_nested(input_t):
      raise ValueError('input_t is expected to be tensor, but got %s' % input_t)
    rank_diff = len(input_t.shape) - len(mask_t.shape)
    for _ in range(rank_diff):
      mask_t = array_ops.expand_dims(mask_t, -1)
    multiples = [1] * fixed_dim + input_t.shape.as_list()[fixed_dim:]
    return array_ops.tile(mask_t, multiples)

  if unroll:
    if not time_steps:
      raise ValueError('Unrolling requires a fixed number of timesteps.')
    states = tuple(initial_states)
    successive_states = []
    successive_outputs = []

    # Process the input tensors. The input tensor need to be split on the
    # time_step dim, and reverse if go_backwards is True. In the case of nested
    # input, the input is flattened and then transformed individually.
    # The result of this will be a tuple of lists, each of the item in tuple is
    # list of the tensor with shape (batch, feature)
    def _process_single_input_t(input_t):
      # [FUNCTION DOC]
      # HELPER: local function of rnn (unrolled path). Input input_t: one input tensor with time first.
      # Unstacks it along time into a list of per-step tensors, reversed when going backwards. Returns the
      # list.
      input_t = array_ops_stack.unstack(input_t)  # unstack for time_step dim
      if go_backwards:
        input_t.reverse()
      return input_t

    if nest.is_nested(inputs):
      processed_input = nest.map_structure(_process_single_input_t, inputs)
    else:
      processed_input = (_process_single_input_t(inputs),)

    def _get_input_tensor(time):
      # [FUNCTION DOC]
      # HELPER: local function of rnn (unrolled path). Input time: step index. Collects that step from each
      # processed input and packs them into the original input structure. Returns that structure.
      inp = [t_[time] for t_ in processed_input]
      return nest.pack_sequence_as(inputs, inp)

    if mask is not None:
      mask_list = array_ops_stack.unstack(mask)
      if go_backwards:
        mask_list.reverse()

      for i in range(time_steps):
        inp = _get_input_tensor(i)
        mask_t = mask_list[i]
        output, new_states = step_function(inp,
                                           tuple(states) + tuple(constants))
        tiled_mask_t = _expand_mask(mask_t, output)

        if not successive_outputs:
          prev_output = zeros_like(output)
        else:
          prev_output = successive_outputs[-1]

        output = array_ops.where_v2(tiled_mask_t, output, prev_output)

        flat_states = nest.flatten(states)
        flat_new_states = nest.flatten(new_states)
        tiled_mask_t = tuple(_expand_mask(mask_t, s) for s in flat_states)
        flat_final_states = tuple(
            array_ops.where_v2(m, s, ps)
            for m, s, ps in zip(tiled_mask_t, flat_new_states, flat_states))
        states = nest.pack_sequence_as(states, flat_final_states)

        successive_outputs.append(output)
        successive_states.append(states)
      last_output = successive_outputs[-1]
      new_states = successive_states[-1]
      outputs = array_ops_stack.stack(successive_outputs)

      if zero_output_for_mask:
        last_output = array_ops.where_v2(
            _expand_mask(mask_list[-1], last_output), last_output,
            zeros_like(last_output))
        outputs = array_ops.where_v2(
            _expand_mask(mask, outputs, fixed_dim=2), outputs,
            zeros_like(outputs))

    else:  # mask is None
      for i in range(time_steps):
        inp = _get_input_tensor(i)
        output, states = step_function(inp, tuple(states) + tuple(constants))
        successive_outputs.append(output)
        successive_states.append(states)
      last_output = successive_outputs[-1]
      new_states = successive_states[-1]
      outputs = array_ops_stack.stack(successive_outputs)

  else:  # Unroll == False
    states = tuple(initial_states)

    # Create input tensor array, if the inputs is nested tensors, then it will
    # be flattened first, and tensor array will be created one per flattened
    # tensor.
    input_ta = tuple(
        tensor_array_ops.TensorArray(
            dtype=inp.dtype,
            size=time_steps_t,
            tensor_array_name='input_ta_%s' % i)
        for i, inp in enumerate(flatted_inputs))
    input_ta = tuple(
        ta.unstack(input_) if not go_backwards else ta
        .unstack(reverse(input_, 0))
        for ta, input_ in zip(input_ta, flatted_inputs))

    # Get the time(0) input and compute the output for that, the output will be
    # used to determine the dtype of output tensor array. Don't read from
    # input_ta due to TensorArray clear_after_read default to True.
    input_time_zero = nest.pack_sequence_as(inputs,
                                            [inp[0] for inp in flatted_inputs])
    # output_time_zero is used to determine the cell output shape and its dtype.
    # the value is discarded.
    output_time_zero, _ = step_function(
        input_time_zero, tuple(initial_states) + tuple(constants))
    output_ta = tuple(
        tensor_array_ops.TensorArray(
            dtype=out.dtype,
            size=time_steps_t,
            element_shape=out.shape,
            tensor_array_name='output_ta_%s' % i)
        for i, out in enumerate(nest.flatten(output_time_zero)))

    time = constant_op.constant(0, dtype='int32', name='time')

    # We only specify the 'maximum_iterations' when building for XLA since that
    # causes slowdowns on GPU in TF.
    if (not context.executing_eagerly() and
        control_flow_util.GraphOrParentsInXlaContext(ops.get_default_graph())):
      max_iterations = math_ops.reduce_max(input_length)
    else:
      max_iterations = None

    while_loop_kwargs = {
        'cond': lambda time, *_: time < time_steps_t,
        'maximum_iterations': max_iterations,
        'parallel_iterations': 32,
        'swap_memory': True,
    }
    if mask is not None:
      if go_backwards:
        mask = reverse(mask, 0)

      mask_ta = tensor_array_ops.TensorArray(
          dtype=dtypes_module.bool,
          size=time_steps_t,
          tensor_array_name='mask_ta')
      mask_ta = mask_ta.unstack(mask)

      def masking_fn(time):
        # [FUNCTION DOC]
        # HELPER: local function of rnn (masked, non-unrolled path). Input time: scalar step index. Returns the
        # mask value for that step read from the mask tensor array.
        return mask_ta.read(time)

      def compute_masked_output(mask_t, flat_out, flat_mask):
        # [FUNCTION DOC]
        # HELPER: local function of rnn (mask tensor version). Inputs: mask_t (mask at the step), flat_out
        # (flat list of new outputs or states), flat_mask (flat list of values to keep where masked). Expands
        # the mask to each output's shape and selects, per element, the new value where the mask is true and
        # the kept value elsewhere. Returns a tuple of selected tensors.
        tiled_mask_t = tuple(
            _expand_mask(mask_t, o, fixed_dim=len(mask_t.shape))
            for o in flat_out)
        return tuple(
            array_ops.where_v2(m, o, fm)
            for m, o, fm in zip(tiled_mask_t, flat_out, flat_mask))
    elif isinstance(input_length, tensor_lib.Tensor):
      if go_backwards:
        max_len = math_ops.reduce_max(input_length, axis=0)
        rev_input_length = math_ops.subtract(max_len - 1, input_length)

        def masking_fn(time):
          # [FUNCTION DOC]
          # HELPER: local function of rnn (length based masking, going backwards). Input time: step index.
          # Returns a boolean tensor, true for samples whose reversed length is below the step, comparing with
          # the per-sample length offset from the maximum length.
          return math_ops.less(rev_input_length, time)
      else:

        def masking_fn(time):
          # [FUNCTION DOC]
          # HELPER: local function of rnn (length based masking, going forward). Input time: step index. Returns
          # a boolean tensor, true for samples whose valid length is greater than the step.
          return math_ops.greater(input_length, time)

      def compute_masked_output(mask_t, flat_out, flat_mask):
        # [FUNCTION DOC]
        # HELPER: local function of rnn (length based masking). Inputs: mask_t (per-sample boolean mask),
        # flat_out and flat_mask (matching flat lists). Selects the new value where the mask is true and the
        # kept value otherwise, without expanding the mask (relies on the selection op to broadcast it along
        # the batch axis). Returns a tuple of tensors.
        return tuple(
            array_ops.where(mask_t, o, zo)
            for (o, zo) in zip(flat_out, flat_mask))
    else:
      masking_fn = None

    if masking_fn is not None:
      # Mask for the T output will be base on the output of T - 1. In the case
      # T = 0, a zero filled tensor will be used.
      flat_zero_output = tuple(array_ops.zeros_like(o)
                               for o in nest.flatten(output_time_zero))
      def _step(time, output_ta_t, prev_output, *states):
        # [FUNCTION DOC]
        # HELPER: loop body of rnn used when masking is active. Inputs: time (step index), output_ta_t (tuple
        # of output tensor arrays), prev_output (flat tuple of outputs of the previous step) and the states
        # as extra positional values. Reads the step's input, runs the user step function, masks the output
        # (with zeros or the previous output depending on the zero-output option) and the new states (keeping
        # the old states where masked, and giving new states the old static shape), writes the masked outputs
        # into the tensor arrays. Returns a tuple of the next time, updated output arrays, the masked outputs
        # and the new states.
        """RNN step function.

        Args:
            time: Current timestep value.
            output_ta_t: TensorArray.
            prev_output: tuple of outputs from time - 1.
            *states: List of states.

        Returns:
            Tuple: `(time + 1, output_ta_t, output) + tuple(new_states)`
        """
        current_input = tuple(ta.read(time) for ta in input_ta)
        # maybe set shape.
        current_input = nest.pack_sequence_as(inputs, current_input)
        mask_t = masking_fn(time)
        output, new_states = step_function(current_input,
                                           tuple(states) + tuple(constants))
        # mask output
        flat_output = nest.flatten(output)
        flat_mask_output = (flat_zero_output if zero_output_for_mask
                            else nest.flatten(prev_output))
        flat_new_output = compute_masked_output(mask_t, flat_output,
                                                flat_mask_output)

        # mask states
        flat_state = nest.flatten(states)
        flat_new_state = nest.flatten(new_states)
        for state, new_state in zip(flat_state, flat_new_state):
          if isinstance(new_state, tensor_lib.Tensor):
            new_state.set_shape(state.shape)
        flat_final_state = compute_masked_output(mask_t, flat_new_state,
                                                 flat_state)
        new_states = nest.pack_sequence_as(new_states, flat_final_state)

        output_ta_t = tuple(
            ta.write(time, out)
            for ta, out in zip(output_ta_t, flat_new_output))
        return (time + 1, output_ta_t,
                tuple(flat_new_output)) + tuple(new_states)

      final_outputs = while_loop.while_loop(
          body=_step,
          loop_vars=(time, output_ta, flat_zero_output) + states,
          **while_loop_kwargs)
      # Skip final_outputs[2] which is the output for final timestep.
      new_states = final_outputs[3:]
    else:
      def _step(time, output_ta_t, *states):
        # [FUNCTION DOC]
        # HELPER: loop body of rnn used when there is no masking. Inputs: time (step index), output_ta_t
        # (tuple of output tensor arrays) and the states as extra positional values. Reads and packs the
        # step's input, runs the user step function, copies static shapes of old states to new states, writes
        # the flattened output into the tensor arrays and packs new states in the structure of the initial
        # states. Returns a tuple of the next time, updated output arrays and the new states.
        """RNN step function.

        Args:
            time: Current timestep value.
            output_ta_t: TensorArray.
            *states: List of states.

        Returns:
            Tuple: `(time + 1,output_ta_t) + tuple(new_states)`
        """
        current_input = tuple(ta.read(time) for ta in input_ta)
        current_input = nest.pack_sequence_as(inputs, current_input)
        output, new_states = step_function(current_input,
                                           tuple(states) + tuple(constants))
        flat_state = nest.flatten(states)
        flat_new_state = nest.flatten(new_states)
        for state, new_state in zip(flat_state, flat_new_state):
          if isinstance(new_state, tensor_lib.Tensor):
            new_state.set_shape(state.shape)

        flat_output = nest.flatten(output)
        output_ta_t = tuple(
            ta.write(time, out) for ta, out in zip(output_ta_t, flat_output))
        new_states = nest.pack_sequence_as(initial_states, flat_new_state)
        return (time + 1, output_ta_t) + tuple(new_states)

      final_outputs = while_loop.while_loop(
          body=_step, loop_vars=(time, output_ta) + states, **while_loop_kwargs)
      new_states = final_outputs[2:]

    output_ta = final_outputs[1]

    outputs = tuple(o.stack() for o in output_ta)
    last_output = tuple(o[-1] for o in outputs)

    outputs = nest.pack_sequence_as(output_time_zero, outputs)
    last_output = nest.pack_sequence_as(output_time_zero, last_output)

  # static shape inference
  def set_shape(output_):
    # [FUNCTION DOC]
    # HELPER: local function of rnn performing static shape inference. Input output_: any object. If it
    # is a tensor, sets its static first two dimensions to the known time steps and batch size (mutating
    # its static shape) and returns it; other objects are returned unchanged.
    if isinstance(output_, tensor_lib.Tensor):
      shape = output_.shape.as_list()
      shape[0] = time_steps
      shape[1] = batch
      output_.set_shape(shape)
    return output_

  outputs = nest.map_structure(set_shape, outputs)

  if not time_major:
    outputs = nest.map_structure(swap_batch_timestep, outputs)

  return last_output, outputs, new_states


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def switch(condition, then_expression, else_expression):
  # [FUNCTION DOC]
  # Switches between two operations depending on a scalar or per-sample condition.
  # Inputs: condition (integer or boolean tensor, cast to boolean), then_expression and else_expression
  # (tensors or callables returning tensors, of the same shape). With a scalar condition uses a
  # conditional op, wrapping non-callables into functions. Otherwise evaluates the callables and uses an
  # element selection op, raising a value error if the rank of the condition exceeds that of the
  # expressions; for conditions of rank above one the condition is reshaped and tiled to the expression
  # shape. Returns the selected tensor.
  """Switches between two operations depending on a scalar value.

  Note that both `then_expression` and `else_expression`
  should be symbolic tensors of the *same shape*.

  Args:
      condition: tensor (`int` or `bool`).
      then_expression: either a tensor, or a callable that returns a tensor.
      else_expression: either a tensor, or a callable that returns a tensor.

  Returns:
      The selected tensor.

  Raises:
      ValueError: If rank of `condition` is greater than rank of expressions.
  """
  if condition.dtype != dtypes_module.bool:
    condition = math_ops.cast(condition, 'bool')
  cond_ndim = ndim(condition)
  if not cond_ndim:
    if not callable(then_expression):

      def then_expression_fn():
        # [FUNCTION DOC]
        # HELPER: local wrapper in switch that turns a non-callable then-expression into a callable.
        # No inputs; returns the captured expression.
        return then_expression
    else:
      then_expression_fn = then_expression
    if not callable(else_expression):

      def else_expression_fn():
        # [FUNCTION DOC]
        # HELPER: local wrapper in switch that turns a non-callable else-expression into a callable.
        # No inputs; returns the captured expression.
        return else_expression
    else:
      else_expression_fn = else_expression
    x = cond.cond(condition, then_expression_fn, else_expression_fn)
  else:
    # tf.where needs its condition tensor
    # to be the same shape as its two
    # result tensors
    if callable(then_expression):
      then_expression = then_expression()
    if callable(else_expression):
      else_expression = else_expression()
    expr_ndim = ndim(then_expression)
    if cond_ndim > expr_ndim:
      raise ValueError('Rank of `condition` should be less than or'
                       ' equal to rank of `then_expression` and '
                       '`else_expression`. ndim(condition)=' + str(cond_ndim) +
                       ', ndim(then_expression)'
                       '=' + str(expr_ndim))
    if cond_ndim > 1:
      ndim_diff = expr_ndim - cond_ndim
      cond_shape = array_ops.concat(
          [array_ops.shape(condition), [1] * ndim_diff], axis=0)
      condition = array_ops.reshape(condition, cond_shape)
      expr_shape = array_ops.shape(then_expression)
      shape_diff = expr_shape - cond_shape
      tile_shape = array_ops.where_v2(shape_diff > 0, expr_shape,
                                      array_ops.ones_like(expr_shape))
      condition = array_ops.tile(condition, tile_shape)
    x = array_ops.where_v2(condition, then_expression, else_expression)
  return x


@doc_controls.do_not_generate_docs
def in_train_phase(x, alt, training=None):
  # [FUNCTION DOC]
  # Selects x in training phase and alt otherwise. Inputs: x and alt (tensors or callables returning
  # tensors with the same shape) and training (optional: boolean, integer or scalar tensor). When
  # nothing is given it first takes the training value from the current layer call context and then
  # falls back to the global learning phase. If the resulting value is a plain number or boolean one or
  # true, returns x (calling it if callable); zero or false returns alt (calling it if callable);
  # otherwise (tensor or other value) delegates to switch. Returns the selection.
  """Selects `x` in train phase, and `alt` otherwise.

  Note that `alt` should have the *same shape* as `x`.

  Args:
      x: What to return in train phase
          (tensor or callable that returns a tensor).
      alt: What to return otherwise
          (tensor or callable that returns a tensor).
      training: Optional scalar tensor
          (or Python boolean, or Python integer)
          specifying the learning phase.

  Returns:
      Either `x` or `alt` based on the `training` flag.
      the `training` flag defaults to `K.learning_phase()`.
  """
  from tensorflow.python.keras.engine import base_layer_utils  # pylint: disable=g-import-not-at-top
  if training is None:
    training = base_layer_utils.call_context().training

  if training is None:
    training = learning_phase()

  # TODO(b/138862903): Handle the case when training is tensor.
  if not tensor_util.is_tf_type(training):
    if training == 1 or training is True:
      if callable(x):
        return x()
      else:
        return x

    elif training == 0 or training is False:
      if callable(alt):
        return alt()
      else:
        return alt

  # else: assume learning phase is a placeholder tensor.
  x = switch(training, x, alt)
  return x


@doc_controls.do_not_generate_docs
def in_test_phase(x, alt, training=None):
  # [FUNCTION DOC]
  # Selects x in test phase and alt otherwise; implemented by calling the train-phase selector with the
  # two arguments swapped. Inputs: x, alt, training (optional learning phase value). Returns the
  # selection.
  """Selects `x` in test phase, and `alt` otherwise.

  Note that `alt` should have the *same shape* as `x`.

  Args:
      x: What to return in test phase
          (tensor or callable that returns a tensor).
      alt: What to return otherwise
          (tensor or callable that returns a tensor).
      training: Optional scalar tensor
          (or Python boolean, or Python integer)
          specifying the learning phase.

  Returns:
      Either `x` or `alt` based on `K.learning_phase`.
  """
  return in_train_phase(alt, x, training=training)


# NN OPERATIONS


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def relu(x, alpha=0., max_value=None, threshold=0):
  # [FUNCTION DOC]
  # Rectified linear unit with optional slope, saturation and threshold. Inputs: x (tensor, array or
  # list), alpha (slope below the threshold, default zero), max_value (saturation, nothing for none),
  # threshold (default zero). A nonzero alpha with no maximum and zero threshold uses the leaky op;
  # otherwise computes the negative part, applies a threshold mask or the plain/capped-at-six native ops
  # or clipping, and subtracts alpha times the negative part. Returns the tensor.
  """Rectified linear unit.

  With default values, it returns element-wise `max(x, 0)`.

  Otherwise, it follows:
  `f(x) = max_value` for `x >= max_value`,
  `f(x) = x` for `threshold <= x < max_value`,
  `f(x) = alpha * (x - threshold)` otherwise.

  Args:
      x: A tensor or variable.
      alpha: A scalar, slope of negative section (default=`0.`).
      max_value: float. Saturation threshold.
      threshold: float. Threshold value for thresholded activation.

  Returns:
      A tensor.
  """
  # While x can be a tensor or variable, we also see cases where
  # numpy arrays, lists, tuples are passed as well.
  # lists, tuples do not have 'dtype' attribute.
  dtype = getattr(x, 'dtype', floatx())
  if alpha != 0.:
    if max_value is None and threshold == 0:
      return nn.leaky_relu(x, alpha=alpha)

    if threshold != 0:
      negative_part = nn.relu(-x + threshold)
    else:
      negative_part = nn.relu(-x)

  clip_max = max_value is not None

  if threshold != 0:
    # computes x for x > threshold else 0
    x = x * math_ops.cast(math_ops.greater(x, threshold), dtype=dtype)
  elif max_value == 6:
    # if no threshold, then can use nn.relu6 native TF op for performance
    x = nn.relu6(x)
    clip_max = False
  else:
    x = nn.relu(x)

  if clip_max:
    max_value = _constant_to_tensor(max_value, x.dtype.base_dtype)
    zero = _constant_to_tensor(0, x.dtype.base_dtype)
    x = clip_ops.clip_by_value(x, zero, max_value)

  if alpha != 0.:
    alpha = _to_tensor(alpha, x.dtype.base_dtype)
    x -= alpha * negative_part
  return x


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def elu(x, alpha=1.):
  # [FUNCTION DOC]
  # Exponential linear unit. Inputs: x and alpha (scale of the negative part, default one). Returns the
  # native op result when alpha is one, otherwise the same with the negative part multiplied by alpha.
  """Exponential linear unit.

  Args:
      x: A tensor or variable to compute the activation function for.
      alpha: A scalar, slope of negative section.

  Returns:
      A tensor.
  """
  res = nn.elu(x)
  if alpha == 1:
    return res
  else:
    return array_ops.where_v2(x > 0, res, alpha * res)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def softmax(x, axis=-1):
  # [FUNCTION DOC]
  # Softmax. Inputs: x and axis (default last). Returns the tensor of probabilities.
  """Softmax of a tensor.

  Args:
      x: A tensor or variable.
      axis: The dimension softmax would be performed on.
          The default is -1 which indicates the last dimension.

  Returns:
      A tensor.
  """
  return nn.softmax(x, axis=axis)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def softplus(x):
  # [FUNCTION DOC]
  # Softplus of a tensor. Input x. Returns a tensor.
  """Softplus of a tensor.

  Args:
      x: A tensor or variable.

  Returns:
      A tensor.
  """
  return math_ops.softplus(x)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def softsign(x):
  # [FUNCTION DOC]
  # Softsign of a tensor. Input x. Returns a tensor.
  """Softsign of a tensor.

  Args:
      x: A tensor or variable.

  Returns:
      A tensor.
  """
  return nn.softsign(x)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def categorical_crossentropy(target, output, from_logits=False, axis=-1):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says a value error is raised for an invalid axis; this function has no
  # explicit axis validation (errors would come from the underlying ops).
  # Categorical crossentropy between target and output tensors. Inputs: target (same shape as output),
  # output (probabilities, or logits when from_logits is true), from_logits (boolean), axis (channel
  # axis, default last). Converts inputs to tensors and checks shape compatibility. If the output carries
  # cached logits (produced by a softmax or sigmoid activation) those are used and a warning is issued
  # if from_logits was true. If using logits, or if the output comes straight from a softmax op in graph
  # mode, uses the fused softmax crossentropy on logits. Otherwise renormalizes the probabilities to sum
  # to one, clips them by epsilon and computes the negative sum of target times log. Returns the loss
  # tensor (reduced over the axis). Side effects: may emit a warning.
  """Categorical crossentropy between an output tensor and a target tensor.

  Args:
      target: A tensor of the same shape as `output`.
      output: A tensor resulting from a softmax
          (unless `from_logits` is True, in which
          case `output` is expected to be the logits).
      from_logits: Boolean, whether `output` is the
          result of a softmax, or is a tensor of logits.
      axis: Int specifying the channels axis. `axis=-1` corresponds to data
          format `channels_last`, and `axis=1` corresponds to data format
          `channels_first`.

  Returns:
      Output tensor.

  Raises:
      ValueError: if `axis` is neither -1 nor one of the axes of `output`.

  Example:

  >>> a = tf.constant([1., 0., 0., 0., 1., 0., 0., 0., 1.], shape=[3,3])
  >>> print(a)
  tf.Tensor(
    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]], shape=(3, 3), dtype=float32)
  >>> b = tf.constant([.9, .05, .05, .05, .89, .06, .05, .01, .94], shape=[3,3])
  >>> print(b)
  tf.Tensor(
    [[0.9  0.05 0.05]
     [0.05 0.89 0.06]
     [0.05 0.01 0.94]], shape=(3, 3), dtype=float32)
  >>> loss = tf.keras.backend.categorical_crossentropy(a, b)
  >>> print(np.around(loss, 5))
  [0.10536 0.11653 0.06188]
  >>> loss = tf.keras.backend.categorical_crossentropy(a, a)
  >>> print(np.around(loss, 5))
  [0. 0. 0.]

  """
  target = tensor_conversion.convert_to_tensor_v2_with_dispatch(target)
  output = tensor_conversion.convert_to_tensor_v2_with_dispatch(output)
  target.shape.assert_is_compatible_with(output.shape)

  # Use logits whenever they are available. `softmax` and `sigmoid`
  # activations cache logits on the `output` Tensor.
  if hasattr(output, '_keras_logits'):
    output = output._keras_logits  # pylint: disable=protected-access
    if from_logits:
      warnings.warn(
          '"`categorical_crossentropy` received `from_logits=True`, but '
          'the `output` argument was produced by a sigmoid or softmax '
          'activation and thus does not represent logits. Was this intended?"')
    from_logits = True

  if from_logits:
    return nn.softmax_cross_entropy_with_logits_v2(
        labels=target, logits=output, axis=axis)

  if (not isinstance(output, (ops.EagerTensor, variables_module.Variable)) and
      output.op.type == 'Softmax') and not hasattr(output, '_keras_history'):
    # When softmax activation function is used for output operation, we
    # use logits from the softmax function directly to compute loss in order
    # to prevent collapsing zero when training.
    # See b/117284466
    assert len(output.op.inputs) == 1
    output = output.op.inputs[0]
    return nn.softmax_cross_entropy_with_logits_v2(
        labels=target, logits=output, axis=axis)

  # scale preds so that the class probas of each sample sum to 1
  output = output / math_ops.reduce_sum(output, axis, True)
  # Compute cross entropy from probabilities.
  epsilon_ = _constant_to_tensor(epsilon(), output.dtype.base_dtype)
  output = clip_ops.clip_by_value(output, epsilon_, 1. - epsilon_)
  return -math_ops.reduce_sum(target * math_ops.log(output), axis)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def sparse_categorical_crossentropy(target, output, from_logits=False, axis=-1):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says a value error is raised when the axis is not minus one or an
  # existing axis; the only value error here is for a non-last axis when the output rank is unknown.
  # Categorical crossentropy with integer targets. Inputs: target (integer tensor), output
  # (probabilities or logits), from_logits (boolean), axis (default last). Prefers cached logits (with a
  # warning when from_logits was true), uses the input of a softmax op when found, or clips and takes the
  # log of probabilities. Moves the class axis to the end by transposing when needed, casts targets to
  # 64-bit integers, flattens targets and outputs when the target rank is not one less than the output
  # rank, computes sparse softmax crossentropy (inside the Keras graph for symbolic tensors), and
  # reshapes the result back for outputs of rank three or more. Returns the loss tensor.
  """Categorical crossentropy with integer targets.

  Args:
      target: An integer tensor.
      output: A tensor resulting from a softmax
          (unless `from_logits` is True, in which
          case `output` is expected to be the logits).
      from_logits: Boolean, whether `output` is the
          result of a softmax, or is a tensor of logits.
      axis: Int specifying the channels axis. `axis=-1` corresponds to data
          format `channels_last`, and `axis=1` corresponds to data format
          `channels_first`.

  Returns:
      Output tensor.

  Raises:
      ValueError: if `axis` is neither -1 nor one of the axes of `output`.
  """
  target = tensor_conversion.convert_to_tensor_v2_with_dispatch(target)
  output = tensor_conversion.convert_to_tensor_v2_with_dispatch(output)

  # Use logits whenever they are available. `softmax` and `sigmoid`
  # activations cache logits on the `output` Tensor.
  if hasattr(output, '_keras_logits'):
    output = output._keras_logits  # pylint: disable=protected-access
    if from_logits:
      warnings.warn(
          '"`sparse_categorical_crossentropy` received `from_logits=True`, but '
          'the `output` argument was produced by a sigmoid or softmax '
          'activation and thus does not represent logits. Was this intended?"')
    from_logits = True
  elif (not from_logits and
        not isinstance(output, (ops.EagerTensor, variables_module.Variable)) and
        output.op.type == 'Softmax') and not hasattr(output, '_keras_history'):
    # When softmax activation function is used for output operation, we
    # use logits from the softmax function directly to compute loss in order
    # to prevent collapsing zero when training.
    # See b/117284466
    assert len(output.op.inputs) == 1
    output = output.op.inputs[0]
    from_logits = True
  elif not from_logits:
    epsilon_ = _constant_to_tensor(epsilon(), output.dtype.base_dtype)
    output = clip_ops.clip_by_value(output, epsilon_, 1 - epsilon_)
    output = math_ops.log(output)

  if isinstance(output.shape, (tuple, list)):
    output_rank = len(output.shape)
  else:
    output_rank = output.shape.ndims
  if output_rank is not None:
    axis %= output_rank
    if axis != output_rank - 1:
      permutation = list(
          itertools.chain(range(axis), range(axis + 1, output_rank), [axis]))
      output = array_ops.transpose(output, perm=permutation)
  elif axis != -1:
    raise ValueError(
        'Cannot compute sparse categorical crossentropy with `axis={}` on an '
        'output tensor with unknown rank'.format(axis))

  target = cast(target, 'int64')

  # Try to adjust the shape so that rank of labels = rank of logits - 1.
  output_shape = array_ops.shape_v2(output)
  target_rank = target.shape.ndims

  update_shape = (
      target_rank is not None and output_rank is not None and
      target_rank != output_rank - 1)
  if update_shape:
    target = flatten(target)
    output = array_ops.reshape(output, [-1, output_shape[-1]])

  if py_any(_is_symbolic_tensor(v) for v in [target, output]):
    with get_graph().as_default():
      res = nn.sparse_softmax_cross_entropy_with_logits_v2(
          labels=target, logits=output)
  else:
    res = nn.sparse_softmax_cross_entropy_with_logits_v2(
        labels=target, logits=output)

  if update_shape and output_rank >= 3:
    # If our output includes timesteps or spatial dimensions we need to reshape
    return array_ops.reshape(res, output_shape[:-1])
  else:
    return res


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def binary_crossentropy(target, output, from_logits=False):
  # [FUNCTION DOC]
  # KNOWN ISSUE: in the probability path the output is clipped by epsilon and then epsilon is added again
  # inside both log terms, so epsilon is applied twice; this slightly differs from the plain formula.
  # Binary crossentropy between target and output. Inputs: target (same shape as output), output (tensor),
  # from_logits (boolean; if the output carries cached logits those are used with a warning when
  # from_logits was true). With logits, or output coming from a sigmoid op in graph mode, uses the
  # sigmoid crossentropy op on logits. Otherwise computes the negative of target times log plus
  # (one minus target) times log of one minus output. Returns the element-wise loss tensor.
  """Binary crossentropy between an output tensor and a target tensor.

  Args:
      target: A tensor with the same shape as `output`.
      output: A tensor.
      from_logits: Whether `output` is expected to be a logits tensor.
          By default, we consider that `output`
          encodes a probability distribution.

  Returns:
      A tensor.
  """
  target = tensor_conversion.convert_to_tensor_v2_with_dispatch(target)
  output = tensor_conversion.convert_to_tensor_v2_with_dispatch(output)

  # Use logits whenever they are available. `softmax` and `sigmoid`
  # activations cache logits on the `output` Tensor.
  if hasattr(output, '_keras_logits'):
    output = output._keras_logits  # pylint: disable=protected-access
    if from_logits:
      warnings.warn(
          '"`binary_crossentropy` received `from_logits=True`, but the `output`'
          ' argument was produced by a sigmoid or softmax activation and thus '
          'does not represent logits. Was this intended?"')
    from_logits = True

  if from_logits:
    return nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)

  if (not isinstance(output, (ops.EagerTensor, variables_module.Variable)) and
      output.op.type == 'Sigmoid') and not hasattr(output, '_keras_history'):
    # When sigmoid activation function is used for output operation, we
    # use logits from the sigmoid function directly to compute loss in order
    # to prevent collapsing zero when training.
    assert len(output.op.inputs) == 1
    output = output.op.inputs[0]
    return nn.sigmoid_cross_entropy_with_logits(labels=target, logits=output)

  epsilon_ = _constant_to_tensor(epsilon(), output.dtype.base_dtype)
  output = clip_ops.clip_by_value(output, epsilon_, 1. - epsilon_)

  # Compute cross entropy from probabilities.
  bce = target * math_ops.log(output + epsilon())
  bce += (1 - target) * math_ops.log(1 - output + epsilon())
  return -bce


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def sigmoid(x):
  # [FUNCTION DOC]
  # Element-wise sigmoid. Input x. Returns a tensor.
  """Element-wise sigmoid.

  Args:
      x: A tensor or variable.

  Returns:
      A tensor.
  """
  return nn.sigmoid(x)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def hard_sigmoid(x):
  # [FUNCTION DOC]
  # Piecewise linear approximation of the sigmoid: slope 0.2 and offset 0.5, clipped into the range zero
  # to one. Input x. Returns a tensor.
  """Segment-wise linear approximation of sigmoid.

  Faster than sigmoid.
  Returns `0.` if `x < -2.5`, `1.` if `x > 2.5`.
  In `-2.5 <= x <= 2.5`, returns `0.2 * x + 0.5`.

  Args:
      x: A tensor or variable.

  Returns:
      A tensor.
  """
  point_two = _constant_to_tensor(0.2, x.dtype.base_dtype)
  point_five = _constant_to_tensor(0.5, x.dtype.base_dtype)
  x = math_ops.multiply(x, point_two)
  x = math_ops.add(x, point_five)
  x = clip_ops.clip_by_value(x, 0., 1.)
  return x


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def tanh(x):
  # [FUNCTION DOC]
  # Element-wise hyperbolic tangent. Input x. Returns a tensor.
  """Element-wise tanh.

  Args:
      x: A tensor or variable.

  Returns:
      A tensor.
  """
  return nn.tanh(x)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def dropout(x, level, noise_shape=None, seed=None):
  # [FUNCTION DOC]
  # Randomly sets entries to zero while scaling the rest. Inputs: x, level (fraction of entries to drop),
  # noise_shape (optional shape of the random keep flags, broadcastable to x), seed (optional; drawn
  # from numpy randomness when absent). Returns the tensor. Side effect: consumes numpy randomness.
  """Sets entries in `x` to zero at random, while scaling the entire tensor.

  Args:
      x: tensor
      level: fraction of the entries in the tensor
          that will be set to 0.
      noise_shape: shape for randomly generated keep/drop flags,
          must be broadcastable to the shape of `x`
      seed: random seed to ensure determinism.

  Returns:
      A tensor.
  """
  if seed is None:
    seed = np.random.randint(10e6)
  return nn.dropout_v2(x, rate=level, noise_shape=noise_shape, seed=seed)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def l2_normalize(x, axis=None):
  # [FUNCTION DOC]
  # Normalizes a tensor with respect to its L2 norm along an axis. Inputs: x and axis (optional).
  # Returns the normalized tensor.
  """Normalizes a tensor wrt the L2 norm alongside the specified axis.

  Args:
      x: Tensor or variable.
      axis: axis along which to perform normalization.

  Returns:
      A tensor.
  """
  return nn.l2_normalize(x, axis=axis)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def in_top_k(predictions, targets, k):
  # [FUNCTION DOC]
  # Checks whether targets are within the top k predictions. Inputs: predictions (batch by classes
  # float tensor), targets (integer vector of batch size), k (integer). Returns a boolean vector.
  """Returns whether the `targets` are in the top `k` `predictions`.

  Args:
      predictions: A tensor of shape `(batch_size, classes)` and type `float32`.
      targets: A 1D tensor of length `batch_size` and type `int32` or `int64`.
      k: An `int`, number of top elements to consider.

  Returns:
      A 1D tensor of length `batch_size` and type `bool`.
      `output[i]` is `True` if `predictions[i, targets[i]]` is within top-`k`
      values of `predictions[i]`.
  """
  return nn.in_top_k(predictions, targets, k)


# CONVOLUTIONS


def _preprocess_conv1d_input(x, data_format):
  # [FUNCTION DOC]
  # HELPER: prepares the input of one-dimensional convolution ops.
  # KNOWN ISSUE: the docstring says it transposes and casts, but no cast is done, only the transpose.
  # Inputs: x and data_format (channels_last or channels_first). Chooses the native format NWC or, if
  # channels-first and NCHW ops are supported, NCW; otherwise transposes channels-first data to NWC.
  # Returns a pair: the (possibly transposed) tensor and the TensorFlow format string.
  """Transpose and cast the input before the conv1d.

  Args:
      x: input tensor.
      data_format: string, `"channels_last"` or `"channels_first"`.

  Returns:
      A tensor.
  """
  tf_data_format = 'NWC'  # to pass TF Conv2dNative operations
  if data_format == 'channels_first':
    if not _has_nchw_support():
      x = array_ops.transpose(x, (0, 2, 1))  # NCW -> NWC
    else:
      tf_data_format = 'NCW'
  return x, tf_data_format


def _preprocess_conv2d_input(x, data_format, force_transpose=False):
  # [FUNCTION DOC]
  # HELPER: prepares the input of two-dimensional convolution, pooling and related ops.
  # KNOWN ISSUE: the docstring says it transposes and casts, but no cast is done, only the transpose.
  # Inputs: x, data_format, force_transpose (if true channels-first data is always transposed to
  # channels-last; otherwise only when NCHW is unsupported). Returns a pair of the tensor and the
  # TensorFlow format string (NHWC or NCHW).
  """Transpose and cast the input before the conv2d.

  Args:
      x: input tensor.
      data_format: string, `"channels_last"` or `"channels_first"`.
      force_transpose: Boolean. If True, the input will always be transposed
          from NCHW to NHWC if `data_format` is `"channels_first"`.
          If False, the transposition only occurs on CPU (GPU ops are
          assumed to support NCHW).

  Returns:
      A tensor.
  """
  tf_data_format = 'NHWC'
  if data_format == 'channels_first':
    if not _has_nchw_support() or force_transpose:
      x = array_ops.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
    else:
      tf_data_format = 'NCHW'
  return x, tf_data_format


def _preprocess_conv3d_input(x, data_format):
  # [FUNCTION DOC]
  # HELPER: prepares the input of three-dimensional convolution and pooling ops.
  # KNOWN ISSUE: the docstring says it transposes and casts, but no cast is done, only the transpose.
  # Inputs: x and data_format. Uses NDHWC, or NCDHW when channels-first is requested and supported,
  # otherwise transposes channels-first input to NDHWC. Returns the tensor and the format string.
  """Transpose and cast the input before the conv3d.

  Args:
      x: input tensor.
      data_format: string, `"channels_last"` or `"channels_first"`.

  Returns:
      A tensor.
  """
  tf_data_format = 'NDHWC'
  if data_format == 'channels_first':
    if not _has_nchw_support():
      x = array_ops.transpose(x, (0, 2, 3, 4, 1))
    else:
      tf_data_format = 'NCDHW'
  return x, tf_data_format


def _preprocess_padding(padding):
  # [FUNCTION DOC]
  # HELPER: converts Keras padding names to TensorFlow ones. Input padding: string same or valid.
  # Returns the upper-case form; any other value raises a value error.
  """Convert keras' padding to TensorFlow's padding.

  Args:
      padding: string, one of 'same' , 'valid'

  Returns:
      a string, one of 'SAME', 'VALID'.

  Raises:
      ValueError: if invalid `padding'`
  """
  if padding == 'same':
    padding = 'SAME'
  elif padding == 'valid':
    padding = 'VALID'
  else:
    raise ValueError('Invalid padding: ' + str(padding))
  return padding


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def conv1d(x,
           kernel,
           strides=1,
           padding='valid',
           data_format=None,
           dilation_rate=1):
  # [FUNCTION DOC]
  # One-dimensional convolution. Inputs: x, kernel, strides (integer), padding (same, valid or causal),
  # data_format (defaults to the configured format; invalid raises a value error), dilation_rate
  # (integer). Causal padding is implemented by padding the left of the time axis by the dilated kernel
  # extent and then using valid padding. Transposes as needed for the device, runs the generic
  # convolution op and transposes back to channels-first when necessary. Returns the output tensor.
  """1D convolution.

  Args:
      x: Tensor or variable.
      kernel: kernel tensor.
      strides: stride integer.
      padding: string, `"same"`, `"causal"` or `"valid"`.
      data_format: string, one of "channels_last", "channels_first".
      dilation_rate: integer dilate rate.

  Returns:
      A tensor, result of 1D convolution.

  Raises:
      ValueError: if `data_format` is neither `channels_last` or
      `channels_first`.
  """
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))

  kernel_shape = kernel.shape.as_list()
  if padding == 'causal':
    # causal (dilated) convolution:
    left_pad = dilation_rate * (kernel_shape[0] - 1)
    x = temporal_padding(x, (left_pad, 0))
    padding = 'valid'
  padding = _preprocess_padding(padding)

  x, tf_data_format = _preprocess_conv1d_input(x, data_format)
  x = nn.convolution(
      input=x,
      filter=kernel,
      dilation_rate=dilation_rate,
      strides=strides,
      padding=padding,
      data_format=tf_data_format)
  if data_format == 'channels_first' and tf_data_format == 'NWC':
    x = array_ops.transpose(x, (0, 2, 1))  # NWC -> NCW
  return x


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def conv2d(x,
           kernel,
           strides=(1, 1),
           padding='valid',
           data_format=None,
           dilation_rate=(1, 1)):
  # [FUNCTION DOC]
  # Two-dimensional convolution. Inputs: x, kernel, strides (pair), padding (same or valid),
  # data_format (default configured; invalid raises a value error), dilation_rate (pair). Handles the
  # channels-first transposition and returns the output tensor in the requested format.
  """2D convolution.

  Args:
      x: Tensor or variable.
      kernel: kernel tensor.
      strides: strides tuple.
      padding: string, `"same"` or `"valid"`.
      data_format: `"channels_last"` or `"channels_first"`.
      dilation_rate: tuple of 2 integers.

  Returns:
      A tensor, result of 2D convolution.

  Raises:
      ValueError: if `data_format` is neither `channels_last` or
      `channels_first`.
  """
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))

  x, tf_data_format = _preprocess_conv2d_input(x, data_format)
  padding = _preprocess_padding(padding)
  x = nn.convolution(
      input=x,
      filter=kernel,
      dilation_rate=dilation_rate,
      strides=strides,
      padding=padding,
      data_format=tf_data_format)
  if data_format == 'channels_first' and tf_data_format == 'NHWC':
    x = array_ops.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
  return x


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def conv2d_transpose(x,
                     kernel,
                     output_shape,
                     strides=(1, 1),
                     padding='valid',
                     data_format=None,
                     dilation_rate=(1, 1)):
  # [FUNCTION DOC]
  # Two-dimensional transposed convolution. Inputs: x, kernel, output_shape (integer vector or tuple;
  # a null batch entry is replaced by the dynamic batch size), strides (tuple), padding, data_format
  # (invalid raises a value error), dilation_rate (pair). Forces transposition for dilated
  # channels-first input because the dilated version supports only NHWC. Uses the plain transposed
  # convolution for unit dilation, otherwise the atrous version (asserting both dilation rates are
  # equal). Returns the output tensor in the requested format.
  """2D deconvolution (i.e.

  transposed convolution).

  Args:
      x: Tensor or variable.
      kernel: kernel tensor.
      output_shape: 1D int tensor for the output shape.
      strides: strides tuple.
      padding: string, `"same"` or `"valid"`.
      data_format: string, `"channels_last"` or `"channels_first"`.
      dilation_rate: Tuple of 2 integers.

  Returns:
      A tensor, result of transposed 2D convolution.

  Raises:
      ValueError: if `data_format` is neither `channels_last` or
      `channels_first`.
  """
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))

  # `atrous_conv2d_transpose` only supports NHWC format, even on GPU.
  if data_format == 'channels_first' and dilation_rate != (1, 1):
    force_transpose = True
  else:
    force_transpose = False

  x, tf_data_format = _preprocess_conv2d_input(x, data_format, force_transpose)

  if data_format == 'channels_first' and tf_data_format == 'NHWC':
    output_shape = (output_shape[0], output_shape[2], output_shape[3],
                    output_shape[1])
  if output_shape[0] is None:
    output_shape = (shape(x)[0],) + tuple(output_shape[1:])

  if isinstance(output_shape, (tuple, list)):
    output_shape = array_ops_stack.stack(list(output_shape))

  padding = _preprocess_padding(padding)
  if tf_data_format == 'NHWC':
    strides = (1,) + strides + (1,)
  else:
    strides = (1, 1) + strides

  if dilation_rate == (1, 1):
    x = nn.conv2d_transpose(x, kernel, output_shape, strides,
                            padding=padding,
                            data_format=tf_data_format)
  else:
    assert dilation_rate[0] == dilation_rate[1]
    x = nn.atrous_conv2d_transpose(
        x,
        kernel,
        output_shape,
        rate=dilation_rate[0],
        padding=padding)
  if data_format == 'channels_first' and tf_data_format == 'NHWC':
    x = array_ops.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
  return x


def separable_conv1d(x,
                     depthwise_kernel,
                     pointwise_kernel,
                     strides=1,
                     padding='valid',
                     data_format=None,
                     dilation_rate=1):
  # [FUNCTION DOC]
  # One-dimensional separable convolution (depthwise followed by pointwise). Inputs: x,
  # depthwise_kernel, pointwise_kernel, strides (integer or tuple), padding, data_format (invalid
  # raises a value error), dilation_rate (integer or tuple). Implemented by adding a spatial axis,
  # running the two-dimensional separable op and removing the axis. Returns the output tensor. Note it
  # has no API dispatch decorator unlike similar functions.
  """1D convolution with separable filters.

  Args:
      x: input tensor
      depthwise_kernel: convolution kernel for the depthwise convolution.
      pointwise_kernel: kernel for the 1x1 convolution.
      strides: stride integer.
      padding: string, `"same"` or `"valid"`.
      data_format: string, `"channels_last"` or `"channels_first"`.
      dilation_rate: integer dilation rate.

  Returns:
      Output tensor.

  Raises:
      ValueError: if `data_format` is neither `channels_last` or
      `channels_first`.
  """
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))

  if isinstance(strides, int):
    strides = (strides,)
  if isinstance(dilation_rate, int):
    dilation_rate = (dilation_rate,)

  x, tf_data_format = _preprocess_conv1d_input(x, data_format)
  padding = _preprocess_padding(padding)
  if not isinstance(strides, tuple):
    strides = tuple(strides)
  if tf_data_format == 'NWC':
    spatial_start_dim = 1
    strides = (1,) + strides * 2 + (1,)
  else:
    spatial_start_dim = 2
    strides = (1, 1) + strides * 2
  x = array_ops.expand_dims(x, spatial_start_dim)
  depthwise_kernel = array_ops.expand_dims(depthwise_kernel, 0)
  pointwise_kernel = array_ops.expand_dims(pointwise_kernel, 0)
  dilation_rate = (1,) + dilation_rate

  x = nn.separable_conv2d(
      x,
      depthwise_kernel,
      pointwise_kernel,
      strides=strides,
      padding=padding,
      rate=dilation_rate,
      data_format=tf_data_format)

  x = array_ops.squeeze(x, [spatial_start_dim])

  if data_format == 'channels_first' and tf_data_format == 'NWC':
    x = array_ops.transpose(x, (0, 2, 1))  # NWC -> NCW

  return x


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def separable_conv2d(x,
                     depthwise_kernel,
                     pointwise_kernel,
                     strides=(1, 1),
                     padding='valid',
                     data_format=None,
                     dilation_rate=(1, 1)):
  # [FUNCTION DOC]
  # Two-dimensional separable convolution. Inputs: x, depthwise_kernel, pointwise_kernel, strides
  # (tuple of two; otherwise a value error), padding, data_format (invalid raises a value error),
  # dilation_rate (pair). Returns the output tensor in the requested format.
  """2D convolution with separable filters.

  Args:
      x: input tensor
      depthwise_kernel: convolution kernel for the depthwise convolution.
      pointwise_kernel: kernel for the 1x1 convolution.
      strides: strides tuple (length 2).
      padding: string, `"same"` or `"valid"`.
      data_format: string, `"channels_last"` or `"channels_first"`.
      dilation_rate: tuple of integers,
          dilation rates for the separable convolution.

  Returns:
      Output tensor.

  Raises:
      ValueError: if `data_format` is neither `channels_last` or
      `channels_first`.
      ValueError: if `strides` is not a tuple of 2 integers.
  """
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))
  if len(strides) != 2:
    raise ValueError('`strides` must be a tuple of 2 integers.')

  x, tf_data_format = _preprocess_conv2d_input(x, data_format)
  padding = _preprocess_padding(padding)
  if not isinstance(strides, tuple):
    strides = tuple(strides)
  if tf_data_format == 'NHWC':
    strides = (1,) + strides + (1,)
  else:
    strides = (1, 1) + strides

  x = nn.separable_conv2d(
      x,
      depthwise_kernel,
      pointwise_kernel,
      strides=strides,
      padding=padding,
      rate=dilation_rate,
      data_format=tf_data_format)
  if data_format == 'channels_first' and tf_data_format == 'NHWC':
    x = array_ops.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
  return x


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def depthwise_conv2d(x,
                     depthwise_kernel,
                     strides=(1, 1),
                     padding='valid',
                     data_format=None,
                     dilation_rate=(1, 1)):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring is copied from the separable convolution and mentions separable filters;
  # this function applies only a depthwise convolution (there is no pointwise kernel) and does not check
  # the length of strides.
  # Two-dimensional depthwise convolution. Inputs: x, depthwise_kernel, strides (tuple), padding,
  # data_format (invalid raises a value error), dilation_rate (pair). Returns the output tensor in the
  # requested format.
  """2D convolution with separable filters.

  Args:
      x: input tensor
      depthwise_kernel: convolution kernel for the depthwise convolution.
      strides: strides tuple (length 2).
      padding: string, `"same"` or `"valid"`.
      data_format: string, `"channels_last"` or `"channels_first"`.
      dilation_rate: tuple of integers,
          dilation rates for the separable convolution.

  Returns:
      Output tensor.

  Raises:
      ValueError: if `data_format` is neither `channels_last` or
      `channels_first`.
  """
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))

  x, tf_data_format = _preprocess_conv2d_input(x, data_format)
  padding = _preprocess_padding(padding)
  if tf_data_format == 'NHWC':
    strides = (1,) + strides + (1,)
  else:
    strides = (1, 1) + strides

  x = nn.depthwise_conv2d(
      x,
      depthwise_kernel,
      strides=strides,
      padding=padding,
      rate=dilation_rate,
      data_format=tf_data_format)
  if data_format == 'channels_first' and tf_data_format == 'NHWC':
    x = array_ops.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
  return x


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def conv3d(x,
           kernel,
           strides=(1, 1, 1),
           padding='valid',
           data_format=None,
           dilation_rate=(1, 1, 1)):
  # [FUNCTION DOC]
  # Three-dimensional convolution. Inputs: x, kernel, strides (triple), padding, data_format (invalid
  # raises a value error), dilation_rate (triple). Returns the output in the requested format.
  """3D convolution.

  Args:
      x: Tensor or variable.
      kernel: kernel tensor.
      strides: strides tuple.
      padding: string, `"same"` or `"valid"`.
      data_format: string, `"channels_last"` or `"channels_first"`.
      dilation_rate: tuple of 3 integers.

  Returns:
      A tensor, result of 3D convolution.

  Raises:
      ValueError: if `data_format` is neither `channels_last` or
      `channels_first`.
  """
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))

  x, tf_data_format = _preprocess_conv3d_input(x, data_format)
  padding = _preprocess_padding(padding)
  x = nn.convolution(
      input=x,
      filter=kernel,
      dilation_rate=dilation_rate,
      strides=strides,
      padding=padding,
      data_format=tf_data_format)
  if data_format == 'channels_first' and tf_data_format == 'NDHWC':
    x = array_ops.transpose(x, (0, 4, 1, 2, 3))
  return x


def conv3d_transpose(x,
                     kernel,
                     output_shape,
                     strides=(1, 1, 1),
                     padding='valid',
                     data_format=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: a tuple or list output shape is stacked into a tensor at the start, so the later check
  # for a null batch entry in the output shape cannot behave as intended for such input (a null entry
  # would already have failed when stacking). The function also lacks the API dispatch decorators of its
  # neighbors.
  # Three-dimensional transposed convolution. Inputs: x, kernel, output_shape (vector or tuple),
  # strides (tuple), padding, data_format (invalid raises a value error). Handles channels-first
  # transposition and returns the output tensor in the requested format.
  """3D deconvolution (i.e.

  transposed convolution).

  Args:
      x: input tensor.
      kernel: kernel tensor.
      output_shape: 1D int tensor for the output shape.
      strides: strides tuple.
      padding: string, "same" or "valid".
      data_format: string, `"channels_last"` or `"channels_first"`.

  Returns:
      A tensor, result of transposed 3D convolution.

  Raises:
      ValueError: if `data_format` is neither `channels_last` or
      `channels_first`.
  """
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))
  if isinstance(output_shape, (tuple, list)):
    output_shape = array_ops_stack.stack(output_shape)

  x, tf_data_format = _preprocess_conv3d_input(x, data_format)

  if data_format == 'channels_first' and tf_data_format == 'NDHWC':
    output_shape = (output_shape[0], output_shape[2], output_shape[3],
                    output_shape[4], output_shape[1])
  if output_shape[0] is None:
    output_shape = (array_ops.shape(x)[0],) + tuple(output_shape[1:])
    output_shape = array_ops_stack.stack(list(output_shape))

  padding = _preprocess_padding(padding)
  if tf_data_format == 'NDHWC':
    strides = (1,) + strides + (1,)
  else:
    strides = (1, 1) + strides

  x = nn.conv3d_transpose(
      x,
      kernel,
      output_shape,
      strides,
      padding=padding,
      data_format=tf_data_format)
  if data_format == 'channels_first' and tf_data_format == 'NDHWC':
    x = array_ops.transpose(x, (0, 4, 1, 2, 3))
  return x


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def pool2d(x,
           pool_size,
           strides=(1, 1),
           padding='valid',
           data_format=None,
           pool_mode='max'):
  # [FUNCTION DOC]
  # Two-dimensional pooling. Inputs: x, pool_size (pair), strides (pair), padding, data_format, pool_mode
  # (max or avg). Raises value errors for an invalid data format, pool_size or strides not of length two
  # and an invalid pool mode. Returns the pooled tensor in the requested format.
  """2D Pooling.

  Args:
      x: Tensor or variable.
      pool_size: tuple of 2 integers.
      strides: tuple of 2 integers.
      padding: string, `"same"` or `"valid"`.
      data_format: string, `"channels_last"` or `"channels_first"`.
      pool_mode: string, `"max"` or `"avg"`.

  Returns:
      A tensor, result of 2D pooling.

  Raises:
      ValueError: if `data_format` is neither `"channels_last"` or
      `"channels_first"`.
      ValueError: if `pool_size` is not a tuple of 2 integers.
      ValueError: if `strides` is not a tuple of 2 integers.
      ValueError: if `pool_mode` is neither `"max"` or `"avg"`.
  """
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))
  if len(pool_size) != 2:
    raise ValueError('`pool_size` must be a tuple of 2 integers.')
  if len(strides) != 2:
    raise ValueError('`strides` must be a tuple of 2 integers.')

  x, tf_data_format = _preprocess_conv2d_input(x, data_format)
  padding = _preprocess_padding(padding)
  if tf_data_format == 'NHWC':
    strides = (1,) + strides + (1,)
    pool_size = (1,) + pool_size + (1,)
  else:
    strides = (1, 1) + strides
    pool_size = (1, 1) + pool_size

  if pool_mode == 'max':
    x = nn.max_pool(
        x, pool_size, strides, padding=padding, data_format=tf_data_format)
  elif pool_mode == 'avg':
    x = nn.avg_pool(
        x, pool_size, strides, padding=padding, data_format=tf_data_format)
  else:
    raise ValueError('Invalid pooling mode: ' + str(pool_mode))

  if data_format == 'channels_first' and tf_data_format == 'NHWC':
    x = array_ops.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
  return x


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def pool3d(x,
           pool_size,
           strides=(1, 1, 1),
           padding='valid',
           data_format=None,
           pool_mode='max'):
  # [FUNCTION DOC]
  # Three-dimensional pooling. Inputs: x, pool_size (triple), strides (triple), padding, data_format,
  # pool_mode (max or avg). Raises value errors for an invalid data format or pool mode. Does not check
  # the lengths of pool_size or strides. Returns the pooled tensor.
  """3D Pooling.

  Args:
      x: Tensor or variable.
      pool_size: tuple of 3 integers.
      strides: tuple of 3 integers.
      padding: string, `"same"` or `"valid"`.
      data_format: string, `"channels_last"` or `"channels_first"`.
      pool_mode: string, `"max"` or `"avg"`.

  Returns:
      A tensor, result of 3D pooling.

  Raises:
      ValueError: if `data_format` is neither `"channels_last"` or
      `"channels_first"`.
      ValueError: if `pool_mode` is neither `"max"` or `"avg"`.
  """
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))

  x, tf_data_format = _preprocess_conv3d_input(x, data_format)
  padding = _preprocess_padding(padding)
  if tf_data_format == 'NDHWC':
    strides = (1,) + strides + (1,)
    pool_size = (1,) + pool_size + (1,)
  else:
    strides = (1, 1) + strides
    pool_size = (1, 1) + pool_size

  if pool_mode == 'max':
    x = nn.max_pool3d(
        x, pool_size, strides, padding=padding, data_format=tf_data_format)
  elif pool_mode == 'avg':
    x = nn.avg_pool3d(
        x, pool_size, strides, padding=padding, data_format=tf_data_format)
  else:
    raise ValueError('Invalid pooling mode: ' + str(pool_mode))

  if data_format == 'channels_first' and tf_data_format == 'NDHWC':
    x = array_ops.transpose(x, (0, 4, 1, 2, 3))
  return x


def local_conv(inputs,
               kernel,
               kernel_size,
               strides,
               output_shape,
               data_format=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the input is indexed with a list of slices instead of a tuple of slices, which newer
  # tensor indexing implementations may interpret differently from the intended multi-axis slicing.
  # N-dimensional convolution with unshared weights (locally connected). Inputs: inputs (batch of
  # rank n plus two), kernel (output items by feature dimension by output channels), kernel_size
  # (tuple of window sizes), strides (tuple), output_shape (tuple of spatial output sizes), data_format
  # (invalid raises a value error). For each output position takes the corresponding window, flattens
  # it, concatenates the windows, applies a batch dot with the kernel, reshapes and permutes the result
  # to the data format. Returns the output tensor.
  """Apply N-D convolution with un-shared weights.

  Args:
      inputs: (N+2)-D tensor with shape
          (batch_size, channels_in, d_in1, ..., d_inN)
          if data_format='channels_first', or
          (batch_size, d_in1, ..., d_inN, channels_in)
          if data_format='channels_last'.
      kernel: the unshared weight for N-D convolution,
          with shape (output_items, feature_dim, channels_out), where
          feature_dim = np.prod(kernel_size) * channels_in,
          output_items = np.prod(output_shape).
      kernel_size: a tuple of N integers, specifying the
          spatial dimensions of the N-D convolution window.
      strides: a tuple of N integers, specifying the strides
          of the convolution along the spatial dimensions.
      output_shape: a tuple of (d_out1, ..., d_outN) specifying the spatial
          dimensionality of the output.
      data_format: string, "channels_first" or "channels_last".

  Returns:
      An (N+2)-D tensor with shape:
      (batch_size, channels_out) + output_shape
      if data_format='channels_first', or:
      (batch_size,) + output_shape + (channels_out,)
      if data_format='channels_last'.

  Raises:
      ValueError: if `data_format` is neither
      `channels_last` nor `channels_first`.
  """
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))

  kernel_shape = int_shape(kernel)
  feature_dim = kernel_shape[1]
  channels_out = kernel_shape[-1]
  ndims = len(output_shape)
  spatial_dimensions = list(range(ndims))

  xs = []
  output_axes_ticks = [range(axis_max) for axis_max in output_shape]
  for position in itertools.product(*output_axes_ticks):
    slices = [slice(None)]

    if data_format == 'channels_first':
      slices.append(slice(None))

    slices.extend(
        slice(position[d] * strides[d], position[d] * strides[d] +
              kernel_size[d]) for d in spatial_dimensions)

    if data_format == 'channels_last':
      slices.append(slice(None))

    xs.append(reshape(inputs[slices], (1, -1, feature_dim)))

  x_aggregate = concatenate(xs, axis=0)
  output = batch_dot(x_aggregate, kernel)
  output = reshape(output, output_shape + (-1, channels_out))

  if data_format == 'channels_first':
    permutation = [ndims, ndims + 1] + spatial_dimensions
  else:
    permutation = [ndims] + spatial_dimensions + [ndims + 1]

  return permute_dimensions(output, permutation)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def local_conv1d(inputs, kernel, kernel_size, strides, data_format=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the return shapes in the docstring are swapped; channels_first yields batch, filters,
  # output length and channels_last yields batch, output length, filters.
  # One-dimensional locally connected convolution. Inputs: inputs, kernel (output length by feature
  # dimension by filters), kernel_size and strides (single-element tuples), data_format. Derives the
  # output length from the kernel and delegates to the general local convolution. Returns the tensor.
  """Apply 1D conv with un-shared weights.

  Args:
      inputs: 3D tensor with shape:
          (batch_size, steps, input_dim)
          if data_format is "channels_last" or
          (batch_size, input_dim, steps)
          if data_format is "channels_first".
      kernel: the unshared weight for convolution,
          with shape (output_length, feature_dim, filters).
      kernel_size: a tuple of a single integer,
          specifying the length of the 1D convolution window.
      strides: a tuple of a single integer,
          specifying the stride length of the convolution.
      data_format: the data format, channels_first or channels_last.

  Returns:
      A 3d tensor with shape:
      (batch_size, output_length, filters)
      if data_format='channels_first'
      or 3D tensor with shape:
      (batch_size, filters, output_length)
      if data_format='channels_last'.
  """
  output_shape = (kernel.shape[0],)
  return local_conv(inputs,
                    kernel,
                    kernel_size,
                    strides,
                    output_shape,
                    data_format)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def local_conv2d(inputs,
                 kernel,
                 kernel_size,
                 strides,
                 output_shape,
                 data_format=None):
  # [FUNCTION DOC]
  # Two-dimensional locally connected convolution. Inputs: inputs, kernel, kernel_size (pair), strides
  # (pair), output_shape (rows and columns), data_format. Delegates to the general local convolution and
  # returns the output tensor.
  """Apply 2D conv with un-shared weights.

  Args:
      inputs: 4D tensor with shape:
          (batch_size, filters, new_rows, new_cols)
          if data_format='channels_first'
          or 4D tensor with shape:
          (batch_size, new_rows, new_cols, filters)
          if data_format='channels_last'.
      kernel: the unshared weight for convolution,
          with shape (output_items, feature_dim, filters).
      kernel_size: a tuple of 2 integers, specifying the
          width and height of the 2D convolution window.
      strides: a tuple of 2 integers, specifying the strides
          of the convolution along the width and height.
      output_shape: a tuple with (output_row, output_col).
      data_format: the data format, channels_first or channels_last.

  Returns:
      A 4D tensor with shape:
      (batch_size, filters, new_rows, new_cols)
      if data_format='channels_first'
      or 4D tensor with shape:
      (batch_size, new_rows, new_cols, filters)
      if data_format='channels_last'.
  """
  return local_conv(inputs,
                    kernel,
                    kernel_size,
                    strides,
                    output_shape,
                    data_format)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def bias_add(x, bias, data_format=None):
  # [FUNCTION DOC]
  # Adds a bias to a tensor. Inputs: x, bias (vector, or tensor with rank one less than x) and
  # data_format (default configured; invalid raises a value error). Raises a value error for other bias
  # ranks. A vector bias uses the native bias addition with the matching format; a bias of rank one less
  # than x (for ranks three to five) is reshaped to broadcast and added. Returns the tensor.
  """Adds a bias vector to a tensor.

  Args:
      x: Tensor or variable.
      bias: Bias tensor to add.
      data_format: string, `"channels_last"` or `"channels_first"`.

  Returns:
      Output tensor.

  Raises:
      ValueError: In one of the two cases below:
                  1. invalid `data_format` argument.
                  2. invalid bias shape.
                     the bias should be either a vector or
                     a tensor with ndim(x) - 1 dimension
  """
  if data_format is None:
    data_format = image_data_format()
  if data_format not in {'channels_first', 'channels_last'}:
    raise ValueError('Unknown data_format: ' + str(data_format))
  bias_shape = int_shape(bias)
  if len(bias_shape) != 1 and len(bias_shape) != ndim(x) - 1:
    raise ValueError(
        'Unexpected bias dimensions %d, expect to be 1 or %d dimensions' %
        (len(bias_shape), ndim(x) - 1))

  if len(bias_shape) == 1:
    if data_format == 'channels_first':
      return nn.bias_add(x, bias, data_format='NCHW')
    return nn.bias_add(x, bias, data_format='NHWC')
  if ndim(x) in (3, 4, 5):
    if data_format == 'channels_first':
      bias_reshape_axis = (1, bias_shape[-1]) + bias_shape[:-1]
      return x + reshape(bias, bias_reshape_axis)
    return x + reshape(bias, (1,) + bias_shape)
  return nn.bias_add(x, bias)


# RANDOMNESS


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def random_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
  # [FUNCTION DOC]
  # Returns a tensor of normally distributed random values. Inputs: shape, mean (default zero), stddev
  # (default one), dtype (default floatx), seed (optional; drawn from numpy randomness when absent).
  # Side effect: consumes numpy randomness.
  """Returns a tensor with normal distribution of values.

  It is an alias to `tf.random.normal`.

  Args:
      shape: A tuple of integers, the shape of tensor to create.
      mean: A float, the mean value of the normal distribution to draw samples.
        Default to 0.0.
      stddev: A float, the standard deviation of the normal distribution
        to draw samples. Default to 1.0.
      dtype: `tf.dtypes.DType`, dtype of returned tensor. Default to use Keras
        backend dtype which is float32.
      seed: Integer, random seed. Will use a random numpy integer when not
        specified.

  Returns:
      A tensor with normal distribution of values.

  Example:

  >>> random_normal_tensor = tf.keras.backend.random_normal(shape=(2,3),
  ... mean=0.0, stddev=1.0)
  >>> random_normal_tensor
  <tf.Tensor: shape=(2, 3), dtype=float32, numpy=...,
  dtype=float32)>
  """
  if dtype is None:
    dtype = floatx()
  if seed is None:
    seed = np.random.randint(10e6)
  return random_ops.random_normal(
      shape, mean=mean, stddev=stddev, dtype=dtype, seed=seed)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
  # [FUNCTION DOC]
  # Returns a tensor of uniformly distributed random values. Inputs: shape, minval (default zero), maxval
  # (default one), dtype (default floatx), seed (optional; drawn from numpy randomness).
  """Returns a tensor with uniform distribution of values.

  Args:
      shape: A tuple of integers, the shape of tensor to create.
      minval: A float, lower boundary of the uniform distribution
          to draw samples.
      maxval: A float, upper boundary of the uniform distribution
          to draw samples.
      dtype: String, dtype of returned tensor.
      seed: Integer, random seed.

  Returns:
      A tensor.

  Example:

  >>> random_uniform_tensor = tf.keras.backend.random_uniform(shape=(2,3),
  ... minval=0.0, maxval=1.0)
  >>> random_uniform_tensor
  <tf.Tensor: shape=(2, 3), dtype=float32, numpy=...,
  dtype=float32)>
  """
  if dtype is None:
    dtype = floatx()
  if seed is None:
    seed = np.random.randint(10e6)
  return random_ops.random_uniform(
      shape, minval=minval, maxval=maxval, dtype=dtype, seed=seed)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def random_binomial(shape, p=0.0, dtype=None, seed=None):
  # [FUNCTION DOC]
  # Deprecated alias of random_bernoulli. Inputs: shape, p (probability of one), dtype, seed. Emits a
  # deprecation warning and returns the bernoulli samples.
  """Returns a tensor with random binomial distribution of values.

  DEPRECATED, use `tf.keras.backend.random_bernoulli` instead.

  The binomial distribution with parameters `n` and `p` is the probability
  distribution of the number of successful Bernoulli process. Only supports
  `n` = 1 for now.

  Args:
      shape: A tuple of integers, the shape of tensor to create.
      p: A float, `0. <= p <= 1`, probability of binomial distribution.
      dtype: String, dtype of returned tensor.
      seed: Integer, random seed.

  Returns:
      A tensor.

  Example:

  >>> random_binomial_tensor = tf.keras.backend.random_binomial(shape=(2,3),
  ... p=0.5)
  >>> random_binomial_tensor
  <tf.Tensor: shape=(2, 3), dtype=float32, numpy=...,
  dtype=float32)>
  """
  warnings.warn('`tf.keras.backend.random_binomial` is deprecated, '
                'and will be removed in a future version.'
                'Please use `tf.keras.backend.random_bernoulli` instead.')
  return random_bernoulli(shape, p, dtype, seed)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def random_bernoulli(shape, p=0.0, dtype=None, seed=None):
  # [FUNCTION DOC]
  # Returns a tensor of bernoulli samples. Inputs: shape, p (probability of one, default zero), dtype
  # (default floatx), seed (optional; drawn from numpy randomness). Draws uniform values and returns ones
  # where the draw is at most p and zeros elsewhere.
  """Returns a tensor with random bernoulli distribution of values.

  Args:
      shape: A tuple of integers, the shape of tensor to create.
      p: A float, `0. <= p <= 1`, probability of bernoulli distribution.
      dtype: String, dtype of returned tensor.
      seed: Integer, random seed.

  Returns:
      A tensor.
  """
  if dtype is None:
    dtype = floatx()
  if seed is None:
    seed = np.random.randint(10e6)
  return array_ops.where_v2(
      random_ops.random_uniform(shape, dtype=dtype, seed=seed) <= p,
      array_ops.ones(shape, dtype=dtype), array_ops.zeros(shape, dtype=dtype))


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=None, seed=None):
  # [FUNCTION DOC]
  # Returns a tensor of truncated normal samples (values more than two standard deviations from the mean
  # are re-drawn). Inputs: shape, mean, stddev, dtype (default floatx), seed (optional; drawn from numpy
  # randomness).
  """Returns a tensor with truncated random normal distribution of values.

  The generated values follow a normal distribution
  with specified mean and standard deviation,
  except that values whose magnitude is more than
  two standard deviations from the mean are dropped and re-picked.

  Args:
      shape: A tuple of integers, the shape of tensor to create.
      mean: Mean of the values.
      stddev: Standard deviation of the values.
      dtype: String, dtype of returned tensor.
      seed: Integer, random seed.

  Returns:
      A tensor.
  """
  if dtype is None:
    dtype = floatx()
  if seed is None:
    seed = np.random.randint(10e6)
  return random_ops.truncated_normal(
      shape, mean, stddev, dtype=dtype, seed=seed)


# CTC
# TensorFlow has a native implementation, but it uses sparse tensors
# and therefore requires a wrapper for Keras. The functions below convert
# dense to sparse tensors and also wraps up the beam search code that is
# in TensorFlow's CTC implementation


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def ctc_label_dense_to_sparse(labels, label_lengths):
  # [FUNCTION DOC]
  # Converts dense CTC labels to a sparse tensor. Inputs: labels (batch by maximum label count) and
  # label_lengths (per-sample lengths). Builds a boolean mask of valid label positions using a scan,
  # collects the batch and label indices of valid positions and the corresponding values, and returns a
  # sparse tensor with 64-bit indices and the dense shape of labels.
  """Converts CTC labels from dense to sparse.

  Args:
      labels: dense CTC labels.
      label_lengths: length of the labels.

  Returns:
      A sparse tensor representation of the labels.
  """
  label_shape = array_ops.shape(labels)
  num_batches_tns = array_ops_stack.stack([label_shape[0]])
  max_num_labels_tns = array_ops_stack.stack([label_shape[1]])

  def range_less_than(old_input, current_input):
    # [FUNCTION DOC]
    # HELPER: scan function used by ctc_label_dense_to_sparse. Inputs: old_input (previous accumulator,
    # only its width is used) and current_input (a label length). Returns a one-row boolean mask that is
    # true for positions smaller than that length.
    return array_ops.expand_dims(
        math_ops.range(array_ops.shape(old_input)[1]), 0) < array_ops.fill(
            max_num_labels_tns, current_input)

  init = math_ops.cast(
      array_ops.fill([1, label_shape[1]], 0), dtypes_module.bool)
  dense_mask = functional_ops.scan(
      range_less_than, label_lengths, initializer=init, parallel_iterations=1)
  dense_mask = dense_mask[:, 0, :]

  label_array = array_ops.reshape(
      array_ops.tile(math_ops.range(0, label_shape[1]), num_batches_tns),
      label_shape)
  label_ind = array_ops.boolean_mask(label_array, dense_mask)

  batch_array = array_ops.transpose(
      array_ops.reshape(
          array_ops.tile(math_ops.range(0, label_shape[0]), max_num_labels_tns),
          reverse(label_shape, 0)))
  batch_ind = array_ops.boolean_mask(batch_array, dense_mask)
  indices = array_ops.transpose(
      array_ops.reshape(concatenate([batch_ind, label_ind], axis=0), [2, -1]))

  vals_sparse = array_ops.gather_nd(labels, indices)

  return sparse_tensor.SparseTensor(
      math_ops.cast(indices, dtypes_module.int64), vals_sparse,
      math_ops.cast(label_shape, dtypes_module.int64))


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def ctc_batch_cost(y_true, y_pred, input_length, label_length):
  # [FUNCTION DOC]
  # Computes the CTC loss for each batch element. Inputs: y_true (samples by maximum label length),
  # y_pred (samples by time steps by categories, softmax outputs), input_length and label_length (each
  # samples by one). Converts the labels to sparse form, takes the log of the transposed predictions
  # plus epsilon and calls the CTC loss op. Returns a tensor of shape samples by one.
  """Runs CTC loss algorithm on each batch element.

  Args:
      y_true: tensor `(samples, max_string_length)`
          containing the truth labels.
      y_pred: tensor `(samples, time_steps, num_categories)`
          containing the prediction, or output of the softmax.
      input_length: tensor `(samples, 1)` containing the sequence length for
          each batch item in `y_pred`.
      label_length: tensor `(samples, 1)` containing the sequence length for
          each batch item in `y_true`.

  Returns:
      Tensor with shape (samples,1) containing the
          CTC loss of each element.
  """
  label_length = math_ops.cast(
      array_ops.squeeze(label_length, axis=-1), dtypes_module.int32)
  input_length = math_ops.cast(
      array_ops.squeeze(input_length, axis=-1), dtypes_module.int32)
  sparse_labels = math_ops.cast(
      ctc_label_dense_to_sparse(y_true, label_length), dtypes_module.int32)

  y_pred = math_ops.log(array_ops.transpose(y_pred, perm=[1, 0, 2]) + epsilon())

  return array_ops.expand_dims(
      ctc.ctc_loss(
          inputs=y_pred, labels=sparse_labels, sequence_length=input_length), 1)


@dispatch.add_dispatch_support
@doc_controls.do_not_generate_docs
def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says the log probability tensor has shape top paths, but the CTC decoders
  # return log probabilities with one row per sample and one column per path.
  # Decodes softmax output with CTC. Inputs: y_pred (samples by time steps by categories), input_length
  # (per-sample lengths), greedy (best path when true), beam_width (beam search width, used only if not
  # greedy), top_paths (number of paths returned when not greedy). Returns a pair of a list of dense
  # decoded sequences (padded with minus one) and the log probabilities.
  """Decodes the output of a softmax.

  Can use either greedy search (also known as best path)
  or a constrained dictionary search.

  Args:
      y_pred: tensor `(samples, time_steps, num_categories)`
          containing the prediction, or output of the softmax.
      input_length: tensor `(samples, )` containing the sequence length for
          each batch item in `y_pred`.
      greedy: perform much faster best-path search if `true`.
          This does not use a dictionary.
      beam_width: if `greedy` is `false`: a beam search decoder will be used
          with a beam of this width.
      top_paths: if `greedy` is `false`,
          how many of the most probable paths will be returned.

  Returns:
      Tuple:
          List: if `greedy` is `true`, returns a list of one element that
              contains the decoded sequence.
              If `false`, returns the `top_paths` most probable
              decoded sequences.
              Each decoded sequence has shape (samples, time_steps).
              Important: blank labels are returned as `-1`.
          Tensor `(top_paths, )` that contains
              the log probability of each decoded sequence.
  """
  input_shape = shape(y_pred)
  num_samples, num_steps = input_shape[0], input_shape[1]
  y_pred = math_ops.log(array_ops.transpose(y_pred, perm=[1, 0, 2]) + epsilon())
  input_length = math_ops.cast(input_length, dtypes_module.int32)

  if greedy:
    (decoded, log_prob) = ctc.ctc_greedy_decoder(
        inputs=y_pred, sequence_length=input_length)
  else:
    (decoded, log_prob) = ctc.ctc_beam_search_decoder(
        inputs=y_pred,
        sequence_length=input_length,
        beam_width=beam_width,
        top_paths=top_paths)
  decoded_dense = []
  for st in decoded:
    st = sparse_tensor.SparseTensor(
        st.indices, st.values, (num_samples, num_steps))
    decoded_dense.append(
        sparse_ops.sparse_tensor_to_dense(sp_input=st, default_value=-1))
  return (decoded_dense, log_prob)


# HIGH ORDER FUNCTIONS


@doc_controls.do_not_generate_docs
def map_fn(fn, elems, name=None, dtype=None):
  # [FUNCTION DOC]
  # Applies a function to every element of a tensor and collects the results. Inputs: fn (callable),
  # elems (tensor), name (optional), dtype (output type). Returns the stacked outputs.
  """Map the function fn over the elements elems and return the outputs.

  Args:
      fn: Callable that will be called upon each element in elems
      elems: tensor
      name: A string name for the map node in the graph
      dtype: Output data type.

  Returns:
      Tensor with dtype `dtype`.
  """
  return map_fn_lib.map_fn(fn, elems, name=name, dtype=dtype)


@doc_controls.do_not_generate_docs
def foldl(fn, elems, initializer=None, name=None):
  # [FUNCTION DOC]
  # Reduces elements from left to right with a combining function. Inputs: fn (takes accumulator and
  # element), elems, initializer (optional; first element used when absent), name. Returns the final
  # accumulator.
  """Reduce elems using fn to combine them from left to right.

  Args:
      fn: Callable that will be called upon each element in elems and an
          accumulator, for instance `lambda acc, x: acc + x`
      elems: tensor
      initializer: The first value used (`elems[0]` in case of None)
      name: A string name for the foldl node in the graph

  Returns:
      Tensor with same type and shape as `initializer`.
  """
  return functional_ops.foldl(fn, elems, initializer=initializer, name=name)


@doc_controls.do_not_generate_docs
def foldr(fn, elems, initializer=None, name=None):
  # [FUNCTION DOC]
  # Reduces elements from right to left with a combining function. Inputs: fn, elems, initializer
  # (optional; last element used when absent), name. Returns the final accumulator.
  """Reduce elems using fn to combine them from right to left.

  Args:
      fn: Callable that will be called upon each element in elems and an
          accumulator, for instance `lambda acc, x: acc + x`
      elems: tensor
      initializer: The first value used (`elems[-1]` in case of None)
      name: A string name for the foldr node in the graph

  Returns:
      Same type and shape as initializer
  """
  return functional_ops.foldr(fn, elems, initializer=initializer, name=name)

# Load Keras default configuration from config file if present.
# Set Keras base dir path given KERAS_HOME env variable, if applicable.
# Otherwise either ~/.keras or /tmp.
if 'KERAS_HOME' in os.environ:
  _keras_dir = os.environ.get('KERAS_HOME')
else:
  _keras_base_dir = os.path.expanduser('~')
  _keras_dir = os.path.join(_keras_base_dir, '.keras')
_config_path = os.path.expanduser(os.path.join(_keras_dir, 'keras.json'))
if os.path.exists(_config_path):
  try:
    with open(_config_path) as fh:
      _config = json.load(fh)
  except ValueError:
    _config = {}
  _floatx = _config.get('floatx', floatx())
  assert _floatx in {'float16', 'float32', 'float64'}
  _epsilon = _config.get('epsilon', epsilon())
  assert isinstance(_epsilon, float)
  _image_data_format = _config.get('image_data_format', image_data_format())
  assert _image_data_format in {'channels_last', 'channels_first'}
  set_floatx(_floatx)
  set_epsilon(_epsilon)
  set_image_data_format(_image_data_format)

# Save config file.
if not os.path.exists(_keras_dir):
  try:
    os.makedirs(_keras_dir)
  except OSError:
    # Except permission denied and potential race conditions
    # in multi-threaded environments.
    pass

if not os.path.exists(_config_path):
  _config = {
      'floatx': floatx(),
      'epsilon': epsilon(),
      'backend': 'tensorflow',
      'image_data_format': image_data_format()
  }
  try:
    with open(_config_path, 'w') as f:
      f.write(json.dumps(_config, indent=4))
  except IOError:
    # Except permission denied.
    pass


def configure_and_create_distributed_session(distribution_strategy):
  # [FUNCTION DOC]
  # Creates the global session appropriate for a distribution strategy and stores it as the Keras
  # session. Input distribution_strategy: the strategy object. Defines an inner function that builds the
  # session configuration, merges the config of an existing session (existing values win), handles TPU
  # strategies (strategy configures the config, session targets the TPU master), distribute coordinator
  # worker contexts (merge configs, use the worker master target) or other cases (strategy configures
  # config), and then registers the new session. In multi-worker mode the inner function is run through
  # the distribute coordinator, otherwise it is called directly. Returns nothing. Side effects: session
  # creation and global session replacement.
  """Configure session config and create a session with it."""

  def _create_session(distribution_strategy):
    # [FUNCTION DOC]
    # HELPER: inner function of configure_and_create_distributed_session. Input distribution_strategy:
    # the strategy. Builds the session config, merges in the config of the currently cached session if
    # any, creates a session as described for the outer function, and sets it as the Keras session.
    # Returns nothing.
    """Create the Distributed Strategy session."""
    session_config = get_default_session_config()

    # If a session already exists, merge in its config; in the case there is a
    # conflict, take values of the existing config.
    global _SESSION
    if getattr(_SESSION, 'session', None) and _SESSION.session._config:
      session_config.MergeFrom(_SESSION.session._config)

    if is_tpu_strategy(distribution_strategy):
      # TODO(priyag, yuefengz): Remove this workaround when Distribute
      # Coordinator is integrated with keras and we can create a session from
      # there.
      distribution_strategy.configure(session_config)
      master = distribution_strategy.extended._tpu_cluster_resolver.master()  # pylint: disable=protected-access
      session = session_module.Session(config=session_config, target=master)
    else:
      worker_context = dc.get_current_worker_context()
      if worker_context:
        dc_session_config = worker_context.session_config
        # Merge the default session config to the one from distribute
        # coordinator, which is fine for now since they don't have
        # conflicting configurations.
        dc_session_config.MergeFrom(session_config)
        session = session_module.Session(
            config=dc_session_config, target=worker_context.master_target)
      else:
        distribution_strategy.configure(session_config)
        session = session_module.Session(config=session_config)

    set_session(session)

  if distribution_strategy.extended._in_multi_worker_mode():
    dc.run_distribute_coordinator(
        _create_session,
        distribution_strategy)
  else:
    _create_session(distribution_strategy)


def _is_tpu_strategy_class(clz):
  # [FUNCTION DOC]
  # HELPER: used by is_tpu_strategy. Input clz: a class. Returns true if the class name starts with the
  # TPU strategy name or any base class does (recursive search through the bases).
  is_tpu_strat = lambda k: k.__name__.startswith('TPUStrategy')
  if is_tpu_strat(clz):
    return True
  return py_any(map(_is_tpu_strategy_class, clz.__bases__))


def is_tpu_strategy(strategy):
  # [FUNCTION DOC]
  # Returns whether the input object is an instance of a TPU strategy class or of a subclass. Input
  # strategy: any strategy object (checked by class name through its inheritance chain). Returns a
  # boolean.
  """Returns whether input is a TPUStrategy instance or subclass instance."""
  return _is_tpu_strategy_class(strategy.__class__)


def cast_variables_to_tensor(tensors):
  # [FUNCTION DOC]
  # Converts variables inside a nested structure to tensors by taking an identity of each; other
  # leaves are untouched. Input tensors: structure of tensors and variables. Returns the structure.

  def _cast_variables_to_tensor(tensor):
    # [FUNCTION DOC]
    # HELPER: leaf mapper used by cast_variables_to_tensor. Input tensor: a leaf. Returns an identity
    # tensor for variables and the leaf itself otherwise.
    if isinstance(tensor, variables_module.Variable):
      return array_ops.identity(tensor)
    return tensor

  return nest.map_structure(_cast_variables_to_tensor, tensors)


def _is_symbolic_tensor(x):
  # [FUNCTION DOC]
  # HELPER: used by the sparse crossentropy. Input x: any object. Returns true if it is a TensorFlow
  # tensor type that is not an eager tensor.
  return tensor_util.is_tf_type(x) and not isinstance(x, ops.EagerTensor)


def convert_inputs_if_ragged(inputs):
  # [FUNCTION DOC]
  # Converts ragged tensors in a nested structure to dense tensors. Input inputs: structure of tensors.
  # If none is ragged returns the inputs and nothing. Otherwise converts every ragged tensor to dense
  # and takes the row lengths of the first flattened input (only one mask is supported, applied to all
  # inputs; first ragged dimension, cast to 32-bit integers). Returns a pair of the converted inputs and
  # the row lengths.
  """Converts any ragged tensors to dense."""

  def _convert_ragged_input(inputs):
    # [FUNCTION DOC]
    # HELPER: leaf mapper used by convert_inputs_if_ragged. Input inputs: a leaf. Returns the dense
    # version for ragged tensors, else the leaf unchanged.
    if isinstance(inputs, ragged_tensor.RaggedTensor):
      return inputs.to_tensor()
    return inputs

  flat_inputs = nest.flatten(inputs)
  contains_ragged = py_any(
      isinstance(i, ragged_tensor.RaggedTensor) for i in flat_inputs)

  if not contains_ragged:
    return inputs, None

  inputs = nest.map_structure(_convert_ragged_input, inputs)
  # Multiple mask are not yet supported, so one mask is used on all inputs.
  # We approach this similarly when using row lengths to ignore steps.
  nested_row_lengths = math_ops.cast(flat_inputs[0].nested_row_lengths()[0],
                                     'int32')
  return inputs, nested_row_lengths


def maybe_convert_to_ragged(is_ragged_input, output, nested_row_lengths,
                            go_backwards=False):
  # [FUNCTION DOC]
  # Converts an output back to ragged form if the original input was ragged. Inputs: is_ragged_input
  # (boolean), output (dense tensor), nested_row_lengths (row lengths from the input conversion),
  # go_backwards (whether the time axis of the output was reversed). Returns the output unchanged for
  # non-ragged input; otherwise builds a ragged tensor from the dense output and the row lengths,
  # reversing along the time axis before and after when going backwards.
  """Converts any ragged input back to its initial structure."""
  if not is_ragged_input:
    return output

  if go_backwards:
    # Reverse based on the timestep dim, so that nested_row_lengths will mask
    # from the correct direction. Return the reverse ragged tensor.
    output = reverse(output, [1])
    ragged = ragged_tensor.RaggedTensor.from_tensor(output, nested_row_lengths)
    return reverse(ragged, [1])
  else:
    return ragged_tensor.RaggedTensor.from_tensor(output, nested_row_lengths)


class ContextValueCache(weakref.WeakKeyDictionary):
  # [CLASS DOC]
  # Weak-keyed dictionary that caches values per graph or eager context, similar to a default
  # dictionary. It subclasses the weak key dictionary. A missing key creates a value with the default
  # factory. Looking up the key nothing means the current context: the dummy eager key when executing
  # eagerly, else the current default graph. Lookup walks up from a function graph to its parent graphs
  # so child functions see values created in the enclosing graph. Item access and the default-setting
  # method call the factory (the latter lets keyword arguments be passed); the plain get method behaves
  # like a normal dictionary. Used for the learning phase, tracked variables and tracked optimizers
  # registries.
  """Container that caches (possibly tensor) values based on the context.

  This class is similar to defaultdict, where values may be produced by the
  default factory specified during initialization. This class also has a default
  value for the key (when key is `None`) -- the key is set to the current graph
  or eager context. The default factories for key and value are only used in
  `__getitem__` and `setdefault`. The `.get()` behavior remains the same.

  This object will return the value of the current graph or closest parent graph
  if the current graph is a function. This is to reflect the fact that if a
  tensor is created in eager/graph, child functions may capture that tensor.

  The default factory method may accept keyword arguments (unlike defaultdict,
  which only accepts callables with 0 arguments). To pass keyword arguments to
  `default_factory`, use the `setdefault` method instead of `__getitem__`.

  An example of how this class can be used in different contexts:

  ```
  cache = ContextValueCache(int)

  # Eager mode
  cache[None] += 2
  cache[None] += 4
  assert cache[None] == 6

  # Graph mode
  with tf.Graph().as_default() as g:
    cache[None] += 5
    cache[g] += 3
  assert cache[g] == 8
  ```

  Example of a default factory with arguments:

  ```
  cache = ContextValueCache(lambda x: x + 1)
  g = tf.get_default_graph()

  # Example with keyword argument.
  value = cache.setdefault(key=g, kwargs={'x': 3})
  assert cache[g] == 4
  ```
  """

  def __init__(self, default_factory):
    # [FUNCTION DOC]
    # Constructor. Input default_factory: callable that creates a new value (called without arguments
    # by item access, with keyword arguments by the default-setting method). Stores it and initializes the
    # weak dictionary. Returns nothing.
    self.default_factory = default_factory
    weakref.WeakKeyDictionary.__init__(self)

  def _key(self):
    # [FUNCTION DOC]
    # HELPER: returns the key of the current context: the dummy eager key when executing eagerly, else the
    # current default graph. No inputs.
    if context.executing_eagerly():
      return _DUMMY_EAGER_GRAPH.key
    else:
      return ops.get_default_graph()

  def _get_parent_graph(self, graph):
    # [FUNCTION DOC]
    # HELPER: used by the recursive lookup. Input graph: a function graph. Returns its outer graph, or the
    # dummy eager key when the outer graph is not a function graph and eager execution is on outside
    # functions (works around outer graphs always being regular graphs even in eager mode).
    """Returns the parent graph or dummy eager object."""
    # TODO(b/149317164): Currently FuncGraphs use ops.get_default_graph() as the
    # outer graph. This results in outer_graph always being a Graph,
    # even in eager mode (get_default_graph will create a new Graph if there
    # isn't a default graph). Because of this bug, we have to specially set the
    # key when eager execution is enabled.
    parent_graph = graph.outer_graph
    if (not isinstance(parent_graph, func_graph.FuncGraph) and
        ops.executing_eagerly_outside_functions()):
      return _DUMMY_EAGER_GRAPH.key
    return parent_graph

  def _get_recursive(self, key):
    # [FUNCTION DOC]
    # HELPER: used by item access. Input key: a graph or dummy key. Returns the stored value for the key,
    # or, for function graphs, for the nearest ancestor that has one; returns nothing if none exists.
    """Gets the value at key or the closest parent graph."""
    value = self.get(key)
    if value is not None:
      return value

    # Since FuncGraphs are able to capture tensors and variables from their
    # parent graphs, recursively search to see if there is a value stored for
    # one of the parent graphs.
    if isinstance(key, func_graph.FuncGraph):
      return self._get_recursive(self._get_parent_graph(key))
    return None

  def __getitem__(self, key):
    # [FUNCTION DOC]
    # Returns the cached value for a context, creating it when absent. Input key: nothing (use the current
    # context) or a graph. Searches the key and its parent graphs; if no value is found, stores the result
    # of the default factory under the key and returns it. Side effect: may insert an entry.
    """Gets the value at key (or current context), or sets default value.

    Args:
      key: May be `None` or `Graph`object. When `None`, the key is set to the
        current context.

    Returns:
      Either the cached or default value.
    """
    if key is None:
      key = self._key()

    value = self._get_recursive(key)
    if value is None:
      value = self[key] = self.default_factory()  # pylint:disable=not-callable
    return value

  def setdefault(self, key=None, default=None, kwargs=None):
    # [FUNCTION DOC]
    # Sets the default value when the key is absent and returns the stored value. Inputs: key (nothing for
    # the current context), default (explicit value, optional) and kwargs (mapping of keyword arguments
    # for the default factory). When no default is given and the key is absent, the factory is called with
    # those arguments to create it. Side effect: may insert an entry. Unlike item access it does not search
    # parent graphs.
    """Sets the default value if key is not in dict, and returns the value."""
    if key is None:
      key = self._key()
    kwargs = kwargs or {}

    if default is None and key not in self:
      default = self.default_factory(**kwargs)
    return weakref.WeakKeyDictionary.setdefault(self, key, default)

# This dictionary holds a mapping {graph: learning_phase}. In eager mode, a
# dummy object is used.
# A learning phase is a bool tensor used to run Keras models in
# either train mode (learning_phase == 1) or test mode (learning_phase == 0).
_GRAPH_LEARNING_PHASES = ContextValueCache(_default_learning_phase)

# This dictionary holds a mapping between a graph and variables to initialize
# in the graph.
_GRAPH_VARIABLES = ContextValueCache(object_identity.ObjectIdentityWeakSet)

# This dictionary holds a mapping between a graph and TF optimizers created in
# the graph.
_GRAPH_TF_OPTIMIZERS = ContextValueCache(object_identity.ObjectIdentityWeakSet)
