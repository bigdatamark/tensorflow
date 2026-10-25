# [MODULE DOC]
# Script language operators: lets Python callables run inside TensorFlow graphs and functions.
# Exposes the public py_function (eager execution of a Python function, differentiable once),
# numpy_function and the legacy v1 py_func (numpy based, not differentiable), plus the
# internal registry and glue they share. A global registry of weakly held callables is keyed by
# unique string tokens; the native PyFunc / EagerPyFunc kernels call back into it through a
# trampoline. Import-time side effects: creates the global registry, initializes the native
# py trampoline with it, registers the gradient for EagerPyFunc, and marks PyFunc and
# PyFuncStateless as not differentiable.
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
"""Script Language Operators."""

# pylint: disable=g-bad-name
import functools
import threading
# Used by py_util.cc to get tracebacks.
import traceback  # pylint: disable=unused-import
import weakref

import numpy as np

from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.eager import backprop
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.eager import record
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec
from tensorflow.python.lib.core import _pywrap_py_func
from tensorflow.python.ops import autograph_ops  # pylint: disable=unused-import
from tensorflow.python.ops import gen_script_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import numpy_compat
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export


# Map from EagerPyFunc token to tuple (tape, eager args, eager outputs);
# used for differentiation.
tape_cache = {}


def _maybe_copy_to_context_device(tensor, device_name):
  # [FUNCTION DOC]
  # HELPER: used by EagerFunc to place py_func outputs on the kernel's device.
  # Takes an eager tensor and a device name string. If the tensor's backing device already equals
  # the name, it is returned unchanged; otherwise a copy is made on the current context device
  # (bypassing the placer, so small int tensors may land on GPU). Returns the tensor or its copy.
  # Uses a protected tensor copy method; no other side effects.
  """Copy an EagerTensor to the current device if it's not on `device_name`."""
  in_device = tensor.backing_device
  if device_name == in_device:
    return tensor
  else:
    # Note that EagerTensor._copy bypasses the placer and copies to the context
    # device, which means e.g. int32 Tensors which would normally be forced onto
    # the CPU can instead be placed on the GPU. This is necessary so that the
    # PyFunc kernel always returns Tensors on the device it's executing on.
    return tensor._copy()  # pylint: disable=protected-access


class EagerFunc:
  # [CLASS DOC]
  # HELPER: internal glue between the EagerPyFunc kernel and user Python functions.
  # Callable wrapper that runs a user function eagerly, converts each result to a tensor of the
  # expected dtype, copies results to the op's device, and (when gradients may be needed) records
  # a gradient tape into the module-level tape cache keyed by the op token. Holds the wrapped
  # function, the list of expected output dtypes, whether it is itself a gradient function, and
  # a flag enabling graph-mode gradient support. Created by the internal py_func builder and
  # stored in the function registry.
  """A wrapper for a function owned by an EagerPyFunc."""

  def __init__(self, func, Tout, is_grad_func):
    # [FUNCTION DOC]
    # Constructor. Stores the function to wrap, the list of output dtypes (empty when the function
    # returns nothing), and whether this wrapper is the gradient of another EagerPyFunc.
    # Graph-mode gradient support starts disabled. Returns nothing; no other side effects.
    """Constructs an EagerFunc.

    Args:
      func: The function to wrap.
      Tout: A list of datatypes for the output; an empty list if the output is
        None.
      is_grad_func: Whether this EagerFunc is the gradient of another
        EagerPyFunc.
    """
    self._func = func
    self._out_dtypes = Tout
    self._is_grad_func = is_grad_func
    self._support_graph_mode_gradient = False

  def set_support_graph_mode_gradient(self):
    # [FUNCTION DOC]
    # Turns on the flag that makes later calls always record a gradient tape and stash it in the
    # tape cache, so graph-mode gradient computation can find it. Takes no inputs, returns nothing.
    """Indicates the object shall support gradient ops.

    This function is internally used by _EagerPyFuncGrad to support
    graph mode gradient of EagerFunc via tf.gradient().
    """
    self._support_graph_mode_gradient = True

  def _convert(self, value, dtype):
    # [FUNCTION DOC]
    # HELPER: converts one returned value of the wrapped function into a tensor.
    # Inputs: the value and the required dtype. Raises a runtime error if the value is a resource
    # variable. If the value is null and this is a gradient function, returns a scalar zero of the
    # given dtype (note: a scalar, not shaped like the missing gradient). Otherwise converts to a
    # tensor of that dtype and returns it.
    """Converts `value` to a tensor of type `dtype`, with error checking.

    Args:
      value: The tensor to convert.
      dtype: The desired dtype.

    Returns:
      A tensor of type `dtype`, or a zeros tensor if value is None and
      this function is in fact a gradient function.

    Raises:
      RuntimeError: if `value` is a variable.
    """

    if isinstance(value, resource_variable_ops.ResourceVariable):
      raise RuntimeError(
          "Attempting to return a variable from an eagerly executed py_func. "
          "Only numeric data structures like Tensors or NumPy arrays should "
          "be returned; to return the value of a variable, make sure to obtain "
          "the Tensor backing it by calling `.read_value()` on the variable in "
          f"question: {value}")
    if value is None and self._is_grad_func:
      # Gradient functions may legitimately return a list that contains
      # both Tensors and Python Nones. Unfortunately this breaks the
      # OpKernel, so for now we replace None objects with zeros, which is
      # mathematically correct but will prevent short-circuiting gradient
      # computations.
      #
      # TODO(akshayka): Make it possible to return a list of both Tensors and
      # Nones from an EagerPyFunc.
      return constant_op.constant(0.0, dtype=dtype)
    return ops.convert_to_tensor(value, dtype=dtype)

  def __call__(self, device, token, args):
    # [FUNCTION DOC]
    # Invoked by the registry. Inputs: device name (or null for CPU), the op token, and the list
    # of argument tensors. If gradients may be needed (graph-mode gradient support enabled or a
    # tape could be recording), runs the function under a new gradient tape that watches all
    # trainable input tensors and stores (tape, args, outputs) in the global tape cache under the
    # byte-encoded token; otherwise just runs it. Returns the converted outputs (list, single
    # tensor or null). Side effect: writes to the tape cache.
    """Calls `self._func` in eager mode, recording the tape if needed."""
    use_tape_cache = (
        self._support_graph_mode_gradient or record.could_possibly_record())

    if use_tape_cache:
      with backprop.GradientTape() as tape:
        for tensor in args:
          for t in nest.flatten(tensor):
            if backprop_util.IsTrainable(t):
              tape.watch(t)
        outputs = self._call(device, args)
      tape_cache[compat.as_bytes(token)] = (tape, args, outputs)
    else:
      outputs = self._call(device, args)

    return outputs

  def _call(self, device, args):
    # [FUNCTION DOC]
    # HELPER: executes the wrapped function eagerly for __call__.
    # Enters eager mode, calls the function with the arguments, then (under the given device,
    # null meaning the CPU) converts each result to the expected dtype and copies it to that
    # device. A list or tuple result is converted element-wise against the output dtypes; a null
    # result yields null; any other result is converted with the first output dtype. Returns
    # the resulting list, tensor or null. Errors from the user function or conversion propagate.
    """Passes `args` to `self._func`, which is executed eagerly."""
    with context.eager_mode():
      ret = self._func(*args)
      # copy the returned tensors to the PyFunc op's device if necessary.
      device_name = device
      if device_name is None:
        # "None" here means "CPU", from the nullptr convention with C++ device
        # pointers.
        device_name = "/job:localhost/replica:0/task:0/device:CPU:0"
      with ops.device(device):
        if isinstance(ret, (tuple, list)):
          outputs = [
              _maybe_copy_to_context_device(self._convert(x, dtype=dtype),
                                            device_name)
              for (x, dtype) in zip(ret, self._out_dtypes)
          ]
        elif ret is None:
          outputs = None
        else:
          outputs = _maybe_copy_to_context_device(
              self._convert(ret, dtype=self._out_dtypes[0]), device_name)
    return outputs


class FuncRegistry:
  # [CLASS DOC]
  # HELPER: registry that the native PyFunc kernels call back into.
  # Maps unique string tokens ("pyfunc_" plus a counter) to callables, holding them only weakly
  # (the graph keeps the strong reference). Thread-safe token generation via a lock. When called
  # with a token, device and arguments it dispatches to the callable: eager wrappers run
  # directly, plain callables get numpy arguments and their results are normalized to numpy
  # arrays. A single module-level instance is used.
  """A helper class to keep track of registered py functions.

  FuncRegistry keeps a map from unique tokens (string) to python
  functions, which takes numpy arrays and outputs numpy arrays.
  """

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor: creates the lock, the token counter starting at zero, and an empty mapping
    # that holds registered functions by weak reference. Takes no inputs, returns nothing.
    self._lock = threading.Lock()
    self._unique_id = 0  # GUARDED_BY(self._lock)
    # Only store weakrefs to the functions. The strong reference is stored in
    # the graph.
    self._funcs = weakref.WeakValueDictionary()

  @property
  def _ctx(self):
    # [FUNCTION DOC]
    # HELPER: property giving the native eager context handle.
    # KNOWN ISSUE: nothing in this module reads this property; it may be dead code or used only
    # by native code.
    # Intended so GPU tensors can be moved to CPU for numpy APIs. Ensures the eager context is
    # initialized (side effect) and returns its protected handle object.
    # N.B. This is needed to support calling py_func with GPU tensors,
    # which must be transferred to CPU if used in any of the NumPy APIs.
    context.ensure_initialized()
    return context.context()._handle  # pylint: disable=protected-access

  def insert(self, func):
    # [FUNCTION DOC]
    # Registers a callable under a freshly generated unique token and returns that token string.
    # Only a weak reference is stored, so the caller must keep the callable alive elsewhere.
    """Registers `func` and returns a unique token for this entry."""
    token = self._next_unique_token()
    # Store a weakref to the function
    self._funcs[token] = func
    return token

  def remove(self, token):
    # [FUNCTION DOC]
    # Removes the entry for the given token if present; silently does nothing if absent.
    # Returns nothing. Mutates the registry.
    """Removes the registered function corresponding to `token`."""
    self._funcs.pop(token, None)

  def get(self, token, default=None):
    # [FUNCTION DOC]
    # Looks up the callable for a token. Returns it, or the optional default (null if not given)
    # when the token is unknown or the weakly held function has been collected. No side effects.
    """Gets the registered function corresponding to `token`."""
    return self._funcs.get(token, default)

  @staticmethod
  def _convert(value, dtype=None):
    # [FUNCTION DOC]
    # HELPER: static method that turns a callback result into a numpy array for the kernel.
    # Inputs: any array-like value and an optional numpy dtype. Converts in C order; if that
    # produced a fixed-width byte string array from a non-array input it redoes the conversion
    # as an object array (avoiding zero padding); unicode arrays from non-array input are encoded
    # to utf-8 bytes in an object array; unicode arrays that are the input itself are cast to
    # bytes. Otherwise returns the plain conversion. No side effects.
    """Converts an arg to numpy, avoiding dangerous string and unicode dtypes.

    Numpy pads with zeros when using string and unicode dtypes if different
    components of a tensor have different lengths.  This is bad: ignoring the
    padding is wrong for text data, and removing the padding is wrong for binary
    data.  To avoid this bug, we redo the conversion using an object dtype.
    Additionally, we convert unicode strings to (byte-)strings for
    compatibility.

    Args:
      value: Value to convert to a numpy array.
      dtype: (Optional.) Desired NumPy type for the returned value.

    Returns:
      A numpy array.
    """
    result = numpy_compat.np_asarray(value, dtype=dtype, order="C")
    if result.dtype.char == "S" and result is not value:
      return numpy_compat.np_asarray(value, order="C", dtype=object)
    elif result.dtype.char == "U" and result is not value:
      value = np.vectorize(lambda x: x.encode("utf8"))(value)
      return numpy_compat.np_asarray(value, order="C", dtype=object)
    elif result.dtype.char == "U":
      return result.astype(np.bytes_)
    else:
      return result

  def __call__(self, token, device, args):
    # [FUNCTION DOC]
    # Entry point used by the native trampoline. Inputs: token, device name (used only for eager
    # wrappers) and the argument list. Raises a value error if no function is registered for the
    # token. Eager wrappers are called with device, token and args and their result returned
    # as is. Other callables are called with the arguments; a bytes result is wrapped in a list,
    # list or tuple results are converted element-wise to numpy arrays, single results converted
    # to one array. Side effects: whatever the user function does (and tape cache writes for
    # eager wrappers).
    """Calls the registered function for `token` with args.

    Args:
      token: A key into this `FuncRegistry` identifying which function to call.
      device: Name of the device on which outputs of `token`'s corresponding
        operation should be placed. Used iff the function registered for `token`
        is an EagerPyFunc.
      args: The arguments to pass to the function registered for `token`.

    Returns:
      The output of the function registered for `token`.

    Raises:
      ValueError: if no function is registered for `token`.
    """
    func = self.get(token, None)
    if func is None:
      raise ValueError(f"Could not find callback with key={token} in the "
                       "registry.")
    if isinstance(func, EagerFunc):
      # NB: Different invocations of the same py_func will share the same
      # token, and the entries they stash in the tape_cache will collide.
      # In practice, when executing a graph, this should only happen if
      # the py_func is in a while_loop whose iterations are run in parallel
      # or if the graph is being driven by concurrent session.run() calls.
      #
      # TODO(akshayka): Key the tape cache in a thread-safe way.
      return func(device, token, args)
    else:
      ret = func(*args)
      # Strings seem to lead to a memory leak here if they're not wrapped in a
      # list.
      if isinstance(ret, bytes):
        ret = [ret]
      # Ensures that we return either a single numpy array or a list of numpy
      # arrays.
      if isinstance(ret, (tuple, list)):
        return [self._convert(x) for x in ret]
      else:
        return self._convert(ret)

  def size(self):
    # [FUNCTION DOC]
    # Returns the number of entries currently in the registry (collected functions drop out).
    """Returns how many functions are currently registered."""
    return len(self._funcs)

  def _next_unique_token(self):
    # [FUNCTION DOC]
    # HELPER: produces registry tokens. Under the lock, takes and increments the counter and
    # returns a string made of the prefix "pyfunc_" and the counter value.
    """Returns a unique token."""
    with self._lock:
      uid = self._unique_id
      self._unique_id += 1
    return "pyfunc_%d" % uid


# Global registry for py functions.
_py_funcs = FuncRegistry()

_pywrap_py_func.initialize_py_trampoline(_py_funcs)


def _internal_py_func(func,
                      inp,
                      Tout,
                      stateful=None,
                      use_eager_py_func=False,
                      is_grad_func=False,
                      name=None):
  # [FUNCTION DOC]
  # HELPER: shared implementation behind py_function, numpy_function and the v1 py_func.
  # Inputs: the callable (raises a value error if not callable), the list of input tensors, Tout
  # (a dtype or type spec, or a list/tuple of them), optional stateful flag (only for the numpy
  # style op), a flag choosing the eager op over the plain op, a flag marking gradient functions,
  # and an optional op name. Wraps the callable to skip autograph conversion, converts variables
  # in the inputs to tensors, handles composite tensors by flattening, wraps in an eager wrapper
  # if requested, registers it in the global registry and keeps a strong reference on the
  # outermost graph (and on the original function when it is a plain function). Creates the
  # eager, stateful or stateless py op. Returns the op's outputs: a list, or a single tensor
  # when Tout was not a list/tuple. Mutates the default graph and global registry.
  """See documentation for py_func and eager_py_func."""
  if not callable(func):
    raise ValueError(
        f"Expected func to be callable. Received func={func} of type "
        f"{type(func)}.")

  original_func = func
  func = autograph.do_not_convert(func)
  inp = variable_utils.convert_variables_to_tensors(list(inp))

  # Normalize Tout.
  is_list_or_tuple = isinstance(Tout, (list, tuple))
  Tout = Tout if is_list_or_tuple else [Tout]
  Tout = [_as_dtype_or_type_spec(t) for t in Tout]

  # Check if we need to handle CompositeTensor inputs or outputs.
  handle_composite_tensors = (
      use_eager_py_func and
      (any(isinstance(v, composite_tensor.CompositeTensor) for v in inp) or
       any(isinstance(t, type_spec.TypeSpec) for t in Tout)))
  if handle_composite_tensors:
    func, inp, Tout, out_structure = _wrap_for_composites(func, inp, Tout)

  if use_eager_py_func:
    func = EagerFunc(func, Tout, is_grad_func)

  # Tying the registered function's lifetime with the current default graph is
  # not reliable. For example, a binary may switch graphs in between model
  # training end evaluation, via saved_model. Those binaries work because the
  # original function is global, and break once the registered
  # function is an anonymous lambda, like the one produced by do_not_convert.
  # To avoid breaking those cases, we attach the wrapper to the original
  # function so that their lifetime is connected.
  # TODO(b/144286616): Remove this.
  if tf_inspect.isfunction(original_func):
    # Note: this check is needed because original_func may be a descriptor
    # (https://docs.python.org/3/howto/descriptor.html)
    # and we can't attach attributes to those.
    original_func.ag_dnc_wrapper__ = func

  token = _py_funcs.insert(func)
  # We tie the registered function's lifetime with the current default graph,
  # i.e., when the current graph is destroyed, we remove its py funcs.
  graph = ops.get_default_graph()

  while True:
    current_graph = graph
    if isinstance(graph, function._FuncGraph):  # pylint: disable=protected-access
      graph = graph._outer_graph  # pylint: disable=protected-access
    elif isinstance(graph, func_graph.FuncGraph):
      graph = graph.outer_graph
    if graph is current_graph:
      break

  # TODO(zhifengc): Consider adding a Graph method to collect
  # `cleanup` objects in one of its member.
  if not hasattr(graph, "_py_funcs_used_in_graph"):
    graph._py_funcs_used_in_graph = []  # pylint: disable=protected-access

  # Store a reference to the function in the graph to ensure it stays alive
  # as long as the graph lives. When the graph is destroyed, the function
  # is left to the garbage collector for destruction as well.
  graph._py_funcs_used_in_graph.append(func)  # pylint: disable=protected-access

  if use_eager_py_func:
    result = gen_script_ops.eager_py_func(
        input=inp,
        token=token,
        is_async=context.is_async(),
        Tout=Tout,
        name=name)
  else:
    if stateful:
      result = gen_script_ops.py_func(
          input=inp, token=token, Tout=Tout, name=name)
    else:
      result = gen_script_ops.py_func_stateless(
          input=inp, token=token, Tout=Tout, name=name)

  if handle_composite_tensors and Tout:
    result = nest.pack_sequence_as(
        out_structure, result, expand_composites=True)

  return result if is_list_or_tuple else result[0]


# TODO(akshayka): Implement higher-order derivatives.
@ops.RegisterGradient("EagerPyFunc")
def _EagerPyFuncGrad(op, *dy):
  # [FUNCTION DOC]
  # HELPER: registered gradient function for the EagerPyFunc op.
  # Inputs: the forward op and the upstream gradients. Builds another eager py op that, when run,
  # pops the saved tape from the tape cache (by the op's token) and asks it for gradients of the
  # recorded outputs with respect to the recorded inputs. In graph mode it also tells the forward
  # eager wrapper to record tapes. Returns the gradient op outputs. Only first-order gradients
  # are supported. Side effects: adds ops to the graph, consumes tape cache entries at run time.
  """Computes the gradient of an EagerPyFunc."""

  token = op.get_attr("token")

  def eagerly_executed_grad(*dy):
    # [FUNCTION DOC]
    # HELPER: nested callback run eagerly at gradient time. Takes upstream gradients, removes the
    # saved (tape, inputs, outputs) entry for the forward token from the tape cache (a missing
    # entry raises a key error), and returns the tape's gradients of the outputs with respect to
    # the inputs using the given upstream gradients.
    tape, eager_inputs, eager_outputs = tape_cache.pop(compat.as_bytes(token))
    return tape.gradient(eager_outputs, eager_inputs, output_gradients=dy)

  with ops.control_dependencies(op.outputs):
    gradient_op = _internal_py_func(
        func=eagerly_executed_grad,
        inp=dy,
        Tout=[tensor.dtype for tensor in op.inputs],
        use_eager_py_func=True,
        is_grad_func=True)

  if not context.executing_eagerly():
    # In graph mode, we find the func object from its token and
    # notify the eager func object it needs to support the gradients.
    func = _py_funcs.get(token.decode())
    assert isinstance(func, EagerFunc), (
        f"EagerPyFuncGrad called on a non-EagerFunc object: {func}.")
    func.set_support_graph_mode_gradient()
  return gradient_op


def _check_args_and_maybe_make_decorator(
    script_op, script_op_name, func=None, inp=None, Tout=None, **kwargs
):
  # [FUNCTION DOC]
  # HELPER: argument validation shared by py_function and numpy_function.
  # Inputs: the op-building function, its display name, and the user's func, inp, Tout and extra
  # keyword options. Raises a type error if Tout is missing, if func is null but inp is given, or
  # if func is given but inp is missing. When func is null returns a decorator factory result:
  # a decorator that wraps a function so calling it passes its positional arguments as inputs to
  # the op builder with the given Tout and options. Otherwise returns null (arguments are valid).
  """Checks the arguments and returns a decorator if func is None."""
  if Tout is None:
    raise TypeError(
        "Missing required argument: 'Tout'\n"
        f"  If using {script_op_name} as a decorator, set `Tout`\n"
        "  **by name** above the function:\n"
        f"  `@{script_op_name}(Tout=tout)`"
    )

  if func is None:
    if inp is not None:
      raise TypeError(
          f"Don't set the `inp` argument when using {script_op_name} as a "
          "decorator (`func=None`)."
      )

    def py_function_decorator(fun):
      # [FUNCTION DOC]
      # HELPER: the decorator returned to the user. Takes the function to decorate and returns
      # a wrapper (preserving its metadata) that forwards calls to the op builder.
      @functools.wraps(fun)
      def py_function_wrapper(*args):
        # [FUNCTION DOC]
        # HELPER: wrapper produced by the decorator. Takes positional arguments, passes them as
        # the input list together with the captured Tout and options to the op builder, and
        # returns its result (tensors).
        return script_op(fun, inp=args, Tout=Tout, **kwargs)

      return py_function_wrapper

    return py_function_decorator

  if inp is None:
    raise TypeError(
        "Missing argument `inp`:\n"
        "  You must set the `inp` argument (the list of arguments to the\n"
        f"  function), unless you use `{script_op_name}` as a decorator"
        "(`func=None`)."
    )

  return None


@tf_export("py_function")
@dispatch.add_dispatch_support
def eager_py_func(func=None, inp=None, Tout=None, name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring contains a garbled phrase ("nofunc=None, inp=None, Tout=None
  # tensorflow code") where it should say "use TensorFlow code"; the behavior is unaffected.
  # Public py_function (exported). Runs a Python function eagerly inside a graph or function.
  # Inputs: func (callable, null when used as a decorator), inp (list or single tensor or
  # composite tensor; must be null when decorating), Tout (dtype, type spec, empty list for no
  # result, or a list of those; required), name (optional). Validates arguments (type error on
  # misuse); returns a decorator when func is null, otherwise returns the output tensor(s). When
  # eager execution is active outside functions the op is placed on the host address space
  # device. Side effects: registers the function and adds an op to the current graph.
  # TODO(b/338268835): Remove "pyformat: disable" and the "pyformat: enable"
  # line below if this feature request is implemented.
  # pyformat: disable
  """Wraps a python function into a TensorFlow op that executes it eagerly.

  Using `tf.py_function` inside a `tf.function` allows you to run a python
  function using eager execution, inside the `tf.function`'s graph.
  This has two main effects:

  1. This allows you to use nofunc=None, inp=None, Tout=None tensorflow code
  inside your `tf.function`.
  2. It allows you to run python control logic in a `tf.function` without
  relying on `tf.autograph` to convert the code to use tensorflow control logic
  (tf.cond, tf.while_loop).

  Both of these features can be useful for debugging.

  Since `tf.py_function` operates on `Tensor`s it is still
  differentiable (once).

  There are two ways to use this function:

  ### As a decorator

  Use `tf.py_function` as a decorator to ensure the function always runs
  eagerly.

  When using `tf.py_function` as a decorator:

  * you must set `Tout`
  * you may set `name`
  * you must not set `func` or `inp`

  For example, you might use `tf.py_function` to
  implement the log huber function.

  >>> @tf.py_function(Tout=tf.float32)
  ... def py_log_huber(x, m):
  ...   print('Running with eager execution.')
  ...   if tf.abs(x) <= m:
  ...     return x**2
  ...   else:
  ...     return m**2 * (1 - 2 * tf.math.log(m) + tf.math.log(x**2))

  Under eager execution the function operates normally:

  >>> x = tf.constant(1.0)
  >>> m = tf.constant(2.0)
  >>>
  >>> print(py_log_huber(x,m).numpy())
  Running with eager execution.
  1.0

  Inside a `tf.function` the `tf.py_function` is not converted to a `tf.Graph`.:

  >>> @tf.function
  ... def tf_wrapper(x):
  ...   print('Tracing.')
  ...   m = tf.constant(2.0)
  ...   return py_log_huber(x,m)

  The `tf.py_function` only executes eagerly, and only when the `tf.function`
  is called:

  >>> print(tf_wrapper(x).numpy())
  Tracing.
  Running with eager execution.
  1.0
  >>> print(tf_wrapper(x).numpy())
  Running with eager execution.
  1.0


  Gradients work as expected:

  >>> with tf.GradientTape() as t:
  ...   t.watch(x)
  ...   y = tf_wrapper(x)
  Running with eager execution.
  >>>
  >>> t.gradient(y, x).numpy()
  2.0

  ### Inplace

  You can also skip the decorator and use `tf.py_function` in-place.
  This form is a useful shortcut if you don't control the function's source,
  but it is harder to read.

  >>> # No decorator
  >>> def log_huber(x, m):
  ...   if tf.abs(x) <= m:
  ...     return x**2
  ...   else:
  ...     return m**2 * (1 - 2 * tf.math.log(m) + tf.math.log(x**2))
  >>>
  >>> x = tf.constant(1.0)
  >>> m = tf.constant(2.0)
  >>>
  >>> tf.py_function(func=log_huber, inp=[x, m], Tout=tf.float32).numpy()
  1.0

  ### More info

  You can also use `tf.py_function` to debug your models at runtime
  using Python tools, i.e., you can isolate portions of your code that
  you want to debug, wrap them in Python functions and insert `pdb` tracepoints
  or print statements as desired, and wrap those functions in
  `tf.py_function`.

  For more information on eager execution, see the
  [Eager guide](https://tensorflow.org/guide/eager).

  `tf.py_function` is similar in spirit to `tf.numpy_function`, but unlike
  the latter, the former lets you use TensorFlow operations in the wrapped
  Python function. In particular, while `tf.compat.v1.py_func` only runs on CPUs
  and wraps functions that take NumPy arrays as inputs and return NumPy arrays
  as outputs, `tf.py_function` can be placed on GPUs and wraps functions
  that take Tensors as inputs, execute TensorFlow operations in their bodies,
  and return Tensors as outputs.

  Note: We recommend to avoid using `tf.py_function` outside of prototyping
  and experimentation due to the following known limitations:

  * Calling `tf.py_function` will acquire the Python Global Interpreter Lock
    (GIL) that allows only one thread to run at any point in time. This will
    preclude efficient parallelization and distribution of the execution of the
    program.

  * The body of the function (i.e. `func`) will not be serialized in a
    `GraphDef`. Therefore, you should not use this function if you need to
    serialize your model and restore it in a different environment.

  * The operation must run in the same address space as the Python program
    that calls `tf.py_function()`. If you are using distributed
    TensorFlow, you must run a `tf.distribute.Server` in the same process as the
    program that calls `tf.py_function()` and you must pin the created
    operation to a device in that server (e.g. using `with tf.device():`).

  * Currently `tf.py_function` is not compatible with XLA. Calling
    `tf.py_function` inside `tf.function(jit_compile=True)` will raise an
    error.

  Args:
    func: A Python function that accepts `inp` as arguments, and returns a value
      (or list of values) whose type is described by `Tout`. Do not set `func`
      when using `tf.py_function` as a decorator.
    inp: Input arguments for `func`.  A list whose elements are `Tensor`s or
      `CompositeTensors` (such as `tf.RaggedTensor`); or a single `Tensor` or
      `CompositeTensor`. Do not set `inp` when using `tf.py_function` as a
      decorator.
    Tout: The type(s) of the value(s) returned by `func`. One of the following:
      * If `func` returns a `Tensor` (or a value that can be converted to a
        Tensor): the `tf.DType` for that value.
      * If `func` returns a `CompositeTensor`: The `tf.TypeSpec` for that value.
      * If `func` returns `None`: the empty list (`[]`).
      * If `func` returns a list of `Tensor` and `CompositeTensor` values: a
        corresponding list of `tf.DType`s and `tf.TypeSpec`s for each value.
    name: A name for the operation (optional).

  Returns:
    * If `func` is `None` this returns a decorator that will ensure the
    decorated function will always run with eager execution even if called
    from a `tf.function`/`tf.Graph`.
    * If used `func` is not `None` this executes `func` with eager execution
    and returns the result: a `Tensor`, `CompositeTensor`, or list of
    `Tensor` and `CompositeTensor`; or an empty list if `func` returns `None`.
  """
  # pyformat: enable

  decorator = _check_args_and_maybe_make_decorator(
      eager_py_func, "tf.py_function", func=func, inp=inp, Tout=Tout, name=name
  )
  if decorator is not None:
    return decorator

  if ops.executing_eagerly_outside_functions():
    with ops.device(context.context().host_address_space()):
      return _internal_py_func(
          func=func, inp=inp, Tout=Tout, use_eager_py_func=True, name=name)

  return _internal_py_func(
      func=func, inp=inp, Tout=Tout, use_eager_py_func=True, name=name)


def py_func_common(func, inp, Tout, stateful=True, name=None):
  # [FUNCTION DOC]
  # HELPER: common implementation of the numpy-style py_func and numpy_function.
  # Inputs: func taking numpy arrays, inp list of tensors, Tout dtype(s), stateful flag (default
  # true) and optional name. When executing eagerly it simply calls func on numpy copies of the
  # inputs, flattens and converts non-null results to tensors, and unwraps a single result.
  # In graph mode it builds a py op (stateful or stateless), placed on the host address space
  # device when eager is enabled outside functions. Returns tensor(s) the function computes.
  """Wraps a python function and uses it as a TensorFlow op.

  Given a python function `func`, which takes numpy arrays as its
  arguments and returns numpy arrays as its outputs, wrap this function as an
  operation in a TensorFlow graph. The following snippet constructs a simple
  TensorFlow graph that invokes the `np.sinh()` NumPy function as a operation
  in the graph:

  ```python
  def my_func(x):
    # x will be a numpy array with the contents of the placeholder below
    return np.sinh(x)
  input = tf.compat.v1.placeholder(tf.float32)
  y = tf.compat.v1.py_func(my_func, [input], tf.float32)
  ```

  **N.B.** The `tf.compat.v1.py_func()` operation has the following known
  limitations:

  * The body of the function (i.e. `func`) will not be serialized in a
    `GraphDef`. Therefore, you should not use this function if you need to
    serialize your model and restore it in a different environment.

  * The operation must run in the same address space as the Python program
    that calls `tf.compat.v1.py_func()`. If you are using distributed
    TensorFlow, you
    must run a `tf.distribute.Server` in the same process as the program that
    calls
    `tf.compat.v1.py_func()` and you must pin the created operation to a device
    in that
    server (e.g. using `with tf.device():`).

  Note: It produces tensors of unknown shape and rank as shape inference
    does not work on arbitrary Python code.
    If you need the shape, you need to set it based on statically
    available information.

    E.g.
    ```python
    import tensorflow as tf
    import numpy as np

    def make_synthetic_data(i):
        return np.cast[np.uint8](i) * np.ones([20,256,256,3],
                dtype=np.float32) / 10.

    def preprocess_fn(i):
        ones = tf.py_function(make_synthetic_data,[i],tf.float32)
        ones.set_shape(tf.TensorShape([None, None, None, None]))
        ones = tf.image.resize(ones, [224,224])
        return ones

    ds = tf.data.Dataset.range(10)
    ds = ds.map(preprocess_fn)
    ```

  Args:
    func: A Python function, which accepts `ndarray` objects as arguments and
      returns a list of `ndarray` objects (or a single `ndarray`). This function
      must accept as many arguments as there are tensors in `inp`, and these
      argument types will match the corresponding `tf.Tensor` objects in `inp`.
      The returns `ndarray`s must match the number and types defined `Tout`.
      Important Note: Input and output numpy `ndarray`s of `func` are not
      guaranteed to be copies. In some cases their underlying memory will be
      shared with the corresponding TensorFlow tensors. In-place modification or
      storing `func` input or return values in python datastructures without
      explicit (np.)copy can have non-deterministic consequences.
    inp: A list of `Tensor` objects.
    Tout: A list or tuple of tensorflow data types or a single tensorflow data
      type if there is only one, indicating what `func` returns.
    stateful: (Boolean.) If True, the function should be considered stateful. If
      a function is stateless, when given the same input it will return the same
      output and have no observable side effects. Optimizations such as common
      sub-expression elimination are only performed on stateless operations.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` or a single `Tensor` which `func` computes.

  @compatibility(TF2)

  This name was deprecated and removed in TF2, but `tf.numpy_function` is a
  near-exact replacement, just drop the `stateful` argument (all
  `tf.numpy_function` calls are considered stateful). It is compatible with
  eager execution and `tf.function`.

  `tf.py_function` is a close but not an exact replacement, passing TensorFlow
  tensors to the wrapped function instead of NumPy arrays, which provides
  gradients and can take advantage of accelerators.

  Before:

  >>> def fn_using_numpy(x):
  ...   x[0] = 0.
  ...   return x
  >>> tf.compat.v1.py_func(fn_using_numpy, inp=[tf.constant([1., 2.])],
  ...     Tout=tf.float32, stateful=False)
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0., 2.], dtype=float32)>

  After:

  >>> tf.numpy_function(fn_using_numpy, inp=[tf.constant([1., 2.])],
  ...     Tout=tf.float32)
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0., 2.], dtype=float32)>

  @end_compatibility
  """
  if context.executing_eagerly():
    result = func(*[np.array(x) for x in inp])
    result = nest.flatten(result)

    result = [x if x is None else ops.convert_to_tensor(x) for x in result]
    if len(result) == 1:
      # Mimic the automatic unwrapping in graph-mode py_func
      result, = result
    return result

  if ops.executing_eagerly_outside_functions():
    with ops.device(context.context().host_address_space()):
      return _internal_py_func(
          func=func,
          inp=inp,
          Tout=Tout,
          stateful=stateful,
          use_eager_py_func=False,
          name=name)

  return _internal_py_func(
      func=func,
      inp=inp,
      Tout=Tout,
      stateful=stateful,
      use_eager_py_func=False,
      name=name)


@deprecation.deprecated(
    date=None,
    instructions="""tf.py_func is deprecated in TF V2. Instead, there are two
    options available in V2.
    - tf.py_function takes a python function which manipulates tf eager
    tensors instead of numpy arrays. It's easy to convert a tf eager tensor to
    an ndarray (just call tensor.numpy()) but having access to eager tensors
    means `tf.py_function`s can use accelerators such as GPUs as well as
    being differentiable using a gradient tape.
    - tf.numpy_function maintains the semantics of the deprecated tf.py_func
    (it is not differentiable, and manipulates numpy arrays). It drops the
    stateful argument making all functions stateful.
    """)
@tf_export(v1=["py_func"])
@dispatch.add_dispatch_support
def py_func(func, inp, Tout, stateful=True, name=None):
  # [FUNCTION DOC]
  # Deprecated v1 public py_func. Thin forwarder to the common implementation with the same
  # inputs (function, input tensors, output types, stateful flag, name); returns its result.
  # Its docstring is assigned from the common implementation after the definition.
  return py_func_common(func, inp, Tout, stateful, name=name)


py_func.__doc__ = "%s" % py_func_common.__doc__


@tf_export("numpy_function")
@dispatch.add_dispatch_support
def numpy_function(func=None, inp=None, Tout=None, stateful=True, name=None):
  # [FUNCTION DOC]
  # Public numpy_function (exported). Wraps a function taking and returning numpy arrays as an op.
  # Inputs: func (null when decorating), inp (list of tensors; null when decorating), Tout
  # (required), stateful flag (default true), name. Raises a type error on bad argument
  # combinations. Returns a decorator when func is null, otherwise the output tensor(s).
  # Not differentiable. Registers the function and adds an op to the graph.
  """Wraps a python function and uses it as a TensorFlow op.

  Given a python function `func` wrap this function as an operation in a
  `tf.function`. `func` must take numpy arrays as its arguments and
  return numpy arrays as its outputs.

  There are two ways to use `tf.numpy_function`.

  ### As a decorator

  When using `tf.numpy_function` as a decorator:

  * you must set `Tout`
  * you may set `name`
  * you must not set `func` or `inp`

  >>> @tf.numpy_function(Tout=tf.float32)
  ... def my_numpy_func(x):
  ...   # x will be a numpy array with the contents of the input to the
  ...   # tf.function
  ...   print(f'executing eagerly, {x=}')
  ...   return np.sinh(x)

  The function runs eagerly:

  >>> my_numpy_func(1.0).numpy()
  executing eagerly, x=1.0
  1.17520

  The behavior doesn't change inside a `tf.function`:

  >>> @tf.function(input_signature=[tf.TensorSpec(None, tf.float32)])
  ... def tf_function(input):
  ...   y = tf.numpy_function(my_numpy_func, [input], tf.float32)
  ...   return y
  >>> tf_function(tf.constant(1.)).numpy()
  executing eagerly, x=array(1.)
  1.17520

  ### Inplace

  This form can be useful if you don't control the function's source,
  but it is harder to read.

  Here is the same function with no decorator:

  >>> def my_func(x):
  ...   # x will be a numpy array with the contents of the input to the
  ...   # tf.function
  ...   print(f'executing eagerly, {x=}')
  ...   return np.sinh(x)

  To run `tf.numpy_function` in-place, pass the function, its inputs, and the
  output type in a single call to `tf.numpy_function`:

  >>> tf.numpy_function(my_func, [tf.constant(1.0)], tf.float32)
  executing eagerly, x=array(1.)
  1.17520

  ### More info

  Comparison to `tf.py_function`:
  `tf.py_function` and `tf.numpy_function` are very similar, except that
  `tf.numpy_function` takes numpy arrays, and not `tf.Tensor`s. If you want the
  function to contain `tf.Tensors`, and have any TensorFlow operations executed
  in the function be differentiable, please use `tf.py_function`.

  Note: We recommend to avoid using `tf.numpy_function` outside of
  prototyping and experimentation due to the following known limitations:

  * Calling `tf.numpy_function` will acquire the Python Global Interpreter Lock
    (GIL) that allows only one thread to run at any point in time. This will
    preclude efficient parallelization and distribution of the execution of the
    program. Therefore, you are discouraged to use `tf.numpy_function` outside
    of prototyping and experimentation.

  * The body of the function (i.e. `func`) will not be serialized in a
    `tf.SavedModel`. Therefore, you should not use this function if you need to
    serialize your model and restore it in a different environment.

  * The operation must run in the same address space as the Python program
    that calls `tf.numpy_function()`. If you are using distributed
    TensorFlow, you must run a `tf.distribute.Server` in the same process as the
    program that calls `tf.numpy_function`  you must pin the created
    operation to a device in that server (e.g. using `with tf.device():`).

  * Currently `tf.numpy_function` is not compatible with XLA. Calling
    `tf.numpy_function` inside `tf.function(jit_compile=True)` will raise an
    error.

  * Since the function takes numpy arrays, you cannot take gradients
    through a numpy_function. If you require something that is differentiable,
    please consider using tf.py_function.

  Args:
    func: A Python function, which accepts `numpy.ndarray` objects as arguments
      and returns a list of `numpy.ndarray` objects (or a single
      `numpy.ndarray`). This function must accept as many arguments as there are
      tensors in `inp`, and these argument types will match the corresponding
      `tf.Tensor` objects in `inp`. The returns `numpy.ndarray`s must match the
      number and types defined `Tout`. Important Note: Input and output
      `numpy.ndarray`s of `func` are not guaranteed to be copies. In some cases
      their underlying memory will be shared with the corresponding TensorFlow
      tensors. In-place modification or storing `func` input or return values in
      python datastructures without explicit (np.)copy can have
      non-deterministic consequences.
    inp: A list of `tf.Tensor` objects.
    Tout: A list or tuple of tensorflow data types or a single tensorflow data
      type if there is only one, indicating what `func` returns.
    stateful: (Boolean.) Setting this argument to False tells the runtime to
      treat the function as stateless, which enables certain optimizations. A
      function is stateless when given the same input it will return the same
      output and have no side effects; its only purpose is to have a return
      value. The behavior for a stateful function with the `stateful` argument
      False is undefined. In particular, caution should be taken when mutating
      the input arguments as this is a stateful operation.
    name: (Optional) A name for the operation.

  Returns:
    * If `func` is `None` this returns a decorator that will ensure the
      decorated function will always run with eager execution even if called
      from a `tf.function`/`tf.Graph`.
    * If used `func` is not `None` this executes `func` with eager execution
      and returns the result: A single or list of `tf.Tensor` which `func`
      computes.
  """
  decorator = _check_args_and_maybe_make_decorator(
      numpy_function,
      "tf.numpy_function",
      func=func,
      inp=inp,
      Tout=Tout,
      stateful=stateful,
      name=name,
  )
  if decorator is not None:
    return decorator

  return py_func_common(func, inp, Tout, stateful=stateful, name=name)


def _as_dtype_or_type_spec(t):
  # [FUNCTION DOC]
  # HELPER: normalizes one output specification. Type specs are returned unchanged; anything
  # else is converted to a dtype (which may raise a type error for invalid values).
  return t if isinstance(t, type_spec.TypeSpec) else dtypes.as_dtype(t)


def _wrap_for_composites(func, inp, Tout):
  # [FUNCTION DOC]
  # HELPER: lets py_function handle composite tensors. Inputs: the user function, the input
  # list and the output specs. Flattens composite inputs into plain tensors, flattens Tout to
  # dtypes (tensor specs become their dtype), and builds a wrapper that repacks inputs, calls
  # the function, validates results and flattens them. Returns a tuple of the wrapper, flattened
  # inputs, flattened output dtypes and the original output structure.
  """Wraps user inputs to support composite tensors for `py_function`.

  1. Flattens `inp` to a list of Tensors (by flattening any composite tensors).
  2. Creates a wrapper function for `func` that expects flat inputs and:
     - Packs the inputs into the input structure expected by `func`.
     - Calls `func` with the packed inputs.
     - Checks that `func`'s output matches `Tout`.
     - Flattens func`'s output to a list of Tensors (flattening any composite
       tensors).

  Args:
    func: The function to wrap (`func` argument to `py_function`).
    inp: The input arguments for func (`inp` argument to `py_function`).
    Tout: The expected output types for func (`Tout` argument to `py_function).

  Returns:
    A tuple `(func, inp, Tout, out_structure)`, where `func` is the wrapped
    function, `inp` is the flattened inputs, `Tout` is the list of expected
    dtypes for the flattened outputs, and `out_structure` is the expected
    output structure (which can be used to pack the output tensors).
  """
  in_structure = [
      v if isinstance(v, composite_tensor.CompositeTensor) else 1 for v in inp
  ]
  inp = nest.flatten_up_to(in_structure, inp, expand_composites=True)
  out_structure = Tout
  Tout = [
      v.dtype if isinstance(v, tensor_spec.TensorSpec) else v
      for v in nest.flatten(Tout, expand_composites=True)
  ]

  def wrapped_func(*flat_inp):
    # [FUNCTION DOC]
    # HELPER: nested wrapper run by the py op. Takes flattened input tensors, repacks them into
    # the original structure, calls the user function, and returns a flat list of outputs
    # (empty if no output expected; a single value is treated as a one element list). For outputs
    # expected to be composite (non-plain type specs) it checks compatibility and flattens them;
    # for plain tensor outputs it rejects composite values. Raises a value error with a
    # description of the mismatch when a result does not match the expected output types.
    structured_inp = nest.pack_sequence_as(
        in_structure, flat_inp, expand_composites=True)
    out = func(*structured_inp)
    if not out_structure:
      return []  # Ignore return value if none is requested/expected.
    if not isinstance(out, (list, tuple)):
      out = [out]  # func may return a single value instead of a list.
    flat_out = []
    for elt, expected_type in zip(out, out_structure):
      if (isinstance(expected_type, type_spec.TypeSpec) and
          not isinstance(expected_type, tensor_spec.TensorSpec)):
        if not expected_type.is_compatible_with(elt):
          # pylint: disable=protected-access
          raise ValueError(
              f"py_function: func={func} returned {out!r}, "
              f"which did not match Tout={out_structure!r}.\nIn particular, "
              f"{elt!r} is not compatible with {expected_type!r}.")
        flat_out.extend(nest.flatten(elt, expand_composites=True))
      else:
        # Pro-actively check if the return value is a composite tensor when
        # we expect a Tensor.  We would catch this later (when we call
        # convert_to_tensor), but checking it here lets us give a better
        # error message.
        if isinstance(elt, composite_tensor.CompositeTensor):
          raise ValueError(
              f"py_function: func={func} returned {out!r}, "
              f"which did not match Tout={out_structure!r}.\nIn particular, "
              f"{elt!r} is not a Tensor.")
        flat_out.append(elt)
    return flat_out

  return wrapped_func, inp, Tout, out_structure


ops.NotDifferentiable("PyFunc")
ops.NotDifferentiable("PyFuncStateless")
