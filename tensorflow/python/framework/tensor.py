# [MODULE DOC]
# Core symbolic tensor types for TensorFlow's Python framework. Defines the abstract Tensor base
# class (shared by graph tensors and eager tensors: dtype, shape, naming, operator overloading
# hooks, equality/hash behavior, conversion refusal for numpy), the TensorSpec type
# specification, its bounded variant, and related helpers such as a spec-from-tensor
# constructor, a placeholder-style value holder, and the tensor-like and conversion
# registrations. Many public classes are exported to the public API namespace by decorators.
# Import-time side effects: creates a monitoring gauge for the tensor-equality flag, registers
# type specs with the type-spec registry and the trace type / nested-structure coder, registers
# conversion functions with the tensor conversion registry, and reads the TF2 enabled setting
# to choose the default of tensor equality semantics.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tensor and TensorSpec classes."""

from typing import Optional, Type

import numpy as np

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.function import trace_type
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.eager import record
from tensorflow.python.framework import common_shapes
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import stack
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.ops import handle_data_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import core as core_tf_types
from tensorflow.python.types import internal
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export


_tensor_equality_api_usage_gauge = monitoring.BoolGauge(
    "/tensorflow/api/enable_tensor_equality",
    "Whether ops.enable_tensor_equality() is called.")


def _override_helper(clazz_object, operator, func):
  # [FUNCTION DOC]
  # HELPER: used by the tensor classes' operator-override hook to install operator overloads.
  # Inputs: clazz_object (the class to modify, such as the tensor or sparse tensor class),
  # operator (string name of a special method), func (replacement callable). Raises a value
  # error if the operator name is not in the tensor class's allowed overloadable set. Otherwise
  # sets the attribute on the class (side effect: mutates the class). Returns nothing.
  """Overrides (string) operator on Tensors to call func.

  Args:
    clazz_object: the class to override for; either Tensor or SparseTensor.
    operator: the string name of the operator to override.
    func: the function that replaces the overridden operator.

  Raises:
    ValueError: If operator is not allowed to be overwritten.
  """
  if operator not in Tensor.OVERLOADABLE_OPERATORS:
    raise ValueError(f"Overriding {operator} is disallowed. "
                     f"Allowed operators are {Tensor.OVERLOADABLE_OPERATORS}.")
  setattr(clazz_object, operator, func)


def _eval_using_default_session(tensors, feed_dict, graph, session=None):
  # [FUNCTION DOC]
  # HELPER: backs graph-mode tensor evaluation helpers.
  # Evaluates one tensor or a list of tensors in a session. Inputs: tensors (single tensor or
  # list), feed_dict (mapping from tensors or names to values), graph (the graph the tensors
  # belong to), session (optional explicit session). If no session is given the current default
  # session is used. Raises a value error if there is no default session, or if the chosen
  # session's graph is not the given graph. Returns the session run result: one array for a single
  # tensor or a list of arrays. Side effect: executes the graph in the session.
  """Uses the default session to evaluate one or more tensors.

  Args:
    tensors: A single Tensor, or a list of Tensor objects.
    feed_dict: A dictionary that maps Tensor objects (or tensor names) to lists,
      numpy ndarrays, TensorProtos, or strings.
    graph: The graph in which the tensors are defined.
    session: (Optional) A different session to use to evaluate "tensors".

  Returns:
    Either a single numpy ndarray if "tensors" is a single tensor; or a list
    of numpy ndarrays that each correspond to the respective element in
    "tensors".

  Raises:
    ValueError: If no default session is available; the default session
      does not have "graph" as its graph; or if "session" is specified,
      and it does not have "graph" as its graph.
  """
  if session is None:
    session = stack.get_default_session()
    if session is None:
      raise ValueError("Cannot evaluate tensor using `eval()`: No default "
                       "session is registered. Use `with "
                       "sess.as_default()` or pass an explicit session to "
                       "`eval(session=sess)`")
    if session.graph is not graph:
      raise ValueError("Cannot use the default session to evaluate tensor: "
                       "the tensor's graph is different from the session's "
                       "graph. Pass an explicit session to "
                       "`eval(session=sess)`.")
  else:
    if session.graph is not graph:
      raise ValueError("Cannot use the given session to evaluate tensor: "
                       "the tensor's graph is different from the session's "
                       "graph.")
  return session.run(tensors, feed_dict)


def _add_error_prefix(msg, *, name=None):
  # [FUNCTION DOC]
  # HELPER: formats error messages with an optional op name prefix.
  # KNOWN ISSUE: duplicate of the identically named helper in the tensor conversion registry
  # module, and it appears unused within this module.
  # Returns the message unchanged if name is absent, else the name, a colon and the message.
  return msg if name is None else f"{name}: {msg}"


class _TensorIterator(object):
  # [CLASS DOC]
  # HELPER: iterator returned by tensor iteration in eager mode.
  # Walks the leading dimension of a tensor by indexing it with successive integers until a
  # known limit. Holds the tensor, the current index and the limit (slots, no instance dict).
  # Performs no validation; the caller supplies a known first dimension.
  """Iterates over the leading dim of a Tensor. Performs no error checks."""

  __slots__ = ["_tensor", "_index", "_limit"]

  def __init__(self, tensor, dim0):
    # [FUNCTION DOC]
    # Constructor. Inputs: tensor (the tensor to iterate) and dim0 (integer count of elements
    # along the leading dimension). Starts the position at zero and stores the limit.
    self._tensor = tensor
    self._index = 0
    self._limit = dim0

  def __iter__(self):
    # [FUNCTION DOC]
    # Iterator protocol: returns the iterator itself.
    return self

  def __next__(self):
    # [FUNCTION DOC]
    # Returns the tensor slice at the current index and advances the position. Raises the
    # standard stop-iteration signal once the index equals the limit. Side effect: increments
    # the stored index.
    if self._index == self._limit:
      raise StopIteration
    result = self._tensor[self._index]
    self._index += 1
    return result

  next = __next__  # python2.x compatibility.


@tf_export("Tensor", "experimental.numpy.ndarray", v1=["Tensor"])
class Tensor(internal.NativeObject, core_tf_types.Symbol):
  # [CLASS DOC]
  # Abstract base of all symbolic and eager tensors: a multidimensional array with a single
  # element type and a (possibly partially known) shape. Exported publicly (also under the numpy
  # ndarray alias in the numpy-compat namespace). Subclasses (graph tensor, eager tensor) supply
  # the underlying storage and the private attributes this class relies on: dtype, name, a
  # shape description, operation, value index and device. This class provides the property
  # accessors for dtype, name, shape and rank, shape manipulation, iteration (eager only),
  # string forms, hashing governed by a class-level equality flag, refusal of numpy and length
  # conversion for symbolic tensors, and a class-level set of operator names that may be
  # overloaded by other modules.
  """A `tf.Tensor` represents a multidimensional array of elements.

  All elements are of a single known data type.

  When writing a TensorFlow program, the main object that is
  manipulated and passed around is the `tf.Tensor`.

  A `tf.Tensor` has the following properties:

  * a single data type (float32, int32, or string, for example)
  * a shape

  TensorFlow supports eager execution and graph execution.  In eager
  execution, operations are evaluated immediately.  In graph
  execution, a computational graph is constructed for later
  evaluation.

  TensorFlow defaults to eager execution.  In the example below, the
  matrix multiplication results are calculated immediately.

  >>> # Compute some values using a Tensor
  >>> c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
  >>> d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
  >>> e = tf.matmul(c, d)
  >>> print(e)
  tf.Tensor(
  [[1. 3.]
   [3. 7.]], shape=(2, 2), dtype=float32)

  Note that during eager execution, you may discover your `Tensors` are actually
  of type `EagerTensor`.  This is an internal detail, but it does give you
  access to a useful function, `numpy`:

  >>> type(e)
  <class '...ops.EagerTensor'>
  >>> print(e.numpy())
    [[1. 3.]
     [3. 7.]]

  In TensorFlow, `tf.function`s are a common way to define graph execution.

  A Tensor's shape (that is, the rank of the Tensor and the size of
  each dimension) may not always be fully known.  In `tf.function`
  definitions, the shape may only be partially known.

  Most operations produce tensors of fully-known shapes if the shapes of their
  inputs are also fully known, but in some cases it's only possible to find the
  shape of a tensor at execution time.

  A number of specialized tensors are available: see `tf.Variable`,
  `tf.constant`, `tf.placeholder`, `tf.sparse.SparseTensor`, and
  `tf.RaggedTensor`.

  Caution: when constructing a tensor from a numpy array or pandas dataframe
  the underlying buffer may be re-used:

  ```python
  a = np.array([1, 2, 3])
  b = tf.constant(a)
  a[0] = 4
  print(b)  # tf.Tensor([4 2 3], shape=(3,), dtype=int64)
  ```

  Note: this is an implementation detail that is subject to change and users
  should not rely on this behaviour.

  For more on Tensors, see the [guide](https://tensorflow.org/guide/tensor).
  """
  # List of Python operators that we allow to override.
  OVERLOADABLE_OPERATORS = {
      # Binary.
      "__add__",
      "__radd__",
      "__sub__",
      "__rsub__",
      "__mul__",
      "__rmul__",
      "__div__",
      "__rdiv__",
      "__truediv__",
      "__rtruediv__",
      "__floordiv__",
      "__rfloordiv__",
      "__mod__",
      "__rmod__",
      "__lt__",
      "__le__",
      "__gt__",
      "__ge__",
      "__ne__",
      "__eq__",
      "__and__",
      "__rand__",
      "__or__",
      "__ror__",
      "__xor__",
      "__rxor__",
      "__getitem__",
      "__pow__",
      "__rpow__",
      # Unary.
      "__invert__",
      "__neg__",
      "__abs__",
      "__matmul__",
      "__rmatmul__"
  }

  # Whether to allow hashing or numpy-style equality
  _USE_EQUALITY = tf2.enabled()

  def __getattr__(self, name):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the final line calls the normal attribute lookup but discards its result and
    # returns nothing; since this hook only runs after normal lookup failed, that call raises
    # the attribute error itself, so the behavior is correct only by accident.
    # Fallback for missing attributes. Input: name of the missing attribute. For a fixed set of
    # numpy-style names (transpose, reshape, size, and similar) raises an attribute error whose
    # message tells the user to enable numpy behavior. For any other name it retries normal
    # lookup, which raises an attribute error again.
    if name in {"T", "astype", "ravel", "transpose", "reshape", "clip", "size",
                "tolist", "data"}:
      # TODO(wangpeng): Export the enable_numpy_behavior knob
      raise AttributeError(
          f"{type(self).__name__} object has no attribute '{name}'. " + """
        If you are looking for numpy-related methods, please run the following:
        tf.experimental.numpy.experimental_enable_numpy_behavior()
      """)
    self.__getattribute__(name)

  @property
  def dtype(self):
    # [FUNCTION DOC]
    # Read-only property returning the element data type object stored by the subclass.
    """The `DType` of elements in this tensor."""
    return self._dtype

  @property
  def name(self):
    # [FUNCTION DOC]
    # Read-only property returning the tensor's name as stored by the subclass (for graph
    # tensors, the producing operation name and output index).
    return self._name

  @property
  def shape(self) -> tensor_shape.TensorShape:
    # [FUNCTION DOC]
    # Read-only property giving the static shape. Lazily builds a shape object from the
    # subclass's raw shape description (a pair of dimension sizes and an unknown-rank flag) and
    # caches it; unknown rank yields an unknown shape, otherwise a shape from the dimension list
    # (negative or null sizes meaning unknown). Side effect: sets the cached shape value.
    """Returns a `tf.TensorShape` that represents the shape of this tensor.

    >>> t = tf.constant([1,2,3,4,5])
    >>> t.shape
    TensorShape([5])

    `tf.Tensor.shape` is equivalent to `tf.Tensor.get_shape()`.

    In a `tf.function` or when building a model using
    `tf.keras.Input`, they return the build-time shape of the
    tensor, which may be partially unknown.

    A `tf.TensorShape` is not a tensor. Use `tf.shape(t)` to get a tensor
    containing the shape, calculated at runtime.

    See `tf.Tensor.get_shape()`, and `tf.TensorShape` for details and examples.
    """
    if self._shape_val is None:
      dims, unknown_shape = self._shape
      if unknown_shape:
        self._shape_val = tensor_shape.unknown_shape()
      else:
        self._shape_val = tensor_shape.TensorShape(dims)
    return self._shape_val

  @property
  def ndim(self):
    # [FUNCTION DOC]
    # Read-only property returning the rank (number of dimensions) of the static shape, or the
    # null value when the rank is unknown.
    return self.shape.rank

  def _disallow(self, task):
    # [FUNCTION DOC]
    # HELPER: shared by the disallowed-operation checks below.
    # Always raises an operator-not-allowed-in-graph error. Input: task, a phrase describing
    # the forbidden action, used to build a message with advice about eager mode, function
    # decoration and autograph source visibility. Never returns.
    raise errors.OperatorNotAllowedInGraphError(
        f"{task} is not allowed."
        " You can attempt the following resolutions to the problem:"
        " If you are running in Graph mode, use Eager execution mode"
        " or decorate this function with @tf.function."
        " If you are using AutoGraph, you can try decorating this function"
        " with @tf.function. If that does not work, then you may be using"
        " an unsupported feature or your source code may not be visible"
        " to AutoGraph. See"
        " https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md#access-to-source-code"
        " for more information.")

  def _disallow_bool_casting(self):
    # [FUNCTION DOC]
    # HELPER: called by boolean conversion hooks of symbolic tensors.
    # Always raises the operator-not-allowed error stating a symbolic tensor cannot be used as
    # a Python boolean.
    self._disallow("Using a symbolic `tf.Tensor` as a Python `bool`")

  def _disallow_iteration(self):
    # [FUNCTION DOC]
    # HELPER: called by the iteration hook when not executing eagerly.
    # Always raises the operator-not-allowed error stating a symbolic tensor cannot be iterated.
    self._disallow("Iterating over a symbolic `tf.Tensor`")

  def __iter__(self):
    # [FUNCTION DOC]
    # Iteration over the leading dimension. If not executing eagerly, raises the disallowed
    # iteration error. Otherwise determines the first dimension size (raising a type error for
    # unknown shape, scalar, or unknown first dimension) and returns an iterator yielding
    # successive slices.
    if not context.executing_eagerly():
      self._disallow_iteration()

    first_dim = self._get_first_dim()
    return _TensorIterator(self, first_dim)

  def _get_first_dim(self):
    # [FUNCTION DOC]
    # HELPER: used by iteration to get the number of elements along axis zero.
    # Returns the integer size of the first dimension. Raises a type error if the rank is unknown,
    # if the tensor is a scalar, or if the first dimension is unknown.
    shape = self._shape_tuple()
    if shape is None:
      raise TypeError("Cannot iterate over a tensor with unknown shape.")
    if not shape:
      raise TypeError("Cannot iterate over a scalar tensor.")
    if shape[0] is None:
      raise TypeError(
          "Cannot iterate over a tensor with unknown first dimension.")
    return shape[0]

  def _shape_as_list(self):
    # [FUNCTION DOC]
    # HELPER: shape conversion utility.
    # Returns the static shape as a list of integer sizes (null entries for unknown dimensions),
    # or the null value when the rank is unknown.
    if self.shape.ndims is not None:
      return [dim.value for dim in self.shape.dims]
    else:
      return None

  def _shape_tuple(self):
    # [FUNCTION DOC]
    # HELPER: shape conversion utility.
    # Returns the static shape as a tuple of sizes (null entries for unknown dimensions), or the
    # null value when the rank is unknown.
    shape = self._shape_as_list()
    if shape is None:
      return None
    return tuple(shape)

  def _record_tape(self, capture):
    # [FUNCTION DOC]
    # HELPER: gradient plumbing for captured values in function graphs.
    # Records on the active gradient tape a pseudo-operation named for a captured value, taking
    # this tensor as input and the capture as output, with identity forward and backward
    # functions so gradients flow between them. Input: capture, the tensor representing the
    # captured value. Returns nothing. Side effect: writes to the gradient tape.
    """Connect this graph tensor with capture for gradients calculation."""
    record.record_operation(
        "captured_value",
        [self], [capture],
        backward_function=lambda x: [x],
        forward_function=lambda x: [x])

  def get_shape(self) -> tensor_shape.TensorShape:
    # [FUNCTION DOC]
    # Method form of the shape property: takes no inputs and returns the same static shape
    # object (possibly partially or fully unknown in graph mode). No side effects beyond the
    # shape caching done by the property.
    """Returns a `tf.TensorShape` that represents the shape of this tensor.

    In eager execution the shape is always fully-known.

    >>> a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    >>> print(a.shape)
    (2, 3)

    `tf.Tensor.get_shape()` is equivalent to `tf.Tensor.shape`.


    When executing in a `tf.function` or building a model using
    `tf.keras.Input`, `Tensor.shape` may return a partial shape (including
    `None` for unknown dimensions). See `tf.TensorShape` for more details.

    >>> inputs = tf.keras.Input(shape = [10])
    >>> # Unknown batch size
    >>> print(inputs.shape)
    (None, 10)

    The shape is computed using shape inference functions that are
    registered for each `tf.Operation`.

    The returned `tf.TensorShape` is determined at *build* time, without
    executing the underlying kernel. It is not a `tf.Tensor`. If you need a
    shape *tensor*, either convert the `tf.TensorShape` to a `tf.constant`, or
    use the `tf.shape(tensor)` function, which returns the tensor's shape at
    *execution* time.

    This is useful for debugging and providing early errors. For
    example, when tracing a `tf.function`, no ops are being executed, shapes
    may be unknown (See the [Concrete Functions
    Guide](https://www.tensorflow.org/guide/concrete_function) for details).

    >>> @tf.function
    ... def my_matmul(a, b):
    ...   result = a@b
    ...   # the `print` executes during tracing.
    ...   print("Result shape: ", result.shape)
    ...   return result

    The shape inference functions propagate shapes to the extent possible:

    >>> f = my_matmul.get_concrete_function(
    ...   tf.TensorSpec([None,3]),
    ...   tf.TensorSpec([3,5]))
    Result shape: (None, 5)

    Tracing may fail if a shape mismatch can be detected:

    >>> cf = my_matmul.get_concrete_function(
    ...   tf.TensorSpec([None,3]),
    ...   tf.TensorSpec([4,5]))
    Traceback (most recent call last):
    ...
    ValueError: Dimensions must be equal, but are 3 and 4 for 'matmul' (op:
    'MatMul') with input shapes: [?,3], [4,5].

    In some cases, the inferred shape may have unknown dimensions. If
    the caller has additional information about the values of these
    dimensions, `tf.ensure_shape` or `Tensor.set_shape()` can be used to augment
    the inferred shape.

    >>> @tf.function
    ... def my_fun(a):
    ...   a = tf.ensure_shape(a, [5, 5])
    ...   # the `print` executes during tracing.
    ...   print("Result shape: ", a.shape)
    ...   return a

    >>> cf = my_fun.get_concrete_function(
    ...   tf.TensorSpec([None, None]))
    Result shape: (5, 5)

    Returns:
      A `tf.TensorShape` representing the shape of this tensor.
    """
    return self.shape

  def set_shape(self, shape):
    # [FUNCTION DOC]
    # Refines the static shape of this tensor. Input: shape, anything convertible to a shape
    # (shape object, shape proto, list, tuple, or null meaning unknown). Clears the cached
    # shape, converts the input to a shape object, builds a list of dimension sizes with minus
    # one for unknown dimensions plus an unknown-rank flag, and passes both to the subclass's
    # private shape-setting hook, which merges them with the current shape (eager tensors treat
    # this as an assertion). Raises a value error if incompatible with the current shape.
    # Returns nothing. Side effect: mutates the tensor's shape metadata (and the underlying
    # graph shape) without any runtime check.
    """Updates the shape of this tensor.

    Note: It is recommended to use `tf.ensure_shape` instead of
    `Tensor.set_shape`, because `tf.ensure_shape` provides better checking for
    programming errors and can create guarantees for compiler
    optimization.

    With eager execution this operates as a shape assertion.
    Here the shapes match:

    >>> t = tf.constant([[1,2,3]])
    >>> t.set_shape([1, 3])

    Passing a `None` in the new shape allows any value for that axis:

    >>> t.set_shape([1,None])

    An error is raised if an incompatible shape is passed.

    >>> t.set_shape([1,5])
    Traceback (most recent call last):
    ...
    ValueError: Tensor's shape (1, 3) is not compatible with supplied
    shape [1, 5]

    When executing in a `tf.function`, or building a model using
    `tf.keras.Input`, `Tensor.set_shape` will *merge* the given `shape` with
    the current shape of this tensor, and set the tensor's shape to the
    merged value (see `tf.TensorShape.merge_with` for details):

    >>> t = tf.keras.Input(shape=[None, None, 3])
    >>> print(t.shape)
    (None, None, None, 3)

    Dimensions set to `None` are not updated:

    >>> t.set_shape([None, 224, 224, None])
    >>> print(t.shape)
    (None, 224, 224, 3)

    The main use case for this is to provide additional shape information
    that cannot be inferred from the graph alone.

    For example if you know all the images in a dataset have shape [28,28,3] you
    can set it with `tf.set_shape`:

    >>> @tf.function
    ... def load_image(filename):
    ...   raw = tf.io.read_file(filename)
    ...   image = tf.image.decode_png(raw, channels=3)
    ...   # the `print` executes during tracing.
    ...   print("Initial shape: ", image.shape)
    ...   image.set_shape([28, 28, 3])
    ...   print("Final shape: ", image.shape)
    ...   return image

    Trace the function, see the [Concrete Functions
    Guide](https://www.tensorflow.org/guide/concrete_function) for details.

    >>> cf = load_image.get_concrete_function(
    ...     tf.TensorSpec([], dtype=tf.string))
    Initial shape:  (None, None, 3)
    Final shape: (28, 28, 3)

    Similarly the `tf.io.parse_tensor` function could return a tensor with
    any shape, even the `tf.rank` is unknown. If you know that all your
    serialized tensors will be 2d, set it with `set_shape`:

    >>> @tf.function
    ... def my_parse(string_tensor):
    ...   result = tf.io.parse_tensor(string_tensor, out_type=tf.float32)
    ...   # the `print` executes during tracing.
    ...   print("Initial shape: ", result.shape)
    ...   result.set_shape([None, None])
    ...   print("Final shape: ", result.shape)
    ...   return result

    Trace the function

    >>> concrete_parse = my_parse.get_concrete_function(
    ...     tf.TensorSpec([], dtype=tf.string))
    Initial shape:  <unknown>
    Final shape:  (None, None)

    Make sure it works:

    >>> t = tf.ones([5,3], dtype=tf.float32)
    >>> serialized = tf.io.serialize_tensor(t)
    >>> print(serialized.dtype)
    <dtype: 'string'>
    >>> print(serialized.shape)
    ()
    >>> t2 = concrete_parse(serialized)
    >>> print(t2.shape)
    (5, 3)

    Caution: `set_shape` ensures that the applied shape is compatible with
    the existing shape, but it does not check at runtime. Setting
    incorrect shapes can result in inconsistencies between the
    statically-known graph and the runtime value of tensors. For runtime
    validation of the shape, use `tf.ensure_shape` instead. It also modifies
    the `shape` of the tensor.

    >>> # Serialize a rank-3 tensor
    >>> t = tf.ones([5,5,5], dtype=tf.float32)
    >>> serialized = tf.io.serialize_tensor(t)
    >>> # The function still runs, even though it `set_shape([None,None])`
    >>> t2 = concrete_parse(serialized)
    >>> print(t2.shape)
    (5, 5, 5)

    Args:
      shape: A `TensorShape` representing the shape of this tensor, a
        `TensorShapeProto`, a list, a tuple, or None.

    Raises:
      ValueError: If `shape` is not compatible with the current shape of
        this tensor.
    """
    # Reset cached shape.
    self._shape_val = None

    # We want set_shape to be reflected in the C API graph for when we run it.
    if not isinstance(shape, tensor_shape.TensorShape):
      shape = tensor_shape.TensorShape(shape)
    dim_list = []
    if shape.dims is None:
      unknown_shape = True
    else:
      unknown_shape = False
      for dim in shape.dims:
        if dim.value is None:
          dim_list.append(-1)
        else:
          dim_list.append(dim.value)
    self._set_shape(dim_list, unknown_shape)

  def _as_node_def_input(self):
    # [FUNCTION DOC]
    # HELPER: used when building node definitions in graphs.
    # Returns the string used in a node's input list to refer to this tensor: the producing
    # operation's name when this is output zero, otherwise the name, a colon and the output
    # index. Asserts that the operation has a non-empty name (assertion error otherwise).
    # KNOWN ISSUE: the docstring says a value error is raised for an unnamed operation, but the
    # code uses an assertion, which raises an assertion error (and is skipped when asserts are
    # disabled).
    """Return a value to use for the NodeDef "input" attribute.

    The returned string can be used in a NodeDef "input" attribute
    to indicate that the NodeDef uses this Tensor as input.

    Raises:
      ValueError: if this Tensor's Operation does not have a name.

    Returns:
      a string.
    """
    assert self._op.name
    if self.value_index == 0:
      return self._op.name
    else:
      return "%s:%d" % (self._op.name, self.value_index)

  def __str__(self):
    # [FUNCTION DOC]
    # Human-readable form: the word Tensor with the quoted name, followed by the shape if the
    # rank is known, the dtype name if present, and the device if non-empty.
    return "Tensor(\"%s\"%s%s%s)" % (
        self.name,
        (", shape=%s" %
         self.get_shape()) if self.get_shape().ndims is not None else "",
        (", dtype=%s" % self._dtype.name) if self._dtype else "",
        (", device=%s" % self.device) if self.device else "")

  def __repr__(self):
    # [FUNCTION DOC]
    # Debug representation showing the tensor name, static shape and dtype name in angle
    # brackets.
    return "<tf.Tensor '%s' shape=%s dtype=%s>" % (self.name, self.get_shape(),
                                                   self._dtype.name)

  def __hash__(self):
    # [FUNCTION DOC]
    # Hash by identity. When the class-level equality flag is on and the tensor is not in a
    # graph or its graph is a function-building graph, raises a type error telling the user to
    # use the reference wrapper as a key instead. Otherwise returns the object identity.
    g = getattr(self, "graph", None)
    if (Tensor._USE_EQUALITY and (g is None or g.building_function)):
      raise TypeError("Tensor is unhashable. "
                      "Instead, use tensor.ref() as the key.")
    else:
      return id(self)

  # NOTE(mrry): This enables the Tensor's overloaded "right" binary
  # operators to run when the left operand is an ndarray, because it
  # accords the Tensor class higher priority than an ndarray, or a
  # numpy matrix.
  # TODO(mrry): Convert this to using numpy's __numpy_ufunc__
  # mechanism, which allows more control over how Tensors interact
  # with ndarrays.
  __array_priority__ = 100

  def __array__(self, dtype=None):
    # [FUNCTION DOC]
    # Numpy conversion hook for symbolic tensors. Input: dtype, ignored. Always raises a
    # not-implemented error explaining a symbolic tensor cannot be converted to a numpy array.
    # (Eager tensors override this.)
    del dtype
    raise NotImplementedError(
        f"Cannot convert a symbolic tf.Tensor ({self.name}) to a numpy array."
        f" This error may indicate that you're trying to pass a Tensor to"
        f" a NumPy call, which is not supported.")

  def __len__(self):
    # [FUNCTION DOC]
    # Length hook for symbolic tensors. Always raises a type error advising use of the shape
    # instead. (Eager tensors override this.)
    raise TypeError(f"len is not well defined for a symbolic Tensor "
                    f"({self.name}). Please call `x.shape` rather than "
                    f"`len(x)` for shape information.")

  # TODO(mdan): This convoluted machinery is hard to maintain. Clean up.
  @staticmethod
  def _override_operator(operator, func):
    # [FUNCTION DOC]
    # HELPER: static hook used by the math-ops modules to install operator overloads on Tensor.
    # Inputs: operator (special-method name string) and func (replacement callable). Delegates to
    # the module override helper, raising a value error for operators not in the allowed set.
    # Side effect: mutates the tensor class.
    _override_helper(Tensor, operator, func)

  def __bool__(self):  # pylint: disable=invalid-bool-returned
    # [FUNCTION DOC]
    # KNOWN ISSUE: docstring says a type error is raised; the error raised is an
    # operator-not-allowed-in-graph error.
    # Boolean conversion hook that always fails. Takes no inputs. Delegates to the
    # disallow-bool-casting helper, which raises the framework's graph-operator error.
    """Dummy method to prevent a tensor from being used as a Python `bool`.

    This overload raises a `TypeError` when the user inadvertently
    treats a `Tensor` as a boolean (most commonly in an `if` or `while`
    statement), in code that was not converted by AutoGraph. For example:

    ```python
    if tf.constant(True):  # Will raise.
      # ...

    if tf.constant(5) < tf.constant(7):  # Will raise.
      # ...
    ```

    Raises:
      `TypeError`.
    """
    self._disallow_bool_casting()

  def __nonzero__(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: legacy Python 2 name that Python 3 never calls; dead code kept for
    # compatibility. The docstring's type error claim is also inaccurate (an
    # operator-not-allowed error is raised).
    # Same behavior as the boolean hook: always raises via the disallow-bool-casting helper.
    """Dummy method to prevent a tensor from being used as a Python `bool`.

    This is the Python 2.x counterpart to `__bool__()` above.

    Raises:
      `TypeError`.
    """
    self._disallow_bool_casting()

  def eval(self, feed_dict=None, session=None):
    # [FUNCTION DOC]
    # Evaluates this tensor in a graph-mode session (v1 workflow). Inputs: feed_dict (optional
    # mapping from tensors to feed values) and session (optional explicit session; otherwise the
    # default session). Returns a numpy array. Raises a value error (from the evaluation helper)
    # if no usable session exists or the session's graph differs from the tensor's graph.
    # Side effect: runs graph operations needed to compute the value. Requires a graph tensor
    # (uses its graph attribute).
    """Evaluates this tensor in a `Session`.

    Note: If you are not using `compat.v1` libraries, you should not need this,
    (or `feed_dict` or `Session`).  In eager execution (or within `tf.function`)
    you do not need to call `eval`.

    Calling this method will execute all preceding operations that
    produce the inputs needed for the operation that produces this
    tensor.

    *N.B.* Before invoking `Tensor.eval()`, its graph must have been
    launched in a session, and either a default session must be
    available, or `session` must be specified explicitly.

    Args:
      feed_dict: A dictionary that maps `Tensor` objects to feed values. See
        `tf.Session.run` for a description of the valid feed values.
      session: (Optional.) The `Session` to be used to evaluate this tensor. If
        none, the default session will be used.

    Returns:
      A numpy array corresponding to the value of this tensor.
    """
    return _eval_using_default_session(self, feed_dict, self.graph, session)

  @deprecation.deprecated(None, "Use ref() instead.")
  def experimental_ref(self):
    # [FUNCTION DOC]
    # Deprecated alias (emits a deprecation warning) for the ref method; returns the same
    # hashable reference wrapper.
    return self.ref()

  def ref(self):
    # [FUNCTION DOC]
    # Returns a hashable wrapper object holding this tensor, compared by identity of the
    # tensor, usable as a set member or mapping key since tensors themselves may be unhashable.
    # The wrapper's dereference method returns the original tensor. No inputs, no side effects.
    # tf.Variable also has the same ref() API.  If you update the
    # documentation here, please update tf.Variable.ref() as well.
    """Returns a hashable reference object to this Tensor.

    The primary use case for this API is to put tensors in a set/dictionary.
    We can't put tensors in a set/dictionary as `tensor.__hash__()` is no longer
    available starting Tensorflow 2.0.

    The following will raise an exception starting 2.0

    >>> x = tf.constant(5)
    >>> y = tf.constant(10)
    >>> z = tf.constant(10)
    >>> tensor_set = {x, y, z}
    Traceback (most recent call last):
      ...
    TypeError: Tensor is unhashable. Instead, use tensor.ref() as the key.
    >>> tensor_dict = {x: 'five', y: 'ten'}
    Traceback (most recent call last):
      ...
    TypeError: Tensor is unhashable. Instead, use tensor.ref() as the key.

    Instead, we can use `tensor.ref()`.

    >>> tensor_set = {x.ref(), y.ref(), z.ref()}
    >>> x.ref() in tensor_set
    True
    >>> tensor_dict = {x.ref(): 'five', y.ref(): 'ten', z.ref(): 'ten'}
    >>> tensor_dict[y.ref()]
    'ten'

    Also, the reference object provides `.deref()` function that returns the
    original Tensor.

    >>> x = tf.constant(5)
    >>> x.ref().deref()
    <tf.Tensor: shape=(), dtype=int32, numpy=5>
    """
    return object_identity.Reference(self)

  def __tf_tracing_type__(self, signature_context):
    # [FUNCTION DOC]
    # Produces the trace type used for function-retracing decisions. Input: signature_context
    # (unused tracing context). For resource or variant dtypes, copies any shape-inference handle
    # data attached to the tensor into a new dtype object; otherwise uses the dtype directly.
    # Returns a tensor spec with this tensor's shape and that dtype.
    if self.dtype == dtypes.resource or self.dtype == dtypes.variant:
      shape_inference_handle_data = handle_data_util.get_handle_data(self)
      handle_data = (
          dtypes.HandleData(shape_inference_handle_data)
          if shape_inference_handle_data
          else None
      )
      dtype = dtypes.DType(self.dtype._type_enum, handle_data)
    else:
      dtype = self.dtype
    spec = TensorSpec(self.shape, dtype)
    return spec

  def __tf_tensor__(
      self, dtype: Optional[dtypes.DType] = None, name: Optional[str] = None
      ) -> "Tensor":
    # [FUNCTION DOC]
    # Conversion hook used by the conversion registry. Inputs: dtype (optional requested element
    # type) and name (optional name used to prefix errors). Returns this same tensor when no
    # dtype is requested or the dtype is compatible with the tensor's dtype. Raises a value
    # error if the requested dtype is incompatible.
    if dtype is not None and not dtype.is_compatible_with(self.dtype):
      raise ValueError(
          _add_error_prefix(
              f"Tensor conversion requested dtype {dtype.name} "
              f"for Tensor with dtype {self.dtype.name}: {self!r}",
              name=name))
    return self


@tf_export(v1=["enable_tensor_equality"])
def enable_tensor_equality():
  # [FUNCTION DOC]
  # Turns on element-wise tensor equality (and therefore unhashable tensors). Takes no inputs,
  # returns nothing. Side effects: logs at verbose level, sets the monitoring gauge to true and
  # sets the tensor class-level equality flag to true, affecting all tensors process-wide.
  """Compare Tensors with element-wise comparison and thus be unhashable.

  Comparing tensors with element-wise allows comparisons such as
  tf.Variable(1.0) == 1.0. Element-wise equality implies that tensors are
  unhashable. Thus tensors can no longer be directly used in sets or as a key in
  a dictionary.
  """
  logging.vlog(1, "Enabling tensor equality")
  _tensor_equality_api_usage_gauge.get_cell().set(True)
  Tensor._USE_EQUALITY = True  # pylint: disable=protected-access


@tf_export(v1=["disable_tensor_equality"])
def disable_tensor_equality():
  # [FUNCTION DOC]
  # Turns off element-wise tensor equality, restoring legacy identity-based equality and
  # hashability. Takes no inputs, returns nothing. Side effects: logs at verbose level, sets the
  # monitoring gauge to false and clears the class-level equality flag process-wide.
  """Compare Tensors by their id and be hashable.

  This is a legacy behaviour of TensorFlow and is highly discouraged.
  """
  logging.vlog(1, "Disabling tensor equality")
  _tensor_equality_api_usage_gauge.get_cell().set(False)
  Tensor._USE_EQUALITY = False  # pylint: disable=protected-access


# TODO(b/249802365): Sanitize all TensorSpec names.
def sanitize_spec_name(name: str) -> str:
  # [FUNCTION DOC]
  # HELPER: normalizes spec names for use as parameter or node names.
  # Input: name, a string. Returns the word unknown for an empty or null name; otherwise lowers
  # the case, replaces every non-alphanumeric character with an underscore, and, if the result
  # does not start with a letter, prefixes it with a tensor-and-underscore marker. Pure.
  """Sanitizes Spec names. Matches Graph Node and Python naming conventions.

  Without sanitization, names that are not legal Python parameter names can be
  set which makes it challenging to represent callables supporting the named
  calling capability.

  Args:
    name: The name to sanitize.

  Returns:
    A string that meets Python parameter conventions.
  """
  if not name:
    return "unknown"

  # Lower case and replace non-alphanumeric chars with '_'
  swapped = "".join([c if c.isalnum() else "_" for c in name.lower()])

  if swapped[0].isalpha():
    return swapped
  else:
    return "tensor_" + swapped


def get_op_name(tensor_name):
  # [FUNCTION DOC]
  # HELPER: name parsing utility for graph tensor names.
  # KNOWN ISSUE: splitting on the colon unpacks exactly two pieces, so a name containing more
  # than one colon raises an unpacking value error with an unhelpful message.
  # Input: tensor_name, the full tensor name string, possibly with a leading caret for a control
  # dependency and a colon plus output index suffix. Returns the operation name (caret and
  # suffix removed). Raises a value error if the name is empty or null.
  """Extract the Op name from a Tensor name.

  The Op name is everything before a colon, if present,
  not including any ^ prefix denoting a control dependency.

  Args:
    tensor_name: the full name of a Tensor in the graph.
  Returns:
    The name of the Op of which the given Tensor is an output.
  Raises:
    ValueError: if tensor_name is None or empty.
  """
  if not tensor_name:
    raise ValueError(
        f"Tensor name cannot be empty or None. Received: {tensor_name}.")

  # Control dependency inputs start with ^.
  if tensor_name.startswith("^"):
    tensor_name = tensor_name[1:]
  if ":" in tensor_name:
    op_name, _ = tensor_name.split(":")
    return op_name
  return tensor_name


class DenseSpec(type_spec.TypeSpec):
  # [CLASS DOC]
  # HELPER: base class for specs of dense array-like values (used by the tensor spec and its
  # bounded variant). Stores a shape, a dtype and an optional name (slots); equality compares
  # exact type, shape, dtype and name, while hashing ignores the name. Provides compatibility
  # checks and the legacy dataset output type/shape/class accessors. Its component specs
  # property is the spec itself.
  """Describes a dense object with shape, dtype, and name."""

  __slots__ = ["_shape", "_dtype", "_name"]

  _component_specs = property(lambda self: self)

  def __init__(self, shape, dtype=dtypes.float32, name=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: shape (anything convertible to a shape), dtype (anything convertible
    # to a data type, default 32-bit float) and name (optional string). Stores the converted
    # shape and dtype and the name. Raises a type error if conversion of shape or dtype fails.
    """Creates a TensorSpec.

    Args:
      shape: Value convertible to `tf.TensorShape`. The shape of the tensor.
      dtype: Value convertible to `tf.DType`. The type of the tensor values.
      name: Optional name for the Tensor.

    Raises:
      TypeError: If shape is not convertible to a `tf.TensorShape`, or dtype is
        not convertible to a `tf.DType`.
    """
    self._shape = tensor_shape.TensorShape(shape)
    self._dtype = dtypes.as_dtype(dtype)
    self._name = name

  @property
  def shape(self):
    # [FUNCTION DOC]
    # Read-only property returning the stored shape object.
    """Returns the `TensorShape` that represents the shape of the tensor."""
    return self._shape

  @property
  def dtype(self):
    # [FUNCTION DOC]
    # Read-only property returning the stored element data type.
    """Returns the `dtype` of elements in the tensor."""
    return self._dtype

  @property
  def name(self):
    # [FUNCTION DOC]
    # Read-only property returning the optional name (null if none was given).
    """Returns the (optionally provided) name of the described tensor."""
    return self._name

  def is_compatible_with(self, spec_or_value):
    # [FUNCTION DOC]
    # Checks compatibility. Input: spec_or_value, another dense spec or a value of this spec's
    # value type. Returns true only if the input has an accepted type, its dtype is compatible
    # with this dtype, and its shape is compatible with this shape. Name is ignored.
    return (isinstance(spec_or_value, (DenseSpec, self.value_type)) and
            self._dtype.is_compatible_with(spec_or_value.dtype) and
            self._shape.is_compatible_with(spec_or_value.shape))

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a string with the concrete class name, the shape, the dtype representation and
    # the name representation.
    return "{}(shape={}, dtype={}, name={})".format(
        type(self).__name__, self.shape, repr(self.dtype), repr(self.name))

  def __hash__(self):
    # [FUNCTION DOC]
    # Hash from the shape and dtype only; the name is deliberately excluded, which stays
    # consistent with equality because equal specs share shape and dtype.
    return hash((self._shape, self.dtype))

  def __eq__(self, other):
    # [FUNCTION DOC]
    # Equality: true only when both objects have exactly the same class and equal shape, dtype
    # and name. Input: other, any object.
    # pylint: disable=protected-access
    return (type(self) is type(other) and self._shape == other._shape and
            self._dtype == other._dtype and self._name == other._name)

  def __ne__(self, other):
    # [FUNCTION DOC]
    # Inequality: the negation of equality.
    return not self == other

  def _serialize(self):
    # [FUNCTION DOC]
    # Type-spec serialization hook: returns the triple of shape, dtype and name that can be
    # used to reconstruct the spec.
    return (self._shape, self._dtype, self._name)

  def _to_legacy_output_types(self):
    # [FUNCTION DOC]
    # HELPER: legacy dataset API support. Returns the dtype.
    return self._dtype

  def _to_legacy_output_shapes(self):
    # [FUNCTION DOC]
    # HELPER: legacy dataset API support. Returns the shape.
    return self._shape

  def _to_legacy_output_classes(self):
    # [FUNCTION DOC]
    # HELPER: legacy dataset API support. Returns the value type class this spec describes.
    return self.value_type


@tf_export("TensorSpec")
@type_spec_registry.register("tf.TensorSpec")
class TensorSpec(DenseSpec, type_spec.BatchableTypeSpec,
                 trace_type.Serializable, internal.TensorSpec):
  # [CLASS DOC]
  # Public type specification of a tensor: shape, dtype and optional name. Registered in the
  # type-spec registry under the public name and exported to the public API. Used to constrain
  # function inputs (input signatures), as a trace type for retracing decisions, for batching
  # and unbatching specs, for serialization to and from a proto, and to create graph
  # placeholders. Adds no instance state beyond the dense spec base (empty slots).
  """Describes the type of a tf.Tensor.

  >>> t = tf.constant([[1,2,3],[4,5,6]])
  >>> tf.TensorSpec.from_tensor(t)
  TensorSpec(shape=(2, 3), dtype=tf.int32, name=None)

  Contains metadata for describing the nature of `tf.Tensor` objects
  accepted or returned by some TensorFlow APIs.

  For example, it can be used to constrain the type of inputs accepted by
  a tf.function:

  >>> @tf.function(input_signature=[tf.TensorSpec([1, None])])
  ... def constrained_foo(t):
  ...   print("tracing...")
  ...   return t

  Now the `tf.function` is able to assume that `t` is always of the type
  `tf.TensorSpec([1, None])` which will avoid retracing as well as enforce the
  type restriction on inputs.

  As a result, the following call with tensor of type `tf.TensorSpec([1, 2])`
  triggers a trace and succeeds:
  >>> constrained_foo(tf.constant([[1., 2]])).numpy()
  tracing...
  array([[1., 2.]], dtype=float32)

  The following subsequent call with tensor of type `tf.TensorSpec([1, 4])`
  does not trigger a trace and succeeds:
  >>> constrained_foo(tf.constant([[1., 2, 3, 4]])).numpy()
  array([[1., 2., 3., 4.], dtype=float32)

  But the following call with tensor of type `tf.TensorSpec([2, 2])` fails:
  >>> constrained_foo(tf.constant([[1., 2], [3, 4]])).numpy()
  Traceback (most recent call last):
  ...
  TypeError: Binding inputs to tf.function `constrained_foo` failed ...

  """

  __slots__ = []

  @classmethod
  def experimental_type_proto(cls) -> Type[struct_pb2.TensorSpecProto]:
    # [FUNCTION DOC]
    # Class method for serialization support: returns the proto message class used to
    # serialize tensor specs.
    """Returns the type of proto associated with TensorSpec serialization."""
    return struct_pb2.TensorSpecProto

  @classmethod
  def experimental_from_proto(
      cls, proto: struct_pb2.TensorSpecProto) -> "TensorSpec":
    # [FUNCTION DOC]
    # Class method that rebuilds a tensor spec from its serialized proto. Input: proto, a tensor
    # spec proto. Returns a new spec with the shape decoded from the proto, the proto's dtype,
    # and the name, or null when the proto's name is empty.
    """Returns a TensorSpec instance based on the serialized proto."""
    return TensorSpec(
        shape=tensor_shape.TensorShape.experimental_from_proto(proto.shape),
        dtype=proto.dtype,
        name=proto.name if proto.name else None)

  def experimental_as_proto(self) -> struct_pb2.TensorSpecProto:
    # [FUNCTION DOC]
    # Serializes this spec. Returns a tensor spec proto holding the shape proto, the dtype's
    # enumeration value and the name (a null name is passed through to the proto field, which
    # leaves it empty or may fail depending on the proto library).
    """Returns a proto representation of the TensorSpec instance."""
    return struct_pb2.TensorSpecProto(
        shape=self.shape.experimental_as_proto(),
        dtype=self.dtype.experimental_as_proto().datatype,
        name=self.name)

  def is_compatible_with(self, spec_or_tensor):  # pylint:disable=useless-super-delegation,arguments-renamed
    # [FUNCTION DOC]
    # Compatibility check that simply delegates to the dense spec base implementation. Input:
    # spec_or_tensor, a tensor spec or a tensor. Returns true when dtypes are compatible and
    # shapes are compatible; names are ignored.
    """Returns True if spec_or_tensor is compatible with this TensorSpec.

    Two tensors are considered compatible if they have the same dtype
    and their shapes are compatible (see `tf.TensorShape.is_compatible_with`).

    Args:
      spec_or_tensor: A tf.TensorSpec or a tf.Tensor

    Returns:
      True if spec_or_tensor is compatible with self.
    """
    return super(TensorSpec, self).is_compatible_with(spec_or_tensor)

  def is_subtype_of(self, other):
    # [FUNCTION DOC]
    # Trace-type subtype test. Input: other, any object. Returns false unless other is a tensor
    # spec. Otherwise true when this spec has no name or the same name as other, its shape is a
    # subtype of other's shape and its dtype is a subtype of other's dtype.
    if not isinstance(other, TensorSpec):
      return False

    return (
        (not self.name or self.name == other.name)
        and self.shape.is_subtype_of(other.shape)
        and self.dtype.is_subtype_of(other.dtype)
    )

  def placeholder_value(self, placeholder_context):
    # [FUNCTION DOC]
    # Creates a graph placeholder for this spec while tracing a function. Input:
    # placeholder_context, describing the target graph, naming scope, whether to only unnest,
    # whether to clear control dependencies, and an optional composite device name. If only
    # unnesting is requested, returns the spec itself. Otherwise creates a placeholder (outside
    # control flow when requested) named by the spec name or the naming scope, records the
    # user-specified name as an attribute on the op, copies resource/variant handle data onto the
    # placeholder when set, and records a composite device attribute when provided. Returns the
    # placeholder tensor. Side effects: adds operations and attributes to the context graph.
    """Generates a graph placeholder with the given TensorSpec information."""
    if placeholder_context.unnest_only:
      return self

    name = self.name or placeholder_context.naming_scope
    context_graph = placeholder_context.context_graph
    if placeholder_context.with_none_control_dependencies:
      # Note: setting ops.control_dependencies(None) ensures we always put
      # capturing placeholders outside of any control flow context.
      with context_graph.control_dependencies(None):
        placeholder = self._graph_placeholder(context_graph, name=name)
    else:
      placeholder = self._graph_placeholder(context_graph, name=name)

    if name is not None:
      # Record the requested/user-specified name in case it's different than
      # the uniquified name, for validation when exporting signatures.
      placeholder.op._set_attr(  # pylint: disable=protected-access
          "_user_specified_name",
          attr_value_pb2.AttrValue(s=compat.as_bytes(name)))

    handle_data = self.dtype._handle_data  # pylint: disable=protected-access
    if (
        handle_data is not None
        and handle_data.shape_inference.is_set
        and handle_data.shape_inference.shape_and_type
    ):
      handle_data_util.set_handle_data(placeholder, handle_data.shape_inference)

    # Record the composite device as an attribute to the placeholder.
    # This attribute would be propagated into the arg_attr of the FunctionDef.
    # Currently, a packed eager tensor is always placed on a CompositeDevice.
    if placeholder_context.composite_device_name is not None:
      placeholder.op._set_attr(  # pylint: disable=protected-access
          "_composite_device",
          attr_value_pb2.AttrValue(s=compat.as_bytes(
              placeholder_context.composite_device_name)))

    return placeholder

  def _graph_placeholder(self, graph, name=None):
    # [FUNCTION DOC]
    # HELPER: used by the placeholder-creation method to add the placeholder operation.
    # Inputs: graph (target graph) and name (optional op name). Builds dtype and shape
    # attributes from the base dtype and shape of this spec and creates a placeholder operation
    # in the graph through its internal op-creation path. If creation fails with a value error
    # (for example the name is not a valid op name), logs a warning and retries without a name.
    # If op callbacks are active, invokes them and uses the first callback output as the
    # result when they return one. Returns the placeholder tensor. Side effects: adds an
    # operation to the graph, may log a warning and may run registered op callbacks.
    """Graph-only version of tf.compat.v1.placeholder(), for internal use only."""
    dtype = self.dtype.base_dtype
    shape = self.shape
    dtype_value = attr_value_pb2.AttrValue(type=dtype.as_datatype_enum)
    if isinstance(shape, (list, tuple)):
      shape = tensor_shape.TensorShape(shape)
    shape = attr_value_pb2.AttrValue(shape=shape.as_proto())
    attrs = {"dtype": dtype_value, "shape": shape}
    try:
      op = graph._create_op_internal(  # pylint: disable=protected-access
          "Placeholder", [], [dtype], input_types=[],
          attrs=attrs, name=name)
    except ValueError as e:
      # TODO(b/262413656) Sometimes parameter names are not valid op names, in
      # which case an unnamed placeholder is created instead. Update this logic
      # to sanitize the name instead of falling back on unnamed placeholders.
      logging.warning(e)
      op = graph._create_op_internal(  # pylint: disable=protected-access
          "Placeholder", [], [dtype], input_types=[], attrs=attrs)
    (result,) = op.outputs
    if op_callbacks.should_invoke_op_callbacks():
      # TODO(b/147670703): Once the special-op creation code paths
      # are unified. Remove this `if` block.
      callback_outputs = op_callbacks.invoke_op_callbacks(
          "Placeholder", tuple(), attrs, tuple(op.outputs),
          op_name=name, graph=graph)
      if callback_outputs is not None:
        (result,) = callback_outputs
    return result

  def to_tensors(self, value):
    # [FUNCTION DOC]
    # Flattens a value into a list of tensors for this spec. Input: value, a tensor or a Python
    # value castable to this spec. Casts it with an internal casting context, then raises a type
    # error if its shape is not a subtype of this spec's shape. Returns a one-element list
    # holding the cast tensor.
    value = self.cast(value, trace_type.InternalCastContext())
    if not value.shape.is_subtype_of(self.shape):
      raise TypeError(
          f"Received tensor of shape {value.shape} instead of {self.shape}"
      )
    return [value]

  def from_tensors(self, tensors):
    # [FUNCTION DOC]
    # Rebuilds a value from flat tensors. Input: tensors, an iterator over tensors; exactly one
    # item is consumed (raises the stop-iteration signal if empty). If the dtype carries
    # resource handle data, copies its shape-inference part onto the tensor (side effect:
    # mutates the tensor's handle data). Returns the tensor.
    tensor = next(tensors)
    handle_data = self.dtype._handle_data  # pylint: disable=protected-access
    if handle_data:
      handle_data_util.set_handle_data(tensor, handle_data.shape_inference)
    return tensor

  def flatten(self):
    # [FUNCTION DOC]
    # Returns a one-element list containing this spec (a tensor spec is already a leaf).
    return [self]

  def cast(self, value, casting_context):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the in-code comment and docstring suggest only Python primitives are
    # converted and dtype changes are unsupported, but a differing-dtype tensor is only rejected
    # indirectly through the subtype check, and the assertion for spec inputs is skipped when
    # assertions are disabled.
    # Casts a value to a tensor that is a subtype of this spec. Inputs: value (tensor, spec or
    # Python/numpy value) and casting_context (supplies an allow-specs flag). If specs are
    # allowed and value is a tensor spec, asserts it is a subtype of this spec and returns this
    # spec. Otherwise non-tensors are converted using this spec's dtype through the conversion
    # registry. Builds a spec from the value's shape and dtype with this name; if that is not a
    # subtype of this spec but this spec is a subtype of it (value shape is less specific), sets
    # the value's shape to this spec's shape (mutating the value's static shape); otherwise
    # raises a type error. Returns the tensor.
    """Cast value to a tensor that is a subtype of this TensorSpec."""
    # This method is mainly used to cast Python primitives to tensor.
    # Currently, cast tensor to tensor with different types are not supported.
    # For example, casting int32 to float32 would raise a ValueError.
    if casting_context.allow_specs and isinstance(value, TensorSpec):
      assert value.is_subtype_of(self), f"Can not cast {value!r} to {self!r}"
      return self

    if not isinstance(value, Tensor):
      value = tensor_conversion_registry.convert(value, self.dtype)
    value_spec = TensorSpec(value.shape, value.dtype, self.name)

    if not value_spec.is_subtype_of(self):
      if self.is_subtype_of(value_spec):
        value.set_shape(self.shape)
      else:
        raise TypeError(f"Can not cast {value_spec!r} to {self!r}")

    return value

  def _alias_id(self):
    # [FUNCTION DOC]
    # HELPER: used to detect duplicate resource tensors when tracing.
    # Returns the alias identifier from the dtype's resource handle data, or the null value when
    # the dtype has no handle data.
    """Returns an id specifying identical tensors to avoid duplication."""
    alias_id = None
    if self.dtype._handle_data:  # pylint: disable=protected-access
      alias_id = self.dtype._handle_data.alias_id  # pylint: disable=protected-access
    return alias_id

  @classmethod
  def from_spec(cls, spec, name=None):
    # [FUNCTION DOC]
    # Class method that copies a spec. Inputs: spec (any type spec with shape, dtype and name)
    # and name (optional; falls back to the spec's name when absent or empty). Returns a new
    # instance of the calling class with the same shape and dtype. Does not copy any bounds.
    """Returns a `TensorSpec` with the same shape and dtype as `spec`.

    >>> spec = tf.TensorSpec(shape=[8, 3], dtype=tf.int32, name="OriginalName")
    >>> tf.TensorSpec.from_spec(spec, "NewName")
    TensorSpec(shape=(8, 3), dtype=tf.int32, name='NewName')

    Args:
      spec: The `TypeSpec` used to create the new `TensorSpec`.
      name: The name for the new `TensorSpec`.  Defaults to `spec.name`.
    """
    return cls(spec.shape, spec.dtype, name or spec.name)

  @classmethod
  def from_tensor(cls, tensor, name=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says the name defaults to the tensor's op name, but that only
    # happens for symbolic tensors; for eager (value) tensors the name stays null when omitted.
    # Class method building a spec describing a tensor. Inputs: tensor (an eager value tensor or
    # symbolic tensor) and name (optional). Eager tensors yield a spec with their shape, dtype
    # and the given name; symbolic ones use the given name or else the op name. Raises a value
    # error for any other input type.
    """Returns a `TensorSpec` that describes `tensor`.

    >>> tf.TensorSpec.from_tensor(tf.constant([1, 2, 3]))
    TensorSpec(shape=(3,), dtype=tf.int32, name=None)

    Args:
      tensor: The `tf.Tensor` that should be described.
      name: A name for the `TensorSpec`.  Defaults to `tensor.op.name`.

    Returns:
      A `TensorSpec` that describes `tensor`.
    """
    if isinstance(tensor, core_tf_types.Value):
      return TensorSpec(tensor.shape, tensor.dtype, name)
    elif isinstance(tensor, core_tf_types.Symbol):
      # TODO(b/249802365): Return a sanitized version of op name or no name.
      return TensorSpec(tensor.shape, tensor.dtype, name or tensor.op.name)
    else:
      raise ValueError(
          f"`tensor` should be a tf.Tensor, but got type {type(tensor)}.")

  @property
  def value_type(self):
    # [FUNCTION DOC]
    # Read-only property: returns the tensor class, the Python type of values this spec describes.
    """The Python type for values that are compatible with this TypeSpec."""
    return Tensor

  def _to_components(self, value):
    # [FUNCTION DOC]
    # Type-spec hook: asserts the value is a tensor (assertion error otherwise, skipped if
    # assertions are disabled) and returns it unchanged as the single component.
    assert isinstance(value, core_tf_types.Tensor)
    return value

  def _from_components(self, components):
    # [FUNCTION DOC]
    # Type-spec hook: returns the components unchanged, since a tensor is its own component.
    return components

  def _from_compatible_tensor_list(self, tensor_list):
    # [FUNCTION DOC]
    # Type-spec hook rebuilding a value from a flat list. Input: tensor_list, which must hold
    # exactly one tensor (asserted). Sets that tensor's static shape to this spec's shape
    # (mutating the tensor) and returns it.
    # TODO(b/112266545): It would be cleaner to create a new `ensure_shape()`
    # op here and return that, instead of mutating the input's shape using
    # `Tensor.set_shape()`. However, that would add extra ops, which could
    # impact performance. When this bug is resolved, we should be able to add
    # the `ensure_shape()` ops and optimize them away using contextual shape
    # information.
    assert len(tensor_list) == 1
    tensor_list[0].set_shape(self._shape)
    return tensor_list[0]

  def _to_batchable_tensor_list(self, value, batched=False):
    # [FUNCTION DOC]
    # Converts a value to its batchable component. Inputs: value (a tensor) and batched (flag,
    # default false, saying the value already carries a batch dimension). Raises a value error
    # when batched is true and the merged shape of this spec and the value has rank zero.
    # Returns the value itself via the components hook (a tensor, not a list).
    if batched and self._shape.merge_with(value.shape).ndims == 0:
      raise ValueError("Unbatching a tensor is only supported for rank >= 1")
    return self._to_components(value)

  def _batch(self, batch_size):
    # [FUNCTION DOC]
    # Returns a new spec with a leading batch dimension of the given size (may be null for
    # unknown) prepended to this shape, keeping the dtype. The name is dropped.
    return TensorSpec(
        tensor_shape.TensorShape([batch_size]).concatenate(self._shape),
        self._dtype)

  def _unbatch(self):
    # [FUNCTION DOC]
    # Returns a new spec with the leading dimension removed, keeping the dtype (name dropped).
    # Raises a value error if the rank is zero.
    if self._shape.ndims == 0:
      raise ValueError("Unbatching a tensor is only supported for rank >= 1")
    return TensorSpec(self._shape[1:], self._dtype)

  @property
  def _flat_tensor_specs(self):
    # [FUNCTION DOC]
    # Read-only property: a one-element list containing this spec.
    return [self]

  def _to_tensor_list(self, value):
    # [FUNCTION DOC]
    # Input: value, a tensor. Returns a one-element list holding it (asserts it is a tensor).
    return [self._to_components(value)]

  def _to_batched_tensor_list(self, value):
    # [FUNCTION DOC]
    # Same as the plain tensor-list conversion: returns a one-element list with the tensor.
    return self._to_tensor_list(value)

  # TODO(b/206014848): Helper function to support logic that does not consider
  # Tensor name. Will be removed once load-bearing usages of Tensor name are
  # fixed.
  def _without_tensor_names(self) -> "TensorSpec":
    # [FUNCTION DOC]
    # HELPER: temporary support for logic that ignores tensor names.
    # Returns this spec if it has no name, otherwise a new spec with the same shape and dtype
    # and no name.
    """Returns a version of `TensorSpec` with the name removed."""
    if self.name is None:
      return self
    else:
      return TensorSpec(self.shape, self.dtype)

trace_type.register_serializable(TensorSpec)
trace_type.register_tensor_type(TensorSpec)


class _TensorSpecCodec:
  # [CLASS DOC]
  # HELPER: codec plugged into the nested-structure coder to serialize and deserialize plain
  # tensor specs to and from the structured-value proto. Stateless; one instance is registered at
  # import time. Bounded specs are excluded because they have their own codec.
  """Codec for `TensorSpec`."""

  def can_encode(self, pyobj):
    # [FUNCTION DOC]
    # Returns true if the object is a tensor spec that is not a bounded tensor spec.
    # BoundedTensorSpec has its own decoder.
    return (isinstance(pyobj, TensorSpec) and
            not isinstance(pyobj, BoundedTensorSpec))

  def do_encode(self, tensor_spec_value, encode_fn):
    # [FUNCTION DOC]
    # Encodes a tensor spec. Inputs: tensor_spec_value (the spec) and encode_fn (callback that
    # encodes nested values such as shape and dtype). Returns a structured-value proto whose
    # tensor-spec field holds the encoded shape, dtype and name.
    encoded_tensor_spec = struct_pb2.StructuredValue()
    encoded_tensor_spec.tensor_spec_value.CopyFrom(
        struct_pb2.TensorSpecProto(
            shape=encode_fn(tensor_spec_value.shape).tensor_shape_value,
            dtype=encode_fn(tensor_spec_value.dtype).tensor_dtype_value,
            name=tensor_spec_value.name))
    return encoded_tensor_spec

  def can_decode(self, value):
    # [FUNCTION DOC]
    # Returns true if the structured-value proto has its tensor-spec field set.
    return value.HasField("tensor_spec_value")

  def do_decode(self, value, decode_fn):
    # [FUNCTION DOC]
    # Decodes a structured-value proto into a tensor spec. Inputs: value (the proto) and
    # decode_fn (callback that decodes nested shape and dtype). An empty stored name becomes a
    # null name. Returns the new spec.
    name = value.tensor_spec_value.name
    return TensorSpec(
        shape=decode_fn(
            struct_pb2.StructuredValue(
                tensor_shape_value=value.tensor_spec_value.shape)),
        dtype=decode_fn(
            struct_pb2.StructuredValue(
                tensor_dtype_value=value.tensor_spec_value.dtype)),
        name=(name if name else None))


nested_structure_coder.register_codec(_TensorSpecCodec())


# TODO(b/133606651): Should is_compatible_with should check min/max bounds?
@type_spec_registry.register("tf.BoundedTensorSpec")
class BoundedTensorSpec(TensorSpec, trace_type.Serializable):
  # [CLASS DOC]
  # A tensor spec that additionally carries inclusive minimum and maximum bounds, stored as
  # read-only numpy arrays of the spec's dtype (slots). Registered in the type-spec registry
  # and serializable through its own proto and codec. Bounds are informational: compatibility
  # checks do not consider them, but equality compares them with numeric closeness while the
  # hash ignores them.
  """A `TensorSpec` that specifies minimum and maximum values.

  Example usage:
  ```python
  spec = tensor_spec.BoundedTensorSpec((1, 2, 3), tf.float32, 0, (5, 5, 5))
  tf_minimum = tf.convert_to_tensor(spec.minimum, dtype=spec.dtype)
  tf_maximum = tf.convert_to_tensor(spec.maximum, dtype=spec.dtype)
  ```

  Bounds are meant to be inclusive. This is especially important for
  integer types. The following spec will be satisfied by tensors
  with values in the set {0, 1, 2}:
  ```python
  spec = tensor_spec.BoundedTensorSpec((3, 5), tf.int32, 0, 2)
  ```
  """

  __slots__ = ("_minimum", "_maximum")

  def __init__(self, shape, dtype, minimum, maximum, name=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: shape and dtype (as for the base spec), minimum and maximum (numbers
    # or sequences of inclusive bounds, broadcastable to the shape), name (optional). Calls the
    # base constructor, then raises a value error if either bound is null or its shape cannot
    # broadcast with the spec shape. Stores each bound as a numpy array of the spec's dtype with
    # writing disabled.
    """Initializes a new `BoundedTensorSpec`.

    Args:
      shape: Value convertible to `tf.TensorShape`. The shape of the tensor.
      dtype: Value convertible to `tf.DType`. The type of the tensor values.
      minimum: Number or sequence specifying the minimum element bounds
        (inclusive). Must be broadcastable to `shape`.
      maximum: Number or sequence specifying the maximum element bounds
        (inclusive). Must be broadcastable to `shape`.
      name: Optional string containing a semantic name for the corresponding
        array. Defaults to `None`.

    Raises:
      ValueError: If `minimum` or `maximum` are not provided or not
        broadcastable to `shape`.
      TypeError: If the shape is not an iterable or if the `dtype` is an invalid
        numpy dtype.
    """
    super(BoundedTensorSpec, self).__init__(shape, dtype, name)

    if minimum is None:
      raise ValueError("`minimum` can not be None.")
    if maximum is None:
      raise ValueError("`maximum` can not be None.")

    try:
      minimum_shape = np.shape(minimum)
      common_shapes.broadcast_shape(
          tensor_shape.TensorShape(minimum_shape), self.shape)
    except ValueError as exception:
      raise ValueError(
          f"`minimum` {minimum} is not compatible with shape {self.shape}."
      ) from exception

    try:
      maximum_shape = np.shape(maximum)
      common_shapes.broadcast_shape(
          tensor_shape.TensorShape(maximum_shape), self.shape)
    except ValueError as exception:
      raise ValueError(
          f"`maximum` {maximum} is not compatible with shape {self.shape}."
      ) from exception

    self._minimum = np.array(minimum, dtype=self.dtype.as_numpy_dtype)
    self._minimum.setflags(write=False)

    self._maximum = np.array(maximum, dtype=self.dtype.as_numpy_dtype)
    self._maximum.setflags(write=False)

  @classmethod
  def experimental_type_proto(cls) -> Type[struct_pb2.BoundedTensorSpecProto]:
    # [FUNCTION DOC]
    # Class method: returns the proto message class used to serialize bounded tensor specs.
    """Returns the type of proto associated with BoundedTensorSpec serialization."""
    return struct_pb2.BoundedTensorSpecProto

  @classmethod
  def experimental_from_proto(
      cls, proto: struct_pb2.BoundedTensorSpecProto) -> "BoundedTensorSpec":
    # [FUNCTION DOC]
    # Class method rebuilding a bounded spec from its proto. Input: proto, a bounded spec proto.
    # Decodes the shape, dtype, name (empty becomes null) and converts the minimum and maximum
    # tensor protos to numpy arrays. Returns the new spec.
    """Returns a BoundedTensorSpec instance based on the serialized proto."""
    return BoundedTensorSpec(
        shape=tensor_shape.TensorShape.experimental_from_proto(proto.shape),
        dtype=proto.dtype,
        minimum=tensor_util.MakeNdarray(proto.minimum),
        maximum=tensor_util.MakeNdarray(proto.maximum),
        name=proto.name if proto.name else None)

  def experimental_as_proto(self) -> struct_pb2.BoundedTensorSpecProto:
    # [FUNCTION DOC]
    # Serializes this spec into a bounded spec proto holding the shape proto, the dtype's
    # enumeration value, the name, and the minimum and maximum bounds as tensor protos.
    """Returns a proto representation of the BoundedTensorSpec instance."""
    return struct_pb2.BoundedTensorSpecProto(
        shape=self.shape.experimental_as_proto(),
        dtype=self.dtype.experimental_as_proto().datatype,
        minimum=tensor_util.make_tensor_proto(self._minimum),
        maximum=tensor_util.make_tensor_proto(self._maximum),
        name=self.name)

  @classmethod
  def from_spec(cls, spec):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the signature differs from the parent class method (no name parameter), and
    # the docstring says a plain tensor spec is returned although a bounded spec is built. It
    # also ignores the calling class and always constructs this exact class.
    # Class method that copies a spec into a bounded spec. Input: spec, any type spec. Uses its
    # minimum and maximum if present, otherwise the smallest and largest values of its dtype, and
    # keeps its shape, dtype and name. Returns the new bounded spec.
    """Returns a `TensorSpec` with the same shape and dtype as `spec`.

    If `spec` is a `BoundedTensorSpec`, then the new spec's bounds are set to
    `spec.minimum` and `spec.maximum`; otherwise, the bounds are set to
    `spec.dtype.min` and `spec.dtype.max`.

    >>> spec = tf.TensorSpec(shape=[8, 3], dtype=tf.int32, name="x")
    >>> BoundedTensorSpec.from_spec(spec)
    BoundedTensorSpec(shape=(8, 3), dtype=tf.int32, name='x',
        minimum=array(-2147483648, dtype=int32),
        maximum=array(2147483647, dtype=int32))

    Args:
      spec: The `TypeSpec` used to create the new `BoundedTensorSpec`.
    """
    dtype = dtypes.as_dtype(spec.dtype)
    minimum = getattr(spec, "minimum", dtype.min)
    maximum = getattr(spec, "maximum", dtype.max)
    return BoundedTensorSpec(spec.shape, dtype, minimum, maximum, spec.name)

  @property
  def minimum(self):
    # [FUNCTION DOC]
    # Read-only property returning the stored read-only numpy array of inclusive lower bounds.
    """Returns a NumPy array specifying the minimum bounds (inclusive)."""
    return self._minimum

  @property
  def maximum(self):
    # [FUNCTION DOC]
    # Read-only property returning the stored read-only numpy array of inclusive upper bounds.
    """Returns a NumPy array specifying the maximum bounds (inclusive)."""
    return self._maximum

  def cast(self, value, casting_context):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the bounds are not checked when casting; values outside the minimum and
    # maximum are accepted.
    # Casts a value to a tensor compatible with this spec. Inputs: value and casting_context
    # (with an allow-specs flag). If specs are allowed and value is a bounded spec, asserts it is
    # a subtype of this spec and returns this spec. Otherwise delegates to the plain tensor spec
    # cast using this shape, dtype and name, and returns its result (which may raise a type error).
    if casting_context.allow_specs and isinstance(value, BoundedTensorSpec):
      assert value.is_subtype_of(self), f"Can not cast {value!r} to {self!r}"
      return self

    actual_spec = TensorSpec(shape=self.shape, dtype=self.dtype, name=self.name)
    return actual_spec.cast(value, casting_context)  # pylint: disable=protected-access

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a string with the class name, shape, dtype, name, minimum and maximum.
    s = "BoundedTensorSpec(shape={}, dtype={}, name={}, minimum={}, maximum={})"
    return s.format(self.shape, repr(self.dtype), repr(self.name),
                    repr(self.minimum), repr(self.maximum))

  def __eq__(self, other):
    # [FUNCTION DOC]
    # Equality: requires the base spec equality (same class, shape, dtype, name) and that both
    # minimum and maximum arrays are numerically close (tolerance-based, not exact). Input:
    # other, any object (bounds are accessed only after the base check passes).
    tensor_spec_eq = super(BoundedTensorSpec, self).__eq__(other)
    return (tensor_spec_eq and np.allclose(self.minimum, other.minimum) and
            np.allclose(self.maximum, other.maximum))

  def __hash__(self):
    # [FUNCTION DOC]
    # Hash from shape and dtype only; the name and the bounds are excluded, which is consistent
    # with equality (equal specs have equal shape and dtype).
    return hash((self._shape, self.dtype))

  def __reduce__(self):
    # [FUNCTION DOC]
    # Pickling support: returns the class and the constructor arguments (shape, dtype, minimum,
    # maximum, name) needed to recreate this spec.
    return BoundedTensorSpec, (self._shape, self._dtype, self._minimum,
                               self._maximum, self._name)

  def _serialize(self):
    # [FUNCTION DOC]
    # Type-spec serialization hook: returns shape, dtype, minimum, maximum and name.
    return (self._shape, self._dtype, self._minimum, self._maximum, self._name)


class _BoundedTensorSpecCodec:
  # [CLASS DOC]
  # HELPER: codec plugged into the nested-structure coder to serialize and deserialize bounded
  # tensor specs to and from the structured-value proto. Stateless; one instance is registered at
  # import time.
  """Codec for `BoundedTensorSpec`."""

  def can_encode(self, pyobj):
    # [FUNCTION DOC]
    # Returns true if the object is a bounded tensor spec.
    return isinstance(pyobj, BoundedTensorSpec)

  def do_encode(self, bounded_tensor_spec_value, encode_fn):
    # [FUNCTION DOC]
    # Encodes a bounded spec. Inputs: bounded_tensor_spec_value (the spec) and encode_fn
    # (callback encoding shape and dtype). Returns a structured-value proto whose bounded spec
    # field holds the encoded shape, dtype, name and the minimum and maximum as tensor protos.
    """Returns an encoded proto for the given `tf.BoundedTensorSpec`."""
    encoded_bounded_tensor_spec = struct_pb2.StructuredValue()
    encoded_bounded_tensor_spec.bounded_tensor_spec_value.CopyFrom(
        struct_pb2.BoundedTensorSpecProto(
            shape=encode_fn(bounded_tensor_spec_value.shape).tensor_shape_value,
            dtype=encode_fn(bounded_tensor_spec_value.dtype).tensor_dtype_value,
            name=bounded_tensor_spec_value.name,
            minimum=tensor_util.make_tensor_proto(
                bounded_tensor_spec_value.minimum),
            maximum=tensor_util.make_tensor_proto(
                bounded_tensor_spec_value.maximum)))
    return encoded_bounded_tensor_spec

  def can_decode(self, value):
    # [FUNCTION DOC]
    # Returns true if the structured-value proto has its bounded tensor spec field set.
    return value.HasField("bounded_tensor_spec_value")

  def do_decode(self, value, decode_fn):
    # [FUNCTION DOC]
    # Decodes a structured-value proto into a bounded spec. Inputs: value (the proto) and
    # decode_fn (callback decoding shape and dtype). Converts bound protos to numpy arrays; an
    # empty name becomes null. Returns the new spec.
    btsv = value.bounded_tensor_spec_value
    name = btsv.name
    return BoundedTensorSpec(
        shape=decode_fn(
            struct_pb2.StructuredValue(tensor_shape_value=btsv.shape)),
        dtype=decode_fn(
            struct_pb2.StructuredValue(tensor_dtype_value=btsv.dtype)),
        minimum=tensor_util.MakeNdarray(btsv.minimum),
        maximum=tensor_util.MakeNdarray(btsv.maximum),
        name=(name if name else None))


nested_structure_coder.register_codec(_BoundedTensorSpecCodec())

trace_type.register_serializable(BoundedTensorSpec)

# Note: we do not include Tensor names when constructing TypeSpecs.
type_spec.register_type_spec_from_value_converter(
    Tensor, lambda tensor: TensorSpec(tensor.shape, tensor.dtype))

type_spec.register_type_spec_from_value_converter(
    np.ndarray, lambda array: TensorSpec(array.shape, array.dtype))
