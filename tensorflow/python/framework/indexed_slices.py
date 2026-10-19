# [MODULE DOC]
# Defines IndexedSlices, TensorFlow's sparse representation of a set of tensor slices (values plus
# first-dimension indices plus an optional dense shape), typically produced as gradients of gather-like
# ops; its gradient adapter, its type specification (IndexedSlicesSpec), a plain named-tuple value
# form used with numpy data in graph mode, and conversion helpers that turn arbitrary objects into
# tensors or leave IndexedSlices untouched, including densification with a large-size warning.
# Import-time side effects: registers the IndexedSlicesSpec codec with the nested structure coder and
# registers a tensor-conversion function so IndexedSlices can be converted to dense tensors.
# Several names are exported under public TensorFlow names.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Indexed slices."""

# pylint: disable=g-bad-name
import collections
import warnings

import numpy as np

from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import composite_tensor_gradient
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.types import internal
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export


class IndexedSlicesCompositeTensorGradient(
    composite_tensor_gradient.CompositeTensorGradient):
  # [CLASS DOC]
  # Gradient adapter telling the composite-tensor gradient machinery how to treat IndexedSlices: the
  # IndexedSlices object itself is the single gradient component (it is not split into its values,
  # indices and shape tensors). A single stateless instance is attached to IndexedSlices.
  """CompositeTensorGradient for IndexedSlices."""

  def get_gradient_components(self, value):
    # [FUNCTION DOC]
    # Returns the given composite value (an IndexedSlices) unchanged, so it acts as its own sole
    # gradient component. No validation, no side effects.
    return value

  def replace_gradient_components(self, value, component_grads):
    # [FUNCTION DOC]
    # Rebuilds a composite value from gradient components. Ignores the original value and returns
    # component_grads as is, since the gradient of an IndexedSlices is already a single object.
    # No validation, no side effects.
    return component_grads


# TODO(mdan): Should IndexedSlices be a "tensor"?
@tf_export("IndexedSlices")
class IndexedSlices(
    internal.IndexedSlices,
    internal.NativeObject,
    composite_tensor.CompositeTensor):
  # [CLASS DOC]
  # Sparse representation of a subset of the slices (along the first dimension) of a larger dense
  # tensor: holds a values tensor, a one-dimensional integer indices tensor, and an optional dense
  # shape tensor describing the full tensor. Mostly used as the gradient of gather-like ops. It is
  # a composite tensor (its type spec is an IndexedSlicesSpec) and a native-object tensor-like type,
  # exported publicly. Instances only hold references to the three tensors in private attributes and
  # delegate name, device, op, dtype and graph queries to the values tensor. Not a real tensor, so
  # it does no validation of the components at construction. Class attribute with the composite
  # gradient adapter tells gradient code to treat the whole object as one component.
  """A sparse representation of a set of tensor slices at given indices.

  This class is a simple wrapper for a pair of `Tensor` objects:

  * `values`: A `Tensor` of any dtype with shape `[D0, D1, ..., Dn]`.
  * `indices`: A 1-D integer `Tensor` with shape `[D0]`.

  An `IndexedSlices` is typically used to represent a subset of a larger
  tensor `dense` of shape `[LARGE0, D1, .. , DN]` where `LARGE0 >> D0`.
  The values in `indices` are the indices in the first dimension of
  the slices that have been extracted from the larger tensor.

  The dense tensor `dense` represented by an `IndexedSlices` `slices` has

  ```python
  dense[slices.indices[i], :, :, :, ...] = slices.values[i, :, :, :, ...]
  ```

  The `IndexedSlices` class is used principally in the definition of
  gradients for operations that have sparse gradients
  (e.g. `tf.gather`).

  >>> v = tf.Variable([[0.,1, 2], [2, 3, 4], [4, 5, 6], [6, 7, 8]])
  >>> with tf.GradientTape() as tape:
  ...   r = tf.gather(v, [1,3])
  >>> index_slices = tape.gradient(r,v)
  >>> index_slices
  <...IndexedSlices object ...>
  >>> index_slices.indices.numpy()
  array([1, 3], dtype=int32)
  >>> index_slices.values.numpy()
  array([[1., 1., 1.],
         [1., 1., 1.]], dtype=float32)

  Contrast this representation with
  `tf.sparse.SparseTensor`,
  which uses multi-dimensional indices and scalar values.
  """

  def __init__(self, values, indices, dense_shape=None):
    # [FUNCTION DOC]
    # Stores the three components without validating or converting them. values is a tensor of any
    # type holding the slice contents, indices a one-dimensional integer tensor of first-dimension
    # positions, and dense_shape an optional tensor with the shape of the corresponding dense tensor
    # (null if unknown). Returns nothing; raises nothing.
    """Creates an `IndexedSlices`."""
    self._values = values
    self._indices = indices
    self._dense_shape = dense_shape

  @property
  def values(self):
    # [FUNCTION DOC]
    # Read-only property returning the stored values tensor (the slice contents).
    """A `Tensor` containing the values of the slices."""
    return self._values

  @property
  def indices(self):
    # [FUNCTION DOC]
    # Read-only property returning the stored indices tensor (positions along the first dimension).
    """A 1-D `Tensor` containing the indices of the slices."""
    return self._indices

  @property
  def dense_shape(self):
    # [FUNCTION DOC]
    # Read-only property returning the stored dense shape tensor, or null when it was not provided.
    """A 1-D `Tensor` containing the shape of the corresponding dense tensor."""
    return self._dense_shape

  @property
  def shape(self):
    # [FUNCTION DOC]
    # Read-only property giving the static shape of the corresponding dense tensor. If no dense
    # shape tensor was given, returns a fully unknown shape; otherwise tries to evaluate the dense
    # shape tensor statically into a shape object, with unknown dimensions where the value cannot
    # be determined at graph-construction time.
    """Gets the `tf.TensorShape` representing the shape of the dense tensor.

    Returns:
      A `tf.TensorShape` object.
    """
    if self._dense_shape is None:
      return tensor_shape.TensorShape(None)

    return tensor_util.constant_value_as_shape(self._dense_shape)

  @property
  def name(self):
    # [FUNCTION DOC]
    # Read-only property returning the name of the values tensor (this object has no name of its
    # own). Fails with the values tensor's own error if it has no name, such as in eager mode.
    """The name of this `IndexedSlices`."""
    return self.values.name

  @property
  def device(self):
    # [FUNCTION DOC]
    # Read-only property returning the device name of the values tensor (a string, or null/empty if
    # unplaced).
    """The name of the device on which `values` will be produced, or `None`."""
    return self.values.device

  @property
  def op(self) -> ops.Operation:
    # [FUNCTION DOC]
    # Read-only property returning the graph operation that produces the values tensor. Only the
    # values tensor is considered; the indices and dense shape may come from other operations.
    """The `Operation` that produces `values` as an output."""
    return self.values.op

  @property
  def dtype(self):
    # [FUNCTION DOC]
    # Read-only property returning the element data type of the values tensor (not of the indices).
    """The `DType` of elements in this tensor."""
    return self.values.dtype

  @property
  def graph(self) -> ops.Graph:
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it is the graph containing values, indices and shape tensors,
    # but the code only returns the graph of the values tensor and does not check the others.
    # Read-only property returning the graph of the values tensor.
    """The `Graph` that contains the values, indices, and shape tensors."""
    return self._values.graph

  def __str__(self):
    # [FUNCTION DOC]
    # Returns a human-readable string showing the indices, the values and, only when it is not
    # null, the dense shape, each formatted with its own string form.
    return "IndexedSlices(indices=%s, values=%s%s)" % (
        self._indices, self._values,
        (", dense_shape=%s" %
         (self._dense_shape,)) if self._dense_shape is not None else "")

  def __neg__(self):
    # [FUNCTION DOC]
    # Unary negation: returns a new IndexedSlices whose values are the negated values tensor, with
    # the same indices and dense shape objects. Does not modify this instance.
    return IndexedSlices(-self.values, self.indices, self.dense_shape)

  __composite_gradient__ = IndexedSlicesCompositeTensorGradient()

  @property
  def _type_spec(self):
    # [FUNCTION DOC]
    # Composite-tensor hook: builds the IndexedSlicesSpec describing this instance. The indices
    # shape is the indices shape merged with the first dimension of the values shape (raises a
    # value error if incompatible). The dense shape starts as unknown first dimension followed by
    # the remaining dimensions of the values shape; if a dense shape tensor exists, it is merged
    # with that tensor's statically known value and its data type is recorded, else the dense shape
    # data type is null. The spec also records the values and indices data types.
    indices_shape = self._indices.shape.merge_with(self._values.shape[:1])
    dense_shape = tensor_shape.TensorShape([None]).concatenate(
        self._values.shape[1:])
    if self._dense_shape is not None:
      dense_shape_dtype = self._dense_shape.dtype
      dense_shape = dense_shape.merge_with(
          tensor_util.constant_value_as_shape(self._dense_shape))
    else:
      dense_shape_dtype = None
    return IndexedSlicesSpec(dense_shape, self.dtype, self._indices.dtype,
                             dense_shape_dtype, indices_shape)

  def _shape_invariant_to_type_spec(self, shape):
    # [FUNCTION DOC]
    # Composite-tensor hook used by while-loops: converts a shape invariant supplied for the values
    # tensor into a type spec. Indices shape is the first dimension of the given shape; dense shape
    # is an unknown first dimension followed by the remaining dimensions of the given shape. The
    # dense shape data type is taken from the current dense shape tensor, or null if there is none.
    # Returns an IndexedSlicesSpec with this instance's values and indices data types.
    # From tf.while_loop docs: "If a loop variable is an IndexedSlices, the
    # shape invariant must be a shape invariant of the values tensor of the
    # IndexedSlices. It means the shapes of the three tensors of the
    # IndexedSlices are (shape, [shape[0]], [shape.ndims])."
    indices_shape = shape[:1]
    dense_shape = tensor_shape.TensorShape([None]).concatenate(shape[1:])
    if self._dense_shape is None:
      dense_shape_dtype = None
    else:
      dense_shape_dtype = self._dense_shape.dtype
    return IndexedSlicesSpec(dense_shape, self.dtype, self._indices.dtype,
                             dense_shape_dtype, indices_shape)

  def consumers(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the internal consumers helper called here is not defined in this class; if no base
    # class supplies it this raises an attribute error, so behavior depends on inherited code.
    # Returns the consumers computed by that internal helper, which must come from a parent class.
    return self._consumers()


IndexedSlicesValue = collections.namedtuple(
    "IndexedSlicesValue", ["values", "indices", "dense_shape"])


@tf_export("IndexedSlicesSpec")
class IndexedSlicesSpec(type_spec.TypeSpec):
  # [CLASS DOC]
  # Type specification for IndexedSlices (exported publicly). Records the dense shape, the data
  # types of the values, indices and optional dense-shape components, and the shape of the indices
  # component, all in private slots. Its value type is IndexedSlices. It describes component specs
  # (values, indices and, if a dense shape data type is set, the dense shape vector) and converts
  # between IndexedSlices and tuples of component tensors. Registered with the nested structure
  # coder at import time.
  """Type specification for a `tf.IndexedSlices`."""

  __slots__ = ["_shape", "_values_dtype", "_indices_dtype",
               "_dense_shape_dtype", "_indices_shape"]

  value_type = property(lambda self: IndexedSlices)

  def __init__(self, shape=None, dtype=dtypes.float32,
               indices_dtype=dtypes.int64, dense_shape_dtype=None,
               indices_shape=None):
    # [FUNCTION DOC]
    # Builds a spec. shape: dense shape (shape-like, or null for unknown, default null).
    # dtype: data type of values (default 32-bit float). indices_dtype: data type of indices
    # (default 64-bit integer; docs say 32 or 64-bit integer but that is not enforced here).
    # dense_shape_dtype: data type of the dense shape tensor, or null if there is none (default).
    # indices_shape: shape-like for the indices component (null means unknown) which is forced to
    # have rank one; raises a value error if its rank differs from one. All inputs are converted to
    # canonical shape and data type objects. Returns nothing.
    """Constructs a type specification for a `tf.IndexedSlices`.

    Args:
      shape: The dense shape of the `IndexedSlices`, or `None` to allow any
        dense shape.
      dtype: `tf.DType` of values in the `IndexedSlices`.
      indices_dtype: `tf.DType` of the `indices` in the `IndexedSlices`.  One
        of `tf.int32` or `tf.int64`.
      dense_shape_dtype: `tf.DType` of the `dense_shape` in the `IndexedSlices`.
        One of `tf.int32`, `tf.int64`, or `None` (if the `IndexedSlices` has
        no `dense_shape` tensor).
      indices_shape: The shape of the `indices` component, which indicates
        how many slices are in the `IndexedSlices`.
    """
    self._shape = tensor_shape.as_shape(shape)
    self._values_dtype = dtypes.as_dtype(dtype)
    self._indices_dtype = dtypes.as_dtype(indices_dtype)
    if dense_shape_dtype is None:
      self._dense_shape_dtype = None
    else:
      self._dense_shape_dtype = dtypes.as_dtype(dense_shape_dtype)
    self._indices_shape = tensor_shape.as_shape(indices_shape).with_rank(1)

  def _serialize(self):
    # [FUNCTION DOC]
    # Returns the spec's defining state as a tuple: dense shape, values data type, indices data
    # type, dense shape data type (possibly null), and indices shape. Used for equality, hashing
    # and serialization.
    return (self._shape, self._values_dtype, self._indices_dtype,
            self._dense_shape_dtype, self._indices_shape)

  @property
  def _component_specs(self):
    # [FUNCTION DOC]
    # Returns a tuple of tensor specs for the components: values (shape is the indices shape
    # followed by the dense shape minus its first dimension, with the values data type), indices
    # (the indices shape and indices data type), and, only when a dense shape data type is set, a
    # vector spec whose length is the rank of the dense shape (may be unknown) with that data type.
    value_shape = self._indices_shape.concatenate(self._shape[1:])
    specs = [
        tensor_spec.TensorSpec(value_shape, self._values_dtype),
        tensor_spec.TensorSpec(self._indices_shape, self._indices_dtype)]
    if self._dense_shape_dtype is not None:
      specs.append(
          tensor_spec.TensorSpec([self._shape.ndims], self._dense_shape_dtype))
    return tuple(specs)

  def _to_components(self, value):
    # [FUNCTION DOC]
    # Splits an IndexedSlices (or value tuple with the same fields) into a tuple of its component
    # tensors: values and indices, plus the dense shape only when it is not null.
    if value.dense_shape is None:
      return (value.values, value.indices)
    else:
      return (value.values, value.indices, value.dense_shape)

  def _from_components(self, tensor_list):
    # [FUNCTION DOC]
    # Rebuilds a value from a sequence of two or three components. If every component is a numpy
    # array and the TF2 behavior is not enabled, returns a plain named-tuple IndexedSlicesValue
    # (using a null dense shape when only two components are given); otherwise returns a real
    # IndexedSlices built from the components in order.
    if (all(isinstance(t, np.ndarray) for t in tensor_list) and
        not tf2.enabled()):
      if len(tensor_list) == 2:
        return IndexedSlicesValue(tensor_list[0], tensor_list[1], None)
      else:
        return IndexedSlicesValue(*tensor_list)
    else:
      return IndexedSlices(*tensor_list)


nested_structure_coder.register_codec(
    nested_structure_coder.BuiltInTypeSpecCodec(
        IndexedSlicesSpec, struct_pb2.TypeSpecProto.INDEXED_SLICES_SPEC
    )
)


@tf_export(v1=["convert_to_tensor_or_indexed_slices"])
def convert_to_tensor_or_indexed_slices(value, dtype=None, name=None):
  # [FUNCTION DOC]
  # Public version-1 API. Converts value to a tensor unless it is already an IndexedSlices or sparse
  # tensor, which are returned unchanged. dtype optionally requires a data type; name optionally
  # names a newly created tensor. Returns a tensor, IndexedSlices or sparse tensor. Raises a value
  # error if dtype is incompatible with the existing IndexedSlices value. Thin wrapper over the
  # internal version with reference-style output turned off.
  """Converts the given object to a `Tensor` or an `IndexedSlices`.

  If `value` is an `IndexedSlices` or `SparseTensor` it is returned
  unmodified. Otherwise, it is converted to a `Tensor` using
  `convert_to_tensor()`.

  Args:
    value: An `IndexedSlices`, `SparseTensor`, or an object that can be consumed
      by `convert_to_tensor()`.
    dtype: (Optional.) The required `DType` of the returned `Tensor` or
      `IndexedSlices`.
    name: (Optional.) A name to use if a new `Tensor` is created.

  Returns:
    A `Tensor`, `IndexedSlices`, or `SparseTensor` based on `value`.

  Raises:
    ValueError: If `dtype` does not match the element type of `value`.
  """
  return internal_convert_to_tensor_or_indexed_slices(
      value=value, dtype=dtype, name=name, as_ref=False)


def internal_convert_to_tensor_or_indexed_slices(value,
                                                 dtype=None,
                                                 name=None,
                                                 as_ref=False):
  # [FUNCTION DOC]
  # HELPER: shared implementation behind the public single-value conversion function.
  # KNOWN ISSUE: the docstring says only IndexedSlices or sparse tensors are returned unmodified,
  # but any native-object value, including ordinary tensors, is passed through (after only a dtype
  # compatibility check); a todo in the code also questions this broad behavior.
  # Behavior by case: an eager tensor seen while not executing eagerly (graph building) is converted
  # to a graph tensor; any native-object value is returned unmodified after, when dtype is given,
  # checking compatibility; everything else is converted to a tensor with the given dtype, name and
  # as_ref. Raises a value error if a requested dtype is incompatible with the native-object value.
  # as_ref asks for a reference-type result.
  """Converts the given object to a `Tensor` or an `IndexedSlices`.

  If `value` is an `IndexedSlices` or `SparseTensor` it is returned
  unmodified. Otherwise, it is converted to a `Tensor` using
  `convert_to_tensor()`.

  Args:
    value: An `IndexedSlices`, `SparseTensor`, or an object that can be consumed
      by `convert_to_tensor()`.
    dtype: (Optional.) The required `DType` of the returned `Tensor` or
      `IndexedSlices`.
    name: (Optional.) A name to use if a new `Tensor` is created.
    as_ref: True if the caller wants the results as ref tensors.

  Returns:
    A `Tensor`, `IndexedSlices`, or `SparseTensor` based on `value`.

  Raises:
    ValueError: If `dtype` does not match the element type of `value`.
  """
  if isinstance(value, ops.EagerTensor) and not context.executing_eagerly():
    return ops.convert_to_tensor(value, dtype=dtype, name=name, as_ref=as_ref)
  # TODO(mdan): Name says tensor_or_indexed_slices. So do explicitly just that?
  elif isinstance(value, internal.NativeObject):
    if dtype and not dtypes.as_dtype(dtype).is_compatible_with(value.dtype):
      raise ValueError(
          "Incompatible tensor conversion requested to `dtype` "
          f"{dtypes.as_dtype(dtype).name} for `value` ({value}) with dtype"
          f" {value.dtype.name}.")
    return value
  else:
    return ops.convert_to_tensor(value, dtype=dtype, name=name, as_ref=as_ref)


def internal_convert_n_to_tensor_or_indexed_slices(values,
                                                   dtype=None,
                                                   name=None,
                                                   as_ref=False):
  # [FUNCTION DOC]
  # HELPER: shared implementation behind the public list conversion function.
  # KNOWN ISSUE: the docstring lists a type error for missing converters and a runtime error for bad
  # converter results, but this function itself only raises the not-iterable type error; the others
  # come only from lower layers.
  # Converts every element of the iterable values using the single-value internal conversion,
  # keeping null elements as null. If name is given, the element at position i is named with the
  # name, an underscore and i. dtype and as_ref are forwarded to each conversion. Returns a list.
  # Raises a type error if values is not iterable; conversion errors from elements propagate.
  """Converts `values` to a list of `Tensor` or `IndexedSlices` objects.

  Any `IndexedSlices` or `SparseTensor` objects in `values` are returned
  unmodified.

  Args:
    values: An iterable of `None`, `IndexedSlices`, `SparseTensor`, or objects
      that can be consumed by `convert_to_tensor()`.
    dtype: (Optional.) The required `DType` of the returned `Tensor` or
      `IndexedSlices`.
    name: (Optional.) A name prefix to used when a new `Tensor` is created, in
      which case element `i` will be given the name `name + '_' + i`.
    as_ref: True if the caller wants the results as ref tensors.

  Returns:
    A list of `Tensor`, `IndexedSlices`, `SparseTensor` and/or `None` objects.

  Raises:
    TypeError: If no conversion function is registered for an element in
      `values`.
    RuntimeError: If a registered conversion function returns an invalid
      value.
  """
  if not isinstance(values, collections_abc.Iterable):
    raise TypeError("Argument `values` must be iterable.")
  ret = []
  for i, value in enumerate(values):
    if value is None:
      ret.append(value)
    else:
      n = None if name is None else "%s_%d" % (name, i)
      ret.append(
          internal_convert_to_tensor_or_indexed_slices(
              value, dtype=dtype, name=n, as_ref=as_ref))
  return ret


def convert_n_to_tensor_or_indexed_slices(values, dtype=None, name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it returns Output objects and describes values as a list, but
  # any iterable is accepted and tensors are returned; also the dtype description is garbled.
  # Converts each element of values to a tensor, leaving IndexedSlices, sparse tensors and null
  # elements unchanged. dtype optionally fixes the data type, name is an optional prefix used to
  # name elements as the prefix, an underscore and the position. Returns a list. Raises a type
  # error for non-iterable input. Thin wrapper over the internal version with reference output off.
  """Converts `values` to a list of `Output` or `IndexedSlices` objects.

  Any `IndexedSlices` or `SparseTensor` objects in `values` are returned
  unmodified.

  Args:
    values: A list of `None`, `IndexedSlices`, `SparseTensor`, or objects that
      can be consumed by `convert_to_tensor()`.
    dtype: (Optional.) The required `DType` of the returned `Tensor`
      `IndexedSlices`.
    name: (Optional.) A name prefix to used when a new `Tensor` is created, in
      which case element `i` will be given the name `name + '_' + i`.

  Returns:
    A list of `Tensor`, `IndexedSlices`, and/or `SparseTensor` objects.

  Raises:
    TypeError: If no conversion function is registered for an element in
      `values`.
    RuntimeError: If a registered conversion function returns an invalid
      value.
  """
  return internal_convert_n_to_tensor_or_indexed_slices(
      values=values, dtype=dtype, name=name, as_ref=False)


# Warn the user if we convert a sparse representation to dense with at
# least this number of elements.
_LARGE_SPARSE_NUM_ELEMENTS = 100000000


def _indexed_slices_to_tensor(value, dtype=None, name=None, as_ref=False):
  # [FUNCTION DOC]
  # HELPER: registered as the tensor conversion function for IndexedSlices.
  # KNOWN ISSUE: as_ref is documented as requesting a ref but is deliberately ignored; also dtype
  # must be a real data type object (its compatibility method is called), not a loose type name.
  # Densifies the IndexedSlices by summing values into segments given by indices over the first
  # dimension of the dense shape, producing a dense tensor (optionally named). Raises a value error
  # if dtype is incompatible with the values data type, or if the dense shape is null. In graph
  # mode, if the dense shape can be evaluated statically and has at least one hundred million
  # elements, emits a warning about large memory use. Potentially expensive.
  """Converts an IndexedSlices object `value` to a Tensor.

  NOTE(mrry): This function is potentially expensive.

  Args:
    value: An ops.IndexedSlices object.
    dtype: The dtype of the Tensor to be returned.
    name: Optional name to use for the returned Tensor.
    as_ref: True if a ref is requested.

  Returns:
    A dense Tensor representing the values in the given IndexedSlices.

  Raises:
    ValueError: If the IndexedSlices does not have the same dtype.
  """
  _ = as_ref
  if dtype and not dtype.is_compatible_with(value.dtype):
    raise ValueError(
        f"Incompatible tensor conversion requested to `dtype` {dtype.name} for "
        f"IndexedSlices ({value}) with dtype {value.dtype.name}")
  if value.dense_shape is None:
    raise ValueError(
        "Tensor conversion requested for IndexedSlices for argument `value` "
        f"without dense_shape: {value!s}")
  # TODO(mrry): Consider adding static shape information to
  # IndexedSlices, to avoid using numpy here.
  if not context.executing_eagerly():
    dense_shape_value = tensor_util.constant_value(value.dense_shape)
    if dense_shape_value is not None:
      num_elements = np.prod(dense_shape_value)
      if num_elements >= _LARGE_SPARSE_NUM_ELEMENTS:
        warnings.warn(
            "Converting sparse IndexedSlices to a dense Tensor with %d "
            "elements. This may consume a large amount of memory." %
            num_elements)
  return gen_math_ops.unsorted_segment_sum(
      value.values, value.indices, value.dense_shape[0], name=name)


tensor_conversion_registry.register_tensor_conversion_function(
    IndexedSlices, _indexed_slices_to_tensor)
