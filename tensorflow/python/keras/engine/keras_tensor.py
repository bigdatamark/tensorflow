# [MODULE DOC]
# Defines KerasTensor, the symbolic placeholder object that flows through Keras Functional API
# construction. A KerasTensor wraps a type specification (plus an optional statically inferred
# value and an optional name) instead of real data. The module also provides specialised
# subclasses for sparse tensors, ragged tensors and legacy user-registered symbolic types, an
# iterator helper, an ordered registry mapping tensor classes to KerasTensor classes, and
# module-level conversion functions (tensor to KerasTensor, type spec to KerasTensor, KerasTensor
# to graph placeholder).
# Import-time side effects: the operator overloads of the base Tensor class are copied onto
# KerasTensor, and a handful of ragged-tensor operators (slicing, add, multiply and their reversed
# forms) are copied onto RaggedKerasTensor. The registry list is created at import time.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Keras Input Tensor used to track functional API Topology."""

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec as type_spec_module
from tensorflow.python.keras.utils import object_identity
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_operators  # pylint: disable=unused-import
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import nest

# pylint: disable=g-classes-have-attributes


# Tensorflow tensors have a maximum rank of 254
# (See `MaxDimensions()` in //tensorflow/core/framework/tensor_shape.h )
# So we do not try to infer values for int32 tensors larger than this,
# As they cannot represent shapes.
_MAX_TENSOR_RANK = 254


class KerasTensor(object):
  # [CLASS DOC]
  # Symbolic stand-in for a Keras layer input or output while a Functional model is being built.
  # Holds a type specification (private attribute for the spec), an optional statically inferred
  # value (used mainly for small integer shape tensors) and an optional, not necessarily unique
  # name. Instances are normally created by the conversion helpers in this module or by Input, not
  # by users. Behaves tensor-like (shape, dtype, name, reference, set_shape) but refuses
  # hashing, length, numpy conversion and the op attribute, raising type errors that explain
  # dispatching to lambda layers. Tensor operators are overloaded onto the class at import time.
  """A representation of a Keras in/output during Functional API construction.

  `KerasTensor`s are tensor-like objects that represent the symbolic inputs
  and outputs of Keras layers during Functional model construction. They are
  comprised of the `tf.TypeSpec` of the (Composite)Tensor that will be
  consumed/produced in the corresponding location of the Functional model.

  KerasTensors are intended as a private API, so users should never need to
  directly instantiate `KerasTensor`s.

  **Building Functional Models with KerasTensors**
  `tf.keras.Input` produces `KerasTensor`s that represent the symbolic inputs
  to your model.

  Passing a `KerasTensor` to a `tf.keras.Layer` `__call__` lets the layer know
  that you are building a Functional model. The layer __call__ will
  infer the output signature and return `KerasTensor`s with `tf.TypeSpec`s
  corresponding to the symbolic outputs of that layer call. These output
  `KerasTensor`s will have all of the internal KerasHistory metadata attached
  to them that Keras needs to construct a Functional Model.

  Currently, layers infer the output signature by:
    * creating a scratch `FuncGraph`
    * making placeholders in the scratch graph that match the input typespecs
    * Calling `layer.call` on these placeholders
    * extracting the signatures of the outputs before clearing the scratch graph

  (Note: names assigned to KerasTensors by this process are not guaranteed to
  be unique, and are subject to implementation details).

  `tf.nest` methods are used to insure all of the inputs/output data
  structures get maintained, with elements swapped between KerasTensors and
  placeholders.

  In rare cases (such as when directly manipulating shapes using Keras layers),
  the layer may be able to partially infer the value of the output in addition
  to just inferring the signature.
  When this happens, the returned KerasTensor will also contain the inferred
  value information. Follow-on layers can use this information.
  during their own output signature inference.
  E.g. if one layer produces a symbolic `KerasTensor` that the next layer uses
  as the shape of its outputs, partially knowing the value helps infer the
  output shape.

  **Automatically converting TF APIs to layers**:
  If you passing a `KerasTensor` to a TF API that supports dispatching,
  Keras will automatically turn that API call into a lambda
  layer in the Functional model, and return KerasTensors representing the
  symbolic outputs.

  Most TF APIs that take only tensors as input and produce output tensors
  will support dispatching.

  Calling a `tf.function` does not support dispatching, so you cannot pass
  `KerasTensor`s as inputs to a `tf.function`.

  Higher-order APIs that take methods which produce tensors (e.g. `tf.while`,
  `tf.map_fn`, `tf.cond`) also do not currently support dispatching. So, you
  cannot directly pass KerasTensors as inputs to these APIs either. If you
  want to use these APIs inside of a Functional model, you must put them inside
  of a custom layer.

  Args:
    type_spec: The `tf.TypeSpec` for the symbolic input created by
      `tf.keras.Input`, or symbolically inferred for the output
      during a symbolic layer `__call__`.
    inferred_value: (Optional) a non-symbolic static value, possibly partially
      specified, that could be symbolically inferred for the outputs during
      a symbolic layer `__call__`. This will generally only happen when
      grabbing and manipulating `tf.int32` shapes directly as tensors.
      Statically inferring values in this way and storing them in the
      KerasTensor allows follow-on layers to infer output signatures
      more effectively. (e.g. when using a symbolic shape tensor to later
      construct a tensor with that shape).
    name: (optional) string name for this KerasTensor. Names automatically
      generated by symbolic layer `__call__`s are not guaranteed to be unique,
      and are subject to implementation details.
  """

  def __init__(self, type_spec, inferred_value=None, name=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: type_spec, a type specification object (required); inferred_value,
    # optional statically known value, defaults to nothing; name, optional string, defaults to
    # nothing. Stores all three as private state. Raises a value error if type_spec is not a type
    # specification instance. Returns nothing.
    """Constructs a KerasTensor."""
    if not isinstance(type_spec, type_spec_module.TypeSpec):
      raise ValueError('KerasTensors must be constructed with a `tf.TypeSpec`.')

    self._type_spec = type_spec
    self._inferred_value = inferred_value
    self._name = name

  @property
  def type_spec(self):
    # [FUNCTION DOC]
    # Read-only property returning the stored type specification. No inputs, no side effects.
    """Returns the `tf.TypeSpec` symbolically inferred for this Keras output."""
    return self._type_spec

  @property
  def shape(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: reads the private shape attribute of the type specification directly, so it is
    # only valid for dense, sparse and ragged specs; other spec kinds may lack it (see the TODO).
    # Read-only property returning the shape recorded in the type specification. No side effects.
    """Returns the `TensorShape` symbolically inferred for this Keras output."""
    # TODO(kaftan): This is only valid for normal/sparse/ragged tensors.
    # may need to raise an error when it's not valid for a type_spec,
    # but some keras code (e.g. build-related stuff) will likely fail when
    # it can't access shape or dtype
    return self._type_spec._shape  # pylint: disable=protected-access

  @classmethod
  def from_tensor(cls, tensor):
    # [FUNCTION DOC]
    # Class method building a KerasTensor from a traced tensor or composite tensor. Input: tensor,
    # the traced value. For a plain dense tensor it records the name and type spec; if the dtype is
    # 32-bit integer with rank known and below two (it may be a shape), it asks the shape
    # machinery of the current scratch graph for static value information, keeping it as a list
    # unless it is unknown or longer than the maximum tensor rank (then nothing is stored). This
    # branch always builds the base KerasTensor class, not the calling subclass. For any other
    # input it builds an instance of the calling class from the name and inferred type spec.
    # Returns the new KerasTensor. Side effect: creates a throwaway ones op in the current graph.
    """Convert a traced (composite)tensor to a representative KerasTensor."""
    if isinstance(tensor, tensor_lib.Tensor):
      name = getattr(tensor, 'name', None)
      type_spec = type_spec_module.type_spec_from_value(tensor)
      inferred_value = None
      if (type_spec.dtype == dtypes.int32 and type_spec.shape.rank is not None
          and type_spec.shape.rank < 2):
        # If this tensor might be representing shape information,
        # (dtype=int32, rank of 0 or 1, not too large to represent a shape)
        # we attempt to capture any value information tensorflow's
        # shape handling can extract from the current scratch graph.
        #
        # Even though keras layers each trace in their own scratch
        # graph, this shape value info extraction allows us to capture
        # a sizable and useful subset of the C++ shape value inference TF can do
        # if all tf ops appear in the same graph when using shape ops.
        #
        # Examples of things this cannot infer concrete dimensions for
        # that the full single-graph C++ shape inference sometimes can are:
        # * cases where the shape tensor is cast out of int32 before being
        #   manipulated w/ floating point numbers then converted back
        # * cases where int32 tensors w/ rank >= 2 are manipulated before being
        #   used as a shape tensor
        # * cases where int32 tensors too large to represent shapes are
        #   manipulated to a smaller size before being used as a shape tensor
        inferred_value = array_ops.ones(shape=tensor).shape
        if inferred_value.dims:
          inferred_value = inferred_value.as_list()
          if len(inferred_value) > _MAX_TENSOR_RANK:
            inferred_value = None
        else:
          inferred_value = None

      return KerasTensor(type_spec, inferred_value=inferred_value, name=name)
    else:
      # Fallback to the generic arbitrary-typespec KerasTensor
      name = getattr(tensor, 'name', None)
      type_spec = type_spec_module.type_spec_from_value(tensor)
      return cls(type_spec, name=name)

  @classmethod
  def from_type_spec(cls, type_spec, name=None):
    # [FUNCTION DOC]
    # Class method creating an instance of the calling class from a type specification (required)
    # and an optional name. No inferred value is recorded. Returns the new instance. It has no
    # docstring of its own.
    return cls(type_spec=type_spec, name=name)

  def _to_placeholder(self):
    # [FUNCTION DOC]
    # HELPER: used by layer tracing to build scratch-graph inputs from symbolic Keras values.
    # Creates graph placeholder(s) for this symbolic value in the current graph. If an inferred
    # value exists, it is re-injected by taking the shape of an int32 placeholder whose shape is
    # that value (so partially unknown entries survive), converted back to a scalar when the spec
    # has rank zero. Otherwise it maps over the type spec (including composite components) and
    # makes one placeholder per component with matching dtype and shape. Returns a tensor or a
    # nested structure of tensors. Side effect: adds ops to the current default graph.
    """Convert this KerasTensor to a placeholder in a graph."""
    # If there is an inferred value for this tensor, inject the inferred value
    if self._inferred_value is not None:
      # If we suspect this KerasTensor might be representing a shape tensor,
      # and we were able to extract value information with TensorFlow's shape
      # handling when making the KerasTensor, we construct the placeholder by
      # re-injecting the inferred value information into the graph. We
      # do this injection through the shape of a placeholder, because that
      # allows us to specify partially-unspecified shape values.
      #
      # See the comment on value extraction inside `from_tensor` for more info.
      inferred_value = array_ops.shape(
          array_ops.placeholder(
              shape=self._inferred_value, dtype=dtypes.int32))
      if self.type_spec.shape.rank == 0:
        # `tf.shape` always returns a rank-1, we may need to turn it back to a
        # scalar.
        inferred_value = inferred_value[0]
      return inferred_value

    # Use the generic conversion from typespec to a placeholder.
    def component_to_placeholder(component):
      # [FUNCTION DOC]
      # HELPER: per-component mapper for the generic placeholder conversion above.
      # Input: component, a leaf spec with dtype and shape. Returns a new placeholder with that
      # dtype and shape in the current graph.
      return array_ops.placeholder(component.dtype, component.shape)

    return nest.map_structure(
        component_to_placeholder, self.type_spec, expand_composites=True)

  def get_shape(self) -> tensor_shape.TensorShape:
    # [FUNCTION DOC]
    # Tensor-compatible accessor returning the same symbolic shape as the shape property. No
    # side effects.
    return self.shape

  def __len__(self):
    # [FUNCTION DOC]
    # Length is deliberately unsupported: always raises a type error explaining that the value
    # is a Keras symbolic input/output, that the API called may lack dispatching, and that asserting
    # on a symbolic value also triggers this. Takes no inputs and never returns.
    raise TypeError('Keras symbolic inputs/outputs do not '
                    'implement `__len__`. You may be '
                    'trying to pass Keras symbolic inputs/outputs '
                    'to a TF API that does not register dispatching, '
                    'preventing Keras from automatically '
                    'converting the API call to a lambda layer '
                    'in the Functional Model. This error will also get raised '
                    'if you try asserting a symbolic input/output directly.')

  @property
  def op(self):
    # [FUNCTION DOC]
    # Property that is deliberately unsupported: reading it always raises a type error explaining
    # that symbolic Keras values have no op and that the TF API used may not dispatch to a lambda
    # layer. Never returns.
    raise TypeError('Keras symbolic inputs/outputs do not '
                    'implement `op`. You may be '
                    'trying to pass Keras symbolic inputs/outputs '
                    'to a TF API that does not register dispatching, '
                    'preventing Keras from automatically '
                    'converting the API call to a lambda layer '
                    'in the Functional Model.')

  def __hash__(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the message is built by formatting the description of the object directly after
    # the opening text with no separating space, so the error text runs together.
    # Hashing is deliberately unsupported: always raises a type error telling the caller to use
    # the reference method to get a hashable key. Never returns.
    raise TypeError('Tensors are unhashable. (%s)'
                    'Instead, use tensor.ref() as the key.' % self)

  # Note: This enables the KerasTensor's overloaded "right" binary
  # operators to run when the left operand is an ndarray, because it
  # accords the Tensor class higher priority than an ndarray, or a
  # numpy matrix.
  # In the future explore changing this to using numpy's __numpy_ufunc__
  # mechanism, which allows more control over how Tensors interact
  # with ndarrays.
  __array_priority__ = 100

  def __array__(self):
    # [FUNCTION DOC]
    # Numpy conversion hook that is deliberately unsupported: always raises a type error saying a
    # symbolic Keras value cannot become a numpy array and hinting at the non-dispatching API case.
    # Never returns.
    raise TypeError(
        'Cannot convert a symbolic Keras input/output to a numpy array. '
        'This error may indicate that you\'re trying to pass a symbolic value '
        'to a NumPy call, which is not supported. Or, '
        'you may be trying to pass Keras symbolic inputs/outputs '
        'to a TF API that does not register dispatching, '
        'preventing Keras from automatically '
        'converting the API call to a lambda layer '
        'in the Functional Model.')

  @property
  def is_tensor_like(self):
    # [FUNCTION DOC]
    # Property marking this object as tensor-like for TensorFlow utilities; always true.
    return True

  def set_shape(self, shape):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the incompatibility error message is built from two adjacent string pieces with
    # no space between "not" and "compatible".
    # Narrows the symbolic shape. Input: shape, a shape object or anything convertible to one.
    # Unknown dimensions in the new shape are filled from the current shape when ranks line up
    # (a mismatched rank would index out of range if the current shape is shorter). Raises a value
    # error if the merged shape is incompatible with the current one. Otherwise overwrites the
    # shape stored inside the type specification (mutating the spec in place). Returns nothing.
    """Updates the shape of this KerasTensor. Mimics `tf.Tensor.set_shape()`."""
    if not isinstance(shape, tensor_shape.TensorShape):
      shape = tensor_shape.TensorShape(shape)
    if shape.dims is not None:
      dim_list = [dim.value for dim in shape.dims]
      for dim in range(len(dim_list)):
        if dim_list[dim] is None and self.shape.dims is not None:
          dim_list[dim] = self.shape.dims[dim]
      shape = tensor_shape.TensorShape(dim_list)
    if not self.shape.is_compatible_with(shape):
      raise ValueError(
          "Keras symbolic input/output's shape %s is not"
          "compatible with supplied shape %s" %
          (self.shape, shape))
    else:
      self._type_spec._shape = shape  # pylint: disable=protected-access

  def __str__(self):
    # [FUNCTION DOC]
    # Builds a readable description string listing the type spec, the inferred value if any, the
    # name if any, and, when the object carries layer history, which layer created it. Returns
    # the string. No side effects.
    symbolic_description = ''
    inferred_value_string = ''
    name_string = ''

    if hasattr(self, '_keras_history'):
      layer = self._keras_history.layer
      symbolic_description = (
          ', description="created by layer \'%s\'"' % (layer.name,))
    if self._inferred_value is not None:
      inferred_value_string = (
          ', inferred_value=%s' % self._inferred_value)
    if self.name is not None:
      name_string = ', name=\'%s\'' % self._name
    return 'KerasTensor(type_spec=%s%s%s%s)' % (
        self.type_spec, inferred_value_string,
        name_string, symbolic_description)

  def __repr__(self):
    # [FUNCTION DOC]
    # Builds a compact representation: for dense tensor specs it shows shape and dtype name,
    # otherwise the type spec; appends the inferred value if present and the creating layer name
    # if layer history is attached. Returns the string. No side effects.
    symbolic_description = ''
    inferred_value_string = ''
    if isinstance(self.type_spec, tensor_lib.TensorSpec):
      type_spec_string = 'shape=%s dtype=%s' % (self.shape, self.dtype.name)
    else:
      type_spec_string = 'type_spec=%s' % self.type_spec

    if hasattr(self, '_keras_history'):
      layer = self._keras_history.layer
      symbolic_description = ' (created by layer \'%s\')' % (layer.name,)
    if self._inferred_value is not None:
      inferred_value_string = (
          ' inferred_value=%s' % self._inferred_value)
    return '<KerasTensor: %s%s%s>' % (
        type_spec_string, inferred_value_string, symbolic_description)

  @property
  def dtype(self):
    # [FUNCTION DOC]
    # Read-only property returning the data type from the private dtype attribute of the type
    # specification (valid only for specs that define one, per the TODO). No side effects.
    """Returns the `dtype` symbolically inferred for this Keras output."""
    # TODO(kaftan): This is only valid for normal/sparse/ragged tensors.
    # may need to raise an error when it's not valid for a type_spec,
    # but some keras code (e.g. build-related stuff) will likely fail when
    # it can't access shape or dtype
    return self._type_spec._dtype  # pylint: disable=protected-access

  def ref(self):
    # [FUNCTION DOC]
    # Returns an identity-based hashable wrapper around this object, suitable as a set member or
    # mapping key. No inputs, no side effects.
    """Returns a hashable reference object to this KerasTensor.

    The primary use case for this API is to put KerasTensors in a
    set/dictionary. We can't put tensors in a set/dictionary as
    `tensor.__hash__()` is not available and tensor equality (`==`) is supposed
    to produce a tensor representing if the two inputs are equal.

    See the documentation of `tf.Tensor.ref()` for more info.
    """
    return object_identity.Reference(self)

  def __iter__(self):
    # [FUNCTION DOC]
    # Iterates over the leading dimension. Raises a type error if the rank is unknown, the value
    # is a scalar, or the first dimension is unknown. Otherwise returns an iterator object that
    # yields successive slices by indexing the symbolic value, for the known first-dimension count.
    shape = None
    if self.shape.ndims is not None:
      shape = [dim.value for dim in self.shape.dims]

    if shape is None:
      raise TypeError('Cannot iterate over a Tensor with unknown shape.')
    if not shape:
      raise TypeError('Cannot iterate over a scalar.')
    if shape[0] is None:
      raise TypeError(
          'Cannot iterate over a Tensor with unknown first dimension.')
    return _KerasTensorIterator(self, shape[0])

  @property
  def name(self):
    # [FUNCTION DOC]
    # Read-only property returning the optional name given at construction (may be nothing).
    """Returns the (non-unique, optional) name of this symbolic Keras value."""
    return self._name

  @classmethod
  def _overload_all_operators(cls, tensor_class):  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # HELPER: import-time setup that copies tensor behaviour onto this class.
    # Class method. Input: tensor_class, the tensor class to copy from. For each operator name in
    # the base tensor's overloadable operator list, installs the same implementation on this class;
    # also installs the legacy experimental reference method if the tensor class has it. Mutates
    # the class. Returns nothing.
    """Register overloads for all operators."""
    for operator in tensor_lib.Tensor.OVERLOADABLE_OPERATORS:
      cls._overload_operator(tensor_class, operator)

    # We include `experimental_ref` for versions of TensorFlow that
    # still include the deprecated method in Tensors.
    if hasattr(tensor_class, 'experimental_ref'):
      cls._overload_operator(tensor_class, 'experimental_ref')

  @classmethod
  def _overload_operator(cls, tensor_class, operator):  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # HELPER: installs one operator on this class; used by the operator-overloading setup.
    # Class method. Inputs: tensor_class, the class to fetch from; operator, the attribute name.
    # Looks the attribute up on tensor_class (raises an attribute error if missing), unwraps the
    # underlying function if it is an unbound method wrapper (Python 2 compatibility, now mostly
    # dead), and sets it on this class. Mutates the class. Returns nothing.
    """Overload an operator with the same implementation as a base Tensor class.

    We pull the operator out of the class dynamically to avoid ordering issues.

    Args:
      tensor_class: The (Composite)Tensor to get the method from.
      operator: string. The operator name.
    """
    tensor_oper = getattr(tensor_class, operator)

    # Compatibility with Python 2:
    # Python 2 unbound methods have type checks for the first arg,
    # so we need to extract the underlying function
    tensor_oper = getattr(tensor_oper, '__func__', tensor_oper)

    setattr(cls, operator, tensor_oper)


KerasTensor._overload_all_operators(tensor_lib.Tensor)  # pylint: disable=protected-access


class SparseKerasTensor(KerasTensor):
  # [CLASS DOC]
  # KerasTensor variant for sparse tensors. Only overrides placeholder creation so the dense
  # shape of top-level sparse tensors is preserved. Selected through the module registry.
  """A specialized KerasTensor representation for `tf.sparse.SparseTensor`s.

  Specifically, it specializes the conversion to a placeholder in order
  to maintain dense shape information.
  """

  def _to_placeholder(self):
    # [FUNCTION DOC]
    # HELPER: placeholder creation used during layer tracing.
    # Creates a sparse placeholder with the spec's dtype and shape in the current graph, avoiding
    # the generic mapping which would lose dense shape information. Only top-level sparse tensors
    # keep shape info; nested ones do not. Returns the sparse placeholder; adds ops to the graph.
    spec = self.type_spec

    # nest.map_structure loses dense shape information for sparse tensors.
    # So, we special-case sparse placeholder creation.
    # This only preserves shape information for top-level sparse tensors;
    # not for sparse tensors that are nested inside another composite
    # tensor.
    return array_ops.sparse_placeholder(dtype=spec.dtype, shape=spec.shape)


class RaggedKerasTensor(KerasTensor):
  # [CLASS DOC]
  # KerasTensor variant for ragged tensors. Builds ragged placeholders that retain shape
  # information for uniform dimensions, exposes the ragged rank, and (after the class body, at
  # module level) borrows ragged versions of slicing, addition and multiplication operators.
  """A specialized KerasTensor representation for `tf.RaggedTensor`s.

  Specifically, it:

  1. Specializes the conversion to a placeholder in order
  to maintain shape information for non-ragged dimensions.
  2. Overloads the KerasTensor's operators with the RaggedTensor versions
  when they don't match the `tf.Tensor` versions
  3. Exposes some of the instance method/attribute that are unique to
  the RaggedTensor API (such as ragged_rank).
  """

  def _to_placeholder(self):
    # [FUNCTION DOC]
    # HELPER: placeholder creation used during layer tracing.
    # Falls back to the base implementation if the ragged rank is zero or the rank is unknown.
    # Otherwise creates a placeholder for the flat values (shape after the ragged dimensions), then
    # wraps it from the innermost ragged axis outwards: for a variable-size axis it adds a row
    # splits placeholder (length one more than the known count of rows, or unknown) and builds a
    # ragged tensor from row splits; for a fixed-size axis it builds one from a constant uniform
    # row length. Validation is disabled. Returns a ragged tensor; adds ops to the current graph.
    ragged_spec = self.type_spec
    if ragged_spec.ragged_rank == 0 or ragged_spec.shape.rank is None:
      return super(RaggedKerasTensor, self)._to_placeholder()

    flat_shape = ragged_spec.shape[ragged_spec.ragged_rank:]
    result = array_ops.placeholder(ragged_spec.dtype, flat_shape)

    known_num_splits = []
    prod = 1
    for axis_size in ragged_spec.shape:
      if prod is not None:
        if axis_size is None or (
            getattr(axis_size, 'value', True) is None):
          prod = None
        else:
          prod = prod * axis_size
      known_num_splits.append(prod)

    for axis in range(ragged_spec.ragged_rank, 0, -1):
      axis_size = ragged_spec.shape[axis]
      if axis_size is None or (getattr(axis_size, 'value', True) is None):
        num_splits = known_num_splits[axis-1]
        if num_splits is not None:
          num_splits = num_splits + 1
        splits = array_ops.placeholder(
            ragged_spec.row_splits_dtype, [num_splits])
        result = ragged_tensor.RaggedTensor.from_row_splits(
            result, splits, validate=False)
      else:
        rowlen = constant_op.constant(axis_size, ragged_spec.row_splits_dtype)
        result = ragged_tensor.RaggedTensor.from_uniform_row_length(
            result, rowlen, validate=False)
    return result

  @property
  def ragged_rank(self):
    # [FUNCTION DOC]
    # Property returning the number of ragged dimensions from the type specification.
    return self.type_spec.ragged_rank

# Overload slicing
RaggedKerasTensor._overload_operator(ragged_tensor.RaggedTensor, '__getitem__')  # pylint: disable=protected-access

# Overload math ops
RaggedKerasTensor._overload_operator(ragged_tensor.RaggedTensor, '__add__')  # pylint: disable=protected-access
RaggedKerasTensor._overload_operator(ragged_tensor.RaggedTensor, '__radd__')  # pylint: disable=protected-access
RaggedKerasTensor._overload_operator(ragged_tensor.RaggedTensor, '__mul__')  # pylint: disable=protected-access
RaggedKerasTensor._overload_operator(ragged_tensor.RaggedTensor, '__rmul__')  # pylint: disable=protected-access


# TODO(b/161487382):
# Special-case user-registered symbolic objects (registered by the
# private `register_symbolic_tensor_type` method) by passing them between
# scratch graphs directly.
# This is needed to not break Tensorflow probability
# while they finish migrating to composite tensors.
class UserRegisteredSpec(type_spec_module.TypeSpec):
  # [CLASS DOC]
  # HELPER: minimal type specification wrapping the shape and dtype of a legacy user-registered
  # symbolic object, for UserRegisteredTypeKerasTensor. All component and serialization hooks
  # raise a not-implemented error, so it cannot be used for real specs operations.
  """TypeSpec to represent user-registered symbolic objects."""

  def __init__(self, shape, dtype):
    # [FUNCTION DOC]
    # KNOWN ISSUE: does not call the parent constructor, and stores the dtype both as a private and
    # a public attribute.
    # Inputs: shape and dtype of the wrapped object. Stores them as attributes. Returns nothing.
    self.shape = shape
    self._dtype = dtype
    self.dtype = dtype

  def _component_specs(self):
    # [FUNCTION DOC]
    # HELPER: required type-spec hook; unsupported here, always raises a not-implemented error.
    raise NotImplementedError

  def _from_components(self, components):
    # [FUNCTION DOC]
    # HELPER: required type-spec hook; input components is ignored. Always raises a
    # not-implemented error.
    raise NotImplementedError

  def _serialize(self):
    # [FUNCTION DOC]
    # HELPER: required type-spec hook; unsupported, always raises a not-implemented error.
    raise NotImplementedError

  def _to_components(self, value):
    # [FUNCTION DOC]
    # HELPER: required type-spec hook; input value is ignored. Always raises a not-implemented
    # error.
    raise NotImplementedError

  def value_type(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the base class defines this as a property, but here it is a plain method, so
    # code reading it as an attribute (such as converting a type spec to a KerasTensor) gets a
    # bound method rather than a class.
    # Always raises a not-implemented error.
    raise NotImplementedError


# TODO(b/161487382):
# Special-case user-registered symbolic objects (registered by the
# private `register_symbolic_tensor_type` method) by passing them between
# scratch graphs directly.
# This is needed to not break Tensorflow probability
# while they finish migrating to composite tensors.
class UserRegisteredTypeKerasTensor(KerasTensor):
  # [CLASS DOC]
  # KerasTensor that wraps a legacy user-registered symbolic object (such as ones registered
  # through the private symbolic-type registration API) and passes that object itself between
  # scratch graphs as the placeholder. Cannot be created from a type spec.
  """KerasTensor that represents legacy register_symbolic_tensor_type."""

  def __init__(self, user_registered_symbolic_object):
    # [FUNCTION DOC]
    # KNOWN ISSUE: passes the name as the second positional argument to the parent constructor,
    # whose second parameter is the inferred value, so the name is stored as the inferred value
    # and the actual name is left unset.
    # Input: user_registered_symbolic_object, any object with shape and dtype (and optionally a
    # name). Keeps a reference to it, builds a spec from its shape and dtype, then initialises the
    # parent. Returns nothing.
    x = user_registered_symbolic_object
    self._user_registered_symbolic_object = x
    type_spec = UserRegisteredSpec(x.shape, x.dtype)
    name = getattr(x, 'name', None)

    super(UserRegisteredTypeKerasTensor, self).__init__(type_spec, name)

  @classmethod
  def from_tensor(cls, tensor):
    # [FUNCTION DOC]
    # Class method wrapping the given user-registered symbolic object in a new instance and
    # returning it.
    return cls(tensor)

  @classmethod
  def from_type_spec(cls, type_spec, name=None):
    # [FUNCTION DOC]
    # Unsupported: always raises a not-implemented error saying a KerasTensor cannot be built
    # directly from a type spec. Inputs type_spec and name are only used for the message/ignored.
    raise NotImplementedError('You cannot instantiate a KerasTensor '
                              'directly from TypeSpec: %s' % type_spec)

  def _to_placeholder(self):
    # [FUNCTION DOC]
    # HELPER: placeholder creation used during layer tracing.
    # Returns the original wrapped symbolic object unchanged (no new graph ops).
    return self._user_registered_symbolic_object


class _KerasTensorIterator(object):
  # [CLASS DOC]
  # HELPER: iterator returned by KerasTensor iteration. Walks indices from zero up to a known
  # limit and yields the result of indexing the tensor at each position. Does no validation.
  """Iterates over the leading dim of a KerasTensor. Performs 0 error checks."""

  def __init__(self, tensor, dim0):
    # [FUNCTION DOC]
    # HELPER: constructor. Inputs: tensor, the symbolic value to iterate; dim0, the number of
    # items (first dimension size). Starts the position at zero. Returns nothing.
    self._tensor = tensor
    self._index = 0
    self._limit = dim0

  def __iter__(self):
    # [FUNCTION DOC]
    # HELPER: iterator protocol; returns the iterator itself.
    return self

  def __next__(self):
    # [FUNCTION DOC]
    # HELPER: iterator protocol. Raises the stop-iteration signal once the limit is reached;
    # otherwise returns the tensor indexed at the current position and advances the position.
    if self._index == self._limit:
      raise StopIteration
    result = self._tensor[self._index]
    self._index += 1
    return result


# Specify the mappings of tensor class to KerasTensor class.
# This is specifically a list instead of a dict for now because
# 1. we do a check w/ isinstance because a key lookup based on class
#    would miss subclasses
# 2. a list allows us to control lookup ordering
# We include tensor.Tensor -> KerasTensor in the first position as a fastpath,
# *and* include object -> KerasTensor at the end as a catch-all.
# We can re-visit these choices in the future as needed.
keras_tensor_classes = [
    (tensor_lib.Tensor, KerasTensor),
    (sparse_tensor.SparseTensor, SparseKerasTensor),
    (ragged_tensor.RaggedTensor, RaggedKerasTensor),
    (object, KerasTensor)
]


def register_keras_tensor_specialization(cls, keras_tensor_subclass):
  # [FUNCTION DOC]
  # Registers a specialised KerasTensor subclass. Inputs: cls, a tensor class to match; and
  # keras_tensor_subclass, the KerasTensor class to use for it. Inserts the pair just before the
  # final catch-all entry, so later registrations come after earlier ones but before the fallback.
  # Mutates the module-level registry list. Returns nothing.
  """Register a specialized KerasTensor subclass for a Tensor type."""
  # We always leave (object, KerasTensor) at the end as a generic fallback
  keras_tensor_classes.insert(-1, (cls, keras_tensor_subclass))


def keras_tensor_to_placeholder(x):
  # [FUNCTION DOC]
  # HELPER: used when tracing layers in scratch graphs, typically via structure mapping.
  # Input: x, any value. If it is a KerasTensor, returns the placeholder built from it (adding
  # graph ops); otherwise returns x unchanged.
  """Construct a graph placeholder to represent a KerasTensor when tracing."""
  if isinstance(x, KerasTensor):
    return x._to_placeholder()  # pylint: disable=protected-access
  else:
    return x


def keras_tensor_from_tensor(tensor):
  # [FUNCTION DOC]
  # Converts a traced tensor or composite tensor to the best-matching KerasTensor subclass: scans
  # the registry in order and uses the first entry whose tensor class matches the instance, then
  # calls that class's conversion. If the input carries a mask attribute, the mask is converted
  # recursively and attached to the result. Input: tensor. Returns the new KerasTensor. Mutates
  # only the new object.
  """Convert a traced (composite)tensor to a representative KerasTensor."""
  # Create a specialized KerasTensor that supports instance methods,
  # operators, and additional value inference if possible
  keras_tensor_cls = None
  for tensor_type, cls in keras_tensor_classes:
    if isinstance(tensor, tensor_type):
      keras_tensor_cls = cls
      break

  out = keras_tensor_cls.from_tensor(tensor)

  if hasattr(tensor, '_keras_mask'):
    out._keras_mask = keras_tensor_from_tensor(tensor._keras_mask)  # pylint: disable=protected-access
  return out


def keras_tensor_from_type_spec(type_spec, name=None):
  # [FUNCTION DOC]
  # Converts a type specification to the best-matching KerasTensor subclass: reads the spec's
  # value type and uses the first registry entry for which it is a subclass of the registered
  # tensor class, then builds an instance from the spec and optional name. Inputs: type_spec
  # (required), name (optional, defaults to nothing). Returns the KerasTensor. Raises a type error
  # if the value type is not a class (for example the legacy user-registered spec).
  """Convert a TypeSpec to a representative KerasTensor."""
  # Create a specialized KerasTensor that supports instance methods,
  # operators, and additional value inference if possible
  keras_tensor_cls = None
  value_type = type_spec.value_type
  for tensor_type, cls in keras_tensor_classes:
    if issubclass(value_type, tensor_type):
      keras_tensor_cls = cls
      break

  return keras_tensor_cls.from_type_spec(type_spec, name=name)
