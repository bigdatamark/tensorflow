# [MODULE DOC]
# Operations converting between extension-type (composite tensor) values and scalar variant
# tensors. Exposes functions that encode a composite value, including a serialized description
# of its type, into a variant, and decode a variant back into a value of a given type spec.
# Import-time side effect: registers gradient functions for the two underlying variant ops.
# Gradients convert in the opposite direction of the forward op.
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
# =============================================================================
"""Operations for ExtensionTypes (aka Composite Tensors)."""

from tensorflow.core.protobuf import composite_tensor_variant_pb2
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import gen_composite_tensor_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.util import nest


def composite_tensor_to_variants(value, type_spec=None, name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring lists only a value error for an incompatible type spec, but the
  # function also raises a type error when value is not a composite tensor.
  # Inputs: value (composite tensor to encode), type_spec (optional description of the value's
  # type; defaults to the value's own type spec), name (optional op name). Checks the type spec
  # is compatible with the value, serializes it into metadata, flattens the value into its
  # component tensors and builds the op that packs components and metadata into one scalar
  # variant tensor, which is returned. Raises type error or value error as above. Adds an op to
  # the current graph (or executes eagerly).
  """Encodes `value` as a scalar variant tensor.

  Args:
    value: The `ExtensionType` value to encode.
    type_spec: Information about the value's type that should be included in the
      encoding.
    name: Optional name for the operation.

  Returns:
    A Tensor with shape=`()` and dtype=`tf.variant`.

  Raises:
    ValueError: If `type_spec` is not compatible with `value`.
  """
  if not isinstance(value, composite_tensor.CompositeTensor):
    raise TypeError("Expected `value` to be a CompositeTensor. "
                    f"Received {type(value)}.")

  if type_spec is None:
    type_spec = value._type_spec  # pylint: disable=protected-access
  if not type_spec.is_compatible_with(value):
    raise ValueError(f"`type_spec` {type_spec} is not compatible with `value` "
                     f"{value!r}.")
  metadata = composite_tensor_variant_pb2.CompositeTensorVariantMetadata()
  metadata.type_spec_proto.CopyFrom(
      nested_structure_coder.encode_structure(type_spec).type_spec_value)

  return gen_composite_tensor_ops.CompositeTensorVariantFromComponents(
      components=nest.flatten(value, expand_composites=True),
      metadata=metadata.SerializeToString(),
      name=name)


def composite_tensor_from_variant(encoded, type_spec, name=None):
  # [FUNCTION DOC]
  # Inputs: encoded (scalar variant tensor produced by the encoding function), type_spec (type
  # description of the original value, used for the serialized metadata and for the number and
  # dtypes of components), name (optional op name). Validates that encoded is a tensor with
  # variant dtype and compatible with a scalar shape (type error if not, shape errors for other
  # shapes), then builds the op that unpacks the variant into component tensors and packs them
  # into a value of the given type. Returns that composite value. The op raises an
  # invalid-argument error at run time if the encoding does not match the type spec.
  """Returns the `ExtensionType` value encoded by a variant scalar tensor.

  Args:
    encoded: A Tensor returned by `composite_tensor_to_variants`.
    type_spec: The `TypeSpec` of the original value.  This is used to determine
      the number and types of the component tensors that comprise the decoded
      value.  Must be compatible with the `TypeSpec` serilized in `encoded`.
    name: Optional name for the operation.

  Returns:
    An `ExtensionType` value that is compatible with `TypeSpec`.

  Raises:
    TypeError: If `encoded` is not a Tensor with dtype=variant.
    InvalidArgumentError: If `encoded` is not compatible with `type_spec`.
  """
  if not isinstance(encoded, tensor.Tensor):
    raise TypeError(f"Expected `encoded` to be a Tensor, got {encoded!r}.")
  if encoded.dtype != dtypes.variant:
    raise TypeError("Expected `encoded` to have dtype=variant, got "
                    f"{encoded!r}.")
  encoded.shape.assert_is_compatible_with(())

  metadata = composite_tensor_variant_pb2.CompositeTensorVariantMetadata()
  metadata.type_spec_proto.CopyFrom(
      nested_structure_coder.encode_structure(type_spec).type_spec_value)

  component_dtypes = [
      t.dtype for t in nest.flatten(type_spec, expand_composites=True)
  ]

  components = gen_composite_tensor_ops.CompositeTensorVariantToComponents(
      encoded=encoded,
      metadata=metadata.SerializeToString(),
      Tcomponents=component_dtypes,
      name=name)
  return nest.pack_sequence_as(type_spec, components, expand_composites=True)


@ops.RegisterGradient("CompositeTensorVariantFromComponents")
def _composite_tensor_to_variants_grad(op, grad):
  # [FUNCTION DOC]
  # HELPER: gradient function registered for the op that builds a variant from components.
  # Inputs: op (the forward operation) and grad (gradient w.r.t. its variant output). Decodes
  # the incoming gradient variant into component tensors using the forward op's stored metadata
  # and component dtypes. Returns those components as the gradients of the inputs.
  return gen_composite_tensor_ops.CompositeTensorVariantToComponents(
      encoded=grad,
      metadata=op.get_attr("metadata"),
      Tcomponents=op.get_attr("Tcomponents"))


@ops.RegisterGradient("CompositeTensorVariantToComponents")
def _composite_tensor_from_variant_grad(op, *grad):
  # [FUNCTION DOC]
  # HELPER: gradient function registered for the op that splits a variant into components.
  # Inputs: op (the forward operation) and one gradient per component output (entries may be
  # nothing/null). Asserts the number of gradients equals the number of outputs. Replaces any
  # missing gradient by the forward output itself, then repacks the components into a variant
  # with the forward op's metadata. Returns that variant as the gradient of the encoded input.
  assert len(grad) == len(op.outputs)
  # `components` is `op.outputs`, but with any tensors for which we're
  # taking the gradient replaced by the corresponding value from `grad`.
  components = [
      op.outputs[i] if grad[i] is None else grad[i] for i in range(len(grad))
  ]
  return gen_composite_tensor_ops.CompositeTensorVariantFromComponents(
      components=components, metadata=op.get_attr("metadata"))
