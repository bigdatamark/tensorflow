# [MODULE DOC]
# Test module for the composite tensor ops that encode composite tensors (ragged and sparse
# tensors) into scalar variant tensors and decode them back. Covers round trips, decoding and
# encoding error messages, empty and invalid variant inputs, serialization through tensor protos
# and gradient flow through encode and decode. Tests run in both graph and eager modes.
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
"""Tests for composite_tensor_ops."""

from absl.testing import parameterized

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import composite_tensor_ops
from tensorflow.python.ops import gen_composite_tensor_ops
from tensorflow.python.ops import gen_list_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest
from tensorflow.python.util import nest


@test_util.run_all_in_graph_and_eager_modes
class ExtensionTypeTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for converting composite tensors to and from variant tensors, run in both graph and
  # eager modes. Several tests are parameterized by named value factories (ragged and sparse
  # tensors) and expected error messages. No shared fixtures.

  @parameterized.named_parameters([
      ('Ragged', lambda: ragged_factory_ops.constant([[1, 2], [3], [4, 5, 6]])),
      ('Sparse', lambda: sparse_ops.from_dense([[0, 0, 3, 0], [1, 2, 0, 0]])),
  ])
  def testEncodeAndDecode(self, value_factory):
    # [FUNCTION DOC]
    # Scenario: a ragged or sparse tensor (made by the factory argument) is encoded to a variant.
    # Expected: the encoding has variant dtype and rank zero; decoding it with the original type
    # spec gives a value whose spec is compatible with the original and whose flattened
    # components are equal to the original components, in equal number.
    value = value_factory()

    encoded = composite_tensor_ops.composite_tensor_to_variants(value)
    self.assertEqual(encoded.dtype, dtypes.variant)
    self.assertEqual(encoded.shape.rank, 0)

    decoded = composite_tensor_ops.composite_tensor_from_variant(
        encoded, value._type_spec)
    self.assertTrue(value._type_spec.is_compatible_with(decoded._type_spec))
    value_components = nest.flatten(value, expand_composites=True)
    decoded_components = nest.flatten(decoded, expand_composites=True)
    self.assertLen(value_components, len(decoded_components))
    for v, d in zip(value_components, decoded_components):
      self.assertAllEqual(v, d)

  @parameterized.named_parameters([
      ('WrongType', lambda: ragged_factory_ops.constant([[1]]),
       sparse_tensor.SparseTensorSpec([None, None], dtypes.int32),
       r'Expected a SPARSE_TENSOR_SPEC \(based on `type_spec`\), but `encoded` '
       'contains a RAGGED_TENSOR_SPEC'),
      ('WrongNumComponents', lambda: ragged_factory_ops.constant([[1]]),
       ragged_tensor.RaggedTensorSpec([None, None, None], dtypes.int32),
       'Encoded value has 2 tensor components; expected 3 components'),
      ('WrongDType', lambda: ragged_factory_ops.constant([[1]]),
       ragged_tensor.RaggedTensorSpec([None, None], dtypes.float32),
       'Tensor component 0 had dtype DT_INT32; expected dtype DT_FLOAT'),
  ])
  def testDecodingErrors(self, value, spec, message):
    # [FUNCTION DOC]
    # Scenario: a ragged tensor (from the factory argument) is encoded, then decoded with a
    # mismatched type spec (wrong tensor kind, wrong number of components or wrong dtype).
    # Expected: evaluating the decode raises an invalid-argument error whose text matches the
    # given message pattern.
    encoded = composite_tensor_ops.composite_tensor_to_variants(value())
    with self.assertRaisesRegex(errors.InvalidArgumentError, message):
      self.evaluate(
          composite_tensor_ops.composite_tensor_from_variant(encoded, spec))

  @parameterized.named_parameters([
      ('IncompatibleSpec', lambda: ragged_factory_ops.constant([[1]]),
       ragged_tensor.RaggedTensorSpec([None, None, None], dtypes.int32),
       r'`type_spec` .* is not compatible with `value` .*'),
  ])
  def testEncodingErrors(self, value, spec, message):
    # [FUNCTION DOC]
    # Scenario: a ragged tensor is encoded while passing an explicit type spec that is
    # incompatible with it. Expected: a value error whose text matches the given message pattern.
    with self.assertRaisesRegex(ValueError, message):
      composite_tensor_ops.composite_tensor_to_variants(value(), spec)

  def testDecodingEmptyNonScalarTensorError(self):
    # [FUNCTION DOC]
    # Scenario: only in eager mode (returns early otherwise, since an empty variant tensor cannot
    # be built in graph mode), the raw variant-to-components op is applied to an empty
    # variant tensor. Expected: an invalid-argument error saying it must not be an empty variant
    # tensor.
    if not context.executing_eagerly():
      # Creating a variant tensor of an empty list is not allowed in eager mode.
      return

    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'must not be an empty variant tensor'):
      gen_composite_tensor_ops.CompositeTensorVariantToComponents(
          encoded=constant_op.constant([], dtype=dtypes.variant),
          metadata='',
          Tcomponents=[dtypes.int32])

  def testDecodingInvalidEncodedInputError(self):
    # [FUNCTION DOC]
    # Scenario: the raw variant-to-components op is given an empty tensor list variant, which is
    # not a composite tensor encoding. Expected: an invalid-argument error saying the input is
    # not a valid composite tensor variant tensor.
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'not a valid CompositeTensorVariant tensor'):
      self.evaluate(
          gen_composite_tensor_ops.CompositeTensorVariantToComponents(
              encoded=gen_list_ops.EmptyTensorList(
                  element_dtype=dtypes.int32,
                  element_shape=[1, 2],
                  max_num_elements=2),
              metadata='',
              Tcomponents=[dtypes.int32]))

  def testRoundTripThroughTensorProto(self):
    # [FUNCTION DOC]
    # Scenario: a ragged tensor is encoded to a variant, serialized to a tensor proto string,
    # parsed back as a variant tensor and decoded with the original spec. Expected: the decoded
    # value equals the original ragged tensor.
    value = ragged_factory_ops.constant([[1, 2], [3], [4, 5, 6]])
    encoded = composite_tensor_ops.composite_tensor_to_variants(value)
    proto = parsing_ops.SerializeTensor(tensor=encoded)
    parsed = parsing_ops.ParseTensor(serialized=proto, out_type=dtypes.variant)
    decoded = composite_tensor_ops.composite_tensor_from_variant(
        parsed, value._type_spec)
    self.assertAllEqual(value, decoded)

  def testGradient(self):
    # [FUNCTION DOC]
    # Scenario: a ragged float tensor is doubled, encoded, decoded, and its values are multiplied
    # by a range of six numbers. The gradient of the result values with respect to the input values
    # is computed with a gradient tape in eager mode or the graph gradient function otherwise.
    # Expected: the gradient equals 0, 2, 4, 6, 8, 10 (the range times the factor two).

    def func(x):
      # [FUNCTION DOC]
      # Helper for the gradient test: doubles the ragged input, round-trips it through the variant
      # encoding and decoding with its own spec, and returns the decoded ragged tensor with values
      # scaled elementwise by a range of six floats.
      x2 = composite_tensor_ops.composite_tensor_to_variants(x * 2)
      x3 = composite_tensor_ops.composite_tensor_from_variant(x2, x._type_spec)
      return x3.with_values(x3.values * math_ops.range(6.0))

    x = ragged_factory_ops.constant([[1.0, 2.0, 3.0], [4.0], [5.0, 6.0]])
    if context.executing_eagerly():
      with backprop.GradientTape() as t:
        t.watch(x.values)
        y = func(x)
        g = t.gradient(y.values, x.values)
    else:
      y = func(x)
      g = gradients_impl.gradients(ys=y.values, xs=x.values)[0]
    self.assertAllClose(g, [0.0, 2.0, 4.0, 6.0, 8.0, 10.0])


if __name__ == '__main__':
  googletest.main()
