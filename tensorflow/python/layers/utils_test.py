# [MODULE DOC]
# Unit tests for the layers utility module: format conversion, argument normalization, convolution
# length arithmetic, and constant-value extraction from conditional helpers. Uses the standard
# TensorFlow test case base class and runs via the test main entry point when executed directly.
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
"""Tests for tf.layers.utils."""

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class ConvUtilsTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the pure-Python convolution and argument-normalization helpers of the layers
  # utility module. Stateless; each test method calls helpers and asserts on returned values.

  def testConvertDataFormat(self):
    # [FUNCTION DOC]
    # Checks that both data formats map to the right layout strings for ranks 3, 4 and 5, and that
    # an unknown format name raises a value error.
    self.assertEqual('NCDHW', utils.convert_data_format('channels_first', 5))
    self.assertEqual('NCHW', utils.convert_data_format('channels_first', 4))
    self.assertEqual('NCW', utils.convert_data_format('channels_first', 3))
    self.assertEqual('NHWC', utils.convert_data_format('channels_last', 4))
    self.assertEqual('NWC', utils.convert_data_format('channels_last', 3))
    self.assertEqual('NDHWC', utils.convert_data_format('channels_last', 5))

    with self.assertRaises(ValueError):
      utils.convert_data_format('invalid', 2)

  def testNormalizeTuple(self):
    # [FUNCTION DOC]
    # Checks that a single integer is repeated to the requested length and a matching tuple is
    # returned unchanged, and that a wrong-length tuple or a null value raises a value error.
    self.assertEqual((2, 2, 2), utils.normalize_tuple(2, n=3, name='strides'))
    self.assertEqual(
        (2, 1, 2), utils.normalize_tuple((2, 1, 2), n=3, name='strides'))

    with self.assertRaises(ValueError):
      utils.normalize_tuple((2, 1), n=3, name='strides')

    with self.assertRaises(ValueError):
      utils.normalize_tuple(None, n=3, name='strides')

  def testNormalizeDataFormat(self):
    # [FUNCTION DOC]
    # Checks that mixed-case or upper-case format names are lowercased to the canonical names and
    # that an invalid name raises a value error.
    self.assertEqual(
        'channels_last', utils.normalize_data_format('Channels_Last'))
    self.assertEqual(
        'channels_first', utils.normalize_data_format('CHANNELS_FIRST'))

    with self.assertRaises(ValueError):
      utils.normalize_data_format('invalid')

  def testNormalizePadding(self):
    # [FUNCTION DOC]
    # Checks that same and valid padding names are lowercased regardless of input case and that an
    # invalid padding name raises a value error.
    self.assertEqual('same', utils.normalize_padding('SAME'))
    self.assertEqual('valid', utils.normalize_padding('VALID'))

    with self.assertRaises(ValueError):
      utils.normalize_padding('invalid')

  def testConvOutputLength(self):
    # [FUNCTION DOC]
    # Checks convolution output lengths for same, valid and full padding with strides 1 and 2, and
    # one case with dilation 2, against hand-computed expected integers.
    self.assertEqual(4, utils.conv_output_length(4, 2, 'same', 1, 1))
    self.assertEqual(2, utils.conv_output_length(4, 2, 'same', 2, 1))
    self.assertEqual(3, utils.conv_output_length(4, 2, 'valid', 1, 1))
    self.assertEqual(2, utils.conv_output_length(4, 2, 'valid', 2, 1))
    self.assertEqual(5, utils.conv_output_length(4, 2, 'full', 1, 1))
    self.assertEqual(3, utils.conv_output_length(4, 2, 'full', 2, 1))
    self.assertEqual(2, utils.conv_output_length(5, 2, 'valid', 2, 2))

  def testConvInputLength(self):
    # [FUNCTION DOC]
    # Checks the inverse input-length computation for same, valid and full padding with strides 1
    # and 2 against hand-computed expected integers.
    self.assertEqual(3, utils.conv_input_length(4, 2, 'same', 1))
    self.assertEqual(2, utils.conv_input_length(2, 2, 'same', 2))
    self.assertEqual(4, utils.conv_input_length(3, 2, 'valid', 1))
    self.assertEqual(4, utils.conv_input_length(2, 2, 'valid', 2))
    self.assertEqual(3, utils.conv_input_length(4, 2, 'full', 1))
    self.assertEqual(4, utils.conv_input_length(3, 2, 'full', 2))

  def testDeconvOutputLength(self):
    # [FUNCTION DOC]
    # Checks transposed-convolution output lengths for same, valid and full padding with strides 1
    # and 2 against hand-computed expected integers.
    self.assertEqual(4, utils.deconv_output_length(4, 2, 'same', 1))
    self.assertEqual(8, utils.deconv_output_length(4, 2, 'same', 2))
    self.assertEqual(5, utils.deconv_output_length(4, 2, 'valid', 1))
    self.assertEqual(8, utils.deconv_output_length(4, 2, 'valid', 2))
    self.assertEqual(3, utils.deconv_output_length(4, 2, 'full', 1))
    self.assertEqual(6, utils.deconv_output_length(4, 2, 'full', 2))


class ConstantValueTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the constant-value and smart conditional helpers of the layers utility
  # module. Stateless; one test method.

  @test_util.run_deprecated_v1
  def testConstantValue(self):
    # [FUNCTION DOC]
    # Runs in legacy graph mode. Builds two constant tensors as branches and checks that the constant
    # value of the smart conditional result is the first or second constant for boolean and for
    # integer 1/0 predicates, is nothing for a dynamic placeholder predicate, and that passing a
    # plain integer other than 0/1 directly to the constant-value helper raises a type error.
    f1 = lambda: constant_op.constant(5)
    f2 = lambda: constant_op.constant(32)

    # Boolean pred
    self.assertEqual(5, utils.constant_value(utils.smart_cond(True, f1, f2)))
    self.assertEqual(32, utils.constant_value(utils.smart_cond(False, f1, f2)))

    # Integer pred
    self.assertEqual(5, utils.constant_value(utils.smart_cond(1, f1, f2)))
    self.assertEqual(32, utils.constant_value(utils.smart_cond(0, f1, f2)))

    # Unknown pred
    pred = array_ops.placeholder_with_default(True, shape=())
    self.assertIsNone(utils.constant_value(utils.smart_cond(pred, f1, f2)))

    #Error case
    with self.assertRaises(TypeError):
      utils.constant_value(5)


if __name__ == '__main__':
  test.main()
