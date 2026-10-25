# [MODULE DOC]
# Unit tests for the ragged squeeze operation. Compares ragged squeeze results against the dense
# squeeze on equivalent data (including data converted from dense tensors), checks genuinely
# ragged and string inputs against expected ragged values, and checks the errors: invalid-argument
# when a requested dimension is not of size one, a value error when no axis is given, and a type
# error when the axis is a tensor rather than a list. Importing it only defines the test class.
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
"""Tests for ragged.squeeze."""

from absl.testing import parameterized

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.ragged import ragged_conversion_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_squeeze_op
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedSqueezeTest(test_util.TensorFlowTestCase,
                        parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the ragged squeeze operation, run in both graph and eager modes (class
  # decorator). Stateless; every test builds ragged or dense constants from nested lists and
  # compares ragged squeeze output with dense squeeze output or an expected ragged tensor.

  @parameterized.parameters([
      {
          'input_list': []
      },
      {
          'input_list': [[]],
          'squeeze_ranks': [0]
      },
      {
          'input_list': [[[[], []], [[], []]]],
          'squeeze_ranks': [0]
      },
  ])
  def test_passing_empty(self, input_list, squeeze_ranks=None):
    # [FUNCTION DOC]
    # Tests squeeze on empty nested lists. Inputs: a nested list and optional list of dimensions
    # to squeeze. Squeezes the ragged constant and the dense constant of the same data, converts
    # the ragged result to dense and asserts equality. Returns nothing.
    rt = ragged_squeeze_op.squeeze(
        ragged_factory_ops.constant(input_list), squeeze_ranks)
    dt = array_ops.squeeze(constant_op.constant(input_list), squeeze_ranks)
    self.assertAllEqual(ragged_conversion_ops.to_tensor(rt), dt)

  @parameterized.parameters([
      {
          'input_list': [[1]],
          'squeeze_ranks': [0]
      },
      {
          'input_list': [[1]],
          'squeeze_ranks': [0, 1]
      },
      {
          'input_list': [[1, 2]],
          'squeeze_ranks': [0]
      },
      {
          'input_list': [[1], [2]],
          'squeeze_ranks': [1]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [0]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [1]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [3]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [0, 3]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [0, 1]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [1, 3]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [0, 1, 3]
      },
      {
          'input_list': [[[1], [2]], [[3], [4]]],
          'squeeze_ranks': [2]
      },
      {
          'input_list': [[1], [2]],
          'squeeze_ranks': [-1]
      },
  ])
  def test_passing_simple(self, input_list, squeeze_ranks=None):
    # [FUNCTION DOC]
    # Tests squeeze on small uniform nested lists with various positive and negative dimension
    # lists. Squeezes the ragged constant and the dense constant of the same data and asserts the
    # ragged result converted to dense equals the dense result. Returns nothing.
    rt = ragged_squeeze_op.squeeze(
        ragged_factory_ops.constant(input_list), squeeze_ranks)
    dt = array_ops.squeeze(constant_op.constant(input_list), squeeze_ranks)
    self.assertAllEqual(ragged_conversion_ops.to_tensor(rt), dt)

  @parameterized.parameters([
      # ragged_conversion_ops.from_tensor does not work for this
      # {'input_list': [1]},
      {
          'input_list': [[1]],
          'squeeze_ranks': [0]
      },
      {
          'input_list': [[1, 2]],
          'squeeze_ranks': [0]
      },
      {
          'input_list': [[1], [2]],
          'squeeze_ranks': [1]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [0]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [1]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [3]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [0, 3]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [0, 1]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [1, 3]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [0, 1, 3]
      },
      {
          'input_list': [[[1], [2]], [[3], [4]]],
          'squeeze_ranks': [2]
      },
  ])
  def test_passing_simple_from_dense(self, input_list, squeeze_ranks=None):
    # [FUNCTION DOC]
    # Same scenarios as the simple test, but the ragged tensor is produced by converting a dense
    # constant to ragged form. Squeezes both the ragged and the dense version over the given
    # dimensions and asserts the ragged result converted back to dense equals the dense result.
    # Returns nothing.
    dt = constant_op.constant(input_list)
    rt = ragged_conversion_ops.from_tensor(dt)
    rt_s = ragged_squeeze_op.squeeze(rt, squeeze_ranks)
    dt_s = array_ops.squeeze(dt, squeeze_ranks)
    self.assertAllEqual(ragged_conversion_ops.to_tensor(rt_s), dt_s)

  @parameterized.parameters([
      {
          'input_list': [[[[[[1]], [[1, 2]]]], [[[[]], [[]]]]]],
          'output_list': [[[1], [1, 2]], [[], []]],
          'squeeze_ranks': [0, 2, 4]
      },
      {
          'input_list': [[[[[[1]], [[1, 2]]]], [[[[]], [[]]]]]],
          'output_list': [[[[[1]], [[1, 2]]]], [[[[]], [[]]]]],
          'squeeze_ranks': [0]
      },
  ])
  def test_passing_ragged(self, input_list, output_list, squeeze_ranks=None):
    # [FUNCTION DOC]
    # Tests squeeze on genuinely ragged nested data. Inputs: nested list, expected nested list and
    # dimensions to squeeze. Squeezes the ragged constant and asserts it equals the ragged
    # constant built from the expected list. Returns nothing.
    rt = ragged_factory_ops.constant(input_list)
    rt_s = ragged_squeeze_op.squeeze(rt, squeeze_ranks)
    ref = ragged_factory_ops.constant(output_list)
    self.assertAllEqual(rt_s, ref)

  def test_passing_text(self):
    # [FUNCTION DOC]
    # Tests squeeze on a deeply nested ragged tensor of single-character strings spelling two
    # sentences. Squeezes dimensions zero, one, three, six and seven and asserts the result equals
    # the expected shallower ragged tensor of characters. Returns nothing.
    rt = ragged_factory_ops.constant([[[[[[[['H']], [['e']], [['l']], [['l']],
                                           [['o']]],
                                          [[['W']], [['o']], [['r']], [['l']],
                                           [['d']], [['!']]]]],
                                        [[[[['T']], [['h']], [['i']], [['s']]],
                                          [[['i']], [['s']]],
                                          [[['M']], [['e']], [['h']], [['r']],
                                           [['d']], [['a']], [['d']]],
                                          [[['.']]]]]]]])
    output_list = [[['H', 'e', 'l', 'l', 'o'], ['W', 'o', 'r', 'l', 'd', '!']],
                   [['T', 'h', 'i', 's'], ['i', 's'],
                    ['M', 'e', 'h', 'r', 'd', 'a', 'd'], ['.']]]
    ref = ragged_factory_ops.constant(output_list)
    rt_s = ragged_squeeze_op.squeeze(rt, [0, 1, 3, 6, 7])
    self.assertAllEqual(rt_s, ref)

  @parameterized.parameters([
      {
          'input_list': [[]],
          'squeeze_ranks': [1]
      },
      {
          'input_list': [[1, 2]],
          'squeeze_ranks': [1]
      },
      {
          'input_list': [[1], [2]],
          'squeeze_ranks': [0]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [0, 2]
      },
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [2]
      },
      {
          'input_list': [[[1], [2]], [[3], [4]]],
          'squeeze_ranks': [0]
      },
      {
          'input_list': [[[1], [2]], [[3], [4]]],
          'squeeze_ranks': [1]
      },
      {
          'input_list': [[], []],
          'squeeze_ranks': [1]
      },
      {
          'input_list': [[[], []], [[], []]],
          'squeeze_ranks': [1]
      },
  ])
  def test_failing_InvalidArgumentError(self, input_list, squeeze_ranks):
    # [FUNCTION DOC]
    # Tests that squeezing a dimension whose size is not one fails. Inputs: nested list and the
    # dimensions to squeeze. Evaluates the squeeze and asserts an invalid-argument error is
    # raised. Returns nothing.
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(
          ragged_squeeze_op.squeeze(
              ragged_factory_ops.constant(input_list), squeeze_ranks))

  @parameterized.parameters([
      {
          'input_list': [[]]
      },
      {
          'input_list': [[1]]
      },
      {
          'input_list': [[1, 2]]
      },
      {
          'input_list': [[[1], [2]], [[3], [4]]]
      },
      {
          'input_list': [[1]]
      },
      {
          'input_list': [[[1], [2]], [[3], [4]]]
      },
      {
          'input_list': [[[[12], [11]]]]
      },
  ])
  def test_failing_no_squeeze_dim_specified(self, input_list):
    # [FUNCTION DOC]
    # Tests that squeezing a ragged tensor without naming dimensions raises a value error. Input:
    # a nested list. Asserts the error on the call. Returns nothing.
    with self.assertRaises(ValueError):
      ragged_squeeze_op.squeeze(ragged_factory_ops.constant(input_list))

  @parameterized.parameters([
      {
          'input_list': [[[[12], [11]]]],
          'squeeze_ranks': [0, 1, 3]
      },
  ])
  def test_failing_axis_is_not_a_list(self, input_list, squeeze_ranks):
    # [FUNCTION DOC]
    # Tests that passing the dimensions as a tensor instead of a list raises a type error. Inputs:
    # nested list and dimension list (converted to a constant tensor inside the test). Asserts the
    # type error on the call. Returns nothing.
    with self.assertRaises(TypeError):
      tensor_ranks = constant_op.constant(squeeze_ranks)
      ragged_squeeze_op.squeeze(
          ragged_factory_ops.constant(input_list), tensor_ranks)


if __name__ == '__main__':
  googletest.main()
