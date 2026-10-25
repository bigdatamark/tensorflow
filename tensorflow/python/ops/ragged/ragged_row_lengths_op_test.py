# [MODULE DOC]
# Unit tests for the row-lengths method of ragged tensors: parameterized cases for different
# ranks, ragged ranks and axes (including negative axes) and error cases for out-of-range axes.
# Runs the test main when executed as a script.
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
"""Tests for ragged.row_lengths."""

from absl.testing import parameterized

from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedRowLengthsOp(test_util.TensorFlowTestCase,
                         parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the row-lengths method of ragged tensors, run in graph and eager modes with
  # parameterized inputs. No shared fixtures.

  @parameterized.parameters([
      # Docstring Example
      dict(
          rt_input=[[[3, 1, 4], [1]], [], [[5, 9], [2]], [[6]], []],
          expected=[2, 0, 2, 1, 0]),
      dict(
          rt_input=[[[3, 1, 4], [1]], [], [[5, 9], [2]], [[6]], []],
          axis=2,
          expected=[[3, 1], [], [2, 1], [1], []]),

      # 2D Tensor (1 ragged dimension)
      dict(
          rt_input=[['a'], ['b', 'c', 'd'], ['e'], [], ['f']],
          expected=[1, 3, 1, 0, 1]),
      dict(
          rt_input=[['a'], ['b', 'c', 'd'], ['e'], [], ['f']],
          axis=0,
          expected=5),
      dict(
          rt_input=[['a', 'b', 'c', 'd', 'e', 'f', 'g']],
          expected=[7]),
      dict(
          rt_input=[[], ['a', 'b', 'c', 'd', 'e', 'f', 'g'], []],
          expected=[0, 7, 0]),
      dict(
          rt_input=[],
          ragged_rank=1,
          expected=[]),
      dict(
          rt_input=[],
          ragged_rank=1,
          axis=0,
          expected=0),

      # 3D Tensor (1 ragged dimension)
      dict(
          rt_input=[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10]]],
          ragged_rank=1,
          axis=0,
          expected=2),
      dict(
          rt_input=[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10]]],
          ragged_rank=1,
          axis=1,
          expected=[3, 2]),
      dict(
          rt_input=[[[1, 2], [3, 4], [5, 6]], [[7, 8], [9, 10]]],
          ragged_rank=1,
          axis=2,
          expected=[[2, 2, 2], [2, 2]],
          expected_ragged_rank=1),

      # 3D Tensor (2 ragged dimensions)
      dict(
          rt_input=[[[1, 2], [3, 4, 5], []], [[6, 7, 8, 9], [10]]],
          axis=0,
          expected=2),
      dict(
          rt_input=[[[1, 2], [3, 4, 5], []], [[6, 7, 8, 9], [10]]],
          axis=-3,
          expected=2),
      dict(
          rt_input=[[[1, 2], [3, 4, 5], []], [[6, 7, 8, 9], [10]]],
          axis=1,
          expected=[3, 2]),
      dict(
          rt_input=[[[1, 2], [3, 4, 5], []], [[6, 7, 8, 9], [10]]],
          axis=-2,
          expected=[3, 2]),
      dict(
          rt_input=[[[1, 2], [3, 4, 5], []], [[6, 7, 8, 9], [10]]],
          axis=2,
          expected=[[2, 3, 0], [4, 1]],
          expected_ragged_rank=1),
      dict(
          rt_input=[[[1, 2], [3, 4, 5], []], [[6, 7, 8, 9], [10]]],
          axis=-1,
          expected=[[2, 3, 0], [4, 1]],
          expected_ragged_rank=1),
  ])  # pyformat: disable
  def testRowLengths(self,
                     rt_input,
                     expected,
                     axis=1,
                     ragged_rank=None,
                     expected_ragged_rank=None):
    # [FUNCTION DOC]
    # Parameters: rt_input (nested list), expected (expected lengths), axis (defaults to one),
    # ragged_rank (optional, for building the constant) and expected_ragged_rank (optional).
    # Builds a ragged tensor, computes row lengths along the axis, asserts they equal the
    # expectation, and if an expected ragged rank is given asserts the result has that ragged
    # rank (or, when the result is not ragged, that the expected ragged rank is zero).
    rt = ragged_factory_ops.constant(rt_input, ragged_rank=ragged_rank)
    lengths = rt.row_lengths(axis)
    self.assertAllEqual(lengths, expected)
    if expected_ragged_rank is not None:
      if isinstance(lengths, ragged_tensor.RaggedTensor):
        self.assertEqual(lengths.ragged_rank, expected_ragged_rank)
      else:
        self.assertEqual(0, expected_ragged_rank)

  @parameterized.parameters([
      dict(  # axis=2 out of bounds: expected -2<=axis<2.
          rt_input=[[10, 20], [30]],
          axis=2,
          exception=(ValueError, errors.InvalidArgumentError)),
      dict(  # axis=-3 out of bounds: expected -2<=axis<2.
          rt_input=[[2, 3, 0], [4, 1, 2]],
          axis=-3,
          exception=(ValueError, errors.InvalidArgumentError)),
  ])
  def testErrors(self, rt_input, exception, message=None, axis=1):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no case supplies a message, so the expected message is null and the
    # regular-expression assertion accepts any error text; only the error type is checked.
    # Parameters: rt_input (nested list), exception (expected error type or tuple of types),
    # optional message and axis (defaults to one). Asserts that computing row lengths along an
    # out-of-bounds axis raises the expected error.
    rt = ragged_factory_ops.constant(rt_input)
    with self.assertRaisesRegex(exception, message):
      rt.row_lengths(axis)


if __name__ == '__main__':
  googletest.main()
