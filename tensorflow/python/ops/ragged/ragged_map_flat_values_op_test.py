# [MODULE DOC]
# Unit tests for the function that applies an operation to the flat values of ragged tensors
# and rewraps the result with the original row partitions. Covers single, multiple, scalar,
# list, keyword and mixed arguments, non-element-wise ops, various ragged ranks, inner values
# only, partition mismatch errors, output shape errors and uniform row length preservation.
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
"""Tests for ragged_functional_ops.map_flat_values."""

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_functional_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedMapInnerValuesOpTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for applying an operation to the flat values of ragged tensors, run in graph and
  # eager modes. Uses a shared assertion helper and small ragged constants as fixtures.

  def assertRaggedMapInnerValuesReturns(self,
                                        op,
                                        expected,
                                        args=(),
                                        kwargs=None):
    # [FUNCTION DOC]
    # HELPER: shared assertion for the tests. Inputs: op, the operation to apply; expected,
    # nested list; args, positional arguments (default empty); kwargs, keyword arguments
    # (default none, treated as empty). Applies the flat-values mapper and asserts equality.
    kwargs = kwargs or {}
    result = ragged_functional_ops.map_flat_values(op, *args, **kwargs)
    self.assertAllEqual(result, expected)

  def testDocStringExamples(self):
    # [FUNCTION DOC]
    # Checks the documented examples: ones-like on a ragged tensor, elementwise multiply of the
    # tensor with itself, and adding a scalar; asserts the expected nested results.
    """Test the examples in apply_op_to_ragged_values.__doc__."""
    rt = ragged_factory_ops.constant([[1, 2, 3], [], [4, 5], [6]])
    v1 = ragged_functional_ops.map_flat_values(array_ops.ones_like, rt)
    v2 = ragged_functional_ops.map_flat_values(math_ops.multiply, rt, rt)
    v3 = ragged_functional_ops.map_flat_values(math_ops.add, rt, 5)
    self.assertAllEqual(v1, [[1, 1, 1], [], [1, 1], [1]])
    self.assertAllEqual(v2, [[1, 4, 9], [], [16, 25], [36]])
    self.assertAllEqual(v3, [[6, 7, 8], [], [9, 10], [11]])

  def testOpWithSingleRaggedTensorArg(self):
    # [FUNCTION DOC]
    # Applies zeros-like to one ragged tensor; expects zeros with the same row structure.
    tensor = ragged_factory_ops.constant([[1, 2, 3], [], [4, 5]])
    self.assertRaggedMapInnerValuesReturns(
        op=array_ops.zeros_like,
        args=(tensor,),
        expected=[[0, 0, 0], [], [0, 0]])

  def testOpWithTwoRaggedTensorArgs(self):
    # [FUNCTION DOC]
    # Multiplies two ragged tensors with identical row structure; expects elementwise products.
    x = ragged_factory_ops.constant([[3, 1, 4], [], [1, 5]])
    y = ragged_factory_ops.constant([[1, 2, 3], [], [4, 5]])
    self.assertRaggedMapInnerValuesReturns(
        op=math_ops.multiply, args=(x, y), expected=[[3, 2, 12], [], [4, 25]])

  def testOpWithRaggedTensorAndScalarArgs(self):
    # [FUNCTION DOC]
    # Multiplies a plain scalar (first argument) by a ragged tensor; expects each value scaled.
    y = ragged_factory_ops.constant([[1, 2, 3], [], [4, 5]])
    self.assertRaggedMapInnerValuesReturns(
        op=math_ops.multiply, args=(5, y), expected=[[5, 10, 15], [], [20, 25]])

  def testOpWithThreeRaggedTensorArgs(self):
    # [FUNCTION DOC]
    # Applies a three-argument conditional select to a boolean ragged tensor and two string
    # ragged tensors; expects picks from the first string tensor where true, else the second.
    condition = ragged_factory_ops.constant(
        [[True, True, False], [], [True, False]])  # pyformat: disable
    x = ragged_factory_ops.constant([['a', 'b', 'c'], [], ['d', 'e']])
    y = ragged_factory_ops.constant([['A', 'B', 'C'], [], ['D', 'E']])
    self.assertRaggedMapInnerValuesReturns(
        op=array_ops.where_v2,
        args=(condition, x, y),
        expected=[[b'a', b'b', b'C'], [], [b'd', b'E']])

  def testOpWithRaggedTensorListArg(self):
    # [FUNCTION DOC]
    # Passes a list of ragged tensors (with one repeated) to an n-ary add; expects the
    # elementwise sum with row structure preserved.
    x = ragged_factory_ops.constant([[1, 2, 3], [], [4, 5]])
    y = ragged_factory_ops.constant([[10, 20, 30], [], [40, 50]])
    self.assertRaggedMapInnerValuesReturns(
        op=math_ops.add_n,
        args=([x, y, x],),
        expected=[[12, 24, 36], [], [48, 60]])

  def testOpWithKeywordArgs(self):
    # [FUNCTION DOC]
    # Passes the two ragged tensors to multiply purely as keyword arguments; expects products.
    x = ragged_factory_ops.constant([[3, 1, 4], [], [1, 5]])
    y = ragged_factory_ops.constant([[1, 2, 3], [], [4, 5]])
    self.assertRaggedMapInnerValuesReturns(
        op=math_ops.multiply,
        kwargs=dict(x=x, y=y),
        expected=[[3, 2, 12], [], [4, 25]])

  def testOpWithMixedPositionalAndKeywordArgs(self):
    # [FUNCTION DOC]
    # Passes one ragged tensor positionally and one by keyword to multiply; expects products.
    x = ragged_factory_ops.constant([[3, 1, 4], [], [1, 5]])
    y = ragged_factory_ops.constant([[1, 2, 3], [], [4, 5]])
    self.assertRaggedMapInnerValuesReturns(
        op=math_ops.multiply,
        args=(x,),
        kwargs=dict(y=y),
        expected=[[3, 2, 12], [], [4, 25]])

  def testNonElementWiseOp(self):
    # [FUNCTION DOC]
    # Applies a sum-reduction over axis one to the flat values of a ragged tensor whose values
    # are vectors; expects each inner vector reduced to its sum, with rows preserved.
    x = ragged_factory_ops.constant(
        [[[3, 1, 4], [1, 5, 9], [2, 6, 5]], [], [[3, 5, 8], [9, 7, 9]]],
        ragged_rank=1)
    self.assertRaggedMapInnerValuesReturns(
        op=math_ops.reduce_sum,
        kwargs={
            'input_tensor': x,
            'axis': 1,
        },
        expected=[[8, 15, 13], [], [16, 25]])

  def testOpWithRaggedRankGreaterThanOne(self):
    # [FUNCTION DOC]
    # Multiplies pairs of tensors of ragged rank zero (plain lists), one, two and three;
    # asserts elementwise products with the nested structure preserved at every rank.
    # ragged_rank=0
    x0 = [3, 1, 4, 1, 5, 9, 2, 6, 5]
    y0 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    self.assertAllEqual(
        math_ops.multiply(x0, y0), [3, 2, 12, 4, 25, 54, 14, 48, 45])

    # ragged_rank=1
    x1 = ragged_factory_ops.constant([[3, 1, 4], [], [1, 5], [9, 2], [6, 5]])
    y1 = ragged_factory_ops.constant([[1, 2, 3], [], [4, 5], [6, 7], [8, 9]])
    self.assertRaggedMapInnerValuesReturns(
        op=math_ops.multiply,
        args=(x1, y1),
        expected=[[3, 2, 12], [], [4, 25], [54, 14], [48, 45]])

    # ragged_rank=2
    x2 = ragged_factory_ops.constant([[[3, 1, 4]], [], [[], [1, 5]],
                                      [[9, 2], [6, 5]]])
    y2 = ragged_factory_ops.constant([[[1, 2, 3]], [], [[], [4, 5]],
                                      [[6, 7], [8, 9]]])
    self.assertRaggedMapInnerValuesReturns(
        op=math_ops.multiply,
        args=(x2, y2),
        expected=[[[3, 2, 12]],          # row 0
                  [],                    # row 1
                  [[], [4, 25]],         # row 2
                  [[54, 14], [48, 45]]   # row 3
                 ])  # pyformat: disable

    # ragged_rank=3
    x3 = ragged_factory_ops.constant([[[[3, 1, 4]], []], [], [[[], [1, 5]]],
                                      [[[9, 2], [6, 5]]]])
    y3 = ragged_factory_ops.constant([[[[1, 2, 3]], []], [], [[[], [4, 5]]],
                                      [[[6, 7], [8, 9]]]])
    self.assertRaggedMapInnerValuesReturns(
        op=math_ops.multiply,
        args=(x3, y3),
        expected=[
            [[[3, 2, 12]], []],       # row 0
            [],                       # row 1
            [[[], [4, 25]]],          # row 2
            [[[54, 14], [48, 45]]]    # row 3
        ])  # pyformat: disable

  def testOpWithRaggedRankThree(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says rank three, but the tensors built have ragged rank two.
    # Multiplies two nested ragged tensors; expects elementwise products.
    x = ragged_factory_ops.constant([[[3, 1, 4]], [], [[], [1, 5]]])
    y = ragged_factory_ops.constant([[[1, 2, 3]], [], [[], [4, 5]]])
    self.assertRaggedMapInnerValuesReturns(
        op=math_ops.multiply,
        args=(x, y),
        expected=[[[3, 2, 12]], [], [[], [4, 25]]])

  def testOpWithInnerValuesOnly(self):
    # [FUNCTION DOC]
    # Passes only dense tensors (a matrix and a scalar), no ragged input; expects the plain
    # operation result.
    x = constant_op.constant([[1, 2], [3, 4], [5, 6]])
    y = constant_op.constant(2)
    self.assertRaggedMapInnerValuesReturns(
        op=math_ops.multiply, args=(x, y), expected=[[2, 4], [6, 8], [10, 12]])

  def testRaggedTensorSplitsRaggedRankMismatchError(self):
    # [FUNCTION DOC]
    # Adds ragged tensors of different ragged ranks; expects a value error saying all ragged
    # inputs must share the same ragged rank.
    x = ragged_factory_ops.constant([[3, 1, 4], [], [1, 5]])
    y = ragged_factory_ops.constant([[[3, 1, 4], []], [], [[1, 5]]])
    with self.assertRaisesRegex(
        ValueError, r'All ragged inputs must have the same ragged_rank.'):
      ragged_functional_ops.map_flat_values(math_ops.add, x, y)

  def testRaggedTensorSplitsValueMismatchError(self):
    # [FUNCTION DOC]
    # Adds ragged tensors whose row partitions differ; expects a value or invalid-argument error
    # about incompatible partitions. Then adds tensors whose flat values have different outer
    # sizes (one using an unknown-shape row-splits placeholder); expects a value error listing
    # the mismatched sizes.
    x = ragged_factory_ops.constant([[3, 1, 4], [], [1, 5]])
    y = ragged_factory_ops.constant([[1], [2, 3], [4, 5]])
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                r'partitions have incompatible'):
      ragged_functional_ops.map_flat_values(math_ops.add, x, y)

    z_splits = array_ops.placeholder_with_default(
        constant_op.constant([0, 3], dtypes.int64), None)
    z = ragged_tensor.RaggedTensor.from_row_splits([0, 1, 2], z_splits)
    with self.assertRaisesRegex(
        ValueError,
        r"Input RaggedTensors' flat_values must all have the same "
        r'outer-dimension size.  Got sizes: \{3, 5\}'):
      ragged_functional_ops.map_flat_values(math_ops.add, x, z)

  def testRaggedTensorShapeMismatchError(self):
    # [FUNCTION DOC]
    # Applies an argmax (which collapses the outer dimension) to a ragged tensor; expects a value
    # error saying the output must keep the outer-dimension size of the flat values.
    x = ragged_factory_ops.constant([[1, 2, 3], [4, 5]])
    with self.assertRaisesRegex(
        ValueError, r'tf.ragged.map_flat_values requires that the output of '
        '`op` have the same outer-dimension size as flat_values of any ragged '
        r'inputs. \(output shape: \(\); expected outer dimension size: 5\)'):
      ragged_functional_ops.map_flat_values(math_ops.argmax, x)

  def testRaggedTensorSplitsMismatchErrorAtRuntime(self):
    # [FUNCTION DOC]
    # Builds two ragged tensors with same-size values but different row splits supplied through
    # unknown-shape placeholders, so the mismatch is only found at run time. Evaluates the add
    # and expects an error about incompatible partitions.
    splits1 = array_ops.placeholder_with_default(
        constant_op.constant([0, 3, 3, 5], dtypes.int64), None)
    splits2 = array_ops.placeholder_with_default(
        constant_op.constant([0, 1, 3, 5], dtypes.int64), None)
    x = ragged_tensor.RaggedTensor.from_row_splits([3, 1, 4, 1, 5], splits1)
    y = ragged_tensor.RaggedTensor.from_row_splits([1, 2, 3, 4, 5], splits2)
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                r'partitions have incompatible'):
      self.evaluate(ragged_functional_ops.map_flat_values(math_ops.add, x, y))

  def testRaggedMapFnPreservesUniformRowLength(self):
    # [FUNCTION DOC]
    # Uses two equal tensors, one with a uniform row length and one without. Adds them in both
    # orders, and via an n-ary add of lists, asserting the result keeps the uniform row length.
    # x and y are equal, except that x has uniform_row_length and y does not.
    x = ragged_tensor.RaggedTensor.from_uniform_row_length(
        ragged_factory_ops.constant([[1, 2], [3]]), uniform_row_length=2)
    y = ragged_factory_ops.constant([[[1, 2], [3]]])

    a = ragged_functional_ops.map_flat_values(math_ops.add, x, y)
    self.assertAllEqual(x.uniform_row_length, a.uniform_row_length)

    b = ragged_functional_ops.map_flat_values(math_ops.add, y, x)
    self.assertAllEqual(x.uniform_row_length, b.uniform_row_length)

    c = ragged_functional_ops.map_flat_values(math_ops.add_n, [x, x])
    self.assertAllEqual(x.uniform_row_length, c.uniform_row_length)

    d = ragged_functional_ops.map_flat_values(math_ops.add_n, [y, x, y])
    self.assertAllEqual(x.uniform_row_length, d.uniform_row_length)


if __name__ == '__main__':
  googletest.main()
