# [MODULE DOC]
# Unit tests for the batch-slicing support of linear operators (the slicing module in the linear
# algebra ops package). Most tests call the private single-parameter slicing helper with a
# zero-filled parameter tensor, a number-of-dimensions mapping, a tuple of slice specifications
# and a batch shape, and assert the resulting shape; the last tests slice real linear operators.
# Covers steps, ranges, integer and tensor indices, new axes, ellipsis, broadcast batch
# dimensions and error cases. All tests run in graph and eager modes.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.ops.linalg import slicing
from tensorflow.python.platform import test

linalg = linalg_lib


class _MakeSlices(object):
  # [CLASS DOC]
  # HELPER: tiny utility whose indexing returns the index expression itself as a tuple, letting
  # tests write slice specifications with normal indexing notation. A single module-level
  # instance is created for the tests.

  def __getitem__(self, slices):
    # [FUNCTION DOC]
    # HELPER: returns the given index expression as a tuple; a tuple is returned unchanged and
    # a single index or slice is wrapped in a one-element tuple.
    return slices if isinstance(slices, tuple) else (slices,)


make_slices = _MakeSlices()


@test_util.run_all_in_graph_and_eager_modes
class SlicingTest(test.TestCase):
  # [CLASS DOC]
  # Test case for batch slicing of parameters and linear operators; every test runs in graph and
  # eager modes. Uses the module-level slice builder to write slice specifications.
  """Tests for slicing LinearOperators."""

  def test_single_param_slice_withstep_broadcastdim(self):
    # [FUNCTION DOC]
    # Scenario: a parameter with a size-one batch dimension (shape one by one by three) is sliced
    # with a negative step and an out-of-range start, against batch shape two by seven. Expects
    # the shape unchanged, since broadcast dimensions are not sliced.
    event_dim = 3
    sliced = slicing._slice_single_param(
        array_ops.zeros([1, 1, event_dim]),
        param_ndims_to_matrix_ndims=1,
        slices=make_slices[44:-52:-3, -94::],
        batch_shape=constant_op.constant([2, 7], dtype=dtypes.int32))
    self.assertAllEqual((1, 1, event_dim), self.evaluate(sliced).shape)

  def test_single_param_slice_stop_leadingdim(self):
    # [FUNCTION DOC]
    # Scenario: a 7x6x5x4x3 parameter with two matrix dimensions is sliced to the first two
    # entries of the leading batch dimension. Expects shape 2x6x5x4x3.
    sliced = slicing._slice_single_param(
        array_ops.zeros([7, 6, 5, 4, 3]),
        param_ndims_to_matrix_ndims=2,
        slices=make_slices[:2],
        batch_shape=constant_op.constant([7, 6, 5], dtype=dtypes.int32))
    self.assertAllEqual((2, 6, 5, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_stop_trailingdim(self):
    # [FUNCTION DOC]
    # Scenario: an ellipsis followed by a stop-two slice selects the last batch dimension.
    # Expects shape 7x6x2x4x3.
    sliced = slicing._slice_single_param(
        array_ops.zeros([7, 6, 5, 4, 3]),
        param_ndims_to_matrix_ndims=2,
        slices=make_slices[..., :2],
        batch_shape=constant_op.constant([7, 6, 5]))
    self.assertAllEqual((7, 6, 2, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_stop_broadcastdim(self):
    # [FUNCTION DOC]
    # Scenario: the second batch dimension has size one in the parameter but six in the batch
    # shape; a stop-two slice is applied to it. Expects the size-one dimension left as is.
    sliced = slicing._slice_single_param(
        array_ops.zeros([7, 1, 5, 4, 3]),
        param_ndims_to_matrix_ndims=2,
        slices=make_slices[:, :2],
        batch_shape=constant_op.constant([7, 6, 5]))
    self.assertAllEqual((7, 1, 5, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_newaxis_leading(self):
    # [FUNCTION DOC]
    # Scenario: a new axis is inserted after the first batch dimension. Expects shape
    # 7x1x6x5x4x3.
    sliced = slicing._slice_single_param(
        array_ops.zeros([7, 6, 5, 4, 3]),
        param_ndims_to_matrix_ndims=2,
        slices=make_slices[:, array_ops.newaxis],
        batch_shape=constant_op.constant([7, 6, 5]))
    self.assertAllEqual((7, 1, 6, 5, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_newaxis_trailing(self):
    # [FUNCTION DOC]
    # Scenario: an ellipsis, then a new axis, then a full slice, so the new axis is inserted just
    # before the last batch dimension. Expects shape 7x6x1x5x4x3.
    sliced = slicing._slice_single_param(
        array_ops.zeros([7, 6, 5, 4, 3]),
        param_ndims_to_matrix_ndims=2,
        slices=make_slices[..., array_ops.newaxis, :],
        batch_shape=constant_op.constant([7, 6, 5]))
    self.assertAllEqual((7, 6, 1, 5, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_start(self):
    # [FUNCTION DOC]
    # Scenario: a start-two slice on the second batch dimension (size six). Expects that
    # dimension to shrink to four: shape 7x4x5x4x3.
    sliced = slicing._slice_single_param(
        array_ops.zeros([7, 6, 5, 4, 3]),
        param_ndims_to_matrix_ndims=2,
        slices=make_slices[:, 2:],
        batch_shape=constant_op.constant([7, 6, 5]))
    self.assertAllEqual((7, 4, 5, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_start_broadcastdim(self):
    # [FUNCTION DOC]
    # Scenario: a start-two slice on a batch dimension that is size one in the parameter (but six
    # in the batch shape). Expects the broadcast dimension left untouched: shape 7x1x5x4x3.
    sliced = slicing._slice_single_param(
        array_ops.zeros([7, 1, 5, 4, 3]),
        param_ndims_to_matrix_ndims=2,
        slices=make_slices[:, 2:],
        batch_shape=constant_op.constant([7, 6, 5]))
    self.assertAllEqual((7, 1, 5, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_int(self):
    # [FUNCTION DOC]
    # Scenario: an integer index on the second batch dimension. Expects that dimension removed:
    # shape 7x5x4x3.
    sliced = slicing._slice_single_param(
        array_ops.zeros([7, 6, 5, 4, 3]),
        param_ndims_to_matrix_ndims=2,
        slices=make_slices[:, 2],
        batch_shape=constant_op.constant([7, 6, 5]))
    self.assertAllEqual((7, 5, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_int_broadcastdim(self):
    # [FUNCTION DOC]
    # Scenario: an integer index on a batch dimension that is size one in the parameter. Expects
    # the dimension removed anyway: shape 7x5x4x3.
    sliced = slicing._slice_single_param(
        array_ops.zeros([7, 1, 5, 4, 3]),
        param_ndims_to_matrix_ndims=2,
        slices=make_slices[:, 2],
        batch_shape=constant_op.constant([7, 6, 5]))
    self.assertAllEqual((7, 5, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_tensor(self):
    # [FUNCTION DOC]
    # Scenario: the parameter has unknown static shape and the index on the second batch
    # dimension is a scalar tensor (value two) rather than a Python integer. Expects that
    # dimension removed: shape 7x5x4x3.
    param = array_ops.placeholder_with_default(
        array_ops.zeros([7, 6, 5, 4, 3]), shape=None)
    idx = array_ops.placeholder_with_default(
        constant_op.constant(2, dtype=dtypes.int32), shape=[])
    sliced = slicing._slice_single_param(
        param,
        param_ndims_to_matrix_ndims=2,
        slices=make_slices[:, idx],
        batch_shape=constant_op.constant([7, 6, 5]))
    self.assertAllEqual((7, 5, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_tensor_broadcastdim(self):
    # [FUNCTION DOC]
    # Scenario: like the tensor index test but the indexed batch dimension is size one in the
    # parameter. Expects the dimension removed: shape 7x5x4x3.
    param = array_ops.placeholder_with_default(
        array_ops.zeros([7, 1, 5, 4, 3]), shape=None)
    idx = array_ops.placeholder_with_default(
        constant_op.constant(2, dtype=dtypes.int32), shape=[])
    sliced = slicing._slice_single_param(
        param,
        param_ndims_to_matrix_ndims=2,
        slices=make_slices[:, idx],
        batch_shape=constant_op.constant([7, 6, 5]))
    self.assertAllEqual((7, 5, 4, 3), self.evaluate(sliced).shape)

  def test_single_param_slice_broadcast_batch(self):
    # [FUNCTION DOC]
    # Scenario: the parameter has fewer batch dimensions than the batch shape (4x3 versus 7x4x3)
    # and the slice uses an ellipsis, new axis, start-two slice and another new axis. Expects the
    # result shape to equal the shape of the same slice applied to a one-by-four-by-three zero
    # array, with a trailing size-one event dimension appended.
    sliced = slicing._slice_single_param(
        array_ops.zeros([4, 3, 1]),  # batch = [4, 3], event = [1]
        param_ndims_to_matrix_ndims=1,
        slices=make_slices[..., array_ops.newaxis, 2:, array_ops.newaxis],
        batch_shape=constant_op.constant([7, 4, 3]))
    self.assertAllEqual(
        list(array_ops.zeros([1, 4, 3])[
            ..., array_ops.newaxis, 2:, array_ops.newaxis].shape) + [1],
        self.evaluate(sliced).shape)

  def test_single_param_slice_broadcast_batch_leading_newaxis(self):
    # [FUNCTION DOC]
    # Scenario: same as the broadcast batch test but the slice also starts with a new axis.
    # Expects the shape to equal that of the same slice applied to a one-by-four-by-three zero
    # array with a trailing size-one event dimension appended.
    sliced = slicing._slice_single_param(
        array_ops.zeros([4, 3, 1]),  # batch = [4, 3], event = [1]
        param_ndims_to_matrix_ndims=1,
        slices=make_slices[
            array_ops.newaxis, ..., array_ops.newaxis, 2:, array_ops.newaxis],
        batch_shape=constant_op.constant([7, 4, 3]))
    expected = array_ops.zeros(
        [1, 4, 3])[
            array_ops.newaxis, ..., array_ops.newaxis,
            2:, array_ops.newaxis].shape + [1]
    self.assertAllEqual(expected, self.evaluate(sliced).shape)

  def test_single_param_multi_ellipsis(self):
    # [FUNCTION DOC]
    # Scenario: a slice specification contains two ellipses. Expects a value error whose message
    # says multiple ellipses were found.
    with self.assertRaisesRegex(ValueError, 'Found multiple `...`'):
      slicing._slice_single_param(
          array_ops.zeros([7, 6, 5, 4, 3]),
          param_ndims_to_matrix_ndims=2,
          slices=make_slices[:, ..., 2, ...],
          batch_shape=constant_op.constant([7, 6, 5]))

  def test_single_param_too_many_slices(self):
    # [FUNCTION DOC]
    # Scenario: the slice specification has more entries than the batch has dimensions. Expects
    # an index error, value error or invalid-argument error.
    with self.assertRaises(
        (IndexError, ValueError, errors.InvalidArgumentError)):
      slicing._slice_single_param(
          array_ops.zeros([7, 6, 5, 4, 3]),
          param_ndims_to_matrix_ndims=2,
          slices=make_slices[:, :3, ..., -2:, :],
          batch_shape=constant_op.constant([7, 6, 5]))

  def test_slice_single_param_operator(self):
    # [FUNCTION DOC]
    # Scenario: a full-matrix linear operator with a random 1x4x3x2x2 matrix is indexed using an
    # ellipsis, new axis, start-two slice and new axis. Expects the sliced operator's batch shape
    # tensor to equal the shape produced by the same slice on a one-by-four-by-three zero array.
    matrix = linear_operator_test_util.random_normal(
        shape=[1, 4, 3, 2, 2], dtype=dtypes.float32)
    operator = linalg.LinearOperatorFullMatrix(matrix, is_square=True)
    sliced = operator[..., array_ops.newaxis, 2:, array_ops.newaxis]

    self.assertAllEqual(
        list(array_ops.zeros([1, 4, 3])[
            ..., array_ops.newaxis, 2:, array_ops.newaxis].shape),
        sliced.batch_shape_tensor())

  def test_slice_nested_operator(self):
    # [FUNCTION DOC]
    # Scenario: a Kronecker operator combining a block-diagonal operator (two diagonal operators
    # with different batch shapes) and a full-matrix operator with batch shape 4x1x1x1. Slices
    # it with an integer index, with an integer plus new axis, and with a new axis alone. Expects
    # batch shapes 3x5x2, 3x5x2x1 and 4x3x5x2x1 respectively.
    linop = linalg.LinearOperatorKronecker([
        linalg.LinearOperatorBlockDiag([
            linalg.LinearOperatorDiag(array_ops.ones([1, 2, 2])),
            linalg.LinearOperatorDiag(array_ops.ones([3, 5, 2, 2]))]),
        linalg.LinearOperatorFullMatrix(
            linear_operator_test_util.random_normal(
                shape=[4, 1, 1, 1, 3, 3], dtype=dtypes.float32))])
    self.assertAllEqual(linop[0, ...].batch_shape_tensor(), [3, 5, 2])
    self.assertAllEqual(linop[
        0, ..., array_ops.newaxis].batch_shape_tensor(), [3, 5, 2, 1])
    self.assertAllEqual(linop[
        ..., array_ops.newaxis].batch_shape_tensor(), [4, 3, 5, 2, 1])


if __name__ == '__main__':
  test.main()
