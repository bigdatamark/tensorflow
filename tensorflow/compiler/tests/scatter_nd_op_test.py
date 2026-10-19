# [MODULE DOC]
# Tests for the scatter-into-zeros op and the tensor scatter add, subtract and update ops under
# the XLA test harness. Module-level private helpers implement a numpy reference scatter; four
# test classes cover basic cases, random variable-rank cases, shape errors, out-of-range indices,
# and scalar updates. No import-time side effects beyond imports; as a script starts the runner.
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
"""Tests for tensorflow.ops.tf.scatter_nd."""

import functools

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


def _AsType(v, vtype):
  # [FUNCTION DOC]
  # HELPER: Type-conversion utility for the scatter test data.
  # Converts v to the numpy type vtype: if v is a numpy array its element type is cast, otherwise
  # vtype is called on v as a scalar conversion. Returns the converted value.
  return v.astype(vtype) if isinstance(v, np.ndarray) else vtype(v)


def _FlatInnerDims(tensor, ndims=2):
  # [FUNCTION DOC]
  # HELPER: Numpy reference reshaping utility for the scatter tests.
  # Inputs: tensor is a numpy array; ndims (default two) is the number of dimensions of the
  # result. Collapses all leading dimensions into one, keeping the last ndims minus one
  # dimensions. Returns the reshaped array.
  shape = list(tensor.shape)
  return tensor.reshape(
      [functools.reduce(lambda x, y: x * y, shape[:-ndims + 1], 1)] +
      shape[-ndims + 1:])


def _FlatOuterDims(tensor, ndims=2):
  # [FUNCTION DOC]
  # HELPER: Numpy reference reshaping utility for the scatter tests.
  # Inputs: tensor is a numpy array; ndims (default two) is the number of dimensions of the
  # result. Keeps the first ndims minus one dimensions and collapses all remaining dimensions
  # into one. Returns the reshaped array.
  shape = list(tensor.shape)
  return tensor.reshape(
      shape[:ndims - 1] +
      [functools.reduce(lambda x, y: x * y, shape[ndims - 1:], 1)])


def _NumpyScatterNd(ref, indices, updates, op):
  # [FUNCTION DOC]
  # HELPER: Numpy reference scatter used to produce expected values in the scatter tests.
  # Inputs: ref is the numpy array scattered into; indices is an integer array whose last
  # dimension is the index depth; updates holds one slice per index tuple; op is a binary
  # callable combining the existing slice and the update slice. The reference is flattened to
  # a view, each index tuple selects a slice and it is replaced by the result of op; repeated
  # indices are applied in order. Note this mutates ref in place through the flattened view.
  # Returns the result reshaped to the shape of ref.
  ixdim = indices.shape[-1]
  num_updates = indices.size // ixdim
  total_nd = len(ref.shape)
  slice_size = 1
  for i in range(ixdim, total_nd):
    slice_size *= ref.shape[i]
  flat_indices = _FlatInnerDims(indices)
  flat_updates = updates.reshape((num_updates, slice_size))
  output_flat = _FlatOuterDims(ref, ixdim + 1)
  for ix_updates, ix_output in enumerate(flat_indices):
    ix_output = tuple(ix_output)
    output_flat[ix_output] = op(output_flat[ix_output],
                                flat_updates[ix_updates])
  return output_flat.reshape(ref.shape)


def _NumpyUpdate(indices, updates, shape):
  # [FUNCTION DOC]
  # HELPER: Numpy reference for scatter into zeros, used as expected-value generator.
  # Inputs: indices integer array, updates numpy array, shape the output shape. Creates a zero
  # array of that shape with the element type of updates and overwrites the indexed slices with
  # the updates (the last write wins for repeated indices). Returns the resulting array.
  ref = np.zeros(shape, dtype=updates.dtype)
  return _NumpyScatterNd(ref, indices, updates, lambda p, u: u)


class ScatterNdTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the scatter-into-zeros op. Covers small hand-written cases, randomized
  # comparisons with a numpy reference over several ranks and index depths, shape validation
  # errors, and out-of-range indices (which must not fail). Holds no state of its own.

  def _VariableRankTest(self,
                        np_scatter,
                        tf_scatter,
                        vtype,
                        itype,
                        repeat_indices=False):
    # [FUNCTION DOC]
    # HELPER: Randomized comparison driver for the scatter tests.
    # Inputs: np_scatter and tf_scatter are callables taking indices, updates and an output shape
    # and returning the expected and device results; vtype and itype are the numpy value and
    # index types; repeat_indices (default false) makes some index tuples duplicates of others.
    # Seeds the random generator with eight (global randomness side effect). For several
    # (reference shape, indices shape) pairs ranging from rank two to rank three with index depth
    # one to three, picks distinct random index tuples in the indexable area (optionally
    # repeating some), builds random normal updates of matching slice shape, runs both scatters
    # and asserts the results are close. Returns nothing.
    np.random.seed(8)
    ref_shapes = [(3, 6), (3, 6), (3, 6, 9), (3, 6, 9), (3, 6, 9), (3, 6, 9)]
    indices_shapes = [(2,), (2, 2), (2,), (2, 2), (2, 3), (2, 3, 3)]
    for ref_shape, indices_shape in zip(ref_shapes, indices_shapes):
      num_updates = indices_shape[0]
      ixdim = indices_shape[-1]

      indexable_area_shape = ()
      for i in range(ixdim):
        indexable_area_shape += (ref_shape[i],)
      all_indices = [
          list(coord)
          for coord, _ in np.ndenumerate(np.empty(indexable_area_shape, vtype))
      ]
      np.random.shuffle(all_indices)
      indices = np.array(all_indices[:num_updates])

      if num_updates > 1 and repeat_indices:
        indices = indices[:num_updates // 2]
        for _ in range(num_updates - num_updates // 2):
          indices = np.append(
              indices, [indices[np.random.randint(num_updates // 2)]], axis=0)
        np.random.shuffle(indices)
      indices = _AsType(indices[:num_updates], itype)

      updates_shape = (num_updates,)
      for i in range(ixdim, len(ref_shape)):
        updates_shape += (ref_shape[i],)
      updates = _AsType(np.random.randn(*(updates_shape)), vtype)

      # Scatter via numpy
      np_out = np_scatter(indices, updates, ref_shape)
      # Scatter via tensorflow
      tf_out = tf_scatter(indices, updates, ref_shape)

      self.assertAllClose(np_out, tf_out)

  def _VariableRankTests(self, np_scatter, tf_scatter):
    # [FUNCTION DOC]
    # HELPER: Runs the randomized comparison for every numeric value type combined with each of
    # 32-bit and 64-bit index types supported by the device. Inputs are the numpy and device
    # scatter callables. Returns nothing.
    for vtype in self.numeric_types:
      for itype in set([np.int32, np.int64]).intersection(set(self.int_types)):
        self._VariableRankTest(np_scatter, tf_scatter, vtype, itype)

  def _runScatterNd(self, indices, updates, shape):
    # [FUNCTION DOC]
    # HELPER: Runs the scatter-into-zeros op on the test device.
    # Inputs: indices and updates are numpy arrays; shape is the output shape as a list.
    # Creates placeholders for both, builds the op in the device scope, and evaluates it in a
    # session with the arrays fed. Returns the resulting numpy array.
    with self.session():
      updates_placeholder = array_ops.placeholder(updates.dtype)
      indices_placeholder = array_ops.placeholder(indices.dtype)
      with self.test_scope():
        output = array_ops.scatter_nd(indices_placeholder, updates_placeholder,
                                      shape)
      feed_dict = {updates_placeholder: updates, indices_placeholder: indices}
      return output.eval(feed_dict=feed_dict)

  def testSimple(self):
    # [FUNCTION DOC]
    # Scatters four values into a length-eight zero vector at indices four, three, one and seven
    # and expects each value at its position with zeros elsewhere. Returns nothing.
    indices = np.array([[4], [3], [1], [7]], dtype=np.int32)
    updates = np.array([9, 10, 11, 12], dtype=np.float32)
    expected = np.array([0, 11, 0, 10, 9, 0, 0, 12], dtype=np.int32)
    self.assertAllEqual(expected, self._runScatterNd(indices, updates, [8]))

  def testRepeatedIndices(self):
    # [FUNCTION DOC]
    # Scatters four values to indices zero, one, zero, one of a length-two vector and expects
    # the values at repeated indices to be summed, giving twenty and twenty-two. Returns nothing.
    indices = np.array([[0], [1], [0], [1]], dtype=np.int32)
    updates = np.array([9, 10, 11, 12], dtype=np.float32)
    expected = np.array([20, 22], dtype=np.int32)
    self.assertAllEqual(expected, self._runScatterNd(indices, updates, [2]))

  def testSimple2(self):
    # [FUNCTION DOC]
    # Scatters two scalars using two-component indices into a three-by-two zero matrix and
    # expects them at the second row's two positions. Returns nothing.
    indices = np.array([[1, 0], [1, 1]], dtype=np.int32)
    updates = np.array([11., 12.], dtype=np.float32)
    expected = np.array([[0., 0.], [11., 12.], [0., 0.]], dtype=np.float32)
    self.assertAllEqual(expected, self._runScatterNd(indices, updates, [3, 2]))

  def testSimple3(self):
    # [FUNCTION DOC]
    # Scatters one whole row of two values to index one of a three-by-two zero matrix and
    # expects that row replaced, other rows zero. Returns nothing.
    indices = np.array([[1]], dtype=np.int32)
    updates = np.array([[11., 12.]], dtype=np.float32)
    expected = np.array([[0., 0.], [11., 12.], [0., 0.]])
    self.assertAllEqual(expected, self._runScatterNd(indices, updates, [3, 2]))

  def testVariableRankUpdate(self):
    # [FUNCTION DOC]
    # Runs the randomized multi-rank comparison of the device scatter against the numpy
    # scatter-into-zeros reference. Returns nothing.
    self._VariableRankTests(_NumpyUpdate, self._runScatterNd)

  def testExtraIndicesDimensions(self):
    # [FUNCTION DOC]
    # Uses an index tensor with an extra leading batch dimension (shape one, one, two) and zero
    # updates of shape one by one into a two-by-two output; expects an all-zero result.
    # Returns nothing.
    indices = np.zeros([1, 1, 2], np.int32)
    updates = np.zeros([1, 1], np.int32)
    expected = np.zeros([2, 2], dtype=np.int32)
    self.assertAllEqual(expected, self._runScatterNd(indices, updates, [2, 2]))

  @test_util.disable_mlir_bridge("Error messages differ")
  def testRank3InvalidShape1(self):
    # [FUNCTION DOC]
    # Disabled for the MLIR bridge because error messages differ. Uses indices of shape three,
    # two, two with updates of shape two, two, two (leading dimensions mismatch) targeting a
    # two-cubed output; expects an invalid-argument error about the required updates shape.
    # Returns nothing.
    indices = np.zeros([3, 2, 2], np.int32)
    updates = np.zeros([2, 2, 2], np.int32)
    with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError,
                                             "Must have updates.shape"):
      self._runScatterNd(indices, updates, [2, 2, 2])

  @test_util.disable_mlir_bridge("Error messages differ")
  def testRank3InvalidShape2(self):
    # [FUNCTION DOC]
    # Disabled for the MLIR bridge because error messages differ. Uses indices of shape two,
    # two, one with updates of shape two, two (missing trailing slice dimension) targeting a
    # two-cubed output; expects an invalid-argument error about the required updates shape.
    # Returns nothing.
    indices = np.zeros([2, 2, 1], np.int32)
    updates = np.zeros([2, 2], np.int32)
    with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError,
                                             "Must have updates.shape"):
      self._runScatterNd(indices, updates, [2, 2, 2])

  def testScatterOutOfRange(self):
    # [FUNCTION DOC]
    # Scatters three values into a length-six vector. Runs once with all indices in range, then
    # with a negative index, then with an index equal to the size. The out-of-range runs only
    # need to complete without raising; their output is implementation defined and not checked.
    # Returns nothing.
    updates = np.array([-3, -4, -5]).astype(np.float32)

    # Indices all in range, no problem.
    indices = np.array([[2], [0], [5]], dtype=np.int32)
    self._runScatterNd(indices, updates, [6])

    # Indices out of range should not fail. It produces implementation-defined
    # output.
    indices = np.array([[-1], [0], [5]], dtype=np.int32)
    self._runScatterNd(indices, updates, [6])
    indices = np.array([[2], [0], [6]], dtype=np.int32)
    self._runScatterNd(indices, updates, [6])


class ScatterNdTensorTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the tensor scatter add, subtract and update ops, each applied to a
  # length-eight vector of ones with four vector updates. Holds no state of its own.

  def _runScatter(self, op):
    # [FUNCTION DOC]
    # HELPER: Runs a tensor scatter op for the tests of this class.
    # Input: op is a callable taking a base tensor, indices and updates (a tensor scatter add,
    # subtract or update). Builds placeholders for indices four, three, one, seven and updates
    # nine through twelve, a base vector of eight ones, applies op in the device scope within a
    # session, and returns the resulting numpy array.
    indices_np = np.array([[4], [3], [1], [7]], dtype=np.int32)
    updates_np = np.array([9, 10, 11, 12], dtype=np.float32)
    with self.session() as sess, self.test_scope():
      indices = array_ops.placeholder(indices_np.dtype, shape=indices_np.shape)
      updates = array_ops.placeholder(updates_np.dtype, shape=updates_np.shape)
      t = array_ops.ones([8], dtype=np.float32)

      out = op(t, indices, updates)
      return sess.run(out, feed_dict={indices: indices_np, updates: updates_np})

  def testAdd(self):
    # [FUNCTION DOC]
    # Applies the tensor scatter add and expects each update added to the base ones at its
    # index. Returns nothing.
    self.assertAllEqual(
        self._runScatter(array_ops.tensor_scatter_add),
        np.array([1, 12, 1, 11, 10, 1, 1, 13], dtype=np.float32))

  def testSub(self):
    # [FUNCTION DOC]
    # Applies the tensor scatter subtract and expects each update subtracted from the base ones
    # at its index. Returns nothing.
    self.assertAllEqual(
        self._runScatter(array_ops.tensor_scatter_sub),
        np.array([1, -10, 1, -9, -8, 1, 1, -11], dtype=np.float32))

  def testUpdate(self):
    # [FUNCTION DOC]
    # Applies the tensor scatter update and expects the base ones replaced by the updates at
    # their indices. Returns nothing.
    self.assertAllEqual(
        self._runScatter(array_ops.tensor_scatter_update),
        np.array([1, 11, 1, 10, 9, 1, 1, 12], dtype=np.float32))


class ScatterNdTensorScalarUpdateTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for tensor scatter update and add when the updates are a single scalar
  # broadcast to all four indexed positions of a length-eight vector of ones. No own state.

  def _runScatter(self, op):
    # [FUNCTION DOC]
    # HELPER: Runs a tensor scatter op for the tests of this class.
    # Input: op is a callable taking a base tensor, indices and updates. Builds placeholders for
    # indices four, three, one, seven and a scalar update of nine, a base vector of eight ones,
    # applies op in the device scope within a session, and returns the resulting numpy array.
    indices_np = np.array([[4], [3], [1], [7]], dtype=np.int32)
    updates_np = np.array(9, dtype=np.float32)
    with self.session() as sess, self.test_scope():
      indices = array_ops.placeholder(indices_np.dtype, shape=indices_np.shape)
      updates = array_ops.placeholder(updates_np.dtype, shape=updates_np.shape)
      t = array_ops.ones([8], dtype=np.float32)

      out = op(t, indices, updates)
      return sess.run(out, feed_dict={indices: indices_np, updates: updates_np})

  def testUpdate(self):
    # [FUNCTION DOC]
    # Applies the tensor scatter update with a scalar update and expects nine at each indexed
    # position and ones elsewhere. Returns nothing.
    self.assertAllEqual(
        self._runScatter(array_ops.tensor_scatter_update),
        np.array([1, 9, 1, 9, 9, 1, 1, 9], dtype=np.float32))

  def testAdd(self):
    # [FUNCTION DOC]
    # Applies the tensor scatter add with a scalar update and expects nine added at each indexed
    # position, giving ten there and ones elsewhere. Returns nothing.
    self.assertAllEqual(
        self._runScatter(array_ops.tensor_scatter_add),
        np.array([1, 10, 1, 10, 10, 1, 1, 10], dtype=np.float32))

if __name__ == "__main__":
  test.main()
