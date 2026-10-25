# [MODULE DOC]
# Test module for the stateful scatter-nd minimum and maximum ops on variables, V1 only. Contains
# numpy reference implementations of an N-dimensional scatter (with helpers for casting and
# flattening dimensions) and a test class that compares TensorFlow results against them and checks
# out-of-range index errors. No import-time side effects.
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
"""Tests for scatter_nd_ops that only work in V1."""

import functools

import numpy as np

from tensorflow.python.framework import test_util
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import test


def _AsType(v, vtype):
  # [FUNCTION DOC]
  # HELPER: Test utility for building typed numpy inputs for the scatter tests.
  # Converts v to the numpy type vtype: if v is a numpy array it is cast with the array cast
  # method, otherwise vtype is called on it as a scalar constructor. Returns the converted value.
  return v.astype(vtype) if isinstance(v, np.ndarray) else vtype(v)


def _FlatInnerDims(tensor, ndims=2):
  # [FUNCTION DOC]
  # HELPER: Numpy reference helper for the scatter-nd tests.
  # Reshapes a numpy array so that all leading dimensions except the last (ndims minus 1) are
  # collapsed into one by multiplying their sizes, keeping the trailing dimensions. With the
  # default ndims of two the result is a matrix. Returns the reshaped array; input is not mutated.
  shape = list(tensor.shape)
  return tensor.reshape(
      [functools.reduce(lambda x, y: x * y, shape[:-ndims + 1], 1)] +
      shape[-ndims + 1:])


def _FlatOuterDims(tensor, ndims=2):
  # [FUNCTION DOC]
  # HELPER: Numpy reference helper for the scatter-nd tests.
  # Reshapes a numpy array keeping the first (ndims minus 1) dimensions and collapsing all the
  # remaining trailing dimensions into one by multiplying their sizes. Returns the reshaped array;
  # the input is not mutated.
  shape = list(tensor.shape)
  return tensor.reshape(
      shape[:ndims - 1] +
      [functools.reduce(lambda x, y: x * y, shape[ndims - 1:], 1)])


def _NumpyScatterNd(ref, indices, updates, op):
  # [FUNCTION DOC]
  # HELPER: Numpy reference implementation used to compute expected scatter-nd results.
  # Inputs: ref, the array being updated; indices, an integer array whose last dimension is the
  # index depth; updates, the update values; op, a binary function combining the existing slice
  # with an update (for example element-wise minimum). Flattens the problem to a list of index
  # tuples, applies op for each index in order (so repeated indices accumulate), and returns the
  # result reshaped to the shape of ref. Works on a flattened view of ref, so ref itself may be
  # modified in place as a side effect when the view shares memory; also returns the array.
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


def _NumpyMin(ref, indices, updates):
  # [FUNCTION DOC]
  # HELPER: Numpy reference for scatter-nd minimum. Delegates to the generic numpy scatter
  # helper using element-wise minimum and returns its result.
  return _NumpyScatterNd(ref, indices, updates, np.minimum)


def _NumpyMax(ref, indices, updates):
  # [FUNCTION DOC]
  # HELPER: Numpy reference for scatter-nd maximum. Delegates to the generic numpy scatter
  # helper using element-wise maximum and returns its result.
  return _NumpyScatterNd(ref, indices, updates, np.maximum)


class StatefulScatterNdTest(test.TestCase):
  # [CLASS DOC]
  # Test case for stateful scatter-nd min/max ops on V1 variables. Provides helper methods that
  # generate random refs, indices and updates and compare against numpy references.

  def _VariableRankTest(self,
                        np_scatter,
                        tf_scatter,
                        vtype,
                        itype,
                        repeat_indices=False):
    # [FUNCTION DOC]
    # HELPER: Shared driver for the scatter-nd tests in this class.
    # Inputs: np_scatter, numpy reference function; tf_scatter, TensorFlow scatter op under test;
    # vtype, numpy value type; itype, numpy index type; repeat_indices, optional flag (default
    # false) that makes half of the indices duplicates. Seeds numpy randomness with a fixed value,
    # then for six paired ref/index shapes (ranks two and three) builds random unique indices,
    # random updates and a random ref on the GPU-enabled device, applies the numpy reference to a
    # copy, runs the TensorFlow op on an initialized variable, and asserts both results are close.
    # Returns nothing; side effects are reseeding numpy's global random state.
    np.random.seed(8)
    ref_shapes = [(3, 6), (3, 6), (3, 6, 9), (3, 6, 9), (3, 6, 9), (3, 6, 9)]
    indices_shapes = [(2,), (2, 2), (2,), (2, 2), (2, 3), (2, 3, 3)]
    with test_util.device(use_gpu=True):
      for ref_shape, indices_shape in zip(ref_shapes, indices_shapes):
        num_updates = indices_shape[0]
        ixdim = indices_shape[-1]

        indexable_area_shape = ()
        for i in range(ixdim):
          indexable_area_shape += (ref_shape[i],)
        all_indices = [
            list(coord) for coord, _ in np.ndenumerate(
                np.empty(indexable_area_shape, vtype))
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
        ref = _AsType(np.random.randn(*(ref_shape)), vtype)

        # Scatter via numpy
        new = ref.copy()
        np_scatter(new, indices, updates)
        # Scatter via tensorflow
        ref_var = variable_v1.VariableV1(ref)
        self.evaluate(ref_var.initializer)
        self.evaluate(tf_scatter(ref_var, indices, updates))

        # Compare
        self.assertAllClose(new, self.evaluate(ref_var))

  def _ScatterRepeatIndicesTest(self, np_scatter, tf_scatter):
    # [FUNCTION DOC]
    # HELPER: Runs the shared variable-rank driver with repeated indices for every combination of
    # value types (int32, float16, float32, float64) and index types (int32, int64). Takes the
    # numpy reference and the TensorFlow scatter op; returns nothing.
    for vtype in (np.int32, np.float16, np.float32, np.float64):
      for itype in (np.int32, np.int64):
        self._VariableRankTest(
            np_scatter, tf_scatter, vtype, itype, repeat_indices=True)

  @test_util.run_v1_only("Don't need to test VariableV1 in TF2")
  def testScatterRepeatIndicesMinMax(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring says it tests scatter_add, but the body tests scatter-nd minimum
    # and maximum.
    # Scenario: run the repeated-indices comparison for scatter-nd minimum against the numpy
    # minimum reference and for scatter-nd maximum against the numpy maximum reference.
    """This tests scatter_add using indices that repeat."""
    self._ScatterRepeatIndicesTest(_NumpyMin, state_ops.scatter_nd_min)
    self._ScatterRepeatIndicesTest(_NumpyMax, state_ops.scatter_nd_max)

  @test_util.run_v1_only("Don't need to test VariableV1 in TF2")
  def testScatterOutOfRangeCpu(self):
    # [FUNCTION DOC]
    # Scenario: for both scatter-nd minimum and maximum on CPU, a six-element float variable is
    # updated with three updates. In-range indices succeed; then indices containing minus one and
    # indices containing six (both outside the variable's size) are evaluated and each is expected
    # to raise an op error stating the index does not index into shape six.
    for op in (state_ops.scatter_nd_min, state_ops.scatter_nd_max):
      params = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32)
      updates = np.array([-3, -4, -5]).astype(np.float32)
      with self.cached_session(use_gpu=False):
        ref = variable_v1.VariableV1(params)
        self.evaluate(ref.initializer)

        # Indices all in range, no problem.
        indices = np.array([[2], [0], [5]])
        self.evaluate(op(ref, indices, updates))

        # Test some out of range errors.
        indices = np.array([[-1], [0], [5]])
        with self.assertRaisesOpError(
            r"indices\[0\] = \[-1\] does not index into shape \[6\]"):
          op(ref, indices, updates).eval()

        indices = np.array([[2], [0], [6]])
        with self.assertRaisesOpError(
            r"indices\[2\] = \[6\] does not index into shape \[6\]"):
          op(ref, indices, updates).eval()


if __name__ == "__main__":
  test.main()
