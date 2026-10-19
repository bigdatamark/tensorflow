# [MODULE DOC]
# Test module for the one-dimensional-index scatter ops on variables: scatter_update, add, sub,
# mul, div, min and max. It defines numpy reference implementations (array and scalar update
# forms), two lookup tables mapping each TensorFlow op to its numpy reference, and one test class
# that compares variable results against numpy for many value/index types, index shapes, repeated
# indices and scalar updates, plus out-of-range index errors and a determinism error check.
# Import-time side effects: only builds the two lookup tables; run as a test program.
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
"""Tests for tensorflow.ops.tf.scatter."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


def _AsType(v, vtype):
  # [FUNCTION DOC]
  # HELPER: converts test data to a requested numeric type. If v is a numpy array returns a copy
  # cast to vtype, otherwise calls vtype on v (scalar conversion) and returns the result.
  return v.astype(vtype) if isinstance(v, np.ndarray) else vtype(v)


def _NumpyAdd(ref, indices, updates):
  # [FUNCTION DOC]
  # HELPER: numpy reference for scatter-add with one update per index. Loops over every position
  # of the indices array and adds the matching update element to the referenced row of ref, so
  # repeated indices accumulate (plain numpy advanced assignment would not). Modifies ref in
  # place and returns nothing.
  # Since numpy advanced assignment does not support repeated indices,
  # we run a simple loop to perform scatter_add.
  for i, indx in np.ndenumerate(indices):
    ref[indx] += updates[i]


def _NumpyAddScalar(ref, indices, update):
  # [FUNCTION DOC]
  # HELPER: numpy reference for scatter-add with one scalar update applied at every index.
  # Adds the scalar to each referenced row of ref in place (repeats accumulate); returns nothing.
  for _, indx in np.ndenumerate(indices):
    ref[indx] += update


def _NumpySub(ref, indices, updates):
  # [FUNCTION DOC]
  # HELPER: numpy reference for scatter-subtract with one update per index. Subtracts the
  # matching update element from the referenced row of ref in place (repeats accumulate);
  # returns nothing.
  for i, indx in np.ndenumerate(indices):
    ref[indx] -= updates[i]


def _NumpySubScalar(ref, indices, update):
  # [FUNCTION DOC]
  # HELPER: numpy reference for scatter-subtract with one scalar update applied at every index.
  # Subtracts the scalar from each referenced row of ref in place; returns nothing.
  for _, indx in np.ndenumerate(indices):
    ref[indx] -= update


def _NumpyMul(ref, indices, updates):
  # [FUNCTION DOC]
  # HELPER: numpy reference for scatter-multiply with one update per index. Multiplies the
  # referenced row of ref by the matching update element in place; returns nothing.
  for i, indx in np.ndenumerate(indices):
    ref[indx] *= updates[i]


def _NumpyMulScalar(ref, indices, update):
  # [FUNCTION DOC]
  # HELPER: numpy reference for scatter-multiply with one scalar update applied at every index.
  # Multiplies each referenced row of ref by the scalar in place; returns nothing.
  for _, indx in np.ndenumerate(indices):
    ref[indx] *= update


def _NumpyDiv(ref, indices, updates):
  # [FUNCTION DOC]
  # HELPER: numpy reference for scatter-divide with one update per index. Divides the referenced
  # row of ref by the matching update element in place; returns nothing. Callers must keep
  # updates away from zero.
  for i, indx in np.ndenumerate(indices):
    ref[indx] /= updates[i]


def _NumpyDivScalar(ref, indices, update):
  # [FUNCTION DOC]
  # HELPER: numpy reference for scatter-divide with one scalar update applied at every index.
  # Divides each referenced row of ref by the scalar in place; returns nothing.
  for _, indx in np.ndenumerate(indices):
    ref[indx] /= update


def _NumpyMin(ref, indices, updates):
  # [FUNCTION DOC]
  # HELPER: numpy reference for scatter-minimum with one update per index. Replaces each
  # referenced row of ref by the element-wise minimum with the matching update, in place;
  # returns nothing.
  for i, indx in np.ndenumerate(indices):
    ref[indx] = np.minimum(ref[indx], updates[i])


def _NumpyMinScalar(ref, indices, update):
  # [FUNCTION DOC]
  # HELPER: numpy reference for scatter-minimum with a scalar update applied at every index.
  # Replaces each referenced row of ref by its element-wise minimum with the scalar, in place;
  # returns nothing.
  for _, indx in np.ndenumerate(indices):
    ref[indx] = np.minimum(ref[indx], update)


def _NumpyMax(ref, indices, updates):
  # [FUNCTION DOC]
  # HELPER: numpy reference for scatter-maximum with one update per index. Replaces each
  # referenced row of ref by the element-wise maximum with the matching update, in place;
  # returns nothing.
  for i, indx in np.ndenumerate(indices):
    ref[indx] = np.maximum(ref[indx], updates[i])


def _NumpyMaxScalar(ref, indices, update):
  # [FUNCTION DOC]
  # HELPER: numpy reference for scatter-maximum with a scalar update applied at every index.
  # Replaces each referenced row of ref by its element-wise maximum with the scalar, in place;
  # returns nothing.
  for _, indx in np.ndenumerate(indices):
    ref[indx] = np.maximum(ref[indx], update)


def _NumpyUpdate(ref, indices, updates):
  # [FUNCTION DOC]
  # HELPER: numpy reference for scatter-update with one update per index. Overwrites each
  # referenced row of ref with the matching update element in place (with repeated indices the
  # last write wins); returns nothing.
  for i, indx in np.ndenumerate(indices):
    ref[indx] = updates[i]


def _NumpyUpdateScalar(ref, indices, update):
  # [FUNCTION DOC]
  # HELPER: numpy reference for scatter-update with one scalar update applied at every index.
  # Overwrites each referenced row of ref with the scalar in place; returns nothing.
  for _, indx in np.ndenumerate(indices):
    ref[indx] = update


_TF_OPS_TO_NUMPY = {
    state_ops.scatter_update: _NumpyUpdate,
    state_ops.scatter_add: _NumpyAdd,
    state_ops.scatter_sub: _NumpySub,
    state_ops.scatter_mul: _NumpyMul,
    state_ops.scatter_div: _NumpyDiv,
    state_ops.scatter_min: _NumpyMin,
    state_ops.scatter_max: _NumpyMax,
}

_TF_OPS_TO_NUMPY_SCALAR = {
    state_ops.scatter_update: _NumpyUpdateScalar,
    state_ops.scatter_add: _NumpyAddScalar,
    state_ops.scatter_sub: _NumpySubScalar,
    state_ops.scatter_mul: _NumpyMulScalar,
    state_ops.scatter_div: _NumpyDivScalar,
    state_ops.scatter_min: _NumpyMinScalar,
    state_ops.scatter_max: _NumpyMaxScalar,
}


class ScatterTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the variable scatter ops (update, add, sub, mul, div, min, max) indexed by a
  # one-dimensional index space. Most tests are thin wrappers that run a shared comparison driver
  # for one op with options for repeated indices and scalar updates; others cover boolean
  # update, out-of-range errors on CPU and the GPU determinism error. No shared fixtures.

  def _VariableRankTest(self,
                        tf_scatter,
                        vtype,
                        itype,
                        repeat_indices=False,
                        updates_are_scalar=False):
    # [FUNCTION DOC]
    # HELPER: shared driver comparing a TensorFlow scatter op against its numpy reference.
    # Inputs: tf_scatter (op from the lookup tables), vtype (numpy value type), itype (numpy index
    # type), repeat_indices (default false; when true half of the indices are repeats),
    # updates_are_scalar (default false; when true a single random scalar is the update).
    # For index shapes scalar, 2, 3x7, 3x4x7 and extra update shapes none, 5, 5x9 it builds
    # shuffled indices, random updates (clipped away from zero so division is safe; for int32 the
    # clipping magnitude is one with random signs), applies the numpy reference to a copy and the
    # op to a variable, then asserts closeness with looser tolerances for half and bfloat16.
    # Seeds numpy randomness with a fixed value (global state change). Returns nothing.
    np.random.seed(8)
    with self.cached_session():
      for indices_shape in (), (2,), (3, 7), (3, 4, 7):
        for extra_shape in (), (5,), (5, 9):
          # Generate random indices with no duplicates for easy numpy comparison
          size = np.prod(indices_shape, dtype=itype)
          first_dim = 3 * size
          indices = np.arange(first_dim)
          np.random.shuffle(indices)
          indices = indices[:size]
          if size > 1 and repeat_indices:
            # Add some random repeats.
            indices = indices[:size // 2]
            for _ in range(size - size // 2):
              # Randomly append some repeats.
              indices = np.append(indices,
                                  indices[np.random.randint(size // 2)])
            np.random.shuffle(indices)
          indices = indices.reshape(indices_shape)
          if updates_are_scalar:
            updates = _AsType(np.random.randn(), vtype)
          else:
            updates = _AsType(
                np.random.randn(*(indices_shape + extra_shape)), vtype)

          # Clips small values to avoid division by zero.
          threshold = np.array(1e-4, dtype=vtype)
          sign = np.sign(updates)
          if vtype == np.int32:
            threshold = 1
            sign = np.random.choice([-1, 1], updates.shape)
          updates = np.where(
              np.abs(updates) < threshold, threshold * sign, updates)

          old = _AsType(np.random.randn(*((first_dim,) + extra_shape)), vtype)

          # Scatter via numpy
          new = old.copy()
          if updates_are_scalar:
            np_scatter = _TF_OPS_TO_NUMPY_SCALAR[tf_scatter]
          else:
            np_scatter = _TF_OPS_TO_NUMPY[tf_scatter]
          np_scatter(new, indices, updates)
          # Scatter via tensorflow
          ref = variables.Variable(old)
          self.evaluate(ref.initializer)
          self.evaluate(tf_scatter(ref, indices, updates))
          self.assertAllCloseAccordingToType(
              self.evaluate(ref),
              new,
              half_rtol=5e-3,
              half_atol=5e-3,
              bfloat16_rtol=5e-2,
              bfloat16_atol=5e-2)

  def _VariableRankTests(self,
                         tf_scatter,
                         repeat_indices=False,
                         updates_are_scalar=False):
    # [FUNCTION DOC]
    # HELPER: runs the comparison driver for one scatter op across value and index types.
    # Value types are float32, float64 and bfloat16, plus int32 and float16 except for divide
    # (integer division is excluded and float16 is numerically unstable for divide); index types
    # are int32 and int64. Passes through the repeat_indices and updates_are_scalar flags
    # (both default false). Returns nothing; fails on any mismatch.
    vtypes = [np.float32, np.float64, dtypes.bfloat16.as_numpy_dtype]
    if tf_scatter != state_ops.scatter_div:
      vtypes.append(np.int32)
      # float16 is numerically unstable for div
      vtypes.append(np.float16)

    for vtype in vtypes:
      for itype in (np.int32, np.int64):
        self._VariableRankTest(tf_scatter, vtype, itype, repeat_indices,
                               updates_are_scalar)

  def testVariableRankUpdate(self):
    # [FUNCTION DOC]
    # Runs the all-types comparison for scatter_update with unique indices and array updates.
    self._VariableRankTests(state_ops.scatter_update, False)

  def testVariableRankAdd(self):
    # [FUNCTION DOC]
    # Runs the all-types comparison for scatter_add with unique indices and array updates.
    self._VariableRankTests(state_ops.scatter_add, False)

  def testVariableRankSub(self):
    # [FUNCTION DOC]
    # Runs the all-types comparison for scatter_sub with unique indices and array updates.
    self._VariableRankTests(state_ops.scatter_sub, False)

  def testVariableRankMul(self):
    # [FUNCTION DOC]
    # Runs the all-types comparison for scatter_mul with unique indices and array updates.
    self._VariableRankTests(state_ops.scatter_mul, False)

  def testVariableRankDiv(self):
    # [FUNCTION DOC]
    # Runs the comparison for scatter_div with unique indices and array updates (integer and
    # float16 types are skipped for divide).
    self._VariableRankTests(state_ops.scatter_div, False)

  def testVariableRankMin(self):
    # [FUNCTION DOC]
    # Runs the all-types comparison for scatter_min with unique indices and array updates.
    self._VariableRankTests(state_ops.scatter_min, False)

  def testVariableRankMax(self):
    # [FUNCTION DOC]
    # Runs the all-types comparison for scatter_max with unique indices and array updates.
    self._VariableRankTests(state_ops.scatter_max, False)

  def testRepeatIndicesAdd(self):
    # [FUNCTION DOC]
    # Runs the all-types comparison for scatter_add with repeated indices (duplicates must
    # accumulate) and array updates.
    self._VariableRankTests(state_ops.scatter_add, True)

  def testRepeatIndicesSub(self):
    # [FUNCTION DOC]
    # Runs the all-types comparison for scatter_sub with repeated indices (duplicates must
    # accumulate) and array updates.
    self._VariableRankTests(state_ops.scatter_sub, True)

  def testRepeatIndicesMul(self):
    # [FUNCTION DOC]
    # Runs the all-types comparison for scatter_mul with repeated indices and array updates.
    self._VariableRankTests(state_ops.scatter_mul, True)

  def testRepeatIndicesDiv(self):
    # [FUNCTION DOC]
    # Runs the comparison for scatter_div with repeated indices and array updates (integer and
    # float16 types are skipped for divide).
    self._VariableRankTests(state_ops.scatter_div, True)

  def testRepeatIndicesMin(self):
    # [FUNCTION DOC]
    # Runs the all-types comparison for scatter_min with repeated indices and array updates.
    self._VariableRankTests(state_ops.scatter_min, True)

  def testRepeatIndicesMax(self):
    # [FUNCTION DOC]
    # Runs the all-types comparison for scatter_max with repeated indices and array updates.
    self._VariableRankTests(state_ops.scatter_max, True)

  def testVariableRankUpdateScalar(self):
    # [FUNCTION DOC]
    # Runs the all-types comparison for scatter_update with unique indices and a scalar update.
    self._VariableRankTests(state_ops.scatter_update, False, True)

  def testVariableRankAddScalar(self):
    # [FUNCTION DOC]
    # Runs the all-types comparison for scatter_add with unique indices and a scalar update.
    self._VariableRankTests(state_ops.scatter_add, False, True)

  def testVariableRankSubScalar(self):
    # [FUNCTION DOC]
    # Runs the all-types comparison for scatter_sub with unique indices and a scalar update.
    self._VariableRankTests(state_ops.scatter_sub, False, True)

  def testVariableRankMulScalar(self):
    # [FUNCTION DOC]
    # Runs the all-types comparison for scatter_mul with unique indices and a scalar update.
    self._VariableRankTests(state_ops.scatter_mul, False, True)

  def testVariableRankDivScalar(self):
    # [FUNCTION DOC]
    # Runs the comparison for scatter_div with unique indices and a scalar update (integer and
    # float16 types are skipped for divide).
    self._VariableRankTests(state_ops.scatter_div, False, True)

  def testVariableRankMinScalar(self):
    # [FUNCTION DOC]
    # Runs the all-types comparison for scatter_min with unique indices and a scalar update.
    self._VariableRankTests(state_ops.scatter_min, False, True)

  def testVariableRankMaxScalar(self):
    # [FUNCTION DOC]
    # Runs the all-types comparison for scatter_max with unique indices and a scalar update.
    self._VariableRankTests(state_ops.scatter_max, False, True)

  def testRepeatIndicesAddScalar(self):
    # [FUNCTION DOC]
    # Runs the all-types comparison for scatter_add with repeated indices and a scalar update.
    self._VariableRankTests(state_ops.scatter_add, True, True)

  def testRepeatIndicesSubScalar(self):
    # [FUNCTION DOC]
    # Runs the all-types comparison for scatter_sub with repeated indices and a scalar update.
    self._VariableRankTests(state_ops.scatter_sub, True, True)

  def testRepeatIndicesMulScalar(self):
    # [FUNCTION DOC]
    # Runs the all-types comparison for scatter_mul with repeated indices and a scalar update.
    self._VariableRankTests(state_ops.scatter_mul, True, True)

  def testRepeatIndicesDivScalar(self):
    # [FUNCTION DOC]
    # Runs the comparison for scatter_div with repeated indices and a scalar update (integer and
    # float16 types are skipped for divide).
    self._VariableRankTests(state_ops.scatter_div, True, True)

  def testRepeatIndicesMinScalar(self):
    # [FUNCTION DOC]
    # Runs the all-types comparison for scatter_min with repeated indices and a scalar update.
    self._VariableRankTests(state_ops.scatter_min, True, True)

  def testRepeatIndicesMaxScalar(self):
    # [FUNCTION DOC]
    # Runs the all-types comparison for scatter_max with repeated indices and a scalar update.
    self._VariableRankTests(state_ops.scatter_max, True, True)

  def testBooleanScatterUpdate(self):
    # [FUNCTION DOC]
    # Scenario (runs only when no GPU is available, otherwise it silently does nothing): on a
    # two-element boolean variable, scatter_update index one with true (scalar index) and index
    # zero (given as a 64-bit constant) with false. Expects the variable to end as false, true.
    if not test.is_gpu_available():
      with self.session(use_gpu=False):
        var = variables.Variable([True, False])
        update0 = state_ops.scatter_update(var, 1, True)
        update1 = state_ops.scatter_update(
            var, constant_op.constant(
                0, dtype=dtypes.int64), False)
        self.evaluate(var.initializer)

        self.evaluate([update0, update1])

        self.assertAllEqual([False, True], self.evaluate(var))

  def testScatterOutOfRangeCpu(self):
    # [FUNCTION DOC]
    # Scenario (runs only when no GPU is available, otherwise passes without checking): for every
    # scatter op in the lookup table, on a length-six float variable, in-range indices succeed,
    # while index -1 and index 6 must each raise an op error saying that indices entry is not
    # in the range zero to six (exclusive).
    for op, _ in _TF_OPS_TO_NUMPY.items():
      params = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32)
      updates = np.array([-3, -4, -5]).astype(np.float32)
      if not test.is_gpu_available():
        with self.session(use_gpu=False):
          ref = variables.Variable(params)
          self.evaluate(ref.initializer)

          # Indices all in range, no problem.
          indices = np.array([2, 0, 5])
          self.evaluate(op(ref, indices, updates))

          # Test some out of range errors.
          indices = np.array([-1, 0, 5])
          with self.assertRaisesOpError(
              r'indices\[0\] = -1 is not in \[0, 6\)'):
            self.evaluate(op(ref, indices, updates))

          indices = np.array([2, 0, 6])
          with self.assertRaisesOpError(r'indices\[2\] = 6 is not in \[0, 6\)'):
            self.evaluate(op(ref, indices, updates))

  # TODO(fpmc): Re-enable this test when gpu_pip test actually runs on a GPU.
  def _disabledTestScatterOutOfRangeGpu(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: dead code, never collected by the test runner because of its leading
    # underscore. Its guard is also inverted: it returns immediately when a GPU IS available, so
    # if enabled it would only run on machines without a GPU, where forcing the GPU cannot work.
    # Intended scenario: for every scatter op, with forced GPU placement, in-range and
    # out-of-range indices (-1 and 6) must all run without failure, since the GPU kernel ignores
    # out-of-range indices.
    if test.is_gpu_available():
      return
    for op, _ in _TF_OPS_TO_NUMPY.items():
      params = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32)
      updates = np.array([-3, -4, -5]).astype(np.float32)
      # With GPU, the code ignores indices that are out of range.
      # We don't test the implementation; just test there's no failures.
      with test_util.force_gpu():
        ref = variables.Variable(params)
        self.evaluate(ref.initializer)

        # Indices all in range, no problem.
        indices = np.array([2, 0, 5])
        self.evaluate(op(ref, indices, updates))

        # Indices out of range should not fail.
        indices = np.array([-1, 0, 5])
        self.evaluate(op(ref, indices, updates))
        indices = np.array([2, 0, 6])
        self.evaluate(op(ref, indices, updates))

  @test_util.run_v1_only("ResrouceVariable has deterministic scatter "
                         "implementation")
  @test_util.run_cuda_only
  def testDeterminismExceptionThrowing(self):
    # [FUNCTION DOC]
    # Scenario (graph-mode only, CUDA builds only): with op determinism enabled, scatter_update
    # with repeated indices on a legacy reference variable must raise an unimplemented-argument
    # style error stating that determinism is not yet supported in the GPU implementation of
    # the scatter ops. Resource variables are excluded because their scatter is deterministic.
    v = ref_variable.RefVariable(np.array([1., 2., 3.]))
    indices = np.array([0, 0, 0])
    updates = np.array([-3, -4, -5]).astype(np.float32)
    with test_util.deterministic_ops():
      with self.assertRaisesRegex(
          errors.UnimplementedError,
          "Determinism is not yet supported in GPU implementation of Scatter "
          "ops"):
        self.evaluate(state_ops.scatter_update(v, indices, updates))




if __name__ == '__main__':
  test.main()
