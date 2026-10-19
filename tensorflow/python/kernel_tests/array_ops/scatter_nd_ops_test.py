# [MODULE DOC]
# Test module for the N-dimensional scatter operations: in-place scatter_nd_update/add/sub on
# variables, the functional scatter_nd (and its non-aliasing-add variant), the tensor_scatter_*
# ops (update, add, sub, min, max), and the scatter_min/max/add/sub ops on variables.
# It defines numpy reference implementations used as ground truth, plus test case classes
# covering values, error messages for bad shapes/indices, gradients, and determinism mode.
# Determinism subclasses rerun the inherited tests with op determinism enabled.
# No import-time side effects beyond defining a dtype tuple and the test classes; the file is run
# as a test program through the standard test main entry point.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

GRADIENT_TESTS_DTYPES = (dtypes.bfloat16, dtypes.float16, dtypes.float32,
                         dtypes.float64)


def _AsType(v, vtype):
  # [FUNCTION DOC]
  # HELPER: converts test data to a requested numeric type for the scatter tests.
  # Takes a value v and a numeric type vtype. If v is a numpy array it returns a copy cast to
  # vtype; otherwise it calls vtype on the value (scalar conversion) and returns the result.
  # No side effects, raises whatever the conversion raises.
  return v.astype(vtype) if isinstance(v, np.ndarray) else vtype(v)


def _FlatInnerDims(tensor, ndims=2):
  # [FUNCTION DOC]
  # HELPER: numpy reference for flattening the leading dimensions, used by _NumpyScatterNd.
  # Takes a numpy array and ndims (default 2, number of trailing dims kept). Returns a reshaped
  # array whose leading dimensions are multiplied into one and whose last (ndims - 1) dimensions
  # are kept. Does not modify the input array (reshape may return a view).
  shape = list(tensor.shape)
  return tensor.reshape(
      [functools.reduce(lambda x, y: x * y, shape[:-ndims + 1], 1)] +
      shape[-ndims + 1:])


def _FlatOuterDims(tensor, ndims=2):
  # [FUNCTION DOC]
  # HELPER: numpy reference for flattening the trailing dimensions, used by _NumpyScatterNd.
  # Takes a numpy array and ndims (default 2). Returns a reshaped array that keeps the first
  # (ndims - 1) dimensions and multiplies all remaining dimensions into one final dimension.
  # The result is usually a view of the input, so writes to it modify the original array.
  shape = list(tensor.shape)
  return tensor.reshape(
      shape[:ndims - 1] +
      [functools.reduce(lambda x, y: x * y, shape[ndims - 1:], 1)])


def _NumpyScatterNd(ref, indices, updates, op):
  # [FUNCTION DOC]
  # HELPER: numpy reference implementation of N-dimensional scatter used as ground truth.
  # Inputs: ref (numpy array to scatter into), indices (integer array whose last dimension is the
  # index depth), updates (array of values, one slice per index tuple), op (binary function that
  # combines the existing slice with an update slice). Each index tuple selects a slice of ref
  # and the slice is replaced by op applied to the old slice and the update, in order, so
  # repeated indices accumulate. Returns the result with the shape of ref. Because the flattened
  # view of ref is written through, the input ref array is modified in place as a side effect
  # (when the reshape produces a view), which the callers rely on.
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


def _NumpyUpdate(ref, indices, updates):
  # [FUNCTION DOC]
  # HELPER: numpy reference for scatter-update. Overwrites the indexed slices of ref with updates
  # via the generic scatter helper and returns the result; ref may be modified in place.
  return _NumpyScatterNd(ref, indices, updates, lambda p, u: u)


def _NumpyAdd(ref, indices, updates):
  # [FUNCTION DOC]
  # HELPER: numpy reference for scatter-add. Adds the updates into the indexed slices of ref
  # (repeated indices accumulate) and returns the result; ref may be modified in place.
  return _NumpyScatterNd(ref, indices, updates, lambda p, u: p + u)


def _NumpySub(ref, indices, updates):
  # [FUNCTION DOC]
  # HELPER: numpy reference for scatter-subtract. Subtracts the updates from the indexed slices of
  # ref (repeated indices accumulate) and returns the result; ref may be modified in place.
  return _NumpyScatterNd(ref, indices, updates, lambda p, u: p - u)


def _NumpyMul(ref, indices, updates):
  # [FUNCTION DOC]
  # HELPER: numpy reference for scatter-multiply. Multiplies the indexed slices of ref by the
  # updates and returns the result. Currently unused because the multiply tests are commented
  # out until the corresponding op is needed.
  return _NumpyScatterNd(ref, indices, updates, lambda p, u: p * u)


def _NumpyDiv(ref, indices, updates):
  # [FUNCTION DOC]
  # HELPER: numpy reference for scatter-divide. Divides the indexed slices of ref by the updates
  # and returns the result. Currently unused because the divide tests are commented out until
  # the corresponding op is needed.
  return _NumpyScatterNd(ref, indices, updates, lambda p, u: p / u)


def _NumpyMin(ref, indices, updates):
  # [FUNCTION DOC]
  # HELPER: numpy reference for scatter-minimum. Replaces each indexed slice of ref by the
  # element-wise minimum with the update and returns the result. Not referenced by any test in
  # this file.
  return _NumpyScatterNd(ref, indices, updates, np.minimum)


def _NumpyMax(ref, indices, updates):
  # [FUNCTION DOC]
  # HELPER: numpy reference for scatter-maximum. Replaces each indexed slice of ref by the
  # element-wise maximum with the update and returns the result. Not referenced by any test in
  # this file.
  return _NumpyScatterNd(ref, indices, updates, np.maximum)


@test_util.with_eager_op_as_function
class StatefulScatterNdTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the in-place N-dimensional scatter ops on variables (scatter_nd_update, add, sub)
  # and the one-dimensional scatter_min/max/add/sub ops on both reference and resource
  # variables. Compares against numpy reference implementations for many dtypes and ranks,
  # checks out-of-range index errors on CPU, shape validation errors, and concurrent updates.
  # Subclassed by a determinism variant that reruns every test with op determinism enabled.

  def _VariableRankTest(self,
                        np_scatter,
                        tf_scatter,
                        vtype,
                        itype,
                        repeat_indices=False):
    # [FUNCTION DOC]
    # HELPER: shared driver for the variable-rank scatter tests.
    # Inputs: np_scatter (numpy reference that updates its first argument in place), tf_scatter
    # (graph op under test taking variable, indices, updates), vtype (numpy value type),
    # itype (numpy index type), repeat_indices (default false; when true about half of the
    # indices are duplicates of the others). For several reference/index shape pairs of rank 2
    # and 3 it builds shuffled unique (or partly repeated) indices, random updates and a random
    # reference, applies the numpy scatter to a copy and the TensorFlow op to a variable, and
    # asserts they match (tolerance 1e-6, or 1e-2 for bfloat16). Seeds numpy's random generator
    # with a fixed value (global state change). Returns nothing.
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
        tol = 1e-6 if vtype != dtypes.bfloat16.as_numpy_dtype else 1e-2
        self.assertAllClose(new, self.evaluate(ref_var), rtol=tol, atol=tol)

  def _VariableRankTests(self, np_scatter, tf_scatter):
    # [FUNCTION DOC]
    # HELPER: runs the variable-rank driver for every combination of value types (int32, float16,
    # float32, float64, complex64, complex128, bfloat16) and index types (int32, int64) using the
    # given numpy reference and TensorFlow scatter op. Returns nothing; fails on any mismatch.
    for vtype in (np.int32, np.float16, np.float32, np.float64, np.complex64,
                  np.complex128, dtypes.bfloat16.as_numpy_dtype):
      for itype in (np.int32, np.int64):
        self._VariableRankTest(np_scatter, tf_scatter, vtype, itype)

  def testSimple(self):
    # [FUNCTION DOC]
    # Scenario: scatter-update four scalar values at indices 4, 3, 1, 7 into a zero vector of
    # length eight, for each of several dtypes (ints, floats, complex, bfloat16), in a variable.
    # Runs on GPU when available. Expects the result to equal the vector with those positions
    # set to 9, 10, 11, 12 respectively.
    indices = constant_op.constant([[4], [3], [1], [7]], dtype=dtypes.int32)
    for dtype in (dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64,
                  dtypes.complex64, dtypes.complex128,
                  dtypes.bfloat16.as_numpy_dtype):
      updates = constant_op.constant([9, 10, 11, 12], dtype=dtype)
      ref = variables.Variable([0, 0, 0, 0, 0, 0, 0, 0], dtype=dtype)
      expected = np.array([0, 11, 0, 10, 9, 0, 0, 12])
      scatter = state_ops.scatter_nd_update(ref, indices, updates)
      init = variables.global_variables_initializer()

      with test_util.use_gpu():
        self.evaluate(init)
        result = self.evaluate(scatter)
        self.assertAllClose(result, expected)

  @test_util.run_in_graph_and_eager_modes
  def testString(self):
    # [FUNCTION DOC]
    # Scenario: scatter-update a variable holding eight strings at indices 4, 3, 1, 7 with four
    # new strings. Runs in both graph and eager modes. Expects the updated positions to hold the
    # new byte strings and the others to keep their original values.
    ref = variables.Variable(["qq", "ww", "ee", "rr", "", "", "", ""])
    indices = constant_op.constant([[4], [3], [1], [7]])
    updates = constant_op.constant(["aa", "dd", "cc", "bb"])
    update = state_ops.scatter_nd_update(ref, indices, updates)
    self.evaluate(variables.global_variables_initializer())
    self.assertAllEqual(
        self.evaluate(update),
        [b"qq", b"cc", b"ee", b"dd", b"aa", b"", b"", b"bb"])

  def testSimpleResource(self):
    # [FUNCTION DOC]
    # Scenario: same scalar scatter-update as the simple test but on a resource variable of
    # int32, float32 and bfloat16 values, on GPU if available. Initializes the variable, applies
    # the update and expects the variable to equal the vector with positions 4, 3, 1, 7 set to
    # 9, 10, 11, 12.
    indices = constant_op.constant([[4], [3], [1], [7]], dtype=dtypes.int32)
    for dtype in (dtypes.int32, dtypes.float32, dtypes.bfloat16):
      updates = constant_op.constant([9, 10, 11, 12], dtype=dtype)
      ref = resource_variable_ops.ResourceVariable([0, 0, 0, 0, 0, 0, 0, 0],
                                                   dtype=dtype)
      expected = np.array([0, 11, 0, 10, 9, 0, 0, 12])
      scatter = state_ops.scatter_nd_update(ref, indices, updates)

      with test_util.device(use_gpu=True):
        self.evaluate(ref.initializer)
        self.evaluate(scatter)
        self.assertAllClose(ref, expected)

  def testSimple2(self):
    # [FUNCTION DOC]
    # Scenario: scatter-update two full row elements, indices (1,0) and (1,1), of a 3x2 float
    # variable with the values 11 and 12. Expects row one to become 11, 12 and others stay zero.
    indices = constant_op.constant([[1, 0], [1, 1]], dtype=dtypes.int32)
    updates = constant_op.constant([11., 12.], dtype=dtypes.float32)
    ref = variables.Variable([[0., 0.], [0., 0.], [0., 0.]],
                             dtype=dtypes.float32)
    expected = np.array([[0., 0.], [11., 12.], [0., 0.]])
    scatter = state_ops.scatter_nd_update(ref, indices, updates)
    init = variables.global_variables_initializer()

    with self.session():
      self.evaluate(init)
      result = self.evaluate(scatter)
      self.assertAllClose(result, expected)

  def testSimple3(self):
    # [FUNCTION DOC]
    # Scenario: scatter-update a whole row slice: index one with a single update row of
    # 11 and 12 into a 3x2 zero float variable. Expects row one to hold 11, 12 and the rest zero.
    indices = constant_op.constant([[1]], dtype=dtypes.int32)
    updates = constant_op.constant([[11., 12.]], dtype=dtypes.float32)
    ref = variables.Variable([[0., 0.], [0., 0.], [0., 0.]],
                             dtype=dtypes.float32)
    expected = np.array([[0., 0.], [11., 12.], [0., 0.]])
    scatter = state_ops.scatter_nd_update(ref, indices, updates)
    init = variables.global_variables_initializer()

    with self.session():
      self.evaluate(init)
      result = self.evaluate(scatter)
      self.assertAllClose(result, expected)

  def testVariableRankUpdate(self):
    # [FUNCTION DOC]
    # Runs the variable-rank comparison for scatter_nd_update against the numpy update reference
    # over all value/index type combinations.
    self._VariableRankTests(_NumpyUpdate, state_ops.scatter_nd_update)

  def testVariableRankAdd(self):
    # [FUNCTION DOC]
    # Runs the variable-rank comparison for scatter_nd_add against the numpy add reference
    # over all value/index type combinations.
    self._VariableRankTests(_NumpyAdd, state_ops.scatter_nd_add)

  def testVariableRankSub(self):
    # [FUNCTION DOC]
    # Runs the variable-rank comparison for scatter_nd_sub against the numpy subtract reference
    # over all value/index type combinations.
    self._VariableRankTests(_NumpySub, state_ops.scatter_nd_sub)

  # TODO(ebrevdo): Re-enable when we need ScatterNdMul.
  # def testVariableRankMul(self):
  #   self._VariableRankTests(_NumpyMul, state_ops.scatter_nd_mul)

  # TODO(ebrevdo): Re-enable when we need ScatterNdDiv.
  # def testVariableRankDiv(self):
  #   self._VariableRankTests(_NumpyDiv, state_ops.scatter_nd_div)

  def _ScatterRepeatIndicesTest(self, np_scatter, tf_scatter):
    # [FUNCTION DOC]
    # HELPER: like the all-types variable-rank runner but with repeated indices enabled.
    # Takes a numpy reference and a TensorFlow scatter op; loops over value types (int32,
    # float16, float32, float64, bfloat16) and index types (int32, int64) and invokes the
    # single-case driver with duplicate indices. Returns nothing; fails on mismatch.
    for vtype in (np.int32, np.float16, np.float32, np.float64,
                  dtypes.bfloat16.as_numpy_dtype):
      for itype in (np.int32, np.int64):
        self._VariableRankTest(
            np_scatter, tf_scatter, vtype, itype, repeat_indices=True)

  def testScatterRepeatIndices(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it tests scatter_add, but the body exercises both the add
    # and the subtract N-dimensional scatter ops (the multiply/divide variants are commented out).
    # Scenario: scatter add and subtract with repeated indices, compared with numpy references
    # over all supported value and index types; accumulated results must match.
    """This tests scatter_add using indices that repeat."""
    self._ScatterRepeatIndicesTest(_NumpyAdd, state_ops.scatter_nd_add)
    self._ScatterRepeatIndicesTest(_NumpySub, state_ops.scatter_nd_sub)
    # TODO(ebrevdo): Re-enable when we need ScatterNdMul and ScatterNdDiv.
    # self._ScatterRepeatIndicesTest(_NumpyMul, state_ops.scatter_nd_mul)
    # self._ScatterRepeatIndicesTest(_NumpyDiv, state_ops.scatter_nd_div)

  # TODO(simister): Re-enable once binary size increase due to
  # extra templating is back under control and this op is re-enabled
  # def testBooleanScatterUpdate(self):
  #   with self.session(use_gpu=False) as session:
  #     var = tf.Variable([True, False])
  #     update0 = tf.compat.v1.scatter_nd_update(var, [[1]], [True])
  #     update1 = tf.compat.v1.scatter_nd_update(
  #         var, tf.constant(
  #             [[0]], dtype=tf.int64), [False])
  #     self.evaluate(var.initializer)
  #     session.run([update0, update1])
  #     self.assertAllEqual([False, True], self.evaluate(var))

  @test_util.disable_xla("b/205330448")
  def testScatterOutOfRangeCpu(self):
    # [FUNCTION DOC]
    # Scenario (CPU only, skipped under XLA): for scatter_nd add, sub and update on a float vector
    # of length six. In-range indices succeed. Index -1 and index 6 must raise an op error whose
    # message says the indices entry does not index into shape [6], naming the offending entry.
    # TODO(simister): Re-enable once binary size increase due to
    # scatter_nd ops is under control.
    #  tf.scatter_nd_mul, tf.scatter_nd_div,
    for op in (state_ops.scatter_nd_add, state_ops.scatter_nd_sub,
               state_ops.scatter_nd_update):
      params = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32)
      updates = np.array([-3, -4, -5]).astype(np.float32)
      with test_util.device(use_gpu=False):
        ref = variable_v1.VariableV1(params)
        self.evaluate(ref.initializer)

        # Indices all in range, no problem.
        indices = np.array([[2], [0], [5]])
        self.evaluate(op(ref, indices, updates))

        # Test some out of range errors.
        indices = np.array([[-1], [0], [5]])
        with self.assertRaisesOpError(
            r"indices\[0\] = \[-1\] does not index into shape \[6\]"):
          self.evaluate(op(ref, indices, updates))

        indices = np.array([[2], [0], [6]])
        with self.assertRaisesOpError(
            r"indices\[2\] = \[6\] does not index into shape \[6\]"):
          self.evaluate(op(ref, indices, updates))

  def testRank3ValidShape(self):
    # [FUNCTION DOC]
    # Scenario: scatter_nd_update with rank-3 indices and updates (all 2x2x2) on a 2x2x2 variable.
    # Expects the static output shape to equal the variable shape; no execution needed.
    indices = array_ops.zeros([2, 2, 2], dtypes.int32)
    updates = array_ops.zeros([2, 2, 2], dtypes.int32)
    shape = np.array([2, 2, 2])
    ref = variables.Variable(array_ops.zeros(shape, dtypes.int32))
    self.assertAllEqual(
        state_ops.scatter_nd_update(ref, indices,
                                    updates).get_shape().as_list(), shape)

  @test_util.disable_xla("b/123337890")  # Error messages differ
  def testResVarInvalidOutputShape(self):
    # [FUNCTION DOC]
    # Scenario (skipped under XLA because error messages differ): scatter_nd_update on a scalar
    # (rank-zero) float variable. Expects an op error saying the output must be at least
    # one-dimensional when the update is evaluated.
    res = variables.Variable(
        initial_value=lambda: array_ops.zeros(shape=[], dtype=dtypes.float32),
        dtype=dtypes.float32)
    with self.cached_session():
      self.evaluate(res.initializer)
      with self.assertRaisesOpError("Output must be at least 1-D"):
        state_ops.scatter_nd_update(res, [[0]], [0.22]).eval()

  def testExtraIndicesDimensions(self):
    # [FUNCTION DOC]
    # Scenario: scatter_nd_update with indices of shape 1x1x2 and updates of shape 1x1 on a 2x2
    # integer variable (an extra batch dimension in indices). Expects the static result shape to
    # be 2x2 and, after initialization and evaluation, an all-zero 2x2 result.
    indices = array_ops.zeros([1, 1, 2], dtypes.int32)
    updates = array_ops.zeros([1, 1], dtypes.int32)
    shape = np.array([2, 2])
    ref = variables.Variable(array_ops.zeros(shape, dtypes.int32))
    scatter_update = state_ops.scatter_nd_update(ref, indices, updates)
    self.assertAllEqual(scatter_update.get_shape().as_list(), shape)

    expected_result = np.zeros([2, 2], dtype=np.int32)
    with self.cached_session():
      self.evaluate(ref.initializer)
      self.assertAllEqual(expected_result, self.evaluate(scatter_update))

  def testRank3InvalidShape1(self):
    # [FUNCTION DOC]
    # Scenario: scatter_nd_update where the leading dimension of indices (3) differs from that of
    # updates (2). Expects an invalid-argument or value error whose message mentions a
    # dimension range of indices.
    indices = array_ops.zeros([3, 2, 2], dtypes.int32)
    updates = array_ops.zeros([2, 2, 2], dtypes.int32)
    shape = np.array([2, 2, 2])
    ref = variables.Variable(array_ops.zeros(shape, dtypes.int32))
    with self.assertRaisesWithPredicateMatch(
        (errors.InvalidArgumentError, ValueError),
        r"Dimensions \[\d,\d\) of indices\[shape="):
      state_ops.scatter_nd_update(ref, indices, updates)

  def testRank3InvalidShape2(self):
    # [FUNCTION DOC]
    # Scenario: scatter_nd_update where updates (2x2) lack the trailing slice dimension implied
    # by indices of shape 2x2x1 on a 2x2x2 variable. Expects an invalid-argument or value error
    # whose message mentions a dimension range of the input.
    indices = array_ops.zeros([2, 2, 1], dtypes.int32)
    updates = array_ops.zeros([2, 2], dtypes.int32)
    shape = np.array([2, 2, 2])
    ref = variables.Variable(array_ops.zeros(shape, dtypes.int32))
    with self.assertRaisesWithPredicateMatch(
        (errors.InvalidArgumentError, ValueError),
        r"Dimensions \[\d,\d\) of input\[shape="):
      state_ops.scatter_nd_update(ref, indices, updates)

  def testConcurrentUpdates(self):
    # [FUNCTION DOC]
    # Scenario: scatter_nd_add of 10000 random float64 values, all at the same index (0, 1) of a
    # 2x2 zero variable. Expects element (0, 1) to equal the sum of all values and the rest zero,
    # within default closeness tolerance (checked with a plain assert, not a test assertion).
    num_updates = 10000
    update_values = np.random.rand(num_updates)
    ref = variables.Variable(np.zeros([2, 2]), dtype=dtypes.float64)
    indices = constant_op.constant([[0, 1]] * num_updates, dtype=dtypes.int32)
    updates = constant_op.constant(update_values, dtype=dtypes.float64)

    expected_result = np.zeros([2, 2], dtype=np.float64)
    expected_result[0, 1] = np.sum(update_values)

    scatter = state_ops.scatter_nd_add(ref, indices, updates)
    init = variables.global_variables_initializer()

    self.evaluate(init)
    result = self.evaluate(scatter)
    assert np.allclose(result, expected_result)

  @test_util.run_in_graph_and_eager_modes
  def testMin(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager): scatter_min on a reference variable and a resource variable of
    # eight ones with indices 4, 3, 1, 7 and updates 0, 2, -1, 2. Expects each position to hold
    # the smaller of old and new value, both in the returned value and in the variable.
    variable = variables.Variable(array_ops.ones([8], dtype=dtypes.int32))
    resource_variable = resource_variable_ops.ResourceVariable(
        array_ops.ones([8], dtype=dtypes.int32))
    indices = constant_op.constant([4, 3, 1, 7])
    updates = constant_op.constant([0, 2, -1, 2], dtype=dtypes.int32)

    for ref in (variable, resource_variable):
      min_result = state_ops.scatter_min(ref, indices, updates)
      self.evaluate(ref.initializer)

      expected_result = constant_op.constant([1, -1, 1, 1, 0, 1, 1, 1])
      self.assertAllEqual(self.evaluate(min_result), expected_result)
      self.assertAllEqual(self.evaluate(ref), expected_result)

  @test_util.run_in_graph_and_eager_modes
  def testMax(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager): scatter_max on a reference variable and a resource variable of
    # eight ones with indices 4, 3, 1, 7 and updates 0, 2, -1, 2. Expects each position to hold
    # the larger of old and new value, both in the returned value and in the variable.
    variable = variables.Variable(array_ops.ones([8], dtype=dtypes.int32))
    resource_variable = resource_variable_ops.ResourceVariable(
        array_ops.ones([8], dtype=dtypes.int32))
    indices = constant_op.constant([4, 3, 1, 7])
    updates = constant_op.constant([0, 2, -1, 2], dtype=dtypes.int32)

    for ref in (variable, resource_variable):
      max_result = state_ops.scatter_max(ref, indices, updates)
      self.evaluate(ref.initializer)

      expected_result = constant_op.constant([1, 1, 1, 2, 1, 1, 1, 2])
      self.assertAllEqual(self.evaluate(max_result), expected_result)
      self.assertAllEqual(self.evaluate(ref), expected_result)

  @test_util.run_in_graph_and_eager_modes
  def testAdd(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager): scatter_add on a reference variable and a resource variable of
    # eight ones with indices 4, 3, 1, 7 and updates 0, 2, -1, 3. Expects the updates to be added
    # to the old values, in both the returned value and the variable.
    variable = variables.Variable(array_ops.ones([8], dtype=dtypes.int32))
    resource_variable = resource_variable_ops.ResourceVariable(
        array_ops.ones([8], dtype=dtypes.int32))
    indices = constant_op.constant([4, 3, 1, 7])
    updates = constant_op.constant([0, 2, -1, 3], dtype=dtypes.int32)

    for ref in (variable, resource_variable):
      add_result = state_ops.scatter_add(ref, indices, updates)
      self.evaluate(ref.initializer)

      expected_result = constant_op.constant([1, 0, 1, 3, 1, 1, 1, 4])
      self.assertAllEqual(self.evaluate(add_result), expected_result)
      self.assertAllEqual(self.evaluate(ref), expected_result)

  @test_util.run_in_graph_and_eager_modes
  def testSub(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager): scatter_sub on a reference variable and a resource variable of
    # eight ones with indices 4, 3, 1, 7 and updates 0, 2, -1, 2. Expects the updates to be
    # subtracted from the old values, in both the returned value and the variable.
    variable = variables.Variable(array_ops.ones([8], dtype=dtypes.int32))
    resource_variable = resource_variable_ops.ResourceVariable(
        array_ops.ones([8], dtype=dtypes.int32))
    indices = constant_op.constant([4, 3, 1, 7])
    updates = constant_op.constant([0, 2, -1, 2], dtype=dtypes.int32)

    for ref in (variable, resource_variable):
      sub_result = state_ops.scatter_sub(ref, indices, updates)
      self.evaluate(ref.initializer)

      expected_result = constant_op.constant([1, 2, 1, -1, 1, 1, 1, -1])
      self.assertAllEqual(self.evaluate(sub_result), expected_result)
      self.assertAllEqual(self.evaluate(ref), expected_result)

  # TODO(fpmc): Re-enable this test when gpu_pip test actually runs on a GPU.
  def _disabledTestScatterOutOfRangeGpu(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: dead code. The leading underscore and "disabled" name mean the test runner
    # never collects it, so it never runs (a comment says to re-enable it once GPU tests run).
    # Intended scenario: only when built with CUDA, scatter_nd add, sub and update on a GPU with
    # in-range and out-of-range indices; on GPU out-of-range indices are ignored, so it only
    # checks that nothing fails. Note it passes one-dimensional indices, unlike the CPU test.
    if not test.IsBuiltWithCuda():
      return
    # TODO(simister): Re-enable once binary size increase due to
    # scatter_nd ops is under control.
    # tf.scatter_nd_mul, tf.scatter_nd_div,
    for op in (state_ops.scatter_nd_add, state_ops.scatter_nd_sub,
               state_ops.scatter_nd_update):
      params = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32)
      updates = np.array([-3, -4, -5]).astype(np.float32)
      # With GPU, the code ignores indices that are out of range.
      # We don't test the implementation; just test there's no failures.
      with self.cached_session(force_gpu=True):
        ref = variables.Variable(params)
        self.evaluate(ref.initializer)

        # Indices all in range, no problem.
        indices = np.array([2, 0, 5])
        op(ref, indices, updates).eval()

        # Indices out of range should not fail.
        indices = np.array([-1, 0, 5])
        op(ref, indices, updates).eval()
        indices = np.array([2, 0, 6])
        op(ref, indices, updates).eval()


class StatefulScatterNdDeterminismTest(StatefulScatterNdTest):
  # [CLASS DOC]
  # Reruns all stateful scatter tests with TensorFlow op determinism enabled (set up before and
  # reset after each test) and adds a test that repeated scatter updates give identical results.

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the parent setup, then enables global op determinism (global state change undone in
    # tear down). Takes and returns nothing.
    super().setUp()
    config.enable_op_determinism()

  def tearDown(self):
    # [FUNCTION DOC]
    # Calls the parent tear down, then disables global op determinism to restore the default.
    # Takes and returns nothing.
    super().tearDown()
    config.disable_op_determinism()

  def testDeterminism(self):
    # [FUNCTION DOC]
    # Scenario: scatter_nd_update of 100000 random values all at the same index of a one-element
    # variable, repeated six times with fresh variables. Expects every run to return exactly the
    # same value as the first, as required under determinism mode.
    ref = variables.Variable(array_ops.zeros([1]))
    indices = array_ops.zeros([100000, 1], dtypes.int32)
    values = np.random.randn(100000)
    self.evaluate(variables.global_variables_initializer())
    val = self.evaluate(state_ops.scatter_nd_update(ref, indices, values))
    for _ in range(5):
      ref2 = variables.Variable(array_ops.zeros([1]))
      self.evaluate(variables.global_variables_initializer())
      val2 = self.evaluate(state_ops.scatter_nd_update(ref2, indices, values))
      self.assertAllEqual(val, val2)


@test_util.with_eager_op_as_function
class ScatterNdTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the functional scatter_nd op that builds a new tensor from indices, updates and a
  # shape: value checks for bool and string types, static shape handling with unknown shapes,
  # error messages for invalid or empty output shapes, gradients with respect to updates (and
  # input for the non-aliasing variant) and smoke tests for batched index shapes.
  # The class attribute non_aliasing_add_test (false here) tells the gradient tests whether to
  # also check the input gradient. Subclasses override the scatter_nd method to test other ops.
  non_aliasing_add_test = False

  def scatter_nd(self, indices, updates, shape, input_=None):
    # [FUNCTION DOC]
    # HELPER: hook through which the tests call the op under test, so subclasses can swap it.
    # Takes indices, updates, shape (output shape) and input_ (ignored here, deleted unused).
    # Returns the tensor produced by the scatter_nd op.
    del input_  # input_ is not used in scatter_nd
    return array_ops.scatter_nd(indices, updates, shape)

  @test_util.run_in_graph_and_eager_modes
  def testBool(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager): scatter four boolean values into a length-eight vector, first
    # with unique indices, then with a repeated index carrying identical values. Expects true at
    # the scattered true positions and false elsewhere in both cases.
    indices = constant_op.constant([[4], [3], [1], [7]], dtype=dtypes.int32)
    updates = constant_op.constant([False, True, False, True],
                                   dtype=dtypes.bool)
    expected = np.array([False, False, False, True, False, False, False, True])
    scatter = self.scatter_nd(indices, updates, shape=(8,))
    result = self.evaluate(scatter)
    self.assertAllEqual(expected, result)

    # Same indice is updated twice by same value.
    indices = constant_op.constant([[4], [3], [3], [7]], dtype=dtypes.int32)
    updates = constant_op.constant([False, True, True, True], dtype=dtypes.bool)
    expected = np.array([False, False, False, True, False, False, False, True])
    scatter = self.scatter_nd(indices, updates, shape=(8,))
    result = self.evaluate(scatter)
    self.assertAllEqual(expected, result)

  @test_util.run_in_graph_and_eager_modes
  def testInvalidShape(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager): call scatter_nd with one-dimensional indices where a nested
    # index list is required. Expects an invalid-argument error in eager mode and a value error
    # in graph mode.
    # TODO(apassos) figure out how to unify these errors
    with self.assertRaises(errors.InvalidArgumentError if context
                           .executing_eagerly() else ValueError):
      array_ops.scatter_nd(
          indices=[0],  # this should be indices=[[0]]
          updates=[0.0],
          shape=[1])

  def testString(self):
    # [FUNCTION DOC]
    # Scenario: scatter four strings into a length-eight vector of empty strings. Checks unique
    # indices, a repeated index with the same value (expects concatenation of both strings) and a
    # repeated index with different values (accepts either concatenation order because the
    # order of duplicate updates is not defined). Evaluated in cached sessions.
    indices = constant_op.constant([[4], [3], [1], [7]], dtype=dtypes.int32)
    updates = constant_op.constant(["four", "three", "one", "seven"],
                                   dtype=dtypes.string)
    expected = np.array(
        [b"", b"one", b"", b"three", b"four", b"", b"", b"seven"])
    scatter = self.scatter_nd(indices, updates, shape=(8,))
    with self.cached_session() as sess:
      result = self.evaluate(scatter)
      self.assertAllEqual(expected, result)

    # Same indice is updated twice by same value.
    indices = constant_op.constant([[4], [3], [3], [7]], dtype=dtypes.int32)
    updates = constant_op.constant(["a", "b", "b", "c"], dtype=dtypes.string)
    expected = np.array([b"", b"", b"", b"bb", b"a", b"", b"", b"c"])
    scatter = self.scatter_nd(indices, updates, shape=(8,))
    with self.cached_session() as sess:
      result = self.evaluate(scatter)
      self.assertAllEqual(expected, result)

    # Same indice is updated twice by different value.
    indices = constant_op.constant([[4], [3], [3], [7]], dtype=dtypes.int32)
    updates = constant_op.constant(["a", "b", "c", "d"], dtype=dtypes.string)
    expected = [
        np.array([b"", b"", b"", b"bc", b"a", b"", b"", b"d"]),
        np.array([b"", b"", b"", b"cb", b"a", b"", b"", b"d"])
    ]
    scatter = self.scatter_nd(indices, updates, shape=(8,))
    with self.cached_session() as sess:
      result = self.evaluate(scatter)
      self.assertTrue(
          np.array_equal(result, expected[0]) or
          np.array_equal(result, expected[1]))

  def testRank3ValidShape(self):
    # [FUNCTION DOC]
    # Scenario: scatter_nd with rank-3 indices and updates (2x2x2) and output shape 2x2x2.
    # Expects the static result shape to equal the requested shape.
    indices = array_ops.zeros([2, 2, 2], dtypes.int32)
    updates = array_ops.zeros([2, 2, 2], dtypes.int32)
    shape = np.array([2, 2, 2])
    self.assertAllEqual(
        self.scatter_nd(indices, updates, shape).get_shape().as_list(), shape)

  def testExtraIndicesDimensions(self):
    # [FUNCTION DOC]
    # Scenario: scatter_nd with indices 1x1x2 and updates 1x1 into a 2x2 output. Expects the
    # static shape to be 2x2 and the evaluated result to be an all-zero 2x2 integer array.
    indices = array_ops.zeros([1, 1, 2], dtypes.int32)
    updates = array_ops.zeros([1, 1], dtypes.int32)
    shape = np.array([2, 2])
    scatter = self.scatter_nd(indices, updates, shape)
    self.assertAllEqual(scatter.get_shape().as_list(), shape)
    expected_result = np.zeros([2, 2], dtype=np.int32)
    self.assertAllEqual(expected_result, self.evaluate(scatter))

  def testUndefinedIndicesShape(self):
    # [FUNCTION DOC]
    # Scenario (graph building only): indices is a placeholder of unknown shape. Expects scatter_nd
    # to build the graph without raising; nothing is executed or asserted beyond that.
    # Placeholders are only valid in Graph.
    with ops.Graph().as_default():
      indices = array_ops.placeholder(dtypes.int32, shape=None)
      updates = array_ops.placeholder(dtypes.int32, shape=[2, 2, 2])
      shape = constant_op.constant([2, 2, 2], dtypes.int32)
      self.scatter_nd(indices, updates, shape)

  def testUndefinedUpdatesShape(self):
    # [FUNCTION DOC]
    # Scenario (graph building only): updates is a placeholder of unknown shape. Expects
    # scatter_nd to build the graph without raising; nothing is executed or asserted.
    # Placeholders are only valid in Graph.
    with ops.Graph().as_default():
      indices = array_ops.placeholder(dtypes.int32, shape=[2, 2, 2])
      updates = array_ops.placeholder(dtypes.int32, shape=None)
      shape = constant_op.constant([2, 2, 2], dtypes.int32)
      self.scatter_nd(indices, updates, shape)

  def testUndefinedOutputShape(self):
    # [FUNCTION DOC]
    # Scenario (graph building only): the output shape is a placeholder vector of unknown length.
    # Expects scatter_nd to build the graph without raising; nothing is executed or asserted.
    # Placeholders are only valid in Graph.
    with ops.Graph().as_default():
      indices = array_ops.placeholder(dtypes.int32, shape=[2, 2, 2])
      updates = array_ops.placeholder(dtypes.int32, shape=[2, 2, 2])
      shape = array_ops.placeholder(dtypes.int32, shape=[None])
      self.scatter_nd(indices, updates, shape)

  def testEmptyOutputShape1(self):
    # [FUNCTION DOC]
    # Scenario: constant indices and updates (2x2x2) targeting an output of shape 0x3x2 that has
    # no elements. Expects an invalid-argument or value error saying indices and updates were
    # specified for an empty output.
    indices = array_ops.zeros([2, 2, 2], dtypes.int32)
    updates = array_ops.zeros([2, 2, 2], dtypes.int32)
    shape = constant_op.constant([0, 3, 2], dtypes.int32)

    with self.assertRaisesWithPredicateMatch(
        (errors.InvalidArgumentError, ValueError),
        "Indices and updates specified for empty"):
      self.scatter_nd(indices, updates, shape)

  def testEmptyOutputShape2(self):
    # [FUNCTION DOC]
    # Scenario (graph mode): indices and updates are placeholders of unknown shape, output shape
    # is 0x3x2. Feeding 2x2x2 integer zeros at run time must raise an op error saying indices and
    # updates were specified for an empty input or output.
    with ops.Graph().as_default():
      indices = array_ops.placeholder(dtypes.int32, shape=None)
      updates = array_ops.placeholder(dtypes.int32, shape=None)
      shape = constant_op.constant([0, 3, 2], dtypes.int32)

      with self.cached_session():
        with self.assertRaisesOpError(
            "Indices and updates specified for empty (input|output)"):
          self.scatter_nd(indices, updates, shape).eval(
              feed_dict={
                  indices: np.zeros([2, 2, 2], dtype=np.int32),
                  updates: np.zeros([2, 2, 2], dtype=np.int32)
              })

  def testEmptyOutputShape3(self):
    # [FUNCTION DOC]
    # Scenario: empty indices, empty updates and an output shape of a single zero dimension.
    # Expects scatter_nd to succeed and return an empty result (size zero).
    indices = array_ops.zeros([0], dtypes.int32)
    updates = array_ops.zeros([0], dtypes.int32)
    shape = constant_op.constant([0], dtypes.int32)
    scatter = self.scatter_nd(indices, updates, shape)

    with self.cached_session():
      self.assertEqual(self.evaluate(scatter).size, 0)

  def testRank3InvalidShape1(self):
    # [FUNCTION DOC]
    # Scenario: indices leading dimension (3) differs from updates leading dimension (2).
    # Expects an invalid-argument or value error whose message mentions a dimension range of
    # indices.
    indices = array_ops.zeros([3, 2, 2], dtypes.int32)
    updates = array_ops.zeros([2, 2, 2], dtypes.int32)
    shape = np.array([2, 2, 2])
    with self.assertRaisesWithPredicateMatch(
        (errors.InvalidArgumentError, ValueError),
        r"Dimensions \[\d\,\d\) of indices\[shape="):
      self.scatter_nd(indices, updates, shape)

  def testRank3InvalidShape2(self):
    # [FUNCTION DOC]
    # Scenario: updates (2x2) lack the trailing slice dimension implied by indices 2x2x1 and
    # output 2x2x2. Expects an invalid-argument or value error whose message mentions a dimension
    # range of the input.
    indices = array_ops.zeros([2, 2, 1], dtypes.int32)
    updates = array_ops.zeros([2, 2], dtypes.int32)
    shape = np.array([2, 2, 2])
    with self.assertRaisesWithPredicateMatch(
        (errors.InvalidArgumentError, ValueError),
        r"Dimensions \[\d\,\d\) of input\[shape="):
      self.scatter_nd(indices, updates, shape)

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def testGradientsRank2ElementUpdate(self, use_tape):
    # [FUNCTION DOC]
    # Parameterized by use_tape (a real gradient tape, or the graph-mode abstraction; runs both
    # forms when eager). For each gradient test dtype (bfloat16, float16, float32, float64):
    # scatter two scalar updates into a 2x2 zero tensor, then take gradients of the output with
    # respect to updates and the input using given upstream gradients. Expects the updates
    # gradient to be the upstream values at the scattered positions, and, only when the
    # non-aliasing flag is set, the input gradient to equal the upstream gradient.
    for dtype in GRADIENT_TESTS_DTYPES:
      with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
        indices = constant_op.constant([[0, 0], [1, 1]], dtype=dtypes.int32)
        updates = constant_op.constant([1, 4], dtype=dtype)
        tape.watch(updates)
        shape = constant_op.constant([2, 2], dtype=dtypes.int32)
        input_ = array_ops.zeros(shape, dtype=dtype)
        tape.watch(input_)
        outputs = self.scatter_nd(indices, updates, shape, input_)

        grad_vals = constant_op.constant([[1, 2], [3, 4]], dtype=dtype)

        updates_grad, input_grad = tape.gradient([outputs], [updates, input_],
                                                 [grad_vals])
      expected_updates_grad = np.array([1, 4], dtype=dtype.as_numpy_dtype)
      expected_input_grad = np.array([[1, 2], [3, 4]],
                                     dtype=dtype.as_numpy_dtype)
      self.assertAllEqual(expected_updates_grad, self.evaluate(updates_grad))
      if self.non_aliasing_add_test:
        self.assertAllEqual(expected_input_grad, self.evaluate(input_grad))

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def testGradientsRank2SliceUpdate(self, use_tape):
    # [FUNCTION DOC]
    # Parameterized by use_tape as in the element-update gradient test. For each gradient test
    # dtype, scatter two row slices (indices 1 and 0) into a 2x2 zero tensor and take gradients
    # with respect to updates and input. Expects the updates gradient to be the upstream rows
    # swapped back to update order, and, when the non-aliasing flag is set, the input gradient
    # to equal the upstream gradient.
    for dtype in GRADIENT_TESTS_DTYPES:
      with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
        indices = constant_op.constant([[1], [0]], dtype=dtypes.int32)
        updates = constant_op.constant([[3, 4], [1, 2]], dtype=dtype)
        tape.watch(updates)
        shape = constant_op.constant([2, 2], dtype=dtypes.int32)
        input_ = array_ops.zeros(shape, dtype=dtype)
        tape.watch(input_)
        outputs = self.scatter_nd(indices, updates, shape, input_)

        grad_vals = constant_op.constant([[3, 4], [1, 2]], dtype=dtype)
        updates_grad, input_grad = tape.gradient([outputs], [updates, input_],
                                                 [grad_vals])
      expected_updates_grad = np.array([[1, 2], [3, 4]],
                                       dtype=dtype.as_numpy_dtype)
      expected_input_grad = np.array([[3, 4], [1, 2]],
                                     dtype=dtype.as_numpy_dtype)
      self.assertAllEqual(expected_updates_grad, self.evaluate(updates_grad))
      if self.non_aliasing_add_test:
        self.assertAllEqual(expected_input_grad, self.evaluate(input_grad))

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def testGradientsRank3SliceUpdate(self, use_tape):
    # [FUNCTION DOC]
    # Parameterized by use_tape. For each gradient test dtype, scatter a batch of 2x2 index
    # pairs with 2x2x2 updates into a 2x2x2 zero tensor and take gradients with respect to
    # updates and input. Expects the updates gradient to be the upstream gradient gathered at the
    # scattered positions, and, when the non-aliasing flag is set, the input gradient to equal
    # the upstream gradient.
    for dtype in GRADIENT_TESTS_DTYPES:
      with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
        indices = constant_op.constant([[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
                                       dtype=dtypes.int32)
        updates = constant_op.constant([[[5, 7], [2, 4]], [[1, 3], [6, 8]]],
                                       dtype=dtype)
        tape.watch(updates)
        shape = constant_op.constant([2, 2, 2], dtype=dtypes.int32)
        input_ = array_ops.zeros(shape, dtype=dtype)
        tape.watch(input_)
        outputs = self.scatter_nd(indices, updates, shape, input_)

        grad_vals = constant_op.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                                         dtype=dtype)
        updates_grad, input_grad = tape.gradient([outputs], [updates, input_],
                                                 [grad_vals])
      expected_updates_grad = np.array([[[3, 4], [5, 6]], [[1, 2], [7, 8]]],
                                       dtype=dtype.as_numpy_dtype)
      expected_input_grad = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]],
                                     dtype=dtype.as_numpy_dtype)
      self.assertAllEqual(expected_updates_grad, self.evaluate(updates_grad))
      if self.non_aliasing_add_test:
        self.assertAllEqual(expected_input_grad, self.evaluate(input_grad))

  @parameterized.parameters(set((True, context.executing_eagerly())))
  def testGradientsRank7SliceUpdate(self, use_tape):
    # [FUNCTION DOC]
    # Parameterized by use_tape. Same gradient check as the rank-3 case but with rank-seven
    # indices, updates and output (shape 1x1x2x1x1x2x2) for each gradient test dtype. Expects
    # the updates gradient to be the upstream gradient gathered at the scattered positions, and,
    # when the non-aliasing flag is set, the input gradient to equal the upstream gradient.
    for dtype in GRADIENT_TESTS_DTYPES:
      with test_util.AbstractGradientTape(use_tape=use_tape) as tape:
        indices = constant_op.constant(
            [[[[[[[0, 0, 0, 0, 0, 1], [0, 0, 1, 0, 0, 0]]]],
               [[[[0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 1]]]]]]],
            dtype=dtypes.int32)
        updates = constant_op.constant(
            [[[[[[[5, 6], [2, 4]]]], [[[[1, 3], [6, 8]]]]]]], dtype=dtype)
        tape.watch(updates)
        shape = constant_op.constant([1, 1, 2, 1, 1, 2, 2], dtype=dtypes.int32)
        input_ = array_ops.zeros(shape, dtype=dtype)
        tape.watch(input_)
        outputs = self.scatter_nd(indices, updates, shape, input_)

        grad_vals = constant_op.constant(
            [[[[[[[1, 2], [3, 4]]]], [[[[5, 6], [7, 8]]]]]]], dtype=dtype)
        updates_grad, input_grad = tape.gradient([outputs], [updates, input_],
                                                 [grad_vals])
      expected_updates_grad = np.array(
          [[[[[[[3, 4], [5, 6]]]], [[[[1, 2], [7, 8]]]]]]],
          dtype=dtype.as_numpy_dtype)
      expected_input_grad = np.array(
          [[[[[[[1, 2], [3, 4]]]], [[[[5, 6], [7, 8]]]]]]],
          dtype=dtype.as_numpy_dtype)
      self.assertAllEqual(expected_updates_grad, self.evaluate(updates_grad))
      if self.non_aliasing_add_test:
        self.assertAllEqual(expected_input_grad, self.evaluate(input_grad))

  def testScatterNdRepeatedIndicesAdd(self):
    # [FUNCTION DOC]
    # Scenario: 100000 random values all scattered to index zero of a one-element output.
    # Expects the result to be close to the sum of all values (duplicates accumulate).
    indices = array_ops.zeros([100000, 1], dtypes.int32)
    values = np.random.randn(100000)
    shape = [1]
    val = self.evaluate(self.scatter_nd(indices, values, shape))
    self.assertAllClose([np.sum(values)], val)

  def testSmokeScatterNdBatch2DSliceDim2(self):
    # [FUNCTION DOC]
    # Smoke test: zero indices of shape 3x5x2 and zero updates 3x5x7 into a 4x6x7 output.
    # Only checks that evaluation succeeds without error.
    indices = array_ops.zeros([3, 5, 2], dtype=dtypes.int32)
    values = array_ops.zeros([3, 5, 7])
    shape = [4, 6, 7]
    self.evaluate(self.scatter_nd(indices, values, shape))

  def testSmokeScatterNdBatch1DSliceDim2(self):
    # [FUNCTION DOC]
    # Smoke test: empty indices (0x2) and empty updates (0x7) into a 4x6x7 output. Only checks
    # that evaluation succeeds without error.
    indices = array_ops.zeros([0, 2], dtype=dtypes.int32)
    values = array_ops.zeros([0, 7])
    shape = [4, 6, 7]
    self.evaluate(self.scatter_nd(indices, values, shape))

  def testSmokeScatterNdBatch1DSliceDim3ShapeRank7(self):
    # [FUNCTION DOC]
    # Smoke test: one index triple (1x3) and updates 1x6x7x8x9 into a rank-seven output of shape
    # 3x4x5x6x7x8x9. Only checks that evaluation succeeds without error.
    indices = array_ops.zeros([1, 3], dtype=dtypes.int32)
    values = array_ops.zeros([1, 6, 7, 8, 9])
    shape = [3, 4, 5, 6, 7, 8, 9]
    self.evaluate(self.scatter_nd(indices, values, shape))

  def testSmokeScatterNdBatch2DSliceDim3ShapeRank7(self):
    # [FUNCTION DOC]
    # Smoke test: indices 1x2x3 and updates 1x2x6x7x8x9 into a rank-seven output of shape
    # 3x4x5x6x7x8x9. Only checks that evaluation succeeds without error.
    indices = array_ops.zeros([1, 2, 3], dtype=dtypes.int32)
    values = array_ops.zeros([1, 2, 6, 7, 8, 9])
    shape = [3, 4, 5, 6, 7, 8, 9]
    self.evaluate(self.scatter_nd(indices, values, shape))


class ScatterNdNonAliasingAddTest(ScatterNdTest):
  # [CLASS DOC]
  # Reruns the functional scatter tests against the non-aliasing-add op, which adds updates into
  # an input tensor (zeros by default) and returns a new tensor. Sets the class flag so gradient
  # tests also verify the gradient with respect to the input. String tests are stubbed out.
  non_aliasing_add_test = True

  def scatter_nd(self, indices, updates, shape, input_=None):
    # [FUNCTION DOC]
    # HELPER: override of the op hook used by the inherited tests. Takes indices, updates, shape
    # and optional input_; when input_ is missing it creates zeros of the given shape with the
    # dtype of updates. Returns the result of the non-aliasing scatter-add of updates into input_.
    # Note it reads the dtype attribute of updates, so updates must be a tensor, not a raw array.
    input_ = (
        input_ if input_ is not None else array_ops.zeros(
            shape, dtype=updates.dtype))
    return array_ops.scatter_nd_non_aliasing_add(input_, indices, updates)

  def testString(self):
    # [FUNCTION DOC]
    # Placeholder that overrides the inherited string test with a no-op because string values are
    # not supported by the non-aliasing add op yet. Does nothing.
    # Not supported yet.
    pass

  # TODO(testString): Enable this test when the above testString is enabled.
  def testStringWithEagerOpAsFunctionEnabled(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: no inherited test has this name, so it overrides nothing; it appears to be a
    # leftover stub of a renamed test and does nothing.
    # Placeholder no-op test for string support, awaiting the same feature as the string test.
    # Not supported yet.
    pass


class ScatterNdDeterminismTest(ScatterNdTest):
  # [CLASS DOC]
  # Reruns all functional scatter tests with op determinism enabled around each test and adds a
  # repeatability test for scattering many duplicate indices.

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the parent setup, then enables global op determinism (global state change undone in
    # tear down). Takes and returns nothing.
    super().setUp()
    config.enable_op_determinism()

  def tearDown(self):
    # [FUNCTION DOC]
    # Calls the parent tear down, then disables global op determinism to restore the default.
    # Takes and returns nothing.
    super().tearDown()
    config.disable_op_determinism()

  def testDeterminism(self):
    # [FUNCTION DOC]
    # Scenario: scatter 100000 random values to a single index of a one-element output, repeated
    # six times. Expects every run to return exactly the same value as the first run.
    indices = array_ops.zeros([100000, 1], dtypes.int32)
    values = np.random.randn(100000)
    shape = [1]
    val = self.evaluate(self.scatter_nd(indices, values, shape))
    for _ in range(5):
      val2 = self.evaluate(self.scatter_nd(indices, values, shape))
      self.assertAllEqual(val, val2)


class ScatterNdNonAliasingAddDeterminismTest(ScatterNdDeterminismTest,
                                             ScatterNdNonAliasingAddTest):
  # [CLASS DOC]
  # Combination of the determinism tests and the non-aliasing-add variant: runs all inherited
  # scatter tests against the non-aliasing add op with op determinism enabled. Adds no behavior.
  pass


class ScatterNdTensorTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the tensor_scatter_update/add/sub/min/max ops that return a new tensor instead of
  # modifying a variable: values for numeric and string types, numerical and manual gradient
  # checks, forwarding inside a compiled function, and repeated-index behavior.
  # Subclassed by a determinism variant.

  @test_util.run_in_graph_and_eager_modes
  def testUpdateAddSub(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager): for int32 and float32, apply tensor_scatter update, add and sub
    # to a tensor of eight ones with indices 4, 3, 1, 7 and values 9 to 12. Expects the update
    # to overwrite, the add to add and the sub to subtract the values at those positions.
    for dtype in (dtypes.int32, dtypes.float32):
      indices = constant_op.constant([[4], [3], [1], [7]])
      updates = constant_op.constant([9, 10, 11, 12], dtype=dtype)
      t = array_ops.ones([8], dtype=dtype)
      assigned = array_ops.tensor_scatter_update(t, indices, updates)
      added = array_ops.tensor_scatter_add(t, indices, updates)
      subbed = array_ops.tensor_scatter_sub(t, indices, updates)

      self.assertAllEqual(assigned,
                          constant_op.constant([1, 11, 1, 10, 9, 1, 1, 12]))
      self.assertAllEqual(added,
                          constant_op.constant([1, 12, 1, 11, 10, 1, 1, 13]))
      self.assertAllEqual(subbed,
                          constant_op.constant([1, -10, 1, -9, -8, 1, 1, -11]))

  def testUpdateAddSubGradients(self):
    # [FUNCTION DOC]
    # Scenario: numerically verify gradients of tensor_scatter update, add and sub, both with
    # respect to the target tensor and with respect to the updates (a lambda wraps each op for
    # the gradient checker). Expects theoretical and numerical Jacobians to agree within 5e-4.
    with self.cached_session():
      indices = constant_op.constant([[3], [1]])
      updates = constant_op.constant([9, 10], dtype=dtypes.float32)
      x = array_ops.ones([4], dtype=dtypes.float32)

      theoretical, numerical = gradient_checker_v2.compute_gradient(
          lambda x: array_ops.tensor_scatter_update(x, indices, updates), [x])
      self.assertAllClose(theoretical, numerical, 5e-4, 5e-4)
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          lambda x: array_ops.tensor_scatter_add(x, indices, updates), [x])
      self.assertAllClose(theoretical, numerical, 5e-4, 5e-4)
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          lambda x: array_ops.tensor_scatter_sub(x, indices, updates), [x])
      self.assertAllClose(theoretical, numerical, 5e-4, 5e-4)

      theoretical, numerical = gradient_checker_v2.compute_gradient(
          lambda updates: array_ops.tensor_scatter_update(x, indices, updates),
          [updates])
      self.assertAllClose(theoretical, numerical, 5e-4, 5e-4)
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          lambda updates: array_ops.tensor_scatter_add(x, indices, updates),
          [updates])
      self.assertAllClose(theoretical, numerical, 5e-4, 5e-4)
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          lambda updates: array_ops.tensor_scatter_sub(x, indices, updates),
          [updates])
      self.assertAllClose(theoretical, numerical, 5e-4, 5e-4)

  @test_util.run_in_graph_and_eager_modes
  def testUpdateMinMax(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager): for int32 and float32, apply tensor_scatter update, min and max
    # to eight ones with indices 4, 3, 1, 7 and values 0, 2, -1, 2. Expects overwrite, element-wise
    # minimum and element-wise maximum results at those positions.
    for dtype in (dtypes.int32, dtypes.float32):
      indices = constant_op.constant([[4], [3], [1], [7]])
      updates = constant_op.constant([0, 2, -1, 2], dtype=dtype)
      t = array_ops.ones([8], dtype=dtype)
      assigned = array_ops.tensor_scatter_update(t, indices, updates)
      min_result = array_ops.tensor_scatter_min(t, indices, updates)
      max_result = array_ops.tensor_scatter_max(t, indices, updates)

      self.assertAllEqual(assigned,
                          constant_op.constant([1, -1, 1, 2, 0, 1, 1, 2]))
      self.assertAllEqual(min_result,
                          constant_op.constant([1, -1, 1, 1, 0, 1, 1, 1]))
      self.assertAllEqual(max_result,
                          constant_op.constant([1, 1, 1, 2, 1, 1, 1, 2]))

  def testUpdateMinMaxGradients(self):
    # [FUNCTION DOC]
    # Scenario: checks gradients of tensor_scatter_max and tensor_scatter_min with respect to the
    # target tensor and to the updates, for int32 and int64 index types, using a nested helper
    # run inside a cached session. Because the derivative is discontinuous where values tie,
    # theoretical Jacobians are compared with manually written ones (ties share the gradient
    # equally, about one third each) within 5e-4.

    # Loop body as a function to avoid go/gpylint-faq#cell-var-from-loop.
    def _TestFn(dtype):
      # [FUNCTION DOC]
      # HELPER: loop body of the min/max gradient test. Takes the index dtype (int32 or int64).
      # Builds a tensor of four ones, indices 1, 2, 3, 3 and updates 2.0, 0.5, 1.0, 1.0; computes
      # theoretical gradients of scatter max and min with respect to the tensor and the updates and
      # asserts they match hand-computed Jacobians within 5e-4. Returns nothing.
      x = array_ops.ones([4], dtype=dtypes.float32)
      indices = constant_op.constant([[1], [2], [3], [3]], dtype=dtype)
      updates = constant_op.constant([2.0, 0.5, 1.0, 1.0], dtype=dtypes.float32)

      theoretical, _ = gradient_checker_v2.compute_gradient(
          lambda x: array_ops.tensor_scatter_max(x, indices, updates), [x])
      # Numerical gradient doesn't work for degenerate values because the
      # derivative is not continuous. The manually entered gradient divides
      # the gradient among all contributing elements at the discontinuity.
      manual = array_ops.reshape(
          array_ops.matrix_diag([1.0, 0.0, 1.0, 0.3333]), (1, 4, 4))
      self.assertAllClose(theoretical, manual, 5e-4, 5e-4)

      theoretical, _ = gradient_checker_v2.compute_gradient(
          lambda x: array_ops.tensor_scatter_min(x, indices, updates), [x])
      manual = array_ops.reshape(
          array_ops.matrix_diag([1.0, 1.0, 0.0, 0.3333]), (1, 4, 4))
      self.assertAllClose(theoretical, manual, 5e-4, 5e-4)

      theoretical, _ = gradient_checker_v2.compute_gradient(
          lambda updates: array_ops.tensor_scatter_max(x, indices, updates),
          [updates])
      manual = constant_op.constant(
          [[[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.3333, 0.3333]]],
          dtype=dtypes.float32)
      self.assertAllClose(theoretical, manual, 5e-4, 5e-4)

      theoretical, _ = gradient_checker_v2.compute_gradient(
          lambda updates: array_ops.tensor_scatter_min(x, indices, updates),
          [updates])
      manual = constant_op.constant(
          [[[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.3333, 0.3333]]],
          dtype=dtypes.float32)
      self.assertAllClose(theoretical, manual, 5e-4, 5e-4)

    with self.cached_session():
      for dtype in (dtypes.int32, dtypes.int64):
        _TestFn(dtype)

  def testTensorScatterUpdateWithForwarding(self):
    # [FUNCTION DOC]
    # Scenario: for int32 and float32, call tensor_scatter_update inside a compiled function on a
    # freshly created tensor (which allows input buffer forwarding). Expects the returned tensor
    # to have values 11, 10, 9, 12 at indices 1, 3, 4, 7 and ones elsewhere.
    for dtype in (dtypes.int32, dtypes.float32):

      @def_function.function
      def _TestFn():
        # [FUNCTION DOC]
        # HELPER: compiled function body for the forwarding test. Takes no arguments, builds the
        # ones tensor, indices and updates for the captured loop dtype, and returns the result of
        # tensor_scatter_update. No side effects.
        indices = constant_op.constant([[4], [3], [1], [7]])
        updates = constant_op.constant([9, 10, 11, 12], dtype=dtype)  # pylint: disable=cell-var-from-loop
        t = array_ops.ones([8], dtype=dtype)  # pylint: disable=cell-var-from-loop

        return array_ops.tensor_scatter_update(t, indices, updates)

      self.assertAllEqual(_TestFn(), [1, 11, 1, 10, 9, 1, 1, 12])

  @test_util.run_in_graph_and_eager_modes
  def testTensorScatterUpdateWithStrings(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager): tensor_scatter_update on a string tensor of eight "hello"
    # values with indices 4, 3, 1, 7 and new strings. Expects those positions to hold the new
    # strings and the others to remain "hello".
    indices = constant_op.constant([[4], [3], [1], [7]])
    updates = constant_op.constant(["there", "there", "there", "12"],
                                   dtype=dtypes.string)
    tensor = constant_op.constant([
        "hello", "hello", "hello", "hello", "hello", "hello", "hello", "hello"
    ],
                                  dtype=dtypes.string)
    updated = array_ops.tensor_scatter_update(tensor, indices, updates)

    self.assertAllEqual(
        updated,
        constant_op.constant([
            "hello", "there", "hello", "there", "there", "hello", "hello", "12"
        ]))

  @test_util.run_in_graph_and_eager_modes
  def testUpdateRepeatedIndices1D(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager; skipped when a GPU is available since duplicate-index updates are
    # non-deterministic there): update index five twice with values 4 and 8 in a 10x1 zero tensor.
    # Expects the last update (8) to win on CPU.
    if test_util.is_gpu_available():
      self.skipTest("Duplicate indices scatter is non-deterministic on GPU")
    a = array_ops.zeros([10, 1])
    b = array_ops.tensor_scatter_update(a, [[5], [5]], [[4], [8]])
    self.assertAllEqual(
        b,
        constant_op.constant([[0.], [0.], [0.], [0.], [0.], [8.], [0.], [0.],
                              [0.], [0.]]))

  @test_util.run_in_graph_and_eager_modes
  def testUpdateRepeatedIndices2D(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager; skipped when a GPU is available): update row six twice (with
    # ranges 11-20 then 10-19) and row five once in a 10x10 zero tensor. Expects row six to hold
    # the last update, the values 10 through 19.
    if test_util.is_gpu_available():
      self.skipTest("Duplicate indices scatter is non-deterministic on GPU")
    a = array_ops.zeros([10, 10])
    b = array_ops.tensor_scatter_update(
        a, [[5], [6], [6]],
        [math_ops.range(10),
         math_ops.range(11, 21),
         math_ops.range(10, 20)])
    self.assertAllEqual(
        b[6],
        constant_op.constant([10., 11., 12., 13., 14., 15., 16., 17., 18.,
                              19.]))


class ScatterNdTensorDeterminismTest(ScatterNdTensorTest):
  # [CLASS DOC]
  # Reruns all tensor-scatter tests with op determinism enabled around each test and adds a
  # repeatability test for tensor_scatter_update with many duplicate indices.

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the parent setup, then enables global op determinism (global state change undone in
    # tear down). Takes and returns nothing.
    super().setUp()
    config.enable_op_determinism()

  def tearDown(self):
    # [FUNCTION DOC]
    # Calls the parent tear down, then disables global op determinism to restore the default.
    # Takes and returns nothing.
    super().tearDown()
    config.disable_op_determinism()

  def testDeterminism(self):
    # [FUNCTION DOC]
    # Scenario: tensor_scatter_update of 100000 random values all at index zero of a one-element
    # tensor, repeated six times. Expects every run to return exactly the same value as the first.
    a = array_ops.zeros([1])
    indices = array_ops.zeros([100000, 1], dtypes.int32)
    values = np.random.randn(100000)
    val = self.evaluate(array_ops.tensor_scatter_update(a, indices, values))
    for _ in range(5):
      val2 = self.evaluate(array_ops.tensor_scatter_update(a, indices, values))
      self.assertAllEqual(val, val2)


if __name__ == "__main__":
  test.main()
