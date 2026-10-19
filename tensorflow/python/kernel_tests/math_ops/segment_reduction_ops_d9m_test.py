# [MODULE DOC]
# Tests that segment reduction ops (sorted and unsorted), indexed-slices to tensor conversion and
# gather backpropagation run on the GPU when deterministic op execution is enabled. Run as a
# script, the main block turns on op determinism before starting the test runner. Tests are
# CUDA-only and otherwise skipped.
#
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for deterministic functionality of segment reduction ops."""

from tensorflow.python.eager import backprop
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class SegmentReductionDeterminismExceptionsTest(test.TestCase):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring says unimplemented-error exceptions are checked, but no test here
  # asserts any exception; they only verify that the ops run without raising. The error message
  # constant is never used.
  # GPU-oriented test case exercising segment reduction ops, indexed-slices conversion and gather
  # backprop with deterministic op execution enabled (the main block enables it). Shared fixture
  # is the input-building helper and a class constant holding an expected error message.
  """Test d9m-unimplemented exceptions from the segment reduction ops.

  Test that tf.errors.UnimplementedError is thrown or not thrown, as
  appropriate, by the GPU code-paths for segment reduction ops when
  deterministic ops are enabled.

  This test assumes that the base op test runs all the same test cases when
  deterministic ops are not enabled and will therefore detect erroneous
  exception throwing in those cases.
  """

  def _input(self, data_type, segment_ids_type):
    # [FUNCTION DOC]
    # HELPER: builds the shared test inputs. Inputs: the element type for the data and the element
    # type for the segment ids. Returns a triple: a 2x4 constant of the numbers one to eight in the
    # data type, a constant segment id vector of zero and one, and the number of segments (two).
    data = constant_op.constant([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=data_type)
    segment_ids = constant_op.constant([0, 1], dtype=segment_ids_type)
    num_segments = 2
    return data, segment_ids, num_segments

  @test_util.run_cuda_only
  def testSortedOps(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the mapping saying which ops should throw for float is built but never used, and
    # no exception is asserted; the test only checks that evaluation does not raise.
    # GPU-only test. Runs each sorted segment op (max, min, prod, sum) on float16, float32 and
    # float64 data with both segment id types on the GPU and evaluates the result.
    op_should_throw_for_float = {
        math_ops.segment_max: False,
        math_ops.segment_min: False,
        math_ops.segment_prod: True,
        math_ops.segment_sum: True,
    }
    for op, should_throw_for_float in op_should_throw_for_float.items():
      for segment_ids_type in [dtypes.int32, dtypes.int64]:
        for data_type in [dtypes.float16, dtypes.float32, dtypes.float64]:
          with self.cached_session(force_gpu=True):
            data, segment_ids, _ = self._input(data_type, segment_ids_type)
            result = op(data, segment_ids)
            self.evaluate(result)

  _UNSORTED_ERROR_MESSAGE = ("Deterministic GPU implementation of unsorted " +
                             "segment reduction op not available.")

  @test_util.run_cuda_only
  @test_util.run_in_graph_and_eager_modes
  def testUnsortedOps(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the mapping saying which ops should throw for float is built but never used, and
    # no exception is asserted; the test only checks that evaluation does not raise.
    # GPU-only test, run in graph and eager modes. Runs each unsorted segment op (max, min, mean,
    # sqrt-n, prod, sum) for several float and int32 types and both segment id types, skipping
    # sqrt-n with int32 data, and evaluates the result on the GPU.
    op_should_throw_for_float = {
        math_ops.unsorted_segment_max: False,
        math_ops.unsorted_segment_min: False,
        math_ops.unsorted_segment_mean: True,  # uses unsorted_segment_sum
        math_ops.unsorted_segment_sqrt_n: True,  # uses unsorted_segment_sum
        math_ops.unsorted_segment_prod: True,
        math_ops.unsorted_segment_sum: True,
    }
    with self.session(force_gpu=True):
      for op, should_throw_for_float in op_should_throw_for_float.items():
        for segment_ids_type in [dtypes.int32, dtypes.int64]:
          for data_type in [
              dtypes.float16, dtypes.float32, dtypes.float64, dtypes.int32
          ]:
            if (op == math_ops.unsorted_segment_sqrt_n and
                data_type == dtypes.int32):  # sqrt_n doesn't support int32
              continue
            data, segment_ids, num_segments = self._input(
                data_type, segment_ids_type)
            result = op(data, segment_ids, num_segments)
            self.evaluate(result)

  @test_util.run_cuda_only
  def testUnsortedOpsComplex(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite the class description, no exception is asserted either way.
    # GPU-only test. Runs the unsorted segment sum on complex64 and complex128 data with both
    # segment id types and evaluates the result, expecting no exception.
    for op in [
        math_ops.unsorted_segment_sum,
    ]:
      for data_type in [dtypes.complex64, dtypes.complex128]:
        for segment_ids_type in [dtypes.int32, dtypes.int64]:
          with self.cached_session(force_gpu=True):
            data, segment_ids, num_segments = self._input(
                data_type, segment_ids_type)
            result = op(data, segment_ids, num_segments)
            self.evaluate(result)

  @test_util.run_cuda_only
  @test_util.run_in_graph_and_eager_modes
  def testConvertToTensor(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite the class description, no exception is asserted either way.
    # GPU-only test, run in graph and eager modes. For floating types (and complex types unless
    # built for ROCm) and both segment id types, builds an indexed-slices object from the test
    # values and ids and converts it to a dense tensor, then evaluates it. Passes if no exception
    # is raised.
    with self.session(force_gpu=True):
      dtypes_to_test = [dtypes.float16, dtypes.float32, dtypes.float64]
      if not test.is_built_with_rocm():
        dtypes_to_test += [dtypes.complex64, dtypes.complex128]
      for data_type in dtypes_to_test:
        for segment_ids_type in [dtypes.int32, dtypes.int64]:
          values, indices, _ = self._input(data_type, segment_ids_type)
          sparse_value = indexed_slices.IndexedSlices(
              values, indices, dense_shape=values.shape)
          result = ops.convert_to_tensor(sparse_value)
          self.evaluate(result)

  @test_util.run_cuda_only
  def testGatherBackprop(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite the class description, no exception is asserted either way.
    # GPU-only test. For each floating type (plus complex types unless built for ROCm) and each
    # segment id type, wraps the data in a variable, gathers rows by the id tensor under a gradient
    # tape, computes the gradient (a scatter-add style backprop) and assigns it back to the
    # variable, evaluating the result. Passes if no exception is raised.
    dtypes_to_test = [dtypes.float16, dtypes.float32, dtypes.float64]
    if not test.is_built_with_rocm():
      dtypes_to_test += [dtypes.complex64, dtypes.complex128]
    for data_type in dtypes_to_test:
      for segment_ids_type in [dtypes.int32, dtypes.int64]:
        with self.cached_session(force_gpu=True):
          params, indices, _ = self._input(data_type, segment_ids_type)
          params = variables.Variable(params)
          with backprop.GradientTape() as tape:
            tape.watch(params)
            op_output = array_ops.gather(params, indices)
          gradient = tape.gradient(op_output, params)
          self.evaluate(params.assign(gradient))


if __name__ == "__main__":
  config.enable_op_determinism()
  test.main()
