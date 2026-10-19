# [MODULE DOC]
# Unit tests for the n-ary element-wise addition op (add_n), covering many dtypes, input counts
# from one to ten, tensors of unknown static shape, and variant-typed tensors.
# Contains a single test class and a test-runner entry point; no import-time side effects.
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
"""Tests for aggregate_ops."""

import numpy as np

from tensorflow.core.framework import tensor_pb2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test


class AddNTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the add_n op. Holds a class constant giving the largest number of summands tried
  # (ten), chosen so every special-cased small input count and one full loop iteration of the
  # kernel are exercised. Provides helpers for the dtype list and random data generation.
  # AddN special-cases adding the first M inputs to make (N - M) divisible by 8,
  # after which it adds the remaining (N - M) tensors 8 at a time in a loop.
  # Test N in [1, 10] so we check each special-case from 1 to 9 and one
  # iteration of the loop.
  _MAX_N = 10

  def _supported_types(self):
    # [FUNCTION DOC]
    # HELPER: provides the dtype list for the add_n tests.
    # Takes no inputs. Returns a list of tensor dtypes to test: when a GPU is available a smaller
    # list (half, bfloat16, float32, float64, complex64, complex128, int64), otherwise the CPU
    # list which additionally includes int8, int16 and int32. No side effects beyond querying
    # GPU availability.
    if test.is_gpu_available():
      return [
          dtypes.float16, dtypes.bfloat16, dtypes.float32, dtypes.float64,
          dtypes.complex64, dtypes.complex128, dtypes.int64
      ]
    return [
        dtypes.int8,
        dtypes.int16,
        dtypes.int32,
        dtypes.int64,
        dtypes.bfloat16,
        dtypes.float16,
        dtypes.float32,
        dtypes.float64,
        dtypes.complex64,
        dtypes.complex128,
    ]

  def _buildData(self, shape, dtype):
    # [FUNCTION DOC]
    # HELPER: generates random input arrays for the add_n tests.
    # Inputs: shape (sequence of dimension sizes) and dtype (a TensorFlow dtype). Returns a numpy
    # array of standard-normal random values cast to the dtype's numpy type; for complex dtypes
    # the imaginary part is ten times the real part so values are distinguishable. Consumes
    # numpy global random state.
    data = np.random.randn(*shape).astype(dtype.as_numpy_dtype)
    # For complex types, add an index-dependent imaginary component so we can
    # tell we got the right value.
    if dtype.is_complex:
      return data + 10j * data
    return data

  def testAddN(self):
    # [FUNCTION DOC]
    # Test: with a fixed numpy seed, for every supported dtype and every input count from one to
    # the maximum, builds random 2x2 arrays, sums them with add_n, and compares against a numpy
    # sum over the stacked inputs. Asserts closeness with dtype-aware tolerances (looser for
    # half precision). Runs inside a session.
    np.random.seed(12345)
    with self.session():
      for dtype in self._supported_types():
        for count in range(1, self._MAX_N + 1):
          data = [self._buildData((2, 2), dtype) for _ in range(count)]
          actual = self.evaluate(math_ops.add_n(data))
          expected = np.sum(np.vstack(
              [np.expand_dims(d, 0) for d in data]), axis=0)
          self.assertAllCloseAccordingToType(
              expected,
              actual,
              float_rtol=5e-6,
              float_atol=5e-6,
              half_rtol=5e-3,
              half_atol=5e-3,
          )

  @test_util.run_deprecated_v1
  def testUnknownShapes(self):
    # [FUNCTION DOC]
    # Test (graph mode only, deprecated-v1 decorator): for each dtype and input count, feeds the
    # same array through an unshaped placeholder repeated count times into add_n and runs it in a
    # session. Asserts the result is close to the numpy sum of count copies, with dtype-aware
    # tolerances. Verifies add_n works when static shapes are unknown.
    np.random.seed(12345)
    with self.session() as sess:
      for dtype in self._supported_types():
        data = self._buildData((2, 2), dtype)
        for count in range(1, self._MAX_N + 1):
          data_ph = array_ops.placeholder(dtype=dtype)
          actual = sess.run(math_ops.add_n([data_ph] * count), {data_ph: data})
          expected = np.sum(np.vstack([np.expand_dims(data, 0)] * count),
                            axis=0)
          self.assertAllCloseAccordingToType(
              expected,
              actual,
              half_rtol=5e-3,
              half_atol=5e-3,
          )

  @test_util.run_deprecated_v1
  def testVariant(self):
    # [FUNCTION DOC]
    # Test (graph mode only): checks add_n over scalar variant tensors holding integers. Builds
    # 100 constant variants, then 127 random subsets of 1 to 28 of them, sums each subset with
    # add_n, converts the results to strings, and asserts each equals the variant debug string
    # for the numpy integer sum. Runs on CPU only because variant copies between CPU and GPU
    # are unsupported.

    def create_constant_variant(value):
      # [FUNCTION DOC]
      # HELPER: builds a scalar variant constant for the variant add_n test.
      # Input: an integer value. Returns a constant tensor of variant dtype whose serialized
      # payload has type name "int" (matching the C++ variant registration) and metadata equal
      # to the 32-bit integer's raw bytes.
      return constant_op.constant(
          tensor_pb2.TensorProto(
              dtype=dtypes.variant.as_datatype_enum,
              tensor_shape=tensor_shape.TensorShape([]).as_proto(),
              variant_val=[
                  tensor_pb2.VariantTensorDataProto(
                      # Match registration in variant_op_registry.cc
                      type_name=b"int",
                      metadata=np.array(value, dtype=np.int32).tobytes())
              ]))

    # TODO(ebrevdo): Re-enable use_gpu=True once non-DMA Variant
    # copying between CPU and GPU is supported.
    with self.session(use_gpu=False):
      num_tests = 127
      values = list(range(100))
      variant_consts = [create_constant_variant(x) for x in values]
      sum_count_indices = np.random.randint(1, 29, size=num_tests)
      sum_indices = [
          np.random.randint(100, size=count) for count in sum_count_indices]
      expected_sums = [np.sum(x) for x in sum_indices]
      variant_sums = [math_ops.add_n([variant_consts[i] for i in x])
                      for x in sum_indices]

      # We use as_string() to get the Variant DebugString for the
      # variant_sums; we know its value so we can check via string equality
      # here.
      #
      # Right now, non-numpy-compatible objects cannot be returned from a
      # session.run call; similarly, objects that can't be converted to
      # native numpy types cannot be passed to ops.convert_to_tensor.
      variant_sums_string = string_ops.as_string(variant_sums)
      self.assertAllEqual(
          variant_sums_string,
          ["Variant<type: int value: {}>".format(s).encode("utf-8")
           for s in expected_sums])


if __name__ == "__main__":
  test.main()
