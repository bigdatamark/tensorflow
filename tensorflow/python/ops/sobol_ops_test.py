# [MODULE DOC]
# Support for sorting tensors: public sort and argsort along a constant axis in ascending or
# descending direction, implemented on top of top-k (with transposition for non-last axes and
# order reversal tricks for ascending). Defines a module-level table mapping direction names to
# implementations. No other import-time side effects.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests Sobol sequence generator."""
import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


class SobolSampleOpTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the Sobol low-discrepancy sequence sample op: known values, skipping, static and
  # dynamic output shapes, default dtype, and input validation errors. No shared fixtures.

  def test_basic(self):
    # [FUNCTION DOC]
    # Scenario: generate 4 samples of dimension 2 in float64 and float32. Expects the first four
    # known Sobol points (0.5,0.5), (0.75,0.25), (0.25,0.75), (0.375,0.375) within 0.001.
    for dtype in [np.float64, np.float32]:
      expected = np.array([[.5, .5], [.75, .25], [.25, .75], [.375, .375]])
      sample = self.evaluate(math_ops.sobol_sample(2, 4, dtype=dtype))
      self.assertAllClose(expected, sample, 0.001)

  def test_more_known_values(self):
    # [FUNCTION DOC]
    # Scenario: generate 31 samples of dimension 5 in both float types. Expects the evaluated
    # values to match a hard coded table of the first 31 five-dimensional Sobol points within
    # 0.001.
    for dtype in [np.float64, np.float32]:
      sample = math_ops.sobol_sample(5, 31, dtype=dtype)
      expected = [[0.50, 0.50, 0.50, 0.50, 0.50],
                  [0.75, 0.25, 0.25, 0.25, 0.75],
                  [0.25, 0.75, 0.75, 0.75, 0.25],
                  [0.375, 0.375, 0.625, 0.875, 0.375],
                  [0.875, 0.875, 0.125, 0.375, 0.875],
                  [0.625, 0.125, 0.875, 0.625, 0.625],
                  [0.125, 0.625, 0.375, 0.125, 0.125],
                  [0.1875, 0.3125, 0.9375, 0.4375, 0.5625],
                  [0.6875, 0.8125, 0.4375, 0.9375, 0.0625],
                  [0.9375, 0.0625, 0.6875, 0.1875, 0.3125],
                  [0.4375, 0.5625, 0.1875, 0.6875, 0.8125],
                  [0.3125, 0.1875, 0.3125, 0.5625, 0.9375],
                  [0.8125, 0.6875, 0.8125, 0.0625, 0.4375],
                  [0.5625, 0.4375, 0.0625, 0.8125, 0.1875],
                  [0.0625, 0.9375, 0.5625, 0.3125, 0.6875],
                  [0.09375, 0.46875, 0.46875, 0.65625, 0.28125],
                  [0.59375, 0.96875, 0.96875, 0.15625, 0.78125],
                  [0.84375, 0.21875, 0.21875, 0.90625, 0.53125],
                  [0.34375, 0.71875, 0.71875, 0.40625, 0.03125],
                  [0.46875, 0.09375, 0.84375, 0.28125, 0.15625],
                  [0.96875, 0.59375, 0.34375, 0.78125, 0.65625],
                  [0.71875, 0.34375, 0.59375, 0.03125, 0.90625],
                  [0.21875, 0.84375, 0.09375, 0.53125, 0.40625],
                  [0.15625, 0.15625, 0.53125, 0.84375, 0.84375],
                  [0.65625, 0.65625, 0.03125, 0.34375, 0.34375],
                  [0.90625, 0.40625, 0.78125, 0.59375, 0.09375],
                  [0.40625, 0.90625, 0.28125, 0.09375, 0.59375],
                  [0.28125, 0.28125, 0.15625, 0.21875, 0.71875],
                  [0.78125, 0.78125, 0.65625, 0.71875, 0.21875],
                  [0.53125, 0.03125, 0.40625, 0.46875, 0.46875],
                  [0.03125, 0.53125, 0.90625, 0.96875, 0.96875]]
      self.assertAllClose(expected, self.evaluate(sample), .001)

  def test_skip(self):
    # [FUNCTION DOC]
    # Scenario: for both float types, sample dimension 10 with 67 points and no skip, and sample
    # 50 points skipping the first 17. Expects the skipped sample to equal the rows of the
    # unskipped sample after the first 17.
    dim = 10
    n = 50
    skip = 17
    for dtype in [np.float64, np.float32]:
      sample_noskip = math_ops.sobol_sample(dim, n + skip, dtype=dtype)
      sample_skip = math_ops.sobol_sample(dim, n, skip, dtype=dtype)

      self.assertAllClose(
          self.evaluate(sample_noskip)[skip:, :], self.evaluate(sample_skip))

  def test_static_shape(self):
    # [FUNCTION DOC]
    # Scenario: sample 100 points of dimension 10 with constant arguments. Expects the statically
    # inferred shape to be 100 by 10.
    s = math_ops.sobol_sample(10, 100, dtype=np.float32)
    self.assertAllEqual([100, 10], s.shape.as_list())

  def test_static_shape_using_placeholder_for_dim(self):
    # [FUNCTION DOC]
    # Scenario: inside a traced function whose scalar int32 argument is the dimension, with 100
    # results fixed. Expects the traced static shape to be 100 by unknown, and the evaluated
    # result for dimension 10 to have shape 100 by 10.

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(shape=[], dtype=dtypes.int32)])
    def f(dim):
      # [FUNCTION DOC]
      # Test helper traced function: samples with the given dimension tensor and 100 results,
      # asserts the static shape is 100 by unknown, and returns the sample.
      s = math_ops.sobol_sample(dim, 100, dtype=dtypes.float32)
      assert s.shape.as_list() == [100, None]
      return s

    self.assertAllEqual([100, 10], self.evaluate(f(10)).shape)

  def test_static_shape_using_placeholder_for_num_results(self):
    # [FUNCTION DOC]
    # Scenario: inside a traced function whose scalar int32 argument is the number of results,
    # with dimension 10 fixed. Expects the traced static shape to be unknown by 10, and the
    # evaluated result for 100 results to have shape 100 by 10.

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(shape=[], dtype=dtypes.int32)])
    def f(num_results):
      # [FUNCTION DOC]
      # Test helper traced function: samples dimension 10 with the given result count tensor,
      # asserts the static shape is unknown by 10, and returns the sample.
      s = math_ops.sobol_sample(10, num_results, dtype=dtypes.float32)
      assert s.shape.as_list() == [None, 10]
      return s

    self.assertAllEqual([100, 10], self.evaluate(f(100)).shape)

  def test_static_shape_using_only_placeholders(self):
    # [FUNCTION DOC]
    # Scenario: a traced function taking both dimension and result count as scalar int32
    # arguments. Expects a fully unknown static shape inside the trace and a 100 by 10 evaluated
    # shape when called with 10 and 100.

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec(shape=[], dtype=dtypes.int32)] *
        2)
    def f(dim, num_results):
      # [FUNCTION DOC]
      # Test helper traced function: samples with both given tensors, asserts the static shape is
      # entirely unknown (two unknown dimensions), and returns the sample.
      s = math_ops.sobol_sample(dim, num_results, dtype=dtypes.float32)
      assert s.shape.as_list() == [None, None]
      return s

    self.assertAllEqual([100, 10], self.evaluate(f(10, 100)).shape)

  def test_dynamic_shape(self):
    # [FUNCTION DOC]
    # Scenario: sample 100 points of dimension 10 and evaluate. Expects the evaluated array
    # shape to be 100 by 10.
    s = math_ops.sobol_sample(10, 100, dtype=dtypes.float32)
    self.assertAllEqual([100, 10], self.evaluate(s).shape)

  def test_default_dtype(self):
    # [FUNCTION DOC]
    # Scenario: create the sample op without giving a dtype. Expects the output dtype to default
    # to float32.
    # Create an op without specifying the dtype. Dtype should be float32 in
    # this case.
    s = math_ops.sobol_sample(10, 100)
    self.assertEqual(dtypes.float32, s.dtype)

  @test_util.run_in_graph_and_eager_modes
  def test_non_scalar_input(self):
    # [FUNCTION DOC]
    # Scenario: call the raw sample op with a non-scalar result count and skip tensors. Expects
    # a value error or invalid-argument error saying the shape must be rank 0 or the argument
    # must be a scalar. Graph and eager modes.
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                r'Shape must be rank 0 but is rank 1|'
                                r'\w+ must be a scalar'):
      self.evaluate(gen_math_ops.sobol_sample(
          dim=7,
          num_results=constant_op.constant([1, 0]),
          skip=constant_op.constant([1])))

  @test_util.run_in_graph_and_eager_modes
  def test_dim_num_results_overflow(self):
    # [FUNCTION DOC]
    # Scenario: request dimension 2560 with 16384000 results, whose product exceeds the 32 bit
    # limit. Expects a value or invalid-argument error stating the product must be less than
    # 2147483647. Graph and eager modes.
    with self.assertRaisesRegex(
        (ValueError, errors.InvalidArgumentError),
        r'num_results\*dim must be less than 2147483647'):
      self.evaluate(
          gen_math_ops.sobol_sample(
              dim=2560, num_results=16384000, skip=0, dtype=dtypes.float32))

  @test_util.run_in_graph_and_eager_modes
  def test_num_results_skip_overflow(self):
    # [FUNCTION DOC]
    # Scenario: request one result with skip equal to 2147483647, so results plus skip overflows.
    # Expects a value or invalid-argument error stating the sum must be less than 2147483647.
    # Graph and eager modes.
    with self.assertRaisesRegex(
        (ValueError, errors.InvalidArgumentError),
        r'num_results\+skip must be less than 2147483647',
    ):
      self.evaluate(
          gen_math_ops.sobol_sample(
              dim=1, num_results=1, skip=2147483647, dtype=dtypes.float32
          )
      )


if __name__ == '__main__':
  googletest.main()
