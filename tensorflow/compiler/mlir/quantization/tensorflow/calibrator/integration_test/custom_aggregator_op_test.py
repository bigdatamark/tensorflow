# [MODULE DOC]
# Integration tests for the custom aggregator op (a calibration op that passes its input through
# while recording min, max and optionally a histogram). Runs in graph mode with a session.
# Imports the full framework (and the native wrapper) for its side effect of loading the op.
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
"""Tests for Custom Aggregator op."""

import tensorflow  # pylint: disable=unused-import

from tensorflow.compiler.mlir.quantization.stablehlo import quantization_config_pb2 as stablehlo_quant_config_pb2
from tensorflow.compiler.mlir.quantization.tensorflow.calibrator import custom_aggregator_op_wrapper
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

_CalibrationMethod = (
    stablehlo_quant_config_pb2.CalibrationOptions.CalibrationMethod
)


class CustomAggregatorTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the custom aggregator op wrapper across min-max, average-min-max and histogram
  # calibration methods. Fixture: eager execution is disabled before each test.

  def setUp(self):
    # [FUNCTION DOC]
    # Runs the parent setup, then disables eager execution globally so graph sessions are used.
    super(CustomAggregatorTest, self).setUp()
    ops.disable_eager_execution()

  def testBypassAndMinMax(self):
    # [FUNCTION DOC]
    # Scenario: min-max aggregator on the constant 1..5. Expected: output equals the input,
    # min is 1, max is 5 and the histogram is empty.
    with self.session():
      input_tensor = array_ops.constant(
          [1.0, 2.0, 3.0, 4.0, 5.0], dtypes.float32
      )

      aggregator = custom_aggregator_op_wrapper.custom_aggregator(
          input_tensor,
          id='1',
          calibration_method=_CalibrationMethod.CALIBRATION_METHOD_MIN_MAX,
      )
      aggregator_output = self.evaluate(aggregator)
      self.assertAllEqual(aggregator_output.output, [1.0, 2.0, 3.0, 4.0, 5.0])
      self.assertEqual(aggregator_output.min, 1.0)
      self.assertEqual(aggregator_output.max, 5.0)
      self.assertEmpty(aggregator_output.histogram)

  def testTwoIdentities(self):
    # [FUNCTION DOC]
    # Scenario: two min-max aggregators with different ids (2 and 3) on 1..5 and its negation.
    # Expected: each passes its input through with its own min and max (1/5 and -5/-1) and an
    # empty histogram, showing independent statistics per id.
    with self.session():
      input_tensor1 = array_ops.constant(
          [1.0, 2.0, 3.0, 4.0, 5.0], dtypes.float32
      )
      aggregator1 = custom_aggregator_op_wrapper.custom_aggregator(
          input_tensor1,
          '2',
          calibration_method=_CalibrationMethod.CALIBRATION_METHOD_MIN_MAX,
      )
      aggregator1_output = self.evaluate(aggregator1)
      self.assertAllEqual(aggregator1_output.output, [1.0, 2.0, 3.0, 4.0, 5.0])
      self.assertEqual(aggregator1_output.min, 1.0)
      self.assertEqual(aggregator1_output.max, 5.0)
      self.assertEmpty(aggregator1_output.histogram)

      input_tensor2 = array_ops.constant(
          [-1.0, -2.0, -3.0, -4.0, -5.0], dtypes.float32
      )
      aggregator2 = custom_aggregator_op_wrapper.custom_aggregator(
          input_tensor2,
          '3',
          calibration_method=_CalibrationMethod.CALIBRATION_METHOD_MIN_MAX,
      )
      aggregator2_output = self.evaluate(aggregator2)
      self.assertAllEqual(
          aggregator2_output.output, [-1.0, -2.0, -3.0, -4.0, -5.0]
      )
      self.assertEqual(aggregator2_output.min, -5.0)
      self.assertEqual(aggregator2_output.max, -1.0)
      self.assertEmpty(aggregator2_output.histogram)

  def testBypassAndAverageMinMax(self):
    # [FUNCTION DOC]
    # Scenario: average-min-max aggregators, both using id 6, on two tensors (range -50..50 and
    # -100..100). Expected: outputs pass through; each reported min and max equals that call's own
    # extremes (-50/50 then -100/100); histograms are empty.
    # KNOWN ISSUE: the test name and shared id suggest checking averaging across calls, but the
    # asserts only check per-call min and max, not an average.
    with self.session():
      input_tensor1 = array_ops.constant(
          [-50.0, -25.0, 0.0, 25.0, 50.0], dtypes.float32
      )
      aggregator1 = custom_aggregator_op_wrapper.custom_aggregator(
          input_tensor1,
          '6',
          calibration_method=_CalibrationMethod.CALIBRATION_METHOD_AVERAGE_MIN_MAX,
      )
      aggregator1_output = self.evaluate(aggregator1)
      self.assertAllEqual(
          aggregator1_output.output,
          [-50.0, -25.0, 0.0, 25.0, 50.0],
      )
      self.assertEqual(aggregator1_output.min, -50.0)
      self.assertEqual(aggregator1_output.max, 50.0)
      self.assertEmpty(aggregator1_output.histogram)

      input_tensor2 = array_ops.constant(
          [-100.0, -50.0, 0.0, 50.0, 100.0], dtypes.float32
      )
      aggregator2 = custom_aggregator_op_wrapper.custom_aggregator(
          input_tensor2,
          '6',
          calibration_method=_CalibrationMethod.CALIBRATION_METHOD_AVERAGE_MIN_MAX,
      )
      aggregator2_output = self.evaluate(aggregator2)
      self.assertAllEqual(
          aggregator2_output.output, [-100.0, -50.0, 0.0, 50.0, 100.0]
      )
      self.assertEqual(aggregator2_output.min, -100.0)
      self.assertEqual(aggregator2_output.max, 100.0)
      self.assertEmpty(aggregator2_output.histogram)

  def testHistogramCalibration(self):
    # [FUNCTION DOC]
    # Scenario: MSE brute-force histogram aggregator with 512 bins on values 1, 1, 3, 4, 6.
    # Expected: output passes through; min 1, max 6; histogram has 512 entries summing to 5, with
    # 2 counts in bin 0 and one count each in bins 128, 192 and 320.
    with self.session():
      input_tensor = array_ops.constant(
          [1.0, 1.0, 3.0, 4.0, 6.0], dtypes.float32
      )

      aggregator = custom_aggregator_op_wrapper.custom_aggregator(
          input_tensor,
          id='7',
          calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE,
          num_bins=512,
      )
      aggregator_output = self.evaluate(aggregator)
      self.assertAllEqual(aggregator_output.output, [1.0, 1.0, 3.0, 4.0, 6.0])
      self.assertEqual(aggregator_output.min, 1.0)
      self.assertEqual(aggregator_output.max, 6.0)

      self.assertLen(aggregator_output.histogram, 512)
      self.assertEqual(sum(aggregator_output.histogram), 5)
      self.assertEqual(aggregator_output.histogram[0], 2)
      self.assertEqual(aggregator_output.histogram[128], 1)
      self.assertEqual(aggregator_output.histogram[192], 1)
      self.assertEqual(aggregator_output.histogram[320], 1)


if __name__ == '__main__':
  test.main()
