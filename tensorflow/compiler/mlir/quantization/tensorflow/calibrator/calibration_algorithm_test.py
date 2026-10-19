# [MODULE DOC]
# Unit tests for the calibration algorithm module: checks that the public min/max function returns
# the expected range for the min-max, average-min-max and the four histogram-based methods.
# No import-time side effects besides test discovery; runs through the TensorFlow test runner.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for CalibrationAlgorithm."""
from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.mlir.quantization.stablehlo import quantization_config_pb2 as stablehlo_quant_config_pb2
from tensorflow.compiler.mlir.quantization.tensorflow.calibrator import calibration_algorithm
from tensorflow.compiler.mlir.quantization.tensorflow.calibrator import calibration_statistics_pb2 as calib_stats_pb2
from tensorflow.python.platform import test

_CalibrationMethod = (
    stablehlo_quant_config_pb2.CalibrationOptions.CalibrationMethod
)


class CalibrationAlgorithmTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case (with parameterized test support) covering the calibration algorithm entry point.
  # No shared fixtures; each test builds its own options and statistics messages.

  def test_min_max_max(self):
    # [FUNCTION DOC]
    # Scenario: min-max method with global min 1.0 and global max 5.0 in the statistics.
    # Action: call the public min/max function. Expected: the pair (1.0, 5.0).
    calib_opts = stablehlo_quant_config_pb2.CalibrationOptions(
        calibration_method=_CalibrationMethod.CALIBRATION_METHOD_MIN_MAX
    )
    statistics = calib_stats_pb2.CalibrationStatistics()
    statistics.min_max_statistics.global_min = 1.0
    statistics.min_max_statistics.global_max = 5.0

    min_value, max_value = calibration_algorithm.get_min_max_value(
        statistics, calib_opts
    )
    self.assertAllEqual((min_value, max_value), (1.0, 5.0))

  def test_average_min_max(self):
    # [FUNCTION DOC]
    # Scenario: average-min-max method with min sum 5, max sum 50 and 5 samples.
    # Action: call the public min/max function. Expected: the pair (1.0, 10.0).
    calib_opts = stablehlo_quant_config_pb2.CalibrationOptions(
        calibration_method=_CalibrationMethod.CALIBRATION_METHOD_AVERAGE_MIN_MAX
    )
    statistics = calib_stats_pb2.CalibrationStatistics()
    statistics.average_min_max_statistics.min_sum = 5.0
    statistics.average_min_max_statistics.max_sum = 50.0
    statistics.average_min_max_statistics.num_samples = 5

    min_value, max_value = calibration_algorithm.get_min_max_value(
        statistics, calib_opts
    )
    self.assertAllEqual((min_value, max_value), (1.0, 10.0))

  @parameterized.named_parameters(
      {
          "testcase_name": "with_histogram_percentile",
          "calibration_options": stablehlo_quant_config_pb2.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_PERCENTILE,
              calibration_parameters=stablehlo_quant_config_pb2.CalibrationOptions.CalibrationParameters(
                  min_percentile=0.001, max_percentile=99.999
              ),
          ),
      },
      {
          "testcase_name": "with_histogram_mse_bruteforce",
          "calibration_options": stablehlo_quant_config_pb2.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE,
              calibration_parameters=stablehlo_quant_config_pb2.CalibrationOptions.CalibrationParameters(),
          ),
      },
      {
          "testcase_name": "with_histogram_mse_max_frequency",
          "calibration_options": stablehlo_quant_config_pb2.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY,
              calibration_parameters=stablehlo_quant_config_pb2.CalibrationOptions.CalibrationParameters(),
          ),
      },
      {
          "testcase_name": "with_histogram_mse_symmetric",
          "calibration_options": stablehlo_quant_config_pb2.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_SYMMETRIC,
              calibration_parameters=stablehlo_quant_config_pb2.CalibrationOptions.CalibrationParameters(),
          ),
      },
  )
  def test_histogram_calibration_methods(self, calibration_options):
    # [FUNCTION DOC]
    # Parameterized over histogram percentile, MSE brute force, MSE max frequency and MSE
    # symmetric options (each given as calibration_options). Setup: a 501-bin histogram with
    # lower bound 0 and bin width 1; single outlier counts in the first and last bins and a
    # peak of 1000 at bin 250 decreasing by one per bin over 200 bins on each side.
    # Action: call the public min/max function. Expected: min within 49 to 51 and max within
    # 449 to 451, i.e. outliers are discarded.
    statistics = calib_stats_pb2.CalibrationStatistics()
    statistics.histogram_statistics.lower_bound = 0.0
    statistics.histogram_statistics.bin_width = 1.0

    hist_freq = np.zeros(501, dtype=np.int32)

    # Advanced calibration methods that use histograms detect outliers, so they
    # don't use the outliers as min/max values.
    hist_freq[0] = 1
    hist_freq[-1] = 1

    # The majority of the data exists around the center.
    hist_freq[250] = 1000
    for i in range(1, 201):
      hist_freq[250 - i] = 1000 - i
      hist_freq[250 + i] = 1000 - i

    statistics.histogram_statistics.hist_freq.extend(hist_freq.tolist())

    # Histogram calibration methods should remove outliers.
    min_value, max_value = calibration_algorithm.get_min_max_value(
        statistics, calibration_options
    )

    # Since the min/max values may differ slightly for each calibration method,
    # also check the nearby values.
    self.assertAllInRange(min_value, 49, 51)
    self.assertAllInRange(max_value, 449, 451)


if __name__ == "__main__":
  test.main()
