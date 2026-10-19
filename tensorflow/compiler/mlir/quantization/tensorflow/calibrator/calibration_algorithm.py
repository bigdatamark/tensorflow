# [MODULE DOC]
# Calibration algorithms that turn collected calibration statistics into a (min, max) range used
# for quantization. Each algorithm is a private class registered, through a decorator, in a
# module-level registry keyed by the calibration method enum of the quantization options.
# Algorithms: global min/max, average min/max, histogram percentile, and three histogram
# mean-squared-error searches (brute force, expand from max frequency, expand from center).
# The only public entry point is the module-level function that looks up the algorithm for the
# configured method and runs it. Import-time side effect: fills the registry (asserting that no
# method is registered twice).
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
"""Defines CalibrationAlgorithm for calculating min and max values calculated by calibration method."""
import abc
import itertools
import logging

import numpy as np

from tensorflow.compiler.mlir.quantization.stablehlo import quantization_config_pb2 as stablehlo_quant_config_pb2
from tensorflow.compiler.mlir.quantization.tensorflow.calibrator import calibration_statistics_pb2 as calib_stats_pb2


_CalibrationMethod = (
    stablehlo_quant_config_pb2.CalibrationOptions.CalibrationMethod
)
_REGISTRY = {}


def _implements(calib_method: _CalibrationMethod):
  # [FUNCTION DOC]
  # HELPER: registers calibration algorithm classes in the module registry.
  # Class-decorator factory. Input: a calibration method enum value. Returns a decorator that
  # stores the decorated class in the registry under that method and returns the class unchanged.
  # The decorator asserts (assertion error) that the method is not already registered.
  def decorator(cls):
    # [FUNCTION DOC]
    # HELPER: inner decorator produced by the registration factory.
    # Input: an algorithm class. Registers it for the captured method and returns it unchanged.
    # Raises an assertion error if the method already has a registered class.
    assert calib_method not in _REGISTRY
    _REGISTRY[calib_method] = cls
    return cls

  return decorator


class _CalibrationAlgorithmBase(abc.ABC):
  # [CLASS DOC]
  # HELPER: abstract base shared by all calibration algorithms.
  # Holds the calibration statistics message and the calibration options message given at
  # construction. Subclasses implement the method that returns the (min, max) pair.
  """Abstract base class for calibration algorithm."""

  def __init__(
      self,
      statistics: calib_stats_pb2.CalibrationStatistics,
      calib_opts: stablehlo_quant_config_pb2.CalibrationOptions,
  ):
    # [FUNCTION DOC]
    # Stores the statistics (collected calibration statistics message) and calibration options
    # (message with method and parameters) on the instance. Returns nothing, no validation.
    self._statistics = statistics
    self._calib_opts = calib_opts

  @abc.abstractmethod
  def get_min_max_value(self) -> tuple[float, float]:
    # [FUNCTION DOC]
    # Abstract method: subclasses return a pair (min value, max value) of floats computed from
    # the statistics. This base body does nothing.
    pass


class _HistogramCalibrationAlgorithmBase(_CalibrationAlgorithmBase):
  # [CLASS DOC]
  # HELPER: shared base for the histogram-based calibration algorithms.
  # State: bin width, lower bound, per-bin frequencies (array), number of bins, number of bits
  # (fixed at 8) and the array of bin midpoints. Provides the quantization-error computation and
  # the range-expansion search used by subclasses.
  """Base class for histogram calibrators."""

  def __init__(
      self,
      statistics: calib_stats_pb2.CalibrationStatistics,
      calib_opts: stablehlo_quant_config_pb2.CalibrationOptions,
  ):
    # [FUNCTION DOC]
    # Calls the base constructor, then reads the histogram statistics: bin width, lower bound and
    # frequencies. Derives the bin count, fixes the quantization bit width at 8, and computes the
    # evenly spaced array of bin midpoints (lower bound plus half a bin, stepping by bin width).
    # Inputs: statistics message and calibration options message. Returns nothing.
    """Builds histogram using statistics.histogram_statistics.

    lower_bound                                    hist_mid
         v                                            v
         |=========|=========|=========|=========|=========|
                    bin width

    Args:
      statistics: Collected calibration statistics.
      calib_opts: Calibration options used for calculating min and max.
    """
    super().__init__(statistics, calib_opts)
    hist_stats = statistics.histogram_statistics
    self._bin_width = hist_stats.bin_width
    self._lower_bound = hist_stats.lower_bound
    self._hist_freq = np.array(hist_stats.hist_freq)
    self._num_bins = len(self._hist_freq)
    self._num_bits = 8
    # i-th bin has a range [bins[i], bins[i + 1]).
    # bins[i] = lower_bound + i * bin_width
    # bins[i + 1] = lower_bound + (i + 1) * bin_width
    # So hist_mids[i] = (lower_bound + bin_width / 2) + bin_width * i
    first_mid = self._lower_bound + self._bin_width / 2
    last_mid = first_mid + (self._num_bins - 1) * self._bin_width
    self._hist_mids = np.linspace(first_mid, last_mid, self._num_bins)

  def _get_dequantized_hist_mids_after_quantize(
      self, quant_min: float, quant_max: float
  ) -> np.ndarray:
    # [FUNCTION DOC]
    # HELPER: computes per-bin quantization round trip for the error search.
    # Simulates 8-bit asymmetric quantization over the range from quant_min to quant_max (floats):
    # scale is range divided by 255, zero point is minus quant_min over scale. Scale is floored
    # at 1e-9 in magnitude and zero point capped at 9e9 in magnitude, then zero point is rounded.
    # Bin midpoints are divided by scale, rounded, shifted by zero point, clipped to 0..255, and
    # dequantized. Returns the array of dequantized midpoints.
    # Note: scale is computed before the guard, so equal quant_min and quant_max divide by zero
    # (a numpy warning with infinite values, or a Python error for plain floats).
    """Quantizes and dequantizes hist_mids using quant_min and quant_max.

    Quantization converts the range of numbers from [quant_min, quant_max] to
    [0, 2^num_bits - 1]. Values less than quant_min are converted to 0, and
    values greater than quant_max are converted to 2^num_bits - 1.

    The histogram represents the distribution of the data, and our goal is to
    find the quant_min and quant_max that best describe this distribution. To do
    this, we quantize hist_mids using quant_min and quant_max and dequantize
    them again. Then the difference between hist_mids and dequantized hist_mids
    equates to quantization error when using quant_min and quant_max.


    Args:
      quant_min: The minimum real value that can be represented by a quantized
        value.
      quant_max: The maximum real value that can be represented by a quantized
        value.

    Returns:
      dequantized hist_mids after quantizing by quant_min and quant_max
    """
    maxbound = 2**self._num_bits - 1
    minbound = 0
    scale = (quant_max - quant_min) / maxbound
    zero_point = -quant_min / scale

    # Limit the range of zero_point and scale in case (quant_max - quant_min)
    # is unusually small.
    if abs(zero_point) > 9e9:
      zero_point = 9e9
    if abs(scale) < 1e-9:
      scale = 1e-9

    zero_point = round(zero_point)
    quantized_hist_mids = np.clip(
        np.round(self._hist_mids / scale) + zero_point, minbound, maxbound
    )
    dequantized_hist_mids = scale * (quantized_hist_mids - zero_point)
    return dequantized_hist_mids

  def _get_weighted_mean_squared_error(
      self, quant_min, quant_max
  ) -> tuple[float, float, float]:
    # [FUNCTION DOC]
    # HELPER: error metric for the histogram range searches.
    # KNOWN ISSUE: docstring says it returns a tuple of the error, but it actually returns a
    # three-element tuple (error, quant_min, quant_max), echoing the inputs for easy comparison.
    # Inputs: quant_min and quant_max (floats, candidate range). Quantizes and dequantizes bin
    # midpoints over that range, then returns the sum over bins of squared difference times the
    # bin frequency, together with the range. No side effects.
    """Gets mean squared error between hist_mids and dequantized hist_mids.

    Quantization converts the range of numbers from [quant_min, quant_max] to
    [0, 2^num_bits - 1]. Values less than quant_min are converted to 0, and
    values greater than quant_max are converted to 2^num_bits - 1.

    Args:
      quant_min: The minimum real value that can be represented by a quantized
        value.
      quant_max: The maximum real value that can be represented by a quantized
        value.

    Returns:
      (error, quant_min, quant_max): Tuple of weighted mean squared error.
      error = (hist_mids - dequantized_hist_mids)**2 * hist_freq
    """
    dequantized_hist_mids = self._get_dequantized_hist_mids_after_quantize(
        quant_min, quant_max
    )
    squared_error = (self._hist_mids - dequantized_hist_mids) ** 2
    weighted_error = np.sum(squared_error * self._hist_freq)
    return (weighted_error, quant_min, quant_max)

  def _get_min_max_value_by_expanding_range(
      self, start_idx: int
  ) -> tuple[float, float]:
    # [FUNCTION DOC]
    # HELPER: shared search used by the expanding-range histogram algorithms.
    # Input: start_idx, integer bin index where the range starts as a single bin. Alternately
    # widens the range one bin to the left, then right (skipping a side that hit the edge) until
    # it spans all bins, evaluating the weighted squared error each step and keeping the lowest
    # (ties broken by smaller min, then max). Returns (min, max) midpoints of the best range.
    # If there is only one bin (or the range already spans all) the loop never runs and the
    # returned values are infinity.
    """Starting from start_idx, expand left and right alternately to find the min value of mse loss.

    Args:
      start_idx: Index to start quantization.

    Returns:
      (min_value, max_value): Min and max calculated.
    """
    # Tuple of (mse_error, quant_min, quant_max).
    mse_min = (float('inf'), float('inf'), float('inf'))
    left, right = start_idx, start_idx

    # If this value is true, it moves left, otherwise it moves right.
    move_left = True
    while not (left == 0 and right == self._num_bins - 1):
      # Decrease left if right can't be moved or move_left is true.
      if (move_left and left > 0) or (right == self._num_bins - 1):
        left = max(left - 1, 0)
      # Else increase right.
      else:
        right = min(right + 1, self._num_bins - 1)
      # Toogle the move_left.
      move_left = not move_left
      quant_min, quant_max = self._hist_mids[left], self._hist_mids[right]
      mse_tuple = self._get_weighted_mean_squared_error(quant_min, quant_max)
      mse_min = min(mse_tuple, mse_min)
    # Extract (quant_min, quant_max) from (mse_error, quant_min, quant_max).
    min_value, max_value = mse_min[1], mse_min[2]
    return min_value, max_value


@_implements(_CalibrationMethod.CALIBRATION_METHOD_MIN_MAX)
class _MinMax(_CalibrationAlgorithmBase):
  # [CLASS DOC]
  # Registered for the min-max calibration method. Uses the recorded global min and global max.
  """MinMaxCalibrationAlgorithm for calculating min and max values of calibration result.

  MinMax calibration calculates the global min and global max values.

  global min = min of given sample inputs
  global max = max of given sample inputs
  """

  def get_min_max_value(self) -> tuple[float, float]:
    # [FUNCTION DOC]
    # Returns the pair (global min, global max) read directly from the min-max statistics.
    # No validation, no side effects.
    """Calculates the global min and max values.

    Returns:
      (min_value, max_value): Min and max calculated using MinMax
    """
    return (
        self._statistics.min_max_statistics.global_min,
        self._statistics.min_max_statistics.global_max,
    )


@_implements(_CalibrationMethod.CALIBRATION_METHOD_AVERAGE_MIN_MAX)
class _AverageMinMax(_CalibrationAlgorithmBase):
  # [CLASS DOC]
  # Registered for the average-min-max calibration method. Averages per-sample min and max.
  """AverageMinMaxCalibrationAlgorithm for calculating min and max values of calibration result.

  AverageMinMax calibration calculates the average of min and max values.
  average of min = sum of min values / number of samples
  average of max = sum of max values / number of samples
  """

  def get_min_max_value(self) -> tuple[float, float]:
    # [FUNCTION DOC]
    # Returns (sum of mins / sample count, sum of maxes / sample count) from the average
    # min-max statistics. Raises a value error (mentioning the options) if the sample count is 0.
    """Calculates the average of min and max values.

    Returns:
      (min_value, max_value): Min and max calculated using AverageMinMax

    Raises:
      ValueError: num_samples is 0.
    """
    average_min_max_statistics = self._statistics.average_min_max_statistics
    # num_samples is guaranteed to be larger than 0 because
    # get_statistics_from_calibrator throws an exception if num_samples == 0.
    num_samples = average_min_max_statistics.num_samples
    if num_samples == 0:
      raise ValueError(
          'num_samples must not be 0 when calibration method is'
          f' AverageMinMax: {self._calib_opts}'
      )
    min_value, max_value = (
        average_min_max_statistics.min_sum / num_samples,
        average_min_max_statistics.max_sum / num_samples,
    )

    return min_value, max_value


@_implements(_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_PERCENTILE)
class _HistogramPercentile(_HistogramCalibrationAlgorithmBase):
  # [CLASS DOC]
  # Registered for the histogram-percentile method. Picks bin midpoints at configured percentiles.
  """HistogramPercentile for calculating min and max values of calibration result."""

  def get_min_max_value(self) -> tuple[float, float]:
    # [FUNCTION DOC]
    # Builds the normalized cumulative frequency of the histogram, converts the configured min and
    # max percentiles (0-100) to fractions, finds bin indices by sorted search (min: first bin whose
    # cumulative value exceeds the fraction; max: first bin reaching it) and returns the midpoints
    # of those bins. A histogram with zero total frequency yields not-a-number cumulative values;
    # an index beyond the last bin can raise an index error.
    """Calculates min and max from statistics using calibration options.

    A "percentile" is a statistical concept that represents the value below
    which a given percentage of data falls in a dataset. It involves sorting the
    data from smallest to largest and then finding the value at a specified
    percentage position. For example, the 0.01 percentile represents the value
    in a given data set that corresponds to the lowest 0.01% of the data.

    HistogramPercentile calibration uses min_percentile and max_percentile to
    find min and max.

    min_percentile and max_percentile must be in range [0, 100].
    min_percentile is 0.001 by default.
    max_percentile is 99.999 by default.

    Returns:
      (min_value, max_value): Min and max calculated using HistogramPercentile
    """
    total_freq = sum(self._hist_freq)
    # hist_freq_cumsum is dividing cumulative sum of hist_freq by total_freq
    # hist_freq_cumsum's value is in range [0, 1] by its definition
    hist_freq_cumsum = np.cumsum(self._hist_freq) / total_freq

    # min_percentile and max_percentile are converted from [0, 100] to [0, 1].
    min_quantile, max_quantile = (
        self._calib_opts.calibration_parameters.min_percentile / 100.0,
        self._calib_opts.calibration_parameters.max_percentile / 100.0,
    )

    # Get index of min/max quantile.
    min_quantile_idx, max_quantile_idx = (
        np.searchsorted(hist_freq_cumsum, min_quantile, side='right'),
        np.searchsorted(hist_freq_cumsum, max_quantile, side='left'),
    )

    # Get value of min/max quantile index.
    min_value, max_value = (
        self._hist_mids[min_quantile_idx],
        self._hist_mids[max_quantile_idx],
    )

    return min_value, max_value


@_implements(_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE)
class _HistogramMseBruteforce(_HistogramCalibrationAlgorithmBase):
  # [CLASS DOC]
  # Registered for the histogram MSE brute-force method. Tries every pair of bin midpoints.
  """HistogramMseBruteforce for calculating min and max values of calibration result."""

  def get_min_max_value(self) -> tuple[float, float]:
    # [FUNCTION DOC]
    # Logs a warning when there are more than 512 bins (quadratic cost). Evaluates the weighted
    # squared quantization error for every pair of distinct bins (left before right) and returns
    # the (min, max) midpoints with the lowest error. With fewer than two bins returns infinities.
    """Finds the optimal quant_min and quant_max by testing all possible cases.

    It guarantees optimal quant_min and quant_max for the representative
    dataset, but not for the test dataset.

    Returns:
      (min_value, max_value): Min and max calculated using
      HistogramMseBruteforce.
    """
    if self._num_bins > 512:
      logging.warning(
          'num_bins=%d is too large. The HISTOGRAM_MSE_BRUTEFORCE method tests'
          ' all histogram mid value pairs, so it may take a long time.',
          self._num_bins,
      )
    # Tuple of (mse_error, quant_min, quant_max).
    mse_min = (float('inf'), float('inf'), float('inf'))

    # Calculate the error for all hist_mid pairs.
    for left, right in itertools.combinations(range(self._num_bins), 2):
      quant_min, quant_max = self._hist_mids[left], self._hist_mids[right]
      mse_tuple = self._get_weighted_mean_squared_error(quant_min, quant_max)
      mse_min = min(mse_tuple, mse_min)
    min_value, max_value = mse_min[1], mse_min[2]

    return min_value, max_value


@_implements(_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY)
class _HistogramMseMaxFrequency(_HistogramCalibrationAlgorithmBase):
  # [CLASS DOC]
  # Registered for the histogram MSE max-frequency method. Expands outward from the fullest bin.
  """HistogramMseMaxFrequency for calculating min and max values of calibration result."""

  def get_min_max_value(self) -> tuple[float, float]:
    # [FUNCTION DOC]
    # Finds the index of the bin with the highest frequency and returns the result of the shared
    # range-expansion search started there: a (min, max) pair of bin midpoints.
    """Finds min and max starting from the index of the max frequency.

     The HistogramMseMaxFrequency method starts from the bin with the highest
     frequency and expands the range to both sides. This performs well when data
     is well spread on both sides of the max frequency.

    Returns:
      (min_value, max_value): Min and max calculated using method to expand the
      range based on max frequency.
    """
    # Find the index of max frequency.
    freq_max_idx = np.argmax(self._hist_freq)
    return self._get_min_max_value_by_expanding_range(freq_max_idx)


@_implements(_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_SYMMETRIC)
class _HistogramMseSymmetric(_HistogramCalibrationAlgorithmBase):
  # [CLASS DOC]
  # Registered for the histogram MSE symmetric method. Expands outward from the center bin.
  """HistogramMseSymmetric for calculating min and max values of calibration result."""

  def get_min_max_value(self) -> tuple[float, float]:
    # [FUNCTION DOC]
    # Runs the shared range-expansion search starting at the middle bin (bin count divided by two,
    # rounded down) and returns the (min, max) midpoints. Note that despite the name the resulting
    # range is not forced to be symmetric around zero.
    """Finds min and max starting from the center index.

    The HistogramMseSymmetric method starts from the center bin and expands the
    range to both sides. This works better when the data is well-centered.

    Returns:
      (min_value, max_value): Min and max calculated using the method starting
      from center and expanding.
    """

    # This function is currently only called in this method, but will be used in
    # other methods in the future.
    return self._get_min_max_value_by_expanding_range(self._num_bins // 2)


def get_min_max_value(
    statistics: calib_stats_pb2.CalibrationStatistics,
    calib_opts: stablehlo_quant_config_pb2.CalibrationOptions,
) -> tuple[float, float]:
  # [FUNCTION DOC]
  # Public entry point. Inputs: statistics (collected calibration statistics message) and
  # calib_opts (options message naming the calibration method and parameters). Looks up the
  # registered algorithm class for the method, builds it and returns its (min, max) pair.
  # Raises a value error if the method has no registered algorithm.
  """Calculates min and max from statistics using calibration options.

  Args:
    statistics: Collected calibration statistics.
    calib_opts: Calibration options used for calculating min and max.

  Returns:
    (min_value, max_value): Min and max calculated using calib_opts.

  Raises:
    ValueError: Unsupported calibration method is given.
  """
  calib_method = calib_opts.calibration_method
  if calib_method not in _REGISTRY:
    raise ValueError(f'Unsupported calibration method: {calib_method}')

  calibration_algorithm = _REGISTRY[calib_method](statistics, calib_opts)
  return calibration_algorithm.get_min_max_value()
