# [MODULE DOC]
# Tests that the window functions of the signal ops (Hann, Hamming, Vorbis) convert to TFLite and
# that the converted model gives the same numeric output as running the TensorFlow function
# directly. Uses the shared signal test helper for conversion and interpretation.
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
"""Tests for window_ops."""

from absl.testing import parameterized
import numpy as np

from tensorflow.lite.python.kernel_tests.signal import test_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.ops.signal import window_ops
from tensorflow.python.platform import test


@tf_test_util.run_all_in_graph_and_eager_modes
class WindowOpsTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case (run in both graph and eager modes) checking TFLite conversion of
  # signal window functions. No shared fixtures.

  @parameterized.parameters(
      # Only float32 is supported.
      (window_ops.hann_window, 10, False, dtypes.float32),
      (window_ops.hann_window, 10, True, dtypes.float32),
      (window_ops.hamming_window, 10, False, dtypes.float32),
      (window_ops.hamming_window, 10, True, dtypes.float32),
      (window_ops.vorbis_window, 12, None, dtypes.float32),
  )
  def test_tflite_convert(self, window_fn, window_length, periodic, dtype):
    # [FUNCTION DOC]
    # Parameters come from the decorator: a window function, an integer window length, a periodic
    # flag (nothing for the Vorbis window) and a data type (float32 only). Converts a function
    # computing the window from a scalar integer length to TFLite, runs the model with the length,
    # computes the expected window with TensorFlow, and asserts they are close (tolerance one
    # millionth).

    def fn(window_length):
      # [FUNCTION DOC]
      # Test helper: computes the window for the given length tensor, passing the periodic flag; if
      # the window function rejects that argument (type error), calls it without the flag.
      try:
        return window_fn(window_length, periodic=periodic, dtype=dtype)
      except TypeError:
        return window_fn(window_length, dtype=dtype)

    tflite_model = test_util.tflite_convert(
        fn, [tensor_spec.TensorSpec(shape=[], dtype=dtypes.int32)]
    )
    window_length = np.array(window_length).astype(np.int32)
    (actual_output,) = test_util.evaluate_tflite_model(
        tflite_model, [window_length]
    )

    expected_output = self.evaluate(fn(window_length))
    self.assertAllClose(actual_output, expected_output, rtol=1e-6, atol=1e-6)


if __name__ == '__main__':
  test.main()
