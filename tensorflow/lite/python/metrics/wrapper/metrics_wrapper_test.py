# [MODULE DOC]
# Tests for the metrics wrapper module: fetching collected errors when none exist, and checking
# that a failed conversion of an unsupported op reports a non-empty error list.
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
"""TFLite metrics_wrapper module test cases."""

import tensorflow as tf

from tensorflow.lite.python import lite
from tensorflow.lite.python.convert import ConverterError
from tensorflow.lite.python.metrics.wrapper import metrics_wrapper
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class MetricsWrapperTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the metrics wrapper error retrieval function. No shared fixtures.

  def test_basic_retrieve_collected_errors_empty(self):
    # [FUNCTION DOC]
    # Retrieves collected errors when no conversion has failed; asserts the list is empty.
    errors = metrics_wrapper.retrieve_collected_errors()
    self.assertEmpty(errors)

  def test_basic_retrieve_collected_errors_not_empty(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: if conversion unexpectedly succeeds, the captured errors variable is never set
    # and the assertion raises a name error rather than a clean failure.
    # Converts a cosh function with the default converter, which is expected to fail with a
    # converter error; takes the errors attached to that exception and asserts they are non-empty.

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float32)])
    def func(x):
      # [FUNCTION DOC]
      # Test helper: returns hyperbolic cosine of the input (an op needing Select TF ops).
      return tf.cosh(x)

    converter = lite.TFLiteConverterV2.from_concrete_functions(
        [func.get_concrete_function()], func)
    try:
      converter.convert()
    except ConverterError as err:
      # retrieve_collected_errors is already captured in err.errors
      captured_errors = err.errors
    self.assertNotEmpty(captured_errors)


if __name__ == "__main__":
  test.main()
