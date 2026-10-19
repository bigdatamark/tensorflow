# [MODULE DOC]
# Smoke tests for the portable metrics helper. Each test creates the metrics object and calls one
# reporting method, expecting no failure. Note it imports the metrics module that is selected by
# the build, not explicitly the portable one.
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
"""TensorFlow Lite Python metrics helpr TFLiteMetrics check."""
from tensorflow.lite.python.metrics import metrics
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class MetricsPortableTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Smoke tests that the metrics object can be created and its reporting methods called without
  # error. No assertions beyond not raising.

  def test_TFLiteMetrics_creation_success(self):
    # [FUNCTION DOC]
    # Creates the metrics object with no arguments; expects no error.
    metrics.TFLiteMetrics()

  def test_debugger_creation_counter_increase_success(self):
    # [FUNCTION DOC]
    # Creates the metrics object and increments the debugger creation counter; expects no error.
    stub = metrics.TFLiteMetrics()
    stub.increase_counter_debugger_creation()

  def test_interpreter_creation_counter_increase_success(self):
    # [FUNCTION DOC]
    # Creates the metrics object and increments the interpreter creation counter; expects no error.
    stub = metrics.TFLiteMetrics()
    stub.increase_counter_interpreter_creation()

  def test_converter_attempt_counter_increase_success(self):
    # [FUNCTION DOC]
    # Creates the metrics object and increments the conversion attempt counter; expects no error.
    stub = metrics.TFLiteMetrics()
    stub.increase_counter_converter_attempt()

  def test_converter_success_counter_increase_success(self):
    # [FUNCTION DOC]
    # Creates the metrics object and increments the conversion success counter; expects no error.
    stub = metrics.TFLiteMetrics()
    stub.increase_counter_converter_success()

  def test_converter_params_set_success(self):
    # [FUNCTION DOC]
    # Creates the metrics object and sets one converter parameter; expects no error.
    stub = metrics.TFLiteMetrics()
    stub.set_converter_param('name', 'value')


if __name__ == '__main__':
  test.main()
