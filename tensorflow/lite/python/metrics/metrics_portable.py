# [MODULE DOC]
# Portable implementation of the TFLite metrics helper: every method is a no-op. Chooses the
# interface module from the tensorflow package or from the standalone tflite_runtime package
# depending on where this file is located. Helper module for code that needs a metrics object but
# has no monitoring backend.
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
"""Python TFLite metrics helper."""
import os
from typing import Optional, Text

# pylint: disable=g-import-not-at-top
if not os.path.splitext(__file__)[0].endswith(
    os.path.join('tflite_runtime', 'metrics_portable')):
  # This file is part of tensorflow package.
  from tensorflow.lite.python.metrics import metrics_interface  # type: ignore
else:
  # This file is part of tflite_runtime package.
  from tflite_runtime import metrics_interface  # type: ignore
# pylint: enable=g-import-not-at-top


class TFLiteMetrics(metrics_interface.TFLiteMetricsInterface):
  # [CLASS DOC]
  # No-op implementation of the metrics interface for environments without monitoring (for example
  # the standalone runtime package). Holds no state; every method does nothing.
  """TFLite metrics helper."""

  def __init__(self,
               model_hash: Optional[Text] = None,
               model_path: Optional[Text] = None) -> None:
    # [FUNCTION DOC]
    # Constructor; accepts an optional model hash and optional model path (both ignored, no
    # validation) and does nothing.
    pass

  def increase_counter_debugger_creation(self):
    # [FUNCTION DOC]
    # No-op.
    pass

  def increase_counter_interpreter_creation(self):
    # [FUNCTION DOC]
    # No-op.
    pass

  def increase_counter_converter_attempt(self):
    # [FUNCTION DOC]
    # No-op.
    pass

  def increase_counter_converter_success(self):
    # [FUNCTION DOC]
    # No-op.
    pass

  def set_converter_param(self, name, value):
    # [FUNCTION DOC]
    # No-op; name and value are ignored.
    pass

  def set_converter_error(self, error_data):
    # [FUNCTION DOC]
    # No-op; the error data is ignored.
    pass

  def set_converter_latency(self, value):
    # [FUNCTION DOC]
    # No-op; the latency value is ignored.
    pass


class TFLiteConverterMetrics(TFLiteMetrics):
  # [CLASS DOC]
  # Converter specialization of the no-op metrics; adds export related operations that also do
  # nothing, so converter code can call them regardless of environment.
  """Similar to TFLiteMetrics but specialized for converter."""

  def __del__(self):
    # [FUNCTION DOC]
    # Destructor; no-op (the non-portable version exports metrics here).
    pass

  def set_export_required(self):
    # [FUNCTION DOC]
    # No-op.
    pass

  def export_metrics(self):
    # [FUNCTION DOC]
    # No-op.
    pass
