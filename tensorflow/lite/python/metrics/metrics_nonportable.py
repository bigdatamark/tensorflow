# [MODULE DOC]
# Non-portable (full TensorFlow) implementation of the TFLite metrics helper. At import time it
# creates monitoring counters and gauges (debugger creation, interpreter creation, conversion
# attempt/success, conversion parameters, conversion errors, conversion latency). Defines a metrics
# class that updates them and a converter specialization that exports collected metrics through the
# native metrics wrapper, once per instance, when exporting or when the object is destroyed.
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
from typing import Optional, Text
import uuid

from tensorflow.compiler.mlir.lite.metrics import converter_error_data_pb2
from tensorflow.lite.python.metrics import metrics_interface
from tensorflow.lite.python.metrics.wrapper import metrics_wrapper
from tensorflow.python.eager import monitoring

_counter_debugger_creation = monitoring.Counter(
    '/tensorflow/lite/quantization_debugger/created',
    'Counter for the number of debugger created.')

_counter_interpreter_creation = monitoring.Counter(
    '/tensorflow/lite/interpreter/created',
    'Counter for number of interpreter created in Python.', 'language')

# The following are conversion metrics. Attempt and success are kept separated
# instead of using a single metric with a label because the converter may
# raise exceptions if conversion failed. That may lead to cases when we are
# unable to capture the conversion attempt. Increasing attempt count at the
# beginning of conversion process and the success count at the end is more
# suitable in these cases.
_counter_conversion_attempt = monitoring.Counter(
    '/tensorflow/lite/convert/attempt',
    'Counter for number of conversion attempts.')

_counter_conversion_success = monitoring.Counter(
    '/tensorflow/lite/convert/success',
    'Counter for number of successful conversions.')

_gauge_conversion_params = monitoring.StringGauge(
    '/tensorflow/lite/convert/params',
    'Gauge for keeping conversion parameters.', 'name')

_gauge_conversion_errors = monitoring.StringGauge(
    '/tensorflow/lite/convert/errors',
    'Gauge for collecting conversion errors. The value represents the error '
    'message.', 'component', 'subcomponent', 'op_name', 'error_code')

_gauge_conversion_latency = monitoring.IntGauge(
    '/tensorflow/lite/convert/latency', 'Conversion latency in ms.')


class TFLiteMetrics(metrics_interface.TFLiteMetricsInterface):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring lists model_hash and model_path attributes, but the class stores no
  # attributes at all; the constructor only validates its arguments.
  # Metrics helper that reports to process-wide monitoring counters and gauges defined at module
  # level. Methods update those shared metrics; instances hold no state of their own.
  """TFLite metrics helper for prod (borg) environment.

  Attributes:
    model_hash: A string containing the hash of the model binary.
    model_path: A string containing the path of the model for debugging
      purposes.
  """

  def __init__(self,
               model_hash: Optional[Text] = None,
               model_path: Optional[Text] = None) -> None:
    # [FUNCTION DOC]
    # Constructor. Optional model_hash and model_path (text). Raises a value error unless both are
    # given or both are missing (empty counts as missing). Stores nothing; there is a placeholder
    # branch for a future service stub that currently does nothing.
    del self  # Temporarily removing self until parameter logic is implemented.
    if model_hash and not model_path or not model_hash and model_path:
      raise ValueError('Both model metadata(model_hash, model_path) should be '
                       'given at the same time.')
    if model_hash:
      # TODO(b/180400857): Create stub once the service is implemented.
      pass

  def increase_counter_debugger_creation(self):
    # [FUNCTION DOC]
    # Adds one to the global quantization debugger creation counter.
    _counter_debugger_creation.get_cell().increase_by(1)

  def increase_counter_interpreter_creation(self):
    # [FUNCTION DOC]
    # Adds one to the global interpreter creation counter, using the language label python.
    _counter_interpreter_creation.get_cell('python').increase_by(1)

  def increase_counter_converter_attempt(self):
    # [FUNCTION DOC]
    # Adds one to the global conversion attempt counter.
    _counter_conversion_attempt.get_cell().increase_by(1)

  def increase_counter_converter_success(self):
    # [FUNCTION DOC]
    # Adds one to the global successful conversion counter.
    _counter_conversion_success.get_cell().increase_by(1)

  def set_converter_param(self, name, value):
    # [FUNCTION DOC]
    # Sets the global conversion parameter gauge cell labeled by name to value (text); a later
    # call with the same name overwrites the earlier value.
    _gauge_conversion_params.get_cell(name).set(value)

  def set_converter_error(
      self, error_data: converter_error_data_pb2.ConverterErrorData):
    # [FUNCTION DOC]
    # Records a converter error in the global errors gauge. error_data is a converter error data
    # message. The cell is labeled by component, subcomponent, operator name and the symbolic name
    # of the error code, and its value is set to the error message text. Returns nothing.
    error_code_str = converter_error_data_pb2.ConverterErrorData.ErrorCode.Name(
        error_data.error_code)
    _gauge_conversion_errors.get_cell(
        error_data.component,
        error_data.subcomponent,
        error_data.operator.name,
        error_code_str,
    ).set(error_data.error_message)

  def set_converter_latency(self, value):
    # [FUNCTION DOC]
    # Sets the global conversion latency gauge to value (integer milliseconds).
    _gauge_conversion_latency.get_cell().set(value)


class TFLiteConverterMetrics(TFLiteMetrics):
  # [CLASS DOC]
  # Converter specialization of the metrics helper. Each instance owns a native metrics exporter
  # created with a fresh random session id, and an exported flag. Metrics are exported by an
  # explicit call or automatically when the object is destroyed if not yet exported.
  """Similar to TFLiteMetrics but specialized for converter.

  A unique session id will be created for each new TFLiteConverterMetrics.
  """

  def __init__(self) -> None:
    # [FUNCTION DOC]
    # Constructor, no arguments. Calls the parent constructor with no model metadata, generates a
    # random hexadecimal session id, creates the native metrics exporter for that session, and
    # marks metrics as not yet exported.
    super(TFLiteConverterMetrics, self).__init__()
    session_id = uuid.uuid4().hex
    self._metrics_exporter = metrics_wrapper.MetricsWrapper(session_id)
    self._exported = False

  def __del__(self):
    # [FUNCTION DOC]
    # Destructor. If metrics were not exported yet, exports them now. Note that if the constructor
    # failed before the flag was set, this could raise an attribute error during cleanup.
    if not self._exported:
      self.export_metrics()

  def set_export_required(self):
    # [FUNCTION DOC]
    # Marks metrics as not yet exported so a later export or destruction exports them again.
    self._exported = False

  def export_metrics(self):
    # [FUNCTION DOC]
    # Exports the collected metrics through the native exporter and marks them as exported.
    self._metrics_exporter.ExportMetrics()
    self._exported = True
