# [MODULE DOC]
# Defines the vocabulary and decorator used to attribute TFLite converter failures to a pipeline
# stage. Exposes the Component enum (coarse converter phases), the SubComponent enum (finer
# stages, each tied to one component), the ConverterError exception, and the convert_phase
# decorator factory. Wrapped converter functions report any exception to the TFLite converter
# metrics collector tagged with the component and subcomponent. Importing has no side effects
# beyond loading the error-data protocol buffer and metrics modules.
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
"""Utilities for collecting TFLite metrics."""

import collections
import enum
import functools
from typing import Text

from tensorflow.compiler.mlir.lite.metrics import converter_error_data_pb2
from tensorflow.lite.python.metrics import metrics


class Component(enum.Enum):
  # [CLASS DOC]
  # Enumeration of the three coarse converter phases: preparing the TensorFlow model, converting
  # it to the TFLite format, and optimizing the TFLite model (quantization/sparsification).
  # Member values are the same strings as the member names and are what gets written into error
  # reports as the component.
  """Enum class defining name of the converter components."""
  # Validate the given input and prepare and optimize TensorFlow Model.
  PREPARE_TF_MODEL = "PREPARE_TF_MODEL"

  # Convert to TFLite model format.
  CONVERT_TF_TO_TFLITE_MODEL = "CONVERT_TF_TO_TFLITE_MODEL"

  # RUN quantization and sparsification.
  OPTIMIZE_TFLITE_MODEL = "OPTIMIZE_TFLITE_MODEL"


SubComponentItem = collections.namedtuple("SubComponentItem",
                                          ["name", "component"])


class SubComponent(SubComponentItem, enum.Enum):
  # [CLASS DOC]
  # Enumeration of finer converter stages. Each member's value is a pair of a name string and the
  # owning Component (or nothing for the unspecified member). The class overrides string
  # conversion and the name property so both yield the plain stage name, and exposes the owning
  # component through a property. Used by the phase decorator to validate that a subcomponent
  # belongs to the declared component.
  """Enum class defining name of the converter subcomponents.

  This enum only defines the subcomponents in Python, there might be more
  subcomponents defined in C++.
  """

  def __str__(self):
    # [FUNCTION DOC]
    # Returns the plain stage name string stored in the member's value, not the default
    # enumeration representation.
    return self.value.name

  @property
  def name(self):
    # [FUNCTION DOC]
    # Property returning the stage name string held in the member's value (overrides the
    # built-in enumeration member name, which is the attribute name).
    return self.value.name

  @property
  def component(self):
    # [FUNCTION DOC]
    # Property returning the Component this subcomponent belongs to; nothing for the
    # unspecified member.
    return self.value.component

  # The subcomponent name is unspecified.
  UNSPECIFIED = SubComponentItem("UNSPECIFIED", None)

  # Valid the given input and parameters.
  VALIDATE_INPUTS = SubComponentItem("VALIDATE_INPUTS",
                                     Component.PREPARE_TF_MODEL)

  # Load GraphDef from SavedModel.
  LOAD_SAVED_MODEL = SubComponentItem("LOAD_SAVED_MODEL",
                                      Component.PREPARE_TF_MODEL)

  # Convert a SavedModel to frozen graph.
  FREEZE_SAVED_MODEL = SubComponentItem("FREEZE_SAVED_MODEL",
                                        Component.PREPARE_TF_MODEL)

  # Save a Keras model to SavedModel.
  CONVERT_KERAS_TO_SAVED_MODEL = SubComponentItem(
      "CONVERT_KERAS_TO_SAVED_MODEL", Component.PREPARE_TF_MODEL)

  # Save Concrete functions to SavedModel.
  CONVERT_CONCRETE_FUNCTIONS_TO_SAVED_MODEL = SubComponentItem(
      "CONVERT_CONCRETE_FUNCTIONS_TO_SAVED_MODEL", Component.PREPARE_TF_MODEL)

  # Convert a Keras model to a frozen graph.
  FREEZE_KERAS_MODEL = SubComponentItem("FREEZE_KERAS_MODEL",
                                        Component.PREPARE_TF_MODEL)

  # Replace all the variables with constants in a ConcreteFunction.
  FREEZE_CONCRETE_FUNCTION = SubComponentItem("FREEZE_CONCRETE_FUNCTION",
                                              Component.PREPARE_TF_MODEL)

  # Run grappler optimization.
  OPTIMIZE_TF_MODEL = SubComponentItem("OPTIMIZE_TF_MODEL",
                                       Component.PREPARE_TF_MODEL)

  # Convert using the old TOCO converter.
  CONVERT_GRAPHDEF_USING_DEPRECATED_CONVERTER = SubComponentItem(
      "CONVERT_GRAPHDEF_USING_DEPRECATED_CONVERTER",
      Component.CONVERT_TF_TO_TFLITE_MODEL)

  # Convert a GraphDef to TFLite model.
  CONVERT_GRAPHDEF = SubComponentItem("CONVERT_GRAPHDEF",
                                      Component.CONVERT_TF_TO_TFLITE_MODEL)

  # Convert a SavedModel to TFLite model.
  CONVERT_SAVED_MODEL = SubComponentItem("CONVERT_SAVED_MODEL",
                                         Component.CONVERT_TF_TO_TFLITE_MODEL)

  # Convert a Jax HLO to TFLite model.
  CONVERT_JAX_HLO = SubComponentItem("CONVERT_JAX_HLO",
                                     Component.CONVERT_TF_TO_TFLITE_MODEL)

  # Do quantization by the deprecated quantizer.
  QUANTIZE_USING_DEPRECATED_QUANTIZER = SubComponentItem(
      "QUANTIZE_USING_DEPRECATED_QUANTIZER", Component.OPTIMIZE_TFLITE_MODEL)

  # Do calibration.
  CALIBRATE = SubComponentItem("CALIBRATE", Component.OPTIMIZE_TFLITE_MODEL)

  # Do quantization by MLIR.
  QUANTIZE = SubComponentItem("QUANTIZE", Component.OPTIMIZE_TFLITE_MODEL)

  # Do sparsification by MLIR.
  SPARSIFY = SubComponentItem("SPARSIFY", Component.OPTIMIZE_TFLITE_MODEL)


class ConverterError(Exception):
  # [CLASS DOC]
  # Exception raised when model conversion fails. Holds a list of structured converter error
  # records (errors), populated by appending explicitly or by recognizing known message
  # patterns at construction. The phase decorator reports these records to the metrics collector.
  """Raised when an error occurs during model conversion."""

  def __init__(self, message):
    # [FUNCTION DOC]
    # Constructor. Takes the error message string, passes it to the base exception, creates an
    # empty errors list and then scans the message for known patterns to add structured records.
    super(ConverterError, self).__init__(message)
    self.errors = []
    self._parse_error_message(message)

  def append_error(self,
                   error_data: converter_error_data_pb2.ConverterErrorData):
    # [FUNCTION DOC]
    # Adds one structured converter error record (error_data) to this exception's errors list.
    # Returns nothing.
    self.errors.append(error_data)

  def _parse_error_message(self, message):
    # [FUNCTION DOC]
    # HELPER: constructor helper that derives structured error records from raw message text.
    # Checks the message for a known substring (currently only the message about failing to
    # functionalize control flow version 1 operations). On the first match it builds an error
    # record holding the full message and the matching error code (unsupported control flow v1),
    # appends it and stops. Returns nothing; mutates the errors list.
    """If the message matches a pattern, assigns the associated error code.

    It is difficult to assign an error code to some errrors in MLIR side, Ex:
    errors thrown by other components than TFLite or not using mlir::emitError.
    This function try to detect them by the error message and assign the
    corresponding error code.

    Args:
      message: The error message of this exception.
    """
    error_code_mapping = {
        "Failed to functionalize Control Flow V1 ops. Consider using Control "
        "Flow V2 ops instead. See https://www.tensorflow.org/api_docs/python/"
        "tf/compat/v1/enable_control_flow_v2.":
            converter_error_data_pb2.ConverterErrorData
            .ERROR_UNSUPPORTED_CONTROL_FLOW_V1,
    }
    for pattern, error_code in error_code_mapping.items():
      if pattern in message:
        error_data = converter_error_data_pb2.ConverterErrorData()
        error_data.error_message = message
        error_data.error_code = error_code
        self.append_error(error_data)
        return


def convert_phase(component, subcomponent=SubComponent.UNSPECIFIED):
  # [FUNCTION DOC]
  # Decorator factory. Takes a Component and an optional SubComponent (default unspecified) and
  # returns a decorator that wraps a function so any exception it raises is reported to the
  # TFLite converter metrics collector and then re-raised unchanged (without chained context).
  # Raises ValueError immediately if component or subcomponent is not a valid enum member, or if
  # a specified subcomponent does not belong to the given component.
  # Reporting: records from a ConverterError's errors list are each reported; if it has none, or
  # for any other exception, a record is built with the stringified message. Reporting always
  # overwrites the record's component, but sets the subcomponent only if empty. Side effect:
  # calls the metrics singleton to store the converter error.
  """The decorator to identify converter component and subcomponent.

  Args:
    component: Converter component name.
    subcomponent: Converter subcomponent name.

  Returns:
    Forward the result from the wrapped function.

  Raises:
    ValueError: if component and subcomponent name is not valid.
  """
  if component not in Component:
    raise ValueError("Given component name not found")
  if subcomponent not in SubComponent:
    raise ValueError("Given subcomponent name not found")
  if (subcomponent != SubComponent.UNSPECIFIED and
      subcomponent.component != component):
    raise ValueError("component and subcomponent name don't match")

  def report_error(error_data: converter_error_data_pb2.ConverterErrorData):
    # [FUNCTION DOC]
    # HELPER: nested reporter used by the wrapper created in convert_phase.
    # Stamps the error record with the enclosing component value, fills the subcomponent name if
    # not already set, and passes it to the converter metrics object to record. Returns nothing.
    # Always overwrites the component information, but only overwrites the
    # subcomponent if it is not available.
    error_data.component = component.value
    if not error_data.subcomponent:
      error_data.subcomponent = subcomponent.name
    tflite_metrics = metrics.TFLiteConverterMetrics()
    tflite_metrics.set_converter_error(error_data)

  def report_error_message(error_message: Text):
    # [FUNCTION DOC]
    # HELPER: nested reporter used by the wrapper created in convert_phase.
    # Wraps a plain message string in a new error record and reports it via the nested report
    # function. Returns nothing.
    error_data = converter_error_data_pb2.ConverterErrorData()
    error_data.error_message = error_message
    report_error(error_data)

  def actual_decorator(func):
    # [FUNCTION DOC]
    # The decorator returned by the factory. Takes the function to wrap and returns a wrapper
    # that preserves the original function's metadata.

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
      # [FUNCTION DOC]
      # Calls the wrapped function with all given positional and keyword arguments and returns
      # its result. On a converter error, reports its structured records (or its message if it
      # has none); on any other exception, reports the message. Both are re-raised as is.
      try:
        return func(*args, **kwargs)
      except ConverterError as converter_error:
        if converter_error.errors:
          for error_data in converter_error.errors:
            report_error(error_data)
        else:
          report_error_message(str(converter_error))
        raise converter_error from None  # Re-throws the exception.
      except Exception as error:
        report_error_message(str(error))
        raise error from None  # Re-throws the exception.

    return wrapper

  return actual_decorator
