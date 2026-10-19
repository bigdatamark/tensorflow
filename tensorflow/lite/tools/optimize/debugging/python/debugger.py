# [MODULE DOC]
# Implements the TFLite quantization debugger. It provides an options holder
# (QuantizationDebugOptions) and the debugger class (QuantizationDebugger) that runs a quantized
# model instrumented with numeric-verify operators (or builds one from a converter), feeds it a
# user supplied dataset, and computes per-layer and whole-model error metrics against float
# results. Both classes are exported under the lite experimental API namespace.
# Module-level state: default per-layer metric functions (element count, standard deviation,
# mean error, max absolute error, mean squared error) and the numeric-verify operator name.
# Import-time side effects: API export registration of the two classes.
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
"""Python TF-Lite QuantizationDebugger."""
import collections
import csv
import re
from typing import (Any, Callable, Dict, IO, Iterable, List, Mapping, Optional,
                    Sequence, Tuple)

import numpy as np

from tensorflow.lite.python import convert
from tensorflow.lite.python import interpreter as _interpreter
from tensorflow.lite.python.metrics import metrics as metrics_stub  # type: ignore
from tensorflow.python.util import tf_export


# TODO(b/198099651): move converter implementation out of lite.py
TFLiteConverter = Any  # importing tf.lite creates circular dependency

# Returns metrics based on difference of values for quantized/float ops.
_DEFAULT_LAYER_DEBUG_METRICS = {
    'num_elements': lambda diffs: diffs.size,
    'stddev': np.std,
    'mean_error': np.average,
    'max_abs_error': lambda diffs: np.max(np.abs(diffs)),
    'mean_squared_error': lambda diffs: np.average(diffs**2),
}

_NUMERIC_VERIFY_OP_NAME = 'NumericVerify'


def _get_quant_params(
    tensor_detail: Mapping[str, Any]) -> Optional[Tuple[float, int]]:
  # [FUNCTION DOC]
  # HELPER: quantization parameter lookup for the debugger's tensor conversion.
  # Takes a tensor detail mapping from an interpreter and returns the first scale and first zero
  # point as a pair, or nothing/null when the quantization parameters are empty or the scale or
  # zero point list is empty. Raises a key error if the quantization parameters key is absent.
  """Returns first scale and zero point from tensor detail, if present."""
  quant_params = tensor_detail['quantization_parameters']
  if not quant_params:
    return None
  if quant_params['scales'] and quant_params['zero_points']:
    return (quant_params['scales'][0], quant_params['zero_points'][0])
  return None


@tf_export.tf_export('lite.experimental.QuantizationDebugOptions')
class QuantizationDebugOptions:
  # [CLASS DOC]
  # Plain holder of settings for QuantizationDebugger: optional user metric mappings (per-layer
  # difference metrics, whole-model output metrics, per-layer direct compare metrics), op names
  # and tensor names to exclude from quantization, and a flag to fully quantize the model.
  # All values are stored as public attributes under the same names as the constructor arguments.
  # Exported in the public lite experimental API.
  """Debug options to set up a given QuantizationDebugger."""

  def __init__(self,
               layer_debug_metrics: Optional[Mapping[str,
                                                     Callable[[np.ndarray],
                                                              float]]] = None,
               model_debug_metrics: Optional[Mapping[
                   str, Callable[[Sequence[np.ndarray], Sequence[np.ndarray]],
                                 float]]] = None,
               layer_direct_compare_metrics: Optional[Mapping[str, Callable[
                   [Sequence[np.ndarray], Sequence[np.ndarray], float, int],
                   float]]] = None,
               denylisted_ops: Optional[List[str]] = None,
               denylisted_nodes: Optional[List[str]] = None,
               fully_quantize: bool = False) -> None:
    # [FUNCTION DOC]
    # Constructor. All arguments are optional. The three metric arguments are mappings from a
    # metric name to a function (per-layer difference, model output pair, direct compare with
    # scale and zero point respectively). denylisted_ops and denylisted_nodes are lists of op
    # names and output tensor names to leave unquantized. fully_quantize (default false) also
    # quantizes model inputs and outputs. Stores everything as attributes. Raises a value error
    # if any metric name appears more than once across the three metric mappings.
    """Initializes debugger options.

    Args:
      layer_debug_metrics: a dict to specify layer debug functions
        {function_name_str: function} where the function accepts result of
          NumericVerify Op, which is value difference between float and
          dequantized op results. The function returns single scalar value.
      model_debug_metrics: a dict to specify model debug functions
        {function_name_str: function} where the function accepts outputs from
          two models, and returns single scalar value for a metric. (e.g.
          accuracy, IoU)
      layer_direct_compare_metrics: a dict to specify layer debug functions
        {function_name_str: function}. The signature is different from that of
          `layer_debug_metrics`, and this one gets passed (original float value,
          original quantized value, scale, zero point). The function's
          implementation is responsible for correctly dequantize the quantized
          value to compare. Use this one when comparing diff is not enough.
          (Note) quantized value is passed as int8, so cast to int32 is needed.
      denylisted_ops: a list of op names which is expected to be removed from
        quantization.
      denylisted_nodes: a list of op's output tensor names to be removed from
        quantization.
      fully_quantize: Bool indicating whether to fully quantize the model.
        Besides model body, the input/output will be quantized as well.
        Corresponding to mlir_quantize's fully_quantize parameter.

    Raises:
      ValueError: when there are duplicate keys
    """
    self.layer_debug_metrics = layer_debug_metrics
    self.model_debug_metrics = model_debug_metrics
    self.layer_direct_compare_metrics = layer_direct_compare_metrics

    keys = []
    for metrics in [
        layer_debug_metrics, model_debug_metrics, layer_direct_compare_metrics
    ]:
      if metrics is not None:
        keys.extend(metrics.keys())
    if len(keys) != len(set(keys)):
      raise ValueError('Provided metrics have duplicate keys.')

    self.denylisted_ops = denylisted_ops
    self.denylisted_nodes = denylisted_nodes
    self.fully_quantize = fully_quantize


@tf_export.tf_export('lite.experimental.QuantizationDebugger')
class QuantizationDebugger:
  # [CLASS DOC]
  # Runs a quantized TFLite model that contains numeric-verify operators over a user dataset and
  # gathers error statistics. Can be created either from model files/bytes (quantized debug
  # model, optionally a float model) or from a converter, in which case it calibrates and
  # quantizes the model itself. State: interpreters for the quantized and optional float model,
  # the options object, the defining-op lookup, numeric-verify tensor and op details, and after
  # a run the layer statistics and model statistics attributes. Reports a debugger creation
  # metric. Exported in the public lite experimental API.
  """Debugger for Quantized TensorFlow Lite debug mode models.

  This can run the TensorFlow Lite converted models equipped with debug ops and
  collect debug information. This debugger calculates statistics from
  user-defined post-processing functions as well as default ones.
  """

  def __init__(self,
               quant_debug_model_path: Optional[str] = None,
               quant_debug_model_content: Optional[bytes] = None,
               float_model_path: Optional[str] = None,
               float_model_content: Optional[bytes] = None,
               debug_dataset: Optional[Callable[
                   [], Iterable[Sequence[np.ndarray]]]] = None,
               debug_options: Optional[QuantizationDebugOptions] = None,
               converter: Optional[TFLiteConverter] = None) -> None:
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it runs the model, but the constructor only sets up
    # interpreters and statistics containers; the run method does the actual running.
    # Constructor. Inputs: quantized debug model path or bytes, float model path or bytes (only
    # used when model output metrics are requested), a factory returning the dataset iterator,
    # debug options (defaults to a fresh options object) and an optional converter. With a
    # converter: if model metrics are wanted it first converts a float model (clearing the
    # converter optimizations temporarily and restoring them), then configures the converter for
    # calibration and converts a calibrated model, and builds the quantized (and float)
    # interpreters from it. Without a converter: creates the interpreter from the given debug
    # model, preserving all tensors when direct compare metrics are used, and a float interpreter
    # when model metrics are wanted. Finally initializes statistics, which raises a value error
    # if the model has no numeric-verify operators. Raises value errors from converter checks.
    """Runs the TFLite debugging model with given debug options.

    Args:
      quant_debug_model_path: Path to the quantized debug TFLite model file.
      quant_debug_model_content: Content of the quantized debug TFLite model.
      float_model_path: Path to float TFLite model file.
      float_model_content: Content of the float TFLite model.
      debug_dataset: a factory function that returns dataset generator which is
        used to generate input samples (list of np.ndarray) for the model. The
        generated elements must have same types and shape as inputs to the
        model.
      debug_options: Debug options to debug the given model.
      converter: Optional, use converter instead of quantized model.

    Raises:
      ValueError: If the debugger was unable to be created.

    Attributes:
      layer_statistics: results of error metrics for each NumericVerify op
        results. in {layer_name: {metric_name: metric}} format.
      model_statistics: results of error metrics for difference between float
        and quantized models. in {metric_name: metric} format.
    """
    self._data_gen = debug_dataset
    self._debug_options = debug_options or QuantizationDebugOptions()
    self.converter = None
    self.calibrated_model = None
    self.float_model = None
    self._float_interpreter = None
    if converter is not None:
      if self._debug_options.model_debug_metrics:
        old_optimizations = converter.optimizations
        self.converter = self._set_converter_options_for_float(converter)
        self.float_model = self.converter.convert()
        converter.optimizations = old_optimizations

      self.converter = self._set_converter_options_for_calibration(converter)
      self.calibrated_model = self.converter.convert()
      # Converter should be already set up with all options
      self._init_from_converter(
          self._debug_options,
          self.converter,
          self.calibrated_model,
          float_model=self.float_model)
    else:
      self._quant_interpreter = _interpreter.Interpreter(
          quant_debug_model_path,
          quant_debug_model_content,
          experimental_preserve_all_tensors=(
              self._debug_options.layer_direct_compare_metrics is not None))
      if self._debug_options.model_debug_metrics:
        self._float_interpreter = _interpreter.Interpreter(
            float_model_path, float_model_content)
    self._initialize_stats()

  @property
  def options(self) -> QuantizationDebugOptions:
    # [FUNCTION DOC]
    # Property getter. Returns the current debug options object.
    return self._debug_options

  @options.setter
  def options(self, options: QuantizationDebugOptions) -> None:
    # [FUNCTION DOC]
    # Property setter. Replaces the debug options. If the debugger was not built from a converter
    # (no converter or calibrated model) nothing else happens. Otherwise the quantized model is
    # regenerated with the new options (rebuilding interpreters) and statistics are reset, which
    # can raise a value error if the new model has no numeric-verify operators.
    self._debug_options = options
    if not self.converter or not self.calibrated_model:
      return
    self._init_from_converter(
        self._debug_options,
        self.converter,
        self.calibrated_model,
        float_model=self.float_model)
    self._initialize_stats()

  def _initialize_stats(self):
    # [FUNCTION DOC]
    # HELPER: finishes setup for the constructor and the options setter.
    # Builds a lookup from tensor index to the index of the operator producing it, resets the
    # cached numeric-verify details, and raises a value error if the quantized model has no
    # numeric-verify operators (not in debug mode). Merges the default layer metrics with user
    # supplied ones (user entries win on name clashes), clears layer and model statistics, and
    # creates a metrics recorder and increments its debugger creation counter.
    """Helper function initializes stats."""
    # TODO(b/177749613) : Fix the dependency on tf.lite._get_ops_details()
    # Following code is needed to get op's name from the output tensor index,
    # since NumericVerify op only provides its quantized input tensor index.
    self._defining_op = dict()
    for op_info in self._quant_interpreter._get_ops_details():  # pylint: disable=protected-access
      self._defining_op.update(
          {tensor_idx: op_info['index'] for tensor_idx in op_info['outputs']})

    self._numeric_verify_tensor_details = None
    self._numeric_verify_op_details = None
    if not self._get_numeric_verify_tensor_details():
      raise ValueError('Please check if the quantized model is in debug mode')

    self._layer_debug_metrics = _DEFAULT_LAYER_DEBUG_METRICS.copy()
    if self._debug_options.layer_debug_metrics:
      self._layer_debug_metrics.update(self._debug_options.layer_debug_metrics)

    self.layer_statistics = None
    self.model_statistics = None

    self._metrics = metrics_stub.TFLiteMetrics()
    self._metrics.increase_counter_debugger_creation()

  def _get_quantized_model(self, is_debug: bool) -> bytes:
    # [FUNCTION DOC]
    # HELPER: shared implementation of the public model getters.
    # Quantizes the stored calibrated model with the converter's per-channel setting and the
    # current options; is_debug selects whether numeric-verify operators are inserted. Returns
    # the model bytes. Raises a value error if the debugger was not created from a converter.
    if not self.converter:
      raise ValueError('No converter found, use this function with the '
                       'converter option in the constructor.')

    return convert.mlir_quantize(
        self.calibrated_model,
        disable_per_channel=self.converter._experimental_disable_per_channel,  # pylint: disable=protected-access
        fully_quantize=self._debug_options.fully_quantize,
        enable_numeric_verify=is_debug,
        denylisted_ops=self._debug_options.denylisted_ops,
        denylisted_nodes=self._debug_options.denylisted_nodes)

  def get_nondebug_quantized_model(self) -> bytes:
    # [FUNCTION DOC]
    # Returns the bytes of the quantized model without numeric-verify instrumentation, produced
    # from the calibrated model. Requires the debugger to have been created from a converter,
    # otherwise raises a value error.
    """Returns a non-instrumented quantized model.

    Convert the quantized model with the initialized converter and
    return bytes for nondebug model. The model will not be instrumented with
    numeric verification operations.

    Returns:
      Model bytes corresponding to the model.
    Raises:
      ValueError: if converter is not passed to the debugger.
    """
    return self._get_quantized_model(is_debug=False)

  def get_debug_quantized_model(self) -> bytes:
    # [FUNCTION DOC]
    # Returns the bytes of the quantized model with numeric-verify instrumentation, produced from
    # the calibrated model. Requires the debugger to have been created from a converter,
    # otherwise raises a value error.
    """Returns an instrumented quantized model.

    Convert the quantized model with the initialized converter and
    return bytes for model. The model will be instrumented with numeric
    verification operations and should only be used for debugging.

    Returns:
      Model bytes corresponding to the model.
    Raises:
      ValueError: if converter is not passed to the debugger.
    """
    return self._get_quantized_model(is_debug=True)

  def _init_from_converter(self,
                           options: QuantizationDebugOptions,
                           converter: TFLiteConverter,
                           calibrated_model: Optional[bytes] = None,
                           float_model: Optional[bytes] = None) -> None:
    # [FUNCTION DOC]
    # HELPER: builds interpreters from a converter's calibrated model.
    # Inputs: options object, converter, calibrated model bytes and optional float model bytes.
    # Quantizes the calibrated model with numeric verify enabled using the options (exclusion
    # lists, full quantization) and the converter's per-channel setting, stores the result as the
    # quantized model attribute, creates the quantized interpreter, and resets the float
    # interpreter to a new one built from the float model when given (otherwise nothing/null).
    # Returns nothing.
    """Convert the model and apply options.

    Converts the quantized model and initializes a quantized model interpreter
    with the quantized model. Returns a float model interpreter if float model
    is provided.

    Args:
      options: a QuantizationDebugOptions object.
      converter: an initialized tf.lite.TFLiteConverter.
      calibrated_model: Calibrated model bytes.
      float_model: Float model bytes.
    """
    self.quant_model = convert.mlir_quantize(
        calibrated_model,
        disable_per_channel=converter._experimental_disable_per_channel,  # pylint: disable=protected-access
        fully_quantize=options.fully_quantize,
        enable_numeric_verify=True,
        denylisted_ops=options.denylisted_ops,
        denylisted_nodes=options.denylisted_nodes)
    self._quant_interpreter = _interpreter.Interpreter(
        model_content=self.quant_model)
    self._float_interpreter = None
    if float_model is not None:
      self._float_interpreter = _interpreter.Interpreter(
          model_content=float_model)

  def _set_converter_options_for_float(
      self, converter: TFLiteConverter) -> TFLiteConverter:
    # [FUNCTION DOC]
    # HELPER: prepares the converter to produce a float model.
    # KNOWN ISSUE: the docstring mentions verification and experimental options, but the code
    # only clears the optimizations list when it is non-empty.
    # Mutates the converter in place and returns the same converter.
    """Verify converter options and set required experimental options."""
    if converter.optimizations:
      converter.optimizations = []
    return converter

  def _set_converter_options_for_calibration(
      self, converter: TFLiteConverter) -> TFLiteConverter:
    # [FUNCTION DOC]
    # HELPER: prepares the converter to produce a calibrated (not yet quantized) model.
    # Raises a value error if the converter has no optimizations set or no representative
    # dataset. Otherwise enables the experimental MLIR quantizer and the calibrate-only mode on
    # the converter (mutating it) and returns the same converter.
    """Verify converter options and set required experimental options."""
    if not converter.optimizations:
      raise ValueError(
          'converter object must set optimizations to lite.Optimize.DEFAULT')
    if not converter.representative_dataset:
      raise ValueError('converter object must set representative_dataset')

    converter.experimental_mlir_quantizer = True
    converter._experimental_calibrate_only = True  # pylint: disable=protected-access
    return converter

  def run(self) -> None:
    # [FUNCTION DOC]
    # Runs the debug dataset through the quantized model and stores aggregated per-layer
    # statistics in the layer statistics attribute. If model output metrics were configured it
    # also runs the float model and stores aggregated model statistics. Returns nothing.
    """Runs models and gets metrics."""
    self.layer_statistics = self._collect_layer_statistics()
    if self._debug_options.model_debug_metrics:
      self.model_statistics = self._collect_model_statistics()

  def _collect_layer_statistics(self) -> Dict[str, Dict[str, float]]:
    # [FUNCTION DOC]
    # HELPER: per-layer measurement step for run.
    # For each sample from the dataset factory: sets the quantized interpreter inputs (resizing
    # and allocating on the first sample), invokes it, and for every numeric-verify output tensor
    # applies each per-layer metric function to the tensor values (the float-minus-dequantized
    # differences). If direct compare metrics exist, it also finds the numeric-verify operator's
    # float and quantized input tensors and calls each direct compare function with the float
    # tensor, quantized tensor, first scale and first zero point of the quantized input.
    # Finally each collected list is reduced to one value with a mean that ignores not-a-number
    # entries. Returns a mapping from tensor name to a mapping from metric name to value.
    """Collects layer statistics by applying layer debug metrics.

    For all data from the given RepresentativeDataset, collect statistics per
    example by getting the NumericVerify op results in _quant_interpreter
    and calculating layer debug metrics on the results.

    Returns:
      aggregated per-layer statistics of NumericVerify results.
      {layer_name: {metric_name: metric}}
    """
    layer_statistics = collections.defaultdict(
        lambda: collections.defaultdict(list))

    initialize = True
    for tensor_data in self._data_gen():
      self._set_input_tensors(self._quant_interpreter, tensor_data, initialize)
      initialize = False

      # Run the model.
      self._quant_interpreter.invoke()

      # Collect the statistics of this invoke result.
      for tensor_detail in self._get_numeric_verify_tensor_details():
        tensor_name = tensor_detail['name']  # pytype: disable=unsupported-operands  # dynamic-method-lookup
        diffs = self._quant_interpreter.get_tensor(tensor_detail['index'])  # pytype: disable=unsupported-operands  # dynamic-method-lookup
        for metric_name, metric_fn in self._layer_debug_metrics.items():
          layer_statistics[tensor_name][metric_name].append(metric_fn(diffs))

      if self._debug_options.layer_direct_compare_metrics is not None:
        for tensor_detail in self._get_numeric_verify_tensor_details():
          tensor_name = tensor_detail['name']  # pytype: disable=unsupported-operands  # dynamic-method-lookup
          op_idx = self._defining_op[tensor_detail['index']]  # pytype: disable=unsupported-operands  # dynamic-method-lookup
          op_detail = self._quant_interpreter._get_op_details(op_idx)  # pylint: disable=protected-access
          q_idx, f_idx = op_detail['inputs']
          quant_input_detail = self._quant_interpreter._get_tensor_details(  # pylint: disable=protected-access
              q_idx, subgraph_index=0)
          for (metric_name, metric_fn
              ) in self._debug_options.layer_direct_compare_metrics.items():
            layer_statistics[tensor_name][metric_name].append(
                metric_fn(
                    self._quant_interpreter.get_tensor(f_idx),
                    self._quant_interpreter.get_tensor(q_idx),
                    quant_input_detail['quantization_parameters']['scales'][0],
                    quant_input_detail['quantization_parameters']['zero_points']
                    [0]))

    # Calculate final aggregated metrics for each layer.
    for metrics in layer_statistics.values():
      for metric_name in metrics:
        metrics[metric_name] = np.nanmean(metrics[metric_name])

    return layer_statistics

  def _collect_model_statistics(self) -> Dict[str, float]:
    # [FUNCTION DOC]
    # HELPER: model-level measurement step for run.
    # KNOWN ISSUE: the docstring says a model results attribute is filled; the code instead
    # returns the aggregated mapping. If no float interpreter exists, the float outputs passed to
    # the metric functions are an empty list.
    # For each dataset sample, runs the quantized interpreter and (when present) the float
    # interpreter, converting int8 outputs back to float using their quantization parameters,
    # and applies every model metric function to the pair of output lists. Returns a mapping from
    # metric name to the mean of its per-sample values.
    """Collects model output metrics.

    For all data from the given RepresentativeDataset, collect all model output
    results from float model & quantized debug model, and calculate metrics
    by using model output functions. As a result, self.model_results is filled,

    where self.model_results[model_output_function_name] = `aggregated model
    output function value` (a scalar).

    Returns:
      aggregated per-model output discrepancy metrics.
      {metric_name: aggregated_metric}
    """

    model_statistics = collections.defaultdict(list)

    initialize = True
    for tensor_data in self._data_gen():
      # Run quantized debug model and collect output results.
      self._set_input_tensors(self._quant_interpreter, tensor_data, initialize)
      self._quant_interpreter.invoke()
      quant_tensor_data = self._get_output_tensors(self._quant_interpreter)

      # Run float model if it's initialized.
      float_tensor_data = []
      if self._float_interpreter:
        self._set_input_tensors(
            self._float_interpreter, tensor_data, initialize)
        self._float_interpreter.invoke()
        float_tensor_data = self._get_output_tensors(self._float_interpreter)

      initialize = False

      # Calculate the metrics.
      for (metric_name,
           metric_fn) in self._debug_options.model_debug_metrics.items():
        model_statistics[metric_name].append(
            metric_fn(float_tensor_data, quant_tensor_data))

    # Calculate final aggregated metrics for each outputs.
    return {
        metric_name: np.mean(metric)
        for metric_name, metric in model_statistics.items()
    }

  def _set_input_tensors(self, interpreter: _interpreter.Interpreter,
                         tensor_data: Sequence[np.ndarray],
                         initialize: bool) -> None:
    # [FUNCTION DOC]
    # HELPER: feeds one dataset sample to an interpreter.
    # Inputs: an interpreter, a sequence of numpy arrays (one per model input) and a flag that is
    # true for the first sample, in which case input tensors are resized to the sample shapes and
    # tensors are allocated. Float32 data destined for an int8 input is quantized with the input's
    # first scale and zero point when quantization parameters exist. Sets the input tensors;
    # returns nothing. Raises a value error when the number of arrays differs from the number of
    # model inputs.
    """Sets input tensors into TFLite model Interpreter.

    Args:
      interpreter: a tf.lite.Interpreter object with allocated tensors.
      tensor_data: a list of Numpy array data.
      initialize: set to true when input is first set for the interpreter, to
        set input shapes and allocate tensors.

    Raises:
      ValueError: when inputs can't be set, or size of provided inputs does not
      match size of model inputs.
    """
    input_details = interpreter.get_input_details()
    if len(input_details) != len(tensor_data):
      raise ValueError(
          'Number of inputs provided ({}) does not match number of inputs to '
          'the model ({})'.format(len(tensor_data), len(input_details)))

    if initialize:
      for input_detail, tensor in zip(input_details, tensor_data):
        interpreter.resize_tensor_input(input_detail['index'], tensor.shape)
      interpreter.allocate_tensors()

    for input_detail, tensor in zip(input_details, tensor_data):
      if tensor.dtype == np.float32 and input_detail['dtype'] == np.int8:
        quant_params = _get_quant_params(input_detail)
        if quant_params:
          scale, zero_point = quant_params
          tensor = np.round((tensor / scale) + zero_point).astype(np.int8)
      interpreter.set_tensor(input_detail['index'], tensor)

  def _get_output_tensors(
      self, interpreter: _interpreter.Interpreter) -> List[np.ndarray]:
    # [FUNCTION DOC]
    # HELPER: reads model outputs after an invocation.
    # Returns a list of numpy arrays, one per output tensor of the interpreter. Outputs of int8
    # type with quantization parameters are dequantized to float32 using the first scale and zero
    # point; others are returned as read.
    """Returns output tensors of given TFLite model Interpreter.

    Args:
      interpreter: a tf.lite.Interpreter object with allocated tensors.

    Returns:
      a list of numpy arrays representing output tensor results.
    """

    outputs = []
    for output_detail in interpreter.get_output_details():
      tensor = interpreter.get_tensor(output_detail['index'])
      if output_detail['dtype'] == np.int8:
        quant_params = _get_quant_params(output_detail)
        if quant_params:
          scale, zero_point = quant_params
          tensor = ((tensor.astype(np.float32) - zero_point) * scale).astype(
              np.float32)
      outputs.append(tensor)

    return outputs

  def _get_numeric_verify_tensor_details(self) -> List[str]:
    # [FUNCTION DOC]
    # HELPER: lazily collects numeric-verify tensor information.
    # KNOWN ISSUE: the type hint and docstring claim a list of names, but the code returns a list
    # of tensor detail mappings (output tensor of each numeric-verify operator).
    # On first use (or when the cached list is empty) scans the quantized interpreter's operators
    # for numeric-verify ones, caching the output tensor details and a mapping from tensor name to
    # operator info. Returns the cached list of tensor details (possibly empty).
    """Returns all names of all tensors from NumericVerify op."""
    # pylint: disable=protected-access
    if not self._numeric_verify_tensor_details:
      self._numeric_verify_tensor_details = []
      self._numeric_verify_op_details = {}
      for op_info in self._quant_interpreter._get_ops_details():
        if op_info['op_name'] == _NUMERIC_VERIFY_OP_NAME:
          self._numeric_verify_tensor_details.append(
              self._quant_interpreter._get_tensor_details(
                  op_info['outputs'][0], subgraph_index=0))
          tensor_name = self._numeric_verify_tensor_details[-1]['name']
          self._numeric_verify_op_details[tensor_name] = op_info
    # pylint: enable=protected-access
    return self._numeric_verify_tensor_details

  def _get_operand_name_and_index(self,
                                  numeric_verify_name: str) -> Tuple[str, int]:
    # [FUNCTION DOC]
    # HELPER: name parsing for the statistics dump.
    # Takes the numeric-verify output tensor name, which has a fixed prefix, the quantized tensor
    # name and a colon-separated index. Splits at the last colon, strips the prefix and slash,
    # and drops one trailing digit of the remaining name if it ends with a digit. Returns a pair
    # of the resulting tensor name and the integer after the colon. Raises an error if the text
    # has no colon or the index is not an integer.
    """Gets the index and name of NumericVerify Op's quantized input tensor.

    Args:
      numeric_verify_name: name of the NumericVerify op's output tensor. It has
        format of `NumericVerify/{quantized_tensor_name}:{quantized_tensor_idx}`

    Returns:
      Tuple of (tensor_name, tensor_idx) for quantized op's output tensor.
    """
    tensor_name, tensor_idx = numeric_verify_name.rsplit(':', 1)
    float_tensor_name = tensor_name[len(_NUMERIC_VERIFY_OP_NAME) + 1:]
    if re.match(r'\d', float_tensor_name[-1]):
      float_tensor_name = float_tensor_name[:-1]

    return (float_tensor_name, int(tensor_idx))

  def layer_statistics_dump(self, file: IO[str]) -> None:
    # [FUNCTION DOC]
    # Writes the layer statistics as comma-separated values to a file or file-like object. The
    # header lists operator name, tensor index, all layer metric names, direct compare metric
    # names (when configured), scale, zero point and tensor name. Writes one row per layer in the
    # statistics (none if run has not been called), filling operator name, tensor index, scale
    # and zero point from the quantized interpreter's details. Returns nothing.
    """Dumps layer statistics into file, in csv format.

    Args:
      file: file, or file-like object to write.
    """
    # order of `fields` is the order of fields in csv.
    fields = ['op_name', 'tensor_idx'] + list(self._layer_debug_metrics.keys())
    if self._debug_options.layer_direct_compare_metrics is not None:
      fields += list(self._debug_options.layer_direct_compare_metrics.keys())
    fields += ['scale', 'zero_point', 'tensor_name']
    writer = csv.DictWriter(file, fields)
    writer.writeheader()
    if self.layer_statistics:
      for name, metrics in self.layer_statistics.items():
        data = metrics.copy()
        (data['tensor_name'], _) = self._get_operand_name_and_index(name)
        data['tensor_idx'] = self._numeric_verify_op_details[name]['inputs'][0]
        data['op_name'] = self._quant_interpreter._get_op_details(  # pylint: disable=protected-access
            self._defining_op[data['tensor_idx']])['op_name']
        details = self._quant_interpreter._get_tensor_details(  # pylint: disable=protected-access
            data['tensor_idx'], subgraph_index=0)
        data['scale'], data['zero_point'] = (
            details['quantization_parameters']['scales'][0],
            details['quantization_parameters']['zero_points'][0])
        writer.writerow(data)
