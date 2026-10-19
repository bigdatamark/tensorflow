# [MODULE DOC]
# Python wrapper around the native calibration wrapper used for post training quantization. Defines
# a helper to add intermediate tensors to a model and a calibrator class that feeds representative
# samples to a model, then returns a calibrated or quantized model. The native extension is loaded
# lazily on first use.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Python wrapper for post training quantization with calibration."""
import numpy as np

from tensorflow.lite.python.convert_phase import Component
from tensorflow.lite.python.convert_phase import convert_phase
from tensorflow.lite.python.convert_phase import SubComponent
from tensorflow.lite.python.interpreter import Interpreter
from tensorflow.python.framework import dtypes
from tensorflow.python.util.lazy_loader import LazyLoader

# Lazy load since some of the performance benchmark skylark rules
# break dependencies. Must use double quotes to match code internal rewrite
# rule.
_calibration_wrapper = LazyLoader(
    "_calibration_wrapper",
    globals(),
    (
        "tensorflow.lite.python.optimize."
        "_pywrap_tensorflow_lite_calibration_wrapper"
    ),
)


def add_intermediate_tensors(model_content):
  # [FUNCTION DOC]
  # Takes serialized TFLite model bytes and returns the model bytes produced by the native routine
  # that adds intermediate tensors to fused ops where needed (so they can be calibrated).
  """Adds intermediate tensors to fused op if needed."""
  return _calibration_wrapper.AddIntermediateTensors(model_content)


class Calibrator:
  # [CLASS DOC]
  # Internal class wrapping the native calibration wrapper for one float model. State: the native
  # calibrator, the model bytes, and a lazily created interpreter (only needed to map named,
  # signature based inputs to positional inputs). Used to run representative data through the model
  # to collect min/max statistics, optionally followed by quantization.
  """Calibrates a floating point model and then quantizes it.

  This is an internal class, not a public interface.
  """

  def __init__(
      self,
      model_content,
      custom_op_registerers_by_name=None,
      custom_op_registerers_by_func=None,
  ):
    # [FUNCTION DOC]
    # Constructor. model_content is the serialized model bytes (required, must be non-empty).
    # custom_op_registerers_by_name and custom_op_registerers_by_func are optional lists (default
    # empty) of registerer symbol names or functions for custom ops. Creates the native calibrator;
    # raises a value error if model_content is empty, if native creation raises (message includes the
    # cause), or if the created calibrator is falsy. Leaves the interpreter unset.
    """Constructor.

    Args:
      model_content: Content of a TF-Lite Flatbuffer file.
      custom_op_registerers_by_name: List of str (symbol names) that take a
        pointer to a MutableOpResolver and register custom ops.
      custom_op_registerers_by_func: List of functions that take a pointer to a
        MutableOpResolver and register custom ops.

    Raises:
      ValueError: If the calibrator was unable to open the model.
    """
    if not model_content:
      raise ValueError("`model_content` must be specified.")
    if custom_op_registerers_by_name is None:
      custom_op_registerers_by_name = []
    if custom_op_registerers_by_func is None:
      custom_op_registerers_by_func = []
    try:
      self._calibrator = _calibration_wrapper.CalibrationWrapper(
          model_content,
          custom_op_registerers_by_name,
          custom_op_registerers_by_func,
      )
      self._model_content = model_content
    except Exception as e:
      raise ValueError("Failed to parse the model: %s." % e)
    if not self._calibrator:
      raise ValueError("Failed to parse the model.")
    self._interpreter = None

  def _create_input_array_from_dict(self, signature_key, inputs):
    # [FUNCTION DOC]
    # HELPER: orders named inputs for the calibration feed.
    # Uses the interpreter's signature runner for signature_key (nothing selects the default) to
    # get input details, sorts them by tensor index, and returns a list of the values from the
    # inputs mapping in that order. A missing input name raises a key error. Requires the
    # interpreter to exist already.
    input_array = []
    signature_runner = self._interpreter.get_signature_runner(signature_key)
    input_details = sorted(
        signature_runner.get_input_details().items(),
        key=lambda item: item[1]["index"],
    )
    for input_name, _ in input_details:
      input_array.append(inputs[input_name])
    return input_array

  def _feed_tensors(self, dataset_gen, resize_input):
    # [FUNCTION DOC]
    # HELPER: feeds all calibration samples to the native calibrator.
    # dataset_gen is a callable returning an iterable of samples; each sample is a list of input
    # arrays, a mapping of input names to arrays, or a pair of signature key and such a mapping.
    # resize_input says whether to prepare the calibrator with the sample shapes. For each new
    # signature key (once per key) it prepares the calibrator, then feeds the sample. Raises a
    # value error for unsupported sample types or a pair whose second item is not a mapping.
    # Creates the interpreter lazily for mapping-based samples.
    """Feed tensors to the calibrator."""
    initialized = {}

    for sample in dataset_gen():
      if isinstance(sample, tuple):
        if not isinstance(sample[1], dict):
          raise ValueError(
              "You need to provide either a dictionary with input "
              "names and values in the second argument in the "
              "tuple"
          )
        # Convert signature based inputs to the tensor index based data.
        if self._interpreter is None:
          self._interpreter = Interpreter(model_content=self._model_content)
        signature_key = sample[0]
        input_array = self._create_input_array_from_dict(
            signature_key, sample[1]
        )
      elif isinstance(sample, dict):
        # Convert signature based inputs to the tensor index based data.
        if self._interpreter is None:
          self._interpreter = Interpreter(model_content=self._model_content)
        signature_key = None
        input_array = self._create_input_array_from_dict(None, sample)
      elif isinstance(sample, list):
        signature_key = None
        input_array = sample
      else:
        raise ValueError(
            "You need to provide either a dictionary with input "
            "names and values, a tuple with signature key and a "
            "dictionary with input names and values, or an array "
            "with input values in the order of input tensors of "
            "the graph in the representative_dataset function. "
            "Unsupported value from dataset: {}.".format(sample)
        )

      if signature_key not in initialized:
        initialized[signature_key] = True
        if resize_input:
          if signature_key is not None:
            self._calibrator.Prepare(
                [list(s.shape) for s in input_array], signature_key
            )
          else:
            self._calibrator.Prepare([list(s.shape) for s in input_array])
        else:
          if signature_key is not None:
            self._calibrator.Prepare(signature_key)
          else:
            self._calibrator.Prepare()
      if signature_key is not None:
        self._calibrator.FeedTensor(input_array, signature_key)
      else:
        self._calibrator.FeedTensor(input_array)

  @convert_phase(
      Component.OPTIMIZE_TFLITE_MODEL,
      SubComponent.QUANTIZE_USING_DEPRECATED_QUANTIZER,
  )
  def calibrate_and_quantize(
      self,
      dataset_gen,
      input_type,
      output_type,
      allow_float,
      activations_type=dtypes.int8,
      bias_type=dtypes.int32,
      resize_input=True,
      disable_per_channel=False,
      disable_per_channel_quantization_for_dense_layers=False,
  ):
    # [FUNCTION DOC]
    # Runs calibration with the samples from dataset_gen and returns the quantized model bytes.
    # Parameters: dataset_gen (callable producing samples), input_type and output_type (tensor
    # data types of the real-valued model input and output), allow_float (boolean; false forbids
    # float fallback), activations_type (default 8-bit integer), bias_type (default 32-bit
    # integer), resize_input (default true, resizes inputs to sample shapes),
    # disable_per_channel and disable_per_channel_quantization_for_dense_layers (booleans,
    # default false). Errors from sample handling or the native quantizer propagate.
    # Decorated to record the conversion phase for quantization with the deprecated quantizer.
    """Calibrates the model with specified generator and then quantizes it.

    The input shapes of the calibrator are resized with the calibration data if
    `resize_input` is set.

    Returns:
      A quantized model.

    Args:
      dataset_gen: A generator that generates calibration samples.
      input_type: A tf.dtype representing the desired real-value input type.
      output_type: A tf.dtype representing the desired real-value output type.
      allow_float: A boolean. False if the resulting model cannot perform float
        computation, useful when targeting an integer-only backend. If False, an
        error will be thrown if an operation cannot be quantized, otherwise the
        model will fallback to float ops.
      activations_type: A tf.dtype representing the desired type for
        activations.
      bias_type: A tf.dtype representing the desired type for bias.
      resize_input: A boolean. True if the shape of the sample data is different
        from the input.
      disable_per_channel: A boolean. True if disabling per-channel
        quantization.
      disable_per_channel_quantization_for_dense_layers: A boolean. True if
        disabling per-channel quantization only in Dense layers.
    """
    self._feed_tensors(dataset_gen, resize_input)
    return self._calibrator.QuantizeModel(
        np.dtype(input_type.as_numpy_dtype()).num,
        np.dtype(output_type.as_numpy_dtype()).num,
        allow_float,
        np.dtype(activations_type.as_numpy_dtype()).num,
        np.dtype(bias_type.as_numpy_dtype()).num,
        disable_per_channel,
        disable_per_channel_quantization_for_dense_layers,
    )

  @convert_phase(
      Component.OPTIMIZE_TFLITE_MODEL,
      SubComponent.QUANTIZE_USING_DEPRECATED_QUANTIZER,
  )
  def calibrate_and_quantize_single(
      self,
      dataset_gen,
      input_type,
      output_type,
      allow_float,
      op_output_name,
      resize_input=True,
  ):
    # [FUNCTION DOC]
    # Like the full quantize method but quantizes only the op whose output is named
    # op_output_name (text), using default activation and bias types. Parameters: dataset_gen,
    # input_type, output_type, allow_float, op_output_name, resize_input (default true). Feeds
    # the samples, then returns the quantized model bytes. Note: allow_float is passed to the
    # native call before the op name. Decorated to record the conversion phase.
    """Calibrates the model with specified generator and then quantizes it.

    Only the single op with output op_output_name will be quantized.
    The input shapes of the calibrator are resized with the calibration data.

    Returns:
      A quantized model.

    Args:
      dataset_gen: A generator that generates calibration samples.
      input_type: A tf.dtype representing the desired real-value input type.
      output_type: A tf.dtype representing the desired real-value output type.
      allow_float: A boolean. False if the resulting model cannot perform float
        computation, useful when targeting an integer-only backend. If False, an
        error will be thrown if an operation cannot be quantized, otherwise the
        model will fallback to float ops.
      op_output_name: A string, only this op will be quantized.
      resize_input: A boolean. True if the shape of the sample data is different
        from the input.
    """
    self._feed_tensors(dataset_gen, resize_input)
    return self._calibrator.QuantizeModel(
        np.dtype(input_type.as_numpy_dtype()).num,
        np.dtype(output_type.as_numpy_dtype()).num,
        allow_float,
        op_output_name,
    )

  @convert_phase(Component.OPTIMIZE_TFLITE_MODEL, SubComponent.CALIBRATE)
  def calibrate(self, dataset_gen):
    # [FUNCTION DOC]
    # Feeds samples from dataset_gen (callable producing samples), always resizing inputs, and
    # returns the model bytes with min/max calibration statistics recorded. Decorated to record
    # the calibrate conversion phase.
    """Calibrates the model with specified generator.

    Returns:
      A model with min and max calibration stats.

    Args:
      dataset_gen: A generator that generates calibration samples.
    """
    self._feed_tensors(dataset_gen, resize_input=True)
    return self._calibrator.Calibrate()
