# [MODULE DOC]
# HELPER: Thin wrapper layer over the native TFLite converter extension module.
# Exposes small functions that forward to the native converter API (convert, experimental MLIR
# quantize, sparsify, custom op definition registration, error retrieval, flat buffer to MLIR).
# Importing it first imports the main TensorFlow native bindings (required before the converter
# wrapper), the native converter module and the quantization Python function library, so the
# heavy native modules load only when this wrapper is imported (the lazy-loading arrangement).
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
"""Wraps TFLite Converter interface with python lazy loader."""
# We need to import pywrap_tensorflow prior to the converter wrapper.
# pylint: disable=invalid-import-order,g-bad-import-order
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.compiler.mlir.lite.python import _pywrap_converter_api
from tensorflow.compiler.mlir.quantization.tensorflow.python import py_function_lib


def wrapped_convert(
    model_flags_str,
    converter_flags_str,
    input_data_str,
    debug_info_str,
):
  # [FUNCTION DOC]
  # HELPER: Forwards a model conversion request to the native converter.
  # Parameters (all serialized strings/bytes): model_flags_str, converter_flags_str, input_data_str
  # (the model contents) and debug_info_str. Always passes extended-return as false and a new
  # Python function library object for quantization callbacks. Returns the native converter's
  # result (the converted model data); errors surface as exceptions from the native layer.
  """Wraps Convert with lazy loader."""
  return _pywrap_converter_api.Convert(
      model_flags_str,
      converter_flags_str,
      input_data_str,
      False,  # extended_return
      debug_info_str,
      py_function_lib.PyFunctionLibrary(),
  )


def wrapped_experimental_mlir_quantize(
    input_data_str,
    disable_per_channel,
    fully_quantize,
    inference_type,
    input_data_type,
    output_data_type,
    enable_numeric_verify,
    enable_whole_model_verify,
    denylisted_ops,
    denylisted_nodes,
    enable_variable_quantization,
    disable_per_channel_for_dense_layers,
    debug_options_str,
):
  # [FUNCTION DOC]
  # HELPER: Forwards an experimental MLIR quantization request to the native converter.
  # input_data_str: serialized model; disable_per_channel: turn off per-channel quantization;
  # fully_quantize: quantize all ops; inference_type, input_data_type, output_data_type: type
  # selections; enable_numeric_verify and enable_whole_model_verify: verification switches;
  # denylisted_ops and denylisted_nodes: names excluded from quantization; enable_variable_
  # quantization: also quantize variables; disable_per_channel_for_dense_layers: per-channel
  # opt-out for dense layers; debug_options_str: serialized debug options. All passed through
  # unchanged in the same order. Returns the native result (the quantized model data).
  """Wraps experimental mlir quantize model."""
  return _pywrap_converter_api.ExperimentalMlirQuantizeModel(
      input_data_str,
      disable_per_channel,
      fully_quantize,
      inference_type,
      input_data_type,
      output_data_type,
      enable_numeric_verify,
      enable_whole_model_verify,
      denylisted_ops,
      denylisted_nodes,
      enable_variable_quantization,
      disable_per_channel_for_dense_layers,
      debug_options_str,
  )


def wrapped_experimental_mlir_sparsify(input_data_str):
  # [FUNCTION DOC]
  # HELPER: Forwards a sparsification request. input_data_str: serialized model. Returns the
  # native result (the sparsified model data).
  """Wraps experimental mlir sparsify model."""
  return _pywrap_converter_api.ExperimentalMlirSparsifyModel(input_data_str)


def wrapped_register_custom_opdefs(custom_opdefs_list):
  # [FUNCTION DOC]
  # HELPER: Registers custom operation definitions with the native converter. custom_opdefs_list:
  # list of operation definition strings. Side effect: native global registration. Returns the
  # native result.
  """Wraps RegisterCustomOpdefs with lazy loader."""
  return _pywrap_converter_api.RegisterCustomOpdefs(custom_opdefs_list)


def wrapped_retrieve_collected_errors():
  # [FUNCTION DOC]
  # HELPER: Fetches the error messages collected by the native converter, e.g. after a failed
  # conversion. No parameters. Returns the native result (a collection of error strings).
  """Wraps RetrieveCollectedErrors with lazy loader."""
  return _pywrap_converter_api.RetrieveCollectedErrors()


def wrapped_flat_buffer_file_to_mlir(model, input_is_filepath):
  # [FUNCTION DOC]
  # HELPER: Converts a flat buffer model to MLIR text through the native converter. model: either
  # a file path or the model contents; input_is_filepath: true if model is a path. Returns the
  # native result (the MLIR representation as text).
  # KNOWN ISSUE: The docstring names the native function for file conversion, but the call goes
  # to the native function that converts flat buffers generally, so the name differs.
  """Wraps FlatBufferFileToMlir with lazy loader."""
  return _pywrap_converter_api.FlatBufferToMlir(model, input_is_filepath)
