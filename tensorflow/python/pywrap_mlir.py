# [MODULE DOC]
# HELPER: Python-friendly wrappers around the native MLIR bindings.
# Re-exports everything from the native MLIR extension and adds snake-case functions that encode
# text arguments to bytes and join list arguments into delimited strings before calling the
# native entry points: importing a graph definition or a concrete function into MLIR, converting
# saved models (v2, v1, v1 lite) to MLIR, running a pass pipeline on MLIR text, writing
# bytecode, and converting a TFLite flatbuffer to TOSA bytecode. Importing it first loads the
# core native TensorFlow library.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Python module for MLIR functions exported by pybind11."""

# pylint: disable=invalid-import-order, g-bad-import-order, wildcard-import, unused-import, undefined-variable
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.eager import context
from tensorflow.python._pywrap_mlir import *


def import_graphdef(
    graphdef,
    pass_pipeline,
    show_debug_info,
    input_names=None,
    input_data_types=None,
    input_data_shapes=None,
    output_names=[],
):
  # [FUNCTION DOC]
  # Imports a graph definition into MLIR and returns the resulting MLIR text.
  # Inputs: graphdef (converted to text then bytes), pass_pipeline (text of passes to run),
  # show_debug_info (boolean), and optionally input_names, input_data_types (lists of text, joined
  # with commas), input_data_shapes (list of text, joined with colons) and output_names (list of
  # text, default empty list; a shared mutable default, though not modified here).
  # If input_names is given, all of these are passed to the native importer; otherwise only the
  # graph, pipeline and debug flag are passed and the other optional arguments are ignored.
  if input_names is not None:
    return ImportGraphDef(
        str(graphdef).encode('utf-8'),
        pass_pipeline.encode('utf-8'),
        show_debug_info,
        ','.join(input_names).encode('utf-8'),
        ','.join(input_data_types).encode('utf-8'),
        ':'.join(input_data_shapes).encode('utf-8'),
        ','.join(output_names).encode('utf-8'),
    )
  return ImportGraphDef(
      str(graphdef).encode('utf-8'),
      pass_pipeline.encode('utf-8'),
      show_debug_info,
  )


def import_function(concrete_function, pass_pipeline, show_debug_info):
  # [FUNCTION DOC]
  # Imports a concrete function into MLIR and returns the MLIR text.
  # Inputs: concrete_function (its function definition is serialized to text), pass_pipeline
  # (text of passes), show_debug_info (boolean). Side effect: ensures the eager context is
  # initialized and passes its native handle to the importer.
  ctxt = context.context()
  ctxt.ensure_initialized()
  return ImportFunction(
      ctxt._handle,
      str(concrete_function.function_def).encode('utf-8'),
      pass_pipeline.encode('utf-8'),
      show_debug_info,
  )


def experimental_convert_saved_model_to_mlir(
    saved_model_path, exported_names, show_debug_info
):
  # [FUNCTION DOC]
  # Converts a SavedModel (v2 style) to MLIR text. Inputs: saved_model_path and exported_names
  # (both turned into text then bytes) and show_debug_info (boolean). Returns the native result.
  return ExperimentalConvertSavedModelToMlir(
      str(saved_model_path).encode('utf-8'),
      str(exported_names).encode('utf-8'),
      show_debug_info,
  )


def experimental_convert_saved_model_v1_to_mlir_lite(
    saved_model_path, exported_names, tags, upgrade_legacy, show_debug_info
):
  # [FUNCTION DOC]
  # Converts a v1 SavedModel to MLIR using the lite path. Inputs: saved_model_path, exported_names
  # and tags (each turned into text then bytes), upgrade_legacy (boolean to upgrade legacy ops)
  # and show_debug_info (boolean). Returns the native result.
  return ExperimentalConvertSavedModelV1ToMlirLite(
      str(saved_model_path).encode('utf-8'),
      str(exported_names).encode('utf-8'),
      str(tags).encode('utf-8'),
      upgrade_legacy,
      show_debug_info,
  )


def experimental_convert_saved_model_v1_to_mlir(
    saved_model_path,
    exported_names,
    tags,
    lift_variables,
    include_variables_in_initializers,
    upgrade_legacy,
    show_debug_info,
):
  # [FUNCTION DOC]
  # Converts a v1 SavedModel to MLIR. Inputs: saved_model_path, exported_names and tags (each
  # turned into text then bytes), lift_variables and include_variables_in_initializers (booleans
  # controlling variable handling), upgrade_legacy (boolean) and show_debug_info (boolean).
  # Returns the native result.
  return ExperimentalConvertSavedModelV1ToMlir(
      str(saved_model_path).encode('utf-8'),
      str(exported_names).encode('utf-8'),
      str(tags).encode('utf-8'),
      lift_variables,
      include_variables_in_initializers,
      upgrade_legacy,
      show_debug_info,
  )


def experimental_run_pass_pipeline(mlir_txt, pass_pipeline, show_debug_info):
  # [FUNCTION DOC]
  # Runs a textual pass pipeline over MLIR text. Inputs: mlir_txt and pass_pipeline (text, encoded
  # to bytes) and show_debug_info (boolean). Returns the native result (transformed MLIR).
  return ExperimentalRunPassPipeline(
      mlir_txt.encode('utf-8'), pass_pipeline.encode('utf-8'), show_debug_info
  )


def experimental_write_bytecode(filename, mlir_txt):
  # [FUNCTION DOC]
  # Writes MLIR text to a file as MLIR bytecode. Inputs: filename (text) and mlir_txt (text), both
  # encoded to bytes. Side effect: file written by the native layer. Returns the native result.
  return ExperimentalWriteBytecode(filename.encode('utf-8'), mlir_txt.encode())


def experimental_tflite_to_tosa_bytecode(
    flatbuffer,
    bytecode,
    use_external_constant=False,
    ordered_input_arrays=None,
    ordered_output_arrays=None,
):
  # [FUNCTION DOC]
  # Converts a TFLite flatbuffer to TOSA bytecode. Inputs: flatbuffer and bytecode (text values,
  # most likely file paths for input and output, encoded to bytes), use_external_constant
  # (boolean, default false), ordered_input_arrays and ordered_output_arrays (optional lists;
  # nothing is treated as an empty list). Returns the native result.
  if ordered_input_arrays is None:
    ordered_input_arrays = []
  if ordered_output_arrays is None:
    ordered_output_arrays = []
  return ExperimentalTFLiteToTosaBytecode(
      flatbuffer.encode('utf-8'),
      bytecode.encode('utf-8'),
      use_external_constant,
      ordered_input_arrays,
      ordered_output_arrays,
  )
