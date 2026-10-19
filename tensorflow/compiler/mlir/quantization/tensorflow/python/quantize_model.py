# [MODULE DOC]
# Public TF quantization API from SavedModel to SavedModel (exported as the experimental
# quantize-saved-model function). It fills in default quantization options, validates them, and
# dispatches to static-range (post-training with calibration, or quantization-aware-trained),
# dynamic-range, or weight-only quantization implemented by the native wrapper module. The
# resulting SavedModel is written to the output directory and loaded back.
# Import-time side effects: registers the quantization options, method, component spec and
# unit-wise spec message classes under the experimental quantization public API names, and the
# main function under its public name. Most functions here are private helpers of that API.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Defines TF Quantization API from SavedModel to SavedModel."""

import tempfile
from typing import Mapping, Optional

from absl import logging

from tensorflow.compiler.mlir.quantization.stablehlo import quantization_config_pb2 as stablehlo_quant_config_pb2
from tensorflow.compiler.mlir.quantization.tensorflow import quantization_options_pb2 as quant_opts_pb2
from tensorflow.compiler.mlir.quantization.tensorflow.python import py_function_lib
from tensorflow.compiler.mlir.quantization.tensorflow.python import pywrap_quantize_model
from tensorflow.compiler.mlir.quantization.tensorflow.python import representative_dataset as repr_dataset
from tensorflow.compiler.mlir.quantization.tensorflow.python import save_model
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.lib.io import file_io
from tensorflow.python.saved_model import load as saved_model_load
from tensorflow.python.saved_model import loader_impl as saved_model_loader
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.trackable import autotrackable
from tensorflow.python.util import tf_export

# Type aliases for quant_opts_pb2 messages.
_QuantizationOptions = tf_export.tf_export(
    'quantization.experimental.QuantizationOptions'
)(quant_opts_pb2.QuantizationOptions)

_QuantizationMethod = tf_export.tf_export(
    'quantization.experimental.QuantizationMethod'
)(quant_opts_pb2.QuantizationMethod)

_QuantizationComponentSpec = tf_export.tf_export(
    'quantization.experimental.QuantizationComponentSpec'
)(quant_opts_pb2.QuantizationComponentSpec)

_UnitWiseQuantizationSpec = tf_export.tf_export(
    'quantization.experimental.UnitWiseQuantizationSpec'
)(quant_opts_pb2.UnitWiseQuantizationSpec)

_PresetMethod = _QuantizationMethod.PresetMethod
_CalibrationMethod = (
    stablehlo_quant_config_pb2.CalibrationOptions.CalibrationMethod
)

_QuantizationComponent = _QuantizationComponentSpec.QuantizationComponent
_TensorType = _QuantizationComponentSpec.TensorType

_RepresentativeDatasetFile = quant_opts_pb2.RepresentativeDatasetFile

# Mapping of signature def key -> SignatureDef.
_SignatureDefMap = Mapping[str, meta_graph_pb2.SignatureDef]

# Default minimum number of elements in the weights for them to be quantized
# during dynamic range quantization (DRQ) and weight-only quantization.
_DYNAMIC_RANGE_DEFAULT_MIN_NUM_ELEMENTS_FOR_WEIGHTS = 1024


def _is_qat_saved_model(saved_model_path: str):
  # [FUNCTION DOC]
  # HELPER: detects quantization-aware-trained models for the quantization dispatcher.
  # Input: SavedModel directory path. Parses the SavedModel and returns true if any graph node,
  # or any node in any library function, has an op name starting with the fake-quantization
  # prefix; otherwise false. Reads files.
  """Checks if the SavedModel is QAT-enabled by looking for 'FakeQuant' ops."""
  saved_model_proto = saved_model_loader.parse_saved_model(saved_model_path)
  for meta_graph in saved_model_proto.meta_graphs:
    if any(
        node.op.startswith('FakeQuant') for node in meta_graph.graph_def.node
    ):
      return True
    for function in meta_graph.graph_def.library.function:
      if any(node.op.startswith('FakeQuant') for node in function.node_def):
        return True
  return False


def _serialize_signature_def_map(
    signature_def_map: _SignatureDefMap,
) -> dict[str, bytes]:
  # [FUNCTION DOC]
  # HELPER: prepares signature definitions for the native wrapper.
  # Input: mapping from signature key to signature definition message. Returns a new mapping from
  # key to the serialized bytes of each definition.
  """Serializes SignatureDef values in `signature_def_map`.

  Args:
    signature_def_map: Signature key -> SignatureDef mapping.

  Returns:
    Signature def map where the values (`SignatureDef`) are serialized.
  """
  signature_def_map_serialized = {}
  for key, signature_def in signature_def_map.items():
    signature_def_map_serialized[key] = signature_def.SerializeToString()

  return signature_def_map_serialized


def _save_representative_dataset(
    representative_dataset: repr_dataset.RepresentativeDatasetOrMapping,
    signature_def_map: _SignatureDefMap,
) -> Mapping[str, _RepresentativeDatasetFile]:
  # [FUNCTION DOC]
  # HELPER: converts a user-provided dataset into dataset files for calibration.
  # KNOWN ISSUE: each temporary file is created with a low-level call whose open file descriptor
  # is discarded and never closed (leak), and the temp files are never deleted here.
  # Inputs: representative_dataset (one dataset or mapping from signature key to dataset) and
  # signature_def_map (key to signature definition). If a mapping, its keys must equal the
  # signature keys (value error otherwise); if not a mapping, there must be exactly one
  # signature (value error if several), and the dataset is assigned to it. Creates a temporary
  # record file per signature (named with the key as prefix) and saves the data there, validating
  # sample keys against signature inputs. Returns a mapping from key to dataset file message.
  """Saves the representative dataset to temporary TFRecord files.

  Args:
    representative_dataset: Representative dataset used for the calibration
      step. Representative datasets should exist for each signature def key in
      `signature_def_keys`.
    signature_def_map: Signature def key -> SignatureDef mapping.

  Returns:
    A map from signature key to the saved representative dataset file.
  """
  if isinstance(representative_dataset, Mapping):
    if set(signature_def_map.keys()) != set(representative_dataset.keys()):
      raise ValueError(
          'The signature keys and the keys of representative dataset map '
          f'do not match. Signature keys: {set(signature_def_map.keys())}, '
          f'representative dataset map: {set(representative_dataset.keys())}.'
      )
    representative_dataset_map = representative_dataset
  elif len(signature_def_map.keys()) > 1:
    raise ValueError(
        'Representative dataset is not a mapping (got: '
        f'{type(representative_dataset)}), but there is more than one '
        'signature key provided. Please provide a map of '
        '{signature_key -> dataset} with more than one signature key.'
    )
  else:
    representative_dataset_map = {
        list(signature_def_map.keys())[0]: representative_dataset,
    }

  # Save the representative dataset to temporary TFRecord files.
  path_map = {}
  expected_input_key_map = {}
  for signature_key, signature_def in signature_def_map.items():
    # Filepath is the second return value of mkstemp.
    _, path_map[signature_key] = tempfile.mkstemp(
        suffix='.tfrecord', prefix=signature_key
    )
    expected_input_key_map[signature_key] = signature_def.inputs.keys()

  return repr_dataset.TfRecordRepresentativeDatasetSaver(
      path_map=path_map,
      expected_input_key_map=expected_input_key_map,
  ).save(representative_dataset_map)


def _run_static_range_qat(
    src_saved_model_path: str,
    dst_saved_model_path: str,
    quant_opts: _QuantizationOptions,
    signature_def_map: _SignatureDefMap,
) -> None:
  # [FUNCTION DOC]
  # HELPER: QAT branch of static-range quantization.
  # Inputs: source and destination SavedModel paths, quantization options message, and signature
  # definition mapping. Logs, then calls the native QAT quantizer with serialized options, the
  # option's signature keys, serialized signature definitions and a new Python function library.
  # Returns nothing; writes the destination model.
  """Runs static-range quantization for a Quantization-Aware Trained model.

  Runs the quantization for a model trained using QAT.

  Args:
    src_saved_model_path: Path to the source SavedModel directory.
    dst_saved_model_path: Path to the destination SavedModel directory.
    quant_opts: Quantization options.
    signature_def_map: Signature def key -> SignatureDef mapping.
  """
  logging.info('Running static-range quantization for QAT model.')

  pywrap_quantize_model.quantize_qat_model(
      src_saved_model_path,
      dst_saved_model_path,
      quantization_options_serialized=quant_opts.SerializeToString(),
      signature_keys=list(quant_opts.signature_keys),
      signature_def_map_serialized=_serialize_signature_def_map(
          signature_def_map
      ),
      py_function_library=py_function_lib.PyFunctionLibrary(),
  )


def _run_static_range_ptq(
    src_saved_model_path: str,
    dst_saved_model_path: str,
    quant_opts: _QuantizationOptions,
    representative_dataset: Mapping[str, _RepresentativeDatasetFile],
    signature_def_map: _SignatureDefMap,
) -> None:
  # [FUNCTION DOC]
  # HELPER: post-training branch of static-range quantization.
  # KNOWN ISSUE: docstring mentions raising a value error for an invalid signature; this function
  # raises nothing itself (errors can only come from the native call).
  # Inputs: source and destination paths, options message, mapping from signature key to dataset
  # file message, and signature definition mapping. Serializes the definitions and dataset files,
  # then calls the native post-training static-range quantizer (which runs calibration through
  # the Python function library). Returns nothing; writes the destination model.
  """Runs static-range Post-Training Quantization.

  Runs static-range PTQ for the model. Runs the calibration step with
  `representative_dataset` to collect statistics required for quantization. This
  produces the quantized GraphDef along with the SignatureDefs which might have
  been modified according to the changes in the graph.

  Args:
    src_saved_model_path: Path to the source SavedModel directory.
    dst_saved_model_path: Path to the destination SavedModel directory.
    quant_opts: Quantization options.
    representative_dataset: A map from signature key to the saved representative
      dataset file.
    signature_def_map: Signature def key -> SignatureDef mapping.

  Raises:
    ValueError if the graph doesn't contain a valid signature.
  """
  logging.info('Running static-range post-training quantization.')

  signature_def_map_serialized = _serialize_signature_def_map(signature_def_map)

  # `quantize_ptq_static_range` requires `RepresentativeDatasetFile`s to be
  # serialized. Serialize the values to match the type.
  dataset_file_map_serialized = {
      signature_key: dataset_file.SerializeToString()
      for signature_key, dataset_file in representative_dataset.items()
  }
  pywrap_quantize_model.quantize_ptq_static_range(
      src_saved_model_path,
      dst_saved_model_path,
      quantization_options_serialized=quant_opts.SerializeToString(),
      signature_keys=list(quant_opts.signature_keys),
      signature_def_map_serialized=signature_def_map_serialized,
      py_function_library=py_function_lib.PyFunctionLibrary(),
      representative_dataset_file_map_serialized=dataset_file_map_serialized,
  )


def _static_range_quantize(
    src_saved_model_path: str,
    dst_saved_model_path: str,
    quantization_options: _QuantizationOptions,
    representative_dataset: Optional[
        repr_dataset.RepresentativeDatasetOrMapping
    ] = None,
) -> autotrackable.AutoTrackable:
  # [FUNCTION DOC]
  # HELPER: static-range branch of the public quantization API.
  # KNOWN ISSUE: the warning about min_num_elements_for_weights fires whenever it is positive,
  # which is always the case after defaults are applied by the public entry point; also the
  # docstring's runtime error is never raised here.
  # Inputs: source and destination paths, options message, and optional representative dataset.
  # Detects QAT models or the no-quantize method, reads signatures, raises a value error if a
  # dataset is given both as argument and in the options, or if none is available for a non-QAT
  # model. Saves a provided dataset to temporary files, then runs the QAT or post-training path.
  # Returns the quantized SavedModel loaded from the destination.
  """Quantizes the given SavedModel via static range quantization.

  If the model is not trained with Quantization-Aware Training (QAT) technique,
  it requires `representative_dataset` to collect statistics required for
  quantization. If non-None `representative_dataset` is provided with a QAT
  model input, `representative_dataset` will be ignored.

  Args:
    src_saved_model_path: Path to the saved model. When representative_dataset
      is not provided, this should be a model trained with QAT.
    dst_saved_model_path: The path to save the output SavedModel. The directory
      will be overwritten if not empty.
    quantization_options: QuantizationOptions proto describing quantization
      related config.
    representative_dataset: a generator that returns a dictionary in {input_key:
      input_value} format or a tuple with signature key and a dictionary in
      {input_key: input_value} format that feeds calibration data for quantizing
      model. This should be provided when the model is not a QAT model.

  Returns:
    A SavedModel object with TF quantization applied.

  Raises:
    ValueError: when representative_dataset is not provided for non-QAT model.
    RuntimeError: When a MetaGraphDef could not be found associated with `tags`
      in the SavedModel.
  """
  logging.info(
      'Running static range quantization on model: %s', src_saved_model_path
  )
  logging.info('QuantizationOptions: \n%s', quantization_options)

  is_qat_saved_model_or_method_no_quantize = _is_qat_saved_model(
      src_saved_model_path
  ) or (
      quantization_options.quantization_method.preset_method
      == _QuantizationMethod.METHOD_NO_QUANTIZE
  )
  signature_def_map = save_model.get_signatures_from_saved_model(
      src_saved_model_path,
      quantization_options.signature_keys,
      set(quantization_options.tags),
  )

  if (
      representative_dataset is not None
      and quantization_options.representative_datasets
  ):
    raise ValueError(
        'Do not specify both the `representative_dataset` argument and'
        ' the `representative_datasets` field in `QuantizationOptions`.'
    )

  saved_representative_dataset = quantization_options.representative_datasets
  if representative_dataset is not None:
    saved_representative_dataset = _save_representative_dataset(
        representative_dataset, signature_def_map
    )

  # Checks if the model is from QAT or method is METHOD_NO_QUANTIZE.
  if (
      not saved_representative_dataset
      and not is_qat_saved_model_or_method_no_quantize
  ):
    raise ValueError(
        'When `representative_dataset` is not provided, the model should be '
        'trained with quantization-aware training (QAT).'
    )
  if quantization_options.min_num_elements_for_weights > 0:
    logging.warn(
        'min_num_elements_for_weights is set but is not supported for the '
        'Post-training static range quantization. '
        'The flag is ignored.'
    )

  if is_qat_saved_model_or_method_no_quantize:
    _run_static_range_qat(
        src_saved_model_path,
        dst_saved_model_path,
        quantization_options,
        signature_def_map,
    )
  else:
    _run_static_range_ptq(
        src_saved_model_path,
        dst_saved_model_path,
        quantization_options,
        saved_representative_dataset,
        signature_def_map,
    )

  return saved_model_load.load(dst_saved_model_path)


def _dynamic_range_quantize(
    src_saved_model_path: str,
    dst_saved_model_path: str,
    quantization_options: _QuantizationOptions,
) -> autotrackable.AutoTrackable:
  # [FUNCTION DOC]
  # HELPER: dynamic-range branch of the public quantization API.
  # Inputs: source and destination paths and options message. Raises a value error for QAT
  # models. Reads signatures, calls the native dynamic-range quantizer, and returns the quantized
  # SavedModel loaded from the destination. Writes files.
  """Quantizes the given SavedModel via post-training dynamic range quantization.

  Args:
    src_saved_model_path: Path to the saved model.
    dst_saved_model_path: The path to save the output SavedModel. The directory
      will be overwritten if not empty.
    quantization_options: QuantizationOptions proto describing quantization
      related config.

  Returns:
    A SavedModel object with TF quantization applied.

  Raises:
    ValueError: when the model is QAT model.
  """
  mode_str = 'dynamic-range quantization'
  if _is_qat_saved_model(src_saved_model_path):
    raise ValueError(
        'The models trained with quantization-aware training (QAT) is not '
        'supported for %s.' % mode_str
    )

  logging.info(
      'Running post-training %s on model: %s', mode_str, src_saved_model_path
  )
  logging.info('QuantizationOptions: \n%s', quantization_options)

  signature_def_map = save_model.get_signatures_from_saved_model(
      src_saved_model_path,
      quantization_options.signature_keys,
      quantization_options.tags,
  )

  # Apply post-training dynamic range quantization to the model.
  pywrap_quantize_model.quantize_ptq_dynamic_range(
      src_saved_model_path,
      dst_saved_model_path,
      quantization_options_serialized=quantization_options.SerializeToString(),
      signature_keys=list(quantization_options.signature_keys),
      signature_def_map_serialized=_serialize_signature_def_map(
          signature_def_map
      ),
      py_function_library=py_function_lib.PyFunctionLibrary(),
  )

  return saved_model_load.load(dst_saved_model_path)


def _weight_only_quantize(
    src_saved_model_path: str,
    dst_saved_model_path: str,
    quantization_options: quant_opts_pb2.QuantizationOptions,
) -> autotrackable.AutoTrackable:
  # [FUNCTION DOC]
  # HELPER: weight-only branch of the public quantization API.
  # Inputs: source and destination paths and options message. Raises a value error for QAT
  # models. Reads signatures, calls the native weight-only quantizer, and returns the quantized
  # SavedModel loaded from the destination. Writes files.
  """Quantizes the given SavedModel via weight-only quantization.

  Args:
    src_saved_model_path: Path to the saved model.
    dst_saved_model_path: The path to save the output SavedModel. The directory
      will be overwritten if not empty.
    quantization_options: QuantizationOptions proto describing quantization
      related config.

  Returns:
    A SavedModel object with TF quantization applied.

  Raises:
    ValueError: when the model is QAT model.
  """
  mode_str = 'weight-only quantization'

  # QAT weight-only is not supported yet.
  if _is_qat_saved_model(src_saved_model_path):
    raise ValueError(
        'The models trained with quantization-aware training (QAT) is not '
        'supported for %s.' % mode_str
    )

  logging.info(
      'Running post-training %s on model: %s', mode_str, src_saved_model_path
  )
  logging.info('QuantizationOptions: \n%s', quantization_options)

  signature_def_map = save_model.get_signatures_from_saved_model(
      src_saved_model_path,
      list(quantization_options.signature_keys),
      set(quantization_options.tags),
  )

  pywrap_quantize_model.quantize_weight_only(
      src_saved_model_path,
      dst_saved_model_path,
      quantization_options_serialized=quantization_options.SerializeToString(),
      signature_def_map_serialized=_serialize_signature_def_map(
          signature_def_map
      ),
      py_function_library=py_function_lib.PyFunctionLibrary(),
  )

  return saved_model_load.load(dst_saved_model_path)


def _verify_output_dir(output_dir: Optional[str], overwrite: bool) -> None:
  # [FUNCTION DOC]
  # HELPER: pre-check of the output location for the public quantization API.
  # Inputs: output_dir (path or null) and overwrite flag. Raises a file-exists error if the
  # directory exists, is non-empty and overwrite is false. A null path is accepted. Creates and
  # modifies nothing. Returns nothing.
  """Verifies the output directory.

  Raises an error if `output_dir` is not suitable for writing the output saved
  model.

  Args:
    output_dir: Output directory.
    overwrite: An option allowing to overwrite the existing output directory if
      set to true. Does not actually create or modify the `output_dir` in this
      function.

  Raises:
    FileExistsError: Iff `output_dir` is not empty and `overwrite` is false.
  """
  dir_not_empty = (
      output_dir is not None
      and file_io.file_exists_v2(output_dir)
      and file_io.list_directory_v2(output_dir)
  )

  if dir_not_empty and not overwrite:
    raise FileExistsError(
        f'Output directory already exists: {output_dir} . '
        'Please set overwrite_output_directory to true to '
        'overwrite the existing directory.'
    )


def _populate_quantization_component_spec(
    quant_method: _QuantizationMethod,
) -> None:
  # [FUNCTION DOC]
  # HELPER: fills component specs of a quantization method message (modified in place).
  # KNOWN ISSUE: the final checks raise value errors with misleading messages (the first passes
  # the method as a separate argument; the second says "at least one" while requiring exactly
  # one), and the typo "componet" appears in comments.
  # Input: quantization method message. Builds defaults per preset: static or dynamic range int8
  # gives int8 activation, int8 weight and int32 bias; weight-only int8 gives int8 weight.
  # User specs override defaults per component; weight and activation must be int8 and others
  # (bias) int32, else a value error. Replaces the message's spec list with the result and
  # verifies the count (3 for static/dynamic range, 1 for weight-only), raising a value error.
  """Populates default values for QuantizationComponentSpec.

  Args:
    quant_method: The quantization method to be updated.
  """
  # Make sure creating one spec per component.
  updated_component_spec = dict()

  # Populate default configuration.
  if (
      quant_method.preset_method == _PresetMethod.METHOD_STATIC_RANGE_INT8
      or quant_method.preset_method == _PresetMethod.METHOD_DYNAMIC_RANGE_INT8
  ):
    updated_component_spec[_QuantizationComponent.COMPONENT_ACTIVATION] = (
        _QuantizationComponentSpec(
            quantization_component=_QuantizationComponent.COMPONENT_ACTIVATION,
            tensor_type=_TensorType.TENSORTYPE_INT_8,
        )
    )
    updated_component_spec[_QuantizationComponent.COMPONENT_WEIGHT] = (
        _QuantizationComponentSpec(
            quantization_component=_QuantizationComponent.COMPONENT_WEIGHT,
            tensor_type=_TensorType.TENSORTYPE_INT_8,
        )
    )
    updated_component_spec[_QuantizationComponent.COMPONENT_BIAS] = (
        _QuantizationComponentSpec(
            quantization_component=_QuantizationComponent.COMPONENT_BIAS,
            tensor_type=_TensorType.TENSORTYPE_INT_32,
        )
    )
  elif (
      quant_method.preset_method
      == _PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
  ):
    updated_component_spec[_QuantizationComponent.COMPONENT_WEIGHT] = (
        _QuantizationComponentSpec(
            quantization_component=_QuantizationComponent.COMPONENT_WEIGHT,
            tensor_type=_TensorType.TENSORTYPE_INT_8,
        )
    )

  # Override if quantization_component_spec is specified.
  if quant_method.quantization_component_specs:
    # Check if the component spec is supported configuration in TF-Quant.
    for component_spec in quant_method.quantization_component_specs:
      if component_spec.quantization_component in [
          _QuantizationComponent.COMPONENT_WEIGHT,
          _QuantizationComponent.COMPONENT_ACTIVATION,
      ]:
        if component_spec.tensor_type != _TensorType.TENSORTYPE_INT_8:
          raise ValueError(
              'Only int8 precision is supported for input operands.'
          )
      else:
        if component_spec.tensor_type != _TensorType.TENSORTYPE_INT_32:
          raise ValueError('Only int32 precision is supported for bias.')
      # Update with the custom spec.
      updated_component_spec[component_spec.quantization_component] = (
          component_spec
      )

  # Update the componet spec
  del quant_method.quantization_component_specs[:]
  quant_method.quantization_component_specs.extend(
      updated_component_spec.values()
  )

  if (
      quant_method.preset_method == _PresetMethod.METHOD_STATIC_RANGE_INT8
      or quant_method.preset_method == _PresetMethod.METHOD_DYNAMIC_RANGE_INT8
  ) and (len(quant_method.quantization_component_specs) != 3):
    raise ValueError('Only 3 components are needed for', quant_method)
  elif (
      quant_method.preset_method
      == _PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
  ) and len(quant_method.quantization_component_specs) != 1:
    raise ValueError('At least one component spec needs to be specified.')


def _populate_unitwise_quantization_specs(
    quantization_options: _QuantizationOptions,
) -> None:
  # [FUNCTION DOC]
  # HELPER: validation step for unit-wise quantization settings.
  # Input: quantization options message (modified in place). Does nothing if there are no
  # unit-wise specs. For each spec: raises a value error if it has no units or a unit has neither
  # op type nor node name; fills its component specs via the component-spec helper; and, if it has
  # component specs that differ (after sorting by component) from the top-level method's, raises
  # a value error since only the same method as the top level is supported.
  """Verifies and pupulates unitwise quantization specs."""
  if not quantization_options.unit_wise_quantization_specs:
    return

  sorted_top_level_component_specs = sorted(
      quantization_options.quantization_method.quantization_component_specs,
      key=lambda x: x.quantization_component,
  )

  for unitwise_spec in quantization_options.unit_wise_quantization_specs:
    if not unitwise_spec.unit:
      raise ValueError(
          'UnitWiseQuantizationSpec must contain at least one unit.'
      )

    for unit in unitwise_spec.unit:
      if not unit.op_type and not unit.node_name:
        raise ValueError('Either `op_type` or `node_name` must be specified.')

    _populate_quantization_component_spec(unitwise_spec.quantization_method)

    component_specs = (
        unitwise_spec.quantization_method.quantization_component_specs
    )
    if component_specs and (
        sorted_top_level_component_specs
        != sorted(component_specs, key=lambda x: x.quantization_component)
    ):
      raise ValueError(
          'Currently unit-wise quantization spec only supports NO_QUANTIZE and'
          ' same quantization method as the top-level `quantization_method`'
      )


def _populate_calibration_options(
    quantization_options: quant_opts_pb2.QuantizationOptions,
):
  # [FUNCTION DOC]
  # HELPER: fills calibration defaults (options modified in place).
  # KNOWN ISSUE: for histogram MSE methods the activation tensor type is read by indexing the
  # repeated component spec list with the activation component enum value, i.e. by position, not
  # by looking up the spec for the activation component; this can pick the wrong spec or fail.
  # Unspecified method becomes min-max. Histogram percentile gets defaults of 512 bins, 0.001 min
  # percentile and 99.999 max percentile when unset. Histogram MSE methods require int8
  # activation (value error otherwise) and default to 512 bins. If a calibration data directory
  # is set, creates it empty (overwriting only when regeneration is forced). Returns nothing.
  """Populates default values for CalibrationOptions.

  Args:
    quantization_options: An instance of QuantizationOptions with a field
      specifying CalibrationOptions
  """

  calib_opts = quantization_options.calibration_options
  if (
      calib_opts.calibration_method
      == _CalibrationMethod.CALIBRATION_METHOD_UNSPECIFIED
  ):
    calib_opts.calibration_method = (
        _CalibrationMethod.CALIBRATION_METHOD_MIN_MAX
    )
  elif (
      calib_opts.calibration_method
      == _CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_PERCENTILE
  ):
    if not calib_opts.calibration_parameters.num_bins:
      calib_opts.calibration_parameters.num_bins = 512
    if not calib_opts.calibration_parameters.min_percentile:
      calib_opts.calibration_parameters.min_percentile = 0.001
    if not calib_opts.calibration_parameters.max_percentile:
      calib_opts.calibration_parameters.max_percentile = 99.999
  # Check the activation_tensor_type of HISTOGRAM_MSE methods.
  elif calib_opts.calibration_method in [
      _CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE,
      _CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY,
      _CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_SYMMETRIC,
  ]:
    activation_tensor_type = (
        quantization_options.quantization_method.quantization_component_specs[
            _QuantizationComponent.COMPONENT_ACTIVATION
        ].tensor_type
    )
    # Unlike the HISTOGRAM_PERCENTILE method, the HISTOGRAM_MSE method uses
    # num_bits because it actually quantizes and dequantizes values.
    if activation_tensor_type != _TensorType.TENSORTYPE_INT_8:
      raise ValueError(
          'Only TENSORTYPE_INT_8 is supported for HISTOGRAM_MSE calibration'
          f' methods. calibration_method={calib_opts.calibration_method}'
      )

    if not calib_opts.calibration_parameters.num_bins:
      calib_opts.calibration_parameters.num_bins = 512

  if calib_opts.calibration_data_dir:
    save_model.create_empty_output_dir(
        calib_opts.calibration_data_dir,
        overwrite=calib_opts.force_regenerate_calibration_data,
    )


def _populate_quantization_options_default_values(
    quantization_options: _QuantizationOptions,
) -> None:
  # [FUNCTION DOC]
  # HELPER: normalizes and validates the options message before quantization (modified in place).
  # KNOWN ISSUE: docstring says the unspecified op set defaults to XLA and lists only a
  # per-channel check, but the code also forces graph-mode calibration to true unconditionally,
  # sets a debugger log directory default, and applies several other validations. One error
  # message has a missing space ("quantizationvia").
  # Defaults: XLA op set, serving tag, default serving signature key, freeze variables true,
  # static range int8 method, minimum weight elements 1024 (with a warning), per-channel false.
  # Raises value errors for: legacy weight-only flag, per-channel with unsupported op set or
  # method, weight-only with TF or uniform-quantized op set, StableHLO with methods other than
  # static range or weight-only, debugger enabled without a type, whole-model debugger without
  # an unquantized model path. Then fills component specs, validates unit-wise specs and, for
  # static range int8, fills calibration options (which may create a directory).
  """Populates default values for QuantizationOptions.

  Populates unspecified or unset fields of QuantizationOptions with the default
  values.

  * If `op_set` is unspecified, it defaults to `OpSet.XLA`.
  * If `freeze_all_variables` is not set, it defaults to `True`.
  * Check if configurations are set correctly:
    - Per-channel quantization is supported for Uniform Quantized opset only.

  Args:
    quantization_options: An instance of QuantizationOptions.
  """
  if quantization_options.op_set == quant_opts_pb2.OpSet.OP_SET_UNSPECIFIED:
    quantization_options.op_set = quant_opts_pb2.OpSet.XLA

  if not quantization_options.tags:
    quantization_options.tags.append(tag_constants.SERVING)

  if not quantization_options.signature_keys:
    quantization_options.signature_keys.append(
        signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    )

  if not quantization_options.HasField('freeze_all_variables'):
    quantization_options.freeze_all_variables = True

  if quantization_options.enable_legacy_weight_only:
    raise ValueError(
        'Legacy weight-only is deprecated. Use weight-only quantization method.'
    )

  # Converter assumes options are specified. So set SRQ explicitly.
  if (
      quantization_options.quantization_method.preset_method
      == _PresetMethod.METHOD_UNSPECIFIED
  ):
    logging.debug(
        '"preset_method" for QuantizationMethod is not specified.'
        'Static range quantization is used by default.'
    )
    quantization_options.quantization_method.preset_method = (
        _PresetMethod.METHOD_STATIC_RANGE_INT8
    )

  # Check default quantization option values for weight-only quantization.
  # TODO(b/242805842): Find good minimum_elements_for_weights number for server.
  # please also update default value in tflite converter:
  # tensorflow/compiler/mlir/lite/tf_to_tfl_flatbuffer.cc;l=201
  if quantization_options.min_num_elements_for_weights == 0:
    quantization_options.min_num_elements_for_weights = (
        _DYNAMIC_RANGE_DEFAULT_MIN_NUM_ELEMENTS_FOR_WEIGHTS
    )
    logging.warning(
        (
            'QuantizationOptions.min_num_elements_for_weights is not set (0).'
            ' Setting to the default value: %d.'
        ),
        _DYNAMIC_RANGE_DEFAULT_MIN_NUM_ELEMENTS_FOR_WEIGHTS,
    )

  if not quantization_options.HasField('enable_per_channel_quantization'):
    quantization_options.enable_per_channel_quantization = False

  if quantization_options.enable_per_channel_quantization and not (
      (
          quantization_options.op_set == quant_opts_pb2.OpSet.UNIFORM_QUANTIZED
          or quantization_options.quantization_method.preset_method
          == _PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
      )
      or (
          quantization_options.op_set
          in (quant_opts_pb2.OpSet.XLA, quant_opts_pb2.OpSet.STABLEHLO)
          and quantization_options.quantization_method.preset_method
          == _PresetMethod.METHOD_STATIC_RANGE_INT8
      )
  ):
    raise ValueError(
        'Currently, per-channel quantization is supported for Uniform Quantized'
        ' opset, weight only quantization, or XLA/StableHLO opset with static'
        ' range quantization.'
    )

  if (
      quantization_options.quantization_method.preset_method
      == _PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
      and (
          quantization_options.op_set == quant_opts_pb2.OpSet.UNIFORM_QUANTIZED
          or quantization_options.op_set == quant_opts_pb2.OpSet.TF
      )
  ):
    raise ValueError('TF/Uniform quantized opset does not support weight-only.')

  if (quantization_options.op_set == quant_opts_pb2.OpSet.STABLEHLO) and (
      quantization_options.quantization_method.preset_method
      != _PresetMethod.METHOD_STATIC_RANGE_INT8
      and quantization_options.quantization_method.preset_method
      != _PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
  ):
    raise ValueError(
        'StableHLO quantized opset currently only supports static range'
        ' quantization and weight-only quantizationvia TF Quantizer.'
    )

  # Set `force_graph_mode_calibration` to True to avoid skipping op execution,
  # which are not connected to return ops, during calibration execution.
  # TODO: b/335031954 - Bring back support to run calibration in Eager mode.
  logging.debug(
      'Setting `force_graph_mode_calibration = True` to ensure the calibration'
      ' mode is executed properly.'
  )
  quantization_options.force_graph_mode_calibration = True

  if quantization_options.HasField('debugger_config'):
    if not quantization_options.debugger_config.log_dir_path:
      quantization_options.debugger_config.log_dir_path = '/tmp/dumps'

    if (
        quantization_options.debugger_config.debugger_type
        == stablehlo_quant_config_pb2.DebuggerConfig.DebuggerType.DEBUGGER_TYPE_UNSPECIFIED
    ):
      raise ValueError(
          'Debugger is enabled but debugger type was not specified.'
      )

    if (
        quantization_options.debugger_config.debugger_type
        == stablehlo_quant_config_pb2.DebuggerConfig.DebuggerType.DEBUGGER_TYPE_WHOLE_MODEL
        and not quantization_options.debugger_config.unquantized_dump_model_path
    ):
      raise ValueError(
          'Debugger type whole model verify was used but'
          ' unquantized_dump_model_path was not specified.'
      )

  # Check and populate quantization component spec.
  _populate_quantization_component_spec(
      quantization_options.quantization_method
  )
  # Verify and populate unit-wise quantization specs.
  _populate_unitwise_quantization_specs(quantization_options)

  if (
      quantization_options.quantization_method.preset_method
      == _PresetMethod.METHOD_STATIC_RANGE_INT8
  ):
    # Check and populate calibration options.
    _populate_calibration_options(quantization_options)


@tf_export.tf_export('quantization.experimental.quantize_saved_model')
def quantize(
    saved_model_path: str,
    output_directory: Optional[str] = None,
    quantization_options: Optional[_QuantizationOptions] = None,
    representative_dataset: Optional[
        repr_dataset.RepresentativeDatasetOrMapping
    ] = None,
    *,
    overwrite_output_directory: bool = False,
) -> autotrackable.AutoTrackable:
  # [FUNCTION DOC]
  # KNOWN ISSUE: docstring says it may return none if no quantization is performed, but every
  # path returns a loaded model or raises. The unsupported-method error message lacks the
  # formatting prefix, so it prints the placeholder text instead of the method value. The
  # output-directory check runs before the default temp directory is created.
  # Public entry point (exported). Inputs: saved model path; optional output directory (a new
  # temporary directory if null); optional options message (defaults created if null; mutated
  # in place with defaults); optional representative dataset; keyword-only overwrite flag
  # (default false). Verifies the output directory, fills defaults, then dispatches on the
  # preset method: static range int8 or no-quantize to static range, dynamic range int8, or
  # static-range weight-only int8; other methods raise a value error. Returns the quantized
  # SavedModel loaded from the output directory. Writes files.
  """Quantizes the SavedModel with the given quantization options.

  Example usage:
  ```python
  # Quantizing a model trained with QAT.
  quantization_options = tf.quantization.experimental.QuantizationOptions(
      signature_keys=['your_signature_key'],
  )
  tf.quantization.experimental.quantize_saved_model(
      '/tmp/input_model',
      '/tmp/output_model',
      quantization_options=quantization_options,
  )

  # When quantizing a model trained without QAT (Post-Training Quantization),
  # a representative dataset is required.
  representative_dataset = [{"input": tf.random.uniform(shape=(3, 3))}
                        for _ in range(256)]
  tf.quantization.experimental.quantize_saved_model(
      '/tmp/input_model',
      '/tmp/output_model',
      quantization_options=quantization_options,
      representative_dataset={'your_signature_key': representative_dataset},
    )

  # In addition to preset quantization methods, fine-grained control of
  # quantization for each component is also supported.
  _QuantizationComponentSpec = (
      tf.quantization.experimental.QuantizationComponentSpec
  )
  quantization_options = tf.quantization.experimental.QuantizationOptions(
      signature_keys=['your_signature_key'],
      quantization_method=tf.quantization.experimental.QuantizationMethod(
          quantization_component_specs=[
              _QuantizationComponentSpec(
                  quantization_component=(
                      _QuantizationComponentSpec.COMPONENT_ACTIVATION
                  ),
                  tensor_type=_QuantizationComponentSpec.TENSORTYPE_INT_8,
              )
          ]
      )
  )
  tf.quantization.experimental.quantize_saved_model(
      '/tmp/input_model',
      '/tmp/output_model',
      quantization_options=quantization_options,
  )
  ```

  Args:
    saved_model_path: Path to the saved model. When representative_dataset is
      not provided, this should be a model trained with QAT.
    output_directory: The path to save the output SavedModel. Set
      `overwrite_output_directory` to `True` to overwrite any existing contents
      in the directory if not empty.
    quantization_options: A set of options for quantization. If None, it uses
      post-training static range quantization with XLA opset by default.
    representative_dataset: an iterator that returns a dictionary of {input_key:
      input_value} or a map from signature key to a dictionary of {input_key:
      input_value} that feeds calibration data for quantizing model. The
      representative should be provided when the model is a PTQ model. It can be
      provided either via this parameter or via the `representative_datasets`
      field in `QuantizationOptions`.
    overwrite_output_directory: If set to true, overwrites the output directory
      iff it isn't empty. The default value is false.

  Returns:
    A SavedModel object with TF quantization applied, or None if no quantization
    is performed.

  Raises:
    ValueError: When 1) representative_dataset is not provided for non QAT model
      for enabling static range quantization, 2) invalid value is provided as
      a quantization method, or 3) provide representative dataset via both
      argument and QuantizationOptions.
    ValueError: When the specified quantization method is not yet supported.
  """
  _verify_output_dir(output_directory, overwrite_output_directory)

  # Set default values for None arguments.
  if output_directory is None:
    output_directory = tempfile.mkdtemp()

  if quantization_options is None:
    quantization_options = _QuantizationOptions()

  _populate_quantization_options_default_values(quantization_options)

  method: _QuantizationMethod = quantization_options.quantization_method
  if (
      method.preset_method == _PresetMethod.METHOD_STATIC_RANGE_INT8
      or method.preset_method == _PresetMethod.METHOD_NO_QUANTIZE
  ):
    return _static_range_quantize(
        saved_model_path,
        output_directory,
        quantization_options,
        representative_dataset,
    )
  elif method.preset_method == _PresetMethod.METHOD_DYNAMIC_RANGE_INT8:
    return _dynamic_range_quantize(
        saved_model_path,
        output_directory,
        quantization_options,
    )
  elif (
      method.preset_method == _PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
  ):
    return _weight_only_quantize(
        saved_model_path,
        output_directory,
        quantization_options,
    )
  else:
    raise ValueError(
        'Quantization method {method.preset_method} is not supported.'
    )
