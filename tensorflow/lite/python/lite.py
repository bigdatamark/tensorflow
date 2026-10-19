# [MODULE DOC]
# Main public entry point of the TensorFlow Lite Python converter. Defines the user-facing
# configuration types (Optimize enum, RepresentativeDataset, TargetSpec), the QuantizationMode
# helper that decides which quantization scheme the user options imply, the converter base class
# and the concrete converters for TF2 (SavedModel, Keras, concrete functions, JAX) and for the
# legacy TF1 flows (SavedModel, session, frozen graph, Keras file), plus a deprecated TOCO
# converter alias. Many names are exported to the public tf.lite namespace through the export
# decorator, and several symbols from sibling modules (interpreter, op hints, debugger,
# converter errors, op sets) are re-exported here for backward compatibility.
# Import-time side effects: registers the public API exports and imports the microfrontend op
# and profiling protos so they are available in the pip package. No flags are defined.
#
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""TensorFlow Lite tooling helper functionality."""

import enum
import functools
import gc
import pprint
import shutil
import sys
import tempfile
import time
import warnings

from absl import logging

from google.protobuf import text_format as _text_format
from google.protobuf.message import DecodeError
from tensorflow.compiler.mlir.quantization.stablehlo import quantization_config_pb2 as qc
from tensorflow.compiler.mlir.quantization.tensorflow.python import representative_dataset as rd
from tensorflow.core.framework import graph_pb2 as _graph_pb2
from tensorflow.lite.experimental.microfrontend.python.ops import audio_microfrontend_op  # pylint: disable=unused-import
# The following imports are needed to make the model_runtime_info_pb2
# and profiling_info_pb2 protos available via the litert PIP package.
from tensorflow.lite.profiling.proto import model_runtime_info_pb2  # pylint: disable=unused-import
from tensorflow.lite.profiling.proto import profiling_info_pb2  # pylint: disable=unused-import
from tensorflow.lite.python import conversion_metadata_schema_py_generated as conversion_metadata_fb
from tensorflow.lite.python import lite_constants as constants
from tensorflow.lite.python.convert import build_conversion_flags as _build_conversion_flags
from tensorflow.lite.python.convert import convert_graphdef as _convert_graphdef
from tensorflow.lite.python.convert import convert_graphdef_with_arrays as _convert_graphdef_with_arrays
from tensorflow.lite.python.convert import convert_jax_hlo as _convert_jax_hlo
from tensorflow.lite.python.convert import convert_saved_model as _convert_saved_model
from tensorflow.lite.python.convert import ConverterError  # pylint: disable=unused-import
from tensorflow.lite.python.convert import deduplicate_readonly_buffers as _deduplicate_readonly_buffers
from tensorflow.lite.python.convert import mlir_quantize as _mlir_quantize
from tensorflow.lite.python.convert import mlir_sparsify as _mlir_sparsify
from tensorflow.lite.python.convert import OpsSet
from tensorflow.lite.python.convert import toco_convert  # pylint: disable=unused-import
from tensorflow.lite.python.convert_phase import Component
from tensorflow.lite.python.convert_phase import convert_phase
from tensorflow.lite.python.convert_phase import SubComponent
from tensorflow.lite.python.convert_saved_model import freeze_saved_model as _freeze_saved_model
from tensorflow.lite.python.interpreter import Interpreter  # pylint: disable=unused-import
from tensorflow.lite.python.interpreter import load_delegate  # pylint: disable=unused-import
from tensorflow.lite.python.interpreter import OpResolverType  # pylint: disable=unused-import
from tensorflow.lite.python.metrics import metrics
from tensorflow.lite.python.op_hint import convert_op_hints_to_stubs  # pylint: disable=unused-import
from tensorflow.lite.python.op_hint import is_ophint_converted as _is_ophint_converted
from tensorflow.lite.python.op_hint import OpHint  # pylint: disable=unused-import
from tensorflow.lite.python.optimize import calibrator as _calibrator
from tensorflow.lite.python.util import _jit
from tensorflow.lite.python.util import build_debug_info_func as _build_debug_info_func
from tensorflow.lite.python.util import convert_debug_info_func as _convert_debug_info_func
from tensorflow.lite.python.util import freeze_graph as _freeze_graph
from tensorflow.lite.python.util import get_debug_info as _get_debug_info
from tensorflow.lite.python.util import get_grappler_config as _get_grappler_config
from tensorflow.lite.python.util import get_model_hash as _get_model_hash
from tensorflow.lite.python.util import get_save_spec as _get_save_spec
from tensorflow.lite.python.util import get_sparsity_modes as _get_sparsity_modes
from tensorflow.lite.python.util import get_tensor_name as _get_tensor_name
from tensorflow.lite.python.util import get_tensors_from_tensor_names as _get_tensors_from_tensor_names
from tensorflow.lite.python.util import get_tf_type_name as _get_tf_type_name
from tensorflow.lite.python.util import is_frozen_graph as _is_frozen_graph
from tensorflow.lite.python.util import model_input_signature as _model_input_signature
from tensorflow.lite.python.util import modify_model_io_type as _modify_model_io_type
from tensorflow.lite.python.util import populate_conversion_metadata as _populate_conversion_metadata
from tensorflow.lite.python.util import run_graph_optimizations as _run_graph_optimizations
from tensorflow.lite.python.util import set_tensor_shapes as _set_tensor_shapes
from tensorflow.lite.python.util import trace_model_call as _trace_model_call
from tensorflow.lite.tools import flatbuffer_utils
from tensorflow.lite.tools.optimize.debugging.python.debugger import QuantizationDebugger  # pylint: disable=unused-import
from tensorflow.lite.tools.optimize.debugging.python.debugger import QuantizationDebugOptions  # pylint: disable=unused-import
from tensorflow.python.client import session as _session
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function as _def_function
from tensorflow.python.eager import function as _function
from tensorflow.python.framework import byte_swap_tensor as bst
from tensorflow.python.framework import convert_to_constants as _convert_to_constants
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import versions
from tensorflow.python.framework.errors_impl import NotFoundError as _NotFoundError
from tensorflow.python.framework.importer import import_graph_def as _import_graph_def
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import loader_impl as _loader_impl
from tensorflow.python.saved_model import save as _save
from tensorflow.python.saved_model import save_options as _save_options
from tensorflow.python.saved_model import signature_constants as _signature_constants
from tensorflow.python.saved_model import tag_constants as _tag_constants
from tensorflow.python.saved_model.load import load as _load
from tensorflow.python.saved_model.loader_impl import parse_saved_model_with_debug_info as _parse_saved_model_with_debug_info
from tensorflow.python.util import deprecation as _deprecation
from tensorflow.python.util import keras_deps
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export as _tf_export


@_tf_export("lite.Optimize")
class Optimize(enum.Enum):
  # [CLASS DOC]
  # Enumeration of optimization strategies the converter can apply. Members are DEFAULT,
  # OPTIMIZE_FOR_SIZE and OPTIMIZE_FOR_LATENCY (the latter two deprecated aliases of default
  # behavior) and EXPERIMENTAL_SPARSITY. Each member's value is its own name as a string.
  # Exported publicly as the lite optimize enum.
  """Enum defining the optimizations to apply when generating a tflite model.

  DEFAULT
      The default optimization strategy that enables post-training quantization.
      The type of post-training quantization that will be used is dependent on
      the other converter options supplied. Refer to the
      [documentation](https://ai.google.dev/edge/litert/models/post_training_quantization)
      for further information on the types available and how to use them.

  OPTIMIZE_FOR_SIZE
      Deprecated. Does the same as DEFAULT.

  OPTIMIZE_FOR_LATENCY
      Deprecated. Does the same as DEFAULT.

  EXPERIMENTAL_SPARSITY
      Experimental flag, subject to change.

      Enable optimization by taking advantage of the sparse model weights
      trained with pruning.

      The converter will inspect the sparsity pattern of the model weights and
      do its best to improve size and latency.
      The flag can be used alone to optimize float32 models with sparse weights.
      It can also be used together with the DEFAULT optimization mode to
      optimize quantized models with sparse weights.
  """

  # Default optimization strategy that quantizes model weights. Enhanced
  # optimizations are gained by providing a representative dataset that
  # quantizes biases and activations as well.
  # Converter will do its best to reduce size and latency, while minimizing
  # the loss in accuracy.
  DEFAULT = "DEFAULT"

  # Deprecated. Does the same as DEFAULT.
  OPTIMIZE_FOR_SIZE = "OPTIMIZE_FOR_SIZE"

  # Deprecated. Does the same as DEFAULT.
  OPTIMIZE_FOR_LATENCY = "OPTIMIZE_FOR_LATENCY"

  # Experimental flag, subject to change.
  # Enable optimization by taking advantage of the sparse model weights trained
  # with pruning.
  #
  # The converter will inspect the sparsity pattern of the model weights and do
  # its best to improve size and latency.
  # The flag can be used alone to optimize float32 models with sparse weights.
  # It can also be used together with the DEFAULT optimization mode to optimize
  # quantized models with sparse weights.
  EXPERIMENTAL_SPARSITY = "EXPERIMENTAL_SPARSITY"

  def __str__(self):
    # [FUNCTION DOC]
    # Returns the string form of the member's value (its name text) rather than the default
    # enum representation. Takes no arguments besides the member itself.
    return str(self.value)


# TODO(b/198099651): move converter implementation out of lite.py
@_tf_export("lite.RepresentativeDataset")
class RepresentativeDataset:
  # [CLASS DOC]
  # Thin wrapper holding a user supplied generator function that yields calibration samples
  # used by post-training quantization. Attribute: input_gen, the generator function.
  # Instances are created by users or by QuantizationMode when a bare generator is supplied.
  """Representative dataset used to optimize the model.

  This is a generator function that provides a small dataset to calibrate or
  estimate the range, i.e, (min, max) of all floating-point arrays in the model
  (such as model input, activation outputs of intermediate layers, and model
  output) for quantization. Usually, this is a small subset of a few hundred
  samples randomly chosen, in no particular order, from the training or
  evaluation dataset.
  """

  def __init__(self, input_gen):
    # [FUNCTION DOC]
    # Constructor. Stores the given generator function (input_gen) unchanged as an attribute.
    # No validation is done. Returns nothing.
    """Creates a representative dataset.

    Args:
      input_gen: A generator function that generates input samples for the model
        and has the same order, type and shape as the inputs to the model.
        Usually, this is a small subset of a few hundred samples randomly
        chosen, in no particular order, from the training or evaluation dataset.
    """
    self.input_gen = input_gen


@_tf_export("lite.TargetSpec")
class TargetSpec:
  # [CLASS DOC]
  # Plain configuration holder describing the target device: which operator sets are allowed,
  # which data types are supported, user select-TF-op names, and supported backends. Also holds
  # private experimental fields (custom op registerer list and an accumulation type hint).
  # Instances are created by users and stored on converters as the target specification.
  """Specification of target device used to optimize the model.

  Attributes:
    supported_ops: Experimental flag, subject to change. Set of `tf.lite.OpsSet`
      options, where each option represents a set of operators supported by the
      target device. (default {tf.lite.OpsSet.TFLITE_BUILTINS}))
    supported_types: Set of `tf.dtypes.DType` data types supported on the target
      device. If initialized, optimization might be driven by the smallest type
      in this set. (default set())
    experimental_select_user_tf_ops: Experimental flag, subject to change. Set
      of user's TensorFlow operators' names that are required in the TensorFlow
      Lite runtime. These ops will be exported as select TensorFlow ops in the
      model (in conjunction with the tf.lite.OpsSet.SELECT_TF_OPS flag). This is
      an advanced feature that should only be used if the client is using TF ops
      that may not be linked in by default with the TF ops that are provided
      when using the SELECT_TF_OPS path. The client is responsible for linking
      these ops into the target runtime.
    experimental_supported_backends: Experimental flag, subject to change. Set
      containing names of supported backends. Currently only "GPU" is supported,
      more options will be available later.
  """

  def __init__(
      self,
      supported_ops=None,
      supported_types=None,
      experimental_select_user_tf_ops=None,
      experimental_supported_backends=None,
  ):
    # [FUNCTION DOC]
    # Constructor. Parameters: supported_ops (set of op-set enum members, default is a set with
    # only the builtin ops), supported_types (set of data types, default empty),
    # experimental_select_user_tf_ops (set of op name strings, default empty), and
    # experimental_supported_backends (set of backend names, default nothing). Missing values
    # are replaced by the defaults above. Also initializes an empty list of custom op
    # registerers and a nothing-valued accumulation type hint. Returns nothing.
    if supported_ops is None:
      supported_ops = {OpsSet.TFLITE_BUILTINS}
    self.supported_ops = supported_ops
    if supported_types is None:
      supported_types = set()
    self.supported_types = supported_types
    if experimental_select_user_tf_ops is None:
      experimental_select_user_tf_ops = set()
    self.experimental_select_user_tf_ops = experimental_select_user_tf_ops
    self.experimental_supported_backends = experimental_supported_backends
    self._experimental_custom_op_registerers = []
    # Hint for the supported accumulation type used for inference. Typically
    # used for fp16 post-training quantization, where some models can use fp16
    # accumulators instead of the typical fp32 type.
    self._experimental_supported_accumulation_type = None


class QuantizationMode:
  # [CLASS DOC]
  # HELPER: internal helper for the converters that classifies the requested quantization.
  # Derives, from the user's optimizations, target spec, representative dataset and graph
  # definition, which quantization flavor applies (post-training int8 or int16x8 with or without
  # float fallback, dynamic range, float16/bfloat16, quantization-aware training, low-bit QAT)
  # and produces the matching converter flags and activation/bias types. The constructor
  # validates option combinations and may mutate the target spec and wrap the dataset.
  """QuantizationMode determines the quantization type from user options."""

  def __init__(
      self,
      optimizations,
      target_spec,
      representative_dataset,
      graph_def,
      disable_per_channel=False,
      experimental_new_dynamic_range_quantizer=False,
      experimental_low_bit_qat=False,
      full_integer_quantization_bias_type=None,
      experimental_mlir_variable_quantization=False,
      experimental_qdq_annotation=False,
  ):
    # [FUNCTION DOC]
    # Constructor. Parameters: optimizations (collection of Optimize members), target_spec
    # (TargetSpec), representative_dataset (generator function, dataset wrapper, or nothing),
    # graph_def (graph definition used to detect training-time quantization ops, may be
    # nothing), disable_per_channel (bool), experimental_new_dynamic_range_quantizer (bool),
    # experimental_low_bit_qat (bool), full_integer_quantization_bias_type (optional dtype),
    # experimental_mlir_variable_quantization (bool), experimental_qdq_annotation (bool).
    # Logs a warning for each deprecated optimization present. If int8 is required it validates
    # the int8 setup (which may change supported ops and wrap the dataset); if float16 is
    # required it validates that variable quantization is off; finally validates the bias type.
    # May raise a value error from those validations. Returns nothing.
    self._optimizations = optimizations
    for deprecated_optimization in [
        Optimize.OPTIMIZE_FOR_SIZE,
        Optimize.OPTIMIZE_FOR_LATENCY,
    ]:
      if deprecated_optimization in self._optimizations:
        logging.warning(
            (
                "Optimization option %s is deprecated, please use"
                " optimizations=[Optimize.DEFAULT] instead."
            ),
            deprecated_optimization,
        )

    self._experimental_qdq_annotation = experimental_qdq_annotation
    self._target_spec = target_spec
    self._representative_dataset = representative_dataset
    self._graph_def = graph_def
    if self._is_int8_target_required():
      self._validate_int8_required()

    self.enable_mlir_variable_quantization = (
        experimental_mlir_variable_quantization
    )
    if self._is_float16_target_required():
      self._validate_float16_required()
    self._disable_per_channel = disable_per_channel

    self._enable_new_dynamic_range_quantizer = (
        experimental_new_dynamic_range_quantizer
    )
    # Allow training with lower than 8 bit weights to be converted
    # to constants with trained scale.
    self._experimental_low_bit_qat = experimental_low_bit_qat

    self._full_integer_quantization_bias_type = (
        full_integer_quantization_bias_type
    )
    self._validate_full_integer_quantization_bias_type()

  def is_post_training_int8_only_quantization(self):
    # [FUNCTION DOC]
    # Returns true when an optimization is enabled, a representative dataset exists, int16x8 is
    # not requested, float ops are not allowed, and int8 is required (integer-only int8 PTQ).
    return (
        self.is_any_optimization_enabled()
        and self._representative_dataset is not None
        and not self._is_int16x8_target_required()
        and not self.is_allow_float()
        and self._is_int8_target_required()
    )

  def is_post_training_int8_quantization_with_float_fallback(self):
    # [FUNCTION DOC]
    # Returns true when an optimization is enabled, a representative dataset exists, int16x8 is
    # not requested, float ops are allowed, and the smallest supported type is int8.
    return (
        self.is_any_optimization_enabled()
        and self._representative_dataset is not None
        and not self._is_int16x8_target_required()
        and self.is_allow_float()
        and self._smallest_supported_type() == _dtypes.int8
    )

  def is_post_training_int8_quantization(self):
    # [FUNCTION DOC]
    # Returns true if either int8-only or int8-with-float-fallback post-training quantization
    # applies.
    return (
        self.is_post_training_int8_only_quantization()
        or self.is_post_training_int8_quantization_with_float_fallback()
    )

  def is_post_training_int16x8_only_quantization(self):
    # [FUNCTION DOC]
    # Returns true when an optimization is enabled, a representative dataset exists, the
    # int16-activation/int8-weight op set is requested, and float ops are not allowed.
    return (
        self.is_any_optimization_enabled()
        and self._representative_dataset is not None
        and self._is_int16x8_target_required()
        and not self.is_allow_float()
    )

  def is_post_training_int16x8_quantization_with_float_fallback(self):
    # [FUNCTION DOC]
    # Returns true when an optimization is enabled, a representative dataset exists, the
    # int16x8 op set is requested, and float ops are allowed.
    return (
        self.is_any_optimization_enabled()
        and self._representative_dataset is not None
        and self._is_int16x8_target_required()
        and self.is_allow_float()
    )

  def is_post_training_int16x8_quantization(self):
    # [FUNCTION DOC]
    # Returns true if either int16x8-only or int16x8-with-float-fallback quantization applies.
    return (
        self.is_post_training_int16x8_only_quantization()
        or self.is_post_training_int16x8_quantization_with_float_fallback()
    )

  def is_post_training_integer_quantization(self):
    # [FUNCTION DOC]
    # Returns true if any post-training integer quantization (int8 or int16x8 variants) applies.
    return (
        self.is_post_training_int8_quantization()
        or self.is_post_training_int16x8_quantization()
    )

  def is_low_bit_quantize_aware_training(self):
    # [FUNCTION DOC]
    # Returns true when an optimization is enabled, the graph contains training-time
    # quantization ops, and the experimental low-bit QAT option was set.
    return (
        self.is_any_optimization_enabled()
        and self.is_quantization_aware_trained_model()
        and self._experimental_low_bit_qat
    )

  def is_quantization_aware_training(self):
    # [FUNCTION DOC]
    # Returns true unconditionally if the QDQ annotation option is set; otherwise true when an
    # optimization is enabled, the graph has training-time quantization ops, and low-bit QAT
    # is not in effect.
    if self._experimental_qdq_annotation:
      return True
    return (
        self.is_any_optimization_enabled()
        and self.is_quantization_aware_trained_model()
        and not self.is_low_bit_quantize_aware_training()
    )

  def is_integer_quantization(self):
    # [FUNCTION DOC]
    # Returns true if post-training integer quantization, quantization-aware training, or
    # low-bit quantization-aware training applies.
    return (
        self.is_post_training_integer_quantization()
        or self.is_quantization_aware_training()
        or self.is_low_bit_quantize_aware_training()
    )

  def is_post_training_dynamic_range_quantization(self):
    # [FUNCTION DOC]
    # Returns true when an optimization is enabled, there is no representative dataset, the
    # graph is not quantization-aware trained, and the smallest supported type is int8.
    # Post-training dynamic range quantization is only enabled if post-training
    # int8 quantization and training time quantization was not done.
    return (
        self.is_any_optimization_enabled()
        and self._representative_dataset is None
        and not self.is_quantization_aware_trained_model()
        and self._smallest_supported_type() == _dtypes.int8
    )

  def is_post_training_float16_quantization(self):
    # [FUNCTION DOC]
    # Returns true when an optimization is enabled, the smallest supported type is two bytes
    # wide, and float16 is among the supported types.
    return (
        self.is_any_optimization_enabled()
        and self._smallest_supported_type().size == 2
        and _dtypes.float16 in self._target_spec.supported_types
    )

  def is_bfloat16_quantization(self):
    # [FUNCTION DOC]
    # Returns true when an optimization is enabled, the smallest supported type is two bytes
    # wide, and bfloat16 is among the supported types.
    return (
        self.is_any_optimization_enabled()
        and self._smallest_supported_type().size == 2
        and _dtypes.bfloat16 in self._target_spec.supported_types
    )

  def activations_type(self):
    # [FUNCTION DOC]
    # Returns the activation data type: int16 if integer quantization with the int16x8 op set,
    # int8 for other integer quantization, otherwise float32.
    if self.is_integer_quantization():
      if self._is_int16x8_target_required():
        return _dtypes.int16
      else:
        return _dtypes.int8
    else:
      return _dtypes.float32

  def bias_type(self):
    # [FUNCTION DOC]
    # Returns the bias data type: the user-specified full integer quantization bias type if
    # set; otherwise int64 for int16 activations, int32 for int8 activations, else float32.
    if self._full_integer_quantization_bias_type:
      return self._full_integer_quantization_bias_type

    if self.activations_type() == _dtypes.int16:
      return _dtypes.int64
    elif self.activations_type() == _dtypes.int8:
      return _dtypes.int32
    else:
      return _dtypes.float32

  def converter_flags(self, inference_ty=None, inference_input_ty=None):
    # [FUNCTION DOC]
    # Builds the mapping of converter flag names to values for the detected quantization mode.
    # Parameters: inference_ty (optional dtype overriding the inference type in the integer
    # case; otherwise the activation type is used, and float32 in the fallback case) and
    # inference_input_ty (optional dtype, used only in the fallback case where it is passed
    # through unchanged). Cases in order: integer quantization (float input, no dynamic range
    # or float16 flags, low-bit QAT toggles tensor range inference and fake-quant bit widths,
    # variable quantization flag); dynamic range (float inference, post-training quantize on,
    # per-channel disable flag, MLIR dynamic range quantizer flag, variable quantization flag);
    # float16 (float16 quantization on, accumulation type hint from the target spec, bfloat16
    # allowance, dynamic range quantizer flag, variable quantization flag); otherwise a plain
    # float configuration with the bfloat16 allowance. Returns the mapping; no side effects.
    """Flags to the converter."""

    if self.is_integer_quantization():
      is_low_bit_qat = self.is_low_bit_quantize_aware_training()
      return {
          "inference_type": (
              inference_ty
              if inference_ty is not None
              else self.activations_type()
          ),
          "inference_input_type": _dtypes.float32,
          "post_training_quantize": False,  # disable dynamic range quantization
          "quantize_to_float16": False,  # disable float16 quantization
          "disable_infer_tensor_range": is_low_bit_qat,
          "use_fake_quant_num_bits": is_low_bit_qat,
          "enable_mlir_variable_quantization": (
              self.enable_mlir_variable_quantization
          ),
      }
    elif self.is_post_training_dynamic_range_quantization():
      return {
          "inference_type": _dtypes.float32,
          "inference_input_type": _dtypes.float32,
          "post_training_quantize": True,  # enable dynamic range quantization
          "quantize_to_float16": False,  # disable float16 quantization
          # experimental: disable per-channel (per-axis) quantization.
          "disable_per_channel_quantization": self._disable_per_channel,
          "enable_mlir_dynamic_range_quantizer": (
              self._enable_new_dynamic_range_quantizer
          ),
          "enable_mlir_variable_quantization": (
              self.enable_mlir_variable_quantization
          ),
      }
    elif self.is_post_training_float16_quantization():
      return {
          "inference_type": _dtypes.float32,
          "inference_input_type": _dtypes.float32,
          "post_training_quantize": True,
          "quantize_to_float16": True,  # enable float16 quantization
          # pylint: disable=protected-access
          "accumulation_type": (
              self._target_spec._experimental_supported_accumulation_type
          ),
          # pylint: enable=protected-access
          "allow_bfloat16": self.is_bfloat16_quantization(),
          "enable_mlir_dynamic_range_quantizer": (
              self._enable_new_dynamic_range_quantizer
          ),
          "enable_mlir_variable_quantization": (
              self.enable_mlir_variable_quantization
          ),
      }
    else:
      # Note this might still trigger (uint8) quantization to be compatible with
      # the old converter.
      return {
          "inference_type": (
              inference_ty if inference_ty is not None else _dtypes.float32
          ),
          "inference_input_type": inference_input_ty,
          "post_training_quantize": False,  # enable dynamic range quantization
          "quantize_to_float16": False,  # disable float16 quantization
          "allow_bfloat16": self.is_bfloat16_quantization(),
      }

  # Below are helpers for the above functions.

  def _validate_int8_required(self):
    # [FUNCTION DOC]
    # HELPER: validation step used by the QuantizationMode constructor.
    # Checks int8 options. Raises a value error if the op set is exactly the int8 builtin set
    # but supported types are neither empty nor exactly int8, and if no representative dataset
    # is given while not quantization-aware training. Side effects: when supported types is
    # exactly int8, replaces the target spec's supported ops with only the int8 builtin set;
    # wraps a bare dataset generator in the dataset wrapper class. Returns nothing.
    """Int8 mode requires certain parameters to exist and be compatible."""
    # Validate target_spec attibute.
    if set(self._target_spec.supported_ops) == {
        OpsSet.TFLITE_BUILTINS_INT8
    } and not (
        set(self._target_spec.supported_types) == set()
        or set(self._target_spec.supported_types) == {_dtypes.int8}
    ):
      raise ValueError(
          "As full integer quantization has been enabled by setting "
          "`target_spec.supported_ops`={tf.lite.OpsSet.TFLITE_BUILTINS_INT8}, "
          "thus `target_spec.supported_types` should be left uninitizalized "
          "or set to {tf.int8}."
      )
    if set(self._target_spec.supported_types) == {_dtypes.int8}:
      self._target_spec.supported_ops = {OpsSet.TFLITE_BUILTINS_INT8}

    # Check if representative_dataset is specified.
    if (
        not self._representative_dataset
        and not self.is_quantization_aware_training()
    ):
      raise ValueError(
          "For full integer quantization, a "
          "`representative_dataset` must be specified."
      )

    # Update represenative dataset to the expected format.
    if self._representative_dataset:
      if not isinstance(self._representative_dataset, RepresentativeDataset):
        self._representative_dataset = RepresentativeDataset(
            self._representative_dataset
        )

  def _validate_float16_required(self):
    # [FUNCTION DOC]
    # HELPER: validation step used by the QuantizationMode constructor.
    # Raises a value error if MLIR variable quantization is enabled, since it is supported only
    # with full integer quantization. Returns nothing.
    """Float16 mode requires certain parameters to exist and be compatible."""
    if self.enable_mlir_variable_quantization:
      raise ValueError(
          "`_experimental_variable_quantization` is only supported for full"
          " integer quantization."
      )

  def _validate_full_integer_quantization_bias_type(self):
    # [FUNCTION DOC]
    # HELPER: validation step used by the QuantizationMode constructor.
    # Does nothing if no bias type was requested. Otherwise raises a value error if activations
    # are float32 (not integer quantization), if int8 activations use a bias type other than
    # int32, or if int16 activations use a bias type other than int32 or int64. Returns nothing.
    """Validates bias type for full interger quantization."""
    bias_type = self._full_integer_quantization_bias_type
    if not bias_type:
      return

    if self.activations_type() == _dtypes.float32:
      raise ValueError(
          "`full_integer_quantization_bias_type` is only supported for full"
          " integer quantization."
      )

    if self.activations_type() == _dtypes.int8 and bias_type != _dtypes.int32:
      raise ValueError(
          "Expected bias type to be `dtypes.int32` for Int8Quant. "
          f"Current setting bias type: {bias_type}"
      )

    if (
        self.activations_type() == _dtypes.int16
        and bias_type != _dtypes.int32
        and bias_type != _dtypes.int64
    ):
      raise ValueError(
          "Expected bias type to be `dtypes.int32` or `dtypes.int64` for "
          f"Int16Quant. Current setting bias type: {bias_type}"
      )

  def _is_int8_target_required(self):
    # [FUNCTION DOC]
    # HELPER: predicate for QuantizationMode.
    # Returns true if the int8 builtin op set is among the supported ops or the supported types
    # are exactly int8.
    return (
        OpsSet.TFLITE_BUILTINS_INT8 in set(self._target_spec.supported_ops)
    ) or (set(self._target_spec.supported_types) == set([_dtypes.int8]))

  def _is_int16x8_target_required(self):
    # [FUNCTION DOC]
    # HELPER: predicate for QuantizationMode.
    # Returns true if the experimental int16-activation/int8-weight op set is among the
    # supported ops.
    return (
        OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8
        in set(self._target_spec.supported_ops)
    )

  def is_allow_float(self):
    # [FUNCTION DOC]
    # Returns true if the supported ops include the float builtin set or the select-TF-ops set,
    # meaning float operators may remain in the model.
    return (OpsSet.TFLITE_BUILTINS in set(self._target_spec.supported_ops)) or (
        OpsSet.SELECT_TF_OPS in set(self._target_spec.supported_ops)
    )

  def _is_float16_target_required(self):
    # [FUNCTION DOC]
    # HELPER: predicate for QuantizationMode.
    # Returns true if float16 is among the target spec's supported types.
    return _dtypes.float16 in self._target_spec.supported_types

  def is_any_optimization_enabled(self):
    # [FUNCTION DOC]
    # Returns true if the optimizations collection contains the default, size or latency
    # optimization (the sparsity option alone does not count).
    return bool(
        set(self._optimizations).intersection([
            Optimize.OPTIMIZE_FOR_LATENCY,
            Optimize.OPTIMIZE_FOR_SIZE,
            Optimize.DEFAULT,
        ])
    )

  def _smallest_supported_type(self):
    # [FUNCTION DOC]
    # HELPER: used by the quantization mode predicates.
    # Returns the supported data type with the smallest byte size from the target spec, or
    # int8 when no supported types were given.
    if self._target_spec.supported_types:
      return min(self._target_spec.supported_types, key=lambda x: x.size)
    else:
      # The default smallest supported type is INT8.
      return _dtypes.int8

  def is_quantization_aware_trained_model(self):
    # [FUNCTION DOC]
    # Scans the graph definition (top-level nodes and every library function's nodes) for
    # training-time quantization ops such as fake-quant and quantize-and-dequantize variants.
    # Returns true if any is found; false if none or if no graph definition was provided.
    """Checks if the graph contains any training-time quantization ops."""
    training_quant_ops = frozenset({
        "FakeQuantWithMinMaxVars",
        "FakeQuantWithMinMaxVarsPerChannel",
        "FakeQuantWithMinMaxArgs",
        "QuantizeAndDequantizeV2",
        "QuantizeAndDequantizeV3",
    })

    if self._graph_def:
      for node_def in self._graph_def.node:
        if node_def.op in training_quant_ops:
          return True
      for function in self._graph_def.library.function:
        for node_def in function.node_def:
          if node_def.op in training_quant_ops:
            return True
    return False


class TFLiteConverterBase:
  # [CLASS DOC]
  # Shared superclass of the V1 and V2 converters. Holds the common user-facing options
  # (optimizations, representative dataset, target spec, custom ops, new converter/quantizer
  # switches, StableHLO quantizer settings), many private experimental flags, debug/IR dump
  # parameters, metrics collection and the conversion metadata record. Provides the shared
  # quantization, sparsification, calibration, metadata and validation helpers used by
  # subclasses. The class-level original model type is set temporarily by factory methods so
  # that the constructor can record where the model came from.
  """Converter superclass to share functionality between V1 and V2 converters."""

  # Stores the original model type temporarily to transmit the information
  # from the factory class methods to TFLiteConverterBase init function.
  _original_model_type = conversion_metadata_fb.ModelType.NONE

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor with no arguments. Initializes all converter options to defaults (empty
    # optimizations, builtin target spec, new converter and quantizer on, resource variables
    # on, dynamic range quantizer on, etc.), creates a metrics recorder, and builds a fresh
    # conversion metadata record filled with the TensorFlow version and the original model type
    # taken from the class level marker. Also sets the debug/IR dump options to nothing.
    # Returns nothing.
    self.optimizations = set()
    self.representative_dataset = None
    self.target_spec = TargetSpec()
    self.allow_custom_ops = False
    self.experimental_new_converter = True
    self.experimental_new_quantizer = True
    self.experimental_enable_resource_variables = True
    self._experimental_calibrate_only = False
    self._experimental_sparsify_model = False
    self._experimental_disable_per_channel = False
    self._debug_info = None  # contains the stack traces of all the original
    # nodes in the `GraphDef` to the converter.
    self.saved_model_dir = None
    self._saved_model_tags = None
    self._saved_model_version = 0
    self._saved_model_exported_names = []
    self._tflite_metrics = metrics.TFLiteConverterMetrics()
    self._collected_converter_params = {}
    self.unfold_batchmatmul = False
    self.legalize_custom_tensor_list_ops = False
    self._experimental_lower_tensor_list_ops = True
    self._experimental_default_to_single_batch_in_tensor_list_ops = False
    self._experimental_unfold_large_splat_constant = False
    self._experimental_tf_quantization_mode = None
    # If unset, bias:int32 is by default except 16x8 quant.
    # For 16x8 quant, bias:int64 is used to prevent any overflow by default.
    # The accumulator type will be the same as bias type set by
    # full_integer_quantization_bias_type.
    self._experimental_full_integer_quantization_bias_type = None
    # Provides specs for quantization, whether preset or custom.
    self._experimental_quantization_options = None  # Deprecated
    # Whether to use StableHLO Quantizer instead of TFLite Quantizer.
    self.experimental_use_stablehlo_quantizer = False
    # Quantization configuration to pass to StableHLO Quantizer.
    self.experimental_stablehlo_quantizer_config = None
    # Initializes conversion metadata.
    self.exclude_conversion_metadata = False
    self._metadata = conversion_metadata_fb.ConversionMetadataT()
    self._metadata.environment = conversion_metadata_fb.EnvironmentT()
    self._metadata.options = conversion_metadata_fb.ConversionOptionsT()
    self._metadata.environment.tensorflowVersion = versions.__version__
    self._metadata.environment.modelType = self._get_original_model_type()
    self._experimental_enable_dynamic_update_slice = False
    self._experimental_preserve_assert_op = False
    self._experimental_guarantee_all_funcs_one_use = False

    # When the value is true, the MLIR quantantizer triggers dynamic range
    # quantization in MLIR instead of the old quantizer. Used only if
    # experimental_new_quantizer is on.
    self.experimental_new_dynamic_range_quantizer = True
    # Experimental flag to enable low-bit QAT in 8 bit.
    self._experimental_low_bit_qat = False
    # Experimental flag to add all TF ops (including custom TF ops) to the
    # converted model as flex ops.
    self._experimental_allow_all_select_tf_ops = False

    self._experimental_variable_quantization = False
    self._experimental_disable_fuse_mul_and_fc = False
    self._experimental_use_buffer_offset = False
    self._experimental_reduce_type_precision = False
    self._experimental_qdq_conversion_mode = None
    self._experimental_disable_per_channel_quantization_for_dense_layers = False
    self._experimental_enable_composite_direct_lowering = True
    self.model_origin_framework = constants.UNSET
    self.canonicalizing_inf_as_min_max_float = True
    self._experimental_strict_qdq = False
    self._experimental_unsafe_fuse_dynamic_shaped_broadcast = False

    # Debug parameters
    self.ir_dump_dir = None
    self.ir_dump_pass_regex = None
    self.ir_dump_func_regex = None
    self.enable_timing = None
    self.print_ir_before = None
    self.print_ir_after = None
    self.print_ir_module_scope = None
    self.elide_elementsattrs_if_larger = None
    self.serialize_debug_metadata = False

  def _grappler_config(self, optimizers=None):
    # [FUNCTION DOC]
    # HELPER: builds the Grappler configuration for the converters.
    # Parameter optimizers: optional list of Grappler optimizer names; it is mutated in place
    # when given (names are appended). Adds the constant folding pass when the new MLIR
    # converter is disabled, and the layout pass when the only supported op set is select TF
    # ops. Returns the Grappler config object built by the shared utility.
    """Creates a tf.compat.v1.ConfigProto for configuring Grappler.

    Args:
      optimizers: List of strings that represents the list of optimizers.

    Returns:
      tf.ConfigProto.
    """
    if not optimizers:
      optimizers = []
    # MLIR converter will take care of constant folding instead of grappler.
    if not self.experimental_new_converter:
      optimizers.append("constfold")

    is_only_flex_enabled = set([OpsSet.SELECT_TF_OPS]) == set(
        self.target_spec.supported_ops
    )
    if is_only_flex_enabled:
      # The layout optimizer turns NHCW to NCHW. This provides performance
      # optimizations when Flex mode is enabled. However, this is not compatible
      # with builtin ops.
      optimizers.append("layout")
    return _get_grappler_config(optimizers)

  def _quantize(
      self,
      result,
      input_type,
      output_type,
      activations_type,
      bias_type,
      allow_float,
      enable_variable_quantization,
      debug_options,
  ):
    # [FUNCTION DOC]
    # HELPER: runs calibration and quantization of a converted float model.
    # Parameters: result (serialized model bytes), input_type and output_type (dtypes for model
    # inputs and outputs), activations_type and bias_type (dtypes), allow_float (bool, whether
    # float ops may remain), enable_variable_quantization (bool), debug_options (a quantization
    # debug options message that is serialized for the MLIR quantizer).
    # Splits the target spec's custom op registerers into names and functions, wraps the
    # representative dataset into the wrapper class if needed (mutating the converter), adds
    # intermediate tensors to the model and creates a calibrator. Calibrates when calibrate-only
    # mode or the new quantizer is on. Returns the calibrated model if calibrate-only mode is
    # set; else the MLIR quantized model if the new quantizer is on and activations are not
    # int16; otherwise the result of the calibrator's calibrate-and-quantize path.
    """Quantize the model."""
    # pylint: disable=protected-access
    custom_op_registerers_by_name = [
        x
        for x in self.target_spec._experimental_custom_op_registerers
        if isinstance(x, str)
    ]
    custom_op_registerers_by_func = [
        x
        for x in self.target_spec._experimental_custom_op_registerers
        if not isinstance(x, str)
    ]
    # pylint: enable=protected-access
    if not isinstance(self.representative_dataset, RepresentativeDataset):
      self.representative_dataset = RepresentativeDataset(
          self.representative_dataset
      )

    # Add intermediate tensors to the model if needed.
    result = _calibrator.add_intermediate_tensors(result)
    calibrate_quantize = _calibrator.Calibrator(
        result, custom_op_registerers_by_name, custom_op_registerers_by_func
    )
    if self._experimental_calibrate_only or self.experimental_new_quantizer:
      calibrated = calibrate_quantize.calibrate(
          self.representative_dataset.input_gen
      )

    if self._experimental_calibrate_only:
      return calibrated
    elif self.experimental_new_quantizer and (
        activations_type != _dtypes.int16
    ):
      return _mlir_quantize(
          calibrated,
          self._experimental_disable_per_channel,
          input_data_type=input_type,
          output_data_type=output_type,
          enable_variable_quantization=enable_variable_quantization,
          disable_per_channel_for_dense_layers=self._experimental_disable_per_channel_quantization_for_dense_layers,
          debug_options_str=debug_options.SerializeToString(),
      )
    else:
      return calibrate_quantize.calibrate_and_quantize(
          self.representative_dataset.input_gen,
          input_type,
          output_type,
          allow_float,
          activations_type,
          bias_type,
          disable_per_channel=self._experimental_disable_per_channel,
          disable_per_channel_quantization_for_dense_layers=self._experimental_disable_per_channel_quantization_for_dense_layers,
      )

  def _is_unknown_shapes_allowed(self):
    # [FUNCTION DOC]
    # HELPER: returns whether unknown tensor dimensions may be passed to the converter, which
    # is true only when the new converter is enabled.
    # Unknown dimensions are only allowed with the new converter.
    return self.experimental_new_converter

  def _get_base_converter_args(self):
    # [FUNCTION DOC]
    # HELPER: gathers the converter options shared by all conversion paths.
    # Returns a mapping from argument names to values built from the converter's attributes
    # (input format as TF graph def, custom ops, debug info, target ops, quantization, IR dump
    # and debug options, etc.). When a saved model directory is set, adds the saved model
    # directory, version, tags and exported names. Logs a warning if the deprecated
    # quantization options are set. If the StableHLO quantizer is enabled it adds a
    # quantization config to the mapping (via the assign-or-default helper); if instead a
    # StableHLO config was supplied without enabling the quantizer it raises a value error.
    """Returns the base converter args.

    Returns:
      {key str: val}
    """
    args = {
        "input_format": constants.TENSORFLOW_GRAPHDEF,
        "allow_custom_ops": self.allow_custom_ops,
        "debug_info": self._debug_info,
        "target_ops": self.target_spec.supported_ops,
        "select_user_tf_ops": self.target_spec.experimental_select_user_tf_ops,
        "supported_backends": self.target_spec.experimental_supported_backends,
        "unfold_batchmatmul": self.unfold_batchmatmul,
        "legalize_custom_tensor_list_ops": self.legalize_custom_tensor_list_ops,
        "lower_tensor_list_ops": self._experimental_lower_tensor_list_ops,
        "unfold_large_splat_constant": (
            self._experimental_unfold_large_splat_constant
        ),
        "default_to_single_batch_in_tensor_list_ops": (
            self._experimental_default_to_single_batch_in_tensor_list_ops
        ),
        "tf_quantization_mode": self._experimental_tf_quantization_mode,
        "experimental_enable_resource_variables": (
            self.experimental_enable_resource_variables
        ),
        "enable_dynamic_update_slice": (
            self._experimental_enable_dynamic_update_slice
        ),
        "preserve_assert_op": self._experimental_preserve_assert_op,
        "guarantee_all_funcs_one_use": (
            self._experimental_guarantee_all_funcs_one_use
        ),
        "allow_all_select_tf_ops": self._experimental_allow_all_select_tf_ops,
        "disable_fuse_mul_and_fc": self._experimental_disable_fuse_mul_and_fc,
        "quantization_options": self._experimental_quantization_options,
        "ir_dump_dir": self.ir_dump_dir,
        "ir_dump_pass_regex": self.ir_dump_pass_regex,
        "ir_dump_func_regex": self.ir_dump_func_regex,
        "enable_timing": self.enable_timing,
        "print_ir_before": self.print_ir_before,
        "print_ir_after": self.print_ir_after,
        "print_ir_module_scope": self.print_ir_module_scope,
        "elide_elementsattrs_if_larger": self.elide_elementsattrs_if_larger,
        "use_buffer_offset": self._experimental_use_buffer_offset,
        "reduce_type_precision": self._experimental_reduce_type_precision,
        "use_stablehlo_quantizer": self.experimental_use_stablehlo_quantizer,
        "stablehlo_quantizer_config": (
            self.experimental_stablehlo_quantizer_config
        ),
        "qdq_conversion_mode": self._experimental_qdq_conversion_mode,
        "strict_qdq_mode": self._experimental_strict_qdq,
        "disable_per_channel_quantization_for_dense_layers": (
            self._experimental_disable_per_channel_quantization_for_dense_layers
        ),
        "enable_composite_direct_lowering": (
            self._experimental_enable_composite_direct_lowering
        ),
        "model_origin_framework": self.model_origin_framework,
        "canonicalizing_inf_as_min_max_float": (
            self.canonicalizing_inf_as_min_max_float
        ),
        "serialize_debug_metadata": self.serialize_debug_metadata,
        "unsafe_fuse_dynamic_shaped_broadcast": (
            self._experimental_unsafe_fuse_dynamic_shaped_broadcast
        ),
    }

    if self.saved_model_dir:
      args.update({
          "saved_model_dir": self.saved_model_dir,
          "saved_model_version": self._saved_model_version,
          "saved_model_tags": self._saved_model_tags,
          "saved_model_exported_names": self._saved_model_exported_names,
      })

    if self._experimental_quantization_options:
      logging.warning(
          "Configs from custom methods in experimental_quantization_options"
          " may not produce a valid tflite model. Note that currently this"
          " option only supports StableHLO path. Setting this option in TFLite"
          " path will be a no-op."
      )

    if self.experimental_use_stablehlo_quantizer:
      self._assign_stablehlo_quantization_config_or_populate_default(args)
    elif self.experimental_stablehlo_quantizer_config is not None:
      raise ValueError(
          "QuantizationConfig should be provided only when"
          " experimental_use_stablehlo_quantizer is set to true."
      )

    return args

  def _assign_stablehlo_quantization_config_or_populate_default(self, args):
    # [FUNCTION DOC]
    # HELPER: fills the StableHLO quantizer configuration into the converter argument mapping.
    # Parameter args: mapping of argument names to values; mutated by setting the quantization
    # config entry. If a user config exists and the default optimization is not requested, it
    # is used as is. If the default optimization and a representative dataset are present, the
    # model must come from a saved model with exactly one exported signature key (else a value
    # error is raised); the dataset is written to a temporary record file (side effect: creates
    # a temp file that is not removed here) and a static-range post-training quantization preset
    # with per-channel weights and full integer quantization is built, keeping uniform quantized
    # types unpacked off. Otherwise raises a value error saying only static range and
    # weight-only post-training quantization is supported. Returns nothing.
    """Assigns `QuantizationConfig` to `args` or populate default.

    Args:
      args: Dictionary of argument names and associated values.
    """
    if (
        self.experimental_stablehlo_quantizer_config is not None
        and Optimize.DEFAULT not in self.optimizations
    ):
      args["quantization_config"] = self.experimental_stablehlo_quantizer_config
    elif Optimize.DEFAULT in self.optimizations and self.representative_dataset:
      if len(self._saved_model_exported_names) != 1:
        raise ValueError(
            "StableHLO quantizer is only supported when converting from a"
            " SavedModel with one signature key."
        )

      signature_key = self._saved_model_exported_names[0]

      # Convert a programmatically provided representative dataset to a
      # temporary TFRecord file to be used by the StableHLO quantizer.
      tfrecord_file_path = tempfile.mkstemp(
          suffix=".tfrecord", prefix=signature_key
      )[1]
      rd.TfRecordRepresentativeDatasetSaver(
          {signature_key: tfrecord_file_path}
      ).save({signature_key: self.representative_dataset()})

      quantization_config = qc.QuantizationConfig(
          static_range_ptq_preset=qc.StaticRangePtqPreset(
              representative_datasets=[
                  qc.RepresentativeDatasetConfig(
                      tf_record=qc.TfRecordFile(path=tfrecord_file_path)
                  )
              ],
              enable_per_channel_quantized_weight=True,
              enable_full_int_quantization=True,
          ),
          # For ODML use cases, uniform quantized types should be left intact.
          pipeline_config=qc.PipelineConfig(
              unpack_quantized_types=False,
          ),
      )

      args["quantization_config"] = quantization_config
      # TODO: b/307626463 - Enable StableHLO quantizer DRQ when Optimize.DEFAULT
      # is set without representative dataset.
    else:
      raise ValueError(
          "StableHLO quantizer only supports static-range and weight-only PTQ."
      )

  def _contains_function_with_implements_attr(self, saved_model_proto):
    # [FUNCTION DOC]
    # HELPER: for saved model import detection. Looks at the first meta graph of the given saved
    # model message and returns true if any library function carries an implements attribute
    # (either the underscore or the api-prefixed name), false otherwise.
    meta_graph = saved_model_proto.meta_graphs[0]
    for function in meta_graph.graph_def.library.function:
      if function.attr.get("_implements", None) or function.attr.get(
          "api_implements", None
      ):
        return True
    return False

  def _parse_saved_model_args(self, always_enable_saved_model_import=False):
    # [FUNCTION DOC]
    # HELPER: decides whether the MLIR saved model import path is used.
    # Parameter always_enable_saved_model_import: bool, default false. Clears the saved model
    # directory (so the frozen graph path is used) when the new converter is off, when the saved
    # model cannot be read (OS error), when the import is not forced and no function has an
    # implements attribute, or when the schema version is zero (also logs a warning). Raises a
    # value error for schema versions other than 1 or 2. Otherwise records the schema version
    # and makes sure the exported names attribute is a list. Mutates converter state; returns
    # nothing.
    """Parses SavedModel arguments from the given Keras/RNN SavedModel.

    Args:
      always_enable_saved_model_import: Bool. When the value is true, it enables
        MLIR saved model import path regardless of checking the conditions.
    """
    if not self.experimental_new_converter:
      self.saved_model_dir = None
      return
    if self.saved_model_dir:
      try:
        saved_model_proto, _ = _parse_saved_model_with_debug_info(
            self.saved_model_dir
        )
      except OSError:
        # If it fails to read the given saved model, it will fall back to the
        # frozen graph def path.
        self.saved_model_dir = None
        return
      if (
          not always_enable_saved_model_import
          and not self._contains_function_with_implements_attr(
              saved_model_proto
          )
      ):
        self.saved_model_dir = None
        return

      if not self._saved_model_exported_names:
        self._saved_model_exported_names = []
      self._saved_model_version = saved_model_proto.saved_model_schema_version
      if self._saved_model_version == 0:
        self.saved_model_dir = None
        logging.warning("SavedModel schema version is zero.")
        return
      if self._saved_model_version not in [1, 2]:
        raise ValueError(
            "SavedModel file format({0}) is not supported".format(
                self._saved_model_version
            )
        )

  def _sparsify_model(self):
    # [FUNCTION DOC]
    # HELPER: returns true if the experimental sparsity optimization is among the optimizations.
    return Optimize.EXPERIMENTAL_SPARSITY in self.optimizations

  def _increase_conversion_attempt_metric(self):
    # [FUNCTION DOC]
    # HELPER: increments the converter-attempt counter in the metrics recorder. Returns nothing.
    self._tflite_metrics.increase_counter_converter_attempt()

  def _increase_conversion_success_metric(self):
    # [FUNCTION DOC]
    # HELPER: increments the converter-success counter in the metrics recorder. Returns nothing.
    self._tflite_metrics.increase_counter_converter_success()

  @classmethod
  def _set_original_model_type(cls, model_type):
    # [FUNCTION DOC]
    # HELPER: class method used by factory constructors to record where the model came from.
    # Parameter model_type: a model type enum from the conversion metadata schema. Raises a
    # value error if it is the "none" value. Otherwise stores it in the class level marker that
    # the next constructor call will consume. Returns nothing.
    """Stores the original model type."""
    if model_type == conversion_metadata_fb.ModelType.NONE:
      raise ValueError("The original model type should be specified.")
    cls._original_model_type = model_type

  def _get_original_model_type(self):
    # [FUNCTION DOC]
    # HELPER: one-time getter for the class level original model type. Returns the stored value
    # and resets the marker on the base class to "none" so that later converters do not inherit
    # it.
    """One-time getter to return original model type and set it to NONE."""
    model_type = TFLiteConverterBase._original_model_type
    TFLiteConverterBase._original_model_type = (
        conversion_metadata_fb.ModelType.NONE
    )
    return model_type

  def _save_conversion_params_metric(
      self, graph_def=None, inference_type=None, inference_input_type=None
  ):
    # [FUNCTION DOC]
    # HELPER: records conversion parameters in the metrics recorder and the metadata record.
    # Parameters: graph_def (optional graph definition used for quantization-aware-training
    # detection), inference_type and inference_input_type (optional dtypes forwarded to the
    # quantization mode flags). Merges the base converter arguments, version info, optimization
    # mode booleans, quantization flags and accumulation type into the collected parameter
    # mapping (mutating it), formats every value as text (enums by value, collections sorted and
    # comma joined, empty ones as the word None, others pretty printed), sets each as a converter
    # param metric and marks the export as required. Also fills the metadata options: custom op
    # allowance, select TF ops enabled/forced, and the list of model optimization modes
    # (float16, dynamic range, full integer, int16, quantization-aware training). May raise value
    # errors from the argument or quantization mode validation. Returns nothing.
    """Set conversion parameter metrics."""
    converter_kwargs = self._collected_converter_params
    converter_kwargs.update(self._get_base_converter_args())

    # Optimization parameters.
    quant_mode = QuantizationMode(
        self.optimizations,
        self.target_spec,
        self.representative_dataset,
        graph_def,
        self._experimental_disable_per_channel,
        self.experimental_new_dynamic_range_quantizer,
        self._experimental_low_bit_qat,
        self._experimental_full_integer_quantization_bias_type,
        self._experimental_variable_quantization,
        self._experimental_strict_qdq,
    )
    converter_kwargs.update({
        "tf_version": self._metadata.environment.tensorflowVersion,
        "api_version": self._metadata.environment.apiVersion,
        "original_model_format": self._metadata.environment.modelType,
        "optimization_default": quant_mode.is_any_optimization_enabled(),
        "optimization_post_training_dynamic_range": (
            quant_mode.is_post_training_dynamic_range_quantization()
        ),
        "optimization_post_training_float16": (
            quant_mode.is_post_training_float16_quantization()
        ),
        "optimization_post_training_integer_quantize": (
            quant_mode.is_post_training_integer_quantization()
        ),
        "optimization_qat": quant_mode.is_quantization_aware_training(),
        "optimization_low_bit_qat": (
            quant_mode.is_low_bit_quantize_aware_training()
        ),
        "optimization_sparsify": self._sparsify_model(),
        "activations_type": quant_mode.activations_type(),
    })
    converter_kwargs.update(
        quant_mode.converter_flags(inference_type, inference_input_type)
    )

    # pylint: disable=protected-access
    if self.target_spec._experimental_supported_accumulation_type:
      converter_kwargs.update({
          "accumulation_type": (
              self.target_spec._experimental_supported_accumulation_type
          )
      })
    # pylint: enable=protected-access

    def format_element(elem):
      # [FUNCTION DOC]
      # HELPER: formats one value for metrics. Enum members give their value as text; anything
      # else is pretty printed to text.
      if isinstance(elem, enum.Enum):
        return str(elem.value)
      return pprint.pformat(elem)

    def format_param(param):
      # [FUNCTION DOC]
      # HELPER: formats a metric value. Lists, tuples and sets become a comma joined string of
      # their sorted formatted elements (the word None when empty); other values are formatted
      # as a single element.
      if isinstance(param, (list, tuple, set)):
        if not param:
          return "None"  # Return None if empty.
        string_list = [format_element(x) for x in param]
        return ",".join(sorted(string_list))
      return format_element(param)

    for key, value in converter_kwargs.items():
      self._tflite_metrics.set_converter_param(key, format_param(value))
    self._tflite_metrics.set_export_required()

    # Set conversion option metadata.
    self._metadata.options.allowCustomOps = self.allow_custom_ops
    self._metadata.options.enableSelectTfOps = (
        OpsSet.SELECT_TF_OPS in self.target_spec.supported_ops
    )
    self._metadata.options.forceSelectTfOps = set(
        [OpsSet.SELECT_TF_OPS]
    ) == set(self.target_spec.supported_ops)
    self._metadata.options.modelOptimizationModes = []

    if quant_mode.is_post_training_float16_quantization():
      self._metadata.options.modelOptimizationModes.append(
          conversion_metadata_fb.ModelOptimizationMode.PTQ_FLOAT16
      )

    if quant_mode.is_post_training_dynamic_range_quantization():
      self._metadata.options.modelOptimizationModes.append(
          conversion_metadata_fb.ModelOptimizationMode.PTQ_DYNAMIC_RANGE
      )

    if quant_mode.is_post_training_int8_quantization():
      self._metadata.options.modelOptimizationModes.append(
          conversion_metadata_fb.ModelOptimizationMode.PTQ_FULL_INTEGER
      )

    if quant_mode.is_post_training_int16x8_quantization():
      self._metadata.options.modelOptimizationModes.append(
          conversion_metadata_fb.ModelOptimizationMode.PTQ_INT16
      )

    if quant_mode.is_quantization_aware_training():
      self._metadata.options.modelOptimizationModes.append(
          conversion_metadata_fb.ModelOptimizationMode.QUANTIZATION_AWARE_TRAINING
      )

  def _set_conversion_latency_metric(self, value):
    # [FUNCTION DOC]
    # HELPER: stores the conversion latency (a number, milliseconds) in the metrics recorder.
    self._tflite_metrics.set_converter_latency(value)

  @convert_phase(Component.OPTIMIZE_TFLITE_MODEL)
  def _optimize_tflite_model(
      self, model, quant_mode, debug_options, quant_io=True
  ):
    # [FUNCTION DOC]
    # HELPER: post-conversion optimization stage (a tracked conversion phase).
    # Parameters: model (serialized flatbuffer bytes), quant_mode (QuantizationMode),
    # debug_options (quantization debug options message), quant_io (bool, default true; when
    # false the quantizer is told to keep float model inputs/outputs). If integer quantization
    # applies and the StableHLO quantizer is off: for post-training integer quantization runs
    # the calibrate/quantize step; then adjusts model input/output types unless the MLIR
    # quantizer already handled them. Sparsifies the model when requested. Unless buffer offsets
    # are used, deduplicates read-only buffers (skipped silently with a warning on any failure,
    # and skipped if the model already uses buffer offsets). Returns the resulting model bytes.
    """Apply optimizations on a TFLite model."""

    # Disable TFLite quantization pass when
    # `experimental_use_stablehlo_quantizer` is set to `True`. StableHLO
    # Quantizer performs quantization during the conversion step, which happens
    # before `_optimize_tflite_model`.
    if (
        quant_mode.is_integer_quantization()
        and not self.experimental_use_stablehlo_quantizer
    ):
      in_type, out_type = self.inference_input_type, self.inference_output_type

      if quant_mode.is_post_training_integer_quantization():
        q_in_type = in_type if in_type and quant_io else _dtypes.float32
        q_out_type = out_type if out_type and quant_io else _dtypes.float32
        q_activations_type = quant_mode.activations_type()
        q_bias_type = quant_mode.bias_type()
        q_allow_float = quant_mode.is_allow_float()
        q_variable_quantization = quant_mode.enable_mlir_variable_quantization
        model = self._quantize(
            model,
            q_in_type,
            q_out_type,
            q_activations_type,
            q_bias_type,
            q_allow_float,
            q_variable_quantization,
            debug_options,
        )

      m_in_type = in_type if in_type else _dtypes.float32
      m_out_type = out_type if out_type else _dtypes.float32
      # Skip updating model io types if MLIR quantizer already takes care of it
      if not (
          quant_mode.is_post_training_integer_quantization()
          and self.experimental_new_quantizer
          and quant_io
          and (m_in_type in [_dtypes.int8, _dtypes.uint8, _dtypes.float32])
          and (m_out_type in [_dtypes.int8, _dtypes.uint8, _dtypes.float32])
      ):
        model = _modify_model_io_type(model, m_in_type, m_out_type)

    if self._sparsify_model():
      model = _mlir_sparsify(model)

    if not self._experimental_use_buffer_offset:
      try:
        model_object = flatbuffer_utils.convert_bytearray_to_object(model)
        if _check_model_use_buffer_offset(model_object):
          return model
        model = _deduplicate_readonly_buffers(model)
      except Exception:  # pylint: disable=broad-except
        # Skip buffer deduplication when flatbuffer library is not ready to be
        # utilized.
        logging.warning(
            "Buffer deduplication procedure will be skipped when flatbuffer "
            "library is not properly loaded"
        )

    return model

  def _convert_and_export_metrics(self, convert_func, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: runs a conversion function while recording and exporting metrics.
    # KNOWN ISSUE: the docstring says it returns a decorator; it actually runs the conversion
    # immediately and returns the converted model bytes.
    # Parameters: convert_func (unbound conversion function taking the converter first), plus
    # positional and keyword arguments forwarded to it. Increments the attempt counter, saves
    # conversion params (without a graph definition), times the call by CPU process time, counts
    # success if the result is non-empty, records latency in ms and exports metrics. Unless
    # metadata is excluded or buffer offsets are used (or the result already uses buffer
    # offsets), parses the model, adds sparsity modes, model hash and the conversion metadata
    # and serializes it again. Returns the model bytes (with metadata when applicable).
    """Wraps around convert function to export metrics.

    Args:
      convert_func: The convert function to wrap.
      *args: Positional arguments of the convert function.
      **kwargs: The keyword arguments of the convert function.

    Returns:
      The decorator to wrap the convert function.
    """
    self._increase_conversion_attempt_metric()
    self._save_conversion_params_metric()
    start_time = time.process_time()
    result = convert_func(self, *args, **kwargs)
    elapsed_time_ms = (time.process_time() - start_time) * 1000
    if result:
      self._increase_conversion_success_metric()
    self._set_conversion_latency_metric(round(elapsed_time_ms))
    self._tflite_metrics.export_metrics()
    if self.exclude_conversion_metadata or self._experimental_use_buffer_offset:
      return result
    # TODO(b/286886803): add support for adding user metadata with
    # use_buffer_offset flags
    model_object = flatbuffer_utils.convert_bytearray_to_object(result)
    if _check_model_use_buffer_offset(model_object):
      return result
    # Populates the conversion metadata.
    # TODO(b/202090541): Collects sparsity block size information.
    sparsity_modes = _get_sparsity_modes(model_object)
    model_hash = _get_model_hash(model_object)
    self._metadata.options.modelOptimizationModes.extend(sparsity_modes)
    self._metadata.environment.modelHash = model_hash
    model_object = _populate_conversion_metadata(model_object, self._metadata)
    return flatbuffer_utils.convert_object_to_bytearray(model_object)


def _check_model_use_buffer_offset(model_object):
  # [FUNCTION DOC]
  # HELPER: inspects a parsed model object (flatbuffer object form). Returns true if its
  # metadata list contains an entry named buffer location (constants stored by offset), false
  # if there is no metadata or no such entry.
  """Checks if a model object uses buffer offsets to store constant buffers.

  Args:
    model_object: tflite model, a python object

  Returns:
    True of the model_object has the metadata entry "buffer_location"
    False otherwise
  """
  if not model_object.metadata:
    return False
  for meta in model_object.metadata:
    if meta.name.decode("utf-8") == "buffer_location":
      return True

  return False


def _export_metrics(convert_func):
  # [FUNCTION DOC]
  # HELPER: decorator factory for conversion methods. Takes the conversion method and returns
  # a wrapper that calls the converter's metrics-exporting runner with that method.
  """The decorator around convert function to export metrics."""

  @functools.wraps(convert_func)
  def wrapper(self, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: wrapper produced by the metrics decorator. Calls the converter's
    # convert-and-export-metrics method with the wrapped function and the given arguments, and
    # returns its result (the converted model bytes).
    # pylint: disable=protected-access
    return self._convert_and_export_metrics(convert_func, *args, **kwargs)
    # pylint: enable=protected-access

  return wrapper


class TFLiteConverterBaseV2(TFLiteConverterBase):
  # [CLASS DOC]
  # Intermediate base class for the V2 (TensorFlow 2) converters. Adds the inference input and
  # output type options (default float32), marks API version 2 in the metadata, and provides
  # shared steps: input/output type validation, saved model loading, input validation with
  # debug info extraction, Grappler optimization, and the common conversion pipeline pieces.
  """Converter subclass to share functionality between V2 converters."""

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor with no arguments. Calls the base constructor, sets the inference input and
    # output types to float32 and records API version 2 in the conversion metadata.
    """Constructor for TFLiteConverter."""
    super(TFLiteConverterBaseV2, self).__init__()
    self.inference_input_type = _dtypes.float32
    self.inference_output_type = _dtypes.float32
    self._metadata.environment.apiVersion = 2

  def _validate_inference_input_output_types(self, quant_mode):
    # [FUNCTION DOC]
    # HELPER: validates the inference input and output types against the quantization mode.
    # Parameter quant_mode: QuantizationMode. For integer quantization the allowed types are
    # float32 plus int16 (int16x8 mode) or float32, int8, uint8 and int16. For anything else
    # only float32 is allowed. Raises a value error naming the allowed types if either the
    # input or output type is not allowed. Returns nothing.
    """Validate inference_input_type and inference_output_type flags."""
    default_types = [_dtypes.float32]
    # We support integer input/output for integer quantized models only.
    if quant_mode.is_integer_quantization():
      if quant_mode.is_post_training_int16x8_quantization():
        all_types = default_types + [_dtypes.int16]
      else:
        all_types = default_types + [_dtypes.int8, _dtypes.uint8, _dtypes.int16]
      if (
          self.inference_input_type not in all_types
          or self.inference_output_type not in all_types
      ):
        all_types_names = ["tf." + t.name for t in all_types]
        raise ValueError(
            "The inference_input_type and inference_output_type "
            "must be in {}.".format(all_types_names)
        )
    elif (
        self.inference_input_type not in default_types
        or self.inference_output_type not in default_types
    ):
      raise ValueError(
          "The inference_input_type and inference_output_type "
          "must be tf.float32."
      )

  @convert_phase(Component.PREPARE_TF_MODEL, SubComponent.LOAD_SAVED_MODEL)
  def _load_saved_model(self, saved_model_dir, saved_model_tags):
    # [FUNCTION DOC]
    # HELPER: loads a saved model graph for conversion (tracked conversion phase).
    # Parameters: saved_model_dir (directory path), saved_model_tags (set of tag strings).
    # Loads the graph into a fresh graph object and fetches the default serving signature.
    # Returns a triple: the graph definition, the list of input tensors and the list of output
    # tensors from that signature. Raises whatever the loader raises if the model or signature
    # is missing.
    """Load graph_def from saved model with the default serving signature key.

    Args:
      saved_model_dir: Directory of the SavedModel.
      saved_model_tags: Set of tags identifying the MetaGraphDef within the
        SavedModel to analyze.

    Returns:
      graph_def: The loaded GraphDef.
      input_tensors: List of input tensors.
      output_tensors: List of output tensors.
    """
    graph = _ops.Graph()
    saved_model = _loader_impl.SavedModelLoader(saved_model_dir)
    saved_model.load_graph(graph, tags=saved_model_tags)
    meta_graph = saved_model.get_meta_graph_def_from_tags(saved_model_tags)
    graph_def = meta_graph.graph_def
    signature_def = meta_graph.signature_def[
        _signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    ]
    input_tensors = [
        graph.get_tensor_by_name(signature_def.inputs[key].name)
        for key in signature_def.inputs
    ]
    output_tensors = [
        graph.get_tensor_by_name(signature_def.outputs[key].name)
        for key in signature_def.outputs
    ]
    return graph_def, input_tensors, output_tensors

  @convert_phase(Component.PREPARE_TF_MODEL, SubComponent.VALIDATE_INPUTS)
  def _validate_inputs(self, graph_def, input_tensors):
    # [FUNCTION DOC]
    # HELPER: input validation phase of V2 conversion.
    # Parameters: graph_def (TensorFlow graph definition), input_tensors (list of tensors).
    # Saves conversion params metrics, builds and stores the quantization mode on the converter
    # (which may raise a value error for bad options), validates inference input/output types,
    # and, when unknown shapes are not allowed, raises a value error if a non-leading dimension
    # of an input is unknown, or sets an unknown leading dimension to 1 (mutating the tensor
    # shape). Finally extracts debug info (stack traces) from the first concrete function graph
    # or from the trackable object's graph debug info and stores it on the converter.
    # Returns nothing.
    """Validate the input parameters.

    Args:
      graph_def: The TensorFlow GraphDef.
      input_tensors: List of input tensors.

    Raise:
      ValueError: Input shape is not specified. Invalid quantization parameters.
    """
    # Update conversion params with graph_def.
    self._save_conversion_params_metric(graph_def)
    self._quant_mode = QuantizationMode(
        self.optimizations,
        self.target_spec,
        self.representative_dataset,
        graph_def,
        self._experimental_disable_per_channel,
        self.experimental_new_dynamic_range_quantizer,
        self._experimental_low_bit_qat,
        self._experimental_full_integer_quantization_bias_type,
        self._experimental_variable_quantization,
        self._experimental_strict_qdq,
    )
    self._validate_inference_input_output_types(self._quant_mode)

    if not self._is_unknown_shapes_allowed():
      # Checks dimensions in input tensor.
      for tensor in input_tensors:
        # Note that shape_list might be empty for scalar shapes.
        shape_list = tensor.shape.as_list()
        if None in shape_list[1:]:
          raise ValueError(
              "None is only supported in the 1st dimension. Tensor '{0}' has "
              "invalid shape '{1}'.".format(
                  _get_tensor_name(tensor), shape_list
              )
          )
        elif shape_list and shape_list[0] is None:
          # Set the batch size to 1 if undefined.
          shape = tensor.shape.as_list()
          shape[0] = 1
          tensor.set_shape(shape)

    if self._trackable_obj is None or not hasattr(
        self._trackable_obj, "graph_debug_info"
    ):
      self._debug_info = _get_debug_info(
          _build_debug_info_func(self._funcs[0].graph), graph_def
      )
    else:
      self._debug_info = _get_debug_info(
          _convert_debug_info_func(self._trackable_obj.graph_debug_info),
          graph_def,
      )

  @convert_phase(Component.PREPARE_TF_MODEL, SubComponent.OPTIMIZE_TF_MODEL)
  def _optimize_tf_model(
      self, graph_def, input_tensors, output_tensors, frozen_func
  ):
    # [FUNCTION DOC]
    # HELPER: Grappler optimization phase of V2 conversion (tracked conversion phase).
    # Parameters: graph_def (frozen graph definition), input_tensors and output_tensors (lists
    # of tensors), frozen_func (the frozen concrete function whose graph is used). Builds the
    # Grappler config; only if it lists at least one optimizer, runs the graph optimizations
    # with it (otherwise Grappler would run its default set, which is avoided on purpose).
    # Returns the optimized graph definition, or the unchanged input when nothing was run.
    """Run a Grappler pass to optimize the TensorFlow graph.

    Args:
      graph_def: Frozen GraphDef to be optimized.
      input_tensors: List of input tensors.
      output_tensors: List of output tensors.
      frozen_func: TensorFlow Graph.

    Returns:
      The optimized TensorFlow graph.
    """
    grappler_config = self._grappler_config()
    # Skip running grappler when there are no optimizers to run. If not,
    # grappler will run with the default optimizer set and it will lead to
    # causing an unexpected behavior.
    if grappler_config.graph_options.rewrite_options.optimizers:
      graph_def = _run_graph_optimizations(
          graph_def,
          input_tensors,
          output_tensors,
          config=grappler_config,
          graph=frozen_func.graph,
      )
    return graph_def

  def _convert_from_saved_model(self, graph_def):
    # [FUNCTION DOC]
    # HELPER: converts directly from the configured saved model directory via the MLIR saved
    # model import path.
    # Parameter graph_def: graph definition of the model, used only for statistics and the
    # quantization-aware-training check. Saves conversion params, builds the quantization mode
    # (may raise a value error), validates inference input/output types, assembles converter
    # arguments (resource variables flag, base args, quantization flags), runs the saved model
    # conversion and then the post-conversion optimization stage (quantizing inputs/outputs only
    # if the new quantizer is on). Returns the converted model bytes.
    """Helper method that converts saved model.

    Args:
      graph_def: GraphDef object for the model, used only for stats.

    Returns:
      The converted TFLite model.
    """
    # Update conversion params with graph_def.
    self._save_conversion_params_metric(graph_def)
    # Get quantization options and do some sanity checks.
    quant_mode = QuantizationMode(
        self.optimizations,
        self.target_spec,
        self.representative_dataset,
        graph_def,
        self._experimental_disable_per_channel,
        self.experimental_new_dynamic_range_quantizer,
        self._experimental_low_bit_qat,
        self._experimental_full_integer_quantization_bias_type,
        self._experimental_variable_quantization,
        self._experimental_strict_qdq,
    )
    self._validate_inference_input_output_types(quant_mode)
    converter_kwargs = {
        "enable_tflite_resource_variables": (
            self.experimental_enable_resource_variables
        )
    }
    converter_kwargs.update(self._get_base_converter_args())
    converter_kwargs.update(quant_mode.converter_flags())

    result = _convert_saved_model(**converter_kwargs)
    return self._optimize_tflite_model(
        result,
        quant_mode,
        _build_conversion_flags(**converter_kwargs).debug_options,
        quant_io=self.experimental_new_quantizer,
    )

  def convert(self, graph_def, input_tensors, output_tensors):
    # [FUNCTION DOC]
    # Shared V2 conversion step from a frozen graph definition (subclasses call it through the
    # superclass after preparing the graph).
    # Parameters: graph_def (frozen graph definition), input_tensors and output_tensors (lists
    # of tensors). Validates inputs and builds the quantization mode (value errors on bad shapes
    # or quantization options), collects converter args and quantization flags, logs a
    # deprecation warning when the new converter is off or an info message when it is on, runs
    # the graph definition conversion and then the post-conversion optimizations. Returns the
    # converted model bytes.
    """Converts a TensorFlow GraphDef based on instance variables.

    Args:
      graph_def: Frozen TensorFlow GraphDef.
      input_tensors: List of input tensors.
      output_tensors: List of output tensors.

    Returns:
      The converted data in serialized format.

    Raises:
      ValueError:
        No concrete function is specified.
        Multiple concrete functions are specified.
        Input shape is not specified.
        Invalid quantization parameters.
    """
    self._validate_inputs(graph_def, input_tensors)
    converter_kwargs = self._get_base_converter_args()
    converter_kwargs.update(self._quant_mode.converter_flags())
    if not self.experimental_new_converter:
      logging.warning(
          "Please consider switching to the new converter by setting "
          "experimental_new_converter=True. "
          "The old converter is deprecated."
      )
    else:
      logging.info(
          "Using new converter: If you encounter a problem "
          "please file a bug. You can opt-out "
          "by setting experimental_new_converter=False"
      )

    # Converts model.
    result = _convert_graphdef(
        input_data=graph_def,
        input_tensors=input_tensors,
        output_tensors=output_tensors,
        **converter_kwargs,
    )

    return self._optimize_tflite_model(
        result,
        self._quant_mode,
        _build_conversion_flags(**converter_kwargs).debug_options,
        quant_io=self.experimental_new_quantizer,
    )


class TFLiteSavedModelConverterV2(TFLiteConverterBaseV2):
  # [CLASS DOC]
  # V2 converter whose source is a SavedModel directory. Holds the saved model directory, tags,
  # exported names and an optional trackable object kept alive so variables are not collected.
  # Created by the public converter's saved model factory; its convert method uses the MLIR
  # saved model import path when possible and falls back to freezing the graph otherwise.
  """Converts the given SavedModel into TensorFlow Lite model.

  Attributes:
      saved_model_dir: Directory of the SavedModel.
  """

  def __init__(
      self,
      saved_model_dir,
      saved_model_tags=None,
      saved_model_exported_names=None,
      trackable_obj=None,
  ):
    # [FUNCTION DOC]
    # Constructor. Parameters: saved_model_dir (path of the SavedModel), saved_model_tags
    # (optional set of tags), saved_model_exported_names (optional list of signature names),
    # trackable_obj (optional object kept referenced). Stores them, then parses the saved model
    # arguments with import forced on, which may clear the directory (unreadable model or
    # converter disabled) or raise a value error for unsupported schema versions.
    """Constructor for TFLiteConverter.

    Args:
      saved_model_dir: Directory of the SavedModel.
      saved_model_tags: Set of tags identifying the MetaGraphDef within the
        SavedModel to analyze. All tags in the tag set must be present. (default
        {tf.saved_model.SERVING}).
      saved_model_exported_names: Names to be exported when the saved model
        import path is on.
      trackable_obj: tf.AutoTrackable object associated with `funcs`. A
        reference to this object needs to be maintained so that Variables do not
        get garbage collected since functions have a weak reference to
        Variables. This is only required when the tf.AutoTrackable object is not
        maintained by the user (e.g. `from_saved_model`).
    """
    super(TFLiteSavedModelConverterV2, self).__init__()
    self.saved_model_dir = saved_model_dir
    self._saved_model_tags = saved_model_tags
    self._saved_model_exported_names = saved_model_exported_names
    self._trackable_obj = trackable_obj
    self._parse_saved_model_args(always_enable_saved_model_import=True)

  @_export_metrics
  def convert(self):
    # [FUNCTION DOC]
    # Converts the saved model to a TFLite model (wrapped to record metrics). Takes no
    # arguments. Loads the default serving signature graph. If the saved model import path is
    # not usable (directory cleared or new converter off), freezes the saved model, clears the
    # directory and converts via the shared graph definition path. Otherwise loads the object to
    # extract debug info, frees it and forces garbage collection, then converts through the
    # saved model import path. Returns converted model bytes.
    # KNOWN ISSUE: the debug info branch used when the loaded object is nothing refers to a
    # functions list attribute that this class never sets; the branch would fail if reached.
    # The docstring's mention of concrete function errors does not apply to this converter.
    """Converts a TensorFlow GraphDef based on instance variables.

    Returns:
      The converted data in serialized format.

    Raises:
      ValueError:
        No concrete function is specified.
        Multiple concrete functions are specified.
        Input shape is not specified.
        Invalid quantization parameters.
    """
    graph_def, input_tensors, output_tensors = self._load_saved_model(
        self.saved_model_dir, self._saved_model_tags
    )
    # If we can't use saved model importer, then fallback
    # to frozen graph conversion path.
    if self.saved_model_dir is None or not self.experimental_new_converter:
      graph_def, _, _, _ = _freeze_saved_model(
          self.saved_model_dir,
          None,
          None,
          None,
          self._saved_model_tags,
          _signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY,
      )
      # We make sure to clear the saved_model_dir as there is some
      # legacy code down in the caller that checks this.
      self.saved_model_dir = None
      return super(TFLiteSavedModelConverterV2, self).convert(
          graph_def, input_tensors, output_tensors
      )

    trackable_obj = _load(self.saved_model_dir, self._saved_model_tags)
    if trackable_obj is None:
      self._debug_info = _get_debug_info(
          _build_debug_info_func(self._funcs[0].graph), graph_def
      )
    else:
      self._debug_info = _get_debug_info(
          _convert_debug_info_func(trackable_obj.graph_debug_info),
          graph_def,
      )

    del trackable_obj
    gc.collect()
    return self._convert_from_saved_model(graph_def)


class TFLiteKerasModelConverterV2(TFLiteConverterBaseV2):
  # [CLASS DOC]
  # V2 converter whose source is a Keras model. Holds the model, an optional trackable object
  # and a flag controlling lowering through a saved model. Convert first tries exporting the
  # model to a temporary saved model and converting that; if this is not possible it freezes
  # the Keras model into a graph and converts it.
  """Converts the given Keras model into TensorFlow Lite model."""

  def __init__(self, keras_model, trackable_obj=None):
    # [FUNCTION DOC]
    # Constructor. Parameters: keras_model (Keras model object), trackable_obj (optional object
    # kept referenced so variables are not collected). Stores them and enables the lower to
    # saved model option. Returns nothing.
    """Constructor for TFLiteConverter.

    Args:
      keras_model: tf.Keras.Model.
      trackable_obj: tf.AutoTrackable object associated with `funcs`. A
        reference to this object needs to be maintained so that Variables do not
        get garbage collected since functions have a weak reference to
        Variables. This is only required when the tf.AutoTrackable object is not
        maintained by the user (e.g. `from_saved_model`).
    """
    super(TFLiteKerasModelConverterV2, self).__init__()
    self._keras_model = keras_model
    self._trackable_obj = trackable_obj
    self.experimental_lower_to_saved_model = True

  @convert_phase(
      Component.PREPARE_TF_MODEL, SubComponent.CONVERT_KERAS_TO_SAVED_MODEL
  )
  def _convert_keras_to_saved_model(self, output_dir):
    # [FUNCTION DOC]
    # HELPER: exports the Keras model as a saved model and reloads it (tracked conversion phase).
    # Parameter output_dir: directory to write the saved model into (side effect: files written).
    # Keras 3 models are exported through an export archive with a serving endpoint traced with
    # training off, using the model's input signature (functional/sequential) or its save spec
    # (raises a value error, swallowed by the broad handler, if the model was never called);
    # other models are saved through the regular saved model saver with debug info. Any failure
    # during export returns three nothing values so the caller falls back. On success sets the
    # converter's saved model dir, serving tag and default signature name, parses saved model
    # args (may clear the directory), and if still set loads graph def, inputs and outputs and
    # replaces the trackable object with the reloaded object. Returns the triple of graph
    # definition, input tensors and output tensors, or three nothing values.
    """Save Keras model to the SavedModel format.

    Args:
      output_dir: The output directory to save the SavedModel.

    Returns:
      graph_def: The frozen GraphDef.
      input_tensors: List of input tensors.
      output_tensors: List of output tensors.
    """
    try:

      def _is_keras_3():
        # [FUNCTION DOC]
        # HELPER: detects a Keras 3 model. Tries to import the keras package; returns true if
        # its version string starts with 3 and the converter's model is a Keras layer, false
        # otherwise or if keras cannot be imported.
        """Returns true if _keras_model is a Keras 3+ model."""
        try:
          import keras  # pylint: disable=g-import-not-at-top

          return keras.__version__.startswith("3") and isinstance(
              self._keras_model, keras.layers.Layer
          )
        except ImportError:
          return False

      if _is_keras_3():
        import keras  # pylint: disable=g-import-not-at-top

        # Keras 3 model `export` by default saves model.__call__ with
        # training=True. Need to export the model call with training=False for
        # inference only and TFLite conversion.
        export_archive = keras.export.ExportArchive()
        export_archive.track(self._keras_model)
        if isinstance(
            self._keras_model,
            (keras.src.models.Functional, keras.src.models.Sequential),
        ):
          input_signature = nest.map_structure(
              lambda x: tensor_spec.TensorSpec(
                  x.shape, dtype=x.dtype, name=x.name
              ),
              self._keras_model.inputs,
          )
          if isinstance(input_signature, list) and len(input_signature) > 1:
            input_signature = [input_signature]
        else:
          save_spec = _get_save_spec(self._keras_model)
          if not save_spec:
            raise ValueError(
                "The model provided has never been called. "
                "It must be called at least once before export."
            )
          input_signature = [save_spec]
        inference_fn = functools.partial(
            self._keras_model.__call__, training=False
        )
        export_archive.add_endpoint("serve", inference_fn, input_signature)
        export_archive.write_out(output_dir)
      else:
        _save.save(
            self._keras_model,
            output_dir,
            options=_save_options.SaveOptions(save_debug_info=True),
        )
    except Exception:  # pylint: disable=broad-except
      # When storing the given keras model to a saved model is failed, let's
      # use original keras model conversion pipeline.
      return None, None, None
    self.saved_model_dir = output_dir
    self._saved_model_tags = set([_tag_constants.SERVING])
    self._saved_model_exported_names = [
        _signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    ]
    self._parse_saved_model_args(
        always_enable_saved_model_import=self.experimental_lower_to_saved_model
    )
    if self.saved_model_dir:
      graph_def, input_tensors, output_tensors = self._load_saved_model(
          self.saved_model_dir, self._saved_model_tags
      )
      self._trackable_obj = _load(self.saved_model_dir, self._saved_model_tags)
      return graph_def, input_tensors, output_tensors
    return None, None, None

  @convert_phase(Component.PREPARE_TF_MODEL, SubComponent.FREEZE_KERAS_MODEL)
  def _freeze_keras_model(self):
    # [FUNCTION DOC]
    # HELPER: fallback preparation (tracked conversion phase) that freezes the Keras model.
    # Takes no arguments. If the model's call is not already a traced function, derives an input
    # signature keeping the user's batch size; traces the model, gets its concrete function,
    # stores it as the only entry of the converter's function list (side effect), converts
    # variables to constants without lowering control flow. Returns four values: the frozen
    # graph definition, the input tensors (resource-typed ones excluded), the output tensors
    # and the frozen function.
    """Freeze Keras model to frozen graph.

    Returns:
      graph_def: The frozen GraphDef.
      input_tensors: List of input tensors.
      output_tensors: List of output tensors.
      frozen_func: The frozen ConcreteFunction.
    """
    input_signature = None
    # If the model's call is not a `tf.function`, then we need to first get its
    # input signature from `model_input_signature` method. We can't directly
    # call `trace_model_call` because otherwise the batch dimension is set
    # to None.
    # Once we have better support for dynamic shapes, we can remove this.
    if not isinstance(self._keras_model.call, _def_function.Function):
      # Pass `keep_original_batch_size=True` will ensure that we get an input
      # signature including the batch dimension specified by the user.
      # TODO(b/169898786): Use the Keras public API when TFLite moves out of TF
      input_signature = _model_input_signature(
          self._keras_model, keep_original_batch_size=True
      )

    # TODO(b/169898786): Use the Keras public API when TFLite moves out of TF
    func = _trace_model_call(self._keras_model, input_signature)
    concrete_func = func.get_concrete_function()
    self._funcs = [concrete_func]

    frozen_func, graph_def = (
        _convert_to_constants.convert_variables_to_constants_v2_as_graph(
            self._funcs[0], lower_control_flow=False
        )
    )

    input_tensors = [
        tensor
        for tensor in frozen_func.inputs
        if tensor.dtype != _dtypes.resource
    ]
    output_tensors = frozen_func.outputs
    return graph_def, input_tensors, output_tensors, frozen_func

  def _convert_as_saved_model(self):
    # [FUNCTION DOC]
    # HELPER: tries converting the Keras model through a temporary saved model.
    # Creates a temporary directory (always removed afterwards), exports the model into it and,
    # if the export left a saved model directory set, converts via the shared graph definition
    # conversion. Returns the converted bytes, or nothing if the export was not possible.
    """Converts a Keras model as a saved model.

    Returns:
      The converted data in serialized format.
    """
    temp_dir = tempfile.mkdtemp()
    try:
      graph_def, input_tensors, output_tensors = (
          self._convert_keras_to_saved_model(temp_dir)
      )
      if self.saved_model_dir:
        return super(TFLiteKerasModelConverterV2, self).convert(
            graph_def, input_tensors, output_tensors
        )
    finally:
      shutil.rmtree(temp_dir, True)

  @_export_metrics
  def convert(self):
    # [FUNCTION DOC]
    # Converts the Keras model to a TFLite model (wrapped to record metrics). Takes no
    # arguments. First tries the saved model route; if that gives a result, returns it.
    # Otherwise freezes the Keras model, runs Grappler optimization and converts through the
    # shared graph definition path. Returns converted model bytes; errors (value errors for
    # shapes or quantization options) propagate.
    """Converts a keras model based on instance variables.

    Returns:
      The converted data in serialized format.

    Raises:
      ValueError:
        Multiple concrete functions are specified.
        Input shape is not specified.
        Invalid quantization parameters.
    """
    saved_model_convert_result = self._convert_as_saved_model()
    if saved_model_convert_result:
      return saved_model_convert_result

    graph_def, input_tensors, output_tensors, frozen_func = (
        self._freeze_keras_model()
    )

    graph_def = self._optimize_tf_model(
        graph_def, input_tensors, output_tensors, frozen_func
    )

    return super(TFLiteKerasModelConverterV2, self).convert(
        graph_def, input_tensors, output_tensors
    )


class TFLiteFrozenGraphConverterV2(TFLiteConverterBaseV2):
  # [CLASS DOC]
  # V2 converter whose source is a single TensorFlow concrete function (given as a list).
  # Holds the function list, an optional trackable object and the lower to saved model flag.
  # Convert tries to serialize the functions as a saved model first, and otherwise freezes the
  # concrete function and converts the frozen graph.
  """Converts the given frozen graph into TensorFlow Lite model."""

  def __init__(self, funcs, trackable_obj=None):
    # [FUNCTION DOC]
    # Constructor. Parameters: funcs (list of concrete functions without duplicates),
    # trackable_obj (optional object kept referenced so variables are not collected). Stores
    # them and enables the lower to saved model option. Returns nothing.
    """Constructor for TFLiteConverter.

    Args:
      funcs: List of TensorFlow ConcreteFunctions. The list should not contain
        duplicate elements.
      trackable_obj: tf.AutoTrackable object associated with `funcs`. A
        reference to this object needs to be maintained so that Variables do not
        get garbage collected since functions have a weak reference to
        Variables. This is only required when the tf.AutoTrackable object is not
        maintained by the user (e.g. `from_saved_model`).
    """
    super(TFLiteFrozenGraphConverterV2, self).__init__()
    self._funcs = funcs
    self._trackable_obj = trackable_obj
    self.experimental_lower_to_saved_model = True

  @convert_phase(
      Component.PREPARE_TF_MODEL, SubComponent.FREEZE_CONCRETE_FUNCTION
  )
  def _freeze_concrete_function(self):
    # [FUNCTION DOC]
    # HELPER: freezes the single concrete function (tracked conversion phase). Raises a value
    # error if the function list is empty or has more than one entry. Converts variables to
    # constants without lowering control flow. Returns four values: frozen graph definition,
    # input tensors (resource-typed excluded), output tensors and the frozen function.
    """Convert the given ConcreteFunction to frozen graph.

    Returns:
      graph_def: The frozen GraphDef.
      input_tensors: List of input tensors.
      output_tensors: List of output tensors.
      frozen_func: The frozen ConcreteFunction.

    Raises:
      ValueError: none or multiple ConcreteFunctions provided.
    """
    if len(self._funcs) == 0:  # pylint: disable=g-explicit-length-test
      raise ValueError("No ConcreteFunction is specified.")

    if len(self._funcs) > 1:
      raise ValueError(
          "This converter can only convert a single "
          "ConcreteFunction. Converting multiple functions is "
          "under development."
      )

    frozen_func, graph_def = (
        _convert_to_constants.convert_variables_to_constants_v2_as_graph(
            self._funcs[0], lower_control_flow=False
        )
    )

    input_tensors = [
        tensor
        for tensor in frozen_func.inputs
        if tensor.dtype != _dtypes.resource
    ]
    output_tensors = frozen_func.outputs
    return graph_def, input_tensors, output_tensors, frozen_func

  @convert_phase(
      Component.PREPARE_TF_MODEL,
      SubComponent.CONVERT_CONCRETE_FUNCTIONS_TO_SAVED_MODEL,
  )
  def _convert_concrete_functions_to_saved_model(self, output_dir):
    # [FUNCTION DOC]
    # HELPER: serializes the concrete functions as a saved model and reloads it (tracked phase).
    # Parameter output_dir: directory to write to (side effect: files written). Raises a value
    # error if the function list is empty. Returns three nothing values (meaning fall back to
    # freezing) if lowering is disabled, if no trackable object was supplied or it is itself a
    # function, or if saving fails for any reason. Otherwise saves with signatures (the default
    # serving key for one function, or each function's graph name for several), then sets the
    # saved model dir, serving tag and signature names, parses saved model args (forced import;
    # may clear the directory) and, if still set, loads and returns graph definition, input
    # tensors and output tensors, replacing the trackable object with the reloaded one.
    """Save concrete functions to the SavedModel format.

    Args:
      output_dir: The output directory to save the SavedModel.

    Returns:
      graph_def: The frozen GraphDef.
      input_tensors: List of input tensors.
      output_tensors: List of output tensors.
    """
    if len(self._funcs) == 0:  # pylint: disable=g-explicit-length-test
      raise ValueError("No ConcreteFunction is specified.")

    if not self.experimental_lower_to_saved_model:
      return None, None, None

    # Without the provided trackable obj, it is not able to serialize the given
    # concrete functions as a saved model format. Also when trackable obj is
    # a function, use the original concrete function conversion pipeline.
    if not self._trackable_obj or isinstance(
        self._trackable_obj,
        (_function.ConcreteFunction, _def_function.Function),
    ):
      return None, None, None

    signatures = {}
    signature_keys = []
    try:
      if len(self._funcs) == 1:
        signatures[_signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY] = (
            self._funcs[0]
        )
        signature_keys = [
            _signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
        ]
      else:
        for func in self._funcs:
          signatures[func.graph.name] = func
          signature_keys.append(func.graph.name)

      _save.save(
          self._trackable_obj,
          output_dir,
          signatures=signatures,
          options=_save_options.SaveOptions(save_debug_info=True),
      )
    except Exception:  # pylint: disable=broad-except
      # When storing the given concrete function to a saved model is failed,
      # let's use original concrete function conversion pipeline.
      return None, None, None

    self.saved_model_dir = output_dir
    self._saved_model_tags = set([_tag_constants.SERVING])
    self._saved_model_exported_names = signature_keys
    self._parse_saved_model_args(always_enable_saved_model_import=True)
    if self.saved_model_dir:
      graph_def, input_tensors, output_tensors = self._load_saved_model(
          self.saved_model_dir, self._saved_model_tags
      )
      self._trackable_obj = _load(self.saved_model_dir, self._saved_model_tags)
      return graph_def, input_tensors, output_tensors
    return None, None, None

  def _convert_as_saved_model(self):
    # [FUNCTION DOC]
    # HELPER: tries converting through a temporary saved model. Creates a temporary directory
    # (always removed afterwards), serializes the concrete functions into it and, if a saved
    # model directory results, validates inputs and converts via the saved model import path.
    # Returns the converted bytes, or nothing if serialization was not possible.
    """Converts the given concrete functions as a saved model format.

    Returns:
      The converted data in serialized format.
    """
    temp_dir = tempfile.mkdtemp()
    try:
      graph_def, input_tensors, _ = (
          self._convert_concrete_functions_to_saved_model(temp_dir)
      )
      if self.saved_model_dir:
        self._validate_inputs(graph_def, input_tensors)
        return self._convert_from_saved_model(graph_def)
    finally:
      shutil.rmtree(temp_dir, True)
    return None

  @_export_metrics
  def convert(self):
    # [FUNCTION DOC]
    # Converts the concrete function to a TFLite model (wrapped to record metrics). Takes no
    # arguments. If lowering to a saved model is enabled, tries that route and returns its
    # result when available. Otherwise freezes the concrete function, runs Grappler optimization
    # and converts through the shared graph definition path. Returns converted model bytes.
    # Raises value errors for an empty or multi-function list, unknown input shapes (when not
    # allowed) or invalid quantization options.
    """Converts a TensorFlow GraphDef based on instance variables.

    Returns:
      The converted data in serialized format.

    Raises:
      ValueError:
        No concrete function is specified.
        Multiple concrete functions are specified.
        Input shape is not specified.
        Invalid quantization parameters.
    """
    if self.experimental_lower_to_saved_model:
      saved_model_convert_result = self._convert_as_saved_model()
      if saved_model_convert_result:
        return saved_model_convert_result

    graph_def, input_tensors, output_tensors, frozen_func = (
        self._freeze_concrete_function()
    )

    graph_def = self._optimize_tf_model(
        graph_def, input_tensors, output_tensors, frozen_func
    )

    return super(TFLiteFrozenGraphConverterV2, self).convert(
        graph_def, input_tensors, output_tensors
    )


class TFLiteJaxConverterV2(TFLiteConverterBaseV2):
  # [CLASS DOC]
  # V2 converter whose source is one or more JAX serving functions with example inputs. Holds
  # the serving function list and the named input placeholders; convert lowers the function to
  # HLO and runs the HLO-to-TFLite conversion.
  """Converts the given jax model into TensorFlow Lite model."""

  def __init__(self, serving_funcs, inputs):
    # [FUNCTION DOC]
    # Constructor. Parameters: serving_funcs (list of JAX serving functions with parameters
    # already inlined) and inputs (list of pairs of input name and example array giving shape
    # and type). Stores them after calling the base constructor. Returns nothing.
    """Constructor for TFLiteConverter.

    Args:
      serving_funcs: A list functions of the serving func of the jax module, the
        model params should already be inlined. (e.g., `serving_func =
        functools.partial(model, params=params)`)
      inputs: Array of input tensor placeholders tuple,s like `jnp.zeros`. For
        example, wrapped in an array like "[('input1', input1), ('input2',
        input2)]]".

    Jax functions are polymorphic, for example:

    ```python
    def add(a, b):
      return a + b
    ```

    Will yield different computations if different input signatures are passed
    in: Pass `add(10.0, 20.0)` will yield a scalar `add` while pass
    `add(np.random((100, 1)), np.random(100, 100))` will yield a broadcasting
    add.  We will need the input information to do tracing for the converter
    to properly convert the model. So it's important to pass in the desired
    `input placeholders` with the correct input shape/type.

    In the converted tflite model, the function name will be default to "main",
    the output names will be the traced outputs. The output ordering shall
    match the serving function.
    """  # fmt: skip

    super(TFLiteJaxConverterV2, self).__init__()
    self._serving_funcs = serving_funcs
    self._inputs = inputs

  @_export_metrics
  def convert(self):
    # [FUNCTION DOC]
    # Converts the JAX serving function to a TFLite model (wrapped to record metrics). Takes no
    # arguments. Raises import error if the JAX jit helper is unavailable; value errors if no
    # serving function or inputs are given, if the counts of inputs and functions differ, if
    # inputs are not a list or tuple, if more than one function is supplied (unsupported), if
    # the first input group is not a list or tuple, or if tracing and lowering to HLO fails
    # (any underlying error is replaced by a generic failed-to-convert message). Splits the
    # named input pairs into names and arrays, lowers the function for the CPU platform to a
    # serialized HLO proto, builds converter arguments (HLO content, input names, proto format
    # flag, base args), builds the quantization mode with no graph definition, validates
    # inference types, adds quantization flags, runs the HLO conversion and then the
    # post-conversion optimizations. Returns the converted model bytes.
    """Converts a Jax serving func based on instance variables.

    Returns:
      The converted data in serialized format.

    Raises:
      ImportError:
        If cannot import the jit from jax.
      ValueError:
        No serving function is specified.
        Input tensors are not specified.
        The truth value of an array with more than one element is ambiguous.
        Failed to convert the given Jax function to hlo.
    """
    if not _jit:
      raise ImportError("Cannot import jit from jax.")

    if not self._serving_funcs:
      raise ValueError("No serving func is specified.")

    if not self._inputs:
      raise ValueError("Input tensors are not specified.")

    if len(self._inputs) != len(self._serving_funcs):
      msg = (
          "Input tensor mapping len {} does not match serving func len {}."
          .format(len(self._inputs), len(self._serving_funcs))
      )
      raise ValueError(msg)

    if not isinstance(self._inputs, (tuple, list)):
      raise ValueError(
          "Input tensors should be pass in a tuple list wrapped in an array."
      )

    # TODO(b/197690428): Support multiple functions.
    # Currently only support one serving function.
    if len(self._serving_funcs) > 1:
      raise ValueError("Currently only support single serving function.")

    if not isinstance(self._inputs[0], (tuple, list)):
      raise ValueError("The input placeholders are not a dictionary.")

    input_names = []
    ordered_inputs = []
    for input_name, tensor in self._inputs[0]:
      input_names.append(input_name)
      ordered_inputs.append(tensor)

    try:
      hlo_proto = (
          _jit(self._serving_funcs[0])
          .trace(*ordered_inputs)
          .lower(lowering_platforms=("cpu",))
          .compiler_ir("hlo")
          .as_serialized_hlo_module_proto()
      )
    except Exception:  # pylint: disable=broad-except
      raise ValueError("Failed to convert the given Jax function to hlo.")

    # We need to set the hlo proto, and here we use serialized proto format
    # since it's more compact.
    converter_kwargs = {
        "input_content": hlo_proto,
        "input_names": input_names,
        "is_proto_format": True,
    }
    converter_kwargs.update(self._get_base_converter_args())

    # Get quantization options and do some checks.
    quant_mode = QuantizationMode(
        self.optimizations,
        self.target_spec,
        self.representative_dataset,
        None,
        experimental_qdq_annotation=self._experimental_strict_qdq,
    )
    self._validate_inference_input_output_types(quant_mode)
    converter_kwargs.update(quant_mode.converter_flags())
    result = _convert_jax_hlo(**converter_kwargs)

    return self._optimize_tflite_model(
        result,
        quant_mode,
        _build_conversion_flags(**converter_kwargs).debug_options,
        quant_io=self.experimental_new_quantizer,
    )


@_tf_export("lite.TFLiteConverter", v1=[])
class TFLiteConverterV2(TFLiteFrozenGraphConverterV2):
  # [CLASS DOC]
  # Public TensorFlow 2 converter (exported as the lite converter class). Extends the frozen
  # graph converter and is normally obtained through its factory class methods: from concrete
  # functions, from a saved model directory, from a Keras model or (experimental, deprecated)
  # from JAX; the factories may return one of the specialized converter subclasses or, in
  # graph mode, the legacy V1 converter instead. All user options (optimizations,
  # representative dataset, target spec, inference types, custom ops, metadata exclusion,
  # new converter/quantizer switches, resource variables, debug metadata) are attributes
  # inherited from the base classes. Its own convert simply delegates to the superclass.
  """Converts a TensorFlow model into TensorFlow Lite model.

  Attributes:
    optimizations: Experimental flag, subject to change. Set of optimizations to
      apply. e.g {tf.lite.Optimize.DEFAULT}. (default None, must be None or a
      set of values of type `tf.lite.Optimize`)
    representative_dataset: A generator function used for integer quantization
      where each generated sample has the same order, type and shape as the
      inputs to the model. Usually, this is a small subset of a few hundred
      samples randomly chosen, in no particular order, from the training or
      evaluation dataset. This is an optional attribute, but required for full
      integer quantization, i.e, if `tf.int8` is the only supported type in
      `target_spec.supported_types`. Refer to `tf.lite.RepresentativeDataset`.
      (default None)
    target_spec: Experimental flag, subject to change. Specifications of the
      target device, including supported ops set, supported types and a set 
      of user's defined TensorFlow operators required in the TensorFlow Lite 
      runtime. Refer to `tf.lite.TargetSpec`.
    inference_input_type: Data type of the input layer. Note that integer types
      (tf.int8 and tf.uint8) are currently only supported for post-training
      integer quantization and quantization-aware training. (default tf.float32,
      must be in {tf.float32, tf.int8, tf.uint8})
    inference_output_type: Data type of the output layer. Note that integer
      types (tf.int8 and tf.uint8) are currently only supported for 
      post-training integer quantization and quantization-aware training. 
      (default tf.float32, must be in {tf.float32, tf.int8, tf.uint8})
    allow_custom_ops: Boolean indicating whether to allow custom operations.
      When False, any unknown operation is an error. When True, custom ops are
      created for any op that is unknown. The developer needs to provide these
      to the TensorFlow Lite runtime with a custom resolver. (default False)
    exclude_conversion_metadata: Whether not to embed the conversion metadata
      into the converted model. (default False)
    experimental_new_converter: Experimental flag, subject to change. Enables
      MLIR-based conversion. (default True)
    experimental_new_quantizer: Experimental flag, subject to change. Enables
      MLIR-based quantization conversion instead of Flatbuffer-based conversion.
      (default True)
    experimental_enable_resource_variables: Experimental flag, subject to
      change. Enables [resource
      variables](https://tensorflow.org/guide/migrate/tf1_vs_tf2#resourcevariables_instead_of_referencevariables)
      to be converted by this converter. This is only allowed if the
      from_saved_model interface is used. (default True)
    serialize_debug_metadata: Enables serializing debug metadata into the TFLite 
      model. (default False)

  Example usage:

  ```python
  # Converting a SavedModel to a TensorFlow Lite model.
  converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
  tflite_model = converter.convert()

  # Converting a tf.Keras model to a TensorFlow Lite model.
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()

  # Converting ConcreteFunctions to a TensorFlow Lite model.
  converter = tf.lite.TFLiteConverter.from_concrete_functions([func], model)
  tflite_model = converter.convert()

  # Converting a Jax model to a TensorFlow Lite model.
  converter = tf.lite.TFLiteConverter.experimental_from_jax(
      [func], [[ ('input1', input1), ('input2', input2)]])
  tflite_model = converter.convert()
  ```
  """  # fmt: skip

  # pylint: disable=useless-super-delegation
  def __init__(self, funcs, trackable_obj=None):
    # [FUNCTION DOC]
    # Constructor. Parameters: funcs (list of concrete functions without duplicates) and
    # trackable_obj (optional object kept referenced so variables are not collected). Simply
    # forwards both to the frozen graph converter constructor. Returns nothing.
    """Constructor for TFLiteConverter.

    Args:
      funcs: List of TensorFlow ConcreteFunctions. The list should not contain
        duplicate elements.
      trackable_obj: tf.AutoTrackable object associated with `funcs`. A
        reference to this object needs to be maintained so that Variables do not
        get garbage collected since functions have a weak reference to
        Variables. This is only required when the tf.AutoTrackable object is not
        maintained by the user (e.g. `from_saved_model`).
    """
    super(TFLiteConverterV2, self).__init__(funcs, trackable_obj)

  @classmethod
  def from_concrete_functions(cls, funcs, trackable_obj=None):
    # [FUNCTION DOC]
    # Factory class method creating a converter from concrete functions.
    # Parameters: funcs (list of concrete functions; the converter currently handles only a
    # single one), trackable_obj (optional owning object; omitting it logs a deprecation warning
    # and uses the deprecated conversion path). Records the original model type as concrete
    # functions for the metadata. Raises a value error if any item is not a concrete function
    # (with a hint to call the get-concrete-function method when it is an untraced function).
    # Returns a new converter instance of the calling class.
    """Creates a TFLiteConverter object from ConcreteFunctions.

    Args:
      funcs: List of TensorFlow ConcreteFunctions. The list should not contain
        duplicate elements. Currently converter can only convert a single
        ConcreteFunction. Converting multiple functions is under development.
      trackable_obj:   An `AutoTrackable` object (typically `tf.module`)
        associated with `funcs`. A reference to this object needs to be
        maintained so that Variables do not get garbage collected since
        functions have a weak reference to Variables.

    Returns:
      TFLiteConverter object.

    Raises:
      Invalid input type.
    """
    # pylint: disable=protected-access
    TFLiteConverterBase._set_original_model_type(
        conversion_metadata_fb.ModelType.TF_CONCRETE_FUNCTIONS
    )
    # pylint: enable=protected-access
    if trackable_obj is None:
      logging.warning(
          "Please consider providing the trackable_obj argument in the "
          "from_concrete_functions. Providing without the trackable_obj "
          "argument is deprecated and it will use the deprecated conversion "
          "path."
      )
    for func in funcs:
      if not isinstance(func, _function.ConcreteFunction):
        message = "This function takes in a list of ConcreteFunction."
        if isinstance(func, _def_function.Function):
          message += (
              " To get the ConcreteFunction from a Function,"
              " call get_concrete_function."
          )
        raise ValueError(message)
    return cls(funcs, trackable_obj)

  @classmethod
  def from_saved_model(cls, saved_model_dir, signature_keys=None, tags=None):
    # [FUNCTION DOC]
    # Factory class method creating a converter from a SavedModel directory.
    # Parameters: saved_model_dir (path), signature_keys (optional list of signature names;
    # default all signatures of the model), tags (optional set of tags, default the serving
    # tag). Records the model type as saved model. If eager execution is disabled it logs a
    # warning and returns the legacy V1 converter (raising a value error when more than one
    # signature key is given). Otherwise loads the model in eager mode; raises a value error if
    # there are no signatures or an unknown key is requested. For models exported by the Keras
    # export feature (several signatures, a serve endpoint and no default save signature) it
    # renames the serve endpoint to serving default on the loaded object (side effect on that
    # object) and uses that single key. Builds the saved model converter; if it kept a saved
    # model directory, returns it, otherwise returns an instance of the calling class built
    # from the signature functions and the loaded object.
    """Creates a TFLiteConverter object from a SavedModel directory.

    Args:
      saved_model_dir: SavedModel directory to convert.
      signature_keys: List of keys identifying SignatureDef containing inputs
        and outputs. Elements should not be duplicated. By default the
        `signatures` attribute of the MetaGraphdef is used. (default
        saved_model.signatures)
      tags: Set of tags identifying the MetaGraphDef within the SavedModel to
        analyze. All tags in the tag set must be present. (default
        {tf.saved_model.SERVING} or {'serve'})

    Returns:
      TFLiteConverter object.

    Raises:
      Invalid signature keys.
    """
    # pylint: disable=protected-access
    TFLiteConverterBase._set_original_model_type(
        conversion_metadata_fb.ModelType.TF_SAVED_MODEL
    )
    # pylint: enable=protected-access
    # When run without eager enabled, this will return the legacy
    # TFLiteConverter.
    if not context.executing_eagerly():
      signature_key = None
      if signature_keys:
        if len(signature_keys) != 1:
          raise ValueError("Only support a single signature key.")
        else:
          signature_key = signature_keys[0]
      logging.warning(
          "Invoking the TF1 implementation of TFLiteConverter "
          "because eager is disabled. Consider enabling eager."
      )
      return TFLiteConverter.from_saved_model(
          saved_model_dir, signature_key=signature_key, tag_set=tags
      )

    if tags is None:
      tags = set([_tag_constants.SERVING])

    with context.eager_mode():
      saved_model = _load(saved_model_dir, tags)
    if not signature_keys:
      signature_keys = list(saved_model.signatures.keys())

    if not signature_keys:
      raise ValueError("Only support at least one signature key.")

    # Distinguishes SavedModel artifacts created by `model.export`
    # from SavedModel created by `model.save`/`tf.saved_model.save`.
    if (
        len(signature_keys) > 1
        and hasattr(saved_model, "serve")  # `model.export` default endpoint
        and not hasattr(saved_model, "_default_save_signature")
        # `_default_save_signature` does not exist for `model.export` artifacts.
    ):
      # Default `serve` endpoint for `model.export` should be copied
      # to `serving_default` to prevent issues in TF Lite serving.
      saved_model.serving_default = saved_model.serve
      delattr(saved_model, "serve")
      signature_keys = ["serving_default"]

    funcs = []
    for key in signature_keys:
      if key not in saved_model.signatures:
        raise ValueError(
            "Invalid signature key '{}' found. Valid keys are '{}'.".format(
                key, ",".join(saved_model.signatures)
            )
        )
      funcs.append(saved_model.signatures[key])

    saved_model_converter = TFLiteSavedModelConverterV2(
        saved_model_dir, tags, signature_keys
    )
    if saved_model_converter.saved_model_dir:
      return saved_model_converter

    return cls(funcs, saved_model)

  @classmethod
  def from_keras_model(cls, model):
    # [FUNCTION DOC]
    # Factory class method creating a converter from a Keras model (parameter model). Records
    # the original model type as Keras and returns a Keras model V2 converter (not an instance
    # of the calling class).
    """Creates a TFLiteConverter object from a Keras model.

    Args:
      model: tf.Keras.Model

    Returns:
      TFLiteConverter object.
    """
    # pylint: disable=protected-access
    TFLiteConverterBase._set_original_model_type(
        conversion_metadata_fb.ModelType.KERAS_MODEL
    )
    # pylint: enable=protected-access
    return TFLiteKerasModelConverterV2(model)

  @classmethod
  @_deprecation.deprecated(
      None,
      "Use `jax2tf.convert` and (`lite.TFLiteConverter.from_saved_model`"
      " or `lite.TFLiteConverter.from_concrete_functions`) instead.",
  )
  def experimental_from_jax(cls, serving_funcs, inputs):
    # [FUNCTION DOC]
    # Deprecated experimental factory class method creating a converter from JAX functions.
    # Parameters: serving_funcs (list of JAX functions with weights applied) and inputs (list of
    # lists of named input placeholder pairs matching the functions). Records the model type as
    # JAX and returns a JAX V2 converter. The deprecation decorator emits a warning when used.
    # Experimental API, subject to changes.
    # TODO(b/197690428): Currently only supports single function.
    """Creates a TFLiteConverter object from a Jax model with its inputs.

    Args:
      serving_funcs: An array of Jax functions with all the weights applied
        already.
      inputs: An array of Jax input placeholders tuples list, e.g.,
        jnp.zeros(INPUT_SHAPE). Each tuple list should correspond with the
        serving function.

    Returns:
      TFLiteConverter object.
    """
    # pylint: disable=protected-access
    TFLiteConverterBase._set_original_model_type(
        conversion_metadata_fb.ModelType.JAX
    )
    # pylint: enable=protected-access
    return TFLiteJaxConverterV2(serving_funcs, inputs)

  # pylint: disable=useless-super-delegation
  def convert(self):
    # [FUNCTION DOC]
    # Public conversion entry point. Takes no arguments and delegates entirely to the frozen
    # graph converter's convert method, returning the converted model bytes. Errors propagate
    # (value errors for missing/multiple functions, unknown shapes or bad quantization options).
    """Converts a TensorFlow GraphDef based on instance variables.

    Returns:
      The converted data in serialized format.

    Raises:
      ValueError:
        No concrete function is specified.
        Multiple concrete functions are specified.
        Input shape is not specified.
        Invalid quantization parameters.
    """
    return super(TFLiteConverterV2, self).convert()


class TFLiteConverterBaseV1(TFLiteConverterBase):
  # [CLASS DOC]
  # Intermediate base class of the legacy TensorFlow 1 converters. Adds V1 options (inference
  # type and input/output types, output format, quantized input stats, default range stats,
  # control dependency and fake quant flags, Graphviz dump and conversion summary settings),
  # the debug info function, and marks API version 1. It intercepts access to two deprecated
  # properties (post training quantize and target ops) and maps them to optimizations and the
  # target spec. Provides V1 input validation, Grappler optimization and the shared convert.
  # Subclasses must set the graph definition and input/output tensors (or arrays) it uses.
  """Converter subclass to share functionality between V1 converters."""

  def __init__(self, experimental_debug_info_func):
    # [FUNCTION DOC]
    # Constructor. Parameter experimental_debug_info_func: function that retrieves graph debug
    # info for a set of nodes of the graph definition (may be nothing). Calls the base
    # constructor, then sets V1 defaults: float32 inference type, no input/output types, the
    # flatbuffer output format, empty quantized input stats, drop control dependency on, other
    # flags off or empty, and API version 1 in the metadata. Returns nothing.
    """Constructor for TFLiteConverter.

    Args:
      experimental_debug_info_func: An experimental function to retrieve the
        graph debug info for a set of nodes from the `graph_def`.
    """
    super(TFLiteConverterBaseV1, self).__init__()
    self.inference_type = _dtypes.float32
    self.inference_input_type = None
    self.inference_output_type = None
    self.output_format = constants.TFLITE
    self.quantized_input_stats = {}
    self.default_ranges_stats = None
    self.drop_control_dependency = True
    self.reorder_across_fake_quant = False
    self.change_concat_input_ranges = False
    self.dump_graphviz_dir = None
    self.dump_graphviz_video = False
    self.conversion_summary_dir = None
    self._debug_info_func = experimental_debug_info_func
    self._metadata.environment.apiVersion = 1

  def __setattr__(self, name, value):
    # [FUNCTION DOC]
    # Intercepts attribute assignment to support two deprecated properties. Assigning the post
    # training quantize property emits a deprecation warning and instead sets optimizations to
    # a list with the default optimization (true value) or an empty list (false value).
    # Assigning the target ops property warns and sets the target spec's supported ops. Both
    # return without storing the named attribute. All other names are stored normally.
    if name == "post_training_quantize":
      warnings.warn(
          "Property %s is deprecated, "
          "please use optimizations=[Optimize.DEFAULT]"
          " instead." % name
      )
      if value:
        self.optimizations = [Optimize.DEFAULT]
      else:
        self.optimizations = []
      return
    if name == "target_ops":
      warnings.warn(
          "Property %s is deprecated, please use "
          "target_spec.supported_ops instead." % name
      )
      self.target_spec.supported_ops = value
      return
    object.__setattr__(self, name, value)

  def __getattribute__(self, name):
    # [FUNCTION DOC]
    # Intercepts attribute reads to support the same two deprecated properties. Reading the post
    # training quantize property warns and returns whether the default optimization is among the
    # optimizations; reading the target ops property warns and returns the target spec's
    # supported ops. All other names use normal attribute lookup.
    if name == "post_training_quantize":
      warnings.warn(
          "Property %s is deprecated, "
          "please use optimizations=[Optimize.DEFAULT]"
          " instead." % name
      )
      return Optimize.DEFAULT in set(self.optimizations)
    if name == "target_ops":
      warnings.warn(
          "Property %s is deprecated, please use "
          "target_spec.supported_ops instead." % name
      )
      return self.target_spec.supported_ops
    return object.__getattribute__(self, name)

  def _validate_quantized_input_stats(self, converter_kwargs, quant_mode):
    # [FUNCTION DOC]
    # HELPER: checks that quantized input stats are supplied when needed.
    # Parameters: converter_kwargs (mapping of converter arguments, read for the inference type,
    # inference input type and quantized input stats) and quant_mode (QuantizationMode). Stats
    # are required when the inference type or input type is int8 or uint8 and the mode is not
    # post-training integer quantization. Raises a value error naming both types if stats are
    # required but empty. Returns nothing.
    """Ensure the `quantized_input_stats` flag is provided if required."""

    quantized_types = frozenset({_dtypes.int8, _dtypes.uint8})

    requires_quantized_input_stats = (
        converter_kwargs["inference_type"] in quantized_types
        or converter_kwargs["inference_input_type"] in quantized_types
    ) and not quant_mode.is_post_training_integer_quantization()

    if (
        requires_quantized_input_stats
        and not converter_kwargs["quantized_input_stats"]
    ):
      raise ValueError(
          "The `quantized_input_stats` flag must be defined when either "
          "`inference_type` flag or `inference_input_type` flag is set to "
          "tf.int8 or tf.uint8. Currently, `inference_type={}` and "
          "`inference_input_type={}`.".format(
              _get_tf_type_name(converter_kwargs["inference_type"]),
              _get_tf_type_name(converter_kwargs["inference_input_type"]),
          )
      )

  @convert_phase(Component.PREPARE_TF_MODEL, SubComponent.VALIDATE_INPUTS)
  def _validate_inputs(self, input_tensors, quantized_input_stats):
    # [FUNCTION DOC]
    # HELPER: V1 input validation phase (tracked conversion phase).
    # Parameters: input_tensors (list of tensors, may be nothing when arrays are used) and
    # quantized_input_stats (mapping from input names to pairs of mean and standard deviation,
    # may be empty). If unknown shapes are not allowed and tensors are valid: raises a value
    # error if a tensor has no shape or an unknown non-leading dimension, and sets an unknown
    # batch dimension to 1 across inputs. If stats are given, collects them in input array
    # order into the converter's stats list (side effect) and raises a value error listing
    # inputs that lack stats; otherwise clears the stats list to nothing.
    """Validate input parameters.

    Args:
      input_tensors: List of input tensors.
      quantized_input_stats: Map of input tensor names to a tuple of floats
        representing the mean and standard deviation of the training data.

    Raises:
      ValueError:
        Input shape is not specified.
        Quantization input stats is required but not provided.
    """

    if not self._is_unknown_shapes_allowed() and self._has_valid_tensors():
      # Checks dimensions in input tensor.
      for tensor in input_tensors:
        shape = tensor.shape
        if not shape:
          raise ValueError(
              "Provide an input shape for input array '{0}'.".format(
                  _get_tensor_name(tensor)
              )
          )
        # Note that shape_list might be empty for scalar shapes.
        shape_list = shape.as_list()
        if None in shape_list[1:]:
          raise ValueError(
              "None is only supported in the 1st dimension. Tensor '{0}' has "
              "invalid shape '{1}'.".format(
                  _get_tensor_name(tensor), shape_list
              )
          )
        elif shape_list and shape_list[0] is None:
          self._set_batch_size(batch_size=1)

    # Get quantization stats. Ensures there is one stat per name if the stats
    # are specified.
    if quantized_input_stats:
      self._quantized_stats = []
      invalid_stats = []
      for name in self.get_input_arrays():
        if name in quantized_input_stats:
          self._quantized_stats.append(quantized_input_stats[name])
        else:
          invalid_stats.append(name)

      if invalid_stats:
        raise ValueError(
            "Quantization input stats are not available for input "
            "tensors '{0}'.".format(",".join(invalid_stats))
        )
    else:
      self._quantized_stats = None

  @convert_phase(Component.PREPARE_TF_MODEL, SubComponent.OPTIMIZE_TF_MODEL)
  def _optimize_tf_model(
      self, graph_def, input_tensors, output_tensors, quant_mode
  ):
    # [FUNCTION DOC]
    # HELPER: V1 Grappler optimization phase (tracked conversion phase).
    # Parameters: graph_def (frozen graph definition), input_tensors and output_tensors (lists),
    # quant_mode (QuantizationMode). Returns the graph unchanged when converting from a saved
    # model directory or when the graph holds training-time quantization ops (to avoid constant
    # folding them). Otherwise disables lowering with switch/merge, runs the function inlining
    # graph optimization and returns the optimized graph; any exception is swallowed and the
    # original graph is returned.
    """Run a Grappler pass to optimize the TensorFlow graph.

    Args:
      graph_def: Frozen GraphDef to be optimized.
      input_tensors: List of input tensors.
      output_tensors: List of output tensors.
      quant_mode: the quantization mode.

    Returns:
      The optimized TensorFlow graph.
    """
    # Disable grappler constant folding if there are training quant ops.
    if self.saved_model_dir or quant_mode.is_quantization_aware_trained_model():
      return graph_def

    try:
      # TODO(b/150163103): Merge `disabling lower using switch merge' calls.
      # Grappler will also try to lower while loop into switch merge
      # representation which is undesired for Ophints, so we simply remove
      # those attributes to prevent Grappler from doing so.
      graph = _convert_to_constants.disable_lower_using_switch_merge(graph_def)
      # Run function inlining optimization to ensure any models generated
      # through the from_frozen_graph path have been inlined.
      optimized_graph = _run_graph_optimizations(
          graph,
          input_tensors,
          output_tensors,
          config=self._grappler_config(["function"]),
      )
      return optimized_graph
    except Exception:  # pylint: disable=broad-except
      return graph_def

  def convert(self):
    # [FUNCTION DOC]
    # Shared V1 conversion pipeline; subclasses set the graph definition and input/output
    # tensors or arrays beforehand. Takes no arguments. Validates inputs and quantized stats,
    # builds the quantization mode, optimizes the graph, extracts debug info, and assembles
    # converter arguments (base args, quantization flags with the V1 inference types, output
    # format, quantized stats, ranges, control dependency, fake quant, Graphviz and summary
    # options). Validates the quantized input stats requirement (value error), logs a warning
    # or info about the new converter, then converts either from tensors (graph definition path)
    # or from named arrays with shapes. Finally applies post-conversion optimizations and returns
    # the converted bytes (a flatbuffer or Graphviz text depending on the output format).
    """Converts a TensorFlow GraphDef based on instance variables.

    Returns:
      The converted data in serialized format, either a TFLite Flatbuffer or
      a Graphviz graph depending on value in `output_format`.

    Raises:
      ValueError:
        Input shape is not specified.
        None value for dimension in input_tensor.
    """
    self._validate_inputs(self._input_tensors, self.quantized_input_stats)

    quant_mode = QuantizationMode(
        self.optimizations,
        self.target_spec,
        self.representative_dataset,
        self._graph_def,
        self._experimental_disable_per_channel,
        self.experimental_new_dynamic_range_quantizer,
        self._experimental_low_bit_qat,
        self._experimental_full_integer_quantization_bias_type,
        self._experimental_variable_quantization,
        self._experimental_strict_qdq,
    )

    optimized_graph = self._optimize_tf_model(
        self._graph_def, self._input_tensors, self._output_tensors, quant_mode
    )

    self._debug_info = _get_debug_info(self._debug_info_func, optimized_graph)

    converter_kwargs = self._get_base_converter_args()
    converter_kwargs.update(
        quant_mode.converter_flags(
            self.inference_type, self.inference_input_type
        )
    )
    converter_kwargs.update({
        "output_format": self.output_format,
        "quantized_input_stats": self._quantized_stats,
        "default_ranges_stats": self.default_ranges_stats,
        "drop_control_dependency": self.drop_control_dependency,
        "reorder_across_fake_quant": self.reorder_across_fake_quant,
        "change_concat_input_ranges": self.change_concat_input_ranges,
        "dump_graphviz_dir": self.dump_graphviz_dir,
        "dump_graphviz_video": self.dump_graphviz_video,
        "conversion_summary_dir": self.conversion_summary_dir,
    })

    self._validate_quantized_input_stats(converter_kwargs, quant_mode)
    if not self.experimental_new_converter:
      logging.warning(
          "Please consider switching to the new converter by setting "
          "experimental_new_converter=True. "
          "The old converter is deprecated."
      )
    else:
      logging.info(
          "Using experimental converter: If you encountered a problem "
          "please file a bug. You can opt-out "
          "by setting experimental_new_converter=False"
      )
    # Converts model.
    if self._has_valid_tensors():
      result = _convert_graphdef(
          input_data=optimized_graph,
          input_tensors=self._input_tensors,
          output_tensors=self._output_tensors,
          **converter_kwargs,
      )
    else:
      result = _convert_graphdef_with_arrays(
          input_data=optimized_graph,
          input_arrays_with_shape=self._input_arrays_with_shape,
          output_arrays=self._output_arrays,
          control_output_arrays=self._control_output_arrays,
          **converter_kwargs,
      )

    return self._optimize_tflite_model(
        result,
        quant_mode,
        _build_conversion_flags(**converter_kwargs).debug_options,
        quant_io=self.experimental_new_quantizer,
    )

  def get_input_arrays(self):
    # [FUNCTION DOC]
    # Returns the list of input names: tensor names when tensors are available, otherwise the
    # names from the stored input arrays with shapes. Takes no arguments.
    """Returns a list of the names of the input tensors.

    Returns:
      List of strings.
    """
    if self._has_valid_tensors():
      return [_get_tensor_name(tensor) for tensor in self._input_tensors]
    else:
      return [name for name, _ in self._input_arrays_with_shape]

  def _has_valid_tensors(self):
    # [FUNCTION DOC]
    # HELPER: tells whether input and output tensors were initialized.
    # KNOWN ISSUE: the docstring says it returns a bool, but the expression returns the output
    # tensors collection itself when inputs are set (truthy or falsy list), not a strict bool.
    # Returns false if input tensors are nothing, else the output tensors value.
    """Checks if the input and output tensors have been initialized.

    Returns:
      Bool.
    """
    return self._input_tensors is not None and self._output_tensors

  def _set_batch_size(self, batch_size):
    # [FUNCTION DOC]
    # HELPER: fixes an undefined batch dimension. Parameter batch_size: integer to use.
    # Raises a value error if no valid tensors exist. Otherwise, for each input tensor whose
    # first dimension is unknown, sets that dimension to the batch size (mutates tensor shapes).
    """Sets the first dimension of the input tensor to `batch_size`.

    Args:
      batch_size: Batch size for the model. Replaces the first dimension of an
        input size array if undefined. (default 1)

    Raises:
      ValueError: input_tensor is not defined.
    """
    if not self._has_valid_tensors():
      raise ValueError(
          "The batch size cannot be set for this model. Please "
          "use input_shapes parameter."
      )

    for tensor in self._input_tensors:
      shape = tensor.shape.as_list()
      if shape[0] is None:
        shape[0] = batch_size
        tensor.set_shape(shape)

  def _is_unknown_shapes_allowed(self):
    # [FUNCTION DOC]
    # HELPER: V1 override deciding whether unknown shapes are allowed. Returns false if the
    # graph came from op hint conversion, if the base rule disallows it (new converter off), or
    # if a conversion summary directory is set (logs a warning, since that uses the old
    # converter); returns true otherwise.
    # Ophint Converted nodes will need the shapes to be known.
    if _is_ophint_converted(self._graph_def):
      return False

    if not super(TFLiteConverterBaseV1, self)._is_unknown_shapes_allowed():
      return False

    # `conversion_summary_dir` calls the old converter. Unknown shapes are only
    # supported by the MLIR converter.
    if self.conversion_summary_dir:
      logging.warning(
          "`conversion_summary_dir` does not work with unknown shapes. "
          "Graphs with unknown shapes might be different than when this flag "
          "is disabled."
      )
      return False
    return True

  def _save_conversion_params_metric(self):
    # [FUNCTION DOC]
    # HELPER: V1 override that records conversion parameters. Takes no extra arguments (a
    # narrower signature than the base method). Adds the V1 options (output format, ranges,
    # control dependency, fake quant, Graphviz and summary settings) to the collected parameter
    # mapping, then calls the base implementation with the stored graph definition and the
    # inference types. Returns nothing.
    self._collected_converter_params.update({
        "output_format": self.output_format,
        "default_ranges_stats": self.default_ranges_stats,
        "drop_control_dependency": self.drop_control_dependency,
        "reorder_across_fake_quant": self.reorder_across_fake_quant,
        "change_concat_input_ranges": self.change_concat_input_ranges,
        "dump_graphviz_dir": self.dump_graphviz_dir,
        "dump_graphviz_video": self.dump_graphviz_video,
        "conversion_summary_dir": self.conversion_summary_dir,
    })
    super(TFLiteConverterBaseV1, self)._save_conversion_params_metric(
        self._graph_def, self.inference_type, self.inference_input_type
    )


class TFLiteSavedModelConverter(TFLiteConverterBaseV1):
  # [CLASS DOC]
  # Legacy V1 converter whose source is a SavedModel with a single signature key. The
  # constructor freezes the saved model into a graph definition plus input and output tensors;
  # convert is wrapped to export metrics and delegates to the shared V1 pipeline.
  """Converts the given SavedModel into TensorFlow Lite model.

  Attributes:
      saved_model_dir: Directory of the SavedModel.
  """

  def __init__(
      self,
      saved_model_dir,
      saved_model_tags,
      saved_model_exported_names,
      experimental_debug_info_func=None,
  ):
    # [FUNCTION DOC]
    # Constructor. Parameters: saved_model_dir (path), saved_model_tags (set of tags),
    # saved_model_exported_names (list that must contain exactly one signature key),
    # experimental_debug_info_func (optional debug info function). Raises a value error unless
    # exactly one key is given. Freezes the saved model (reads from disk) and stores the
    # resulting graph definition, input tensors and output tensors, then parses saved model
    # args (which may clear the directory). Returns nothing.
    """Constructor for TFLiteConverter.

    Args:
      saved_model_dir: Directory of the SavedModel.
      saved_model_tags: Set of tags identifying the MetaGraphDef within the
        SavedModel to analyze. All tags in the tag set must be present. (default
        {tf.saved_model.SERVING}).
      saved_model_exported_names: Names to be exported when the saved model
        import path is on.
      experimental_debug_info_func: An experimental function to retrieve the
        graph debug info for a set of nodes from the `graph_def`.

    Raises:
      ValueError: Invalid arguments.
    """
    super(TFLiteSavedModelConverter, self).__init__(
        experimental_debug_info_func
    )
    self.saved_model_dir = saved_model_dir
    self._saved_model_tags = saved_model_tags
    self._saved_model_exported_names = saved_model_exported_names

    if len(self._saved_model_exported_names) != 1:
      raise ValueError("Only supports a single signature key.")

    signature_key = self._saved_model_exported_names[0]

    result = _freeze_saved_model(
        self.saved_model_dir,
        None,
        None,
        None,
        self._saved_model_tags,
        signature_key,
    )
    self._graph_def = result[0]
    self._input_tensors = result[1]
    self._output_tensors = result[2]
    self._parse_saved_model_args()

  @_export_metrics
  def convert(self):
    # [FUNCTION DOC]
    # Converts the frozen saved model (wrapped to record metrics). Takes no arguments and simply
    # delegates to the shared V1 pipeline, returning the converted bytes; errors propagate.
    """Converts a TensorFlow GraphDef based on instance variables.

    Note that in the converted TensorFlow Lite model, the input tensor's order
    might be changed each time `convert` is called. To access input tensor
    information, please consider using the `SignatureRunner` API
    (`interpreter.get_signature_runner`).

    Returns:
      The converted data in serialized format, either a TFLite Flatbuffer or
      a Graphviz graph depending on value in `output_format`.

    Raises:
      ValueError:
        Input shape is not specified.
        None value for dimension in input_tensor.
    """
    return super(TFLiteSavedModelConverter, self).convert()


class TFLiteKerasModelConverter(TFLiteConverterBaseV1):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring says it converts a SavedModel; it actually converts a Keras HDF5
  # model file.
  # Legacy V1 converter whose source is a Keras model file. The constructor loads the model,
  # freezes it into a graph definition and tensors, and optionally applies input shapes.
  """Converts the given SavedModel into TensorFlow Lite model."""

  def __init__(
      self,
      model_file,
      input_arrays=None,
      input_shapes=None,
      output_arrays=None,
      custom_objects=None,
  ):
    # [FUNCTION DOC]
    # Constructor. Parameters: model_file (path of an HDF5 Keras model file), input_arrays
    # (optional list of input tensor names; unsupported in eager mode), input_shapes (optional
    # mapping from input names to shape lists; applied to the input tensors), output_arrays
    # (optional list of output tensor names; unsupported in eager mode), custom_objects
    # (optional mapping from names to custom classes/functions for deserialization).
    # In eager mode: raises a value error if input or output arrays are given; loads the model,
    # traces it, freezes variables into constants, applies shapes and stores the model, graph
    # definition, tensors and a debug info function built from the frozen graph. In graph mode:
    # clears the Keras session, loads the model, resolves tensors from names (or the model's own
    # inputs/outputs), applies shapes, freezes the session graph and stores the same state.
    # Side effects: reads the file; clears the global Keras session in graph mode.
    """Constructor for TFLiteConverter.

    Args:
      model_file: Full filepath of HDF5 file containing the tf.keras model.
      input_arrays: List of input tensors to freeze graph with. Uses input
        arrays from SignatureDef when none are provided. (default None)
      input_shapes: Dict of strings representing input tensor names to list of
        integers representing input shapes (e.g., {"foo" : [1, 16, 16, 3]}).
        Automatically determined when input shapes is None (e.g., {"foo" :
        None}). (default None)
      output_arrays: List of output tensors to freeze graph with. Uses output
        arrays from SignatureDef when none are provided. (default None)
      custom_objects: Dict mapping names (strings) to custom classes or
        functions to be considered during model deserialization. (default None)

    Raises:
      ValueError: Invalid arguments.
    """
    super(TFLiteKerasModelConverter, self).__init__(
        experimental_debug_info_func=None
    )
    # Handles Keras when Eager mode is enabled.
    if context.executing_eagerly():
      if input_arrays or output_arrays:
        raise ValueError(
            "`input_arrays` and `output_arrays` are unsupported "
            "with Eager mode. If your model requires any of these "
            "parameters, please use disable_eager_execution()."
        )

      keras_model = keras_deps.get_load_model_function()(
          model_file, custom_objects
      )
      function = _trace_model_call(keras_model)
      concrete_func = function.get_concrete_function()

      frozen_func = _convert_to_constants.convert_variables_to_constants_v2(
          concrete_func, lower_control_flow=False
      )
      _set_tensor_shapes(frozen_func.inputs, input_shapes)
      self._keras_model = keras_model
      self._graph_def = frozen_func.graph.as_graph_def()
      self._input_tensors = frozen_func.inputs
      self._output_tensors = frozen_func.outputs
      self._debug_info_func = _build_debug_info_func(frozen_func.graph)
      return

    # Handles Keras when Eager mode is disabled.
    keras_deps.get_clear_session_function()()
    keras_model = keras_deps.get_load_model_function()(
        model_file, custom_objects
    )
    sess = keras_deps.get_get_session_function()()

    # Get input and output tensors.
    if input_arrays:
      input_tensors = _get_tensors_from_tensor_names(sess.graph, input_arrays)
    else:
      input_tensors = keras_model.inputs

    if output_arrays:
      output_tensors = _get_tensors_from_tensor_names(sess.graph, output_arrays)
    else:
      output_tensors = keras_model.outputs
    _set_tensor_shapes(input_tensors, input_shapes)

    graph_def = _freeze_graph(sess, input_tensors, output_tensors)
    self._keras_model = keras_model
    self._graph_def = graph_def
    self._input_tensors = input_tensors
    self._output_tensors = output_tensors
    self._debug_info_func = _build_debug_info_func(sess.graph)

  @convert_phase(Component.PREPARE_TF_MODEL, SubComponent.FREEZE_KERAS_MODEL)
  def _freeze_keras_model(self, output_dir):
    # [FUNCTION DOC]
    # HELPER: V1 step (tracked conversion phase) that exports the Keras model as a saved model.
    # KNOWN ISSUE: the name suggests freezing a model, but it saves to a saved model directory
    # and returns nothing useful.
    # Parameter output_dir: directory to write to (side effect: files written). Saves the model
    # in the TF format; on any failure returns nothing and leaves the converter unchanged so
    # the original pipeline is used. Otherwise freezes the saved model with the serving tag and
    # default signature, records the saved model dir, tags and signature, parses saved model
    # args (which may clear the directory) and, if still set, replaces the graph definition,
    # tensors and debug info function with those of the saved model. Returns nothing.
    """Save Keras model to Saved Model format.

    Args:
      output_dir: The output directory to save the SavedModel.
    """
    try:
      self._keras_model.save(output_dir, save_format="tf")
    except Exception:  # pylint: disable=broad-except
      # When storing the given keras model to a saved model is failed, let's
      # use original keras model conversion pipeline.
      return None
    tag_set = set([_tag_constants.SERVING])
    signature_key = _signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    graph_def, input_tensors, output_tensors, sess_graph = _freeze_saved_model(
        output_dir, None, None, None, tag_set, signature_key
    )

    self.saved_model_dir = output_dir
    self._saved_model_tags = tag_set
    self._saved_model_exported_names = [signature_key]
    self._parse_saved_model_args()
    if self.saved_model_dir:
      self._graph_def = graph_def
      self._input_tensors = input_tensors
      self._output_tensors = output_tensors
      self._debug_info_func = _build_debug_info_func(sess_graph)

  def _convert_as_saved_model(self):
    # [FUNCTION DOC]
    # HELPER: tries converting through a temporary saved model. Creates a temporary directory
    # (always removed afterwards), exports the Keras model into it and, if that left a saved
    # model directory set, runs the shared V1 conversion. Returns the converted bytes or
    # nothing when the export was not possible.
    """Converts a Keras model as a saved model.

    Returns:
      The converted data in serialized format.
    """
    temp_dir = tempfile.mkdtemp()
    try:
      self._freeze_keras_model(temp_dir)
      if self.saved_model_dir:
        return super(TFLiteKerasModelConverter, self).convert()
    finally:
      shutil.rmtree(temp_dir, True)

  @_export_metrics
  def convert(self):
    # [FUNCTION DOC]
    # Converts the Keras model file (wrapped to record metrics). Takes no arguments. Tries the
    # saved model route first and returns its result when available; otherwise runs the shared
    # V1 conversion on the frozen graph. Returns the converted bytes; value errors for unknown
    # input shapes or missing quantization stats propagate.
    """Converts a Keras model based on instance variables.

    Returns:
      The converted data in serialized format, either a TFLite Flatbuffer or
      a Graphviz graph depending on value in `output_format`.

    Raises:
      ValueError:
        Input shape is not specified.
        None value for dimension in input_tensor.
    """
    saved_model_convert_result = self._convert_as_saved_model()
    if saved_model_convert_result:
      return saved_model_convert_result

    return super(TFLiteKerasModelConverter, self).convert()


class TFLiteFrozenGraphConverter(TFLiteConverterBaseV1):
  # [CLASS DOC]
  # Legacy V1 converter whose source is a frozen graph definition, given either with tensors
  # that are loadable in TensorFlow or, for graphs with ops TensorFlow cannot resolve, with
  # named input arrays with shapes and output array names. Holds the graph definition, tensors,
  # arrays and control output arrays. Base class of the public V1 converter.
  """Converts the given frozen graph def into TensorFlow Lite model."""

  def __init__(
      self,
      graph_def,
      input_tensors,
      output_tensors,
      input_arrays_with_shape=None,
      output_arrays=None,
      experimental_debug_info_func=None,
  ):
    # [FUNCTION DOC]
    # Constructor. Parameters: graph_def (frozen graph definition), input_tensors and
    # output_tensors (lists, or nothing for graphs TensorFlow cannot load),
    # input_arrays_with_shape (optional list of pairs of name and shape list),
    # output_arrays (optional list of names), experimental_debug_info_func (optional).
    # Stores the graph and tensors, clears control output arrays, and keeps the arrays only
    # when tensors are not valid. Logs a warning when arrays are given alongside tensors (the
    # arrays are then ignored). Returns nothing.
    """Constructor for TFLiteConverter.

    Args:
      graph_def: Frozen TensorFlow GraphDef.
      input_tensors: List of input tensors. Type and shape are computed using
        `foo.shape` and `foo.dtype`.
      output_tensors: List of output tensors (only .name is used from this).
      input_arrays_with_shape: Tuple of strings representing input tensor names
        and list of integers representing input shapes (e.g., [("foo", [1, 16,
        16, 3])]). Use only when graph cannot be loaded into TensorFlow and when
        `input_tensors` and `output_tensors` are None. (default None)
      output_arrays: List of output tensors to freeze graph with. Use only when
        graph cannot be loaded into TensorFlow and when `input_tensors` and
        `output_tensors` are None. (default None)
      experimental_debug_info_func: An experimental function to retrieve the
        graph debug info for a set of nodes from the `graph_def`.

    Raises:
      ValueError: Invalid arguments.
    """
    super(TFLiteFrozenGraphConverter, self).__init__(
        experimental_debug_info_func
    )
    self._graph_def = graph_def
    self._input_tensors = input_tensors
    self._output_tensors = output_tensors
    self._control_output_arrays = None

    # Attributes are used by models that cannot be loaded into TensorFlow.
    if not self._has_valid_tensors():
      self._input_arrays_with_shape = input_arrays_with_shape
      self._output_arrays = output_arrays

    if input_tensors is not None and input_arrays_with_shape is not None:
      logging.warning(
          "input_arrays_with_shape will be ignored when both the "
          "given input_tensors and input_arrays_with_shape are not "
          "None."
      )

    if output_tensors is not None and output_arrays is not None:
      logging.warning(
          "output_arrays will be ignored when both the given "
          "output_tensors and output_arrays are not None."
      )

  @_export_metrics
  def convert(self):
    # [FUNCTION DOC]
    # Converts the frozen graph (wrapped to record metrics). Takes no arguments. If tensors are
    # not valid, raises a value error unless both input arrays with shapes and
    # output/control-output arrays are defined. Then delegates to the shared V1 pipeline and
    # returns the converted bytes.
    """Converts a TensorFlow GraphDef based on instance variables.

    Returns:
      The converted data in serialized format, either a TFLite Flatbuffer or
      a Graphviz graph depending on value in `output_format`.

    Raises:
      ValueError:
        Input shape is not specified.
        None value for dimension in input_tensor.
    """
    if not self._has_valid_tensors():
      if not self._input_arrays_with_shape or not (
          self._output_arrays or self._control_output_arrays
      ):
        raise ValueError(
            "If input_tensors and output_tensors are None, both "
            "input_arrays_with_shape and output_arrays|control_output_arrays "
            "must be defined."
        )
    return super(TFLiteFrozenGraphConverter, self).convert()


@_tf_export(v1=["lite.TFLiteConverter"])
class TFLiteConverter(TFLiteFrozenGraphConverter):
  # [CLASS DOC]
  # Public legacy (TensorFlow 1 style) converter, exported only under the v1 lite namespace.
  # Extends the frozen graph converter and is normally created via factory class methods from
  # a session, a frozen graph file, a SavedModel or a Keras model file; some factories return
  # specialized V1 converter subclasses instead of this class. All V1 options (inference
  # types, quantized input stats, output format, Graphviz dumps, etc.) are inherited
  # attributes. Its convert simply delegates to the superclass.
  """Convert a TensorFlow model into `output_format`.

  This is used to convert from a TensorFlow GraphDef, SavedModel or tf.keras
  model into either a TFLite FlatBuffer or graph visualization.

  Attributes:
    optimizations: Experimental flag, subject to change. Set of optimizations to
      apply. e.g {tf.lite.Optimize.DEFAULT}. (default None, must be None or a
      set of values of type `tf.lite.Optimize`)
    representative_dataset: A generator function used for integer quantization
      where each generated sample has the same order, type and shape as the
      inputs to the model. Usually, this is a small subset of a few hundred
      samples randomly chosen, in no particular order, from the training or
      evaluation dataset. This is an optional attribute, but required for full
      integer quantization, i.e, if `tf.int8` is the only supported type in
      `target_spec.supported_types`. Refer to `tf.lite.RepresentativeDataset`.
      (default None)
    target_spec: Experimental flag, subject to change. Specifications of target
      device, including supported ops set, supported types and a set of user's
      defined TensorFlow operators required in the TensorFlow Lite runtime.
      Refer to `tf.lite.TargetSpec`.
    inference_type: Data type of numeric arrays, excluding the input layer.
      (default tf.float32, must be in {tf.float32, tf.int8, tf.uint8})
    inference_input_type: Data type of the numeric arrays in the input layer. If
      `inference_input_type` is in {tf.int8, tf.uint8}, then
      `quantized_input_stats` must be provided. (default is the value assigned
      to `inference_type`, must be in {tf.float32, tf.int8, tf.uint8})
    inference_output_type: Data type of the numeric arrays in the output layer.
      (default is the value assigned to `inference_type`, must be in
      {tf.float32, tf.int8, tf.uint8})
    quantized_input_stats: Map of input tensor names to a tuple of floats
      representing the mean and standard deviation of the training data. (e.g.,
      {"foo" : (0., 1.)}). Required if `inference_input_type` is tf.int8 or
      tf.uint8. (default None)
    default_ranges_stats: Tuple of integers (min, max) representing range values
      for all numeric arrays without a specified range. Intended for
      experimenting with quantization via "dummy quantization". (default None)
    allow_custom_ops: Boolean indicating whether to allow custom operations.
      When False any unknown operation is an error. When True, custom ops are
      created for any op that is unknown. The developer will need to provide
      these to the TensorFlow Lite runtime with a custom resolver. (default
      False)
    drop_control_dependency: Boolean indicating whether to drop control
      dependencies silently. This is due to TFLite not supporting control
      dependencies. (default True)
    reorder_across_fake_quant: Boolean indicating whether to reorder FakeQuant
      nodes in unexpected locations. Used when the location of the FakeQuant
      nodes is preventing graph transformations necessary to convert the graph.
      Results in a graph that differs from the quantized training graph,
      potentially causing differing arithmetic behavior. (default False)
    change_concat_input_ranges: Boolean to change behavior of min/max ranges for
      inputs and outputs of the concat operator for quantized models. Changes
      the ranges of concat operator overlap when true. (default False)
    output_format: Output file format. (default
      tf.compat.v1.lite.constants.TFLITE, must be in
      {tf.compat.v1.lite.constants.TFLITE,
      tf.compat.v1.lite.constants.GRAPHVIZ_DOT})
    dump_graphviz_dir: Full filepath of folder to dump the graphs at various
      stages of processing GraphViz .dot files. Preferred over
      `output_format=tf.compat.v1.lite.constants.GRAPHVIZ_DOT` in order to keep
      the requirements of the output file. (default None)
    dump_graphviz_video: Boolean indicating whether to dump the GraphViz .dot
      files after every graph transformation. Requires the `dump_graphviz_dir`
      flag to be specified. (default False)
    conversion_summary_dir: Full path of the directory to store conversion logs.
      (default None)
    exclude_conversion_metadata: Whether not to embed the conversion metadata
      into the converted model. (default False)
    target_ops: Deprecated. Please use `target_spec.supported_ops` instead.
    post_training_quantize: Deprecated. Please use `optimizations` instead and
      set it to `{tf.lite.Optimize.DEFAULT}`. (default False)
    experimental_new_converter: Experimental flag, subject to change. Enables
      MLIR-based conversion. (default True)
    experimental_new_quantizer: Experimental flag, subject to change. Enables
      MLIR-based quantization conversion instead of Flatbuffer-based conversion.
      (default True)  Example usage:  ```python # Converting a GraphDef from
      session. converter = tf.compat.v1.lite.TFLiteConverter.from_session( sess,
      in_tensors, out_tensors) tflite_model = converter.convert()
      open("converted_model.tflite", "wb").write(tflite_model)  # Converting a
      GraphDef from file. converter =
      tf.compat.v1.lite.TFLiteConverter.from_frozen_graph( graph_def_file,
      input_arrays, output_arrays) tflite_model = converter.convert()
      open("converted_model.tflite", "wb").write(tflite_model)  # Converting a
      SavedModel. converter =
      tf.compat.v1.lite.TFLiteConverter.from_saved_model( saved_model_dir)
      tflite_model = converter.convert() open("converted_model.tflite",
      "wb").write(tflite_model)  # Converting a tf.keras model. converter =
      tf.compat.v1.lite.TFLiteConverter.from_keras_model_file( keras_model)
      tflite_model = converter.convert() open("converted_model.tflite",
      "wb").write(tflite_model) ```
  """

  # pylint: disable=useless-super-delegation
  def __init__(
      self,
      graph_def,
      input_tensors,
      output_tensors,
      input_arrays_with_shape=None,
      output_arrays=None,
      experimental_debug_info_func=None,
  ):
    # [FUNCTION DOC]
    # Constructor. Parameters match the frozen graph converter (graph definition, input and
    # output tensors, optional input arrays with shapes, optional output arrays, optional debug
    # info function) and are forwarded to it unchanged. Returns nothing.
    """Constructor for TFLiteConverter.

    Args:
      graph_def: Frozen TensorFlow GraphDef.
      input_tensors: List of input tensors. Type and shape are computed using
        `foo.shape` and `foo.dtype`.
      output_tensors: List of output tensors (only .name is used from this).
      input_arrays_with_shape: Tuple of strings representing input tensor names
        and list of integers representing input shapes (e.g., [("foo" : [1, 16,
        16, 3])]). Use only when graph cannot be loaded into TensorFlow and when
        `input_tensors` and `output_tensors` are None. (default None)
      output_arrays: List of output tensors to freeze graph with. Use only when
        graph cannot be loaded into TensorFlow and when `input_tensors` and
        `output_tensors` are None. (default None)
      experimental_debug_info_func: An experimental function to retrieve the
        graph debug info for a set of nodes from the `graph_def`.

    Raises:
      ValueError: Invalid arguments.
    """
    super(TFLiteConverter, self).__init__(
        graph_def,
        input_tensors,
        output_tensors,
        input_arrays_with_shape,
        output_arrays,
        experimental_debug_info_func,
    )

  @classmethod
  def from_session(cls, sess, input_tensors, output_tensors):
    # [FUNCTION DOC]
    # Factory class method creating a converter from a TensorFlow session. Parameters: sess
    # (session), input_tensors and output_tensors (lists of tensors). Records the model type as
    # session, freezes the session's graph into a graph definition, and returns an instance of
    # the calling class with a debug info function built from the session graph.
    """Creates a TFLiteConverter class from a TensorFlow Session.

    Args:
      sess: TensorFlow Session.
      input_tensors: List of input tensors. Type and shape are computed using
        `foo.shape` and `foo.dtype`.
      output_tensors: List of output tensors (only .name is used from this).

    Returns:
      TFLiteConverter class.
    """
    # pylint: disable=protected-access
    TFLiteConverterBase._set_original_model_type(
        conversion_metadata_fb.ModelType.TF_SESSION
    )
    # pylint: enable=protected-access
    graph_def = _freeze_graph(sess, input_tensors, output_tensors)
    return cls(
        graph_def,
        input_tensors,
        output_tensors,
        experimental_debug_info_func=_build_debug_info_func(sess.graph),
    )

  @classmethod
  def from_frozen_graph(
      cls, graph_def_file, input_arrays, output_arrays, input_shapes=None
  ):
    # [FUNCTION DOC]
    # Factory class method creating a converter from a frozen graph definition file.
    # Parameters: graph_def_file (path), input_arrays and output_arrays (lists of tensor names),
    # input_shapes (optional mapping from input names to shape lists). Records the model type as
    # graph definition. Inside a fresh graph and session: raises an IO error if the file is
    # missing; parses it as binary, falling back to text format (prints a notice about large
    # allocation warnings), raising an IO error when neither works; byte-swaps tensor content on
    # big-endian hosts. Tries importing the graph; if an op is unknown to TensorFlow, the model
    # is handled as an array-only graph, which needs input_shapes with exactly the input array
    # names (value errors otherwise) and returns an instance built from arrays. If importable:
    # raises a value error when the graph is not frozen, resolves tensors by name (errors for
    # invalid names), applies shapes, and returns an instance built from the session graph.
    """Creates a TFLiteConverter class from a file containing a frozen GraphDef.

    Args:
      graph_def_file: Full filepath of file containing frozen GraphDef.
      input_arrays: List of input tensors to freeze graph with.
      output_arrays: List of output tensors to freeze graph with.
      input_shapes: Dict of strings representing input tensor names to list of
        integers representing input shapes (e.g., {"foo" : [1, 16, 16, 3]}).
        Automatically determined when input shapes is None (e.g., {"foo" :
        None}). (default None)

    Returns:
      TFLiteConverter class.

    Raises:
      IOError:
        File not found.
        Unable to parse input file.
      ValueError:
        The graph is not frozen.
        input_arrays or output_arrays contains an invalid tensor name.
        input_shapes is not correctly defined when required
    """
    # pylint: disable=protected-access
    TFLiteConverterBase._set_original_model_type(
        conversion_metadata_fb.ModelType.TF_GRAPH_DEF
    )
    # pylint: enable=protected-access
    with _ops.Graph().as_default():
      with _session.Session() as sess:
        # Read GraphDef from file.
        if not gfile.Exists(graph_def_file):
          raise IOError("File '{0}' does not exist.".format(graph_def_file))
        with gfile.GFile(graph_def_file, "rb") as f:
          file_content = f.read()

        try:
          graph_def = _graph_pb2.GraphDef()
          graph_def.ParseFromString(file_content)
        except (_text_format.ParseError, DecodeError):
          try:
            print("Ignore 'tcmalloc: large alloc' warnings.")

            if not isinstance(file_content, str):
              file_content = file_content.decode("utf-8")
            graph_def = _graph_pb2.GraphDef()
            _text_format.Merge(file_content, graph_def)
          except (_text_format.ParseError, DecodeError):
            raise IOError(
                "Unable to parse input file '{}'.".format(graph_def_file)
            )

        if sys.byteorder == "big":
          bst.swap_tensor_content_in_graph_node(graph_def, "little", "big")

        # Handles models with custom TFLite ops that cannot be resolved in
        # TensorFlow.
        load_model_in_session = True
        try:
          _import_graph_def(graph_def, name="")
        except _NotFoundError:
          load_model_in_session = False

        if load_model_in_session:
          # Check if graph is frozen.
          if not _is_frozen_graph(sess):
            raise ValueError("Please freeze the graph using freeze_graph.py.")

          # Get input and output tensors.
          input_tensors = _get_tensors_from_tensor_names(
              sess.graph, input_arrays
          )
          output_tensors = _get_tensors_from_tensor_names(
              sess.graph, output_arrays
          )
          _set_tensor_shapes(input_tensors, input_shapes)

          return cls(sess.graph_def, input_tensors, output_tensors)
        else:
          if not input_shapes:
            raise ValueError("input_shapes must be defined for this model.")
          if set(input_arrays) != set(input_shapes.keys()):
            raise ValueError(
                "input_shapes must contain a value for each item "
                "in input_array."
            )

          input_arrays_with_shape = [
              (name, input_shapes[name]) for name in input_arrays
          ]
          return cls(
              graph_def,
              input_tensors=None,
              output_tensors=None,
              input_arrays_with_shape=input_arrays_with_shape,
              output_arrays=output_arrays,
          )

  @classmethod
  def from_saved_model(
      cls,
      saved_model_dir,
      input_arrays=None,
      input_shapes=None,
      output_arrays=None,
      tag_set=None,
      signature_key=None,
  ):
    # [FUNCTION DOC]
    # Factory class method creating a V1 converter from a SavedModel.
    # Parameters: saved_model_dir (path), input_arrays and output_arrays (optional name lists,
    # defaulting to the signature's), input_shapes (optional mapping from names to shapes),
    # tag_set (optional, default the serving tag), signature_key (optional, default serving
    # default). Records the model type as saved model. First builds the saved model converter;
    # if it keeps a saved model directory, returns it (note the array/shape arguments are then
    # not used). Otherwise freezes the saved model with the given arrays and shapes and returns
    # an instance of the calling class with its graph definition, tensors and debug info.
    """Creates a TFLiteConverter class from a SavedModel.

    Args:
      saved_model_dir: SavedModel directory to convert.
      input_arrays: List of input tensors to freeze graph with. Uses input
        arrays from SignatureDef when none are provided. (default None)
      input_shapes: Dict of strings representing input tensor names to list of
        integers representing input shapes (e.g., {"foo" : [1, 16, 16, 3]}).
        Automatically determined when input shapes is None (e.g., {"foo" :
        None}). (default None)
      output_arrays: List of output tensors to freeze graph with. Uses output
        arrays from SignatureDef when none are provided. (default None)
      tag_set: Set of tags identifying the MetaGraphDef within the SavedModel to
        analyze. All tags in the tag set must be present. (default
        {tf.saved_model.SERVING})
      signature_key: Key identifying SignatureDef containing inputs and outputs.
        (default tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY)

    Returns:
      TFLiteConverter class.
    """
    # pylint: disable=protected-access
    TFLiteConverterBase._set_original_model_type(
        conversion_metadata_fb.ModelType.TF_SAVED_MODEL
    )
    # pylint: enable=protected-access
    if tag_set is None:
      tag_set = set([_tag_constants.SERVING])
    if signature_key is None:
      signature_key = _signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    saved_model_converter = TFLiteSavedModelConverter(
        saved_model_dir, tag_set, [signature_key]
    )
    if saved_model_converter.saved_model_dir:
      return saved_model_converter

    result = _freeze_saved_model(
        saved_model_dir,
        input_arrays,
        input_shapes,
        output_arrays,
        tag_set,
        signature_key,
    )

    return cls(
        graph_def=result[0],
        input_tensors=result[1],
        output_tensors=result[2],
        experimental_debug_info_func=_build_debug_info_func(result[3]),
    )

  @classmethod
  def from_keras_model_file(
      cls,
      model_file,
      input_arrays=None,
      input_shapes=None,
      output_arrays=None,
      custom_objects=None,
  ):
    # [FUNCTION DOC]
    # Factory class method creating a converter from a Keras HDF5 model file. Parameters:
    # model_file (path), input_arrays and output_arrays (optional name lists), input_shapes
    # (optional mapping), custom_objects (optional mapping for deserialization). Records the
    # model type as Keras and returns a Keras model V1 converter (not an instance of the
    # calling class); the constructor may raise value errors for invalid arguments.
    """Creates a TFLiteConverter class from a tf.keras model file.

    Args:
      model_file: Full filepath of HDF5 file containing the tf.keras model.
      input_arrays: List of input tensors to freeze graph with. Uses input
        arrays from SignatureDef when none are provided. (default None)
      input_shapes: Dict of strings representing input tensor names to list of
        integers representing input shapes (e.g., {"foo" : [1, 16, 16, 3]}).
        Automatically determined when input shapes is None (e.g., {"foo" :
        None}). (default None)
      output_arrays: List of output tensors to freeze graph with. Uses output
        arrays from SignatureDef when none are provided. (default None)
      custom_objects: Dict mapping names (strings) to custom classes or
        functions to be considered during model deserialization. (default None)

    Returns:
      TFLiteConverter class.
    """
    # pylint: disable=protected-access
    TFLiteConverterBase._set_original_model_type(
        conversion_metadata_fb.ModelType.KERAS_MODEL
    )
    # pylint: enable=protected-access
    return TFLiteKerasModelConverter(
        model_file, input_arrays, input_shapes, output_arrays, custom_objects
    )

  # pylint: disable=useless-super-delegation
  def convert(self):
    # [FUNCTION DOC]
    # Public V1 conversion entry point. Takes no arguments and delegates to the frozen graph
    # converter's convert method, returning the converted bytes (a flatbuffer or Graphviz text
    # according to the output format). Errors propagate.
    """Converts a TensorFlow GraphDef based on instance variables.

    Returns:
      The converted data in serialized format, either a TFLite Flatbuffer or
      a Graphviz graph depending on value in `output_format`.

    Raises:
      ValueError:
        Input shape is not specified.
        None value for dimension in input_tensor.
    """
    return super(TFLiteConverter, self).convert()


@_tf_export(v1=["lite.TocoConverter"])
class TocoConverter:
  # [CLASS DOC]
  # Deprecated compatibility class (exported only under the v1 namespace) that only offers
  # class methods forwarding to the equivalent factory methods of the V1 converter class. Each
  # forwarder is wrapped by a deprecation decorator that logs a warning when used. It is never
  # instantiated and holds no state.
  """Convert a TensorFlow model into `output_format`.

  This class has been deprecated. Please use `lite.TFLiteConverter` instead.
  """

  @classmethod
  @_deprecation.deprecated(
      None, "Use `lite.TFLiteConverter.from_session` instead."
  )
  def from_session(cls, sess, input_tensors, output_tensors):
    # [FUNCTION DOC]
    # Deprecated forwarder to the V1 converter's from-session factory with the same arguments
    # (session, input tensors, output tensors). Returns the created converter.
    """Creates a TocoConverter class from a TensorFlow Session."""
    return TFLiteConverter.from_session(sess, input_tensors, output_tensors)

  @classmethod
  @_deprecation.deprecated(
      None, "Use `lite.TFLiteConverter.from_frozen_graph` instead."
  )
  def from_frozen_graph(
      cls, graph_def_file, input_arrays, output_arrays, input_shapes=None
  ):
    # [FUNCTION DOC]
    # Deprecated forwarder to the V1 converter's from-frozen-graph factory with the same
    # arguments (file path, input array names, output array names, optional input shapes).
    # Returns the created converter.
    """Creates a TocoConverter class from a file containing a frozen graph."""
    return TFLiteConverter.from_frozen_graph(
        graph_def_file, input_arrays, output_arrays, input_shapes
    )

  @classmethod
  @_deprecation.deprecated(
      None, "Use `lite.TFLiteConverter.from_saved_model` instead."
  )
  def from_saved_model(
      cls,
      saved_model_dir,
      input_arrays=None,
      input_shapes=None,
      output_arrays=None,
      tag_set=None,
      signature_key=None,
  ):
    # [FUNCTION DOC]
    # Deprecated forwarder to the V1 converter's from-saved-model factory with the same
    # arguments (directory, optional arrays, shapes, tag set and signature key). Returns the
    # created converter.
    """Creates a TocoConverter class from a SavedModel."""
    return TFLiteConverter.from_saved_model(
        saved_model_dir,
        input_arrays,
        input_shapes,
        output_arrays,
        tag_set,
        signature_key,
    )

  @classmethod
  @_deprecation.deprecated(
      None, "Use `lite.TFLiteConverter.from_keras_model_file` instead."
  )
  def from_keras_model_file(
      cls, model_file, input_arrays=None, input_shapes=None, output_arrays=None
  ):
    # [FUNCTION DOC]
    # Deprecated forwarder to the V1 converter's from-Keras-model-file factory.
    # KNOWN ISSUE: unlike the target factory it does not accept or forward a custom objects
    # argument, so custom Keras classes cannot be supplied through this class.
    # Parameters: model file path, optional input arrays, input shapes and output arrays.
    # Returns the created converter.
    """Creates a TocoConverter class from a tf.keras model file."""
    return TFLiteConverter.from_keras_model_file(
        model_file, input_arrays, input_shapes, output_arrays
    )
