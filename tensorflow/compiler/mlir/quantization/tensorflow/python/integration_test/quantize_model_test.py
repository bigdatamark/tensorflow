# [MODULE DOC]
# KNOWN ISSUE: the per-channel op name constant is a generator expression, not a tuple or list, so it
# can be iterated only once per process; later uses see an empty sequence.
# Integration tests for the TensorFlow quantizer entry point (the quantize function of the
# quantize_model module) in the TF-quantization MLIR pipeline.
# The module builds small saved models (Keras-free tf.Module models and TF1 graph-mode saved models),
# runs post-training static range (PTQ), quantization-aware (QAT, models with fake-quant ops), dynamic
# range and weight-only quantization to several target op sets (TF, XLA, uniform quantized, StableHLO),
# then inspects the output graph definition, signature keys, model size ratio and numerical fidelity.
# Test classes: QuantizationOptionsTest, TensorNamePreservationTest, StaticRangeQuantizationTest,
# DynamicRangeQuantizationTest, WeightOnlyQuantizationTest, DebuggerTest, CalibrationOptionsTest and
# SelectiveQuantizationTest. All derive from the QuantizedModelTest base class in
# quantize_model_test_base, which supplies temporary input/output saved-model directories and graph
# inspection helpers (contains-op, count-ops, size ratio, model builders).
# Module-level helpers: type aliases for protobuf enums, the per-channel op name constants, a variable
# node detector and a variable finder for TF1 graph definitions, and the MultipleSignatureModel fixture.
# Import-time effects: imports tensorflow and defines module constants only; no registrations.
# Running the file as a program starts the TensorFlow test runner.
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
"""Tests for quantize_model."""
# TODO(b/264234648): Refactor and cleanup this file.
import itertools
import os
from typing import Mapping, Optional, Sequence, Tuple, Union

from absl.testing import parameterized
import numpy as np
import tensorflow  # pylint: disable=unused-import

from tensorflow.compiler.mlir.quantization.common.python import testing
from tensorflow.compiler.mlir.quantization.stablehlo import quantization_config_pb2 as stablehlo_quant_config_pb2
from tensorflow.compiler.mlir.quantization.tensorflow import quantization_options_pb2 as quant_opts_pb2
from tensorflow.compiler.mlir.quantization.tensorflow.python import quantize_model
from tensorflow.compiler.mlir.quantization.tensorflow.python import representative_dataset as repr_dataset
from tensorflow.compiler.mlir.quantization.tensorflow.python import save_model
from tensorflow.compiler.mlir.quantization.tensorflow.python.integration_test import quantize_model_test_base
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import node_def_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.lib.io import tf_record
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import builder
from tensorflow.python.saved_model import loader_impl as saved_model_loader
from tensorflow.python.saved_model import save as saved_model_save
from tensorflow.python.saved_model import save_options
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils_impl
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.types import core


# Type aliases for quantization method protobuf enums.
_PresetMethod = quant_opts_pb2.QuantizationMethod.PresetMethod
_CalibrationMethod = (
    stablehlo_quant_config_pb2.CalibrationOptions.CalibrationMethod
)

_QuantizationComponent = (
    quant_opts_pb2.QuantizationComponentSpec.QuantizationComponent
)
_TensorType = quant_opts_pb2.QuantizationComponentSpec.TensorType

_TensorShape = Sequence[Union[int, None]]

_PER_CHANNEL_QUANTIZED_OPS = (
    'UniformQuantizedConvolution',
    'UniformQuantizedConvolutionHybrid',
    'UniformQuantizedDotHybrid',
)

_DebuggerConfig = stablehlo_quant_config_pb2.DebuggerConfig

# Lists of ops whose channel dimension should be changed if per_channel
# quantization is enabled. Respectively refers to (scale, zero_point).
_SUFFIXES = ('/filter1', '/filter2')
_PER_CHANNEL_OP_NAMES = (
    f'{op}{suffix}'
    for op, suffix in itertools.product(_PER_CHANNEL_QUANTIZED_OPS, _SUFFIXES)
)


def _is_variable(node_def: node_def_pb2.NodeDef) -> bool:
  # [FUNCTION DOC]
  # HELPER: predicate used by the variable finder of this test module.
  # Returns true if the given node definition (graph node protobuf) is a resource variable handle node,
  # that is, its op type equals the variable-handle op. Takes one node definition, returns a boolean,
  # no side effects.
  """Determines whether `node_def` is a variable node.

  Args:
    node_def: `NodeDef` to test whether it is a variable or not.

  Returns:
    Returns True if it is a variable.
  """
  return node_def.op == 'VarHandleOp'


def _find_variables(
    graph_def: graph_pb2.GraphDef,
) -> Mapping[str, node_def_pb2.NodeDef]:
  # [FUNCTION DOC]
  # HELPER: used by the unfreeze-constants tests to compare graph variables against the checkpoint.
  # KNOWN ISSUE: the docstring says it maps each variable's shared_name, but the second loop (over
  # library functions) never reads the function node's own shared_name; it reuses the shared_name left
  # over from the last top-level variable, so function-level variables overwrite a single stale key (and
  # it raises an unbound-name error if there was no top-level variable but a function has one).
  # Collects variable handle nodes of a TF1-style graph definition into a mapping from shared name to
  # node definition. First it scans the top-level nodes and records each variable node under its
  # shared_name attribute (decoded as text). Then it scans every function in the graph library.
  # Input: graph_def, the graph definition to search. Output: the mapping. No side effects.
  """Finds all variables within `graph_def`.

  This function makes sense for TF 1 graphs only, as it depends on
  `shared_name`.

  Args:
    graph_def: `GraphDef` to find variables from.

  Returns:
    A mapping of `shared_name` -> `NodeDef` corresponding to a variable op.
  """
  variable_nodes = {}

  for var_node in filter(_is_variable, graph_def.node):
    shared_name = str(var_node.attr['shared_name'].s, encoding='utf-8')
    variable_nodes[shared_name] = var_node

  for func in graph_def.library.function:
    for var_node in filter(_is_variable, func.node_def):
      variable_nodes[shared_name] = var_node

  return variable_nodes


class MultipleSignatureModel(module.Module):
  # [CLASS DOC]
  # Test fixture: a saved-model module with two traced functions (matmul and conv), used where the
  # quantizer must handle several signatures. Holds a random matmul weight tensor (4 by 3) and a
  # random float32 convolution filter (shape 2,3,3,2) as attributes. Instances are created with no
  # arguments and are saved with explicit signature keys by the tests.
  """A model with 2 signatures.

  Used to test where the quantizer has to handle multiple signatures.
  """

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor. Takes no arguments. Creates the matmul filter as a random uniform tensor in the range
    # minus one to one with shape 4 by 3 and the convolution filter as a numpy random array in the range
    # minus ten to ten with shape 2,3,3,2 (float32). Returns nothing.
    self.matmul_filters = random_ops.random_uniform(
        shape=(4, 3), minval=-1.0, maxval=1.0
    )
    self.conv_filters = np.random.uniform(
        low=-10, high=10, size=(2, 3, 3, 2)
    ).astype('f4')

  @def_function.function(
      input_signature=[
          tensor_spec.TensorSpec(shape=[1, 4], dtype=dtypes.float32)
      ]
  )
  def matmul(self, matmul_input: core.Tensor) -> Mapping[str, core.Tensor]:
    # [FUNCTION DOC]
    # Traced function (fixed input spec: float32 tensor of shape 1 by 4). Multiplies the input by the
    # stored matmul filter and returns a mapping with one key, output, holding the product.
    # Input: matmul_input, the left operand tensor. No side effects.
    """Performs a matrix multiplication.

    Args:
      matmul_input: Input tensor to matmul with the filter.

    Returns:
      A map of: output key -> output result.
    """
    out = math_ops.matmul(matmul_input, self.matmul_filters)

    return {'output': out}

  @def_function.function(
      input_signature=[
          tensor_spec.TensorSpec(shape=(1, 3, 4, 3), dtype=dtypes.float32)
      ]
  )
  def conv(self, conv_input: core.Tensor) -> Mapping[str, core.Tensor]:
    # [FUNCTION DOC]
    # Traced function (fixed input spec: float32 tensor of shape 1,3,4,3). Applies a 2D convolution with
    # the stored filter, strides 1,1,2,1, no dilation, SAME padding and NHWC layout. Returns a mapping
    # with one key, output, holding the convolution result. Input: conv_input, the image tensor.
    """Performs a 2D convolution operation.

    Args:
      conv_input: Input tensor to perform convolution on.

    Returns:
      A map of: output key -> output result.
    """
    out = nn_ops.conv2d(
        conv_input,
        self.conv_filters,
        strides=[1, 1, 2, 1],
        dilations=[1, 1, 1, 1],
        padding='SAME',
        data_format='NHWC',
    )

    return {'output': out}


# TODO(b/280208261): Add unit tests for comparing unquantized and
# quantized results
@test_util.run_all_in_graph_and_eager_modes
class QuantizationOptionsTest(quantize_model_test_base.QuantizedModelTest):
  # [CLASS DOC]
  # Tests for how the quantizer interprets the QuantizationOptions protocol buffer: default method,
  # invalid or unspecified methods, component specs and the forced graph-mode calibration option.
  # The class is decorated to run every test in both graph and eager modes. Shared fixtures: the nested
  # SimpleModel and a representative-data generator method.
  """Test cases regarding the use of QuantizationOptions proto.

  Run all tests cases in both the graph mode (default in TF1) and the eager mode
  (default in TF2) to ensure support for when TF2 is disabled.
  """

  class SimpleModel(module.Module):
    # [CLASS DOC]
    # Nested fixture: a minimal module with one random 4 by 3 float32 weight, whose call method does a
    # matmul of a 1 by 4 input with it. Created with no arguments by the tests and saved as a saved model.

    def __init__(self):
      # [FUNCTION DOC]
      # Constructor. Creates the weight attribute as a random uniform float32 array in the range minus one
      # to one with shape 4 by 3. No arguments, returns nothing.
      self.filters = np.random.uniform(low=-1.0, high=1.0, size=(4, 3)).astype(
          'f4'
      )

    @def_function.function(
        input_signature=[
            tensor_spec.TensorSpec(shape=[1, 4], dtype=dtypes.float32)
        ]
    )
    def __call__(self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
      # [FUNCTION DOC]
      # Traced call method (fixed input spec: float32 tensor of shape 1 by 4). Multiplies the input by the
      # weight and returns a mapping with the single key output. Input: input_tensor, the left operand.
      """Performs a matrix multiplication.

      Args:
        input_tensor: Input tensor to matmul with the filter.

      Returns:
        A map of: output key -> output result.
      """

      out = math_ops.matmul(input_tensor, self.filters)
      return {'output': out}

  def _simple_model_data_gen(self) -> repr_dataset.RepresentativeDataset:
    # [FUNCTION DOC]
    # HELPER: supplies representative calibration data for the tests of this class.
    # Generator that yields eight samples. Each sample is a mapping from the input name input_tensor to
    # a tensor of shape 1 by 4 filled with random float32 values in the range 0 to 150. Takes no arguments.
    """Creates an interable of representative samples.

    Yields:
      Representative samples, which is basically a mapping of: input key ->
      input value.
    """
    for _ in range(8):
      yield {
          'input_tensor': ops.convert_to_tensor(
              np.random.uniform(low=0, high=150, size=(1, 4)).astype('f4')
          ),
      }

  def test_static_range_quantization_by_default(self):
    # [FUNCTION DOC]
    # Test: calling the quantizer with no options defaults to static range quantization.
    # Saves the simple model, quantizes it with default options and a representative dataset, asserts a
    # model comes back with only the serving_default signature, then asserts that quantizing without a
    # representative dataset raises a value error mentioning representative_dataset (which proves static
    # range calibration was selected).
    model = self.SimpleModel()

    saved_model_save.save(model, self._input_saved_model_path)

    # Use default QuantizationOptions.
    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        representative_dataset=self._simple_model_data_gen(),
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    # Indirectly prove that it is performing a static-range quantization
    # by checking that it complains about representative_dataset when it is
    # not provided.
    with self.assertRaisesRegex(ValueError, 'representative_dataset'):
      quantize_model.quantize(self._input_saved_model_path)

  def test_method_unspecified_raises_value_error(self):
    # [FUNCTION DOC]
    # Test: an options message whose preset method is explicitly unspecified is rejected.
    # Saves the simple model, builds options with the unspecified preset method and asserts the quantizer
    # raises a value error.
    model = self.SimpleModel()

    saved_model_save.save(model, self._input_saved_model_path)

    options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_UNSPECIFIED
        )
    )

    with self.assertRaises(ValueError):
      quantize_model.quantize(
          self._input_saved_model_path, quantization_options=options
      )

  def test_predefined_method_component_spec(self):
    # [FUNCTION DOC]
    # Test: populating component specs for the preset static range int8 method.
    # Calls the private component-spec population routine on the options and asserts that exactly three
    # component specs (activation, weight and bias) were generated. No model is saved.
    options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        )
    )
    quantize_model._populate_quantization_component_spec(
        options.quantization_method
    )

    # Quantize activation, weight and bias for static range quantization.
    self.assertLen(options.quantization_method.quantization_component_specs, 3)

  def test_invalid_spec_raise_value_error(self):
    # [FUNCTION DOC]
    # Test: an invalid user-supplied component spec is rejected.
    # Builds a method with a single spec that asks for a 4-bit activation, which is not a valid
    # configuration, and asserts that populating the component specs raises a value error.
    options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            quantization_component_specs=[
                quant_opts_pb2.QuantizationComponentSpec(
                    quantization_component=(
                        _QuantizationComponent.COMPONENT_ACTIVATION
                    ),
                    tensor_type=_TensorType.TENSORTYPE_INT_4,
                )
            ]
        )
    )

    with self.assertRaises(ValueError):
      # Activation 4bit is not a valid configuration.
      quantize_model._populate_quantization_component_spec(
          options.quantization_method
      )

  def test_invalid_method_raises_value_error(self):
    # [FUNCTION DOC]
    # Test: a preset method value that is not a valid enum member is rejected.
    # Saves the simple model, builds options with preset method set to minus one, and asserts that the
    # quantizer raises a value error.
    model = self.SimpleModel()

    saved_model_save.save(model, self._input_saved_model_path)

    # Set an invalid value of -1 to QuantizationMethod.preset_method.
    options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(preset_method=-1)
    )

    with self.assertRaises(ValueError):
      quantize_model.quantize(
          self._input_saved_model_path, quantization_options=options
      )

  def test_drq_per_channel_for_non_uniform_opset_raises_value_error(
      self,
  ):
    # [FUNCTION DOC]
    # Test: per-channel dynamic range quantization is rejected for the plain TF op set.
    # Saves the simple model and builds dynamic range int8 options with the TF op set and per-channel
    # quantization enabled; asserts the quantizer raises a value error.
    model = self.SimpleModel()

    saved_model_save.save(model, self._input_saved_model_path)

    options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_DYNAMIC_RANGE_INT8
        ),
        op_set=quant_opts_pb2.TF,
        enable_per_channel_quantization=True,
    )

    with self.assertRaises(ValueError):
      quantize_model.quantize(
          self._input_saved_model_path, quantization_options=options
      )

  def test_force_graph_mode_calibration(self):
    # [FUNCTION DOC]
    # Test: the force-graph-mode-calibration option makes calibration run in graph mode.
    # Saves the simple model, quantizes it with static range options and the option enabled while
    # capturing INFO logs (temporarily raising the global logging verbosity to INFO and restoring it
    # afterwards), and asserts the logs are not empty and contain the message that calibration is
    # executed in graph mode. Side effect: changes and restores the TensorFlow logger verbosity.
    model = self.SimpleModel()

    saved_model_save.save(model, self._input_saved_model_path)

    options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        op_set=quant_opts_pb2.TF,
        force_graph_mode_calibration=True,
    )

    with self.assertLogs(level='INFO') as info_logs:
      # Save the logger verbosity.
      prev_log_level = logging.get_verbosity()
      logging.set_verbosity(logging.INFO)

      try:
        quantize_model.quantize(
            self._input_saved_model_path,
            quantization_options=options,
            representative_dataset=self._simple_model_data_gen(),
        )
      finally:
        # Restore the logger verbosity.
        logging.set_verbosity(prev_log_level)

      self.assertNotEmpty(info_logs.records)
      self.assertTrue(
          self._any_log_contains(
              'Calibration step is executed in graph mode.',
              info_logs.records,
          )
      )


class TensorNamePreservationTest(quantize_model_test_base.QuantizedModelTest):
  # [CLASS DOC]
  # Tests that quantization keeps the input and output tensor names / signature maps of the model
  # unchanged, including awkward cases such as duplicate outputs. Uses the shared base class fixtures.

  def test_preserving_input_output_tensor_names(self):
    # [FUNCTION DOC]
    # Test: signature maps survive static range quantization for multiple signatures.
    # Defines a local module with three traced functions (one returning several top-k derived outputs,
    # one returning the same tensor under two output names after fake-quant, one returning only some
    # outputs), saves it with all three signatures, reads the signature map, quantizes with static range
    # int8 and the TF op set (no representative dataset is passed), reads the output signature map, and
    # asserts the original keys are the expected set and that the original and converted signature maps
    # are equal.
    class MultiSignatureModel(module.Module):
      # [CLASS DOC]
      # Local fixture module with three traced functions that share the same named float32 input of shape
      # 32: multiple_output_ops, duplicate_outputs and return_higher_index_only.

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(
                  name='input', shape=[32], dtype=dtypes.float32
              ),
          ]
      )
      def multiple_output_ops(
          self, input_tensor: core.Tensor
      ) -> Mapping[str, core.Tensor]:
        # [FUNCTION DOC]
        # Traced function. Computes the top four values and indices of the input, adds 2 to the values and
        # returns a mapping with keys indices, adj_values and values. Input: input_tensor, vector of 32.
        k = array_ops.constant(4, dtype=dtypes.int32)
        values, indices = nn_ops.top_k(input_tensor, k, name='TopK')
        adj_values = values + 2
        return {'indices': indices, 'adj_values': adj_values, 'values': values}

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(
                  name='input', shape=[32], dtype=dtypes.float32
              ),
          ]
      )
      def duplicate_outputs(
          self, input_tensor: core.Tensor
      ) -> Mapping[str, core.Tensor]:
        # [FUNCTION DOC]
        # Traced function. Passes the input through a fake-quant op (range minus 0.1 to 0.2, 8 bits), adds 2
        # and returns the same tensor under two output keys, adj_values_1 and adj_values_2, to test
        # duplicate output tensors. Input: input_tensor, vector of 32.
        q_input = array_ops.fake_quant_with_min_max_args(
            input_tensor, min=-0.1, max=0.2, num_bits=8, narrow_range=False
        )
        adj_values = q_input + 2
        return {'adj_values_1': adj_values, 'adj_values_2': adj_values}

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(
                  name='input', shape=[32], dtype=dtypes.float32
              ),
          ]
      )
      def return_higher_index_only(
          self, input_tensor: core.Tensor
      ) -> Mapping[str, core.Tensor]:
        # [FUNCTION DOC]
        # Traced function. Computes the top four values and indices of the input, adds 2 to the values and
        # returns a mapping with only indices and adj_values (the highest-index output of the sibling function
        # is not returned). Input: input_tensor, vector of 32.
        k = array_ops.constant(4, dtype=dtypes.int32)
        values, indices = nn_ops.top_k(input_tensor, k, name='TopK')
        adj_values = values + 2
        return {'indices': indices, 'adj_values': adj_values}

    model = MultiSignatureModel()
    signatures = {
        'multiple_output_ops': model.multiple_output_ops,
        'duplicate_outputs': model.duplicate_outputs,
        'return_higher_index_only': model.return_higher_index_only,
    }
    saved_model_save.save(
        model, self._input_saved_model_path, signatures=signatures
    )

    tags = {tag_constants.SERVING}
    original_signature_map = save_model.get_signatures_from_saved_model(
        self._input_saved_model_path,
        signature_keys=signatures.keys(),
        tags=tags,
    )

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signatures.keys(),
        op_set=quant_opts_pb2.TF,
    )
    quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    converted_signature_map = save_model.get_signatures_from_saved_model(
        self._output_saved_model_path,
        signature_keys=signatures.keys(),
        tags=tags,
    )

    # The original and converted model should have the same signature map.
    self.assertAllInSet(
        list(original_signature_map.keys()), set(signatures.keys())
    )
    self.assertDictEqual(original_signature_map, converted_signature_map)

  def test_duplicated_tensor_name(self):
    # [FUNCTION DOC]
    # Test: a TF1 graph whose fake-quant output feeds a sqrt and identity keeps its signature map.
    # Builds a graph in a session (placeholder, fake-quant, sqrt, identity named output), saves it as a
    # TF1 saved model with a predict signature under the key main, quantizes it with static range int8 and
    # the TF op set, and asserts the signature map read back from the output equals the original.
    with session.Session(graph=ops.Graph()) as sess:
      input_tensor = array_ops.placeholder(
          dtypes.float32, shape=[], name='input'
      )
      q_input = array_ops.fake_quant_with_min_max_args(
          input_tensor, min=-0.1, max=0.2, num_bits=8, narrow_range=False
      )
      sqrt = math_ops.sqrt(q_input, name='sqrt')
      identity = array_ops.identity(sqrt, name='output')

      input_map = {'input': input_tensor}
      output_map = {'sqrt': identity}
      signature = signature_def_utils_impl.predict_signature_def(
          inputs=input_map, outputs=output_map
      )
      signature_map = {'main': signature}

      tags = {tag_constants.SERVING}
      v1_builder = builder.SavedModelBuilder(self._input_saved_model_path)
      v1_builder.add_meta_graph_and_variables(
          sess, tags, signature_def_map=signature_map
      )
      v1_builder.save()

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_map.keys(),
        op_set=quant_opts_pb2.TF,
    )
    quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    converted_signature_map = save_model.get_signatures_from_saved_model(
        self._output_saved_model_path,
        signature_keys=signature_map.keys(),
        tags=tags,
    )
    # The original and converted model should have the same signature map.
    self.assertDictEqual(signature_map, converted_signature_map)


class StaticRangeQuantizationTest(quantize_model_test_base.QuantizedModelTest):
  # [CLASS DOC]
  # Tests for static range (post-training with calibration, and quantization-aware) quantization across
  # op sets: matmul, convolution, einsum, gather, depthwise convolution, 3D convolution, while loops,
  # lookup tables, variables, TF1 saved models, signatures, representative dataset formats, function
  # aliases and output directory handling. Uses model builders and inspection helpers from the base class.

  @parameterized.parameters(
      testing.parameter_combinations([{
          'shapes': [
              ([3, 3], [3, 3]),
              ([3, None], [None, 3]),
              ([None, None], [None, None]),
              ([4, 3, 3], [4, 3, 3]),
              ([4, 3, None], [4, None, 3]),
              ([None, None, None], [None, None, None]),
          ],
          'activation_fn': [None, nn_ops.relu, nn_ops.relu6],
          'has_bias': [True, False],
          'use_kernel': [True, False],
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_qat_matmul_model(
      self,
      shapes: Sequence[Tuple[_TensorShape, _TensorShape]],
      activation_fn: Optional[ops.Operation],
      has_bias: bool,
      use_kernel: bool,
  ):
    # [FUNCTION DOC]
    # Parameterized QAT test: matmul model with fake-quant ops, with or without bias, activation and
    # kernel as constant.
    # Parameters: shapes (pair of shapes for the two operands, possibly with unknown dimensions, which are
    # set to 5 for the real data), activation_fn (none, relu or relu6), has_bias, use_kernel (constant
    # kernel versus second input). Builds a local module, saves it with its signature, quantizes it with
    # static range int8 for the TF op set, asserts the signature key, closeness of outputs (tolerance 0.1)
    # and a quantized function call in the graph; then quantizes into the XLA op set (two input tensors
    # enabled when no kernel) and asserts a dot-general op is present and results are close to the
    # original and to the TF output path.
    n = 5
    x_shape = [v if v is not None else n for v in shapes[0]]
    y_shape = [v if v is not None else n for v in shapes[1]]

    class MatmulModel(module.Module):
      # [CLASS DOC]
      # Local fixture module for the QAT matmul test. Holds an optional bias tensor, a random kernel and
      # the fake-quant min/max constants. Exposes two traced functions, with and without a constant kernel,
      # both delegating to the private matmul method.

      def __init__(self, bias: Optional[core.Tensor]):
        # [FUNCTION DOC]
        # Constructor. Stores the optional bias, creates a random float32 kernel with the right operand shape
        # and sets the fake-quant minimum and maximum triples. Input: bias, a tensor or nothing.
        self._bias = bias
        self._kernel = np.random.uniform(size=y_shape).astype('f4')
        self._min = (-0.8, -0.8, -0.9)
        self._max = (0.9, 0.9, 1.0)

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(
                  name='x', shape=shapes[0], dtype=dtypes.float32
              )
          ]
      )
      def matmul_with_kernel(self, x: core.Tensor) -> Mapping[str, core.Tensor]:
        # [FUNCTION DOC]
        # Traced function with one named input x (shape given by the test parameters). Multiplies x by the
        # stored kernel through the private matmul helper and returns its result mapping. Input: x.
        return self._matmul(x, self._kernel)

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(
                  name='x', shape=shapes[0], dtype=dtypes.float32
              ),
              tensor_spec.TensorSpec(
                  name='y', shape=shapes[1], dtype=dtypes.float32
              ),
          ]
      )
      def matmul_without_kernel(
          self, x: core.Tensor, y: core.Tensor
      ) -> Mapping[str, core.Tensor]:
        # [FUNCTION DOC]
        # Traced function with two named inputs x and y. Multiplies them through the private matmul helper
        # and returns the result mapping. Inputs: x and y, the operands.
        return self._matmul(x, y)

      def _matmul(self, x, y):
        # [FUNCTION DOC]
        # HELPER: shared body of the two traced matmul functions above.
        # Fake-quantizes both operands (8 bits, not narrow range), multiplies them, adds the bias if present,
        # applies the activation if one was chosen, fake-quantizes the result and returns a mapping with the
        # single key output. Inputs: x and y, the operand tensors.
        x = array_ops.fake_quant_with_min_max_vars(
            x,
            min=ops.convert_to_tensor(self._min[0]),
            max=ops.convert_to_tensor(self._max[0]),
            num_bits=8,
            narrow_range=False,
        )
        y = array_ops.fake_quant_with_min_max_vars(
            y,
            min=ops.convert_to_tensor(self._min[1]),
            max=ops.convert_to_tensor(self._max[1]),
            num_bits=8,
            narrow_range=False,
        )

        out = math_ops.matmul(x, y)
        if self._bias is not None:
          out = nn_ops.bias_add(out, self._bias)
        if activation_fn is not None:
          out = activation_fn(out)
        out = array_ops.fake_quant_with_min_max_vars(
            out,
            min=ops.convert_to_tensor(self._min[2]),
            max=ops.convert_to_tensor(self._max[2]),
            num_bits=8,
            narrow_range=False,
        )
        return {'output': out}

    bias = None
    if has_bias:
      bias_shape = shapes[1][-1]
      if bias_shape is not None:
        bias = array_ops.constant(
            np.random.uniform(size=[shapes[1][-1]]), dtype=dtypes.float32
        )
    model = MatmulModel(bias)
    x = array_ops.constant(
        np.random.uniform(size=x_shape), dtype=dtypes.float32
    )
    y = array_ops.constant(
        np.random.uniform(size=y_shape), dtype=dtypes.float32
    )
    if use_kernel:
      model.matmul = model.matmul_with_kernel
      model_inputs = {'x': x}
    else:
      model.matmul = model.matmul_without_kernel
      model_inputs = {'x': x, 'y': y}

    saved_model_save.save(
        model, self._input_saved_model_path, signatures=model.matmul
    )

    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

    # Check the converted model with TF opset as the baseline.
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=quant_opts_pb2.TF,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {signature_key}
    )

    expected_outputs = model.matmul(**model_inputs)
    got_outputs = converted_model.signatures[signature_key](**model_inputs)
    self.assertAllClose(expected_outputs, got_outputs, atol=1e-1)

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

    # Check the converted model in the XLA opset.
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=quant_opts_pb2.XLA,
        enable_two_input_tensors=not use_kernel,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path_2,
        quantization_options,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {signature_key}
    )
    loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path_2
    )
    graphdef = loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_op(graphdef, 'XlaDotV2'))

    new_outputs = converted_model.signatures[signature_key](**model_inputs)

    # The difference between TF and XLA path is expected to be small (smaller
    # or equal to 1 in the quantized domain).
    self.assertAllClose(new_outputs, expected_outputs, atol=1e-1)

  @parameterized.parameters(
      testing.parameter_combinations([{
          'activation_fn': [None, nn_ops.relu, nn_ops.relu6],
          'has_bias': [True, False],
          'has_batch_norm': [True, False],
          'target_opset': [quant_opts_pb2.XLA],
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_qat_conv_model(
      self,
      activation_fn: Optional[ops.Operation],
      has_bias: bool,
      has_batch_norm: bool,
      target_opset: quant_opts_pb2.OpSet,
  ):
    # [FUNCTION DOC]
    # Parameterized QAT test: 2D convolution with fake-quant ops, optional bias, batch norm and activation.
    # Parameters: activation_fn, has_bias, has_batch_norm (applied only when an activation is given) and
    # target_opset (the XLA op set). Builds a local conv module, saves it, quantizes for the TF op set
    # and checks signature, output closeness (tight tolerance) and quantized function calls; then
    # quantizes for the target op set and, for XLA, checks a convolution op with the expected node name
    # exists, and that outputs of both paths are close.
    class ConvModel(module.Module):
      # [CLASS DOC]
      # Local fixture module: a single traced 2D convolution (input 1,3,4,3, filter 2,3,3,2) wrapped with
      # fake-quant ops on the input, per-channel filter and output, plus optional bias, batch norm and
      # activation depending on enclosing test parameters.

      def __init__(self):
        # [FUNCTION DOC]
        # Constructor. Creates the filter value as a random float32 array in the range minus 0.5 to 0.5 with
        # shape 2,3,3,2. No arguments, returns nothing.
        self.filter_value = np.random.uniform(
            low=-0.5, high=0.5, size=(2, 3, 3, 2)
        ).astype('f4')

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(
                  name='input', shape=[1, 3, 4, 3], dtype=dtypes.float32
              ),
          ]
      )
      def conv(self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
        # [FUNCTION DOC]
        # Traced function (fixed input spec: float32 of shape 1,3,4,3, named input). Fake-quantizes the
        # input, filter (per channel) and output, runs a SAME-padded convolution named sample/conv2d, then
        # optionally adds bias, and, if an activation is chosen, optionally applies inference-mode fused batch
        # norm followed by the activation. Returns a mapping with one key, output. Input: input_tensor.
        """Performs a 2D convolution operation.

        Args:
          input_tensor: Input tensor to perform convolution on.

        Returns:
          A map of: output key -> output result.
        """
        q_input = array_ops.fake_quant_with_min_max_args(
            input_tensor, min=-0.1, max=0.2, num_bits=8, narrow_range=False
        )
        filter_tensor = ops.convert_to_tensor(self.filter_value)
        filter_min = array_ops.identity(
            array_ops.constant([-0.5, -0.5], dtype=dtypes.float32)
        )
        filter_max = array_ops.identity(
            array_ops.constant([0.5, 0.5], dtype=dtypes.float32)
        )
        q_filter = array_ops.fake_quant_with_min_max_vars_per_channel(
            filter_tensor, filter_min, filter_max, num_bits=8, narrow_range=True
        )
        bias = array_ops.constant([0.1, 0.2], dtype=dtypes.float32)
        scale, offset = [1.0] * 2, [0.5] * 2
        mean, variance = scale, offset
        out = nn_ops.conv2d(
            q_input,
            q_filter,
            strides=[1, 1, 2, 1],
            dilations=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC',
            name='sample/conv2d',
        )
        if has_bias:
          out = nn_ops.bias_add(out, bias, data_format='NHWC')
        if activation_fn is not None:
          # The accuracy is not good when having FusedBatchNorm without
          # activation in this test.
          if has_batch_norm:
            # Fusing is supported for non-training case.
            out, _, _, _, _, _ = nn_ops.fused_batch_norm_v3(
                out, scale, offset, mean, variance, is_training=False
            )
          out = activation_fn(out)
        out_min = array_ops.constant([-0.18, -0.32], dtype=dtypes.float32)
        out_max = array_ops.constant([0.5, 0.5], dtype=dtypes.float32)
        q_out = array_ops.fake_quant_with_min_max_vars_per_channel(
            out, min=out_min, max=out_max, num_bits=8, narrow_range=True
        )
        return {'output': q_out}

    model = ConvModel()
    saved_model_save.save(model, self._input_saved_model_path)

    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

    # Check the converted model with TF opset as the baseline.
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=quant_opts_pb2.TF,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {signature_key}
    )

    input_data = np.random.uniform(
        low=-0.1, high=0.2, size=(1, 3, 4, 3)
    ).astype('f4')
    expected_outputs = model.conv(input_data)
    got_outputs = converted_model.signatures[signature_key](
        input=ops.convert_to_tensor(input_data)
    )
    self.assertAllClose(expected_outputs, got_outputs, atol=0.00323)

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

    # Check the converted model in the target opset.
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=target_opset,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path_2,
        quantization_options,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {signature_key}
    )
    loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path_2
    )
    graphdef = loader.get_meta_graph_def_from_tags(tags).graph_def
    if target_opset == quant_opts_pb2.XLA:
      self.assertTrue(
          self._contains_op(graphdef, 'XlaConvV2', node_name='sample/conv2d.*')
      )

    new_outputs = converted_model.signatures[signature_key](
        input=ops.convert_to_tensor(input_data)
    )
    # The difference between TF and XLA path is expected to be small (smaller
    # or equal to 1 in the quantized domain).
    self.assertAllClose(new_outputs, got_outputs, atol=0.00154)

  # Currently, only some specific forms of equantions are supported for
  # batchmatmul conversion.
  @parameterized.parameters(
      testing.parameter_combinations([{
          'equation': ('abc,cd->abd', 'abcd,cde->abe'),
          'shape_unknown': (True, False),
          'activation_fn': (None, nn_ops.relu, nn_ops.relu6),
          'has_bias': (True, False),
          'use_kernel': (True, False),
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_qat_einsum_model_with_batchmatmul_conversion(
      self,
      equation: str,
      shape_unknown: bool,
      activation_fn: Optional[ops.Operation],
      has_bias: bool,
      use_kernel: bool,
  ):
    # [FUNCTION DOC]
    # Parameterized QAT test: einsum models whose equations can be converted to batch matmul.
    # Parameters: equation (two supported equation strings), shape_unknown (dynamic shapes), activation_fn,
    # has_bias, use_kernel (constant versus input second operand). Builds an einsum model with fake-quant
    # via base class helpers, saves it, quantizes into the TF op set and asserts closeness, signature and
    # quantized function call; then quantizes into XLA and asserts a dot op exists and outputs are close.
    x_shape, y_shape, bias_shape, x_signature, y_signature = (
        self._prepare_sample_einsum_datashapes(
            equation, shape_unknown, has_bias and not shape_unknown
        )
    )
    model = self._create_einsum_model(
        equation,
        y_shape,
        x_signature,
        y_signature,
        bias_shape,
        activation_fn,
        is_qat_model=True,
    )
    x = array_ops.constant(
        np.random.uniform(size=x_shape), dtype=dtypes.float32
    )
    y = array_ops.constant(
        np.random.uniform(size=y_shape), dtype=dtypes.float32
    )
    if use_kernel:
      model.einsum = model.einsum_with_kernel
      model_inputs = {'x': x}
    else:
      model.einsum = model.einsum_without_kernel
      model_inputs = {'x': x, 'y': y}

    saved_model_save.save(
        model, self._input_saved_model_path, signatures=model.einsum
    )

    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

    # Check the converted model with TF opset as the baseline.
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=quant_opts_pb2.TF,
        enable_two_input_tensors=not use_kernel,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {signature_key}
    )

    expected_outputs = model.einsum(**model_inputs)
    got_outputs = converted_model.signatures[signature_key](**model_inputs)
    self.assertAllClose(expected_outputs, got_outputs, atol=1e-1)

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

    # Check the converted model in the XLA opset.
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=quant_opts_pb2.XLA,
        enable_two_input_tensors=not use_kernel,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path_2,
        quantization_options,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {signature_key}
    )
    loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path_2
    )
    graphdef = loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_op(graphdef, 'XlaDotV2'))

    new_outputs = converted_model.signatures[signature_key](**model_inputs)

    # The difference between TF and XLA path is expected to be small (smaller
    # or equal to 1 in the quantized domain).
    self.assertAllClose(new_outputs, expected_outputs, atol=1e-1)

  # Equations only supported for XLA operations.
  @parameterized.parameters(
      testing.parameter_combinations([{
          'equation': ('abc,acd->abd', 'abcd,aecd->acbe'),
          'shape_unknown': (True, False),
          'activation_fn': (None, nn_ops.relu, nn_ops.relu6),
          'has_bias': (True, False),
          'use_kernel': (True, False),
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_qat_einsum_model_with_xla(
      self,
      equation: str,
      shape_unknown: bool,
      activation_fn: Optional[ops.Operation],
      has_bias: bool,
      use_kernel: bool,
  ):
    # [FUNCTION DOC]
    # Parameterized QAT test: einsum equations that are supported only by the XLA op set.
    # Parameters as in the batch-matmul variant (equation, shape_unknown, activation_fn, has_bias,
    # use_kernel). Saves a QAT einsum model, quantizes into XLA with static range int8 and asserts the
    # signature, a dot op in the graph and closeness of outputs to the float model.
    x_shape, y_shape, bias_shape, x_signature, y_signature = (
        self._prepare_sample_einsum_datashapes(
            equation, shape_unknown, has_bias and not shape_unknown
        )
    )
    model = self._create_einsum_model(
        equation,
        y_shape,
        x_signature,
        y_signature,
        bias_shape,
        activation_fn,
        is_qat_model=True,
    )

    x = array_ops.constant(
        np.random.uniform(size=x_shape), dtype=dtypes.float32
    )
    y = array_ops.constant(
        np.random.uniform(size=y_shape), dtype=dtypes.float32
    )
    if use_kernel:
      model.einsum = model.einsum_with_kernel
      model_inputs = {'x': x}
    else:
      model.einsum = model.einsum_without_kernel
      model_inputs = {'x': x, 'y': y}

    saved_model_save.save(
        model, self._input_saved_model_path, signatures=model.einsum
    )

    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

    # Check the converted model in the XLA opset.
    expected_outputs = model.einsum(**model_inputs)
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=quant_opts_pb2.XLA,
        enable_two_input_tensors=not use_kernel,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {signature_key}
    )
    loader = saved_model_loader.SavedModelLoader(self._output_saved_model_path)
    graphdef = loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_op(graphdef, 'XlaDotV2'))

    outputs = converted_model.signatures[signature_key](**model_inputs)

    self.assertAllClose(outputs, expected_outputs, atol=1e-1)

  # Equations NOT supported for XLA operations.
  @parameterized.parameters(
      testing.parameter_combinations([{
          'equation': ('aecd,abcd->acbe', 'abc,acd->adb'),
          'use_kernel': (True, False),
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_qat_einsum_model_not_supported_with_xla(
      self,
      equation: str,
      use_kernel: bool,
  ):
    # [FUNCTION DOC]
    # Parameterized QAT test: einsum equations that XLA does not support are left unconverted.
    # Parameters: equation and use_kernel. Saves a QAT einsum model, quantizes with the XLA op set and
    # asserts the signature exists but no XLA dot op appears in the output graph.
    _, y_shape, _, x_signature, y_signature = (
        self._prepare_sample_einsum_datashapes(equation)
    )

    model = self._create_einsum_model(
        equation,
        y_shape,
        x_signature,
        y_signature,
        bias_shape=None,
        activation_fn=None,
        is_qat_model=True,
    )

    if use_kernel:
      model.einsum = model.einsum_with_kernel
    else:
      model.einsum = model.einsum_without_kernel

    saved_model_save.save(
        model, self._input_saved_model_path, signatures=model.einsum
    )

    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

    # Check the converted model does NOT have XLA opset.
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=quant_opts_pb2.XLA,
        enable_two_input_tensors=not use_kernel,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path_2,
        quantization_options,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {signature_key}
    )
    loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path_2
    )
    graphdef = loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertFalse(self._contains_op(graphdef, 'XlaDotV2'))

  @test_util.run_in_graph_and_eager_modes
  def test_qat_gather_and_conv_model(
      self,
  ):
    # [FUNCTION DOC]
    # Test: QAT gather plus convolution model quantized to XLA gets smaller.
    # Builds the int32 gather and convolution model with fake-quant ops (large filter), saves it,
    # quantizes with static range int8 and XLA, and asserts the output model is under half of the input
    # model size.
    input_type = dtypes.int32
    model = self._create_simple_gather_and_conv_model(
        input_type,
        filter_shape=(2, 3, 3, 1024),
        is_qat_model=True,
    )

    saved_model_save.save(model, self._input_saved_model_path)

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.XLA,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    self.assertIsNotNone(converted_model)
    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.5,
    )

  def test_qat_vocab_table_lookup_model(self):
    # [FUNCTION DOC]
    # Test: QAT model with a vocabulary hash table lookup still works after quantization.
    # Creates and saves a TF1 model with a vocabulary table, asserts expected input and output keys, quantizes
    # with a list representative dataset, then loads the output in a session and asserts it contains a
    # quantized function call and the signature, and that looking up three words returns indices 1, 2 and 0.
    tags = {tag_constants.SERVING}
    signature_def_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    # Create and save a simple model that involves a hash table.
    inputs, outputs = self._create_and_save_vocab_table_lookup_qat_model_tf1(
        self._input_saved_model_path, tags, signature_def_key
    )

    # Make sure that the desired input key and output key is present.
    self.assertIn('input_vocabs', inputs.keys())
    self.assertIn('lookup', outputs.keys())

    # Representative dataset is composed of a set of vocabs for table lookup.
    repr_ds = [
        {'input_vocabs': np.array([b'hello', b'model', b'quantization'])}
        for _ in range(4)
    ]

    signature_def_keys = [signature_def_key]
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_def_keys,
        op_set=quant_opts_pb2.TF,
    )

    quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=repr_ds,
    )

    # Tests table lookup to make sure the table has been initialized
    # successfully.
    with session.Session(graph=ops.Graph()) as sess:
      output_meta_graph_def = saved_model_loader.load(
          sess, tags=tags, export_dir=self._output_saved_model_path
      )

      # The graph should contain a quantized function call (it contains a
      # single f32 matmul node).
      self.assertTrue(
          self._contains_quantized_function_call(
              output_meta_graph_def.graph_def
          )
      )
      self.assertCountEqual(
          output_meta_graph_def.signature_def.keys(), signature_def_keys
      )

      signature_def = output_meta_graph_def.signature_def[signature_def_key]

      input_tensor_name = signature_def.inputs['input_vocabs'].name
      input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)

      lookup_tensor_name = signature_def.outputs['lookup'].name
      lookup_tensor = sess.graph.get_tensor_by_name(lookup_tensor_name)

      lookup_val = sess.run(
          lookup_tensor,
          feed_dict={
              input_tensor: np.array([b'model', b'quantization', b'hello'])
          },
      )

      self.assertAllClose(lookup_val, [1.0, 2.0, 0.0])

  def test_qat_file_init_hash_table_lookup_model_tf1(self):
    # [FUNCTION DOC]
    # Test: QAT model with a file-initialized hash table lookup keeps its table after quantization.
    # Same flow as the vocabulary table test but the table is initialized from a file; asserts the quantized
    # function call and that looking up a missing word gives the default minus one and known words give 2 and 1.
    tags = {tag_constants.SERVING}
    signature_def_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    # Create and save a simple model that involves a hash table.
    inputs, outputs = self._create_and_save_file_init_hash_table_qat_model_tf1(
        self._input_saved_model_path, tags, signature_def_key
    )

    # Make sure that the desired input key and output key is present.
    self.assertIn('input_vocabs', inputs.keys())
    self.assertIn('lookup', outputs.keys())

    # Representative dataset is composed of a set of vocabs for table lookup.
    repr_ds = [
        {'input_vocabs': np.array([b'static', b'range', b'quantization'])}
        for _ in range(4)
    ]

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_def_key],
        op_set=quant_opts_pb2.TF,
    )
    signature_def_keys = [signature_def_key]

    quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=repr_ds,
    )

    # Tests table lookup to make sure the table has been initialized
    # successfully.
    with session.Session(graph=ops.Graph()) as sess:
      output_meta_graph_def = saved_model_loader.load(
          sess, tags=tags, export_dir=self._output_saved_model_path
      )

      # The graph should contain a quantized function call (it contains a
      # single f32 matmul node).
      self.assertTrue(
          self._contains_quantized_function_call(
              output_meta_graph_def.graph_def
          )
      )
      self.assertCountEqual(
          output_meta_graph_def.signature_def.keys(), signature_def_keys
      )

      signature_def = output_meta_graph_def.signature_def[signature_def_key]
      input_tensor_name = signature_def.inputs['input_vocabs'].name
      input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)
      lookup_tensor_name = signature_def.outputs['lookup'].name
      lookup_tensor = sess.graph.get_tensor_by_name(lookup_tensor_name)

      lookup_val = sess.run(
          lookup_tensor,
          feed_dict={
              input_tensor: np.array([b'dynamic', b'quantization', b'range'])
          },
      )

      # "dynamic" is not in the table: -1 (default value)
      self.assertAllClose(lookup_val, [-1.0, 2.0, 1.0])

  # Run this test only with the eager mode.
  @test_util.run_v2_only
  def test_ptq_model_with_variable(self):
    # [FUNCTION DOC]
    # Test: PTQ of a convolution model that keeps its filter as a variable (eager mode only).
    # Defines a local module with a variable filter, saves it, quantizes with static range int8 and the TF
    # op set using a generated representative dataset, and asserts the signature and a quantized function
    # call in the output.
    class ConvModelWithVariable(module.Module):
      # [CLASS DOC]
      # Local fixture module performing a single 2D convolution with its filter stored as a variable.
      """A simple model that performs a single convolution to the input tensor.

      It keeps the filter as a tf.Variable.
      """

      def __init__(self) -> None:
        # [FUNCTION DOC]
        # Constructor. Creates the filter as a variable initialized with random values (shape 2,3,3,2,
        # range minus 1 to 1). Returns nothing.
        """Initializes the filter variable."""
        self.filters = variables.Variable(
            random_ops.random_uniform(
                shape=(2, 3, 3, 2), minval=-1.0, maxval=1.0
            )
        )

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(
                  name='input', shape=(1, 3, 4, 3), dtype=dtypes.float32
              ),
          ]
      )
      def __call__(self, x: core.Tensor) -> Mapping[str, core.Tensor]:
        # [FUNCTION DOC]
        # Traced call method (input named input, shape 1,3,4,3). Convolves with the variable filter using
        # strides 1,1,2,1, SAME padding, NHWC and returns a mapping with the key output. Input: x.
        """Performs a 2D convolution operation.

        Args:
          x: Input tensor to perform convolution on.

        Returns:
          A map of: output key -> output result.
        """
        out = nn_ops.conv2d(
            x,
            self.filters,
            strides=[1, 1, 2, 1],
            dilations=[1, 1, 1, 1],
            padding='SAME',
            data_format='NHWC',
        )
        return {'output': out}

    def gen_data() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: representative data generator for the variable test.
      # Yields eight samples, each a mapping from input to a random tensor of shape 1,3,4,3 in range 0 to 150.
      """Creates an interable of representative samples.

      Yields:
        Representative samples, which is basically a mapping of: input key ->
        input value.
      """
      for _ in range(8):
        yield {
            'input': random_ops.random_uniform(
                shape=(1, 3, 4, 3), minval=0, maxval=150
            )
        }

    model = ConvModelWithVariable()
    saved_model_save.save(model, self._input_saved_model_path)

    signature_keys = [signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_keys,
        op_set=quant_opts_pb2.TF,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=gen_data(),
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), signature_keys
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  # Check only the most simple case and the most complicated cases.
  @parameterized.named_parameters(
      {
          'testcase_name': 'none',
          'activation_fn': None,
          'has_bias': False,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'dilation',
          'activation_fn': None,
          'has_bias': False,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'relu',
          'activation_fn': nn_ops.relu,
          'has_bias': False,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'dilation_relu',
          'activation_fn': nn_ops.relu,
          'has_bias': False,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'relu6',
          'activation_fn': nn_ops.relu6,
          'has_bias': False,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'dilation_relu6',
          'activation_fn': nn_ops.relu6,
          'has_bias': False,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'bn',
          'activation_fn': None,
          'has_bias': False,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'dilation_bn',
          'activation_fn': None,
          'has_bias': False,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'with_bias',
          'activation_fn': None,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'dilation_with_bias',
          'activation_fn': None,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'with_bias_and_relu6',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'dilation_with_bias_and_relu6',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'with_bias_and_bn_and_relu6',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'dilation_with_bias_and_bn_and_relu6',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'with_bias_and_relu6_to_xla_per_tensor',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'with_bias_and_relu6_to_xla_per_channel',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': True,
      },
      {
          'testcase_name': 'dilation_with_bias_and_relu6_to_xla',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'dilation_with_bias_and_relu6_to_xla_per_channel',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': True,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'with_bias_and_bn_and_relu6_to_xla',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'with_bias_and_bn_and_relu6_to_xla_per_channel',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': True,
      },
      {
          'testcase_name': 'dilation_with_bias_and_bn_and_relu6_to_xla',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': (
              'dilation_with_bias_and_bn_and_relu6_to_xla_per_channel'
          ),
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': True,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'with_bias_and_relu6_to_xla_dynamic',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': True,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'with_bias_and_relu6_to_xla_dynamic_per_channel',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': True,
          'enable_per_channel_quantization': True,
      },
      {
          'testcase_name': 'dilation_with_bias_and_relu6_to_xla_dynamic',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': True,
          'enable_per_channel_quantization': False,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': (
              'dilation_with_bias_and_relu6_to_xla_dynamic_per_channel'
          ),
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': True,
          'enable_per_channel_quantization': True,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'with_bias_and_bn_and_relu6_to_xla_dynamic',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': True,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': (
              'with_bias_and_bn_and_relu6_to_xla_dynamic_per_channel'
          ),
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': True,
          'enable_per_channel_quantization': True,
      },
      {
          'testcase_name': 'dilation_with_bias_and_bn_and_relu6_to_xla_dynamic',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': True,
          'enable_per_channel_quantization': False,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': (
              'dilation_with_bias_and_bn_and_relu6_to_xla_dynamic_per_channel'
          ),
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': True,
          'enable_per_channel_quantization': True,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'with_bias_and_relu6_to_uq',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.UNIFORM_QUANTIZED,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'dilation_with_bias_and_relu6_to_uq',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.UNIFORM_QUANTIZED,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'with_bias_and_bn_and_relu6_to_uq',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.UNIFORM_QUANTIZED,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'dilation_with_bias_and_bn_and_relu6_to_uq',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.UNIFORM_QUANTIZED,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'with_bias_and_relu6_to_uq_per_channel',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.UNIFORM_QUANTIZED,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': True,
      },
      {
          'testcase_name': 'dilation_with_bias_and_relu6_to_uq_per_channel',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.UNIFORM_QUANTIZED,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': True,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'with_bias_and_bn_and_relu6_to_uq_per_channel',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.UNIFORM_QUANTIZED,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': True,
      },
      {
          'testcase_name': (
              'dilation_with_bias_and_bn_and_relu6_to_uq_per_channel'
          ),
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.UNIFORM_QUANTIZED,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': True,
          'dilations': [1, 2, 2, 1],
      },
      {
          'testcase_name': 'with_bias_and_relu6_to_stablehlo_per_channel',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.STABLEHLO,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': True,
      },
  )
  @test_util.run_in_graph_and_eager_modes
  def test_conv_ptq_model(
      self,
      activation_fn: Optional[ops.Operation],
      has_bias: bool,
      has_batch_norm: bool,
      target_opset: quant_opts_pb2.OpSet,
      input_shape_dynamic: bool,
      enable_per_channel_quantization: bool,
      dilations: Sequence[int] = None,
  ):
    # [FUNCTION DOC]
    # KNOWN ISSUE: uses the module-level per-channel op names generator twice, so the second count in the
    # equality assertions sees an exhausted iterator.
    # Parameterized PTQ test for 2D convolution. Test cases cover activation, bias, batch norm, dilation,
    # static or dynamic input shape, per-channel and the TF, XLA, uniform quantized and StableHLO op sets.
    # Parameters: activation_fn, has_bias, has_batch_norm, target_opset, input_shape_dynamic,
    # enable_per_channel_quantization, dilations (optional, defaults to none).
    # Builds a conv model, generates 500 random representative samples, quantizes with static range int8,
    # then asserts: signature; for non per-channel, outputs close to float (tolerance 0.06); for XLA, an XLA
    # conv op and, if per channel, a constant with the per-channel size; for uniform quantized, the proper
    # conv op, matching quantization axis attributes and no float conv; for StableHLO, an XLA call-module
    # op; otherwise a quantized function call; and no fused batch norm remains.
    input_shape = [None, None, None, 3] if input_shape_dynamic else [1, 3, 4, 3]
    filter_shape = [2, 3, 3, 2]
    strides = [1, 1, 1, 1]

    model = self._create_conv2d_model(
        input_shape,
        filter_shape,
        has_bias,
        has_batch_norm,
        activation_fn,
        strides,
        dilations,
    )
    saved_model_save.save(model, self._input_saved_model_path)

    # Generate model input data.
    rng = np.random.default_rng(seed=1234)
    static_input_shape = [dim if dim is not None else 2 for dim in input_shape]
    input_data = ops.convert_to_tensor(
        rng.uniform(low=0.0, high=1.0, size=static_input_shape).astype(
            np.float32
        )
    )

    def data_gen() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: representative data generator for the convolution PTQ test.
      # Yields 500 samples, each mapping input_tensor to uniform random float32 values in 0 to 1 with the
      # test's static input shape, drawn from the test's seeded random generator.
      for _ in range(500):
        yield {
            'input_tensor': rng.uniform(
                low=0.0, high=1.0, size=static_input_shape
            ).astype(np.float32)
        }

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
        enable_per_channel_quantization=enable_per_channel_quantization,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen(),
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def

    # The difference between float model and target path quantized model is
    # expected to be small.
    # The atol value is arbitrary.
    if not enable_per_channel_quantization:
      expected_outputs = model.conv(input_data)
      target_outputs = converted_model.signatures['serving_default'](
          input_tensor=ops.convert_to_tensor(input_data)
      )
      self.assertAllClose(target_outputs, expected_outputs, atol=0.06)

    if target_opset == quant_opts_pb2.XLA:
      self.assertTrue(self._contains_op(output_graphdef, 'XlaConvV2'))
      if enable_per_channel_quantization:
        per_channel_size_attr = attr_value_pb2.AttrValue(
            list=attr_value_pb2.AttrValue.ListValue(
                shape=[
                    tensor_shape_pb2.TensorShapeProto(
                        dim=[
                            tensor_shape_pb2.TensorShapeProto.Dim(
                                size=filter_shape[-1]
                            )
                        ]
                    )
                ]
            )
        )
        self.assertTrue(
            self._contains_op(
                output_graphdef,
                'Const',
                '_output_shapes',
                per_channel_size_attr,
            )
        )
    elif target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      self.assertTrue(
          self._contains_op(output_graphdef, 'UniformQuantizedConvolution')
      )
      if enable_per_channel_quantization:
        quantized_axis = 3
        quantized_dim_size_attr = attr_value_pb2.AttrValue(
            list=attr_value_pb2.AttrValue.ListValue(
                shape=[
                    tensor_shape_pb2.TensorShapeProto(
                        dim=[
                            tensor_shape_pb2.TensorShapeProto.Dim(
                                size=filter_shape[quantized_axis]
                            )
                        ]
                    )
                ]
            )
        )
      else:
        quantized_axis = -1
        # Empty dimension. Per-tensor quantization has singular channel.
        quantized_dim_size_attr = attr_value_pb2.AttrValue(
            list=attr_value_pb2.AttrValue.ListValue(
                shape=[tensor_shape_pb2.TensorShapeProto()]
            )
        )
      quantized_axis_attr = attr_value_pb2.AttrValue(i=quantized_axis)
      self.assertEqual(
          self._count_ops(
              output_graphdef,
              _PER_CHANNEL_QUANTIZED_OPS,
              'rhs_quantization_axis',
              quantized_axis_attr,
          ),
          self._count_ops(output_graphdef, _PER_CHANNEL_QUANTIZED_OPS),
      )
      self.assertEqual(
          self._count_ops(
              output_graphdef,
              _PER_CHANNEL_OP_NAMES,
              '_output_shapes',
              quantized_dim_size_attr,
              get_op_name=True,
          ),
          self._count_ops(
              output_graphdef,
              _PER_CHANNEL_OP_NAMES,
              get_op_name=True,
          ),
      )
      self.assertFalse(self._contains_op(output_graphdef, 'Conv2D'))
    elif target_opset == quant_opts_pb2.STABLEHLO:
      # This is to verify the invocation of StableHLO quantizer works. More
      # thorough functional tests are in StableHLO quantizer directory.
      self.assertTrue(self._contains_op(output_graphdef, 'XlaCallModule'))
    else:
      self.assertTrue(self._contains_quantized_function_call(output_graphdef))
    self.assertFalse(self._contains_op(output_graphdef, 'FusedBatchNormV3'))

  @parameterized.named_parameters(
      ('to_tf_with_int32_input_type', dtypes.int32, quant_opts_pb2.TF),
      ('to_xla_with_int32_input_type', dtypes.int32, quant_opts_pb2.XLA),
      ('to_xla_with_int64_input_type', dtypes.int64, quant_opts_pb2.XLA),
      (
          'to_uq_with_int32_input_type',
          dtypes.int32,
          quant_opts_pb2.UNIFORM_QUANTIZED,
      ),
  )
  @test_util.run_v2_only
  def test_gather_and_conv_model(
      self, input_type: dtypes, target_opset: quant_opts_pb2.OpSet
  ):
    # [FUNCTION DOC]
    # Parameterized PTQ test (eager only): gather followed by convolution with large filter.
    # Parameters: input_type (int32 or int64 index type) and target_opset (TF, XLA, uniform quantized).
    # Builds the model, quantizes with a generated representative dataset, then asserts the signature, size
    # ratio (above 0.68 for uniform quantized, below one third otherwise), the proper conv op, and that
    # quantized outputs are within a very loose tolerance of the float outputs.
    model = self._create_simple_gather_and_conv_model(
        input_type, filter_shape=(2, 3, 3, 1024)
    )
    saved_model_save.save(model, self._input_saved_model_path)

    data_gen = self._create_data_generator(
        input_key='input_tensor',
        shape=[500],
        minval=0,
        maxval=64,
        dtype=input_type,
    )

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def

    if target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      self.assertGreater(
          testing.get_size_ratio(
              self._output_saved_model_path, self._input_saved_model_path
          ),
          0.68,
      )
      self.assertTrue(
          self._contains_op(output_graphdef, 'UniformQuantizedConvolution')
      )
    else:
      # Due to other meta data, the compression is not exactly 1/4.
      self.assertLess(
          testing.get_size_ratio(
              self._output_saved_model_path, self._input_saved_model_path
          ),
          1 / 3,
      )
      if target_opset == quant_opts_pb2.XLA:
        self.assertTrue(self._contains_op(output_graphdef, 'XlaConvV2'))
      else:
        self.assertTrue(self._contains_quantized_function_call(output_graphdef))

    test_data = np.random.uniform(low=0, high=64, size=(32)).astype(
        input_type.as_numpy_dtype
    )
    original_outputs = model.model(test_data)['output']
    quantized_output = converted_model.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(test_data)
    )['output']
    self.assertAllClose(original_outputs, quantized_output, atol=442.7)

  @test_util.run_v2_only
  def test_while_op_model(
      self,
  ):
    # [FUNCTION DOC]
    # Test (eager only): PTQ of a model containing a while loop quantized to XLA.
    # Builds a while-loop model, quantizes with two representative samples, asserts the signature, that
    # the convolution outside the loop became an int8 XLA convolution, and that the convolution inside the
    # loop body is still a float convolution (known limitation noted in the test).
    input_shape = (1, 5, 5, 32)
    model = self._create_while_model(input_shape)
    saved_model_save.save(model, self._input_saved_model_path)

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.XLA,
    )

    def data_gen() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: representative data generator for the while loop test. Yields two samples, each mapping
      # input_tensor to random float32 values in 0 to 150 with the model input shape.
      for _ in range(2):
        yield {
            'input_tensor': ops.convert_to_tensor(
                np.random.uniform(low=0, high=150, size=input_shape).astype(
                    'f4'
                )
            ),
        }

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen(),
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    loader = saved_model_loader.SavedModelLoader(self._output_saved_model_path)
    output_graphdef = loader.get_meta_graph_def_from_tags(tags).graph_def

    # Convolution ouside the while op is quantized.
    self.assertTrue(
        self._contains_op(
            output_graphdef,
            op_name='XlaConvV2',
            attr_name='RhsT',
            attr_val=attr_value_pb2.AttrValue(type=types_pb2.DT_INT8),
        )
    )
    # TODO: b/294783597 - [Converter][TF-Quantizer] Support quantization for the
    # ops in the while op body for both SRQ and WO
    # Convolution inside the while op is not quantized.
    self.assertTrue(
        self._contains_op(
            output_graphdef,
            op_name='Conv2D',
            attr_name='T',
            attr_val=attr_value_pb2.AttrValue(type=types_pb2.DT_FLOAT),
        )
    )

  # Check only the most simple case and the most complicated cases.
  @parameterized.named_parameters(
      {
          'testcase_name': 'none',
          'activation_fn': None,
          'has_bias': False,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'relu',
          'activation_fn': nn_ops.relu,
          'has_bias': False,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'relu6',
          'activation_fn': nn_ops.relu6,
          'has_bias': False,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'bn',
          'activation_fn': None,
          'has_bias': False,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'with_bias',
          'activation_fn': None,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'with_bias_and_relu6',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'with_bias_and_bn_and_relu6',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.TF,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'with_bias_and_relu6_to_xla',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'with_bias_and_relu6_to_xla_per_channel',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': True,
      },
      {
          'testcase_name': 'with_bias_and_bn_and_relu6_to_xla',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'with_bias_and_bn_and_relu6_to_xla_per_channel',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': True,
      },
      {
          'testcase_name': 'with_bias_and_relu6_to_xla_dynamic',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': True,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'with_bias_and_relu6_to_xla_dynamic_per_channel',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': True,
          'enable_per_channel_quantization': True,
      },
      {
          'testcase_name': 'with_bias_and_bn_and_relu6_to_xla_dynamic',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': True,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': (
              'with_bias_and_bn_and_relu6_to_xla_dynamic_per_channel'
          ),
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.XLA,
          'input_shape_dynamic': True,
          'enable_per_channel_quantization': True,
      },
      {
          'testcase_name': 'with_bias_and_relu6_to_uq',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.UNIFORM_QUANTIZED,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'with_bias_and_bn_and_relu6_to_uq',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.UNIFORM_QUANTIZED,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': False,
      },
      {
          'testcase_name': 'with_bias_and_relu6_to_uq_per_channel',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.UNIFORM_QUANTIZED,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': True,
      },
      {
          'testcase_name': 'with_bias_and_bn_and_relu6_to_uq_per_channel',
          'activation_fn': nn_ops.relu6,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.UNIFORM_QUANTIZED,
          'input_shape_dynamic': False,
          'enable_per_channel_quantization': True,
      },
  )
  @test_util.run_in_graph_and_eager_modes
  def test_depthwise_conv_ptq_model(
      self,
      activation_fn: Optional[ops.Operation],
      has_bias: bool,
      has_batch_norm: bool,
      target_opset: quant_opts_pb2.OpSet,
      input_shape_dynamic: bool,
      enable_per_channel_quantization: bool,
  ):
    # [FUNCTION DOC]
    # KNOWN ISSUE: reuses the exhausted per-channel op names generator for the second count.
    # Parameterized PTQ test for depthwise convolution.
    # Parameters: activation_fn, has_bias, has_batch_norm, target_opset, input_shape_dynamic,
    # enable_per_channel_quantization. Builds a depthwise conv model, quantizes with eight random samples,
    # and asserts signature and op expectations per op set (XLA keeps the depthwise op and has per-channel
    # constants of channel times multiplier size; uniform quantized converts to quantized conv with the
    # right axis attributes; otherwise a quantized function call) and no fused batch norm.
    input_shape = [None, None, None, 3] if input_shape_dynamic else [1, 3, 4, 3]
    filter_shape = [2, 3, 3, 1]
    model = self._create_depthwise_conv2d_model(
        input_shape, filter_shape, has_bias, has_batch_norm, activation_fn
    )
    saved_model_save.save(model, self._input_saved_model_path)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: representative data generator for the depthwise test. Yields eight samples mapping
      # input_tensor to random float32 values in 0 to 150 with shape 1,3,4,3.
      for _ in range(8):
        yield {
            'input_tensor': ops.convert_to_tensor(
                np.random.uniform(low=0, high=150, size=(1, 3, 4, 3)).astype(
                    'f4'
                )
            ),
        }

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
        enable_per_channel_quantization=enable_per_channel_quantization,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen(),
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    if target_opset == quant_opts_pb2.XLA:
      self.assertTrue(
          self._contains_op(output_graphdef, 'DepthwiseConv2dNative')
      )
      if enable_per_channel_quantization:
        per_channel_size_attr = attr_value_pb2.AttrValue(
            list=attr_value_pb2.AttrValue.ListValue(
                shape=[
                    tensor_shape_pb2.TensorShapeProto(
                        dim=[
                            tensor_shape_pb2.TensorShapeProto.Dim(
                                size=filter_shape[-1] * filter_shape[2]
                            )
                        ]
                    )
                ]
            )
        )
        self.assertTrue(
            self._contains_op(
                output_graphdef,
                'Const',
                '_output_shapes',
                per_channel_size_attr,
            )
        )
    elif target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      self.assertTrue(
          self._contains_op(output_graphdef, 'UniformQuantizedConvolution')
      )
      if enable_per_channel_quantization:
        quantized_axis = 3
        quantized_dim_size_attr = attr_value_pb2.AttrValue(
            list=attr_value_pb2.AttrValue.ListValue(
                shape=[
                    tensor_shape_pb2.TensorShapeProto(
                        dim=[
                            tensor_shape_pb2.TensorShapeProto.Dim(
                                # Depthwise conv is reshaped to [H,W,1,CxM].
                                size=filter_shape[quantized_axis]
                                * filter_shape[2]
                            )
                        ]
                    )
                ]
            )
        )
      else:
        quantized_axis = -1
        # Empty dimension. Per-tensor quantization has singular channel.
        quantized_dim_size_attr = attr_value_pb2.AttrValue(
            list=attr_value_pb2.AttrValue.ListValue(
                shape=[tensor_shape_pb2.TensorShapeProto()]
            )
        )
      quantized_axis_attr = attr_value_pb2.AttrValue(i=quantized_axis)
      self.assertEqual(
          self._count_ops(
              output_graphdef,
              _PER_CHANNEL_QUANTIZED_OPS,
              'rhs_quantization_axis',
              quantized_axis_attr,
          ),
          self._count_ops(output_graphdef, _PER_CHANNEL_QUANTIZED_OPS),
      )
      self.assertEqual(
          self._count_ops(
              output_graphdef,
              _PER_CHANNEL_OP_NAMES,
              '_output_shapes',
              quantized_dim_size_attr,
              get_op_name=True,
          ),
          self._count_ops(
              output_graphdef,
              _PER_CHANNEL_OP_NAMES,
              get_op_name=True,
          ),
      )
      self.assertFalse(self._contains_op(output_graphdef, 'Conv2D'))
    else:
      self.assertTrue(self._contains_quantized_function_call(output_graphdef))
    self.assertFalse(self._contains_op(output_graphdef, 'FusedBatchNormV3'))

  @parameterized.parameters(
      *testing.parameter_combinations([
          {
              'activation_fn': [None, nn_ops.relu, nn_ops.relu6],
              'has_bias': [True, False],
              'batch_sizes': [([], []), ([10], [10]), ([2, 3], [2, 3])],
              'target_opset': [quant_opts_pb2.XLA],
          },
          # Test broadcastable batch sizes.
          {
              'activation_fn': [None],
              'has_bias': [True],
              'batch_sizes': [
                  ([2], []),
                  ([], [2]),
                  ([1], [2]),
                  ([None], []),
              ],
              'target_opset': [quant_opts_pb2.XLA],
          },
          {
              'activation_fn': [None, nn_ops.relu, nn_ops.relu6],
              'has_bias': [True, False],
              'batch_sizes': [([], []), ([10], [10]), ([2, 3], [2, 3])],
              'target_opset': [quant_opts_pb2.UNIFORM_QUANTIZED],
          },
      ])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_matmul_ptq_model(
      self,
      activation_fn: Optional[ops.Operation],
      has_bias: bool,
      batch_sizes: Sequence[int],
      target_opset: quant_opts_pb2.OpSet,
  ):
    # [FUNCTION DOC]
    # Parameterized PTQ test for matmul (including batched shapes).
    # Parameters: activation_fn, has_bias, batch_sizes (pair of lhs and rhs batch dimension lists, possibly
    # with unknown dimensions) and target_opset. Builds a matmul model, calibrates with 500 seeded random
    # samples, quantizes with static range int8, asserts signature and for XLA a dot op, for uniform
    # quantized a quantized dot op named like the matmul; and that outputs are close to the float model
    # (tolerance 0.25 for uniform quantized because of requantization, 0.13 otherwise).
    lhs_batch_size, rhs_batch_size = batch_sizes
    input_shape = (*lhs_batch_size, 1, 1024)
    filter_shape = (*rhs_batch_size, 1024, 3)
    static_input_shape = [dim if dim is not None else 2 for dim in input_shape]
    model = self._create_matmul_model(
        input_shape,
        filter_shape,
        self._input_saved_model_path,
        has_bias,
        activation_fn,
    )
    rng = np.random.default_rng(seed=1234)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: representative data generator for the matmul PTQ test. Yields 500 samples mapping
      # input_tensor to uniform random float32 values in 0 to 1 using the test's seeded generator.
      for _ in range(500):
        yield {
            'input_tensor': rng.uniform(
                low=0.0, high=1.0, size=static_input_shape
            ).astype(np.float32)
        }

    tags = {tag_constants.SERVING}

    input_data = ops.convert_to_tensor(
        rng.uniform(low=0.0, high=1.0, size=static_input_shape).astype(
            np.float32
        )
    )
    expected_outputs = model.matmul(input_data)

    # Check the converted model in the target opset.
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen(),
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )
    loader = saved_model_loader.SavedModelLoader(self._output_saved_model_path)
    output_graphdef = loader.get_meta_graph_def_from_tags(tags).graph_def
    if target_opset == quant_opts_pb2.XLA:
      self.assertTrue(
          self._contains_op(
              output_graphdef, 'XlaDotV2', node_name='sample/matmul.*'
          )
      )
    elif target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      self.assertTrue(
          self._contains_op(
              output_graphdef,
              'UniformQuantizedDot',
              node_name='sample/matmul.*',
          )
      )

    new_outputs = converted_model.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(input_data)
    )
    # The difference between TF and target path is expected to be small.
    # The atol value is arbitrary.
    # Currently, Uniform Quantized Opset are producing non-optimal graphs:
    # unnecessary requantization followed by dequantization, so the error will
    # be higher.
    if target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      self.assertAllClose(new_outputs, expected_outputs, atol=0.25)
    else:
      self.assertAllClose(new_outputs, expected_outputs, atol=0.13)

  @parameterized.named_parameters(
      {
          'testcase_name': 'with_biasadd',
          'input_shape': (32, 16),
          'filter_shape': (16, 8),
          'bias_size': 4,
          'use_biasadd': True,
          'activation_fn': nn_ops.relu,
      },
      {
          'testcase_name': 'with_addv2',
          'input_shape': (32, 16),
          'filter_shape': (16, 8),
          'bias_size': 4,
          'use_biasadd': False,
          'activation_fn': nn_ops.relu,
      },
  )
  def test_matmul_with_reshape_and_bias_ptq_model(
      self, input_shape, filter_shape, bias_size, activation_fn, use_biasadd
  ):
    # [FUNCTION DOC]
    # Parameterized PTQ test: matmul with reshape and bias (either bias add or add op) quantized to XLA.
    # Parameters: input_shape, filter_shape, bias_size, activation_fn, use_biasadd. Calibrates with five
    # seeded random samples, quantizes, and asserts outputs are close to the float model (tolerance 0.05).
    model = self._create_matmul_model(
        input_shape,
        filter_shape,
        self._input_saved_model_path,
        True,
        activation_fn,
        bias_size,
        use_biasadd,
    )

    rng = np.random.default_rng(seed=1234)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: representative data generator for this test. Yields five samples mapping input_tensor to
      # random float32 values in 0 to 1 with the input shape.
      for _ in range(5):
        yield {
            'input_tensor': rng.uniform(
                low=0.0, high=1.0, size=input_shape
            ).astype(np.float32)
        }

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.OpSet.XLA,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen(),
    )

    input_data = ops.convert_to_tensor(
        rng.uniform(low=0.0, high=1.0, size=input_shape).astype(np.float32)
    )
    expected_outputs = model.matmul(input_data)

    got_outputs = converted_model.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(input_data)
    )

    self.assertAllClose(expected_outputs, got_outputs, atol=0.05)

  @parameterized.parameters(
      ('abc,cde->abde', quant_opts_pb2.XLA),
      ('abc,dce->abde', quant_opts_pb2.XLA),
  )
  def test_einsum_ptq_model(
      self,
      equation: str,
      target_opset: quant_opts_pb2.OpSet,
  ):
    # [FUNCTION DOC]
    # Parameterized PTQ test for einsum with bias and relu.
    # Parameters: equation (two strings) and target_opset (XLA). Builds an einsum model saved with the
    # default signature, quantizes first with the TF op set (asserts quantized function call, output
    # tolerance 0.097) and then with the target op set (asserts a dot op for XLA and that outputs are
    # close to both the TF-path and float outputs).
    _, y_shape, bias_shape, x_signature, y_signature = (
        self._prepare_sample_einsum_datashapes(equation, use_bias=True)
    )

    model = self._create_einsum_model(
        equation,
        y_shape,
        x_signature,
        y_signature,
        bias_shape,
        activation_fn=nn_ops.relu,
    )

    signatures = {
        'serving_default': model.einsum_with_kernel.get_concrete_function(),
    }

    saved_model_save.save(model, self._input_saved_model_path, signatures)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: representative data generator for the einsum PTQ test. Yields four samples mapping x to
      # random float32 values in 0 to 1 with the einsum input shape.
      for _ in range(4):
        yield {
            'x': ops.convert_to_tensor(
                np.random.uniform(low=0.0, high=1.0, size=x_signature).astype(
                    'f4'
                )
            ),
        }

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.TF,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen(),
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

    input_data = ops.convert_to_tensor(
        np.random.uniform(low=0.0, high=1.0, size=x_signature).astype('f4')
    )
    expected_outputs = model.einsum_with_kernel(input_data)
    got_outputs = converted_model.signatures['serving_default'](
        x=ops.convert_to_tensor(input_data)
    )
    self.assertAllClose(expected_outputs, got_outputs, atol=0.097)

    # Check the converted model in the target opset.
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path_2,
        quantization_options,
        representative_dataset=data_gen(),
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path_2
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    if target_opset == quant_opts_pb2.XLA:
      self.assertTrue(self._contains_op(output_graphdef, 'XlaDotV2'))

    new_outputs = converted_model.signatures['serving_default'](
        x=ops.convert_to_tensor(input_data)
    )
    # The difference between TF and target path is expected to be small.
    self.assertAllClose(new_outputs, got_outputs, atol=0.097)
    self.assertAllClose(new_outputs, expected_outputs, atol=0.057)

  def test_reuse_calibration_data(self):
    # [FUNCTION DOC]
    # Test: calibration statistics saved in a directory are reused on the second quantization run.
    # Saves a gather and conv model, quantizes to XLA with min-max calibration, forced graph-mode
    # calibration and a calibration data directory, capturing INFO logs (verbosity raised and restored).
    # First run: asserts the graph-mode calibration log message appears and an XLA conv op exists. Second
    # run, overwriting the output: asserts the message does not appear. Finally asserts both models give the
    # same outputs. Side effects: temporary calibration directory, global logging verbosity changes.
    model = self._create_simple_gather_and_conv_model(
        dtypes.int32, filter_shape=(2, 3, 3, 1024)
    )
    saved_model_save.save(model, self._input_saved_model_path)

    data_gen = self._create_data_generator(
        input_key='input_tensor',
        shape=[50],
        minval=0,
        maxval=64,
        dtype=dtypes.int32,
    )

    tags = {tag_constants.SERVING}

    calibration_data_dir = self.create_tempdir('calibration_data').full_path
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.XLA,
        force_graph_mode_calibration=True,
        calibration_options=stablehlo_quant_config_pb2.CalibrationOptions(
            calibration_method=_CalibrationMethod.CALIBRATION_METHOD_MIN_MAX,
            calibration_data_dir=calibration_data_dir,
        ),
    )

    # Run quantization the first time, calibration is expected to be run.
    with self.assertLogs(level='INFO') as info_logs:
      # Save the logger verbosity.
      prev_log_level = logging.get_verbosity()
      logging.set_verbosity(logging.INFO)
      try:
        converted_model1 = quantize_model.quantize(
            self._input_saved_model_path,
            self._output_saved_model_path,
            quantization_options,
            representative_dataset=data_gen,
        )
      finally:
        # Restore the logger verbosity.
        logging.set_verbosity(prev_log_level)

      self.assertNotEmpty(info_logs.records)
      self.assertTrue(
          self._any_log_contains(
              'Calibration step is executed in graph mode.',
              info_logs.records,
          )
      )
      self.assertIsNotNone(converted_model1)
      self.assertCountEqual(
          converted_model1.signatures._signatures.keys(), {'serving_default'}
      )

      output_loader = saved_model_loader.SavedModelLoader(
          self._output_saved_model_path
      )
      output_graphdef = output_loader.get_meta_graph_def_from_tags(
          tags
      ).graph_def
      self.assertTrue(self._contains_op(output_graphdef, 'XlaConvV2'))

    # Run quantization the first time, calibration is expected to be skipped.
    with self.assertLogs(level='INFO') as info_logs:
      # Save the logger verbosity.
      prev_log_level = logging.get_verbosity()
      logging.set_verbosity(logging.INFO)
      try:
        converted_model2 = quantize_model.quantize(
            self._input_saved_model_path,
            self._output_saved_model_path,
            quantization_options,
            representative_dataset=data_gen,
            overwrite_output_directory=True,
        )
      finally:
        # Restore the logger verbosity.
        logging.set_verbosity(prev_log_level)

      self.assertNotEmpty(info_logs.records)
      self.assertFalse(
          self._any_log_contains(
              'Calibration step is executed in graph mode.',
              info_logs.records,
          )
      )
      self.assertIsNotNone(converted_model2)
      self.assertCountEqual(
          converted_model2.signatures._signatures.keys(), {'serving_default'}
      )

      # Expect two models to produce the same results.
      test_data = ops.convert_to_tensor(
          np.random.uniform(low=0, high=64, size=(32)).astype(
              dtypes.int32.as_numpy_dtype
          )
      )
      new_outputs_1 = converted_model1.signatures['serving_default'](
          input_tensor=test_data
      )['output']
      new_outputs_2 = converted_model2.signatures['serving_default'](
          input_tensor=test_data
      )['output']
      self.assertAllClose(new_outputs_1, new_outputs_2)

  @test_util.run_in_graph_and_eager_modes
  def test_function_alias_preserved(self):
    # [FUNCTION DOC]
    # Test: function aliases of the original saved model survive PTQ to XLA.
    # Saves a conv model with a function alias for its conv function, quantizes with two seeded samples,
    # and asserts the output meta graph still has exactly that alias and that the aliased function
    # contains an XLA conv op.
    model = self._create_conv2d_model(
        input_shape=(1, 3, 4, 3), filter_shape=(2, 3, 3, 2)
    )

    signatures = {
        'serving_default': model.conv.get_concrete_function(),
    }
    save_opts = save_options.SaveOptions(
        function_aliases={'conv_func': model.conv}
    )

    saved_model_save.save(
        model, self._input_saved_model_path, signatures, save_opts
    )

    def data_gen() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: representative data generator for the function alias test. Yields two samples mapping
      # input_tensor to random float32 values in 0 to 150 with shape 1,3,4,3 from a seeded generator.
      rng = np.random.default_rng(seed=123)
      for _ in range(2):
        yield {
            'input_tensor': rng.uniform(
                low=0, high=150, size=(1, 3, 4, 3)
            ).astype(np.float32),
        }

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.OpSet.XLA,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen(),
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    # Test whether the aliased function exists.
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )

    # Confirm that the function alias is preserved.
    meta_graph_def = output_loader.get_meta_graph_def_from_tags(tags)
    function_aliases = meta_graph_def.meta_info_def.function_aliases
    self.assertNotEmpty(function_aliases)
    self.assertCountEqual(function_aliases.values(), {'conv_func'})

    # Test that the aliased function contains a quantized op.
    for func_name, alias in function_aliases.items():
      if alias == 'conv_func':
        for func in meta_graph_def.graph_def.library.function:
          if func.signature.name == func_name:
            self.assertTrue(
                self._contains_op_with_name_and_attribute(
                    func.node_def,
                    op_name='XlaConvV2',
                    attr_name='',
                    attr_val=None,
                )
            )

  @test_util.run_in_graph_and_eager_modes
  def test_function_alias_preserved_in_qat(self):
    # [FUNCTION DOC]
    # Test: function aliases survive QAT quantization of an einsum model to XLA.
    # Saves a QAT einsum model with an alias for its einsum function, quantizes without a representative
    # dataset, and asserts the alias remains and the aliased function contains an XLA dot op.
    _, y_shape, _, x_signature, y_signature = (
        self._prepare_sample_einsum_datashapes('ab,bc->ac')
    )
    model = self._create_einsum_model(
        'ab,bc->ac', y_shape, x_signature, y_signature, is_qat_model=True
    )

    signatures = {
        'serving_default': model.einsum_with_kernel.get_concrete_function(),
    }
    save_opts = save_options.SaveOptions(
        function_aliases={'einsum_with_kernel': model.einsum_with_kernel}
    )

    saved_model_save.save(
        model, self._input_saved_model_path, signatures, save_opts
    )

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.OpSet.XLA,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    # Test whether the aliased function exists.
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )

    # Confirm that the function alias is preserved.
    meta_graph_def = output_loader.get_meta_graph_def_from_tags(tags)
    function_aliases = meta_graph_def.meta_info_def.function_aliases
    self.assertNotEmpty(function_aliases)
    self.assertCountEqual(function_aliases.values(), {'einsum_with_kernel'})

    # Test that the aliased function contains a quantized op.
    for func_name, alias in function_aliases.items():
      if alias == 'einsum_with_kernel':
        for func in meta_graph_def.graph_def.library.function:
          if func.signature.name == func_name:
            self.assertTrue(
                self._contains_op_with_name_and_attribute(
                    func.node_def,
                    op_name='XlaDotV2',
                    attr_name='',
                    attr_val=None,
                )
            )

  def test_matmul_ptq_model_with_unfreeze_constants(self):
    # [FUNCTION DOC]
    # Test: with freeze_all_variables disabled, large constants are unfrozen into variables.
    # Creates a matmul model with a large weight (over the 64 KiB constant threshold), quantizes with
    # static range int8 and the TF op set, loads the output in a session, asserts a quantized function call,
    # exactly one variable in the graph, and that its name and shape match the checkpoint listing.
    # Uses large weight to exceed the constant size threshold of 64KiB
    # (specified by `kDefaultConstantSizeThresholdInBytes`) for unfreezing.
    self._create_matmul_model(
        input_shape=(1, 20),
        weight_shape=(20, 4096),
        saved_model_path=self._input_saved_model_path,
    )

    repr_ds = self._create_data_generator(
        input_key='input_tensor', shape=(1, 20), num_examples=2
    )

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.TF,
        freeze_all_variables=False,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=repr_ds,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    # Test that the quantized model successfully loads without error.
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    with session.Session(graph=ops.Graph()) as sess:
      output_meta_graph_def = output_loader.load(sess, tags)

    # Confirms that quantization is applied to the model.
    output_graphdef = output_meta_graph_def.graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

    # Tests that there are variables in the model.
    variable_node_defs = _find_variables(output_graphdef)
    self.assertLen(variable_node_defs, 1)

    # Reads the variables from the checkpoint file and matches with the
    # variables found in the graph.
    checkpoint_path = os.path.join(
        self._output_saved_model_path, 'variables', 'variables'
    )
    var_name_and_shapes = checkpoint_utils.list_variables(checkpoint_path)

    # Checks that each variable's name and shape match.
    self.assertEqual(len(variable_node_defs), len(var_name_and_shapes))
    for var_name, shape in var_name_and_shapes:
      self.assertIn(var_name, variable_node_defs)
      self.assertEqual(
          shape,
          tensor_shape.TensorShape(
              variable_node_defs[var_name].attr['shape'].shape
          ),
      )

  @parameterized.named_parameters(
      ('use_constant_with_int32_input', dtypes.int32, False, True),
      ('use_variable_with_int32_input', dtypes.int32, True, True),
      ('use_constant_with_int64_input', dtypes.int64, False, True),
      ('use_variable_with_int64_input', dtypes.int64, True, True),
      ('small_gather_use_constant', dtypes.int32, False, False),
      ('small_gather_use_variable', dtypes.int32, True, False),
  )
  @test_util.run_v2_only
  def test_gather_model(
      self, input_type, use_variable, expect_quantized_gather
  ):
    # [FUNCTION DOC]
    # Parameterized PTQ test (eager only): gather quantization for constant and variable embeddings.
    # Parameters: input_type, use_variable, expect_quantized_gather (small gathers are opted out through
    # the minimum number of elements option, 1024 versus 8192). Quantizes to XLA and asserts the signature
    # and that the model size shrinks below one third if gather is quantized, else stays above two thirds.
    model = self._create_gather_model(input_type, use_variable)

    saved_model_save.save(model, self._input_saved_model_path)

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.XLA,
        # Gather op is opt-outed if the size is smaller than the threshold.
        min_num_elements_for_weights=1024 if expect_quantized_gather else 8192,
    )

    data_gen = self._create_data_generator(
        input_key='input_tensor',
        shape=[6],
        minval=0,
        maxval=10,
        dtype=input_type,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    if expect_quantized_gather:
      # Due to other meta data, the compression is not exactly 1/4.
      self.assertLess(
          testing.get_size_ratio(
              self._output_saved_model_path, self._input_saved_model_path
          ),
          1 / 3,
      )
    else:
      self.assertGreater(
          testing.get_size_ratio(
              self._output_saved_model_path, self._input_saved_model_path
          ),
          2 / 3,
      )

  @test_util.run_in_graph_and_eager_modes
  def test_model_ptq_use_representative_samples_list(self):
    # [FUNCTION DOC]
    # Test: representative dataset given as a list of mappings with tensor values.
    # Creates a matmul model, quantizes with eight random tensor samples and asserts the signature and a
    # quantized function call.
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=self._input_saved_model_path,
    )

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.TF,
    )

    representative_dataset: repr_dataset.RepresentativeDataset = [
        {
            'input_tensor': random_ops.random_uniform(shape=(1, 1024)),
        }
        for _ in range(8)
    ]

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options=quantization_options,
        representative_dataset=representative_dataset,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_model_ptq_use_ndarray_representative_dataset(self):
    # [FUNCTION DOC]
    # Test: representative dataset given as a list of mappings with numpy arrays.
    # Creates a matmul model, quantizes with four seeded numpy samples and asserts the signature and a
    # quantized function call.
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=self._input_saved_model_path,
    )

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.TF,
    )

    # Use np.ndarrays instead of tf.Tensors for the representative dataset.
    rng = np.random.default_rng(seed=1234)
    representative_dataset = [
        {'input_tensor': rng.uniform(size=(1, 1024)).astype(np.float32)}
        for _ in range(4)
    ]

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options=quantization_options,
        representative_dataset=representative_dataset,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_model_ptq_use_python_list_representative_dataset(self):
    # [FUNCTION DOC]
    # Test: representative dataset given as nested Python lists.
    # Creates a matmul model, quantizes with four samples of plain lists and asserts the signature and a
    # quantized function call.
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=self._input_saved_model_path,
    )

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.TF,
    )

    # Use plain python lists as representative samples.
    representative_dataset = [
        {
            'input_tensor': [[i * 0.1 for i in range(1024)]],
        }
        for _ in range(4)
    ]

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options=quantization_options,
        representative_dataset=representative_dataset,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_model_ptq_use_representative_samples_file(self):
    # [FUNCTION DOC]
    # Test: representative dataset loaded from TFRecord files named in the options.
    # Saves samples to a record file via the saver, then asserts that passing both the dataset argument
    # and the representative datasets option raises a value error; and that using only the option works,
    # giving the signature and a quantized function call.
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=self._input_saved_model_path,
    )

    rng = np.random.default_rng(seed=1234)
    representative_dataset: repr_dataset.RepresentativeDataset = [
        {'input_tensor': rng.uniform(size=(1, 1024)).astype(np.float32)}
        for _ in range(4)
    ]
    dataset_file_map = repr_dataset.TfRecordRepresentativeDatasetSaver(
        {'serving_default': os.path.join(self._input_saved_model_path, 'repr')}
    ).save({'serving_default': representative_dataset})

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.TF,
        representative_datasets=dataset_file_map,
    )

    with self.assertRaisesRegex(
        ValueError,
        'Do not specify both the `representative_dataset` argument and the'
        ' `representative_datasets` field in `QuantizationOptions`',
    ):
      quantize_model.quantize(
          self._input_saved_model_path,
          self._output_saved_model_path,
          quantization_options=quantization_options,
          representative_dataset=representative_dataset,
      )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options=quantization_options,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_model_ptq_call_twice(self):
    # [FUNCTION DOC]
    # Test: the quantizer can be called twice on the same input model.
    # Quantizes a matmul model twice into two different output directories with the same options and
    # dataset, asserting the signature and a quantized function call each time.
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=self._input_saved_model_path,
    )

    tags = {tag_constants.SERVING}
    signature_def_keys = [signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_def_keys,
        op_set=quant_opts_pb2.TF,
    )

    representative_dataset: repr_dataset.RepresentativeDataset = [
        {
            'input_tensor': random_ops.random_uniform(shape=(1, 1024)),
        }
        for _ in range(8)
    ]

    # Test the first run.
    converted_model_1 = quantize_model.quantize(
        self._input_saved_model_path,
        output_directory=self._output_saved_model_path,
        quantization_options=quantization_options,
        representative_dataset=representative_dataset,
    )

    self.assertIsNotNone(converted_model_1)
    self.assertCountEqual(
        converted_model_1.signatures._signatures.keys(), signature_def_keys
    )
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

    # Test the second run on the same model.
    converted_model_2 = quantize_model.quantize(
        self._input_saved_model_path,
        output_directory=self._output_saved_model_path_2,
        quantization_options=quantization_options,
        representative_dataset=representative_dataset,
    )

    self.assertIsNotNone(converted_model_2)
    self.assertCountEqual(
        converted_model_2.signatures._signatures.keys(), signature_def_keys
    )
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path_2
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  def test_model_ptq_preserving_assets_extra(self):
    # [FUNCTION DOC]
    # Test: extra assets directory contents are preserved in the quantized model.
    # Creates a matmul model, writes a file under the assets.extra directory, quantizes, and asserts the
    # file exists in the output. Side effect: writes a file in the temporary input model directory.
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=self._input_saved_model_path,
    )
    asset_filename = 'assets.extra/tf_serving_warmup_requests'
    file_io.create_dir_v2(
        os.path.join(self._input_saved_model_path, 'assets.extra')
    )
    file_io.write_string_to_file(
        filename=os.path.join(self._input_saved_model_path, asset_filename),
        file_content='Test content',
    )

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.TF,
    )

    # Use plain python lists as representative samples.
    representative_dataset = [
        {
            'input_tensor': [[i * 0.1 for i in range(1024)]],
        }
        for _ in range(4)
    ]

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options=quantization_options,
        representative_dataset=representative_dataset,
    )
    self.assertIsNotNone(converted_model)
    # Check if the assets.extra file exists in the output model.
    self.assertTrue(
        file_io.file_exists_v2(
            os.path.join(self._output_saved_model_path, asset_filename)
        )
    )

  # tf.data.Dataset is as an Iterable (thus can be used as representative
  # dataset) only in TF2 (eager mode).
  @test_util.run_v2_only
  def test_model_ptq_use_tf_dataset_for_representative_dataset(self):
    # [FUNCTION DOC]
    # Test (eager only): a tf.data dataset is accepted as the representative dataset.
    # Builds a dataset from eight generated samples, quantizes the matmul model and asserts the signature
    # and a quantized function call.
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=self._input_saved_model_path,
    )

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.TF,
    )

    representative_samples = [
        {
            'input_tensor': random_ops.random_uniform(shape=(1, 1024)),
        }
        for _ in range(8)
    ]

    # Construct a tf.data.Dataset from the representative samples.
    representative_dataset = dataset_ops.DatasetV2.from_generator(
        lambda: representative_samples,
        output_signature={
            'input_tensor': tensor_spec.TensorSpec(
                shape=(1, 1024), dtype=dtypes.float32
            ),
        },
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options=quantization_options,
        representative_dataset=representative_dataset,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_model_ptq_no_representative_sample_not_quantized(self):
    # [FUNCTION DOC]
    # Test: an empty representative dataset leaves the model unquantized.
    # Quantizes a matmul model with an empty list as dataset, asserts a model with the signature is
    # produced and the graph has no quantized function call.
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=self._input_saved_model_path,
    )

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        # Put no sample into the representative dataset to make calibration
        # impossible.
        representative_dataset=[],
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    # Model is not quantized because there was no sample data for calibration.
    self.assertFalse(self._contains_quantized_function_call(output_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_model_ptq_with_uncalibrated_subgraph(self):
    # [FUNCTION DOC]
    # Test: when only one branch of a conditional is exercised, only that branch is quantized.
    # Defines a local branching model, calibrates with inputs whose sum never exceeds the branch threshold
    # (so only the false branch runs), quantizes, and asserts the signature, a quantized function call
    # overall, that the false-branch function has a quantized call and the true-branch function does not.
    class IfModel(module.Module):
      # [CLASS DOC]
      # Local fixture module: two sets of random weight and bias, and a traced function that picks one
      # matmul plus bias add depending on whether the input sum is greater than ten.
      """A model that contains a branching op."""

      def __init__(self):
        # [FUNCTION DOC]
        # Constructor. Creates two random 4 by 3 weights and two random bias vectors of length three
        # (float32, range minus 1 to 1). Returns nothing.
        self.filters_0 = np.random.uniform(
            low=-1.0, high=1.0, size=(4, 3)
        ).astype('f4')
        self.bias_0 = np.random.uniform(low=-1.0, high=1.0, size=(3,)).astype(
            'f4'
        )

        self.filters_1 = np.random.uniform(
            low=-1.0, high=1.0, size=(4, 3)
        ).astype('f4')
        self.bias_1 = np.random.uniform(low=-1.0, high=1.0, size=(3,)).astype(
            'f4'
        )

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(shape=[1, 4], dtype=dtypes.float32)
          ]
      )
      def model_fn(self, x: core.Tensor) -> Mapping[str, core.Tensor]:
        # [FUNCTION DOC]
        # Traced function (input shape 1 by 4). If the sum of the input exceeds 10 it applies the first
        # weight and bias, otherwise the second pair. Returns a mapping with key output. Input: x.
        """Runs the input tensor to a branched operations.

        The graph is branched by a condition whether the sum of elements of `x`
        is greater than 10.

        Args:
          x: Input tensor.

        Returns:
          A map of: output key -> output result.
        """
        if math_ops.reduce_sum(x) > 10.0:
          out = math_ops.matmul(x, self.filters_0)
          out = nn_ops.bias_add(out, self.bias_0)
          return {'output': out}

        out = math_ops.matmul(x, self.filters_1)
        out = nn_ops.bias_add(out, self.bias_1)
        return {'output': out}

    model = IfModel()
    saved_model_save.save(model, self._input_saved_model_path)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: representative data generator. Yields eight samples mapping x to random float32 values in
      # 0 to 1 with shape 1 by 4, so the true branch is never taken.
      for _ in range(8):
        yield {
            'x': ops.convert_to_tensor(
                np.random.uniform(low=0.0, high=1.0, size=(1, 4)).astype('f4')
            ),
        }

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.TF,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen(),
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

    # Tests that the false branch contains a quantized function call whereas the
    # true branch doesn't.
    def _is_quantized_function_call_node(
        node_def: node_def_pb2.NodeDef,
    ) -> bool:
      # [FUNCTION DOC]
      # HELPER: predicate for the branch check. Returns true if the node is a partitioned call whose
      # called function name starts with the quantized prefix. Input: node_def. Returns a boolean.
      return node_def.op == 'PartitionedCall' and node_def.attr[
          'f'
      ].func.name.startswith('quantized_')

    for func in output_graphdef.library.function:
      if func.signature.name.startswith('cond_false'):
        self.assertTrue(
            any(map(_is_quantized_function_call_node, func.node_def))
        )
      elif func.signature.name.startswith('cond_true'):
        self.assertFalse(
            any(map(_is_quantized_function_call_node, func.node_def))
        )

  # Run this test only with the eager mode.
  @test_util.run_v2_only
  def test_ptq_model_with_multiple_signatures(self):
    # [FUNCTION DOC]
    # Test (eager only): PTQ of a model with two signatures using record-file datasets.
    # Saves the two-signature fixture, writes per-signature representative data with the record saver,
    # quantizes with the dataset mapping in the options, and asserts both signatures exist and a quantized
    # function call is in the graph.
    # Create and save a model having 2 signatures.
    model = MultipleSignatureModel()

    signatures = {
        'sig1': model.matmul.get_concrete_function(
            tensor_spec.TensorSpec(shape=(1, 4), dtype=dtypes.float32)
        ),
        'sig2': model.conv.get_concrete_function(
            tensor_spec.TensorSpec(shape=(1, 3, 4, 3), dtype=dtypes.float32)
        ),
    }
    saved_model_save.save(
        model, self._input_saved_model_path, signatures=signatures
    )

    def data_gen_sig1() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: generator of representative data for signature sig1. Yields four samples mapping
      # matmul_input to random tensors of shape 1 by 4.
      """Generates tuple-style samples for signature 'sig1'.

      The first element of the tuple identifies the signature key the input data
      is for.

      Yields:
        Representative sample for 'sig1'.
      """
      for _ in range(4):
        yield {'matmul_input': random_ops.random_uniform(shape=(1, 4))}

    def data_gen_sig2() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: generator of representative data for signature sig2. Yields four samples mapping
      # conv_input to random tensors of shape 1,3,4,3.
      """Generates tuple-style samples for signature 'sig2'.

      The first element of the tuple identifies the signature key the input data
      is for.

      Yields:
        Representative sample for 'sig2'.
      """
      for _ in range(4):
        yield {'conv_input': random_ops.random_uniform(shape=(1, 3, 4, 3))}

    dataset_file_map = repr_dataset.TfRecordRepresentativeDatasetSaver({
        'sig1': os.path.join(self._input_saved_model_path, 'sig1_repr'),
        'sig2': os.path.join(self._input_saved_model_path, 'sig2_repr'),
    }).save({
        'sig1': data_gen_sig1(),
        'sig2': data_gen_sig2(),
    })
    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['sig1', 'sig2'],
        op_set=quant_opts_pb2.TF,
        representative_datasets=dataset_file_map,
    )
    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        output_directory=self._output_saved_model_path,
        quantization_options=quantization_options,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'sig1', 'sig2'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  # Run this test only with the eager mode.
  @test_util.run_v2_only
  def test_ptq_multiple_signatures_invalid_dataset_raises_value_error(self):
    # [FUNCTION DOC]
    # Test (eager only): a single flat dataset for a multi-signature model is invalid.
    # Saves the two-signature fixture and passes a plain list of samples; asserts the quantizer raises an
    # error saying the representative dataset is not a mapping (a mapping keyed by signature is needed).
    # Create and save a model having 2 signatures.
    model = MultipleSignatureModel()

    signatures = {
        'sig1': model.matmul.get_concrete_function(
            tensor_spec.TensorSpec(shape=(1, 4), dtype=dtypes.float32)
        ),
        'sig2': model.conv.get_concrete_function(
            tensor_spec.TensorSpec(shape=(1, 3, 4, 3), dtype=dtypes.float32)
        ),
    }
    saved_model_save.save(
        model, self._input_saved_model_path, signatures=signatures
    )

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags={tag_constants.SERVING},
        signature_keys=['sig1', 'sig2'],
    )

    # Use a dict-style samples instead of tuple-style samples. This is invalid
    # because for a model multiple signatures one must use tuple-style samples.
    invalid_dataset: repr_dataset.RepresentativeDataset = [
        {'matmul_input': random_ops.random_uniform(shape=(1, 4))}
        for _ in range(8)
    ]

    with self.assertRaisesRegex(
        Exception, 'Representative dataset is not a mapping'
    ):
      quantize_model.quantize(
          self._input_saved_model_path,
          output_directory=self._output_saved_model_path,
          quantization_options=quantization_options,
          representative_dataset=invalid_dataset,
      )

  @test_util.run_in_graph_and_eager_modes
  def test_ptq_model_with_tf1_saved_model_with_variable_for_conv2d(self):
    # [FUNCTION DOC]
    # Test: PTQ of a TF1 saved model with a variable-based convolution.
    # Creates and saves the TF1 model, quantizes with a generated dataset and the TF op set, asserts the
    # signature and a quantized function call.
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

    input_placeholder = self._create_and_save_tf1_conv_model(
        self._input_saved_model_path,
        signature_key,
        tags,
        input_key='x',
        output_key='output',
        use_variable=True,
    )

    signature_keys = [signature_key]

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_keys,
        op_set=quant_opts_pb2.TF,
    )

    data_gen = self._create_data_generator(
        input_key='x', shape=input_placeholder.shape
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), signature_keys
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  @parameterized.named_parameters(
      ('use_constant_with_int32_input', dtypes.int32, False),
      ('use_variable_with_int32_input', dtypes.int32, True),
      ('use_constant_with_int64_input', dtypes.int64, False),
      ('use_variable_with_int64_input', dtypes.int64, True),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_ptq_model_with_tf1_saved_model_with_variable_for_gather(
      self, input_type, use_variable
  ):
    # [FUNCTION DOC]
    # Parameterized test: PTQ of a TF1 gather model with constant or variable embeddings.
    # Parameters: input_type (int32 or int64) and use_variable. Quantizes with the TF op set and generated
    # indices and asserts the signature and a quantized function call.
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

    input_placeholder = self._create_and_save_tf1_gather_model(
        self._input_saved_model_path,
        signature_key,
        tags,
        input_key='x',
        output_key='output',
        input_type=input_type,
        use_variable=use_variable,
    )

    signature_keys = [signature_key]

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_keys,
        op_set=quant_opts_pb2.TF,
    )

    data_gen = self._create_data_generator(
        input_key='x',
        shape=input_placeholder.shape,
        minval=0,
        maxval=10,
        dtype=input_type,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), signature_keys
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  def test_ptq_model_with_variable_tf1_saved_model_unfreeze_constants(self):
    # [FUNCTION DOC]
    # Test: TF1 saved model with a variable and large filter, unfreezing constants enabled.
    # Quantizes with freeze_all_variables off, loads in a session, asserts a quantized function call,
    # one variable in the graph, and that variable name and shape match the checkpoint.
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

    input_placeholder = self._create_and_save_tf1_conv_model(
        self._input_saved_model_path,
        signature_key,
        tags,
        input_key='x',
        output_key='output',
        input_shape=(1, 16, 16, 8),
        # Uses large filter to exceed the constant size threshold of 64KiB
        # (specified by `kDefaultConstantSizeThresholdInBytes`) for unfreezing.
        filter_shape=(256, 8, 8, 16),
        use_variable=True,
    )

    signature_keys = [signature_key]

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_keys,
        op_set=quant_opts_pb2.TF,
        freeze_all_variables=False,
    )

    repr_ds = self._create_data_generator(
        input_key='x', shape=input_placeholder.shape, num_examples=2
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=repr_ds,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    # Confirm that the quantized model loads successfully.
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )

    with session.Session(graph=ops.Graph()) as sess:
      output_meta_graph_def = output_loader.load(sess, tags)

    # Checks that quantization is applied.
    output_graphdef = output_meta_graph_def.graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

    # Tests that there are variables in the model.
    variable_node_defs = _find_variables(output_graphdef)
    self.assertLen(variable_node_defs, 1)

    # Reads the variables from the checkpoint file and matches with the
    # variables found in the graph.
    checkpoint_path = os.path.join(
        self._output_saved_model_path, 'variables', 'variables'
    )
    var_name_and_shapes = checkpoint_utils.list_variables(checkpoint_path)

    # Checks that each variable's name and shape match.
    self.assertEqual(len(variable_node_defs), len(var_name_and_shapes))
    for var_name, shape in var_name_and_shapes:
      self.assertIn(var_name, variable_node_defs)
      self.assertEqual(
          shape,
          tensor_shape.TensorShape(
              variable_node_defs[var_name].attr['shape'].shape
          ),
      )

  @test_util.run_in_graph_and_eager_modes
  def test_ptq_model_with_tf1_saved_model(self):
    # [FUNCTION DOC]
    # Test: PTQ of a TF1 saved model with a constant convolution.
    # Creates and saves the model, quantizes with generated data for the TF op set, asserts the signature
    # and a quantized function call.
    tags = {tag_constants.SERVING}
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    input_placeholder = self._create_and_save_tf1_conv_model(
        self._input_saved_model_path,
        signature_key,
        tags,
        input_key='p',
        output_key='output',
        use_variable=False,
    )

    signature_keys = [signature_key]

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_keys,
        op_set=quant_opts_pb2.TF,
    )

    data_gen = self._create_data_generator(
        input_key='p', shape=input_placeholder.shape
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), signature_keys
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_ptq_model_with_tf1_saved_model_multiple_signatures(self):
    # [FUNCTION DOC]
    # Test: TF1 saved model with two signatures, sig1 and sig2, and per-signature datasets.
    # Builds two conv graphs in one session, saves them under two signature keys, quantizes with a mapping
    # from signature key to generator, and asserts both signatures and a quantized function call.
    tags = {tag_constants.SERVING}

    # Create two models and add them to a same SavedModel under different
    # signature keys.
    with ops.Graph().as_default(), session.Session() as sess:
      in_placeholder_1, output_tensor_1 = self._create_simple_tf1_conv_model()
      sig_def_1 = signature_def_utils_impl.predict_signature_def(
          inputs={'x1': in_placeholder_1}, outputs={'output1': output_tensor_1}
      )

      in_placeholder_2, output_tensor_2 = self._create_simple_tf1_conv_model()
      sig_def_2 = signature_def_utils_impl.predict_signature_def(
          inputs={'x2': in_placeholder_2}, outputs={'output2': output_tensor_2}
      )

      v1_builder = builder.SavedModelBuilder(self._input_saved_model_path)
      v1_builder.add_meta_graph_and_variables(
          sess,
          tags,
          signature_def_map={
              'sig1': sig_def_1,
              'sig2': sig_def_2,
          },
      )

      v1_builder.save()

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['sig1', 'sig2'],
        op_set=quant_opts_pb2.TF,
    )

    def data_gen_sig1() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: generator for signature sig1. Yields four samples mapping x1 to random tensors with the
      # first placeholder shape.
      """Generates tuple-style samples.

      The first element of the tuple identifies the signature key the input data
      is for.

      Yields:
        Representative samples for signature 'sig1'.
      """
      for _ in range(4):
        yield {'x1': random_ops.random_uniform(shape=in_placeholder_1.shape)}

    def data_gen_sig2() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: generator for signature sig2. Yields four samples mapping x2 to random tensors with the
      # second placeholder shape.
      """Generates tuple-style samples.

      The first element of the tuple identifies the signature key the input data
      is for.

      Yields:
        Representative samples for signature 'sig2'.
      """
      for _ in range(4):
        yield {'x2': random_ops.random_uniform(shape=in_placeholder_2.shape)}

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        output_directory=self._output_saved_model_path,
        quantization_options=quantization_options,
        representative_dataset={
            'sig1': data_gen_sig1(),
            'sig2': data_gen_sig2(),
        },
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'sig1', 'sig2'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_ptq_model_with_tf1_saved_model_invalid_input_key_raises_value_error(
      self,
  ):
    # [FUNCTION DOC]
    # Test: representative samples with an input key that does not exist raise an error.
    # Saves a TF1 conv model with input key x, supplies samples keyed invalid_input_key and asserts the
    # quantizer raises an exception about invalid input keys.
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

    input_placeholder = self._create_and_save_tf1_conv_model(
        self._input_saved_model_path,
        signature_key,
        tags,
        input_key='x',
        output_key='output',
        use_variable=False,
    )

    signature_keys = [signature_key]

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_keys,
    )

    # Representative generator function that yields with an invalid input key.
    invalid_data_gen = self._create_data_generator(
        input_key='invalid_input_key', shape=input_placeholder.shape
    )

    with self.assertRaisesRegex(
        Exception,
        'Invalid input keys for representative sample.',
    ):
      quantize_model.quantize(
          self._input_saved_model_path,
          self._output_saved_model_path,
          quantization_options,
          representative_dataset=invalid_data_gen,
      )

  @test_util.run_in_graph_and_eager_modes
  def test_ptq_model_with_non_default_tags(self):
    # [FUNCTION DOC]
    # Test: PTQ of a TF1 model saved with non-default tags (training and gpu).
    # Quantizes using those same tags and asserts the signature and a quantized function call; the output
    # is loaded with the same tags.
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    # Use a different set of tags other than {"serve"}.
    tags = {tag_constants.TRAINING, tag_constants.GPU}

    # Non-default tags are usually used when saving multiple metagraphs in TF1.
    input_placeholder = self._create_and_save_tf1_conv_model(
        self._input_saved_model_path,
        signature_key,
        tags,
        input_key='input',
        output_key='output',
        use_variable=True,
    )

    signature_keys = [signature_key]

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_keys,
        op_set=quant_opts_pb2.TF,
    )

    data_gen = self._create_data_generator(
        input_key='input', shape=input_placeholder.shape
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), signature_keys
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_ptq_model_with_wrong_tags_raises_error(self):
    # [FUNCTION DOC]
    # Test: asking for the serving tags when the model was saved with other tags fails.
    # Saves with training and gpu tags, quantizes with the serving tag, and asserts a runtime error that
    # no MetaGraphDef with those tags could be found.
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    save_tags = {tag_constants.TRAINING, tag_constants.GPU}

    input_placeholder = self._create_and_save_tf1_conv_model(
        self._input_saved_model_path,
        signature_key,
        save_tags,
        input_key='input',
        output_key='output',
        use_variable=True,
    )

    # Try to use a different set of tags to quantize.
    tags = {tag_constants.SERVING}
    signature_keys = [signature_key]
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_keys,
    )

    data_gen = self._create_data_generator(
        input_key='input', shape=input_placeholder.shape
    )
    with self.assertRaisesRegex(
        RuntimeError,
        "MetaGraphDef associated with tags {'serve'} could not be found",
    ):
      quantize_model.quantize(
          self._input_saved_model_path,
          self._output_saved_model_path,
          quantization_options,
          representative_dataset=data_gen,
      )

  def test_ptq_vocab_table_lookup_model(self):
    # [FUNCTION DOC]
    # Test: PTQ keeps a TF1 vocabulary hash table working.
    # Saves a vocabulary table model, quantizes with a list dataset, loads the output in a session and
    # asserts quantized function call, signature, and lookup of three words gives indices 1, 2 and 0.
    tags = {tag_constants.SERVING}
    signature_def_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    # Create and save a simple model that involves a hash table.
    inputs, outputs = self._create_and_save_vocab_table_lookup_model_tf1(
        self._input_saved_model_path, tags, signature_def_key
    )

    # Make sure that the desired input key and output key is present.
    self.assertIn('input_vocabs', inputs.keys())
    self.assertIn('lookup', outputs.keys())

    # Representative dataset is composed of a set of vocabs for table lookup.
    repr_ds = [
        {'input_vocabs': np.array([b'hello', b'model', b'quantization'])}
        for _ in range(4)
    ]

    signature_def_keys = [signature_def_key]
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_def_keys,
        op_set=quant_opts_pb2.TF,
    )

    quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=repr_ds,
    )

    # Tests table lookup to make sure the table has been initialized
    # successfully.
    with session.Session(graph=ops.Graph()) as sess:
      output_meta_graph_def = saved_model_loader.load(
          sess, tags=tags, export_dir=self._output_saved_model_path
      )

      # The graph should contain a quantized function call (it contains a
      # single f32 matmul node).
      self.assertTrue(
          self._contains_quantized_function_call(
              output_meta_graph_def.graph_def
          )
      )
      self.assertCountEqual(
          output_meta_graph_def.signature_def.keys(), signature_def_keys
      )

      signature_def = output_meta_graph_def.signature_def[signature_def_key]

      input_tensor_name = signature_def.inputs['input_vocabs'].name
      input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)

      lookup_tensor_name = signature_def.outputs['lookup'].name
      lookup_tensor = sess.graph.get_tensor_by_name(lookup_tensor_name)

      lookup_val = sess.run(
          lookup_tensor,
          feed_dict={
              input_tensor: np.array([b'model', b'quantization', b'hello'])
          },
      )

      self.assertAllClose(lookup_val, [1.0, 2.0, 0.0])

  def test_ptq_file_init_hash_table_lookup_model(self):
    # [FUNCTION DOC]
    # Test: PTQ keeps a TF1 file-initialized hash table working.
    # Same as the vocabulary test but with a file-based table; asserts a missing word maps to minus one and
    # the known words to 2 and 1.
    tags = {tag_constants.SERVING}
    signature_def_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    # Create and save a simple model that involves a hash table.
    inputs, outputs = self._create_and_save_file_init_hash_table_model_tf1(
        self._input_saved_model_path, tags, signature_def_key
    )

    # Make sure that the desired input key and output key is present.
    self.assertIn('input_vocabs', inputs.keys())
    self.assertIn('lookup', outputs.keys())

    # Representative dataset is composed of a set of vocabs for table lookup.
    repr_ds = [
        {'input_vocabs': np.array([b'static', b'range', b'quantization'])}
        for _ in range(4)
    ]

    signature_def_keys = [signature_def_key]
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_def_keys,
        op_set=quant_opts_pb2.TF,
    )

    quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=repr_ds,
    )

    # Tests table lookup to make sure the table has been initialized
    # successfully.
    with session.Session(graph=ops.Graph()) as sess:
      output_meta_graph_def = saved_model_loader.load(
          sess, tags=tags, export_dir=self._output_saved_model_path
      )

      # The graph should contain a quantized function call (it contains a
      # single f32 matmul node).
      self.assertTrue(
          self._contains_quantized_function_call(
              output_meta_graph_def.graph_def
          )
      )
      self.assertCountEqual(
          output_meta_graph_def.signature_def.keys(), signature_def_keys
      )

      signature_def = output_meta_graph_def.signature_def[signature_def_key]

      input_tensor_name = signature_def.inputs['input_vocabs'].name
      input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)

      lookup_tensor_name = signature_def.outputs['lookup'].name
      lookup_tensor = sess.graph.get_tensor_by_name(lookup_tensor_name)

      lookup_val = sess.run(
          lookup_tensor,
          feed_dict={
              input_tensor: np.array([b'dynamic', b'quantization', b'range'])
          },
      )

      # "dynamic" is not in the table: -1 (default value)
      self.assertAllClose(lookup_val, [-1.0, 2.0, 1.0])

  @parameterized.named_parameters(
      ('none', None, False, False, quant_opts_pb2.TF, False, 'SAME'),
      ('relu', nn_ops.relu, False, False, quant_opts_pb2.TF, False, 'SAME'),
      ('relu6', nn_ops.relu6, False, False, quant_opts_pb2.TF, False, 'SAME'),
      ('with_bias', None, True, False, quant_opts_pb2.TF, False, 'SAME'),
      (
          'with_bias_and_relu',
          nn_ops.relu,
          True,
          False,
          quant_opts_pb2.TF,
          False,
          'SAME',
      ),
      (
          'with_bias_and_relu6',
          nn_ops.relu6,
          True,
          False,
          quant_opts_pb2.TF,
          False,
          'SAME',
      ),
      ('none_to_xla', None, False, False, quant_opts_pb2.XLA, False, 'SAME'),
      (
          'with_bias_and_relu6_to_xla',
          nn_ops.relu6,
          True,
          False,
          quant_opts_pb2.XLA,
          False,
          'SAME',
      ),
      (
          'with_bias_to_xla_dynamic',
          None,
          True,
          False,
          quant_opts_pb2.XLA,
          True,
          'SAME',
      ),
      (
          'none_to_xla_padding_valid',
          None,
          False,
          False,
          quant_opts_pb2.XLA,
          False,
          'VALID',
      ),
      (
          'with_bias_and_relu6_to_xla_padding_valid',
          nn_ops.relu6,
          True,
          False,
          quant_opts_pb2.XLA,
          False,
          'VALID',
      ),
      (
          'with_bias_to_xla_dynamic_padding_valid',
          None,
          True,
          False,
          quant_opts_pb2.XLA,
          True,
          'VALID',
      ),
  )
  def test_conv3d_ptq_model(
      self,
      activation_fn: Optional[ops.Operation],
      has_bias: bool,
      has_batch_norm: bool,
      target_opset: quant_opts_pb2.OpSet,
      input_shape_dynamic: bool,
      padding: str,
  ):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the has_batch_norm parameter is never used.
    # Parameterized PTQ test for 3D convolution.
    # Parameters: activation_fn, has_bias, has_batch_norm (accepted but not used by the body),
    # target_opset, input_shape_dynamic and padding (SAME or VALID). Builds a local 3D conv model, calibrates
    # with 500 random samples, quantizes with TF op set (asserts outputs close to float, tolerance 0.00494,
    # and a quantized call), then with the target op set (asserts XLA conv for XLA, and closeness of both
    # quantized paths).
    input_shape = [1, 3, 4, 3, 3]
    if input_shape_dynamic:
      input_shape = [None, None, None, None, 3]

    class ConvModel(module.Module):
      # [CLASS DOC]
      # Local fixture module: a traced 3D convolution (filter 2,3,3,3,2) with optional bias and activation
      # and the padding selected by the enclosing test.

      def __init__(self):
        # [FUNCTION DOC]
        # Constructor. Creates a random float32 filter (shape 2,3,3,3,2, range minus 0.5 to 0.5) and a random
        # bias of length two (range 0 to 0.2). Returns nothing.
        self.filters = np.random.uniform(
            low=-0.5, high=0.5, size=(2, 3, 3, 3, 2)
        ).astype('f4')
        self.bias = np.random.uniform(low=0.0, high=0.2, size=(2)).astype('f4')

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(shape=input_shape, dtype=dtypes.float32)
          ]
      )
      def conv3d(self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
        # [FUNCTION DOC]
        # Traced function (float32 input of the test's shape). Performs a 3D convolution with strides
        # 1,1,2,1,1, NDHWC layout and chosen padding, optionally adds bias and applies the activation.
        # Returns a mapping with key output. Input: input_tensor.
        """Performs a 3D convolution operation.

        Args:
          input_tensor: Input tensor to perform convolution on.

        Returns:
          A map of: output key -> output result.
        """
        out = nn_ops.conv3d(
            input_tensor,
            self.filters,
            strides=[1, 1, 2, 1, 1],
            dilations=[1, 1, 1, 1, 1],
            padding=padding,
            data_format='NDHWC',
        )
        if has_bias:
          out = nn_ops.bias_add(out, self.bias)
        if activation_fn is not None:
          out = activation_fn(out)
        return {'output': out}

    model = ConvModel()
    saved_model_save.save(model, self._input_saved_model_path)

    repr_ds = []
    for _ in range(500):
      repr_ds.append({
          'input_tensor': ops.convert_to_tensor(
              np.random.uniform(
                  low=-0.1, high=0.2, size=(1, 3, 4, 3, 3)
              ).astype('f4')
          ),
      })

    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

    # Check the converted model with TF opset as the baseline.
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=quant_opts_pb2.TF,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=repr_ds,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {signature_key}
    )

    input_data = np.random.uniform(
        low=-0.1, high=0.2, size=(1, 3, 4, 3, 3)
    ).astype('f4')
    expected_outputs = model.conv3d(input_data)
    got_outputs = converted_model.signatures[signature_key](
        input_tensor=ops.convert_to_tensor(input_data)
    )
    self.assertAllClose(expected_outputs, got_outputs, atol=0.00494)

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

    # Check the converted model in the target opset.
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=target_opset,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path_2,
        quantization_options,
        representative_dataset=repr_ds,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {signature_key}
    )
    loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path_2
    )
    graphdef = loader.get_meta_graph_def_from_tags(tags).graph_def
    if target_opset == quant_opts_pb2.XLA:
      self.assertTrue(self._contains_op(graphdef, 'XlaConvV2'))

    new_outputs = converted_model.signatures[signature_key](
        input_tensor=ops.convert_to_tensor(input_data)
    )
    # The quantized model in XLA opset is expected to have similar fidelity
    # compared to the quantized model in TF opset.
    self.assertAllClose(new_outputs, got_outputs, atol=0.00306)
    self.assertAllClose(new_outputs, expected_outputs, atol=0.00494)

  # Tests the case of having a signature key of `main` because it is a
  # special name in the TF quantizer's MLIR pipeline that should be treated
  # with care.
  @test_util.run_in_graph_and_eager_modes
  def test_ptq_model_with_signature_key_main(self):
    # [FUNCTION DOC]
    # Test: a signature key named main is handled correctly by the quantizer pipeline.
    # Saves a TF1 conv model under the key main, quantizes it, asserts the signature, a quantized function
    # call and that the original main function was renamed to main_0.
    signature_key = 'main'
    tags = {tag_constants.SERVING}

    input_placeholder = self._create_and_save_tf1_conv_model(
        self._input_saved_model_path,
        signature_key,
        tags,
        input_key='x',
        output_key='output',
        use_variable=True,
    )

    signature_keys = [signature_key]
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_keys,
        op_set=quant_opts_pb2.TF,
    )

    data_gen = self._create_data_generator(
        input_key='x', shape=input_placeholder.shape
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), signature_keys
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

    # Makes sure that the original function identified by the signature key
    # `main` is renamed to `main_0` (see `InsertMainFunctionPass` for details).
    self.assertTrue(
        any(
            map(
                lambda func: func.signature.name == 'main_0',
                output_graphdef.library.function,
            )
        )
    )


class DynamicRangeQuantizationTest(quantize_model_test_base.QuantizedModelTest):
  # [CLASS DOC]
  # Tests for dynamic range quantization (no calibration data needed) across the XLA, uniform quantized
  # and TF op sets: einsum, matmul, convolution, depthwise convolution, gather, minimum element
  # thresholds, wrong tags, output directory handling and table models. Runs in graph and eager modes.
  """Test cases for dynamic range quantization.

  Tries to run all tests cases in both the graph mode (default in TF1) and the
  eager mode (default in TF2) to ensure support for when TF2 is disabled.
  """

  @parameterized.parameters(
      (True, quant_opts_pb2.XLA),
      (False, quant_opts_pb2.XLA),
      (True, quant_opts_pb2.UNIFORM_QUANTIZED),
      (False, quant_opts_pb2.UNIFORM_QUANTIZED),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_einsum_model(
      self,
      constant_y_operand: bool,
      target_opset: quant_opts_pb2.OpSet,
  ):
    # [FUNCTION DOC]
    # Parameterized test: dynamic range quantization of an einsum model.
    # Parameters: constant_y_operand (kernel constant or input) and target_opset (XLA or uniform
    # quantized). Saves the model, quantizes, and asserts the signature and that no XLA dot op appears;
    # for uniform quantized a batch matmul is present, otherwise an einsum op remains (einsum is not yet
    # quantized, per the TODO in the test).
    equation = 'abc,cde->abde'
    _, y_shape, bias_shape, x_signature, y_signature = (
        self._prepare_sample_einsum_datashapes(equation, use_bias=True)
    )

    model = self._create_einsum_model(
        equation,
        y_shape,
        x_signature,
        y_signature,
        bias_shape,
        activation_fn=nn_ops.relu,
    )

    if constant_y_operand:
      signatures = {
          'serving_default': model.einsum_with_kernel.get_concrete_function(),
      }
    else:
      signatures = {
          'serving_default': (
              model.einsum_without_kernel.get_concrete_function()
          ),
      }

    saved_model_save.save(model, self._input_saved_model_path, signatures)

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_DYNAMIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def

    # TODO(b/286489783): Support Einsum
    if target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      self.assertFalse(self._contains_op(output_graphdef, 'XlaDotV2'))
      self.assertTrue(self._contains_op(output_graphdef, 'BatchMatMulV2'))
    else:
      self.assertFalse(self._contains_op(output_graphdef, 'XlaDotV2'))
      self.assertTrue(self._contains_op(output_graphdef, 'Einsum'))

  @parameterized.named_parameters(
      ('to_tf_per_tensor', quant_opts_pb2.TF, False),
      ('to_xla_per_tensor', quant_opts_pb2.XLA, False),
      (
          'to_uniform_quantized_per_tensor',
          quant_opts_pb2.UNIFORM_QUANTIZED,
          False,
      ),
      (
          'to_uniform_quantized_per_channel',
          quant_opts_pb2.UNIFORM_QUANTIZED,
          True,
      ),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_matmul_model(
      self,
      target_opset: quant_opts_pb2.OpSet,
      enable_per_channel_quantization: bool,
  ):
    # [FUNCTION DOC]
    # Parameterized test: dynamic range quantization of a matmul model.
    # Parameters: target_opset (TF, XLA, uniform quantized) and enable_per_channel_quantization.
    # Asserts the signature and, per op set, the proper quantized dot op (and axis attribute for per
    # channel), or a quantized function call with a retained matmul for TF.
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=self._input_saved_model_path,
    )

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_DYNAMIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
        enable_per_channel_quantization=enable_per_channel_quantization,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def

    if target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      self.assertTrue(
          self._contains_op(output_graphdef, 'UniformQuantizedDotHybrid')
      )
      self.assertFalse(self._contains_op(output_graphdef, 'MatMul'))
      if enable_per_channel_quantization:
        quantized_axis_attr = attr_value_pb2.AttrValue(i=-1)
        self.assertTrue(
            self._contains_op(
                output_graphdef,
                'UniformQuantizedDotHybrid',
                'rhs_quantization_axis',
                quantized_axis_attr,
            )
        )
    elif target_opset == quant_opts_pb2.XLA:
      self.assertTrue(self._contains_op(output_graphdef, 'XlaDotV2'))
      self.assertFalse(self._contains_op(output_graphdef, 'MatMul'))
    else:
      self.assertTrue(self._contains_quantized_function_call(output_graphdef))
      self.assertTrue(self._contains_op(output_graphdef, 'MatMul'))

  @parameterized.named_parameters(
      ('to_tf_per_tensor', quant_opts_pb2.TF, False),
      ('to_xla_per_tensor', quant_opts_pb2.XLA, False),
      (
          'to_uniform_quantized_per_tensor',
          quant_opts_pb2.UNIFORM_QUANTIZED,
          False,
      ),
      (
          'to_uniform_quantized_per_channel',
          quant_opts_pb2.UNIFORM_QUANTIZED,
          True,
      ),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_conv_model(
      self,
      target_opset: quant_opts_pb2.OpSet,
      enable_per_channel_quantization: bool,
  ):
    # [FUNCTION DOC]
    # Parameterized test: dynamic range quantization of a convolution with bias, batch norm and relu6.
    # Parameters: target_opset and enable_per_channel_quantization. Asserts the signature and per op set
    # the expected hybrid or XLA conv op, per-channel axis and size attributes, and that float conv is
    # removed (or retained for TF with a quantized function call).
    filter_shape = (2, 3, 512, 2)

    model = self._create_conv2d_model(
        input_shape=(1, 3, 4, 512),
        filter_shape=filter_shape,
        has_bias=True,
        has_batch_norm=True,
        activation_fn=nn_ops.relu6,
    )

    saved_model_save.save(model, self._input_saved_model_path)

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_DYNAMIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
        enable_per_channel_quantization=enable_per_channel_quantization,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def

    if enable_per_channel_quantization:
      quantized_axis = 3
      quantized_axis_attr = attr_value_pb2.AttrValue(i=quantized_axis)
      quantized_dim_size_attr = attr_value_pb2.AttrValue(
          list=attr_value_pb2.AttrValue.ListValue(
              shape=[
                  tensor_shape_pb2.TensorShapeProto(
                      dim=[
                          tensor_shape_pb2.TensorShapeProto.Dim(
                              size=filter_shape[quantized_axis]
                          )
                      ]
                  )
              ]
          )
      )

    if target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      self.assertTrue(
          self._contains_op(
              output_graphdef, 'UniformQuantizedConvolutionHybrid'
          )
      )
      self.assertFalse(self._contains_op(output_graphdef, 'Conv2D'))
      if enable_per_channel_quantization:
        self.assertTrue(
            self._contains_op(
                output_graphdef,
                'UniformQuantizedConvolutionHybrid',
                'rhs_quantization_axis',
                quantized_axis_attr,
            )
        )
        self.assertTrue(
            self._contains_op(
                output_graphdef,
                'Const',
                '_output_shapes',
                quantized_dim_size_attr,
            )
        )
    elif target_opset == quant_opts_pb2.XLA:
      self.assertTrue(self._contains_op(output_graphdef, 'XlaConvV2'))
      self.assertFalse(self._contains_op(output_graphdef, 'Conv2D'))
    else:
      self.assertTrue(self._contains_quantized_function_call(output_graphdef))
      self.assertTrue(self._contains_op(output_graphdef, 'Conv2D'))

  @parameterized.named_parameters(
      ('to_tf_per_tensor', quant_opts_pb2.TF, False),
      ('to_xla_per_tensor', quant_opts_pb2.XLA, False),
      (
          'to_uniform_quantized_per_tensor',
          quant_opts_pb2.UNIFORM_QUANTIZED,
          False,
      ),
      (
          'to_uniform_quantized_per_channel',
          quant_opts_pb2.UNIFORM_QUANTIZED,
          True,
      ),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_depthwise_conv_model(
      self,
      target_opset: quant_opts_pb2.OpSet,
      enable_per_channel_quantization: bool,
  ):
    # [FUNCTION DOC]
    # Parameterized test: dynamic range quantization of a depthwise convolution.
    # Parameters: target_opset and enable_per_channel_quantization. Strides are 1,2,2,1 (only the middle two
    # are expected for uniform quantized). Asserts strides attributes, per channel axis and size attributes,
    # and that the depthwise op is replaced for uniform and XLA op sets.
    filter_shape = (2, 3, 1024, 2)
    strides = (1, 2, 2, 1)

    model = self._create_depthwise_conv2d_model(
        input_shape=(1, 3, 4, 1024), filter_shape=filter_shape, strides=strides
    )

    saved_model_save.save(model, self._input_saved_model_path)

    tags = [tag_constants.SERVING]

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_DYNAMIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
        enable_per_channel_quantization=enable_per_channel_quantization,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def

    # Uniform Quantized op takes only the first and the second values for
    # strides.
    strides_to_check = (
        (strides[1], strides[2])
        if target_opset == quant_opts_pb2.UNIFORM_QUANTIZED
        else strides
    )
    strides_attr = attr_value_pb2.AttrValue(
        list=attr_value_pb2.AttrValue.ListValue(i=strides_to_check)
    )

    if enable_per_channel_quantization:
      quantized_axis_attr = attr_value_pb2.AttrValue(i=3)
      quantized_dim_size_attr = attr_value_pb2.AttrValue(
          list=attr_value_pb2.AttrValue.ListValue(
              shape=[
                  tensor_shape_pb2.TensorShapeProto(
                      dim=[
                          tensor_shape_pb2.TensorShapeProto.Dim(
                              size=filter_shape[2] * filter_shape[3]
                          )
                      ]
                  )
              ]
          )
      )

    if target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      self.assertTrue(
          self._contains_op(
              output_graphdef,
              'UniformQuantizedConvolutionHybrid',
              'window_strides',
              strides_attr,
          )
      )
      self.assertFalse(
          self._contains_op(output_graphdef, 'DepthwiseConv2dNative')
      )
      if enable_per_channel_quantization:
        self.assertTrue(
            self._contains_op(
                output_graphdef,
                'UniformQuantizedConvolutionHybrid',
                'rhs_quantization_axis',
                quantized_axis_attr,
            )
        )
        self.assertTrue(
            self._contains_op(
                output_graphdef,
                'Const',
                '_output_shapes',
                quantized_dim_size_attr,
            )
        )
    elif target_opset == quant_opts_pb2.XLA:
      self.assertTrue(self._contains_op(output_graphdef, 'XlaConvV2'))
      self.assertFalse(
          self._contains_op(output_graphdef, 'DepthwiseConv2dNative')
      )
    else:
      self.assertTrue(self._contains_quantized_function_call(output_graphdef))
      self.assertTrue(
          self._contains_op(
              output_graphdef, 'DepthwiseConv2dNative', 'strides', strides_attr
          )
      )

  @parameterized.named_parameters(
      ('to_tf_use_constant', quant_opts_pb2.TF, False),
      ('to_xla_use_constant', quant_opts_pb2.XLA, False),
      (
          'to_uniform_quantized_use_constant',
          quant_opts_pb2.UNIFORM_QUANTIZED,
          False,
      ),
      ('to_tf_use_variable', quant_opts_pb2.TF, True),
      ('to_xla_use_variable', quant_opts_pb2.XLA, True),
      (
          'to_uniform_quantized_use_variable',
          quant_opts_pb2.UNIFORM_QUANTIZED,
          True,
      ),
  )
  @test_util.run_v2_only
  def test_gather_model(
      self, target_opset: quant_opts_pb2.OpSet, use_variable: bool
  ):
    # [FUNCTION DOC]
    # Parameterized test (eager only): dynamic range quantization of a gather model.
    # Parameters: target_opset and use_variable. Saves an int64 gather model, quantizes and asserts the
    # signature and the size ratio (above 0.65 for uniform quantized, below one third otherwise).
    input_type = dtypes.int64
    model = self._create_gather_model(input_type, use_variable)
    saved_model_save.save(model, self._input_saved_model_path)

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_DYNAMIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    if target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      self.assertGreater(
          testing.get_size_ratio(
              self._output_saved_model_path, self._input_saved_model_path
          ),
          0.65,
      )
    else:
      # Due to other meta data, the compression is not exactly 1/4.
      self.assertLess(
          testing.get_size_ratio(
              self._output_saved_model_path, self._input_saved_model_path
          ),
          1 / 3,
      )

  @parameterized.named_parameters(
      ('to_tf_with_int32_input_type', dtypes.int32, quant_opts_pb2.TF),
      ('to_xla_with_int32_input_type', dtypes.int32, quant_opts_pb2.XLA),
      ('to_xla_with_int64_input_type', dtypes.int64, quant_opts_pb2.XLA),
      (
          'to_uq_with_int32_input_type',
          dtypes.int32,
          quant_opts_pb2.UNIFORM_QUANTIZED,
      ),
  )
  @test_util.run_v2_only
  def test_gather_and_conv_model(
      self, input_type: dtypes, target_opset: quant_opts_pb2.OpSet
  ):
    # [FUNCTION DOC]
    # Parameterized test (eager only): dynamic range quantization of gather followed by convolution.
    # Parameters: input_type and target_opset. Asserts the signature, the size ratio per op set and the
    # expected conv op (hybrid conv, XLA conv or quantized function call).
    model = self._create_simple_gather_and_conv_model(
        input_type, filter_shape=(2, 3, 3, 1024)
    )
    saved_model_save.save(model, self._input_saved_model_path)

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_DYNAMIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def

    if target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      self.assertGreater(
          testing.get_size_ratio(
              self._output_saved_model_path, self._input_saved_model_path
          ),
          0.65,
      )
      self.assertTrue(
          self._contains_op(
              output_graphdef, 'UniformQuantizedConvolutionHybrid'
          )
      )
    else:
      # Due to other meta data, the compression is not exactly 1/4.
      self.assertLess(
          testing.get_size_ratio(
              self._output_saved_model_path, self._input_saved_model_path
          ),
          1 / 3,
      )
      if target_opset == quant_opts_pb2.XLA:
        self.assertTrue(self._contains_op(output_graphdef, 'XlaConvV2'))
      else:
        self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  @test_util.run_in_graph_and_eager_modes
  def test_conv_model_with_wrong_tags_raises_error(self):
    # [FUNCTION DOC]
    # Test: dynamic range quantization with tags that do not match the saved TF1 model fails.
    # Saves with training and gpu tags, quantizes with serving tags and the uniform quantized op set and
    # asserts an error saying the tags could not be found, of the runtime error class.
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    save_tags = {tag_constants.TRAINING, tag_constants.GPU}

    input_placeholder = self._create_and_save_tf1_conv_model(
        self._input_saved_model_path,
        signature_key,
        save_tags,
        input_key='input',
        output_key='output',
        use_variable=True,
    )

    tags = {tag_constants.SERVING}
    signature_keys = [signature_key]
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_DYNAMIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_keys,
        op_set=quant_opts_pb2.OpSet.UNIFORM_QUANTIZED,
    )

    # Try to use a different set of tags to quantize.
    data_gen = self._create_data_generator(
        input_key='input', shape=input_placeholder.shape
    )

    # StatusNotOk error. `Exception` is used here because importing
    # `StatusNotOk` may break the open-sourced version of TensorFlow.
    with self.assertRaisesRegex(
        Exception,
        'could not be found in SavedModel, with available tags',
    ) as raises:
      quantize_model.quantize(
          self._input_saved_model_path,
          self._output_saved_model_path,
          quantization_options,
          representative_dataset=data_gen,
      )

    self.assertEqual(raises.exception.__class__.__name__, 'RuntimeError')

  @parameterized.named_parameters(
      ('quantize', True, 0),
      ('not_quantize', False, 10000),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_minimum_elements_for_weights(self, quantize, num_elements):
    # [FUNCTION DOC]
    # Parameterized test: minimum number of weight elements for dynamic range quantization.
    # Parameters: quantize (whether the weight is expected to be quantized) and num_elements (threshold
    # 0 or 10000). Asserts the signature and that an int8 quantized constant exists when quantized, and
    # does not exist when the threshold exceeds the weight size.
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=self._input_saved_model_path,
    )

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_DYNAMIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.OpSet.UNIFORM_QUANTIZED,
    )
    quantization_options.min_num_elements_for_weights = num_elements

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    type_attr = attr_value_pb2.AttrValue(type=types_pb2.DT_QINT8)
    if quantize:
      self.assertTrue(
          self._contains_op(output_graphdef, 'Const', 'dtype', type_attr)
      )
    else:
      self.assertFalse(
          self._contains_op(output_graphdef, 'Const', 'dtype', type_attr)
      )

  @parameterized.named_parameters(
      ('to_tf_use_constant', quant_opts_pb2.TF, False),
      ('to_xla_use_constant', quant_opts_pb2.XLA, False),
      (
          'to_uniform_quantized_use_constant',
          quant_opts_pb2.UNIFORM_QUANTIZED,
          False,
      ),
      ('to_tf_use_variable', quant_opts_pb2.TF, True),
      ('to_xla_use_variable', quant_opts_pb2.XLA, True),
      (
          'to_uniform_quantized_use_variable',
          quant_opts_pb2.UNIFORM_QUANTIZED,
          True,
      ),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_gather_model_tf1(
      self, target_opset: quant_opts_pb2.OpSet, use_variable: bool
  ):
    # [FUNCTION DOC]
    # Parameterized test: dynamic range quantization of TF1 gather models.
    # Parameters: target_opset and use_variable. Saves a TF1 int32 gather model, quantizes and asserts the
    # signature and size ratio against thresholds that depend on op set and variable usage.
    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}

    _ = self._create_and_save_tf1_gather_model(
        self._input_saved_model_path,
        signature_key,
        tags,
        input_key='x',
        output_key='output',
        input_type=dtypes.int32,
        use_variable=use_variable,
    )

    signature_keys = [signature_key]
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_DYNAMIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_keys,
        op_set=target_opset,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), signature_keys
    )

    if target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      threshold = 0.45 if use_variable else 0.7
      self.assertGreater(
          testing.get_size_ratio(
              self._output_saved_model_path, self._input_saved_model_path
          ),
          threshold,
      )

    else:
      threshold = 0.19 if use_variable else 0.42
      self.assertLess(
          testing.get_size_ratio(
              self._output_saved_model_path, self._input_saved_model_path
          ),
          threshold,
      )

  @test_util.run_in_graph_and_eager_modes
  def test_non_empty_directory_raises_file_exists_error(self):
    # [FUNCTION DOC]
    # Test: quantizing into a non-empty output directory raises an error by default.
    # Writes a dummy file into the output directory and asserts the quantizer raises a file-exists error.
    # Side effect: writes a file in the temporary output directory.
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=self._input_saved_model_path,
    )

    # Create a file inside the output directory.
    file_io.write_string_to_file(
        filename=os.path.join(self._output_saved_model_path, 'dummy_file.txt'),
        file_content='Test content',
    )

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_DYNAMIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
    )

    with self.assertRaisesRegex(
        FileExistsError, 'Output directory already exists'
    ):
      quantize_model.quantize(
          self._input_saved_model_path,
          self._output_saved_model_path,
          quantization_options,
      )

  @test_util.run_in_graph_and_eager_modes
  def test_non_empty_directory_overwritten(self):
    # [FUNCTION DOC]
    # Test: a non-empty output directory can be overwritten when requested.
    # Writes a dummy file into the output directory, quantizes with overwrite enabled, and asserts the
    # signature and a quantized function call.
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=self._input_saved_model_path,
    )

    # Create a file inside the output directory.
    file_io.write_string_to_file(
        filename=os.path.join(self._output_saved_model_path, 'dummy_file.txt'),
        file_content='Test content',
    )

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_DYNAMIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.TF,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        overwrite_output_directory=True,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_quantized_function_call(output_graphdef))

  def test_table_initialized_when_model_has_table_tf1(self):
    # [FUNCTION DOC]
    # Test: hash table initialization is preserved by dynamic range quantization.
    # Saves a TF1 vocabulary table model, quantizes, loads in a session and asserts the signature and that
    # looking up three words gives 1, 2 and 0.
    tags = {tag_constants.SERVING}
    signature_def_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    # Create and save a simple model that involves a hash table.
    inputs, outputs = self._create_and_save_vocab_table_lookup_model_tf1(
        self._input_saved_model_path, tags, signature_def_key
    )

    # Make sure that the desired input key and output key is present.
    self.assertIn('input_vocabs', inputs.keys())
    self.assertIn('lookup', outputs.keys())

    signature_def_keys = [signature_def_key]
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_DYNAMIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=signature_def_keys,
    )

    quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )

    # Tests table lookup to make sure the table has been initialized
    # successfully.
    with session.Session(graph=ops.Graph()) as sess:
      output_meta_graph_def = saved_model_loader.load(
          sess, tags=tags, export_dir=self._output_saved_model_path
      )

      self.assertCountEqual(
          output_meta_graph_def.signature_def.keys(), signature_def_keys
      )

      signature_def = output_meta_graph_def.signature_def[signature_def_key]

      input_tensor_name = signature_def.inputs['input_vocabs'].name
      input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)

      lookup_tensor_name = signature_def.outputs['lookup'].name
      lookup_tensor = sess.graph.get_tensor_by_name(lookup_tensor_name)

      lookup_val = sess.run(
          lookup_tensor,
          feed_dict={
              input_tensor: np.array([b'model', b'quantization', b'hello'])
          },
      )

      self.assertAllClose(lookup_val, [1.0, 2.0, 0.0])

  def test_file_init_hash_table_lookup_model(self):
    # [FUNCTION DOC]
    # Test: file-initialized hash table is preserved by dynamic range quantization.
    # Quantizes a TF1 model with a file table and asserts the signature and lookup results of minus one for
    # a missing word and 2 and 1 for known words.
    tags = {tag_constants.SERVING}
    signature_def_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY

    # Create and save a simple model that involves a hash table.
    inputs, outputs = self._create_and_save_file_init_hash_table_model_tf1(
        self._input_saved_model_path, tags, signature_def_key
    )
    # Make sure that the desired input key and output key is present.
    self.assertIn('input_vocabs', inputs.keys())
    self.assertIn('lookup', outputs.keys())

    signature_def_keys = [signature_def_key]
    quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options=quant_opts_pb2.QuantizationOptions(
            quantization_method=quant_opts_pb2.QuantizationMethod(
                preset_method=_PresetMethod.METHOD_DYNAMIC_RANGE_INT8
            ),
            tags=tags,
            signature_keys=signature_def_keys,
        ),
    )

    # Tests table lookup to make sure the table has been initialized
    # successfully.
    with session.Session(graph=ops.Graph()) as sess:
      output_meta_graph_def = saved_model_loader.load(
          sess, tags=tags, export_dir=self._output_saved_model_path
      )

      self.assertCountEqual(
          output_meta_graph_def.signature_def.keys(), signature_def_keys
      )

      signature_def = output_meta_graph_def.signature_def[signature_def_key]

      input_tensor_name = signature_def.inputs['input_vocabs'].name
      input_tensor = sess.graph.get_tensor_by_name(input_tensor_name)

      lookup_tensor_name = signature_def.outputs['lookup'].name
      lookup_tensor = sess.graph.get_tensor_by_name(lookup_tensor_name)

      lookup_val = sess.run(
          lookup_tensor,
          feed_dict={
              input_tensor: np.array([b'dynamic', b'quantization', b'range'])
          },
      )

      # "dynamic" is not in the table: -1 (default value)
      self.assertAllClose(lookup_val, [-1.0, 2.0, 1.0])


class WeightOnlyQuantizationTest(quantize_model_test_base.QuantizedModelTest):
  # [CLASS DOC]
  # Tests for weight-only quantization (static range weight-only int8 preset) for einsum, matmul,
  # convolution, depthwise convolution, gather, function aliases and while loops. Runs in graph and eager
  # modes. Mostly checks size reduction and the presence of int8 constants.
  """Test cases for weight-only quantization.

  Run all tests cases in both the graph mode (default in TF1) and the eager mode
  (default in TF2) to ensure support for when TF2 is disabled.
  """

  @test_util.run_in_graph_and_eager_modes
  def test_einsum_model(
      self,
  ):
    # [FUNCTION DOC]
    # Test: weight-only quantization of an einsum model to XLA.
    # Quantizes with the weight-only preset, asserts the signature, an int8 constant exists and the output
    # is smaller than half of the input size.
    equation = 'abc,cde->abde'
    _, y_shape, bias_shape, x_signature, y_signature = (
        self._prepare_sample_einsum_datashapes(equation, use_bias=True)
    )

    model = self._create_einsum_model(
        equation,
        y_shape,
        x_signature,
        y_signature,
        bias_shape,
        activation_fn=nn_ops.relu,
    )

    # Use constant y operand.
    signatures = {
        'serving_default': model.einsum_with_kernel.get_concrete_function(),
    }

    saved_model_save.save(model, self._input_saved_model_path, signatures)

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.XLA,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def

    self.assertTrue(
        self._contains_op(
            output_graphdef,
            op_name='Const',
            attr_name='dtype',
            attr_val=attr_value_pb2.AttrValue(type=types_pb2.DT_INT8),
        )
    )
    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.5,
    )

  @parameterized.named_parameters(
      ('to_xla_per_tensor', quant_opts_pb2.XLA, False),
      ('stablehlo_per_channel', quant_opts_pb2.STABLEHLO, True),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_matmul_model(
      self,
      target_opset: quant_opts_pb2.OpSet,
      enable_per_channel_quantization: bool,
  ):
    # [FUNCTION DOC]
    # Parameterized test: weight-only quantization of matmul.
    # Parameters: target_opset (XLA or StableHLO) and enable_per_channel_quantization. Asserts signature,
    # XLA dot op for XLA or call-module op for StableHLO, and size ratio under 0.3.
    input_shape = (1, 512)

    self._create_matmul_model(
        input_shape=input_shape,
        weight_shape=(512, 2),
        saved_model_path=self._input_saved_model_path,
    )

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
        enable_per_channel_quantization=enable_per_channel_quantization,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def

    if target_opset == quant_opts_pb2.XLA:
      self.assertTrue(self._contains_op(output_graphdef, 'XlaDotV2'))
    elif target_opset == quant_opts_pb2.STABLEHLO:
      # This is to verify the invocation of StableHLO quantizer works. More
      # thorough functional tests are in StableHLO quantizer directory.
      self.assertTrue(self._contains_op(output_graphdef, 'XlaCallModule'))

    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.3,
    )

  @parameterized.named_parameters(
      ('to_xla_per_tensor', quant_opts_pb2.XLA, False),
      ('stablehlo_per_channel', quant_opts_pb2.STABLEHLO, True),
      # TODO: b/289761265 - [Converter Component][TF-Quantizer] Improve Weight-
      # only Quantization
      # Enable this back once new weight-only quantizer is supported for per-
      # channel quantization.
      # ('to_xla_per_channel', quant_opts_pb2.XLA, True),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_conv_model(
      self,
      target_opset: quant_opts_pb2.OpSet,
      enable_per_channel_quantization: bool,
  ):
    # [FUNCTION DOC]
    # Parameterized test: weight-only quantization of a convolution with relu6.
    # Parameters: target_opset and enable_per_channel_quantization. Asserts the signature, size ratio
    # under 0.31, per channel constants for XLA, the XLA conv or call-module op, and output closeness to
    # the float model with a small tolerance.
    input_shape = (1, 3, 4, 512)
    filter_shape = (2, 3, 512, 2)
    model = self._create_conv2d_model(
        input_shape=input_shape,
        filter_shape=filter_shape,
        has_bias=False,
        has_batch_norm=False,
        activation_fn=nn_ops.relu6,
    )
    saved_model_save.save(model, self._input_saved_model_path)

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
        enable_per_channel_quantization=enable_per_channel_quantization,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def

    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.31,
    )

    if enable_per_channel_quantization and target_opset == quant_opts_pb2.XLA:
      per_channel_size_attr = attr_value_pb2.AttrValue(
          list=attr_value_pb2.AttrValue.ListValue(
              shape=[
                  tensor_shape_pb2.TensorShapeProto(
                      dim=[
                          tensor_shape_pb2.TensorShapeProto.Dim(
                              size=filter_shape[-1]
                          )
                      ]
                  )
              ]
          )
      )
      self.assertTrue(
          self._contains_op(
              output_graphdef, 'Const', '_output_shapes', per_channel_size_attr
          )
      )
    if target_opset == quant_opts_pb2.XLA:
      self.assertTrue(self._contains_op(output_graphdef, 'XlaConvV2'))
    elif target_opset == quant_opts_pb2.STABLEHLO:
      # This is to verify the invocation of StableHLO quantizer works. More
      # thorough functional tests are in StableHLO quantizer directory.
      self.assertTrue(self._contains_op(output_graphdef, 'XlaCallModule'))

    input_tensor = array_ops.constant(
        np.random.uniform(low=0, high=0.1, size=input_shape),
        dtype=dtypes.float32,
    )
    original_output = model.conv(input_tensor)
    quantized_output = converted_model.signatures['serving_default'](
        input_tensor
    )

    threshold = 0.015 if enable_per_channel_quantization else 0.02
    self.assertAllClose(original_output, quantized_output, atol=threshold)

  @parameterized.named_parameters(
      ('to_xla_per_tensor', quant_opts_pb2.XLA, False),
      # TODO: b/289761265 - [Converter Component][TF-Quantizer] Improve Weight-
      # only Quantization
      # Enable this back once new weight-only quantizer is supported for per-
      # channel quantization.
      # ('to_xla_per_channel', quant_opts_pb2.XLA, True),
  )
  @test_util.run_in_graph_and_eager_modes
  def test_depthwise_conv2d_model(
      self,
      target_opset: quant_opts_pb2.OpSet,
      enable_per_channel_quantization: bool,
  ):
    # [FUNCTION DOC]
    # Parameterized test: weight-only quantization of a depthwise convolution (XLA, per tensor).
    # Parameters: target_opset and enable_per_channel_quantization. Asserts the signature, size ratio, optional
    # per-channel constants, and closeness to the float model with a loose tolerance.
    input_shape = (1, 3, 4, 512)
    filter_shape = (2, 3, 512, 2)
    strides = (1, 2, 2, 1)

    model = self._create_depthwise_conv2d_model(
        input_shape=input_shape, filter_shape=filter_shape, strides=strides
    )

    saved_model_save.save(model, self._input_saved_model_path)

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
        enable_per_channel_quantization=enable_per_channel_quantization,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def

    # Due to other meta data, the compression is not exactly 1/4.
    size_threshold = 0.5 if enable_per_channel_quantization else 0.33
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        size_threshold,
    )

    if enable_per_channel_quantization:
      per_channel_size_attr = attr_value_pb2.AttrValue(
          list=attr_value_pb2.AttrValue.ListValue(
              shape=[
                  tensor_shape_pb2.TensorShapeProto(
                      dim=[
                          tensor_shape_pb2.TensorShapeProto.Dim(
                              size=filter_shape[2] * filter_shape[3]
                          ),
                      ]
                  )
              ]
          )
      )
      self.assertTrue(
          self._contains_op(
              output_graphdef, 'Const', '_output_shapes', per_channel_size_attr
          )
      )

    input_tensor = array_ops.constant(
        np.random.uniform(low=-0.1, high=0.1, size=input_shape),
        dtype=dtypes.float32,
    )
    original_output = model.depthwise_conv(input_tensor)
    quantized_output = converted_model.signatures['serving_default'](
        input_tensor
    )

    threshold = 0.68 if enable_per_channel_quantization else 1.3
    self.assertAllClose(original_output, quantized_output, atol=threshold)

  @parameterized.named_parameters(
      ('to_tf_use_constant', quant_opts_pb2.TF, False),
      ('to_xla_use_constant', quant_opts_pb2.XLA, False),
      (
          'to_uniform_quantized_use_constant',
          quant_opts_pb2.UNIFORM_QUANTIZED,
          False,
      ),
      ('to_tf_use_variable', quant_opts_pb2.TF, True),
      ('to_xla_use_variable', quant_opts_pb2.XLA, True),
      (
          'to_uniform_quantized_use_variable',
          quant_opts_pb2.UNIFORM_QUANTIZED,
          True,
      ),
  )
  @test_util.run_v2_only
  def test_gather_model(
      self, target_opset: quant_opts_pb2.OpSet, use_variable: bool
  ):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the size ratio is computed from the base-class input and output paths, not the temporary
    # directories where this test saved and wrote the model, so it does not measure the model under test.
    # Parameterized test (eager only): weight-only quantization of gather models.
    # Parameters: target_opset and use_variable. Saves the model into its own temporary directories. For TF
    # and uniform quantized op sets, asserts a value error that weight-only is unsupported and returns early.
    # For XLA, quantizes and asserts the signature and a size ratio under 0.3.
    input_type = dtypes.int64
    model = self._create_gather_model(input_type, use_variable)
    input_saved_model_path = self.create_tempdir('input').full_path
    saved_model_save.save(model, input_saved_model_path)

    tags = {tag_constants.SERVING}
    output_directory = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
    )

    if target_opset != quant_opts_pb2.XLA:
      # Uniform quantized opset is not supported for weight-only
      with self.assertRaisesRegex(
          ValueError, 'TF/Uniform quantized opset does not support weight-only.'
      ):
        converted_model = quantize_model.quantize(
            input_saved_model_path,
            output_directory,
            quantization_options,
        )
      return

    else:
      converted_model = quantize_model.quantize(
          input_saved_model_path,
          output_directory,
          quantization_options,
      )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )
    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.3,
    )

  @parameterized.named_parameters(
      ('to_tf_with_int32_input_type', dtypes.int32, quant_opts_pb2.TF),
      ('to_xla_with_int32_input_type', dtypes.int32, quant_opts_pb2.XLA),
      ('to_xla_with_int64_input_type', dtypes.int64, quant_opts_pb2.XLA),
      (
          'to_uq_with_int32_input_type',
          dtypes.int32,
          quant_opts_pb2.UNIFORM_QUANTIZED,
      ),
  )
  @test_util.run_v2_only
  def test_gather_and_conv_model(
      self, input_type: dtypes, target_opset: quant_opts_pb2.OpSet
  ):
    # [FUNCTION DOC]
    # Parameterized test (eager only): weight-only quantization of gather plus conv.
    # Parameters: input_type and target_opset. For non-XLA op sets asserts a value error that weight-only is
    # not supported and returns. For XLA asserts the signature, an XLA conv op and a size ratio under one third.
    model = self._create_simple_gather_and_conv_model(
        input_type, filter_shape=(2, 3, 3, 1024)
    )
    saved_model_save.save(model, self._input_saved_model_path)

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
    )

    if target_opset != quant_opts_pb2.XLA:
      # Uniform quantized opset is not supported for weight-only
      with self.assertRaisesRegex(
          ValueError, 'TF/Uniform quantized opset does not support weight-only.'
      ):
        converted_model = quantize_model.quantize(
            self._input_saved_model_path,
            self._output_saved_model_path,
            quantization_options,
        )
      return
    else:
      converted_model = quantize_model.quantize(
          self._input_saved_model_path,
          self._output_saved_model_path,
          quantization_options,
      )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    self.assertTrue(self._contains_op(output_graphdef, 'XlaConvV2'))
    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        1 / 3,
    )

  @test_util.run_in_graph_and_eager_modes
  def test_function_alias_preserved(self):
    # [FUNCTION DOC]
    # Test: function aliases are preserved by weight-only quantization.
    # Saves a gather and conv model with an alias for its main function, quantizes to XLA with the weight
    # threshold lowered to 1, and asserts the alias remains and the aliased function contains an int8 constant.
    # Prepare test model
    function_alias = 'conv_func'
    tags = {tag_constants.SERVING}
    input_type, filter_shape = dtypes.int64, (2, 3, 3, 2)
    model = self._create_simple_gather_and_conv_model(input_type, filter_shape)
    save_opts = save_options.SaveOptions(
        function_aliases={function_alias: model.model}
    )
    signatures = {
        'serving_default': model.model.get_concrete_function(),
    }
    saved_model_save.save(
        model, self._input_saved_model_path, signatures, save_opts
    )

    # Quantize the model
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=quant_opts_pb2.QuantizationMethod.PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.XLA,
        min_num_elements_for_weights=1,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    # Check if function alias is preserved
    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    meta_graph_def = output_loader.get_meta_graph_def_from_tags(tags)
    function_aliases = meta_graph_def.meta_info_def.function_aliases
    self.assertNotEmpty(function_aliases)
    self.assertCountEqual(function_aliases.values(), {function_alias})

    # Test that the aliased function contains a quantized op.
    for func_name, alias in function_aliases.items():
      if alias == function_alias:
        for func in meta_graph_def.graph_def.library.function:
          if func.signature.name == func_name:
            self.assertTrue(
                self._contains_op_with_name_and_attribute(
                    func.node_def,
                    op_name='Const',
                    attr_name='dtype',
                    attr_val=attr_value_pb2.AttrValue(type=types_pb2.DT_INT8),
                )
            )

  @test_util.run_v2_only
  def test_while_op_model(
      self,
  ):
    # [FUNCTION DOC]
    # Test (eager only): weight-only quantization of a model with a while loop.
    # Asserts the signature, an int8 XLA conv outside the loop and a still-float conv inside the loop body.
    model = self._create_while_model()
    saved_model_save.save(model, self._input_saved_model_path)

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.XLA,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    loader = saved_model_loader.SavedModelLoader(self._output_saved_model_path)
    output_graphdef = loader.get_meta_graph_def_from_tags(tags).graph_def

    # Convolution ouside the while op is quantized.
    self.assertTrue(
        self._contains_op(
            output_graphdef,
            op_name='XlaConvV2',
            attr_name='RhsT',
            attr_val=attr_value_pb2.AttrValue(type=types_pb2.DT_INT8),
        )
    )
    # TODO: b/294783597 - [Converter][TF-Quantizer] Support quantization for the
    # ops in the while op body for both SRQ and WO
    # Convolution inside the while op is not quantized.
    self.assertTrue(
        self._contains_op(
            output_graphdef,
            op_name='Conv2D',
            attr_name='T',
            attr_val=attr_value_pb2.AttrValue(type=types_pb2.DT_FLOAT),
        )
    )


class DebuggerTest(quantize_model_test_base.QuantizedModelTest):
  # [CLASS DOC]
  # Tests for the quantization debugger (dumping tensors of the original and quantized models for a
  # single conv layer or the whole model). Provides helpers to run a saved model in a TF1 session and to
  # read dumped tensor record files.

  def _run_model_in_sess(self, model_dir, tags, signature_key, sample_inputs):
    # [FUNCTION DOC]
    # HELPER: runs a saved model in a graph-mode session for the debugger tests.
    # Loads the model with the given tags from model_dir, finds the signature, and for each sample feeds
    # the inputs and collects the first output. Needed because the tensor dump op works only in graph mode.
    # Inputs: model_dir, tags, signature_key and sample_inputs (list of mappings from input key to value).
    # Returns a list of output arrays (one per sample).
    with tensorflow.compat.v1.Session(graph=tensorflow.Graph()) as sess:
      meta_graph = saved_model_loader.load(sess, tags, export_dir=model_dir)
      signature_def = meta_graph.signature_def[signature_key]

      # DumpTensorOp only works in graph mode.
      # Execute the model using session to run DumpTensorOp.
      output_tensor_names = [
          output_tensor_info.name
          for output_tensor_info in signature_def.outputs.values()
      ]

      output_values = []
      for sample_input in sample_inputs:
        feed_dict = {}
        for input_key, input_value in sample_input.items():
          input_tensor_name = signature_def.inputs[input_key].name
          feed_dict[input_tensor_name] = input_value

        # Obtain the output of the model.
        output_values.append(
            sess.run(output_tensor_names, feed_dict=feed_dict)[0]
        )
    return output_values

  def _read_tensor_array_file(self, file_path):
    # [FUNCTION DOC]
    # HELPER: reads dumped tensors for the debugger tests. Opens a ZLIB-compressed record file, parses each
    # record as a tensor protocol buffer, converts to arrays and returns them as one stacked array.
    # Input: file_path of the record file.
    tensor_protos = []
    for raw_record in tf_record.tf_record_iterator(file_path, options='ZLIB'):
      tensor_protos.append(
          tensorflow.make_ndarray(tensor_pb2.TensorProto.FromString(raw_record))
      )
    return np.array(tensor_protos)

  @parameterized.named_parameters(
      {
          'testcase_name': 'none',
          'activation_fn': None,
          'has_bias': False,
      },
      {
          'testcase_name': 'relu',
          'activation_fn': nn_ops.relu,
          'has_bias': False,
      },
      {
          'testcase_name': 'with_bias',
          'activation_fn': None,
          'has_bias': True,
      },
      {
          'testcase_name': 'with_bias_and_relu',
          'activation_fn': nn_ops.relu,
          'has_bias': True,
      },
  )
  def test_conv2d_ptq_model_whole_model_verify(self, activation_fn, has_bias):
    # [FUNCTION DOC]
    # Parameterized test: whole-model debugger for a conv model (activation and bias variants).
    # Parameters: activation_fn and has_bias. Quantizes to XLA with the whole-model debugger type, an
    # unquantized dump model path and a log directory. For both the unquantized dump model and the quantized
    # model it runs the model on two samples, reads the dumped tensor file, asserts it matches the model
    # outputs, and checks the quant unit file (node name Conv2D and function name pattern).
    # Side effects: temporary directories for dumps and logs.
    input_shape = [None, None, None, 3]
    filter_shape = [2, 3, 3, 2]

    model = self._create_conv2d_model(
        input_shape,
        filter_shape,
        activation_fn=activation_fn,
        has_bias=has_bias,
    )
    saved_model_save.save(model, self._input_saved_model_path)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: representative data generator. Yields eight samples mapping input_tensor to random
      # float32 values in 0 to 150 with shape 1,3,4,3.
      for _ in range(8):
        yield {
            'input_tensor': ops.convert_to_tensor(
                np.random.uniform(low=0, high=150, size=(1, 3, 4, 3)).astype(
                    'f4'
                )
            ),
        }

    tags = {tag_constants.SERVING}

    unquantized_dump_model_path = self.create_tempdir().full_path
    log_dir_path = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        op_set=quant_opts_pb2.XLA,
        debugger_config=_DebuggerConfig(
            debugger_type=_DebuggerConfig.DebuggerType.DEBUGGER_TYPE_WHOLE_MODEL,
            unquantized_dump_model_path=unquantized_dump_model_path,
            log_dir_path=log_dir_path,
        ),
        tags=tags,
        signature_keys=['serving_default'],
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen(),
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    sample_inputs = [
        {'input_tensor': np.random.uniform(low=0, high=1, size=(16, 3, 4, 3))},
        {'input_tensor': np.random.uniform(low=0, high=1, size=(16, 3, 4, 3))},
    ]

    # Check if output of the model and value saved by DumpTensorOp matches.
    # Verify for both unquantized model and quantized model.
    for model_path, file_name in [
        [unquantized_dump_model_path, 'unquantized_tensor_data.pb'],
        [self._output_saved_model_path, 'quantized_tensor_data.pb'],
    ]:
      output_values = self._run_model_in_sess(
          model_path, tags, 'serving_default', sample_inputs
      )

      # Find the dump file and parse it.
      folder = os.path.join(log_dir_path, os.listdir(log_dir_path)[0])
      dump_file_path = os.path.join(log_dir_path, folder, file_name)
      dump_file_numpy = self._read_tensor_array_file(dump_file_path)

      # Since the model only has one conv2d and its output is directly used as
      # the output of the model, output of the model and conv2d's dump value
      # should be the same.
      self.assertAllClose(output_values, dump_file_numpy)

      # Verify if quant_unit.pb file was created correctly.
      quant_unit_file_path = os.path.join(log_dir_path, folder, 'quant_unit.pb')

      quant_unit = (
          quant_opts_pb2.UnitWiseQuantizationSpec.QuantizationUnit.FromString(
              open(quant_unit_file_path, 'rb').read()
          )
      )

      self.assertEqual(quant_unit.node_name, 'Conv2D')
      self.assertRegex(quant_unit.func_name, r'^__inference_conv_\d+')

  @parameterized.parameters(
      testing.parameter_combinations([{
          'activation_fn': [None, nn_ops.relu, nn_ops.relu6],
          'has_bias': [True, False],
          'debugger_type': [
              _DebuggerConfig.DEBUGGER_TYPE_INT_PER_LAYER,
              _DebuggerConfig.DEBUGGER_TYPE_FLOAT_PER_LAYER,
          ],
          'target_opset': [quant_opts_pb2.XLA, quant_opts_pb2.STABLEHLO],
      }])
  )
  def test_conv2d_ptq_model_per_layer_verify(
      self,
      activation_fn: Optional[ops.Operation],
      has_bias: bool,
      debugger_type: _DebuggerConfig.DebuggerType,
      target_opset: quant_opts_pb2.OpSet,
  ):
    # [FUNCTION DOC]
    # Parameterized test: per-layer debugger (integer or float per layer) for XLA and StableHLO op sets.
    # Parameters: activation_fn, has_bias, debugger_type, target_opset. For StableHLO the dynamic spatial
    # dimensions are made static. Quantizes with debugger config, runs original and output models on
    # samples, reads the dump files, asserts unquantized dumps equal the original outputs, and that
    # output of the debugging model equals the quantized dump (int per layer) or the original output
    # (float per layer). Checks the quant unit file: node Conv2D for XLA, or an empty node name and a
    # composite function pattern for StableHLO.
    # TODO: b/326114903 - Support dynamic input dimensions after 0th rank in
    # op_set=STABLEHLO.
    input_shape_dynamic = target_opset != quant_opts_pb2.STABLEHLO
    concrete_input_shape = [None, 3, 4, 3]
    input_shape = (
        [None, None, None, 3] if input_shape_dynamic else concrete_input_shape
    )
    filter_shape = [2, 3, 3, 2]

    model = self._create_conv2d_model(
        input_shape,
        filter_shape,
        activation_fn=activation_fn,
        has_bias=has_bias,
    )
    saved_model_save.save(model, self._input_saved_model_path)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: representative data generator. Yields eight samples mapping input_tensor to random float32
      # values in 0 to 150 with batch size one and the concrete spatial shape.
      data_input_size = [1] + concrete_input_shape[1:]
      for _ in range(8):
        yield {
            'input_tensor': ops.convert_to_tensor(
                np.random.uniform(low=0, high=150, size=data_input_size).astype(
                    'f4'
                )
            ),
        }

    tags = {tag_constants.SERVING}

    log_dir_path = self.create_tempdir().full_path

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        op_set=target_opset,
        debugger_config=_DebuggerConfig(
            debugger_type=debugger_type,
            log_dir_path=log_dir_path,
        ),
        tags=tags,
        signature_keys=['serving_default'],
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen(),
    )

    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    sample_input_size = [16] + concrete_input_shape[1:]
    sample_inputs = [
        {
            'input_tensor': np.random.uniform(
                low=0, high=1, size=sample_input_size
            )
        },
        {
            'input_tensor': np.random.uniform(
                low=0, high=1, size=sample_input_size
            )
        },
    ]

    output_value_from_original_model = self._run_model_in_sess(
        self._input_saved_model_path, tags, 'serving_default', sample_inputs
    )
    output_value_from_debugging_model = self._run_model_in_sess(
        self._output_saved_model_path, tags, 'serving_default', sample_inputs
    )

    # Find the both quantized and unquantized dump file.
    folder = os.path.join(log_dir_path, os.listdir(log_dir_path)[0])
    unquantized_dump_file_path = os.path.join(
        log_dir_path, folder, 'unquantized_tensor_data.pb'
    )
    quantized_dump_file_path = os.path.join(
        log_dir_path, folder, 'quantized_tensor_data.pb'
    )

    unquantized_dump_file_numpy = self._read_tensor_array_file(
        unquantized_dump_file_path
    )
    quantized_dump_file_numpy = self._read_tensor_array_file(
        quantized_dump_file_path
    )

    # Since the model only has one conv2d and its output is directly used as
    # the output of the model, output of the model and conv2d's dump value
    # should be the same.
    self.assertAllClose(
        output_value_from_original_model, unquantized_dump_file_numpy
    )
    # The output_value_from_debugging_model of DEBUGGER_TYPE_INT_PER_LAYER is
    # a quantized value, while for DEBUGGER_TYPE_FLOAT_PER_LAYER, it's an
    # unquantized value. Therefore there are different verifications for the
    # output value.
    if debugger_type == _DebuggerConfig.DEBUGGER_TYPE_INT_PER_LAYER:
      self.assertAllClose(
          output_value_from_debugging_model, quantized_dump_file_numpy
      )
    else:  # debugger_type == _DebuggerConfig.DEBUGGER_TYPE_FLOAT_PER_LAYER:
      self.assertAllClose(
          output_value_from_debugging_model, output_value_from_original_model
      )

    # Verify if quant_unit.pb file was created correctly.
    quant_unit_file_path = os.path.join(log_dir_path, folder, 'quant_unit.pb')
    quant_unit = (
        quant_opts_pb2.UnitWiseQuantizationSpec.QuantizationUnit.FromString(
            open(quant_unit_file_path, 'rb').read()
        )
    )

    if target_opset == quant_opts_pb2.XLA:
      self.assertEqual(quant_unit.node_name, 'Conv2D')
      self.assertRegex(quant_unit.func_name, r'^__inference_conv_\d+')
    elif target_opset == quant_opts_pb2.STABLEHLO:
      self.assertEqual(quant_unit.node_name, '_empty_node')
      self.assertRegex(
          quant_unit.func_name, r'^composite_conv_([a-zA-Z_0-9]+_)*fn_\d+'
      )
    else:
      assert False, f'Please add assertion for the op_set: {target_opset}.'


@test_util.run_all_in_graph_and_eager_modes
class CalibrationOptionsTest(quantize_model_test_base.QuantizedModelTest):
  # [CLASS DOC]
  # Tests for the CalibrationOptions protocol buffer: calibration method selection and default
  # parameters. Runs all tests in both graph and eager modes.
  """Test cases regarding the use of CalibrationOptions proto.

  Run all tests cases in both the graph mode (default in TF1) and the eager mode
  (default in TF2) to ensure support for when TF2 is disabled.
  """

  @parameterized.parameters(
      testing.parameter_combinations([{
          'target_opset': [
              quant_opts_pb2.TF,
              quant_opts_pb2.XLA,
              quant_opts_pb2.UNIFORM_QUANTIZED,
          ],
          'calibration_options': [
              stablehlo_quant_config_pb2.CalibrationOptions(
                  calibration_method=_CalibrationMethod.CALIBRATION_METHOD_MIN_MAX
              ),
              stablehlo_quant_config_pb2.CalibrationOptions(
                  calibration_method=_CalibrationMethod.CALIBRATION_METHOD_AVERAGE_MIN_MAX
              ),
              stablehlo_quant_config_pb2.CalibrationOptions(
                  calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_PERCENTILE,
                  calibration_parameters=stablehlo_quant_config_pb2.CalibrationOptions.CalibrationParameters(
                      num_bins=32,
                  ),
              ),
              stablehlo_quant_config_pb2.CalibrationOptions(
                  calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE,
                  calibration_parameters=stablehlo_quant_config_pb2.CalibrationOptions.CalibrationParameters(
                      num_bins=32,
                  ),
              ),
              stablehlo_quant_config_pb2.CalibrationOptions(
                  calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY,
                  calibration_parameters=stablehlo_quant_config_pb2.CalibrationOptions.CalibrationParameters(
                      num_bins=32,
                  ),
              ),
              stablehlo_quant_config_pb2.CalibrationOptions(
                  calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_SYMMETRIC,
                  calibration_parameters=stablehlo_quant_config_pb2.CalibrationOptions.CalibrationParameters(
                      num_bins=32,
                  ),
              ),
          ],
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_conv_ptq_model_by_calibration_options(
      self,
      target_opset: quant_opts_pb2.OpSet,
      calibration_options: stablehlo_quant_config_pb2.CalibrationOptions,
  ):
    # [FUNCTION DOC]
    # KNOWN ISSUE: per-channel is hard coded off, so the per-channel branch is dead code, and the
    # per-channel op names generator is used twice.
    # Parameterized test: PTQ of a conv model with each calibration method and op set.
    # Parameters: target_opset (TF, XLA, uniform quantized) and calibration_options (min-max, average
    # min-max, and four histogram methods with 32 bins). Builds a conv model with bias, batch norm and
    # relu6, calibrates with ten random samples, quantizes with overwriting, and asserts the signature,
    # closeness (tolerance 0.1) and op expectations per op set (no fused batch norm).
    has_bias = True
    has_batch_norm = True
    activation_fn = nn_ops.relu6
    enable_per_channel_quantization = False

    input_shape = [1, 3, 4, 3]
    filter_shape = [2, 3, 3, 2]

    model = self._create_conv2d_model(
        input_shape, filter_shape, has_bias, has_batch_norm, activation_fn
    )
    saved_model_save.save(model, self._input_saved_model_path)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: representative data generator. Yields ten samples mapping input_tensor to random float32
      # values in 0 to 10 with shape 1,3,4,3.
      for _ in range(10):
        yield {
            'input_tensor': ops.convert_to_tensor(
                np.random.uniform(low=0, high=10, size=(1, 3, 4, 3)).astype(
                    'f4'
                )
            ),
        }

    tags = {tag_constants.SERVING}

    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
        enable_per_channel_quantization=enable_per_channel_quantization,
        calibration_options=calibration_options,
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen(),
        overwrite_output_directory=True,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {'serving_default'}
    )

    sample_input = ops.convert_to_tensor(
        np.random.uniform(low=0, high=10, size=(1, 3, 4, 3)).astype('f4')
    )
    expected_outputs = model.conv(sample_input)
    got_outputs = converted_model.signatures['serving_default'](sample_input)
    self.assertAllClose(expected_outputs, got_outputs, atol=1e-1)

    output_loader = saved_model_loader.SavedModelLoader(
        self._output_saved_model_path
    )
    output_graphdef = output_loader.get_meta_graph_def_from_tags(tags).graph_def
    if target_opset == quant_opts_pb2.XLA:
      self.assertTrue(self._contains_op(output_graphdef, 'XlaConvV2'))
    elif target_opset == quant_opts_pb2.UNIFORM_QUANTIZED:
      self.assertTrue(
          self._contains_op(output_graphdef, 'UniformQuantizedConvolution')
      )
      if enable_per_channel_quantization:
        quantized_axis = 3
        quantized_dim_size_attr = attr_value_pb2.AttrValue(
            list=attr_value_pb2.AttrValue.ListValue(
                shape=[
                    tensor_shape_pb2.TensorShapeProto(
                        dim=[
                            tensor_shape_pb2.TensorShapeProto.Dim(
                                size=filter_shape[quantized_axis]
                            )
                        ]
                    )
                ]
            )
        )
      else:
        quantized_axis = -1
        # Empty dimension. Per-tensor quantization has singular channel.
        quantized_dim_size_attr = attr_value_pb2.AttrValue(
            list=attr_value_pb2.AttrValue.ListValue(
                shape=[tensor_shape_pb2.TensorShapeProto()]
            )
        )
      quantized_axis_attr = attr_value_pb2.AttrValue(i=quantized_axis)
      self.assertEqual(
          self._count_ops(
              output_graphdef,
              _PER_CHANNEL_QUANTIZED_OPS,
              'rhs_quantization_axis',
              quantized_axis_attr,
          ),
          self._count_ops(output_graphdef, _PER_CHANNEL_QUANTIZED_OPS),
      )
      self.assertEqual(
          self._count_ops(
              output_graphdef,
              _PER_CHANNEL_OP_NAMES,
              '_output_shapes',
              quantized_dim_size_attr,
              get_op_name=True,
          ),
          self._count_ops(
              output_graphdef,
              _PER_CHANNEL_OP_NAMES,
              get_op_name=True,
          ),
      )
      self.assertFalse(self._contains_op(output_graphdef, 'Conv2D'))
    else:
      self.assertTrue(self._contains_quantized_function_call(output_graphdef))
    self.assertFalse(self._contains_op(output_graphdef, 'FusedBatchNormV3'))

  @parameterized.named_parameters(
      {
          'testcase_name': 'with_calibration_method_unspecified',
          'calibration_options': stablehlo_quant_config_pb2.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_UNSPECIFIED
          ),
          'default_calibration_options': stablehlo_quant_config_pb2.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_MIN_MAX
          ),
      },
      {
          'testcase_name': 'with_histogram_percentile',
          'calibration_options': stablehlo_quant_config_pb2.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_PERCENTILE
          ),
          'default_calibration_options': stablehlo_quant_config_pb2.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_PERCENTILE,
              calibration_parameters=stablehlo_quant_config_pb2.CalibrationOptions.CalibrationParameters(
                  num_bins=512,
                  min_percentile=0.001,
                  max_percentile=99.999,
              ),
          ),
      },
      {
          'testcase_name': 'with_histogram_mse_bruteforce',
          'calibration_options': stablehlo_quant_config_pb2.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE
          ),
          'default_calibration_options': stablehlo_quant_config_pb2.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE,
              calibration_parameters=stablehlo_quant_config_pb2.CalibrationOptions.CalibrationParameters(
                  num_bins=512
              ),
          ),
      },
      {
          'testcase_name': 'with_histogram_mse_max_frequency',
          'calibration_options': stablehlo_quant_config_pb2.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY
          ),
          'default_calibration_options': stablehlo_quant_config_pb2.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY,
              calibration_parameters=stablehlo_quant_config_pb2.CalibrationOptions.CalibrationParameters(
                  num_bins=512
              ),
          ),
      },
      {
          'testcase_name': 'with_histogram_mse_symmetric',
          'calibration_options': stablehlo_quant_config_pb2.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_SYMMETRIC
          ),
          'default_calibration_options': stablehlo_quant_config_pb2.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_SYMMETRIC,
              calibration_parameters=stablehlo_quant_config_pb2.CalibrationOptions.CalibrationParameters(
                  num_bins=512
              ),
          ),
      },
  )
  @test_util.run_in_graph_and_eager_modes
  def test_default_calibration_options(
      self,
      calibration_options: stablehlo_quant_config_pb2.CalibrationOptions,
      default_calibration_options: stablehlo_quant_config_pb2.CalibrationOptions,
  ):
    # [FUNCTION DOC]
    # Parameterized test: default calibration parameters are filled in.
    # Parameters: calibration_options (given) and default_calibration_options (expected). Calls the private
    # routines to populate component specs and calibration options on a static range options message, then
    # asserts the calibration method, bin count and minimum and maximum percentile equal the expected values.
    # Modifies only the local options message.
    quant_opts = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        calibration_options=calibration_options,
    )
    quantize_model._populate_quantization_component_spec(
        quant_opts.quantization_method
    )
    quantize_model._populate_calibration_options(quant_opts)

    self.assertEqual(
        quant_opts.calibration_options.calibration_method,
        default_calibration_options.calibration_method,
    )
    self.assertEqual(
        quant_opts.calibration_options.calibration_parameters.num_bins,
        default_calibration_options.calibration_parameters.num_bins,
    )
    self.assertEqual(
        quant_opts.calibration_options.calibration_parameters.min_percentile,
        default_calibration_options.calibration_parameters.min_percentile,
    )
    self.assertEqual(
        quant_opts.calibration_options.calibration_parameters.max_percentile,
        default_calibration_options.calibration_parameters.max_percentile,
    )

  @parameterized.named_parameters(
      {
          'testcase_name': 'none',
          'activation_fn': None,
          'has_bias': False,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
      },
      {
          'testcase_name': 'relu',
          'activation_fn': nn_ops.relu,
          'has_bias': False,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
      },
      {
          'testcase_name': 'bn',
          'activation_fn': None,
          'has_bias': False,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.TF,
      },
      {
          'testcase_name': 'with_bias',
          'activation_fn': None,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
      },
      {
          'testcase_name': 'with_bias_and_relu',
          'activation_fn': nn_ops.relu,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.TF,
      },
      {
          'testcase_name': 'with_bias_and_bn_and_relu',
          'activation_fn': nn_ops.relu,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.TF,
      },
      {
          'testcase_name': 'with_bias_and_relu_to_xla',
          'activation_fn': nn_ops.relu,
          'has_bias': True,
          'has_batch_norm': False,
          'target_opset': quant_opts_pb2.XLA,
      },
      {
          'testcase_name': 'with_bias_and_bn_and_relu_to_xla',
          'activation_fn': nn_ops.relu,
          'has_bias': True,
          'has_batch_norm': True,
          'target_opset': quant_opts_pb2.XLA,
      },
  )
  @test_util.run_in_graph_and_eager_modes
  def test_conv_ptq_with_outlier_representative_data(
      self,
      activation_fn: Optional[ops.Operation],
      has_bias: bool,
      has_batch_norm: bool,
      target_opset: quant_opts_pb2.OpSet,
  ):
    # [FUNCTION DOC]
    # Parameterized test: average min-max calibration is more robust to outliers than min-max.
    # Parameters: activation_fn, has_bias, has_batch_norm, target_opset (TF or XLA). Calibrates with a first
    # sample containing extreme values (minus and plus 1000) and ten normal samples, quantizes once with
    # min-max and once with average min-max, compares each quantized output with the float output on a
    # sample and asserts the average min-max mean square error is smaller.
    input_shape = [1, 3, 4, 3]
    filter_shape = [2, 3, 3, 2]

    model = self._create_conv2d_model(
        input_shape, filter_shape, has_bias, has_batch_norm, activation_fn
    )
    saved_model_save.save(model, self._input_saved_model_path)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: representative data generator with an outlier. Yields one sample with two extreme values and
      # then ten normal random samples (0 to 10), each mapping input_tensor to a tensor of shape 1,3,4,3.
      outlier = np.random.uniform(low=0, high=10, size=(1, 3, 4, 3)).astype(
          'f4'
      )
      outlier[0][0][0][0:2] = [-1000, 1000]
      yield {'input_tensor': ops.convert_to_tensor(outlier)}
      for _ in range(10):
        yield {
            'input_tensor': ops.convert_to_tensor(
                np.random.uniform(low=0, high=10, size=(1, 3, 4, 3)).astype(
                    'f4'
                )
            ),
        }

    tags = {tag_constants.SERVING}

    quantization_options_min_max = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
        enable_per_channel_quantization=False,
        calibration_options=stablehlo_quant_config_pb2.CalibrationOptions(
            calibration_method=_CalibrationMethod.CALIBRATION_METHOD_MIN_MAX,
        ),
    )

    converted_model_min_max = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options_min_max,
        representative_dataset=data_gen(),
        overwrite_output_directory=True,
    )

    self.assertIsNotNone(converted_model_min_max)
    self.assertCountEqual(
        converted_model_min_max.signatures._signatures.keys(),
        {'serving_default'},
    )

    quantization_options_average_min_max = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=target_opset,
        enable_per_channel_quantization=False,
        calibration_options=stablehlo_quant_config_pb2.CalibrationOptions(
            calibration_method=_CalibrationMethod.CALIBRATION_METHOD_AVERAGE_MIN_MAX,
        ),
    )

    converted_model_average_min_max = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options_average_min_max,
        representative_dataset=data_gen(),
        overwrite_output_directory=True,
    )

    self.assertIsNotNone(converted_model_average_min_max)
    self.assertCountEqual(
        converted_model_average_min_max.signatures._signatures.keys(),
        {'serving_default'},
    )

    sample_input = ops.convert_to_tensor(
        np.random.uniform(low=0, high=10, size=(1, 3, 4, 3)).astype('f4')
    )

    original_output = model.conv(sample_input)['output']
    min_max_output = converted_model_min_max.signatures['serving_default'](
        input_tensor=sample_input
    )['output']
    average_min_max_output = converted_model_average_min_max.signatures[
        'serving_default'
    ](input_tensor=sample_input)['output']

    def get_mean_square_error(x, y):
      # [FUNCTION DOC]
      # HELPER: computes the mean squared error between two tensors, evaluating it either eagerly or in a
      # session depending on the mode. Inputs: x and y tensors. Returns a number.
      ret = tensorflow.reduce_mean(tensorflow.square(tensorflow.subtract(x, y)))
      try:
        ret = ret.numpy()
      except AttributeError:
        ret = ret.eval()
      return ret

    min_max_mse = get_mean_square_error(original_output, min_max_output)
    average_min_max_mse = get_mean_square_error(
        original_output, average_min_max_output
    )

    self.assertLess(average_min_max_mse, min_max_mse)


class SelectiveQuantizationTest(quantize_model_test_base.QuantizedModelTest):
  # [CLASS DOC]
  # Tests for selective quantization through unit-wise quantization specs (opting ops in or out of
  # quantization), including validation errors and QAT, PTQ and gather cases.
  """Test cases regarding selective quantization."""

  def test_unitwise_spec_with_no_units(self):
    # [FUNCTION DOC]
    # Test: a unit-wise spec with an empty unit list is rejected.
    # Saves a QAT einsum model, supplies a spec with no units and asserts a value error saying at least
    # one unit is required.
    _, y_shape, bias_shape, x_signature, y_signature = (
        self._prepare_sample_einsum_datashapes('abc,acd->abd')
    )
    model = self._create_einsum_model(
        'abc,acd->abd',
        y_shape,
        x_signature,
        y_signature,
        bias_shape,
        is_qat_model=True,
    )
    saved_model_save.save(
        model, self._input_saved_model_path, signatures=model.einsum_with_kernel
    )

    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=quant_opts_pb2.XLA,
        unit_wise_quantization_specs=[
            quant_opts_pb2.UnitWiseQuantizationSpec(
                unit=[],
                quantization_method=quant_opts_pb2.QuantizationMethod(
                    preset_method=quant_opts_pb2.QuantizationMethod.METHOD_NO_QUANTIZE
                ),
            )
        ],
    )

    with self.assertRaisesRegex(
        ValueError, 'UnitWiseQuantizationSpec must contain at least one unit.'
    ):
      quantize_model.quantize(
          self._input_saved_model_path,
          self._output_saved_model_path,
          quantization_options,
      )

  def test_unitwise_spec_missing_unit_info(self):
    # [FUNCTION DOC]
    # Test: a unit entry with neither op type nor node name is rejected.
    # Asserts a value error stating either the op type or the node name must be specified.
    _, y_shape, bias_shape, x_signature, y_signature = (
        self._prepare_sample_einsum_datashapes('abc,acd->abd')
    )
    model = self._create_einsum_model(
        'abc,acd->abd',
        y_shape,
        x_signature,
        y_signature,
        bias_shape,
        is_qat_model=True,
    )
    saved_model_save.save(
        model, self._input_saved_model_path, signatures=model.einsum_with_kernel
    )

    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=quant_opts_pb2.XLA,
        unit_wise_quantization_specs=[
            quant_opts_pb2.UnitWiseQuantizationSpec(
                unit=[
                    quant_opts_pb2.UnitWiseQuantizationSpec.QuantizationUnit(),
                ],
                quantization_method=quant_opts_pb2.QuantizationMethod(
                    preset_method=quant_opts_pb2.QuantizationMethod.METHOD_NO_QUANTIZE
                ),
            )
        ],
    )

    with self.assertRaisesRegex(
        ValueError, 'Either `op_type` or `node_name` must be specified.'
    ):
      quantize_model.quantize(
          self._input_saved_model_path,
          self._output_saved_model_path,
          quantization_options,
      )

  def test_unitwise_spec_unsupported_method(self):
    # [FUNCTION DOC]
    # Test: a unit-wise spec with an unsupported method is rejected.
    # Uses weight-only int8 for a Conv2D unit and asserts a value error that only the no-quantize method and
    # the same method as the top level are supported.
    _, y_shape, bias_shape, x_signature, y_signature = (
        self._prepare_sample_einsum_datashapes('abc,acd->abd')
    )
    model = self._create_einsum_model(
        'abc,acd->abd',
        y_shape,
        x_signature,
        y_signature,
        bias_shape,
        is_qat_model=True,
    )
    saved_model_save.save(
        model, self._input_saved_model_path, signatures=model.einsum_with_kernel
    )

    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=quant_opts_pb2.XLA,
        unit_wise_quantization_specs=[
            quant_opts_pb2.UnitWiseQuantizationSpec(
                unit=[
                    quant_opts_pb2.UnitWiseQuantizationSpec.QuantizationUnit(
                        op_type='Conv2D',
                    ),
                ],
                quantization_method=quant_opts_pb2.QuantizationMethod(
                    preset_method=quant_opts_pb2.QuantizationMethod.METHOD_STATIC_RANGE_WEIGHT_ONLY_INT8
                ),
            )
        ],
    )

    with self.assertRaisesRegex(
        ValueError,
        'Currently unit-wise quantization spec only supports NO_QUANTIZE and'
        ' same quantization method as the top-level',
    ):
      quantize_model.quantize(
          self._input_saved_model_path,
          self._output_saved_model_path,
          quantization_options,
      )

  def test_selective_quantization_qat(self):
    # [FUNCTION DOC]
    # Test: einsum can be excluded from quantization in a QAT model.
    # Specifies no-quantize for the einsum op, quantizes with XLA, asserts the signature, that an einsum
    # op remains and that no XLA dot op exists.
    _, y_shape, bias_shape, x_signature, y_signature = (
        self._prepare_sample_einsum_datashapes('abc,acd->abd')
    )
    model = self._create_einsum_model(
        'abc,acd->abd',
        y_shape,
        x_signature,
        y_signature,
        bias_shape,
        is_qat_model=True,
    )
    saved_model_save.save(
        model, self._input_saved_model_path, signatures=model.einsum_with_kernel
    )

    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=quant_opts_pb2.XLA,
        unit_wise_quantization_specs=[
            quant_opts_pb2.UnitWiseQuantizationSpec(
                unit=[
                    quant_opts_pb2.UnitWiseQuantizationSpec.QuantizationUnit(
                        op_type='Einsum',
                    ),
                ],
                quantization_method=quant_opts_pb2.QuantizationMethod(
                    preset_method=quant_opts_pb2.QuantizationMethod.METHOD_NO_QUANTIZE
                ),
            )
        ],
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {signature_key}
    )
    loader = saved_model_loader.SavedModelLoader(self._output_saved_model_path)
    graphdef = loader.get_meta_graph_def_from_tags(tags).graph_def
    # The Einsum ops shouldn't be quantized.
    self.assertTrue(self._contains_op(graphdef, 'Einsum'))
    self.assertFalse(self._contains_op(graphdef, 'XlaDotV2'))

  def test_selective_quantization_ptq(self):
    # [FUNCTION DOC]
    # Test: einsum can be excluded from quantization in a PTQ model.
    # Same as the QAT case but without fake quant, calibrating with ten seeded samples.
    x_shape, y_shape, bias_shape, x_signature, y_signature = (
        self._prepare_sample_einsum_datashapes('abc,acd->abd')
    )
    model = self._create_einsum_model(
        'abc,acd->abd',
        y_shape,
        x_signature,
        y_signature,
        bias_shape,
        is_qat_model=False,
    )
    saved_model_save.save(
        model, self._input_saved_model_path, signatures=model.einsum_with_kernel
    )

    signature_key = signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=[signature_key],
        op_set=quant_opts_pb2.XLA,
        unit_wise_quantization_specs=[
            quant_opts_pb2.UnitWiseQuantizationSpec(
                unit=[
                    quant_opts_pb2.UnitWiseQuantizationSpec.QuantizationUnit(
                        op_type='Einsum',
                    ),
                ],
                quantization_method=quant_opts_pb2.QuantizationMethod(
                    preset_method=quant_opts_pb2.QuantizationMethod.METHOD_NO_QUANTIZE
                ),
            )
        ],
    )

    rng = np.random.default_rng(seed=1234)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: representative data generator. Yields ten samples mapping x to uniform random float32
      # values in 0 to 1 using the test's seeded generator with the einsum input shape.
      for _ in range(10):
        yield {
            'x': rng.uniform(low=0.0, high=1.0, size=x_shape).astype(np.float32)
        }

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
        representative_dataset=data_gen(),
    )
    self.assertIsNotNone(converted_model)
    self.assertCountEqual(
        converted_model.signatures._signatures.keys(), {signature_key}
    )
    loader = saved_model_loader.SavedModelLoader(self._output_saved_model_path)
    graphdef = loader.get_meta_graph_def_from_tags(tags).graph_def
    # The Einsum ops shouldn't be quantized.
    self.assertTrue(self._contains_op(graphdef, 'Einsum'))
    self.assertFalse(self._contains_op(graphdef, 'XlaDotV2'))

  def test_selective_quantization_on_gather(
      self,
  ):
    # [FUNCTION DOC]
    # Test: gather can be excluded from quantization while the unquantized conv stays as is.
    # Saves a QAT gather and conv model, specifies no-quantize for the gather op, quantizes with XLA, asserts
    # a plain conv op remains and that the model size ratio of input over output is below 1.15, meaning
    # the gather weights were not quantized.
    input_type = dtypes.int32
    model = self._create_simple_gather_and_conv_model(
        input_type,
        filter_shape=(2, 3, 3, 1024),
        is_qat_model=True,
    )

    saved_model_save.save(model, self._input_saved_model_path)

    tags = {tag_constants.SERVING}
    quantization_options = quant_opts_pb2.QuantizationOptions(
        quantization_method=quant_opts_pb2.QuantizationMethod(
            preset_method=_PresetMethod.METHOD_STATIC_RANGE_INT8
        ),
        tags=tags,
        signature_keys=['serving_default'],
        op_set=quant_opts_pb2.XLA,
        unit_wise_quantization_specs=[
            quant_opts_pb2.UnitWiseQuantizationSpec(
                unit=[
                    quant_opts_pb2.UnitWiseQuantizationSpec.QuantizationUnit(
                        op_type='GatherV2',
                    ),
                ],
                quantization_method=quant_opts_pb2.QuantizationMethod(
                    preset_method=quant_opts_pb2.QuantizationMethod.METHOD_NO_QUANTIZE
                ),
            )
        ],
    )

    converted_model = quantize_model.quantize(
        self._input_saved_model_path,
        self._output_saved_model_path,
        quantization_options,
    )
    self.assertIsNotNone(converted_model)
    loader = saved_model_loader.SavedModelLoader(self._output_saved_model_path)
    graphdef = loader.get_meta_graph_def_from_tags(tags).graph_def
    # The Conv2D op shouldn't be quantized as it has no FakeQuant on input.
    self.assertTrue(self._contains_op(graphdef, 'Conv2D'))
    # If the Gather op is quantized, input_model_size / output_model_size > 2.
    self.assertLess(
        testing.get_size_ratio(
            self._input_saved_model_path, self._output_saved_model_path
        ),
        1.15,
    )


if __name__ == '__main__':
  test.main()
