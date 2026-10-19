# [MODULE DOC]
# Integration tests for the StableHLO quantize_model entry point (quantize a saved model).
# Contains three test classes: static range post-training quantization (matmul, same-scale ops,
# conv, einsum, gather, calibration reuse, deny-list specs, report file), calibration option
# variations, and weight-only quantization (matmul, conv, add, report file). Tests build small
# models using the shared base class, write representative datasets as record files, quantize
# through the public quantization function with a configuration message, reload the output and
# compare numerics, inspect the StableHLO text, and check the size ratio against the input.
# Defines a module constant alias for the calibration method enum. Runs the test main when
# executed as a script. No import-time registrations.
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
import os
import re
from typing import Mapping, Optional, Sequence

from absl.testing import parameterized
import numpy as np

from google.protobuf import text_format
from tensorflow.compiler.mlir.quantization.common.python import testing
from tensorflow.compiler.mlir.quantization.stablehlo import quantization_config_pb2 as qc
from tensorflow.compiler.mlir.quantization.stablehlo.python import quantization
from tensorflow.compiler.mlir.quantization.stablehlo.python.integration_test import quantize_model_test_base
from tensorflow.compiler.mlir.quantization.tensorflow.python import representative_dataset as repr_dataset
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.module import module
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.types import core

_CalibrationMethod = qc.CalibrationOptions.CalibrationMethod


# Test cases for Static Range Quantization.
# Tries to run all tests cases in both the graph mode (default in TF1) and the
# eager mode (default in TF2) to ensure support for when TF2 is disabled.
class StaticRangeQuantizationTest(quantize_model_test_base.QuantizedModelTest):
  # [CLASS DOC]
  # Tests of static range post-training quantization through the quantize saved model function.
  # Uses the shared base fixtures for temporary paths and model builders. Covers matmul and
  # batch matmul, same-scale ops, convolution, einsum, gather, calibration data reuse, deny-list
  # quantization specs and the quantization report file.

  @parameterized.parameters(
      testing.parameter_combinations([{
          'bias_fn': (
              None,
              nn_ops.bias_add,
          ),
          'activation_fn': (
              None,
              nn_ops.relu,
              nn_ops.relu6,
          ),
          'dim_sizes': (
              # tf.MatMul cases.
              ([None, 1024], [1024, 3]),  # dynamic batch dim.
              ([1, 1024], [1024, 3]),
              # tf.BatchMatMul cases.
              ([10, 1, 1024], [10, 1024, 3]),
              ([2, 3, 1, 1024], [2, 3, 1024, 3]),
          ),
          'merge_fusion_with_dequantize': (False, True),
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_matmul_ptq_model(
      self,
      bias_fn: Optional[ops.Operation],
      activation_fn: Optional[ops.Operation],
      dim_sizes: Sequence[int],
      merge_fusion_with_dequantize: bool,
  ):
    # [FUNCTION DOC]
    # Parameterized scenario (bias on or off, activation none, relu or relu6, four shape pairs
    # covering matmul and batch matmul with one dynamic batch case, merge fusion with dequantize
    # on or off). Builds and saves a float matmul model, writes 100 random representative samples
    # to a record file, quantizes with a static range preset (serving tag) and the fusion flag.
    # Expected: the output has only the serving default signature; the StableHLO has an int8
    # dot_general, an int32 add when bias is used; outputs match the float model within loose
    # tolerances; the saved size ratio is under 0.65. With merge fusion, relu6 and relu appear
    # explicitly as clamp and maximum right before return; without it neither pattern appears.
    lhs_dim_size, rhs_dim_size = dim_sizes
    input_shape = (*lhs_dim_size,)
    filter_shape = (*rhs_dim_size,)
    static_input_shape = [dim if dim is not None else 2 for dim in input_shape]
    model = self._create_matmul_model(
        input_shape,
        filter_shape,
        self._input_saved_model_path,
        bias_fn,
        activation_fn,
    )

    rng = np.random.default_rng(seed=42)
    input_data = ops.convert_to_tensor(
        rng.uniform(low=0.0, high=1.0, size=static_input_shape).astype(
            np.float32
        )
    )

    def data_gen() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: representative dataset generator for the test above.
      # Yields 100 samples, each a mapping from input_tensor to random uniform float32 data in
      # zero to one of the static input shape (dynamic dimensions replaced by two), drawn from the
      # shared seeded generator.
      for _ in range(100):
        yield {
            'input_tensor': rng.uniform(
                low=0.0, high=1.0, size=static_input_shape
            ).astype(np.float32)
        }

    dataset_path = self.create_tempfile('tfrecord').full_path
    path_map = {'serving_default': dataset_path}
    repr_dataset.TfRecordRepresentativeDatasetSaver(path_map).save(
        {'serving_default': data_gen()}
    )

    config = qc.QuantizationConfig(
        static_range_ptq_preset=qc.StaticRangePtqPreset(
            representative_datasets=[
                qc.RepresentativeDatasetConfig(
                    tf_record=qc.TfRecordFile(path=dataset_path)
                )
            ]
        ),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
        pipeline_config=qc.PipelineConfig(
            merge_fusion_with_dequantize=merge_fusion_with_dequantize
        ),
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    expected_outputs = model.matmul(input_data)

    root = load.load(self._output_saved_model_path)
    self.assertCountEqual(root.signatures.keys(), {'serving_default'})

    new_outputs = root.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(input_data)
    )
    module_str = self._extract_first_xla_call_module_op(
        self._output_saved_model_path
    )
    self.assertTrue(re.search('stablehlo.dot_general.*xi8>', module_str))
    if bias_fn:
      self.assertTrue(re.search('stablehlo.add.*xi32>', module_str))
    # Consider if there is a way to check if activation fusion is properly
    # done in MLIR level.
    # Tests that the quantized graph outputs similar values. The rtol and atol
    # values are arbitrary.
    self.assertAllClose(new_outputs, expected_outputs, rtol=0.3, atol=0.2)

    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.65,
    )

    if merge_fusion_with_dequantize:
      # Check activation functions are explicitly present.
      # If present the last op before return should be stablehlo.clamp for relu6
      # and stablehlo.maximum for relu.
      if activation_fn is nn_ops.relu6:
        self.assertRegex(module_str, r'stablehlo.clamp.*\n.*return')
      elif activation_fn is nn_ops.relu:
        self.assertRegex(module_str, r'stablehlo.maximum.*\n.*return')
    else:
      # Check activation functions are implicit.
      self.assertNotRegex(module_str, r'stablehlo.clamp.*\n.*return')
      self.assertNotRegex(module_str, r'stablehlo.maximum.*\n.*return')

  @parameterized.parameters(
      testing.parameter_combinations([{
          'same_scale_op': (
              'concatenate',
              'gather',
              'max_pool',
              'pad',
              'reshape',
              'select',
              'slice',
              'transpose',
          ),
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_matmul_and_same_scale_ptq_model(
      self,
      same_scale_op: str,
  ):
    # [FUNCTION DOC]
    # Parameterized over same-scale op names (concatenate, gather, max_pool, pad, reshape, select,
    # slice, transpose). Builds and saves a matmul followed by that op on a rank-4 input, creates
    # representative data, and quantizes with a static range preset.
    # Expected: only the serving default signature; outputs close to the float model (relative
    # tolerance 0.03, absolute 0.2); saved size ratio under 0.65.
    input_shape = (2, 3, 1, 1024)
    filter_shape = (2, 3, 1024, 3)
    static_input_shape = [dim if dim is not None else 2 for dim in input_shape]

    model = self._create_matmul_and_same_scale_model(
        input_shape,
        filter_shape,
        self._input_saved_model_path,
        same_scale_op,
    )

    rng = np.random.default_rng(seed=42)
    input_data = ops.convert_to_tensor(
        rng.uniform(low=0.0, high=1.0, size=static_input_shape).astype(
            np.float32
        )
    )

    def data_gen() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: representative dataset generator; yields 100 mappings from input_tensor to random
      # float32 data in zero to one with the static input shape, from the shared seeded generator.
      for _ in range(100):
        yield {
            'input_tensor': rng.uniform(
                low=0.0, high=1.0, size=static_input_shape
            ).astype(np.float32)
        }

    dataset_path = self.create_tempfile('tfrecord').full_path
    path_map = {'serving_default': dataset_path}
    repr_dataset.TfRecordRepresentativeDatasetSaver(path_map).save(
        {'serving_default': data_gen()}
    )

    config = qc.QuantizationConfig(
        static_range_ptq_preset=qc.StaticRangePtqPreset(
            representative_datasets=[
                qc.RepresentativeDatasetConfig(
                    tf_record=qc.TfRecordFile(path=dataset_path)
                )
            ]
        ),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    expected_outputs = model.matmul_and_same_scale(input_data)

    root = load.load(self._output_saved_model_path)
    self.assertCountEqual(root.signatures.keys(), {'serving_default'})

    new_outputs = root.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(input_data)
    )
    # Tests that the quantized graph outputs similar values. The rtol and atol
    # values are arbitrary.
    self.assertAllClose(new_outputs, expected_outputs, rtol=0.03, atol=0.2)

    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.65,
    )

  @parameterized.parameters(
      testing.parameter_combinations([{
          'same_scale_op': (
              'reshape',  # This corresponds to stablehlo.dynamic_reshape
              'slice',  # This corresponds to stablehlo.dynamic_slice.
              # TODO: b/326242075 - Support other same-scale ops.
          ),
          'dim_sizes': (([None, 1024], [1024, 3]),),
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_matmul_and_same_scale_ptq_model_dynamic(
      self,
      same_scale_op: str,
      dim_sizes: Sequence[int],
  ):
    # [FUNCTION DOC]
    # Same as the static same-scale test but with a dynamic batch dimension and only reshape and
    # slice (which map to dynamic reshape and dynamic slice in StableHLO). Quantizes with a static
    # range preset and checks the serving signature, output closeness (relative 0.03, absolute
    # 0.2) and a saved size ratio under 0.6.
    input_dim_size, filter_dim_size = dim_sizes
    input_shape = (*input_dim_size,)
    filter_shape = (*filter_dim_size,)
    static_input_shape = [dim if dim is not None else 2 for dim in input_shape]

    model = self._create_matmul_and_same_scale_model(
        input_shape,
        filter_shape,
        self._input_saved_model_path,
        same_scale_op,
    )

    rng = np.random.default_rng(seed=42)
    input_data = ops.convert_to_tensor(
        rng.uniform(low=0.0, high=1.0, size=static_input_shape).astype(
            np.float32
        )
    )

    def data_gen() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: representative dataset generator; yields 100 mappings from input_tensor to random
      # float32 data in zero to one with the static input shape.
      for _ in range(100):
        yield {
            'input_tensor': rng.uniform(
                low=0.0, high=1.0, size=static_input_shape
            ).astype(np.float32)
        }

    dataset_path = self.create_tempfile('tfrecord').full_path
    path_map = {'serving_default': dataset_path}
    repr_dataset.TfRecordRepresentativeDatasetSaver(path_map).save(
        {'serving_default': data_gen()}
    )

    config = qc.QuantizationConfig(
        static_range_ptq_preset=qc.StaticRangePtqPreset(
            representative_datasets=[
                qc.RepresentativeDatasetConfig(
                    tf_record=qc.TfRecordFile(path=dataset_path)
                )
            ]
        ),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    expected_outputs = model.matmul_and_same_scale(input_data)

    root = load.load(self._output_saved_model_path)
    self.assertCountEqual(root.signatures.keys(), {'serving_default'})

    new_outputs = root.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(input_data)
    )
    # Tests that the quantized graph outputs similar values. The rtol and atol
    # values are arbitrary.
    self.assertAllClose(new_outputs, expected_outputs, rtol=0.03, atol=0.2)

    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.6,
    )

  @parameterized.parameters(
      testing.parameter_combinations([{
          'bias_fn': (
              None,
              nn_ops.bias_add,
          ),
          'activation_fn': (
              None,
              nn_ops.relu,
              nn_ops.relu6,
          ),
          'has_batch_norm': (False, True),
          'input_shape_dynamic': (
              False,
              True,
          ),
          'enable_per_channel_quantized_weight': (
              False,
              True,
          ),
          'merge_fusion_with_dequantize': (False, True),
          'has_func_alias': (False, True),
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_conv_ptq_model(
      self,
      bias_fn: Optional[ops.Operation],
      activation_fn: Optional[ops.Operation],
      has_batch_norm: bool,
      input_shape_dynamic: bool,
      enable_per_channel_quantized_weight: bool,
      merge_fusion_with_dequantize: bool,
      dilations: Sequence[int] = None,
      has_func_alias: bool = False,
  ):
    # [FUNCTION DOC]
    # KNOWN ISSUE: every batch-norm case returns early after the model is built (the two early
    # return conditions together cover all batch-norm cases, both guarded by TODO bug notes), so
    # batch-norm variants verify nothing. The dilations argument is never parameterized and stays
    # null.
    # Parameterized convolution scenario: bias, activation, batch norm, dynamic batch input, per
    # channel weight quantization, merge fusion with dequantize, and function alias presence.
    # Builds and saves a float conv model, quantizes with a static range preset, then expects:
    # int8 convolution and int32 add (when bias) in the StableHLO, outputs close to the float
    # model (relative 0.02, absolute 0.05), size ratio under 0.61, explicit or implicit
    # activation ops depending on the fusion flag, and, when aliases are used, the output model
    # contains exactly the expected alias.
    input_shape = (None, 3, 4, 3) if input_shape_dynamic else (1, 3, 4, 3)
    filter_shape = (2, 3, 3, 2)
    strides = (1, 1, 1, 1)
    model = self._create_conv2d_model(
        input_shape,
        filter_shape,
        self._input_saved_model_path,
        bias_fn,
        activation_fn,
        has_batch_norm,
        strides,
        dilations,
        'SAME',
        has_func_alias,
    )
    # TODO: b/331809306 - Investigate why these test fail then re-enable.
    if has_batch_norm and (bias_fn or not input_shape_dynamic):
      return

    # TODO: b/331120943 - Re-enable this after correctly handling quantization
    # granularity per quantizable scope.
    if has_batch_norm and (not bias_fn and input_shape_dynamic):
      return

    # Generate model input data.
    rng = np.random.default_rng(seed=42)
    static_input_shape = [dim if dim is not None else 2 for dim in input_shape]
    input_data = ops.convert_to_tensor(
        rng.uniform(low=0.0, high=1.0, size=static_input_shape).astype(
            np.float32
        )
    )

    def data_gen() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: representative dataset generator; yields 100 mappings from input_tensor to random
      # float32 data in zero to one with the static input shape.
      for _ in range(100):
        yield {
            'input_tensor': rng.uniform(
                low=0.0, high=1.0, size=static_input_shape
            ).astype(np.float32)
        }

    dataset_path = self.create_tempfile('tfrecord').full_path
    path_map = {'serving_default': dataset_path}
    repr_dataset.TfRecordRepresentativeDatasetSaver(path_map).save(
        {'serving_default': data_gen()}
    )

    config = qc.QuantizationConfig(
        static_range_ptq_preset=qc.StaticRangePtqPreset(
            representative_datasets=[
                qc.RepresentativeDatasetConfig(
                    tf_record=qc.TfRecordFile(path=dataset_path)
                )
            ],
            enable_per_channel_quantized_weight=enable_per_channel_quantized_weight,
        ),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
        pipeline_config=qc.PipelineConfig(
            merge_fusion_with_dequantize=merge_fusion_with_dequantize
        ),
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    expected_outputs = model.conv2d(input_data)

    root = load.load(self._output_saved_model_path)
    self.assertCountEqual(root.signatures.keys(), {'serving_default'})

    new_outputs = root.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(input_data)
    )
    module_str = self._extract_first_xla_call_module_op(
        self._output_saved_model_path
    )
    self.assertTrue(re.search('stablehlo.convolution.*xi8>', module_str))
    if bias_fn:
      self.assertTrue(re.search('stablehlo.add.*xi32>', module_str))
    # Consider if there is a way to check if activation fusion is properly
    # done in MLIR level.
    # Tests that the quantized graph outputs similar values. The rtol and atol
    # values are arbitrary.
    self.assertAllClose(new_outputs, expected_outputs, rtol=0.02, atol=0.05)

    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.61,
    )

    if merge_fusion_with_dequantize:
      # Check activation functions are explicitly present.
      # If present the last op before return should be stablehlo.clamp for relu6
      # and stablehlo.maximum for relu.
      if activation_fn is nn_ops.relu6:
        self.assertRegex(module_str, r'stablehlo.clamp.*\n.*return')
      elif activation_fn is nn_ops.relu:
        self.assertRegex(module_str, r'stablehlo.maximum.*\n.*return')
    else:
      # Check activation functions are implicit.
      self.assertNotRegex(module_str, r'stablehlo.clamp.*\n.*return')
      self.assertNotRegex(module_str, r'stablehlo.maximum.*\n.*return')

    if has_func_alias:
      func_aliases = self._get_function_aliases(
          self._output_saved_model_path, [tag_constants.SERVING]
      )
      self.assertCountEqual(
          func_aliases.values(), [quantize_model_test_base.FUNC_ALIAS]
      )

  @parameterized.parameters(
      testing.parameter_combinations([{
          'equation': (
              'abc,cde->abde',
              'abc,dce->abde',
          ),
      }])
  )
  def test_einsum_ptq_model(
      self,
      equation: str,
  ):
    # [FUNCTION DOC]
    # Parameterized over two einsum equations. Builds shapes with a bias, saves an einsum model,
    # quantizes with a static range preset and a generated dataset, then expects: serving default
    # signature only, outputs close to the float model (relative 0.02, absolute 0.04) when called
    # with input x, and a saved size ratio under 0.65.
    _, y_shape, bias_shape, x_signature, y_signature = (
        self._prepare_sample_einsum_datashapes(equation, use_bias=True)
    )

    model = self._create_einsum_model(
        self._input_saved_model_path,
        equation,
        y_shape,
        x_signature,
        y_signature,
        bias_shape,
    )

    # Generate model input data.
    rng = np.random.default_rng(seed=42)
    input_data = ops.convert_to_tensor(
        rng.uniform(low=0.0, high=1.0, size=x_signature).astype('f4')
    )

    def data_gen() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: representative dataset generator; yields 100 mappings from x to random float32
      # data in zero to one (using the global random state, not the seeded generator).
      for _ in range(100):
        yield {
            'x': ops.convert_to_tensor(
                np.random.uniform(low=0.0, high=1.0, size=x_signature).astype(
                    'f4'
                )
            ),
        }

    dataset_path = self.create_tempfile('tfrecord').full_path
    path_map = {'serving_default': dataset_path}
    repr_dataset.TfRecordRepresentativeDatasetSaver(path_map).save(
        {'serving_default': data_gen()}
    )

    config = qc.QuantizationConfig(
        static_range_ptq_preset=qc.StaticRangePtqPreset(
            representative_datasets=[
                qc.RepresentativeDatasetConfig(
                    tf_record=qc.TfRecordFile(path=dataset_path)
                )
            ]
        ),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    expected_outputs = model.einsum_with_kernel(input_data)

    root = load.load(self._output_saved_model_path)
    self.assertCountEqual(root.signatures.keys(), {'serving_default'})

    new_outputs = root.signatures['serving_default'](
        x=ops.convert_to_tensor(input_data)
    )
    # Tests that the quantized graph outputs similar values. The rtol and atol
    # values are arbitrary.
    self.assertAllClose(new_outputs, expected_outputs, rtol=0.02, atol=0.04)

    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.65,
    )

  def test_reuse_calibration_data(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the comment before the second run says first time, but it is the second run.
    # Scenario: einsum model quantized with min-max calibration and a calibration data directory.
    # Run one: expects the log message that calibration runs in graph mode and an int8 dot_general
    # in the output. Run two with the same config into another output directory: expects that
    # message to be absent (calibration data reused) and the same int8 dot_general. Finally both
    # outputs must produce close results for the same input.
    _, y_shape, bias_shape, x_signature, y_signature = (
        self._prepare_sample_einsum_datashapes('abc,cde->abde', use_bias=True)
    )

    self._create_einsum_model(
        self._input_saved_model_path,
        'abc,cde->abde',
        y_shape,
        x_signature,
        y_signature,
        bias_shape,
    )

    # Generate model input data.
    rng = np.random.default_rng(seed=42)
    input_data = ops.convert_to_tensor(
        rng.uniform(low=0.0, high=1.0, size=x_signature).astype('f4')
    )

    def data_gen() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: representative dataset generator; yields 100 mappings from x to random float32
      # data in zero to one (global random state).
      for _ in range(100):
        yield {
            'x': ops.convert_to_tensor(
                np.random.uniform(low=0.0, high=1.0, size=x_signature).astype(
                    'f4'
                )
            ),
        }

    dataset_path = self.create_tempfile('tfrecord').full_path
    path_map = {'serving_default': dataset_path}
    repr_dataset.TfRecordRepresentativeDatasetSaver(path_map).save(
        {'serving_default': data_gen()}
    )

    calibration_data_dir = self.create_tempdir('calibration_data').full_path
    config = qc.QuantizationConfig(
        static_range_ptq_preset=qc.StaticRangePtqPreset(
            representative_datasets=[
                qc.RepresentativeDatasetConfig(
                    tf_record=qc.TfRecordFile(path=dataset_path)
                )
            ]
        ),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
        calibration_options=qc.CalibrationOptions(
            calibration_method=_CalibrationMethod.CALIBRATION_METHOD_MIN_MAX,
            calibration_data_dir=calibration_data_dir,
        ),
    )

    # Run quantization the first time, calibration is expected to be run.
    with self.assertLogs(level='INFO') as info_logs:
      quantization.quantize_saved_model(
          self._input_saved_model_path,
          self._output_saved_model_path,
          config,
      )
      self.assertTrue(
          self._any_log_contains(
              'Calibration step is executed in graph mode.',
              info_logs.records,
          )
      )
      module_str = self._extract_first_xla_call_module_op(
          self._output_saved_model_path
      )
      self.assertTrue(
          re.search('stablehlo.dot_general.*xi8>.*xi8>.*xi32>', module_str)
      )

    # Run quantization the first time, calibration is expected to be skipped.
    output_saved_model_path_2 = self.create_tempdir('output2').full_path
    with self.assertLogs(level='INFO') as info_logs:
      quantization.quantize_saved_model(
          self._input_saved_model_path,
          output_saved_model_path_2,
          config,
      )
      self.assertFalse(
          self._any_log_contains(
              'Calibration step is executed in graph mode.',
              info_logs.records,
          )
      )
      module_str = self._extract_first_xla_call_module_op(
          output_saved_model_path_2
      )
      self.assertTrue(
          re.search('stablehlo.dot_general.*xi8>.*xi8>.*xi32>', module_str)
      )

    # Expect both quantized model to produce the same results.
    root = load.load(self._output_saved_model_path)
    self.assertCountEqual(root.signatures.keys(), {'serving_default'})
    new_outputs_1 = root.signatures['serving_default'](
        x=ops.convert_to_tensor(input_data)
    )

    root = load.load(output_saved_model_path_2)
    self.assertCountEqual(root.signatures.keys(), {'serving_default'})
    new_outputs_2 = root.signatures['serving_default'](
        x=ops.convert_to_tensor(input_data)
    )

    self.assertAllClose(new_outputs_1, new_outputs_2)

  @parameterized.named_parameters(
      ('use_constant_with_int32_input', np.int32, False),
      ('use_variable_with_int32_input', np.int32, True),
      ('use_constant_with_int64_input', np.int64, False),
      ('use_variable_with_int64_input', np.int64, True),
  )
  @test_util.run_v2_only
  def test_gather_model(self, input_type, use_variable):
    # [FUNCTION DOC]
    # Named parameterized scenario over int32 and int64 index types, with the weight stored as a
    # constant or a variable. Builds a gather model (only available in v2 mode), saves it,
    # quantizes with a static range preset using random indices, and expects an int8 gather in
    # the output and a saved size ratio under one third.
    model = self._create_gather_model(input_type, use_variable)

    save.save(model, self._input_saved_model_path)

    rng = np.random.default_rng(seed=42)
    static_input_shape = [6]

    def data_gen() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: representative dataset generator; yields 100 mappings from input_tensor to random
      # values in zero to ten cast to the given index type, with shape of six elements.
      for _ in range(100):
        yield {
            'input_tensor': rng.uniform(
                low=0.0, high=10, size=static_input_shape
            ).astype(input_type)
        }

    dataset_path = self.create_tempfile('tfrecord').full_path
    path_map = {'serving_default': dataset_path}
    repr_dataset.TfRecordRepresentativeDatasetSaver(path_map).save(
        {'serving_default': data_gen()}
    )

    config = qc.QuantizationConfig(
        static_range_ptq_preset=qc.StaticRangePtqPreset(
            representative_datasets=[
                qc.RepresentativeDatasetConfig(
                    tf_record=qc.TfRecordFile(path=dataset_path)
                )
            ]
        ),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    root = load.load(self._output_saved_model_path)
    self.assertCountEqual(root.signatures.keys(), {'serving_default'})
    module_str = self._extract_first_xla_call_module_op(
        self._output_saved_model_path
    )
    self.assertTrue(re.search('stablehlo.gather.*xi8>', module_str))

    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        1 / 3,
    )

  def test_when_preset_not_srq_raises_error(self):
    # [FUNCTION DOC]
    # Scenario: quantizing with an empty configuration (no preset). Expects a value error whose
    # message says that only static-range post-training quantization is supported.
    self._create_matmul_model(
        input_shape=(1, 1024),
        weight_shape=(1024, 3),
        saved_model_path=self._input_saved_model_path,
    )

    config = qc.QuantizationConfig()
    with self.assertRaisesRegex(ValueError, 'only supports static-range PTQ'):
      quantization.quantize_saved_model(
          self._input_saved_model_path,
          self._output_saved_model_path,
          config,
      )

  @test_util.run_in_graph_and_eager_modes
  def test_ptq_denylist_basic(self):
    # [FUNCTION DOC]
    # Scenario: a matmul model quantized with a specification that matches the composite dot
    # general function name pattern and sets no quantization. Expects outputs nearly identical to
    # the float model (relative 1e-6), the serving signature only, and a size ratio under 0.4
    # (the model is not shrunk since no weights are quantized).
    """Tests that the op is not quantized when no quantization is enabled."""
    input_shape = (1, 2)
    model = self._create_matmul_model(
        input_shape,
        weight_shape=(2, 3),
        saved_model_path=self._input_saved_model_path,
    )

    rng = np.random.default_rng(1230)
    random_tensor_gen_fn = lambda: rng.uniform(
        low=0.0, high=1.0, size=input_shape
    ).astype(np.float32)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: representative dataset generator; yields 50 mappings from input_tensor to values
      # produced by the shared random tensor function.
      for _ in range(50):
        yield {'input_tensor': random_tensor_gen_fn()}

    dataset_path = self.create_tempfile('tfrecord').full_path
    path_map = {'serving_default': dataset_path}
    repr_dataset.TfRecordRepresentativeDatasetSaver(path_map).save(
        {'serving_default': data_gen()}
    )

    config = qc.QuantizationConfig(
        static_range_ptq_preset=qc.StaticRangePtqPreset(
            representative_datasets=[
                qc.RepresentativeDatasetConfig(
                    tf_record=qc.TfRecordFile(path=dataset_path)
                )
            ]
        ),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
        # Disable quantization for the quantizable unit (lifted function) whose
        # function name starts with "composite_dot_general".
        specs=qc.QuantizationSpecs(
            specs=[
                qc.QuantizationSpec(
                    matcher=qc.MatcherSpec(
                        function_name=qc.FunctionNameMatcherSpec(
                            regex='composite_dot_general.*'
                        )
                    ),
                    method=qc.Method(no_quantization={}),
                )
            ]
        ),
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    input_data = ops.convert_to_tensor(random_tensor_gen_fn())
    expected_outputs = model.matmul(input_data)

    root = load.load(self._output_saved_model_path)
    self.assertCountEqual(root.signatures.keys(), {'serving_default'})

    new_outputs = root.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(input_data)
    )

    # Indirectly tests that the model is not quantized by asserting that there
    # are negligible numeric difference.
    self.assertAllClose(new_outputs, expected_outputs, rtol=0.000001)

    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.4,
    )

  @test_util.run_in_graph_and_eager_modes
  def test_ptq_selective_denylist(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring is copied from the basic deny-list test; this test checks that
    # only the first of two matmuls is left unquantized (partial quantization).
    # Scenario: saves a two-matmul model, quantizes with a specification that disables
    # quantization for the first composite dot general unit only. Expects outputs within relative
    # 0.011 of float (partial quantization) and a saved size ratio under 0.55.
    """Tests that the op is not quantized when no quantization is enabled."""

    rng = np.random.default_rng(1230)
    random_tensor_gen_fn = lambda shape: rng.uniform(
        low=-1.0, high=1.0, size=shape
    ).astype(np.float32)

    class TwoMatmulModel(module.Module):
      # [CLASS DOC]
      # Local test model with a single traced function chaining two matrix multiplications with
      # random constant weights of shapes two by three and three by four.
      """A model with two matmul ops."""

      @def_function.function
      def matmul(self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
        # [FUNCTION DOC]
        # Traced function: multiplies the input by a freshly generated random two by three weight
        # then by a random three by four weight. Returns a mapping with key output holding the
        # result.
        """Performs a matrix multiplication.

        Args:
          input_tensor: Input tensor to matmul with the filter.

        Returns:
          A 'output' -> output tensor mapping
        """
        out = math_ops.matmul(input_tensor, random_tensor_gen_fn((2, 3)))
        out = math_ops.matmul(out, random_tensor_gen_fn((3, 4)))
        return {'output': out}

    model = TwoMatmulModel()
    input_shape = (1, 2)

    save.save(
        model,
        self._input_saved_model_path,
        signatures=model.matmul.get_concrete_function(
            tensor_spec.TensorSpec(
                shape=input_shape, dtype=dtypes.float32, name='input_tensor'
            )
        ),
    )

    def data_gen() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: representative dataset generator; yields 50 mappings from input_tensor to random
      # values (minus one to one) of the input shape.
      for _ in range(50):
        yield {'input_tensor': random_tensor_gen_fn(input_shape)}

    dataset_path = self.create_tempfile('tfrecord').full_path
    path_map = {'serving_default': dataset_path}
    repr_dataset.TfRecordRepresentativeDatasetSaver(path_map).save(
        {'serving_default': data_gen()}
    )

    config = qc.QuantizationConfig(
        static_range_ptq_preset=qc.StaticRangePtqPreset(
            representative_datasets=[
                qc.RepresentativeDatasetConfig(
                    tf_record=qc.TfRecordFile(path=dataset_path)
                ),
            ],
        ),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
        # Disable quantization for the quantizable unit (lifted function) whose
        # function name matches "composite_dot_general_fn_1".
        # "composite_dot_general_fn_2" will be quantized.
        specs=qc.QuantizationSpecs(
            specs=[
                qc.QuantizationSpec(
                    matcher=qc.MatcherSpec(
                        function_name=qc.FunctionNameMatcherSpec(
                            regex='composite_dot_general_fn_1'
                        )
                    ),
                    method=qc.Method(no_quantization={}),
                )
            ]
        ),
    )

    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    input_data = ops.convert_to_tensor(random_tensor_gen_fn(input_shape))
    expected_outputs = model.matmul(input_data)

    root = load.load(self._output_saved_model_path)
    self.assertCountEqual(root.signatures.keys(), {'serving_default'})

    new_outputs = root.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(input_data)
    )

    # Indirectly tests that the model is only partially quantized.
    self.assertAllClose(new_outputs, expected_outputs, rtol=0.011)

    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.55,
    )

  @test_util.run_in_graph_and_eager_modes
  def test_ptq_quantization_method_not_applied_when_matcher_mismatch(self):
    # [FUNCTION DOC]
    # Scenario: a matmul model quantized with a specification whose name pattern matches no
    # quantizable unit. Expects the matmul to be quantized anyway (outputs close within relative
    # 0.04 but not nearly identical), the serving signature only, and a size ratio under 0.4.
    """Tests that quantization method is not applied to unmatched units."""
    input_shape = (1, 2)
    model = self._create_matmul_model(
        input_shape,
        weight_shape=(2, 3),
        saved_model_path=self._input_saved_model_path,
    )

    rng = np.random.default_rng(1230)
    random_tensor_gen_fn = lambda: rng.uniform(
        low=0.0, high=1.0, size=input_shape
    ).astype(np.float32)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: representative dataset generator; yields 50 mappings from input_tensor to random
      # values from the shared random tensor function.
      for _ in range(50):
        yield {'input_tensor': random_tensor_gen_fn()}

    dataset_path = self.create_tempfile('tfrecord').full_path
    path_map = {'serving_default': dataset_path}
    repr_dataset.TfRecordRepresentativeDatasetSaver(path_map).save(
        {'serving_default': data_gen()}
    )

    config = qc.QuantizationConfig(
        static_range_ptq_preset=qc.StaticRangePtqPreset(
            representative_datasets=[
                qc.RepresentativeDatasetConfig(
                    tf_record=qc.TfRecordFile(path=dataset_path)
                )
            ]
        ),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
        specs=qc.QuantizationSpecs(
            specs=[
                qc.QuantizationSpec(
                    # Provide a regex that wouldn't match any quantizable units.
                    matcher=qc.MatcherSpec(
                        function_name=qc.FunctionNameMatcherSpec(
                            regex='.*invalid_function_name.*'
                        ),
                    ),
                    method=qc.Method(no_quantization={}),
                ),
            ],
        ),
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    input_data = ops.convert_to_tensor(random_tensor_gen_fn())
    expected_outputs = model.matmul(input_data)

    root = load.load(self._output_saved_model_path)
    self.assertCountEqual(root.signatures.keys(), {'serving_default'})

    new_outputs = root.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(input_data)
    )

    # Tests that the quantized graph outputs similar values. They also shouldn't
    # be exactly the same. Indirectly proves that the `FunctionNameMatcherSpec`
    # with regex '.*invalid_function_name.*' did not match the quantizable unit.
    self.assertAllClose(new_outputs, expected_outputs, rtol=0.04)
    self.assertNotAllClose(new_outputs, expected_outputs, 1e-7)

    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.4,
    )

  def test_save_quantization_report_file(self):
    # [FUNCTION DOC]
    # Scenario: quantizes a matmul model with static range preset and a report file path. Expects
    # the report file to exist and its text-format content to equal a results message listing one
    # quantizable unit (composite dot general function one) with the static range method.
    """Tests that the quantization report file is created.

    Also test that it is populated with textproto of `QuantizationResults`.
    """
    input_shape = (1, 16)
    filter_shape = (16, 3)
    self._create_matmul_model(
        input_shape,
        filter_shape,
        self._input_saved_model_path,
    )

    rng = np.random.default_rng(seed=42)

    def data_gen() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: representative dataset generator; yields 100 mappings from input_tensor to random
      # float32 data in zero to one with the input shape.
      for _ in range(100):
        yield {
            'input_tensor': rng.uniform(
                low=0.0, high=1.0, size=input_shape
            ).astype(np.float32)
        }

    dataset_path = self.create_tempfile('tfrecord').full_path
    path_map = {'serving_default': dataset_path}
    repr_dataset.TfRecordRepresentativeDatasetSaver(path_map).save(
        {'serving_default': data_gen()}
    )

    report_file_path = self.create_tempfile('report.txtpb').full_path
    config = qc.QuantizationConfig(
        static_range_ptq_preset=qc.StaticRangePtqPreset(
            representative_datasets=[
                qc.RepresentativeDatasetConfig(
                    tf_record=qc.TfRecordFile(path=dataset_path)
                )
            ]
        ),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
        report_file_path=report_file_path,
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    # Test the contents of the report file, which is a textproto of
    # `QuantizationResults`.
    self.assertTrue(os.path.exists(report_file_path))
    with open(report_file_path, 'r') as f:
      quantization_results_textpb = f.read()

    results = qc.QuantizationResults()
    text_format.Parse(quantization_results_textpb, results)

    self.assertProtoEquals(
        expected_message_maybe_ascii=r"""
        results {
          quantizable_unit { name: "composite_dot_general_fn_1" }
          method { static_range_ptq {} }
        }
        """,
        validate_message=results,
    )


@test_util.run_all_in_graph_and_eager_modes
class CalibrationOptionsTest(quantize_model_test_base.QuantizedModelTest):
  # [CLASS DOC]
  # Tests for the calibration options configuration (calibration method and parameters). All
  # tests run in both graph and eager modes.
  """Test cases regarding the use of CalibrationOptions proto.

  Run all tests cases in both the graph mode (default in TF1) and the eager mode
  (default in TF2) to ensure support for when TF2 is disabled.
  """

  @parameterized.parameters(
      {
          'calibration_options': qc.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_MIN_MAX
          )
      },
      {
          'calibration_options': qc.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_AVERAGE_MIN_MAX
          ),
      },
      {
          'calibration_options': qc.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_PERCENTILE,
              calibration_parameters=qc.CalibrationOptions.CalibrationParameters(
                  num_bins=10,
              ),
          ),
      },
      {
          'calibration_options': qc.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_BRUTEFORCE,
              calibration_parameters=qc.CalibrationOptions.CalibrationParameters(
                  num_bins=10,
              ),
          ),
      },
      {
          'calibration_options': qc.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_MAX_FREQUENCY,
              calibration_parameters=qc.CalibrationOptions.CalibrationParameters(
                  num_bins=10,
              ),
          ),
      },
      {
          'calibration_options': qc.CalibrationOptions(
              calibration_method=_CalibrationMethod.CALIBRATION_METHOD_HISTOGRAM_MSE_SYMMETRIC,
              calibration_parameters=qc.CalibrationOptions.CalibrationParameters(
                  num_bins=10,
              ),
          ),
      },
  )
  @test_util.run_in_graph_and_eager_modes
  def test_conv_ptq_model_by_calibration_options(
      self,
      calibration_options: qc.CalibrationOptions,
  ):
    # [FUNCTION DOC]
    # Parameterized over six calibration option settings: min-max, average min-max, histogram
    # percentile, histogram MSE brute force, histogram MSE max frequency and histogram MSE
    # symmetric (histogram ones with ten bins). Builds a conv model with bias, relu6 and batch
    # norm, quantizes with a static range preset (per channel off) and those options using
    # random data in zero to ten. Expects the serving signature, outputs close to float
    # (relative 0.02, absolute 0.5) and a size ratio under 0.46.
    bias_fn = nn_ops.bias_add
    activation_fn = nn_ops.relu6
    enable_per_channel_quantized_weight = False
    has_batch_norm = True
    dilations = None
    input_shape = (1, 3, 4, 3)
    filter_shape = (2, 3, 3, 2)
    strides = (1, 1, 1, 1)
    model = self._create_conv2d_model(
        input_shape,
        filter_shape,
        self._input_saved_model_path,
        bias_fn,
        activation_fn,
        has_batch_norm,
        strides,
        dilations,
    )

    # Generate model input data.
    input_data = ops.convert_to_tensor(
        np.random.uniform(low=0.0, high=10, size=input_shape).astype('f4')
    )

    def data_gen() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: representative dataset generator; yields 100 mappings from input_tensor to random
      # float32 data in zero to ten with the input shape (global random state).
      for _ in range(100):
        yield {
            'input_tensor': ops.convert_to_tensor(
                np.random.uniform(low=0, high=10, size=input_shape).astype('f4')
            ),
        }

    dataset_path = self.create_tempfile('tfrecord').full_path
    path_map = {'serving_default': dataset_path}
    repr_dataset.TfRecordRepresentativeDatasetSaver(path_map).save(
        {'serving_default': data_gen()}
    )

    config = qc.QuantizationConfig(
        static_range_ptq_preset=qc.StaticRangePtqPreset(
            representative_datasets=[
                qc.RepresentativeDatasetConfig(
                    tf_record=qc.TfRecordFile(path=dataset_path)
                )
            ],
            enable_per_channel_quantized_weight=enable_per_channel_quantized_weight,
        ),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
        calibration_options=calibration_options,
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    expected_outputs = model.conv2d(input_data)

    root = load.load(self._output_saved_model_path)
    self.assertCountEqual(root.signatures.keys(), {'serving_default'})

    new_outputs = root.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(input_data)
    )
    # Tests that the quantized graph outputs similar values. The rtol and atol
    # values are arbitrary.
    self.assertAllClose(new_outputs, expected_outputs, rtol=0.02, atol=0.5)

    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.46,
    )


class WeightOnlyQuantizationTest(quantize_model_test_base.QuantizedModelTest):
  # [CLASS DOC]
  # Tests of weight-only post-training quantization (only weights are quantized to int8 and
  # dequantized by multiplication; computations stay float). No representative dataset is
  # needed.

  @parameterized.parameters(
      testing.parameter_combinations([{
          'bias_fn': (
              None,
              nn_ops.bias_add,
          ),
          'activation_fn': (
              None,
              nn_ops.relu,
              nn_ops.relu6,
          ),
          'dim_sizes': (
              # tf.MatMul cases.
              ([None, 1024], [1024, 3]),  # dynamic batch dim.
              ([1, 1024], [1024, 3]),
              # tf.BatchMatMul cases.
              ([10, 1, 1024], [10, 1024, 3]),
              ([2, 3, 1, 1024], [2, 3, 1024, 3]),
          ),
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_matmul_weight_only_model(
      self,
      bias_fn: Optional[ops.Operation],
      activation_fn: Optional[ops.Operation],
      dim_sizes: Sequence[int],
  ):
    # [FUNCTION DOC]
    # Parameterized over bias, activation and four shape pairs (matmul and batch matmul, one
    # dynamic). Quantizes a float matmul model with the weight-only preset. Expects: outputs
    # close to float (relative 0.03, absolute 0.2); a multiply op (dequantization) and a float32
    # dot_general in the StableHLO; size ratio under 0.3.
    lhs_dim_size, rhs_dim_size = dim_sizes
    input_shape = (*lhs_dim_size,)
    filter_shape = (*rhs_dim_size,)
    static_input_shape = [dim if dim is not None else 2 for dim in input_shape]
    model = self._create_matmul_model(
        input_shape,
        filter_shape,
        self._input_saved_model_path,
        bias_fn,
        activation_fn,
    )

    rng = np.random.default_rng(1234)
    input_data = ops.convert_to_tensor(
        rng.uniform(low=0.0, high=1.0, size=static_input_shape).astype(
            np.float32
        )
    )

    config = qc.QuantizationConfig(
        weight_only_ptq_preset=qc.WeightOnlyPtqPreset(),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    expected_outputs = model.matmul(input_data)

    root = load.load(self._output_saved_model_path)
    self.assertCountEqual(root.signatures.keys(), {'serving_default'})

    new_outputs = root.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(input_data)
    )
    # Tests that the quantized graph outputs similar values. The rtol and atol
    # values are arbitrary.
    self.assertAllClose(new_outputs, expected_outputs, rtol=0.03, atol=0.2)

    module_str = self._extract_first_xla_call_module_op(
        self._output_saved_model_path
    )

    # Tests that the output graph contains multiply for symmetric
    # dequantization.
    self.assertTrue(re.search('stablehlo.multiply', module_str))
    # Tests that the output graph contains float dot_general.
    self.assertTrue(
        re.search('stablehlo.dot_general.*xf32>.*xf32>.*xf32>', module_str)
    )

    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.3,
    )

  @parameterized.parameters(
      testing.parameter_combinations([{
          'bias_fn': (
              None,
              nn_ops.bias_add,
          ),
          'activation_fn': (
              None,
              nn_ops.relu,
              nn_ops.relu6,
          ),
          'has_batch_norm': (False,),
          'input_shape_dynamic': (
              False,
              True,
          ),
          'has_func_alias': (False, True),
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_conv_weight_only_model(
      self,
      bias_fn: Optional[ops.Operation],
      activation_fn: Optional[ops.Operation],
      has_batch_norm: bool,
      input_shape_dynamic: bool,
      dilations: Sequence[int] = None,
      has_func_alias: bool = False,
  ):
    # [FUNCTION DOC]
    # Parameterized over bias, activation, dynamic input and function alias presence (batch norm
    # off). Quantizes a float conv model with the weight-only preset. Expects: serving signature,
    # outputs close to float (relative 0.03, absolute 0.2), a multiply op and a float32
    # convolution in the StableHLO (the comment there mentions dot_general), expected function
    # alias retained when used, size ratio under 0.4.
    input_shape = (None, 3, 4, 3) if input_shape_dynamic else (1, 3, 4, 3)
    filter_shape = (2, 3, 3, 2)
    strides = (1, 1, 1, 1)
    model = self._create_conv2d_model(
        input_shape,
        filter_shape,
        self._input_saved_model_path,
        bias_fn,
        activation_fn,
        has_batch_norm,
        strides,
        dilations,
        'SAME',
        has_func_alias,
    )

    rng = np.random.default_rng(1234)
    static_input_shape = [dim if dim is not None else 2 for dim in input_shape]
    input_data = ops.convert_to_tensor(
        rng.uniform(low=0.0, high=1.0, size=static_input_shape).astype(
            np.float32
        )
    )

    config = qc.QuantizationConfig(
        weight_only_ptq_preset=qc.WeightOnlyPtqPreset(),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    expected_outputs = model.conv2d(input_data)

    root = load.load(self._output_saved_model_path)
    self.assertCountEqual(root.signatures.keys(), {'serving_default'})

    new_outputs = root.signatures['serving_default'](
        input_tensor=ops.convert_to_tensor(input_data)
    )
    # Tests that the quantized graph outputs similar values. The rtol and atol
    # values are arbitrary.
    self.assertAllClose(new_outputs, expected_outputs, rtol=0.03, atol=0.2)

    module_str = self._extract_first_xla_call_module_op(
        self._output_saved_model_path
    )

    # Tests that the output graph contains multiply op for symmetric
    # dequantization.
    self.assertTrue(re.search('stablehlo.multiply', module_str))
    # Tests that the output graph contains float dot_general.
    self.assertTrue(
        re.search('stablehlo.convolution.*xf32>.*xf32>.*xf32>', module_str)
    )

    if has_func_alias:
      func_aliases = self._get_function_aliases(
          self._output_saved_model_path, [tag_constants.SERVING]
      )
      self.assertCountEqual(
          func_aliases.values(), [quantize_model_test_base.FUNC_ALIAS]
      )

    # Due to other meta data, the compression is not exactly 1/4.
    self.assertLess(
        testing.get_size_ratio(
            self._output_saved_model_path, self._input_saved_model_path
        ),
        0.4,
    )

  @parameterized.parameters(
      testing.parameter_combinations([{
          'shape_dynamic': (
              False,
              True,
          ),
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_add_ptq_model(
      self,
      shape_dynamic: bool,
  ):
    # [FUNCTION DOC]
    # Parameterized over static or dynamic input shape. Saves an add-only model and quantizes it
    # with a static range preset. Expects exactly one XlaCallModule op in the output and the
    # add to stay float32 (not quantized).
    input_shape = (None, 3, 4, 3) if shape_dynamic else (2, 3, 4, 3)
    self._create_add_model(
        input_shape,
        self._input_saved_model_path,
    )

    # Generate model input data.
    rng = np.random.default_rng(seed=42)
    static_input_shape = [dim if dim is not None else 2 for dim in input_shape]

    def data_gen() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: representative dataset generator; yields 100 mappings from input_tensor to random
      # float32 data in zero to one with the static input shape.
      for _ in range(100):
        yield {
            'input_tensor': rng.uniform(
                low=0.0, high=1.0, size=static_input_shape
            ).astype(np.float32)
        }

    dataset_path = self.create_tempfile('tfrecord').full_path
    path_map = {'serving_default': dataset_path}
    repr_dataset.TfRecordRepresentativeDatasetSaver(path_map).save(
        {'serving_default': data_gen()}
    )

    config = qc.QuantizationConfig(
        static_range_ptq_preset=qc.StaticRangePtqPreset(
            representative_datasets=[
                qc.RepresentativeDatasetConfig(
                    tf_record=qc.TfRecordFile(path=dataset_path)
                )
            ],
        ),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    self.assertEqual(
        self._get_num_xla_call_module_op(self._output_saved_model_path), 1
    )
    module_str = self._extract_first_xla_call_module_op(
        self._output_saved_model_path
    )

    # Check add is not quantized.
    self.assertTrue(re.search(r'stablehlo.add.*f32>', module_str))

  @parameterized.parameters(
      testing.parameter_combinations([{
          'shape_dynamic': (
              False,
              True,
          ),
      }])
  )
  @test_util.run_in_graph_and_eager_modes
  def test_add_weight_only_model(
      self,
      shape_dynamic: bool,
  ):
    # [FUNCTION DOC]
    # Parameterized over static or dynamic input shape. Saves an add-only model and quantizes it
    # with the weight-only preset. Expects exactly one XlaCallModule op and the add to stay
    # float32 (not quantized).
    input_shape = (None, 3, 4, 3) if shape_dynamic else (2, 3, 4, 3)
    self._create_add_model(
        input_shape,
        self._input_saved_model_path,
    )

    # Generate model input data.
    rng = np.random.default_rng(seed=42)
    static_input_shape = [dim if dim is not None else 2 for dim in input_shape]

    def data_gen() -> repr_dataset.RepresentativeDataset:
      # [FUNCTION DOC]
      # HELPER: representative dataset generator; yields 100 mappings from input_tensor to random
      # float32 data in zero to one with the static input shape. The generated dataset file is
      # written but never referenced by the weight-only configuration.
      for _ in range(100):
        yield {
            'input_tensor': rng.uniform(
                low=0.0, high=1.0, size=static_input_shape
            ).astype(np.float32)
        }

    dataset_path = self.create_tempfile('tfrecord').full_path
    path_map = {'serving_default': dataset_path}
    repr_dataset.TfRecordRepresentativeDatasetSaver(path_map).save(
        {'serving_default': data_gen()}
    )

    config = qc.QuantizationConfig(
        weight_only_ptq_preset=qc.WeightOnlyPtqPreset(),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    self.assertEqual(
        self._get_num_xla_call_module_op(self._output_saved_model_path), 1
    )
    module_str = self._extract_first_xla_call_module_op(
        self._output_saved_model_path
    )

    # Check add is not quantized.
    self.assertTrue(re.search(r'stablehlo.add.*f32>', module_str), module_str)

  def test_save_quantization_report_file(self):
    # [FUNCTION DOC]
    # Scenario: quantizes a conv model with the weight-only preset and a report file path. Expects
    # the report file to exist and its text-format content to equal a results message listing the
    # composite conv function one unit with a weight-only method with an input quantized type
    # entry for operand one.
    """Tests that the quantization report file is created.

    Also test that it is populated with textproto of `QuantizationResults`.
    """

    input_shape = (1, 3, 4, 3)
    filter_shape = (2, 3, 3, 2)
    self._create_conv2d_model(
        input_shape,
        filter_shape,
        self._input_saved_model_path,
    )

    report_file_path = self.create_tempfile('report.txtpb').full_path
    config = qc.QuantizationConfig(
        weight_only_ptq_preset=qc.WeightOnlyPtqPreset(),
        tf_saved_model=qc.TfSavedModelConfig(tags=[tag_constants.SERVING]),
        report_file_path=report_file_path,
    )
    quantization.quantize_saved_model(
        self._input_saved_model_path,
        self._output_saved_model_path,
        config,
    )

    # Test the contents of the report file, which is a textproto of
    # `QuantizationResults`.
    self.assertTrue(os.path.exists(report_file_path))
    with open(report_file_path, 'r') as f:
      quantization_results_textpb = f.read()

    results = qc.QuantizationResults()
    text_format.Parse(quantization_results_textpb, results)

    self.assertProtoEquals(
        expected_message_maybe_ascii=r"""
        results {
          quantizable_unit { name: "composite_conv_fn_1" }
          method {
            weight_only_ptq {
              input_quantized_types {
                key: 1
                value { dimension_specs {} }
              }
            }
          }
        }
        """,
        validate_message=results,
    )


if __name__ == '__main__':
  test.main()
