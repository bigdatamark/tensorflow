# [MODULE DOC]
# HELPER: helper module for the StableHLO quantize_model integration tests.
# Defines the base test class with fixtures that create and save small models, and helpers that
# inspect quantized saved models. Also defines a constant function alias name used by tests.
# Imports MLIR bindings and the StableHLO dialect; no import-time registrations beyond that.
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
"""Base test class for quantize_model Tests."""

from typing import List, Mapping, Optional, Sequence, Tuple

from absl.testing import parameterized
from mlir import ir
from mlir.dialects import stablehlo as stablehlo_dialect
import numpy as np
import tensorflow  # pylint: disable=unused-import

from tensorflow.compiler.mlir.stablehlo import stablehlo
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import save as saved_model_save
from tensorflow.python.types import core


FUNC_ALIAS = 'some_alias'


class QuantizedModelTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # HELPER: shared base class for the StableHLO quantization integration tests.
  # Provides temporary directory attributes plus helper methods that build and save small models
  # (matmul, matmul with same-scale op, conv2d, gather, add, einsum), inspect quantized saved
  # models (count or extract XlaCallModule ops, read function aliases) and search log records.
  """Base test class for StableHLO quant tests."""

  def setUp(self) -> None:
    # [FUNCTION DOC]
    # Per-test fixture: calls the parent setup and creates three temporary directories (input,
    # output and a second output) whose paths are stored as attributes for saved models. The test
    # framework removes them after each test. Returns nothing.
    super().setUp()

    # Many test cases for quantization involve creating and saving the input
    # model and saving the output quantized model. These two member
    # attributes can be used to specify the paths for such models,
    # respectively. These paths will be cleaned up after each test case.
    self._input_saved_model_path = self.create_tempdir('input').full_path
    self._output_saved_model_path = self.create_tempdir('output').full_path
    # Extra output path occasionally used for comparing two different
    # quantized models.
    self._output_saved_model_path_2 = self.create_tempdir('output2').full_path

  def _extract_first_xla_call_module_op(
      self, output_saved_model_path: str
  ) -> str:
    # [FUNCTION DOC]
    # HELPER: inspection helper for saved models.
    # KNOWN ISSUE: only library functions are searched; XlaCallModule ops in the top-level graph
    # nodes are ignored (unlike the counting helper).
    # Loads the saved model at the path, scans the library functions of the serving default
    # signature graph for the first XlaCallModule op, deserializes its module attribute from the
    # portable artifact into MLIR bytecode, parses it and returns the StableHLO module as text.
    # Raises a value error when no such op exists.
    """Extracts the first XlaCallModule op from output saved model to string."""
    root = load.load(output_saved_model_path)
    tf_graph_def = root.signatures['serving_default'].graph.as_graph_def()
    for function in tf_graph_def.library.function:
      for node_def in function.node_def:
        if node_def.op == 'XlaCallModule':
          with ir.Context() as context:
            stablehlo_dialect.register_dialect(context)
            # Serialization in VHLO dialect.
            serialized = node_def.attr.get('module').s
            # MLIR bytecode matching StableHLO version.
            mlir_bytecode = stablehlo.deserialize_portable_artifact_str(
                serialized)
            stablehlo_module = ir.Module.parse(mlir_bytecode, context=context)
            return str(stablehlo_module)
    raise ValueError('No XlaCallModule found in saved model.')

  def _get_num_xla_call_module_op(self, output_saved_model_path: str) -> int:
    # [FUNCTION DOC]
    # HELPER: inspection helper for saved models.
    # Loads the saved model at the path, takes the graph of the serving default signature and
    # counts operations of type XlaCallModule both in the main graph and in every library
    # function. Returns that count as an integer.
    """Gets the number of XlaCallModule ops in the output saved model."""
    root = load.load(output_saved_model_path)
    tf_graph_def = root.signatures['serving_default'].graph.as_graph_def()
    count = 0
    for node_def in tf_graph_def.node:
      if node_def.op == 'XlaCallModule':
        count += 1
    for function in tf_graph_def.library.function:
      for node_def in function.node_def:
        if node_def.op == 'XlaCallModule':
          count += 1
    return count

  def _get_function_aliases(
      self, output_saved_model_path: str, tags: List[str]
  ) -> dict[str, str]:
    # [FUNCTION DOC]
    # HELPER: inspection helper for saved models.
    # Loads the saved model at the path, selects the meta graph with the given tags (list of
    # strings) and returns its mapping from function alias to function name. Raises an error from
    # the loader when no meta graph matches the tags.
    """Gets the function aliases in the output saved model."""
    loader = loader_impl.SavedModelLoader(output_saved_model_path)
    return loader.get_meta_graph_def_from_tags(
        tags
    ).meta_info_def.function_aliases

  def _create_matmul_model(
      self,
      input_shape: Sequence[int],
      weight_shape: Sequence[int],
      saved_model_path: str,
      bias_fn: Optional[ops.Operation] = None,
      activation_fn: Optional[ops.Operation] = None,
  ) -> module.Module:
    # [FUNCTION DOC]
    # HELPER: test fixture that builds and saves a matmul model.
    # Input: input_shape and weight_shape, saved_model_path, optional bias_fn (callable taking the
    # product and bias) and activation_fn (callable on the output). Saves the model with a float32
    # signature for the matmul function and returns the model instance. Side effect: writes a
    # saved model to disk.
    class MatmulModel(module.Module):
      # [CLASS DOC]
      # Local test model: random filters, an optional random bias, and a traced matmul function
      # optionally followed by bias and activation functions captured from the enclosing builder.
      """A simple model with a single matmul.

      Bias and activation function are optional.
      """

      def __init__(
          self,
          weight_shape: Sequence[int],
      ) -> None:
        # [FUNCTION DOC]
        # Creates random uniform filters (range minus one to one) of weight_shape and, when the
        # enclosing builder has a bias function, a random bias vector sized to the last weight
        # dimension. Does not call the parent constructor.
        """Initializes a MatmulModel.

        Args:
          weight_shape: Shape of the weight tensor.
        """
        self.filters = np.random.uniform(low=-1.0, high=1.0, size=weight_shape)

        if bias_fn is not None:
          self.bias = np.random.uniform(
              low=-1.0, high=1.0, size=weight_shape[-1]
          )

      def has_reshape(self) -> bool:
        # [FUNCTION DOC]
        # KNOWN ISSUE: broken and dead code. It refers to attributes (a bias function and a bias
        # size) that this model never defines, so calling it would raise an attribute error.
        # Intended to report whether the bias needs reshaping versus the filters' last dimension.
        return self.bias_fn() and self.bias_size != self.filters.shape[-1]

      @def_function.function
      def matmul(self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
        # [FUNCTION DOC]
        # Traced function multiplying the input by the stored filters, then applying the enclosing
        # builder's bias function (called with the product and the stored bias) if given, then its
        # activation function if given. Returns a mapping with key output holding the result.
        """Performs a matrix multiplication.

        Depending on self.bias_fn and self.activation_fn, it may add a bias
        term or go through the activaction function.

        Args:
          input_tensor: Input tensor to matmul with the filter.

        Returns:
          A map of: output key -> output result.
        """
        out = math_ops.matmul(input_tensor, self.filters, name='sample/matmul')
        if bias_fn is not None:
          out = bias_fn(out, self.bias)
        if activation_fn is not None:
          out = activation_fn(out)
        return {'output': out}

    model = MatmulModel(weight_shape)
    saved_model_save.save(
        model,
        saved_model_path,
        signatures=model.matmul.get_concrete_function(
            tensor_spec.TensorSpec(
                shape=input_shape, dtype=dtypes.float32, name='input_tensor'
            )
        ),
    )
    return model

  def _any_log_contains(
      self, substring: str, log_record_list: List['logging.LogRecord']
  ) -> bool:
    # [FUNCTION DOC]
    # HELPER: assertion helper for tests that capture log output.
    # Returns true when the message of any log record in the list contains the given substring,
    # false otherwise (including for an empty list). No side effects.
    """Returns True if any of the log contains a given substring.

    Args:
      substring: A piece of string to check whether it exists in the log
        message.
      log_record_list: A list of `absl.logging.LogRecord`s.

    Returns:
      True if and only if the substring exists in any of the log in
      `log_record_list`.
    """
    return any(
        map(
            lambda log_record: substring in str(log_record.message),
            log_record_list,
        )
    )

  def _create_matmul_and_same_scale_model(
      self,
      input_shape: Sequence[int],
      weight_shape: Sequence[int],
      saved_model_path: str,
      same_scale_op: str,
  ) -> module.Module:
    # [FUNCTION DOC]
    # HELPER: test fixture that builds and saves a matmul followed by a same-scale op.
    # Input: input_shape, weight_shape, saved_model_path and same_scale_op (name such as
    # concatenate, gather, max_pool, pad, reshape, select, slice or transpose). Saves the model
    # with a float32 signature and returns the model instance. Side effect: writes a saved model
    # to disk. Unknown op names fail later when the function is traced during saving.
    class MatmulAndSameScaleModel(module.Module):
      # [CLASS DOC]
      # Local test model: a matrix multiplication with random filters followed by one same-scale
      # op (an op whose output shares the input's quantization scale) selected by name.
      """A simple model with a same-scale op.

      Op name in StableHLO dialect is given as a string.
      """

      def __init__(
          self,
          weight_shape: Sequence[int],
          same_scale_op: str,
      ) -> None:
        # [FUNCTION DOC]
        # KNOWN ISSUE: the docstring says it initializes a MatmulModel, but this is the
        # same-scale model; also it claims an error for unknown names, which is only raised later
        # when the traced function runs.
        # Creates random uniform filters (range minus one to one) of weight_shape and stores the
        # same-scale op name. Does not call the parent constructor.
        """Initializes a MatmulModel.

        Args:
          weight_shape: Shape of the weight tensor.
          same_scale_op: Name of the same-scale op to be tested. Raises error
            when an unknown name is given.
        """
        self.filters = np.random.uniform(low=-1.0, high=1.0, size=weight_shape)
        self.same_scale_op = same_scale_op

      @def_function.function
      def matmul_and_same_scale(
          self, input_tensor: core.Tensor
      ) -> Mapping[str, core.Tensor]:
        # [FUNCTION DOC]
        # Traced function that multiplies the input by the stored filters, then applies one
        # same-scale op chosen by name: concatenate (with ones), gather (index zero), max pool,
        # pad (constant, one on each side), reshape (flatten), select (random seeded boolean
        # condition versus ones), slice (size one in every dimension from origin) or transpose.
        # Raises a not-implemented error for any other name. Returns a mapping with key output.
        """Performs a matrix multiplication.

        Args:
          input_tensor: Input tensor to matmul with the filter.

        Returns:
          A map of: output key -> output result.
        """
        out = math_ops.matmul(input_tensor, self.filters, name='sample/matmul')

        if self.same_scale_op == 'concatenate':
          ones = array_ops.ones_like(out)
          out = array_ops.concat([out, ones], 0)
        elif self.same_scale_op == 'gather':
          out = array_ops.gather(out, indices=[0], axis=0)
        elif self.same_scale_op == 'max_pool':
          out = nn_ops.max_pool(out, ksize=3, strides=1, padding='SAME')
        elif self.same_scale_op == 'pad':
          paddings = array_ops.ones(
              (array_ops.rank(out), 2), dtype=dtypes.int32
          )
          out = array_ops.pad(out, paddings, 'CONSTANT')
        elif self.same_scale_op == 'reshape':
          out = array_ops.reshape(out, [-1])
        elif self.same_scale_op == 'select':
          rng = np.random.default_rng(seed=1234)
          condition = ops.convert_to_tensor(
              rng.uniform(low=0.0, high=1.0, size=out.shape) < 0.5
          )
          ones = array_ops.ones_like(out)
          out = math_ops.select(condition, out, ones)
        elif self.same_scale_op == 'slice':
          begin = array_ops.zeros((array_ops.rank(out)), dtype=dtypes.int32)
          size = array_ops.ones((array_ops.rank(out)), dtype=dtypes.int32)
          out = array_ops.slice(out, begin, size)
        elif self.same_scale_op == 'transpose':
          out = array_ops.transpose(out)
        else:
          raise NotImplementedError(
              '{} is not implemented for integration test.'.format(
                  self.same_scale_op
              )
          )

        return {'output': out}

    model = MatmulAndSameScaleModel(weight_shape, same_scale_op)
    saved_model_save.save(
        model,
        saved_model_path,
        signatures=model.matmul_and_same_scale.get_concrete_function(
            tensor_spec.TensorSpec(
                shape=input_shape, dtype=dtypes.float32, name='input_tensor'
            )
        ),
    )
    return model

  def _create_conv2d_model(
      self,
      input_shape: Sequence[int],
      filter_shape: Sequence[int],
      saved_model_path: str,
      bias_fn: Optional[ops.Operation] = None,
      activation_fn: Optional[ops.Operation] = None,
      has_batch_norm: bool = False,
      strides: Sequence[int] = (1, 1, 1, 1),
      dilations: Sequence[int] = (1, 1, 1, 1),
      padding: str = 'SAME',
      has_func_alias: bool = False,
  ) -> module.Module:
    # [FUNCTION DOC]
    # HELPER: test fixture that builds and saves a 2D convolution model.
    # Input: input_shape and filter_shape (last filter dimension is the output channel count),
    # saved_model_path, optional bias_fn and activation_fn (bias is applied whenever bias_fn is
    # not null; activation is called on the output), has_batch_norm, strides, dilations (default
    # all ones), padding (default same) and has_func_alias (register the conv function under the
    # module constant alias name when saving). Saves the model with a signature for the conv
    # function and returns the model instance. Side effect: writes a saved model to disk.
    class ConvModel(module.Module):
      # [CLASS DOC]
      # Local test model with a convolution whose per-channel filters have differing ranges, an
      # optional bias, optional batch normalization and optional activation, as configured by the
      # enclosing builder.
      """A simple model with a single conv2d, bias and relu."""

      def __init__(self):
        # [FUNCTION DOC]
        # Builds the filters so each output channel has a different value range (channel i draws
        # uniformly from minus (i+1) to plus (i+1)), stacked along the last axis as float32, and a
        # random float32 bias in the range zero to ten per output channel. Does not call the
        # parent constructor.
        self.out_channel_size = filter_shape[-1]

        # This ensures filters will have different value range per out channel
        self.filters = np.stack(
            [
                np.random.uniform(
                    low=-(i + 1), high=(i + 1), size=filter_shape[:-1]
                ).astype('f4')
                for i in range(self.out_channel_size)
            ],
            axis=-1,
        )

        self.bias = np.random.uniform(
            low=0, high=10, size=(self.out_channel_size)
        ).astype('f4')

      @def_function.function
      def conv2d(self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
        # [FUNCTION DOC]
        # Traced function applying a 2D convolution (channels-last layout) of the input with the
        # stored filters using the enclosing builder's strides, dilations and padding. Then
        # optionally adds the bias (if a bias function was supplied; the supplied function itself
        # is not used, a standard bias add is), applies inference-mode fused batch normalization
        # when requested, and applies the activation function if given. Returns a mapping with key
        # output holding the result.
        """Performs a 2D convolution operation.

        Args:
          input_tensor: Input tensor to perform convolution on.

        Returns:
          A map of: output key -> output result.
        """
        scale = [1.0] * self.out_channel_size
        offset = [0.5] * self.out_channel_size
        mean, variance = scale, offset
        out = nn_ops.conv2d(
            input_tensor,
            self.filters,
            strides=strides,
            dilations=dilations,
            padding=padding,
            data_format='NHWC',
            name='sample/conv',
        )
        if bias_fn is not None:
          out = nn_ops.bias_add(out, self.bias)
        if has_batch_norm:
          # Fusing is supported for non-training case.
          out, _, _, _, _, _ = nn_ops.fused_batch_norm_v3(
              out, scale, offset, mean, variance, is_training=False
          )
        if activation_fn is not None:
          out = activation_fn(out)
        return {'output': out}

    model = ConvModel()
    save_options = None
    if has_func_alias:
      save_options = tensorflow.saved_model.SaveOptions(
          function_aliases={FUNC_ALIAS: model.conv2d}
      )
    saved_model_save.save(
        model,
        saved_model_path,
        signatures=model.conv2d.get_concrete_function(
            tensor_spec.TensorSpec(
                shape=input_shape, dtype=dtypes.float32, name='input_tensor'
            )
        ),
        options=save_options,
    )
    return model

  def _create_gather_model(self, input_type, use_variable) -> module.Module:
    # [FUNCTION DOC]
    # HELPER: test fixture creating a gather model (not saved to disk).
    # Input: input_type, the element type of the six-element index tensor; use_variable, whether
    # the weight is a variable. Returns the model instance. Unlike sibling builders it does not
    # save the model; callers must save it themselves.
    class GatherModel(module.Module):
      # [CLASS DOC]
      # Local test model holding one 128 by 32 weight (variable or constant) and a traced call that
      # gathers rows from it.
      """A simple model with a single gather."""

      def __init__(self, use_variable):
        # [FUNCTION DOC]
        # Creates a random normal float32 weight of shape 128 by 32. If use_variable is true the
        # weight is stored as a trainable variable, otherwise as a plain constant array. Calls the
        # parent constructor.
        """Initializes a GatherModel.

        Args:
          use_variable: If True, creates a variable for weight.
        """
        super().__init__()
        w_val = np.random.randn(128, 32).astype('f4')
        if use_variable:
          self.w = variables.Variable(w_val)
        else:
          self.w = w_val

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(
                  shape=[6], dtype=input_type, name='input_tensor'
              )
          ]
      )
      def __call__(
          self, input_tensor: core.Tensor
      ) -> Mapping[str, core.Tensor]:
        # [FUNCTION DOC]
        # Traced call with a fixed input signature of a rank-1 tensor of six elements whose type
        # is the one given to the enclosing builder. Gathers rows of the stored weight at the
        # given indices. Returns a mapping with key output holding the gathered rows.
        """Performs a gather operation."""
        out = array_ops.gather_v2(self.w, input_tensor)
        return {'output': out}

    return GatherModel(use_variable)

  def _create_add_model(
      self,
      shape: Sequence[int],
      saved_model_path: str,
  ) -> module.Module:
    # [FUNCTION DOC]
    # HELPER: test fixture that builds and saves a model computing input plus input.
    # Input: shape of the float32 input tensor, and the directory to save to. Saves the model with
    # a signature built from the add function. Returns the model instance. Side effect: writes a
    # saved model to disk.
    class AddModel(module.Module):
      # [CLASS DOC]
      # Local stateless test model with one traced function adding its input to itself.
      """A simple model with a single add."""

      def __init__(self):
        # [FUNCTION DOC]
        # Constructor that intentionally does nothing; the model has no state.
        pass

      @def_function.function
      def add(self, input_tensor: core.Tensor) -> Mapping[str, core.Tensor]:
        # [FUNCTION DOC]
        # Traced function adding the input tensor to itself. Input: one tensor. Returns a mapping
        # with key output holding the doubled tensor.
        """Performs an add operation.

        Args:
          input_tensor: Input tensor to perform add on.

        Returns:
          A map of: output key -> output result.
        """
        out = math_ops.add(input_tensor, input_tensor)
        return {'output': out}

    model = AddModel()
    saved_model_save.save(
        model,
        saved_model_path,
        signatures=model.add.get_concrete_function(
            tensor_spec.TensorSpec(
                shape=shape, dtype=dtypes.float32, name='input_tensor'
            )
        ),
    )
    return model

  # Prepares sample einsum input data shapes.
  # This function returns:
  # 1. Shape for input 1
  # 2. Shape for input 2
  # 3. Shape for bias
  # 4. Signature for input 1 (Could contain None dimension)
  # 5. Signature for input 2 (Could contain None dimension)
  def _prepare_sample_einsum_datashapes(
      self,
      equation: str,
      generate_unknown_shape_signature: bool = False,
      use_bias: bool = False,
  ) -> Tuple[
      List[Optional[int]],
      List[Optional[int]],
      Optional[List[Optional[int]]],
      List[Optional[int]],
      List[Optional[int]],
  ]:
    # [FUNCTION DOC]
    # HELPER: test fixture computing sample shapes for an einsum equation.
    # Parses the equation text (two operand labels separated by a comma, then an arrow and output
    # labels) and maps each single-letter label to a fixed size (a 4, b 32, c 64, d 128, e 8;
    # other labels map to an unknown value). If use_bias is true, the bias shape is the size of
    # the last output label only. If generate_unknown_shape_signature is true, the signatures
    # keep sizes only for labels shared by both operands (contracting dimensions) and use unknown
    # for the rest; otherwise signatures equal the shapes.
    # Returns a five-item tuple: x shape, y shape, bias shape (nothing when no bias),
    # x signature, y signature. No side effects.
    # 1. Parse equation.
    comma_pos = equation.find(',')
    arrow_pos = equation.find('->')
    x_labels = equation[0:comma_pos]
    y_labels = equation[comma_pos + 1 : arrow_pos]
    out_labels = equation[arrow_pos + 1 :]

    # 2. Create sample shapes.
    label_to_size = {'a': 4, 'b': 32, 'c': 64, 'd': 128, 'e': 8}
    x_shape = [label_to_size.get(x_label) for x_label in x_labels]
    y_shape = [label_to_size.get(y_label) for y_label in y_labels]
    bias_shape = None
    if use_bias:
      bias_shape = [label_to_size.get(out_label) for out_label in out_labels]
      bias_shape = bias_shape[-1:]
    contracting_dims = set()

    x_signature = list(x_shape)
    y_signature = list(y_shape)
    if generate_unknown_shape_signature:
      for c in x_labels:
        if c in y_labels:
          contracting_dims.add(c)
      x_signature = [
          None if c not in contracting_dims else x_shape[cidx]
          for cidx, c in enumerate(x_labels)
      ]
      y_signature = [
          None if c not in contracting_dims else y_shape[cidx]
          for cidx, c in enumerate(y_labels)
      ]
    return x_shape, y_shape, bias_shape, x_signature, y_signature

  def _create_einsum_model(
      self,
      saved_model_path: str,
      equation: str,
      y_shape: Sequence[int],
      x_signature: Sequence[Optional[int]],
      y_signature: Sequence[Optional[int]],
      bias_shape: Optional[Sequence[int]] = None,
  ) -> module.Module:
    # [FUNCTION DOC]
    # HELPER: test fixture that builds and saves an einsum model.
    # Defines a local model class computing an einsum for the given equation text, with a random
    # kernel of shape y_shape and an optional random bias of bias_shape (omit for no bias).
    # x_signature and y_signature are the input shapes of the traced functions (entries may be
    # unknown). Saves the model to saved_model_path with the serving default signature bound to
    # the kernel variant taking only x. Returns the model instance. Side effect: writes a saved
    # model to disk.
    class EinsumModel(module.Module):
      # [CLASS DOC]
      # Local test model computing an einsum with the equation captured from the enclosing
      # builder. Holds an optional constant bias, a random kernel, and two traced entry points:
      # one using the kernel as the second operand, the other taking it as a runtime input.
      """Einsum class."""

      def __init__(self):
        # [FUNCTION DOC]
        # Builds the einsum model state from the enclosing builder's arguments: an optional
        # constant float32 bias of the given bias shape filled with random uniform values, a
        # random float32 kernel of shape y_shape, and two unused min and max range tuples.
        # Does not call the parent constructor.
        self._bias = None
        if bias_shape is not None:
          self._bias = array_ops.constant(
              np.random.uniform(size=bias_shape), dtype=dtypes.float32
          )

        self._kernel = np.random.uniform(size=y_shape).astype('f4')
        self._min = (-0.8, -0.8, -0.9)
        self._max = (0.9, 0.9, 1.0)

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(
                  name='x', shape=x_signature, dtype=dtypes.float32
              )
          ]
      )
      def einsum_with_kernel(self, x: core.Tensor) -> Mapping[str, core.Tensor]:
        # [FUNCTION DOC]
        # Traced function taking one float32 tensor x. Uses the model's stored random kernel
        # as the second einsum operand (a constant weight). Returns a mapping with key output
        # holding the result.
        return self._einsum(x, self._kernel)

      @def_function.function(
          input_signature=[
              tensor_spec.TensorSpec(
                  name='x', shape=x_signature, dtype=dtypes.float32
              ),
              tensor_spec.TensorSpec(
                  name='y', shape=y_signature, dtype=dtypes.float32
              ),
          ]
      )
      def einsum_without_kernel(
          self, x: core.Tensor, y: core.Tensor
      ) -> Mapping[str, core.Tensor]:
        # [FUNCTION DOC]
        # Traced function taking two float32 tensors x and y (shapes from the signatures given to
        # the model builder, possibly with unknown dimensions). Treats y as a runtime input rather
        # than a constant weight. Returns a mapping with key output holding the einsum result.
        return self._einsum(x, y)

      def _einsum(self, x, y):
        # [FUNCTION DOC]
        # HELPER: shared computation for the two einsum entry points of the test model.
        # Evaluates the enclosing einsum equation on the two input tensors x and y, then adds the
        # bias (if one was configured) along the last dimension. Returns a mapping with the single
        # key output holding the result tensor.

        out = tensorflow.einsum(equation, x, y)
        if self._bias is not None:
          out = nn_ops.bias_add(out, self._bias)
        return {'output': out}

    model = EinsumModel()
    signatures = {
        'serving_default': model.einsum_with_kernel.get_concrete_function(
            tensor_spec.TensorSpec(
                name='x', shape=x_signature, dtype=dtypes.float32
            )
        ),
    }
    saved_model_save.save(model, saved_model_path, signatures=signatures)
    return model
