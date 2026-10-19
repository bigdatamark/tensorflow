# [MODULE DOC]
# Tests for the Grappler automatic mixed precision graph optimizer, which rewrites eligible graph
# nodes to run in half precision (cuda mode, float16) or bfloat16 (mkl mode, oneDNN) and inserts
# casts. Contains many small private builders for graph pieces (convolutions, batch norm, pooling,
# matmul with activation, loops, LSTM cells), helpers to configure and run sessions with and
# without the optimizer, helpers to count and look up cast nodes, two non-inlined function
# definitions, and one test class that builds graphs, runs them with and without the rewrite and
# checks which nodes got the lower precision output type and that numeric results stay close.
# Standalone test program.
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
"""Tests for Grappler AutoMixedPrecision."""

import os
import re

from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import tf2
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.layers import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import sysconfig as sysconfig_lib
from tensorflow.python.platform import test
from tensorflow.python.training import adam
from tensorflow.python.training import gradient_descent
from tensorflow.python.util import _pywrap_utils


def _input(shape):
  # [FUNCTION DOC]
  # HELPER: graph builder for the tests in this module.
  # Input: shape, a list of dimension sizes. Returns a new variable initialized with seeded truncated
  # normal random values of that shape (seed zero, so deterministic across calls).
  """Generates an input of a given shape."""
  return variables.Variable(random_ops.truncated_normal(shape, seed=0))


def _weight(shape):
  # [FUNCTION DOC]
  # HELPER: graph builder for the tests in this module.
  # Input: shape, a list of dimension sizes. Returns a new variable initialized lazily with a seeded
  # Glorot uniform initializer (seed zero). Initialization is wrapped in a callable so that creation
  # inside loops works.
  """Generates a weight of a given shape."""
  # Note that the lambda is needed to allow construction inside loops.
  return variables.Variable(lambda: init_ops.glorot_uniform_initializer(seed=0)
                            (shape))


def _bias(shape):
  # [FUNCTION DOC]
  # HELPER: graph builder for the tests in this module.
  # Input: shape, a list of dimension sizes. Returns a constant tensor of that shape filled with 0.1.
  """Generates a bias of a given shape."""
  return constant_op.constant(0.1, shape=shape)


def _conv2d(x, w):
  # [FUNCTION DOC]
  # HELPER: graph builder for the tests in this module.
  # Inputs: x, a four-dimensional input tensor; w, a filter tensor. Returns a 2D convolution with
  # stride one in every dimension and same padding.
  """Returns a 2d convolution layer with full stride."""
  return nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def _conv3d(x, w):
  # [FUNCTION DOC]
  # HELPER: graph builder for the tests in this module.
  # Inputs: x, a five-dimensional input tensor; w, a filter tensor. Returns a 3D convolution with
  # stride one in every dimension and same padding.
  """Returns a 3d convolution layer with full stride."""
  return nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='SAME')


def _max_pool_2x2(x):
  # [FUNCTION DOC]
  # HELPER: graph builder for the tests in this module.
  # Input: x, a four-dimensional feature map. Returns a max pooling with a two by two window, stride
  # two and same padding (halving the spatial size).
  """Downsamples a feature map by 2X."""
  return nn.max_pool(
      x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def _fused_batchnorm(x, scale, offset):
  # [FUNCTION DOC]
  # HELPER: graph builder for the tests in this module.
  # Inputs: x, the input tensor; scale and offset, per-channel parameters. Returns the result of the
  # fused batch normalization in training mode: a tuple of output, batch mean and batch variance.
  """Batchnorm."""
  return nn_impl.fused_batch_norm(
      x, scale=scale, offset=offset, is_training=True)


def _conv_bn(x):
  # [FUNCTION DOC]
  # HELPER: graph builder for the tests in this module.
  # Input: x, a tensor reshaped to batches of eight by eight single-channel images. Builds a 3x3
  # convolution with six output channels followed by training-mode fused batch normalization and an
  # identity op. Returns the identity output (the normalized tensor). Creates new variables.
  """Conv followed by batchnorm."""
  i = array_ops.reshape(x, [-1, 8, 8, 1])
  f = _weight([3, 3, 1, 6])
  x = _conv2d(i, f)
  s = _weight([6])
  o = _weight([6])
  y, _, _ = _fused_batchnorm(x, s, o)
  y = array_ops.identity(y)
  return y


def _conv3d_bn(x):
  # [FUNCTION DOC]
  # HELPER: graph builder for the tests in this module.
  # Input: x, a tensor reshaped to batches of eight-cubed single-channel volumes. Builds a 3x3x3 3D
  # convolution with six output channels, reshapes the result to a four-dimensional tensor, applies
  # training-mode fused batch normalization and an identity op. Returns the identity output.
  # Creates new variables.
  """Conv3D followed by batchnorm."""
  i = array_ops.reshape(x, [-1, 8, 8, 8, 1])
  f = _weight([3, 3, 3, 1, 6])
  x = _conv3d(i, f)
  s = _weight([6])
  o = _weight([6])
  x = array_ops.reshape(x, [-1, 8, 8, 6])
  y, _, _ = _fused_batchnorm(x, s, o)
  y = array_ops.identity(y)
  return y


def _matmul_act(x):
  # [FUNCTION DOC]
  # HELPER: graph builder for the tests in this module, also used as a loop body.
  # Input: x, a tensor with 64 elements, reshaped to eight by eight. Multiplies it by a new eight by
  # eight weight variable and applies the rectified linear activation. Returns the activation output.
  """Matmul followed by activation."""
  i = array_ops.reshape(x, [8, 8])
  f = _weight([8, 8])
  x = math_ops.matmul(i, f)
  y = nn.relu(x)
  return y


def _conv_pool(x):
  # [FUNCTION DOC]
  # HELPER: graph builder for the tests in this module.
  # Input: x, a tensor reshaped to eight by eight single-channel images. Applies two stages of
  # convolution, bias add, rectified linear activation and max pooling (first 6 then 4 output
  # channels), with new weight variables and constant biases. Returns the second pooled output.
  """(Conv -> bias -> relu -> max_pool) x2."""
  x_image = array_ops.reshape(x, [-1, 8, 8, 1])
  w_conv1 = _weight([3, 3, 1, 6])
  b_conv1 = _bias([6])
  h_conv1 = nn.relu(nn.bias_add(_conv2d(x_image, w_conv1), b_conv1))
  h_pool1 = _max_pool_2x2(h_conv1)
  w_conv2 = _weight([3, 3, 6, 4])
  b_conv2 = _bias([4])
  h_conv2 = nn.relu(nn.bias_add(_conv2d(h_pool1, w_conv2), b_conv2))
  h_pool2 = _max_pool_2x2(h_conv2)
  return h_pool2


def _depthwise_conv2d(x, w):
  # [FUNCTION DOC]
  # HELPER: graph builder for the tests in this module.
  # Inputs: x, a four-dimensional input tensor; w, a depthwise filter. Returns a depthwise 2D
  # convolution with stride one and same padding.
  """Returns a 2d depthwise convolution layer with full stride."""
  return nn.depthwise_conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def _simple_loop(x, functor):
  # [FUNCTION DOC]
  # HELPER: graph builder for the loop tests in this module.
  # Inputs: x, the initial loop value; functor, a callable applied to the value on each iteration.
  # Builds a while loop running four iterations from a counter of zero, applying functor to the
  # carried value each time. Returns the final loop variables: the counter and the carried value.
  """Simple loop whose body is provided by the functor."""
  init = (constant_op.constant(0), x)
  c = lambda i, j: i < 4
  b = lambda i, j: (i + 1, functor(j))
  ij = while_loop.while_loop(c, b, init)
  return ij


def _loop_vars_intertwined(x0, y0, functor_x, functor_y):
  # [FUNCTION DOC]
  # HELPER: graph builder for the loop tests in this module.
  # Inputs: x0 and y0, initial values; functor_x and functor_y, callables. Builds a while loop with
  # four loop variables (two counters, x, y) that runs while the second counter is below four. Each
  # iteration swaps the roles of the counters and applies functor_y to y and functor_x to x, with
  # the results placed in swapped positions so the loop variables are intertwined. Returns the final
  # loop variables as a tuple.
  """Loop whose loop variables are intertwined."""
  c = lambda i, j, x, y: j < 4
  b = lambda i, j, x, y: (j + 1, i + 1, functor_y(y), functor_x(x))
  init = (constant_op.constant(0), constant_op.constant(0), x0, y0)
  ijzw = while_loop.while_loop(c, b, init)
  return ijzw


def _lstm_cell(prev_c, prev_h, x):
  # [FUNCTION DOC]
  # HELPER: graph builder used by the recurrent LSTM test.
  # Inputs: prev_c, previous cell state; prev_h, previous hidden state; x, the input; all with 8 rows.
  # Computes the four gates from one matrix multiplication of the concatenated input and hidden
  # state with a new weight variable (8 by 16), adds a constant bias, applies sigmoid to input,
  # forget and output gates and tanh to the candidate cell, and returns the new cell state and the
  # new hidden state.
  """Create an LSTM cell."""
  # i: input gate
  # f: forget gate
  # o: output gate
  # c: cell state
  # x: input
  # h: embedding
  bias = _bias([4])
  w = _weight([8, 16])
  ifoc = math_ops.matmul(array_ops.concat([x, prev_h], axis=1), w)
  i, f, o, c = array_ops.split(ifoc, 4, axis=1)
  i = math_ops.sigmoid(nn.bias_add(i, bias))
  f = math_ops.sigmoid(nn.bias_add(f, bias))
  o = math_ops.sigmoid(nn.bias_add(o, bias))
  c = math_ops.tanh(nn.bias_add(c, bias))
  next_c = f * prev_c + i * c
  next_h = o * math_ops.tanh(next_c)
  return next_c, next_h


def _recurrent_lstm(c, h):
  # [FUNCTION DOC]
  # HELPER: graph builder for the recurrent LSTM test.
  # Inputs: c and h, the initial cell and hidden states (8 by 4). Builds a dynamic single-layer LSTM
  # as a while loop of four steps reading inputs from a tensor array filled with four constants (8
  # by 4, value 0.1). Returns the final loop variables: step counter, cell state, hidden state and
  # the tensor array.
  """Dynamic single-layer LSTM with TensorArray."""

  def cond(i, c, h, ta_x):
    # [FUNCTION DOC]
    # HELPER: loop condition of the recurrent LSTM. Ignores the states and the array; returns whether
    # the counter i is below four.
    del c, h, ta_x
    return i < 4

  def body(i, c, h, ta_x):
    # [FUNCTION DOC]
    # HELPER: loop body of the recurrent LSTM. Reads the i-th input from the tensor array, runs one
    # LSTM cell step with the carried states, and returns the incremented counter, the new cell
    # state, the new hidden state and the array.
    x = ta_x.read(i)
    next_c, next_h = _lstm_cell(c, h, x)
    return (i + 1, next_c, next_h, ta_x)

  ta_x = tensor_array_ops.TensorArray(dtype=dtypes.float32, size=4)
  for i in range(0, 4):
    ta_x = ta_x.write(
        i, constant_op.constant(0.1, shape=[8, 4], dtype=dtypes.float32))
  init = (constant_op.constant(0), c, h, ta_x)
  r = while_loop.while_loop(cond, body, init)
  return r


def _make_node_with_color(color, input_tensor, name=None):
  # [FUNCTION DOC]
  # HELPER: graph builder for the loop propagation tests.
  # Inputs: color, a one-letter string (case ignored); input_tensor, the tensor to build on; name,
  # optional node name (default none). The color selects the optimizer list type: w (allow) builds a
  # matrix multiply with a new weight of the input shape; g (infer) adds 0.1; c (clear) applies
  # relu; b (deny) squares then takes the square root (via powers). Returns the new tensor.
  # Raises a value error for any other color letter.
  """Returns a node representative of the specified list type."""
  color = color.lower()
  if color == 'w':  # Allow node
    weights = _weight(input_tensor.get_shape().as_list())
    return math_ops.matmul(input_tensor, weights, name=name)
  if color == 'g':  # Infer node
    return math_ops.add(input_tensor, 0.1, name=name)
  if color == 'c':  # Clear node
    return nn.relu(input_tensor, name=name)
  if color == 'b':  # Deny node
    return math_ops.pow(math_ops.pow(input_tensor, 2.), 0.5, name=name)
  raise ValueError('Invalid node color: ' + str(color))


def _build_simple_loop_graph(inp_colors, body_colors, out_colors):
  # [FUNCTION DOC]
  # HELPER: graph builder for the loop propagation tests.
  # Inputs: inp_colors, body_colors, out_colors, strings of color letters (see the node builder).
  # Starts from an eight by eight input variable, chains nodes named input_N for the input colors,
  # runs a simple loop whose body chains nodes named body_N for the body colors, chains nodes named
  # output_N for the output colors, and ends with an identity op. Returns the final tensor.
  """Builds a test graph with a simple loop."""
  a = _input([8, 8])
  for i, color in enumerate(inp_colors):
    a = _make_node_with_color(color, a, 'input_%i' % i)

  def body(x):
    # [FUNCTION DOC]
    # HELPER: loop body for the simple loop graph. Input: x, the carried value. Applies one node per
    # body color letter (named body_N) in order and returns the result.
    for i, color in enumerate(body_colors):
      x = _make_node_with_color(color, x, 'body_%i' % i)
    return x

  _, a = _simple_loop(a, body)
  for i, color in enumerate(out_colors):
    a = _make_node_with_color(color, a, 'output_%i' % i)
  a = array_ops.identity(a)
  return a


def _get_config(auto_mixed_precision_mode):
  # [FUNCTION DOC]
  # HELPER: session configuration for the tests.
  # Input: auto_mixed_precision_mode, cuda, mkl or none. Returns a session configuration with the
  # arithmetic optimization and remapping rewriters switched off (so nodes are not merged or fused),
  # the minimum graph node limit disabled, cost model building enabled, and the optimizer level
  # set to minus one. Turns on the half-precision rewrite for cuda or the oneDNN bfloat16 rewrite
  # for mkl; for none leaves both off. Any other value fails an assertion.
  """Returns a ConfigProto with auto mixed precision enabled if appropriate."""
  rewrite_config = rewriter_config_pb2.RewriterConfig(
      # do not remove duplicated nodes
      arithmetic_optimization=rewriter_config_pb2.RewriterConfig.OFF,
      # do not turn Conv2D and other nodes into _FusedConv2D
      remapping=rewriter_config_pb2.RewriterConfig.OFF,
  )
  if auto_mixed_precision_mode == 'cuda':
    rewrite_config.auto_mixed_precision = rewriter_config_pb2.RewriterConfig.ON
  elif auto_mixed_precision_mode == 'mkl':
    rewrite_config.auto_mixed_precision_onednn_bfloat16 = (
        rewriter_config_pb2.RewriterConfig.ON)
  else:
    assert auto_mixed_precision_mode is None
  rewrite_config.min_graph_nodes = -1
  graph_options = config_pb2.GraphOptions(
      rewrite_options=rewrite_config, build_cost_model=1)
  config = config_pb2.ConfigProto(graph_options=graph_options)
  config.graph_options.optimizer_options.opt_level = -1
  return config


def _get_device(auto_mixed_precision_mode):
  # [FUNCTION DOC]
  # HELPER: device selection for the tests.
  # Input: auto_mixed_precision_mode. Returns the CPU device name for mkl, otherwise an empty string
  # (let placement choose).
  """Returns the device to run on. If mode is mkl, run on CPU"""
  if auto_mixed_precision_mode == 'mkl':
    return '/cpu:0'
  else:
    return ''


def _is_cast_to_fp16(node_name):
  # [FUNCTION DOC]
  # HELPER: node-name classifier for the cast counters.
  # Input: node_name. Returns a match result (truthy) if the name looks like a cast-to-float16 node
  # inserted by the optimizer, otherwise nothing.
  return re.match('.*-CastToFp16-[0-9]-AutoMixedPrecision$', node_name)


def _is_cast_to_bf16(node_name):
  # [FUNCTION DOC]
  # HELPER: node-name classifier for the cast counters.
  # Input: node_name. Returns a match result (truthy) if the name looks like a cast-to-bfloat16 node
  # inserted by the optimizer, otherwise nothing.
  return re.match('.*-CastToBf16-[0-9]-AutoMixedPrecision$', node_name)


def _is_cast_to_fp32(node_name):
  # [FUNCTION DOC]
  # HELPER: node-name classifier for the cast counters.
  # Input: node_name. Returns a match result (truthy) if the name looks like a cast-to-float32 node
  # inserted by the optimizer, otherwise nothing.
  return re.match('.*-CastToFp32-[0-9]-AutoMixedPrecision$', node_name)


def _count_casts(mode, nodes):
  # [FUNCTION DOC]
  # HELPER: assertion support for the tests.
  # KNOWN ISSUE: the docstring says it counts casts to f16 and fp32, but in mkl mode it returns the
  # bfloat16 count first. Also, the fp32 check is an else-if attached to the bf16 check, so a node
  # matched as bf16 is never counted as fp32 (harmless given the name patterns).
  # Inputs: mode, cuda or mkl; nodes, a sequence of graph nodes. Counts nodes by cast name pattern.
  # Returns a pair (lower precision cast count, float32 cast count). Fails an assertion if there is
  # a bf16 cast in cuda mode, an fp16 cast in mkl mode, or the mode is not one of the two.
  """Counts the number of casts to f16 and fp32."""
  num_to_fp16 = 0
  num_to_bf16 = 0
  num_to_fp32 = 0
  for node in nodes:
    if _is_cast_to_fp16(node.name):
      num_to_fp16 += 1
    if _is_cast_to_bf16(node.name):
      num_to_bf16 += 1
    elif _is_cast_to_fp32(node.name):
      num_to_fp32 += 1
  if mode == 'cuda':
    assert num_to_bf16 == 0
    return num_to_fp16, num_to_fp32
  else:
    assert mode == 'mkl'
    assert num_to_fp16 == 0
    return num_to_bf16, num_to_fp32


def _build_node_map(nodes):
  # [FUNCTION DOC]
  # HELPER: lookup builder for the tests.
  # Input: nodes, a sequence of nodes with a name field. Returns a mapping from node name to node
  # (later duplicates overwrite earlier ones).
  node_map = {}
  for node in nodes:
    node_map[node.name] = node
  return node_map


def _example_noninlined_funcdef_shape(op):
  # [FUNCTION DOC]
  # HELPER: shape function for the example non-inlined function definitions.
  # Input: op, an operation. Returns a list holding the shape of the first input (output shape is
  # the same as the input shape).
  return [op.inputs[0].shape]


@function.Defun(
    shape_func=_example_noninlined_funcdef_shape,
    func_name='example_noninlined_funcdef_grad',
    noinline=True)
def _example_noninlined_funcdef_grad(features, grad):
  # [FUNCTION DOC]
  # HELPER: gradient function of the example non-inlined function (declared as a non-inlined
  # function definition with a custom shape function).
  # Inputs: features, the forward input; grad, the incoming gradient. Computes the derivative of
  # the Swish activation (sigmoid times one plus features times one minus sigmoid) and returns the
  # incoming gradient multiplied by it.
  """Gradient of Swish function defined below."""
  sigmoid_features = math_ops.sigmoid(features)
  activation_grad = (
      sigmoid_features * (1.0 + features * (1.0 - sigmoid_features)))
  return grad * activation_grad


@function.Defun(
    grad_func=_example_noninlined_funcdef_grad,
    shape_func=_example_noninlined_funcdef_shape,
    func_name='example_noninlined_funcdef',
    noinline=True)
def _example_noninlined_funcdef(features):
  # [FUNCTION DOC]
  # HELPER: example function definition that is kept as a separate (non-inlined) function in the
  # graph, with the gradient and shape functions above and the name for that function.
  # Input: features, a tensor. Returns the Swish activation: features times the sigmoid of features.
  """Computes the Swish activation function: `x * sigmoid(x)`."""
  return features * math_ops.sigmoid(features)


class AutoMixedPrecisionTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the auto mixed precision optimizer. Each test builds a graph (variables, ops,
  # loops) under a device chosen by the mode, runs it with and without the optimizer through
  # sessions, then checks the output dtype of chosen nodes in the cost graph and numeric closeness.
  # Most tests are parameterized over the modes cuda (float16) and mkl (bfloat16). Fixture: the
  # setup and teardown methods temporarily set the environment variable that tells the optimizer to
  # ignore performance heuristics.
  """Tests the Grappler auto mixed precision optimizer."""
  IGNORE_PERF_VAR = 'TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'

  # TODO(benbarsdell): Add tests for eager mode with a tf.function.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture. Calls the base setup, remembers the current value of the environment variable that
    # makes the optimizer ignore performance heuristics, and sets it to one so the tests also run on
    # older GPUs. Side effect: modifies the process environment.
    super(AutoMixedPrecisionTest, self).setUp()
    # Enable the CUDA tests to be run on pre-Volta GPUs by telling the grappler
    # pass to ignore performance and always transform the graph.
    self._original_ignore_perf_value = os.getenv(self.IGNORE_PERF_VAR)
    os.environ[self.IGNORE_PERF_VAR] = '1'

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture. Restores the environment variable to its saved value, or removes it if it was unset,
    # then calls the base teardown. Side effect: modifies the process environment.
    if self._original_ignore_perf_value is not None:
      os.environ[self.IGNORE_PERF_VAR] = self._original_ignore_perf_value
    else:
      del os.environ[self.IGNORE_PERF_VAR]
    super(AutoMixedPrecisionTest, self).tearDown()

  def _lower_precision_dtype(self, mode):
    # [FUNCTION DOC]
    # HELPER: for the tests. Input: mode. Returns float16 for cuda and bfloat16 otherwise.
    return dtypes.float16 if mode == 'cuda' else dtypes.bfloat16

  def _assert_output_f16(self, mode, node_map, node_name, output_port=0):
    # [FUNCTION DOC]
    # HELPER: assertion for the tests. Inputs: mode; node_map, mapping from node names to cost-graph
    # nodes; node_name; output_port, which output to check (default zero). Asserts the output dtype
    # of that node equals the lower precision dtype for the mode (the name says f16 but bfloat16 is
    # used for mkl). Fails the test otherwise; a missing name raises a key error.
    self.assertEqual(node_map[node_name].output_info[output_port].dtype,
                     self._lower_precision_dtype(mode).as_datatype_enum)

  def _run(self, mode, fetches):
    # [FUNCTION DOC]
    # HELPER: for the tests. Inputs: mode; fetches, tensors to evaluate. First runs in a session
    # without the optimizer (initializing variables) to get reference values; then runs in a
    # session configured for the mode and collects run metadata. Returns a triple: reference values,
    # optimized-run values and the cost graph from the run metadata.
    """Runs the graph and returns the evaluation of the fetches."""
    with session.Session(config=_get_config(None)) as sess:
      sess.run(variables.global_variables_initializer())
      output_val_ref = self.evaluate(fetches)

    with session.Session(config=_get_config(mode)) as sess:
      sess.run(variables.global_variables_initializer())
      metadata = config_pb2.RunMetadata()
      output_val = sess.run(fetches, run_metadata=metadata)

    return output_val_ref, output_val, metadata.cost_graph

  def _maybe_skip(self, mode):
    # [FUNCTION DOC]
    # HELPER: for the tests. Input: mode. Skips the test if mode is cuda and no CUDA GPU is
    # available, if mode is mkl and oneDNN is not enabled, or if mode is mkl and the CPU lacks
    # support for the bfloat16 oneDNN data type (AVX512f machines only).
    if mode == 'cuda' and not test.is_gpu_available(cuda_only=True):
      self.skipTest('No GPU is available')
    if mode == 'mkl' and not test_util.IsMklEnabled():
      self.skipTest('MKL is not enabled')
    # Test will fail on machines without AVX512f, e.g., Broadwell
    is_avx512f = _pywrap_utils.IsDataTypeSupportedByOneDNNOnThisCPU(
        dtypes.bfloat16
    )
    if mode == 'mkl' and not is_avx512f:
      self.skipTest('Skipping test due to non-AVX512f machine')

  def _run_simple_loop_test(self, mode, inp, body, out):
    # [FUNCTION DOC]
    # HELPER: shared body of the propagation tests. Inputs: mode; inp, body, out, strings of color
    # letters for the input, loop body and output sections. Uppercase letters mean the node is
    # expected to get the lower precision type, lowercase means it stays float32.
    # KNOWN ISSUE: the docstring says DT_HALF, but in mkl mode the expected type is bfloat16. The
    # loop variable expected_types is also reused for different things (list of lists then a loop
    # variable), which is confusing but works.
    # Skips if unsupported. Builds the loop graph under the device and a fixed seed, runs it, maps
    # the cost graph, and checks each section node (input_N, while/body_N, output_N) has the expected
    # dtype, printing every mismatch and then asserting there were none. Finally asserts numeric
    # closeness with looser tolerances for mkl.
    """Runs a test of a simple loop.

    The loop has different node colors in different sections of the graph. The
    arguments must be strings where each character represents the color of a
    node in that section of the graph: w = allow, g = infer, c = clear,
    b = deny. CAPITALIZED characters indicate that the node is expected to be
    changed to DT_HALF during graph optimization.

    inp -> loop [ body ] -> out.

    Args:
      mode: Either 'cuda' or 'mkl'.
      inp: A string of letters indicating the colors and expected dtypes of the
        input nodes.
      body: A string of letters indicating the colors and expected dtypes of the
        body nodes.
      out: A string of letters indicating the colors and expected dtypes of the
        output nodes.
    """
    self._maybe_skip(mode)
    with ops.device(_get_device(mode)):
      random_seed.set_random_seed(0)
      expected_types = []
      for section in [inp, body, out]:
        section_expected_types = []
        for color in section:
          if color.isupper():
            expected_type = self._lower_precision_dtype(mode).as_datatype_enum
          else:
            expected_type = types_pb2.DT_FLOAT
          section_expected_types.append(expected_type)
        expected_types.append(section_expected_types)
      a = _build_simple_loop_graph(inp, body, out)
    output_val_ref, output_val, cost_graph = self._run(mode, a)
    node_map = _build_node_map(cost_graph.node)

    section_names = ['input', 'while/body', 'output']
    all_types_correct = True
    for section_name, expected_types in zip(section_names, expected_types):
      for i, expected_type in enumerate(expected_types):
        node_name = section_name + '_%i' % i
        output_port = 0
        optimized_type = node_map[node_name].output_info[output_port].dtype
        if optimized_type != expected_type:
          print('Expected node %s to have type %s but got type %s' %
                (node_name, expected_type, optimized_type))
          all_types_correct = False
    self.assertTrue(all_types_correct)
    if mode == 'mkl':
      self.assertAllClose(output_val_ref, output_val, atol=2e-2, rtol=2e-2)
    else:
      self.assertAllClose(output_val_ref, output_val, atol=2e-3, rtol=1e-3)

  @parameterized.parameters(['cuda', 'mkl'])
  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_conv_bn(self, mode):
    # [FUNCTION DOC]
    # Test (modes cuda and mkl, graph mode only, not run with XLA). Skips if the mode is unsupported.
    # Scenario: two stacked convolution plus batch norm blocks on a 2x8x8x1 input. Asserts the first
    # convolution, batch norm and second convolution outputs are in the lower precision dtype, that
    # there are three casts to lower precision and one cast back to float32, and that the optimized
    # output is close to the reference (tolerance 1e-2 for mkl, 2e-3 on ROCm, otherwise 1e-3).
    """Test graph with convolution followed by batch norm."""
    self._maybe_skip(mode)
    with ops.device(_get_device(mode)):
      random_seed.set_random_seed(0)
      x = _input([2, 8, 8, 1])
      x = _conv_bn(x)
      output = _conv_bn(x)

    output_val_ref, output_val, cost_graph = self._run(mode, output)
    node_map = _build_node_map(cost_graph.node)
    num_to_f16, num_to_fp32 = _count_casts(mode, cost_graph.node)

    self._assert_output_f16(mode, node_map, 'Conv2D')
    self._assert_output_f16(mode, node_map, 'FusedBatchNormV3')
    self._assert_output_f16(mode, node_map, 'Conv2D_1')
    self.assertEqual(num_to_f16, 3)  # Before Conv2D:0, Conv2D:1, Conv2D_1:1
    self.assertEqual(num_to_fp32, 1)  # After FusedBatchNormV3:0
    if mode == 'mkl':
      tol = 1e-2
    elif test.is_built_with_rocm():
      # Bump up the tolerance for the ROCm platform
      # The default tolerance (1e-3) results in a tiny fraction (<1%) of
      # miscompares on ROCm platform, and hence the tolerance bump
      tol = 2e-3
    else:
      tol = 1e-3
    self.assertAllClose(output_val_ref, output_val, atol=tol, rtol=tol)

  @parameterized.parameters(['cuda', 'mkl'])
  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_conv3d_bn(self, mode):
    # [FUNCTION DOC]
    # Test (modes cuda and mkl, graph mode only, not run with XLA). Skips if the mode is unsupported,
    # and always skips for cuda (waiting for a newer cuDNN).
    # KNOWN ISSUE: the docstring says plain convolution but the test builds 3D convolution plus batch
    # norm blocks.
    # Scenario: two stacked 3D convolution plus batch norm blocks on a 2x8x8x8x1 input. Asserts the
    # first 3D convolution, batch norm and second 3D convolution are in lower precision, three casts
    # to lower precision and one to float32, and closeness to the reference with tolerance 1e-2.
    """Test graph with convolution followed by batch norm."""
    self._maybe_skip(mode)
    if mode == 'cuda':
      # TODO(reedwm): enable these tests when cuDNN is upgraded to >= 7.6.2.
      self.skipTest('Test case should be skipped when cuDNN < 7.6.2')
    with ops.device(_get_device(mode)):
      random_seed.set_random_seed(0)
      x = _input([2, 8, 8, 8, 1])
      x = _conv3d_bn(x)
      output = _conv3d_bn(x)

    output_val_ref, output_val, cost_graph = self._run(mode, output)
    node_map = _build_node_map(cost_graph.node)
    num_to_fp16, num_to_fp32 = _count_casts(mode, cost_graph.node)

    self._assert_output_f16(mode, node_map, 'Conv3D')
    self._assert_output_f16(mode, node_map, 'FusedBatchNormV3')
    self._assert_output_f16(mode, node_map, 'Conv3D_1')
    self.assertEqual(num_to_fp16, 3)  # Before Conv3D:0, Conv3D:1, Conv3D_1:1
    self.assertEqual(num_to_fp32, 1)  # After FusedBatchNormV3:0
    self.assertAllClose(output_val_ref, output_val, atol=1e-2, rtol=1e-2)

  @parameterized.parameters(['cuda', 'mkl'])
  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_conv3d(self, mode):
    # [FUNCTION DOC]
    # Test (modes cuda and mkl, graph mode only, not run with XLA). Skips if unsupported, and always
    # skips for cuda (waiting for a newer cuDNN).
    # KNOWN ISSUE: the graph is run twice and the results of the first run are overwritten, so the
    # first run only serves the node checks; the numeric comparison uses the second run.
    # Scenario: a 3D convolution of a 2x8x8x8x1 input with a 3x3x3 filter, followed by an identity,
    # plus gradients of the output with respect to input and filter computed by gradient descent.
    # Asserts the forward convolution and both gradient (backprop input and filter) nodes are in lower
    # precision, and results are close to the reference (tolerance 5e-2 for mkl, else 1e-3).
    """Test grad ops with convolution3d graph."""
    self._maybe_skip(mode)
    if mode == 'cuda':
      # TODO(reedwm): enable these tests when cuDNN is upgraded to >= 7.6.2.
      self.skipTest('Test case should be skipped when cuDNN < 7.6.2')
    with ops.device(_get_device(mode)):
      random_seed.set_random_seed(0)
      x = _input([2, 8, 8, 8, 1])
      f = _weight([3, 3, 3, 1, 6])
      y = _conv3d(x, f)
      y = array_ops.identity(y)
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.01)
      g = optimizer.compute_gradients(y, [x, f])
      output = (y, g)

    output_val_ref, output_val, cost_graph = self._run(mode, output)
    node_map = _build_node_map(cost_graph.node)
    self._assert_output_f16(mode, node_map, 'Conv3D')
    self._assert_output_f16(mode, node_map,
                            'gradients/Conv3D_grad/Conv3DBackpropInputV2')
    self._assert_output_f16(mode, node_map,
                            'gradients/Conv3D_grad/Conv3DBackpropFilterV2')

    output_val_ref, output_val, cost_graph = self._run(mode, output)
    tol = 5e-2 if mode == 'mkl' else 1e-3
    self.assertAllClose(output_val_ref, output_val, atol=tol, rtol=tol)

  @parameterized.parameters(['cuda', 'mkl'])
  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_conv_bn_dropout(self, mode):
    # [FUNCTION DOC]
    # Test (modes cuda and mkl, graph mode only, not run with XLA). Skips if unsupported.
    # KNOWN ISSUE: the ROCm check uses the build-info function without calling it, so it is always
    # truthy and the tolerance is always 2e-3 instead of 1e-3 outside mkl.
    # Scenario: convolution plus batch norm, dropout with rate 0.5, an addition named addition, a
    # second convolution plus batch norm, an identity, and gradients with respect to the input.
    # Asserts the first convolution, batch norm, the addition and the second convolution are in lower
    # precision (dropout is intentionally not checked), and numeric closeness to the reference (5e-2
    # for mkl). The graph is run again for the comparison.
    """Test dropout precision of convolution batch norm graph."""
    self._maybe_skip(mode)
    with ops.device(_get_device(mode)):
      random_seed.set_random_seed(0)
      x = _input([2, 8, 8, 1])
      y = _conv_bn(x)
      y = nn.dropout(y, rate=0.5)
      y = math_ops.add(y, 1, name='addition')
      y = _conv_bn(y)
      y = array_ops.identity(y)
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.01)
      g = optimizer.compute_gradients(y, [x])
      output = (y, g)

    output_val_ref, output_val, cost_graph = self._run(mode, output)
    node_map = _build_node_map(cost_graph.node)
    self._assert_output_f16(mode, node_map, 'Conv2D')
    self._assert_output_f16(mode, node_map, 'FusedBatchNormV3')
    # We do not assert dropout's dtype because we do not want to rely on the
    # node names of dropout's internal implementation.
    self._assert_output_f16(mode, node_map, 'addition')
    self._assert_output_f16(mode, node_map, 'Conv2D_1')

    output_val_ref, output_val, cost_graph = self._run(mode, output)
    # Bump up the tolerance for the ROCm platform
    # The default tolerance (1e-3) results in a tiny fraction (<1%) of
    # miscompares on ROCm platform, and hence the tolerance bump
    tol = 2e-3 if test.is_built_with_rocm else 1e-3
    tol = 5e-2 if mode == 'mkl' else tol
    self.assertAllClose(output_val_ref, output_val, atol=tol, rtol=tol)

  # TODO(reedwm): Fix and enable this test with MKL. Currently this crashes with
  # MKL
  @parameterized.parameters(['cuda'])
  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_conv_pool(self, mode):
    # [FUNCTION DOC]
    # Test (cuda only because it crashes with mkl, graph mode only, not run with XLA).
    # Scenario: two stages of convolution, relu and max pooling on a 2x8x8x1 input. Asserts the first
    # convolution, relu, max pool and second convolution are in lower precision, five casts to lower
    # precision and one to float32, and numeric closeness (tolerance 1e-3, 5e-3 for mkl).
    """Test graph with convolution followed by pooling."""
    self._maybe_skip(mode)
    with ops.device(_get_device(mode)):
      random_seed.set_random_seed(0)
      x = _input([2, 8, 8, 1])
      output = _conv_pool(x)

    output_val_ref, output_val, cost_graph = self._run(mode, output)
    node_map = _build_node_map(cost_graph.node)
    num_to_f16, num_to_fp32 = _count_casts(mode, cost_graph.node)

    self._assert_output_f16(mode, node_map, 'Conv2D')
    self._assert_output_f16(mode, node_map, 'Relu')
    self._assert_output_f16(mode, node_map, 'MaxPool')
    self._assert_output_f16(mode, node_map, 'Conv2D_1')
    self.assertEqual(num_to_f16, 5)
    self.assertEqual(num_to_fp32, 1)
    tol = 5e-3 if mode == 'mkl' else 1e-3
    self.assertAllClose(output_val_ref, output_val, atol=tol, rtol=tol)

  # TODO(benbarsdell): This test has not been tried with MKL.
  @parameterized.parameters(['cuda'])
  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_depthwise_conv2d(self, mode):
    # [FUNCTION DOC]
    # Test (cuda only; not tried with mkl; graph mode only, not run with XLA). Skips if unsupported
    # or if the build info reports a cuDNN version below eight (depthwise ops are only enabled then).
    # Scenario: a depthwise convolution of a 2x8x8x1 input with a 3x3 filter of depth multiplier four,
    # plus gradients with respect to input and filter. Asserts the forward node and both backprop
    # nodes are in lower precision and the results match the reference with tolerance 2e-3.
    """Test grad ops with depthwise convolution2d graph."""
    self._maybe_skip(mode)
    cudnn_version_str = sysconfig_lib.get_build_info().get(
        'cudnn_version', '0.0')
    cudnn_version = tuple([int(x) for x in cudnn_version_str.split('.')])
    if cudnn_version < (8,):
      # Depthwise conv2d ops are only enabled in auto_mixed_precision as of
      # cuDNN v8.
      self.skipTest('cuDNN version >= 8 required')
    with ops.device(_get_device(mode)):
      random_seed.set_random_seed(0)
      x = _input([2, 8, 8, 1])
      f = _weight([3, 3, 1, 4])
      y = _depthwise_conv2d(x, f)
      y = array_ops.identity(y)
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.01)
      g = optimizer.compute_gradients(y, [x, f])
      output = (y, g)

    output_val_ref, output_val, cost_graph = self._run(mode, output)
    node_map = _build_node_map(cost_graph.node)
    self._assert_output_f16(mode, node_map, 'depthwise')
    self._assert_output_f16(
        mode, node_map,
        'gradients/depthwise_grad/DepthwiseConv2dNativeBackpropInput')
    self._assert_output_f16(
        mode, node_map,
        'gradients/depthwise_grad/DepthwiseConv2dNativeBackpropFilter')

    output_val_ref, output_val, cost_graph = self._run(mode, output)
    tol = 2e-3
    self.assertAllClose(output_val_ref, output_val, atol=tol, rtol=tol)

  @parameterized.parameters(['cuda', 'mkl'])
  @test_util.run_v1_only('b/138749235')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_simple_loop(self, mode):
    # [FUNCTION DOC]
    # Test (modes cuda and mkl, version-one graph mode only, not run with XLA). Skips if unsupported.
    # Scenario: a four-iteration loop whose body is a matrix multiply plus relu on an 8x8 input, and
    # gradients with respect to the input. Asserts the in-loop matrix multiply and relu are in lower
    # precision and results are close to the reference (tolerance 1e-2 for mkl, else 1e-3).
    """Test graph with while loop."""
    self._maybe_skip(mode)
    with ops.device(_get_device(mode)):
      random_seed.set_random_seed(0)
      x = _input([8, 8])
      y = _simple_loop(x, _matmul_act)[1]
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.01)
      g = optimizer.compute_gradients(y, [x])
      output = (y, g)

    output_val_ref, output_val, cost_graph = self._run(mode, output)
    node_map = _build_node_map(cost_graph.node)

    self._assert_output_f16(mode, node_map, 'while/MatMul')
    self._assert_output_f16(mode, node_map, 'while/Relu')
    tol = 1e-2 if mode == 'mkl' else 1e-3
    self.assertAllClose(output_val_ref, output_val, atol=tol, rtol=tol)

  @parameterized.parameters(['cuda', 'mkl'])
  @test_util.run_v1_only('b/138749235')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_loop_with_vars_intertwined(self, mode):
    # [FUNCTION DOC]
    # Test (modes cuda and mkl, version-one graph mode only, not run with XLA). Skips if unsupported.
    # Scenario: a loop with intertwined loop variables, whose two bodies are matrix multiply plus relu,
    # starting from a tensor of ones and an 8x8 input; gradients of one output with respect to the
    # input. Asserts both in-loop matrix multiplies and relus are in lower precision and results are
    # close to the reference (tolerance 5e-3 for mkl, else 1e-3).
    """Test graph with intertwined while loops."""
    self._maybe_skip(mode)
    with ops.device(_get_device(mode)):
      random_seed.set_random_seed(0)
      x = _input([8, 8])
      _, _, k, l = _loop_vars_intertwined(
          array_ops.ones(array_ops.shape(x)), x, _matmul_act, _matmul_act)
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.01)
      g = optimizer.compute_gradients(k, [x])
      output = (k, l, g)

    output_val_ref, output_val, cost_graph = self._run(mode, output)
    node_map = _build_node_map(cost_graph.node)

    self._assert_output_f16(mode, node_map, 'while/MatMul')
    self._assert_output_f16(mode, node_map, 'while/Relu')
    self._assert_output_f16(mode, node_map, 'while/MatMul_1')
    self._assert_output_f16(mode, node_map, 'while/Relu_1')
    tol = 5e-3 if mode == 'mkl' else 1e-3
    self.assertAllClose(output_val_ref, output_val, atol=tol, rtol=tol)

  @parameterized.parameters(['cuda'])
  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_multi_paths(self, mode):
    # [FUNCTION DOC]
    # Test (cuda only, graph mode only, not run with XLA). Skips if unsupported.
    # KNOWN ISSUE: the docstring is identical to the one of the second multi-path test; this one uses
    # convolution and pooling on three channel slices while the other uses matrix multiplies.
    # Scenario: a 2x8x8x3 input split into three channels, each passed through the convolution and
    # pooling stack, concatenated and passed to an identity; gradients with respect to the input.
    # Asserts the split, every convolution, relu and max pool (six suffixes) and the concat are in
    # lower precision, and results are close (absolute tolerance 1e-2 on ROCm, else 1e-3).
    """Test graph with multiple paths."""
    self._maybe_skip(mode)
    with ops.device(_get_device(mode)):
      random_seed.set_random_seed(0)
      x = _input([2, 8, 8, 3])
      x1, x2, x3 = array_ops.split(x, num_or_size_splits=3, axis=3)
      y1 = _conv_pool(x1)
      y2 = _conv_pool(x2)
      y3 = _conv_pool(x3)
      y = array_ops.concat([y1, y2, y3], axis=3)
      y = array_ops.identity(y)
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.01)
      g = optimizer.compute_gradients(y, [x])
      output = (y, g)

    output_val_ref, output_val, cost_graph = self._run(mode, output)
    node_map = _build_node_map(cost_graph.node)

    self._assert_output_f16(mode, node_map, 'split')
    for suffix in [''] + ['_%i' % i for i in range(1, 6)]:
      self._assert_output_f16(mode, node_map, 'Conv2D' + suffix)
      self._assert_output_f16(mode, node_map, 'Relu' + suffix)
      self._assert_output_f16(mode, node_map, 'MaxPool' + suffix)
    self._assert_output_f16(mode, node_map, 'concat')
    atol = 1e-2 if test.is_built_with_rocm() else 1e-3
    self.assertAllClose(output_val_ref, output_val, atol=atol, rtol=1e-3)

  @parameterized.parameters(['cuda', 'mkl'])
  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_multi_paths_2(self, mode):
    # [FUNCTION DOC]
    # Test (modes cuda and mkl, graph mode only, not run with XLA). Skips if unsupported.
    # Scenario: two matrix-multiply-plus-relu paths from the same 8x8 input added together and to the
    # input itself; gradients with respect to the input. Asserts both matrix multiplies and relus are
    # in lower precision and results are close (tolerance 2e-2 for mkl, 1e-2 on ROCm, else 1e-3).
    """Test graph with multiple paths."""
    self._maybe_skip(mode)
    with ops.device(_get_device(mode)):
      random_seed.set_random_seed(0)
      x = _input([8, 8])
      y1 = _matmul_act(x)
      y2 = _matmul_act(x)
      y = y1 + y2 + x
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.01)
      g = optimizer.compute_gradients(y, [x])
      output = (g, y)

    output_val_ref, output_val, cost_graph = self._run(mode, output)
    node_map = _build_node_map(cost_graph.node)

    self._assert_output_f16(mode, node_map, 'MatMul')
    self._assert_output_f16(mode, node_map, 'Relu')
    self._assert_output_f16(mode, node_map, 'MatMul_1')
    self._assert_output_f16(mode, node_map, 'Relu_1')
    if mode == 'mkl':
      tol = 2e-2
    elif test.is_built_with_rocm():
      # Bump up the tolerance for the ROCm platform
      # The default tolerance (1e-3) results in a tiny fraction (<1%) of
      # miscompares on ROCm platform, and hence the tolerance bump
      tol = 1e-2
    else:
      tol = 1e-3
    self.assertAllClose(output_val_ref, output_val, atol=tol, rtol=tol)

  @parameterized.parameters(['cuda'])  # MKL doesn't support bf16 Sigmoid
  @test_util.run_v1_only('b/138749235')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_recurrent_lstm(self, mode):
    # [FUNCTION DOC]
    # Test (cuda only because mkl lacks bfloat16 sigmoid; version-one graph mode only, not run with
    # XLA). Skips if unsupported.
    # Scenario: a dynamic recurrent LSTM from two 8x4 initial states and gradients of the final hidden
    # state with respect to both. Asserts the in-loop concat, matrix multiply, split, three sigmoids
    # and two tanh nodes are in lower precision and results are close (tolerance 1e-3).
    """Test graph with recurrent lstm."""
    self._maybe_skip(mode)
    with ops.device(_get_device(mode)):
      random_seed.set_random_seed(0)
      init_c = _input([8, 4])
      init_h = _input([8, 4])
      _, _, h, _ = _recurrent_lstm(init_c, init_h)
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.01)
      g = optimizer.compute_gradients(h, [init_c, init_h])
      output = (h, g)

    output_val_ref, output_val, cost_graph = self._run(mode, output)
    node_map = _build_node_map(cost_graph.node)

    self._assert_output_f16(mode, node_map, 'while/concat')
    self._assert_output_f16(mode, node_map, 'while/MatMul')
    self._assert_output_f16(mode, node_map, 'while/split')
    self._assert_output_f16(mode, node_map, 'while/Sigmoid')
    self._assert_output_f16(mode, node_map, 'while/Sigmoid_1')
    self._assert_output_f16(mode, node_map, 'while/Sigmoid_2')
    self._assert_output_f16(mode, node_map, 'while/Tanh')
    self._assert_output_f16(mode, node_map, 'while/Tanh_1')
    self.assertAllClose(output_val_ref, output_val, atol=1e-3, rtol=1e-3)

  @parameterized.parameters(['cuda', 'mkl'])
  @test_util.run_v1_only('v1 loop test')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_propagation_through_simple_loop_1(self, mode):
    # [FUNCTION DOC]
    # Test (modes cuda and mkl, version-one only). Runs the shared loop propagation check with input
    # section w, body c and output c: the allow node before the loop is lowered, clear nodes follow.
    self._run_simple_loop_test(mode, 'W', 'C', 'C')

  @parameterized.parameters(['cuda', 'mkl'])
  @test_util.run_v1_only('v1 loop test')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_propagation_through_simple_loop_2(self, mode):
    # [FUNCTION DOC]
    # Test (modes cuda and mkl, version-one only). Runs the shared loop propagation check with input
    # c, body c and output w: a clear node in the loop before an allow node after it.
    self._run_simple_loop_test(mode, 'C', 'C', 'W')

  @parameterized.parameters(['cuda', 'mkl'])
  @test_util.run_v1_only('v1 loop test')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_propagation_through_simple_loop_3(self, mode):
    # [FUNCTION DOC]
    # Test (modes cuda and mkl, version-one only). Runs the shared loop propagation check with input
    # w, body g and output w: an infer node inside the loop between allow nodes.
    self._run_simple_loop_test(mode, 'W', 'G', 'W')

  @parameterized.parameters(['cuda', 'mkl'])
  @test_util.run_v1_only('v1 loop test')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_propagation_through_simple_loop_4(self, mode):
    # [FUNCTION DOC]
    # Test (modes cuda and mkl, version-one only). Runs the shared loop propagation check with input
    # w, body of infer, deny, infer and output w: a deny node blocks lowering inside the loop.
    self._run_simple_loop_test(mode, 'W', 'gbg', 'W')

  @parameterized.parameters(['cuda', 'mkl'])
  @test_util.run_v1_only('b/138749235')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_propagation_through_simple_loop_5(self, mode):
    # [FUNCTION DOC]
    # Test (modes cuda and mkl, version-one only). Runs the shared loop propagation check with input
    # b (deny), body of infer, allow, clear, and output c.
    self._run_simple_loop_test(mode, 'b', 'gWC', 'c')

  @parameterized.parameters(['cuda', 'mkl'])
  @test_util.run_v1_only('b/138749235')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_propagation_through_simple_loop_6(self, mode):
    # [FUNCTION DOC]
    # Test (modes cuda and mkl, version-one only). Runs the shared loop propagation check with input
    # b (deny), body of clear, allow, clear, infer, and output c.
    self._run_simple_loop_test(mode, 'b', 'CWCG', 'C')

  @parameterized.parameters(['cuda', 'mkl'])
  @test_util.run_v1_only('b/138749235')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_propagation_through_simple_loop_7(self, mode):
    # [FUNCTION DOC]
    # Test (modes cuda and mkl, version-one only). Runs the shared loop propagation check with input
    # c, body of infer, allow, clear, infer, and output c.
    self._run_simple_loop_test(mode, 'C', 'GWCG', 'C')

  @parameterized.parameters(['cuda', 'mkl'])
  @test_util.run_v1_only('b/138749235')
  @test_util.disable_xla('This test does not pass with XLA')
  def test_propagation_through_simple_loop_8(self, mode):
    # [FUNCTION DOC]
    # Test (modes cuda and mkl, version-one only). Runs the shared loop propagation check with input
    # c, body of clear, infer, deny, infer, allow, clear, and output g.
    self._run_simple_loop_test(mode, 'C', 'CgbgWC', 'g')

  @parameterized.parameters(['cuda', 'mkl'])
  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_noninlined_funcdef(self, mode):
    # [FUNCTION DOC]
    # Test (modes cuda and mkl, graph mode only, not run with XLA). Skips if unsupported.
    # Scenario: an 8x8 matrix multiply plus relu followed by the non-inlined Swish function definition,
    # with gradients with respect to the input. This needs the optimizer to handle an operation
    # definition that only exists in the graph function library. Asserts the matrix multiply output
    # is in lower precision and results are close (tolerance 1e-2 for mkl, else 1e-3; absolute
    # tolerance 1e-2 on ROCm).
    """Test graph with non-inlined function subgraph.

    This requires the grappler pass to handle an OpDef that only appears in the
    graph's function registry instead of the global op registry.

    Args:
      mode: Either 'cuda' or 'mkl'.
    """
    self._maybe_skip(mode)
    with ops.device(_get_device(mode)):
      random_seed.set_random_seed(0)
      x = _input([8, 8])
      y = _matmul_act(x)
      y = _example_noninlined_funcdef(y)
      optimizer = gradient_descent.GradientDescentOptimizer(learning_rate=0.01)
      g = optimizer.compute_gradients(y, [x])
      output = (g, y)

    output_val_ref, output_val, cost_graph = self._run(mode, output)
    node_map = _build_node_map(cost_graph.node)

    self._assert_output_f16(mode, node_map, 'MatMul')
    tol = 1e-2 if mode == 'mkl' else 1e-3
    atol = 1e-2 if test.is_built_with_rocm() else tol
    self.assertAllClose(output_val_ref, output_val, atol=atol, rtol=tol)

  @parameterized.parameters(['cuda', 'mkl'])
  @test_util.run_deprecated_v1
  @test_util.disable_xla('This test does not pass with XLA')
  def test_ingraph_train_loop(self, mode):
    # [FUNCTION DOC]
    # Test (modes cuda and mkl, graph mode only, not run with XLA). Skips if unsupported, and skips
    # when TensorFlow 2 behavior is enabled (the test uses non-resource variables only in TF1).
    # Scenario: a while loop of 100 iterations around a training step: each iteration reads a batch
    # from a one-shot iterator over random data (64 examples of 32 features, 100 classes), applies a
    # dense layer, computes sparse softmax cross entropy and runs an Adam update. Random seeds are set
    # for TensorFlow and NumPy. Asserts the in-loop dense matrix multiply and its gradient matrix
    # multiply are in lower precision and the loss is close to the reference (tolerance 1e-3). This
    # checks handling of Enter ops in front of reads of non-resource variables.
    """Tests a graph containing a while loop around a training update.

    This requires the grappler pass to take special care with its handling of
    Enter ops that appear in front of reads from non-resource variables. See
    the use of NodeImplicitlyReadsVariable in auto_mixed_precision.cc.

    Args:
      mode: Either 'cuda' or 'mkl'.
    """
    self._maybe_skip(mode)
    if tf2.enabled():
      # This test tests non-resource variables, which are only used in TF1.
      self.skipTest('TensorFlow 1 required')
    with ops.device(_get_device(mode)):
      random_seed.set_random_seed(1234)
      np.random.seed(1234)
      num_iter, bs, nchan, nclass = 100, 64, 32, 100

      data = np.random.normal(size=(bs * num_iter, nchan)).astype(np.float32)
      labels = np.random.randint(nclass, size=(bs * num_iter,))
      ds = dataset_ops.Dataset.from_tensor_slices((data, labels))
      ds = ds.batch(bs).prefetch(3)
      it = ds.make_one_shot_iterator()

      def body(_, i):
        # [FUNCTION DOC]
        # HELPER: loop body for the training loop test. Inputs: an ignored loss value, and counter i.
        # Increments the counter, fetches the next batch from the iterator, builds a dense layer, computes
        # sparse softmax cross entropy, minimizes it with an Adam optimizer over the dense layer weights,
        # and returns the loss (after the update through a control dependency) and the counter.
        i += 1
        x, yt = it.get_next()
        dense = layers.Dense(nclass)
        y = dense(x)
        loss = losses.sparse_softmax_cross_entropy(yt, y)
        opt = adam.AdamOptimizer()
        train_op = opt.minimize(loss, var_list=dense.trainable_weights)
        with ops.control_dependencies([train_op]):
          loss = array_ops.identity(loss)
        return loss, i

      begin, end = constant_op.constant(0), constant_op.constant(num_iter)
      loss, _ = while_loop.while_loop(lambda loss, i: math_ops.less(i, end),
                                      body, [0.0, begin])

    output_val_ref, output_val, cost_graph = self._run(mode, loss)
    node_map = _build_node_map(cost_graph.node)

    self._assert_output_f16(mode, node_map, 'while/dense/MatMul')
    self._assert_output_f16(mode, node_map,
                            'while/gradients/while/dense/MatMul_grad/MatMul_1')
    self.assertAllClose(output_val_ref, output_val, atol=1e-3, rtol=1e-3)

  # TODO(benbarsdell): Add tests for list ops (TensorList*) that pass through
  # graph source/sink nodes, similar to the TensorListThroughFunction C++ test.
  # Tests here will have the advantage of catching changes in the types of ops
  # that are added to the graph.


if __name__ == '__main__':
  test.main()
