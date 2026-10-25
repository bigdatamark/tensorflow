# [MODULE DOC]
# Tests and benchmarks for the pfor-based jacobian and batch_jacobian ops. Defines small model
# builders (fully connected, LSTM, dynamic LSTM, MNIST convnet) and factory functions that build
# the same jacobian, hessian, per-example gradient or per-example jacobian twice: once with pfor
# and once with a sequential loop, so tests can assert the two agree. Contains a test case
# class (graph mode only) and a benchmark class that times the pfor versus loop variants.
# Most definitions here are helpers for those two classes. Running the file as a script runs
# the test main entry point.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for jacobian and batch_jacobian ops."""

import functools
import os
import time

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.layers import layers as tf_layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients as gradient_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.losses import losses
from tensorflow.python.ops.parallel_for import control_flow_ops
from tensorflow.python.ops.parallel_for import gradients
from tensorflow.python.platform import test
from tensorflow.python.util import nest


class FullyConnectedModel:
  # [CLASS DOC]
  # HELPER: small test model used to build fully connected networks for jacobian tests.
  # Holds a list of dense layers (each with relu activation and the same width) created at
  # construction. Calling the instance applies the layers in sequence to an input tensor.

  def __init__(self, activation_size, num_layers):
    # [FUNCTION DOC]
    # Creates num_layers dense layers, each with activation_size units and relu activation, and
    # stores them in a private list. Takes the layer width and the number of layers; returns
    # nothing. Layer variables are created lazily on first call.
    self._layers = [
        tf_layers.Dense(activation_size, activation=nn.relu)
        for _ in range(num_layers)
    ]

  def __call__(self, inp):
    # [FUNCTION DOC]
    # Runs the input tensor through every stored dense layer in order and returns the final
    # activation tensor. Side effect: builds layer variables on the first call.
    activation = inp
    for layer in self._layers:
      activation = layer(activation)
    return activation


def fully_connected_model_fn(batch_size, activation_size, num_layers):
  # [FUNCTION DOC]
  # HELPER: builds the fully connected model graph for the jacobian tests.
  # Creates a random normal input of shape batch by activation size and a model with the given
  # number of layers, applies it, and returns the pair of input tensor and output tensor.
  model = FullyConnectedModel(activation_size, num_layers)
  inp = random_ops.random_normal([batch_size, activation_size])
  return inp, model(inp)


def lstm_model_fn(batch_size, state_size, steps, inputs_size=None):
  # [FUNCTION DOC]
  # HELPER: builds an unrolled basic LSTM graph for the jacobian tests.
  # Creates one random normal input per step (width inputs_size, defaulting to state_size when
  # not given), a basic LSTM cell, and unrolls it for the given number of steps from a zero
  # state. Returns the initial cell-state tensor and the final cell-state tensor.
  inputs_size = inputs_size or state_size
  inputs = [
      random_ops.random_normal([batch_size, inputs_size]) for _ in range(steps)
  ]
  cell = rnn_cell.BasicLSTMCell(state_size)
  init_state = cell.zero_state(batch_size, dtypes.float32)
  state = init_state
  for inp in inputs:
    _, state = cell(inp, state)
  return init_state.c, state.c


def dynamic_lstm_model_fn(batch_size, state_size, max_steps):
  # [FUNCTION DOC]
  # HELPER: builds a dynamic RNN graph with constant inputs for the (disabled) dynamic LSTM test.
  # Inputs and per-example sequence lengths are random constants drawn once at graph build time
  # so repeated runs agree. Returns the inputs tensor and the result of running a dynamic RNN
  # over a basic LSTM cell (outputs and final state) from a zero initial state.
  # We make inputs and sequence_length constant so that multiple session.run
  # calls produce the same result.
  inputs = constant_op.constant(
      np.random.rand(batch_size, max_steps, state_size), dtype=dtypes.float32)
  sequence_length = constant_op.constant(
      np.random.randint(0, size=[batch_size], high=max_steps + 1),
      dtype=dtypes.int32)

  cell = rnn_cell.BasicLSTMCell(state_size)
  initial_state = cell.zero_state(batch_size, dtypes.float32)
  return inputs, rnn.dynamic_rnn(
      cell,
      inputs,
      sequence_length=sequence_length,
      initial_state=initial_state)


def create_fc_batch_jacobian(batch_size, activation_size, num_layers):
  # [FUNCTION DOC]
  # HELPER: builds the batch jacobian of a fully connected model both ways.
  # Returns a pair: the batch jacobian of the model output with respect to its input computed
  # with pfor, and the same computed with a sequential loop.
  inp, output = fully_connected_model_fn(batch_size, activation_size,
                                         num_layers)
  pfor_jacobian = gradients.batch_jacobian(output, inp, use_pfor=True)
  while_jacobian = gradients.batch_jacobian(output, inp, use_pfor=False)
  return pfor_jacobian, while_jacobian


def create_lstm_batch_jacobian(batch_size, state_size, steps, inputs_size=None):
  # [FUNCTION DOC]
  # HELPER: builds the batch jacobian of an unrolled LSTM's final state with respect to its
  # initial state, both with pfor and with a sequential loop. The optional inputs_size sets the
  # per-step input width. Returns the pair (pfor result, loop result).
  inp, output = lstm_model_fn(batch_size, state_size, steps,
                              inputs_size=inputs_size)
  pfor_jacobian = gradients.batch_jacobian(output, inp, use_pfor=True)
  while_jacobian = gradients.batch_jacobian(output, inp, use_pfor=False)
  return pfor_jacobian, while_jacobian


def create_dynamic_lstm_batch_jacobian(batch_size, state_size, max_steps):
  # [FUNCTION DOC]
  # HELPER: builds the batch jacobian of a dynamic LSTM's final cell state with respect to its
  # inputs. The pfor version uses the batch jacobian op. The loop version is built by statically
  # unrolling one gradient per state unit, because the loop mode of the op does not work here.
  # Returns the pair (pfor jacobian tensor, list of per-unit gradient tensors).
  inp, (_, final_state) = dynamic_lstm_model_fn(batch_size, state_size,
                                                max_steps)
  pfor_jacobian = gradients.batch_jacobian(final_state.c, inp, use_pfor=True)
  # Note that use_pfor=False does not work above given the current limitations
  # on implementation of while_loop. So we statically unroll the looping in the
  # jacobian computation.
  while_gradients = [
      gradient_ops.gradients(array_ops.gather(final_state.c, i, axis=1), inp)[0]
      for i in range(state_size)
  ]
  return pfor_jacobian, while_gradients


def create_lstm_batch_hessian(batch_size, state_size, steps):
  # [FUNCTION DOC]
  # HELPER: builds the batch hessian of an unrolled LSTM's final state with respect to its
  # initial state. The first jacobian is flattened per example and differentiated again.
  # KNOWN ISSUE: the "while" hessian is not an independent loop-only computation; it reuses the
  # pfor jacobian as input (nested loops do not work), only the second derivative uses a loop.
  # Returns the pair (pfor hessian, loop-based hessian).
  inp, output = lstm_model_fn(batch_size, state_size, steps)
  pfor_jacobian = gradients.batch_jacobian(output, inp, use_pfor=True)
  pfor_jacobian = array_ops.reshape(pfor_jacobian, [batch_size, -1])
  pfor_hessian = gradients.batch_jacobian(pfor_jacobian, inp, use_pfor=True)
  # TODO(agarwal): using two nested while_loop doesn't seem to work here.
  # Hence we use pfor_jacobian for computing while_hessian.
  while_jacobian = pfor_jacobian
  while_hessian = gradients.batch_jacobian(while_jacobian, inp, use_pfor=False)
  return pfor_hessian, while_hessian


def create_lstm_hessian(batch_size, state_size, steps):
  # [FUNCTION DOC]
  # HELPER: builds the full hessian of an unrolled LSTM output with respect to all trainable
  # variables, as lists of jacobians of jacobians.
  # KNOWN ISSUE: the loop-based hessians reuse the pfor jacobians as their first level (nested
  # loops do not work), so only the second level differs between the two results.
  # Returns the pair (pfor hessians, loop hessians). Uses all trainable variables in the graph.
  _, output = lstm_model_fn(batch_size, state_size, steps)
  weights = variables.trainable_variables()
  pfor_jacobians = gradients.jacobian(output, weights, use_pfor=True)
  pfor_hessians = [
      gradients.jacobian(x, weights, use_pfor=True) for x in pfor_jacobians
  ]
  # TODO(agarwal): using two nested while_loop doesn't seem to work here.
  # Hence we use pfor_jacobians for computing while_hessians.
  while_jacobians = pfor_jacobians
  while_hessians = [
      gradients.jacobian(x, weights, use_pfor=False) for x in while_jacobians
  ]
  return pfor_hessians, while_hessians


def create_fc_per_eg_grad(batch_size, activation_size, num_layers):
  # [FUNCTION DOC]
  # HELPER: builds per-example gradients of an l2 loss of a fully connected network plus a
  # one-unit projection layer, with respect to all trainable variables. For each example index
  # the model is applied to that single example; the iteration is vectorized with pfor and also
  # run with a sequential for-loop. Returns the pair (pfor outputs, loop outputs).
  inp = random_ops.random_normal([batch_size, activation_size])
  layers = [
      tf_layers.Dense(activation_size, activation=nn.relu)
      for _ in range(num_layers)
  ]
  projection = tf_layers.Dense(1)

  def model_fn(activation):
    # [FUNCTION DOC]
    # HELPER: per-example model for create_fc_per_eg_grad. Applies the dense layers and the
    # projection to the given activation, takes its l2 loss and returns the list of gradients of
    # that loss with respect to all trainable variables.
    for layer in layers:
      activation = layer(activation)
    activation = projection(activation)
    activation = nn.l2_loss(activation)
    return gradient_ops.gradients(activation, variables.trainable_variables())

  def loop_fn(i):
    # [FUNCTION DOC]
    # HELPER: iteration body. Selects example i from the input, adds a batch dimension of one and
    # returns the per-example gradients computed by the model function.
    return model_fn(array_ops.expand_dims(array_ops.gather(inp, i), 0))

  pfor_outputs = control_flow_ops.pfor(loop_fn, batch_size)
  loop_fn_dtypes = [x.dtype for x in variables.trainable_variables()]
  while_outputs = control_flow_ops.for_loop(loop_fn, loop_fn_dtypes, batch_size)
  return pfor_outputs, while_outputs


def create_lstm_per_eg_grad(batch_size, state_size, steps, inputs_size=None):
  # [FUNCTION DOC]
  # HELPER: builds per-example gradients of the l2 loss of an unrolled LSTM's final cell state
  # with respect to all trainable variables. Random inputs (width inputs_size, default
  # state_size) are created per step. Each iteration slices one example from the inputs and
  # initial state. Computed with pfor and with a for-loop; returns (pfor outputs, loop outputs).
  inputs_size = inputs_size or state_size
  inputs = [
      random_ops.random_normal([batch_size, inputs_size]) for _ in range(steps)
  ]
  cell = rnn_cell.BasicLSTMCell(state_size)
  init_state = cell.zero_state(batch_size, dtypes.float32)

  def model_fn(inps, init_state):
    # [FUNCTION DOC]
    # HELPER: per-example model for create_lstm_per_eg_grad. Unrolls the LSTM cell over the given
    # list of step inputs from the given initial state, takes the l2 loss of the final cell
    # state and returns gradients of it with respect to all trainable variables.
    state = init_state
    for inp in inps:
      _, state = cell(inp, state)
    output = nn.l2_loss(state.c)
    return gradient_ops.gradients(output, variables.trainable_variables())

  def loop_fn(i):
    # [FUNCTION DOC]
    # HELPER: iteration body. Slices example i (with a batch dimension of one) out of every step
    # input and out of the initial state tuple, then returns the per-example gradients from the
    # model function.
    loop_inputs = [
        array_ops.expand_dims(array_ops.gather(x, i), 0) for x in inputs
    ]
    loop_init_state = rnn_cell.LSTMStateTuple(
        *[array_ops.expand_dims(array_ops.gather(x, i), 0) for x in init_state])
    return model_fn(loop_inputs, loop_init_state)

  pfor_outputs = control_flow_ops.pfor(loop_fn, batch_size)
  loop_fn_dtypes = [x.dtype for x in variables.trainable_variables()]
  while_outputs = control_flow_ops.for_loop(loop_fn, loop_fn_dtypes, batch_size)
  return pfor_outputs, while_outputs


# Importing the code from tensorflow_models seems to cause errors. Hence we
# duplicate the model definition here.
# TODO(agarwal): Use the version in tensorflow_models/official instead.
class Mnist(tf_layers.Layer):
  # [CLASS DOC]
  # HELPER: small MNIST convolutional classifier layer duplicated here for the tests (importing
  # it from the models repository caused errors). Holds two convolutions, two dense layers,
  # dropout and max pooling, plus the input reshape size that depends on the data format.
  # Instances are called with images and a training flag and return class logits.

  def __init__(self, data_format):
    # [FUNCTION DOC]
    # Builds the layers: two same-padded 5x5 convolutions (32 and 64 filters, relu), a 1024-unit
    # relu dense layer, a 10-unit dense layer, dropout with rate 0.4 and 2x2 max pooling.
    # data_format must be channels first or channels last; it sets the reshape size of the input.
    # Any other value triggers an assertion failure. Returns nothing.
    """Creates a model for classifying a hand-written digit.

    Args:
      data_format: Either 'channels_first' or 'channels_last'.
    """
    super(Mnist, self).__init__()
    if data_format == "channels_first":
      self._input_shape = [-1, 1, 28, 28]
    else:
      assert data_format == "channels_last"
      self._input_shape = [-1, 28, 28, 1]

    self.conv1 = tf_layers.Conv2D(
        32, 5, padding="same", data_format=data_format, activation=nn.relu)
    self.conv2 = tf_layers.Conv2D(
        64, 5, padding="same", data_format=data_format, activation=nn.relu)
    self.fc1 = tf_layers.Dense(1024, activation=nn.relu)
    self.fc2 = tf_layers.Dense(10)
    self.dropout = tf_layers.Dropout(0.4)
    self.max_pool2d = tf_layers.MaxPooling2D(
        (2, 2), (2, 2), padding="same", data_format=data_format)

  def __call__(self, inputs, training):
    # [FUNCTION DOC]
    # Classifies a batch of images: reshapes the input to four dimensions, applies convolution
    # and pooling twice, flattens, applies the dense layer, dropout (active only when training is
    # true) and the final dense layer. Returns the logits tensor with ten classes per example.
    """Add operations to classify a batch of input images.

    Args:
      inputs: A Tensor representing a batch of input images.
      training: A boolean. Set to True to add operations required only when
        training the classifier.

    Returns:
      A logits Tensor with shape [<batch_size>, 10].
    """
    y = array_ops.reshape(inputs, self._input_shape)
    y = self.conv1(y)
    y = self.max_pool2d(y)
    y = self.conv2(y)
    y = self.max_pool2d(y)
    y = tf_layers.flatten(y)
    y = self.fc1(y)
    y = self.dropout(y, training=training)
    return self.fc2(y)


def create_mnist_autobatch(batch_size, data_format, training):
  # [FUNCTION DOC]
  # HELPER: builds three ways of running the MNIST model on random images: pfor over per-image
  # calls, a sequential for-loop over per-image calls, and one manual batched call. Takes batch
  # size, data format and the training flag. Returns the triple (pfor, loop, manual).
  images = random_ops.random_uniform([batch_size, 28, 28])
  model = Mnist(data_format)
  manual = model(images, training=training)

  def loop_fn(i):
    # [FUNCTION DOC]
    # HELPER: iteration body. Selects image i and returns the model logits for that single image.
    image = array_ops.gather(images, i)
    return model(image, training=training)

  pfor_outputs = control_flow_ops.pfor(loop_fn, batch_size)
  while_outputs = control_flow_ops.for_loop(
      loop_fn, dtypes.float32, batch_size)

  return pfor_outputs, while_outputs, manual


def create_mnist_per_eg_grad(batch_size, data_format, training):
  # [FUNCTION DOC]
  # HELPER: builds per-example gradients of the softmax cross-entropy loss of the MNIST model
  # with respect to all trainable variables, using random images and random one-hot labels
  # (drawn with numpy). Computed with pfor and with a for-loop. Returns (pfor, loop) outputs.
  images = random_ops.random_uniform([batch_size, 28, 28])
  sparse_labels = np.random.randint(
      low=0, high=10, size=[batch_size]).astype(np.int32)
  labels = np.zeros((batch_size, 10)).astype(np.float32)
  labels[np.arange(batch_size), sparse_labels] = 1.
  model = Mnist(data_format)

  def loop_fn(i):
    # [FUNCTION DOC]
    # HELPER: iteration body. Selects image and label i, computes the unreduced softmax
    # cross-entropy of the flattened logits and returns its gradients with respect to all
    # trainable variables.
    image = array_ops.gather(images, i)
    label = array_ops.gather(labels, i)
    logits = array_ops.reshape(model(image, training=training), [-1])
    loss = losses.softmax_cross_entropy(
        logits=logits, onehot_labels=label, reduction=losses.Reduction.NONE)
    return gradient_ops.gradients(loss, variables.trainable_variables())

  pfor_outputs = control_flow_ops.pfor(loop_fn, batch_size)
  while_outputs = control_flow_ops.for_loop(
      loop_fn, [dtypes.float32] * len(variables.trainable_variables()),
      batch_size)
  return pfor_outputs, while_outputs


def create_mnist_batch_jacobian(batch_size, data_format, training):
  # [FUNCTION DOC]
  # HELPER: builds the batch jacobian of the MNIST model logits with respect to random input
  # images, using pfor and using a sequential loop. Returns the pair (pfor, loop) jacobians.
  images = random_ops.random_uniform([batch_size, 28, 28])
  model = Mnist(data_format)
  logits = model(images, training=training)

  pfor_jacobian = gradients.batch_jacobian(logits, images, use_pfor=True)
  while_jacobian = gradients.batch_jacobian(logits, images, use_pfor=False)
  return pfor_jacobian, while_jacobian


def create_mnist_per_eg_jacobian(batch_size, data_format, training):
  # [FUNCTION DOC]
  # HELPER: builds per-example jacobians of the flattened MNIST logits with respect to all
  # trainable variables. The outer iteration over examples is done with pfor and with a
  # for-loop, and the inner jacobian likewise uses pfor or a loop. Returns (pfor, loop) outputs.
  images = random_ops.random_uniform([batch_size, 28, 28])
  model = Mnist(data_format)

  def loop_fn(i, use_pfor):
    # [FUNCTION DOC]
    # HELPER: iteration body. Selects image i, flattens the model logits and returns the jacobian
    # of the logits with respect to all trainable variables, computed with pfor or a loop
    # depending on use_pfor.
    image = array_ops.gather(images, i)
    logits = array_ops.reshape(model(image, training=training), [-1])
    return gradients.jacobian(
        logits, variables.trainable_variables(), use_pfor=use_pfor)

  pfor_outputs = control_flow_ops.pfor(
      functools.partial(loop_fn, use_pfor=True),
      batch_size)
  while_outputs = control_flow_ops.for_loop(
      functools.partial(loop_fn, use_pfor=False),
      [dtypes.float32] * len(variables.trainable_variables()), batch_size)
  return pfor_outputs, while_outputs


def create_fc_per_eg_jacobians(batch_size, activation_size, num_layers):
  # [FUNCTION DOC]
  # HELPER: builds jacobians of a fully connected model output with respect to all trainable
  # variables, two ways: a whole-batch jacobian, and per-example jacobians computed by iterating
  # over examples with pfor and with a for-loop. Returns the triple (batch jacobians, per-example
  # pfor jacobians, per-example loop jacobians).
  model = FullyConnectedModel(activation_size=activation_size,
                              num_layers=num_layers)
  inp = random_ops.random_normal([batch_size, activation_size])
  output = model(inp)
  jacobians = gradients.jacobian(output, variables.trainable_variables())

  def loop_fn(i, use_pfor):
    # [FUNCTION DOC]
    # HELPER: iteration body. Takes example i with a batch dimension of one, runs the model,
    # flattens the output and returns its jacobian with respect to all trainable variables,
    # computed with pfor or a loop depending on use_pfor.
    inp_i = array_ops.expand_dims(array_ops.gather(inp, i), 0)
    output = array_ops.reshape(model(inp_i), [-1])
    return gradients.jacobian(
        output, variables.trainable_variables(), use_pfor=use_pfor)

  per_eg_jacobians_pfor = control_flow_ops.pfor(
      functools.partial(loop_fn, use_pfor=True),
      batch_size)
  per_eg_jacobians_while = control_flow_ops.for_loop(
      functools.partial(loop_fn, use_pfor=False),
      [dtypes.float32] * len(variables.trainable_variables()), batch_size)
  return jacobians, per_eg_jacobians_pfor, per_eg_jacobians_while


@test_util.run_v1_only("b/122612051")
class GradientsTest(test.TestCase):
  # [CLASS DOC]
  # Graph-mode-only (v1) tests for the jacobian and batch_jacobian ops: no-path behavior, fixed,
  # scan, while-loop and unknown shapes, parallel iteration settings, bad shapes, and
  # equivalence of pfor and loop results on fully connected, LSTM and MNIST models, hessians,
  # per-example gradients and sparse gradients. Shared fixture: run_and_assert_equal.

  def run_and_assert_equal(self, targets1, targets2, atol=1e-4, rtol=1e-4):
    # [FUNCTION DOC]
    # HELPER: shared assertion for the tests in this class.
    # Flattens both target structures (which must have the same number of entries, else an
    # assertion fails), initializes all global variables, evaluates both sets in one run and
    # asserts each pair is close within the given absolute and relative tolerances.
    targets1 = nest.flatten(targets1)
    targets2 = nest.flatten(targets2)
    assert len(targets1) == len(targets2)
    init = variables.global_variables_initializer()
    self.evaluate(init)
    outputs = self.evaluate(targets1 + targets2)
    n = len(outputs) // 2
    for i in range(n):
      self.assertAllClose(outputs[i], outputs[i + n], rtol=rtol, atol=atol)

  def test_no_path(self):
    # [FUNCTION DOC]
    # Scenario: output and input are unrelated constants. For both jacobian and batch jacobian,
    # with pfor on and off, the result is expected to be nothing/null.
    for grad_func in [gradients.jacobian, gradients.batch_jacobian]:
      for use_pfor in [True, False]:
        x = constant_op.constant([[1.0]])
        y = constant_op.constant([[2.0]])
        self.assertIsNone(grad_func(y, x, use_pfor=use_pfor))

  def test_jacobian_fixed_shape(self):
    # [FUNCTION DOC]
    # Scenario: output is x transposed times x for a fixed 2x2 random x. The jacobian from pfor and
    # from the loop must both equal a reference built from per-element gradients.
    x = random_ops.random_uniform([2, 2])
    y = math_ops.matmul(x, x, transpose_a=True)
    jacobian_pfor = gradients.jacobian(y, x, use_pfor=True)
    jacobian_while = gradients.jacobian(y, x, use_pfor=False)
    answer = ops.convert_to_tensor([[
        gradient_ops.gradients(y[0][0], x)[0],
        gradient_ops.gradients(y[0][1], x)[0]
    ], [
        gradient_ops.gradients(y[1][0], x)[0],
        gradient_ops.gradients(y[1][1], x)[0]
    ]])
    self.run_and_assert_equal(answer, jacobian_pfor)
    self.run_and_assert_equal(answer, jacobian_while)

  def test_jacobian_scan_shape(self):
    # [FUNCTION DOC]
    # Scenario: output comes from a scan over six elements starting from a 3x4 tensor. Expects the
    # static shape of the jacobian to be 6, 3, 4, 3, 4.
    # Shape x: [3, 4]
    x = random_ops.random_uniform([3, 4])
    elems = random_ops.random_uniform([6])
    # Shape y: [6, 3, 4]
    y = functional_ops.scan(lambda a, e: a + e, elems, initializer=x)
    jacobian = gradients.jacobian(y, x)

    expected_shape = [6, 3, 4, 3, 4]
    self.assertAllEqual(expected_shape, jacobian.shape.as_list())

  def test_jacobian_while_loop_shape(self):
    # [FUNCTION DOC]
    # Scenario: output is a 2x3 slice of the result of a while loop over a 3x4 tensor. Expects the
    # static jacobian shape to be 2, 3, 3, 4.
    # Shape x: [3, 4]
    x = random_ops.random_uniform([3, 4])
    _, y = while_loop.while_loop(lambda i, a: i > 5., lambda i, a:
                                 (i + 1, a + i), (constant_op.constant(0.), x))
    # Shape y: [2, 3]
    y = y[:2, :3]
    jacobian = gradients.jacobian(y, x)

    expected_shape = [2, 3, 3, 4]
    self.assertAllEqual(expected_shape, jacobian.shape.as_list())

  def test_jacobian_unknown_shape(self):
    # [FUNCTION DOC]
    # Scenario: input is a placeholder with unknown dimensions, fed a 2x2 value in a session.
    # The pfor and loop jacobians of x transposed times x must both match the reference built
    # from per-element gradients.
    with self.cached_session() as sess:
      x = array_ops.placeholder(dtypes.float32, shape=[None, None])
      y = math_ops.matmul(x, x, transpose_a=True)
      jacobian_pfor = gradients.jacobian(y, x, use_pfor=True)
      jacobian_while = gradients.jacobian(y, x, use_pfor=False)
      answer = ops.convert_to_tensor([[
          gradient_ops.gradients(y[0][0], x)[0],
          gradient_ops.gradients(y[0][1], x)[0]
      ], [
          gradient_ops.gradients(y[1][0], x)[0],
          gradient_ops.gradients(y[1][1], x)[0]
      ]])
      ans, pfor_value, while_value = sess.run(
          [answer, jacobian_pfor, jacobian_while],
          feed_dict={x: [[1, 2], [3, 4]]})
      self.assertAllClose(ans, pfor_value)
      self.assertAllClose(ans, while_value)

  def test_jacobian_parallel_iterations(self):
    # [FUNCTION DOC]
    # Scenario: jacobian of a 2x2 matrix product computed with parallel_iterations of 2 and of 3.
    # The two results must be close (the knob must not change values).
    x = constant_op.constant([[1., 2], [3, 4]])
    y = math_ops.matmul(x, x)
    self.assertAllClose(gradients.jacobian(y, x, parallel_iterations=2),
                        gradients.jacobian(y, x, parallel_iterations=3))

  def test_batch_jacobian_bad_shapes(self):
    # [FUNCTION DOC]
    # Scenario: output has batch 3 and input batch 2. Expects batch_jacobian to raise a value
    # error whose message mentions the first dimension of the output.
    x = random_ops.random_uniform([2, 2])
    y = random_ops.random_uniform([3, 2])
    with self.assertRaisesRegex(ValueError, "Need first dimension of `output`"):
      gradients.batch_jacobian(y, x, use_pfor=True)

  def test_batch_jacobian_bad_unknown_shapes(self):
    # [FUNCTION DOC]
    # Scenario: input is a placeholder of unknown shape and output concatenates it with itself, so
    # the batch sizes differ. Graph building succeeds; running with a 2x2 feed is expected to
    # raise an invalid-argument error mentioning a failed assertion.
    with self.cached_session() as sess:
      x = array_ops.placeholder(dtypes.float32)
      y = array_ops.concat([x, x], axis=0)
      jacobian = gradients.batch_jacobian(y, x)
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  "assertion failed"):
        sess.run(jacobian, feed_dict={x: [[1, 2], [3, 4]]})

  def test_batch_jacobian_fixed_shape(self):
    # [FUNCTION DOC]
    # Scenario: output is the elementwise square of a random 2x3x5 tensor. Both the pfor and loop
    # batch jacobians must equal a stack of per-example diagonal matrices of twice the input.
    x = random_ops.random_uniform([2, 3, 5])
    y = x * x
    batch_jacobian_pfor = gradients.batch_jacobian(y, x, use_pfor=True)
    batch_jacobian_while = gradients.batch_jacobian(y, x, use_pfor=False)
    two_x = 2 * x
    answer = array_ops_stack.stack(
        [array_ops.diag(two_x[0]),
         array_ops.diag(two_x[1])])
    self.run_and_assert_equal(answer, batch_jacobian_pfor)
    self.run_and_assert_equal(answer, batch_jacobian_while)

  def test_batch_jacobian_unknown_shape(self):
    # [FUNCTION DOC]
    # Scenario: unknown-shape placeholder squared elementwise, fed a 2x2 value. The pfor and loop
    # batch jacobians evaluated in a session must match stacked diagonals of twice the input.
    with self.cached_session() as sess:
      x = array_ops.placeholder(dtypes.float32)
      y = x * x
      batch_jacobian_pfor = gradients.batch_jacobian(y, x, use_pfor=True)
      batch_jacobian_while = gradients.batch_jacobian(y, x, use_pfor=False)
      two_x = 2 * x
      answer = array_ops_stack.stack(
          [array_ops.diag(two_x[0]),
           array_ops.diag(two_x[1])])
      ans, pfor_value, while_value = sess.run(
          [answer, batch_jacobian_pfor, batch_jacobian_while],
          feed_dict={x: [[1, 2], [3, 4]]})
      self.assertAllClose(ans, pfor_value)
      self.assertAllClose(ans, while_value)

  def test_batch_jacobian_parallel_iterations(self):
    # [FUNCTION DOC]
    # Scenario: batch jacobian of a matrix product with constant inputs, computed with
    # parallel_iterations of 2 and of 3. The two results must be close.
    x = constant_op.constant([[1., 2], [3, 4]])
    w = constant_op.constant([[1., 2, 3, 4], [5, 6, 7, 8]])
    y = math_ops.matmul(x, w)
    self.assertAllClose(gradients.batch_jacobian(y, x, parallel_iterations=2),
                        gradients.batch_jacobian(y, x, parallel_iterations=3))

  def test_fc_batch_jacobian(self):
    # [FUNCTION DOC]
    # Scenario: fully connected model, batch 8, width 4, two layers. The pfor and loop batch
    # jacobians must agree.
    pfor_jacobian, while_jacobian = create_fc_batch_jacobian(8, 4, 2)
    self.run_and_assert_equal(pfor_jacobian, while_jacobian)

  def test_lstm_batch_jacobian(self):
    # [FUNCTION DOC]
    # Scenario: two-step LSTM with state size 4, batch 8 and input width 128. The pfor and loop
    # batch jacobians must agree.
    pfor_jacobian, while_jacobian = create_lstm_batch_jacobian(8, 4, 2,
                                                               inputs_size=128)
    self.run_and_assert_equal(pfor_jacobian, while_jacobian)

  @test_util.disable_xla("This test never passed for XLA")
  def DISABLED_test_dynamic_lstm_batch_jacobian(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name has a DISABLED prefix so the test runner never collects or runs it
    # (dead test; the decorator notes it never passed under XLA). Also the loop over four units
    # is hard-coded and assumes state size 4.
    # Scenario: dynamic LSTM with batch 8, state 4, three steps. Initializes variables and
    # evaluates the pfor batch jacobian; each per-unit loop gradient is evaluated in a session and
    # must match the corresponding slice of the pfor jacobian.
    pfor_jacobian, while_gradients = create_dynamic_lstm_batch_jacobian(8, 4, 3)
    with session.Session() as sess:
      init = variables.global_variables_initializer()
      self.evaluate(init)
      pfor = self.evaluate(pfor_jacobian)
      for i in range(4):
        while_i = sess.run(while_gradients[i])
        self.assertAllClose(while_i, pfor[:, i, ...])

  def test_lstm_hessian(self):
    # [FUNCTION DOC]
    # Scenario: tiny LSTM (batch 2, state 2, two steps). The hessians with respect to trainable
    # variables from the pfor and loop paths must agree.
    pfor_hessian, while_hessian = create_lstm_hessian(2, 2, 2)
    self.run_and_assert_equal(pfor_hessian, while_hessian)

  def test_lstm_batch_hessian(self):
    # [FUNCTION DOC]
    # Scenario: tiny LSTM (batch 2, state 2, two steps). The batch hessians from the pfor and
    # loop paths must agree.
    pfor_hessian, while_hessian = create_lstm_batch_hessian(2, 2, 2)
    self.run_and_assert_equal(pfor_hessian, while_hessian)

  def test_fc_per_eg_grad(self):
    # [FUNCTION DOC]
    # Scenario: fully connected model (batch 8, width 4, two layers). Per-example gradients from
    # pfor and from the loop must agree.
    pfor_outputs, while_outputs = create_fc_per_eg_grad(8, 4, 2)
    self.run_and_assert_equal(pfor_outputs, while_outputs)

  def test_lstm_per_eg_grad(self):
    # [FUNCTION DOC]
    # Scenario: LSTM (batch 8, state 4, two steps). Per-example gradients from pfor and from the
    # loop must agree.
    pfor_outputs, while_outputs = create_lstm_per_eg_grad(8, 4, 2)
    self.run_and_assert_equal(pfor_outputs, while_outputs)

  def test_mnist_per_eg_grad(self):
    # [FUNCTION DOC]
    # Scenario: MNIST model, four examples, inference mode so dropout is deterministic. Sets an
    # environment variable disabling non-fused Winograd convolution during the test (side effect)
    # and removes it afterwards. Uses channels first if a GPU is available. Per-example gradients
    # from pfor and the loop must agree within a relative tolerance of 1e-3.
    # KNOWN ISSUE: the variable is not removed if the assertion fails, since cleanup is not in a
    # finally block.
    # It looks like CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED
    # configuration of Winograd can cause low precision output resulting in
    # tests failing. So we disable that here.
    os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "0"
    data_format = ("channels_first"
                   if test.is_gpu_available() else "channels_last")
    # Note that we are setting training=False here so that dropout produces
    # the same result with pfor and with while_loop.
    pfor_outputs, while_outputs = create_mnist_per_eg_grad(
        4, data_format, training=False)
    self.run_and_assert_equal(pfor_outputs, while_outputs, rtol=1e-3)
    os.environ.pop("TF_ENABLE_WINOGRAD_NONFUSED", None)

  def test_mnist_per_eg_jacobian(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the environment variable set for the test is not removed if an assertion fails
    # (no finally block), so it can leak into later tests.
    # Scenario: MNIST model, two examples, inference mode. Disables non-fused Winograd convolution
    # through an environment variable (side effect), picks channels first only if a GPU exists,
    # and asserts the per-example jacobians from pfor and the loop agree within relative
    # tolerance 1e-3; then removes the variable.
    # It looks like CUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED
    # configuration of Winograd can cause low precision output resulting in
    # tests failing. So we disable that here.
    os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "0"
    data_format = ("channels_first"
                   if test.is_gpu_available() else "channels_last")
    # Note that we are setting training=False here so that dropout produces
    # the same result with pfor and with while_loop.
    pfor_outputs, while_outputs = create_mnist_per_eg_jacobian(
        2, data_format, training=False)
    self.run_and_assert_equal(pfor_outputs, while_outputs, rtol=1e-3)
    os.environ.pop("TF_ENABLE_WINOGRAD_NONFUSED", None)

  def test_fc_jacobian(self):
    # [FUNCTION DOC]
    # Scenario: fully connected model (batch 8, width 4, two layers). The whole-batch jacobian
    # with respect to trainable variables must match the per-example jacobians computed with pfor
    # and with a loop (relative tolerance 2e-3, absolute 1e-3).
    jacobians, per_eg_jacobians_pfor, per_eg_jacobians_while = (
        create_fc_per_eg_jacobians(batch_size=8,
                                   activation_size=4,
                                   num_layers=2))
    self.run_and_assert_equal(jacobians, per_eg_jacobians_pfor,
                              rtol=2e-3, atol=1e-3)
    self.run_and_assert_equal(jacobians, per_eg_jacobians_while,
                              rtol=2e-3, atol=1e-3)

  def test_indexed_slice(self):
    # [FUNCTION DOC]
    # Scenario: output is an embedding lookup (sparse gradient) of rows 0 and 2 of a 3x2 random
    # table. The jacobian computed with the loop must equal the one from pfor (dense results).
    inp = random_ops.random_uniform([3, 2])
    output = nn.embedding_lookup(inp, [0, 2])
    pfor_jacobian = gradients.jacobian(output, inp, use_pfor=True)
    while_jacobian = gradients.jacobian(output, inp, use_pfor=False)
    self.run_and_assert_equal(while_jacobian, pfor_jacobian)


class GradientsBenchmarks(test.Benchmark):
  # [CLASS DOC]
  # Benchmarks comparing pfor against sequential loops for jacobians, hessians, per-example
  # gradients and jacobians on fully connected, LSTM and MNIST models. Each benchmark builds the
  # graph in a fresh graph and reports wall time through the shared _run helper.

  def _run(self, targets, iters, name=None):
    # [FUNCTION DOC]
    # HELPER: timing driver for the benchmarks in this class.
    # Reduces each target to its first element (so GPU work is forced to finish), creates a
    # session, initializes variables, runs once as warm-up, then runs iters times and measures
    # wall time. Reports the benchmark under name (optional) in milliseconds per iteration and
    # returns that average.

    def _done(t):
      # [FUNCTION DOC]
      # HELPER: reduces a tensor to a single scalar (its first flattened element) to force the
      # computation to complete; assumes the tensor is not empty.
      # Note that we don't use tf.control_dependencies since that will not make
      # sure that the computation on GPU has actually finished. So we fetch the
      # first element of the output, and assume that this will not be called on
      # empty tensors.
      return array_ops.gather(array_ops.reshape(t, [-1]), 0)

    targets = [_done(x) for x in nest.flatten(targets)]
    sess = session.Session()
    with sess:
      init = variables.global_variables_initializer()
      self.evaluate(init)
      self.evaluate(targets)
      begin = time.time()
      for _ in range(iters):
        self.evaluate(targets)
      end = time.time()
    avg_time_ms = (1000 * (end - begin)) / iters
    self.report_benchmark(iters=iters, wall_time=avg_time_ms, name=name)
    return avg_time_ms

  def benchmark_fc_batch_jacobian(self):
    # [FUNCTION DOC]
    # Benchmark: batch jacobian of a 20-layer fully connected model (batch 100, width 32), pfor
    # (100 iterations) versus loop (20 iterations), in a fresh graph.
    with ops.Graph().as_default():
      pfor_jacobian, while_jacobian = create_fc_batch_jacobian(100, 32, 20)
      self._run(pfor_jacobian, 100, name="fc_batch_jacobian_pfor")
      self._run(while_jacobian, 20, name="fc_batch_jacobian_while")

  def benchmark_lstm_batch_jacobian(self):
    # [FUNCTION DOC]
    # Benchmark: batch jacobian of an eight-step LSTM (batch 100, state 32, input width 128),
    # pfor versus loop, in a fresh graph.
    with ops.Graph().as_default():
      pfor_jacobian, while_jacobian = create_lstm_batch_jacobian(
          100, 32, 8, inputs_size=128)
      self._run(pfor_jacobian, 100, name="lstm_batch_jacobian_pfor")
      self._run(while_jacobian, 20, name="lstm_batch_jacobian_while")

  def benchmark_lstm_hessian(self):
    # [FUNCTION DOC]
    # Benchmark: hessian of a ten-step LSTM (batch 2, state 2) with pfor versus the loop variant
    # (which reuses pfor jacobians), in a fresh graph.
    with ops.Graph().as_default():
      pfor_hessian, while_hessian = create_lstm_hessian(2, 2, 10)
      self._run(pfor_hessian, 20, name="lstm_hessian_pfor")
      self._run(while_hessian, 3, name="lstm_hessian_while_pfor")

  def benchmark_lstm_batch_hessian(self):
    # [FUNCTION DOC]
    # Benchmark: batch hessian of a ten-step LSTM (batch 4, state 4), pfor versus loop variant,
    # in a fresh graph.
    with ops.Graph().as_default():
      pfor_hessian, while_hessian = create_lstm_batch_hessian(4, 4, 10)
      self._run(pfor_hessian, 100, name="lstm_batch_hessian_pfor")
      self._run(while_hessian, 20, name="lstm_batch_hessian_while_pfor")

  def benchmark_fc_per_eg_grad(self):
    # [FUNCTION DOC]
    # Benchmark: per-example gradients of a three-layer fully connected model (batch 100, width
    # 32), pfor versus loop, in a fresh graph.
    with ops.Graph().as_default():
      pfor_outputs, while_outputs = create_fc_per_eg_grad(100, 32, 3)
      self._run(pfor_outputs, 100, name="fc_per_eg_grad_pfor")
      self._run(while_outputs, 20, name="fc_per_eg_grad_while")

  def benchmark_lstm_per_eg_grad(self):
    # [FUNCTION DOC]
    # Benchmark: per-example gradients of an eight-step LSTM (batch 100, state 32), pfor versus
    # loop, in a fresh graph.
    with ops.Graph().as_default():
      pfor_outputs, while_outputs = create_lstm_per_eg_grad(100, 32, 8)
      self._run(pfor_outputs, 100, name="lstm_per_eg_grad_pfor")
      self._run(while_outputs, 20, name="lstm_per_eg_grad_while")

  def benchmark_mnist_autobatch(self):
    # [FUNCTION DOC]
    # Benchmark: MNIST inference on 100 images, comparing automatic batching with pfor, a
    # sequential loop, and a manual batched call, in a fresh graph (channels first on GPU).
    with ops.Graph().as_default():
      data_format = ("channels_first"
                     if test.is_gpu_available() else "channels_last")
      pfor_outputs, while_outputs, manual = create_mnist_autobatch(
          100, data_format, training=False)
      self._run(pfor_outputs, 100, name="mnist_pfor")
      self._run(while_outputs, 20, name="mnist_while")
      self._run(manual, 100, name="mnist_manual")

  def benchmark_mnist_per_eg_grad(self):
    # [FUNCTION DOC]
    # Benchmark: per-example gradients of the MNIST model in training mode (batch 128), pfor
    # versus loop, in a fresh graph.
    with ops.Graph().as_default():
      data_format = ("channels_first"
                     if test.is_gpu_available() else "channels_last")
      pfor_outputs, while_outputs = create_mnist_per_eg_grad(
          128, data_format, training=True)
      self._run(pfor_outputs, 20, name="mnist_per_eg_grad_pfor")
      self._run(while_outputs, 20, name="mnist_per_eg_grad_while")

  def benchmark_mnist_per_eg_jacobian(self):
    # [FUNCTION DOC]
    # Benchmark: per-example jacobians of the MNIST model in training mode (batch 16), pfor
    # versus loop, in a fresh graph (channels first on GPU).
    with ops.Graph().as_default():
      if test.is_gpu_available():
        data_format = "channels_first"
      else:
        data_format = "channels_last"
      pfor_outputs, while_outputs = create_mnist_per_eg_jacobian(
          16, data_format, training=True)
      self._run(pfor_outputs, 20, name="mnist_per_eg_jacobian_pfor")
      self._run(while_outputs, 20, name="mnist_per_eg_jacobian_while")

  def benchmark_mnist_batch_jacobian(self):
    # [FUNCTION DOC]
    # Benchmark: batch jacobian of the MNIST logits with respect to the images in training mode
    # (batch 128), pfor versus loop, in a fresh graph (channels first on GPU).
    with ops.Graph().as_default():
      if test.is_gpu_available():
        data_format = "channels_first"
      else:
        data_format = "channels_last"
      pfor_outputs, while_outputs = create_mnist_batch_jacobian(
          128, data_format, training=True)
      self._run(pfor_outputs, 20, name="mnist_batch_jacobian_pfor")
      self._run(while_outputs, 20, name="mnist_batch_jacobian_while")

  def benchmark_fc_per_eg_jacobian(self):
    # [FUNCTION DOC]
    # Benchmark: for a three-layer fully connected model (batch 128, width 32), times the full
    # jacobian, the per-example jacobians with pfor, and the per-example jacobians with a loop,
    # in a fresh graph.
    with ops.Graph().as_default():
      jacobians, per_eg_jacobians_pfor, per_eg_jacobians_while = (
          create_fc_per_eg_jacobians(batch_size=128,
                                     activation_size=32,
                                     num_layers=3))
      self._run(jacobians, 30, name="fc_jacobians_pfor")
      self._run(per_eg_jacobians_pfor, 100,
                name="fc_per_eg_jacobians_pfor")
      self._run(per_eg_jacobians_while, 10,
                name="fc_per_eg_jacobians_while")


if __name__ == "__main__":
  test.main()
