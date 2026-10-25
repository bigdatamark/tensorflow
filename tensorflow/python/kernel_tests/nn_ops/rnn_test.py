# [MODULE DOC]
# Unit tests and benchmarks for the legacy recurrent network driver (rnn module) and the RNN cell
# implementations. Defines several tiny test cells (plus one, scalar state, unbalanced tuple
# output, tensor-array state), the RNNTest case (validation errors, batch size inference, state
# types, initial state creation, cell building, checkpoint interchange between LSTM variants),
# module-level benchmark graph builders and timers, and a benchmark class. Importing the gradient
# modules registers gradient functions. Running as a script starts the test runner.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for rnn module."""

import os
import time
import timeit

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables as variables_lib
import tensorflow.python.ops.data_flow_grad  # pylint: disable=unused-import
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
import tensorflow.python.ops.sparse_grad  # pylint: disable=unused-import
import tensorflow.python.ops.tensor_array_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.training import saver


class Plus1RNNCell(rnn_cell_impl.RNNCell):
  # [CLASS DOC]
  # HELPER: tiny recurrent cell used by the tests in this module as a simple cell with
  # predictable behaviour. Output size and state size are both five; each step returns the input
  # plus one as output and the state plus one as the new state. Holds no variables.
  """RNN Cell generating (output, new_state) = (input + 1, state + 1)."""

  @property
  def output_size(self):
    # [FUNCTION DOC]
    # Property: size of the cell output, always five.
    return 5

  @property
  def state_size(self):
    # [FUNCTION DOC]
    # Property: size of the cell state, always five.
    return 5

  def call(self, input_, state, scope=None):
    # [FUNCTION DOC]
    # One time step. Inputs: input tensor, state tensor, and an unused optional scope. Returns a
    # pair of the input plus one and the state plus one. No side effects.
    return (input_ + 1, state + 1)


class ScalarStateRNNCell(rnn_cell_impl.RNNCell):
  # [CLASS DOC]
  # HELPER: test cell whose state is a single scalar integer counter. Output size is one; each
  # step passes the input through unchanged and increments the state.
  # KNOWN ISSUE: the docstring claims the output is the input plus one, but the code returns the
  # input unchanged.
  """RNN Cell generating (output, new_state) = (input + 1, state + 1)."""

  @property
  def output_size(self):
    # [FUNCTION DOC]
    # Property: size of the cell output, always one.
    return 1

  @property
  def state_size(self):
    # [FUNCTION DOC]
    # Property: the state is a scalar, so returns an empty (rank-zero) tensor shape.
    return tensor_shape.TensorShape([])

  def zero_state(self, batch_size, dtype):
    # [FUNCTION DOC]
    # KNOWN ISSUE: ignores the dtype argument and always returns an int32 scalar.
    # Inputs: batch size and dtype (both ignored). Returns a scalar integer zero tensor as the
    # initial state.
    return array_ops.zeros([], dtype=dtypes.int32)

  def call(self, input_, state, scope=None):
    # [FUNCTION DOC]
    # One time step. Inputs: input tensor, state tensor, and an unused optional scope. Returns a
    # pair of the unchanged input and the state plus one. No side effects.
    return (input_, state + 1)


class UnbalancedOutputRNNCell(rnn_cell_impl.RNNCell):
  # [CLASS DOC]
  # HELPER: test cell with a nested (tuple) output of different sizes: the input itself (size
  # one) and the input concatenated with itself (size two), and a scalar integer counter state
  # that is incremented every step.
  # KNOWN ISSUE: the docstring claims the output is the input plus one, but the code returns a
  # tuple of the input and the doubled input.
  """RNN Cell generating (output, new_state) = (input + 1, state + 1)."""

  @property
  def output_size(self):
    # [FUNCTION DOC]
    # Property: returns a pair of tensor shapes, one of length one and one of length two, i.e. an
    # unbalanced nested output structure.
    return  tensor_shape.TensorShape(1), tensor_shape.TensorShape((2))

  @property
  def state_size(self):
    # [FUNCTION DOC]
    # Property: the state is a scalar, so returns an empty (rank-zero) tensor shape.
    return tensor_shape.TensorShape([])

  def zero_state(self, batch_size, dtype):
    # [FUNCTION DOC]
    # KNOWN ISSUE: ignores the dtype argument and always returns an int32 scalar.
    # Inputs: batch size and dtype (both ignored). Returns a scalar integer zero tensor as the
    # initial state.
    return array_ops.zeros([], dtype=dtypes.int32)

  def call(self, input_, state, scope=None):
    # [FUNCTION DOC]
    # One time step. Inputs: input tensor, state tensor, and an unused optional scope. Concatenates
    # the input with itself along the last axis. Returns a pair: a tuple of the input and the
    # concatenation as the output, and the state plus one as the new state.
    concatenated = array_ops.concat((input_, input_), axis=-1)
    return (input_, concatenated), state + 1


class TensorArrayStateRNNCell(rnn_cell_impl.RNNCell):
  # [CLASS DOC]
  # HELPER: test cell whose state is a pair of an integer step counter and a dynamically sized
  # tensor array that records every input. Output size is one; the output is the input unchanged.
  """RNN Cell its state as a TensorArray."""

  @property
  def output_size(self):
    # [FUNCTION DOC]
    # Property: size of the cell output, always one.
    return 1

  @property
  def state_size(self):
    # [FUNCTION DOC]
    # Property: returns a pair of a scalar tensor shape (the counter) and an empty tuple standing
    # in for the tensor array part.
    return (tensor_shape.TensorShape([]), ())

  def zero_state(self, batch_size, dtype):
    # [FUNCTION DOC]
    # Inputs: batch size (ignored) and dtype used for the array elements. Returns a pair: an int32
    # scalar zero counter and an empty, dynamically growing tensor array of the given dtype.
    return (array_ops.zeros([], dtype=dtypes.int32),
            tensor_array_ops.TensorArray(
                dtype=dtype, size=0, dynamic_size=True))

  def call(self, input_, state, scope=None):
    # [FUNCTION DOC]
    # One time step. Inputs: input tensor, state pair (counter, tensor array), unused optional
    # scope. Writes the input into the array at the index given by the counter. Returns a pair:
    # the unchanged input as output and the new state (counter plus one, updated array).
    new_array = state[1].write(state[0], input_)
    return (input_, (state[0] + 1, new_array))


class RNNTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the dynamic RNN driver and the recurrent cell base class: sequence length and
  # dtype validation, batch size inference, scalar / unbalanced / tensor-array states, initial
  # state validation, cell building, and checkpoint interchange between two LSTM cell variants.
  # Many tests run in both graph and eager modes.

  def setUp(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: does not call the parent setUp, so the base test fixture setup is skipped.
    # Fixture: stores a fixed seed on the instance and seeds the global NumPy random generator
    # with it for reproducibility. Returns nothing.
    self._seed = 23489
    np.random.seed(self._seed)

  @test_util.run_in_graph_and_eager_modes
  def testInvalidSequenceLengthShape(self):
    # [FUNCTION DOC]
    # Scenario: runs the dynamic RNN with a simple cell and a two-dimensional sequence-length
    # value (not a vector). Expects a value error saying it must be a vector. Runs in graph and
    # eager modes (constant input in eager mode, placeholder in graph mode).
    cell = Plus1RNNCell()
    if context.executing_eagerly():
      inputs = [constant_op.constant(np.ones((3, 4)))]
    else:
      inputs = [array_ops.placeholder(dtypes.float32, shape=(3, 4))]
    with self.assertRaisesRegex(ValueError, "must be a vector"):
      rnn.dynamic_rnn(
          cell,
          array_ops_stack.stack(inputs),
          dtype=dtypes.float32,
          sequence_length=[[4]])

  @test_util.run_in_graph_and_eager_modes
  def testInvalidDtype(self):
    # [FUNCTION DOC]
    # Scenario: for each of the basic RNN, GRU, basic LSTM and LSTM cells, builds the cell with
    # int32 dtype and runs the dynamic RNN on int32 input. Expects a value error saying the cell
    # only supports floating point. Runs in graph and eager modes.
    if context.executing_eagerly():
      inputs = np.zeros((3, 4, 5), dtype=np.int32)
    else:
      inputs = array_ops.placeholder(dtypes.int32, shape=(3, 4, 5))

    cells = [
        rnn_cell_impl.BasicRNNCell,
        rnn_cell_impl.GRUCell,
        rnn_cell_impl.BasicLSTMCell,
        rnn_cell_impl.LSTMCell,
    ]
    for cell_cls in cells:
      with self.cached_session():
        with self.assertRaisesRegex(ValueError,
                                    "RNN cell only supports floating"):
          cell = cell_cls(2, dtype=dtypes.int32)
          rnn.dynamic_rnn(cell, inputs, dtype=dtypes.int32)

  @test_util.run_in_graph_and_eager_modes
  def testBatchSizeFromInput(self):
    # [FUNCTION DOC]
    # Scenario: runs the dynamic RNN with the simple plus-one cell, with and without an explicit
    # initial state, and asserts the batch dimension (three) of the output and final state. In
    # graph mode it also repeats with an unknown batch dimension and asserts the output and state
    # batch dimensions are unknown. Runs in graph and eager modes.
    cell = Plus1RNNCell()
    in_eager_mode = context.executing_eagerly()
    # With static batch size
    if in_eager_mode:
      inputs = np.zeros((3, 4, 5), dtype=np.float32)
      initial_state = np.zeros((3, 5), dtype=np.float32)
    else:
      inputs = array_ops.placeholder(dtypes.float32, shape=(3, 4, 5))
      initial_state = array_ops.placeholder(dtypes.float32, shape=(3, 5))

    # - Without initial_state
    outputs, state = rnn.dynamic_rnn(cell, inputs, dtype=dtypes.float32)
    self.assertEqual(3, outputs.shape[0])
    self.assertEqual(3, state.shape[0])

    # - With initial_state
    outputs, state = rnn.dynamic_rnn(
        cell, inputs, initial_state=initial_state)
    self.assertEqual(3, outputs.shape[0])
    self.assertEqual(3, state.shape[0])

    # Without static batch size
    # Tensor shapes are fully determined with eager execution enabled,
    # so only run this test for graph construction.
    if not in_eager_mode:
      inputs = array_ops.placeholder(dtypes.float32, shape=(None, 4, 5))
      # - Without initial_state
      outputs, state = rnn.dynamic_rnn(cell, inputs, dtype=dtypes.float32)
      self.assertEqual(None, outputs.shape.dims[0].value)
      self.assertEqual(None, state.shape.dims[0].value)
      # - With initial_state
      outputs, state = rnn.dynamic_rnn(
          cell,
          inputs,
          initial_state=array_ops.placeholder(dtypes.float32, shape=(None, 5)))
      self.assertEqual(None, outputs.shape.dims[0].value)
      self.assertEqual(None, state.shape.dims[0].value)

  @test_util.run_in_graph_and_eager_modes
  def testScalarStateIsAccepted(self):
    # [FUNCTION DOC]
    # Scenario: runs the dynamic RNN with the scalar-state cell on one sequence of four steps of
    # length-one inputs with sequence length four. Asserts the outputs equal the inputs and the
    # final state equals four (the number of steps). Runs in graph and eager modes.
    cell = ScalarStateRNNCell()
    in_eager_mode = context.executing_eagerly()

    if in_eager_mode:
      inputs = np.array([[[1], [2], [3], [4]]], dtype=np.float32)
    else:
      inputs = array_ops.placeholder(dtypes.float32, shape=(1, 4, 1))

    with self.cached_session() as sess:
      outputs, state = rnn.dynamic_rnn(
          cell, inputs, dtype=dtypes.float32, sequence_length=[4])
      if not in_eager_mode:
        outputs, state = sess.run(
            [outputs, state], feed_dict={inputs: [[[1], [2], [3], [4]]]})

    self.assertAllEqual([[[1], [2], [3], [4]]], outputs)
    self.assertAllEqual(4, state)

  @test_util.run_in_graph_and_eager_modes
  def testUnbalancedOutputIsAccepted(self):
    # [FUNCTION DOC]
    # Scenario: runs the dynamic RNN with the cell returning a tuple output of different sizes.
    # Asserts the outputs are a tuple, the first element equals the inputs, the second equals the
    # inputs doubled along the last axis, and the final state is four. Runs in graph and eager
    # modes.
    cell = UnbalancedOutputRNNCell()
    in_eager_mode = context.executing_eagerly()

    if in_eager_mode:
      inputs = np.array([[[1], [2], [3], [4]]], dtype=np.float32)
    else:
      inputs = array_ops.placeholder(dtypes.float32, shape=(1, 4, 1))

    with self.cached_session() as sess:
      outputs, state = rnn.dynamic_rnn(
          cell, inputs, dtype=dtypes.float32, sequence_length=[4])
      if not in_eager_mode:
        outputs, state = sess.run(
            [outputs, state], feed_dict={inputs: [[[1], [2], [3], [4]]]})

    self.assertIsInstance(outputs, tuple)
    self.assertAllEqual([[[1], [2], [3], [4]]], outputs[0])
    self.assertAllEqual([[[1, 1], [2, 2], [3, 3], [4, 4]]], outputs[1])
    self.assertAllEqual(4, state)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testEagerMemory(self):
    # [FUNCTION DOC]
    # Scenario: in eager mode, runs the dynamic RNN with the tensor-array-state cell once. The
    # decorator asserts no new Python objects leak after execution; there are no explicit
    # assertions in the body.
    with context.eager_mode():
      cell = TensorArrayStateRNNCell()
      inputs = np.array([[[1], [2], [3], [4]]], dtype=np.float32)
      rnn.dynamic_rnn(cell, inputs, dtype=dtypes.float32, sequence_length=[4])

  @test_util.run_in_graph_and_eager_modes
  @test_util.run_v1_only("b/120545219")
  def testTensorArrayStateIsAccepted(self):
    # [FUNCTION DOC]
    # Scenario: runs the dynamic RNN with the tensor-array-state cell over four steps, stacks the
    # array in the final state, and asserts the outputs equal the inputs, the counter is four and
    # the stacked array holds the four inputs in order. Runs in graph and eager modes; skipped
    # for non-v1 behaviour via the v1-only decorator (a known bug reference).
    cell = TensorArrayStateRNNCell()
    in_eager_mode = context.executing_eagerly()

    if in_eager_mode:
      inputs = np.array([[[1], [2], [3], [4]]], dtype=np.float32)
    else:
      inputs = array_ops.placeholder(dtypes.float32, shape=(1, 4, 1))

    with self.cached_session() as sess:
      outputs, state = rnn.dynamic_rnn(
          cell, inputs, dtype=dtypes.float32, sequence_length=[4])
      state = (state[0], state[1].stack())
      if not in_eager_mode:
        outputs, state = sess.run(
            [outputs, state], feed_dict={
                inputs: [[[1], [2], [3], [4]]]
            })

    self.assertAllEqual([[[1], [2], [3], [4]]], outputs)
    self.assertAllEqual(4, state[0])
    self.assertAllEqual([[[1]], [[2]], [[3]], [[4]]], state[1])

  @test_util.run_deprecated_v1
  def testCellGetInitialState(self):
    # [FUNCTION DOC]
    # Scenario: exercises initial-state creation on a basic RNN cell of five units. Expects value
    # errors when batch size and dtype are both missing, when the given batch size (python integer
    # or tensor) differs from the input's batch dimension, and when the given dtype differs from
    # the input's. Then asserts that deriving the state from the input alone, or from a tensor
    # batch size and the input dtype, gives a state of unknown batch and five units with the input
    # dtype. Graph-mode only (deprecated v1).
    cell = rnn_cell_impl.BasicRNNCell(5)
    with self.assertRaisesRegex(ValueError,
                                "batch_size and dtype cannot be None"):
      cell.get_initial_state(None, None, None)

    inputs = array_ops.placeholder(dtypes.float32, shape=(None, 4, 1))
    with self.assertRaisesRegex(
        ValueError, "batch size from input tensor is different from"):
      cell.get_initial_state(inputs=inputs, batch_size=50, dtype=None)

    with self.assertRaisesRegex(
        ValueError, "batch size from input tensor is different from"):
      cell.get_initial_state(
          inputs=inputs, batch_size=constant_op.constant(50), dtype=None)

    with self.assertRaisesRegex(ValueError,
                                "dtype from input tensor is different from"):
      cell.get_initial_state(inputs=inputs, batch_size=None, dtype=dtypes.int16)

    initial_state = cell.get_initial_state(
        inputs=inputs, batch_size=None, dtype=None)
    self.assertEqual(initial_state.shape.as_list(), [None, 5])
    self.assertEqual(initial_state.dtype, inputs.dtype)

    batch = array_ops.shape(inputs)[0]
    dtype = inputs.dtype
    initial_state = cell.get_initial_state(None, batch, dtype)
    self.assertEqual(initial_state.shape.as_list(), [None, 5])
    self.assertEqual(initial_state.dtype, inputs.dtype)

  def _assert_cell_builds(self, cell_class, dtype, batch_size, in_size,
                          out_size):
    # [FUNCTION DOC]
    # HELPER: checks that a cell class can be built and applied.
    # Inputs: cell class, dtype, batch size, input size, output size (number of units). Builds
    # the cell for the given input shape, gets an initial state, applies the cell to a zero input,
    # and asserts the output shape equals batch size by units. Returns nothing.
    cell = cell_class(out_size, dtype=dtype)
    in_shape = tensor_shape.TensorShape((batch_size, in_size))
    cell.build(in_shape)
    state_output = cell.get_initial_state(
        inputs=None, batch_size=batch_size, dtype=dtype)
    cell_output, _ = cell(array_ops.zeros(in_shape, dtype), state_output)
    self.assertAllEqual([batch_size, out_size], cell_output.shape.as_list())

  @test_util.run_in_graph_and_eager_modes
  def testCellsBuild(self):
    # [FUNCTION DOC]
    # Scenario: builds and applies the basic RNN, basic LSTM, GRU and LSTM cells in float32 and
    # float64 with batch five, input size seven and three units, via the shared build helper.
    # Runs in graph and eager modes.
    f32 = dtypes.float32
    f64 = dtypes.float64
    self._assert_cell_builds(rnn_cell_impl.BasicRNNCell, f32, 5, 7, 3)
    self._assert_cell_builds(rnn_cell_impl.BasicRNNCell, f64, 5, 7, 3)
    self._assert_cell_builds(rnn_cell_impl.BasicLSTMCell, f32, 5, 7, 3)
    self._assert_cell_builds(rnn_cell_impl.BasicLSTMCell, f64, 5, 7, 3)
    self._assert_cell_builds(rnn_cell_impl.GRUCell, f32, 5, 7, 3)
    self._assert_cell_builds(rnn_cell_impl.GRUCell, f64, 5, 7, 3)
    self._assert_cell_builds(rnn_cell_impl.LSTMCell, f32, 5, 7, 3)
    self._assert_cell_builds(rnn_cell_impl.LSTMCell, f64, 5, 7, 3)

  @test_util.run_deprecated_v1
  def testBasicLSTMCellInterchangeWithLSTMCell(self):
    # [FUNCTION DOC]
    # Scenario: in one graph, builds a basic LSTM cell with one unit, sets its bias to ten for all
    # four gates and saves a checkpoint in the test temp directory. In a fresh graph, builds an
    # LSTM cell with the basic cell's name, restores the checkpoint and asserts its bias equals the
    # saved values, showing variable-name compatibility. Side effect: writes a checkpoint file.
    # Graph-mode only (deprecated v1).
    with self.session(graph=ops_lib.Graph()) as sess:
      basic_cell = rnn_cell_impl.BasicLSTMCell(1)
      basic_cell(array_ops.ones([1, 1]),
                 state=basic_cell.get_initial_state(inputs=None,
                                                    batch_size=1,
                                                    dtype=dtypes.float32))
      self.evaluate([v.initializer for v in basic_cell.variables])
      self.evaluate(basic_cell._bias.assign([10.] * 4))
      save = saver.Saver()
      prefix = os.path.join(self.get_temp_dir(), "ckpt")
      save_path = save.save(sess, prefix)

    with self.session(graph=ops_lib.Graph()) as sess:
      lstm_cell = rnn_cell_impl.LSTMCell(1, name="basic_lstm_cell")
      lstm_cell(array_ops.ones([1, 1]),
                state=lstm_cell.get_initial_state(inputs=None,
                                                  batch_size=1,
                                                  dtype=dtypes.float32))
      self.evaluate([v.initializer for v in lstm_cell.variables])
      save = saver.Saver()
      save.restore(sess, save_path)
      self.assertAllEqual([10.] * 4, self.evaluate(lstm_cell._bias))

######### Benchmarking RNN code


def _static_vs_dynamic_rnn_benchmark_static(inputs_list_t, sequence_length):
  # [FUNCTION DOC]
  # HELPER: builds the static-RNN graph used by the benchmark comparison.
  # Inputs: list of per-step input tensors (each batch by depth) and a sequence-length value.
  # Creates a peephole LSTM cell (untupled state, seeded uniform initialiser) with units equal to
  # the input depth, runs the static RNN, takes gradients of the outputs and final state with
  # respect to all trainable variables in the default graph, and returns a grouped op depending on
  # the final state, gradients and outputs. Adds ops and variables to the current graph.
  (_, input_size) = inputs_list_t[0].get_shape().as_list()
  initializer = init_ops.random_uniform_initializer(-0.01, 0.01, seed=127)
  cell = rnn_cell_impl.LSTMCell(
      num_units=input_size,
      use_peepholes=True,
      initializer=initializer,
      state_is_tuple=False)
  outputs, final_state = rnn.static_rnn(
      cell,
      inputs_list_t,
      sequence_length=sequence_length,
      dtype=dtypes.float32)

  trainable_variables = ops_lib.get_collection(
      ops_lib.GraphKeys.TRAINABLE_VARIABLES)
  gradients = gradients_impl.gradients(outputs + [final_state],
                                       trainable_variables)

  return control_flow_ops.group(final_state, *(gradients + outputs))


def _static_vs_dynamic_rnn_benchmark_dynamic(inputs_t, sequence_length):
  # [FUNCTION DOC]
  # HELPER: builds the dynamic-RNN graph used by the benchmark comparison.
  # Inputs: a batch-by-time-by-depth input tensor and a sequence-length value. Creates the same
  # peephole LSTM cell as the static helper, runs the dynamic RNN, takes gradients of outputs and
  # final state with respect to all trainable variables in the default graph, and returns a
  # grouped op depending on the final state, outputs and gradients. Adds ops to the current graph.
  (unused_0, unused_1, input_size) = inputs_t.get_shape().as_list()
  initializer = init_ops.random_uniform_initializer(-0.01, 0.01, seed=127)
  cell = rnn_cell_impl.LSTMCell(
      num_units=input_size,
      use_peepholes=True,
      initializer=initializer,
      state_is_tuple=False)
  outputs, final_state = rnn.dynamic_rnn(
      cell, inputs_t, sequence_length=sequence_length, dtype=dtypes.float32)

  trainable_variables = ops_lib.get_collection(
      ops_lib.GraphKeys.TRAINABLE_VARIABLES)
  gradients = gradients_impl.gradients([outputs, final_state],
                                       trainable_variables)

  return control_flow_ops.group(final_state, outputs, *gradients)


def graph_creation_static_vs_dynamic_rnn_benchmark(max_time):
  # [FUNCTION DOC]
  # Benchmark helper comparing graph-construction time of static and dynamic RNNs.
  # Input: number of time steps. Seeds the global random generator, makes random sequence lengths
  # and random float inputs (batch and units fixed at 512), then times five constructions of each
  # graph kind with the standard timer and prints the step count, both total times and their
  # ratio. Returns a pair of the total static and total dynamic construction times in seconds;
  # side effects are printing and creating temporary sessions and graphs.
  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  # These parameters don't matter
  batch_size = 512
  num_units = 512

  # Set up sequence lengths
  np.random.seed([127])
  sequence_length = np.random.randint(0, max_time, size=batch_size)
  inputs_list = [
      np.random.randn(batch_size, num_units).astype(np.float32)
      for _ in range(max_time)
  ]
  inputs = np.dstack(inputs_list).transpose([0, 2, 1])  # batch x time x depth

  def _create_static_rnn():
    # [FUNCTION DOC]
    # HELPER: timed unit of work. In a fresh graph and session, wraps each input in a non-trainable
    # variable and builds the static-RNN benchmark graph. Takes nothing, returns nothing.
    with session.Session(config=config, graph=ops_lib.Graph()):
      inputs_list_t = [
          variables_lib.Variable(
              x, trainable=False).value() for x in inputs_list
      ]
      _static_vs_dynamic_rnn_benchmark_static(inputs_list_t, sequence_length)

  def _create_dynamic_rnn():
    # [FUNCTION DOC]
    # HELPER: timed unit of work. In a fresh graph and session, wraps the stacked input in a
    # non-trainable variable and builds the dynamic-RNN benchmark graph. Takes nothing, returns
    # nothing.
    with session.Session(config=config, graph=ops_lib.Graph()):
      inputs_t = variables_lib.Variable(inputs, trainable=False).value()
      _static_vs_dynamic_rnn_benchmark_dynamic(inputs_t, sequence_length)

  delta_static = timeit.timeit(_create_static_rnn, number=5)
  delta_dynamic = timeit.timeit(_create_dynamic_rnn, number=5)

  print("%d \t %f \t %f \t %f" %
        (max_time, delta_static, delta_dynamic, delta_dynamic / delta_static))
  return delta_static, delta_dynamic


def _timer(sess, ops):
  # [FUNCTION DOC]
  # HELPER: shared timing routine for the benchmark functions.
  # Inputs: a session and an op to run. Runs the op twice as warm-up, then twenty timed runs
  # using wall-clock time. Returns the average seconds per run. Side effect: executes the op.
  # Warm in
  for _ in range(2):
    sess.run(ops)

  # Timing run
  runs = 20
  start = time.time()
  for _ in range(runs):
    sess.run(ops)
  end = time.time()
  return (end - start) / float(runs)


def static_vs_dynamic_rnn_benchmark(batch_size, max_time, num_units, use_gpu):
  # [FUNCTION DOC]
  # Benchmark helper comparing run time of a statically unrolled RNN with a dynamic RNN.
  # Inputs: batch size, number of time steps, units (input depth), and a flag allowing GPU
  # placement (otherwise ops are pinned to the CPU). Seeds the global random generator, creates
  # random lengths and inputs, builds each graph in its own session with variables initialised,
  # times each with the timing helper, prints a row of results. Returns a pair of average seconds
  # per run for the static and dynamic versions.
  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  # Set up sequence lengths
  np.random.seed([127])
  sequence_length = np.random.randint(0, max_time, size=batch_size)
  inputs_list = [
      np.random.randn(batch_size, num_units).astype(np.float32)
      for _ in range(max_time)
  ]
  inputs = np.dstack(inputs_list).transpose([0, 2, 1])  # batch x time x depth

  # Using rnn()
  with session.Session(config=config, graph=ops_lib.Graph()) as sess:
    with ops_lib.device("/cpu:0" if not use_gpu else None):
      inputs_list_t = [
          variables_lib.Variable(
              x, trainable=False).value() for x in inputs_list
      ]
      ops = _static_vs_dynamic_rnn_benchmark_static(inputs_list_t,
                                                    sequence_length)
    variables_lib.global_variables_initializer().run()
    delta_static = _timer(sess, ops)

  # Using dynamic_rnn()
  with session.Session(config=config, graph=ops_lib.Graph()) as sess:
    with ops_lib.device("/cpu:0" if not use_gpu else None):
      inputs_t = variables_lib.Variable(inputs, trainable=False).value()
      ops = _static_vs_dynamic_rnn_benchmark_dynamic(inputs_t, sequence_length)
    variables_lib.global_variables_initializer().run()
    delta_dynamic = _timer(sess, ops)

  print("%d \t %d \t %d \t %s \t %f \t %f \t %f" %
        (batch_size, max_time, num_units, use_gpu, delta_static, delta_dynamic,
         delta_dynamic / delta_static))

  return delta_static, delta_dynamic


def _half_seq_len_vs_unroll_half_rnn_benchmark(inputs_list_t, sequence_length):
  # [FUNCTION DOC]
  # HELPER: builds the graph for the half-sequence-length versus half-unroll benchmark.
  # Inputs: list of per-step input tensors and a sequence-length value. Identical in construction
  # to the static-RNN benchmark helper: peephole LSTM, static RNN, gradients over all trainable
  # variables, returns a grouped op over final state, gradients and outputs.
  (_, input_size) = inputs_list_t[0].get_shape().as_list()
  initializer = init_ops.random_uniform_initializer(-0.01, 0.01, seed=127)
  cell = rnn_cell_impl.LSTMCell(
      num_units=input_size,
      use_peepholes=True,
      initializer=initializer,
      state_is_tuple=False)
  outputs, final_state = rnn.static_rnn(
      cell,
      inputs_list_t,
      sequence_length=sequence_length,
      dtype=dtypes.float32)

  trainable_variables = ops_lib.get_collection(
      ops_lib.GraphKeys.TRAINABLE_VARIABLES)
  gradients = gradients_impl.gradients(outputs + [final_state],
                                       trainable_variables)

  return control_flow_ops.group(final_state, *(gradients + outputs))


def half_seq_len_vs_unroll_half_rnn_benchmark(batch_size, max_time, num_units,
                                              use_gpu):
  # [FUNCTION DOC]
  # Benchmark helper comparing a full static unroll with halved sequence lengths against a
  # half-length static unroll.
  # Inputs: batch size, number of time steps, units, and a flag allowing GPU placement (otherwise
  # CPU). Seeds the global random generator, uses full-length sequence lengths halved in both
  # runs, times each graph in its own session, prints a row of results. Returns a pair of average
  # seconds per run: half sequence length, then half unroll.
  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  # Set up sequence lengths
  np.random.seed([127])
  sequence_length = max_time * np.ones((batch_size,))
  inputs_list = [
      np.random.randn(batch_size, num_units).astype(np.float32)
      for _ in range(max_time)
  ]

  # Halve the sequence length, full static unroll
  with session.Session(config=config, graph=ops_lib.Graph()) as sess:
    with ops_lib.device("/cpu:0" if not use_gpu else None):
      inputs_list_t = [
          variables_lib.Variable(
              x, trainable=False).value() for x in inputs_list
      ]
      ops = _half_seq_len_vs_unroll_half_rnn_benchmark(inputs_list_t,
                                                       sequence_length / 2)
    variables_lib.global_variables_initializer().run()
    delta_half_seq_len = _timer(sess, ops)

  # Halve the unroll size, don't use sequence length
  with session.Session(config=config, graph=ops_lib.Graph()) as sess:
    with ops_lib.device("/cpu:0" if not use_gpu else None):
      inputs_list_t = [
          variables_lib.Variable(
              x, trainable=False).value() for x in inputs_list
      ]
      ops = _half_seq_len_vs_unroll_half_rnn_benchmark(
          inputs_list_t[:(max_time // 2)], sequence_length / 2)
    variables_lib.global_variables_initializer().run()
    delta_unroll_half = _timer(sess, ops)
  print("%d \t %d \t\t %d \t %s \t %f \t\t %f \t\t %f" %
        (batch_size, max_time, num_units, use_gpu, delta_half_seq_len,
         delta_unroll_half, delta_half_seq_len / delta_unroll_half))

  return delta_half_seq_len, delta_unroll_half


def _concat_state_vs_tuple_state_rnn_benchmark(inputs_list_t, sequence_length,
                                               state_is_tuple):
  # [FUNCTION DOC]
  # HELPER: builds the graph for the concatenated-state versus tuple-state benchmark.
  # Inputs: list of per-step input tensors, sequence lengths, and a flag choosing tuple state for
  # the peephole LSTM cell. Runs the static RNN, flattens the final state into a list, takes
  # gradients over all trainable variables, and returns a grouped op over final state, gradients
  # and outputs.
  (_, input_size) = inputs_list_t[0].get_shape().as_list()
  initializer = init_ops.random_uniform_initializer(-0.01, 0.01, seed=127)
  cell = rnn_cell_impl.LSTMCell(
      num_units=input_size,
      use_peepholes=True,
      initializer=initializer,
      state_is_tuple=state_is_tuple)
  outputs, final_state = rnn.static_rnn(
      cell,
      inputs_list_t,
      sequence_length=sequence_length,
      dtype=dtypes.float32)

  final_state = list(final_state) if state_is_tuple else [final_state]

  trainable_variables = ops_lib.get_collection(
      ops_lib.GraphKeys.TRAINABLE_VARIABLES)
  gradients = gradients_impl.gradients(outputs + final_state,
                                       trainable_variables)

  return control_flow_ops.group(*(final_state + gradients + outputs))


def concat_state_vs_tuple_state_rnn_benchmark(batch_size, max_time, num_units,
                                              use_gpu):
  # [FUNCTION DOC]
  # Benchmark helper comparing a static unrolled LSTM using concatenated state with one using
  # tuple state.
  # Inputs: batch size, number of time steps, units, and a flag allowing GPU placement (otherwise
  # CPU). Seeds the global random generator, builds each variant in its own session with full
  # sequence lengths, times them, prints a row of results. Returns a pair of average seconds per
  # run: concatenated state, then tuple state.
  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  # Set up sequence lengths
  np.random.seed([127])
  sequence_length = max_time * np.ones((batch_size,))
  inputs_list = [
      np.random.randn(batch_size, num_units).astype(np.float32)
      for _ in range(max_time)
  ]

  # Run with concatenated states (default)
  with session.Session(config=config, graph=ops_lib.Graph()) as sess:
    with ops_lib.device("/cpu:0" if not use_gpu else None):
      inputs_list_t = [
          variables_lib.Variable(
              x, trainable=False).value() for x in inputs_list
      ]
      ops = _concat_state_vs_tuple_state_rnn_benchmark(
          inputs_list_t, sequence_length, state_is_tuple=False)
    variables_lib.global_variables_initializer().run()
    delta_concat_state = _timer(sess, ops)

  # Run with tuple states (new)
  with session.Session(config=config, graph=ops_lib.Graph()) as sess:
    with ops_lib.device("/cpu:0" if not use_gpu else None):
      inputs_list_t = [
          variables_lib.Variable(
              x, trainable=False).value() for x in inputs_list
      ]
      ops = _concat_state_vs_tuple_state_rnn_benchmark(
          inputs_list_t, sequence_length, state_is_tuple=True)
    variables_lib.global_variables_initializer().run()
    delta_tuple_state = _timer(sess, ops)
  print("%d \t %d \t %d \t %s \t %f \t\t %f \t\t %f" %
        (batch_size, max_time, num_units, use_gpu, delta_concat_state,
         delta_tuple_state, delta_concat_state / delta_tuple_state))

  return delta_concat_state, delta_tuple_state


def _dynamic_rnn_swap_memory_benchmark(inputs_t, sequence_length, swap_memory):
  # [FUNCTION DOC]
  # HELPER: builds the dynamic-RNN graph for the memory-swap benchmarks.
  # Inputs: a batch-by-time-by-depth input tensor, sequence lengths, and a flag passed to the
  # dynamic RNN to allow swapping tensors to host memory. Same peephole LSTM setup as the other
  # helpers; returns a grouped op over final state, outputs and gradients.
  (unused_0, unused_1, input_size) = inputs_t.get_shape().as_list()
  initializer = init_ops.random_uniform_initializer(-0.01, 0.01, seed=127)
  cell = rnn_cell_impl.LSTMCell(
      num_units=input_size,
      use_peepholes=True,
      initializer=initializer,
      state_is_tuple=False)
  outputs, final_state = rnn.dynamic_rnn(
      cell,
      inputs_t,
      sequence_length=sequence_length,
      swap_memory=swap_memory,
      dtype=dtypes.float32)

  trainable_variables = ops_lib.get_collection(
      ops_lib.GraphKeys.TRAINABLE_VARIABLES)
  gradients = gradients_impl.gradients([outputs, final_state],
                                       trainable_variables)

  return control_flow_ops.group(final_state, outputs, *gradients)


def dynamic_rnn_swap_memory_benchmark(batch_size, max_time, num_units):
  # [FUNCTION DOC]
  # Benchmark helper comparing a dynamic LSTM with and without memory swapping.
  # Inputs: batch size, number of time steps, units. Seeds the global random generator, builds the
  # graph twice (swap off, swap on) in separate sessions, times each, prints a row including the
  # ratio. Returns a pair of average seconds per run: no swap, then swap.
  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  # Set up sequence lengths
  np.random.seed([127])
  sequence_length = np.random.randint(0, max_time, size=batch_size)
  inputs_list = [
      np.random.randn(batch_size, num_units).astype(np.float32)
      for _ in range(max_time)
  ]
  inputs = np.dstack(inputs_list).transpose([0, 2, 1])  # batch x time x depth

  # No memory swap
  with session.Session(config=config, graph=ops_lib.Graph()) as sess:
    inputs_t = variables_lib.Variable(inputs, trainable=False).value()
    ops = _dynamic_rnn_swap_memory_benchmark(
        inputs_t, sequence_length, swap_memory=False)
    variables_lib.global_variables_initializer().run()
    no_swap = _timer(sess, ops)

  # Memory swap
  with session.Session(config=config, graph=ops_lib.Graph()) as sess:
    inputs_t = variables_lib.Variable(inputs, trainable=False).value()
    ops = _dynamic_rnn_swap_memory_benchmark(
        inputs_t, sequence_length, swap_memory=True)
    variables_lib.global_variables_initializer().run()
    swap = _timer(sess, ops)

  print("%d \t %d \t %d \t %f \t %f \t %f" %
        (batch_size, max_time, num_units, no_swap, swap, swap / no_swap))
  return no_swap, swap


def rnn_long_sequence_benchmark(batch_size, seqlen, num_units, dynamic,
                                swap_memory, nn):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the static branch ignores the swap flag, and the sequence lengths are a plain
  # list rather than an array; results are only printed, nothing is returned.
  # Benchmark helper for long sequences. Inputs: batch size, sequence length, units, a flag
  # choosing dynamic (true) or static RNN, the memory-swap flag (dynamic only), and the number of
  # repetitions. Seeds the global random generator, then repeatedly builds the chosen graph in a
  # fresh session, times it, and prints a row with elapsed time and time per step.
  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  # Set up sequence lengths
  np.random.seed([127])
  sequence_length = [seqlen for _ in range(batch_size)]
  inputs_list = [
      np.random.randn(batch_size, num_units).astype(np.float32)
      for _ in range(seqlen)
  ]
  inputs = np.dstack(inputs_list).transpose([0, 2, 1])  # batch x time x depth

  for _ in range(nn):
    if dynamic:
      with session.Session(config=config, graph=ops_lib.Graph()) as sess:
        inputs_t = variables_lib.Variable(inputs, trainable=False).value()
        ops = _dynamic_rnn_swap_memory_benchmark(
            inputs_t, sequence_length, swap_memory=swap_memory)
        variables_lib.global_variables_initializer().run()
        elapsed = _timer(sess, ops)
    else:
      with session.Session(config=config, graph=ops_lib.Graph()) as sess:
        inputs_list_t = [
            variables_lib.Variable(
                x, trainable=False).value() for x in inputs_list
        ]
        ops = _static_vs_dynamic_rnn_benchmark_static(inputs_list_t,
                                                      sequence_length)
        variables_lib.global_variables_initializer().run()
        elapsed = _timer(sess, ops)

    print("%d \t %d \t %d \t %s \t %f \t %f" % (batch_size, seqlen, num_units,
                                                dynamic, elapsed,
                                                elapsed / seqlen))


class BenchmarkRNN(test.Benchmark):
  # [CLASS DOC]
  # Benchmark class that drives the module-level benchmark helpers over parameter grids and
  # reports wall times to the benchmark framework: graph creation time, static versus dynamic
  # run time, memory swapping, half sequence length versus half unroll, and concatenated versus
  # tuple LSTM state. Stateless.

  def benchmarkGraphCreationStaticVsDynamicLSTM(self):
    # [FUNCTION DOC]
    # Benchmark: for 1, 25 and 50 time steps, measures graph-creation time for static and dynamic
    # RNNs, prints a table, and reports two named results (static and dynamic) per step count.
    print("Graph Creation: Static Unroll vs. Dynamic Unroll LSTM")
    print("max_t \t dt(static) \t dt(dynamic) \t dt(dynamic)/dt(static)")
    for max_time in (1, 25, 50):
      s_dt, d_dt = graph_creation_static_vs_dynamic_rnn_benchmark(max_time)
      self.report_benchmark(
          name="graph_creation_time_static_T%02d" % max_time,
          iters=5,
          wall_time=s_dt)
      self.report_benchmark(
          name="graph_creation_time_dynamic_T%02d" % max_time,
          iters=5,
          wall_time=d_dt)

  def benchmarkStaticUnrollVsDynamicFlowLSTM(self):
    # [FUNCTION DOC]
    # Benchmark: batch 256, 50 steps, units 512/256/128, with and without GPU placement; compares
    # run time of static unrolling with dynamic RNN and reports two named results per setting.
    print("Calculation: Static Unroll with Dynamic Flow LSTM "
          "vs. Dynamic Unroll LSTM")
    print("batch \t max_t \t units \t gpu \t dt(static) \t dt(dynamic) "
          "\t dt(dynamic)/dt(static)")
    for batch_size in (256,):
      for max_time in (50,):
        for num_units in (512, 256, 128):
          for use_gpu in (False, True):
            s_dt, d_dt = static_vs_dynamic_rnn_benchmark(batch_size, max_time,
                                                         num_units, use_gpu)
            self.report_benchmark(
                name="static_unroll_time_T%02d_B%03d_N%03d_gpu_%s" %
                (max_time, batch_size, num_units, use_gpu),
                iters=20,
                wall_time=s_dt)
            self.report_benchmark(
                name="dynamic_unroll_time_T%02d_B%03d_N%03d_gpu_%s" %
                (max_time, batch_size, num_units, use_gpu),
                iters=20,
                wall_time=d_dt)

  def benchmarkDynamicLSTMNoMemorySwapVsMemorySwap(self):
    # [FUNCTION DOC]
    # Benchmark: batch 256 and 512, 100 steps, units 512/256/128; compares dynamic LSTM run time
    # without and with memory swapping and reports two named results per setting.
    print("Calculation: Dynamic LSTM No Memory Swap vs. Memory Swap")
    print("batch \t max_t \t units \t no_swap \t swap \t swap/no_swap")
    for batch_size in (256, 512):
      for max_time in (100,):
        for num_units in (512, 256, 128):
          no_swap, swap = dynamic_rnn_swap_memory_benchmark(batch_size,
                                                            max_time, num_units)
          self.report_benchmark(
              name="dynamic_lstm_no_memory_swap_T%02d_B%03d_N%03d" %
              (max_time, batch_size, num_units),
              iters=20,
              wall_time=no_swap)
          self.report_benchmark(
              name="dynamic_lstm_with_memory_swap_T%02d_B%03d_N%03d" %
              (max_time, batch_size, num_units),
              iters=20,
              wall_time=swap)

  def benchmarkStaticUnrollHalfSequenceLengthVsHalfUnroll(self):
    # [FUNCTION DOC]
    # Benchmark: batch 128, 50 steps, 256 units, with and without GPU placement; compares halved
    # sequence length on a full unroll against a half-length unroll and reports two named results
    # per setting.
    print("Calculation: Static Unroll with Halved Sequence Length "
          "vs. Half Static Unroll")
    print("batch \t full_t \t units \t gpu \t dt(half_seq_len) "
          "\t dt(unroll_half) \t dt(half_seq_len)/dt(unroll_half)")
    for batch_size in (128,):
      for max_time in (50,):
        for num_units in (256,):
          for use_gpu in (False, True):
            s_dt, d_dt = half_seq_len_vs_unroll_half_rnn_benchmark(batch_size,
                                                                   max_time,
                                                                   num_units,
                                                                   use_gpu)
            self.report_benchmark(
                name="half_seq_len_time_T%02d_B%03d_N%03d_gpu_%s" %
                (max_time, batch_size, num_units, use_gpu),
                iters=20,
                wall_time=s_dt)
            self.report_benchmark(
                name="unroll_half_time_T%02d_B%03d_N%03d_gpu_%s" %
                (max_time, batch_size, num_units, use_gpu),
                iters=20,
                wall_time=d_dt)

  def benchmarkStaticUnrollStateConcatVsStateTuple(self):
    # [FUNCTION DOC]
    # Benchmark: batch 16 and 128, 50 steps, units 16 and 128, with and without GPU placement;
    # compares static unrolled LSTM with concatenated state versus tuple state and reports two
    # named results per setting.
    print("Calculation: Static Unroll with Concatenated State "
          "vs. Tuple State")
    print("batch \t time \t units \t gpu \t dt(concat_state) "
          "\t dt(tuple_state) \t dt(concat_state)/dt(tuple_state)")
    for batch_size in (
        16,
        128,):
      for max_time in (50,):
        for num_units in (
            16,
            128,):
          for use_gpu in (False, True):
            c_dt, t_dt = concat_state_vs_tuple_state_rnn_benchmark(batch_size,
                                                                   max_time,
                                                                   num_units,
                                                                   use_gpu)
            self.report_benchmark(
                name="concat_state_time_T%02d_B%03d_N%03d_gpu_%s" %
                (max_time, batch_size, num_units, use_gpu),
                iters=20,
                wall_time=c_dt)
            self.report_benchmark(
                name="tuple_state_time_T%02d_B%03d_N%03d_gpu_%s" %
                (max_time, batch_size, num_units, use_gpu),
                iters=20,
                wall_time=t_dt)

  def _benchmarkDynamicLSTMMemorySwapLongSeq(self):
    # [FUNCTION DOC]
    # HELPER: disabled benchmark (underscore prefix keeps it from being discovered) for long
    # sequences with memory swapping, originally for a research paper. Uses batch 512, 512 units,
    # dynamic RNN with swapping; does two warm-up repetitions at length 800, then three
    # repetitions for each length from 100 to 1000 in steps of 100. Prints results; reports none.
    """The memory swapping test for the SOSP submission."""
    print("Calculation: Long LSTM Sequence")
    print("batch \t len \t units \t dynamic \t elapsed_t \t elapsed_t/len")
    batch_size = 512
    seqlen = 800
    num_units = 512
    dynamic = True
    swap_memory = True
    # Some warming up.
    if swap_memory:
      rnn_long_sequence_benchmark(batch_size, seqlen, num_units,
                                  dynamic, swap_memory, 2)
    # Measure the performance.
    for slen in range(100, 1100, 100):
      rnn_long_sequence_benchmark(batch_size, slen, num_units, dynamic,
                                  swap_memory, 3)

if __name__ == "__main__":
  test.main()
