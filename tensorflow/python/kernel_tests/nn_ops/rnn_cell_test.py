# [MODULE DOC]
# Test module for the legacy (graph-oriented) RNN cell library and the static and dynamic RNN
# drivers. It defines small dummy cells (a plus-one cell, a multi-dimensional dummy LSTM, a nested
# tuple cell) and state-saver stand-ins used as fixtures, then three test cases: RNNTest (basic
# static RNN, dropout, sequence-length masking, variable scoping), LSTMTest (LSTM cell with/without
# projection, sharding, clipping, state savers, dynamic-versus-static equivalence, and error
# handling of the raw LSTM block kernels) and BidirectionalRNNTest (bidirectional static/dynamic RNN
# behavior, plus further cell tests later in the file). Many tests are restricted to graph mode.
# It exposes no public API; running it as a main program starts the TensorFlow test runner.
# No import-time side effects beyond imports.
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
"""Tests for RNN cells."""

import itertools
import os

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_rnn_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.trackable import autotrackable
from tensorflow.python.util import nest


class Plus1RNNCell(rnn_cell.RNNCell):
  # [CLASS DOC]
  # HELPER: dummy cell fixture for the RNN driver tests in this file.
  # Stateless toy recurrent cell. Each step returns the input plus one as output and the previous
  # state plus one as the new state, with both output size and state size fixed at five. It holds
  # no variables, so it is used to verify unrolling, masking and shapes in the RNN drivers.
  """RNN Cell generating (output, new_state) = (input + 1, state + 1)."""

  @property
  def output_size(self):
    # [FUNCTION DOC]
    # HELPER: property required by the cell interface for Plus1RNNCell.
    # Returns the constant integer 5 as the size of the cell output. Takes no inputs, no side effects.
    return 5

  @property
  def state_size(self):
    # [FUNCTION DOC]
    # HELPER: property required by the cell interface for Plus1RNNCell.
    # Returns the constant integer 5 as the size of the cell state. Takes no inputs, no side effects.
    return 5

  def __call__(self, input_, state, scope=None):
    # [FUNCTION DOC]
    # HELPER: single-step computation of the Plus1RNNCell fixture.
    # Inputs: input_ is the step input tensor, state is the previous state tensor, scope is an
    # ignored optional variable scope. Returns a pair of the input plus one (the output) and the
    # state plus one (the new state). Raises nothing and creates no variables.
    return (input_ + 1, state + 1)


class DummyMultiDimensionalLSTM(rnn_cell.RNNCell):
  # [CLASS DOC]
  # HELPER: dummy cell fixture for testing RNN drivers with multi-dimensional per-step data.
  # Toy cell whose output and its two state components (a hidden and a cell part) are tensors of an
  # arbitrary, caller-chosen shape without time and batch dimensions. Each step adds one to the
  # input and to both state parts. It stores the shape in private attributes for the size
  # properties and has no variables.
  """LSTM Cell generating (output, new_state) = (input + 1, state + 1).

  The input to this cell may have an arbitrary number of dimensions that follow
  the preceding 'Time' and 'Batch' dimensions.
  """

  def __init__(self, dims):
    # [FUNCTION DOC]
    # Constructor. Input dims must be a tuple of integers giving the per-step shape without time or
    # batch dimensions. Raises a type error if dims is not a tuple. Stores the dims and builds the
    # output size (one shape object) and state size (a pair of identical shape objects).
    """Initialize the Multi-dimensional LSTM cell.

    Args:
      dims: tuple that contains the dimensions of the output of the cell,
      without including 'Time' or 'Batch' dimensions.
    """
    if not isinstance(dims, tuple):
      raise TypeError("The dimensions passed to DummyMultiDimensionalLSTM "
                      "should be a tuple of ints.")
    self._dims = dims
    self._output_size = tensor_shape.TensorShape(self._dims)
    self._state_size = (tensor_shape.TensorShape(self._dims),
                        tensor_shape.TensorShape(self._dims))

  @property
  def output_size(self):
    # [FUNCTION DOC]
    # HELPER: property required by the cell interface.
    # Returns the shape object computed in the constructor as the output size of the cell.
    return self._output_size

  @property
  def state_size(self):
    # [FUNCTION DOC]
    # HELPER: property required by the cell interface.
    # Returns the pair of shape objects computed in the constructor as the two-part state size.
    return self._state_size

  def __call__(self, input_, state, scope=None):
    # [FUNCTION DOC]
    # HELPER: single-step computation of the dummy multi-dimensional LSTM fixture.
    # Inputs: input_ is the step input tensor, state is a pair of tensors (hidden, cell), scope is an
    # ignored optional variable scope. Returns the input plus one as output and a pair holding each
    # state part plus one as the new state. No variables are created, nothing is raised.
    h, c = state
    return (input_ + 1, (h + 1, c + 1))


class NestedRNNCell(rnn_cell.RNNCell):
  # [CLASS DOC]
  # HELPER: dummy cell fixture for testing RNN drivers with nested (tuple) inputs, outputs, states.
  # Toy stateless cell whose input and output are pairs of tensors (size five each) and whose state
  # is a pair of tensors (size six each). Each step adds one to every component.
  """RNN Cell generating (output, new_state) = (input + 1, state + 1).

  The input, output and state of this cell is a tuple of two tensors.
  """

  @property
  def output_size(self):
    # [FUNCTION DOC]
    # HELPER: property required by the cell interface.
    # Returns a pair of integers, both 5, describing the two output components.
    return (5, 5)

  @property
  def state_size(self):
    # [FUNCTION DOC]
    # HELPER: property required by the cell interface.
    # Returns a pair of integers, both 6, describing the two state components.
    return (6, 6)

  def __call__(self, input_, state, scope=None):
    # [FUNCTION DOC]
    # HELPER: single-step computation of the nested-tuple cell fixture.
    # Inputs: input_ is a pair of tensors, state is a pair of tensors (hidden, cell), scope is an
    # ignored optional variable scope. Returns a pair: the output pair (each input part plus one)
    # and the new state pair (each state part plus one). Raises a value error if either argument
    # is not a pair (unpacking fails). No variables are created.
    h, c = state
    x, y = input_
    return ((x + 1, y + 1), (h + 1, c + 1))


class TestStateSaver(object):
  # [CLASS DOC]
  # HELPER: fake state saver passed to the state-saving static RNN in tests.
  # Provides zero initial states per name via the state method and records saved final states in a
  # public mapping from state names to tensors via save_state. Holds the batch size and a state
  # size that is either a single size or a mapping from state names to sizes.

  def __init__(self, batch_size, state_size):
    # [FUNCTION DOC]
    # Constructor. batch_size is the integer batch dimension of generated states; state_size is an
    # integer, a tuple of integers, or a mapping from state name to such a value. Stores both and
    # creates an empty mapping of saved states. Nothing is raised here.
    self._batch_size = batch_size
    self._state_size = state_size
    self.saved_state = {}

  def state(self, name):
    # [FUNCTION DOC]
    # Returns an all-zero float tensor of shape batch size followed by the state size to use as the
    # initial value of the named state. name selects the entry when the state size is a mapping;
    # otherwise it is ignored. An integer size is treated as a one-element tuple. Raises a type
    # error if the selected size is neither an integer nor a tuple. Creates a graph op.

    if isinstance(self._state_size, dict):
      state_size = self._state_size[name]
    else:
      state_size = self._state_size
    if isinstance(state_size, int):
      state_size = (state_size,)
    elif isinstance(state_size, tuple):
      pass
    else:
      raise TypeError("state_size should either be an int or a tuple")

    return array_ops.zeros((self._batch_size,) + state_size)

  def save_state(self, name, state):
    # [FUNCTION DOC]
    # Records the given state tensor under the given name in the saved-state mapping (side effect,
    # overwriting any earlier entry) and returns an identity op of that tensor so callers can
    # depend on the save. Inputs: name (string key), state (tensor). Raises nothing.
    self.saved_state[name] = state
    return array_ops.identity(state)

  @property
  def batch_size(self):
    # [FUNCTION DOC]
    # Property returning the integer batch size given at construction.
    return self._batch_size

  @property
  def state_size(self):
    # [FUNCTION DOC]
    # Property returning the state size (integer, tuple, or name-to-size mapping) given at
    # construction.
    return self._state_size


class TestStateSaverWithCounters(TestStateSaver):
  # [CLASS DOC]
  # HELPER: instrumented state saver for the static state-saving RNN tests.
  # Extends TestStateSaver by counting, with two integer variables, how many times the state and
  # save_state methods are executed in the graph (each call adds a control dependency that
  # increments the matching counter). Counter variables are exposed through read-only properties.
  """Class wrapper around TestStateSaver.

  A dummy class used for testing of static_state_saving_rnn. It helps test if
  save_state and state functions got called same number of time when we
  evaluate output of rnn cell and state or either of them separately. It
  inherits from the TestStateSaver and adds the counters for calls of functions.
  """

  @test_util.run_v1_only("b/124229375")
  def __init__(self, batch_size, state_size):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the v1-only test decorator is applied to a constructor, which is unusual; it is
    # meant for test methods and wraps construction (skipping or altering it under v2 behavior).
    # Constructor. Calls the base constructor with batch_size and state_size, then creates two
    # integer counter variables (initially zero) for state calls and save_state calls. Side effect:
    # creates two graph variables that must be initialized before use.
    super(TestStateSaverWithCounters, self).__init__(batch_size, state_size)
    self._num_state_calls = variable_v1.VariableV1(0)
    self._num_save_state_calls = variable_v1.VariableV1(0)

  def state(self, name):
    # [FUNCTION DOC]
    # Same as the base state method (returns a zero tensor for the named state) but first
    # registers an increment of the state-call counter as a control dependency, so evaluating the
    # returned tensor bumps the counter. Input: name of the state. Returns the zero tensor.
    with ops.control_dependencies(
        [state_ops.assign_add(self._num_state_calls, 1)]):
      return super(TestStateSaverWithCounters, self).state(name)

  def save_state(self, name, state):
    # [FUNCTION DOC]
    # Same as the base save_state method (records the tensor under the name and returns an
    # identity op) but with a control dependency incrementing the save-call counter. Inputs: name
    # and state tensor. Returns the identity tensor.
    with ops.control_dependencies([state_ops.assign_add(
        self._num_save_state_calls, 1)]):
      return super(TestStateSaverWithCounters, self).save_state(name, state)

  @property
  def num_state_calls(self):
    # [FUNCTION DOC]
    # Property returning the counter variable tracking how many times state was evaluated.
    return self._num_state_calls

  @property
  def num_save_state_calls(self):
    # [FUNCTION DOC]
    # Property returning the counter variable tracking how many times save_state was evaluated.
    return self._num_save_state_calls


class RNNTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the basic static RNN driver using the plus-one dummy cell. Covers output and
  # state values, shape and dtype checks, dropout wrapping, sequence-length masking and variable
  # scope naming. Most tests run in graph mode only. Fixture: setUp seeds numpy's random generator.

  def setUp(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: does not call the parent setUp, so base-class test fixture setup is skipped.
    # Fixture. Stores a fixed integer seed on the instance and seeds the numpy random generator
    # with it so random input values are reproducible. Returns nothing.
    self._seed = 23489
    np.random.seed(self._seed)

  @test_util.run_v1_only("b/124229375")
  def testInvalidSequenceLengthShape(self):
    # [FUNCTION DOC]
    # Scenario: build the static RNN over one placeholder input but pass a scalar integer as the
    # sequence length. Expectation: a value error whose message says the length must be a vector.
    cell = Plus1RNNCell()
    inputs = [array_ops.placeholder(dtypes.float32, shape=(3, 4))]
    with self.assertRaisesRegex(ValueError, "must be a vector"):
      rnn.static_rnn(cell, inputs, dtype=dtypes.float32, sequence_length=4)

  @test_util.run_v1_only("b/124229375")
  def testRNN(self):
    # [FUNCTION DOC]
    # Scenario: unroll the plus-one cell over eight placeholder inputs of batch 2 and width 5 with
    # the static RNN. Checks the number, shape and dtype of outputs. Runs a session with random
    # input and asserts every output equals the input plus one and the final state equals the
    # number of steps in every element.
    cell = Plus1RNNCell()
    batch_size = 2
    input_size = 5
    max_length = 8  # unrolled up to this length
    inputs = max_length * [
        array_ops.placeholder(dtypes.float32, shape=(batch_size, input_size))
    ]
    outputs, state = rnn.static_rnn(cell, inputs, dtype=dtypes.float32)
    self.assertEqual(len(outputs), len(inputs))
    for out, inp in zip(outputs, inputs):
      self.assertEqual(out.get_shape(), inp.get_shape())
      self.assertEqual(out.dtype, inp.dtype)

    with self.session() as sess:
      input_value = np.random.randn(batch_size, input_size)
      values = sess.run(outputs + [state], feed_dict={inputs[0]: input_value})

      # Outputs
      for v in values[:-1]:
        self.assertAllClose(v, input_value + 1.0)

      # Final state
      self.assertAllClose(values[-1],
                          max_length * np.ones(
                              (batch_size, input_size), dtype=np.float32))

  @test_util.run_v1_only("b/124229375")
  def testDropout(self):
    # [FUNCTION DOC]
    # Scenario: wrap the plus-one cell in a dropout wrapper with an almost-zero input keep
    # probability and a fixed seed. Asserts the wrapper tracks the wrapped cell as a child, runs the
    # plain and dropped RNNs in separate scopes, and checks the plain outputs equal input plus one
    # while the dropped outputs (input all zeroed) equal one everywhere.
    cell = Plus1RNNCell()
    full_dropout_cell = rnn_cell.DropoutWrapper(
        cell, input_keep_prob=1e-6, seed=0)
    self.assertIn("cell", full_dropout_cell._trackable_children())
    self.assertIs(full_dropout_cell._trackable_children()["cell"], cell)
    batch_size = 2
    input_size = 5
    max_length = 8
    inputs = max_length * [
        array_ops.placeholder(dtypes.float32, shape=(batch_size, input_size))
    ]
    with variable_scope.variable_scope("share_scope"):
      outputs, state = rnn.static_rnn(cell, inputs, dtype=dtypes.float32)
    with variable_scope.variable_scope("drop_scope"):
      dropped_outputs, _ = rnn.static_rnn(
          full_dropout_cell, inputs, dtype=dtypes.float32)
    self.assertEqual(len(outputs), len(inputs))
    for out, inp in zip(outputs, inputs):
      self.assertEqual(out.get_shape().as_list(), inp.get_shape().as_list())
      self.assertEqual(out.dtype, inp.dtype)

    with self.session() as sess:
      input_value = np.random.randn(batch_size, input_size)
      values = sess.run(outputs + [state], feed_dict={inputs[0]: input_value})
      full_dropout_values = sess.run(
          dropped_outputs, feed_dict={
              inputs[0]: input_value
          })

      for v in values[:-1]:
        self.assertAllClose(v, input_value + 1.0)
      for d_v in full_dropout_values[:-1]:  # Add 1.0 to dropped_out (all zeros)
        self.assertAllClose(d_v, np.ones_like(input_value))

  @test_util.run_v1_only("b/124229375")
  def testDynamicCalculation(self):
    # [FUNCTION DOC]
    # Scenario: static RNN with the plus-one cell and a placeholder of per-example sequence
    # lengths, fed lengths 2 and 3. Asserts outputs are computed for the first two steps, at step
    # three only for the longer example (zero for the other), zero afterwards, and that the final
    # states are copied from the last valid step of each example.
    cell = Plus1RNNCell()
    sequence_length = array_ops.placeholder(dtypes.int64)
    batch_size = 2
    input_size = 5
    max_length = 8
    inputs = max_length * [
        array_ops.placeholder(dtypes.float32, shape=(batch_size, input_size))
    ]
    with variable_scope.variable_scope("drop_scope"):
      dynamic_outputs, dynamic_state = rnn.static_rnn(
          cell, inputs, sequence_length=sequence_length, dtype=dtypes.float32)
    self.assertEqual(len(dynamic_outputs), len(inputs))

    with self.session() as sess:
      input_value = np.random.randn(batch_size, input_size)
      dynamic_values = sess.run(
          dynamic_outputs,
          feed_dict={
              inputs[0]: input_value,
              sequence_length: [2, 3]
          })
      dynamic_state_value = sess.run(
          [dynamic_state],
          feed_dict={
              inputs[0]: input_value,
              sequence_length: [2, 3]
          })

      # outputs are fully calculated for t = 0, 1
      for v in dynamic_values[:2]:
        self.assertAllClose(v, input_value + 1.0)

      # outputs at t = 2 are zero for entry 0, calculated for entry 1
      self.assertAllClose(dynamic_values[2],
                          np.vstack((np.zeros((input_size)),
                                     1.0 + input_value[1, :])))

      # outputs at t = 3+ are zero
      for v in dynamic_values[3:]:
        self.assertAllEqual(v, np.zeros_like(input_value))

      # the final states are:
      #  entry 0: the values from the calculation at t=1
      #  entry 1: the values from the calculation at t=2
      self.assertAllEqual(dynamic_state_value[0],
                          np.vstack((1.0 * (1 + 1) * np.ones((input_size)),
                                     1.0 * (2 + 1) * np.ones((input_size)))))

  def _testScope(self, factory, prefix="prefix", use_outer_scope=True):
    # [FUNCTION DOC]
    # HELPER: shared body for the variable-scope naming test.
    # KNOWN ISSUE: the call that creates the global variables initializer op is made but its
    # result is never run, so it has no effect on the check.
    # Inputs: factory is a callback that builds an RNN given a scope or scope name; prefix is the
    # expected scope name (a nothing value means the default rnn scope); use_outer_scope chooses
    # whether to pass a variable scope object (true) or just the name string (false). Builds the RNN
    # in a fresh graph, logs the variable names, and asserts every global variable name starts with
    # the prefix followed by a slash. Returns nothing.
    with self.session(graph=ops.Graph()):
      if use_outer_scope:
        with variable_scope.variable_scope(prefix) as scope:
          factory(scope)
      else:
        factory(prefix)

      # check that all the variables names starts
      # with the proper scope.
      variables_lib.global_variables_initializer()
      all_vars = variables_lib.global_variables()
      prefix = prefix or "rnn"
      scope_vars = [v for v in all_vars if v.name.startswith(prefix + "/")]
      tf_logging.info("RNN with scope: %s (%s)" %
                      (prefix, "scope" if use_outer_scope else "str"))
      for v in scope_vars:
        tf_logging.info(v.name)
      self.assertEqual(len(scope_vars), len(all_vars))

  @test_util.run_v1_only("b/124229375")
  def testScope(self):
    # [FUNCTION DOC]
    # Scenario: build a static RNN with the plus-one cell via a nested factory and check variable
    # scoping three ways: with an outer variable scope object, with a scope name string, and with
    # no scope name (default rnn scope). Delegates assertions to the scope-checking helper.

    def factory(scope):
      # [FUNCTION DOC]
      # HELPER: builds the RNN under test for the scope helper.
      # Input: scope, a variable scope object or a name string. Creates eight placeholder inputs
      # (batch 2, width 5), runs the static RNN with the plus-one cell in that scope, and returns
      # its (outputs, final state) pair.
      cell = Plus1RNNCell()
      batch_size = 2
      input_size = 5
      max_length = 8  # unrolled up to this length
      inputs = max_length * [
          array_ops.placeholder(dtypes.float32, shape=(batch_size, input_size))
      ]
      return rnn.static_rnn(cell, inputs, dtype=dtypes.float32, scope=scope)

    self._testScope(factory, use_outer_scope=True)
    self._testScope(factory, use_outer_scope=False)
    self._testScope(factory, prefix=None, use_outer_scope=False)


class LSTMTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the LSTM cell and its interaction with the RNN drivers: dtype handling, plain,
  # clipped, peephole, projected and sharded variants, state savers, weight sharing and scoping,
  # dynamic versus static equivalence, and invalid-argument handling of the raw LSTM block ops.
  # Fixture: setUp seeds numpy and stores a seed used by weight initializers.

  def setUp(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: does not call the parent setUp, so base-class test fixture setup is skipped.
    # Fixture. Stores a fixed integer seed on the instance and seeds numpy's random generator.
    self._seed = 23489
    np.random.seed(self._seed)

  def testDType(self):
    # [FUNCTION DOC]
    # Scenario (regression for a reported dtype issue): build LSTM cells and call build on a ten by
    # fifty shape. Asserts the bias variable is float32 when no dtype is given, and equals the
    # requested dtype for float16, float32 and float64.
    # Test case for GitHub issue 16228
    # Not passing dtype in constructor results in default float32
    lstm = rnn_cell.LSTMCell(10)
    input_tensor = array_ops.ones([10, 50])
    lstm.build(input_tensor.get_shape())
    self.assertEqual(lstm._bias.dtype.base_dtype, dtypes.float32)

    # Explicitly pass dtype in constructor
    for dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
      lstm = rnn_cell.LSTMCell(10, dtype=dtype)
      input_tensor = array_ops.ones([10, 50])
      lstm.build(input_tensor.get_shape())
      self.assertEqual(lstm._bias.dtype.base_dtype, dtype)

  @test_util.run_v1_only("b/124229375")
  def testNoProjNoSharding(self):
    # [FUNCTION DOC]
    # Smoke scenario: static RNN of eight steps with a plain 3-unit LSTM cell (no projection, no
    # sharding, non-tuple state). Asserts each output has static shape batch by units and that the
    # graph runs with random input after variable initialization.
    num_units = 3
    input_size = 5
    batch_size = 2
    max_length = 8
    with self.session(graph=ops.Graph()) as sess:
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=self._seed)
      cell = rnn_cell.LSTMCell(
          num_units, initializer=initializer, state_is_tuple=False)
      inputs = max_length * [
          array_ops.placeholder(dtypes.float32, shape=(batch_size, input_size))
      ]
      outputs, _ = rnn.static_rnn(cell, inputs, dtype=dtypes.float32)
      self.assertEqual(len(outputs), len(inputs))
      for out in outputs:
        self.assertEqual(out.get_shape().as_list(), [batch_size, num_units])

      variables_lib.global_variables_initializer().run()
      input_value = np.random.randn(batch_size, input_size)
      sess.run(outputs, feed_dict={inputs[0]: input_value})

  @test_util.run_v1_only("b/124229375")
  def testCellClipping(self):
    # [FUNCTION DOC]
    # Scenario: peephole LSTM with the cell clip value set to zero in a static RNN. Because the
    # cell state is clipped to zero, asserts every output across all steps is exactly zero, and
    # that output shapes are batch by units.
    num_units = 3
    input_size = 5
    batch_size = 2
    max_length = 8
    with self.session(graph=ops.Graph()) as sess:
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=self._seed)
      cell = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          cell_clip=0.0,
          initializer=initializer,
          state_is_tuple=False)
      inputs = max_length * [
          array_ops.placeholder(dtypes.float32, shape=(batch_size, input_size))
      ]
      outputs, _ = rnn.static_rnn(cell, inputs, dtype=dtypes.float32)
      self.assertEqual(len(outputs), len(inputs))
      for out in outputs:
        self.assertEqual(out.get_shape().as_list(), [batch_size, num_units])

      variables_lib.global_variables_initializer().run()
      input_value = np.random.randn(batch_size, input_size)
      values = sess.run(outputs, feed_dict={inputs[0]: input_value})

    for value in values:
      # if cell c is clipped to 0, tanh(c) = 0 => m==0
      self.assertAllEqual(value, np.zeros((batch_size, num_units)))

  @test_util.run_v1_only("b/124229375")
  def testNoProjNoShardingSimpleStateSaver(self):
    # [FUNCTION DOC]
    # Scenario: state-saving static RNN with a non-tuple-state LSTM and the fake state saver whose
    # state size is twice the units. Asserts output shapes, then evaluates the final state and
    # the saved state under the given name and checks they are equal.
    num_units = 3
    input_size = 5
    batch_size = 2
    max_length = 8
    with self.session(graph=ops.Graph()) as sess:
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=self._seed)
      state_saver = TestStateSaver(batch_size, 2 * num_units)
      cell = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=False,
          initializer=initializer,
          state_is_tuple=False)
      inputs = max_length * [
          array_ops.placeholder(dtypes.float32, shape=(batch_size, input_size))
      ]
      with variable_scope.variable_scope("share_scope"):
        outputs, state = rnn.static_state_saving_rnn(
            cell, inputs, state_saver=state_saver, state_name="save_lstm")
      self.assertEqual(len(outputs), len(inputs))
      for out in outputs:
        self.assertEqual(out.get_shape().as_list(), [batch_size, num_units])

      variables_lib.global_variables_initializer().run()
      input_value = np.random.randn(batch_size, input_size)
      (last_state_value, saved_state_value) = sess.run(
          [state, state_saver.saved_state["save_lstm"]],
          feed_dict={
              inputs[0]: input_value
          })
      self.assertAllEqual(last_state_value, saved_state_value)

  @test_util.run_v1_only("b/124229375")
  def testNoProjNoShardingTupleStateSaver(self):
    # [FUNCTION DOC]
    # Scenario: state-saving static RNN with a tuple-state LSTM and two state names (cell and
    # hidden). Asserts output shapes and that the evaluated final state tuple equals the two
    # saved states recorded by the fake state saver.
    num_units = 3
    input_size = 5
    batch_size = 2
    max_length = 8
    with self.session(graph=ops.Graph()) as sess:
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=self._seed)
      state_saver = TestStateSaver(batch_size, num_units)
      cell = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=False,
          initializer=initializer,
          state_is_tuple=True)
      inputs = max_length * [
          array_ops.placeholder(dtypes.float32, shape=(batch_size, input_size))
      ]
      with variable_scope.variable_scope("share_scope"):
        outputs, state = rnn.static_state_saving_rnn(
            cell, inputs, state_saver=state_saver, state_name=("c", "m"))
      self.assertEqual(len(outputs), len(inputs))
      for out in outputs:
        self.assertEqual(out.get_shape().as_list(), [batch_size, num_units])

      variables_lib.global_variables_initializer().run()
      input_value = np.random.randn(batch_size, input_size)
      last_and_saved_states = sess.run(
          state + (state_saver.saved_state["c"], state_saver.saved_state["m"]),
          feed_dict={
              inputs[0]: input_value
          })
      self.assertEqual(4, len(last_and_saved_states))
      self.assertAllEqual(last_and_saved_states[:2], last_and_saved_states[2:])

  @test_util.run_v1_only("b/124229375")
  def testNoProjNoShardingNestedTupleStateSaver(self):
    # [FUNCTION DOC]
    # Scenario: a stack of four tuple-state LSTM cells of growing size run through the state-saving
    # static RNN with a fake saver keyed by eight state names. Asserts the nested state structure,
    # that outputs have the last cell's width, and that all eight final state tensors equal the
    # corresponding saved states.
    num_units = 3
    input_size = 5
    batch_size = 2
    max_length = 8
    with self.session(graph=ops.Graph()) as sess:
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=self._seed)
      state_saver = TestStateSaver(
          batch_size, {
              "c0": num_units,
              "m0": num_units,
              "c1": num_units + 1,
              "m1": num_units + 1,
              "c2": num_units + 2,
              "m2": num_units + 2,
              "c3": num_units + 3,
              "m3": num_units + 3
          })

      def _cell(i):
        # [FUNCTION DOC]
        # HELPER: builds one tuple-state LSTM cell for the stacked-cell scenario.
        # Input: i, integer offset added to the unit count. Returns an LSTM cell with units plus i,
        # no peepholes and the shared initializer.
        return rnn_cell.LSTMCell(
            num_units + i,
            use_peepholes=False,
            initializer=initializer,
            state_is_tuple=True)

      # This creates a state tuple which has 4 sub-tuples of length 2 each.
      cell = rnn_cell.MultiRNNCell(
          [_cell(i) for i in range(4)], state_is_tuple=True)

      self.assertEqual(len(cell.state_size), 4)
      for i in range(4):
        self.assertEqual(len(cell.state_size[i]), 2)

      inputs = max_length * [
          array_ops.placeholder(dtypes.float32, shape=(batch_size, input_size))
      ]

      state_names = (("c0", "m0"), ("c1", "m1"), ("c2", "m2"), ("c3", "m3"))
      with variable_scope.variable_scope("share_scope"):
        outputs, state = rnn.static_state_saving_rnn(
            cell, inputs, state_saver=state_saver, state_name=state_names)
      self.assertEqual(len(outputs), len(inputs))

      # Final output comes from _cell(3) which has state size num_units + 3
      for out in outputs:
        self.assertEqual(out.get_shape().as_list(), [batch_size, num_units + 3])

      variables_lib.global_variables_initializer().run()
      input_value = np.random.randn(batch_size, input_size)
      last_states = sess.run(
          list(nest.flatten(state)), feed_dict={
              inputs[0]: input_value
          })
      saved_states = sess.run(
          list(state_saver.saved_state.values()),
          feed_dict={
              inputs[0]: input_value
          })
      self.assertEqual(8, len(last_states))
      self.assertEqual(8, len(saved_states))
      flat_state_names = nest.flatten(state_names)
      named_saved_states = dict(
          zip(state_saver.saved_state.keys(), saved_states))

      for i in range(8):
        self.assertAllEqual(last_states[i],
                            named_saved_states[flat_state_names[i]])

  @test_util.run_v1_only("b/124229375")
  def testProjNoSharding(self):
    # [FUNCTION DOC]
    # Smoke scenario: peephole LSTM with a projection layer of four units and no sharding in a
    # static RNN with unknown batch dimension. Asserts the output count and that the graph runs.
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    max_length = 8
    with self.session(graph=ops.Graph()) as sess:
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=self._seed)
      inputs = max_length * [
          array_ops.placeholder(dtypes.float32, shape=(None, input_size))
      ]
      cell = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          num_proj=num_proj,
          initializer=initializer,
          state_is_tuple=False)
      outputs, _ = rnn.static_rnn(cell, inputs, dtype=dtypes.float32)
      self.assertEqual(len(outputs), len(inputs))

      variables_lib.global_variables_initializer().run()
      input_value = np.random.randn(batch_size, input_size)
      sess.run(outputs, feed_dict={inputs[0]: input_value})

  def _testStateTupleWithProjAndSequenceLength(self):
    # [FUNCTION DOC]
    # HELPER: shared scenario body, apparently not invoked by any test method here.
    # KNOWN ISSUE: the name starts with an underscore and no test calls it, so it is never run.
    # Scenario: two peephole projected LSTM cells, one with tuple state and one without, share
    # weights by reusing the scope, run over eight steps with sequence lengths 4 and 6. Asserts
    # state types (tuple versus tensor), identical outputs, and that the non-tuple state equals the
    # horizontal concatenation of the tuple state parts.
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    max_length = 8
    sequence_length = [4, 6]
    with self.session(graph=ops.Graph()) as sess:
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=self._seed)
      inputs = max_length * [
          array_ops.placeholder(dtypes.float32, shape=(None, input_size))
      ]
      cell_notuple = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          num_proj=num_proj,
          initializer=initializer,
          state_is_tuple=False)
      cell_tuple = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          num_proj=num_proj,
          initializer=initializer,
          state_is_tuple=True)
      with variable_scope.variable_scope("root") as scope:
        outputs_notuple, state_notuple = rnn.static_rnn(
            cell_notuple,
            inputs,
            dtype=dtypes.float32,
            sequence_length=sequence_length,
            scope=scope)
        scope.reuse_variables()
        # TODO(ebrevdo): For this test, we ensure values are identical and
        # therefore the weights here are tied.  In the future, we may consider
        # making the state_is_tuple property mutable so we can avoid
        # having to do this - especially if users ever need to reuse
        # the parameters from different RNNCell instances.  Right now,
        # this seems an unrealistic use case except for testing.
        cell_tuple._scope = cell_notuple._scope  # pylint: disable=protected-access
        outputs_tuple, state_tuple = rnn.static_rnn(
            cell_tuple,
            inputs,
            dtype=dtypes.float32,
            sequence_length=sequence_length,
            scope=scope)
      self.assertEqual(len(outputs_notuple), len(inputs))
      self.assertEqual(len(outputs_tuple), len(inputs))
      self.assertTrue(isinstance(state_tuple, tuple))
      self.assertTrue(isinstance(state_notuple, tensor.Tensor))

      variables_lib.global_variables_initializer().run()
      input_value = np.random.randn(batch_size, input_size)
      outputs_notuple_v = sess.run(
          outputs_notuple, feed_dict={
              inputs[0]: input_value
          })
      outputs_tuple_v = sess.run(
          outputs_tuple, feed_dict={
              inputs[0]: input_value
          })
      self.assertAllEqual(outputs_notuple_v, outputs_tuple_v)

      (state_notuple_v,) = sess.run(
          (state_notuple,), feed_dict={
              inputs[0]: input_value
          })
      state_tuple_v = sess.run(state_tuple, feed_dict={inputs[0]: input_value})
      self.assertAllEqual(state_notuple_v, np.hstack(state_tuple_v))

  @test_util.run_v1_only("b/124229375")
  def testProjSharding(self):
    # [FUNCTION DOC]
    # Smoke scenario: peephole LSTM with projection and with weight sharding (two unit shards,
    # three projection shards) in a static RNN. Asserts the output count and that the graph runs.
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    num_proj_shards = 3
    num_unit_shards = 2
    max_length = 8
    with self.session(graph=ops.Graph()) as sess:
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=self._seed)

      inputs = max_length * [
          array_ops.placeholder(dtypes.float32, shape=(None, input_size))
      ]

      cell = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          num_proj=num_proj,
          num_unit_shards=num_unit_shards,
          num_proj_shards=num_proj_shards,
          initializer=initializer,
          state_is_tuple=False)

      outputs, _ = rnn.static_rnn(cell, inputs, dtype=dtypes.float32)

      self.assertEqual(len(outputs), len(inputs))

      variables_lib.global_variables_initializer().run()
      input_value = np.random.randn(batch_size, input_size)
      sess.run(outputs, feed_dict={inputs[0]: input_value})

  @test_util.run_v1_only("b/124229375")
  def testDoubleInput(self):
    # [FUNCTION DOC]
    # Scenario: sharded, projected, peephole LSTM fed float64 placeholders with an explicit float64
    # zero initial state. Asserts the first output value has dtype float64 like the input.
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    num_proj_shards = 3
    num_unit_shards = 2
    max_length = 8
    with self.session(graph=ops.Graph()) as sess:
      initializer = init_ops.random_uniform_initializer(-1, 1, seed=self._seed)
      inputs = max_length * [
          array_ops.placeholder(dtypes.float64, shape=(None, input_size))
      ]

      cell = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          num_proj=num_proj,
          num_unit_shards=num_unit_shards,
          num_proj_shards=num_proj_shards,
          initializer=initializer,
          state_is_tuple=False)

      outputs, _ = rnn.static_rnn(
          cell,
          inputs,
          initial_state=cell.zero_state(batch_size, dtypes.float64))

      self.assertEqual(len(outputs), len(inputs))

      variables_lib.global_variables_initializer().run()
      input_value = np.asarray(
          np.random.randn(batch_size, input_size), dtype=np.float64)
      values = sess.run(outputs, feed_dict={inputs[0]: input_value})
      self.assertEqual(values[0].dtype, input_value.dtype)

  @test_util.run_v1_only("b/124229375")
  def testShardNoShardEquivalentOutput(self):
    # [FUNCTION DOC]
    # Scenario: two LSTMs with the same constant initializer, one sharded and one not, run in
    # separate scopes on identical inputs. Asserts output and state lists have equal length and
    # values agree within a tolerance of one thousandth.
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    num_proj_shards = 3
    num_unit_shards = 2
    max_length = 8
    with self.session(graph=ops.Graph()) as sess:
      inputs = max_length * [
          array_ops.placeholder(dtypes.float32, shape=(None, input_size))
      ]
      initializer = init_ops.constant_initializer(0.001)

      cell_noshard = rnn_cell.LSTMCell(
          num_units,
          num_proj=num_proj,
          use_peepholes=True,
          initializer=initializer,
          num_unit_shards=num_unit_shards,
          num_proj_shards=num_proj_shards,
          state_is_tuple=False)

      cell_shard = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          initializer=initializer,
          num_proj=num_proj,
          state_is_tuple=False)

      with variable_scope.variable_scope("noshard_scope"):
        outputs_noshard, state_noshard = rnn.static_rnn(
            cell_noshard, inputs, dtype=dtypes.float32)
      with variable_scope.variable_scope("shard_scope"):
        outputs_shard, state_shard = rnn.static_rnn(
            cell_shard, inputs, dtype=dtypes.float32)

      self.assertEqual(len(outputs_noshard), len(inputs))
      self.assertEqual(len(outputs_noshard), len(outputs_shard))

      variables_lib.global_variables_initializer().run()
      input_value = np.random.randn(batch_size, input_size)
      feeds = dict((x, input_value) for x in inputs)
      values_noshard = sess.run(outputs_noshard, feed_dict=feeds)
      values_shard = sess.run(outputs_shard, feed_dict=feeds)
      state_values_noshard = sess.run([state_noshard], feed_dict=feeds)
      state_values_shard = sess.run([state_shard], feed_dict=feeds)
      self.assertEqual(len(values_noshard), len(values_shard))
      self.assertEqual(len(state_values_noshard), len(state_values_shard))
      for (v_noshard, v_shard) in zip(values_noshard, values_shard):
        self.assertAllClose(v_noshard, v_shard, atol=1e-3)
      for (s_noshard, s_shard) in zip(state_values_noshard, state_values_shard):
        self.assertAllClose(s_noshard, s_shard, atol=1e-3)

  @test_util.run_v1_only("b/124229375")
  def testDoubleInputWithDropoutAndDynamicCalculation(self):
    # [FUNCTION DOC]
    # Smoke scenario: float64 inputs, a dropout-wrapped sharded projected LSTM, a sequence-length
    # placeholder fed lengths 2 and 3 and a float64 initial state. Asserts the output and the final
    # state keep dtype float64.
    """Smoke test for using LSTM with doubles, dropout, dynamic calculation."""

    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    num_proj_shards = 3
    num_unit_shards = 2
    max_length = 8
    with self.session(graph=ops.Graph()) as sess:
      sequence_length = array_ops.placeholder(dtypes.int64)
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=self._seed)
      inputs = max_length * [
          array_ops.placeholder(dtypes.float64, shape=(None, input_size))
      ]

      cell = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          num_proj=num_proj,
          num_unit_shards=num_unit_shards,
          num_proj_shards=num_proj_shards,
          initializer=initializer,
          state_is_tuple=False)
      dropout_cell = rnn_cell.DropoutWrapper(cell, 0.5, seed=0)

      outputs, state = rnn.static_rnn(
          dropout_cell,
          inputs,
          sequence_length=sequence_length,
          initial_state=cell.zero_state(batch_size, dtypes.float64))

      self.assertEqual(len(outputs), len(inputs))

      variables_lib.global_variables_initializer().run(feed_dict={
          sequence_length: [2, 3]
      })
      input_value = np.asarray(
          np.random.randn(batch_size, input_size), dtype=np.float64)
      values = sess.run(
          outputs, feed_dict={
              inputs[0]: input_value,
              sequence_length: [2, 3]
          })
      state_value = sess.run(
          [state], feed_dict={
              inputs[0]: input_value,
              sequence_length: [2, 3]
          })
      self.assertEqual(values[0].dtype, input_value.dtype)
      self.assertEqual(state_value[0].dtype, input_value.dtype)

  @test_util.run_v1_only("b/124229375")
  def testSharingWeightsWithReuse(self):
    # [FUNCTION DOC]
    # Scenario: run one LSTM twice, the second time in the same variable scope with reuse enabled,
    # and a different LSTM (other initializer) in another scope. Asserts the two runs of the same
    # cell give identical outputs while the other cell's outputs differ by a norm above a small
    # threshold.
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    max_length = 8
    with self.session(graph=ops.Graph()) as sess:
      initializer = init_ops.random_uniform_initializer(-1, 1, seed=self._seed)
      initializer_d = init_ops.random_uniform_initializer(
          -1, 1, seed=self._seed + 1)
      inputs = max_length * [
          array_ops.placeholder(dtypes.float32, shape=(None, input_size))
      ]
      cell = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          num_proj=num_proj,
          initializer=initializer,
          state_is_tuple=False)
      cell_d = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          num_proj=num_proj,
          initializer=initializer_d,
          state_is_tuple=False)

      with variable_scope.variable_scope("share_scope"):
        outputs0, _ = rnn.static_rnn(cell, inputs, dtype=dtypes.float32)
      with variable_scope.variable_scope("share_scope", reuse=True):
        outputs1, _ = rnn.static_rnn(cell, inputs, dtype=dtypes.float32)
      with variable_scope.variable_scope("diff_scope"):
        outputs2, _ = rnn.static_rnn(cell_d, inputs, dtype=dtypes.float32)

      variables_lib.global_variables_initializer().run()
      input_value = np.random.randn(batch_size, input_size)
      output_values = sess.run(
          outputs0 + outputs1 + outputs2, feed_dict={
              inputs[0]: input_value
          })
      outputs0_values = output_values[:max_length]
      outputs1_values = output_values[max_length:2 * max_length]
      outputs2_values = output_values[2 * max_length:]
      self.assertEqual(len(outputs0_values), len(outputs1_values))
      self.assertEqual(len(outputs0_values), len(outputs2_values))
      for o1, o2, o3 in zip(outputs0_values, outputs1_values, outputs2_values):
        # Same weights used by both RNNs so outputs should be the same.
        self.assertAllEqual(o1, o2)
        # Different weights used so outputs should be different.
        self.assertTrue(np.linalg.norm(o1 - o3) > 1e-6)

  @test_util.run_v1_only("b/124229375")
  def testSharingWeightsWithDifferentNamescope(self):
    # [FUNCTION DOC]
    # Scenario: reuse one LSTM's weights in the same variable scope but under two different name
    # scopes. Asserts the output values of both runs are exactly equal, showing variable sharing
    # does not depend on the name scope.
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    max_length = 8
    with self.session(graph=ops.Graph()) as sess:
      initializer = init_ops.random_uniform_initializer(-1, 1, seed=self._seed)
      inputs = max_length * [
          array_ops.placeholder(dtypes.float32, shape=(None, input_size))
      ]
      cell = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          num_proj=num_proj,
          initializer=initializer,
          state_is_tuple=False)

      with ops.name_scope("scope0"):
        with variable_scope.variable_scope("share_scope"):
          outputs0, _ = rnn.static_rnn(cell, inputs, dtype=dtypes.float32)
      with ops.name_scope("scope1"):
        with variable_scope.variable_scope("share_scope", reuse=True):
          outputs1, _ = rnn.static_rnn(cell, inputs, dtype=dtypes.float32)

      variables_lib.global_variables_initializer().run()
      input_value = np.random.randn(batch_size, input_size)
      output_values = sess.run(
          outputs0 + outputs1, feed_dict={
              inputs[0]: input_value
          })
      outputs0_values = output_values[:max_length]
      outputs1_values = output_values[max_length:]
      self.assertEqual(len(outputs0_values), len(outputs1_values))
      for out0, out1 in zip(outputs0_values, outputs1_values):
        self.assertAllEqual(out0, out1)

  @test_util.run_v1_only("b/124229375")
  def testDynamicRNNAllowsUnknownTimeDimension(self):
    # [FUNCTION DOC]
    # Smoke scenario: build the dynamic RNN with a GRU cell over a placeholder whose time
    # dimension is unknown. Expectation: graph construction does not raise; nothing is asserted.
    inputs = array_ops.placeholder(dtypes.float32, shape=[1, None, 20])
    cell = rnn_cell.GRUCell(30)
    # Smoke test, this should not raise an error
    rnn.dynamic_rnn(cell, inputs, dtype=dtypes.float32)

  @test_util.run_in_graph_and_eager_modes
  def testDynamicRNNWithTupleStates(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager modes): a tuple-state peephole projected LSTM run by both the
    # static RNN and the time-major dynamic RNN with sequence lengths 4 and 6, sharing weights.
    # Asserts both final states are the LSTM state tuple type with consistent indexing, and that
    # outputs and states match (exactly, or within tolerance when XLA is enabled).
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    max_length = 8
    sequence_length = [4, 6]
    in_graph_mode = not context.executing_eagerly()
    with self.session(graph=ops.Graph()) as sess:
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=self._seed)
      if in_graph_mode:
        inputs = max_length * [
            array_ops.placeholder(dtypes.float32, shape=(None, input_size))
        ]
      else:
        inputs = max_length * [
            constant_op.constant(
                np.random.randn(batch_size, input_size).astype(np.float32))
        ]
      inputs_c = array_ops_stack.stack(inputs)
      cell = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          num_proj=num_proj,
          initializer=initializer,
          state_is_tuple=True)
      with variable_scope.variable_scope("root") as scope:
        outputs_static, state_static = rnn.static_rnn(
            cell,
            inputs,
            dtype=dtypes.float32,
            sequence_length=sequence_length,
            scope=scope)
        scope.reuse_variables()
        outputs_dynamic, state_dynamic = rnn.dynamic_rnn(
            cell,
            inputs_c,
            dtype=dtypes.float32,
            time_major=True,
            sequence_length=sequence_length,
            scope=scope)
      self.assertTrue(isinstance(state_static, rnn_cell.LSTMStateTuple))
      self.assertTrue(isinstance(state_dynamic, rnn_cell.LSTMStateTuple))
      self.assertIs(state_static[0], state_static.c)
      self.assertIs(state_static[1], state_static.h)
      self.assertIs(state_dynamic[0], state_dynamic.c)
      self.assertIs(state_dynamic[1], state_dynamic.h)

      if in_graph_mode:
        variables_lib.global_variables_initializer().run()
        input_value = np.random.randn(batch_size, input_size)
        outputs_static = sess.run(
            outputs_static, feed_dict={
                inputs[0]: input_value
            })
        outputs_dynamic = sess.run(
            outputs_dynamic, feed_dict={
                inputs[0]: input_value
            })
        state_static = sess.run(
            state_static, feed_dict={
                inputs[0]: input_value
            })
        state_dynamic = sess.run(
            state_dynamic, feed_dict={
                inputs[0]: input_value
            })

      comparison_fn = self.assertAllEqual
      if test_util.is_xla_enabled():
        comparison_fn = self.assertAllClose
      if in_graph_mode:
        comparison_fn(outputs_static, outputs_dynamic)
      else:
        self.assertAllEqual(
            array_ops_stack.stack(outputs_static), outputs_dynamic)
      comparison_fn(np.hstack(state_static), np.hstack(state_dynamic))

  @test_util.run_in_graph_and_eager_modes
  def testDynamicRNNWithNestedTupleStates(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager modes): four stacked tuple-state LSTM cells compared between the
    # static and dynamic RNN with weight sharing and sequence lengths 4 and 6. Asserts the state
    # structure, zero-state shapes, and equality of flattened outputs and states between drivers.
    num_units = 3
    input_size = 5
    batch_size = 2
    num_proj = 4
    max_length = 8
    sequence_length = [4, 6]
    in_graph_mode = not context.executing_eagerly()
    with self.session(graph=ops.Graph()) as sess:
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=self._seed)
      if in_graph_mode:
        inputs = max_length * [
            array_ops.placeholder(dtypes.float32, shape=(None, input_size))
        ]
      else:
        inputs = max_length * [
            constant_op.constant(
                np.random.randn(batch_size, input_size).astype(np.float32))
        ]
      inputs_c = array_ops_stack.stack(inputs)

      def _cell(i):
        # [FUNCTION DOC]
        # HELPER: builds one peephole projected tuple-state LSTM cell for the stacked scenario.
        # Input: i, integer added to both the unit count and the projection size. Returns the cell.
        return rnn_cell.LSTMCell(
            num_units + i,
            use_peepholes=True,
            num_proj=num_proj + i,
            initializer=initializer,
            state_is_tuple=True)

      # This creates a state tuple which has 4 sub-tuples of length 2 each.
      cell = rnn_cell.MultiRNNCell(
          [_cell(i) for i in range(4)], state_is_tuple=True)

      self.assertEqual(len(cell.state_size), 4)
      for i in range(4):
        self.assertEqual(len(cell.state_size[i]), 2)

      test_zero = cell.zero_state(1, dtypes.float32)
      self.assertEqual(len(test_zero), 4)
      for i in range(4):
        self.assertEqual(test_zero[i][0].get_shape()[1], cell.state_size[i][0])
        self.assertEqual(test_zero[i][1].get_shape()[1], cell.state_size[i][1])

      with variable_scope.variable_scope("root") as scope:
        outputs_static, state_static = rnn.static_rnn(
            cell,
            inputs,
            dtype=dtypes.float32,
            sequence_length=sequence_length,
            scope=scope)
        scope.reuse_variables()
        outputs_dynamic, state_dynamic = rnn.dynamic_rnn(
            cell,
            inputs_c,
            dtype=dtypes.float32,
            time_major=True,
            sequence_length=sequence_length,
            scope=scope)

      if in_graph_mode:
        input_value = np.random.randn(batch_size, input_size)
        variables_lib.global_variables_initializer().run()
        outputs_static = sess.run(
            outputs_static, feed_dict={
                inputs[0]: input_value
            })
        outputs_dynamic = sess.run(
            outputs_dynamic, feed_dict={
                inputs[0]: input_value
            })
        state_static = sess.run(
            nest.flatten(state_static), feed_dict={
                inputs[0]: input_value
            })
        state_dynamic = sess.run(
            nest.flatten(state_dynamic), feed_dict={
                inputs[0]: input_value
            })

      comparison_fn = self.assertAllEqual
      if test_util.is_xla_enabled():
        comparison_fn = self.assertAllClose
      if in_graph_mode:
        self.assertAllClose(outputs_static, outputs_dynamic)
      else:
        self.assertAllEqual(
            array_ops_stack.stack(outputs_static), outputs_dynamic)
        state_static = nest.flatten(state_static)
        state_dynamic = nest.flatten(state_dynamic)
      comparison_fn(np.hstack(state_static), np.hstack(state_dynamic))

  def _testDynamicEquivalentToStaticRNN(self, use_sequence_length):
    # [FUNCTION DOC]
    # HELPER: shared body for the static versus dynamic RNN equivalence tests.
    # Input: use_sequence_length, boolean; when true random per-example lengths are used, else none.
    # Builds a peephole projected LSTM, runs it with the static RNN then the dynamic RNN on the same
    # random time-major input (in separate graphs; in eager mode one cell is reused because of an
    # operation-seed limitation). In graph mode also computes gradients of outputs, final state and
    # individual outputs with respect to inputs and trainable variables. Asserts outputs, final
    # states and all gradients are close between the two drivers. Returns nothing.
    time_steps = 8
    num_units = 3
    num_proj = 4
    input_size = 5
    batch_size = 2

    input_values = np.random.randn(time_steps, batch_size, input_size).astype(
        np.float32)

    if use_sequence_length:
      sequence_length = np.random.randint(0, time_steps, size=batch_size)
    else:
      sequence_length = None

    in_graph_mode = not context.executing_eagerly()

    # TODO(b/68017812): Eager ignores operation seeds, so we need to create a
    # single cell and reuse it across the static and dynamic RNNs. Remove this
    # special case once is fixed.
    if not in_graph_mode:
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=self._seed)
      cell = rnn_cell.LSTMCell(
          num_units,
          use_peepholes=True,
          initializer=initializer,
          num_proj=num_proj,
          state_is_tuple=False)

    ########### Step 1: Run static graph and generate readouts
    with self.session(graph=ops.Graph()) as sess:
      if in_graph_mode:
        concat_inputs = array_ops.placeholder(
            dtypes.float32, shape=(time_steps, batch_size, input_size))
      else:
        concat_inputs = constant_op.constant(input_values)
      inputs = array_ops_stack.unstack(concat_inputs)
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=self._seed)

      # TODO(akshayka): Remove special case once b/68017812 is fixed.
      if in_graph_mode:
        cell = rnn_cell.LSTMCell(
            num_units,
            use_peepholes=True,
            initializer=initializer,
            num_proj=num_proj,
            state_is_tuple=False)

      with variable_scope.variable_scope("dynamic_scope"):
        outputs_static, state_static = rnn.static_rnn(
            cell, inputs, sequence_length=sequence_length, dtype=dtypes.float32)

      if in_graph_mode:
        # Generate gradients of sum of outputs w.r.t. inputs
        static_gradients = gradients_impl.gradients(
            outputs_static + [state_static], [concat_inputs])
        # Generate gradients of individual outputs w.r.t. inputs
        static_individual_gradients = nest.flatten([
            gradients_impl.gradients(y, [concat_inputs])
            for y in [outputs_static[0], outputs_static[-1], state_static]
        ])
        # Generate gradients of individual variables w.r.t. inputs
        trainable_variables = ops.get_collection(
            ops.GraphKeys.TRAINABLE_VARIABLES)
        assert len(trainable_variables) > 1, (
            "Count of trainable variables: %d" % len(trainable_variables))
        # pylint: disable=bad-builtin
        static_individual_variable_gradients = nest.flatten([
            gradients_impl.gradients(y, trainable_variables)
            for y in [outputs_static[0], outputs_static[-1], state_static]
        ])
        # Generate gradients and run sessions to obtain outputs
        feeds = {concat_inputs: input_values}
        # Initialize
        variables_lib.global_variables_initializer().run(feed_dict=feeds)
        # Test forward pass
        values_static = sess.run(outputs_static, feed_dict=feeds)
        (state_value_static,) = sess.run((state_static,), feed_dict=feeds)

        # Test gradients to inputs and variables w.r.t. outputs & final state
        static_grad_values = sess.run(static_gradients, feed_dict=feeds)

        static_individual_grad_values = sess.run(
            static_individual_gradients, feed_dict=feeds)

        static_individual_var_grad_values = sess.run(
            static_individual_variable_gradients, feed_dict=feeds)

    ########## Step 2: Run dynamic graph and generate readouts
    with self.session(graph=ops.Graph()) as sess:
      if in_graph_mode:
        concat_inputs = array_ops.placeholder(
            dtypes.float32, shape=(time_steps, batch_size, input_size))
      else:
        concat_inputs = constant_op.constant(input_values)
      initializer = init_ops.random_uniform_initializer(
          -0.01, 0.01, seed=self._seed)

      # TODO(akshayka): Remove this special case once b/68017812 is
      # fixed.
      if in_graph_mode:
        cell = rnn_cell.LSTMCell(
            num_units,
            use_peepholes=True,
            initializer=initializer,
            num_proj=num_proj,
            state_is_tuple=False)

      with variable_scope.variable_scope("dynamic_scope"):
        outputs_dynamic, state_dynamic = rnn.dynamic_rnn(
            cell,
            inputs=concat_inputs,
            sequence_length=sequence_length,
            time_major=True,
            dtype=dtypes.float32)
        split_outputs_dynamic = array_ops_stack.unstack(
            outputs_dynamic, time_steps)

      if in_graph_mode:

        # Generate gradients of sum of outputs w.r.t. inputs
        dynamic_gradients = gradients_impl.gradients(
            split_outputs_dynamic + [state_dynamic], [concat_inputs])

        # Generate gradients of several individual outputs w.r.t. inputs
        dynamic_individual_gradients = nest.flatten([
            gradients_impl.gradients(y, [concat_inputs])
            for y in [
                split_outputs_dynamic[0], split_outputs_dynamic[-1],
                state_dynamic
            ]
        ])

        # Generate gradients of individual variables w.r.t. inputs
        trainable_variables = ops.get_collection(
            ops.GraphKeys.TRAINABLE_VARIABLES)
        assert len(trainable_variables) > 1, (
            "Count of trainable variables: %d" % len(trainable_variables))
        dynamic_individual_variable_gradients = nest.flatten([
            gradients_impl.gradients(y, trainable_variables)
            for y in [
                split_outputs_dynamic[0], split_outputs_dynamic[-1],
                state_dynamic
            ]
        ])

        feeds = {concat_inputs: input_values}

        # Initialize
        variables_lib.global_variables_initializer().run(feed_dict=feeds)

        # Test forward pass
        values_dynamic = sess.run(split_outputs_dynamic, feed_dict=feeds)
        (state_value_dynamic,) = sess.run((state_dynamic,), feed_dict=feeds)

        # Test gradients to inputs and variables w.r.t. outputs & final state
        dynamic_grad_values = sess.run(dynamic_gradients, feed_dict=feeds)

        dynamic_individual_grad_values = sess.run(
            dynamic_individual_gradients, feed_dict=feeds)

        dynamic_individual_var_grad_values = sess.run(
            dynamic_individual_variable_gradients, feed_dict=feeds)

    ######### Step 3: Comparisons
    if not in_graph_mode:
      values_static = outputs_static
      values_dynamic = split_outputs_dynamic
      state_value_static = state_static
      state_value_dynamic = state_dynamic

    self.assertEqual(len(values_static), len(values_dynamic))
    for (value_static, value_dynamic) in zip(values_static, values_dynamic):
      self.assertAllClose(value_static, value_dynamic)
    self.assertAllClose(state_value_static, state_value_dynamic)

    if in_graph_mode:

      self.assertAllClose(static_grad_values, dynamic_grad_values)

      self.assertEqual(
          len(static_individual_grad_values),
          len(dynamic_individual_grad_values))
      self.assertEqual(
          len(static_individual_var_grad_values),
          len(dynamic_individual_var_grad_values))

      for i, (a, b) in enumerate(
          zip(static_individual_grad_values, dynamic_individual_grad_values)):
        tf_logging.info("Comparing individual gradients iteration %d" % i)
        self.assertAllClose(a, b)

      for i, (a, b) in enumerate(
          zip(static_individual_var_grad_values,
              dynamic_individual_var_grad_values)):
        tf_logging.info(
            "Comparing individual variable gradients iteration %d" % i)
        self.assertAllClose(a, b)

  @test_util.run_in_graph_and_eager_modes
  def testDynamicEquivalentToStaticRNN(self):
    # [FUNCTION DOC]
    # Runs the equivalence helper without sequence lengths (graph and eager modes).
    self._testDynamicEquivalentToStaticRNN(use_sequence_length=False)

  @test_util.run_in_graph_and_eager_modes
  def testDynamicEquivalentToStaticRNNWithSequenceLength(self):
    # [FUNCTION DOC]
    # Runs the equivalence helper with random sequence lengths (graph and eager modes).
    self._testDynamicEquivalentToStaticRNN(use_sequence_length=True)

  @test_util.run_in_graph_and_eager_modes
  def testLSTMBlockCellErrorHandling(self):
    # [FUNCTION DOC]
    # Scenario: call the raw LSTM block cell kernel with constants whose peephole weight shapes
    # (a scalar and mismatched vectors) are inconsistent with the other inputs. Expectation: an
    # invalid-argument error is raised on evaluation.
    forget_bias = 1
    cell_clip = 0
    use_peephole = False
    x = constant_op.constant(0.837607, shape=[28, 29], dtype=dtypes.float32)
    cs_prev = constant_op.constant(0, shape=[28, 17], dtype=dtypes.float32)
    h_prev = constant_op.constant(
        0.592631638, shape=[28, 17], dtype=dtypes.float32)
    w = constant_op.constant(0.887386262, shape=[46, 68], dtype=dtypes.float32)
    wci = constant_op.constant(0, shape=[], dtype=dtypes.float32)
    wcf = constant_op.constant(0, shape=[17], dtype=dtypes.float32)
    wco = constant_op.constant(
        0.592631638, shape=[28, 17], dtype=dtypes.float32)
    b = constant_op.constant(0.75259006, shape=[68], dtype=dtypes.float32)
    with self.assertRaises(errors_impl.InvalidArgumentError):
      self.evaluate(
          gen_rnn_ops.lstm_block_cell(
              x=x,
              cs_prev=cs_prev,
              h_prev=h_prev,
              w=w,
              wci=wci,
              wcf=wcf,
              wco=wco,
              b=b,
              forget_bias=forget_bias,
              cell_clip=cell_clip,
              use_peephole=use_peephole))

  @test_util.run_in_graph_and_eager_modes
  def testLSTMBlockCellEmptyInputRaisesError(self):
    # [FUNCTION DOC]
    # Scenario: call the raw LSTM block cell kernel with half-precision constants where several
    # inputs have a zero-sized dimension. Expectation: an invalid-argument error whose message
    # contains the phrase "is empty".
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError, "is empty"):
      self.evaluate(
          gen_rnn_ops.lstm_block_cell(
              x=constant_op.constant(0, shape=[2, 16], dtype=dtypes.half),
              cs_prev=constant_op.constant(0, shape=[2, 0], dtype=dtypes.half),
              h_prev=constant_op.constant(0, shape=[2, 0], dtype=dtypes.half),
              w=constant_op.constant(0, shape=[16, 0], dtype=dtypes.half),
              wci=constant_op.constant(0, shape=[5], dtype=dtypes.half),
              wcf=constant_op.constant(0, shape=[16], dtype=dtypes.half),
              wco=constant_op.constant(0, shape=[13], dtype=dtypes.half),
              b=constant_op.constant(0, shape=[0], dtype=dtypes.half),
              forget_bias=112.66590343649887,
              cell_clip=67.12389445926587,
              use_peephole=False,
          )
      )

  @test_util.run_in_graph_and_eager_modes
  def testLSTMBlockCellGradErrorHandling(self):
    # [FUNCTION DOC]
    # Scenario: call the second-version block LSTM gradient kernel with inputs of the wrong rank
    # (rank-3 tensors where lower rank is required). Expectation: either a value error or an
    # invalid-argument error whose message contains "must be rank".
    use_peephole = False
    seq_len_max = constant_op.constant(1, shape=[], dtype=dtypes.int64)
    x = constant_op.constant(0.504355371, shape=[1, 1, 1], dtype=dtypes.float32)
    cs_prev = constant_op.constant(
        0.504355371, shape=[1, 1, 1], dtype=dtypes.float32)
    h_prev = constant_op.constant(
        0.504355371, shape=[1, 1], dtype=dtypes.float32)
    w = constant_op.constant(0.504355371, shape=[1, 1], dtype=dtypes.float32)
    wci = constant_op.constant(0.504355371, shape=[1], dtype=dtypes.float32)
    wcf = constant_op.constant(0.504355371, shape=[1], dtype=dtypes.float32)
    wco = constant_op.constant(0.504355371, shape=[1], dtype=dtypes.float32)
    b = constant_op.constant(0.504355371, shape=[1], dtype=dtypes.float32)
    i = constant_op.constant(0.504355371, shape=[1, 1, 1], dtype=dtypes.float32)
    cs = constant_op.constant(
        0.504355371, shape=[1, 1, 1], dtype=dtypes.float32)
    f = constant_op.constant(0.504355371, shape=[1, 1, 1], dtype=dtypes.float32)
    o = constant_op.constant(0.504355371, shape=[1, 1, 1], dtype=dtypes.float32)
    ci = constant_op.constant(
        0.504355371, shape=[1, 1, 1], dtype=dtypes.float32)
    co = constant_op.constant(
        0.504355371, shape=[1, 1, 1], dtype=dtypes.float32)
    h = constant_op.constant(0.504355371, shape=[1, 1, 1], dtype=dtypes.float32)
    cs_grad = constant_op.constant(
        0.504355371, shape=[1, 1, 1], dtype=dtypes.float32)
    h_grad = constant_op.constant(
        0.504355371, shape=[1, 1, 1], dtype=dtypes.float32)
    with self.assertRaisesRegex((ValueError, errors_impl.InvalidArgumentError),
                                "must be rank"):
      self.evaluate(
          gen_rnn_ops.block_lstm_grad_v2(
              seq_len_max=seq_len_max,
              x=x,
              cs_prev=cs_prev,
              h_prev=h_prev,
              w=w,
              wci=wci,
              wcf=wcf,
              wco=wco,
              b=b,
              i=i,
              cs=cs,
              f=f,
              o=o,
              ci=ci,
              co=co,
              h=h,
              cs_grad=cs_grad,
              h_grad=h_grad,
              use_peephole=use_peephole))

  def testLSTMBlockInvalidArgument(self):
    # [FUNCTION DOC]
    # Scenario (regression for a reported crash): call the full block LSTM op with random inputs
    # that include zero-sized dimensions and an oddly shaped sequence-length tensor, with negative
    # forget bias and clip. Expectation: an invalid-argument error is raised rather than a crash.
    # Test case for GitHub issue 58175
    forget_bias = -121.22699269620765
    cell_clip = -106.82307555235684
    use_peephole = False
    seq_len_max = math_ops.saturate_cast(
        random_ops.random_uniform(
            [13, 11, 0], minval=0, maxval=64, dtype=dtypes.int64
        ),
        dtype=dtypes.int64,
    )
    x = random_ops.random_uniform([1, 3, 15], dtype=dtypes.float32)
    cs_prev = random_ops.random_uniform([3, 0], dtype=dtypes.float32)
    h_prev = random_ops.random_uniform([3, 0], dtype=dtypes.float32)
    w = random_ops.random_uniform([15, 0], dtype=dtypes.float32)
    wci = random_ops.random_uniform([0], dtype=dtypes.float32)
    wcf = random_ops.random_uniform([0], dtype=dtypes.float32)
    wco = random_ops.random_uniform([0], dtype=dtypes.float32)
    b = random_ops.random_uniform([0], dtype=dtypes.float32)
    with self.assertRaises(errors_impl.InvalidArgumentError):
      self.evaluate(
          gen_rnn_ops.BlockLSTM(
              forget_bias=forget_bias,
              cell_clip=cell_clip,
              use_peephole=use_peephole,
              seq_len_max=seq_len_max,
              x=x,
              cs_prev=cs_prev,
              h_prev=h_prev,
              w=w,
              wci=wci,
              wcf=wcf,
              wco=wco,
              b=b,
          )
      )


class BidirectionalRNNTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the static and dynamic bidirectional RNN drivers using LSTM cells with identical
  # initialization in both directions, so the backward output is the forward output reversed in
  # time. Has builder helpers returning inputs, outputs, states and sequence lengths, check helpers
  # parameterized over shape knowledge, state tuples, time-major layout and sequence lengths, and
  # variable-scope naming tests. Fixture: setUp seeds numpy.

  def setUp(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: does not call the parent setUp, so base-class test fixture setup is skipped.
    # Fixture. Stores a fixed integer seed on the instance and seeds numpy's random generator.
    self._seed = 23489
    np.random.seed(self._seed)

  def _createBidirectionalRNN(self, use_shape, use_sequence_length, scope=None):
    # [FUNCTION DOC]
    # HELPER: builds the static bidirectional LSTM graph for the tests of that driver.
    # Inputs: use_shape (boolean) chooses a fully known batch dimension versus an unknown one;
    # use_sequence_length (boolean) adds an integer sequence-length placeholder; scope is an
    # optional variable scope or name. Builds forward and backward LSTM cells (non-tuple state)
    # with the same seeded initializer, runs the static bidirectional RNN over eight placeholders,
    # asserts the number of outputs and that each output has width twice the units, and stacks the
    # outputs. Returns a tuple of: random input value, the input placeholder list, stacked
    # outputs, forward state, backward state, and the sequence-length placeholder (or nothing).
    num_units = 3
    input_size = 5
    batch_size = 2
    max_length = 8

    initializer = init_ops.random_uniform_initializer(
        -0.01, 0.01, seed=self._seed)
    sequence_length = array_ops.placeholder(
        dtypes.int64) if use_sequence_length else None
    cell_fw = rnn_cell.LSTMCell(
        num_units, input_size, initializer=initializer, state_is_tuple=False)
    cell_bw = rnn_cell.LSTMCell(
        num_units, input_size, initializer=initializer, state_is_tuple=False)
    inputs = max_length * [
        array_ops.placeholder(
            dtypes.float32,
            shape=(batch_size, input_size) if use_shape else (None, input_size))
    ]
    outputs, state_fw, state_bw = rnn.static_bidirectional_rnn(
        cell_fw,
        cell_bw,
        inputs,
        dtype=dtypes.float32,
        sequence_length=sequence_length,
        scope=scope)
    self.assertEqual(len(outputs), len(inputs))
    for out in outputs:
      self.assertEqual(out.get_shape().as_list(),
                       [batch_size if use_shape else None, 2 * num_units])

    input_value = np.random.randn(batch_size, input_size)
    outputs = array_ops_stack.stack(outputs)

    return input_value, inputs, outputs, state_fw, state_bw, sequence_length

  def _testBidirectionalRNN(self, use_shape):
    # [FUNCTION DOC]
    # HELPER: shared body of the bidirectional static RNN test with sequence lengths.
    # KNOWN ISSUE: the in-body comment says the backward output occupies depths 4 to 5 but the
    # assertions (correctly) use depths 3 to 5; the comment is off by one.
    # Input: use_shape, whether the batch dimension is statically known. Builds the graph with
    # sequence lengths 2 and 3 fed in, runs it, and asserts that forward outputs equal the
    # backward outputs mirrored within each example's valid length, and that final forward and
    # backward states are the same.
    with self.session(graph=ops.Graph()) as sess:
      input_value, inputs, outputs, state_fw, state_bw, sequence_length = (
          self._createBidirectionalRNN(use_shape, True))
      variables_lib.global_variables_initializer().run()
      # Run with pre-specified sequence length of 2, 3
      out, s_fw, s_bw = sess.run(
          [outputs, state_fw, state_bw],
          feed_dict={
              inputs[0]: input_value,
              sequence_length: [2, 3]
          })

      # Since the forward and backward LSTM cells were initialized with the
      # same parameters, the forward and backward output has to be the same,
      # but reversed in time. The format is output[time][batch][depth], and
      # due to depth concatenation (as num_units=3 for both RNNs):
      # - forward output:  out[][][depth] for 0 <= depth < 3
      # - backward output: out[][][depth] for 4 <= depth < 6
      #
      # First sequence in batch is length=2
      # Check that the time=0 forward output is equal to time=1 backward output
      self.assertAllClose(out[0][0][0], out[1][0][3])
      self.assertAllClose(out[0][0][1], out[1][0][4])
      self.assertAllClose(out[0][0][2], out[1][0][5])
      # Check that the time=1 forward output is equal to time=0 backward output
      self.assertAllClose(out[1][0][0], out[0][0][3])
      self.assertAllClose(out[1][0][1], out[0][0][4])
      self.assertAllClose(out[1][0][2], out[0][0][5])

      # Second sequence in batch is length=3
      # Check that the time=0 forward output is equal to time=2 backward output
      self.assertAllClose(out[0][1][0], out[2][1][3])
      self.assertAllClose(out[0][1][1], out[2][1][4])
      self.assertAllClose(out[0][1][2], out[2][1][5])
      # Check that the time=1 forward output is equal to time=1 backward output
      self.assertAllClose(out[1][1][0], out[1][1][3])
      self.assertAllClose(out[1][1][1], out[1][1][4])
      self.assertAllClose(out[1][1][2], out[1][1][5])
      # Check that the time=2 forward output is equal to time=0 backward output
      self.assertAllClose(out[2][1][0], out[0][1][3])
      self.assertAllClose(out[2][1][1], out[0][1][4])
      self.assertAllClose(out[2][1][2], out[0][1][5])
      # Via the reasoning above, the forward and backward final state should be
      # exactly the same
      self.assertAllClose(s_fw, s_bw)

  def _testBidirectionalRNNWithoutSequenceLength(self, use_shape):
    # [FUNCTION DOC]
    # HELPER: shared body of the bidirectional static RNN test without sequence lengths.
    # Input: use_shape, whether the batch dimension is statically known. Runs the graph and
    # asserts that, for all eight steps, the forward output at step i equals the backward output at
    # the mirrored step, and that final forward and backward states are close.
    with self.session(graph=ops.Graph()) as sess:
      input_value, inputs, outputs, state_fw, state_bw, _ = (
          self._createBidirectionalRNN(use_shape, False))
      variables_lib.global_variables_initializer().run()
      out, s_fw, s_bw = sess.run(
          [outputs, state_fw, state_bw], feed_dict={
              inputs[0]: input_value
          })

      # Since the forward and backward LSTM cells were initialized with the
      # same parameters, the forward and backward output has to be the same,
      # but reversed in time. The format is output[time][batch][depth], and
      # due to depth concatenation (as num_units=3 for both RNNs):
      # - forward output:  out[][][depth] for 0 <= depth < 3
      # - backward output: out[][][depth] for 4 <= depth < 6
      #
      # Both sequences in batch are length=8.  Check that the time=i
      # forward output is equal to time=8-1-i backward output
      for i in range(8):
        self.assertAllClose(out[i][0][0:3], out[8 - 1 - i][0][3:6])
        self.assertAllClose(out[i][1][0:3], out[8 - 1 - i][1][3:6])
      # Via the reasoning above, the forward and backward final state should be
      # exactly the same
      self.assertAllClose(s_fw, s_bw)

  @test_util.run_v1_only("b/124229375")
  def testBidirectionalRNN(self):
    # [FUNCTION DOC]
    # Runs the static bidirectional check with sequence lengths for both unknown and known batch
    # shapes (graph mode only).
    self._testBidirectionalRNN(use_shape=False)
    self._testBidirectionalRNN(use_shape=True)

  @test_util.run_v1_only("b/124229375")
  def testBidirectionalRNNWithoutSequenceLength(self):
    # [FUNCTION DOC]
    # Runs the static bidirectional check without sequence lengths for both unknown and known
    # batch shapes (graph mode only).
    self._testBidirectionalRNNWithoutSequenceLength(use_shape=False)
    self._testBidirectionalRNNWithoutSequenceLength(use_shape=True)

  def _createBidirectionalDynamicRNN(self,
                                     use_shape,
                                     use_state_tuple,
                                     use_time_major,
                                     use_sequence_length,
                                     scope=None):
    # [FUNCTION DOC]
    # HELPER: builds the dynamic bidirectional LSTM graph for the tests of that driver.
    # Inputs: use_shape (known versus unknown batch dimension), use_state_tuple (tuple versus
    # concatenated LSTM state), use_time_major (time-major versus batch-major inputs and outputs),
    # use_sequence_length (add a sequence-length placeholder), scope (optional scope or name).
    # Builds two seeded LSTM cells, stacks eight placeholders (transposing when batch-major), runs
    # the bidirectional dynamic RNN, concatenates the two directions' outputs, and asserts the
    # resulting static shape. Returns a tuple of: random input value, input placeholders, outputs,
    # forward state, backward state, and the sequence-length placeholder (or nothing).
    num_units = 3
    input_size = 5
    batch_size = 2
    max_length = 8

    initializer = init_ops.random_uniform_initializer(
        -0.01, 0.01, seed=self._seed)
    sequence_length = (
        array_ops.placeholder(dtypes.int64) if use_sequence_length else None)
    cell_fw = rnn_cell.LSTMCell(
        num_units, initializer=initializer, state_is_tuple=use_state_tuple)
    cell_bw = rnn_cell.LSTMCell(
        num_units, initializer=initializer, state_is_tuple=use_state_tuple)
    inputs = max_length * [
        array_ops.placeholder(
            dtypes.float32,
            shape=(batch_size if use_shape else None, input_size))
    ]
    inputs_c = array_ops_stack.stack(inputs)
    if not use_time_major:
      inputs_c = array_ops.transpose(inputs_c, [1, 0, 2])
    outputs, states = rnn.bidirectional_dynamic_rnn(
        cell_fw,
        cell_bw,
        inputs_c,
        sequence_length,
        dtype=dtypes.float32,
        time_major=use_time_major,
        scope=scope)
    outputs = array_ops.concat(outputs, 2)
    state_fw, state_bw = states
    outputs_shape = [None, max_length, 2 * num_units]
    if use_shape:
      outputs_shape[0] = batch_size
    if use_time_major:
      outputs_shape[0], outputs_shape[1] = outputs_shape[1], outputs_shape[0]
    self.assertEqual(outputs.get_shape().as_list(), outputs_shape)

    input_value = np.random.randn(batch_size, input_size)

    return input_value, inputs, outputs, state_fw, state_bw, sequence_length

  def _testBidirectionalDynamicRNN(self, use_shape, use_state_tuple,
                                   use_time_major, use_sequence_length):
    # [FUNCTION DOC]
    # HELPER: shared body of the dynamic bidirectional RNN check for one option combination.
    # KNOWN ISSUE: the in-body comment says the backward output occupies depths 4 to 5 while the
    # assertions use depths 3 to 5.
    # Inputs: four booleans (known batch shape, tuple state, time-major layout, sequence lengths).
    # Builds and runs the graph (lengths 2 and 3 when used), swaps axes for batch-major results,
    # then asserts forward outputs mirror backward outputs within each valid length (or over the
    # full eight steps without lengths) and that forward and backward final states match.
    with self.session(graph=ops.Graph()) as sess:
      input_value, inputs, outputs, state_fw, state_bw, sequence_length = (
          self._createBidirectionalDynamicRNN(
              use_shape, use_state_tuple, use_time_major, use_sequence_length))
      variables_lib.global_variables_initializer().run()
      # Run with pre-specified sequence length of 2, 3
      feed_dict = ({sequence_length: [2, 3]} if use_sequence_length else {})
      feed_dict.update({inputs[0]: input_value})
      if use_state_tuple:
        out, c_fw, m_fw, c_bw, m_bw = sess.run(
            [outputs, state_fw[0], state_fw[1], state_bw[0], state_bw[1]],
            feed_dict=feed_dict)
        s_fw = (c_fw, m_fw)
        s_bw = (c_bw, m_bw)
      else:
        feed_dict.update({inputs[0]: input_value})
        out, s_fw, s_bw = sess.run(
            [outputs, state_fw, state_bw], feed_dict=feed_dict)

      # Since the forward and backward LSTM cells were initialized with the
      # same parameters, the forward and backward output has to be the same,
      # but reversed in time. The format is output[time][batch][depth], and
      # due to depth concatenation (as num_units=3 for both RNNs):
      # - forward output:  out[][][depth] for 0 <= depth < 3
      # - backward output: out[][][depth] for 4 <= depth < 6
      #
      if not use_time_major:
        out = np.swapaxes(out, 0, 1)

      if use_sequence_length:
        # First sequence in batch is length=2
        # Check that the t=0 forward output is equal to t=1 backward output
        self.assertEqual(out[0][0][0], out[1][0][3])
        self.assertEqual(out[0][0][1], out[1][0][4])
        self.assertEqual(out[0][0][2], out[1][0][5])
        # Check that the t=1 forward output is equal to t=0 backward output
        self.assertEqual(out[1][0][0], out[0][0][3])
        self.assertEqual(out[1][0][1], out[0][0][4])
        self.assertEqual(out[1][0][2], out[0][0][5])

        # Second sequence in batch is length=3
        # Check that the t=0 forward output is equal to t=2 backward output
        self.assertEqual(out[0][1][0], out[2][1][3])
        self.assertEqual(out[0][1][1], out[2][1][4])
        self.assertEqual(out[0][1][2], out[2][1][5])
        # Check that the t=1 forward output is equal to t=1 backward output
        self.assertEqual(out[1][1][0], out[1][1][3])
        self.assertEqual(out[1][1][1], out[1][1][4])
        self.assertEqual(out[1][1][2], out[1][1][5])
        # Check that the t=2 forward output is equal to t=0 backward output
        self.assertEqual(out[2][1][0], out[0][1][3])
        self.assertEqual(out[2][1][1], out[0][1][4])
        self.assertEqual(out[2][1][2], out[0][1][5])
        # Via the reasoning above, the forward and backward final state should
        # be exactly the same
        self.assertAllClose(s_fw, s_bw)
      else:  # not use_sequence_length
        max_length = 8  # from createBidirectionalDynamicRNN
        for t in range(max_length):
          self.assertAllEqual(out[t, :, 0:3], out[max_length - t - 1, :, 3:6])
        self.assertAllClose(s_fw, s_bw)

  @test_util.run_v1_only("b/124229375")
  def testBidirectionalDynamicRNN(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the comment in the body claims 2^5 option values and five flags, but the code
    # takes the product over four booleans, giving 16 combinations.
    # Runs the dynamic bidirectional check over every combination of the four boolean options
    # (known shape, tuple state, time-major, sequence lengths). Graph mode only.
    # Generate 2^5 option values
    # from [True, True, True, True, True] to [False, False, False, False, False]
    options = itertools.product([True, False], repeat=4)
    for option in options:
      self._testBidirectionalDynamicRNN(
          use_shape=option[0],
          use_state_tuple=option[1],
          use_time_major=option[2],
          use_sequence_length=option[3])

  def _testScope(self, factory, prefix="prefix", use_outer_scope=True):
    # [FUNCTION DOC]
    # HELPER: shared body for the bidirectional variable-scope naming tests.
    # KNOWN ISSUE: the call that creates the global variables initializer op is made but never run,
    # so it has no effect.
    # Inputs: factory is a callback that builds a bidirectional RNN given a scope (nothing, a
    # string or a scope object); prefix is the expected scope name (nothing means the default
    # bidirectional scope); use_outer_scope selects passing a scope object versus a string. Builds
    # in a fresh graph, logs variable names, and asserts every global variable name begins with the
    # prefix and a slash. Returns nothing.
    # REMARKS: factory(scope) is a function accepting a scope
    #          as an argument, such scope can be None, a string
    #          or a VariableScope instance.
    with self.session(graph=ops.Graph()):
      if use_outer_scope:
        with variable_scope.variable_scope(prefix) as scope:
          factory(scope)
      else:
        factory(prefix)

      # check that all the variables names starts
      # with the proper scope.
      variables_lib.global_variables_initializer()
      all_vars = variables_lib.global_variables()
      prefix = prefix or "bidirectional_rnn"
      scope_vars = [v for v in all_vars if v.name.startswith(prefix + "/")]
      tf_logging.info("BiRNN with scope: %s (%s)" %
                      (prefix, "scope" if use_outer_scope else "str"))
      for v in scope_vars:
        tf_logging.info(v.name)
      self.assertEqual(len(scope_vars), len(all_vars))

  @test_util.run_v1_only("b/124229375")
  def testBidirectionalRNNScope(self):
    # [FUNCTION DOC]
    # Scenario: build the static bidirectional RNN through a factory and check variable naming with
    # an outer scope object, a scope name string and the default scope, using the scope helper.

    def factory(scope):
      # [FUNCTION DOC]
      # HELPER: builds the static bidirectional RNN (known shape, with sequence length) in the
      # given scope and returns the builder's result tuple.
      return self._createBidirectionalRNN(
          use_shape=True, use_sequence_length=True, scope=scope)

    self._testScope(factory, use_outer_scope=True)
    self._testScope(factory, use_outer_scope=False)
    self._testScope(factory, prefix=None, use_outer_scope=False)

  @test_util.run_v1_only("b/124229375")
  def testBidirectionalDynamicRNNScope(self):
    # [FUNCTION DOC]
    # Scenario: build the dynamic bidirectional RNN (tuple state, sequence lengths, known shape)
    # in both time-major and batch-major layouts and check variable naming for an outer scope
    # object, a name string and the default scope, using the scope helper.

    def get_factory(use_time_major):
      # [FUNCTION DOC]
      # HELPER: creates a scope-taking builder for the scope helper.
      # Input: use_time_major, whether the dynamic RNN uses time-major layout. Returns the nested
      # factory callback.

      def factory(scope):
        # [FUNCTION DOC]
        # HELPER: builds the dynamic bidirectional RNN in the given scope with the layout chosen
        # by the enclosing call, and returns the builder's result tuple.
        return self._createBidirectionalDynamicRNN(
            use_shape=True,
            use_state_tuple=True,
            use_sequence_length=True,
            use_time_major=use_time_major,
            scope=scope)

      return factory

    self._testScope(get_factory(True), use_outer_scope=True)
    self._testScope(get_factory(True), use_outer_scope=False)
    self._testScope(get_factory(True), prefix=None, use_outer_scope=False)
    self._testScope(get_factory(False), use_outer_scope=True)
    self._testScope(get_factory(False), use_outer_scope=False)
    self._testScope(get_factory(False), prefix=None, use_outer_scope=False)


class MultiDimensionalLSTMTest(test.TestCase):
  # [CLASS DOC]
  # Test case checking that all RNN drivers (static, dynamic, static bidirectional, static
  # state-saving) accept per-step inputs with several feature dimensions, using the dummy
  # multi-dimensional cell. Fixture: setUp seeds numpy.

  def setUp(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: does not call the parent setUp, so base-class test fixture setup is skipped.
    # Fixture. Stores a fixed integer seed on the instance and seeds numpy's random generator.
    self._seed = 23489
    np.random.seed(self._seed)

  @test_util.run_v1_only("b/124229375")
  def testMultiDimensionalLSTMAllRNNContainers(self):
    # [FUNCTION DOC]
    # Scenario: run the dummy multi-dimensional cell (feature shape 3 by 4 by 5) through static,
    # dynamic, bidirectional and state-saving drivers with sequence lengths 4 and 6. Asserts output
    # static shapes (doubled feature axis for bidirectional), that static, dynamic and
    # state-saving outputs are equal, that bidirectional output equals static output duplicated
    # along the feature axis, and that all final states agree.
    feature_dims = (3, 4, 5)
    input_size = feature_dims
    batch_size = 2
    max_length = 8
    sequence_length = [4, 6]
    with self.session(graph=ops.Graph()) as sess:
      inputs = max_length * [
          array_ops.placeholder(dtypes.float32, shape=(None,) + input_size)
      ]
      inputs_using_dim = max_length * [
          array_ops.placeholder(
              dtypes.float32, shape=(batch_size,) + input_size)
      ]
      inputs_c = array_ops_stack.stack(inputs)
      # Create a cell for the whole test. This is fine because the cell has no
      # variables.
      cell = DummyMultiDimensionalLSTM(feature_dims)
      state_saver = TestStateSaver(batch_size, input_size)
      outputs_static, state_static = rnn.static_rnn(
          cell, inputs, dtype=dtypes.float32, sequence_length=sequence_length)
      outputs_dynamic, state_dynamic = rnn.dynamic_rnn(
          cell,
          inputs_c,
          dtype=dtypes.float32,
          time_major=True,
          sequence_length=sequence_length)
      outputs_bid, state_fw, state_bw = rnn.static_bidirectional_rnn(
          cell,
          cell,
          inputs_using_dim,
          dtype=dtypes.float32,
          sequence_length=sequence_length)
      outputs_sav, state_sav = rnn.static_state_saving_rnn(
          cell,
          inputs_using_dim,
          sequence_length=sequence_length,
          state_saver=state_saver,
          state_name=("h", "c"))

      self.assertEqual(outputs_dynamic.get_shape().as_list(),
                       inputs_c.get_shape().as_list())
      for out, inp in zip(outputs_static, inputs):
        self.assertEqual(out.get_shape().as_list(), inp.get_shape().as_list())
      for out, inp in zip(outputs_bid, inputs_using_dim):
        input_shape_list = inp.get_shape().as_list()
        # fwd and bwd activations are concatenated along the second dim.
        input_shape_list[1] *= 2
        self.assertEqual(out.get_shape().as_list(), input_shape_list)

      variables_lib.global_variables_initializer().run()

      input_total_size = (batch_size,) + input_size
      input_value = np.random.randn(*input_total_size)
      outputs_static_v = sess.run(
          outputs_static, feed_dict={
              inputs[0]: input_value
          })
      outputs_dynamic_v = sess.run(
          outputs_dynamic, feed_dict={
              inputs[0]: input_value
          })
      outputs_bid_v = sess.run(
          outputs_bid, feed_dict={
              inputs_using_dim[0]: input_value
          })
      outputs_sav_v = sess.run(
          outputs_sav, feed_dict={
              inputs_using_dim[0]: input_value
          })

      self.assertAllEqual(outputs_static_v, outputs_dynamic_v)
      self.assertAllEqual(outputs_static_v, outputs_sav_v)
      outputs_static_array = np.array(outputs_static_v)
      outputs_static_array_double = np.concatenate(
          (outputs_static_array, outputs_static_array), axis=2)
      outputs_bid_array = np.array(outputs_bid_v)
      self.assertAllEqual(outputs_static_array_double, outputs_bid_array)

      state_static_v = sess.run(
          state_static, feed_dict={
              inputs[0]: input_value
          })
      state_dynamic_v = sess.run(
          state_dynamic, feed_dict={
              inputs[0]: input_value
          })
      state_bid_fw_v = sess.run(
          state_fw, feed_dict={
              inputs_using_dim[0]: input_value
          })
      state_bid_bw_v = sess.run(
          state_bw, feed_dict={
              inputs_using_dim[0]: input_value
          })
      state_sav_v = sess.run(
          state_sav, feed_dict={
              inputs_using_dim[0]: input_value
          })
      self.assertAllEqual(np.hstack(state_static_v), np.hstack(state_dynamic_v))
      self.assertAllEqual(np.hstack(state_static_v), np.hstack(state_sav_v))
      self.assertAllEqual(np.hstack(state_static_v), np.hstack(state_bid_fw_v))
      self.assertAllEqual(np.hstack(state_static_v), np.hstack(state_bid_bw_v))


class NestedLSTMTest(test.TestCase):
  # [CLASS DOC]
  # Test case checking that all RNN drivers handle cells whose inputs, outputs and states are
  # nested tuples, using the nested dummy cell. Fixture: setUp seeds numpy.

  def setUp(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: does not call the parent setUp, so base-class test fixture setup is skipped.
    # Fixture. Stores a fixed integer seed on the instance and seeds numpy's random generator.
    self._seed = 23489
    np.random.seed(self._seed)

  @test_util.run_v1_only("b/124229375")
  def testNestedIOLSTMAllRNNContainers(self):
    # [FUNCTION DOC]
    # Scenario: run the nested tuple dummy cell (pairs of inputs and outputs, pair state) through
    # the dynamic, static, bidirectional and state-saving drivers with sequence lengths 4 and 6.
    # Asserts matching nested shapes, static outputs equal to transposed dynamic outputs and to
    # state-saving outputs, bidirectional outputs equal to static outputs duplicated along the
    # feature axis, and equal final states across drivers.
    input_size = 5
    batch_size = 2
    state_size = 6
    max_length = 8
    sequence_length = [4, 6]
    with self.session(graph=ops.Graph()) as sess:
      state_saver = TestStateSaver(batch_size, state_size)
      single_input = (array_ops.placeholder(
          dtypes.float32, shape=(None, input_size)),
                      array_ops.placeholder(
                          dtypes.float32, shape=(None, input_size)))
      inputs = max_length * [single_input]
      inputs_c = (array_ops_stack.stack([input_[0] for input_ in inputs]),
                  array_ops_stack.stack([input_[1] for input_ in inputs]))
      single_input_using_dim = (array_ops.placeholder(
          dtypes.float32, shape=(batch_size, input_size)),
                                array_ops.placeholder(
                                    dtypes.float32,
                                    shape=(batch_size, input_size)))
      inputs_using_dim = max_length * [single_input_using_dim]

      # Create a cell for the whole test. This is fine because the cell has no
      # variables.
      cell = NestedRNNCell()
      outputs_dynamic, state_dynamic = rnn.dynamic_rnn(
          cell,
          inputs_c,
          dtype=dtypes.float32,
          time_major=True,
          sequence_length=sequence_length)
      outputs_static, state_static = rnn.static_rnn(
          cell, inputs, dtype=dtypes.float32, sequence_length=sequence_length)
      outputs_bid, state_fw, state_bw = rnn.static_bidirectional_rnn(
          cell,
          cell,
          inputs_using_dim,
          dtype=dtypes.float32,
          sequence_length=sequence_length)
      outputs_sav, state_sav = rnn.static_state_saving_rnn(
          cell,
          inputs_using_dim,
          sequence_length=sequence_length,
          state_saver=state_saver,
          state_name=("h", "c"))

      def _assert_same_shape(input1, input2, double=False):
        # [FUNCTION DOC]
        # HELPER: shape comparison for the nested driver scenario.
        # Inputs: two possibly nested structures of tensors, and a boolean double that, when true,
        # expects the second structure's second dimension to be twice the first's. Flattens both,
        # pairs the tensors and asserts the static shapes agree. Returns nothing.
        flat_input1 = nest.flatten(input1)
        flat_input2 = nest.flatten(input2)
        for inp1, inp2 in zip(flat_input1, flat_input2):
          input_shape = inp1.get_shape().as_list()
          if double:
            input_shape[1] *= 2
          self.assertEqual(input_shape, inp2.get_shape().as_list())

      _assert_same_shape(inputs_c, outputs_dynamic)
      _assert_same_shape(inputs, outputs_static)
      _assert_same_shape(inputs_using_dim, outputs_sav)
      _assert_same_shape(inputs_using_dim, outputs_bid, double=True)

      variables_lib.global_variables_initializer().run()

      input_total_size = (batch_size, input_size)
      input_value = (np.random.randn(*input_total_size),
                     np.random.randn(*input_total_size))
      outputs_dynamic_v = sess.run(
          outputs_dynamic, feed_dict={
              single_input: input_value
          })
      outputs_static_v = sess.run(
          outputs_static, feed_dict={
              single_input: input_value
          })
      outputs_sav_v = sess.run(
          outputs_sav, feed_dict={
              single_input_using_dim: input_value
          })
      outputs_bid_v = sess.run(
          outputs_bid, feed_dict={
              single_input_using_dim: input_value
          })

      self.assertAllEqual(outputs_static_v,
                          np.transpose(outputs_dynamic_v, (1, 0, 2, 3)))
      self.assertAllEqual(outputs_static_v, outputs_sav_v)
      outputs_static_array = np.array(outputs_static_v)
      outputs_static_array_double = np.concatenate(
          (outputs_static_array, outputs_static_array), axis=3)
      outputs_bid_array = np.array(outputs_bid_v)
      self.assertAllEqual(outputs_static_array_double, outputs_bid_array)

      state_dynamic_v = sess.run(
          state_dynamic, feed_dict={
              single_input: input_value
          })
      state_static_v = sess.run(
          state_static, feed_dict={
              single_input: input_value
          })
      state_bid_fw_v = sess.run(
          state_fw, feed_dict={
              single_input_using_dim: input_value
          })
      state_bid_bw_v = sess.run(
          state_bw, feed_dict={
              single_input_using_dim: input_value
          })
      state_sav_v = sess.run(
          state_sav, feed_dict={
              single_input_using_dim: input_value
          })
      self.assertAllEqual(np.hstack(state_static_v), np.hstack(state_dynamic_v))
      self.assertAllEqual(np.hstack(state_static_v), np.hstack(state_sav_v))
      self.assertAllEqual(np.hstack(state_static_v), np.hstack(state_bid_fw_v))
      self.assertAllEqual(np.hstack(state_static_v), np.hstack(state_bid_bw_v))


class StateSaverRNNTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the state-saving static RNN: variable scoping and (currently skipped) checks
  # that state-saver methods are evaluated the same number of times. Uses the fake state savers
  # defined in this file. Fixture: setUp seeds numpy.

  def setUp(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: does not call the parent setUp, so base-class test fixture setup is skipped.
    # Fixture. Stores a fixed integer seed on the instance and seeds numpy's random generator.
    self._seed = 23489
    np.random.seed(self._seed)

  def _factory(self, scope, state_saver):
    # [FUNCTION DOC]
    # HELPER: builds the state-saving RNN graph for the tests of this class.
    # Inputs: scope (variable scope, name or nothing) and state_saver (fake saver whose state size
    # is twice the unit count). Creates a non-tuple LSTM, eight zero inputs, and runs the
    # state-saving static RNN under the name save_lstm. Returns a tuple of outputs, final state and
    # the same state saver.
    num_units = state_saver.state_size // 2
    batch_size = state_saver.batch_size
    input_size = 5
    max_length = 8
    initializer = init_ops.random_uniform_initializer(
        -0.01, 0.01, seed=self._seed)
    cell = rnn_cell.LSTMCell(
        num_units,
        use_peepholes=False,
        initializer=initializer,
        state_is_tuple=False)
    inputs = max_length * [
        array_ops.zeros(dtype=dtypes.float32, shape=(batch_size, input_size))
    ]
    out, state = rnn.static_state_saving_rnn(
        cell,
        inputs,
        state_saver=state_saver,
        state_name="save_lstm",
        scope=scope)
    return out, state, state_saver

  def _testScope(self, prefix="prefix", use_outer_scope=True):
    # [FUNCTION DOC]
    # HELPER: shared body for the state-saver RNN variable-scope test.
    # KNOWN ISSUE: the variables initializer op is created only in the string-scope branch and
    # is never run, so it has no effect.
    # Inputs: prefix (expected scope name, nothing means default rnn) and use_outer_scope (pass a
    # scope object versus a string). Builds the state-saving RNN with a fake saver in a fresh graph
    # and asserts all global variable names start with the prefix and a slash. Returns nothing.
    num_units = 3
    batch_size = 2
    state_saver = TestStateSaver(batch_size, 2 * num_units)

    with self.session(graph=ops.Graph()):
      if use_outer_scope:
        with variable_scope.variable_scope(prefix) as scope:
          self._factory(scope=scope, state_saver=state_saver)
      else:
        self._factory(scope=prefix, state_saver=state_saver)
        variables_lib.global_variables_initializer()

      # check that all the variables names starts
      # with the proper scope.
      all_vars = variables_lib.global_variables()
      prefix = prefix or "rnn"
      scope_vars = [v for v in all_vars if v.name.startswith(prefix + "/")]
      tf_logging.info("RNN with scope: %s (%s)" %
                      (prefix, "scope" if use_outer_scope else "str"))
      for v in scope_vars:
        tf_logging.info(v.name)
      self.assertEqual(len(scope_vars), len(all_vars))

  def testStateSaverRNNScope(self):
    # [FUNCTION DOC]
    # Checks variable naming of the state-saving RNN with an outer scope object, a name string and
    # the default scope, via the scope helper.
    self._testScope(use_outer_scope=True)
    self._testScope(use_outer_scope=False)
    self._testScope(prefix=None, use_outer_scope=False)

  def testStateSaverCallsSaveState(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test skips itself immediately (referencing a known bug), so everything
    # after the skip call is dead code and nothing is verified.
    # Intended scenario: with a counting state saver, evaluate outputs and state together, outputs
    # alone and state alone, and assert the state-call counter equals the save-state-call counter.
    """Test that number of calls to state and save_state is equal.

    Test if the order of actual evaluating or skipping evaluation of out,
    state tensors, which are the output tensors from static_state_saving_rnn,
    have influence on number of calls to save_state and state methods of
    state_saver object (the number of calls should be same.)
    """
    self.skipTest("b/124196246 Breakage for sess.run([out, ...]): 2 != 1")

    num_units = 3
    batch_size = 2
    state_saver = TestStateSaverWithCounters(batch_size, 2 * num_units)
    out, state, state_saver = self._factory(scope=None, state_saver=state_saver)

    with self.cached_session() as sess:
      sess.run(variables_lib.global_variables_initializer())
      sess.run(variables_lib.local_variables_initializer())

      _, _, num_state_calls, num_save_state_calls = sess.run([
          out,
          state,
          state_saver.num_state_calls,
          state_saver.num_save_state_calls])
      self.assertEqual(num_state_calls, num_save_state_calls)

      _, num_state_calls, num_save_state_calls = sess.run([
          out,
          state_saver.num_state_calls,
          state_saver.num_save_state_calls])
      self.assertEqual(num_state_calls, num_save_state_calls)

      _, num_state_calls, num_save_state_calls = sess.run([
          state,
          state_saver.num_state_calls,
          state_saver.num_save_state_calls])
      self.assertEqual(num_state_calls, num_save_state_calls)

class GRUTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the GRU cell inside the dynamic RNN: a smoke run with random sequence lengths
  # and a variable-scope naming check. Fixture: setUp seeds numpy.

  def setUp(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: does not call the parent setUp, so base-class test fixture setup is skipped.
    # Fixture. Stores a fixed integer seed on the instance and seeds numpy's random generator.
    self._seed = 23489
    np.random.seed(self._seed)

  @test_util.run_v1_only("b/124229375")
  def testDynamic(self):
    # [FUNCTION DOC]
    # Smoke scenario: time-major dynamic RNN with a GRU cell over a placeholder of eight steps and
    # random sequence lengths. Initializes variables and runs outputs and state; nothing is
    # asserted beyond the run not failing.
    time_steps = 8
    num_units = 3
    input_size = 5
    batch_size = 2

    input_values = np.random.randn(time_steps, batch_size, input_size)

    sequence_length = np.random.randint(0, time_steps, size=batch_size)

    with self.session(graph=ops.Graph()) as sess:
      concat_inputs = array_ops.placeholder(
          dtypes.float32, shape=(time_steps, batch_size, input_size))

      cell = rnn_cell.GRUCell(num_units=num_units)

      with variable_scope.variable_scope("dynamic_scope"):
        outputs_dynamic, state_dynamic = rnn.dynamic_rnn(
            cell,
            inputs=concat_inputs,
            sequence_length=sequence_length,
            time_major=True,
            dtype=dtypes.float32)

      feeds = {concat_inputs: input_values}

      # Initialize
      variables_lib.global_variables_initializer().run(feed_dict=feeds)

      sess.run([outputs_dynamic, state_dynamic], feed_dict=feeds)

  def _testScope(self, factory, prefix="prefix", use_outer_scope=True):
    # [FUNCTION DOC]
    # HELPER: shared body for the GRU variable-scope naming test.
    # KNOWN ISSUE: the variables initializer op is created only in the string-scope branch and is
    # never run, so it has no effect.
    # Inputs: factory (callback building a dynamic GRU RNN given a scope or name), prefix (expected
    # scope name, nothing means default rnn) and use_outer_scope (scope object versus string).
    # Asserts all global variable names start with the prefix and a slash. Returns nothing.
    with self.session(graph=ops.Graph()):
      if use_outer_scope:
        with variable_scope.variable_scope(prefix) as scope:
          factory(scope)
      else:
        factory(prefix)
        variables_lib.global_variables_initializer()

      # check that all the variables names starts
      # with the proper scope.
      all_vars = variables_lib.global_variables()
      prefix = prefix or "rnn"
      scope_vars = [v for v in all_vars if v.name.startswith(prefix + "/")]
      tf_logging.info("RNN with scope: %s (%s)" %
                      (prefix, "scope" if use_outer_scope else "str"))
      for v in scope_vars:
        tf_logging.info(v.name)
      self.assertEqual(len(scope_vars), len(all_vars))

  @test_util.run_v1_only("b/124229375")
  def testDynamicScope(self):
    # [FUNCTION DOC]
    # Scenario: build a time-major dynamic GRU RNN with random sequence lengths through a nested
    # factory and check variable naming for a scope object, a name string and the default scope.
    time_steps = 8
    num_units = 3
    input_size = 5
    batch_size = 2
    sequence_length = np.random.randint(0, time_steps, size=batch_size)

    def factory(scope):
      # [FUNCTION DOC]
      # HELPER: builds the dynamic GRU RNN in the given scope for the scope helper.
      # Input: scope (object, string or nothing). Creates a time-major placeholder, a GRU cell, and
      # returns the (outputs, final state) pair from the dynamic RNN.
      concat_inputs = array_ops.placeholder(
          dtypes.float32, shape=(time_steps, batch_size, input_size))
      cell = rnn_cell.GRUCell(num_units=num_units)
      return rnn.dynamic_rnn(
          cell,
          inputs=concat_inputs,
          sequence_length=sequence_length,
          time_major=True,
          dtype=dtypes.float32,
          scope=scope)

    self._testScope(factory, use_outer_scope=True)
    self._testScope(factory, use_outer_scope=False)
    self._testScope(factory, prefix=None, use_outer_scope=False)


class RawRNNTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the low-level raw RNN driver driven by a user loop function. Compares it with
  # the dynamic RNN (outputs, states, gradients), and exercises loop state handling and emitting
  # outputs with a structure different from the cell output. Fixture: setUp seeds numpy.

  def setUp(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: does not call the parent setUp, so base-class test fixture setup is skipped.
    # Fixture. Stores a fixed integer seed on the instance and seeds numpy's random generator.
    self._seed = 23489
    np.random.seed(self._seed)

  @test_util.run_v1_only("b/124229375")
  def _testRawRNN(self, max_time):
    # [FUNCTION DOC]
    # HELPER: shared body for the raw RNN versus dynamic RNN comparison tests.
    # Input: max_time, the number of time steps (zero is allowed). Builds a tuple-state LSTM, a
    # loop function that feeds inputs from a tensor array until each example's sequence length is
    # reached, runs the raw RNN and then the dynamic RNN with shared weights, and computes
    # gradients of both. Asserts outputs and final states are close, and (only when max_time is
    # positive) that gradients with respect to inputs and variables are close. Returns nothing.
    with self.session(graph=ops.Graph()) as sess:
      batch_size = 16
      input_depth = 4
      num_units = 3

      inputs = array_ops.placeholder(
          shape=(max_time, batch_size, input_depth), dtype=dtypes.float32)
      sequence_length = array_ops.placeholder(
          shape=(batch_size,), dtype=dtypes.int32)
      inputs_ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, size=array_ops.shape(inputs)[0])
      inputs_ta = inputs_ta.unstack(inputs)

      cell = rnn_cell.LSTMCell(num_units, state_is_tuple=True)

      def loop_fn(time_, cell_output, cell_state, unused_loop_state):
        # [FUNCTION DOC]
        # HELPER: loop callback handed to the raw RNN driver.
        # Inputs: time step, previous cell output (nothing at step zero), previous cell state, and
        # an unused loop state. Emits the cell output, starts from the zero state at step zero and
        # otherwise passes the state through, marks examples finished when the time reaches their
        # sequence length, and reads the next input from the tensor array (a zero dummy on the very
        # last iteration). Returns the tuple of finished flags, next input, next state, emitted
        # output and a nothing loop state.
        emit_output = cell_output  # == None for time == 0
        if cell_output is None:  # time == 0
          next_state = cell.zero_state(batch_size, dtypes.float32)
        else:
          next_state = cell_state  # copy state through
        elements_finished = (time_ >= sequence_length)
        finished = math_ops.reduce_all(elements_finished)
        # For the very final iteration, we must emit a dummy input
        next_input = cond.cond(
            finished,
            lambda: array_ops.zeros([batch_size, input_depth], dtype=dtypes.float32),
            lambda: inputs_ta.read(time_))
        return (elements_finished, next_input, next_state, emit_output, None)

      reuse_scope = variable_scope.get_variable_scope()

      outputs_ta, final_state, _ = rnn.raw_rnn(cell, loop_fn, scope=reuse_scope)
      outputs = outputs_ta.stack()

      reuse_scope.reuse_variables()
      outputs_dynamic_rnn, final_state_dynamic_rnn = rnn.dynamic_rnn(
          cell,
          inputs,
          time_major=True,
          dtype=dtypes.float32,
          sequence_length=sequence_length,
          scope=reuse_scope)

      variables = variables_lib.trainable_variables()
      gradients = gradients_impl.gradients([outputs, final_state],
                                           [inputs] + variables)
      gradients_dynamic_rnn = gradients_impl.gradients(
          [outputs_dynamic_rnn, final_state_dynamic_rnn], [inputs] + variables)

      variables_lib.global_variables_initializer().run()

      rand_input = np.random.randn(max_time, batch_size, input_depth)
      if max_time == 0:
        rand_seq_len = np.zeros(batch_size)
      else:
        rand_seq_len = np.random.randint(max_time, size=batch_size)

      # To ensure same output lengths for dynamic_rnn and raw_rnn
      rand_seq_len[0] = max_time

      (outputs_val, outputs_dynamic_rnn_val, final_state_val,
       final_state_dynamic_rnn_val) = sess.run(
           [outputs, outputs_dynamic_rnn, final_state, final_state_dynamic_rnn],
           feed_dict={
               inputs: rand_input,
               sequence_length: rand_seq_len
           })

      self.assertAllClose(outputs_dynamic_rnn_val, outputs_val)
      self.assertAllClose(final_state_dynamic_rnn_val, final_state_val)

      # NOTE: Because with 0 time steps, raw_rnn does not have shape
      # information about the input, it is impossible to perform
      # gradients comparisons as the gradients eval will fail.  So
      # this case skips the gradients test.
      if max_time > 0:
        self.assertEqual(len(gradients), len(gradients_dynamic_rnn))
        gradients_val = sess.run(
            gradients,
            feed_dict={
                inputs: rand_input,
                sequence_length: rand_seq_len
            })
        gradients_dynamic_rnn_val = sess.run(
            gradients_dynamic_rnn,
            feed_dict={
                inputs: rand_input,
                sequence_length: rand_seq_len
            })
        self.assertEqual(len(gradients_val), len(gradients_dynamic_rnn_val))
        input_gradients_val = gradients_val[0]
        input_gradients_dynamic_rnn_val = gradients_dynamic_rnn_val[0]
        self.assertAllClose(input_gradients_val,
                            input_gradients_dynamic_rnn_val)
        for i in range(1, len(gradients_val)):
          self.assertAllClose(gradients_dynamic_rnn_val[i], gradients_val[i])

  @test_util.run_v1_only("b/124229375")
  def testRawRNNZeroLength(self):
    # [FUNCTION DOC]
    # Runs the raw-versus-dynamic comparison with zero time steps (gradient checks are skipped for
    # this case). Graph mode only.
    # NOTE: Because with 0 time steps, raw_rnn does not have shape
    # information about the input, it is impossible to perform
    # gradients comparisons as the gradients eval will fail.  So this
    # case skips the gradients test.
    self._testRawRNN(max_time=0)

  def testRawRNN(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: unlike its zero-length sibling, this test has no graph-only decorator, though
    # the shared helper it calls is decorated as graph-only.
    # Runs the raw-versus-dynamic comparison with ten time steps, including gradients.
    self._testRawRNN(max_time=10)

  @test_util.run_v1_only("b/124229375")
  def testLoopState(self):
    # [FUNCTION DOC]
    # Scenario: raw RNN whose loop function carries an integer loop state that is incremented each
    # step after the first. Runs ten steps and asserts the final loop state evaluates to ten.
    with self.session(graph=ops.Graph()):
      max_time = 10
      batch_size = 16
      input_depth = 4
      num_units = 3

      inputs = np.random.randn(max_time, batch_size, input_depth)
      inputs_ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, size=array_ops.shape(inputs)[0])
      inputs_ta = inputs_ta.unstack(inputs)

      cell = rnn_cell.LSTMCell(num_units, state_is_tuple=True)

      def loop_fn(time_, cell_output, cell_state, loop_state):
        # [FUNCTION DOC]
        # HELPER: loop callback for the raw RNN with an integer loop state.
        # Inputs: time step, previous output (nothing at step zero), previous state, loop state.
        # At step zero sets the loop state to zero and the zero cell state; afterwards adds one to
        # the loop state and passes the cell state through. All examples finish at the maximum
        # time. Returns the tuple of finished flags, next input, next state, emitted output, loop
        # state.
        if cell_output is None:
          loop_state = constant_op.constant([0])
          next_state = cell.zero_state(batch_size, dtypes.float32)
        else:
          loop_state = array_ops_stack.stack(
              [array_ops.squeeze(loop_state) + 1])
          next_state = cell_state
        emit_output = cell_output  # == None for time == 0
        elements_finished = array_ops.tile([time_ >= max_time], [batch_size])
        finished = math_ops.reduce_all(elements_finished)
        # For the very final iteration, we must emit a dummy input
        next_input = cond.cond(
            finished,
            lambda: array_ops.zeros([batch_size, input_depth], dtype=dtypes.float32),
            lambda: inputs_ta.read(time_))
        return (elements_finished, next_input, next_state, emit_output,
                loop_state)

      r = rnn.raw_rnn(cell, loop_fn)
      loop_state = r[-1]
      self.assertEqual([10], self.evaluate(loop_state))

  @test_util.run_v1_only("b/124229375")
  def testLoopStateWithTensorArray(self):
    # [FUNCTION DOC]
    # Scenario: raw RNN whose loop state is a growable tensor array of integers, writing the
    # previous entry plus the time index at each step. Runs four steps and asserts the stacked
    # final loop state equals one, two, four, seven, eleven.
    with self.session(graph=ops.Graph()):
      max_time = 4
      batch_size = 16
      input_depth = 4
      num_units = 3

      inputs = np.random.randn(max_time, batch_size, input_depth)
      inputs_ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, size=array_ops.shape(inputs)[0])
      inputs_ta = inputs_ta.unstack(inputs)

      cell = rnn_cell.LSTMCell(num_units, state_is_tuple=True)

      def loop_fn(time_, cell_output, cell_state, loop_state):
        # [FUNCTION DOC]
        # HELPER: loop callback for the raw RNN with a tensor array loop state.
        # Inputs: time step, previous output (nothing at step zero), previous state, loop state.
        # At step zero creates a growable integer tensor array holding one at index zero; later
        # writes the previous entry plus the time index at the current index. Cell state is passed
        # through; all examples finish at the maximum time. Returns the tuple of finished flags,
        # next input, next state, emitted output and loop state.
        if cell_output is None:
          loop_state = tensor_array_ops.TensorArray(
              dynamic_size=True,
              size=0,
              dtype=dtypes.int32,
              clear_after_read=False)
          loop_state = loop_state.write(0, 1)
          next_state = cell.zero_state(batch_size, dtypes.float32)
        else:
          loop_state = loop_state.write(time_,
                                        loop_state.read(time_ - 1) + time_)
          next_state = cell_state
        emit_output = cell_output  # == None for time == 0
        elements_finished = array_ops.tile([time_ >= max_time], [batch_size])
        finished = math_ops.reduce_all(elements_finished)
        # For the very final iteration, we must emit a dummy input
        next_input = cond.cond(
            finished,
            lambda: array_ops.zeros([batch_size, input_depth], dtype=dtypes.float32),
            lambda: inputs_ta.read(time_))
        return (elements_finished, next_input, next_state, emit_output,
                loop_state)

      r = rnn.raw_rnn(cell, loop_fn)
      loop_state = r[-1]
      loop_state = loop_state.stack()
      self.assertAllEqual([1, 2, 2 + 2, 4 + 3, 7 + 4], loop_state)

  @test_util.run_v1_only("b/124229375")
  def testEmitDifferentStructureThanCellOutput(self):
    # [FUNCTION DOC]
    # Scenario: raw RNN whose loop function emits a pair of integer tensors (one with a shape
    # that depends on a placeholder fed at run time) instead of the cell output. Asserts two output
    # tensor arrays with the expected integer dtypes and that stacked emitted values are all ones
    # with shapes time by batch by two by three and time by batch by one.
    with self.session(graph=ops.Graph()) as sess:
      max_time = 10
      batch_size = 16
      input_depth = 4
      num_units = 3

      inputs = np.random.randn(max_time, batch_size, input_depth)
      inputs_ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, size=array_ops.shape(inputs)[0])
      inputs_ta = inputs_ta.unstack(inputs)
      # Verify emit shapes may be unknown by feeding a placeholder that
      # determines an emit shape.
      unknown_dim = array_ops.placeholder(dtype=dtypes.int32)

      cell = rnn_cell.LSTMCell(num_units, state_is_tuple=True)

      def loop_fn(time_, cell_output, cell_state, _):
        # [FUNCTION DOC]
        # HELPER: loop callback for the emit-structure scenario.
        # Inputs: time step, previous output (nothing at step zero), previous state, unused loop
        # state. Emits a pair of integer zero tensors at step zero and pair of ones (batch-sized,
        # one with run-time-dependent width) afterwards; cell state is zero then passed through; all
        # examples finish at the maximum time. Returns finished flags, next input, next state,
        # emitted pair and a nothing loop state.
        if cell_output is None:
          emit_output = (array_ops.zeros([2, 3], dtype=dtypes.int32),
                         array_ops.zeros([unknown_dim], dtype=dtypes.int64))
          next_state = cell.zero_state(batch_size, dtypes.float32)
        else:
          emit_output = (array_ops.ones([batch_size, 2, 3], dtype=dtypes.int32),
                         array_ops.ones(
                             [batch_size, unknown_dim], dtype=dtypes.int64))
          next_state = cell_state
        elements_finished = array_ops.tile([time_ >= max_time], [batch_size])
        finished = math_ops.reduce_all(elements_finished)
        # For the very final iteration, we must emit a dummy input
        next_input = cond.cond(
            finished,
            lambda: array_ops.zeros([batch_size, input_depth], dtype=dtypes.float32),
            lambda: inputs_ta.read(time_))
        return (elements_finished, next_input, next_state, emit_output, None)

      r = rnn.raw_rnn(cell, loop_fn)
      output_ta = r[0]
      self.assertEqual(2, len(output_ta))
      self.assertEqual([dtypes.int32, dtypes.int64],
                       [ta.dtype for ta in output_ta])
      output = [ta.stack() for ta in output_ta]
      output_vals = sess.run(output, feed_dict={unknown_dim: 1})
      self.assertAllEqual(
          np.ones((max_time, batch_size, 2, 3), np.int32), output_vals[0])
      self.assertAllEqual(
          np.ones((max_time, batch_size, 1), np.int64), output_vals[1])

  def _testScope(self, factory, prefix="prefix", use_outer_scope=True):
    # [FUNCTION DOC]
    # HELPER: shared body for the raw RNN variable-scope naming test.
    # KNOWN ISSUE: the variables initializer op is created only in the string-scope branch and is
    # never run, so it has no effect.
    # Inputs: factory (callback building a raw RNN given a scope or name), prefix (expected scope
    # name, nothing means default rnn) and use_outer_scope (scope object versus string). Builds in
    # a fresh graph and asserts every global variable name starts with the prefix and a slash.
    with self.session(graph=ops.Graph()):
      if use_outer_scope:
        with variable_scope.variable_scope(prefix) as scope:
          factory(scope)
      else:
        factory(prefix)
        variables_lib.global_variables_initializer()

      # check that all the variables names starts
      # with the proper scope.
      all_vars = variables_lib.global_variables()
      prefix = prefix or "rnn"
      scope_vars = [v for v in all_vars if v.name.startswith(prefix + "/")]
      tf_logging.info("RNN with scope: %s (%s)" %
                      (prefix, "scope" if use_outer_scope else "str"))
      for v in scope_vars:
        tf_logging.info(v.name)
      self.assertEqual(len(scope_vars), len(all_vars))

  @test_util.run_v1_only("b/124229375")
  def testRawRNNScope(self):
    # [FUNCTION DOC]
    # Scenario: build a raw RNN with an LSTM and a sequence-length-driven loop function through a
    # nested factory, then check variable naming for a scope object, a name string and the
    # default scope via the scope helper.
    max_time = 10
    batch_size = 16
    input_depth = 4
    num_units = 3

    def factory(scope):
      # [FUNCTION DOC]
      # HELPER: builds the raw RNN in the given scope for the scope helper.
      # Input: scope (object, string or nothing). Creates input and sequence-length placeholders,
      # a tensor array of inputs, an LSTM cell and a loop function, and returns the raw RNN result.
      inputs = array_ops.placeholder(
          shape=(max_time, batch_size, input_depth), dtype=dtypes.float32)
      sequence_length = array_ops.placeholder(
          shape=(batch_size,), dtype=dtypes.int32)
      inputs_ta = tensor_array_ops.TensorArray(
          dtype=dtypes.float32, size=array_ops.shape(inputs)[0])
      inputs_ta = inputs_ta.unstack(inputs)

      cell = rnn_cell.LSTMCell(num_units, state_is_tuple=True)

      def loop_fn(time_, cell_output, cell_state, unused_loop_state):
        # [FUNCTION DOC]
        # HELPER: loop callback for the raw RNN used by the scope test.
        # Same behavior as the comparison test's callback: emits the cell output, zero state at
        # step zero then state pass-through, finishes examples at their sequence length and reads
        # inputs from the tensor array. Returns the five-element loop result tuple (loop state is
        # nothing).
        emit_output = cell_output  # == None for time == 0
        if cell_output is None:  # time == 0
          next_state = cell.zero_state(batch_size, dtypes.float32)
        else:
          next_state = cell_state

        elements_finished = (time_ >= sequence_length)
        finished = math_ops.reduce_all(elements_finished)
        # For the very final iteration, we must emit a dummy input
        next_input = cond.cond(
            finished,
            lambda: array_ops.zeros([batch_size, input_depth], dtype=dtypes.float32),
            lambda: inputs_ta.read(time_))
        return (elements_finished, next_input, next_state, emit_output, None)

      return rnn.raw_rnn(cell, loop_fn, scope=scope)

    self._testScope(factory, use_outer_scope=True)
    self._testScope(factory, use_outer_scope=False)
    self._testScope(factory, prefix=None, use_outer_scope=False)


class DeviceWrapperCell(rnn_cell.RNNCell):
  # [CLASS DOC]
  # HELPER: test-local wrapper cell for the device placement tests.
  # Delegates size properties and the step computation to a wrapped cell, but executes the wrapped
  # cell under a device scope when a device string is configured. Holds the wrapped cell and the
  # device (or nothing for no placement).
  """Class to ensure cell calculation happens on a specific device."""

  def __init__(self, cell, device):
    # [FUNCTION DOC]
    # KNOWN ISSUE: does not call the base-class constructor, so base cell initialization is
    # skipped.
    # Constructor. cell is the cell to wrap; device is a device name string or nothing. Stores both.
    self._cell = cell
    self._device = device

  @property
  def output_size(self):
    # [FUNCTION DOC]
    # Property returning the wrapped cell's output size.
    return self._cell.output_size

  @property
  def state_size(self):
    # [FUNCTION DOC]
    # Property returning the wrapped cell's state size.
    return self._cell.state_size

  def __call__(self, input_, state, scope=None):
    # [FUNCTION DOC]
    # Runs one step of the wrapped cell. Inputs: input_ tensor, state, optional scope forwarded to
    # the wrapped cell. If a device was configured, the call happens inside that device scope.
    # Returns whatever the wrapped cell returns (output and new state).
    if self._device is not None:
      with ops.device(self._device):
        return self._cell(input_, state, scope=scope)
    else:
      return self._cell(input_, state, scope=scope)


class TensorArrayOnCorrectDeviceTest(test.TestCase):
  # [CLASS DOC]
  # Test case verifying, via run-time step statistics, that the dynamic RNN places tensor array
  # operations on the expected devices (CPU or GPU) depending on where the RNN, cell and inputs
  # are placed. All tests silently return when no GPU is available.

  def _execute_rnn_on(self,
                      rnn_device=None,
                      cell_device=None,
                      input_device=None):
    # [FUNCTION DOC]
    # HELPER: runs a dynamic RNN with device placement options and returns its trace metadata.
    # Inputs (all optional device name strings or nothing): rnn_device is the device scope for
    # building the RNN, cell_device the device used by the wrapped LSTM cell, input_device the
    # device on which the input constant is created. Builds a peephole LSTM wrapped in the device
    # cell, random inputs and sequence lengths, runs the outputs with full tracing after
    # initializing variables, and returns the run metadata containing per-device step stats.
    batch_size = 3
    time_steps = 7
    input_size = 5
    num_units = 10

    cell = rnn_cell.LSTMCell(num_units, use_peepholes=True)
    gpu_cell = DeviceWrapperCell(cell, cell_device)
    inputs = np.random.randn(batch_size, time_steps, input_size).astype(
        np.float32)
    sequence_length = np.random.randint(0, time_steps, size=batch_size)

    if input_device is not None:
      with ops.device(input_device):
        inputs = constant_op.constant(inputs)

    if rnn_device is not None:
      with ops.device(rnn_device):
        outputs, _ = rnn.dynamic_rnn(
            gpu_cell,
            inputs,
            sequence_length=sequence_length,
            dtype=dtypes.float32)
    else:
      outputs, _ = rnn.dynamic_rnn(
          gpu_cell,
          inputs,
          sequence_length=sequence_length,
          dtype=dtypes.float32)

    with self.session() as sess:
      opts = config_pb2.RunOptions(trace_level=config_pb2.RunOptions.FULL_TRACE)
      run_metadata = config_pb2.RunMetadata()
      variables_lib.global_variables_initializer().run()
      sess.run(outputs, options=opts, run_metadata=run_metadata)

    return run_metadata

  def _retrieve_cpu_gpu_stats(self, run_metadata):
    # [FUNCTION DOC]
    # HELPER: extracts per-device node statistics from run metadata.
    # Input: run_metadata, the trace metadata of a session run. Scans device statistics; takes the
    # one whose device name ends with cpu:0 and the one ending exactly with gpu:0. Returns a pair
    # (CPU node stats, GPU node stats), each nothing if the device is not present.
    cpu_stats = None
    gpu_stats = None
    step_stats = run_metadata.step_stats
    for ds in step_stats.dev_stats:
      if "cpu:0" in ds.device[-5:].lower():
        cpu_stats = ds.node_stats
      if "gpu:0" == ds.device[-5:].lower():
        gpu_stats = ds.node_stats
    return cpu_stats, gpu_stats

  @test_util.run_v1_only("b/124229375")
  def testRNNOnCPUCellOnGPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: returns silently (counts as a pass) when no GPU is available, so the check is
    # not performed on CPU-only machines.
    # Scenario: RNN scope on the CPU, cell on the GPU. Asserts tensor array write and gather
    # operations appear on the GPU and not the CPU, and read and scatter operations appear on the
    # CPU and not the GPU.
    if not test.is_gpu_available():
      return  # Test requires access to a GPU

    gpu_dev = test.gpu_device_name()
    run_metadata = self._execute_rnn_on(
        rnn_device="/cpu:0", cell_device=gpu_dev)
    cpu_stats, gpu_stats = self._retrieve_cpu_gpu_stats(run_metadata)

    def _assert_in(op_str, in_stats, out_stats):
      # [FUNCTION DOC]
      # HELPER: placement assertion. Inputs: op_str (operation name fragment), in_stats (node
      # stats expected to contain it), out_stats (node stats expected not to). Asserts some node in
      # the first contains the fragment and none in the second does.
      self.assertTrue(any(op_str in s.node_name for s in in_stats))
      self.assertFalse(any(op_str in s.node_name for s in out_stats))

    # Writes happen at output of RNN cell
    _assert_in("TensorArrayWrite", gpu_stats, cpu_stats)
    # Gather happens on final TensorArray
    _assert_in("TensorArrayGather", gpu_stats, cpu_stats)
    # Reads happen at input to RNN cell
    _assert_in("TensorArrayRead", cpu_stats, gpu_stats)
    # Scatters happen to get initial input into TensorArray
    _assert_in("TensorArrayScatter", cpu_stats, gpu_stats)

  @test_util.run_v1_only("b/124229375")
  def testRNNOnCPUCellOnCPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: returns silently (counts as a pass) when no GPU is available.
    # Scenario: RNN and cell on the CPU while the input constant lives on the GPU. Asserts all
    # tensor array operations are on the CPU and none on the GPU.
    if not test.is_gpu_available():
      return  # Test requires access to a GPU

    gpu_dev = test.gpu_device_name()
    run_metadata = self._execute_rnn_on(
        rnn_device="/cpu:0", cell_device="/cpu:0", input_device=gpu_dev)
    cpu_stats, gpu_stats = self._retrieve_cpu_gpu_stats(run_metadata)

    def _assert_in(op_str, in_stats, out_stats):
      # [FUNCTION DOC]
      # HELPER: placement assertion. Inputs: op_str (operation name fragment), in_stats (node
      # stats expected to contain it), out_stats (node stats expected not to). Asserts some node in
      # the first contains the fragment and none in the second does.
      self.assertTrue(any(op_str in s.node_name for s in in_stats))
      self.assertFalse(any(op_str in s.node_name for s in out_stats))

    # All TensorArray operations happen on CPU
    _assert_in("TensorArray", cpu_stats, gpu_stats)

  @test_util.run_v1_only("b/124229375")
  def testInputOnGPUCellNotDeclared(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: returns silently (counts as a pass) when no GPU is available.
    # Scenario: input on the GPU with no device declared for the RNN or cell. Asserts all tensor
    # array operations run on the GPU and none on the CPU.
    if not test.is_gpu_available():
      return  # Test requires access to a GPU

    gpu_dev = test.gpu_device_name()
    run_metadata = self._execute_rnn_on(input_device=gpu_dev)
    cpu_stats, gpu_stats = self._retrieve_cpu_gpu_stats(run_metadata)

    def _assert_in(op_str, in_stats, out_stats):
      # [FUNCTION DOC]
      # HELPER: placement assertion. Inputs: op_str (operation name fragment), in_stats (node
      # stats expected to contain it), out_stats (node stats expected not to). Asserts some node in
      # the first contains the fragment and none in the second does.
      self.assertTrue(any(op_str in s.node_name for s in in_stats))
      self.assertFalse(any(op_str in s.node_name for s in out_stats))

    # Everything happens on GPU
    _assert_in("TensorArray", gpu_stats, cpu_stats)


class RNNCellTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the individual cell implementations and wrappers of the cell implementation
  # module: basic RNN, GRU, basic LSTM, LSTM (with projection), multi-layer cell, residual, dropout
  # and device wrappers. Covers variable naming and trainability, numeric smoke results with
  # constant initializers, state tuple types, checkpointing, configuration serialization and
  # saved-model export. Many tests are graph-mode only.

  @test_util.run_v1_only("b/124229375")
  def testBasicRNNCell(self):
    # [FUNCTION DOC]
    # Scenario: apply a two-unit basic RNN cell to zero input and state with constant weights.
    # Asserts the trainable variable names (weights and bias under the cell scope), that there are
    # no non-trainable variables, and that the output has shape one by two.
    with self.cached_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 2])
        m = array_ops.zeros([1, 2])
        cell = rnn_cell_impl.BasicRNNCell(2)
        g, _ = cell(x, m)
        self.assertEqual([
            "root/basic_rnn_cell/%s:0" % rnn_cell_impl._WEIGHTS_VARIABLE_NAME,
            "root/basic_rnn_cell/%s:0" % rnn_cell_impl._BIAS_VARIABLE_NAME
        ], [v.name for v in cell.trainable_variables])
        self.assertFalse(cell.non_trainable_variables)
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run([g], {
            x: np.array([[1., 1.]]),
            m: np.array([[0.1, 0.1]])
        })
        self.assertEqual(res[0].shape, (1, 2))

  @test_util.run_v1_only("b/124229375")
  def testBasicRNNCellNotTrainable(self):
    # [FUNCTION DOC]
    # Scenario: build a basic RNN cell inside a variable scope whose custom getter forces every
    # variable to be non-trainable. Asserts the cell reports no trainable variables, lists its
    # weights and bias as non-trainable, and produces an output of shape one by two.
    with self.cached_session() as sess:

      def not_trainable_getter(getter, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: custom variable getter for the scenario. Forces the trainable option to false in
        # the keyword arguments and delegates to the original getter, returning its variable.
        kwargs["trainable"] = False
        return getter(*args, **kwargs)

      with variable_scope.variable_scope(
          "root",
          initializer=init_ops.constant_initializer(0.5),
          custom_getter=not_trainable_getter):
        x = array_ops.zeros([1, 2])
        m = array_ops.zeros([1, 2])
        cell = rnn_cell_impl.BasicRNNCell(2)
        g, _ = cell(x, m)
        self.assertFalse(cell.trainable_variables)
        self.assertEqual([
            "root/basic_rnn_cell/%s:0" % rnn_cell_impl._WEIGHTS_VARIABLE_NAME,
            "root/basic_rnn_cell/%s:0" % rnn_cell_impl._BIAS_VARIABLE_NAME
        ], [v.name for v in cell.non_trainable_variables])
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run([g], {
            x: np.array([[1., 1.]]),
            m: np.array([[0.1, 0.1]])
        })
        self.assertEqual(res[0].shape, (1, 2))

  @test_util.run_v1_only("b/124229375")
  def testGRUCell(self):
    # [FUNCTION DOC]
    # Scenario: smoke test of a two-unit GRU cell with constant weights, first with input width
    # equal to units, then (in another scope) with a wider input. Asserts the outputs equal fixed
    # reference numbers within the default tolerance.
    with self.cached_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 2])
        m = array_ops.zeros([1, 2])
        g, _ = rnn_cell_impl.GRUCell(2)(x, m)
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run([g], {
            x: np.array([[1., 1.]]),
            m: np.array([[0.1, 0.1]])
        })
        # Smoke test
        self.assertAllClose(res[0], [[0.175991, 0.175991]])
      with variable_scope.variable_scope(
          "other", initializer=init_ops.constant_initializer(0.5)):
        # Test GRUCell with input_size != num_units.
        x = array_ops.zeros([1, 3])
        m = array_ops.zeros([1, 2])
        g, _ = rnn_cell_impl.GRUCell(2)(x, m)
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run([g], {
            x: np.array([[1., 1., 1.]]),
            m: np.array([[0.1, 0.1]])
        })
        # Smoke test
        self.assertAllClose(res[0], [[0.156736, 0.156736]])

  @test_util.run_v1_only("b/124229375")
  def testBasicLSTMCell(self):
    # [FUNCTION DOC]
    # Scenario: for half and single precision, a two-layer multi-layer cell of basic LSTM cells
    # (non-tuple state) with constant weights. Asserts the initial dtype is unset and later
    # inferred from the input, tracked child cell names, that the configuration can be produced,
    # exact variable names and trainability, result count, and smoke values for output and memory
    # within a loose tolerance. Then checks a separate cell with input width different from units.
    for dtype in [dtypes.float16, dtypes.float32]:
      np_dtype = dtype.as_numpy_dtype
      with self.session(graph=ops.Graph()) as sess:
        with variable_scope.variable_scope(
            "root", initializer=init_ops.constant_initializer(0.5)):
          x = array_ops.zeros([1, 2], dtype=dtype)
          m = array_ops.zeros([1, 8], dtype=dtype)
          cell = rnn_cell_impl.MultiRNNCell(
              [
                  rnn_cell_impl.BasicLSTMCell(2, state_is_tuple=False)
                  for _ in range(2)
              ],
              state_is_tuple=False)
          self.assertEqual(cell.dtype, None)
          self.assertIn("cell-0", cell._trackable_children())
          self.assertIn("cell-1", cell._trackable_children())
          cell.get_config()  # Should not throw an error
          g, out_m = cell(x, m)
          # Layer infers the input type.
          self.assertEqual(cell.dtype, dtype.name)
          expected_variable_names = [
              "root/multi_rnn_cell/cell_0/basic_lstm_cell/%s:0" %
              rnn_cell_impl._WEIGHTS_VARIABLE_NAME,
              "root/multi_rnn_cell/cell_0/basic_lstm_cell/%s:0" %
              rnn_cell_impl._BIAS_VARIABLE_NAME,
              "root/multi_rnn_cell/cell_1/basic_lstm_cell/%s:0" %
              rnn_cell_impl._WEIGHTS_VARIABLE_NAME,
              "root/multi_rnn_cell/cell_1/basic_lstm_cell/%s:0" %
              rnn_cell_impl._BIAS_VARIABLE_NAME
          ]
          self.assertEqual(expected_variable_names,
                           [v.name for v in cell.trainable_variables])
          self.assertFalse(cell.non_trainable_variables)
          sess.run([variables_lib.global_variables_initializer()])
          res = sess.run([g, out_m], {
              x: np.array([[1., 1.]]),
              m: 0.1 * np.ones([1, 8])
          })
          self.assertEqual(len(res), 2)
          variables = variables_lib.global_variables()
          self.assertEqual(expected_variable_names, [v.name for v in variables])
          # The numbers in results were not calculated, this is just a
          # smoke test.
          self.assertAllClose(res[0], np.array(
              [[0.240, 0.240]], dtype=np_dtype), 1e-2)
          expected_mem = np.array(
              [[0.689, 0.689, 0.448, 0.448, 0.398, 0.398, 0.240, 0.240]],
              dtype=np_dtype)
          self.assertAllClose(res[1], expected_mem, 1e-2)
        with variable_scope.variable_scope(
            "other", initializer=init_ops.constant_initializer(0.5)):
          # Test BasicLSTMCell with input_size != num_units.
          x = array_ops.zeros([1, 3], dtype=dtype)
          m = array_ops.zeros([1, 4], dtype=dtype)
          g, out_m = rnn_cell_impl.BasicLSTMCell(2, state_is_tuple=False)(x, m)
          sess.run([variables_lib.global_variables_initializer()])
          res = sess.run(
              [g, out_m], {
                  x: np.array([[1., 1., 1.]], dtype=np_dtype),
                  m: 0.1 * np.ones([1, 4], dtype=np_dtype)
              })
          self.assertEqual(len(res), 2)

  @test_util.run_v1_only("b/124229375")
  def testBasicLSTMCellDimension0Error(self):
    # [FUNCTION DOC]
    # Scenario: a basic LSTM cell is applied to an input with batch size three and a state with
    # batch size two. Expectation: a value error is raised.
    """Tests that dimension 0 in both(x and m) shape must be equal."""
    with self.cached_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        num_units = 2
        state_size = num_units * 2
        batch_size = 3
        input_size = 4
        x = array_ops.zeros([batch_size, input_size])
        m = array_ops.zeros([batch_size - 1, state_size])
        with self.assertRaises(ValueError):
          g, out_m = rnn_cell_impl.BasicLSTMCell(
              num_units, state_is_tuple=False)(x, m)
          sess.run([variables_lib.global_variables_initializer()])
          sess.run(
              [g, out_m], {
                  x: 1 * np.ones([batch_size, input_size]),
                  m: 0.1 * np.ones([batch_size - 1, state_size])
              })

  def testBasicLSTMCellStateSizeError(self):
    # [FUNCTION DOC]
    # Scenario: a basic LSTM cell (non-tuple state) is given a state whose width is three times
    # the units instead of two times. Expectation: a value error or an invalid-argument error.
    """Tests that state_size must be num_units * 2."""
    with self.cached_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        num_units = 2
        state_size = num_units * 3  # state_size must be num_units * 2
        batch_size = 3
        input_size = 4
        x = array_ops.zeros([batch_size, input_size])
        m = array_ops.zeros([batch_size, state_size])
        with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
          g, out_m = rnn_cell_impl.BasicLSTMCell(
              num_units, state_is_tuple=False)(x, m)
          sess.run([variables_lib.global_variables_initializer()])
          sess.run(
              [g, out_m], {
                  x: 1 * np.ones([batch_size, input_size]),
                  m: 0.1 * np.ones([batch_size, state_size])
              })

  @test_util.run_v1_only("b/124229375")
  def testBasicLSTMCellStateTupleType(self):
    # [FUNCTION DOC]
    # Scenario: a two-layer multi-layer cell of tuple-state basic LSTM cells. Asserts the state
    # size is a tuple of LSTM state tuples, that passing plain tuples as state yields LSTM state
    # tuples as the new states, and the same when passing the cell's zero state.
    with self.cached_session():
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 2])
        m0 = (array_ops.zeros([1, 2]),) * 2
        m1 = (array_ops.zeros([1, 2]),) * 2
        cell = rnn_cell_impl.MultiRNNCell(
            [rnn_cell_impl.BasicLSTMCell(2) for _ in range(2)],
            state_is_tuple=True)
        self.assertTrue(isinstance(cell.state_size, tuple))
        self.assertTrue(
            isinstance(cell.state_size[0], rnn_cell_impl.LSTMStateTuple))
        self.assertTrue(
            isinstance(cell.state_size[1], rnn_cell_impl.LSTMStateTuple))

        # Pass in regular tuples
        _, (out_m0, out_m1) = cell(x, (m0, m1))
        self.assertTrue(isinstance(out_m0, rnn_cell_impl.LSTMStateTuple))
        self.assertTrue(isinstance(out_m1, rnn_cell_impl.LSTMStateTuple))

        # Pass in LSTMStateTuples
        variable_scope.get_variable_scope().reuse_variables()
        zero_state = cell.zero_state(1, dtypes.float32)
        self.assertTrue(isinstance(zero_state, tuple))
        self.assertTrue(isinstance(zero_state[0], rnn_cell_impl.LSTMStateTuple))
        self.assertTrue(isinstance(zero_state[1], rnn_cell_impl.LSTMStateTuple))
        _, (out_m0, out_m1) = cell(x, zero_state)
        self.assertTrue(isinstance(out_m0, rnn_cell_impl.LSTMStateTuple))
        self.assertTrue(isinstance(out_m1, rnn_cell_impl.LSTMStateTuple))

  @test_util.run_v1_only("b/124229375")
  def testBasicLSTMCellWithStateTuple(self):
    # [FUNCTION DOC]
    # Scenario: a multi-layer cell with tuple state over non-tuple basic LSTM cells, called with
    # two state tensors. Asserts three results and that output and the two new memories equal
    # fixed reference values (matching the non-tuple variant).
    with self.cached_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 2])
        m0 = array_ops.zeros([1, 4])
        m1 = array_ops.zeros([1, 4])
        cell = rnn_cell_impl.MultiRNNCell(
            [
                rnn_cell_impl.BasicLSTMCell(2, state_is_tuple=False)
                for _ in range(2)
            ],
            state_is_tuple=True)
        g, (out_m0, out_m1) = cell(x, (m0, m1))
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run(
            [g, out_m0, out_m1], {
                x: np.array([[1., 1.]]),
                m0: 0.1 * np.ones([1, 4]),
                m1: 0.1 * np.ones([1, 4])
            })
        self.assertEqual(len(res), 3)
        # The numbers in results were not calculated, this is just a smoke test.
        # Note, however, these values should match the original
        # version having state_is_tuple=False.
        self.assertAllClose(res[0], [[0.24024698, 0.24024698]])
        expected_mem0 = np.array(
            [[0.68967271, 0.68967271, 0.44848421, 0.44848421]])
        expected_mem1 = np.array(
            [[0.39897051, 0.39897051, 0.24024698, 0.24024698]])
        self.assertAllClose(res[1], expected_mem0)
        self.assertAllClose(res[2], expected_mem1)

  @test_util.run_v1_only("b/124229375")
  def testLSTMCell(self):
    # [FUNCTION DOC]
    # Scenario: LSTM cell with eight units, projection to six and non-tuple state, batch of three
    # distinct inputs. Asserts output shape batch by projection, state shape batch by units plus
    # projection, and that the outputs and states of examples differ from the first example's.
    with self.cached_session() as sess:
      num_units = 8
      num_proj = 6
      state_size = num_units + num_proj
      batch_size = 3
      input_size = 2
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([batch_size, input_size])
        m = array_ops.zeros([batch_size, state_size])
        cell = rnn_cell_impl.LSTMCell(
            num_units=num_units,
            num_proj=num_proj,
            forget_bias=1.0,
            state_is_tuple=False)
        output, state = cell(x, m)
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run(
            [output, state], {
                x: np.array([[1., 1.], [2., 2.], [3., 3.]]),
                m: 0.1 * np.ones((batch_size, state_size))
            })
        self.assertEqual(len(res), 2)
        # The numbers in results were not calculated, this is mostly just a
        # smoke test.
        self.assertEqual(res[0].shape, (batch_size, num_proj))
        self.assertEqual(res[1].shape, (batch_size, state_size))
        # Different inputs so different outputs and states
        for i in range(1, batch_size):
          self.assertTrue(
              float(np.linalg.norm((res[0][0, :] - res[0][i, :]))) > 1e-6)
          self.assertTrue(
              float(np.linalg.norm((res[1][0, :] - res[1][i, :]))) > 1e-6)

  @test_util.run_v1_only("b/124229375")
  def testLSTMCellVariables(self):
    # [FUNCTION DOC]
    # Scenario: build an LSTM cell with projection by calling it once. Asserts the global variable
    # names in creation order are the cell kernel, the bias and the projection kernel.
    with self.cached_session():
      num_units = 8
      num_proj = 6
      state_size = num_units + num_proj
      batch_size = 3
      input_size = 2
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([batch_size, input_size])
        m = array_ops.zeros([batch_size, state_size])
        cell = rnn_cell_impl.LSTMCell(
            num_units=num_units,
            num_proj=num_proj,
            forget_bias=1.0,
            state_is_tuple=False)
        cell(x, m)  # Execute to create variables
      variables = variables_lib.global_variables()
      self.assertEqual(variables[0].op.name, "root/lstm_cell/kernel")
      self.assertEqual(variables[1].op.name, "root/lstm_cell/bias")
      self.assertEqual(variables[2].op.name, "root/lstm_cell/projection/kernel")

  @test_util.run_in_graph_and_eager_modes
  def testWrapperCheckpointing(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager modes): for each of the dropout wrapper, residual wrapper and a
    # multi-layer wrapper around a one-unit basic RNN cell, run it once, set the bias to forty,
    # save a checkpoint to a temporary directory, change the bias to zero, restore, and assert all
    # checkpointed values were consumed and the bias is forty again. Writes a checkpoint file.
    for wrapper_type in [
        rnn_cell_impl.DropoutWrapper,
        rnn_cell_impl.ResidualWrapper,
        lambda cell: rnn_cell_impl.MultiRNNCell([cell])]:
      cell = rnn_cell_impl.BasicRNNCell(1)
      wrapper = wrapper_type(cell)
      wrapper(array_ops.ones([1, 1]),
              state=wrapper.zero_state(batch_size=1, dtype=dtypes.float32))
      self.evaluate([v.initializer for v in cell.variables])
      checkpoint = trackable_utils.Checkpoint(wrapper=wrapper)
      prefix = os.path.join(self.get_temp_dir(), "ckpt")
      self.evaluate(cell._bias.assign([40.]))
      save_path = checkpoint.save(prefix)
      self.evaluate(cell._bias.assign([0.]))
      checkpoint.restore(save_path).assert_consumed().run_restore_ops()
      self.assertAllEqual([40.], self.evaluate(cell._bias))

  @test_util.run_in_graph_and_eager_modes
  def testResidualWrapper(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager modes): wrap a three-unit GRU cell with the residual wrapper.
    # Asserts configuration works, the wrapped cell is a tracked child, the wrapper output equals
    # the cell output plus the input, and the new state is unchanged by the wrapper.
    wrapper_type = rnn_cell_impl.ResidualWrapper
    x = ops.convert_to_tensor(np.array([[1., 1., 1.]]))
    m = ops.convert_to_tensor(np.array([[0.1, 0.1, 0.1]]))
    base_cell = rnn_cell_impl.GRUCell(
        3, kernel_initializer=init_ops.constant_initializer(0.5),
        bias_initializer=init_ops.constant_initializer(0.5))
    g, m_new = base_cell(x, m)
    wrapper_object = wrapper_type(base_cell)
    wrapper_object.get_config()  # Should not throw an error

    self.assertIn("cell", wrapper_object._trackable_children())
    self.assertIs(wrapper_object._trackable_children()["cell"], base_cell)

    g_res, m_new_res = wrapper_object(x, m)
    self.evaluate([variables_lib.global_variables_initializer()])
    res = self.evaluate([g, g_res, m_new, m_new_res])
    # Residual connections
    self.assertAllClose(res[1], res[0] + [1., 1., 1.])
    # States are left untouched
    self.assertAllClose(res[2], res[3])

  @test_util.run_in_graph_and_eager_modes
  def testResidualWrapperWithSlice(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager modes): residual wrapper with a custom residual function that
    # slices the five-wide input down to the cell's three units before adding. Asserts the output
    # equals the cell output plus the sliced input and states are unchanged.
    wrapper_type = rnn_cell_impl.ResidualWrapper
    x = ops.convert_to_tensor(np.array([[1., 1., 1., 1., 1.]]))
    m = ops.convert_to_tensor(np.array([[0.1, 0.1, 0.1]]))
    base_cell = rnn_cell_impl.GRUCell(
        3, kernel_initializer=init_ops.constant_initializer(0.5),
        bias_initializer=init_ops.constant_initializer(0.5))
    g, m_new = base_cell(x, m)

    def residual_with_slice_fn(inp, out):
      # [FUNCTION DOC]
      # HELPER: custom residual function. Inputs: inp (wrapper input tensor) and out (cell output
      # tensor). Takes the first three features of inp for all rows and returns them added to out.
      inp_sliced = array_ops.slice(inp, [0, 0], [-1, 3])
      return inp_sliced + out

    g_res, m_new_res = wrapper_type(
        base_cell, residual_with_slice_fn)(x, m)
    self.evaluate([variables_lib.global_variables_initializer()])
    res_g, res_g_res, res_m_new, res_m_new_res = self.evaluate(
        [g, g_res, m_new, m_new_res])
    # Residual connections
    self.assertAllClose(res_g_res, res_g + [1., 1., 1.])
    # States are left untouched
    self.assertAllClose(res_m_new, res_m_new_res)

  def testDeviceWrapper(self):
    # [FUNCTION DOC]
    # Scenario: wrap a GRU cell in the device wrapper pinned to the first CPU. Asserts the config
    # can be produced, the wrapped cell is a tracked child, and the output tensor's device name
    # mentions cpu:0.
    wrapper_type = rnn_cell_impl.DeviceWrapper
    x = array_ops.zeros([1, 3])
    m = array_ops.zeros([1, 3])
    cell = rnn_cell_impl.GRUCell(3)
    wrapped_cell = wrapper_type(cell, "/cpu:0")
    wrapped_cell.get_config()  # Should not throw an error
    self.assertEqual(wrapped_cell._trackable_children()["cell"], cell)

    outputs, _ = wrapped_cell(x, m)
    self.assertIn("cpu:0", outputs.device.lower())

  def _retrieve_cpu_gpu_stats(self, run_metadata):
    # [FUNCTION DOC]
    # HELPER: extracts per-device node statistics from run metadata (same logic as the helper in
    # the tensor array placement test case).
    # Input: run_metadata, trace metadata of a session run. Returns a pair (CPU node stats, GPU
    # node stats), each nothing if that device does not appear.
    cpu_stats = None
    gpu_stats = None
    step_stats = run_metadata.step_stats
    for ds in step_stats.dev_stats:
      if "cpu:0" in ds.device[-5:].lower():
        cpu_stats = ds.node_stats
      if "gpu:0" == ds.device[-5:].lower():
        gpu_stats = ds.node_stats
    return cpu_stats, gpu_stats

  @test_util.run_v1_only("b/124229375")
  def testDeviceWrapperDynamicExecutionNodesAreAllProperlyLocated(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: returns silently (counts as a pass) when no GPU is available.
    # Scenario: dynamic RNN built under a CPU scope with a GRU cell wrapped for the GPU, run with
    # full tracing. Asserts no GRU cell nodes run on the CPU and some run on the GPU.
    if not test.is_gpu_available():
      # Can't perform this test w/o a GPU
      return

    gpu_dev = test.gpu_device_name()
    with self.session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 1, 3])
        cell = rnn_cell_impl.DeviceWrapper(rnn_cell_impl.GRUCell(3), gpu_dev)
        with ops.device("/cpu:0"):
          outputs, _ = rnn.dynamic_rnn(
              cell=cell, inputs=x, dtype=dtypes.float32)
        run_metadata = config_pb2.RunMetadata()
        opts = config_pb2.RunOptions(
            trace_level=config_pb2.RunOptions.FULL_TRACE)

        sess.run([variables_lib.global_variables_initializer()])
        _ = sess.run(outputs, options=opts, run_metadata=run_metadata)

      cpu_stats, gpu_stats = self._retrieve_cpu_gpu_stats(run_metadata)
      self.assertFalse([s for s in cpu_stats if "gru_cell" in s.node_name])
      self.assertTrue([s for s in gpu_stats if "gru_cell" in s.node_name])

  @test_util.run_v1_only("b/124229375")
  def testMultiRNNCell(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the last assertion checks the truth of a non-empty list of booleans, which is
    # always true, so the weight dtype check is vacuous.
    # Scenario: two stacked two-unit GRU cells with a concatenated (non-tuple) state. Asserts the
    # new state equals fixed reference values and the cell has eight weights.
    with self.cached_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 2])
        m = array_ops.zeros([1, 4])
        multi_rnn_cell = rnn_cell_impl.MultiRNNCell(
            [rnn_cell_impl.GRUCell(2) for _ in range(2)],
            state_is_tuple=False)
        _, ml = multi_rnn_cell(x, m)
        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run(ml, {
            x: np.array([[1., 1.]]),
            m: np.array([[0.1, 0.1, 0.1, 0.1]])
        })
        # The numbers in results were not calculated, this is just a smoke test.
        self.assertAllClose(res, [[0.175991, 0.175991, 0.13248, 0.13248]])
        self.assertEqual(len(multi_rnn_cell.weights), 2 * 4)
        self.assertTrue(
            [x.dtype == dtypes.float32 for x in multi_rnn_cell.weights])

  @test_util.run_v1_only("b/124229375")
  def testMultiRNNCellWithStateTuple(self):
    # [FUNCTION DOC]
    # Scenario: stacked GRU cells with tuple state. Asserts passing a single concatenated state
    # raises a value error saying a tuple was expected, and that passing a proper tuple of states
    # gives results equal to those of the non-tuple variant's reference values.
    with self.cached_session() as sess:
      with variable_scope.variable_scope(
          "root", initializer=init_ops.constant_initializer(0.5)):
        x = array_ops.zeros([1, 2])
        m_bad = array_ops.zeros([1, 4])
        m_good = (array_ops.zeros([1, 2]), array_ops.zeros([1, 2]))

        # Test incorrectness of state
        with self.assertRaisesRegex(ValueError, "Expected state .* a tuple"):
          rnn_cell_impl.MultiRNNCell(
              [rnn_cell_impl.GRUCell(2) for _ in range(2)],
              state_is_tuple=True)(x, m_bad)

        _, ml = rnn_cell_impl.MultiRNNCell(
            [rnn_cell_impl.GRUCell(2) for _ in range(2)],
            state_is_tuple=True)(x, m_good)

        sess.run([variables_lib.global_variables_initializer()])
        res = sess.run(
            ml, {
                x: np.array([[1., 1.]]),
                m_good[0]: np.array([[0.1, 0.1]]),
                m_good[1]: np.array([[0.1, 0.1]])
            })

        # The numbers in results were not calculated, this is just a
        # smoke test.  However, these numbers should match those of
        # the test testMultiRNNCell.
        self.assertAllClose(res[0], [[0.175991, 0.175991]])
        self.assertAllClose(res[1], [[0.13248, 0.13248]])

  def testDeviceWrapperSerialization(self):
    # [FUNCTION DOC]
    # Scenario: get the configuration of a device wrapper around an LSTM cell, replace the stored
    # cell entry with a real cell rebuilt from its config (to avoid a reverse dependency), and
    # rebuild the wrapper from the configuration. Asserts the rebuilt wrapper has an equal
    # configuration and is the same wrapper type.
    wrapper_cls = rnn_cell_impl.DeviceWrapper
    cell = rnn_cell_impl.LSTMCell(10)
    wrapper = wrapper_cls(cell, "/cpu:0")
    config = wrapper.get_config()

    # Replace the cell in the config with real cell instance to work around the
    # reverse keras dependency issue.
    config_copy = config.copy()
    config_copy["cell"] = rnn_cell_impl.LSTMCell.from_config(
        config_copy["cell"]["config"])
    reconstructed_wrapper = wrapper_cls.from_config(config_copy)
    self.assertDictEqual(config, reconstructed_wrapper.get_config())
    self.assertIsInstance(reconstructed_wrapper, wrapper_cls)

  def testResidualWrapperSerialization(self):
    # [FUNCTION DOC]
    # Scenario: round-trip the residual wrapper through its configuration three times: with the
    # default residual function (config equality and type checked), with an anonymous function
    # (the rebuilt function maps inputs one and two to four), and with a named function (rebuilt
    # function maps one and two to five). The cell entry in each config is replaced with a real
    # rebuilt cell before reconstruction.
    wrapper_cls = rnn_cell_impl.ResidualWrapper
    cell = rnn_cell_impl.LSTMCell(10)
    wrapper = wrapper_cls(cell)
    config = wrapper.get_config()

    # Replace the cell in the config with real cell instance to work around the
    # reverse keras dependency issue.
    config_copy = config.copy()
    config_copy["cell"] = rnn_cell_impl.LSTMCell.from_config(
        config_copy["cell"]["config"])
    reconstructed_wrapper = wrapper_cls.from_config(config_copy)
    self.assertDictEqual(config, reconstructed_wrapper.get_config())
    self.assertIsInstance(reconstructed_wrapper, wrapper_cls)

    wrapper = wrapper_cls(cell, residual_fn=lambda i, o: i + i + o)
    config = wrapper.get_config()

    config_copy = config.copy()
    config_copy["cell"] = rnn_cell_impl.LSTMCell.from_config(
        config_copy["cell"]["config"])
    reconstructed_wrapper = wrapper_cls.from_config(config_copy)
    # Assert the reconstructed function will perform the math correctly.
    self.assertEqual(reconstructed_wrapper._residual_fn(1, 2), 4)

    def residual_fn(inputs, outputs):
      # [FUNCTION DOC]
      # HELPER: named residual function for the serialization scenario. Returns three times the
      # inputs plus the outputs.
      return inputs * 3 + outputs

    wrapper = wrapper_cls(cell, residual_fn=residual_fn)
    config = wrapper.get_config()

    config_copy = config.copy()
    config_copy["cell"] = rnn_cell_impl.LSTMCell.from_config(
        config_copy["cell"]["config"])
    reconstructed_wrapper = wrapper_cls.from_config(config_copy)
    # Assert the reconstructed function will perform the math correctly.
    self.assertEqual(reconstructed_wrapper._residual_fn(1, 2), 5)

  def testDropoutWrapperSerialization(self):
    # [FUNCTION DOC]
    # Scenario: round-trip the dropout wrapper through its configuration three times: with defaults
    # (config equality and type checked), with an anonymous state-filter function that returns
    # true (rebuilt filter returns true), and with a named filter returning false (rebuilt filter
    # returns false). The cell entry in each config is replaced with a real rebuilt cell.
    wrapper_cls = rnn_cell_impl.DropoutWrapper
    cell = rnn_cell_impl.LSTMCell(10)
    wrapper = wrapper_cls(cell)
    config = wrapper.get_config()

    config_copy = config.copy()
    config_copy["cell"] = rnn_cell_impl.LSTMCell.from_config(
        config_copy["cell"]["config"])
    reconstructed_wrapper = wrapper_cls.from_config(config_copy)
    self.assertDictEqual(config, reconstructed_wrapper.get_config())
    self.assertIsInstance(reconstructed_wrapper, wrapper_cls)

    wrapper = wrapper_cls(cell, dropout_state_filter_visitor=lambda s: True)
    config = wrapper.get_config()

    config_copy = config.copy()
    config_copy["cell"] = rnn_cell_impl.LSTMCell.from_config(
        config_copy["cell"]["config"])
    reconstructed_wrapper = wrapper_cls.from_config(config_copy)
    self.assertTrue(reconstructed_wrapper._dropout_state_filter(None))

    def dropout_state_filter_visitor(unused_state):
      # [FUNCTION DOC]
      # HELPER: named state filter for the serialization scenario. Ignores its argument and
      # returns false (never drop out any state part).
      return False

    wrapper = wrapper_cls(
        cell, dropout_state_filter_visitor=dropout_state_filter_visitor)
    config = wrapper.get_config()

    config_copy = config.copy()
    config_copy["cell"] = rnn_cell_impl.LSTMCell.from_config(
        config_copy["cell"]["config"])
    reconstructed_wrapper = wrapper_cls.from_config(config_copy)
    self.assertFalse(reconstructed_wrapper._dropout_state_filter(None))

  def testSavedModel(self):
    # [FUNCTION DOC]
    # Scenario: export and reload an LSTM cell step function. Skipped on GPU machines (known
    # bug). Builds a trackable root holding an LSTM cell and a compiled function with a fixed
    # input signature that applies the cell to a zero state, saves it to a temporary directory,
    # loads it back, and asserts the loaded function output equals the original on zero input.
    # Side effect: writes a saved model to the test temp directory.
    if test_util.is_gpu_available():
      self.skipTest("b/175887901")

    with self.cached_session():
      root = autotrackable.AutoTrackable()
      root.cell = rnn_cell_impl.LSTMCell(8)
      @def_function.function(input_signature=[tensor.TensorSpec([3, 8])])
      def call(x):
        # [FUNCTION DOC]
        # HELPER: compiled step function that is saved. Input x is a three by eight float tensor.
        # Creates the cell's zero state for batch three with x's dtype, applies the cell once, and
        # returns only the output (the new state is discarded).
        state = root.cell.zero_state(3, dtype=x.dtype)
        y, _ = root.cell(x, state)
        return y
      root.call = call
      expected = root.call(array_ops.zeros((3, 8)))
      self.evaluate(variables_lib.global_variables_initializer())

      save_dir = os.path.join(self.get_temp_dir(), "saved_model")
      save.save(root, save_dir)
      loaded = load.load(save_dir)
      self.evaluate(variables_lib.global_variables_initializer())
      self.assertAllClose(
          expected, loaded.call(array_ops.zeros((3, 8))))


@test_util.run_all_in_graph_and_eager_modes
@test_util.run_all_without_tensor_float_32(
    "Uses an LSTMCell, which calls matmul")
class DropoutWrapperTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the dropout wrapper cell in graph and eager modes (class decorators run every
  # test in both and disable reduced-precision matmul). Uses a shared helper that runs a dynamic
  # RNN with a dropout-wrapped LSTM and checks properties such as keep-everything equivalence,
  # dropping outputs, inputs or state, seeding, and variational (time-consistent) masks.

  def _testDropoutWrapper(self,
                          batch_size=None,
                          time_steps=None,
                          parallel_iterations=None,
                          wrapper_type=None,
                          scope="root",
                          **kwargs):
    # [FUNCTION DOC]
    # HELPER: shared runner for the dropout wrapper tests.
    # Inputs: batch_size and time_steps (integers; if both are nothing a fixed two-step, batch-one
    # example is used, otherwise random normal time-major inputs of width three); parallel_iterations
    # (optional integer for the dynamic RNN); wrapper_type (the wrapper class to apply); scope
    # (variable scope name, default root); extra keyword arguments (keep probabilities, seed,
    # variational option, input size, and so on) forwarded to the wrapper along with the input
    # dtype. Wraps a three-unit LSTM with constant weights, runs a time-major dynamic RNN from a
    # constant initial state, and initializes variables. Asserts output shape (time, batch, 3) and
    # that the final cell and hidden states have shape (batch, 3). Returns a list of the evaluated
    # outputs and the final state tuple.
    if batch_size is None and time_steps is None:
      # 2 time steps, batch size 1, depth 3
      batch_size = 1
      time_steps = 2
      x = constant_op.constant(
          [[[2., 2., 2.]], [[1., 1., 1.]]], dtype=dtypes.float32)
      m = rnn_cell_impl.LSTMStateTuple(
          *[constant_op.constant([[0.1, 0.1, 0.1]], dtype=dtypes.float32)] * 2)
    else:
      x = constant_op.constant(
          np.random.randn(time_steps, batch_size, 3).astype(np.float32))
      m = rnn_cell_impl.LSTMStateTuple(*[
          constant_op.
          constant([[0.1, 0.1, 0.1]] * batch_size, dtype=dtypes.float32)] * 2)
    outputs, final_state = rnn.dynamic_rnn(
        cell=wrapper_type(
            rnn_cell_impl.LSTMCell(
                3, initializer=init_ops.constant_initializer(0.5)),
            dtype=x.dtype, **kwargs),
        time_major=True,
        parallel_iterations=parallel_iterations,
        inputs=x,
        initial_state=m,
        scope=scope)
    self.evaluate([variables_lib.global_variables_initializer()])
    res = self.evaluate([outputs, final_state])
    self.assertEqual(res[0].shape, (time_steps, batch_size, 3))
    self.assertEqual(res[1].c.shape, (batch_size, 3))
    self.assertEqual(res[1].h.shape, (batch_size, 3))
    return res

  def testDropoutWrapperProperties(self):
    # [FUNCTION DOC]
    # Scenario (regression for a reported issue): wrap a ten-unit basic RNN cell. Asserts the
    # wrapper exposes the wrapped cell and reports state and output sizes of ten.
    wrapper_type = rnn_cell_impl.DropoutWrapper
    cell = rnn_cell_impl.BasicRNNCell(10)
    wrapper = wrapper_type(cell)
    # Github issue 15810
    self.assertEqual(wrapper.wrapped_cell, cell)
    self.assertEqual(wrapper.state_size, 10)
    self.assertEqual(wrapper.output_size, 10)

  def testDropoutWrapperZeroState(self):
    # [FUNCTION DOC]
    # Scenario: wrap a custom cell whose zero-state method returns a marker string. Asserts the
    # wrapper's own zero-state method returns that marker, i.e. it delegates to the wrapped cell.
    wrapper_type = rnn_cell_impl.DropoutWrapper

    class _Cell(rnn_cell_impl.BasicRNNCell):
      # [CLASS DOC]
      # HELPER: test-local ten-unit basic RNN cell whose zero-state method is overridden to
      # return a fixed marker string, used to check delegation by the dropout wrapper.

      def zero_state(self, batch_size=None, dtype=None):
        # [FUNCTION DOC]
        # HELPER: overridden zero-state method. Ignores the optional batch size and dtype and
        # returns the fixed marker string.
        return "wrapped_cell_zero_state"
    wrapper = wrapper_type(_Cell(10))
    self.assertEqual(wrapper.zero_state(10, dtypes.float32),
                     "wrapped_cell_zero_state")

  def testDropoutWrapperKeepAllConstantInput(self):
    # [FUNCTION DOC]
    # Scenario: dropout wrapper with all keep probabilities given as a constant tensor equal to
    # one. Asserts outputs and final states equal the known no-dropout reference values.
    wrapper_type = rnn_cell_impl.DropoutWrapper
    keep = array_ops.ones([])
    res = self._testDropoutWrapper(
        input_keep_prob=keep, output_keep_prob=keep, state_keep_prob=keep,
        wrapper_type=wrapper_type)
    true_full_output = np.array(
        [[[0.751109, 0.751109, 0.751109]], [[0.895509, 0.895509, 0.895509]]],
        dtype=np.float32)
    true_full_final_c = np.array(
        [[1.949385, 1.949385, 1.949385]], dtype=np.float32)
    self.assertAllClose(true_full_output, res[0])
    self.assertAllClose(true_full_output[1], res[1].h)
    self.assertAllClose(true_full_final_c, res[1].c)

  def testDropoutWrapperKeepAll(self):
    # [FUNCTION DOC]
    # Scenario: dropout wrapper with all keep probabilities given by a variable equal to one.
    # Asserts outputs and final states equal the known no-dropout reference values.
    wrapper_type = rnn_cell_impl.DropoutWrapper
    keep = variable_scope.get_variable("all", initializer=1.0)
    res = self._testDropoutWrapper(
        input_keep_prob=keep, output_keep_prob=keep, state_keep_prob=keep,
        wrapper_type=wrapper_type)
    true_full_output = np.array(
        [[[0.751109, 0.751109, 0.751109]], [[0.895509, 0.895509, 0.895509]]],
        dtype=np.float32)
    true_full_final_c = np.array(
        [[1.949385, 1.949385, 1.949385]], dtype=np.float32)
    self.assertAllClose(true_full_output, res[0])
    self.assertAllClose(true_full_output[1], res[1].h)
    self.assertAllClose(true_full_final_c, res[1].c)

  def testDropoutWrapperWithSeed(self):
    # [FUNCTION DOC]
    # Scenario: run the helper twice with keep probability one half, the same global and wrapper
    # seeds and one parallel iteration, in different scopes. Asserts outputs and final cell and
    # hidden states are identical between the two runs. Side effect: sets the global random seed.
    wrapper_type = rnn_cell_impl.DropoutWrapper
    keep_some = 0.5
    random_seed.set_random_seed(2)
    ## Use parallel_iterations = 1 in both calls to
    ## _testDropoutWrapper to ensure the (per-time step) dropout is
    ## consistent across both calls.  Otherwise the seed may not end
    ## up being munged consistently across both graphs.
    res_standard_1 = self._testDropoutWrapper(
        input_keep_prob=keep_some,
        output_keep_prob=keep_some,
        state_keep_prob=keep_some,
        seed=10,
        parallel_iterations=1,
        wrapper_type=wrapper_type,
        scope="root_1")
    random_seed.set_random_seed(2)
    res_standard_2 = self._testDropoutWrapper(
        input_keep_prob=keep_some,
        output_keep_prob=keep_some,
        state_keep_prob=keep_some,
        seed=10,
        parallel_iterations=1,
        wrapper_type=wrapper_type,
        scope="root_2")
    self.assertAllClose(res_standard_1[0], res_standard_2[0])
    self.assertAllClose(res_standard_1[1].c, res_standard_2[1].c)
    self.assertAllClose(res_standard_1[1].h, res_standard_2[1].h)

  def testDropoutWrapperKeepNoOutput(self):
    # [FUNCTION DOC]
    # Scenario: output keep probability almost zero, input and state keep one. Asserts all
    # outputs are zero while the final hidden and cell states match the no-dropout reference.
    wrapper_type = rnn_cell_impl.DropoutWrapper
    keep_all = variable_scope.get_variable("all", initializer=1.0)
    keep_none = variable_scope.get_variable("none", initializer=1e-6)
    res = self._testDropoutWrapper(
        input_keep_prob=keep_all,
        output_keep_prob=keep_none,
        state_keep_prob=keep_all,
        wrapper_type=wrapper_type)
    true_full_output = np.array(
        [[[0.751109, 0.751109, 0.751109]], [[0.895509, 0.895509, 0.895509]]],
        dtype=np.float32)
    true_full_final_c = np.array(
        [[1.949385, 1.949385, 1.949385]], dtype=np.float32)
    self.assertAllClose(np.zeros(res[0].shape), res[0])
    self.assertAllClose(true_full_output[1], res[1].h)
    self.assertAllClose(true_full_final_c, res[1].c)

  def testDropoutWrapperKeepNoStateExceptLSTMCellMemory(self):
    # [FUNCTION DOC]
    # Scenario: state keep probability almost zero, input and output keep one. Asserts the first
    # output is unchanged, the second output differs (it sees the zeroed state), the final hidden
    # state is zero, and the final cell (memory) state is a reference value because the wrapper
    # never drops the memory part of an LSTM state tuple by default.
    wrapper_type = rnn_cell_impl.DropoutWrapper
    keep_all = variable_scope.get_variable("all", initializer=1.0)
    keep_none = variable_scope.get_variable("none", initializer=1e-6)
    # Even though we dropout state, by default DropoutWrapper never
    # drops out the memory ("c") term of an LSTMStateTuple.
    res = self._testDropoutWrapper(
        input_keep_prob=keep_all,
        output_keep_prob=keep_all,
        state_keep_prob=keep_none,
        wrapper_type=wrapper_type)
    true_c_state = np.array([[1.713925, 1.713925, 1.713925]], dtype=np.float32)
    true_full_output = np.array(
        [[[0.751109, 0.751109, 0.751109]], [[0.895509, 0.895509, 0.895509]]],
        dtype=np.float32)
    self.assertAllClose(true_full_output[0], res[0][0])
    # Second output is modified by zero input state
    self.assertGreater(np.linalg.norm(true_full_output[1] - res[0][1]), 1e-4)
    # h state has been set to zero
    self.assertAllClose(np.zeros(res[1].h.shape), res[1].h)
    # c state of an LSTMStateTuple is NEVER modified.
    self.assertAllClose(true_c_state, res[1].c)

  def testDropoutWrapperKeepNoInput(self):
    # [FUNCTION DOC]
    # Scenario: input keep probability almost zero, output and state keep one. Asserts the
    # outputs, final hidden state and final cell state all differ noticeably from the no-dropout
    # reference values because inputs were zeroed.
    wrapper_type = rnn_cell_impl.DropoutWrapper
    keep_all = variable_scope.get_variable("all", initializer=1.0)
    keep_none = variable_scope.get_variable("none", initializer=1e-6)
    true_full_output = np.array(
        [[[0.751109, 0.751109, 0.751109]], [[0.895509, 0.895509, 0.895509]]],
        dtype=np.float32)
    true_full_final_c = np.array(
        [[1.949385, 1.949385, 1.949385]], dtype=np.float32)
    # All outputs are different because inputs are zeroed out
    res = self._testDropoutWrapper(
        input_keep_prob=keep_none,
        output_keep_prob=keep_all,
        state_keep_prob=keep_all,
        wrapper_type=wrapper_type)
    self.assertGreater(np.linalg.norm(res[0] - true_full_output), 1e-4)
    self.assertGreater(np.linalg.norm(res[1].h - true_full_output[1]), 1e-4)
    self.assertGreater(np.linalg.norm(res[1].c - true_full_final_c), 1e-4)

  def testDropoutWrapperRecurrentOutput(self):
    # [FUNCTION DOC]
    # Scenario: variational (time-consistent) dropout with output keep probability 0.8, batch five
    # and seven time steps. Asserts the pattern of zeroed output entries is the same at every time
    # step.
    wrapper_type = rnn_cell_impl.DropoutWrapper
    keep_some = 0.8
    keep_all = variable_scope.get_variable("all", initializer=1.0)
    res = self._testDropoutWrapper(
        input_keep_prob=keep_all,
        output_keep_prob=keep_some,
        state_keep_prob=keep_all,
        variational_recurrent=True,
        wrapper_type=wrapper_type,
        input_size=3,
        batch_size=5,
        time_steps=7)
    # Ensure the same dropout pattern for all time steps
    output_mask = np.abs(res[0]) > 1e-6
    for m in output_mask[1:]:
      self.assertAllClose(output_mask[0], m)

  def testDropoutWrapperRecurrentStateInputAndOutput(self):
    # [FUNCTION DOC]
    # Scenario: variational dropout with input, output and state keep probabilities 0.9. Smoke
    # asserts the output mask is the same at every time step and across batch entries, and that
    # the final cell and hidden state masks are the same across batch entries.
    wrapper_type = rnn_cell_impl.DropoutWrapper
    keep_some = 0.9
    res = self._testDropoutWrapper(
        input_keep_prob=keep_some,
        output_keep_prob=keep_some,
        state_keep_prob=keep_some,
        variational_recurrent=True,
        wrapper_type=wrapper_type,
        input_size=3,
        batch_size=5,
        time_steps=7)

    # Smoke test for the state/input masks.
    output_mask = np.abs(res[0]) > 1e-6
    for time_step in output_mask:
      # Ensure the same dropout output pattern for all time steps
      self.assertAllClose(output_mask[0], time_step)
      for batch_entry in time_step:
        # Assert all batch entries get the same mask
        self.assertAllClose(batch_entry, time_step[0])

    # For state, ensure all batch entries have the same mask
    state_c_mask = np.abs(res[1].c) > 1e-6
    state_h_mask = np.abs(res[1].h) > 1e-6
    for batch_entry in state_c_mask:
      self.assertAllClose(batch_entry, state_c_mask[0])
    for batch_entry in state_h_mask:
      self.assertAllClose(batch_entry, state_h_mask[0])

  def testDropoutWrapperRecurrentStateInputAndOutputWithSeed(self):
    # [FUNCTION DOC]
    # Scenario: run the variational dropout helper twice with identical global, numpy and wrapper
    # seeds in different scopes. Asserts the first run has batch- and time-consistent output and
    # state masks, and that outputs and final cell and hidden states are identical across runs.
    # Side effects: sets the global and numpy random seeds.
    wrapper_type = rnn_cell_impl.DropoutWrapper
    keep_some = 0.9
    random_seed.set_random_seed(2347)
    np.random.seed(23487)
    res0 = self._testDropoutWrapper(
        input_keep_prob=keep_some,
        output_keep_prob=keep_some,
        state_keep_prob=keep_some,
        variational_recurrent=True,
        wrapper_type=wrapper_type,
        input_size=3,
        batch_size=5,
        time_steps=7,
        seed=-234987,
        scope="root_0")
    random_seed.set_random_seed(2347)
    np.random.seed(23487)
    res1 = self._testDropoutWrapper(
        input_keep_prob=keep_some,
        output_keep_prob=keep_some,
        state_keep_prob=keep_some,
        variational_recurrent=True,
        wrapper_type=wrapper_type,
        input_size=3,
        batch_size=5,
        time_steps=7,
        seed=-234987,
        scope="root_1")

    output_mask = np.abs(res0[0]) > 1e-6
    for time_step in output_mask:
      # Ensure the same dropout output pattern for all time steps
      self.assertAllClose(output_mask[0], time_step)
      for batch_entry in time_step:
        # Assert all batch entries get the same mask
        self.assertAllClose(batch_entry, time_step[0])

    # For state, ensure all batch entries have the same mask
    state_c_mask = np.abs(res0[1].c) > 1e-6
    state_h_mask = np.abs(res0[1].h) > 1e-6
    for batch_entry in state_c_mask:
      self.assertAllClose(batch_entry, state_c_mask[0])
    for batch_entry in state_h_mask:
      self.assertAllClose(batch_entry, state_h_mask[0])

    # Ensure seeded calculation is identical.
    self.assertAllClose(res0[0], res1[0])
    self.assertAllClose(res0[1].c, res1[1].c)
    self.assertAllClose(res0[1].h, res1[1].h)


if __name__ == "__main__":
  test.main()
