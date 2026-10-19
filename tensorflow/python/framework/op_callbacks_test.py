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
# [MODULE DOC]
# Unit tests for the op callback mechanism (op_callbacks module). Defines a recording callback
# helper class that logs eager op executions, eager FuncGraph executions and graph-construction
# op creations (types, names, attributes, inputs, graphs, instrumented runtime values), and two
# test classes: one for normal behavior (stack per thread, eager/graph/control-flow/dataset/
# sparse/gradient capture, output overriding) and one for error conditions. Running the file
# as a script enables eager execution and runs the test main.
"""Unit tests for op_callback."""

import threading

import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.util import compat


# Keep all the hard-coded op type strings in one place so they are easy to
# change all at once in the face of any possible future op type name changes.
_ADD_OP = b"AddV2"
_ASSIGN_ADD_VARIABLE_OP = b"AssignAddVariableOp"
_CONSTANT_OP = b"Const"
_COS_OP = b"Cos"
_ENTER_OP = b"Enter"
_EXIT_OP = b"Exit"
_GREATER_OP = b"Greater"
_IDENTITY_OP = b"Identity"
_IF_OP = b"If"
_LESS_OP = b"Less"
_LOG_OP = b"Log"
_MERGE_OP = b"Merge"
_MATMUL_OP = b"MatMul"
_MUL_OP = b"Mul"
_NEXT_ITERATION_OP = b"NextIteration"
_PLACEHOLDER_OP = b"Placeholder"
_POW_OP = b"Pow"
_READ_VARIABLE_OP = b"ReadVariableOp"
_SIN_OP = b"Sin"
_SPARSE_TENSOR_DENSE_MATMUL_OP = b"SparseTensorDenseMatMul"
_SQRT_OP = b"Sqrt"
_SQUARE_OP = b"Square"
_STATELESS_IF_OP = b"StatelessIf"
_SWITCH_OP = b"Switch"
_UNIQUE_OP = b"Unique"
_VAR_HANDLE_OP = b"VarHandleOp"
_WHILE_OP = b"While"


class _NumpyFunctionCallback(object):
  # [CLASS DOC]
  # HELPER: recording op-callback fixture used by the tests in this file.
  # Holds parallel lists for eager events (op types, names, attributes, graphs, inputs) and graph
  # construction events (op types, names, attributes, graphs, graph versions, inputs), plus a
  # mapping from op name to the list of runtime values captured. Its callback method is
  # registered as an op callback; in graph mode it can wrap each output in a numpy function that
  # records the runtime value without changing it.

  def __init__(self, instrument_graph_ops=True, float_only=False):
    # [FUNCTION DOC]
    # Stores configuration and initializes all recording lists via reset. Inputs:
    # instrument_graph_ops (bool, default true) whether to wrap graph op outputs with recording
    # numpy functions; float_only (bool, default false) restricts that instrumentation to
    # floating-point outputs. Returns nothing.
    self.instrument_graph_ops = instrument_graph_ops
    self._float_only = float_only
    self.reset()

  def callback(self, op_type, inputs, attrs, outputs, op_name=None, graph=None):
    # [FUNCTION DOC]
    # The op callback itself. Inputs follow the op callback contract: op type, input tensors,
    # attribute tuple, output tensors, optional op name and graph. If no graph is given the
    # event is eager: the type, name, attributes, graph and inputs are appended to the eager
    # lists and nothing is returned. Otherwise the graph-side lists (including the graph
    # version) are appended; if instrumentation is disabled the outputs are returned as is.
    # Otherwise each output is replaced by a numpy-function wrapper (with shape restored) that
    # records its runtime value under the op name, except for control-flow, identity, variable
    # handle, placeholder and constant ops (and non-float outputs when float_only), which pass
    # through unchanged. Returns the list of possibly instrumented outputs.
    is_eager = not graph
    if is_eager:
      self.eager_op_types.append(
          compat.as_bytes(op_type) if op_type else op_type)
      self.eager_op_names.append(
          compat.as_bytes(op_name) if op_name else op_name)
      self.eager_attrs.append(attrs)
      self.eager_graphs.append(graph)
      self.eager_inputs.append(inputs)
    else:
      self.graph_op_types.append(
          compat.as_bytes(op_type) if op_type else op_type)
      self.graph_op_names.append(
          compat.as_bytes(op_name) if op_name else op_name)
      self.graph_attrs.append(attrs)
      self.graph_graphs.append(graph)
      self.graph_graph_versions.append(graph.version)
      self.graph_inputs.append(inputs)

      if not self.instrument_graph_ops:
        return outputs

      # Instrument the graph with numpy_function.
      instrumented_outputs = []
      for output in outputs:
        if compat.as_bytes(op_type) in (_ENTER_OP, _EXIT_OP, _IF_OP, _MERGE_OP,
                                        _NEXT_ITERATION_OP, _STATELESS_IF_OP,
                                        _SWITCH_OP, _WHILE_OP, _IDENTITY_OP,
                                        _VAR_HANDLE_OP, _PLACEHOLDER_OP,
                                        _CONSTANT_OP):
          # TODO(cais): Overriding the output of StatelessIf, If and While ops
          # currently fails with error. Investigate (b/139668453).
          # Avoid instrumenting Identity ops as well, as they are inserted
          # by tf.function/AutoGraph for marshalling outputs.
          instrumented_output = output
        else:
          def record(ndarray_value):
            # [FUNCTION DOC]
            # HELPER: runtime recorder for the instrumented graph outputs.
            # Appends the given array value to the list kept for this op's name in the
            # internal arrays mapping (creating the list if needed) and returns the value
            # unchanged. Captures the op name from the enclosing callback invocation.
            if compat.as_bytes(op_name) not in self.graph_internal_ndarrays:
              self.graph_internal_ndarrays[compat.as_bytes(op_name)] = []
            self.graph_internal_ndarrays[compat.as_bytes(op_name)].append(
                ndarray_value)
            return ndarray_value

          if self._float_only and not output.dtype.is_floating:
            instrumented_output = output
          else:
            instrumented_output = script_ops.numpy_function(
                record, [output], output.dtype)
            instrumented_output.set_shape(output.shape)
        instrumented_outputs.append(instrumented_output)

      return instrumented_outputs

  def reset(self):
    # [FUNCTION DOC]
    # Clears all recorded state: replaces every eager and graph recording list with a new empty
    # list and the internal arrays mapping with an empty mapping. No inputs or return value.
    self.eager_op_types = []
    self.eager_op_names = []
    self.eager_attrs = []
    self.eager_graphs = []
    self.eager_inputs = []
    self.graph_op_types = []
    self.graph_op_names = []
    self.graph_attrs = []
    self.graph_graphs = []
    self.graph_graph_versions = []
    self.graph_inputs = []

    # A dict mapping tensor name (e.g., "MatMut_10") to a list of ndarrays.
    # The list is the history of the tensor's computation result inside
    # `tf.Graph`s (`FuncGraph`s).
    # For an op with multiple output tensors, the outputs are interleaved in
    # the list.
    self.graph_internal_ndarrays = {}


class OpCallbacksTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the op callback feature under eager and graph modes: callback registration,
  # per-thread isolation, capture of eager ops, eager function executions, graph construction
  # (including control flow, datasets, sparse tensors, gradient tapes), attributes, inputs and
  # output overriding. Uses the recording callback helper; tearDown clears all callbacks.

  def tearDown(self):
    # [FUNCTION DOC]
    # Removes all op callbacks registered in the current thread so tests do not leak state,
    # then runs the parent tearDown.
    op_callbacks.clear_op_callbacks()
    super(OpCallbacksTest, self).tearDown()

  def testSingleThreadedStack(self):
    # [FUNCTION DOC]
    # Scenario: add two callbacks then remove them in reverse order on one thread. Asserts the
    # context's callback collection grows to size one then two, contains the expected callables
    # after each step, and shrinks back to zero.
    ctx = context.context()
    instrument_0 = _NumpyFunctionCallback()
    instrument_1 = _NumpyFunctionCallback()

    op_callbacks.add_op_callback(instrument_0.callback)
    self.assertEqual(1, len(ctx.op_callbacks))
    self.assertIn(instrument_0.callback, ctx.op_callbacks)

    op_callbacks.add_op_callback(instrument_1.callback)
    self.assertEqual(2, len(ctx.op_callbacks))
    self.assertIn(instrument_0.callback, ctx.op_callbacks)
    self.assertIn(instrument_1.callback, ctx.op_callbacks)

    op_callbacks.remove_op_callback(instrument_1.callback)
    self.assertEqual(1, len(ctx.op_callbacks))
    self.assertIn(instrument_0.callback, ctx.op_callbacks)

    op_callbacks.remove_op_callback(instrument_0.callback)
    self.assertEqual(0, len(ctx.op_callbacks))

  def testMultiThreadedStacks(self):
    # [FUNCTION DOC]
    # Scenario: the main thread and a second thread each register their own recording callback
    # and build a different traced function (sin/square versus log/sqrt). Asserts there is no
    # cross-talk: each callback sees only the ops built on its own thread (placeholder, log,
    # sqrt for the second; sin, square for the main) and the computed values are correct.
    # Instrument for the main thread.
    instrument_0 = _NumpyFunctionCallback()

    # Instrument for the to-be-created thread.
    instrument_1 = _NumpyFunctionCallback()

    def thread1_job():
      # [FUNCTION DOC]
      # HELPER: body of the secondary thread in the multi-threaded stack test.
      # Registers the second recording callback on this thread, traces and runs a function
      # computing the square root of the log of a constant, and asserts the numeric result.
      op_callbacks.add_op_callback(instrument_1.callback)

      @def_function.function
      def func1(x):
        # [FUNCTION DOC]
        # Traced function: returns the square root of the natural log of the input tensor.
        return math_ops.sqrt(math_ops.log(x))

      x = constant_op.constant(4.0)
      self.assertAllClose(func1(x), np.sqrt(np.log(4.0)))

    thread1 = threading.Thread(target=thread1_job)

    # Start job on separate thread.
    thread1.start()

    # Run something on the main thread.
    op_callbacks.add_op_callback(instrument_0.callback)

    @def_function.function
    def func0(x):
      # [FUNCTION DOC]
      # Traced function: returns the square of the sine of the input tensor (main thread).
      return math_ops.square(math_ops.sin(x))

    x = constant_op.constant(4.0)
    self.assertAllClose(func0(x), np.square(np.sin(4.0)))

    thread1.join()

    # Assert that there is no cross-talk between the main thread
    # and the created thread.
    self.assertIn(_PLACEHOLDER_OP, instrument_1.graph_op_types)
    self.assertIn(_LOG_OP, instrument_1.graph_op_types)
    self.assertIn(_SQRT_OP, instrument_1.graph_op_types)
    self.assertNotIn(_SIN_OP, instrument_1.graph_op_types)
    self.assertNotIn(_SQUARE_OP, instrument_1.graph_op_types)

    self.assertNotIn(_LOG_OP, instrument_0.graph_op_types)
    self.assertNotIn(_SQRT_OP, instrument_0.graph_op_types)
    self.assertIn(_SIN_OP, instrument_0.graph_op_types)
    self.assertIn(_SQUARE_OP, instrument_0.graph_op_types)

  def testEagerOpExecution(self):
    # [FUNCTION DOC]
    # Scenario: with a callback registered, eagerly compute the square of the log of a constant.
    # Asserts the numeric result; that the eager record holds the log and square op types in
    # order, with null names and graphs; that each recorded inputs entry is a one-element tuple
    # with the expected value; and that no graph-side events were recorded.
    instrument = _NumpyFunctionCallback()

    x = constant_op.constant(6.0)
    op_callbacks.add_op_callback(instrument.callback)
    y = math_ops.square(math_ops.log(x))
    self.assertAllClose(y, np.square(np.log(6.0)))

    self.assertEqual(instrument.eager_op_types, [_LOG_OP, _SQUARE_OP])
    # Op names are unavailable under eager mode.
    self.assertEqual(instrument.eager_op_names, [None, None])
    self.assertEqual(instrument.eager_graphs, [None, None])
    self.assertEqual(len(instrument.eager_inputs), 2)
    self.assertEqual(len(instrument.eager_inputs[0]), 1)
    self.assertIsInstance(instrument.eager_inputs[0], tuple)
    self.assertEqual(instrument.eager_inputs[0][0], x)
    self.assertEqual(len(instrument.eager_inputs[1]), 1)
    self.assertIsInstance(instrument.eager_inputs[1], tuple)
    self.assertAllClose(instrument.eager_inputs[1][0], np.log(6.0))
    self.assertFalse(instrument.graph_op_types)
    self.assertFalse(instrument.graph_op_names)
    self.assertFalse(instrument.graph_attrs)
    self.assertFalse(instrument.graph_graphs)
    self.assertFalse(instrument.graph_inputs)

  def testMultiThreadedEagerOpExecution(self):
    # [FUNCTION DOC]
    # Scenario: two threads register separate callbacks and run different eager ops (log then
    # square on the new thread, cosine on the main thread). Asserts each callback recorded only
    # its own thread's op types and that op names are null in eager mode.
    # Instrument for the main thread.
    instrument_0 = _NumpyFunctionCallback()

    # Instrument for the to-be-created thread.
    instrument_1 = _NumpyFunctionCallback()

    def thread_1_job():
      # [FUNCTION DOC]
      # HELPER: secondary thread body. Creates a constant, registers the second callback,
      # eagerly computes square of log, removes the callback, and returns the result.
      x = constant_op.constant(6.0)
      op_callbacks.add_op_callback(instrument_1.callback)
      y = math_ops.square(math_ops.log(x))
      op_callbacks.remove_op_callback(instrument_1.callback)
      return y

    thread_1 = threading.Thread(target=thread_1_job)
    thread_1.start()

    # While thread_1 is ongoing, do something on the main thread.
    x = constant_op.constant(2.0)
    op_callbacks.add_op_callback(instrument_0.callback)
    y = math_ops.cos(x)
    self.assertAllClose(y, np.cos(2.0))
    op_callbacks.remove_op_callback(instrument_0.callback)

    thread_1.join()

    self.assertEqual(instrument_0.eager_op_types, [_COS_OP])
    self.assertEqual(instrument_0.eager_op_names, [None])
    self.assertEqual(instrument_1.eager_op_types, [_LOG_OP, _SQUARE_OP])
    self.assertEqual(instrument_1.eager_op_names, [None, None])

  def testEagerFunctionExecution(self):
    # [FUNCTION DOC]
    # Scenario: a traced square-of-log function is called once per dtype (float32 and float64)
    # before registering the callback so tracing is not recorded, then called again with the
    # callback active. Asserts numeric results; that the eager record has two null-named events
    # whose types include each concrete function's graph name; that no graph events exist; and
    # that the recorded inputs are tuples whose first element is the corresponding argument.
    instrument = _NumpyFunctionCallback()

    @def_function.function
    def square_log(x):
      # [FUNCTION DOC]
      # Traced function: square of the natural log of the input.
      return math_ops.square(math_ops.log(x))

    # Call the function once, so that the graph construction won't show up
    # in the callback.
    x_float32 = constant_op.constant(6.0, dtype=dtypes.float32)
    x_float64 = constant_op.constant(6.0, dtype=dtypes.float64)
    square_log(x_float32)
    square_log(x_float64)

    op_callbacks.add_op_callback(instrument.callback)
    y = square_log(x_float32)
    self.assertAllClose(y, np.square(np.log(6.0)))
    y = square_log(x_float64)
    self.assertAllClose(y, np.square(np.log(6.0)))

    self.assertEqual(instrument.eager_op_names, [None, None])
    self.assertFalse(instrument.graph_op_types)
    self.assertFalse(instrument.graph_op_names)
    self.assertFalse(instrument.graph_inputs)

    # Each of the two dtypes should be associated with its own FuncGraph.
    self.assertIn(
        square_log.get_concrete_function(x_float32).name,
        instrument.eager_op_types)
    self.assertIn(
        square_log.get_concrete_function(x_float64).name,
        instrument.eager_op_types)

    self.assertEqual(len(instrument.eager_inputs), 2)
    self.assertIsInstance(instrument.eager_inputs[0], tuple)
    self.assertEqual(instrument.eager_inputs[0][0], x_float32)
    self.assertIsInstance(instrument.eager_inputs[1], tuple)
    self.assertEqual(instrument.eager_inputs[1][0], x_float64)

  def testMultiThreadedEagerFunctionExecution(self):
    # [FUNCTION DOC]
    # Scenario: a traced function is pre-called for two dtypes; then the main thread runs the
    # float64 variant and a second thread runs the float32 variant, each with its own callback.
    # Asserts each callback recorded only the concrete-function name for its own dtype, with
    # null op names, and no graph-construction events.
    # Instrument for the main thread.
    instrument_0 = _NumpyFunctionCallback()

    # Instrument for the to-be-created thread.
    instrument_1 = _NumpyFunctionCallback()

    @def_function.function
    def square_log(x):
      # [FUNCTION DOC]
      # Traced function: square of the natural log of the input.
      return math_ops.square(math_ops.log(x))

    # Call the function once, so that the graph construction won't show up
    # in the callback.
    x_float32 = constant_op.constant(6.0, dtype=dtypes.float32)
    x_float64 = constant_op.constant(6.0, dtype=dtypes.float64)
    square_log(x_float32)
    square_log(x_float64)

    def thread_1_job():
      # [FUNCTION DOC]
      # HELPER: secondary thread body. Registers the second callback and runs the traced
      # function on the float32 input.
      op_callbacks.add_op_callback(instrument_1.callback)
      square_log(x_float32)

    thread_1 = threading.Thread(target=thread_1_job)
    thread_1.start()

    # In the meantime, run some computation on the main thread.
    op_callbacks.add_op_callback(instrument_0.callback)
    square_log(x_float64)

    thread_1.join()

    # Each of the two dtypes should be associated with its own FuncGraph.
    self.assertIn(
        square_log.get_concrete_function(x_float64).name,
        instrument_0.eager_op_types)
    self.assertEqual(instrument_0.eager_op_names, [None])
    self.assertFalse(instrument_0.graph_op_types)
    self.assertIn(
        square_log.get_concrete_function(x_float32).name,
        instrument_1.eager_op_types)
    self.assertEqual(instrument_1.eager_op_names, [None])
    self.assertFalse(instrument_1.graph_op_types)

  @test_util.run_in_graph_and_eager_modes
  def testSimpleGraphConstructionScopeOutsideFunction(self):
    # [FUNCTION DOC]
    # Scenario: register the instrumenting callback outside a traced function that computes
    # log of two plus the unique values of a vector, then run it. Asserts results; that unique,
    # add and log op types were seen during graph construction with matching name counts; and
    # that the runtime values recorded for each op match expected arrays (counts asserted only
    # in eager mode because graph-mode sessions run twice).
    instrument = _NumpyFunctionCallback()

    op_callbacks.add_op_callback(instrument.callback)

    @def_function.function
    def log_2plus_unique_x(x):
      # [FUNCTION DOC]
      # Traced function: computes unique values and positions of the input, returns log of two
      # plus the unique values together with the positions.
      unique_values, unique_pos = array_ops.unique(x)
      return math_ops.log(2.0 + unique_values), unique_pos

    x = constant_op.constant([-1.0, -1.0, 0.0], dtype=dtypes.float32)
    y1, y2 = log_2plus_unique_x(x)
    self.assertAllClose(y1, [0.0, np.log(2.0)])
    self.assertAllClose(y2, [0, 0, 1])

    self.assertIn(_UNIQUE_OP, instrument.graph_op_types)
    self.assertIn(_ADD_OP, instrument.graph_op_types)
    self.assertIn(_LOG_OP, instrument.graph_op_types)
    self.assertEqual(
        len(instrument.graph_op_names), len(instrument.graph_op_types))

    # Check the graph internal ndarrays recorded at runtime.
    unique_op_outputs = instrument.graph_internal_ndarrays[_UNIQUE_OP]
    if context.executing_eagerly():
      # b/140810696: The run_in_graph_and_eager_modes decorator runs
      # Session.run() twice. We can't assert on the number of outputs in
      # that case.
      self.assertEqual(len(unique_op_outputs), 2)
    self.assertAllClose(unique_op_outputs[0], [-1.0, 0.0])
    self.assertAllClose(unique_op_outputs[1], [0, 0, 1])
    add_op_outputs = instrument.graph_internal_ndarrays[b"add"]
    if context.executing_eagerly():
      self.assertEqual(len(add_op_outputs), 1)
    self.assertAllClose(add_op_outputs[0], [1.0, 2.0])
    log_op_outputs = instrument.graph_internal_ndarrays[_LOG_OP]
    if context.executing_eagerly():
      self.assertEqual(len(log_op_outputs), 1)
    self.assertAllClose(log_op_outputs[0], [0.0, np.log(2.0)])

  @test_util.run_in_graph_and_eager_modes
  def testPadOp(self):
    # [FUNCTION DOC]
    # Scenario: a traced function pads a 2x2 float matrix with one row and two columns of zeros
    # on each side. Asserts the output equals the expected 4x6 zero-padded array and that the
    # runtime value recorded for the padding op equals the same array.
    instrument = _NumpyFunctionCallback()

    op_callbacks.add_op_callback(instrument.callback)

    @def_function.function
    def my_pad(x, padding):
      # [FUNCTION DOC]
      # Traced function: pads the input tensor using the given paddings and returns it.
      return array_ops.pad(x, padding)

    x = constant_op.constant([[1, 2], [3, 4]], dtype=dtypes.float32)
    paddings = [[1, 1], [2, 2]]

    y = my_pad(x, paddings)
    expected_output = np.array([
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 2, 0, 0],
        [0, 0, 3, 4, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ], dtype=np.float32)
    self.assertAllClose(y, expected_output)
    self.assertAllClose(
        instrument.graph_internal_ndarrays[b"Pad"][0], expected_output)

  @test_util.run_in_graph_and_eager_modes
  def testSimpleGraphConstructionWithCallbackReturningNone(self):
    # [FUNCTION DOC]
    # Scenario: register a callback that only records op types and returns nothing, then run a
    # traced log-of-one-plus-x function. Asserts the result is correct (graph construction is not
    # disturbed by a null return) and that add and log op types were recorded.
    """Test that callbacks that return None works."""
    op_types = []
    def no_return_callback(op_type,
                           inputs,
                           attrs,
                           outputs,
                           op_name=None,
                           graph=None):
      # [FUNCTION DOC]
      # Test callback: appends the op type (as bytes) to the enclosing list; ignores all other
      # arguments and returns nothing.
      del inputs, attrs, outputs, op_name, graph  # Unused.
      op_types.append(compat.as_bytes(op_type))

    op_callbacks.add_op_callback(no_return_callback)

    @def_function.function
    def log1p(x):
      # [FUNCTION DOC]
      # Traced function: returns the natural log of one plus the input.
      return math_ops.log(1.0 + x)
    x = constant_op.constant(3.0)
    y = log1p(x)

    self.assertAllClose(y, np.log(4.0))
    self.assertIn(_ADD_OP, op_types)
    self.assertIn(_LOG_OP, op_types)

  @test_util.run_in_graph_and_eager_modes
  def testGraphConstructionInputsAndGraphAreCapturedCorrectly(self):
    # [FUNCTION DOC]
    # Scenario: run the unique/add/log traced function with a non-instrumenting callback.
    # Asserts the recorded input tuples per op (unique takes a placeholder; add takes a constant
    # and the unique output; log takes the add output), that one graph is recorded per op, and,
    # in eager mode, that graph versions increase monotonically.
    instrument = _NumpyFunctionCallback(instrument_graph_ops=False)

    op_callbacks.add_op_callback(instrument.callback)

    @def_function.function
    def log_2plus_unique_x(x):
      # [FUNCTION DOC]
      # Traced function: computes unique values and positions of the input, returns log of two
      # plus the unique values together with the positions.
      unique_values, unique_pos = array_ops.unique(x)
      return math_ops.log(2.0 + unique_values), unique_pos

    x = constant_op.constant([-1.0, -1.0, 0.0], dtype=dtypes.float32)
    y1, y2 = log_2plus_unique_x(x)
    self.assertAllClose(y1, [0.0, np.log(2.0)])
    self.assertAllClose(y2, [0, 0, 1])

    # Check the recorded input tensors.
    self.assertEqual(
        len(instrument.graph_inputs), len(instrument.graph_op_types))
    unique_inputs = instrument.graph_inputs[instrument.graph_op_types.index(
        _UNIQUE_OP)]
    self.assertIsInstance(unique_inputs, tuple)
    self.assertEqual(len(unique_inputs), 1)
    self.assertEqual(
        compat.as_bytes(unique_inputs[0].op.op_def.name), _PLACEHOLDER_OP)

    add_inputs = instrument.graph_inputs[instrument.graph_op_types.index(
        _ADD_OP)]
    self.assertIsInstance(add_inputs, tuple)
    self.assertEqual(len(add_inputs), 2)
    self.assertEqual(
        compat.as_bytes(add_inputs[0].op.op_def.name), _CONSTANT_OP)
    self.assertEqual(compat.as_bytes(add_inputs[1].op.op_def.name), _UNIQUE_OP)

    log_inputs = instrument.graph_inputs[instrument.graph_op_types.index(
        _LOG_OP)]
    self.assertIsInstance(log_inputs, tuple)
    self.assertEqual(len(log_inputs), 1)
    self.assertEqual(compat.as_bytes(log_inputs[0].op.op_def.name), _ADD_OP)

    # Check the recorded graphs.
    self.assertEqual(
        len(instrument.graph_graphs), len(instrument.graph_op_types))
    self.assertGreater(len(instrument.graph_graph_versions), 1)
    if context.executing_eagerly():
      for i in range(len(instrument.graph_graph_versions) - 1):
        self.assertGreater(instrument.graph_graph_versions[i + 1],
                           instrument.graph_graph_versions[i])

  @test_util.run_in_graph_and_eager_modes
  def testEagerGraphOpConstructionSimpleGraphScopeInsideFunction(self):
    # [FUNCTION DOC]
    # Scenario: the callback is added and removed inside the traced function body, around the
    # unique, add and log ops but before the sine op. Asserts the result; that unique, add and
    # log were captured while sine was not; matching name counts; and the runtime values
    # recorded for unique, add and log.
    instrument = _NumpyFunctionCallback()

    @def_function.function
    def log_2plus_unique_x(x):
      # [FUNCTION DOC]
      # Traced function: registers the callback, builds unique and log-of-two-plus ops, removes
      # the callback, then returns the sine of the intermediate result. Side effect: toggles
      # callback registration during tracing.
      op_callbacks.add_op_callback(instrument.callback)
      unique_values, _ = array_ops.unique(x)
      y = math_ops.log(2.0 + unique_values)
      op_callbacks.remove_op_callback(instrument.callback)
      return math_ops.sin(y)

    x = constant_op.constant([-1.0, -1.0, 0.0], dtype=dtypes.float32)
    output = log_2plus_unique_x(x)
    self.assertAllClose(output, np.sin([0.0, np.log(2.0)]))

    # The following ops should have been captured by the callback
    # because they were constructed within the scope of `op_callback()`.
    self.assertIn(_UNIQUE_OP, instrument.graph_op_types)
    self.assertIn(_ADD_OP, instrument.graph_op_types)
    self.assertIn(_LOG_OP, instrument.graph_op_types)
    # The "Sin" op should not have been captured, because it was constructed
    # outside the scope of `op_callback()`.
    self.assertNotIn(_SIN_OP, instrument.graph_op_types)
    self.assertEqual(
        len(instrument.graph_op_names), len(instrument.graph_op_types))

    # Check the graph internal ndarrays recorded at runtime.
    unique_op_outputs = instrument.graph_internal_ndarrays[_UNIQUE_OP]
    self.assertEqual(len(unique_op_outputs), 2)
    self.assertAllClose(unique_op_outputs[0], [-1.0, 0.0])
    self.assertAllClose(unique_op_outputs[1], [0, 0, 1])
    add_op_outputs = instrument.graph_internal_ndarrays[b"add"]
    self.assertEqual(len(add_op_outputs), 1)
    self.assertAllClose(add_op_outputs[0], [1.0, 2.0])
    log_op_outputs = instrument.graph_internal_ndarrays[_LOG_OP]
    self.assertEqual(len(log_op_outputs), 1)
    self.assertAllClose(log_op_outputs[0], [0.0, np.log(2.0)])

  def testEagerOpAttributesAreCapture(self):
    # [FUNCTION DOC]
    # Scenario: eagerly run a matmul with transpose_a true and transpose_b false. Asserts the
    # product, that exactly one eager attribute record exists and is a tuple of alternating
    # keys and values holding the expected booleans, and that no graph attributes were recorded.
    instrument = _NumpyFunctionCallback()
    m = constant_op.constant([[1.0, -1.0], [0.0, 1.0]])
    x = constant_op.constant([[-2.0], [3.0]])
    op_callbacks.add_op_callback(instrument.callback)
    y = math_ops.matmul(m, x, transpose_a=True, transpose_b=False)
    self.assertAllClose(y, [[-2.0], [5.0]])

    self.assertEqual(len(instrument.eager_attrs), 1)
    self.assertIsInstance(instrument.eager_attrs[0], tuple)
    self.assertEqual(
        instrument.eager_attrs[0][instrument.eager_attrs[0].index("transpose_a")
                                  + 1], True)
    self.assertEqual(
        instrument.eager_attrs[0][instrument.eager_attrs[0].index("transpose_b")
                                  + 1], False)
    self.assertEqual(len(instrument.graph_attrs), 0)

  @test_util.run_in_graph_and_eager_modes
  def testGraphOpAttributesAreCapture(self):
    # [FUNCTION DOC]
    # Scenario: a traced function does a matmul with transpose_a true and transpose_b false.
    # Asserts the product; that the matmul graph record's attributes are a tuple whose values
    # (attribute messages) have the expected boolean fields; and, in eager mode, that one eager
    # attribute tuple was recorded for the function execution.
    instrument = _NumpyFunctionCallback()

    @def_function.function
    def my_matmul(m, x):
      # [FUNCTION DOC]
      # Traced function: matrix product of m (transposed) and x.
      return math_ops.matmul(m, x, transpose_a=True, transpose_b=False)

    m = constant_op.constant([[1.0, -1.0], [0.0, 1.0]])
    x = constant_op.constant([[-2.0], [3.0]])
    op_callbacks.add_op_callback(instrument.callback)
    y = my_matmul(m, x)
    self.assertAllClose(y, [[-2.0], [5.0]])

    index = instrument.graph_op_types.index(_MATMUL_OP)
    self.assertIsInstance(instrument.graph_attrs[index], tuple)
    self.assertEqual(
        instrument.graph_attrs[index][
            instrument.graph_attrs[index].index("transpose_a") + 1].b, True)
    self.assertEqual(
        instrument.graph_attrs[index][
            instrument.graph_attrs[index].index("transpose_b") + 1].b, False)
    if context.executing_eagerly():
      self.assertEqual(len(instrument.eager_attrs), 1)
      self.assertIsInstance(instrument.eager_attrs[0], tuple)

  @test_util.run_in_graph_and_eager_modes
  def testEagerGraphOpConstructionIfControlFlow(self):
    # [FUNCTION DOC]
    # Scenario: a traced function with a conditional on whether input is greater than zero
    # (squares if so, cubes otherwise) is run with a negative input. Asserts the result is
    # minus sixty-four; that conditional, greater and power op types were captured; and that the
    # recorded runtime values for greater (false) and power (minus sixty-four) are right, using
    # a cond-scope name prefix in eager mode.
    instrument = _NumpyFunctionCallback()
    op_callbacks.add_op_callback(instrument.callback)

    @def_function.function
    def my_function_with_cond(x):
      # [FUNCTION DOC]
      # Traced function: returns the square of x if x is greater than zero, else its cube.
      if math_ops.greater(x, 0.0):
        return x**2.0
      else:
        return x**3.0

    x = constant_op.constant(-4.0)
    self.assertAllClose(my_function_with_cond(x), -64.0)

    self.assertIn(_IF_OP, instrument.graph_op_types)
    self.assertIn(_GREATER_OP, instrument.graph_op_types)
    self.assertIn(_POW_OP, instrument.graph_op_types)
    self.assertEqual(
        len(instrument.graph_op_names), len(instrument.graph_op_types))

    # Check the graph internal ndarrays recorded at runtime.
    greater_op_outputs = instrument.graph_internal_ndarrays[_GREATER_OP]
    self.assertEqual(len(greater_op_outputs), 1)
    self.assertAllClose(greater_op_outputs[0], False)
    prefix = b"cond/" if context.executing_eagerly() else b""
    pow_op_outputs = instrument.graph_internal_ndarrays[b"%spow" % prefix]
    self.assertEqual(len(pow_op_outputs), 1)
    self.assertAllClose(pow_op_outputs[0], -64.0)

  def testEagerGraphOpConstructionWhileLoopControlFlow(self):
    # [FUNCTION DOC]
    # Scenario: a traced while loop doubles an accumulator variable four times while a counter
    # variable increments. Asserts the accumulator ends at sixteen; that while, less and
    # variable-assign-add op types were captured; and that the recorded runtime values for the
    # in-loop variable read are one, two, four, eight and for the less comparison four trues
    # then a false.
    instrument = _NumpyFunctionCallback()
    op_callbacks.add_op_callback(instrument.callback)

    @def_function.function
    def my_function_with_while(counter, lim, accum):
      # [FUNCTION DOC]
      # Traced function: while counter is less than lim, adds accum to itself and increments
      # counter by one. Mutates both variables; returns nothing.
      while math_ops.less(counter, lim):
        accum.assign_add(accum)
        counter.assign_add(1.0)

    counter = variables.Variable(0.0)
    lim = constant_op.constant(4.0, dtype=dtypes.float32)
    accum = variables.Variable(1.0)
    my_function_with_while(counter, lim, accum)

    self.assertAllClose(accum.read_value(), 16.0)
    self.assertIn(_WHILE_OP, instrument.graph_op_types)
    self.assertIn(_LESS_OP, instrument.graph_op_types)
    self.assertIn(_ASSIGN_ADD_VARIABLE_OP, instrument.graph_op_types)
    self.assertEqual(
        len(instrument.graph_op_names), len(instrument.graph_op_types))

    # Check the graph internal ndarrays recorded at runtime.
    read_variable_op_outputs = instrument.graph_internal_ndarrays[
        b"while/" + _READ_VARIABLE_OP]
    self.assertAllClose(read_variable_op_outputs, [1.0, 2.0, 4.0, 8.0])
    less_op_outputs = instrument.graph_internal_ndarrays[b"while/" + _LESS_OP]
    self.assertAllClose(less_op_outputs, [True, True, True, True, False])

  # TODO(cais): The following isn't decorated with
  # `@test_util.run_in_graph_and_eager_modes` because of some apparent
  # between `Dataset.map()` and `numpy_function()` used by
  # `_NumpyFunctionCallback`. Maybe investigate.
  def testDatasetMapTest(self):
    # [FUNCTION DOC]
    # Scenario: a dataset of ten floats is batched by two and mapped through a log of square
    # plus one function, then consumed by a one-shot iterator. Asserts the first two batches'
    # values, that square, add and log were captured as graph ops, and that the eager type and
    # name lists have equal length.
    instrument = _NumpyFunctionCallback()

    op_callbacks.add_op_callback(instrument.callback)

    tensor = constant_op.constant(
        [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])

    def map_fn(x):
      # [FUNCTION DOC]
      # Dataset map function: returns the natural log of the square of x plus one.
      return math_ops.log(math_ops.square(x) + 1)

    dataset = dataset_ops.Dataset.from_tensor_slices(tensor).batch(2).map(
        map_fn)
    iterator = dataset_ops.make_one_shot_iterator(dataset)

    self.assertAllClose(iterator.next(), np.log([1.25, 2]))
    self.assertAllClose(iterator.next(), np.log([3.25, 5]))

    self.assertIn(_SQUARE_OP, instrument.graph_op_types)
    self.assertIn(_ADD_OP, instrument.graph_op_types)
    self.assertIn(_LOG_OP, instrument.graph_op_types)
    self.assertEqual(
        len(instrument.eager_op_types), len(instrument.eager_op_names))

  def testSparseTensorEagerExecution(self):
    # [FUNCTION DOC]
    # Scenario: eagerly multiply a 4x5 sparse value (three non-zero entries) by a 5x1 tensor of
    # ones. Asserts the dense result, that the sparse-dense matmul op type was recorded as an
    # eager event, and that no graph events were recorded.
    instrument = _NumpyFunctionCallback()
    op_callbacks.add_op_callback(instrument.callback)

    indices = [[1, 2], [2, 0], [3, 4]]
    values = [0.0, 8.0, -2.0]
    shape = [4, 5]
    sp = sparse_tensor.SparseTensorValue(indices, values, shape)
    w = ops.convert_to_tensor(np.ones([5, 1], np.float32))

    y = sparse_ops.sparse_tensor_dense_matmul(sp, w)
    self.assertAllClose(y, [[0.0], [0.0], [8.0], [-2.0]])
    self.assertIn(_SPARSE_TENSOR_DENSE_MATMUL_OP, instrument.eager_op_types)
    self.assertFalse(instrument.graph_op_types)

  @test_util.run_in_graph_and_eager_modes
  def testSparseTensorFuncGraph(self):
    # [FUNCTION DOC]
    # Scenario: a traced function multiplies a sparse value by a dense tensor of ones. Asserts
    # the dense result; that the sparse-dense matmul op type appears among graph events; in eager
    # mode that the concrete function name appears among eager events; and that the recorded
    # runtime output equals the expected values.
    instrument = _NumpyFunctionCallback()
    op_callbacks.add_op_callback(instrument.callback)

    @def_function.function
    def dense_matmul(sp, w):
      # [FUNCTION DOC]
      # Traced function: returns the sparse-tensor by dense-matrix product of sp and w.
      return sparse_ops.sparse_tensor_dense_matmul(sp, w)

    indices = [[1, 2], [2, 0], [3, 4]]
    values = [0.0, 8.0, -2.0]
    shape = [4, 5]
    sp = sparse_tensor.SparseTensorValue(indices, values, shape)
    w = ops.convert_to_tensor(np.ones([5, 1], np.float32))
    y = dense_matmul(sp, w)
    self.assertAllClose(y, [[0.0], [0.0], [8.0], [-2.0]])
    self.assertIn(_SPARSE_TENSOR_DENSE_MATMUL_OP, instrument.graph_op_types)
    if context.executing_eagerly():
      self.assertIn(
          dense_matmul.get_concrete_function(sp, w).name,
          instrument.eager_op_types)

    # Check the graph internal ndarrays recorded at runtime.
    sparse_matmul_outputs = instrument.graph_internal_ndarrays[
        _SPARSE_TENSOR_DENSE_MATMUL_OP + b"/" + _SPARSE_TENSOR_DENSE_MATMUL_OP]
    if context.executing_eagerly():
      self.assertEqual(len(sparse_matmul_outputs), 1)
    self.assertAllClose(sparse_matmul_outputs[0], [[0.0], [0.0], [8.0], [-2.0]])

  @test_util.run_in_graph_and_eager_modes
  def testOverrideDTypeInFuncGraph(self):
    # [FUNCTION DOC]
    # Scenario: a callback casts every graph op output to float64 (leaving constants and
    # placeholders alone), applied to a traced add-one-times-two function on a float32 input.
    # Asserts the output dtype is float64 and the value is eight.
    def to_float64(op_type, inputs, attrs, outputs, op_name=None, graph=None):
      # [FUNCTION DOC]
      # Test callback: returns outputs unchanged for constant and placeholder ops, otherwise a
      # list of the outputs cast to double precision. Other arguments are ignored.
      del inputs, attrs, op_name, graph  # Unused.
      if op_type in ("Const", "Placeholder"):
        return outputs
      else:
        return [math_ops.cast(output, dtypes.float64) for output in outputs]

    op_callbacks.add_op_callback(to_float64)

    @def_function.function
    def add_1_times_2(x):
      # [FUNCTION DOC]
      # Traced function: returns the input plus one, multiplied by two.
      return (x + 1.0) * 2.0

    x = constant_op.constant(3.0, dtype=dtypes.float32)
    y = add_1_times_2(x)
    self.assertEqual(y.dtype, dtypes.float64)
    self.assertAllClose(y, 8.0)

  def testNoOutputOpUnderEagerExecution(self):
    # [FUNCTION DOC]
    # Scenario: add two constants eagerly, then run a group op (which has no outputs) on the
    # sum with a callback active. Asserts the group call returns nothing and that only the add
    # op was recorded as an eager event.
    instrument = _NumpyFunctionCallback()

    x = constant_op.constant(10.0)
    y = constant_op.constant(20.0)
    op_callbacks.add_op_callback(instrument.callback)
    z = x + y
    w = control_flow_ops.group([z])
    self.assertIsNone(w)
    self.assertEqual(instrument.eager_op_types, [_ADD_OP])

  def testOpCallbackCapturesConstTensors(self):
    # [FUNCTION DOC]
    # Scenario: a traced function computes x times two plus three. Asserts the result is
    # twenty-three for input ten and that exactly two constant ops (the literals) were captured
    # during graph construction.
    instrument = _NumpyFunctionCallback()
    op_callbacks.add_op_callback(instrument.callback)

    @def_function.function
    def times_two_plus_three(x):
      # [FUNCTION DOC]
      # Traced function: returns the input times two plus three.
      return x * 2.0 + 3.0

    self.assertAllClose(times_two_plus_three(constant_op.constant(10.0)), 23.0)
    self.assertEqual(instrument.graph_op_types.count(b"Const"), 2)

  @test_util.run_in_graph_and_eager_modes
  def testOpCallbackWorksWithGradientTape(self):
    # [FUNCTION DOC]
    # Scenario: a traced function uses a gradient tape to differentiate the sine of the square
    # of a variable (initialized to three; in graph mode the initializer is run first). Asserts
    # the gradient equals the chain-rule value; that square, sine, multiply and cosine ops were
    # captured (the last two come from backprop); and that the recorded cosine runtime value
    # under the gradient-tape scope has one entry equal to the cosine of nine.
    instrument = _NumpyFunctionCallback()
    op_callbacks.add_op_callback(instrument.callback)

    v = variables.Variable(3.0, dtype=dtypes.float32)
    if not context.executing_eagerly():
      self.evaluate(v.initializer)

    @def_function.function
    def get_gradients():
      # [FUNCTION DOC]
      # Traced function: under a gradient tape computes the sine of the square of the captured
      # variable and returns the gradient of that loss with respect to the variable.
      with backprop.GradientTape() as tape:
        loss = math_ops.sin(math_ops.square(v))
        gradients = tape.gradient(loss, v)
      return gradients

    gradients = get_gradients()
    # Applying the chain rule.
    self.assertAllClose(gradients, np.cos(3.0 * 3.0) * 3.0 * 2.0)
    self.assertIn(_SQUARE_OP, instrument.graph_op_types)
    self.assertIn(_SIN_OP, instrument.graph_op_types)
    # The mul and cos ops are created for backprop.
    self.assertIn(_MUL_OP, instrument.graph_op_types)
    self.assertIn(_COS_OP, instrument.graph_op_types)

    # Check the ndarrays from runtime.
    cos_op_outputs = instrument.graph_internal_ndarrays[b"gradient_tape/" +
                                                        _COS_OP]
    self.assertEqual(len(cos_op_outputs), 1)
    self.assertAllClose(cos_op_outputs[0], np.cos(3.0 * 3.0))


class OpCallbacksErrorConditionsTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for error handling of the op callback API: non-callable arguments, removing callbacks
  # that were never registered or already removed, and callbacks returning the wrong number of
  # outputs. tearDown clears all callbacks.

  def tearDown(self):
    # [FUNCTION DOC]
    # Removes all op callbacks registered in the current thread, then runs the parent tearDown.
    op_callbacks.clear_op_callbacks()
    super(OpCallbacksErrorConditionsTest, self).tearDown()

  def testNonCallableObjectArgErrors(self):
    # [FUNCTION DOC]
    # Scenario: add a callback that is a plain number. Asserts a value error whose message says
    # the callback is expected to be callable.
    with self.assertRaisesRegex(ValueError, r"is expected to be callable"):
      op_callbacks.add_op_callback(1337)

  def testRemoveUnregisteredCallbackLeadsToError(self):
    # [FUNCTION DOC]
    # Scenario: remove a callback that was never added. Asserts a key error whose message says
    # it has not been registered.
    instrument = _NumpyFunctionCallback()
    with self.assertRaisesRegex(KeyError, r"has not been registered"):
      op_callbacks.remove_op_callback(instrument.callback)

  def testRemovingCallbackTwiceLeadsToError(self):
    # [FUNCTION DOC]
    # Scenario: add a callback, remove it, then remove it again. Asserts the second removal
    # raises a key error saying it has not been registered.
    instrument = _NumpyFunctionCallback()
    op_callbacks.add_op_callback(instrument.callback)
    op_callbacks.remove_op_callback(instrument.callback)
    with self.assertRaisesRegex(KeyError, r"has not been registered"):
      op_callbacks.remove_op_callback(instrument.callback)

  def testOverridingWithWrongNumberOfTensorOutputsErrors(self):
    # [FUNCTION DOC]
    # Scenario: register a callback that returns two tensors for every single-output op, then
    # call a traced log-of-one-plus-x function. Asserts a value error whose message reports two
    # returned tensors not matching the original count of one.
    def wrong_outputs_callback(op_type,
                               inputs,
                               attrs,
                               outputs,
                               op_name=None,
                               graph=None):
      # [FUNCTION DOC]
      # Test callback: ignores all but outputs and returns a pair made of the first output and
      # its negation, so the count is always two (wrong for single-output ops).
      del op_type, inputs, attrs, op_name, graph  # Unused.
      return outputs[0], math_ops.negative(outputs[0])

    @def_function.function
    def log1p(x):
      # [FUNCTION DOC]
      # Traced function: returns the natural log of one plus the input.
      return math_ops.log(1.0 + x)

    x = constant_op.constant(3.0)
    op_callbacks.add_op_callback(wrong_outputs_callback)
    with self.assertRaisesRegex(
        ValueError,
        r"returned 2 tensors, .* does not match .* \(1\)"):
      log1p(x)


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
