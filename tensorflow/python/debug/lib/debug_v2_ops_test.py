# [MODULE DOC]
# Tests for the internal ops used by the second-generation tfdbg: the debug-identity op that
# writes tensors to a debug-events dump directory (with circular-buffer behavior, control flow,
# several dump roots, and with the events writer created before or after the op runs), the
# numeric-summary op in its several tensor-debug modes (reduced inf/nan, curt health, concise
# health, shape, full health), and the numeric-check op's error messages for infinities and
# not-a-number values. Test classes derive from the dumping-callback test base or the generic
# framework test case. When run as a script it enables eager execution before starting tests.
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
"""Test for the internal ops used by tfdbg v2."""

import os

import numpy as np

from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.debug.lib import debug_events_reader
from tensorflow.python.debug.lib import debug_events_writer
from tensorflow.python.debug.lib import dumping_callback_test_lib
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_debug_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


class DebugIdentityV2OpTest(dumping_callback_test_lib.DumpingCallbackTestBase):
  # [CLASS DOC]
  # Tests for the version-2 debug-identity op when a debug-events writer is created before the
  # op runs. Fixture: a writer on the temporary dump root with a small circular buffer (size
  # four) and a fixed run id, closed at teardown. Tests read the written files with an events
  # reader and check the recorded graph-execution traces.
  """Tests for DebugIdentityV2Op: when DebugEventsWriter is initialized.

  DebugEventsWriter being initialized prior to DebugIdentityV2 ops being invoked
  for the first time is the typical case (e.g., tfdbg2 running on a local
  machine with only local devices.)
  """

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup: calls the parent setUp, sets the circular buffer size to four, stores a run
    # id string, and creates a debug-events writer for the dump root with that run id and buffer
    # size (creating files on disk).
    super(DebugIdentityV2OpTest, self).setUp()
    # Testing using a small circular-buffer size.
    self.circular_buffer_size = 4
    self.tfdbg_run_id = "test_tfdbg_run"
    self.writer = debug_events_writer.DebugEventsWriter(
        self.dump_root, self.tfdbg_run_id, self.circular_buffer_size)

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture cleanup: closes the debug-events writer, then calls the parent tearDown.
    self.writer.Close()
    super(DebugIdentityV2OpTest, self).tearDown()

  @test_util.run_in_graph_and_eager_modes
  def testSingleTensorFullTensorDebugModeWithCircularBufferBehavior(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function squares and square-roots its input, writing each result via
    # the debug-identity op in full-tensor mode under two distinct context ids, and is called
    # three times (buffer size / 2 + 1). Asserted: the reader sees a metadata file with positive
    # start time, a version and a version string with the expected prefix; before flushing, the
    # graph-execution-trace file is empty; after flushing, only the last two iterations (four
    # entries, because of the circular buffer of size four) are present, each with positive wall
    # time, the right context id, op name, slot, full-tensor mode and tensor values; no further
    # entries follow.

    @def_function.function
    def write_debug_trace(x):
      # [FUNCTION DOC]
      # Nested compiled function: computes the square and square root of the input, writes each
      # through the debug-identity op to the dump root (different context ids and op names, slot
      # zero, full-tensor mode), and returns their sum.
      square = math_ops.square(x)
      gen_debug_ops.debug_identity_v2(
          square,
          tfdbg_context_id="deadbeaf",
          op_name="Square",
          output_slot=0,
          tensor_debug_mode=debug_event_pb2.TensorDebugMode.FULL_TENSOR,
          debug_urls=["file://%s" % self.dump_root])

      sqrt = math_ops.sqrt(x)
      gen_debug_ops.debug_identity_v2(
          sqrt,
          tfdbg_context_id="beafdead",
          op_name="Sqrt",
          output_slot=0,
          tensor_debug_mode=debug_event_pb2.TensorDebugMode.FULL_TENSOR,
          debug_urls=["file://%s" % self.dump_root])
      return square + sqrt

    x = np.array([3.0, 4.0])
    # Only the graph-execution trace of the last iteration should be written
    # to self.dump_root.
    for _ in range(self.circular_buffer_size // 2 + 1):
      self.assertAllClose(
          write_debug_trace(x), [9.0 + np.sqrt(3.0), 16.0 + 2.0])

    with debug_events_reader.DebugEventsReader(self.dump_root) as reader:
      # Check that the .metadata DebugEvents data file has been created, even
      # before FlushExecutionFiles() is called.
      self.assertGreater(reader.starting_wall_time(), 0)
      self.assertTrue(reader.tensorflow_version())
      self.assertTrue(reader.tfdbg_file_version().startswith("debug.Event"))

      graph_trace_iter = reader.graph_execution_traces_iterators()[0]
      # Before FlushExecutionFiles() is called, the .graph_execution_traces file
      # ought to be empty.
      with self.assertRaises(StopIteration):
        next(graph_trace_iter)

      # Flush the circular buffer.
      self.writer.FlushExecutionFiles()
      graph_trace_iter = reader.graph_execution_traces_iterators()[0]

      # The circular buffer has a size of 4. So only the data from the
      # last two iterations should have been written to self.dump_root.
      for _ in range(2):
        debug_event = next(graph_trace_iter).debug_event
        self.assertGreater(debug_event.wall_time, 0)
        trace = debug_event.graph_execution_trace
        self.assertEqual(trace.tfdbg_context_id, "deadbeaf")
        self.assertEqual(trace.op_name, "Square")
        self.assertEqual(trace.output_slot, 0)
        self.assertEqual(trace.tensor_debug_mode,
                         debug_event_pb2.TensorDebugMode.FULL_TENSOR)
        tensor_value = tensor_util.MakeNdarray(trace.tensor_proto)
        self.assertAllClose(tensor_value, [9.0, 16.0])

        debug_event = next(graph_trace_iter).debug_event
        self.assertGreater(debug_event.wall_time, 0)
        trace = debug_event.graph_execution_trace
        self.assertEqual(trace.tfdbg_context_id, "beafdead")
        self.assertEqual(trace.op_name, "Sqrt")
        self.assertEqual(trace.output_slot, 0)
        self.assertEqual(trace.tensor_debug_mode,
                         debug_event_pb2.TensorDebugMode.FULL_TENSOR)
        tensor_value = tensor_util.MakeNdarray(trace.tensor_proto)
        self.assertAllClose(tensor_value, [np.sqrt(3.0), 2.0])

      # Only the graph-execution trace of the last iteration should be written
      # to self.dump_root.
      with self.assertRaises(StopIteration):
        next(graph_trace_iter)

  @test_util.run_in_graph_and_eager_modes
  def testControlFlow(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function runs a Collatz-style loop starting at ten, writing the loop
    # variable each iteration via the debug-identity op. After flushing, the reader's graph
    # execution traces are consumed until exhaustion. Asserted: wall times strictly increase,
    # each trace has the expected context id, op name, slot and full-tensor mode, and because of
    # the circular buffer only the last four values of the sequence (sixteen, eight, four, two)
    # were recorded.

    @def_function.function
    def collatz(x):
      # [FUNCTION DOC]
      # Nested compiled function: while the input is greater than one, increments a counter,
      # writes the current value through the debug-identity op, and replaces the value by half
      # if even or three times it plus one if odd. Returns the iteration counter.
      counter = constant_op.constant(0, dtype=dtypes.int32)
      while math_ops.greater(x, 1):
        counter = counter + 1
        gen_debug_ops.debug_identity_v2(
            x,
            tfdbg_context_id="deadbeaf",
            op_name="x",
            output_slot=0,
            tensor_debug_mode=debug_event_pb2.TensorDebugMode.FULL_TENSOR,
            debug_urls=["file://%s" % self.dump_root])
        if math_ops.equal(x % 2, 0):
          x = math_ops.div(x, 2)
        else:
          x = x * 3 + 1
      return counter

    x = constant_op.constant(10, dtype=dtypes.int32)
    self.evaluate(collatz(x))

    self.writer.FlushExecutionFiles()
    with debug_events_reader.DebugEventsReader(self.dump_root) as reader:
      graph_trace_iter = reader.graph_execution_traces_iterators()[0]
      try:
        x_values = []
        timestamp = 0
        while True:
          debug_event = next(graph_trace_iter).debug_event
          self.assertGreater(debug_event.wall_time, timestamp)
          timestamp = debug_event.wall_time
          trace = debug_event.graph_execution_trace
          self.assertEqual(trace.tfdbg_context_id, "deadbeaf")
          self.assertEqual(trace.op_name, "x")
          self.assertEqual(trace.output_slot, 0)
          self.assertEqual(trace.tensor_debug_mode,
                           debug_event_pb2.TensorDebugMode.FULL_TENSOR)
          x_values.append(int(tensor_util.MakeNdarray(trace.tensor_proto)))
      except StopIteration:
        pass

      # Due to the circular buffer, only the last 4 iterations of
      # [10, 5, 16, 8, 4, 2] should have been written.
      self.assertAllEqual(x_values, [16, 8, 4, 2])

  @test_util.run_in_graph_and_eager_modes
  def testTwoDumpRoots(self):
    # [FUNCTION DOC]
    # Scenario: a second writer is created on a sub-directory of the dump root, and a compiled
    # function writes one squared tensor to both dump roots through a single debug-identity op
    # with two URLs. After flushing both writers and closing the second, each root is read.
    # Asserted: each has exactly one trace with the expected context id, empty op name,
    # full-tensor mode and values, and no further entries.
    another_dump_root = os.path.join(self.dump_root, "another")
    another_debug_url = "file://%s" % another_dump_root
    another_writer = debug_events_writer.DebugEventsWriter(
        another_dump_root, "test_tfdbg_run")

    @def_function.function
    def write_debug_trace(x):
      # [FUNCTION DOC]
      # Nested compiled function: squares the input, writes it via the debug-identity op to both
      # dump roots in full-tensor mode, and returns the square plus one.
      # DebugIdentityV2 is a stateful op. It ought to be included by auto
      # control dependency.
      square = math_ops.square(x)
      gen_debug_ops.debug_identity_v2(
          square,
          tfdbg_context_id="deadbeaf",
          tensor_debug_mode=debug_event_pb2.TensorDebugMode.FULL_TENSOR,
          debug_urls=["file://%s" % self.dump_root, another_debug_url])
      return square + 1.0

    x = np.array([3.0, 4.0])
    self.assertAllClose(write_debug_trace(x), np.array([10.0, 17.0]))
    self.writer.FlushExecutionFiles()
    another_writer.FlushExecutionFiles()
    another_writer.Close()

    for debug_root in (self.dump_root, another_dump_root):
      with debug_events_reader.DebugEventsReader(debug_root) as reader:
        graph_trace_iter = reader.graph_execution_traces_iterators()[0]

        debug_event = next(graph_trace_iter).debug_event
        trace = debug_event.graph_execution_trace
        self.assertEqual(trace.tfdbg_context_id, "deadbeaf")
        self.assertEqual(trace.op_name, "")
        self.assertEqual(trace.tensor_debug_mode,
                         debug_event_pb2.TensorDebugMode.FULL_TENSOR)
        tensor_value = tensor_util.MakeNdarray(trace.tensor_proto)
        self.assertAllClose(tensor_value, [9.0, 16.0])

        with self.assertRaises(StopIteration):
          next(graph_trace_iter)


class DebugIdentityV2OpUninitializedWriterTest(
    dumping_callback_test_lib.DumpingCallbackTestBase):
  # [CLASS DOC]
  # Tests for the version-2 debug-identity op when no debug-events writer exists yet at the dump
  # root when the op first runs (as on a remote worker). Uses the dump root fixture of the base
  # class.
  """Tests for DebugIdentityV2Op: when DebugEventsWriter is not initialized.

  This case can occur when DebugIdentityV2Ops are running on a remote
  TensorFlow server (e.g., a TPU worker).
  """

  @test_util.run_in_graph_and_eager_modes
  def testInvokingDebugIdentityV2OpBeforeCreatingDebugEventsWriterWorks(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function writes a squared tensor through the debug-identity op with
    # an explicit circular buffer size of three, and is called six times (twice the buffer size)
    # before any events writer exists for the dump root. Then a writer with the same buffer size
    # is created and both non-execution and execution files are flushed. Asserted: each call
    # returns the square of its input; the reader finds exactly buffer-size traces, and they hold
    # the squares of the last three inputs.
    circular_buffer_size = 3

    @def_function.function
    def write_debug_trace(x):
      # [FUNCTION DOC]
      # Nested compiled function: squares the input, writes it via the debug-identity op to the
      # dump root with the given circular buffer size, and returns the square.
      # DebugIdentityV2 is a stateful op. It ought to be included by auto
      # control dependency.
      square = math_ops.square(x)
      gen_debug_ops.debug_identity_v2(
          square,
          tfdbg_context_id="deadbeaf",
          op_name="Square",
          output_slot=0,
          tensor_debug_mode=debug_event_pb2.TensorDebugMode.FULL_TENSOR,
          debug_urls=["file://%s" % self.dump_root],
          circular_buffer_size=circular_buffer_size)
      return square

    # The DebugIdentityV2 ops are invokes *before* a DebugEventsWriter at the
    # same dump root is created.
    for i in range(circular_buffer_size * 2):
      self.assertAllClose(
          write_debug_trace(np.array([i]).astype(np.float32)), [i**2.0])
    writer = debug_events_writer.DebugEventsWriter(self.dump_root,
                                                   "test_tfdbg_run",
                                                   circular_buffer_size)
    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugEventsReader(self.dump_root) as reader:
      graph_trace_iter = reader.graph_execution_traces_iterators()[0]
      graph_execution_traces = []
      while True:
        try:
          graph_execution_traces.append(
              next(graph_trace_iter).debug_event.graph_execution_trace)
        except StopIteration:
          break
      self.assertLen(graph_execution_traces, circular_buffer_size)
      for i in range(circular_buffer_size):
        self.assertAllClose(
            tensor_util.MakeNdarray(graph_execution_traces[i].tensor_proto),
            [(i + circular_buffer_size)**2.0])


class DebugNumericSummaryV2Test(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the version-2 numeric-summary debug op in its tensor-debug modes (reduced inf/nan
  # in three slots, curt health, concise health, shape, full health), checking summary values
  # for small and large tensors, tensor id limits, consistency across repeated runs and
  # determinism errors, plus tests of the numeric-check op's error messages.

  @test_util.run_in_graph_and_eager_modes
  def testDebugNumericSummaryV2OpReduceInfNanThreeSlots(self):
    # [FUNCTION DOC]
    # Scenario: the numeric-summary op in reduced inf/nan three-slot mode is evaluated on empty,
    # scalar, normal, infinite and not-a-number inputs, and on large zero-filled arrays of
    # several float types with one special value inserted. Asserted: the three-element result
    # holds the most negative infinity indicator, positive infinity indicator and not-a-number
    # indicator, with zeros where absent.

    def debug_summary(x):
      # [FUNCTION DOC]
      # Nested helper: evaluates the numeric-summary op on the input in reduced inf/nan
      # three-slot mode and returns the resulting values.
      return self.evaluate(
          gen_debug_ops.debug_numeric_summary_v2(
              x,
              tensor_debug_mode=(
                  debug_event_pb2.TensorDebugMode.REDUCE_INF_NAN_THREE_SLOTS)))

    self.assertAllEqual(
        debug_summary(constant_op.constant([])), [0.0, 0.0, 0.0])
    self.assertAllEqual(
        debug_summary(constant_op.constant(42.0)), [0.0, 0.0, 0.0])
    self.assertAllEqual(
        debug_summary(constant_op.constant([3.0, 4.0])), [0.0, 0.0, 0.0])
    self.assertAllEqual(
        debug_summary(constant_op.constant(np.array([3.0, -np.inf]))),
        [-np.inf, 0.0, 0.0])
    self.assertAllEqual(
        debug_summary(constant_op.constant(np.array([[0, 0], [np.nan, 0]]))),
        [0.0, 0.0, np.nan])
    self.assertAllEqual(
        debug_summary(
            constant_op.constant(np.array([[0, 0], [np.nan, np.inf]]))),
        [0.0, np.inf, np.nan])
    self.assertAllEqual(
        debug_summary(
            constant_op.constant(np.array([[0, np.inf], [np.nan, -np.inf]]))),
        [-np.inf, np.inf, np.nan])

    x = np.zeros([100, 100], dtype=np.float16)
    x[32, 47] = np.nan
    self.assertAllEqual(
        debug_summary(constant_op.constant(x)), [0.0, 0.0, np.nan])
    x = np.zeros([97, 97], dtype=np.float32)
    x[50, 83] = -np.inf
    self.assertAllEqual(
        debug_summary(constant_op.constant(x)), [-np.inf, 0.0, 0.0])
    x[1, 41] = np.nan
    self.assertAllEqual(
        debug_summary(constant_op.constant(x)), [-np.inf, 0.0, np.nan])
    x = np.zeros([9701], dtype=np.float64)
    x[9700] = np.nan
    self.assertAllEqual(
        debug_summary(constant_op.constant(x)), [0.0, 0.0, np.nan])

  @test_util.run_in_graph_and_eager_modes
  def testDebugNumericSummaryV2OpLargeTensorIDError(self):
    # [FUNCTION DOC]
    # Scenario: for the curt health, concise health and shape modes, the op is evaluated with a
    # tensor id equal to two to the power of 53 (the maximum allowed), then with that id plus
    # one. Asserted: the maximum id succeeds; the larger id raises an invalid-argument error in
    # every mode.
    modes = [
        debug_event_pb2.TensorDebugMode.CURT_HEALTH,
        debug_event_pb2.TensorDebugMode.CONCISE_HEALTH,
        debug_event_pb2.TensorDebugMode.SHAPE,
    ]
    # Maximum allowed tensor_id
    tensor_id = np.power(2, 53, dtype=np.int64)
    for mode in modes:
      self.evaluate(
          gen_debug_ops.debug_numeric_summary_v2(
              constant_op.constant(42.0),
              tensor_debug_mode=mode,
              tensor_id=tensor_id,
              output_dtype=dtypes.float64))
    # Incrementing by one should error
    tensor_id += 1
    for mode in modes:
      with self.assertRaises(errors.InvalidArgumentError):
        self.evaluate(
            gen_debug_ops.debug_numeric_summary_v2(
                constant_op.constant(42.0),
                tensor_debug_mode=mode,
                tensor_id=tensor_id,
                output_dtype=dtypes.float64))

  @test_util.run_in_graph_and_eager_modes
  def testDebugNumericSummaryV2OpCurtHealthValuesSmall(self):
    # [FUNCTION DOC]
    # Scenario: curt-health mode on small inputs (empty, scalar, normal values, and arrays with
    # negative infinity, not-a-number or both). Asserted: the result is the tensor id followed by
    # zero if all values are finite or one if any is infinite or not-a-number.

    def debug_summary(x):
      # [FUNCTION DOC]
      # Nested helper: evaluates the numeric-summary op on the input in curt-health mode using
      # the input's own id as tensor id (double-precision output). Returns a pair of the summary
      # values and the tensor id.
      return self.evaluate(
          gen_debug_ops.debug_numeric_summary_v2(
              x,
              tensor_debug_mode=(debug_event_pb2.TensorDebugMode.CURT_HEALTH),
              tensor_id=x._id,
              output_dtype=dtypes.float64)), x._id

    tensor, tensor_id = debug_summary(constant_op.constant([]))
    self.assertAllEqual(tensor, [tensor_id, 0.0])

    tensor, tensor_id = debug_summary(constant_op.constant(42.0))
    self.assertAllEqual(tensor, [tensor_id, 0.0])

    tensor, tensor_id = debug_summary(constant_op.constant([3.0, 4.0]))
    self.assertAllEqual(tensor, [tensor_id, 0.0])

    tensor, tensor_id = debug_summary(
        constant_op.constant(np.array([3.0, -np.inf])))
    self.assertAllEqual(tensor, [tensor_id, 1.0])

    tensor, tensor_id = debug_summary(
        constant_op.constant(np.array([[0, 0], [np.nan, 0]])))
    self.assertAllEqual(tensor, [tensor_id, 1.0])

    tensor, tensor_id = debug_summary(
        constant_op.constant(np.array([[0, 0], [np.nan, np.inf]])))
    self.assertAllEqual(tensor, [tensor_id, 1.0])

    tensor, tensor_id = debug_summary(
        constant_op.constant(np.array([[0, np.inf], [np.nan, -np.inf]])))
    self.assertAllEqual(tensor, [tensor_id, 1.0])

  @test_util.run_in_graph_and_eager_modes
  def testDebugNumericSummaryV2OpCurtHealthValuesLarge(self):
    # [FUNCTION DOC]
    # Scenario: curt-health mode on large zero-filled arrays of different float types and shapes,
    # each with one or two special values inserted. Asserted: every summary is the tensor id
    # followed by one (a problem was found).

    def debug_summary(x):
      # [FUNCTION DOC]
      # Nested helper: evaluates the numeric-summary op on the input in curt-health mode using
      # the input's id as tensor id (double-precision output). Returns a pair of the summary
      # values and the tensor id.
      return self.evaluate(
          gen_debug_ops.debug_numeric_summary_v2(
              x,
              tensor_debug_mode=(debug_event_pb2.TensorDebugMode.CURT_HEALTH),
              tensor_id=x._id,
              output_dtype=dtypes.float64)), x._id

    x = np.zeros([100, 100], dtype=np.float16)
    x[32, 47] = np.nan
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    self.assertAllEqual(tensor, [tensor_id, 1.0])
    x = np.zeros([97, 97], dtype=np.float32)
    x[50, 83] = -np.inf
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    self.assertAllEqual(tensor, [tensor_id, 1.0])
    x[1, 41] = np.nan
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    self.assertAllEqual(tensor, [tensor_id, 1.0])
    x = np.zeros([9701], dtype=np.float64)
    x[9700] = np.nan
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    self.assertAllEqual(tensor, [tensor_id, 1.0])

  @test_util.run_in_graph_and_eager_modes
  def testDebugNumericSummaryV2OpCurtHealthConsistency(self):
    # [FUNCTION DOC]
    # Scenario: the curt-health summary is computed twice for the same constant tensor, for three
    # different large tensors. Asserted: both runs give identical summary values and identical
    # tensor ids.

    def debug_summary(x):
      # [FUNCTION DOC]
      # Nested helper: evaluates the numeric-summary op on the input in curt-health mode using
      # the input's id as tensor id (double-precision output). Returns a pair of the summary
      # values and the tensor id.
      return self.evaluate(
          gen_debug_ops.debug_numeric_summary_v2(
              x,
              tensor_debug_mode=(debug_event_pb2.TensorDebugMode.CURT_HEALTH),
              tensor_id=x._id,
              output_dtype=dtypes.float64)), x._id

    x = np.zeros([100, 100], dtype=np.float16)
    x[43, 99] = np.nan
    c = constant_op.constant(x)
    tensor_1, tensor_id_1 = debug_summary(c)
    tensor_2, tensor_id_2 = debug_summary(c)
    self.assertAllEqual(tensor_1, tensor_2)
    self.assertEqual(tensor_id_1, tensor_id_2)

    x = np.zeros([100, 100, 50], dtype=np.float64)
    x[0, 0, 1] = np.inf
    c = constant_op.constant(x)
    tensor_1, tensor_id_1 = debug_summary(c)
    tensor_2, tensor_id_2 = debug_summary(c)
    self.assertAllEqual(tensor_1, tensor_2)
    self.assertEqual(tensor_id_1, tensor_id_2)

    c = constant_op.constant(np.ones((100, 200), np.double))
    tensor_1, tensor_id_1 = debug_summary(c)
    tensor_2, tensor_id_2 = debug_summary(c)
    self.assertAllEqual(tensor_1, tensor_2)
    self.assertEqual(tensor_id_1, tensor_id_2)

  @test_util.run_in_graph_and_eager_modes
  def testDebugNumericSummaryV2OpDeterminism(self):
    # [FUNCTION DOC]
    # Scenario: with deterministic ops enabled and a large zero-filled tensor, the op is run in
    # concise-health and full-health modes. Only if a GPU is present is anything asserted: an
    # unimplemented-determinism error with a message naming the mode is expected. On machines
    # without a GPU the test performs no assertions.
    x = np.zeros([100, 100, 50], dtype=np.float64)
    x = constant_op.constant(x)
    modes = (
        debug_event_pb2.TensorDebugMode.CONCISE_HEALTH,
        debug_event_pb2.TensorDebugMode.FULL_HEALTH,
        )
    for mode in modes:
      debug_mode = debug_event_pb2.TensorDebugMode.Name(mode)
      with test_util.deterministic_ops():
        if test_util.config.list_physical_devices("GPU"):
          with self.assertRaisesRegex(
              errors_impl.UnimplementedError, "Determinism is not yet "
              "supported for DebugNumericSummaryV2 when tensor_debug_mode is "
              + debug_mode + "."):
            self.evaluate(
                gen_debug_ops.debug_numeric_summary_v2(
                    x,
                    tensor_debug_mode=mode,
                    tensor_id=x._id,
                    output_dtype=dtypes.float64))

  @test_util.run_in_graph_and_eager_modes
  def testDebugNumericSummaryV2OpConciseHealthSmall(self):
    # [FUNCTION DOC]
    # Scenario: concise-health mode on small inputs (empty, scalar, normal values, negative
    # infinity, not-a-number, both infinities). Asserted: the result is the tensor id, element
    # count, negative-infinity count, positive-infinity count and not-a-number count.

    def debug_summary(x):
      # [FUNCTION DOC]
      # Nested helper: evaluates the numeric-summary op on the input in concise-health mode using
      # the input's id as tensor id (double-precision output). Returns a pair of the summary
      # values and the tensor id.
      return self.evaluate(
          gen_debug_ops.debug_numeric_summary_v2(
              x,
              tensor_debug_mode=(
                  debug_event_pb2.TensorDebugMode.CONCISE_HEALTH),
              tensor_id=x._id,
              output_dtype=dtypes.float64)), x._id

    tensor, tensor_id = debug_summary(constant_op.constant([]))
    self.assertAllEqual(tensor, [tensor_id, 0.0, 0.0, 0.0, 0.0])

    tensor, tensor_id = debug_summary(constant_op.constant(42.0))
    self.assertAllEqual(tensor, [tensor_id, 1.0, 0.0, 0.0, 0.0])

    tensor, tensor_id = debug_summary(constant_op.constant([3.0, 4.0]))
    self.assertAllEqual(tensor, [tensor_id, 2.0, 0.0, 0.0, 0.0])

    tensor, tensor_id = debug_summary(
        constant_op.constant(np.array([3.0, -np.inf])))
    self.assertAllEqual(tensor, [tensor_id, 2.0, 1.0, 0.0, 0.0])

    tensor, tensor_id = debug_summary(
        constant_op.constant(np.array([[0, 0], [np.nan, 0]])))
    self.assertAllEqual(tensor, [tensor_id, 4.0, 0.0, 0.0, 1.0])

    tensor, tensor_id = debug_summary(
        constant_op.constant(np.array([[0, 0], [np.nan, np.inf]])))
    self.assertAllEqual(tensor, [tensor_id, 4.0, 0.0, 1.0, 1.0])

    tensor, tensor_id = debug_summary(
        constant_op.constant(np.array([[0, np.inf], [np.nan, -np.inf]])))
    self.assertAllEqual(tensor, [tensor_id, 4.0, 1.0, 1.0, 1.0])

  @test_util.run_in_graph_and_eager_modes
  def testDebugNumericSummaryV2OpConciseHealthLarge(self):
    # [FUNCTION DOC]
    # Scenario: concise-health mode on large zero-filled arrays of several float types with rows
    # or slices of not-a-number or negative infinity inserted. Asserted: the element count and
    # the per-category counts match the inserted values.

    def debug_summary(x):
      # [FUNCTION DOC]
      # Nested helper: evaluates the numeric-summary op on the input in concise-health mode using
      # the input's id as tensor id (double-precision output). Returns a pair of the summary
      # values and the tensor id.
      return self.evaluate(
          gen_debug_ops.debug_numeric_summary_v2(
              x,
              tensor_debug_mode=(
                  debug_event_pb2.TensorDebugMode.CONCISE_HEALTH),
              tensor_id=x._id,
              output_dtype=dtypes.float64)), x._id

    x = np.zeros([100, 100], dtype=np.float16)
    x[32, :] = np.nan
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    self.assertAllEqual(tensor, [tensor_id, 10000.0, 0.0, 0.0, 100.0])
    x = np.zeros([97, 97], dtype=np.float32)
    x[50, 83:85] = -np.inf
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    self.assertAllEqual(tensor, [tensor_id, 97 * 97, 2.0, 0.0, 0.0])
    x[1:9, 41] = np.nan
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    self.assertAllEqual(tensor, [tensor_id, 97 * 97, 2.0, 0.0, 8.0])
    x = np.zeros([9701], dtype=np.float64)
    x[9700] = np.nan
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    self.assertAllEqual(tensor, [tensor_id, 9701, 0.0, 0.0, 1.0])

  @test_util.run_in_graph_and_eager_modes
  def testDebugNumericSummaryV2OpConciseHealthConsistency(self):
    # [FUNCTION DOC]
    # Scenario: the concise-health summary is computed twice for the same constant tensor, for
    # two different large tensors. Asserted: both runs give identical values and tensor ids.

    def debug_summary(x):
      # [FUNCTION DOC]
      # Nested helper: evaluates the numeric-summary op on the input in concise-health mode using
      # the input's id as tensor id (double-precision output). Returns a pair of the summary
      # values and the tensor id.
      return self.evaluate(
          gen_debug_ops.debug_numeric_summary_v2(
              x,
              tensor_debug_mode=(
                  debug_event_pb2.TensorDebugMode.CONCISE_HEALTH),
              tensor_id=x._id,
              output_dtype=dtypes.float64)), x._id

    # Assert the same op is returns a consistent value
    x = np.zeros([100, 100], dtype=np.float16)
    x[3, 4] = -np.inf
    c = constant_op.constant(x)
    tensor_1, tensor_id_1 = debug_summary(c)
    tensor_2, tensor_id_2 = debug_summary(c)
    self.assertAllEqual(tensor_1, tensor_2)
    self.assertEqual(tensor_id_1, tensor_id_2)

    c = constant_op.constant(np.ones((100, 200), np.double))
    tensor_1, tensor_id_1 = debug_summary(c)
    tensor_2, tensor_id_2 = debug_summary(c)
    self.assertAllEqual(tensor_1, tensor_2)
    self.assertEqual(tensor_id_1, tensor_id_2)

  @test_util.run_in_graph_and_eager_modes
  def testDebugNumericSummaryV2OpShapeEmpty(self):
    # [FUNCTION DOC]
    # Scenario: shape mode on a scalar. Asserted: the ten-element summary holds the tensor id,
    # dtype code one, rank zero, element count one, and zeros for the unused dimension slots.

    def debug_summary(x):
      # [FUNCTION DOC]
      # Nested helper: evaluates the numeric-summary op on the input in shape mode using the
      # input's id as tensor id (double-precision output). Returns a pair of the summary values
      # and the tensor id.
      return self.evaluate(
          gen_debug_ops.debug_numeric_summary_v2(
              x,
              tensor_debug_mode=(debug_event_pb2.TensorDebugMode.SHAPE),
              tensor_id=x._id,
              output_dtype=dtypes.float64)), x._id

    tensor, tensor_id = debug_summary(constant_op.constant(0.0))
    self.assertAllEqual(
        tensor, [tensor_id, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

  @test_util.run_in_graph_and_eager_modes
  def testDebugNumericSummaryV2OpShapeSmall(self):
    # [FUNCTION DOC]
    # Scenario: shape mode on a matrix, a six-dimensional half-precision array with a
    # not-a-number, a vector, and an empty tensor. Asserted: the summary contains the tensor id,
    # a dtype code, the rank, the element count and the dimension sizes padded with zeros.

    def debug_summary(x):
      # [FUNCTION DOC]
      # Nested helper: evaluates the numeric-summary op on the input in shape mode using the
      # input's id as tensor id (double-precision output). Returns a pair of the summary values
      # and the tensor id.
      return self.evaluate(
          gen_debug_ops.debug_numeric_summary_v2(
              x,
              tensor_debug_mode=(debug_event_pb2.TensorDebugMode.SHAPE),
              tensor_id=x._id,
              output_dtype=dtypes.float64)), x._id

    x = np.zeros([3, 4], dtype=np.float32)
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    self.assertAllEqual(
        tensor, [tensor_id, 1.0, 2.0, 12.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0])

    x = np.ones([1, 2, 3, 4, 5, 6], dtype=np.float16)
    x[0, 1, 2, 2, 2, 2] = np.nan
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    self.assertAllEqual(
        tensor,
        [tensor_id, 19, 6.0, 2 * 3 * 4 * 5 * 6, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    x = np.zeros([2], dtype=np.float32)
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    self.assertAllEqual(
        tensor, [tensor_id, 1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    tensor, tensor_id = debug_summary(constant_op.constant([]))
    self.assertAllEqual(
        tensor, [tensor_id, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

  @test_util.run_in_graph_and_eager_modes
  def testDebugNumericSummaryV2OpShapeLarge(self):
    # [FUNCTION DOC]
    # Scenario: shape mode on a seven-dimensional double-precision array of ones. Asserted: the
    # summary records rank seven and the element count, and only the last six dimension sizes
    # (the dimension slots are limited to six).

    def debug_summary(x):
      # [FUNCTION DOC]
      # Nested helper: evaluates the numeric-summary op on the input in shape mode using the
      # input's id as tensor id (double-precision output). Returns a pair of the summary values
      # and the tensor id.
      return self.evaluate(
          gen_debug_ops.debug_numeric_summary_v2(
              x,
              tensor_debug_mode=(debug_event_pb2.TensorDebugMode.SHAPE),
              tensor_id=x._id,
              output_dtype=dtypes.float64)), x._id

    x = np.ones([1, 2, 3, 4, 5, 6, 7], dtype=np.double)
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    self.assertAllEqual(tensor, [
        tensor_id, 2.0, 7.0, 2 * 3 * 4 * 5 * 6 * 7, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0
    ])

  @test_util.run_in_graph_and_eager_modes
  def testDebugNumericSummaryV2OpFullHealthSmall(self):
    # [FUNCTION DOC]
    # Scenario: full-health mode on small inputs of various float types (empty, scalar, vector,
    # negative infinity, not-a-number, both infinities). Asserted: the eleven-element summary
    # holds the tensor id, a placeholder device value, dtype code, rank, element count, and the
    # counts of negative infinities, positive infinities, not-a-numbers, negatives, zeros and
    # positives.

    def debug_summary(x):
      # [FUNCTION DOC]
      # Nested helper: evaluates the numeric-summary op on the input in full-health mode using
      # the input's id as tensor id (double-precision output). Returns a pair of the summary
      # values and the tensor id.
      return self.evaluate(
          gen_debug_ops.debug_numeric_summary_v2(
              x,
              tensor_debug_mode=(debug_event_pb2.TensorDebugMode.FULL_HEALTH),
              tensor_id=x._id,
              output_dtype=dtypes.float64)), x._id

    tensor, tensor_id = debug_summary(constant_op.constant([]))
    expected = [tensor_id, -1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    self.assertAllEqual(tensor, expected)

    tensor, tensor_id = debug_summary(constant_op.constant(42.0))
    expected = [tensor_id, -1, 1, 0, 1, 0, 0, 0, 0, 0, 1]
    self.assertAllEqual(tensor, expected)

    tensor, tensor_id = debug_summary(constant_op.constant([3.0, 4.0]))
    expected = [tensor_id, -1, 1, 1, 2, 0, 0, 0, 0, 0, 2]
    self.assertAllEqual(tensor, expected)

    tensor, tensor_id = debug_summary(
        constant_op.constant(np.array([3, -np.inf], dtype=np.float32)))
    expected = [tensor_id, -1, 1, 1, 2, 1, 0, 0, 0, 0, 1]
    self.assertAllEqual(tensor, expected)

    tensor, tensor_id = debug_summary(
        constant_op.constant(np.array([[0, 0], [np.nan, 0]], dtype=np.float64)))
    expected = [tensor_id, -1, 2, 2, 4, 0, 0, 1, 0, 3, 0]
    self.assertAllEqual(tensor, expected)

    tensor, tensor_id = debug_summary(
        constant_op.constant(
            np.array([[0, 0], [np.nan, np.inf]], dtype=np.float16)))
    expected = [tensor_id, -1, 19, 2, 4, 0, 1, 1, 0, 2, 0]
    self.assertAllEqual(tensor, expected)

    tensor, tensor_id = debug_summary(
        constant_op.constant(
            np.array([[0, np.inf], [np.nan, -np.inf]], dtype=np.float32)))
    expected = [tensor_id, -1, 1, 2, 4, 1, 1, 1, 0, 1, 0]
    self.assertAllEqual(tensor, expected)

  @test_util.run_in_graph_and_eager_modes
  def testDebugNumericSummaryV2OpFullHealthLarge(self):
    # [FUNCTION DOC]
    # Scenario: full-health mode on large arrays of half, single and double precision filled with
    # a mix of zeros, infinities, not-a-numbers, negatives and positives. Asserted: the summary
    # equals the tensor id, a placeholder, a dtype code, followed by rank, size and category
    # counts computed independently by a local helper.

    def debug_summary(x):
      # [FUNCTION DOC]
      # Nested helper: evaluates the numeric-summary op on the input in full-health mode using
      # the input's id as tensor id (double-precision output). Returns a pair of the summary
      # values and the tensor id.
      return self.evaluate(
          gen_debug_ops.debug_numeric_summary_v2(
              x,
              tensor_debug_mode=(debug_event_pb2.TensorDebugMode.FULL_HEALTH),
              tensor_id=x._id,
              output_dtype=dtypes.float64)), x._id

    def tensor_counts(arr):
      # [FUNCTION DOC]
      # Nested helper computing the expected summary tail on the host: takes an array and returns
      # a list of eight numbers: rank, total element count, then counts of negative infinities,
      # positive infinities, not-a-numbers, negative finite values, zeros and positive finite
      # values, found by classifying each element.
      counts = [len(np.shape(arr)), np.size(arr), 0, 0, 0, 0, 0, 0]
      for n in np.ravel(arr):
        if np.isneginf(n):
          counts[2] += 1
        elif np.isposinf(n):
          counts[3] += 1
        elif np.isnan(n):
          counts[4] += 1
        elif n < 0.:
          counts[5] += 1
        elif n == 0.:
          counts[6] += 1
        else:
          counts[7] += 1
      return counts

    x = np.zeros([50, 50], dtype=np.float16)
    x[32, 47] = np.nan
    x[0:4, 3] = np.inf
    x[40:50, 40:50] = 10
    x[3, 20] = -10
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    expected = [tensor_id, -1, 19] + tensor_counts(x)
    self.assertAllEqual(tensor, expected)

    x = np.ones([25, 25, 50], dtype=np.float32) * np.inf
    x[:, :, 1] = np.nan
    x[:, :, 2] = -np.inf
    x[:, :, 3] = -1
    x[:, :, 4] = 0
    x[:, :, 5] = 1
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    expected = [tensor_id, -1, 1] + tensor_counts(x)
    self.assertAllEqual(tensor, expected)
    x[0, 0, 0] = np.nan
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    expected = [
        tensor_id,
        -1,
        1,
    ] + tensor_counts(x)
    self.assertAllEqual(tensor, expected)
    x = np.zeros([9701], dtype=np.float64)
    x[9700] = np.nan
    tensor, tensor_id = debug_summary(constant_op.constant(x))
    expected = [tensor_id, -1, 2] + tensor_counts(x)
    self.assertAllEqual(tensor, expected)

  @test_util.run_in_graph_and_eager_modes
  def testDebugNumericSummaryV2OpFullHealthConsistency(self):
    # [FUNCTION DOC]
    # Scenario: the full-health summary is computed twice for the same constant tensor, for two
    # large tensors with mixed special values. Asserted: both runs give identical values and
    # tensor ids.

    def debug_summary(x):
      # [FUNCTION DOC]
      # Nested helper: evaluates the numeric-summary op on the input in full-health mode using
      # the input's id as tensor id (double-precision output). Returns a pair of the summary
      # values and the tensor id.
      return self.evaluate(
          gen_debug_ops.debug_numeric_summary_v2(
              x,
              tensor_debug_mode=(debug_event_pb2.TensorDebugMode.FULL_HEALTH),
              tensor_id=x._id,
              output_dtype=dtypes.float64)), x._id

    # Assert the same op is returns a consistent value
    x = np.zeros([100, 100], dtype=np.float16)
    x[32, 47] = np.nan
    x[0:4, 3] = np.inf
    x[90:100, 90:100] = 10
    x[3, 20] = -10
    c = constant_op.constant(x)
    tensor_1, tensor_id_1 = debug_summary(c)
    tensor_2, tensor_id_2 = debug_summary(c)
    self.assertAllEqual(tensor_1, tensor_2)
    self.assertEqual(tensor_id_1, tensor_id_2)

    x = np.ones((100, 200, 3, 10), np.double)
    x[1, 30, 2] = 10
    x[5, :, 0, 1] = np.nan
    x[90:100, 150, :, :] = np.inf
    c = constant_op.constant(x)
    tensor_1, tensor_id_1 = debug_summary(c)
    tensor_2, tensor_id_2 = debug_summary(c)
    self.assertAllEqual(tensor_1, tensor_2)
    self.assertEqual(tensor_id_1, tensor_id_2)

  def testCheckNumericsV2OpNegativeAndPositiveInf(self):
    # [FUNCTION DOC]
    # Scenario: in a fresh graph, divides a vector of minus one and one by zeros and passes it
    # through the numeric-check op with a message. Asserted: an invalid-argument error is raised
    # whose text mentions both negative and positive infinity values.
    """Test that CheckNumericsV2 op distinguishes negative and positive infs."""
    with self.session(graph=ops.Graph()):
      t1 = constant_op.constant([-1.0, 1.0])
      t2 = constant_op.constant([0.0, 0.0])
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          r"pass through test.*had -Inf and \+Inf values"):
        self.evaluate(
            array_ops.check_numerics_v2(t1 / t2, message="pass through test"))

  def testCheckNumericsV2OpNegativeAndPositiveInfAndNaN(self):
    # [FUNCTION DOC]
    # Scenario: in a fresh graph, divides minus one, one and zero by zeros and runs the numeric
    # check. Asserted: an invalid-argument error whose text mentions negative infinity, positive
    # infinity and not-a-number values.
    """CheckNumericsV2 op distinguishes - & + infs when nan is present."""
    with self.session(graph=ops.Graph()):
      t1 = constant_op.constant([-1.0, 1.0, 0.0])
      t2 = constant_op.constant([0.0, 0.0, 0.0])
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          r"pass through test.*had -Inf, \+Inf, and NaN values"):
        self.evaluate(
            array_ops.check_numerics_v2(t1 / t2, message="pass through test"))

  def testCheckNumericsV2PositiveInfAndNaN(self):
    # [FUNCTION DOC]
    # Scenario: in a fresh graph, divides zero and one by zeros and runs the numeric check.
    # Asserted: an invalid-argument error whose text mentions positive infinity and
    # not-a-number values (with the sign of the infinity shown).
    """Test that CheckNumericsV2 op shows sign of inf when nan is present."""
    with self.session(graph=ops.Graph()):
      t1 = constant_op.constant([0.0, 1.0])
      t2 = constant_op.constant([0.0, 0.0])
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          r"pass through test.*had \+Inf and NaN values"):
        self.evaluate(
            array_ops.check_numerics_v2(t1 / t2, message="pass through test"))


if __name__ == "__main__":
  ops.enable_eager_execution()
  googletest.main()
