# [MODULE DOC]
# Unit tests for the tfdbg v2 dumping callback. Covers eager and graph execution dumping across
# tensor debug modes, op and dtype filters, nested functions and control flow, repeated enable and
# disable calls, and multi-threaded use, verifying results via the debug events reader. The main
# entry point enables eager execution and runs the test framework.
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
"""Unit tests for tfdbg v2 dumping callback."""

import collections
import os
import shutil
import socket
import tempfile
import threading

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.debug.lib import debug_events_reader
from tensorflow.python.debug.lib import dumping_callback
from tensorflow.python.debug.lib import dumping_callback_test_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


_host_name = socket.gethostname()
_current_file_full_path = os.path.abspath(__file__)


class DumpingCallbackTest(
    dumping_callback_test_lib.DumpingCallbackTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the dumping callback of tfdbg v2 (enable and disable dump functions). Each test
  # uses a fresh temporary dump root created in set-up and removed in tear-down, enables dumping with
  # some configuration, runs eager ops or compiled functions, flushes the writer, and reads the dump
  # back with the debug data reader to verify executions, graphs, traces, source files and errors.
  # Parameterized over tensor debug modes where relevant.

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the parent set-up and creates a fresh temporary directory stored as the dump root.
    super(DumpingCallbackTest, self).setUp()
    self.dump_root = tempfile.mkdtemp()

  def tearDown(self):
    # [FUNCTION DOC]
    # Removes the temporary dump directory if present, disables dumping (reset thread state), then
    # calls the parent tear-down.
    if os.path.isdir(self.dump_root):
      shutil.rmtree(self.dump_root, ignore_errors=True)
    dumping_callback.disable_dump_debug_info()
    super(DumpingCallbackTest, self).tearDown()

  def _verifyStackFrames(self, stack_frames):
    # [FUNCTION DOC]
    # HELPER: shared assertion for the tests. Takes a list of stack frames (file path first) and
    # asserts at least one frame belongs to this test file. Returns nothing.
    """Verify the correctness of the stack frames.

    Currently, it simply asserts that the current file is found in the stack
    frames.
    TODO(cais): Perhaps implement a stricter check later.

    Args:
      stack_frames: The stack frames to verify.
    """
    self.assertTrue([
        frame for frame in stack_frames if frame[0] == _current_file_full_path])

  def _expectedDefaultDeviceName(self):
    # [FUNCTION DOC]
    # HELPER: returns the full device name expected for default op placement: the first GPU under
    # the local job if one exists, otherwise the first CPU.
    gpu_name = test_util.gpu_device_name()
    if gpu_name:
      return "/job:localhost/replica:0/task:0" + gpu_name
    else:
      return "/job:localhost/replica:0/task:0/device:CPU:0"

  def testInvalidTensorDebugModeCausesError(self):
    # [FUNCTION DOC]
    # Scenario: enable dumping with a nonsensical debug mode name. Asserts a value error whose
    # message names the invalid value and lists valid options such as no-tensor.
    with self.assertRaisesRegex(
        ValueError, r"Invalid value in tensor_debug_mode \(\'NONSENSICAL\'\).*"
        r"Valid options.*NO_TENSOR.*"):
      dumping_callback.enable_dump_debug_info(
          self.dump_root, tensor_debug_mode="NONSENSICAL")

  @parameterized.named_parameters(
      ("NoTensor", "NO_TENSOR"),
      ("CurtHealth", "CURT_HEALTH"),
      ("ConciseHealth", "CONCISE_HEALTH"),
      ("Shape", "SHAPE"),
      ("FulHealth", "FULL_HEALTH"),
      ("FullTensor", "FULL_TENSOR"),
  )
  def testEnableDumpDebugInfoLogsTensorDebugModeAsStringName(self,
                                                             tensor_debug_mode):
    # [FUNCTION DOC]
    # Scenario: with the logging info function mocked, enable dumping in each debug mode. Asserts
    # exactly one log message was emitted and it contains the dump root and the debug mode name.
    log_messages = []
    def fake_logging_info(*args):
      # [FUNCTION DOC]
      # HELPER: mock replacement for the logging info function; stores the arguments in the list.
      log_messages.append(args)
    with test.mock.patch.object(
        tf_logging, "info", side_effect=fake_logging_info):
      dumping_callback.enable_dump_debug_info(
          self.dump_root, tensor_debug_mode=tensor_debug_mode)
      self.assertLen(log_messages, 1)
      self.assertIn(self.dump_root, log_messages[0])
      self.assertIn(tensor_debug_mode, log_messages[0])

  def testDisablingTracingCallbackWithoutEnablingFirstIsTolerated(self):
    # [FUNCTION DOC]
    # Scenario: call the disable function without having enabled dumping. Passes if no exception
    # is raised.
    dumping_callback.disable_dump_debug_info()

  @parameterized.named_parameters(
      ("NoTensor", "NO_TENSOR"),
      ("CurtHealth", "CURT_HEALTH"),
      ("ConciseHealth", "CONCISE_HEALTH"),
      ("Shape", "SHAPE"),
      ("FullHealth", "FULL_HEALTH"),
      ("FullTensor", "FULL_TENSOR"),
  )
  def testPureEagerOpExecution(self, tensor_debug_mode):
    # [FUNCTION DOC]
    # Scenario: for every debug mode, run the Collatz sequence from 10 using eager ops. Asserts
    # that the execution file is empty before flushing; afterwards, non-decreasing wall times,
    # expected device names, no graph IDs, input and output tensor IDs, the configured debug mode,
    # per-mode debug tensor values (including full values), host name and stack frames, the exact
    # executed op type sequence, and that no graphs or graph execution traces were recorded.
    """Test dumping data from eager op execution: float32."""

    x = constant_op.constant(10.0)
    zero = constant_op.constant(0.0)
    one = constant_op.constant(1.0)
    two = constant_op.constant(2.0)
    three = constant_op.constant(3.0)
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode=tensor_debug_mode)
    # Use Collatz conjecture as a test case.
    while x > one:
      if math_ops.equal(x % two, zero):
        x = x / two
      else:
        x = x * three + one

    writer.FlushNonExecutionFiles()
    self._readAndCheckMetadataFile()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      # Before FlushExecutionFiles() is called, the .execution file should be
      # empty.
      self.assertFalse(reader.executions())

      # After the flushing, the .execution file should hold the appropriate
      # contents.
      writer.FlushExecutionFiles()
      reader.update()
      executions = reader.executions()
      prev_wall_time = 1
      executed_op_types = []
      tensor_values = collections.defaultdict(lambda: [])
      for execution in executions:
        self.assertGreaterEqual(execution.wall_time, prev_wall_time)
        prev_wall_time = execution.wall_time
        executed_op_types.append(execution.op_type)
        # Check the device name.
        if execution.op_type in ("AddV2", "Mul", "RealDiv"):
          self.assertLen(execution.output_tensor_device_ids, 1)
          self.assertEqual(
              reader.device_name_by_id(execution.output_tensor_device_ids[0]),
              self._expectedDefaultDeviceName(),
              "Unexpected device name from eager op %s" % execution.op_type)

        # No graph IDs should have been logged for eager op executions.
        self.assertFalse(execution.graph_id)
        self.assertTrue(execution.input_tensor_ids)
        self.assertTrue(execution.output_tensor_ids)
        self.assertEqual(
            debug_event_pb2.TensorDebugMode.keys()[execution.tensor_debug_mode],
            tensor_debug_mode)
        if tensor_debug_mode == "NO_TENSOR":
          # Due to the NO_TENSOR tensor debug mode, tensor_protos ought to
          # be empty.
          self.assertFalse(execution.debug_tensor_values)
        elif tensor_debug_mode == "CURT_HEALTH":
          self.assertLen(execution.debug_tensor_values, 1)
          if execution.op_type in ("AddV2", "Mul", "RealDiv"):
            # 1st element: -1 is the unset tensor_id for eager op execution.
            # 2nd element: 0 means there is no inf or nan.
            self.assertAllClose(execution.debug_tensor_values, [[-1.0, 0.0]])
        elif tensor_debug_mode == "CONCISE_HEALTH":
          if execution.op_type in ("AddV2", "Mul", "RealDiv"):
            # 1st element: -1 is the unset tensor_id for eager op execution.
            # 2nd element: each scalar tensor has 1 element.
            # Remaining elements: no -inf, inf or nan in these
            self.assertAllClose(
                execution.debug_tensor_values, [[-1, 1, 0, 0, 0]])
        elif tensor_debug_mode == "FULL_HEALTH":
          if execution.op_type in ("AddV2", "Mul", "RealDiv"):
            # Elements: [
            #   -1 is the unset tensor_id for eager op execution,
            #   device ID (set to -1 for now),
            #   dtype, rank, element_count,
            #   neg_inf_count, pos_inf_count, nan_count
            #   neg_finite_count, zero_count, pos_finite_count]
            self.assertAllClose(
                execution.debug_tensor_values,
                [[-1, -1, 1, 0, 1, 0, 0, 0, 0, 0, 1]])
        elif tensor_debug_mode == "SHAPE":
          if execution.op_type in ("AddV2", "Mul", "RealDiv"):
            # 1st element: -1 is the unset tensor_id for eager op execution.
            # 2nd element: dtype enum value (float32).
            # 3rd element: rank (scalar).
            # 4th element: element count (4).
            # Remaining elements: shape at fixed length (6).
            self.assertAllClose(execution.debug_tensor_values,
                                [[-1, 1, 0, 1, 0, 0, 0, 0, 0, 0]])
        elif tensor_debug_mode == "FULL_TENSOR":
          tensor_values[execution.op_type].append(
              reader.execution_to_tensor_values(execution)[0])

        host_name, stack_frames = reader.read_execution_stack_trace(execution)
        self.assertEqual(host_name, _host_name)
        self._verifyStackFrames(stack_frames)

      if tensor_debug_mode == "FULL_TENSOR":
        self.assertAllClose(tensor_values["Greater"], [1, 1, 1, 1, 1, 1, 0])
        self.assertAllClose(tensor_values["RealDiv"], [5, 8, 4, 2, 1])
        self.assertAllClose(tensor_values["Mul"], [15])
        self.assertAllClose(tensor_values["AddV2"], [16])

      self.assertEqual(
          executed_op_types,
          [
              "Greater",
              "FloorMod",
              "Equal",
              "RealDiv",  # 10 --> 5
              "Greater",
              "FloorMod",
              "Equal",
              "Mul",
              "AddV2",  # 5 --> 16
              "Greater",
              "FloorMod",
              "Equal",
              "RealDiv",  # 16 --> 8
              "Greater",
              "FloorMod",
              "Equal",
              "RealDiv",  # 8 --> 4
              "Greater",
              "FloorMod",
              "Equal",
              "RealDiv",  # 4 --> 2
              "Greater",
              "FloorMod",
              "Equal",
              "RealDiv",  # 2 --> 1
              "Greater"
          ])

      # Due to the pure eager op execution, the .graph file and the
      # .graph_execution_traces file ought to be empty.
      self.assertFalse(reader.outermost_graphs())
      self.assertEqual(reader.num_graph_execution_traces(), 0)

  @parameterized.named_parameters(
      ("CurtHealth", "CURT_HEALTH"),
      ("ConciseHealth", "CONCISE_HEALTH"),
      ("FullHealth", "FULL_HEALTH"),
      ("Shape", "SHAPE"),
  )
  @test_util.run_in_graph_and_eager_modes
  def testModesSummarizingBadNumericalValue(self, tensor_debug_mode):
    # [FUNCTION DOC]
    # Scenario: for the curt-health, concise-health, full-health and shape modes (graph and eager),
    # a compiled function computes the ratio of a sum to a difference of float16 vectors that yields
    # infinities and not-a-number values. Asserts the executed op types (add, subtract, divide) and
    # the exact summary values per mode: bad-value flag, element and bad-value counts, full health
    # fields, or shape fields.
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode=tensor_debug_mode)

    @def_function.function
    def func(x, y):
      # [FUNCTION DOC]
      # Traced function: (x plus y) divided by (x minus y), producing inf or nan for some elements.
      return (x + y) / (x - y)

    x = np.array([-3, -1, 0, 0, 1, 1, 1, 2], dtype=np.float16)
    y = np.array([2, -1, 0, 0, 1, 1, 1, 3], dtype=np.float16)
    # x - y = [-5, 0, 0, 0, 0, 0, 0, -1]
    # (x + y) / (x - y) = [0.2, -inf, nan, nan, inf, inf, inf, -5].
    self.evaluate(func(x, y))
    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      graph_exec_traces = reader.graph_execution_traces()
      executed_op_types = [trace.op_type for trace in graph_exec_traces
                           if trace.op_type not in ["Const", "Placeholder"]]
      self.assertCountEqual(
          executed_op_types,
          ["AddV2", "Sub", "RealDiv"])
      if tensor_debug_mode == "CURT_HEALTH":
        for trace in graph_exec_traces:
          # 1st element: tensor_id, should be >= 0.
          # 2nd element: indicates if there is any inf or nan.
          tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
          self.assertGreaterEqual(tensor_id, 0)
          if trace.op_type == "RealDiv":
            self.assertAllClose(trace.debug_tensor_value, [tensor_id, 1])
          else:
            self.assertAllClose(trace.debug_tensor_value, [tensor_id, 0])
      elif tensor_debug_mode == "CONCISE_HEALTH":
        for trace in graph_exec_traces:
          # 1st element: tensor_id, should be >= 0.
          # 2nd element: element count (8).
          # Remaining 3 elements: The counts of -inf, inf and nan.
          tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
          self.assertGreaterEqual(tensor_id, 0)
          if trace.op_type == "RealDiv":
            self.assertAllClose(trace.debug_tensor_value,
                                [tensor_id, 8, 1, 3, 2])
          else:
            self.assertAllClose(trace.debug_tensor_value,
                                [tensor_id, 8, 0, 0, 0])
      elif tensor_debug_mode == "FULL_HEALTH":
        for trace in graph_exec_traces:
          # Elements: [
          #   -1 is the unset tensor_id for eager op execution,
          #   device ID (set to -1 for now),
          #   dtype, rank, element_count,
          #   neg_inf_count, pos_inf_count, nan_count
          #   neg_finite_count, zero_count, pos_finite_count]
          tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
          self.assertGreaterEqual(tensor_id, 0)
          if trace.op_type == "RealDiv":
            self.assertAllClose(trace.debug_tensor_value,
                                [tensor_id, -1, 19, 1, 8, 1, 3, 2, 1, 0, 1])
          elif trace.op_type == "Sub":
            self.assertAllClose(trace.debug_tensor_value,
                                [tensor_id, -1, 19, 1, 8, 0, 0, 0, 2, 6, 0])
      else:  # SHAPE.
        for trace in graph_exec_traces:
          # 1st element: tensor_id, should be >= 0.
          # 2nd element: dtype enum value (float16 = 19).
          # 3rd element: rank (1)
          # 4th element: element count (8).
          # Remaining elements: shape at fixed length (6).
          tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
          self.assertGreaterEqual(tensor_id, 0)
          self.assertAllClose(trace.debug_tensor_value,
                              [tensor_id, 19, 1, 8, 8, 0, 0, 0, 0, 0])

  @parameterized.named_parameters(
      ("CurtHealth", "CURT_HEALTH"),
      ("FullTensor", "FULL_TENSOR"),
  )
  @test_util.run_in_graph_and_eager_modes
  def testConstTensorsAreCaptured(self, tensor_debug_mode):
    # [FUNCTION DOC]
    # Scenario: curt-health and full-tensor modes with a compiled function multiplying by constant
    # two and adding constant three. Asserts the result 23 and at least three constant-op traces:
    # in curt-health mode each has a two-element value with no inf or nan; in full-tensor mode the
    # values include the argument 10 and constants 2 and 3 (order unspecified).
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode=tensor_debug_mode)
    @def_function.function
    def times_two_plus_three(x):
      # [FUNCTION DOC]
      # Traced function: input times constant two plus constant three.
      return x * constant_op.constant(2.0) + constant_op.constant(3.0)
    self.assertAllEqual(
        self.evaluate(times_two_plus_three(10.0)), 23.0)
    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      const_traces = [trace for trace in reader.graph_execution_traces()
                      if trace.op_type == "Const"]
      self.assertGreaterEqual(len(const_traces), 3)
      if tensor_debug_mode == "CURT_HEALTH":
        # Under CURT_HEALTH, each debug tensor value has the form
        # [tensor_id, has_inf_or_nan].
        self.assertLen(const_traces[0].debug_tensor_value, 2)
        self.assertEqual(const_traces[0].debug_tensor_value[1], 0)
        self.assertLen(const_traces[1].debug_tensor_value, 2)
        self.assertEqual(const_traces[1].debug_tensor_value[1], 0)
        self.assertLen(const_traces[2].debug_tensor_value, 2)
        self.assertEqual(const_traces[2].debug_tensor_value[1], 0)
      else:  # FULL_TENSOR.
        const_tensor_values = [
            reader.graph_execution_trace_to_tensor_value(const_trace)
            for const_trace in const_traces]
        # Avoid making assertion on the particular order of the debug tensors
        # for the three Consts because it may be indeterminate.
        self.assertIn(10.0, const_tensor_values)
        self.assertIn(2.0, const_tensor_values)
        self.assertIn(3.0, const_tensor_values)

  @parameterized.named_parameters(
      ("Shape", "SHAPE"),
  )
  @test_util.run_in_graph_and_eager_modes
  def testBooleanTensors(self, tensor_debug_mode):
    # [FUNCTION DOC]
    # Scenario: shape mode with a compiled function computing not-and of two boolean matrices.
    # Asserts the result, that the traced ops are logical-and then logical-not, and that each trace
    # holds the shape summary (tensor ID, bool dtype code, rank 2, element count 4, shape).
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode=tensor_debug_mode)

    @def_function.function
    def func(x, y):
      # [FUNCTION DOC]
      # Traced function: logical not of the logical and of two boolean inputs.
      return math_ops.logical_not(math_ops.logical_and(x, y))

    x = np.array([[False, False], [True, True]], dtype=np.bool_)
    y = np.array([[False, True], [False, True]], dtype=np.bool_)
    self.assertAllEqual(
        self.evaluate(func(x, y)), [[True, True], [True, False]])

    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      graph_exec_traces = reader.graph_execution_traces()
      executed_op_types = [trace.op_type for trace in graph_exec_traces
                           if trace.op_type not in ["Const", "Placeholder"]]
      self.assertEqual(
          executed_op_types,
          ["LogicalAnd", "LogicalNot"])
      for trace in graph_exec_traces:
        tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
        self.assertGreaterEqual(tensor_id, 0)
        # 1st element: tensor_id, should be >= 0.
        # 2nd element: dtype enum value (bool).
        # 3rd element: rank (2).
        # 4th element: element count (4).
        # Remaining elements: shape at fixed length.
        self.assertAllClose(
            trace.debug_tensor_value, [tensor_id, 10, 2, 4, 2, 2, 0, 0, 0, 0])

  def testListingSourceFiles(self):
    # [FUNCTION DOC]
    # Scenario: enable dumping, run one eager op and flush. Asserts the source file list is a
    # tuple of two-element tuples and contains the host name and path of this test file.
    writer = dumping_callback.enable_dump_debug_info(self.dump_root)
    # Run a simple eager execution event, so that the source files are dumped.
    self.assertAllClose(math_ops.truediv(7.0, 1.0 / 6.0), 42.0)
    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()
    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      source_file_list = reader.source_file_list()
      self.assertIsInstance(source_file_list, tuple)
      for item in source_file_list:
        self.assertIsInstance(item, tuple)
        self.assertLen(item, 2)
      self.assertIn((_host_name, _current_file_full_path), source_file_list)

  def testReadingSourceLines(self):
    # [FUNCTION DOC]
    # Scenario: enable dumping, run one eager op and flush. Asserts that the lines of this test
    # file stored in the dump (read by host name and file path) equal the file's actual lines when
    # split on newlines.
    writer = dumping_callback.enable_dump_debug_info(self.dump_root)
    # Run a simple eager execution event, so that the source-file contents are
    # dumped.
    self.assertAllClose(math_ops.truediv(7.0, 1.0 / 6.0), 42.0)
    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()
    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      with open(_current_file_full_path, "rt") as f:
        file_lines = f.read().split("\n")
      self.assertEqual(
          reader.source_lines(_host_name, _current_file_full_path), file_lines)

  @parameterized.named_parameters(
      ("NoTensor", "NO_TENSOR"),
      ("CurtHealth", "CURT_HEALTH"),
      ("ConciseHealth", "CONCISE_HEALTH"),
      ("FullHealth", "FULL_HEALTH"),
      ("Shape", "SHAPE"),
      ("FullTensor", "FULL_TENSOR"),
  )
  @test_util.run_in_graph_and_eager_modes
  def testNestedFunctionExecutionWithoutControlFlow(self, tensor_debug_mode):
    # [FUNCTION DOC]
    # Scenario: a compiled function calls another compiled function (sin of one plus log of a sum)
    # under every debug mode, graph and eager. Asserts a single outermost graph; in eager mode one
    # execution with correct graph name, inner graph and device name; the graph op digests for add,
    # log and sin with correct owning graphs, output tensor IDs and stack traces; the graph ID stack
    # of each traced op; and mode-specific debug tensor values of the traces.
    x = constant_op.constant(2.0)
    y = constant_op.constant(3.0)
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode=tensor_debug_mode)

    @def_function.function
    def log_sum(x, y):
      # [FUNCTION DOC]
      # Traced inner function: log of the sum of its two inputs.
      return math_ops.log(x + y)

    @def_function.function
    def sin1p_log_sum(x, y):
      # [FUNCTION DOC]
      # Traced outer function: sine of one plus the inner function result.
      return math_ops.sin(1.0 + log_sum(x, y))

    self.assertAllClose(sin1p_log_sum(x, y), np.sin(1.0 + np.log(5.0)))
    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      outermost_graphs = reader.outermost_graphs()
      self.assertLen(outermost_graphs, 1)

      if context.executing_eagerly():
        # NOTE(b/142486213): Execution of the TF function happens with
        # Session.run() in v1 graph mode, so doesn't get logged to the
        # .execution file.
        executions = reader.executions()
        self.assertLen(executions, 1)
        self.assertIn("sin1p_log_sum", executions[0].op_type)
        # Get the executed graph and verify its identity and inner graph.
        graph = reader.graph_by_id(executions[0].graph_id)
        self.assertEqual(graph.name, "sin1p_log_sum")
        self.assertLen(graph.inner_graph_ids, 1)
        inner_graph = reader.graph_by_id(graph.inner_graph_ids[0])
        self.assertEqual(inner_graph.name, "log_sum")
        # Check device names.
        self.assertLen(executions[0].output_tensor_device_ids, 1)
        self.assertEqual(
            reader.device_name_by_id(executions[0].output_tensor_device_ids[0]),
            self._expectedDefaultDeviceName())
        self.assertIn(self._expectedDefaultDeviceName(),
                      set(reader.device_name_map().values()))

      # Verify the recorded graph-building history.
      add_op_digests = reader.graph_op_digests(op_type="AddV2")
      self.assertLen(add_op_digests, 2)
      self.assertEqual(
          reader.graph_by_id(add_op_digests[0].graph_id).name, "log_sum")
      self.assertEqual(
          reader.graph_by_id(add_op_digests[1].graph_id).name, "sin1p_log_sum")
      log_op_digests = reader.graph_op_digests(op_type="Log")
      self.assertLen(log_op_digests, 1)
      self.assertEqual(
          reader.graph_by_id(log_op_digests[0].graph_id).name, "log_sum")
      sin_op_digests = reader.graph_op_digests(op_type="Sin")
      self.assertLen(sin_op_digests, 1)
      self.assertEqual(
          reader.graph_by_id(sin_op_digests[0].graph_id).name, "sin1p_log_sum")

      # Verify the output tensor IDs and the stack traces.
      for op_digest in add_op_digests + log_op_digests + sin_op_digests:
        # These are all single-output ops.
        self.assertLen(op_digest.output_tensor_ids, 1)
        self.assertGreaterEqual(op_digest.output_tensor_ids[0], 0)
        _, stack_frames = reader.read_graph_op_creation_stack_trace(op_digest)
        self._verifyStackFrames(stack_frames)

      graph_exec_traces = [trace for trace in reader.graph_execution_traces()
                           if trace.op_type not in ["Const", "Placeholder"]]
      executed_op_types = [digest.op_type for digest in graph_exec_traces]
      self.assertEqual(
          executed_op_types, ["AddV2", "Log", "AddV2", "Sin"])

      # Verify the graph ID stack of each op.
      # 1st AddV2 op.
      self.assertEqual(
          reader.graph_by_id(graph_exec_traces[0].graph_ids[-1]).name,
          "log_sum")
      self.assertEqual(
          reader.graph_by_id(graph_exec_traces[0].graph_ids[-2]).name,
          "sin1p_log_sum")
      # Log op.
      self.assertEqual(
          reader.graph_by_id(graph_exec_traces[1].graph_ids[-1]).name,
          "log_sum")
      self.assertEqual(
          reader.graph_by_id(graph_exec_traces[1].graph_ids[-2]).name,
          "sin1p_log_sum")
      # 2nd AddV2 op.
      self.assertEqual(
          reader.graph_by_id(graph_exec_traces[2].graph_ids[-1]).name,
          "sin1p_log_sum")
      # Sin op.
      self.assertEqual(
          reader.graph_by_id(graph_exec_traces[3].graph_ids[-1]).name,
          "sin1p_log_sum")

      if tensor_debug_mode == "NO_TENSOR":
        # Under the default NO_TENSOR tensor-debug mode, the tensor_proto ought
        # to be an empty float32 tensor.
        for trace in graph_exec_traces:
          self.assertIsNone(trace.debug_tensor_value)
      elif tensor_debug_mode == "CURT_HEALTH":
        # Test the association between graph exec and prior graph building.
        # In each case, the 1st element of debug_tensor_value is the ID of the
        # symbolic tenosr and the 2nd element is a zero indicating there is no
        # inf or nan.
        self.assertAllClose(  # 1st AddV2 op.
            graph_exec_traces[0].debug_tensor_value,
            [add_op_digests[0].output_tensor_ids[0], 0.0])
        self.assertAllClose(  # Log op.
            graph_exec_traces[1].debug_tensor_value,
            [log_op_digests[0].output_tensor_ids[0], 0.0])
        self.assertAllClose(  # 2nd AddV2 op.
            graph_exec_traces[2].debug_tensor_value,
            [add_op_digests[1].output_tensor_ids[0], 0.0])
        self.assertAllClose(  # Sin op.
            graph_exec_traces[3].debug_tensor_value,
            [sin_op_digests[0].output_tensor_ids[0], 0.0])
      elif tensor_debug_mode == "CONCISE_HEALTH":
        # 1st element: tensor_id.
        # 2nd element: element count. Remaining elements: all zero because there
        # is no -inf, inf or nan.
        # 1st AddV2 op.
        self.assertAllClose(
            graph_exec_traces[0].debug_tensor_value,
            [add_op_digests[0].output_tensor_ids[0], 1.0, 0.0, 0.0, 0.0])
        # Log op.
        self.assertAllClose(
            graph_exec_traces[1].debug_tensor_value,
            [log_op_digests[0].output_tensor_ids[0], 1.0, 0.0, 0.0, 0.0])
        # 2nd AddV2 op.
        self.assertAllClose(
            graph_exec_traces[2].debug_tensor_value,
            [add_op_digests[1].output_tensor_ids[0], 1.0, 0.0, 0.0, 0.0])
        # Sin op.
        self.assertAllClose(
            graph_exec_traces[3].debug_tensor_value,
            [sin_op_digests[0].output_tensor_ids[0], 1.0, 0.0, 0.0, 0.0])
      elif tensor_debug_mode == "FULL_HEALTH":
        # Elements: [
        #   -1 is the unset tensor_id for eager op execution,
        #   device ID (set to -1 for now),
        #   dtype, rank, element_count,
        #   neg_inf_count, pos_inf_count, nan_count
        #   neg_finite_count, zero_count, pos_finite_count]
        # 1st AddV2 op.
        self.assertAllClose(
            graph_exec_traces[0].debug_tensor_value,
            [add_op_digests[0].output_tensor_ids[0],
             -1, 1, 0, 1, 0, 0, 0, 0, 0, 1])
        # Log op.
        self.assertAllClose(
            graph_exec_traces[1].debug_tensor_value,
            [log_op_digests[0].output_tensor_ids[0],
             -1, 1, 0, 1, 0, 0, 0, 0, 0, 1])
        # 2nd AddV2 op.
        self.assertAllClose(
            graph_exec_traces[2].debug_tensor_value,
            [add_op_digests[1].output_tensor_ids[0],
             -1, 1, 0, 1, 0, 0, 0, 0, 0, 1])
        # Sin op.
        self.assertAllClose(
            graph_exec_traces[3].debug_tensor_value,
            [sin_op_digests[0].output_tensor_ids[0],
             -1, 1, 0, 1, 0, 0, 0, 0, 0, 1])
      elif tensor_debug_mode == "SHAPE":
        # 1st element: tensor_id.
        # 2nd element: dtype (float32).
        # 3rd element: rank (scalar).
        # 4th element: element count (1).
        # Remaining elements: shape padded to fixed length (6).
        # 1st AddV2 op.
        self.assertAllClose(
            graph_exec_traces[0].debug_tensor_value,
            [add_op_digests[0].output_tensor_ids[0], 1, 0, 1, 0, 0, 0, 0, 0, 0])
        # Log op.
        self.assertAllClose(
            graph_exec_traces[1].debug_tensor_value,
            [log_op_digests[0].output_tensor_ids[0], 1, 0, 1, 0, 0, 0, 0, 0, 0])
        # 2nd AddV2 op.
        self.assertAllClose(
            graph_exec_traces[2].debug_tensor_value,
            [add_op_digests[1].output_tensor_ids[0], 1, 0, 1, 0, 0, 0, 0, 0, 0])
        # Sin op.
        self.assertAllClose(
            graph_exec_traces[3].debug_tensor_value,
            [sin_op_digests[0].output_tensor_ids[0], 1, 0, 1, 0, 0, 0, 0, 0, 0])
      else:  # FULL_TENSOR.

        full_tensor_values = [
            reader.graph_execution_trace_to_tensor_value(trace)
            for trace in graph_exec_traces]
        self.assertAllClose(
            full_tensor_values[0], 5.0)  # 1st AddV2 op.
        self.assertAllClose(
            full_tensor_values[1], np.log(5.0))  # Log op.
        self.assertAllClose(
            full_tensor_values[2],
            np.log(5.0) + 1.0)  # 2nd AddV2 op.
        self.assertAllClose(
            full_tensor_values[3],
            np.sin(np.log(5.0) + 1.0))  # Sin op.

  @parameterized.named_parameters(
      ("NoTensor", "NO_TENSOR"),
      ("FullTensor", "FULL_TENSOR"),
  )
  @test_util.run_in_graph_and_eager_modes
  def testGraphOpConsumingRelationIsCaptured(self, tensor_debug_mode):
    # [FUNCTION DOC]
    # Scenario: nested compiled functions compute the maximum index of the unique values of sin of
    # one plus log of a sum, in no-tensor and full-tensor modes. Asserts the number of traces per op
    # type and that the recorded graphs report the correct consumers: add consumed by log, second add
    # by sin, sin by unique, and the second output of unique by the max op.
    x = constant_op.constant([2.0, 2.0])
    y = constant_op.constant([3.0, 3.0])
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode=tensor_debug_mode)

    @def_function.function
    def log_sum(x, y):
      # [FUNCTION DOC]
      # Traced inner function: log of the sum of its two inputs.
      return math_ops.log(x + y)

    @def_function.function
    def maxindex_sin1p_log_sum(x, y):
      # [FUNCTION DOC]
      # Traced outer function: applies unique to sin of one plus the inner result and returns the
      # maximum of the index output.
      _, indices = array_ops.unique(math_ops.sin(1.0 + log_sum(x, y)))
      return math_ops.reduce_max(indices)

    maxindex = maxindex_sin1p_log_sum(x, y)
    self.assertAllEqual(maxindex, 0)
    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      traces = reader.graph_execution_traces()
      add_traces = [trace for trace in traces if trace.op_type == "AddV2"]
      log_traces = [trace for trace in traces if trace.op_type == "Log"]
      sin_traces = [trace for trace in traces if trace.op_type == "Sin"]
      unique_traces = [trace for trace in traces if trace.op_type == "Unique"]
      max_traces = [trace for trace in traces if trace.op_type == "Max"]
      self.assertLen(add_traces, 2)
      self.assertLen(log_traces, 1)
      self.assertLen(sin_traces, 1)
      self.assertLen(unique_traces, 2)  # The Unique op outputs two tensors.
      self.assertLen(max_traces, 1)
      graph = reader.graph_by_id(add_traces[0].graph_id)
      # The first AddV2 op is consumed by the Log op.
      self.assertEqual(
          graph.get_op_consumers(add_traces[0].op_name),
          [(0, log_traces[0].op_name, 0)])
      graph = reader.graph_by_id(add_traces[1].graph_id)
      # The second AddV2 op is consumed by the Sin op.
      self.assertEqual(
          graph.get_op_consumers(add_traces[1].op_name),
          [(0, sin_traces[0].op_name, 0)])
      # The last Sin op is consumed by the Unique op.
      self.assertEqual(
          graph.get_op_consumers(sin_traces[0].op_name),
          [(0, unique_traces[0].op_name, 0)])
      # The Unique op's 2nd output tensor is consumed by the Max op.
      self.assertEqual(
          graph.get_op_consumers(unique_traces[0].op_name),
          [(1, max_traces[0].op_name, 0)])

  def testCapturingExecutedGraphIdsOfTwoCompilationsOfSameFunction(self):
    # [FUNCTION DOC]
    # Scenario: one compiled function is called alternately with float32 and float64 inputs (two
    # compilations), twice each, in no-tensor mode. Asserts four executions whose op types start
    # with the function prefix, equal graph IDs for same-dtype calls, different IDs across dtypes,
    # and graph name equal to the function name.
    """Test correct executed IDs of two FuncGraphs from the same Py function."""
    x_float32 = constant_op.constant(np.array(3.5, dtype=np.float32))
    x_float64 = constant_op.constant(np.array(4.5, dtype=np.float64))
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode="NO_TENSOR")

    @def_function.function
    def ceil_times_two(x):
      # [FUNCTION DOC]
      # Traced function: rounds the input up and doubles it.
      return math_ops.ceil(x) * 2.0

    # Four executions, with two different FuncGraphs, which should lead
    # to two unique executed graph IDs (see assertion below).
    self.assertAllClose(ceil_times_two(x_float32), 8.0)
    self.assertAllClose(ceil_times_two(x_float64), 10.0)
    self.assertAllClose(ceil_times_two(x_float32), 8.0)
    self.assertAllClose(ceil_times_two(x_float64), 10.0)
    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()

      executions = reader.executions()
      self.assertLen(executions, 4)
      for execution in executions:
        self.assertStartsWith(execution.op_type, "__inference_ceil_times_two_")
      executed_graph_ids = [execution.graph_id for execution in executions]
      self.assertEqual(executed_graph_ids[0], executed_graph_ids[2])
      self.assertEqual(executed_graph_ids[1], executed_graph_ids[3])
      self.assertNotEqual(executed_graph_ids[0], executed_graph_ids[1])
      self.assertNotEqual(executed_graph_ids[2], executed_graph_ids[3])
      for executed_graph_id in executed_graph_ids:
        self.assertEqual(
            reader.graph_by_id(executed_graph_id).name, "ceil_times_two")

  def testCapturingExecutedGraphIdsOfDuplicateFunctionNames(self):
    # [FUNCTION DOC]
    # Scenario: two instances of a local class each have a same-named compiled method. Each is run
    # twice in no-tensor mode. Asserts four executions with function-style op types, that the same
    # instance yields the same executed graph ID while different instances yield different IDs,
    # and that each graph is named after the method.
    """Two FuncGraphs compiled from Python functions with identical names."""
    x = constant_op.constant(np.array(3.5, dtype=np.float32))
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode="NO_TENSOR")

    class TestClass(object):
      # [CLASS DOC]
      # Local helper class whose compiled method has the same name for every instance, used to
      # check that executed graph IDs are distinguished per instance.

      @def_function.function
      def ceil_times_two(self, x):
        # [FUNCTION DOC]
        # Traced method: rounds the input up and doubles it.
        return math_ops.ceil(x) * 2.0

    # The `ceil_times_two` method of the two objects will be compiled
    # into separate FuncGraphs.
    test_object_1 = TestClass()
    test_object_2 = TestClass()

    # Four executions, with two different FuncGraphs, which should lead
    # to two unique executed graph IDs (see assertion below).
    self.assertAllClose(test_object_1.ceil_times_two(x), 8.0)
    self.assertAllClose(test_object_2.ceil_times_two(x), 8.0)
    self.assertAllClose(test_object_1.ceil_times_two(x), 8.0)
    self.assertAllClose(test_object_2.ceil_times_two(x), 8.0)
    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      executions = reader.executions()
      self.assertLen(executions, 4)
      for execution in executions:
        self.assertStartsWith(execution.op_type, "__inference_ceil_times_two_")
      executed_graph_ids = [execution.graph_id for execution in executions]
      self.assertEqual(executed_graph_ids[0], executed_graph_ids[2])
      self.assertEqual(executed_graph_ids[1], executed_graph_ids[3])
      self.assertNotEqual(executed_graph_ids[0], executed_graph_ids[1])
      self.assertNotEqual(executed_graph_ids[2], executed_graph_ids[3])
      for executed_graph_id in executed_graph_ids:
        self.assertEqual(
            reader.graph_by_id(executed_graph_id).name, "ceil_times_two")

  @parameterized.named_parameters(
      ("AddV2", "AddV2"),
      ("Log", "Log"),
      ("AddV2AndLog", "(AddV2|Log)"),
  )
  @test_util.run_in_graph_and_eager_modes
  def testOpRegex(self, op_regex):
    # [FUNCTION DOC]
    # Scenario: parameterized over op regular expressions (add, log, either). Full-tensor dumping is
    # enabled with the regex, and a nested compiled function computes sin of one plus log of a sum.
    # Asserts that all add, log and sin ops are recorded as graph op creations regardless of the
    # regex, while only the matching ops appear in the graph execution traces, with the expected
    # tensor values.
    x = constant_op.constant(2.0)
    y = constant_op.constant(3.0)
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode="FULL_TENSOR",
        op_regex=op_regex)

    @def_function.function
    def log_sum(x, y):
      # [FUNCTION DOC]
      # Traced inner function: log of the sum of its two inputs.
      return math_ops.log(x + y)

    @def_function.function
    def sin1p_log_sum(x, y):
      # [FUNCTION DOC]
      # Traced outer function: sine of one plus the inner function result.
      return math_ops.sin(1.0 + log_sum(x, y))

    self.assertAllClose(
        self.evaluate(sin1p_log_sum(x, y)), np.sin(1.0 + np.log(5.0)))
    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      graph_op_digests = reader.graph_op_digests()
      op_types = [digest.op_type for digest in graph_op_digests]
      self.assertIn("AddV2", op_types)
      self.assertIn("Log", op_types)
      self.assertIn("Sin", op_types)

      graph_exec_digests = reader.graph_execution_traces(digest=True)
      executed_op_types = [digest.op_type for digest in graph_exec_digests]
      tensor_values = [reader.graph_execution_trace_to_tensor_value(digest)
                       for digest in graph_exec_digests]
      if op_regex == "AddV2":
        self.assertEqual(executed_op_types, ["AddV2", "AddV2"])
        self.assertLen(tensor_values, 2)
        self.assertAllClose(tensor_values[0], 5.0)  # 1st AddV2 op.
        self.assertAllClose(
            tensor_values[1], np.log(5.0) + 1.0)  # 2nd AddV2 op.
      elif op_regex == "Log":
        self.assertEqual(executed_op_types, ["Log"])
        self.assertLen(tensor_values, 1)
        self.assertAllClose(tensor_values[0], np.log(5.0))  # Log op.
      else:  # "(AddV2|Log)"
        self.assertEqual(executed_op_types, ["AddV2", "Log", "AddV2"])
        self.assertLen(tensor_values, 3)
        self.assertAllClose(tensor_values[0], 5.0)  # 1st AddV2 op.
        self.assertAllClose(tensor_values[1], np.log(5.0))  # Log op.
        self.assertAllClose(
            tensor_values[2], np.log(5.0) + 1.0)  # 2nd AddV2 op.

  def testIncorrectTensorDTypeArgFormatLeadsToError(self):
    # [FUNCTION DOC]
    # Scenario: enable dumping with invalid tensor dtype arguments. Asserts a value error naming
    # the expected list, tuple or callable for an empty mapping, a string, and a single dtype
    # object; and a type error for a list containing callables.
    with self.assertRaisesRegex(
        ValueError, r".*expected.*list.*tuple.*callable.*but received.*\{\}"):
      dumping_callback.enable_dump_debug_info(self.dump_root,
                                              tensor_dtypes=dict())
    with self.assertRaisesRegex(
        ValueError, r".*expected.*list.*tuple.*callable.*but received.*"):
      dumping_callback.enable_dump_debug_info(self.dump_root,
                                              tensor_dtypes="float32")
    with self.assertRaisesRegex(
        ValueError, r".*expected.*list.*tuple.*callable.*but received.*"):
      dumping_callback.enable_dump_debug_info(
          self.dump_root, tensor_dtypes=dtypes.float32)
    with self.assertRaises(TypeError):
      dumping_callback.enable_dump_debug_info(self.dump_root, tensor_dtypes=[
          lambda dtype: dtype.is_floating, lambda dtype: dtype.is_integer])

  @parameterized.named_parameters(
      ("float", [dtypes.float32], None),
      ("float_only_sum", ["float32"], "Sum"),
      ("float_no_sum", (dtypes.float32,), "(?!Sum)"),
      ("int", [dtypes.int32], None),
      ("int_via_lambda", lambda dtype: dtype.is_integer, None),
      ("exclude_Sum", None, "(?!Sum)"),
      ("All", None, None),
  )
  @test_util.run_in_graph_and_eager_modes
  def testTensorDTypesAndOpRegexFilters(self,
                                        tensor_dtypes,
                                        op_regex):
    # [FUNCTION DOC]
    # Scenario: parameterized over combinations of tensor dtype filter (list, tuple, callable or
    # none) and op regular expression. A compiled function takes the unique values of a float
    # vector and sums them. Full-tensor dumping is enabled with the filters. Asserts, per
    # combination, which op types (unique, sum) appear in the graph execution traces (ignoring
    # constants and placeholders) and their tensor values (unique values, indices, sum 17).
    xs = constant_op.constant([2., 6., 8., 1., 2.], dtype=dtypes.float32)
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode="FULL_TENSOR",
        tensor_dtypes=tensor_dtypes,
        op_regex=op_regex)

    @def_function.function
    def unique_sum(xs):
      # [FUNCTION DOC]
      # Traced function: computes the unique values of the input, returning their sum together
      # with the index of each input element in the unique list.
      """Sum over the unique values, for testing."""
      unique_xs, indices = array_ops.unique(xs)
      return math_ops.reduce_sum(unique_xs), indices

    y, indices = self.evaluate(unique_sum(xs))
    self.assertAllClose(y, 17.)
    self.assertAllEqual(indices, [0, 1, 2, 3, 0])

    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      graph_exec_digests = reader.graph_execution_traces(digest=True)
      executed_op_types = [digest.op_type for digest in graph_exec_digests
                           if digest.op_type not in ("Const", "Placeholder")]
      tensor_values = [reader.graph_execution_trace_to_tensor_value(digest)
                       for digest in graph_exec_digests
                       if digest.op_type not in ("Const", "Placeholder")]

      if tensor_dtypes == [dtypes.float32] and not op_regex:
        self.assertEqual(executed_op_types, ["Unique", "Sum"])
        self.assertLen(tensor_values, 2)
        self.assertAllClose(tensor_values[0], [2, 6, 8, 1])  # Unique values.
        self.assertAllClose(tensor_values[1], 17.)  # Sum.
      elif tensor_dtypes == ["float32"] and op_regex == "Sum":
        self.assertEqual(executed_op_types, ["Sum"])
        self.assertLen(tensor_values, 1)
        self.assertAllClose(tensor_values[0], 17.)  # Sum.
      elif tensor_dtypes == (dtypes.float32,) and op_regex == "(?!Sum)":
        self.assertEqual(executed_op_types, ["Unique"])
        self.assertLen(tensor_values, 1)
        self.assertAllClose(tensor_values[0], [2, 6, 8, 1])  # Unique values.
      elif tensor_dtypes == [dtypes.int32] and not op_regex:
        self.assertEqual(executed_op_types, ["Unique"])
        self.assertLen(tensor_values, 1)
        self.assertAllEqual(
            tensor_values[0], [0, 1, 2, 3, 0])  # Unique indices.
      elif callable(tensor_dtypes) and not op_regex:
        self.assertEqual(executed_op_types, ["Unique"])
        self.assertLen(tensor_values, 1)
        self.assertAllEqual(
            tensor_values[0], [0, 1, 2, 3, 0])  # Unique indices.
      elif not tensor_dtypes and op_regex == "(?!Sum)":
        self.assertEqual(executed_op_types, ["Unique", "Unique"])
        self.assertLen(tensor_values, 2)
        self.assertAllClose(tensor_values[0], [2, 6, 8, 1])  # Unique values.
        self.assertAllEqual(
            tensor_values[1], [0, 1, 2, 3, 0])  # Unique indices.
      else:  # "All".
        self.assertEqual(executed_op_types, ["Unique", "Unique", "Sum"])
        self.assertLen(tensor_values, 3)
        self.assertAllClose(tensor_values[0], [2, 6, 8, 1])  # Unique values.
        self.assertAllEqual(
            tensor_values[1], [0, 1, 2, 3, 0])  # Unique indices.
        self.assertAllClose(tensor_values[2], 17)  # Sum.

  @parameterized.named_parameters(
      ("NoTensor", "NO_TENSOR"),
      ("CurtHealth", "CURT_HEALTH"),
      ("FullTensor", "FULL_TENSOR"),
  )
  @test_util.run_in_graph_and_eager_modes
  def testFunctionExecutionWithControlFlow(self, tensor_debug_mode):
    # [FUNCTION DOC]
    # Scenario: a compiled function with a while loop that doubles a value four times (result 8),
    # under no-tensor, curt-health and full-tensor modes (graph and eager). Asserts that graph op
    # digests contain the less-than, multiply and add ops; that execution and graph execution trace
    # files are empty before flushing the execution files; and afterwards, in eager mode, that one
    # execution of the function with the expected IDs is recorded, plus per-mode checks on the
    # traced tensor values (loop condition and multiply results in full-tensor mode).
    x = constant_op.constant(0.5, dtype=dtypes.float32)
    times = constant_op.constant(4, dtype=dtypes.int32)
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode=tensor_debug_mode)

    @def_function.function
    def iterative_doubling(x, times):
      # [FUNCTION DOC]
      # Traced function: loops a number of times, doubling the value each time.
      i = constant_op.constant(0, dtype=dtypes.int32)
      while i < times:
        x = x * 2.0
        i += 1
      return x

    self.assertAllClose(self.evaluate(iterative_doubling(x, times)), 8.0)

    writer.FlushNonExecutionFiles()
    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      graph_op_digests = reader.graph_op_digests()
      op_types = [digest.op_type for digest in graph_op_digests]
      self.assertIn("Less", op_types)
      self.assertIn("Mul", op_types)
      self.assertIn("AddV2", op_types)

      # Before FlushExecutionFiles() is called, the .execution and
      # .graph_execution_traces files should be both empty.
      self.assertEqual(reader.num_executions(), 0)
      self.assertEqual(reader.num_graph_execution_traces(), 0)

      # TODO(cais): Backport execution instrumentation to tf.Session.
      writer.FlushExecutionFiles()
      # After the flushing, the .execution file should hold the appropriate
      # contents.
      reader.update()
      if context.executing_eagerly():
        # NOTE(b/142486213): Execution of the TF function happens with
        # Session.run() in v1 graph mode, hence it doesn't get logged to the
        executions = reader.executions()
        self.assertLen(executions, 1)
        executed_op_types = [execution.op_type for execution in executions]
        self.assertIn("iterative_doubling", executions[0].op_type)
        execution = executions[0]
        self.assertLen(execution.input_tensor_ids, 2)
        self.assertLen(execution.output_tensor_ids, 1)
        self.assertEqual(
            debug_event_pb2.TensorDebugMode.keys()[execution.tensor_debug_mode],
            tensor_debug_mode)
        if tensor_debug_mode == "FULL_TENSOR":
          tensor_values = reader.execution_to_tensor_values(execution)
          self.assertAllClose(tensor_values, [8.0])

      graph_exec_traces = reader.graph_execution_traces()
      executed_op_types = [trace.op_type for trace in graph_exec_traces
                           if trace.op_type != "Const"]
      if tensor_debug_mode != "CURT_HEALTH":
        # Less outputs a boolean tensor, which is not tracked under CURT_HEALTH.
        # The Less op should have been executed 5 times.
        self.assertEqual(executed_op_types.count("Less"), 5)
        # The last executed op should be Less.
        self.assertEqual(executed_op_types[-1], "Less")
        # AddV2 produces an int tensor, which is not tracked under CURT_HEALTH.
        # The AddV2 op should have been run, but we refrain from asserting on
        # how many times it's executed.
        self.assertIn("AddV2", executed_op_types)
        for trace in graph_exec_traces:
          self.assertEqual(trace.output_slot, 0)
      # The Mul op should have been executed 4 times.
      self.assertEqual(executed_op_types.count("Mul"), 4)

      tensor_values = [reader.graph_execution_trace_to_tensor_value(trace)
                       for trace in graph_exec_traces]
      if tensor_debug_mode == "NO_TENSOR":
        # Under the default NO_TENSOR tensor-debug mode, the tensor_proto ought
        # to be an empty float32 tensor.
        for tensor_value in tensor_values:
          self.assertAllEqual(tensor_value, [])
      elif tensor_debug_mode == "CURT_HEALTH":
        for trace in graph_exec_traces:
          tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
          # 1st element: tensor_id; 2nd element: 0 indicating no inf or nan.
          self.assertAllClose(trace.debug_tensor_value, [tensor_id, 0.0])
      elif tensor_debug_mode == "FULL_TENSOR":
        less_values = [
            reader.graph_execution_trace_to_tensor_value(trace)
            for trace in graph_exec_traces if trace.op_type == "Less"]
        self.assertAllEqual(less_values, [True, True, True, True, False])
        mul_values = [
            reader.graph_execution_trace_to_tensor_value(trace)
            for trace in graph_exec_traces if trace.op_type == "Mul"]
        self.assertAllClose(mul_values, [1.0, 2.0, 4.0, 8.0])

  def testCallingEnableTracingTwiceWithTheSameDumpRootIsIdempotent(self):
    # [FUNCTION DOC]
    # Scenario: enable dumping twice with the same dump root, then run the unique op twice. Asserts
    # exactly two executions are recorded (not duplicated), each with a positive wall time, op type
    # unique, two outputs, and a stack trace that includes this test file.
    x = constant_op.constant([10.0, 12.0, 10.0])
    dumping_callback.enable_dump_debug_info(self.dump_root)
    writer = dumping_callback.enable_dump_debug_info(self.dump_root)

    for _ in range(2):
      array_ops.unique(x)

    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      executions = reader.executions()
      self.assertLen(executions, 2)
      for execution in executions:
        self.assertGreater(execution.wall_time, 0)
        self.assertEqual(execution.op_type, "Unique")
        self.assertEqual(execution.num_outputs, 2)
        _, stack_frames = reader.read_execution_stack_trace(execution)
        self._verifyStackFrames(stack_frames)

  def testCallingEnableTracingTwiceWithDifferentDumpRootsOverwrites(self):
    # [FUNCTION DOC]
    # Scenario: enable dumping to one directory, then to a new directory, then run the unique op
    # twice. Asserts that the new directory holds two executions of the unique op (two outputs,
    # positive wall time, stack frames including this file) and that the old directory received
    # nothing (no executions, no outermost graphs).
    x = constant_op.constant([10.0, 12.0, 10.0])
    dumping_callback.enable_dump_debug_info(self.dump_root)
    new_dump_root = self.dump_root + "_new_dump_root"
    writer = dumping_callback.enable_dump_debug_info(new_dump_root)

    for _ in range(2):
      array_ops.unique(x)

    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(new_dump_root) as reader:
      reader.update()
      executions = reader.executions()
      self.assertLen(executions, 2)
      for execution in executions:
        self.assertGreater(execution.wall_time, 0)
        self.assertEqual(execution.op_type, "Unique")
        self.assertEqual(execution.num_outputs, 2)
        _, stack_frames = reader.read_execution_stack_trace(execution)
        self._verifyStackFrames(stack_frames)

    with debug_events_reader.DebugDataReader(
        self.dump_root) as old_dump_root_reader:
      old_dump_root_reader.update()
      # The old dump root shouldn't have been written to.
      self.assertEqual(old_dump_root_reader.num_executions(), 0)
      self.assertFalse(old_dump_root_reader.outermost_graphs())

  def testCallingEnableRepeatedlyWithDifferentTensorDebugMode(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says re-enabling with another mode overwrites the earlier mode, but
    # the test actually asserts that a second call with a different mode raises a value error and
    # the first (no-tensor) mode remains in effect.
    # Scenario: enable in no-tensor mode, run a compiled function, verify every traced tensor value
    # is empty, then expect the value error mentioning both modes when enabling with full-tensor.
    """Assert calling enable_dump_debug_info() with two tensor-debug modes.

    It should lead to overwriting of the previously-configured mode.
    """
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode="NO_TENSOR")

    @def_function.function
    def add_1_divide_by_2(x):
      # [FUNCTION DOC]
      # Traced function: adds one to the input and halves the result.
      return (x + 1.0) / 2.0

    self.assertAllClose(add_1_divide_by_2(constant_op.constant(4.0)), 2.5)
    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      graph_exec_digests = reader.graph_execution_traces(digest=True)
      tensor_values = [reader.graph_execution_trace_to_tensor_value(digest)
                       for digest in graph_exec_digests]
      for tensor_value in tensor_values:
        # Under NO_TENSOR mode, each tensor is summarized as an empty float32
        # array.
        self.assertAllEqual(tensor_value, [])

    with self.assertRaisesRegex(
        ValueError, r"already.*NO_TENSOR.*FULL_TENSOR.*not be honored"):
      dumping_callback.enable_dump_debug_info(
          self.dump_root, tensor_debug_mode="FULL_TENSOR")

  @parameterized.named_parameters(
      ("NoTensor", "NO_TENSOR"),
      ("FullTensor", "FULL_TENSOR"),
  )
  def testDisableTracingWorks(self, tensor_debug_mode):
    # [FUNCTION DOC]
    # Scenario: enable dumping (no-tensor and full-tensor modes), immediately disable it, then run
    # an op twice. Asserts that the dump root holds no executions, no graph execution traces and no
    # outermost graphs.
    x = constant_op.constant([10.0, 12.0, 10.0])
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode=tensor_debug_mode)
    dumping_callback.disable_dump_debug_info()

    for _ in range(2):
      array_ops.unique(x)

    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      self.assertEqual(reader.num_executions(), 0)
      self.assertEqual(reader.num_graph_execution_traces(), 0)
      self.assertFalse(reader.outermost_graphs())

  @parameterized.named_parameters(
      ("NoTensor", "NO_TENSOR"),
      ("CurtHealth", "CURT_HEALTH"),
      ("ConciseHealth", "CONCISE_HEALTH"),
      ("FullHealth", "FULL_HEALTH"),
      ("Shape", "SHAPE"),
      ("FullTensor", "FULL_TENSOR"),
  )
  def testMultiThreadedExecutionWithSameSetting(self, tensor_debug_mode):
    # [FUNCTION DOC]
    # Scenario: dumping enabled in the main thread for each debug mode; a compiled function adds
    # twice a variable to another variable, run once on the main thread and three more times from
    # worker threads. Asserts the final value (34), non-decreasing execution wall times, counts of
    # multiply and read-variable ops in the graph execution traces, single-output traces, and the
    # per-mode debug tensor values (empty, health summaries, shape summaries or full values).
    """Dumping from multiple threads using the same setting."""
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode=tensor_debug_mode)
    x = variables.Variable(10.0, dtype=dtypes.float32)
    y = variables.Variable(3.0, dtype=dtypes.float32)

    @def_function.function
    def increase_x():
      # [FUNCTION DOC]
      # Traced function: adds twice the second variable to the first variable and returns the
      # result. Also used as the thread target.
      return x.assign_add(y * 2.0)

    increase_x()

    num_threads = 3
    threads = []
    for _ in range(num_threads):
      threads.append(threading.Thread(target=increase_x))
    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()
    # 10 --> 16 --> 22 --> 28 --> 34.
    self.assertAllClose(x.read_value(), 34.0)

    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      exec_digests = reader.executions(digest=True)
      prev_wall_time = 1
      for exec_digest in exec_digests:
        self.assertGreaterEqual(exec_digest.wall_time, prev_wall_time)
        prev_wall_time = exec_digest.wall_time

      graph_exec_traces = reader.graph_execution_traces()
      executed_op_types = [trace.op_type for trace in graph_exec_traces]
      self.assertEqual(executed_op_types.count("Mul"), 1 + num_threads)
      self.assertEqual(
          executed_op_types.count("ReadVariableOp"), 2 * (1 + num_threads))
      for trace in graph_exec_traces:
        # These are all single-output tensors.
        self.assertEqual(trace.output_slot, 0)

    tensor_values = [reader.graph_execution_trace_to_tensor_value(trace)
                     for trace in graph_exec_traces]
    if tensor_debug_mode == "NO_TENSOR":
      for tensor_value in tensor_values:
        self.assertAllEqual(tensor_value, [])
    elif tensor_debug_mode == "CURT_HEALTH":
      for trace in graph_exec_traces:
        tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
        # 1st element: tensor ID; 2nd element: 0 indicating no inf or nan.
        self.assertAllClose(trace.debug_tensor_value, [tensor_id, 0])
    elif tensor_debug_mode == "CONCISE_HEALTH":
      for trace in graph_exec_traces:
        tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
        # 1st element: tensor ID.
        # 2nd element: element count. Remaining elements: all zero because there
        # is no -inf, inf or nan.
        self.assertAllClose(trace.debug_tensor_value, [tensor_id, 1, 0, 0, 0])
    elif tensor_debug_mode == "FULL_HEALTH":
      for trace in graph_exec_traces:
        tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
        # Elements: [
        #   -1 is the unset tensor_id for eager op execution,
        #   device ID (set to -1 for now),
        #   dtype, rank, element_count,
        #   neg_inf_count, pos_inf_count, nan_count
        #   neg_finite_count, zero_count, pos_finite_count]
        self.assertAllClose(
            trace.debug_tensor_value,
            [tensor_id, -1, 1, 0, 1, 0, 0, 0, 0, 0, 1])
    elif tensor_debug_mode == "SHAPE":
      for trace in graph_exec_traces:
        if trace.op_type == "Mul":
          tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
          mul_value = reader.graph_execution_trace_to_tensor_value(trace)
          # 1st element: tensor_id, should be >= 0.
          # 2nd element: dtype enum value (float32).
          # 3rd element: rank.
          # 4th element: element count.
          self.assertAllClose(mul_value, [tensor_id, 1, 0, 1, 0, 0, 0, 0, 0, 0])
    elif tensor_debug_mode == "FULL_TENSOR":
      mul_values = [
          reader.graph_execution_trace_to_tensor_value(trace)
          for trace in graph_exec_traces if trace.op_type == "Mul"]
      self.assertAllClose(mul_values, [6.0, 6.0, 6.0, 6.0])

  def testMultiThreadedDumpingWithDifferentSettings(self):
    # [FUNCTION DOC]
    # Scenario: two threads enable dumping with different dump roots (full-tensor mode) and mutate
    # separate variables. Skipped on GPU (known flaky). The main thread squares and negates its
    # variable three times; the sub thread disables dumping before its negation. Asserts final
    # variable values, that the first dump root holds the Pow and Neg tensor values of the main
    # thread, and that the second dump root has the Pow value but no negation op execution.
    gpu_name = test_util.gpu_device_name()
    if gpu_name:
      self.skipTest("b/153671240: test is flaky on GPUs")
    dump_root_1 = os.path.join(self.dump_root, "dump_root_1")
    dump_root_2 = os.path.join(self.dump_root, "dump_root_2")
    v1 = variables.Variable(10.0, dtype=dtypes.float32)
    v2 = variables.Variable(3.0, dtype=dtypes.float32)

    def add_negative_v1_squared_to_itself():
      # [FUNCTION DOC]
      # Thread target: enables dumping to the first dump root in full-tensor mode, then three times
      # adds the negated square of the first variable to itself, and flushes the writer.
      writer = dumping_callback.enable_dump_debug_info(
          dump_root_1, tensor_debug_mode="FULL_TENSOR")
      # Run in a loop to facilitate interleaving between threads.
      for _ in range(3):
        v1.assign_add(-(v1 ** 2.0))
      writer.FlushNonExecutionFiles()
      writer.FlushExecutionFiles()

    def add_negative_v2_squared_to_itself():
      # [FUNCTION DOC]
      # Thread target: enables dumping to the second dump root, squares the second variable, then
      # disables dumping in this thread before negating and adding it to the variable, and flushes.
      writer = dumping_callback.enable_dump_debug_info(
          dump_root_2, tensor_debug_mode="FULL_TENSOR")
      v2_squared = v2 ** 2.0
      # Since dumping is disabled before the Neg op is called, no tensor data
      # should be dumped from the op, but this shouldn't affect the dumping of
      # the tensor data from the Neg op in `add_negative_v1_squared_to_itself`.
      # Both behavior is checked below.
      dumping_callback.disable_dump_debug_info()
      negative_v2_squared = -v2_squared
      v2.assign_add(negative_v2_squared)
      writer.FlushNonExecutionFiles()
      writer.FlushExecutionFiles()

    # v2 is mutated on a sub-thread.
    sub_thread = threading.Thread(target=add_negative_v2_squared_to_itself)
    sub_thread.start()
    add_negative_v1_squared_to_itself()  # v1 is mutated on the main thread.
    sub_thread.join()
    # 10 - 10 * 10 = -90.
    # -90 - (-90 * -90) = -8190.
    # -8190 - (-8190 * -8190) = -67084290.
    self.assertAllClose(v1.read_value(), -67084290.0)
    self.assertAllClose(v2.read_value(), -6.0)

    with debug_events_reader.DebugDataReader(dump_root_1) as reader:
      reader.update()
      exec_digests = reader.executions(digest=True)
      v1_squared_values = [
          reader.execution_to_tensor_values(digest)
          for digest in exec_digests if digest.op_type == "Pow"]
      negative_v1_squared_values = [
          reader.execution_to_tensor_values(digest)
          for digest in exec_digests if digest.op_type == "Neg"]
      self.assertAllClose(v1_squared_values, [[100.0], [8100.0], [67076100.0]])
      self.assertAllClose(
          negative_v1_squared_values, [[-100.0], [-8100.0], [-67076100.0]])

    with debug_events_reader.DebugDataReader(dump_root_2) as reader:
      reader.update()
      exec_digests = reader.executions(digest=True)
      executed_op_types = [digest.op_type for digest in exec_digests]
      self.assertNotIn("Neg", executed_op_types)
      v2_squared_values = [
          reader.execution_to_tensor_values(digest)
          for digest in exec_digests if digest.op_type == "Pow"]
      self.assertAllClose(v2_squared_values, [[9.0]])

  @test_util.run_in_graph_and_eager_modes
  def testNestedContextIsCapturedByGraphOpCreationHistory(self):
    # [FUNCTION DOC]
    # Scenario: a compiled function with a while loop that doubles and subtracts one, run four
    # times in no-tensor mode (result 17). Asserts that the less-than op (loop condition context)
    # has a different innermost graph ID than the multiply and subtract ops, while the multiply and
    # subtract ops (loop body context) share the same graph ID.
    x = constant_op.constant(2.0, dtype=dtypes.float32)
    times = constant_op.constant(4, dtype=dtypes.int32)
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode="NO_TENSOR")

    @def_function.function
    def iterative_doubling(x, times):
      # [FUNCTION DOC]
      # Traced function: loops a number of times, doubling the value and subtracting one each time.
      i = constant_op.constant(0, dtype=dtypes.int32)
      while i < times:
        x = x * 2.0 - 1.0
        i += 1
      return x

    # 2 * 2 - 1 = 3; 3 * 2 - 1 = 5; 5 * 2 - 1 = 9; 9 * 2 - 1 = 17.
    self.assertAllClose(self.evaluate(iterative_doubling(x, times)), 17.0)

    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()
    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      less_op_digest = reader.graph_op_digests(op_type="Less")[-1]
      mul_op_digest = reader.graph_op_digests(op_type="Mul")[-1]
      sub_op_digest = reader.graph_op_digests(op_type="Sub")[-1]
      # The Less op is from the while-loop cond context and hence should have
      # a different innermost context ID from the mul and sub ops, which are
      # both from the while-loop body context.
      self.assertNotEqual(less_op_digest.graph_id, mul_op_digest.graph_id)
      self.assertNotEqual(less_op_digest.graph_id, sub_op_digest.graph_id)
      # The Mul and Sub ops are from the same innermost context.
      self.assertEqual(mul_op_digest.graph_id, sub_op_digest.graph_id)

  @parameterized.named_parameters(
      ("NoTensor", "NO_TENSOR"),
      ("Shape", "SHAPE"),
      ("FullTensor", "FULL_TENSOR"),
  )
  @test_util.run_in_graph_and_eager_modes
  def testGraphInputTracingWorksWithConstAndPlaceholderTensors(
      self, tensor_debug_mode):
    # [FUNCTION DOC]
    # Scenario: a compiled function adds a constant to its argument and divides by the argument,
    # run under no-tensor, shape and full-tensor modes (graph and eager). After flushing, checks that
    # recorded graph op digests list the right input tensor names: the add op consumes the placeholder
    # then the constant; the divide op consumes the add output then the placeholder. Also asserts
    # that the add and divide ops were found.
    x = constant_op.constant(2.0)
    writer = dumping_callback.enable_dump_debug_info(
        self.dump_root, tensor_debug_mode=tensor_debug_mode)

    @def_function.function
    def func(x):
      # [FUNCTION DOC]
      # Traced function: adds a constant 4 to the input and divides by the input.
      return (x + constant_op.constant(4.0)) / x

    self.assertAllClose(self.evaluate(func(x)), 3.0)
    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      graph_op_digests = reader.graph_op_digests()
      placeholder_op_name = None
      const_op_name = None
      add_op_name = None
      div_op_name = None
      for op_digest in graph_op_digests:
        if op_digest.op_type == "Placeholder":
          placeholder_op_name = op_digest.op_name
        elif op_digest.op_type == "Const":
          const_op_name = op_digest.op_name
        elif op_digest.op_type == "AddV2":
          add_op_name = op_digest.op_name
          self.assertLen(op_digest.input_names, 2)
          self.assertEqual(op_digest.input_names[0], placeholder_op_name + ":0")
          self.assertEqual(op_digest.input_names[1], const_op_name + ":0")
        elif op_digest.op_type == "RealDiv":
          div_op_name = op_digest
          self.assertLen(op_digest.input_names, 2)
          self.assertEqual(op_digest.input_names[0], add_op_name + ":0")
          self.assertEqual(op_digest.input_names[1], placeholder_op_name + ":0")
      self.assertTrue(add_op_name)
      self.assertTrue(div_op_name)


if __name__ == "__main__":
  ops.enable_eager_execution()
  googletest.main()
