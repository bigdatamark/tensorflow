# [MODULE DOC]
# Tests for the Python DebugEventsWriter (tfdbg2 debug-event file writer) and its interplay with
# the DebugEventsReader / DebugDataReader classes. Covers: thread-safe construction, writing and
# reading source files, stack frames, graph op creations, debugged graphs, executions and graph
# execution traces, circular-buffer semantics, concurrent writes and random reads, range reads,
# multi-file-set (multi-host) dump roots, and JSON conversion of the reader's data objects.
# Test module only; its main block enables eager execution and runs the test main function.
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
"""Tests for the debug events writer Python class."""

import glob
import json as json_lib
import os
import re
import threading
import time

from absl.testing import parameterized

from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.debug.lib import debug_events_reader
from tensorflow.python.debug.lib import debug_events_writer
from tensorflow.python.debug.lib import dumping_callback_test_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import versions
from tensorflow.python.platform import googletest


class DebugEventsWriterTest(dumping_callback_test_lib.DumpingCallbackTestBase,
                            parameterized.TestCase):
  # [CLASS DOC]
  # Test case for DebugEventsWriter, built on the dumping-callback test base (which supplies a
  # temporary dump root directory and a run id) combined with parameterized testing. Each test
  # writes debug events with a writer and verifies them through the reader classes.

  def testMultiThreadedConstructorCallWorks(self):
    # [FUNCTION DOC]
    # Scenario: four threads construct a writer on the same dump root and run id at once. Asserts that
    # exactly one file of each type (metadata, source files, stack frames, graphs) exists, then reads
    # and checks the metadata file.
    def init_writer():
      # [FUNCTION DOC]
      # HELPER: thread target for the multi-threaded constructor test.
      # Constructs a writer on the test dump root with the test run id; returns nothing.
      debug_events_writer.DebugEventsWriter(self.dump_root, self.tfdbg_run_id)

    num_threads = 4
    threads = []
    for _ in range(num_threads):
      thread = threading.Thread(target=init_writer)
      thread.start()
      threads.append(thread)
    for thread in threads:
      thread.join()

    # Verify that there is only one debug event file of each type.
    metadata_paths = glob.glob(os.path.join(self.dump_root, "*.metadata"))
    self.assertLen(metadata_paths, 1)
    source_files_paths = glob.glob(
        os.path.join(self.dump_root, "*.source_files"))
    self.assertLen(source_files_paths, 1)
    stack_frames_paths = glob.glob(
        os.path.join(self.dump_root, "*.stack_frames"))
    self.assertLen(stack_frames_paths, 1)
    graphs_paths = glob.glob(os.path.join(self.dump_root, "*.graphs"))
    self.assertLen(graphs_paths, 1)
    self._readAndCheckMetadataFile()

  def testWriteSourceFilesAndStackFrames(self):
    # [FUNCTION DOC]
    # Scenario: writes ten source-file records and ten stack-frame records, flushes the non-execution
    # files, then reads them back. Asserts the count, file path, host name, lines, frame ids and
    # file index of every record match what was written, in order.
    writer = debug_events_writer.DebugEventsWriter(self.dump_root,
                                                   self.tfdbg_run_id)
    num_protos = 10
    for i in range(num_protos):
      source_file = debug_event_pb2.SourceFile()
      source_file.file_path = "/home/tf2user/main.py"
      source_file.host_name = "machine.cluster"
      source_file.lines.append("print(%d)" % i)
      writer.WriteSourceFile(source_file)

      stack_frame = debug_event_pb2.StackFrameWithId()
      stack_frame.id = "stack_%d" % i
      stack_frame.file_line_col.file_index = i * 10
      writer.WriteStackFrameWithId(stack_frame)

    writer.FlushNonExecutionFiles()

    with debug_events_reader.DebugEventsReader(self.dump_root) as reader:
      actuals = list(item.debug_event.source_file
                     for item in reader.source_files_iterator())
      self.assertLen(actuals, num_protos)
      for i in range(num_protos):
        self.assertEqual(actuals[i].file_path, "/home/tf2user/main.py")
        self.assertEqual(actuals[i].host_name, "machine.cluster")
        self.assertEqual(actuals[i].lines, ["print(%d)" % i])

      actuals = list(item.debug_event.stack_frame_with_id
                     for item in reader.stack_frames_iterator())
      self.assertLen(actuals, num_protos)
      for i in range(num_protos):
        self.assertEqual(actuals[i].id, "stack_%d" % i)
        self.assertEqual(actuals[i].file_line_col.file_index, i * 10)

  def testWriteGraphOpCreationAndDebuggedGraphs(self):
    # [FUNCTION DOC]
    # Scenario: writes ten graph-op-creation records and one debugged-graph record, flushes, and reads
    # the graphs file. Asserts eleven events in order: op types and names for the first ten and the
    # graph id of the final debugged graph.
    writer = debug_events_writer.DebugEventsWriter(self.dump_root,
                                                   self.tfdbg_run_id)
    num_op_creations = 10
    for i in range(num_op_creations):
      graph_op_creation = debug_event_pb2.GraphOpCreation()
      graph_op_creation.op_type = "Conv2D"
      graph_op_creation.op_name = "Conv2D_%d" % i
      writer.WriteGraphOpCreation(graph_op_creation)
    debugged_graph = debug_event_pb2.DebuggedGraph()
    debugged_graph.graph_id = "deadbeaf"
    debugged_graph.graph_name = "MyGraph1"
    writer.WriteDebuggedGraph(debugged_graph)
    writer.FlushNonExecutionFiles()

    reader = debug_events_reader.DebugEventsReader(self.dump_root)
    actuals = list(item.debug_event for item in reader.graphs_iterator())
    self.assertLen(actuals, num_op_creations + 1)
    for i in range(num_op_creations):
      self.assertEqual(actuals[i].graph_op_creation.op_type, "Conv2D")
      self.assertEqual(actuals[i].graph_op_creation.op_name, "Conv2D_%d" % i)
    self.assertEqual(actuals[num_op_creations].debugged_graph.graph_id,
                     "deadbeaf")

  def testConcurrentWritesToNonExecutionFilesWorks(self):
    # [FUNCTION DOC]
    # Scenario: nine threads concurrently write source files, stack frames and graph op creations
    # (three of each kind, each with a counter guarded by a lock), flushing after each write.
    # Asserts each of the three files contains exactly the three expected entries (sorted).
    writer = debug_events_writer.DebugEventsWriter(self.dump_root,
                                                   self.tfdbg_run_id)

    source_file_state = {"counter": 0, "lock": threading.Lock()}

    def writer_source_file():
      # [FUNCTION DOC]
      # HELPER: thread target. Under a lock, builds a source-file record with a unique numbered path,
      # increments the shared counter, writes it and flushes the non-execution files. Returns nothing.
      source_file = debug_event_pb2.SourceFile()
      with source_file_state["lock"]:
        source_file.file_path = "/home/tf2user/file_%d.py" % source_file_state[
            "counter"]
        source_file_state["counter"] += 1
      writer.WriteSourceFile(source_file)
      # More-frequent-than-necessary concurrent flushing is not recommended,
      # but tolerated.
      writer.FlushNonExecutionFiles()

    stack_frame_state = {"counter": 0, "lock": threading.Lock()}

    def write_stack_frame():
      # [FUNCTION DOC]
      # HELPER: thread target. Under a lock, builds a stack-frame record with a unique numbered id,
      # increments the shared counter, writes it and flushes the non-execution files. Returns nothing.
      stack_frame = debug_event_pb2.StackFrameWithId()
      with stack_frame_state["lock"]:
        stack_frame.id = "stack_frame_%d" % stack_frame_state["counter"]
        stack_frame_state["counter"] += 1
      writer.WriteStackFrameWithId(stack_frame)
      # More-frequent-than-necessary concurrent flushing is not recommended,
      # but tolerated.
      writer.FlushNonExecutionFiles()

    graph_op_state = {"counter": 0, "lock": threading.Lock()}

    def write_graph_op_creation():
      # [FUNCTION DOC]
      # HELPER: thread target. Under a lock, builds a graph-op-creation record with a unique numbered
      # op name, increments the shared counter, writes it and flushes the non-execution files.
      # Returns nothing.
      graph_op_creation = debug_event_pb2.GraphOpCreation()
      with graph_op_state["lock"]:
        graph_op_creation.op_name = "Op%d" % graph_op_state["counter"]
        graph_op_state["counter"] += 1
      writer.WriteGraphOpCreation(graph_op_creation)
      # More-frequent-than-necessary concurrent flushing is not recommended,
      # but tolerated.
      writer.FlushNonExecutionFiles()

    num_threads = 9
    threads = []
    for i in range(num_threads):
      if i % 3 == 0:
        target = writer_source_file
      elif i % 3 == 1:
        target = write_stack_frame
      else:
        target = write_graph_op_creation
      thread = threading.Thread(target=target)
      thread.start()
      threads.append(thread)
    for thread in threads:
      thread.join()

    # Verify the content of the .source_files file.
    with debug_events_reader.DebugEventsReader(self.dump_root) as reader:
      source_files_iter = reader.source_files_iterator()
      actuals = list(item.debug_event.source_file for item in source_files_iter)
      file_paths = sorted([actual.file_path for actual in actuals])
      self.assertEqual(file_paths, [
          "/home/tf2user/file_0.py", "/home/tf2user/file_1.py",
          "/home/tf2user/file_2.py"
      ])

    # Verify the content of the .stack_frames file.
    actuals = list(item.debug_event.stack_frame_with_id
                   for item in reader.stack_frames_iterator())
    stack_frame_ids = sorted([actual.id for actual in actuals])
    self.assertEqual(stack_frame_ids,
                     ["stack_frame_0", "stack_frame_1", "stack_frame_2"])

    # Verify the content of the .graphs file.
    actuals = list(item.debug_event.graph_op_creation
                   for item in reader.graphs_iterator())
    graph_op_names = sorted([actual.op_name for actual in actuals])
    self.assertEqual(graph_op_names, ["Op0", "Op1", "Op2"])

  def testWriteAndReadMetadata(self):
    # [FUNCTION DOC]
    # Scenario: records the time, creates and closes a writer, then opens a data reader. Asserts the
    # starting wall time is a float no earlier than the recorded time, the TensorFlow version equals
    # the library version, and a non-empty run id is reported.
    t0 = time.time()
    writer = debug_events_writer.DebugEventsWriter(self.dump_root,
                                                   self.tfdbg_run_id)
    writer.Close()
    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      self.assertIsInstance(reader.starting_wall_time(), float)
      self.assertGreaterEqual(reader.starting_wall_time(), t0)
      self.assertEqual(reader.tensorflow_version(), versions.__version__)
      self.assertTrue(reader.tfdbg_run_id())

  def testWriteExecutionEventsWithCircularBuffer(self):
    # [FUNCTION DOC]
    # Scenario: with the default circular buffer size, writes twice that many execution events.
    # Asserts nothing is on disk before flushing; after flushing, the reader sees only the most
    # recent events (op types offset by the buffer size).
    writer = debug_events_writer.DebugEventsWriter(self.dump_root,
                                                   self.tfdbg_run_id)
    num_execution_events = debug_events_writer.DEFAULT_CIRCULAR_BUFFER_SIZE * 2
    for i in range(num_execution_events):
      execution = debug_event_pb2.Execution()
      execution.op_type = "OpType%d" % i
      writer.WriteExecution(execution)

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      # Before FlushExecutionFiles() is called. No data should have been written
      # to the file.
      reader.update()
      self.assertFalse(reader.executions())

      writer.FlushExecutionFiles()
      reader.update()
      executions = reader.executions()
      for i, execution in enumerate(executions):
        self.assertEqual(
            execution.op_type,
            "OpType%d" % (i + debug_events_writer.DEFAULT_CIRCULAR_BUFFER_SIZE))

  def testWriteExecutionEventsWithoutCircularBufferBehavior(self):
    # [FUNCTION DOC]
    # Scenario: a buffer size of zero disables circular behavior. Writes twice the default size of
    # execution events and flushes. Asserts all events are read back in order.
    # A circular buffer size of 0 abolishes the circular buffer behavior.
    writer = debug_events_writer.DebugEventsWriter(self.dump_root,
                                                   self.tfdbg_run_id, 0)
    num_execution_events = debug_events_writer.DEFAULT_CIRCULAR_BUFFER_SIZE * 2
    for i in range(num_execution_events):
      execution = debug_event_pb2.Execution()
      execution.op_type = "OpType%d" % i
      writer.WriteExecution(execution)
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      executions = reader.executions()
      self.assertLen(executions, num_execution_events)
      for i, execution in enumerate(executions):
        self.assertEqual(execution.op_type, "OpType%d" % i)

  def testWriteGraphExecutionTraceEventsWithCircularBuffer(self):
    # [FUNCTION DOC]
    # Scenario: with the default circular buffer size, writes twice that many graph execution traces.
    # Asserts none are readable before flushing; after flushing exactly buffer-size traces remain,
    # being the most recent ones.
    writer = debug_events_writer.DebugEventsWriter(self.dump_root,
                                                   self.tfdbg_run_id)
    num_execution_events = debug_events_writer.DEFAULT_CIRCULAR_BUFFER_SIZE * 2
    for i in range(num_execution_events):
      trace = debug_event_pb2.GraphExecutionTrace()
      trace.op_name = "Op%d" % i
      writer.WriteGraphExecutionTrace(trace)

    with debug_events_reader.DebugEventsReader(self.dump_root) as reader:
      actuals = list(reader.graph_execution_traces_iterators()[0])
      # Before FlushExecutionFiles() is called. No data should have been written
      # to the file.
      self.assertEmpty(actuals)

      writer.FlushExecutionFiles()
      actuals = list(item.debug_event.graph_execution_trace
                     for item in reader.graph_execution_traces_iterators()[0])
      self.assertLen(actuals, debug_events_writer.DEFAULT_CIRCULAR_BUFFER_SIZE)
      for i in range(debug_events_writer.DEFAULT_CIRCULAR_BUFFER_SIZE):
        self.assertEqual(
            actuals[i].op_name,
            "Op%d" % (i + debug_events_writer.DEFAULT_CIRCULAR_BUFFER_SIZE))

  def testWriteGraphExecutionTraceEventsWithoutCircularBufferBehavior(self):
    # [FUNCTION DOC]
    # Scenario: buffer size zero disables the circular behavior. Writes twice the default size of
    # graph execution traces and flushes. Asserts all are read back, in order.
    # A circular buffer size of 0 abolishes the circular buffer behavior.
    writer = debug_events_writer.DebugEventsWriter(self.dump_root,
                                                   self.tfdbg_run_id, 0)
    num_execution_events = debug_events_writer.DEFAULT_CIRCULAR_BUFFER_SIZE * 2
    for i in range(num_execution_events):
      trace = debug_event_pb2.GraphExecutionTrace()
      trace.op_name = "Op%d" % i
      writer.WriteGraphExecutionTrace(trace)
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugEventsReader(self.dump_root) as reader:
      actuals = list(item.debug_event.graph_execution_trace
                     for item in reader.graph_execution_traces_iterators()[0])
    self.assertLen(actuals, num_execution_events)
    for i in range(num_execution_events):
      self.assertEqual(actuals[i].op_name, "Op%d" % i)

  def testConcurrentWritesToExecutionFiles(self):
    # [FUNCTION DOC]
    # Scenario: small circular buffer (five); twenty threads alternate between writing executions and
    # writing graph-op-creation plus graph-execution-trace pairs, after registering a debugged graph.
    # After flushing, asserts exactly buffer-size executions and traces survive, all unique.
    circular_buffer_size = 5
    writer = debug_events_writer.DebugEventsWriter(self.dump_root,
                                                   self.tfdbg_run_id,
                                                   circular_buffer_size)
    debugged_graph = debug_event_pb2.DebuggedGraph(graph_id="graph1",
                                                   graph_name="graph1")
    writer.WriteDebuggedGraph(debugged_graph)

    execution_state = {"counter": 0, "lock": threading.Lock()}

    def write_execution():
      # [FUNCTION DOC]
      # HELPER: thread target. Under a lock builds an execution event with a unique numbered op type,
      # then writes it (outside the lock). Returns nothing.
      execution = debug_event_pb2.Execution()
      with execution_state["lock"]:
        execution.op_type = "OpType%d" % execution_state["counter"]
        execution_state["counter"] += 1
      writer.WriteExecution(execution)

    graph_execution_trace_state = {"counter": 0, "lock": threading.Lock()}

    def write_graph_execution_trace():
      # [FUNCTION DOC]
      # HELPER: thread target. Under a lock builds a uniquely named graph-op-creation and matching
      # execution trace, then writes both (outside the lock). Returns nothing.
      with graph_execution_trace_state["lock"]:
        op_name = "Op%d" % graph_execution_trace_state["counter"]
        graph_op_creation = debug_event_pb2.GraphOpCreation(
            op_type="FooOp", op_name=op_name, graph_id="graph1")
        trace = debug_event_pb2.GraphExecutionTrace(
            op_name=op_name, tfdbg_context_id="graph1")
        graph_execution_trace_state["counter"] += 1
      writer.WriteGraphOpCreation(graph_op_creation)
      writer.WriteGraphExecutionTrace(trace)

    threads = []
    for i in range(circular_buffer_size * 4):
      if i % 2 == 0:
        target = write_execution
      else:
        target = write_graph_execution_trace
      thread = threading.Thread(target=target)
      thread.start()
      threads.append(thread)
    for thread in threads:
      thread.join()
    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      # Verify the content of the .execution file.
      executions = reader.executions()
      executed_op_types = [execution.op_type for execution in executions]
      self.assertLen(executed_op_types, circular_buffer_size)
      self.assertLen(executed_op_types, len(set(executed_op_types)))

      # Verify the content of the .graph_execution_traces file.
      op_names = [trace.op_name for trace in reader.graph_execution_traces()]
      self.assertLen(op_names, circular_buffer_size)
      self.assertLen(op_names, len(set(op_names)))

  def testConcurrentSourceFileRandomReads(self):
    # [FUNCTION DOC]
    # Scenario: writes 100 source files and flushes, then two threads read disjoint halves of the
    # source lines concurrently (one in reverse order). Asserts every file returns its expected line.
    writer = debug_events_writer.DebugEventsWriter(self.dump_root,
                                                   self.tfdbg_run_id)

    for i in range(100):
      source_file = debug_event_pb2.SourceFile(
          host_name="localhost", file_path="/tmp/file_%d.py" % i)
      source_file.lines.append("# File %d" % i)
      writer.WriteSourceFile(source_file)
    writer.FlushNonExecutionFiles()

    reader = debug_events_reader.DebugDataReader(self.dump_root)
    reader.update()
    lines = [None] * 100
    def read_job_1():
      # [FUNCTION DOC]
      # HELPER: thread target. Reads source lines for files 49 down to 0 into the shared list.
      # Returns nothing.
      # Read in the reverse order to enhance randomness of the read access.
      for i in range(49, -1, -1):
        lines[i] = reader.source_lines("localhost", "/tmp/file_%d.py" % i)
    def read_job_2():
      # [FUNCTION DOC]
      # HELPER: thread target. Reads source lines for files 99 down to 50 into the shared list.
      # Returns nothing.
      for i in range(99, 49, -1):
        lines[i] = reader.source_lines("localhost", "/tmp/file_%d.py" % i)
    thread_1 = threading.Thread(target=read_job_1)
    thread_2 = threading.Thread(target=read_job_2)
    thread_1.start()
    thread_2.start()
    thread_1.join()
    thread_2.join()
    for i in range(100):
      self.assertEqual(lines[i], ["# File %d" % i])

  def testConcurrentExecutionUpdateAndRandomRead(self):
    # [FUNCTION DOC]
    # Scenario: unbounded buffer; a sub-thread keeps writing, flushing and updating the reader while
    # the main thread polls for execution digests. Once one appears, asserts the first execution read
    # randomly has op type of the first written event, then stops the writer thread and joins it.
    circular_buffer_size = -1
    writer = debug_events_writer.DebugEventsWriter(self.dump_root,
                                                   self.tfdbg_run_id,
                                                   circular_buffer_size)

    writer_state = {"counter": 0, "done": False}

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      def write_and_update_job():
        # [FUNCTION DOC]
        # HELPER: sub-thread target. Loops until the shared done flag is set, each time writing a new
        # execution with an increasing counter, flushing execution files and updating the reader.
        while True:
          if writer_state["done"]:
            break
          execution = debug_event_pb2.Execution()
          execution.op_type = "OpType%d" % writer_state["counter"]
          writer_state["counter"] += 1
          writer.WriteExecution(execution)
          writer.FlushExecutionFiles()
          reader.update()
      # On the sub-thread, keep writing and reading new Execution protos.
      write_and_update_thread = threading.Thread(target=write_and_update_job)
      write_and_update_thread.start()
      # On the main thread, do concurrent random read.
      while True:
        exec_digests = reader.executions(digest=True)
        if exec_digests:
          exec_0 = reader.read_execution(exec_digests[0])
          self.assertEqual(exec_0.op_type, "OpType0")
          writer_state["done"] = True
          break
        else:
          time.sleep(0.1)
          continue
      write_and_update_thread.join()

  def testConcurrentExecutionRandomReads(self):
    # [FUNCTION DOC]
    # Scenario: writes 100 executions with an unbounded buffer, flushes, then two threads read
    # disjoint halves of the executions by digest concurrently. Asserts each has the expected op type.
    circular_buffer_size = -1
    writer = debug_events_writer.DebugEventsWriter(self.dump_root,
                                                   self.tfdbg_run_id,
                                                   circular_buffer_size)

    for i in range(100):
      execution = debug_event_pb2.Execution()
      execution.op_type = "OpType%d" % i
      writer.WriteExecution(execution)
    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    reader = debug_events_reader.DebugDataReader(self.dump_root)
    reader.update()
    executions = [None] * 100
    def read_job_1():
      # [FUNCTION DOC]
      # HELPER: thread target. Reads executions 49 down to 0 by digest into the shared list.
      # Returns nothing.
      execution_digests = reader.executions(digest=True)
      # Read in the reverse order to enhance randomness of the read access.
      for i in range(49, -1, -1):
        execution = reader.read_execution(execution_digests[i])
        executions[i] = execution
    def read_job_2():
      # [FUNCTION DOC]
      # HELPER: thread target. Reads executions 99 down to 50 by digest into the shared list.
      # Returns nothing.
      execution_digests = reader.executions(digest=True)
      for i in range(99, 49, -1):
        execution = reader.read_execution(execution_digests[i])
        executions[i] = execution
    thread_1 = threading.Thread(target=read_job_1)
    thread_2 = threading.Thread(target=read_job_2)
    thread_1.start()
    thread_2.start()
    thread_1.join()
    thread_2.join()
    for i in range(100):
      self.assertEqual(executions[i].op_type, "OpType%d" % i)

  def testConcurrentGraphExecutionTraceUpdateAndRandomRead(self):
    # [FUNCTION DOC]
    # Scenario: like the execution update/random-read test but for graph execution traces: a
    # sub-thread keeps writing op creations and traces, flushing and updating the reader, while the
    # main thread polls for trace digests and asserts the first trace has the first op name.
    circular_buffer_size = -1
    writer = debug_events_writer.DebugEventsWriter(self.dump_root,
                                                   self.tfdbg_run_id,
                                                   circular_buffer_size)
    debugged_graph = debug_event_pb2.DebuggedGraph(graph_id="graph1",
                                                   graph_name="graph1")
    writer.WriteDebuggedGraph(debugged_graph)

    writer_state = {"counter": 0, "done": False}

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      def write_and_update_job():
        # [FUNCTION DOC]
        # HELPER: sub-thread target. Loops until the done flag is set, writing a numbered graph op
        # creation and trace, flushing both file kinds and updating the reader. Returns nothing.
        while True:
          if writer_state["done"]:
            break
          op_name = "Op%d" % writer_state["counter"]
          graph_op_creation = debug_event_pb2.GraphOpCreation(
              op_type="FooOp", op_name=op_name, graph_id="graph1")
          writer.WriteGraphOpCreation(graph_op_creation)
          trace = debug_event_pb2.GraphExecutionTrace(
              op_name=op_name, tfdbg_context_id="graph1")
          writer.WriteGraphExecutionTrace(trace)
          writer_state["counter"] += 1
          writer.FlushNonExecutionFiles()
          writer.FlushExecutionFiles()
          reader.update()
      # On the sub-thread, keep writing and reading new GraphExecutionTraces.
      write_and_update_thread = threading.Thread(target=write_and_update_job)
      write_and_update_thread.start()
      # On the main thread, do concurrent random read.
      while True:
        digests = reader.graph_execution_traces(digest=True)
        if digests:
          trace_0 = reader.read_graph_execution_trace(digests[0])
          self.assertEqual(trace_0.op_name, "Op0")
          writer_state["done"] = True
          break
        else:
          time.sleep(0.1)
          continue
      write_and_update_thread.join()

  def testConcurrentGraphExecutionTraceRandomReads(self):
    # [FUNCTION DOC]
    # Scenario: writes 100 op-creation and trace pairs, flushes, then two threads concurrently read
    # disjoint halves of the traces by digest. Asserts each trace has the expected op name.
    circular_buffer_size = -1
    writer = debug_events_writer.DebugEventsWriter(self.dump_root,
                                                   self.tfdbg_run_id,
                                                   circular_buffer_size)
    debugged_graph = debug_event_pb2.DebuggedGraph(graph_id="graph1",
                                                   graph_name="graph1")
    writer.WriteDebuggedGraph(debugged_graph)

    for i in range(100):
      op_name = "Op%d" % i
      graph_op_creation = debug_event_pb2.GraphOpCreation(
          op_type="FooOp", op_name=op_name, graph_id="graph1")
      writer.WriteGraphOpCreation(graph_op_creation)
      trace = debug_event_pb2.GraphExecutionTrace(
          op_name=op_name, tfdbg_context_id="graph1")
      writer.WriteGraphExecutionTrace(trace)
    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()

    reader = debug_events_reader.DebugDataReader(self.dump_root)
    reader.update()
    traces = [None] * 100
    def read_job_1():
      # [FUNCTION DOC]
      # HELPER: thread target. Reads graph execution traces 49 down to 0 by digest. Returns nothing.
      digests = reader.graph_execution_traces(digest=True)
      for i in range(49, -1, -1):
        traces[i] = reader.read_graph_execution_trace(digests[i])
    def read_job_2():
      # [FUNCTION DOC]
      # HELPER: thread target. Reads graph execution traces 99 down to 50 by digest. Returns nothing.
      digests = reader.graph_execution_traces(digest=True)
      for i in range(99, 49, -1):
        traces[i] = reader.read_graph_execution_trace(digests[i])
    thread_1 = threading.Thread(target=read_job_1)
    thread_2 = threading.Thread(target=read_job_2)
    thread_1.start()
    thread_2.start()
    thread_1.join()
    thread_2.join()
    for i in range(100):
      self.assertEqual(traces[i].op_name, "Op%d" % i)

  @parameterized.named_parameters(
      ("Begin1End3", 1, 3, 1, 3),
      ("Begin0End3", 0, 3, 0, 3),
      ("Begin0EndNeg1", 0, -1, 0, 4),
      ("BeginNoneEnd3", None, 3, 0, 3),
      ("Begin2EndNone", 2, None, 2, 5),
      ("BeginNoneEndNone", None, None, 0, 5),
  )
  def testRangeReadingExecutions(self, begin, end, expected_begin,
                                 expected_end):
    # [FUNCTION DOC]
    # Parameterized scenario: writes five executions, closes the writer, and reads with the given
    # begin and end (either may be nothing/null; negative end counts from the back). Asserts the
    # count equals the expected range length and the first and last op types match the range.
    # Parameters: begin, end (requested range), expected_begin, expected_end (resolved bounds).
    writer = debug_events_writer.DebugEventsWriter(
        self.dump_root, self.tfdbg_run_id, circular_buffer_size=-1)
    for i in range(5):
      execution = debug_event_pb2.Execution(op_type="OpType%d" % i)
      writer.WriteExecution(execution)
    writer.FlushExecutionFiles()
    writer.Close()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      executions = reader.executions(begin=begin, end=end)
    self.assertLen(executions, expected_end - expected_begin)
    self.assertEqual(executions[0].op_type, "OpType%d" % expected_begin)
    self.assertEqual(executions[-1].op_type, "OpType%d" % (expected_end - 1))

  @parameterized.named_parameters(
      ("Begin1End3", 1, 3, 1, 3),
      ("Begin0End3", 0, 3, 0, 3),
      ("Begin0EndNeg1", 0, -1, 0, 4),
      ("BeginNoneEnd3", None, 3, 0, 3),
      ("Begin2EndNone", 2, None, 2, 5),
      ("BeginNoneEndNone", None, None, 0, 5),
  )
  def testRangeReadingGraphExecutionTraces(self, begin, end, expected_begin,
                                           expected_end):
    # [FUNCTION DOC]
    # Parameterized scenario: writes a debugged graph and five op-creation/trace pairs, closes the
    # writer, and reads traces with the given begin and end (nullable; negative end counts from the
    # back). Asserts the count and the first and last op names match the expected range.
    # Parameters: begin, end (requested range), expected_begin, expected_end (resolved bounds).
    writer = debug_events_writer.DebugEventsWriter(
        self.dump_root, self.tfdbg_run_id, circular_buffer_size=-1)
    debugged_graph = debug_event_pb2.DebuggedGraph(
        graph_id="graph1", graph_name="graph1")
    writer.WriteDebuggedGraph(debugged_graph)
    for i in range(5):
      op_name = "Op_%d" % i
      graph_op_creation = debug_event_pb2.GraphOpCreation(
          op_name=op_name, graph_id="graph1")
      writer.WriteGraphOpCreation(graph_op_creation)
      trace = debug_event_pb2.GraphExecutionTrace(
          op_name=op_name, tfdbg_context_id="graph1")
      writer.WriteGraphExecutionTrace(trace)
    writer.FlushNonExecutionFiles()
    writer.FlushExecutionFiles()
    writer.Close()

    with debug_events_reader.DebugDataReader(self.dump_root) as reader:
      reader.update()
      traces = reader.graph_execution_traces(begin=begin, end=end)
    self.assertLen(traces, expected_end - expected_begin)
    self.assertEqual(traces[0].op_name, "Op_%d" % expected_begin)
    self.assertEqual(traces[-1].op_name, "Op_%d" % (expected_end - 1))


class MultiSetReaderTest(dumping_callback_test_lib.DumpingCallbackTestBase):
  # [CLASS DOC]
  # Tests that DebugDataReader can combine multiple file sets (as from multiple hosts) found under
  # one dump root, and rejects sets with differing run ids.
  """Test for DebugDataReader for multiple file sets under a dump root."""

  def testReadingTwoFileSetsWithTheSameDumpRootSucceeds(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the read loops use a stale loop variable left over from the earlier writing loop
    # instead of their own counter, so the same index is read on every iteration rather than each
    # of the ten traces per file set.
    # Scenario: writes two file sets in subdirectories with the same run id (the first also holds a
    # debugged graph and two op creations; each has ten traces), moves the second set into the first
    # directory with renamed files, and reads. Asserts twenty trace digests were found and the read
    # traces carry the expected op names.
    # To simulate a multi-host data dump, we first generate file sets in two
    # different directories, with the same tfdbg_run_id, and then combine them.
    tfdbg_run_id = "foo"
    for i in range(2):
      writer = debug_events_writer.DebugEventsWriter(
          os.path.join(self.dump_root, str(i)),
          tfdbg_run_id,
          circular_buffer_size=-1)
      if i == 0:
        debugged_graph = debug_event_pb2.DebuggedGraph(
            graph_id="graph1", graph_name="graph1")
        writer.WriteDebuggedGraph(debugged_graph)
        op_name = "Op_0"
        graph_op_creation = debug_event_pb2.GraphOpCreation(
            op_type="FooOp", op_name=op_name, graph_id="graph1")
        writer.WriteGraphOpCreation(graph_op_creation)
        op_name = "Op_1"
        graph_op_creation = debug_event_pb2.GraphOpCreation(
            op_type="FooOp", op_name=op_name, graph_id="graph1")
        writer.WriteGraphOpCreation(graph_op_creation)
      for _ in range(10):
        trace = debug_event_pb2.GraphExecutionTrace(
            op_name="Op_%d" % i, tfdbg_context_id="graph1")
        writer.WriteGraphExecutionTrace(trace)
        writer.FlushNonExecutionFiles()
        writer.FlushExecutionFiles()

    # Move all files from the subdirectory /1 to subdirectory /0.
    dump_root_0 = os.path.join(self.dump_root, "0")
    src_paths = glob.glob(os.path.join(self.dump_root, "1", "*"))
    for src_path in src_paths:
      dst_path = os.path.join(
          dump_root_0,
          # Rename the file set to avoid file name collision.
          re.sub(r"(tfdbg_events\.\d+)", r"\g<1>1", os.path.basename(src_path)))
      os.rename(src_path, dst_path)

    with debug_events_reader.DebugDataReader(dump_root_0) as reader:
      reader.update()
      # Verify the content of the .graph_execution_traces file.
      trace_digests = reader.graph_execution_traces(digest=True)
      self.assertLen(trace_digests, 20)
      for _ in range(10):
        trace = reader.read_graph_execution_trace(trace_digests[i])
        self.assertEqual(trace.op_name, "Op_0")
      for _ in range(10):
        trace = reader.read_graph_execution_trace(trace_digests[i + 10])
        self.assertEqual(trace.op_name, "Op_1")

  def testReadingTwoFileSetsWithTheDifferentRootsLeadsToError(self):
    # [FUNCTION DOC]
    # Scenario: writes two file sets with different run ids, merges them into one directory with
    # renamed files, and constructs a reader. Asserts a value error stating that multiple (two)
    # tfdbg2 runs were found.
    # To simulate a multi-host data dump, we first generate file sets in two
    # different directories, with different tfdbg_run_ids, and then combine
    # them.
    for i in range(2):
      writer = debug_events_writer.DebugEventsWriter(
          os.path.join(self.dump_root, str(i)),
          "run_id_%d" % i,
          circular_buffer_size=-1)
      writer.FlushNonExecutionFiles()
      writer.FlushExecutionFiles()

    # Move all files from the subdirectory /1 to subdirectory /0.
    dump_root_0 = os.path.join(self.dump_root, "0")
    src_paths = glob.glob(os.path.join(self.dump_root, "1", "*"))
    for src_path in src_paths:
      dst_path = os.path.join(
          dump_root_0,
          # Rename the file set to avoid file name collision.
          re.sub(r"(tfdbg_events\.\d+)", r"\g<1>1", os.path.basename(src_path)))
      os.rename(src_path, dst_path)

    with self.assertRaisesRegex(ValueError,
                                r"Found multiple \(2\) tfdbg2 runs"):
      debug_events_reader.DebugDataReader(dump_root_0)


class DataObjectsTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests the plain data objects of the reader module (execution digests, executions, debugged
  # devices and graphs, op-creation digests, trace digests and traces): their JSON conversion and
  # derived properties.

  def jsonRoundTripCheck(self, obj):
    # [FUNCTION DOC]
    # HELPER: shared assertion for the JSON tests. Serializes the object to text, parses it back,
    # serializes again with sorted keys and asserts equality with the direct sorted serialization.
    # Parameter obj: JSON-serializable value. Returns nothing; fails the test on mismatch.
    self.assertEqual(
        json_lib.dumps(json_lib.loads(json_lib.dumps(obj)), sort_keys=True),
        json_lib.dumps(obj, sort_keys=True))

  def testExecutionDigestWithNoOutputToJson(self):
    # [FUNCTION DOC]
    # Scenario: an execution digest without output device ids is converted to JSON. Asserts the round
    # trip, wall time, op type, and that the output device ids field is null.
    execution_digest = debug_events_reader.ExecutionDigest(
        1234, 5678, "FooOp", output_tensor_device_ids=None)
    json = execution_digest.to_json()
    self.jsonRoundTripCheck(json)
    self.assertEqual(json["wall_time"], 1234)
    self.assertEqual(json["op_type"], "FooOp")
    self.assertEqual(json["output_tensor_device_ids"], None)

  def testExecutionDigestWithTwoOutputsToJson(self):
    # [FUNCTION DOC]
    # Scenario: an execution digest with two output device ids is converted to JSON. Asserts the round
    # trip, wall time, op type, and that the device ids are given as a two-element tuple.
    execution_digest = debug_events_reader.ExecutionDigest(
        1234, 5678, "FooOp", output_tensor_device_ids=[1357, 2468])
    json = execution_digest.to_json()
    self.jsonRoundTripCheck(json)
    self.assertEqual(json["wall_time"], 1234)
    self.assertEqual(json["op_type"], "FooOp")
    self.assertEqual(json["output_tensor_device_ids"], (1357, 2468))

  def testExecutionNoGraphNoInputToJson(self):
    # [FUNCTION DOC]
    # Scenario: an execution with no graph id and no input tensor ids but with outputs and debug
    # values is converted to JSON. Asserts the round trip and that every field is reported as given,
    # with absent fields null and sequences as tuples.
    execution_digest = debug_events_reader.ExecutionDigest(
        1234, 5678, "FooOp", output_tensor_device_ids=[1357])
    execution = debug_events_reader.Execution(
        execution_digest,
        "localhost",
        ("a1", "b2"),
        debug_event_pb2.TensorDebugMode.CURT_HEALTH,
        graph_id=None,
        input_tensor_ids=None,
        output_tensor_ids=[2468],
        debug_tensor_values=([1, 0],))
    json = execution.to_json()
    self.jsonRoundTripCheck(json)
    self.assertEqual(json["wall_time"], 1234)
    self.assertEqual(json["op_type"], "FooOp")
    self.assertEqual(json["output_tensor_device_ids"], (1357,))
    self.assertEqual(json["host_name"], "localhost")
    self.assertEqual(json["stack_frame_ids"], ("a1", "b2"))
    self.assertEqual(json["tensor_debug_mode"],
                     debug_event_pb2.TensorDebugMode.CURT_HEALTH)
    self.assertIsNone(json["graph_id"])
    self.assertIsNone(json["input_tensor_ids"])
    self.assertEqual(json["output_tensor_ids"], (2468,))
    self.assertEqual(json["debug_tensor_values"], ([1, 0],))

  def testExecutionNoGraphNoInputButWithOutputToJson(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says no input but with output, yet the scenario supplies input tensor ids
    # and a graph id and omits the output ids and debug values.
    # Scenario: an execution with a graph id and input tensor ids but no output ids or debug values is
    # converted to JSON. Asserts the round trip and that missing fields are null and the rest match.
    execution_digest = debug_events_reader.ExecutionDigest(
        1234, 5678, "FooOp", output_tensor_device_ids=[1357])
    execution = debug_events_reader.Execution(
        execution_digest,
        "localhost",
        ("a1", "b2"),
        debug_event_pb2.TensorDebugMode.FULL_HEALTH,
        graph_id="abcd",
        input_tensor_ids=[13, 37],
        output_tensor_ids=None,
        debug_tensor_values=None)
    json = execution.to_json()
    self.jsonRoundTripCheck(json)
    self.assertEqual(json["wall_time"], 1234)
    self.assertEqual(json["op_type"], "FooOp")
    self.assertEqual(json["output_tensor_device_ids"], (1357,))
    self.assertEqual(json["host_name"], "localhost")
    self.assertEqual(json["stack_frame_ids"], ("a1", "b2"))
    self.assertEqual(json["tensor_debug_mode"],
                     debug_event_pb2.TensorDebugMode.FULL_HEALTH)
    self.assertEqual(json["graph_id"], "abcd")
    self.assertEqual(json["input_tensor_ids"], (13, 37))
    self.assertIsNone(json["output_tensor_ids"])
    self.assertIsNone(json["debug_tensor_values"])

  @parameterized.named_parameters(
      ("EmptyList", []),
      ("None", None),
  )
  def testExecutionWithNoOutputTensorsReturnsZeroForNumOutputs(
      self, output_tensor_ids):
    # [FUNCTION DOC]
    # Parameterized scenario (empty list or nothing/null for the output tensor ids): builds an
    # execution with those ids and asserts its number of outputs is zero.
    # Parameter output_tensor_ids: empty list or null.
    execution = debug_events_reader.Execution(
        debug_events_reader.ExecutionDigest(1234, 5678, "FooOp"),
        "localhost", ("a1", "b2"),
        debug_event_pb2.TensorDebugMode.FULL_HEALTH,
        graph_id="abcd",
        input_tensor_ids=[13, 37],
        output_tensor_ids=output_tensor_ids,
        debug_tensor_values=None)
    self.assertEqual(execution.num_outputs, 0)

  def testDebuggedDeviceToJons(self):
    # [FUNCTION DOC]
    # Scenario: a debugged device with a name and id is converted to JSON. Asserts the mapping holds
    # the device name and device id.
    debugged_device = debug_events_reader.DebuggedDevice("/TPU:3", 4)
    self.assertEqual(debugged_device.to_json(), {
        "device_name": "/TPU:3",
        "device_id": 4,
    })

  def testDebuggedGraphToJonsWitouthNameInnerOuterGraphIds(self):
    # [FUNCTION DOC]
    # Scenario: a debugged graph with no name and no outer graph id is converted to JSON. Asserts
    # the name and outer id are null and the inner graph id list is empty.
    debugged_graph = debug_events_reader.DebuggedGraph(
        None,
        "b1c2",
        outer_graph_id=None,
    )
    self.assertEqual(
        debugged_graph.to_json(), {
            "name": None,
            "graph_id": "b1c2",
            "outer_graph_id": None,
            "inner_graph_ids": [],
        })

  def testDebuggedGraphToJonsWithNameAndInnerOuterGraphIds(self):
    # [FUNCTION DOC]
    # Scenario: a debugged graph with a name, an outer graph id and two added inner graph ids is
    # converted to JSON. Asserts all fields including the ordered inner graph id list.
    debugged_graph = debug_events_reader.DebuggedGraph(
        "loss_function",
        "b1c2",
        outer_graph_id="a0b1",
    )
    debugged_graph.add_inner_graph_id("c2d3")
    debugged_graph.add_inner_graph_id("c2d3e4")
    self.assertEqual(
        debugged_graph.to_json(), {
            "name": "loss_function",
            "graph_id": "b1c2",
            "outer_graph_id": "a0b1",
            "inner_graph_ids": ["c2d3", "c2d3e4"],
        })

  @parameterized.named_parameters(
      ("EmptyList", []),
      ("None", None),
  )
  def testGraphOpDigestWithNoOutpusReturnsNumOutputsZero(
      self, output_tensor_ids):
    # [FUNCTION DOC]
    # Parameterized scenario (empty list or nothing/null for the output tensor ids): builds an op
    # creation digest with those ids and asserts its number of outputs is zero.
    # Parameter output_tensor_ids: empty list or null.
    op_creation_digest = debug_events_reader.GraphOpCreationDigest(
        1234,
        5678,
        "deadbeef",
        "FooOp",
        "Model_1/Foo_2",
        output_tensor_ids,
        "machine.cluster", ("a1", "a2"),
        input_names=None,
        device_name=None)
    self.assertEqual(op_creation_digest.num_outputs, 0)

  def testGraphOpCreationDigestNoInputNoDeviceNameToJson(self):
    # [FUNCTION DOC]
    # Scenario: an op creation digest with no input names and no device name is converted to JSON.
    # Asserts the round trip, all provided fields (output ids as a tuple) and null input names/device.
    op_creation_digest = debug_events_reader.GraphOpCreationDigest(
        1234,
        5678,
        "deadbeef",
        "FooOp",
        "Model_1/Foo_2", [135],
        "machine.cluster", ("a1", "a2"),
        input_names=None,
        device_name=None)
    json = op_creation_digest.to_json()
    self.jsonRoundTripCheck(json)
    self.assertEqual(json["wall_time"], 1234)
    self.assertEqual(json["graph_id"], "deadbeef")
    self.assertEqual(json["op_type"], "FooOp")
    self.assertEqual(json["op_name"], "Model_1/Foo_2")
    self.assertEqual(json["output_tensor_ids"], (135,))
    self.assertEqual(json["host_name"], "machine.cluster")
    self.assertEqual(json["stack_frame_ids"], ("a1", "a2"))
    self.assertIsNone(json["input_names"])
    self.assertIsNone(json["device_name"])

  def testGraphOpCreationDigestWithInputsAndDeviceNameToJson(self):
    # [FUNCTION DOC]
    # Scenario: an op creation digest with input names and a device name is converted to JSON.
    # Asserts the round trip and every field, with input names given as a tuple.
    op_creation_digest = debug_events_reader.GraphOpCreationDigest(
        1234,
        5678,
        "deadbeef",
        "FooOp",
        "Model_1/Foo_2", [135],
        "machine.cluster", ("a1", "a2"),
        input_names=["Bar_1", "Qux_2"],
        device_name="/device:GPU:0")
    json = op_creation_digest.to_json()
    self.jsonRoundTripCheck(json)
    self.assertEqual(json["wall_time"], 1234)
    self.assertEqual(json["graph_id"], "deadbeef")
    self.assertEqual(json["op_type"], "FooOp")
    self.assertEqual(json["op_name"], "Model_1/Foo_2")
    self.assertEqual(json["output_tensor_ids"], (135,))
    self.assertEqual(json["host_name"], "machine.cluster")
    self.assertEqual(json["stack_frame_ids"], ("a1", "a2"))
    self.assertEqual(json["input_names"], ("Bar_1", "Qux_2"))
    self.assertEqual(json["device_name"], "/device:GPU:0")

  def testGraphExecutionTraceDigestToJson(self):
    # [FUNCTION DOC]
    # Scenario: a graph execution trace digest is converted to JSON. Asserts wall time, op type, op
    # name, output slot and graph id.
    trace_digest = debug_events_reader.GraphExecutionTraceDigest(
        1234, 5678, "FooOp", "Model_1/Foo_2", 1, "deadbeef")
    json = trace_digest.to_json()
    self.assertEqual(json["wall_time"], 1234)
    self.assertEqual(json["op_type"], "FooOp")
    self.assertEqual(json["op_name"], "Model_1/Foo_2")
    self.assertEqual(json["output_slot"], 1)
    self.assertEqual(json["graph_id"], "deadbeef")

  def testGraphExecutionTraceWithTensorDebugValueAndDeviceNameToJson(self):
    # [FUNCTION DOC]
    # Scenario: a graph execution trace with a debug tensor value, device name and graph id chain is
    # converted to JSON. Asserts all digest fields, the graph ids tuple, debug mode, value tuple and
    # device name.
    trace_digest = debug_events_reader.GraphExecutionTraceDigest(
        1234, 5678, "FooOp", "Model_1/Foo_2", 1, "deadbeef")
    trace = debug_events_reader.GraphExecutionTrace(
        trace_digest, ["g1", "g2", "deadbeef"],
        debug_event_pb2.TensorDebugMode.CURT_HEALTH,
        debug_tensor_value=[3, 1], device_name="/device:GPU:0")
    json = trace.to_json()
    self.assertEqual(json["wall_time"], 1234)
    self.assertEqual(json["op_type"], "FooOp")
    self.assertEqual(json["op_name"], "Model_1/Foo_2")
    self.assertEqual(json["output_slot"], 1)
    self.assertEqual(json["graph_id"], "deadbeef")
    self.assertEqual(json["graph_ids"], ("g1", "g2", "deadbeef"))
    self.assertEqual(json["tensor_debug_mode"],
                     debug_event_pb2.TensorDebugMode.CURT_HEALTH)
    self.assertEqual(json["debug_tensor_value"], (3, 1))
    self.assertEqual(json["device_name"], "/device:GPU:0")

  def testGraphExecutionTraceNoTensorDebugValueNoDeviceNameToJson(self):
    # [FUNCTION DOC]
    # Scenario: a graph execution trace in no-tensor mode without a debug value or device name is
    # converted to JSON. Asserts the digest fields and graph ids, and that value and device are null.
    trace_digest = debug_events_reader.GraphExecutionTraceDigest(
        1234, 5678, "FooOp", "Model_1/Foo_2", 1, "deadbeef")
    trace = debug_events_reader.GraphExecutionTrace(
        trace_digest, ["g1", "g2", "deadbeef"],
        debug_event_pb2.TensorDebugMode.NO_TENSOR,
        debug_tensor_value=None, device_name=None)
    json = trace.to_json()
    self.assertEqual(json["wall_time"], 1234)
    self.assertEqual(json["op_type"], "FooOp")
    self.assertEqual(json["op_name"], "Model_1/Foo_2")
    self.assertEqual(json["output_slot"], 1)
    self.assertEqual(json["graph_id"], "deadbeef")
    self.assertEqual(json["graph_ids"], ("g1", "g2", "deadbeef"))
    self.assertEqual(json["tensor_debug_mode"],
                     debug_event_pb2.TensorDebugMode.NO_TENSOR)
    self.assertIsNone(json["debug_tensor_value"])
    self.assertIsNone(json["device_name"])


if __name__ == "__main__":
  ops.enable_eager_execution()
  googletest.main()
