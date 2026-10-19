# [MODULE DOC]
# Unit tests for the remote source module: verifies that graph and eager execution tracebacks,
# and the user source files they reference, are delivered to one or more gRPC debug test servers,
# and that gRPC channels and servers use unlimited message sizes. Not a helper module.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Unit tests for source_remote."""

import os
import traceback

import grpc

from tensorflow.core.debug import debug_service_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import grpc_debug_test_server
from tensorflow.python.debug.lib import source_remote
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
# Import resource_variable_ops for the variables-to-tensor implicit conversion.
from tensorflow.python.ops import resource_variable_ops  # pylint: disable=unused-import
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.util import tf_inspect


def line_number_above():
  # [FUNCTION DOC]
  # HELPER: small test utility for recording source line numbers.
  # Returns the line number of the caller's current line minus one, i.e. the line directly above
  # the call site. Takes no arguments.
  return tf_inspect.stack()[1][2] - 1


class SendTracebacksTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the traceback-sending functions of the remote source module, run against real
  # in-process gRPC debug servers. Shared fixtures: two servers started once per class (stored
  # as class attributes together with their addresses) and the path of this file; the servers'
  # recorded data and the default graph are reset after every test.

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Class-level setup: calls the parent setup, then starts two gRPC debug test servers on
    # separate threads (the first one in polling mode) and stores on the class their ports, URLs,
    # dump directories, threads and server objects, plus localhost address strings for both
    # and the normalized absolute path of this test file.
    test_util.TensorFlowTestCase.setUpClass()
    (cls._server_port, cls._debug_server_url, cls._server_dump_dir,
     cls._server_thread,
     cls._server) = grpc_debug_test_server.start_server_on_separate_thread(
         poll_server=True)
    cls._server_address = "localhost:%d" % cls._server_port
    (cls._server_port_2, cls._debug_server_url_2, cls._server_dump_dir_2,
     cls._server_thread_2,
     cls._server_2) = grpc_debug_test_server.start_server_on_separate_thread()
    cls._server_address_2 = "localhost:%d" % cls._server_port_2
    cls._curr_file_path = os.path.normpath(os.path.abspath(__file__))

  @classmethod
  def tearDownClass(cls):
    # [FUNCTION DOC]
    # Class-level cleanup: stops both debug servers, waiting for each to finish, joins both
    # server threads, then calls the parent class tear-down. Returns nothing.
    # Stop the test server and join the thread.
    cls._server.stop_server().wait()
    cls._server_thread.join()
    cls._server_2.stop_server().wait()
    cls._server_thread_2.join()
    test_util.TensorFlowTestCase.tearDownClass()

  def tearDown(self):
    # [FUNCTION DOC]
    # Per-test cleanup: resets the default graph and clears all data recorded by both debug
    # servers, then calls the parent tear-down. Returns nothing.
    ops.reset_default_graph()
    self._server.clear_data()
    self._server_2.clear_data()
    super(SendTracebacksTest, self).tearDown()

  def _findFirstTraceInsideTensorFlowPyLibrary(self, op):
    # [FUNCTION DOC]
    # HELPER: lets the tests pick a library-internal frame of an op's creation traceback.
    # Takes an operation and returns the first traceback entry whose file is guessed to belong
    # to the TensorFlow Python library, or nothing/null if there is none.
    """Find the first trace of an op that belongs to the TF Python library."""
    for trace in op.traceback:
      if source_utils.guess_is_tensorflow_py_library(trace.filename):
        return trace

  def testSendGraphTracebacksToSingleDebugServer(self):
    # [FUNCTION DOC]
    # Test: sending graph tracebacks to one debug server.
    # Setup: in a session, creates two variables and an add op, recording the line number of
    # each creation, and captures the current stack. Action: sends the graph tracebacks with a
    # dummy run key. Asserts: op tracebacks for the three ops contain this file, the recorded
    # line and this test's name; the origin stack contains the send line; the uploaded source
    # line for the first variable has the expected text; a TensorFlow library file was not
    # uploaded (querying it raises a value error); and the recorded call type is graph
    # execution, the call key is the dummy key, and the graph version matches.
    this_func_name = "testSendGraphTracebacksToSingleDebugServer"
    with session.Session() as sess:
      a = variables.Variable(21.0, name="a")
      a_lineno = line_number_above()
      b = variables.Variable(2.0, name="b")
      b_lineno = line_number_above()
      math_ops.add(a, b, name="x")
      x_lineno = line_number_above()

      send_stack = traceback.extract_stack()
      send_lineno = line_number_above()
      source_remote.send_graph_tracebacks(
          self._server_address, "dummy_run_key", send_stack, sess.graph)

      tb = self._server.query_op_traceback("a")
      self.assertIn((self._curr_file_path, a_lineno, this_func_name), tb)
      tb = self._server.query_op_traceback("b")
      self.assertIn((self._curr_file_path, b_lineno, this_func_name), tb)
      tb = self._server.query_op_traceback("x")
      self.assertIn((self._curr_file_path, x_lineno, this_func_name), tb)

      self.assertIn(
          (self._curr_file_path, send_lineno, this_func_name),
          self._server.query_origin_stack()[-1])

      self.assertEqual(
          "      a = variables.Variable(21.0, name=\"a\")",
          self._server.query_source_file_line(__file__, a_lineno))
      # Files in the TensorFlow code base shouldn not have been sent.
      tf_trace = self._findFirstTraceInsideTensorFlowPyLibrary(a.op)
      tf_trace_file_path = tf_trace.filename
      with self.assertRaises(ValueError):
        self._server.query_source_file_line(tf_trace_file_path, 0)
      self.assertEqual([debug_service_pb2.CallTraceback.GRAPH_EXECUTION],
                       self._server.query_call_types())
      self.assertEqual(["dummy_run_key"], self._server.query_call_keys())
      self.assertEqual(
          [sess.graph.version], self._server.query_graph_versions())

  def testSendGraphTracebacksToTwoDebugServers(self):
    # [FUNCTION DOC]
    # Test: sending graph tracebacks to two debug servers at once.
    # Setup: in a session, creates two variables and an add op, recording the line numbers of
    # each creation, and captures the current stack. Action: sends the graph tracebacks to both
    # server addresses with a dummy run key while wrapping the gRPC channel factory with a
    # recording mock; asserts the channel was opened with unlimited message sizes. For each of
    # the two servers asserts: the op tracebacks contain the expected file, line and test name;
    # the origin stack contains the send line; the source line of the add op was uploaded with
    # the right text; a file from the TensorFlow library was not uploaded (querying it raises a
    # value error); and the call type, call key and graph version match.
    this_func_name = "testSendGraphTracebacksToTwoDebugServers"
    with session.Session() as sess:
      a = variables.Variable(21.0, name="two/a")
      a_lineno = line_number_above()
      b = variables.Variable(2.0, name="two/b")
      b_lineno = line_number_above()
      x = math_ops.add(a, b, name="two/x")
      x_lineno = line_number_above()

      send_traceback = traceback.extract_stack()
      send_lineno = line_number_above()

      with test.mock.patch.object(
          grpc, "insecure_channel",
          wraps=grpc.insecure_channel) as mock_grpc_channel:
        source_remote.send_graph_tracebacks(
            [self._server_address, self._server_address_2],
            "dummy_run_key", send_traceback, sess.graph)
        mock_grpc_channel.assert_called_with(
            test.mock.ANY,
            options=[("grpc.max_receive_message_length", -1),
                     ("grpc.max_send_message_length", -1)])

      servers = [self._server, self._server_2]
      for server in servers:
        tb = server.query_op_traceback("two/a")
        self.assertIn((self._curr_file_path, a_lineno, this_func_name), tb)
        tb = server.query_op_traceback("two/b")
        self.assertIn((self._curr_file_path, b_lineno, this_func_name), tb)
        tb = server.query_op_traceback("two/x")
        self.assertIn((self._curr_file_path, x_lineno, this_func_name), tb)

        self.assertIn(
            (self._curr_file_path, send_lineno, this_func_name),
            server.query_origin_stack()[-1])

        self.assertEqual(
            "      x = math_ops.add(a, b, name=\"two/x\")",
            server.query_source_file_line(__file__, x_lineno))
        tf_trace = self._findFirstTraceInsideTensorFlowPyLibrary(a.op)
        tf_trace_file_path = tf_trace.filename
        with self.assertRaises(ValueError):
          server.query_source_file_line(tf_trace_file_path, 0)
        self.assertEqual([debug_service_pb2.CallTraceback.GRAPH_EXECUTION],
                         server.query_call_types())
        self.assertEqual(["dummy_run_key"], server.query_call_keys())
        self.assertEqual([sess.graph.version], server.query_graph_versions())

  def testSendEagerTracebacksToSingleDebugServer(self):
    # [FUNCTION DOC]
    # Test: sending an eager-execution traceback to a single debug server.
    # Action: captures the current stack, records the line number of the capture, and sends it
    # with the eager sender. Asserts the server recorded exactly one call of type eager
    # execution, and that the last recorded origin stack contains this file, the capture line
    # and this test's name.
    this_func_name = "testSendEagerTracebacksToSingleDebugServer"
    send_traceback = traceback.extract_stack()
    send_lineno = line_number_above()
    source_remote.send_eager_tracebacks(self._server_address, send_traceback)

    self.assertEqual([debug_service_pb2.CallTraceback.EAGER_EXECUTION],
                     self._server.query_call_types())
    self.assertIn((self._curr_file_path, send_lineno, this_func_name),
                  self._server.query_origin_stack()[-1])

  def testGRPCServerMessageSizeLimit(self):
    # [FUNCTION DOC]
    # Test: the gRPC debug test server is created with unlimited send and receive message sizes.
    # Setup: wraps the gRPC server factory with a recording mock, then starts a polling debug
    # server on a separate thread. Asserts the factory was last called with options setting both
    # maximum receive and send message lengths to unlimited. Finally stops the server and joins
    # its thread.
    """Assert gRPC debug server is started with unlimited message size."""
    with test.mock.patch.object(
        grpc, "server", wraps=grpc.server) as mock_grpc_server:
      (_, _, _, server_thread,
       server) = grpc_debug_test_server.start_server_on_separate_thread(
           poll_server=True)
      mock_grpc_server.assert_called_with(
          test.mock.ANY,
          options=[("grpc.max_receive_message_length", -1),
                   ("grpc.max_send_message_length", -1)])
    server.stop_server().wait()
    server_thread.join()


if __name__ == "__main__":
  googletest.main()
