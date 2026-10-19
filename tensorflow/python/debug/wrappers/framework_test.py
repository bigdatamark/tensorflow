# [MODULE DOC]
# Unit tests for the debug wrapper session framework. Defines two concrete test wrapper classes
# (one that records callback activity in a shared observer mapping and dumps to a directory, one
# that returns deliberately invalid actions or URLs) and tests covering session init, run
# behavior, error capture, context-manager and default-session use, closing, thread name
# filtering, and that the wrapper exposes all public methods of regular and monitored sessions.
# No import-time side effects.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Framework of debug-wrapped sessions."""
import os
import tempfile
import threading

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.wrappers import framework
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
# Import resource_variable_ops for the variables-to-tensor implicit conversion.
from tensorflow.python.ops import resource_variable_ops  # pylint: disable=unused-import
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.training import monitored_session
from tensorflow.python.util import tf_inspect


class TestDebugWrapperSession(framework.BaseDebugWrapperSession):
  # [CLASS DOC]
  # HELPER: concrete wrapper session used by the tests. Every callback increments counters and
  # records request details (session, fetches, feeds, performed action, error) in a shared
  # observer mapping; run-start always requests a debug run dumping to a file URL under the given
  # dump root.
  """A concrete implementation of BaseDebugWrapperSession for test."""

  def __init__(self, sess, dump_root, observer, thread_name_filter=None):
    # [FUNCTION DOC]
    # Stores the dump root path and observer mapping before calling the base constructor (the
    # init callback needs them). Inputs: sess, session to wrap; dump_root, directory path;
    # observer, mutable mapping with counters and fields; thread_name_filter, optional pattern.
    # Supply dump root.
    self._dump_root = dump_root

    # Supply observer.
    self._obs = observer

    # Invoke superclass constructor.
    framework.BaseDebugWrapperSession.__init__(
        self, sess, thread_name_filter=thread_name_filter)

  def on_session_init(self, request):
    # [FUNCTION DOC]
    # Increments the init counter, records the request's session in the observer, and returns a
    # response with the proceed action.
    """Override abstract on-session-init callback method."""

    self._obs["sess_init_count"] += 1
    self._obs["request_sess"] = request.session

    return framework.OnSessionInitResponse(
        framework.OnSessionInitAction.PROCEED)

  def on_run_start(self, request):
    # [FUNCTION DOC]
    # Increments the run-start counter, records the request's fetches and feed mapping, and
    # returns a response with the debug-run action and a single file URL for the dump root.
    """Override abstract on-run-start callback method."""

    self._obs["on_run_start_count"] += 1
    self._obs["run_fetches"] = request.fetches
    self._obs["run_feed_dict"] = request.feed_dict

    return framework.OnRunStartResponse(
        framework.OnRunStartAction.DEBUG_RUN,
        ["file://" + self._dump_root])

  def on_run_end(self, request):
    # [FUNCTION DOC]
    # Increments the run-end counter, records the performed action and any captured op error in
    # the observer, and returns an empty run-end response.
    """Override abstract on-run-end callback method."""

    self._obs["on_run_end_count"] += 1
    self._obs["performed_action"] = request.performed_action
    self._obs["tf_error"] = request.tf_error

    return framework.OnRunEndResponse()


class TestDebugWrapperSessionBadAction(framework.BaseDebugWrapperSession):
  # [CLASS DOC]
  # HELPER: concrete wrapper session used to test error handling. It can return an invalid action
  # from the init callback, an invalid action and/or invalid debug URLs from the run-start
  # callback; otherwise it behaves with valid values (proceed, debug run, empty URL list).
  """A concrete implementation of BaseDebugWrapperSession for test.

  This class intentionally puts a bad action value in OnSessionInitResponse
  and/or in OnRunStartAction to test the handling of such invalid cases.
  """

  def __init__(
      self,
      sess,
      bad_init_action=None,
      bad_run_start_action=None,
      bad_debug_urls=None):
    # [FUNCTION DOC]
    # Stores the bad values, then calls the base constructor. Inputs: sess, session to wrap;
    # bad_init_action, optional string to return from init; bad_run_start_action, optional string
    # action for run start; bad_debug_urls, optional value used as the debug URLs. Returns
    # nothing.
    """Constructor.

    Args:
      sess: The TensorFlow Session object to be wrapped.
      bad_init_action: (str) bad action value to be returned during the
        on-session-init callback.
      bad_run_start_action: (str) bad action value to be returned during the
        the on-run-start callback.
      bad_debug_urls: Bad URL values to be returned during the on-run-start
        callback.
    """

    self._bad_init_action = bad_init_action
    self._bad_run_start_action = bad_run_start_action
    self._bad_debug_urls = bad_debug_urls

    # Invoke superclass constructor.
    framework.BaseDebugWrapperSession.__init__(self, sess)

  def on_session_init(self, request):
    # [FUNCTION DOC]
    # Returns an init response with the configured bad action if one was given, otherwise with
    # the proceed action.
    if self._bad_init_action:
      return framework.OnSessionInitResponse(self._bad_init_action)
    else:
      return framework.OnSessionInitResponse(
          framework.OnSessionInitAction.PROCEED)

  def on_run_start(self, request):
    # [FUNCTION DOC]
    # Returns a run-start response using the configured bad debug URLs (or an empty list) and
    # the configured bad action if given, otherwise the debug-run action.
    debug_urls = self._bad_debug_urls or []

    if self._bad_run_start_action:
      return framework.OnRunStartResponse(
          self._bad_run_start_action, debug_urls)
    else:
      return framework.OnRunStartResponse(
          framework.OnRunStartAction.DEBUG_RUN, debug_urls)

  def on_run_end(self, request):
    # [FUNCTION DOC]
    # Returns an empty run-end response.
    return framework.OnRunEndResponse()


@test_util.run_v1_only("Sessions are not available in TF 2.x")
class DebugWrapperSessionTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Graph-mode-only tests of the base wrapper session behavior through the test wrapper
  # classes. Fixture: observer mapping, temporary dump root, a session with graph pruning
  # disabled, constants, a placeholder, two variables, and matrix-product and sum ops.

  def _no_rewrite_session_config(self):
    # [FUNCTION DOC]
    # HELPER: returns a session configuration whose graph rewriter has model pruning disabled, so
    # all nodes are kept and dumped in tests.
    rewriter_config = rewriter_config_pb2.RewriterConfig(
        disable_model_pruning=True)
    graph_options = config_pb2.GraphOptions(rewrite_options=rewriter_config)
    return config_pb2.ConfigProto(graph_options=graph_options)

  def setUp(self):
    # [FUNCTION DOC]
    # Creates the observer mapping with zeroed counters and empty fields, a temporary dump root, a
    # session with the no-pruning configuration, initial values, constants, a double placeholder,
    # two variables, a matrix product, a product with the placeholder and a sum; then initializes
    # the variables. Stores everything on the test instance.
    self._observer = {
        "sess_init_count": 0,
        "request_sess": None,
        "on_run_start_count": 0,
        "run_fetches": None,
        "run_feed_dict": None,
        "on_run_end_count": 0,
        "performed_action": None,
        "tf_error": None,
    }

    self._dump_root = tempfile.mkdtemp()

    self._sess = session.Session(config=self._no_rewrite_session_config())

    self._a_init_val = np.array([[5.0, 3.0], [-1.0, 0.0]])
    self._b_init_val = np.array([[2.0], [-1.0]])
    self._c_val = np.array([[-4.0], [6.0]])

    self._a_init = constant_op.constant(
        self._a_init_val, shape=[2, 2], name="a_init")
    self._b_init = constant_op.constant(
        self._b_init_val, shape=[2, 1], name="b_init")

    self._ph = array_ops.placeholder(dtype=dtypes.float64, name="ph")

    self._a = variables.Variable(self._a_init, name="a1")
    self._b = variables.Variable(self._b_init, name="b")
    self._c = constant_op.constant(self._c_val, shape=[2, 1], name="c")

    # Matrix product of a and b.
    self._p = math_ops.matmul(self._a, self._b, name="p1")

    # Matrix product of a and ph.
    self._q = math_ops.matmul(self._a, self._ph, name="q")

    # Sum of two vectors.
    self._s = math_ops.add(self._p, self._c, name="s")

    # Initialize the variables.
    self._sess.run(self._a.initializer)
    self._sess.run(self._b.initializer)

  def tearDown(self):
    # [FUNCTION DOC]
    # Deletes the temporary dump directory if present and resets the default graph.
    # Tear down temporary dump directory.
    if os.path.isdir(self._dump_root):
      file_io.delete_recursively(self._dump_root)

    ops.reset_default_graph()

  def testSessionInit(self):
    # [FUNCTION DOC]
    # Wraps the session and asserts the init callback ran once with the right session, that the
    # wrapper is a session interface and exposes the same connection string, graph and graph
    # definition, and that partial-run setup raises not-implemented.
    self.assertEqual(0, self._observer["sess_init_count"])

    wrapper_sess = TestDebugWrapperSession(self._sess, self._dump_root,
                                           self._observer)

    # Assert that on-session-init callback is invoked.
    self.assertEqual(1, self._observer["sess_init_count"])

    # Assert that the request to the on-session-init callback carries the
    # correct session object.
    self.assertEqual(self._sess, self._observer["request_sess"])

    # Verify that the wrapper session implements the session.SessionInterface.
    self.assertTrue(isinstance(wrapper_sess, session.SessionInterface))
    self.assertEqual(self._sess.sess_str, wrapper_sess.sess_str)
    self.assertEqual(self._sess.graph, wrapper_sess.graph)
    self.assertEqual(self._sess.graph_def, wrapper_sess.graph_def)

    # Check that the partial_run_setup and partial_run are not implemented for
    # the debug wrapper session.
    with self.assertRaises(NotImplementedError):
      wrapper_sess.partial_run_setup(self._p)

  def testInteractiveSessionInit(self):
    # [FUNCTION DOC]
    # Checks that an interactive session (a session subclass) can be wrapped without error.
    """The wrapper should work also on other subclasses of session.Session."""

    TestDebugWrapperSession(
        session.InteractiveSession(), self._dump_root, self._observer)

  def testSessionRun(self):
    # [FUNCTION DOC]
    # Runs the sum op through the wrapper. Asserts the result is correct, run-start ran once with
    # the right fetches and no feeds, seven tensors were dumped to the dump root, run-end ran once
    # with the debug-run action performed, and no error was captured.
    wrapper = TestDebugWrapperSession(
        self._sess, self._dump_root, self._observer)

    # Check initial state of the observer.
    self.assertEqual(0, self._observer["on_run_start_count"])
    self.assertEqual(0, self._observer["on_run_end_count"])

    s = wrapper.run(self._s)

    # Assert the run return value is correct.
    self.assertAllClose(np.array([[3.0], [4.0]]), s)

    # Assert the on-run-start method is invoked.
    self.assertEqual(1, self._observer["on_run_start_count"])

    # Assert the on-run-start request reflects the correct fetch.
    self.assertEqual(self._s, self._observer["run_fetches"])

    # Assert the on-run-start request reflects the correct feed_dict.
    self.assertIsNone(self._observer["run_feed_dict"])

    # Assert the file debug URL has led to dump on the filesystem.
    dump = debug_data.DebugDumpDir(self._dump_root)
    self.assertEqual(7, len(dump.dumped_tensor_data))

    # Assert the on-run-end method is invoked.
    self.assertEqual(1, self._observer["on_run_end_count"])

    # Assert the performed action field in the on-run-end callback request is
    # correct.
    self.assertEqual(
        framework.OnRunStartAction.DEBUG_RUN,
        self._observer["performed_action"])

    # No TensorFlow runtime error should have happened.
    self.assertIsNone(self._observer["tf_error"])

  def testSessionInitInvalidSessionType(self):
    # [FUNCTION DOC]
    # Tries to wrap a wrapper session (not a real session); asserts a type error mentioning
    # expected and actual types.
    """Attempt to wrap a non-Session-type object should cause an exception."""

    wrapper = TestDebugWrapperSessionBadAction(self._sess)
    with self.assertRaisesRegex(TypeError, "Expected type .*; got type .*"):
      TestDebugWrapperSessionBadAction(wrapper)

  def testSessionInitBadActionValue(self):
    # [FUNCTION DOC]
    # Constructs a wrapper whose init callback returns an invalid action; asserts a value error
    # naming the invalid session-init action value.
    with self.assertRaisesRegex(
        ValueError, "Invalid OnSessionInitAction value: nonsense_action"):
      TestDebugWrapperSessionBadAction(
          self._sess, bad_init_action="nonsense_action")

  def testRunStartBadActionValue(self):
    # [FUNCTION DOC]
    # Builds a wrapper whose run-start callback returns an invalid action; asserts running raises
    # a value error naming the invalid run-start action value.
    wrapper = TestDebugWrapperSessionBadAction(
        self._sess, bad_run_start_action="nonsense_action")

    with self.assertRaisesRegex(
        ValueError, "Invalid OnRunStartAction value: nonsense_action"):
      wrapper.run(self._s)

  def testRunStartBadURLs(self):
    # [FUNCTION DOC]
    # Builds a wrapper whose run-start callback returns a string instead of a list of debug URLs;
    # asserts running raises a type error.
    # debug_urls ought to be a list of str, not a str. So an exception should
    # be raised during a run() call.
    wrapper = TestDebugWrapperSessionBadAction(
        self._sess, bad_debug_urls="file://foo")

    with self.assertRaisesRegex(TypeError, "Expected type .*; got type .*"):
      wrapper.run(self._s)

  def testErrorDuringRun(self):
    # [FUNCTION DOC]
    # Runs the placeholder product with a correctly shaped feed and asserts the result and no
    # captured error; then runs with a mismatched shape and asserts the run-end callback saw an
    # invalid-argument error instead of the error being raised.

    wrapper = TestDebugWrapperSession(self._sess, self._dump_root,
                                      self._observer)

    # No matrix size mismatch.
    self.assertAllClose(
        np.array([[11.0], [-1.0]]),
        wrapper.run(self._q, feed_dict={self._ph: np.array([[1.0], [2.0]])}))
    self.assertEqual(1, self._observer["on_run_end_count"])
    self.assertIsNone(self._observer["tf_error"])

    # Now there should be a matrix size mismatch error.
    wrapper.run(self._q, feed_dict={self._ph: np.array([[1.0], [2.0], [3.0]])})
    self.assertEqual(2, self._observer["on_run_end_count"])
    self.assertTrue(
        isinstance(self._observer["tf_error"], errors.InvalidArgumentError))

  def testUsingWrappedSessionShouldWorkAsContextManager(self):
    # [FUNCTION DOC]
    # Uses the wrapper as a context manager and evaluates a tensor and runs an op inside it;
    # asserts the callbacks were invoked with the expected fetches and counts.
    wrapper = TestDebugWrapperSession(self._sess, self._dump_root,
                                      self._observer)

    with wrapper as sess:
      self.assertAllClose([[3.0], [4.0]], self._s)
      self.assertEqual(1, self._observer["on_run_start_count"])
      self.assertEqual([self._s], self._observer["run_fetches"])
      self.assertEqual(1, self._observer["on_run_end_count"])

      self.assertAllClose(
          [[11.0], [-1.0]],
          sess.run(self._q, feed_dict={self._ph: np.array([[1.0], [2.0]])}))
      self.assertEqual(2, self._observer["on_run_start_count"])
      self.assertEqual(self._q, self._observer["run_fetches"])
      self.assertEqual(2, self._observer["on_run_end_count"])

  def testUsingWrappedSessionShouldSupportEvalWithAsDefault(self):
    # [FUNCTION DOC]
    # Makes the wrapper the default session, creates a constant, evaluates it, and asserts the
    # value is correct and the wrapper observed the constant as the run fetch.
    wrapper = TestDebugWrapperSession(self._sess, self._dump_root,
                                      self._observer)

    with wrapper.as_default():
      foo = constant_op.constant(42, name="foo")
      self.assertEqual(42, self.evaluate(foo))
      self.assertEqual([foo], self._observer["run_fetches"])

  def testWrapperShouldSupportSessionClose(self):
    # [FUNCTION DOC]
    # Creates a wrapper and closes it; expects no error.
    wrapper = TestDebugWrapperSession(self._sess, self._dump_root,
                                      self._observer)
    wrapper.close()

  def testWrapperThreadNameFilterMainThread(self):
    # [FUNCTION DOC]
    # Wraps with a filter matching the main thread. Runs one constant on a child thread and
    # another on the main thread; asserts both results are right but only the main thread's
    # run was dumped (one dump, from the first constant's node).
    wrapper = TestDebugWrapperSession(
        self._sess, self._dump_root, self._observer,
        thread_name_filter="MainThread")

    child_run_output = []
    def child_thread_job():
      # [FUNCTION DOC]
      # HELPER: child thread body; runs the second constant through the wrapper and appends the
      # result to the shared output list.
      child_run_output.append(wrapper.run(self._b_init))

    thread = threading.Thread(name="ChildThread", target=child_thread_job)
    thread.start()
    self.assertAllClose(self._a_init_val, wrapper.run(self._a_init))
    thread.join()
    self.assertAllClose([self._b_init_val], child_run_output)

    dump = debug_data.DebugDumpDir(self._dump_root)
    self.assertEqual(1, dump.size)
    self.assertEqual("a_init", dump.dumped_tensor_data[0].node_name)

  def testWrapperThreadNameFilterChildThread(self):
    # [FUNCTION DOC]
    # Same scenario with a filter matching thread names starting with the child prefix; asserts
    # both runs return correct values but only the child thread's run was dumped (the second
    # constant's node).
    wrapper = TestDebugWrapperSession(
        self._sess, self._dump_root, self._observer,
        thread_name_filter=r"Child.*")

    child_run_output = []
    def child_thread_job():
      # [FUNCTION DOC]
      # HELPER: child thread body; runs the second constant through the wrapper and appends the
      # result to the shared output list.
      child_run_output.append(wrapper.run(self._b_init))

    thread = threading.Thread(name="ChildThread", target=child_thread_job)
    thread.start()
    self.assertAllClose(self._a_init_val, wrapper.run(self._a_init))
    thread.join()
    self.assertAllClose([self._b_init_val], child_run_output)

    dump = debug_data.DebugDumpDir(self._dump_root)
    self.assertEqual(1, dump.size)
    self.assertEqual("b_init", dump.dumped_tensor_data[0].node_name)

  def testWrapperThreadNameFilterBothThreads(self):
    # [FUNCTION DOC]
    # Same scenario with no thread filter; asserts both results are right and both runs were
    # dumped (two dumps, from both constants' nodes). Loads the dump without validation.
    wrapper = TestDebugWrapperSession(
        self._sess, self._dump_root, self._observer,
        thread_name_filter=None)

    child_run_output = []
    def child_thread_job():
      # [FUNCTION DOC]
      # HELPER: child thread body; runs the second constant through the wrapper and appends the
      # result to the shared output list.
      child_run_output.append(wrapper.run(self._b_init))

    thread = threading.Thread(name="ChildThread", target=child_thread_job)
    thread.start()
    self.assertAllClose(self._a_init_val, wrapper.run(self._a_init))
    thread.join()
    self.assertAllClose([self._b_init_val], child_run_output)

    dump = debug_data.DebugDumpDir(self._dump_root, validate=False)
    self.assertEqual(2, dump.size)
    self.assertItemsEqual(
        ["a_init", "b_init"],
        [datum.node_name for datum in dump.dumped_tensor_data])


def _is_public_method_name(method_name):
  # [FUNCTION DOC]
  # HELPER: for the method parity tests. Returns true for names that are public (no leading
  # underscore) or are double-underscore special methods; false for other private names.
  return (method_name.startswith("__") and method_name.endswith("__")
          or not method_name.startswith("_"))


class SessionWrapperPublicMethodParityTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests that the base wrapper session class offers every public method of the regular and the
  # monitored session classes, so it can stand in for them.

  def testWrapperHasAllPublicMethodsOfSession(self):
    # [FUNCTION DOC]
    # Lists public methods of the regular session class and of the base wrapper class, and asserts
    # none of the session's methods is missing from the wrapper.
    session_public_methods = [
        method_tuple[0] for method_tuple in
        tf_inspect.getmembers(session.Session, predicate=tf_inspect.ismethod)
        if _is_public_method_name(method_tuple[0])]
    wrapper_public_methods = [
        method_tuple[0] for method_tuple in
        tf_inspect.getmembers(
            framework.BaseDebugWrapperSession, predicate=tf_inspect.ismethod)
        if _is_public_method_name(method_tuple[0])]
    missing_public_methods = [
        method for method in session_public_methods
        if method not in wrapper_public_methods]
    self.assertFalse(missing_public_methods)

  def testWrapperHasAllPublicMethodsOfMonitoredSession(self):
    # [FUNCTION DOC]
    # Same check against the monitored session class: asserts none of its public methods is
    # missing from the base wrapper class.
    session_public_methods = [
        method_tuple[0] for method_tuple in
        tf_inspect.getmembers(monitored_session.MonitoredSession,
                              predicate=tf_inspect.ismethod)
        if _is_public_method_name(method_tuple[0])]
    wrapper_public_methods = [
        method_tuple[0] for method_tuple in
        tf_inspect.getmembers(
            framework.BaseDebugWrapperSession, predicate=tf_inspect.ismethod)
        if _is_public_method_name(method_tuple[0])]
    missing_public_methods = [
        method for method in session_public_methods
        if method not in wrapper_public_methods]
    self.assertFalse(missing_public_methods)


if __name__ == "__main__":
  googletest.main()
