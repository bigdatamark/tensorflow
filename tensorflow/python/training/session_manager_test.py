# [MODULE DOC]
# Tests for the session manager, which creates, prepares, recovers and waits for training
# sessions. Two test classes are defined: one using the "report uninitialized variables" style
# of readiness check (plus many local-initialization scenarios) and an "obsolete" one that
# repeats a subset using the older "assert variables initialized" readiness op. Tests build
# graphs and variables, write and restore checkpoints in temporary directories, and sometimes
# start an in-process local server. Both classes disable resource variables at class setup, which
# changes global state for the process. Run as the main program through the test runner.
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
"""Tests for SessionManager."""

import os
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.client import session as session_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variables_toggle
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import server_lib
from tensorflow.python.training import session_manager


class SessionManagerTest(test.TestCase):
  # [CLASS DOC]
  # Main test case for the session manager. Class-level fixture disables resource variables so
  # classic reference variables are used. Covers preparing a session (init op, feed dict, init
  # function, local init op, checkpoint waiting and failures), recovering from checkpoints,
  # waiting for a session, and the readiness checks for global and local initialization.

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Class-level fixture run once before the tests. Calls the parent setup and then disables
    # resource variables globally so tests get reference variables. Returns nothing. The change
    # is process-wide and never undone here.
    super(SessionManagerTest, cls).setUpClass()
    resource_variables_toggle.disable_resource_variables()

  def testPrepareSessionSucceeds(self):
    # [FUNCTION DOC]
    # Scenario: a three element variable and a manager whose readiness check reports
    # uninitialized variables. Preparing a session with the global initializer as init op must
    # succeed; asserts the variable then reads one, two, three.
    with ops.Graph().as_default():
      v = variable_v1.VariableV1([1.0, 2.0, 3.0], name="v")
      sm = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables())
      sess = sm.prepare_session(
          "", init_op=variables.global_variables_initializer())
      self.assertAllClose([1.0, 2.0, 3.0], sess.run(v))

  def testPrepareSessionSucceedsWithInitFeedDict(self):
    # [FUNCTION DOC]
    # Scenario: a variable initialized from a placeholder. Preparing a session supplies the
    # placeholder value through the init feed mapping. Asserts the variable reads one, two,
    # three afterwards.
    with ops.Graph().as_default():
      p = array_ops.placeholder(dtypes.float32, shape=(3,))
      v = variable_v1.VariableV1(p, name="v")
      sm = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables())
      sess = sm.prepare_session(
          "",
          init_op=variables.global_variables_initializer(),
          init_feed_dict={p: [1.0, 2.0, 3.0]})
      self.assertAllClose([1.0, 2.0, 3.0], sess.run(v))

  def testPrepareSessionSucceedsWithInitFn(self):
    # [FUNCTION DOC]
    # Scenario: preparing a session with an init function (instead of an init op) that runs the
    # variable's initializer. Asserts the variable reads the value 125 afterwards.
    with ops.Graph().as_default():
      v = variable_v1.VariableV1([125], name="v")
      sm = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables())
      sess = sm.prepare_session(
          "", init_fn=lambda sess: sess.run(v.initializer))
      self.assertAllClose([125], sess.run(v))

  def testPrepareSessionSucceedsWithLocalInitFeedDict(self):
    # [FUNCTION DOC]
    # Scenario: a local variable initialized from a placeholder; the manager is built with the
    # variable's initializer as local init op and a local init feed mapping providing the
    # placeholder. Preparing a session without any init op must succeed; asserts the variable
    # reads one, two, three.
    with ops.Graph().as_default():
      p = array_ops.placeholder(dtypes.float32, shape=(3,))
      v = variable_v1.VariableV1(
          p, name="v", collections=[ops.GraphKeys.LOCAL_VARIABLES])
      sm = session_manager.SessionManager(
          local_init_op=v.initializer,
          local_init_feed_dict={p: [1.0, 2.0, 3.0]},
          ready_op=variables.report_uninitialized_variables())
      sess = sm.prepare_session("")
      self.assertAllClose([1.0, 2.0, 3.0], sess.run(v))

  def testPrepareSessionFails(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says it fails, but the test ends by asserting success once the
    # checkpoint is restored. Also, in the second graph a new manager is constructed and its
    # result is discarded, so the prepare calls there use the manager from the first graph; the
    # first session variable in the failing block is assigned but never used.
    # Scenario: save a checkpoint of a variable in one graph, then in a new graph move the
    # checkpoint directory away and make an empty one. Preparing a session with no init op,
    # waiting for a checkpoint up to two seconds, must raise a runtime error saying no init op,
    # init function or local init op was given. After moving the checkpoint back, the same call
    # must succeed and the variable named v must be initialized.
    checkpoint_dir = os.path.join(self.get_temp_dir(), "prepare_session")
    checkpoint_dir2 = os.path.join(self.get_temp_dir(), "prepare_session2")
    try:
      gfile.DeleteRecursively(checkpoint_dir)
      gfile.DeleteRecursively(checkpoint_dir2)
    except errors.OpError:
      pass  # Ignore
    gfile.MakeDirs(checkpoint_dir)

    with ops.Graph().as_default():
      v = variable_v1.VariableV1([1.0, 2.0, 3.0], name="v")
      sm = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables())
      saver = saver_lib.Saver({"v": v})
      sess = sm.prepare_session(
          "",
          init_op=variables.global_variables_initializer(),
          saver=saver,
          checkpoint_dir=checkpoint_dir)
      self.assertAllClose([1.0, 2.0, 3.0], sess.run(v))
      checkpoint_filename = os.path.join(checkpoint_dir,
                                         "prepare_session_checkpoint")
      saver.save(sess, checkpoint_filename)
    # Create a new Graph and SessionManager and recover.
    with ops.Graph().as_default():
      # Renames the checkpoint directory.
      os.rename(checkpoint_dir, checkpoint_dir2)
      gfile.MakeDirs(checkpoint_dir)
      v = variable_v1.VariableV1([6.0, 7.0, 8.0], name="v")
      with self.cached_session():
        self.assertEqual(False, variable_v1.is_variable_initialized(v).eval())
      session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables())
      saver = saver_lib.Saver({"v": v})
      # This should fail as there's no checkpoint within 2 seconds.
      with self.assertRaisesRegex(
          RuntimeError, "no init_op or init_fn or local_init_op was given"):
        sess = sm.prepare_session(
            "",
            init_op=None,
            saver=saver,
            checkpoint_dir=checkpoint_dir,
            wait_for_checkpoint=True,
            max_wait_secs=2)
      # Rename the checkpoint directory back.
      gfile.DeleteRecursively(checkpoint_dir)
      os.rename(checkpoint_dir2, checkpoint_dir)
      # This should succeed as there's checkpoint.
      sess = sm.prepare_session(
          "",
          init_op=None,
          saver=saver,
          checkpoint_dir=checkpoint_dir,
          wait_for_checkpoint=True,
          max_wait_secs=2)
      self.assertEqual(
          True,
          variable_v1.is_variable_initialized(
              sess.graph.get_tensor_by_name("v:0")).eval(session=sess))

  def _test_recovered_variable(self,
                               checkpoint_dir=None,
                               checkpoint_filename_with_path=None):
    # [FUNCTION DOC]
    # HELPER: shared by the recover-session test to check recovery from a saved checkpoint.
    # Inputs: checkpoint_dir (optional directory holding checkpoints) and
    # checkpoint_filename_with_path (optional explicit checkpoint file); both default to
    # nothing. In a fresh graph it creates a variable with value two, confirms it is not
    # initialized, recovers a session with a saver, and asserts recovery reports initialized,
    # the variable is initialized and now holds one. Errors from recovery (for example when both
    # inputs are given) propagate to the caller. Returns nothing.
    # Create a new Graph and SessionManager and recover from a checkpoint.
    with ops.Graph().as_default():
      v = variable_v1.VariableV1(2, name="v")
      with session_lib.Session():
        self.assertEqual(False, variable_v1.is_variable_initialized(v).eval())
      sm2 = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables())
      saver = saver_lib.Saver({"v": v})
      sess, initialized = sm2.recover_session(
          "",
          saver=saver,
          checkpoint_dir=checkpoint_dir,
          checkpoint_filename_with_path=checkpoint_filename_with_path)
      self.assertTrue(initialized)
      self.assertEqual(
          True,
          variable_v1.is_variable_initialized(
              sess.graph.get_tensor_by_name("v:0")).eval(session=sess))
      self.assertEqual(1, sess.run(v))

  def testRecoverSession(self):
    # [FUNCTION DOC]
    # Scenario: with no checkpoint present, recovering reports not initialized; the variable is
    # then initialized by hand and saved. Recovery in a fresh graph is checked both by
    # directory and by explicit latest checkpoint file path (via the helper). Supplying both the
    # directory and the file path must raise a value error.
    # Create a checkpoint.
    checkpoint_dir = os.path.join(self.get_temp_dir(), "recover_session")
    try:
      gfile.DeleteRecursively(checkpoint_dir)
    except errors.OpError:
      pass  # Ignore
    gfile.MakeDirs(checkpoint_dir)

    with ops.Graph().as_default():
      v = variable_v1.VariableV1(1, name="v")
      sm = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables())
      saver = saver_lib.Saver({"v": v})
      sess, initialized = sm.recover_session(
          "", saver=saver, checkpoint_dir=checkpoint_dir)
      self.assertFalse(initialized)
      sess.run(v.initializer)
      self.assertEqual(1, sess.run(v))
      saver.save(sess, os.path.join(checkpoint_dir,
                                    "recover_session_checkpoint"))
    self._test_recovered_variable(checkpoint_dir=checkpoint_dir)
    self._test_recovered_variable(
        checkpoint_filename_with_path=checkpoint_management.latest_checkpoint(
            checkpoint_dir))
    # Cannot set both checkpoint_dir and checkpoint_filename_with_path.
    with self.assertRaises(ValueError):
      self._test_recovered_variable(
          checkpoint_dir=checkpoint_dir,
          checkpoint_filename_with_path=checkpoint_management.latest_checkpoint(
              checkpoint_dir))

  def testWaitForSessionReturnsNoneAfterTimeout(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says the call returns nothing after a timeout, but the test asserts
    # that a deadline-exceeded error is raised.
    # Scenario: an uninitialized variable and a manager with a one second recovery wait. Waiting
    # for a session with a three second maximum must raise a deadline-exceeded error.
    with ops.Graph().as_default():
      variable_v1.VariableV1(1, name="v")
      sm = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables(),
          recovery_wait_secs=1)

      # Set max_wait_secs to allow us to try a few times.
      with self.assertRaises(errors.DeadlineExceededError):
        sm.wait_for_session(master="", max_wait_secs=3)

  def testInitWithNoneLocalInitOpError(self):
    # [FUNCTION DOC]
    # Scenario: constructing a manager with a ready-for-local-init op but no local init op.
    # Asserts a value error whose message says that passing a ready-for-local-init op requires
    # also passing a local init op.
    # Creating a SessionManager with a None local_init_op but
    # non-None ready_for_local_init_op raises ValueError
    with self.assertRaisesRegex(
        ValueError, "If you pass a ready_for_local_init_op "
        "you must also pass a local_init_op "):
      session_manager.SessionManager(
          ready_for_local_init_op=variables.report_uninitialized_variables(
              variables.global_variables()),
          local_init_op=None)

  def testRecoverSessionWithReadyForLocalInitOp(self):
    # [FUNCTION DOC]
    # Scenario: a checkpoint of variable v is saved. In a new graph a local variable w depends
    # on v; the manager has a ready-for-local-init check over global variables and w's
    # initializer as local init op. Recovering must report initialized, restore v from the
    # checkpoint, and run the local init so w is initialized too. Asserts both are initialized
    # and both read one.
    # Create a checkpoint.
    checkpoint_dir = os.path.join(self.get_temp_dir(),
                                  "recover_session_ready_for_local_init")
    try:
      gfile.DeleteRecursively(checkpoint_dir)
    except errors.OpError:
      pass  # Ignore
    gfile.MakeDirs(checkpoint_dir)

    with ops.Graph().as_default():
      v = variable_v1.VariableV1(1, name="v")
      sm = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables())
      saver = saver_lib.Saver({"v": v})
      sess, initialized = sm.recover_session(
          "", saver=saver, checkpoint_dir=checkpoint_dir)
      self.assertFalse(initialized)
      sess.run(v.initializer)
      self.assertEqual(1, sess.run(v))
      saver.save(sess, os.path.join(checkpoint_dir,
                                    "recover_session_checkpoint"))
    # Create a new Graph and SessionManager and recover.
    with ops.Graph().as_default():
      v = variable_v1.VariableV1(2, name="v")
      w = variable_v1.VariableV1(
          v,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="w")
      with self.cached_session():
        self.assertEqual(False, variable_v1.is_variable_initialized(v).eval())
        self.assertEqual(False, variable_v1.is_variable_initialized(w).eval())
      sm2 = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables(),
          ready_for_local_init_op=variables.report_uninitialized_variables(
              variables.global_variables()),
          local_init_op=w.initializer)
      saver = saver_lib.Saver({"v": v})
      sess, initialized = sm2.recover_session(
          "", saver=saver, checkpoint_dir=checkpoint_dir)
      self.assertTrue(initialized)
      self.assertEqual(
          True,
          variable_v1.is_variable_initialized(
              sess.graph.get_tensor_by_name("v:0")).eval(session=sess))
      self.assertEqual(
          True,
          variable_v1.is_variable_initialized(
              sess.graph.get_tensor_by_name("w:0")).eval(session=sess))
      self.assertEqual(1, sess.run(v))
      self.assertEqual(1, sess.run(w))

  def testRecoverSessionWithReadyForLocalInitOpFailsToReadyLocal(self):
    # [FUNCTION DOC]
    # Scenario: same as the ready-for-local-init recovery test, except the ready-for-local-init
    # check covers all variables including the not-yet-initialized local one, so it never
    # passes. Recovery must restore v but not run the local init op, and report not
    # initialized. Asserts v is initialized, w is not, and v reads one.
    # We use ready_for_local_init_op=report_uninitialized_variables(),
    # which causes recover_session to not run local_init_op, and to return
    # initialized=False

    # Create a checkpoint.
    checkpoint_dir = os.path.join(
        self.get_temp_dir(),
        "recover_session_ready_for_local_init_fails_to_ready_local")
    try:
      gfile.DeleteRecursively(checkpoint_dir)
    except errors.OpError:
      pass  # Ignore
    gfile.MakeDirs(checkpoint_dir)

    with ops.Graph().as_default():
      v = variable_v1.VariableV1(1, name="v")
      sm = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables())
      saver = saver_lib.Saver({"v": v})
      sess, initialized = sm.recover_session(
          "", saver=saver, checkpoint_dir=checkpoint_dir)
      self.assertFalse(initialized)
      sess.run(v.initializer)
      self.assertEqual(1, sess.run(v))
      saver.save(sess, os.path.join(checkpoint_dir,
                                    "recover_session_checkpoint"))
    # Create a new Graph and SessionManager and recover.
    with ops.Graph().as_default():
      v = variable_v1.VariableV1(2, name="v")
      w = variable_v1.VariableV1(
          v,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="w")
      with self.cached_session():
        self.assertEqual(False, variable_v1.is_variable_initialized(v).eval())
        self.assertEqual(False, variable_v1.is_variable_initialized(w).eval())
      sm2 = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables(),
          ready_for_local_init_op=variables.report_uninitialized_variables(),
          local_init_op=w.initializer)
      saver = saver_lib.Saver({"v": v})
      sess, initialized = sm2.recover_session(
          "", saver=saver, checkpoint_dir=checkpoint_dir)
      self.assertFalse(initialized)
      self.assertEqual(
          True,
          variable_v1.is_variable_initialized(
              sess.graph.get_tensor_by_name("v:0")).eval(session=sess))
      self.assertEqual(
          False,
          variable_v1.is_variable_initialized(
              sess.graph.get_tensor_by_name("w:0")).eval(session=sess))
      self.assertEqual(1, sess.run(v))

  def testRecoverSessionNoChkptStillRunsLocalInitOp(self):
    # [FUNCTION DOC]
    # Scenario (backward compatibility): recovery with no saver and no checkpoint directory,
    # with a local init op and no ready-for-local-init check. Asserts recovery reports not
    # initialized yet still runs the local init op exactly so the local variable is
    # initialized and reads one.
    # This test checks for backwards compatibility.
    # In particular, we continue to ensure that recover_session will execute
    # local_init_op exactly once, regardless of whether the session was
    # successfully recovered.
    with ops.Graph().as_default():
      w = variable_v1.VariableV1(
          1,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="w")
      with self.cached_session():
        self.assertEqual(False, variable_v1.is_variable_initialized(w).eval())
      sm2 = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables(),
          ready_for_local_init_op=None,
          local_init_op=w.initializer)
      # Try to recover session from None
      sess, initialized = sm2.recover_session(
          "", saver=None, checkpoint_dir=None)
      # Succeeds because recover_session still run local_init_op
      self.assertFalse(initialized)
      self.assertEqual(
          True,
          variable_v1.is_variable_initialized(
              sess.graph.get_tensor_by_name("w:0")).eval(session=sess))
      self.assertEqual(1, sess.run(w))

  def testRecoverSessionFailsStillRunsLocalInitOp(self):
    # [FUNCTION DOC]
    # Scenario: an empty checkpoint directory exists but holds no checkpoint; recovery is
    # attempted without waiting. Asserts recovery reports not initialized, the global variable
    # stays uninitialized, but the local init op still ran so the local variable is initialized
    # and reads one.
    # Create a checkpoint.
    checkpoint_dir = os.path.join(
        self.get_temp_dir(),
        "recover_session_ready_for_local_init_fails_stil_run")
    try:
      gfile.DeleteRecursively(checkpoint_dir)
    except errors.OpError:
      pass  # Ignore
    gfile.MakeDirs(checkpoint_dir)

    # Create a new Graph and SessionManager and recover.
    with ops.Graph().as_default():
      v = variable_v1.VariableV1(2, name="v")
      w = variable_v1.VariableV1(
          1,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="w")
      with self.cached_session():
        self.assertEqual(False, variable_v1.is_variable_initialized(v).eval())
        self.assertEqual(False, variable_v1.is_variable_initialized(w).eval())
      sm2 = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables(),
          ready_for_local_init_op=None,
          local_init_op=w.initializer)
      saver = saver_lib.Saver({"v": v})
      sess, initialized = sm2.recover_session(
          "",
          saver=saver,
          checkpoint_dir=checkpoint_dir,
          wait_for_checkpoint=False)
      self.assertFalse(initialized)
      self.assertEqual(
          False,
          variable_v1.is_variable_initialized(
              sess.graph.get_tensor_by_name("v:0")).eval(session=sess))
      self.assertEqual(
          True,
          variable_v1.is_variable_initialized(
              sess.graph.get_tensor_by_name("w:0")).eval(session=sess))
      self.assertEqual(1, sess.run(w))

  def testWaitForSessionLocalInit(self):
    # [FUNCTION DOC]
    # Scenario: starts an in-process local server (side effect: opens a server). Global
    # variable v is initialized through a separate session on the server; the manager has a
    # ready-for-local-init check and a local init op for dependent local variable w. Waiting for
    # a session (three seconds maximum) must run the local init. Asserts both variables are
    # initialized and both read one.
    server = server_lib.Server.create_local_server()
    with ops.Graph().as_default() as graph:
      v = variable_v1.VariableV1(1, name="v")
      w = variable_v1.VariableV1(
          v,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="w")
      sm = session_manager.SessionManager(
          graph=graph,
          ready_op=variables.report_uninitialized_variables(),
          ready_for_local_init_op=variables.report_uninitialized_variables(
              variables.global_variables()),
          local_init_op=w.initializer)

      # Initialize v but not w
      s = session_lib.Session(server.target, graph=graph)
      s.run(v.initializer)

      sess = sm.wait_for_session(server.target, max_wait_secs=3)
      self.assertEqual(
          True,
          variable_v1.is_variable_initialized(
              sess.graph.get_tensor_by_name("v:0")).eval(session=sess))
      self.assertEqual(
          True,
          variable_v1.is_variable_initialized(
              sess.graph.get_tensor_by_name("w:0")).eval(session=sess))
      self.assertEqual(1, sess.run(v))
      self.assertEqual(1, sess.run(w))

  def testWaitForSessionWithReadyForLocalInitOpFailsToReadyLocal(self):
    # [FUNCTION DOC]
    # Scenario: the ready-for-local-init check covers the uninitialized local variable and so
    # never passes, meaning the local init op is never run. Waiting for a session with a three
    # second maximum must raise a deadline-exceeded error.
    with ops.Graph().as_default() as graph:
      v = variable_v1.VariableV1(1, name="v")
      w = variable_v1.VariableV1(
          v,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="w")
      sm = session_manager.SessionManager(
          graph=graph,
          ready_op=variables.report_uninitialized_variables(),
          ready_for_local_init_op=variables.report_uninitialized_variables(),
          local_init_op=w.initializer)

      with self.assertRaises(errors_impl.DeadlineExceededError):
        # Time-out because w fails to be initialized,
        # because of overly restrictive ready_for_local_init_op
        sm.wait_for_session("", max_wait_secs=3)

  @test_util.run_v1_only("Requires TF V1 variable behavior.")
  def testWaitForSessionInsufficientReadyForLocalInitCheck(self):
    # [FUNCTION DOC]
    # Runs only under the version 1 variable behavior (decorator). Scenario: no
    # ready-for-local-init check is given, so the local init op runs before the global variable
    # is initialized and the readiness check fails. Waiting for a session must raise a
    # deadline-exceeded error whose message starts with "Session was not ready after waiting".
    with ops.Graph().as_default() as graph:
      v = variable_v1.VariableV1(1, name="v")
      w = variable_v1.VariableV1(
          v,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="w")
      sm = session_manager.SessionManager(
          graph=graph,
          ready_op=variables.report_uninitialized_variables(),
          ready_for_local_init_op=None,
          local_init_op=w.initializer)
    with self.assertRaisesRegex(errors_impl.DeadlineExceededError,
                                "Session was not ready after waiting.*"):
      sm.wait_for_session("", max_wait_secs=3)

  def testPrepareSessionWithReadyForLocalInitOp(self):
    # [FUNCTION DOC]
    # Scenario: global variable v and local variables w (copy of v) and x (three times v), with
    # a list of two local init ops and a ready-for-local-init check over global variables.
    # Preparing a session with only v's initializer must also run the local init ops once v is
    # ready. Asserts all three are initialized and read one, one and three.
    with ops.Graph().as_default():
      v = variable_v1.VariableV1(1, name="v")
      w = variable_v1.VariableV1(
          v,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="w")
      x = variable_v1.VariableV1(
          3 * v,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="x")
      with self.cached_session():
        self.assertEqual(False, variable_v1.is_variable_initialized(v).eval())
        self.assertEqual(False, variable_v1.is_variable_initialized(w).eval())
        self.assertEqual(False, variable_v1.is_variable_initialized(x).eval())
      sm2 = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables(),
          ready_for_local_init_op=variables.report_uninitialized_variables(
              variables.global_variables()),
          local_init_op=[w.initializer, x.initializer])
      sess = sm2.prepare_session("", init_op=v.initializer)
      self.assertEqual(
          True,
          variable_v1.is_variable_initialized(
              sess.graph.get_tensor_by_name("v:0")).eval(session=sess))
      self.assertEqual(
          True,
          variable_v1.is_variable_initialized(
              sess.graph.get_tensor_by_name("w:0")).eval(session=sess))
      self.assertEqual(
          True,
          variable_v1.is_variable_initialized(
              sess.graph.get_tensor_by_name("x:0")).eval(session=sess))
      self.assertEqual(1, sess.run(v))
      self.assertEqual(1, sess.run(w))
      self.assertEqual(3, sess.run(x))

  @test_util.run_v1_only("Requires TF V1 variable behavior.")
  def testPrepareSessionWithPartialInitOp(self):
    # [FUNCTION DOC]
    # Runs only under the version 1 variable behavior. Scenario: a manager with only a list of
    # local init ops (no ready op) covering local variables derived from two global variables,
    # and no init op passed to prepare. Asserts the global variables remain uninitialized while
    # the local ones are initialized with values one and three.
    with ops.Graph().as_default():
      v = variable_v1.VariableV1(1, name="v")
      w = variable_v1.VariableV1(
          v,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="w")
      x = variable_v1.VariableV1(
          3 * v,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="x")
      # TODO(b/70206927): Use ResourceVariables once they are handled properly.
      v_res = variable_v1.VariableV1(1, name="v_res")
      w_res = variable_v1.VariableV1(
          v_res,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="w_res")
      x_res = variable_v1.VariableV1(
          3 * v_res,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="x_res")

      with self.cached_session():
        self.assertEqual(False, variable_v1.is_variable_initialized(v).eval())
        self.assertEqual(False, variable_v1.is_variable_initialized(w).eval())
        self.assertEqual(False, variable_v1.is_variable_initialized(x).eval())
        self.assertEqual(False,
                         variable_v1.is_variable_initialized(v_res).eval())
        self.assertEqual(False,
                         variable_v1.is_variable_initialized(w_res).eval())
        self.assertEqual(False,
                         variable_v1.is_variable_initialized(x_res).eval())
      sm2 = session_manager.SessionManager(local_init_op=[
          w.initializer, x.initializer, w_res.initializer, x_res.initializer
      ])
      sess = sm2.prepare_session("", init_op=None)
      self.assertEqual(
          False,
          variable_v1.is_variable_initialized(
              sess.graph.get_tensor_by_name("v:0")).eval(session=sess))
      self.assertEqual(
          True,
          variable_v1.is_variable_initialized(
              sess.graph.get_tensor_by_name("w:0")).eval(session=sess))
      self.assertEqual(
          True,
          variable_v1.is_variable_initialized(
              sess.graph.get_tensor_by_name("x:0")).eval(session=sess))
      self.assertEqual(1, sess.run(w))
      self.assertEqual(3, sess.run(x))
      self.assertEqual(
          False,
          variable_v1.is_variable_initialized(
              sess.graph.get_tensor_by_name("v_res:0")).eval(session=sess))
      self.assertEqual(
          True,
          variable_v1.is_variable_initialized(
              sess.graph.get_tensor_by_name("w_res:0")).eval(session=sess))
      self.assertEqual(
          True,
          variable_v1.is_variable_initialized(
              sess.graph.get_tensor_by_name("x_res:0")).eval(session=sess))
      self.assertEqual(1, sess.run(w_res))
      self.assertEqual(3, sess.run(x_res))

  def testPrepareSessionWithCyclicInitializer(self):
    # [FUNCTION DOC]
    # Regression test: a variable whose initial value comes from a while loop (cyclic graph
    # dependencies) once caused infinite recursion when building the initializer expression.
    # Preparing a session with the variable's initializer must succeed; asserts the variable
    # reads one and is initialized.
    # Regression test. Previously Variable._build_initializer_expr would enter
    # into an infinite recursion when the variable's initial_value involved
    # cyclic dependencies.
    with ops.Graph().as_default():
      i = while_loop.while_loop(lambda i: i < 1, lambda i: i + 1, [0])
      v = variable_v1.VariableV1(array_ops.identity(i), name="v")
      with self.cached_session():
        self.assertEqual(False, variable_v1.is_variable_initialized(v).eval())
      sm = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables())
      sess = sm.prepare_session("", init_op=v.initializer)
      self.assertEqual(1, sess.run(v))
      self.assertEqual(
          True,
          variable_v1.is_variable_initialized(
              sess.graph.get_tensor_by_name("v:0")).eval(session=sess))

  def testPrepareSessionDidNotInitLocalVariable(self):
    # [FUNCTION DOC]
    # Scenario: a local variable exists but the manager has no local init op, and only the
    # global variable's initializer is passed. Preparing must raise a runtime error whose
    # message starts with "Init operations did not make model ready".
    with ops.Graph().as_default():
      v = variable_v1.VariableV1(1, name="v")
      w = variable_v1.VariableV1(
          v,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="w")
      with self.cached_session():
        self.assertEqual(False, variable_v1.is_variable_initialized(v).eval())
        self.assertEqual(False, variable_v1.is_variable_initialized(w).eval())
      sm2 = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables())
      with self.assertRaisesRegex(RuntimeError,
                                  "Init operations did not make model ready.*"):
        sm2.prepare_session("", init_op=v.initializer)

  def testPrepareSessionDidNotInitLocalVariableList(self):
    # [FUNCTION DOC]
    # Same scenario as the single-op variant but the init op is passed as a list containing the
    # global variable's initializer. Preparing must raise a runtime error saying the init
    # operations did not make the model ready.
    with ops.Graph().as_default():
      v = variable_v1.VariableV1(1, name="v")
      w = variable_v1.VariableV1(
          v,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="w")
      with self.cached_session():
        self.assertEqual(False, variable_v1.is_variable_initialized(v).eval())
        self.assertEqual(False, variable_v1.is_variable_initialized(w).eval())
      sm2 = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables())
      with self.assertRaisesRegex(RuntimeError,
                                  "Init operations did not make model ready"):
        sm2.prepare_session("", init_op=[v.initializer])

  def testPrepareSessionWithReadyNotReadyForLocal(self):
    # [FUNCTION DOC]
    # Scenario: a manager with a ready-for-local-init check over global variables and a local
    # init op, but prepare is called with no init op so the global variable never becomes
    # initialized. Must raise a runtime error saying the init operations did not make the model
    # ready for local init.
    with ops.Graph().as_default():
      v = variable_v1.VariableV1(1, name="v")
      w = variable_v1.VariableV1(
          v,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="w")
      with self.cached_session():
        self.assertEqual(False, variable_v1.is_variable_initialized(v).eval())
        self.assertEqual(False, variable_v1.is_variable_initialized(w).eval())
      sm2 = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables(),
          ready_for_local_init_op=variables.report_uninitialized_variables(
              variables.global_variables()),
          local_init_op=w.initializer)
      with self.assertRaisesRegex(
          RuntimeError,
          "Init operations did not make model ready for local_init"):
        sm2.prepare_session("", init_op=None)

  @test_util.run_v1_only("Requires TF V1 variable behavior.")
  def testPrepareSessionWithInsufficientReadyForLocalInitCheck(self):
    # [FUNCTION DOC]
    # Runs only under the version 1 variable behavior. Scenario: no ready-for-local-init check,
    # a local init op that depends on an uninitialized global variable, and no init op. Preparing
    # must raise a runtime error starting with "Init operations did not make model ready".
    with ops.Graph().as_default():
      v = variable_v1.VariableV1(1, name="v")
      w = variable_v1.VariableV1(
          v,
          trainable=False,
          collections=[ops.GraphKeys.LOCAL_VARIABLES],
          name="w")
      with self.cached_session():
        self.assertEqual(False, variable_v1.is_variable_initialized(v).eval())
        self.assertEqual(False, variable_v1.is_variable_initialized(w).eval())
      sm2 = session_manager.SessionManager(
          ready_op=variables.report_uninitialized_variables(),
          ready_for_local_init_op=None,
          local_init_op=w.initializer)
    with self.assertRaisesRegex(RuntimeError,
                                "Init operations did not make model ready.*"):
      sm2.prepare_session("", init_op=None)


class ObsoleteSessionManagerTest(test.TestCase):
  # [CLASS DOC]
  # Test case repeating a subset of the main session manager tests (prepare, recover, wait)
  # using the older "assert variables initialized" readiness op instead of the report
  # uninitialized variables style. Class-level fixture disables resource variables.

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Class-level fixture run once before the tests. Calls the parent setup and then disables
    # resource variables globally (process-wide, never undone here). Returns nothing.
    super(ObsoleteSessionManagerTest, cls).setUpClass()
    resource_variables_toggle.disable_resource_variables()

  def testPrepareSessionSucceeds(self):
    # [FUNCTION DOC]
    # Scenario: same as the main class version but the manager uses the assert-variables-
    # initialized readiness op. Preparing with the global initializer must succeed; asserts the
    # variable reads one, two, three.
    with ops.Graph().as_default():
      v = variable_v1.VariableV1([1.0, 2.0, 3.0], name="v")
      sm = session_manager.SessionManager(
          ready_op=variables.assert_variables_initialized())
      sess = sm.prepare_session(
          "", init_op=variables.global_variables_initializer())
      self.assertAllClose([1.0, 2.0, 3.0], sess.run(v))

  def testPrepareSessionSucceedsWithInitFeedDict(self):
    # [FUNCTION DOC]
    # Scenario: a placeholder-initialized variable prepared with an init feed mapping, using the
    # assert-variables-initialized readiness op. Asserts the variable reads one, two, three.
    with ops.Graph().as_default():
      p = array_ops.placeholder(dtypes.float32, shape=(3,))
      v = variable_v1.VariableV1(p, name="v")
      sm = session_manager.SessionManager(
          ready_op=variables.assert_variables_initialized())
      sess = sm.prepare_session(
          "",
          init_op=variables.global_variables_initializer(),
          init_feed_dict={p: [1.0, 2.0, 3.0]})
      self.assertAllClose([1.0, 2.0, 3.0], sess.run(v))

  def testPrepareSessionSucceedsWithInitFn(self):
    # [FUNCTION DOC]
    # Scenario: preparing with an init function that runs the variable's initializer, using the
    # assert-variables-initialized readiness op. Asserts the variable reads 125.
    with ops.Graph().as_default():
      v = variable_v1.VariableV1([125], name="v")
      sm = session_manager.SessionManager(
          ready_op=variables.assert_variables_initialized())
      sess = sm.prepare_session(
          "", init_fn=lambda sess: sess.run(v.initializer))
      self.assertAllClose([125], sess.run(v))

  def testPrepareSessionFails(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says it fails, but the test ends by asserting success once the
    # checkpoint is restored. In the second graph a new manager is constructed and discarded, so
    # the prepare calls reuse the manager from the first graph.
    # Scenario: same as the main class version but with the assert-variables-initialized
    # readiness op. With the checkpoint moved away, preparing with checkpoint waiting for two
    # seconds must raise a runtime error saying no init op, init function or local init op was
    # given; after restoring the checkpoint it must succeed and the variable is initialized.
    checkpoint_dir = os.path.join(self.get_temp_dir(), "prepare_session")
    checkpoint_dir2 = os.path.join(self.get_temp_dir(), "prepare_session2")
    try:
      gfile.DeleteRecursively(checkpoint_dir)
      gfile.DeleteRecursively(checkpoint_dir2)
    except errors.OpError:
      pass  # Ignore
    gfile.MakeDirs(checkpoint_dir)

    with ops.Graph().as_default():
      v = variable_v1.VariableV1([1.0, 2.0, 3.0], name="v")
      sm = session_manager.SessionManager(
          ready_op=variables.assert_variables_initialized())
      saver = saver_lib.Saver({"v": v})
      sess = sm.prepare_session(
          "",
          init_op=variables.global_variables_initializer(),
          saver=saver,
          checkpoint_dir=checkpoint_dir)
      self.assertAllClose([1.0, 2.0, 3.0], sess.run(v))
      checkpoint_filename = os.path.join(checkpoint_dir,
                                         "prepare_session_checkpoint")
      saver.save(sess, checkpoint_filename)
    # Create a new Graph and SessionManager and recover.
    with ops.Graph().as_default():
      # Renames the checkpoint directory.
      os.rename(checkpoint_dir, checkpoint_dir2)
      gfile.MakeDirs(checkpoint_dir)
      v = variable_v1.VariableV1([6.0, 7.0, 8.0], name="v")
      with self.cached_session():
        self.assertEqual(False, variable_v1.is_variable_initialized(v).eval())
      session_manager.SessionManager(
          ready_op=variables.assert_variables_initialized())
      saver = saver_lib.Saver({"v": v})
      # This should fail as there's no checkpoint within 2 seconds.
      with self.assertRaisesRegex(
          RuntimeError, "no init_op or init_fn or local_init_op was given"):
        sess = sm.prepare_session(
            "",
            init_op=None,
            saver=saver,
            checkpoint_dir=checkpoint_dir,
            wait_for_checkpoint=True,
            max_wait_secs=2)
      # Rename the checkpoint directory back.
      gfile.DeleteRecursively(checkpoint_dir)
      os.rename(checkpoint_dir2, checkpoint_dir)
      # This should succeed as there's checkpoint.
      sess = sm.prepare_session(
          "",
          init_op=None,
          saver=saver,
          checkpoint_dir=checkpoint_dir,
          wait_for_checkpoint=True,
          max_wait_secs=2)
      self.assertEqual(
          True,
          variable_v1.is_variable_initialized(
              sess.graph.get_tensor_by_name("v:0")).eval(session=sess))

  def testRecoverSession(self):
    # [FUNCTION DOC]
    # Scenario: with the assert-variables-initialized readiness op, recovery with no checkpoint
    # reports not initialized; the variable is then initialized and saved. In a fresh graph
    # recovery restores it: asserts recovery reports initialized, the variable is initialized
    # and reads one.
    # Create a checkpoint.
    checkpoint_dir = os.path.join(self.get_temp_dir(), "recover_session")
    try:
      gfile.DeleteRecursively(checkpoint_dir)
    except errors.OpError:
      pass  # Ignore
    gfile.MakeDirs(checkpoint_dir)

    with ops.Graph().as_default():
      v = variable_v1.VariableV1(1, name="v")
      sm = session_manager.SessionManager(
          ready_op=variables.assert_variables_initialized())
      saver = saver_lib.Saver({"v": v})
      sess, initialized = sm.recover_session(
          "", saver=saver, checkpoint_dir=checkpoint_dir)
      self.assertFalse(initialized)
      sess.run(v.initializer)
      self.assertEqual(1, sess.run(v))
      saver.save(sess, os.path.join(checkpoint_dir,
                                    "recover_session_checkpoint"))
    # Create a new Graph and SessionManager and recover.
    with ops.Graph().as_default():
      v = variable_v1.VariableV1(2, name="v")
      with self.cached_session():
        self.assertEqual(False, variable_v1.is_variable_initialized(v).eval())
      sm2 = session_manager.SessionManager(
          ready_op=variables.assert_variables_initialized())
      saver = saver_lib.Saver({"v": v})
      sess, initialized = sm2.recover_session(
          "", saver=saver, checkpoint_dir=checkpoint_dir)
      self.assertTrue(initialized)
      self.assertEqual(
          True,
          variable_v1.is_variable_initialized(
              sess.graph.get_tensor_by_name("v:0")).eval(session=sess))
      self.assertEqual(1, sess.run(v))

  def testWaitForSessionReturnsNoneAfterTimeout(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says the call returns nothing after a timeout, but the test asserts
    # that a deadline-exceeded error is raised.
    # Scenario: an uninitialized variable, the assert-variables-initialized readiness op and a
    # one second recovery wait. Waiting for a session with a three second maximum must raise a
    # deadline-exceeded error.
    with ops.Graph().as_default():
      variable_v1.VariableV1(1, name="v")
      sm = session_manager.SessionManager(
          ready_op=variables.assert_variables_initialized(),
          recovery_wait_secs=1)

      # Set max_wait_secs to allow us to try a few times.
      with self.assertRaises(errors.DeadlineExceededError):
        sm.wait_for_session(master="", max_wait_secs=3)


if __name__ == "__main__":
  test.main()
