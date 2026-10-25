# [MODULE DOC]
# Unit tests for the session run hooks in the basic_session_run_hooks module. Defines a
# recording checkpoint-saver listener helper and test cases for the second-or-step timer, the
# stop-at-step, logging-tensor, checkpoint-saver (single step, multi step, resource variable),
# step-counter, summary-saver (regular and resource), global-step-waiter, final-ops, feed-fn
# and profiler hooks. Tests use hooked or monitored sessions, mocked clocks and sleep, a mocked
# logger, and a fake summary writer, with temporary directories for checkpoints and traces.
# Running the file executes the test main entry point.
#
# pylint: disable=g-bad-file-header
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
"""Tests for basic_session_run_hooks."""

import os.path
import shutil
import tempfile
import time

from tensorflow.python.client import session as session_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from tensorflow.python.summary import summary as summary_lib
from tensorflow.python.summary.writer import fake_summary_writer
from tensorflow.python.summary.writer import writer_cache
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import monitored_session
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util


# Provide a realistic start time for unit tests where we need to mock out
# calls to time.time().
MOCK_START_TIME = 1484695987.209386


class MockCheckpointSaverListener(
    basic_session_run_hooks.CheckpointSaverListener):
  # [CLASS DOC]
  # HELPER: test double used by the checkpoint saver hook tests.
  # A checkpoint-saver listener that records how many times each callback (begin, before save,
  # after save, end) was invoked, and can be told to request that training stop from the
  # after-save callback. Counts are exposed through a mapping returned by the counts getter.

  def __init__(self):
    # [FUNCTION DOC]
    # HELPER: constructor for the recording listener.
    # Takes no inputs. Sets the four callback counters to zero and the stop-request flag to
    # false. Returns nothing.
    self.begin_count = 0
    self.before_save_count = 0
    self.after_save_count = 0
    self.end_count = 0
    self.ask_for_stop = False

  def begin(self):
    # [FUNCTION DOC]
    # HELPER: listener callback invoked when the hook starts. Increments the begin counter.
    # Takes no inputs and returns nothing.
    self.begin_count += 1

  def before_save(self, session, global_step):
    # [FUNCTION DOC]
    # HELPER: listener callback invoked just before a checkpoint is written. Increments the
    # before-save counter. The session and global step arguments are ignored. Returns nothing.
    self.before_save_count += 1

  def after_save(self, session, global_step):
    # [FUNCTION DOC]
    # HELPER: listener callback invoked right after a checkpoint is written. Increments the
    # after-save counter; the session and global step arguments are ignored. Returns true
    # (requesting that training stop) if the stop-request flag was set, otherwise returns
    # nothing (null value).
    self.after_save_count += 1
    if self.ask_for_stop:
      return True

  def end(self, session, global_step):
    # [FUNCTION DOC]
    # HELPER: listener callback invoked when the hook ends. Increments the end counter; the
    # session and global step arguments are ignored. Returns nothing.
    self.end_count += 1

  def get_counts(self):
    # [FUNCTION DOC]
    # HELPER: reports the recorded callback counts. Takes no inputs and returns a mapping with
    # the keys begin, before_save, after_save and end, each holding an integer count.
    return {
        'begin': self.begin_count,
        'before_save': self.before_save_count,
        'after_save': self.after_save_count,
        'end': self.end_count
    }


class SecondOrStepTimerTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the timer that decides whether an action should trigger based on elapsed seconds
  # or elapsed steps. Covers argument validation, time-based triggering with a mocked clock,
  # step-based triggering, and the elapsed time/steps reported on update. No shared fixtures.

  @test_util.run_deprecated_v1
  def test_raise_in_both_secs_and_steps(self):
    # [FUNCTION DOC]
    # Scenario: construct the timer with both a seconds interval and a steps interval.
    # Expected: a value error is raised, since exactly one of them is allowed.
    with self.assertRaises(ValueError):
      basic_session_run_hooks.SecondOrStepTimer(every_secs=2.0, every_steps=10)

  @test_util.run_deprecated_v1
  def test_raise_in_none_secs_and_steps(self):
    # [FUNCTION DOC]
    # Scenario: construct the timer with neither a seconds nor a steps interval.
    # Expected: a value error is raised.
    with self.assertRaises(ValueError):
      basic_session_run_hooks.SecondOrStepTimer()

  @test.mock.patch.object(time, 'time')
  def test_every_secs(self, mock_time):
    # [FUNCTION DOC]
    # Scenario: a one-second timer with the wall clock mocked to a fixed start time (the mock is
    # injected by the patch decorator). Expected: the timer triggers for the first step, does not
    # trigger for the same or next step right after being updated, and after the mocked clock
    # advances by one second triggers for a newer step but not for the already-seen step.
    mock_time.return_value = MOCK_START_TIME
    timer = basic_session_run_hooks.SecondOrStepTimer(every_secs=1.0)
    self.assertTrue(timer.should_trigger_for_step(1))

    timer.update_last_triggered_step(1)
    self.assertFalse(timer.should_trigger_for_step(1))
    self.assertFalse(timer.should_trigger_for_step(2))

    mock_time.return_value += 1.0
    self.assertFalse(timer.should_trigger_for_step(1))
    self.assertTrue(timer.should_trigger_for_step(2))

  def test_every_steps(self):
    # [FUNCTION DOC]
    # Scenario: a timer that fires every three steps. Expected: it triggers for the first step;
    # after the update at step one it does not trigger for steps one to three and triggers again
    # at step four.
    timer = basic_session_run_hooks.SecondOrStepTimer(every_steps=3)
    self.assertTrue(timer.should_trigger_for_step(1))

    timer.update_last_triggered_step(1)
    self.assertFalse(timer.should_trigger_for_step(1))
    self.assertFalse(timer.should_trigger_for_step(2))
    self.assertFalse(timer.should_trigger_for_step(3))
    self.assertTrue(timer.should_trigger_for_step(4))

  def test_update_last_triggered_step(self):
    # [FUNCTION DOC]
    # Scenario: a timer with a one-step interval is updated at steps 1, 5 and 7. Expected: the
    # first update returns no elapsed seconds and no elapsed steps; later updates return positive
    # elapsed seconds and the step differences of four and then two.
    timer = basic_session_run_hooks.SecondOrStepTimer(every_steps=1)

    elapsed_secs, elapsed_steps = timer.update_last_triggered_step(1)
    self.assertEqual(None, elapsed_secs)
    self.assertEqual(None, elapsed_steps)

    elapsed_secs, elapsed_steps = timer.update_last_triggered_step(5)
    self.assertLess(0, elapsed_secs)
    self.assertEqual(4, elapsed_steps)

    elapsed_secs, elapsed_steps = timer.update_last_triggered_step(7)
    self.assertLess(0, elapsed_secs)
    self.assertEqual(2, elapsed_steps)


class StopAtStepTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the hook that requests a stop once a target global step is reached, configured
  # either by an absolute last step or by a number of steps from the start. Uses a hooked
  # session in a fresh graph with a global step variable; no shared fixtures.

  def test_raise_in_both_last_step_and_num_steps(self):
    # [FUNCTION DOC]
    # Scenario: construct the stop hook with both a step count and a last step.
    # Expected: a value error is raised because only one may be given.
    with self.assertRaises(ValueError):
      basic_session_run_hooks.StopAtStepHook(num_steps=10, last_step=20)

  def test_stop_based_on_last_step(self):
    # [FUNCTION DOC]
    # Scenario: hook with last step 10; the global step is set to 5, 9, 10 and then 11 between
    # runs of a no-op through a hooked session. Expected: the session does not report stop at
    # steps 5 and 9, reports stop at 10, and after the stop flag is manually cleared still
    # reports stop at 11.
    h = basic_session_run_hooks.StopAtStepHook(last_step=10)
    with ops.Graph().as_default():
      global_step = training_util.get_or_create_global_step()
      no_op = control_flow_ops.no_op()
      h.begin()
      with session_lib.Session() as sess:
        mon_sess = monitored_session._HookedSession(sess, [h])
        sess.run(state_ops.assign(global_step, 5))
        h.after_create_session(sess, None)
        mon_sess.run(no_op)
        self.assertFalse(mon_sess.should_stop())
        sess.run(state_ops.assign(global_step, 9))
        mon_sess.run(no_op)
        self.assertFalse(mon_sess.should_stop())
        sess.run(state_ops.assign(global_step, 10))
        mon_sess.run(no_op)
        self.assertTrue(mon_sess.should_stop())
        sess.run(state_ops.assign(global_step, 11))
        mon_sess._should_stop = False
        mon_sess.run(no_op)
        self.assertTrue(mon_sess.should_stop())

  def test_stop_based_on_num_step(self):
    # [FUNCTION DOC]
    # Scenario: hook configured to run ten steps; the hook is told the session started at global
    # step 5, then the step is set to 13, 14, 15 and 16. Expected: no stop through step 14, a
    # stop at 15 (start plus ten), and a stop again at 16 after the stop flag is manually cleared.
    h = basic_session_run_hooks.StopAtStepHook(num_steps=10)

    with ops.Graph().as_default():
      global_step = training_util.get_or_create_global_step()
      no_op = control_flow_ops.no_op()
      h.begin()
      with session_lib.Session() as sess:
        mon_sess = monitored_session._HookedSession(sess, [h])
        sess.run(state_ops.assign(global_step, 5))
        h.after_create_session(sess, None)
        mon_sess.run(no_op)
        self.assertFalse(mon_sess.should_stop())
        sess.run(state_ops.assign(global_step, 13))
        mon_sess.run(no_op)
        self.assertFalse(mon_sess.should_stop())
        sess.run(state_ops.assign(global_step, 14))
        mon_sess.run(no_op)
        self.assertFalse(mon_sess.should_stop())
        sess.run(state_ops.assign(global_step, 15))
        mon_sess.run(no_op)
        self.assertTrue(mon_sess.should_stop())
        sess.run(state_ops.assign(global_step, 16))
        mon_sess._should_stop = False
        mon_sess.run(no_op)
        self.assertTrue(mon_sess.should_stop())

  def test_stop_based_with_multiple_steps(self):
    # [FUNCTION DOC]
    # Scenario: hook configured for ten steps starting at global step 5; the step then jumps
    # directly to 15 in one run. Expected: no stop after the first run and a stop after the jump,
    # showing a step increment larger than one is handled.
    h = basic_session_run_hooks.StopAtStepHook(num_steps=10)

    with ops.Graph().as_default():
      global_step = training_util.get_or_create_global_step()
      no_op = control_flow_ops.no_op()
      h.begin()
      with session_lib.Session() as sess:
        mon_sess = monitored_session._HookedSession(sess, [h])
        sess.run(state_ops.assign(global_step, 5))
        h.after_create_session(sess, None)
        mon_sess.run(no_op)
        self.assertFalse(mon_sess.should_stop())
        sess.run(state_ops.assign(global_step, 15))
        mon_sess.run(no_op)
        self.assertTrue(mon_sess.should_stop())


class LoggingTensorHookTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the hook that logs tensor values every N iterations, every N seconds and/or at
  # the end of a session. The fixture replaces the info logging function with a recorder that
  # stores the latest logged arguments (and still forwards to the real logger), restored in
  # tearDown. Tests check which runs log by searching the recorded message for the tensor name.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: saves the real info logging function, clears the recorded message, and installs a
    # replacement that stores its positional arguments in the recorded message and then calls
    # the real function. Modifies global logging state until tearDown. Returns nothing.
    # Mock out logging calls so we can verify whether correct tensors are being
    # monitored.
    self._actual_log = tf_logging.info
    self.logged_message = None

    def mock_log(*args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: logging replacement installed by setUp. Stores the positional arguments as the
      # last logged message and forwards all arguments to the original info logger.
      # Returns nothing.
      self.logged_message = args
      self._actual_log(*args, **kwargs)

    tf_logging.info = mock_log

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture cleanup: restores the real info logging function saved in setUp. Returns nothing.
    tf_logging.info = self._actual_log

  def test_illegal_args(self):
    # [FUNCTION DOC]
    # Scenario: construct the logging hook with invalid interval settings. Expected: value errors
    # whose messages mention an invalid iteration interval for zero and negative values, and a
    # message about needing exactly one of the iteration or seconds options when both or neither
    # are provided.
    with self.assertRaisesRegex(ValueError, 'nvalid every_n_iter'):
      basic_session_run_hooks.LoggingTensorHook(tensors=['t'], every_n_iter=0)
    with self.assertRaisesRegex(ValueError, 'nvalid every_n_iter'):
      basic_session_run_hooks.LoggingTensorHook(tensors=['t'], every_n_iter=-10)
    with self.assertRaisesRegex(ValueError, 'xactly one of'):
      basic_session_run_hooks.LoggingTensorHook(
          tensors=['t'], every_n_iter=5, every_n_secs=5)
    with self.assertRaisesRegex(ValueError, 'xactly one of'):
      basic_session_run_hooks.LoggingTensorHook(tensors=['t'])

  def test_print_at_end_only(self):
    # [FUNCTION DOC]
    # Scenario: hook created with only the at-end option for a constant tensor; three training
    # runs go through a hooked session. Expected: nothing mentioning the tensor is logged during
    # the runs, and ending the hook logs a message containing the tensor name.
    with ops.Graph().as_default(), session_lib.Session() as sess:
      t = constant_op.constant(42.0, name='foo')
      train_op = constant_op.constant(3)
      hook = basic_session_run_hooks.LoggingTensorHook(
          tensors=[t.name], at_end=True)
      hook.begin()
      mon_sess = monitored_session._HookedSession(sess, [hook])
      self.evaluate(variables_lib.global_variables_initializer())
      self.logged_message = ''
      for _ in range(3):
        mon_sess.run(train_op)
        # assertNotRegexpMatches is not supported by python 3.1 and later
        self.assertEqual(str(self.logged_message).find(t.name), -1)

      hook.end(sess)
      self.assertRegex(str(self.logged_message), t.name)

  def _validate_print_every_n_steps(self, sess, at_end):
    # [FUNCTION DOC]
    # HELPER: shared check for the every-N-iterations logging tests. Inputs: an open session and
    # a flag for whether the hook also logs at end. Creates a hook logging every ten iterations
    # and verifies in a hooked session that the tensor is logged on the first run and on every
    # tenth run after, not logged in between, and that the counter resets properly after an
    # additional run. Finally ends the hook and asserts the tensor is logged only if at-end was
    # requested. Returns nothing; fails through assertions.
    t = constant_op.constant(42.0, name='foo')

    train_op = constant_op.constant(3)
    hook = basic_session_run_hooks.LoggingTensorHook(
        tensors=[t.name], every_n_iter=10, at_end=at_end)
    hook.begin()
    mon_sess = monitored_session._HookedSession(sess, [hook])
    self.evaluate(variables_lib.global_variables_initializer())
    mon_sess.run(train_op)
    self.assertRegex(str(self.logged_message), t.name)
    for _ in range(3):
      self.logged_message = ''
      for _ in range(9):
        mon_sess.run(train_op)
        # assertNotRegexpMatches is not supported by python 3.1 and later
        self.assertEqual(str(self.logged_message).find(t.name), -1)
      mon_sess.run(train_op)
      self.assertRegex(str(self.logged_message), t.name)

    # Add additional run to verify proper reset when called multiple times.
    self.logged_message = ''
    mon_sess.run(train_op)
    # assertNotRegexpMatches is not supported by python 3.1 and later
    self.assertEqual(str(self.logged_message).find(t.name), -1)

    self.logged_message = ''
    hook.end(sess)
    if at_end:
      self.assertRegex(str(self.logged_message), t.name)
    else:
      # assertNotRegexpMatches is not supported by python 3.1 and later
      self.assertEqual(str(self.logged_message).find(t.name), -1)

  def test_print_every_n_steps(self):
    # [FUNCTION DOC]
    # Scenario: runs the every-N-iterations check twice in the same session without the at-end
    # option. Expected: the periodic logging pattern holds both times, verifying proper reset.
    with ops.Graph().as_default(), session_lib.Session() as sess:
      self._validate_print_every_n_steps(sess, at_end=False)
      # Verify proper reset.
      self._validate_print_every_n_steps(sess, at_end=False)

  def test_print_every_n_steps_and_end(self):
    # [FUNCTION DOC]
    # Scenario: same as the periodic logging test but with the at-end option enabled, run twice.
    # Expected: periodic logging as before, plus a log of the tensor when the hook ends.
    with ops.Graph().as_default(), session_lib.Session() as sess:
      self._validate_print_every_n_steps(sess, at_end=True)
      # Verify proper reset.
      self._validate_print_every_n_steps(sess, at_end=True)

  def test_print_first_step(self):
    # [FUNCTION DOC]
    # Scenario: hook given a mapping of tag to tensor and an interval of one iteration. After a
    # single run, expected: the message mentions the tag and contains no elapsed-time text,
    # because the first iteration has no previous trigger to measure from.
    # if it runs every iteration, first iteration has None duration.
    with ops.Graph().as_default(), session_lib.Session() as sess:
      t = constant_op.constant(42.0, name='foo')
      train_op = constant_op.constant(3)
      hook = basic_session_run_hooks.LoggingTensorHook(
          tensors={'foo': t}, every_n_iter=1)
      hook.begin()
      mon_sess = monitored_session._HookedSession(sess, [hook])
      self.evaluate(variables_lib.global_variables_initializer())
      mon_sess.run(train_op)
      self.assertRegex(str(self.logged_message), 'foo')
      # in first run, elapsed time is None.
      self.assertEqual(str(self.logged_message).find('sec'), -1)

  def _validate_print_every_n_secs(self, sess, at_end, mock_time):
    # [FUNCTION DOC]
    # HELPER: shared check for the every-N-seconds logging tests. Inputs: an open session, a
    # flag for logging at end, and the mocked clock whose return value the check advances.
    # Creates a hook logging every second; verifies the tensor is logged on the first run, not
    # logged on an immediate second run, logged again after the mocked clock advances one
    # second, and that ending the hook logs it only when at-end was requested.
    # Returns nothing; fails through assertions.
    t = constant_op.constant(42.0, name='foo')
    train_op = constant_op.constant(3)

    hook = basic_session_run_hooks.LoggingTensorHook(
        tensors=[t.name], every_n_secs=1.0, at_end=at_end)
    hook.begin()
    mon_sess = monitored_session._HookedSession(sess, [hook])
    self.evaluate(variables_lib.global_variables_initializer())

    mon_sess.run(train_op)
    self.assertRegex(str(self.logged_message), t.name)

    # assertNotRegexpMatches is not supported by python 3.1 and later
    self.logged_message = ''
    mon_sess.run(train_op)
    self.assertEqual(str(self.logged_message).find(t.name), -1)
    mock_time.return_value += 1.0

    self.logged_message = ''
    mon_sess.run(train_op)
    self.assertRegex(str(self.logged_message), t.name)

    self.logged_message = ''
    hook.end(sess)
    if at_end:
      self.assertRegex(str(self.logged_message), t.name)
    else:
      # assertNotRegexpMatches is not supported by python 3.1 and later
      self.assertEqual(str(self.logged_message).find(t.name), -1)

  @test.mock.patch.object(time, 'time')
  def test_print_every_n_secs(self, mock_time):
    # [FUNCTION DOC]
    # Scenario: with the clock mocked to a fixed start time (injected by the patch decorator),
    # runs the seconds-based logging check twice without the at-end option. Expected: time-based
    # logging behaves identically both times, verifying proper reset.
    with ops.Graph().as_default(), session_lib.Session() as sess:
      mock_time.return_value = MOCK_START_TIME
      self._validate_print_every_n_secs(sess, at_end=False, mock_time=mock_time)
      # Verify proper reset.
      self._validate_print_every_n_secs(sess, at_end=False, mock_time=mock_time)

  @test.mock.patch.object(time, 'time')
  def test_print_every_n_secs_and_end(self, mock_time):
    # [FUNCTION DOC]
    # Scenario: same as the seconds-based logging test but with the at-end option enabled and the
    # clock mocked (injected by the patch decorator). Expected: time-based logging plus a final
    # log when the hook ends, both times the check runs.
    with ops.Graph().as_default(), session_lib.Session() as sess:
      mock_time.return_value = MOCK_START_TIME
      self._validate_print_every_n_secs(sess, at_end=True, mock_time=mock_time)
      # Verify proper reset.
      self._validate_print_every_n_secs(sess, at_end=True, mock_time=mock_time)

  def test_print_formatter(self):
    # [FUNCTION DOC]
    # Scenario: hook created with a custom formatter that renders the tensor value as a custom
    # prefix string. After one run, expected: the first logged argument equals the formatter's
    # output for the constant value 42.
    with ops.Graph().as_default(), session_lib.Session() as sess:
      t = constant_op.constant(42.0, name='foo')
      train_op = constant_op.constant(3)
      hook = basic_session_run_hooks.LoggingTensorHook(
          tensors=[t.name], every_n_iter=10,
          formatter=lambda items: 'qqq=%s' % items[t.name])
      hook.begin()
      mon_sess = monitored_session._HookedSession(sess, [hook])
      self.evaluate(variables_lib.global_variables_initializer())
      mon_sess.run(train_op)
      self.assertEqual(self.logged_message[0], 'qqq=42.0')


class CheckpointSaverHookTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the hook that periodically saves checkpoints, including argument validation,
  # save triggers, listener callbacks, listener-requested stops, default saver behavior and
  # saving of graph/meta graph/summary data. Fixture: a temporary model directory, a graph with
  # a scaffold, a global step and a train op that increments it by one.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: creates a temporary model directory, a new graph, and in that graph a scaffold,
    # the global step variable and an increment train op, stored as attributes. Returns nothing.
    self.model_dir = tempfile.mkdtemp()
    self.graph = ops.Graph()
    with self.graph.as_default():
      self.scaffold = monitored_session.Scaffold()
      self.global_step = training_util.get_or_create_global_step()
      self.train_op = training_util._increment_global_step(1)

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture cleanup: recursively removes the temporary model directory, ignoring errors.
    # Returns nothing.
    shutil.rmtree(self.model_dir, ignore_errors=True)

  def test_saves_when_saver_and_scaffold_both_missing(self):
    # [FUNCTION DOC]
    # Scenario: hook with a save interval of one step and neither a saver nor a scaffold.
    # Action: begin the hook, finalize the fixture scaffold, run one training step in a hooked
    # session. Expected: a checkpoint is written to the model directory and its stored global
    # step value is one.
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_steps=1)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)
        self.assertEqual(1,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

  def test_raise_when_saver_and_scaffold_both_present(self):
    # [FUNCTION DOC]
    # Scenario: construct the saver hook with both a saver and a scaffold. Expected: a value
    # error is raised because only one may be given.
    with self.assertRaises(ValueError):
      basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, saver=self.scaffold.saver, scaffold=self.scaffold)

  @test_util.run_deprecated_v1
  def test_raise_in_both_secs_and_steps(self):
    # [FUNCTION DOC]
    # Scenario: construct the saver hook with both a save-seconds and a save-steps interval.
    # Expected: a value error is raised.
    with self.assertRaises(ValueError):
      basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_secs=10, save_steps=20)

  @test_util.run_deprecated_v1
  def test_raise_in_none_secs_and_steps(self):
    # [FUNCTION DOC]
    # Scenario: construct the saver hook with neither a seconds nor a steps interval.
    # Expected: a value error is raised.
    with self.assertRaises(ValueError):
      basic_session_run_hooks.CheckpointSaverHook(self.model_dir)

  def test_save_secs_saves_in_first_step(self):
    # [FUNCTION DOC]
    # Scenario: hook with a two-second interval and the fixture scaffold. Action: one training
    # step in a hooked session. Expected: a checkpoint is saved on the very first step, with the
    # stored global step equal to one.
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_secs=2, scaffold=self.scaffold)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)
        self.assertEqual(1,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

  def test_save_secs_calls_listeners_at_begin_and_end(self):
    # [FUNCTION DOC]
    # Scenario: hook with a two-second interval and one recording listener. Action: two training
    # steps (the first saves, the second does not) then ending the hook, which saves again.
    # Expected: listener counts of one begin, two before-save, two after-save and one end.
    with self.graph.as_default():
      listener = MockCheckpointSaverListener()
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir,
          save_secs=2,
          scaffold=self.scaffold,
          listeners=[listener])
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)  # hook runs here
        mon_sess.run(self.train_op)  # hook won't run here, so it does at end
        hook.end(sess)  # hook runs here
      self.assertEqual({
          'begin': 1,
          'before_save': 2,
          'after_save': 2,
          'end': 1
      }, listener.get_counts())

  def test_listener_with_monitored_session(self):
    # [FUNCTION DOC]
    # Scenario: hook saving every step with a recording listener and explicit scaffold, run
    # inside a singular monitored session over two training steps. Expected: the global step
    # is two, and the listener saw one begin, three saves (before and after) and one end.
    with ops.Graph().as_default():
      scaffold = monitored_session.Scaffold()
      global_step = training_util.get_or_create_global_step()
      train_op = training_util._increment_global_step(1)
      listener = MockCheckpointSaverListener()
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir,
          save_steps=1,
          scaffold=scaffold,
          listeners=[listener])
      with monitored_session.SingularMonitoredSession(
          hooks=[hook],
          scaffold=scaffold,
          checkpoint_dir=self.model_dir) as sess:
        sess.run(train_op)
        sess.run(train_op)
        global_step_val = sess.raw_session().run(global_step)
      listener_counts = listener.get_counts()
    self.assertEqual(2, global_step_val)
    self.assertEqual({
        'begin': 1,
        'before_save': 3,
        'after_save': 3,
        'end': 1
    }, listener_counts)

  def test_listener_stops_training_in_after_save(self):
    # [FUNCTION DOC]
    # Scenario: saver hook with a listener whose after-save callback can request a stop, in a
    # singular monitored session. Action: two normal steps, then enable the stop request and run
    # another step. Expected: the session does not report stop after the first two steps and
    # reports stop after the third.
    with ops.Graph().as_default():
      scaffold = monitored_session.Scaffold()
      training_util.get_or_create_global_step()
      train_op = training_util._increment_global_step(1)
      listener = MockCheckpointSaverListener()
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_steps=1, scaffold=scaffold, listeners=[listener])
      with monitored_session.SingularMonitoredSession(
          hooks=[hook], scaffold=scaffold,
          checkpoint_dir=self.model_dir) as sess:
        sess.run(train_op)
        self.assertFalse(sess.should_stop())
        sess.run(train_op)
        self.assertFalse(sess.should_stop())
        listener.ask_for_stop = True
        sess.run(train_op)
        self.assertTrue(sess.should_stop())

  def test_listener_with_default_saver(self):
    # [FUNCTION DOC]
    # Scenario: saver hook with a listener but no scaffold, so the default saver is used, in a
    # monitored session for two steps. Expected: global step two and listener counts of one
    # begin, three saves and one end. A second session in a fresh graph restoring from the same
    # directory must see a global step of two, proving a checkpoint was written.
    with ops.Graph().as_default():
      global_step = training_util.get_or_create_global_step()
      train_op = training_util._increment_global_step(1)
      listener = MockCheckpointSaverListener()
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir,
          save_steps=1,
          listeners=[listener])
      with monitored_session.SingularMonitoredSession(
          hooks=[hook],
          checkpoint_dir=self.model_dir) as sess:
        sess.run(train_op)
        sess.run(train_op)
        global_step_val = sess.raw_session().run(global_step)
      listener_counts = listener.get_counts()
    self.assertEqual(2, global_step_val)
    self.assertEqual({
        'begin': 1,
        'before_save': 3,
        'after_save': 3,
        'end': 1
    }, listener_counts)

    with ops.Graph().as_default():
      global_step = training_util.get_or_create_global_step()
      with monitored_session.SingularMonitoredSession(
          checkpoint_dir=self.model_dir) as sess2:
        global_step_saved_val = sess2.run(global_step)
    self.assertEqual(2, global_step_saved_val)

  def test_two_listeners_with_default_saver(self):
    # [FUNCTION DOC]
    # Scenario: saver hook with two recording listeners and the default saver, two training
    # steps in a monitored session. Expected: global step two and both listeners report the same
    # counts of one begin, three saves and one end.
    with ops.Graph().as_default():
      global_step = training_util.get_or_create_global_step()
      train_op = training_util._increment_global_step(1)
      listener1 = MockCheckpointSaverListener()
      listener2 = MockCheckpointSaverListener()
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir,
          save_steps=1,
          listeners=[listener1, listener2])
      with monitored_session.SingularMonitoredSession(
          hooks=[hook],
          checkpoint_dir=self.model_dir) as sess:
        sess.run(train_op)
        sess.run(train_op)
        global_step_val = sess.raw_session().run(global_step)
      listener1_counts = listener1.get_counts()
      listener2_counts = listener2.get_counts()
    self.assertEqual(2, global_step_val)
    self.assertEqual({
        'begin': 1,
        'before_save': 3,
        'after_save': 3,
        'end': 1
    }, listener1_counts)
    self.assertEqual(listener1_counts, listener2_counts)

    with ops.Graph().as_default():
      global_step = training_util.get_or_create_global_step()
      with monitored_session.SingularMonitoredSession(
          checkpoint_dir=self.model_dir) as sess2:
        global_step_saved_val = sess2.run(global_step)
    self.assertEqual(2, global_step_saved_val)

  @test.mock.patch.object(time, 'time')
  def test_save_secs_saves_periodically(self, mock_time):
    # [FUNCTION DOC]
    # Scenario: hook with a two-second interval; the clock is mocked (injected by the patch
    # decorator) and moved to 0, 0.5, 2.5, 2.6, 2.7 and 10 seconds after the start with one
    # training step at each. Expected: saves happen on the first step, then at 2.5 s and at 10 s
    # but not at the in-between times, verified by reading the stored global step (1, then 3,
    # then 6) from the checkpoint.
    with self.graph.as_default():
      mock_time.return_value = MOCK_START_TIME
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_secs=2, scaffold=self.scaffold)
      hook.begin()
      self.scaffold.finalize()

      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])

        mock_time.return_value = MOCK_START_TIME
        mon_sess.run(self.train_op)  # Saved.

        mock_time.return_value = MOCK_START_TIME + 0.5
        mon_sess.run(self.train_op)  # Not saved.

        self.assertEqual(1,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

        # Simulate 2.5 seconds of sleep.
        mock_time.return_value = MOCK_START_TIME + 2.5
        mon_sess.run(self.train_op)  # Saved.

        mock_time.return_value = MOCK_START_TIME + 2.6
        mon_sess.run(self.train_op)  # Not saved.

        mock_time.return_value = MOCK_START_TIME + 2.7
        mon_sess.run(self.train_op)  # Not saved.

        self.assertEqual(3,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

        # Simulate 7.5 more seconds of sleep (10 seconds from start.
        mock_time.return_value = MOCK_START_TIME + 10
        mon_sess.run(self.train_op)  # Saved.
        self.assertEqual(6,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

  @test.mock.patch.object(time, 'time')
  def test_save_secs_calls_listeners_periodically(self, mock_time):
    # [FUNCTION DOC]
    # Scenario: two-second interval hook with a recording listener and a mocked clock (injected
    # by the patch decorator) advanced through several times up to 7 seconds, then the hook is
    # ended. Expected: listener counts of one begin, four before-save, four after-save and one
    # end, i.e. saves at the first step, at 3.0 s, at 6.5 s and on end.
    with self.graph.as_default():
      mock_time.return_value = MOCK_START_TIME
      listener = MockCheckpointSaverListener()
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir,
          save_secs=2,
          scaffold=self.scaffold,
          listeners=[listener])
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])

        mock_time.return_value = MOCK_START_TIME + 0.5
        mon_sess.run(self.train_op)  # hook runs here

        mock_time.return_value = MOCK_START_TIME + 0.5
        mon_sess.run(self.train_op)

        mock_time.return_value = MOCK_START_TIME + 3.0
        mon_sess.run(self.train_op)  # hook runs here

        mock_time.return_value = MOCK_START_TIME + 3.5
        mon_sess.run(self.train_op)

        mock_time.return_value = MOCK_START_TIME + 4.0
        mon_sess.run(self.train_op)

        mock_time.return_value = MOCK_START_TIME + 6.5
        mon_sess.run(self.train_op)  # hook runs here

        mock_time.return_value = MOCK_START_TIME + 7.0
        mon_sess.run(self.train_op)  # hook won't run here, so it does at end

        mock_time.return_value = MOCK_START_TIME + 7.5
        hook.end(sess)  # hook runs here
      self.assertEqual({
          'begin': 1,
          'before_save': 4,
          'after_save': 4,
          'end': 1
      }, listener.get_counts())

  def test_save_steps_saves_in_first_step(self):
    # [FUNCTION DOC]
    # Scenario: hook with a two-step interval. Action: one training step. Expected: a checkpoint
    # is written on the first step and its stored global step is one.
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_steps=2, scaffold=self.scaffold)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)
        self.assertEqual(1,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

  def test_save_steps_saves_periodically(self):
    # [FUNCTION DOC]
    # Scenario: hook with a two-step interval, five consecutive training steps. Expected: the
    # checkpoint on disk holds global step 1 after two steps (not saved again yet), 3 after the
    # third step (saved), 3 after the fourth (not saved) and 5 after the fifth (saved).
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_steps=2, scaffold=self.scaffold)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)
        mon_sess.run(self.train_op)
        # Not saved
        self.assertEqual(1,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))
        mon_sess.run(self.train_op)
        # saved
        self.assertEqual(3,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))
        mon_sess.run(self.train_op)
        # Not saved
        self.assertEqual(3,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))
        mon_sess.run(self.train_op)
        # saved
        self.assertEqual(5,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

  def test_save_saves_at_end(self):
    # [FUNCTION DOC]
    # Scenario: hook with a two-second interval, two quick training steps, then ending the hook.
    # Expected: ending forces a final save so the checkpoint's stored global step is two.
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_secs=2, scaffold=self.scaffold)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)
        mon_sess.run(self.train_op)
        hook.end(sess)
        self.assertEqual(2,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

  def test_summary_writer_defs(self):
    # [FUNCTION DOC]
    # Scenario: a fake summary writer is installed in place of the real writer and the writer
    # cache is cleared; the saver hook runs its after-create-session step and one training step.
    # Expected: the fake writer received the expected log directory and exactly one meta graph
    # definition built from the graph (with shapes) and the scaffold's saver definition.
    # Side effects: installs the fake writer at start and uninstalls it at the end (not in a
    # cleanup, so a failure before the end leaves it installed); clears the writer cache.
    fake_summary_writer.FakeSummaryWriter.install()
    writer_cache.FileWriterCache.clear()
    summary_writer = writer_cache.FileWriterCache.get(self.model_dir)

    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_steps=2, scaffold=self.scaffold)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        hook.after_create_session(sess, None)
        mon_sess.run(self.train_op)
      summary_writer.assert_summaries(
          test_case=self,
          expected_logdir=self.model_dir,
          expected_added_meta_graphs=[
              meta_graph.create_meta_graph_def(
                  graph_def=self.graph.as_graph_def(add_shapes=True),
                  saver_def=self.scaffold.saver.saver_def)
          ])

    fake_summary_writer.FakeSummaryWriter.uninstall()

  def test_save_checkpoint_before_first_train_step(self):
    # [FUNCTION DOC]
    # Scenario: hook with a two-step interval; the session is created and the hook's
    # after-create-session step runs before any training. Expected: a checkpoint with global step
    # zero already exists, stays at zero after one training step, and is updated to two after the
    # second step.
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_steps=2, scaffold=self.scaffold)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        mon_sess = monitored_session._HookedSession(sess, [hook])
        sess.run(self.scaffold.init_op)
        hook.after_create_session(sess, None)
        # Verifies that checkpoint is saved at step 0.
        self.assertEqual(0,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))
        # Verifies that no checkpoint is saved after one training step.
        mon_sess.run(self.train_op)
        self.assertEqual(0,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))
        # Verifies that checkpoint is saved after save_steps.
        mon_sess.run(self.train_op)
        self.assertEqual(2,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

  def test_save_graph_def(self):
    # [FUNCTION DOC]
    # Scenario: hook with save-graph-definition enabled and a one-step interval. After the
    # after-create-session step, expected: a text graph file exists in the model directory and
    # exactly one meta graph file exists (for step zero); after one training step there are two
    # meta graph files.
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_steps=1, scaffold=self.scaffold,
          save_graph_def=True)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        sess.run(self.scaffold.init_op)
        hook.after_create_session(sess, None)

        self.assertIn('graph.pbtxt', os.listdir(self.model_dir))
        # Should have a single .meta file for step 0
        self.assertLen(gfile.Glob(os.path.join(self.model_dir, '*.meta')), 1)

        mon_sess.run(self.train_op)
        self.assertLen(gfile.Glob(os.path.join(self.model_dir, '*.meta')), 2)

  def test_save_graph_def_false(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: an inline comment says a single meta graph file is expected for step zero, but
    # the assertion actually checks that there are no meta graph files at all.
    # Scenario: hook with save-graph-definition disabled and a one-step interval. Expected: no
    # text graph file in the model directory and no meta graph files, both after the
    # after-create-session step and after one training step.
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_steps=1, scaffold=self.scaffold,
          save_graph_def=False)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        sess.run(self.scaffold.init_op)
        hook.after_create_session(sess, None)

        self.assertNotIn('graph.pbtxt', os.listdir(self.model_dir))
        # Should have a single .meta file for step 0
        self.assertEmpty(gfile.Glob(os.path.join(self.model_dir, '*.meta')))

        mon_sess.run(self.train_op)
        self.assertEmpty(gfile.Glob(os.path.join(self.model_dir, '*.meta')))




class CheckpointSaverHookMultiStepTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the checkpoint saver hook when each session run advances the global step by
  # several steps (five), after telling the hook the steps-per-run value. Fixture: temporary
  # model directory, graph, scaffold, global step and a train op incrementing by five.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: creates a temporary model directory and a new graph holding a scaffold, the global
    # step, and a train op that increments the step by five (the steps-per-run attribute).
    # Returns nothing.
    self.model_dir = tempfile.mkdtemp()
    self.graph = ops.Graph()
    self.steps_per_run = 5
    with self.graph.as_default():
      self.scaffold = monitored_session.Scaffold()
      self.global_step = training_util.get_or_create_global_step()
      self.train_op = training_util._increment_global_step(self.steps_per_run)

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture cleanup: recursively removes the temporary model directory, ignoring errors.
    # Returns nothing.
    shutil.rmtree(self.model_dir, ignore_errors=True)

  def test_save_steps_saves_in_first_step(self):
    # [FUNCTION DOC]
    # Scenario: save interval of ten steps (two runs) with five steps per run. Action: one
    # session run. Expected: a checkpoint is saved on the first run with stored global step five.
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir,
          save_steps=2*self.steps_per_run,
          scaffold=self.scaffold)
      hook._set_steps_per_run(self.steps_per_run)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)
        self.assertEqual(5,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

  def test_save_steps_saves_periodically(self):
    # [FUNCTION DOC]
    # Scenario: save interval of ten steps with five steps per run, five session runs.
    # Expected: the stored checkpoint step is 5 after the first run, still 5 after the second,
    # 15 after the third, still 15 after the fourth and 25 after the fifth, i.e. saves on every
    # other run.
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir,
          save_steps=2*self.steps_per_run,
          scaffold=self.scaffold)
      hook._set_steps_per_run(self.steps_per_run)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)
        # Saved (step=5)
        self.assertEqual(5,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

        mon_sess.run(self.train_op)
        # Not saved (step=10)
        self.assertEqual(5,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

        mon_sess.run(self.train_op)
        # Saved (step=15)
        self.assertEqual(15,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

        mon_sess.run(self.train_op)
        # Not saved (step=20)
        self.assertEqual(15,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

        mon_sess.run(self.train_op)
        # Saved (step=25)
        self.assertEqual(25,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))

  def test_save_steps_saves_at_end(self):
    # [FUNCTION DOC]
    # Scenario: save interval of ten steps with five steps per run; two runs then ending the
    # hook. Expected: the final checkpoint stored on end has global step ten.
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir,
          save_steps=2*self.steps_per_run,
          scaffold=self.scaffold)
      hook._set_steps_per_run(self.steps_per_run)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)
        mon_sess.run(self.train_op)
        hook.end(sess)
        self.assertEqual(10,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))


class ResourceCheckpointSaverHookTest(test.TestCase):
  # [CLASS DOC]
  # Tests that the checkpoint saver hook works when the global step is a resource variable
  # (created under a variable scope that requests resource variables). Fixture: temporary
  # model directory, graph, scaffold, resource global step and an increment train op.
  # Note: unlike its sibling test classes this class defines no tearDown, so the temporary
  # directory is not removed.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: creates a temporary model directory and a new graph with a scaffold, a resource
    # global step created inside a scope named foo, and an increment-by-one train op. Returns
    # nothing.
    self.model_dir = tempfile.mkdtemp()
    self.graph = ops.Graph()
    with self.graph.as_default():
      self.scaffold = monitored_session.Scaffold()
      with variable_scope.variable_scope('foo', use_resource=True):
        self.global_step = training_util.get_or_create_global_step()
      self.train_op = training_util._increment_global_step(1)

  def test_save_steps_saves_periodically(self):
    # [FUNCTION DOC]
    # Scenario: two-step interval hook with a resource-variable global step, five training
    # steps. Expected: stored checkpoint step is 1 after two steps, 3 after the third, 3 after
    # the fourth and 5 after the fifth, matching the non-resource behavior.
    with self.graph.as_default():
      hook = basic_session_run_hooks.CheckpointSaverHook(
          self.model_dir, save_steps=2, scaffold=self.scaffold)
      hook.begin()
      self.scaffold.finalize()
      with session_lib.Session() as sess:
        sess.run(self.scaffold.init_op)
        mon_sess = monitored_session._HookedSession(sess, [hook])
        mon_sess.run(self.train_op)
        mon_sess.run(self.train_op)
        # Not saved
        self.assertEqual(1,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))
        mon_sess.run(self.train_op)
        # saved
        self.assertEqual(3,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))
        mon_sess.run(self.train_op)
        # Not saved
        self.assertEqual(3,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))
        mon_sess.run(self.train_op)
        # saved
        self.assertEqual(5,
                         checkpoint_utils.load_variable(self.model_dir,
                                                        self.global_step.name))


class StepCounterHookTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the hook that measures global steps per second and writes it as a summary, either
  # every N steps or every N seconds, including steps-per-run handling and a warning when the
  # global step does not increase. Uses a fake summary writer and a mocked clock. Fixture: a
  # temporary log directory.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: creates a temporary log directory and stores its path. Returns nothing.
    self.log_dir = tempfile.mkdtemp()

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture cleanup: recursively removes the temporary log directory, ignoring errors.
    # Returns nothing.
    shutil.rmtree(self.log_dir, ignore_errors=True)

  @test.mock.patch.object(time, 'time')
  def test_step_counter_every_n_steps(self, mock_time):
    # [FUNCTION DOC]
    # Scenario: hook with a fake summary writer and an interval of ten steps; the mocked clock
    # (injected by the patch decorator) advances 0.01 s per run for thirty runs, with the warning
    # logger replaced by a mock. Expected: no warning is logged; after ending, summaries exist
    # exactly at steps 11 and 21, each a positive global_step/sec value; the writer saw the
    # expected log directory and graph.
    mock_time.return_value = MOCK_START_TIME
    with ops.Graph().as_default() as g, session_lib.Session() as sess:
      training_util.get_or_create_global_step()
      train_op = training_util._increment_global_step(1)
      summary_writer = fake_summary_writer.FakeSummaryWriter(self.log_dir, g)
      hook = basic_session_run_hooks.StepCounterHook(
          summary_writer=summary_writer, every_n_steps=10)
      hook.begin()
      self.evaluate(variables_lib.global_variables_initializer())
      mon_sess = monitored_session._HookedSession(sess, [hook])
      with test.mock.patch.object(tf_logging, 'warning') as mock_log:
        for _ in range(30):
          mock_time.return_value += 0.01
          mon_sess.run(train_op)
        # logging.warning should not be called.
        self.assertIsNone(mock_log.call_args)
      hook.end(sess)
      summary_writer.assert_summaries(
          test_case=self,
          expected_logdir=self.log_dir,
          expected_graph=g,
          expected_summaries={})
      self.assertItemsEqual([11, 21], summary_writer.summaries.keys())
      for step in [11, 21]:
        summary_value = summary_writer.summaries[step][0].value[0]
        self.assertEqual('global_step/sec', summary_value.tag)
        self.assertGreater(summary_value.simple_value, 0)

  @test.mock.patch.object(time, 'time')
  def test_step_counter_every_n_secs(self, mock_time):
    # [FUNCTION DOC]
    # Scenario: hook with no step interval and a 0.1 second interval; the mocked clock (injected
    # by the patch decorator) advances 0.2 s between three runs. Expected: summaries exist at
    # steps 2 and 3 only, each a positive global_step/sec value, and the fake writer has the
    # expected log directory and graph.
    mock_time.return_value = MOCK_START_TIME
    with ops.Graph().as_default() as g, session_lib.Session() as sess:
      training_util.get_or_create_global_step()
      train_op = training_util._increment_global_step(1)
      summary_writer = fake_summary_writer.FakeSummaryWriter(self.log_dir, g)
      hook = basic_session_run_hooks.StepCounterHook(
          summary_writer=summary_writer, every_n_steps=None, every_n_secs=0.1)

      hook.begin()
      self.evaluate(variables_lib.global_variables_initializer())
      mon_sess = monitored_session._HookedSession(sess, [hook])
      mon_sess.run(train_op)
      mock_time.return_value += 0.2
      mon_sess.run(train_op)
      mock_time.return_value += 0.2
      mon_sess.run(train_op)
      hook.end(sess)

      summary_writer.assert_summaries(
          test_case=self,
          expected_logdir=self.log_dir,
          expected_graph=g,
          expected_summaries={})
      self.assertTrue(summary_writer.summaries, 'No summaries were created.')
      self.assertItemsEqual([2, 3], summary_writer.summaries.keys())
      for summary in summary_writer.summaries.values():
        summary_value = summary[0].value[0]
        self.assertEqual('global_step/sec', summary_value.tag)
        self.assertGreater(summary_value.simple_value, 0)

  def test_global_step_name(self):
    # [FUNCTION DOC]
    # Scenario: the global step is a non-trainable variable named foo under scope bar registered
    # in the global step collection; hook with a one-step interval and a fake writer, two runs.
    # Expected: a summary exists only at step 2 and its tag is derived from the variable's name
    # (the bar/foo name followed by a per-second suffix) rather than a fixed global step name.
    with ops.Graph().as_default() as g, session_lib.Session() as sess:
      with variable_scope.variable_scope('bar'):
        variable_scope.get_variable(
            'foo',
            initializer=0,
            trainable=False,
            collections=[
                ops.GraphKeys.GLOBAL_STEP, ops.GraphKeys.GLOBAL_VARIABLES
            ])
      train_op = training_util._increment_global_step(1)
      summary_writer = fake_summary_writer.FakeSummaryWriter(self.log_dir, g)
      hook = basic_session_run_hooks.StepCounterHook(
          summary_writer=summary_writer, every_n_steps=1, every_n_secs=None)

      hook.begin()
      self.evaluate(variables_lib.global_variables_initializer())
      mon_sess = monitored_session._HookedSession(sess, [hook])
      mon_sess.run(train_op)
      mon_sess.run(train_op)
      hook.end(sess)

      summary_writer.assert_summaries(
          test_case=self,
          expected_logdir=self.log_dir,
          expected_graph=g,
          expected_summaries={})
      self.assertTrue(summary_writer.summaries, 'No summaries were created.')
      self.assertItemsEqual([2], summary_writer.summaries.keys())
      summary_value = summary_writer.summaries[2][0].value[0]
      self.assertEqual('bar/foo/sec', summary_value.tag)

  def test_log_warning_if_global_step_not_increased(self):
    # [FUNCTION DOC]
    # Scenario: a train op that increments the global step by zero, hook with a one-step
    # interval and no summary writer, one initial run then thirty more with the log-first-n
    # logger mocked. Expected: the logged call text says the global step has not been increased.
    with ops.Graph().as_default(), session_lib.Session() as sess:
      training_util.get_or_create_global_step()
      train_op = training_util._increment_global_step(0)  # keep same.
      self.evaluate(variables_lib.global_variables_initializer())
      hook = basic_session_run_hooks.StepCounterHook(
          every_n_steps=1, every_n_secs=None)
      hook.begin()
      mon_sess = monitored_session._HookedSession(sess, [hook])
      mon_sess.run(train_op)  # Run one step to record global step.
      with test.mock.patch.object(tf_logging, 'log_first_n') as mock_log:
        for _ in range(30):
          mon_sess.run(train_op)
        self.assertRegex(
            str(mock_log.call_args), 'global step.*has not been increased')
      hook.end(sess)

  def _setup_steps_per_run_test(self,
                                every_n_steps,
                                steps_per_run,
                                graph,
                                sess):
    # [FUNCTION DOC]
    # HELPER: shared setup for the steps-per-run step counter tests. Inputs: the hook's step
    # interval, the number of steps each run advances, the graph (given to the fake summary
    # writer) and an open session. Creates the global step, a train op incrementing it by
    # steps-per-run, a fake summary writer on the log directory, a step counter hook told the
    # steps-per-run value and begun, initializes variables, and wraps the session in a hooked
    # session. Stores the train op, writer, hook and hooked session as attributes. Returns
    # nothing.
    training_util.get_or_create_global_step()
    self.train_op = training_util._increment_global_step(steps_per_run)
    self.summary_writer = fake_summary_writer.FakeSummaryWriter(
        self.log_dir, graph)
    self.hook = basic_session_run_hooks.StepCounterHook(
        summary_writer=self.summary_writer, every_n_steps=every_n_steps)
    self.hook._set_steps_per_run(steps_per_run)
    self.hook.begin()
    self.evaluate(variables_lib.global_variables_initializer())
    self.mon_sess = monitored_session._HookedSession(sess, [self.hook])

  @test.mock.patch.object(time, 'time')
  def test_steps_per_run_less_than_every_n_steps(self, mock_time):
    # [FUNCTION DOC]
    # Scenario: interval of ten steps with five steps per run (clock mocked via the patch
    # decorator), five runs. Expected: summaries exist exactly at steps 15 and 25, each a
    # positive global_step/sec value.
    mock_time.return_value = MOCK_START_TIME
    with ops.Graph().as_default() as g, session_lib.Session() as sess:
      self._setup_steps_per_run_test(10, 5, g, sess)

      # Logs at 15, 25
      for _ in range(5):
        mock_time.return_value += 0.01
        self.mon_sess.run(self.train_op)

      self.hook.end(sess)
      self.summary_writer.assert_summaries(
          test_case=self,
          expected_logdir=self.log_dir,
          expected_graph=g,
          expected_summaries={})
      self.assertItemsEqual([15, 25], self.summary_writer.summaries.keys())
      for step in [15, 25]:
        summary_value = self.summary_writer.summaries[step][0].value[0]
        self.assertEqual('global_step/sec', summary_value.tag)
        self.assertGreater(summary_value.simple_value, 0)

  @test.mock.patch.object(time, 'time')
  def test_steps_per_run_equal_every_n_steps(self, mock_time):
    # [FUNCTION DOC]
    # Scenario: interval of five steps with five steps per run (clock mocked via the patch
    # decorator), five runs. Expected: summaries at steps 10, 15, 20 and 25, each a positive
    # global_step/sec value.
    mock_time.return_value = MOCK_START_TIME
    with ops.Graph().as_default() as g, session_lib.Session() as sess:
      self._setup_steps_per_run_test(5, 5, g, sess)

      # Logs at 10, 15, 20, 25
      for _ in range(5):
        mock_time.return_value += 0.01
        self.mon_sess.run(self.train_op)

      self.hook.end(sess)
      self.summary_writer.assert_summaries(
          test_case=self,
          expected_logdir=self.log_dir,
          expected_graph=g,
          expected_summaries={})
      self.assertItemsEqual([10, 15, 20, 25],
                            self.summary_writer.summaries.keys())
      for step in [10, 15, 20, 25]:
        summary_value = self.summary_writer.summaries[step][0].value[0]
        self.assertEqual('global_step/sec', summary_value.tag)
        self.assertGreater(summary_value.simple_value, 0)

  @test.mock.patch.object(time, 'time')
  def test_steps_per_run_greater_than_every_n_steps(self, mock_time):
    # [FUNCTION DOC]
    # Scenario: interval of five steps with ten steps per run (clock mocked via the patch
    # decorator), five runs. Expected: summaries at steps 20, 30, 40 and 50 (every run after the
    # first), each a positive global_step/sec value.
    mock_time.return_value = MOCK_START_TIME
    with ops.Graph().as_default() as g, session_lib.Session() as sess:
      self._setup_steps_per_run_test(5, 10, g, sess)

      # Logs at 20, 30, 40, 50
      for _ in range(5):
        mock_time.return_value += 0.01
        self.mon_sess.run(self.train_op)

      self.hook.end(sess)
      self.summary_writer.assert_summaries(
          test_case=self,
          expected_logdir=self.log_dir,
          expected_graph=g,
          expected_summaries={})
      self.assertItemsEqual([20, 30, 40, 50],
                            self.summary_writer.summaries.keys())
      for step in [20, 30, 40, 50]:
        summary_value = self.summary_writer.summaries[step][0].value[0]
        self.assertEqual('global_step/sec', summary_value.tag)
        self.assertGreater(summary_value.simple_value, 0)


@test_util.run_deprecated_v1
class SummarySaverHookTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the hook that evaluates summary ops and writes them to a summary writer on a
  # step or time schedule (the whole class is decorated to run in deprecated V1 graph mode).
  # Fixture: a fake summary writer on a dummy log directory, two scalar summaries over an
  # incrementing variable, a global step and a train op.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: calls the base setUp, creates a fake summary writer for a dummy log directory,
    # a variable that is incremented to produce a first summary value and a second summary
    # of twice that value, the global step and an increment-by-one train op, all stored as
    # attributes. Returns nothing.
    test.TestCase.setUp(self)

    self.log_dir = 'log/dir'
    self.summary_writer = fake_summary_writer.FakeSummaryWriter(self.log_dir)

    var = variables_lib.Variable(0.0)
    tensor = state_ops.assign_add(var, 1.0)
    tensor2 = tensor * 2
    self.summary_op = summary_lib.scalar('my_summary', tensor)
    self.summary_op2 = summary_lib.scalar('my_summary2', tensor2)

    training_util.get_or_create_global_step()
    self.train_op = training_util._increment_global_step(1)

  def test_raise_when_scaffold_and_summary_op_both_missing(self):
    # [FUNCTION DOC]
    # Scenario: construct the summary hook with neither a scaffold nor a summary op.
    # Expected: a value error is raised.
    with self.assertRaises(ValueError):
      basic_session_run_hooks.SummarySaverHook()

  def test_raise_when_scaffold_and_summary_op_both_present(self):
    # [FUNCTION DOC]
    # Scenario: construct the summary hook with both a scaffold and a summary op.
    # Expected: a value error is raised.
    with self.assertRaises(ValueError):
      basic_session_run_hooks.SummarySaverHook(
          scaffold=monitored_session.Scaffold(), summary_op=self.summary_op)

  def test_raise_in_both_secs_and_steps(self):
    # [FUNCTION DOC]
    # Scenario: construct the summary hook with both save-seconds and save-steps.
    # Expected: a value error is raised.
    with self.assertRaises(ValueError):
      basic_session_run_hooks.SummarySaverHook(
          save_secs=10, save_steps=20, summary_writer=self.summary_writer)

  def test_raise_in_none_secs_and_steps(self):
    # [FUNCTION DOC]
    # Scenario: construct the summary hook with both save-seconds and save-steps explicitly
    # set to nothing. Expected: a value error is raised.
    with self.assertRaises(ValueError):
      basic_session_run_hooks.SummarySaverHook(
          save_secs=None, save_steps=None, summary_writer=self.summary_writer)

  def test_save_steps(self):
    # [FUNCTION DOC]
    # Scenario: summary hook saving every eight steps with the first scalar summary op, thirty
    # training runs in a cached session. Expected: the fake writer holds summaries at global
    # steps 1, 9, 17 and 25 with the summary value rising 1.0, 2.0, 3.0, 4.0.
    hook = basic_session_run_hooks.SummarySaverHook(
        save_steps=8,
        summary_writer=self.summary_writer,
        summary_op=self.summary_op)

    with self.cached_session() as sess:
      hook.begin()
      self.evaluate(variables_lib.global_variables_initializer())
      mon_sess = monitored_session._HookedSession(sess, [hook])
      for _ in range(30):
        mon_sess.run(self.train_op)
      hook.end(sess)

    self.summary_writer.assert_summaries(
        test_case=self,
        expected_logdir=self.log_dir,
        expected_summaries={
            1: {
                'my_summary': 1.0
            },
            9: {
                'my_summary': 2.0
            },
            17: {
                'my_summary': 3.0
            },
            25: {
                'my_summary': 4.0
            },
        })

  def test_multiple_summaries(self):
    # [FUNCTION DOC]
    # Scenario: summary hook saving every eight steps with a list of two summary ops, ten runs.
    # Expected: the writer holds summaries at steps 1 and 9, each containing both summaries with
    # the second equal to twice the first (1.0/2.0, then 2.0/4.0).
    hook = basic_session_run_hooks.SummarySaverHook(
        save_steps=8,
        summary_writer=self.summary_writer,
        summary_op=[self.summary_op, self.summary_op2])

    with self.cached_session() as sess:
      hook.begin()
      self.evaluate(variables_lib.global_variables_initializer())
      mon_sess = monitored_session._HookedSession(sess, [hook])
      for _ in range(10):
        mon_sess.run(self.train_op)
      hook.end(sess)

    self.summary_writer.assert_summaries(
        test_case=self,
        expected_logdir=self.log_dir,
        expected_summaries={
            1: {
                'my_summary': 1.0,
                'my_summary2': 2.0
            },
            9: {
                'my_summary': 2.0,
                'my_summary2': 4.0
            },
        })

  @test.mock.patch.object(time, 'time')
  def test_save_secs_saving_once_every_step(self, mock_time):
    # [FUNCTION DOC]
    # Scenario: summary hook with a half-second interval; the mocked clock (injected by the
    # patch decorator) advances by 0.5 s after each of four runs. Expected: a summary is written
    # at every step 1 to 4 with values 1.0 to 4.0.
    mock_time.return_value = MOCK_START_TIME
    hook = basic_session_run_hooks.SummarySaverHook(
        save_secs=0.5,
        summary_writer=self.summary_writer,
        summary_op=self.summary_op)

    with self.cached_session() as sess:
      hook.begin()
      self.evaluate(variables_lib.global_variables_initializer())
      mon_sess = monitored_session._HookedSession(sess, [hook])
      for _ in range(4):
        mon_sess.run(self.train_op)
        mock_time.return_value += 0.5
      hook.end(sess)

    self.summary_writer.assert_summaries(
        test_case=self,
        expected_logdir=self.log_dir,
        expected_summaries={
            1: {
                'my_summary': 1.0
            },
            2: {
                'my_summary': 2.0
            },
            3: {
                'my_summary': 3.0
            },
            4: {
                'my_summary': 4.0
            },
        })

  @test.mock.patch.object(time, 'time')
  def test_save_secs_saving_once_every_three_steps(self, mock_time):
    # [FUNCTION DOC]
    # Scenario: summary hook with a nine-second interval; the mocked clock (injected by the
    # patch decorator) advances 3.1 s after each of eight runs. Expected: summaries are written
    # only at steps 1, 4 and 7 with values 1.0, 2.0 and 3.0.
    mock_time.return_value = 1484695987.209386
    hook = basic_session_run_hooks.SummarySaverHook(
        save_secs=9.,
        summary_writer=self.summary_writer,
        summary_op=self.summary_op)

    with self.cached_session() as sess:
      hook.begin()
      self.evaluate(variables_lib.global_variables_initializer())
      mon_sess = monitored_session._HookedSession(sess, [hook])
      for _ in range(8):
        mon_sess.run(self.train_op)
        mock_time.return_value += 3.1
      hook.end(sess)

    # 24.8 seconds passed (3.1*8), it saves every 9 seconds starting from first:
    self.summary_writer.assert_summaries(
        test_case=self,
        expected_logdir=self.log_dir,
        expected_summaries={
            1: {
                'my_summary': 1.0
            },
            4: {
                'my_summary': 2.0
            },
            7: {
                'my_summary': 3.0
            },
        })


class GlobalStepWaiterHookTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the hook that blocks before each run until the global step reaches a threshold.
  # Covers the no-wait case for threshold zero and the sleep-and-poll loop with a mocked sleep.
  # No shared fixtures.

  def test_not_wait_for_step_zero(self):
    # [FUNCTION DOC]
    # Scenario: waiter hook with a wait-until step of zero. Action: call its before-run step
    # with a run context for a session. Expected: it returns immediately without waiting for the
    # global step to increase (the test would hang otherwise).
    with ops.Graph().as_default():
      training_util.get_or_create_global_step()
      hook = basic_session_run_hooks.GlobalStepWaiterHook(wait_until_step=0)
      hook.begin()
      with session_lib.Session() as sess:
        # Before run should return without waiting gstep increment.
        hook.before_run(
            session_run_hook.SessionRunContext(
                original_args=None, session=sess))

  @test.mock.patch.object(time, 'sleep')
  def test_wait_for_step(self, mock_sleep):
    # [FUNCTION DOC]
    # Scenario: waiter hook that waits until global step 1000, with the sleep function mocked
    # (injected by the patch decorator) so each sleep call advances the step: the first call sets
    # it to 500, the second to 1100, and any further call raises an assertion error. Action: call
    # the hook's before-run step. Expected: it returns after exactly two sleep calls.
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      hook = basic_session_run_hooks.GlobalStepWaiterHook(wait_until_step=1000)
      hook.begin()

      with session_lib.Session() as sess:
        # Mock out calls to time.sleep() to update the global step.

        class Context:
          # [CLASS DOC]
          # HELPER: tiny mutable holder used by the mocked sleep. Its class-level counter records
          # how many times the mocked sleep has been called; it is never instantiated.
          counter = 0

        def mock_sleep_side_effect(seconds):
          # [FUNCTION DOC]
          # HELPER: side effect installed on the mocked sleep. Input: the requested sleep
          # duration, ignored. Increments the call counter; on the first call sets the global
          # step to 500, on the second to 1100, and on any later call raises an assertion error
          # stating the hook should have finished after two sleeps. Returns nothing.
          del seconds  # argument is ignored
          Context.counter += 1
          if Context.counter == 1:
            # The first time sleep() is called, we update the global_step from
            # 0 to 500.
            sess.run(state_ops.assign(gstep, 500))
          elif Context.counter == 2:
            # The second time sleep() is called, we update the global_step from
            # 500 to 1100.
            sess.run(state_ops.assign(gstep, 1100))
          else:
            raise AssertionError(
                'Expected before_run() to terminate after the second call to '
                'time.sleep()')

        mock_sleep.side_effect = mock_sleep_side_effect

        # Run the mocked-out interaction with the hook.
        self.evaluate(variables_lib.global_variables_initializer())
        run_context = session_run_hook.SessionRunContext(
            original_args=None, session=sess)
        hook.before_run(run_context)
        self.assertEqual(Context.counter, 2)


class FinalOpsHookTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the hook that evaluates a set of final ops when the session ends and exposes the
  # result as its final values attribute. Covers scalar, vector, exhausted-input and
  # feed-dictionary cases. No shared fixtures.

  def test_final_ops_is_scalar_tensor(self):
    # [FUNCTION DOC]
    # Scenario: final ops is a scalar constant. Action: begin the hook and end it with a session.
    # Expected: the hook's final values equal the constant.
    with ops.Graph().as_default():
      expected_value = 4
      final_ops = constant_op.constant(expected_value)

      hook = basic_session_run_hooks.FinalOpsHook(final_ops)
      hook.begin()

      with session_lib.Session() as session:
        hook.end(session)
        self.assertEqual(expected_value,
                         hook.final_ops_values)

  def test_final_ops_is_tensor(self):
    # [FUNCTION DOC]
    # Scenario: final ops is a constant vector of six integers. Action: begin and end the hook.
    # Expected: the final values converted to a list equal the original list.
    with ops.Graph().as_default():
      expected_values = [1, 6, 3, 5, 2, 4]
      final_ops = constant_op.constant(expected_values)

      hook = basic_session_run_hooks.FinalOpsHook(final_ops)
      hook.begin()

      with session_lib.Session() as session:
        hook.end(session)
        self.assertListEqual(expected_values,
                             hook.final_ops_values.tolist())

  def test_final_ops_triggers_out_of_range_error(self):
    # [FUNCTION DOC]
    # Scenario: final ops reads the next element of a one-element dataset iterator that has
    # already been fully consumed, with the warning logger mocked. Action: end the hook.
    # Expected: an out-of-range error mentioning end of sequence is raised, and the logged
    # warning text mentions the dependency back to some input source.
    with ops.Graph().as_default():
      dataset = dataset_ops.Dataset.range(1)
      iterator = dataset_ops.make_one_shot_iterator(dataset)
      read_ops = iterator.get_next()
      final_ops = read_ops

      hook = basic_session_run_hooks.FinalOpsHook(final_ops)
      hook.begin()

      with session_lib.Session() as session:
        session.run(read_ops)
        with test.mock.patch.object(tf_logging, 'warning') as mock_log:
          with self.assertRaisesRegex(errors.OutOfRangeError,
                                      'End of sequence'):
            hook.end(session)
          self.assertRegex(
              str(mock_log.call_args), 'dependency back to some input source')

  def test_final_ops_with_dictionary(self):
    # [FUNCTION DOC]
    # Scenario: final ops is a float placeholder and a feed mapping supplies a two-element list
    # for it. Action: begin the hook with the feed mapping and end it. Expected: the final
    # values converted to a list equal the fed list.
    with ops.Graph().as_default():
      expected_values = [4, -3]
      final_ops = array_ops.placeholder(dtype=dtypes.float32)
      final_ops_feed_dict = {final_ops: expected_values}

      hook = basic_session_run_hooks.FinalOpsHook(
          final_ops, final_ops_feed_dict)
      hook.begin()

      with session_lib.Session() as session:
        hook.end(session)
        self.assertListEqual(expected_values,
                             hook.final_ops_values.tolist())


@test_util.run_deprecated_v1
class ResourceSummarySaverHookTest(test.TestCase):
  # [CLASS DOC]
  # Tests that the summary saver hook works with resource variables (the whole class runs in
  # deprecated V1 graph mode). Fixture: a fake summary writer, a scalar summary over an
  # incrementing resource variable, a resource global step and an increment train op.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: calls the base setUp, creates a fake summary writer for a dummy log directory,
    # a resource variable that is incremented to give the summary value, a scalar summary op,
    # a resource global step created under a scope named foo, and an increment-by-one train op,
    # all stored as attributes. Returns nothing.
    test.TestCase.setUp(self)

    self.log_dir = 'log/dir'
    self.summary_writer = fake_summary_writer.FakeSummaryWriter(self.log_dir)

    var = variable_scope.get_variable('var', initializer=0.0, use_resource=True)
    tensor = state_ops.assign_add(var, 1.0)
    self.summary_op = summary_lib.scalar('my_summary', tensor)

    with variable_scope.variable_scope('foo', use_resource=True):
      training_util.create_global_step()
    self.train_op = training_util._increment_global_step(1)

  def test_save_steps(self):
    # [FUNCTION DOC]
    # Scenario: summary hook saving every eight steps with a summary op over a resource
    # variable, thirty training runs. Expected: the fake writer holds summaries at global steps
    # 1, 9, 17 and 25 with values 1.0, 2.0, 3.0 and 4.0.
    hook = basic_session_run_hooks.SummarySaverHook(
        save_steps=8,
        summary_writer=self.summary_writer,
        summary_op=self.summary_op)

    with self.cached_session() as sess:
      hook.begin()
      self.evaluate(variables_lib.global_variables_initializer())
      mon_sess = monitored_session._HookedSession(sess, [hook])
      for _ in range(30):
        mon_sess.run(self.train_op)
      hook.end(sess)

    self.summary_writer.assert_summaries(
        test_case=self,
        expected_logdir=self.log_dir,
        expected_summaries={
            1: {
                'my_summary': 1.0
            },
            9: {
                'my_summary': 2.0
            },
            17: {
                'my_summary': 3.0
            },
            25: {
                'my_summary': 4.0
            },
        })


class FeedFnHookTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the hook that supplies extra feed values from a user callback on every run.
  # No shared fixtures.

  def test_feeding_placeholder(self):
    # [FUNCTION DOC]
    # Scenario: a float placeholder x, an op computing x plus one, and a feed hook whose callback
    # returns a mapping giving x the value 1.0. Action: run the op through a hooked session
    # without feeding x directly. Expected: the result is 2, showing the hook's feed was used.
    with ops.Graph().as_default(), session_lib.Session() as sess:
      x = array_ops.placeholder(dtype=dtypes.float32)
      y = x + 1
      hook = basic_session_run_hooks.FeedFnHook(
          feed_fn=lambda: {x: 1.0})
      hook.begin()
      mon_sess = monitored_session._HookedSession(sess, [hook])
      self.assertEqual(mon_sess.run(y), 2)


class ProfilerHookTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the hook that periodically collects run metadata and writes timeline trace files.
  # Fixture: a temporary output directory, a graph with a global step, a train op that
  # increments it, and a file pattern matching the timeline json files written to the directory.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: calls the base setUp, creates a temporary output directory, a new graph holding
    # the global step and an increment train op, and the glob pattern for timeline files in the
    # output directory. Stored as attributes. Returns nothing.
    super(ProfilerHookTest, self).setUp()
    self.output_dir = tempfile.mkdtemp()
    self.graph = ops.Graph()
    self.filepattern = os.path.join(self.output_dir, 'timeline-*.json')
    with self.graph.as_default():
      self.global_step = training_util.get_or_create_global_step()
      self.train_op = state_ops.assign_add(self.global_step, 1)

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture cleanup: calls the base tearDown and then recursively removes the temporary output
    # directory, ignoring errors. Returns nothing.
    super(ProfilerHookTest, self).tearDown()
    shutil.rmtree(self.output_dir, ignore_errors=True)

  def _count_timeline_files(self):
    # [FUNCTION DOC]
    # HELPER: counts the timeline trace files currently in the output directory. Takes no
    # inputs and returns the number of files matching the fixture's file pattern.
    return len(gfile.Glob(self.filepattern))

  @test_util.run_deprecated_v1
  def test_raise_in_both_secs_and_steps(self):
    # [FUNCTION DOC]
    # Scenario: construct the profiler hook with both save-seconds and save-steps.
    # Expected: a value error is raised.
    with self.assertRaises(ValueError):
      basic_session_run_hooks.ProfilerHook(save_secs=10, save_steps=20)

  @test_util.run_deprecated_v1
  def test_raise_in_none_secs_and_steps(self):
    # [FUNCTION DOC]
    # Scenario: construct the profiler hook with both save-seconds and save-steps set to
    # nothing. Expected: a value error is raised.
    with self.assertRaises(ValueError):
      basic_session_run_hooks.ProfilerHook(save_secs=None, save_steps=None)

  def test_save_secs_does_not_save_in_first_step(self):
    # [FUNCTION DOC]
    # Scenario: profiler hook with a two-second interval in a singular monitored session.
    # Action: one training run. Expected: no timeline files have been written.
    with self.graph.as_default():
      hook = basic_session_run_hooks.ProfilerHook(
          save_secs=2, output_dir=self.output_dir)
      with monitored_session.SingularMonitoredSession(hooks=[hook]) as sess:
        sess.run(self.train_op)
        self.assertEqual(0, self._count_timeline_files())

  @test.mock.patch.object(time, 'time')
  def test_save_secs_saves_periodically(self, mock_time):
    # [FUNCTION DOC]
    # Scenario: profiler hook with a two-second interval; the clock is mocked (injected by the
    # patch decorator) and set to 0, 2.5, 2.6, 4.4 and 4.5 seconds after the start for five runs.
    # Expected: the timeline file count is 0 after the first run, 1 after the run at 2.5 s, still
    # 1 just before the next interval (4.4 s), and 2 after the run at 4.5 s.
    # Pick a fixed start time.
    with self.graph.as_default():
      mock_time.return_value = MOCK_START_TIME
      hook = basic_session_run_hooks.ProfilerHook(
          save_secs=2, output_dir=self.output_dir)
      with monitored_session.SingularMonitoredSession(hooks=[hook]) as sess:
        sess.run(self.train_op)  # Not saved.
        self.assertEqual(0, self._count_timeline_files())
        # Simulate 2.5 seconds of sleep.
        mock_time.return_value = MOCK_START_TIME + 2.5
        sess.run(self.train_op)  # Saved.
        self.assertEqual(1, self._count_timeline_files())

        # Pretend some small amount of time has passed.
        mock_time.return_value = MOCK_START_TIME + 2.6
        sess.run(self.train_op)  # Not saved.
        # Edge test just before we should save the timeline.
        mock_time.return_value = MOCK_START_TIME + 4.4
        sess.run(self.train_op)  # Not saved.
        self.assertEqual(1, self._count_timeline_files())

        mock_time.return_value = MOCK_START_TIME + 4.5
        sess.run(self.train_op)  # Saved.
        self.assertEqual(2, self._count_timeline_files())

  def test_save_steps_does_not_save_in_first_step(self):
    # [FUNCTION DOC]
    # Scenario: profiler hook with a one-step interval in a singular monitored session.
    # Action: one training run. Expected: no timeline files have been written yet.
    with self.graph.as_default():
      hook = basic_session_run_hooks.ProfilerHook(
          save_steps=1, output_dir=self.output_dir)
      with monitored_session.SingularMonitoredSession(hooks=[hook]) as sess:
        sess.run(self.train_op)  # Not saved.
        self.assertEqual(0, self._count_timeline_files())

  def test_save_steps_saves_periodically(self):
    # [FUNCTION DOC]
    # Scenario: profiler hook with a two-step interval, five training runs. Expected: the
    # timeline file count is 0 initially and after the first run, 1 after the second, 1 after
    # the third, 2 after the fourth and 2 after the fifth.
    with self.graph.as_default():
      hook = basic_session_run_hooks.ProfilerHook(
          save_steps=2, output_dir=self.output_dir)
      with monitored_session.SingularMonitoredSession(hooks=[hook]) as sess:
        self.assertEqual(0, self._count_timeline_files())
        sess.run(self.train_op)  # Not saved.
        self.assertEqual(0, self._count_timeline_files())
        sess.run(self.train_op)  # Saved.
        self.assertEqual(1, self._count_timeline_files())
        sess.run(self.train_op)  # Not saved.
        self.assertEqual(1, self._count_timeline_files())
        sess.run(self.train_op)  # Saved.
        self.assertEqual(2, self._count_timeline_files())
        sess.run(self.train_op)  # Not saved.
        self.assertEqual(2, self._count_timeline_files())

  def test_run_metadata_saves(self):
    # [FUNCTION DOC]
    # Scenario: the writer cache is cleared and a fake summary writer installed; the profiler
    # hook has a one-step interval and two training runs happen. Expected: the fake writer for
    # the output directory recorded run metadata under exactly one tag, for step 2 (the second
    # run). Side effects: installs the fake writer, clears the cache, and uninstalls the fake
    # at the end (not in a cleanup, so a failure earlier leaves it installed).
    writer_cache.FileWriterCache.clear()
    fake_summary_writer.FakeSummaryWriter.install()
    fake_writer = writer_cache.FileWriterCache.get(self.output_dir)
    with self.graph.as_default():
      hook = basic_session_run_hooks.ProfilerHook(
          save_steps=1, output_dir=self.output_dir)
      with monitored_session.SingularMonitoredSession(hooks=[hook]) as sess:
        sess.run(self.train_op)  # Not saved.
        sess.run(self.train_op)  # Saved.
        self.assertEqual(
            list(fake_writer._added_run_metadata.keys()), ['step_2'])
    fake_summary_writer.FakeSummaryWriter.uninstall()


if __name__ == '__main__':
  test.main()
