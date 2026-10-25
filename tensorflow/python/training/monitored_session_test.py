# [MODULE DOC]
# Unit tests for the monitored session machinery: Scaffold, MonitoredTrainingSession,
# MonitoredSession, SingularMonitoredSession and the internal wrapper sessions (wrapped,
# coordinated, recoverable, hooked). The tests cover scaffold default/overridden/copied ops,
# checkpoint and summary saving, distribute-coordinator control of savers, automatic recovery
# from aborted/unavailable errors, clean exit on end-of-input errors, propagation of other
# errors, hook interaction (feeds, fetches, run options merging) and step functions.
# The module also defines small fake hooks, sessions and strategies used only by these tests.
# No flags or registrations at import time; it runs through the standard test main entry.
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
"""Tests for monitored_session."""

import collections
import glob
import os
import sys
import threading
import time
import traceback

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import debug_pb2
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.client import session as session_lib
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import distribute_coordinator
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import load as saved_model_load
from tensorflow.python.summary import summary
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import coordinator
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver as saver_lib
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import summary_io
from tensorflow.python.training import training_util


def latest_summaries(base_dir):
  # [FUNCTION DOC]
  # HELPER: shared by the summary-related tests in this file to read back written summaries.
  # Input: base_dir, a directory path that may contain event files. Finds files whose names
  # start with the event-file prefix, sorts them by name and takes the last one (the newest).
  # Returns a list of all events in that file that carry a summary; returns an empty list when
  # no event file exists. Reads files only, no other side effects.
  """Parse summary events from latest event file in base_dir."""
  file_paths = glob.glob(os.path.join(base_dir, 'events.*'))
  file_path = sorted(file_paths)[-1] if file_paths else None
  latest_events = summary_io.summary_iterator(file_path) if file_path else []
  return [e for e in latest_events if e.HasField('summary')]

class ScaffoldTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the Scaffold class: checks which ops are lazily created on finalize, how
  # defaults are derived from the graph, caching of shared collections, user-supplied values,
  # graph finalization and copying settings from another scaffold. Each test builds its own
  # fresh graph so there is no shared fixture.
  """Scaffold tests."""

  def test_nothing_created_before_finalize(self):
    # [FUNCTION DOC]
    # Test: a new scaffold must not have created any of its ops or the saver before finalize
    # is called. Asserts that init op, init feed dict, init function, ready op, ready-for-local-init
    # op, local init op and saver are all null.
    with ops.Graph().as_default():
      scaffold = monitored_session.Scaffold()
      self.assertEqual(None, scaffold.init_op)
      self.assertEqual(None, scaffold.init_feed_dict)
      self.assertEqual(None, scaffold.init_fn)
      self.assertEqual(None, scaffold.ready_op)
      self.assertEqual(None, scaffold.ready_for_local_init_op)
      self.assertEqual(None, scaffold.local_init_op)
      self.assertEqual(None, scaffold.saver)

  def test_defaults_empty_graph(self):
    # [FUNCTION DOC]
    # Test: finalizing a scaffold in a graph with one global and one local variable creates the
    # default ops. Asserts the types of the created ops and saver, and runs them in a session:
    # the ready op reports both variables as uninitialized, the init op fixes the global one,
    # the local init op fixes the local one, after which the ready checks return empty results.
    with ops.Graph().as_default():
      scaffold = monitored_session.Scaffold()
      variable_v1.VariableV1(1, name='my_var')
      variable_v1.VariableV1(
          2, name='my_local_var', collections=[ops.GraphKeys.LOCAL_VARIABLES])
      scaffold.finalize()
      self.assertTrue(isinstance(scaffold.init_op, ops.Operation))
      self.assertEqual(None, scaffold.init_feed_dict)
      self.assertEqual(None, scaffold.init_fn)
      self.assertTrue(isinstance(scaffold.ready_op, tensor.Tensor))
      self.assertTrue(isinstance(
          scaffold.ready_for_local_init_op, tensor.Tensor))
      self.assertTrue(isinstance(scaffold.local_init_op, ops.Operation))
      self.assertEqual(None, scaffold.local_init_feed_dict)
      self.assertTrue(isinstance(scaffold.saver, saver_lib.Saver))
      with self.cached_session() as sess:
        self.assertItemsEqual([b'my_var', b'my_local_var'],
                              sess.run(scaffold.ready_op))
        self.assertItemsEqual([b'my_var'],
                              sess.run(scaffold.ready_for_local_init_op))
        sess.run(scaffold.init_op)
        self.assertEqual(0, len(sess.run(scaffold.ready_for_local_init_op)))
        sess.run(scaffold.local_init_op)
        self.assertEqual(0, len(sess.run(scaffold.ready_op)))

  def test_defaults_no_variables(self):
    # [FUNCTION DOC]
    # Test: finalizing a scaffold in a graph that has only a constant (no variables) still
    # produces an init op, ready op, ready-for-local-init op, local init op and a saver of the
    # expected types, with null feed dictionaries and a null init function.
    with ops.Graph().as_default():
      scaffold = monitored_session.Scaffold()
      constant_op.constant(1, name='my_const')
      scaffold.finalize()
      self.assertTrue(isinstance(scaffold.init_op, ops.Operation))
      self.assertEqual(None, scaffold.init_feed_dict)
      self.assertEqual(None, scaffold.init_fn)
      self.assertTrue(isinstance(scaffold.ready_op, tensor.Tensor))
      self.assertTrue(isinstance(
          scaffold.ready_for_local_init_op, tensor.Tensor))
      self.assertTrue(isinstance(scaffold.local_init_op, ops.Operation))
      self.assertEqual(None, scaffold.local_init_feed_dict)
      self.assertTrue(isinstance(scaffold.saver, saver_lib.Saver))

  def test_caches_values(self):
    # [FUNCTION DOC]
    # Test: two scaffolds finalized in the same graph share the cached ops. Asserts that init
    # op, ready op, ready-for-local-init op, local init op and saver are identical in both.
    with ops.Graph().as_default():
      variable_v1.VariableV1([1])
      scaffold1 = monitored_session.Scaffold()
      scaffold1.finalize()
      scaffold2 = monitored_session.Scaffold()
      scaffold2.finalize()
      self.assertEqual(scaffold1.init_op, scaffold2.init_op)
      self.assertEqual(scaffold1.ready_op, scaffold2.ready_op)
      self.assertEqual(scaffold1.ready_for_local_init_op,
                       scaffold2.ready_for_local_init_op)
      self.assertEqual(scaffold1.local_init_op, scaffold2.local_init_op)
      self.assertEqual(scaffold1.saver, scaffold2.saver)

  def test_raise_error_if_more_than_one_cached_item(self):
    # [FUNCTION DOC]
    # Test: if the saver collection holds two savers, finalizing a scaffold cannot choose one.
    # Asserts that a runtime error mentioning more than one item is raised.
    with ops.Graph().as_default():
      variable_v1.VariableV1([1])
      ops.add_to_collection(ops.GraphKeys.SAVERS, saver_lib.Saver())
      ops.add_to_collection(ops.GraphKeys.SAVERS, saver_lib.Saver())
      with self.assertRaisesRegex(RuntimeError, 'More than one item'):
        monitored_session.Scaffold().finalize()

  def test_uses_passed_values(self):
    # [FUNCTION DOC]
    # Test: values passed explicitly to the Scaffold constructor (init op, feed dicts, init
    # function, ready ops, local init op, saver) are kept as given after finalize, and the init
    # function is still callable. Placeholder numbers are used in place of real ops.
    with ops.Graph().as_default():
      variable_v1.VariableV1([1])
      saver = saver_lib.Saver()
      scaffold = monitored_session.Scaffold(
          init_op=2,
          init_feed_dict=3,
          init_fn=lambda scaffold, sess: 4,
          ready_op=5,
          ready_for_local_init_op=6,
          local_init_op=7,
          local_init_feed_dict=8,
          saver=saver)
      scaffold.finalize()
      self.assertEqual(2, scaffold.init_op)
      self.assertEqual(3, scaffold.init_feed_dict)
      self.assertTrue(callable(scaffold.init_fn))
      self.assertEqual(5, scaffold.ready_op)
      self.assertEqual(6, scaffold.ready_for_local_init_op)
      self.assertEqual(7, scaffold.local_init_op)
      self.assertEqual(8, scaffold.local_init_feed_dict)
      self.assertEqual(saver, scaffold.saver)

  def test_graph_is_finalized(self):
    # [FUNCTION DOC]
    # Test: finalizing a scaffold also finalizes its graph. Asserts that creating a new op
    # afterwards raises a runtime error saying the graph is finalized and cannot be modified.
    with ops.Graph().as_default():
      variable_v1.VariableV1([1])
      monitored_session.Scaffold().finalize()
      with self.assertRaisesRegex(RuntimeError,
                                  'Graph is finalized and cannot be modified'):
        constant_op.constant([0])

  def test_new_scaffold_from_default_scaffold(self):
    # [FUNCTION DOC]
    # Test: a scaffold built with explicit values and a copy-from scaffold created outside the
    # graph uses the explicit values for everything after finalize (explicit arguments win over
    # the copied scaffold's settings). Asserts each attribute equals the explicit value.
    scaffold1 = monitored_session.Scaffold()
    with ops.Graph().as_default():
      variable_v1.VariableV1([1])
      saver = saver_lib.Saver()
      scaffold2 = monitored_session.Scaffold(
          init_op=2,
          init_feed_dict=3,
          init_fn=lambda scaffold, sess: 4,
          ready_op=5,
          ready_for_local_init_op=6,
          local_init_op=7,
          local_init_feed_dict=8,
          saver=saver,
          copy_from_scaffold=scaffold1)

      scaffold2.finalize()
      self.assertEqual(2, scaffold2.init_op)
      self.assertEqual(3, scaffold2.init_feed_dict)
      self.assertTrue(callable(scaffold2.init_fn))
      self.assertEqual(5, scaffold2.ready_op)
      self.assertEqual(6, scaffold2.ready_for_local_init_op)
      self.assertEqual(7, scaffold2.local_init_op)
      self.assertEqual(8, scaffold2.local_init_feed_dict)
      self.assertEqual(saver, scaffold2.saver)

  def test_new_scaffold_from_existing_scaffold(self):
    # [FUNCTION DOC]
    # Test: a second scaffold copying from a first scaffold that has all values set still uses
    # its own explicitly passed values. Asserts every attribute of the second scaffold matches
    # the second scaffold's arguments, not the first scaffold's.
    with ops.Graph().as_default():
      variable_v1.VariableV1([1])
      saver = saver_lib.Saver()
      scaffold1 = monitored_session.Scaffold(
          init_op=2,
          init_feed_dict=3,
          init_fn=lambda scaffold, sess: 4,
          ready_op=5,
          ready_for_local_init_op=6,
          local_init_op=7,
          local_init_feed_dict=8,
          saver=saver)

      scaffold2 = monitored_session.Scaffold(
          init_op=4,
          init_feed_dict=6,
          init_fn=lambda scaffold, sess: 8,
          ready_op=10,
          ready_for_local_init_op=12,
          local_init_op=14,
          local_init_feed_dict=15,
          saver=saver,
          copy_from_scaffold=scaffold1)

      scaffold2.finalize()
      self.assertEqual(4, scaffold2.init_op)
      self.assertEqual(6, scaffold2.init_feed_dict)
      self.assertTrue(callable(scaffold2.init_fn))
      self.assertEqual(10, scaffold2.ready_op)
      self.assertEqual(12, scaffold2.ready_for_local_init_op)
      self.assertEqual(14, scaffold2.local_init_op)
      self.assertEqual(15, scaffold2.local_init_feed_dict)
      self.assertEqual(saver, scaffold2.saver)

  def test_copy_from_scaffold_is_scaffold(self):
    # [FUNCTION DOC]
    # Test: passing something that is not a Scaffold as the copy source is rejected. Asserts a
    # type error whose message says the copy source is not a Scaffold instance.
    with ops.Graph().as_default():
      with self.assertRaisesRegex(
          TypeError, 'copy_from_scaffold is not a Scaffold instance'):
        monitored_session.Scaffold(copy_from_scaffold=1)


def _test_dir(temp_dir, test_name):
  # [FUNCTION DOC]
  # HELPER: creates per-test scratch directories for the tests in this file.
  # Inputs: temp_dir, a base temporary directory path; test_name, the sub-directory name.
  # Joins the two; if the directory already exists, removes every file directly inside it
  # (sub-directories are not handled), otherwise creates it (with parents).
  # Returns the path of the empty directory. Side effects: deletes and creates files on disk.
  """Create an empty dir to use for tests.

  Args:
    temp_dir: Tmp directory path.
    test_name: Name of the test.

  Returns:
    Absolute path to the test directory.
  """
  test_dir = os.path.join(temp_dir, test_name)
  if os.path.isdir(test_dir):
    for f in glob.glob('%s/*' % test_dir):
      os.remove(f)
  else:
    os.makedirs(test_dir)
  return test_dir


class FakeHook(session_run_hook.SessionRunHook):
  # [CLASS DOC]
  # HELPER: test double hook used to observe how a session calls its hooks.
  # Subclass of the session run hook base. Records how often each callback (begin, session
  # creation, before run, after run, end) was called, the last run context and run values it
  # saw, and can be told to request a stop or to return a configured run request.

  def __init__(self):
    # [FUNCTION DOC]
    # HELPER: sets up the recording state of the fake hook: stop flag off, no request to return,
    # an empty per-callback call counter, and no last run context or run values.
    self.should_stop = False
    self.request = None
    self.call_counter = collections.Counter()
    self.last_run_context = None
    self.last_run_values = None

  def begin(self):
    # [FUNCTION DOC]
    # HELPER: hook callback invoked once before the session is created. Only increments the
    # begin counter. Returns nothing.
    self.call_counter['begin'] += 1

  def after_create_session(self, session, coord):  # pylint: disable=unused-argument
    # [FUNCTION DOC]
    # HELPER: hook callback invoked after the session is created. Takes the session and the
    # coordinator but ignores them; only increments the after-create-session counter.
    # Returns nothing.
    self.call_counter['after_create_session'] += 1

  def before_run(self, run_context):
    # [FUNCTION DOC]
    # HELPER: hook callback invoked before each run. Increments the before-run counter, stores
    # the run context for later inspection and returns the configured request (null when no
    # request was set), which the session merges into the run.
    self.call_counter['before_run'] += 1
    self.last_run_context = run_context
    return self.request

  def after_run(self, run_context, run_values):
    # [FUNCTION DOC]
    # HELPER: hook callback invoked after each run. Increments the after-run counter, stores the
    # run values, and if the stop flag was set asks the run context to request a stop.
    # Returns nothing.
    self.call_counter['after_run'] += 1
    self.last_run_values = run_values
    if self.should_stop:
      run_context.request_stop()

  def end(self, session):
    # [FUNCTION DOC]
    # HELPER: hook callback invoked when the session ends. Takes the session (unused) and only
    # increments the end counter. Returns nothing.
    self.call_counter['end'] += 1


class MonitoredTrainingSessionTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the MonitoredTrainingSession factory: checkpoint save/restore by steps and by
  # seconds, restoring from a version-1 saved model, summary saving by steps and by seconds,
  # custom chief-only hooks replacing the default saver, and saving or skipping the graph
  # definition. Tests use per-test directories under the temporary test directory.
  """Tests MonitoredTrainingSession."""

  def test_saving_restoring_checkpoint(self):
    # [FUNCTION DOC]
    # Test: a chief session with a checkpoint directory saves a checkpoint when closed.
    # Runs a global-step increment twice, then opens a second session on the same directory and
    # asserts the global step was restored to 2.
    logdir = _test_dir(self.get_temp_dir(), 'test_saving_restoring_checkpoint')
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      with monitored_session.MonitoredTrainingSession(
          is_chief=True, checkpoint_dir=logdir) as session:
        self.assertEqual(0, session.run(gstep))
        self.assertEqual(1, session.run(do_step))
        self.assertEqual(2, session.run(do_step))
      # A restart will find the checkpoint and recover automatically.
      with monitored_session.MonitoredTrainingSession(
          is_chief=True, checkpoint_dir=logdir) as session:
        self.assertEqual(2, session.run(gstep))

  def test_save_checkpoint_steps(self):
    # [FUNCTION DOC]
    # Test: saving a checkpoint every 100 steps. Runs 100 increments, then restarts a chief
    # session on the directory and asserts the global step is restored as 100.
    logdir = _test_dir(self.get_temp_dir(), 'test_save_checkpoint_steps')
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      new_gstep = state_ops.assign_add(gstep, 1)
      with monitored_session.MonitoredTrainingSession(
          is_chief=True,
          checkpoint_dir=logdir,
          save_checkpoint_steps=100,
          log_step_count_steps=10) as session:
        for _ in range(100):
          session.run(new_gstep)
      # A restart will find the checkpoint and recover automatically.
      with monitored_session.MonitoredTrainingSession(
          is_chief=True, checkpoint_dir=logdir) as session:
        self.assertEqual(100, session.run(gstep))

  def test_save_checkpoint_secs(self):
    # [FUNCTION DOC]
    # Test: saving a checkpoint on a time interval of 0.1 seconds. Runs one step, sleeps 0.2
    # seconds, runs ten more, then restarts and asserts the restored global step is 11.
    logdir = _test_dir(self.get_temp_dir(), 'test_save_checkpoint_secs')
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      new_gstep = state_ops.assign_add(gstep, 1)
      with monitored_session.MonitoredTrainingSession(
          is_chief=True,
          checkpoint_dir=logdir,
          save_checkpoint_secs=0.1,
          log_step_count_steps=10) as session:
        session.run(new_gstep)
        time.sleep(0.2)
        for _ in range(10):
          session.run(new_gstep)
      # A restart will find the checkpoint and recover automatically.
      with monitored_session.MonitoredTrainingSession(
          is_chief=True, checkpoint_dir=logdir) as session:
        self.assertEqual(11, session.run(gstep))

  def test_save_restore_checkpoint_v1_saved_model(self):
    # [FUNCTION DOC]
    # Test: a monitored training session works with a variable loaded from a version-1 saved
    # model. Writes such a model with one variable of value five, loads it, increments the
    # variable inside a session using the model directory as checkpoint directory (asserts six,
    # then seven), then reopens the session and asserts the variable continues from the
    # saved value (eight).

    def _write_v1_simple_saved_model(export_dir):
      # [FUNCTION DOC]
      # HELPER: writes the saved model used by the enclosing test. Input: export_dir, directory to
      # write to. Builds a version-1 saved model with one resource variable of value five in a
      # fresh graph, tagged with a fixed tag, and saves it. Returns nothing; writes files.
      # Create v1 Saved Model with single variable `w0` with value 5.0.
      builder = saved_model_builder.SavedModelBuilder(export_dir)
      with ops.Graph().as_default():
        _ = resource_variable_ops.ResourceVariable(5.0)
        with self.cached_session() as session:
          session.run(variables.global_variables_initializer())
          builder.add_meta_graph_and_variables(session, ['foo'])
      builder.save()

    test_dir = _test_dir(self.get_temp_dir(), 'saved_model')
    _write_v1_simple_saved_model(test_dir)

    with ops.Graph().as_default():
      # Load saved model with `load_v1_in_v2`.
      model = saved_model_load.load(test_dir)
      w0 = model.variables[0]
      # Define operation that increments `w0`.
      w_add = w0.assign_add(1.)
      gstep = training_util.get_or_create_global_step()
      new_gstep = state_ops.assign_add(gstep, 1)

      with monitored_session.MonitoredTrainingSession(
          checkpoint_dir=test_dir) as session:
        w1 = session.run(w_add)
        self.assertEqual(w1, 6.)
        session.run(new_gstep)
        w2 = session.run(w_add)
        self.assertEqual(w2, 7.)

      # Stop and resume training.
      with monitored_session.MonitoredTrainingSession(
          checkpoint_dir=test_dir) as session:
        # `w0` saves its value of 7.
        w3 = session.run(w_add)
        self.assertEqual(w3, 8.)

  def test_summaries_steps(self):
    # [FUNCTION DOC]
    # Test: summary saving every 100 steps. Adds a scalar summary, runs 101 steps, then reads
    # the newest event file and asserts the custom summary tag and the global steps per second
    # tag are both present.
    logdir = _test_dir(self.get_temp_dir(), 'test_summaries_steps')
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      new_gstep = state_ops.assign_add(gstep, 1)
      summary.scalar('my_summary_tag', new_gstep * 2)
      with monitored_session.MonitoredTrainingSession(
          is_chief=True,
          checkpoint_dir=logdir,
          save_summaries_steps=100,
          log_step_count_steps=10) as session:
        for _ in range(101):
          session.run(new_gstep)
    summaries = latest_summaries(logdir)
    tags = [s.summary.value[0].tag for s in summaries]
    self.assertIn('my_summary_tag', tags)
    self.assertIn('global_step/sec', tags)

  def test_summaries_secs(self):
    # [FUNCTION DOC]
    # Test: summary saving on a time interval (step-based saving turned off, 0.1 seconds).
    # Runs a step, sleeps 0.2 seconds, runs 101 more and asserts the custom summary tag and the
    # global steps per second tag appear in the written events.
    logdir = _test_dir(self.get_temp_dir(), 'test_summaries_secs')
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      new_gstep = state_ops.assign_add(gstep, 1)
      summary.scalar('my_summary_tag', new_gstep * 2)
      with monitored_session.MonitoredTrainingSession(
          is_chief=True,
          checkpoint_dir=logdir,
          save_summaries_steps=None,
          save_summaries_secs=0.1,
          log_step_count_steps=10) as session:
        session.run(new_gstep)
        time.sleep(0.2)
        for _ in range(101):
          session.run(new_gstep)
    summaries = latest_summaries(logdir)
    tags = [s.summary.value[0].tag for s in summaries]
    self.assertIn('my_summary_tag', tags)
    self.assertIn('global_step/sec', tags)

  def test_custom_saving(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: uses the same directory name as the save/restore test, so it relies on the
    # directory helper clearing old files rather than having its own directory.
    # Test: a chief-only hook is passed and checkpoint saving by seconds is set to zero.
    # Runs two steps and asserts the custom hook's begin callback ran once. A second session
    # on the same directory must not find a checkpoint (step zero), showing that the default
    # saver was not used in that configuration.
    logdir = _test_dir(self.get_temp_dir(), 'test_saving_restoring_checkpoint')
    fake_hook = FakeHook()
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      with monitored_session.MonitoredTrainingSession(
          is_chief=True,
          checkpoint_dir=logdir,
          chief_only_hooks=[fake_hook],
          save_checkpoint_secs=0) as session:
        self.assertEqual(0, session.run(gstep))
        self.assertEqual(1, session.run(do_step))
        self.assertEqual(2, session.run(do_step))

      # Check whether custom hook called or not
      self.assertEqual(1, fake_hook.call_counter['begin'])
      # A restart will not find the checkpoint, since we didn't save.
      with monitored_session.MonitoredTrainingSession(
          is_chief=True, checkpoint_dir=logdir) as session:
        self.assertEqual(0, session.run(gstep))

  def test_save_graph_def(self):
    # [FUNCTION DOC]
    # Test: with graph saving enabled and a checkpoint every step, the graph text file exists
    # and one meta-graph file exists on session start and two after one step.
    logdir = _test_dir(self.get_temp_dir(), 'test_save_graph_def')
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      new_gstep = state_ops.assign_add(gstep, 1)
      with monitored_session.MonitoredTrainingSession(
          is_chief=True,
          checkpoint_dir=logdir,
          save_checkpoint_steps=1,
          save_graph_def=True) as session:
        self.assertIn('graph.pbtxt', os.listdir(logdir))
        self.assertLen(glob.glob(os.path.join(logdir, '*.meta')), 1)
        session.run(new_gstep)
        self.assertLen(glob.glob(os.path.join(logdir, '*.meta')), 2)

  def test_save_graph_def_false(self):
    # [FUNCTION DOC]
    # Test: with graph saving disabled and a checkpoint every step, no graph text file exists
    # and no meta-graph files are written, before or after a step.
    logdir = _test_dir(self.get_temp_dir(), 'test_save_graph_def')
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      new_gstep = state_ops.assign_add(gstep, 1)
      with monitored_session.MonitoredTrainingSession(
          is_chief=True,
          checkpoint_dir=logdir,
          save_checkpoint_steps=1,
          save_graph_def=False) as session:
        self.assertNotIn('graph.pbtxt', os.listdir(logdir))
        self.assertEmpty(glob.glob(os.path.join(logdir, '*.meta')))
        session.run(new_gstep)
        self.assertEmpty(glob.glob(os.path.join(logdir, '*.meta')))


class MockExtended:
  # [CLASS DOC]
  # HELPER: minimal stand-in for a distribution strategy's extended object, used by the
  # distribute-coordinator tests. Holds the between-graph flag, should-init flag and the
  # should-checkpoint and should-save-summary decisions as plain attributes.

  def __init__(self, between_graph, should_init, should_checkpoint,
               should_save_summary):
    # [FUNCTION DOC]
    # HELPER: stores the four given values (between-graph flag, should-init flag,
    # should-checkpoint decision, should-save-summary decision) as attributes. Returns nothing.
    self.experimental_between_graph = between_graph
    self.experimental_should_init = should_init
    self.should_checkpoint = should_checkpoint
    self.should_save_summary = should_save_summary


class MockStrategy:
  # [CLASS DOC]
  # HELPER: minimal stand-in for a distribution strategy used by the distribute-coordinator
  # tests. Exposes only an extended attribute holding a mock extended object configured from
  # the constructor arguments.

  def __init__(self,
               between_graph=False,
               should_init=True,
               should_checkpoint=None,
               should_save_summary=None):
    # [FUNCTION DOC]
    # HELPER: builds the mock extended object. Optional inputs: between_graph (default false),
    # should_init (default true), should_checkpoint and should_save_summary (default null).
    # Returns nothing.
    self.extended = MockExtended(between_graph, should_init, should_checkpoint,
                                 should_save_summary)


class MonitoredTrainingSessionWithDistributeCoordinatorTest(test.TestCase):
  # [CLASS DOC]
  # Test case verifying that the distribute coordinator's worker context decides whether
  # MonitoredTrainingSession adds summary and checkpoint hooks. Uses mock strategies and worker
  # contexts; summaries are checked via event files and checkpoints via the latest checkpoint.
  """Test distribute coordinator controls summary saving and checkpointing."""

  def test_summary_hook_enabled(self):
    # [FUNCTION DOC]
    # Test: when the strategy says summaries should be saved, running 101 steps inside the
    # worker context writes events containing the custom tag and the steps per second tag.
    context = distribute_coordinator._WorkerContext(
        MockStrategy(should_save_summary=True), None, None, None)

    logdir = _test_dir(self.get_temp_dir(), 'test_summaries_enabled')
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      new_gstep = state_ops.assign_add(gstep, 1)
      summary.scalar('my_summary_tag', new_gstep * 2)
      with context, monitored_session.MonitoredTrainingSession(
          checkpoint_dir=logdir,
          save_summaries_steps=100,
          log_step_count_steps=10) as session:
        for _ in range(101):
          session.run(new_gstep)

    summaries = latest_summaries(logdir)
    tags = [s.summary.value[0].tag for s in summaries]
    self.assertIn('my_summary_tag', tags)
    self.assertIn('global_step/sec', tags)

  def test_summary_hook_disabled(self):
    # [FUNCTION DOC]
    # Test: when the strategy says summaries should not be saved, running 101 steps writes no
    # summary events at all.
    context = distribute_coordinator._WorkerContext(
        MockStrategy(should_save_summary=False), None, None, None)

    logdir = _test_dir(self.get_temp_dir(), 'test_summaries_disabled')
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      new_gstep = state_ops.assign_add(gstep, 1)
      summary.scalar('my_summary_tag', new_gstep * 2)
      with context, monitored_session.MonitoredTrainingSession(
          checkpoint_dir=logdir,
          save_summaries_steps=100,
          log_step_count_steps=10) as session:
        for _ in range(101):
          session.run(new_gstep)

    # No summary is saved.
    summaries = latest_summaries(logdir)
    self.assertEqual(len(summaries), 0)

  def test_checkpoint_hook_enabled(self):
    # [FUNCTION DOC]
    # Test: when the strategy says checkpoints should be saved, a checkpoint is written after
    # 100 steps and a restarted chief session restores global step 100.
    context = distribute_coordinator._WorkerContext(
        MockStrategy(should_checkpoint=True), None, None, None)

    logdir = _test_dir(self.get_temp_dir(), 'test_save_checkpoint_enabled')
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      new_gstep = state_ops.assign_add(gstep, 1)
      with context, monitored_session.MonitoredTrainingSession(
          checkpoint_dir=logdir,
          save_checkpoint_steps=100,
          log_step_count_steps=10) as session:
        for _ in range(100):
          session.run(new_gstep)

      # A restart will find the checkpoint and recover automatically.
      with monitored_session.MonitoredTrainingSession(
          is_chief=True, checkpoint_dir=logdir) as session:
        self.assertEqual(100, session.run(gstep))

  def test_checkpoint_hook_disabled(self):
    # [FUNCTION DOC]
    # Test: when the strategy says checkpoints should not be saved, no checkpoint exists in the
    # directory after 100 steps (the latest checkpoint is null).
    context = distribute_coordinator._WorkerContext(
        MockStrategy(should_checkpoint=False), None, None, None)

    logdir = _test_dir(self.get_temp_dir(), 'test_save_checkpoint_disabled')
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      new_gstep = state_ops.assign_add(gstep, 1)
      with context, monitored_session.MonitoredTrainingSession(
          checkpoint_dir=logdir,
          save_checkpoint_steps=100,
          log_step_count_steps=10) as session:
        for _ in range(100):
          session.run(new_gstep)

    # No checkpoint is saved.
    checkpoint = checkpoint_management.latest_checkpoint(logdir)
    self.assertIsNone(checkpoint)

  def test_checkpoint_hook_enable_on_non_chief_with_collective_ops(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the directory name says disabled although this test covers the enabled-on-
    # non-chief case; it is the same name used by the disabled-hook test.
    # Test: a collective all-reduce strategy marked as non-chief, with worker id one, still
    # saves checkpoints, but into a temporary per-worker sub-directory. Asserts nothing is in
    # the main directory and a checkpoint exists in the temporary worker directory.
    strategy = collective_all_reduce_strategy.CollectiveAllReduceStrategy()
    strategy.extended._is_chief = False

    context = distribute_coordinator._WorkerContext(strategy, None, 'worker', 1)

    logdir = _test_dir(self.get_temp_dir(), 'test_save_checkpoint_disabled')
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      new_gstep = state_ops.assign_add(gstep, 1)
      with context, monitored_session.MonitoredTrainingSession(
          checkpoint_dir=logdir,
          save_checkpoint_steps=100,
          log_step_count_steps=10) as session:
        for _ in range(100):
          session.run(new_gstep)

    # No checkpoint is saved.
    checkpoint = checkpoint_management.latest_checkpoint(logdir)
    self.assertIsNone(checkpoint)

    # But saved to a temporary directory.
    checkpoint = checkpoint_management.latest_checkpoint(
        os.path.join(logdir, 'tmp_worker_1'))
    self.assertIsNotNone(checkpoint)


class StopAtNSession(monitored_session._WrappedSession):
  # [CLASS DOC]
  # HELPER: wrapped-session test double whose stop check returns true once its countdown of N
  # calls reaches zero. Used to prove that should-stop consults the overridable check method.
  """A wrapped session that stops at the N-th call to _check_stop."""

  def __init__(self, sess, n):
    # [FUNCTION DOC]
    # HELPER: wraps the session given as sess and stores n, the number of stop checks that
    # still return false. Returns nothing.
    super(StopAtNSession, self).__init__(sess)
    self._count = n

  def _check_stop(self):
    # [FUNCTION DOC]
    # HELPER: overrides the stop check. Returns true if the countdown is zero; otherwise
    # decrements it and returns false. Mutates the remaining count.
    if self._count == 0:
      return True
    self._count -= 1
    return False


class WrappedSessionTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the internal wrapped session: property forwarding, should-stop behavior,
  # closing and running. Mostly run in legacy graph mode inside a cached test session.
  """_WrappedSession tests."""

  @test_util.run_deprecated_v1
  def test_properties(self):
    # [FUNCTION DOC]
    # Test: the wrapper exposes the wrapped session's graph and session string unchanged.
    with self.cached_session() as sess:
      constant_op.constant(0.0)
      wrapped_sess = monitored_session._WrappedSession(sess)
      self.assertEqual(sess.graph, wrapped_sess.graph)
      self.assertEqual(sess.sess_str, wrapped_sess.sess_str)

  @test_util.run_deprecated_v1
  def test_should_stop_on_close(self):
    # [FUNCTION DOC]
    # Test: should-stop is false on a fresh wrapper and becomes true after close.
    with self.cached_session() as sess:
      wrapped_sess = monitored_session._WrappedSession(sess)
      self.assertFalse(wrapped_sess.should_stop())
      wrapped_sess.close()
      self.assertTrue(wrapped_sess.should_stop())

  @test_util.run_deprecated_v1
  def test_should_stop_uses_check_stop(self):
    # [FUNCTION DOC]
    # Test: should-stop uses the overridable stop check. With a stop-after-three wrapper the
    # first three calls return false and the fourth returns true.
    with self.cached_session() as sess:
      wrapped_sess = StopAtNSession(sess, 3)
      self.assertFalse(wrapped_sess.should_stop())
      self.assertFalse(wrapped_sess.should_stop())
      self.assertFalse(wrapped_sess.should_stop())
      self.assertTrue(wrapped_sess.should_stop())

  @test_util.run_deprecated_v1
  def test_should_stop_delegates_to_wrapped_session(self):
    # [FUNCTION DOC]
    # Test: should-stop delegates to the inner wrapped session. A wrapper around a stop-after-four
    # session returns false four times and then true.
    with self.cached_session() as sess:
      wrapped_sess0 = StopAtNSession(sess, 4)
      wrapped_sess1 = monitored_session._WrappedSession(wrapped_sess0)
      self.assertFalse(wrapped_sess1.should_stop())
      self.assertFalse(wrapped_sess1.should_stop())
      self.assertFalse(wrapped_sess1.should_stop())
      self.assertFalse(wrapped_sess1.should_stop())
      self.assertTrue(wrapped_sess1.should_stop())

  @test_util.run_deprecated_v1
  def test_close_twice(self):
    # [FUNCTION DOC]
    # Test: closing a wrapper twice is allowed; should-stop is true after each close.
    with self.cached_session() as sess:
      wrapped_sess = monitored_session._WrappedSession(sess)
      wrapped_sess.close()
      self.assertTrue(wrapped_sess.should_stop())
      wrapped_sess.close()
      self.assertTrue(wrapped_sess.should_stop())

  @test_util.run_deprecated_v1
  def test_run(self):
    # [FUNCTION DOC]
    # Test: running a fetch with a feed dictionary through the wrapper returns the same value
    # as running directly on the underlying session (42 directly, 51 through the wrapper).
    with self.cached_session() as sess:
      c = constant_op.constant(0)
      v = array_ops.identity(c)
      self.assertEqual(42, sess.run(v, feed_dict={c: 42}))
      wrapped_sess = monitored_session._WrappedSession(sess)
      self.assertEqual(51, wrapped_sess.run(v, feed_dict={c: 51}))


def busy_wait_for_coord_stop(coord):
  # [FUNCTION DOC]
  # HELPER: thread target used by coordinated-session tests. Input: coord, a coordinator.
  # Loops with a one-millisecond sleep until the coordinator says to stop. Returns nothing.
  while not coord.should_stop():
    time.sleep(0.001)


class CoordinatedSessionTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the internal coordinated session, which ties a session to a coordinator:
  # property forwarding, running, stop propagation in both directions, thread shutdown on close
  # and exception trace propagation.
  """_CoordinatedSession tests."""

  @test_util.run_deprecated_v1
  def test_properties(self):
    # [FUNCTION DOC]
    # Test: the coordinated session exposes the wrapped session's graph and session string.
    with self.cached_session() as sess:
      constant_op.constant(0.0)
      coord = coordinator.Coordinator()
      coord_sess = monitored_session._CoordinatedSession(sess, coord)
      self.assertEqual(sess.graph, coord_sess.graph)
      self.assertEqual(sess.sess_str, coord_sess.sess_str)

  @test_util.run_deprecated_v1
  def test_run(self):
    # [FUNCTION DOC]
    # Test: running a fetch with a feed dictionary through the coordinated session returns the
    # fed value (42).
    with self.cached_session() as sess:
      c = constant_op.constant(0)
      v = array_ops.identity(c)
      coord = coordinator.Coordinator()
      coord_sess = monitored_session._CoordinatedSession(sess, coord)
      self.assertEqual(42, coord_sess.run(v, feed_dict={c: 42}))

  @test_util.run_deprecated_v1
  def test_should_stop_on_close(self):
    # [FUNCTION DOC]
    # Test: should-stop is false at first and true after the coordinated session is closed.
    with self.cached_session() as sess:
      coord = coordinator.Coordinator()
      coord_sess = monitored_session._CoordinatedSession(sess, coord)
      self.assertFalse(coord_sess.should_stop())
      coord_sess.close()
      self.assertTrue(coord_sess.should_stop())

  @test_util.run_deprecated_v1
  def test_should_stop_on_coord_stop(self):
    # [FUNCTION DOC]
    # Test: should-stop turns true when the coordinator itself is asked to stop.
    with self.cached_session() as sess:
      coord = coordinator.Coordinator()
      coord_sess = monitored_session._CoordinatedSession(sess, coord)
      self.assertFalse(coord_sess.should_stop())
      coord.request_stop()
      self.assertTrue(coord_sess.should_stop())

  @test_util.run_deprecated_v1
  def test_dont_request_stop_on_exception_in_main_thread(self):
    # [FUNCTION DOC]
    # Test: an error raised in the caller's thread by an invalid fetch (a null fetch, type
    # error) does not request a stop on the coordinator. Normal runs succeed before the error
    # and both the coordinator and session still report not stopped afterwards.
    with self.cached_session() as sess:
      c = constant_op.constant(0)
      v = array_ops.identity(c)
      coord = coordinator.Coordinator()
      coord_sess = monitored_session._CoordinatedSession(sess, coord)
      self.assertFalse(coord_sess.should_stop())
      self.assertEqual(0, coord_sess.run(c))
      self.assertEqual(1, coord_sess.run(v, feed_dict={c: 1}))
      with self.assertRaisesRegex(TypeError, 'None has invalid type'):
        coord_sess.run([None], feed_dict={c: 2})
      self.assertFalse(coord.should_stop())
      self.assertFalse(coord_sess.should_stop())

  @test_util.run_deprecated_v1
  def test_stop_threads_on_close_after_exception(self):
    # [FUNCTION DOC]
    # Test: three busy-wait threads registered with the coordinator are alive during normal
    # runs and after a type error from an invalid fetch. Closing the session stops them all,
    # and both coordinator and session report stopped.
    with self.cached_session() as sess:
      c = constant_op.constant(0)
      v = array_ops.identity(c)
      coord = coordinator.Coordinator()
      threads = [
          threading.Thread(
              target=busy_wait_for_coord_stop, args=(coord,)) for _ in range(3)
      ]
      for t in threads:
        coord.register_thread(t)
        t.start()
      coord_sess = monitored_session._CoordinatedSession(sess, coord)
      self.assertFalse(coord_sess.should_stop())
      for t in threads:
        self.assertTrue(t.is_alive())
      self.assertEqual(0, coord_sess.run(c))
      for t in threads:
        self.assertTrue(t.is_alive())
      self.assertEqual(1, coord_sess.run(v, feed_dict={c: 1}))
      for t in threads:
        self.assertTrue(t.is_alive())
      with self.assertRaisesRegex(TypeError, 'None has invalid type'):
        coord_sess.run([None], feed_dict={c: 2})
      coord_sess.close()
      for t in threads:
        self.assertFalse(t.is_alive())
      self.assertTrue(coord.should_stop())
      self.assertTrue(coord_sess.should_stop())

  def test_stop_threads_on_close(self):
    # [FUNCTION DOC]
    # Test: closing a coordinated session with three registered busy-wait threads stops the
    # coordinator and joins all threads; asserts no thread is alive and should-stop is true.
    with self.cached_session() as sess:
      coord = coordinator.Coordinator()
      threads = [
          threading.Thread(
              target=busy_wait_for_coord_stop, args=(coord,)) for _ in range(3)
      ]
      for t in threads:
        coord.register_thread(t)
        t.start()
      coord_sess = monitored_session._CoordinatedSession(sess, coord)
      coord_sess.close()
      for t in threads:
        self.assertFalse(t.is_alive())
      self.assertTrue(coord.should_stop())
      self.assertTrue(coord_sess.should_stop())

  @test_util.run_deprecated_v1
  def test_propagates_exception_trace(self):
    # [FUNCTION DOC]
    # Test: the original exception trace is preserved. A failing assert op is run through a
    # coordinated session whose coordinator does not treat any exception type as a clean stop.
    # Catches the invalid-argument error, extracts the file where it was first raised and
    # asserts it is one of the two known session files, and specifically the session file (not
    # the monitored session file), proving the trace was not replaced.
    assertion = control_flow_assert.Assert(False, ['This should fail.'])
    with self.cached_session() as sess:
      coord = coordinator.Coordinator(clean_stop_exception_types=())
      coord_sess = monitored_session._CoordinatedSession(sess, coord)
      try:
        coord_sess.run([assertion])
        self.fail('No exception was raised by assertion.')
      except errors_impl.InvalidArgumentError:
        # Extract the name of the file where the exception was first raised.
        _, _, exc_traceback = sys.exc_info()
        tb = traceback.extract_tb(exc_traceback)
        exc_source_file = tb[-1][0]
        exc_source_basename = os.path.basename(exc_source_file)
        # If it's monitored_session.py then the original stack trace was not
        # correctly propagated.
        self.assertIn(
            exc_source_basename, ['session.py', 'monitored_session.py'],
            'The exception was raised from an unrecognized file. This unit '
            'test probably needs to be updated. Traceback:\n%s\n' % tb)
        self.assertEqual(
            exc_source_basename, 'session.py',
            'Original stack trace was not propagated by MonitoredSession. '
            'Traceback:\n%s' % tb)


class AbortAtNSession:
  # [CLASS DOC]
  # HELPER: mock session that delegates runs to a real session but raises an aborted error on
  # the run call after N successful calls. Used to simulate preemption for recovery tests.
  """A mock session that aborts at the N-th run call."""

  def __init__(self, sess, n):
    # [FUNCTION DOC]
    # HELPER: stores the real session and n, the number of successful runs before aborting.
    self._sess = sess
    self._count = n

  def close(self):
    # [FUNCTION DOC]
    # HELPER: close does nothing, so the shared real session is not closed. Returns nothing.
    pass

  def run(self, *args, **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the aborted error is built with the message text in the first positional slot
    # and nothing in the message slot (other tests in this file pass nothing, nothing, then the
    # message), so the error carries a null message. It is still the right type for recovery.
    # HELPER: forwards positional and keyword arguments to the real session while the
    # countdown is above zero (decrementing it) and returns its result. When the countdown is
    # zero raises an aborted error instead.
    if self._count == 0:
      raise errors_impl.AbortedError('Aborted at N', None, None)
    self._count -= 1
    return self._sess.run(*args, **kwargs)


class StopCoordinatorWithException(session_run_hook.SessionRunHook):
  # [CLASS DOC]
  # HELPER: hook that makes the coordinator stop with an exception after N run calls. It starts
  # a side thread registered with the coordinator that waits until the call budget hits zero,
  # then requests a stop with the configured exception (an aborted error by default).
  """With this hook Coordinator throws an exception after N-runs."""

  def __init__(self, calls_before_stopping, exception_to_raise=None):
    # [FUNCTION DOC]
    # HELPER: inputs: calls_before_stopping, number of run calls before the stop; and
    # exception_to_raise, optional error to report (defaults to an aborted error). Creates the
    # lock, an event signalled once the exception is stored, and a flag that the side thread
    # was not started yet.
    self._started_the_side_thread_already = False
    self._lock = threading.Lock()
    self._stored_exception_event = threading.Event()
    self._calls_before_stopping = calls_before_stopping
    self._exception_to_raise = (exception_to_raise or errors_impl.AbortedError(
        None, None, 'Aborted at N'))

  def _maybe_stop_with_exception(self, coord):
    # [FUNCTION DOC]
    # KNOWN ISSUE: busy loop that never sleeps; if the budget never reaches zero the thread
    # spins forever holding and releasing the lock until the process ends.
    # HELPER: thread body. Input: coord, the coordinator. Repeatedly takes the lock; when the
    # remaining call budget is zero raises the configured exception, catches it, passes it to the
    # coordinator's request-stop, sets the stored-exception event and exits the loop.
    while True:
      with self._lock:
        if self._calls_before_stopping == 0:
          try:
            raise self._exception_to_raise
          except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)
            self._stored_exception_event.set()
            break

  def after_create_session(self, session, coord):
    # [FUNCTION DOC]
    # HELPER: hook callback after session creation. Inputs: session (unused) and coord.
    # The first time only, starts a separate thread running the exception-raising loop and
    # registers it with the coordinator (which will join it); later calls do nothing.
    if self._started_the_side_thread_already:
      return

    separate_thread = threading.Thread(
        target=self._maybe_stop_with_exception, args=(coord,))

    coord.register_thread(separate_thread)
    separate_thread.start()
    self._started_the_side_thread_already = True
    # Coordinator will take care of joining `separate_thread`.

  def after_run(self, run_context, run_values):
    # [FUNCTION DOC]
    # HELPER: hook callback after each run. Decrements the call budget under the lock; if it just
    # reached zero waits until the side thread has stored its exception in the coordinator.
    # Returns nothing.
    stopping_now = False
    with self._lock:
      self._calls_before_stopping -= 1
      if self._calls_before_stopping == 0:
        stopping_now = True

    if stopping_now:
      self._stored_exception_event.wait()


class FailTrainingAfterCoordinatorStopped(StopCoordinatorWithException):
  # [CLASS DOC]
  # HELPER: variant of the coordinator-stopping hook where training itself fails after the
  # coordinator was stopped. Mimics a main thread that gets stuck and whose session is
  # discarded, by turning a stored aborted error into a cancelled error.
  """With this hook training encounters an exception after N-runs."""

  def __init__(self, calls_before_stopping):
    # [FUNCTION DOC]
    # HELPER: input calls_before_stopping is passed to the parent hook; also keeps a slot for
    # the coordinator, initially null.
    StopCoordinatorWithException.__init__(self, calls_before_stopping)
    self._coord = None

  def after_create_session(self, session, coord):
    # [FUNCTION DOC]
    # HELPER: remembers the coordinator, then runs the parent callback (which starts the side
    # thread) and returns its result.
    self._coord = coord
    return StopCoordinatorWithException.after_create_session(
        self, session, coord)

  def after_run(self, run_context, run_values):
    # [FUNCTION DOC]
    # HELPER: runs the parent after-run logic, then asks the coordinator to re-raise any
    # exception stored in it. If an aborted error is re-raised it is swallowed and replaced by
    # a cancelled error stating the session was garbage collected, simulating a stuck session.
    StopCoordinatorWithException.after_run(self, run_context, run_values)
    try:
      # After a `run`, an exception could have been stored inside the
      # coordinator.
      self._coord.raise_requested_exception()
    except errors_impl.AbortedError:
      # In real world, the main thread may or may not know about the exception
      # that stopped the coordinator. Because the coordinator has stopped, the
      # main thread could have gotten stuck as well (for example, the
      # coordinator was supposed to execute `FIFOQueue.enqueue` while the main
      # thread is executing a blocking `FIFOQueue.dequeue`). After it got stuck,
      # the session is going to get garbage collected after some time with:
      raise errors_impl.CancelledError(None, None,
                                       'Session got garbage-collected.')


class CountingSessionCreator:
  # [CLASS DOC]
  # HELPER: session creator for tests that always hands back the same existing session and
  # counts how many times a session was requested. It disables closing on that session so it
  # can be reused after recovery.
  """A creator that counts the number of created sessions."""

  def __init__(self, session):
    # [FUNCTION DOC]
    # HELPER: input session is the single real session. Replaces its close method with one that
    # does nothing (mutates the given session) and starts the creation counter at zero.
    self._initial_session = session
    # We only have one session per test case. We can't re-create it, thus
    # it shouldn't be closed.
    self._initial_session.close = lambda *args: None
    self._create_session_calls = 0

  @property
  def number_of_sessions_created(self):
    # [FUNCTION DOC]
    # HELPER: property returning how many times a session was created through this creator.
    return self._create_session_calls

  def create_session(self):
    # [FUNCTION DOC]
    # HELPER: increments the creation counter and returns the same stored session each time.
    self._create_session_calls += 1
    return self._initial_session


class RecoverableSessionTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the internal recoverable session and for MonitoredSession recovery logic.
  # Covers simple properties, recovery after aborted runs, recovery or non-recovery when the
  # coordinator stops with various errors, and the same through step functions and through the
  # raw underlying session.
  """_RecoverableSession tests."""

  class _SessionReturner:
    # [CLASS DOC]
    # HELPER: tiny session creator that returns a preexisting session when asked to create one.

    def __init__(self, sess):
      # [FUNCTION DOC]
      # HELPER: stores the given session. Returns nothing.
      self._sess = sess

    def create_session(self):
      # [FUNCTION DOC]
      # HELPER: returns the stored session.
      return self._sess

  @test_util.run_deprecated_v1
  def test_properties(self):
    # [FUNCTION DOC]
    # Test: a recoverable session exposes the wrapped session's graph and session string.
    with self.cached_session() as sess:
      constant_op.constant(0.0)
      recoverable_sess = monitored_session._RecoverableSession(
          self._SessionReturner(sess))
      self.assertEqual(sess.graph, recoverable_sess.graph)
      self.assertEqual(sess.sess_str, recoverable_sess.sess_str)

  @test_util.run_deprecated_v1
  def test_run(self):
    # [FUNCTION DOC]
    # Test: running a fetch with a feed dictionary through a recoverable session returns the
    # fed value (51).
    with self.cached_session() as sess:
      c = constant_op.constant(0)
      v = array_ops.identity(c)
      recoverable_sess = monitored_session._RecoverableSession(
          self._SessionReturner(sess))
      self.assertEqual(51, recoverable_sess.run(v, feed_dict={c: 51}))

  @test_util.run_deprecated_v1
  def test_recovery(self):
    # [FUNCTION DOC]
    # Test: the recoverable session replaces an aborted session by asking the creator for a new
    # one. Three sessions that abort after one, two and three runs are queued. Asserts the
    # queue shrinks only when recovery happens, that all runs return the fed values, and that
    # once the queue is empty a further abort surfaces the creator's index error about popping
    # from an empty list.
    with self.cached_session() as sess:

      class StackSessionCreator:
        # [CLASS DOC]
        # HELPER: session creator that holds a list of pre-made aborting sessions and hands them
        # out one at a time.

        def __init__(self, sess):
          # [FUNCTION DOC]
          # HELPER: input sess is the real session. Builds three abort-after-N sessions, with N
          # one, two and three, as the list of sessions to hand out.
          self.sessions_to_use = [
              AbortAtNSession(sess, x + 1) for x in range(3)
          ]

        def create_session(self):
          # [FUNCTION DOC]
          # HELPER: removes and returns the first remaining session; raises an index error when the
          # list is empty.
          return self.sessions_to_use.pop(0)

      c = constant_op.constant(0)
      v = array_ops.identity(c)
      session_creator = StackSessionCreator(sess)
      # List of 3 sessions to use for recovery.  The first one aborts
      # after 1 run() call, the second after 2 run calls, the third
      # after 3 run calls.
      self.assertEqual(3, len(session_creator.sessions_to_use))
      # Make the recoverable session uses these 3 sessions in sequence by
      # passing a factory that pops from the session_to_use list.
      recoverable_sess = monitored_session._RecoverableSession(session_creator)
      self.assertEqual(
          2, len(session_creator.sessions_to_use))  # One session popped.
      # Using first session.
      self.assertEqual(51, recoverable_sess.run(v, feed_dict={c: 51}))
      self.assertEqual(
          2, len(session_creator.sessions_to_use))  # Still 2 sessions available
      # This will fail and recover by picking up the second session.
      self.assertEqual(42, recoverable_sess.run(v, feed_dict={c: 42}))
      self.assertEqual(
          1, len(session_creator.sessions_to_use))  # Still 1 session available
      self.assertEqual(33, recoverable_sess.run(v, feed_dict={c: 33}))
      self.assertEqual(
          1, len(session_creator.sessions_to_use))  # Still 1 session available
      # This will fail and recover by picking up the last session.
      self.assertEqual(24, recoverable_sess.run(v, feed_dict={c: 24}))
      self.assertEqual(
          0, len(session_creator.sessions_to_use))  # All sessions used.
      self.assertEqual(11, recoverable_sess.run(v, feed_dict={c: 11}))
      self.assertEqual(0, recoverable_sess.run(v, feed_dict={c: 0}))
      # This will fail and throw a real error as the pop() will fail.
      with self.assertRaisesRegex(IndexError, 'pop from empty list'):
        recoverable_sess.run(v, feed_dict={c: -12})

  @test_util.run_deprecated_v1
  def test_recovery_from_coordinator_exception(self):
    # [FUNCTION DOC]
    # Test: when the coordinator stops with an aborted error after two calls, the monitored
    # session silently recreates the underlying session. Asserts one session created at first,
    # correct values from the first two runs, not stopped, and two sessions created in total.
    with self.cached_session() as test_session:
      session_creator = CountingSessionCreator(test_session)
      session = monitored_session.MonitoredSession(
          session_creator,
          [StopCoordinatorWithException(calls_before_stopping=2)])

      self.assertEqual(1, session_creator.number_of_sessions_created)
      self.assertFalse(session.should_stop())

      c = constant_op.constant(0)
      v = array_ops.identity(c)

      # The coordinator will not abort during this call, since it's the call
      # number 0.
      self.assertEqual(51, session.run(v, feed_dict={c: 51}))
      self.assertFalse(session.should_stop())
      # The coordinator will abort during the next call, since it's the call
      # number 1.
      self.assertEqual(42, session.run(v, feed_dict={c: 42}))
      # Even though the coordinator was asked to stop, the underlying session is
      # recreated and is to be continued.
      self.assertFalse(session.should_stop())
      self.assertEqual(2, session_creator.number_of_sessions_created)

  @test_util.run_deprecated_v1
  def test_recovery_from_non_preemption_in_coordinator(self):
    # [FUNCTION DOC]
    # Test: when the coordinator stops with an unknown (non-recoverable) error, no new session is
    # created. Asserts the session reports should-stop after the second run, only one session
    # was created, and closing raises that unknown error.
    with self.cached_session() as test_session:
      session_creator = CountingSessionCreator(test_session)
      hook = StopCoordinatorWithException(
          calls_before_stopping=2,
          exception_to_raise=errors_impl.UnknownError(
              None, None, 'Some fatal exception inside the coordinator.'))
      session = monitored_session.MonitoredSession(session_creator, [hook])

      self.assertEqual(1, session_creator.number_of_sessions_created)
      self.assertFalse(session.should_stop())

      c = constant_op.constant(0)
      v = array_ops.identity(c)

      # The coordinator will not abort during this call, since it's the call
      # number 0.
      self.assertEqual(51, session.run(v, feed_dict={c: 51}))
      self.assertFalse(session.should_stop())
      # The coordinator will abort during the next call, since it's the call
      # number 1.
      self.assertEqual(42, session.run(v, feed_dict={c: 42}))
      # The coordinator was asked to stop due to non-redeemable error. Training
      # should stop and the session should not be recreated.
      self.assertTrue(session.should_stop())
      self.assertEqual(1, session_creator.number_of_sessions_created)
      with self.assertRaises(errors_impl.UnknownError):
        session.close()

  @test_util.run_deprecated_v1
  def test_recovery_from_session_getting_stuck(self):
    # [FUNCTION DOC]
    # Test: when training fails after the coordinator stopped (stuck session simulated by a
    # cancelled error), the monitored session still recreates the session. Asserts two sessions
    # created and the session not stopped.
    with self.cached_session() as test_session:
      session_creator = CountingSessionCreator(test_session)
      session = monitored_session.MonitoredSession(
          session_creator,
          [FailTrainingAfterCoordinatorStopped(calls_before_stopping=2)])

      self.assertEqual(1, session_creator.number_of_sessions_created)
      self.assertFalse(session.should_stop())

      c = constant_op.constant(0)
      v = array_ops.identity(c)

      # Training will not fail, since it's the call number 0.
      self.assertEqual(51, session.run(v, feed_dict={c: 51}))
      self.assertFalse(session.should_stop())
      # Training will fail during the next call, since it's the call
      # number 1.
      self.assertEqual(42, session.run(v, feed_dict={c: 42}))
      # Even though the coordinator stopped which and training failed, the
      # underlying session is recreated and training is to be continued.
      self.assertFalse(session.should_stop())
      self.assertEqual(2, session_creator.number_of_sessions_created)

  @test_util.run_deprecated_v1
  def test_step_fn_recovery_from_coordinator_exception_when_run_hooks(self):
    # [FUNCTION DOC]
    # Test: same recovery as the plain coordinator-exception test but runs work through step
    # functions that run with hooks. Asserts values 51 and 42, not stopped, and two sessions
    # created.
    with self.cached_session() as test_session:
      session_creator = CountingSessionCreator(test_session)
      session = monitored_session.MonitoredSession(
          session_creator,
          [StopCoordinatorWithException(calls_before_stopping=2)])

      self.assertEqual(1, session_creator.number_of_sessions_created)
      self.assertFalse(session.should_stop())

      c = constant_op.constant(0)
      v = array_ops.identity(c)

      def feed_step_fn(value):
        # [FUNCTION DOC]
        # HELPER: factory. Input value is the number to feed. Returns a step function that runs
        # the fetch with hooks using that fed value.
        def step_fn(step_context):
          # [FUNCTION DOC]
          # HELPER: step function. Input step_context. Returns the result of running the fetch
          # through the hooks with the captured value fed.
          return step_context.run_with_hooks(fetches=v, feed_dict={c: value})
        return step_fn

      # The coordinator will not abort during this call, since it's the call
      # number 0.
      self.assertEqual(51, session.run_step_fn(feed_step_fn(51)))
      self.assertFalse(session.should_stop())
      # The coordinator will abort during the next call, since it's the call
      # number 1.
      self.assertEqual(42, session.run_step_fn(feed_step_fn(42)))
      # Even though the coordinator was asked to stop, the underlying session is
      # recreated and is to be continued.
      self.assertFalse(session.should_stop())
      self.assertEqual(2, session_creator.number_of_sessions_created)

  @test_util.run_deprecated_v1
  def test_recovery_from_non_preemption_in_coordinator_when_run_hooks(self):
    # [FUNCTION DOC]
    # Test: the non-recoverable (unknown) coordinator error case, driven through step functions
    # that run with hooks. Asserts the session stops, only one session was created and closing
    # raises the unknown error.
    with self.cached_session() as test_session:
      session_creator = CountingSessionCreator(test_session)
      hook = StopCoordinatorWithException(
          calls_before_stopping=2,
          exception_to_raise=errors_impl.UnknownError(
              None, None, 'Some fatal exception inside the coordinator.'))
      session = monitored_session.MonitoredSession(session_creator, [hook])

      self.assertEqual(1, session_creator.number_of_sessions_created)
      self.assertFalse(session.should_stop())

      c = constant_op.constant(0)
      v = array_ops.identity(c)

      def feed_step_fn(value):
        # [FUNCTION DOC]
        # HELPER: factory returning a step function for the given value to feed.
        def step_fn(step_context):
          # [FUNCTION DOC]
          # HELPER: step function running the fetch with hooks and the captured fed value.
          return step_context.run_with_hooks(fetches=v, feed_dict={c: value})
        return step_fn

      # The coordinator will not abort during this call, since it's the call
      # number 0.
      self.assertEqual(51, session.run_step_fn(feed_step_fn(51)))
      self.assertFalse(session.should_stop())
      # The coordinator will abort during the next call, since it's the call
      # number 1.
      self.assertEqual(42, session.run_step_fn(feed_step_fn(42)))
      # The coordinator was asked to stop due to non-redeemable error. Training
      # should stop and the session should not be recreated.
      self.assertTrue(session.should_stop())
      self.assertEqual(1, session_creator.number_of_sessions_created)
      with self.assertRaises(errors_impl.UnknownError):
        session.close()

  @test_util.run_deprecated_v1
  def test_recovery_from_session_getting_stuck_when_run_hooks(self):
    # [FUNCTION DOC]
    # Test: the stuck-training case driven through step functions that run with hooks. Asserts
    # that the session is recreated (two sessions) and not stopped.
    with self.cached_session() as test_session:
      session_creator = CountingSessionCreator(test_session)
      session = monitored_session.MonitoredSession(
          session_creator,
          [FailTrainingAfterCoordinatorStopped(calls_before_stopping=2)])

      self.assertEqual(1, session_creator.number_of_sessions_created)
      self.assertFalse(session.should_stop())

      c = constant_op.constant(0)
      v = array_ops.identity(c)

      def feed_step_fn(value):
        # [FUNCTION DOC]
        # HELPER: factory returning a step function for the given value to feed.
        def step_fn(step_context):
          # [FUNCTION DOC]
          # HELPER: step function running the fetch with hooks and the captured fed value.
          return step_context.run_with_hooks(fetches=v, feed_dict={c: value})
        return step_fn

      # Training will not fail, since it's the call number 0.
      self.assertEqual(51, session.run_step_fn(feed_step_fn(51)))
      self.assertFalse(session.should_stop())
      # Training will fail during the next call, since it's the call
      # number 1.
      self.assertEqual(42, session.run_step_fn(feed_step_fn(42)))
      # Even though the coordinator stopped which and training failed, the
      # underlying session is recreated and training is to be continued.
      self.assertFalse(session.should_stop())
      self.assertEqual(2, session_creator.number_of_sessions_created)

  def create_raw_session_with_failing_coordinator(self, session_creator, hook):
    # [FUNCTION DOC]
    # HELPER: builds a monitored session whose raw-session accessor is overridden to return the
    # innermost raw session. Inputs: session_creator and hook, the single hook to install.
    # Returns the monitored session. Running against it exercises the raw session while the
    # hook's coordinator thread independently stops with an exception. Mutates the created
    # session object.
    """Return MonitoredSession that triggers coordinator failures."""
    session = monitored_session.MonitoredSession(session_creator, [hook])
    # We would like to test a situation where during fetches through the
    # raw session, the coordinator fails with an exception.  To do that, we
    # are going to use (raw_session + StopCoordinatorWithException) hook
    # combination that is stored in
    # `MonitoredSession._RecoverableSession._CoordinatedSession._sess`
    # at this point:
    session._tf_sess = lambda: session._sess._sess._sess
    # `run()` on such a session is equivalent to `run()` on the raw session
    # with separate coordinator threads independently stopping with an
    # exception.
    return session

  @test_util.run_deprecated_v1
  def test_step_fn_recovery_from_coordinator_exception_with_raw_session(self):
    # [FUNCTION DOC]
    # Test: the aborted-error recovery through step functions that use the raw session (not
    # hooks) directly. Asserts the values 51 and 42, no stop, and two sessions created.
    with self.cached_session() as test_session:
      session_creator = CountingSessionCreator(test_session)
      session = self.create_raw_session_with_failing_coordinator(
          session_creator,
          StopCoordinatorWithException(calls_before_stopping=2))

      self.assertEqual(1, session_creator.number_of_sessions_created)
      self.assertFalse(session.should_stop())

      c = constant_op.constant(0)
      v = array_ops.identity(c)

      def feed_step_fn(value):
        # [FUNCTION DOC]
        # HELPER: factory taking value to feed and returning a step function for the raw session.

        def step_fn(step_context):
          # [FUNCTION DOC]
          # HELPER: step function. Runs the fetch directly on the step context's session with the
          # captured fed value and returns the result.
          return step_context.session.run(fetches=v, feed_dict={c: value})

        return step_fn

      # The coordinator will not abort during this call, since it's the call
      # number 0.
      self.assertEqual(51, session.run_step_fn(feed_step_fn(51)))
      self.assertFalse(session.should_stop())
      # The coordinator will abort during the next call, since it's the call
      # number 1.
      self.assertEqual(42, session.run_step_fn(feed_step_fn(42)))
      # Even though the coordinator was asked to stop, the underlying session is
      # recreated and is to be continued.
      self.assertFalse(session.should_stop())
      self.assertEqual(2, session_creator.number_of_sessions_created)

  @test_util.run_deprecated_v1
  def test_recovery_from_non_preemption_in_coordinator_with_raw_session(self):
    # [FUNCTION DOC]
    # Test: the unknown, non-recoverable coordinator error with the raw-session setup. Asserts
    # the session stops, a single session was created and closing raises the unknown error.
    with self.cached_session() as test_session:
      session_creator = CountingSessionCreator(test_session)
      session = self.create_raw_session_with_failing_coordinator(
          session_creator,
          StopCoordinatorWithException(
              calls_before_stopping=2,
              exception_to_raise=errors_impl.UnknownError(
                  None, None, 'Some fatal exception inside the coordinator.')))

      self.assertEqual(1, session_creator.number_of_sessions_created)
      self.assertFalse(session.should_stop())

      c = constant_op.constant(0)
      v = array_ops.identity(c)

      def feed_step_fn(value):
        # [FUNCTION DOC]
        # HELPER: factory taking value to feed and returning a step function.

        def step_fn(step_context):
          # [FUNCTION DOC]
          # HELPER: step function running the fetch with hooks and the captured fed value.
          return step_context.run_with_hooks(fetches=v, feed_dict={c: value})

        return step_fn

      # The coordinator will not abort during this call, since it's the call
      # number 0.
      self.assertEqual(51, session.run_step_fn(feed_step_fn(51)))
      self.assertFalse(session.should_stop())
      # The coordinator will abort during the next call, since it's the call
      # number 1.
      self.assertEqual(42, session.run_step_fn(feed_step_fn(42)))
      # The coordinator was asked to stop due to non-redeemable error. Training
      # should stop and the session should not be recreated.
      self.assertTrue(session.should_stop())
      self.assertEqual(1, session_creator.number_of_sessions_created)
      with self.assertRaises(errors_impl.UnknownError):
        session.close()

  @test_util.run_deprecated_v1
  def test_recovery_from_session_getting_stuck_with_raw_session(self):
    # [FUNCTION DOC]
    # Test: the stuck-training case with the raw-session setup. Asserts the session is
    # recreated (two sessions) and is not stopped.
    with self.cached_session() as test_session:
      session_creator = CountingSessionCreator(test_session)
      session = self.create_raw_session_with_failing_coordinator(
          session_creator,
          FailTrainingAfterCoordinatorStopped(calls_before_stopping=2))

      self.assertEqual(1, session_creator.number_of_sessions_created)
      self.assertFalse(session.should_stop())

      c = constant_op.constant(0)
      v = array_ops.identity(c)

      def feed_step_fn(value):
        # [FUNCTION DOC]
        # HELPER: factory taking value to feed and returning a step function.

        def step_fn(step_context):
          # [FUNCTION DOC]
          # HELPER: step function running the fetch with hooks and the captured fed value.
          return step_context.run_with_hooks(fetches=v, feed_dict={c: value})

        return step_fn

      # Training will not fail, since it's the call number 0.
      self.assertEqual(51, session.run_step_fn(feed_step_fn(51)))
      self.assertFalse(session.should_stop())
      # Training will fail during the next call, since it's the call
      # number 1.
      self.assertEqual(42, session.run_step_fn(feed_step_fn(42)))
      # Even though the coordinator stopped which and training failed, the
      # underlying session is recreated and training is to be continued.
      self.assertFalse(session.should_stop())
      self.assertEqual(2, session_creator.number_of_sessions_created)


class FakeSession(monitored_session._WrappedSession):
  # [CLASS DOC]
  # HELPER: wrapped-session test double that records the keyword arguments of the last run call.
  # Used to check that the hooked session passes feed dictionary, options and metadata through.

  def __init__(self, sess):
    # [FUNCTION DOC]
    # HELPER: wraps the given session and initializes the recorded keyword arguments as empty.
    monitored_session._WrappedSession.__init__(self, sess)
    self.args_called = {}

  def run(self, fetches, **kwargs):
    # [FUNCTION DOC]
    # HELPER: run override. Inputs: fetches and any keyword arguments. Records a copy of the
    # keyword arguments, then runs only the fetches on the wrapped session and returns the
    # result, because other arguments are not forwarded to the real session.
    self.args_called = dict(kwargs)
    # Call run only with fetches since we directly pass other arguments.
    return monitored_session._WrappedSession.run(self, fetches)


class HookedSessionTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the internal hooked session, which calls hooks around each run, merges hook
  # feeds and requests with the caller's, and reports should-stop. Uses fake hooks and real
  # sessions in fresh graphs.
  """Tests of _HookedSession."""

  def testRunPassesAllArguments(self):
    # [FUNCTION DOC]
    # Test: the hooked session passes the feed dictionary, run options and run metadata through
    # unchanged. Asserts the output is the fetched value and the fake session recorded exactly
    # those three arguments.
    with ops.Graph().as_default(), session_lib.Session() as sess:
      mock_run = FakeSession(sess)
      mon_sess = monitored_session._HookedSession(sess=mock_run, hooks=[])
      a_tensor = constant_op.constant([0], name='a_tensor')
      self.evaluate(variables.global_variables_initializer())
      output = mon_sess.run(fetches=a_tensor,
                            feed_dict='a_feed',
                            options='an_option',
                            run_metadata='a_metadata')
      self.assertEqual(output, [0])
      self.assertEqual(mock_run.args_called, {
          'feed_dict': 'a_feed',
          'options': 'an_option',
          'run_metadata': 'a_metadata'
      })

  def testCallsHooksBeginEnd(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name suggests begin and end callbacks are checked, but the test asserts
    # the begin and after-create-session counts are zero (the hooked session does not call
    # them) and only checks before-run and after-run counts.
    # Test: with two fake hooks, one run calls each hook once before and after. Asserts the
    # last run values, run context arguments and session, and the per-callback counts.
    with ops.Graph().as_default(), session_lib.Session() as sess:
      mock_hook = FakeHook()
      mock_hook2 = FakeHook()
      mon_sess = monitored_session._HookedSession(
          sess=sess, hooks=[mock_hook, mock_hook2])
      a_tensor = constant_op.constant([0], name='a_tensor')
      self.evaluate(variables.global_variables_initializer())
      mon_sess.run(a_tensor)

      for hook in [mock_hook, mock_hook2]:
        self.assertEqual(
            hook.last_run_values,
            session_run_hook.SessionRunValues(
                results=None,
                options=config_pb2.RunOptions(),
                run_metadata=config_pb2.RunMetadata()))
        self.assertEqual(hook.last_run_context.original_args,
                         session_run_hook.SessionRunArgs(a_tensor))
        self.assertEqual(hook.last_run_context.session, sess)
        self.assertEqual(hook.call_counter['begin'], 0)
        self.assertEqual(hook.call_counter['after_create_session'], 0)
        self.assertEqual(hook.call_counter['before_run'], 1)
        self.assertEqual(hook.call_counter['after_run'], 1)

  def testShouldStop(self):
    # [FUNCTION DOC]
    # Test: the hooked session reports should-stop only after one of its hooks asks to stop.
    # Asserts false after a normal run and true after a run with a hook that requests a stop.
    with ops.Graph().as_default(), session_lib.Session() as sess:
      mock_hook = FakeHook()
      mock_hook2 = FakeHook()
      mon_sess = monitored_session._HookedSession(
          sess=sess, hooks=[mock_hook, mock_hook2])
      constant_op.constant([0], name='a_tensor')
      self.evaluate(variables.global_variables_initializer())

      mon_sess.run(fetches='a_tensor')
      self.assertFalse(mon_sess.should_stop())

      mock_hook.should_stop = True
      mon_sess.run(fetches='a_tensor')
      self.assertTrue(mon_sess.should_stop())

  def testFetchesHookRequests(self):
    # [FUNCTION DOC]
    # Test: hook run requests add fetches. Two hooks each request an extra tensor; asserts the
    # caller's output is unchanged and each hook's run values contain only its own result.
    with ops.Graph().as_default(), session_lib.Session() as sess:
      mock_hook = FakeHook()
      mock_hook2 = FakeHook()
      mon_sess = monitored_session._HookedSession(
          sess=sess, hooks=[mock_hook, mock_hook2])
      a_tensor = constant_op.constant([0], name='a_tensor')
      another_tensor = constant_op.constant([5], name='another_tensor')
      third_tensor = constant_op.constant([10], name='third_tensor')
      mock_hook.request = session_run_hook.SessionRunArgs([another_tensor])
      mock_hook2.request = session_run_hook.SessionRunArgs([third_tensor])
      self.evaluate(variables.global_variables_initializer())

      output = mon_sess.run(fetches=a_tensor)
      self.assertEqual(output, [0])
      self.assertEqual(mock_hook.last_run_values.results, [5])
      self.assertEqual(mock_hook2.last_run_values.results, [10])

  def testOnlyHooksHaveFeeds(self):
    # [FUNCTION DOC]
    # Test: feeds supplied only by hooks are combined. Two hooks feed different inputs of an add
    # operation and the caller feeds nothing; asserts the sum of both fed values (15).
    with ops.Graph().as_default(), session_lib.Session() as sess:
      mock_hook = FakeHook()
      mock_hook2 = FakeHook()
      mon_sess = monitored_session._HookedSession(
          sess=sess, hooks=[mock_hook, mock_hook2])
      a_tensor = constant_op.constant([0], name='a_tensor')
      b_tensor = constant_op.constant([0], name='b_tensor')
      add_tensor = a_tensor + b_tensor
      mock_hook.request = session_run_hook.SessionRunArgs(
          None, feed_dict={a_tensor: [5]})
      mock_hook2.request = session_run_hook.SessionRunArgs(
          None, feed_dict={b_tensor: [10]})
      self.evaluate(variables.global_variables_initializer())

      self.assertEqual(mon_sess.run(fetches=add_tensor), [15])

  def testBothHooksAndUserHaveFeeds(self):
    # [FUNCTION DOC]
    # Test: hook feeds and the caller's feed are merged. Two hooks and the caller each feed one
    # of three inputs; asserts the sum (35) and that the caller's feed dictionary was not
    # modified (still one entry).
    with ops.Graph().as_default(), session_lib.Session() as sess:
      mock_hook = FakeHook()
      mock_hook2 = FakeHook()
      mon_sess = monitored_session._HookedSession(
          sess=sess, hooks=[mock_hook, mock_hook2])
      a_tensor = constant_op.constant([0], name='a_tensor')
      b_tensor = constant_op.constant([0], name='b_tensor')
      c_tensor = constant_op.constant([0], name='c_tensor')
      add_tensor = a_tensor + b_tensor + c_tensor
      mock_hook.request = session_run_hook.SessionRunArgs(
          None, feed_dict={a_tensor: [5]})
      mock_hook2.request = session_run_hook.SessionRunArgs(
          None, feed_dict={b_tensor: [10]})
      self.evaluate(variables.global_variables_initializer())

      feed_dict = {c_tensor: [20]}
      self.assertEqual(
          mon_sess.run(fetches=add_tensor, feed_dict=feed_dict), [35])
      # User feed_dict should not be changed
      self.assertEqual(len(feed_dict), 1)

  def testHooksFeedConflicts(self):
    # [FUNCTION DOC]
    # Test: two hooks feeding the same tensor conflict. Asserts the run raises a runtime error
    # saying the same tensor is fed twice.
    with ops.Graph().as_default(), session_lib.Session() as sess:
      mock_hook = FakeHook()
      mock_hook2 = FakeHook()
      mon_sess = monitored_session._HookedSession(
          sess=sess, hooks=[mock_hook, mock_hook2])
      a_tensor = constant_op.constant([0], name='a_tensor')
      b_tensor = constant_op.constant([0], name='b_tensor')
      add_tensor = a_tensor + b_tensor
      mock_hook.request = session_run_hook.SessionRunArgs(
          None, feed_dict={a_tensor: [5]})
      mock_hook2.request = session_run_hook.SessionRunArgs(
          None, feed_dict={a_tensor: [10]})
      self.evaluate(variables.global_variables_initializer())

      with self.assertRaisesRegex(RuntimeError, 'Same tensor is fed'):
        mon_sess.run(fetches=add_tensor)

  def testHooksAndUserFeedConflicts(self):
    # [FUNCTION DOC]
    # Test: a hook and the caller feeding the same tensor conflict. Asserts a runtime error
    # saying the same tensor is fed twice.
    with ops.Graph().as_default(), session_lib.Session() as sess:
      mock_hook = FakeHook()
      mock_hook2 = FakeHook()
      mon_sess = monitored_session._HookedSession(
          sess=sess, hooks=[mock_hook, mock_hook2])
      a_tensor = constant_op.constant([0], name='a_tensor')
      b_tensor = constant_op.constant([0], name='b_tensor')
      add_tensor = a_tensor + b_tensor
      mock_hook.request = session_run_hook.SessionRunArgs(
          None, feed_dict={a_tensor: [5]})
      mock_hook2.request = session_run_hook.SessionRunArgs(
          None, feed_dict={b_tensor: [10]})
      self.evaluate(variables.global_variables_initializer())

      with self.assertRaisesRegex(RuntimeError, 'Same tensor is fed'):
        mon_sess.run(fetches=add_tensor, feed_dict={b_tensor: [10]})


class RaiseOnceAtCountN(session_run_hook.SessionRunHook):
  # [CLASS DOC]
  # HELPER: hook that raises a given exception once, on the run call where its countdown
  # reaches zero. Used to test retry, recovery and stop handling in monitored sessions.
  """Hook that raises an Exception at step N."""

  def __init__(self, n, ex):
    # [FUNCTION DOC]
    # HELPER: inputs n, the countdown of before-run calls until the raise, and ex, the
    # exception to raise. Also keeps a flag recording whether it already raised.
    self.n = n
    self.ex = ex
    self.raised = False

  def before_run(self, run_context):
    # [FUNCTION DOC]
    # HELPER: before-run callback. Decrements the countdown; the first time it reaches exactly
    # zero marks the flag and raises the stored exception. Otherwise returns nothing/null.
    # Raise the first time we reach step N.
    self.n -= 1
    if 0 == self.n and not self.raised:
      self.raised = True
      raise self.ex
    return None


class RunOptionsMetadataHook(session_run_hook.SessionRunHook):
  # [CLASS DOC]
  # HELPER: hook that requests specific run options (trace level, timeout, output of partition
  # graphs, report on out-of-memory, and one debug tensor watch) before each run, and records
  # the run options and run metadata seen afterwards. Used to test merging of options.
  """A hook that observes & optionally modifies RunOptions and RunMetadata."""

  def __init__(self, trace_level, timeout_in_ms, output_partition_graphs,
               debug_tensor_watch, report_tensor_allocations_upon_oom):
    # [FUNCTION DOC]
    # HELPER: stores trace level, timeout in milliseconds, output-partition-graphs flag, a debug
    # tensor watch and the report-allocations-on-out-of-memory flag, and creates two empty
    # lists for the observed run options and run metadata.
    self._trace_level = trace_level
    self._timeout_in_ms = timeout_in_ms
    self._output_partition_graphs = output_partition_graphs
    self._debug_tensor_watch = debug_tensor_watch
    self._report_tensor_allocations_upon_oom = (
        report_tensor_allocations_upon_oom)

    self.run_options_list = []
    self.run_metadata_list = []

  def before_run(self, run_context):
    # [FUNCTION DOC]
    # HELPER: before-run callback. Builds run options from the stored settings plus the debug
    # tensor watch and returns them wrapped in a run request with no fetches or feeds.
    options = config_pb2.RunOptions(
        trace_level=self._trace_level,
        timeout_in_ms=self._timeout_in_ms,
        output_partition_graphs=self._output_partition_graphs,
        report_tensor_allocations_upon_oom=self
        ._report_tensor_allocations_upon_oom)
    options.debug_options.debug_tensor_watch_opts.extend(
        [self._debug_tensor_watch])
    return session_run_hook.SessionRunArgs(None, None, options=options)

  def after_run(self, run_context, run_values):
    # [FUNCTION DOC]
    # HELPER: after-run callback. Appends the run options and run metadata from the run values
    # to the recording lists. Returns nothing.
    self.run_options_list.append(run_values.options)
    self.run_metadata_list.append(run_values.run_metadata)


class MonitoredSessionTest(test.TestCase):
  # [CLASS DOC]
  # Test case for MonitoredSession: defaults, stop-at-step hooks, checkpoint recovery, retry on
  # transient errors, clean exit on end-of-input errors, exception pass-through, graph
  # finalization, run-options merging, and step function behavior (with hooks, recovery and
  # error cases).
  """MonitoredSession tests."""

  def test_defaults(self):
    # [FUNCTION DOC]
    # Test: a default monitored session initializes variables on creation. Asserts the variable
    # reads zero.
    with ops.Graph().as_default():
      a_var = variable_v1.VariableV1(0)
      with monitored_session.MonitoredSession() as session:
        self.assertEqual(0, session.run(a_var))

  def test_last_step(self):
    # [FUNCTION DOC]
    # Test: stop-at-last-step hook. Runs to step three (stop flag becomes true), saves a
    # checkpoint, then starts a new session restoring it through a scaffold init function with a
    # last step of five; asserts steps three, four, five and stop at five.
    logdir = _test_dir(self.get_temp_dir(), 'test_last_step')
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      # Run till step 3 and save.
      hooks = [basic_session_run_hooks.StopAtStepHook(last_step=3)]
      with monitored_session.MonitoredSession(hooks=hooks) as session:
        self.assertEqual(0, session.run(gstep))
        self.assertFalse(session.should_stop())
        self.assertEqual(1, session.run(do_step))
        self.assertFalse(session.should_stop())
        self.assertEqual(2, session.run(do_step))
        self.assertFalse(session.should_stop())
        self.assertEqual(3, session.run(do_step))
        self.assertTrue(session.should_stop())
        save_path = saver_lib._get_saver_or_default().save(
            session._coordinated_creator.tf_sess,
            os.path.join(logdir, 'step-3'))
      # Run till step 5 and save.
      def load_ckpt(scaffold, sess):
        # [FUNCTION DOC]
        # HELPER: scaffold init function. Inputs scaffold and sess. Restores the saved checkpoint
        # path from the enclosing test into the session. Returns nothing.
        scaffold.saver.restore(sess, save_path)

      session_creator = monitored_session.ChiefSessionCreator(
          monitored_session.Scaffold(init_fn=load_ckpt))
      hooks = [basic_session_run_hooks.StopAtStepHook(last_step=5)]
      with monitored_session.MonitoredSession(
          hooks=hooks, session_creator=session_creator) as session:
        self.assertEqual(3, session.run(gstep))
        self.assertFalse(session.should_stop())
        self.assertEqual(4, session.run(do_step))
        self.assertFalse(session.should_stop())
        self.assertEqual(5, session.run(do_step))
        self.assertTrue(session.should_stop())

  def test_num_steps(self):
    # [FUNCTION DOC]
    # Test: stop-after-number-of-steps hook. Runs three steps and stops, saves a checkpoint,
    # restores it in a new session with a limit of four steps; asserts the step values and that
    # the session stops after the fourth.
    logdir = _test_dir(self.get_temp_dir(), 'test_num_steps')
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      # Do 3 steps and save.
      hooks = [basic_session_run_hooks.StopAtStepHook(num_steps=3)]
      with monitored_session.MonitoredSession(hooks=hooks) as session:
        session.run(do_step)
        self.assertFalse(session.should_stop())
        session.run(do_step)
        self.assertFalse(session.should_stop())
        session.run(do_step)
        self.assertTrue(session.should_stop())
        save_path = saver_lib._get_saver_or_default().save(
            session._coordinated_creator.tf_sess,
            os.path.join(logdir, 'step-3'))
      # Restore and do 4 steps.
      def load_ckpt(scaffold, sess):
        # [FUNCTION DOC]
        # HELPER: scaffold init function. Inputs scaffold and sess. Restores the checkpoint saved
        # earlier in the test. Returns nothing.
        scaffold.saver.restore(sess, save_path)

      session_creator = monitored_session.ChiefSessionCreator(
          scaffold=monitored_session.Scaffold(init_fn=load_ckpt))
      hooks = [basic_session_run_hooks.StopAtStepHook(num_steps=4)]
      with monitored_session.MonitoredSession(
          hooks=hooks, session_creator=session_creator) as session:
        self.assertEqual(4, session.run(do_step))
        self.assertFalse(session.should_stop())
        session.run(do_step)
        self.assertFalse(session.should_stop())
        session.run(do_step)
        self.assertFalse(session.should_stop())
        session.run(do_step)
        self.assertTrue(session.should_stop())

  # This set of tests, verifies the supervised session behavior when exceptions
  # are raised next to the innermost session run() call.

  @test_util.run_deprecated_v1
  def test_recovery(self):
    # [FUNCTION DOC]
    # Test: checkpoint-based recovery. A saver hook saves every step; a restarted session on the
    # same directory restores global step two, both when found by directory and when the
    # checkpoint path is given explicitly.
    logdir = _test_dir(self.get_temp_dir(), 'test_recovery')
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      scaffold = monitored_session.Scaffold()
      # Use a hook to save the model every 100 steps.  It also saves it at
      # the end.
      hooks = [
          basic_session_run_hooks.CheckpointSaverHook(
              logdir, save_steps=1, scaffold=scaffold)
      ]
      with monitored_session.MonitoredSession(
          session_creator=monitored_session.ChiefSessionCreator(
              scaffold, checkpoint_dir=logdir),
          hooks=hooks) as session:
        self.assertEqual(0, session.run(gstep))
        self.assertEqual(1, session.run(do_step))
        self.assertEqual(2, session.run(do_step))
      # A restart will find the checkpoint and recover automatically.
      with monitored_session.MonitoredSession(
          session_creator=monitored_session.ChiefSessionCreator(
              scaffold, checkpoint_dir=logdir)) as session:
        self.assertEqual(2, session.run(gstep))
      # A restart will find the checkpoint and recover automatically.
      with monitored_session.MonitoredSession(
          session_creator=monitored_session.ChiefSessionCreator(
              scaffold,
              checkpoint_filename_with_path=checkpoint_management.
              latest_checkpoint(logdir))) as session:
        self.assertEqual(2, session.run(gstep))

  def test_retry_initialization_on_aborted_error(self):
    # [FUNCTION DOC]
    # Test: an aborted error raised by the scaffold init function on the first attempt is
    # silently retried. Asserts the session is usable (step zero) and the failure was raised.
    # Tests that we silently retry on abort during initialization.
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      self.init_raised_aborted_error = False

      def _init_fn(scaffold, session):
        # [FUNCTION DOC]
        # HELPER: scaffold init function that raises an aborted error the first time it is called
        # and does nothing after. Inputs scaffold and session are unused.
        _, _ = scaffold, session
        if not self.init_raised_aborted_error:
          self.init_raised_aborted_error = True
          raise errors_impl.AbortedError(None, None, 'Abort')

      with monitored_session.MonitoredSession(
          session_creator=monitored_session.ChiefSessionCreator(
              scaffold=monitored_session.Scaffold(
                  init_fn=_init_fn))) as session:
        self.assertFalse(session.should_stop())
        self.assertEqual(0, session.run(gstep))
      self.assertTrue(self.init_raised_aborted_error)

  def _retry_test(self, ex):
    # [FUNCTION DOC]
    # HELPER: shared body of the retry tests. Input ex is the error that a hook raises once at
    # the fourth run. Without a checkpoint saver, the session restarts from fresh state;
    # asserts the step counter returns to one and continues to two, and the hook raised.
    # Tests that we silently retry on error.  Note that this does not test
    # recovery as we do not use a CheckpointSaver in this test.
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      hook = RaiseOnceAtCountN(4, ex)
      with monitored_session.MonitoredSession(hooks=[hook]) as session:
        self.assertEqual(0, session.run(gstep))
        self.assertEqual(1, session.run(do_step))
        self.assertEqual(2, session.run(do_step))
        self.assertFalse(session.should_stop())
        # Here at step 3, the hook triggers and raises AbortedError.  The
        # MonitoredSession automatically retries and restart from a freshly
        # initialized session, so the step is back to 0 and running do_step
        # moves it to 1.
        self.assertEqual(1, session.run(do_step))
        self.assertFalse(session.should_stop())
        self.assertTrue(hook.raised)
        self.assertEqual(2, session.run(do_step))
        self.assertFalse(session.should_stop())

  def test_retry_on_aborted_error(self):
    # [FUNCTION DOC]
    # Test: the retry scenario with an aborted error.
    self._retry_test(errors_impl.AbortedError(None, None, 'Abort'))

  def test_retry_on_unavailable_error(self):
    # [FUNCTION DOC]
    # Test: the retry scenario with an unavailable error.
    self._retry_test(errors_impl.UnavailableError(None, None, 'Unavailable'))

  def test_recover_and_retry_on_aborted_error(self):
    # [FUNCTION DOC]
    # Test: recovery with a checkpoint. A hook raises an aborted error at the fourth run while a
    # saver hook saves each step; asserts the session restores and the step counter continues
    # to three and four with the abort hook having fired.
    # Tests that we silently retry and recover on abort.  This test uses
    # a CheckpointSaver to have something to recover from.
    logdir = _test_dir(self.get_temp_dir(),
                       'test_recover_and_retry_on_aborted_error')
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      scaffold = monitored_session.Scaffold()
      abort_hook = RaiseOnceAtCountN(
          4, errors_impl.AbortedError(None, None, 'Abort'))
      # Save after each step.
      ckpt_hook = basic_session_run_hooks.CheckpointSaverHook(
          logdir, save_steps=1, scaffold=scaffold)
      hooks = [abort_hook, ckpt_hook]
      with monitored_session.MonitoredSession(
          session_creator=monitored_session.ChiefSessionCreator(
              scaffold, checkpoint_dir=logdir),
          hooks=hooks) as session:
        self.assertEqual(0, session.run(gstep))
        self.assertEqual(1, session.run(do_step))
        self.assertEqual(2, session.run(do_step))
        self.assertFalse(session.should_stop())
        # Here at step 3, the hook triggers and raises AbortedError.  The
        # MonitoredSession automatically restores and retries.
        self.assertEqual(3, session.run(do_step))
        self.assertTrue(abort_hook.raised)
        self.assertFalse(session.should_stop())
        self.assertEqual(4, session.run(do_step))
        self.assertFalse(session.should_stop())

  def test_exit_cleanly_on_out_of_range_exception(self):
    # [FUNCTION DOC]
    # Test: an out-of-range error from a hook ends the with block cleanly (the error is
    # swallowed), the code after the failing run is not executed and should-stop is true.
    # Tests that we stop cleanly when OutOfRange is raised.
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      hook = RaiseOnceAtCountN(2, errors_impl.OutOfRangeError(None, None,
                                                              'EOI'))
      session = monitored_session.MonitoredSession(hooks=[hook])
      # session should cleanly exit from the context.
      with session:
        self.assertEqual(0, session.run(gstep))
        self.assertFalse(session.should_stop())
        # Here at step 1, the hook triggers and raises OutOfRange. The
        # session should go into should_stop() mode. It should raise the
        # exception. So next step should not be executed.
        session.run(do_step)
        self.assertTrue(False)
      self.assertTrue(session.should_stop())

  def test_exit_cleanly_on_stop_iteration_exception(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the inline comments say out-of-range error, but the hook raises a stop
    # iteration exception; the behavior verified is the clean exit on stop iteration.
    # Test: a stop iteration from a hook ends the with block cleanly; should-stop is true.
    # Tests that we stop cleanly when OutOfRange is raised.
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      hook = RaiseOnceAtCountN(2, StopIteration)
      session = monitored_session.MonitoredSession(hooks=[hook])
      # session should cleanly exit from the context.
      with session:
        self.assertEqual(0, session.run(gstep))
        self.assertFalse(session.should_stop())
        # Here at step 1, the hook triggers and raises StopIteration. The
        # session should go into should_stop() mode. It should raise the
        # exception. So next step should not be executed.
        session.run(do_step)
        self.assertTrue(False)
      self.assertTrue(session.should_stop())

  def test_regular_exception_pass_through_run(self):
    # [FUNCTION DOC]
    # Test: a regular runtime error from a hook propagates out of the with block. Asserts the
    # error message, that the hook raised and that the session reports should-stop.
    # Tests that regular exceptions just pass through a "with
    # MonitoredSession" block and set the session in stop mode.
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      hook = RaiseOnceAtCountN(4, RuntimeError('regular exception'))
      session = monitored_session.MonitoredSession(hooks=[hook])
      with self.assertRaisesRegex(RuntimeError, 'regular exception'):
        with session:
          self.assertEqual(0, session.run(gstep))
          self.assertEqual(1, session.run(do_step))
          self.assertEqual(2, session.run(do_step))
          self.assertFalse(session.should_stop())
          # This triggers the hook and raises the exception
          session.run(do_step)
          # We should not hit this
          self.assertFalse(True)
      self.assertTrue(hook.raised)
      self.assertTrue(session.should_stop())

  def test_regular_exception_reported_to_coord_pass_through_run(self):
    # [FUNCTION DOC]
    # Test: an error reported to the coordinator from a thread surfaces at exit of the with
    # block even though later runs work. Asserts the runtime error and that the later run was
    # performed.
    # Tests that regular exceptions reported to the coordinator from a thread
    # passes through a "run()" call within a "with MonitoredSession" block and
    # set the session in stop mode.
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      session = monitored_session.MonitoredSession()
      run_performed_without_error = False
      with self.assertRaisesRegex(RuntimeError, 'a thread wants to stop'):
        with session:
          self.assertEqual(0, session.run(gstep))
          # Report an exception through the coordinator.
          try:
            raise RuntimeError('a thread wants to stop')
          except RuntimeError as e:
            session._coordinated_creator.coord.request_stop(e)
          # Call run() which should perform normally.
          self.assertEqual(0, session.run(gstep))
          run_performed_without_error = True
      self.assertTrue(run_performed_without_error)

  def test_regular_exception_reported_to_coord_pass_through_return(self):
    # [FUNCTION DOC]
    # Test: an error reported to the coordinator makes should-stop true and is raised on leaving
    # the with block.
    # Tests that regular exceptions reported to the coordinator from a thread
    # passes through returning from a "with MonitoredSession" block and
    # set the session in stop mode.
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      session = monitored_session.MonitoredSession()
      with self.assertRaisesRegex(RuntimeError, 'a thread wants to stop'):
        with session:
          self.assertEqual(0, session.run(gstep))
          # Report an exception through the coordinator.
          try:
            raise RuntimeError('a thread wants to stop')
          except RuntimeError as e:
            session._coordinated_creator.coord.request_stop(e)
          self.assertTrue(session.should_stop())

  # This set of tests, verifies the session behavior when exceptions are raised
  # from code inside a "with MonitoredSession:" context.

  def test_stop_cleanly_when_no_exception_in_with_body(self):
    # [FUNCTION DOC]
    # Test: with no error in the with body, the session is closed on exit. Asserts step values
    # one and two, not stopped inside, and stopped and closed after.
    # Tests that regular exceptions pass through
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      session = monitored_session.MonitoredSession()
      with session:
        self.assertEqual(1, session.run(do_step))
        self.assertEqual(2, session.run(do_step))
        self.assertFalse(session.should_stop())
      # Should have closed.
      self.assertTrue(session.should_stop())
      self.assertTrue(session._is_closed())

  def test_raises_regular_exceptions_in_with_body(self):
    # [FUNCTION DOC]
    # Test: a regular error raised in the with body propagates; afterwards the session is
    # stopped and closed.
    # Tests that regular exceptions in "with body" are seen outside.
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      session = monitored_session.MonitoredSession()
      # We should see that exception.
      with self.assertRaisesRegex(RuntimeError, 'regular exception'):
        with session:
          self.assertEqual(1, session.run(do_step))
          self.assertEqual(2, session.run(do_step))
          self.assertFalse(session.should_stop())
          # Will be visible outside the "with body".
          raise RuntimeError('regular exception')
      # Should have closed.
      self.assertTrue(session.should_stop())
      self.assertTrue(session._is_closed())

  def test_graph(self):
    # [FUNCTION DOC]
    # Test: the session's graph property equals the default graph it was created in.
    with ops.Graph().as_default() as g:
      with monitored_session.MonitoredSession() as session:
        self.assertEqual(g, session.graph)

  def test_graph_finalized_during_run_unfinalized_after_exit(self):
    # [FUNCTION DOC]
    # Test: the graph is finalized while the session is active and unfinalized after exit.
    with ops.Graph().as_default() as g:
      a_var = variable_v1.VariableV1(0)
      with monitored_session.MonitoredSession() as session:
        self.assertEqual(0, session.run(a_var))
        self.assertTrue(g.finalized)
      self.assertFalse(g.finalized)

  def test_keep_finalized_graph_as_finalized(self):
    # [FUNCTION DOC]
    # Test: a graph already finalized by a scaffold stays finalized after the session exits.
    with ops.Graph().as_default() as g:
      a_var = variable_v1.VariableV1(0)
      monitored_session.Scaffold().finalize()
      with monitored_session.MonitoredSession() as session:
        self.assertEqual(0, session.run(a_var))
        self.assertTrue(g.finalized)
      self.assertTrue(g.finalized)

  def test_merge_run_options_from_hooks(self):
    # [FUNCTION DOC]
    # Test: run options requested by two hooks are merged. Later numeric values override
    # (trace level three, timeout 60000, output of partition graphs true) and the two debug
    # watches are combined; also checks that run metadata with partition graphs was reported.
    """Test for rewriting RunOptions and observing RunMetadata with hooks."""

    with ops.Graph().as_default():
      my_const = constant_op.constant(42, name='my_const')
      _ = constant_op.constant(24, name='my_const_2')

      watch_a = debug_pb2.DebugTensorWatch(
          node_name='my_const',
          output_slot=0,
          debug_ops=['DebugIdentity'],
          debug_urls=[])
      hook_a = RunOptionsMetadataHook(2, 30000, False, watch_a, False)
      watch_b = debug_pb2.DebugTensorWatch(
          node_name='my_const_2',
          output_slot=0,
          debug_ops=['DebugIdentity'],
          debug_urls=[])
      hook_b = RunOptionsMetadataHook(3, 60000, True, watch_b, True)
      with monitored_session.MonitoredSession(
          hooks=[hook_a, hook_b]) as session:
        self.assertEqual(42, session.run(my_const))

        # trace_level=3 should have overridden trace_level=2;
        # timeout_in_ms=60000 should have overridden 30000;
        # output_partition_graphs=True should have overridden False.
        # The two debug tensor watches should have been merged.
        self.assertEqual([
            config_pb2.RunOptions(
                trace_level=3,
                timeout_in_ms=60000,
                output_partition_graphs=True,
                debug_options=debug_pb2.DebugOptions(
                    debug_tensor_watch_opts=[watch_a, watch_b]),
                report_tensor_allocations_upon_oom=True),
        ], hook_b.run_options_list)
        self.assertEqual(1, len(hook_b.run_metadata_list))
        self.assertTrue(
            isinstance(hook_b.run_metadata_list[0], config_pb2.RunMetadata))
        self.assertGreater(len(hook_b.run_metadata_list[0].partition_graphs), 0)

  def test_merge_caller_and_hook_run_options(self):
    # [FUNCTION DOC]
    # Test: run options passed by the caller are merged with a hook's options. Caller trace
    # level and partition output win, hook timeout wins, and debug watches are combined with the
    # caller's first; also checks the metadata contains partition graphs.
    """Test that RunOptions from caller and hooks can be merged properly."""

    with ops.Graph().as_default():
      my_const = constant_op.constant(42, name='my_const')
      _ = constant_op.constant(24, name='my_const_2')

      hook_watch = debug_pb2.DebugTensorWatch(
          node_name='my_const_2',
          output_slot=0,
          debug_ops=['DebugIdentity'],
          debug_urls=[])
      hook = RunOptionsMetadataHook(2, 60000, False, hook_watch, False)
      with monitored_session.MonitoredSession(hooks=[hook]) as session:
        caller_watch = debug_pb2.DebugTensorWatch(
            node_name='my_const',
            output_slot=0,
            debug_ops=['DebugIdentity'],
            debug_urls=[])
        caller_options = config_pb2.RunOptions(
            trace_level=3,
            timeout_in_ms=30000,
            output_partition_graphs=True,
            report_tensor_allocations_upon_oom=True)
        caller_options.debug_options.debug_tensor_watch_opts.extend(
            [caller_watch])
        self.assertEqual(42, session.run(my_const, options=caller_options))

        # trace_level=3 from the caller should override 2 from the hook.
        # timeout_in_ms=60000 from the hook should override from the caller.
        # output_partition_graph=True from the caller should override False
        # from the hook.
        # The two debug watches from the caller and the hook should be merged,
        # in that order.
        self.assertEqual([
            config_pb2.RunOptions(
                trace_level=3,
                timeout_in_ms=60000,
                output_partition_graphs=True,
                debug_options=debug_pb2.DebugOptions(
                    debug_tensor_watch_opts=[caller_watch, hook_watch]),
                report_tensor_allocations_upon_oom=True),
        ], hook.run_options_list)
        self.assertEqual(1, len(hook.run_metadata_list))
        self.assertTrue(
            isinstance(hook.run_metadata_list[0], config_pb2.RunMetadata))
        self.assertGreater(len(hook.run_metadata_list[0].partition_graphs), 0)

  @test_util.run_deprecated_v1
  def test_with_statement_and_close(self):
    # [FUNCTION DOC]
    # Test: closing the session inside its with block raises a runtime error saying the session
    # is already closed (regression test for a reported issue about a clearer error message).
    # Test case for https://github.com/tensorflow/tensorflow/issues/12224
    # where close() inside the with should have a better error message.
    with self.assertRaisesRegex(RuntimeError, 'Session is already closed'):
      with monitored_session.MonitoredSession() as session:
        session.close()

  def test_step_fn_example(self):
    # [FUNCTION DOC]
    # Test: a step function that runs a fetch with hooks and a fed value returns the value
    # (near 3.2).
    with ops.Graph().as_default():
      c = array_ops.placeholder(dtypes.float32)
      v = array_ops.identity(c)

      def step_fn(step_context):
        # [FUNCTION DOC]
        # HELPER: step function. Input step_context; returns the result of running the fetch
        # with hooks and a fed value of 3.2.
        value = step_context.run_with_hooks(fetches=v, feed_dict={c: 3.2})
        return value

      with monitored_session.MonitoredSession() as session:
        self.assertNear(3.2, session.run_step_fn(step_fn), 0.1)

  def test_step_function_stops(self):
    # [FUNCTION DOC]
    # Test: a step function that requests a stop returns nothing and the session reports
    # should-stop.
    with ops.Graph().as_default():

      def step_fn(step_context):
        # [FUNCTION DOC]
        # HELPER: step function that asks the step context to request a stop. Returns nothing.
        step_context.request_stop()

      with monitored_session.MonitoredSession() as session:
        self.assertEqual(None, session.run_step_fn(step_fn))
        self.assertTrue(session.should_stop())

  def test_step_request_stop_without_a_with_block(self):
    # [FUNCTION DOC]
    # Test: requesting a stop from a step function outside a with block raises a stop iteration
    # to the caller, and the session is not marked as stopped.
    with ops.Graph().as_default():
      was_stop_iteration_raised = False

      def step_fn(step_context):
        # [FUNCTION DOC]
        # HELPER: step function that asks the step context to request a stop.
        step_context.request_stop()

      session = monitored_session.MonitoredSession()
      try:
        self.assertEqual(None, session.run_step_fn(step_fn))
      except StopIteration:
        was_stop_iteration_raised = True

      self.assertTrue(was_stop_iteration_raised)
      self.assertFalse(session.should_stop())

  def test_step_request_stop_in_a_loop(self):
    # [FUNCTION DOC]
    # Test: in a loop that runs while not stopped, a step function requesting a stop ends the
    # loop by raising; the line after the call must never run.
    with ops.Graph().as_default():
      def step_fn(step_context):
        # [FUNCTION DOC]
        # HELPER: step function that asks the step context to request a stop.
        step_context.request_stop()

      with monitored_session.MonitoredSession() as session:
        while not session.should_stop():
          _ = session.run_step_fn(step_fn)
          self.fail('An exception should be raised on the line above.')

  def test_step_request_stop_with_returning_a_type(self):
    # [FUNCTION DOC]
    # Test: a step function may return any value, here a text; the session returns it.
    with ops.Graph().as_default():

      def step_fn(step_context):
        # [FUNCTION DOC]
        # HELPER: step function ignoring its context and returning a fixed text.
        del step_context
        return 'a type'

      with monitored_session.MonitoredSession() as session:
        self.assertEqual('a type', session.run_step_fn(step_fn))

  def test_step_with_extra_arguments(self):
    # [FUNCTION DOC]
    # Test: a step function with an extra parameter beyond the context is rejected. Asserts a
    # value error stating a step function may have either one context argument or none.
    with ops.Graph().as_default():

      def step_fn(step_context, extra_foo):
        # [FUNCTION DOC]
        # HELPER: invalid step function with an extra parameter; deletes both inputs.
        del step_context, extra_foo

      with monitored_session.MonitoredSession() as session:
        with self.assertRaisesRegex(
            ValueError,
            '`step_fn` may either have one `step_context` argument'):
          self.assertEqual(None, session.run_step_fn(step_fn))

  def test_step_fn_belongs_to_a_class(self):
    # [FUNCTION DOC]
    # Test: a bound method of an object can serve as step function. Asserts the result is near
    # 3.2.
    with ops.Graph().as_default():
      c = array_ops.placeholder(dtypes.float32)
      v = array_ops.identity(c)

      class Model:
        # [CLASS DOC]
        # HELPER: holder for a step function method, defined inside the test.

        def step_fn(self, step_context):
          # [FUNCTION DOC]
          # HELPER: method used as step function. Input step_context; returns the result of
          # running the fetch with hooks and a fed value of 3.2.
          return step_context.run_with_hooks(fetches=v, feed_dict={c: 3.2})

      with monitored_session.MonitoredSession() as session:
        model = Model()
        self.assertNear(3.2, session.run_step_fn(model.step_fn), 0.1)

  def test_step_fn_belongs_to_a_class_and_has_extra_methods(self):
    # [FUNCTION DOC]
    # Test: a bound method that takes an extra parameter besides self and the context is
    # rejected with the same value error about the allowed step function parameters.
    with ops.Graph().as_default():

      class Model:
        # [CLASS DOC]
        # HELPER: holder for an invalid step function method with an extra parameter.

        def step_fn(self, step_context, extra_foo):
          # [FUNCTION DOC]
          # HELPER: invalid method taking context and an extra argument; deletes both.
          del step_context, extra_foo

      with monitored_session.MonitoredSession() as session:
        with self.assertRaisesRegex(
            ValueError,
            '`step_fn` may either have one `step_context` argument'):
          model = Model()
          self.assertEqual(None, session.run_step_fn(model.step_fn))

  def test_step_fn_with_hooks(self):
    # [FUNCTION DOC]
    # Test: ordering of operations between a step function and hooks. The step function runs a
    # first stage directly, then a second stage with hooks; the hook adds a fetch before the run
    # and checks cumulative values after, running a final stage. Asserts the total of the
    # increments (about 1.5) is returned.
    with ops.Graph().as_default():
      var = resource_variable_ops.ResourceVariable(0.0)

      # This test highlights the interaction of hooks with
      # `Monitoredsession.run_step_fn`.  The order of execution of operations
      # below is:
      #   0.  stage_0
      #   1.  stage_1_0 or stage_1_1 in an undefined order
      #   2.  stage_2

      stage_0 = state_ops.assign_add(var, 0.3)
      stage_1_0 = state_ops.assign_add(var, 0.7)
      # The order of `stage_1_0` and `stage_1_1` is undefined by
      # `MonitoredSession`, but we should be able to assert when both of them
      # are complete.  To obtain a consistent result of adding two different
      # constants to `var`, we rely on a control dependency and
      # `ResourceVariable`.  Otherwise, it is possible that one of the
      # additions overwrites the result of the other addition.
      with ops.control_dependencies([stage_1_0]):
        stage_1_1 = state_ops.assign_add(var, 0.5)
      stage_2 = state_ops.assign_add(var, 1.1)

      class Hook(session_run_hook.SessionRunHook):
        # [CLASS DOC]
        # HELPER: hook that adds a fetch before the run and, after the run, asserts through the
        # test case that the variable and the following stage have the expected cumulative values.

        def __init__(self, testing):
          # [FUNCTION DOC]
          # HELPER: stores the test case used for assertions.
          self._testing = testing

        def before_run(self, run_context):
          # [FUNCTION DOC]
          # HELPER: before-run callback; returns a request fetching the first stage op.
          return session_run_hook.SessionRunArgs(fetches=stage_1_0)

        def after_run(self, run_context, run_values):
          # [FUNCTION DOC]
          # HELPER: after-run callback. Asserts the variable is near 1.5 and, after running the
          # last stage, near 2.6. Returns nothing.
          self._testing.assertNear(0.3 + 0.5 + 0.7,
                                   run_context.session.run(var), 0.1)
          self._testing.assertNear(0.3 + 0.5 + 0.7 + 1.1,
                                   run_context.session.run(stage_2), 0.1)

      def step_fn(step_context):
        # [FUNCTION DOC]
        # HELPER: step function. Runs the first stage directly (asserts near 0.3) and returns the
        # result of running the second stage with hooks.
        self.assertNear(0.3, step_context.session.run(stage_0), 0.1)
        return step_context.run_with_hooks(fetches=stage_1_1)

      with monitored_session.MonitoredSession(hooks=[Hook(self)]) as session:
        self.assertEqual(0.3 + 0.5 + 0.7, session.run_step_fn(step_fn))

  def test_step_fn_has_the_same_hooks_behavior_without_recovery(self):
    # [FUNCTION DOC]
    # Test: the same hook ordering scenario as the previous test, but using a singular monitored
    # session (no recovery). Asserts the same returned total.
    with ops.Graph().as_default():
      var = resource_variable_ops.ResourceVariable(0.0)

      stage_0 = state_ops.assign_add(var, 0.3)
      stage_1_0 = state_ops.assign_add(var, 0.7)
      with ops.control_dependencies([stage_1_0]):
        stage_1_1 = state_ops.assign_add(var, 0.5)
      stage_2 = state_ops.assign_add(var, 1.1)

      class Hook(session_run_hook.SessionRunHook):
        # [CLASS DOC]
        # HELPER: hook adding a fetch before the run and asserting cumulative values after.

        def __init__(self, testing):
          # [FUNCTION DOC]
          # HELPER: stores the test case used for assertions.
          self._testing = testing

        def before_run(self, run_context):
          # [FUNCTION DOC]
          # HELPER: before-run callback returning a request fetching the first stage op.
          return session_run_hook.SessionRunArgs(fetches=stage_1_0)

        def after_run(self, run_context, run_values):
          # [FUNCTION DOC]
          # HELPER: after-run callback asserting variable values near 1.5 and then 2.6.
          self._testing.assertNear(0.3 + 0.5 + 0.7,
                                   run_context.session.run(var), 0.1)
          self._testing.assertNear(0.3 + 0.5 + 0.7 + 1.1,
                                   run_context.session.run(stage_2), 0.1)

      def step_fn(step_context):
        # [FUNCTION DOC]
        # HELPER: step function running one stage directly and another with hooks.
        self.assertNear(0.3, step_context.session.run(stage_0), 0.1)
        return step_context.run_with_hooks(fetches=stage_1_1)

      with monitored_session.SingularMonitoredSession(
          hooks=[Hook(self)]) as session:
        self.assertEqual(0.3 + 0.5 + 0.7, session.run_step_fn(step_fn))

  def test_step_fn_with_hooks_and_request_stop(self):
    # [FUNCTION DOC]
    # Test: when a step function requests a stop, the hooks are not run. Asserts the session
    # stops, the step function returns nothing and neither hook callback was called.
    with ops.Graph().as_default():
      trace_the_hook = {'before_run': False, 'after_run': False}

      class Hook(session_run_hook.SessionRunHook):
        # [CLASS DOC]
        # HELPER: hook that records into a shared mapping that its before-run and after-run
        # callbacks were called.

        def before_run(self, run_context):
          # [FUNCTION DOC]
          # HELPER: marks the before-run entry as called.
          trace_the_hook['before_run'] = True

        def after_run(self, run_context, run_values):
          # [FUNCTION DOC]
          # HELPER: marks the after-run entry as called.
          trace_the_hook['after_run'] = True

      def step_fn(step_context):
        # [FUNCTION DOC]
        # HELPER: step function that requests a stop.
        step_context.request_stop()

      with monitored_session.MonitoredSession(hooks=[Hook()]) as session:
        self.assertEqual(None, session.run_step_fn(step_fn))
        self.assertTrue(session.should_stop())
        # `step_context.request_stop()` in a step_fn interrupts the flow of
        # running the hooks.
        self.assertFalse(trace_the_hook['before_run'])
        self.assertFalse(trace_the_hook['after_run'])

  def test_recovers_from_an_exception_in_step_fn(self):
    # [FUNCTION DOC]
    # Test: a step function raising an aborted error the first time is retried by the session.
    # Asserts the final value (near 3.2) and that the failure path ran.
    trace_the_exception = {'run_already': False}

    with ops.Graph().as_default():
      c = array_ops.placeholder(dtypes.float32)
      v = array_ops.identity(c)

      def step_fn(step_context):
        # [FUNCTION DOC]
        # HELPER: step function that raises an aborted error on its first call, and afterwards
        # returns the result of running the fetch with hooks.
        if not trace_the_exception['run_already']:
          trace_the_exception['run_already'] = True
          raise errors_impl.AbortedError(None, None, 'Abort')

        return step_context.run_with_hooks(fetches=v, feed_dict={c: 3.2})

      with monitored_session.MonitoredSession() as session:
        self.assertNear(3.2, session.run_step_fn(step_fn), 0.1)
      self.assertTrue(trace_the_exception['run_already'])

  def test_recovers_from_an_exception_in_step_fn_after_hooks(self):
    # [FUNCTION DOC]
    # Test: when a step function fails after running with hooks, the whole function is run again
    # after recovery. Asserts the value near 3.2, two executions counted, and the side effect on
    # a variable applied twice (about 0.62).
    trace_the_exception = {'run_already': False, 'side_effect_counter': 0}

    with ops.Graph().as_default():
      c = array_ops.placeholder(dtypes.float32)
      v = array_ops.identity(c)
      graph_state = variable_v1.VariableV1(0.0)
      graph_side_effect = state_ops.assign_add(graph_state, 0.31)

      def step_fn(step_context):
        # [FUNCTION DOC]
        # HELPER: step function. Counts its executions, applies a side effect to a variable, runs
        # the fetch with hooks, raises an aborted error on the first execution only, and
        # otherwise returns the fetched value.
        trace_the_exception['side_effect_counter'] += 1
        step_context.session.run(graph_side_effect)

        value = step_context.run_with_hooks(fetches=v, feed_dict={c: 3.2})

        if not trace_the_exception['run_already']:
          trace_the_exception['run_already'] = True
          raise errors_impl.AbortedError(None, None, 'Abort')

        return value

      with self.cached_session() as test_session:
        with monitored_session.MonitoredSession(
            CountingSessionCreator(test_session)) as session:
          session.run(variables.global_variables_initializer())

          self.assertNear(3.2, session.run_step_fn(step_fn), 0.1)
          self.assertTrue(trace_the_exception['run_already'])
          # Make sure the rest of the body of the step_fn is re-executed upon
          # AbortedError:
          self.assertEqual(2, trace_the_exception['side_effect_counter'])
          self.assertNear(0.62, session.run(graph_state), 0.1)

  def test_step_fn_doesnt_recover_when_it_wasnt_asked_to(self):
    # [FUNCTION DOC]
    # Test: a singular monitored session does not recover from an aborted error in the step
    # function. Asserts the aborted error propagates and that the step function had run.
    trace_the_exception = {'run_already': False}

    with ops.Graph().as_default():
      c = array_ops.placeholder(dtypes.float32)
      v = array_ops.identity(c)

      def step_fn(step_context):
        # [FUNCTION DOC]
        # HELPER: step function that raises an aborted error on its first call and otherwise
        # runs the fetch with hooks and returns the value.
        if not trace_the_exception['run_already']:
          trace_the_exception['run_already'] = True
          raise errors_impl.AbortedError(None, None, 'Abort')

        value = step_context.run_with_hooks(fetches=v, feed_dict={c: 3.2})
        return value

      with monitored_session.SingularMonitoredSession() as session:
        with self.assertRaisesRegex(errors_impl.AbortedError, 'Abort'):
          self.assertNear(3.2, session.run_step_fn(step_fn), 0.1)
          self.fail()

      self.assertTrue(trace_the_exception['run_already'])

  def test_step_fn_exception_from_before_run(self):
    # [FUNCTION DOC]
    # Test: an aborted error raised from a hook's before-run callback during a step function is
    # recovered. Asserts the step result near 1.3, the step function executed twice and the side
    # effect applied twice (about 0.62).
    trace_the_exception = {'run_already': False, 'side_effect_counter': 0}

    with ops.Graph().as_default():
      c = array_ops.placeholder(dtypes.float32)
      v = array_ops.identity(c)
      vv = constant_op.constant(3.2)
      graph_state = variable_v1.VariableV1(0.0)
      graph_side_effect = state_ops.assign_add(graph_state, 0.31)

      class Hook(session_run_hook.SessionRunHook):
        # [CLASS DOC]
        # HELPER: hook whose before-run raises an aborted error the first time, and later
        # requests a constant fetch; after-run checks the fetched constant is near 3.2.

        def __init__(self, testing):
          # [FUNCTION DOC]
          # HELPER: stores the test case used for assertions.
          self._testing = testing

        def before_run(self, run_context):
          # [FUNCTION DOC]
          # HELPER: raises an aborted error on first call; later returns a request for a constant.
          if not trace_the_exception['run_already']:
            trace_the_exception['run_already'] = True
            raise errors_impl.AbortedError(None, None, 'Abort')
          return session_run_hook.SessionRunArgs(fetches=vv)

        def after_run(self, run_context, run_values):
          # [FUNCTION DOC]
          # HELPER: asserts that the hook's fetched result is near 3.2.
          self._testing.assertNear(3.2, run_values.results, 0.1)

      def step_fn(step_context):
        # [FUNCTION DOC]
        # HELPER: step function. Counts executions, applies a graph side effect, then returns the
        # result of running a fetch with a fed value through the hooks.
        trace_the_exception['side_effect_counter'] += 1
        step_context.session.run(graph_side_effect)
        return step_context.run_with_hooks(fetches=v, feed_dict={c: 1.3})

      with self.cached_session() as test_session:
        with monitored_session.MonitoredSession(
            CountingSessionCreator(test_session),
            hooks=[Hook(self)]) as session:
          test_session.run(variables.global_variables_initializer())
          self.assertNear(1.3, session.run_step_fn(step_fn), 0.1)
          self.assertEqual(2, trace_the_exception['side_effect_counter'])
          self.assertNear(0.62, session.run(graph_state), 0.1)


class SingularMonitoredSessionTest(test.TestCase):
  # [CLASS DOC]
  # Test case for SingularMonitoredSession, the variant without automatic recovery: initialization,
  # aborted-error handling, clean exit on out-of-range errors, exception reporting through the
  # coordinator, closing behavior, graph and raw session access.
  """Tests SingularMonitoredSession."""

  def test_handles_initialization(self):
    # [FUNCTION DOC]
    # Test: variables are initialized when the singular session starts; the variable reads zero.
    with ops.Graph().as_default():
      a_var = variable_v1.VariableV1(0)
      with monitored_session.SingularMonitoredSession() as session:
        # If it's not initialized, following statement raises an error.
        self.assertEqual(0, session.run(a_var))

  def test_do_not_handle_aborted_error(self):
    # [FUNCTION DOC]
    # Test: a hook raising an aborted error is not handled by the singular session. Asserts the
    # error propagates both from a run inside the with block and from entering the block.
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()

      class _RaiseAbortedHook(session_run_hook.SessionRunHook):
        # [CLASS DOC]
        # HELPER: hook whose before-run callback always raises an aborted error.

        def before_run(self, run_context):
          # [FUNCTION DOC]
          # HELPER: raises an aborted error. Input run_context is unused.
          raise errors_impl.AbortedError(None, None, 'Abort')

      with monitored_session.SingularMonitoredSession(
          hooks=[_RaiseAbortedHook()]) as session:
        with self.assertRaises(errors_impl.AbortedError):
          self.assertEqual(0, session.run(gstep))

      with self.assertRaises(errors_impl.AbortedError):
        with monitored_session.SingularMonitoredSession(
            hooks=[_RaiseAbortedHook()]) as session:
          self.assertEqual(0, session.run(gstep))

  def test_exit_cleanly_on_out_of_range_exception(self):
    # [FUNCTION DOC]
    # Test: an out-of-range error from a hook ends the with block cleanly in a singular session;
    # code after the failing run is not executed and should-stop is true.
    # Tests that we stop cleanly when OutOfRange is raised.
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      hook = RaiseOnceAtCountN(2, errors_impl.OutOfRangeError(None, None,
                                                              'EOI'))
      session = monitored_session.SingularMonitoredSession(hooks=[hook])
      # session should cleanly exit from the context.
      with session:
        self.assertEqual(0, session.run(gstep))
        self.assertFalse(session.should_stop())
        # Here at step 1, the hook triggers and raises OutOfRange. The
        # session should go into should_stop() mode. It should raise the
        # exception. So next step should not be executed.
        session.run(do_step)
        self.assertTrue(False)
      self.assertTrue(session.should_stop())

  def test_regular_exception_reported_to_coord_pass_through_run(self):
    # [FUNCTION DOC]
    # Test: an error reported to the coordinator from a thread is raised when leaving the with
    # block even though a later run succeeds in a singular session.
    # Tests that regular exceptions reported to the coordinator from a thread
    # passes through a "run()" call within a "with MonitoredSession" block and
    # set the session in stop mode.
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      session = monitored_session.SingularMonitoredSession()
      run_performed_without_error = False
      with self.assertRaisesRegex(RuntimeError, 'a thread wants to stop'):
        with session:
          self.assertEqual(0, session.run(gstep))
          # Report an exception through the coordinator.
          try:
            raise RuntimeError('a thread wants to stop')
          except RuntimeError as e:
            session._coordinated_creator.coord.request_stop(e)
          # Call run() which should perform normally.
          self.assertEqual(0, session.run(gstep))
          run_performed_without_error = True
      self.assertTrue(run_performed_without_error)

  def test_stop_cleanly_when_no_exception_in_with_body(self):
    # [FUNCTION DOC]
    # Test: no error in the with body. Runs two steps, then after exit should-stop is true and
    # the raw session accessor returns null.
    # Tests that regular exceptions pass through
    with ops.Graph().as_default():
      gstep = training_util.get_or_create_global_step()
      do_step = state_ops.assign_add(gstep, 1)
      session = monitored_session.SingularMonitoredSession()
      with session:
        self.assertEqual(1, session.run(do_step))
        self.assertEqual(2, session.run(do_step))
        self.assertFalse(session.should_stop())
      # Should have closed.
      self.assertTrue(session.should_stop())
      self.assertEqual(None, session.raw_session())

  def test_graph(self):
    # [FUNCTION DOC]
    # Test: the singular session's graph equals the default graph it was created in.
    with ops.Graph().as_default() as g:
      with monitored_session.SingularMonitoredSession() as session:
        self.assertEqual(g, session.graph)

  def test_raw_session(self):
    # [FUNCTION DOC]
    # Test: the raw session accessor returns a session object of the standard session type.
    with ops.Graph().as_default():
      with monitored_session.SingularMonitoredSession() as session:
        self.assertTrue(isinstance(session.raw_session(), session_lib.Session))


if __name__ == '__main__':
  test.main()
