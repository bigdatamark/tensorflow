# [MODULE DOC]
# Tests for checkpoint-state file handling and the checkpoint manager: relative/absolute path
# handling when generating, writing and reading the checkpoint-state file, latest-checkpoint
# lookup, existence, modification times and removal of checkpoints written by the V1 saver, and
# CheckpointManager behavior (deletion, keep-all, persisted state, hourly preservation, clock
# reversal, custom numbering and names, init callback, path-like directories, step intervals).
# Test module; run via the test main entry.
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
# =============================================================================
"""Tests for tensorflow.python.training.saver.py."""

import contextlib
import os
import pathlib
import shutil
import tempfile

from google.protobuf import text_format

from tensorflow.core.protobuf import saver_pb2
from tensorflow.python.checkpoint import checkpoint as util
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import saver as saver_module
from tensorflow.python.training.checkpoint_state_pb2 import CheckpointState


class LatestCheckpointWithRelativePaths(test.TestCase):
  # [CLASS DOC]
  # Tests that the V1 saver and latest-checkpoint lookup work with relative save paths and
  # avoid colliding with the checkpoint-state file name. Provides two static context-manager
  # helpers for temporary working/scratch directories.

  @staticmethod
  @contextlib.contextmanager
  def tempWorkingDir(temppath):
    # [FUNCTION DOC]
    # HELPER: test context manager that changes the working directory to the given path for the
    # block and restores the previous directory afterwards (even on error). Yields nothing.
    cwd = os.getcwd()
    os.chdir(temppath)
    try:
      yield
    finally:
      os.chdir(cwd)

  @staticmethod
  @contextlib.contextmanager
  def tempDir():
    # [FUNCTION DOC]
    # HELPER: test context manager that creates a temporary directory, yields its path, and
    # removes it recursively when the block ends (even on error).
    tempdir = tempfile.mkdtemp()
    try:
      yield tempdir
    finally:
      shutil.rmtree(tempdir)

  @test_util.run_deprecated_v1
  def testNameCollision(self):
    # [FUNCTION DOC]
    # Scenario: inside a temporary working directory, save with a V1 saver to a relative path
    # named like the state file. Saving without a step must fail with a value error mentioning a
    # collision; saving with a global step, with a sharded saver, and sharded with a step must
    # succeed, and latest-checkpoint lookup of the training directory must find a checkpoint.
    # Make sure we have a clean directory to work in.
    with self.tempDir() as tempdir:
      # Jump to that directory until this test is done.
      with self.tempWorkingDir(tempdir):
        # Save training snapshots to a relative path.
        traindir = "train"
        os.mkdir(traindir)
        # Collides with the default name of the checkpoint state file.
        filepath = os.path.join(traindir, "checkpoint")

        with self.cached_session() as sess:
          unused_a = variables.Variable(0.0)  # So that Saver saves something.
          self.evaluate(variables.global_variables_initializer())

          # Should fail.
          saver = saver_module.Saver(sharded=False)
          with self.assertRaisesRegex(ValueError, "collides with"):
            saver.save(sess, filepath)

          # Succeeds: the file will be named "checkpoint-<step>".
          saver.save(sess, filepath, global_step=1)
          self.assertIsNotNone(
              checkpoint_management.latest_checkpoint(traindir))

          # Succeeds: the file will be named "checkpoint-<i>-of-<n>".
          saver = saver_module.Saver(sharded=True)
          saver.save(sess, filepath)
          self.assertIsNotNone(
              checkpoint_management.latest_checkpoint(traindir))

          # Succeeds: the file will be named "checkpoint-<step>-<i>-of-<n>".
          saver = saver_module.Saver(sharded=True)
          saver.save(sess, filepath, global_step=1)
          self.assertIsNotNone(
              checkpoint_management.latest_checkpoint(traindir))

  @test_util.run_deprecated_v1
  def testRelativePath(self):
    # [FUNCTION DOC]
    # Scenario: save three snapshots (variable values 0, 1, 2) with a V1 saver to a relative
    # path in a temporary directory. In a fresh graph with a different initial value, look up the
    # latest checkpoint through the relative directory, restore it, and assert the variable equals 2.
    # Make sure we have a clean directory to work in.
    with self.tempDir() as tempdir:

      # Jump to that directory until this test is done.
      with self.tempWorkingDir(tempdir):

        # Save training snapshots to a relative path.
        traindir = "train"
        os.mkdir(traindir)

        filename = "snapshot"
        filepath = os.path.join(traindir, filename)

        with self.cached_session() as sess:
          # Build a simple graph.
          v0 = variables.Variable(0.0)
          inc = v0.assign_add(1.0)

          save = saver_module.Saver({"v0": v0})

          # Record a short training history.
          self.evaluate(variables.global_variables_initializer())
          save.save(sess, filepath, global_step=0)
          self.evaluate(inc)
          save.save(sess, filepath, global_step=1)
          self.evaluate(inc)
          save.save(sess, filepath, global_step=2)

        with self.cached_session() as sess:
          # Build a new graph with different initialization.
          v0 = variables.Variable(-1.0)

          # Create a new saver.
          save = saver_module.Saver({"v0": v0})
          self.evaluate(variables.global_variables_initializer())

          # Get the most recent checkpoint name from the training history file.
          name = checkpoint_management.latest_checkpoint(traindir)
          self.assertIsNotNone(name)

          # Restore "v0" from that checkpoint.
          save.restore(sess, name)
          self.assertEqual(self.evaluate(v0), 2.0)


class CheckpointStateTest(test.TestCase):
  # [CLASS DOC]
  # Tests for generating, writing and reading the checkpoint-state message with absolute,
  # relative and path-like directories, plus failure on incomplete state files.

  def _get_test_dir(self, dirname):
    # [FUNCTION DOC]
    # HELPER: creates (if needed) and returns a subdirectory with the given name under the test
    # temporary directory.
    test_dir = os.path.join(self.get_temp_dir(), dirname)
    gfile.MakeDirs(test_dir)
    return test_dir

  def testAbsPath(self):
    # [FUNCTION DOC]
    # Scenario: generate a state message with an absolute checkpoint path. Expects the path to stay
    # absolute and equal to the input, and the all-paths list to hold exactly that one path.
    save_dir = self._get_test_dir("abs_paths")
    abs_path = os.path.join(save_dir, "model-0")
    ckpt = checkpoint_management.generate_checkpoint_state_proto(
        save_dir, abs_path)
    self.assertEqual(ckpt.model_checkpoint_path, abs_path)
    self.assertTrue(os.path.isabs(ckpt.model_checkpoint_path))
    self.assertEqual(len(ckpt.all_model_checkpoint_paths), 1)
    self.assertEqual(ckpt.all_model_checkpoint_paths[-1], abs_path)

  def testRelPath(self):
    # [FUNCTION DOC]
    # Scenario: generate a state message for a relative save directory and a path inside it.
    # Expects the stored path to be rewritten relative to the directory (without the directory
    # part), and the all-paths list to contain only that path.
    train_dir = "train"
    model = os.path.join(train_dir, "model-0")
    # model_checkpoint_path should have no "train" directory part.
    new_rel_path = "model-0"
    ckpt = checkpoint_management.generate_checkpoint_state_proto(
        train_dir, model)
    self.assertEqual(ckpt.model_checkpoint_path, new_rel_path)
    self.assertEqual(len(ckpt.all_model_checkpoint_paths), 1)
    self.assertEqual(ckpt.all_model_checkpoint_paths[-1], new_rel_path)

  def testAllModelCheckpointPaths(self):
    # [FUNCTION DOC]
    # Scenario: generate state with the all-paths argument null, empty, and a list with a different
    # path. Expects the model path to stay absolute, the newest path to be last in the list, and
    # the list length to be the input length or one when input was null/empty (the model path is
    # appended when missing).
    save_dir = self._get_test_dir("all_models_test")
    abs_path = os.path.join(save_dir, "model-0")
    for paths in [None, [], ["model-2"]]:
      ckpt = checkpoint_management.generate_checkpoint_state_proto(
          save_dir, abs_path, all_model_checkpoint_paths=paths)
      self.assertEqual(ckpt.model_checkpoint_path, abs_path)
      self.assertTrue(os.path.isabs(ckpt.model_checkpoint_path))
      self.assertEqual(
          len(ckpt.all_model_checkpoint_paths), len(paths) if paths else 1)
      self.assertEqual(ckpt.all_model_checkpoint_paths[-1], abs_path)

  def testUpdateCheckpointState(self):
    # [FUNCTION DOC]
    # Scenario: from inside a temporary directory with a relative train directory, write state with
    # the deprecated updater using one absolute and one relative path, then read it back. Expects
    # the relative model path, two listed paths, with the absolute one first and relative last.
    save_dir = self._get_test_dir("update_checkpoint_state")
    os.chdir(save_dir)
    # Make a temporary train directory.
    train_dir = "train"
    os.mkdir(train_dir)
    abs_path = os.path.join(save_dir, "model-0")
    rel_path = os.path.join("train", "model-2")
    checkpoint_management.update_checkpoint_state(
        train_dir, rel_path, all_model_checkpoint_paths=[abs_path, rel_path])
    ckpt = checkpoint_management.get_checkpoint_state(train_dir)
    self.assertEqual(ckpt.model_checkpoint_path, rel_path)
    self.assertEqual(len(ckpt.all_model_checkpoint_paths), 2)
    self.assertEqual(ckpt.all_model_checkpoint_paths[-1], rel_path)
    self.assertEqual(ckpt.all_model_checkpoint_paths[0], abs_path)

  def testFSPath(self):
    # [FUNCTION DOC]
    # Scenario: same as updating state with mixed paths, but reading the state through a path-like
    # (not string) directory object. Expects the model checkpoint path to match the relative path.
    save_dir = self._get_test_dir("fspath")
    os.chdir(save_dir)
    # Make a temporary train directory.
    train_dir = "train"
    os.mkdir(train_dir)
    abs_path = os.path.join(save_dir, "model-0")
    rel_path = os.path.join("train", "model-2")
    checkpoint_management.update_checkpoint_state(
        train_dir, rel_path, all_model_checkpoint_paths=[abs_path, rel_path])
    ckpt = checkpoint_management.get_checkpoint_state(pathlib.Path(train_dir))
    self.assertEqual(ckpt.model_checkpoint_path, rel_path)

  def testUpdateCheckpointStateSaveRelativePaths(self):
    # [FUNCTION DOC]
    # Scenario: write state with the internal updater and relative-path saving, giving a mix of
    # relative and absolute paths. Expects the file on disk to hold only relative paths (two
    # entries, newest last), while reading through the state getter returns absolute paths.
    save_dir = self._get_test_dir("update_checkpoint_state")
    os.chdir(save_dir)
    abs_path2 = os.path.join(save_dir, "model-2")
    rel_path2 = "model-2"
    abs_path0 = os.path.join(save_dir, "model-0")
    rel_path0 = "model-0"
    checkpoint_management.update_checkpoint_state_internal(
        save_dir=save_dir,
        model_checkpoint_path=abs_path2,
        all_model_checkpoint_paths=[rel_path0, abs_path2],
        save_relative_paths=True)

    # File should contain relative paths.
    file_content = file_io.read_file_to_string(
        os.path.join(save_dir, "checkpoint"))
    ckpt = CheckpointState()
    text_format.Merge(file_content, ckpt)
    self.assertEqual(ckpt.model_checkpoint_path, rel_path2)
    self.assertEqual(len(ckpt.all_model_checkpoint_paths), 2)
    self.assertEqual(ckpt.all_model_checkpoint_paths[-1], rel_path2)
    self.assertEqual(ckpt.all_model_checkpoint_paths[0], rel_path0)

    # get_checkpoint_state should return absolute paths.
    ckpt = checkpoint_management.get_checkpoint_state(save_dir)
    self.assertEqual(ckpt.model_checkpoint_path, abs_path2)
    self.assertEqual(len(ckpt.all_model_checkpoint_paths), 2)
    self.assertEqual(ckpt.all_model_checkpoint_paths[-1], abs_path2)
    self.assertEqual(ckpt.all_model_checkpoint_paths[0], abs_path0)

  def testCheckPointStateFailsWhenIncomplete(self):
    # [FUNCTION DOC]
    # Scenario: create an empty state file and read the state. Expects a value error because the
    # model checkpoint path is missing.
    save_dir = self._get_test_dir("checkpoint_state_fails_when_incomplete")
    os.chdir(save_dir)
    ckpt_path = os.path.join(save_dir, "checkpoint")
    ckpt_file = open(ckpt_path, "w")
    ckpt_file.write("")
    ckpt_file.close()
    with self.assertRaises(ValueError):
      checkpoint_management.get_checkpoint_state(save_dir)

  def testCheckPointCompletesRelativePaths(self):
    # [FUNCTION DOC]
    # Scenario: write a state file with relative "./" paths (one model path and two listed paths)
    # and read it. Expects each path to be joined to the directory, in the original order.
    save_dir = self._get_test_dir("checkpoint_completes_relative_paths")
    os.chdir(save_dir)
    ckpt_path = os.path.join(save_dir, "checkpoint")
    ckpt_file = open(ckpt_path, "w")
    ckpt_file.write("""
        model_checkpoint_path: "./model.ckpt-687529"
        all_model_checkpoint_paths: "./model.ckpt-687500"
        all_model_checkpoint_paths: "./model.ckpt-687529"
        """)
    ckpt_file.close()
    ckpt = checkpoint_management.get_checkpoint_state(save_dir)
    self.assertEqual(ckpt.model_checkpoint_path,
                     os.path.join(save_dir, "./model.ckpt-687529"))
    self.assertEqual(ckpt.all_model_checkpoint_paths[0],
                     os.path.join(save_dir, "./model.ckpt-687500"))
    self.assertEqual(ckpt.all_model_checkpoint_paths[1],
                     os.path.join(save_dir, "./model.ckpt-687529"))


class SaverUtilsTest(test.TestCase):
  # [CLASS DOC]
  # Tests for checkpoint existence, modification-time and removal utilities against checkpoints
  # written by the V1 saver in both format versions. Fixture: a base directory under the test
  # temporary directory, created before and recursively deleted after each test.

  def setUp(self):
    # [FUNCTION DOC]
    # Creates the base directory used by the tests under the test temporary directory.
    self._base_dir = os.path.join(self.get_temp_dir(), "saver_utils_test")
    gfile.MakeDirs(self._base_dir)

  def tearDown(self):
    # [FUNCTION DOC]
    # Recursively deletes the base directory created in setup.
    gfile.DeleteRecursively(self._base_dir)

  @test_util.run_deprecated_v1
  def testCheckpointExists(self):
    # [FUNCTION DOC]
    # Scenario: for sharded/non-sharded savers and V1/V2 formats, expects no checkpoint before
    # saving, existence after saving by returned prefix, and existence of the prefix found by the
    # latest-checkpoint lookup.
    for sharded in (False, True):
      for version in (saver_pb2.SaverDef.V2, saver_pb2.SaverDef.V1):
        with self.session(graph=ops_lib.Graph()) as sess:
          unused_v = variables.Variable(1.0, name="v")
          self.evaluate(variables.global_variables_initializer())
          saver = saver_module.Saver(sharded=sharded, write_version=version)

          path = os.path.join(self._base_dir, "%s-%s" % (sharded, version))
          self.assertFalse(
              checkpoint_management.checkpoint_exists(path))  # Not saved yet.

          ckpt_prefix = saver.save(sess, path)
          self.assertTrue(checkpoint_management.checkpoint_exists(ckpt_prefix))

          ckpt_prefix = checkpoint_management.latest_checkpoint(self._base_dir)
          self.assertTrue(checkpoint_management.checkpoint_exists(ckpt_prefix))

  @test_util.run_deprecated_v1
  def testGetCheckpointMtimes(self):
    # [FUNCTION DOC]
    # Scenario: save one V2 and one V1 checkpoint in order and get their modification times.
    # Expects two values with the second not earlier than the first.
    prefixes = []
    for version in (saver_pb2.SaverDef.V2, saver_pb2.SaverDef.V1):
      with self.session(graph=ops_lib.Graph()) as sess:
        unused_v = variables.Variable(1.0, name="v")
        self.evaluate(variables.global_variables_initializer())
        saver = saver_module.Saver(write_version=version)
        prefixes.append(
            saver.save(sess, os.path.join(self._base_dir, str(version))))

    mtimes = checkpoint_management.get_checkpoint_mtimes(prefixes)
    self.assertEqual(2, len(mtimes))
    self.assertTrue(mtimes[1] >= mtimes[0])

  @test_util.run_deprecated_v1
  def testRemoveCheckpoint(self):
    # [FUNCTION DOC]
    # Scenario: for each sharded/format combination, save, confirm it exists, remove it with the
    # matching format version, and expect it no longer exists.
    for sharded in (False, True):
      for version in (saver_pb2.SaverDef.V2, saver_pb2.SaverDef.V1):
        with self.session(graph=ops_lib.Graph()) as sess:
          unused_v = variables.Variable(1.0, name="v")
          self.evaluate(variables.global_variables_initializer())
          saver = saver_module.Saver(sharded=sharded, write_version=version)

          path = os.path.join(self._base_dir, "%s-%s" % (sharded, version))
          ckpt_prefix = saver.save(sess, path)
          self.assertTrue(checkpoint_management.checkpoint_exists(ckpt_prefix))
          checkpoint_management.remove_checkpoint(ckpt_prefix, version)
          self.assertFalse(checkpoint_management.checkpoint_exists(ckpt_prefix))


class CheckpointManagerTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the checkpoint manager covering deletion, retention, persisted state, clock
  # reversal, numbering, naming, initialization, path-like directories and step intervals, in
  # graph and eager modes.

  @test_util.run_in_graph_and_eager_modes
  def testDeletion(self):
    # [FUNCTION DOC]
    # Scenario: manager keeping three checkpoints saves four times. Expects the latest three exist
    # and the first was deleted.
    checkpoint = util.Checkpoint()
    manager = checkpoint_management.CheckpointManager(
        checkpoint, self.get_temp_dir(), max_to_keep=3)
    first_path = manager.save()
    second_path = manager.save()
    third_path = manager.save()
    fourth_path = manager.save()
    self.assertTrue(checkpoint_management.checkpoint_exists(fourth_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(third_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(second_path))
    self.assertFalse(checkpoint_management.checkpoint_exists(first_path))

  @test_util.run_in_graph_and_eager_modes
  def testKeepAll(self):
    # [FUNCTION DOC]
    # Scenario: manager with no limit saves three checkpoints (all exist; latest and list are
    # correct), a new manager continues and lists four, and a third manager with limit three still
    # lists four and keeps them until it saves a fifth, after which only the newest three are
    # listed and the two oldest are deleted.
    checkpoint = util.Checkpoint()
    directory = os.path.join(
        self.get_temp_dir(),
        # Avoid sharing directories between eager and graph
        # TODO(allenl): stop run_in_graph_and_eager_modes reusing directories
        str(context.executing_eagerly()))
    manager = checkpoint_management.CheckpointManager(
        checkpoint, directory, max_to_keep=None)
    first_path = manager.save()
    second_path = manager.save()
    third_path = manager.save()
    self.assertTrue(checkpoint_management.checkpoint_exists(third_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(second_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(first_path))
    self.assertEqual(third_path, manager.latest_checkpoint)
    self.assertEqual([first_path, second_path, third_path],
                     manager.checkpoints)
    del manager
    manager = checkpoint_management.CheckpointManager(
        checkpoint, directory, max_to_keep=None)
    fourth_path = manager.save()
    self.assertEqual([first_path, second_path, third_path, fourth_path],
                     manager.checkpoints)
    del manager
    manager = checkpoint_management.CheckpointManager(
        checkpoint, directory, max_to_keep=3)
    self.assertEqual([first_path, second_path, third_path, fourth_path],
                     manager.checkpoints)
    self.assertTrue(checkpoint_management.checkpoint_exists(fourth_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(third_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(second_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(first_path))
    fifth_path = manager.save()
    self.assertEqual([third_path, fourth_path, fifth_path],
                     manager.checkpoints)
    self.assertTrue(checkpoint_management.checkpoint_exists(fifth_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(fourth_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(third_path))
    self.assertFalse(checkpoint_management.checkpoint_exists(second_path))
    self.assertFalse(checkpoint_management.checkpoint_exists(first_path))

  @test_util.run_in_graph_and_eager_modes
  @test.mock.patch.object(checkpoint_management, "time")
  def testSaveRestoreState(self, mock_time):
    # [FUNCTION DOC]
    # Scenario: with a mocked clock, a manager with limit two saves twice and the state file holds
    # both timestamps. A second manager (hourly preservation of 1.5 hours) recovers the list and
    # latest, then saves at later times: the oldest checkpoint is preserved once enough time has
    # passed since the last preserved timestamp, others are deleted when pushed out. A third
    # manager recovers state, saves again, and the preserved timestamp, existing files and managed
    # lists are asserted at each step.
    directory = self.get_temp_dir()
    mock_time.time.return_value = 3.
    checkpoint = util.Checkpoint()
    first_manager = checkpoint_management.CheckpointManager(
        checkpoint, directory, max_to_keep=2)
    first_time = 10000.
    first_name = os.path.join(directory, "ckpt-1")
    mock_time.time.return_value = first_time
    first_manager.save()
    state = checkpoint_management.get_checkpoint_state(directory)
    second_time = first_time + 3610.
    second_name = os.path.join(directory, "ckpt-2")
    mock_time.time.return_value = second_time
    first_manager.save()
    state = checkpoint_management.get_checkpoint_state(directory)
    self.assertEqual([first_time, second_time],
                     state.all_model_checkpoint_timestamps)
    self.assertEqual([first_name, second_name], first_manager.checkpoints)
    self.assertEqual(second_name, first_manager.latest_checkpoint)
    del first_manager

    second_manager = checkpoint_management.CheckpointManager(
        checkpoint, directory,
        max_to_keep=2, keep_checkpoint_every_n_hours=1.5)
    self.assertEqual([first_name, second_name], second_manager.checkpoints)
    self.assertEqual(second_name, second_manager.latest_checkpoint)
    third_name = os.path.join(directory, "ckpt-3")
    third_time = second_time + 3600. * 0.2
    mock_time.time.return_value = third_time
    second_manager.save()
    self.assertTrue(checkpoint_management.checkpoint_exists(first_name))
    self.assertTrue(checkpoint_management.checkpoint_exists(second_name))
    self.assertEqual([second_name, third_name],
                     second_manager.checkpoints)
    state = checkpoint_management.get_checkpoint_state(directory)
    self.assertEqual(first_time, state.last_preserved_timestamp)
    fourth_time = third_time + 3600. * 0.5
    mock_time.time.return_value = fourth_time
    fourth_name = os.path.join(directory, "ckpt-4")
    second_manager.save()
    self.assertTrue(checkpoint_management.checkpoint_exists(first_name))
    self.assertFalse(checkpoint_management.checkpoint_exists(second_name))
    self.assertEqual([third_name, fourth_name],
                     second_manager.checkpoints)
    fifth_time = fourth_time + 3600. * 0.5
    mock_time.time.return_value = fifth_time
    fifth_name = os.path.join(directory, "ckpt-5")
    second_manager.save()
    self.assertEqual([fourth_name, fifth_name],
                     second_manager.checkpoints)
    state = checkpoint_management.get_checkpoint_state(directory)
    self.assertEqual(first_time, state.last_preserved_timestamp)
    del second_manager
    third_manager = checkpoint_management.CheckpointManager(
        checkpoint, directory,
        max_to_keep=2, keep_checkpoint_every_n_hours=1.5)
    self.assertEqual(fifth_name, third_manager.latest_checkpoint)
    mock_time.time.return_value += 10.
    third_manager.save()
    sixth_name = os.path.join(directory, "ckpt-6")
    state = checkpoint_management.get_checkpoint_state(directory)
    self.assertEqual(fourth_time, state.last_preserved_timestamp)
    self.assertTrue(checkpoint_management.checkpoint_exists(first_name))
    self.assertTrue(checkpoint_management.checkpoint_exists(fourth_name))
    self.assertTrue(checkpoint_management.checkpoint_exists(fifth_name))
    self.assertTrue(checkpoint_management.checkpoint_exists(sixth_name))
    self.assertFalse(checkpoint_management.checkpoint_exists(second_name))
    self.assertFalse(checkpoint_management.checkpoint_exists(third_name))
    self.assertEqual([fifth_name, sixth_name],
                     third_manager.checkpoints)

  @test_util.run_in_graph_and_eager_modes
  def testContinueFromUnmanaged(self):
    # [FUNCTION DOC]
    # Scenario: two checkpoints are saved directly (not via a manager), then a manager with limit
    # two takes over the directory. Restoring its latest gives save counter two; manager saves are
    # tracked from scratch, the unmanaged checkpoints are never deleted, and only the newest two
    # managed ones remain.
    directory = self.get_temp_dir()
    prefix = os.path.join(directory, "unusual_prefix")
    checkpoint = util.Checkpoint()
    first_path = checkpoint.save(prefix)
    second_path = checkpoint.save(prefix)
    del checkpoint
    checkpoint = util.Checkpoint()
    manager = checkpoint_management.CheckpointManager(
        checkpoint, directory, max_to_keep=2)
    checkpoint.restore(manager.latest_checkpoint).run_restore_ops()
    self.assertEqual(2, self.evaluate(checkpoint.save_counter))
    third_path = manager.save()
    self.assertEqual([third_path], manager.checkpoints)
    fourth_path = manager.save()
    self.assertEqual([third_path, fourth_path],
                     manager.checkpoints)
    fifth_path = manager.save()
    self.assertEqual([fourth_path, fifth_path],
                     manager.checkpoints)
    self.assertTrue(checkpoint_management.checkpoint_exists(first_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(second_path))
    self.assertFalse(checkpoint_management.checkpoint_exists(third_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(fourth_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(fifth_path))

  @test_util.run_in_graph_and_eager_modes
  @test.mock.patch.object(checkpoint_management, "time")
  def testClockReset(self, mock_time):
    # [FUNCTION DOC]
    # Scenario: with a mocked clock and hourly preservation, a limit-one manager saves three times
    # an hour apart, preserving the middle checkpoint. The clock is then set back; a new manager
    # logs a warning about being behind the last preserved timestamp, keeps existing checkpoints
    # untracked, and later saves are tracked normally with the preserved timestamp reset to the
    # new (earlier) clock value and a single recorded timestamp.
    directory = self.get_temp_dir()
    mock_time.time.return_value = 10000.
    checkpoint = util.Checkpoint()
    first_manager = checkpoint_management.CheckpointManager(
        checkpoint, directory, max_to_keep=1, keep_checkpoint_every_n_hours=1.)
    first_path = first_manager.save()
    mock_time.time.return_value += 3600.
    second_path = first_manager.save()
    mock_time.time.return_value += 3600.
    third_path = first_manager.save()
    self.assertFalse(checkpoint_management.checkpoint_exists(first_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(second_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(third_path))
    self.assertEqual([third_path], first_manager.checkpoints)
    state = checkpoint_management.get_checkpoint_state(directory)
    self.assertEqual(13600., state.last_preserved_timestamp)
    # Set the clock back in time
    mock_time.time.return_value = 5000.
    del first_manager
    with test.mock.patch.object(logging, "warning") as mock_log:
      second_manager = checkpoint_management.CheckpointManager(
          checkpoint, directory, max_to_keep=1)
      self.assertRegex(
          str(mock_log.call_args),
          "behind the last preserved checkpoint timestamp")
    # We should err on the side of keeping checkpoints around when we're not
    # sure whether they were preserved or not due to clock funkiness.
    self.assertTrue(checkpoint_management.checkpoint_exists(second_path))
    # We know about the existing checkpoints, but they'll never be deleted and
    # so won't go in the CheckpointState proto on save.
    self.assertEqual(third_path, second_manager.latest_checkpoint)
    self.assertEqual([], second_manager.checkpoints)
    mock_time.time.return_value += 10.
    fourth_path = second_manager.save()
    self.assertTrue(checkpoint_management.checkpoint_exists(second_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(third_path))
    self.assertEqual(fourth_path, second_manager.latest_checkpoint)
    self.assertEqual([fourth_path], second_manager.checkpoints)
    mock_time.time.return_value += 10.
    fifth_path = second_manager.save()
    self.assertTrue(checkpoint_management.checkpoint_exists(second_path))
    self.assertTrue(checkpoint_management.checkpoint_exists(third_path))
    self.assertEqual([fifth_path], second_manager.checkpoints)
    state = checkpoint_management.get_checkpoint_state(directory)
    self.assertEqual(5000., state.last_preserved_timestamp)
    self.assertEqual([5020.],
                     state.all_model_checkpoint_timestamps)

  @test_util.run_in_graph_and_eager_modes
  def testCustomNumbering(self):
    # [FUNCTION DOC]
    # Scenario: save using a step variable as the checkpoint number five times, advancing it by
    # two each time: file suffixes follow the variable (0, 2, ...) while the save counter still
    # increments. Then save with a plain integer number; expects its suffix, that it is the latest,
    # and that only the last two checkpoints are recorded in the state.
    directory = self.get_temp_dir()
    step = variables.Variable(0, dtype=dtypes.int64)
    checkpoint = util.Checkpoint(step=step)
    manager = checkpoint_management.CheckpointManager(
        checkpoint, directory, max_to_keep=2)
    self.evaluate(step.initializer)
    for i in range(5):
      path = manager.save(checkpoint_number=step)
      expected_suffix = "-%d" % (2 * i,)
      if not path.endswith(expected_suffix):
        self.fail("%s should have suffix %s" % (path, expected_suffix))
      self.evaluate(step.assign_add(2))
    self.assertEqual(5, self.evaluate(checkpoint.save_counter))
    # Test regular integers
    last_path = manager.save(checkpoint_number=32)
    self.assertIn("-32", last_path)
    self.assertEqual(last_path, manager.latest_checkpoint)
    self.assertEqual(
        last_path, checkpoint_management.latest_checkpoint(directory))
    state = checkpoint_management.get_checkpoint_state(directory)
    # Only the most recent two checkpoints are saved
    self.assertEqual([path, last_path], state.all_model_checkpoint_paths)

  @test_util.run_in_graph_and_eager_modes
  def testCustomCheckpointPrefix(self):
    # [FUNCTION DOC]
    # Scenario: manager with a custom checkpoint name saves with number five and expects the file
    # base name to use that name; a manager with the default name expects the default prefix.
    directory = self.get_temp_dir()
    checkpoint = util.Checkpoint()
    manager = checkpoint_management.CheckpointManager(
        checkpoint, directory, max_to_keep=2, checkpoint_name="ckpt_name")
    path = manager.save(checkpoint_number=5)
    self.assertEqual(os.path.basename(path), "ckpt_name-5")
    manager = checkpoint_management.CheckpointManager(
        checkpoint, directory, max_to_keep=2)
    path = manager.save(checkpoint_number=5)
    self.assertEqual(os.path.basename(path), "ckpt-5")

  @test_util.run_in_graph_and_eager_modes
  def testRestoreOrInitialize(self):
    # [FUNCTION DOC]
    # Scenario: create an initial checkpoint holding a variable with value two; a manager with a
    # variable of value one and an initialization callback that restores that checkpoint. The first
    # restore-or-initialize returns nothing and leaves the variable at two (callback ran); after a
    # save, the next call restores and returns a path.
    directory = self.get_temp_dir()

    # Create a checkpoint for initializing.
    init_prefix = os.path.join(directory, "init")
    init_v = variables.Variable(2.0)
    init_ckpt = util.Checkpoint(v=init_v)
    self.evaluate(init_v.initializer)
    init_path = init_ckpt.save(init_prefix)

    # Create the checkpoint manager.
    ckpt_dir = os.path.join(directory, "ckpt")
    v = variables.Variable(1.0)
    checkpoint = util.Checkpoint(v=v)
    manager = checkpoint_management.CheckpointManager(
        checkpoint,
        ckpt_dir,
        max_to_keep=None,
        init_fn=lambda: checkpoint.restore(init_path).run_restore_ops())
    self.evaluate(v.initializer)

    # First call should call `init_fn`.
    self.assertIsNone(manager.restore_or_initialize())
    self.assertEqual(2.0, self.evaluate(v))

    # Save a checkpoint and second call should restore from the checkpoints.
    manager.save()
    self.assertIsNotNone(manager.restore_or_initialize())

  @test_util.run_in_graph_and_eager_modes
  def testCheckpointManagerFSpathDirectory(self):
    # [FUNCTION DOC]
    # Scenario: manager constructed with a path-like directory and custom name. Expects the save
    # path and the restore-or-initialize result to be the string path of the first numbered
    # checkpoint.
    directory = pathlib.Path(self.get_temp_dir())
    v = variables.Variable(0.0)
    checkpoint = util.Checkpoint(v=v)
    self.evaluate(v.initializer)
    manager = checkpoint_management.CheckpointManager(
        checkpoint, directory, max_to_keep=2, checkpoint_name="ckpt_name")
    save_path = manager.save()
    expected = str(directory / "ckpt_name-1")
    self.assertEqual(expected, save_path)

    restore_path = manager.restore_or_initialize()
    self.assertEqual(str(directory / "ckpt_name-1"), restore_path)

  @test_util.run_in_graph_and_eager_modes
  def testLatestCheckpointFSpathDirectory(self):
    # [FUNCTION DOC]
    # Scenario: after one manager save, the latest-checkpoint lookup given a path-like directory
    # returns the string path of the first numbered checkpoint.
    directory = pathlib.Path(self.get_temp_dir())
    checkpoint = util.Checkpoint()
    manager = checkpoint_management.CheckpointManager(
        checkpoint, directory, max_to_keep=2, checkpoint_name="ckpt_name")
    manager.save()

    cp_dir = checkpoint_management.latest_checkpoint(directory)
    self.assertEqual(str(directory / "ckpt_name-1"), cp_dir)

  @test_util.run_in_graph_and_eager_modes
  def testCheckpointInterval(self):
    # [FUNCTION DOC]
    # Scenario: manager with step counter and interval two. Saves at step 0; skipped at step 1;
    # saved at step 2; skipped when called again at the same step; skipped at step 3; and saved
    # regardless when the interval check is disabled.
    v = variables.Variable(1.0)
    step_counter = variables.Variable(0)
    self.evaluate([v.initializer, step_counter.initializer])
    checkpoint = util.Checkpoint(v=v)
    manager = checkpoint_management.CheckpointManager(
        checkpoint,
        self.get_temp_dir(),
        max_to_keep=None,
        step_counter=step_counter,
        checkpoint_interval=2)

    # step_counter: 0, save an initial checkpoint.
    path = manager.save(check_interval=True)
    self.assertTrue(checkpoint_management.checkpoint_exists(path))

    # step_counter: 1, no checkpoint saved.
    self.evaluate(step_counter.assign_add(1))
    path = manager.save(check_interval=True)
    self.assertIsNone(path)

    # step_counter: 2, checkpoint saved.
    self.evaluate(step_counter.assign_add(1))
    path = manager.save(check_interval=True)
    self.assertTrue(checkpoint_management.checkpoint_exists(path))

    # no checkpoint saved when calling `save` with the same step counter.
    path = manager.save(check_interval=True)
    self.assertIsNone(path)

    # step_counter: 3, no checkpoint saved.
    self.evaluate(step_counter.assign_add(1))
    path = manager.save(check_interval=True)
    self.assertIsNone(path)

    # Always save the checkpoint.
    path = manager.save(check_interval=False)
    self.assertTrue(checkpoint_management.checkpoint_exists(path))

  @test_util.run_in_graph_and_eager_modes
  def testCheckpointIntervalWithLastCheckpointStep(self):
    # [FUNCTION DOC]
    # Scenario: manager with interval two and an initial last checkpoint step of one, step counter
    # starting at one. Saves are skipped at steps one and two and happen at step three.
    v = variables.Variable(1.0)
    step_counter = variables.Variable(1)
    self.evaluate([v.initializer, step_counter.initializer])
    checkpoint = util.Checkpoint(v=v)
    manager = checkpoint_management.CheckpointManager(
        checkpoint,
        self.get_temp_dir(),
        max_to_keep=None,
        step_counter=step_counter,
        checkpoint_interval=2,
        last_checkpoint_step=1)

    # step_counter: 1, no checkpoint saved.
    path = manager.save(check_interval=True)
    self.assertIsNone(path)

    # step_counter: 2, no checkpoint saved since the interval starts at 1.
    self.evaluate(step_counter.assign_add(1))
    path = manager.save(check_interval=True)
    self.assertIsNone(path)

    # step_counter: 3, checkpoint saved.
    self.evaluate(step_counter.assign_add(1))
    path = manager.save(check_interval=True)
    self.assertTrue(checkpoint_management.checkpoint_exists(path))

  @test_util.run_in_graph_and_eager_modes
  def testCheckpointIntervalWithRestore(self):
    # [FUNCTION DOC]
    # Scenario: a checkpoint is saved directly, then a manager with an interval restores from the
    # directory. Since restore records the current step as the last checkpoint step, a save at the
    # same step returns nothing.
    directory = self.get_temp_dir()
    v = variables.Variable(1.0)
    step_counter = variables.Variable(0)
    self.evaluate([v.initializer, step_counter.initializer])

    # Prepare a checkpoint.
    checkpoint = util.Checkpoint(v=v)
    checkpoint.save(os.path.join(directory, "ckpt"))

    manager = checkpoint_management.CheckpointManager(
        checkpoint,
        directory,
        max_to_keep=None,
        step_counter=step_counter,
        checkpoint_interval=2)

    # Restore from the checkpoint.
    self.assertIsNotNone(manager.restore_or_initialize())

    # step_counter: 0, no checkpoint saved because it is restored from the
    # checkpoint with the same step.
    path = manager.save()
    self.assertIsNone(path)


if __name__ == "__main__":
  test.main()
