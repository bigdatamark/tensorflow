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
# [MODULE DOC]
# Unit tests for the name-based checkpoint utilities: load_variable, list_variables,
# load_checkpoint, init_from_checkpoint (scope mapping, partitioned variables, root mapping,
# error cases, resource variables) and wait_for_new_checkpoint. Defines two module-level helper
# functions that write small checkpoints. Only tests; no import-time side effects.
"""Tests for checkpoints tools."""

import os
import pathlib
import time

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import saver as saver_lib


def _create_checkpoints(sess, checkpoint_dir):
  # [FUNCTION DOC]
  # HELPER: test fixture that writes a checkpoint for the checkpoint utility tests.
  # Creates four variables (var1 of shape 1x10, var2 10x10, var3 100x100, and var4 9x9 inside
  # a scope named useful_scope), initializes them in the given session, saves them with a
  # saver under a model prefix in the given directory (global step zero), and returns the four
  # values as arrays in that order. Side effects: adds variables to the default graph, writes
  # files.
  checkpoint_prefix = os.path.join(checkpoint_dir, "model")
  checkpoint_state_name = "checkpoint"
  v1 = variable_scope.get_variable("var1", [1, 10])
  v2 = variable_scope.get_variable("var2", [10, 10])
  v3 = variable_scope.get_variable("var3", [100, 100])
  with variable_scope.variable_scope("useful_scope"):
    v4 = variable_scope.get_variable("var4", [9, 9])
  sess.run(variables.global_variables_initializer())
  v1_value, v2_value, v3_value, v4_value = sess.run([v1, v2, v3, v4])
  saver = saver_lib.Saver()
  saver.save(
      sess,
      checkpoint_prefix,
      global_step=0,
      latest_filename=checkpoint_state_name)
  return v1_value, v2_value, v3_value, v4_value


def _create_partition_checkpoints(sess, checkpoint_dir):
  # [FUNCTION DOC]
  # HELPER: test fixture that writes a checkpoint holding one partitioned variable.
  # Creates variable var1 (shape 100x100, truncated-normal initializer, up to five partitions
  # along the first axis) in a scope named scope, initializes it, saves it with a saver under
  # a model prefix in the given directory, and returns the list of partition values.
  # Side effects: adds variables to the default graph, writes files.
  checkpoint_prefix = os.path.join(checkpoint_dir, "model")
  checkpoint_state_name = "checkpoint"
  with variable_scope.variable_scope("scope"):
    v1 = variable_scope.get_variable(
        name="var1",
        shape=[100, 100],
        initializer=init_ops.truncated_normal_initializer(0.5),
        partitioner=partitioned_variables.min_max_variable_partitioner(
            max_partitions=5, axis=0, min_slice_size=8 << 10))
  sess.run(variables.global_variables_initializer())
  v1_value = sess.run(v1._get_variable_list())
  saver = saver_lib.Saver()
  saver.save(
      sess,
      checkpoint_prefix,
      global_step=0,
      latest_filename=checkpoint_state_name)
  return v1_value


class CheckpointsTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the checkpoint reading and initialization utilities. Each test writes its
  # own checkpoint into the temporary directory with the module helpers and then loads,
  # lists, or initializes variables from it in fresh graphs.

  def testNoCheckpoints(self):
    # [FUNCTION DOC]
    # Scenario: loading a variable from a directory with no checkpoints. Expects a TensorFlow
    # op error to be raised.
    checkpoint_dir = self.get_temp_dir() + "/no_checkpoints"
    with self.assertRaises(errors_impl.OpError):
      self.assertAllEqual(
          checkpoint_utils.load_variable(checkpoint_dir, "var1"), [])

  def testNoTensor(self):
    # [FUNCTION DOC]
    # Scenario: a checkpoint exists but the requested name (var5) is not in it. Expects an op
    # error from load_variable.
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      _, _, _, _ = _create_checkpoints(session, checkpoint_dir)
    with self.assertRaises(errors_impl.OpError):
      self.assertAllEqual(
          checkpoint_utils.load_variable(checkpoint_dir, "var5"), [])

  def testGetTensor(self):
    # [FUNCTION DOC]
    # Scenario: writes a checkpoint, then loads each of the four variables by name (including
    # the scoped one). Expects each loaded value to equal the value that was saved.
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      v1, v2, v3, v4 = _create_checkpoints(session, checkpoint_dir)
    self.assertAllEqual(
        checkpoint_utils.load_variable(checkpoint_dir, "var1"), v1)
    self.assertAllEqual(
        checkpoint_utils.load_variable(checkpoint_dir, "var2"), v2)
    self.assertAllEqual(
        checkpoint_utils.load_variable(checkpoint_dir, "var3"), v3)
    self.assertAllEqual(
        checkpoint_utils.load_variable(checkpoint_dir, "useful_scope/var4"), v4)

  def testGetAllVariables(self):
    # [FUNCTION DOC]
    # Scenario: writes a checkpoint and lists its variables. Expects the sorted list of name
    # and shape pairs for the four saved variables.
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      _create_checkpoints(session, checkpoint_dir)
    self.assertEqual(
        checkpoint_utils.list_variables(checkpoint_dir),
        [("useful_scope/var4", [9, 9]), ("var1", [1, 10]), ("var2", [10, 10]),
         ("var3", [100, 100])])

  def testFSPath(self):
    # [FUNCTION DOC]
    # Scenario: the checkpoint directory is a path-like object. Checks that load_checkpoint,
    # load_variable and list_variables all accept it and return the saved value and the expected
    # sorted name and shape list.
    checkpoint_dir = pathlib.Path(self.get_temp_dir())
    with self.cached_session() as session:
      v1, v2, v3, v4 = _create_checkpoints(session, checkpoint_dir)  # pylint: disable=unused-variable

    reader = checkpoint_utils.load_checkpoint(checkpoint_dir)
    self.assertAllEqual(reader.get_tensor("var1"), v1)

    self.assertAllEqual(
        checkpoint_utils.load_variable(checkpoint_dir, "var1"), v1)

    self.assertEqual(
        checkpoint_utils.list_variables(checkpoint_dir),
        [("useful_scope/var4", [9, 9]), ("var1", [1, 10]), ("var2", [10, 10]),
         ("var3", [100, 100])])

  def testInitFromCheckpoint(self):
    # [FUNCTION DOC]
    # Scenario: in a new graph, creates differently named variables and initializes them from
    # the checkpoint using a dictionary map (name to name and scope to scope) and a list of
    # pairs (including mapping one checkpoint tensor to two variables). After running the
    # initializers expects all values to match the saved ones and the graph definition to stay
    # small, showing tensors are not embedded as constants.
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      v1, v2, v3, v4 = _create_checkpoints(session, checkpoint_dir)

    # New graph and session.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as session:
        with variable_scope.variable_scope("some_scope"):
          my1 = variable_scope.get_variable("my1", [1, 10])
          with variable_scope.variable_scope("some_other_scope"):
            my2 = variable_scope.get_variable("my2", [10, 10])
            with variable_scope.variable_scope("other_useful_scope"):
              my4 = variable_scope.get_variable("var4", [9, 9])
        my3 = variable_scope.get_variable("my3", [100, 100])
        my3b = variable_scope.get_variable("my3b", [100, 100])

        checkpoint_utils.init_from_checkpoint(checkpoint_dir, {
            "var1": "some_scope/my1",
            "useful_scope/": "some_scope/some_other_scope/other_useful_scope/",
        })
        checkpoint_utils.init_from_checkpoint(checkpoint_dir, [
            ("var2", "some_scope/some_other_scope/my2"),
            ("var3", my3),
            ("var3", my3b),
        ])

        session.run(variables.global_variables_initializer())
        self.assertAllEqual(my1.eval(session), v1)
        self.assertAllEqual(my2.eval(session), v2)
        self.assertAllEqual(my3.eval(session), v3)
        self.assertAllEqual(my3b.eval(session), v3)
        self.assertAllEqual(my4.eval(session), v4)

        # Check that tensors are not explicitly in the graph.
        self.assertLess(len(str(session.graph.as_graph_def())), 32000)

  def testInitialValueComesFromCheckpoint(self):
    # [FUNCTION DOC]
    # Scenario: a variable with a zeros initializer has its initialized-value read taken before
    # and after init_from_checkpoint. Expects the earlier read to give zeros before running
    # initializers and, after the global initializer runs, all reads to give the checkpoint
    # value; also confirms the zeros comparison fails an approximate-equality assertion.
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      v1, _, _, _ = _create_checkpoints(session, checkpoint_dir)

    # New graph and session.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as session:
        with variable_scope.variable_scope(
            "some_scope", initializer=init_ops.zeros_initializer()):
          my1 = variable_scope.get_variable("my1", [1, 10])

        before = my1.initialized_value()

        checkpoint_utils.init_from_checkpoint(checkpoint_dir, {"var1": my1})

        after = my1.initialized_value()

        self.assertAllEqual(session.run(before), [[0.0] * 10])
        self.assertAllEqual(session.run(after), v1)

        session.run(variables.global_variables_initializer())

        self.assertAllEqual(session.run(my1), v1)
        self.assertAllEqual(session.run(my1.initialized_value()), v1)
        self.assertAllClose(session.run(before), v1)
        self.assertAllClose(session.run(after), v1)
        with self.assertRaises(AssertionError):
          self.assertAllClose(v1, [[0.0] * 10])

  def testInitWithScopeDoesNotCaptureSuffixes(self):
    # [FUNCTION DOC]
    # Scenario: a scope mapping for useful_scope must not also capture variables in a scope
    # named useful_scope_1. Expects the first variable to load the checkpoint value and the
    # second to keep its own initial value.
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      _, _, _, v4 = _create_checkpoints(session, checkpoint_dir)

    with ops.Graph().as_default() as g:
      with variable_scope.variable_scope("useful_scope"):
        my4 = variable_scope.get_variable("var4", [9, 9])
      with variable_scope.variable_scope("useful_scope_1"):
        my5_init = [[1.0, 2.0], [3.0, 4.0]]
        my5 = variable_scope.get_variable("var5", initializer=my5_init)

      checkpoint_utils.init_from_checkpoint(checkpoint_dir,
                                            {"useful_scope/": "useful_scope/"})
      with self.session(graph=g) as session:
        session.run(variables.global_variables_initializer())
        self.assertAllEqual(my4.eval(session), v4)
        self.assertAllEqual(my5.eval(session), my5_init)

  def testRestoreRunsOnSameDevice(self):
    # [FUNCTION DOC]
    # Scenario: a variable is created under a parameter-server device scope and then
    # initialized from the checkpoint by scope mapping. The test only expects the call to
    # complete without error (device placement of the restore must be compatible).
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      _create_checkpoints(session, checkpoint_dir)

    with ops.Graph().as_default():
      with ops.device("/job:ps"):
        with variable_scope.variable_scope("useful_scope"):
          variable_scope.get_variable("var4", [9, 9])

      checkpoint_utils.init_from_checkpoint(checkpoint_dir,
                                            {"useful_scope/": "useful_scope/"})

  def testInitFromRootCheckpoint(self):
    # [FUNCTION DOC]
    # Scenario: maps the checkpoint root to a scope named some_scope, whose variables share the
    # checkpoint names. Expects all four variables to take the saved values.
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      v1, v2, v3, v4 = _create_checkpoints(session, checkpoint_dir)

    # New graph and session.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as session:
        with variable_scope.variable_scope("some_scope"):
          my1 = variable_scope.get_variable("var1", [1, 10])
          my2 = variable_scope.get_variable("var2", [10, 10])
          my3 = variable_scope.get_variable("var3", [100, 100])
          with variable_scope.variable_scope("useful_scope"):
            my4 = variable_scope.get_variable("var4", [9, 9])

        checkpoint_utils.init_from_checkpoint(checkpoint_dir,
                                              {"/": "some_scope/",})

        session.run(variables.global_variables_initializer())
        self.assertAllEqual(my1.eval(session), v1)
        self.assertAllEqual(my2.eval(session), v2)
        self.assertAllEqual(my3.eval(session), v3)
        self.assertAllEqual(my4.eval(session), v4)

  def testInitToRootCheckpoint(self):
    # [FUNCTION DOC]
    # Scenario: maps the checkpoint root to the graph root with variables at top level and one
    # inside a scope. Expects all four variables to take the saved values.
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      v1, v2, v3, v4 = _create_checkpoints(session, checkpoint_dir)

    # New graph and session.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as session:
        my1 = variable_scope.get_variable("var1", [1, 10])
        my2 = variable_scope.get_variable("var2", [10, 10])
        my3 = variable_scope.get_variable("var3", [100, 100])
        with variable_scope.variable_scope("useful_scope"):
          my4 = variable_scope.get_variable("var4", [9, 9])

        checkpoint_utils.init_from_checkpoint(checkpoint_dir,
                                              {"/": "/",})

        session.run(variables.global_variables_initializer())
        self.assertAllEqual(my1.eval(session), v1)
        self.assertAllEqual(my2.eval(session), v2)
        self.assertAllEqual(my3.eval(session), v3)
        self.assertAllEqual(my4.eval(session), v4)

  def testInitFromPartitionVar(self):
    # [FUNCTION DOC]
    # Scenario: a checkpoint contains one partitioned variable. In a first graph, one variable
    # with the same partitioning is initialized by name and another with a different number of
    # partitions by scope; expects the values to match, the partition counts to differ, and the
    # concatenated values to be equal. In a second graph, a partition list is mapped directly
    # and expected to equal the saved partitions.
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      v1 = _create_partition_checkpoints(session, checkpoint_dir)

    # New graph and session.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as session:
        with variable_scope.variable_scope("some_scope"):
          my1 = variable_scope.get_variable(
              name="my1",
              shape=[100, 100],
              initializer=init_ops.zeros_initializer(),
              partitioner=partitioned_variables.min_max_variable_partitioner(
                  max_partitions=5, axis=0, min_slice_size=8 << 10))
          my1_var_list = my1._get_variable_list()
        # Create another variable with different partitions than the variable in
        # the checkpoint.
        with variable_scope.variable_scope("some_other_scope"):
          my2 = variable_scope.get_variable(
              name="var1",
              shape=[100, 100],
              initializer=init_ops.zeros_initializer(),
              partitioner=partitioned_variables.min_max_variable_partitioner(
                  max_partitions=5, axis=0, min_slice_size=16 << 10))
          my2_var_list = my2._get_variable_list()

        checkpoint_utils.init_from_checkpoint(checkpoint_dir, {
            "scope/var1": "some_scope/my1",
            "scope/": "some_other_scope/"})

        session.run(variables.global_variables_initializer())
        my1_values = session.run(my1_var_list)
        self.assertAllEqual(my1_values, v1)
        my2_values = session.run(my2_var_list)
        # Verify we created different number of partitions.
        self.assertNotEqual(len(my2_values), len(v1))
        # Verify the values were correctly initialized inspite of different
        # partitions.
        full_my2_values = np.concatenate(my2_values, axis=0)
        full_v1_values = np.concatenate(v1, axis=0)
        self.assertAllEqual(full_my2_values, full_v1_values)

    # New graph and session.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as session:
        with variable_scope.variable_scope("some_scope"):
          my1 = variable_scope.get_variable(
              name="my1",
              shape=[100, 100],
              initializer=init_ops.truncated_normal_initializer(0.5),
              partitioner=partitioned_variables.min_max_variable_partitioner(
                  max_partitions=5, axis=0, min_slice_size=8 << 10))
          my1_var_list = my1._get_variable_list()

        checkpoint_utils.init_from_checkpoint(checkpoint_dir,
                                              {"scope/var1": my1_var_list,})

        session.run(variables.global_variables_initializer())
        my1_values = session.run(my1_var_list)
        self.assertAllEqual(my1_values, v1)

  def testInitFromCheckpointMissing(self):
    # [FUNCTION DOC]
    # Scenario: error handling of init_from_checkpoint. Expects an op error for a missing
    # directory, and value errors for: a tensor absent from the checkpoint, a graph variable that
    # does not exist, a shape mismatch, a scope with no matching checkpoint tensors, and a
    # mapping from a non-scope name to a scope name.
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      _, _, _, _ = _create_checkpoints(session, checkpoint_dir)

    # New graph and session.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as session:
        with variable_scope.variable_scope("some_scope"):
          _ = variable_scope.get_variable("my1", [10, 10])
          _ = variable_scope.get_variable(
              "my2", [1, 10],
              dtype=dtypes.int64,
              initializer=init_ops.zeros_initializer())

        # No directory.
        with self.assertRaises(errors_impl.OpError):
          checkpoint_utils.init_from_checkpoint("no_dir",
                                                {"var1": "some_scope/my1"})

        # No variable in checkpoint.
        with self.assertRaises(ValueError):
          checkpoint_utils.init_from_checkpoint(checkpoint_dir,
                                                {"no_var": "some_scope/my1"})

        # No variable in the graph.
        with self.assertRaises(ValueError):
          checkpoint_utils.init_from_checkpoint(checkpoint_dir,
                                                {"var3": "some_scope/no_var"})

        # Shape mismatch.
        with self.assertRaises(ValueError):
          checkpoint_utils.init_from_checkpoint(checkpoint_dir,
                                                {"var1": "some_scope/my1"})

        # Variable 'my1' and 'my2' are missing in given checkpoint scope.
        with self.assertRaises(ValueError):
          checkpoint_utils.init_from_checkpoint(
              checkpoint_dir, {"useful_scope/": "some_scope/"})

        # Mapping is not to scope name.
        with self.assertRaises(ValueError):
          checkpoint_utils.init_from_checkpoint(checkpoint_dir,
                                                {"useful_scope": "some_scope/"})

  def testNoAdditionalReadOpsForResourceVariables(self):
    # [FUNCTION DOC]
    # Scenario: initializes a resource variable from the checkpoint inside a name scope.
    # Expects the value to match after running initializers, and that the only operations
    # created under that scope besides the checkpoint initializer ones are variable assignment
    # and identity ops (no extra read ops).
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      v1, _, _, _ = _create_checkpoints(session, checkpoint_dir)

    # New graph and session.
    with ops.Graph().as_default() as g:
      with self.session(graph=g) as session:
        my1 = resource_variable_ops.ResourceVariable([[0.0] * 10], name="my1")

        with ops.name_scope("init_from_checkpoint"):
          checkpoint_utils.init_from_checkpoint(checkpoint_dir, {"var1": my1})

        # Basic sanity checks:
        session.run(variables.global_variables_initializer())
        self.assertAllEqual(session.run(my1), v1)

    ops_in_init_from_checkpoint_scope = [
        op for op in g.get_operations()
        if (op.name.startswith("init_from_checkpoint/") and
            not op.name.startswith("init_from_checkpoint/checkpoint_initializer"
                                  ) and
            op.type != "AssignVariableOp" and
            op.type != "Identity")
    ]
    self.assertEqual(ops_in_init_from_checkpoint_scope, [])


@test_util.run_all_in_graph_and_eager_modes
class WaitForNewCheckpointTest(test.TestCase):
  # [CLASS DOC]
  # Tests for wait_for_new_checkpoint, run in both graph and eager modes via the class
  # decorator. No shared fixtures.

  def testReturnsNoneAfterTimeout(self):
    # [FUNCTION DOC]
    # Scenario: waits on a nonexistent directory with a one second timeout and half second
    # sleep. Expects a null result, with elapsed wall time greater than half a second and less
    # than about 1.1 seconds.
    start = time.time()
    ret = checkpoint_utils.wait_for_new_checkpoint(
        "/non-existent-dir", "foo", timeout=1.0, seconds_to_sleep=0.5)
    end = time.time()
    self.assertIsNone(ret)

    # We've waited one second.
    self.assertGreater(end, start + 0.5)

    # The timeout kicked in.
    self.assertLess(end, start + 1.1)


if __name__ == "__main__":
  test.main()
