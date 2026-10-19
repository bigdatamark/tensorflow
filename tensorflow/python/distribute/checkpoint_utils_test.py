# [MODULE DOC]
# Tests for restoring variables from a checkpoint (the init-from-checkpoint utility) while a
# distribution strategy is active. Defines a small checkpoint-writing helper and one test class
# that runs graph-mode tests across default, one-device and mirrored strategies, both inside a
# replica context and in the cross-replica context. Not a helper module; it registers no state at
# import time and runs the test main when executed as a script.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for checkpoint_utils.init_from_checkpoint with Distribution Strategy.

These tests are located here instead of as part of
`python.training.CheckpointsTest` because they need access to distribution
strategies which are only present in contrib right now.
TODO(priyag): Move the tests to core `python.training.CheckpointsTest` when
distribution strategy moves out of contrib.
"""

import os

from absl.testing import parameterized

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import checkpoint_utils
from tensorflow.python.training import saver as saver_lib


def _create_checkpoints(sess, checkpoint_dir):
  # [FUNCTION DOC]
  # HELPER: writes the source checkpoint used by the init-from-checkpoint tests.
  # Takes a session and a directory path. Creates two variables (one of shape 1x10, one of shape
  # 10x10) in the current variable scope, initializes all global variables, reads their values,
  # and saves a checkpoint named "model" in the directory with global step zero.
  # Returns a pair with the two variables' values as arrays. Side effects: adds variables and a
  # saver to the current graph and writes checkpoint files to disk.
  checkpoint_prefix = os.path.join(checkpoint_dir, "model")
  checkpoint_state_name = "checkpoint"
  v1 = variable_scope.get_variable("var1", [1, 10])
  v2 = variable_scope.get_variable("var2", [10, 10])
  sess.run(variables.global_variables_initializer())
  v1_value, v2_value = sess.run([v1, v2])
  saver = saver_lib.Saver()
  saver.save(
      sess,
      checkpoint_prefix,
      global_step=0,
      latest_filename=checkpoint_state_name)
  return v1_value, v2_value


class CheckpointUtilsWithDistributionStrategyTest(
    test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case checking that initializing variables from a checkpoint works under
  # several distribution strategies, in and out of replica context. Uses a checkpoint written
  # into the test temp directory by a shared helper method.

  def _get_test_object(self):
    # [FUNCTION DOC]
    # HELPER: fixture creator for the tests in this class.
    # Writes a fresh checkpoint into the test temp directory using a cached session.
    # Returns the checkpoint directory and the saved values of the two variables.
    checkpoint_dir = self.get_temp_dir()
    with self.cached_session() as session:
      v1, v2 = _create_checkpoints(session, checkpoint_dir)
    return checkpoint_dir, v1, v2

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.default_strategy,
              strategy_combinations.one_device_strategy,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.mirrored_strategy_with_two_gpus,
              strategy_combinations
              .mirrored_strategy_with_two_gpus_no_merge_call,
          ],
          in_replica_mode=[True, False],
          mode=["graph"]))
  def testInitFromCheckpoint(self, distribution, in_replica_mode):
    # [FUNCTION DOC]
    # Graph-mode test, parameterized by strategy and by whether to run in replica context.
    # Saves a checkpoint, then under the strategy scope creates two new variables (the second is
    # read-synchronized with mean aggregation), maps the checkpoint variables to them by name,
    # initializes them and asserts that both hold the saved values. In replica mode the work is
    # run through call-for-each-replica, otherwise directly.
    checkpoint_dir, v1_value, v2_value = self._get_test_object()

    def init_and_verify(g):
      # [FUNCTION DOC]
      # HELPER: body run either per replica or directly. Takes the graph under test, creates the
      # new variables, requests initialization from the checkpoint by name mapping, runs the
      # global initializer in a session on that graph and asserts both values equal the saved
      # ones. Returns nothing.
      v1 = variable_scope.get_variable("new_var1", [1, 10])
      v2 = variable_scope.get_variable(
          "new_var2", [10, 10],
          synchronization=variable_scope.VariableSynchronization.ON_READ,
          aggregation=variable_scope.VariableAggregation.MEAN)
      checkpoint_utils.init_from_checkpoint(checkpoint_dir, {
          "var1": "new_var1",
          "var2": "new_var2"
      })
      with self.session(graph=g) as session:
        session.run(variables.global_variables_initializer())
        self.assertAllEqual(v1_value, self.evaluate(v1))
        self.assertAllEqual(v2_value, self.evaluate(v2))

    with ops.Graph().as_default() as g, distribution.scope():
      if in_replica_mode:
        distribution.extended.call_for_each_replica(init_and_verify, args=[g])
      else:
        init_and_verify(g)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.default_strategy,
              strategy_combinations.one_device_strategy,
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.mirrored_strategy_with_two_gpus,
              strategy_combinations
              .mirrored_strategy_with_two_gpus_no_merge_call,
          ],
          in_replica_mode=[True, False],
          mode=["graph"]))
  def testInitFromDifferentNameObject(self, distribution, in_replica_mode):
    # [FUNCTION DOC]
    # Graph-mode test, parameterized like the previous one. Checks that the checkpoint mapping
    # works when the target variable name is a string object freshly built (by concatenation) in
    # each replica, i.e. a different object with equal contents. Asserts the new variable gets
    # the saved value of the first checkpoint variable.
    checkpoint_dir, v1_value, _ = self._get_test_object()

    def init_and_verify(g):
      # [FUNCTION DOC]
      # HELPER: body run per replica or directly. Creates a variable, builds its name by joining
      # two strings, initializes it from the checkpoint using that name, runs the initializer in
      # a session on the given graph and asserts the value equals the saved one. Returns nothing.
      v1 = variable_scope.get_variable("new_var1", [1, 10])
      # Use string add to create new object in each replica
      prefix = "new_"
      suffix = "var1"
      new_var1 = prefix + suffix
      checkpoint_utils.init_from_checkpoint(checkpoint_dir, {
          "var1": new_var1,
      })
      with self.test_session(graph=g) as session:
        session.run(variables.global_variables_initializer())
        self.assertAllEqual(v1_value, self.evaluate(v1))

    with ops.Graph().as_default() as g, distribution.scope():
      if in_replica_mode:
        distribution.extended.call_for_each_replica(init_and_verify, [g])
      else:
        init_and_verify(g)


if __name__ == "__main__":
  test.main()
