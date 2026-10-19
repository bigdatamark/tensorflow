# [MODULE DOC]
# Unit tests for the mirrored value types in the distribute values library: mirrored variables
# (a variable replicated across devices) and plain mirrored values (constants replicated across
# devices). Covers variable property forwarding, saving/restoring checkpoints between mirrored and
# ordinary variables, and arithmetic on mirrored values. Contains small module-level helper
# functions that build mirrored objects, plus the test classes. Running the file as a script
# starts the distribute test runner. No import-time side effects beyond imports.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for the mirrored values library."""

import os

from absl.testing import parameterized

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.distribute import test_util as ds_test_util
from tensorflow.python.distribute import tpu_values
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import saver as saver_lib


def _make_mirrored(distribution=None):
  # [FUNCTION DOC]
  # HELPER: builds the mirrored variable fixture used by the tests in this file.
  # Creates two resource variables named "v" (initial value 1) and "v/replica" (initial value 2),
  # one per device. Devices come from the optional distribution's worker devices; when the
  # distribution is missing (null), a GPU and a CPU device are used. The variables are wrapped in a
  # mirrored variable with sum aggregation: the TPU flavor when the distribution is a TPU strategy,
  # otherwise the regular one. Returns the mirrored variable. Side effect: creates variables in the
  # current graph/variable scope.
  v = []
  if distribution:
    devices = distribution.extended.worker_devices
  else:
    devices = ["/device:GPU:0", "/device:CPU:0"]
  for d, n, init in zip(devices, ["v", "v/replica"], [1., 2.]):
    with ops.device(d):
      v.append(
          variable_scope.get_variable(
              name=n, initializer=init, use_resource=True))

  if (distribution
      is not None) and strategy_test_lib.is_tpu_strategy(distribution):
    var_cls = tpu_values.TPUMirroredVariable
  else:
    var_cls = values_lib.MirroredVariable
  mirrored = var_cls(distribution, v, variable_scope.VariableAggregation.SUM)
  return mirrored


def _make_mirrored_val(init_val=5.0):
  # [FUNCTION DOC]
  # HELPER: builds the plain mirrored value fixture used by the mirrored value tests.
  # Takes an optional float init_val (default 5.0), creates one constant with that value on a GPU
  # device and one on a CPU device, and returns them wrapped in a mirrored value object.
  v = []
  devices = ["/device:GPU:0", "/device:CPU:0"]
  for d, _ in zip(devices, ["v", "v/replica"]):
    with ops.device(d):
      v.append(constant_op.constant(init_val))
  return values_lib.Mirrored(v)


def mirrored_and_tpu_strategy_combinations():
  # [FUNCTION DOC]
  # HELPER: supplies the parameter combinations for the save/restore tests.
  # Takes no arguments. Returns the cross product of four distribution strategies (mirrored with
  # GPU and CPU, mirrored with two GPUs without merge call, TPU, and TPU with packed variables)
  # and the two modes graph and eager.
  return combinations.combine(
      distribution=[
          strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          strategy_combinations.mirrored_strategy_with_two_gpus_no_merge_call,
          strategy_combinations.tpu_strategy,
          strategy_combinations.tpu_strategy_packed_var,
      ],
      mode=["graph", "eager"])


class MirroredVariableTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests basic behavior of a mirrored variable. The class-level session config allows soft device
  # placement so that GPU-requested variables can fall back to the CPU. The context is reset after
  # each test.

  config = config_pb2.ConfigProto()
  config.allow_soft_placement = True

  def tearDown(self):
    # [FUNCTION DOC]
    # Runs the parent teardown, then resets the global eager context so that device and
    # configuration state does not leak between tests. Takes no arguments, returns nothing.
    super().tearDown()
    context._reset_context()

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testProperties(self):
    # [FUNCTION DOC]
    # Scenario: build the two-device mirrored variable fixture (skipped in eager mode when no GPU
    # is present). Asserts that the mirrored variable reports the same name, data type and shape as
    # its first component variable. Runs in both graph and eager modes.
    if context.num_gpus() < 1 and context.executing_eagerly():
      self.skipTest("A GPU is not available for this test in eager mode.")

    mirrored = _make_mirrored()
    v = mirrored.values[0]
    self.assertEqual(v.name, mirrored.name)
    self.assertEqual(v.dtype, mirrored.dtype)
    self.assertEqual(v.shape, mirrored.shape)

  @test_util.run_in_graph_and_eager_modes(config=config)
  def testVariableOnAnotherDevice(self):
    # [FUNCTION DOC]
    # Scenario: wrap a single ordinary resource variable in a mirrored variable with mean
    # aggregation and no distribution strategy. Asserts that the mirrored variable exposes the same
    # name, data type and shape as the wrapped variable. Runs in both graph and eager modes.
    v = variable_scope.get_variable(
        name="v", initializer=[1.], use_resource=True)
    mirrored = values_lib.MirroredVariable(
        None, (v,), variable_scope.VariableAggregation.MEAN)

    self.assertEqual(v.name, mirrored.name)
    self.assertEqual(v.dtype, mirrored.dtype)
    self.assertEqual(v.shape, mirrored.shape)


class MirroredVariableSaveRestoreTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests checkpoint saving and restoring of mirrored variables, both mirrored-to-mirrored and
  # crossing between mirrored and ordinary variables. Holds private helper methods for assigning,
  # saving and restoring; the public tests run for each strategy/mode combination.

  def _assign_mirrored(self, v, new):
    # [FUNCTION DOC]
    # HELPER: sets the component variables of a mirrored variable for the save/restore tests.
    # Takes the mirrored variable v and a list of new values, one per component; assigns each value
    # to the matching component and evaluates the assignment. Returns nothing.
    for var, n in zip(v.values, new):
      self.evaluate(var.assign(n))

  def _save_return_saver(self, sess, var):
    # [FUNCTION DOC]
    # HELPER: saves one variable to a checkpoint and also hands back the saver.
    # Takes a session and a variable (mirrored or ordinary). Builds a saver for that variable and
    # writes a checkpoint with prefix "ckpt" inside the test temp directory. Returns a pair of the
    # saved checkpoint path and the saver. Side effect: writes checkpoint files.
    saver = saver_lib.Saver(var_list=[var])
    test_dir = self.get_temp_dir()
    prefix = os.path.join(test_dir, "ckpt")
    return saver.save(sess, prefix), saver

  def _save(self, sess, var):
    # [FUNCTION DOC]
    # HELPER: saves a variable to a checkpoint and returns only the checkpoint path (discarding the
    # saver). Takes a session and a variable; see the save-and-return-saver helper for details.
    save_path, _ = self._save_return_saver(sess, var)
    return save_path

  def _save_mirrored(self, distribution):
    # [FUNCTION DOC]
    # HELPER: produces a checkpoint from a mirrored variable.
    # In a fresh graph and session, builds a mirrored variable for the given distribution, assigns
    # the component values 3 and 4, saves a checkpoint (which records the first component, 3), then
    # overwrites the components with 5 and 6 so a later restore is observable. Returns the
    # checkpoint path.
    """Save variables with mirroring, returns save_path."""
    with self.session(graph=ops.Graph()) as sess:
      mirrored = _make_mirrored(distribution)

      # Overwrite the initial values.
      self._assign_mirrored(mirrored, [3., 4.])

      # Saves the current value of v[0], 3.
      save_path = self._save(sess, mirrored)

      # Change the values between save and restore.
      self._assign_mirrored(mirrored, [5., 6.])
    return save_path

  def _save_normal(self):
    # [FUNCTION DOC]
    # HELPER: produces a checkpoint from an ordinary (non-mirrored) variable.
    # In a fresh graph and session, creates a resource variable named "v", assigns 3, saves a
    # checkpoint, then changes the variable to 5. Takes no arguments; returns the checkpoint path.
    """Save variables without mirroring, returns save_path."""
    with self.session(graph=ops.Graph()) as sess:
      var = variable_scope.get_variable(
          name="v", initializer=1., use_resource=True)

      # Overwrite the initial value.
      self.evaluate(var.assign(3.))

      # Saves the current value of var, 3.
      save_path = self._save(sess, var)

      # Change the values between save and restore.
      self.evaluate(var.assign(5.))
    return save_path

  def _restore_normal(self, save_path):
    # [FUNCTION DOC]
    # HELPER: restores a checkpoint into an ordinary variable and checks the result.
    # Takes the checkpoint path. In a fresh graph, creates a variable named "v", sets it to 8,
    # restores the checkpoint, and asserts the value is 3. Returns nothing.
    """Restore to variables without mirroring in a fresh graph."""
    with self.session(graph=ops.Graph()) as sess:
      var = variable_scope.get_variable(
          name="v", initializer=7., use_resource=True)

      # Overwrite the initial value.
      self.evaluate(var.assign(8.))

      # Restores the saved value of 3. to `var`.
      saver = saver_lib.Saver(var_list=[var])
      saver.restore(sess, save_path)
      self.assertEqual(3., self.evaluate(var))

  def _restore_mirrored(self, save_path, distribution):
    # [FUNCTION DOC]
    # HELPER: restores a checkpoint into a mirrored variable and checks the result.
    # Takes the checkpoint path and the distribution. In a fresh graph, builds a mirrored variable,
    # sets its components to 7 and 8, restores the checkpoint, and asserts both components now hold
    # 3. Returns nothing.
    """Restore to variables with mirroring in a fresh graph."""
    with self.session(graph=ops.Graph()) as sess:
      mirrored = _make_mirrored(distribution)
      v = mirrored.values

      # Overwrite the initial values.
      self._assign_mirrored(mirrored, [7., 8.])

      # Restores the saved value of 3. to both variables.
      saver = saver_lib.Saver(var_list=[mirrored])
      saver.restore(sess, save_path)
      self.assertEqual([3., 3.], self.evaluate([v[0], v[1]]))

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveAndRestoreMirroredOneGraph(self, distribution):
    # [FUNCTION DOC]
    # Scenario: within one cached session, build a mirrored variable, set components to 3 and 4,
    # save, change them to 5 and 6, then restore with the same saver. Asserts both components equal
    # 3 afterwards. Runs for each strategy/mode combination.
    with self.cached_session() as sess:
      mirrored = _make_mirrored(distribution)
      v = mirrored  .values

      # Overwrite the initial values.
      self._assign_mirrored(mirrored, [3., 4.])

      # Saves the current value of v[0], 3.
      save_path, saver = self._save_return_saver(sess, mirrored)

      # Change the values between save and restore.
      self._assign_mirrored(mirrored, [5., 6.])

      # Restores the saved value of 3. to both variables.
      saver.restore(sess, save_path)
      self.assertEqual([3., 3.], self.evaluate([v[0], v[1]]))

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveMirroredRestoreMirrored(self, distribution):
    # [FUNCTION DOC]
    # Scenario: save a checkpoint from a mirrored variable, then restore it into a freshly built
    # mirrored variable in a new graph; the restore helper asserts both components become 3.
    # Skipped in eager mode when no GPU exists.
    if context.num_gpus() < 1 and context.executing_eagerly():
      # Graph mode can work without GPU because the Placer "moves" the
      # variable to a CPU. In other words, if there is no GPU available, but
      # user requested to create a variable on GPU, Placer will ignore the
      # user request and assign the VarHandleOp to CPU. This requires
      # soft_placement, which is on by default.
      self.skipTest("A GPU is not available for this test in eager mode.")

    save_path = self._save_mirrored(distribution)
    self._restore_mirrored(save_path, distribution)

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveMirroredRestoreNormal(self, distribution):
    # [FUNCTION DOC]
    # Scenario: save a checkpoint from a mirrored variable, then restore it into an ordinary
    # variable in a new graph; the restore helper asserts the value is 3. Skipped in eager mode
    # when no GPU exists.
    if context.num_gpus() < 1 and context.executing_eagerly():
      # Graph mode can work without GPU because the Placer "moves" the
      # variable to a CPU. In other words, if there is no GPU available, but
      # user requested to create a variable on GPU, Placer will ignore the
      # user request and assign the VarHandleOp to CPU. This requires
      # soft_placement, which is on by default.
      self.skipTest("A GPU is not available for this test in eager mode.")

    save_path = self._save_mirrored(distribution)
    self._restore_normal(save_path)

  @combinations.generate(mirrored_and_tpu_strategy_combinations())
  def testSaveNormalRestoreMirrored(self, distribution):
    # [FUNCTION DOC]
    # Scenario: save a checkpoint from an ordinary variable, then restore it into a mirrored
    # variable in a new graph; the restore helper asserts both components become 3. Skipped in
    # eager mode when no GPU exists.
    if context.num_gpus() < 1 and context.executing_eagerly():
      # Graph mode can work without GPU because the Placer "moves" the
      # variable to a CPU. In other words, if there is no GPU available, but
      # user requested to create a variable on GPU, Placer will ignore the
      # user request and assign the VarHandleOp to CPU. This requires
      # soft_placement, which is on by default.
      self.skipTest("A GPU is not available for this test in eager mode.")

    save_path = self._save_normal()
    self._restore_mirrored(save_path, distribution)


class MirroredTest(test.TestCase):
  # [CLASS DOC]
  # Tests the plain mirrored value type (an immutable value replicated across devices), currently
  # its arithmetic behavior.

  def testAddOp(self):
    # [FUNCTION DOC]
    # Scenario: build a mirrored value holding 3 (skipped when no GPU is available). Asserts that
    # adding it to itself gives 6, adding the number 1 gives 4, that the operator form and the
    # explicit add function agree, and that both results have the same type.
    if context.num_gpus() < 1:
      self.skipTest("A GPU is not available for this test.")
    mirrored_val = _make_mirrored_val(init_val=3.)

    self.assertEqual(self.evaluate(constant_op.constant(6.)),
                     self.evaluate(mirrored_val + mirrored_val))
    self.assertEqual(self.evaluate(constant_op.constant(4.)),
                     self.evaluate(mirrored_val + 1))
    self.assertEqual(self.evaluate(mirrored_val + 1),
                     self.evaluate(math_ops.add(mirrored_val, 1)))
    self.assertEqual(type(mirrored_val + 1),
                     type(math_ops.add(mirrored_val, 1)))


if __name__ == "__main__":
  ds_test_util.main()
