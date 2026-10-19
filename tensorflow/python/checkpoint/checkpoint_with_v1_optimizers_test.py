# [MODULE DOC]
# Tests for object-based checkpointing that use v1 (tf.train) optimizers and variable-scope
# templates. Covers slot restoration, graph stability and multi-graph use. Runs via test main.
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
"""Tests for object-based saving which use tf.train.* optimizers."""
import os

from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.checkpoint import checkpoint_options as options
from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import template
from tensorflow.python.ops import variable_scope
from tensorflow.python.trackable import autotrackable
from tensorflow.python.training import adam
from tensorflow.python.training import checkpoint_utils


class CheckpointingTests(test.TestCase):
  # [CLASS DOC]
  # Tests object-based checkpoint saving and restoring with v1 (tf.train) optimizers: deferred
  # slot restoration, skipping slot variables, graph stability across repeated saves/restores,
  # and multiple graphs sharing one optimizer.

  @test_util.run_in_graph_and_eager_modes
  def testDeferredSlotRestoration(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Saves a checkpoint without the optimizer (variable 12) and
    # one with it (variable 13, slot m 14). Restores both into a new, empty root before any
    # variables exist (deferred restoration). Asserts consumption failures until the variable is
    # created, that the variable ends at 12, that the slot is restored to 14 only in eager mode
    # (graph mode needs a minimize call and restore ops), a complaint about beta1 power until the
    # optimizer is fully built, and finally full consumption.
    checkpoint_directory = self.get_temp_dir()

    root = trackable_utils.Checkpoint()
    root.var = trackable_utils.add_variable(
        root, name="var", initializer=0.)
    optimizer = adam.AdamOptimizer(0.1)
    if context.executing_eagerly():
      optimizer.minimize(root.var.read_value)
    else:
      train_op = optimizer.minimize(root.var)
      # Note that `optimizer` has not been added as a dependency of
      # `root`. Create a one-off grouping so that slot variables for `root.var`
      # get initialized too.
      self.evaluate(trackable_utils.gather_initializers(
          trackable_utils.Checkpoint(root=root, optimizer=optimizer)))
      self.evaluate(train_op)
    self.evaluate(state_ops.assign(root.var, 12.))
    no_slots_path = root.save(os.path.join(checkpoint_directory, "no_slots"))
    root.optimizer = optimizer
    self.evaluate(state_ops.assign(root.var, 13.))
    self.evaluate(state_ops.assign(optimizer.get_slot(name="m", var=root.var),
                                   14.))
    slots_path = root.save(os.path.join(checkpoint_directory, "with_slots"))
    new_root = trackable_utils.Checkpoint()
    # Load the slot-containing checkpoint (deferred), then immediately overwrite
    # the non-slot variable (also deferred).
    slot_status = new_root.restore(slots_path)
    no_slot_status = new_root.restore(no_slots_path)
    with self.assertRaises(AssertionError):
      no_slot_status.assert_consumed()
    new_root.var = trackable_utils.add_variable(
        new_root, name="var", shape=[])
    no_slot_status.assert_consumed()
    no_slot_status.run_restore_ops()
    self.assertEqual(12., self.evaluate(new_root.var))
    new_root.optimizer = adam.AdamOptimizer(0.1)
    slot_status.assert_existing_objects_matched()
    with self.assertRaisesRegex(AssertionError, "beta1_power"):
      slot_status.assert_consumed()
    self.assertEqual(12., self.evaluate(new_root.var))
    if context.executing_eagerly():
      # Slot variables are only created with restoring initializers when
      # executing eagerly.
      self.assertEqual(14., self.evaluate(
          new_root.optimizer.get_slot(name="m", var=new_root.var)))
    else:
      self.assertIs(new_root.optimizer.get_slot(name="m", var=new_root.var),
                    None)
    if context.executing_eagerly():
      new_root.optimizer.minimize(new_root.var.read_value)
    else:
      train_op = new_root.optimizer.minimize(new_root.var)
      # The slot variable now exists; restore() didn't create it, but we should
      # now have a restore op for it.
      slot_status.run_restore_ops()
      self.assertEqual(14., self.evaluate(
          new_root.optimizer.get_slot(name="m", var=new_root.var)))
      self.evaluate(train_op)
    slot_status.assert_consumed()

  def testSkipSlotVarRestore(self):
    # [FUNCTION DOC]
    # Saves a checkpoint of a variable (set to 12) and its Adam slot m (set to 14), asserts the
    # exact list of checkpoint keys, then restores into a fresh object with the skip-slot-variables
    # option on: the variable becomes 12 while the slot stays 0. Restoring again with the option
    # off restores the slot to 14.
    def create_trackable():
      # [FUNCTION DOC]
      # HELPER: builds the object under test for testSkipSlotVarRestore.
      # Returns a checkpoint object holding a variable "var" (initial 0) and a v1 Adam optimizer
      # whose slot variables for that variable have been created.
      root = trackable_utils.Checkpoint()
      root.var = trackable_utils.add_variable(root, name="var", initializer=0.0)
      root.optimizer = adam.AdamOptimizer(0.1)
      root.optimizer._create_slots([root.var])
      return root
    root = create_trackable()
    checkpoint_directory = self.get_temp_dir()
    self.evaluate(state_ops.assign(root.var, 12.0))
    self.evaluate(
        state_ops.assign(root.optimizer.get_slot(name="m", var=root.var), 14.0)
    )
    chkpt_path = root.save(os.path.join(checkpoint_directory, "chkpt"))
    new_root = create_trackable()
    chkpt_keys = [x[0] for x in checkpoint_utils.list_variables(chkpt_path)]
    self.assertEqual(
        chkpt_keys,
        [
            "_CHECKPOINTABLE_OBJECT_GRAPH",
            "optimizer/beta1_power/.ATTRIBUTES/VARIABLE_VALUE",
            "optimizer/beta2_power/.ATTRIBUTES/VARIABLE_VALUE",
            "save_counter/.ATTRIBUTES/VARIABLE_VALUE",
            "var/.ATTRIBUTES/VARIABLE_VALUE",
            "var/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE",
            "var/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE",
        ],
    )
    # Restore without slot variables
    slot_restore = new_root.restore(
        chkpt_path,
        options.CheckpointOptions(experimental_skip_slot_variables=True),
    )
    slot_restore.assert_consumed().run_restore_ops()
    # regular variable is restored
    self.assertEqual(self.evaluate(new_root.var), 12.0)
    # Slot variable is not restored
    self.assertEqual(
        self.evaluate(new_root.optimizer.get_slot(name="m", var=new_root.var)),
        0.0,
    )
    # Restore everything
    slot_restore = new_root.restore(
        chkpt_path,
        options.CheckpointOptions(experimental_skip_slot_variables=False),
    )
    slot_restore.assert_consumed().run_restore_ops()
    self.assertEqual(self.evaluate(new_root.var), 12.0)
    # Slot variable also restored
    self.assertEqual(
        self.evaluate(new_root.optimizer.get_slot(name="m", var=new_root.var)),
        14.0,
    )

  def testManySavesGraph(self):
    # [FUNCTION DOC]
    # Graph mode. Builds a variable and v1 Adam optimizer, saves once, records the graph
    # operations, saves again, and asserts the operation list is unchanged (no new ops on later
    # saves).
    """Saves after the first should not modify the graph."""
    with context.graph_mode():
      graph = ops.Graph()
      with graph.as_default(), self.session(graph):
        checkpoint_directory = self.get_temp_dir()
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        obj = trackable_utils.Checkpoint()
        obj.var = variable_scope.get_variable(name="v", initializer=0.)
        obj.opt = adam.AdamOptimizer(0.1)
        obj.opt.minimize(obj.var.read_value())
        self.evaluate(trackable_utils.gather_initializers(obj))
        obj.save(checkpoint_prefix)
        before_ops = graph.get_operations()
        obj.save(checkpoint_prefix)
        self.assertEqual(before_ops, graph.get_operations())

  def testManyRestoresGraph(self):
    # [FUNCTION DOC]
    # Graph mode. Builds a variable and v1 Adam optimizer, saves, restores once, records the graph
    # operations, restores again, and asserts the operation list is unchanged (restore ops are
    # cached after the first restore).
    """Restores after the first should not modify the graph."""
    with context.graph_mode():
      graph = ops.Graph()
      with graph.as_default(), self.session(graph):
        checkpoint_directory = self.get_temp_dir()
        checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
        obj = trackable_utils.Checkpoint()
        obj.var = variable_scope.get_variable(name="v", initializer=0.)
        obj.opt = adam.AdamOptimizer(0.1)
        obj.opt.minimize(obj.var.read_value())
        self.evaluate(trackable_utils.gather_initializers(obj))
        save_path = obj.save(checkpoint_prefix)
        obj.restore(save_path)
        before_ops = graph.get_operations()
        obj.restore(save_path)
        self.assertEqual(before_ops, graph.get_operations())

  def testMultipleGraphsNonSlotVariables(self):
    # [FUNCTION DOC]
    # Graph mode. Builds a model with a shared v1 Adam optimizer in a first graph and sets values
    # (variable 1, slot 2, beta1 power 3). In a second graph it builds the same model, restores
    # nothing, trains, sets values 4/5/6, saves, overwrites, restores and asserts the saved values
    # (4, 5, 6) return. Finally asserts the first graph's values (1, 2, 3) are untouched.
    with context.graph_mode():
      checkpoint_directory = self.get_temp_dir()
      checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
      optimizer = adam.AdamOptimizer(0.001)
      # Construct a model in one graph
      first_graph = ops.Graph()
      first_session = session_lib.Session(graph=first_graph)
      with first_graph.as_default(), first_session.as_default():
        first_variable = resource_variable_ops.ResourceVariable([1.])
        first_root_trackable = trackable_utils.Checkpoint(
            optimizer=optimizer, variable=first_variable)
        train_op = optimizer.minimize(first_variable.read_value)
        self.evaluate(trackable_utils.gather_initializers(
            first_root_trackable))
        self.evaluate(train_op)
        self.evaluate(first_variable.assign([1.]))
        self.evaluate(optimizer.get_slot(
            var=first_variable, name="m").assign([2.]))
        beta1_power, _ = optimizer._get_beta_accumulators()
        self.evaluate(beta1_power.assign(3.))

      # Save and load in a second graph
      second_graph = ops.Graph()
      with second_graph.as_default(), session_lib.Session(graph=second_graph):
        second_variable = resource_variable_ops.ResourceVariable([1.])
        second_root_trackable = trackable_utils.Checkpoint(
            optimizer=optimizer, variable=second_variable)
        train_op = optimizer.minimize(second_variable.read_value)
        second_root_trackable.restore(None).initialize_or_restore()
        self.evaluate(train_op)
        self.evaluate(second_variable.assign([4.]))
        self.evaluate(optimizer.get_slot(
            var=second_variable, name="m").assign([5.]))
        beta1_power, _ = optimizer._get_beta_accumulators()
        self.evaluate(beta1_power.assign(6.))
        save_path = second_root_trackable.save(checkpoint_prefix)
        self.evaluate(second_variable.assign([7.]))
        self.evaluate(optimizer.get_slot(
            var=second_variable, name="m").assign([8.]))
        beta1_power, _ = optimizer._get_beta_accumulators()
        self.assertAllEqual(6., self.evaluate(beta1_power))
        status = second_root_trackable.restore(save_path)
        status.assert_consumed().run_restore_ops()
        self.assertAllEqual([4.], self.evaluate(second_variable))
        self.assertAllEqual([5.], self.evaluate(optimizer.get_slot(
            var=second_variable, name="m")))
        beta1_power, _ = optimizer._get_beta_accumulators()
        self.assertAllEqual(6., self.evaluate(beta1_power))

      # Check that the first graph is unmolested
      with first_graph.as_default(), first_session.as_default():
        self.assertAllEqual([1.], self.evaluate(first_variable))
        self.assertAllEqual([2.], self.evaluate(optimizer.get_slot(
            var=first_variable, name="m")))
        beta1_power, _ = optimizer._get_beta_accumulators()
        self.assertAllEqual(3., self.evaluate(beta1_power))


class _ManualScope(autotrackable.AutoTrackable):
  # [CLASS DOC]
  # HELPER: test fixture for TemplateTests. Auto-trackable object that, when called, creates a
  # variable inside a named variable scope and registers it as its own tracked dependency,
  # emulating a manually managed template. Holds the variable scope after the first call.

  def __call__(self):
    # [FUNCTION DOC]
    # HELPER: test fixture behavior.
    # Opens the variable scope named ManualScope, stores the scope on the instance, then builds
    # the variable while capturing dependencies into this object as a template, so the variable
    # becomes its tracked child. Returns the created variable.
    with variable_scope.variable_scope("ManualScope") as vs:
      self.variable_scope = vs
      with trackable_utils.capture_dependencies(template=self):
        return self._build()

  def _build(self):
    # [FUNCTION DOC]
    # HELPER: creates the variable for _ManualScope.
    # Gets (or creates) a scalar variable named in_manual_scope in the current variable scope and
    # returns it.
    return variable_scope.get_variable(name="in_manual_scope", shape=[])


class TemplateTests(test.TestCase):
  # [CLASS DOC]
  # Tests that variable-scope templates (with v1 optimizers) save and restore through
  # object-based checkpoints.

  @test_util.run_in_graph_and_eager_modes
  def test_trackable_save_restore(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Builds a template ("s1") with two variables and a manual
    # scope, checks list_objects and trackable children, adds it with a v1 Adam optimizer to a
    # checkpoint, sets values 12 and 14, and saves. Then builds a second template ("s2") with a new
    # optimizer, restores, and creates variables. Asserts the template has three trackable children
    # (v, v2, ManualScope), restore is consumed, and values restored are 12, 13 (v plus one) and 14.

    def _templated():
      # [FUNCTION DOC]
      # HELPER: template body for test_trackable_save_restore.
      # Creates two zero-initialized resource variables (v, v2) with variable-scope getters, a
      # manual-scope trackable, and returns a tuple: v, v plus one, v2, the manual scope object
      # and the variable returned by calling it.
      v = variable_scope.get_variable(
          "v", shape=[1], initializer=init_ops.zeros_initializer(),
          use_resource=True)
      v2 = variable_scope.get_variable(
          "v2", shape=[1], initializer=init_ops.zeros_initializer(),
          use_resource=True)
      manual = _ManualScope()
      return v, v + 1., v2, manual, manual()

    save_template = template.make_template("s1", _templated)
    v1_save, _, v2_save, manual_scope, manual_scope_v = save_template()
    self.assertCountEqual([
        id(obj) for obj in
        [v1_save, v2_save, manual_scope, manual_scope_v, save_template]
    ], [id(obj) for obj in trackable_utils.list_objects(save_template)])
    self.assertDictEqual({"in_manual_scope": manual_scope_v},
                         manual_scope._trackable_children())
    optimizer = adam.AdamOptimizer(0.0)
    save_root = trackable_utils.Checkpoint(
        my_template=save_template, optimizer=optimizer)
    optimizer.minimize(v1_save.read_value)
    self.evaluate([v.initializer for v in save_template.variables])
    self.evaluate([v.initializer for v in optimizer.variables()])
    self.evaluate(v1_save.assign([12.]))
    self.evaluate(v2_save.assign([14.]))
    checkpoint_directory = self.get_temp_dir()
    checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")
    save_path = save_root.save(checkpoint_prefix)

    load_template = template.make_template("s2", _templated)
    load_optimizer = adam.AdamOptimizer(0.0)
    load_root = trackable_utils.Checkpoint(
        my_template=load_template, optimizer=load_optimizer)
    status = load_root.restore(save_path)
    var, var_plus_one, var2, _, _ = load_template()
    load_optimizer.minimize(var.read_value)
    self.assertEqual(3, len(load_template._trackable_children()))
    self.assertEqual(set(["v", "v2", "ManualScope"]),
                     load_template._trackable_children().keys())
    status.assert_consumed().run_restore_ops()
    self.assertAllEqual([12.], self.evaluate(var))
    self.assertAllEqual([13.], self.evaluate(var_plus_one))
    self.assertAllEqual([14.], self.evaluate(var2))


if __name__ == "__main__":
  test.main()
