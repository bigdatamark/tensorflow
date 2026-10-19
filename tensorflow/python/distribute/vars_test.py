# [MODULE DOC]
# Tests for variable behavior under distribution strategies (mirrored, multi-worker mirrored and
# TPU, with and without the variable-policy option). Covers "on write" synchronized variables
# (assign, assign add/sub, reads, checkpoint save/restore, scatter ops) and "on read" synchronized
# variables (assignments, aggregation-specific reads, all-reduce, scatter ops that must be
# rejected, checkpoint save/restore).
# Module-level helpers build the parameter combinations used by the test decorators. Running the
# file as a script starts the distributed test runner. Importing it has no side effects beyond
# building the combination lists.
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
"""Tests for the distributed values library."""

import itertools

import uuid
from absl.testing import parameterized
from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.checkpoint import checkpoint_management as ckpt_manager
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.util import variable_utils


def strategy_and_run_tf_function_combinations():
  # [FUNCTION DOC]
  # HELPER: builds the strategy/mode combinations consumed by the variable tests in this file.
  # KNOWN ISSUE: the leading comment lines read like a stray docstring (with a dangling closing
  # quote), and a large part of the combination (the mirrored strategy and multi-worker variants)
  # is commented out, so only the two TPU strategies are actually produced.
  # Takes no arguments. Returns a combination of the TPU strategy and packed-variable TPU strategy
  # with graph and eager modes, the option to run through a compiled function fixed to enabled,
  # and the variable-policy flag set both on and off.
  # Test the combination of different strategies and whether a tf.function
  # is passed into strategy.run."""
  # TODO(b/197981388): re-enable MWMS test
  # return combinations.combine(
  #     distribution=[
  #         strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
  #     ],
  #     mode=["graph", "eager"],
  #     experimental_run_tf_function=[True, False],
  #     use_var_policy=[True, False]) +
  return combinations.combine(
      distribution=[
          strategy_combinations.tpu_strategy,
          strategy_combinations.tpu_strategy_packed_var,
      ],
      mode=["graph", "eager"],
      experimental_run_tf_function=[True],
      use_var_policy=[True, False])


def strategy_with_var_policy():
  # [FUNCTION DOC]
  # HELPER: builds the strategy/mode combinations consumed by the variable tests in this file.
  # Takes no arguments. Returns a combination of the mirrored (GPU and CPU) strategy, the TPU
  # strategy and the packed-variable TPU strategy (multi-worker variants are disabled by a TODO),
  # each in graph and eager modes, with the variable-policy flag both on and off.
  return combinations.combine(
      distribution=[
          strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          # TODO(b/197981388): re-enable MWMS test
          # strategy_combinations.multi_worker_mirrored_2x1_cpu,
          # strategy_combinations.multi_worker_mirrored_2x1_gpu,
          strategy_combinations.tpu_strategy,
          strategy_combinations.tpu_strategy_packed_var,
      ],
      mode=["graph", "eager"],
      use_var_policy=[True, False])


class OnWriteVariableSync(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering variables created under a distribution strategy with the default "on write"
  # synchronization: assignment in replica and cross-replica contexts, reads, initialization,
  # operator overloading and checkpoint save/restore. Uses the parameterized combination decorator
  # to run each test over strategies and modes.

  @combinations.generate(strategy_and_run_tf_function_combinations())
  def testAssign(self, distribution, experimental_run_tf_function):
    # [FUNCTION DOC]
    # Test: for every update kind (assign, assign add, assign sub), aggregation setting and
    # context (cross-replica or replica), creates a variable under the strategy and applies the
    # update with a scalar value. Replica-context SUM is skipped. Expects every component of the
    # variable to end up equal to one, as the variable started at zero.

    def assign(fn, v, update_value, cross_replica):
      # [FUNCTION DOC]
      # HELPER: applies the named update method of a variable with a value, either directly
      # (cross-replica) or through the strategy run (optionally wrapped in a compiled function).
      # Inputs: method name, variable, update value, and whether to use cross-replica context.
      # Returns the update result, gathered across replicas in the replica case.
      update_fn = lambda: getattr(v, fn)(update_value)
      if cross_replica:
        return update_fn()
      else:
        if experimental_run_tf_function:
          update_fn = def_function.function(update_fn)
        return test_util.gather(distribution, distribution.run(update_fn))

    updates = [("assign", 1.), ("assign_add", 1.), ("assign_sub", -1.)]
    aggregations = [
        variables_lib.VariableAggregation.NONE,
        variables_lib.VariableAggregation.SUM,
        variables_lib.VariableAggregation.MEAN,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
    ]
    options = list(
        x for x in itertools.product(updates, aggregations, [True, False]))
    for update, aggregation, cross_replica in options:
      # assign in replica context with SUM does not make sense cause you can
      # just do value * num replicas error is 1. is not a distributed value and
      # is unsupported for aggregation SUM
      if (not cross_replica and aggregation ==
          variables_lib.VariableAggregation.SUM):
        continue
      with distribution.scope():
        v = variable_v1.VariableV1(
            0.,
            aggregation=aggregation)
      self.evaluate(variables_lib.global_variables_initializer())
      fn, update_value = update
      self.evaluate(assign(fn, v, update_value, cross_replica))
      for component in v._values:
        self.assertAllEqual(self.evaluate(component.read_value()),
                            self.evaluate(array_ops.ones_like(component)))

  @combinations.generate(strategy_and_run_tf_function_combinations())
  def testAssignOnWriteVar(self, distribution, experimental_run_tf_function):
    # [FUNCTION DOC]
    # Test: same as the basic assign test but the update values are themselves variables
    # (two, and minus two, with mean aggregation). SUM aggregation is skipped. Expects every
    # component of the target variable to equal two after the update.

    with distribution.scope():
      v_to_assign = variable_v1.VariableV1(
          2., aggregation=variables_lib.VariableAggregation.MEAN)
      v_to_assign_sub = variable_v1.VariableV1(
          -2., aggregation=variables_lib.VariableAggregation.MEAN)

    def assign(fn, v, update_value, cross_replica):
      # [FUNCTION DOC]
      # HELPER: applies the named update method with a value in cross-replica or replica context.
      # Inputs: method name, variable, update value, cross-replica flag. Returns the update result
      # (gathered across replicas in the replica case).
      update_fn = lambda: getattr(v, fn)(update_value)
      if cross_replica:
        return update_fn()
      else:
        if experimental_run_tf_function:
          update_fn = def_function.function(update_fn)
        return test_util.gather(distribution, distribution.run(update_fn))

    updates = [("assign", v_to_assign), ("assign_add", v_to_assign),
               ("assign_sub", v_to_assign_sub)]
    aggregations = [
        variables_lib.VariableAggregation.NONE,
        variables_lib.VariableAggregation.SUM,
        variables_lib.VariableAggregation.MEAN,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
    ]
    options = list(
        x for x in itertools.product(updates, aggregations, [True, False]))
    for update, aggregation, cross_replica in options:
      # assign in replica context with SUM does not make sense cause you can
      # just do value * num replicas error is 1. is not a distributed value and
      # is unsupported for aggregation SUM
      if aggregation == variables_lib.VariableAggregation.SUM:
        continue
      with distribution.scope():
        v = variable_v1.VariableV1(
            0.,
            aggregation=aggregation)
      self.evaluate(variables_lib.global_variables_initializer())
      fn, update_value = update
      self.evaluate(assign(fn, v, update_value, cross_replica))
      for component in v._values:
        self.assertAllEqual(2.0, self.evaluate(component.read_value()))

  @combinations.generate(strategy_and_run_tf_function_combinations())
  def testAssignPerReplicaVal(self, distribution, experimental_run_tf_function):
    # [FUNCTION DOC]
    # Test: assigns values that differ per replica (a per-replica container of twos or minus twos)
    # to variables, only in replica context and not with aggregation NONE. Skipped on TPU
    # strategies. Expects every component to equal four for SUM aggregation and two otherwise.

    if strategy_test_lib.is_tpu_strategy(distribution):
      self.skipTest("Assigning PerReplica values is not supported. See"
                    " sponge/80ba41f8-4220-4516-98ce-bbad48f9f11a.")

    with distribution.scope():
      per_replica_value = values.PerReplica(
          [constant_op.constant(2.0),
           constant_op.constant(2.0)])
      per_replica_sub_value = values.PerReplica(
          [constant_op.constant(-2.0),
           constant_op.constant(-2.0)])

    def assign(fn, v, update_value, cross_replica):
      # [FUNCTION DOC]
      # HELPER: applies the named update method with a value in cross-replica or replica context.
      # Inputs: method name, variable, update value, cross-replica flag. Returns the update result
      # (gathered across replicas in the replica case).
      update_fn = lambda: getattr(v, fn)(update_value)
      if cross_replica:
        return update_fn()
      else:
        if experimental_run_tf_function:
          update_fn = def_function.function(update_fn)
        return test_util.gather(distribution, distribution.run(update_fn))

    updates = [("assign", per_replica_value), ("assign_add", per_replica_value),
               ("assign_sub", per_replica_sub_value)]
    # We don't support assigning PerReplica valus to vars in replica context
    # with aggregation=NONE.
    aggregations = [
        variables_lib.VariableAggregation.SUM,
        variables_lib.VariableAggregation.MEAN,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
    ]
    options = list(
        x for x in itertools.product(updates, aggregations, [True, False]))
    for update, aggregation, cross_replica in options:
      # assign in replica context with SUM does not make sense cause you can
      # just do value * num replicas error is 1. is not a distributed value and
      # is unsupported for aggregation SUM
      if cross_replica:
        # We don't support assigning PerReplica values to MirroredVariables in
        # cross replica context
        continue
      with distribution.scope():
        v = variable_v1.VariableV1(
            0.,
            aggregation=aggregation)
      self.evaluate(variables_lib.global_variables_initializer())
      fn, update_value = update
      self.evaluate(assign(fn, v, update_value, cross_replica))
      if aggregation == variables_lib.VariableAggregation.SUM:
        expected = 4.0
      else:
        expected = 2.0
      for component in v._values:
        self.assertAllEqual(expected, self.evaluate(component.read_value()))

  @combinations.generate(strategy_with_var_policy())
  def testValueInReplicaContext(self, distribution):
    # [FUNCTION DOC]
    # Test: under the strategy, creates a mean-aggregated variable of value one and runs a compiled
    # function in each replica that adds one and then reads the variable value. Expects every
    # replica to see two.
    with distribution.scope():
      v = variables_lib.Variable(
          1., aggregation=variables_lib.VariableAggregation.MEAN)
      self.evaluate(variables_lib.global_variables_initializer())

      @def_function.function
      def f():
        # [FUNCTION DOC]
        # HELPER: per-replica step for the test; adds one to the variable and returns its value
        # after the add (ordered by a control dependency).
        with ops.control_dependencies([v.assign_add(1.)]):
          return v.value()

      results = self.evaluate(
          test_util.gather(distribution, distribution.run(f)))
      for value in results:
        self.assertEqual(2., value)

  @combinations.generate(strategy_with_var_policy())
  def testValueInReplicaContextAssignDirectValue(self, distribution,
                                                 use_var_policy):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite the name, nothing is assigned a direct value, and the variable-policy
    # argument is unused; the body is identical to the previous replica-context value test.
    # Test: runs a compiled function per replica that adds one to a mean-aggregated variable and
    # reads it back. Expects every replica to see two.
    with distribution.scope():
      v = variables_lib.Variable(
          1., aggregation=variables_lib.VariableAggregation.MEAN)
      self.evaluate(variables_lib.global_variables_initializer())

      @def_function.function
      def f():
        # [FUNCTION DOC]
        # HELPER: per-replica step for the test; adds one to the variable and returns its value
        # after the add (ordered by a control dependency).
        with ops.control_dependencies([v.assign_add(1.)]):
          return v.value()

      results = self.evaluate(
          test_util.gather(distribution, distribution.run(f)))
      for value in results:
        self.assertEqual(2., value)

  @combinations.generate(strategy_and_run_tf_function_combinations())
  def testReadValueInReplicaContext(self, distribution,
                                    experimental_run_tf_function):
    # [FUNCTION DOC]
    # Test: for each aggregation setting, creates a zero variable and reads its value in replica
    # context (optionally via a compiled function). Expects each gathered per-replica result to
    # equal the value of the matching component.
    aggregations = [
        variables_lib.VariableAggregation.NONE,
        variables_lib.VariableAggregation.SUM,
        variables_lib.VariableAggregation.MEAN,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
    ]
    for aggregation in aggregations:
      with distribution.scope():
        v = variable_v1.VariableV1(
            0.,
            aggregation=aggregation)
      self.evaluate(variables_lib.global_variables_initializer())
      if experimental_run_tf_function:
        read_var_fn = def_function.function(v.read_value)
      else:
        read_var_fn = v.read_value
      results = self.evaluate(
          test_util.gather(distribution, distribution.run(read_var_fn)))
      for component, value in zip(v._values, results):
        self.assertAllEqual(self.evaluate(component.read_value()), value)

  @combinations.generate(strategy_and_run_tf_function_combinations())
  def testReadValueInCrossReplicaContext(self, distribution,
                                         experimental_run_tf_function):
    # [FUNCTION DOC]
    # Test: for each aggregation setting, creates a variable of value two and reads it in
    # cross-replica context (optionally via a compiled function). Expects every component to
    # equal the single value returned.
    aggregations = [
        variables_lib.VariableAggregation.NONE,
        variables_lib.VariableAggregation.SUM,
        variables_lib.VariableAggregation.MEAN,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
    ]
    for aggregation in aggregations:
      with distribution.scope():
        v = variable_v1.VariableV1(
            2.,
            aggregation=aggregation)
      self.evaluate(variables_lib.global_variables_initializer())

      if experimental_run_tf_function:
        read_var_fn = def_function.function(v.read_value)
      else:
        read_var_fn = v.read_value

      results = read_var_fn()
      for component in v._values:
        self.assertEqual(self.evaluate(component.read_value()),
                         self.evaluate(results))

  @combinations.generate(strategy_with_var_policy())
  def testAssignOutOfScope(self, distribution):
    # [FUNCTION DOC]
    # Test: creates a variable under the strategy, then assigns three outside the strategy scope.
    # Expects the variable and every one of its component variables to read three.
    with distribution.scope():
      mirrored = variables_lib.Variable(1.)
    self.evaluate(mirrored.assign(3.))
    self.assertEqual(self.evaluate(mirrored.read_value()), 3.)
    for component in mirrored.values:
      self.assertEqual(self.evaluate(component.read_value()), 3.)

  @combinations.generate(strategy_with_var_policy())
  def testInitializedToSameValueInsideEagerRun(self, distribution):
    # [FUNCTION DOC]
    # Test (eager only; skipped for multi-worker collective strategy): creates a variable lazily
    # inside a per-replica function inside a compiled step with a random initial value and the
    # global seed cleared. Expects the first two replicas' component values to be equal.
    if not context.executing_eagerly(): self.skipTest("eager only test")
    if isinstance(distribution.extended,
                  collective_all_reduce_strategy.CollectiveAllReduceExtended):
      self.skipTest("Test for more than 1 device per worker only.")
    v = [None]

    @def_function.function
    def step():
      # [FUNCTION DOC]
      # HELPER: compiled step that runs the inner function on every replica.
      # Takes no arguments, returns nothing.

      def f():
        # [FUNCTION DOC]
        # HELPER: per-replica function that creates the variable with a random-normal initial
        # value the first time it is called, storing it in the enclosing list. Returns nothing.
        if v[0] is None:
          v[0] = variables_lib.Variable(random_ops.random_normal([]))

      distribution.run(f)

    context.set_global_seed(None)
    step()
    vals = self.evaluate(v[0].values)
    self.assertAllEqual(vals[0], vals[1])

  @combinations.generate(strategy_with_var_policy())
  def testAggregationOnlyFirstReplica(self, distribution):
    # [FUNCTION DOC]
    # Test (skipped for multi-worker collective strategy): variable with ON_WRITE synchronization
    # and ONLY_FIRST_REPLICA aggregation; each replica assigns its replica index in a compiled
    # function. Expects all per-replica results to equal the first replica's value (zero).
    if isinstance(distribution.extended,
                  collective_all_reduce_strategy.CollectiveAllReduceExtended):
      self.skipTest("b/212945803")
    with distribution.scope():
      v = variable_v1.VariableV1(
          15.,
          synchronization=variables_lib.VariableSynchronization.ON_WRITE,
          aggregation=variables_lib.VariableAggregation.ONLY_FIRST_REPLICA)
    self.evaluate(variables_lib.global_variables_initializer())

    @def_function.function
    def assign():
      # [FUNCTION DOC]
      # HELPER: compiled function assigning the current replica index (as float) to the variable
      # and returning the result.
      ctx = distribute_lib.get_replica_context()
      replica_id = ctx.replica_id_in_sync_group
      return v.assign(math_ops.cast(replica_id, dtypes.float32))

    per_replica_results = self.evaluate(
        test_util.gather(distribution, distribution.run(assign)))
    # The per-replica values should always match the first replicas value.
    self.assertAllEqual(
        array_ops.zeros(distribution.num_replicas_in_sync, dtypes.float32),
        per_replica_results)

  @combinations.generate(strategy_with_var_policy())
  def testInitScope(self, distribution):
    # [FUNCTION DOC]
    # Test (eager only): creates variables inside an initialization scope within a compiled
    # function, the second initialized from the first's value, then adds two to the second in each
    # replica. Expects both replicas to return two.
    if not context.executing_eagerly(): self.skipTest("eager only")

    class C(object):
      # [CLASS DOC]
      # HELPER: trivial plain object used as an attribute holder (variables are stored on it).
      pass

    obj = C()
    obj.w = None
    obj.v = None

    @def_function.function
    def assign():
      # [FUNCTION DOC]
      # HELPER: compiled function that lazily creates two mean-aggregated variables in an
      # initialization scope (running the initializer) and returns the second after adding two.
      with ops.init_scope():
        if obj.w is None:
          obj.w = variables_lib.Variable(
              0., aggregation=variables_lib.VariableAggregation.MEAN)
          obj.v = variables_lib.Variable(
              obj.w.read_value(),
              aggregation=variables_lib.VariableAggregation.MEAN)
          self.evaluate(variables_lib.global_variables_initializer())

      return obj.v.assign_add(2.)

    per_replica_results = self.evaluate(
        test_util.gather(distribution, distribution.run(assign)))
    self.assertAllEqual([2., 2.], per_replica_results)

  @combinations.generate(strategy_with_var_policy())
  def testOperatorOverride(self, distribution):
    # [FUNCTION DOC]
    # Test (skipped in graph mode for multi-worker collective strategy): variable with SUM
    # aggregation and value one. Expects the plain addition of one to evaluate to two, and a
    # compiled per-replica addition to give two in every replica.

    if not context.executing_eagerly() and isinstance(
        distribution.extended,
        collective_all_reduce_strategy.CollectiveAllReduceExtended):
      self.skipTest("b/212954197")

    with distribution.scope():
      v = variable_v1.VariableV1(
          1, aggregation=variables_lib.VariableAggregation.SUM)
      self.evaluate(variables_lib.global_variables_initializer())

    self.assertEqual(2, self.evaluate(v + 1))

    @def_function.function
    def add():
      # [FUNCTION DOC]
      # HELPER: compiled function returning the variable plus one.
      return v + 1

    per_replica_results = self.evaluate(
        test_util.gather(distribution, distribution.run(add)))
    self.assertAllEqual([2, 2], per_replica_results)

  @combinations.generate(
      combinations.combine(
          strategy=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.tpu_strategy_packed_var,
              strategy_combinations.multi_worker_mirrored_2x1_cpu,
              strategy_combinations.multi_worker_mirrored_2x1_gpu,
          ],
          mode=["eager"],
          use_var_policy=[True, False]))
  def testSaveAndRestoreOnWrite(self, strategy):
    # [FUNCTION DOC]
    # Test (eager only): for each aggregation, checks checkpoint round trips of an ON_WRITE
    # variable: save and restore it (expects two); restore that checkpoint into a plain variable
    # (expects two); save a plain variable (three) and restore into the ON_WRITE variable (expects
    # three). Checkpoints are written to uniquely named directories under the temporary folder.
    aggregation = [
        variable_scope.VariableAggregation.NONE,
        variable_scope.VariableAggregation.ONLY_FIRST_REPLICA,
        variable_scope.VariableAggregation.SUM,
        variable_scope.VariableAggregation.MEAN
    ]
    for agg in aggregation:
      v_normal_restore = variables_lib.Variable(1.0)
      v_normal_save = variables_lib.Variable(3.0)
      with strategy.scope():
        v_on_write = variables_lib.Variable(2.0, aggregation=agg)

        # Save ONWRITE Restore ONWRITE
        # Save
        ckpt = trackable_utils.Checkpoint(var=v_on_write)
        manager = ckpt_manager.CheckpointManager(
            ckpt, "/tmp/ckpt_" + str(uuid.uuid4()), max_to_keep=None)
        manager.save()
        # Restore
        ckpt.restore(manager.latest_checkpoint)
        self.assertEqual(2.0, self.evaluate(v_on_write._values[0]))
        self.assertEqual(2.0, self.evaluate(v_on_write.read_value()))

        # Save Mirrored Restore Normal
        # We've already saved Mirrored, so we only need to restore normal
        ckpt_normal = trackable_utils.Checkpoint(var=v_normal_restore)
        ckpt_normal.restore(manager.latest_checkpoint)
        self.assertEqual(2.0, self.evaluate(v_on_write._values[0]))
        self.assertEqual(2.0, self.evaluate(v_normal_restore.read_value()))

        # Save Normal Restore Mirrored
        # Save
        ckpt = trackable_utils.Checkpoint(var=v_normal_save)
        manager_2 = ckpt_manager.CheckpointManager(
            ckpt, "/tmp/ckptckpt_" + str(uuid.uuid4()), max_to_keep=None)
        manager_2.save()
        # Restore
        ckpt_on_write = trackable_utils.Checkpoint(var=v_on_write)
        ckpt_on_write.restore(manager_2.latest_checkpoint)
        self.assertEqual(3.0, self.evaluate(v_on_write._values[0]))
        self.assertEqual(3.0, self.evaluate(v_on_write.read_value()))


ms_combination = combinations.combine(
    distribution=[strategy_combinations.mirrored_strategy_with_gpu_and_cpu],
    mode=["graph", "eager"])
tpu_combination = combinations.combine(
    distribution=[strategy_combinations.tpu_strategy_packed_var],
    mode=["graph", "eager"])


class OnWriteVariableSyncScatterTests(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering scatter update operations (sub, add, div, mul, min, max, update) on ON_WRITE
  # variables under the mirrored and TPU strategies, including unsupported cases that must raise.

  @combinations.generate(ms_combination)
  def testScatterSub(self, distribution):
    # [FUNCTION DOC]
    # Test: mean-aggregated variable of zeros; each replica subtracts a sparse update built from
    # its replica index. Expects both replicas' results to be zero, minus one, minus one.
    with distribution.scope():
      v = variables_lib.Variable(
          [0., 0., 0.], aggregation=variables_lib.VariableAggregation.MEAN)
    self.evaluate(v.initializer)

    @def_function.function
    def scatter_sub():
      # [FUNCTION DOC]
      # HELPER: compiled per-replica function building a sparse update from the replica index and
      # applying scatter subtract to the variable; returns the result.
      ctx = distribute_lib.get_replica_context()
      replica_id = ctx.replica_id_in_sync_group
      value = indexed_slices.IndexedSlices(
          values=array_ops_stack.stack([
              math_ops.cast(replica_id, dtypes.float32),
              math_ops.cast(replica_id + 1, dtypes.float32)
          ]),
          indices=array_ops_stack.stack([replica_id, replica_id + 1]),
          dense_shape=(3,))
      return v.scatter_sub(value)

    per_replica_results = self.evaluate(
        distribution.experimental_local_results(
            distribution.run(scatter_sub)))
    self.assertAllEqual([[0., -1., -1.], [0., -1., -1.]], per_replica_results)

  @combinations.generate(ms_combination)
  def testScatterAdd(self, distribution):
    # [FUNCTION DOC]
    # Test: SUM-aggregated integer variable of zeros; each replica scatter-adds a sparse update
    # derived from its replica index. Expects both replicas to read zero, two, two.
    with distribution.scope():
      v = variables_lib.Variable(
          [0, 0, 0], aggregation=variables_lib.VariableAggregation.SUM)
    self.evaluate(v.initializer)

    @def_function.function
    def scatter_add():
      # [FUNCTION DOC]
      # HELPER: compiled per-replica function applying scatter add of a replica-index-based sparse
      # update to the variable; returns the result.
      ctx = distribute_lib.get_replica_context()
      replica_id = ctx.replica_id_in_sync_group
      value = indexed_slices.IndexedSlices(
          values=array_ops_stack.stack([replica_id, replica_id + 1]),
          indices=array_ops_stack.stack([replica_id, replica_id + 1]),
          dense_shape=(3,))
      return v.scatter_add(value)

    per_replica_results = self.evaluate(
        test_util.gather(distribution, distribution.run(scatter_add)))
    self.assertAllEqual([[0, 2, 2], [0, 2, 2]], per_replica_results)

  @combinations.generate(ms_combination)
  def testScatterDiv(self, distribution):
    # [FUNCTION DOC]
    # Test: SUM-aggregated integer variable of one, six, one; each replica divides selected
    # entries by its replica index plus two. Expects both replicas to read zero, two, one.
    with distribution.scope():
      v = variables_lib.Variable(
          [1, 6, 1], aggregation=variables_lib.VariableAggregation.SUM)
    self.evaluate(v.initializer)

    @def_function.function
    def scatter_div():
      # [FUNCTION DOC]
      # HELPER: compiled per-replica function applying scatter divide with a replica-index-based
      # sparse update; returns the result.
      ctx = distribute_lib.get_replica_context()
      replica_id = ctx.replica_id_in_sync_group
      value = indexed_slices.IndexedSlices(
          values=array_ops.reshape(replica_id + 2, [1]),
          indices=array_ops.reshape(replica_id, [1]),
          dense_shape=(3,))
      return v.scatter_div(value)

    per_replica_results = self.evaluate(
        test_util.gather(distribution, distribution.run(scatter_div)))
    self.assertAllEqual([[0, 2, 1], [0, 2, 1]], per_replica_results)

  @combinations.generate(ms_combination)
  def testScatterMul(self, distribution):
    # [FUNCTION DOC]
    # Test: mean-aggregated float variable of two, one, one; each replica multiplies selected
    # entries by its replica index plus two. Expects both replicas to read two, one and a half, one.
    with distribution.scope():
      v = variables_lib.Variable(
          [2., 1., 1.], aggregation=variables_lib.VariableAggregation.MEAN)
    self.evaluate(v.initializer)

    @def_function.function
    def scatter_mul():
      # [FUNCTION DOC]
      # HELPER: compiled per-replica function applying scatter multiply with a replica-index-based
      # sparse update; returns the result.
      ctx = distribute_lib.get_replica_context()
      replica_id = ctx.replica_id_in_sync_group
      value = indexed_slices.IndexedSlices(
          values=array_ops.reshape(
              math_ops.cast(replica_id + 2, dtypes.float32), [1]),
          indices=array_ops.reshape(replica_id, [1]),
          dense_shape=(3,))
      return v.scatter_mul(value)

    per_replica_results = self.evaluate(
        test_util.gather(distribution, distribution.run(scatter_mul)))
    self.assertAllClose([[2., 1.5, 1.], [2., 1.5, 1.]], per_replica_results)

  @combinations.generate(ms_combination)
  def testScatterMin(self, distribution):
    # [FUNCTION DOC]
    # Test: scatter minimum on two variables, one with SUM aggregation and one with
    # ONLY_FIRST_REPLICA. The SUM one must raise a not-implemented error mentioning the operation;
    # the other must give zero, one, zero in both replicas.
    with distribution.scope():
      v1 = variables_lib.Variable(
          [0, 2, 0], aggregation=variables_lib.VariableAggregation.SUM)
      v2 = variables_lib.Variable(
          [0, 2, 0],
          aggregation=variables_lib.VariableAggregation.ONLY_FIRST_REPLICA)
    self.evaluate(variables_lib.global_variables_initializer())

    @def_function.function
    def scatter_min(v):
      # [FUNCTION DOC]
      # HELPER: compiled function applying scatter minimum of a fixed sparse update to the given
      # variable; returns the result.
      value = indexed_slices.IndexedSlices(
          values=array_ops.identity([1]),
          indices=array_ops.identity([1]),
          dense_shape=(3,))
      return v.scatter_min(value)

    with self.assertRaisesRegex(NotImplementedError, "scatter_min.*"):
      self.evaluate(
          test_util.gather(distribution,
                           distribution.run(scatter_min, args=(v1,))))

    per_replica_results = self.evaluate(
        test_util.gather(distribution,
                         distribution.run(scatter_min, args=(v2,))))
    self.assertAllClose([[0, 1, 0], [0, 1, 0]], per_replica_results)

  @combinations.generate(ms_combination)
  def testScatterMax(self, distribution):
    # [FUNCTION DOC]
    # Test: scatter maximum on two variables, one with SUM aggregation and one with
    # ONLY_FIRST_REPLICA. The SUM one must raise a not-implemented error mentioning the operation;
    # the other must give one, zero, zero in both replicas.
    with distribution.scope():
      v1 = variables_lib.Variable(
          [0, 0, 0], aggregation=variables_lib.VariableAggregation.SUM)
      v2 = variables_lib.Variable(
          [0, 0, 0],
          aggregation=variables_lib.VariableAggregation.ONLY_FIRST_REPLICA)
    self.evaluate(variables_lib.global_variables_initializer())

    @def_function.function
    def scatter_max(v):
      # [FUNCTION DOC]
      # HELPER: compiled function applying scatter maximum of a fixed sparse update to the given
      # variable; returns the result.
      value = indexed_slices.IndexedSlices(
          values=array_ops.identity([1]),
          indices=array_ops.identity([0]),
          dense_shape=(3,))
      return v.scatter_max(value)

    with self.assertRaisesRegex(NotImplementedError, "scatter_max.*"):
      self.evaluate(
          test_util.gather(distribution,
                           distribution.run(scatter_max, args=(v1,))))

    per_replica_results = self.evaluate(
        test_util.gather(distribution,
                         distribution.run(scatter_max, args=(v2,))))
    self.assertAllClose([[1, 0, 0], [1, 0, 0]], per_replica_results)

  @combinations.generate(ms_combination)
  def testScatterUpdate(self, distribution):
    # [FUNCTION DOC]
    # Test: scatter update on two variables, one with SUM aggregation and one with
    # ONLY_FIRST_REPLICA. The SUM one must raise a not-implemented error mentioning the operation;
    # the other must give zero, three, zero in both replicas.
    with distribution.scope():
      v1 = variables_lib.Variable(
          [0, 0, 0], aggregation=variables_lib.VariableAggregation.SUM)
      v2 = variables_lib.Variable(
          [0, 0, 0],
          aggregation=variables_lib.VariableAggregation.ONLY_FIRST_REPLICA)
    self.evaluate(variables_lib.global_variables_initializer())

    @def_function.function
    def scatter_update(v):
      # [FUNCTION DOC]
      # HELPER: compiled function applying scatter update of a fixed sparse update to the given
      # variable; returns the result.
      value = indexed_slices.IndexedSlices(
          values=array_ops.identity([3]),
          indices=array_ops.identity([1]),
          dense_shape=(3,))
      return v.scatter_update(value)

    with self.assertRaisesRegex(NotImplementedError, "scatter_update.*"):
      self.evaluate(
          test_util.gather(distribution,
                           distribution.run(scatter_update, args=(v1,))))

    per_replica_results = self.evaluate(
        test_util.gather(distribution,
                         distribution.run(scatter_update, args=(v2,))))
    self.assertAllClose([[0, 3, 0], [0, 3, 0]], per_replica_results)

  @combinations.generate(ms_combination + tpu_combination)
  def testScatterOpsWithNoneAggregation(self, distribution):
    # [FUNCTION DOC]
    # Test: disables the MLIR bridge, then for a variable with aggregation NONE applies each
    # scatter operation (sub, add, max, min, mul, div, update) with a sparse update of two to the
    # initial four, in a compiled per-replica function. Expects both replicas to read the expected
    # value after each step (two, four, four, two, four, two, two).
    config.disable_mlir_bridge()
    def assert_close(v, op, delta, expect):
      # [FUNCTION DOC]
      # HELPER: applies the named scatter method with the given update in each replica and checks
      # that both replicas return the expected value. Inputs: variable, method name, update, and
      # expected value.
      scatter_op = getattr(v, op)

      @def_function.function
      def scatter_xxx():
        # [FUNCTION DOC]
        # HELPER: compiled function applying the chosen scatter operation with the update.
        return scatter_op(delta)

      per_replica_results = self.evaluate(
          variable_utils.convert_variables_to_tensors(
              distribution.experimental_local_results(
                  distribution.run(scatter_xxx))))
      self.assertAllClose([expect, expect], per_replica_results)

    with distribution.scope():
      v = variables_lib.Variable(
          [4.], aggregation=variables_lib.VariableAggregation.NONE)
    self.evaluate(variables_lib.global_variables_initializer())

    delta = indexed_slices.IndexedSlices(
        values=array_ops.identity([2.]),
        indices=array_ops.identity([0]),
        dense_shape=(1,))

    assert_close(v, "scatter_sub", delta, [2.])
    assert_close(v, "scatter_add", delta, [4.])
    assert_close(v, "scatter_max", delta, [4.])
    assert_close(v, "scatter_min", delta, [2.])
    assert_close(v, "scatter_mul", delta, [4.])
    assert_close(v, "scatter_div", delta, [2.])
    assert_close(v, "scatter_update", delta, [2.])

  @combinations.generate(ms_combination + tpu_combination)
  def testScatterOpsInCrossReplicaContext(self, distribution):
    # [FUNCTION DOC]
    # Test: in cross-replica context, scatter add of a sparse update on a SUM variable of ones, and
    # scatter min on a default variable. Expects three, one, one and one, one, one respectively.
    with distribution.scope():
      v1 = variables_lib.Variable(
          [1, 1, 1], aggregation=variables_lib.VariableAggregation.SUM)
      v2 = variables_lib.Variable([1, 1, 1])
    self.evaluate(variables_lib.global_variables_initializer())

    value = indexed_slices.IndexedSlices(
        values=array_ops.identity([2]),
        indices=array_ops.identity([0]),
        dense_shape=(3,))
    with distribution.scope():
      self.evaluate(v1.scatter_add(value))
      self.assertAllEqual([3, 1, 1], self.evaluate(v1.read_value()))

      self.evaluate(v2.scatter_min(value))
      self.assertAllEqual([1, 1, 1], self.evaluate(v2.read_value()))


class OnReadVariableSyncTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering variables created with ON_READ synchronization under a strategy:
  # assignment, aggregation-dependent reads, all-reduce, operator overloading, initialization and
  # checkpoint save/restore.

  @combinations.generate(strategy_and_run_tf_function_combinations())
  def testAssign(self, distribution, experimental_run_tf_function):
    # [FUNCTION DOC]
    # Test: for every update kind, aggregation and context, assigns scalar updates to ON_READ
    # variables. Cross-replica SUM and NONE are skipped. Expects every component to equal one.

    def assign(fn, v, update_value, cross_replica):
      # [FUNCTION DOC]
      # HELPER: applies the named update method with a value in cross-replica or replica context.
      # Returns the update result (gathered across replicas in the replica case).
      update_fn = lambda: getattr(v, fn)(update_value)
      if cross_replica:
        return update_fn()
      else:
        if experimental_run_tf_function:
          update_fn = def_function.function(update_fn)
        return test_util.gather(distribution, distribution.run(update_fn))

    updates = [("assign", 1.), ("assign_add", 1.), ("assign_sub", -1.)]
    aggregations = [
        variables_lib.VariableAggregation.NONE,
        variables_lib.VariableAggregation.SUM,
        variables_lib.VariableAggregation.MEAN,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
    ]
    options = list(
        x for x in itertools.product(updates, aggregations, [True, False]))
    for update, aggregation, cross_replica in options:
      # VariableAggregation.SUM in cross-replica mode is tested below,
      # VariableAggregation.NONE in cross-replica mode is not supported.
      if cross_replica and aggregation in [
          variables_lib.VariableAggregation.SUM,
          variables_lib.VariableAggregation.NONE,
      ]:
        continue
      with distribution.scope():
        v = variable_v1.VariableV1(
            0.,
            synchronization=variables_lib.VariableSynchronization.ON_READ,
            aggregation=aggregation)
      self.evaluate(variables_lib.global_variables_initializer())
      fn, update_value = update
      self.evaluate(assign(fn, v, update_value, cross_replica))
      for component in v._values:
        self.assertAllEqual(self.evaluate(component.read_value()),
                            self.evaluate(array_ops.ones_like(component)))

  @combinations.generate(strategy_and_run_tf_function_combinations())
  def testAssignOnReadVar(self, distribution, experimental_run_tf_function):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the variables under test are created without ON_READ synchronization, so
    # despite the name they are not on-read variables; the expected-value tables for cross-replica
    # SUM are therefore never exercised either, as SUM is skipped.
    # Test: assigns variable values (two, minus two) to variables with various aggregations in both
    # contexts, skipping SUM, and expects each component to match the table of expected values.

    with distribution.scope():
      v_to_assign = variable_v1.VariableV1(
          2., aggregation=variables_lib.VariableAggregation.MEAN)
      v_to_assign_sub = variable_v1.VariableV1(
          -2., aggregation=variables_lib.VariableAggregation.MEAN)

    def assign(fn, v, update_value, cross_replica):
      # [FUNCTION DOC]
      # HELPER: applies the named update method with a value in cross-replica or replica context.
      # Returns the update result (gathered across replicas in the replica case).
      update_fn = lambda: getattr(v, fn)(update_value)
      if cross_replica:
        return update_fn()
      else:
        if experimental_run_tf_function:
          update_fn = def_function.function(update_fn)
        return test_util.gather(distribution, distribution.run(update_fn))

    updates = [("assign", v_to_assign), ("assign_add", v_to_assign),
               ("assign_sub", v_to_assign_sub)]
    expected_cross_replica = {
        variables_lib.VariableAggregation.SUM: 1.0,
        variables_lib.VariableAggregation.MEAN: 2.0,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA: 2.0
    }
    expected_replica = {
        variables_lib.VariableAggregation.SUM: 2.0,
        variables_lib.VariableAggregation.MEAN: 2.0,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA: 2.0
    }
    # aggregation=NONE is not supported for OnReadVariables.
    aggregations = [
        variables_lib.VariableAggregation.SUM,
        variables_lib.VariableAggregation.MEAN,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
    ]
    options = list(
        x for x in itertools.product(updates, aggregations, [True, False]))
    for update, aggregation, cross_replica in options:
      # assign in replica context with SUM does not make sense cause you can
      # just do value * num replicas error is 1. is not a distributed value and
      # is unsupported for aggregation SUM
      if aggregation == variables_lib.VariableAggregation.SUM:
        continue
      with distribution.scope():
        v = variable_v1.VariableV1(
            0.,
            aggregation=aggregation)
      self.evaluate(variables_lib.global_variables_initializer())
      fn, update_value = update
      self.evaluate(assign(fn, v, update_value, cross_replica))
      if cross_replica:
        for component in v._values:
          self.assertAllEqual(expected_cross_replica.get(aggregation),
                              self.evaluate(component.read_value()))
      else:
        for component in v._values:
          self.assertAllEqual(expected_replica.get(aggregation),
                              self.evaluate(component.read_value()))

  @combinations.generate(strategy_and_run_tf_function_combinations())
  def testAssignPerReplicaVal(self, distribution, experimental_run_tf_function):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test unconditionally skips (after the TPU skip) because assigning
    # per-replica values is unsupported, so the rest of the body is dead code.
    # Intended scenario: assign per-replica values to ON_READ variables and expect four for SUM
    # aggregation and two otherwise.

    if strategy_test_lib.is_tpu_strategy(distribution):
      self.skipTest("Assigning PerReplica values is not supported. See"
                    " sponge/80ba41f8-4220-4516-98ce-bbad48f9f11a.")

    self.skipTest(
        "We don't support assigning PerReplica values in cross "
        "replica context or replica context. see error in "
        "sponge/2b2e54c1-eda6-4534-82e1-c73b1dcd517f."
    )

    with distribution.scope():
      per_replica_value = values.PerReplica(
          [constant_op.constant(2.0),
           constant_op.constant(2.0)])

    def assign(fn, v, update_value, cross_replica):
      # [FUNCTION DOC]
      # HELPER: applies the named update method with a value in cross-replica or replica context.
      # Returns the update result (gathered across replicas in the replica case).
      update_fn = lambda: getattr(v, fn)(update_value)
      if cross_replica:
        return update_fn()
      else:
        if experimental_run_tf_function:
          update_fn = def_function.function(update_fn)
        return test_util.gather(distribution, distribution.run(update_fn))

    updates = [("assign", per_replica_value)]
    # We don't support assigning PerReplica valus to vars in replica context
    # with aggregation=NONE.
    aggregations = [
        variables_lib.VariableAggregation.SUM,
        variables_lib.VariableAggregation.MEAN,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
    ]
    options = list(
        x for x in itertools.product(updates, aggregations, [True, False]))
    for update, aggregation, cross_replica in options:
      # assign in replica context with SUM does not make sense cause you can
      # just do value * num replicas error is 1. is not a distributed value and
      # is unsupported for aggregation SUM
      with distribution.scope():
        v = variable_v1.VariableV1(
            0.,
            synchronization=variables_lib.VariableSynchronization.ON_READ,
            aggregation=aggregation)
      self.evaluate(variables_lib.global_variables_initializer())
      fn, update_value = update
      # with self.assertRaisesRegex(ValueError, "Attempt to convert a value "):
      self.evaluate(assign(fn, v, update_value, cross_replica))
      if aggregation == variables_lib.VariableAggregation.SUM:
        expected = 4.0
      else:
        expected = 2.0
      for component in v._values:
        self.assertAllEqual(expected, self.evaluate(component.read_value()))

  @combinations.generate(strategy_and_run_tf_function_combinations())
  def testAssignDtypeConversion(self, distribution,
                                experimental_run_tf_function):
    # [FUNCTION DOC]
    # Test: assigns integer updates to float ON_READ variables, checking dtype conversion, for all
    # update kinds and aggregations (skipping cross-replica SUM and NONE). Expects every component
    # to equal one.

    def assign(fn, v, update_value, cross_replica):
      # [FUNCTION DOC]
      # HELPER: applies the named update method with a value in cross-replica or replica context.
      # Returns the update result (gathered across replicas in the replica case).
      update_fn = lambda: getattr(v, fn)(update_value)
      if cross_replica:
        return update_fn()
      else:
        if experimental_run_tf_function:
          update_fn = def_function.function(update_fn)
        return test_util.gather(distribution, distribution.run(update_fn))

    updates = [("assign", 1), ("assign_add", 1), ("assign_sub", -1)]
    aggregations = [
        variables_lib.VariableAggregation.NONE,
        variables_lib.VariableAggregation.SUM,
        variables_lib.VariableAggregation.MEAN,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
    ]
    options = list(
        x for x in itertools.product(updates, aggregations, [True, False]))
    for update, aggregation, cross_replica in options:
      # VariableAggregation.SUM in cross-replica mode is tested below,
      # VariableAggregation.NONE in cross-replica mode is not supported.
      if cross_replica and aggregation in [
          variables_lib.VariableAggregation.SUM,
          variables_lib.VariableAggregation.NONE,
      ]:
        continue
      with distribution.scope():
        v = variable_v1.VariableV1(
            0.,
            synchronization=variables_lib.VariableSynchronization.ON_READ,
            aggregation=aggregation)
      self.evaluate(variables_lib.global_variables_initializer())
      fn, update_value = update
      self.evaluate(assign(fn, v, update_value, cross_replica))
      for component in v._values:
        self.assertAllEqual(self.evaluate(component.read_value()),
                            self.evaluate(array_ops.ones_like(component)))

  @combinations.generate(strategy_with_var_policy())
  def testAssignWithAggregationSum(self, distribution):
    # [FUNCTION DOC]
    # Test: assigns the number of replicas to a SUM-aggregated ON_READ variable in cross-replica
    # context. Expects every component to equal one.
    with distribution.scope():
      v = variable_v1.VariableV1(
          0.,
          synchronization=variables_lib.VariableSynchronization.ON_READ,
          aggregation=variables_lib.VariableAggregation.SUM)
    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(v.assign(1. * distribution.num_replicas_in_sync))
    for component in v._values:
      self.assertAllEqual(self.evaluate(component.read_value()),
                          self.evaluate(array_ops.ones_like(component)))

  @combinations.generate(strategy_with_var_policy())
  def testAssignAddSubWithAggregationSum(self, distribution):
    # [FUNCTION DOC]
    # Test: assign add and assign sub on a SUM-aggregated ON_READ variable in cross-replica
    # context. Expects a value error saying the on-read variable does not support them.
    with distribution.scope():
      v = variable_v1.VariableV1(
          0.,
          synchronization=variables_lib.VariableSynchronization.ON_READ,
          aggregation=variables_lib.VariableAggregation.SUM)
    self.evaluate(variables_lib.global_variables_initializer())
    with self.assertRaisesRegex(
        ValueError, "SyncOnReadVariable does not support "):
      self.evaluate(v.assign_add(1.))
    with self.assertRaisesRegex(
        ValueError, "SyncOnReadVariable does not support "):
      self.evaluate(v.assign_sub(1.))

  @combinations.generate(strategy_and_run_tf_function_combinations())
  def testReadValueInReplicaContext(self, distribution,
                                    experimental_run_tf_function):
    # [FUNCTION DOC]
    # Test: for each aggregation, reads an ON_READ variable in replica context (optionally via a
    # compiled function). Expects results to equal the matching component values.
    aggregations = [
        variables_lib.VariableAggregation.NONE,
        variables_lib.VariableAggregation.SUM,
        variables_lib.VariableAggregation.MEAN,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
    ]
    for aggregation in aggregations:
      with distribution.scope():
        v = variable_v1.VariableV1(
            0.,
            synchronization=variables_lib.VariableSynchronization.ON_READ,
            aggregation=aggregation)
      self.evaluate(variables_lib.global_variables_initializer())
      if experimental_run_tf_function:
        read_var_fn = def_function.function(v.read_value)
      else:
        read_var_fn = v.read_value
      results = self.evaluate(
          test_util.gather(distribution, distribution.run(read_var_fn)))
      for component, value in zip(v._values, results):
        self.assertAllEqual(self.evaluate(component.read_value()), value)

  @combinations.generate(strategy_and_run_tf_function_combinations())
  def testReadValueInCrossReplicaContext(self, distribution,
                                         experimental_run_tf_function):
    # [FUNCTION DOC]
    # Test: for SUM, MEAN and first-replica aggregation (initializing the TPU system first on TPU
    # strategies), each replica assigns its replica index; then the variable is read in
    # cross-replica context in several ways. Expects the sum, the mean, or zero respectively.
    aggregations = [
        variables_lib.VariableAggregation.SUM,
        variables_lib.VariableAggregation.MEAN,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
    ]
    for aggregation in aggregations:
      if strategy_test_lib.is_tpu_strategy(distribution):
        resolver = tpu_cluster_resolver.TPUClusterResolver("")
        tpu_cluster_resolver.initialize_tpu_system(resolver)
      with distribution.scope():
        v = variable_v1.VariableV1(
            0.,
            synchronization=variables_lib.VariableSynchronization.ON_READ,
            aggregation=aggregation)
      self.evaluate(variables_lib.global_variables_initializer())

      def assign(v=v):
        # [FUNCTION DOC]
        # HELPER: assigns the current replica index (as float) to the bound variable and returns
        # the result. The variable is bound through a default argument to avoid loop capture.
        ctx = distribute_lib.get_replica_context()
        replica_id = ctx.replica_id_in_sync_group
        return v.assign(math_ops.cast(replica_id, dtypes.float32))

      if experimental_run_tf_function:
        assign = def_function.function(assign)

      self.evaluate(test_util.gather(distribution, distribution.run(assign)))
      num_replicas = distribution.num_replicas_in_sync
      sum_of_replica_values = num_replicas * (num_replicas - 1) / 2.
      if aggregation == variables_lib.VariableAggregation.SUM:
        expected = sum_of_replica_values
      elif aggregation == variables_lib.VariableAggregation.MEAN:
        expected = sum_of_replica_values / num_replicas
      else:
        expected = 0
      self.assertEqual(expected, self.evaluate(v.read_value()), aggregation)
      self.assertEqual(expected, self.evaluate(v.value()), aggregation)
      self.assertEqual(expected, self.evaluate(v), aggregation)
      self.assertEqual(expected, self.evaluate(array_ops.identity(v)),
                       aggregation)

  @combinations.generate(strategy_and_run_tf_function_combinations())
  def testAllReduce(self, distribution, experimental_run_tf_function):
    # [FUNCTION DOC]
    # KNOWN ISSUE: this test lives in the ON_READ class but its variable is created with ON_WRITE
    # synchronization.
    # Test: variable of two with mean aggregation; each replica all-reduces it with a sum and adds
    # its replica index. Expects two times the number of replicas plus the replica index.
    with distribution.scope():
      v = variable_v1.VariableV1(
          2.,
          synchronization=variables_lib.VariableSynchronization.ON_WRITE,
          aggregation=variables_lib.VariableAggregation.MEAN)
    self.evaluate(variables_lib.global_variables_initializer())

    def all_reduce():
      # [FUNCTION DOC]
      # HELPER: per-replica function returning the sum all-reduce of the variable plus the replica
      # index as a float.
      ctx = distribute_lib.get_replica_context()
      replica_id = ctx.replica_id_in_sync_group
      return ctx.all_reduce("SUM", v) + math_ops.cast(replica_id,
                                                      dtypes.float32)

    if experimental_run_tf_function:
      all_reduce = def_function.function(all_reduce)

    per_replica_results = self.evaluate(
        test_util.gather(distribution, distribution.run(all_reduce)))
    expected_result = []
    for i in range(distribution.num_replicas_in_sync):
      expected_result.append(2.0 * distribution.num_replicas_in_sync +
                             1.0 * i)
    self.assertAllEqual(per_replica_results, tuple(expected_result))

  @combinations.generate(strategy_and_run_tf_function_combinations())
  def testAssignPerReplicaBeforeRead(self, distribution,
                                     experimental_run_tf_function):
    # [FUNCTION DOC]
    # Test: for SUM, MEAN and first-replica aggregation, each replica assigns its replica index to
    # an ON_READ variable, and the per-replica results are checked. Expects the replica indices.
    aggregations = [
        variables_lib.VariableAggregation.SUM,
        variables_lib.VariableAggregation.MEAN,
        variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
    ]
    for aggregation in aggregations:
      with distribution.scope():
        v = variable_v1.VariableV1(
            0.,
            synchronization=variables_lib.VariableSynchronization.ON_READ,
            aggregation=aggregation)
      self.evaluate(variables_lib.global_variables_initializer())

      def assign(var=v):
        # [FUNCTION DOC]
        # HELPER: assigns the current replica index (as float) to the bound variable and returns
        # the result. The variable is bound through a default argument.
        ctx = distribute_lib.get_replica_context()
        replica_id = ctx.replica_id_in_sync_group
        return var.assign(math_ops.cast(replica_id, dtypes.float32))

      if experimental_run_tf_function:
        assign = def_function.function(assign)

      per_replica_results = self.evaluate(
          test_util.gather(distribution, distribution.run(assign)))
      expected_result = []
      for i in range(distribution.num_replicas_in_sync):
        expected_result.append(1.0 * i)
      self.assertAllEqual(per_replica_results, tuple(expected_result))

  @combinations.generate(strategy_with_var_policy())
  def testReadValueWithAggregationNoneInCrossReplicaContext(self, distribution):
    # [FUNCTION DOC]
    # Test: reading an ON_READ variable with aggregation NONE in cross-replica context. Expects a
    # value error saying it cannot be converted for that aggregation.
    with distribution.scope():
      v = variable_v1.VariableV1(
          0.,
          synchronization=variables_lib.VariableSynchronization.ON_READ,
          aggregation=variables_lib.VariableAggregation.NONE)
    self.evaluate(variables_lib.global_variables_initializer())
    with self.assertRaisesRegex(
        ValueError, "Could not convert from .* VariableAggregation\\.NONE"):
      self.evaluate(v.read_value())

  @combinations.generate(strategy_with_var_policy())
  def testInitializedToSameValueInsideEagerRun(self, distribution):
    # [FUNCTION DOC]
    # Test (eager only; skipped for multi-worker collective strategy): lazily creates an ON_READ
    # variable with random initial value inside a per-replica function within a compiled step.
    # Expects both replicas' component values to be equal.
    if not context.executing_eagerly(): self.skipTest("eager only")
    if isinstance(distribution.extended,
                  collective_all_reduce_strategy.CollectiveAllReduceExtended):
      self.skipTest("Test for more than 1 device per worker only.")

    v = [None]
    @def_function.function
    def step():
      # [FUNCTION DOC]
      # HELPER: compiled step that runs the inner function on every replica.
      def f():
        # [FUNCTION DOC]
        # HELPER: per-replica function creating the ON_READ variable the first time it is called.
        if v[0] is None:
          v[0] = variables_lib.Variable(
              random_ops.random_normal([]),
              synchronization=variables_lib.VariableSynchronization.ON_READ)

      distribution.run(f)

    context.set_global_seed(None)
    step()
    vals = self.evaluate(v[0].values)
    self.assertAllEqual(vals[0], vals[1])

  @combinations.generate(strategy_with_var_policy())
  def testOperatorOverride(self, distribution):
    # [FUNCTION DOC]
    # Test: mean-aggregated ON_READ variable; replicas are assigned their replica indices; expects
    # adding one in cross-replica context to give one and a half, and per-replica additions to
    # give one and two.

    with distribution.scope():
      v = variable_v1.VariableV1(
          0.0,
          synchronization=variables_lib.VariableSynchronization.ON_READ,
          aggregation=variables_lib.VariableAggregation.MEAN)
      self.evaluate(variables_lib.global_variables_initializer())

      @def_function.function
      def assign():
        # [FUNCTION DOC]
        # HELPER: compiled function assigning the replica index (as float) to the variable.
        ctx = distribute_lib.get_replica_context()
        replica_id = ctx.replica_id_in_sync_group
        return v.assign(math_ops.cast(replica_id, dtypes.float32))

      # Assign different replicas with different values.
      self.evaluate(test_util.gather(distribution, distribution.run(assign)))
      self.assertEqual(1.5, self.evaluate(v + 1))

      @def_function.function
      def add():
        # [FUNCTION DOC]
        # HELPER: compiled function returning the variable plus one.
        return v + 1

      per_replica_results = self.evaluate(
          test_util.gather(distribution, distribution.run(add)))
      self.assertAllEqual([1, 2], per_replica_results)

  @combinations.generate(
      combinations.combine(
          strategy=[
              strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
              strategy_combinations.tpu_strategy,
              strategy_combinations.tpu_strategy_packed_var,
              strategy_combinations.multi_worker_mirrored_2x1_cpu,
              strategy_combinations.multi_worker_mirrored_2x1_gpu,
          ],
          mode=["eager"],
          use_var_policy=[True, False]))
  def testSaveAndRestoreOnRead(self, strategy):
    # [FUNCTION DOC]
    # Test (eager only): checkpoint round trips for ON_READ variables with SUM and MEAN. Replicas
    # are assigned three or four, then saved and restored into the same variable (expects three and
    # a half), into a plain variable (seven for SUM, three and a half for MEAN), and a saved plain
    # variable is restored into the ON_READ variable (one for SUM, two for MEAN). Checkpoints go to
    # uniquely named folders under the temporary directory.
    aggregation = [variable_scope.VariableAggregation.SUM,
                   variable_scope.VariableAggregation.MEAN]
    for agg in aggregation:
      v_normal_restore = variables_lib.Variable(1.0)
      v_normal_save = variables_lib.Variable(2.0)

      with strategy.scope():
        v_on_read = variables_lib.Variable(
            1.0, synchronization=variable_scope.VariableSynchronization.ON_READ,
            aggregation=agg)

        @def_function.function
        def assign_fn():
          # [FUNCTION DOC]
          # HELPER: compiled function assigning three or four to the ON_READ variable depending on
          # whether this is a worker task or the replica index equals one.
          cluster_resolver = strategy.cluster_resolver
          replica_ctx = distribute_lib.get_replica_context()
          if ((cluster_resolver and cluster_resolver.task_type == "worker") or
              math_ops.equal(replica_ctx.replica_id_in_sync_group,
                             constant_op.constant(1))):
            v_on_read.assign(3.)  # pylint:disable=cell-var-from-loop
          else:
            v_on_read.assign(4.)  # pylint:disable=cell-var-from-loop

        strategy.run(assign_fn)

        # Save ONREAD, restore ONREAD
        # Saves v[0] + v[1] = 7 for SUM and 3.5 for MEAN.
        ckpt = trackable_utils.Checkpoint(var=v_on_read)
        manager = ckpt_manager.CheckpointManager(
            ckpt, "/tmp/ckpt_" + str(uuid.uuid4()), max_to_keep=None)
        manager.save()
        # Restores a value of 7/2 = 3.5 for SUM and 3.5 for MEAN.
        ckpt.restore(manager.latest_checkpoint)
        self.assertEqual(3.5, self.evaluate(v_on_read._values[0]))

        # Save ONREAD, restore normal
        ckpt_normal = trackable_utils.Checkpoint(var=v_normal_restore)
        ckpt_normal.restore(manager.latest_checkpoint)
        if agg == variable_scope.VariableAggregation.SUM:
          self.assertEqual(7.0, self.evaluate(v_normal_restore.read_value()))
        else:
          self.assertEqual(3.5, self.evaluate(v_normal_restore.read_value()))

        # Save normal, restore ONREAD
        ckpt = trackable_utils.Checkpoint(var=v_normal_save)
        manager = ckpt_manager.CheckpointManager(
            ckpt, "/tmp/ckpt_" + str(uuid.uuid4()), max_to_keep=None)
        manager.save()
        # Restores a value of 2/2 = 1.0 for SUM and 2.0 for MEAN.
        ckpt_on_read = trackable_utils.Checkpoint(var=v_on_read)
        ckpt_on_read.restore(manager.latest_checkpoint)
        if agg == variable_scope.VariableAggregation.SUM:
          self.assertEqual(1.0, self.evaluate(v_on_read._values[0]))
        else:
          self.assertEqual(2.0, self.evaluate(v_on_read._values[0]))


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
            strategy_combinations.multi_worker_mirrored_2x1_cpu,
            strategy_combinations.multi_worker_mirrored_2x1_gpu,
        ],
        aggregation=[
            variables_lib.VariableAggregation.MEAN,
            variables_lib.VariableAggregation.SUM,
            variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
        ],
        mode=["graph", "eager"],
        use_var_policy=[True, False]))
class SyncOnReadScatterReplicaTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case (parameterized over strategies, aggregations, modes and variable policy) checking
  # that every scatter operation on ON_READ variables with per-replica sparse updates raises a
  # not-implemented error.

  def testScatterSub(self, distribution, aggregation):
    # [FUNCTION DOC]
    # Test: scatter subtract on an ON_READ variable with per-replica sparse updates. Expects a
    # not-implemented error.
    with distribution.scope():
      v = variables_lib.Variable(
          [1., 1., 1.],
          synchronization=variables_lib.VariableSynchronization.ON_READ,
          aggregation=aggregation)
    self.evaluate(v.initializer)

    delta = values.PerReplica([
        indexed_slices.IndexedSlices(
            values=[[0.], [1.]], indices=[0, 1], dense_shape=(3,)),
        indexed_slices.IndexedSlices(
            values=[[1.], [2.]], indices=[1, 2], dense_shape=(3,)),
    ])

    with self.assertRaises(NotImplementedError):
      self.evaluate(distribution.run(v.scatter_sub, args=(delta,)))

  def testScatterAdd(self, distribution, aggregation):
    # [FUNCTION DOC]
    # Test: scatter add on an ON_READ variable with per-replica sparse updates. Expects a
    # not-implemented error.
    with distribution.scope():
      v = variables_lib.Variable(
          [1., 1., 1.],
          synchronization=variables_lib.VariableSynchronization.ON_READ,
          aggregation=aggregation)
    self.evaluate(v.initializer)

    delta = values.PerReplica([
        indexed_slices.IndexedSlices(
            values=[[0.], [1.]], indices=[0, 1], dense_shape=(3,)),
        indexed_slices.IndexedSlices(
            values=[[1.], [2.]], indices=[1, 2], dense_shape=(3,)),
    ])

    with self.assertRaises(NotImplementedError):
      self.evaluate(distribution.run(v.scatter_add, args=(delta,)))

  def testScatterDiv(self, distribution, aggregation):
    # [FUNCTION DOC]
    # Test: scatter divide on an ON_READ variable with per-replica sparse updates. Expects a
    # not-implemented error.
    with distribution.scope():
      v = variables_lib.Variable(
          [2., 6., 1.],
          synchronization=variables_lib.VariableSynchronization.ON_READ,
          aggregation=aggregation)
    self.evaluate(v.initializer)

    delta = values.PerReplica([
        indexed_slices.IndexedSlices(
            values=[[2.], [2.]], indices=[0, 1], dense_shape=(3,)),
        indexed_slices.IndexedSlices(
            values=[[3.], [3.]], indices=[1, 2], dense_shape=(3,)),
    ])

    with self.assertRaises(NotImplementedError):
      self.evaluate(distribution.run(v.scatter_div, args=(delta,)))

  def testScatterMul(self, distribution, aggregation):
    # [FUNCTION DOC]
    # Test: scatter multiply on an ON_READ variable with per-replica sparse updates. Expects a
    # not-implemented error.
    with distribution.scope():
      v = variables_lib.Variable(
          [2., 1., 1.],
          synchronization=variables_lib.VariableSynchronization.ON_READ,
          aggregation=aggregation)
    self.evaluate(v.initializer)

    delta = values.PerReplica([
        indexed_slices.IndexedSlices(
            values=[[2.], [3.]], indices=[0, 1], dense_shape=(3,)),
        indexed_slices.IndexedSlices(
            values=[[4.], [5.]], indices=[1, 2], dense_shape=(3,)),
    ])

    with self.assertRaises(NotImplementedError):
      self.evaluate(distribution.run(v.scatter_mul, args=(delta,)))

  def testScatterMin(self, distribution, aggregation):
    # [FUNCTION DOC]
    # Test: scatter minimum on an ON_READ variable with per-replica sparse updates. Expects a
    # not-implemented error.
    with distribution.scope():
      v = variables_lib.Variable(
          [3., 4., 5.],
          synchronization=variables_lib.VariableSynchronization.ON_READ,
          aggregation=aggregation)
    self.evaluate(v.initializer)

    delta = values.PerReplica([
        indexed_slices.IndexedSlices(
            values=[[1.], [8.]], indices=[0, 1], dense_shape=(3,)),
        indexed_slices.IndexedSlices(
            values=[[9.], [2.]], indices=[1, 2], dense_shape=(3,)),
    ])

    with self.assertRaises(NotImplementedError):
      self.evaluate(distribution.run(v.scatter_min, args=(delta,)))

  def testScatterMax(self, distribution, aggregation):
    # [FUNCTION DOC]
    # Test: scatter maximum on an ON_READ variable with per-replica sparse updates. Expects a
    # not-implemented error.
    with distribution.scope():
      v = variables_lib.Variable(
          [3., 4., 5.],
          synchronization=variables_lib.VariableSynchronization.ON_READ,
          aggregation=aggregation)
    self.evaluate(v.initializer)

    delta = values.PerReplica([
        indexed_slices.IndexedSlices(
            values=[[1.], [8.]], indices=[0, 1], dense_shape=(3,)),
        indexed_slices.IndexedSlices(
            values=[[9.], [2.]], indices=[1, 2], dense_shape=(3,)),
    ])

    with self.assertRaises(NotImplementedError):
      self.evaluate(distribution.run(v.scatter_max, args=(delta,)))

  def testScatterUpdate(self, distribution, aggregation):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite the name, the test calls scatter minimum, not scatter update.
    # Test: runs scatter minimum on an ON_READ variable with per-replica sparse updates. Expects a
    # not-implemented error.
    with distribution.scope():
      v = variables_lib.Variable(
          [0., 0., 0.],
          synchronization=variables_lib.VariableSynchronization.ON_READ,
          aggregation=aggregation)
    self.evaluate(v.initializer)

    delta = values.PerReplica([
        indexed_slices.IndexedSlices(
            values=[[1.], [2.]], indices=[0, 1], dense_shape=(3,)),
        indexed_slices.IndexedSlices(
            values=[[3.], [4.]], indices=[1, 2], dense_shape=(3,)),
    ])

    with self.assertRaises(NotImplementedError):
      self.evaluate(distribution.run(v.scatter_min, args=(delta,)))


if __name__ == "__main__":
  test_util.main()
