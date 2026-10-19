# [MODULE DOC]
# Tests for distributed variables created under distribution strategies (mirrored, TPU,
# central storage, multi-worker mirrored, parameter server). Defines a shared checkpoint
# round-trip helper, a strategy combination helper, a heavily parameterized test class that
# covers variable behavior (type, checkpointing, copying, assignment, update, saving without
# distribution, unsaveable cases) over strategy, synchronization, aggregation and mode, and a
# test class for packed distributed variables in eager mode. Running the file as a script
# starts the distributed test runner. No import-time side effects besides test registration
# through the combination decorators.
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
"""Tests for the distributed variables library."""

import copy
import os

from absl.testing import parameterized
from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import packed_distributed_variable as packed
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import ps_values
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import test_util as ds_test_util
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute import values as values_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.saved_model import save
from tensorflow.python.saved_model import save_context
from tensorflow.python.saved_model import save_options
from tensorflow.python.types import core


def _device_str(d):
  # [FUNCTION DOC]
  # HELPER: small utility for building device names in tests.
  # Takes a device index (integer) and returns the GPU device name string for that index.
  # No side effects. (Appears unused in this file.)
  return "/device:GPU:" + str(d)


def _nested_value(d):
  # [FUNCTION DOC]
  # HELPER: builds sample nested structures for tests.
  # Takes a string suffix and returns a tuple containing strings, a list and a mapping, each
  # with the suffix appended, to exercise nested-structure handling. No side effects.
  # (Appears unused in this file.)
  return ("a" + d, ["b" + d, {"c": "d" + d, "e": "f" + d}, "g" + d], "h" + d)


def mirrored_and_tpu_strategy_combinations():
  # [FUNCTION DOC]
  # HELPER: supplies strategy combinations for parameterized tests.
  # Takes no inputs. Returns a combination of GPU/CPU mirrored, two-GPU mirrored without merge
  # call, TPU, and packed-variable TPU strategies, crossed with graph and eager modes.
  # (Appears unused in this file.)
  return combinations.combine(
      distribution=[
          strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          strategy_combinations.mirrored_strategy_with_two_gpus_no_merge_call,
          strategy_combinations.tpu_strategy,
          strategy_combinations.tpu_strategy_packed_var,
      ],
      mode=["graph", "eager"])


def checkpoint_test_helper(dvar_test, distribution,
                           synchronization, aggregation, enable_async_ckpt):
  # [FUNCTION DOC]
  # HELPER: shared body for the synchronous and asynchronous checkpointing tests.
  # Inputs: the running test case, a distribution strategy, a synchronization mode, an
  # aggregation mode, and a flag enabling asynchronous checkpointing. Creates a distributed
  # variable in the strategy scope, saves a checkpoint to a temp directory, assigns different
  # values, restores, and asserts the original values return. Then repeats a second round of
  # save, assign and restore with new values to cover reuse of an already created CPU copy.
  # Returns nothing; raises assertion failures on mismatch. Writes checkpoint files.
  # This method is added since `testCheckpointing` cannot be parameterized after
  # the entire class is parameterized.
  with distribution.scope():
    v = variables_lib.Variable(
        constant_op.constant([1., 2., 3., 4]),
        synchronization=synchronization,
        aggregation=aggregation)

  dvar_test.evaluate(v.initializer)
  before_save = dvar_test.evaluate(v.read_value())

  # Save random weights into checkpoint.
  checkpoint = trackable_utils.Checkpoint(v=v)
  ckpt_options = checkpoint_options.CheckpointOptions(
      experimental_enable_async_checkpoint=enable_async_ckpt)
  prefix = os.path.join(dvar_test.get_temp_dir(), "ckpt")
  with dvar_test.test_session():
    save_path = checkpoint.save(file_prefix=prefix, options=ckpt_options)

  # Assign inverted value.
  dvar_test.evaluate(v.assign(constant_op.constant([4., 3., 2., 1.])))
  after_assign = dvar_test.evaluate(v.read_value())
  dvar_test.assertNotAllClose(before_save, after_assign)

  # Restore from the checkpoint.
  with dvar_test.test_session():
    checkpoint.restore(save_path).assert_consumed().run_restore_ops()
  after_restore = dvar_test.evaluate(v)
  dvar_test.assertAllClose(before_save, after_restore)

  # Another round of saving/restoring to ensure that the logic of
  # _copy_trackable_to_cpu works when a copy is already created in object_map.
  dvar_test.evaluate(v.assign(constant_op.constant([5., 6., 7., 8.])))
  before_save_1 = dvar_test.evaluate(v.read_value())
  dvar_test.assertNotAllClose(before_save_1, after_restore)
  with dvar_test.test_session():
    save_path = checkpoint.save(file_prefix=prefix, options=ckpt_options)

  dvar_test.evaluate(v.assign(constant_op.constant([8., 7., 6., 5.])))
  after_assign_1 = dvar_test.evaluate(v.read_value())
  dvar_test.assertNotAllClose(before_save_1, after_assign_1)

  with dvar_test.test_session():
    checkpoint.restore(save_path).assert_consumed().run_restore_ops()
  after_restore_1 = dvar_test.evaluate(v)
  dvar_test.assertAllClose(before_save_1, after_restore_1)


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.mirrored_strategy_with_one_cpu,
            strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
            strategy_combinations.mirrored_strategy_with_two_gpus_no_merge_call,
            strategy_combinations.tpu_strategy,
            strategy_combinations.tpu_strategy_packed_var,
            strategy_combinations.tpu_strategy_spmd,
            strategy_combinations.central_storage_strategy_with_gpu_and_cpu,
            strategy_combinations.multi_worker_mirrored_2x1_cpu,
            strategy_combinations.multi_worker_mirrored_2x1_gpu,
            strategy_combinations.multi_worker_mirrored_2x2_gpu,
            strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call,
        ],
        synchronization=[
            variables_lib.VariableSynchronization.ON_READ,
            variables_lib.VariableSynchronization.ON_WRITE,
        ],
        aggregation=[
            variables_lib.VariableAggregation.MEAN,
            variables_lib.VariableAggregation.SUM,
            variables_lib.VariableAggregation.ONLY_FIRST_REPLICA,
        ],
        mode=["graph", "eager"],
        use_var_policy=[True, False]))
class DistributedVariableTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized tests for distributed variables. The decorator crosses many strategies
  # (mirrored, TPU, central storage, multi-worker mirrored) with ON_READ/ON_WRITE
  # synchronization, MEAN/SUM/only-first-replica aggregation, graph/eager mode and the
  # variable-policy flag. Each test method receives the matching parameters from that cross.

  def testExtendsVariable(self, distribution, synchronization, aggregation):
    # [FUNCTION DOC]
    # Creates a variable inside the strategy scope with the given synchronization and
    # aggregation, and asserts it is an instance of the base variable class.
    with distribution.scope():
      v = variables_lib.Variable(
          1., synchronization=synchronization, aggregation=aggregation)
    self.assertIsInstance(v, variables_lib.Variable)

  def testCheckpointing(self, distribution, synchronization, aggregation, mode):
    # [FUNCTION DOC]
    # Skips multi-worker mirrored strategies in graph mode, otherwise runs the shared
    # checkpoint save/assign/restore helper with asynchronous checkpointing disabled and
    # expects the restored values to equal the saved ones.
    if (isinstance(distribution,
                   collective_all_reduce_strategy.CollectiveAllReduceStrategy)
        and mode == "graph"):
      self.skipTest("MWMS combinations tests do not work well in graph mode.")

    checkpoint_test_helper(self, distribution, synchronization, aggregation,
                           enable_async_ckpt=False)

  def testAsyncCheckpointing(self, distribution, synchronization,
                             aggregation, mode):
    # [FUNCTION DOC]
    # Same as the synchronous checkpointing test but with asynchronous checkpointing enabled.
    # Skips multi-worker mirrored strategies in graph mode; otherwise runs the shared helper
    # and expects the restored values to equal the saved ones.
    if (isinstance(distribution,
                   collective_all_reduce_strategy.CollectiveAllReduceStrategy)
        and mode == "graph"):
      self.skipTest("MWMS combinations tests do not work well in graph mode.")

    checkpoint_test_helper(self, distribution, synchronization, aggregation,
                           enable_async_ckpt=True)

  def testTraceback(self, distribution, synchronization, aggregation):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name suggests a traceback check, but it only verifies that creating a
    # second variable with an existing name raises a value error.
    # Skipped in eager mode. In graph mode, creates a resource variable named by get-variable
    # in the strategy scope, then asserts that creating it again raises a value error saying
    # the variable already exists.
    if context.executing_eagerly():
      self.skipTest("does not apply to eager")
    with distribution.scope():
      variable_scope.get_variable(
          name="testVar",
          initializer=1.,
          use_resource=True,
          synchronization=synchronization,
          aggregation=aggregation)
      with self.assertRaisesRegex(ValueError,
                                  "Variable testVar already exists"):
        variable_scope.get_variable(
            name="testVar",
            initializer=1.,
            use_resource=True,
            synchronization=synchronization,
            aggregation=aggregation)

  def testSelectReplica(self, distribution, synchronization, aggregation):
    # [FUNCTION DOC]
    # Creates a distributed variable and asserts that selecting replica zero from it returns
    # the very same object (identity), since variables are not per-replica containers here.
    with distribution.scope():
      v = variables_lib.Variable(
          1., synchronization=synchronization, aggregation=aggregation)
    self.assertIs(v, distribute_utils.select_replica(0, v))

  def testIsTensorLike(self, distribution, synchronization, aggregation):
    # [FUNCTION DOC]
    # Skips TPU strategies in eager mode. Creates a distributed variable and asserts it is an
    # instance of the tensor-like core type both in cross-replica context and inside a
    # function run in replica context.
    if isinstance(distribution.extended,
                  tpu_strategy.TPUExtended) and context.executing_eagerly():
      self.skipTest("TPU doesn't support pure eager")

    with distribution.scope():
      v = variables_lib.Variable(
          0., synchronization=synchronization, aggregation=aggregation)
    # In cross replica context.
    self.assertIsInstance(v, core.Tensor)
    # In replica context.
    distribution.run(lambda v: self.assertIsInstance(v, core.Tensor), args=(v,))

  def testAssignReturnValueIsTensorLike(self, distribution, synchronization,
                                        aggregation):
    # [FUNCTION DOC]
    # Checks that the values returned by assign, assign-sub and assign-add are tensor-like.
    # Skips TPU strategies (eager unsupported, graph has a known bug). Creates a variable,
    # then checks the return values in cross-replica context only for ON_READ variables with
    # non-SUM aggregation, and always checks them inside replica context via a run call.
    if isinstance(distribution.extended, tpu_strategy.TPUExtended):
      if context.executing_eagerly():
        self.skipTest("TPU doesn't support pure eager")
      else:
        self.skipTest("b/152076846")

    with distribution.scope():
      v = variables_lib.Variable(
          0., synchronization=synchronization, aggregation=aggregation)

    def assert_is_tensor_like(v):
      # [FUNCTION DOC]
      # HELPER: assertion helper for the assign return value test.
      # Given a variable, assigns, subtracts and adds a constant-derived tensor and asserts each
      # returned value is a tensor-like object. Returns nothing.
      # We can't use Python literals because they are treated as non-distributed
      # values is not allowed when aggregation is SUM. See
      # `cross_device_ops.reduce_non_distributed_value`.
      delta = array_ops.identity(1.)
      self.assertIsInstance(v.assign(delta), core.Tensor)
      self.assertIsInstance(v.assign_sub(delta), core.Tensor)
      self.assertIsInstance(v.assign_add(delta), core.Tensor)

    # In cross replica context we return a PerReplica which is not Tensor like
    # all the time yet.
    if (synchronization == variables_lib.VariableSynchronization.ON_READ and
        aggregation != variables_lib.VariableAggregation.SUM):
      assert_is_tensor_like(v)

    # In replica context.
    distribution.run(assert_is_tensor_like, args=(v,))

  def testDeepCopy(self, distribution, synchronization, aggregation):
    # [FUNCTION DOC]
    # Eager mode only. Deep-copies a distributed variable both inside and outside the strategy
    # scope, initializes variables, and (except on TPU) runs a comparison in replica context
    # asserting each copy has the same type, aggregation and strategy, distinct component
    # variables with equal devices and values, and a copied (distinct) policy when one exists.
    if not context.executing_eagerly():
      self.skipTest("deepcopy only supported in eager mode")

    with distribution.scope():
      v = variables_lib.Variable(
          0., synchronization=synchronization, aggregation=aggregation)
      in_dist_copy = copy.deepcopy(v)

    out_dist_copy = copy.deepcopy(v)

    def assert_is_deep_copy(v1, v2):
      # [FUNCTION DOC]
      # HELPER: comparison helper for the deep copy test.
      # Takes an original and a copied variable and asserts matching type, aggregation and
      # strategy. For aggregating variables, asserts the wrapped variables are distinct objects
      # of the same type. Otherwise asserts the policy is distinct when present (identical when
      # absent), and that components match in count, device and value but are distinct objects.
      self.assertIsInstance(v2, type(v1))
      self.assertEqual(v1.aggregation, v2.aggregation)
      self.assertEqual(v1.distribute_strategy, v2.distribute_strategy)
      if isinstance(v1, ps_values.AggregatingVariable):
        self.assertIsInstance(v2.get(), type(v1.get()))
        self.assertNotEqual(id(v1.get()), id(v2.get()))
      else:
        if v1._policy:
          self.assertNotEqual(id(v1._policy), id(v2._policy))  # pylint: disable=protected-access
        else:
          self.assertEqual(id(v1._policy), id(v2._policy))  # pylint: disable=protected-access
        self.assertEqual(len(v1.values), len(v2.values))
        for (v1v, v2v) in zip(v1.values, v2.values):
          self.assertEqual(v1v.device, v2v.device)
          self.assertNotEqual(id(v1v), id(v2v))
          self.assertAllEqual(
              self.evaluate(v1.values), self.evaluate(v2.values))

    self.evaluate(variables_lib.global_variables_initializer())
    if not isinstance(distribution.extended, tpu_strategy.TPUExtended):
      distribution.run(assert_is_deep_copy, args=(v, in_dist_copy))
      distribution.run(assert_is_deep_copy, args=(v, out_dist_copy))

  def testAssignSignature(self, distribution, synchronization, aggregation):
    # [FUNCTION DOC]
    # Verifies assign, assign-add and assign-sub accept the same positional and keyword
    # arguments as ordinary variables. Calls them with an inner function, evaluated in
    # cross-replica context (skipped for ON_READ with SUM) and via a run call in replica
    # context (skipped for TPU in eager mode). Asserts nothing explicitly beyond no errors.
    # This test verifies assign*() can be called in the same way as normal
    # variables.
    with distribution.scope():
      v = variables_lib.Variable(
          0., synchronization=synchronization, aggregation=aggregation)

      def assign():
        # [FUNCTION DOC]
        # HELPER: exercises all assign variants on the enclosing variable.
        # Calls assign, assign-add and assign-sub with positional and keyword arguments and no
        # value read-back, then returns a constant tensor so graph mode has something to fetch.
        one = constant_op.constant(1.)
        v.assign(one, True, "assign", False)
        # TODO(b/154017756): SyncOnReadVariable.assign() doesn't support passing
        # value as a keyword argument.
        v.assign(one, use_locking=True, name="assign", read_value=False)
        v.assign_add(one, True, "assign", False)
        v.assign_add(one, use_locking=True, name="assign", read_value=False)
        v.assign_sub(one, True, "assign", False)
        v.assign_sub(one, use_locking=True, name="assign", read_value=False)
        # Return something for graph mode to fetch.
        return constant_op.constant(1)

      self.evaluate(variables_lib.global_variables_initializer())
      if not (synchronization == variables_lib.VariableSynchronization.ON_READ
              and aggregation == variables_lib.VariableAggregation.SUM):
        self.evaluate(distribution.experimental_local_results(assign()))
      if not (isinstance(distribution.extended, tpu_strategy.TPUExtended) and
              context.executing_eagerly()):
        self.evaluate(
            distribution.experimental_local_results(distribution.run(assign)))

  def testStrategyExtendedUpdate(self, distribution, synchronization,
                                 aggregation):
    # [FUNCTION DOC]
    # Skips unless the strategy has exactly two parameter devices, and for ON_WRITE variables
    # with an aggregation. Creates a variable and applies assign, assign-add, assign-sub and a
    # read-based assign through the strategy extended update call using a per-replica value of
    # two entries, asserting the per-device values after each step (1 and 2; 2 and 4; 1 and 2;
    # then 3 and 6).
    if len(distribution.extended.parameter_devices) != 2:
      self.skipTest("n/a: needs exactly two parameter devices")
    if (synchronization == variables_lib.VariableSynchronization.ON_WRITE and
        aggregation != variables_lib.VariableAggregation.NONE):
      self.skipTest("n/a: doesn't apply to ON_WRITE variable with aggregation")
    with distribution.scope():
      v = variables_lib.Variable(
          0., synchronization=synchronization, aggregation=aggregation)
    value = values_lib.PerReplica([1., 2.])

    assign_fn = lambda var, value: var.assign(value)
    self.evaluate(distribution.extended.update(v, assign_fn, args=(value,)))
    self.assertAllEqual(self.evaluate(v.values), [1., 2.])

    assign_add_fn = lambda var, value: var.assign_add(value)
    self.evaluate(distribution.extended.update(v, assign_add_fn, args=(value,)))
    self.assertAllEqual(self.evaluate(v.values), [2., 4.])

    assign_sub_fn = lambda var, value: var.assign_sub(value)
    self.evaluate(distribution.extended.update(v, assign_sub_fn, args=(value,)))
    self.assertAllEqual(self.evaluate(v.values), [1., 2.])

    read_assign_fn = lambda var, value: var.assign_add(var.value() + var.
                                                       read_value())
    self.evaluate(
        distribution.extended.update(v, read_assign_fn, args=(value,)))
    self.assertAllEqual(self.evaluate(v.values), [3., 6.])

  def testSaveNonDistributed(self, distribution, synchronization, aggregation):
    # [FUNCTION DOC]
    # Verifies that a distributed variable behaves like its primary component when a
    # non-distributed model is saved. Skipped for parameter server strategies. For many
    # properties, methods, scatter operations, conversion, control dependencies, operator
    # overloads (float and integer) and indexing, it traces a function under a save context
    # and asserts the graph has no device annotations and captures at most the primary
    # variable handle. Uses a float variable, a vector variable and (unless aggregation is
    # MEAN) an integer variable.
    # This test verifies that the DistributedVariable behave like the primary
    # variable when saving a non-distributed version of the model (the default).
    # The test asserts that the function traced under SaveContext has no device
    # annotations and only reference the primary component of the variable. Note
    # that please avoid capturing other eager tensors in this test to make the
    # assertion easy.

    if isinstance(distribution.extended,
                  parameter_server_strategy.ParameterServerStrategyExtended):
      self.skipTest("b/148689177: AggregatingVariable doesn't "
                    "conform to Variable interface well")

    # tf.function requires the return value to be Tensors, which is not always
    # case for properties and methods of Variable, so we simply discard the
    # return values.
    def _discard_return(f):
      # [FUNCTION DOC]
      # HELPER: wraps a callable so traced functions need not return tensors.
      # Calls the given callable with no arguments, ignores its result and returns nothing.
      f()
      return

    def _test(f, v):
      # [FUNCTION DOC]
      # HELPER: core check for the non-distributed save test.
      # Inputs: a callable to trace and the variable it uses. Traces the callable as a graph
      # function inside a save context with no variable policy, then asserts every operation
      # has an empty device and the function captures at most one tensor, which must be the
      # primary component handle of the variable. Returns nothing.
      # This verifies that the function under SaveContext:
      #   - contains no device annotations.
      #   - only references the primary component of the variable.
      g = def_function.function(lambda: _discard_return(f))
      options = save_options.SaveOptions(
          experimental_variable_policy=save_options.VariablePolicy.NONE)
      with save_context.save_context(options):
        # The graph should contain no device.
        graph = g.get_concrete_function().graph
      for op in graph.get_operations():
        self.assertEqual(op.device, "", msg=str(op))
      # The function should only capture the primary variable. Note that it
      # may not have captures, e.g. v.aggregation.
      captures = list(graph.captures)
      self.assertLessEqual(len(captures), 1)
      if graph.captures:
        self.assertIs(captures[0][0], v._primary.handle)

    def _assert(cond):
      # [FUNCTION DOC]
      # HELPER: builds an assert operation from a boolean condition tensor, printing the
      # condition on failure. Returns the created assert operation.
      return control_flow_assert.Assert(cond, [cond])

    with distribution.scope():
      # We use four variables for convenience reasons. They have no special
      # meaning.
      # - v is used whenever possible.
      # - w is used for scatter and gather, which require the variable to be
      # non-scalar.
      # - y is used when the dtype needs to be integer. Note that aggregation
      # cannot be MEAN for integers.
      v = variables_lib.Variable(
          0.,
          synchronization=synchronization,
          aggregation=aggregation,
          trainable=True)
      w = variables_lib.Variable([0., 0., 0.],
                                 synchronization=synchronization,
                                 aggregation=aggregation,
                                 trainable=True)
      if aggregation != variables_lib.VariableAggregation.MEAN:
        y = variables_lib.Variable(
            0, synchronization=synchronization, aggregation=aggregation)

    # pylint: disable=g-long-lambda

    # tf.Variable properties.
    _test(lambda: self.assertEqual(v.aggregation, aggregation), v)
    _test(lambda: self.assertIs(v.constraint, None), v)
    # TODO(crccw): should we raise an error instead?
    _test(lambda: self.assertEqual(v.device, v._primary.device), v)
    _test(lambda: self.assertEqual(v.dtype, dtypes.float32), v)
    if not context.executing_eagerly():
      _test(lambda: self.assertIs(v.graph, v._primary.graph), v)
    if not context.executing_eagerly():
      _test(lambda: _assert(v.initial_value == 0), v)
    _test(lambda: self.assertIs(v.initializer, v._primary.initializer), v)
    _test(lambda: self.assertEqual(v.name, "Variable:0"), v)
    if not context.executing_eagerly():
      _test(lambda: self.assertIs(v.op, v._primary.op), v)
    _test(lambda: self.assertEqual(v.shape, tensor_shape.TensorShape(())), v)
    _test(lambda: self.assertEqual(v.synchronization, synchronization), v)
    _test(lambda: self.assertEqual(v.trainable, True), v)

    # tf.Variable methods.
    _test(lambda: check_ops.assert_equal_v2(v.assign(1.), 1.), v)
    _test(lambda: check_ops.assert_equal_v2(v.assign_add(1.), 2.), v)
    _test(lambda: check_ops.assert_equal_v2(v.assign_sub(1.), 1.), v)
    # TODO(b/148689177): Implement batch_scatter_update.
    # count_up_to() is skipped since it's deprecated.
    # eval() is skipped since it shouldn't called in a tf.function.
    # experimental_ref() is skipped since it's deprecated.
    # from_proto() is skipped since it shouldn't called in a tf.function.
    # TODO(b/148689177): Implement gather_nd.
    _test(
        lambda: check_ops.assert_equal_v2(v.get_shape(),
                                          tensor_shape.TensorShape(())), v)
    # initialized_value() is skipped since it shouldn't called in a tf.function.
    # load() is skipped since it shouldn't called in a tf.function.
    _test(lambda: check_ops.assert_equal_v2(v.read_value(), 1.), v)
    # ref() is skipped since it shouldn't called in a tf.function.
    _test(
        lambda: check_ops.assert_equal_v2(
            w.scatter_add(_make_index_slices(values=[1., 2.], indices=[0, 2])),
            [1., 0., 2.]), w)
    _test(
        lambda: check_ops.assert_equal_v2(
            w.scatter_div(_make_index_slices(values=[4., 2.], indices=[0, 2])),
            [0.25, 0., 1.]), w)
    _test(
        lambda: check_ops.assert_equal_v2(
            w.scatter_max(_make_index_slices(values=[1., 0.5], indices=[1, 2])),
            [0.25, 1., 1.]), w)
    _test(
        lambda: check_ops.assert_equal_v2(
            w.scatter_min(_make_index_slices(values=[1., 0.5], indices=[0, 1])),
            [0.25, 0.5, 1.]), w)
    _test(
        lambda: check_ops.assert_equal_v2(
            w.scatter_mul(_make_index_slices(values=[2., 0.5], indices=[0, 1])),
            [0.5, 0.25, 1.]), w)
    # TODO(b/148689177): Implement scatter_nd_*
    _test(
        lambda: check_ops.assert_equal_v2(
            w.scatter_sub(_make_index_slices(values=[2., 0.5], indices=[0, 1])),
            [-1.5, -0.25, 1.]), w)
    _test(
        lambda: check_ops.assert_equal_v2(
            w.scatter_update(
                _make_index_slices(values=[2., 0.5], indices=[0, 1])),
            [2., 0.5, 1.]), w)
    # set_shape() is skipped since ResourceVariable doesn't implement it.
    # to_proto() is skipped since it shouldn't called in a tf.function.
    _test(lambda: check_ops.assert_equal_v2(v.value(), 1.), v)

    # DistributedVariable should be treated as ResourceVariable, so it needs to
    # conform to ResourceVariable interface as well.
    _test(lambda: self.assertIs(v.handle, v._primary.handle), v)

    # Convert to tensor.
    _test(lambda: check_ops.assert_equal_v2(ops.convert_to_tensor(v), 1.), v)

    # Control dependency.
    def _with_control_dep():
      # [FUNCTION DOC]
      # HELPER: uses the variable assign operation as a control dependency and returns an
      # identity of a constant, to check control dependency handling under the save context.
      with ops.control_dependencies([v.assign(1.)]):
        return array_ops.identity(1)

    _test(_with_control_dep, v)

    # Operator overloads.
    _test(lambda: check_ops.assert_equal_v2(v.assign(7.), 7.), v)
    _test(lambda: check_ops.assert_equal_v2(v + 1., 8.), v)
    _test(lambda: check_ops.assert_equal_v2(3 + v, 10.), v)
    _test(lambda: check_ops.assert_equal_v2(v + v, 14.), v)
    _test(lambda: check_ops.assert_equal_v2(v - 2., 5.), v)
    _test(lambda: check_ops.assert_equal_v2(v - v, 0.), v)
    _test(lambda: check_ops.assert_equal_v2(v * 2., 14.), v)
    _test(lambda: check_ops.assert_equal_v2(3 * v, 21.), v)
    _test(lambda: check_ops.assert_equal_v2(v * v, 49.), v)
    _test(
        lambda: check_ops.assert_equal_v2(
            math_ops.cast(v / 2., dtypes.float32), 3.5), v)
    _test(
        lambda: check_ops.assert_equal_v2(
            math_ops.cast(14. / v, dtypes.float32), 2.), v)
    _test(lambda: _assert(v < 12.), v)
    _test(lambda: _assert(v <= 12.), v)
    _test(lambda: _assert(not v > 12.), v)
    _test(lambda: _assert(not v >= 12.), v)
    _test(lambda: _assert(not 12. < v), v)
    _test(lambda: _assert(not 12. <= v), v)
    _test(lambda: _assert(12. > v), v)
    _test(lambda: _assert(12. >= v), v)
    _test(lambda: check_ops.assert_near_v2(pow(v, 3.), 343.), v)
    _test(lambda: check_ops.assert_near_v2(pow(2., v), 128.), v)
    _test(lambda: check_ops.assert_equal_v2(abs(v), 7.), v)

    # Operator overloads that only works for integers.
    if aggregation != variables_lib.VariableAggregation.MEAN:
      _test(lambda: check_ops.assert_equal_v2(y.assign(7), 7), y)
      _test(lambda: check_ops.assert_equal_v2(y // 2, 3), y)
      _test(lambda: check_ops.assert_equal_v2(15 // y, 2), y)
      _test(lambda: check_ops.assert_equal_v2(y % 2, 1), y)
      _test(lambda: check_ops.assert_equal_v2(16 % y, 2), y)
      _test(lambda: check_ops.assert_equal_v2(y & 3, 3), y)
      _test(lambda: check_ops.assert_equal_v2(3 & y, 3), y)
      _test(lambda: check_ops.assert_equal_v2(y | 8, 15), y)
      _test(lambda: check_ops.assert_equal_v2(16 | y, 23), y)
      _test(lambda: check_ops.assert_equal_v2(y ^ 3, 4), y)
      _test(lambda: check_ops.assert_equal_v2(11 ^ y, 12), y)
      _test(lambda: check_ops.assert_equal_v2(-y, -7), y)
      _test(lambda: check_ops.assert_equal_v2(~y, ~7), y)

    # Index.
    if isinstance(distribution.extended, tpu_strategy.TPUExtended):
      # TODO(b/161572567): slice assignment doesn't work for TPU.
      _test(lambda: check_ops.assert_equal_v2(w[0], 2.), w)
    else:
      _test(lambda: check_ops.assert_equal_v2(w[0].assign(1.), [1., 0.5, 1.]),
            w)
      _test(lambda: check_ops.assert_equal_v2(w[0], 1.), w)

    # pylint: enable=g-long-lambda

  def testUnsaveable(self, distribution, synchronization, aggregation, mode):
    # [FUNCTION DOC]
    # Verifies that functions which mutate a distributed variable (assign and scatter variants)
    # cannot be saved as signatures, raising a value error mentioning input signatures. Also
    # covers reads of ON_READ variables that must be unsaveable (SUM aggregation, no merge
    # call, or MEAN with collective all-reduce), while other reads must save successfully.
    # Skips parameter server strategies, multi-worker mirrored in graph mode, and strategies
    # without merge call. Writes saved models to a temp directory.
    if isinstance(distribution.extended,
                  parameter_server_strategy.ParameterServerStrategyExtended):
      self.skipTest("n/a: not applicable to AggregatingVariable")
    if (isinstance(distribution,
                   collective_all_reduce_strategy.CollectiveAllReduceStrategy)
        and mode == "graph"):
      self.skipTest("MWMS combinations tests do not work well in graph mode.")
    if not distribution.extended._use_merge_call():
      self.skipTest("Unsupported combination.")
    with distribution.scope():
      v = variables_lib.Variable([1., 1.],
                                 synchronization=synchronization,
                                 aggregation=aggregation)

    with self.cached_session():
      self.evaluate(variables_lib.global_variables_initializer())

    export_dir = self.get_temp_dir()

    def _assert_unsaveable(f):
      # [FUNCTION DOC]
      # HELPER: asserts that a callable cannot be saved as a signature.
      # Traces the callable into a concrete function; if tracing raises not-implemented or
      # value errors, returns silently. Otherwise asserts that saving the variable with that
      # signature raises a value error mentioning input signatures. Writes to the export dir.
      # Ignore if it cannot be traced. Certain combinations are not supported or
      # yet or not allowed.
      try:
        f = def_function.function(f).get_concrete_function()
      except (NotImplementedError, ValueError):
        return
      with self.assertRaisesRegex(ValueError, "f_with_input_signature"):
        save.save(v, export_dir, signatures=f)

    _assert_unsaveable(lambda: v.assign(ops.convert_to_tensor([1., 1.])))
    _assert_unsaveable(lambda: v.assign_add(ops.convert_to_tensor([1., 1.])))
    _assert_unsaveable(lambda: v.assign_sub(ops.convert_to_tensor([1., 1.])))
    _assert_unsaveable(lambda: v.scatter_add(_make_index_slices([1.], [0])))
    _assert_unsaveable(lambda: v.scatter_sub(_make_index_slices([1.], [0])))
    _assert_unsaveable(lambda: v.scatter_mul(_make_index_slices([1.], [0])))
    _assert_unsaveable(lambda: v.scatter_div(_make_index_slices([1.], [0])))
    _assert_unsaveable(lambda: v.scatter_min(_make_index_slices([1.], [0])))
    _assert_unsaveable(lambda: v.scatter_max(_make_index_slices([1.], [0])))
    _assert_unsaveable(lambda: v.scatter_update(_make_index_slices([1.], [0])))
    # Reading a ON_READ variable should be unsaveable if either:
    # 1) CollectiveAllReduceStrategy, and aggregation is MEAN/SUM.
    # 2) aggregation is SUM.
    if (synchronization == variables_lib.VariableSynchronization.ON_READ and
        (aggregation == variables_lib.VariableAggregation.SUM or
         (not distribution.extended._use_merge_call()) or
         (isinstance(distribution.extended,
                     collective_all_reduce_strategy.CollectiveAllReduceExtended)
          and aggregation == variables_lib.VariableAggregation.MEAN))):
      _assert_unsaveable(v.read_value)
      _assert_unsaveable(v.value)
      _assert_unsaveable(lambda: ops.convert_to_tensor(v))
    else:
      # Otherwise reading a variable should be saveable.

      @def_function.function
      def f():
        # [FUNCTION DOC]
        # Graph function that reads the variable value two ways and returns the variable
        # converted to a tensor; saved to show such reads are saveable.
        v.read_value()
        v.value()
        return ops.convert_to_tensor(v)

      with self.cached_session():
        save.save(v, export_dir, signatures=f.get_concrete_function())


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.mirrored_strategy_with_one_cpu,
            strategy_combinations.tpu_strategy,
        ],
        mode=["eager"]))
class PackedDistributedVariableTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Eager-mode tests for packed distributed variables, run on a one-CPU mirrored strategy and
  # a TPU strategy. Covers enabling packed variables and their behavior in save contexts.

  def testPackedVariable(self, distribution):
    # [FUNCTION DOC]
    # Asserts a variable created without enabling packing has no packed variable. Enables
    # packing on the strategy, creates a variable and asserts it holds a packed distributed
    # variable. Assigns each replica its index in a replica context, then asserts that getting
    # the value in each replica context yields a packed variable-and-device object on the
    # matching device whose read value equals the replica index (zero by default).
    with distribution.scope():
      v0 = variables_lib.Variable(0.)
    self.assertIsNone(v0._packed_var)

    distribution._enable_packed_variable_in_eager_mode = True
    with distribution.scope():
      v1 = variables_lib.Variable(0)
      self.assertIsInstance(v1._packed_var, packed.PackedDistributedVariable)

    devices = v1._devices
    for i in range(1, len(devices)):
      with distribute_lib.ReplicaContext(distribution, i):
        v1.assign(i)
    val = v1._get()
    self.assertIsInstance(val, packed.PackedVarAndDevice)
    self.assertEqual(val.device, devices[0])
    self.assertEqual(self.evaluate(val.read_value()), 0)
    for i in range(0, len(devices)):
      with distribute_lib.ReplicaContext(distribution, i):
        val = v1._get()
        self.assertIsInstance(val, packed.PackedVarAndDevice)
        self.assertEqual(val.device, devices[i])
        self.assertEqual(self.evaluate(val.read_value()), i)

  def testIgnorePackedVariableInSaveContext(self, distribution):
    # [FUNCTION DOC]
    # Enables packing on the strategy, creates a variable and asserts its packed variable
    # property is set, then within a save context asserts that property is nothing (ignored).
    distribution._enable_packed_variable_in_eager_mode = True
    with distribution.scope():
      v = variables_lib.Variable(0)
      self.assertIsInstance(v._packed_variable,
                            packed.PackedDistributedVariable)

    options = save_options.SaveOptions()
    with save_context.save_context(options):
      self.assertIsNone(v._packed_variable)


def _make_index_slices(values, indices, dense_shape=None):
  # [FUNCTION DOC]
  # HELPER: builds sparse indexed slices for the scatter tests.
  # Inputs: values, indices, and an optional dense shape (default nothing). Each is passed
  # through an identity op (the shape only when provided). Returns an indexed-slices object.
  if dense_shape:
    dense_shape = array_ops.identity(dense_shape)
  return indexed_slices.IndexedSlices(
      array_ops.identity(values), array_ops.identity(indices), dense_shape)


if __name__ == "__main__":
  ds_test_util.main()
