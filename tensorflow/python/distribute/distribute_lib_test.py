# [MODULE DOC]
# Unit tests for the core strategy machinery in the distribute library: strategy scopes, replica
# contexts, the default strategy, and input contexts. The module defines minimal fake strategies
# (a test strategy, its extended object, and a replica context) that implement just enough of the
# strategy extension interface to exercise the generic code paths, and then tests scope entering and
# exiting (including nesting errors), variable creator behaviour, set-strategy, reductions,
# updates, autograph behaviour inside replica calls, the default strategy, and input context
# properties. No import-time side effects other than defining classes and helpers.
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
"""Test DistributionStrategy, ReplicaContext, and supporting APIs."""

from absl.testing import parameterized

from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib
from tensorflow.python.util import nest


class _TestReplicaContext(distribute_lib.ReplicaContext):
  # [CLASS DOC]
  # HELPER: fake replica context for the tests in this module.
  # A replica context whose merge call ignores the merge function and simply returns the value
  # passed in the keyword argument named test_arg, so tests can verify the context is in use.

  def merge_call(self, fn, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: fake merge call for the test replica context.
    # Ignores the merge function and positional arguments and returns the value of the required
    # keyword argument test_arg. Raises a key-missing error if test_arg is not supplied.
    return kwargs["test_arg"]


def _get_test_variable(name, synchronization, aggregation):
  # [FUNCTION DOC]
  # HELPER: builds the fake "variable" produced by the test strategy.
  # Returns a mapping with the entries name, synchronization and aggregation set to the given
  # values (any types). Used to check which arguments reached variable creation.
  return {
      "name": name,
      "synchronization": synchronization,
      "aggregation": aggregation
  }


def _test_input_fn(input_context):
  # [FUNCTION DOC]
  # HELPER: input function used by the iterator tests.
  # Ignores the input context and returns a dataset that endlessly repeats the single value one.
  del input_context
  return dataset_ops.DatasetV2.from_tensors(1.).repeat()


class _TestStrategy(distribute_lib.Strategy):
  # [CLASS DOC]
  # HELPER: minimal fake strategy for the tests in this module.
  # A Strategy subclass wired to the fake extended object defined below; its variable creation
  # returns a plain mapping instead of a real variable.

  def __init__(self):
    # [FUNCTION DOC]
    # Constructs the strategy with a new fake extended object bound to this strategy. No
    # parameters.
    super(_TestStrategy, self).__init__(_TestExtended(self))


class _TestExtended(distribute_lib.StrategyExtendedV1):
  # [CLASS DOC]
  # HELPER: fake strategy extension implementing the minimum interface for the tests.
  # Runs everything on a single CPU device with a single replica, creates fake variables, runs
  # updates inline, and returns values unchanged for reductions and local results. Holds a
  # single-worker input workers object for the input iterator tests.

  def __init__(self, distribute):
    # [FUNCTION DOC]
    # Constructs the extension for the given owning strategy and creates input workers for one
    # worker with the single device CPU 0.
    super(_TestExtended, self).__init__(distribute)
    worker_device_pairs = [("", ["/device:CPU:0"])]
    self._input_workers = input_lib.InputWorkers(worker_device_pairs)

  def _call_for_each_replica(self, fn, args, kwargs):
    # [FUNCTION DOC]
    # Calls the function once with the given positional arguments (list) and keyword arguments
    # (mapping), inside the fake replica context with replica id zero. Returns the function's
    # result.
    with _TestReplicaContext(
        self._container_strategy(), replica_id_in_sync_group=0):
      return fn(*args, **kwargs)

  def _create_variable(self, next_creator, **kwargs):
    # [FUNCTION DOC]
    # Fake variable creation: ignores the next creator and returns a mapping with the requested
    # name, synchronization and aggregation (all must be present in the keyword arguments,
    # otherwise a key-missing error is raised).
    return _get_test_variable(kwargs["name"], kwargs["synchronization"],
                              kwargs["aggregation"])

  def _make_input_fn_iterator(
      self,
      input_fn,
      replication_mode=distribute_lib.InputReplicationMode.PER_WORKER):
    # [FUNCTION DOC]
    # Builds an input-function iterator for the given input function using the stored input
    # workers and a default input context. The replication mode (default per worker) is ignored.
    # Returns the iterator.
    return input_lib_v1.InputFunctionIterator(input_fn, self._input_workers,
                                              [distribute_lib.InputContext()],
                                              self._container_strategy())

  def _distribute_datasets_from_function(self, dataset_fn, options):
    # [FUNCTION DOC]
    # Calls the dataset function with a default input context and returns the dataset it makes
    # without any distribution. The options are ignored.
    return dataset_fn(distribute_lib.InputContext())

  def _local_results(self, value):
    # [FUNCTION DOC]
    # Returns the given value wrapped in a one-element tuple (single replica).
    return (value,)

  def _reduce_to(self, reduce_op, value, destinations, options):
    # [FUNCTION DOC]
    # Fake reduction: with a single replica the value is returned unchanged; the reduce op,
    # destinations and options are ignored.
    del reduce_op, destinations, options
    return value

  def _experimental_run_steps_on_iterator(self, fn, iterator, iterations,
                                          initial_loop_values=None):
    # [FUNCTION DOC]
    # Runs the function the given number of iterations, each time passing a new multi-step
    # context and the next element of the iterator. Initial loop values are ignored. Returns the
    # multi-step context. Incomplete (the context's run operation is never set).
    # TODO(tomhennigan) This is missing many things (e.g. ctx.run_op).
    ctx = input_lib.MultiStepContext()
    for _ in range(iterations):
      fn(ctx, iterator.get_next())
    return ctx

  def _update(self, var, fn, args, kwargs, group):
    # [FUNCTION DOC]
    # Applies the function to the variable followed by the given arguments by delegating to the
    # non-slot update with the variable as the colocation target. The group flag controls whether
    # the result is returned as is or unwrapped. Returns that result.
    # The implementations of _update() and _update_non_slot() are identical
    # except _update() passes `var` as the first argument to `fn()`.
    return self._update_non_slot(var, fn, (var,) + tuple(args), kwargs, group)

  def _update_non_slot(self, colocate_with, fn, args, kwargs, group):
    # [FUNCTION DOC]
    # Calls the function with the given positional and keyword arguments, ignoring the
    # colocation target. If the group flag is true the result is returned unchanged; otherwise
    # every value in the (possibly nested) result is unwrapped to its per-replica components.
    del colocate_with
    result = fn(*args, **kwargs)
    if group:
      return result
    else:
      return nest.map_structure(self._unwrap, result)

  def _get_local_replica_id(self, replica_id_in_sync_group):
    # [FUNCTION DOC]
    # Maps a replica id within the sync group to a local replica id; with one replica they are
    # identical, so the input is returned.
    return replica_id_in_sync_group


def _assert_in_default_state(t):
  # [FUNCTION DOC]
  # HELPER: shared assertion used by most tests in this module.
  # Given a test case, asserts that no strategy is active: the current replica context is the
  # default one, there is no cross-replica context, the current strategy is the default strategy
  # and has-strategy is false.
  t.assertIs(distribute_lib._get_default_replica_context(),
             distribute_lib.get_replica_context())
  t.assertIs(None, distribute_lib.get_cross_replica_context())
  t.assertFalse(distribute_lib.in_cross_replica_context())
  t.assertIs(
      distribute_lib._get_default_strategy(), distribute_lib.get_strategy())
  t.assertFalse(distribute_lib.has_strategy())


def _run_in_and_out_of_scope(unbound_test_method):
  # [FUNCTION DOC]
  # HELPER: decorator for test methods that take a strategy argument.
  # Takes a test method (test case plus strategy) and returns a wrapper taking only the test
  # case. The wrapper creates a new test strategy and runs the method in the default state, then
  # inside that strategy's scope, then checks the default state is restored, and finally asserts
  # that running the method under a different strategy's scope raises a runtime error about
  # mixing strategy objects.
  def wrapper(test_case):
    # [FUNCTION DOC]
    # HELPER: the wrapped test body; see the enclosing decorator. Takes the test case, returns
    # nothing, and fails the test if any of the scenarios behave unexpectedly.
    dist = _TestStrategy()
    # Running in the default (replica) scope should be supported.
    _assert_in_default_state(test_case)
    unbound_test_method(test_case, dist)
    # As well as running in the strategy scope.
    with dist.scope():
      unbound_test_method(test_case, dist)
    _assert_in_default_state(test_case)
    # When run under a different strategy the test method should fail.
    another_strategy = _TestStrategy()
    msg = "Mixing different .*Strategy objects"
    with test_case.assertRaisesRegex(RuntimeError, msg):
      with another_strategy.scope():
        unbound_test_method(test_case, dist)
  return wrapper


class TestStrategyTest(test.TestCase):
  # [CLASS DOC]
  # Tests the generic strategy behaviour (scopes, nesting errors, variable creation arguments,
  # set-strategy, reductions, updates, autograph) using the fake test strategy defined above.

  def testCallForEachReplica(self):
    # [FUNCTION DOC]
    # Scenario: call a function per replica both outside and inside the strategy scope. Inside,
    # asserts the replica context exists, there is no cross-replica context, a strategy is
    # active and is the test strategy, the fake merge call returns the supplied value, and a
    # created variable carries default (auto) synchronization and no aggregation. Afterwards
    # asserts the default state is restored.
    _assert_in_default_state(self)
    dist = _TestStrategy()

    def run_fn():
      # [FUNCTION DOC]
      # Per-replica body: performs the assertions described in the enclosing test.
      replica_context = distribute_lib.get_replica_context()
      self.assertIsNotNone(replica_context)
      self.assertIs(None, distribute_lib.get_cross_replica_context())
      self.assertFalse(distribute_lib.in_cross_replica_context())
      self.assertTrue(distribute_lib.has_strategy())
      self.assertIs(dist, distribute_lib.get_strategy())
      self.assertEqual("foo", replica_context.merge_call(None, test_arg="foo"))
      expected_value = _get_test_variable(
          "bar", variable_scope.VariableSynchronization.AUTO,
          variable_scope.VariableAggregation.NONE)
      self.assertDictEqual(expected_value,
                           variable_v1.VariableV1(1.0, name="bar"))

    dist.extended.call_for_each_replica(run_fn)
    with dist.scope():
      dist.extended.call_for_each_replica(run_fn)
    _assert_in_default_state(self)

  def testScope(self):
    # [FUNCTION DOC]
    # Scenario: enter the strategy scope. Asserts there is no replica context, the cross-replica
    # context is the strategy, a strategy is active, and a created variable has default
    # synchronization and aggregation; after exit the default state is restored.
    _assert_in_default_state(self)
    dist = _TestStrategy()
    with dist.scope():
      self.assertIs(None, distribute_lib.get_replica_context())
      self.assertIs(dist, distribute_lib.get_cross_replica_context())
      self.assertTrue(distribute_lib.in_cross_replica_context())
      self.assertTrue(distribute_lib.has_strategy())
      self.assertIs(dist, distribute_lib.get_strategy())
      expected_value = _get_test_variable(
          "baz", variable_scope.VariableSynchronization.AUTO,
          variable_scope.VariableAggregation.NONE)
      self.assertDictEqual(expected_value,
                           variable_v1.VariableV1(1.0, name="baz"))
    _assert_in_default_state(self)

  def testScopeDeviceNestingError(self):
    # [FUNCTION DOC]
    # Scenario: the strategy has a default GPU device; enter its scope manually, then open a CPU
    # device scope and try to exit the strategy scope. Expects a runtime error about device scope
    # nesting; then exits properly and asserts the default state is restored.
    _assert_in_default_state(self)
    dist = _TestStrategy()
    # Open a device scope with dist.scope().
    dist.extended._default_device = "/device:GPU:0"
    scope = dist.scope()
    scope.__enter__()
    self.assertIs(dist, distribute_lib.get_strategy())
    with ops.device("/device:CPU:0"):
      with self.assertRaisesRegex(RuntimeError, "Device scope nesting error"):
        scope.__exit__(None, None, None)
    scope.__exit__(None, None, None)
    _assert_in_default_state(self)

  def testScopeVarCreatorNestingError(self):
    # [FUNCTION DOC]
    # Scenario: enter the strategy scope manually, then open a variable creator scope and try to
    # exit the strategy scope. Expects a runtime error about variable creator scope nesting; then
    # exits properly and asserts the default state is restored.

    def creator(next_creator, **kwargs):
      # [FUNCTION DOC]
      # HELPER: pass-through variable creator; forwards all keyword arguments to the next
      # creator and returns its result.
      return next_creator(**kwargs)

    _assert_in_default_state(self)
    dist = _TestStrategy()
    scope = dist.scope()
    scope.__enter__()
    self.assertIs(dist, distribute_lib.get_strategy())
    with variable_scope.variable_creator_scope(creator):
      with self.assertRaisesRegex(RuntimeError,
                                  "Variable creator scope nesting error"):
        scope.__exit__(None, None, None)
    scope.__exit__(None, None, None)
    _assert_in_default_state(self)

  def testScopeVarScopeNestingError(self):
    # [FUNCTION DOC]
    # Scenario (in a fresh graph): enter the strategy scope manually, open a variable scope and
    # try to exit the strategy scope. Expects a runtime error about variable scope nesting, then
    # asserts the default state is restored.
    # We create a new graph here to simplify clean-up, since the error
    # we are triggering happens in the middle of scope.__exit__() and
    # leaves us in a weird state.
    with ops.Graph().as_default():
      _assert_in_default_state(self)
      dist = _TestStrategy()
      scope = dist.scope()
      scope.__enter__()
      self.assertIs(dist, distribute_lib.get_strategy())
      with variable_scope.variable_scope("AA"):
        with self.assertRaisesRegex(RuntimeError,
                                    "Variable scope nesting error"):
          scope.__exit__(None, None, None)
    _assert_in_default_state(self)

  def testSettingSynchronizationAndAggregation(self):
    # [FUNCTION DOC]
    # Scenario: inside the strategy scope create a variable with on-write synchronization and
    # mean aggregation. Asserts these settings reach the strategy's variable creation, and the
    # default state is restored afterwards.
    _assert_in_default_state(self)
    dist = _TestStrategy()
    with dist.scope():
      expected_value = _get_test_variable(
          "baz", variable_scope.VariableSynchronization.ON_WRITE,
          variable_scope.VariableAggregation.MEAN)
      self.assertDictEqual(
          expected_value,
          variable_v1.VariableV1(
              1.0,
              name="baz",
              synchronization=variable_scope.VariableSynchronization.ON_WRITE,
              aggregation=variable_scope.VariableAggregation.MEAN))
    _assert_in_default_state(self)

  def testSetStrategy(self):
    # [FUNCTION DOC]
    # Scenario: globally set a strategy (outside any scope). Asserts it acts as the current
    # cross-replica strategy and creates variables with default settings; setting a second
    # strategy replaces it; setting nothing restores the default state.
    _assert_in_default_state(self)
    dist = _TestStrategy()
    dist2 = _TestStrategy()
    distribute_lib.experimental_set_strategy(dist)
    self.assertIs(None, distribute_lib.get_replica_context())
    self.assertIs(dist, distribute_lib.get_cross_replica_context())
    self.assertTrue(distribute_lib.in_cross_replica_context())
    self.assertTrue(distribute_lib.has_strategy())
    self.assertIs(dist, distribute_lib.get_strategy())
    expected_value = _get_test_variable(
        "baz", variable_scope.VariableSynchronization.AUTO,
        variable_scope.VariableAggregation.NONE)
    self.assertDictEqual(expected_value,
                         variable_v1.VariableV1(1.0, name="baz"))
    distribute_lib.experimental_set_strategy(dist2)
    self.assertIs(dist2, distribute_lib.get_strategy())
    distribute_lib.experimental_set_strategy(None)
    _assert_in_default_state(self)

  def testSetStrategyInScope(self):
    # [FUNCTION DOC]
    # Scenario: inside a strategy scope, try to globally set a new strategy, the same strategy,
    # and nothing. Each must raise a runtime error saying it must not be called inside a strategy
    # scope. Afterwards asserts the default state.
    _assert_in_default_state(self)
    dist = _TestStrategy()
    with dist.scope():
      with self.assertRaisesRegex(
          RuntimeError,
          "Must not be called inside a `tf.distribute.Strategy` scope"):
        distribute_lib.experimental_set_strategy(_TestStrategy())
      with self.assertRaisesRegex(
          RuntimeError,
          "Must not be called inside a `tf.distribute.Strategy` scope"):
        distribute_lib.experimental_set_strategy(dist)
      with self.assertRaisesRegex(
          RuntimeError,
          "Must not be called inside a `tf.distribute.Strategy` scope"):
        distribute_lib.experimental_set_strategy(None)
    _assert_in_default_state(self)

  def testSameScopeNesting(self):
    # [FUNCTION DOC]
    # Scenario: nest scopes of the same strategy (re-entering the same scope object and a second
    # scope object); the strategy stays current throughout. Entering a scope of a different
    # strategy inside raises a runtime error about mixing strategies. After leaving all scopes
    # the default state holds, and a previously created scope can be reused.
    _assert_in_default_state(self)
    dist = _TestStrategy()
    scope_a = dist.scope()
    with scope_a:
      self.assertIs(dist, distribute_lib.get_strategy())
      scope_b = dist.scope()
      with scope_b:
        self.assertIs(dist, distribute_lib.get_strategy())
        with scope_a:
          self.assertIs(dist, distribute_lib.get_strategy())
        self.assertIs(dist, distribute_lib.get_strategy())
      self.assertIs(dist, distribute_lib.get_strategy())
      dist2 = _TestStrategy()
      scope2 = dist2.scope()
      with self.assertRaisesRegex(
          RuntimeError, "Mixing different tf.distribute.Strategy objects"):
        with scope2:
          pass
    _assert_in_default_state(self)
    with scope_b:
      self.assertIs(dist, distribute_lib.get_strategy())
    _assert_in_default_state(self)

  @_run_in_and_out_of_scope
  def testMakeInputFnIterator(self, dist):
    # [FUNCTION DOC]
    # Scenario: make an input-function iterator from the test input function; asserts a
    # non-empty result. Run in and out of scope via the decorator.
    self.assertIsNotNone(dist.make_input_fn_iterator(_test_input_fn))

  @_run_in_and_out_of_scope
  def testReduce(self, dist):
    # [FUNCTION DOC]
    # Scenario: mean-reduce a scalar constant with no axis; the result must equal the input
    # (single replica).
    x = constant_op.constant(1.)
    x_r = dist.reduce(reduce_util.ReduceOp.MEAN, x, axis=None)
    self.assertEqual(self.evaluate(x), self.evaluate(x_r))

  def testReductions_acceptStringOps(self):
    # [FUNCTION DOC]
    # Scenario: reduce, reduce-to and batch-reduce-to are called with string reduce ops in lower
    # and upper case (mean, sum). Asserts each accepts them and returns values equal to the
    # inputs (single replica).
    dist = _TestStrategy()
    for op in ("mean", "MEAN", "sum", "SUM"):
      x = constant_op.constant(1.)
      y = constant_op.constant(1.)
      x_r = dist.reduce(op, x, axis=None)
      self.assertEqual(self.evaluate(x), self.evaluate(x_r))
      x_r = dist.extended.reduce_to(op, x, "/CPU:0")
      self.assertEqual(self.evaluate(x), self.evaluate(x_r))
      x_r, y_r = dist.extended.batch_reduce_to(op,
                                               ((x, "/CPU:0"), (y, "/CPU:0")))
      self.assertEqual(self.evaluate(x), self.evaluate(x_r))
      self.assertEqual(self.evaluate(y), self.evaluate(y_r))

  @_run_in_and_out_of_scope
  def testReduceMeanAxis(self, dist):
    # [FUNCTION DOC]
    # Scenario: mean-reduce a two by two matrix with no axis (unchanged), axis zero (column
    # means), and both axes (overall mean 2.5). Asserts the expected values.
    x = constant_op.constant([[1., 2.], [3., 4.]])
    x_r = dist.reduce(reduce_util.ReduceOp.MEAN, x, axis=None)
    self.assertAllEqual(self.evaluate(x), self.evaluate(x_r))
    x_r = dist.reduce(reduce_util.ReduceOp.MEAN, x, axis=0)
    self.assertAllEqual([2., 3.], self.evaluate(x_r))
    x_r = dist.reduce(reduce_util.ReduceOp.MEAN, x, axis=(0, 1))
    self.assertEqual(2.5, self.evaluate(x_r))

  @_run_in_and_out_of_scope
  def testReduceSumAxis(self, dist):
    # [FUNCTION DOC]
    # Scenario: sum-reduce a two by two matrix with no axis (unchanged), axis zero (column
    # sums), and both axes (total 10). Asserts the expected values.
    x = constant_op.constant([[1., 2.], [3., 4.]])
    x_r = dist.reduce(reduce_util.ReduceOp.SUM, x, axis=None)
    self.assertAllEqual(self.evaluate(x), self.evaluate(x_r))
    x_r = dist.reduce(reduce_util.ReduceOp.SUM, x, axis=0)
    self.assertAllEqual([4., 6.], self.evaluate(x_r))
    x_r = dist.reduce(reduce_util.ReduceOp.SUM, x, axis=(0, 1))
    self.assertEqual(10., self.evaluate(x_r))

  @_run_in_and_out_of_scope
  def testExperimentalRunStepsOnIterator(self, dist):
    # [FUNCTION DOC]
    # Scenario: run one step of a function over a one-shot iterator of an endless dataset of the
    # value one, collecting the evaluated inputs. Asserts exactly one input equal to one was seen.
    all_inputs = []
    dataset = dataset_ops.Dataset.from_tensors(1.).repeat()
    dist.extended.experimental_run_steps_on_iterator(
        lambda _, inputs: all_inputs.append(self.evaluate(inputs)),
        dataset_ops.make_one_shot_iterator(dataset))
    self.assertEqual(all_inputs, [1.])

  @_run_in_and_out_of_scope
  def testReduceTo(self, dist):
    # [FUNCTION DOC]
    # Scenario: mean reduce-to the CPU for a scalar; the result equals the input.
    x = constant_op.constant(1.)
    x_r = dist.extended.reduce_to(reduce_util.ReduceOp.MEAN, x, "/CPU:0")
    self.assertEqual(self.evaluate(x), self.evaluate(x_r))

  @_run_in_and_out_of_scope
  def testBatchReduceTo(self, dist):
    # [FUNCTION DOC]
    # Scenario: mean batch-reduce-to the CPU for two scalar values; each result equals its input.
    x = constant_op.constant(1.)
    y = constant_op.constant(1.)
    x_r, y_r = dist.extended.batch_reduce_to(reduce_util.ReduceOp.MEAN,
                                             ((x, "/CPU:0"), (y, "/CPU:0")))
    self.assertEqual(self.evaluate(x), self.evaluate(x_r))
    self.assertEqual(self.evaluate(y), self.evaluate(y_r))

  @_run_in_and_out_of_scope
  def testUpdate(self, dist):
    # [FUNCTION DOC]
    # Scenario: create a variable in scope and update it with a function and one extra argument.
    # The function asserts it receives the variable itself and the extra argument unchanged.
    with dist.scope():
      v = variables.Variable(1.)
    t = constant_op.constant(2.)

    def assign_fn(vv, tt):
      # [FUNCTION DOC]
      # Update function: asserts the first argument is the variable and the second is the
      # constant passed to the update. Returns nothing.
      self.assertIs(vv, v)
      self.assertIs(tt, t)
    dist.extended.update(v, assign_fn, (t,))

  @_run_in_and_out_of_scope
  def testUpdateAutoGraph(self, dist):
    # [FUNCTION DOC]
    # Scenario: inside a compiled function, update a variable through the strategy. The update
    # function asserts it is being run as autograph-generated code.
    with dist.scope():
      v = variables.Variable(1.)
    t = constant_op.constant(2.)

    def assign_fn(unused_vv, unused_tt):
      # [FUNCTION DOC]
      # Update function: asserts it executes inside autograph-generated code. Ignores its
      # arguments and returns nothing.
      self.assertTrue(converter_testing.is_inside_generated_code())

    @def_function.function  # AutoGraph is default-on only within tf.function
    def test_fn():
      # [FUNCTION DOC]
      # Compiled function that triggers the strategy update with the update function.
      dist.extended.update(v, assign_fn, (t,))

    test_fn()

  @_run_in_and_out_of_scope
  def testUpdateNonSlot(self, dist):
    # [FUNCTION DOC]
    # Scenario: run a non-slot update with a function that records a call; asserts the function
    # was called exactly once.
    t = constant_op.constant(2.)
    update_calls = []
    dist.extended.update_non_slot(t, lambda: update_calls.append(1))
    self.assertEqual(len(update_calls), 1)

  @_run_in_and_out_of_scope
  def testUpdateNonSlotAutoGraph(self, dist):
    # [FUNCTION DOC]
    # Scenario: inside a compiled function, run a non-slot update; the update function asserts
    # it is running as autograph-generated code.
    t = constant_op.constant(2.)

    def update_fn():
      # [FUNCTION DOC]
      # Update function: asserts it executes inside autograph-generated code; returns nothing.
      self.assertTrue(converter_testing.is_inside_generated_code())

    @def_function.function  # AutoGraph is default-on only within tf.function
    def test_fn():
      # [FUNCTION DOC]
      # Compiled function that triggers the non-slot update.
      dist.extended.update_non_slot(t, update_fn)

    test_fn()

  def testClusterResolverDefaultNotImplemented(self):
    # [FUNCTION DOC]
    # Scenario: a strategy that does not set a cluster resolver reports none; after assigning a
    # simple resolver on the extension, the strategy's cluster resolver is that object.
    dist = _TestStrategy()
    self.assertIsNone(dist.cluster_resolver)
    base_cluster_spec = server_lib.ClusterSpec({
        "ps": ["ps0:2222", "ps1:2222"],
        "worker": ["worker0:2222", "worker1:2222", "worker2:2222"]
    })
    cluster_resolver = cluster_resolver_lib.SimpleClusterResolver(
        base_cluster_spec)
    dist.extended._cluster_resolver = cluster_resolver
    self.assertIs(dist.cluster_resolver, cluster_resolver)


# _TestStrategy2 is like _TestStrategy, except it doesn't change variable
# creation.
class _TestStrategy2(distribute_lib.Strategy):
  # [CLASS DOC]
  # HELPER: fake strategy like the first test strategy but whose extension does not override
  # variable creation (variables are created by the normal creator chain).

  def __init__(self):
    # [FUNCTION DOC]
    # Constructs the strategy with a new second-flavour extended object bound to it.
    super(_TestStrategy2, self).__init__(_TestExtended2(self))


class _TestExtended2(_TestExtended):
  # [CLASS DOC]
  # HELPER: extension identical to the first fake extension except that variable creation is
  # passed through to the next creator.

  def _create_variable(self, next_creator, **kwargs):
    # [FUNCTION DOC]
    # Forwards all keyword arguments to the next creator and returns the created variable.
    return next_creator(**kwargs)


class DefaultDistributionStrategyTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests the default strategy and default replica context that are active when no strategy
  # scope has been entered: merge calls, scope as a mostly no-op, running steps, distributing
  # datasets, and API-version-specific strategy types.

  def testMergeCall(self):
    # [FUNCTION DOC]
    # Scenario: from the default replica context call merge-call with a function and an argument.
    # The function asserts it runs in the cross-replica context of the default strategy with no
    # replica context and returns a prefixed string; the test asserts that string is returned.
    _assert_in_default_state(self)

    def merge_fn(dist, s):
      # [FUNCTION DOC]
      # Merge function: asserts the strategy argument is the default strategy and the cross-
      # replica state is active, then returns the given string prefixed with a fixed word.
      self.assertIs(distribute_lib._get_default_strategy(), dist)
      self.assertIs(None, distribute_lib.get_replica_context())
      self.assertIs(dist, distribute_lib.get_cross_replica_context())
      self.assertTrue(distribute_lib.in_cross_replica_context())
      self.assertIs(dist, distribute_lib.get_strategy())
      self.assertFalse(distribute_lib.has_strategy())
      return "foo_" + s

    replica_ctx = distribute_lib.get_replica_context()
    self.assertIs(distribute_lib._get_default_replica_context(), replica_ctx)
    self.assertEqual("foo_bar", replica_ctx.merge_call(merge_fn, args=("bar",)))
    _assert_in_default_state(self)

  def testMergeCallAutoGraph(self):
    # [FUNCTION DOC]
    # Scenario: inside a compiled function, call merge-call on the default replica context; the
    # merge function asserts it is running as autograph-generated code.
    _assert_in_default_state(self)

    def merge_fn(_, s):
      # [FUNCTION DOC]
      # Merge function: asserts it executes in autograph-generated code and returns its second
      # argument unchanged.
      self.assertTrue(converter_testing.is_inside_generated_code())
      return s

    @def_function.function  # AutoGraph is default-on only within tf.function
    def test_fn():
      # [FUNCTION DOC]
      # Compiled function that fetches the replica context and calls merge-call.
      replica_ctx = distribute_lib.get_replica_context()
      replica_ctx.merge_call(merge_fn, args=("bar",))

    test_fn()

  def testScopeMostlyNoOp(self):
    # [FUNCTION DOC]
    # Scenario: enter the default strategy's scope (also re-entered nested); the state stays the
    # default state. Entering another strategy's scope inside it and creating a variable raises
    # a runtime error about mixing strategies, while creating variables in that other strategy's
    # scope before and after is fine.
    _assert_in_default_state(self)

    test_strategy = _TestStrategy2()
    with test_strategy.scope():
      variable_v1.VariableV1(1.0, name="before")

    default_strategy = distribute_lib._get_default_strategy()
    scope = default_strategy.scope()
    with scope:
      _assert_in_default_state(self)

      with test_strategy.scope():
        with self.assertRaisesRegex(
            RuntimeError, "Mixing different tf.distribute.Strategy objects"):
          variable_v1.VariableV1(1.0, name="error")

      with scope:
        _assert_in_default_state(self)

        with test_strategy.scope():
          with self.assertRaisesRegex(
              RuntimeError, "Mixing different tf.distribute.Strategy objects"):
            variable_v1.VariableV1(1.0, name="also_error")

      _assert_in_default_state(self)

    _assert_in_default_state(self)
    with test_strategy.scope():
      variable_v1.VariableV1(1.0, name="after")

  def testExperimentalRunV2(self):
    # [FUNCTION DOC]
    # Scenario: run a step function twice on the default strategy with the next element of a
    # dataset iterator as argument. Only checks that it does not raise.
    default_strategy = distribute_lib._get_default_strategy()
    dataset = dataset_ops.Dataset.range(10).batch(2)
    iterator = default_strategy.extended._make_dataset_iterator(dataset)
    next_val = iterator.get_next()

    def train_step(input_data):
      # [FUNCTION DOC]
      # Step function: returns its input unchanged.
      return input_data

    for _ in range(2):
      default_strategy.run(train_step, args=(next_val,))

  @combinations.generate(combinations.combine(mode=["graph", "eager"]))
  def testDistributedDatasets(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager modes): distribute a range dataset batched by two with the default
    # strategy and read the first element (through iteration in eager mode, an initializable
    # iterator in graph mode). Asserts the first batch contains zero and one.
    default_strategy = distribute_lib._get_default_strategy()
    if context.executing_eagerly():
      dataset_fn = lambda _: dataset_ops.DatasetV2.range(10).batch(2)
      dist_dataset = default_strategy.experimental_distribute_dataset(
          dataset_fn(distribute_lib.InputContext()))
      next_val = next(iter(dist_dataset))
    else:
      dataset_fn = lambda _: dataset_ops.DatasetV1.range(10).batch(2)
      dist_dataset = default_strategy.experimental_distribute_dataset(
          dataset_fn(distribute_lib.InputContext()))
      iterator = dist_dataset.make_initializable_iterator()
      self.evaluate(iterator.initializer)
      next_val = iterator.get_next()
    self.assertAllEqual([0, 1], self.evaluate(next_val))

  @combinations.generate(combinations.combine(mode=["graph", "eager"]))
  def testDistributedDatasetsFromFunction(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager modes): distribute datasets from a dataset function with the
    # default strategy. In eager mode asserts the first batch is zero and one; in graph mode only
    # builds an initializable iterator over the result.
    default_strategy = distribute_lib._get_default_strategy()
    if context.executing_eagerly():
      dataset_fn = lambda _: dataset_ops.DatasetV2.range(10).batch(2)
      dist_dataset_from_func = \
          default_strategy.distribute_datasets_from_function(
              dataset_fn)
      next_val = next(iter(dist_dataset_from_func))
      self.assertAllEqual([0, 1], self.evaluate(next_val))
    else:
      dataset_fn = lambda _: dataset_ops.DatasetV2.range(10).batch(2)
      dist_dataset_from_func = \
        default_strategy.distribute_datasets_from_function(
            dataset_fn)
      dataset_ops.make_initializable_iterator(dist_dataset_from_func)

  @combinations.generate(combinations.combine(tf_api_version=1))
  def testV1(self):
    # [FUNCTION DOC]
    # Scenario (API version 1): the current default strategy is an instance of the version-1
    # strategy class.
    self.assertIsInstance(
        distribute_lib.get_strategy(), distribute_lib.StrategyV1)

  @combinations.generate(combinations.combine(tf_api_version=2))
  def testV2(self):
    # [FUNCTION DOC]
    # Scenario (API version 2): the current default strategy is an instance of the version-2
    # strategy class.
    self.assertIsInstance(
        distribute_lib.get_strategy(), distribute_lib.Strategy)


class InputContextTest(test.TestCase):
  # [CLASS DOC]
  # Tests the input context object: its properties, per-replica batch size computation, and
  # string form.

  def testProperties(self):
    # [FUNCTION DOC]
    # Scenario: build an input context with two pipelines, id one and six replicas; asserts the
    # three properties return those values.
    input_context = distribute_lib.InputContext(
        num_input_pipelines=2, input_pipeline_id=1, num_replicas_in_sync=6)
    self.assertEqual(6, input_context.num_replicas_in_sync)
    self.assertEqual(1, input_context.input_pipeline_id)
    self.assertEqual(2, input_context.num_input_pipelines)

  def testPerReplicaBatchSize(self):
    # [FUNCTION DOC]
    # Scenario: with six replicas, a global batch of twelve gives a per-replica batch of two; a
    # global batch of thirteen (not divisible) raises a value error.
    input_context = distribute_lib.InputContext(
        num_input_pipelines=2, input_pipeline_id=1, num_replicas_in_sync=6)
    self.assertEqual(2, input_context.get_per_replica_batch_size(12))
    with self.assertRaises(ValueError):
      input_context.get_per_replica_batch_size(13)

  def testStr(self):
    # [FUNCTION DOC]
    # Scenario: the string form of an input context reports the pipeline id and the total number
    # of pipelines; checked for two different contexts.
    input_context = distribute_lib.InputContext(
        num_input_pipelines=1, input_pipeline_id=0, num_replicas_in_sync=42)
    self.assertEqual(
        "tf.distribute.InputContext(input pipeline id 0, total: 1)",
        str(input_context))
    input_context = distribute_lib.InputContext(
        num_input_pipelines=3, input_pipeline_id=1, num_replicas_in_sync=42)
    self.assertEqual(
        "tf.distribute.InputContext(input pipeline id 1, total: 3)",
        str(input_context))


if __name__ == "__main__":
  test.main()
