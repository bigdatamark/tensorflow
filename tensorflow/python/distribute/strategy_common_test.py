# [MODULE DOC]
# Tests for behavior common to all distribution strategy classes, run across the strategy
# combinations in eager mode (and graph mode for some). Covers capturing the replica id in traced
# functions, merge calls inside an init scope, conversion of per-replica results to local results
# for several nested structure shapes, reductions (including axis handling), replica-context
# variable updates, replica-context all-reduce and the public all-reduce API for dense, sparse
# (indexed slices), tuple and nested inputs, multi-worker host and device tensor reductions,
# input pipelines with partial last batches, and the cluster resolver property of strategies.
# Contains two small module-level helpers used by several tests. No import-time side effects
# apart from class-level decoration by the combinations generator.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for common methods in strategy classes."""

from absl.testing import parameterized

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute.collective_all_reduce_strategy import CollectiveAllReduceStrategy
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.util import nest


@combinations.generate(
    combinations.combine(
        strategy=[
            strategy_combinations.multi_worker_mirrored_2x1_cpu,
            strategy_combinations.multi_worker_mirrored_2x1_gpu,
        ] + strategy_combinations.all_strategies,
        mode=['eager']))
class StrategyTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Generic strategy tests, run in eager mode over two multi-worker mirrored descriptors and all
  # strategies in the shared all-strategies list. Each test method receives a strategy.

  def testCaptureReplicaId(self, strategy):
    # [FUNCTION DOC]
    # Checks that a compiled function reading the replica id from the replica context can be
    # captured and run through the strategy from a stateful outer function that gets traced twice.
    # Only asserts that running completes without error; nothing is compared.
    m = {}

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # HELPER: compiled function returning the current replica's id within the sync group.
      return distribute_lib.get_replica_context().replica_id_in_sync_group

    @def_function.function
    def g():
      # [FUNCTION DOC]
      # HELPER: outer compiled function that creates a variable on its first trace (making it
      # stateful so it is traced twice) and runs the replica-id function on the strategy.
      # Make g() a stateful function so it's traced twice.
      if m.get('v', None) is None:
        m['v'] = variables.Variable(0.)
      return strategy.run(f)

    g()

  def testMergeCallInitScope(self, strategy):
    # [FUNCTION DOC]
    # Within the strategy scope, runs on each replica a function that enters the init scope and
    # performs a merge call whose merge function returns eleven, then adds one. Asserts the local
    # results equal twelve for every replica on this client.
    with strategy.scope():

      @def_function.function
      def fn():
        # [FUNCTION DOC]
        # HELPER: compiled function that runs the replica function on the strategy and returns
        # the per-replica result.

        def merge_fn(unused_strat):
          # [FUNCTION DOC]
          # HELPER: merge function that ignores the strategy argument and returns the constant
          # eleven.

          y = constant_op.constant(11)
          return y

        def replica_fn():
          # [FUNCTION DOC]
          # HELPER: per-replica function; inside an init scope it performs a merge call, adds one
          # to the merged value and returns it.

          with ops.init_scope():
            y = distribute_lib.get_replica_context().merge_call(merge_fn)
            z = y + 1
            return z

        return strategy.run(replica_fn)

      result = strategy.experimental_local_results(fn())
      self.assertAllClose(result, [12] * _get_num_replicas_per_client(strategy))


@combinations.generate(
    combinations.combine(
        distribution=[
            strategy_combinations.mirrored_strategy_with_cpu_1_and_2,
            strategy_combinations.multi_worker_mirrored_2x2_gpu,
            strategy_combinations.tpu_strategy
        ],
        mode=['graph', 'eager']))
class StrategyLocalResultTest(test.TestCase):
  # [CLASS DOC]
  # Tests that the strategy's local-results conversion preserves nested structures and values
  # returned from per-replica functions. Runs in graph and eager mode over a two-CPU mirrored,
  # a multi-worker mirrored and a TPU descriptor; each test receives a distribution.

  def testLocalResultForDictionary(self, distribution):
    # [FUNCTION DOC]
    # Runs a function returning a mapping of two constants on every replica and asserts the local
    # results are a pair of identical mappings.

    @def_function.function
    def model_fn():
      # [FUNCTION DOC]
      # HELPER: returns a mapping with float values one and two under keys a and b.
      return {'a': constant_op.constant(1.), 'b': constant_op.constant(2.)}

    with distribution.scope():
      result = distribution.run(model_fn)
      got = self.evaluate(distribution.experimental_local_results(result))
      self.assertEqual(got, ({'a': 1., 'b': 2.}, {'a': 1., 'b': 2.}))

  def testLocalResultForList(self, distribution):
    # [FUNCTION DOC]
    # Runs a function returning a list of two constants and asserts the local results are a pair
    # of identical lists.

    @def_function.function
    def model_fn():
      # [FUNCTION DOC]
      # HELPER: returns a list holding float constants one and two.
      return [constant_op.constant(1.), constant_op.constant(2.)]

    with distribution.scope():
      result = distribution.run(model_fn)
      got = self.evaluate(distribution.experimental_local_results(result))
      self.assertEqual(got, ([1., 2.], [1., 2.]))

  def testLocalResultForTuple(self, distribution):
    # [FUNCTION DOC]
    # Runs a function returning a tuple of three constants and asserts the local results are a pair
    # of identical tuples.

    @def_function.function
    def model_fn():
      # [FUNCTION DOC]
      # HELPER: returns a tuple of float constants one, two and three.
      return (constant_op.constant(1.), constant_op.constant(2.),
              constant_op.constant(3.))

    with distribution.scope():
      result = distribution.run(model_fn)
      got = self.evaluate(distribution.experimental_local_results(result))
      self.assertEqual(got, ((1., 2., 3.), (1., 2., 3.)))

  def testLocalResultForNestedStruct(self, distribution):
    # [FUNCTION DOC]
    # Runs a function returning a tuple of two mappings of constants and asserts the local results
    # are a pair of identical copies of that nested structure.

    @def_function.function
    def model_fn():
      # [FUNCTION DOC]
      # HELPER: returns a pair of mappings: (a=1, b=2) and (a=4, b=6) as float constants.
      return ({
          'a': constant_op.constant(1.),
          'b': constant_op.constant(2.)
      }, {
          'a': constant_op.constant(4.),
          'b': constant_op.constant(6.)
      })

    with distribution.scope():
      result = distribution.run(model_fn)
      got = self.evaluate(distribution.experimental_local_results(result))
      self.assertEqual(got, (({
          'a': 1.,
          'b': 2.
      }, {
          'a': 4.,
          'b': 6.
      }), ({
          'a': 1.,
          'b': 2.
      }, {
          'a': 4.,
          'b': 6.
      })))

  def testLocalResultForNestedStructWithoutTensor(self, distribution):
    # [FUNCTION DOC]
    # Runs a function returning a mapping of plain Python floats (no tensors) and asserts the local
    # results form a tuple of two identical mappings.

    @def_function.function
    def model_fn():
      # [FUNCTION DOC]
      # HELPER: returns a mapping of plain float numbers one and two under keys a and b.
      return {'a': 1., 'b': 2.}

    with distribution.scope():
      result = distribution.run(model_fn)
      v = self.evaluate(distribution.experimental_local_results(result))
      self.assertIsInstance(v, tuple)
      self.assertAllEqual(v, ({'a': 1., 'b': 2.}, {'a': 1., 'b': 2.}))

  def testLocalResultForScalarValue(self, distribution):
    # [FUNCTION DOC]
    # Runs a function returning each replica's local replica index and asserts the local results
    # are a tuple equal to zero and one (so this expects exactly two replicas).

    @def_function.function
    def model_fn():
      # [FUNCTION DOC]
      # HELPER: returns the local replica index derived from the replica id in the sync group.
      return distribution.extended._get_local_replica_id(
          distribute_lib.get_replica_context().replica_id_in_sync_group)

    with distribution.scope():
      result = distribution.run(model_fn)
      v = self.evaluate(distribution.experimental_local_results(result))
      self.assertIsInstance(v, tuple)
      self.assertEqual(v, (0, 1))

  def testLocalResultForDictionaryDifferentReplicas(self, distribution):
    # [FUNCTION DOC]
    # Runs a function whose mapping values depend on the replica index and asserts replica zero
    # got (a=1, b=2) and replica one got (a=2, b=3) in the local results.

    @def_function.function
    def model_fn():
      # [FUNCTION DOC]
      # HELPER: returns a mapping with a equal to local replica index plus one and b equal to
      # index plus two, as floats.
      replica_id = distribution.extended._get_local_replica_id(
          distribute_lib.get_replica_context().replica_id_in_sync_group)
      return {
          'a': math_ops.cast(replica_id + 1, dtype=float),
          'b': math_ops.cast(replica_id + 2, dtype=float)
      }

    with distribution.scope():
      result = distribution.run(model_fn)
      got = self.evaluate(distribution.experimental_local_results(result))
      self.assertAllEqual(got, ({'a': 1., 'b': 2.}, {'a': 2., 'b': 3.}))

  def testLocalResultForTensor(self, distribution):
    # [FUNCTION DOC]
    # Runs a function returning a two-element tensor and asserts the local results are a pair of
    # identical two-element values.

    @def_function.function
    def model_fn():
      # [FUNCTION DOC]
      # HELPER: returns a float vector with values two and three.
      return constant_op.constant([2., 3.])

    with distribution.scope():
      result = distribution.run(model_fn)
      v = self.evaluate(distribution.experimental_local_results(result))
      self.assertAllEqual(v, ([2., 3.], [2., 3.]))


@combinations.generate(
    combinations.combine(
        strategy=[
            strategy_combinations.multi_worker_mirrored_2x1_cpu,
            strategy_combinations.multi_worker_mirrored_2x1_gpu,
        ] + strategy_combinations.all_strategies,
        mode=['eager']))
class ReduceTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests the strategy reduce method, in eager mode over two multi-worker mirrored descriptors
  # and all strategies in the shared all-strategies list.

  def testBasic(self, strategy):
    # [FUNCTION DOC]
    # Builds a per-replica value of one on every replica, then sums it with reduce both eagerly
    # and as a compiled function, inside and outside the strategy scope. Asserts every result
    # equals the number of replicas in sync.
    per_replica_value = strategy.experimental_distribute_values_from_function(
        lambda _: array_ops.ones((), dtypes.float32))

    def fn_eager():
      # [FUNCTION DOC]
      # HELPER: sums the captured per-replica value across replicas with no axis and returns it.

      return strategy.reduce(
          reduce_util.ReduceOp.SUM, value=per_replica_value, axis=None)

    fn_graph = def_function.function(fn_eager)
    # Run reduce under the strategy scope to explicitly enter
    # strategy default_device scope.
    with strategy.scope():
      self.assertEqual(fn_eager().numpy(), 1.0 * strategy.num_replicas_in_sync)
      self.assertEqual(fn_graph().numpy(), 1.0 * strategy.num_replicas_in_sync)

    # Run reduce without a strategy scope to implicitly enter
    # strategy default_device scope.
    self.assertEqual(fn_eager().numpy(), 1.0 * strategy.num_replicas_in_sync)
    self.assertEqual(fn_graph().numpy(), 1.0 * strategy.num_replicas_in_sync)

  def testAxis(self, strategy):
    # [FUNCTION DOC]
    # Runs a function returning the vector one, two on each replica, then reduces along axis zero
    # with mean and with sum. Asserts the mean equals one and a half and the sum equals three times
    # the number of replicas in sync.

    @def_function.function
    def fn():
      # [FUNCTION DOC]
      # HELPER: returns a float vector with values one and two.
      return constant_op.constant([1., 2.])

    x = strategy.run(fn)

    x_m = strategy.reduce(reduce_util.ReduceOp.MEAN, x, axis=0)
    self.assertEqual(1.5, x_m)
    x_s = strategy.reduce(reduce_util.ReduceOp.SUM, x, axis=0)
    self.assertEqual(3 * strategy.num_replicas_in_sync, x_s)


@combinations.generate(
    combinations.combine(
        strategy=[
            strategy_combinations.default_strategy,
            strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
            strategy_combinations.mirrored_strategy_with_two_gpus_no_merge_call,
            strategy_combinations.tpu_strategy,
            strategy_combinations.tpu_strategy_packed_var,
            strategy_combinations.multi_worker_mirrored_2x1_cpu,
            strategy_combinations.multi_worker_mirrored_2x2_gpu,
            strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call,
        ],
        update_fn=['assign', 'assign_add', 'assign_sub'],
        tf_function=[True, False],
        mode=['eager']))
class ReplicaCtxUpdateTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests the replica context private update method on distributed variables for assign,
  # assign-add and assign-sub, with and without compiling, over several strategies including ones
  # with merge calls disabled. Eager mode only.

  def testDenseUpdate(self, strategy, tf_function, update_fn):
    # [FUNCTION DOC]
    # Skips TPU strategies when not compiling. Creates a variable with initial value five under the
    # strategy scope, runs on each replica a function that applies the chosen update twice (first
    # with a tensor two, then with the plain number one), optionally compiled. Asserts the local
    # results on every replica client equal the expected value: assign gives one, assign-add gives
    # eight and assign-sub gives two.
    if strategy_test_lib.is_tpu_strategy(strategy) and (not tf_function):
      self.skipTest('Skip TPUStrategy + eager combination.')
    with strategy.scope():
      distributed_variable1 = variables.Variable(5.0)

    def replica_fn():
      # [FUNCTION DOC]
      # HELPER: per-replica function that applies the selected update operation to the distributed
      # variable through the replica context, first with a tensor value and then with a plain
      # number. Returns nothing.
      value = array_ops.constant(2.)
      python_literal = 1.
      replica_context = distribute_lib.get_replica_context()
      fn_sets = {
          'assign': lambda var, value: var.assign(value),
          'assign_add': lambda var, value: var.assign_add(value),
          'assign_sub': lambda var, value: var.assign_sub(value),
      }
      replica_context._update(
          distributed_variable1, fn_sets[update_fn], args=(value,))
      replica_context._update(
          distributed_variable1, fn_sets[update_fn], args=(python_literal,))

    if tf_function:
      replica_fn = def_function.function(replica_fn)
    strategy.run(replica_fn)

    expected_result = {'assign': 1., 'assign_add': 8., 'assign_sub': 2.}
    self.assertAllEqual(
        strategy.experimental_local_results(distributed_variable1),
        [expected_result[update_fn]] * _get_num_replicas_per_client(strategy))


@combinations.generate(
    combinations.combine(
        strategy=[
            strategy_combinations.multi_worker_mirrored_2x1_cpu,
            strategy_combinations.multi_worker_mirrored_2x1_gpu,
            strategy_combinations.multi_worker_mirrored_2x2_gpu,
            strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call,
            strategy_combinations.tpu_strategy,
        ] + strategy_combinations.strategies_minus_tpu,
        tf_function=[combinations.tf_function, combinations.no_tf_function],
        mode=['eager']))
class ReplicaCtxAllReduceTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests the strategy extended private replica-context all-reduce for dense, sparse, nested and
  # synchronized-on-read inputs, with compiled and uncompiled execution. Eager mode only, over
  # multi-worker mirrored, TPU and all non-TPU strategies.

  def testDense(self, strategy, tf_function):
    # [FUNCTION DOC]
    # Skips TPU strategies without compilation. Each replica sums the constant one via the private
    # replica-context all-reduce; asserts the first local result equals the number of replicas.
    if (strategy_test_lib.is_tpu_strategy(strategy) and
        tf_function is combinations.no_tf_function):
      self.skipTest('Skip TPUStrategy + eager combination.')

    @tf_function
    def fn():
      # [FUNCTION DOC]
      # HELPER: runs the replica function on the strategy and returns the local results.

      def replica_fn():
        # [FUNCTION DOC]
        # HELPER: all-reduces (sum) the constant one across replicas and returns the result.
        value = array_ops.identity(1.0)
        reduced = strategy.extended._replica_ctx_all_reduce(
            reduce_util.ReduceOp.SUM, value)
        return reduced

      return strategy.experimental_local_results(strategy.run(replica_fn))

    got = fn()[0]
    self.assertEqual(got, 1.0 * strategy.num_replicas_in_sync)

  def testSparse(self, strategy, tf_function):
    # [FUNCTION DOC]
    # Skipped without compilation. Each replica all-reduces (sum) a sparse indexed-slices value
    # (one at index zero, dense shape five by one) through the private replica-context call.
    # Asserts the dense conversion equals the same slices with the value scaled by replica count.
    if tf_function is combinations.no_tf_function:
      self.skipTest('Skip IndexedSlices + eager combination.')

    @tf_function
    def fn():
      # [FUNCTION DOC]
      # HELPER: runs the replica function on the strategy and returns the local results.

      def replica_fn():
        # [FUNCTION DOC]
        # HELPER: all-reduces (sum) a one-row sparse indexed-slices value and returns the result.
        value = indexed_slices.IndexedSlices(
            values=array_ops.identity([[1.0]]),
            indices=array_ops.identity([0]),
            dense_shape=array_ops.identity([5, 1]))
        reduced = strategy.extended._replica_ctx_all_reduce(
            reduce_util.ReduceOp.SUM, value)
        return reduced

      return strategy.experimental_local_results(strategy.run(replica_fn))

    got = fn()[0]
    expect = indexed_slices.IndexedSlices(
        values=array_ops.identity([[1.0 * strategy.num_replicas_in_sync]]),
        indices=array_ops.identity([0]),
        dense_shape=array_ops.identity([5, 1]))
    self.assertAllEqual(
        ops.convert_to_tensor(got), ops.convert_to_tensor(expect))

  def testNestedInput(self, strategy, tf_function):
    # [FUNCTION DOC]
    # Skipped without compilation. Each replica all-reduces (sum) a tuple of a dense scalar, a
    # sparse slice, another dense scalar and another sparse slice through the private
    # replica-context call. Asserts every element is scaled by the replica count.
    if tf_function is combinations.no_tf_function:
      self.skipTest('Skip IndexedSlices + eager combination.')

    @tf_function
    def fn():
      # [FUNCTION DOC]
      # HELPER: runs the replica function on the strategy and returns the local results.

      def replica_fn():
        # [FUNCTION DOC]
        # HELPER: all-reduces (sum) a nested tuple of dense and sparse values and returns it.
        value = (array_ops.identity(1.0),
                 indexed_slices.IndexedSlices(
                     values=array_ops.identity([[1.0]]),
                     indices=array_ops.identity([0]),
                     dense_shape=array_ops.identity([5, 1])),
                 array_ops.identity(2.0),
                 indexed_slices.IndexedSlices(
                     values=array_ops.identity([[2.0]]),
                     indices=array_ops.identity([1]),
                     dense_shape=array_ops.identity([5, 1])))
        reduced = strategy.extended._replica_ctx_all_reduce(
            reduce_util.ReduceOp.SUM, value)
        return reduced

      return strategy.experimental_local_results(strategy.run(replica_fn))

    got = fn()[0]
    expect = (1.0 * strategy.num_replicas_in_sync,
              indexed_slices.IndexedSlices(
                  values=array_ops.identity(
                      [[1.0 * strategy.num_replicas_in_sync]]),
                  indices=array_ops.identity([0]),
                  dense_shape=array_ops.identity([5, 1])),
              2.0 * strategy.num_replicas_in_sync,
              indexed_slices.IndexedSlices(
                  values=array_ops.identity(
                      [[2.0 * strategy.num_replicas_in_sync]]),
                  indices=array_ops.identity([1]),
                  dense_shape=array_ops.identity([5, 1])))

    self.assertAllClose(
        nest.map_structure(ops.convert_to_tensor, got),
        nest.map_structure(ops.convert_to_tensor, expect))

  def testSyncOnReadVariableInput(self, strategy, tf_function):
    # [FUNCTION DOC]
    # Skips strategies that do not use synchronized-on-read variables (anything other than mirrored,
    # multi-worker mirrored and TPU), TPU without compilation, and multi-worker mirrored with
    # compilation (known bug reference in the skip message). Creates an on-read variable, has each
    # replica assign three times its replica id and all-reduces (sum) the variable through the
    # replica context. Asserts that every local replica on this client sees the sum of the replica
    # ids times three.
    if (not strategy_test_lib.is_mirrored_strategy(strategy) and
        not strategy_test_lib.is_multi_worker_mirrored_strategy(strategy) and
        not strategy_test_lib.is_tpu_strategy(strategy)):
      self.skipTest('Skip strategies not using SyncOnReadVariables.')
    if (strategy_test_lib.is_tpu_strategy(strategy) and
        tf_function is combinations.no_tf_function):
      self.skipTest('Skip TPUStrategy + eager combination.')
    if (strategy_test_lib.is_multi_worker_mirrored_strategy(strategy) and
        tf_function is combinations.tf_function):
      self.skipTest('Skip MWMS + graph combination until b/228512201 is fixed.')

    with strategy.scope():
      var = variables.Variable(
          0.0,
          synchronization=variables.VariableSynchronization.ON_READ,
          aggregation=variables.VariableAggregation.ONLY_FIRST_REPLICA)

    @tf_function
    def replica_fn():
      # [FUNCTION DOC]
      # HELPER: per-replica function that assigns three times the replica id (as float) to the
      # on-read variable and returns the sum all-reduce of the variable across replicas.
      replica_context = distribute_lib.get_replica_context()
      replica_id = replica_context.replica_id_in_sync_group
      var.assign(math_ops.cast(replica_id, dtype=float) * 3.0)

      return replica_context.all_reduce(reduce_util.ReduceOp.SUM, var)

    if strategy_test_lib.is_multi_worker_mirrored_strategy(strategy):
      client_local_replica_num = strategy.extended._num_devices_per_worker
    else:
      client_local_replica_num = strategy.num_replicas_in_sync

    workers_num = strategy.num_replicas_in_sync
    expected_sum = sum(range(workers_num)) * 3.0

    # Expand the values on each replica if multiple devices are used; otherwise
    # simple read the value of the Tensor.
    result = strategy.run(replica_fn)
    if hasattr(result, 'values'):
      result = result.values
    result = nest.flatten(result)

    # Iterate through all replicas and verify the reduce sum result.
    for i in range(client_local_replica_num):
      self.assertEqual(result[i].numpy(), expected_sum)


@combinations.generate(
    combinations.combine(
        strategy=[
            strategy_combinations.multi_worker_mirrored_2x1_cpu,
            strategy_combinations.multi_worker_mirrored_2x1_gpu,
            strategy_combinations.multi_worker_mirrored_2x2_gpu,
            strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call,
            strategy_combinations.tpu_strategy,
        ] + strategy_combinations.strategies_minus_tpu,
        tf_function=[combinations.tf_function, combinations.no_tf_function],
        mode=['eager']))
class AllReduceTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests the public replica-context all-reduce for dense, sparse, tuple-of-sparse and nested
  # inputs, with compiled and uncompiled execution. Eager mode only, over the same strategies as
  # the replica-context all-reduce tests.

  def testDense(self, strategy, tf_function):
    # [FUNCTION DOC]
    # Skips TPU strategies without compilation. Each replica sums the constant one with the public
    # all-reduce; asserts the first local result equals the number of replicas.
    if (strategy_test_lib.is_tpu_strategy(strategy) and
        tf_function is combinations.no_tf_function):
      self.skipTest('Skip TPUStrategy + eager combination.')

    @tf_function
    def fn():
      # [FUNCTION DOC]
      # HELPER: runs the replica function on the strategy and returns the local results.

      def replica_fn():
        # [FUNCTION DOC]
        # HELPER: all-reduces (sum) the constant one across replicas and returns the result.
        value = array_ops.identity(1.0)
        rep_ctx = distribute_lib.get_replica_context()
        reduced = rep_ctx.all_reduce(reduce_util.ReduceOp.SUM, value)
        return reduced

      return strategy.experimental_local_results(strategy.run(replica_fn))

    got = fn()[0]
    self.assertEqual(got, 1.0 * strategy.num_replicas_in_sync)

  def testSparse(self, strategy, tf_function):
    # [FUNCTION DOC]
    # Skipped without compilation. Each replica all-reduces with mean a one-row sparse
    # indexed-slices value through the public call. Asserts (except on TPU) that the result is still
    # an indexed-slices object and that its dense conversion equals the unchanged input (mean of
    # identical values).
    if tf_function is combinations.no_tf_function:
      self.skipTest('Skip IndexedSlices + eager combination.')

    @tf_function
    def fn():
      # [FUNCTION DOC]
      # HELPER: runs the replica function on the strategy and returns the local results.

      def replica_fn():
        # [FUNCTION DOC]
        # HELPER: all-reduces (mean) a one-row sparse indexed-slices value and returns the result.
        value = indexed_slices.IndexedSlices(
            values=array_ops.identity([[1.0]]),
            indices=array_ops.identity([0]),
            dense_shape=array_ops.identity([5, 1]))
        rep_ctx = distribute_lib.get_replica_context()
        reduced = rep_ctx.all_reduce(reduce_util.ReduceOp.MEAN, value)
        return reduced

      return strategy.experimental_local_results(strategy.run(replica_fn))

    got = fn()[0]

    if not strategy_test_lib.is_tpu_strategy(strategy):
      self.assertIsInstance(got, indexed_slices.IndexedSlices)
    expect = indexed_slices.IndexedSlices(
        values=array_ops.identity([[1.0]]),
        indices=array_ops.identity([0]),
        dense_shape=array_ops.identity([5, 1]))
    self.assertAllEqual(
        ops.convert_to_tensor(got), ops.convert_to_tensor(expect))

  def testSparseTuple(self, strategy, tf_function):
    # [FUNCTION DOC]
    # Skipped without compilation. Each replica all-reduces (sum) a list of two sparse
    # indexed-slices values (values one and two at index zero). Asserts (except on TPU) that each
    # result element is an indexed-slices object and that the dense conversions equal the inputs
    # scaled by the replica count.
    if tf_function is combinations.no_tf_function:
      self.skipTest('Skip IndexedSlices + eager combination.')

    @tf_function
    def fn():
      # [FUNCTION DOC]
      # HELPER: runs the replica function on the strategy and returns the local results.

      def replica_fn():
        # [FUNCTION DOC]
        # HELPER: all-reduces (sum) a list of two sparse values and returns the reduced list.
        value1 = indexed_slices.IndexedSlices(
            values=array_ops.identity([[1.0]]),
            indices=array_ops.identity([0]),
            dense_shape=array_ops.identity([5, 1]))
        value2 = indexed_slices.IndexedSlices(
            values=array_ops.identity([[2.0]]),
            indices=array_ops.identity([0]),
            dense_shape=array_ops.identity([5, 1]))
        rep_ctx = distribute_lib.get_replica_context()
        reduced = rep_ctx.all_reduce(reduce_util.ReduceOp.SUM, [value1, value2])
        return reduced

      return strategy.experimental_local_results(strategy.run(replica_fn))

    got = fn()[0]

    if not strategy_test_lib.is_tpu_strategy(strategy):
      for g in got:
        self.assertIsInstance(g, indexed_slices.IndexedSlices)
    expect = [
        indexed_slices.IndexedSlices(
            values=array_ops.identity([[1.0 * strategy.num_replicas_in_sync]]),
            indices=array_ops.identity([0]),
            dense_shape=array_ops.identity([5, 1])),
        indexed_slices.IndexedSlices(
            values=array_ops.identity([[2.0 * strategy.num_replicas_in_sync]]),
            indices=array_ops.identity([0]),
            dense_shape=array_ops.identity([5, 1]))
    ]
    self.assertAllEqual(
        nest.map_structure(ops.convert_to_tensor, got),
        nest.map_structure(ops.convert_to_tensor, expect))

  def testNestedInput(self, strategy, tf_function):
    # [FUNCTION DOC]
    # Skipped without compilation. Each replica all-reduces (sum) a tuple of a dense scalar, a
    # sparse slice, another dense scalar and another sparse slice through the public call.
    # Asserts every element is scaled by the replica count.
    if tf_function is combinations.no_tf_function:
      self.skipTest('Skip IndexedSlices + eager combination.')

    @tf_function
    def fn():
      # [FUNCTION DOC]
      # HELPER: runs the replica function on the strategy and returns the local results.

      def replica_fn():
        # [FUNCTION DOC]
        # HELPER: all-reduces (sum) a nested tuple of dense and sparse values and returns it.
        value = (array_ops.identity(1.0),
                 indexed_slices.IndexedSlices(
                     values=array_ops.identity([[1.0]]),
                     indices=array_ops.identity([0]),
                     dense_shape=array_ops.identity([5, 1])),
                 array_ops.identity(2.0),
                 indexed_slices.IndexedSlices(
                     values=array_ops.identity([[2.0]]),
                     indices=array_ops.identity([1]),
                     dense_shape=array_ops.identity([5, 1])))
        rep_ctx = distribute_lib.get_replica_context()
        reduced = rep_ctx.all_reduce(reduce_util.ReduceOp.SUM, value)
        return reduced

      return strategy.experimental_local_results(strategy.run(replica_fn))

    got = fn()[0]
    expect = (1.0 * strategy.num_replicas_in_sync,
              indexed_slices.IndexedSlices(
                  values=array_ops.identity(
                      [[1.0 * strategy.num_replicas_in_sync]]),
                  indices=array_ops.identity([0]),
                  dense_shape=array_ops.identity([5, 1])),
              2.0 * strategy.num_replicas_in_sync,
              indexed_slices.IndexedSlices(
                  values=array_ops.identity(
                      [[2.0 * strategy.num_replicas_in_sync]]),
                  indices=array_ops.identity([1]),
                  dense_shape=array_ops.identity([5, 1])))

    self.assertAllClose(
        nest.map_structure(ops.convert_to_tensor, got),
        nest.map_structure(ops.convert_to_tensor, expect))


def _make_indexed_slices(values, indices, dense_shape):
  # [FUNCTION DOC]
  # HELPER: builds a sparse indexed-slices object from three plain array-like arguments (values,
  # indices and dense shape), converting each to a constant tensor, and returns it.
  # KNOWN ISSUE: this helper is never called anywhere in this file (dead code).
  tensor = indexed_slices.IndexedSlices(
      values=constant_op.constant(values),
      indices=constant_op.constant(indices),
      dense_shape=constant_op.constant(dense_shape))
  return tensor


def _get_num_replicas_per_client(strategy):
  # [FUNCTION DOC]
  # HELPER: number of replicas hosted by this client process, for the tests' expected values.
  # For a collective all-reduce strategy returns the larger of one and the first accelerator count
  # from its cluster resolver; for any other strategy returns the total replicas in sync.
  if isinstance(strategy, CollectiveAllReduceStrategy):
    resolver = strategy.cluster_resolver
    return max(nest.flatten(resolver.num_accelerators())[0], 1)
  else:
    return strategy.num_replicas_in_sync


@combinations.generate(
    combinations.combine(
        strategy=[
            strategy_combinations.multi_worker_mirrored_2x1_cpu,
            strategy_combinations.multi_worker_mirrored_2x1_gpu,
        ],
        mode=['eager']))
class DistributedCollectiveAllReduceStrategyTest(
    strategy_test_lib.DistributionTestBase,
    parameterized.TestCase):
  # [CLASS DOC]
  # Multi-worker (chief plus one worker, CPU or GPU) collective all-reduce tests on top of the
  # shared distribution test base: dataset input pipelines including partial last batches and
  # reductions of host and device tensors. Expected values assume exactly two workers.

  def testDatasetFromFunction(self, strategy):
    # [FUNCTION DOC]
    # Distributes a dataset built from a function (range of one hundred, repeated, batched by the
    # per-replica batch size of a global size ten, sharded per input pipeline), takes one element
    # and sums it. Asserts the sum is ten on the chief and thirty-five on the worker.
    def dataset_fn(input_context):
      # [FUNCTION DOC]
      # HELPER: dataset factory taking an input context; builds a repeated, batched range
      # dataset with a per-replica batch size from a global batch of ten and shards it by the
      # context's pipeline count and id. Returns the dataset.
      global_batch_size = 10
      batch_size = input_context.get_per_replica_batch_size(global_batch_size)
      d = dataset_ops.DatasetV2.range(100).repeat().batch(batch_size)
      return d.shard(input_context.num_input_pipelines,
                     input_context.input_pipeline_id)

    expected_sum_on_workers = {'chief': 10, 'worker': 35}
    input_iterator = iter(
        strategy.distribute_datasets_from_function(dataset_fn))

    @def_function.function
    def run(iterator):
      # [FUNCTION DOC]
      # HELPER: compiled function that fetches the next element from the distributed iterator
      # and returns the per-replica local results.
      return strategy.experimental_local_results(iterator.get_next())

    result = run(input_iterator)
    sum_value = math_ops.reduce_sum(result)
    self.assertEqual(
        sum_value.numpy(),
        expected_sum_on_workers[multi_worker_test_base.get_task_type()])

  def testSimpleInputFromDatasetLastPartialBatch(self, strategy):
    # [FUNCTION DOC]
    # Distributes a range-of-fourteen dataset batched by eight without dropping the remainder,
    # runs one full batch, then an incomplete batch. Asserts the incomplete batch holds eight to
    # ten on the chief and eleven to thirteen on the worker.
    global_batch_size = 8
    dataset = dataset_ops.DatasetV2.range(14).batch(
        global_batch_size, drop_remainder=False)
    input_iterator = iter(strategy.experimental_distribute_dataset(dataset))

    @def_function.function
    def run(input_iterator):
      # [FUNCTION DOC]
      # HELPER: compiled function that runs an identity function on the strategy with the next
      # element of the distributed iterator and returns the result.
      return strategy.run(lambda x: x, args=(next(input_iterator),))

    # Let the complete batch go.
    run(input_iterator)

    # `result` is an incomplete batch
    result = run(input_iterator)
    expected_data_on_workers = {'chief': [8, 9, 10], 'worker': [11, 12, 13]}
    self.assertAllEqual(
        expected_data_on_workers[multi_worker_test_base.get_task_type()],
        result.numpy(),
    )

  def testSimpleInputFromFnLastPartialBatch(self, strategy):
    # [FUNCTION DOC]
    # Distributes a function-built range-of-fourteen dataset (per-replica batch from a global size
    # of eight, remainder kept, sharded per pipeline), runs one full batch, then an incomplete
    # batch. Asserts the incomplete batch holds eight to eleven on the chief and twelve and
    # thirteen on the worker.

    def dataset_fn(input_context):
      # [FUNCTION DOC]
      # HELPER: dataset factory; builds a range-of-fourteen dataset batched by the per-replica
      # size (without dropping the remainder) and shards it by the context's pipeline count and id.
      global_batch_size = 8
      batch_size = input_context.get_per_replica_batch_size(global_batch_size)
      dataset = dataset_ops.DatasetV2.range(14).batch(
          batch_size, drop_remainder=False)
      return dataset.shard(input_context.num_input_pipelines,
                           input_context.input_pipeline_id)

    input_iterator = iter(
        strategy.distribute_datasets_from_function(dataset_fn))

    @def_function.function
    def run(input_iterator):
      # [FUNCTION DOC]
      # HELPER: compiled function that runs an identity function on the strategy with the next
      # element of the distributed iterator and returns the result.
      return strategy.run(lambda x: x, args=(next(input_iterator),))

    # Let the complete batch go.
    run(input_iterator)
    # `result` is an incomplete batch
    result = run(input_iterator)

    expected_data_on_worker = {'chief': [8, 9, 10, 11], 'worker': [12, 13]}
    self.assertAllEqual(
        expected_data_on_worker[multi_worker_test_base.get_task_type()],
        result.numpy())

  def testReduceHostTensor(self, strategy):
    # [FUNCTION DOC]
    # Sums a plain host tensor of one across the two workers with reduce and asserts the result is two.
    reduced = strategy.reduce(
        reduce_util.ReduceOp.SUM, array_ops.identity(1.), axis=None)
    self.assertEqual(reduced.numpy(), 2.)

  def testReduceToHostTensor(self, strategy):
    # [FUNCTION DOC]
    # Sums a host tensor of one with the extended reduce-to (destination is the value itself) and
    # asserts the result is two.
    value = array_ops.identity(1.)
    reduced = strategy.extended.reduce_to(reduce_util.ReduceOp.SUM, value,
                                          value)
    self.assertEqual(reduced.numpy(), 2.)

  def testBatchReduceToHostTensor(self, strategy):
    # [FUNCTION DOC]
    # Sums two pairs of a host tensor of one with the extended batch reduce-to and asserts the
    # result is a list of two, two.
    value = array_ops.identity(1.)
    reduced = strategy.extended.batch_reduce_to(reduce_util.ReduceOp.SUM,
                                                [(value, value),
                                                 (value, value)])
    self.assertAllEqual([2., 2.], reduced)

  def testReduceDeviceTensors(self, strategy):
    # [FUNCTION DOC]
    # Creates a per-replica value of one by running on the strategy, sums it with reduce (no
    # axis) and asserts the result is two.
    value = strategy.run(lambda: array_ops.identity(1.))
    reduced = strategy.reduce(reduce_util.ReduceOp.SUM, value, axis=None)
    self.assertEqual(reduced.numpy(), 2.)

  def testReduceToDeviceTensors(self, strategy):
    # [FUNCTION DOC]
    # Creates a per-replica value of one, sums it with the extended reduce-to (destination is
    # the value itself) and asserts the result is two.
    value = strategy.run(lambda: array_ops.identity(1.))
    reduced = strategy.extended.reduce_to(reduce_util.ReduceOp.SUM, value,
                                          value)
    self.assertEqual(reduced.numpy(), 2.)

  def testBatchReduceToDeviceTensors(self, strategy):
    # [FUNCTION DOC]
    # Creates a per-replica value of one, sums two pairs of it with the extended batch reduce-to
    # and asserts the result is a list of two, two.
    value = strategy.run(lambda: array_ops.identity(1.))
    reduced = strategy.extended.batch_reduce_to(reduce_util.ReduceOp.SUM,
                                                [(value, value),
                                                 (value, value)])
    self.assertAllEqual([2., 2.], reduced)

  # TODO(crccw): add a test that mixes device and host tensors after multi
  # worker strategy combinations can run on a fixed number of GPUs.


class StrategyClusterResolverTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests the cluster resolver property of strategies in eager mode over a multi-worker mirrored
  # descriptor and all strategies in the shared all-strategies list.

  @combinations.generate(
      combinations.combine(
          strategy=[strategy_combinations.multi_worker_mirrored_2x1_cpu] +
          strategy_combinations.all_strategies,
          mode=['eager']))
  def testClusterResolverProperty(self, strategy):
    # [FUNCTION DOC]
    # Asserts the cluster resolver is nothing for strategies other than collective all-reduce
    # and TPU. For those two, asserts the resolver is the same object inside the strategy
    # scope and has the cluster spec, master, accelerator count, task id and task type
    # attributes; for collective all-reduce also asserts task id zero and a chief or worker task type.
    # CollectiveAllReduceStrategy and TPUStrategy must have a cluster resolver.
    # `None` otherwise.
    resolver = strategy.cluster_resolver
    if (not isinstance(strategy, CollectiveAllReduceStrategy) and
        not strategy_test_lib.is_tpu_strategy(strategy)):
      self.assertIsNone(resolver)
      return

    with strategy.scope():
      self.assertIs(strategy.cluster_resolver, resolver)

    self.assertTrue(hasattr(resolver, 'cluster_spec'))
    self.assertTrue(hasattr(resolver, 'master'))
    self.assertTrue(hasattr(resolver, 'num_accelerators'))
    self.assertTrue(hasattr(resolver, 'task_id'))
    self.assertTrue(hasattr(resolver, 'task_type'))
    if isinstance(strategy, CollectiveAllReduceStrategy):
      self.assertEqual(resolver.task_id, 0)
      self.assertAllInSet(resolver.task_type, ['chief', 'worker'])


if __name__ == '__main__':
  test_util.main()
