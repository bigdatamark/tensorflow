# [MODULE DOC]
# Tests for the multi-worker collective all-reduce strategy. Provides a test-object factory,
# a shared base test class, and test classes for a three-worker cluster (with and without a
# chief), single-worker mode, logical device handling, strategy-combination eager tests, and
# compatibility of the deprecated experimental class. Many tests need GPUs, TPUs or an
# in-process cluster; the main entry point disables logical device configuration. No import-time
# side effects beyond module-level aliases for the strategy classes.
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
"""Tests for CollectiveAllReduceStrategy."""

import copy
import functools

from absl.testing import parameterized
import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute.cluster_resolver import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib


CollectiveAllReduceStrategy = (
    collective_all_reduce_strategy.CollectiveAllReduceStrategy)
CollectiveAllReduceExtended = (
    collective_all_reduce_strategy.CollectiveAllReduceExtended)
_CollectiveAllReduceStrategyExperimental = (
    collective_all_reduce_strategy._CollectiveAllReduceStrategyExperimental)


# TODO(b/231630416): Create more tests to cover the case that strategy uses
# different number of GPUs than the number of physical devices.
def create_test_objects(cluster_spec=None,
                        task_type=None,
                        task_id=None,
                        num_gpus=None,
                        num_tpus=None):
  # [FUNCTION DOC]
  # HELPER: builds a strategy and a session target for tests. Parameters: cluster specification
  # (optional), task type and task id (needed together with the spec), GPU count (nothing means
  # the visible GPU count) and TPU count (nothing means the list of physical TPUs). Initializes
  # the TPU system if any TPUs. With a full cluster description builds a resolver for that task
  # and a target address of that task; otherwise an empty-cluster resolver and an empty target.
  # Returns a pair: the new strategy and the target string.
  if num_gpus is None:
    num_gpus = context.num_gpus()
  if num_tpus is None:
    num_tpus = context.context().list_physical_devices('TPU')
  if num_tpus:
    tpu_cluster_resolver.initialize_tpu_system()

  if cluster_spec and task_type and task_id is not None:
    cluster_resolver = cluster_resolver_lib.SimpleClusterResolver(
        cluster_spec=multi_worker_util.normalize_cluster_spec(cluster_spec),
        task_type=task_type,
        task_id=task_id,
        num_accelerators={'GPU': num_gpus, 'TPU': num_tpus})
    target = 'grpc://' + cluster_spec[task_type][task_id]
  else:
    cluster_resolver = cluster_resolver_lib.SimpleClusterResolver(
        server_lib.ClusterSpec({}),
        num_accelerators={'GPU': num_gpus, 'TPU': num_tpus},
    )
    target = ''

  strategy = collective_all_reduce_strategy.CollectiveAllReduceStrategy(
      cluster_resolver=cluster_resolver)

  return strategy, target


class CollectiveAllReduceStrategyTestBase(
    multi_worker_test_base.MultiWorkerTestBase):
  # [CLASS DOC]
  # HELPER: shared base class for the strategy tests. Offers a strategy factory (optionally with
  # an explicit device list) and reusable test routines for minimizing a loss in graph mode,
  # variable initialization and input-function iterators. Each test gets a fresh collective key
  # base so keys are never reused.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: bumps the strategy class's collective key base by one hundred thousand so every
    # test uses distinct collective keys, then calls the parent setup.
    # We use a different key_base for each test so that collective keys won't be
    # reused.
    CollectiveAllReduceStrategy._collective_key_base += 100000
    super(CollectiveAllReduceStrategyTestBase, self).setUp()

  def _get_test_object(self,
                       task_type,
                       task_id,
                       num_gpus=0,
                       num_tpus=0,
                       use_devices_arg=False):
    # [FUNCTION DOC]
    # HELPER: creates a strategy and target for a given task. Parameters: task type, task id, GPU
    # count, TPU count (both default zero) and whether to use an explicit device list. If the
    # device-list option is on and GPUs were requested, replaces the strategy's extended
    # implementation with one built from a list of that many GPU device names, and manually sets
    # the retrace-per-device flag (true for more than one GPU) because the base constructor is
    # bypassed. Returns the strategy and target.
    strategy, target = create_test_objects(
        cluster_spec=self._cluster_spec,
        task_type=task_type,
        task_id=task_id,
        num_gpus=num_gpus,
        num_tpus=num_tpus)

    if use_devices_arg and num_gpus > 0:
      devices = ['GPU:%d' % i for i in range(num_gpus)]
      # Temporary workaround to manually set the `_extended` field before device
      # initialization is exposed as a public interface.
      strategy._extended = CollectiveAllReduceExtended(
          container_strategy=strategy,
          cluster_resolver=None,
          communication_options=collective_util.Options(),
          devices=devices)
      # Manually set the field since the workaround bypasses the base
      # constructor, resulting in the absence of this field.
      strategy._extended._retrace_functions_for_each_device = (num_gpus > 1)

    return strategy, target

  def _test_minimize_loss_graph(self, task_type, task_id, num_gpus):
    # [FUNCTION DOC]
    # HELPER: shared graph-mode training check. Parameters: task type, task id and GPU count.
    # Builds the strategy, creates a one-by-one kernel variable, defines a squared-error loss
    # toward one, computes gradients per replica, reduces them with a sum and applies gradient
    # descent updates. Returns early with true if more local GPUs are configured than exist.
    # Otherwise initializes variables, runs ten steps and asserts the final error is smaller than
    # the initial error.
    distribution, master_target = self._get_test_object(task_type, task_id,
                                                        num_gpus)
    with ops.Graph().as_default(), \
         self.cached_session(target=master_target) as sess, \
         distribution.scope():
      initializer = functools.partial(
          init_ops_v2.GlorotUniform(), (1, 1), dtype=dtypes.float32)
      kernel = variables.Variable(
          initial_value=initializer,
          name='gpu_%d/kernel' % distribution.extended._num_devices_per_worker,
          trainable=True)

      def loss_fn(x):
        # [FUNCTION DOC]
        # HELPER: squared error of the product of the input with the kernel against one.
        # Returns a scalar loss.
        y = array_ops.reshape(
            gen_math_ops.mat_mul(x, kernel), []) - constant_op.constant(1.)
        return y * y

      # TODO(yuefengz, apassos): eager.backprop.implicit_grad is not safe for
      # multiple graphs (b/111216820).
      def grad_fn(x):
        # [FUNCTION DOC]
        # HELPER: computes the loss for an input and returns a list of pairs of gradient and
        # variable for all trainable variables in the graph.
        loss = loss_fn(x)
        var_list = (
            variables.trainable_variables() + ops.get_collection(
                ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES))
        grads = gradients.gradients(loss, var_list)
        ret = list(zip(grads, var_list))
        return ret

      def update(v, g):
        # [FUNCTION DOC]
        # HELPER: update function subtracting the gradient scaled by a learning rate of 0.05 from
        # the variable, with locking. Returns the assignment result.
        return v.assign_sub(0.05 * g, use_locking=True)

      one = constant_op.constant([[1.]])

      def step():
        # [FUNCTION DOC]
        # HELPER: builds one optimization step in the graph. Gets gradient-variable pairs from
        # each replica, reads each variable, sum-reduces the gradient to the variable's devices
        # and updates the variable. Returns two lists: variable values before and after the update.
        """Perform one optimization step."""
        # Run forward & backward to get gradients, variables list.
        g_v = distribution.extended.call_for_each_replica(grad_fn, args=[one])
        # Update the variables using the gradients and the update() function.
        before_list = []
        after_list = []
        for g, v in g_v:
          fetched = distribution.extended.read_var(v)
          before_list.append(fetched)
          with ops.control_dependencies([fetched]):
            # TODO(yuefengz): support non-Mirrored variable as destinations.
            g = distribution.extended.reduce_to(
                reduce_util.ReduceOp.SUM, g, destinations=v)
            with ops.control_dependencies(
                distribution.extended.update(v, update, args=(g,),
                                             group=False)):
              after_list.append(distribution.extended.read_var(v))
        return before_list, after_list

      before_out, after_out = step()

      if (distribution.extended._local_device_type == 'GPU' and
          context.num_gpus() < distribution.extended._num_devices_per_worker):
        return True

      sess.run(variables.global_variables_initializer())

      for i in range(10):
        b, a = sess.run((before_out, after_out))
        if i == 0:
          before, = b
        after, = a

      error_before = abs(before - 1)
      error_after = abs(after - 1)
      # Error should go down
      self.assertLess(error_after, error_before)

  def _test_variable_initialization(self, task_type, task_id, num_gpus):
    # [FUNCTION DOC]
    # HELPER: shared graph-mode check of variable initialization. Parameters: task type, task id
    # and GPU count. Creates a randomly initialized two-by-three variable on every replica,
    # reduces it with a mean, initializes variables and asserts the first replica's value is
    # close to the reduced mean (so all replicas started with the same value).
    distribution, master_target = self._get_test_object(task_type, task_id,
                                                        num_gpus)
    with ops.Graph().as_default(), \
         self.cached_session(target=master_target) as sess, \
         distribution.scope():

      def model_fn():
        # [FUNCTION DOC]
        # HELPER: replica function creating the random-initialized variable (values between one
        # and ten) and returning an identity of it.
        x = variable_scope.get_variable(
            'x',
            shape=(2, 3),
            initializer=init_ops.random_uniform_initializer(
                1.0, 10.0, dtype=dtypes.float32))
        return array_ops.identity(x)

      x = distribution.extended.call_for_each_replica(model_fn)
      reduced_x = distribution.reduce(reduce_util.ReduceOp.MEAN, x, axis=None)
      x = distribution.experimental_local_results(x)[0]

      sess.run(variables.global_variables_initializer())

      x_value, reduced_x_value = sess.run([x, reduced_x])
      self.assertTrue(
          np.allclose(x_value, reduced_x_value, atol=1e-5),
          msg=('x_value = %r, reduced_x_value = %r' % (x_value,
                                                       reduced_x_value)))

  def _test_input_fn_iterator(self,
                              task_type,
                              task_id,
                              num_gpus,
                              input_fn,
                              expected_values,
                              test_reinitialize=True,
                              ignore_order=False,
                              use_devices_arg=False):
    # [FUNCTION DOC]
    # HELPER: shared graph-mode check of input-function iterators. Parameters: task type, task id,
    # GPU count, the input function, the list of expected per-step values (one group per step),
    # whether to re-initialize and iterate again (default true), whether to ignore order within a
    # step (default false) and whether to use an explicit device list (default false). Iterates
    # to exhaustion comparing each step's per-replica values with expectations, asserts the
    # out-of-range error at the end and, if requested, repeats after re-initializing.
    distribution, master_target = self._get_test_object(
        task_type, task_id, num_gpus, use_devices_arg=use_devices_arg)
    devices = distribution.extended.worker_devices

    with ops.Graph().as_default(), \
         self.cached_session(target=master_target) as sess:
      iterator = distribution.make_input_fn_iterator(input_fn)
      sess.run(iterator.initializer)

      for expected_value in expected_values:
        next_element = iterator.get_next()
        computed_value = sess.run([distribute_utils.select_replica(
            r, next_element) for r in range(len(devices))])
        if ignore_order:
          self.assertCountEqual(list(expected_value), list(computed_value))
        else:
          self.assertEqual(list(expected_value), list(computed_value))

      with self.assertRaises(errors.OutOfRangeError):
        next_element = iterator.get_next()
        sess.run([distribute_utils.select_replica(r, next_element)
                  for r in range(len(devices))])

      # After re-initializing the iterator, should be able to iterate again.
      if test_reinitialize:
        sess.run(iterator.initializer)

        for expected_value in expected_values:
          next_element = iterator.get_next()
          computed_value = sess.run([
              distribute_utils.select_replica(r, next_element)
              for r in range(len(devices))])
          if ignore_order:
            self.assertCountEqual(list(expected_value), list(computed_value))
          else:
            self.assertEqual(list(expected_value), list(computed_value))


class DistributedCollectiveAllReduceStrategyTest(
    CollectiveAllReduceStrategyTestBase,
    strategy_test_lib.DistributionTestBase,
    parameterized.TestCase):
  # [CLASS DOC]
  # Tests the strategy on an in-process cluster of three workers: replica counts, dataset
  # prefetch placement, minimize-loss, variable initialization, input iterators and session
  # config updating. Also inherits the generic strategy tests.

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Class-level fixture: starts an in-process cluster with three workers and no parameter
    # servers and stores its specification on the class.
    """Create a local cluster with 3 workers."""
    cls._cluster_spec = multi_worker_test_base.create_in_process_cluster(
        num_workers=3, num_ps=0)

  @combinations.generate(combinations.combine(mode=['graph']))
  def test_num_replicas_in_sync(self):
    # [FUNCTION DOC]
    # Graph-mode test: with two GPUs per task, asserts the strategy's replica count equals two
    # times the number of chief and worker tasks in the cluster.
    distribution, _ = create_test_objects(
        cluster_spec=self._cluster_spec,
        task_type='worker',
        task_id=0,
        num_gpus=2)
    num_workers = len(self._cluster_spec.get('chief', []) +
                      self._cluster_spec.get('worker', []))
    self.assertEqual(2 * num_workers,
                     distribution.num_replicas_in_sync)

  @combinations.generate(combinations.combine(
      mode=['graph'],
      prefetch_to_device=[None, True]))
  def test_prefetch_to_device_dataset(self, prefetch_to_device):
    # [FUNCTION DOC]
    # Graph-mode test parameterized by the fetch-to-device input option (unset or enabled).
    # Distributes a batched range dataset, takes the first element of its initializable iterator
    # and asserts every per-replica tensor lives on a GPU. Skips if the distributed dataset is not
    # the first-generation type.
    distribution, _ = self._get_test_object(
        task_type='worker', task_id=0, num_gpus=2)
    if prefetch_to_device is None:
      input_options = None
    else:
      input_options = distribute_lib.InputOptions(
          experimental_fetch_to_device=prefetch_to_device)
    dataset = dataset_ops.Dataset.range(100)
    dataset = dataset.batch(distribution.num_replicas_in_sync)
    dataset = distribution.experimental_distribute_dataset(
        dataset, options=input_options)
    if isinstance(dataset, input_lib_v1.DistributedDatasetV1):
      item = dataset.make_initializable_iterator().get_next()
    else:
      self.skipTest('unsupported test combination')
    device_types = {
        tf_device.DeviceSpec.from_string(tensor.device).device_type for
        tensor in item.values}
    self.assertAllEqual(list(device_types), ['GPU'])

  @combinations.generate(combinations.combine(mode=['graph']))
  def test_prefetch_to_host_dataset(self):
    # [FUNCTION DOC]
    # Graph-mode test like the device-prefetch one but with fetch-to-device disabled: asserts all
    # per-replica tensors of the first element are on a CPU. Skips if the distributed dataset is
    # not the first-generation type.
    distribution, _ = self._get_test_object(
        task_type='worker', task_id=0, num_gpus=2)
    input_options = distribute_lib.InputOptions(
        experimental_fetch_to_device=False)
    dataset = dataset_ops.Dataset.range(100)
    dataset = dataset.batch(distribution.num_replicas_in_sync)
    dataset = distribution.experimental_distribute_dataset(
        dataset, options=input_options)
    if isinstance(dataset, input_lib_v1.DistributedDatasetV1):
      item = dataset.make_initializable_iterator().get_next()
    else:
      self.skipTest('unsupported test combination')
    device_types = {
        tf_device.DeviceSpec.from_string(tensor.device).device_type for
        tensor in item.values}
    self.assertAllEqual(list(device_types), ['CPU'])

  @combinations.generate(
      combinations.combine(mode=['graph'], required_gpus=[0, 1, 2]))
  def testMinimizeLossGraph(self, required_gpus):
    # [FUNCTION DOC]
    # Graph-mode test on a three-worker cluster, parameterized by GPUs per worker. Runs the
    # shared minimize-loss routine on every task and expects the training error to shrink.
    self._run_between_graph_clients(self._test_minimize_loss_graph,
                                    self._cluster_spec, required_gpus)

  @combinations.generate(
      combinations.combine(mode=['graph'], required_gpus=[0, 1, 2]))
  def testVariableInitialization(self, required_gpus):
    # [FUNCTION DOC]
    # Graph-mode test on a three-worker cluster, parameterized by GPUs per worker. Runs the
    # shared variable-initialization routine on every task, expecting a variable's initial value
    # to agree with its reduced mean across replicas.
    self._run_between_graph_clients(
        self._test_variable_initialization,
        self._cluster_spec,
        num_gpus=required_gpus)

  @combinations.generate(
      combinations.combine(
          mode=['graph'], required_gpus=[0, 1, 2], use_dataset=[True, False]))
  def testMakeInputFnIterator(self, required_gpus, use_dataset):
    # [FUNCTION DOC]
    # Graph-mode test on a three-worker cluster, parameterized by GPUs per worker and by whether
    # the input function returns a dataset or a get-next callable. On each task, builds an input
    # function that also checks the input context (three pipelines, this task's id, replicas in
    # sync equal to three times the devices per worker) and verifies the iterator yields the
    # expected per-replica values, ignoring order in the callable case.
    def _worker_fn(task_type, task_id, required_gpus):
      # [FUNCTION DOC]
      # HELPER: per-task body run on every worker. Builds a range dataset of twenty elements
      # (or a one-shot-iterator get-next callable), computes the expected grouped values for the
      # number of devices per worker (at least one, CPU when no GPUs), wraps the function with the
      # input-context checks and runs the shared iterator test routine.
      if use_dataset:
        fn = lambda: dataset_ops.Dataset.range(20)
      else:
        def fn():
          # [FUNCTION DOC]
          # HELPER: input function returning the get-next callable of a one-shot iterator over
          # a range dataset of twenty elements.
          dataset = dataset_ops.Dataset.range(20)
          it = dataset_ops.make_one_shot_iterator(dataset)
          return it.get_next
      # We use CPU as the device when required_gpus = 0
      devices_per_worker = max(1, required_gpus)
      expected_values = [[i+j for j in range(devices_per_worker)]
                         for i in range(0, 20, devices_per_worker)]

      input_fn = self._input_fn_to_test_input_context(
          fn,
          expected_num_replicas_in_sync=3*devices_per_worker,
          expected_num_input_pipelines=3,
          expected_input_pipeline_id=task_id)
      self._test_input_fn_iterator(
          task_type,
          task_id,
          required_gpus,
          input_fn,
          expected_values,
          test_reinitialize=use_dataset,
          ignore_order=not use_dataset)

    self._run_between_graph_clients(_worker_fn, self._cluster_spec,
                                    required_gpus)

  @combinations.generate(combinations.combine(mode=['graph']))
  def testUpdateConfigProto(self):
    # [FUNCTION DOC]
    # Graph-mode test of session configuration updating for worker 1 of a three-worker cluster.
    # Starts from a config with a placeholder device filter and a placeholder scoped-allocator op,
    # updates it through the strategy, and asserts: the collective group leader is worker 0, the
    # device filters contain only worker 1, the scoped-allocator optimization is on, and the
    # enabled op list is exactly the collective reduce op.
    strategy, _ = self._get_test_object(
        task_type='worker', task_id=1, num_gpus=2)

    config_proto = config_pb2.ConfigProto(device_filters=['to_be_overridden'])
    rewrite_options = config_proto.graph_options.rewrite_options
    rewrite_options.scoped_allocator_opts.enable_op.append('to_be_removed')

    new_config = strategy.update_config_proto(config_proto)

    # Verify group leader
    self.assertEqual('/job:worker/replica:0/task:0',
                     new_config.experimental.collective_group_leader)

    # Verify device filters.
    self.assertEqual(['/job:worker/task:1'], new_config.device_filters)

    # Verify rewrite options.
    new_rewrite_options = new_config.graph_options.rewrite_options
    self.assertEqual(rewriter_config_pb2.RewriterConfig.ON,
                     new_rewrite_options.scoped_allocator_optimization)
    self.assertEqual(['CollectiveReduce'],
                     new_rewrite_options.scoped_allocator_opts.enable_op)


class DistributedCollectiveAllReduceStrategyTestWithChief(
    CollectiveAllReduceStrategyTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests the strategy on an in-process cluster of one chief and three workers, reusing the
  # shared test routines of the base class.

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Class-level fixture: starts an in-process cluster with three workers, no parameter servers
    # and a chief, and stores its specification on the class.
    """Create a local cluster with 3 workers and 1 chief."""
    cls._cluster_spec = multi_worker_test_base.create_in_process_cluster(
        num_workers=3, num_ps=0, has_chief=True)

  @combinations.generate(
      combinations.combine(mode=['graph'], required_gpus=[0, 1, 2]))
  def testMinimizeLossGraph(self, required_gpus):
    # [FUNCTION DOC]
    # Graph-mode test on a cluster with a chief plus three workers, parameterized by number of
    # GPUs per worker. Runs the shared minimize-loss routine on every task and expects the
    # error of a one-variable model to shrink after training.
    self._run_between_graph_clients(self._test_minimize_loss_graph,
                                    self._cluster_spec, required_gpus)

  @combinations.generate(
      combinations.combine(mode=['graph'], required_gpus=[0, 1, 2]))
  def testVariableInitialization(self, required_gpus):
    # [FUNCTION DOC]
    # Graph-mode test on the chief-plus-workers cluster, parameterized by GPUs per worker. Runs
    # the shared variable-initialization routine on every task, expecting a variable's initial
    # value to agree with its reduced mean across replicas.
    self._run_between_graph_clients(
        self._test_variable_initialization,
        self._cluster_spec,
        num_gpus=required_gpus)


class SingleWorkerCollectiveAllReduceStrategy(
    CollectiveAllReduceStrategyTestBase, strategy_test_lib.DistributionTestBase,
    strategy_test_lib.TwoDeviceDistributionTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests the strategy in single-worker mode (no cluster), with and without an explicit device
  # list, across graph and eager modes and varying GPU counts. Covers construction errors,
  # training, replica counts, input iterators, reductions, all-reduce, datasets, deep copy,
  # summaries and trainable variables, mostly via shared routines from the test base classes.

  @combinations.generate(combinations.combine(mode=['eager']))
  def testStrategyInitializationError(self):
    # [FUNCTION DOC]
    # Eager test: constructing the extended implementation with both a cluster resolver and an
    # explicit device list must raise a value error saying they cannot be set at the same time.
    with self.assertRaisesRegex(
        ValueError,
        'cluster_resolver and devices cannot be set at the same time'):
      _ = collective_all_reduce_strategy.CollectiveAllReduceExtended(
          container_strategy=None,
          cluster_resolver=multi_worker_test_base.create_in_process_cluster(
              num_workers=3, num_ps=0),
          communication_options=collective_util.Options(),
          devices=['GPU:0', 'GPU:1'])

  @combinations.generate(
      combinations.combine(
          mode=['graph', 'eager'],
          required_gpus=[0, 1, 2],
          use_devices_arg=[True, False]))
  def testMinimizeLoss(self, required_gpus, use_devices_arg):
    # [FUNCTION DOC]
    # Single-worker minimize-loss test parameterized by mode, GPU count and whether to pass an
    # explicit device list. In eager mode builds the strategy and runs the shared eager routine;
    # in graph mode runs the shared graph routine. Expects the training error to shrink.
    # Collective ops doesn't support strategy with one device.
    if context.executing_eagerly():
      strategy, _ = self._get_test_object(
          None, None, required_gpus, use_devices_arg=use_devices_arg)
      self._test_minimize_loss_eager(strategy)
    else:
      self._test_minimize_loss_graph(None, None, required_gpus)

  @combinations.generate(
      combinations.combine(
          mode=['eager'], required_gpus=[1, 2], use_devices_arg=[True, False]))
  def testNumReplicasInSync(self, required_gpus, use_devices_arg):
    # [FUNCTION DOC]
    # Eager test: with one or two GPUs (with or without an explicit device list), asserts the
    # strategy's replica count equals the GPU count.
    strategy, _ = self._get_test_object(
        None, None, required_gpus, use_devices_arg=use_devices_arg)
    self.assertEqual(required_gpus, strategy.num_replicas_in_sync)

  @combinations.generate(
      combinations.combine(
          mode=['eager'],
          required_tpus=[0, 1, 2],
          use_devices_arg=[True, False]))
  def testMinimizeLossTPU(self, required_tpus, use_devices_arg):
    # [FUNCTION DOC]
    # Eager test parameterized by TPU count (zero to two): builds the strategy for that many TPUs
    # and runs the shared eager minimize-loss routine.
    # KNOWN ISSUE: the explicit-device option has no effect here, since the helper only uses it
    # when the GPU count is positive and this test passes zero GPUs.
    strategy, _ = self._get_test_object(
        None, None, num_tpus=required_tpus, use_devices_arg=use_devices_arg)
    self._test_minimize_loss_eager(strategy)

  @combinations.generate(
      combinations.combine(
          mode=['graph', 'eager'],
          required_gpus=[0, 1, 2],
          use_devices_arg=[True, False]))
  def testCallAndMergeExceptions(self, required_gpus, use_devices_arg):
    # [FUNCTION DOC]
    # Test parameterized by mode, GPU count and device-list option: runs the shared routine that
    # checks exceptions raised inside replica calls and merge calls propagate correctly.
    strategy, _ = self._get_test_object(
        None, None, num_gpus=required_gpus, use_devices_arg=use_devices_arg)
    self._test_call_and_merge_exceptions(strategy)

  @combinations.generate(
      combinations.combine(
          mode=['graph'],
          required_gpus=2,
          use_dataset=[True, False],
          use_devices_arg=[True, False]))
  def testMakeInputFnIterator(self, required_gpus, use_dataset,
                              use_devices_arg):
    # [FUNCTION DOC]
    # Graph-mode single-worker test with two GPUs, parameterized by dataset-versus-callable input
    # function and device-list option. Builds an input function over a range of ten elements that
    # checks the input context (one pipeline, id zero, replicas in sync equal to GPU count) and
    # verifies the iterator yields consecutive pairs per step, ignoring order in the callable
    # case.
    # KNOWN ISSUE: the device-list option is never passed on to the iterator routine.
    if use_dataset:
      fn = lambda: dataset_ops.Dataset.range(5 * required_gpus)
    else:
      def fn():
        # [FUNCTION DOC]
        # HELPER: input function returning the get-next callable of a one-shot iterator over a
        # range dataset of five times the GPU count elements.
        dataset = dataset_ops.Dataset.range(5 * required_gpus)
        it = dataset_ops.make_one_shot_iterator(dataset)
        return it.get_next

    expected_values = [
        range(i, i + required_gpus) for i in range(0, 10, required_gpus)
    ]

    input_fn = self._input_fn_to_test_input_context(
        fn,
        expected_num_replicas_in_sync=required_gpus,
        expected_num_input_pipelines=1,
        expected_input_pipeline_id=0)
    self._test_input_fn_iterator(
        None,
        None,
        required_gpus,
        input_fn,
        expected_values,
        test_reinitialize=use_dataset,
        ignore_order=not use_dataset)

  @combinations.generate(
      combinations.combine(
          mode=['graph', 'eager'],
          required_gpus=[0, 1, 2],
          use_devices_arg=[True, False]))
  def testReduceToCpu(self, required_gpus, use_devices_arg):
    # [FUNCTION DOC]
    # Test parameterized by mode, GPU count and device-list option. Runs a function returning
    # each replica's id as a float, reduces it with a sum to the host and asserts the result is
    # the sum of all replica ids.
    strategy, _ = self._get_test_object(
        None, None, required_gpus, use_devices_arg=use_devices_arg)
    with strategy.scope():
      result = strategy.extended.call_for_each_replica(_replica_id_f32)
      reduced = strategy.reduce(reduce_util.ReduceOp.SUM, result, axis=None)
      expected = sum(range(strategy.num_replicas_in_sync))
      self.assertEqual(expected, self.evaluate(reduced))

  @combinations.generate(
      combinations.combine(
          mode=['graph'], required_gpus=2, use_devices_arg=[True, False]))
  def testAllReduceSum(self, required_gpus, use_devices_arg):
    # [FUNCTION DOC]
    # Graph-mode test with two GPUs: in a cached session on the strategy's target, runs the
    # shared all-reduce-sum routine.
    distribution, target = self._get_test_object(
        None, None, num_gpus=required_gpus, use_devices_arg=use_devices_arg)
    with self.cached_session(target=target):
      self._test_all_reduce_sum(distribution)

  @combinations.generate(
      combinations.combine(
          mode=['graph'], required_gpus=2, use_devices_arg=[True, False]))
  def testAllReduceSumGradients(self, required_gpus, use_devices_arg):
    # [FUNCTION DOC]
    # Graph-mode test with two GPUs: runs the shared routine checking gradients of an all-reduce
    # sum.
    distribution, target = self._get_test_object(
        None, None, num_gpus=required_gpus, use_devices_arg=use_devices_arg)
    with self.cached_session(target=target):
      self._test_all_reduce_sum_gradients(distribution)

  @combinations.generate(
      combinations.combine(
          mode=['graph'], required_gpus=2, use_devices_arg=[True, False]))
  def testAllReduceSumGradientTape(self, required_gpus, use_devices_arg):
    # [FUNCTION DOC]
    # Graph-mode test with two GPUs: runs the shared routine checking an all-reduce sum under a
    # gradient tape.
    distribution, target = self._get_test_object(
        None, None, num_gpus=required_gpus, use_devices_arg=use_devices_arg)
    with self.cached_session(target=target):
      self._test_all_reduce_sum_gradient_tape(distribution)

  @combinations.generate(
      combinations.combine(
          mode=['graph'], required_gpus=2, use_devices_arg=[True, False]))
  def testAllReduceMean(self, required_gpus, use_devices_arg):
    # [FUNCTION DOC]
    # Graph-mode test with two GPUs: runs the shared all-reduce-mean routine in a cached session.
    distribution, target = self._get_test_object(
        None, None, num_gpus=required_gpus, use_devices_arg=use_devices_arg)
    with self.cached_session(target=target):
      self._test_all_reduce_mean(distribution)

  @combinations.generate(
      combinations.combine(
          mode=['graph'], required_gpus=2, use_devices_arg=[True, False]))
  def testAllReduceMeanGradients(self, required_gpus, use_devices_arg):
    # [FUNCTION DOC]
    # Graph-mode test with two GPUs: runs the shared routine checking gradients of an all-reduce
    # mean.
    distribution, target = self._get_test_object(
        None, None, num_gpus=required_gpus, use_devices_arg=use_devices_arg)
    with self.cached_session(target=target):
      self._test_all_reduce_mean_gradients(distribution)

  @combinations.generate(
      combinations.combine(
          mode=['graph'], required_gpus=2, use_devices_arg=[True, False]))
  def testAllReduceMeanGradientTape(self, required_gpus, use_devices_arg):
    # [FUNCTION DOC]
    # Graph-mode test with two GPUs: runs the shared routine checking an all-reduce mean under a
    # gradient tape.
    distribution, target = self._get_test_object(
        None, None, num_gpus=required_gpus, use_devices_arg=use_devices_arg)
    with self.cached_session(target=target):
      self._test_all_reduce_mean_gradient_tape(distribution)

  @combinations.generate(
      combinations.combine(
          mode=['graph'], required_gpus=2, use_devices_arg=[True, False]))
  def testNumpyDataset(self, required_gpus, use_devices_arg):
    # [FUNCTION DOC]
    # Graph-mode test with two GPUs: runs the shared routine that builds a dataset from in-memory
    # arrays through the strategy, using a cached session on the strategy's target.
    strategy, target = self._get_test_object(
        None, None, num_gpus=required_gpus, use_devices_arg=use_devices_arg)
    self._test_numpy_dataset(
        strategy, session=self.cached_session(target=target))

  @combinations.generate(
      combinations.combine(
          mode=['eager'], required_gpus=2, use_devices_arg=[True, False]))
  def testReplicateDataset(self, required_gpus, use_devices_arg):
    # [FUNCTION DOC]
    # Eager test with two GPUs: builds a range dataset of ten elements through an input function
    # that checks the input context (one pipeline, id zero, two replicas) and verifies iterating
    # the distributed dataset yields consecutive pairs.
    strategy, _ = self._get_test_object(
        None, None, num_gpus=required_gpus, use_devices_arg=use_devices_arg)
    dataset_fn = lambda: dataset_ops.Dataset.range(10)
    expected_values = [[i, i + 1] for i in range(0, 10, 2)]
    input_fn = self._input_fn_to_test_input_context(
        dataset_fn,
        expected_num_replicas_in_sync=required_gpus,
        expected_num_input_pipelines=1,
        expected_input_pipeline_id=0)
    self._test_input_fn_iterable(strategy, input_fn, expected_values)

  @combinations.generate(
      combinations.combine(mode=['graph'], use_devices_arg=[True, False]))
  def testDeepCopy(self, use_devices_arg):
    # [FUNCTION DOC]
    # Graph-mode test: a single-worker strategy (with or without an explicit device list) can be
    # deep copied without error. Asserts nothing else.
    distribution, _ = self._get_test_object(
        None, None, use_devices_arg=use_devices_arg)
    copy.deepcopy(distribution)

  @combinations.generate(
      combinations.combine(
          mode=['graph', 'eager'],
          required_gpus=[0, 1, 2],
          use_devices_arg=[True, False]))
  def testSummaryForReplicaZeroOnly(self, required_gpus, use_devices_arg):
    # [FUNCTION DOC]
    # Test parameterized by mode, GPU count and device-list option: runs the shared routine
    # checking that summaries are recorded only for the first replica.
    strategy, target = self._get_test_object(
        None, None, num_gpus=required_gpus, use_devices_arg=use_devices_arg)
    with self.cached_session(target=target):
      self._test_summary_for_replica_zero_only(strategy)

  @combinations.generate(
      combinations.combine(
          mode=['graph', 'eager'],
          required_gpus=[0, 1, 2],
          use_devices_arg=[True, False]))
  def testTrainableVariables(self, required_gpus, use_devices_arg):
    # [FUNCTION DOC]
    # Test parameterized by mode, GPU count and device-list option: runs the shared routine
    # checking which variables are reported as trainable under the strategy.
    strategy, _ = self._get_test_object(
        None, None, num_gpus=required_gpus, use_devices_arg=use_devices_arg)
    self._test_trainable_variable(strategy)


class LogicalDeviceTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests that creating the strategy does not disturb logical device configuration of GPUs.

  @combinations.generate(combinations.combine(mode=['eager'], required_gpus=1))
  def testKeepLogicalDevice(self):
    # [FUNCTION DOC]
    # Eager test needing a GPU (skipped with several GPUs). Resets the runtime context, makes a
    # one-worker cluster resolver, splits each physical GPU into logical GPUs with a small memory
    # limit, then creates the strategy and asserts the number of logical GPUs is unchanged
    # (twice the physical count). Resets the context again afterwards.
    gpus = tf_config.list_physical_devices('GPU')
    if len(gpus) > 1:
      self.skipTest('Skip logical device test on multi GPUs, since partial GPU '
                    'virtualization is not permitted.')
    # Cannot change logical device after the context initialization.
    context._reset_context()  # pylint: disable=protected-access
    cluster_spec = multi_worker_test_base.create_cluster_spec(
        has_chief=False, num_workers=1)
    resolver = cluster_resolver_lib.SimpleClusterResolver(
        cluster_spec=multi_worker_util.normalize_cluster_spec(cluster_spec),
        task_type='worker',
        task_id=0)

    logical_gpus = len(gpus) * 2
    for i, device in enumerate(gpus):
      n = (i + 1) * logical_gpus // len(gpus) - i * logical_gpus // len(gpus)
      assert n > 0  # guaranteed if count >= len(devices)
      configs = []
      for ordinal in range(n):
        config = context.LogicalDeviceConfiguration(
            memory_limit=64,
            experimental_device_ordinal=ordinal)
        configs.append(config)

      tf_config.set_logical_device_configuration(device, configs)

    collective_all_reduce_strategy.CollectiveAllReduceStrategy(
        cluster_resolver=resolver)
    # Since we create two logical GPUs out of the last GPU, there should be one
    # more logical GPUs than physical GPUs.
    self.assertLen(tf_config.list_logical_devices('GPU'), logical_gpus)
    context._reset_context()  # pylint: disable=protected-access


@combinations.generate(
    combinations.combine(
        strategy=[
            strategy_combinations.multi_worker_mirrored_2x1_cpu,
            strategy_combinations.multi_worker_mirrored_2x1_gpu,
            strategy_combinations.multi_worker_mirrored_2x2_gpu,
            strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call,
        ],
        mode=['eager']))
class CollectiveAllReduceStrategyV2Test(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Eager tests run over several multi-worker mirrored strategy configurations (CPU and GPU, with
  # and without merge calls) supplied by the strategy combinations decorator.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: calls the parent setup and skips the test when TPUs are present.
    super().setUp()
    if context.context().list_physical_devices('TPU'):
      self.skipTest('Test not supported on TPUs')

  def test_replica_id_in_sync_group(self, strategy):
    # [FUNCTION DOC]
    # Runs a function on all replicas returning the replica id in the sync group and the
    # local replica id, gathers the results and asserts the first equals the range over all
    # replicas in sync and the second repeats the local device range once per worker.

    def replica_fn():
      # [FUNCTION DOC]
      # HELPER: replica function returning the current replica's id in the sync group and its
      # local replica id.
      replica_ctx = distribute_lib.get_replica_context()
      return replica_ctx.replica_id_in_sync_group, replica_ctx._replica_id

    results = test_util.gather(strategy, strategy.run(replica_fn))
    self.assertAllEqual(list(range(strategy.extended._num_replicas_in_sync)),
                        results[0].numpy())
    self.assertAllEqual(
        list(range(len(strategy.extended.worker_devices))) *
        strategy.extended._num_workers, results[1].numpy())

  def test_deep_copy_not_allowed(self, strategy):
    # [FUNCTION DOC]
    # Starts the health-check thread (disabled by default in tests) to mimic real use, then
    # asserts that deep copying the strategy raises a value error, both normally and inside a new
    # graph. Always stops the health-check thread afterwards.
    # Check health is disabled in tests by default. We need to enable it for
    # this test to simulate the real world.
    strategy.extended._start_check_health_thread()
    try:
      with self.assertRaisesRegex(ValueError, 'cannot be deep copied'):
        copy.deepcopy(strategy)
      with self.assertRaisesRegex(ValueError, 'cannot be deep copied'):
        with ops.Graph().as_default():
          copy.deepcopy(strategy)
    finally:
      strategy.extended._stop_check_health_thread()


class ExperimentalCompatibilityTest(test.TestCase):
  # [CLASS DOC]
  # Tests that the deprecated experimental strategy class and the main strategy class are
  # interchangeable for instance checks and share a name.

  def testIsInstance(self):
    # [FUNCTION DOC]
    # Creates one main strategy and one experimental strategy and asserts each is an instance of
    # both classes (people special-case the strategy type, so both endpoints must work).
    # It's not uncommon for people to special case MultiWorkerMirroredStrategy,
    # so we need to make sure isinstance check works for combinations between
    # the experimental and non-experimental endpoints.
    strategy = CollectiveAllReduceStrategy()
    experimental_strategy = _CollectiveAllReduceStrategyExperimental()
    self.assertIsInstance(strategy, CollectiveAllReduceStrategy)
    self.assertIsInstance(strategy, _CollectiveAllReduceStrategyExperimental)
    self.assertIsInstance(experimental_strategy, CollectiveAllReduceStrategy)
    self.assertIsInstance(experimental_strategy,
                          _CollectiveAllReduceStrategyExperimental)

  def testName(self):
    # [FUNCTION DOC]
    # Asserts both the main and the experimental class report the same class name.
    self.assertEqual(CollectiveAllReduceStrategy.__name__,
                     'CollectiveAllReduceStrategy')
    self.assertEqual(_CollectiveAllReduceStrategyExperimental.__name__,
                     'CollectiveAllReduceStrategy')


def _replica_id_f32():
  # [FUNCTION DOC]
  # HELPER: replica function returning the current replica's id in the sync group as a 32-bit
  # float; used by the reduce-to-host test.
  return math_ops.cast(
      distribute_lib.get_replica_context()
      .replica_id_in_sync_group, dtypes.float32)


if __name__ == '__main__':
  # TODO(b/172304955): enable logical devices.
  test_util.main(config_logical_devices=False)
