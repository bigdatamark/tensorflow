# [MODULE DOC]
# Test module for repeated collective operations across five worker processes launched with the
# multi-process runner: repeated all-reduce, and variable initialization with changing shapes
# broadcast from worker zero. Import-time side effects: disables the collective strategy's
# health-check feature globally for this process (it breaks the barrier the tests rely on),
# optionally registers a dill pickling hook so the test case can be sent to child processes, and
# defines the worker count constant. The main block uses the runner's test entry point.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for continuous runs using cross-worker collective ops."""

import json
import os

from absl.testing import parameterized
import numpy as np

from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base as test_base
from tensorflow.python.distribute import reduce_util
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import config
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope


try:
  import dill  # pylint:disable=g-import-not-at-top

  _REGISTER_DECORATOR = dill.register
except ImportError:
  _REGISTER_DECORATOR = lambda fn, *_: fn


# TODO(b/151232436): This test doesn't work with check health enabled because it
# relies on barrier around creating strategies. Check health performs
# communications inside strategy constructor, which makes the barrier
# ineffective.
CollectiveAllReduceExtended = (
    collective_all_reduce_strategy.CollectiveAllReduceExtended)
CollectiveAllReduceExtended._enable_check_health = False


NUM_WORKERS = 5


# TODO(b/143286947): expand the test to cover fault tolerance and elasticity
class MultiWorkerContinuousRunTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case exercising continuous collective-op runs on a cluster of five workers
  # started by the multi-process runner. Fixture: per-test GPU detection that picks a local device
  # (first GPU if any, else CPU) stored on the instance.

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the base setup, then runs the GPU setup helper in setup mode. Returns nothing.
    super(MultiWorkerContinuousRunTest, self).setUp()
    self._maybe_setup_gpus(setup=True)

  def _maybe_setup_gpus(self, setup=False):
    # [FUNCTION DOC]
    # HELPER: fixture helper for the tests. Input: setup flag (default false). Records the list of
    # physical GPUs and the local device name (first GPU if any, else CPU). When GPUs exist and the
    # flag is false (i.e. inside a worker process), configures the first GPU with one logical device
    # limited to 64 megabytes so several workers can share it. Returns nothing.
    self._gpus = config.list_physical_devices('GPU')
    self._local_device = '/device:GPU:0' if self._gpus else '/device:CPU:0'

    if self._gpus and not setup:
      # Set virtual GPU with memory limit of 64MB so that multiple worker
      # processes can share the physical GPU
      config.set_logical_device_configuration(
          self._gpus[0], [context.LogicalDeviceConfiguration(64)])

  @combinations.generate(combinations.combine(mode=['eager']))
  def testAllReduceContinuousRun(self, mode):
    # [FUNCTION DOC]
    # Scenario: five workers each create a collective all-reduce strategy, wait at the shared
    # barrier, and run a compiled function that mean-reduces a two-by-two tensor filled with the
    # worker index; repeated twenty times. Expected: every worker sees all elements equal to the mean
    # of worker indices (half of workers minus one). The run is skipped if an unavailable error
    # occurs. Mode is the (eager-only) parameter.
    tensor_shape = [2, 2]

    def worker_step_fn(worker_id):
      # [FUNCTION DOC]
      # HELPER: one iteration inside a worker. Input: worker index. Creates a strategy, waits at the
      # barrier, runs a mean reduction of a tensor filled with the index, and asserts the result
      # is close to the expected mean across all workers. Returns nothing.
      strategy = collective_all_reduce_strategy.CollectiveAllReduceStrategy()
      # Make sure the processeses are in sync after updating the cluster
      multi_process_runner.get_barrier().wait()

      @def_function.function
      def run_reduce():
        # [FUNCTION DOC]
        # HELPER: compiled function that, on the local device, mean-reduces a tensor of ones
        # scaled by the worker index across replicas and returns the reduced tensor.
        with ops.device(self._local_device):
          t_in = array_ops.ones(tensor_shape) * worker_id
          return strategy.reduce(reduce_util.ReduceOp.MEAN, t_in, axis=None)

      t_out = run_reduce()
      # Element values from the workers are
      #     0, 1, ..., (NUM_WORKERS - 1)
      expected_mean = (NUM_WORKERS - 1) / 2
      expected_out = np.ones(tensor_shape) * expected_mean
      self.assertAllClose(t_out, expected_out)

    def worker_fn():
      # [FUNCTION DOC]
      # HELPER: function run in each worker process. Sets up GPUs, reads the worker index from the
      # cluster-config environment variable, and runs the step function twenty times.
      self._maybe_setup_gpus()
      tf_config = json.loads(os.environ['TF_CONFIG'])
      worker_id = tf_config['task']['index']
      for _ in range(20):
        worker_step_fn(worker_id)

    with test_util.skip_if_error(self, errors_impl.UnavailableError):
      multi_process_runner.run(
          worker_fn,
          cluster_spec=test_base.create_cluster_spec(num_workers=NUM_WORKERS))

  @combinations.generate(combinations.combine(mode=['eager']))
  def testVariableInitializationWithChangingShape(self, mode):
    # [FUNCTION DOC]
    # Scenario: five workers repeatedly (twenty iterations, with growing tensor rank from one to
    # twenty) create a strategy, wait at the barrier, and create a variable whose initial value is
    # ones on worker zero and zeros elsewhere. Expected: the value broadcast from worker zero wins,
    # so every worker sees all ones. Skipped on an unavailable error. Mode is the eager-only
    # parameter.

    def worker_step_fn(worker_id, num_dims):
      # [FUNCTION DOC]
      # HELPER: one iteration inside a worker. Inputs: worker index and number of dimensions of the
      # tensor (each of size two). Creates a strategy, waits at the barrier, calls the variable
      # creation function on each replica, and asserts the result is all ones.
      strategy = collective_all_reduce_strategy.CollectiveAllReduceStrategy()
      # Make sure the processeses are in sync after updating the cluster
      multi_process_runner.get_barrier().wait()
      tensor_shape = [2] * num_dims

      def variable_fn():
        # [FUNCTION DOC]
        # HELPER: replica function. Creates (or gets) a variable named x on the local device whose
        # initial value is ones on worker zero and zeros on others, and returns an identity of it.
        with ops.device(self._local_device):
          # The initial value will be broadcasted from worker 0 to others.
          initial_value = (array_ops.ones(tensor_shape) if worker_id == 0 else
                           array_ops.zeros(tensor_shape))
          var = variable_scope.get_variable(name='x', initializer=initial_value)
          return array_ops.identity(var)

      t_out = strategy.extended.call_for_each_replica(variable_fn)
      expected_out = np.ones(tensor_shape)
      self.assertAllClose(t_out, expected_out)

    def worker_fn():
      # [FUNCTION DOC]
      # HELPER: function run in each worker process. Sets up GPUs, reads the worker index from the
      # cluster-config environment variable, and runs the step function twenty times with the
      # number of dimensions growing from one.
      self._maybe_setup_gpus()
      tf_config = json.loads(os.environ['TF_CONFIG'])
      worker_id = tf_config['task']['index']
      for i in range(20):
        worker_step_fn(worker_id, num_dims=(i + 1))

    with test_util.skip_if_error(self, errors_impl.UnavailableError):
      multi_process_runner.run(
          worker_fn,
          cluster_spec=test_base.create_cluster_spec(num_workers=NUM_WORKERS))


@_REGISTER_DECORATOR(MultiWorkerContinuousRunTest)
def _save_test_case(pickler, obj):
  # [FUNCTION DOC]
  # HELPER: custom dill pickling hook for the test case class, registered when dill is available.
  # Inputs: the pickler and the test case instance. Makes the instance pickle as a fresh instance
  # built by a reconstruct function, so test-case objects can be sent to child processes without
  # their state. Returns the pickler's save result.
  def reconstruct(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: factory used when unpickling; ignores all arguments and returns a new test case
    # instance.
    del args, kwargs
    return MultiWorkerContinuousRunTest()

  return pickler.save_reduce(reconstruct, (), obj=obj)


if __name__ == '__main__':
  multi_process_runner.test_main()
