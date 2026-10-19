# [MODULE DOC]
# Unit tests for the strategy combination descriptors in the strategy combinations module.
# They check that the two and four replica strategy lists really yield the expected replica
# counts, that the two-virtual-CPU mirrored descriptor works, and that with TF2 behavior disabled
# or enabled each descriptor family creates the matching V1 or V2 strategy class.
# Runs via the standard TensorFlow test main entry point; no import-time side effects.
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
"""Tests for a little bit of strategy_combinations."""

from absl.testing import parameterized

from tensorflow.python import tf2
from tensorflow.python.distribute import central_storage_strategy
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import one_device_strategy
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class StrategyCombinationsTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for replica counts of the multi-replica strategy lists and the two-CPU mirrored
  # descriptor. Parameterized by the combinations framework over strategies and graph/eager modes.

  @combinations.generate(
      combinations.combine(
          strategy=strategy_combinations.two_replica_strategies,
          mode=["graph", "eager"]))
  def testTwoReplicaStrategy(self, strategy):
    # [FUNCTION DOC]
    # For each two-replica strategy, runs a function returning one on every replica, sums across
    # replicas and asserts the total equals two.
    with strategy.scope():

      @def_function.function
      def one():
        # [FUNCTION DOC]
        # HELPER: compiled function returning a constant one as a float tensor, run on each replica.
        return array_ops.identity(1.)

      one_per_replica = strategy.run(one)
      num_replicas = strategy.reduce(
          reduce_util.ReduceOp.SUM, one_per_replica, axis=None)
      self.assertEqual(self.evaluate(num_replicas), 2.)

  @combinations.generate(
      combinations.combine(
          strategy=strategy_combinations.four_replica_strategies,
          mode=["graph", "eager"]))
  def testFourReplicaStrategy(self, strategy):
    # [FUNCTION DOC]
    # For each four-replica strategy, runs a function returning one on every replica, sums across
    # replicas and asserts the total equals four.
    with strategy.scope():

      @def_function.function
      def one():
        # [FUNCTION DOC]
        # HELPER: compiled function returning a constant one as a float tensor, run on each replica.
        return array_ops.identity(1.)

      one_per_replica = strategy.run(one)
      num_replicas = strategy.reduce(
          reduce_util.ReduceOp.SUM, one_per_replica, axis=None)
      self.assertEqual(self.evaluate(num_replicas), 4.)

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.mirrored_strategy_with_cpu_1_and_2
          ],
          mode=["graph", "eager"]))
  def testMirrored2CPUs(self, distribution):
    # [FUNCTION DOC]
    # Runs a constant-one function on each replica of the two-virtual-CPU mirrored strategy,
    # sums across replicas and asserts the result is two.
    with distribution.scope():
      one_per_replica = distribution.run(lambda: constant_op.constant(1))
      num_replicas = distribution.reduce(
          reduce_util.ReduceOp.SUM, one_per_replica, axis=None)
      self.assertEqual(2, self.evaluate(num_replicas))


class V1StrategyTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Verifies that, with TF2 behavior disabled, each descriptor family creates the V1 strategy
  # class (except multi-worker mirrored, which only has a V2 class).

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the parent setup and then disables TF2 behavior globally (side effect on global
    # state; it is not re-enabled here).
    super().setUp()
    tf2.disable()

  @combinations.generate(
      combinations.combine(strategy=[
          strategy_combinations.one_device_strategy,
          strategy_combinations.one_device_strategy_gpu,
          strategy_combinations.one_device_strategy_gpu_on_worker_1,
          strategy_combinations.one_device_strategy_on_worker_1
      ]))
  def testOneDevice(self, strategy):
    # [FUNCTION DOC]
    # Asserts each one-device descriptor yields a V1 one-device strategy.
    self.assertIsInstance(strategy, one_device_strategy.OneDeviceStrategyV1)

  @combinations.generate(
      combinations.combine(strategy=[
          strategy_combinations.mirrored_strategy_with_cpu_1_and_2,
          strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          strategy_combinations.mirrored_strategy_with_one_cpu,
          strategy_combinations.mirrored_strategy_with_one_gpu,
          strategy_combinations.mirrored_strategy_with_two_gpus,
      ]))
  def testMirrored(self, strategy):
    # [FUNCTION DOC]
    # Asserts each mirrored descriptor yields a V1 mirrored strategy.
    self.assertIsInstance(strategy, mirrored_strategy.MirroredStrategyV1)

  @combinations.generate(
      combinations.combine(strategy=[
          strategy_combinations.multi_worker_mirrored_2x1_cpu,
          strategy_combinations.multi_worker_mirrored_2x1_gpu,
          strategy_combinations.multi_worker_mirrored_2x2_gpu,
          strategy_combinations.multi_worker_mirrored_4x1_cpu,
      ]))
  def testMultiWorkerMirrored(self, strategy):
    # [FUNCTION DOC]
    # Asserts each multi-worker mirrored descriptor yields the collective all-reduce strategy
    # (the only supported version) even with TF2 disabled.
    # MultiWorkerMirroredStrategy combinations only supports V2.
    self.assertIsInstance(
        strategy, collective_all_reduce_strategy.CollectiveAllReduceStrategy)

  @combinations.generate(
      combinations.combine(strategy=[
          strategy_combinations.central_storage_strategy_with_gpu_and_cpu,
          strategy_combinations.central_storage_strategy_with_two_gpus,
      ]))
  def testCentralStorage(self, strategy):
    # [FUNCTION DOC]
    # Asserts each central storage descriptor yields a V1 central storage strategy.
    self.assertIsInstance(strategy,
                          central_storage_strategy.CentralStorageStrategyV1)

  @combinations.generate(
      combinations.combine(strategy=strategy_combinations.tpu_strategies))
  def testTPU(self, strategy):
    # [FUNCTION DOC]
    # Asserts each TPU descriptor in the TPU strategy list yields a V1 TPU strategy.
    self.assertIsInstance(strategy, tpu_strategy.TPUStrategyV1)


class V2StrategyTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Verifies that, with TF2 behavior enabled, each descriptor family creates the V2 strategy
  # class, including parameter server strategies.

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the parent setup and then enables TF2 behavior globally (side effect on global state).
    super().setUp()
    tf2.enable()

  @combinations.generate(
      combinations.combine(strategy=[
          strategy_combinations.one_device_strategy,
          strategy_combinations.one_device_strategy_gpu,
          strategy_combinations.one_device_strategy_gpu_on_worker_1,
          strategy_combinations.one_device_strategy_on_worker_1
      ]))
  def testOneDevice(self, strategy):
    # [FUNCTION DOC]
    # Asserts each one-device descriptor yields a V2 one-device strategy.
    self.assertIsInstance(strategy, one_device_strategy.OneDeviceStrategy)

  @combinations.generate(
      combinations.combine(strategy=[
          strategy_combinations.mirrored_strategy_with_cpu_1_and_2,
          strategy_combinations.mirrored_strategy_with_gpu_and_cpu,
          strategy_combinations.mirrored_strategy_with_one_cpu,
          strategy_combinations.mirrored_strategy_with_one_gpu,
          strategy_combinations.mirrored_strategy_with_two_gpus,
      ]))
  def testMirrored(self, strategy):
    # [FUNCTION DOC]
    # Asserts each mirrored descriptor yields a V2 mirrored strategy.
    self.assertIsInstance(strategy, mirrored_strategy.MirroredStrategy)

  @combinations.generate(
      combinations.combine(strategy=[
          strategy_combinations.multi_worker_mirrored_2x1_cpu,
          strategy_combinations.multi_worker_mirrored_2x1_gpu,
          strategy_combinations.multi_worker_mirrored_2x2_gpu,
          strategy_combinations.multi_worker_mirrored_4x1_cpu,
      ]))
  def testMultiWorkerMirrored(self, strategy):
    # [FUNCTION DOC]
    # Asserts each multi-worker mirrored descriptor yields the collective all-reduce strategy.
    self.assertIsInstance(
        strategy, collective_all_reduce_strategy.CollectiveAllReduceStrategy)

  @combinations.generate(
      combinations.combine(strategy=[
          strategy_combinations.central_storage_strategy_with_gpu_and_cpu,
          strategy_combinations.central_storage_strategy_with_two_gpus,
      ]))
  def testCentralStorage(self, strategy):
    # [FUNCTION DOC]
    # Asserts each central storage descriptor yields a V2 central storage strategy.
    self.assertIsInstance(strategy,
                          central_storage_strategy.CentralStorageStrategy)

  @combinations.generate(
      combinations.combine(strategy=strategy_combinations.tpu_strategies))
  def testTPU(self, strategy):
    # [FUNCTION DOC]
    # Asserts each TPU descriptor yields either the TPU strategy or the V2 TPU strategy class.
    self.assertIsInstance(
        strategy, (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV2))

  @combinations.generate(
      combinations.combine(strategy=[
          strategy_combinations.parameter_server_strategy_3worker_2ps_cpu,
          strategy_combinations.parameter_server_strategy_1worker_2ps_cpu,
          strategy_combinations.parameter_server_strategy_3worker_2ps_1gpu,
          strategy_combinations.parameter_server_strategy_1worker_2ps_1gpu,
      ]))
  def testParameterServer(self, strategy):
    # [FUNCTION DOC]
    # Asserts each parameter server descriptor yields a V2 parameter server strategy.
    self.assertIsInstance(
        strategy, parameter_server_strategy_v2.ParameterServerStrategyV2)


if __name__ == "__main__":
  test_util.main()
