# [MODULE DOC]
# Tests MirroredStrategy on a remote single worker in eager mode, reusing shared test bodies
# from the strategy test library. Requires at least one GPU on the remote worker.
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
"""Tests for MirroredStrategy."""

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import test


def get_gpus():
  # [FUNCTION DOC]
  # HELPER: lists the names of logical GPU devices that belong to a job (remote devices) in the
  # current eager context. Returns a list of device name strings; empty if none.
  gpus = context.context().list_logical_devices("GPU")
  actual_gpus = []
  for gpu in gpus:
    if "job" in gpu.name:
      actual_gpus.append(gpu.name)
  return actual_gpus


@combinations.generate(
    combinations.combine(
        distribution=[
            combinations.NamedDistribution(
                "Mirrored",
                # pylint: disable=g-long-lambda
                lambda: mirrored_strategy.MirroredStrategy(get_gpus()),
                required_gpus=1)
        ],
        mode=["eager"]))
class RemoteSingleWorkerMirroredStrategyEager(
    multi_worker_test_base.SingleWorkerTestBaseEager,
    strategy_test_lib.RemoteSingleWorkerMirroredStrategyBase):
  # [CLASS DOC]
  # Eager-mode tests of mirrored strategy over a remote single worker, mixing the single worker
  # eager test base with the shared remote mirrored strategy test bodies. Each test receives the
  # strategy built over the remote GPUs.

  def _get_num_gpus(self):
    # [FUNCTION DOC]
    # HELPER: returns the number of remote GPUs, used by the shared test base.
    return len(get_gpus())

  def testNumReplicasInSync(self, distribution):
    # [FUNCTION DOC]
    # Runs the shared check that the strategy reports the expected number of replicas in sync.
    self._testNumReplicasInSync(distribution)

  def testMinimizeLoss(self, distribution):
    # [FUNCTION DOC]
    # Runs the shared check that minimizing a loss works with the strategy.
    self._testMinimizeLoss(distribution)

  def testDeviceScope(self, distribution):
    # [FUNCTION DOC]
    # Runs the shared check of device scoping behavior with the strategy.
    self._testDeviceScope(distribution)

  def testMakeInputFnIteratorWithDataset(self, distribution):
    # [FUNCTION DOC]
    # Runs the shared check of building an input iterator from an input function returning a dataset.
    self._testMakeInputFnIteratorWithDataset(distribution)

  def testMakeInputFnIteratorWithCallable(self, distribution):
    # [FUNCTION DOC]
    # Runs the shared check of building an input iterator from an input function returning a callable.
    self._testMakeInputFnIteratorWithCallable(distribution)


if __name__ == "__main__":
  test.main()
