# [MODULE DOC]
# Tests for the private dataset replicate transformation, which copies a dataset onto a list of
# other devices and returns a mapping from device name to the replicated dataset.
# Three test cases are defined: local replication across virtual CPU devices in one process,
# eager replication across a multi-process cluster (set through the eager context server
# definition), and graph-mode replication across a local cluster driven through a session.
# Helper function builds a cluster server definition. No import-time side effects besides imports;
# the test cases reset the global eager context in teardown.
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
"""Tests for the private `replicate()` transformation."""
from absl.testing import parameterized

from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import combinations
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib


class LocalReplicateTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests replicating datasets between logical CPU devices of the local process. The constructor
  # splits the first physical CPU into three logical devices and stores their names as device0,
  # device1 and device2; tests build a dataset on device0, replicate it to devices 1 and 2 and
  # check that every copy produces the same elements.

  def __init__(self, methodName="runTest"):  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # Constructor. Takes the test method name (default is the standard run-test name). Configures
    # three logical devices on the first physical CPU (global device configuration side effect;
    # fails if the context was already initialized differently) and records the three device names.
    super(LocalReplicateTest, self).__init__(methodName)
    cpus = config.list_physical_devices("CPU")
    # Set 3 virtual CPUs
    config.set_logical_device_configuration(cpus[0], [
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration()
    ])
    self._device0 = "/device:CPU:0"
    self._device1 = "/device:CPU:1"
    self._device2 = "/device:CPU:2"

  def tearDown(self):
    # [FUNCTION DOC]
    # Test cleanup: calls the base teardown, clears the current device scope by entering a null
    # device scope, and resets the global eager context so later tests are unaffected.
    super().tearDown()
    # Clear the current device scope to avoid polluting other test cases.
    ops.device(None).__enter__()
    # Reset the context to avoid polluting other test cases.
    context._reset_context()

  @combinations.generate(test_base.default_test_combinations())
  def testBasic(self):
    # [FUNCTION DOC]
    # Builds a range dataset of 100 on device0, replicates it to devices 1 and 2, and asserts that
    # the original and both replicas each produce the numbers 0 to 99 when iterated on their device.
    with ops.device(self._device0):
      dataset0 = dataset_ops.Dataset.range(100)
    replicated_ds = distribute.replicate(dataset0,
                                         [self._device1, self._device2])
    dataset1 = replicated_ds[self._device1]
    dataset2 = replicated_ds[self._device2]

    with ops.device(self._device0):
      self.assertDatasetProduces(dataset0, range(100))
    with ops.device(self._device1):
      self.assertDatasetProduces(dataset1, range(100))
    with ops.device(self._device2):
      self.assertDatasetProduces(dataset2, range(100))

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorsWithDataset(self):
    # [FUNCTION DOC]
    # Wraps a range dataset as a single element via from-tensors, flattens it back with flat-map,
    # replicates to devices 1 and 2, and asserts all three datasets yield 0 to 99.
    with ops.device(self._device0):
      dataset0 = dataset_ops.Dataset.range(100)
      dataset0 = dataset_ops.Dataset.from_tensors(dataset0)
      dataset0 = dataset0.flat_map(lambda x: x)
    replicated_ds = distribute.replicate(dataset0,
                                         [self._device1, self._device2])
    dataset1 = replicated_ds[self._device1]
    dataset2 = replicated_ds[self._device2]

    with ops.device(self._device0):
      self.assertDatasetProduces(dataset0, range(100))
    with ops.device(self._device1):
      self.assertDatasetProduces(dataset1, range(100))
    with ops.device(self._device2):
      self.assertDatasetProduces(dataset2, range(100))

  @combinations.generate(test_base.default_test_combinations())
  def testFromTensorSlicesWithDataset(self):
    # [FUNCTION DOC]
    # Same as the from-tensors variant but nests the dataset using from-tensor-slices of a one
    # element list before flat-mapping; replicas on devices 1 and 2 must yield 0 to 99.
    with ops.device(self._device0):
      dataset0 = dataset_ops.Dataset.range(100)
      dataset0 = dataset_ops.Dataset.from_tensor_slices([dataset0])
      dataset0 = dataset0.flat_map(lambda x: x)
    replicated_ds = distribute.replicate(dataset0,
                                         [self._device1, self._device2])
    dataset1 = replicated_ds[self._device1]
    dataset2 = replicated_ds[self._device2]

    with ops.device(self._device0):
      self.assertDatasetProduces(dataset0, range(100))
    with ops.device(self._device1):
      self.assertDatasetProduces(dataset1, range(100))
    with ops.device(self._device2):
      self.assertDatasetProduces(dataset2, range(100))

  @combinations.generate(test_base.default_test_combinations())
  def testVariableInput(self):
    # [FUNCTION DOC]
    # Maps a range dataset through an increment of a resource counter variable on device0,
    # replicates it, initializes the variable, and asserts each replica yields 1 to 100 (each
    # starting from the shared variable state). The original is iterated last because in graph
    # mode replication must happen before the variable is modified.
    with ops.device(self._device0):
      counter_var = variable_scope.get_variable(
          "counter", (), dtypes.int32, use_resource=True)
      dataset0 = dataset_ops.Dataset.range(100).map(
          lambda _: counter_var.assign_add(1))
    replicated_ds = distribute.replicate(dataset0,
                                         [self._device1, self._device2])
    dataset1 = replicated_ds[self._device1]
    dataset2 = replicated_ds[self._device2]
    self.evaluate(counter_var.initializer)
    with ops.device(self._device1):
      self.assertDatasetProduces(
          dataset1, range(1, 101), requires_initialization=True)
    with ops.device(self._device2):
      self.assertDatasetProduces(
          dataset2, range(1, 101), requires_initialization=True)
    # Iterate through the original device last so that replication happens
    # before counter_var is modified. The order only matters in graph mode.
    with ops.device(self._device0):
      self.assertDatasetProduces(
          dataset0, range(1, 101), requires_initialization=True)

  @combinations.generate(test_base.default_test_combinations())
  def testExternalStatePolicyIgnore(self):
    # [FUNCTION DOC]
    # Builds a dataset whose map uses a random op (external state) with the external state policy
    # option set to ignore, replicates it, and checks that 100 elements can be drawn from the
    # original and both replicas without error.
    with ops.device(self._device0):
      dataset0 = dataset_ops.Dataset.range(100).map(
          lambda _: random_ops.random_uniform(  # pylint:disable=g-long-lambda
              [],
              minval=1,
              maxval=10,
              dtype=dtypes.float32))
      opt = options_lib.Options()
      opt.experimental_external_state_policy = (
          options_lib.ExternalStatePolicy.IGNORE)
      dataset0 = dataset0.with_options(opt)
    replicated_ds = distribute.replicate(dataset0,
                                         [self._device1, self._device2])
    dataset1 = replicated_ds[self._device1]
    dataset2 = replicated_ds[self._device2]

    with ops.device(self._device0):
      get_next0 = self.getNext(dataset0)
    with ops.device(self._device1):
      get_next1 = self.getNext(dataset1)
    with ops.device(self._device2):
      get_next2 = self.getNext(dataset2)

    for _ in range(100):
      self.evaluate(get_next0())
      self.evaluate(get_next1())
      self.evaluate(get_next2())

  @combinations.generate(test_base.default_test_combinations())
  def testExternalStatePolicyWarn(self):
    # [FUNCTION DOC]
    # Same scenario as the ignore variant but with the policy set to warn: replication succeeds
    # (a warning is logged) and 100 elements are drawn from each of the three datasets.
    with ops.device(self._device0):
      dataset0 = dataset_ops.Dataset.range(100).map(
          lambda _: random_ops.random_uniform(  # pylint:disable=g-long-lambda
              [],
              minval=1,
              maxval=10,
              dtype=dtypes.float32))
      opt = options_lib.Options()
      opt.experimental_external_state_policy = (
          options_lib.ExternalStatePolicy.WARN)
      dataset0 = dataset0.with_options(opt)
    replicated_ds = distribute.replicate(dataset0,
                                         [self._device1, self._device2])
    dataset1 = replicated_ds[self._device1]
    dataset2 = replicated_ds[self._device2]

    with ops.device(self._device0):
      get_next0 = self.getNext(dataset0)
    with ops.device(self._device1):
      get_next1 = self.getNext(dataset1)
    with ops.device(self._device2):
      get_next2 = self.getNext(dataset2)

    for _ in range(100):
      self.evaluate(get_next0())
      self.evaluate(get_next1())
      self.evaluate(get_next2())

  @combinations.generate(test_base.default_test_combinations())
  def testExternalStatePolicyFail(self):
    # [FUNCTION DOC]
    # Same scenario with the policy set to fail: asserts that replicating (or iterating) the
    # stateful dataset raises a failed-precondition error.
    with ops.device(self._device0):
      dataset0 = dataset_ops.Dataset.range(100).map(
          lambda _: random_ops.random_uniform(  # pylint:disable=g-long-lambda
              [],
              minval=1,
              maxval=10,
              dtype=dtypes.float32))
      opt = options_lib.Options()
      opt.experimental_external_state_policy = (
          options_lib.ExternalStatePolicy.FAIL)
      dataset0 = dataset0.with_options(opt)
    with self.assertRaises(errors.FailedPreconditionError):
      replicated_ds = distribute.replicate(dataset0,
                                           [self._device1, self._device2])
      dataset1 = replicated_ds[self._device1]
      dataset2 = replicated_ds[self._device2]

      with ops.device(self._device0):
        get_next0 = self.getNext(dataset0)
      with ops.device(self._device1):
        get_next1 = self.getNext(dataset1)
      with ops.device(self._device2):
        get_next2 = self.getNext(dataset2)

      for _ in range(100):
        self.evaluate(get_next0())
        self.evaluate(get_next1())
        self.evaluate(get_next2())


def _get_server_def(job_name, local_server_port, remote_server_addresses,
                    task_index):
  # [FUNCTION DOC]
  # HELPER: builds the cluster server definition used by the eager cluster replicate tests.
  # Inputs: job name, port of the local server (task 0), list of remote server addresses (tasks
  # 1..n), and the task index of this process. Returns a server definition message for a single job
  # using the grpc protocol, where task 0 is localhost on the given port.
  """Returns a server def with a single job + multiple tasks."""
  cluster_def = cluster_pb2.ClusterDef()
  job_def = cluster_def.job.add()
  job_def.name = job_name
  job_def.tasks[0] = "localhost:%d" % local_server_port

  for i, remote_server_address in enumerate(remote_server_addresses, start=1):
    job_def.tasks[i] = remote_server_address

  server_def = tensorflow_server_pb2.ServerDef(
      cluster=cluster_def,
      job_name=job_name,
      task_index=task_index,
      protocol="grpc")

  return server_def


class EagerClusterReplicateTest(test_base.DatasetTestBase,
                                parameterized.TestCase):
  # [CLASS DOC]
  # Tests replicating datasets across tasks of an in-process eager cluster (one local context plus
  # two local gRPC servers). Device names refer to tasks 0, 1 and 2 of one job. Eager mode only.

  def __init__(self, methodName="runTest"):  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # Constructor taking the test method name; records the job name and the three task device names.
    super(EagerClusterReplicateTest, self).__init__(methodName)
    self._job_name = "remove_device"
    self._device0 = "/job:%s/replica:0/task:0/device:CPU:0" % self._job_name
    self._device1 = "/job:%s/replica:0/task:1/device:CPU:0" % self._job_name
    self._device2 = "/job:%s/replica:0/task:2/device:CPU:0" % self._job_name

  def setUp(self):
    # [FUNCTION DOC]
    # Starts two local servers, stores their addresses (without the protocol prefix), picks an
    # unused port and sets the eager context server definition so task 0 is this process and the
    # two servers are tasks 1 and 2. Side effect: changes global eager context state.
    super(EagerClusterReplicateTest, self).setUp()

    # TODO(b/171412104): Move create server to __init__ once tfrt support it.
    self._cached_server1 = server_lib.Server.create_local_server()
    self._cached_server2 = server_lib.Server.create_local_server()
    self._cached_server1_target = self._cached_server1.target[len("grpc://"):]
    self._cached_server2_target = self._cached_server2.target[len("grpc://"):]

    # Start the local server.
    local_port = pywrap_tfe.TF_PickUnusedPortOrDie()
    context.set_server_def(
        server_def=_get_server_def(
            self._job_name,
            local_server_port=local_port,
            remote_server_addresses=[
                self._cached_server1_target, self._cached_server2_target
            ],
            task_index=0))

  def tearDown(self):
    # [FUNCTION DOC]
    # Test cleanup: base teardown, clear the device scope, and reset the global eager context.
    super().tearDown()
    # Clear the current device scope to avoid polluting other test cases.
    ops.device(None).__enter__()
    # Reset the context to avoid polluting other test cases.
    context._reset_context()

  @combinations.generate(
      combinations.combine(tf_api_version=[2], mode=["eager"]))
  def testBasic(self):
    # [FUNCTION DOC]
    # Replicates a range dataset of 100 from task 0 to tasks 1 and 2 and asserts all three produce
    # 0 to 99 on their respective devices.
    with ops.device(self._device0):
      dataset0 = dataset_ops.Dataset.range(100)
    replicated_ds = distribute.replicate(dataset0,
                                         [self._device1, self._device2])
    dataset1 = replicated_ds[self._device1]
    dataset2 = replicated_ds[self._device2]
    with ops.device(self._device0):
      self.assertDatasetProduces(dataset0, range(100))
    with ops.device(self._device1):
      self.assertDatasetProduces(dataset1, range(100))
    with ops.device(self._device2):
      self.assertDatasetProduces(dataset2, range(100))

  @combinations.generate(
      combinations.combine(tf_api_version=[2], mode=["eager"]))
  def testMap(self):
    # [FUNCTION DOC]
    # Replicates a range dataset mapped by doubling and asserts all three yield the even numbers
    # 0 to 198.
    with ops.device(self._device0):
      dataset0 = dataset_ops.Dataset.range(100).map(lambda x: x * 2)
    replicated_ds = distribute.replicate(dataset0,
                                         [self._device1, self._device2])
    dataset1 = replicated_ds[self._device1]
    dataset2 = replicated_ds[self._device2]
    with ops.device(self._device0):
      self.assertDatasetProduces(dataset0, range(0, 200, 2))
    with ops.device(self._device1):
      self.assertDatasetProduces(dataset1, range(0, 200, 2))
    with ops.device(self._device2):
      self.assertDatasetProduces(dataset2, range(0, 200, 2))

  @combinations.generate(
      combinations.combine(tf_api_version=[2], mode=["eager"]))
  def testVariableInput(self):
    # [FUNCTION DOC]
    # Replicates a dataset whose map increments a resource counter variable living on task 0 and
    # asserts each of the three datasets yields 1 to 100 (the original is iterated first here, and
    # each replica still starts from the counter's initial value).
    with ops.device(self._device0):
      counter_var = variable_scope.get_variable(
          "counter", (), dtypes.int32, use_resource=True)
      dataset0 = dataset_ops.Dataset.range(100).map(
          lambda _: counter_var.assign_add(1))
    replicated_ds = distribute.replicate(dataset0,
                                         [self._device1, self._device2])
    dataset1 = replicated_ds[self._device1]
    dataset2 = replicated_ds[self._device2]
    with ops.device(self._device0):
      self.assertDatasetProduces(
          dataset0, range(1, 101), requires_initialization=True)
    with ops.device(self._device1):
      self.assertDatasetProduces(
          dataset1, range(1, 101), requires_initialization=True)
    with ops.device(self._device2):
      self.assertDatasetProduces(
          dataset2, range(1, 101), requires_initialization=True)


class GraphClusterReplicateTest(test_base.DatasetTestBase,
                                parameterized.TestCase):
  # [CLASS DOC]
  # Tests dataset replication across tasks of a local multi-process cluster in graph mode, using a
  # session connected to the first worker. Graph mode with the version 1 API only.

  def setUp(self):
    # [FUNCTION DOC]
    # Creates a local cluster of three workers (two CPU devices each), stores the device names of
    # the three tasks and the target address of the first worker for sessions.
    super(GraphClusterReplicateTest, self).setUp()
    # Start the local server.
    worker_config = config_pb2.ConfigProto()
    worker_config.device_count["CPU"] = 2
    worker, _ = test_util.create_local_cluster(
        3, 0, worker_config=worker_config)
    self._device0 = "/job:worker/replica:0/task:0/device:CPU:0"
    self._device1 = "/job:worker/replica:0/task:1/device:CPU:0"
    self._device2 = "/job:worker/replica:0/task:2/device:CPU:0"
    self._target = worker[0].target

  def tearDown(self):
    # [FUNCTION DOC]
    # Test cleanup: base teardown, clear the device scope, and reset the global eager context.
    super().tearDown()
    # Clear the current device scope to avoid polluting other test cases.
    ops.device(None).__enter__()
    # Reset the context to avoid polluting other test cases.
    context._reset_context()

  @combinations.generate(
      combinations.combine(tf_api_version=[1], mode=["graph"]))
  def testBasic(self):
    # [FUNCTION DOC]
    # Replicates a range dataset of 100 to tasks 1 and 2, then in a session on the cluster target
    # asserts that for each step all three get-next ops return the same value as the step index.
    with ops.device(self._device0):
      dataset0 = dataset_ops.Dataset.range(100)
    replicated_ds = distribute.replicate(dataset0,
                                         [self._device1, self._device2])
    dataset1 = replicated_ds[self._device1]
    dataset2 = replicated_ds[self._device2]
    with ops.device(self._device0):
      get_next = self.getNext(dataset0)
    with ops.device(self._device1):
      get_next1 = self.getNext(dataset1)
    with ops.device(self._device2):
      get_next2 = self.getNext(dataset2)

    with session.Session(self._target) as sess:
      for i in range(100):
        self.assertEqual(i, sess.run(get_next()))
        self.assertEqual(i, sess.run(get_next1()))
        self.assertEqual(i, sess.run(get_next2()))

  @combinations.generate(
      combinations.combine(tf_api_version=[1], mode=["graph"]))
  def testMap(self):
    # [FUNCTION DOC]
    # Same as the basic graph test but with a doubling map: each step all three iterators must
    # return twice the step index.
    with ops.device(self._device0):
      dataset0 = dataset_ops.Dataset.range(100).map(lambda x: x * 2)
    replicated_ds = distribute.replicate(dataset0,
                                         [self._device1, self._device2])
    dataset1 = replicated_ds[self._device1]
    dataset2 = replicated_ds[self._device2]
    with ops.device(self._device0):
      get_next = self.getNext(dataset0)
    with ops.device(self._device1):
      get_next1 = self.getNext(dataset1)
    with ops.device(self._device2):
      get_next2 = self.getNext(dataset2)

    with session.Session(self._target) as sess:
      for i in range(100):
        self.assertEqual(i * 2, sess.run(get_next()))
        self.assertEqual(i * 2, sess.run(get_next1()))
        self.assertEqual(i * 2, sess.run(get_next2()))

  @combinations.generate(
      combinations.combine(tf_api_version=[1], mode=["graph"]))
  def testVariableInput(self):
    # [FUNCTION DOC]
    # Replicates a dataset whose map modifies a resource variable on task 0 to task 1, creates an
    # initializable iterator there, and asserts that running its initializer in a session raises an
    # op error, since stateful ops across processes in functions are unsupported.
    with ops.device(self._device0):
      counter_var = variable_scope.get_variable(
          "counter", (), dtypes.int32, use_resource=True)
      dataset0 = dataset_ops.Dataset.range(100).map(
          lambda _: counter_var.assign_add(1))
    replicated_ds = distribute.replicate(dataset0,
                                         [self._device1, self._device2])
    dataset1 = replicated_ds[self._device1]
    with ops.device(self._device1):
      it1 = dataset_ops.make_initializable_iterator(dataset1)
    # We don't support stateful ops across processes in functions as of now.
    with session.Session(self._target) as sess:
      with self.assertRaises(errors.OpError):
        sess.run(it1.initializer)


if __name__ == "__main__":
  test.main()
