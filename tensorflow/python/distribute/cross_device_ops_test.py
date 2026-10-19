# [MODULE DOC]
# Multi-process tests for the cross-device reduction implementations in cross_device_ops
# (collective all-reduce, plus the NCCL, ring and auto communication implementations), and for
# the collective replica launcher in cross_device_utils. Defines small helpers (building
# per-replica values, enabling collective ops in a worker process, shared pools of worker
# processes) and the test classes. At import time it tries to import an optional pickling
# library to register picklers, creates two global pools of worker processes (one and two
# processes) so expensive process start-up is shared, and defines module aliases.
# Tests need multiple processes, and some need GPUs; many are skipped without enough hardware.
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
"""Tests for CrossDeviceOps."""

import collections
import os
import threading
import time

from absl.testing import parameterized

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.distribute import cluster_resolver as cluster_resolver_lib
from tensorflow.python.distribute import collective_util
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import cross_device_ops as cross_device_ops_lib
from tensorflow.python.distribute import cross_device_utils
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute import values as value_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest


try:
  import dill  # pylint:disable=g-import-not-at-top

  _REGISTER_DECORATOR = dill.register
except ImportError:
  _REGISTER_DECORATOR = lambda fn, *_: fn

CollectiveReplicaLauncher = cross_device_utils.CollectiveReplicaLauncher
CommunicationImplementation = collective_util.CommunicationImplementation
ReduceOp = reduce_util.ReduceOp
IndexedSlicesValue = indexed_slices.IndexedSlicesValue
IndexedSlices = indexed_slices.IndexedSlices


def make_per_replica_value(value, devices):
  # [FUNCTION DOC]
  # HELPER: small shared test utility that builds inputs for reduction tests.
  # Builds a per-replica values object with one component per device. value may be a callable
  # taking the device index and returning that device's value, a list indexed by device index,
  # or a single value used for all devices. Each component is copied with an identity op placed
  # on its device; indexed-slices values (the plain value form) are converted to indexed
  # slices whose values, indices and dense shape are each copied on the device. devices is a
  # list of device strings. Returns the per-replica values object.
  """Creates a `PerReplica` object whose values reside in `devices`.

  Args:
    value: a tensor-convertible value or a `IndexedSlicesValue`, or a callable
      that takes one argument (`device_idx`) and should return the value that is
      going to be created on devices[device_idx].
    devices: a list of device strings to create `PerReplica` values on.

  Returns:
    A `PerReplica` object.
  """
  values = []
  for device_idx, device in enumerate(devices):
    if callable(value):
      v = value(device_idx)
    elif isinstance(value, list):
      v = value[device_idx]
    else:
      v = value
    if isinstance(v, IndexedSlicesValue):
      with ops.device(device):
        values.append(
            IndexedSlices(
                values=array_ops.identity(v.values),
                indices=array_ops.identity(v.indices),
                dense_shape=array_ops.identity(v.dense_shape),
            )
        )
    else:
      with ops.device(device):
        values.append(array_ops.identity(v))
  return value_lib.PerReplica(values)


def enable_collective_ops():
  # [FUNCTION DOC]
  # HELPER: setup function run inside each worker process of the pools.
  # Enables collective ops in the current process: resolves the cluster from the environment
  # configuration, configures the first worker as collective leader, builds a server definition
  # (cluster, job name, task index, protocol from the resolver) and enables collectives with it
  # on the global context. Finally resets the launcher class switches to their default test
  # values (unique instance keys on, ordering token off). Takes no arguments, returns nothing.
  # Side effects: mutates the global eager context and class-level launcher flags.
  """Enable collectives in the current process."""
  cluster_resolver = cluster_resolver_lib.TFConfigClusterResolver()
  context.context().configure_collective_ops(
      collective_leader="'/job:worker/replica:0/task:0'"
  )
  config_proto = config_pb2.ConfigProto()
  config_proto.experimental.collective_group_leader = (
      "/job:worker/replica:0/task:0"
  )
  server_def = tensorflow_server_pb2.ServerDef(
      cluster=cluster_resolver.cluster_spec().as_cluster_def(),
      default_session_config=config_proto,
      job_name=cluster_resolver.task_type,
      task_index=cluster_resolver.task_id,
      protocol=cluster_resolver.rpc_layer,
  )
  context.context().enable_collective_ops(server_def)
  # Recover default flag values.
  CollectiveReplicaLauncher._prefer_unique_instance_key = True
  CollectiveReplicaLauncher._prefer_ordering_token = False


class MultiProcessPoolRunner:
  # [CLASS DOC]
  # HELPER: thin test wrapper that owns a pool of worker processes.
  # Builds a cluster description with the requested number of workers and stores a pool runner
  # for it in the attribute named runner. Used only to create the module-level shared pools.

  def __init__(self, num_processes):
    # [FUNCTION DOC]
    # Constructor. num_processes (integer) is the number of worker tasks. Creates a cluster
    # specification mapping with that many workers and a multi-process pool runner on it,
    # stored as the runner attribute. Starts worker processes as a side effect. Returns nothing.
    cluster_spec_dict = multi_worker_test_base.create_cluster_spec(
        num_workers=num_processes
    )
    self.runner = multi_process_runner.MultiProcessPoolRunner(cluster_spec_dict)


# Global MultiProcessPoolRunners that can be shared by test cases to avoid
# expensive initialization cost of TensorFlow in new processes.
#
# Note that they have to be globals and can't be owned by test classes because
# usually fn usually captures the test class instance, and test class
# instance can't be pickled if it has mpr as a member (it is not allowed to
# pickle Process objects).
# TODO(crccw): Use `num_workers` combination once it is ready.
global_mpr_2p = MultiProcessPoolRunner(num_processes=2)
global_mpr_1p = MultiProcessPoolRunner(num_processes=1)


def get_global_mpr(num_processes):
  # [FUNCTION DOC]
  # HELPER: lookup for the shared worker process pools.
  # Returns the pool runner of the global one-process pool when num_processes is 1, or of the
  # two-process pool when it is 2. Raises a value error for any other count.
  if num_processes == 1:
    return global_mpr_1p.runner
  elif num_processes == 2:
    return global_mpr_2p.runner
  else:
    raise ValueError(
        "get_global_mpr: num_processes must be 1 or 2, got %d" % num_processes
    )


class CollectiveOpsTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests the collective all-reduce cross-device ops (reduce and batch reduce, dense and
  # sparse values, eager and function-graph modes, different communication implementations,
  # process and GPU counts, unique-instance-key settings), using the shared global pools of
  # worker processes. Provides shared fixtures: a collective builder, a value-to-list
  # converter, a named tuple of run options, and reduce-and-verify helpers that run a replica
  # function in every worker process.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture: before every test, re-enables collective ops in all worker processes of
    # both the one-process and two-process pools, which gives fresh collective state and test
    # isolation. Returns nothing.
    super().setUp()
    # Enabling collectives can be done in "setUpClass", but requires using
    # different collective_keys in different tests as collectives are reused
    # across tests. Always resetting collective ops before each test offers
    # better test isolation.
    global_mpr_1p.runner.run(enable_collective_ops)
    global_mpr_2p.runner.run(enable_collective_ops)

  def make_collective(self, num_processes, gpu_per_process):
    # [FUNCTION DOC]
    # HELPER: builds the object under test inside a worker process.
    # num_processes: how many processes take part; gpu_per_process: GPUs used by each process
    # (zero means a single CPU device). Resolves the cluster from the environment, lists the
    # local devices of this task (one CPU, or the requested GPUs), computes the group size as
    # processes times local devices, and creates a collective all-reduce object with default
    # options. Returns a tuple of the collective object, the list of local device strings, and
    # this process's task id.
    """Returns collectives and other info to be used in tests.

    Args:
      num_processes: an integer indicating the number of processes that
        participate in the collective.
      gpu_per_process: number of GPUs (0 if no GPUs) used by each process.

    Returns:
     A tuple of (collective, devices, pid) where collective is a instance
     of `CollectiveAllReduce`, devices are a list of local devices (str)
     attached to the current process, and pid is the id of this process among
     all participant processes.
    """

    cluster_resolver = cluster_resolver_lib.TFConfigClusterResolver()
    devices = [
        "/job:worker/replica:0/task:%d/device:CPU:0" % cluster_resolver.task_id
    ]
    if gpu_per_process > 0:
      devices = [
          "/job:worker/replica:0/task:%d/device:GPU:%d"
          % (cluster_resolver.task_id, i)
          for i in range(gpu_per_process)
      ]
    group_size = num_processes * len(devices)
    collective = cross_device_ops_lib.CollectiveAllReduce(
        devices=devices,
        group_size=group_size,
        options=collective_util.Options(),
    )
    return collective, devices, cluster_resolver.task_id

  def as_list(self, value):
    # [FUNCTION DOC]
    # HELPER: normalizes reduction results for comparison.
    # Converts a tensor or indexed slices into a one-element list, and a mirrored value into
    # the list of its per-device components. Raises a value error for any other type.
    """An utility to convert a `Mirrored`, `Tensor` or `IndexedSlices` to a list.

    The reason it exists is to provide a uniformed view of returned value of
    "reduce" calls, especially across tf.function boundaries. Returning
    `Mirrored` from a tf.function will only evaluate the primary value, which
    makes collective ops of non-primary device being pruned, and will eventually
    cause hanging.

    Args:
      value: the value to convert, can be one of `Mirrored`, `Tensor` and
        `IndexedSlices`.

    Returns:
      A list of `Tensor` or `IndexedSlices`.
    """
    if isinstance(value, tensor_lib.Tensor):
      return [value]
    elif isinstance(value, IndexedSlices):
      return [value]
    elif isinstance(value, value_lib.Mirrored):
      return value.values
    else:
      raise ValueError("unwrap: unsupported input type: %s" % type(value))

  RunOptions = collections.namedtuple(  # pylint: disable=invalid-name
      "RunOptions",
      [
          "mode",  # A list of str from ["eager", "func_graph"]
          "num_processes",
          "gpus_per_process",
          "reduce_op",
          "communication_options",
          "prefer_unique_instance_key",
      ],
  )
  RunOptions.__new__.__defaults__ = (
      ["eager", "func_graph"],
      2,
      0,
      ReduceOp.SUM,
      collective_util.Options(),
      True,
  )

  def reduce_and_verify(self, inputs, expect, options):
    # [FUNCTION DOC]
    # HELPER: shared test driver for single reduce tests.
    # inputs: list of tensors or indexed slices, one per replica across all processes (indexed
    # by process id times local device count plus device index). expect: expected value for one
    # replica. options: the run options named tuple. Runs a replica function in every process
    # of the right pool. In each process it sets the unique-instance-key switch, builds the
    # collective, makes per-replica inputs, reduces with the chosen operation and communication
    # options, checks the result type is mirrored when more than one GPU is used, that result
    # devices equal the local devices, and that all values match the expectation, in eager mode
    # and/or inside a function graph depending on the options.
    # Side effect: sets a class-level launcher flag in the worker processes.
    """Reduce the given `inputs` and verify the output matches `expect`.

    Args:
      inputs: a list of `Tensor` or `IndexedSlices`, where i-th value will be
        fed to i-th replica.
      expect: a `Tensor` or `IndexedSlices`. This should be the expected value
        for one replica.
      options: a `RunOpotions` instance.
    """

    def replica_fn():
      # [FUNCTION DOC]
      # HELPER: body executed in every worker process by reduce_and_verify.
      # Applies the unique-instance-key option, builds the collective, then defines and runs the
      # reduction in eager and/or function-graph mode and compares with the expected value.
      CollectiveReplicaLauncher._prefer_unique_instance_key = (
          options.prefer_unique_instance_key
      )
      collective, devices, pid = self.make_collective(
          options.num_processes, options.gpus_per_process
      )

      def reduce_fn():
        # [FUNCTION DOC]
        # HELPER: nested function of reduce_and_verify, run eagerly or traced as a function.
        # Creates per-replica input values for this process's devices, reduces them with the
        # collective using the configured reduce operation and communication options, asserts
        # the result is mirrored when several GPUs are used, converts the result to a list
        # and asserts its devices equal the local devices. Returns a list of tensors.
        value_fn = lambda device_idx: inputs[pid * len(devices) + device_idx]
        per_replica_value = make_per_replica_value(value_fn, devices)
        reduced_values = collective.reduce(
            options.reduce_op,
            per_replica_value,
            per_replica_value,
            options.communication_options,
        )
        if options.gpus_per_process > 1:
          self.assertIsInstance(reduced_values, value_lib.Mirrored)
        reduced_values = self.as_list(reduced_values)
        self.assertAllEqual(devices, [v.device for v in reduced_values])
        return [ops.convert_to_tensor(v) for v in reduced_values]

      per_replica_expect = [ops.convert_to_tensor(expect)] * len(devices)

      if "eager" in options.mode:
        got = reduce_fn()
        self.assertAllClose(got, per_replica_expect)

      if "func_graph" in options.mode:
        got = def_function.function(reduce_fn)()
        self.assertAllClose(got, per_replica_expect)

    get_global_mpr(options.num_processes).run(replica_fn)

  def batch_reduce_and_verify(self, inputs, expect, options):
    # [FUNCTION DOC]
    # HELPER: shared test driver for batch reduce tests.
    # inputs: a two-level nested list (per replica, then per batch element) of tensors or
    # indexed slices. expect: list of expected values for one replica. options: run options
    # named tuple. Runs a replica function in every process of the right pool: sets the
    # unique-instance-key switch, builds the collective, creates a pair of per-replica value
    # (used as both source and destination) for every batch element, batch-reduces them, checks
    # mirrored type when several GPUs are used, that devices match, and that values equal the
    # expectation, eagerly and/or in a function graph per the options.
    """Batch reduce the given `inputs` and verify the output matches `expect`.

    Args:
      inputs: a 2-level nested list of `Tensor` or `IndexedSlices`, where i-th
        value will be fed to i-th replica.
      expect: a list of `Tensor` or `IndexedSlices`. This should be the expected
        value for one replica.
      options: a `RunOpotions` instance.
    """

    def replica_fn():
      # [FUNCTION DOC]
      # HELPER: body executed in every worker process by batch_reduce_and_verify.
      # Applies the unique-instance-key option, builds the collective, then defines and runs the
      # batch reduction in eager and/or function-graph mode and compares with the expectation.
      CollectiveReplicaLauncher._prefer_unique_instance_key = (
          options.prefer_unique_instance_key
      )
      collective, devices, pid = self.make_collective(
          options.num_processes, options.gpus_per_process
      )

      def batch_reduce_fn():
        # [FUNCTION DOC]
        # HELPER: nested function of batch_reduce_and_verify, run eagerly or traced.
        # Builds per-replica values for every batch element, batch-reduces them with the
        # configured operation and options, asserts mirrored results when several GPUs are
        # used, asserts devices match local devices for each result, and returns the results as
        # nested lists of tensors.
        batch_size = len(inputs[0])
        value_dst_pairs = []
        for i in range(batch_size):

          def value_fn(device_idx, idx=i):
            # [FUNCTION DOC]
            # HELPER: nested value provider. Returns the idx-th batch element of the input that
            # belongs to this process's device with the given index (idx is bound to the loop
            # position through a default argument).
            return inputs[pid * len(devices) + device_idx][idx]

          per_replica_value = make_per_replica_value(value_fn, devices)
          value_dst_pairs.append((per_replica_value, per_replica_value))
        reduced_values = collective.batch_reduce(
            options.reduce_op, value_dst_pairs, options.communication_options
        )
        if options.gpus_per_process > 1:
          for v in reduced_values:
            self.assertIsInstance(v, value_lib.Mirrored)
        reduced_values = [self.as_list(v) for v in reduced_values]
        for v in reduced_values:
          self.assertAllEqual(devices, [t.device for t in v])
        return nest.map_structure(ops.convert_to_tensor, reduced_values)

      per_replica_expect = nest.map_structure(
          lambda x: [ops.convert_to_tensor(x)] * len(devices), expect
      )

      if "eager" in options.mode:
        got = batch_reduce_fn()
        self.assertAllClose(got, per_replica_expect)

      if "func_graph" in options.mode:
        got = def_function.function(batch_reduce_fn)()
        self.assertAllClose(got, per_replica_expect)

    get_global_mpr(options.num_processes).run(replica_fn)

  @combinations.generate(
      combinations.combine(
          num_processes=[1, 2],
          required_gpus=[0, 1, 2],
          implementation=[
              CommunicationImplementation.AUTO,
              CommunicationImplementation.RING,
              CommunicationImplementation.NCCL,
          ],
          reduce_op=[ReduceOp.SUM, ReduceOp.MEAN],
          prefer_unique_instance_key=[True, False],
      )
  )
  def testReduceDense(
      self,
      num_processes,
      required_gpus,
      implementation,
      reduce_op,
      prefer_unique_instance_key,
  ):
    # [FUNCTION DOC]
    # Scenario: reduce one dense scalar per replica across 1 or 2 processes with 0, 1 or 2 GPUs
    # each, for each communication implementation (auto, ring, NCCL), reduce operation (sum,
    # mean) and unique-instance-key setting (generated combinations). Skips CPU plus NCCL, and
    # NCCL or auto when process count and GPU count differ. Inputs are consecutive values
    # starting at one, one per replica. Expected: sum or mean of the inputs seen on every
    # replica (for example three for two replicas summed), in eager and function-graph modes.
    if (
        required_gpus == 0
        and implementation == CommunicationImplementation.NCCL
    ):
      self.skipTest("Skip CPU + NCCL combination")
    if (
        num_processes != required_gpus
        and implementation == CommunicationImplementation.NCCL
    ):
      self.skipTest(
          "Skip NCCL combination with mismatched process and GPU "
          "count. NCCL requires physical GPUs for every process."
      )
    if (
        num_processes != required_gpus
        and implementation == CommunicationImplementation.AUTO
    ):
      self.skipTest(
          "Skip potential NCCL combination (AUTO) with mismatched "
          "process and GPU count. NCCL requires physical GPUs for "
          "every process."
      )
    options = self.RunOptions(
        num_processes=num_processes,
        gpus_per_process=required_gpus,
        reduce_op=reduce_op,
        communication_options=collective_util.Options(
            implementation=implementation
        ),
        prefer_unique_instance_key=prefer_unique_instance_key,
    )
    group_size = options.num_processes * (options.gpus_per_process or 1)

    inputs_data = [1.0, 2.0, 3.0, 4.0]
    inputs = inputs_data[0:group_size]

    if group_size == 1:
      expect = 1.0
    if group_size == 2:
      expect = 3.0 if reduce_op == ReduceOp.SUM else 1.5
    elif group_size == 4:
      expect = 10.0 if reduce_op == ReduceOp.SUM else 2.5

    self.reduce_and_verify(inputs, expect, options)

  @combinations.generate(
      combinations.combine(
          num_processes=[1, 2],
          required_gpus=[0, 1, 2],
          implementation=[
              CommunicationImplementation.AUTO,
              CommunicationImplementation.RING,
              CommunicationImplementation.NCCL,
          ],
          # TODO(b/166682130): add MEAN reduce once the bug is fixed.
          reduce_op=ReduceOp.SUM,
          prefer_unique_instance_key=[True, False],
      )
  )
  def testReduceSparse(
      self,
      num_processes,
      required_gpus,
      implementation,
      reduce_op,
      prefer_unique_instance_key,
  ):
    # [FUNCTION DOC]
    # Scenario: reduce indexed slices (two rows each, differing indices) across 1 or 2
    # processes with 0 to 2 GPUs each, for each communication implementation and unique-key
    # setting; reduce operation is only sum (mean is disabled pending a known bug). Skips the
    # same hardware combinations as the dense test. Only function-graph mode is run because
    # sparse reduce is unsupported eagerly. Expected: the concatenation of all replicas'
    # values and indices in replica order (sparse reduce gathers rather than merges duplicates).
    if (
        required_gpus == 0
        and implementation == CommunicationImplementation.NCCL
    ):
      self.skipTest("Skip CPU + NCCL combination")
    if (
        num_processes != required_gpus
        and implementation == CommunicationImplementation.NCCL
    ):
      self.skipTest(
          "Skip NCCL combination with mismatched process and GPU "
          "count. NCCL requires physical GPUs for every process."
      )
    if (
        num_processes != required_gpus
        and implementation == CommunicationImplementation.AUTO
    ):
      self.skipTest(
          "Skip potential NCCL combination (AUTO) with mismatched "
          "process and GPU count. NCCL requires physical GPUs for "
          "every process."
      )
    options = self.RunOptions(
        mode=["func_graph"],  # Sparse reduce is not supported in eager.
        num_processes=num_processes,
        gpus_per_process=required_gpus,
        reduce_op=reduce_op,
        communication_options=collective_util.Options(
            implementation=implementation
        ),
        prefer_unique_instance_key=prefer_unique_instance_key,
    )
    group_size = options.num_processes * (options.gpus_per_process or 1)

    inputs_data = [
        IndexedSlicesValue(
            values=[[1.0], [2.0]], indices=[0, 1], dense_shape=[10, 1]
        ),
        IndexedSlicesValue(
            values=[[3.0], [4.0]], indices=[1, 2], dense_shape=[10, 1]
        ),
        IndexedSlicesValue(
            values=[[5.0], [6.0]], indices=[7, 8], dense_shape=[10, 1]
        ),
        IndexedSlicesValue(
            values=[[7.0], [8.0]], indices=[3, 2], dense_shape=[10, 1]
        ),
    ]
    inputs = inputs_data[0:group_size]

    if group_size == 1:
      expect = IndexedSlices(
          values=[[1.0], [2.0]], indices=[0, 1], dense_shape=[10, 1]
      )
    elif group_size == 2:
      expect = IndexedSlices(
          values=[[1.0], [2.0], [3.0], [4.0]],
          indices=[0, 1, 1, 2],
          dense_shape=[10, 1],
      )
    elif group_size == 4:
      expect = IndexedSlices(
          values=[[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]],
          indices=[0, 1, 1, 2, 7, 8, 3, 2],
          dense_shape=[10, 1],
      )

    self.reduce_and_verify(inputs, expect, options)

  @combinations.generate(
      combinations.combine(prefer_unique_instance_key=[True, False])
  )
  def testReduceSparseVariableLength(self, prefer_unique_instance_key):
    # [FUNCTION DOC]
    # Scenario: two processes with one CPU device each reduce indexed slices of different
    # lengths (one row versus three rows), with unique instance keys both on and off. Runs in
    # function-graph mode only with the sum operation. Expected: all rows from both replicas
    # concatenated in replica order, exercising the padding path of the sparse reduction.
    # One device per process, 2 processes, 2 replicas in total.
    inputs = [
        IndexedSlicesValue(values=[[1.0]], indices=[0], dense_shape=[10, 1]),
        IndexedSlicesValue(
            values=[[2.0], [3.0], [4.0]], indices=[0, 1, 2], dense_shape=[10, 1]
        ),
    ]
    expect = IndexedSlices(
        values=[[1.0], [2.0], [3.0], [4.0]],
        indices=[0, 0, 1, 2],
        dense_shape=[10, 1],
    )
    self.reduce_and_verify(
        inputs,
        expect,
        self.RunOptions(
            mode=["func_graph"],  # Sparse reduce is not supported in eager.
            num_processes=2,
            reduce_op=ReduceOp.SUM,
            prefer_unique_instance_key=prefer_unique_instance_key,
        ),
    )

  @combinations.generate(
      combinations.combine(
          num_processes=[1, 2],
          required_gpus=[0, 1, 2],
          implementation=[
              CommunicationImplementation.AUTO,
              CommunicationImplementation.RING,
              CommunicationImplementation.NCCL,
          ],
          reduce_op=[ReduceOp.SUM, ReduceOp.MEAN],
          prefer_unique_instance_key=[True, False],
      )
  )
  def testBatchReduceDense(
      self,
      num_processes,
      required_gpus,
      implementation,
      reduce_op,
      prefer_unique_instance_key,
  ):
    # [FUNCTION DOC]
    # Scenario: batch-reduce a pair of dense scalars per replica across 1 or 2 processes with 0
    # to 2 GPUs each, for each communication implementation, reduce operation and unique-key
    # setting. Same skip rules as the single dense reduce test. Expected: element-wise sum or
    # mean over replicas for each batch element, on every replica, eager and function graph.
    if (
        required_gpus == 0
        and implementation == CommunicationImplementation.NCCL
    ):
      self.skipTest("Skip CPU + NCCL combination")
    if (
        num_processes != required_gpus
        and implementation == CommunicationImplementation.NCCL
    ):
      self.skipTest(
          "Skip NCCL combination with mismatched process and GPU "
          "count. NCCL requires physical GPUs for every process."
      )
    if (
        num_processes != required_gpus
        and implementation == CommunicationImplementation.AUTO
    ):
      self.skipTest(
          "Skip potential NCCL combination (AUTO) with mismatched "
          "process and GPU count. NCCL requires physical GPUs for "
          "every process."
      )

    options = self.RunOptions(
        num_processes=num_processes,
        gpus_per_process=required_gpus,
        reduce_op=reduce_op,
        communication_options=collective_util.Options(
            implementation=implementation
        ),
        prefer_unique_instance_key=prefer_unique_instance_key,
    )
    group_size = options.num_processes * (options.gpus_per_process or 1)

    inputs_data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    inputs = inputs_data[0:group_size]

    if group_size == 1:
      expect = [1.0, 2.0]
    if group_size == 2:
      expect = [4.0, 6.0] if reduce_op == ReduceOp.SUM else [2.0, 3.0]
    elif group_size == 4:
      expect = [16.0, 20.0] if reduce_op == ReduceOp.SUM else [4.0, 5.0]

    self.batch_reduce_and_verify(inputs, expect, options)

  @combinations.generate(
      combinations.combine(
          num_processes=[1, 2],
          required_gpus=[0, 1, 2],
          implementation=[
              CommunicationImplementation.AUTO,
              CommunicationImplementation.RING,
              CommunicationImplementation.NCCL,
          ],
          # TODO(b/166682130): add MEAN reduce once the bug is fixed.
          reduce_op=ReduceOp.SUM,
          prefer_unique_instance_key=[True, False],
      )
  )
  def testBatchReduceSparse(
      self,
      num_processes,
      required_gpus,
      implementation,
      reduce_op,
      prefer_unique_instance_key,
  ):
    # [FUNCTION DOC]
    # KNOWN ISSUE: in the four-replica case the expected second slices declare a dense shape
    # of five by two, while every input of that element has dense shape five by one; the
    # expectation looks inconsistent with the inputs (it only escapes failure if shapes are not
    # compared or the case is not run, since four replicas need many devices).
    # Scenario: batch-reduce pairs of indexed slices (different dense shapes per element)
    # across 1 or 2 processes with 0 to 2 GPUs each, per implementation and unique-key setting;
    # sum only (mean disabled pending a known bug). Same skip rules as the dense tests;
    # function-graph mode only. Expected: for each element the concatenation of all replicas'
    # values and indices in replica order.
    if (
        required_gpus == 0
        and implementation == CommunicationImplementation.NCCL
    ):
      self.skipTest("Skip CPU + NCCL combination")
    if (
        num_processes != required_gpus
        and implementation == CommunicationImplementation.NCCL
    ):
      self.skipTest(
          "Skip NCCL combination with mismatched process and GPU "
          "count. NCCL requires physical GPUs for every process."
      )
    if (
        num_processes != required_gpus
        and implementation == CommunicationImplementation.AUTO
    ):
      self.skipTest(
          "Skip potential NCCL combination (AUTO) with mismatched "
          "process and GPU count. NCCL requires physical GPUs for "
          "every process."
      )

    options = self.RunOptions(
        mode=["func_graph"],  # Sparse reduce is not supported in eager.
        num_processes=num_processes,
        gpus_per_process=required_gpus,
        reduce_op=reduce_op,
        communication_options=collective_util.Options(
            implementation=implementation
        ),
        prefer_unique_instance_key=prefer_unique_instance_key,
    )
    group_size = options.num_processes * (options.gpus_per_process or 1)

    inputs_data = (
        [
            IndexedSlicesValue(
                values=[[1.0], [2.0]], indices=[0, 1], dense_shape=[10, 1]
            ),
            IndexedSlicesValue(
                values=[[3.0], [4.0]], indices=[1, 2], dense_shape=[5, 1]
            ),
        ],
        [
            IndexedSlicesValue(
                values=[[5.0], [6.0]], indices=[1, 2], dense_shape=[10, 1]
            ),
            IndexedSlicesValue(
                values=[[7.0], [8.0]], indices=[0, 1], dense_shape=[5, 1]
            ),
        ],
        [
            IndexedSlicesValue(
                values=[[9.0], [10.0]], indices=[3, 4], dense_shape=[10, 1]
            ),
            IndexedSlicesValue(
                values=[[11.0], [12.0]], indices=[3, 4], dense_shape=[5, 1]
            ),
        ],
        [
            IndexedSlicesValue(
                values=[[13.0], [14.0]], indices=[8, 9], dense_shape=[10, 1]
            ),
            IndexedSlicesValue(
                values=[[15.0], [16.0]], indices=[3, 4], dense_shape=[5, 1]
            ),
        ],
    )
    inputs = inputs_data[0:group_size]

    if group_size == 1:
      expect = [
          IndexedSlices(
              values=[[1.0], [2.0]], indices=[0, 1], dense_shape=[10, 1]
          ),
          IndexedSlices(
              values=[[3.0], [4.0]], indices=[1, 2], dense_shape=[5, 1]
          ),
      ]
    if group_size == 2:
      expect = [
          IndexedSlices(
              values=[[1.0], [2.0], [5.0], [6.0]],
              indices=[0, 1, 1, 2],
              dense_shape=[10, 1],
          ),
          IndexedSlices(
              values=[[3.0], [4.0], [7.0], [8.0]],
              indices=[1, 2, 0, 1],
              dense_shape=[5, 1],
          ),
      ]
    elif group_size == 4:
      expect = [
          IndexedSlices(
              values=[
                  [1.0],
                  [2.0],
                  [5.0],
                  [6.0],
                  [9.0],
                  [10.0],
                  [13.0],
                  [14.0],
              ],
              indices=[0, 1, 1, 2, 3, 4, 8, 9],
              dense_shape=[10, 1],
          ),
          IndexedSlices(
              values=[
                  [3.0],
                  [4.0],
                  [7.0],
                  [8.0],
                  [11.0],
                  [12.0],
                  [15.0],
                  [16.0],
              ],
              indices=[1, 2, 0, 1, 3, 4, 3, 4],
              dense_shape=[5, 2],
          ),
      ]
    self.batch_reduce_and_verify(inputs, expect, options)

  def testBatchReduceMixedDenseAndSparse(self):
    # [FUNCTION DOC]
    # Scenario: two processes, CPU only, batch-reduce a batch mixing two dense scalars and two
    # indexed slices per replica, sum, function-graph mode only. Expected: dense elements are
    # summed across replicas; sparse elements are concatenated in replica order, with the
    # original dense shapes.

    options = self.RunOptions(
        num_processes=2,
        gpus_per_process=0,
        reduce_op=ReduceOp.SUM,
        mode=["func_graph"],
    )

    inputs_data = [
        [
            1.0,
            2.0,
            IndexedSlicesValue(
                values=[[1.0], [2.0]], indices=[0, 1], dense_shape=[10, 1]
            ),
            IndexedSlicesValue(
                values=[[3.0], [4.0]], indices=[1, 2], dense_shape=[5, 1]
            ),
        ],
        [
            3.0,
            4.0,
            IndexedSlicesValue(
                values=[[5.0], [6.0]], indices=[1, 2], dense_shape=[10, 1]
            ),
            IndexedSlicesValue(
                values=[[7.0], [8.0]], indices=[0, 1], dense_shape=[5, 1]
            ),
        ],
    ]

    expect = [
        4.0,
        6.0,
        IndexedSlices(
            values=[[1.0], [2.0], [5.0], [6.0]],
            indices=[0, 1, 1, 2],
            dense_shape=[10, 1],
        ),
        IndexedSlices(
            values=[[3.0], [4.0], [7.0], [8.0]],
            indices=[1, 2, 0, 1],
            dense_shape=[5, 1],
        ),
    ]

    self.batch_reduce_and_verify(inputs_data, expect, options)

  @combinations.generate(
      combinations.combine(
          num_processes=[1, 2],
          required_gpus=[0, 1, 2],
          implementation=[
              CommunicationImplementation.AUTO,
              CommunicationImplementation.RING,
              CommunicationImplementation.NCCL,
          ],
          reduce_op=[ReduceOp.SUM, ReduceOp.MEAN],
      )
  )
  def testAllReduceDense(
      self, num_processes, required_gpus, implementation, reduce_op
  ):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test is unconditionally skipped (after the hardware skips) because of a
    # tracked infrastructure bug, so everything after the skip, including the replica function
    # and assertions, never runs.
    # Scenario (when enabled): call the private all-reduce of the collective object in each
    # worker for a scalar of one and for a tuple of two scalars, inside function graphs, with
    # sum and mean over combinations of processes, GPUs and implementations. Expected: sum
    # gives each value times the group size, mean gives the original values, on every device.
    if (
        required_gpus == 0
        and implementation == CommunicationImplementation.NCCL
    ):
      self.skipTest("Skip CPU + NCCL combination")
    if (
        num_processes != required_gpus
        and implementation == CommunicationImplementation.NCCL
    ):
      self.skipTest(
          "Skip NCCL combination with mismatched process and GPU "
          "count. NCCL requires physical GPUs for every process."
      )
    if (
        num_processes != required_gpus
        and implementation == CommunicationImplementation.AUTO
    ):
      self.skipTest(
          "Skip potential NCCL combination (AUTO) with mismatched "
          "process and GPU count. NCCL requires physical GPUs for "
          "every process."
      )
    self.skipTest(
        "b/435404154: As we moved from NVIDIA CUDA base image to Ubuntu 22.04"
        " with NVIDIA Driver 580 installed for RBE, this test is failing and"
        " needs to be addressed as part of the bug."
    )

    def replica_fn():
      # [FUNCTION DOC]
      # HELPER: body run in every worker process by testAllReduceDense. Builds the collective,
      # runs the all-reduce of a scalar and of a pair of scalars in function graphs, and checks
      # results against sum or mean expectations.
      collective, devices, _ = self.make_collective(
          num_processes, required_gpus
      )
      options = collective_util.Options(implementation=implementation)
      group_size = num_processes * (required_gpus or 1)

      @def_function.function
      def collective_all_reduce():
        # [FUNCTION DOC]
        # HELPER: traced function. For each local device, creates a scalar of one on that device
        # and all-reduces it with the collective's private all-reduce using the replica index.
        # Returns the list of per-device results.
        results = []
        for replica_id, device in enumerate(devices):
          with ops.device(device):
            value = constant_op.constant(1.0)
            results.append(
                collective._all_reduce(reduce_op, value, replica_id, options)
            )
        return results

      got = collective_all_reduce()
      if reduce_op == ReduceOp.SUM:
        expect = [1.0 * group_size] * len(devices)
      elif reduce_op == ReduceOp.MEAN:
        expect = [1.0] * len(devices)
      self.assertAllClose(got, expect)

      @def_function.function
      def collective_batch_all_reduce():
        # [FUNCTION DOC]
        # HELPER: traced function. For each local device, creates a pair of scalars (one and
        # two) on that device and all-reduces the pair together. Returns the list of per-device
        # results.
        results = []
        for replica_id, device in enumerate(devices):
          with ops.device(device):
            value = (constant_op.constant(1.0), constant_op.constant(2.0))
            results.append(
                collective._all_reduce(reduce_op, value, replica_id, options)
            )
        return results

      got = collective_batch_all_reduce()
      if reduce_op == ReduceOp.SUM:
        expect = [(1.0 * group_size, 2.0 * group_size)] * len(devices)
      elif reduce_op == ReduceOp.MEAN:
        expect = [(1.0, 2.0)] * len(devices)
      self.assertAllClose(got, expect)

    get_global_mpr(num_processes).run(replica_fn)

  @combinations.generate(
      combinations.combine(
          num_processes=[1, 2],
          required_gpus=[0, 1, 2],
          implementation=[
              CommunicationImplementation.AUTO,
              CommunicationImplementation.RING,
              CommunicationImplementation.NCCL,
          ],
          reduce_op=[ReduceOp.SUM, ReduceOp.MEAN],
      )
  )
  def testAllReduceSparse(
      self, num_processes, required_gpus, implementation, reduce_op
  ):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test is skipped unconditionally as its very first action (tracked
    # infrastructure bug), so the hardware skips and the whole replica function are dead code.
    # Scenario (when enabled): call the private all-reduce of the collective object in each
    # worker with a single indexed slices and with a pair of indexed slices, inside function
    # graphs, for sum and mean. Expected: sum yields values multiplied by the group size with
    # unchanged indices; mean yields the original values.
    self.skipTest(
        "b/435404154: As we moved from NVIDIA CUDA base image to Ubuntu 22.04"
        " with NVIDIA Driver 580 installed for RBE, this test is failing and"
        " needs to be addressed as part of the bug."
    )
    if (
        required_gpus == 0
        and implementation == CommunicationImplementation.NCCL
    ):
      self.skipTest("Skip CPU + NCCL combination")
    if (
        num_processes != required_gpus
        and implementation == CommunicationImplementation.NCCL
    ):
      self.skipTest(
          "Skip NCCL combination with mismatched process and GPU "
          "count. NCCL requires physical GPUs for every process."
      )
    if (
        num_processes != required_gpus
        and implementation == CommunicationImplementation.AUTO
    ):
      self.skipTest(
          "Skip potential NCCL combination (AUTO) with mismatched "
          "process and GPU count. NCCL requires physical GPUs for "
          "every process."
      )

    def replica_fn():
      # [FUNCTION DOC]
      # HELPER: body run in every worker process by testAllReduceSparse. Builds the collective,
      # runs the all-reduce of indexed slices (single and pair) in function graphs, and compares
      # with the sum or mean expectation after converting to dense tensors.
      collective, devices, _ = self.make_collective(
          num_processes, required_gpus
      )
      options = collective_util.Options(implementation=implementation)
      group_size = num_processes * (required_gpus or 1)

      @def_function.function
      def collective_all_reduce():
        # [FUNCTION DOC]
        # HELPER: traced function. For each local device, builds an indexed slices with a
        # single row of one at index zero and dense shape five by one, all-reduces it with the
        # private all-reduce, and returns the list of per-device results.
        results = []
        for replica_id, device in enumerate(devices):
          with ops.device(device):
            value = IndexedSlices(
                values=array_ops.identity([[1.0]]),
                indices=array_ops.identity([0]),
                dense_shape=array_ops.identity([5, 1]),
            )
            results.append(
                collective._all_reduce(reduce_op, value, replica_id, options)
            )
        return results

      got = collective_all_reduce()
      if reduce_op == ReduceOp.SUM:
        expect = [IndexedSlices([[1.0 * group_size]], [0], [5, 1])] * len(
            devices
        )
      elif reduce_op == ReduceOp.MEAN:
        expect = [IndexedSlices([[1.0]], [0], [5, 1])] * len(devices)
      self.assertAllClose(
          nest.map_structure(ops.convert_to_tensor, got),
          nest.map_structure(ops.convert_to_tensor, expect),
      )

      @def_function.function
      def collective_batch_all_reduce():
        # [FUNCTION DOC]
        # HELPER: traced function. For each local device, builds a pair of indexed slices (rows
        # of one at index zero and three at index two, dense shape five by one), all-reduces the
        # pair together, and returns the list of per-device results.
        results = []
        for replica_id, device in enumerate(devices):
          with ops.device(device):
            value = (
                IndexedSlices(
                    array_ops.identity([[1.0]]),
                    array_ops.identity([0]),
                    array_ops.identity([5, 1]),
                ),
                IndexedSlices(
                    array_ops.identity([[3.0]]),
                    array_ops.identity([2]),
                    array_ops.identity([5, 1]),
                ),
            )
            results.append(
                collective._all_reduce(reduce_op, value, replica_id, options)
            )
        return results

      got = collective_batch_all_reduce()
      if reduce_op == ReduceOp.SUM:
        expect = [(
            IndexedSlices([[1.0 * group_size]], [0], [5, 1]),
            IndexedSlices([[3.0 * group_size]], [2], [5, 1]),
        )] * len(devices)
      elif reduce_op == ReduceOp.MEAN:
        expect = [(
            IndexedSlices([[1.0]], [0], [5, 1]),
            IndexedSlices([[3.0]], [2], [5, 1]),
        )] * len(devices)
      self.assertAllClose(
          nest.map_structure(ops.convert_to_tensor, got),
          nest.map_structure(ops.convert_to_tensor, expect),
      )

    get_global_mpr(num_processes).run(replica_fn)

  @combinations.generate(
      combinations.combine(
          num_processes=2,
          required_gpus=0,
          implementation=CommunicationImplementation.AUTO,
          reduce_op=ReduceOp.SUM,
      )
  )
  def testAllReduceMixedDenseAndSparse(
      self, num_processes, required_gpus, implementation, reduce_op
  ):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the only generated combination is two processes, zero GPUs, auto
    # implementation, which satisfies the mismatched-process-and-GPU skip condition, so the
    # test always skips and the body never runs.
    # Scenario (when it runs): two CPU-only processes call the collective's private all-reduce
    # on a tuple mixing two indexed slices and two dense scalars, sum, auto implementation, in a
    # function graph. Expected: every element is multiplied by the group size (indices
    # unchanged) on every device.

    if (
        num_processes != required_gpus
        and implementation == CommunicationImplementation.AUTO
    ):
      self.skipTest(
          "Skip potential NCCL combination (AUTO) with mismatched "
          "process and GPU count. NCCL requires physical GPUs for "
          "every process."
      )

    def replica_fn():
      # [FUNCTION DOC]
      # HELPER: body run in every worker process by testAllReduceMixedDenseAndSparse. Builds the
      # collective, all-reduces a mixed sparse and dense tuple in a function graph and compares
      # with the group-size multiple of each element after converting to dense tensors.
      collective, devices, _ = self.make_collective(
          num_processes, required_gpus
      )
      options = collective_util.Options(implementation=implementation)
      group_size = num_processes * (required_gpus or 1)

      @def_function.function
      def collective_batch_all_reduce():
        # [FUNCTION DOC]
        # HELPER: traced function. For each local device, builds a tuple of an indexed slices,
        # a scalar, another indexed slices and another scalar, all-reduces the tuple with the
        # private all-reduce, and returns the list of per-device results.
        results = []
        for replica_id, device in enumerate(devices):
          with ops.device(device):
            value = (
                IndexedSlices(
                    array_ops.identity([[1.0]]),
                    array_ops.identity([0]),
                    array_ops.identity([5, 1]),
                ),
                array_ops.identity(1.0),
                IndexedSlices(
                    array_ops.identity([[3.0]]),
                    array_ops.identity([2]),
                    array_ops.identity([5, 1]),
                ),
                array_ops.identity(2.0),
            )
            results.append(
                collective._all_reduce(reduce_op, value, replica_id, options)
            )
        return results

      got = collective_batch_all_reduce()
      expect = [(
          IndexedSlices([[1.0 * group_size]], [0], [5, 1]),
          1.0 * group_size,
          IndexedSlices([[3.0 * group_size]], [2], [5, 1]),
          2.0 * group_size,
      )] * len(devices)
      self.assertAllClose(
          nest.map_structure(ops.convert_to_tensor, got),
          nest.map_structure(ops.convert_to_tensor, expect),
      )

    get_global_mpr(num_processes).run(replica_fn)

  @combinations.generate(
      combinations.combine(
          num_processes=[1, 2],
          required_gpus=[0, 1, 2],
          axis=[0, 1, 2],
          func_mode=["eager", "func_graph"],
          implementation=[
              CommunicationImplementation.AUTO,
              CommunicationImplementation.RING,
              CommunicationImplementation.NCCL,
          ],
          prefer_unique_instance_key=[True, False],
      )
  )
  def testAllGatherSameShape(
      self,
      num_processes,
      required_gpus,
      implementation,
      func_mode,
      axis,
      prefer_unique_instance_key,
  ):
    # [FUNCTION DOC]
    # Scenario: every replica holds the same small three-dimensional float tensor; the
    # collective gather is run along each of the axes zero, one and two, in eager or function
    # graph mode, over combinations of processes, GPUs, implementations and unique-key
    # settings. Same hardware skip rules as the reduce tests. Expected: the result on every
    # device equals the input concatenated group-size times along the chosen axis; outside eager
    # mode the result devices are also checked.

    if (
        required_gpus == 0
        and implementation == CommunicationImplementation.NCCL
    ):
      self.skipTest("Skip CPU + NCCL combination")
    if (
        num_processes != required_gpus
        and implementation == CommunicationImplementation.NCCL
    ):
      self.skipTest(
          "Skip NCCL combination with mismatched process and GPU "
          "count. NCCL requires physical GPUs for every process."
      )
    if (
        num_processes != required_gpus
        and implementation == CommunicationImplementation.AUTO
    ):
      self.skipTest(
          "Skip potential NCCL combination (AUTO) with mismatched "
          "process and GPU count. NCCL requires physical GPUs for "
          "every process."
      )

    def replica_fn():
      # [FUNCTION DOC]
      # HELPER: body run in every worker process by testAllGatherSameShape. Sets the
      # unique-instance-key switch, builds the collective, gathers identical per-replica values
      # along the axis in eager or function-graph mode and compares with the expected
      # concatenation.
      CollectiveReplicaLauncher._prefer_unique_instance_key = (
          prefer_unique_instance_key
      )
      collective, devices, _ = self.make_collective(
          num_processes, required_gpus
      )
      options = collective_util.Options(implementation=implementation)
      value = constant_op.constant([[[1, 2], [1, 2]]], dtype=dtypes.float32)

      def gather_fn():
        # [FUNCTION DOC]
        # HELPER: gathers the per-replica copies of the value along the axis with the
        # collective, converts the result to a list, checks devices (only outside eager mode,
        # where the device attribute is unreliable), and returns a list of tensors.
        per_replica_value = make_per_replica_value(value, devices)
        gathered_values = collective._gather(
            per_replica_value, per_replica_value, axis=axis, options=options
        )
        gathered_values = self.as_list(gathered_values)
        # Skip checking devices in eager. In eager the device attribute doesn't
        # reflect the actual device of the tensor.
        if not context.executing_eagerly():
          self.assertAllEqual(devices, [v.device for v in gathered_values])
        return [ops.convert_to_tensor(v) for v in gathered_values]

      group_size = num_processes * (required_gpus or 1)
      expect = array_ops.concat([value] * group_size, axis=axis)
      per_replica_expect = [ops.convert_to_tensor(expect)] * len(devices)

      if func_mode == "eager":
        result = gather_fn()
        self.assertAllClose(result, per_replica_expect)

      if func_mode == "func_graph":
        result = def_function.function(gather_fn)()
        self.assertAllClose(result, per_replica_expect)

    get_global_mpr(num_processes).run(replica_fn)

  @combinations.generate(
      combinations.combine(
          num_processes=[1, 2],
          required_gpus=[0, 1, 2],
          implementation=[CommunicationImplementation.RING],
      )
  )
  def testCollectiveV2ControlFlow(
      self, num_processes, required_gpus, implementation
  ):
    # [FUNCTION DOC]
    # Scenario: with the ring implementation over 1 or 2 processes and 0 to 2 GPUs, a collective
    # reduce is placed inside both branches of a conditional in a function graph (the condition
    # is a runtime false value), to check that collectives work inside control flow. Expected:
    # the summed value averaged over local devices equals the number of replicas times the
    # input of one.

    def replica_fn():
      # [FUNCTION DOC]
      # HELPER: body run in every worker process by testCollectiveV2ControlFlow. Enables unique
      # instance keys, builds the collective and per-replica inputs of one, runs the traced
      # function and asserts the result.
      CollectiveReplicaLauncher._prefer_unique_instance_key = True
      collective, devices, _ = self.make_collective(
          num_processes, required_gpus
      )
      options = collective_util.Options(implementation=implementation)
      value = make_per_replica_value(constant_op.constant([1.0]), devices)

      @def_function.function
      def reduce_fn():
        # [FUNCTION DOC]
        # HELPER: traced function that returns the result of a conditional whose both branches
        # run the same collective reduce, averaged over local devices.

        def cond_body():
          # [FUNCTION DOC]
          # HELPER: branch body. Sums the per-replica inputs with the collective, then returns
          # the sum of the local results divided by the number of local devices.
          reduced = collective.reduce(
              reduce_util.ReduceOp.SUM, value, value, options
          )
          return math_ops.add_n(self.as_list(reduced)) / len(devices)

        return cond.cond(array_ops.identity(False), cond_body, cond_body)

      num_replicas = num_processes * len(devices)
      self.assertAllEqual(reduce_fn(), [1.0 * num_replicas])

    get_global_mpr(num_processes).run(replica_fn)

  @combinations.generate(
      combinations.combine(
          num_processes=1,
          required_gpus=2,
          implementation=[
              CommunicationImplementation.RING,
              CommunicationImplementation.NCCL,
          ],
          prefer_unique_instance_key=[True, False],
      )
  )
  def testMultiThreadedCollectiveLaunchNoInterleave(
      self,
      num_processes,
      required_gpus,
      implementation,
      prefer_unique_instance_key,
  ):
    # [FUNCTION DOC]
    # Scenario: one process with two GPUs, ring or NCCL, unique keys on and off. Two threads
    # launch batch reductions of two values each, with an artificial delay inserted into the
    # underlying all-reduce so that kernel launches would interleave across devices in an
    # order that deadlocks NCCL if launch order were not enforced. Expected: both batch
    # reductions finish; the main thread's sums are all four and the other thread's all two.
    # Skips NCCL when process and GPU counts differ (always so for the generated GPU count).

    if (
        num_processes != required_gpus
        and implementation == CommunicationImplementation.NCCL
    ):
      self.skipTest(
          "Skip NCCL combination with mismatched process and GPU "
          "count. NCCL requires physical GPUs for every process."
      )
    if (
        num_processes != required_gpus
        and implementation == CommunicationImplementation.AUTO
    ):
      self.skipTest(
          "Skip potential NCCL combination (AUTO) with mismatched "
          "process and GPU count. NCCL requires physical GPUs for "
          "every process."
      )

    def replica_fn():
      # [FUNCTION DOC]
      # HELPER: body run in the worker process by the multi-threaded launch test. Sets the
      # unique-key switch, builds the collective and two per-replica inputs, patches the
      # all-reduce op with a delaying wrapper, and runs two batch reductions concurrently,
      # one in a helper thread and one in the calling thread, asserting both results.
      CollectiveReplicaLauncher._prefer_unique_instance_key = (
          prefer_unique_instance_key
      )
      collective, devices, _ = self.make_collective(
          num_processes, required_gpus
      )
      options = collective_util.Options(implementation=implementation)

      # We would like to simulate the following sequence:
      #   thread-0  device0                 device1
      #   thread-1          device0 device1
      # If the kernel launch sequence is as-is the program will deadlock since
      # NCCL requires the launch order to be same on each device.
      v0 = make_per_replica_value(1.0, devices)
      v1 = make_per_replica_value(2.0, devices)

      # Add a delay to collective_ops.all_reduce according to the input tensors
      # index in `sequence.`
      sequence = [v0.values[0], v1.values[0], v1.values[1], v0.values[1]]
      all_reduce = collective_ops.all_reduce

      def delayed_all_reduce(input_tensor, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: replacement for the all-reduce op used while patched. If the input tensor is
        # one of the four listed tensors (identity comparison), sleeps for as many seconds as
        # its position in that list, to force a chosen launch order; then forwards the call,
        # with all remaining arguments, to the original all-reduce and returns its result.
        for idx, v in enumerate(sequence):
          if input_tensor is v:
            time.sleep(idx)
            break
        return all_reduce(input_tensor, *args, **kwargs)

      with test.mock.patch.object(
          collective_ops, "all_reduce", delayed_all_reduce
      ):
        # We only use NCCL for batch reduce with two or more values, so we use
        # two values here.

        def thread_fn():
          # [FUNCTION DOC]
          # HELPER: runs in the helper thread. Batch-reduces two pairs of the first input with
          # a sum and asserts each result holds the value two on both devices.
          reduced = collective.batch_reduce(
              reduce_util.ReduceOp.SUM, [(v0, v0), (v0, v0)], options
          )
          self.assertAllEqual(reduced[0].values, [2.0, 2.0])
          self.assertAllEqual(reduced[1].values, [2.0, 2.0])

        t = threading.Thread(target=thread_fn)
        t.start()
        reduced = collective.batch_reduce(
            reduce_util.ReduceOp.SUM, [(v1, v1), (v1, v1)], options
        )
        self.assertAllEqual(reduced[0].values, [4.0, 4.0])
        self.assertAllEqual(reduced[1].values, [4.0, 4.0])
        t.join()

    get_global_mpr(num_processes).run(replica_fn)

  @combinations.generate(
      combinations.combine(
          num_processes=1,
          required_gpus=2,
          implementation=[
              CommunicationImplementation.RING,
              CommunicationImplementation.NCCL,
          ],
          prefer_unique_instance_key=[True, False],
      )
  )
  def testInputsAreFunctionArgs(
      self,
      num_processes,
      required_gpus,
      implementation,
      prefer_unique_instance_key,
  ):
    # [FUNCTION DOC]
    # Scenario: one process with two GPUs, ring or NCCL, unique keys on and off. A traced
    # function receives per-replica values as arguments and batch-reduces two pairs of them.
    # Expected: inside the function the argument components have no device placement, the
    # reduced components are placed on the matching devices, and every reduced value is two.

    if (
        num_processes != required_gpus
        and implementation == CommunicationImplementation.NCCL
    ):
      self.skipTest(
          "Skip NCCL combination with mismatched process and GPU "
          "count. NCCL requires physical GPUs for every process."
      )
    if (
        num_processes != required_gpus
        and implementation == CommunicationImplementation.AUTO
    ):
      self.skipTest(
          "Skip potential NCCL combination (AUTO) with mismatched "
          "process and GPU count. NCCL requires physical GPUs for "
          "every process."
      )

    def replica_fn():
      # [FUNCTION DOC]
      # HELPER: body run in the worker process by testInputsAreFunctionArgs. Sets the unique-key
      # switch, builds the collective, creates per-replica inputs of one, calls the traced
      # function and asserts all reduced values equal two.
      CollectiveReplicaLauncher._prefer_unique_instance_key = (
          prefer_unique_instance_key
      )
      collective, devices, _ = self.make_collective(
          num_processes, required_gpus
      )
      options = collective_util.Options(implementation=implementation)

      @def_function.function
      def reduce_fn(v):
        # [FUNCTION DOC]
        # HELPER: traced function taking a per-replica value. Asserts the argument components
        # have empty device placement, batch-reduces two pairs of it with a sum, asserts the
        # reduced components sit on the two local devices, and returns the two lists of
        # per-device values (returning mirrored values would only evaluate the primary one and
        # cause a hang).
        # Function inputs don't have device placement.
        self.assertEqual(v.values[0].device, "")
        self.assertEqual(v.values[1].device, "")
        # We only use NCCL for batch reduce with two or more values, so we use
        # two values here.
        reduced = collective.batch_reduce(
            reduce_util.ReduceOp.SUM, [(v, v), (v, v)], options
        )
        self.assertEqual(reduced[0].values[0].device, devices[0])
        self.assertEqual(reduced[0].values[1].device, devices[1])
        self.assertEqual(reduced[1].values[0].device, devices[0])
        self.assertEqual(reduced[1].values[1].device, devices[1])
        # Returning Mirrored only evaluates the primary value, which causes
        # hanging,
        return [reduced[0].values, reduced[1].values]

      v = make_per_replica_value(1.0, devices)
      reduced = reduce_fn(v)
      self.assertAllClose(reduced, [[2.0, 2.0], [2.0, 2.0]])

    get_global_mpr(num_processes).run(replica_fn)

  @combinations.generate(
      combinations.combine(
          num_processes=2,
          required_gpus=[0, 1],
          implementation=[
              CommunicationImplementation.RING,
              CommunicationImplementation.NCCL,
          ],
          prefer_unique_instance_key=[True, False],
      )
  )
  def testTimeoutReduceDense(
      self,
      num_processes,
      implementation,
      required_gpus,
      prefer_unique_instance_key,
  ):
    # [FUNCTION DOC]
    # Scenario: two processes (0 or 1 GPUs each, ring or NCCL, unique keys on and off); only
    # the first worker launches a dense reduce with a one second timeout in a function graph,
    # while the other worker never joins. Expected: a deadline-exceeded error is raised. Skips
    # CPU plus NCCL and NCCL or auto with mismatched process and GPU counts (so NCCL with one
    # GPU per process also skips, since two processes differ from one GPU).
    # The parameter order differs from other tests, but parameters are passed by name.

    if (
        required_gpus == 0
        and implementation == CommunicationImplementation.NCCL
    ):
      self.skipTest("Skip CPU + NCCL combination")
    if (
        num_processes != required_gpus
        and implementation == CommunicationImplementation.NCCL
    ):
      self.skipTest(
          "Skip NCCL combination with mismatched process and GPU "
          "count. NCCL requires physical GPUs for every process."
      )
    if (
        num_processes != required_gpus
        and implementation == CommunicationImplementation.AUTO
    ):
      self.skipTest(
          "Skip potential NCCL combination (AUTO) with mismatched "
          "process and GPU count. NCCL requires physical GPUs for "
          "every process."
      )

    def replica_fn():
      # [FUNCTION DOC]
      # HELPER: body run in every worker process by testTimeoutReduceDense. Sets the unique-key
      # switch and builds the collective; every worker except the first returns immediately.
      # The first worker runs the reduce with a one second timeout and asserts it raises a
      # deadline-exceeded error. (Its comment says three workers while only two are used.)
      CollectiveReplicaLauncher._prefer_unique_instance_key = (
          prefer_unique_instance_key
      )
      collective, devices, task_id = self.make_collective(
          num_processes, required_gpus
      )
      if task_id != 0:
        return

      v = make_per_replica_value(1.0, devices)
      options = collective_util.Options(
          timeout_seconds=1.0, implementation=implementation
      )

      @def_function.function
      def reduce_dense():
        # [FUNCTION DOC]
        # HELPER: traced function that sums the dense per-replica value with the collective,
        # using the one second timeout options, and returns the result.
        return collective.reduce(reduce_util.ReduceOp.SUM, v, v, options)

      # The collective should time out because we only launch it on worker-0,
      # while there're three workers in total.
      with self.assertRaises(errors.DeadlineExceededError):
        reduce_dense()

    get_global_mpr(num_processes).run(replica_fn)

  @combinations.generate(
      combinations.combine(
          num_processes=2,
          required_gpus=[0, 1],
          implementation=[
              CommunicationImplementation.RING,
              CommunicationImplementation.NCCL,
          ],
          prefer_unique_instance_key=[True, False],
      )
  )
  def testTimeoutBatchReduceDense(
      self,
      num_processes,
      implementation,
      required_gpus,
      prefer_unique_instance_key,
  ):
    # [FUNCTION DOC]
    # Scenario: same as the dense reduce timeout test but with a batch reduce of two pairs of
    # dense values: only the first of two workers launches it with a one second timeout.
    # Expected: a deadline-exceeded error. Same hardware skip rules.
    if (
        required_gpus == 0
        and implementation == CommunicationImplementation.NCCL
    ):
      self.skipTest("Skip CPU + NCCL combination")
    if (
        num_processes != required_gpus
        and implementation == CommunicationImplementation.NCCL
    ):
      self.skipTest(
          "Skip NCCL combination with mismatched process and GPU "
          "count. NCCL requires physical GPUs for every process."
      )
    if (
        num_processes != required_gpus
        and implementation == CommunicationImplementation.AUTO
    ):
      self.skipTest(
          "Skip potential NCCL combination (AUTO) with mismatched "
          "process and GPU count. NCCL requires physical GPUs for "
          "every process."
      )

    def replica_fn():
      # [FUNCTION DOC]
      # HELPER: body run in every worker process by testTimeoutBatchReduceDense. Non-first
      # workers return immediately; the first builds dense inputs and runs the batch reduce with
      # a one second timeout, asserting a deadline-exceeded error.
      CollectiveReplicaLauncher._prefer_unique_instance_key = (
          prefer_unique_instance_key
      )
      collective, devices, task_id = self.make_collective(
          num_processes, required_gpus
      )
      if task_id != 0:
        return

      v = make_per_replica_value(1.0, devices)
      options = collective_util.Options(
          timeout_seconds=1.0, implementation=implementation
      )

      @def_function.function
      def batch_reduce_dense():
        # [FUNCTION DOC]
        # HELPER: traced function that batch-reduces two pairs of the dense value with a sum
        # and the timeout options, and returns the results.
        return collective.batch_reduce(
            reduce_util.ReduceOp.SUM, [(v, v), (v, v)], options
        )

      # The collective should time out because we only launch it on worker-0,
      # while there're two workers in total.
      with self.assertRaises(errors.DeadlineExceededError):
        batch_reduce_dense()

    get_global_mpr(num_processes).run(replica_fn)

  @combinations.generate(
      combinations.combine(
          num_processes=2,
          required_gpus=[0, 1],
          implementation=[
              CommunicationImplementation.RING,
              CommunicationImplementation.NCCL,
          ],
          prefer_unique_instance_key=[True, False],
      )
  )
  def testTimeoutReduceSparse(
      self,
      num_processes,
      implementation,
      required_gpus,
      prefer_unique_instance_key,
  ):
    # [FUNCTION DOC]
    # Scenario: same as the dense reduce timeout test but the reduced value is an indexed
    # slices; only the first of two workers launches it with a one second timeout.
    # Expected: a deadline-exceeded error. Same hardware skip rules.
    if (
        required_gpus == 0
        and implementation == CommunicationImplementation.NCCL
    ):
      self.skipTest("Skip CPU + NCCL combination")
    if (
        num_processes != required_gpus
        and implementation == CommunicationImplementation.NCCL
    ):
      self.skipTest(
          "Skip NCCL combination with mismatched process and GPU "
          "count. NCCL requires physical GPUs for every process."
      )
    if (
        num_processes != required_gpus
        and implementation == CommunicationImplementation.AUTO
    ):
      self.skipTest(
          "Skip potential NCCL combination (AUTO) with mismatched "
          "process and GPU count. NCCL requires physical GPUs for "
          "every process."
      )

    def replica_fn():
      # [FUNCTION DOC]
      # HELPER: body run in every worker process by testTimeoutReduceSparse. Non-first workers
      # return immediately; the first builds an indexed slices input and runs the reduce with a
      # one second timeout, asserting a deadline-exceeded error.
      CollectiveReplicaLauncher._prefer_unique_instance_key = (
          prefer_unique_instance_key
      )
      collective, devices, task_id = self.make_collective(
          num_processes, required_gpus
      )
      if task_id != 0:
        return

      v = make_per_replica_value(
          IndexedSlicesValue(
              values=[[4.0, 6.0]], indices=[1], dense_shape=[5, 2]
          ),
          devices,
      )
      options = collective_util.Options(
          timeout_seconds=1.0, implementation=implementation
      )

      @def_function.function
      def reduce_sparse():
        # [FUNCTION DOC]
        # HELPER: traced function that sums the sparse per-replica value with the collective,
        # using the timeout options, and returns the result.
        return collective.reduce(reduce_util.ReduceOp.SUM, v, v, options)

      # The collective should time out because we only launch it on worker-0,
      # while there're two workers in total.
      with self.assertRaises(errors.DeadlineExceededError):
        reduce_sparse()

    get_global_mpr(num_processes).run(replica_fn)

  @combinations.generate(
      combinations.combine(
          num_processes=2,
          required_gpus=[0, 1],
          implementation=[
              CommunicationImplementation.RING,
              CommunicationImplementation.NCCL,
          ],
          prefer_unique_instance_key=[True, False],
      )
  )
  def testTimeoutBatchReduceSparse(
      self,
      num_processes,
      required_gpus,
      implementation,
      prefer_unique_instance_key,
  ):
    # [FUNCTION DOC]
    # Scenario: same as the dense batch timeout test but with indexed slices values; only the
    # first of two workers launches the batch reduce with a one second timeout.
    # Expected: a deadline-exceeded error. Same hardware skip rules.
    if (
        required_gpus == 0
        and implementation == CommunicationImplementation.NCCL
    ):
      self.skipTest("Skip CPU + NCCL combination")
    if (
        num_processes != required_gpus
        and implementation == CommunicationImplementation.NCCL
    ):
      self.skipTest(
          "Skip NCCL combination with mismatched process and GPU "
          "count. NCCL requires physical GPUs for every process."
      )
    if (
        num_processes != required_gpus
        and implementation == CommunicationImplementation.AUTO
    ):
      self.skipTest(
          "Skip potential NCCL combination (AUTO) with mismatched "
          "process and GPU count. NCCL requires physical GPUs for "
          "every process."
      )

    def replica_fn():
      # [FUNCTION DOC]
      # HELPER: body run in every worker process by testTimeoutBatchReduceSparse. Non-first
      # workers return immediately; the first builds an indexed slices input and runs the batch
      # reduce with a one second timeout, asserting a deadline-exceeded error.
      CollectiveReplicaLauncher._prefer_unique_instance_key = (
          prefer_unique_instance_key
      )
      collective, devices, task_id = self.make_collective(
          num_processes, required_gpus
      )
      if task_id != 0:
        return

      v = make_per_replica_value(
          IndexedSlicesValue(
              values=[[4.0, 6.0]], indices=[1], dense_shape=[5, 2]
          ),
          devices,
      )
      options = collective_util.Options(
          timeout_seconds=1.0, implementation=implementation
      )

      @def_function.function
      def batch_reduce_sparse():
        # [FUNCTION DOC]
        # HELPER: traced function that batch-reduces two pairs of the sparse value with a sum
        # and the timeout options, and returns the results.
        return collective.batch_reduce(
            reduce_util.ReduceOp.SUM, [(v, v), (v, v)], options
        )

      # The collective should time out because we only launch it on worker-0,
      # while there're two workers in total.
      with self.assertRaises(errors.DeadlineExceededError):
        batch_reduce_sparse()

    get_global_mpr(num_processes).run(replica_fn)

  @combinations.generate(combinations.combine(num_processes=1, required_gpus=2))
  def testNcclOrdering(self, num_processes, required_gpus):
    # [FUNCTION DOC]
    # Scenario: one process with two GPUs, NCCL implementation, with ordering tokens enabled.
    # A traced function performs many sum reductions of dense and sparse values: directly,
    # inside nested traced functions, in a conditional and in loops. Expected: in the resulting
    # graph, on each device, the collective, conditional, loop and function-call operations
    # (including device-less ones) cover all those types and execute in a strictly sequential
    # order according to a topological sort. Skips when process and GPU counts differ.

    if num_processes != required_gpus:
      self.skipTest(
          "Skip NCCL combination with mismatched process and GPU "
          "count. NCCL requires physical GPUs for every process."
      )

    def replica_fn():
      # [FUNCTION DOC]
      # HELPER: body run in the worker process by testNcclOrdering. Turns on unique instance
      # keys and ordering tokens (class-level switches), builds a NCCL collective and dense and
      # sparse inputs, traces the outer function, then inspects its graph: collects ordered
      # operation types per device, topologically sorts operations, and for each device checks
      # all expected operation types are present and executed in sequence.
      CollectiveReplicaLauncher._prefer_unique_instance_key = True
      CollectiveReplicaLauncher._prefer_ordering_token = True
      collective, devices, _ = self.make_collective(
          num_processes, required_gpus
      )
      options = collective_util.Options(
          implementation=CommunicationImplementation.NCCL
      )

      v_dense = make_per_replica_value([1.0, 1.0], devices)
      v_sparse = make_per_replica_value(
          [
              IndexedSlicesValue([[4.0, 6.0], [5.0, 6.0]], [1, 3], [5, 2]),
              IndexedSlicesValue([[4.0, 6.0], [5.0, 6.0]], [1, 3], [5, 2]),
          ],
          devices,
      )

      @def_function.function
      def nested_dense():
        # [FUNCTION DOC]
        # HELPER: traced inner function performing one sum reduction of the dense input.
        collective.reduce(reduce_util.ReduceOp.SUM, v_dense, v_dense, options)

      @def_function.function
      def nested_sparse():
        # [FUNCTION DOC]
        # HELPER: traced inner function performing one sum reduction of the sparse input.
        collective.reduce(reduce_util.ReduceOp.SUM, v_sparse, v_sparse, options)

      # All collectives, function calls, if clause and while loops should be
      # chained by control dependencies, so that the execution order is
      # deterministic.
      @def_function.function
      def f():
        # [FUNCTION DOC]
        # HELPER: outer traced function whose graph is inspected. Chains sparse and dense sum
        # reductions directly, through nested traced functions, inside a conditional branch (one
        # branch each for dense and sparse) and inside loops, then repeats dense and sparse.
        # Returns nothing; its graph is what the test checks.
        # pylint: disable=pointless-statement
        collective.reduce(reduce_util.ReduceOp.SUM, v_sparse, v_sparse, options)
        # reducing dense value.
        collective.reduce(reduce_util.ReduceOp.SUM, v_dense, v_dense, options)
        # reducing sparse value.
        collective.reduce(reduce_util.ReduceOp.SUM, v_sparse, v_sparse, options)
        # reduce dense value in nested tf.function.
        nested_dense()
        # reduce sparse value in nested tf.function.
        nested_sparse()
        # reduce dense value in tf.cond.
        if array_ops.identity(1.0) > array_ops.identity(2.0):
          collective.reduce(reduce_util.ReduceOp.SUM, v_dense, v_dense, options)
        else:
          v_dense
        # reduce sparse value in tf.cond.
        if array_ops.identity(1.0) > array_ops.identity(2.0):
          v_sparse
        else:
          collective.reduce(
              reduce_util.ReduceOp.SUM, v_sparse, v_sparse, options
          )
        # reduce dense value in tf.while_loop.
        i = array_ops.identity(1)
        while i < 3:
          collective.reduce(reduce_util.ReduceOp.SUM, v_dense, v_dense, options)
          i += 1
        # reduce sparse value in tf.while_loop.
        i = array_ops.identity(1)
        while i < 3:
          collective.reduce(
              reduce_util.ReduceOp.SUM, v_sparse, v_sparse, options
          )
          i += 1
        # reducing dense and sparse value again.
        collective.reduce(reduce_util.ReduceOp.SUM, v_dense, v_dense, options)
        collective.reduce(reduce_util.ReduceOp.SUM, v_sparse, v_sparse, options)
        # pylint: enable=pointless-statement

      graph = f.get_concrete_function().graph
      should_be_ordered = set([
          "CollectiveReduceV2",
          "CollectiveGatherV2",
          "If",
          "While",
          "StatefulPartitionedCall",
      ])
      nodes_by_device = {}
      for op in graph.get_operations():
        if op.type in should_be_ordered:
          if op.device not in nodes_by_device:
            nodes_by_device[op.device] = []
          nodes_by_device[op.device].append(op)
      order = test_util.topological_sort_operations(graph.get_operations())
      for device in devices:
        device = device_util.canonicalize(device)
        # Those function ops don't have device annotations, but they contain
        # collectives for both devices so we always include them.
        operations = nodes_by_device[device] + nodes_by_device[""]
        # Verify that we get all types of nodes we want.
        self.assertEqual(set(op.type for op in operations), should_be_ordered)
        test_util.assert_sequential_execution(order, operations)

    get_global_mpr(num_processes).run(replica_fn)


@_REGISTER_DECORATOR(CollectiveOpsTest)
def _save_test_case(pickler, obj):
  # [FUNCTION DOC]
  # HELPER: custom pickling hook registered (when the optional pickling library is available)
  # for the test class so worker processes can receive test-case instances captured by closures.
  # pickler is the pickler object, obj is the test case instance. Saves a reduction that
  # rebuilds a fresh default instance of the test class in the receiving process, dropping the
  # original instance's state. Returns whatever the pickler's save call returns.
  def reconstruct(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: factory used when unpickling. Ignores all arguments and returns a new instance of
    # the test class.
    del args, kwargs
    return CollectiveOpsTest()

  return pickler.save_reduce(reconstruct, (), obj=obj)


if __name__ == "__main__":
  # Set default inter op thread pool size to one to ensure we don't exhaust the
  # thread pool with the additional executors to run collectives in eager.
  os.environ["TF_NUM_INTEROP_THREADS"] = "1"
  # TODO(b/172304955): figure why logical devices doesn't work.
  test_util.main(config_logical_devices=False)
