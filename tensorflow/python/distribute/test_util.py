# [MODULE DOC]
# HELPER: helper module of shared utilities for tf.distribute tests.
# Provides: a cluster-parameter container and lazy in-process cluster definition, gather and
# per-replica construction helpers, logical device setup, an all-in-one test main, graph
# topological-sort and sequential-execution assertions, thread-leak helpers, an objgraph-based
# back-reference dump, TPU strategy detection, and eager context reset. Import-time side
# effect: tries to import the optional objgraph library, falling back to nothing if missing.
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
"""Test utilities."""

import collections
import dataclasses
import functools
import io
import itertools
import threading

from absl import app

from tensorflow.python.compat import v2_compat
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import multi_process_runner
from tensorflow.python.distribute import multi_worker_test_base
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute import values
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.util import nest

try:
  import objgraph  # pylint:disable=g-import-not-at-top
except ImportError:
  objgraph = None


@dataclasses.dataclass
class TestClusterParams:
  # [CLASS DOC]
  # Plain data container (dataclass) describing a reusable in-process test cluster: the cluster
  # specification mapping (or nothing/null until lazily created), the maximum number of worker
  # servers and the maximum number of parameter-server servers it can provide. Mutated by
  # get_cluster_def, which fills in the cluster on first use.
  cluster: dict
  max_num_worker: int
  max_num_ps: int


def get_cluster_def(cluster_params, num_workers, num_ps):
  # [FUNCTION DOC]
  # HELPER: test helper that hands out a cluster definition from a lazily created shared cluster.
  # Inputs: cluster_params (a TestClusterParams holding the cluster and size limits), num_workers
  # and num_ps (integers, how many worker and parameter-server tasks the caller wants).
  # Raises a value error if either requested count exceeds the maximum in cluster_params. On
  # first use creates an in-process cluster with the maximum numbers of workers and parameter
  # servers and stores it back into cluster_params (side effect: starts servers, mutates the
  # argument). Returns a mapping with a worker entry and a ps entry listing the first
  # num_workers worker addresses and first num_ps parameter-server addresses.
  if (num_workers > cluster_params.max_num_worker or
      num_ps > cluster_params.max_num_ps):
    raise ValueError("Requesting more servers than the maximum, adjust"
                     "cluster params' max_num_ps and max_num_worker")
  if cluster_params.cluster is None:
    cluster_params.cluster = multi_worker_test_base.create_in_process_cluster(
        num_workers=cluster_params.max_num_worker,
        num_ps=cluster_params.max_num_ps)
  return {
      "worker": cluster_params.cluster["worker"][:num_workers],
      "ps": cluster_params.cluster["ps"][:num_ps],
  }


def gather(strategy, value):
  # [FUNCTION DOC]
  # HELPER: test-only stand-in for an all-gather API.
  # Applies the single-value gather helper to every leaf of the (possibly nested) value
  # structure, using the given strategy. Inputs: strategy (a distribution strategy) and value
  # (nested structure of per-replica values or tensors). Returns a structure of the same shape
  # whose leaves are tensors with one extra leading dimension holding the replica/worker values.
  # No side effects.
  """Gathers value from all workers.

  This is intended for tests before we implement an official all-gather API.

  Args:
    strategy: a `tf.distribute.Strategy`.
    value: a nested structure of n-dim `tf.distribute.DistributedValue` of
      `tf.Tensor`, or of a `tf.Tensor` if the strategy only has one replica.
      Cannot contain tf.sparse.SparseTensor.

  Returns:
    a (n+1)-dim `tf.Tensor`.
  """
  return nest.map_structure(functools.partial(_gather, strategy), value)


def _gather(strategy, value):
  # [FUNCTION DOC]
  # HELPER: single-leaf implementation behind gather.
  # If value is not a distributed value it is wrapped as a one-replica per-replica value after
  # converting to a tensor. For strategies that are not the collective all-reduce kind, stacks
  # the per-replica component values along a new first axis. For collective all-reduce, asserts
  # the number of local worker devices equals the number of components, adds a leading axis of
  # size one to each, and calls the strategy gather along axis zero across workers. Returns the
  # gathered tensor. Uses private attributes of distributed values.
  """Gathers a single value."""
  # pylint: disable=protected-access
  if not isinstance(value, values.DistributedValues):
    value = values.PerReplica([ops.convert_to_tensor(value)])
  if not isinstance(strategy.extended,
                    collective_all_reduce_strategy.CollectiveAllReduceExtended):
    return array_ops_stack.stack(value._values)
  assert len(strategy.extended.worker_devices) == len(value._values)
  inputs = [array_ops.expand_dims_v2(v, axis=0) for v in value._values]
  return strategy.gather(values.PerReplica(inputs), axis=0)
  # pylint: enable=protected-access


def set_logical_devices_to_at_least(device, num):
  # [FUNCTION DOC]
  # HELPER: test helper that splits physical devices into enough logical devices.
  # Inputs: device (device type name such as GPU or CPU) and num (integer, minimum number of
  # logical devices wanted). Raises a value error if num is below one and a runtime error if no
  # physical device of that type exists. Returns nothing; does nothing if enough physical
  # devices already exist. Otherwise configures the last physical device to be split into the
  # needed number of logical devices (GPU logical devices get a 2048 megabyte memory limit) so
  # that the total reaches num. Side effect: changes global logical device configuration, which
  # must happen before the runtime is initialized.
  """Create logical devices of at least a given number."""
  if num < 1:
    raise ValueError("`num` must be at least 1 not %r" % (num,))
  physical_devices = config.list_physical_devices(device)
  if not physical_devices:
    raise RuntimeError("No {} found".format(device))
  if len(physical_devices) >= num:
    return
  # By default each physical device corresponds to one logical device. We create
  # multiple logical devices for the last physical device so that we have `num`
  # logical devices.
  num = num - len(physical_devices) + 1
  logical_devices = []
  for _ in range(num):
    if device.upper() == "GPU":
      logical_devices.append(
          context.LogicalDeviceConfiguration(memory_limit=2048))
    else:
      logical_devices.append(context.LogicalDeviceConfiguration())
  # Create logical devices from the last device since sometimes the first GPU
  # is the primary graphic card and may have less memory available.
  config.set_logical_device_configuration(physical_devices[-1], logical_devices)


def _set_logical_devices():
  # [FUNCTION DOC]
  # HELPER: callback used by main to create two logical devices.
  # If any physical GPU exists, ensures at least two logical GPUs; if any physical CPU exists,
  # ensures at least two logical CPUs. Takes no arguments, returns nothing; changes global device
  # configuration.
  if config.list_physical_devices("GPU"):
    set_logical_devices_to_at_least("GPU", 2)
  if config.list_physical_devices("CPU"):
    set_logical_devices_to_at_least("CPU", 2)


def main(enable_v2_behavior=True, config_logical_devices=True):
  # [FUNCTION DOC]
  # HELPER: shared entry point for tf.distribute test files.
  # Inputs: enable_v2_behavior (boolean, default true; chooses between enabling and disabling
  # TF2 behavior) and config_logical_devices (boolean, default true; when set registers a
  # callback to run after application init that creates multiple logical devices). Then runs the
  # multi-process-runner test main, which runs the tests (and handles subprocess execution).
  # Side effects: changes global v2 behavior flags and registers an init callback. Returns
  # whatever the test main does (normally exits the process).
  """All-in-one main function for tf.distribute tests."""
  if config_logical_devices:
    app.call_after_init(_set_logical_devices)
  if enable_v2_behavior:
    v2_compat.enable_v2_behavior()
  else:
    v2_compat.disable_v2_behavior()
  multi_process_runner.test_main()


def _op_dependencies(op):
  # [FUNCTION DOC]
  # HELPER: graph-analysis utility for topological_sort_operations and the dependency check.
  # Input: an operation. Returns a list of the operations it depends on: producers of its input
  # tensors followed by its control-input operations. Asserts each control input is an operation.
  """Returns the data and control dependencies of a tf.Operation combined."""
  deps = []
  for node in itertools.chain(op.inputs, op.control_inputs):
    if isinstance(node, tensor.Tensor):
      node = node.op
    assert isinstance(node, ops.Operation)
    deps.append(node)
  return deps


def topological_sort_operations(operations):
  # [FUNCTION DOC]
  # HELPER: test utility for checking execution ordering in graphs.
  # Input: operations, a list of operations from one graph. Computes in-degree counts over the
  # reversed list and their dependency edges (an edge points from an operation to what it
  # depends on), then repeatedly takes operations with zero in-degree from a first-in first-out
  # queue. Returns a mapping from each operation to its topological position (zero based).
  # Ties are intentionally ordered unstably. Dependencies not in the input list are also
  # counted, so the final assertion that the result covers exactly the given operations fails
  # (assertion error) if dependencies lie outside the list or the graph has a cycle.
  """Topological sorts a list of operations.

  This does a topological sort of the operations in a graph. The edges include
  both data dependencies and control dependencies. Note that the edge goes from
  an operation to its dependencies.

  The sort is intentionally unstable, reversing orders of operations and
  dependencies on ties.

  Args:
    operations: a list of tf.Operation in the same graph.

  Returns:
    A map from a tf.Operation to its topological order.
  """
  in_degrees = collections.OrderedDict()
  for op in reversed(operations):
    if op not in in_degrees:
      in_degrees[op] = 0
    for next_op in reversed(_op_dependencies(op)):
      in_degrees[next_op] = in_degrees.get(next_op, 0) + 1
  nexts = []
  for op, in_degree in in_degrees.items():
    if in_degree == 0:
      nexts.append(op)
  order = {}
  next_order = 0
  while nexts:
    op, nexts = nexts[0], nexts[1:]
    order[op] = next_order
    next_order += 1
    for next_op in reversed(_op_dependencies(op)):
      in_degrees[next_op] -= 1
      if in_degrees[next_op] == 0:
        nexts.append(next_op)
  assert len(order) == len(operations)
  return order


def _exists_dependency(start, end):
  # [FUNCTION DOC]
  # HELPER: reachability check used by assert_sequential_execution.
  # Breadth-first walks the dependency edges from start; returns true if end is reached, false
  # otherwise. Inputs are two operations.
  """Returns whether there exists a dependency chain from start to end."""
  nexts = [start]
  while nexts:
    op, nexts = nexts[0], nexts[1:]
    for next_op in _op_dependencies(op):
      if next_op == end:
        return True
      nexts.append(next_op)
  return False


def assert_sequential_execution(order, operations):
  # [FUNCTION DOC]
  # HELPER: test assertion that a set of operations forms a deterministic chain.
  # Inputs: order (mapping from operation to topological position, as produced by
  # topological_sort_operations) and operations (list, any order). Sorts the operations by
  # position and requires a dependency path between every adjacent pair. On failure prints the
  # whole graph definition to standard output and raises an assertion error naming the two
  # operations. Returns nothing.
  """Asserts there's a deterministic execution order between the operations.

  Args:
    order: a map from a tf.Operation to its topological order.
    operations: a list of operations that should be executed sequentially. It
      can be given in any order.
  """
  # Topological ordering guarantees that, if there's a dependency from N_a to
  # N_b, then order[N_a] < order[N_b]. If there do exist a path of dependencies
  # among the operations, it always goes from a operation with a smaller
  # topological order to one with a larger topological order. Therefore, we only
  # need to sort the operations by their topological orders, and verify that
  # there's a path of dependency between adjacent pairs.
  operations = sorted(operations, key=lambda op: order[op])
  for i in range(len(operations) - 1):
    if not _exists_dependency(operations[i], operations[i + 1]):
      print(operations[i].graph.as_graph_def())
      raise AssertionError(
          "No dependency between {} and {}. Graph is dumped to stdout.".format(
              operations[i].name, operations[i + 1].name))


def get_running_threads():
  # [FUNCTION DOC]
  # HELPER: thread-leak test utility. Returns a set with the names of all currently alive
  # threads in the process (threads whose name is nothing are skipped, which in practice never
  # happens).
  """Returns a set of all running thread names."""
  running_threads = set()
  for thread in threading.enumerate():
    if thread.name is not None:
      running_threads.add(thread.name)
  return running_threads


def has_thread(prefix, running_threads):
  # [FUNCTION DOC]
  # HELPER: thread-leak test utility. Inputs: prefix (string) and running_threads (collection of
  # thread-name strings). Returns true if any name starts with the prefix, else false.
  """Returns whether any 'running_threads' is prefixed with 'prefix'.

  Args:
    prefix: The prefix of the expected thread name.
    running_threads: A collection of the running thread names.
  """
  for thread in running_threads:
    if thread.startswith(prefix):
      return True
  return False


def show_backref(target, max_depth=3):
  # [FUNCTION DOC]
  # HELPER: debugging aid for memory-leak and reference-cycle investigations.
  # Inputs: target (any object) and max_depth (integer, default three, how many levels of
  # referrers to follow). Uses the optional objgraph library to render the back-reference graph
  # into an in-memory text buffer and returns the resulting dot-format string. Raises a
  # not-implemented error if objgraph could not be imported.
  """Returns a dot graph of all the objects that are referencing the target.

  A object referencing graph is useful to debug memory leak like circular
  reference. objgraph provides a good visualization of the memory graph than
  most python built-in utilities like gc.get_referrers(), which are not
  human-readable sometimes.

  The dot graph will be written to a string IO object, and can be rendered with
  graphviz in operating system.
  E.g. dot -Tpng {$dot_graph} -o output.png
  Args:
    target: The target object for the memory graph.
    max_depth: The maximum depth of the graph. By default 3 layers of references
      are used. Increases this a lot may result in the graph growing too big.

  Returns:
    A string that contains the object reference graph.
  Raises:
    NotImplementedError: if objgraph is not installed.
  """
  if objgraph is None:
    raise NotImplementedError("objgraph is not installed.")
  string_io = io.StringIO()
  objgraph.show_backrefs(target, max_depth=max_depth, output=string_io)
  graph = string_io.getvalue()
  string_io.close()
  return graph


def create_per_replica(strategy, value_list):
  # [FUNCTION DOC]
  # HELPER: test utility that builds a per-replica value with a distinct tensor per device.
  # Inputs: strategy (distribution strategy) and value_list (list with one value per worker
  # device). Raises a value error if the lengths differ. Converts each value to a tensor placed
  # on the matching worker device and returns a per-replica value wrapping those tensors.
  """Creates a PerReplica of Tensors from the value_list."""
  if len(strategy.extended.worker_devices) != len(value_list):
    raise ValueError(
        "the length of values must be the same as the number of worker devices")
  tensors = []
  for device, value in zip(strategy.extended.worker_devices, value_list):
    with ops.device(device):
      tensors.append(ops.convert_to_tensor(value))
  return values.PerReplica(tensors)


def is_tpu_strategy(strategy):
  # [FUNCTION DOC]
  # HELPER: test utility. Returns true if the strategy is an instance of any of the TPU strategy
  # classes (original, V1 or V2), else false.
  """Returns whether the strategy is a TPU strategy."""
  return isinstance(strategy,
                    (tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV1,
                     tpu_strategy.TPUStrategyV2))


def reset_context():
  # [FUNCTION DOC]
  # HELPER: test utility that resets the global eager context by calling the context module's
  # private reset routine. Takes nothing, returns nothing; side effect is discarding eager
  # runtime state (devices, configuration).
  """Resets eager context."""
  context._reset_context()  # pylint: disable=protected-access
