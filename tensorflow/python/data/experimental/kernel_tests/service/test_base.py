# [MODULE DOC]
# HELPER: shared test infrastructure for the tf.data service kernel tests.
# Defines constants (placeholder work directory, no-work-directory marker, short heartbeat and
# timeout values, default protocol), a function listing cluster configurations, a worker wrapper,
# an in-process cluster of a dispatcher plus workers, a base test class with helpers for building
# distributed and coordinated-read datasets, and a temporary directory helper. No import-time
# side effects other than defining these names.
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
"""Test base for tf.data service tests."""

import os
import shutil
import tempfile

from tensorflow.core.protobuf import service_config_pb2
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest

# This will be resolved to a tmp directory by `start_dispatch_server`.
TMP_WORK_DIR = "tmp_work_dir_placeholder"
# `""` indicates not to use a work directory.
NO_WORK_DIR = ""
# We use a faster than normal heartbeat interval so that tests run faster.
TEST_HEARTBEAT_INTERVAL_MS = 100
TEST_DISPATCHER_TIMEOUT_MS = 5000
TEST_WORKER_TIMEOUT_MS = 200
TEST_JOB_GC_CHECK_INTERNAL_MS = 1000
TEST_SNAPSHOT_MAX_CHUNK_SIZE_BYTES = 16 << 10  # 16 KB
PROTOCOL = "grpc"


def all_cluster_configurations():
  # [FUNCTION DOC]
  # HELPER: returns test parameter combinations describing cluster setups: a temporary work
  # directory with fault tolerance on or off, plus no work directory with fault tolerance off.
  # Takes no inputs; returns the joined list of combinations.
  with_work_dir = combinations.combine(
      work_dir=TMP_WORK_DIR, fault_tolerant_mode=[True, False])
  without_work_dir = combinations.combine(
      work_dir=NO_WORK_DIR, fault_tolerant_mode=False)
  return with_work_dir + without_work_dir


def _make_worker(
    dispatcher_address,
    protocol,
    data_transfer_protocol,
    shutdown_quiet_period_ms=0,
    port=0,
    worker_tags=None,
    cross_trainer_cache_size_bytes=None,
    snapshot_max_chunk_size_bytes=TEST_SNAPSHOT_MAX_CHUNK_SIZE_BYTES,
):
  # [FUNCTION DOC]
  # HELPER: builds (without starting) a data service worker server. Parameters: dispatcher
  # address; protocol; data transfer protocol (may be nothing); shutdown quiet period in
  # milliseconds (default 0); port (0 picks one); worker tags (optional); cross trainer cache
  # size in bytes (optional); max snapshot chunk size in bytes (defaults to the test constant).
  # Uses fast test heartbeat and dispatcher timeout settings and default addresses derived from
  # the default worker configuration. Returns the unstarted worker server.
  """Creates a worker server."""
  defaults = server_lib.WorkerConfig(dispatcher_address=dispatcher_address)
  config_proto = service_config_pb2.WorkerConfig(
      dispatcher_address=dispatcher_address,
      worker_address=defaults.worker_address,
      port=port,
      protocol=protocol,
      worker_tags=worker_tags,
      heartbeat_interval_ms=TEST_HEARTBEAT_INTERVAL_MS,
      dispatcher_timeout_ms=TEST_DISPATCHER_TIMEOUT_MS,
      data_transfer_protocol=data_transfer_protocol,
      data_transfer_address=defaults.data_transfer_address,
      shutdown_quiet_period_ms=shutdown_quiet_period_ms,
      cross_trainer_cache_size_bytes=cross_trainer_cache_size_bytes,
      snapshot_max_chunk_size_bytes=snapshot_max_chunk_size_bytes,
  )
  return server_lib.WorkerServer(config_proto, start=False)


# pylint: disable=protected-access
class TestWorker:
  # [CLASS DOC]
  # HELPER: thin wrapper around a data service worker server for tests. Remembers the dispatcher
  # address, protocols, quiet period, port and running flag, and offers start, stop, restart,
  # join and introspection (task count, snapshot progress, address).
  """A tf.data service worker."""

  def __init__(
      self,
      dispatcher_address,
      shutdown_quiet_period_ms,
      protocol=PROTOCOL,
      data_transfer_protocol=None,
      port=0,
      worker_tags=None,
      cross_trainer_cache_size_bytes=None,
      snapshot_max_chunk_size_bytes=TEST_SNAPSHOT_MAX_CHUNK_SIZE_BYTES,
  ):
    # [FUNCTION DOC]
    # Constructor. Parameters: dispatcher address; shutdown quiet period in ms; protocol
    # (default grpc); data transfer protocol (optional); port (0 picks); worker tags; cross
    # trainer cache size; snapshot chunk size limit. Creates the server without starting it and
    # marks the worker as not running.
    self._dispatcher_address = dispatcher_address
    self._shutdown_quiet_period_ms = shutdown_quiet_period_ms
    self._server = _make_worker(
        dispatcher_address,
        protocol,
        data_transfer_protocol,
        shutdown_quiet_period_ms,
        port=port,
        worker_tags=worker_tags,
        cross_trainer_cache_size_bytes=cross_trainer_cache_size_bytes,
        snapshot_max_chunk_size_bytes=snapshot_max_chunk_size_bytes,
    )
    self._running = False
    self._protocol = protocol
    self._data_transfer_protocol = data_transfer_protocol

  def stop(self):
    # [FUNCTION DOC]
    # Stops the underlying server and marks the worker as not running.
    self._server._stop()
    self._running = False

  def start(self):
    # [FUNCTION DOC]
    # Starts the server, records the port it actually bound (parsed from its address) and marks
    # the worker as running.
    self._server.start()
    self._port = int(self._server._address.split(":")[1])
    self._running = True

  def restart(self, use_same_port=True):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the new server is built without the original worker tags, cache size or
    # snapshot chunk size settings, so a restarted worker silently uses defaults for those.
    # Parameter: use_same_port (default true) reuses the previously bound port, otherwise any
    # free port is chosen. Stops the worker if running, creates and starts a new server and
    # updates the stored port. Returns nothing.
    """Restarts the worker, stopping it first if it is already running."""
    if self._running:
      self.stop()
    port = 0
    if use_same_port:
      port = self._port
    self._server = _make_worker(self._dispatcher_address,
                                self._protocol,
                                self._data_transfer_protocol,
                                self._shutdown_quiet_period_ms, port)
    self._server.start()
    self._port = int(self._server._address.split(":")[1])
    self._running = True

  def join(self):
    # [FUNCTION DOC]
    # Blocks until the worker server terminates.
    self._server.join()

  def num_tasks(self):
    # [FUNCTION DOC]
    # Returns the number of tasks currently on this worker.
    return self._server._num_tasks()

  def snapshot_task_progresses(self):
    # [FUNCTION DOC]
    # Returns the progress information of the snapshot tasks running on this worker.
    return self._server._snapshot_task_progresses()

  def worker_address(self):
    # [FUNCTION DOC]
    # Returns the address (host and port text) of the worker server.
    return self._server._address


class TestCluster:
  # [CLASS DOC]
  # HELPER: in-process tf.data service cluster for tests, made of one dispatcher server and a
  # list of test workers. Remembers the settings needed to recreate the dispatcher. Offers
  # adding, starting, stopping and restarting workers or the dispatcher, and queries about
  # registered workers, tasks and snapshot streams. Cleaned up when destroyed.
  """Test tf.data service cluster."""

  def __init__(
      self,
      num_workers,
      dispatcher_port=0,
      work_dir=TMP_WORK_DIR,
      fault_tolerant_mode=True,
      job_gc_check_interval_ms=TEST_JOB_GC_CHECK_INTERNAL_MS,
      job_gc_timeout_ms=None,
      worker_timeout_ms=TEST_WORKER_TIMEOUT_MS,
      worker_shutdown_quiet_period_ms=0,
      snapshot_max_chunk_size_bytes=TEST_SNAPSHOT_MAX_CHUNK_SIZE_BYTES,
      worker_max_concurrent_snapshots=0,
      start=True,
      protocol=PROTOCOL,
      data_transfer_protocol=None,
  ):
    # [FUNCTION DOC]
    # Constructor; see the docstring for the parameters. When the work directory equals the
    # temporary placeholder, a fresh temporary directory is created. Stores the settings, builds
    # the dispatcher (started only when start is true) and adds the requested number of workers
    # (started when start is true). Starts servers as a side effect.
    """Creates a tf.data service test cluster.

    Args:
      num_workers: The number of workers to initially add to the cluster.
      dispatcher_port: The port to use for the dispatcher.
      work_dir: The work directory to use for the dispatcher. If set to
        `TMP_WORK_DIR`, the cluster will create a new temporary directory to use
        as the work directory. If set to `NO_WORK_DIR`, no work directory will
        be used.
      fault_tolerant_mode: Whether the dispatcher should write its state to a
        journal so that it can recover from restarts.
      job_gc_check_interval_ms: How often the dispatcher should scan through to
        delete old and unused jobs, in milliseconds.
      job_gc_timeout_ms: How long a job needs to be unused before it becomes a
        candidate for garbage collection, in milliseconds.
      worker_timeout_ms: How long to wait for a worker to heartbeat before
        considering it missing, in milliseconds.
      worker_shutdown_quiet_period_ms: When shutting down a worker, how long to
        wait for the gRPC server to process the final requests.
      snapshot_max_chunk_size_bytes: The maximum size of a distributed snapshot
        chunk file.
      worker_max_concurrent_snapshots: The maximum number of snapshots a worker
        can concurrently process.
      start: Whether to immediately start the servers in the cluster. If
        `False`, the servers can be started later by calling
        `start_dispatcher()` and `start_workers()`.
      protocol: The protocol to use for communicating with the tf.data service,
        e.g. "grpc".
      data_transfer_protocol: (Optional.) The protocol to use for transferring
        data with the tf.data service.
    """
    if work_dir == TMP_WORK_DIR:
      work_dir = tempfile.mkdtemp(dir=googletest.GetTempDir())
    self._worker_shutdown_quiet_period_ms = worker_shutdown_quiet_period_ms
    self._snapshot_max_chunk_size_bytes = snapshot_max_chunk_size_bytes
    self._protocol = protocol
    self._data_transfer_protocol = data_transfer_protocol
    self._job_gc_check_interval_ms = job_gc_check_interval_ms
    self._job_gc_timeout_ms = job_gc_timeout_ms
    self._worker_timeout_ms = worker_timeout_ms
    self._worker_max_concurrent_snapshots = worker_max_concurrent_snapshots
    self.dispatcher = server_lib.DispatchServer(
        server_lib.DispatcherConfig(
            port=dispatcher_port,
            work_dir=work_dir,
            protocol=protocol,
            fault_tolerant_mode=fault_tolerant_mode,
            job_gc_check_interval_ms=job_gc_check_interval_ms,
            job_gc_timeout_ms=job_gc_timeout_ms,
            worker_timeout_ms=worker_timeout_ms,
            worker_max_concurrent_snapshots=worker_max_concurrent_snapshots,
        ),
        start=start,
    )

    self.workers = []
    for _ in range(num_workers):
      self.add_worker(start=start)

  def dispatcher_address(self):
    # [FUNCTION DOC]
    # Returns the dispatcher address (host and port) obtained by dropping the protocol prefix of
    # the dispatcher target.
    return self.dispatcher.target.split("://")[1]

  def add_worker(self, start=True):
    # [FUNCTION DOC]
    # Parameter: start (default true) whether to start the new worker. Creates a worker connected
    # to this cluster's dispatcher with the cluster's protocols and chunk size limit, optionally
    # starts it, and appends it to the worker list.
    worker = TestWorker(
        self.dispatcher_address(),
        self._worker_shutdown_quiet_period_ms,
        self._protocol,
        self._data_transfer_protocol,
        snapshot_max_chunk_size_bytes=self._snapshot_max_chunk_size_bytes,
    )
    if start:
      worker.start()
    self.workers.append(worker)

  def start_dispatcher(self):
    # [FUNCTION DOC]
    # Starts the dispatcher server (for clusters created without starting).
    self.dispatcher.start()

  def start_workers(self):
    # [FUNCTION DOC]
    # Starts every worker in the cluster.
    for worker in self.workers:
      worker.start()

  def stop_dispatcher(self):
    # [FUNCTION DOC]
    # Stops the dispatcher server.
    # pylint: disable=protected-access
    self.dispatcher._stop()

  def restart_worker(self, index):
    # [FUNCTION DOC]
    # Restarts the worker at the given position in the worker list (same port).
    self.workers[index].restart()

  def stop_worker(self, index):
    # [FUNCTION DOC]
    # Stops the worker at the given position in the worker list.
    self.workers[index].stop()

  def stop_workers(self):
    # [FUNCTION DOC]
    # Stops all workers in the cluster.
    for worker in self.workers:
      worker.stop()

  # pylint: disable=protected-access
  def restart_dispatcher(self):
    # [FUNCTION DOC]
    # Stops the dispatcher and creates a new one on the same port with the same work directory
    # and settings (the docstring explains it needs fault tolerance). Raises a value error if
    # the dispatcher was not configured with fault tolerant mode. The new dispatcher replaces the
    # old one and is started immediately.
    """Stops `dispatcher` and creates a new dispatcher with the same port.

    Restarting is supported only when the dispatcher is configured with
    `fault_tolerant_mode=True`.
    """
    if not self.dispatcher._config.fault_tolerant_mode:
      raise ValueError(
          "Trying to restart the dispatcher without fault-tolerance.")
    port = int(self.dispatcher_address().split(":")[1])
    self.dispatcher._stop()
    self.dispatcher = server_lib.DispatchServer(
        server_lib.DispatcherConfig(
            port=port,
            work_dir=self.dispatcher._config.work_dir,
            protocol=self._protocol,
            fault_tolerant_mode=self.dispatcher._config.fault_tolerant_mode,
            job_gc_check_interval_ms=self._job_gc_check_interval_ms,
            job_gc_timeout_ms=self._job_gc_timeout_ms,
            worker_timeout_ms=self._worker_timeout_ms,
            worker_max_concurrent_snapshots=
            self._worker_max_concurrent_snapshots,
        )
    )

  def num_registered_workers(self):
    # [FUNCTION DOC]
    # Returns how many workers the dispatcher currently has registered.
    return self.dispatcher._num_workers()

  def num_tasks_on_workers(self):
    # [FUNCTION DOC]
    # Returns the total number of tasks across all workers of the cluster.
    return sum(worker.num_tasks() for worker in self.workers)

  def snapshot_streams(self, path):
    # [FUNCTION DOC]
    # Parameter: path of a snapshot directory. Returns the dispatcher's information about the
    # streams of that snapshot.
    return self.dispatcher._snapshot_streams(path)

  def __del__(self):
    # [FUNCTION DOC]
    # Destructor: clears the worker list first and then removes the dispatcher reference so
    # shutdown happens in a clean order.
    # Destroy workers before the dispatcher for clean shutdown.
    self.workers.clear()
    del self.dispatcher


class TestBase(test_base.DatasetTestBase):
  # [CLASS DOC]
  # Base class for tf.data service tests, extending the dataset test base. Holds the default data
  # transfer protocol and default compression applied by its dataset building helpers, and
  # provides helpers to make clusters, distributed datasets, coordinated-read datasets and to
  # validate coordinated read results.
  """Base class for tf.data service tests."""

  def setUp(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: does not call the parent setup, so base class fixture setup is skipped.
    # Initializes the default data transfer protocol to nothing and default compression to automatic.
    self.default_data_transfer_protocol = None
    self.default_compression = "AUTO"

  def set_default_data_transfer_protocol(self, protocol):
    # [FUNCTION DOC]
    # Sets the data transfer protocol used by default for clusters and distributed datasets made
    # through this test.
    self.default_data_transfer_protocol = protocol

  def set_default_compression(self, compression):
    # [FUNCTION DOC]
    # Sets the compression setting used by default for distributed datasets made through this test.
    self.default_compression = compression

  def make_test_cluster(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Creates a test cluster passing all arguments through; if no data transfer protocol is given
    # the test's default is used. Returns the cluster.
    if "data_transfer_protocol" not in kwargs:
      kwargs["data_transfer_protocol"] = self.default_data_transfer_protocol
    return TestCluster(*args, **kwargs)

  def make_distributed_dataset(self,
                               dataset,
                               cluster,
                               processing_mode="parallel_epochs",
                               **kwargs):
    # [FUNCTION DOC]
    # Parameters: dataset to distribute; cluster (provides the dispatcher address); processing
    # mode (defaults to parallel epochs; may be a sharding policy); extra keyword options passed
    # to the distribute operation. Forces a short task refresh interval hint (20 ms) and fills in
    # the default data transfer protocol and default compression when not supplied. Returns the
    # dataset with the service distribution applied.
    kwargs["task_refresh_interval_hint_ms"] = 20
    if "data_transfer_protocol" not in kwargs:
      kwargs["data_transfer_protocol"] = self.default_data_transfer_protocol
    if "compression" not in kwargs:
      kwargs["compression"] = self.default_compression

    # pylint: disable=protected-access
    return dataset.apply(
        data_service_ops._distribute(
            processing_mode,
            cluster.dispatcher_address(),
            **kwargs))

  def make_distributed_range_dataset(self,
                                     num_elements,
                                     cluster,
                                     **kwargs):
    # [FUNCTION DOC]
    # Parameters: number of elements; cluster; extra options forwarded to the distributed dataset
    # builder. Creates a range dataset and returns it distributed through the cluster.
    dataset = dataset_ops.Dataset.range(num_elements)
    return self.make_distributed_dataset(dataset, cluster, **kwargs)

  def make_coordinated_read_dataset(
      self,
      cluster,
      num_consumers,
      sharding_policy=data_service_ops.ShardingPolicy.OFF):
    # [FUNCTION DOC]
    # Builds a dataset simulating the given number of consumers doing coordinated reads against a
    # cluster (see the docstring). The source starts with a zero then random numbers; each
    # number x (taken modulo a power of two to avoid overflow) expands to a run of consecutive
    # values of group size. One distributed dataset per consumer (job named "test") is created
    # and all are read in parallel by interleaving. Raises a value error for any sharding policy
    # other than off or dynamic. Returns the interleaved dataset.
    """Creates a dataset that performs coordinated reads.

    The dataset simulates `num_consumers` consumers by using parallel
    interleave to read with `num_consumers` threads, one for each consumer. The
    nth element of the dataset is produced by consumer `n % num_consumers`.

    The dataset executed on each worker will produce groups of `num_consumers`
    sequentially increasing numbers. For example, if `num_consumers=3` a worker
    dataset could produce [0, 1, 2, 9, 10, 11, 21, 22, 23]. This enables
    `checkCoordinatedReadGroups` below to assess whether the values received in
    each step came from the same group.

    Args:
      cluster: A tf.data service `TestCluster`.
      num_consumers: The number of consumers to simulate.
      sharding_policy: The sharding policy to use. Currently only OFF and
        DYNAMIC are supported.

    Returns:
      A dataset that simulates reading with `num_consumers` consumers.
    """
    if sharding_policy not in [
        data_service_ops.ShardingPolicy.OFF,
        data_service_ops.ShardingPolicy.DYNAMIC
    ]:
      raise ValueError(f"Unsupported sharding policy: {sharding_policy}")
    # Start from 0 so that we can detect when a new worker is added with
    # ShardingPolicy.OFF.
    ds = dataset_ops.Dataset.from_tensors(math_ops.cast(0, dtypes.int64))
    ds = ds.concatenate(dataset_ops.Dataset.random())
    # Ensure that all elements in the same group are consecutive.
    def make_group(x):
      # [FUNCTION DOC]
      # Flat-map function: reduces x modulo a power of two (preventing integer overflow) and
      # returns a range dataset of num_consumers consecutive integers starting at that value
      # times num_consumers.
      # Avoid overflowing an int64 in (x+1)*num_consumers below.
      x = x % (2**32)
      return dataset_ops.Dataset.range(x*num_consumers, (x+1)*num_consumers)
    ds = ds.flat_map(make_group)
    consumers = []
    for consumer_index in range(num_consumers):
      consumers.append(
          self.make_distributed_dataset(
              ds,
              cluster,
              job_name="test",
              processing_mode=sharding_policy,
              consumer_index=consumer_index,
              num_consumers=num_consumers))
    # Use parallel interleave to read from consumers in parallel.
    ds = dataset_ops.Dataset.from_tensor_slices(consumers)
    ds = ds.interleave(
        lambda x: x,
        cycle_length=num_consumers,
        num_parallel_calls=num_consumers)
    return ds

  def checkCoordinatedReadGroups(self, results, num_consumers):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring names a misspelled dataset builder ("coordinted"); the real one
    # is the coordinated read dataset maker.
    # Parameters: list of produced elements; number of consumers (group size). Splits results into
    # consecutive groups of that size and asserts each group is made of consecutive integers
    # (same worker), failing with a message listing bad groups and all groups.
    """Validates results from a `make_coordinted_read_dataset` dataset.

    Each group of `num_consumers` results should be consecutive, indicating that
    they were produced by the same worker.

    Args:
      results: The elements produced by the dataset.
      num_consumers: The number of consumers.
    """
    groups = [
        results[start:start + num_consumers]
        for start in range(0, len(results), num_consumers)
    ]
    incorrect_groups = []
    for group in groups:
      # Check that each group of `num_consumers` results are consecutive.
      for offset in range(1, len(group)):
        if group[0] + offset != group[offset]:
          incorrect_groups.append(group)
          break
    self.assertEmpty(
        incorrect_groups,
        "Incorrect groups: {}.\nAll groups: {}".format(incorrect_groups,
                                                       groups))

  def read(self, get_next, results, count):
    # [FUNCTION DOC]
    # HELPER: calls the get-next callable count times, evaluating each result and appending it to
    # the results list (mutated in place). Returns nothing.
    for _ in range(count):
      results.append(self.evaluate(get_next()))


class TempDir:
  # [CLASS DOC]
  # HELPER: temporary directory holder for snapshot tests. Stores the path of a snapshot
  # subdirectory name inside a fresh temp directory; usable as a path-like object; the directory
  # tree is removed when the object is destroyed.
  """Temporary directory for unit testing."""

  def __init__(self):
    # [FUNCTION DOC]
    # Creates a fresh temporary directory under the test temp location and stores the path of a
    # snapshot subdirectory name inside another newly created directory in it. Creates directories.
    temp_dir = tempfile.mkdtemp(dir=googletest.GetTempDir())
    self._path = os.path.join(
        tempfile.mkdtemp(dir=temp_dir), "tf_data_snapshot")

  @property
  def full_path(self) -> str:
    # [FUNCTION DOC]
    # Property returning the stored snapshot path as text.
    return self._path

  def __fspath__(self) -> str:
    # [FUNCTION DOC]
    # Path-like protocol hook: returns the stored snapshot path as text.
    return self._path

  def __del__(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: only the snapshot path is removed; the parent temp directories created in the
    # constructor are left behind.
    # Destructor: recursively deletes the snapshot path, ignoring a missing path.
    try:
      shutil.rmtree(self.full_path)
    except FileNotFoundError:
      pass
