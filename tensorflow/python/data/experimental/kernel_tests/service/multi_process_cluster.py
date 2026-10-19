# [MODULE DOC]
# HELPER: helper module for the tf.data service tests that need both in-process (local) and
# out-of-process (remote) workers.
# Defines a process class that runs one worker server in a child process and a cluster class that
# owns a dispatcher plus local and remote workers, with restart support, and a test main entry
# point. No import-time side effects besides a module constant for the worker shutdown quiet period.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""tf.data service test-cluster with local and remote workers."""

import tempfile

from tensorflow.core.protobuf import data_service_pb2
from tensorflow.core.protobuf import service_config_pb2
from tensorflow.python.data.experimental.kernel_tests.service import test_base as data_service_test_base
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.distribute import multi_process_lib
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest

_WORKER_SHUTDOWN_QUIET_PERIOD_MS = 100


# pylint: disable=protected-access
class _RemoteWorkerProcess(multi_process_lib.Process):
  # [CLASS DOC]
  # HELPER: child process used by the multi-process cluster to host one tf.data service worker.
  # Holds the dispatcher address, port, worker tags and the write end of a pipe through which the
  # started worker's address is sent back to the parent. Running the process starts the worker
  # and blocks until it stops.
  """Runs a worker server in a new process to simulate a remote worker."""

  def __init__(self, dispatcher_address, port, worker_tags, pipe_writer):
    # [FUNCTION DOC]
    # Constructor. Parameters: dispatcher address (host and port text), port the worker should
    # listen on, worker tags (list of strings or nothing), and the writing end of a pipe used to
    # report the worker address. Stores them for use in the child process.
    super(_RemoteWorkerProcess, self).__init__()
    self._dispatcher_address = dispatcher_address
    self._port = port
    self._worker_tags = worker_tags
    self._pipe_writer = pipe_writer

  def run(self):
    # [FUNCTION DOC]
    # Process entry point executed in the child process; simply starts the worker and blocks.
    self.start_worker()

  def start_worker(self):
    # [FUNCTION DOC]
    # Creates and starts a test worker connected to the dispatcher (using the shutdown quiet
    # period constant, the configured port and tags), sends the worker's address through the
    # pipe to the parent, then waits until the worker finishes. Returns nothing.
    self._worker = data_service_test_base.TestWorker(
        self._dispatcher_address,
        _WORKER_SHUTDOWN_QUIET_PERIOD_MS,
        port=self._port,
        worker_tags=self._worker_tags)
    self._worker.start()
    self._pipe_writer.send(self._worker.worker_address())
    self._worker.join()


class MultiProcessCluster:
  # [CLASS DOC]
  # HELPER: test fixture representing a tf.data service cluster with one dispatcher, a number of
  # local workers (in the test process) and remote workers (in child processes). State: a
  # temporary work directory, the deployment mode, the dispatcher server, a list of local
  # workers and a list of (address, process) pairs for remote workers. Created by tests and
  # passed to the dataset distribution helpers. Everything is stopped when the object is
  # destroyed.
  """tf.data service cluster with local and remote workers.

  Represents a cluster with a dispatcher, `num_local_workers` local workers, and
  `num_remote_workers` remote workers. Remote workers run in separate processes.
  This is useful to test reading from local in-process workers. For example:

  ```
  cluster = multi_process_cluster.MultiProcessCluster(
      num_local_workers=1, num_remote_workers=3)
  num_elements = 10
  dataset = self.make_distributed_range_dataset(
      num_elements, cluster, target_workers="LOCAL")
  self.assertDatasetProduces(dataset, list(range(num_elements)))
  ```
  """

  def __init__(self,
               num_local_workers,
               num_remote_workers,
               worker_tags=None,
               worker_addresses=None,
               deployment_mode=data_service_pb2.DEPLOYMENT_MODE_COLOCATED):
    # [FUNCTION DOC]
    # Constructor. Parameters: number of local workers; number of remote workers; optional worker
    # tags applied to all workers; optional list of worker addresses given to the dispatcher;
    # deployment mode (defaults to colocated). Creates a temporary work directory, starts the
    # dispatcher, then the local workers, then the remote workers. Starts servers and processes.
    self._work_dir = tempfile.mkdtemp(dir=googletest.GetTempDir())
    self._deployment_mode = deployment_mode
    self._start_dispatcher(worker_addresses)
    self._start_local_workers(num_local_workers, worker_tags)
    self._start_remote_workers(num_remote_workers, worker_tags)

  def _start_dispatcher(self, worker_addresses, port=0):
    # [FUNCTION DOC]
    # HELPER: starts the dispatcher server. Parameters: list of worker addresses (or nothing) to
    # put in the dispatcher config; port (zero means pick an unused port). Uses grpc, the
    # cluster work directory, fault tolerant mode and the deployment mode, and stores the
    # started server. Returns nothing.
    if port == 0:
      port = test_util.pick_unused_port()
    self._dispatcher = server_lib.DispatchServer(
        service_config_pb2.DispatcherConfig(
            port=port,
            protocol="grpc",
            work_dir=self._work_dir,
            fault_tolerant_mode=True,
            worker_addresses=worker_addresses,
            deployment_mode=self._deployment_mode),
        start=True)

  def _start_local_workers(self, num_workers, worker_tags=None):
    # [FUNCTION DOC]
    # HELPER: resets the list of local workers and starts the requested number of them with the
    # optional worker tags.
    self._local_workers = []
    for _ in range(num_workers):
      self.start_local_worker(worker_tags)

  def _start_remote_workers(self, num_workers, worker_tags=None):
    # [FUNCTION DOC]
    # HELPER: resets the list of remote workers and starts the requested number of them in child
    # processes, with the optional worker tags.
    # List of (worker address, remote worker process) tuples.
    self._remote_workers = []
    for _ in range(num_workers):
      self.start_remote_worker(worker_tags)

  def start_local_worker(self, worker_tags=None):
    # [FUNCTION DOC]
    # Starts one in-process test worker on an unused port, connected to the dispatcher, with
    # optional worker tags, and appends it to the local worker list. Returns nothing.
    worker = data_service_test_base.TestWorker(
        self.dispatcher_address(),
        _WORKER_SHUTDOWN_QUIET_PERIOD_MS,
        port=test_util.pick_unused_port(),
        worker_tags=worker_tags)
    worker.start()
    self._local_workers.append(worker)

  def start_remote_worker(self, worker_tags=None):
    # [FUNCTION DOC]
    # Spawns a child process hosting a worker (unused port, optional tags), waits to receive its
    # address through a one-way pipe, and records the address and process pair. Blocks until the
    # worker reports its address. Returns nothing.
    """Runs a tf.data service worker in a remote process."""

    pipe_reader, pipe_writer = multi_process_lib.multiprocessing.Pipe(
        duplex=False)
    worker_process = _RemoteWorkerProcess(
        self.dispatcher_address(),
        port=test_util.pick_unused_port(),
        worker_tags=worker_tags,
        pipe_writer=pipe_writer)
    worker_process.start()
    worker_address = pipe_reader.recv()
    self._remote_workers.append((worker_address, worker_process))

  def restart_dispatcher(self):
    # [FUNCTION DOC]
    # Stops the dispatcher and starts a new one on the same port, telling it the addresses of all
    # local and remote workers. Returns nothing.
    port = int(self.dispatcher_address().split(":")[1])
    self._dispatcher._stop()
    self._start_dispatcher(
        worker_addresses=(self.local_worker_addresses() +
                          self.remote_worker_addresses()),
        port=port)

  def restart_local_workers(self):
    # [FUNCTION DOC]
    # Restarts every local worker in place. Remote workers are untouched. Returns nothing.
    for worker in self._local_workers:
      worker.restart()

  def dispatcher_address(self):
    # [FUNCTION DOC]
    # Returns the dispatcher's address as host and port text.
    return self._dispatcher._address

  def local_worker_addresses(self):
    # [FUNCTION DOC]
    # Returns a list with the address of each local worker.
    return [worker.worker_address() for worker in self._local_workers]

  def remote_worker_addresses(self):
    # [FUNCTION DOC]
    # Returns a list with the address of each remote worker.
    return [worker_address for (worker_address, _) in self._remote_workers]

  def _stop(self):
    # [FUNCTION DOC]
    # HELPER: shuts the cluster down: stops local workers, kills the remote worker processes and
    # stops the dispatcher. Returns nothing.
    for worker in self._local_workers:
      worker.stop()
    for (_, worker_process) in self._remote_workers:
      worker_process.kill()
    self._dispatcher._stop()

  def __del__(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: if the constructor failed before the worker lists or dispatcher were created,
    # stopping here would fail with a missing attribute error; no guard exists.
    # Destructor: stops the whole cluster when the object is garbage collected.
    self._stop()


def test_main():
  # [FUNCTION DOC]
  # HELPER: entry point for test files in this package; delegates to the multi-process test
  # library main so child processes work. Takes nothing, returns what the library main does.
  """Main function to be called within `__main__` of a test file."""
  multi_process_lib.test_main()
