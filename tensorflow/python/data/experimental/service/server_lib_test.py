# [MODULE DOC]
# Unit tests for the tf.data service server library (server_lib): starting, stopping and joining
# dispatcher and worker servers, port and work directory configuration, fault tolerance validation,
# worker counting and profiler connectivity. Also defines a small helper that hands out unused
# local ports without reusing them within the process. Runs through the TensorFlow test main.
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
"""Tests for tf.data service server lib."""

import logging
import tempfile
import threading
import unittest
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.platform import test
from tensorflow.python.profiler import profiler_client

_portpicker_import_error = None
try:
  import portpicker  # pylint: disable=g-import-not-at-top
except ImportError as _error:  # pylint: disable=invalid-name
  _portpicker_import_error = _error
  portpicker = None

ASSIGNED_PORTS = set()
lock = threading.Lock()


def pick_unused_port():
  # [FUNCTION DOC]
  # HELPER: shared by the tests that need a free local port.
  # Returns an integer port above 10000 that the port picker library reports as free and that this
  # module has not handed out before (tracked in a module-level set under a lock, logged at info
  # level). Raises the saved import error if the port picker library is unavailable, and skips the
  # calling test if the library cannot find any free port.
  """Returns an unused and unassigned local port."""

  if _portpicker_import_error:
    raise _portpicker_import_error  # pylint: disable=raising-bad-type

  global ASSIGNED_PORTS
  with lock:
    while True:
      try:
        port = portpicker.pick_unused_port()
      except portpicker.NoFreePortFoundError:
        raise unittest.SkipTest("Flakes in portpicker library do not represent "
                                "TensorFlow errors.")
      if port > 10000 and port not in ASSIGNED_PORTS:
        ASSIGNED_PORTS.add(port)
        logging.info("Using local port %r", port)
        return port


class ServerLibTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering DispatchServer, WorkerServer and their configuration classes in server_lib.
  # No shared fixtures; each test creates its own in-process servers.

  def testStartDispatcher(self):
    # [FUNCTION DOC]
    # Creates a dispatcher without starting it, then starts it. Expects no error.
    dispatcher = server_lib.DispatchServer(start=False)
    dispatcher.start()

  def testStartDispatcherWithPortConfig(self):
    # [FUNCTION DOC]
    # Starts a dispatcher configured with a specific unused port. Asserts that the target string is
    # the grpc protocol, the local host and that port.
    port = pick_unused_port()
    config = server_lib.DispatcherConfig(port=port)
    dispatcher = server_lib.DispatchServer(config=config, start=True)
    self.assertEqual(dispatcher.target, "grpc://localhost:{}".format(port))

  def testStartDispatcherWithWorkDirConfig(self):
    # [FUNCTION DOC]
    # Starts a dispatcher configured with a temporary work directory. Expects no error.
    temp_dir = tempfile.mkdtemp()
    config = server_lib.DispatcherConfig(work_dir=temp_dir)
    dispatcher = server_lib.DispatchServer(  # pylint: disable=unused-variable
        config=config, start=True)

  def testStartDispatcherWithFaultTolerantConfig(self):
    # [FUNCTION DOC]
    # Starts a dispatcher with fault tolerant mode enabled and a temporary work directory.
    # Expects no error.
    temp_dir = tempfile.mkdtemp()
    config = server_lib.DispatcherConfig(
        work_dir=temp_dir, fault_tolerant_mode=True)
    dispatcher = server_lib.DispatchServer(  # pylint: disable=unused-variable
        config=config, start=True)

  def testStartDispatcherWithWrongFaultTolerantConfig(self):
    # [FUNCTION DOC]
    # Builds a configuration enabling fault tolerant mode with no work directory. Asserts that
    # creating the dispatcher raises a value error saying a work dir must be configured.
    config = server_lib.DispatcherConfig(fault_tolerant_mode=True)
    error = "Cannot enable fault tolerant mode without configuring a work dir"
    with self.assertRaisesRegex(ValueError, error):
      dispatcher = server_lib.DispatchServer(  # pylint: disable=unused-variable
          config=config, start=True)

  def testMultipleStartDispatcher(self):
    # [FUNCTION DOC]
    # Creates an already started dispatcher and starts it again. Expects the second start to be
    # harmless (no error).
    dispatcher = server_lib.DispatchServer(start=True)
    dispatcher.start()

  def testStartWorker(self):
    # [FUNCTION DOC]
    # Creates a dispatcher and a worker pointing at it without starting the worker, then starts the
    # worker. Expects no error.
    dispatcher = server_lib.DispatchServer()
    worker = server_lib.WorkerServer(
        server_lib.WorkerConfig(dispatcher._address), start=False)
    worker.start()

  def testStartWorkerWithPortConfig(self):
    # [FUNCTION DOC]
    # Starts a worker configured with a specific unused port. Asserts that the worker address is
    # the local host with that port.
    dispatcher = server_lib.DispatchServer()
    port = pick_unused_port()
    worker = server_lib.WorkerServer(
        server_lib.WorkerConfig(dispatcher._address, port=port), start=True)
    self.assertEqual(worker._address, "localhost:{}".format(port))

  def testMultipleStartWorker(self):
    # [FUNCTION DOC]
    # Creates an already started worker and starts it again. Expects no error.
    dispatcher = server_lib.DispatchServer()
    worker = server_lib.WorkerServer(
        server_lib.WorkerConfig(dispatcher._address), start=True)
    worker.start()

  def testStopDispatcher(self):
    # [FUNCTION DOC]
    # Stops a dispatcher twice. Expects stopping to be idempotent (no error).
    dispatcher = server_lib.DispatchServer()
    dispatcher.stop()
    dispatcher.stop()

  def testStopWorker(self):
    # [FUNCTION DOC]
    # Stops a worker twice. Expects stopping to be idempotent (no error).
    dispatcher = server_lib.DispatchServer()
    worker = server_lib.WorkerServer(
        server_lib.WorkerConfig(dispatcher._address))
    worker.stop()
    worker.stop()

  def testStopStartDispatcher(self):
    # [FUNCTION DOC]
    # Stops a dispatcher and then tries to start it. Asserts a runtime error stating that a server
    # cannot be started after it has been stopped.
    dispatcher = server_lib.DispatchServer()
    dispatcher.stop()
    with self.assertRaisesRegex(
        RuntimeError, "Server cannot be started after it has been stopped"):
      dispatcher.start()

  def testStopStartWorker(self):
    # [FUNCTION DOC]
    # Stops a worker and then tries to start it. Asserts a runtime error stating that a server
    # cannot be started after it has been stopped.
    dispatcher = server_lib.DispatchServer()
    worker = server_lib.WorkerServer(
        server_lib.WorkerConfig(dispatcher._address))
    worker.stop()
    with self.assertRaisesRegex(
        RuntimeError, "Server cannot be started after it has been stopped"):
      worker.start()

  def testJoinDispatcher(self):
    # [FUNCTION DOC]
    # Stops a dispatcher and then joins it. Expects join to return without blocking or error.
    dispatcher = server_lib.DispatchServer()
    dispatcher.stop()
    dispatcher.join()

  def testJoinWorker(self):
    # [FUNCTION DOC]
    # Stops a worker and then joins it. Expects join to return without blocking or error.
    dispatcher = server_lib.DispatchServer()
    worker = server_lib.WorkerServer(
        server_lib.WorkerConfig(dispatcher._address))
    worker.stop()
    worker.join()

  def testDispatcherNumWorkers(self):
    # [FUNCTION DOC]
    # Starts a dispatcher and checks its registered worker count is 0, then adds one worker and
    # checks the count is 1, then a second worker and checks the count is 2.
    dispatcher = server_lib.DispatchServer()
    self.assertEqual(0, dispatcher._num_workers())
    worker1 = server_lib.WorkerServer(  # pylint: disable=unused-variable
        server_lib.WorkerConfig(dispatcher._address))
    self.assertEqual(1, dispatcher._num_workers())
    worker2 = server_lib.WorkerServer(  # pylint: disable=unused-variable
        server_lib.WorkerConfig(dispatcher._address))
    self.assertEqual(2, dispatcher._num_workers())

  def testProfileWorker(self):
    # [FUNCTION DOC]
    # Starts a dispatcher and a worker, then asks the profiler client to trace the worker for a
    # short duration with nothing running. Asserts a runtime error whose message starts with an
    # unavailable status saying no trace event was collected, proving the profiler service is
    # reachable on the worker.
    dispatcher = server_lib.DispatchServer()
    worker = server_lib.WorkerServer(
        server_lib.WorkerConfig(dispatcher._address)
    )
    # Test the profilers are successfully started and connected to profiler
    # service on the worker. Since there is no op running, it is expected to
    # return RuntimeError with no trace events collected string.
    with self.assertRaises(RuntimeError) as error:
      profiler_client.trace(worker._address, tempfile.mkdtemp(), duration_ms=10)
    self.assertStartsWith(
        str(error.exception), "UNAVAILABLE: No trace event was collected"
    )


if __name__ == "__main__":
  test.main()
