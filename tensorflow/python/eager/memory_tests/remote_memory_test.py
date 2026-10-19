# [MODULE DOC]
# Memory-leak test for remote eager execution: checks that running a local compiled function
# on an input tensor living on a remote worker does not leak a reference to the remote tensor.
# Starts an in-process local server and connects to it as a remote host. Uses the shared leak
# assertion helper and skips when the memory profiler is unavailable. Running as a script
# starts the test main.
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
"""Tests for memory leaks in remote eager execution."""

from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.eager.memory_tests import memory_test_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.training import server_lib


class RemoteWorkerMemoryTest(test.TestCase):
  # [CLASS DOC]
  # Test case for memory leaks with remote workers. Each instance starts its own local
  # in-process server in its constructor and remembers the server's address (without the
  # protocol prefix) for connecting.

  def __init__(self, method):
    # [FUNCTION DOC]
    # Constructor. Input: method, the name of the test method this instance runs (passed to the
    # base class). Creates a local server in-process and stores the server and its target
    # address with the grpc scheme prefix removed. Side effect: a server is started for every
    # test instance created, even for tests that are later skipped.
    super(RemoteWorkerMemoryTest, self).__init__(method)

    # used for remote worker tests
    self._cached_server = server_lib.Server.create_local_server()
    self._cached_server_target = self._cached_server.target[len("grpc://"):]

  def testMemoryLeakInLocalCopy(self):
    # [FUNCTION DOC]
    # Test: connects to the local server as a remote host, then repeats 100 times a workload that
    # creates a 1000 by 1000 integer zeros tensor on the remote worker device and passes it to a
    # local compiled identity function. Memory growth must stay under 50 megabytes. Skipped if the
    # memory profiler is unavailable. Side effect: configures the remote connection globally.
    if not memory_test_util.memory_profiler_is_available():
      self.skipTest("memory_profiler required to run this test")

    remote.connect_to_remote_host(self._cached_server_target)

    # Run a function locally with the input on a remote worker and ensure we
    # do not leak a reference to the remote tensor.

    @def_function.function
    def local_func(i):
      # [FUNCTION DOC]
      # Compiled function that returns its input unchanged.
      return i

    def func():
      # [FUNCTION DOC]
      # Workload: allocates the zeros tensor on the remote worker device, then calls the local
      # compiled function with it (forcing a copy to the local device); returns nothing.
      with ops.device("job:worker/replica:0/task:0/device:CPU:0"):
        x = array_ops.zeros([1000, 1000], dtypes.int32)

      local_func(x)

    memory_test_util.assert_no_leak(
        func, num_iters=100, increase_threshold_absolute_mb=50)


if __name__ == "__main__":
  test.main()
