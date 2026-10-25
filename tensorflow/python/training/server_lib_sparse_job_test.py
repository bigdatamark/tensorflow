# [MODULE DOC]
# Test module for placing computation on a server whose job has a sparse set of task indices.
# One test case, split out of the main server library tests because multiple servers with
# different configurations in one file were flaky.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf.GrpcServer."""

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib


class SparseJobTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering servers defined with a sparse (non-contiguous) task index. No shared
  # fixtures.

  # TODO(b/34465411): Starting multiple servers with different configurations
  # in the same test is flaky. Move this test case back into
  # "server_lib_test.py" when this is no longer the case.
  @test_util.run_deprecated_v1
  def testSparseJob(self):
    # [FUNCTION DOC]
    # Scenario: starts a server for a job named local that defines only task index 37. A
    # constant is created on the device for that job and task, and a session to the server
    # target evaluates it. Expected: the value is 1.0, proving placement on a sparse task index
    # works. Graph-mode (deprecated v1) test.
    server = server_lib.Server({"local": {37: "localhost:0"}})
    with ops.device("/job:local/task:37"):
      a = constant_op.constant(1.0)

    with session.Session(server.target) as sess:
      self.assertEqual(1.0, self.evaluate(a))


if __name__ == "__main__":
  test.main()
