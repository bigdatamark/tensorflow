# [MODULE DOC]
# Test module for session reset with multiple resource containers on a local gRPC server.
# Contains one test case, kept in its own file because starting several servers with different
# configurations in one test file was flaky. Runs under the standard TensorFlow test main.
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
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib


class MultipleContainersTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering resetting selected containers on a server through the session reset
  # call. No shared fixtures.

  # Verifies behavior of tf.Session.reset() with multiple containers using
  # tf.container.
  # TODO(b/34465411): Starting multiple servers with different configurations
  # in the same test is flaky. Move this test case back into
  # "server_lib_test.py" when this is no longer the case.
  @test_util.run_deprecated_v1
  def testMultipleContainers(self):
    # [FUNCTION DOC]
    # Scenario: two variables are created in two different named containers (test0 and test1)
    # and a local server is started. After initializing and checking values (1.0 and 2.0), only
    # container test0 is reset on the server. Expected: the old session running the second
    # variable fails with an aborted error; a new session to the same target finds the first
    # variable uninitialized (failed-precondition error) while the second variable is still
    # valid with value 2.0. Graph-mode (deprecated v1) test.
    with ops.container("test0"):
      v0 = variables.Variable(1.0, name="v0")
    with ops.container("test1"):
      v1 = variables.Variable(2.0, name="v0")
    server = server_lib.Server.create_local_server()
    sess = session.Session(server.target)
    sess.run(variables.global_variables_initializer())
    self.assertAllEqual(1.0, sess.run(v0))
    self.assertAllEqual(2.0, sess.run(v1))

    # Resets container. Session aborts.
    session.Session.reset(server.target, ["test0"])
    with self.assertRaises(errors_impl.AbortedError):
      sess.run(v1)

    # Connects to the same target. Device memory for the v0 would have
    # been released, so it will be uninitialized. But v1 should still
    # be valid.
    sess = session.Session(server.target)
    with self.assertRaises(errors_impl.FailedPreconditionError):
      sess.run(v0)
    self.assertAllEqual(2.0, sess.run(v1))


if __name__ == "__main__":
  test.main()
