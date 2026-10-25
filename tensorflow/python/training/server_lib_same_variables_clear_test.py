# [MODULE DOC]
# Test module for resetting a whole server target and its effect on sessions and variables.
# One test case, split out of the main server library tests because starting multiple servers
# with different configurations in one file was flaky.
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
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib


class SameVariablesClearTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the reset of a server target with no container names (clears all
  # containers) while multiple sessions share variables on that server. No shared fixtures.

  # Verifies behavior of tf.Session.reset().
  # TODO(b/34465411): Starting multiple servers with different configurations
  # in the same test is flaky. Move this test case back into
  # "server_lib_test.py" when this is no longer the case.
  @test_util.run_deprecated_v1
  def testSameVariablesClear(self):
    # [FUNCTION DOC]
    # Scenario: a local server is started and a graph with two matrix variables and their
    # product is built. Two sessions to the same target both compute the product after
    # initialization and must agree (the single value 4). Then the whole target is reset: the
    # old session must fail with an aborted error. A new session finds the variables
    # uninitialized (failed-precondition error); after re-initializing them the product is again
    # 4. Finally the session is closed. Graph-mode (deprecated v1) test.
    server = server_lib.Server.create_local_server()

    # Creates a graph with 2 variables.
    v0 = variables.Variable([[2, 1]], name="v0")
    v1 = variables.Variable([[1], [2]], name="v1")
    v2 = math_ops.matmul(v0, v1)

    # Verifies that both sessions connecting to the same target return
    # the same results.
    sess_1 = session.Session(server.target)
    sess_2 = session.Session(server.target)
    sess_1.run(variables.global_variables_initializer())
    self.assertAllEqual([[4]], sess_1.run(v2))
    self.assertAllEqual([[4]], sess_2.run(v2))

    # Resets target. sessions abort. Use sess_2 to verify.
    session.Session.reset(server.target)
    with self.assertRaises(errors_impl.AbortedError):
      self.assertAllEqual([[4]], sess_2.run(v2))

    # Connects to the same target. Device memory for the variables would have
    # been released, so they will be uninitialized.
    sess_2 = session.Session(server.target)
    with self.assertRaises(errors_impl.FailedPreconditionError):
      sess_2.run(v2)
    # Reinitializes the variables.
    sess_2.run(variables.global_variables_initializer())
    self.assertAllEqual([[4]], sess_2.run(v2))
    sess_2.close()


if __name__ == "__main__":
  test.main()
