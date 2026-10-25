# [MODULE DOC]
# Test module checking that variables placed on a local gRPC server persist across separate
# sessions connected to the same server when nothing is reset. One test case, split out of the
# main server library tests because multiple servers in one file were flaky.
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
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib


class SameVariablesNoClearTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering state sharing between successive sessions of one server without any
  # reset. No shared fixtures.

  # Verifies behavior of multiple variables with multiple sessions connecting to
  # the same server.
  # TODO(b/34465411): Starting multiple servers with different configurations
  # in the same test is flaky. Move this test case back into
  # "server_lib_test.py" when this is no longer the case.
  @test_util.run_v1_only(
      "This exercises tensor lookup via names which is not supported in V2.")
  def testSameVariablesNoClear(self):
    # [FUNCTION DOC]
    # Scenario: starts a local server. In a first session, creates two matrix variables, their
    # product, initializes them and checks the product is the single value 4. After that session
    # closes, a second session to the same server looks up the variables by tensor name in the
    # default graph, builds a new product and must also get 4, showing variable state survived
    # across sessions. Runs only in v1 graph mode because it looks up tensors by name.
    server = server_lib.Server.create_local_server()

    with session.Session(server.target) as sess_1:
      v0 = variable_v1.VariableV1([[2, 1]], name="v0")
      v1 = variable_v1.VariableV1([[1], [2]], name="v1")
      v2 = math_ops.matmul(v0, v1)
      sess_1.run([v0.initializer, v1.initializer])
      self.assertAllEqual([[4]], sess_1.run(v2))

    with session.Session(server.target) as sess_2:
      new_v0 = ops.get_default_graph().get_tensor_by_name("v0:0")
      new_v1 = ops.get_default_graph().get_tensor_by_name("v1:0")
      new_v2 = math_ops.matmul(new_v0, new_v1)
      self.assertAllEqual([[4]], sess_2.run(new_v2))


if __name__ == "__main__":
  test.main()
