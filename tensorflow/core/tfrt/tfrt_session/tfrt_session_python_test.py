# [MODULE DOC]
# Tests that a classic graph-mode Session works when the experimental TFRT runtime option is
# enabled in its configuration. Contains one test class with two matrix multiplication tests.
# No import-time side effects; running as main starts the googletest runner.

# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.python.client.session.Session (with tfrt_session)."""

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest


class TfrtSessionPythonTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case covering sessions configured to use the TFRT runtime. The shared fixture is a
  # configuration object with the experimental use-TFRT option on, created in setUp and kept in
  # a private attribute.

  def setUp(self):
    # [FUNCTION DOC]
    # Runs the parent setUp, then stores a session configuration with the experimental TFRT
    # option enabled for use by the tests. Returns nothing.
    super(TfrtSessionPythonTest, self).setUp()
    self._config = config_pb2.ConfigProto(
        experimental=config_pb2.ConfigProto.Experimental(use_tfrt=True))

  def testUseExistingGraph(self):
    # [FUNCTION DOC]
    # Scenario: build a fresh graph on the CPU with two 1x1 constants (6 and 7) and a matrix
    # multiplication. Open a session passing that graph explicitly plus the TFRT config, evaluate
    # the product, and assert it equals a 1x1 matrix holding 42.
    with ops.Graph().as_default() as g, ops.device('/cpu:0'):
      a = constant_op.constant(6.0, shape=[1, 1])
      b = constant_op.constant(7.0, shape=[1, 1])
      c = math_ops.matmul(a, b, name='matmul')
    with session.Session(graph=g, config=self._config):
      result = c.eval()
      self.assertAllEqual(result, [[42.0]])

  def testUseDefaultGraph(self):
    # [FUNCTION DOC]
    # Scenario: same 1x1 matrix multiplication (6 times 7) built in a graph made default, but the
    # session is opened without passing a graph, so it must pick up the default graph, using the
    # TFRT config. Asserts the evaluated result equals a 1x1 matrix holding 42.
    with ops.Graph().as_default(), ops.device('/cpu:0'):
      a = constant_op.constant(6.0, shape=[1, 1])
      b = constant_op.constant(7.0, shape=[1, 1])
      c = math_ops.matmul(a, b, name='matmul')
      with session.Session(config=self._config):
        result = c.eval()
        self.assertAllEqual(result, [[42.0]])


if __name__ == '__main__':
  googletest.main()
