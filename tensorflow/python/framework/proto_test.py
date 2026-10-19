# [MODULE DOC]
# Test module for protobuf-related behavior of the framework. It currently contains a single
# disabled test (its name starts with an underscore so the runner never collects it) that checks
# that a graph definition holding a constant larger than 64 megabytes can be serialized and
# parsed back. Running the file directly starts the standard TensorFlow test runner. No
# import-time side effects.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Protobuf related tests."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


class ProtoTest(test.TestCase):
  # [CLASS DOC]
  # Test case holding protobuf serialization checks for graph definitions. Currently its only
  # test is disabled, so no test in this class is actually collected. Uses the standard
  # TensorFlow test case base class.

  # TODO(vrv): re-enable this test once we figure out how this can
  # pass the pip install test (where the user is expected to have
  # protobuf installed).
  def _testLargeProto(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the leading underscore means the test runner never runs this method; it is
    # disabled pending a fix for the pip install test environment (see the TODO above it).
    # Scenario: build a constant of zeros with a size above 64 megabytes in the default graph,
    # export the graph definition, serialize it to a byte string, parse the bytes into a fresh
    # graph definition, and assert that the parsed message equals the original. Expected: the
    # protobuf library handles messages larger than 64 megabytes without error.
    # create a constant of size > 64MB.
    a = constant_op.constant(np.zeros([1024, 1024, 17]))
    # Serialize the resulting graph def.
    gdef = a.op.graph.as_graph_def()
    serialized = gdef.SerializeToString()
    unserialized = ops.Graph().as_graph_def()
    # Deserialize back. Protobuf python library should support
    # protos larger than 64MB.
    unserialized.ParseFromString(serialized)
    self.assertProtoEquals(unserialized, gdef)


if __name__ == "__main__":
  test.main()
