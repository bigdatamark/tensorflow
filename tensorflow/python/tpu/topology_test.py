# [MODULE DOC]
# Unit test module for the TPU topology class: checks that a topology survives a round trip
# through its serialized form. Runs under the standard TensorFlow test runner; no import-time
# side effects.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

"""Tests for topology.py."""

from tensorflow.python.platform import test
from tensorflow.python.tpu import topology


class TopologyTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the topology class. No shared fixtures; each test builds its own
  # topology objects.

  def testSerialization(self):
    # [FUNCTION DOC]
    # Scenario: build a topology from a mesh shape of 1,1,1,2 and two devices of one task at
    # core coordinates 0 and 1. Action: serialize it and construct a second topology from the
    # serialized bytes. Asserts the recovered mesh shape and device coordinates equal the
    # original's.
    """Tests if the class is able to generate serialized strings."""
    original_topology = topology.Topology(
        mesh_shape=[1, 1, 1, 2],
        device_coordinates=[[[0, 0, 0, 0], [0, 0, 0, 1]]],
    )
    serialized_str = original_topology.serialized()
    new_topology = topology.Topology(serialized=serialized_str)

    # Make sure the topology recovered from serialized str is same as the
    # original topology.
    self.assertAllEqual(
        original_topology.mesh_shape, new_topology.mesh_shape)
    self.assertAllEqual(
        original_topology.device_coordinates, new_topology.device_coordinates)

if __name__ == "__main__":
  test.main()
