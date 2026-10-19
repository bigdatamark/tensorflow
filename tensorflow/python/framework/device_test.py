# [MODULE DOC]
# Unit tests for the framework device module helpers: merging device strings under nested device
# scopes, canonicalizing device names and validating device strings. Parameterized over the legacy
# and immutable device spec types. No import-time side effects; runs the test runner as a script.
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
"""Tests for tensorflow.python.framework.device."""
from absl.testing import parameterized

from tensorflow.python.eager import context
from tensorflow.python.framework import device
from tensorflow.python.framework import device_spec
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest

TEST_V1_AND_V2 = (("v1", device_spec.DeviceSpecV1), ("v2",
                                                     device_spec.DeviceSpecV2))


class DeviceTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the device helper functions (merge, canonical name, validity check). Uses the
  # module level pairing of names and spec types for parameterized tests; no shared fixtures.

  @parameterized.named_parameters(*TEST_V1_AND_V2)
  def testMerge(self, DeviceSpec):  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # Parameterized test: parses a device string with the given spec type and checks it round-trips.
    # Then, only when not executing eagerly, nests device scopes using merged-device functions and
    # checks that variables created inside get device strings combining the scopes, with inner
    # scopes overriding the outer job or device type. Input is the spec type.
    d = DeviceSpec.from_string("/job:muu/task:1/device:MyFunnyDevice:2")
    self.assertEqual("/job:muu/task:1/device:MyFunnyDevice:2", d.to_string())

    if not context.executing_eagerly():
      with ops.device(device.merge_device("/device:GPU:0")):
        var1 = variables.Variable(1.0)
        self.assertEqual("/device:GPU:0", var1.device)
        with ops.device(device.merge_device("/job:worker")):
          var2 = variables.Variable(1.0)
          self.assertEqual("/job:worker/device:GPU:0", var2.device)
          with ops.device(device.merge_device("/device:CPU:0")):
            var3 = variables.Variable(1.0)
            self.assertEqual("/job:worker/device:CPU:0", var3.device)
            with ops.device(device.merge_device("/job:ps")):
              var4 = variables.Variable(1.0)
              self.assertEqual("/job:ps/device:CPU:0", var4.device)

  def testCanonicalName(self):
    # [FUNCTION DOC]
    # Tests that the canonical name function reorders job, replica, task and device components
    # into the standard order and leaves already canonical names unchanged.
    self.assertEqual("/job:foo/replica:0",
                     device.canonical_name("/job:foo/replica:0"))
    self.assertEqual("/job:foo/replica:0",
                     device.canonical_name("/replica:0/job:foo"))

    self.assertEqual("/job:foo/replica:0/task:0",
                     device.canonical_name("/job:foo/replica:0/task:0"))
    self.assertEqual("/job:foo/replica:0/task:0",
                     device.canonical_name("/job:foo/task:0/replica:0"))

    self.assertEqual("/device:CPU:0", device.canonical_name("/device:CPU:0"))
    self.assertEqual("/device:GPU:2", device.canonical_name("/device:GPU:2"))

    self.assertEqual(
        "/job:foo/replica:0/task:0/device:GPU:0",
        device.canonical_name("/job:foo/replica:0/task:0/device:GPU:0"))
    self.assertEqual(
        "/job:foo/replica:0/task:0/device:GPU:0",
        device.canonical_name("/device:GPU:0/task:0/replica:0/job:foo"))

  def testCheckValid(self):
    # [FUNCTION DOC]
    # Tests the validity check: a well formed string passes; non-integer replica or task values
    # raise a value error about an invalid integer literal; an unknown attribute raises a value
    # error naming it; two device types raise a value error about multiple device types.
    device.check_valid("/job:foo/replica:0")

    with self.assertRaisesRegex(ValueError, "invalid literal for int"):
      device.check_valid("/job:j/replica:foo")

    with self.assertRaisesRegex(ValueError, "invalid literal for int"):
      device.check_valid("/job:j/task:bar")

    # Assume no one will register a device type named "barcpugpu"
    with self.assertRaisesRegex(ValueError, "Unknown attribute 'barcpugpu'"):
      device.check_valid("/barcpugpu:muu/baz:2")

    with self.assertRaisesRegex(ValueError,
                                "Multiple device types are not allowed"):
      device.check_valid("/cpu:0/device:GPU:2")


if __name__ == "__main__":
  googletest.main()
