# [MODULE DOC]
# Tests for registering custom devices with the eager context, using a logging custom device
# provided by the custom device test utility module. Checks that ops under the custom device run
# there and that the context can identify custom device names. Running as a script enables eager
# execution and starts the test main. No import-time side effects.

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

from tensorflow.python.eager import context
from tensorflow.python.eager import custom_device_testutil
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


class CustomDeviceTest(test.TestCase):
  # [CLASS DOC]
  # Test case for custom device registration in the eager context. The fixture resets the
  # global eager context so each test registers devices on a clean state.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: calls the base setup and resets the global eager context. Returns nothing.
    super().setUp()
    context._reset_context()

  def testRegisterCustomDevice(self):
    # [FUNCTION DOC]
    # Gets a logging custom device and its arrived/executed flags for a custom device name,
    # registers it with the context, and asserts both flags start unset. Runs a constant and a
    # multiplication under that device scope and asserts the executed flag is set but the
    # arrived (copy onto device) flag is not. Finally asserts that reading the result as a numpy
    # value raises an internal error saying it is trying to copy.
    device_name = '/job:localhost/replica:0/task:0/device:CUSTOM:0'
    device, device_info, arrived_flag, executed_flag = (
        custom_device_testutil.GetLoggingDeviceCapsules(device_name))
    context.register_custom_device(device, device_name, device_info)
    self.assertFalse(custom_device_testutil.FlagValue(arrived_flag))
    self.assertFalse(custom_device_testutil.FlagValue(executed_flag))
    with ops.device(device_name):
      x = constant_op.constant(1.)
      y = x * constant_op.constant(2.)
    self.assertTrue(custom_device_testutil.FlagValue(executed_flag))
    # There was no copy onto the device. Actually I'm not sure how to trigger
    # that from Python.
    self.assertFalse(custom_device_testutil.FlagValue(arrived_flag))
    with self.assertRaisesRegex(errors.InternalError, 'Trying to copy'):
      y.numpy()

  def testIsCustomDevice(self):
    # [FUNCTION DOC]
    # Registers a logging custom device, then asserts the context reports its name as a custom
    # device and reports an ordinary CPU name as not a custom device.
    device_name = '/job:localhost/replica:0/task:0/device:CUSTOM:0'
    device, device_info, _, _ = (
        custom_device_testutil.GetLoggingDeviceCapsules(device_name))
    context.register_custom_device(device, device_name, device_info)

    self.assertTrue(context.is_custom_device(device_name))
    self.assertFalse(context.is_custom_device('cpu:0'))


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
