# [MODULE DOC]
# Unit tests for the DTensor configuration helpers (client id, job names, device listing, and
# parsing of the DTENSOR_JOBS environment variable). Defines one test class and a main entry point.
# Importing it has no side effects beyond importing DTensor config and test utilities.
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
"""Tests for the open source DTensor Python API."""

import os

# pylint: disable=g-direct-tensorflow-import
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.eager import context
from tensorflow.python.framework import device as tf_device
from tensorflow.python.platform import test as tf_test
# pylint: enable=g-direct-tensorflow-import


class ConfigTest(tf_test.TestCase):
  # [CLASS DOC]
  # Test case covering the DTensor config module: environment-derived values such as client id and
  # job names, local/global device listing, and validation of the jobs environment variable.
  # Each test starts with two logical CPU devices (and two GPUs if a GPU is present).

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup. Calls the parent setup, then resets the logical devices to two CPUs, and
    # also to two GPUs when a physical GPU is present. Takes no arguments and returns nothing.
    super().setUp()
    test_util.reset_logical_devices('CPU', 2)
    if test_util.is_gpu_present():
      test_util.reset_logical_devices('GPU', 2)

  def tearDown(self):
    # [FUNCTION DOC]
    # Test fixture cleanup. Removes the DTensor jobs environment variable (using the config
    # module's private name for it) so tests do not leak state, then calls the parent teardown.
    # Returns nothing.
    os.environ.pop(config._DT_JOBS, [])
    super().tearDown()

  def test_env_vars(self):
    # [FUNCTION DOC]
    # With no DTensor environment set, checks the defaults: client id 0, one client, job name
    # "localhost", full job name "localhost/replica:0/task:0", and an empty jobs list.
    self.assertEqual(config.client_id(), 0)
    self.assertEqual(config.num_clients(), 1)
    self.assertEqual(config.job_name(), 'localhost')
    self.assertEqual(config.full_job_name(), 'localhost/replica:0/task:0')
    self.assertEqual(config.jobs(), [])

  def test_list_devices(self):
    # [FUNCTION DOC]
    # Checks that local devices of the preferred device type are the two expected device
    # specifications (task 0 of the localhost job, indexes 0 and 1), and that the local and global
    # device counts are both 2. Also asserts that these calls did not initialize the eager context.
    device_type = config.preferred_device_type()

    local_devices = [
        tf_device.DeviceSpec.from_string(
            f'/job:localhost/replica:0/task:0/device:{device_type}:0'),
        tf_device.DeviceSpec.from_string(
            f'/job:localhost/replica:0/task:0/device:{device_type}:1'),
    ]

    self.assertEqual(config.local_devices(device_type), local_devices)
    self.assertEqual(config.num_local_devices(device_type), 2)
    self.assertEqual(config.num_global_devices(device_type), 2)
    # The eager context should not be initialized by any of the calls
    self.assertFalse(context.context()._initialized)  # pylint: disable=protected-access

  def test_sort_jobs_with_bns_names(self):
    # [FUNCTION DOC]
    # Sets the jobs environment variable to comma-joined lists of 16 bns-style job names. Expects
    # the config jobs list to equal the input when names are in increasing task order, with or
    # without a port suffix. For a decreasing task order it expects a value error mentioning an
    # unexpected jobs setting. Mutates the environment variable (cleaned up in teardown).
    # bns names must be sorted in the bns order.
    dtensor_jobs = [
        '/bns/localhost/{task_id}'.format(task_id=i) for i in range(16)
    ]
    os.environ[config._DT_JOBS] = ','.join(dtensor_jobs)
    self.assertListEqual(dtensor_jobs, config.jobs())

    dtensor_jobs = [
        '/bns/localhost/{task_id}:8888'.format(task_id=i) for i in range(16)
    ]
    os.environ[config._DT_JOBS] = ','.join(dtensor_jobs)
    self.assertListEqual(dtensor_jobs, config.jobs())

    dtensor_jobs = [
        '/bns/localhost/{task_id}'.format(task_id=100 - i) for i in range(16)
    ]
    os.environ[config._DT_JOBS] = ','.join(dtensor_jobs)
    with self.assertRaisesRegex(ValueError, 'Unexpected DTENSOR_JOBS'):
      config.jobs()

  def test_jobs_with_ip_port(self):
    # [FUNCTION DOC]
    # Sets the jobs environment variable to 16 host:port entries with descending port numbers and
    # expects the config jobs list to return them unchanged, since non-bns addresses are not
    # required to be sorted.
    # The ip port format is not a bns address, and not required to sorted.
    dtensor_jobs = ['localhost:{port}'.format(port=16 - i) for i in range(16)]
    os.environ[config._DT_JOBS] = ','.join(dtensor_jobs)
    self.assertListEqual(dtensor_jobs, config.jobs())


if __name__ == '__main__':
  tf_test.main()
