# [MODULE DOC]
# Test module for the session device-listing API: direct sessions, the low-level C session
# wrapper, a local gRPC server session, and a session using ClusterSpec propagation. Exposes
# nothing; the main block runs the tests. No import-time side effects.
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

"""Tests for tensorflow.python.client.session.Session's list_devices API."""

from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import pywrap_tf_session as tf_session
from tensorflow.python.client import session
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest
from tensorflow.python.training import server_lib


class SessionListDevicesTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for listing devices of a session. No shared fixtures.

  def testListDevices(self):
    # [FUNCTION DOC]
    # Lists devices of a default local session and asserts the local CPU device is present and
    # every device has a non-zero incarnation.
    with session.Session() as sess:
      devices = sess.list_devices()
      self.assertTrue('/job:localhost/replica:0/task:0/device:CPU:0' in set(
          [d.name for d in devices]), devices)
      # All valid device incarnations must be non-zero.
      self.assertTrue(all(d.incarnation != 0 for d in devices))

  def testInvalidDeviceNumber(self):
    # [FUNCTION DOC]
    # Uses the low-level session API: creates a session on the default graph, gets its raw
    # device list, and asks for the memory size of the device at an index equal to the count
    # (out of range). Asserts an invalid-argument error, then frees the list and closes the
    # session.
    opts = tf_session.TF_NewSessionOptions()
    with ops.get_default_graph()._c_graph.get() as c_graph:
      c_session = tf_session.TF_NewSession(c_graph, opts)
    raw_device_list = tf_session.TF_SessionListDevices(c_session)
    size = tf_session.TF_DeviceListCount(raw_device_list)
    with self.assertRaises(errors.InvalidArgumentError):
      tf_session.TF_DeviceListMemoryBytes(raw_device_list, size)
    tf_session.TF_DeleteDeviceList(raw_device_list)
    tf_session.TF_CloseSession(c_session)

  def testListDevicesGrpcSession(self):
    # [FUNCTION DOC]
    # Starts a local server, opens a session on its target, lists devices, and asserts the CPU
    # device is present and incarnations are non-zero.
    server = server_lib.Server.create_local_server()
    with session.Session(server.target) as sess:
      devices = sess.list_devices()
      self.assertTrue(
          '/job:localhost/replica:0/task:0/device:CPU:0' in set(
              [d.name for d in devices]), devices)
      # All valid device incarnations must be non-zero.
      self.assertTrue(all(d.incarnation != 0 for d in devices))

  def testListDevicesClusterSpecPropagation(self):
    # [FUNCTION DOC]
    # Starts two local servers, builds a two-task worker cluster passed in the session config,
    # lists devices from a session on the first server, and asserts CPU devices of both worker
    # tasks appear and all incarnations are non-zero.
    server1 = server_lib.Server.create_local_server()
    server2 = server_lib.Server.create_local_server()

    cluster_def = cluster_pb2.ClusterDef()
    job = cluster_def.job.add()
    job.name = 'worker'
    job.tasks[0] = server1.target[len('grpc://'):]
    job.tasks[1] = server2.target[len('grpc://'):]
    config = config_pb2.ConfigProto(cluster_def=cluster_def)
    with session.Session(server1.target, config=config) as sess:
      devices = sess.list_devices()
      device_names = set(d.name for d in devices)
      self.assertTrue(
          '/job:worker/replica:0/task:0/device:CPU:0' in device_names)
      self.assertTrue(
          '/job:worker/replica:0/task:1/device:CPU:0' in device_names)
      # All valid device incarnations must be non-zero.
      self.assertTrue(all(d.incarnation != 0 for d in devices))


if __name__ == '__main__':
  googletest.main()
