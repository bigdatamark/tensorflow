# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# [MODULE DOC]
# Unit tests for the debug_data module of the legacy TensorFlow Debugger. Covers conversion
# between device names and dump directory names, the bad-numerical-value predicate for several
# element types, parsing of dump file paths into tensor datum objects, and loading of dump root
# directories (device discovery, time origin, size, node listing, duplicate node name handling,
# filesystem pattern matching through the TensorFlow filesystem layer, and slash normalization
# of node names). Uses temporary directories and fake dump files; no import-time side effects.
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
"""Tests for tfdbg module debug_data."""
import os
import platform
import tempfile

import numpy as np

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import file_io
from tensorflow.python.platform import gfile
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test


class DeviceNamePathConversionTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests the two conversion functions between device names and device directory names.

  def testDeviceNameToDevicePath(self):
    # [FUNCTION DOC]
    # Converts a full job/replica/task/cpu device name to a path and expects the metadata
    # prefix, device tag and comma-separated pieces with colons replaced by underscores.
    self.assertEqual(
        debug_data.METADATA_FILE_PREFIX
        + debug_data.DEVICE_TAG
        + ",job_ps,replica_1,task_2,cpu_0",
        debug_data.device_name_to_device_path("/job:ps/replica:1/task:2/cpu:0"),
    )

  def testDevicePathToDeviceName(self):
    # [FUNCTION DOC]
    # Converts a device directory name back to a device name and expects the original
    # job/replica/task/cpu name with slashes and colons restored.
    self.assertEqual(
        "/job:ps/replica:1/task:2/cpu:0",
        debug_data.device_path_to_device_name(
            debug_data.METADATA_FILE_PREFIX
            + debug_data.DEVICE_TAG
            + ",job_ps,replica_1,task_2,cpu_0"
        ),
    )


class HasNanOrInfTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests the bad-numerical-value predicate against arrays with not-a-number or infinite
  # values, clean arrays, empty arrays, wrapper objects and several element types. Shared
  # fixture: a dummy tensor datum.

  def setUp(self):
    # [FUNCTION DOC]
    # Creates a dummy tensor datum object (for a nonexistent dump file) and stores it for the
    # tests; the predicate ignores it.
    self._dummy_datum = dummy_datum = debug_data.DebugTensorDatum(
        "/foo", "bar_0_DebugIdentity_42"
    )

  def testNaN(self):
    # [FUNCTION DOC]
    # An array containing not-a-number values must be reported as bad (true).
    a = np.array([np.nan, np.nan, 7.0])
    self.assertTrue(debug_data.has_inf_or_nan(self._dummy_datum, a))

  def testInf(self):
    # [FUNCTION DOC]
    # An array containing infinite values must be reported as bad (true).
    a = np.array([np.inf, np.inf, 7.0])
    self.assertTrue(debug_data.has_inf_or_nan(self._dummy_datum, a))

  def testNanAndInf(self):
    # [FUNCTION DOC]
    # An array containing both infinite and not-a-number values must be reported as bad (true).
    a = np.array([np.inf, np.nan, 7.0])
    self.assertTrue(debug_data.has_inf_or_nan(self._dummy_datum, a))

  def testNoNanOrInf(self):
    # [FUNCTION DOC]
    # An array of finite floats must be reported as not bad (false).
    a = np.array([0.0, 0.0, 7.0])
    self.assertFalse(debug_data.has_inf_or_nan(self._dummy_datum, a))

  def testEmpty(self):
    # [FUNCTION DOC]
    # An empty array must be reported as not bad (false).
    a = np.array([])
    self.assertFalse(debug_data.has_inf_or_nan(self._dummy_datum, a))

  def testInconvertibleTensorProto(self):
    # [FUNCTION DOC]
    # Wrapper objects for unconvertible tensors, both marked uninitialized and initialized, must
    # be reported as not bad (false).
    self.assertFalse(
        debug_data.has_inf_or_nan(
            self._dummy_datum,
            debug_data.InconvertibleTensorProto(
                tensor_pb2.TensorProto(), initialized=False
            ),
        )
    )
    self.assertFalse(
        debug_data.has_inf_or_nan(
            self._dummy_datum,
            debug_data.InconvertibleTensorProto(
                tensor_pb2.TensorProto(), initialized=True
            ),
        )
    )

  def testDTypeComplexWorks(self):
    # [FUNCTION DOC]
    # Complex arrays: finite values give false; adding a not-a-number element gives true.
    a = np.array([1j, 3j, 3j, 7j], dtype=np.complex128)
    self.assertFalse(debug_data.has_inf_or_nan(self._dummy_datum, a))

    b = np.array([1j, 3j, 3j, 7j, np.nan], dtype=np.complex128)
    self.assertTrue(debug_data.has_inf_or_nan(self._dummy_datum, b))

  def testDTypeIntegerWorks(self):
    # [FUNCTION DOC]
    # An integer array must be reported as not bad (false).
    a = np.array([1, 3, 3, 7], dtype=np.int16)
    self.assertFalse(debug_data.has_inf_or_nan(self._dummy_datum, a))

  def testDTypeStringGivesFalse(self):
    # [FUNCTION DOC]
    # A string array must be reported as not bad (false) since the checks do not apply.
    """isnan and isinf are not applicable to strings."""

    a = np.array(["s", "p", "a", "m"])
    self.assertFalse(debug_data.has_inf_or_nan(self._dummy_datum, a))

  def testDTypeObjectGivesFalse(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says object but the array uses a structured (record) element type
    # with a string field and a float vector field. Expects false.
    # A structured-element array must be reported as not bad (false).
    dt = np.dtype([("spam", np.str_, 16), ("eggs", np.float64, (2,))])
    a = np.array([("spam", (8.0, 7.0)), ("eggs", (6.0, 5.0))], dtype=dt)
    self.assertFalse(debug_data.has_inf_or_nan(self._dummy_datum, a))


class DebugTensorDatumTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests parsing of dump file paths by the tensor datum class.

  def testDebugDatum(self):
    # [FUNCTION DOC]
    # Builds a datum from a relative path with a device directory, two scope directories and a
    # file name for slot 2 of a node, then asserts debug op, node name (scopes joined), output
    # slot, tensor name, timestamp, watch key, full file path, and that both string and
    # representation forms equal the expected brace-wrapped text.
    dump_root = "/tmp/tfdbg_1"
    debug_dump_rel_path = (
        debug_data.METADATA_FILE_PREFIX
        + debug_data.DEVICE_TAG
        + ",job_localhost,replica_0,task_0,cpu_0"
        + "/ns1/ns2/node_a_1_2_DebugIdentity_1472563253536385"
    )

    datum = debug_data.DebugTensorDatum(dump_root, debug_dump_rel_path)

    self.assertEqual("DebugIdentity", datum.debug_op)
    self.assertEqual("ns1/ns2/node_a_1", datum.node_name)
    self.assertEqual(2, datum.output_slot)
    self.assertEqual("ns1/ns2/node_a_1:2", datum.tensor_name)
    self.assertEqual(1472563253536385, datum.timestamp)
    self.assertEqual("ns1/ns2/node_a_1:2:DebugIdentity", datum.watch_key)
    self.assertEqual(
        os.path.join(dump_root, debug_dump_rel_path), datum.file_path
    )
    self.assertEqual(
        "{DebugTensorDatum (/job:localhost/replica:0/task:0/cpu:0) "
        "%s:%d @ %s @ %d}"
        % (datum.node_name, datum.output_slot, datum.debug_op, datum.timestamp),
        str(datum),
    )
    self.assertEqual(
        "{DebugTensorDatum (/job:localhost/replica:0/task:0/cpu:0) "
        "%s:%d @ %s @ %d}"
        % (datum.node_name, datum.output_slot, datum.debug_op, datum.timestamp),
        repr(datum),
    )

  def testDumpSizeBytesIsNoneForNonexistentFilePath(self):
    # [FUNCTION DOC]
    # Builds a datum whose file does not exist and asserts the dump size is null.
    dump_root = "/tmp/tfdbg_1"
    debug_dump_rel_path = "ns1/ns2/node_foo_1_2_DebugIdentity_1472563253536385"
    datum = debug_data.DebugTensorDatum(dump_root, debug_dump_rel_path)

    self.assertIsNone(datum.dump_size_bytes)


class DebugDumpDirTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests the dump directory class: error handling, multi-device dumps, duplicate node names,
  # empty directories, filesystem pattern calls and slash normalization. Each test works in a
  # fresh temporary dump root directory.

  def setUp(self):
    # [FUNCTION DOC]
    # Creates a temporary directory and stores its path as the dump root.
    self._dump_root = tempfile.mkdtemp()

  def tearDown(self):
    # [FUNCTION DOC]
    # Recursively deletes the temporary dump root directory.
    # Tear down temporary dump directory.
    file_io.delete_recursively(self._dump_root)

  def _makeDataDirWithMultipleDevicesAndDuplicateNodeNames(self):
    # [FUNCTION DOC]
    # HELPER: fixture builder for the dump directory tests.
    # Creates three device directories (a CPU and two GPUs) under the dump root, each holding one
    # empty dump file for the same node name, with timestamps ending in 86 (CPU), 85 (first
    # GPU) and 87 (second GPU). Returns nothing. Opened file handles are not explicitly closed.
    cpu_0_dir = os.path.join(
        self._dump_root,
        debug_data.METADATA_FILE_PREFIX
        + debug_data.DEVICE_TAG
        + ",job_localhost,replica_0,task_0,cpu_0",
    )
    gpu_0_dir = os.path.join(
        self._dump_root,
        debug_data.METADATA_FILE_PREFIX
        + debug_data.DEVICE_TAG
        + ",job_localhost,replica_0,task_0,device_GPU_0",
    )
    gpu_1_dir = os.path.join(
        self._dump_root,
        debug_data.METADATA_FILE_PREFIX
        + debug_data.DEVICE_TAG
        + ",job_localhost,replica_0,task_0,device_GPU_1",
    )
    os.makedirs(cpu_0_dir)
    os.makedirs(gpu_0_dir)
    os.makedirs(gpu_1_dir)
    open(
        os.path.join(cpu_0_dir, "node_foo_1_2_DebugIdentity_1472563253536386"),
        "wb",
    )
    open(
        os.path.join(gpu_0_dir, "node_foo_1_2_DebugIdentity_1472563253536385"),
        "wb",
    )
    open(
        os.path.join(gpu_1_dir, "node_foo_1_2_DebugIdentity_1472563253536387"),
        "wb",
    )

  def testDebugDumpDir_nonexistentDumpRoot(self):
    # [FUNCTION DOC]
    # Constructing the dump directory object for a nonexistent path must raise an IO error whose
    # message says the directory does not exist.
    with self.assertRaisesRegex(IOError, "does not exist"):
      debug_data.DebugDumpDir(tempfile.mkdtemp() + "_foo")

  def testDebugDumpDir_invalidFileNamingPattern(self):
    # [FUNCTION DOC]
    # Places a dump file with too few underscore-separated fields in a device directory and
    # expects the dump directory constructor to raise a value error about the naming pattern.
    # File name with too few underscores should lead to an exception.
    device_dir = os.path.join(
        self._dump_root,
        debug_data.METADATA_FILE_PREFIX
        + debug_data.DEVICE_TAG
        + ",job_localhost,replica_0,task_0,cpu_0",
    )
    os.makedirs(device_dir)
    open(os.path.join(device_dir, "node1_DebugIdentity_1234"), "wb")

    with self.assertRaisesRegex(
        ValueError, "does not conform to the naming pattern"
    ):
      debug_data.DebugDumpDir(self._dump_root)

  def testDebugDumpDir_validDuplicateNodeNamesWithMultipleDevices(self):
    # [FUNCTION DOC]
    # Builds the three-device fixture and one graph per device, each with a single node of the
    # same name. Loads the dump directory with those graphs and asserts: the three devices are
    # listed; the time origin is the smallest timestamp; size is three; listing nodes for an
    # unknown device raises a value error; listing all nodes yields the name three times; listing
    # for the CPU device yields it once.
    self._makeDataDirWithMultipleDevicesAndDuplicateNodeNames()

    graph_cpu_0 = graph_pb2.GraphDef()
    node = graph_cpu_0.node.add()
    node.name = "node_foo_1"
    node.op = "FooOp"
    node.device = "/job:localhost/replica:0/task:0/cpu:0"
    graph_gpu_0 = graph_pb2.GraphDef()
    node = graph_gpu_0.node.add()
    node.name = "node_foo_1"
    node.op = "FooOp"
    node.device = "/job:localhost/replica:0/task:0/device:GPU:0"
    graph_gpu_1 = graph_pb2.GraphDef()
    node = graph_gpu_1.node.add()
    node.name = "node_foo_1"
    node.op = "FooOp"
    node.device = "/job:localhost/replica:0/task:0/device:GPU:1"

    dump_dir = debug_data.DebugDumpDir(
        self._dump_root,
        partition_graphs=[graph_cpu_0, graph_gpu_0, graph_gpu_1],
    )

    self.assertCountEqual(
        [
            "/job:localhost/replica:0/task:0/cpu:0",
            "/job:localhost/replica:0/task:0/device:GPU:0",
            "/job:localhost/replica:0/task:0/device:GPU:1",
        ],
        dump_dir.devices(),
    )
    self.assertEqual(1472563253536385, dump_dir.t0)
    self.assertEqual(3, dump_dir.size)

    with self.assertRaisesRegex(ValueError, r"Invalid device name: "):
      dump_dir.nodes("/job:localhost/replica:0/task:0/device:GPU:2")
    self.assertCountEqual(
        ["node_foo_1", "node_foo_1", "node_foo_1"], dump_dir.nodes()
    )
    self.assertCountEqual(
        ["node_foo_1"],
        dump_dir.nodes(device_name="/job:localhost/replica:0/task:0/cpu:0"),
    )

  def testDuplicateNodeNamesInGraphDefOfSingleDeviceRaisesException(self):
    # [FUNCTION DOC]
    # Builds the three-device fixture and graphs where the second GPU's graph contains the same
    # node name twice. Expects loading the dump directory to raise a value error reporting a
    # duplicate node name on a device.
    self._makeDataDirWithMultipleDevicesAndDuplicateNodeNames()
    graph_cpu_0 = graph_pb2.GraphDef()
    node = graph_cpu_0.node.add()
    node.name = "node_foo_1"
    node.op = "FooOp"
    node.device = "/job:localhost/replica:0/task:0/cpu:0"
    graph_gpu_0 = graph_pb2.GraphDef()
    node = graph_gpu_0.node.add()
    node.name = "node_foo_1"
    node.op = "FooOp"
    node.device = "/job:localhost/replica:0/task:0/device:GPU:0"
    graph_gpu_1 = graph_pb2.GraphDef()
    node = graph_gpu_1.node.add()
    node.name = "node_foo_1"
    node.op = "FooOp"
    node.device = "/job:localhost/replica:0/task:0/device:GPU:1"
    node = graph_gpu_1.node.add()  # Here is the duplicate.
    node.name = "node_foo_1"
    node.op = "FooOp"
    node.device = "/job:localhost/replica:0/task:0/device:GPU:1"

    with self.assertRaisesRegex(ValueError, r"Duplicate node name on device "):
      debug_data.DebugDumpDir(
          self._dump_root,
          partition_graphs=[graph_cpu_0, graph_gpu_0, graph_gpu_1],
      )

  def testDebugDumpDir_emptyDumpDir(self):
    # [FUNCTION DOC]
    # Loads an empty dump root and asserts the time origin is null and the dumped tensor data
    # list is empty.
    dump_dir = debug_data.DebugDumpDir(self._dump_root)

    self.assertIsNone(dump_dir.t0)
    self.assertEqual([], dump_dir.dumped_tensor_data)

  def testDebugDumpDir_usesGfileGlob(self):
    # [FUNCTION DOC]
    # Skipped on Windows (where the standard library matcher is used). Builds the fixture, patches
    # the filesystem pattern function with a fake returning nothing, loads the dump directory and
    # asserts the fake was called (in any order) for core metadata, fetches info, feed keys info
    # and device directory patterns under the dump root.
    if platform.system() == "Windows":
      self.skipTest("gfile.Glob is not used on Windows.")

    self._makeDataDirWithMultipleDevicesAndDuplicateNodeNames()

    def fake_gfile_glob(glob_pattern):
      # [FUNCTION DOC]
      # HELPER: stand-in for the filesystem pattern function; ignores the pattern and returns an
      # empty list.
      del glob_pattern
      return []

    with test.mock.patch.object(
        gfile, "Glob", side_effect=fake_gfile_glob, autospec=True
    ) as fake:
      debug_data.DebugDumpDir(self._dump_root)
      expected_calls = [
          test.mock.call(
              os.path.join(
                  self._dump_root,
                  (
                      debug_data.METADATA_FILE_PREFIX
                      + debug_data.CORE_METADATA_TAG
                      + "*"
                  ),
              )
          ),
          test.mock.call(
              os.path.join(
                  self._dump_root,
                  (
                      debug_data.METADATA_FILE_PREFIX
                      + debug_data.FETCHES_INFO_FILE_TAG
                      + "*"
                  ),
              )
          ),
          test.mock.call(
              os.path.join(
                  self._dump_root,
                  (
                      debug_data.METADATA_FILE_PREFIX
                      + debug_data.FEED_KEYS_INFO_FILE_TAG
                      + "*"
                  ),
              )
          ),
          test.mock.call(
              os.path.join(
                  self._dump_root,
                  (
                      debug_data.METADATA_FILE_PREFIX
                      + debug_data.DEVICE_TAG
                      + "*"
                  ),
              )
          ),
      ]
      fake.assert_has_calls(expected_calls, any_order=True)

  def testValidationSucceedsOnDoubleSlashNodeName(self):
    # [FUNCTION DOC]
    # Creates one dump file in a scope directory and a graph whose node name has a doubled slash.
    # Loading the dump directory with validation must succeed (names are normalized), with size
    # one and the first listed node name containing the single-slash form.
    device_dir = os.path.join(
        self._dump_root,
        debug_data.device_name_to_device_path(
            "/job:localhost/replica:0/task:0/cpu:0"
        ),
    )
    node_scope_dir = os.path.join(device_dir, "scope_A")
    os.makedirs(node_scope_dir)
    file_io.write_string_to_file(
        os.path.join(node_scope_dir, "op_B_0_DebugIdentity_12345"), "dummy"
    )
    graph_def = graph_pb2.GraphDef()
    node = graph_def.node.add()
    # Previously double slash would have caused validation to fail. b/429335661
    node.name = "scope_A//op_B"
    node.op = "NoOp"
    node.device = "/job:localhost/replica:0/task:0/cpu:0"
    dump_dir = debug_data.DebugDumpDir(
        self._dump_root, partition_graphs=[graph_def]
    )
    self.assertEqual(1, dump_dir.size)
    self.assertIn("scope_A/op_B", dump_dir.nodes()[0])


if __name__ == "__main__":
  googletest.main()
