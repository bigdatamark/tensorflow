# [MODULE DOC]
# Tests for the node file writer, a runtime feature that records the node definitions of executed
# ops (with input shapes, dtypes and small host-memory int32 input values) to files in a directory
# named by an environment variable. The tests run eager ops and parse the written file. Helper
# methods decode the length-prefixed binary records. Not a helper module.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

import io
import os
import struct
import tempfile

import numpy as np

from tensorflow.core.framework import node_def_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.framework.test_util import IsMklEnabled
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.platform import test


@test_util.with_eager_op_as_function
class NodeFileWriterTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the node file writer. Class-level setup points the writer directory environment
  # variable at a temporary directory, triggers a matrix multiply to create the node file and finds
  # it (one file for the GPU or CPU device). Each test opens that file positioned at its end so it
  # only sees newly written records. Runs with eager op as function variants.
  """Tests for NodeFileWriter."""

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Class fixture. Creates a temporary directory and stores it on the class, sets the process
    # environment variable for the writer directory (side effect, never restored), runs a small
    # eager matrix multiply to initialize the writer, then locates the single node-definitions file
    # for device index 0 (GPU if any physical GPU exists, else CPU) and stores its path on the
    # class. Fails with an assertion if there is not exactly one such file or its name does not
    # start with the node definitions prefix.
    super().setUpClass()
    # Set TF_NODE_FILE_WRITER_DIRECTORY, which is where NodeFileWriter will
    # write to.
    cls.node_dir = tempfile.TemporaryDirectory(suffix='NodeFileWriterTest')
    os.environ['TF_NODE_FILE_WRITER_DIRECTORY'] = cls.node_dir.name
    # Initializes the NodeFileWriter, causing the node file to be created.
    with context.eager_mode():
      gen_math_ops.mat_mul(array_ops.ones((1, 1)), array_ops.ones((1, 1)))
    # Find the node file.
    device = 'GPU' if config.list_physical_devices('GPU') else 'CPU'
    files_with_device = {
        file for file in os.listdir(cls.node_dir.name)
        if f'_{device}_0_' in file
    }
    assert len(files_with_device) == 1, (
        f'Expected to create exactly one test_nodes file in directory '
        f'{cls.node_dir.name} with string _{device}_0_ but found '
        f'{len(files_with_device)}: {files_with_device}')
    (file,) = files_with_device
    assert file.startswith('node_defs_')
    cls.node_filename = os.path.join(cls.node_dir.name, file)

  @classmethod
  def tearDownClass(cls):
    # [FUNCTION DOC]
    # Class fixture teardown. Calls the parent teardown and then deletes the temporary directory
    # created in class setup.
    super().tearDownClass()
    cls.node_dir.cleanup()

  def setUp(self):
    # [FUNCTION DOC]
    # Per-test fixture. Opens the node file for binary reading and seeks to its end so that only
    # node definitions written during the test are read.
    super().setUp()
    self.node_file = open(self.node_filename, 'rb')
    # Seek to end of file, so only newly written NodeDefs are read in each test.
    self.node_file.seek(0, io.SEEK_END)

  def tearDown(self):
    # [FUNCTION DOC]
    # Per-test teardown. Calls the parent teardown and closes the node file handle.
    super().tearDown()
    self.node_file.close()

  def _get_new_node_defs(self):
    # [FUNCTION DOC]
    # HELPER: test utility for reading node definitions written by the writer in these tests.
    # Reads everything appended to the node file since the last call. Each record is an 8-byte
    # little-endian unsigned size followed by a serialized node definition. Parses each, skipping
    # internal ops (argument, return value, no-op, and receive variants) when eager op as function
    # is enabled. Returns the list of remaining node definitions. Asserts that the whole file
    # content was consumed exactly.
    """Gets new NodeDefs written by the NodeFileWriter.

    Returns:
      A list of new NodeDefs in the file written by NodeDefWriter since the last
      time this method was called.
    """
    node_def_bytes = self.node_file.read()
    node_defs = []
    cur_pos = 0
    while cur_pos < len(node_def_bytes):
      size_bytes = node_def_bytes[cur_pos:cur_pos + 8]
      (size,) = struct.unpack('<Q', size_bytes)
      cur_pos += 8
      node_def = node_def_pb2.NodeDef()
      node_def.ParseFromString(node_def_bytes[cur_pos:cur_pos + size])
      # When running eager op as function is enabled we expect these extra nodes
      # to show up in the list of executed nodes.
      ignored_ops = []
      if context.run_eager_op_as_function_enabled():
        ignored_ops.extend(['_Arg', '_Retval', 'NoOp'])
        # TODO(b/206047926): Fix or remove _Recv/_HostRecv from the ignored_ops.
        ignored_ops.extend(['_Recv', '_HostRecv'])
      if node_def.op not in ignored_ops:
        node_defs.append(node_def)
      cur_pos += size
    self.assertEqual(cur_pos, len(node_def_bytes))
    return node_defs

  def _get_input_shapes(self, node_def):
    # [FUNCTION DOC]
    # HELPER: test utility. Given a node definition, returns a list of tuples of dimension sizes,
    # one per input, taken from the input shapes attribute.
    input_shapes = []
    for shape_attr in node_def.attr['_input_shapes'].list.shape:
      shape = tuple(a.size for a in shape_attr.dim)
      input_shapes.append(shape)
    return input_shapes

  def _get_input_dtypes(self, node_def):
    # [FUNCTION DOC]
    # HELPER: test utility. Given a node definition, returns the list of data types of its inputs
    # (converted to dtype objects) from the input dtypes attribute.
    input_dtypes = []
    for dtype_attr in node_def.attr['_input_dtypes'].list.type:
      input_dtypes.append(dtypes.as_dtype(dtype_attr))
    return input_dtypes

  def _get_input_tensor(self, node_def, input_index):
    # [FUNCTION DOC]
    # HELPER: test utility. Given a node definition and an input index, returns the recorded input
    # value as a numpy array, or nothing if that input's value was not recorded in the node def.
    tensor_proto = node_def.attr.get(f'_input_tensor_{input_index}')
    if tensor_proto is None:
      return None
    return tensor_util.MakeNdarray(tensor_proto.tensor)

  @test_util.disable_xla('b/201684914')
  def test_simple(self):
    # [FUNCTION DOC]
    # Test: in eager mode runs two matrix multiplies (float32 and float64) and asserts two node
    # definitions are written with the right op name (allowing the oneDNN-rewritten name when
    # enabled), input dtypes and shapes. Then runs another float32 multiply with a different first
    # input shape and asserts a new node def is written with the new shapes. Skipped under XLA.
    with context.eager_mode():
      x32 = constant_op.constant(np.ones((2, 3)).astype(np.float32))
      y32 = constant_op.constant(np.ones((3, 2)).astype(np.float32))
      x64 = constant_op.constant(np.ones((2, 3)).astype(np.float64))
      y64 = constant_op.constant(np.ones((3, 2)).astype(np.float64))
      gen_math_ops.mat_mul(x32, y32)
      gen_math_ops.mat_mul(x64, y64)
      node_defs = self._get_new_node_defs()
      self.assertLen(node_defs, 2)
      node_def1, node_def2 = node_defs  # pylint: disable=unbalanced-tuple-unpacking
      if not IsMklEnabled():
        self.assertEqual(node_def1.op, 'MatMul')
      else:
        # Under certain conditions ops can be rewritten by oneDNN optimization
        # pass.
        self.assertIn(node_def1.op, ['MatMul', '_MklMatMul'])

      self.assertEqual(
          self._get_input_dtypes(node_def1), [dtypes.float32, dtypes.float32])
      self.assertEqual(self._get_input_shapes(node_def1), [(2, 3), (3, 2)])
      self.assertEqual(node_def2.op, 'MatMul')
      self.assertEqual(
          self._get_input_dtypes(node_def2), [dtypes.float64, dtypes.float64])
      self.assertEqual(self._get_input_shapes(node_def2), [(2, 3), (3, 2)])

      # The node is written again if the input shapes are different
      x32 = constant_op.constant(np.ones((4, 3)).astype(np.float32))
      gen_math_ops.mat_mul(x32, y32)
      node_defs = self._get_new_node_defs()
      self.assertLen(node_defs, 1)
      (node_def3,) = node_defs  # pylint: disable=unbalanced-tuple-unpacking
      if not IsMklEnabled():
        self.assertEqual(node_def3.op, 'MatMul')
      else:
        # Under certain conditions ops can be rewritten by oneDNN optimization
        # pass.
        self.assertIn(node_def3.op, ['MatMul', '_MklMatMul'])
      self.assertEqual(
          self._get_input_dtypes(node_def3), [dtypes.float32, dtypes.float32])
      self.assertEqual(self._get_input_shapes(node_def3), [(4, 3), (3, 2)])

  @test_util.disable_xla('b/201684914')
  def test_host_int32_inputs(self):
    # [FUNCTION DOC]
    # Test: runs a pad op with float input, int32 paddings and float constant value in eager mode.
    # Asserts one node def is written with the pad op name, expected dtypes and shapes, and that
    # only the int32 host-memory paddings input has its value recorded (others are absent).
    # Skipped under XLA.
    with context.eager_mode():
      x = constant_op.constant(np.ones((2, 2)).astype(np.float32))
      paddings = constant_op.constant([[1, 2], [3, 4]])
      constant_values = constant_op.constant(0.)
      gen_array_ops.pad_v2(x, paddings, constant_values)
      node_defs = self._get_new_node_defs()
      self.assertLen(node_defs, 1)
      (node_def,) = node_defs  # pylint: disable=unbalanced-tuple-unpacking
      self.assertEqual(node_def.op, 'PadV2')
      self.assertEqual(
          self._get_input_dtypes(node_def),
          [dtypes.float32, dtypes.int32, dtypes.float32])
      self.assertEqual(self._get_input_shapes(node_def), [(2, 2), (2, 2), ()])
      self.assertIsNone(self._get_input_tensor(node_def, 0))
      self.assertAllEqual(
          self._get_input_tensor(node_def, 1), np.array([[1, 2], [3, 4]]))
      self.assertIsNone(self._get_input_tensor(node_def, 2))

  @test_util.disable_xla('b/201684914')
  def test_skipped_ops(self):
    # [FUNCTION DOC]
    # Test: verifies ops the writer skips. A cast op (on a hardcoded skip list) writes nothing. Two
    # convolutions with same shapes but different values write only one node def (duplicates are
    # skipped). A pad op whose int32 host input has more than 10 elements writes nothing. Skipped
    # under XLA.
    with context.eager_mode():
      x = constant_op.constant(np.ones((1, 1, 1, 1)).astype(np.float32))

      # Cast is on the hardcoded list of ops to skip
      gen_math_ops.cast(x, dtypes.float64)
      self.assertEmpty(self._get_new_node_defs())

      gen_nn_ops.conv2d(x, x, [1, 1, 1, 1], 'SAME')
      y = constant_op.constant(np.zeros((1, 1, 1, 1)).astype(np.float32))
      # Duplicate ops are skipped, even if input values are different
      gen_nn_ops.conv2d(x, y, [1, 1, 1, 1], 'SAME')
      self.assertLen(self._get_new_node_defs(), 1)

      x = constant_op.constant(np.ones((1, 1, 1, 1, 1, 1)).astype(np.float32))
      paddings = constant_op.constant(np.ones((6, 2)).astype(np.int32))
      constant_values = constant_op.constant(0.)
      # If an host int32 input has more than 10 elements, the op is skipped
      gen_array_ops.pad_v2(x, paddings, constant_values)
      self.assertEmpty(self._get_new_node_defs())


if __name__ == '__main__':
  test.main()
