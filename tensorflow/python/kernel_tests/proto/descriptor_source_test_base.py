# [MODULE DOC]
# HELPER: shared base class for the descriptor-source tests of the proto decode/encode ops.
# Defines DescriptorSourceTestBase, a test case that is parameterized by which module supplies the
# decode_proto and encode_proto ops. It builds a descriptor set by hand for the subset of the test
# example proto used by the simple test case, and checks that decoding then re-encoding a message
# reproduces the original bytes when the descriptor comes from a file or an in-memory bytes string,
# and that the local descriptor database fails. Concrete test files subclass it. No import-time
# side effects beyond imports.
# =============================================================================
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
"""Tests for proto ops reading descriptors from other sources."""
# Python3 preparedness imports.
import os

import numpy as np

from google.protobuf.descriptor_pb2 import FieldDescriptorProto
from google.protobuf.descriptor_pb2 import FileDescriptorSet
from tensorflow.python.framework import dtypes
from tensorflow.python.kernel_tests.proto import proto_op_test_base as test_base
from tensorflow.python.platform import test


class DescriptorSourceTestBase(test.TestCase):
  # [CLASS DOC]
  # HELPER: base class reused by concrete descriptor-source test modules.
  # Test case holding the decode and encode modules (supplied at construction) and the round-trip
  # helpers. It defines one real test (testWithFileDescriptorSet) that runs the round trip with
  # different descriptor sources.
  """Base class for testing descriptor sources."""

  def __init__(self, decode_module, encode_module, methodName='runTest'):  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # Constructor. Takes the module providing the decode op (decode_module), the module providing the
    # encode op (encode_module) and the test method name (default runTest), passes the name to the
    # base test case and stores both modules on the instance for later use. Returns nothing.
    """DescriptorSourceTestBase initializer.

    Args:
      decode_module: a module containing the `decode_proto_op` method
      encode_module: a module containing the `encode_proto_op` method
      methodName: the name of the test method (same as for test.TestCase)
    """

    super(DescriptorSourceTestBase, self).__init__(methodName)
    self._decode_module = decode_module
    self._encode_module = encode_module

  # NOTE: We generate the descriptor programmatically instead of via a compiler
  # because of differences between different versions of the compiler.
  #
  # The generated descriptor should capture the subset of `test_example.proto`
  # used in `test_base.simple_test_case()`.
  def _createDescriptorProto(self):
    # [FUNCTION DOC]
    # HELPER: builds the descriptor input for testWithFileDescriptorSet.
    # Programmatically builds a file descriptor set with two files: a types file defining a data type
    # enum, and a test example file defining the test case, test value and field spec messages with
    # the fields used by the simple test case. Takes no inputs and returns the descriptor set message.
    # No side effects.
    proto = FileDescriptorSet()

    file_proto = proto.file.add(
        name='types.proto', package='tensorflow', syntax='proto3')
    enum_proto = file_proto.enum_type.add(name='DataType')
    enum_proto.value.add(name='DT_DOUBLE', number=0)
    enum_proto.value.add(name='DT_BOOL', number=1)

    file_proto = proto.file.add(
        name='test_example.proto',
        package='tensorflow.contrib.proto',
        dependency=['types.proto'])
    message_proto = file_proto.message_type.add(name='TestCase')
    message_proto.field.add(
        name='values',
        number=1,
        type=FieldDescriptorProto.TYPE_MESSAGE,
        type_name='.tensorflow.contrib.proto.TestValue',
        label=FieldDescriptorProto.LABEL_REPEATED)
    message_proto.field.add(
        name='shapes',
        number=2,
        type=FieldDescriptorProto.TYPE_INT32,
        label=FieldDescriptorProto.LABEL_REPEATED)
    message_proto.field.add(
        name='sizes',
        number=3,
        type=FieldDescriptorProto.TYPE_INT32,
        label=FieldDescriptorProto.LABEL_REPEATED)
    message_proto.field.add(
        name='fields',
        number=4,
        type=FieldDescriptorProto.TYPE_MESSAGE,
        type_name='.tensorflow.contrib.proto.FieldSpec',
        label=FieldDescriptorProto.LABEL_REPEATED)

    message_proto = file_proto.message_type.add(
        name='TestValue')
    message_proto.field.add(
        name='double_value',
        number=1,
        type=FieldDescriptorProto.TYPE_DOUBLE,
        label=FieldDescriptorProto.LABEL_REPEATED)
    message_proto.field.add(
        name='bool_value',
        number=2,
        type=FieldDescriptorProto.TYPE_BOOL,
        label=FieldDescriptorProto.LABEL_REPEATED)

    message_proto = file_proto.message_type.add(
        name='FieldSpec')
    message_proto.field.add(
        name='name',
        number=1,
        type=FieldDescriptorProto.TYPE_STRING,
        label=FieldDescriptorProto.LABEL_OPTIONAL)
    message_proto.field.add(
        name='dtype',
        number=2,
        type=FieldDescriptorProto.TYPE_ENUM,
        type_name='.tensorflow.DataType',
        label=FieldDescriptorProto.LABEL_OPTIONAL)
    message_proto.field.add(
        name='value',
        number=3,
        type=FieldDescriptorProto.TYPE_MESSAGE,
        type_name='.tensorflow.contrib.proto.TestValue',
        label=FieldDescriptorProto.LABEL_OPTIONAL)

    return proto

  def _writeProtoToFile(self, proto):
    # [FUNCTION DOC]
    # HELPER: persists a descriptor set so it can be used as a file descriptor source.
    # Serializes the given proto and writes it to a file named descriptor.pb inside the test temp
    # directory (overwriting any existing file). Returns the file path.
    fn = os.path.join(self.get_temp_dir(), 'descriptor.pb')
    with open(fn, 'wb') as f:
      f.write(proto.SerializeToString())
    return fn

  def _testRoundtrip(self, descriptor_source):
    # [FUNCTION DOC]
    # HELPER: round-trip check parameterized by descriptor source.
    # Takes a descriptor source (a bytes string such as the local database marker, a file path, or a
    # bytes-prefixed serialized descriptor). Decodes a serialized simple test case with the decode
    # module, re-encodes the result with the encode module using the same source, runs both in a
    # session, and asserts the output has the same shape and identical bytes as the input. Errors from
    # the ops propagate to the caller. Returns nothing.
    # Numpy silently truncates the strings if you don't specify dtype=object.
    in_bufs = np.array(
        [test_base.ProtoOpTestBase.simple_test_case().SerializeToString()],
        dtype=object)
    message_type = 'tensorflow.contrib.proto.TestCase'
    field_names = ['values', 'shapes', 'sizes', 'fields']
    tensor_types = [dtypes.string, dtypes.int32, dtypes.int32, dtypes.string]

    with self.cached_session() as sess:
      sizes, field_tensors = self._decode_module.decode_proto(
          in_bufs,
          message_type=message_type,
          field_names=field_names,
          output_types=tensor_types,
          descriptor_source=descriptor_source)

      out_tensors = self._encode_module.encode_proto(
          sizes,
          field_tensors,
          message_type=message_type,
          field_names=field_names,
          descriptor_source=descriptor_source)

      out_bufs, = sess.run([out_tensors])

      # Check that the re-encoded tensor has the same shape.
      self.assertEqual(in_bufs.shape, out_bufs.shape)

      # Compare the input and output.
      for in_buf, out_buf in zip(in_bufs.flat, out_bufs.flat):
        # Check that the input and output serialized messages are identical.
        # If we fail here, there is a difference in the serialized
        # representation but the new serialization still parses. This could
        # be harmless (a change in map ordering?) or it could be bad (e.g.
        # loss of packing in the encoding).
        self.assertEqual(in_buf, out_buf)

  def testWithFileDescriptorSet(self):
    # [FUNCTION DOC]
    # Test: first asserts that using the local descriptor database fails with an op error saying no
    # descriptor was found for the message type. Then writes a hand-built descriptor set to a file and
    # asserts the round trip succeeds with the file path, and finally with the serialized descriptor
    # passed inline as a bytes string.
    # First try parsing with a local proto db, which should fail.
    with self.assertRaisesOpError('No descriptor found for message type'):
      self._testRoundtrip(b'local://')

    # Now try parsing with a FileDescriptorSet which contains the test proto.
    proto = self._createDescriptorProto()
    proto_file = self._writeProtoToFile(proto)
    self._testRoundtrip(proto_file)

    # Finally, try parsing the descriptor as a serialized string.
    self._testRoundtrip(b'bytes://' + proto.SerializeToString())
