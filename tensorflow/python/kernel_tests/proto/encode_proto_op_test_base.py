# [MODULE DOC]
# HELPER: shared base class for the encode_proto op tests.
# Provides a table-driven test base checking that encode_proto is a lossless inverse of decode_proto
# for the specified fields, plus negative tests for bad arguments (unknown field, wrong dtype, wrong
# sizes shape, inconsistent value shapes). Test cases come from the common proto op test base and are
# run both with the plain and the packed message variants. Concrete test modules subclass it and
# supply the ops modules. No import-time side effects beyond imports.
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
"""Table-driven test for encode_proto op.

It tests that encode_proto is a lossless inverse of decode_proto
(for the specified fields).
"""
# Python3 readiness boilerplate
from absl.testing import parameterized
import numpy as np

from google.protobuf import text_format

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.kernel_tests.proto import proto_op_test_base as test_base
from tensorflow.python.kernel_tests.proto import test_example_pb2
from tensorflow.python.ops import array_ops


class EncodeProtoOpTestBase(test_base.ProtoOpTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # HELPER: base class for encode_proto tests.
  # Combines the shared proto op test fixtures with parameterized test support. Holds the decode and
  # encode modules given at construction and defines the error tests and the round-trip tests.
  """Base class for testing proto encoding ops."""

  def __init__(self, decode_module, encode_module, methodName='runTest'):  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # Constructor. Takes the module providing decode_proto, the module providing encode_proto and the
    # test method name (default runTest). Calls the base constructor with the method name and stores
    # both modules on the instance. Returns nothing.
    """EncodeProtoOpTestBase initializer.

    Args:
      decode_module: a module containing the `decode_proto_op` method
      encode_module: a module containing  the `encode_proto_op` method
      methodName: the name of the test method (same as for test.TestCase)
    """

    super(EncodeProtoOpTestBase, self).__init__(methodName)
    self._decode_module = decode_module
    self._encode_module = encode_module

  def testBadSizesShape(self):
    # [FUNCTION DOC]
    # Test: passes a scalar sizes value to encode_proto. Expects an invalid-argument error about the
    # field shape when running eagerly, or a value error saying the shape must be at least rank 2 in
    # graph mode.
    if context.executing_eagerly():
      expected_error = (errors.InvalidArgumentError,
                        r'Invalid shape for field double_value.')
    else:
      expected_error = (ValueError,
                        r'Shape must be at least rank 2 but is rank 0')
    with self.assertRaisesRegex(*expected_error):
      self.evaluate(
          self._encode_module.encode_proto(
              sizes=1,
              values=[np.double(1.0)],
              message_type='tensorflow.contrib.proto.TestValue',
              field_names=['double_value']))

  def testBadInputs(self):
    # [FUNCTION DOC]
    # Test: asserts encode_proto rejects bad inputs with op errors: an unknown field name; a value of
    # the wrong dtype for a field; sizes with the wrong rank (checked eagerly or via placeholders in
    # graph mode); and values whose shapes disagree up to the last dimension. Each case checks the
    # specific error message.
    # Invalid field name
    with self.assertRaisesOpError('Unknown field: non_existent_field'):
      self.evaluate(
          self._encode_module.encode_proto(
              sizes=[[1]],
              values=[np.array([[0.0]], dtype=np.int32)],
              message_type='tensorflow.contrib.proto.TestValue',
              field_names=['non_existent_field']))

    # Incorrect types.
    with self.assertRaisesOpError('Incompatible type for field double_value.'):
      self.evaluate(
          self._encode_module.encode_proto(
              sizes=[[1]],
              values=[np.array([[0.0]], dtype=np.int32)],
              message_type='tensorflow.contrib.proto.TestValue',
              field_names=['double_value']))

    # Incorrect shapes of sizes.
    for sizes_value in 1, np.array([[[0, 0]]]):
      with self.assertRaisesOpError(
          r'sizes should be batch_size \+ \[len\(field_names\)\]'):
        if context.executing_eagerly():
          self.evaluate(
              self._encode_module.encode_proto(
                  sizes=sizes_value,
                  values=[np.array([[0.0]])],
                  message_type='tensorflow.contrib.proto.TestValue',
                  field_names=['double_value']))
        else:
          with self.cached_session():
            sizes = array_ops.placeholder(dtypes.int32)
            values = array_ops.placeholder(dtypes.float64)
            self._encode_module.encode_proto(
                sizes=sizes,
                values=[values],
                message_type='tensorflow.contrib.proto.TestValue',
                field_names=['double_value']).eval(feed_dict={
                    sizes: sizes_value,
                    values: [[0.0]]
                })

    # Inconsistent shapes of values.
    with self.assertRaisesOpError('Values must match up to the last dimension'):
      if context.executing_eagerly():
        self.evaluate(
            self._encode_module.encode_proto(
                sizes=[[1, 1]],
                values=[np.array([[0.0]]),
                        np.array([[0], [0]])],
                message_type='tensorflow.contrib.proto.TestValue',
                field_names=['double_value', 'int32_value']))
      else:
        with self.cached_session():
          values1 = array_ops.placeholder(dtypes.float64)
          values2 = array_ops.placeholder(dtypes.int32)
          (self._encode_module.encode_proto(
              sizes=[[1, 1]],
              values=[values1, values2],
              message_type='tensorflow.contrib.proto.TestValue',
              field_names=['double_value', 'int32_value']).eval(feed_dict={
                  values1: [[0.0]],
                  values2: [[0], [0]]
              }))

  def _testRoundtrip(self, in_bufs, message_type, fields):
    # [FUNCTION DOC]
    # HELPER: shared round-trip check for the parameterized round-trip tests.
    # Takes an array of serialized messages, the message type name and a list of field specs (each
    # with a name and dtype). Decodes with the decode module, re-encodes with the encode module in a
    # session and asserts the output has the same shape, parses to equal messages, and is
    # byte-identical to the input. Returns nothing.

    field_names = [f.name for f in fields]
    out_types = [f.dtype for f in fields]

    with self.cached_session() as sess:
      sizes, field_tensors = self._decode_module.decode_proto(
          in_bufs,
          message_type=message_type,
          field_names=field_names,
          output_types=out_types)

      out_tensors = self._encode_module.encode_proto(
          sizes,
          field_tensors,
          message_type=message_type,
          field_names=field_names)

      out_bufs, = sess.run([out_tensors])

      # Check that the re-encoded tensor has the same shape.
      self.assertEqual(in_bufs.shape, out_bufs.shape)

      # Compare the input and output.
      for in_buf, out_buf in zip(in_bufs.flat, out_bufs.flat):
        in_obj = test_example_pb2.TestValue()
        in_obj.ParseFromString(in_buf)

        out_obj = test_example_pb2.TestValue()
        out_obj.ParseFromString(out_buf)

        # Check that the deserialized objects are identical.
        self.assertEqual(in_obj, out_obj)

        # Check that the input and output serialized messages are identical.
        # If we fail here, there is a difference in the serialized
        # representation but the new serialization still parses. This could
        # be harmless (a change in map ordering?) or it could be bad (e.g.
        # loss of packing in the encoding).
        self.assertEqual(in_buf, out_buf)

  @parameterized.named_parameters(
      *test_base.ProtoOpTestBase.named_parameters(extension=False))
  def testRoundtrip(self, case):
    # [FUNCTION DOC]
    # Parameterized test over the shared cases (without extensions). Serializes the case values into
    # an object-dtype array reshaped to the case shapes, then runs the round-trip check against the
    # plain test value message type. Returns the helper result (nothing).
    in_bufs = [value.SerializeToString() for value in case.values]

    # np.array silently truncates strings if you don't specify dtype=object.
    in_bufs = np.reshape(np.array(in_bufs, dtype=object), list(case.shapes))
    return self._testRoundtrip(
        in_bufs, 'tensorflow.contrib.proto.TestValue', case.fields)

  @parameterized.named_parameters(
      *test_base.ProtoOpTestBase.named_parameters(extension=False))
  def testRoundtripPacked(self, case):
    # [FUNCTION DOC]
    # Parameterized test over the shared cases (without extensions) using packed encoding. Each value
    # is converted to text format and re-parsed as the packed test value message (the text form is
    # identical for packed and unpacked fields), serialized, reshaped to the case shapes, and run
    # through the round-trip check with the packed message type. Returns the helper result (nothing).
    # Now try with the packed serialization.
    # We test the packed representations by loading the same test cases using
    # PackedTestValue instead of TestValue. To do this we rely on the text
    # format being the same for packed and unpacked fields, and reparse the test
    # message using the packed version of the proto.
    in_bufs = [
        text_format.Parse(
            text_format.MessageToString(value),
            test_example_pb2.PackedTestValue(),
        ).SerializeToString()
        for value in case.values
    ]

    # np.array silently truncates strings if you don't specify dtype=object.
    in_bufs = np.reshape(np.array(in_bufs, dtype=object), list(case.shapes))
    return self._testRoundtrip(
        in_bufs, 'tensorflow.contrib.proto.PackedTestValue', case.fields)
