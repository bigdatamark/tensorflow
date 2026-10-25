# [MODULE DOC]
# HELPER: shared base test case for the decode_proto op, parameterised by the module that provides
# the decode function, so several front ends can reuse the same tests.
# Defines DecodeProtoOpTestBase which decodes serialized test messages (binary, text, packed,
# disordered fields, sanitised input) through the supplied module and compares sizes and values
# with the expectations in the shared proto test cases, plus tests for corrupt input, unexpected
# output types and out-of-order repeated fields. Depends on the shared proto test base and the
# generated test message definitions. No import-time side effects; no script entry point.
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
"""Tests for decode_proto op."""

# Python3 preparedness imports.
import itertools

from absl.testing import parameterized
import numpy as np


from google.protobuf import text_format

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.kernel_tests.proto import proto_op_test_base as test_base
from tensorflow.python.kernel_tests.proto import test_example_pb2


class DecodeProtoOpTestBase(test_base.ProtoOpTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterised base test case for proto decoding. Stores the module that supplies the decode
  # function (chosen by the concrete subclass) and defines the shared decode tests, which run once
  # per named test case from the proto test base. Combines the proto test-case provider with the
  # parameterized test mixin.
  """Base class for testing proto decoding ops."""

  def __init__(self, decode_module, methodName='runTest'):  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # Constructor. Inputs: the module exposing the decode function, and the name of the test
    # method to run (optional, defaults to the standard run-test name). Passes the method name to
    # the parent constructor and stores the module for use by the tests. Returns nothing.
    """DecodeProtoOpTestBase initializer.

    Args:
      decode_module: a module containing the `decode_proto_op` method
      methodName: the name of the test method (same as for test.TestCase)
    """

    super(DecodeProtoOpTestBase, self).__init__(methodName)
    self._decode_module = decode_module

  def _compareValues(self, fd, vs, evs):
    # [FUNCTION DOC]
    # HELPER: compares decoded values of one field with the expected values.
    # Inputs: field descriptor, sequence of decoded values, sequence of expected values. Fails
    # the test if the lengths differ. Floats are compared with a relative closeness test; strings
    # are compared with the expected value encoded as ASCII bytes; other types by equality.
    # Returns nothing; raises on assertion failure.
    """Compare lists/arrays of field values."""

    if len(vs) != len(evs):
      self.fail('Field %s decoded %d outputs, expected %d' %
                (fd.name, len(vs), len(evs)))
    for i, ev in enumerate(evs):
      # Special case fuzzy match for float32. TensorFlow seems to mess with
      # MAX_FLT slightly and the test doesn't work otherwise.
      # TODO(nix): ask on TF list about why MAX_FLT doesn't pass through.
      if fd.cpp_type == fd.CPPTYPE_FLOAT:
        # Numpy isclose() is better than assertIsClose() which uses an absolute
        # value comparison.
        self.assertTrue(
            np.isclose(vs[i], ev), 'expected %r, actual %r' % (ev, vs[i]))
      elif fd.cpp_type == fd.CPPTYPE_STRING:
        # In Python3 string tensor values will be represented as bytes, so we
        # reencode the proto values to match that.
        self.assertEqual(vs[i], ev.encode('ascii'))
      else:
        # Doubles and other types pass through unscathed.
        self.assertEqual(vs[i], ev)

  def _compareProtos(self, batch_shape, sizes, fields, field_dict):
    # [FUNCTION DOC]
    # HELPER: checks decoded field tensors against the expected field specifications.
    # KNOWN ISSUE: the sizes argument is accepted but never used.
    # Inputs: batch shape (list), repeat-count matrix (unused), list of field specs with expected
    # values, and a mapping from field name to decoded array. For each field asserts the dtype and
    # that all but the last dimension equal the batch shape. Nested-message and extension fields
    # are parsed from bytes and compared as messages; other fields are mapped from the tensor
    # type (or the enum special case) to the matching value list and compared via the value
    # helper. Fails the test on an unhandled type. Returns nothing.
    """Compare protos of type TestValue.

    Args:
      batch_shape: the shape of the input tensor of serialized messages.
      sizes: int matrix of repeat counts returned by decode_proto
      fields: list of test_example_pb2.FieldSpec (types and expected values)
      field_dict: map from field names to decoded numpy tensors of values
    """

    # Check that expected values match.
    for field in fields:
      values = field_dict[field.name]
      self.assertEqual(dtypes.as_dtype(values.dtype), field.dtype)

      if 'ext_value' in field.name:
        fd = test_example_pb2.PrimitiveValue()
      else:
        fd = field.value.DESCRIPTOR.fields_by_name[field.name]

      # Values has the same shape as the input plus an extra
      # dimension for repeats.
      self.assertEqual(list(values.shape)[:-1], batch_shape)

      # Nested messages are represented as TF strings, requiring
      # some special handling.
      if field.name == 'message_value' or 'ext_value' in field.name:
        vs = []
        for buf in values.flat:
          msg = test_example_pb2.PrimitiveValue()
          msg.ParseFromString(buf)
          vs.append(msg)
        if 'ext_value' in field.name:
          evs = field.value.Extensions[test_example_pb2.ext_value]
        else:
          evs = getattr(field.value, field.name)
        if len(vs) != len(evs):
          self.fail('Field %s decoded %d outputs, expected %d' %
                    (fd.name, len(vs), len(evs)))
        for v, ev in zip(vs, evs):
          self.assertEqual(v, ev)
        continue

      tf_type_to_primitive_value_field = {
          dtypes.bool:
              'bool_value',
          dtypes.float32:
              'float_value',
          dtypes.float64:
              'double_value',
          dtypes.int8:
              'int8_value',
          dtypes.int32:
              'int32_value',
          dtypes.int64:
              'int64_value',
          dtypes.string:
              'string_value',
          dtypes.uint8:
              'uint8_value',
          dtypes.uint32:
              'uint32_value',
          dtypes.uint64:
              'uint64_value',
      }
      if field.name in ['enum_value', 'enum_value_with_default']:
        tf_field_name = 'enum_value'
      else:
        tf_field_name = tf_type_to_primitive_value_field.get(field.dtype)
      if tf_field_name is None:
        self.fail('Unhandled tensorflow type %d' % field.dtype)

      self._compareValues(fd, values.flat,
                          getattr(field.value, tf_field_name))

  def _runDecodeProtoTests(self, fields, case_sizes, batch_shape, batch,
                           message_type, message_format, sanitize,
                           force_disordered=False):
    # [FUNCTION DOC]
    # HELPER: runs one decode scenario end to end for the decode tests.
    # Inputs: field specs, expected size list, batch shape, list of serialized messages, message
    # type name, message format ('text' or 'binary'), sanitize flag, and an optional flag to
    # prepend a serialized message of extra high-numbered fields to every message so fields arrive
    # out of order (asserts sanitize is off in that case). Builds an object-typed array reshaped
    # to the batch shape, calls the decode function of the stored module in a cached session,
    # checks the sizes tensor shape and values, then compares decoded values via the proto
    # comparison helper. Returns nothing; raises on assertion failure.
    """Run decode tests on a batch of messages.

    Args:
      fields: list of test_example_pb2.FieldSpec (types and expected values)
      case_sizes: expected sizes array
      batch_shape: the shape of the input tensor of serialized messages
      batch: list of serialized messages
      message_type: descriptor name for messages
      message_format: format of messages, 'text' or 'binary'
      sanitize: whether to sanitize binary protobuf inputs
      force_disordered: whether to force fields encoded out of order.
    """

    if force_disordered:
      # Exercise code path that handles out-of-order fields by prepending extra
      # fields with tag numbers higher than any real field. Note that this won't
      # work with sanitization because that forces reserialization using a
      # trusted decoder and encoder.
      assert not sanitize
      extra_fields = test_example_pb2.ExtraFields()
      extra_fields.string_value = 'IGNORE ME'
      extra_fields.bool_value = False
      extra_msg = extra_fields.SerializeToString()
      batch = [extra_msg + msg for msg in batch]

    # Numpy silently truncates the strings if you don't specify dtype=object.
    batch = np.array(batch, dtype=object)
    batch = np.reshape(batch, batch_shape)

    field_names = [f.name for f in fields]
    output_types = [f.dtype for f in fields]

    with self.cached_session() as sess:
      sizes, vtensor = self._decode_module.decode_proto(
          batch,
          message_type=message_type,
          field_names=field_names,
          output_types=output_types,
          message_format=message_format,
          sanitize=sanitize)

      vlist = sess.run([sizes] + vtensor)
      sizes = vlist[0]
      # Values is a list of tensors, one for each field.
      value_tensors = vlist[1:]

      # Check that the repeat sizes are correct.
      self.assertTrue(
          np.all(np.array(sizes.shape) == batch_shape + [len(field_names)]))

      # Check that the decoded sizes match the expected sizes.
      self.assertEqual(len(sizes.flat), len(case_sizes))
      self.assertTrue(
          np.all(sizes.flat == np.array(
              case_sizes, dtype=np.int32)))

      field_dict = dict(zip(field_names, value_tensors))

      self._compareProtos(batch_shape, sizes, fields, field_dict)

  @parameterized.named_parameters(*test_base.ProtoOpTestBase.named_parameters())
  def testBinary(self, case):
    # [FUNCTION DOC]
    # Parameterised scenario: serializes each test value to binary and decodes the batch without
    # sanitisation using the test value message type; sizes and values must match the case.
    batch = [value.SerializeToString() for value in case.values]
    self._runDecodeProtoTests(
        case.fields,
        case.sizes,
        list(case.shapes),
        batch,
        'tensorflow.contrib.proto.TestValue',
        'binary',
        sanitize=False)

  @parameterized.named_parameters(*test_base.ProtoOpTestBase.named_parameters())
  def testBinaryDisordered(self, case):
    # [FUNCTION DOC]
    # Parameterised scenario: like the binary test, but every serialized message is prefixed with
    # extra unknown fields so the decoder must handle out-of-order fields; sanitisation is off.
    batch = [value.SerializeToString() for value in case.values]
    self._runDecodeProtoTests(
        case.fields,
        case.sizes,
        list(case.shapes),
        batch,
        'tensorflow.contrib.proto.TestValue',
        'binary',
        sanitize=False,
        force_disordered=True)

  @parameterized.named_parameters(
      *test_base.ProtoOpTestBase.named_parameters(extension=False))
  def testPacked(self, case):
    # [FUNCTION DOC]
    # Parameterised scenario (cases without extensions): converts each test value to text, parses
    # that text into the packed-field variant of the message, serializes it, and decodes the batch
    # with the packed message type; sizes and values must match the case.
    # Now try with the packed serialization.
    #
    # We test the packed representations by loading the same test case using
    # PackedTestValue instead of TestValue. To do this we rely on the text
    # format being the same for packed and unpacked fields, and reparse the
    # test message using the packed version of the proto.
    packed_batch = [
        text_format.Parse(
            text_format.MessageToString(value),
            test_example_pb2.PackedTestValue(),
        ).SerializeToString()
        for value in case.values
    ]

    self._runDecodeProtoTests(
        case.fields,
        case.sizes,
        list(case.shapes),
        packed_batch,
        'tensorflow.contrib.proto.PackedTestValue',
        'binary',
        sanitize=False)

  @parameterized.named_parameters(*test_base.ProtoOpTestBase.named_parameters())
  def testText(self, case):
    # [FUNCTION DOC]
    # Parameterised scenario: renders each test value in protobuf text format and decodes the
    # batch with the text message format and no sanitisation; sizes and values must match.
    text_batch = [text_format.MessageToString(value) for value in case.values]

    self._runDecodeProtoTests(
        case.fields,
        case.sizes,
        list(case.shapes),
        text_batch,
        'tensorflow.contrib.proto.TestValue',
        'text',
        sanitize=False)

  @parameterized.named_parameters(*test_base.ProtoOpTestBase.named_parameters())
  def testSanitizerGood(self, case):
    # [FUNCTION DOC]
    # Parameterised scenario: decodes valid binary messages with sanitisation enabled (messages
    # are re-serialised by a trusted codec first); results must match the case.
    batch = [value.SerializeToString() for value in case.values]
    self._runDecodeProtoTests(
        case.fields,
        case.sizes,
        list(case.shapes),
        batch,
        'tensorflow.contrib.proto.TestValue',
        'binary',
        sanitize=True)

  @parameterized.parameters((False), (True))
  def testCorruptProtobuf(self, sanitize):
    # [FUNCTION DOC]
    # Parameterised on the sanitize flag (false and true). Scenario: decodes a plain non-protobuf
    # string as a binary message of the test case type, requesting one integer field. Expects a
    # data-loss error saying the binary protobuf could not be parsed (or the buffer was not fully
    # consumed).
    corrupt_proto = 'This is not a binary protobuf'

    # Numpy silently truncates the strings if you don't specify dtype=object.
    batch = np.array(corrupt_proto, dtype=object)
    msg_type = 'tensorflow.contrib.proto.TestCase'
    field_names = ['sizes']
    field_types = [dtypes.int32]

    with self.assertRaisesRegex(
        errors.DataLossError, 'Unable to parse binary protobuf'
        '|Failed to consume entire buffer'):
      self.evaluate(
          self._decode_module.decode_proto(
              batch,
              message_type=msg_type,
              field_names=field_names,
              output_types=field_types,
              sanitize=sanitize))

  def testUnexpectedType(self):
    # [FUNCTION DOC]
    # Scenario: serializes a message with an int64 field and asks the decoder to output that field
    # as a string. Expects an invalid-argument error saying the output type is unexpected
    # (int64 versus string).
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        'Unexpected output type.*int64 to DT_STRING'):
      msg = test_example_pb2.TestValue(
          int64_value_with_default=3
      ).SerializeToString()
      msg_type = 'tensorflow.contrib.proto.TestValue'
      field_names = ['int64_value_with_default']
      field_types = [dtypes.string]
      self.evaluate(
          self._decode_module.decode_proto(
              msg,
              message_type=msg_type,
              field_names=field_names,
              output_types=field_types,
          )
      )

  def testOutOfOrderRepeated(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the comment says all combinations of two fields are tested, but the loop over
    # the number of fields to parse stops before two, so only zero and one field subsets are run
    # and the case of parsing both fields is never exercised.
    # Scenario: builds three message fragments (one double field and two nested message entries),
    # concatenates them in every permutation, and decodes selected field subsets. Asserts that the
    # double value and the nested message values are returned in the expected order (which depends
    # on which message fragment appears first in the permutation), with a failure message naming
    # the permutation and combination.
    fragments = [
        test_example_pb2.TestValue(double_value=[1.0]).SerializeToString(),
        test_example_pb2.TestValue(
            message_value=[test_example_pb2.PrimitiveValue(
                string_value='abc')]).SerializeToString(),
        test_example_pb2.TestValue(
            message_value=[test_example_pb2.PrimitiveValue(
                string_value='def')]).SerializeToString()
    ]
    all_fields_to_parse = ['double_value', 'message_value']
    field_types = {
        'double_value': dtypes.double,
        'message_value': dtypes.string,
    }
    # Test against all 3! permutations of fragments, and for each permutation
    # test parsing all possible combination of 2 fields.
    for indices in itertools.permutations(range(len(fragments))):
      proto = b''.join(fragments[i] for i in indices)
      for i in indices:
        if i == 1:
          expected_message_values = [
              test_example_pb2.PrimitiveValue(
                  string_value='abc').SerializeToString(),
              test_example_pb2.PrimitiveValue(
                  string_value='def').SerializeToString(),
          ]
          break
        if i == 2:
          expected_message_values = [
              test_example_pb2.PrimitiveValue(
                  string_value='def').SerializeToString(),
              test_example_pb2.PrimitiveValue(
                  string_value='abc').SerializeToString(),
          ]
          break

      expected_field_values = {
          'double_value': [[1.0]],
          'message_value': [expected_message_values],
      }

      for num_fields_to_parse in range(len(all_fields_to_parse)):
        for comb in itertools.combinations(
            all_fields_to_parse, num_fields_to_parse):
          parsed_values = self.evaluate(
              self._decode_module.decode_proto(
                  [proto],
                  message_type='tensorflow.contrib.proto.TestValue',
                  field_names=comb,
                  output_types=[field_types[f] for f in comb],
                  sanitize=False)).values
          self.assertLen(parsed_values, len(comb))
          for field_name, parsed in zip(comb, parsed_values):
            self.assertAllEqual(parsed, expected_field_values[field_name],
                                'perm: {}, comb: {}'.format(indices, comb))
