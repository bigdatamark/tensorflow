# [MODULE DOC]
# KNOWN ISSUE: in the equality test class the not-equal helper is spelled with a plural name
# that the tests never call, so their not-equal assertions fall back to the built-in text
# inequality check on the raw strings rather than the message comparison under test.
# Unit tests for the protocol buffer comparison utilities in the sibling compare module, using the
# generated test messages Large, Medium, Small, Labeled and WithMap. Covers: structural equality
# of messages (ProtoEq), float closeness (including NaN, infinity and subnormal values),
# normalization of numeric fields, the assertion function with its diff-style failure messages,
# and the mixin class. Runs through the standard test runner when executed as a script.
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
"""Tests for python.util.protobuf.compare."""

import copy
import re
import sys
import textwrap


from google.protobuf import text_format

from tensorflow.python.platform import googletest
from tensorflow.python.util.protobuf import compare
from tensorflow.python.util.protobuf import compare_test_pb2


def LargePbs(*args):
  # [FUNCTION DOC]
  # HELPER: shared test utility. Inputs: any number of text-format strings. Parses each into a new
  # Large test message (merging text into an empty message) and returns the list of messages in the
  # same order. Raises a parse error on malformed text.
  """Converts ASCII string Large PBs to messages."""
  return [text_format.Merge(arg, compare_test_pb2.Large()) for arg in args]


class ProtoEqTest(googletest.TestCase):
  # [CLASS DOC]
  # KNOWN ISSUE: the not-equal helper below is named with a plural form that unittest never calls;
  # tests call the singular not-equal assertion, which just compares the raw strings.
  # Tests for the structural message equality function. Overrides the equality assertion so that
  # its two text arguments are parsed into Large messages and compared with that function.

  def assertNotEquals(self, a, b):
    # [FUNCTION DOC]
    # HELPER: custom assertion helper for this test class.
    # KNOWN ISSUE: never invoked; tests call the singular built-in not-equal method, so this
    # message-level check does not run.
    # Inputs: a and b, text-format strings. Parses both into Large messages and asserts that the
    # equality function returns false. Fails the test otherwise.
    """Asserts that ProtoEq says a != b."""
    a, b = LargePbs(a, b)
    googletest.TestCase.assertEqual(self, compare.ProtoEq(a, b), False)

  def assertEqual(self, a, b):
    # [FUNCTION DOC]
    # HELPER: overrides the standard equality assertion for this class (without the optional message
    # argument). Inputs: a and b, text-format strings. Parses both into Large messages and asserts
    # that the equality function returns true; fails the test otherwise.
    """Asserts that ProtoEq says a == b."""
    a, b = LargePbs(a, b)
    googletest.TestCase.assertEqual(self, compare.ProtoEq(a, b), True)

  def testPrimitives(self):
    # [FUNCTION DOC]
    # Test: equality of plain strings. Asserts the function reports equal for identical text and
    # not equal for different text.
    googletest.TestCase.assertEqual(self, True, compare.ProtoEq('a', 'a'))
    googletest.TestCase.assertEqual(self, False, compare.ProtoEq('b', 'a'))

  def testEmpty(self):
    # [FUNCTION DOC]
    # Test: two empty messages are equal.
    self.assertEqual('', '')

  def testPrimitiveFields(self):
    # [FUNCTION DOC]
    # Test: single-valued string, 64-bit integer, float, boolean and enumeration fields. Asserts
    # that presence versus absence and different values count as different, and identical values
    # as equal. (The not-equal checks go through the built-in text comparison, see class notes.)
    self.assertNotEqual('string_: "a"', '')
    self.assertEqual('string_: "a"', 'string_: "a"')
    self.assertNotEqual('string_: "b"', 'string_: "a"')
    self.assertNotEqual('string_: "ab"', 'string_: "aa"')

    self.assertNotEqual('int64_: 0', '')
    self.assertEqual('int64_: 0', 'int64_: 0')
    self.assertNotEqual('int64_: -1', '')
    self.assertNotEqual('int64_: 1', 'int64_: 0')
    self.assertNotEqual('int64_: 0', 'int64_: -1')

    self.assertNotEqual('float_: 0.0', '')
    self.assertEqual('float_: 0.0', 'float_: 0.0')
    self.assertNotEqual('float_: -0.1', '')
    self.assertNotEqual('float_: 3.14', 'float_: 0')
    self.assertNotEqual('float_: 0', 'float_: -0.1')
    self.assertEqual('float_: -0.1', 'float_: -0.1')

    self.assertNotEqual('bool_: true', '')
    self.assertNotEqual('bool_: false', '')
    self.assertNotEqual('bool_: true', 'bool_: false')
    self.assertEqual('bool_: false', 'bool_: false')
    self.assertEqual('bool_: true', 'bool_: true')

    self.assertNotEqual('enum_: A', '')
    self.assertNotEqual('enum_: B', 'enum_: A')
    self.assertNotEqual('enum_: C', 'enum_: B')
    self.assertEqual('enum_: C', 'enum_: C')

  def testRepeatedPrimitives(self):
    # [FUNCTION DOC]
    # Test: repeated integer fields. Asserts that order, extra elements and duplicates matter, and
    # that identical lists are equal.
    self.assertNotEqual('int64s: 0', '')
    self.assertEqual('int64s: 0', 'int64s: 0')
    self.assertNotEqual('int64s: 1', 'int64s: 0')
    self.assertNotEqual('int64s: 0 int64s: 0', '')
    self.assertNotEqual('int64s: 0 int64s: 0', 'int64s: 0')
    self.assertNotEqual('int64s: 1 int64s: 0', 'int64s: 0')
    self.assertNotEqual('int64s: 0 int64s: 1', 'int64s: 0')
    self.assertNotEqual('int64s: 1', 'int64s: 0 int64s: 2')
    self.assertNotEqual('int64s: 2 int64s: 0', 'int64s: 1')
    self.assertEqual('int64s: 0 int64s: 0', 'int64s: 0 int64s: 0')
    self.assertEqual('int64s: 0 int64s: 1', 'int64s: 0 int64s: 1')
    self.assertNotEqual('int64s: 1 int64s: 0', 'int64s: 0 int64s: 0')
    self.assertNotEqual('int64s: 1 int64s: 0', 'int64s: 0 int64s: 1')
    self.assertNotEqual('int64s: 1 int64s: 0', 'int64s: 0 int64s: 2')
    self.assertNotEqual('int64s: 1 int64s: 1', 'int64s: 1 int64s: 0')
    self.assertNotEqual('int64s: 1 int64s: 1', 'int64s: 1 int64s: 0 int64s: 2')

  def testMessage(self):
    # [FUNCTION DOC]
    # Test: nested sub-message fields combined with scalar fields. Asserts that presence of a
    # sub-message, its contents and neighbouring scalar fields all affect equality.
    self.assertNotEqual('small <>', '')
    self.assertEqual('small <>', 'small <>')
    self.assertNotEqual('small < strings: "a" >', '')
    self.assertNotEqual('small < strings: "a" >', 'small <>')
    self.assertEqual('small < strings: "a" >', 'small < strings: "a" >')
    self.assertNotEqual('small < strings: "b" >', 'small < strings: "a" >')
    self.assertNotEqual(
        'small < strings: "a" strings: "b" >', 'small < strings: "a" >'
    )

    self.assertNotEqual('string_: "a"', 'small <>')
    self.assertNotEqual('string_: "a"', 'small < strings: "b" >')
    self.assertNotEqual('string_: "a"', 'small < strings: "b" strings: "c" >')
    self.assertNotEqual('string_: "a" small <>', 'small <>')
    self.assertNotEqual('string_: "a" small <>', 'small < strings: "b" >')
    self.assertEqual('string_: "a" small <>', 'string_: "a" small <>')
    self.assertNotEqual(
        'string_: "a" small < strings: "a" >', 'string_: "a" small <>'
    )
    self.assertEqual('string_: "a" small < strings: "a" >',
                     'string_: "a" small < strings: "a" >')
    self.assertNotEqual(
        'string_: "a" small < strings: "a" >',
        'int64_: 1 small < strings: "a" >',
    )
    self.assertNotEqual('string_: "a" small < strings: "a" >', 'int64_: 1')
    self.assertNotEqual('string_: "a"', 'int64_: 1 small < strings: "a" >')
    self.assertNotEqual(
        'string_: "a" int64_: 0 small < strings: "a" >',
        'int64_: 1 small < strings: "a" >',
    )
    self.assertNotEqual(
        'string_: "a" int64_: 1 small < strings: "a" >',
        'string_: "a" int64_: 0 small < strings: "a" >',
    )
    self.assertEqual('string_: "a" int64_: 0 small < strings: "a" >',
                     'string_: "a" int64_: 0 small < strings: "a" >')

  def testNestedMessage(self):
    # [FUNCTION DOC]
    # Test: two-level nesting (medium messages holding lists of small ones). Asserts that the
    # number and contents of nested messages and other fields affect equality.
    self.assertNotEqual('medium <>', '')
    self.assertEqual('medium <>', 'medium <>')
    self.assertNotEqual('medium < smalls <> >', 'medium <>')
    self.assertEqual('medium < smalls <> >', 'medium < smalls <> >')
    self.assertNotEqual(
        'medium < smalls <> smalls <> >', 'medium < smalls <> >'
    )
    self.assertEqual('medium < smalls <> smalls <> >',
                     'medium < smalls <> smalls <> >')

    self.assertNotEqual('medium < int32s: 0 >', 'medium < smalls <> >')

    self.assertNotEqual(
        'medium < smalls < strings: "a"> >', 'medium < smalls <> >'
    )

  def testTagOrder(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring lists the small field twice and omits some fields; it is only a
    # reference listing and nothing asserts on it.
    # Test: messages that set different subsets of fields (ordered by field number). Asserts that
    # different set-field combinations, values and message kinds are not equal, identical ones are.
    """Tests that different fields are ordered by tag number.

    For reference, here are the relevant tag numbers from compare_test.proto:
      optional string string_ = 1;
      optional int64 int64_ = 2;
      optional float float_ = 3;
      optional Small small = 8;
      optional Medium medium = 7;
      optional Small small = 8;
    """
    self.assertNotEqual(
        'string_: "a"                      ',
        '             int64_: 1            ',
    )
    self.assertNotEqual(
        'string_: "a" int64_: 2            ',
        '             int64_: 1            ',
    )
    self.assertNotEqual(
        'string_: "b" int64_: 1            ',
        'string_: "a" int64_: 2            ',
    )
    self.assertEqual('string_: "a" int64_: 1            ',
                     'string_: "a" int64_: 1            ')
    self.assertNotEqual(
        'string_: "a" int64_: 1 float_: 0.0',
        'string_: "a" int64_: 1            ',
    )
    self.assertEqual('string_: "a" int64_: 1 float_: 0.0',
                     'string_: "a" int64_: 1 float_: 0.0')
    self.assertNotEqual(
        'string_: "a" int64_: 1 float_: 0.1',
        'string_: "a" int64_: 1 float_: 0.0',
    )
    self.assertNotEqual(
        'string_: "a" int64_: 2 float_: 0.0',
        'string_: "a" int64_: 1 float_: 0.1',
    )
    self.assertNotEqual(
        'string_: "a"                      ',
        '             int64_: 1 float_: 0.1',
    )
    self.assertNotEqual(
        'string_: "a"           float_: 0.0',
        '             int64_: 1            ',
    )
    self.assertNotEqual(
        'string_: "b"           float_: 0.0',
        'string_: "a" int64_: 1            ',
    )

    self.assertNotEqual('string_: "a"', 'small < strings: "a" >')
    self.assertNotEqual(
        'string_: "a" small < strings: "a" >', 'small < strings: "b" >'
    )
    self.assertNotEqual(
        'string_: "a" small < strings: "b" >',
        'string_: "a" small < strings: "a" >',
    )
    self.assertEqual('string_: "a" small < strings: "a" >',
                     'string_: "a" small < strings: "a" >')

    self.assertNotEqual(
        'string_: "a" medium <>', 'string_: "a" small < strings: "a" >'
    )
    self.assertNotEqual(
        'string_: "a" medium < smalls <> >',
        'string_: "a" small < strings: "a" >',
    )
    self.assertNotEqual('medium <>', 'small < strings: "a" >')
    self.assertNotEqual('medium <> small <>', 'small < strings: "a" >')
    self.assertNotEqual('medium < smalls <> >', 'small < strings: "a" >')
    self.assertNotEqual(
        'medium < smalls < strings: "a" > >', 'small < strings: "b" >'
    )

  def testIsClose(self):
    # [FUNCTION DOC]
    # Test: the float closeness function. Asserts equal integers are close and that two nearby
    # values are close under a loose tolerance but not under a tight one.
    self.assertTrue(compare.isClose(1, 1, 1e-10))
    self.assertTrue(compare.isClose(65061.0420, 65061.0322, 1e-5))
    self.assertFalse(compare.isClose(65061.0420, 65061.0322, 1e-7))

  def testIsCloseNan(self):
    # [FUNCTION DOC]
    # Test: closeness with not-a-number values. Asserts two NaNs are close and a NaN is not close to
    # a number or to infinity, in either argument order.
    self.assertTrue(compare.isClose(float('nan'), float('nan'), 1e-10))
    self.assertFalse(compare.isClose(float('nan'), 1, 1e-10))
    self.assertFalse(compare.isClose(1, float('nan'), 1e-10))
    self.assertFalse(compare.isClose(float('nan'), float('inf'), 1e-10))

  def testIsCloseInf(self):
    # [FUNCTION DOC]
    # Test: closeness with infinities. Asserts same-signed infinities are close, opposite signs are
    # not, and an infinity is not close to a finite number in either order.
    self.assertTrue(compare.isClose(float('inf'), float('inf'), 1e-10))
    self.assertTrue(compare.isClose(float('-inf'), float('-inf'), 1e-10))
    self.assertFalse(compare.isClose(float('-inf'), float('inf'), 1e-10))
    self.assertFalse(compare.isClose(float('inf'), 1, 1e-10))
    self.assertFalse(compare.isClose(1, float('inf'), 1e-10))

  def testIsCloseSubnormal(self):
    # [FUNCTION DOC]
    # Test: closeness with a tiny subnormal-range value. Asserts it is close to itself and not close
    # to one.
    x = sys.float_info.min * sys.float_info.epsilon
    self.assertTrue(compare.isClose(x, x, 1e-10))
    self.assertFalse(compare.isClose(x, 1, 1e-10))


class NormalizeNumbersTest(googletest.TestCase):
  # [CLASS DOC]
  # Tests for the numeric-field normalization function using the Large and Medium test messages
  # and a message with maps.
  """Tests for NormalizeNumberFields()."""

  def testNormalizesInts(self):
    # [FUNCTION DOC]
    # Test: a 64-bit integer field, set through the constructor, by assignment and with a very large
    # value, stays an integer type after normalization. Asserts the type each time.
    pb = compare_test_pb2.Large(int64_=4)
    compare.NormalizeNumberFields(pb)
    self.assertIsInstance(pb.int64_, int)

    pb.int64_ = 4
    compare.NormalizeNumberFields(pb)
    self.assertIsInstance(pb.int64_, int)

    pb.int64_ = 9999999999999999
    compare.NormalizeNumberFields(pb)
    self.assertIsInstance(pb.int64_, int)

  def testNormalizesRepeatedInts(self):
    # [FUNCTION DOC]
    # Test: every element of a repeated 64-bit integer field is an integer after normalization.
    pb = compare_test_pb2.Large(int64s=[1, 400, 999999999999999])
    compare.NormalizeNumberFields(pb)
    self.assertIsInstance(pb.int64s[0], int)
    self.assertIsInstance(pb.int64s[1], int)
    self.assertIsInstance(pb.int64s[2], int)

  def testNormalizesFloats(self):
    # [FUNCTION DOC]
    # Test: two messages whose float field differs only beyond about six digits compare unequal
    # before normalization and equal after it.
    pb1 = compare_test_pb2.Large(float_=1.2314352351231)
    pb2 = compare_test_pb2.Large(float_=1.231435)
    self.assertNotEqual(pb1.float_, pb2.float_)
    compare.NormalizeNumberFields(pb1)
    compare.NormalizeNumberFields(pb2)
    self.assertEqual(pb1.float_, pb2.float_)

  def testNormalizesRepeatedFloats(self):
    # [FUNCTION DOC]
    # Test: elements of a repeated float field inside a nested message are rounded so that each is
    # approximately the same six-digit value after normalization.
    pb = compare_test_pb2.Large(
        medium=compare_test_pb2.Medium(floats=[0.111111111, 0.111111])
    )
    compare.NormalizeNumberFields(pb)
    for value in pb.medium.floats:
      self.assertAlmostEqual(0.111111, value)

  def testNormalizesDoubles(self):
    # [FUNCTION DOC]
    # Test: two messages whose double field differs only beyond seven digits compare unequal before
    # normalization and equal after it.
    pb1 = compare_test_pb2.Large(double_=1.2314352351231)
    pb2 = compare_test_pb2.Large(double_=1.2314352)
    self.assertNotEqual(pb1.double_, pb2.double_)
    compare.NormalizeNumberFields(pb1)
    compare.NormalizeNumberFields(pb2)
    self.assertEqual(pb1.double_, pb2.double_)

  def testNormalizesMaps(self):
    # [FUNCTION DOC]
    # Test: normalization over a message containing a map of messages and a map of strings.
    # Only checks that it completes without error; there are no assertions on the result.
    pb = compare_test_pb2.WithMap()
    pb.value_message[4].strings.extend(['a', 'b', 'c'])
    pb.value_string['d'] = 'e'
    compare.NormalizeNumberFields(pb)


class AssertTest(googletest.TestCase):
  # [CLASS DOC]
  # Tests for the proto assertion function, checking both passing cases and the exact diff text in
  # failure messages. Provides local helper methods to wrap text arguments into messages and to
  # check passing and failing cases.
  """Tests assertProtoEqual()."""

  def assertProtoEqual(self, a, b, **kwargs):
    # [FUNCTION DOC]
    # HELPER: local wrapper around the module under test. Inputs: a and b, either both text strings
    # (parsed into Large messages) or messages; extra keyword options are passed through. Calls the
    # assertion function with this test case. Fails the test on difference.
    if isinstance(a, str) and isinstance(b, str):
      a, b = LargePbs(a, b)
    compare.assertProtoEqual(self, a, b, **kwargs)

  def assertAll(self, a, **kwargs):
    # [FUNCTION DOC]
    # HELPER: asserts a value is equal to itself. Inputs: a, text or message; keyword options passed
    # through. Calls the wrapper assertion with a on both sides. Fails the test if it raises.
    """Checks that all possible asserts pass."""
    self.assertProtoEqual(a, a, **kwargs)

  def assertSameNotEqual(self, a, b):
    # [FUNCTION DOC]
    # HELPER: asserts that the assertion function fails. Inputs: a and b, text or messages.
    # Expects an assertion error from the wrapper and fails the test if none is raised.
    """Checks that assertProtoEqual() fails."""
    self.assertRaises(AssertionError, self.assertProtoEqual, a, b)

  def assertNone(self, a, b, message, **kwargs):
    # [FUNCTION DOC]
    # HELPER: test utility.
    # KNOWN ISSUE: the name suggests a none/null check, but the helper checks for a failure message.
    # Asserts that the assertion function fails with a given message. Inputs: a and b, text
    # or messages; message, the expected failure text (indentation removed, then matched literally
    # as a pattern); keyword options passed through. Expects an assertion error whose text matches.
    """Checks that all possible asserts fail with the given message."""
    message = re.escape(textwrap.dedent(message))
    self.assertRaisesRegex(AssertionError, message, self.assertProtoEqual, a, b,
                           **kwargs)

  def testCheckInitialized(self):
    # [FUNCTION DOC]
    # Test: the initialization check using a message with a required field. Cases: neither message
    # initialized (fails with initialization errors when checking, passes when not checking); only
    # the first initialized (fails with initialization errors, or with a diff showing the required
    # field when not checking); both initialized (passes when equal, fails with a diff otherwise).
    # neither is initialized
    a = compare_test_pb2.Labeled(optional=1)
    self.assertNone(a, a, 'Initialization errors: ', check_initialized=True)
    self.assertAll(a, check_initialized=False)

    # a is initialized, b isn't
    b = copy.deepcopy(a)
    a.required = 2
    self.assertNone(a, b, 'Initialization errors: ', check_initialized=True)
    self.assertNone(
        a,
        b,
        """
                    - required: 2
                      optional: 1
                    """,
        check_initialized=False)

    # both are initialized
    a = compare_test_pb2.Labeled(required=2)
    self.assertAll(a, check_initialized=True)
    self.assertAll(a, check_initialized=False)

    b = copy.deepcopy(a)
    b.required = 3
    message = """
              - required: 2
              ?           ^
              + required: 3
              ?           ^
              """
    self.assertNone(a, b, message, check_initialized=True)
    self.assertNone(a, b, message, check_initialized=False)

  def testAssertEqualWithStringArg(self):
    # [FUNCTION DOC]
    # Test: the first argument may be a text-format string. Asserts that it equals a Large message
    # with the same string and float values.
    pb = compare_test_pb2.Large(string_='abc', float_=1.234)
    compare.assertProtoEqual(self, """
          string_: 'abc'
          float_: 1.234
        """, pb)

  def testNormalizesNumbers(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite the name, number normalization is not enabled in this test.
    # Test: two identically built messages pass the assertion with default options.
    pb1 = compare_test_pb2.Large(int64_=4)
    pb2 = compare_test_pb2.Large(int64_=4)
    compare.assertProtoEqual(self, pb1, pb2)

  def testNormalizesFloat(self):
    # [FUNCTION DOC]
    # Test: with number normalization enabled, a double field set to a float value and to an
    # integer value of the same quantity pass as equal.
    pb1 = compare_test_pb2.Large(double_=4.0)
    pb2 = compare_test_pb2.Large(double_=4)
    compare.assertProtoEqual(self, pb1, pb2, normalize_numbers=True)

  def testLargeProtoData(self):
    # [FUNCTION DOC]
    # Test: messages whose text form exceeds sixty-five thousand characters (thousands of repeated
    # entries) take the diff-based path. Asserts that equal data passes and that data with an extra
    # entry raises an assertion error.
    # Proto size should be larger than 2**16.
    number_of_entries = 2**13
    string_value = 'dummystr'  # Has length of 2**3.
    pb1_txt = 'strings: "dummystr"\n' * number_of_entries
    pb2 = compare_test_pb2.Small(strings=[string_value] * number_of_entries)
    compare.assertProtoEqual(self, pb1_txt, pb2)

    with self.assertRaises(AssertionError):
      compare.assertProtoEqual(self, pb1_txt + 'strings: "Should fail."', pb2)

  def testPrimitives(self):
    # [FUNCTION DOC]
    # Test: a string field passes when identical and fails with a diff message marking the changed
    # line when values differ.
    self.assertAll('string_: "x"')
    self.assertNone('string_: "x"', 'string_: "y"', """
                    - string_: "x"
                    ?           ^
                    + string_: "y"
                    ?           ^
                    """)

  def testRepeatedPrimitives(self):
    # [FUNCTION DOC]
    # Test: repeated integers. Identical lists pass; reordered, extra or duplicate elements fail;
    # the failure text shows an added line or a removed and added pair with markers.
    self.assertAll('int64s: 0 int64s: 1')

    self.assertSameNotEqual('int64s: 0 int64s: 1', 'int64s: 1 int64s: 0')
    self.assertSameNotEqual('int64s: 0 int64s: 1 int64s: 2',
                            'int64s: 2 int64s: 1 int64s: 0')

    self.assertSameNotEqual('int64s: 0', 'int64s: 0 int64s: 0')
    self.assertSameNotEqual('int64s: 0 int64s: 1',
                            'int64s: 1 int64s: 0 int64s: 1')

    self.assertNone('int64s: 0', 'int64s: 0 int64s: 2', """
                      int64s: 0
                    + int64s: 2
                    """)
    self.assertNone('int64s: 0 int64s: 1', 'int64s: 0 int64s: 2', """
                      int64s: 0
                    - int64s: 1
                    ?         ^
                    + int64s: 2
                    ?         ^
                    """)

  def testMessage(self):
    # [FUNCTION DOC]
    # Test: nested messages. Identical messages pass; reordered or duplicated nested strings and
    # integers fail; the failure text shows added or removed nested blocks and changed values.
    self.assertAll('medium: {}')
    self.assertAll('medium: { smalls: {} }')
    self.assertAll('medium: { int32s: 1 smalls: {} }')
    self.assertAll('medium: { smalls: { strings: "x" } }')
    self.assertAll(
        'medium: { smalls: { strings: "x" } } small: { strings: "y" }')

    self.assertSameNotEqual('medium: { smalls: { strings: "x" strings: "y" } }',
                            'medium: { smalls: { strings: "y" strings: "x" } }')
    self.assertSameNotEqual(
        'medium: { smalls: { strings: "x" } smalls: { strings: "y" } }',
        'medium: { smalls: { strings: "y" } smalls: { strings: "x" } }')

    self.assertSameNotEqual(
        'medium: { smalls: { strings: "x" strings: "y" strings: "x" } }',
        'medium: { smalls: { strings: "y" strings: "x" } }')
    self.assertSameNotEqual(
        'medium: { smalls: { strings: "x" } int32s: 0 }',
        'medium: { int32s: 0 smalls: { strings: "x" } int32s: 0 }')

    self.assertNone('medium: {}', 'medium: { smalls: { strings: "x" } }', """
                      medium {
                    +   smalls {
                    +     strings: "x"
                    +   }
                      }
                    """)
    self.assertNone('medium: { smalls: { strings: "x" } }',
                    'medium: { smalls: {} }', """
                      medium {
                        smalls {
                    -     strings: "x"
                        }
                      }
                    """)
    self.assertNone('medium: { int32s: 0 }', 'medium: { int32s: 1 }', """
                      medium {
                    -   int32s: 0
                    ?           ^
                    +   int32s: 1
                    ?           ^
                      }
                    """)

  def testMsgPassdown(self):
    # [FUNCTION DOC]
    # Test: the custom message argument is passed through, so a failing comparison raises an
    # assertion error whose text contains that message.
    self.assertRaisesRegex(
        AssertionError,
        'test message passed down',
        self.assertProtoEqual,
        'medium: {}',
        'medium: { smalls: { strings: "x" } }',
        msg='test message passed down')

  def testRepeatedMessage(self):
    # [FUNCTION DOC]
    # Test: repeated nested messages. Identical lists pass; reordered or differently sized lists
    # fail; the failure text shows added empty blocks or a changed string value.
    self.assertAll('medium: { smalls: {} smalls: {} }')
    self.assertAll('medium: { smalls: { strings: "x" } } medium: {}')
    self.assertAll('medium: { smalls: { strings: "x" } } medium: { int32s: 0 }')
    self.assertAll('medium: { smalls: {} smalls: { strings: "x" } } small: {}')

    self.assertSameNotEqual('medium: { smalls: { strings: "x" } smalls: {} }',
                            'medium: { smalls: {} smalls: { strings: "x" } }')

    self.assertSameNotEqual('medium: { smalls: {} }',
                            'medium: { smalls: {} smalls: {} }')
    self.assertSameNotEqual('medium: { smalls: {} smalls: {} } medium: {}',
                            'medium: {} medium: {} medium: { smalls: {} }')
    self.assertSameNotEqual(
        'medium: { smalls: { strings: "x" } smalls: {} }',
        'medium: { smalls: {} smalls: { strings: "x" } smalls: {} }')

    self.assertNone('medium: {}', 'medium: {} medium { smalls: {} }', """
                      medium {
                    +   smalls {
                    +   }
                      }
                    """)
    self.assertNone('medium: { smalls: {} smalls: { strings: "x" } }',
                    'medium: { smalls: {} smalls: { strings: "y" } }', """
                      medium {
                        smalls {
                        }
                        smalls {
                    -     strings: "x"
                    ?               ^
                    +     strings: "y"
                    ?               ^
                        }
                      }
                    """)


class MixinTests(compare.ProtoAssertions, googletest.TestCase):
  # [CLASS DOC]
  # Tests the mixin class: the test case inherits the assertion method from it and calls it as a
  # method instead of using the module function.

  def testAssertEqualWithStringArg(self):
    # [FUNCTION DOC]
    # Test: the mixin method accepts a text-format string as the first argument and passes when it
    # matches a Large message with the same string and float values.
    pb = compare_test_pb2.Large(string_='abc', float_=1.234)
    self.assertProtoEqual("""
          string_: 'abc'
          float_: 1.234
        """, pb)


if __name__ == '__main__':
  googletest.main()
