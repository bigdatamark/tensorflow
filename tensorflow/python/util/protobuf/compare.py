# [MODULE DOC]
# HELPER: test-support utilities for comparing protocol buffer messages.
# Exposes: a float closeness check treating equal infinities and any two NaNs as equal; a routine
# that replaces approximately equal floats of the actual message by the expected ones so a text
# comparison ignores tiny differences; a test assertion function (assertProtoEqual) that
# compares two messages (or a text-format string and a message) and fails the test with a readable
# multi-line diff; a normalizer for numeric field types and precision; a structural equality
# function for messages (ProtoEq) with ordered list semantics for repeated fields; and a mixin
# class that exposes the assertion as a test-case method. No import-time side effects beyond
# importing the protobuf libraries.
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

"""Utility functions for comparing proto2 messages in Python.

ProtoEq() compares two proto2 messages for equality.

ClearDefaultValuedFields() recursively clears the fields that are set to their
default values. This is useful for comparing protocol buffers where the
semantics of unset fields and default valued fields are the same.

assertProtoEqual() is useful for unit tests.  It produces much more helpful
output than assertEqual() for proto2 messages, e.g. this:

  outer {
    inner {
-     strings: "x"
?               ^
+     strings: "y"
?               ^
    }
  }

...compared to the default output from assertEqual() that looks like this:

AssertionError: <my.Msg object at 0x9fb353c> != <my.Msg object at 0x9fb35cc>

Call it inside your unit test's googletest.TestCase subclasses like this:

  from tensorflow.python.util.protobuf import compare

  class MyTest(googletest.TestCase):
    ...
    def testXXX(self):
      ...
      compare.assertProtoEqual(self, a, b)

Alternatively:

  from tensorflow.python.util.protobuf import compare

  class MyTest(compare.ProtoAssertions, googletest.TestCase):
    ...
    def testXXX(self):
      ...
      self.assertProtoEqual(a, b)
"""

import collections.abc as collections_abc
import difflib
import math

from google.protobuf import descriptor
from google.protobuf import descriptor_pool
from google.protobuf import message
from google.protobuf import text_format


# TODO(alankelly): Distinguish between signalling and quiet NaNs.
def isClose(x, y, relative_tolerance):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says relative_tolerance is optional and that strings are compared
  # when it is missing; the function requires it and never compares strings.
  # Inputs: x and y, floats; relative_tolerance, float multiplier. Returns true if both are NaN
  # (a NaN and a non-NaN give false), if either is infinite and they are equal, or otherwise when
  # the absolute difference is at most the tolerance times the larger absolute value. Pure.
  """Returns True if x is close to y given the relative tolerance or if x and y are both inf, both -inf, or both NaNs.

  This function does not distinguish between signalling and non-signalling NaN.

  Args:
    x: float value to be compared
    y: float value to be compared
    relative_tolerance: float. The allowable difference between the two values
      being compared is determined by multiplying the relative tolerance by the
      maximum of the two values. If this is not provided, then all floats are
      compared using string comparison.
  """
  # NaNs are considered equal.
  if math.isnan(x) or math.isnan(y):
    return math.isnan(x) == math.isnan(y)

  if math.isinf(x) or math.isinf(y):
    return x == y

  return abs(x - y) <= relative_tolerance * max(abs(x), abs(y))


def checkFloatEqAndReplace(self, expected, actual, relative_tolerance):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # HELPER: pre-processing step for proto comparison with a float tolerance.
  # Inputs: self, the test case (unused except for passing along); expected and actual, protobuf
  # messages; relative_tolerance, float. Walks the set fields of both messages pairwise (assumes the
  # same fields are set in the same order), and for float and double fields (single or repeated)
  # overwrites the actual value with the expected one whenever they are close. Recurses into
  # nested messages, groups and map values that are messages. Returns nothing. Side effect:
  # mutates the actual message in place; values that are not close are left untouched.
  """Recursively replaces the floats in actual with those in expected iff they are approximately equal.

  This is done because string equality will consider values such as 5.0999999999
  and 5.1 as not being equal, despite being extremely close.

  Args:
    self: googletest.TestCase
    expected: expected values
    actual: actual values
    relative_tolerance: float, relative tolerance.
  """

  for expected_fields, actual_fields in zip(
      expected.ListFields(), actual.ListFields()
  ):
    is_repeated = True
    expected_desc, expected_values = expected_fields
    actual_values = actual_fields[1]
    if expected_desc.label != descriptor.FieldDescriptor.LABEL_REPEATED:
      is_repeated = False
      expected_values = [expected_values]
      actual_values = [actual_values]

    if (
        expected_desc.type == descriptor.FieldDescriptor.TYPE_FLOAT
        or expected_desc.type == descriptor.FieldDescriptor.TYPE_DOUBLE
    ):
      for i, (x, y) in enumerate(zip(expected_values, actual_values)):
        # Replace the actual value with the expected value if the test passes,
        # otherwise leave it and let it fail in the next test so that the error
        # message is nicely formatted
        if isClose(x, y, relative_tolerance):
          if is_repeated:
            getattr(actual, actual_fields[0].name)[i] = x
          else:
            setattr(actual, actual_fields[0].name, x)

    if (
        expected_desc.type == descriptor.FieldDescriptor.TYPE_MESSAGE
        or expected_desc.type == descriptor.FieldDescriptor.TYPE_GROUP
    ):
      if (
          expected_desc.type == descriptor.FieldDescriptor.TYPE_MESSAGE
          and expected_desc.message_type.has_options
          and expected_desc.message_type.GetOptions().map_entry
      ):
        # This is a map, only recurse if it has type message type.
        if (
            expected_desc.message_type.fields_by_number[2].type
            == descriptor.FieldDescriptor.TYPE_MESSAGE
        ):
          for e_v, a_v in zip(
              iter(expected_values.values()), iter(actual_values.values())
          ):
            checkFloatEqAndReplace(
                self,
                expected=e_v,
                actual=a_v,
                relative_tolerance=relative_tolerance,
            )
      else:
        for v, a in zip(expected_values, actual_values):
          # recursive step
          checkFloatEqAndReplace(
              self, expected=v, actual=a, relative_tolerance=relative_tolerance
          )


def assertProtoEqual(
    self,
    a,
    b,
    check_initialized=True,
    normalize_numbers=False,
    msg=None,
    relative_tolerance=None,
):  # pylint: disable=invalid-name(
  # [FUNCTION DOC]
  # KNOWN ISSUE: the signature line ends with a stray opening parenthesis inside the lint comment.
  # Test assertion: fails the test with a readable diff if two protos differ.
  # Inputs: self, the test case; a, a message or a text-format string (parsed using the class of b
  # and the default descriptor pool); b, a message; check_initialized, whether to fail when either
  # message has missing required fields (default true); normalize_numbers, whether to normalize
  # numeric field types and precision first (default false, mutates both messages); msg, optional
  # failure message; relative_tolerance, optional float enabling tolerant float comparison (mutates
  # a). Compares the text-format renderings: a multi-line equality assertion when both are short,
  # otherwise a unified diff of the lines; a failure is raised when the diff is non-empty.
  # Returns nothing; raises the test failure exception on initialization errors or differences.
  # Parse errors from the text parser propagate.
  """Fails with a useful error if a and b aren't equal.

  Comparison of repeated fields matches the semantics of
  unittest.TestCase.assertEqual(), ie order and extra duplicates fields matter.

  Args:
    self: googletest.TestCase
    a: proto2 PB instance, or text string representing one.
    b: proto2 PB instance -- message.Message or subclass thereof.
    check_initialized: boolean, whether to fail if either a or b isn't
      initialized.
    normalize_numbers: boolean, whether to normalize types and precision of
      numbers before comparison.
    msg: if specified, is used as the error message on failure.
    relative_tolerance: float, relative tolerance. If this is not provided, then
      all floats are compared using string comparison otherwise, floating point
      comparisons are done using the relative tolerance provided.
  """
  pool = descriptor_pool.Default()
  if isinstance(a, str):
    a = text_format.Parse(a, b.__class__(), descriptor_pool=pool)

  for pb in a, b:
    if check_initialized:
      errors = pb.FindInitializationErrors()
      if errors:
        self.fail('Initialization errors: %s\n%s' % (errors, pb))
    if normalize_numbers:
      NormalizeNumberFields(pb)

  if relative_tolerance is not None:
    checkFloatEqAndReplace(
        self, expected=b, actual=a, relative_tolerance=relative_tolerance
    )

  a_str = text_format.MessageToString(a, descriptor_pool=pool)
  b_str = text_format.MessageToString(b, descriptor_pool=pool)

  # Some Python versions would perform regular diff instead of multi-line
  # diff if string is longer than 2**16. We substitute this behavior
  # with a call to unified_diff instead to have easier-to-read diffs.
  # For context, see: https://bugs.python.org/issue11763.
  if len(a_str) < 2**16 and len(b_str) < 2**16:
    self.assertMultiLineEqual(a_str, b_str, msg=msg)
  else:
    diff = ''.join(
        difflib.unified_diff(a_str.splitlines(True), b_str.splitlines(True)))
    if diff:
      self.fail('%s :\n%s' % (msg, diff))


def NormalizeNumberFields(pb):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says floats are normalized to five digits of precision; the code
  # rounds single-precision floats to six digits and doubles to seven, and 64-bit integers are
  # converted with the same plain integer conversion as 32-bit ones (no long type distinction).
  # Input: pb, a protobuf message. Converts the set integer, enumeration, float and double fields
  # (single or repeated) to normalized Python numbers, rewriting the field values, and recurses into
  # nested messages, groups and map fields whose values are messages. Returns the same message
  # object, modified in place.
  """Normalizes types and precisions of number fields in a protocol buffer.

  Due to subtleties in the python protocol buffer implementation, it is possible
  for values to have different types and precision depending on whether they
  were set and retrieved directly or deserialized from a protobuf. This function
  normalizes integer values to ints and longs based on width, 32-bit floats to
  five digits of precision to account for python always storing them as 64-bit,
  and ensures doubles are floating point for when they're set to integers.

  Modifies pb in place. Recurses into nested objects.

  Args:
    pb: proto2 message.

  Returns:
    the given pb, modified in place.
  """
  for desc, values in pb.ListFields():
    is_repeated = True
    if desc.label != descriptor.FieldDescriptor.LABEL_REPEATED:
      is_repeated = False
      values = [values]

    normalized_values = None

    # We force 32-bit values to int and 64-bit values to long to make
    # alternate implementations where the distinction is more significant
    # (e.g. the C++ implementation) simpler.
    if desc.type in (descriptor.FieldDescriptor.TYPE_INT64,
                     descriptor.FieldDescriptor.TYPE_UINT64,
                     descriptor.FieldDescriptor.TYPE_SINT64):
      normalized_values = [int(x) for x in values]
    elif desc.type in (descriptor.FieldDescriptor.TYPE_INT32,
                       descriptor.FieldDescriptor.TYPE_UINT32,
                       descriptor.FieldDescriptor.TYPE_SINT32,
                       descriptor.FieldDescriptor.TYPE_ENUM):
      normalized_values = [int(x) for x in values]
    elif desc.type == descriptor.FieldDescriptor.TYPE_FLOAT:
      normalized_values = [round(x, 6) for x in values]
    elif desc.type == descriptor.FieldDescriptor.TYPE_DOUBLE:
      normalized_values = [round(float(x), 7) for x in values]

    if normalized_values is not None:
      if is_repeated:
        pb.ClearField(desc.name)
        getattr(pb, desc.name).extend(normalized_values)
      else:
        setattr(pb, desc.name, normalized_values[0])

    if (desc.type == descriptor.FieldDescriptor.TYPE_MESSAGE or
        desc.type == descriptor.FieldDescriptor.TYPE_GROUP):
      if (desc.type == descriptor.FieldDescriptor.TYPE_MESSAGE and
          desc.message_type.has_options and
          desc.message_type.GetOptions().map_entry):
        # This is a map, only recurse if the values have a message type.
        if (desc.message_type.fields_by_number[2].type ==
            descriptor.FieldDescriptor.TYPE_MESSAGE):
          for v in iter(values.values()):
            NormalizeNumberFields(v)
      else:
        for v in values:
          # recursive step
          NormalizeNumberFields(v)

  return pb


def _IsMap(value):
  # [FUNCTION DOC]
  # HELPER: structural check used by ProtoEq. Input: value, any object. Returns true if it is a
  # mapping (including protobuf map containers), else false.
  return isinstance(value, collections_abc.Mapping)


def _IsRepeatedContainer(value):
  # [FUNCTION DOC]
  # HELPER: structural check used by ProtoEq. Input: value, any object. Returns false for text
  # strings and for anything that cannot be iterated; true for any other iterable (including
  # repeated protobuf fields, lists, and byte strings).
  if isinstance(value, str):
    return False
  try:
    iter(value)
    return True
  except TypeError:
    return False


def ProtoEq(a, b):
  # [FUNCTION DOC]
  # Recursive equality of protobuf messages. Inputs: a and b, messages, maps, repeated containers
  # or primitive values. Messages are viewed as mappings from field number to value, repeated
  # fields as mappings from element index to value; equal when both have exactly the same keys and
  # all values are equal recursively; non-container values are compared with plain equality.
  # Returns true or false. Pure, no side effects.
  """Compares two proto2 objects for equality.

  Recurses into nested messages. Uses list (not set) semantics for comparing
  repeated fields, ie duplicates and order matter.

  Args:
    a: A proto2 message or a primitive.
    b: A proto2 message or a primitive.

  Returns:
    `True` if the messages are equal.
  """
  def Format(pb):
    # [FUNCTION DOC]
    # HELPER: nested helper of ProtoEq that converts a value to a comparable form. Input: pb, a
    # message, map, repeated container or primitive. Returns a mapping from field number to value
    # for messages (set fields only), a plain mapping copy for maps, a mapping from index to value
    # for repeated containers, otherwise the input unchanged.
    """Returns a dictionary or unchanged pb bases on its type.

    Specifically, this function returns a dictionary that maps tag
    number (for messages) or element index (for repeated fields) to
    value, or just pb unchanged if it's neither.

    Args:
      pb: A proto2 message or a primitive.
    Returns:
      A dict or unchanged pb.
    """
    if isinstance(pb, message.Message):
      return dict((desc.number, value) for desc, value in pb.ListFields())
    elif _IsMap(pb):
      return dict(pb.items())
    elif _IsRepeatedContainer(pb):
      return dict(enumerate(list(pb)))
    else:
      return pb

  a, b = Format(a), Format(b)

  # Base case
  if not isinstance(a, dict) or not isinstance(b, dict):
    return a == b

  # This list performs double duty: it compares two messages by tag value *or*
  # two repeated fields by element, in order. the magic is in the format()
  # function, which converts them both to the same easily comparable format.
  for tag in sorted(set(a.keys()) | set(b.keys())):
    if tag not in a or tag not in b:
      return False
    else:
      # Recursive step
      if not ProtoEq(a[tag], b[tag]):
        return False

  # Didn't find any values that differed, so they're equal!
  return True


class ProtoAssertions(object):
  # [CLASS DOC]
  # Mixin for unit test classes (placed before the test-case base class) that adds a proto equality
  # assertion method delegating to the module-level assertion function. No state of its own.
  """Mix this into a googletest.TestCase class to get proto2 assertions.

  Usage:

  class SomeTestCase(compare.ProtoAssertions, googletest.TestCase):
    ...
    def testSomething(self):
      ...
      self.assertProtoEqual(a, b)

  See module-level definitions for method documentation.
  """

  # pylint: disable=invalid-name
  def assertProtoEqual(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Method form of the module-level assertion. Inputs: any positional and keyword arguments,
    # passed through unchanged with the test case as the first argument (so the arguments are the
    # expected/actual messages and options described for the module-level function). Returns
    # nothing; fails the test on difference, exactly like the function it delegates to.
    return assertProtoEqual(self, *args, **kwargs)
