# [MODULE DOC]
# Unit tests for tensorflow/python/framework/tensor_shape.py: the legacy Dimension class and the
# TensorShape class (construction, merging, slicing, concatenation, compatibility, proto
# conversion, string forms, equality, pickling, serialization and trace-type subtyping).
# Contains four test case classes and runs the TensorFlow test runner when executed as a script.
# No import-time side effects beyond imports.
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
"""Functional tests for shape inference helper classes."""
from absl.testing import parameterized

from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.function import trace_type
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.platform import googletest


class DimensionTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case covering the legacy Dimension class: construction, arithmetic, comparison, equality,
  # merging, string forms, pickling support and error handling. Uses the standard TensorFlow test
  # base class; no shared fixtures, each test builds its own Dimension objects.

  def testDimension(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: several assertions are repeated verbatim (the doubled-by-two check and the
    # greater-than check each appear twice), so they add no coverage.
    # Scenario: a known dimension of size 12. Asserts value and integer conversion, equality, and that
    # addition, subtraction, multiplication and floor division work against Dimensions and plain
    # integers on both sides (including list repetition by a Dimension), merge_with on equal values and
    # integers, all ordering comparisons, inequality to a tuple, and that merging with a different
    # known size (13) raises a value error.
    dim = tensor_shape.Dimension(12)
    self.assertEqual(12, dim.value)
    self.assertEqual(12, int(dim))
    self.assertEqual(dim, tensor_shape.Dimension(12))
    self.assertEqual(
        tensor_shape.Dimension(15), dim + tensor_shape.Dimension(3))
    self.assertEqual(tensor_shape.Dimension(15), dim + 3)
    self.assertEqual(tensor_shape.Dimension(15), 3 + dim)
    self.assertEqual(tensor_shape.Dimension(9), dim - 3)
    self.assertEqual(tensor_shape.Dimension(1), 13 - dim)
    self.assertEqual(
        tensor_shape.Dimension(24), dim * tensor_shape.Dimension(2))
    self.assertEqual(tensor_shape.Dimension(24), dim * 2)
    self.assertEqual(tensor_shape.Dimension(24), 2 * dim)
    self.assertEqual([4] * 12, [4] * dim)
    self.assertEqual(12 * [4], dim * [4])
    self.assertEqual(tensor_shape.Dimension(24), 2 * dim)
    self.assertEqual(
        tensor_shape.Dimension(6), dim // tensor_shape.Dimension(2))
    self.assertEqual(tensor_shape.Dimension(6), dim // 2)
    self.assertEqual(tensor_shape.Dimension(0), 2 // dim)
    self.assertEqual(
        tensor_shape.Dimension(12), dim.merge_with(tensor_shape.Dimension(12)))
    self.assertEqual(tensor_shape.Dimension(12), dim.merge_with(12))
    self.assertLess(tensor_shape.Dimension(12), tensor_shape.Dimension(13))
    self.assertGreater(tensor_shape.Dimension(13), tensor_shape.Dimension(12))
    self.assertLessEqual(tensor_shape.Dimension(12), tensor_shape.Dimension(12))
    self.assertLessEqual(tensor_shape.Dimension(12), tensor_shape.Dimension(13))
    self.assertGreater(tensor_shape.Dimension(13), tensor_shape.Dimension(12))
    self.assertGreaterEqual(
        tensor_shape.Dimension(12), tensor_shape.Dimension(12))
    self.assertGreaterEqual(
        tensor_shape.Dimension(13), tensor_shape.Dimension(12))
    self.assertNotEqual(dim, (12,))
    with self.assertRaises(ValueError):
      dim.merge_with(tensor_shape.Dimension(13))

  def testUnknownDimension(self):
    # [FUNCTION DOC]
    # Scenario: unknown dimensions. Asserts the value of an unknown Dimension is null, and that adding,
    # multiplying, floor-dividing and merging two unknown dimensions all give an unknown value, while
    # every ordering comparison between two unknown dimensions returns null.
    dim = tensor_shape.Dimension(None)
    self.assertIsNone(dim.value)
    self.assertEqual(dim.value, tensor_shape.Dimension(None).value)
    self.assertEqual(
        tensor_shape.Dimension(None).value,
        (dim + tensor_shape.Dimension(None)).value)
    self.assertEqual(
        tensor_shape.Dimension(None).value,
        (dim * tensor_shape.Dimension(None)).value)
    self.assertEqual(
        tensor_shape.Dimension(None).value,
        (dim // tensor_shape.Dimension(None)).value)
    self.assertEqual(
        tensor_shape.Dimension(None).value,
        dim.merge_with(tensor_shape.Dimension(None)).value)
    self.assertIsNone(
        tensor_shape.Dimension(None) < tensor_shape.Dimension(None))
    self.assertIsNone(
        tensor_shape.Dimension(None) <= tensor_shape.Dimension(None))
    self.assertIsNone(
        tensor_shape.Dimension(None) > tensor_shape.Dimension(None))
    self.assertIsNone(
        tensor_shape.Dimension(None) >= tensor_shape.Dimension(None))

  def testKnownAndUnknownDimensions(self):
    # [FUNCTION DOC]
    # Scenario: mixing known (12) and unknown dimensions. Asserts sum, product and floor quotient in
    # either order are unknown, merging yields the known value 12 in either order, and all ordering
    # comparisons between a known and an unknown dimension return null in both orders.
    known = tensor_shape.Dimension(12)
    unknown = tensor_shape.Dimension(None)
    self.assertEqual(
        tensor_shape.Dimension(None).value, (known + unknown).value)
    self.assertEqual(
        tensor_shape.Dimension(None).value, (unknown + known).value)
    self.assertEqual(
        tensor_shape.Dimension(None).value, (known * unknown).value)
    self.assertEqual(
        tensor_shape.Dimension(None).value, (unknown * known).value)
    self.assertEqual(
        tensor_shape.Dimension(None).value, (known // unknown).value)
    self.assertEqual(
        tensor_shape.Dimension(None).value, (unknown // known).value)
    self.assertEqual(tensor_shape.Dimension(12), known.merge_with(unknown))
    self.assertEqual(tensor_shape.Dimension(12), unknown.merge_with(known))
    self.assertIsNone(tensor_shape.Dimension(12) < tensor_shape.Dimension(None))
    self.assertIsNone(
        tensor_shape.Dimension(12) <= tensor_shape.Dimension(None))
    self.assertIsNone(tensor_shape.Dimension(12) > tensor_shape.Dimension(None))
    self.assertIsNone(
        tensor_shape.Dimension(12) >= tensor_shape.Dimension(None))
    self.assertIsNone(tensor_shape.Dimension(None) < tensor_shape.Dimension(12))
    self.assertIsNone(
        tensor_shape.Dimension(None) <= tensor_shape.Dimension(12))
    self.assertIsNone(tensor_shape.Dimension(None) > tensor_shape.Dimension(12))
    self.assertIsNone(
        tensor_shape.Dimension(None) >= tensor_shape.Dimension(12))

  def testAsDimension(self):
    # [FUNCTION DOC]
    # Tests the as_dimension helper: a Dimension and an integer both map to an equal known Dimension,
    # and an unknown Dimension or null both map to a Dimension with a null value.
    self.assertEqual(
        tensor_shape.Dimension(12),
        tensor_shape.as_dimension(tensor_shape.Dimension(12)))
    self.assertEqual(tensor_shape.Dimension(12), tensor_shape.as_dimension(12))
    self.assertEqual(
        tensor_shape.Dimension(None).value,
        tensor_shape.as_dimension(tensor_shape.Dimension(None)).value)
    self.assertEqual(
        tensor_shape.Dimension(None).value,
        tensor_shape.as_dimension(None).value)

  def testEquality(self):
    # [FUNCTION DOC]
    # Tests equality: equal and unequal known dimensions compare as expected; comparing with an
    # unknown dimension (either side, or both unknown) returns null; comparing an unknown dimension
    # with a non-integral float gives a definite non-null result that is not equal; comparing with the
    # null value returns null; a known dimension is not equal to a float.
    self.assertEqual(tensor_shape.Dimension(12), tensor_shape.Dimension(12))
    self.assertNotEqual(tensor_shape.Dimension(12), tensor_shape.Dimension(13))
    self.assertIsNone(
        tensor_shape.Dimension(12) == tensor_shape.Dimension(None))
    self.assertIsNone(
        tensor_shape.Dimension(None) == tensor_shape.Dimension(12))
    self.assertIsNone(
        tensor_shape.Dimension(None) == tensor_shape.Dimension(None))

    # None indicates ambiguous comparison, but comparison vs the wrong type
    # is unambiguously False.
    self.assertIsNotNone(tensor_shape.Dimension(None) == 12.99)
    self.assertNotEqual(tensor_shape.Dimension(None), 12.99)

    # pylint: disable=singleton-comparison, g-equals-none
    self.assertIsNone(tensor_shape.Dimension(None) == None)
    # pylint: enable=singleton-comparison, g-equals-none
    self.assertNotEqual(tensor_shape.Dimension(12), 12.99)

  def testInequality(self):
    # [FUNCTION DOC]
    # Tests inequality, mirror image of the equality test: known values behave normally, comparisons
    # involving an unknown dimension return null, and comparison with a non-integral float is a
    # definite result (not null) and not equal; comparison with the null value returns null.
    self.assertNotEqual(tensor_shape.Dimension(12), tensor_shape.Dimension(13))
    self.assertEqual(tensor_shape.Dimension(12), tensor_shape.Dimension(12))
    self.assertIsNone(
        tensor_shape.Dimension(12) != tensor_shape.Dimension(None))
    self.assertIsNone(
        tensor_shape.Dimension(None) != tensor_shape.Dimension(12))
    self.assertIsNone(
        tensor_shape.Dimension(None) != tensor_shape.Dimension(None))

    # None indicates ambiguous comparison, but comparison vs the wrong type
    # is unambiguously False.
    self.assertIsNotNone(tensor_shape.Dimension(None) != 12.99)
    self.assertNotEqual(tensor_shape.Dimension(None), 12.99)

    self.assertIsNone(tensor_shape.Dimension(None) != None)  # pylint: disable=g-equals-none
    self.assertNotEqual(tensor_shape.Dimension(12), 12.99)

  def testIsCompatibleWithError(self):
    # [FUNCTION DOC]
    # Tests error reporting of the compatibility check: an empty list argument raises a type error
    # saying the value must be integer or null, and a negative integer raises a value error saying it
    # must be non-negative.
    with self.assertRaisesRegex(TypeError, "must be integer or None"):
      tensor_shape.Dimension(42).is_compatible_with([])

    with self.assertRaisesRegex(ValueError, "must be >= 0"):
      tensor_shape.Dimension(42).is_compatible_with(-1)

  def testMergeWithError(self):
    # [FUNCTION DOC]
    # Tests error reporting of merging: an empty list raises a type error (must be integer or null) and
    # a negative integer raises a value error (must be non-negative).
    with self.assertRaisesRegex(TypeError, "must be integer or None"):
      tensor_shape.Dimension(42).merge_with([])

    with self.assertRaisesRegex(ValueError, "must be >= 0"):
      tensor_shape.Dimension(42).merge_with(-1)

  def testRepr(self):
    # [FUNCTION DOC]
    # Tests debug representation: a known dimension shows its integer inside the class-name text and an
    # unknown one shows the null word.
    self.assertEqual(repr(tensor_shape.Dimension(7)), "Dimension(7)")
    self.assertEqual(repr(tensor_shape.Dimension(None)), "Dimension(None)")

  def testStr(self):
    # [FUNCTION DOC]
    # Tests text conversion: a known dimension gives its decimal digits and an unknown one gives a
    # question mark.
    self.assertEqual(str(tensor_shape.Dimension(7)), "7")
    self.assertEqual(str(tensor_shape.Dimension(None)), "?")

  def testUnsupportedType(self):
    # [FUNCTION DOC]
    # Tests that constructing a Dimension from a data type object (not integer-like) raises a type
    # error.
    with self.assertRaises(TypeError):
      tensor_shape.Dimension(dtypes.string)

  def testBool(self):
    # [FUNCTION DOC]
    # Tests truthiness: dimension of size one is true; size zero and unknown dimensions are false.
    one = tensor_shape.Dimension(1)
    zero = tensor_shape.Dimension(0)
    has_none = tensor_shape.Dimension(None)
    self.assertTrue(one)
    self.assertFalse(zero)
    self.assertFalse(has_none)

  def testMod(self):
    # [FUNCTION DOC]
    # Tests the modulo operator with dimensions 9 and 4: both Dimension-by-Dimension and
    # Dimension-by-integer give 1, and integer-by-Dimension (the reflected form) gives 4.
    four = tensor_shape.Dimension(4)
    nine = tensor_shape.Dimension(9)
    self.assertEqual(nine % four, 1)
    # test both __mod__ and __rmod__.
    self.assertEqual(nine % 4, 1)
    self.assertEqual(4 % nine, 4)

  def testReduce(self):
    # [FUNCTION DOC]
    # Tests pickling support: the reduce result is the Dimension class plus an argument tuple with the
    # value 5, and calling the class with those arguments rebuilds an equal Dimension.
    dim = tensor_shape.Dimension(5)
    ctor, args = dim.__reduce__()
    self.assertEqual(ctor, tensor_shape.Dimension)
    self.assertEqual(args, (5,))
    reconstructed = ctor(*args)
    self.assertEqual(reconstructed, dim)

  def testDiv(self):
    # [FUNCTION DOC]
    # Related to a GitHub issue about the division operator. Asserts that true division between two
    # Dimensions, a Dimension and an integer, and an integer and a Dimension each raise a type error
    # whose message names both operand types and suggests floor division.
    # Note: This test is related to GitHub issue 25790.
    six = tensor_shape.Dimension(6)
    two = tensor_shape.Dimension(2)
    message = (r"unsupported operand type\(s\) for /: "
               r"'Dimension' and 'Dimension', please use // instead")
    with self.assertRaisesRegex(TypeError, message):
      _ = six / two
    message = (r"unsupported operand type\(s\) for /: "
               r"'Dimension' and 'int', please use // instead")
    with self.assertRaisesRegex(TypeError, message):
      _ = six / 2
    message = (r"unsupported operand type\(s\) for /: "
               r"'int' and 'Dimension', please use // instead")
    with self.assertRaisesRegex(TypeError, message):
      _ = 6 / two


class SerilizationTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # KNOWN ISSUE: the class name is misspelled (Serilization instead of Serialization).
  # Test case covering serialization of TensorShape through the trace-type serialize/deserialize
  # functions. Uses the standard TensorFlow test base class and has no shared fixtures.

  def testSerialization(self):
    # [FUNCTION DOC]
    # Scenario: three shapes (fully known, partially known and unknown rank). Each is serialized and
    # deserialized with the trace-type helpers, and the result is asserted to equal the original.
    shape_1 = tensor_shape.TensorShape([1, 2, 3])
    shape_2 = tensor_shape.TensorShape([None, 2, None])
    shape_3 = tensor_shape.TensorShape(None)

    self.assertEqual(
        trace_type.deserialize(trace_type.serialize(shape_1)), shape_1)
    self.assertEqual(
        trace_type.deserialize(trace_type.serialize(shape_2)), shape_2)
    self.assertEqual(
        trace_type.deserialize(trace_type.serialize(shape_3)), shape_3)


class ShapeTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case (also parameterized) covering TensorShape behavior: known, partial and unknown shapes,
  # merging, slicing, concatenation and addition, most-specific-compatible shape, proto conversion,
  # string form, equality, list conversion and pickling. Parameterization is used for the
  # concatenation and addition tests.

  def testUnknownShape(self):
    # [FUNCTION DOC]
    # Scenario: a shape of unknown rank. Asserts assert_is_fully_defined raises a value error, rank is
    # null, length raises a value error about unknown rank, the shape is falsy, its dimension list is
    # null, and iterating raises a value error about unknown rank.
    s = tensor_shape.TensorShape(None)
    # pylint: disable=g-error-prone-assert-raises
    with self.assertRaisesRegex(ValueError, "Shape .+ is not fully defined"):
      s.assert_is_fully_defined()
    # pylint: enable=g-error-prone-assert-raises
    self.assertIsNone(s.rank)
    with self.assertRaisesRegex(
        ValueError, "Cannot take the length of shape with unknown rank."):
      len(s)
    self.assertFalse(s)
    self.assertIsNone(s.dims)
    with self.assertRaisesRegex(
        ValueError, "Cannot iterate over a shape with unknown rank."):
      for _ in tensor_shape.TensorShape(None):
        pass

  def testFullyDefinedShape(self):
    # [FUNCTION DOC]
    # Scenario: a fully defined shape built from three Dimension objects (3, 4, 7). Asserts it passes
    # the fully-defined assertion, has rank and length 3, is truthy, passes the rank assertion, its
    # dimension list and indexed items are the Dimension objects, as_list gives the integers,
    # compatibility with a list of the same sizes holds, same-rank assertion with another shape passes,
    # and iterating yields values equal to the sizes.
    s = tensor_shape.TensorShape([
        tensor_shape.Dimension(3),
        tensor_shape.Dimension(4),
        tensor_shape.Dimension(7)
    ])
    s.assert_is_fully_defined()
    self.assertEqual(s.rank, 3)
    self.assertLen(s, 3)
    self.assertTrue(s)
    s.assert_has_rank(3)
    self.assertEqual([
        tensor_shape.Dimension(3),
        tensor_shape.Dimension(4),
        tensor_shape.Dimension(7)
    ], s.dims)
    self.assertEqual(tensor_shape.Dimension(3), s[0])
    self.assertEqual(tensor_shape.Dimension(4), s[1])
    self.assertEqual(tensor_shape.Dimension(7), s[2])
    self.assertEqual([3, 4, 7], s.as_list())
    s.assert_is_compatible_with([3, 4, 7])
    s.assert_same_rank([6, 3, 7])
    for d1, d2 in zip(s, [3, 4, 7]):
      assert tensor_shape.dimension_value(d1) == d2

  def testPartiallyDefinedShape(self):
    # [FUNCTION DOC]
    # Scenario: a partially defined shape (3, unknown, 7). Asserts the fully-defined assertion raises a
    # value error, rank and length are 3, the shape is truthy, rank assertion passes, indexing and the
    # dimension list give the expected Dimensions (unknown in the middle), same-rank assertion passes,
    # and iteration yields the expected values including null for the unknown one.
    s = tensor_shape.TensorShape([
        tensor_shape.Dimension(3),
        tensor_shape.Dimension(None),
        tensor_shape.Dimension(7)
    ])
    # pylint: disable=g-error-prone-assert-raises
    with self.assertRaisesRegex(ValueError, "Shape .+ is not fully defined"):
      s.assert_is_fully_defined()
    # pylint: enable=g-error-prone-assert-raises
    self.assertEqual(s.rank, 3)
    self.assertLen(s, 3)
    self.assertTrue(s)
    s.assert_has_rank(3)
    self.assertEqual(tensor_shape.Dimension(3), s[0])
    self.assertEqual(tensor_shape.Dimension(None).value, s.dims[1].value)
    self.assertEqual(tensor_shape.Dimension(7), s.dims[2])
    s.assert_same_rank([6, 3, 7])
    for d1, d2 in zip(s, [3, None, 7]):
      assert tensor_shape.dimension_value(d1) == d2

  def testMergeFullShapes(self):
    # [FUNCTION DOC]
    # Tests merging fully known shapes: merging identical shapes gives the same list; merging shapes
    # with conflicting sizes raises a value error.
    self.assertEqual([3, 4, 7],
                     tensor_shape.TensorShape([3, 4, 7]).merge_with(
                         tensor_shape.TensorShape([3, 4, 7])).as_list())
    with self.assertRaises(ValueError):
      tensor_shape.TensorShape([3, 4, 7]).merge_with(
          tensor_shape.TensorShape([6, 3, 7]))

  def testMergePartialShapes(self):
    # [FUNCTION DOC]
    # Tests merging two partially known shapes whose unknown positions are complementary; the merged
    # shape is the fully known list of three sizes.
    s1 = tensor_shape.TensorShape([
        tensor_shape.Dimension(3),
        tensor_shape.Dimension(None),
        tensor_shape.Dimension(7)
    ])
    s2 = tensor_shape.TensorShape([
        tensor_shape.Dimension(None),
        tensor_shape.Dimension(4),
        tensor_shape.Dimension(7)
    ])
    self.assertEqual([3, 4, 7], s1.merge_with(s2).as_list())

  def testMergeFullAndUnknownShape(self):
    # [FUNCTION DOC]
    # Tests merging a fully known shape with an unknown-rank shape; the result is the known shape's
    # list of sizes.
    self.assertEqual([3, 4, 7],
                     tensor_shape.TensorShape([3, 4, 7]).merge_with(
                         tensor_shape.TensorShape(None)).as_list())

  def testSlice(self):
    # [FUNCTION DOC]
    # Tests slicing and indexing: on a known five-element shape, an integer index gives the expected
    # Dimension and a slice is compatible with the expected three-element shape; on an unknown shape,
    # an integer index gives an unknown value and a slice of length three is compatible with a shape of
    # three unknown dimensions.
    known = tensor_shape.TensorShape([0, 1, 2, 3, 4])
    self.assertEqual(tensor_shape.Dimension(2), known[2])
    tensor_shape.TensorShape([1, 2, 3]).assert_is_compatible_with(known[1:4])

    unknown = tensor_shape.TensorShape(None)
    self.assertEqual(
        tensor_shape.Dimension(None).value,
        tensor_shape.dimension_value(unknown[2]))
    tensor_shape.TensorShape([None, None,
                              None]).assert_is_compatible_with(unknown[1:4])

  @parameterized.named_parameters(
      ("Concatenate", lambda x, y: x.concatenate(y)),
      ("Add", lambda x, y: x + y), ("RAdd", lambda x, y: y.__radd__(x)))
  def testConcatenate(self, concatenate_fn):
    # [FUNCTION DOC]
    # Parameterized test over three ways of concatenating (the concatenate method, the addition
    # operator and the reflected-addition call). Input: concatenate_fn, a callback combining two
    # shapes. Asserts known plus known, known plus unknown, unknown plus known and unknown plus unknown
    # are all compatible with the four-dimension expected shape (the unknown cases yield a fully
    # unknown shape, which is compatible).
    tensor_shape.TensorShape([1, 2, 3, 4]).assert_is_compatible_with(
        concatenate_fn(
            tensor_shape.TensorShape([1, 2]), tensor_shape.TensorShape([3, 4])))
    tensor_shape.TensorShape([1, 2, 3, 4]).assert_is_compatible_with(
        concatenate_fn(
            tensor_shape.TensorShape([1, 2]), tensor_shape.TensorShape(None)))
    tensor_shape.TensorShape([1, 2, 3, 4]).assert_is_compatible_with(
        concatenate_fn(
            tensor_shape.TensorShape(None), tensor_shape.TensorShape([3, 4])))
    tensor_shape.TensorShape([1, 2, 3, 4]).assert_is_compatible_with(
        concatenate_fn(
            tensor_shape.TensorShape(None), tensor_shape.TensorShape(None)))

  @parameterized.named_parameters(
      ("Concatenate", lambda x, y: x.concatenate(y)),
      ("Add", lambda x, y: x + y))
  def testConcatenateWithDimension(self, concatenate_fn):
    # [FUNCTION DOC]
    # Parameterized test over the concatenate method and addition operator. Input: concatenate_fn, a
    # callback combining a shape with a Dimension. Asserts that a two-element shape combined with a
    # Dimension of size 3 is compatible with a three-element shape.
    tensor_shape.TensorShape([1, 2, 3]).assert_is_compatible_with(
        concatenate_fn(
            tensor_shape.TensorShape([1, 2]), tensor_shape.Dimension(3)))

  @parameterized.named_parameters(("List", [3, 4, 5]), ("Tuple", (3, 4, 5)))
  def testAdd_nonTensorShape(self, addend):
    # [FUNCTION DOC]
    # Parameterized test over a list and a tuple. Input: addend, a sequence of sizes. Asserts that
    # adding a one-element TensorShape (size 2) and the sequence returns a TensorShape compatible with
    # the shape of sizes two, three, four, five.
    two = tensor_shape.TensorShape([2])
    result = two + addend
    self.assertIsInstance(result, tensor_shape.TensorShape)
    tensor_shape.TensorShape([2, 3, 4, 5]).assert_is_compatible_with(result)

  @parameterized.named_parameters(("List", [2, 3, 4]), ("Tuple", (2, 3, 4)))
  def testRAdd_nonTensorShape(self, addend):
    # [FUNCTION DOC]
    # Parameterized test over a list and a tuple. Input: addend, a sequence of sizes (2, 3, 4). Asserts
    # that adding the sequence on the left of a one-element TensorShape (size 5) uses reflected
    # addition, returns a TensorShape and is compatible with the shape of sizes two, three, four, five.
    five = tensor_shape.TensorShape([5])
    result = addend + five
    self.assertIsInstance(result, tensor_shape.TensorShape)
    tensor_shape.TensorShape([2, 3, 4, 5]).assert_is_compatible_with(result)

  def _testMostSpecificCompatibleShapeHelper(self, x, y, expected):
    # [FUNCTION DOC]
    # HELPER: shared check for the most-specific-compatible-shape test.
    # Inputs: x and y, values convertible to shapes (lists or null); expected, the expected result
    # sizes as a list or null for unknown rank. Computes the most specific compatible shape of the two
    # and asserts its dimension list is null when expected is null, otherwise that its list equals
    # expected.
    mcs = tensor_shape.TensorShape(x).most_specific_compatible_shape(
        tensor_shape.TensorShape(y))
    mcs_dims = mcs.dims
    if expected is None or mcs_dims is None:
      self.assertIs(expected, mcs_dims)
    else:
      self.assertEqual(expected, mcs.as_list())

  def testMostSpecificCompatibleShape(self):
    # [FUNCTION DOC]
    # Tests most_specific_compatible_shape with cases: unknown on either side, different ranks (both
    # orders), identical shapes, and partially known shapes where differing or unknown positions become
    # unknown, using the shared helper for each case.
    self._testMostSpecificCompatibleShapeHelper([1, 2], None, None)
    self._testMostSpecificCompatibleShapeHelper(None, [1, 2], None)
    self._testMostSpecificCompatibleShapeHelper([1, 2], [1, 2, 3, 4], None)
    self._testMostSpecificCompatibleShapeHelper([1, 2, 3, 4], [1, 2], None)
    self._testMostSpecificCompatibleShapeHelper([1, 2], [1, 2], [1, 2])
    self._testMostSpecificCompatibleShapeHelper([None, 2, 3], [1, 1, 3],
                                                [None, None, 3])
    self._testMostSpecificCompatibleShapeHelper([1, 1, 3], [None, 2, 3],
                                                [None, None, 3])

  def testTruedivFails(self):
    # [FUNCTION DOC]
    # Tests that true division of dimensions is rejected: floor division of two unknown dimensions has
    # a null value, while true division raises a type error mentioning unsupported operand types.
    unknown = tensor_shape.Dimension(None)
    self.assertEqual((unknown // unknown).value, None)
    with self.assertRaisesRegex(TypeError, r"unsupported operand type"):
      unknown / unknown  # pylint: disable=pointless-statement

  def testConvertFromProto(self):
    # [FUNCTION DOC]
    # Tests conversion of TensorShapeProto messages. Builds protos for an empty shape, a fully known
    # shape and a partial one (-1 means unknown). Asserts both the constructor and the as_shape helper
    # give equal shapes, and for the partial shape that unknown and known dimension values come out as
    # null, 37 and 42 and that it is compatible with the equivalent partial shape.

    def make_tensor_shape_proto(shape):
      # [FUNCTION DOC]
      # HELPER: builds a TensorShapeProto for the enclosing test.
      # Input: shape, an iterable of integer sizes (-1 for unknown). Returns a proto with one dimension
      # entry per size.
      return tensor_shape_pb2.TensorShapeProto(
          dim=[tensor_shape_pb2.TensorShapeProto.Dim(size=x) for x in shape])

    proto = make_tensor_shape_proto([])
    self.assertEqual(
        tensor_shape.TensorShape([]), tensor_shape.TensorShape(proto))
    self.assertEqual(tensor_shape.TensorShape([]), tensor_shape.as_shape(proto))

    proto = make_tensor_shape_proto([1, 37, 42])
    self.assertEqual(
        tensor_shape.TensorShape([1, 37, 42]), tensor_shape.TensorShape(proto))
    self.assertEqual(
        tensor_shape.TensorShape([1, 37, 42]), tensor_shape.as_shape(proto))

    partial_proto_shape = tensor_shape.as_shape(
        make_tensor_shape_proto([-1, 37, 42]))
    partial_shape = tensor_shape.TensorShape([None, 37, 42])
    self.assertEqual(partial_proto_shape, partial_shape)
    self.assertEqual(tensor_shape.dimension_value(partial_proto_shape[0]), None)
    self.assertEqual(tensor_shape.dimension_value(partial_proto_shape[1]), 37)
    self.assertEqual(tensor_shape.dimension_value(partial_proto_shape[2]), 42)
    self.assertTrue(partial_shape.is_compatible_with(partial_proto_shape))

  def testStr(self):
    # [FUNCTION DOC]
    # Tests string form of shapes: unknown rank gives the unknown marker; unknown shapes of rank one,
    # two, three show question marks (replaced by the word None for comparison); a partial shape; a
    # scalar shape gives empty parentheses; one-element shapes show a trailing comma; and multi-element
    # shapes show comma-separated sizes.
    self.assertEqual("<unknown>", str(tensor_shape.unknown_shape()))
    self.assertEqual(
        "(None,)",
        str(tensor_shape.unknown_shape(rank=1)).replace("?", "None"))
    self.assertEqual(
        "(None, None)",
        str(tensor_shape.unknown_shape(rank=2)).replace("?", "None"))
    self.assertEqual(
        "(None, None, None)",
        str(tensor_shape.unknown_shape(rank=3)).replace("?", "None"))
    self.assertEqual(
        "(32, None, 1, 9)",
        str(tensor_shape.TensorShape([32, None, 1, 9])).replace("?", "None"))
    self.assertEqual("()", str(tensor_shape.TensorShape([])))
    self.assertEqual("(7,)", str(tensor_shape.TensorShape([7])))
    self.assertEqual("(3, 8)", str(tensor_shape.TensorShape([3, 8])))
    self.assertEqual("(4, 5, 2)", str(tensor_shape.TensorShape([4, 5, 2])))

  def testAsProto(self):
    # [FUNCTION DOC]
    # Tests conversion to a proto: the unknown-rank flag is set only for the unknown shape, and not set
    # for unknown-dimension shapes of known rank or for known and partial shapes.
    self.assertTrue(tensor_shape.unknown_shape().as_proto().unknown_rank)
    self.assertFalse(tensor_shape.unknown_shape(rank=3).as_proto().unknown_rank)
    self.assertFalse(
        tensor_shape.TensorShape([1, 2, 3]).as_proto().unknown_rank)
    self.assertFalse(
        tensor_shape.TensorShape([1, None, 3]).as_proto().unknown_rank)

  def testEquality(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the first equality assertion is duplicated and the unknown-versus-known inequality
    # assertions are repeated, so some checks add no coverage.
    # Tests shape equality: two shapes with equal sizes compare equal; a shape with an unknown
    # dimension differs from the fully known one; an unknown-rank shape differs from a known shape in
    # both orders and two unknown-rank shapes are equal in both orders.
    s1 = tensor_shape.TensorShape([
        tensor_shape.Dimension(3),
        tensor_shape.Dimension(4),
        tensor_shape.Dimension(7)
    ])
    s2 = tensor_shape.TensorShape([
        tensor_shape.Dimension(3),
        tensor_shape.Dimension(4),
        tensor_shape.Dimension(7)
    ])
    s3 = tensor_shape.TensorShape(
        [tensor_shape.Dimension(3),
         tensor_shape.Dimension(4), None])

    self.assertEqual(s1, s2)
    self.assertEqual(s1, s2)

    # Test with an unknown shape in s3
    self.assertNotEqual(s1, s3)

    unk0 = tensor_shape.unknown_shape()
    self.assertNotEqual(unk0, s1)
    self.assertNotEqual(s1, unk0)
    self.assertNotEqual(unk0, s1)
    self.assertNotEqual(s1, unk0)

    unk1 = tensor_shape.unknown_shape()
    self.assertEqual(unk0, unk1)
    self.assertEqual(unk1, unk0)

  def testAsList(self):
    # [FUNCTION DOC]
    # Tests list conversion: an unknown-rank shape raises a value error saying conversion is not
    # defined for unknown shapes; a shape from unknown_shape with rank two gives two null entries; a
    # tuple-built shape gives its sizes with null for the unknown.
    with self.assertRaisesRegex(ValueError,
                                "not defined on an unknown TensorShape"):
      tensor_shape.unknown_shape().as_list()
    self.assertAllEqual([None, None], tensor_shape.unknown_shape(2).as_list())
    self.assertAllEqual([2, None, 4],
                        tensor_shape.TensorShape((2, None, 4)).as_list())

  def testReduce(self):
    # [FUNCTION DOC]
    # Tests pickling support: the reduce result is the TensorShape class plus an argument tuple holding
    # the list of Dimension objects, and calling the class with those arguments rebuilds an equal
    # shape.
    shape = tensor_shape.TensorShape([2, 3])
    ctor, args = shape.__reduce__()
    self.assertEqual(ctor, tensor_shape.TensorShape)
    self.assertEqual(args,
                     ([tensor_shape.Dimension(2),
                       tensor_shape.Dimension(3)],))
    reconstructed = ctor(*args)
    self.assertEqual(reconstructed, shape)


class TraceTypeTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case covering the trace-type behavior of TensorShape: subtype relation and most-specific
  # common supertype computation. Uses the standard TensorFlow test base class; no shared fixtures.

  def testSubtypeOfEqualTypes(self):
    # [FUNCTION DOC]
    # Tests subtyping of two separately created equal shapes: each is a subtype of the other.
    type_1 = tensor_shape.TensorShape([1, 2, 3])
    type_2 = tensor_shape.TensorShape([1, 2, 3])

    self.assertTrue(type_2.is_subtype_of(type_1))
    self.assertTrue(type_1.is_subtype_of(type_2))

  def testReflexivity(self):
    # [FUNCTION DOC]
    # Tests that a partially known shape is a subtype of itself (reflexivity).
    type_1 = tensor_shape.TensorShape([1, None, 3])
    self.assertTrue(type_1.is_subtype_of(type_1))

  def testSubtypeOfShapeless(self):
    # [FUNCTION DOC]
    # Tests subtyping against an unknown-rank shape: the shapes are not equal, an unknown-rank shape
    # is not a subtype of a known shape, and the known shape is a subtype of the unknown one.
    type_1 = tensor_shape.TensorShape(None)
    type_2 = tensor_shape.TensorShape([1, 2, 3])
    self.assertNotEqual(type_1, type_2)
    self.assertFalse(type_1.is_subtype_of(type_2))
    self.assertTrue(type_2.is_subtype_of(type_1))

  def testSubtypeOfDimlessShape(self):
    # [FUNCTION DOC]
    # Tests subtyping with an all-unknown-dimension shape versus a known shape of the same rank: not
    # equal, the all-unknown shape is not a subtype of the known one, and the known one is a subtype.
    type_1 = tensor_shape.TensorShape([None, None, None])
    type_2 = tensor_shape.TensorShape([1, 2, 3])
    self.assertNotEqual(type_1, type_2)
    self.assertFalse(type_1.is_subtype_of(type_2))
    self.assertTrue(type_2.is_subtype_of(type_1))

  def testSubtypeOfPartialShape(self):
    # [FUNCTION DOC]
    # Tests subtyping with a partially known shape versus a known shape: not equal, the partial shape
    # is not a subtype of the known one, and the known one is a subtype of the partial one.
    type_1 = tensor_shape.TensorShape([None, 2, None])
    type_2 = tensor_shape.TensorShape([1, 2, 3])
    self.assertNotEqual(type_1, type_2)
    self.assertFalse(type_1.is_subtype_of(type_2))
    self.assertTrue(type_2.is_subtype_of(type_1))

  def testSupertypeOfEqualTypes(self):
    # [FUNCTION DOC]
    # Tests the most specific common supertype of two equal known shapes: the result equals the shape,
    # checked from both sides.
    type_1 = tensor_shape.TensorShape([1, 2, 3])
    type_2 = tensor_shape.TensorShape([1, 2, 3])
    self.assertEqual(type_1.most_specific_common_supertype([type_2]), type_1)
    self.assertEqual(type_2.most_specific_common_supertype([type_1]), type_2)

  def testSupertypeOfRelatedTypes(self):
    # [FUNCTION DOC]
    # Tests the most specific common supertype of a known shape and a partial shape that generalizes
    # it: the result is the partial shape, checked from both sides.
    type_1 = tensor_shape.TensorShape([1, 2, 3])
    type_2 = tensor_shape.TensorShape([None, 2, 3])
    self.assertEqual(type_1.most_specific_common_supertype([type_2]), type_2)
    self.assertEqual(type_2.most_specific_common_supertype([type_1]), type_2)

  def testSupertypeOfUnrelatedTypes(self):
    # [FUNCTION DOC]
    # Tests the most specific common supertype of two shapes that disagree in positions: the result has
    # unknown size where sizes differ or are unknown and keeps the agreeing size (2); checked from both
    # sides.
    type_1 = tensor_shape.TensorShape([1, 2, 4])
    type_2 = tensor_shape.TensorShape([None, 2, 3])
    self.assertEqual(
        type_1.most_specific_common_supertype([type_2]),
        tensor_shape.TensorShape([None, 2, None]))
    self.assertEqual(
        type_2.most_specific_common_supertype([type_1]),
        tensor_shape.TensorShape([None, 2, None]))


if __name__ == "__main__":
  googletest.main()
