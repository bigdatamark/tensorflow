# [MODULE DOC]
# Unit tests for the data-specific nested structure utilities: flatten, pack, nesting detection,
# structure assertions, map, and up-to variants, with named tuples, dicts, ordered dicts, data
# classes, sparse and ragged values. Runs the tests when executed as a program.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for utilities working with arbitrarily nested structures."""

import collections
import dataclasses

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.util import nest
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test


@dataclasses.dataclass
class MaskedTensor:
  # [CLASS DOC]
  # HELPER: small test data class pairing a boolean mask with a tensor value. Implements the
  # flatten and unflatten hooks so nest utilities treat it as a composite with one component
  # (the value) and the mask as metadata. Used only by the data class tests below.
  mask: bool
  value: tensor.Tensor

  def __tf_flatten__(self):
    # [FUNCTION DOC]
    # HELPER: hook used by nest utilities to decompose the value. Returns a pair: metadata (a
    # tuple holding the mask) and components (a tuple holding the value tensor).
    metadata = (self.mask,)
    components = (self.value,)
    return metadata, components

  def __tf_unflatten__(self, metadata, components):
    # [FUNCTION DOC]
    # HELPER: hook used by nest utilities to rebuild the value. metadata is a tuple whose first
    # entry is the mask; components is a tuple whose first entry is the value tensor. Returns a
    # new masked tensor holding them.
    mask = metadata[0]
    value = components[0]
    return MaskedTensor(mask=mask, value=value)


class NestTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test class for the data-specific nest utilities (flatten, pack, map, structure assertions,
  # up-to variants) including data class, sparse and ragged value handling. Uses the dataset
  # test base class and the default test combinations.

  @combinations.generate(test_base.default_test_combinations())
  def testFlattenAndPack(self):
    # [FUNCTION DOC]
    # Test: basic flatten and pack. Asserts flattening nested tuples and named tuples gives the
    # leaves in order; packing a flat list restores the structure (named tuple fields intact);
    # scalars flatten to a one-element list and pack from one-element lists; and errors: value
    # error when a scalar structure gets two elements, type error when the flat sequence is not
    # a sequence, value error when element counts differ.
    structure = ((3, 4), 5, (6, 7, (9, 10), 8))
    flat = ["a", "b", "c", "d", "e", "f", "g", "h"]
    self.assertEqual(nest.flatten(structure), [3, 4, 5, 6, 7, 9, 10, 8])
    self.assertEqual(
        nest.pack_sequence_as(structure, flat), (("a", "b"), "c",
                                                 ("d", "e", ("f", "g"), "h")))
    point = collections.namedtuple("Point", ["x", "y"])
    structure = (point(x=4, y=2), ((point(x=1, y=0),),))
    flat = [4, 2, 1, 0]
    self.assertEqual(nest.flatten(structure), flat)
    restructured_from_flat = nest.pack_sequence_as(structure, flat)
    self.assertEqual(restructured_from_flat, structure)
    self.assertEqual(restructured_from_flat[0].x, 4)
    self.assertEqual(restructured_from_flat[0].y, 2)
    self.assertEqual(restructured_from_flat[1][0][0].x, 1)
    self.assertEqual(restructured_from_flat[1][0][0].y, 0)

    self.assertEqual([5], nest.flatten(5))
    self.assertEqual([np.array([5])], nest.flatten(np.array([5])))

    self.assertEqual("a", nest.pack_sequence_as(5, ["a"]))
    self.assertEqual(
        np.array([5]), nest.pack_sequence_as("scalar", [np.array([5])]))

    with self.assertRaisesRegex(ValueError, "Argument `structure` is a scalar"):
      nest.pack_sequence_as("scalar", [4, 5])

    with self.assertRaisesRegex(TypeError, "flat_sequence"):
      nest.pack_sequence_as([4, 5], "bad_sequence")

    with self.assertRaises(ValueError):
      nest.pack_sequence_as([5, 6, [7, 8]], ["a", "b", "c"])

  @combinations.generate(test_base.default_test_combinations())
  def testDataclassIsNested(self):
    # [FUNCTION DOC]
    # Test: a data class value that provides flatten and unflatten hooks is reported as nested.
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    self.assertTrue(nest.is_nested(mt))

  @combinations.generate(test_base.default_test_combinations())
  def testFlattenDataclass(self):
    # [FUNCTION DOC]
    # Test: flattening a data class value yields exactly one leaf, the wrapped tensor.
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    leaves = nest.flatten(mt)
    self.assertLen(leaves, 1)
    self.assertAllEqual(leaves[0], [1])

  @combinations.generate(test_base.default_test_combinations())
  def testPackDataclass(self):
    # [FUNCTION DOC]
    # Test: packing leaves back into a data class value. Asserts the result has the same class
    # and the same mask and value as the original; packing with a template having a different
    # mask keeps that template's mask but takes the value from the supplied leaves.
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    leaves = nest.flatten(mt)
    reconstructed_mt = nest.pack_sequence_as(mt, leaves)
    self.assertIsInstance(reconstructed_mt, MaskedTensor)
    self.assertEqual(reconstructed_mt.mask, mt.mask)
    self.assertAllEqual(reconstructed_mt.value, mt.value)

    mt2 = MaskedTensor(mask=False, value=constant_op.constant([2]))
    reconstructed_mt = nest.pack_sequence_as(mt2, leaves)
    self.assertIsInstance(reconstructed_mt, MaskedTensor)
    self.assertFalse(reconstructed_mt.mask)
    self.assertAllEqual(reconstructed_mt.value, [1])

  @combinations.generate(test_base.default_test_combinations())
  def testDataclassMapStructure(self):
    # [FUNCTION DOC]
    # Test: mapping a doubling function over a data class value; asserts the result is the same
    # class, the mask is unchanged and the wrapped tensor is doubled.
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    mt_doubled = nest.map_structure(lambda x: x * 2, mt)
    self.assertIsInstance(mt_doubled, MaskedTensor)
    self.assertEqual(mt_doubled.mask, True)
    self.assertAllEqual(mt_doubled.value, [2])

  @combinations.generate(test_base.default_test_combinations())
  def testDataclassAssertSameStructure(self):
    # [FUNCTION DOC]
    # Test: same-structure assertion with data class values. Two instances with different field
    # values match; comparing with a tuple raises a type error about nested structure; a subclass
    # instance matches when type checking is off and raises a type error about sequence type
    # when it is on.
    mt1 = MaskedTensor(mask=True, value=constant_op.constant([1]))
    mt2 = MaskedTensor(mask=False, value=constant_op.constant([2]))
    nest.assert_same_structure(mt1, mt2)

    mt3 = (1, 2)

    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        TypeError,
        "don't have the same nested structure",
    ):
      nest.assert_same_structure(mt1, mt3)

    class SubMaskedTensor(MaskedTensor):
      # [CLASS DOC]
      # Local subclass of the masked tensor data class with no additions; used to check that a
      # subclass instance differs in sequence type from the base class.
      pass

    mt_subclass = SubMaskedTensor(mask=True, value=constant_op.constant([1]))
    nest.assert_same_structure(mt1, mt_subclass, check_types=False)
    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        TypeError,
        "don't have the same sequence type",
    ):
      nest.assert_same_structure(mt1, mt_subclass)

  @combinations.generate(test_base.default_test_combinations())
  def testDataclassAssertShallowStructure(self):
    # [FUNCTION DOC]
    # Test: shallow-structure assertion with a data class value as a leaf. Asserts a matching
    # shallow tuple passes; a longer input raises a value error about sequence length; a dict
    # input passes when type checking is off and raises a type error about sequence type
    # otherwise.
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    structure1 = ("a", "b")
    structure2 = (mt, "c")
    nest.assert_shallow_structure(structure1, structure2)

    structure3 = (mt, "d", "e")

    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        ValueError,
        "don't have the same sequence length",
    ):
      nest.assert_shallow_structure(structure1, structure3)

    structure4 = {"a": mt, "b": "c"}
    nest.assert_shallow_structure(structure1, structure4, check_types=False)
    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        TypeError,
        "don't have the same sequence type",
    ):
      nest.assert_shallow_structure(structure1, structure4)

  @combinations.generate(test_base.default_test_combinations())
  def testFlattenDictOrder(self):
    # [FUNCTION DOC]
    # Test: flattening a plain dict and an ordered dict with the same entries; asserts both
    # produce values ordered by sorted key.
    """`flatten` orders dicts by key, including OrderedDicts."""
    ordered = collections.OrderedDict([("d", 3), ("b", 1), ("a", 0), ("c", 2)])
    plain = {"d": 3, "b": 1, "a": 0, "c": 2}
    ordered_flat = nest.flatten(ordered)
    plain_flat = nest.flatten(plain)
    self.assertEqual([0, 1, 2, 3], ordered_flat)
    self.assertEqual([0, 1, 2, 3], plain_flat)

  @combinations.generate(test_base.default_test_combinations())
  def testPackDictOrder(self):
    # [FUNCTION DOC]
    # Test: packing a flat sequence into a plain dict or an ordered dict assigns values in sorted
    # key order, while the ordered dict keeps its own key order. Asserts both reconstructions.
    """Packing orders dicts by key, including OrderedDicts."""
    ordered = collections.OrderedDict([("d", 0), ("b", 0), ("a", 0), ("c", 0)])
    plain = {"d": 0, "b": 0, "a": 0, "c": 0}
    seq = [0, 1, 2, 3]
    ordered_reconstruction = nest.pack_sequence_as(ordered, seq)
    plain_reconstruction = nest.pack_sequence_as(plain, seq)
    self.assertEqual(
        collections.OrderedDict([("d", 3), ("b", 1), ("a", 0), ("c", 2)]),
        ordered_reconstruction)
    self.assertEqual({"d": 3, "b": 1, "a": 0, "c": 2}, plain_reconstruction)

  @combinations.generate(test_base.default_test_combinations())
  def testFlattenAndPackWithDicts(self):
    # [FUNCTION DOC]
    # Test: flatten and pack on a mix of tuples, named tuples, dicts and ordered dicts. Asserts
    # flattening orders dict values by key, packing with another structure of the same shape
    # reproduces the original, and the ordered dict is recreated with its key order preserved.
    # A nice messy mix of tuples, lists, dicts, and `OrderedDict`s.
    named_tuple = collections.namedtuple("A", ("b", "c"))
    mess = (
        "z",
        named_tuple(3, 4),
        {
            "c": (
                1,
                collections.OrderedDict([
                    ("b", 3),
                    ("a", 2),
                ]),
            ),
            "b": 5
        },
        17
    )

    flattened = nest.flatten(mess)
    self.assertEqual(flattened, ["z", 3, 4, 5, 1, 2, 3, 17])

    structure_of_mess = (
        14,
        named_tuple("a", True),
        {
            "c": (
                0,
                collections.OrderedDict([
                    ("b", 9),
                    ("a", 8),
                ]),
            ),
            "b": 3
        },
        "hi everybody",
    )

    unflattened = nest.pack_sequence_as(structure_of_mess, flattened)
    self.assertEqual(unflattened, mess)

    # Check also that the OrderedDict was created, with the correct key order.
    unflattened_ordered_dict = unflattened[2]["c"][1]
    self.assertIsInstance(unflattened_ordered_dict, collections.OrderedDict)
    self.assertEqual(list(unflattened_ordered_dict.keys()), ["b", "a"])

  @combinations.generate(test_base.default_test_combinations())
  def testFlattenSparseValue(self):
    # [FUNCTION DOC]
    # Test: a sparse tensor value is treated as a single atomic leaf. Asserts flattening a single
    # value, a list of values (the list is a leaf), a nested tuple and a dict of values.
    st = sparse_tensor.SparseTensorValue([[0]], [0], [1])
    single_value = st
    list_of_values = [st, st, st]
    nest_of_values = ((st), ((st), (st)))
    dict_of_values = {"foo": st, "bar": st, "baz": st}
    self.assertEqual([st], nest.flatten(single_value))
    self.assertEqual([[st, st, st]], nest.flatten(list_of_values))
    self.assertEqual([st, st, st], nest.flatten(nest_of_values))
    self.assertEqual([st, st, st], nest.flatten(dict_of_values))

  @combinations.generate(test_base.default_test_combinations())
  def testFlattenRaggedValue(self):
    # [FUNCTION DOC]
    # Test: a ragged tensor value is treated as a single atomic leaf. Asserts flattening a single
    # value, a list of values (the list is a leaf), a nested tuple and a dict of values.
    rt = ragged_factory_ops.constant_value([[[0]], [[1]]])
    single_value = rt
    list_of_values = [rt, rt, rt]
    nest_of_values = ((rt), ((rt), (rt)))
    dict_of_values = {"foo": rt, "bar": rt, "baz": rt}
    self.assertEqual([rt], nest.flatten(single_value))
    self.assertEqual([[rt, rt, rt]], nest.flatten(list_of_values))
    self.assertEqual([rt, rt, rt], nest.flatten(nest_of_values))
    self.assertEqual([rt, rt, rt], nest.flatten(dict_of_values))

  @combinations.generate(test_base.default_test_combinations())
  def testIsNested(self):
    # [FUNCTION DOC]
    # Test: nesting detection. Asserts tuples and dicts are nested, while strings, lists, empty
    # lists, sets, tensors, numpy arrays, sparse tensor values and ragged tensor values are not.
    self.assertFalse(nest.is_nested("1234"))
    self.assertFalse(nest.is_nested([1, 3, [4, 5]]))
    self.assertTrue(nest.is_nested(((7, 8), (5, 6))))
    self.assertFalse(nest.is_nested([]))
    self.assertFalse(nest.is_nested(set([1, 2])))
    ones = array_ops.ones([2, 3])
    self.assertFalse(nest.is_nested(ones))
    self.assertFalse(nest.is_nested(math_ops.tanh(ones)))
    self.assertFalse(nest.is_nested(np.ones((4, 5))))
    self.assertTrue(nest.is_nested({"foo": 1, "bar": 2}))
    self.assertFalse(
        nest.is_nested(sparse_tensor.SparseTensorValue([[0]], [0], [1])))
    self.assertFalse(
        nest.is_nested(ragged_factory_ops.constant_value([[[0]], [[1]]])))

  @combinations.generate(test_base.default_test_combinations())
  def testAssertSameStructure(self):
    # [FUNCTION DOC]
    # Test: same-structure assertion. Asserts identical nesting passes regardless of leaf values
    # (strings, floats, arrays, tensors); mismatched element counts or nesting raise a value
    # error; different named tuple types or tuple versus named tuple raise a type error; dicts
    # versus tuples raise a type error about sequence type; different dict keys raise a value
    # error about keys; and disabling type checking relaxes the type and key mismatches.
    structure1 = (((1, 2), 3), 4, (5, 6))
    structure2 = ((("foo1", "foo2"), "foo3"), "foo4", ("foo5", "foo6"))
    structure_different_num_elements = ("spam", "eggs")
    structure_different_nesting = (((1, 2), 3), 4, 5, (6,))
    structure_dictionary = {"foo": 2, "bar": 4, "baz": {"foo": 5, "bar": 6}}
    structure_dictionary_diff_nested = {
        "foo": 2,
        "bar": 4,
        "baz": {
            "foo": 5,
            "baz": 6
        }
    }
    nest.assert_same_structure(structure1, structure2)
    nest.assert_same_structure("abc", 1.0)
    nest.assert_same_structure("abc", np.array([0, 1]))
    nest.assert_same_structure("abc", constant_op.constant([0, 1]))

    with self.assertRaisesRegex(ValueError,
                                "don't have the same nested structure"):
      nest.assert_same_structure(structure1, structure_different_num_elements)

    with self.assertRaisesRegex(ValueError,
                                "don't have the same nested structure"):
      nest.assert_same_structure((0, 1), np.array([0, 1]))

    with self.assertRaisesRegex(ValueError,
                                "don't have the same nested structure"):
      nest.assert_same_structure(0, (0, 1))

    with self.assertRaisesRegex(ValueError,
                                "don't have the same nested structure"):
      nest.assert_same_structure(structure1, structure_different_nesting)

    named_type_0 = collections.namedtuple("named_0", ("a", "b"))
    named_type_1 = collections.namedtuple("named_1", ("a", "b"))
    self.assertRaises(TypeError, nest.assert_same_structure, (0, 1),
                      named_type_0("a", "b"))

    nest.assert_same_structure(named_type_0(3, 4), named_type_0("a", "b"))

    self.assertRaises(TypeError, nest.assert_same_structure,
                      named_type_0(3, 4), named_type_1(3, 4))

    with self.assertRaisesRegex(ValueError,
                                "don't have the same nested structure"):
      nest.assert_same_structure(named_type_0(3, 4), named_type_0((3,), 4))

    with self.assertRaisesRegex(ValueError,
                                "don't have the same nested structure"):
      nest.assert_same_structure(((3,), 4), (3, (4,)))

    structure1_list = {"a": ((1, 2), 3), "b": 4, "c": (5, 6)}
    structure2_list = {"a": ((1, 2), 3), "b": 4, "d": (5, 6)}
    with self.assertRaisesRegex(TypeError, "don't have the same sequence type"):
      nest.assert_same_structure(structure1, structure1_list)
    nest.assert_same_structure(structure1, structure2, check_types=False)
    nest.assert_same_structure(structure1, structure1_list, check_types=False)
    with self.assertRaisesRegex(ValueError, "don't have the same set of keys"):
      nest.assert_same_structure(structure1_list, structure2_list)
    with self.assertRaisesRegex(ValueError, "don't have the same set of keys"):
      nest.assert_same_structure(structure_dictionary,
                                 structure_dictionary_diff_nested)
    nest.assert_same_structure(
        structure_dictionary,
        structure_dictionary_diff_nested,
        check_types=False)
    nest.assert_same_structure(
        structure1_list, structure2_list, check_types=False)

  @combinations.generate(test_base.default_test_combinations())
  def testMapStructure(self):
    # [FUNCTION DOC]
    # Test: mapping over structures. Asserts mapping an increment keeps structure and values,
    # mapping a two-argument sum over two structures works, scalars are accepted, and errors are
    # raised: type error for a non-callable function, value error for mismatched nesting, type
    # error for mismatched sequence types, and value errors for unknown keyword arguments.
    structure1 = (((1, 2), 3), 4, (5, 6))
    structure2 = (((7, 8), 9), 10, (11, 12))
    structure1_plus1 = nest.map_structure(lambda x: x + 1, structure1)
    nest.assert_same_structure(structure1, structure1_plus1)
    self.assertAllEqual(
        [2, 3, 4, 5, 6, 7],
        nest.flatten(structure1_plus1))
    structure1_plus_structure2 = nest.map_structure(
        lambda x, y: x + y, structure1, structure2)
    self.assertEqual(
        (((1 + 7, 2 + 8), 3 + 9), 4 + 10, (5 + 11, 6 + 12)),
        structure1_plus_structure2)

    self.assertEqual(3, nest.map_structure(lambda x: x - 1, 4))

    self.assertEqual(7, nest.map_structure(lambda x, y: x + y, 3, 4))

    with self.assertRaisesRegex(TypeError, "callable"):
      nest.map_structure("bad", structure1_plus1)

    with self.assertRaisesRegex(ValueError, "same nested structure"):
      nest.map_structure(lambda x, y: None, 3, (3,))

    with self.assertRaisesRegex(TypeError, "same sequence type"):
      nest.map_structure(lambda x, y: None, ((3, 4), 5), {"a": (3, 4), "b": 5})

    with self.assertRaisesRegex(ValueError, "same nested structure"):
      nest.map_structure(lambda x, y: None, ((3, 4), 5), (3, (4, 5)))

    with self.assertRaisesRegex(ValueError, "same nested structure"):
      nest.map_structure(lambda x, y: None, ((3, 4), 5), (3, (4, 5)),
                         check_types=False)

    with self.assertRaisesRegex(ValueError, "Only valid keyword argument"):
      nest.map_structure(lambda x: None, structure1, foo="a")

    with self.assertRaisesRegex(ValueError, "Only valid keyword argument"):
      nest.map_structure(lambda x: None, structure1, check_types=False, foo="a")

  @combinations.generate(test_base.default_test_combinations())
  def testAssertShallowStructure(self):
    # [FUNCTION DOC]
    # Test: shallow-structure assertion. Asserts a value error on different sequence lengths, a
    # type error on different sequence types (tuple versus dict) unless type checking is turned
    # off, a value error when nested dicts have different keys, and no error when two ordered
    # dicts have the same keys in different order.
    inp_ab = ("a", "b")
    inp_abc = ("a", "b", "c")
    expected_message = (
        "The two structures don't have the same sequence length. Input "
        "structure has length 2, while shallow structure has length 3.")
    with self.assertRaisesRegex(ValueError, expected_message):
      nest.assert_shallow_structure(inp_abc, inp_ab)

    inp_ab1 = ((1, 1), (2, 2))
    inp_ab2 = {"a": (1, 1), "b": (2, 2)}
    expected_message = (
        "The two structures don't have the same sequence type. Input structure "
        "has type 'tuple', while shallow structure has type "
        "'dict'.")
    with self.assertRaisesRegex(TypeError, expected_message):
      nest.assert_shallow_structure(inp_ab2, inp_ab1)
    nest.assert_shallow_structure(inp_ab2, inp_ab1, check_types=False)

    inp_ab1 = {"a": (1, 1), "b": {"c": (2, 2)}}
    inp_ab2 = {"a": (1, 1), "b": {"d": (2, 2)}}
    expected_message = (
        r"The two structures don't have the same keys. Input "
        r"structure has keys \['c'\], while shallow structure has "
        r"keys \['d'\].")
    with self.assertRaisesRegex(ValueError, expected_message):
      nest.assert_shallow_structure(inp_ab2, inp_ab1)

    inp_ab = collections.OrderedDict([("a", 1), ("b", (2, 3))])
    inp_ba = collections.OrderedDict([("b", (2, 3)), ("a", 1)])
    nest.assert_shallow_structure(inp_ab, inp_ba)

  @combinations.generate(test_base.default_test_combinations())
  def testFlattenUpTo(self):
    # [FUNCTION DOC]
    # Test: flattening an input tree up to a shallow tree. Asserts partial flattening of nested
    # tuples and dicts retains deeper substructure as single elements; that non-sequence shallow
    # trees (strings, numbers) give a one-element list holding the input; and that a sequence
    # shallow tree with a non-sequence input raises a type error mentioning that input must also
    # be a sequence (checked for strings and integers). Flattening the shallow tree against
    # itself yields its own leaves.
    input_tree = (((2, 2), (3, 3)), ((4, 9), (5, 5)))
    shallow_tree = ((True, True), (False, True))
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [(2, 2), (3, 3), (4, 9), (5, 5)])
    self.assertEqual(flattened_shallow_tree, [True, True, False, True])

    input_tree = ((("a", 1), (("b", 2), (("c", 3), (("d", 4))))))
    shallow_tree = (("level_1", ("level_2", ("level_3", ("level_4")))))
    input_tree_flattened_as_shallow_tree = nest.flatten_up_to(shallow_tree,
                                                              input_tree)
    input_tree_flattened = nest.flatten(input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [("a", 1), ("b", 2), ("c", 3), ("d", 4)])
    self.assertEqual(input_tree_flattened, ["a", 1, "b", 2, "c", 3, "d", 4])

    ## Shallow non-list edge-case.
    # Using iterable elements.
    input_tree = ["input_tree"]
    shallow_tree = "shallow_tree"
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    input_tree = ("input_tree_0", "input_tree_1")
    shallow_tree = "shallow_tree"
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    # Using non-iterable elements.
    input_tree = (0,)
    shallow_tree = 9
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    input_tree = (0, 1)
    shallow_tree = 9
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    ## Both non-list edge-case.
    # Using iterable elements.
    input_tree = "input_tree"
    shallow_tree = "shallow_tree"
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    # Using non-iterable elements.
    input_tree = 0
    shallow_tree = 0
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    ## Input non-list edge-case.
    # Using iterable elements.
    input_tree = "input_tree"
    shallow_tree = ("shallow_tree",)
    expected_message = ("If shallow structure is a sequence, input must also "
                        "be a sequence. Input has type: 'str'.")
    with self.assertRaisesRegex(TypeError, expected_message):
      flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree, list(shallow_tree))

    input_tree = "input_tree"
    shallow_tree = ("shallow_tree_9", "shallow_tree_8")
    with self.assertRaisesRegex(TypeError, expected_message):
      flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree, list(shallow_tree))

    # Using non-iterable elements.
    input_tree = 0
    shallow_tree = (9,)
    expected_message = ("If shallow structure is a sequence, input must also "
                        "be a sequence. Input has type: 'int'.")
    with self.assertRaisesRegex(TypeError, expected_message):
      flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree, list(shallow_tree))

    input_tree = 0
    shallow_tree = (9, 8)
    with self.assertRaisesRegex(TypeError, expected_message):
      flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree, list(shallow_tree))

    # Using dict.
    input_tree = {"a": ((2, 2), (3, 3)), "b": ((4, 9), (5, 5))}
    shallow_tree = {"a": (True, True), "b": (False, True)}
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [(2, 2), (3, 3), (4, 9), (5, 5)])
    self.assertEqual(flattened_shallow_tree, [True, True, False, True])

  @combinations.generate(test_base.default_test_combinations())
  def testMapStructureUpTo(self):
    # [FUNCTION DOC]
    # Test: mapping a function over inputs flattened up to a shallow tree. With a named tuple of
    # two numbers and a matching named tuple of operation tuples, asserts the result fields
    # are computed from sum-then-product values. With nested name and data tuples, asserts the
    # result is a nested tuple of strings built from the length of each data entry and its name.
    ab_tuple = collections.namedtuple("ab_tuple", "a, b")
    op_tuple = collections.namedtuple("op_tuple", "add, mul")
    inp_val = ab_tuple(a=2, b=3)
    inp_ops = ab_tuple(a=op_tuple(add=1, mul=2), b=op_tuple(add=2, mul=3))
    out = nest.map_structure_up_to(
        inp_val, lambda val, ops: (val + ops.add) * ops.mul, inp_val, inp_ops)
    self.assertEqual(out.a, 6)
    self.assertEqual(out.b, 15)

    data_list = ((2, 4, 6, 8), ((1, 3, 5, 7, 9), (3, 5, 7)))
    name_list = ("evens", ("odds", "primes"))
    out = nest.map_structure_up_to(
        name_list, lambda name, sec: "first_{}_{}".format(len(sec), name),
        name_list, data_list)
    self.assertEqual(out, ("first_4_evens", ("first_5_odds", "first_3_primes")))


if __name__ == "__main__":
  test.main()
