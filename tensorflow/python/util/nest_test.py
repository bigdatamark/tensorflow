# [MODULE DOC]
# Unit tests for the nest utility module (tensorflow.python.util.nest), which flattens, packs,
# maps over and compares arbitrarily nested structures (lists, tuples, namedtuples, dicts,
# ordered dicts, custom mappings, attrs classes, dataclasses implementing the custom nest
# protocol, mapping views and, optionally, composite tensors).
# Contents: small helper classes (a custom read-only mapping, a list subclass, a sequence whose
# item access fails, and two dataclasses MaskedTensor and NestedMaskedTensor that implement the
# flatten/unflatten protocol), the large NestTest test case, and NestBenchmark, a micro-benchmark
# of structure comparison.
# Import-time side effects: tries to import the optional attrs package (falls back to nothing if
# missing); defines namedtuple types at class level; running the file as a program starts the
# TensorFlow test runner. This is a test module, not a helper module.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
import collections.abc
import dataclasses
import time
from typing import NamedTuple

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.util import nest
from tensorflow.python.util.nest_util import CustomNestProtocol

try:
  import attr  # pylint:disable=g-import-not-at-top
except ImportError:
  attr = None


class _CustomMapping(collections.abc.Mapping):
  # [CLASS DOC]
  # HELPER: test fixture for the nest tests (custom mapping type that is not a plain dict).
  # Minimal read-only mapping that wraps an internal dict. It is built from the same arguments
  # as a dict and supports lookup, iteration and length, so nest must treat it as a mapping
  # (sorted by key) without it being a dict subclass.

  def __init__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: constructor of the test mapping fixture.
    # Builds the wrapped dictionary from any positional and keyword arguments accepted by a
    # dict constructor and stores it privately. Returns nothing; raises whatever the dict
    # constructor raises for invalid arguments.
    self._wrapped = dict(*args, **kwargs)

  def __getitem__(self, key):
    # [FUNCTION DOC]
    # HELPER: mapping protocol method of the test fixture.
    # Returns the value stored for the given key in the wrapped dictionary; a missing key
    # raises a key error.
    return self._wrapped[key]

  def __iter__(self):
    # [FUNCTION DOC]
    # HELPER: mapping protocol method of the test fixture.
    # Returns an iterator over the keys of the wrapped dictionary, in insertion order.
    return iter(self._wrapped)

  def __len__(self):
    # [FUNCTION DOC]
    # HELPER: mapping protocol method of the test fixture.
    # Returns the number of entries in the wrapped dictionary.
    return len(self._wrapped)


class _CustomList(list):
  # [CLASS DOC]
  # HELPER: test fixture for the nest tests.
  # Empty subclass of list, used to check that a list subclass is treated as having the same
  # shallow structure as a plain list when types are compared.
  pass


class _CustomSequenceThatRaisesException(collections.abc.Sequence):
  # [CLASS DOC]
  # HELPER: test fixture for the nest tests.
  # Sequence that claims to have exactly one element but raises a value error whenever an item
  # is read. Used to verify that errors raised while flattening are propagated, not swallowed.

  def __len__(self):
    # [FUNCTION DOC]
    # HELPER: sequence protocol method of the failing-sequence fixture.
    # Always returns one, so consumers believe the sequence has a single element to fetch.
    return 1

  def __getitem__(self, item):
    # [FUNCTION DOC]
    # HELPER: sequence protocol method of the failing-sequence fixture.
    # Takes the requested index and always raises a value error whose message contains the
    # text "Cannot get item" and the index. Never returns a value.
    raise ValueError("Cannot get item: %s" % item)


@dataclasses.dataclass
class MaskedTensor:
  # [CLASS DOC]
  # HELPER: test fixture implementing the custom nest protocol.
  # Dataclass with two fields: a boolean mask (kept as static metadata) and a tensor value (the
  # single component that nest flattens). It provides flatten and unflatten methods so nest
  # recognises it as a nested structure with exactly one leaf, and a length of one so the
  # arity checks of the map-up-to functions succeed. Used by the Dataclass tests of NestTest.
  mask: bool
  value: tensor.Tensor

  def __tf_flatten__(self):
    # [FUNCTION DOC]
    # HELPER: nest custom protocol hook for the test dataclass.
    # Returns a pair: the metadata (a one-element tuple holding the mask) and the components
    # (a one-element tuple holding the tensor value). No side effects.
    metadata = (self.mask,)
    components = (self.value,)
    return metadata, components

  @classmethod
  def __tf_unflatten__(cls, metadata, components):
    # [FUNCTION DOC]
    # HELPER: nest custom protocol hook for the test dataclass (class method).
    # Inputs: the metadata tuple produced by the flatten hook and the components tuple of
    # (possibly transformed) leaves. Returns a new instance whose mask is the first metadata
    # item and whose value is the first component.
    mask = metadata[0]
    value = components[0]
    return MaskedTensor(mask=mask, value=value)

  def __eq__(self, other):
    # [FUNCTION DOC]
    # HELPER: equality for the test dataclass.
    # KNOWN ISSUE: when the masks are equal the result is a tensor (reduce-all of an elementwise
    # comparison), not a plain boolean as the equality protocol suggests; only the mask check
    # yields a plain boolean. Tests rely on this being truthy in assertions.
    # Compares the mask and requires every element of the tensor values to be equal.
    return self.mask == other.mask and math_ops.reduce_all(
        self.value == other.value
    )

  def __len__(self):
    # [FUNCTION DOC]
    # HELPER: arity hook for the test dataclass.
    # Always returns one (the number of components). The in-code comment explains that the map
    # up-to helpers use it to verify that structures have compatible arity.
    # Used by `nest.map_structure_up_to` and related functions to verify the
    # arity compatibility.
    return 1


class MaskedTensor2(MaskedTensor):
  # [CLASS DOC]
  # HELPER: test fixture for the nest tests.
  # Empty subclass of MaskedTensor with identical behavior but a different type, used to check
  # shallow-structure comparison with and without type checking.
  pass


@dataclasses.dataclass
class NestedMaskedTensor:
  # [CLASS DOC]
  # HELPER: test fixture implementing the custom nest protocol.
  # Dataclass with a boolean mask (metadata) and a MaskedTensor value (the single component), so
  # it nests two levels deep and has one leaf tensor. Implements flatten/unflatten hooks and a
  # length of one. Used to test dataclasses nested inside dataclasses.
  mask: bool
  value: MaskedTensor

  @classmethod
  def nested_masked_tensor_with_opposite_masks(cls, mask, inner_value):
    # [FUNCTION DOC]
    # HELPER: factory for the nested dataclass fixture (class method).
    # Inputs: the outer boolean mask and the tensor used as the innermost value. Returns a
    # nested dataclass whose inner MaskedTensor has the logical opposite of the outer mask.
    # Makes it easy to tell outer and inner metadata apart in assertions.
    return NestedMaskedTensor(
        mask=mask, value=MaskedTensor(mask=not mask, value=inner_value)
    )

  def __tf_flatten__(self):
    # [FUNCTION DOC]
    # HELPER: nest custom protocol hook for the nested test dataclass.
    # Returns the metadata (one-element tuple with the outer mask) and the components (one-element
    # tuple holding the inner MaskedTensor, which nest then flattens further).
    metadata = (self.mask,)
    components = (self.value,)
    return metadata, components

  @classmethod
  def __tf_unflatten__(cls, metadata, components):
    # [FUNCTION DOC]
    # HELPER: nest custom protocol hook for the nested test dataclass (class method).
    # Rebuilds an instance from the metadata tuple and the components tuple, taking the mask from
    # the first metadata item and the inner value from the first component.
    mask = metadata[0]
    value = components[0]
    return NestedMaskedTensor(mask=mask, value=value)

  def __eq__(self, other):
    # [FUNCTION DOC]
    # HELPER: equality for the nested test dataclass.
    # KNOWN ISSUE: the result may be a tensor rather than a plain boolean, because the inner
    # comparison of MaskedTensor values returns a tensor when the inner masks match.
    # Compares the outer masks and then delegates to the equality of the inner values.
    return self.mask == other.mask and self.value == other.value

  def __len__(self):
    # [FUNCTION DOC]
    # HELPER: arity hook for the nested test dataclass.
    # Always returns one, the number of components.
    return 1


class NestTest(parameterized.TestCase, test.TestCase):
  # [CLASS DOC]
  # Test case (parameterized, based on the TensorFlow test case) covering the public functions
  # of the nest module: flatten, pack-sequence-as, is-nested, flatten of dictionary items,
  # structure assertions, map-structure and its path/up-to variants, shallow-structure helpers,
  # path generation, list-to-tuple conversion and namedtuple helpers. Also covers dataclasses
  # implementing the custom nest protocol and attrs classes.
  # Shared fixtures are class attributes: several namedtuple types (including pairs that share
  # a name or field names), regular-expression patterns for expected error messages, an empty
  # namedtuple, and (when attrs is installed) three attrs sample classes.

  PointXY = collections.namedtuple("Point", ["x", "y"])  # pylint: disable=invalid-name
  unsafe_map_pattern = ("nest cannot guarantee that it is safe to map one to "
                        "the other.")
  bad_pack_pattern = ("Attempted to pack value:\n  .+\ninto a structure, but "
                      "found incompatible type `<(type|class) 'str'>` instead.")

  if attr:
    class BadAttr(object):
      # [CLASS DOC]
      # HELPER: test fixture for NestTest (only defined when attrs is installed).
      # Class that has an attrs-attributes marker that is not iterable. Flattening an instance
      # must fail with a type error.
      """Class that has a non-iterable __attrs_attrs__."""
      __attrs_attrs__ = None

    @attr.s
    class SampleAttr(object):
      # [CLASS DOC]
      # HELPER: test fixture for NestTest (only defined when attrs is installed).
      # attrs class with two attributes, field1 and field2, declared in that order. Used to
      # check flattening, packing and recognition of attrs instances.
      field1 = attr.ib()
      field2 = attr.ib()

    @attr.s
    class UnsortedSampleAttr(object):
      # [CLASS DOC]
      # HELPER: test fixture for NestTest (only defined when attrs is installed).
      # attrs class with three attributes declared in the order field3, field1, field2 (not
      # sorted). Used to check that mapping over an attrs instance preserves its fields.
      field3 = attr.ib()
      field1 = attr.ib()
      field2 = attr.ib()

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassCustomProtocol(self):
    # [FUNCTION DOC]
    # Test: a MaskedTensor wrapping a constant tensor is an instance of the custom nest protocol
    # type. Asserts the instance check succeeds. Checks for leaked Python objects when run eagerly.
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    self.assertIsInstance(mt, CustomNestProtocol)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassIsNested(self):
    # [FUNCTION DOC]
    # Test: nest recognises a MaskedTensor dataclass as a nested structure. Asserts the is-nested
    # query returns true. Checks for leaked Python objects when run eagerly.
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    self.assertTrue(nest.is_nested(mt))

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassFlatten(self):
    # [FUNCTION DOC]
    # Test: flattening a MaskedTensor yields a list with exactly one leaf, the wrapped tensor.
    # Asserts the length is one and the leaf equals the original values.
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    leaves = nest.flatten(mt)
    self.assertLen(leaves, 1)
    self.assertAllEqual(leaves[0], [1])

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassFlattenUpToCompatible(self):
    # [FUNCTION DOC]
    # Test: flatten-up-to between a dataclass and plain lists with type checking disabled.
    # Scenarios: a one-element list as shallow tree over a dataclass input (leaf is the tensor),
    # a dataclass as shallow tree over a list input (leaf is the list element), then the same two
    # directions with a nested list and a nested dataclass. Asserts the flattened leaves match.
    simple_list = [2]
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    flattened_mt = nest.flatten_up_to(
        shallow_tree=simple_list, input_tree=mt, check_types=False
    )
    # Expected flat_path_mt = [Tensor([1])]
    self.assertAllEqual(flattened_mt[0], [1])
    flattened_list = nest.flatten_up_to(
        shallow_tree=mt, input_tree=simple_list, check_types=False
    )
    self.assertEqual(flattened_list, [2])

    nested_list = [[2]]
    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([3])
    )
    flattened_nmt = nest.flatten_up_to(
        shallow_tree=nested_list, input_tree=nmt, check_types=False
    )
    # Expected flattened_nmt = [Tensor([3])]
    self.assertAllEqual(flattened_nmt[0], [3])

    flat_path_nested_list = nest.flatten_up_to(
        shallow_tree=nmt, input_tree=nested_list, check_types=False
    )
    self.assertAllEqual(flat_path_nested_list, [2])

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassFlattenUpToIncompatible(self):
    # [FUNCTION DOC]
    # Test: flatten-up-to rejects incompatible dataclass/list combinations.
    # With type checking on, a list against a dataclass (either direction) must raise a type
    # error with the exact mismatching-types message. With type checking off, a shallow tree that
    # is deeper than the input must raise a type error saying that a sequence shallow structure
    # requires a sequence input.
    simple_list = [2]
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))

    # When `check_types=True` is set, `flatten_up_to` would fail when input_tree
    # and shallow_tree args don't have the same type
    with self.assertRaisesWithLiteralMatch(  # pylint: disable=g-error-prone-assert-raises
        TypeError,
        nest.STRUCTURES_HAVE_MISMATCHING_TYPES.format(
            shallow_type=type(simple_list), input_type=type(mt)
        ),
    ):
      nest.flatten_up_to(
          shallow_tree=simple_list, input_tree=mt, check_types=True
      )

    with self.assertRaisesWithLiteralMatch(  # pylint: disable=g-error-prone-assert-raises
        TypeError,
        nest.STRUCTURES_HAVE_MISMATCHING_TYPES.format(
            shallow_type=type(mt), input_type=type(simple_list)
        ),
    ):
      nest.flatten_up_to(
          shallow_tree=mt, input_tree=simple_list, check_types=True
      )

    nested_list = [[1]]
    # Although `check_types=False` is set, this assertion would fail because the
    # shallow_tree component has a deeper structure than the input_tree
    # component.
    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        TypeError,
        "If shallow structure is a sequence, input must also be a sequence",
    ):
      nest.flatten_up_to(
          shallow_tree=nested_list, input_tree=mt, check_types=False
      )

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassFlattenWithTuplePathsUpToCompatible(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: a comment in the body says the expected nested-dataclass path is the one-step
    # path, but the assertion checks the two-step path (0, 0); the comment is stale.
    # Test: flatten-with-tuple-paths-up-to between dataclasses and lists, type checking off.
    # Asserts that paths and values are correct for a list shallow tree over a dataclass (path
    # of one step), a dataclass shallow tree over a list, and the nested-dataclass variants
    # (two-step path with the inner tensor value).
    simple_list = [2]
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    flat_path_mt = nest.flatten_with_tuple_paths_up_to(
        shallow_tree=simple_list, input_tree=mt, check_types=False
    )
    # Expected flat_path_mt = [((0,), Tensor([1]))]
    self.assertEqual(flat_path_mt[0][0], (0,))
    self.assertAllEqual(flat_path_mt[0][1], [1])

    flat_path_list = nest.flatten_with_tuple_paths_up_to(
        shallow_tree=mt, input_tree=simple_list, check_types=False
    )
    self.assertAllEqual(flat_path_list, [[(0,), 2]])

    nested_list = [[2]]
    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([3])
    )
    flat_path_nmt = nest.flatten_with_tuple_paths_up_to(
        shallow_tree=nested_list, input_tree=nmt, check_types=False
    )
    # Expected flat_path_nmt = [((0,), Tensor([3]))]
    self.assertAllEqual(flat_path_nmt[0][0], [0, 0])
    self.assertAllEqual(flat_path_nmt[0][1], [3])

    flat_path_nested_list = nest.flatten_with_tuple_paths_up_to(
        shallow_tree=nmt, input_tree=nested_list, check_types=False
    )
    self.assertAllEqual(flat_path_nested_list, [[(0, 0), 2]])

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassFlattenWithTuplePathsUpToIncompatible(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the last scenario calls the plain flatten-up-to function, not the
    # tuple-paths variant this test is named for, so the path variant is not exercised there.
    # Test: with type checking on, mixing a list and a dataclass (either direction) must raise a
    # type error with the exact mismatching-types message. Finally, with type checking off, a
    # shallow tree deeper than the input must raise a type error about sequence inputs.
    simple_list = [2]
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    with self.assertRaisesWithLiteralMatch(  # pylint: disable=g-error-prone-assert-raises
        TypeError,
        nest.STRUCTURES_HAVE_MISMATCHING_TYPES.format(
            shallow_type=type(simple_list), input_type=type(mt)
        ),
    ):
      nest.flatten_with_tuple_paths_up_to(
          shallow_tree=simple_list, input_tree=mt, check_types=True
      )

    with self.assertRaisesWithLiteralMatch(  # pylint: disable=g-error-prone-assert-raises
        TypeError,
        nest.STRUCTURES_HAVE_MISMATCHING_TYPES.format(
            shallow_type=type(mt), input_type=type(simple_list)
        ),
    ):
      nest.flatten_with_tuple_paths_up_to(
          shallow_tree=mt, input_tree=simple_list, check_types=True
      )

    nested_list2 = [[[2]]]
    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([3])
    )

    # Although `check_types=False` is set, this assertion would fail because the
    # shallow_tree component has a deeper structure than the input_tree
    # component.
    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        TypeError,
        "If shallow structure is a sequence, input must also be a sequence",
    ):
      nest.flatten_up_to(
          shallow_tree=nested_list2, input_tree=nmt, check_types=False
      )

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassFlattenAndPack(self):
    # [FUNCTION DOC]
    # Test: round trip. Flatten a MaskedTensor, then pack the leaves back using the original as
    # template. Asserts the result is a MaskedTensor equal to the original.
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    leaves = nest.flatten(mt)
    reconstructed_mt = nest.pack_sequence_as(mt, leaves)
    self.assertIsInstance(reconstructed_mt, MaskedTensor)
    self.assertEqual(reconstructed_mt, mt)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassMapStructure(self):
    # [FUNCTION DOC]
    # Test: map-structure with a doubling function over a MaskedTensor. Asserts the result is a
    # MaskedTensor with the unchanged mask and a doubled tensor value.
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    mt_doubled = nest.map_structure(lambda x: x * 2, mt)
    self.assertIsInstance(mt_doubled, MaskedTensor)
    self.assertEqual(mt_doubled.mask, True)
    self.assertAllEqual(mt_doubled.value, [2])

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassMapStructureWithPaths(self):
    # [FUNCTION DOC]
    # Test: map-structure-with-paths over three MaskedTensors and over two nested dataclasses,
    # using a local function that returns the path and the sum of the leaves. Asserts that the
    # metadata comes from the first input, the string path is "0" (or "0/0" when nested) and
    # the leaf sums are correct.
    mt = MaskedTensor(mask=False, value=constant_op.constant([1]))
    mt2 = MaskedTensor(mask=True, value=constant_op.constant([2]))
    mt3 = MaskedTensor(mask=True, value=constant_op.constant([3]))

    def path_sum(path, *tensors):
      # [FUNCTION DOC]
      # HELPER: local mapping function of the string-path dataclass test.
      # Inputs: a path string and any number of tensors. Returns a pair of the path and the sum.
      return (path, sum(tensors))

    mt_combined_with_path = nest.map_structure_with_paths(
        path_sum, mt, mt2, mt3
    )
    self.assertIsInstance(mt_combined_with_path, MaskedTensor)
    # metadata uses the one from the first input (mt).
    self.assertEqual(mt_combined_with_path.mask, False)
    # Tesnor index is '0' for the only compoenent in MaskedTensor.
    self.assertAllEqual(mt_combined_with_path.value[0], "0")
    # sum of all input tensors.
    self.assertAllEqual(mt_combined_with_path.value[1], [6])

    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([4])
    )
    nmt2 = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=False, inner_value=constant_op.constant([5])
    )
    nmt_combined_with_path = nest.map_structure_with_paths(path_sum, nmt, nmt2)
    self.assertIsInstance(nmt_combined_with_path, NestedMaskedTensor)
    self.assertEqual(nmt_combined_with_path.mask, True)
    self.assertEqual(nmt_combined_with_path.value.mask, False)
    self.assertAllEqual(nmt_combined_with_path.value.value[0], "0/0")
    self.assertAllEqual(nmt_combined_with_path.value.value[1], [9])

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassMapStructureWithTuplePaths(self):
    # [FUNCTION DOC]
    # Test: same as the string-path variant but using tuple paths. Maps a local function that
    # returns the tuple path and the sum of leaves over three MaskedTensors and two nested
    # dataclasses. Asserts metadata from the first input, paths (0,) and (0, 0), and sums.
    mt = MaskedTensor(mask=False, value=constant_op.constant([1]))
    mt2 = MaskedTensor(mask=True, value=constant_op.constant([2]))
    mt3 = MaskedTensor(mask=True, value=constant_op.constant([3]))

    def tuple_path_sum(tuple_path, *tensors):
      # [FUNCTION DOC]
      # HELPER: local mapping function of this test.
      # Inputs: a tuple path and any number of tensors. Returns a pair of the path and the sum.
      return (tuple_path, sum(tensors))

    mt_combined_with_path = nest.map_structure_with_tuple_paths(
        tuple_path_sum, mt, mt2, mt3
    )
    self.assertIsInstance(mt_combined_with_path, MaskedTensor)
    # metadata uses the one from the first input (mt).
    self.assertEqual(mt_combined_with_path.mask, False)
    # Tesnor index is 0 for the only compoenent in MaskedTensor.
    self.assertAllEqual(mt_combined_with_path.value[0], (0,))
    # sum of all input tensors.
    self.assertAllEqual(mt_combined_with_path.value[1], [6])

    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([4])
    )
    nmt2 = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=False, inner_value=constant_op.constant([5])
    )
    nmt_combined_with_path = nest.map_structure_with_tuple_paths(
        tuple_path_sum, nmt, nmt2
    )
    self.assertIsInstance(nmt_combined_with_path, NestedMaskedTensor)
    self.assertEqual(nmt_combined_with_path.mask, True)
    self.assertEqual(nmt_combined_with_path.value.mask, False)
    self.assertAllEqual(nmt_combined_with_path.value.value[0], (0, 0))
    self.assertAllEqual(nmt_combined_with_path.value.value[1], [9])

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassMapStructureUpTo(self):
    # [FUNCTION DOC]
    # Test: map-structure-up-to using a dataclass template as the shallow structure. A local
    # function sums its arguments. Asserts the result type, that the mask comes from the
    # template, and the sum of the input tensors, for both flat and nested dataclasses.
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    mt2 = MaskedTensor(mask=True, value=constant_op.constant([2]))
    mt3 = MaskedTensor(mask=True, value=constant_op.constant([3]))
    mt_out_template = MaskedTensor(mask=False, value=constant_op.constant([4]))

    def sum_tensors(*tensors):
      # [FUNCTION DOC]
      # HELPER: local mapping function of the map-up-to dataclass test.
      # Returns the sum of all tensors passed as arguments.
      return sum(tensors)

    mt_combined_with_path = nest.map_structure_up_to(
        mt_out_template, sum_tensors, mt, mt2, mt3
    )
    self.assertIsInstance(mt_combined_with_path, MaskedTensor)
    # metadata uses the one from the first arg (mt_out_template).
    self.assertEqual(mt_combined_with_path.mask, False)
    # sum of all input tensors.
    self.assertAllEqual(mt_combined_with_path.value, [6])

    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([4])
    )
    nmt2 = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([5])
    )
    nmt_out = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=False, inner_value=constant_op.constant([6])
    )
    nmt_combined_with_path = nest.map_structure_up_to(
        nmt_out, sum_tensors, nmt, nmt2
    )
    self.assertIsInstance(nmt_combined_with_path, NestedMaskedTensor)
    self.assertEqual(nmt_combined_with_path.mask, False)
    self.assertEqual(nmt_combined_with_path.value.mask, True)
    self.assertAllEqual(nmt_combined_with_path.value.value, [9])

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassMapStructureWithTuplePathsUoTo(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name contains a typo ("UoTo" instead of "UpTo").
    # Test: map-structure-with-tuple-paths-up-to using dataclass templates. A local function
    # returns the tuple path and the sum of the leaves. Asserts template metadata is used, the
    # paths are (0,) and (0, 0) for flat and nested cases, and the sums are correct.
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    mt2 = MaskedTensor(mask=True, value=constant_op.constant([2]))
    mt3 = MaskedTensor(mask=True, value=constant_op.constant([3]))
    mt_out_template = MaskedTensor(mask=False, value=constant_op.constant([4]))

    def tuple_path_sum(tuple_path, *tensors):
      # [FUNCTION DOC]
      # HELPER: local mapping function of this test.
      # Inputs: a tuple path and any number of tensors. Returns a pair of the path and the sum.
      return (tuple_path, sum(tensors))

    mt_combined_with_path = nest.map_structure_with_tuple_paths_up_to(
        mt_out_template, tuple_path_sum, mt, mt2, mt3
    )
    self.assertIsInstance(mt_combined_with_path, MaskedTensor)
    # metadata uses the one from the first arg (mt_out_template).
    self.assertEqual(mt_combined_with_path.mask, False)
    # Tesnor index is 0 for the only compoenent in MaskedTensor.
    self.assertAllEqual(mt_combined_with_path.value[0], (0,))
    # sum of all input tensors.
    self.assertAllEqual(mt_combined_with_path.value[1], [6])

    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([4])
    )
    nmt2 = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([5])
    )
    nmt_out = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=False, inner_value=constant_op.constant([6])
    )
    nmt_combined_with_path = nest.map_structure_with_tuple_paths_up_to(
        nmt_out, tuple_path_sum, nmt, nmt2
    )
    self.assertIsInstance(nmt_combined_with_path, NestedMaskedTensor)
    self.assertEqual(nmt_combined_with_path.mask, False)
    self.assertEqual(nmt_combined_with_path.value.mask, True)
    self.assertAllEqual(nmt_combined_with_path.value.value[0], (0, 0))
    self.assertAllEqual(nmt_combined_with_path.value.value[1], [9])

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testNestedDataclassIsNested(self):
    # [FUNCTION DOC]
    # Test: both a MaskedTensor and a nested dataclass are reported as nested by is-nested.
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    self.assertTrue(nest.is_nested(mt))

    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([1])
    )
    self.assertTrue(nest.is_nested(nmt))

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassAssertShallowStructure(self):
    # [FUNCTION DOC]
    # Test: assert-shallow-structure with dataclasses. Two dataclasses with the same component
    # count pass in both directions, including a subclass when type checking is off. Mixing a
    # MaskedTensor with a nested dataclass raises a mismatching-types type error with type
    # checking on, passes with it off in the shallow-to-deep direction, and raises a type error
    # about sequences in the deep-to-shallow direction.
    # These assertions are expected to pass: two dataclasses with the same
    # component size are considered to have the same shallow structure.
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    mt2 = MaskedTensor(mask=False, value=constant_op.constant([2, 3]))
    nest.assert_shallow_structure(
        shallow_tree=mt, input_tree=mt2, check_types=True
    )
    nest.assert_shallow_structure(
        shallow_tree=mt2, input_tree=mt, check_types=True
    )

    mt3 = MaskedTensor2(mask=True, value=constant_op.constant([1]))
    # These assertions are expected to pass: two dataclasses with the same
    # component size are considered to have the same shallow structure.
    nest.assert_shallow_structure(
        shallow_tree=mt, input_tree=mt3, check_types=False
    )
    nest.assert_shallow_structure(
        shallow_tree=mt3, input_tree=mt, check_types=False
    )

    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([1])
    )

    # This assertion is expected to fail, when `check_types=True`, because the
    # shallow_tree type is not the same as input_tree.
    with self.assertRaisesWithLiteralMatch(  # pylint: disable=g-error-prone-assert-raises
        TypeError,
        nest.STRUCTURES_HAVE_MISMATCHING_TYPES.format(
            shallow_type=type(mt), input_type=type(nmt)
        ),
    ):
      nest.assert_shallow_structure(
          shallow_tree=mt, input_tree=nmt, check_types=True
      )

    # This assertion is expected to pass: the shallow_tree component contains
    # the shallow structure of the input_tree component.
    nest.assert_shallow_structure(
        shallow_tree=mt, input_tree=nmt, check_types=False
    )

    # This assertion is expected to fail: the shallow_tree component has
    # a deeper structure than the input_tree component.
    with self.assertRaisesRegex(  # pylint: disable=g-error-prone-assert-raises
        TypeError,
        "If shallow structure is a sequence, input must also be a sequence",
    ):
      nest.assert_shallow_structure(
          shallow_tree=nmt, input_tree=mt, check_types=False
      )

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassGetTraverseShallowStructure(self):
    # [FUNCTION DOC]
    # Test: get-traverse-shallow-structure over a nested dataclass with three traversal
    # functions. One accepting both dataclass types gives a nested dataclass whose inner value
    # is false; one accepting anything not a list gives an inner value of true; one accepting
    # only tensors gives false for the whole structure. Each result is also checked to be a
    # valid shallow structure of the input.
    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([1])
    )
    traverse_result = nest.get_traverse_shallow_structure(
        lambda s: isinstance(s, (NestedMaskedTensor, MaskedTensor)), nmt
    )
    self.assertIsInstance(traverse_result, NestedMaskedTensor)
    self.assertEqual(traverse_result.mask, nmt.mask)
    self.assertIsInstance(traverse_result.value, MaskedTensor)
    self.assertEqual(traverse_result.value.value, False)
    nest.assert_shallow_structure(traverse_result, nmt)

    traverse_result2 = nest.get_traverse_shallow_structure(
        lambda s: not isinstance(s, list), nmt
    )
    self.assertIsInstance(traverse_result2, NestedMaskedTensor)
    self.assertEqual(traverse_result2.mask, nmt.mask)
    self.assertIsInstance(traverse_result2.value, MaskedTensor)
    # Expected traverse_result2.value.value is True since it can pass the
    # traverse function, but there is no more flattening for the Tensor value.
    self.assertEqual(traverse_result2.value.value, True)
    nest.assert_shallow_structure(traverse_result2, nmt)

    traverse_result3 = nest.get_traverse_shallow_structure(
        lambda s: isinstance(s, tensor.Tensor), nmt
    )
    # Expected `traverse_result3 = False` because `nmt` doesn't pass the
    # traverse function.
    self.assertEqual(traverse_result3, False)
    nest.assert_shallow_structure(traverse_result3, nmt)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testNestedDataclassFlatten(self):
    # [FUNCTION DOC]
    # Test: flattening a nested dataclass yields one leaf, the innermost tensor.
    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([1])
    )
    leaves = nest.flatten(nmt)
    self.assertLen(leaves, 1)
    self.assertAllEqual(leaves[0], [1])

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testNestedDataclassFlattenAndPack(self):
    # [FUNCTION DOC]
    # Test: flatten then pack a nested dataclass; asserts the result has the nested dataclass
    # type and equals the original.
    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([1])
    )
    leaves = nest.flatten(nmt)
    reconstructed_mt = nest.pack_sequence_as(nmt, leaves)
    self.assertIsInstance(reconstructed_mt, NestedMaskedTensor)
    self.assertEqual(reconstructed_mt, nmt)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testNestedDataclassMapStructure(self):
    # [FUNCTION DOC]
    # Test: map-structure doubling over a nested dataclass. Asserts the result type, that both
    # masks are preserved, and that the inner tensor is doubled.
    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([1])
    )
    mt_doubled = nest.map_structure(lambda x: x * 2, nmt)
    expected = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([2])
    )

    self.assertIsInstance(mt_doubled, NestedMaskedTensor)
    self.assertEqual(mt_doubled.mask, expected.mask)
    self.assertEqual(mt_doubled.value.mask, expected.value.mask)
    self.assertAllEqual(mt_doubled.value.value, expected.value.value)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassYieldFlatPaths(self):
    # [FUNCTION DOC]
    # Test: yield-flat-paths for a dataclass (path 0), a nested dataclass (path 0, 0) and a
    # dictionary mixing both plus a list. Asserts the paths are produced in sorted key order.
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    mt_flat_paths = list(nest.yield_flat_paths(mt))
    self.assertEqual(mt_flat_paths, [(0,)])

    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([2])
    )
    nmt_flat_paths = list(nest.yield_flat_paths(nmt))
    self.assertEqual(nmt_flat_paths, [(0, 0)])

    dict_mt_nmt = {"mt": mt, "nmt": nmt, "mt_nmt_list": [mt, nmt]}
    dict_mt_nmt_flat_paths = list(nest.yield_flat_paths(dict_mt_nmt))
    self.assertEqual(
        dict_mt_nmt_flat_paths,
        [
            ("mt", 0),
            ("mt_nmt_list", 0, 0),
            ("mt_nmt_list", 1, 0, 0),
            ("nmt", 0, 0),
        ],
    )

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassFlattenWithStringPaths(self):
    # [FUNCTION DOC]
    # Test: flatten-with-joined-string-paths using a slash separator on dataclasses and a
    # dictionary of them. Asserts joined paths such as "0", "0/0", "mt/0" and "nmt/0/0" and the
    # associated leaf values.
    sep = "/"
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    mt_flat_paths = nest.flatten_with_joined_string_paths(mt, separator=sep)
    self.assertEqual(mt_flat_paths[0][0], "0")
    self.assertAllEqual(mt_flat_paths[0][1], [1])

    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([2])
    )
    nmt_flat_paths = nest.flatten_with_joined_string_paths(nmt, separator=sep)
    self.assertEqual(nmt_flat_paths[0][0], "0/0")
    self.assertAllEqual(nmt_flat_paths[0][1], [2])

    dict_mt_nmt = {"mt": mt, "nmt": nmt}
    dict_mt_nmt_flat_paths = nest.flatten_with_joined_string_paths(
        dict_mt_nmt, separator=sep
    )
    self.assertEqual(dict_mt_nmt_flat_paths[0][0], "mt/0")
    self.assertAllEqual(dict_mt_nmt_flat_paths[0][1], [1])
    self.assertEqual(dict_mt_nmt_flat_paths[1][0], "nmt/0/0")
    self.assertAllEqual(dict_mt_nmt_flat_paths[1][1], [2])

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassFlattenWithTuplePaths(self):
    # [FUNCTION DOC]
    # Test: flatten-with-tuple-paths on dataclasses and a dictionary of them. Asserts the tuple
    # paths, such as (0,) and ("nmt", 0, 0), and the leaf values.
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    mt_flat_paths = nest.flatten_with_tuple_paths(mt)
    self.assertEqual(mt_flat_paths[0][0], (0,))
    self.assertAllEqual(mt_flat_paths[0][1], [1])

    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([2])
    )
    nmt_flat_paths = nest.flatten_with_tuple_paths(nmt)
    self.assertEqual(nmt_flat_paths[0][0], (0, 0))
    self.assertAllEqual(nmt_flat_paths[0][1], [2])

    dict_mt_nmt = {"mt": mt, "nmt": nmt}
    dict_mt_nmt_flat_paths = nest.flatten_with_tuple_paths(dict_mt_nmt)
    self.assertEqual(dict_mt_nmt_flat_paths[0][0], ("mt", 0))
    self.assertAllEqual(dict_mt_nmt_flat_paths[0][1], [1])
    self.assertEqual(dict_mt_nmt_flat_paths[1][0], ("nmt", 0, 0))
    self.assertAllEqual(dict_mt_nmt_flat_paths[1][1], [2])

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testDataclassListToTuple(self):
    # [FUNCTION DOC]
    # Test: list-to-tuple converts lists nested inside a mixed structure that contains
    # dataclasses, dictionaries and None into tuples. Asserts the result has the same structure
    # as the expected all-tuple structure (dataclass leaves are left as they are).
    mt = MaskedTensor(mask=True, value=constant_op.constant([1]))
    nmt = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=True, inner_value=constant_op.constant([2])
    )
    input_sequence = [mt, (nmt, {"a": [mt, nmt, (mt,)]}, None, nmt, [[[mt]]])]

    mt2 = MaskedTensor(mask=True, value=constant_op.constant([3]))
    nmt2 = NestedMaskedTensor.nested_masked_tensor_with_opposite_masks(
        mask=False, inner_value=constant_op.constant([2])
    )
    results = nest.list_to_tuple(input_sequence)
    expected = (
        mt2,
        (nmt2, {"a": (mt2, nmt2, (mt2,))}, None, nmt2, (((mt2,),),)),
    )
    nest.assert_same_structure(results, expected)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testAttrsFlattenAndPack(self):
    # [FUNCTION DOC]
    # Test: attrs classes. Skipped when attrs is not installed. Asserts the attrs detection
    # helper is false for a list and true for an instance, flattening gives the field values in
    # order, packing restores an equal instance, and flattening an instance of the class with a
    # non-iterable attributes marker raises a type error.
    if attr is None:
      self.skipTest("attr module is unavailable.")

    field_values = [1, 2]
    sample_attr = NestTest.SampleAttr(*field_values)
    self.assertFalse(nest._is_attrs(field_values))
    self.assertTrue(nest._is_attrs(sample_attr))
    flat = nest.flatten(sample_attr)
    self.assertEqual(field_values, flat)
    restructured_from_flat = nest.pack_sequence_as(sample_attr, flat)
    self.assertIsInstance(restructured_from_flat, NestTest.SampleAttr)
    self.assertEqual(restructured_from_flat, sample_attr)

    # Check that flatten fails if attributes are not iterable
    with self.assertRaisesRegex(TypeError, "object is not iterable"):
      flat = nest.flatten(NestTest.BadAttr())

  @parameterized.parameters(
      {"values": [1, 2, 3]},
      {"values": [{"B": 10, "A": 20}, [1, 2], 3]},
      {"values": [(1, 2), [3, 4], 5]},
      {"values": [PointXY(1, 2), 3, 4]},
  )
  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testAttrsMapStructure(self, values):
    # [FUNCTION DOC]
    # Test (parameterized over several lists of field values, including dicts, nested lists and
    # namedtuples). Skipped when attrs is unavailable. Builds an attrs instance from the values,
    # maps an identity function over it and asserts the result equals the input.
    if attr is None:
      self.skipTest("attr module is unavailable.")

    structure = NestTest.UnsortedSampleAttr(*values)
    new_structure = nest.map_structure(lambda x: x, structure)
    self.assertEqual(structure, new_structure)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testFlattenAndPack(self):
    # [FUNCTION DOC]
    # Test: basic flatten and pack-sequence-as behavior. Covers nested tuples and namedtuples,
    # scalars and numpy arrays (flatten gives a one-item list), packing a scalar template,
    # a value-error for packing several values into a scalar, a type error for packing a string
    # into a sequence template, and a value-error for a length mismatch.
    structure = ((3, 4), 5, (6, 7, (9, 10), 8))
    flat = ["a", "b", "c", "d", "e", "f", "g", "h"]
    self.assertEqual(nest.flatten(structure), [3, 4, 5, 6, 7, 9, 10, 8])
    self.assertEqual(
        nest.pack_sequence_as(structure, flat), (("a", "b"), "c",
                                                 ("d", "e", ("f", "g"), "h")))
    structure = (NestTest.PointXY(x=4, y=2),
                 ((NestTest.PointXY(x=1, y=0),),))
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

    with self.assertRaisesRegex(ValueError, self.unsafe_map_pattern):
      nest.pack_sequence_as("scalar", [4, 5])

    with self.assertRaisesRegex(TypeError, self.bad_pack_pattern):
      nest.pack_sequence_as([4, 5], "bad_sequence")

    with self.assertRaises(ValueError):
      nest.pack_sequence_as([5, 6, [7, 8]], ["a", "b", "c"])

  @parameterized.parameters({"mapping_type": collections.OrderedDict},
                            {"mapping_type": _CustomMapping})
  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testFlattenDictOrder(self, mapping_type):
    # [FUNCTION DOC]
    # Test (parameterized over ordered dict and the custom mapping): flatten orders mapping
    # values by sorted key, giving the same result as for a plain dictionary with the same keys.
    """`flatten` orders dicts by key, including OrderedDicts."""
    ordered = mapping_type([("d", 3), ("b", 1), ("a", 0), ("c", 2)])
    plain = {"d": 3, "b": 1, "a": 0, "c": 2}
    ordered_flat = nest.flatten(ordered)
    plain_flat = nest.flatten(plain)
    self.assertEqual([0, 1, 2, 3], ordered_flat)
    self.assertEqual([0, 1, 2, 3], plain_flat)

  @parameterized.parameters({"mapping_type": collections.OrderedDict},
                            {"mapping_type": _CustomMapping})
  def testPackDictOrder(self, mapping_type):
    # [FUNCTION DOC]
    # Test (parameterized over ordered dict and the custom mapping): pack-sequence-as fills a
    # mapping template in sorted key order. Asserts the result for an ordered or custom mapping
    # keeps the template type and for a plain dictionary is a dictionary, with correct values.
    """Packing orders dicts by key, including OrderedDicts."""
    custom = mapping_type([("d", 0), ("b", 0), ("a", 0), ("c", 0)])
    plain = {"d": 0, "b": 0, "a": 0, "c": 0}
    seq = [0, 1, 2, 3]
    custom_reconstruction = nest.pack_sequence_as(custom, seq)
    plain_reconstruction = nest.pack_sequence_as(plain, seq)
    self.assertIsInstance(custom_reconstruction, mapping_type)
    self.assertIsInstance(plain_reconstruction, dict)
    self.assertEqual(
        mapping_type([("d", 3), ("b", 1), ("a", 0), ("c", 2)]),
        custom_reconstruction)
    self.assertEqual({"d": 3, "b": 1, "a": 0, "c": 2}, plain_reconstruction)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testFlattenAndPackMappingViews(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the existing docstring claims flatten orders dicts by key, but the assertions
    # show the opposite for mapping views of an ordered dictionary: insertion order is kept.
    # (The text appears copied from the dictionary-order test.)
    # Test: flattening the keys, values and items views of an ordered dictionary yields them in
    # insertion order, and packing with the items view as template restores the pairs.
    """`flatten` orders dicts by key, including OrderedDicts."""
    ordered = collections.OrderedDict([("d", 3), ("b", 1), ("a", 0), ("c", 2)])

    # test flattening
    ordered_keys_flat = nest.flatten(ordered.keys())
    ordered_values_flat = nest.flatten(ordered.values())
    ordered_items_flat = nest.flatten(ordered.items())
    self.assertEqual([3, 1, 0, 2], ordered_values_flat)
    self.assertEqual(["d", "b", "a", "c"], ordered_keys_flat)
    self.assertEqual(["d", 3, "b", 1, "a", 0, "c", 2], ordered_items_flat)

    # test packing
    self.assertEqual([("d", 3), ("b", 1), ("a", 0), ("c", 2)],
                     nest.pack_sequence_as(ordered.items(), ordered_items_flat))

  Abc = collections.namedtuple("A", ("b", "c"))  # pylint: disable=invalid-name

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testFlattenAndPack_withDicts(self):
    # [FUNCTION DOC]
    # Test: a mixed structure of strings, namedtuples, dictionaries, custom mappings and ordered
    # dictionaries is flattened in sorted-key order, then packed using a different structure as
    # template. Asserts the leaf order, equality with the original, and that the ordered
    # dictionary and custom mapping types and key order are preserved.
    # A nice messy mix of tuples, lists, dicts, and `OrderedDict`s.
    mess = [
        "z",
        NestTest.Abc(3, 4), {
            "d": _CustomMapping({
                41: 4
            }),
            "c": [
                1,
                collections.OrderedDict([
                    ("b", 3),
                    ("a", 2),
                ]),
            ],
            "b": 5
        }, 17
    ]

    flattened = nest.flatten(mess)
    self.assertEqual(flattened, ["z", 3, 4, 5, 1, 2, 3, 4, 17])

    structure_of_mess = [
        14,
        NestTest.Abc("a", True),
        {
            "d": _CustomMapping({
                41: 42
            }),
            "c": [
                0,
                collections.OrderedDict([
                    ("b", 9),
                    ("a", 8),
                ]),
            ],
            "b": 3
        },
        "hi everybody",
    ]

    unflattened = nest.pack_sequence_as(structure_of_mess, flattened)
    self.assertEqual(unflattened, mess)

    # Check also that the OrderedDict was created, with the correct key order.
    unflattened_ordered_dict = unflattened[2]["c"][1]
    self.assertIsInstance(unflattened_ordered_dict, collections.OrderedDict)
    self.assertEqual(list(unflattened_ordered_dict.keys()), ["b", "a"])

    unflattened_custom_mapping = unflattened[2]["d"]
    self.assertIsInstance(unflattened_custom_mapping, _CustomMapping)
    self.assertEqual(list(unflattened_custom_mapping.keys()), [41])

  def testFlatten_numpyIsNotFlattened(self):
    # [FUNCTION DOC]
    # Test: a numpy array is treated as a single leaf; flatten returns a one-element list.
    structure = np.array([1, 2, 3])
    flattened = nest.flatten(structure)
    self.assertLen(flattened, 1)

  def testFlatten_stringIsNotFlattened(self):
    # [FUNCTION DOC]
    # Test: a string is a single leaf. Flatten gives one item, and packing it into a string
    # template gives back the original string.
    structure = "lots of letters"
    flattened = nest.flatten(structure)
    self.assertLen(flattened, 1)
    unflattened = nest.pack_sequence_as("goodbye", flattened)
    self.assertEqual(structure, unflattened)

  def testPackSequenceAs_notIterableError(self):
    # [FUNCTION DOC]
    # Test: packing a string as the flat sequence into a string template raises a type error
    # matching the bad-pack pattern.
    with self.assertRaisesRegex(TypeError, self.bad_pack_pattern):
      nest.pack_sequence_as("hi", "bye")

  def testPackSequenceAs_wrongLengthsError(self):
    # [FUNCTION DOC]
    # Test: packing three items into a template with two leaves raises a value error stating
    # that the structure had 2 atoms but the flat sequence had 3 items.
    with self.assertRaisesRegex(
        ValueError, "Structure had 2 atoms, but flat_sequence had 3 items."):
      nest.pack_sequence_as(["hello", "world"],
                            ["and", "goodbye", "again"])

  def testPackSequenceAs_CompositeTensor(self):
    # [FUNCTION DOC]
    # Test: with composite expansion enabled, a ragged tensor template has two leaves, so packing
    # a single-item flat sequence raises a value error (2 atoms versus 1 item).
    val = ragged_tensor.RaggedTensor.from_row_splits(values=[1],
                                                     row_splits=[0, 1])
    with self.assertRaisesRegex(
        ValueError, "Structure had 2 atoms, but flat_sequence had 1 items."):
      nest.pack_sequence_as(val, [val], expand_composites=True)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testIsNested(self):
    # [FUNCTION DOC]
    # Test: is-nested is false for strings, sets, tensors and numpy arrays, and true for lists,
    # tuples (also empty), dictionaries and dictionary key, value and item views.
    self.assertFalse(nest.is_nested("1234"))
    self.assertTrue(nest.is_nested([1, 3, [4, 5]]))
    self.assertTrue(nest.is_nested(((7, 8), (5, 6))))
    self.assertTrue(nest.is_nested([]))
    self.assertTrue(nest.is_nested({"a": 1, "b": 2}))
    self.assertTrue(nest.is_nested({"a": 1, "b": 2}.keys()))
    self.assertTrue(nest.is_nested({"a": 1, "b": 2}.values()))
    self.assertTrue(nest.is_nested({"a": 1, "b": 2}.items()))
    self.assertFalse(nest.is_nested(set([1, 2])))
    ones = array_ops.ones([2, 3])
    self.assertFalse(nest.is_nested(ones))
    self.assertFalse(nest.is_nested(math_ops.tanh(ones)))
    self.assertFalse(nest.is_nested(np.ones((4, 5))))

  @parameterized.parameters({"mapping_type": _CustomMapping},
                            {"mapping_type": dict})
  def testFlattenDictItems(self, mapping_type):
    # [FUNCTION DOC]
    # Test (parameterized over the custom mapping and dict): flatten-dict-items expands
    # dictionary entries whose keys and values are matching nested tuples into a flat
    # dictionary. Also asserts a type error for a non-dictionary, a value error for duplicate
    # flattened keys, and a value error when key and value structures have different sizes.
    dictionary = mapping_type({(4, 5, (6, 8)): ("a", "b", ("c", "d"))})
    flat = {4: "a", 5: "b", 6: "c", 8: "d"}
    self.assertEqual(nest.flatten_dict_items(dictionary), flat)

    with self.assertRaises(TypeError):
      nest.flatten_dict_items(4)

    bad_dictionary = mapping_type({(4, 5, (4, 8)): ("a", "b", ("c", "d"))})
    with self.assertRaisesRegex(ValueError, "not unique"):
      nest.flatten_dict_items(bad_dictionary)

    another_bad_dictionary = mapping_type({
        (4, 5, (6, 8)): ("a", "b", ("c", ("d", "e")))
    })
    with self.assertRaisesRegex(
        ValueError, "Key had [0-9]* elements, but value had [0-9]* elements"):
      nest.flatten_dict_items(another_bad_dictionary)

  # pylint does not correctly recognize these as class names and
  # suggests to use variable style under_score naming.
  # pylint: disable=invalid-name
  Named0ab = collections.namedtuple("named_0", ("a", "b"))
  Named1ab = collections.namedtuple("named_1", ("a", "b"))
  SameNameab = collections.namedtuple("same_name", ("a", "b"))
  SameNameab2 = collections.namedtuple("same_name", ("a", "b"))
  SameNamexy = collections.namedtuple("same_name", ("x", "y"))
  SameName1xy = collections.namedtuple("same_name_1", ("x", "y"))
  SameName1xy2 = collections.namedtuple("same_name_1", ("x", "y"))
  NotSameName = collections.namedtuple("not_same_name", ("a", "b"))
  # pylint: enable=invalid-name

  class SameNamedType1(SameNameab):
    # [CLASS DOC]
    # HELPER: test fixture for NestTest.
    # Empty subclass of the same-name namedtuple type; used to check that a subclass of a
    # namedtuple is not considered the same structure type as its parent.
    pass

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testAssertSameStructure(self):
    # [FUNCTION DOC]
    # Test: assert-same-structure. Valid pairs pass (including scalars of different types).
    # Raises a value error with a detailed message for differing sizes, sequence versus scalar
    # and nesting differences; a type error for different sequence types or different
    # namedtuple names/fields/subclasses; passes when type checking is off; raises a value
    # error for different dictionary keys; and treats namedtuples of equal name and fields as
    # the same.
    structure1 = (((1, 2), 3), 4, (5, 6))
    structure2 = ((("foo1", "foo2"), "foo3"), "foo4", ("foo5", "foo6"))
    structure_different_num_elements = ("spam", "eggs")
    structure_different_nesting = (((1, 2), 3), 4, 5, (6,))
    nest.assert_same_structure(structure1, structure2)
    nest.assert_same_structure("abc", 1.0)
    nest.assert_same_structure("abc", np.array([0, 1]))
    nest.assert_same_structure("abc", constant_op.constant([0, 1]))

    with self.assertRaisesRegex(
        ValueError,
        ("The two structures don't have the same nested structure\\.\n\n"
         "First structure:.*?\n\n"
         "Second structure:.*\n\n"
         "More specifically: Substructure "
         r'"type=tuple str=\(\(1, 2\), 3\)" is a sequence, while '
         'substructure "type=str str=spam" is not\n'
         "Entire first structure:\n"
         r"\(\(\(\., \.\), \.\), \., \(\., \.\)\)\n"
         "Entire second structure:\n"
         r"\(\., \.\)")):
      nest.assert_same_structure(structure1, structure_different_num_elements)

    with self.assertRaisesRegex(
        ValueError,
        ("The two structures don't have the same nested structure\\.\n\n"
         "First structure:.*?\n\n"
         "Second structure:.*\n\n"
         r'More specifically: Substructure "type=list str=\[0, 1\]" '
         r'is a sequence, while substructure "type=ndarray str=\[0 1\]" '
         "is not")):
      nest.assert_same_structure([0, 1], np.array([0, 1]))

    with self.assertRaisesRegex(
        ValueError,
        ("The two structures don't have the same nested structure\\.\n\n"
         "First structure:.*?\n\n"
         "Second structure:.*\n\n"
         r'More specifically: Substructure "type=list str=\[0, 1\]" '
         'is a sequence, while substructure "type=int str=0" '
         "is not")):
      nest.assert_same_structure(0, [0, 1])

    self.assertRaises(TypeError, nest.assert_same_structure, (0, 1), [0, 1])

    with self.assertRaisesRegex(ValueError,
                                ("don't have the same nested structure\\.\n\n"
                                 "First structure: .*?\n\nSecond structure: ")):
      nest.assert_same_structure(structure1, structure_different_nesting)

    self.assertRaises(TypeError, nest.assert_same_structure, (0, 1),
                      NestTest.Named0ab("a", "b"))

    nest.assert_same_structure(NestTest.Named0ab(3, 4),
                               NestTest.Named0ab("a", "b"))

    self.assertRaises(TypeError, nest.assert_same_structure,
                      NestTest.Named0ab(3, 4), NestTest.Named1ab(3, 4))

    with self.assertRaisesRegex(ValueError,
                                ("don't have the same nested structure\\.\n\n"
                                 "First structure: .*?\n\nSecond structure: ")):
      nest.assert_same_structure(NestTest.Named0ab(3, 4),
                                 NestTest.Named0ab([3], 4))

    with self.assertRaisesRegex(ValueError,
                                ("don't have the same nested structure\\.\n\n"
                                 "First structure: .*?\n\nSecond structure: ")):
      nest.assert_same_structure([[3], 4], [3, [4]])

    structure1_list = [[[1, 2], 3], 4, [5, 6]]
    with self.assertRaisesRegex(TypeError, "don't have the same sequence type"):
      nest.assert_same_structure(structure1, structure1_list)
    nest.assert_same_structure(structure1, structure2, check_types=False)
    nest.assert_same_structure(structure1, structure1_list, check_types=False)

    with self.assertRaisesRegex(ValueError, "don't have the same set of keys"):
      nest.assert_same_structure({"a": 1}, {"b": 1})

    nest.assert_same_structure(NestTest.SameNameab(0, 1),
                               NestTest.SameNameab2(2, 3))

    # This assertion is expected to pass: two namedtuples with the same
    # name and field names are considered to be identical.
    nest.assert_same_structure(
        NestTest.SameNameab(NestTest.SameName1xy(0, 1), 2),
        NestTest.SameNameab2(NestTest.SameName1xy2(2, 3), 4))

    expected_message = "The two structures don't have the same.*"
    with self.assertRaisesRegex(ValueError, expected_message):
      nest.assert_same_structure(
          NestTest.SameNameab(0, NestTest.SameNameab2(1, 2)),
          NestTest.SameNameab2(NestTest.SameNameab(0, 1), 2))

    self.assertRaises(TypeError, nest.assert_same_structure,
                      NestTest.SameNameab(0, 1), NestTest.NotSameName(2, 3))

    self.assertRaises(TypeError, nest.assert_same_structure,
                      NestTest.SameNameab(0, 1), NestTest.SameNamexy(2, 3))

    self.assertRaises(TypeError, nest.assert_same_structure,
                      NestTest.SameNameab(0, 1), NestTest.SameNamedType1(2, 3))

  EmptyNT = collections.namedtuple("empty_nt", "")  # pylint: disable=invalid-name

  def testHeterogeneousComparison(self):
    # [FUNCTION DOC]
    # Test: a plain dictionary and a custom mapping with the same keys have the same structure
    # in both argument orders; no error is raised.
    nest.assert_same_structure({"a": 4}, _CustomMapping(a=3))
    nest.assert_same_structure(_CustomMapping(b=3), {"b": 4})

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testMapStructure(self):
    # [FUNCTION DOC]
    # Test: map-structure over nested tuples with one and two structures, scalars, a default
    # dictionary and empty structures (type preserved). Asserts errors: type error for a
    # non-callable function, value error without structures, mismatching lengths or nesting,
    # type error for different sequence types unless type checking is off, and a value error
    # for an unknown keyword argument.
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

    structure3 = collections.defaultdict(list)
    structure3["a"] = [1, 2, 3, 4]
    structure3["b"] = [2, 3, 4, 5]

    expected_structure3 = collections.defaultdict(list)
    expected_structure3["a"] = [2, 3, 4, 5]
    expected_structure3["b"] = [3, 4, 5, 6]
    self.assertEqual(expected_structure3,
                     nest.map_structure(lambda x: x + 1, structure3))

    # Empty structures
    self.assertEqual((), nest.map_structure(lambda x: x + 1, ()))
    self.assertEqual([], nest.map_structure(lambda x: x + 1, []))
    self.assertEqual({}, nest.map_structure(lambda x: x + 1, {}))
    self.assertEqual(NestTest.EmptyNT(), nest.map_structure(lambda x: x + 1,
                                                            NestTest.EmptyNT()))

    # This is checking actual equality of types, empty list != empty tuple
    self.assertNotEqual((), nest.map_structure(lambda x: x + 1, []))

    with self.assertRaisesRegex(TypeError, "callable"):
      nest.map_structure("bad", structure1_plus1)

    with self.assertRaisesRegex(ValueError, "at least one structure"):
      nest.map_structure(lambda x: x)

    with self.assertRaisesRegex(ValueError, "same number of elements"):
      nest.map_structure(lambda x, y: None, (3, 4), (3, 4, 5))

    with self.assertRaisesRegex(ValueError, "same nested structure"):
      nest.map_structure(lambda x, y: None, 3, (3,))

    with self.assertRaisesRegex(TypeError, "same sequence type"):
      nest.map_structure(lambda x, y: None, ((3, 4), 5), [(3, 4), 5])

    with self.assertRaisesRegex(ValueError, "same nested structure"):
      nest.map_structure(lambda x, y: None, ((3, 4), 5), (3, (4, 5)))

    structure1_list = [[[1, 2], 3], 4, [5, 6]]
    with self.assertRaisesRegex(TypeError, "same sequence type"):
      nest.map_structure(lambda x, y: None, structure1, structure1_list)

    nest.map_structure(lambda x, y: None, structure1, structure1_list,
                       check_types=False)

    with self.assertRaisesRegex(ValueError, "same nested structure"):
      nest.map_structure(lambda x, y: None, ((3, 4), 5), (3, (4, 5)),
                         check_types=False)

    with self.assertRaisesRegex(ValueError, "Only valid keyword argument.*foo"):
      nest.map_structure(lambda x: None, structure1, foo="a")

    with self.assertRaisesRegex(ValueError, "Only valid keyword argument.*foo"):
      nest.map_structure(lambda x: None, structure1, check_types=False, foo="a")

  ABTuple = collections.namedtuple("ab_tuple", "a, b")  # pylint: disable=invalid-name

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testMapStructureWithStrings(self):
    # [FUNCTION DOC]
    # Test: map-structure keeps strings as leaves. Repeats strings by counts held in a second
    # namedtuple structure, then reverses every string of a namedtuple. Asserts values and that
    # the result has the same structure as the input.
    inp_a = NestTest.ABTuple(a="foo", b=("bar", "baz"))
    inp_b = NestTest.ABTuple(a=2, b=(1, 3))
    out = nest.map_structure(lambda string, repeats: string * repeats,
                             inp_a,
                             inp_b)
    self.assertEqual("foofoo", out.a)
    self.assertEqual("bar", out.b[0])
    self.assertEqual("bazbazbaz", out.b[1])

    nt = NestTest.ABTuple(a=("something", "something_else"),
                          b="yet another thing")
    rev_nt = nest.map_structure(lambda x: x[::-1], nt)
    # Check the output is the correct structure, and all strings are reversed.
    nest.assert_same_structure(nt, rev_nt)
    self.assertEqual(nt.a[0][::-1], rev_nt.a[0])
    self.assertEqual(nt.a[1][::-1], rev_nt.a[1])
    self.assertEqual(nt.b[::-1], rev_nt.b)

  def testMapStructureOverPlaceholders(self):
    # [FUNCTION DOC]
    # Test (graph mode only, uses a new graph): map-structure adds two tuples of placeholders.
    # Asserts the output has the same structure and shapes, then runs a session with random
    # fed values and checks the sums.
    # Test requires placeholders and thus requires graph mode
    with ops.Graph().as_default():
      inp_a = (array_ops.placeholder(dtypes.float32, shape=[3, 4]),
               array_ops.placeholder(dtypes.float32, shape=[3, 7]))
      inp_b = (array_ops.placeholder(dtypes.float32, shape=[3, 4]),
               array_ops.placeholder(dtypes.float32, shape=[3, 7]))

      output = nest.map_structure(lambda x1, x2: x1 + x2, inp_a, inp_b)

      nest.assert_same_structure(output, inp_a)
      self.assertShapeEqual(np.zeros((3, 4)), output[0])
      self.assertShapeEqual(np.zeros((3, 7)), output[1])

      feed_dict = {
          inp_a: (np.random.randn(3, 4), np.random.randn(3, 7)),
          inp_b: (np.random.randn(3, 4), np.random.randn(3, 7))
      }

      with self.cached_session() as sess:
        output_np = sess.run(output, feed_dict=feed_dict)
      self.assertAllClose(output_np[0],
                          feed_dict[inp_a][0] + feed_dict[inp_b][0])
      self.assertAllClose(output_np[1],
                          feed_dict[inp_a][1] + feed_dict[inp_b][1])

  def testAssertShallowStructure(self):
    # [FUNCTION DOC]
    # Test: assert-shallow-structure. Raises a value error for longer shallow tree, type error
    # for mismatched sequence types (unless type checks are off), a value error for invalid
    # dictionary keys, and passes for ordered dictionaries in different order, equal-shape
    # namedtuples, list subclasses and a variable spec versus a variable (with or without
    # expanding composites).
    inp_ab = ["a", "b"]
    inp_abc = ["a", "b", "c"]
    with self.assertRaisesWithLiteralMatch(  # pylint: disable=g-error-prone-assert-raises
        ValueError,
        nest.STRUCTURES_HAVE_MISMATCHING_LENGTHS.format(
            input_length=len(inp_ab), shallow_length=len(inp_abc)
        ),
    ):
      nest.assert_shallow_structure(inp_abc, inp_ab)

    inp_ab1 = [(1, 1), (2, 2)]
    inp_ab2 = [[1, 1], [2, 2]]
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        nest.STRUCTURES_HAVE_MISMATCHING_TYPES.format(
            shallow_type=type(inp_ab2[0]), input_type=type(inp_ab1[0])
        ),
    ):
      nest.assert_shallow_structure(inp_ab2, inp_ab1)
    nest.assert_shallow_structure(inp_ab2, inp_ab1, check_types=False)

    inp_ab1 = {"a": (1, 1), "b": {"c": (2, 2)}}
    inp_ab2 = {"a": (1, 1), "b": {"d": (2, 2)}}
    with self.assertRaisesWithLiteralMatch(
        ValueError, nest.SHALLOW_TREE_HAS_INVALID_KEYS.format(["d"])
    ):
      nest.assert_shallow_structure(inp_ab2, inp_ab1)

    inp_ab = collections.OrderedDict([("a", 1), ("b", (2, 3))])
    inp_ba = collections.OrderedDict([("b", (2, 3)), ("a", 1)])
    nest.assert_shallow_structure(inp_ab, inp_ba)

    # This assertion is expected to pass: two namedtuples with the same
    # name and field names are considered to be identical.
    inp_shallow = NestTest.SameNameab(1, 2)
    inp_deep = NestTest.SameNameab2(1, [1, 2, 3])
    nest.assert_shallow_structure(inp_shallow, inp_deep, check_types=False)
    nest.assert_shallow_structure(inp_shallow, inp_deep, check_types=True)

    # This assertion is expected to pass: two list-types with same number
    # of fields are considered identical.
    inp_shallow = _CustomList([1, 2])
    inp_deep = [1, 2]
    nest.assert_shallow_structure(inp_shallow, inp_deep, check_types=False)
    nest.assert_shallow_structure(inp_shallow, inp_deep, check_types=True)

    # This assertion is expected to pass: a VariableSpec with alias_id and
    # a Variable are considered identical.
    inp_shallow = resource_variable_ops.VariableSpec(None, alias_id=0)
    inp_deep = resource_variable_ops.ResourceVariable(1.)
    nest.assert_shallow_structure(inp_shallow, inp_deep,
                                  expand_composites=False)
    nest.assert_shallow_structure(inp_shallow, inp_deep,
                                  expand_composites=True)

  def testFlattenUpTo(self):
    # [FUNCTION DOC]
    # Test: flatten-up-to over many shapes. Shallow trees ending at scalars or strings, mappings
    # (values, not keys, returned), namedtuples, ordered dictionaries, scalar shallow trees over
    # sequence inputs, scalars on both sides, and type errors when the shallow tree is a
    # sequence but the input is not. The last scenario checks that assert-shallow-structure
    # raises the mismatching-lengths value error for an input longer than the shallow tree.
    # Shallow tree ends at scalar.
    input_tree = [[[2, 2], [3, 3]], [[4, 9], [5, 5]]]
    shallow_tree = [[True, True], [False, True]]
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [[2, 2], [3, 3], [4, 9], [5, 5]])
    self.assertEqual(flattened_shallow_tree, [True, True, False, True])

    # Shallow tree ends at string.
    input_tree = [[("a", 1), [("b", 2), [("c", 3), [("d", 4)]]]]]
    shallow_tree = [["level_1", ["level_2", ["level_3", ["level_4"]]]]]
    input_tree_flattened_as_shallow_tree = nest.flatten_up_to(shallow_tree,
                                                              input_tree)
    input_tree_flattened = nest.flatten(input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [("a", 1), ("b", 2), ("c", 3), ("d", 4)])
    self.assertEqual(input_tree_flattened, ["a", 1, "b", 2, "c", 3, "d", 4])

    # Make sure dicts are correctly flattened, yielding values, not keys.
    input_tree = {"a": 1, "b": {"c": 2}, "d": [3, (4, 5)]}
    shallow_tree = {"a": 0, "b": 0, "d": [0, 0]}
    input_tree_flattened_as_shallow_tree = nest.flatten_up_to(shallow_tree,
                                                              input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [1, {"c": 2}, 3, (4, 5)])

    # Namedtuples.
    ab_tuple = NestTest.ABTuple
    input_tree = ab_tuple(a=[0, 1], b=2)
    shallow_tree = ab_tuple(a=0, b=1)
    input_tree_flattened_as_shallow_tree = nest.flatten_up_to(shallow_tree,
                                                              input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [[0, 1], 2])

    # Nested dicts, OrderedDicts and namedtuples.
    input_tree = collections.OrderedDict(
        [("a", ab_tuple(a=[0, {"b": 1}], b=2)),
         ("c", {"d": 3, "e": collections.OrderedDict([("f", 4)])})])
    shallow_tree = input_tree
    input_tree_flattened_as_shallow_tree = nest.flatten_up_to(shallow_tree,
                                                              input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree, [0, 1, 2, 3, 4])
    shallow_tree = collections.OrderedDict([("a", 0), ("c", {"d": 3, "e": 1})])
    input_tree_flattened_as_shallow_tree = nest.flatten_up_to(shallow_tree,
                                                              input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [ab_tuple(a=[0, {"b": 1}], b=2),
                      3,
                      collections.OrderedDict([("f", 4)])])
    shallow_tree = collections.OrderedDict([("a", 0), ("c", 0)])
    input_tree_flattened_as_shallow_tree = nest.flatten_up_to(shallow_tree,
                                                              input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [ab_tuple(a=[0, {"b": 1}], b=2),
                      {"d": 3, "e": collections.OrderedDict([("f", 4)])}])

    ## Shallow non-list edge-case.
    # Using iterable elements.
    input_tree = ["input_tree"]
    shallow_tree = "shallow_tree"
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    input_tree = ["input_tree_0", "input_tree_1"]
    shallow_tree = "shallow_tree"
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    # Using non-iterable elements.
    input_tree = [0]
    shallow_tree = 9
    flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    input_tree = [0, 1]
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
    shallow_tree = ["shallow_tree"]
    expected_message = ("If shallow structure is a sequence, input must also "
                        "be a sequence. Input has type: <(type|class) 'str'>.")
    with self.assertRaisesRegex(TypeError, expected_message):
      flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree, shallow_tree)

    input_tree = "input_tree"
    shallow_tree = ["shallow_tree_9", "shallow_tree_8"]
    with self.assertRaisesRegex(TypeError, expected_message):
      flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree, shallow_tree)

    # Using non-iterable elements.
    input_tree = 0
    shallow_tree = [9]
    expected_message = ("If shallow structure is a sequence, input must also "
                        "be a sequence. Input has type: <(type|class) 'int'>.")
    with self.assertRaisesRegex(TypeError, expected_message):
      flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree, shallow_tree)

    input_tree = 0
    shallow_tree = [9, 8]
    with self.assertRaisesRegex(TypeError, expected_message):
      flattened_input_tree = nest.flatten_up_to(shallow_tree, input_tree)
    flattened_shallow_tree = nest.flatten_up_to(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree, shallow_tree)

    input_tree = [(1,), (2,), 3]
    shallow_tree = [(1,), (2,)]
    expected_message = nest.STRUCTURES_HAVE_MISMATCHING_LENGTHS.format(
        input_length=len(input_tree), shallow_length=len(shallow_tree)
    )
    with self.assertRaisesRegex(ValueError, expected_message):  # pylint: disable=g-error-prone-assert-raises
      nest.assert_shallow_structure(shallow_tree, input_tree)

  def testFlattenWithTuplePathsUpTo(self):
    # [FUNCTION DOC]
    # Test: flatten-with-tuple-paths-up-to using a local function that returns the lists of
    # paths and values. Covers the same families of cases as the flatten-up-to test (scalar and
    # string leaves, mappings, namedtuples, ordered dictionaries, scalar shallow trees), checking
    # paths and values; a value error for mismatching lengths; and exact-message type errors
    # when the shallow tree is a sequence and the input is not.
    def get_paths_and_values(shallow_tree, input_tree):
      # [FUNCTION DOC]
      # HELPER: local utility of the flatten-with-tuple-paths-up-to test.
      # Inputs: shallow tree and input tree. Calls flatten-with-tuple-paths-up-to and splits the
      # resulting pairs into two lists. Returns a pair: list of paths, list of values. Errors
      # from the library call propagate.
      path_value_pairs = nest.flatten_with_tuple_paths_up_to(
          shallow_tree, input_tree)
      paths = [p for p, _ in path_value_pairs]
      values = [v for _, v in path_value_pairs]
      return paths, values

    # Shallow tree ends at scalar.
    input_tree = [[[2, 2], [3, 3]], [[4, 9], [5, 5]]]
    shallow_tree = [[True, True], [False, True]]
    (flattened_input_tree_paths,
     flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree_paths,
                     [(0, 0), (0, 1), (1, 0), (1, 1)])
    self.assertEqual(flattened_input_tree, [[2, 2], [3, 3], [4, 9], [5, 5]])
    self.assertEqual(flattened_shallow_tree_paths,
                     [(0, 0), (0, 1), (1, 0), (1, 1)])
    self.assertEqual(flattened_shallow_tree, [True, True, False, True])

    # Shallow tree ends at string.
    input_tree = [[("a", 1), [("b", 2), [("c", 3), [("d", 4)]]]]]
    shallow_tree = [["level_1", ["level_2", ["level_3", ["level_4"]]]]]
    (input_tree_flattened_as_shallow_tree_paths,
     input_tree_flattened_as_shallow_tree) = get_paths_and_values(shallow_tree,
                                                                  input_tree)
    input_tree_flattened_paths = [p for p, _ in
                                  nest.flatten_with_tuple_paths(input_tree)]
    input_tree_flattened = nest.flatten(input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree_paths,
                     [(0, 0), (0, 1, 0), (0, 1, 1, 0), (0, 1, 1, 1, 0)])
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [("a", 1), ("b", 2), ("c", 3), ("d", 4)])

    self.assertEqual(input_tree_flattened_paths,
                     [(0, 0, 0), (0, 0, 1),
                      (0, 1, 0, 0), (0, 1, 0, 1),
                      (0, 1, 1, 0, 0), (0, 1, 1, 0, 1),
                      (0, 1, 1, 1, 0, 0), (0, 1, 1, 1, 0, 1)])
    self.assertEqual(input_tree_flattened, ["a", 1, "b", 2, "c", 3, "d", 4])

    # Make sure dicts are correctly flattened, yielding values, not keys.
    input_tree = {"a": 1, "b": {"c": 2}, "d": [3, (4, 5)]}
    shallow_tree = {"a": 0, "b": 0, "d": [0, 0]}
    (input_tree_flattened_as_shallow_tree_paths,
     input_tree_flattened_as_shallow_tree) = get_paths_and_values(shallow_tree,
                                                                  input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree_paths,
                     [("a",), ("b",), ("d", 0), ("d", 1)])
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [1, {"c": 2}, 3, (4, 5)])

    # Namedtuples.
    ab_tuple = collections.namedtuple("ab_tuple", "a, b")
    input_tree = ab_tuple(a=[0, 1], b=2)
    shallow_tree = ab_tuple(a=0, b=1)
    (input_tree_flattened_as_shallow_tree_paths,
     input_tree_flattened_as_shallow_tree) = get_paths_and_values(shallow_tree,
                                                                  input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree_paths,
                     [("a",), ("b",)])
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [[0, 1], 2])

    # Nested dicts, OrderedDicts and namedtuples.
    input_tree = collections.OrderedDict(
        [("a", ab_tuple(a=[0, {"b": 1}], b=2)),
         ("c", {"d": 3, "e": collections.OrderedDict([("f", 4)])})])
    shallow_tree = input_tree
    (input_tree_flattened_as_shallow_tree_paths,
     input_tree_flattened_as_shallow_tree) = get_paths_and_values(shallow_tree,
                                                                  input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree_paths,
                     [("a", "a", 0),
                      ("a", "a", 1, "b"),
                      ("a", "b"),
                      ("c", "d"),
                      ("c", "e", "f")])
    self.assertEqual(input_tree_flattened_as_shallow_tree, [0, 1, 2, 3, 4])
    shallow_tree = collections.OrderedDict([("a", 0), ("c", {"d": 3, "e": 1})])
    (input_tree_flattened_as_shallow_tree_paths,
     input_tree_flattened_as_shallow_tree) = get_paths_and_values(shallow_tree,
                                                                  input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree_paths,
                     [("a",),
                      ("c", "d"),
                      ("c", "e")])
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [ab_tuple(a=[0, {"b": 1}], b=2),
                      3,
                      collections.OrderedDict([("f", 4)])])
    shallow_tree = collections.OrderedDict([("a", 0), ("c", 0)])
    (input_tree_flattened_as_shallow_tree_paths,
     input_tree_flattened_as_shallow_tree) = get_paths_and_values(shallow_tree,
                                                                  input_tree)
    self.assertEqual(input_tree_flattened_as_shallow_tree_paths,
                     [("a",), ("c",)])
    self.assertEqual(input_tree_flattened_as_shallow_tree,
                     [ab_tuple(a=[0, {"b": 1}], b=2),
                      {"d": 3, "e": collections.OrderedDict([("f", 4)])}])

    ## Shallow non-list edge-case.
    # Using iterable elements.
    input_tree = ["input_tree"]
    shallow_tree = "shallow_tree"
    (flattened_input_tree_paths,
     flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree_paths, [()])
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree_paths, [()])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    input_tree = ["input_tree_0", "input_tree_1"]
    shallow_tree = "shallow_tree"
    (flattened_input_tree_paths,
     flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree_paths, [()])
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree_paths, [()])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    # Test case where len(shallow_tree) < len(input_tree)
    input_tree = {"a": "A", "b": "B", "c": "C"}
    shallow_tree = {"a": 1, "c": 2}

    with self.assertRaisesWithLiteralMatch(  # pylint: disable=g-error-prone-assert-raises
        ValueError,
        nest.STRUCTURES_HAVE_MISMATCHING_LENGTHS.format(
            input_length=len(input_tree), shallow_length=len(shallow_tree)
        ),
    ):
      get_paths_and_values(shallow_tree, input_tree)

    # Using non-iterable elements.
    input_tree = [0]
    shallow_tree = 9
    (flattened_input_tree_paths,
     flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree_paths, [()])
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree_paths, [()])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    input_tree = [0, 1]
    shallow_tree = 9
    (flattened_input_tree_paths,
     flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree_paths, [()])
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree_paths, [()])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    ## Both non-list edge-case.
    # Using iterable elements.
    input_tree = "input_tree"
    shallow_tree = "shallow_tree"
    (flattened_input_tree_paths,
     flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree_paths, [()])
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree_paths, [()])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    # Using non-iterable elements.
    input_tree = 0
    shallow_tree = 0
    (flattened_input_tree_paths,
     flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_input_tree_paths, [()])
    self.assertEqual(flattened_input_tree, [input_tree])
    self.assertEqual(flattened_shallow_tree_paths, [()])
    self.assertEqual(flattened_shallow_tree, [shallow_tree])

    ## Input non-list edge-case.
    # Using iterable elements.
    input_tree = "input_tree"
    shallow_tree = ["shallow_tree"]
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        nest.IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ.format(type(input_tree)),
    ):
      (flattened_input_tree_paths,
       flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree_paths, [(0,)])
    self.assertEqual(flattened_shallow_tree, shallow_tree)

    input_tree = "input_tree"
    shallow_tree = ["shallow_tree_9", "shallow_tree_8"]
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        nest.IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ.format(type(input_tree)),
    ):
      (flattened_input_tree_paths,
       flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree_paths, [(0,), (1,)])
    self.assertEqual(flattened_shallow_tree, shallow_tree)

    # Using non-iterable elements.
    input_tree = 0
    shallow_tree = [9]
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        nest.IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ.format(type(input_tree)),
    ):
      (flattened_input_tree_paths,
       flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree_paths, [(0,)])
    self.assertEqual(flattened_shallow_tree, shallow_tree)

    input_tree = 0
    shallow_tree = [9, 8]
    with self.assertRaisesWithLiteralMatch(
        TypeError,
        nest.IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ.format(type(input_tree)),
    ):
      (flattened_input_tree_paths,
       flattened_input_tree) = get_paths_and_values(shallow_tree, input_tree)
    (flattened_shallow_tree_paths,
     flattened_shallow_tree) = get_paths_and_values(shallow_tree, shallow_tree)
    self.assertEqual(flattened_shallow_tree_paths, [(0,), (1,)])
    self.assertEqual(flattened_shallow_tree, shallow_tree)

  def testMapStructureUpTo(self):
    # [FUNCTION DOC]
    # Test: map-structure-up-to over namedtuples, lists and dictionaries, passing a shallow
    # structure and applying a function to the deeper parts of the inputs. Asserts results.
    # Dictionaries with different keys, or a dictionary against a custom mapping with different
    # keys, raise a value error with the invalid-keys message.
    # Named tuples.
    ab_tuple = collections.namedtuple("ab_tuple", "a, b")
    op_tuple = collections.namedtuple("op_tuple", "add, mul")
    inp_val = ab_tuple(a=2, b=3)
    inp_ops = ab_tuple(a=op_tuple(add=1, mul=2), b=op_tuple(add=2, mul=3))
    out = nest.map_structure_up_to(
        inp_val, lambda val, ops: (val + ops.add) * ops.mul, inp_val, inp_ops)
    self.assertEqual(out.a, 6)
    self.assertEqual(out.b, 15)

    # Lists.
    data_list = [[2, 4, 6, 8], [[1, 3, 5, 7, 9], [3, 5, 7]]]
    name_list = ["evens", ["odds", "primes"]]
    out = nest.map_structure_up_to(
        name_list, lambda name, sec: "first_{}_{}".format(len(sec), name),
        name_list, data_list)
    self.assertEqual(out, ["first_4_evens", ["first_5_odds", "first_3_primes"]])

    # Dicts.
    inp_val = dict(a=2, b=3)
    inp_ops = dict(a=dict(add=1, mul=2), b=dict(add=2, mul=3))
    out = nest.map_structure_up_to(
        inp_val,
        lambda val, ops: (val + ops["add"]) * ops["mul"], inp_val, inp_ops)
    self.assertEqual(out["a"], 6)
    self.assertEqual(out["b"], 15)

    # Non-equal dicts.
    inp_val = dict(a=2, b=3)
    inp_ops = dict(a=dict(add=1, mul=2), c=dict(add=2, mul=3))
    with self.assertRaisesWithLiteralMatch(
        ValueError, nest.SHALLOW_TREE_HAS_INVALID_KEYS.format(["b"])
    ):
      nest.map_structure_up_to(
          inp_val,
          lambda val, ops: (val + ops["add"]) * ops["mul"], inp_val, inp_ops)

    # Dict+custom mapping.
    inp_val = dict(a=2, b=3)
    inp_ops = _CustomMapping(a=dict(add=1, mul=2), b=dict(add=2, mul=3))
    out = nest.map_structure_up_to(
        inp_val,
        lambda val, ops: (val + ops["add"]) * ops["mul"], inp_val, inp_ops)
    self.assertEqual(out["a"], 6)
    self.assertEqual(out["b"], 15)

    # Non-equal dict/mapping.
    inp_val = dict(a=2, b=3)
    inp_ops = _CustomMapping(a=dict(add=1, mul=2), c=dict(add=2, mul=3))
    with self.assertRaisesWithLiteralMatch(
        ValueError, nest.SHALLOW_TREE_HAS_INVALID_KEYS.format(["b"])
    ):
      nest.map_structure_up_to(
          inp_val,
          lambda val, ops: (val + ops["add"]) * ops["mul"], inp_val, inp_ops)

  def testGetTraverseShallowStructure(self):
    # [FUNCTION DOC]
    # Test: get-traverse-shallow-structure with a scalar-returning traversal function
    # (not descending into tuples) and with one returning a boolean structure for tuples.
    # Results must equal the expected boolean structures and be shallow structures of the
    # input. Errors: type error when the function returns a structure for a scalar, a non-boolean
    # scalar, or a structure that is not made of booleans.
    scalar_traverse_input = [3, 4, (1, 2, [0]), [5, 6], {"a": (7,)}, []]
    scalar_traverse_r = nest.get_traverse_shallow_structure(
        lambda s: not isinstance(s, tuple),
        scalar_traverse_input)
    self.assertEqual(scalar_traverse_r,
                     [True, True, False, [True, True], {"a": False}, []])
    nest.assert_shallow_structure(scalar_traverse_r,
                                  scalar_traverse_input)

    structure_traverse_input = [(1, [2]), ([1], 2)]
    structure_traverse_r = nest.get_traverse_shallow_structure(
        lambda s: (True, False) if isinstance(s, tuple) else True,
        structure_traverse_input)
    self.assertEqual(structure_traverse_r,
                     [(True, False), ([True], False)])
    nest.assert_shallow_structure(structure_traverse_r,
                                  structure_traverse_input)

    with self.assertRaisesRegex(TypeError, "returned structure"):
      nest.get_traverse_shallow_structure(lambda _: [True], 0)

    with self.assertRaisesRegex(TypeError, "returned a non-bool scalar"):
      nest.get_traverse_shallow_structure(lambda _: 1, [1])

    with self.assertRaisesRegex(TypeError,
                                "didn't return a depth=1 structure of bools"):
      nest.get_traverse_shallow_structure(lambda _: [1], [1])

  def testYieldFlatStringPaths(self):
    # [FUNCTION DOC]
    # Test: yield-flat-paths on empty lists, scalars, lists, dictionaries (sorted keys) and
    # their nestings, including string keys that look like numbers. Compares the yielded paths
    # with the expected list for every case in a loop.
    for inputs_expected in ({"inputs": [], "expected": []},
                            {"inputs": 3, "expected": [()]},
                            {"inputs": [3], "expected": [(0,)]},
                            {"inputs": {"a": 3}, "expected": [("a",)]},
                            {"inputs": {"a": {"b": 4}},
                             "expected": [("a", "b")]},
                            {"inputs": [{"a": 2}], "expected": [(0, "a")]},
                            {"inputs": [{"a": [2]}], "expected": [(0, "a", 0)]},
                            {"inputs": [{"a": [(23, 42)]}],
                             "expected": [(0, "a", 0, 0), (0, "a", 0, 1)]},
                            {"inputs": [{"a": ([23], 42)}],
                             "expected": [(0, "a", 0, 0), (0, "a", 1)]},
                            {"inputs": {"a": {"a": 2}, "c": [[[4]]]},
                             "expected": [("a", "a"), ("c", 0, 0, 0)]},
                            {"inputs": {"0": [{"1": 23}]},
                             "expected": [("0", 0, "1")]}):
      inputs = inputs_expected["inputs"]
      expected = inputs_expected["expected"]
      self.assertEqual(list(nest.yield_flat_paths(inputs)), expected)

  # We cannot define namedtuples within @parameterized argument lists.
  # pylint: disable=invalid-name
  Foo = collections.namedtuple("Foo", ["a", "b"])
  Bar = collections.namedtuple("Bar", ["c", "d"])
  # pylint: enable=invalid-name

  @parameterized.parameters([
      dict(inputs=[], expected=[]),
      dict(inputs=[23, "42"], expected=[("0", 23), ("1", "42")]),
      dict(inputs=[[[[108]]]], expected=[("0/0/0/0", 108)]),
      dict(inputs=Foo(a=3, b=Bar(c=23, d=42)),
           expected=[("a", 3), ("b/c", 23), ("b/d", 42)]),
      dict(inputs=Foo(a=Bar(c=23, d=42), b=Bar(c=0, d="thing")),
           expected=[("a/c", 23), ("a/d", 42), ("b/c", 0), ("b/d", "thing")]),
      dict(inputs=Bar(c=42, d=43),
           expected=[("c", 42), ("d", 43)]),
      dict(inputs=Bar(c=[42], d=43),
           expected=[("c/0", 42), ("d", 43)]),
  ])
  def testFlattenWithStringPaths(self, inputs, expected):
    # [FUNCTION DOC]
    # Test (parameterized over structures with expected joined-path/value pairs): asserts that
    # flatten-with-joined-string-paths with a slash separator returns the expected pairs.
    self.assertEqual(
        nest.flatten_with_joined_string_paths(inputs, separator="/"),
        expected)

  @parameterized.parameters([
      dict(inputs=[], expected=[]),
      dict(inputs=[23, "42"], expected=[((0,), 23), ((1,), "42")]),
      dict(inputs=[[[[108]]]], expected=[((0, 0, 0, 0), 108)]),
      dict(inputs=Foo(a=3, b=Bar(c=23, d=42)),
           expected=[(("a",), 3), (("b", "c"), 23), (("b", "d"), 42)]),
      dict(inputs=Foo(a=Bar(c=23, d=42), b=Bar(c=0, d="thing")),
           expected=[(("a", "c"), 23), (("a", "d"), 42), (("b", "c"), 0),
                     (("b", "d"), "thing")]),
      dict(inputs=Bar(c=42, d=43),
           expected=[(("c",), 42), (("d",), 43)]),
      dict(inputs=Bar(c=[42], d=43),
           expected=[(("c", 0), 42), (("d",), 43)]),
  ])
  def testFlattenWithTuplePaths(self, inputs, expected):
    # [FUNCTION DOC]
    # Test (parameterized over structures with expected tuple-path/value pairs): asserts that
    # flatten-with-tuple-paths returns the expected pairs.
    self.assertEqual(nest.flatten_with_tuple_paths(inputs), expected)

  @parameterized.named_parameters(
      ("tuples", (1, 2), (3, 4), True, (("0", 4), ("1", 6))),
      ("dicts", {"a": 1, "b": 2}, {"b": 4, "a": 3}, True,
       {"a": ("a", 4), "b": ("b", 6)}),
      ("mixed", (1, 2), [3, 4], False, (("0", 4), ("1", 6))),
      ("nested",
       {"a": [2, 3], "b": [1, 2, 3]}, {"b": [5, 6, 7], "a": [8, 9]}, True,
       {"a": [("a/0", 10), ("a/1", 12)],
        "b": [("b/0", 6), ("b/1", 8), ("b/2", 10)]}))
  def testMapWithPathsCompatibleStructures(self, s1, s2, check_types, expected):
    # [FUNCTION DOC]
    # Test (named parameters: tuples, dicts, mixed types, nested): map-structure-with-paths with
    # a local function that returns the joined string path and the sum of values. Asserts the
    # result equals the expected structure; the type check flag is passed per case.
    def format_sum(path, *values):
      # [FUNCTION DOC]
      # HELPER: local mapping function of the compatible-structures test.
      # Inputs: a path string and any number of values. Returns a pair of the path and the
      # sum of the values.
      return (path, sum(values))
    result = nest.map_structure_with_paths(format_sum, s1, s2,
                                           check_types=check_types)
    self.assertEqual(expected, result)

  @parameterized.named_parameters(
      ("tuples", (1, 2, 3), (4, 5), ValueError),
      ("dicts", {"a": 1}, {"b": 2}, ValueError),
      ("mixed", (1, 2), [3, 4], TypeError),
      ("nested",
       {"a": [2, 3, 4], "b": [1, 3]},
       {"b": [5, 6], "a": [8, 9]},
       ValueError
      ))
  def testMapWithPathsIncompatibleStructures(self, s1, s2, error_type):
    # [FUNCTION DOC]
    # Test (named parameters with expected error type): map-structure-with-paths over
    # incompatible structures must raise the given error (value error for size or key
    # mismatches, type error for mixed sequence types).
    with self.assertRaises(error_type):
      nest.map_structure_with_paths(lambda path, *s: 0, s1, s2)

  @parameterized.named_parameters([
      dict(testcase_name="Tuples", s1=(1, 2), s2=(3, 4),
           check_types=True, expected=(((0,), 4), ((1,), 6))),
      dict(testcase_name="Dicts", s1={"a": 1, "b": 2}, s2={"b": 4, "a": 3},
           check_types=True, expected={"a": (("a",), 4), "b": (("b",), 6)}),
      dict(testcase_name="Mixed", s1=(1, 2), s2=[3, 4],
           check_types=False, expected=(((0,), 4), ((1,), 6))),
      dict(testcase_name="Nested",
           s1={"a": [2, 3], "b": [1, 2, 3]},
           s2={"b": [5, 6, 7], "a": [8, 9]},
           check_types=True,
           expected={"a": [(("a", 0), 10), (("a", 1), 12)],
                     "b": [(("b", 0), 6), (("b", 1), 8), (("b", 2), 10)]}),
  ])
  def testMapWithTuplePathsCompatibleStructures(
      self, s1, s2, check_types, expected):
    # [FUNCTION DOC]
    # Test (named parameters: tuples, dicts, mixed, nested): map-structure-with-tuple-paths with
    # a local function that returns the tuple path and the sum of values. Asserts the result
    # equals the expected structure; the type check flag is passed per case.
    def path_and_sum(path, *values):
      # [FUNCTION DOC]
      # HELPER: local mapping function of the compatible-structures test.
      # Inputs: a tuple path and any number of values. Returns a pair of the path and the sum.
      return path, sum(values)
    result = nest.map_structure_with_tuple_paths(
        path_and_sum, s1, s2, check_types=check_types)
    self.assertEqual(expected, result)

  @parameterized.named_parameters([
      dict(testcase_name="Tuples", s1=(1, 2, 3), s2=(4, 5),
           error_type=ValueError),
      dict(testcase_name="Dicts", s1={"a": 1}, s2={"b": 2},
           error_type=ValueError),
      dict(testcase_name="Mixed", s1=(1, 2), s2=[3, 4], error_type=TypeError),
      dict(testcase_name="Nested",
           s1={"a": [2, 3, 4], "b": [1, 3]},
           s2={"b": [5, 6], "a": [8, 9]},
           error_type=ValueError)
  ])
  def testMapWithTuplePathsIncompatibleStructures(self, s1, s2, error_type):
    # [FUNCTION DOC]
    # Test (named parameters with expected error type): map-structure-with-tuple-paths over
    # incompatible structures must raise the given error type.
    with self.assertRaises(error_type):
      nest.map_structure_with_tuple_paths(lambda path, *s: 0, s1, s2)

  def testFlattenCustomSequenceThatRaisesException(self):  # b/140746865
    # [FUNCTION DOC]
    # Test (regression for a referenced bug): flattening a sequence whose item access raises a
    # value error must propagate that error (message "Cannot get item").
    seq = _CustomSequenceThatRaisesException()
    with self.assertRaisesRegex(ValueError, "Cannot get item"):
      nest.flatten(seq)

  def testListToTuple(self):
    # [FUNCTION DOC]
    # Test: list-to-tuple recursively converts lists to tuples inside tuples and dictionaries
    # (keeping None and scalars). Asserts same structure as the expected nested tuples.
    input_sequence = [1, (2, {3: [4, 5, (6,)]}, None, 7, [[[8]]])]
    expected = (1, (2, {3: (4, 5, (6,))}, None, 7, (((8,),),)))
    nest.assert_same_structure(
        nest.list_to_tuple(input_sequence),
        expected,
    )

  def testInvalidCheckTypes(self):
    # [FUNCTION DOC]
    # Test: assert-same-structure with a tensor passed as the type-check flag or as the
    # expand-composites flag, with structurally different tensor arguments, must raise a value
    # error or a type error.
    with self.assertRaises((ValueError, TypeError)):
      nest.assert_same_structure(
          nest1=array_ops.zeros((1)),
          nest2=array_ops.ones((1, 1, 1)),
          check_types=array_ops.ones((2)))
    with self.assertRaises((ValueError, TypeError)):
      nest.assert_same_structure(
          nest1=array_ops.zeros((1)),
          nest2=array_ops.ones((1, 1, 1)),
          expand_composites=array_ops.ones((2)))

  def testIsNamedtuple(self):
    # [FUNCTION DOC]
    # Test: is-namedtuple. True for a classic namedtuple, a subclass of it and a typed named
    # tuple instance; false for the classes themselves, for numbers, strings and plain tuples,
    # and for a tuple subclass that merely has a non-string fields attribute.
    # A classic namedtuple.
    Foo = collections.namedtuple("Foo", ["a", "b"])
    self.assertTrue(nest.is_namedtuple(Foo(1, 2)))

    # A subclass of it.
    class SubFoo(Foo):
      # [CLASS DOC]
      # HELPER: local fixture of the is-namedtuple test.
      # Subclass of a classic namedtuple that adds one method, to check subclass detection.

      def extra_method(self, x):
        # [FUNCTION DOC]
        # HELPER: local fixture method. Returns the first field of the tuple plus x.
        return self.a + x

    self.assertTrue(nest.is_namedtuple(SubFoo(1, 2)))

    # A typing.NamedTuple.
    class TypedFoo(NamedTuple):
      # [CLASS DOC]
      # HELPER: local fixture of the is-namedtuple and same-namedtuples tests.
      # Typed named tuple with two integer fields, a and b.
      a: int
      b: int
    self.assertTrue(nest.is_namedtuple(TypedFoo(1, 2)))

    # Their types are not namedtuple values themselves.
    self.assertFalse(nest.is_namedtuple(Foo))
    self.assertFalse(nest.is_namedtuple(SubFoo))
    self.assertFalse(nest.is_namedtuple(TypedFoo))

    # These values don't have namedtuple types.
    self.assertFalse(nest.is_namedtuple(123))
    self.assertFalse(nest.is_namedtuple("abc"))
    self.assertFalse(nest.is_namedtuple((123, "abc")))

    class SomethingElseWithFields(tuple):
      # [CLASS DOC]
      # HELPER: local fixture of the is-namedtuple test.
      # Tuple subclass whose constructor sets a fields attribute to a list of integers (not
      # strings), so it must not be recognised as a namedtuple.

      def __init__(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: constructor of the fake-namedtuple fixture.
        # Passes the arguments to the parent constructor and then sets a fields attribute to the
        # list one, two, three. Returns nothing.
        super().__init__(*args, **kwargs)
        self._fields = [1, 2, 3]  # Not str, as expected for a namedtuple.

    self.assertFalse(nest.is_namedtuple(SomethingElseWithFields()))

  def testSameNamedtuples(self):
    # [FUNCTION DOC]
    # Test: same-namedtuples is true for instances of one namedtuple type or of two
    # equivalent definitions (same name and fields), and false when names or field names differ.
    # Also true for an equivalent typed named tuple definition.
    # A classic namedtuple and an equivalent cppy.
    Foo1 = collections.namedtuple("Foo", ["a", "b"])
    Foo2 = collections.namedtuple("Foo", ["a", "b"])
    self.assertTrue(nest.same_namedtuples(Foo1(1, 2), Foo1(3, 4)))
    self.assertTrue(nest.same_namedtuples(Foo1(1, 2), Foo2(3, 4)))

    # Non-equivalent namedtuples.
    Bar = collections.namedtuple("Bar", ["a", "b"])
    self.assertFalse(nest.same_namedtuples(Foo1(1, 2), Bar(1, 2)))
    FooXY = collections.namedtuple("Foo", ["x", "y"])
    self.assertFalse(nest.same_namedtuples(Foo1(1, 2), FooXY(1, 2)))

    # An equivalent subclass from the typing module
    class Foo(NamedTuple):
      # [CLASS DOC]
      # HELPER: local fixture of the same-namedtuples test.
      # Typed named tuple with the same name and field names as the classic Foo namedtuple.
      a: int
      b: int
    self.assertTrue(nest.same_namedtuples(Foo1(1, 2), Foo(3, 4)))


class NestBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Micro-benchmark of the structure comparison function. Provides a shared runner that
  # reports wall time per iteration and benchmark methods that use it for small and large
  # structures.

  def run_and_report(self, s1, s2, name):
    # [FUNCTION DOC]
    # HELPER: shared runner of the benchmark methods.
    # Inputs: two structures to compare and a benchmark name. Performs a warm-up loop of 100
    # assert-same-structure calls, then times 30000 calls and reports the iteration count and
    # average wall time per call under the given name. Returns nothing.
    burn_iter, test_iter = 100, 30000

    for _ in range(burn_iter):
      nest.assert_same_structure(s1, s2)

    t0 = time.time()
    for _ in range(test_iter):
      nest.assert_same_structure(s1, s2)
    t1 = time.time()

    self.report_benchmark(iters=test_iter, wall_time=(t1 - t0) / test_iter,
                          name=name)

  def benchmark_assert_structure(self):
    # [FUNCTION DOC]
    # Benchmark: compares a six-element integer structure with a same-shaped string structure,
    # then the same pair repeated ten times (sixty elements), reporting each through the
    # shared runner under separate names.
    s1 = (((1, 2), 3), 4, (5, 6))
    s2 = ((("foo1", "foo2"), "foo3"), "foo4", ("foo5", "foo6"))
    self.run_and_report(s1, s2, "assert_same_structure_6_elem")

    s1 = (((1, 2), 3), 4, (5, 6)) * 10
    s2 = ((("foo1", "foo2"), "foo3"), "foo4", ("foo5", "foo6")) * 10
    self.run_and_report(s1, s2, "assert_same_structure_60_elem")


if __name__ == "__main__":
  test.main()
