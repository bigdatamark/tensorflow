# [MODULE DOC]
# KNOWN ISSUE: The named tuple type used for index specifications is given the type name
# "IndexTest" while it is bound to the module variable IndexSpec, and the table of gradient
# indexing cases (static indexing grad tests) is defined but never used by any test below.
# Test module for the numpy-compatible indexing and indexed-update API of the TensorFlow
# numpy ops package. It defines tables of indexing cases (static slicing, integer, ellipsis
# and new-axis indexing, advanced integer-array indexing, and mixed forms) as lists of
# named groups of specs pairing an input shape with an indexer. Two test classes consume
# them: one checks that reading elements via indexing on tensors matches plain numpy and
# compiles correctly, and one checks the index update, add, min and max operations and the
# dynamic slice and dynamic update slice extensions against numpy reference
# implementations. Test base class and helpers come from the sibling test utility, config,
# extensions and wrapper modules. Import time side effect: command line flags are parsed
# through the config object. Running the file as a script disables soft device placement
# and starts the absl test runner.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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


import collections
import enum
from functools import partial
import itertools
import unittest

from absl.testing import absltest
from absl.testing import parameterized
import numpy as onp

from tensorflow.python.framework import config as tf_config
from tensorflow.python.ops.numpy_ops.tests.config import config
import tensorflow.python.ops.numpy_ops.tests.extensions as nje
import tensorflow.python.ops.numpy_ops.tests.np_wrapper as tnp
import tensorflow.python.ops.numpy_ops.tests.test_util as jtu
from tensorflow.python.util import nest

config.parse_flags_with_absl()


# We disable the whitespace continuation check in this file because otherwise it
# makes the test name formatting unwieldy.
# pylint: disable=bad-continuation
# We also disable undefined-variable till we start enabling tests.
# pylint: disable=undefined-variable


def subvals(lst, replace):
  # [FUNCTION DOC]
  # HELPER: small utility for the indexing tests that rebuild index tuples.
  # Returns a tuple copy of the sequence lst in which, for every pair of position and value
  # in replace (an iterable of such pairs), the element at that position is substituted by
  # the value. The input is not mutated. Raises an index error if a position is out of range.
  lst = list(lst)
  for i, v in replace:
    lst[i] = v
  return tuple(lst)


float_dtypes = [onp.float32, onp.float64]
int_dtypes = [onp.int32, onp.int64]
bool_types = [onp.bool_]
default_dtypes = float_dtypes + int_dtypes
all_dtypes = float_dtypes + int_dtypes + bool_types

IndexSpec = collections.namedtuple("IndexTest", ["shape", "indexer"])


suppress_deprecated_indexing_warnings = partial(
  jtu.ignore_warning, category=FutureWarning,
  message='Using a non-tuple sequence.*')


STATIC_INDEXING_TESTS = [
    ("OneIntIndex", [
        IndexSpec(shape=(3,), indexer=1),
        IndexSpec(shape=(3, 3), indexer=0),
        IndexSpec(shape=(3, 4, 5), indexer=2),
        IndexSpec(shape=(3,), indexer=-1),
        IndexSpec(shape=(3,), indexer=-2),
    ]),
    ("TwoIntIndices", [
        IndexSpec(shape=(3, 3), indexer=(2, 1)),
        IndexSpec(shape=(3, 4, 5), indexer=(1, 2)),
        IndexSpec(shape=(3, 4, 5), indexer=(-1, 2)),
    ]),
    ("ThreeIntIndices", [IndexSpec((3, 4, 5), indexer=(1, 2, 3))]),
    ("OneSliceIndex", [
        IndexSpec(shape=(10,), indexer=slice(1, 3)),
        IndexSpec(shape=(10,), indexer=slice(1, -1)),
        IndexSpec(shape=(10,), indexer=slice(None, -1)),
        IndexSpec(shape=(10,), indexer=slice(None, None, None)),
        IndexSpec(shape=(10, 8), indexer=slice(1, 3)),
        IndexSpec(shape=(10, 8), indexer=slice(1, None)),
        IndexSpec(shape=(10, 8), indexer=slice(None, 3)),
        IndexSpec(shape=(10, 8), indexer=slice(-3, None)),
    ]),
    ("OneSliceIndexNegativeStride", [
        IndexSpec(shape=(10,), indexer=slice(3, 1, -1)),
        IndexSpec(shape=(10,), indexer=slice(1, 8, -1)),  # empty result
        IndexSpec(shape=(10,), indexer=slice(None, 1, -2)),
        IndexSpec(shape=(10,), indexer=slice(None, None, -1)),
        IndexSpec(shape=(10, 8), indexer=slice(3, 1, -1)),
        IndexSpec(shape=(10, 8), indexer=slice(0, 8, -1)),  # empty result
        IndexSpec(shape=(10, 8), indexer=slice(None, None, -1)),
    ]),
    ("OneSliceIndexNonUnitStride", [
        IndexSpec(shape=(10,), indexer=slice(0, 8, 2)),
        IndexSpec(shape=(10,), indexer=slice(0, 8, 3)),
        IndexSpec(shape=(10,), indexer=slice(1, 3, 2)),
        IndexSpec(shape=(10,), indexer=slice(1, None, 2)),
        IndexSpec(shape=(10,), indexer=slice(None, 1, -2)),
        IndexSpec(shape=(10, 8), indexer=slice(1, 8, 3)),
        IndexSpec(shape=(10, 8), indexer=slice(None, None, 2)),
        IndexSpec(shape=(10, 8), indexer=slice(None, 1, -2)),
        IndexSpec(shape=(10, 8), indexer=slice(None, None, -2)),
    ]),
    ("TwoSliceIndices", [
        IndexSpec(shape=(10, 8), indexer=(slice(1, 3), slice(0, 2))),
        IndexSpec(shape=(10, 8), indexer=(slice(1, None), slice(None, 2))),
        IndexSpec(
            shape=(10, 8), indexer=(slice(None, None, -1), slice(None, 2))),
        IndexSpec(shape=(10, 8, 3), indexer=(slice(1, 3), slice(0, 2))),
        IndexSpec(shape=(10, 8, 3), indexer=(slice(1, 3), slice(0, None))),
        IndexSpec(shape=(10, 8, 3), indexer=(slice(1, None), slice(0, 2))),
    ]),
    ("OneColonIndex", [
        IndexSpec(shape=(3,), indexer=slice(None)),
        IndexSpec(shape=(3, 4), indexer=slice(None)),
    ]),
    ("MultipleColonIndices", [
        IndexSpec(shape=(3, 4), indexer=(slice(None), slice(None))),
        IndexSpec(shape=(3, 4, 5), indexer=(slice(None), slice(None))),
    ]),
    ("MixedSliceIndices", [
        IndexSpec(shape=(10, 4), indexer=(slice(None), slice(0, 2))),
        IndexSpec(shape=(10, 4), indexer=(1, slice(None))),
    ]),
    ("EllipsisIndex", [
        IndexSpec(shape=(3,), indexer=Ellipsis),
        IndexSpec(shape=(3, 4), indexer=Ellipsis),
        IndexSpec(shape=(3, 4, 5), indexer=(0, Ellipsis)),
        IndexSpec(shape=(3, 4, 5), indexer=(Ellipsis, 2, 3)),
    ]),
    ("NoneIndex", [
        IndexSpec(shape=(), indexer=None),
        IndexSpec(shape=(), indexer=(None, None)),
        IndexSpec(shape=(), indexer=(Ellipsis, None)),
        IndexSpec(shape=(3,), indexer=None),
        IndexSpec(shape=(3, 4), indexer=None),
        IndexSpec(shape=(3, 4), indexer=(Ellipsis, None)),
        IndexSpec(shape=(3, 4), indexer=(0, None, Ellipsis)),
        IndexSpec(shape=(3, 4, 5), indexer=(1, None, Ellipsis)),
    ]),
    ("EmptyIndex", [
        IndexSpec(shape=(), indexer=()),
        IndexSpec(shape=(3,), indexer=()),
        IndexSpec(shape=(3, 4), indexer=()),
    ]),
]

STATIC_INDEXING_GRAD_TESTS = [
    ("OneIntIndex", [
        IndexSpec(shape=(3,), indexer=1),
        IndexSpec(shape=(3, 3), indexer=0),
        IndexSpec(shape=(3, 4, 5), indexer=2),
        IndexSpec(shape=(3,), indexer=-1),
        IndexSpec(shape=(3,), indexer=-2),
    ]),
    ("TwoIntIndices", [
        IndexSpec(shape=(3, 3), indexer=(2, 1)),
        IndexSpec(shape=(3, 4, 5), indexer=(1, 2)),
        IndexSpec(shape=(3, 4, 5), indexer=(-1, 2)),
    ]),
    ("ThreeIntIndices", [IndexSpec((3, 4, 5), indexer=(1, 2, 3))]),
    ("OneSliceIndex", [
        IndexSpec(shape=(5,), indexer=slice(1, 3)),
        IndexSpec(shape=(5,), indexer=slice(1, -1)),
        IndexSpec(shape=(5,), indexer=slice(None, -1)),
        IndexSpec(shape=(5,), indexer=slice(None, None, None)),
        IndexSpec(shape=(5, 4), indexer=slice(1, 3)),
        IndexSpec(shape=(5, 4), indexer=slice(1, None)),
        IndexSpec(shape=(5, 4), indexer=slice(None, 3)),
        IndexSpec(shape=(5, 4), indexer=slice(-3, None)),
    ]),
    ("TwoSliceIndices", [
        IndexSpec(shape=(5, 4), indexer=(slice(1, 3), slice(0, 2))),
        IndexSpec(shape=(5, 4), indexer=(slice(1, None), slice(None, 2))),
        IndexSpec(shape=(5, 4, 3), indexer=(slice(1, 3), slice(0, 2))),
        IndexSpec(shape=(5, 4, 3), indexer=(slice(1, 3), slice(0, None))),
        IndexSpec(shape=(5, 4, 3), indexer=(slice(1, None), slice(0, 2))),
    ]),
    ("OneColonIndex", [
        IndexSpec(shape=(3,), indexer=slice(None)),
        IndexSpec(shape=(3, 4), indexer=slice(None)),
    ]),
    ("MultipleColonIndices", [
        IndexSpec(shape=(3, 4), indexer=(slice(None), slice(None))),
        IndexSpec(shape=(3, 4, 5), indexer=(slice(None), slice(None))),
    ]),
    ("MixedSliceIndices", [
        IndexSpec(shape=(5, 4), indexer=(slice(None), slice(0, 2))),
        IndexSpec(shape=(5, 4), indexer=(1, slice(None))),
    ]),
    ("EllipsisIndex", [
        IndexSpec(shape=(3,), indexer=Ellipsis),
        IndexSpec(shape=(3, 4), indexer=Ellipsis),
        IndexSpec(shape=(3, 4, 5), indexer=(0, Ellipsis)),
        IndexSpec(shape=(3, 4, 5), indexer=(Ellipsis, 2, 3)),
    ]),
    ("NoneIndex", [
        IndexSpec(shape=(), indexer=None),
        IndexSpec(shape=(), indexer=(None, None)),
        IndexSpec(shape=(), indexer=(Ellipsis, None)),
        IndexSpec(shape=(3,), indexer=None),
        IndexSpec(shape=(3, 4), indexer=None),
        IndexSpec(shape=(3, 4), indexer=(Ellipsis, None)),
        IndexSpec(shape=(3, 4), indexer=(0, None, Ellipsis)),
        IndexSpec(shape=(3, 4, 5), indexer=(1, None, Ellipsis)),
    ]),
    # TODO(mattjj): these fail for uninteresting dtype reasons
    # ("EmptyIndex",
    #  [IndexSpec(shape=(), indexer=()),
    #   IndexSpec(shape=(3,), indexer=()),
    #   IndexSpec(shape=(3, 4), indexer=()),
    #   ]),
]

ADVANCED_INDEXING_TESTS = [
    ("One1DIntArrayIndex",
     [IndexSpec(shape=(3,), indexer=onp.array([0, 1])),
     IndexSpec(shape=(3, 3), indexer=onp.array([1, 2, 1])),
     IndexSpec(shape=(3, 4, 5), indexer=onp.array([0, 2, 0, 1])),
     IndexSpec(shape=(3,), indexer=onp.array([-1, 1])),
     IndexSpec(shape=(3,), indexer=onp.array([-2, -1])),
     IndexSpec(shape=(0,), indexer=onp.array([], dtype=onp.int32)),
     ]),
    ("One2DIntArrayIndex",
     [IndexSpec(shape=(3,), indexer=onp.array([[0, 0]])),
     IndexSpec(shape=(3, 3), indexer=onp.array([[1, 2, 1],
                                                [0, 1, -1]])),
     IndexSpec(shape=(3, 4, 5), indexer=onp.array([[0, 2, 0, 1],
                                                   [-1, -2, 1, 0]])),
     ]),
    ("Two1DIntArrayIndicesNoBroadcasting",
     [IndexSpec(shape=(3, 3), indexer=(onp.array([0, 1]),
                                       onp.array([1, 2]))),
     IndexSpec(shape=(3, 4, 5), indexer=(onp.array([0, 2, 0, 1]),
                                         onp.array([-1, 0, -1, 2]))),
     ]),
    ("Two1DIntArrayIndicesWithBroadcasting",
     [IndexSpec(shape=(3, 3), indexer=(onp.array([[0, 1]]),
                                       onp.array([1, 2]))),
     IndexSpec(shape=(3, 4, 5), indexer=(onp.array([[0, 2, 0, 1]]),
                                         onp.array([-1, 0, -1, 2]))),
     ]),
    ("TupleOfListsOfPythonInts",
     [IndexSpec(shape=(3, 4, 5), indexer=([0, 1])),
     IndexSpec(shape=(3, 4, 5), indexer=([[0], [-1]], [[2, 3, 0, 3]])),
     ]),
    ("TupleOfPythonIntsAndIntArrays",
     [IndexSpec(shape=(3, 4, 5), indexer=(0, onp.array([0, 1]))),
     IndexSpec(shape=(3, 4, 5), indexer=(0, 1,
                                         onp.array([[2, 3, 0, 3]]))),
     ]),
    ("TupleOfListsOfPythonIntsAndIntArrays",
     [IndexSpec(shape=(3, 4, 5), indexer=([0, 1], onp.array([0]))),
     IndexSpec(shape=(3, 4, 5), indexer=([[0], [-1]],
                                         onp.array([[2, 3, 0, 3]]))),
     ]),
]

ADVANCED_INDEXING_TESTS_NO_REPEATS = [
    ("One1DIntArrayIndex",
     [IndexSpec(shape=(3,), indexer=onp.array([0, 1])),
      IndexSpec(shape=(3, 3), indexer=onp.array([1, 2, 0])),
      IndexSpec(shape=(3, 4, 5), indexer=onp.array([0, 2, 1])),
      IndexSpec(shape=(3,), indexer=onp.array([-1, 1])),
      IndexSpec(shape=(3,), indexer=onp.array([-2, -1])),
      # Fails with a TF/XLA error.
      # IndexSpec(shape=(0,), indexer=onp.array([], dtype=onp.int32)),
     ]),
    ("One2DIntArrayIndex",
     [IndexSpec(shape=(3,), indexer=onp.array([[0, 1]])),
      IndexSpec(shape=(6, 6), indexer=onp.array([[1, 2, 0],
                                                 [3, 4, -1]])),
     ]),
    ("Two1DIntArrayIndicesNoBroadcasting",
     [IndexSpec(shape=(3, 3), indexer=(onp.array([0, 1]),
                                       onp.array([1, 2]))),
      IndexSpec(shape=(4, 5, 6), indexer=(onp.array([0, 2, 1, 3]),
                                          onp.array([-1, 0, -2, 1]))),
     ]),
    ("Two1DIntArrayIndicesWithBroadcasting",
     [IndexSpec(shape=(3, 3), indexer=(onp.array([[0, 1]]),
                                       onp.array([1, 2]))),
      IndexSpec(shape=(4, 5, 6), indexer=(onp.array([[0, 2, -1, 1]]),
                                          onp.array([-1, 0, -2, 2]))),
     ]),
    ("TupleOfListsOfPythonInts",
     [IndexSpec(shape=(3, 4, 5), indexer=([0, 1])),
      IndexSpec(shape=(3, 4, 5), indexer=([[0], [-1]], [[2, 3, 0]])),
     ]),
    ("TupleOfPythonIntsAndIntArrays",
     [IndexSpec(shape=(3, 4, 5), indexer=(0, onp.array([0, 1]))),
      IndexSpec(shape=(3, 4, 5), indexer=(0, 1,
                                          onp.array([[2, 3, 0]]))),
     ]),
    ("TupleOfListsOfPythonIntsAndIntArrays",
     [IndexSpec(shape=(3, 4, 5), indexer=([0, 1], onp.array([0]))),
      IndexSpec(shape=(3, 4, 5), indexer=([[0], [-1]],
                                          onp.array([[2, 3, 0]]))),
     ]),
]

MIXED_ADVANCED_INDEXING_TESTS_NO_REPEATS = [
    ("SlicesAndOneIntArrayIndex",
     [IndexSpec(shape=(2, 3), indexer=(onp.array([0, 1]), slice(1, 2))),
     IndexSpec(shape=(2, 3), indexer=(slice(0, 2),
                                      onp.array([0, 2]))),
     IndexSpec(shape=(3, 4, 5), indexer=(Ellipsis,
                                         onp.array([0, 2]),
                                         slice(None))),
     IndexSpec(shape=(3, 4, 5), indexer=(Ellipsis,
                                         onp.array([[0, 2], [1, 3]]),
                                         slice(None))),
     ]),
    ("SlicesAndTwoIntArrayIndices",
     [IndexSpec(shape=(3, 4, 5), indexer=(Ellipsis,
                                          onp.array([0, 2]),
                                          onp.array([-1, 2]))),
     IndexSpec(shape=(3, 4, 5), indexer=(onp.array([0, 2]),
                                         Ellipsis,
                                         onp.array([-1, 2]))),
     IndexSpec(shape=(3, 4, 5), indexer=(onp.array([0, 2]),
                                         onp.array([-1, 2]),
                                         Ellipsis)),
     IndexSpec(shape=(3, 4, 5), indexer=(onp.array([0, 2]),
                                         onp.array([-1, 2]),
                                         slice(1, 3))),
     IndexSpec(shape=(3, 4, 5), indexer=(onp.array([0, 2]),
                                         slice(1, 3),
                                         onp.array([-1, 2]))),
     IndexSpec(shape=(3, 4, 5), indexer=(onp.array([0, 2, -2]),
                                         slice(None, None, 2),
                                         onp.array([-1, 2, 1]))),
     ]),
    ("NonesAndIntArrayIndices",
     [IndexSpec(shape=(3, 4, 5), indexer=(onp.array([0, 2]),
                                          None,
                                          onp.array([-1, 2]))),
     IndexSpec(shape=(3, 4, 5), indexer=(onp.array([0, 2]),
                                         None,
                                         None,
                                         onp.array([-1, 2]))),
     IndexSpec(shape=(3, 4, 5), indexer=(Ellipsis,
                                         onp.array([0, 2]),
                                         None,
                                         None,
                                         onp.array([-1, 2]))),
     ]),
    ("IntArrayWithInt32Type",
     [IndexSpec(shape=(3, 4), indexer=(Ellipsis, onp.array(1, dtype=onp.int32)))
     ]),
]

MIXED_ADVANCED_INDEXING_TESTS = MIXED_ADVANCED_INDEXING_TESTS_NO_REPEATS + [
    ("SlicesAndOneIntArrayIndex",
     [
     IndexSpec(shape=(3, 4, 5), indexer=(Ellipsis,
                                         onp.array([[0, 2], [1, 1]]),
                                         slice(None))),
     ]),
    ("SlicesAndTwoIntArrayIndices",
     [IndexSpec(shape=(3, 4, 5), indexer=(onp.array([0, 2, -2]),
                                         slice(None, None, 2),
                                         onp.array([-1, 2, -1]))),
      IndexSpec(shape=(3, 4, 5), indexer=(onp.array([[0, 2], [2, 0]]),
                                          Ellipsis,
                                          onp.array([[1, 0], [1, 0]]))),
     ]),]


def dynamic_slice_reference(operand, start_indices, slice_sizes):
  # [FUNCTION DOC]
  # HELPER: pure numpy reference implementation used to check the dynamic slice extension.
  # Inputs: operand is a numpy array, start_indices the per-axis start positions, slice_sizes
  # the requested per-axis sizes. Returns a new zero-initialised array of shape slice_sizes
  # and operand's dtype, filled with the in-bounds part of the requested window (the window
  # is clipped at the array end, the remainder stays zero). No side effects.
  out = onp.zeros(slice_sizes, dtype=operand.dtype)
  idx = tuple(slice(start, start+size)
              for start, size in zip(start_indices, slice_sizes))
  section = operand[idx]
  out[tuple(slice(None, stop) for stop in section.shape)] = section
  return out


def dynamic_update_slice_reference(operand, update, start_indices):
  # [FUNCTION DOC]
  # HELPER: pure numpy reference implementation used to check the dynamic update slice
  # extension. Returns a copy of operand in which the block starting at start_indices and
  # having the shape of update is overwritten with update. The operand is not mutated.
  # Raises a numpy shape error if the block does not fit in the operand.
  slices = tuple(map(
      slice, start_indices, onp.add(start_indices, update.shape)))
  updated_operand = onp.copy(operand)
  updated_operand[slices] = update
  return updated_operand


class IndexingTest(jtu.TestCase):
  # [CLASS DOC]
  # Test case covering read-only indexing of tensors created through the numpy wrapper:
  # static and dynamic slicing, integer, advanced (array) and boolean indexing, error cases,
  # and the dynamic slice extension. Each test compares against plain numpy results and
  # (for parameterized ones) also checks graph compilation. Inherits the numpy comparison
  # and compile-and-check helpers from the shared test utility base class.
  """Tests for Numpy indexing translation rules."""

  @parameterized.named_parameters(jtu.cases_from_list({
      "testcase_name": "{}_inshape={}_indexer={}".format(
          name, jtu.format_shape_dtype_string( shape, dtype), indexer),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory, "indexer": indexer
  } for name, index_specs in STATIC_INDEXING_TESTS
    for shape, indexer in index_specs
    for dtype in all_dtypes
    for rng_factory in [jtu.rand_default]))
  def testStaticIndexing(self, shape, dtype, rng_factory, indexer):
    # [FUNCTION DOC]
    # Parameterized over every static indexing case, every dtype and the default random
    # generator factory. Builds a random array of the given shape and dtype, indexes it with
    # the constant indexer (integers, slices, ellipsis, new axes, empty tuple) both in numpy
    # and on a tensor made by the wrapper, and asserts equal results including dtypes. Then
    # asserts the tensor version also compiles and agrees, with incomplete input shapes.
    # TODO(rohanj): Revisit passing in self.rng() to this to customize further.
    # This would need updating lax_numpy_test as well.
    rng = rng_factory()
    args_maker = lambda: [rng(shape, dtype)]
    onp_fun = lambda x: x[indexer]
    jnp_fun = lambda x: tnp.asarray(x)[indexer]
    self._CheckAgainstNumpy(onp_fun, jnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True,
                          check_incomplete_shape=True)

  def _ReplaceSlicesWithTuples(self, idx):
    # [FUNCTION DOC]
    # HELPER: prepares indexers so slice bounds can be passed as ordinary runtime arguments.
    # Input: idx, a slice, an integer, or a possibly nested tuple or list of those. Returns a
    # pair: the structure with each slice replaced by a (start, stop, step) triple whose
    # missing (nothing) entries became zero, and a function that rebuilds the original
    # structure from such a triple structure, restoring nothing at the replaced positions.
    # Non-slice leaves are returned unchanged. Empty tuples or lists are treated as leaves.
    """Helper method to replace slices with tuples for dynamic indexing args."""
    if isinstance(idx, slice):
      triple = idx.start, idx.stop, idx.step
      isnone = [i for i, elt in enumerate(triple) if elt is None]
      zeros = itertools.repeat(0)
      nones = itertools.repeat(None)
      out = subvals(triple, zip(isnone, zeros))
      return out, lambda out: slice(*subvals(out, zip(isnone, nones)))
    elif isinstance(idx, (tuple, list)) and idx:
      t = type(idx)
      elts, packs = zip(*map(self._ReplaceSlicesWithTuples, idx))
      return elts, lambda elts: t((pack(i) for pack, i in zip(packs, elts)))
    else:
      return idx, lambda x: x

  @parameterized.named_parameters(
      {"testcase_name": "{}_inshape={}_indexer={}"
       .format(name, jtu.format_shape_dtype_string(shape, dtype), indexer),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory, "indexer": indexer}
      for name, index_specs in [
          ("OneSliceIndex",
           [IndexSpec(shape=(5,), indexer=slice(1, 3)),
            IndexSpec(shape=(5, 4), indexer=slice(1, 3))]),
          ("TwoSliceIndices",
           [IndexSpec(shape=(5, 4), indexer=(slice(1, 3), slice(0, 2))),
            IndexSpec(shape=(5, 4, 3), indexer=(slice(1, 3), slice(0, 2)))]),
          ("NonUnitStrides", [
              IndexSpec(shape=(3,), indexer=slice(None, None, -1)),
              IndexSpec(shape=(3, 3), indexer=slice(0, 3, -2)),
              IndexSpec(shape=(3, 4, 5), indexer=slice(0, 4, 2))
          ]),
          ("OnlyStartOrStopDynamic", [
              IndexSpec(shape=(5, 4), indexer=(slice(None, 3), slice(0, 2))),
              IndexSpec(shape=(5, 4, 3), indexer=(slice(1, 3), slice(0, None)))
          ]),
      ]
      for shape, indexer in index_specs
      for dtype in all_dtypes
      for rng_factory in [jtu.rand_default])
  def testDynamicIndexingWithSlices(self, shape, dtype, rng_factory, indexer):
    # [FUNCTION DOC]
    # Parameterized over slice-based indexers (single and double slices, negative and non-unit
    # strides) and dtypes. The slices are converted to triples passed as runtime arguments of
    # the function under test, then rebuilt inside it. Asserts that indexing a tensor matches
    # numpy and that the compiled version agrees (with eager shape evaluation and the forced
    # compile check turned off because of compile-time-constant requirements).
    rng = rng_factory()
    unpacked_indexer, pack_indexer = self._ReplaceSlicesWithTuples(indexer)

    def onp_fun(x, unpacked_indexer):
      # [FUNCTION DOC]
      # HELPER: function under test. Rebuilds the indexer from its triple form and returns
      # the indexed value of x. Works on numpy arrays or tensors; no side effects.
      indexer = pack_indexer(unpacked_indexer)
      return x[indexer]

    jnp_fun = lambda x, idx: onp_fun(tnp.asarray(x), idx)

    args_maker = lambda: [rng(shape, dtype), unpacked_indexer]
    self._CheckAgainstNumpy(onp_fun, jnp_fun, args_maker, check_dtypes=True)
    # TODO(wangpeng): check_xla_forced_compile is turned off because some
    # compile-time-constant requirements are violated. Investigate and turn it
    # on.
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True,
                          check_eval_on_shapes=False,
                          check_incomplete_shape=True,
                          check_xla_forced_compile=False)

  @parameterized.named_parameters(
      {"testcase_name": "{}_inshape={}_indexer={}"
       .format(name, jtu.format_shape_dtype_string(shape, dtype), indexer),
       "shape": shape, "dtype": dtype, "rng_factory": rng_factory, "indexer": indexer}
      for name, index_specs in [
          ("OneIntIndex",
           [IndexSpec(shape=(3,), indexer=1),
            IndexSpec(shape=(3, 3), indexer=0),
            IndexSpec(shape=(3, 4, 5), indexer=2),
            IndexSpec(shape=(3,), indexer=-1),
            IndexSpec(shape=(3,), indexer=-2)]),
          ("TwoIntIndices",
           [IndexSpec(shape=(3, 3), indexer=(2, 1)),
            IndexSpec(shape=(3, 4, 5), indexer=(1, 2)),
            IndexSpec(shape=(3, 4, 5), indexer=(-1, 2))]),
          ("ThreeIntIndices",
           [IndexSpec((3, 4, 5), indexer=(1, 2, 3))]),
      ]
      for shape, indexer in index_specs
      for dtype in all_dtypes
      for rng_factory in [jtu.rand_default])
  def testDynamicIndexingWithIntegers(self, shape, dtype, rng_factory, indexer):
    # [FUNCTION DOC]
    # Parameterized over integer and integer-tuple indexers and dtypes. The indexer is passed
    # as a runtime argument (not a constant) to the function under test. Asserts that the
    # tensor result matches numpy and that the compiled function agrees, with incomplete
    # input shapes allowed.
    # TODO(rohanj): Revisit passing in self.rng() to this to customize further.
    # This would need updating lax_numpy_test as well.
    rng = rng_factory()
    unpacked_indexer, pack_indexer = self._ReplaceSlicesWithTuples(indexer)

    def onp_fun(x, unpacked_indexer):
      # [FUNCTION DOC]
      # HELPER: function under test. Rebuilds the indexer from its argument form and returns
      # x indexed with it. Works on numpy arrays or tensors; no side effects.
      indexer = pack_indexer(unpacked_indexer)
      return x[indexer]

    jnp_fun = lambda x, idx: onp_fun(tnp.asarray(x), idx)

    args_maker = lambda: [rng(shape, dtype), unpacked_indexer]
    self._CheckAgainstNumpy(onp_fun, jnp_fun, args_maker, check_dtypes=True)
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True,
                          check_incomplete_shape=True)

  @parameterized.named_parameters(
      {"testcase_name": "_{}_inshape={}_indexer={}"  # pylint: disable=g-complex-comprehension
       .format(name, jtu.format_shape_dtype_string(shape, dtype), indexer),
       "name": name, "shape": shape, "dtype": dtype, "rng_factory": rng_factory,
       "indexer": indexer}
      for name, index_specs in ADVANCED_INDEXING_TESTS
      for shape, indexer in index_specs
      for dtype in all_dtypes
      for rng_factory in [jtu.rand_default])
  def testAdvancedIntegerIndexing(self, name, shape, dtype, rng_factory,
                                  indexer):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the forced-compile check is meant to be skipped for the case named
    # "ListOfPythonIntsAndIntArrays", but no case in the advanced indexing table has that name,
    # so the check is always enabled and the intended exemption never applies.
    # Parameterized over integer-array indexing cases and dtypes. The indexer is passed as a
    # runtime argument. Asserts that indexing a tensor equals numpy and that the compiled
    # version agrees with incomplete shapes allowed.
    rng = rng_factory()
    args_maker = lambda: [rng(shape, dtype), indexer]
    onp_fun = lambda x, idx: x[idx]
    jnp_fun = lambda x, idx: onp_fun(tnp.asarray(x), idx)

    self._CheckAgainstNumpy(onp_fun, jnp_fun, args_maker, check_dtypes=True)
    # TODO(wangpeng): check_xla_forced_compile is turned off for
    # ListOfPythonIntsAndIntArrays because it throws "The number of output
    # elements has to equal to number of input elements that are sliced when
    # input indices are not constant". Investigate and turn it on.
    check_xla = (name != "ListOfPythonIntsAndIntArrays")
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True,
                          check_incomplete_shape=True,
                          check_xla_forced_compile=check_xla)

  @parameterized.named_parameters(
      {"testcase_name": "_{}_inshape={}_indexer={}"  # pylint: disable=g-complex-comprehension
       .format(name, jtu.format_shape_dtype_string(shape, dtype), indexer),
       "name": name, "shape": shape, "dtype": dtype, "rng_factory": rng_factory,
       "indexer": indexer}
      for name, index_specs in MIXED_ADVANCED_INDEXING_TESTS
      for shape, indexer in index_specs
      for dtype in all_dtypes
      for rng_factory in [jtu.rand_default])
  def testMixedAdvancedIntegerIndexing(self, name, shape, dtype, rng_factory,
                                       indexer):
    # [FUNCTION DOC]
    # Parameterized over indexers mixing integer arrays with slices, ellipsis and new axes.
    # Only array elements are passed as runtime arguments (other elements replaced by empty
    # placeholders) and the full indexer is rebuilt inside the function under test. Asserts
    # the tensor result equals numpy and the compiled version agrees; the forced-compile
    # check is skipped for the case named IntArrayWithInt32Type.
    rng = rng_factory()
    indexer_with_dummies = [e if isinstance(e, onp.ndarray) else ()
                            for e in indexer]
    substitutes = [(i, e) for i, e in enumerate(indexer)
                   if not isinstance(e, onp.ndarray)]
    args_maker = lambda: [rng(shape, dtype), indexer_with_dummies]

    def np_fun(x, indexer_with_dummies):
      # [FUNCTION DOC]
      # HELPER: function under test. Restores the non-array indexer elements at their
      # recorded positions, rebuilds an indexer of the original container type and returns
      # x indexed with it. No side effects.
      idx = type(indexer)(subvals(indexer_with_dummies, substitutes))
      return x[idx]

    jnp_fun = lambda x, idx: np_fun(tnp.asarray(x), idx)

    self._CheckAgainstNumpy(np_fun, jnp_fun, args_maker, check_dtypes=True)
    # TODO(wangpeng): check_xla_forced_compile is turned off for
    # IntArrayWithInt32Type because it throws "The number of output elements has
    # to equal to number of input elements that are sliced when input indices
    # are not constant". Investigate and turn it on.
    check_xla = (name != "IntArrayWithInt32Type")
    self._CompileAndCheck(jnp_fun, args_maker, check_dtypes=True,
                          check_incomplete_shape=True,
                          check_xla_forced_compile=check_xla)

  def testAdvancedIndexingManually(self):
    # [FUNCTION DOC]
    # Builds a random three-dimensional array and an integer index array, and for three
    # indexing expressions (index array on the middle axis, index arrays combined with
    # ellipsis and new axis, and a mix of ellipsis, a column-shaped index and new axis)
    # asserts that the plain result equals the result of the jit-compiled extension version,
    # including dtypes.
    x = onp.random.RandomState(0).randn(3, 4, 5)
    index_array = onp.array([0, 2, -1, 0])

    op = lambda x, index_array: x[..., index_array, :]
    cop = nje.jit(op)

    a1 = op(x, index_array)
    a2 = cop(x, index_array)

    self.assertAllClose(a1, a2, check_dtypes=True)

    op = lambda x, index_array: x[..., index_array, :, index_array, None]
    cop = nje.jit(op)

    a1 = op(x, index_array)
    a2 = cop(x, index_array)

    self.assertAllClose(a1, a2, check_dtypes=True)

    op = lambda x, index_array: x[index_array, ..., index_array[:, None], None]
    cop = nje.jit(op)

    a1 = op(x, index_array)
    a2 = cop(x, index_array)

    self.assertAllClose(a1, a2, check_dtypes=True)

  # Note that we don't currently allow __iter__ in graph mode. So this test only
  # iterates over eager tensor.
  def testUnpacking(self):
    # [FUNCTION DOC]
    # Asserts that iterating (unpacking into three names) over an eager tensor produced by the
    # wrapper gives the same sum as for a numpy array of three elements. Only eager tensors
    # are iterated because iteration is not allowed in graph mode.

    def foo(x):
      # [FUNCTION DOC]
      # HELPER: unpacks a three-element sequence x and returns the sum of its elements.
      # Raises a value error if x does not have exactly three elements.
      a, b, c = x
      return a + b + c

    a1 = foo(onp.arange(3))
    a2 = foo(tnp.arange(3))

    self.assertAllClose(a1, a2, check_dtypes=True)

  def testBooleanIndexingArray1D(self):
    # [FUNCTION DOC]
    # Indexes a tensor of three consecutive integers with a numpy boolean mask and asserts the
    # result equals numpy's masked selection (dtype not checked).
    idx = onp.array([True, True, False])
    x = tnp.asarray(onp.arange(3))
    ans = x[idx]
    expected = onp.arange(3)[idx]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testBooleanIndexingList1D(self):
    # [FUNCTION DOC]
    # Same as the one-dimensional array mask test but the boolean mask is a plain list of
    # booleans; asserts the tensor result equals numpy's (dtype not checked).
    idx = [True, True, False]
    x = tnp.asarray(onp.arange(3))
    ans = x[idx]
    expected = onp.arange(3)[idx]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testBooleanIndexingArray2DBroadcast(self):
    # [FUNCTION DOC]
    # Selects rows of a 4 by 2 tensor using a one-dimensional numpy boolean mask over the
    # first axis; asserts the result equals numpy's row selection (dtype not checked).
    idx = onp.array([True, True, False, True])
    x = onp.arange(8).reshape(4, 2)
    ans = tnp.asarray(x)[idx]
    expected = x[idx]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testBooleanIndexingList2DBroadcast(self):
    # [FUNCTION DOC]
    # Same as the two-dimensional broadcast mask test but the row mask is a plain list of
    # booleans; asserts the result equals numpy's row selection (dtype not checked).
    idx = [True, True, False, True]
    x = onp.arange(8).reshape(4, 2)
    ans = tnp.asarray(x)[idx]
    expected = x[idx]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testBooleanIndexingArray2D(self):
    # [FUNCTION DOC]
    # Indexes a 4 by 2 tensor with a full-shape two-dimensional boolean mask and asserts the
    # flattened selection equals numpy's (dtype not checked).
    idx = onp.array([[True, False],
                     [False, True],
                     [False, False],
                     [True, True]])
    x = onp.arange(8).reshape(4, 2)
    ans = tnp.asarray(x)[idx]
    expected = x[idx]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testBooleanIndexingDynamicShape(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the roles are swapped relative to the other tests: the numpy array is
    # indexed to produce the "ans" value, and the tensor-indexed result is named "expected";
    # the comparison is symmetric so it still checks equality, but the naming is misleading.
    # Indexes a numpy zero vector with a boolean mask and a tensor copy with the same mask,
    # asserting both selections are equal including dtype.
    x = onp.zeros(3)
    i = onp.array([True, True, False])
    ans = x[i]
    expected = tnp.asarray(x)[i]
    self.assertAllClose(ans, expected, check_dtypes=True)

  def testIssue187(self):
    # [FUNCTION DOC]
    # Regression test for an old issue. First asserts (implicitly, by not raising) that
    # indexing a tensor of ones with two integer lists does not crash. Then indexes a numpy
    # array with two integer lists inside a jit-compiled function and asserts equality with
    # plain numpy (dtype not checked).
    x = tnp.ones((5, 5))
    x[[0, 2, 4], [0, 2, 4]]  # doesn't crash

    x = onp.arange(25).reshape((5, 5))
    ans = nje.jit(lambda x: x[[0, 2, 4], [0, 2, 4]])(x)
    expected = x[[0, 2, 4], [0, 2, 4]]
    self.assertAllClose(ans, expected, check_dtypes=False)

  # TODO(agarwal): Fix this use case.
  @jtu.disable
  def testIndexingEmptyDimension(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test is disabled by the disable decorator (marked as a to-do to fix),
    # so it never runs; it also refers to JAX behaviour and expects out-of-bounds errors that
    # the tensor implementation may not raise.
    # Intended scenario: for a tensor with a zero-sized second axis, slicing and new-axis
    # indexing work, while a scalar index into the empty axis should raise an index error
    # (in numpy, directly, and under jit compilation).
    # Issue 2671: XLA error when indexing into dimension of size 0
    x = tnp.ones((2, 0))
    # The following work, even on axis 1 of size 0
    _ = x[0, :] + x[0, None] + x[0, 1:] + x[0, 1:3:2]

    with self.assertRaisesRegex(IndexError,
                                "index .* is out of bounds for axis .* with size 0"):
      _ = onp.ones((2, 0))[0, 0]  # The numpy error
    with self.assertRaisesRegex(IndexError,
                                "index is out of bounds for axis .* with size 0"):
      _ = x[0, 0]  # JAX indexing
    with self.assertRaisesRegex(IndexError,
                                "index is out of bounds for axis .* with size 0"):
      nje.jit(lambda i: x[0, i])(0)  # JAX indexing under jit

  def testBooleanIndexingWithEmptyResult(self):
    # [FUNCTION DOC]
    # Indexes a one-element tensor with an all-false boolean mask (must not crash) and asserts
    # the empty result equals numpy's (dtype not checked).
    # based on a TensorFlow Probability test that started failing after #1623
    x = tnp.array([-1])
    mask = tnp.array([False])
    ans = x[mask]  # doesn't crash

    expected =  onp.array([-1])[onp.array([False])]
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testFloatIndexingError(self):
    # [FUNCTION DOC]
    # Asserts that using a float as an index raises an index error whose message says only
    # integers, slices and similar are valid: first for numpy (to confirm the expected
    # behaviour), then for tensors with a float scalar index, with a tuple containing a float,
    # and inside a jit-compiled function with a runtime float index.
    error_regex = "only integers, slices.*are valid indices"
    # Verify onp behavior
    with self.assertRaisesRegex(IndexError, error_regex):
      _ = onp.zeros((2, 2))[(0, 0.)]
    # Test tnp
    with self.assertRaisesRegex(IndexError, error_regex):
      tnp.zeros(2)[0.]  # pylint: disable=expression-not-assigned
    with self.assertRaisesRegex(IndexError, error_regex):
      tnp.zeros((2, 2))[(0, 0.)]  # pylint: disable=expression-not-assigned
    # Test with jit
    with self.assertRaisesRegex(IndexError, error_regex):
      nje.jit(lambda idx: tnp.zeros((2, 2))[idx])((0, 0.0))

  def testIndexOutOfBounds(self):  # https://github.com/google/jax/issues/2245
    # [FUNCTION DOC]
    # Asserts that slicing a five-element tensor of ones with an end bound beyond its length
    # is clipped and returns the whole tensor (equal values and dtype).
    array = tnp.ones(5)
    self.assertAllClose(array, array[:10], check_dtypes=True)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_start_indices={}_size_indices={}".format(  # pylint: disable=g-complex-comprehension
          jtu.format_shape_dtype_string(shape, dtype),
          start_indices, size_indices),
       "shape": shape, "dtype": dtype, "start_indices": start_indices,
       "size_indices": size_indices, "rng_factory": rng_factory}
      for shape, start_indices, size_indices in [
        [(3,), onp.array((1,)), (1,)],
        [(5, 3), (1, 1), (3, 1)],
        [(5, 3), (1, -2), (3, 1)],
        [(5, 3), onp.array((1, 1)), (3, 1)],
        [(7, 5, 3), onp.array((4, 1, 0)), (2, 0, 1)],
        [(), (), ()],
      ]
      for dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testDynamicSlice(self, shape, dtype, start_indices, size_indices,
                       rng_factory):
    # [FUNCTION DOC]
    # Parameterized over shapes, start positions (including negative ones) and slice sizes.
    # Runs the dynamic slice extension, with start positions passed as a runtime array, through
    # the compile-and-check helper to verify the compiled and eager results agree.
    rng = rng_factory()
    args_maker = lambda: [rng(shape, dtype), onp.array(start_indices)]
    op = lambda x, starts: nje.dynamic_slice(x, starts, size_indices)
    self._CompileAndCheck(op, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_start_indices={}_size_indices={}".format(  # pylint: disable=g-complex-comprehension
          jtu.format_shape_dtype_string(shape, dtype),
          start_indices, size_indices),
       "shape": shape, "dtype": dtype, "start_indices": start_indices,
       "size_indices": size_indices, "rng_factory": rng_factory}
      for shape, start_indices, size_indices in [
        [(3,), (1,), (1,)],
        [(5, 3), (1, 1), (3, 1)],
        [(5, 3), (1, -2), (3, 1)],
        [(7, 5, 3), (4, 1, 0), (2, 0, 1)],
        [(), (), ()],
      ]
      for dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testDynamicSliceAgainstNumpy(self, shape, dtype, start_indices,
                                   size_indices, rng_factory):
    # [FUNCTION DOC]
    # Parameterized over shapes, start positions and slice sizes. Asserts that the dynamic
    # slice extension equals the pure numpy reference implementation of dynamic slicing.
    rng = rng_factory()
    args_maker = lambda: [rng(shape, dtype), onp.array(start_indices)]
    op = lambda x, s: nje.dynamic_slice(x, s, size_indices)
    numpy_op = lambda x, s: dynamic_slice_reference(x, s, size_indices)
    self._CheckAgainstNumpy(numpy_op, op, args_maker)

  def testDynamicSliceInDim(self):
    # [FUNCTION DOC]
    # Takes a random 6 by 7 integer array and asserts that the dynamic slice along one
    # dimension (start 2, size 3, default axis) equals the plain slice of rows 2 to 4.
    rng = jtu.rand_default()
    x = rng((6, 7), onp.int32)
    self.assertAllClose(
        nje.dynamic_slice_in_dim(x, 2, 3), x[2:5], check_dtypes=True
    )


def _broadcastable_shapes(shape):
  # [FUNCTION DOC]
  # HELPER: generates update shapes for the indexed-update tests.
  # Generator that yields, as lists of dimension sizes, every shape that can broadcast to the
  # given shape: each dimension either kept or replaced by one, with leading dimensions also
  # droppable (shorter shapes). Duplicates can be yielded when a dimension is already one.
  """Returns all shapes that broadcast to `shape`."""
  def f(rshape):
    # [FUNCTION DOC]
    # HELPER: recursive generator working on a reversed dimension list. Yields the empty
    # list, then every suffix choice combined with the first dimension kept, and (when that
    # dimension is not one) with it replaced by one.
    yield []
    if rshape:
      for s in f(rshape[1:]):
        yield rshape[0:1] + s
      if rshape[0] != 1:
        for s in f(rshape[1:]):
          yield [1] + s
  for x in f(list(reversed(shape))):
    yield list(reversed(x))


def _update_shape(shape, indexer):
  # [FUNCTION DOC]
  # HELPER: computes the shape of the region selected by indexer in an array of the given
  # shape, by indexing a temporary zero array and returning its shape (a tuple).
  return onp.zeros(shape)[indexer].shape


class UpdateOps(enum.Enum):
  # [CLASS DOC]
  # HELPER: enumeration of the indexed-update operations under test (plain update, add,
  # minimum and maximum; multiplication is commented out). Each member also offers two
  # functions taking the member as first argument: one computing the numpy reference result
  # and one calling the matching extension function. Member values are 0, 1, 3 and 4.
  UPDATE = 0
  ADD = 1
  # MUL = 2
  MIN = 3
  MAX = 4

  def np_fn(op, indexer, x, y):  # pylint: disable=no-self-argument
    # [FUNCTION DOC]
    # HELPER: numpy reference for an indexed update. The first argument is the operation
    # member. Copies array x, then at the indexed region stores y (update), the sum with y,
    # the elementwise minimum, or the elementwise maximum depending on the member. Returns the
    # new array; x is not mutated.
    x = x.copy()
    x[indexer] = {
      UpdateOps.UPDATE: lambda: y,
      UpdateOps.ADD: lambda: x[indexer] + y,
      # UpdateOps.MUL: lambda: x[indexer] * y,
      UpdateOps.MIN: lambda: onp.minimum(x[indexer], y),
      UpdateOps.MAX: lambda: onp.maximum(x[indexer], y),
    }[op]()
    return x

  def tfnp_fn(op, indexer, x, y):  # pylint: disable=no-self-argument
    # [FUNCTION DOC]
    # HELPER: dispatches to the extension indexed-update function matching the operation
    # member (update, add, min or max) and returns its result for array x, indexer and
    # update values y. Raises a key error for a member without an implementation.
    return {
        UpdateOps.UPDATE: nje.index_update,
        UpdateOps.ADD: nje.index_add,
        # UpdateOps.MUL: nje.index_mul,
        UpdateOps.MIN: nje.index_min,
        UpdateOps.MAX: nje.index_max,
    }[op](x, indexer, y)


# a test to workaround b/123559667
def has_non_trivial_stride(indexer):
  # [FUNCTION DOC]
  # HELPER: decides whether forced compilation checks must be skipped (workaround for a
  # known bug). Returns true if any slice found when flattening the indexer structure has a
  # step other than one, minus one or nothing.
  def has(idx):
    # [FUNCTION DOC]
    # HELPER: returns true if idx is a slice with a step other than one, minus one or nothing.
    return isinstance(idx, slice) and idx.step not in (1, -1, None)
  return any(has(idx) for idx in nest.flatten(indexer))


class IndexedUpdateTest(jtu.TestCase):
  # [CLASS DOC]
  # Test case covering the indexed update extension functions (update, add, min, max) and the
  # dynamic update slice functions. Compares against numpy-based references, checks
  # compilation, checks gradients, and uses the update-operation enumeration and indexing
  # case tables defined at module level.

  @parameterized.named_parameters(jtu.cases_from_list({  # pylint: disable=g-complex-comprehension
      "testcase_name": "_{}_{}_{}_{}".format(
          jtu.format_shape_dtype_string(shape, dtype), indexer,
          jtu.format_shape_dtype_string(update_shape, update_dtype), op.name),
      "shape": shape, "dtype": dtype, "rng_factory": rng_factory,
      "indexer": indexer, "update_shape": update_shape,
      "update_dtype": update_dtype, "op": op
  } for name, index_specs in STATIC_INDEXING_TESTS
    for shape, indexer in index_specs
    for op in UpdateOps
    for dtype in (all_dtypes if op == UpdateOps.UPDATE else default_dtypes)
    for update_shape in _broadcastable_shapes(_update_shape(shape, indexer))
    for update_dtype in all_dtypes
    for rng_factory in [jtu.rand_default]))
  def testStaticIndexing(self, shape, dtype, update_shape, update_dtype,
                         rng_factory, indexer, op):
    # [FUNCTION DOC]
    # Parameterized over static indexing cases, update operations, dtypes and every update
    # shape that broadcasts to the indexed region. Asserts the extension update function equals
    # the numpy reference, then that the compiled version agrees. The experimental and forced
    # compile checks are disabled for strided indexers and for slices ending at 8 with step
    # minus one (known compiler errors).
    rng = rng_factory()
    args_maker = lambda: [rng(shape, dtype), rng(update_shape, update_dtype)]
    np_fn = lambda x, y: UpdateOps.np_fn(op, indexer, x, y)
    tfnp_fn = lambda x, y: UpdateOps.tfnp_fn(op, indexer, x, y)
    self._CheckAgainstNumpy(np_fn, tfnp_fn, args_maker)
    # TODO(wangpeng): When indexer is slice(_, 8, -1), XLA throws error "Missing
    # xla_context 0-th output from". Investigate.
    check_xla = (not has_non_trivial_stride(indexer) and  # b/123559667
                 not (isinstance(indexer, slice) and indexer.stop == 8 and
                      indexer.step == -1))
    self._CompileAndCheck(tfnp_fn, args_maker, check_incomplete_shape=True,
                          check_experimental_compile=check_xla,
                          check_xla_forced_compile=check_xla)

  @parameterized.named_parameters(jtu.cases_from_list({  # pylint: disable=g-complex-comprehension
      "testcase_name": "_{}_{}_{}_{}".format(
          jtu.format_shape_dtype_string(shape, dtype), indexer,
          jtu.format_shape_dtype_string(update_shape, update_dtype), op.name),
      "shape": shape, "dtype": dtype, "rng_factory": rng_factory,
      "indexer": indexer, "update_shape": update_shape,
      "update_dtype": update_dtype, "op": op
  } for name, index_specs in ADVANCED_INDEXING_TESTS_NO_REPEATS
    for shape, indexer in index_specs
    for op in UpdateOps
    for dtype in (all_dtypes if op == UpdateOps.UPDATE else default_dtypes)
    for update_shape in _broadcastable_shapes(_update_shape(shape, indexer))
    for update_dtype in all_dtypes
    for rng_factory in [jtu.rand_default]))
  def testAdvancedIndexing(self, shape, dtype, update_shape, update_dtype,
                           rng_factory, indexer, op):
    # [FUNCTION DOC]
    # Same scenario as the static update test but for integer-array indexers without repeated
    # indices (so results are well defined), with all update operations and broadcastable
    # update shapes. Asserts agreement with the numpy reference and with the compiled version
    # (incomplete shapes allowed).
    rng = rng_factory()
    args_maker = lambda: [rng(shape, dtype), rng(update_shape, update_dtype)]
    np_fn = lambda x, y: UpdateOps.np_fn(op, indexer, x, y)
    tfnp_fn = lambda x, y: UpdateOps.tfnp_fn(op, indexer, x, y)
    self._CheckAgainstNumpy(np_fn, tfnp_fn, args_maker)
    self._CompileAndCheck(tfnp_fn, args_maker, check_incomplete_shape=True)

  @parameterized.named_parameters(jtu.cases_from_list({  # pylint: disable=g-complex-comprehension
      "testcase_name": "_{}_{}_{}_{}".format(
          jtu.format_shape_dtype_string(shape, dtype), indexer,
          jtu.format_shape_dtype_string(update_shape, update_dtype), op.name),
      "shape": shape, "dtype": dtype, "rng_factory": rng_factory,
      "indexer": indexer, "update_shape": update_shape,
      "update_dtype": update_dtype, "op": op
  } for name, index_specs in MIXED_ADVANCED_INDEXING_TESTS_NO_REPEATS
    for shape, indexer in index_specs
    for op in UpdateOps
    for dtype in (all_dtypes if op == UpdateOps.UPDATE else default_dtypes)
    for update_shape in _broadcastable_shapes(_update_shape(shape, indexer))
    for update_dtype in all_dtypes
    for rng_factory in [jtu.rand_default]))
  def testMixedAdvancedIndexing(self, shape, dtype, update_shape, update_dtype,
                                rng_factory, indexer, op):
    # [FUNCTION DOC]
    # Same scenario for indexers mixing integer arrays with slices, ellipsis and new axes
    # (without repeated indices). Asserts agreement with the numpy reference, then with the
    # compiled version; experimental and forced compile checks are skipped if the indexer has
    # a non-trivial stride.
    rng = rng_factory()
    args_maker = lambda: [rng(shape, dtype), rng(update_shape, update_dtype)]
    np_fn = lambda x, y: UpdateOps.np_fn(op, indexer, x, y)
    tfnp_fn = lambda x, y: UpdateOps.tfnp_fn(op, indexer, x, y)
    self._CheckAgainstNumpy(np_fn, tfnp_fn, args_maker)
    check_xla = not has_non_trivial_stride(indexer)  # b/123559667
    self._CompileAndCheck(tfnp_fn, args_maker, check_incomplete_shape=True,
                          check_experimental_compile=check_xla,
                          check_xla_forced_compile=check_xla)

  @parameterized.named_parameters(jtu.cases_from_list({  # pylint: disable=g-complex-comprehension
      "testcase_name": "_{}_{}_{}_{}".format(
          jtu.format_shape_dtype_string(shape, dtype), indexer,
          jtu.format_shape_dtype_string(update_shape, update_dtype), op.name),
      "shape": shape, "dtype": dtype, "rng_factory": rng_factory,
      "indexer": indexer, "update_shape": update_shape,
      "update_dtype": update_dtype, "op": op
  } for name, index_specs in STATIC_INDEXING_TESTS
    for shape, indexer in index_specs
    for op in [UpdateOps.ADD, UpdateOps.UPDATE]
    for dtype in float_dtypes
    for update_shape in _broadcastable_shapes(_update_shape(shape, indexer))
    for update_dtype in float_dtypes
    for rng_factory in [jtu.rand_default]))
  def testStaticIndexingGrads(self, shape, dtype, update_shape, update_dtype,
                              rng_factory, indexer, op):
    # [FUNCTION DOC]
    # KNOWN ISSUE: draws its cases from the general static indexing table, not from the
    # dedicated gradient table defined at module level, which stays unused.
    # Parameterized over static indexers, the add and update operations, and float dtypes.
    # Checks numerically (with loose tolerances) that gradients of the update function with
    # respect to the array and the update values are correct.
    rng = rng_factory()
    tfnp_fn = lambda x, y: UpdateOps.tfnp_fn(op, indexer, x, y)
    x = rng(shape, dtype)
    y = rng(update_shape, update_dtype)
    self.check_grads(tfnp_fn, (x, y), rtol=1e-3, atol=1e-3, delta=1.)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_start_indices={}_update_shape={}".format(  # pylint: disable=g-complex-comprehension
          jtu.format_shape_dtype_string(shape, dtype),
          start_indices, update_shape),
       "shape": shape, "dtype": dtype, "start_indices": start_indices,
       "update_shape": update_shape, "rng_factory": rng_factory}
      for shape, start_indices, update_shape in [
        [(3,), (1,), (1,)],
        [(5, 3), (1, 1), (3, 1)],
        [(5, 3), (1, -2), (3, 1)],
        [(7, 5, 3), (4, 1, 0), (2, 0, 1)],
        [(), (), ()],
      ]
      for dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testDynamicUpdateSlice(self, shape, dtype, start_indices, update_shape,
                             rng_factory):
    # [FUNCTION DOC]
    # Parameterized over shapes, start positions and update shapes. Runs the dynamic update
    # slice extension through the compile-and-check helper to verify compiled and eager
    # results agree; incomplete-shape checking is off because the update shape must be known.
    rng = rng_factory()
    def args_maker():
      # [FUNCTION DOC]
      # HELPER: argument factory. Returns a list of a random array, a random update array and
      # the start positions as an array.
      return [rng(shape, dtype), rng(update_shape, dtype),
              onp.array(start_indices)]
    # update's shape must be fully known.
    # TODO(wangpeng): Support turning off check_incomplete_shape for individual
    #   arguments.
    self._CompileAndCheck(
        nje.dynamic_update_slice, args_maker, check_incomplete_shape=False
    )

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_shape={}_start_indices={}_update_shape={}".format(  # pylint: disable=g-complex-comprehension
          jtu.format_shape_dtype_string(shape, dtype),
          start_indices, update_shape),
       "shape": shape, "dtype": dtype, "start_indices": start_indices,
       "update_shape": update_shape, "rng_factory": rng_factory}
      for shape, start_indices, update_shape in [
        [(3,), (1,), (1,)],
        [(5, 3), (1, 1), (3, 1)],
        [(5, 3), (1, -2), (3, 1)],
        [(7, 5, 3), (4, 1, 0), (2, 0, 1)],
        [(), (), ()],
      ]
      for dtype in default_dtypes
      for rng_factory in [jtu.rand_default]))
  def testDynamicUpdateSliceAgainstNumpy(self, shape, dtype, start_indices,
                                         update_shape, rng_factory):
    # [FUNCTION DOC]
    # Parameterized over shapes, start positions and update shapes. Asserts the dynamic update
    # slice extension equals the numpy reference implementation.
    rng = rng_factory()
    def args_maker():
      # [FUNCTION DOC]
      # HELPER: argument factory. Returns a list of a random array, a random update array and
      # the start positions as an array.
      return [rng(shape, dtype), rng(update_shape, dtype),
              onp.array(start_indices)]
    self._CheckAgainstNumpy(
        dynamic_update_slice_reference, nje.dynamic_update_slice, args_maker
    )

  def testDynamicUpdateSliceInDim(self):
    # [FUNCTION DOC]
    # Takes a random 6 by 7 array and a 3 by 7 update, computes the expected result by writing
    # the update into rows 2 to 4 of a copy, and asserts the dynamic update along dimension
    # zero at start 2 equals it, including dtype.
    rng = jtu.rand_default()
    x = rng((6, 7), onp.int32)
    y = rng((3, 7), onp.int32)
    z = x.copy()
    z[2:5] = y
    self.assertAllClose(
        nje.dynamic_update_slice_in_dim(x, y, 2, 0), z, check_dtypes=True
    )


if __name__ == "__main__":
  tf_config.set_soft_device_placement(False)
  absltest.main()
