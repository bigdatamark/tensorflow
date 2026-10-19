# [MODULE DOC]
# Tests for sorting-related operators on XLA-compiled devices: the XLA sort, key-value sort and
# variadic sort ops (with custom comparators, stable and unstable), argsort, top_k and in_top_k.
# Also holds a small benchmark class that times stable and unstable variadic sorts, with and
# without XLA JIT. Results are compared against numpy reference implementations.
# No import-time side effects other than defining a module-level list of key dtypes.
# Run as a test binary via the test main entry point.
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
# ==============================================================================
"""Tests for sorting operators."""

import unittest

from absl.testing import parameterized
import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.compiler.tf2xla.python import xla
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import sort_ops
from tensorflow.python.platform import test

ALL_KEY_TYPES = [
    dtypes.bfloat16.as_numpy_dtype, np.float16, np.float32, np.float64,
    np.int32, np.uint32, np.int16, np.uint16, np.int8, np.uint8
]


class XlaSortOpTest(xla_test.XLATestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering sort, key/value sort, variadic sort, argsort, top_k and in_top_k on the XLA
  # test device. Combines the XLA test base class (device scope, session, supported numeric types)
  # with parameterized testing. Has no instance state of its own.

  def _assertOpOutputMatchesExpected(self, op, args, expected):
    # [FUNCTION DOC]
    # HELPER: shared assertion helper for the sort tests in this class.
    # Builds a placeholder for each numpy array in args (matching dtype and shape), calls the
    # given op on the placeholders inside the XLA test device scope, runs it in a session feeding
    # the arrays, and checks each result against the matching entry of expected with a relative
    # tolerance of 1e-3. A single tensor output is wrapped into a list. Returns nothing; fails
    # via test assertions.
    """Tests that op(*args) == expected."""
    with self.session() as session:
      with self.test_scope():
        placeholders = [
            array_ops.placeholder(dtypes.as_dtype(arg.dtype), arg.shape)
            for arg in args
        ]
        feeds = {placeholders[i]: args[i] for i in range(0, len(args))}
        output = op(*placeholders)
        if isinstance(output, tensor.Tensor):
          output = [output]

      results = session.run(output, feeds)
      for result, v in zip(results, expected):
        self.assertAllClose(v, result, rtol=1e-3)

  def _shuffled_arange(self, shape, dtype):
    # [FUNCTION DOC]
    # HELPER: generates distinct test input data for the sort tests.
    # Returns a numpy array of the given shape and dtype holding the integers from zero up to the
    # element count minus one, randomly shuffled (uses the global numpy random state).
    x = np.arange(np.prod(shape), dtype=dtype)
    np.random.shuffle(x)
    return x.reshape(shape)

  def _supported_key_types(self):
    # [FUNCTION DOC]
    # HELPER: selects which key dtypes to exercise for the current test device.
    # Returns the set of dtypes from the module-level key type list that are also in the device's
    # supported numeric types. Asserts the set is not empty.
    supported_key_types = set(ALL_KEY_TYPES)
    res = supported_key_types.intersection(self.numeric_types)
    assert res
    return res

  def testSort(self):
    # [FUNCTION DOC]
    # For each supported key type, sorts a shuffled range of 101 values with the XLA sort op and
    # expects the ascending range as the result.
    for dtype in self._supported_key_types():
      x = self._shuffled_arange((101,), dtype)
      self._assertOpOutputMatchesExpected(
          xla.sort, [x], expected=[np.arange(101, dtype=dtype)])

  def testKeyValueSort(self):
    # [FUNCTION DOC]
    # For every pair of supported key and value types (skipping any pair involving unsigned 8-bit,
    # which is known to fail for unexplained reasons), sorts a shuffled range of 101 keys together
    # with values equal to the negated keys using the XLA key-value sort op. Expects the keys in
    # ascending order and the values as the negated ascending range.
    for key_type in self._supported_key_types():
      for value_type in self._supported_key_types():
        if key_type == np.uint8 or value_type == np.uint8:
          # I do not understand why the test fails on uint8. We plan to
          # deprecate xla.key_value_sort in favor of xla.variadic_sort anyway.
          continue
        x = self._shuffled_arange((101,), key_type)
        y = (-x).astype(value_type)
        self._assertOpOutputMatchesExpected(
            xla.key_value_sort, [x, y],
            expected=[
                np.arange(101, dtype=key_type),
                -np.arange(101, dtype=value_type)
            ])

  # Flip is the only reliable way to get a descending sort across any dimension.
  # 1. -np.sort(-x) doesn't work with unsigned integers.
  # 2. np.sort(x, axis=a)[::-1] is not generic over axis wher reversing array
  # 3. x.argsort() either requires "-1" (first option) or flip, so the same.
  def _descendingSort(self, x, dimension):
    # [FUNCTION DOC]
    # HELPER: numpy reference for a descending sort.
    # Sorts array x ascending along the given dimension and then flips it along that dimension.
    # Returns the resulting array. Works for unsigned integers, unlike negating before sorting.
    b = np.sort(x, axis=dimension)
    return np.flip(b, axis=dimension)

  @parameterized.parameters(0, 1, 2)
  def testMisleadingComparator(self, dimension):
    # [FUNCTION DOC]
    # Parameterized over the sort dimension (0, 1, 2). For each supported key type, sorts a
    # shuffled array of shape 4x3x4 with the variadic sort op using a custom comparator that
    # implements "greater than" by swapping operands of a less-than test. Expects a descending
    # sort along the dimension.
    shape = (4, 3, 4)
    for key_type in self._supported_key_types():
      x = self._shuffled_arange(shape, key_type)
      expected = self._descendingSort(x, dimension)

      # pylint: disable=cell-var-from-loop
      @function.Defun(key_type, key_type)
      def compare_gt(x1, x2):
        # [FUNCTION DOC]
        # HELPER: comparator for the variadic sort. Returns true when the first element is
        # greater than the second, written via a reversed less-than comparison.
        return x2 < x1  # "greater than" with misleading "<" sign

      def wrap_sort(x):
        # [FUNCTION DOC]
        # HELPER: adapter passed to the output-checking helper. Runs the unstable variadic sort
        # on the single input along the enclosing test's dimension with the greater-than
        # comparator and returns the sort result.
        return xla.variadic_sort([x],
                                 dimension=dimension,
                                 is_stable=False,
                                 comparator=compare_gt)

      self._assertOpOutputMatchesExpected(wrap_sort, [x], expected=[expected])

  @parameterized.parameters(0, 1, 2)
  def testVariadicSortDimension(self, dimension):
    # [FUNCTION DOC]
    # Parameterized over the sort dimension (0, 1, 2). For each supported key type, sorts a
    # shuffled 2x3x4 array with the unstable variadic sort and a less-than comparator along the
    # dimension; expects the same result as numpy's ascending sort along that axis.
    shape = (2, 3, 4)
    for key_type in self._supported_key_types():
      x = self._shuffled_arange(shape, key_type)
      expected = np.sort(x, axis=dimension)

      @function.Defun(key_type, key_type)
      def compare_lt(x1, x2):
        # [FUNCTION DOC]
        # HELPER: comparator for the variadic sort; true when the first element is less than
        # the second.
        return x1 < x2

      def wrap_sort(x):
        # [FUNCTION DOC]
        # HELPER: adapter that runs the unstable variadic sort on the single input along the
        # enclosing test's dimension with the less-than comparator and returns the result.
        return xla.variadic_sort([x],
                                 dimension=dimension,
                                 is_stable=False,
                                 comparator=compare_lt)

      self._assertOpOutputMatchesExpected(wrap_sort, [x], expected=[expected])

  @parameterized.parameters(0, 1, 2)
  def testVariadicSortReverse(self, dimension):
    # [FUNCTION DOC]
    # Parameterized over the sort dimension (0, 1, 2). For each supported key type, sorts a
    # shuffled 100x3x4 array with the unstable variadic sort using a greater-than comparator;
    # expects a descending sort along the dimension.
    shape = (100, 3, 4)
    for key_type in self._supported_key_types():
      x = self._shuffled_arange(shape, key_type)
      expected = self._descendingSort(x, dimension)

      @function.Defun(key_type, key_type)
      def compare_gt(x1, x2):
        # [FUNCTION DOC]
        # HELPER: comparator for the variadic sort; true when the first element is greater than
        # the second.
        return x1 > x2

      def wrap_sort(x):
        # [FUNCTION DOC]
        # HELPER: adapter that runs the unstable variadic sort on the single input along the
        # enclosing test's dimension with the greater-than comparator and returns the result.
        return xla.variadic_sort([x],
                                 dimension=dimension,
                                 is_stable=False,
                                 comparator=compare_gt)

      self._assertOpOutputMatchesExpected(wrap_sort, [x], expected=[expected])

  @parameterized.product(dimension=[0, 1, 2], key_type=ALL_KEY_TYPES)
  def testVariadicSortSeveral(self, dimension, key_type):
    # [FUNCTION DOC]
    # Parameterized over dimension (0, 1, 2) and key type. Skipped for old numpy versions lacking
    # take-along-axis; returns early if the key type is unsupported on the device. For every pair
    # of supported value types, sorts three 2x3x4 shuffled arrays together with the variadic sort,
    # comparing only the first array (the key). Expects the first array sorted and the other two
    # permuted in the same way (computed with argsort and take-along-axis).
    if np.__version__ < "1.15":
      raise unittest.SkipTest("np.take_along_axis was added in 1.15")
    if key_type not in self._supported_key_types():
      return
    shape = (2, 3, 4)
    for value_type_1 in self._supported_key_types():
      for value_type_2 in self._supported_key_types():
        inputs = [
            self._shuffled_arange(shape, key_type),
            self._shuffled_arange(shape, value_type_1),
            self._shuffled_arange(shape, value_type_2)
        ]

        # The first array is sorted, and the others are shuffled the same way
        sorted_indices = np.argsort(inputs[0], axis=dimension)
        expected = [
            np.take_along_axis(inp, sorted_indices, axis=dimension)
            for inp in inputs
        ]
        self.assertAllEqual(np.sort(inputs[0], axis=dimension), expected[0])

        @function.Defun(key_type, key_type, value_type_1, value_type_1,
                        value_type_2, value_type_2)
        def compare_lt(x1, x2, y1, y2, z1, z2):
          # [FUNCTION DOC]
          # HELPER: comparator over three operand pairs; ignores the value operands and returns
          # whether the first key is less than the second.
          del y1, y2, z1, z2
          return x1 < x2

        def wrap_sort(*args):
          # [FUNCTION DOC]
          # HELPER: adapter that runs the unstable variadic sort over all given arrays along the
          # enclosing test's dimension with the key comparator and returns the sorted outputs.
          return xla.variadic_sort(
              args,  # Pass the arguments as a tuple
              comparator=compare_lt,
              dimension=dimension,
              is_stable=False)

        self._assertOpOutputMatchesExpected(
            wrap_sort, inputs, expected=expected)

  @parameterized.parameters(ALL_KEY_TYPES)
  @test_util.disable_mlir_bridge("Not supported yet")
  def testVariadicSortLexicographic(self, key_type_2):
    # [FUNCTION DOC]
    # Parameterized over the second key type; disabled for the MLIR bridge. Skipped on old numpy;
    # returns early if the type is unsupported. Sorts three arrays of 20 elements with the
    # variadic sort using a lexicographic comparator on the first two (the first has many ties
    # since it holds only small values from four integer types), the third just follows along.
    # Expects the result of a numpy argsort on a combined key applied to all three arrays.
    # Three inputs: the first two are used for lexicographic sort, and the
    # third is just swapped accordingly.
    # The first array will contain only 0 and 1, to test lexicographic order
    if np.__version__ < "1.15":
      raise unittest.SkipTest("np.take_along_axis was added in 1.15")
    shape = (20,)
    if key_type_2 not in self._supported_key_types():
      return
    for key_type_1 in [np.int16, np.uint16, np.int32, np.uint32]:
      for value_type in self._supported_key_types():
        inputs = [
            # Ensure that some keys in the first input are equal
            np.random.uniform(0, 2, shape).astype(key_type_1),
            self._shuffled_arange(shape, key_type_2),
            self._shuffled_arange(shape, value_type)
        ]
        # The first two arrays are sorted lexicographically, and the third
        # is shuffled the same way
        sorted_indices = np.argsort(100 * inputs[0] + inputs[1])
        expected = [
            np.take_along_axis(inp, sorted_indices, axis=0) for inp in inputs
        ]

        @function.Defun(key_type_1, key_type_1, key_type_2, key_type_2,
                        value_type, value_type)
        def compare_lexicographic(x1, x2, y1, y2, z1, z2):
          # [FUNCTION DOC]
          # HELPER: comparator for the variadic sort; true if the first key is smaller, or the
          # first keys are equal and the second key is smaller. Ignores the third operand pair.
          del z1, z2
          return math_ops.logical_or(
              x1 < x2, math_ops.logical_and(math_ops.equal(x1, x2), y1 < y2))

        def wrap_sort(*args):
          # [FUNCTION DOC]
          # HELPER: adapter that runs the unstable variadic sort over all given arrays along
          # dimension zero with the lexicographic comparator and returns the sorted outputs.
          return xla.variadic_sort(
              args,  # Pass the arguments as a tuple
              comparator=compare_lexicographic,
              dimension=0,
              is_stable=False)

        self._assertOpOutputMatchesExpected(
            wrap_sort, inputs, expected=expected)

  @parameterized.product(dimension=[0, 1, 2], key_type=ALL_KEY_TYPES)
  def testVariadicSortSeveralStable(self, dimension, key_type):
    # [FUNCTION DOC]
    # Parameterized over dimension and key type; returns early if the key type is unsupported.
    # For every pair of supported value types, runs the stable variadic sort on three 2x3x4
    # arrays whose first (key) array is all zeros, so every comparison ties. Expects the outputs
    # to equal the inputs unchanged, verifying stability.
    shape = (2, 3, 4)
    if key_type not in self._supported_key_types():
      return
    for value_type_1 in self._supported_key_types():
      for value_type_2 in self._supported_key_types():
        # The first input is all 0s, there should be no changes for
        # stable sort.
        inputs = [
            np.zeros(shape, key_type),
            self._shuffled_arange(shape, value_type_1),
            self._shuffled_arange(shape, value_type_2)
        ]

        @function.Defun(key_type, key_type, value_type_1, value_type_1,
                        value_type_2, value_type_2)
        def compare_lt(x1, x2, y1, y2, z1, z2):
          # [FUNCTION DOC]
          # HELPER: comparator over three operand pairs; ignores the value operands and returns
          # whether the first key is less than the second.
          del y1, y2, z1, z2
          return x1 < x2

        def wrap_sort(*args):
          # [FUNCTION DOC]
          # HELPER: adapter that runs the stable variadic sort over all given arrays along the
          # enclosing test's dimension with the key comparator and returns the sorted outputs.
          return xla.variadic_sort(
              args,  # Pass the arguments as a tuple
              comparator=compare_lt,
              dimension=dimension,
              is_stable=True)

        self._assertOpOutputMatchesExpected(wrap_sort, inputs, expected=inputs)

  @parameterized.product(dimension=[0, 1, 2], dtype=ALL_KEY_TYPES)
  def testArgsort(self, dimension, dtype):
    # [FUNCTION DOC]
    # Parameterized over dimension and dtype; returns early for unsupported dtypes. Runs the
    # stable argsort op along the dimension on a shuffled 2x3x4 array and expects the same
    # indices as numpy's stable argsort.
    shape = (2, 3, 4)
    if dtype not in self._supported_key_types():
      return

    def argsort(v, axis=dimension):
      # [FUNCTION DOC]
      # HELPER: adapter that calls the stable argsort op on v along the given axis (defaulting
      # to the enclosing test's dimension) and returns the index tensor.
      return sort_ops.argsort(v, axis, stable=True)

    x = self._shuffled_arange(shape, dtype)
    self._assertOpOutputMatchesExpected(
        argsort, [x], expected=[np.argsort(x, axis=dimension, kind="stable")]
    )

  @parameterized.product(
      dtype=[
          dtypes.bfloat16.as_numpy_dtype,
          np.float16,
          np.float32,
          np.float64,
          np.int32,
          np.uint32,
          np.int64,
          np.uint64,
          np.uint8,
          np.int8,
      ],
      rank=[1, 2, 3],
  )
  def testTopK(self, dtype, rank):
    # [FUNCTION DOC]
    # Parameterized over dtype (several float and integer types) and rank (1 to 3). Does nothing
    # if the dtype is not supported on the device. Builds a shuffled array (a range tiled to the
    # rank, shuffled along the last axis; small sizes for half-precision and 8-bit types to avoid
    # duplicate values, 200000 elements otherwise) and, for several k values including zero and
    # the full size, runs the sorted top_k op. Expects the k largest values (descending) and
    # their indices, derived from numpy argsort and sort.
    if dtype in self.numeric_types:
      # Use small input size for bfloat16. Otherwise, we'll get duplicate values
      # after conversion to bfloat16, so the possible resulting index array is
      # no longer unique.
      if dtype in (dtypes.bfloat16.as_numpy_dtype, np.float16):
        array_size = 20
        k_options = [0, 1, 2, 10, 20]
      elif dtype in (dtypes.uint8.as_numpy_dtype, dtypes.int8.as_numpy_dtype):
        array_size = 111
        k_options = [0, 1, 2, 10, 20]
      else:
        array_size = 200 * 1000
        k_options = [0, 1, 2, 10, 20, 100, 1000, 200 * 1000]

      # Tile array to tensor of specified rank, then shuffle along the last dim
      x = np.arange(array_size)
      x = np.tile(x, (2,) * (rank - 1) + (1,))
      np.apply_along_axis(np.random.shuffle, -1, x)

      sorted_indices = x.argsort(axis=-1)[..., ::-1]
      sorted_values = np.sort(x, axis=-1)[..., ::-1]
      for k in k_options:
        indices = sorted_indices[..., :k]
        expected = sorted_values[..., :k]

        def topk(v, k=k):
          # [FUNCTION DOC]
          # HELPER: adapter that calls the sorted top_k op on v with the loop's current k and
          # returns its values and indices.
          return nn_ops.top_k(v, k=k, sorted=True)

        self._assertOpOutputMatchesExpected(
            topk,
            [x.astype(dtype)],
            expected=[expected.astype(dtype), indices],
        )

  def testTopKZeros(self):
    # [FUNCTION DOC]
    # For each supported floating-point type, runs top_k with k=4 on an input mixing positive
    # zeros, negative zeros, 3 and -4. Expects values 3, 0, 0, 0 and indices 3, 0, 2, 6, i.e.
    # zeros of either sign tie and are ordered by lowest index first.
    """Tests that positive and negative zeros sort correctly."""
    supported_types = set(
        [dtypes.bfloat16.as_numpy_dtype, np.float16, np.float32, np.float64])
    for dtype in supported_types.intersection(self.numeric_types):
      with self.session() as sess:
        p = array_ops.placeholder(dtype)
        with self.test_scope():
          topk = nn_ops.top_k(p, k=4)
        results = sess.run(
            topk,
            {p: np.array([0., -0., 0., 3., -0., -4., 0., -0.], dtype=dtype)})
        self.assertAllEqual(np.array([3., 0., 0., 0.], dtype=dtype), results[0])
        self.assertEqual(list([3, 0, 2, 6]), list(results[1]))

  def testTopKInfinities(self):
    # [FUNCTION DOC]
    # For each supported floating-point type, runs top_k with k=6 on a six-element input that
    # includes positive and negative infinity. Expects the full descending order (positive
    # infinity first, negative infinity last) with indices 2, 1, 0, 4, 5, 3.
    """Tests that positive and negative infinity sort correctly."""
    supported_types = set(
        [dtypes.bfloat16.as_numpy_dtype, np.float16, np.float32, np.float64])
    for dtype in supported_types.intersection(self.numeric_types):
      with self.session() as sess:
        p = array_ops.placeholder(dtype)
        with self.test_scope():
          topk = nn_ops.top_k(p, k=6)
        results = sess.run(topk, {
            p:
                np.array([1, 2, float("inf"), -float("inf"), -1, -2],
                         dtype=dtype)
        })
        self.assertAllEqual(
            np.array([float("inf"), 2.0, 1.0, -1.0, -2.0, -float("inf")],
                     dtype=dtype), results[0])
        self.assertEqual(list([2, 1, 0, 4, 5, 3]), list(results[1]))

  @parameterized.named_parameters(
      ("Int32", np.int32),
      ("Int64", np.uint64),
  )
  def testInTopK(self, dtype):
    # [FUNCTION DOC]
    # Named-parameterized over target integer type. KNOWN ISSUE: the case named "Int64" actually
    # uses the unsigned 64-bit type. Does nothing if the dtype is unsupported. Builds a shuffled
    # 16 x 200000 prediction matrix and random target indices, and for several k values checks
    # the in_top_k op against a numpy computation of whether each target is among the k largest.
    if dtype in self.numeric_types:
      array_size = 200 * 1000
      k_options = [0, 1, 2, 10, 20, 100, 1000, 200 * 1000]
      batch = 16
      for x in [np.arange(batch * array_size)]:
        np.random.shuffle(x)
        x = np.reshape(x, [batch, array_size])
        y = np.random.randint(0, array_size, size=batch)
        for k in k_options:
          indices = x.argsort(axis=1)[::, -1:-k - 1:-1]
          expected = [y[i] in indices[i] for i in range(batch)]

          def in_topk(predictions, targets, k=k):
            # [FUNCTION DOC]
            # HELPER: adapter that calls the in_top_k op with the loop's current k and returns
            # a boolean tensor per row.
            return nn_ops.in_top_k(predictions, targets, k)

          self._assertOpOutputMatchesExpected(
              in_topk,
              [x.astype(np.float32), y.astype(dtype)],
              expected=[expected])


class SortOpsBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Benchmark class timing the variadic sort op on a 100001-element array for float16, float32
  # and float64, stable and unstable, each with and without XLA JIT. Each public benchmark method
  # delegates to a shared helper.
  """Microbenchmarks for the sort ops."""

  def _benchmarkSort(self, name, dtype, is_stable, use_xla_jit):
    # [FUNCTION DOC]
    # HELPER: shared implementation of all benchmark methods in this class.
    # Parameters: name (label prefix for the benchmark), dtype (numpy dtype of the data),
    # is_stable (whether to request a stable sort), use_xla_jit (whether to compile with the XLA
    # JIT). Defines a comparator and a builder function that creates a shuffled array and sorts
    # it with the variadic sort, and runs it on the CPU through the XLA test benchmark utility.
    # The reported benchmark name combines stability, label and shape. Returns nothing.

    def get_shuffled_arr(sorted_arr, shape):
      # [FUNCTION DOC]
      # HELPER: returns a randomly shuffled copy of sorted_arr reshaped to the given shape.
      shuffled = sorted_arr.copy()
      np.random.shuffle(shuffled)
      return shuffled.reshape(shape)

    @function.Defun(dtype, dtype)
    def compare_lt(x1, x2):
      # [FUNCTION DOC]
      # HELPER: comparator for the variadic sort; true when the first element is less than the
      # second.
      return x1 < x2

    def builder_fn():
      # [FUNCTION DOC]
      # HELPER: builder callback for the benchmark utility. Creates a shuffled array of 100001
      # elements and applies the variadic sort along dimension zero with the enclosing stability
      # setting. Returns a pair: the benchmark name string and a list holding the sort result.
      shape = (100001,)
      sorted_arr = np.arange(np.prod(shape), dtype=dtype)
      shuffled = get_shuffled_arr(sorted_arr, shape)
      given_result = xla.variadic_sort(
          [shuffled], dimension=0, is_stable=is_stable, comparator=compare_lt
      )
      stable_str = "stable" if is_stable else "unstable"
      return "%s_%s.shape%s" % (stable_str, name, shape), [given_result]

    xla_test.Benchmark(self, builder_fn, use_xla_jit=use_xla_jit, device="cpu")

  def benchmarkStableSortF16(self):
    # [FUNCTION DOC]
    # Benchmarks the stable sort of float16 data without XLA JIT.
    self._benchmarkSort(
        "sort_f16", dtype=np.float16, is_stable=True, use_xla_jit=False
    )

  def benchmarkStableSortF32(self):
    # [FUNCTION DOC]
    # Benchmarks the stable sort of float32 data without XLA JIT.
    self._benchmarkSort(
        "sort_f32", dtype=np.float32, is_stable=True, use_xla_jit=False
    )

  def benchmarkStableSortF64(self):
    # [FUNCTION DOC]
    # Benchmarks the stable sort of float64 data without XLA JIT.
    self._benchmarkSort(
        "sort_f64", dtype=np.float64, is_stable=True, use_xla_jit=False
    )

  def benchmarkStableSortF16XLA(self):
    # [FUNCTION DOC]
    # Benchmarks the stable sort of float16 data with XLA JIT enabled.
    self._benchmarkSort(
        "sort_f16", dtype=np.float16, is_stable=True, use_xla_jit=True
    )

  def benchmarkStableSortF32XLA(self):
    # [FUNCTION DOC]
    # Benchmarks the stable sort of float32 data with XLA JIT enabled.
    self._benchmarkSort(
        "sort_f32", dtype=np.float32, is_stable=True, use_xla_jit=True
    )

  def benchmarkStableSortF64XLA(self):
    # [FUNCTION DOC]
    # Benchmarks the stable sort of float64 data with XLA JIT enabled.
    self._benchmarkSort(
        "sort_f64", dtype=np.float64, is_stable=True, use_xla_jit=True
    )

  def benchmarkUnstableSortF16(self):
    # [FUNCTION DOC]
    # Benchmarks the unstable sort of float16 data without XLA JIT.
    self._benchmarkSort(
        "sort_f16", dtype=np.float16, is_stable=False, use_xla_jit=False
    )

  def benchmarkUnstableSortF32(self):
    # [FUNCTION DOC]
    # Benchmarks the unstable sort of float32 data without XLA JIT.
    self._benchmarkSort(
        "sort_f32", dtype=np.float32, is_stable=False, use_xla_jit=False
    )

  def benchmarkUnstableSortF64(self):
    # [FUNCTION DOC]
    # Benchmarks the unstable sort of float64 data without XLA JIT.
    self._benchmarkSort(
        "sort_f64", dtype=np.float64, is_stable=False, use_xla_jit=False
    )

  def benchmarkUnstableSortF16XLA(self):
    # [FUNCTION DOC]
    # Benchmarks the unstable sort of float16 data with XLA JIT enabled.
    self._benchmarkSort(
        "sort_f16", dtype=np.float16, is_stable=False, use_xla_jit=True
    )

  def benchmarkUnstableSortF32XLA(self):
    # [FUNCTION DOC]
    # Benchmarks the unstable sort of float32 data with XLA JIT enabled.
    self._benchmarkSort(
        "sort_f32", dtype=np.float32, is_stable=False, use_xla_jit=True
    )

  def benchmarkUnstableSortF64XLA(self):
    # [FUNCTION DOC]
    # Benchmarks the unstable sort of float64 data with XLA JIT enabled.
    self._benchmarkSort(
        "sort_f64", dtype=np.float64, is_stable=False, use_xla_jit=True
    )


if __name__ == "__main__":
  test.main()
