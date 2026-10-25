# [MODULE DOC]
# Tests for the array creation, manipulation and reduction functions of TF NumPy (np_array_ops),
# checked against real NumPy as the reference. Test cases: ArrayCreationTest (empty, zeros, ones, eye,
# full, array, asarray, arange, diag and similar), ArrayMethodsTest (reductions, copy, reshape,
# transpose, padding, taking, splitting and others), ArrayManipulationTest (broadcast_to, ix_),
# ArrayMathTest (arithmetic of Python constants with arrays) and StringArrayTest (string handling).
# Many tests run the same operation over a grid of input representations (Python lists, tensors, NumPy
# arrays, TF NumPy arrays) and compare dtype, shape and values. A module-level helper configures two
# logical CPU devices once per process. Run as a script it enables eager execution, uses the legacy
# dtype conversion mode, enables NumPy methods on tensors and starts the test runner.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tf numpy array methods."""

import itertools
import operator
import sys
from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_math_ops
from tensorflow.python.platform import test


_virtual_devices_ready = False


def set_up_virtual_devices():
  # [FUNCTION DOC]
  # HELPER: shared test fixture setup for the test cases in this file.
  # Splits the first physical CPU into two logical CPU devices, once per process, guarded by a
  # module-level flag. Takes no inputs and returns nothing. Side effects: changes global device
  # configuration and sets the flag. Fails if the runtime has already been initialized.
  global _virtual_devices_ready
  if _virtual_devices_ready:
    return
  physical_devices = config.list_physical_devices('CPU')
  config.set_logical_device_configuration(
      physical_devices[0], [
          context.LogicalDeviceConfiguration(),
          context.LogicalDeviceConfiguration()
      ])
  _virtual_devices_ready = True


class ArrayCreationTest(test.TestCase):
  # [CLASS DOC]
  # Tests for array creation functions (empty, zeros, ones, their like-variants, eye, identity, full,
  # array, asarray, asanyarray, ascontiguousarray, arange, diag, diagflat, vander and IndexedSlices
  # conversion). Fixtures built in setUp: lists of shapes in several representations, dtypes and
  # source arrays in several representations. Also holds comparison helpers (shape, dtype, full match).

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture. Calls the base setUp and the virtual device setup, then builds: shape_transforms (identity,
    # NumPy int array, TF NumPy array, tensor shape type) and all_shapes (every Python shape converted by
    # every transform, excluding scalar arrays and tensors, which crash NumPy empty-like calls); all_types
    # (Python int/float and NumPy int16/32/64, float16/32/64, complex64/128); array_transforms (identity,
    # tensor conversion, NumPy array, TF NumPy array) and all_arrays (sample data converted by each).
    # KNOWN ISSUE: the Python-major-version check is always true on supported interpreters.
    super(ArrayCreationTest, self).setUp()
    set_up_virtual_devices()
    python_shapes = [
        0, 1, 2, (), (1,), (2,), (1, 2, 3), [], [1], [2], [1, 2, 3]
    ]
    self.shape_transforms = [
        lambda x: x, lambda x: np.array(x, dtype=int),
        lambda x: np_array_ops.array(x, dtype=int), tensor_shape.TensorShape
    ]

    self.all_shapes = []
    for fn in self.shape_transforms:
      self.all_shapes.extend([fn(s) for s in python_shapes])

    if sys.version_info.major == 3:
      # There is a bug of np.empty (and alike) in Python 3 causing a crash when
      # the `shape` argument is an np_arrays.ndarray scalar (or tf.Tensor
      # scalar).
      def not_ndarray_scalar(s):
        # [FUNCTION DOC]
        # HELPER: filter predicate used by setUp. Input s is a candidate shape. Returns true unless s is a
        # zero-dimensional array (workaround for a crash in NumPy empty-like calls); no side effects.
        return not (isinstance(s, np_arrays.ndarray) and s.ndim == 0)

      self.all_shapes = list(filter(not_ndarray_scalar, self.all_shapes))

    self.all_types = [
        int, float, np.int16, np.int32, np.int64, np.float16, np.float32,
        np.float64, np.complex64, np.complex128
    ]

    source_array_data = [
        1,
        5.5,
        7,
        (),
        (8, 10.),
        ((), ()),
        ((1, 4), (2, 8)),
        [],
        [7],
        [8, 10.],
        [[], []],
        [[1, 4], [2, 8]],
        ([], []),
        ([1, 4], [2, 8]),
        [(), ()],
        [(1, 4), (2, 8)],
    ]

    self.array_transforms = [
        lambda x: x,
        ops.convert_to_tensor,
        np.array,
        np_array_ops.array,
    ]
    self.all_arrays = []
    for fn in self.array_transforms:
      self.all_arrays.extend([fn(s) for s in source_array_data])

  def testEmpty(self):
    # [FUNCTION DOC]
    # Checks empty creation against NumPy for every shape, then for every shape and dtype pair; for each
    # asserts only matching shape and dtype (values are uninitialized so not compared).
    for s in self.all_shapes:
      actual = np_array_ops.empty(s)
      expected = np.empty(s)
      msg = 'shape: {}'.format(s)
      self.match_shape(actual, expected, msg)
      self.match_dtype(actual, expected, msg)

    for s, t in itertools.product(self.all_shapes, self.all_types):
      actual = np_array_ops.empty(s, t)
      expected = np.empty(s, t)
      msg = 'shape: {}, dtype: {}'.format(s, t)
      self.match_shape(actual, expected, msg)
      self.match_dtype(actual, expected, msg)

  def testEmptyLike(self):
    # [FUNCTION DOC]
    # Checks the empty-like function over all sample arrays, then over arrays crossed with all dtypes,
    # asserting matching shape and dtype against NumPy.
    for a in self.all_arrays:
      actual = np_array_ops.empty_like(a)
      expected = np.empty_like(a)
      msg = 'array: {}'.format(a)
      self.match_shape(actual, expected, msg)
      self.match_dtype(actual, expected, msg)

    for a, t in itertools.product(self.all_arrays, self.all_types):
      actual = np_array_ops.empty_like(a, t)
      expected = np.empty_like(a, t)
      msg = 'array: {} type: {}'.format(a, t)
      self.match_shape(actual, expected, msg)
      self.match_dtype(actual, expected, msg)

  def testZeros(self):
    # [FUNCTION DOC]
    # Checks zeros creation against NumPy for all shapes, and all shape and dtype pairs; asserts full
    # match of dtype, shape and values.
    for s in self.all_shapes:
      actual = np_array_ops.zeros(s)
      expected = np.zeros(s)
      msg = 'shape: {}'.format(s)
      self.match(actual, expected, msg)

    for s, t in itertools.product(self.all_shapes, self.all_types):
      actual = np_array_ops.zeros(s, t)
      expected = np.zeros(s, t)
      msg = 'shape: {}, dtype: {}'.format(s, t)
      self.match(actual, expected, msg)

  def testZerosLike(self):
    # [FUNCTION DOC]
    # Checks the zeros-like function over all sample arrays, and arrays crossed with all dtypes, against
    # NumPy with full match.
    for a in self.all_arrays:
      actual = np_array_ops.zeros_like(a)
      expected = np.zeros_like(a)
      msg = 'array: {}'.format(a)
      self.match(actual, expected, msg)

    for a, t in itertools.product(self.all_arrays, self.all_types):
      actual = np_array_ops.zeros_like(a, t)
      expected = np.zeros_like(a, t)
      msg = 'array: {} type: {}'.format(a, t)
      self.match(actual, expected, msg)

  def testOnes(self):
    # [FUNCTION DOC]
    # Checks ones creation against NumPy for all shapes and all shape and dtype pairs with full match.
    for s in self.all_shapes:
      actual = np_array_ops.ones(s)
      expected = np.ones(s)
      msg = 'shape: {}'.format(s)
      self.match(actual, expected, msg)

    for s, t in itertools.product(self.all_shapes, self.all_types):
      actual = np_array_ops.ones(s, t)
      expected = np.ones(s, t)
      msg = 'shape: {}, dtype: {}'.format(s, t)
      self.match(actual, expected, msg)

  def testOnesLike(self):
    # [FUNCTION DOC]
    # Checks the ones-like function over all sample arrays and array and dtype pairs with full match.
    for a in self.all_arrays:
      actual = np_array_ops.ones_like(a)
      expected = np.ones_like(a)
      msg = 'array: {}'.format(a)
      self.match(actual, expected, msg)

    for a, t in itertools.product(self.all_arrays, self.all_types):
      actual = np_array_ops.ones_like(a, t)
      expected = np.ones_like(a, t)
      msg = 'array: {} type: {}'.format(a, t)
      self.match(actual, expected, msg)

  def testEye(self):
    # [FUNCTION DOC]
    # Checks identity-matrix creation (eye) against NumPy for sizes up to three: rows only, with diagonal
    # offsets, with columns, and combined offset and columns, first with default dtype and then for every
    # dtype. Asserts full match each time.
    n_max = 3
    m_max = 3

    for n in range(1, n_max + 1):
      self.match(np_array_ops.eye(n), np.eye(n))
      for k in range(-n, n + 1):
        self.match(np_array_ops.eye(n, k=k), np.eye(n, k=k))
      for m in range(1, m_max + 1):
        self.match(np_array_ops.eye(n, m), np.eye(n, m))
        for k in range(-n, m):
          self.match(np_array_ops.eye(n, k=k), np.eye(n, k=k))
          self.match(np_array_ops.eye(n, m, k), np.eye(n, m, k))

    for dtype in self.all_types:
      for n in range(1, n_max + 1):
        self.match(np_array_ops.eye(n, dtype=dtype), np.eye(n, dtype=dtype))
        for k in range(-n, n + 1):
          self.match(
              np_array_ops.eye(n, k=k, dtype=dtype),
              np.eye(n, k=k, dtype=dtype))
        for m in range(1, m_max + 1):
          self.match(
              np_array_ops.eye(n, m, dtype=dtype), np.eye(n, m, dtype=dtype))
          for k in range(-n, m):
            self.match(
                np_array_ops.eye(n, k=k, dtype=dtype),
                np.eye(n, k=k, dtype=dtype))
            self.match(
                np_array_ops.eye(n, m, k, dtype=dtype),
                np.eye(n, m, k, dtype=dtype))

  def testIdentity(self):
    # [FUNCTION DOC]
    # Checks the identity function for sizes one to three with default dtype and every dtype against
    # NumPy with full match.
    n_max = 3

    for n in range(1, n_max + 1):
      self.match(np_array_ops.identity(n), np.identity(n))

    for dtype in self.all_types:
      for n in range(1, n_max + 1):
        self.match(
            np_array_ops.identity(n, dtype=dtype), np.identity(n, dtype=dtype))

  def testFull(self):
    # [FUNCTION DOC]
    # Checks the full function with many fill values (scalars, lists, nested lists) and shapes, with fill
    # values and shapes converted by each array and shape transform, with and without an explicit dtype
    # from all dtypes; asserts match against NumPy.
    # List of 2-tuples of fill value and shape.
    data = [
        (5, ()),
        (5, (7,)),
        (5., (7,)),
        ([5, 8], (2,)),
        ([5, 8], (3, 2)),
        ([[5], [8]], (2, 3)),
        ([[5], [8]], (3, 2, 5)),
        ([[5.], [8.]], (3, 2, 5)),
        ([[3, 4], [5, 6], [7, 8]], (3, 3, 2)),
    ]
    for f, s in data:
      for fn1, fn2 in itertools.product(self.array_transforms,
                                        self.shape_transforms):
        fill_value = fn1(f)
        shape = fn2(s)
        self.match(
            np_array_ops.full(shape, fill_value), np.full(shape, fill_value))
        for dtype in self.all_types:
          self.match(
              np_array_ops.full(shape, fill_value, dtype=dtype),
              np.full(shape, fill_value, dtype=dtype))

  def testFullLike(self):
    # [FUNCTION DOC]
    # Checks the full-like function using prototype arrays built by TF NumPy or NumPy zeros builders in
    # all dtypes, with many fill values and optional explicit dtype; asserts match against NumPy.
    # List of 2-tuples of fill value and shape.
    data = [
        (5, ()),
        (5, (7,)),
        (5., (7,)),
        ([5, 8], (2,)),
        ([5, 8], (3, 2)),
        ([[5], [8]], (2, 3)),
        ([[5], [8]], (3, 2, 5)),
        ([[5.], [8.]], (3, 2, 5)),
    ]
    zeros_builders = [np_array_ops.zeros, np.zeros]
    for f, s in data:
      for fn1, fn2, arr_dtype in itertools.product(self.array_transforms,
                                                   zeros_builders,
                                                   self.all_types):
        fill_value = fn1(f)
        arr = fn2(s, arr_dtype)
        self.match(
            np_array_ops.full_like(arr, fill_value),
            np.full_like(arr, fill_value))
        for dtype in self.all_types:
          self.match(
              np_array_ops.full_like(arr, fill_value, dtype=dtype),
              np.full_like(arr, fill_value, dtype=dtype))

  def testArray(self):
    # [FUNCTION DOC]
    # Checks the array function. First compares to NumPy over all arrays, dtypes, minimum-dimension values
    # (0, 1, 2, 5) and copy flag values. Then verifies identity semantics when copy is false: the same
    # object is returned unless a minimum-dimension or dtype requirement forces a new one, both on the
    # default device and under the second CPU. Finally asserts a copy made under the second CPU is backed
    # by that device while the original is not.
    ndmins = [0, 1, 2, 5]
    for a, dtype, ndmin, copy in itertools.product(self.all_arrays,
                                                   self.all_types, ndmins,
                                                   [True, False]):
      self.match(
          np_array_ops.array(a, dtype=dtype, ndmin=ndmin, copy=copy),
          np.array(a, dtype=dtype, ndmin=ndmin))

    zeros_list = np_array_ops.zeros(5)

    def test_copy_equal_false():
      # [FUNCTION DOC]
      # HELPER: inner check of testArray, run on the default and second CPU devices.
      # KNOWN ISSUE: each identity assertion is written twice in a row, so the duplicated checks add nothing.
      # Takes no inputs (uses the enclosing zeros array). Asserts that with copy false the same object is
      # returned by default, when the minimum dimensions is already satisfied, and when the dtype is
      # already satisfied, and that a different object is returned when more dimensions or an int dtype
      # are requested.
      # Backing tensor is the same if copy=False, other attributes being None.
      self.assertIs(np_array_ops.array(zeros_list, copy=False), zeros_list)
      self.assertIs(np_array_ops.array(zeros_list, copy=False), zeros_list)

      # Backing tensor is different if ndmin is not satisfied.
      self.assertIsNot(
          np_array_ops.array(zeros_list, copy=False, ndmin=2),
          zeros_list)
      self.assertIsNot(
          np_array_ops.array(zeros_list, copy=False, ndmin=2),
          zeros_list)
      self.assertIs(
          np_array_ops.array(zeros_list, copy=False, ndmin=1),
          zeros_list)
      self.assertIs(
          np_array_ops.array(zeros_list, copy=False, ndmin=1),
          zeros_list)

      # Backing tensor is different if dtype is not satisfied.
      self.assertIsNot(
          np_array_ops.array(zeros_list, copy=False, dtype=int),
          zeros_list)
      self.assertIsNot(
          np_array_ops.array(zeros_list, copy=False, dtype=int),
          zeros_list)
      self.assertIs(
          np_array_ops.array(zeros_list, copy=False, dtype=float),
          zeros_list)
      self.assertIs(
          np_array_ops.array(zeros_list, copy=False, dtype=float),
          zeros_list)

    test_copy_equal_false()
    with ops.device('CPU:1'):
      test_copy_equal_false()

    self.assertNotIn('CPU:1', zeros_list.backing_device)
    with ops.device('CPU:1'):
      self.assertIn(
          'CPU:1', np_array_ops.array(zeros_list, copy=True).backing_device)
      self.assertIn(
          'CPU:1', np_array_ops.array(np.array(0), copy=True).backing_device)

  def testAsArray(self):
    # [FUNCTION DOC]
    # Checks asarray over all arrays and dtypes against NumPy, and that an existing array is returned
    # as the same object when no dtype is given (also under the second CPU) but a new object when a dtype is given.
    for a, dtype in itertools.product(self.all_arrays, self.all_types):
      self.match(
          np_array_ops.asarray(a, dtype=dtype), np.asarray(a, dtype=dtype))

    zeros_list = np_array_ops.zeros(5)
    # Same instance is returned if no dtype is specified and input is ndarray.
    self.assertIs(np_array_ops.asarray(zeros_list), zeros_list)
    with ops.device('CPU:1'):
      self.assertIs(np_array_ops.asarray(zeros_list), zeros_list)
    # Different instance is returned if dtype is specified and input is ndarray.
    self.assertIsNot(np_array_ops.asarray(zeros_list, dtype=int), zeros_list)

  def testAsAnyArray(self):
    # [FUNCTION DOC]
    # Same as testAsArray but for asanyarray: match against NumPy for all arrays and dtypes, same-object
    # return when no dtype is given (default and second CPU), different object when dtype is given.
    for a, dtype in itertools.product(self.all_arrays, self.all_types):
      self.match(
          np_array_ops.asanyarray(a, dtype=dtype),
          np.asanyarray(a, dtype=dtype))
    zeros_list = np_array_ops.zeros(5)
    # Same instance is returned if no dtype is specified and input is ndarray.
    self.assertIs(np_array_ops.asanyarray(zeros_list), zeros_list)
    with ops.device('CPU:1'):
      self.assertIs(np_array_ops.asanyarray(zeros_list), zeros_list)
    # Different instance is returned if dtype is specified and input is ndarray.
    self.assertIsNot(np_array_ops.asanyarray(zeros_list, dtype=int), zeros_list)

  def testAsContiguousArray(self):
    # [FUNCTION DOC]
    # Checks ascontiguousarray over all arrays and dtypes against NumPy with full match.
    for a, dtype in itertools.product(self.all_arrays, self.all_types):
      self.match(
          np_array_ops.ascontiguousarray(a, dtype=dtype),
          np.ascontiguousarray(a, dtype=dtype))

  def testARange(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: for a zero step, the statements after the first call inside the raises-check block
    # KNOWN ISSUE: (the dtype variant) can never execute because the first call already raises, so they are dead code.
    # Checks arange against NumPy across integer and float starts and stops, integer steps (floats are
    # deliberately not tested because NumPy is inconsistent), and all dtypes. Dtype variants are skipped
    # when a bound is a float because of known bugs. A zero step must raise a value error.
    int_values = np.arange(-3, 3).tolist()
    float_values = np.arange(-3.5, 3.5).tolist()
    all_values = int_values + float_values
    for dtype in self.all_types:
      for start in all_values:
        msg = 'dtype:{} start:{}'.format(dtype, start)
        self.match(np_array_ops.arange(start), np.arange(start), msg=msg)
        self.match(
            np_array_ops.arange(start, dtype=dtype),
            np.arange(start, dtype=dtype),
            msg=msg)
        for stop in all_values:
          msg = 'dtype:{} start:{} stop:{}'.format(dtype, start, stop)
          self.match(
              np_array_ops.arange(start, stop), np.arange(start, stop), msg=msg)
          # TODO(srbs): Investigate and remove check.
          # There are some bugs when start or stop is float and dtype is int.
          if not isinstance(start, float) and not isinstance(stop, float):
            self.match(
                np_array_ops.arange(start, stop, dtype=dtype),
                np.arange(start, stop, dtype=dtype),
                msg=msg)
          # Note: We intentionally do not test with float values for step
          # because numpy.arange itself returns inconsistent results. e.g.
          # np.arange(0.5, 3, step=0.5, dtype=int) returns
          # array([0, 1, 2, 3, 4])
          for step in int_values:
            msg = 'dtype:{} start:{} stop:{} step:{}'.format(
                dtype, start, stop, step)
            if not step:
              with self.assertRaises(ValueError):
                self.match(
                    np_array_ops.arange(start, stop, step),
                    np.arange(start, stop, step),
                    msg=msg)
                if not isinstance(start, float) and not isinstance(stop, float):
                  self.match(
                      np_array_ops.arange(start, stop, step, dtype=dtype),
                      np.arange(start, stop, step, dtype=dtype),
                      msg=msg)
            else:
              self.match(
                  np_array_ops.arange(start, stop, step),
                  np.arange(start, stop, step),
                  msg=msg)
              if not isinstance(start, float) and not isinstance(stop, float):
                self.match(
                    np_array_ops.arange(start, stop, step, dtype=dtype),
                    np.arange(start, stop, step, dtype=dtype),
                    msg=msg)

  def testDiag(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the inner helper reassigns its input with each transform in turn, so later transforms are
    # KNOWN ISSUE: applied to already-converted values rather than the original input.
    # Checks the diag function against NumPy for two-dimensional and one-dimensional inputs, using several
    # input representations and diagonal offsets from -3 to 2.
    array_transforms = [
        lambda x: x,  # Identity,
        ops.convert_to_tensor,
        np.array,
        lambda x: np.array(x, dtype=np.float32),
        lambda x: np.array(x, dtype=np.float64),
        np_array_ops.array,
        lambda x: np_array_ops.array(x, dtype=np.float32),
        lambda x: np_array_ops.array(x, dtype=np.float64)
    ]

    def run_test(arr):
      # [FUNCTION DOC]
      # HELPER: inner driver of testDiag. Input arr is the initial data. For each transform, converts arr
      # (cumulatively) and asserts diag matches NumPy, with default and each offset.
      for fn in array_transforms:
        arr = fn(arr)
        self.match(
            np_array_ops.diag(arr), np.diag(arr), msg='diag({})'.format(arr))
        for k in range(-3, 3):
          self.match(
              np_array_ops.diag(arr, k),
              np.diag(arr, k),
              msg='diag({}, k={})'.format(arr, k))

    # 2-d arrays.
    run_test(np.arange(9).reshape((3, 3)).tolist())
    run_test(np.arange(6).reshape((2, 3)).tolist())
    run_test(np.arange(6).reshape((3, 2)).tolist())
    run_test(np.arange(3).reshape((1, 3)).tolist())
    run_test(np.arange(3).reshape((3, 1)).tolist())
    run_test([[5]])
    run_test([[]])
    run_test([[], []])

    # 1-d arrays.
    run_test([])
    run_test([1])
    run_test([1, 2])

  def testDiagFlat(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the inner helper reassigns its input with each transform in turn (cumulative conversion).
    # Checks the diagflat function against NumPy for one, two and three dimensional inputs, using several
    # representations and offsets from -3 to 2.
    array_transforms = [
        lambda x: x,  # Identity,
        ops.convert_to_tensor,
        np.array,
        lambda x: np.array(x, dtype=np.float32),
        lambda x: np.array(x, dtype=np.float64),
        np_array_ops.array,
        lambda x: np_array_ops.array(x, dtype=np.float32),
        lambda x: np_array_ops.array(x, dtype=np.float64)
    ]

    def run_test(arr):
      # [FUNCTION DOC]
      # HELPER: inner driver of testDiagFlat. Input arr is the initial data. For each transform converts arr
      # (cumulatively) and asserts diagflat matches NumPy with default and each offset.
      for fn in array_transforms:
        arr = fn(arr)
        self.match(
            np_array_ops.diagflat(arr),
            np.diagflat(arr),
            msg='diagflat({})'.format(arr))
        for k in range(-3, 3):
          self.match(
              np_array_ops.diagflat(arr, k),
              np.diagflat(arr, k),
              msg='diagflat({}, k={})'.format(arr, k))

    # 1-d arrays.
    run_test([])
    run_test([1])
    run_test([1, 2])
    # 2-d arrays.
    run_test([[]])
    run_test([[5]])
    run_test([[], []])
    run_test(np.arange(4).reshape((2, 2)).tolist())
    run_test(np.arange(2).reshape((2, 1)).tolist())
    run_test(np.arange(2).reshape((1, 2)).tolist())
    # 3-d arrays
    run_test(np.arange(8).reshape((2, 2, 2)).tolist())

  def match_shape(self, actual, expected, msg=None):
    # [FUNCTION DOC]
    # HELPER: assertion helper for this test case. Inputs actual and expected are array-like with a shape;
    # msg is an optional prefix. Asserts the shapes are equal; if msg is given, it is replaced by a
    # formatted message including both shapes. Raises a test failure on mismatch.
    if msg:
      msg = 'Shape match failed for: {}. Expected: {} Actual: {}'.format(
          msg, expected.shape, actual.shape)
    self.assertEqual(actual.shape, expected.shape, msg=msg)

  def match_dtype(self, actual, expected, msg=None):
    # [FUNCTION DOC]
    # HELPER: assertion helper for this test case. Inputs actual and expected have a dtype; msg optional.
    # Asserts the dtypes are equal with a formatted message naming both when msg is given.
    if msg:
      msg = 'Dtype match failed for: {}. Expected: {} Actual: {}.'.format(
          msg, expected.dtype, actual.dtype)
    self.assertEqual(actual.dtype, expected.dtype, msg=msg)

  def match(self, actual, expected, msg=None, almost=False, decimal=7):
    # [FUNCTION DOC]
    # HELPER: main comparison assertion for this test case.
    # Inputs: actual (value under test), expected (NumPy reference), msg optional text, almost (boolean,
    # default false) selects approximate value comparison, decimal (default 7) is the precision used then.
    # Asserts actual is a TF NumPy array, then dtype and shape match, then values match exactly (as a list
    # for scalars, sequence otherwise) or almost equal.
    msg_ = 'Expected: {} Actual: {}'.format(expected, actual)
    if msg:
      msg = '{} {}'.format(msg_, msg)
    else:
      msg = msg_
    self.assertIsInstance(actual, np_arrays.ndarray)
    self.match_dtype(actual, expected, msg)
    self.match_shape(actual, expected, msg)
    if not almost:
      if not actual.shape.rank:
        self.assertEqual(actual.tolist(), expected.tolist())
      else:
        self.assertSequenceEqual(actual.tolist(), expected.tolist())
    else:
      np.testing.assert_almost_equal(
          actual.tolist(), expected.tolist(), decimal=decimal)

  def testIndexedSlices(self):
    # [FUNCTION DOC]
    # Builds a sparse indexed-slices object with two rows of ones in a ten-by-three dense shape, converts
    # it with the array function without copying and asserts it equals the dense scattered tensor.
    dtype = dtypes.int64
    iss = indexed_slices.IndexedSlices(
        values=np_array_ops.ones([2, 3], dtype=dtype),
        indices=constant_op.constant([1, 9]),
        dense_shape=[10, 3])
    a = np_array_ops.array(iss, copy=False)
    expected = array_ops.scatter_nd([[1], [9]],
                                    array_ops.ones([2, 3], dtype=dtype),
                                    [10, 3])
    self.assertAllEqual(expected, a)

  def testVander(self):
    # [FUNCTION DOC]
    # Checks vander with zero columns and decreasing order on two values equals NumPy's result.
    tf_res = np_array_ops.vander([-1.0, 1.0], N=0, increasing=False)
    np_res = np.vander(np.array([-1.0, 1.0]), N=0)
    self.assertAllEqual(tf_res, np_res)


class ArrayMethodsTest(test.TestCase):
  # [CLASS DOC]
  # Tests for array methods and reductions (all, any, compress, copy, cumprod, cumsum, imag, real, amax,
  # amin, mean, std, var, prod, sum, size, ravel, repeat, around, reshape, expand_dims, squeeze, transpose,
  # pad, take, take_along_axis, where, shape, swapaxes, moveaxis, ndim, isscalar, split, hsplit, sign).
  # setUp provides array_transforms; comparison helpers are defined at the end of the class.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture. Calls the base setUp and virtual device setup, then stores array_transforms (identity,
    # tensor conversion, NumPy array, TF NumPy array).
    super(ArrayMethodsTest, self).setUp()
    set_up_virtual_devices()
    self.array_transforms = [
        lambda x: x,
        ops.convert_to_tensor,
        np.array,
        np_array_ops.array,
    ]

  def testAllAny(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the helper reassigns its input with each transform so conversions accumulate.
    # Checks all and any against NumPy for scalars, empty, booleans, floats and ints, with axis (single or
    # tuple) and keepdims options.

    def run_test(arr, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner driver of testAllAny. Input arr plus extra arguments and keyword arguments for the
      # functions; for each transform asserts the all and any results match NumPy.
      for fn in self.array_transforms:
        arr = fn(arr)
        self.match(
            np_array_ops.all(arr, *args, **kwargs),
            np.all(arr, *args, **kwargs))
        self.match(
            np_array_ops.any(arr, *args, **kwargs),
            np.any(arr, *args, **kwargs))

    run_test(0)
    run_test(1)
    run_test([])
    run_test([[True, False], [True, True]])
    run_test([[True, False], [True, True]], axis=0)
    run_test([[True, False], [True, True]], axis=0, keepdims=True)
    run_test([[True, False], [True, True]], axis=1)
    run_test([[True, False], [True, True]], axis=1, keepdims=True)
    run_test([[True, False], [True, True]], axis=(0, 1))
    run_test([[True, False], [True, True]], axis=(0, 1), keepdims=True)
    run_test([5.2, 3.5], axis=0)
    run_test([1, 0], axis=0)

  def testCompress(self):
    # [FUNCTION DOC]
    # Checks compress with many condition and array pairs (booleans and 0/1 ints, mismatched lengths, scalars
    # and 2D arrays) and axes, comparing to NumPy with the condition cast to boolean.

    def run_test(condition, arr, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner driver of testCompress; converts condition and array with each pair of transforms and
      # asserts the result matches NumPy.
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arg1 = fn1(condition)
          arg2 = fn2(arr)
          self.match(
              np_array_ops.compress(arg1, arg2, *args, **kwargs),
              np.compress(
                  np.asarray(arg1).astype(np.bool_), arg2, *args, **kwargs))

    run_test([True], 5)
    run_test([False], 5)
    run_test([], 5)
    run_test([True, False, True], [1, 2, 3])
    run_test([True, False], [1, 2, 3])
    run_test([False, True], [[1, 2], [3, 4]])
    run_test([1, 0, 1], [1, 2, 3])
    run_test([1, 0], [1, 2, 3])
    run_test([0, 1], [[1, 2], [3, 4]])
    run_test([True], [[1, 2], [3, 4]])
    run_test([False, True], [[1, 2], [3, 4]], axis=1)
    run_test([False, True], [[1, 2], [3, 4]], axis=0)
    run_test([False, True], [[1, 2], [3, 4]], axis=-1)
    run_test([False, True], [[1, 2], [3, 4]], axis=-2)

  def testCopy(self):
    # [FUNCTION DOC]
    # Checks copy against NumPy for several inputs. Then asserts a copy of a scalar array requested
    # under the second CPU is backed by that device while the original is not.

    def run_test(arr, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner driver of testCopy; for each transform asserts the copy matches NumPy.
      for fn in self.array_transforms:
        arg = fn(arr)
        self.match(
            np_array_ops.copy(arg, *args, **kwargs),
            np.copy(arg, *args, **kwargs))

    run_test([])
    run_test([1, 2, 3])
    run_test([1., 2., 3.])
    run_test([True])
    run_test(np.arange(9).reshape((3, 3)).tolist())

    a = np_array_ops.asarray(0)
    self.assertNotIn('CPU:1', a.backing_device)
    with ops.device('CPU:1'):
      self.assertIn('CPU:1', np_array_ops.array(a, copy=True)
                    .backing_device)
      self.assertIn('CPU:1', np_array_ops.array(np.array(0), copy=True)
                    .backing_device)

  def testCumProdAndSum(self):
    # [FUNCTION DOC]
    # Checks cumprod and cumsum against NumPy for empty, int, float inputs, with explicit dtypes and axes.

    def run_test(arr, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner driver of testCumProdAndSum; for each transform asserts both functions match NumPy.
      for fn in self.array_transforms:
        arg = fn(arr)
        self.match(
            np_array_ops.cumprod(arg, *args, **kwargs),
            np.cumprod(arg, *args, **kwargs))
        self.match(
            np_array_ops.cumsum(arg, *args, **kwargs),
            np.cumsum(arg, *args, **kwargs))

    run_test([])
    run_test([1, 2, 3])
    run_test([1, 2, 3], dtype=float)
    run_test([1, 2, 3], dtype=np.float32)
    run_test([1, 2, 3], dtype=np.float64)
    run_test([1., 2., 3.])
    run_test([1., 2., 3.], dtype=int)
    run_test([1., 2., 3.], dtype=np.int32)
    run_test([1., 2., 3.], dtype=np.int64)
    run_test([[1, 2], [3, 4]], axis=1)
    run_test([[1, 2], [3, 4]], axis=0)
    run_test([[1, 2], [3, 4]], axis=-1)
    run_test([[1, 2], [3, 4]], axis=-2)

  def testImag(self):
    # [FUNCTION DOC]
    # Checks imag against NumPy for real and complex scalars, empty and lists (comparing to an array built
    # from the NumPy result because NumPy may return a scalar).

    def run_test(arr, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner driver of testImag; for each transform asserts the result matches NumPy.
      for fn in self.array_transforms:
        arg = fn(arr)
        self.match(
            np_array_ops.imag(arg, *args, **kwargs),
            # np.imag may return a scalar so we convert to a np.ndarray.
            np.array(np.imag(arg, *args, **kwargs)))

    run_test(1)
    run_test(5.5)
    run_test(5 + 3j)
    run_test(3j)
    run_test([])
    run_test([1, 2, 3])
    run_test([1 + 5j, 2 + 3j])
    run_test([[1 + 5j, 2 + 3j], [1 + 7j, 2 + 8j]])

  def testAMaxAMin(self):
    # [FUNCTION DOC]
    # Checks amax and amin against NumPy for several inputs, axes (including negative and tuple), keepdims,
    # with axis passed in each representation. Also asserts passing an output argument raises a value error.

    def run_test(arr, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner driver of testAMaxAMin; pops the axis keyword, crosses array and axis transforms and
      # asserts both amax and amin match NumPy.
      axis = kwargs.pop('axis', None)
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arr_arg = fn1(arr)
          axis_arg = fn2(axis) if axis is not None else None
          self.match(
              np_array_ops.amax(arr_arg, axis=axis_arg, *args, **kwargs),
              np.amax(arr_arg, axis=axis, *args, **kwargs))
          self.match(
              np_array_ops.amin(arr_arg, axis=axis_arg, *args, **kwargs),
              np.amin(arr_arg, axis=axis, *args, **kwargs))

    run_test([1, 2, 3])
    run_test([1., 2., 3.])
    run_test([[1, 2], [3, 4]], axis=1)
    run_test([[1, 2], [3, 4]], axis=0)
    run_test([[1, 2], [3, 4]], axis=-1)
    run_test([[1, 2], [3, 4]], axis=-2)
    run_test([[1, 2], [3, 4]], axis=(0, 1))
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(0, 2))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), axis=(0, 2), keepdims=True)
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(2, 0))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), axis=(2, 0), keepdims=True)
    self.assertRaises(ValueError, np_array_ops.amax, np.ones([2, 2]), out=[])
    self.assertRaises(ValueError, np_array_ops.amin, np.ones([2, 2]), out=[])

  def testMean(self):
    # [FUNCTION DOC]
    # Checks mean against NumPy for ints, floats, explicit int dtype, axes, tuple axes and keepdims, and
    # asserts an output argument raises a value error.

    def run_test(arr, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner driver of testMean; crosses array and axis transforms and asserts the result matches NumPy.
      axis = kwargs.pop('axis', None)
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arr_arg = fn1(arr)
          axis_arg = fn2(axis) if axis is not None else None
          self.match(
              np_array_ops.mean(arr_arg, axis=axis_arg, *args, **kwargs),
              np.mean(arr_arg, axis=axis, *args, **kwargs))

    run_test([1, 2, 1])
    run_test([1., 2., 1.])
    run_test([1., 2., 1.], dtype=int)
    run_test([[1, 2], [3, 4]], axis=1)
    run_test([[1, 2], [3, 4]], axis=0)
    run_test([[1, 2], [3, 4]], axis=-1)
    run_test([[1, 2], [3, 4]], axis=-2)
    run_test([[1, 2], [3, 4]], axis=(0, 1))
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(0, 2))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), axis=(0, 2), keepdims=True)
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(2, 0))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), axis=(2, 0), keepdims=True)
    self.assertRaises(ValueError, np_array_ops.mean, np.ones([2, 2]), out=[])

  def testStd(self):
    # [FUNCTION DOC]
    # Checks std against NumPy for ints, floats, complex values, axes, tuple axes and keepdims. Unlike
    # the variance test, no output-argument check is made.

    def run_test(arr, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner driver of testStd; crosses array and axis transforms and asserts the result matches NumPy.
      axis = kwargs.pop('axis', None)
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arr_arg = fn1(arr)
          axis_arg = fn2(axis) if axis is not None else None
          self.match(
              np_array_ops.std(arr_arg, axis=axis_arg, *args, **kwargs),
              np.std(arr_arg, axis=axis, *args, **kwargs))

    run_test([1, 2, 1])
    run_test([1., 2., 1.])
    run_test([1.j, 2., 1.j])
    run_test([[1, 2], [3, 4]], axis=1)
    run_test([[1, 2], [3, 4]], axis=0)
    run_test([[1, 2], [3, 4]], axis=-1)
    run_test([[1, 2], [3, 4]], axis=-2)
    run_test([[1, 2], [3, 4]], axis=(0, 1))
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(0, 2))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), axis=(0, 2), keepdims=True)
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(2, 0))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), axis=(2, 0), keepdims=True)

  def testVar(self):
    # [FUNCTION DOC]
    # Checks var against NumPy for ints, floats, complex, int64 dtype, axes and keepdims, and asserts an
    # output argument raises a value error.

    def run_test(arr, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner driver of testVar; crosses array and axis transforms and asserts the result matches NumPy.
      axis = kwargs.pop('axis', None)
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arr_arg = fn1(arr)
          axis_arg = fn2(axis) if axis is not None else None
          self.match(
              np_array_ops.var(arr_arg, axis=axis_arg, *args, **kwargs),
              np.var(arr_arg, axis=axis, *args, **kwargs))

    run_test([1, 2, 1])
    run_test([1., 2., 1.])
    run_test([1.j, 2., 1.j])
    run_test([1., 2., 1.], dtype=np.int64)
    run_test([[1, 2], [3, 4]], axis=1)
    run_test([[1, 2], [3, 4]], axis=0)
    run_test([[1, 2], [3, 4]], axis=-1)
    run_test([[1, 2], [3, 4]], axis=-2)
    run_test([[1, 2], [3, 4]], axis=(0, 1))
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(0, 2))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), axis=(0, 2), keepdims=True)
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(2, 0))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), axis=(2, 0), keepdims=True)
    self.assertRaises(ValueError, np_array_ops.var, np.ones([2, 2]), out=[])

  def testProd(self):
    # [FUNCTION DOC]
    # Checks prod against NumPy for ints, floats, int16 arrays, axes, tuple axes and keepdims.

    def run_test(arr, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner driver of testProd; for each transform asserts the result matches NumPy.
      for fn in self.array_transforms:
        arg = fn(arr)
        self.match(
            np_array_ops.prod(arg, *args, **kwargs),
            np.prod(arg, *args, **kwargs))

    run_test([1, 2, 3])
    run_test([1., 2., 3.])
    run_test(np.array([1, 2, 3], dtype=np.int16))
    run_test([[1, 2], [3, 4]], axis=1)
    run_test([[1, 2], [3, 4]], axis=0)
    run_test([[1, 2], [3, 4]], axis=-1)
    run_test([[1, 2], [3, 4]], axis=-2)
    run_test([[1, 2], [3, 4]], axis=(0, 1))
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(0, 2))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), axis=(0, 2), keepdims=True)
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(2, 0))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), axis=(2, 0), keepdims=True)

  def _testReduce(self, math_fun, np_fun, name):
    # [FUNCTION DOC]
    # HELPER: shared reduction test driver. Inputs: math_fun the TF NumPy function, np_fun the NumPy
    # reference, name a label for failure messages. Runs both over scalars, vectors and a matrix with
    # axes none, 0, 1, tuples and keepdims, with the axis given in each of several representations, and
    # asserts matching results with a descriptive message.
    axis_transforms = [
        lambda x: x,  # Identity,
        ops.convert_to_tensor,
        np.array,
        np_array_ops.array,
        lambda x: np_array_ops.array(x, dtype=np.float32),
        lambda x: np_array_ops.array(x, dtype=np.float64),
    ]

    def run_test(a, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner driver of _testReduce; pops the axis keyword, crosses array and axis transforms, and
      # asserts the TF NumPy result matches NumPy.
      axis = kwargs.pop('axis', None)
      for fn1 in self.array_transforms:
        for fn2 in axis_transforms:
          arg1 = fn1(a)
          axis_arg = fn2(axis) if axis is not None else None
          self.match(
              math_fun(arg1, axis=axis_arg, **kwargs),
              np_fun(arg1, axis=axis, **kwargs),
              msg='{}({}, axis={}, keepdims={})'.format(name, arg1, axis,
                                                        kwargs.get('keepdims')))

    run_test(5)
    run_test([2, 3])
    run_test([[2, -3], [-6, 7]])
    run_test([[2, -3], [-6, 7]], axis=0)
    run_test([[2, -3], [-6, 7]], axis=0, keepdims=True)
    run_test([[2, -3], [-6, 7]], axis=1)
    run_test([[2, -3], [-6, 7]], axis=1, keepdims=True)
    run_test([[2, -3], [-6, 7]], axis=(0, 1))
    run_test([[2, -3], [-6, 7]], axis=(1, 0))

  def testSum(self):
    # [FUNCTION DOC]
    # Runs the shared reduction test for sum against NumPy sum.
    self._testReduce(np_array_ops.sum, np.sum, 'sum')

  def testAmax(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: this is a second test of amax (testAMaxAMin already covers it), run through the shared reduction driver.
    # Runs the shared reduction test for the maximum function against NumPy.
    self._testReduce(np_array_ops.amax, np.amax, 'amax')

  def testSize(self):
    # [FUNCTION DOC]
    # Checks size against NumPy for arrays, scalars and tensors, with and without an axis; asserts a
    # call with an axis on a NumPy array raises a not-implemented error. Also traces a function with a
    # float64 input signature that returns the size and asserts it gives 6 for a three-by-two input.

    def run_test(arr, axis=None):
      # [FUNCTION DOC]
      # HELPER: inner driver of testSize. Inputs arr and optional axis; asserts size equals the NumPy size.
      onp_arr = np.array(arr)
      self.assertEqual(np_array_ops.size(arr, axis), np.size(onp_arr, axis))

    run_test(np_array_ops.array([1]))
    run_test(np_array_ops.array([1, 2, 3, 4, 5]))
    run_test(np_array_ops.ones((2, 3, 2)))
    run_test(np_array_ops.ones((3, 2)))
    run_test(np_array_ops.zeros((5, 6, 7)))
    run_test(1)
    run_test(np_array_ops.ones((3, 2, 1)))
    run_test(constant_op.constant(5))
    run_test(constant_op.constant([1, 1, 1]))
    self.assertRaises(NotImplementedError, np_array_ops.size, np.ones((2, 2)),
                      1)

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(dtype=dtypes.float64, shape=None)])
    def f(arr):
      # [FUNCTION DOC]
      # HELPER: traced function of testSize. Input arr is a float64 tensor; converts it to an array and returns its size.
      arr = np_array_ops.asarray(arr)
      return np_array_ops.size(arr)

    self.assertEqual(f(np_array_ops.ones((3, 2))).numpy(), 6)

  def testRavel(self):
    # [FUNCTION DOC]
    # Checks ravel against NumPy for scalars, empty and nested lists.

    def run_test(arr, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner driver of testRavel; for each transform asserts the result matches NumPy.
      for fn in self.array_transforms:
        arg = fn(arr)
        self.match(
            np_array_ops.ravel(arg, *args, **kwargs),
            np.ravel(arg, *args, **kwargs))

    run_test(5)
    run_test(5.)
    run_test([])
    run_test([[]])
    run_test([[], []])
    run_test([1, 2, 3])
    run_test([1., 2., 3.])
    run_test([[1, 2], [3, 4]])
    run_test(np.arange(8).reshape((2, 2, 2)).tolist())

  def testReal(self):
    # [FUNCTION DOC]
    # Checks real against NumPy for real and complex scalars, empty and lists (via an array built from the NumPy result).

    def run_test(arr, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner driver of testReal; for each transform asserts the result matches NumPy.
      for fn in self.array_transforms:
        arg = fn(arr)
        self.match(
            np_array_ops.real(arg, *args, **kwargs),
            np.array(np.real(arg, *args, **kwargs)))

    run_test(1)
    run_test(5.5)
    run_test(5 + 3j)
    run_test(3j)
    run_test([])
    run_test([1, 2, 3])
    run_test([1 + 5j, 2 + 3j])
    run_test([[1 + 5j, 2 + 3j], [1 + 7j, 2 + 8j]])

  def testRepeat(self):
    # [FUNCTION DOC]
    # Checks repeat against NumPy with scalar and list repeat counts, with and without axes (including negative).

    def run_test(arr, repeats, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner driver of testRepeat; crosses array and repeats transforms and asserts a match with NumPy.
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arr_arg = fn1(arr)
          repeats_arg = fn2(repeats)
          self.match(
              np_array_ops.repeat(arr_arg, repeats_arg, *args, **kwargs),
              np.repeat(arr_arg, repeats_arg, *args, **kwargs))

    run_test(1, 2)
    run_test([1, 2], 2)
    run_test([1, 2], [2])
    run_test([1, 2], [1, 2])
    run_test([[1, 2], [3, 4]], 3, axis=0)
    run_test([[1, 2], [3, 4]], 3, axis=1)
    run_test([[1, 2], [3, 4]], [3], axis=0)
    run_test([[1, 2], [3, 4]], [3], axis=1)
    run_test([[1, 2], [3, 4]], [3, 2], axis=0)
    run_test([[1, 2], [3, 4]], [3, 2], axis=1)
    run_test([[1, 2], [3, 4]], [3, 2], axis=-1)
    run_test([[1, 2], [3, 4]], [3, 2], axis=-2)

  def testAround(self):
    # [FUNCTION DOC]
    # Checks around against NumPy for scalars, empty and lists with positive, zero and negative decimals.

    def run_test(arr, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner driver of testAround; for each transform asserts the result matches NumPy.
      for fn in self.array_transforms:
        arg = fn(arr)
        self.match(
            np_array_ops.around(arg, *args, **kwargs),
            np.around(arg, *args, **kwargs))

    run_test(5.5)
    run_test(5.567, decimals=2)
    run_test([])
    run_test([1.27, 2.49, 2.75], decimals=1)
    run_test([23.6, 45.1], decimals=-1)

  def testReshape(self):
    # [FUNCTION DOC]
    # Checks reshape against NumPy for scalars, empty and vectors into various new shapes.

    def run_test(arr, newshape, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner driver of testReshape; crosses array and shape transforms and asserts a match with NumPy.
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arr_arg = fn1(arr)
          newshape_arg = fn2(newshape)
          self.match(
              np_array_ops.reshape(arr_arg, newshape_arg, *args, **kwargs),
              np.reshape(arr_arg, newshape, *args, **kwargs))

    run_test(5, [-1])
    run_test([], [-1])
    run_test([1, 2, 3], [1, 3])
    run_test([1, 2, 3], [3, 1])
    run_test([1, 2, 3, 4], [2, 2])
    run_test([1, 2, 3, 4], [2, 1, 2])

  def testExpandDims(self):
    # [FUNCTION DOC]
    # Checks expand_dims against NumPy for a vector at axes 0 and 1.

    def run_test(arr, axis):
      # [FUNCTION DOC]
      # HELPER: inner driver of testExpandDims; asserts the result for the given array and axis matches NumPy.
      self.match(np_array_ops.expand_dims(arr, axis), np.expand_dims(arr, axis))

    run_test([1, 2, 3], 0)
    run_test([1, 2, 3], 1)

  def testSqueeze(self):
    # [FUNCTION DOC]
    # Checks squeeze against NumPy for scalars, empty and arrays with unit dimensions, with various axes.
    # The reference is computed on a NumPy array copy because NumPy ignores the axis for non-array inputs.

    def run_test(arr, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner driver of testSqueeze; for each transform asserts the result matches NumPy squeeze.
      for fn in self.array_transforms:
        arg = fn(arr)
        # Note: np.squeeze ignores the axis arg for non-ndarray objects.
        # This looks like a bug: https://github.com/numpy/numpy/issues/8201
        # So we convert the arg to np.ndarray before passing to np.squeeze.
        self.match(
            np_array_ops.squeeze(arg, *args, **kwargs),
            np.squeeze(np.array(arg), *args, **kwargs))

    run_test(5)
    run_test([])
    run_test([5])
    run_test([[1, 2, 3]])
    run_test([[[1], [2], [3]]])
    run_test([[[1], [2], [3]]], axis=0)
    run_test([[[1], [2], [3]]], axis=2)
    run_test([[[1], [2], [3]]], axis=(0, 2))
    run_test([[[1], [2], [3]]], axis=-1)
    run_test([[[1], [2], [3]]], axis=-3)

  def testTranspose(self):
    # [FUNCTION DOC]
    # Checks transpose against NumPy for scalars, empty, vectors and a three-dimensional array with all
    # six axis permutations, with the axes in each representation.

    def run_test(arr, axes=None):
      # [FUNCTION DOC]
      # HELPER: inner driver of testTranspose; crosses array and axes transforms and asserts a match with NumPy.
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arr_arg = fn1(arr)
          axes_arg = fn2(axes) if axes is not None else None
          self.match(
              np_array_ops.transpose(arr_arg, axes_arg),
              np.transpose(arr_arg, axes))

    run_test(5)
    run_test([])
    run_test([5])
    run_test([5, 6, 7])
    run_test(np.arange(30).reshape(2, 3, 5).tolist())
    run_test(np.arange(30).reshape(2, 3, 5).tolist(), [0, 1, 2])
    run_test(np.arange(30).reshape(2, 3, 5).tolist(), [0, 2, 1])
    run_test(np.arange(30).reshape(2, 3, 5).tolist(), [1, 0, 2])
    run_test(np.arange(30).reshape(2, 3, 5).tolist(), [1, 2, 0])
    run_test(np.arange(30).reshape(2, 3, 5).tolist(), [2, 0, 1])
    run_test(np.arange(30).reshape(2, 3, 5).tolist(), [2, 1, 0])

  def match_shape(self, actual, expected, msg=None):
    # [FUNCTION DOC]
    # HELPER: assertion helper for this test case. Asserts the two shapes are equal; msg, if given, is
    # replaced by a formatted message naming both shapes.
    if msg:
      msg = 'Shape match failed for: {}. Expected: {} Actual: {}'.format(
          msg, expected.shape, actual.shape)
    self.assertEqual(actual.shape, expected.shape, msg=msg)

  def match_dtype(self, actual, expected, msg=None):
    # [FUNCTION DOC]
    # HELPER: assertion helper for this test case. Asserts the two dtypes are equal; msg, if given, is
    # replaced by a formatted message naming both dtypes.
    if msg:
      msg = 'Dtype match failed for: {}. Expected: {} Actual: {}.'.format(
          msg, expected.dtype, actual.dtype)
    self.assertEqual(actual.dtype, expected.dtype, msg=msg)

  def match(self, actual, expected, msg=None, check_dtype=True):
    # [FUNCTION DOC]
    # KNOWN ISSUE: both branches of the rank check call the same close-equality assertion, so the check is redundant.
    # HELPER: main comparison assertion. Inputs: actual, expected, msg optional, check_dtype (default true).
    # Asserts actual is a TF NumPy array, optionally dtype equal, shape equal, and values close in tolerance.
    msg_ = 'Expected: {} Actual: {}'.format(expected, actual)
    if msg:
      msg = '{} {}'.format(msg_, msg)
    else:
      msg = msg_
    self.assertIsInstance(actual, np_arrays.ndarray)
    if check_dtype:
      self.match_dtype(actual, expected, msg)
    self.match_shape(actual, expected, msg)
    if not actual.shape.rank:
      self.assertAllClose(actual.tolist(), expected.tolist())
    else:
      self.assertAllClose(actual.tolist(), expected.tolist())

  def testPad(self):
    # [FUNCTION DOC]
    # Checks pad on a two-by-three matrix with padding of one row and two columns in constant, reflect and
    # symmetric modes against the expected explicit matrices.
    t = [[1, 2, 3], [4, 5, 6]]
    paddings = [[
        1,
        1,
    ], [2, 2]]
    self.assertAllEqual(
        np_array_ops.pad(t, paddings, 'constant'),
        [[0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 2, 3, 0, 0], [0, 0, 4, 5, 6, 0, 0],
         [0, 0, 0, 0, 0, 0, 0]])

    self.assertAllEqual(
        np_array_ops.pad(t, paddings, 'reflect'),
        [[6, 5, 4, 5, 6, 5, 4], [3, 2, 1, 2, 3, 2, 1], [6, 5, 4, 5, 6, 5, 4],
         [3, 2, 1, 2, 3, 2, 1]])

    self.assertAllEqual(
        np_array_ops.pad(t, paddings, 'symmetric'),
        [[2, 1, 1, 2, 3, 3, 2], [2, 1, 1, 2, 3, 3, 2], [5, 4, 4, 5, 6, 6, 5],
         [5, 4, 4, 5, 6, 6, 5]])

  def testTake(self):
    # [FUNCTION DOC]
    # Checks take with flat indices, nested indices, a 2D source, and a random three-dimensional source
    # with axis one against NumPy.
    a = [4, 3, 5, 7, 6, 8]
    indices = [0, 1, 4]
    self.assertAllEqual([4, 3, 6], np_array_ops.take(a, indices))
    indices = [[0, 1], [2, 3]]
    self.assertAllEqual([[4, 3], [5, 7]], np_array_ops.take(a, indices))
    a = [[4, 3, 5], [7, 6, 8]]
    self.assertAllEqual([[4, 3], [5, 7]], np_array_ops.take(a, indices))
    a = np.random.rand(2, 16, 3)
    axis = 1
    self.assertAllEqual(
        np.take(a, indices, axis=axis),
        np_array_ops.take(a, indices, axis=axis))

  def testTakeAlongAxis(self):
    # [FUNCTION DOC]
    # Checks take_along_axis on a random 2x3 float matrix with random int64 indices along axis 1 against NumPy.
    rng = np.random.default_rng()
    x = rng.standard_normal((2, 3)).astype(np.float32)
    ind = rng.integers(0, 3, (2, 5)).astype(np.int64)
    out_expected = np.take_along_axis(x, ind, axis=1)
    out = np_array_ops.take_along_axis(x, ind, axis=1)
    self.assertAllEqual(out, out_expected)

  def testWhere(self):
    # [FUNCTION DOC]
    # Checks where with a one-element condition broadcast against two arrays gives a matrix of ones.
    self.assertAllEqual([[1.0, 1.0], [1.0, 1.0]],
                        np_array_ops.where([True], [1.0, 1.0],
                                           [[0, 0], [0, 0]]))

  def testShape(self):
    # [FUNCTION DOC]
    # Checks the shape function returns one by two for a nested list of one row of two.
    self.assertAllEqual((1, 2), np_array_ops.shape([[0, 0]]))

  def testSwapaxes(self):
    # [FUNCTION DOC]
    # Checks swapaxes on a 1x3 matrix and a 2x2x2 array with positive and negative axis pairs.
    x = [[1, 2, 3]]
    self.assertAllEqual([[1], [2], [3]], np_array_ops.swapaxes(x, 0, 1))
    self.assertAllEqual([[1], [2], [3]], np_array_ops.swapaxes(x, -2, -1))
    x = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    self.assertAllEqual([[[0, 4], [2, 6]], [[1, 5], [3, 7]]],
                        np_array_ops.swapaxes(x, 0, 2))
    self.assertAllEqual([[[0, 4], [2, 6]], [[1, 5], [3, 7]]],
                        np_array_ops.swapaxes(x, -3, -1))

  def testMoveaxis(self):
    # [FUNCTION DOC]
    # Checks moveaxis on a random six-dimensional array with many source and destination combinations
    # (positive, negative, ranges, reversed ranges, empty) against NumPy.

    def _test(*args):
      # [FUNCTION DOC]
      # HELPER: inner driver of testMoveaxis; passes its arguments to both NumPy and TF NumPy moveaxis and asserts equal.
      # pylint: disable=no-value-for-parameter
      expected = np.moveaxis(*args)
      raw_ans = np_array_ops.moveaxis(*args)

      self.assertAllEqual(expected, raw_ans)

    a = np.random.rand(1, 2, 3, 4, 5, 6)

    # Basic
    _test(a, (0, 2), (3, 5))
    _test(a, (0, 2), (-1, -3))
    _test(a, (-6, -4), (3, 5))
    _test(a, (-6, -4), (-1, -3))
    _test(a, 0, 4)
    _test(a, -6, -2)
    _test(a, tuple(range(6)), tuple(range(6)))
    _test(a, tuple(range(6)), tuple(reversed(range(6))))
    _test(a, (), ())

  def testNdim(self):
    # [FUNCTION DOC]
    # Checks ndim returns 0 for a scalar and 1 for a two-element list.
    self.assertAllEqual(0, np_array_ops.ndim(0.5))
    self.assertAllEqual(1, np_array_ops.ndim([1, 2]))

  def testIsscalar(self):
    # [FUNCTION DOC]
    # Checks isscalar is true for float, int and boolean and false for a list.
    self.assertTrue(np_array_ops.isscalar(0.5))
    self.assertTrue(np_array_ops.isscalar(5))
    self.assertTrue(np_array_ops.isscalar(False))
    self.assertFalse(np_array_ops.isscalar([1, 2]))

  def assertListEqual(self, a, b):
    # [FUNCTION DOC]
    # KNOWN ISSUE: overrides the standard list-equality assertion of the base test class with a different
    # KNOWN ISSUE: meaning (compare lengths, then each element with array-aware equality), which can confuse readers.
    # HELPER: assertion helper for the split tests. Inputs a and b are lists of arrays; asserts equal
    # lengths and that each pair of elements is equal.
    self.assertAllEqual(len(a), len(b))
    for x, y in zip(a, b):
      self.assertAllEqual(x, y)

  def testSplit(self):
    # [FUNCTION DOC]
    # Checks split of arange(9) into three equal parts and of arange(8) at indices 3, 5, 6, 10 (giving
    # five parts, the last empty).
    x = np_array_ops.arange(9)
    y = np_array_ops.split(x, 3)
    self.assertListEqual([([0, 1, 2]), ([3, 4, 5]), ([6, 7, 8])], y)

    x = np_array_ops.arange(8)
    y = np_array_ops.split(x, [3, 5, 6, 10])
    self.assertListEqual([([0, 1, 2]), ([3, 4]), ([5]), ([6, 7]), ([])], y)

  def testHSplitBecomesVsplitFor1DInput(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says hsplit becomes vsplit for one-dimensional input, but the test only
    # KNOWN ISSUE: asserts hsplit into two halves of a 1D array inside a traced function.
    # Traces a function that calls hsplit with two sections on a 1D array; asserts the halves are [0,1] and [2,3].
    @def_function.function
    def f(arr):
      # [FUNCTION DOC]
      # HELPER: traced function of the hsplit test. Input arr is an array; returns hsplit into two parts.
      return np_array_ops.hsplit(arr, 2)

    x = np_array_ops.arange(4)
    self.assertListEqual([[0, 1], [2, 3]], f(x))

  def testSign(self):
    # [FUNCTION DOC]
    # Checks sign against NumPy for float, int and complex dtypes and shapes including empty dimensions,
    # using seeded random data.
    state = np.random.RandomState(0)
    test_types = [np.float16, np.float32, np.float64, np.int32, np.int64,
                  np.complex64, np.complex128]
    test_shapes = [(), (1,), (2, 3, 4), (2, 3, 0, 4)]

    for dtype in test_types:
      for shape in test_shapes:
        if np.issubdtype(dtype, np.complexfloating):
          arr = (np.asarray(state.randn(*shape) * 100, dtype=dtype) +
                 1j * np.asarray(state.randn(*shape) * 100, dtype=dtype))
        else:
          arr = np.asarray(state.randn(*shape) * 100, dtype=dtype)
        self.match(np_array_ops.sign(arr), np.sign(arr))


class ArrayManipulationTest(test.TestCase):
  # [CLASS DOC]
  # Tests for array manipulation functions broadcast_to and ix_. setUp provides array_transforms; match
  # helpers compare dtype, shape and values.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture. Calls the base setUp and stores array_transforms (identity, tensor conversion, NumPy array, TF NumPy array).
    super(ArrayManipulationTest, self).setUp()
    self.array_transforms = [
        lambda x: x,
        ops.convert_to_tensor,
        np.array,
        np_array_ops.array,
    ]

  def testBroadcastTo(self):
    # [FUNCTION DOC]
    # Checks broadcast_to against NumPy for scalars, vectors and nested lists into several target shapes.

    def run_test(arr, shape):
      # [FUNCTION DOC]
      # HELPER: inner driver of testBroadcastTo; for each transform asserts the result matches NumPy.
      for fn in self.array_transforms:
        arg1 = fn(arr)
        self.match(
            np_array_ops.broadcast_to(arg1, shape),
            np.broadcast_to(arg1, shape))

    run_test(1, 2)
    run_test(1, (2, 2))
    run_test([1, 2], (2, 2))
    run_test([[1], [2]], (2, 2))
    run_test([[1, 2]], (3, 2))
    run_test([[[1, 2]], [[3, 4]], [[5, 6]]], (3, 4, 2))

  def testIx_(self):
    # [FUNCTION DOC]
    # Checks ix_ against NumPy for combinations (with replacement) of boolean lists, an int range and an
    # empty int64 array, comparing each returned array.
    possible_arys = [[True, True], [True, False], [False, False],
                     list(range(5)), np_array_ops.empty(0, dtype=np.int64)]
    for r in range(len(possible_arys)):
      for arys in itertools.combinations_with_replacement(possible_arys, r):
        tnp_ans = np_array_ops.ix_(*arys)
        onp_ans = np.ix_(*arys)
        for t, o in zip(tnp_ans, onp_ans):
          self.match(t, o)

  def match_shape(self, actual, expected, msg=None):
    # [FUNCTION DOC]
    # HELPER: assertion helper. Asserts equal shapes; msg, if given, is replaced by a formatted shape message.
    if msg:
      msg = 'Shape match failed for: {}. Expected: {} Actual: {}'.format(
          msg, expected.shape, actual.shape)
    self.assertEqual(actual.shape, expected.shape, msg=msg)

  def match_dtype(self, actual, expected, msg=None):
    # [FUNCTION DOC]
    # HELPER: assertion helper. Asserts equal dtypes; msg, if given, is replaced by a formatted dtype message.
    if msg:
      msg = 'Dtype match failed for: {}. Expected: {} Actual: {}.'.format(
          msg, expected.dtype, actual.dtype)
    self.assertEqual(actual.dtype, expected.dtype, msg=msg)

  def match(self, actual, expected, msg=None):
    # [FUNCTION DOC]
    # HELPER: main comparison assertion. Inputs actual, expected, msg optional. Asserts actual is a TF NumPy
    # array, then dtype, shape, and exact values (list equality for scalars, sequence equality otherwise).
    msg_ = 'Expected: {} Actual: {}'.format(expected, actual)
    if msg:
      msg = '{} {}'.format(msg_, msg)
    else:
      msg = msg_
    self.assertIsInstance(actual, np_arrays.ndarray)
    self.match_dtype(actual, expected, msg)
    self.match_shape(actual, expected, msg)
    if not actual.shape.rank:
      self.assertEqual(actual.tolist(), expected.tolist())
    else:
      self.assertSequenceEqual(actual.tolist(), expected.tolist())


class ArrayMathTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for arithmetic between Python constants and TF NumPy arrays.

  @parameterized.named_parameters(
      ('complex_mul_1', 2j, [2], int, [4j], operator.mul),
      ('complex_mul_2', 2j, [0], int, [0], operator.mul),
      ('complex_mul_3', 2j, [-2.0], float, [-4j], operator.mul),
      ('complex_mul_4', 2j, [2j], complex, [-4], operator.mul),
      ('float_mul_1', 2.0, [2], int, [4], operator.mul),
      ('float_mul_2', 2.0, [0], int, [0], operator.mul),
      ('float_mul_3', 2.0, [-2.0], float, [-4], operator.mul),
      ('float_mul_4', 2.0, [2j], complex, [4j], operator.mul))
  def testConstantBinOp(self, a, b, b_type, expected_result, test_func):
    # [FUNCTION DOC]
    # Parameterized by named cases (complex and float constants multiplied by int, float or complex arrays).
    # Inputs: a the constant, b the list value, b_type its dtype, expected_result the expected values,
    # test_func the operator. Creates an array from b, applies the operator with a first, then asserts the
    # result is close (inexact dtype) or exactly equal (otherwise) to the expected result.
    b = np_array_ops.array(b, dtype=b_type)
    result = test_func(a, b)
    if np.issubdtype(result.dtype.as_numpy_dtype, np.inexact):
      self.assertAllClose(result, expected_result)
    else:
      self.assertAllEqual(result, expected_result)


class StringArrayTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for string and bytes inputs to array creation functions. Defines a class-level
  # decorator carrying named cases: text (with and without embedded zero characters) and bytes inputs,
  # with dtype U8, S8 or none, each expected as bytes because TensorFlow always encodes text as bytes.

  StringParameters = parameterized.named_parameters(  # pylint: disable=invalid-name
      # Tensorflow always encodes python string into bytes, regardless of
      # requested dtype.
      ('str_u8', 'abcde\U0001f005', 'U8', b'abcde\xf0\x9f\x80\x85'),
      ('str_s8', 'abcde\U0001f005', 'S8', b'abcde\xf0\x9f\x80\x85'),
      ('str_none', 'abcde\U0001f005', None, b'abcde\xf0\x9f\x80\x85'),
      ('zstr_u8', '\0abcde\U0001f005', 'U8', b'\0abcde\xf0\x9f\x80\x85'),
      ('zstr_s8', '\0abcde\U0001f005', 'S8', b'\0abcde\xf0\x9f\x80\x85'),
      ('zstr_none', '\0abcde\U0001f005', None, b'\0abcde\xf0\x9f\x80\x85'),
      ('bytes_u8', b'abcdef', 'U8', b'abcdef'),
      ('bytes_s8', b'abcdef', 'S8', b'abcdef'),
      ('bytes_none', b'abcdef', None, b'abcdef'),
      ('zbytes_u8', b'\0abcdef', 'U8', b'\0abcdef'),
      ('zbytes_s8', b'\0abcdef', 'S8', b'\0abcdef'),
      ('zbytes_none', b'\0abcdef', None, b'\0abcdef'),
  )

  @StringParameters
  def testArray(self, a, dtype, a_as_bytes):
    # [FUNCTION DOC]
    # Parameterized (a input, dtype, a_as_bytes expected). Creates an array with the array function and
    # asserts its NumPy value is a bytes object equal to the expected bytes.
    b = np_array_ops.array(a, dtype=dtype)
    self.assertIsInstance(b.numpy(), bytes)
    self.assertEqual(b.numpy(), a_as_bytes)

  @StringParameters
  def testAsArray(self, a, dtype, a_as_bytes):
    # [FUNCTION DOC]
    # Parameterized like testArray but uses the asarray function; asserts bytes with the expected value.
    b = np_array_ops.asarray(a, dtype=dtype)
    self.assertIsInstance(b.numpy(), bytes)
    self.assertEqual(b.numpy(), a_as_bytes)

  @StringParameters
  def testZerosLike(self, a, dtype, unused_a_as_bytes):
    # [FUNCTION DOC]
    # Parameterized (unused expected-bytes). Calls zeros_like on the string with the dtype and asserts the
    # value is an empty bytes object.
    b = np_array_ops.zeros_like(a, dtype=dtype)
    self.assertIsInstance(b.numpy(), bytes)
    self.assertEqual(b.numpy(), b'')

  @StringParameters
  def testEmptyLike(self, a, dtype, unused_a_as_bytes):
    # [FUNCTION DOC]
    # Parameterized (unused expected-bytes). Calls empty_like on the string with the dtype and asserts the
    # value is an empty bytes object.
    b = np_array_ops.empty_like(a, dtype=dtype)
    self.assertIsInstance(b.numpy(), bytes)
    self.assertEqual(b.numpy(), b'')


if __name__ == '__main__':
  ops.enable_eager_execution()
  ops.set_dtype_conversion_mode('legacy')
  np_math_ops.enable_numpy_methods_on_tensor()
  test.main()
