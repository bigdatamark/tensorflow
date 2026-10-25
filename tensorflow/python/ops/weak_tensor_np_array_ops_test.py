# [MODULE DOC]
# Unit tests for TF-NumPy array functions and methods operating on weak tensors, covering array
# creation (empty_like, zeros_like, ones, full_like, array, asarray, arange, diag and others)
# and manipulation/reduction (copy, cumulative ops, imag/real, amax/amin, mean, std, var, prod,
# sum, size, ravel, repeat, around, reshape, squeeze, transpose, swapaxes, moveaxis). Verifies
# result type (weak tensor versus tensor), inferred dtype, shape and values against numpy. Imports
# the weak tensor ops module for its patching side effect, and defines a helper to split the CPU
# into two virtual devices. When run as a script it enables eager execution, sets dtype
# conversion mode to "all" and enables numpy methods on tensors before running tests.
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
"""Tests for tf numpy array methods on WeakTensor."""

import itertools
import sys

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import flexible_dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework.weak_tensor import WeakTensor
from tensorflow.python.ops import weak_tensor_ops  # pylint: disable=unused-import
from tensorflow.python.ops import weak_tensor_test_util
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_math_ops
from tensorflow.python.platform import test


_virtual_devices_ready = False
_get_weak_tensor = weak_tensor_test_util.get_weak_tensor

_all_types = [
    np.int32,
    np.int64,
    np.float32,
    np.float64,
    np.complex128,
]

_NP_TO_TF = dtypes._NP_TO_TF

_NP_to_TF_result_inferred_types = {
    np.dtype(np.int32): dtypes.int32,
    np.dtype(np.int64): dtypes.int32,
    np.dtype(np.float32): dtypes.float32,
    np.dtype(np.float64): dtypes.float32,
    np.dtype(np.complex128): dtypes.complex128,
    dtypes.int32: dtypes.int32,
    dtypes.int64: dtypes.int32,
    dtypes.float32: dtypes.float32,
    dtypes.float64: dtypes.float32,
    dtypes.complex128: dtypes.complex128,
}


def set_up_virtual_devices():
  # [FUNCTION DOC]
  # HELPER: shared test utility that prepares the device configuration for the tests in this file.
  # Splits the first physical CPU device into two logical (virtual) CPU devices, so tests can place
  # work on a second CPU device. Guarded by a module-level flag so it takes effect only once per
  # process; later calls return immediately. Takes no inputs and returns nothing. Side effects:
  # changes global device configuration and sets the module flag. Raises whatever the configuration
  # call raises if the runtime was already initialized with a different configuration.
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
  # Tests TF-NumPy array-creation functions (empty_like, zeros_like, ones, ones_like, full_like,
  # array, asarray, asanyarray, ascontiguousarray, arange, diag, diagflat) under the new dtype
  # conversion rules, checking that results are weak tensors for weak/python inputs without an
  # explicit dtype and plain tensors when a dtype is given. Fixture (built in setUp): lists of shape
  # and array-like inputs passed through several transforms. Provides comparison helpers for shape,
  # dtype and result type.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup. Calls the parent setup, configures two virtual CPU devices, then builds:
    # all_shapes (a collection of python, numpy, TF-NumPy and tensor-shape forms of several shapes,
    # with zero-dimensional TF-NumPy scalar shapes removed on Python 3 because they can crash numpy
    # empty-style functions), a list of array transforms (identity, TF-NumPy array, weak tensor) and
    # all_arrays (every sample array-like input under every transform). Returns nothing.
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
        # HELPER: filter predicate used while building the shape list in setUp. Takes a candidate shape
        # and returns false only if it is a zero-dimensional TF-NumPy array, true otherwise.
        return not (isinstance(s, np_arrays.ndarray) and s.ndim == 0)

      self.all_shapes = list(filter(not_ndarray_scalar, self.all_shapes))

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
        np_array_ops.array,
        _get_weak_tensor,
    ]
    self.all_arrays = []
    for fn in self.array_transforms:
      self.all_arrays.extend([fn(s) for s in source_array_data])

  def testEmptyLikeOnWeakInputs(self):
    # [FUNCTION DOC]
    # Test: empty_like on every sample input must match numpy in shape, and without a dtype must
    # return a weak tensor of the inferred TF dtype; with each explicit dtype it must return a plain
    # tensor with the requested dtype and the numpy shape.
    for a in self.all_arrays:
      expected = np.empty_like(a)
      actual = np_array_ops.empty_like(a)
      msg = 'array: {}'.format(a)
      self.match_shape(actual, expected, msg)
      self.match_dtype_and_type(
          actual,
          _NP_to_TF_result_inferred_types[expected.dtype],
          WeakTensor,
          msg,
      )

    for a, t in itertools.product(self.all_arrays, _all_types):
      actual = np_array_ops.empty_like(a, t)
      expected = np.empty_like(a, t)
      msg = 'array: {} type: {}'.format(a, t)
      self.match_shape(actual, expected, msg)
      # empty_like returns a Tensor if dtype is specified.
      self.match_dtype_and_type(actual, expected.dtype, tensor.Tensor, msg)

  def testZerosLikeOnWeakInputs(self):
    # [FUNCTION DOC]
    # Test: zeros_like on every sample input matches numpy values, shape and dtype, returning a weak
    # tensor of the inferred dtype without a dtype argument, and a plain tensor of the given dtype
    # when one of the supported dtypes is specified.
    for a in self.all_arrays:
      actual = np_array_ops.zeros_like(a)
      expected = np.zeros_like(a)
      msg = 'array: {}'.format(a)
      self.match_expected_attrs(
          actual,
          expected,
          _NP_to_TF_result_inferred_types[expected.dtype],
          WeakTensor,
          msg,
      )

    for a, t in itertools.product(self.all_arrays, _all_types):
      actual = np_array_ops.zeros_like(a, t)
      expected = np.zeros_like(a, t)
      msg = 'array: {} type: {}'.format(a, t)
      self.match_expected_attrs(
          actual, expected, expected.dtype, tensor.Tensor, msg
      )

  def testOnes(self):
    # [FUNCTION DOC]
    # Test: ones for every sample shape; without dtype the result is a plain tensor whose dtype is the
    # inferred TF dtype of numpy's default result, and with each explicit dtype it is a plain tensor of
    # that dtype; values and shape must equal numpy's.
    for s in self.all_shapes:
      actual = np_array_ops.ones(s)
      expected = np.ones(s)
      msg = 'shape: {}'.format(s)
      self.match_expected_attrs(
          actual,
          expected,
          _NP_to_TF_result_inferred_types[expected.dtype],
          tensor.Tensor,
          msg,
      )

    for s, t in itertools.product(self.all_shapes, _all_types):
      actual = np_array_ops.ones(s, t)
      expected = np.ones(s, t)
      msg = 'shape: {}, dtype: {}'.format(s, t)
      self.match_expected_attrs(
          actual, expected, expected.dtype, tensor.Tensor, msg
      )

  def testOnesLike(self):
    # [FUNCTION DOC]
    # Test: ones_like on every sample input matches numpy; weak tensor of inferred dtype when no dtype
    # is given, plain tensor of the requested dtype otherwise.
    for a in self.all_arrays:
      actual = np_array_ops.ones_like(a)
      expected = np.ones_like(a)
      msg = 'array: {}'.format(a)
      self.match_expected_attrs(
          actual,
          expected,
          _NP_to_TF_result_inferred_types[expected.dtype],
          WeakTensor,
          msg,
      )

    for a, t in itertools.product(self.all_arrays, _all_types):
      actual = np_array_ops.ones_like(a, t)
      expected = np.ones_like(a, t)
      msg = 'array: {} type: {}'.format(a, t)
      self.match_expected_attrs(
          actual, expected, expected.dtype, tensor.Tensor, msg
      )

  def testFullLike(self):
    # [FUNCTION DOC]
    # Test: full_like over several fill values (python scalars, lists, nested lists) and shapes, with
    # fill values under each transform and prototype arrays built with TF-NumPy or numpy zeros in each
    # dtype. The prototype is converted to a weak tensor; without dtype the result must be a weak
    # tensor with the numpy-expected dtype and values; with explicit dtype the prototype is used
    # directly and the result must be a plain tensor of that dtype.
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
      for fn1, fn2, arr_dtype in itertools.product(
          self.array_transforms, zeros_builders, _all_types
      ):
        fill_value = fn1(f)
        arr = fn2(s, arr_dtype)
        wt_arr = _get_weak_tensor(arr)
        expected = np.full_like(arr, fill_value)
        self.match_expected_attrs(
            np_array_ops.full_like(wt_arr, fill_value),
            expected,
            expected.dtype,
            WeakTensor,
        )
        for dtype in _all_types:
          self.match_expected_attrs(
              np_array_ops.full_like(arr, fill_value, dtype=dtype),
              np.full_like(arr, fill_value, dtype=dtype),
              _NP_TO_TF[dtype],
              tensor.Tensor,
          )

  def testArray(self):
    # [FUNCTION DOC]
    # Test: array over every sample input, dtype, minimum-dimension count (0, 1, 2, 5) and copy flag.
    # With dtype it must return a plain tensor of that dtype matching numpy; without dtype it must
    # return a weak tensor of the inferred dtype matching numpy values and shape.
    ndmins = [0, 1, 2, 5]
    for a, dtype, ndmin, copy in itertools.product(
        self.all_arrays, _all_types, ndmins, [True, False]
    ):
      # Dtype specified.
      self.match_expected_attrs(
          np_array_ops.array(a, dtype=dtype, ndmin=ndmin, copy=copy),
          np.array(a, dtype=dtype, ndmin=ndmin),
          dtype,
          tensor.Tensor,
      )
      # No dtype specified.
      actual = np_array_ops.array(a, ndmin=ndmin, copy=copy)
      expected = np.array(a, ndmin=ndmin)
      self.match_expected_attrs(
          actual,
          expected,
          _NP_to_TF_result_inferred_types[expected.dtype],
          WeakTensor,
      )

  def testAsArray(self):
    # [FUNCTION DOC]
    # Test: asarray over every sample input and dtype (plain tensor of that dtype when specified,
    # weak tensor of the inferred dtype otherwise). Also checks identity semantics: asarray of an
    # existing TF-NumPy array without dtype returns the very same object, even under the second
    # virtual CPU device, while giving a dtype returns a different object.
    for a, dtype in itertools.product(self.all_arrays, _all_types):
      # Dtype specified.
      self.match_expected_attrs(
          np_array_ops.asarray(a, dtype=dtype),
          np.asarray(a, dtype=dtype),
          _NP_TO_TF[dtype],
          tensor.Tensor,
      )
      # No dtype specified.
      actual = np_array_ops.asarray(a)
      expected = np.asarray(a)
      self.match_expected_attrs(
          actual,
          expected,
          _NP_to_TF_result_inferred_types[expected.dtype],
          WeakTensor,
      )

    zeros_list = np_array_ops.zeros(5)
    # Same instance is returned if no dtype is specified and input is ndarray.
    self.assertIs(np_array_ops.asarray(zeros_list), zeros_list)
    with ops.device('CPU:1'):
      self.assertIs(np_array_ops.asarray(zeros_list), zeros_list)
    # Different instance is returned if dtype is specified and input is ndarray.
    self.assertIsNot(np_array_ops.asarray(zeros_list, dtype=int), zeros_list)

  def testAsAnyArray(self):
    # [FUNCTION DOC]
    # Test: same scenario as the asarray test but for asanyarray: type/dtype expectations for given and
    # inferred dtypes, plus same-object return for an existing array without dtype (also under the
    # second virtual CPU) and a new object when a dtype is given.
    for a, dtype in itertools.product(self.all_arrays, _all_types):
      # Dtype specified.
      self.match_expected_attrs(
          np_array_ops.asanyarray(a, dtype=dtype),
          np.asanyarray(a, dtype=dtype),
          _NP_TO_TF[dtype],
          tensor.Tensor,
      )
      # No dtype specified.
      actual = np_array_ops.asanyarray(a)
      expected = np.asanyarray(a)
      self.match_expected_attrs(
          actual,
          expected,
          _NP_to_TF_result_inferred_types[expected.dtype],
          WeakTensor,
      )

    zeros_list = np_array_ops.zeros(5)
    # Same instance is returned if no dtype is specified and input is ndarray.
    self.assertIs(np_array_ops.asanyarray(zeros_list), zeros_list)
    with ops.device('CPU:1'):
      self.assertIs(np_array_ops.asanyarray(zeros_list), zeros_list)
    # Different instance is returned if dtype is specified and input is ndarray.
    self.assertIsNot(np_array_ops.asanyarray(zeros_list, dtype=int), zeros_list)

  def testAsContiguousArray(self):
    # [FUNCTION DOC]
    # Test: ascontiguousarray over every sample input and dtype; plain tensor of the dtype when
    # specified, weak tensor of the inferred dtype otherwise, matching numpy values and shapes.
    for a, dtype in itertools.product(self.all_arrays, _all_types):
      # Dtype specified.
      self.match_expected_attrs(
          np_array_ops.ascontiguousarray(a, dtype=dtype),
          np.ascontiguousarray(a, dtype=dtype),
          _NP_TO_TF[dtype],
          tensor.Tensor,
      )
      # No dtype specified.
      actual = np_array_ops.ascontiguousarray(a)
      expected = np.ascontiguousarray(a)
      self.match_expected_attrs(
          actual,
          expected,
          _NP_to_TF_result_inferred_types[expected.dtype],
          WeakTensor,
      )

  def testARange(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: in the zero-step case every statement, including the comparison and the explicit
    # dtype call, sits inside the expected-error block, so only the first arange call (which raises a
    # value error) executes and the remaining statements in that block are never reached; they are
    # dead code and verify nothing.
    # Test: arange with start only, start and stop, and start, stop and step, over integer and float
    # values (negative, zero, positive) and every dtype. With dtype the result is a plain tensor of that
    # dtype; without dtype it is a weak tensor of the inferred dtype. Float start/stop combined with an
    # explicit dtype are skipped because of known numpy/TF differences, and float steps are not tested.
    # A step of zero must raise a value error.
    int_values = np.arange(-3, 3).tolist()
    float_values = np.arange(-3.5, 3.5).tolist()
    all_values = int_values + float_values
    for dtype in _all_types:
      for start in all_values:
        msg = 'dtype:{} start:{}'.format(dtype, start)
        # Dtype specified.
        self.match_expected_attrs(
            np_array_ops.arange(start, dtype=dtype),
            np.arange(start, dtype=dtype),
            _NP_TO_TF[dtype],
            tensor.Tensor,
            msg=msg,
        )
        # No dtype specified.
        actual = np_array_ops.arange(start)
        expected = np.arange(start)
        self.match_expected_attrs(
            actual,
            expected,
            _NP_to_TF_result_inferred_types[expected.dtype],
            WeakTensor,
            msg=msg,
        )

        for stop in all_values:
          msg = 'dtype:{} start:{} stop:{}'.format(dtype, start, stop)
          # TODO(srbs): Investigate and remove check.
          # There are some bugs when start or stop is float and dtype is int.
          if not isinstance(start, float) and not isinstance(stop, float):
            # Dtype specfieid.
            self.match_expected_attrs(
                np_array_ops.arange(start, stop, dtype=dtype),
                np.arange(start, stop, dtype=dtype),
                _NP_TO_TF[dtype],
                tensor.Tensor,
                msg=msg,
            )
            # No dtype specified.
            actual = np_array_ops.arange(start, stop)
            expected = np.arange(start, stop)
            self.match_expected_attrs(
                actual,
                expected,
                _NP_to_TF_result_inferred_types[expected.dtype],
                WeakTensor,
                msg=msg,
            )
          # Note: We intentionally do not test with float values for step
          # because numpy.arange itself returns inconsistent results. e.g.
          # np.arange(0.5, 3, step=0.5, dtype=int) returns
          # array([0, 1, 2, 3, 4])
          for step in int_values:
            msg = 'dtype:{} start:{} stop:{} step:{}'.format(
                dtype, start, stop, step)
            if not step:
              with self.assertRaises(ValueError):
                actual = np_array_ops.arange(start, stop, step)
                expected = np.arange(start, stop, step)
                self.match_expected_attrs(
                    actual,
                    expected,
                    _NP_to_TF_result_inferred_types[expected.dtype],
                    WeakTensor,
                    msg=msg,
                )
                if not isinstance(start, float) and not isinstance(stop, float):
                  self.match_expected_attrs(
                      np_array_ops.arange(start, stop, step, dtype=dtype),
                      np.arange(start, stop, step, dtype=dtype),
                      _NP_TO_TF[dtype],
                      tensor.Tensor,
                      msg=msg,
                  )
            else:
              if not isinstance(start, float) and not isinstance(stop, float):
                actual = np_array_ops.arange(start, stop, step)
                expected = np.arange(start, stop, step)
                self.match_expected_attrs(
                    actual,
                    expected,
                    _NP_to_TF_result_inferred_types[expected.dtype],
                    WeakTensor,
                    msg=msg,
                )
                self.match_expected_attrs(
                    np_array_ops.arange(start, stop, step, dtype=dtype),
                    np.arange(start, stop, step, dtype=dtype),
                    _NP_TO_TF[dtype],
                    tensor.Tensor,
                    msg=msg,
                )

  def testDiag(self):
    # [FUNCTION DOC]
    # Test: diag on two-dimensional inputs (square, wide, tall, single row/column, empty) and
    # one-dimensional inputs, for the default diagonal and offsets from -3 to 2, under identity, weak
    # tensor and TF-NumPy transforms. Results must be weak tensors with numpy's values, shape and
    # the inferred dtype.
    array_transforms = [
        lambda x: x,  # Identity,
        _get_weak_tensor,
        np_array_ops.array,
    ]

    def run_test(arr):
      # [FUNCTION DOC]
      # HELPER: inner checker for the diag test.
      # KNOWN ISSUE: the loop reassigns its own input name with each transform result, so the transforms
      # are applied cumulatively (the second transform receives the output of the first) instead of each
      # being applied to the original input.
      # Compares diag (default and each offset) with numpy: weak tensor result, inferred dtype, shape and
      # values. Takes the array-like input; returns nothing.
      for fn in array_transforms:
        arr = fn(arr)
        actual = np_array_ops.diag(arr)
        expected = np.diag(arr)
        self.match_expected_attrs(
            actual,
            expected,
            _NP_to_TF_result_inferred_types[expected.dtype],
            WeakTensor,
            msg='diag({})'.format(arr),
        )
        for k in range(-3, 3):
          actual = np_array_ops.diag(arr, k)
          expected = np.diag(arr, k)
          self.match_expected_attrs(
              actual,
              expected,
              _NP_to_TF_result_inferred_types[expected.dtype],
              WeakTensor,
              msg='diag({}, k={})'.format(arr, k),
          )

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
    # Test: diagflat on empty, one-dimensional, two-dimensional and three-dimensional inputs with
    # the default and offsets from -3 to 2, under three input transforms. Results must be weak tensors
    # equal to numpy in values, shape and inferred dtype.
    array_transforms = [
        lambda x: x,  # Identity,
        _get_weak_tensor,
        np_array_ops.array,
    ]

    def run_test(arr):
      # [FUNCTION DOC]
      # HELPER: inner checker for the diagflat test.
      # KNOWN ISSUE: the loop reassigns its own input name with each transform result, so the transforms
      # are applied cumulatively rather than each to the original input.
      # Compares diagflat (default and each offset) against numpy for weak tensor result type, inferred
      # dtype, shape and values. Returns nothing.
      for fn in array_transforms:
        arr = fn(arr)
        actual = np_array_ops.diagflat(arr)
        expected = np.diagflat(arr)
        self.match_expected_attrs(
            actual,
            expected,
            _NP_to_TF_result_inferred_types[expected.dtype],
            WeakTensor,
            msg='diagflat({})'.format(arr),
        )
        for k in range(-3, 3):
          actual = np_array_ops.diagflat(arr, k)
          expected = np.diagflat(arr, k)
          self.match_expected_attrs(
              actual,
              expected,
              _NP_to_TF_result_inferred_types[expected.dtype],
              WeakTensor,
              msg='diagflat({})'.format(arr),
          )

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
    # HELPER: assertion helper shared by the tests in this class.
    # Asserts the shapes of actual and expected are equal. If a message is given it is expanded into a
    # failure message that mentions both shapes. Returns nothing; fails by assertion.
    if msg:
      msg = 'Shape match failed for: {}. Expected: {} Actual: {}'.format(
          msg, expected.shape, actual.shape)
    self.assertEqual(actual.shape, expected.shape, msg=msg)

  def match_dtype_and_type(self, actual, expected_dtype, res_type, msg=None):
    # [FUNCTION DOC]
    # HELPER: assertion helper shared by the tests in this class.
    # Asserts that actual is an instance of the expected result type (res_type) and that its dtype
    # equals expected_dtype. If a message is given, a detailed message listing expected and actual
    # dtype and type is built and used for the dtype assertion. Returns nothing.
    if msg:
      msg = (
          'Dtype and type match failed for: {}. Expected dtype: {} Actual'
          ' dtype: {}. Expected type: {} Actual type: {}.'.format(
              msg, expected_dtype, actual.dtype, res_type, type(actual)
          )
      )
    self.assertIsInstance(actual, res_type)
    self.assertEqual(actual.dtype, expected_dtype, msg=msg)

  def match_expected_attrs(
      self,
      actual,
      expected,
      expected_dtype,
      res_type,
      msg=None,
      almost=False,
      decimal=7,
  ):
    # [FUNCTION DOC]
    # HELPER: main assertion helper shared by the tests in this class.
    # KNOWN ISSUE: the result-type assertion is performed twice, once directly and once inside the
    # dtype-and-type helper; the first is redundant. Also the decimal argument only matters when the
    # almost flag is set.
    # Checks actual against expected (a numpy result): result type, dtype, shape, and values. Inputs:
    # expected_dtype and res_type (class the result must be an instance of), optional msg appended to a
    # base message showing expected and actual, almost (compare approximately), decimal (precision for
    # approximate comparison, default 7). With exact comparison scalar results are compared as lists
    # by equality and non-scalars as equal sequences. Returns nothing.
    msg_ = 'Expected: {} Actual: {}'.format(expected, actual)
    if msg:
      msg = '{} {}'.format(msg_, msg)
    else:
      msg = msg_
    self.assertIsInstance(actual, res_type)
    self.match_dtype_and_type(actual, expected_dtype, res_type, msg)
    self.match_shape(actual, expected, msg)
    if not almost:
      if not actual.shape.rank:
        self.assertEqual(actual.tolist(), expected.tolist())
      else:
        self.assertSequenceEqual(actual.tolist(), expected.tolist())
    else:
      np.testing.assert_almost_equal(
          actual.tolist(), expected.tolist(), decimal=decimal)


class ArrayMethodsTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests TF-NumPy array methods and reductions (copy, cumprod/cumsum, imag, real, amax/amin, mean,
  # std, var, prod, sum, size, ravel, repeat, around, reshape, expand_dims, squeeze, transpose,
  # shape, swapaxes, moveaxis, ndim) with weak tensor, array and python inputs, verifying
  # result type (weak tensor versus plain tensor when a dtype is passed), inferred dtype, shape and
  # values against numpy. Fixture: virtual CPU devices and a list of array transforms.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup. Calls the parent setup, configures two virtual CPU devices, and stores in
    # array_transforms three input transforms (identity, weak tensor conversion, TF-NumPy array
    # conversion). Returns nothing.
    super(ArrayMethodsTest, self).setUp()
    set_up_virtual_devices()
    self.array_transforms = [
        lambda x: x,
        _get_weak_tensor,
        np_array_ops.array,
    ]

  def testCopy(self):
    # [FUNCTION DOC]
    # Test: copy of empty, integer, float and 2-D inputs under each transform returns a weak tensor
    # matching numpy. Also checks device placement: a scalar array created by default is not backed by
    # the second CPU, while copying it (or a numpy scalar) with the copy flag under a second-CPU device
    # scope yields an array backed by that device.

    def run_test(arr, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner checker for the copy test. For each input transform, copies the input with the
      # extra arguments in TF-NumPy and numpy and checks weak tensor type, inferred dtype, shape and
      # values. Returns nothing.
      for fn in self.array_transforms:
        arg = fn(arr)
        actual = np_array_ops.copy(arg, *args, **kwargs)
        expected = np.copy(arg, *args, **kwargs)
        self.match_expected_attrs(
            actual,
            expected,
            _NP_to_TF_result_inferred_types[expected.dtype],
            WeakTensor,
            msg='copy({})'.format(arr),
        )

    run_test([])
    run_test([1, 2, 3])
    run_test([1., 2., 3.])
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
    # Test: cumprod and cumsum for integer and float lists with and without explicit dtype (python and
    # numpy dtypes) and with axes 0, 1, -1, -2 on 2-D data. Values must equal numpy; without dtype the
    # result is a weak tensor of the inferred dtype, with dtype a plain tensor of the dtype's promoted
    # type.

    def run_test(arr, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner checker for the cumulative product/sum test. For each input transform, runs both
      # operations with the given extra arguments, compares with numpy, and checks result type and
      # dtype depending on whether a dtype keyword was given. Returns nothing.
      for fn in self.array_transforms:
        arg = fn(arr)
        # Cumprod Test
        actual = np_array_ops.cumprod(arg, *args, **kwargs)
        expected = np.cumprod(arg, *args, **kwargs)
        self.assertAllEqual(actual, expected)
        if kwargs.get('dtype', None) is None:
          self.match_dtype_and_type(
              actual,
              _NP_to_TF_result_inferred_types[expected.dtype],
              WeakTensor,
          )
        else:
          self.match_dtype_and_type(
              actual,
              flexible_dtypes.result_type(kwargs['dtype'])[0],
              tensor.Tensor,
          )

        # Cumsum Test
        actual = np_array_ops.cumsum(arg, *args, **kwargs)
        expected = np.cumsum(arg, *args, **kwargs)
        self.assertAllEqual(actual, expected)
        if kwargs.get('dtype', None) is None:
          self.match_dtype_and_type(
              actual,
              _NP_to_TF_result_inferred_types[expected.dtype],
              WeakTensor,
          )
        else:
          self.match_dtype_and_type(
              actual,
              flexible_dtypes.result_type(kwargs['dtype'])[0],
              tensor.Tensor,
          )

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
    # Test: imag of integer, float, complex scalars and lists/nested lists, under each transform.
    # Result is always a weak tensor; expected dtype is given per case (a weak complex input yields
    # float64, weak ints stay int32, weak floats float32).

    def run_test(arr, dtype):
      # [FUNCTION DOC]
      # HELPER: inner checker for the imag test. Takes an input and the expected TF dtype; for each
      # transform compares imag with numpy (wrapped into an array since numpy may return a scalar),
      # checking weak tensor type, dtype, shape and values.
      for fn in self.array_transforms:
        arg = fn(arr)
        actual = np_array_ops.imag(arg)
        # np.imag may return a scalar so we convert to a np.ndarray.
        expected = np.array(np.imag(arg))
        self.match_expected_attrs(actual, expected, dtype, WeakTensor)

    # Weak complex128 input returns float64.
    run_test(1, dtypes.int32)
    run_test(5.5, dtypes.float32)
    run_test(5 + 3j, dtypes.float64)
    run_test(3j, dtypes.float64)
    run_test([], dtypes.float32)
    run_test([1, 2, 3], dtypes.int32)
    run_test([1 + 5j, 2 + 3j], dtypes.float64)
    run_test([[1 + 5j, 2 + 3j], [1 + 7j, 2 + 8j]], dtypes.float64)

  def testAMaxAMin(self):
    # [FUNCTION DOC]
    # Test: amax and amin for lists and nested lists with various axes (single, negative, tuples,
    # reversed tuples) and keepdims, with input and axis each under every transform; results must be
    # weak tensors matching numpy. Finally asserts that passing an output argument raises a value
    # error for both functions.

    def run_test(arr, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner checker for the amax/amin test. Extracts the axis keyword, then for every
      # combination of input transform and axis transform runs both reductions in TF-NumPy and numpy and
      # checks weak tensor type, inferred dtype, shape and values. Returns nothing.
      axis = kwargs.pop('axis', None)
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arr_arg = fn1(arr)
          axis_arg = fn2(axis) if axis is not None else None
          actual = np_array_ops.amax(arr_arg, axis=axis_arg, *args, **kwargs)
          expected = np.amax(arr_arg, axis=axis, *args, **kwargs)
          self.match_expected_attrs(
              actual,
              expected,
              _NP_to_TF_result_inferred_types[expected.dtype],
              WeakTensor,
              msg='amax({})'.format(arr),
          )
          actual = np_array_ops.amin(arr_arg, axis=axis_arg, *args, **kwargs)
          expected = np.amin(arr_arg, axis=axis, *args, **kwargs)
          self.match_expected_attrs(
              actual,
              expected,
              _NP_to_TF_result_inferred_types[expected.dtype],
              WeakTensor,
              msg='amin({})'.format(arr),
          )

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
    # Test: mean for integer/float lists, with and without dtype, and nested lists with several axes and
    # keepdims, with input and axis under every transform. Without dtype results are weak tensors with
    # the inferred dtype; with dtype a plain tensor of the promoted dtype. Passing an output argument
    # must raise a value error.

    def run_test(arr, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner checker for the mean test. For each combination of input and axis transforms,
      # computes mean in TF-NumPy and numpy and checks type, dtype (depending on whether a dtype keyword
      # was passed), shape and values. Returns nothing.
      axis = kwargs.pop('axis', None)
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arr_arg = fn1(arr)
          axis_arg = fn2(axis) if axis is not None else None
          actual = np_array_ops.mean(arr_arg, axis=axis_arg, *args, **kwargs)
          expected = np.mean(arr_arg, axis=axis, *args, **kwargs)
          dtype = kwargs.get('dtype', None)
          if dtype is None:
            self.match_expected_attrs(
                actual,
                expected,
                _NP_to_TF_result_inferred_types[expected.dtype],
                WeakTensor,
            )
          else:
            self.match_expected_attrs(
                actual,
                expected,
                flexible_dtypes.result_type(dtype)[0],
                tensor.Tensor,
            )
    run_test([1, 2, 1])
    run_test([1.0, 2.0, 1.0])
    run_test([1.0, 2.0, 1.0], dtype=int)
    run_test([[1, 2], [3, 4]], axis=1)
    run_test([[1, 2], [3, 4]], axis=0)
    run_test([[1, 2], [3, 4]], axis=-1)
    run_test([[1, 2], [3, 4]], axis=-2)
    run_test([[1, 2], [3, 4]], axis=(0, 1))
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(0, 2))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(),
        axis=(0, 2),
        keepdims=True,
    )
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(2, 0))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(),
        axis=(2, 0),
        keepdims=True,
    )
    self.assertRaises(ValueError, np_array_ops.mean, np.ones([2, 2]), out=[])

  def testStd(self):
    # [FUNCTION DOC]
    # Test: std for integer, float and complex inputs and nested lists with various axes and keepdims.
    # Results must be weak tensors; the expected dtype is inferred from numpy except for the complex
    # case, which must be float64.
    def run_test(arr, res_dtype, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner checker for the std test.
      # KNOWN ISSUE: the result dtype parameter is overwritten inside the loop, so after the first
      # iteration the dtype derived from numpy is reused for all later iterations.
      # Takes input, expected result dtype (nothing means infer from numpy), and extra arguments
      # including axis. For every combination of transforms compares std with numpy and checks weak
      # tensor type, dtype, shape and values.
      axis = kwargs.pop('axis', None)
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arr_arg = fn1(arr)
          axis_arg = fn2(axis) if axis is not None else None
          actual = np_array_ops.std(arr_arg, axis=axis_arg, *args, **kwargs)
          expected = np.std(arr_arg, axis=axis, *args, **kwargs)
          res_dtype = (
              _NP_to_TF_result_inferred_types[expected.dtype]
              if res_dtype is None
              else res_dtype
          )
          self.match_expected_attrs(actual, expected, res_dtype, WeakTensor)

    run_test([1, 2, 1], res_dtype=None)
    run_test([1.0, 2.0, 1.0], res_dtype=None)
    run_test([1.0j, 2.0, 1.0j], res_dtype=dtypes.float64)
    run_test([[1, 2], [3, 4]], res_dtype=None, axis=1)
    run_test([[1, 2], [3, 4]], res_dtype=None, axis=0)
    run_test([[1, 2], [3, 4]], res_dtype=None, axis=-1)
    run_test([[1, 2], [3, 4]], res_dtype=None, axis=-2)
    run_test([[1, 2], [3, 4]], res_dtype=None, axis=(0, 1))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), res_dtype=None, axis=(0, 2)
    )
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(),
        res_dtype=None,
        axis=(0, 2),
        keepdims=True,
    )
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), res_dtype=None, axis=(2, 0)
    )
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(),
        res_dtype=None,
        axis=(2, 0),
        keepdims=True,
    )

  def testVar(self):
    # [FUNCTION DOC]
    # Test: var for integer, float, complex inputs, an explicit int64 dtype case, and nested lists with
    # various axes and keepdims; results are weak tensors without dtype and plain tensors with a dtype.
    # Passing an output argument must raise a value error.

    def run_test(arr, res_dtype, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner checker for the var test.
      # KNOWN ISSUE: the result dtype parameter is overwritten inside the loop, so the first inferred
      # value is reused for subsequent iterations.
      # Takes input, expected result dtype (nothing means infer from numpy) and extra arguments. For all
      # transform combinations compares var with numpy and checks result type (plain tensor when a dtype
      # keyword was given, else weak), dtype, shape and values.
      axis = kwargs.pop('axis', None)
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arr_arg = fn1(arr)
          axis_arg = fn2(axis) if axis is not None else None
          actual = np_array_ops.var(arr_arg, axis=axis_arg, *args, **kwargs)
          expected = np.var(arr_arg, axis=axis, *args, **kwargs)
          dtype = kwargs.get('dtype', None)
          res_type = tensor.Tensor if dtype is not None else WeakTensor
          res_dtype = (
              _NP_to_TF_result_inferred_types[expected.dtype]
              if res_dtype is None
              else res_dtype
          )
          self.match_expected_attrs(actual, expected, res_dtype, res_type)

    # Input of weak complex type (complex 128) always outputs float64.
    run_test([1, 2, 1], res_dtype=None)
    run_test([1.0, 2.0, 1.0], res_dtype=None)
    run_test([1.0j, 2.0, 1.0j], res_dtype=dtypes.float64)
    run_test([1.0, 2.0, 1.0], res_dtype=dtypes.int64, dtype=np.int64)
    run_test([[1, 2], [3, 4]], res_dtype=None, axis=1)
    run_test([[1, 2], [3, 4]], res_dtype=None, axis=0)
    run_test([[1, 2], [3, 4]], res_dtype=None, axis=-1)
    run_test([[1, 2], [3, 4]], res_dtype=None, axis=-2)
    run_test([[1, 2], [3, 4]], res_dtype=None, axis=(0, 1))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), res_dtype=None, axis=(0, 2)
    )
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(),
        axis=(0, 2),
        res_dtype=None,
        keepdims=True,
    )
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(), res_dtype=None, axis=(2, 0)
    )
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(),
        axis=(2, 0),
        res_dtype=None,
        keepdims=True,
    )
    self.assertRaises(ValueError, np_array_ops.var, np.ones([2, 2]), out=[])

  def testProd(self):
    # [FUNCTION DOC]
    # Test: prod for integer and float lists, a numpy array with dtype, and nested lists with axes and
    # keepdims. Without dtype results are weak tensors with the inferred dtype; with dtype they are
    # plain tensors of the promoted dtype; values and shape equal numpy's.
    def run_test(arr, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner checker for the prod test. For each input transform, computes prod with the extra
      # arguments in TF-NumPy and numpy and checks result type, dtype, shape and values depending on
      # whether a dtype keyword was supplied.
      for fn in self.array_transforms:
        arg = fn(arr)
        actual = np_array_ops.prod(arg, *args, **kwargs)
        expected = np.prod(arg, *args, **kwargs)
        dtype = kwargs.get('dtype', None)
        if dtype is None:
          self.match_expected_attrs(
              actual,
              expected,
              _NP_to_TF_result_inferred_types[expected.dtype],
              WeakTensor,
          )
        else:
          self.match_expected_attrs(
              actual,
              expected,
              flexible_dtypes.result_type(dtype)[0],
              tensor.Tensor,
          )

    run_test([1, 2, 3])
    run_test([1.0, 2.0, 3.0])
    run_test(np.array([1, 2, 3]), dtype=np.int32)
    run_test([[1, 2], [3, 4]], axis=1)
    run_test([[1, 2], [3, 4]], axis=0)
    run_test([[1, 2], [3, 4]], axis=-1)
    run_test([[1, 2], [3, 4]], axis=-2)
    run_test([[1, 2], [3, 4]], axis=(0, 1))
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(0, 2))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(),
        axis=(0, 2),
        keepdims=True,
    )
    run_test(np.arange(8).reshape((2, 2, 2)).tolist(), axis=(2, 0))
    run_test(
        np.arange(8).reshape((2, 2, 2)).tolist(),
        axis=(2, 0),
        keepdims=True,
    )

  def _testReduce(self, math_fun, np_fun, name):
    # [FUNCTION DOC]
    # HELPER: shared check used by the sum and amax tests.
    # Given a TF-NumPy reduction (math_fun), the numpy reference (np_fun) and a name for messages,
    # runs it on scalar, list and nested-list inputs with axis values (none, 0, 1, tuples) and keepdims,
    # with input and axis each under every transform, asserting weak tensor results with inferred dtype
    # and numpy-equal shape and values.
    axis_transforms = [
        lambda x: x,  # Identity,
        np_array_ops.array,
        _get_weak_tensor,
    ]

    def run_test(a, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner checker for the reduction helper. Takes the input and keyword arguments (axis is
      # extracted, others such as keepdims are forwarded), and for each combination of input and axis
      # transforms compares the TF-NumPy and numpy reductions.
      axis = kwargs.pop('axis', None)
      for fn1 in self.array_transforms:
        for fn2 in axis_transforms:
          arg1 = fn1(a)
          axis_arg = fn2(axis) if axis is not None else None
          actual = math_fun(arg1, axis=axis_arg, **kwargs)
          expected = np_fun(arg1, axis=axis, **kwargs)
          self.match_expected_attrs(
              actual,
              expected,
              _NP_to_TF_result_inferred_types[expected.dtype],
              WeakTensor,
              msg='{}({}, axis={}, keepdims={})'.format(
                  name, arg1, axis, kwargs.get('keepdims')
              ),
          )

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
    # Test: sum, checked through the shared reduction helper against numpy sum.
    self._testReduce(np_array_ops.sum, np.sum, 'sum')

  def testAmax(self):
    # [FUNCTION DOC]
    # Test: amax, checked through the shared reduction helper against numpy amax.
    self._testReduce(np_array_ops.amax, np.amax, 'amax')

  def testSize(self):
    # [FUNCTION DOC]
    # Test: size of various TF-NumPy arrays, python scalars and constants equals numpy's size (with
    # optional axis). Asking for the size along an axis of a plain numpy array must raise a
    # not-implemented error. Finally a compiled function with an unknown-shape float input signature
    # converts its argument to an array and returns its size; calling it with a 3 by 2 array must
    # give 6.

    def run_test(arr, axis=None):
      # [FUNCTION DOC]
      # HELPER: inner checker for the size test. Takes an array-like and an optional axis and asserts
      # that TF-NumPy size equals numpy size of the same data.
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

    @def_function.function(
        input_signature=[
            tensor_spec.TensorSpec(dtype=dtypes.float32, shape=None)
        ]
    )
    def f(arr):
      # [FUNCTION DOC]
      # Compiled function used by the size test (input signature: float32 tensor of unknown shape).
      # Converts the argument with asarray and returns its size. Returns the size value.
      arr = np_array_ops.asarray(arr)
      return np_array_ops.size(arr)

    self.assertEqual(f(np_array_ops.ones((3, 2))).numpy(), 6)

  def testRavel(self):
    # [FUNCTION DOC]
    # Test: ravel of scalars, empty and nested lists, and multi-dimensional inputs under each
    # transform returns a weak tensor matching numpy in values, shape and inferred dtype.

    def run_test(arr, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner checker for the ravel test. For each input transform, flattens with extra
      # arguments in TF-NumPy and numpy and compares weak tensor type, dtype, shape and values.
      for fn in self.array_transforms:
        arg = fn(arr)
        actual = np_array_ops.ravel(arg, *args, **kwargs)
        expected = np.ravel(arg, *args, **kwargs)
        self.match_expected_attrs(
            actual,
            expected,
            _NP_to_TF_result_inferred_types[expected.dtype],
            WeakTensor,
        )

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
    # Test: real of scalars, complex scalars, and lists under each transform; always a weak tensor.
    # Complex weak inputs must give float64; other cases use the dtype inferred from numpy.

    def run_test(arr, res_dtype, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner checker for the real test.
      # KNOWN ISSUE: the expected dtype parameter is overwritten in the loop, so the value inferred on
      # the first iteration is reused later.
      # Takes input, expected dtype (nothing means infer from numpy) and extra arguments; for each
      # transform compares real with numpy for weak tensor type, dtype, shape and values.
      for fn in self.array_transforms:
        arg = fn(arr)
        actual = np_array_ops.real(arg, *args, **kwargs)
        expected = np.array(np.real(arg, *args, **kwargs))
        res_dtype = (
            _NP_to_TF_result_inferred_types[expected.dtype]
            if res_dtype is None
            else res_dtype
        )
        self.match_expected_attrs(
            actual,
            expected,
            res_dtype,
            WeakTensor,
        )

    run_test(1, None)
    run_test(5.5, None)
    run_test(5 + 3j, dtypes.float64)
    run_test(3j, dtypes.float64)
    run_test([], None)
    run_test([1, 2, 3], None)
    run_test([1 + 5j, 2 + 3j], dtypes.float64)
    run_test([[1 + 5j, 2 + 3j], [1 + 7j, 2 + 8j]], dtypes.float64)

  def testRepeat(self):
    # [FUNCTION DOC]
    # Test: repeat with scalar and list repeat counts and with axes 0, 1, -1, -2, with both the array
    # and the repeats under every transform; results must be weak tensors matching numpy.

    def run_test(arr, repeats, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner checker for the repeat test. For each combination of transforms applied to the
      # array and the repeat counts, compares repeat in TF-NumPy and numpy for weak tensor type, dtype,
      # shape and values.
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arr_arg = fn1(arr)
          repeats_arg = fn2(repeats)
          actual = np_array_ops.repeat(arr_arg, repeats_arg, *args, **kwargs)
          expected = np.repeat(arr_arg, repeats_arg, *args, **kwargs)
          self.match_expected_attrs(
              actual,
              expected,
              _NP_to_TF_result_inferred_types[expected.dtype],
              WeakTensor,
          )

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
    # Test: around (rounding) for floats, empty list, lists with positive and negative decimals under
    # each transform; weak tensor results matching numpy.

    def run_test(arr, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner checker for the around test. For each input transform, rounds with the extra
      # arguments in TF-NumPy and numpy and compares type, dtype, shape and values.
      for fn in self.array_transforms:
        arg = fn(arr)
        actual = np_array_ops.around(arg, *args, **kwargs)
        expected = np.around(arg, *args, **kwargs)
        self.match_expected_attrs(
            actual,
            expected,
            _NP_to_TF_result_inferred_types[expected.dtype],
            WeakTensor,
        )

    run_test(5.5)
    run_test(5.567, decimals=2)
    run_test([])
    run_test([1.27, 2.49, 2.75], decimals=1)
    run_test([23.6, 45.1], decimals=-1)

  def testReshape(self):
    # [FUNCTION DOC]
    # Test: reshape of scalars, empty list and short lists to various new shapes (including -1) with
    # array and shape under every transform; weak tensor results matching numpy.

    def run_test(arr, newshape, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner checker for the reshape test. For each combination of transforms on the array and
      # the new shape, compares reshape in TF-NumPy and numpy for type, dtype, shape and values.
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arr_arg = fn1(arr)
          newshape_arg = fn2(newshape)
          actual = np_array_ops.reshape(arr_arg, newshape_arg, *args, **kwargs)
          expected = np.reshape(arr_arg, newshape, *args, **kwargs)
          self.match_expected_attrs(
              actual,
              expected,
              _NP_to_TF_result_inferred_types[expected.dtype],
              WeakTensor,
          )

    run_test(5, [-1])
    run_test([], [-1])
    run_test([1, 2, 3], [1, 3])
    run_test([1, 2, 3], [3, 1])
    run_test([1, 2, 3, 4], [2, 2])
    run_test([1, 2, 3, 4], [2, 1, 2])

  def testExpandDims(self):
    # [FUNCTION DOC]
    # Test: expand_dims of a list at axes 0 and 1 gives a weak tensor matching numpy.

    def run_test(arr, axis):
      # [FUNCTION DOC]
      # HELPER: inner checker for the expand_dims test. Takes a list and an axis, compares expand_dims
      # with numpy for weak tensor type, inferred dtype, shape and values (no input transforms).
      actual = np_array_ops.expand_dims(arr, axis)
      expected = np.expand_dims(arr, axis)
      self.match_expected_attrs(
          actual,
          expected,
          _NP_to_TF_result_inferred_types[expected.dtype],
          WeakTensor,
      )

    run_test([1, 2, 3], 0)
    run_test([1, 2, 3], 1)

  def testSqueeze(self):
    # [FUNCTION DOC]
    # Test: squeeze of scalars, empty, single-element and nested lists with various axes (single,
    # negative, tuple); weak tensor results matching numpy.

    def run_test(arr, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner checker for the squeeze test. For each input transform, compares squeeze in
      # TF-NumPy with numpy squeeze applied to an explicit numpy array copy of the input, because numpy
      # ignores the axis for non-array inputs (a numpy bug noted in the code).
      for fn in self.array_transforms:
        arg = fn(arr)
        # Note: np.squeeze ignores the axis arg for non-ndarray objects.
        # This looks like a bug: https://github.com/numpy/numpy/issues/8201
        # So we convert the arg to np.ndarray before passing to np.squeeze.
        actual = np_array_ops.squeeze(arg, *args, **kwargs)
        expected = np.squeeze(np.array(arg), *args, **kwargs)
        self.match_expected_attrs(
            actual,
            expected,
            _NP_to_TF_result_inferred_types[expected.dtype],
            WeakTensor,
        )

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
    # Test: transpose of scalars, vectors and a 2x3x5 array with default and all six axis permutations,
    # with array and axes under every transform; weak tensor results matching numpy.

    def run_test(arr, axes=None):
      # [FUNCTION DOC]
      # HELPER: inner checker for the transpose test. Takes the array and optional axes (defaults to
      # none); for each combination of transforms compares transpose in TF-NumPy and numpy.
      for fn1 in self.array_transforms:
        for fn2 in self.array_transforms:
          arr_arg = fn1(arr)
          axes_arg = fn2(axes) if axes is not None else None
          actual = np_array_ops.transpose(arr_arg, axes_arg)
          expected = np.transpose(arr_arg, axes)
          self.match_expected_attrs(
              actual,
              expected,
              _NP_to_TF_result_inferred_types[expected.dtype],
              WeakTensor,
          )

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
    # HELPER: assertion helper shared by the tests in this class. Asserts the shapes of actual and
    # expected are equal, with a descriptive failure message when a message is supplied.
    if msg:
      msg = 'Shape match failed for: {}. Expected: {} Actual: {}'.format(
          msg, expected.shape, actual.shape)
    self.assertEqual(actual.shape, expected.shape, msg=msg)

  def match_dtype_and_type(self, actual, expected_dtype, res_type, msg=None):
    # [FUNCTION DOC]
    # HELPER: assertion helper shared by the tests in this class. Asserts that actual is an instance of
    # res_type and that its dtype equals expected_dtype, using a descriptive message (when msg is
    # supplied) listing expected and actual dtype and type.
    if msg:
      msg = (
          'Dtype and type match failed for: {}. Expected dtype: {} Actual'
          ' dtype: {}. Expected type: {} Actual type: {}.'.format(
              msg, expected_dtype, actual.dtype, res_type, type(actual)
          )
      )
    self.assertIsInstance(actual, res_type, msg=msg)
    self.assertEqual(actual.dtype, expected_dtype, msg=msg)

  def match_expected_attrs(
      self, actual, expected, expected_dtype, res_type, msg=None
  ):
    # [FUNCTION DOC]
    # HELPER: main assertion helper shared by the tests in this class.
    # KNOWN ISSUE: both branches of the scalar-versus-nonscalar check perform the identical
    # approximate comparison, so the conditional has no effect.
    # Builds a failure message from expected and actual (plus optional msg), checks dtype and result
    # type via the dtype-and-type helper, the shape via the shape helper, and finally compares values
    # with approximate equality. Returns nothing.
    msg_ = 'Expected: {} Actual: {}'.format(expected, actual)
    if msg:
      msg = '{} {}'.format(msg_, msg)
    else:
      msg = msg_
    self.match_dtype_and_type(actual, expected_dtype, res_type, msg)
    self.match_shape(actual, expected, msg)
    if not actual.shape.rank:
      self.assertAllClose(actual.tolist(), expected.tolist())
    else:
      self.assertAllClose(actual.tolist(), expected.tolist())

  def testShape(self):
    # [FUNCTION DOC]
    # Test: shape of a nested list with one row of two elements equals the pair (1, 2).
    self.assertAllEqual((1, 2), np_array_ops.shape([[0, 0]]))

  @parameterized.parameters(
      ([[1, 2, 3]], 0, 1, [[1], [2], [3]]),
      ([[1, 2, 3]], -2, -1, [[1], [2], [3]]),
      (
          [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
          0,
          2,
          [[[0, 4], [2, 6]], [[1, 5], [3, 7]]],
      ),
      (
          [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
          -3,
          -1,
          [[[0, 4], [2, 6]], [[1, 5], [3, 7]]],
      ),
  )
  def testSwapaxes(self, x, axis1, axis2, expected):
    # [FUNCTION DOC]
    # Parameterized test: swapaxes on nested lists with positive and negative axis pairs must return a
    # weak tensor whose values equal the given expected nested list. Inputs: x, axis1, axis2 and
    # expected values.
    actual = np_array_ops.swapaxes(x, axis1, axis2)
    self.assertIsInstance(actual, WeakTensor)
    self.assertAllEqual(actual, expected)

  def testMoveaxis(self):
    # [FUNCTION DOC]
    # Test: moveaxis on a random six-dimensional array for tuples and single integers of source and
    # destination axes (positive, negative, full permutations, reversed, empty), each result must be a
    # weak tensor equal to numpy's moveaxis. Uses random data (values are not checked for specific
    # numbers).

    def _test(a, *args):
      # [FUNCTION DOC]
      # HELPER: inner checker for the moveaxis test. Takes an array and the source and destination
      # arguments; converts the array to a weak tensor, calls moveaxis, and asserts a weak tensor result
      # equal to numpy's answer.
      # pylint: disable=no-value-for-parameter
      expected = np.moveaxis(a, *args)
      wt_a = _get_weak_tensor(a)
      raw_ans = np_array_ops.moveaxis(wt_a, *args)

      self.assertIsInstance(raw_ans, WeakTensor)
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
    # Test: ndim of a float scalar is 0 and of a two-element list is 1.
    self.assertAllEqual(0, np_array_ops.ndim(0.5))
    self.assertAllEqual(1, np_array_ops.ndim([1, 2]))


if __name__ == '__main__':
  ops.enable_eager_execution()
  ops.set_dtype_conversion_mode('all')
  np_math_ops.enable_numpy_methods_on_tensor()
  test.main()
