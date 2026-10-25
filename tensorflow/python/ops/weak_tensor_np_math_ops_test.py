# [MODULE DOC]
# Unit tests for TF-NumPy mathematical functions operating on weak tensors, verifying result
# types (weak tensor vs array), dtype inference, shapes and values against numpy. At import it
# imports the weak tensor ops module for its patching side effect. Run as a script it enables
# tensor equality, numpy methods on tensors, eager execution and the "all" dtype conversion mode
# before starting the test runner.
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
"""Tests for tf numpy mathematical methods on WeakTensor."""

import itertools

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import flexible_dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.framework import weak_tensor
from tensorflow.python.ops import weak_tensor_ops  # pylint: disable=unused-import
from tensorflow.python.ops import weak_tensor_test_util
from tensorflow.python.ops.numpy_ops import np_array_ops
from tensorflow.python.ops.numpy_ops import np_arrays
from tensorflow.python.ops.numpy_ops import np_math_ops
from tensorflow.python.ops.numpy_ops import np_random
from tensorflow.python.platform import googletest


_get_weak_tensor = weak_tensor_test_util.get_weak_tensor

_NP_to_TF_result_inferred_types = {
    np.dtype(np.int32): dtypes.int32,
    np.dtype(np.int64): dtypes.int32,
    np.dtype(np.float32): dtypes.float32,
    np.dtype(np.float64): dtypes.float32,
    np.dtype(np.complex128): dtypes.complex128,
}


class MathTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests TF-NumPy math functions (log, exp, tanh, sqrt, argsort, argmax/argmin, average, ptp,
  # flatten, isinf, random dtype) with weak tensor, plain array and python inputs, under the
  # new dtype conversion mode. Fixture: a list of input transforms and numpy scalar types.
  # Shared helpers: a unary-op checker and a match comparison method.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup. Calls the parent setup, then creates a list of three input transforms
    # (identity, conversion to a weak tensor, conversion to a TF-NumPy array) stored in the
    # array_transforms attribute, and a list of four numpy scalar types (int32, int64, float32,
    # float64) stored in the types attribute. Returns nothing.
    super(MathTest, self).setUp()
    self.array_transforms = [
        lambda x: x,  # Identity,
        _get_weak_tensor,
        np_array_ops.array,
    ]
    self.types = [np.int32, np.int64, np.float32, np.float64]

  def _testUnaryOp(self, math_fun, np_fun, name, weak_result):
    # [FUNCTION DOC]
    # HELPER: shared check used by the unary math tests in this class.
    # Runs math_fun over a scalar, a 1-D list and a 2-D nested list, each passed through every
    # input transform. If weak_result is true and the input is inferred as weak, asserts the result
    # is a weak tensor; otherwise asserts it is a TF-NumPy array. Then uses the match method to
    # compare dtype, shape and values against the numpy reference function np_fun (name is only
    # used in failure messages). Returns nothing; fails through assertions.
    def run_test(a):
      # [FUNCTION DOC]
      # HELPER: inner loop of the unary test helper. For one input value a, applies every input
      # transform, checks the result type of the op under test (weak tensor vs array depending on
      # the weak_result flag and the inferred weakness), and compares with the numpy reference.
      for fn in self.array_transforms:
        arg1 = fn(a)
        if weak_result and flexible_dtypes.result_type(arg1)[1]:
          self.assertIsInstance(math_fun(arg1), weak_tensor.WeakTensor)
        else:
          self.assertIsInstance(math_fun(arg1), np_arrays.ndarray)
        self.match(
            math_fun(arg1), np_fun(arg1), msg='{}({})'.format(name, arg1))

    run_test(5)
    run_test([2, 3])
    run_test([[2, -3], [-6, 7]])

  def testLog(self):
    # [FUNCTION DOC]
    # Test: runs the unary-op helper for the TF-NumPy log function against numpy log, expecting a
    # weak tensor result for weak inputs and matching dtype, shape and values for all transforms.
    self._testUnaryOp(np_math_ops.log, np.log, 'log', True)

  def testExp(self):
    # [FUNCTION DOC]
    # Test: runs the unary-op helper for the TF-NumPy exp function against numpy exp, expecting a
    # weak tensor result for weak inputs and matching dtype, shape and values for all transforms.
    self._testUnaryOp(np_math_ops.exp, np.exp, 'exp', True)

  def testTanh(self):
    # [FUNCTION DOC]
    # Test: runs the unary-op helper for the TF-NumPy tanh function against numpy tanh, expecting a
    # weak tensor result for weak inputs and matching dtype, shape and values for all transforms.
    self._testUnaryOp(np_math_ops.tanh, np.tanh, 'tanh', True)

  def testSqrt(self):
    # [FUNCTION DOC]
    # Test: runs the unary-op helper for the TF-NumPy sqrt function against numpy sqrt, expecting a
    # weak tensor result for weak inputs and matching dtype, shape and values for all transforms.
    self._testUnaryOp(np_math_ops.sqrt, np.sqrt, 'sqrt', True)

  def match(self, actual, expected, msg='', check_dtype=True):
    # [FUNCTION DOC]
    # HELPER: shared assertion helper for the tests in this class.
    # Compares a TF result (actual) with a numpy result (expected). When check_dtype is true
    # (default) the actual dtype must equal the TF type that the module's mapping assigns to the
    # numpy dtype (for example 64-bit numpy types map to 32-bit TF types under the new promotion
    # rules); the lookup raises a key error for numpy dtypes not in the mapping. Always asserts equal
    # shapes and then values within a relative tolerance of one millionth. msg is appended to
    # failure messages. Returns nothing.
    if check_dtype:
      self.assertEqual(
          actual.dtype,
          _NP_to_TF_result_inferred_types[expected.dtype],
          'Dtype mismatch.\nActual: {}\nExpected: {}\n{}'.format(
              actual.dtype.as_numpy_dtype,
              _NP_to_TF_result_inferred_types[expected.dtype],
              msg,
          ),
      )
    self.assertEqual(
        actual.shape, expected.shape,
        'Shape mismatch.\nActual: {}\nExpected: {}\n{}'.format(
            actual.shape, expected.shape, msg))
    np.testing.assert_allclose(actual.tolist(), expected.tolist(), rtol=1e-6)

  def testArgsort(self):
    # [FUNCTION DOC]
    # Test: argsort via the unary helper (result is a plain array, not weak, since it returns
    # indices), then checks stability: argsort of 100 equal zeros with the stable kind must equal
    # the sequence 0..99.
    self._testUnaryOp(np_math_ops.argsort, np.argsort, 'argsort', False)

    # Test stability
    r = np.arange(100)
    a = np.zeros(100)
    np.testing.assert_equal(np_math_ops.argsort(a, kind='stable'), r)

  def testArgMaxArgMin(self):
    # [FUNCTION DOC]
    # Test: for several scalar and nested-list inputs under every input transform, compares
    # argmax and argmin with numpy, both with no axis and for every valid axis from negative
    # rank to rank minus one (scalars are treated as 1-D of size one). Dtype is not checked
    # because indices are 64-bit.
    data = [
        0,
        5,
        [1],
        [1, 2, 3],
        [[1, 2, 3]],
        [[4, 6], [7, 8]],
        [[[4, 6], [9, 10]], [[7, 8], [12, 34]]],
    ]
    for fn, d in itertools.product(self.array_transforms, data):
      arr = fn(d)
      # argmax and argmin returns indices (int64 type).
      self.match(np_math_ops.argmax(arr), np.argmax(arr), check_dtype=False)
      self.match(np_math_ops.argmin(arr), np.argmin(arr), check_dtype=False)
      if hasattr(arr, 'shape'):
        ndims = len(arr.shape)
      else:
        ndims = np_array_ops.array(arr, copy=False).ndim
      if ndims == 0:
        # Numpy flattens the scalar ndarray and treats it as a 1-d array of
        # size 1.
        ndims = 1
      for axis in range(-ndims, ndims):
        self.match(
            np_math_ops.argmax(arr, axis=axis),
            np.argmax(arr, axis=axis),
            check_dtype=False,
        )
        self.match(
            np_math_ops.argmin(arr, axis=axis),
            np.argmin(arr, axis=axis),
            check_dtype=False,
        )

  def testAverageWrongShape(self):
    # [FUNCTION DOC]
    # Test: average with weights whose shape is incompatible with the data (wrong length, wrong
    # axis length, scalar weights with an axis, mismatched sizes) must raise an invalid-argument
    # error in each of four cases.
    with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError, r''):
      np_math_ops.average(np.ones([2, 3]), weights=np.ones([2, 4]))
    with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError, r''):
      np_math_ops.average(np.ones([2, 3]), axis=0, weights=np.ones([2, 4]))
    with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError, r''):
      np_math_ops.average(np.ones([2, 3]), axis=0, weights=np.ones([]))
    with self.assertRaisesWithPredicateMatch(errors.InvalidArgumentError, r''):
      np_math_ops.average(np.ones([2, 3]), axis=0, weights=np.ones([5]))

  def testPtp(self):
    # [FUNCTION DOC]
    # Test: peak-to-peak (max minus min) of integer and float lists and 2-D data with various
    # axes (0, 1, -1, -2), under every input transform, compared with numpy without dtype check.

    def run_test(arr, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: inner loop of the peak-to-peak test. For the given array and extra arguments,
      # applies each input transform and compares the TF-NumPy result with numpy (dtype ignored).
      for fn in self.array_transforms:
        arg = fn(arr)
        self.match(
            np_math_ops.ptp(arg, *args, **kwargs),
            np.ptp(arg, *args, **kwargs),
            check_dtype=False,
        )

    run_test([1, 2, 3])
    run_test([1., 2., 3.])
    run_test([[1, 2], [3, 4]], axis=1)
    run_test([[1, 2], [3, 4]], axis=0)
    run_test([[1, 2], [3, 4]], axis=-1)
    run_test([[1, 2], [3, 4]], axis=-2)

  # Test that enable_numpy_methods() gets called when weak_tensor_ops is
  # imported.
  @parameterized.parameters([
      'T', 'ndim', 'size', 'data', '__pos__', '__round__', 'tolist', 'flatten',
      'transpose', 'reshape', 'ravel', 'clip', 'astype', 'max', 'mean', 'min'])
  def testNumpyMethodsOnTensor(self, np_method):
    # [FUNCTION DOC]
    # Parameterized test over numpy-style attribute names (transpose, reshape, mean, size, and
    # others). Converts a small list to a tensor and asserts the tensor has that attribute,
    # verifying numpy methods were enabled when the weak tensor ops module was imported.
    a = ops.convert_to_tensor([1, 2])
    self.assertTrue(hasattr(a, np_method))

  def testFlatten(self):
    # [FUNCTION DOC]
    # Test: flatten on a weak tensor built from a 2x2x2 array returns the same values as numpy
    # for orders C and F; orders A and K are compared against the C-order numpy result (valid
    # for a C-contiguous input); an invalid order string must raise a value error.
    a1 = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    a2 = _get_weak_tensor(a1)
    self.assertAllEqual(a1.flatten('C'), a2.flatten('C'))
    self.assertAllEqual(a1.flatten('F'), a2.flatten('F'))
    self.assertAllEqual(a1.flatten('C'), a2.flatten('A'))
    self.assertAllEqual(a1.flatten('C'), a2.flatten('K'))
    with self.assertRaises(ValueError):
      a2.flatten('invalid')

  def testIsInf(self):
    # [FUNCTION DOC]
    # Test: the extreme 32-bit integer values as weak tensors are reported as not infinite, not
    # positive infinity and not negative infinity.
    x1 = _get_weak_tensor(-2147483648)
    x2 = _get_weak_tensor(2147483647)
    self.assertFalse(np_math_ops.isinf(x1))
    self.assertFalse(np_math_ops.isinf(x2))
    self.assertFalse(np_math_ops.isposinf(x1))
    self.assertFalse(np_math_ops.isposinf(x2))
    self.assertFalse(np_math_ops.isneginf(x1))
    self.assertFalse(np_math_ops.isneginf(x2))

  def testRandomOpsReturnFloat32(self):
    # [FUNCTION DOC]
    # Test: TF-NumPy random functions (rand, standard normal, uniform with bounds) return float32
    # tensors whose shapes equal those of the numpy equivalents. Uses random data; only dtype and
    # shape are asserted.
    x = np_random.rand(2, 50)
    np_x = np.random.rand(2, 50)
    self.assertEqual(x.dtype, dtypes.float32)
    self.assertEqual(np_x.shape, x.shape)

    x = np_random.standard_normal(50)
    np_x = np.random.standard_normal(50)
    self.assertEqual(x.dtype, dtypes.float32)
    self.assertEqual(np_x.shape, x.shape)

    x = np_random.uniform(low=-1, high=0, size=(50, 50))
    np_x = np.random.uniform(low=-1, high=0, size=(50, 50))
    self.assertEqual(x.dtype, dtypes.float32)
    self.assertEqual(np_x.shape, x.shape)


if __name__ == '__main__':
  tensor.enable_tensor_equality()
  np_math_ops.enable_numpy_methods_on_tensor()
  ops.enable_eager_execution()
  ops.set_dtype_conversion_mode('all')
  googletest.main()
