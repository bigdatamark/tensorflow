# [MODULE DOC]
# Tests for the TF NumPy array type (an alias of the tensor class) and the tensor methods and
# operator overloads it relies on: dtype access, type conversion, arithmetic and comparison
# operators across mixed dtypes, Python scalar conversion, hashing and composite-tensor flatten/pack.
# Importing np_math_ops is required for the operator overloads. Run as a script it enables eager
# execution, turns on equality semantics for tensors, selects the legacy dtype conversion mode,
# enables NumPy methods on tensors, and starts the test runner.
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
"""Tests for ndarray."""

import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.numpy_ops import np_arrays
# Required for operator overloads
from tensorflow.python.ops.numpy_ops import np_math_ops  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.util import nest


class ArrayTest(test.TestCase):
  # [CLASS DOC]
  # Test case for array behavior: dtype, astype, operators with promotion, comparisons, scalar
  # conversions, hashing and composite flattening. Holds a class-level table of expected dtype triples
  # for true division. Helper methods run one operator over many dtype combinations.

  def testDtype(self):
    # [FUNCTION DOC]
    # Creates an int64 zeros tensor of shape one by two and asserts its dtype maps to the NumPy int64
    # type and that converting zero via that NumPy type compares equal to zero.
    a = array_ops.zeros(shape=[1, 2], dtype=dtypes.int64)
    self.assertIs(a.dtype.as_numpy_dtype, np.int64)
    np_dt = a.dtype.as_numpy_dtype
    self.assertAllEqual(0, np_dt(0))

  def testAstype(self):
    # [FUNCTION DOC]
    # Converts a float32 scalar to int32 with the astype method and asserts the dtype and value 1; then
    # converts a float32 vector to boolean and asserts boolean dtype with values false and true.
    a = ops.convert_to_tensor(value=1.1, dtype=dtypes.float32).astype(np.int32)
    self.assertIs(a.dtype.as_numpy_dtype, np.int32)
    self.assertAllEqual(1, a)
    a = ops.convert_to_tensor(value=[0.0, 1.1], dtype=dtypes.float32).astype(
        np.bool_)
    self.assertIs(a.dtype.as_numpy_dtype, np.bool_)
    self.assertAllEqual([False, True], a)

  def testNeg(self):
    # [FUNCTION DOC]
    # Asserts that negating a float vector gives the element-wise negatives.
    a = ops.convert_to_tensor(value=[1.0, 2.0])
    self.assertAllEqual([-1.0, -2.0], -a)  # pylint: disable=invalid-unary-operand-type

  def _testBinOp(self, a, b, out, f, types=None):
    # [FUNCTION DOC]
    # HELPER: shared driver for the binary operator tests.
    # Inputs: a, b operand values; out expected result (value or tensor); f a callable applying the
    # operator; types optional list of (a type, b type, expected output type) triples (defaults to a
    # table covering int32, int64, float32 and float64 promotion). Converts operands to int32 tensors,
    # then for each triple casts operands, applies f, asserts the output NumPy dtype is the expected one,
    # casts the expected output and compares (close-equality for floating types, exact otherwise).
    # Raises a test failure on mismatch.
    a = ops.convert_to_tensor(value=a, dtype=np.int32)
    b = ops.convert_to_tensor(value=b, dtype=np.int32)
    if not isinstance(out, np_arrays.ndarray):
      out = ops.convert_to_tensor(value=out, dtype=np.int32)
    if types is None:
      types = [[np.int32, np.int32, np.int32], [np.int64, np.int32, np.int64],
               [np.int32, np.int64, np.int64],
               [np.float32, np.int32, np.float64],
               [np.int32, np.float32, np.float64],
               [np.float32, np.float32, np.float32],
               [np.float64, np.float32, np.float64],
               [np.float32, np.float64, np.float64]]
    for a_type, b_type, out_type in types:
      o = f(a.astype(a_type), b.astype(b_type))
      self.assertIs(o.dtype.as_numpy_dtype, out_type)
      out = out.astype(out_type)
      if np.issubdtype(out_type, np.inexact):
        self.assertAllClose(out, o)
      else:
        self.assertAllEqual(out, o)

  def testAdd(self):
    # [FUNCTION DOC]
    # Tests addition via the add method on integer vectors across dtype combinations; expects 4 and 6.
    self._testBinOp([1, 2], [3, 4], [4, 6], lambda a, b: a.__add__(b))

  def testRadd(self):
    # [FUNCTION DOC]
    # Tests reflected addition via the radd method; expects 4 and 6.
    self._testBinOp([1, 2], [3, 4], [4, 6], lambda a, b: b.__radd__(a))

  def testSub(self):
    # [FUNCTION DOC]
    # Tests subtraction via the sub method; expects -2 and -3.
    self._testBinOp([1, 2], [3, 5], [-2, -3], lambda a, b: a.__sub__(b))

  def testRsub(self):
    # [FUNCTION DOC]
    # Tests reflected subtraction via the rsub method; expects -2 and -3.
    self._testBinOp([1, 2], [3, 5], [-2, -3], lambda a, b: b.__rsub__(a))

  def testMul(self):
    # [FUNCTION DOC]
    # Tests multiplication via the mul method; expects 3 and 8.
    self._testBinOp([1, 2], [3, 4], [3, 8], lambda a, b: a.__mul__(b))

  def testRmul(self):
    # [FUNCTION DOC]
    # Tests reflected multiplication via the rmul method; expects 3 and 8.
    self._testBinOp([1, 2], [3, 4], [3, 8], lambda a, b: b.__rmul__(a))

  def testPow(self):
    # [FUNCTION DOC]
    # Tests power via the pow method with bases 4, 5 and exponents 3, 2; expects 64 and 25.
    self._testBinOp([4, 5], [3, 2], [64, 25], lambda a, b: a.__pow__(b))

  def testRpow(self):
    # [FUNCTION DOC]
    # Tests reflected power via the rpow method; expects 64 and 25.
    self._testBinOp([4, 5], [3, 2], [64, 25], lambda a, b: b.__rpow__(a))

  _truediv_types = [[np.int32, np.int32, np.float64],
                    [np.int64, np.int32, np.float64],
                    [np.int32, np.int64, np.float64],
                    [np.float32, np.int32, np.float64],
                    [np.int32, np.float32, np.float64],
                    [np.float32, np.float32, np.float32],
                    [np.float64, np.float32, np.float64],
                    [np.float32, np.float64, np.float64]]

  def testTruediv(self):
    # [FUNCTION DOC]
    # Tests true division via the truediv method on 3,5 by 2,4 with the dedicated promotion table
    # (integers divide to float64); expects 1.5 and 1.25.
    self._testBinOp([3, 5], [2, 4],
                    ops.convert_to_tensor(value=[1.5, 1.25]),
                    lambda a, b: a.__truediv__(b),
                    types=self._truediv_types)

  def testRtruediv(self):
    # [FUNCTION DOC]
    # Tests reflected true division via the rtruediv method with the same table; expects 1.5 and 1.25.
    self._testBinOp([3, 5], [2, 4],
                    ops.convert_to_tensor(value=[1.5, 1.25]),
                    lambda a, b: b.__rtruediv__(a),
                    types=self._truediv_types)

  def _testCmp(self, a, b, out, f):
    # [FUNCTION DOC]
    # HELPER: shared driver for the comparison operator tests.
    # Inputs: a, b operand values; out expected boolean result; f a callable applying the comparison.
    # Converts operands to int32 tensors and, for a fixed list of dtype pairs, casts both operands,
    # applies f and asserts the result equals out. Raises a test failure on mismatch.
    a = ops.convert_to_tensor(value=a, dtype=np.int32)
    b = ops.convert_to_tensor(value=b, dtype=np.int32)

    types = [[np.int32, np.int32], [np.int64, np.int32], [np.int32, np.int64],
             [np.float32, np.int32], [np.int32, np.float32],
             [np.float32, np.float32], [np.float64, np.float32],
             [np.float32, np.float64]]
    for a_type, b_type in types:
      o = f(a.astype(a_type), b.astype(b_type))
      self.assertAllEqual(out, o)

  def testLt(self):
    # [FUNCTION DOC]
    # Tests the less-than method on 1,2,3 versus 3,2,1; expects true, false, false.
    self._testCmp([1, 2, 3], [3, 2, 1], [True, False, False],
                  lambda a, b: a.__lt__(b))

  def testLe(self):
    # [FUNCTION DOC]
    # Tests the less-or-equal method; expects true, true, false.
    self._testCmp([1, 2, 3], [3, 2, 1], [True, True, False],
                  lambda a, b: a.__le__(b))

  def testGt(self):
    # [FUNCTION DOC]
    # Tests the greater-than method; expects false, false, true.
    self._testCmp([1, 2, 3], [3, 2, 1], [False, False, True],
                  lambda a, b: a.__gt__(b))

  def testGe(self):
    # [FUNCTION DOC]
    # Tests the greater-or-equal method; expects false, true, true.
    self._testCmp([1, 2, 3], [3, 2, 1], [False, True, True],
                  lambda a, b: a.__ge__(b))

  def testEq(self):
    # [FUNCTION DOC]
    # Tests the equality method; expects false, true, false.
    self._testCmp([1, 2, 3], [3, 2, 1], [False, True, False],
                  lambda a, b: a.__eq__(b))

  def testNe(self):
    # [FUNCTION DOC]
    # Tests the not-equal method; expects true, false, true.
    self._testCmp([1, 2, 3], [3, 2, 1], [True, False, True],
                  lambda a, b: a.__ne__(b))

  def testInt(self):
    # [FUNCTION DOC]
    # Converts a scalar tensor to a Python integer and asserts type and value 10.
    v = 10
    u = int(ops.convert_to_tensor(value=v))
    self.assertIsInstance(u, int)
    self.assertAllEqual(v, u)

  def testFloat(self):
    # [FUNCTION DOC]
    # Converts a scalar tensor to a Python float and asserts type and value 21.32 within tolerance.
    v = 21.32
    u = float(ops.convert_to_tensor(value=v))
    self.assertIsInstance(u, float)
    self.assertAllClose(v, u)

  def testBool(self):
    # [FUNCTION DOC]
    # Converts scalar tensors to booleans: 10 and 0.1 are true, 0 and 0.0 are false; asserts the type.
    b = bool(ops.convert_to_tensor(value=10))
    self.assertIsInstance(b, bool)
    self.assertTrue(b)
    self.assertFalse(bool(ops.convert_to_tensor(value=0)))
    self.assertTrue(bool(ops.convert_to_tensor(value=0.1)))
    self.assertFalse(bool(ops.convert_to_tensor(value=0.0)))

  def testHash(self):
    # [FUNCTION DOC]
    # Asserts hashing a tensor raises a type error saying tensors are unhashable and to use the ref
    # method, both eagerly and inside a traced function.
    a = ops.convert_to_tensor(value=10)
    def eager():
      # [FUNCTION DOC]
      # HELPER: eager case of testHash; hashes the enclosing tensor (expected to raise).
      hash(a)
    def graph():
      # [FUNCTION DOC]
      # HELPER: graph case of testHash; calls a traced function that hashes the enclosing tensor.
      @def_function.function
      def f(x):
        # [FUNCTION DOC]
        # HELPER: traced function of testHash's graph case. Input x is a tensor; hashes it (raises a type error).
        hash(x)
      f(a)
    for f in [eager, graph]:
      with self.assertRaisesRegex(
          TypeError,
          r'Tensor is unhashable. Instead, use tensor.ref\(\) as the key.'):
        f()

  def testFromToCompositeTensor(self):
    # [FUNCTION DOC]
    # Flattens a list of two tensors with composite expansion and asserts two plain tensors result; packs
    # them back to the original structure and asserts two arrays that are close to the originals.
    tensors = [ops.convert_to_tensor(0.1), ops.convert_to_tensor(0.2)]

    flattened = nest.flatten(tensors, expand_composites=True)
    # Each ndarray contains only one tensor, so the flattened output should be
    # just 2 tensors in a list.
    self.assertLen(flattened, 2)
    self.assertIsInstance(flattened[0], tensor.Tensor)
    self.assertIsInstance(flattened[1], tensor.Tensor)

    repacked = nest.pack_sequence_as(tensors, flattened, expand_composites=True)
    self.assertLen(repacked, 2)
    self.assertIsInstance(repacked[0], np_arrays.ndarray)
    self.assertIsInstance(repacked[1], np_arrays.ndarray)

    self.assertAllClose(tensors, repacked)


if __name__ == '__main__':
  # TODO(wangpeng): Test in graph mode as well. Also test in V2 (the requirement
  # for setting _USE_EQUALITY points to V2 behavior not being on).
  ops.enable_eager_execution()
  tensor.Tensor._USE_EQUALITY = True
  ops.set_dtype_conversion_mode('legacy')
  np_math_ops.enable_numpy_methods_on_tensor()
  test.main()
