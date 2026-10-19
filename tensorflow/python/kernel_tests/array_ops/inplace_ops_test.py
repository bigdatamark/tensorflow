# [MODULE DOC]
# Test module for the in-place update, add and subtract operations, the alias-in-place add, and
# the empty / empty-like tensor constructors. Compares results against numpy arrays mutated the
# same way. Defines a module list of basic dtypes and one test class; no import-time side
# effects beyond imports.
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
"""Tests for inplace_ops."""
import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import inplace_ops
from tensorflow.python.platform import test as test_lib


BASIC_TYPES = [
    dtypes.float32,
    dtypes.int8,
    dtypes.uint8,
    dtypes.int32,
    dtypes.int64,
    dtypes.uint64,
    dtypes.bfloat16,
]


class InplaceOpsTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the in-place row update/add/subtract operations and empty-tensor creation,
  # using numpy arrays as the reference. Tests run on GPU when available. No shared fixtures.

  def testBasicUpdate(self):
    # [FUNCTION DOC]
    # Test scenario: for each basic dtype, start with a seven-by-three tensor of ones and a
    # matching numpy array; replace row three, then the last row (index minus one) with twos, then
    # row five given as a scalar index with sevens, comparing with the numpy mirror after each
    # step. Returns nothing.
    for dtype in BASIC_TYPES:
      with test_util.use_gpu():
        x = array_ops.ones([7, 3], dtype)
        y = np.ones([7, 3], dtype.as_numpy_dtype)
        self.assertAllClose(x, y)
        x = inplace_ops.inplace_update(x, [3], array_ops.ones([1, 3], dtype))
        y[3, :] = 1
        self.assertAllClose(x, y)
        x = inplace_ops.inplace_update(x, [-1],
                                       array_ops.ones([1, 3], dtype) * 2)
        y[-1, :] = 2
        self.assertAllClose(x, y)
        x = inplace_ops.inplace_update(x, 5, array_ops.ones([3], dtype) * 7)
        y[5, :] = 7
        self.assertAllClose(x, y)

  def testBasicUpdateBool(self):
    # [FUNCTION DOC]
    # Test scenario: same as the basic update test but with boolean tensors: set row three to
    # true, the last row to false and row five (scalar index) to false, checking against numpy
    # after each update. Returns nothing.
    with test_util.use_gpu():
      x = array_ops.ones([7, 3], dtypes.bool)
      y = np.ones([7, 3], dtypes.bool.as_numpy_dtype)
      self.assertAllClose(x, y)
      x = inplace_ops.inplace_update(x, [3], array_ops.ones([1, 3],
                                                            dtypes.bool))
      y[3, :] = True
      self.assertAllClose(x, y)
      x = inplace_ops.inplace_update(x, [-1],
                                     array_ops.zeros([1, 3], dtypes.bool))
      y[-1, :] = False
      self.assertAllClose(x, y)
      x = inplace_ops.inplace_update(x, 5, array_ops.zeros([3], dtypes.bool))
      y[5, :] = False
      self.assertAllClose(x, y)

  def testBasicAdd(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the first add goes through the array_ops namespace while every other call in
    # the file uses the inplace_ops module, an inconsistency that may hide the module under test.
    # Test scenario: for each basic dtype, add ones to row three, twos to the last row, sevens to
    # row five (scalar index), and 99 to every row (index given as nothing), comparing with the
    # numpy array updated with the same additions after each step. Returns nothing.
    for dtype in BASIC_TYPES:
      with test_util.use_gpu():
        x = array_ops.ones([7, 3], dtype)
        y = np.ones([7, 3], dtype.as_numpy_dtype)
        self.assertAllClose(x, y)
        x = array_ops.inplace_add(x, [3], array_ops.ones([1, 3], dtype))
        y[3, :] += 1
        self.assertAllClose(x, y)
        x = inplace_ops.inplace_add(x, [-1], array_ops.ones([1, 3], dtype) * 2)
        y[-1, :] += 2
        self.assertAllClose(x, y)
        x = inplace_ops.inplace_add(x, 5, array_ops.ones([3], dtype) * 7)
        y[5, :] += 7
        self.assertAllClose(x, y)
        x = inplace_ops.inplace_add(x, None, array_ops.ones([7, 3], dtype) * 99)
        y[:, :] += 99
        self.assertAllClose(x, y)

  def testBasicSub(self):
    # [FUNCTION DOC]
    # Test scenario: mirror of the add test using subtraction: subtract 1 from row three, 2 from
    # the last row, 7 from row five (scalar index) and 99 from all rows (index nothing), for each
    # basic dtype, comparing with numpy after each step. Returns nothing.
    for dtype in BASIC_TYPES:
      with test_util.use_gpu():
        x = array_ops.ones([7, 3], dtype)
        y = np.ones([7, 3], dtype.as_numpy_dtype)
        self.assertAllClose(x, y)
        x = inplace_ops.inplace_sub(x, [3], array_ops.ones([1, 3], dtype))
        y[3, :] -= 1
        self.assertAllClose(x, y)
        x = inplace_ops.inplace_sub(x, [-1], array_ops.ones([1, 3], dtype) * 2)
        y[-1, :] -= 2
        self.assertAllClose(x, y)
        x = inplace_ops.inplace_sub(x, 5, array_ops.ones([3], dtype) * 7)
        y[5, :] -= 7
        self.assertAllClose(x, y)
        x = inplace_ops.inplace_sub(x, None, array_ops.ones([7, 3], dtype) * 99)
        y[:, :] -= 99
        self.assertAllClose(x, y)

  def testRandom(self):
    # [FUNCTION DOC]
    # Test scenario (randomized, uses numpy random state): start with a zero tensor of shape
    # 100 by 3 by 5. Twenty times pick ten distinct random row indices and random integer
    # values, randomly choose update, add or subtract, apply to both the tensor and a numpy
    # mirror, and compare. Returns nothing.
    with test_util.use_gpu():
      d0, d1, d2 = 100, 3, 5
      x = array_ops.zeros([d0, d1, d2])
      y = np.zeros([d0, d1, d2])
      for _ in range(20):
        idx = np.random.choice(d0, d0 // 10, replace=False)
        val = np.random.randint(10, size=(d0 // 10, d1, d2))
        op = np.random.randint(3)
        if op == 0:
          x = inplace_ops.inplace_update(x, idx, val)
          y[idx, :] = val
        elif op == 1:
          x = inplace_ops.inplace_add(x, idx, val)
          y[idx, :] += val
        elif op == 2:
          x = inplace_ops.inplace_sub(x, idx, val)
          y[idx, :] -= val
        self.assertAllClose(x, y)

  def testRandom1D(self):
    # [FUNCTION DOC]
    # Test scenario: randomized like the multi-dimensional test but on a one-dimensional zero
    # tensor of length 100, so each index selects a single element. Twenty rounds of random
    # update, add or subtract compared against a numpy mirror. Returns nothing.
    with test_util.use_gpu():
      d0 = 100
      x = array_ops.zeros([d0])
      y = np.zeros([d0])
      for _ in range(20):
        idx = np.random.choice(d0, d0 // 10, replace=False)
        val = np.random.randint(10, size=(d0 // 10))
        op = np.random.randint(3)
        if op == 0:
          x = inplace_ops.inplace_update(x, idx, val)
          y[idx] = val
        elif op == 1:
          x = inplace_ops.inplace_add(x, idx, val)
          y[idx] += val
        elif op == 2:
          x = inplace_ops.inplace_sub(x, idx, val)
          y[idx] -= val
        self.assertAllClose(x, y)

  def testAlias(self):
    # [FUNCTION DOC]
    # Test scenario: a two-by-three tensor of ones gets row zero incremented by one, two, three
    # through the aliasing in-place add (which mutates its input buffer). An identity of the
    # original tensor is taken under a control dependency on the add result, then the original,
    # add result and identity are evaluated; expects the add result to equal the identity,
    # showing the original buffer was modified. Returns nothing.
    with test_util.use_gpu():
      x = array_ops.ones([2, 3])
      y = inplace_ops.alias_inplace_add(x, [0], [[1, 2, 3]])
      with ops.control_dependencies([y]):
        z = array_ops.identity(x)
        _, vy, vz = self.evaluate([x, y, z])
      self.assertAllClose(vy, vz)

  def testError(self):
    # [FUNCTION DOC]
    # Test scenario: three invalid in-place updates must each raise an invalid-argument error:
    # an index that is not a vector ("must be a vector"), a value whose shape does not match
    # the target ("x and v shape doesn't match"), and an index list longer than the target's
    # first dimension ("i and x shape doesn't match"). Returns nothing.
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "must be a vector"):
      _ = self.evaluate(inplace_ops.inplace_update([[1.]], [[0]], [[10]]))
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "x and v shape doesn't match"):
      _ = self.evaluate(inplace_ops.inplace_update([[1.]], [0], [10]))
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "i and x shape doesn't match"):
      _ = self.evaluate(inplace_ops.inplace_update([[1.]], [0, 1], [[10]]))

  def testEmpty(self):
    # [FUNCTION DOC]
    # Test scenario: for several numeric/bool dtypes and shapes (scalar, vector, matrices, and
    # shapes with zero-size dimensions), the empty constructor and the empty-like constructor
    # must return arrays with the requested shape and dtype; with initialization enabled the
    # contents must be zeros. Also, a one-by-two string tensor created with or without
    # initialization must contain empty byte strings. Returns nothing.
    for dtype in [
        dtypes.float32, dtypes.float64, dtypes.int32, dtypes.int64, dtypes.bool,
        dtypes.uint8, dtypes.bfloat16
    ]:
      with test_util.use_gpu():
        test_shapes = [(), (1,), (2, 3), (0, 2), (2, 3, 5), (2, 0, 5)]
        for shape in test_shapes:
          val = self.evaluate(inplace_ops.empty(shape, dtype))
          self.assertEqual(val.shape, shape)
          self.assertEqual(val.dtype, dtype.as_numpy_dtype)
          val = self.evaluate(inplace_ops.empty(shape, dtype, init=True))
          self.assertEqual(val.shape, shape)
          self.assertEqual(val.dtype, dtype.as_numpy_dtype)
          self.assertAllEqual(val, np.zeros(shape, dtype.as_numpy_dtype))
          val = self.evaluate(
              inplace_ops.empty_like(array_ops.zeros(shape, dtype)))
          self.assertEqual(val.shape, shape)
          self.assertEqual(val.dtype, dtype.as_numpy_dtype)
          val = self.evaluate(inplace_ops.empty_like(
              array_ops.zeros(shape, dtype), init=True))
          self.assertEqual(val.shape, shape)
          self.assertEqual(val.dtype, dtype.as_numpy_dtype)
          self.assertAllEqual(val, np.zeros(shape, dtype.as_numpy_dtype))

    with test_util.use_gpu():
      val = self.evaluate(inplace_ops.empty((1, 2), dtypes.string, init=True))
      self.assertEqual(val.tolist(), [[b"", b""]])

      val = self.evaluate(inplace_ops.empty((1, 2), dtypes.string, init=False))
      self.assertEqual(val.tolist(), [[b"", b""]])

  def testInplaceOpOnEmptyTensors(self):
    # [FUNCTION DOC]
    # Test scenario: for every basic dtype and each of add, subtract and update, apply the op to
    # a seven-by-zero tensor with a one-by-zero update, once with index three and once with index
    # nothing. Expects no failure and the result to still equal the empty numpy array.
    # Returns nothing.
    op_fns = [
        inplace_ops.inplace_add,
        inplace_ops.inplace_sub,
        inplace_ops.inplace_update,
    ]
    for dtype in BASIC_TYPES:
      for op_fn in op_fns:
        with test_util.use_gpu():
          x = array_ops.zeros([7, 0], dtype)
          y = np.zeros([7, 0], dtype.as_numpy_dtype)
          self.assertAllClose(x, y)
          x = op_fn(x, [3], array_ops.ones([1, 0], dtype))
          self.assertAllClose(x, y)
          x = op_fn(x, None, array_ops.ones([1, 0], dtype))
          self.assertAllClose(x, y)


if __name__ == "__main__":
  test_lib.main()
