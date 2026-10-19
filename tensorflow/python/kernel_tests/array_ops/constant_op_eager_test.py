# [MODULE DOC]
# Eager-execution tests for constant creation and the basic filling ops: constant, zeros,
# zeros_like, ones, ones_like, fill and implicit conversion of Python/numpy values to tensors.
# Tests run with the eager context (the default), read results with the numpy accessor and compare
# against numpy references. Covers dtype inference, shape handling (explicit, implicit, too big or
# too small), strings with embedded nulls, custom sequence objects and invalid-length objects.
# Has no import-time side effects; the bottom runs the test main entry point when executed directly.
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
"""Tests for ConstantOp."""

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.util import compat


# TODO(josh11b): add tests with lists/tuples, Shape.
# TODO(ashankar): Collapse with tests in constant_op_test.py and use something
# like the test_util.run_in_graph_and_eager_modes decorator to confirm
# equivalence between graph and eager execution.
class ConstantTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the constant op in eager mode: tensors built from numpy arrays, Python lists,
  # shapes and dimensions, strings and odd sequence objects. Shared helpers compare conversion
  # results on the CPU and (when present) a GPU against numpy.

  def _testCpu(self, x):
    # [FUNCTION DOC]
    # HELPER: shared check used by the per-dtype constant tests.
    # Converts the given value to a tensor under the CPU device scope and compares its numpy value with
    # the numpy array of the same value. Floating and complex dtypes are compared approximately, all
    # others exactly. Input x is any array-like value. Returns nothing; fails the test on mismatch.
    np_ans = np.array(x)
    with context.device("/device:CPU:0"):
      tf_ans = ops.convert_to_tensor(x).numpy()
    if np_ans.dtype in [np.float32, np.float64, np.complex64, np.complex128]:
      self.assertAllClose(np_ans, tf_ans)
    else:
      self.assertAllEqual(np_ans, tf_ans)

  def _testGpu(self, x):
    # [FUNCTION DOC]
    # HELPER: GPU counterpart of the CPU check used by the per-dtype constant tests.
    # KNOWN ISSUE: when no GPU device is available the name lookup yields nothing and the body is
    # silently skipped, so nothing is verified on such machines.
    # Otherwise it converts input x under the GPU device scope and compares with numpy, approximately for
    # floating and complex dtypes and exactly for the rest. Returns nothing.
    device = test_util.gpu_device_name()
    if device:
      np_ans = np.array(x)
      with context.device(device):
        tf_ans = ops.convert_to_tensor(x).numpy()
      if np_ans.dtype in [np.float32, np.float64, np.complex64, np.complex128]:
        self.assertAllClose(np_ans, tf_ans)
      else:
        self.assertAllEqual(np_ans, tf_ans)

  def _testAll(self, x):
    # [FUNCTION DOC]
    # HELPER: runs both the CPU check and the GPU check for one input value x.
    # Returns nothing; failures come from the two helpers.
    self._testCpu(x)
    self._testGpu(x)

  def testFloat(self):
    # [FUNCTION DOC]
    # Tests float32 constants: arrays of ascending, random and empty (zero-sized dimension) values
    # round-trip. Also checks that Python lists of floats, mixed floats and ints, and ints too large for
    # 32 bits mixed with a float all infer the 32-bit float dtype and keep their values.
    self._testAll(np.arange(-15, 15).reshape([2, 3, 5]).astype(np.float32))
    self._testAll(
        np.random.normal(size=30).reshape([2, 3, 5]).astype(np.float32))
    self._testAll(np.empty((2, 0, 5)).astype(np.float32))

    orig = [-1.0, 2.0, 0.0]
    tf_ans = constant_op.constant(orig)
    self.assertEqual(dtypes_lib.float32, tf_ans.dtype)
    self.assertAllClose(np.array(orig), tf_ans.numpy())

    # Mix floats and ints
    orig = [-1.5, 2, 0]
    tf_ans = constant_op.constant(orig)
    self.assertEqual(dtypes_lib.float32, tf_ans.dtype)
    self.assertAllClose(np.array(orig), tf_ans.numpy())

    orig = [-5, 2.5, 0]
    tf_ans = constant_op.constant(orig)
    self.assertEqual(dtypes_lib.float32, tf_ans.dtype)
    self.assertAllClose(np.array(orig), tf_ans.numpy())

    # Mix floats and ints that don't fit in int32
    orig = [1, 2**42, 0.5]
    tf_ans = constant_op.constant(orig)
    self.assertEqual(dtypes_lib.float32, tf_ans.dtype)
    self.assertAllClose(np.array(orig), tf_ans.numpy())

  def testDouble(self):
    # [FUNCTION DOC]
    # Tests float64 constants round-trip for ascending, random and empty arrays and for an explicit
    # double dtype with mixed ints and floats. Checks that an integer not exactly representable as a
    # double is rounded, and that an integer beyond the double range raises a value error mentioning an
    # out-of-range integer.
    self._testAll(np.arange(-15, 15).reshape([2, 3, 5]).astype(np.float64))
    self._testAll(
        np.random.normal(size=30).reshape([2, 3, 5]).astype(np.float64))
    self._testAll(np.empty((2, 0, 5)).astype(np.float64))

    orig = [-5, 2.5, 0]
    tf_ans = constant_op.constant(orig, dtypes_lib.float64)
    self.assertEqual(dtypes_lib.float64, tf_ans.dtype)
    self.assertAllClose(np.array(orig), tf_ans.numpy())

    # This integer is not exactly representable as a double, gets rounded.
    tf_ans = constant_op.constant(2**54 + 1, dtypes_lib.float64)
    self.assertEqual(2**54, tf_ans.numpy())

    # This integer is larger than all non-infinite numbers representable
    # by a double, raises an exception.
    with self.assertRaisesRegex(ValueError, "out-of-range integer"):
      constant_op.constant(10**310, dtypes_lib.float64)

  def testInt32(self):
    # [FUNCTION DOC]
    # Tests int32 constants: ascending, random, empty arrays and a small Python list all round-trip
    # through conversion to tensors on CPU and GPU.
    self._testAll(np.arange(-15, 15).reshape([2, 3, 5]).astype(np.int32))
    self._testAll(
        (100 * np.random.normal(size=30)).reshape([2, 3, 5]).astype(np.int32))
    self._testAll(np.empty((2, 0, 5)).astype(np.int32))
    self._testAll([-1, 2])

  def testInt64(self):
    # [FUNCTION DOC]
    # Tests int64 constants round-trip. Checks that a Python list with values outside the 32-bit range
    # infers a 64-bit integer dtype with correct values, and that a list holding a value beyond 64 bits
    # raises a value error mentioning an out-of-range integer.
    self._testAll(np.arange(-15, 15).reshape([2, 3, 5]).astype(np.int64))
    self._testAll(
        (100 * np.random.normal(size=30)).reshape([2, 3, 5]).astype(np.int64))
    self._testAll(np.empty((2, 0, 5)).astype(np.int64))
    # Should detect out of range for int32 and use int64 instead.
    orig = [2, 2**48, -2**48]
    tf_ans = constant_op.constant(orig)
    self.assertEqual(dtypes_lib.int64, tf_ans.dtype)
    self.assertAllClose(np.array(orig), tf_ans.numpy())

    # Out of range for an int64
    with self.assertRaisesRegex(ValueError, "out-of-range integer"):
      constant_op.constant([2**72])

  def testComplex64(self):
    # [FUNCTION DOC]
    # Tests complex64 constants round-trip for ascending, random and empty arrays scaled by a complex
    # factor, on CPU and GPU.
    self._testAll(
        (1 + 2j) * np.arange(-15, 15).reshape([2, 3, 5]).astype(np.complex64))
    self._testAll(
        (1 + 2j) *
        np.random.normal(size=30).reshape([2, 3, 5]).astype(np.complex64))
    self._testAll(np.empty((2, 0, 5)).astype(np.complex64))

  def testComplex128(self):
    # [FUNCTION DOC]
    # Tests complex128 constants round-trip for ascending, random and empty arrays scaled by a complex
    # factor, on CPU and GPU.
    self._testAll(
        (1 + 2j) * np.arange(-15, 15).reshape([2, 3, 5]).astype(np.complex128))
    self._testAll(
        (1 + 2j) *
        np.random.normal(size=30).reshape([2, 3, 5]).astype(np.complex128))
    self._testAll(np.empty((2, 0, 5)).astype(np.complex128))

  @test_util.disable_tfrt("support creating string tensors from empty "
                          "numpy arrays.")
  def testString(self):
    # [FUNCTION DOC]
    # Tests string constants: a numpy array of byte strings and an empty string array convert to
    # tensors and match numpy, checked on CPU only. Disabled under the TFRT runtime because empty numpy
    # string arrays are not supported there.
    val = [compat.as_bytes(str(x)) for x in np.arange(-15, 15)]
    self._testCpu(np.array(val).reshape([2, 3, 5]))
    self._testCpu(np.empty((2, 0, 5)).astype(np.str_))

  def testStringWithNulls(self):
    # [FUNCTION DOC]
    # Tests that byte strings containing null bytes survive conversion: lengths and contents of simple
    # and nested values are preserved. The nested case compares through plain list conversion because
    # array comparison would lose the null terminators.
    val = ops.convert_to_tensor(b"\0\0\0\0").numpy()
    self.assertEqual(len(val), 4)
    self.assertEqual(val, b"\0\0\0\0")

    val = ops.convert_to_tensor(b"xx\0xx").numpy()
    self.assertEqual(len(val), 5)
    self.assertAllEqual(val, b"xx\0xx")

    nested = [[b"\0\0\0\0", b"xx\0xx"], [b"\0_\0_\0_\0", b"\0"]]
    val = ops.convert_to_tensor(nested).numpy()
    # NOTE(mrry): Do not use assertAllEqual, because it converts nested to a
    #   numpy array, which loses the null terminators.
    self.assertEqual(val.tolist(), nested)

  def testStringConstantOp(self):
    # [FUNCTION DOC]
    # Tests string constants made from a single text value and from a list of values, both without and
    # inside an explicit CPU device scope; decoded or compared values must equal the originals.
    s = constant_op.constant("uiuc")
    self.assertEqual(s.numpy().decode("utf-8"), "uiuc")
    s_array = constant_op.constant(["mit", "stanford"])
    self.assertAllEqual(s_array.numpy(), ["mit", "stanford"])

    with ops.device("/cpu:0"):
      s = constant_op.constant("cmu")
      self.assertEqual(s.numpy().decode("utf-8"), "cmu")

      s_array = constant_op.constant(["berkeley", "ucla"])
      self.assertAllEqual(s_array.numpy(), ["berkeley", "ucla"])

  def testExplicitShapeNumPy(self):
    # [FUNCTION DOC]
    # Tests that a numpy float array with an explicit shape equal to its own shape keeps that static
    # shape.
    c = constant_op.constant(
        np.arange(-15, 15).reshape([2, 3, 5]).astype(np.float32),
        shape=[2, 3, 5])
    self.assertEqual(c.get_shape(), [2, 3, 5])

  def testImplicitShapeNumPy(self):
    # [FUNCTION DOC]
    # Tests that a numpy float array without a shape argument yields a constant whose static shape is
    # the array shape.
    c = constant_op.constant(
        np.arange(-15, 15).reshape([2, 3, 5]).astype(np.float32))
    self.assertEqual(c.get_shape(), [2, 3, 5])

  def testExplicitShapeList(self):
    # [FUNCTION DOC]
    # Tests that a flat Python list of seven numbers with an explicit shape of seven yields a constant of
    # that shape.
    c = constant_op.constant([1, 2, 3, 4, 5, 6, 7], shape=[7])
    self.assertEqual(c.get_shape(), [7])

  def testExplicitShapeFill(self):
    # [FUNCTION DOC]
    # Tests that a scalar value with an explicit vector shape is broadcast: the constant has the
    # requested shape and every element equals the scalar.
    c = constant_op.constant(12, shape=[7])
    self.assertEqual(c.get_shape(), [7])
    self.assertAllEqual([12, 12, 12, 12, 12, 12, 12], c.numpy())

  def testExplicitShapeReshape(self):
    # [FUNCTION DOC]
    # Tests that a numpy array can be reshaped through the shape argument: a 2 by 3 by 5 array with the
    # shape 5 by 2 by 3 gives that static shape.
    c = constant_op.constant(
        np.arange(-15, 15).reshape([2, 3, 5]).astype(np.float32),
        shape=[5, 2, 3])
    self.assertEqual(c.get_shape(), [5, 2, 3])

  def testImplicitShapeList(self):
    # [FUNCTION DOC]
    # Tests that a flat Python list without an explicit shape yields a vector-shaped constant.
    c = constant_op.constant([1, 2, 3, 4, 5, 6, 7])
    self.assertEqual(c.get_shape(), [7])

  def testExplicitShapeNumber(self):
    # [FUNCTION DOC]
    # Tests that a scalar with an explicit shape of one element yields a constant of that shape.
    c = constant_op.constant(1, shape=[1])
    self.assertEqual(c.get_shape(), [1])

  def testImplicitShapeNumber(self):
    # [FUNCTION DOC]
    # Tests that a plain scalar yields a constant with an empty (scalar) shape.
    c = constant_op.constant(1)
    self.assertEqual(c.get_shape(), [])

  def testShapeTooBig(self):
    # [FUNCTION DOC]
    # Tests that requesting a shape of ten elements for a list of seven elements raises a type error.
    with self.assertRaises(TypeError):
      constant_op.constant([1, 2, 3, 4, 5, 6, 7], shape=[10])

  def testShapeTooSmall(self):
    # [FUNCTION DOC]
    # Tests that requesting a shape of five elements for a list of seven elements raises a type error.
    with self.assertRaises(TypeError):
      constant_op.constant([1, 2, 3, 4, 5, 6, 7], shape=[5])

  def testShapeWrong(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: this repeats the previous too-small-shape test with the same input and shape; the only
    # difference is the explicit unspecified message pattern, so it adds little coverage and its name
    # does not describe a distinct scenario.
    # Tests that a seven-element list with a shape of five elements raises a type error.
    with self.assertRaisesRegex(TypeError, None):
      constant_op.constant([1, 2, 3, 4, 5, 6, 7], shape=[5])

  def testShape(self):
    # [FUNCTION DOC]
    # Tests that the static shape of a constant built from a one-element list can be passed to the
    # shared conversion check (CPU and GPU) and converts back to an equal value.
    self._testAll(constant_op.constant([1]).get_shape())

  def testDimension(self):
    # [FUNCTION DOC]
    # Tests that a single dimension object taken from a static shape converts to a tensor via the shared
    # conversion check.
    x = constant_op.constant([1]).shape[0]
    self._testAll(x)

  def testDimensionList(self):
    # [FUNCTION DOC]
    # Tests that a list holding a dimension object converts to a tensor, alone and when mixed with plain
    # integers on either side.
    x = [constant_op.constant([1]).shape[0]]
    self._testAll(x)

    # Mixing with regular integers is fine too
    self._testAll([1] + x)
    self._testAll(x + [1])

  def testDimensionTuple(self):
    # [FUNCTION DOC]
    # Tests that tuples containing a dimension object, alone or combined with plain integers in either
    # order, convert to tensors.
    x = constant_op.constant([1]).shape[0]
    self._testAll((x,))
    self._testAll((1, x))
    self._testAll((x, 1))

  def testInvalidLength(self):
    # [FUNCTION DOC]
    # Tests error handling for sequence objects whose length method reports a negative size. Builds a
    # helper list type with the bad length and expects value errors mentioning that length should be
    # non-negative when it is used alone or nested at various positions; a mix with plain integers
    # raises a value error about mixed types. Some further cases are left commented out as known gaps
    # in shape checking.

    class BadList(list):
      # [CLASS DOC]
      # HELPER: local list subclass for the invalid-length test. It contains the elements one, two and
      # three but its length method reports a negative number, to exercise constant conversion error
      # handling.

      def __init__(self):
        # [FUNCTION DOC]
        # HELPER: constructor that fills the list with the elements one, two and three via the parent
        # constructor. Takes no extra inputs and returns nothing.
        super(BadList, self).__init__([1, 2, 3])  # pylint: disable=invalid-length-returned

      def __len__(self):  # pylint: disable=invalid-length-returned
        # [FUNCTION DOC]
        # HELPER: deliberately returns the invalid negative length one below zero. Takes no inputs.
        return -1

    with self.assertRaisesRegex(ValueError, "should return >= 0"):
      constant_op.constant([BadList()])
    with self.assertRaisesRegex(ValueError, "mixed types"):
      constant_op.constant([1, 2, BadList()])
    with self.assertRaisesRegex(ValueError, "should return >= 0"):
      constant_op.constant(BadList())
    with self.assertRaisesRegex(ValueError, "should return >= 0"):
      constant_op.constant([[BadList(), 2], 3])
    with self.assertRaisesRegex(ValueError, "should return >= 0"):
      constant_op.constant([BadList(), [1, 2, 3]])
    with self.assertRaisesRegex(ValueError, "should return >= 0"):
      constant_op.constant([BadList(), []])

    # TODO(allenl, josh11b): These cases should return exceptions rather than
    # working (currently shape checking only checks the first element of each
    # sequence recursively). Maybe the first one is fine, but the second one
    # silently truncating is rather bad.

    # with self.assertRaisesRegex(ValueError, "should return >= 0"):
    #   constant_op.constant([[3, 2, 1], BadList()])
    # with self.assertRaisesRegex(ValueError, "should return >= 0"):
    #   constant_op.constant([[], BadList()])

  def testSparseValuesRaiseErrors(self):
    # [FUNCTION DOC]
    # Tests that ragged (non-rectangular) nested lists raise value errors: with an explicit 32-bit
    # integer dtype the error mentions a non-rectangular sequence; without a dtype, and for a three
    # row ragged list, any value error is accepted.
    with self.assertRaisesRegex(ValueError, "non-rectangular Python sequence"):
      constant_op.constant([[1, 2], [3]], dtype=dtypes_lib.int32)

    with self.assertRaisesRegex(ValueError, None):
      constant_op.constant([[1, 2], [3]])

    with self.assertRaisesRegex(ValueError, None):
      constant_op.constant([[1, 2], [3], [4, 5]])

  # TODO(ashankar): This test fails with graph construction since
  # tensor_util.make_tensor_proto (invoked from constant_op.constant)
  # does not handle iterables (it relies on numpy conversion).
  # For consistency, should graph construction handle Python objects
  # that implement the sequence protocol (but not numpy conversion),
  # or should eager execution fail on such sequences?
  def testCustomSequence(self):
    # [FUNCTION DOC]
    # Tests that an object implementing only the Python sequence protocol (length, iteration and keyed
    # access that fails for key zero) converts to a constant by iteration, giving the values one and
    # three. Eager-only: graph construction cannot convert such objects, per the comment above.

    # This is inspired by how many objects in pandas are implemented:
    # - They implement the Python sequence protocol
    # - But may raise a KeyError on __getitem__(self, 0)
    # See https://github.com/tensorflow/tensorflow/issues/20347
    class MySeq(object):
      # [CLASS DOC]
      # HELPER: local sequence-like class used by the custom sequence test. Reports length two, iterates
      # over the values one and three, and raises a key error for any key other than one or three,
      # mimicking pandas-like objects.

      def __getitem__(self, key):
        # [FUNCTION DOC]
        # HELPER: keyed access. Returns the key itself for keys one and three; raises a key error for any
        # other key (including zero).
        if key != 1 and key != 3:
          raise KeyError(key)
        return key

      def __len__(self):
        # [FUNCTION DOC]
        # HELPER: reports a fixed length of two.
        return 2

      def __iter__(self):
        # [FUNCTION DOC]
        # HELPER: returns an iterator over the list of values one and three.
        l = list([1, 3])
        return l.__iter__()

    self.assertAllEqual([1, 3], self.evaluate(constant_op.constant(MySeq())))


class AsTensorTest(test.TestCase):
  # [CLASS DOC]
  # Test case for implicit conversion of values to tensors in eager mode: tensor inputs pass through
  # unchanged and plain Python values become eager tensors.

  def testAsTensorForTensorInput(self):
    # [FUNCTION DOC]
    # Tests that converting something that is already a tensor returns the very same object.
    t = constant_op.constant(10.0)
    x = ops.convert_to_tensor(t)
    self.assertIs(t, x)

  def testAsTensorForNonTensorInput(self):
    # [FUNCTION DOC]
    # Tests that converting a Python float produces an eager tensor instance.
    x = ops.convert_to_tensor(10.0)
    self.assertTrue(isinstance(x, ops.EagerTensor))


class ZerosTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the zeros op in eager mode: shapes given as lists, tuples or tensors, dynamic sizes
  # and dtype handling.

  def _Zeros(self, shape):
    # [FUNCTION DOC]
    # HELPER: builds zeros of the given shape, asserts the static shape equals the request, and returns
    # the numpy value.
    ret = array_ops.zeros(shape)
    self.assertEqual(shape, ret.get_shape())
    return ret.numpy()

  def testConst(self):
    # [FUNCTION DOC]
    # Tests that zeros of shape two by three equals a numpy array of zeros with that shape.
    self.assertTrue(
        np.array_equal(self._Zeros([2, 3]), np.array([[0] * 3] * 2)))

  def testScalar(self):
    # [FUNCTION DOC]
    # Tests scalar zeros built from an empty list, an empty tuple and an empty integer tensor; each
    # gives the value zero.
    self.assertEqual(0, self._Zeros([]))
    self.assertEqual(0, self._Zeros(()))
    scalar = array_ops.zeros(constant_op.constant([], dtype=dtypes_lib.int32))
    self.assertEqual(0, scalar.numpy())

  def testDynamicSizes(self):
    # [FUNCTION DOC]
    # Tests that zeros built from the dynamic shape of a filled tensor has matching values and shape
    # (two by three of zeros).
    np_ans = np.array([[0] * 3] * 2)
    # Creates a tensor of 2 x 3.
    d = array_ops.fill([2, 3], 12., name="fill")
    # Constructs a tensor of zeros of the same dimensions as "d".
    z = array_ops.zeros(array_ops.shape(d))
    out = z.numpy()
    self.assertAllEqual(np_ans, out)
    self.assertShapeEqual(np_ans, d)
    self.assertShapeEqual(np_ans, z)

  def testDtype(self):
    # [FUNCTION DOC]
    # Tests zeros dtypes: the default is 32-bit float for both static and dynamic shapes, and for a list
    # of explicit dtypes (floats, ints, complex, 64-bit int, bool) the result has that dtype, shape two
    # by three, and only zero values. String dtype is left out per a TODO.
    d = array_ops.fill([2, 3], 12., name="fill")
    self.assertEqual(d.get_shape(), [2, 3])
    # Test default type for both constant size and dynamic size
    z = array_ops.zeros([2, 3])
    self.assertEqual(z.dtype, dtypes_lib.float32)
    self.assertEqual([2, 3], z.get_shape())
    self.assertAllEqual(z.numpy(), np.zeros([2, 3]))
    z = array_ops.zeros(array_ops.shape(d))
    self.assertEqual(z.dtype, dtypes_lib.float32)
    self.assertEqual([2, 3], z.get_shape())
    self.assertAllEqual(z.numpy(), np.zeros([2, 3]))
    # Test explicit type control
    for dtype in [
        dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int32,
        dtypes_lib.uint8, dtypes_lib.int16, dtypes_lib.int8,
        dtypes_lib.complex64, dtypes_lib.complex128, dtypes_lib.int64,
        dtypes_lib.bool,
        # TODO(josh11b): Support string type here.
        # dtypes_lib.string
    ]:
      z = array_ops.zeros([2, 3], dtype=dtype)
      self.assertEqual(z.dtype, dtype)
      self.assertEqual([2, 3], z.get_shape())
      z_value = z.numpy()
      self.assertFalse(np.any(z_value))
      self.assertEqual((2, 3), z_value.shape)
      z = array_ops.zeros(array_ops.shape(d), dtype=dtype)
      self.assertEqual(z.dtype, dtype)
      self.assertEqual([2, 3], z.get_shape())
      z_value = z.numpy()
      self.assertFalse(np.any(z_value))
      self.assertEqual((2, 3), z_value.shape)


class ZerosLikeTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the zeros_like op in eager mode across dtypes and cross-dtype requests.

  def _compareZeros(self, dtype, use_gpu):
    # [FUNCTION DOC]
    # HELPER: checks zeros_like for one dtype. Builds a two by three tensor of ones (using a bytes
    # numpy type for strings), applies zeros_like and asserts dtype, static shape, all-zero values and
    # value shape. The use_gpu parameter is accepted but never used.
    # Creates a tensor of non-zero values with shape 2 x 3.
    # NOTE(kearnes): The default numpy dtype associated with tf.string is
    # np.object_ (and can't be changed without breaking a lot things), which
    # causes a TypeError in constant_op.constant below. Here we catch the
    # special case of tf.string and set the numpy dtype appropriately.
    if dtype == dtypes_lib.string:
      numpy_dtype = np.bytes_
    else:
      numpy_dtype = dtype.as_numpy_dtype
    d = constant_op.constant(np.ones((2, 3), dtype=numpy_dtype), dtype=dtype)
    # Constructs a tensor of zeros of the same dimensions and type as "d".
    z_var = array_ops.zeros_like(d)
    # Test that the type is correct
    self.assertEqual(z_var.dtype, dtype)
    # Test that the shape is correct
    self.assertEqual([2, 3], z_var.get_shape())

    # Test that the value is correct
    z_value = z_var.numpy()
    self.assertFalse(np.any(z_value))
    self.assertEqual((2, 3), z_value.shape)

  @test_util.disable_tfrt("b/169112823: unsupported dtype for Op:ZerosLike.")
  def testZerosLikeCPU(self):
    # [FUNCTION DOC]
    # Tests zeros_like for a set of numeric dtypes through the shared helper without GPU. Disabled under
    # the TFRT runtime because of an unsupported dtype for the op. String is excluded per a TODO.
    for dtype in [
        dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int32,
        dtypes_lib.uint8, dtypes_lib.int16, dtypes_lib.int8,
        dtypes_lib.complex64, dtypes_lib.complex128, dtypes_lib.int64,
        # TODO(josh11b): Support string type here.
        # dtypes_lib.string
    ]:
      self._compareZeros(dtype, use_gpu=False)

  @test_util.disable_tfrt("b/169112823: unsupported dtype for Op:ZerosLike.")
  def testZerosLikeGPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name and the use_gpu flag suggest GPU coverage, but the helper ignores that flag,
    # so the same device selection as the CPU variant is used.
    # Tests zeros_like for dtypes including bool, via the shared helper with the GPU flag set. Disabled
    # under the TFRT runtime for an unsupported dtype.
    for dtype in [
        dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int32,
        dtypes_lib.bool, dtypes_lib.int64,
        # TODO(josh11b): Support string type here.
        # dtypes_lib.string
    ]:
      self._compareZeros(dtype, use_gpu=True)

  @test_util.disable_tfrt("b/169112823: unsupported dtype for Op:ZerosLike.")
  def testZerosLikeDtype(self):
    # [FUNCTION DOC]
    # Tests zeros_like with an explicit output dtype for numpy inputs of 32-bit float and complex64,
    # for every input and output combination (including ones that cannot be cast): result dtype, shape
    # and all-zero values are checked. Disabled under the TFRT runtime.
    # Make sure zeros_like works even for dtypes that cannot be cast between
    shape = (3, 5)
    dtypes = np.float32, np.complex64
    for in_type in dtypes:
      x = np.arange(15).astype(in_type).reshape(*shape)
      for out_type in dtypes:
        y = array_ops.zeros_like(x, dtype=out_type).numpy()
        self.assertEqual(y.dtype, out_type)
        self.assertEqual(y.shape, shape)
        self.assertAllEqual(y, np.zeros(shape, dtype=out_type))


class OnesTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the ones op in eager mode: shapes given as lists, tuples or tensors, dynamic sizes
  # and dtype handling.

  def _Ones(self, shape):
    # [FUNCTION DOC]
    # HELPER: builds ones of the given shape, asserts the static shape equals the request, and returns
    # the numpy value.
    ret = array_ops.ones(shape)
    self.assertEqual(shape, ret.get_shape())
    return ret.numpy()

  def testConst(self):
    # [FUNCTION DOC]
    # Tests that ones of shape two by three equals a numpy array of ones with that shape.
    self.assertTrue(np.array_equal(self._Ones([2, 3]), np.array([[1] * 3] * 2)))

  def testScalar(self):
    # [FUNCTION DOC]
    # Tests scalar ones built from an empty list, an empty tuple and an empty integer tensor; each
    # gives the value one.
    self.assertEqual(1, self._Ones([]))
    self.assertEqual(1, self._Ones(()))
    scalar = array_ops.ones(constant_op.constant([], dtype=dtypes_lib.int32))
    self.assertEqual(1, scalar.numpy())

  def testDynamicSizes(self):
    # [FUNCTION DOC]
    # Tests that ones built from the dynamic shape of a filled tensor has matching values and shape.
    np_ans = np.array([[1] * 3] * 2)
    # Creates a tensor of 2 x 3.
    d = array_ops.fill([2, 3], 12., name="fill")
    # Constructs a tensor of ones of the same dimensions as "d".
    z = array_ops.ones(array_ops.shape(d))
    out = z.numpy()
    self.assertAllEqual(np_ans, out)
    self.assertShapeEqual(np_ans, d)
    self.assertShapeEqual(np_ans, z)

  def testDtype(self):
    # [FUNCTION DOC]
    # Tests ones dtypes: default 32-bit float for static and dynamic shapes, and explicit numeric and
    # bool dtypes keep their dtype, shape two by three and all-one values.
    d = array_ops.fill([2, 3], 12., name="fill")
    self.assertEqual(d.get_shape(), [2, 3])
    # Test default type for both constant size and dynamic size
    z = array_ops.ones([2, 3])
    self.assertEqual(z.dtype, dtypes_lib.float32)
    self.assertEqual([2, 3], z.get_shape())
    self.assertAllEqual(z.numpy(), np.ones([2, 3]))
    z = array_ops.ones(array_ops.shape(d))
    self.assertEqual(z.dtype, dtypes_lib.float32)
    self.assertEqual([2, 3], z.get_shape())
    self.assertAllEqual(z.numpy(), np.ones([2, 3]))
    # Test explicit type control
    for dtype in (dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int32,
                  dtypes_lib.uint8, dtypes_lib.int16, dtypes_lib.int8,
                  dtypes_lib.complex64, dtypes_lib.complex128, dtypes_lib.int64,
                  dtypes_lib.bool):
      z = array_ops.ones([2, 3], dtype=dtype)
      self.assertEqual(z.dtype, dtype)
      self.assertEqual([2, 3], z.get_shape())
      self.assertAllEqual(z.numpy(), np.ones([2, 3]))
      z = array_ops.ones(array_ops.shape(d), dtype=dtype)
      self.assertEqual(z.dtype, dtype)
      self.assertEqual([2, 3], z.get_shape())
      self.assertAllEqual(z.numpy(), np.ones([2, 3]))


class OnesLikeTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the ones_like op in eager mode.

  def testOnesLike(self):
    # [FUNCTION DOC]
    # Tests ones_like for a list of numeric dtypes: a two by three tensor of ones of that dtype is
    # converted, and the result must keep the dtype, hold all ones and keep shape two by three.
    for dtype in [
        dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int32,
        dtypes_lib.uint8, dtypes_lib.int16, dtypes_lib.int8,
        dtypes_lib.complex64, dtypes_lib.complex128, dtypes_lib.int64
    ]:
      numpy_dtype = dtype.as_numpy_dtype
      # Creates a tensor of non-zero values with shape 2 x 3.
      d = constant_op.constant(np.ones((2, 3), dtype=numpy_dtype), dtype=dtype)
      # Constructs a tensor of zeros of the same dimensions and type as "d".
      z_var = array_ops.ones_like(d)
      # Test that the type is correct
      self.assertEqual(z_var.dtype, dtype)
      z_value = z_var.numpy()

      # Test that the value is correct
      self.assertTrue(np.array_equal(z_value, np.array([[1] * 3] * 2)))
      self.assertEqual([2, 3], z_var.get_shape())


class FillTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the fill op in eager mode, across dtypes and invalid dimension arguments.

  def _compare(self, dims, val, np_ans, use_gpu):
    # [FUNCTION DOC]
    # HELPER: runs fill for dimensions dims and value val on the first GPU if use_gpu is true and one
    # exists, else on the CPU, and compares the numpy result with the expected array np_ans
    # approximately. Returns nothing.
    ctx = context.context()
    device = "GPU:0" if (use_gpu and ctx.num_gpus()) else "CPU:0"
    with ops.device(device):
      tf_ans = array_ops.fill(dims, val, name="fill")
      out = tf_ans.numpy()
    self.assertAllClose(np_ans, out)

  def _compareAll(self, dims, val, np_ans):
    # [FUNCTION DOC]
    # HELPER: runs the fill comparison twice, once requesting the CPU and once requesting a GPU (which
    # falls back to the CPU when none exists).
    self._compare(dims, val, np_ans, False)
    self._compare(dims, val, np_ans, True)

  def testFillFloat(self):
    # [FUNCTION DOC]
    # Tests fill with a 32-bit float value into shape two by three.
    np_ans = np.array([[3.1415] * 3] * 2).astype(np.float32)
    self._compareAll([2, 3], np_ans[0][0], np_ans)

  def testFillDouble(self):
    # [FUNCTION DOC]
    # Tests fill with a 64-bit float value into shape two by three.
    np_ans = np.array([[3.1415] * 3] * 2).astype(np.float64)
    self._compareAll([2, 3], np_ans[0][0], np_ans)

  def testFillInt32(self):
    # [FUNCTION DOC]
    # Tests fill with a 32-bit integer value into shape two by three.
    np_ans = np.array([[42] * 3] * 2).astype(np.int32)
    self._compareAll([2, 3], np_ans[0][0], np_ans)

  def testFillInt64(self):
    # [FUNCTION DOC]
    # Tests fill with a negative 64-bit integer value into shape two by three.
    np_ans = np.array([[-42] * 3] * 2).astype(np.int64)
    self._compareAll([2, 3], np_ans[0][0], np_ans)

  def testFillComplex64(self):
    # [FUNCTION DOC]
    # Tests fill with a complex64 value into shape two by three, on the CPU only.
    np_ans = np.array([[0.15] * 3] * 2).astype(np.complex64)
    self._compare([2, 3], np_ans[0][0], np_ans, use_gpu=False)

  def testFillComplex128(self):
    # [FUNCTION DOC]
    # Tests fill with a complex128 value into shape two by three, on the CPU only.
    np_ans = np.array([[0.15] * 3] * 2).astype(np.complex128)
    self._compare([2, 3], np_ans[0][0], np_ans, use_gpu=False)

  def testFillString(self):
    # [FUNCTION DOC]
    # Tests fill with a byte-string value into shape two by three; the result equals the expected
    # string array.
    np_ans = np.array([[b"yolo"] * 3] * 2)
    tf_ans = array_ops.fill([2, 3], np_ans[0][0], name="fill").numpy()
    self.assertAllEqual(np_ans, tf_ans)

  def testFillNegative(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the two scalar-looking entries in the list of shapes are bare integers, not
    # one-element tuples, so they test a scalar dims argument rather than a negative dimension vector.
    # Tests that fill raises an invalid-argument error for shapes containing negative sizes or a
    # negative scalar as dims.
    for shape in (-1,), (2, -1), (-1, 2), (-2), (-3):
      with self.assertRaises(errors_impl.InvalidArgumentError):
        array_ops.fill(shape, 7)

  def testShapeFunctionEdgeCases(self):
    # [FUNCTION DOC]
    # Tests shape edge cases of fill: dims that is not a vector, and a fill value that is not a
    # scalar, each raise an invalid-argument error.
    # Non-vector dimensions.
    with self.assertRaises(errors_impl.InvalidArgumentError):
      array_ops.fill([[0, 1], [2, 3]], 1.0)

    # Non-scalar value.
    with self.assertRaises(errors_impl.InvalidArgumentError):
      array_ops.fill([3, 2], [1.0, 2.0])


if __name__ == "__main__":
  test.main()
