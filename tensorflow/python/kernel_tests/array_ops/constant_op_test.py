# [MODULE DOC]
# Graph-mode (session-based) tests for constant creation and the basic tensor-building ops:
# constant, zeros, zeros_like, ones, ones_like, fill, placeholder, placeholder_with_default and
# identity, plus implicit conversion of values and shapes to tensors. Most tests are marked to run
# under the deprecated version-one graph behavior and evaluate results through sessions. Includes
# a text-format graph definition from an older version to check placeholder shape handling.
# Counterpart of the eager-mode constant tests. No import-time side effects; running the file
# directly invokes the test main entry point.
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

from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat


class ConstantTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the constant op in graph mode: dtype round trips (bfloat16, half, float, double,
  # int, 4-bit ints, complex, string, variant), static shape handling, error cases for ragged data,
  # and eager object-leak checks. Shared helpers compare converted values against numpy.

  def _testCpu(self, x):
    # [FUNCTION DOC]
    # HELPER: shared check for the per-dtype tests. Converts value x to a tensor and evaluates it in a
    # CPU-only session, then compares with numpy: approximately for floating or complex dtypes,
    # exactly otherwise. Returns nothing.
    np_ans = np.array(x)
    with self.cached_session(use_gpu=False):
      tf_ans = ops.convert_to_tensor(x).eval()
    dtype = dtypes_lib.as_dtype(np_ans.dtype)
    if dtype.is_floating or dtype.is_complex:
      self.assertAllClose(np_ans, tf_ans)
    else:
      self.assertAllEqual(np_ans, tf_ans)

  def _testGpu(self, x):
    # [FUNCTION DOC]
    # HELPER: like the CPU check but uses the default cached session, so it runs on a GPU when one is
    # available. Compares the evaluated tensor with numpy approximately for floating or complex dtypes
    # and exactly otherwise. Returns nothing.
    np_ans = np.array(x)
    with self.cached_session():
      tf_ans = ops.convert_to_tensor(x).eval()
    dtype = dtypes_lib.as_dtype(np_ans.dtype)
    if dtype.is_floating or dtype.is_complex:
      self.assertAllClose(np_ans, tf_ans)
    else:
      self.assertAllEqual(np_ans, tf_ans)

  def _testAll(self, x):
    # [FUNCTION DOC]
    # HELPER: runs the CPU check and then the default-device check for value x.
    self._testCpu(x)
    self._testGpu(x)

  def testInvalidDType(self):
    # [FUNCTION DOC]
    # Regression test for a GitHub issue: passing a dtype object where the value belongs, with a string
    # as the dtype, must raise a type error.
    # Test case for GitHub issue 18474
    with self.assertRaises(TypeError):
      constant_op.constant(dtypes_lib.string, "[,]")

  @test_util.run_deprecated_v1
  def testBFloat16(self):
    # [FUNCTION DOC]
    # Tests bfloat16 constants round-trip (ascending, random and empty arrays) on CPU and the default
    # device. Runs under deprecated version-one graph behavior.
    bfloat16 = dtypes_lib.bfloat16.as_numpy_dtype
    self._testAll(np.arange(-15, 15).reshape([2, 3, 5]).astype(bfloat16))
    self._testAll(
        np.random.normal(size=30).reshape([2, 3, 5]).astype(bfloat16))
    self._testAll(np.empty((2, 0, 5)).astype(bfloat16))

  @test_util.run_deprecated_v1
  def testHalf(self):
    # [FUNCTION DOC]
    # Tests half-precision float constants round-trip (ascending, random and empty arrays) on CPU and
    # the default device.
    self._testAll(np.arange(-15, 15).reshape([2, 3, 5]).astype(np.float16))
    self._testAll(
        np.random.normal(size=30).reshape([2, 3, 5]).astype(np.float16))
    self._testAll(np.empty((2, 0, 5)).astype(np.float16))

  @test_util.run_deprecated_v1
  def testFloat(self):
    # [FUNCTION DOC]
    # Tests 32-bit float constants round-trip (ascending, random and empty arrays).
    self._testAll(np.arange(-15, 15).reshape([2, 3, 5]).astype(np.float32))
    self._testAll(
        np.random.normal(size=30).reshape([2, 3, 5]).astype(np.float32))
    self._testAll(np.empty((2, 0, 5)).astype(np.float32))

  @test_util.run_deprecated_v1
  def testDouble(self):
    # [FUNCTION DOC]
    # Tests 64-bit float constants round-trip (ascending, random and empty arrays).
    self._testAll(np.arange(-15, 15).reshape([2, 3, 5]).astype(np.float64))
    self._testAll(
        np.random.normal(size=30).reshape([2, 3, 5]).astype(np.float64))
    self._testAll(np.empty((2, 0, 5)).astype(np.float64))

  @test_util.run_deprecated_v1
  def testInt32(self):
    # [FUNCTION DOC]
    # Tests 32-bit integer constants round-trip (ascending, random and empty arrays).
    self._testAll(np.arange(-15, 15).reshape([2, 3, 5]).astype(np.int32))
    self._testAll((100 * np.random.normal(size=30)).reshape([2, 3, 5]).astype(
        np.int32))
    self._testAll(np.empty((2, 0, 5)).astype(np.int32))

  @test_util.run_deprecated_v1
  def testInt64(self):
    # [FUNCTION DOC]
    # Tests 64-bit integer constants round-trip (ascending, random and empty arrays).
    self._testAll(np.arange(-15, 15).reshape([2, 3, 5]).astype(np.int64))
    self._testAll((100 * np.random.normal(size=30)).reshape([2, 3, 5]).astype(
        np.int64))
    self._testAll(np.empty((2, 0, 5)).astype(np.int64))

  @test_util.run_deprecated_v1
  @test_util.disable_xla("b/183567451: XLA doesn't yet support int4")
  def testInt4(self):
    # [FUNCTION DOC]
    # Tests signed and unsigned 4-bit integer constants round-trip: a range covering the full dtype
    # span, random values and an empty array. Skipped under XLA, which lacks 4-bit support.
    for dtype in [dtypes_lib.int4, dtypes_lib.uint4]:
      np_dtype = dtype.as_numpy_dtype
      self._testAll(
          np.arange(dtype.min, dtype.max + 1)
          .reshape([2, 4, 2])
          .astype(np_dtype)
      )
      self._testAll(
          (7 * np.random.normal(size=30)).reshape([2, 3, 5]).astype(np_dtype)
      )
      self._testAll(np.empty((2, 0, 5)).astype(np_dtype))

  @test_util.run_deprecated_v1
  def testComplex64(self):
    # [FUNCTION DOC]
    # Tests complex64 constants round-trip (ascending, random, empty arrays with a complex factor).
    self._testAll(
        (1 + 2j) * np.arange(-15, 15).reshape([2, 3, 5]).astype(np.complex64))
    self._testAll(
        (1 + 2j) *
        np.random.normal(size=30).reshape([2, 3, 5]).astype(np.complex64))
    self._testAll(np.empty((2, 0, 5)).astype(np.complex64))

  @test_util.run_deprecated_v1
  def testComplex128(self):
    # [FUNCTION DOC]
    # Tests complex128 constants round-trip (ascending, random, empty arrays with a complex factor).
    self._testAll(
        (1 + 2j) * np.arange(-15, 15).reshape([2, 3, 5]).astype(np.complex128))
    self._testAll(
        (1 + 2j) *
        np.random.normal(size=30).reshape([2, 3, 5]).astype(np.complex128))
    self._testAll(np.empty((2, 0, 5)).astype(np.complex128))

  @test_util.run_deprecated_v1
  def testString(self):
    # [FUNCTION DOC]
    # Tests string constants round-trip on CPU only: an array of byte strings and an empty string array.
    self._testCpu(
        np.array([compat.as_bytes(str(x)) for x in np.arange(-15, 15)]).reshape(
            [2, 3, 5]))
    self._testCpu(np.empty((2, 0, 5)).astype(np.str_))

  @test_util.run_deprecated_v1
  def testVariant(self):
    # [FUNCTION DOC]
    # Tests a constant holding a variant value built from a tensor proto with one integer payload.
    # Checks the stored value attribute on the op equals the proto, then runs a print op over it as a
    # smoke test that execution works. CPU only, since copying variants between CPU and GPU is not
    # supported yet (TODO).
    # TODO(ebrevdo): Re-enable use_gpu=True once non-DMA Variant
    # copying between CPU and GPU is supported.
    with self.session(use_gpu=False):
      variant_tensor = tensor_pb2.TensorProto(
          dtype=dtypes_lib.variant.as_datatype_enum,
          tensor_shape=tensor_shape.TensorShape([]).as_proto(),
          variant_val=[
              tensor_pb2.VariantTensorDataProto(
                  # Match registration in variant_op_registry.cc
                  type_name=b"int",
                  metadata=np.array(1, dtype=np.int32).tobytes())
          ])
      const = constant_op.constant(variant_tensor)
      const_value = const.op.get_attr("value")

      # Ensure we stored the tensor proto properly.
      self.assertProtoEquals(variant_tensor, const_value)

      # Smoke test -- ensure this executes without trouble.
      # Right now, non-numpy-compatible objects cannot be returned from a
      # session.run call; similarly, objects that can't be converted to
      # native numpy types cannot be passed to ops.convert_to_tensor.
      # TODO(ebrevdo): Add registration mechanism for
      # ops.convert_to_tensor and for session.run output.
      logging_const_op = logging_ops.Print(
          const, [const],
          message="Variant storing an int, decoded const value:").op
      logging_const_op.run()

  @test_util.run_deprecated_v1
  def testStringWithNulls(self):
    # [FUNCTION DOC]
    # Tests byte strings with null bytes (all nulls, embedded nulls, nested lists) survive conversion
    # and evaluation; lengths and contents are preserved. The nested case compares via list conversion
    # because array comparison would drop null terminators.
    with self.cached_session():
      val = ops.convert_to_tensor(b"\0\0\0\0").eval()
    self.assertEqual(len(val), 4)
    self.assertEqual(val, b"\0\0\0\0")

    with self.cached_session():
      val = ops.convert_to_tensor(b"xx\0xx").eval()
    self.assertEqual(len(val), 5)
    self.assertAllEqual(val, b"xx\0xx")
    nested = [[b"\0\0\0\0", b"xx\0xx"], [b"\0_\0_\0_\0", b"\0"]]

    with self.cached_session():
      val = ops.convert_to_tensor(nested).eval()
    # NOTE(mrry): Do not use assertAllEqual, because it converts nested to a
    #   numpy array, which loses the null terminators.
    self.assertEqual(val.tolist(), nested)

  def testExplicitShapeNumPy(self):
    # [FUNCTION DOC]
    # Tests that a numpy float array with an explicit matching shape, built in a fresh graph, keeps the
    # static shape two by three by five.
    with ops.Graph().as_default():
      c = constant_op.constant(
          np.arange(-15, 15).reshape([2, 3, 5]).astype(np.float32),
          shape=[2, 3, 5])
    self.assertEqual(c.get_shape(), [2, 3, 5])

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testEagerMemory(self):
    # [FUNCTION DOC]
    # Eager-mode leak check: a decorator asserts that creating a small constant while executing eagerly
    # creates no new Python objects that remain alive. The body only creates the constant.
    """Tests PyObject refs are managed correctly when executing eagerly."""
    constant_op.constant([[1.]])

  def testImplicitShapeNumPy(self):
    # [FUNCTION DOC]
    # Tests that a numpy float array without shape argument gives the array's static shape in a graph.
    with ops.Graph().as_default():
      c = constant_op.constant(
          np.arange(-15, 15).reshape([2, 3, 5]).astype(np.float32))
    self.assertEqual(c.get_shape(), [2, 3, 5])

  def testExplicitShapeList(self):
    # [FUNCTION DOC]
    # Tests that a flat list of seven numbers with shape seven gives that static shape in a graph.
    with ops.Graph().as_default():
      c = constant_op.constant([1, 2, 3, 4, 5, 6, 7], shape=[7])
    self.assertEqual(c.get_shape(), [7])

  def testImplicitShapeList(self):
    # [FUNCTION DOC]
    # Tests that a flat list of seven numbers without a shape gives a vector shape in a graph.
    with ops.Graph().as_default():
      c = constant_op.constant([1, 2, 3, 4, 5, 6, 7])
    self.assertEqual(c.get_shape(), [7])

  def testExplicitShapeNumber(self):
    # [FUNCTION DOC]
    # Tests that a scalar with a one-element shape gives that static shape in a graph.
    with ops.Graph().as_default():
      c = constant_op.constant(1, shape=[1])
    self.assertEqual(c.get_shape(), [1])

  def testImplicitShapeNumber(self):
    # [FUNCTION DOC]
    # Tests that a plain scalar gives an empty static shape in a graph.
    with ops.Graph().as_default():
      c = constant_op.constant(1)
    self.assertEqual(c.get_shape(), [])

  def testShapeInconsistent(self):
    # [FUNCTION DOC]
    # Tests inconsistent shapes: the legacy version-one constant function accepts a list of seven with a
    # shape of ten and reports the requested shape, while the current constant function raises a type
    # error saying the tensor shape was unexpected.
    with ops.Graph().as_default():
      c = constant_op.constant_v1([1, 2, 3, 4, 5, 6, 7], shape=[10])
    self.assertEqual(c.get_shape(), [10])

    with ops.Graph().as_default():
      with self.assertRaisesRegex(TypeError, "Expected Tensor's shape"):
        c = constant_op.constant([1, 2, 3, 4, 5, 6, 7], shape=[10])

  def testPromotionShapes(self):
    # [FUNCTION DOC]
    # Tests shape promotion: a one-element list or a scalar with a requested shape of ten is expanded,
    # and the result has the static shape ten.
    with ops.Graph().as_default():
      c = constant_op.constant([7], shape=[10])
    self.assertEqual(c.get_shape(), [10])
    with ops.Graph().as_default():
      c = constant_op.constant(3, shape=[10])
    self.assertEqual(c.get_shape(), [10])

  # pylint: disable=g-long-lambda
  def testShapeWrong(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the surrounding comments disable and re-enable a long-lambda lint rule, but no
    # lambda is used in this test.
    # Tests too-small requested shape: the legacy constant function raises a value error saying too many
    # elements were provided, while the current constant function raises a type error about the tensor's
    # expected shape.
    with ops.Graph().as_default():
      with self.assertRaisesRegex(ValueError, "Too many elements provided."):
        constant_op.constant_v1([1, 2, 3, 4, 5, 6, 7], shape=[5])
      with self.assertRaisesRegex(TypeError, "Expected Tensor's shape"):
        constant_op.constant([1, 2, 3, 4, 5, 6, 7], shape=[5])

  # pylint: enable=g-long-lambda
  # TODO(b/35396543): Temporarily disable: suspicion that
  # this is causing test timeouts.
  def _testTooLargeConstant(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: dead code. The leading underscore keeps the test runner from collecting it (disabled
    # due to suspected timeouts) and the created constant is never used.
    # Intended to check that a constant from a very large array raises a value error saying the tensor
    # proto content exceeds two gigabytes.
    with ops.Graph().as_default():
      large_array = np.zeros((512, 1024, 1024), dtype=np.float32)
      with self.assertRaisesRegex(
          ValueError,
          "Cannot create a tensor proto whose content is larger than 2GB."):
        c = constant_op.constant(large_array)

  # TODO(b/35396543): Temporarily disable: suspicion that
  # this is causing test timeouts.
  def _testTooLargeGraph(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: dead code. The leading underscore keeps the test runner from collecting it (disabled
    # due to suspected timeouts) and the two constants are never used afterwards.
    # Intended to check that serializing a graph holding two large constants to a graph definition
    # raises a value error about exceeding two gigabytes.
    with ops.Graph().as_default() as g:
      large_array = np.zeros((256, 1024, 1024), dtype=np.float32)
      c = constant_op.constant(large_array)
      d = constant_op.constant(large_array)
      with self.assertRaisesRegex(ValueError,
                                  "GraphDef cannot be larger than 2GB."):
        g.as_graph_def()

  @test_util.run_deprecated_v1
  def testSparseValuesRaiseErrors(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the constants are assigned to a variable that is never read; only the raised errors
    # matter.
    # Tests ragged nested lists raise value errors in graph mode: with an explicit integer dtype the
    # error mentions setting an array element with a sequence; without a dtype the error mentions a
    # non-dense tensor or an inhomogeneous shape; also for a three-row ragged list.
    with self.assertRaisesRegex(ValueError,
                                "setting an array element with a sequence"):
      c = constant_op.constant([[1, 2], [3]], dtype=dtypes_lib.int32)

    with self.assertRaisesRegex(
        ValueError, "(Expected.*to be a dense tensor|inhomogeneous shape)"):
      c = constant_op.constant([[1, 2], [3]])

    with self.assertRaisesRegex(
        ValueError, "(Expected.*to be a dense tensor|inhomogeneous shape)"):
      c = constant_op.constant([[1, 2], [3], [4, 5]])


class AsTensorTest(test.TestCase):
  # [CLASS DOC]
  # Test case for implicit conversion to tensors in graph mode: tensors pass through, Python values
  # become graph tensors, and shapes and dimensions become integer tensors.

  def testAsTensorForTensorInput(self):
    # [FUNCTION DOC]
    # Tests that converting an existing graph tensor returns the same object.
    with ops.Graph().as_default():
      t = constant_op.constant(10.0)
      x = ops.convert_to_tensor(t)
    self.assertIs(t, x)

  def testAsTensorForNonTensorInput(self):
    # [FUNCTION DOC]
    # Tests that converting a Python float inside a graph yields a graph tensor instance.
    with ops.Graph().as_default():
      x = ops.convert_to_tensor(10.0)
    self.assertTrue(isinstance(x, tensor.Tensor))

  def testAsTensorForShapeInput(self):
    # [FUNCTION DOC]
    # Tests conversion of static shape objects: fully known shapes become 32-bit integer vectors with
    # the right values, including a shape with a dimension of two to the power 31 minus one; larger
    # dimensions become 64-bit unless an explicit 32-bit dtype is requested, which raises a value error
    # about a dimension being too large; explicit 64-bit dtype works for small shapes; shapes can
    # be passed as the shape argument of reshape. Partially known or unknown-rank shapes raise a value
    # error saying the shape is partially known; asking for a float dtype raises a type error.
    with self.cached_session():
      x = ops.convert_to_tensor(tensor_shape.TensorShape([]))
      self.assertEqual(dtypes_lib.int32, x.dtype)
      self.assertAllEqual([], self.evaluate(x))

      x = ops.convert_to_tensor(tensor_shape.TensorShape([1, 2, 3]))
      self.assertEqual(dtypes_lib.int32, x.dtype)
      self.assertAllEqual([1, 2, 3], self.evaluate(x))

      x = ops.convert_to_tensor(tensor_shape.TensorShape([2**31-1, 2, 3]))
      self.assertEqual(dtypes_lib.int32, x.dtype)
      self.assertAllEqual([2**31 - 1, 2, 3], self.evaluate(x))

      x = ops.convert_to_tensor(tensor_shape.TensorShape([2**31-1, 2, 3]),
                                dtype=dtypes_lib.int32)
      self.assertEqual(dtypes_lib.int32, x.dtype)
      self.assertAllEqual([2**31 - 1, 2, 3], self.evaluate(x))

      x = ops.convert_to_tensor(tensor_shape.TensorShape([2**31, 2, 3]))
      self.assertEqual(dtypes_lib.int64, x.dtype)
      self.assertAllEqual([2**31, 2, 3], self.evaluate(x))

      x = ops.convert_to_tensor(tensor_shape.TensorShape([2**31, 2, 3]),
                                dtype=dtypes_lib.int64)
      self.assertEqual(dtypes_lib.int64, x.dtype)
      self.assertAllEqual([2**31, 2, 3], self.evaluate(x))

      with self.assertRaisesRegex(ValueError,
                                  "a dimension is too large.*int64"):
        x = ops.convert_to_tensor(tensor_shape.TensorShape([2**31, 2, 3]),
                                  dtype=dtypes_lib.int32)

      x = ops.convert_to_tensor(
          tensor_shape.TensorShape([1, 2, 3]), dtype=dtypes_lib.int64)
      self.assertEqual(dtypes_lib.int64, x.dtype)
      self.assertAllEqual([1, 2, 3], self.evaluate(x))

      x = array_ops.reshape(
          array_ops.zeros([6]), tensor_shape.TensorShape([2, 3]))
      self.assertAllEqual([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], self.evaluate(x))

    with self.assertRaisesRegex(ValueError, "partially known"):
      ops.convert_to_tensor(tensor_shape.TensorShape(None))

    with self.assertRaisesRegex(ValueError, "partially known"):
      ops.convert_to_tensor(tensor_shape.TensorShape([1, None, 64]))

    with self.assertRaises(TypeError):
      ops.convert_to_tensor(
          tensor_shape.TensorShape([1, 2, 3]), dtype=dtypes_lib.float32)

  @test_util.run_deprecated_v1
  def testAsTensorForDimensionInput(self):
    # [FUNCTION DOC]
    # Tests conversion of single dimension objects to scalar integer tensors (32-bit by default, 64-bit
    # on request) with value two. For an unknown shape, the dimension of an unknown-rank or unknown
    # size raises a value error whose message depends on the shape behavior version: one saying None
    # values are unsupported under the version-two behavior, otherwise one about an unknown dimension.
    with self.cached_session():
      x = ops.convert_to_tensor(tensor_shape.TensorShape([1, 2, 3])[1])
      self.assertEqual(dtypes_lib.int32, x.dtype)
      self.assertAllEqual(2, self.evaluate(x))

      x = ops.convert_to_tensor(
          tensor_shape.TensorShape([1, 2, 3])[1], dtype=dtypes_lib.int64)
      self.assertEqual(dtypes_lib.int64, x.dtype)
      self.assertAllEqual(2, self.evaluate(x))

    shape = tensor_shape.TensorShape(None)
    if shape._v2_behavior:
      with self.assertRaisesRegex(ValueError, "None values not supported"):
        ops.convert_to_tensor(shape[1])
      with self.assertRaisesRegex(ValueError, "None values not supported"):
        ops.convert_to_tensor(tensor_shape.TensorShape([1, None, 64])[1])
    else:
      with self.assertRaisesRegex(ValueError, "unknown Dimension"):
        ops.convert_to_tensor(shape[1])
      with self.assertRaisesRegex(ValueError, "unknown Dimension"):
        ops.convert_to_tensor(tensor_shape.TensorShape([1, None, 64])[1])


class IdentityOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the identity op in graph mode.

  def testIdTensor(self):
    # [FUNCTION DOC]
    # Tests identity on a constant vector: the op input is a tensor and the resulting node definition
    # matches the expected identity op with float type and an input named input.
    with ops.Graph().as_default():
      x = constant_op.constant(2.0, shape=[6], name="input")
      id_op = array_ops.identity(x, name="id")
    self.assertTrue(isinstance(id_op.op.inputs[0], tensor.Tensor))
    self.assertProtoEquals("name: 'id' op: 'Identity' input: 'input' "
                           "attr { key: 'T' value { type: DT_FLOAT } }",
                           id_op.op.node_def)


class ZerosTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the zeros op in graph mode: shapes, dynamic sizes, dtypes including string and
  # quantized types.

  def _Zeros(self, shape):
    # [FUNCTION DOC]
    # HELPER: builds zeros of the given shape in a cached session, asserts the static shape equals the
    # request, and returns the evaluated value.
    with self.cached_session():
      ret = array_ops.zeros(shape)
      self.assertEqual(shape, ret.get_shape())
      return self.evaluate(ret)

  def testConst(self):
    # [FUNCTION DOC]
    # Tests that zeros of shape two by three equals a numpy zero array of that shape.
    self.assertTrue(
        np.array_equal(self._Zeros([2, 3]), np.array([[0] * 3] * 2)))

  def testScalar(self):
    # [FUNCTION DOC]
    # Tests scalar zeros from empty list, empty tuple and empty integer tensor; each evaluates to zero.
    self.assertEqual(0, self._Zeros([]))
    self.assertEqual(0, self._Zeros(()))
    with self.cached_session():
      scalar = array_ops.zeros(constant_op.constant([], dtype=dtypes_lib.int32))
      self.assertEqual(0, self.evaluate(scalar))

  def testDynamicSizes(self):
    # [FUNCTION DOC]
    # Tests zeros whose shape comes from the dynamic shape of a filled tensor; values and shapes must
    # match a numpy zero array.
    np_ans = np.array([[0] * 3] * 2)
    with self.cached_session():
      # Creates a tensor of 2 x 3.
      d = array_ops.fill([2, 3], 12., name="fill")
      # Constructs a tensor of zeros of the same dimensions as "d".
      z = array_ops.zeros(array_ops.shape(d))
      out = self.evaluate(z)
    self.assertAllEqual(np_ans, out)
    self.assertShapeEqual(np_ans, d)
    self.assertShapeEqual(np_ans, z)

  @test_util.run_deprecated_v1
  def testDtype(self):
    # [FUNCTION DOC]
    # Tests zeros dtypes: default 32-bit float for static and dynamic shapes, and explicit numeric,
    # bool and string dtypes keep their dtype and shape and evaluate to all-zero values.
    with self.cached_session():
      d = array_ops.fill([2, 3], 12., name="fill")
      self.assertEqual(d.get_shape(), [2, 3])
      # Test default type for both constant size and dynamic size
      z = array_ops.zeros([2, 3])
      self.assertEqual(z.dtype, dtypes_lib.float32)
      self.assertEqual([2, 3], z.get_shape())
      self.assertAllEqual(z, np.zeros([2, 3]))
      z = array_ops.zeros(array_ops.shape(d))
      self.assertEqual(z.dtype, dtypes_lib.float32)
      self.assertEqual([2, 3], z.get_shape())
      self.assertAllEqual(z, np.zeros([2, 3]))
      # Test explicit type control
      for dtype in [
          dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int32,
          dtypes_lib.uint8, dtypes_lib.int16, dtypes_lib.int8,
          dtypes_lib.complex64, dtypes_lib.complex128, dtypes_lib.int64,
          dtypes_lib.bool, dtypes_lib.string
      ]:
        z = array_ops.zeros([2, 3], dtype=dtype)
        self.assertEqual(z.dtype, dtype)
        self.assertEqual([2, 3], z.get_shape())
        z_value = self.evaluate(z)
        self.assertFalse(np.any(z_value))
        self.assertEqual((2, 3), z_value.shape)
        z = array_ops.zeros(array_ops.shape(d), dtype=dtype)
        self.assertEqual(z.dtype, dtype)
        self.assertEqual([2, 3], z.get_shape())
        z_value = self.evaluate(z)
        self.assertFalse(np.any(z_value))
        self.assertEqual((2, 3), z_value.shape)

  @test_util.disable_tfrt("b/169901260")
  def testQint8Dtype(self):
    # [FUNCTION DOC]
    # Tests zeros with 8-bit quantized integer dtype: dtype and shape are right and the values, cast to
    # 32-bit integer for comparison with numpy, are all zero. Disabled under the TFRT runtime.
    dtype = dtypes_lib.qint8
    z = array_ops.zeros([2, 3], dtype=dtype)
    self.assertEqual(z.dtype, dtype)
    self.assertEqual([2, 3], z.get_shape())
    # cast to int32 so that it can be compred with numpy
    # where [qint|quint][8|16] are not available.
    z_value = self.evaluate(math_ops.cast(z, dtypes_lib.int32))
    self.assertFalse(np.any(z_value))

  @test_util.disable_tfrt("b/169901260")
  def testQint16Dtype(self):
    # [FUNCTION DOC]
    # Tests zeros with 16-bit quantized integer dtype: dtype and shape are right and the values, cast to
    # 32-bit integer, are all zero. Disabled under the TFRT runtime.
    dtype = dtypes_lib.qint16
    z = array_ops.zeros([2, 3], dtype=dtype)
    self.assertEqual(z.dtype, dtype)
    self.assertEqual([2, 3], z.get_shape())
    # cast to int32 so that it can be compred with numpy
    # where [qint|quint][8|16] are not available.
    z_value = self.evaluate(math_ops.cast(z, dtypes_lib.int32))
    self.assertFalse(np.any(z_value))

  @test_util.disable_tfrt("b/169901260")
  def testQint32Dtype(self):
    # [FUNCTION DOC]
    # Tests zeros with 32-bit quantized integer dtype: dtype and shape are right and the values, cast to
    # 32-bit integer, are all zero. Disabled under the TFRT runtime.
    dtype = dtypes_lib.qint32
    z = array_ops.zeros([2, 3], dtype=dtype)
    self.assertEqual(z.dtype, dtype)
    self.assertEqual([2, 3], z.get_shape())
    # cast to int32 so that it can be compred with numpy
    # where [qint|quint][8|16] are not available.
    z_value = self.evaluate(math_ops.cast(z, dtypes_lib.int32))
    self.assertFalse(np.any(z_value))


class ZerosLikeTest(test.TestCase):
  # [CLASS DOC]
  # Test case for zeros_like in graph mode, including unknown shapes via placeholders and variant
  # values.

  def _compareZeros(self, dtype, fully_defined_shape, use_gpu):
    # [FUNCTION DOC]
    # HELPER: checks zeros_like for one dtype. Parameters: dtype, whether the input has a fully defined
    # shape (otherwise a placeholder fed at evaluation time), and whether to use the GPU session.
    # Builds a two by three tensor of ones (byte strings for the string dtype), applies zeros_like and
    # asserts dtype, static shape when known, all-zero values and a value shape of two by three.
    with self.cached_session(use_gpu=use_gpu):
      # Creates a tensor of non-zero values with shape 2 x 3.
      # NOTE(kearnes): The default numpy dtype associated with tf.string is
      # np.object_ (and can't be changed without breaking a lot things), which
      # causes a TypeError in constant_op.constant below. Here we catch the
      # special case of tf.string and set the numpy dtype appropriately.
      if dtype == dtypes_lib.string:
        numpy_dtype = np.bytes_
      else:
        numpy_dtype = dtype.as_numpy_dtype
      if fully_defined_shape:
        d = constant_op.constant(
            np.ones((2, 3), dtype=numpy_dtype), dtype=dtype)
      else:
        d = array_ops.placeholder(dtype=dtype)
      # Constructs a tensor of zeros of the same dimensions and type as "d".
      z_var = array_ops.zeros_like(d)
      # Test that the type is correct
      self.assertEqual(z_var.dtype, dtype)
      # Test that the shape is correct
      if fully_defined_shape:
        self.assertEqual([2, 3], z_var.get_shape())

      # Test that the value is correct
      feed_dict = {}
      if not fully_defined_shape:
        feed_dict[d] = np.ones((2, 3), dtype=numpy_dtype)
      z_value = z_var.eval(feed_dict=feed_dict)
      self.assertFalse(np.any(z_value))
      self.assertEqual((2, 3), z_value.shape)

  @test_util.run_deprecated_v1
  def testZerosLikeCPU(self):
    # [FUNCTION DOC]
    # Tests zeros_like on CPU for many dtypes (including string) with both defined and undefined shape.
    for dtype in [
        dtypes_lib.half, dtypes_lib.float32, dtypes_lib.float64,
        dtypes_lib.int8, dtypes_lib.uint8, dtypes_lib.int16, dtypes_lib.uint16,
        dtypes_lib.int32, dtypes_lib.int64, dtypes_lib.bool,
        dtypes_lib.complex64, dtypes_lib.complex128, dtypes_lib.string
    ]:
      self._compareZeros(dtype, fully_defined_shape=False, use_gpu=False)
      self._compareZeros(dtype, fully_defined_shape=True, use_gpu=False)

  @test_util.run_deprecated_v1
  def testZerosLikeGPU(self):
    # [FUNCTION DOC]
    # Tests zeros_like for a subset of dtypes with both defined and undefined shape, selecting the
    # GPU session in the helper.
    for dtype in [
        dtypes_lib.half, dtypes_lib.float32, dtypes_lib.float64,
        dtypes_lib.int32, dtypes_lib.int64, dtypes_lib.complex64,
        dtypes_lib.complex128, dtypes_lib.bool
    ]:
      self._compareZeros(dtype, fully_defined_shape=False, use_gpu=True)
      self._compareZeros(dtype, fully_defined_shape=True, use_gpu=True)

  @test_util.run_deprecated_v1
  def testZerosLikePartialShape(self):
    # [FUNCTION DOC]
    # Tests that zeros_like of a placeholder with partially unknown shape preserves the static shape
    # list.
    d = array_ops.placeholder(dtypes_lib.float32, shape=[None, 4, None])
    z = array_ops.zeros_like(d)
    self.assertEqual(d.get_shape().as_list(), z.get_shape().as_list())

  @test_util.run_deprecated_v1
  def testZerosLikeDtype(self):
    # [FUNCTION DOC]
    # Tests zeros_like with an explicit dtype for numpy inputs of float and complex type, for each
    # input/output pairing: result dtype, shape and all-zero values.
    # Make sure zeros_like works even for dtypes that cannot be cast between
    with self.cached_session():
      shape = (3, 5)
      dtypes = np.float32, np.complex64
      for in_type in dtypes:
        x = np.arange(15).astype(in_type).reshape(*shape)
        for out_type in dtypes:
          y = array_ops.zeros_like(x, dtype=out_type).eval()
          self.assertEqual(y.dtype, out_type)
          self.assertEqual(y.shape, shape)
          self.assertAllEqual(y, np.zeros(shape, dtype=out_type))

  @test_util.run_deprecated_v1
  def testZerosLikeVariant(self):
    # [FUNCTION DOC]
    # Tests zeros_like on a variant tensor holding an integer payload, built from a tensor proto, on CPU
    # only (GPU support missing, TODO). Prints both input and output through a print op and runs it as
    # a smoke test that no error occurs.
    # TODO(ebrevdo): Re-enable use_gpu=True once non-DMA Variant
    # copying between CPU and GPU is supported AND we register a
    # ZerosLike callback for GPU for Variant storing primitive types
    # in variant_op_registry.cc.
    with self.session(use_gpu=False):
      variant_tensor = tensor_pb2.TensorProto(
          dtype=dtypes_lib.variant.as_datatype_enum,
          tensor_shape=tensor_shape.TensorShape([]).as_proto(),
          variant_val=[
              tensor_pb2.VariantTensorDataProto(
                  # Match registration in variant_op_registry.cc
                  type_name=b"int",
                  metadata=np.array(1, dtype=np.int32).tobytes())
          ])
      const_variant = constant_op.constant(variant_tensor)
      zeros_like = array_ops.zeros_like(const_variant)
      zeros_like_op = logging_ops.Print(
          zeros_like, [const_variant, zeros_like],
          message="Variant storing an int, input and output of zeros_like:").op

      # Smoke test -- ensure this executes without trouble.
      # Right now, non-numpy-compatible objects cannot be returned from a
      # session.run call; similarly, objects that can't be converted to
      # native numpy types cannot be passed to ops.convert_to_tensor.
      # TODO(ebrevdo): Add registration mechanism for
      # ops.convert_to_tensor and for session.run output.
      zeros_like_op.run()


class OnesTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the ones op in graph mode: shapes, dynamic sizes, automatic packing of tensor
  # dimensions, dtypes and quantized types.

  def _Ones(self, shape):
    # [FUNCTION DOC]
    # HELPER: builds ones of the given shape in a cached session, asserts the static shape equals the
    # request, and returns the evaluated value.
    with self.cached_session():
      ret = array_ops.ones(shape)
      self.assertEqual(shape, ret.get_shape())
      return self.evaluate(ret)

  def testConst(self):
    # [FUNCTION DOC]
    # Tests that ones of shape two by three equals a numpy array of ones of that shape.
    self.assertTrue(np.array_equal(self._Ones([2, 3]), np.array([[1] * 3] * 2)))

  def testScalar(self):
    # [FUNCTION DOC]
    # Tests scalar ones from empty list, empty tuple and empty integer tensor; each evaluates to one.
    self.assertEqual(1, self._Ones([]))
    self.assertEqual(1, self._Ones(()))
    with self.cached_session():
      scalar = array_ops.ones(constant_op.constant([], dtype=dtypes_lib.int32))
      self.assertEqual(1, self.evaluate(scalar))

  def testDynamicSizes(self):
    # [FUNCTION DOC]
    # Tests ones whose shape comes from the dynamic shape of a filled tensor; values and shapes match.
    np_ans = np.array([[1] * 3] * 2)
    with self.cached_session():
      # Creates a tensor of 2 x 3.
      d = array_ops.fill([2, 3], 12., name="fill")
      # Constructs a tensor of ones of the same dimensions as "d".
      z = array_ops.ones(array_ops.shape(d))
      out = self.evaluate(z)
    self.assertAllEqual(np_ans, out)
    self.assertShapeEqual(np_ans, d)
    self.assertShapeEqual(np_ans, z)

  @test_util.run_deprecated_v1
  def testAutoPack(self):
    # [FUNCTION DOC]
    # Tests that a shape list made of scalar integer placeholders is packed automatically: ones fed
    # with height four and width sixteen evaluates to a four by sixteen array of ones.
    with self.cached_session():
      h = array_ops.placeholder(dtypes_lib.int32, shape=[])
      w = array_ops.placeholder(dtypes_lib.int32, shape=[])
      z = array_ops.ones([h, w])
      out = z.eval(feed_dict={h: 4, w: 16})
    self.assertAllEqual(out, np.array([[1] * 16] * 4))

  @test_util.run_deprecated_v1
  def testDtype(self):
    # [FUNCTION DOC]
    # Tests ones dtypes: default 32-bit float for static and dynamic shapes; explicit numeric and bool
    # dtypes keep dtype, shape two by three and all-one values.
    with self.cached_session():
      d = array_ops.fill([2, 3], 12., name="fill")
      self.assertEqual(d.get_shape(), [2, 3])
      # Test default type for both constant size and dynamic size
      z = array_ops.ones([2, 3])
      self.assertEqual(z.dtype, dtypes_lib.float32)
      self.assertEqual([2, 3], z.get_shape())
      self.assertAllEqual(z, np.ones([2, 3]))
      z = array_ops.ones(array_ops.shape(d))
      self.assertEqual(z.dtype, dtypes_lib.float32)
      self.assertEqual([2, 3], z.get_shape())
      self.assertAllEqual(z, np.ones([2, 3]))
      # Test explicit type control
      for dtype in (dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int32,
                    dtypes_lib.uint8, dtypes_lib.int16, dtypes_lib.int8,
                    dtypes_lib.complex64, dtypes_lib.complex128,
                    dtypes_lib.int64, dtypes_lib.bool):
        z = array_ops.ones([2, 3], dtype=dtype)
        self.assertEqual(z.dtype, dtype)
        self.assertEqual([2, 3], z.get_shape())
        self.assertAllEqual(z, np.ones([2, 3]))
        z = array_ops.ones(array_ops.shape(d), dtype=dtype)
        self.assertEqual(z.dtype, dtype)
        self.assertEqual([2, 3], z.get_shape())
        self.assertAllEqual(z, np.ones([2, 3]))

  @test_util.disable_tfrt("b/169901260")
  def testQintDtype(self):
    # [FUNCTION DOC]
    # Tests ones with an unsigned 8-bit quantized dtype inside a compiled function with autograph
    # turned off; the result cast to 32-bit integer must be all nonzero. Disabled under the TFRT runtime.

    @def_function.function(autograph=False)
    def f():
      # [FUNCTION DOC]
      # HELPER: inner traced function that creates a two by three tensor of quantized ones and returns it
      # cast to 32-bit integer. Takes no inputs.
      return math_ops.cast(
          array_ops.ones([2, 3], dtype=dtypes_lib.quint8), dtypes_lib.int32)

    value = self.evaluate(f())
    self.assertTrue(np.all(value))


class OnesLikeTest(test.TestCase):
  # [CLASS DOC]
  # Test case for ones_like in graph mode.

  def testOnesLike(self):
    # [FUNCTION DOC]
    # Tests ones_like for many dtypes: a two by three tensor of ones of that dtype is converted, and
    # the result keeps dtype, equals all ones and has static shape two by three.
    for dtype in [
        dtypes_lib.float32, dtypes_lib.float64, dtypes_lib.int8,
        dtypes_lib.uint8, dtypes_lib.int16, dtypes_lib.uint16, dtypes_lib.int32,
        dtypes_lib.int64, dtypes_lib.bool, dtypes_lib.complex64,
        dtypes_lib.complex128
    ]:
      numpy_dtype = dtype.as_numpy_dtype
      with self.cached_session():
        # Creates a tensor of non-zero values with shape 2 x 3.
        d = constant_op.constant(
            np.ones(
                (2, 3), dtype=numpy_dtype), dtype=dtype)
        # Constructs a tensor of zeros of the same dimensions and type as "d".
        z_var = array_ops.ones_like(d)
        # Test that the type is correct
        self.assertEqual(z_var.dtype, dtype)
        z_value = self.evaluate(z_var)

      # Test that the value is correct
      self.assertTrue(np.array_equal(z_value, np.array([[1] * 3] * 2)))
      self.assertEqual([2, 3], z_var.get_shape())

  @test_util.run_deprecated_v1
  def testOnesLikePartialShape(self):
    # [FUNCTION DOC]
    # Tests that ones_like of a placeholder with partially unknown shape preserves the static shape.
    d = array_ops.placeholder(dtypes_lib.float32, shape=[None, 4, None])
    z = array_ops.ones_like(d)
    self.assertEqual(d.get_shape().as_list(), z.get_shape().as_list())


class FillTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the fill op in graph mode: dtypes, negative dimensions, shape inference and
  # gradients.

  def _compare(self, dims, val, np_ans, use_gpu):
    # [FUNCTION DOC]
    # HELPER: evaluates fill for dimensions dims and value val in a cached session (GPU if use_gpu is
    # true) and compares exactly with expected array np_ans. Returns nothing.
    with self.cached_session(use_gpu=use_gpu):
      tf_ans = array_ops.fill(dims, val, name="fill")
      out = self.evaluate(tf_ans)
    self.assertAllEqual(np_ans, out)
    # Fill does not set the shape.
    # self.assertShapeEqual(np_ans, tf_ans)

  def _compareAll(self, dims, val, np_ans):
    # [FUNCTION DOC]
    # HELPER: runs the fill comparison once with GPU disabled and once enabled.
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

  @test_util.disable_xla("b/183567451: XLA doesn't yet support int4")
  def testFillInt4(self):
    # [FUNCTION DOC]
    # Tests fill with signed and unsigned 4-bit integer values into shape two by three. Skipped under
    # XLA, which lacks 4-bit support.
    np_ans = np.array([[-6] * 3] * 2).astype(dtypes_lib.int4.as_numpy_dtype)
    self._compareAll([2, 3], np_ans[0][0], np_ans)
    np_ans = np.array([[11] * 3] * 2).astype(dtypes_lib.uint4.as_numpy_dtype)
    self._compareAll([2, 3], np_ans[0][0], np_ans)

  def testFillComplex64(self):
    # [FUNCTION DOC]
    # Tests fill with a complex64 value into shape two by three.
    np_ans = np.array([[0.15 + 0.3j] * 3] * 2).astype(np.complex64)
    self._compareAll([2, 3], np_ans[0][0], np_ans)

  def testFillComplex128(self):
    # [FUNCTION DOC]
    # Tests fill with a complex128 value into shape two by three.
    np_ans = np.array([[0.15 + 0.3j] * 3] * 2).astype(np.complex128)
    self._compareAll([2, 3], np_ans[0][0], np_ans)

  @test_util.run_deprecated_v1
  def testFillString(self):
    # [FUNCTION DOC]
    # Tests fill with a byte-string value on CPU; the result equals the expected string array.
    np_ans = np.array([[b"yolo"] * 3] * 2)
    with self.session(use_gpu=False):
      tf_ans = array_ops.fill([2, 3], np_ans[0][0], name="fill").eval()
    self.assertAllEqual(np_ans, tf_ans)

  @test_util.run_deprecated_v1
  def testFillNegative(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the shape list contains bare integers for the last two entries, not one-element
    # tuples, so they test scalar dims rather than negative vectors.
    # Tests negative dims: static negative shapes raise value errors at graph construction time; the
    # same shapes fed through a placeholder raise invalid-argument errors when evaluated.
    with self.cached_session():
      for shape in (-1,), (2, -1), (-1, 2), (-2), (-3):
        with self.assertRaises(ValueError):
          array_ops.fill(shape, 7)

      # Using a placeholder so this won't be caught in static analysis.
      dims = array_ops.placeholder(dtypes_lib.int32)
      fill_t = array_ops.fill(dims, 3.0)
      for shape in (-1,), (2, -1), (-1, 2), (-2), (-3):
        with self.assertRaises(errors_impl.InvalidArgumentError):
          fill_t.eval({dims: shape})

  @test_util.run_deprecated_v1
  def testShapeFunctionEdgeCases(self):
    # [FUNCTION DOC]
    # Tests shape inference edge cases: non-vector dims and non-scalar value raise value errors; dims
    # fed from a placeholder of known length four gives four unknown dimensions; a list mixing a
    # scalar placeholder and 17 gives an unknown dimension followed by seventeen.
    # Non-vector dimensions.
    with self.assertRaises(ValueError):
      array_ops.fill([[0, 1], [2, 3]], 1.0)

    # Non-scalar value.
    with self.assertRaises(ValueError):
      array_ops.fill([3, 2], [1.0, 2.0])

    # Partial dimension information.
    f = array_ops.fill(array_ops.placeholder(dtypes_lib.int32, shape=(4,)), 3.0)
    self.assertEqual([None, None, None, None], f.get_shape().as_list())

    f = array_ops.fill(
        [array_ops.placeholder(
            dtypes_lib.int32, shape=()), 17], 1.0)
    self.assertEqual([None, 17], f.get_shape().as_list())

  @test_util.run_deprecated_v1
  def testGradient(self):
    # [FUNCTION DOC]
    # Tests the gradient of fill with respect to the scalar value: compares the computed gradient with a
    # numerical one for shape three by two and requires the error below one thousandth.
    with self.cached_session():
      in_v = constant_op.constant(5.0)
      out_shape = [3, 2]
      out_filled = array_ops.fill(out_shape, in_v)
      err = gradient_checker.compute_gradient_error(in_v, [], out_filled,
                                                    out_shape)
    self.assertLess(err, 1e-3)


class PlaceholderTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the placeholder op in graph mode: dtype and shape feeding, unknown and partial
  # shapes, error messages, control dependencies, printing and loading an older graph definition.

  @test_util.run_deprecated_v1
  def testDtype(self):
    # [FUNCTION DOC]
    # Tests that a float placeholder returns the fed array when fed through identity, and that
    # evaluating without feeding raises an op error saying a value must be fed for placeholder p with
    # dtype float.
    with self.cached_session():
      p = array_ops.placeholder(dtypes_lib.float32, shape=(10, 10), name="p")
      p_identity = array_ops.identity(p)
      feed_array = np.random.rand(10, 10)
      self.assertAllClose(
          p_identity.eval(feed_dict={p: feed_array}), feed_array)

      with self.assertRaisesOpError(
          "must feed a value for placeholder tensor 'p' with dtype float"):
        self.evaluate(p_identity)

  @test_util.run_deprecated_v1
  def testShape(self):
    # [FUNCTION DOC]
    # Tests a placeholder with fixed shape ten by ten: feeding a matching array works; not feeding
    # raises an op error that mentions dtype and shape; feeding a value of the wrong shape raises a
    # value error saying the shape cannot be fed.
    with self.cached_session():
      p = array_ops.placeholder(dtypes_lib.float32, shape=(10, 10), name="p")
      p_identity = array_ops.identity(p)
      feed_array = np.random.rand(10, 10)
      self.assertAllClose(
          p_identity.eval(feed_dict={p: feed_array}), feed_array)

      with self.assertRaisesOpError(
          "must feed a value for placeholder tensor 'p' with dtype float and "
          r"shape \[10,10\]"):
        self.evaluate(p_identity)

      with self.assertRaisesWithPredicateMatch(
          ValueError, lambda e: "Cannot feed value of shape" in str(e)):
        p_identity.eval(feed_dict={p: feed_array[:5, :5]})

  @test_util.run_deprecated_v1
  def testUnknownShape(self):
    # [FUNCTION DOC]
    # Tests a placeholder with unknown shape accepts fed arrays of different ranks and shapes.
    with self.cached_session():
      p = array_ops.placeholder(dtypes_lib.float32, shape=None, name="p")
      p_identity = array_ops.identity(p)
      # can feed anything
      feed_array = np.random.rand(10, 3)
      self.assertAllClose(
          p_identity.eval(feed_dict={p: feed_array}), feed_array)
      feed_array = np.random.rand(4, 2, 5)
      self.assertAllClose(
          p_identity.eval(feed_dict={p: feed_array}), feed_array)

  @test_util.run_deprecated_v1
  def testScalarShape(self):
    # [FUNCTION DOC]
    # Tests a scalar placeholder: feeding the value five returns five.
    with self.cached_session():
      p = array_ops.placeholder(dtypes_lib.float32, shape=[], name="p")
      p_identity = array_ops.identity(p)
      self.assertAllClose(p_identity.eval(feed_dict={p: 5}), 5)

  @test_util.run_deprecated_v1
  def testPartialShape(self):
    # [FUNCTION DOC]
    # Tests a placeholder with one unknown and one known dimension: matching feed works and a feed with
    # the wrong second dimension raises a value error saying the shape cannot be fed.
    with self.cached_session():
      p = array_ops.placeholder(dtypes_lib.float32, shape=[None, 3], name="p")
      p_identity = array_ops.identity(p)
      feed_array = np.random.rand(10, 3)
      self.assertAllClose(
          p_identity.eval(feed_dict={p: feed_array}), feed_array)

      with self.assertRaisesWithPredicateMatch(
          ValueError, lambda e: "Cannot feed value of shape" in str(e)):
        p_identity.eval(feed_dict={p: feed_array[:5, :2]})

  @test_util.run_deprecated_v1
  def testPartialShapeWhenNotFed(self):
    # [FUNCTION DOC]
    # Tests that evaluating a partial-shape placeholder without feeding raises an op error about the
    # missing placeholder value, not a shape error.
    with self.cached_session():
      p = array_ops.placeholder(dtypes_lib.float32, shape=[None, 3], name="p")
      p_identity = array_ops.identity(p)

      # Should trigger an operator error, not a shape error.
      with self.assertRaisesOpError(
          "must feed a value for placeholder tensor 'p' with dtype float"):
        self.evaluate(p_identity)

  @test_util.run_deprecated_v1
  def testControlDependency(self):
    # [FUNCTION DOC]
    # Tests that a constant created under a control dependency on a placeholder, multiplied by the
    # placeholder, evaluates when the placeholder is fed two: the result is ten.
    with self.cached_session():
      p = array_ops.placeholder(dtypes_lib.int32, shape=[], name="p")
      with ops.control_dependencies([p]):
        c = constant_op.constant(5, dtypes_lib.int32)
      d = math_ops.multiply(p, c)
      val = np.array(2).astype(np.int64)
      self.assertEqual(10, d.eval(feed_dict={p: val}))

  @test_util.run_deprecated_v1
  def testBadShape(self):
    # [FUNCTION DOC]
    # Tests that creating a placeholder with a negative dimension raises a value error.
    with self.assertRaises(ValueError):
      array_ops.placeholder(dtypes_lib.float32, shape=(-1, 10))

  @test_util.run_deprecated_v1
  def testTensorStr(self):
    # [FUNCTION DOC]
    # Tests the printed representation of placeholders: unknown shape, fixed shape, and partial shape
    # with a quantized dtype, where the unknown dimension is shown as None or a question mark depending
    # on the shape behavior version.
    a = array_ops.placeholder(dtypes_lib.float32, shape=None, name="a")
    self.assertEqual("<tf.Tensor 'a:0' shape=<unknown> dtype=float32>", repr(a))

    b = array_ops.placeholder(dtypes_lib.int32, shape=(32, 40), name="b")
    self.assertEqual("<tf.Tensor 'b:0' shape=(32, 40) dtype=int32>", repr(b))

    c = array_ops.placeholder(dtypes_lib.qint32, shape=(32, None, 2), name="c")
    if c.shape._v2_behavior:
      self.assertEqual(
          "<tf.Tensor 'c:0' shape=(32, None, 2) dtype=qint32>", repr(c))
    else:
      self.assertEqual(
          "<tf.Tensor 'c:0' shape=(32, ?, 2) dtype=qint32>", repr(c))

  @test_util.run_deprecated_v1
  def testOldGraph(self):
    # [FUNCTION DOC]
    # Tests loading a graph definition produced by an earlier version, in which the placeholder shape
    # is empty and not unknown-rank. With producer version 21 that empty shape means any shape, so
    # feeding a two-element vector must give two plus one results of two and three. Imports the graph in
    # a cached session and evaluates the add output.
    # Load graph generated from earlier version of TF where
    # placeholder shape was not set.
    #
    # a = tf.compat.v1.placeholder(tf.float32)
    # b = a + 1.0
    #
    # Older graph's default shape is 'shape {}', not 'shape {
    # unknown_rank: true }'
    graph = """
node {
  name: "Placeholder"
  op: "Placeholder"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "shape"
    value {
      shape {
      }
    }
  }
}
node {
  name: "add/y"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
        }
        float_val: 1.0
      }
    }
  }
}
node {
  name: "add"
  op: "Add"
  input: "Placeholder"
  input: "add/y"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
}
versions {
  producer: 21
}
"""
    gdef = graph_pb2.GraphDef()
    text_format.Merge(graph, gdef)
    with self.cached_session():
      p, ret = importer.import_graph_def(
          gdef, return_elements=["Placeholder:0", "add:0"])

      # Feed in a vector of two elements.  Since the producer version
      # of 21, a shape of {} is interpreted as "any shape".  If
      # producer version were 22, then we'd get a shape mismatch
      # error.
      self.assertAllEqual([2.0, 3.0], ret.eval(feed_dict={p: [1.0, 2.0]}))


class PlaceholderWithDefaultTest(test.TestCase):
  # [CLASS DOC]
  # Test case for placeholder_with_default in graph mode: shape enforcement on feeds and gradients.

  @test_util.run_deprecated_v1
  def testFullShape(self):
    # [FUNCTION DOC]
    # Tests placeholder_with_default with a fixed two by two shape: evaluates to the default, accepts
    # a fed array of the same shape and raises a value error for a wrongly shaped feed.
    with self.session(force_gpu=test_util.is_gpu_available()):
      p = array_ops.placeholder_with_default([[2, 2], [2, 2]], shape=[2, 2])
      a = array_ops.identity(p)
      self.assertAllEqual([[2, 2], [2, 2]], self.evaluate(a))
      self.assertAllEqual(
          [[3, 3], [3, 3]], a.eval(feed_dict={p: [[3, 3], [3, 3]]}))

      with self.assertRaises(ValueError):
        a.eval(feed_dict={p: [[6, 6, 6], [6, 6, 6]]})

  @test_util.run_deprecated_v1
  def testPartialShape(self):
    # [FUNCTION DOC]
    # Tests placeholder_with_default with a vector of unknown length: default works, feeds of other
    # lengths work and a rank-two feed raises a value error.
    with self.session(force_gpu=test_util.is_gpu_available()):
      p = array_ops.placeholder_with_default([1, 2, 3], shape=[None])
      a = array_ops.identity(p)
      self.assertAllEqual([1, 2, 3], self.evaluate(a))
      self.assertAllEqual([3, 37], a.eval(feed_dict={p: [3, 37]}))

      with self.assertRaises(ValueError):
        a.eval(feed_dict={p: [[2, 2], [2, 2]]})

  @test_util.run_deprecated_v1
  def testNoShape(self):
    # [FUNCTION DOC]
    # Tests placeholder_with_default with no shape constraint: default works and feeds of different
    # ranks are accepted.
    with self.session(force_gpu=test_util.is_gpu_available()):
      p = array_ops.placeholder_with_default([17], shape=None)
      a = array_ops.identity(p)
      self.assertAllEqual([17], self.evaluate(a))
      self.assertAllEqual([3, 37], a.eval(feed_dict={p: [3, 37]}))
      self.assertAllEqual(
          [[3, 3], [3, 3]], a.eval(feed_dict={p: [[3, 3], [3, 3]]}))

  @test_util.run_deprecated_v1
  def testGradient(self):
    # [FUNCTION DOC]
    # Tests the gradient through placeholder_with_default fed by a placeholder: gradient error between
    # analytic and numerical values for shape five by seven must be below one thousandth.
    with self.session(force_gpu=test_util.is_gpu_available()):
      x = array_ops.placeholder(dtypes_lib.float32, [5, 7])
      y = array_ops.placeholder_with_default(x, None)
      err = gradient_checker.compute_gradient_error(x, [5, 7], y, [5, 7])
      self.assertLess(err, 1e-3)


if __name__ == "__main__":
  test.main()
