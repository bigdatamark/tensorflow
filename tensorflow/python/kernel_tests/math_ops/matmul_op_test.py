# [MODULE DOC]
# Tests for the matrix multiplication operations in the math ops module. Covers the mixed input/output
# type batch matmul (version 3), the matrix-vector product helper, shape and dimension error
# reporting for the infix matrix-multiply operator, and a large generated family of matmul value
# tests and matmul gradient tests parameterized over data types, matrix sizes, static versus
# dynamic shapes, and transpose/adjoint options.
# The generated tests are attached to the MatMulTest and MatMulGradientTest classes only when the
# file is run as a script (in the main block), so they do not exist when the module is merely
# imported. Has no import-time side effects other than defining classes and helpers.
#
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
"""Tests for tensorflow.ops.math_ops.matmul."""

import operator

import numpy as np

from tensorflow.python import tf2
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test as test_lib

# TODO(yangzihao): Currently matmul autotuning is disabled by default. Use
# os.environ["TF_MATMUL_AUTOTUNE_ENABLE"] = "1" to enable it.


@test_util.with_eager_op_as_function
class MatMulMixedType(test_lib.TestCase):
  # [CLASS DOC]
  # Test case covering batch matmul version 3, where the output element type may differ from
  # the input element types. Has no shared fixtures; each test builds its own numpy inputs.
  """Simple test for tf.matmul where Tout is different from T."""

  def testBatchMatMulV3OutputType(self):
    # [FUNCTION DOC]
    # Multiplies two small 2x2 matrices of 8-bit integers (signed, then unsigned), with the second
    # one adjointed, asking for 32-bit integer output. Asserts the result shape is 2x2 and the
    # values equal the expected integer products. Takes no inputs and returns nothing.
    # TODO(shivaniagrawal): uint8 is not supported for mixed matmul type in XLA.
    for (a_dtype, b_dtype) in [(np.int8, np.int8), (np.uint8, np.uint8)]:
      a = np.array([[1, 2], [3, 4]], dtype=a_dtype)
      b = np.array([[1, 2], [3, 4]], dtype=b_dtype)
      c = math_ops.batch_mat_mul_v3(a, b, adj_y=True, Tout=np.int32)
      self.assertAllEqual((2, 2), c.shape)
      self.assertAllEqual([[5, 11], [11, 25]], c)

  def testBatchMatMulV3MixedPrec(self):
    # [FUNCTION DOC]
    # Multiplies an 8-bit integer matrix by a bfloat16 matrix (second adjointed) requesting
    # bfloat16 output. Asserts the result shape is 2x2 and the values match the expected products.
    # Takes no inputs and returns nothing.
    # TODO(shivaniagrawal): uint8 is not supported for mixed matmul type in XLA.
    np_bf16 = dtypes.bfloat16.as_numpy_dtype
    a = np.array([[1, 2], [3, 4]], dtype=np.int8)
    b = np.array([[1, 2], [3, 4]], dtype=np_bf16)
    c = math_ops.batch_mat_mul_v3(a, b, adj_y=True, Tout=np_bf16)
    self.assertAllEqual((2, 2), c.shape)
    self.assertAllEqual([[5, 11], [11, 25]], c)


@test_util.with_eager_op_as_function
class MatVecTest(test_lib.TestCase):
  # [CLASS DOC]
  # Test case for the matrix-vector product helper, which is a thin wrapper over matmul.
  # No shared fixtures.
  """Simple test for matvec, which is sugar on top of matmul."""

  def testTwoByTwoCase(self):
    # [FUNCTION DOC]
    # Multiplies a 2x2 integer matrix with a length-2 vector and asserts the result has shape (2,)
    # and holds the expected row-wise dot products. Returns nothing.
    a = np.array([[1, 2], [3, 4]])
    b = np.array([5, 6])
    c = math_ops.matvec(a, b)
    self.assertAllEqual((2,), c.shape)
    self.assertAllEqual([5 + 2 * 6, 3 * 5 + 4 * 6], c)

  def testEmpty(self):
    # [FUNCTION DOC]
    # Checks matmul with an adjointed first operand when one operand has a zero-sized dimension
    # (3x0 matrix against a 3x2 matrix, in both operand orders). Asserts only that the result
    # shape equals the numpy-computed shape (empty results). Returns nothing.
    full = np.array([[1., 2.], [3., 4.], [5., 6.]])
    empty = np.empty([3, 0])
    self.assertShapeEqual(
        np.matmul(full.T, empty), math_ops.matmul(full, empty, adjoint_a=True))
    self.assertShapeEqual(
        np.matmul(empty.T, full), math_ops.matmul(empty, full, adjoint_a=True))


def _AddTest(test, op_name, testcase_name, fn):
  # [FUNCTION DOC]
  # HELPER: registers dynamically generated test functions on a test class in the main block.
  # Builds a test method name from the operation name and case name, and attaches the given
  # function to the test class wrapped so that it runs only in deprecated graph mode.
  # Inputs: test is the test class, op_name and testcase_name are strings, fn is the test
  # function. Raises a runtime error if a method of that name already exists on the class.
  # Side effect: mutates the test class. Returns nothing.
  test_name = "_".join(["test", op_name, testcase_name])
  if hasattr(test, test_name):
    raise RuntimeError("Test %s defined more than once" % test_name)
  setattr(test, test_name, test_util.deprecated_graph_mode_only(fn))


def _GetTransposedMatrices(x, x_name, kwargs):
  # [FUNCTION DOC]
  # HELPER: computes the effective operand to feed to matmul in the generated tests.
  # Inputs: x is a numpy matrix, x_name is "a" or "b" naming which operand it is, and kwargs is a
  # mapping that must contain transpose and adjoint flags for that operand (keys built from the
  # name). Returns the transpose of x if the transpose flag is exactly true, the conjugate
  # transpose if the adjoint flag is exactly true, otherwise x unchanged. Raises a key error if the
  # flags are missing. No side effects.
  if kwargs["transpose_" + x_name] is True:
    return x.T
  elif kwargs["adjoint_" + x_name] is True:
    return np.conj(x.T)
  else:
    return x


@test_util.with_eager_op_as_function
class MatMulTest(test_lib.TestCase):
  # [CLASS DOC]
  # Initially empty test case; the main block fills it with one generated test per combination of
  # static-shape flag, data type, matrix sizes and transpose/adjoint options, comparing matmul
  # output against numpy.
  pass  # Filled in below


def _GetMatMulTest(a_np_, b_np_, use_static_shape_, **kwargs_):
  # [FUNCTION DOC]
  # HELPER: factory that builds one matmul value-test method for the generated MatMulTest cases.
  # Inputs: a_np_ and b_np_ are numpy matrices (the logical operands), use_static_shape_ chooses
  # constants (static shapes) versus placeholders fed at run time, and the keyword arguments are
  # the transpose/adjoint flags forwarded to matmul. Returns the test function (taking a test
  # case instance), not a result.

  @test_util.run_without_tensor_float_32("Tests matmul")
  def Test(self):
    # [FUNCTION DOC]
    # Generated test body. Computes the numpy reference product of the two matrices, falls back to
    # CPU for half precision if the GPU lacks half matmul support (printing a message), feeds
    # suitably transposed or conjugated operands to matmul, and asserts the result is close to the
    # reference with dtype-dependent tolerances. Returns nothing.
    np_val = np.matrix(a_np_) * np.matrix(b_np_)

    use_gpu = True
    if a_np_.dtype is np.float16 and (
        not test_util.GpuSupportsHalfMatMulAndConv()):
      use_gpu = False
      print("Built without fp16 matmul support for Cuda, running test on CPU.")

    # Transpose and possibly conjugate a_np_ and b_np_ according to the
    # attributes such that tf.matmul(effective_a_np, effective_b_np, **kwargs)
    # results in a valid matrix multiplication and produces the same result as
    # np.matrix(a_np_) * np.matrix(b_np_)
    effective_a_np = _GetTransposedMatrices(a_np_, "a", kwargs_)
    effective_b_np = _GetTransposedMatrices(b_np_, "b", kwargs_)
    with self.cached_session() as sess, test_util.device(use_gpu):
      if use_static_shape_:
        a = constant_op.constant(effective_a_np)
        b = constant_op.constant(effective_b_np)
        res = math_ops.matmul(a, b, **kwargs_)
        tf_val = self.evaluate(res)
      else:
        a = array_ops.placeholder(a_np_.dtype)
        b = array_ops.placeholder(b_np_.dtype)
        res = math_ops.matmul(a, b, **kwargs_)
        tf_val = sess.run(res, feed_dict={a: effective_a_np, b: effective_b_np})

    self.assertAllCloseAccordingToType(
        tf_val,
        np_val,
        float_rtol=3e-5,
        float_atol=3e-5,
        half_rtol=0.2,
        half_atol=0.2)

  return Test


@test_util.with_eager_op_as_function
class MatMulGradientTest(test_lib.TestCase):
  # [CLASS DOC]
  # Initially empty test case; the main block fills it with generated gradient tests that compare
  # theoretical and numerical matmul gradients for each parameter combination.
  pass  # Will be filled in below.


def _GetMatMulGradientTest(a_np_, b_np_, use_static_shape_, **kwargs_):
  # [FUNCTION DOC]
  # HELPER: factory that builds one matmul gradient-test method for the generated
  # MatMulGradientTest cases. Inputs: a_np_ and b_np_ numpy matrices, use_static_shape_ flag, and
  # keyword transpose/adjoint flags forwarded to matmul. Returns the test function.

  @test_util.run_without_tensor_float_32("Tests matmul")
  def Test(self):
    # [FUNCTION DOC]
    # Generated test body. Skips when shapes are not static, when the type is integer or float16,
    # or for bfloat16 without a GPU. Otherwise computes a finite-difference step and tolerance from
    # the type's machine epsilon (hard-coded for bfloat16), then checks the theoretical versus
    # numerical gradient of matmul with respect to each operand in turn. Returns nothing.
    if not use_static_shape_ or a_np_.dtype in (np.int32, np.int64, np.float16):
      self.skipTest("Skipping infeasible gradient test.")

    if (a_np_.dtype == dtypes.bfloat16.as_numpy_dtype and
        not test_util.is_gpu_available()):
      self.skipTest("The bfloat16 tests might fail on CPU")

    # Transpose and possibly conjugate a_np_ and b_np_ according to the
    # attributes such that tf.matmul(effective_a_np, effective_b_np, **kwargs)
    # results in a valid matrix multiplication and produces the same result as
    # np.matrix(a_np_) * np.matrix(b_np_)
    effective_a_np = _GetTransposedMatrices(a_np_, "a", kwargs_)
    effective_b_np = _GetTransposedMatrices(b_np_, "b", kwargs_)

    # np.finfo doesn't support bfloat16. So, we manually compute the eps which
    # defines the difference between 1.0 and the next smallest representable
    # float larger than 1.0. For bfloat16, the difference is 1/128.
    if a_np_.dtype == dtypes.bfloat16.as_numpy_dtype:
      epsilon = 0.0078125
    else:
      epsilon = np.finfo(a_np_.dtype).eps
    delta = epsilon**(1.0 / 3.0)
    tol = 20 * delta
    with self.session():
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          lambda x: math_ops.matmul(x, effective_b_np, **kwargs_),
          [effective_a_np],
          delta=delta)
      self.assertAllClose(theoretical, numerical, rtol=tol, atol=tol)

      theoretical, numerical = gradient_checker_v2.compute_gradient(
          lambda x: math_ops.matmul(effective_a_np, x, **kwargs_),
          [effective_b_np],
          delta=delta)
      self.assertAllClose(theoretical, numerical, rtol=tol, atol=tol)

  return Test


try:
  # @ operator supported since python 3.5.
  infix_matmul = operator.matmul
except AttributeError:

  # For earlier versions of python, emulate regular behavior.
  # Useful to build and test for 3.5+ on earlier versions.
  def infix_matmul(x, y):  # pylint: disable=invalid-name
    # [FUNCTION DOC]
    # KNOWN ISSUE: dead code on any supported Python version, since the operator always exists.
    # HELPER: fallback emulation of the infix matrix-multiply operator, defined only when the
    # standard library lacks it. Tries the left operand's matmul method, then the right operand's
    # reflected method if the types differ and the first gave no result. Returns the product.
    # Raises a type error if neither operand supports the operation.
    try:
      r = type(x).__matmul__(x, y)
    except AttributeError:
      r = NotImplemented
    if r is NotImplemented and type(x) is not type(y):
      try:
        r = type(y).__rmatmul__(y, x)
      except AttributeError:
        r = NotImplemented
    if r is NotImplemented:
      raise TypeError("unsupported operand type(s) for @: '{}' and '{}'"
                      .format(type(x).__name__, type(y).__name__))
    return r


@test_util.with_eager_op_as_function
class MatMulInfixOperatorTest(test_lib.TestCase):
  # [CLASS DOC]
  # Test case for the infix matrix-multiply operator on tensors: error reporting for bad operand
  # shapes and equivalence with the matmul function. No shared fixtures.

  def testMismatchedShape(self):
    # [FUNCTION DOC]
    # Applies the infix operator to a rank-1 tensor and a rank-2 tensor and asserts that an
    # exception is raised whose message matches one of several accepted rank/ndims complaints.
    # Returns nothing.
    with self.assertRaisesRegex(
        Exception, (r"(In\[0\] and In\[1\] has different ndims|In\[0\] "
                    r"ndims must be >= 2|Shape must be rank 2 but is rank 1)")):
      infix_matmul(
          ops.convert_to_tensor([10.0, 20.0, 30.0]),
          ops.convert_to_tensor([[40.0, 50.0], [60.0, 70.0]]))

  def testMismatchedDimensions(self):
    # [FUNCTION DOC]
    # Applies the infix operator to a 1x3 and a 2x2 tensor (inner dimensions disagree) and asserts
    # an exception is raised whose message matches a size-incompatible or dimensions-must-be-equal
    # complaint. Returns nothing.
    with self.assertRaisesRegex(
        Exception, r"(Matrix size-incompatible: In\[0\]: .* In\[1\]|"
        r"Dimensions must be equal)"):
      infix_matmul(
          ops.convert_to_tensor([[10.0, 20.0, 30.0]]),
          ops.convert_to_tensor([[40.0, 50.0], [60.0, 70.0]]))

  @test_util.run_v1_only("Tensor.op is generally not applicable in TF 2")
  def testInfixMatmulIsTfMatmul(self):
    # [FUNCTION DOC]
    # Graph-mode-only test (skipped in TF2). Multiplies a 1x3 by a 3x2 tensor with the infix
    # operator and asserts the producing graph operation has type MatMul. Returns nothing.
    a = ops.convert_to_tensor([[10.0, 20.0, 30.0]])
    b = ops.convert_to_tensor([[40.0, 50.0], [60.0, 70.0], [80.0, 90.0]])
    c = infix_matmul(a, b)
    self.assertEqual(c.op.type, "MatMul")

  def testInfixMatmulDoesDotProduct(self):
    # [FUNCTION DOC]
    # Computes the product of a 1x3 and a 3x2 tensor with both the infix operator and the matmul
    # function, and asserts the two results are equal. Returns nothing.
    a = ops.convert_to_tensor([[10.0, 20.0, 30.0]])
    b = ops.convert_to_tensor([[40.0, 50.0], [60.0, 70.0], [80.0, 90.0]])
    c = infix_matmul(a, b)
    d = math_ops.matmul(a, b)
    self.assertAllEqual(c, d)


if __name__ == "__main__":
  sizes = [1, 3, 5]
  trans_options = [[False, False], [True, False], [False, True]]
  dtypes_to_test = [
      np.int32, np.int64, np.float16, np.float32, np.float64, np.complex64,
      np.complex128, dtypes.bfloat16.as_numpy_dtype
  ]

  # TF2 does not support placeholders under eager so we skip it
  for use_static_shape in set([True, tf2.enabled()]):
    for dtype in dtypes_to_test:
      for m in sizes:
        for n in sizes:
          for k in sizes:
            # Construct compatible random matrices a_np of size [m, k] and b_np
            # of size [k, n].
            # Add seed value to make the tests for bfloat16 stable.
            if dtype == dtypes.bfloat16.as_numpy_dtype:
              np.random.seed(12)
            a_np = np.random.normal(-5, 5, m * k).astype(dtype).reshape([m, k])
            if dtype in (np.complex64, np.complex128):
              a_np.imag = np.random.normal(-5, 5,
                                           m * k).astype(dtype).reshape([m, k])
            b_np = np.random.normal(-5, 5, k * n).astype(dtype).reshape([k, n])
            if dtype in (np.complex64, np.complex128):
              b_np.imag = np.random.normal(-5, 5,
                                           k * n).astype(dtype).reshape([k, n])
            for adjoint_a, transpose_a in trans_options:
              for adjoint_b, transpose_b in trans_options:
                name = "%s_%s_%s_%s_%s_%s_%s_%s_%s" % (
                    use_static_shape, dtype.__name__, m, n, k, adjoint_a,
                    transpose_a, adjoint_b, transpose_b)
                _AddTest(MatMulTest, "MatMulTest", name,
                         _GetMatMulTest(
                             a_np,
                             b_np,
                             use_static_shape,
                             adjoint_a=adjoint_a,
                             transpose_a=transpose_a,
                             adjoint_b=adjoint_b,
                             transpose_b=transpose_b))
                _AddTest(MatMulGradientTest, "MatMulGradientTest", name,
                         _GetMatMulGradientTest(
                             a_np,
                             b_np,
                             use_static_shape,
                             adjoint_a=adjoint_a,
                             transpose_a=transpose_a,
                             adjoint_b=adjoint_b,
                             transpose_b=transpose_b))

  test_lib.main()
