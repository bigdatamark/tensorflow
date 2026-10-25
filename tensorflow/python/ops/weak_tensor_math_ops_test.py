# [MODULE DOC]
# Tests that math_ops functions and operators behave correctly with weak tensors (WeakTensor) under
# the "all" dtype conversion mode. Test cases cover: reductions (sum, variance, standard deviation,
# count of nonzero, log-sum-exp) and their result types for weak, python, numpy and regular tensor
# inputs; rounding, sign gradient, reciprocal with no NaN, equality with none and floats, erfcinv,
# argmax/argmin, casting with full type information, reflected-operator dispatch and error
# raising, scalar multiplication on variables and indexed slices, comparison ops (equal, maximum,
# minimum) mixing python/numpy/tensor inputs, in-place variable updates (allowed and disallowed
# dtype combinations, listed in module-level tables), floor/truncate division edge cases for
# integers, and division with no NaN.
# Import-time side effects: defines module-level tables of allowed and disallowed (variable dtype,
# delta) combinations. Running as a script sets the dtype conversion mode to "all" before launching
# the test runner.
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
"""Tests for tensorflow.ops.math_ops on WeakTensor."""

import itertools

from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import full_type_pb2
from tensorflow.python import tf2
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.framework.weak_tensor import WeakTensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import weak_tensor_ops  # pylint: disable=unused-import
from tensorflow.python.ops import weak_tensor_test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import googletest

_convert_to_input_type = weak_tensor_test_util.convert_to_input_type
_get_weak_tensor = weak_tensor_test_util.get_weak_tensor


@test_util.run_all_in_graph_and_eager_modes
class ReduceTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case (run in graph and eager modes) for reduction ops in math_ops with weak tensor
  # semantics: sum, variance, standard deviation, count of nonzero. Checks values, result types
  # (weak versus regular tensor depending on the input kind) and error cases. Parameterized.
  # Test unary ops with optional dtype arg.

  @parameterized.parameters(
      ("WeakTensor", WeakTensor),
      ("Python", WeakTensor),
      ("NumPy", tensor.Tensor),
      ("Tensor", tensor.Tensor),
  )
  def testReduceAllDims(self, input_type, result_type):
    # [FUNCTION DOC]
    # Parameterized by the kind of input (weak tensor, python list, numpy array, regular tensor)
    # and the expected result class. Converts a 2x3 int32 matrix to that input kind, sums all
    # elements on the preferred (GPU if available) device, and asserts the result class and that
    # the value is 21.
    test_input = _convert_to_input_type(
        [[1, 2, 3], [4, 5, 6]], input_type, dtypes.int32
    )
    with test_util.device(use_gpu=True):
      res = math_ops.reduce_sum(test_input)
      self.assertIsInstance(res, result_type)
      self.assertEqual(self.evaluate(res), 21)

  def testReduceExtendType(self):
    # [FUNCTION DOC]
    # Sums a large random 1000x1000 float32 matrix once as a float32 weak tensor and once after
    # casting to bfloat16, then asserts the bfloat16 sum is close (tolerance 1e-3) to the float32
    # sum cast to bfloat16 (checking the reduction accumulates in a wider type).
    test_in = np.random.randn(1000, 1000).astype(np.float32)
    in_f32 = _get_weak_tensor(test_in, dtypes.float32)
    in_bfl6 = math_ops.cast(test_in, dtypes.bfloat16)

    out_f32 = self.evaluate(math_ops.reduce_sum(in_f32))
    out_bf16 = self.evaluate(math_ops.reduce_sum(in_bfl6))
    expected = math_ops.cast(out_f32, dtypes.bfloat16)

    self.assertAllClose(out_bf16, expected, 1e-3)

  def testCountNonzero(self):
    # [FUNCTION DOC]
    # Checks count_nonzero gives 2 for a weak int32 matrix with two nonzero entries, 2 for the
    # boolean mask derived from it, and 512 for an int8 numpy array of 512 twos (guarding against
    # accumulating in int8).
    # simple case
    x = _get_weak_tensor([[0, -2, 0], [4, 0, 0]], dtypes.int32)
    self.assertEqual(self.evaluate(math_ops.count_nonzero(x)), 2)

    # boolean input
    x = math_ops.not_equal(x, 0)
    self.assertEqual(self.evaluate(math_ops.count_nonzero(x)), 2)

    # would overflow if int8 would be used for internal calculations
    x = 2 * np.ones(512, dtype=np.int8)
    self.assertEqual(self.evaluate(math_ops.count_nonzero(x)), 512)

  @parameterized.parameters(
      ("WeakTensor", WeakTensor),
      ("Python", WeakTensor),
      ("NumPy", tensor.Tensor),
      ("Tensor", tensor.Tensor),
  )
  def testReduceExplicitAxes(self, input_type, result_type):
    # [FUNCTION DOC]
    # Parameterized by input kind and expected result class. Sums a 2x3 int32 matrix along axis 0
    # and -2 (expects 5, 7, 9), axis 1 and -1 (expects 6, 15), and many axis tuples or no axis
    # (expects 21), asserting the result class each time.
    x = _convert_to_input_type([[1, 2, 3], [4, 5, 6]], input_type, dtypes.int32)
    with test_util.device(use_gpu=True):
      for axis in (0, -2):
        res = math_ops.reduce_sum(x, axis=axis)
        self.assertIsInstance(res, result_type)
        self.assertAllEqual(res, [5, 7, 9])
      for axis in (1, -1):
        res = math_ops.reduce_sum(x, axis=axis)
        self.assertIsInstance(res, result_type)
        self.assertAllEqual(res, [6, 15])
      for axis in (None, (0, 1), (1, 0), (-1, 0), (0, -1), (-2, 1), (1, -2),
                   (-1, -2), (-2, -1)):
        res = math_ops.reduce_sum(x, axis=axis)
        self.assertIsInstance(res, result_type)
        self.assertEqual(self.evaluate(res), 21)

  def testReduceInvalidAxis(self):
    # [FUNCTION DOC]
    # In eager mode returns immediately without checking anything (the shape check only happens
    # at graph construction time). In graph mode, passes a rank-2 axis array to reduce_sum and
    # asserts a value error saying the axis must be at most rank 1.
    if context.executing_eagerly():
      # The shape check is in run a graph construction time. In eager mode,
      # it misses the check, magically return result given wrong shape.
      return
    x = _get_weak_tensor([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    axis = np.array([[0], [1]])
    with self.assertRaisesRegex(ValueError, "must be at most rank 1"):
      math_ops.reduce_sum(x, axis)

  def testReduceVar(self):
    # [FUNCTION DOC]
    # Checks reduce_variance: zero for an all-zero float32 weak matrix (also along axis 0); a type
    # error ("must be either real or complex") for an integer weak matrix; 0.25 for float weak and
    # numpy float32 inputs (matching numpy); and the per-column variances for a ragged tensor.
    x = _get_weak_tensor([[0, 0, 0], [0, 0, 0]], dtype=dtypes.float32)
    self.assertAllClose(self.evaluate(math_ops.reduce_variance(x)), 0)
    self.assertAllClose(
        self.evaluate(math_ops.reduce_variance(x, axis=0)), [0, 0, 0])

    x = _get_weak_tensor([[1, 2, 1, 1], [1, 1, 0, 1]])
    with self.assertRaisesRegex(TypeError, "must be either real or complex"):
      math_ops.reduce_variance(x)

    x = _get_weak_tensor([[1.0, 2.0, 1.0, 1.0], [1.0, 1.0, 0.0, 1.0]])
    self.assertEqual(self.evaluate(math_ops.reduce_variance(x)), 0.25)
    x_np = np.array([[1, 2, 1, 1], [1, 1, 0, 1]], "float32")
    self.assertEqual(np.var(x_np), 0.25)
    self.assertEqual(self.evaluate(math_ops.reduce_variance(x_np)), 0.25)

    x = ragged_factory_ops.constant([[5., 1., 4., 1.], [], [5., 9., 2.], [5.],
                                     []])
    self.assertAllClose(math_ops.reduce_variance(x, axis=0), [0., 16., 1., 0.])

  def testReduceVarComplex(self):
    # [FUNCTION DOC]
    # For a python complex list, a complex64 array and a complex128 array, asserts reduce_variance
    # returns 1.0 (matching numpy) with a real result dtype: float64, float32 and float64
    # respectively.
    # Ensure that complex values are handled to be consistent with numpy
    complex_ys = [([0 - 1j, 0 + 1j], dtypes.float64),
                  (np.array([0 - 1j, 0 + 1j], "complex64"), dtypes.float32),
                  (np.array([0 - 1j, 0 + 1j], "complex128"), dtypes.float64)]
    for y, dtype in complex_ys:
      y_result = math_ops.reduce_variance(y)
      self.assertEqual(np.var(y), 1.0)
      self.assertEqual(self.evaluate(y_result), 1.0)
      self.assertEqual(y_result.dtype, dtype)

  def testReduceStd(self):
    # [FUNCTION DOC]
    # Checks reduce_std: zero for an all-zero weak matrix (and along axis 0); a type error for an
    # integer weak matrix; 0.5 for a python nested float list with a WeakTensor result; 0.5 for
    # the numpy version with a regular Tensor result; and column standard deviations for a ragged
    # tensor.
    x = _get_weak_tensor([[0, 0, 0], [0, 0, 0]], dtypes.float32)
    self.assertAllClose(self.evaluate(math_ops.reduce_std(x)), 0)
    self.assertAllClose(
        self.evaluate(math_ops.reduce_std(x, axis=0)), [0, 0, 0])

    x = _get_weak_tensor([[1, 2, 1, 1], [1, 1, 0, 1]])
    with self.assertRaisesRegex(TypeError, "must be either real or complex"):
      math_ops.reduce_std(x)

    x = [[1., 2., 1., 1.], [1., 1., 0., 1.]]
    res = math_ops.reduce_std(x)
    self.assertEqual(self.evaluate(res), 0.5)
    self.assertIsInstance(res, WeakTensor)
    x_np = np.array(x)
    self.assertEqual(np.std(x_np), 0.5)
    self.assertEqual(self.evaluate(math_ops.reduce_std(x_np)), 0.5)
    self.assertIsInstance(math_ops.reduce_std(x_np), tensor.Tensor)

    x = ragged_factory_ops.constant([[5., 1., 4., 1.], [], [5., 9., 2.], [5.],
                                     []])
    self.assertAllClose(math_ops.reduce_std(x, axis=0), [0., 4., 1., 0.])

  def testReduceStdComplex(self):
    # [FUNCTION DOC]
    # For a python complex list, a complex64 array and a complex128 array, asserts reduce_std
    # returns 1.0 (matching numpy) with a real result dtype: float64, float32 and float64.
    # Ensure that complex values are handled to be consistent with numpy
    complex_ys = [([0 - 1j, 0 + 1j], dtypes.float64),
                  (np.array([0 - 1j, 0 + 1j], "complex64"), dtypes.float32),
                  (np.array([0 - 1j, 0 + 1j], "complex128"), dtypes.float64)]
    for y, dtype in complex_ys:
      y_result = math_ops.reduce_std(y)
      self.assertEqual(np.std(y), 1.0)
      self.assertEqual(self.evaluate(y_result), 1.0)
      self.assertEqual(y_result.dtype, dtype)


@test_util.run_all_in_graph_and_eager_modes
class LogSumExpTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case (graph and eager modes) for reduce_logsumexp: axis handling, keeping dimensions,
  # numerical stability for very large and very small values, and negative infinity input.
  # Compares against a numpy reference on random float16, float32 and float64 data.

  def testReduceLogSumExp(self):
    # [FUNCTION DOC]
    # For float16, float32 and float64 random 5x5 numpy arrays, asserts reduce_logsumexp over all
    # elements is close to the numpy log of the sum of exponentials.
    for dtype in [np.float16, np.float32, np.double]:
      x_np = np.random.rand(5, 5).astype(dtype)
      with test_util.use_gpu():
        y_tf_np = math_ops.reduce_logsumexp(x_np)
        y_np = np.log(np.sum(np.exp(x_np)))
        self.assertAllClose(y_tf_np, y_np)

  def testReductionIndices(self):
    # [FUNCTION DOC]
    # For float16, float32 and float64 random 5x5 arrays, reduces with logsumexp over a list
    # containing axis 0, asserts the static shape matches the numpy reference and the values are
    # close.
    for dtype in [np.float16, np.float32, np.double]:
      x_np = np.random.rand(5, 5).astype(dtype)
      with test_util.use_gpu():
        y_tf = math_ops.reduce_logsumexp(x_np, axis=[0])
        y_np = np.log(np.sum(np.exp(x_np), axis=0))
        self.assertShapeEqual(y_np, y_tf)
        y_tf_np = self.evaluate(y_tf)
        self.assertAllClose(y_tf_np, y_np)

  def testReductionIndices2(self):
    # [FUNCTION DOC]
    # Same as the previous reduction-indices test but passes axis 0 as a plain integer instead of
    # a list; asserts shape and values match the numpy reference.
    for dtype in [np.float16, np.float32, np.double]:
      x_np = np.random.rand(5, 5).astype(dtype)
      with test_util.use_gpu():
        y_tf = math_ops.reduce_logsumexp(x_np, axis=0)
        y_np = np.log(np.sum(np.exp(x_np), axis=0))
        self.assertShapeEqual(y_np, y_tf)
        y_tf_np = self.evaluate(y_tf)
        self.assertAllClose(y_tf_np, y_np)

  def testKeepDims(self):
    # [FUNCTION DOC]
    # For float16, float32 and float64 random 5x5 arrays, reduces with logsumexp keeping
    # dimensions; asserts the result rank equals the input rank and values match the numpy
    # reference.
    for dtype in [np.float16, np.float32, np.double]:
      x_np = np.random.rand(5, 5).astype(dtype)
      with test_util.use_gpu():
        y_tf_np = math_ops.reduce_logsumexp(x_np, keepdims=True)
        self.assertEqual(y_tf_np.shape.rank, x_np.ndim)
        y_np = np.log(np.sum(np.exp(x_np), keepdims=True))
        self.assertAllClose(y_tf_np, y_np)

  def testOverflow(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the "overflow" check on the naive numpy computation does not test TensorFlow;
    # it manually raises a runtime warning when the naive result is infinite, then asserts that
    # the raised warning matches (a self-contained sanity check of the numpy reference).
    # For float32 and float64 inputs 1000..1003 it confirms the naive numpy formula overflows, then
    # asserts reduce_logsumexp of the weak tensor equals the numerically stable reference.
    x = [1000, 1001, 1002, 1003]
    for dtype in [np.float32, np.double]:
      x_np = np.array(x, dtype=dtype)
      max_np = np.max(x_np)
      with self.assertRaisesRegex(RuntimeWarning,
                                  "overflow encountered in exp"):
        out = np.log(np.sum(np.exp(x_np)))
        if out == np.inf:
          raise RuntimeWarning("overflow encountered in exp")

      with test_util.use_gpu():
        x_tf = _get_weak_tensor(x_np, shape=x_np.shape)
        y_tf_np = math_ops.reduce_logsumexp(x_tf)
        y_np = np.log(np.sum(np.exp(x_np - max_np))) + max_np
        self.assertAllClose(y_tf_np, y_np)

  def testUnderflow(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: as in the overflow test, the "divide by zero" check exercises only the naive
    # numpy formula (a warning is manually raised when its result is negative infinity), not
    # TensorFlow.
    # For float32 and float64 inputs -1000..-1003 it confirms the naive formula underflows, then
    # asserts reduce_logsumexp of the weak tensor equals the numerically stable reference.
    x = [-1000, -1001, -1002, -1003]
    for dtype in [np.float32, np.double]:
      x_np = np.array(x, dtype=dtype)
      max_np = np.max(x_np)
      with self.assertRaisesRegex(RuntimeWarning,
                                  "divide by zero encountered in log"):
        out = np.log(np.sum(np.exp(x_np)))
        if out == -np.inf:
          raise RuntimeWarning("divide by zero encountered in log")

      with test_util.use_gpu():
        x_tf = _get_weak_tensor(x_np, shape=x_np.shape)
        y_tf_np = math_ops.reduce_logsumexp(x_tf)
        y_np = np.log(np.sum(np.exp(x_np - max_np))) + max_np
        self.assertAllClose(y_tf_np, y_np)

  def testInfinity(self):
    # [FUNCTION DOC]
    # Asserts reduce_logsumexp of negative infinity evaluates to negative infinity.
    with test_util.use_gpu():
      res = math_ops.reduce_logsumexp(-np.inf)
      self.assertEqual(-np.inf, self.evaluate(res))


@test_util.run_all_in_graph_and_eager_modes
class RoundTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case (graph and eager modes) for math_ops.round on weak tensor inputs.

  def testRounding(self):
    # [FUNCTION DOC]
    # Builds values from -5.0 to 4.75 in steps of 0.25 as float32, float64 and int32 numpy arrays,
    # wraps them as weak tensors, rounds with math_ops.round, and asserts closeness (absolute
    # tolerance 1e-2) to numpy rounding.
    x = np.arange(-5.0, 5.0, .25)
    for dtype in [np.float32, np.double, np.int32]:
      x_np = np.array(x, dtype=dtype)
      with test_util.device(use_gpu=True):
        x_tf = _get_weak_tensor(x_np, shape=x_np.shape)
        y_tf = math_ops.round(x_tf)
        y_tf_np = self.evaluate(y_tf)
        y_np = np.round(x_np)
        self.assertAllClose(y_tf_np, y_np, atol=1e-2)


class SignTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for math_ops.sign gradient behavior with complex input.

  def test_complex_sign_gradient(self):
    # [FUNCTION DOC]
    # In eager mode, records the sign of the complex number 1+1j on a gradient tape and asserts the
    # gradient of the sign with respect to the input is close to 0.353553 - 0.353553j.
    with context.eager_mode():
      x = math_ops.complex(1., 1.)
      with backprop.GradientTape() as t:
        t.watch(x)
        y = math_ops.sign(x)
      self.assertAllClose(
          t.gradient(y, x), math_ops.complex(0.353553, -0.353553))


@test_util.run_all_in_graph_and_eager_modes
class ReciprocalNoNanTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case (graph and eager modes) for math_ops.reciprocal_no_nan on weak tensors. The class
  # attribute allowed_dtypes lists the dtypes tested: float32, float64 and complex128.

  allowed_dtypes = [dtypes.float32, dtypes.float64, dtypes.complex128]

  def testBasic(self):
    # [FUNCTION DOC]
    # For each allowed dtype, applies reciprocal_no_nan to the weak tensor 1, 2, 0, 4 and asserts
    # the result equals 1, 0.5, 0 (zero maps to zero, not infinity), 0.25 with matching dtype.
    for dtype in self.allowed_dtypes:
      x = _get_weak_tensor([1.0, 2.0, 0.0, 4.0], dtype=dtype)

      y = math_ops.reciprocal_no_nan(x)

      target = _get_weak_tensor([1.0, 0.5, 0.0, 0.25], dtype=dtype)

      self.assertAllEqual(y, target)
      self.assertEqual(y.dtype.base_dtype, target.dtype.base_dtype)

  def testInverse(self):
    # [FUNCTION DOC]
    # For each allowed dtype, builds a random 5x5x5 weak tensor from the values 0, 1, 2, 4, 5 and
    # asserts that applying reciprocal_no_nan twice returns the original values and dtype.
    for dtype in self.allowed_dtypes:
      x = np.random.choice([0, 1, 2, 4, 5], size=(5, 5, 5))
      x = _get_weak_tensor(x, dtype=dtype)

      y = math_ops.reciprocal_no_nan(math_ops.reciprocal_no_nan(x))

      self.assertAllClose(y, x)
      self.assertEqual(y.dtype.base_dtype, x.dtype.base_dtype)


class EqualityTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for tensor equality and inequality of weak tensors against nothing
  # (null) and float literals.

  @test_util.run_all_in_graph_and_eager_modes
  def testEqualityNone(self):
    # [FUNCTION DOC]
    # Runs in graph and eager modes. Asserts a float32 weak tensor is not equal to the null value
    # in either operand order, that tensor_equals with null is false and tensor_not_equals with
    # null is true.
    x = _get_weak_tensor([1.0, 2.0, 0.0, 4.0], dtype=dtypes.float32)
    self.assertNotEqual(x, None)
    self.assertNotEqual(None, x)
    self.assertFalse(math_ops.tensor_equals(x, None))
    self.assertTrue(math_ops.tensor_not_equals(x, None))

  @parameterized.named_parameters(
      (f"-is_equals={is_equals}-float_literal_type={type(float_literal)}"  # pylint: disable=g-complex-comprehension
       f"-float_literal={float_literal}", is_equals, float_literal)
      for float_literal in [4.6, np.float32(4.6), 4.4, np.float32(4.4)]
      for is_equals in [True, False])
  def testEqualityNoDowncast(self, is_equals, float_literal):
    # [FUNCTION DOC]
    # Named-parameterized over whether equality or inequality is tested and over float literals
    # (python floats and numpy float32, values 4.6 and 4.4). Skips (known type promotion bug)
    # for numpy float32 when TF2 behavior is enabled, or python floats when it is not. Otherwise
    # compares the weak integer 4 with the literal; a type error is accepted as a pass; else the
    # (evaluated) result must equal the opposite of is_equals, i.e. 4 is never equal to the
    # literal, proving the literal is not downcast to an integer.
    if (tf2.enabled() and isinstance(float_literal, np.float32) or
        not tf2.enabled() and isinstance(float_literal, float)):
      # TODO(b/199262800): Remove this skip
      self.skipTest("There is a bug in type promotion.")
    if is_equals:
      op = math_ops.tensor_equals
    else:
      op = math_ops.tensor_not_equals
    x = _get_weak_tensor(4)
    try:
      result = op(x, float_literal)
      if isinstance(result, tensor.Tensor):
        result = self.evaluate(result)
    except TypeError:
      # Throwing a TypeError is OK
      return
    self.assertEqual(result, not is_equals)


@test_util.run_all_in_graph_and_eager_modes
class ErfcinvTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case (graph and eager modes) for math_ops.erfcinv on weak tensors.

  def testErfcinv(self):
    # [FUNCTION DOC]
    # Builds 10000 random float32 values in 0.1 to 1.9 as a weak tensor, applies erfcinv then erfc
    # and asserts the round trip is close to the original values.
    values = _get_weak_tensor(
        np.random.uniform(0.1, 1.9, size=int(1e4)).astype(np.float32)
    )
    approx_id = math_ops.erfc(math_ops.erfcinv(values))
    self.assertAllClose(values, self.evaluate(approx_id))


@test_util.run_all_in_graph_and_eager_modes
class ArgMaxMinTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case (graph and eager modes) for math_ops.argmax and argmin on weak tensors, compared
  # against numpy. Provides helpers for random weak tensor generation and the dtypes to test.

  def _generateRandomWeakTensor(self, dtype, shape):
    # [FUNCTION DOC]
    # HELPER: random input generator for the argmax/argmin tests.
    # Inputs: dtype, a TensorFlow dtype; shape, the output shape. For integer dtypes draws random
    # integers across the full range of the dtype (inclusive); otherwise draws uniform values in
    # -1 to 1. Returns the data wrapped as a weak tensor of that dtype.
    if dtype.is_integer:
      array = np.random.default_rng().integers(
          low=dtype.min, high=dtype.max, size=shape, endpoint=True)
      return _get_weak_tensor(array, dtype=dtype)
    else:
      array = np.random.default_rng().uniform(low=-1.0, high=1.0, size=shape)
      return _get_weak_tensor(array, dtype=dtype)

  def _getValidDtypes(self):
    # [FUNCTION DOC]
    # HELPER: supplies the dtypes iterated by the argmax/argmin tests.
    # Returns a tuple of float32, float64, int32 and int64.
    return (dtypes.float32, dtypes.float64, dtypes.int32, dtypes.int64)

  def testArgMax(self):
    # [FUNCTION DOC]
    # For each valid dtype, creates a random 24x8 weak tensor and asserts argmax along each axis
    # equals the numpy argmax of its evaluated values.
    shape = (24, 8)
    for dtype in self._getValidDtypes():
      tf_values = self._generateRandomWeakTensor(dtype, shape)
      np_values = self.evaluate(tf_values)
      for axis in range(0, len(shape)):
        np_max = np.argmax(np_values, axis=axis)
        tf_max = math_ops.argmax(tf_values, axis=axis)
        self.assertAllEqual(tf_max, np_max)

  def testArgMaxReturnsFirstOccurence(self):
    # [FUNCTION DOC]
    # For each valid dtype, checks argmax along axis 1 of a small matrix with ties returns the
    # first maximal index (as numpy does), and that argmax of a long (193681) all-zero tensor is 0
    # (exercising multithreaded and GPU paths).
    for dtype in self._getValidDtypes():
      values = _get_weak_tensor(
          [[10, 11, 15, 15, 10], [12, 12, 10, 10, 12]], dtype=dtype
      )
      self.assertAllEqual(
          math_ops.argmax(values, axis=1),
          np.argmax(self.evaluate(values), axis=1))

      # Long tensor to ensure works with multithreading/GPU
      values = array_ops.zeros(shape=(193681,), dtype=dtype)
      self.assertAllEqual(math_ops.argmax(values), 0)

  def testArgMaxUint16(self):
    # [FUNCTION DOC]
    # Same as the basic argmax test but requests an unsigned 16-bit output type and compares with
    # the numpy argmax along each axis of a random 24x8 weak tensor for every valid dtype.
    shape = (24, 8)
    for dtype in self._getValidDtypes():
      tf_values = self._generateRandomWeakTensor(dtype, shape)
      np_values = self.evaluate(tf_values)
      for axis in range(0, len(shape)):
        np_max = np.argmax(np_values, axis=axis)
        tf_max = math_ops.argmax(
            tf_values, axis=axis, output_type=dtypes.uint16)
        self.assertAllEqual(tf_max, np_max)

  def testArgMin(self):
    # [FUNCTION DOC]
    # For each valid dtype, creates a random 24x8 weak tensor and asserts argmin along each axis
    # equals the numpy argmin of its evaluated values.
    shape = (24, 8)
    for dtype in self._getValidDtypes():
      tf_values = self._generateRandomWeakTensor(dtype, shape)
      np_values = self.evaluate(tf_values)
      for axis in range(0, len(shape)):
        np_min = np.argmin(np_values, axis=axis)
        tf_min = math_ops.argmin(tf_values, axis=axis)
        self.assertAllEqual(tf_min, np_min)

  def testArgMinReturnsFirstOccurence(self):
    # [FUNCTION DOC]
    # For each valid dtype, checks argmin along axis 1 of a small matrix with ties returns the
    # first minimal index (as numpy does), and that argmin of a long (193681) all-zero tensor is 0.
    for dtype in self._getValidDtypes():
      values = _get_weak_tensor(
          [[10, 11, 15, 15, 10], [12, 12, 10, 10, 12]], dtype=dtype
      )
      self.assertAllEqual(
          math_ops.argmin(values, axis=1),
          np.argmin(self.evaluate(values), axis=1))

      # Long tensor to ensure works with multithreading/GPU
      values = array_ops.zeros(shape=(193681,), dtype=dtype)
      self.assertAllEqual(math_ops.argmin(values), 0)


class CastTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for math_ops.cast interacting with full type information on tensor array flows.

  def testCastWithFullType(self):
    # [FUNCTION DOC]
    # Defines and calls a traced function that casts a tensor array flow to the variant dtype,
    # attaches a full type definition (a product with one array argument) to the cast op, builds a
    # new tensor array from that flow, writes a weak 1 into index 0 and stacks it. Asserts the
    # stacked result evaluates to a list holding 1.

    @def_function.function
    def test_fn():
      # [FUNCTION DOC]
      # HELPER: nested traced function for the enclosing test.
      # Takes no inputs; builds a one-element int32 tensor array, casts its flow to variant, sets
      # full type information on the cast op, rebuilds the array from that flow, writes weak 1 at
      # index 0, and returns the stacked tensor.
      ta = tensor_array_ops.TensorArray(dtypes.int32, size=1)
      h = math_ops.cast(ta.flow, dtypes.variant)

      t = full_type_pb2.FullTypeDef(
          type_id=full_type_pb2.TFT_PRODUCT,
          args=[full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_ARRAY)])
      h.op.experimental_set_type(t)

      ta = tensor_array_ops.TensorArray(dtypes.int32, flow=h)
      ta = ta.write(0, _get_weak_tensor(1))
      return ta.stack()

    self.assertAllEqual(self.evaluate(test_fn()), [1])


@test_util.run_all_in_graph_and_eager_modes
class BinaryOpsTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case (graph and eager modes) for binary operator dispatch of tensors and weak tensors to
  # the reflected operators of right-hand-side objects.

  def testRHSDispatchingAndErrorRaising(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: in the not-implemented scenarios the second statement inside each
    # assertion-raises block can never run, because the first statement already raises; the weak
    # tensor variants of those two scenarios are effectively untested.
    # Picks the expected error (value error in eager mode, type error in graph mode) and message.
    # Defines local classes with reflected addition/modulo returning true/false, raising a type
    # error, returning not-implemented, or defining nothing. Asserts that a regular or weak tensor
    # combined with the first class yields the class's result, and that the error scenarios raise
    # the expected error with the expected message pattern.
    if context.executing_eagerly():
      error = ValueError
      error_message = r"Attempt to convert a value .* with an unsupported type"
    else:
      error = TypeError
      error_message = r"Failed to convert elements of .* to Tensor"

    class RHSReturnsTrue:
      # [CLASS DOC]
      # HELPER: local test double whose reflected addition returns true and reflected modulo
      # returns false, used to verify right-hand-side dispatch.

      def __radd__(self, other):
        # [FUNCTION DOC]
        # Reflected addition; ignores the other operand and returns true.
        return True

      def __rmod__(self, other):
        # [FUNCTION DOC]
        # Reflected modulo; ignores the other operand and returns false.
        return False

    a = array_ops.ones([1], dtype=dtypes.int32) + RHSReturnsTrue()
    self.assertEqual(a, True)

    a = _get_weak_tensor(5, dtype=dtypes.int32) + RHSReturnsTrue()
    self.assertEqual(a, True)

    a = array_ops.ones([1], dtype=dtypes.float32) % RHSReturnsTrue()
    self.assertEqual(a, False)

    a = _get_weak_tensor(5, dtype=dtypes.float32) % RHSReturnsTrue()
    self.assertEqual(a, False)

    class RHSRaisesError:
      # [CLASS DOC]
      # HELPER: local test double whose reflected addition always raises a type error.

      def __radd__(self, other):
        # [FUNCTION DOC]
        # Reflected addition; always raises a type error stating the right-hand side is not
        # implemented.
        raise TypeError("RHS not implemented")

    with self.assertRaisesRegex(error, error_message):
      a = array_ops.ones([1], dtype=dtypes.int32) + RHSRaisesError()
      self.evaluate(a)

    with self.assertRaisesRegex(error, error_message):
      a = _get_weak_tensor([1], dtype=dtypes.int32) + RHSRaisesError()
      self.evaluate(a)

    class RHSReturnsNotImplemented:
      # [CLASS DOC]
      # HELPER: local test double whose reflected addition reports not-implemented.

      def __radd__(self, other):
        # [FUNCTION DOC]
        # Reflected addition; ignores the other operand and returns the not-implemented marker.
        return NotImplemented

    with self.assertRaisesRegex(error, error_message):
      a = array_ops.ones([1], dtype=dtypes.int32) + RHSReturnsNotImplemented()
      self.evaluate(a)

      a = _get_weak_tensor([1], dtype=dtypes.int32) + RHSReturnsNotImplemented()
      self.evaluate(a)

    class RHSNotImplemented:
      # [CLASS DOC]
      # HELPER: local test double with no operator methods at all.
      pass

    with self.assertRaisesRegex(error, error_message):
      a = array_ops.ones([1], dtype=dtypes.int32) + RHSNotImplemented()
      self.evaluate(a)

      a = _get_weak_tensor([1], dtype=dtypes.int32) + RHSNotImplemented()
      self.evaluate(a)


@test_util.run_all_in_graph_and_eager_modes
class ScalarMulTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case (graph and eager modes) for math_ops.scalar_mul with variables and indexed slices.

  def testAcceptsRefs(self):
    # [FUNCTION DOC]
    # Creates a variable holding 10 (resource variable in eager mode, regular variable in graph
    # mode), multiplies it by scalar 3, initializes variables, and asserts the result is 30.
    if context.executing_eagerly():
      var = resource_variable_ops.ResourceVariable(10, name="var")
    else:
      var = variables.Variable(10)
    result = math_ops.scalar_mul(3, var)
    init = variables.global_variables_initializer()
    with test_util.device(use_gpu=True):
      self.evaluate(init)
      self.assertEqual(30, self.evaluate(result))

  def testAcceptsIndexedSlices(self):
    # [FUNCTION DOC]
    # Builds an indexed-slices value (three rows at indices 0, 2, 5), multiplies by scalar -3, and
    # asserts the resulting values are scaled and the indices are unchanged.
    values = constant_op.constant([2, 3, 5, 7, 0, -1], shape=[3, 2])
    indices = constant_op.constant([0, 2, 5])
    # Test that patched scalar_mul works with IndexedSlices.
    x = math_ops.scalar_mul(-3, indexed_slices.IndexedSlices(values, indices))
    with test_util.device(use_gpu=True):
      self.assertAllEqual(
          self.evaluate(x.values), [[-6, -9], [-15, -21], [0, 3]]
      )
      self.assertAllEqual(self.evaluate(x.indices), [0, 2, 5])


class ComparisonOps(parameterized.TestCase, test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for equal, maximum and minimum with mixed python, numpy and tensor operands of
  # different dtypes (checks promotion and results).

  def test_math_equal(self):
    # [FUNCTION DOC]
    # Asserts math_ops.equal is true when comparing python 1, numpy 1, or a float32 constant 1,
    # against an equal-valued constant of the same or a different dtype.
    self.assertAllEqual(math_ops.equal(1, constant_op.constant(1)), True)
    self.assertAllEqual(
        math_ops.equal(np.int_(1), constant_op.constant(1)), True
    )
    self.assertAllEqual(
        math_ops.equal(
            constant_op.constant(1, dtypes.float32),
            constant_op.constant(1, dtypes.int32),
        ),
        True,
    )

  def test_math_maximum(self):
    # [FUNCTION DOC]
    # Checks math_ops.maximum and gen_math_ops.maximum with mixed operand kinds (python int,
    # numpy int, float32 and int32 constants); asserts the expected maximum values, including that
    # mixing a numpy int with a float32 constant gives a float32 result.
    # Test math_ops.maximum.
    self.assertAllEqual(math_ops.maximum(1, constant_op.constant(2)), 2)
    self.assertAllEqual(
        math_ops.maximum(np.int_(1), constant_op.constant(1.5, dtypes.float32)),
        np.array(1.5, np.float32),
    )
    self.assertAllEqual(
        math_ops.maximum(
            constant_op.constant(5, dtypes.float32),
            constant_op.constant(1, dtypes.int32),
        ),
        5,
    )

    # Test gen_math_ops.maximum.
    self.assertAllEqual(gen_math_ops.maximum(1, constant_op.constant(2)), 2)
    self.assertAllEqual(
        gen_math_ops.maximum(
            np.int_(1), constant_op.constant(1.5, dtypes.float32)
        ),
        np.array(1.5, np.float32),
    )
    self.assertAllEqual(
        gen_math_ops.maximum(
            constant_op.constant(5, dtypes.float32),
            constant_op.constant(1, dtypes.int32),
        ),
        5,
    )

  def test_math_minimum(self):
    # [FUNCTION DOC]
    # Checks math_ops.minimum and gen_math_ops.minimum with mixed operand kinds (python int,
    # numpy int, float32 and int32 constants); asserts the expected minimum values.
    # Test math_ops.minimum.
    self.assertAllEqual(math_ops.minimum(1, constant_op.constant(2)), 1)
    self.assertAllEqual(
        math_ops.minimum(np.int_(1), constant_op.constant(1.1, dtypes.float32)),
        1,
    )
    self.assertAllEqual(
        math_ops.minimum(
            constant_op.constant(5, dtypes.float32),
            constant_op.constant(-1, dtypes.int32),
        ),
        -1,
    )

    # Test gen_math_ops.minimum.
    self.assertAllEqual(gen_math_ops.minimum(1, constant_op.constant(2)), 1)
    self.assertAllEqual(
        gen_math_ops.minimum(
            np.int_(1), constant_op.constant(1.1, dtypes.float32)
        ),
        1,
    )
    self.assertAllEqual(
        gen_math_ops.minimum(
            constant_op.constant(5, dtypes.float32),
            constant_op.constant(-1, dtypes.int32),
        ),
        -1,
    )


allowed_var_op_input_combinations = [
    (dtypes.uint8, 10),
    (dtypes.uint8, "weak_i64"),
    (dtypes.uint8, dtypes.uint8),
    (dtypes.uint16, 10),
    (dtypes.uint16, "weak_i64"),
    (dtypes.uint16, dtypes.uint8),
    (dtypes.uint16, dtypes.uint16),
    (dtypes.uint32, 10),
    (dtypes.uint32, "weak_i64"),
    (dtypes.uint32, dtypes.uint8),
    (dtypes.uint32, dtypes.uint32),
    (dtypes.uint64, 10),
    (dtypes.uint64, "weak_i64"),
    (dtypes.uint64, dtypes.uint32),
    (dtypes.uint64, dtypes.uint64),
    (dtypes.int8, 10),
    (dtypes.int8, "weak_i64"),
    (dtypes.int8, dtypes.int8),
    (dtypes.int16, 10),
    (dtypes.int16, "weak_i64"),
    (dtypes.int16, dtypes.uint8),
    (dtypes.int16, dtypes.int8),
    (dtypes.int16, dtypes.int16),
    (dtypes.int32, 10),
    (dtypes.int32, "weak_i64"),
    (dtypes.int32, dtypes.uint16),
    (dtypes.int32, dtypes.int16),
    (dtypes.int32, dtypes.int32),
    (dtypes.int64, 10),
    (dtypes.int64, "weak_i64"),
    (dtypes.int64, dtypes.uint32),
    (dtypes.int64, dtypes.int32),
    (dtypes.int64, dtypes.int64),
    (dtypes.bfloat16, 10),
    (dtypes.bfloat16, "weak_i64"),
    (dtypes.bfloat16, 1.0),
    (dtypes.bfloat16, "weak_f64"),
    (dtypes.bfloat16, dtypes.int32),
    (dtypes.bfloat16, dtypes.bfloat16),
    (dtypes.float16, 10),
    (dtypes.float16, "weak_i64"),
    (dtypes.float16, 1.0),
    (dtypes.float16, "weak_f64"),
    (dtypes.float16, dtypes.int32),
    (dtypes.float16, dtypes.float16),
    (dtypes.float32, 10),
    (dtypes.float32, "weak_i64"),
    (dtypes.float32, 1.0),
    (dtypes.float32, "weak_f64"),
    (dtypes.float32, dtypes.int32),
    (dtypes.float32, dtypes.float32),
    (dtypes.float64, 10),
    (dtypes.float64, "weak_i64"),
    (dtypes.float64, 1.0),
    (dtypes.float64, "weak_f64"),
    (dtypes.float64, dtypes.int32),
    (dtypes.float64, dtypes.float64),
    (dtypes.complex64, 10),
    (dtypes.complex64, "weak_i64"),
    (dtypes.complex64, 1.0),
    (dtypes.complex64, "weak_f64"),
    (dtypes.complex64, 1.0 + 2.0j),
    (dtypes.complex64, "weak_c128"),
    (dtypes.complex64, dtypes.int32),
    (dtypes.complex64, dtypes.complex64),
    (dtypes.complex128, 10),
    (dtypes.complex128, "weak_i64"),
    (dtypes.complex128, 1.0),
    (dtypes.complex128, "weak_f64"),
    (dtypes.complex128, 1.0 + 2.0j),
    (dtypes.complex128, "weak_c128"),
    (dtypes.complex128, dtypes.int32),
    (dtypes.complex128, dtypes.float64),
    (dtypes.complex128, dtypes.complex128),
]


disallowed_var_op_input_combinations = [
    (dtypes.uint8, 1.0),
    (dtypes.uint8, "weak_f64"),
    (dtypes.uint8, dtypes.int8),
    (dtypes.uint8, dtypes.uint16),
    (dtypes.uint16, 1.0),
    (dtypes.uint16, "weak_f64"),
    (dtypes.uint16, dtypes.int8),
    (dtypes.uint16, dtypes.uint32),
    (dtypes.uint32, 1.0),
    (dtypes.uint32, "weak_f64"),
    (dtypes.uint32, dtypes.int32),
    (dtypes.uint32, dtypes.uint64),
    (dtypes.uint64, 1.0),
    (dtypes.uint64, "weak_f64"),
    (dtypes.uint64, dtypes.int8),
    (dtypes.uint64, dtypes.float16),
    (dtypes.int8, 1.0),
    (dtypes.int8, "weak_f64"),
    (dtypes.int8, dtypes.int16),
    (dtypes.int8, dtypes.float32),
    (dtypes.int8, dtypes.complex64),
    (dtypes.int16, 1.0),
    (dtypes.int16, "weak_f64"),
    (dtypes.int16, dtypes.int32),
    (dtypes.int16, dtypes.float32),
    (dtypes.int16, dtypes.complex64),
    (dtypes.int32, 1.0),
    (dtypes.int32, "weak_f64"),
    (dtypes.int32, dtypes.int64),
    (dtypes.int32, dtypes.float16),
    (dtypes.int32, dtypes.complex64),
    (dtypes.int64, 1.0),
    (dtypes.int64, "weak_f64"),
    (dtypes.int64, dtypes.uint64),
    (dtypes.int64, dtypes.float16),
    (dtypes.int64, dtypes.complex64),
    (dtypes.bfloat16, 1.0 + 2.0j),
    (dtypes.bfloat16, "weak_c128"),
    (dtypes.bfloat16, dtypes.float16),
    (dtypes.bfloat16, dtypes.float32),
    (dtypes.float16, 1.0 + 2.0j),
    (dtypes.float16, "weak_c128"),
    (dtypes.float16, dtypes.bfloat16),
    (dtypes.float16, dtypes.float32),
    (dtypes.float64, 1.0 + 2.0j),
    (dtypes.float64, "weak_c128"),
    (dtypes.float64, dtypes.complex64),
    (dtypes.complex64, dtypes.float64),
    (dtypes.complex64, dtypes.complex128),
]


def _weak_tensor_from_str(s):
  # [FUNCTION DOC]
  # HELPER: converts the string names used in the variable-update dtype tables to weak tensors.
  # Input: s, one of the strings naming a weak int64, weak float64 or weak complex128 scalar.
  # Returns a weak tensor holding 1, 1.0 or 1+2j with the corresponding dtype. Raises a value
  # error for any other string.
  if s == "weak_i64":
    return _get_weak_tensor(1, dtype=dtypes.int64)
  elif s == "weak_f64":
    return _get_weak_tensor(1.0, dtype=dtypes.float64)
  elif s == "weak_c128":
    return _get_weak_tensor(1.0 + 2.0j, dtype=dtypes.complex128)
  else:
    raise ValueError(f"Unsupported str: {s}")


@test_util.run_all_in_graph_and_eager_modes
class VariableInplaceOpsTest(
    parameterized.TestCase, test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Parameterized test case (graph and eager modes) for in-place update methods of resource
  # variables (assign, add-assign, subtract-assign) with weak tensors and regular tensors of
  # various dtypes, driven by the module-level allowed and disallowed combination tables.

  @parameterized.parameters(
      itertools.product(
          allowed_var_op_input_combinations,
          ("assign", "assign_add", "assign_sub")))
  def testAllowedDtypes(self, v_dtype_and_delta, op):
    # [FUNCTION DOC]
    # Parameterized over allowed (variable dtype, delta) pairs and the three update methods. The
    # delta is a python number, a named weak tensor, or a constant of the given dtype. Creates a
    # variable of the dtype holding 10, applies the update, and asserts the result keeps the
    # variable's dtype.
    v_dtype, delta = v_dtype_and_delta
    if isinstance(delta, dtypes.DType):
      delta = constant_op.constant(1, delta)
    elif isinstance(delta, str):
      delta = _weak_tensor_from_str(delta)

    var = resource_variable_ops.ResourceVariable(10, dtype=v_dtype)
    result = getattr(var, op)(delta)
    with test_util.device(use_gpu=True):
      self.assertEqual(result.dtype, v_dtype)

  @parameterized.parameters(
      itertools.product(
          disallowed_var_op_input_combinations,
          ("assign", "assign_add", "assign_sub")))
  def testDisallowedDtypes(self, v_dtype_and_delta, op):
    # [FUNCTION DOC]
    # Parameterized over disallowed (variable dtype, delta) pairs and the three update methods.
    # Builds the delta as in the allowed test, creates a variable holding 10, and asserts the
    # update raises a type error.
    v_dtype, delta = v_dtype_and_delta
    if isinstance(delta, dtypes.DType):
      delta = constant_op.constant(1, delta)
    elif isinstance(delta, str):
      delta = _weak_tensor_from_str(delta)

    var = resource_variable_ops.ResourceVariable(10, dtype=v_dtype)
    with self.assertRaises(TypeError):
      _ = getattr(var, op)(delta)


@test_util.run_all_in_graph_and_eager_modes
class DivAndModTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case (graph and eager modes) for floor and truncate division and modulo on int32 and
  # int64 weak tensors at the edges of the integer range. Holds numpy reference implementations
  # that avoid overflow crashes, and an edge-case data generator.

  def numpySafeFloorDivInt(self, x, y):
    # [FUNCTION DOC]
    # HELPER: numpy reference for integer floor division in the division tests.
    # Inputs: x and y, integer numpy arrays. Returns floor division of x by y, except that entries
    # where x is the minimum integer value and y is -1 are set to the minimum value (the expected
    # overflow behaviour, instead of numpy's 0). Modifies only the result array.
    z = x // y
    # Numpy produces 0 for INT_MIN/-1, but we expect an overflow to INT_MIN
    # so that (INT_MIN/-1) + (INT_MIN % -1) = INT_MIN + 0 = INT_MIN.
    z[(x == np.iinfo(x.dtype).min) & (y == -1)] = np.iinfo(x.dtype).min
    return z

  def numpySafeFloorModInt(self, x, y):
    # [FUNCTION DOC]
    # HELPER: numpy reference for integer floor modulo in the division tests.
    # Computes x minus the safe floor quotient times y (avoids numpy's floating point exception for
    # the minimum value modulo -1). Inputs are integer numpy arrays; returns the remainder array.
    # Numpy crashes with a FPE for INT_MIN % -1.
    z = self.numpySafeFloorDivInt(x, y)
    return x - z * y

  def numpySafeTruncateDivInt(self, x, y):
    # [FUNCTION DOC]
    # HELPER: numpy reference for integer truncating division in the division tests.
    # Starts from the safe floor quotient and adds one where the remainder is nonzero and the
    # operands have opposite signs (rounding toward zero). Inputs are integer numpy arrays;
    # returns the quotient array.
    z = self.numpySafeFloorDivInt(x, y)
    # Round up if non-zero remainder and inputs have opposite signs.
    z[(x != z * y) & ((x < 0) != (y < 0))] += 1
    return z

  def numpySafeTruncateModInt(self, x, y):
    # [FUNCTION DOC]
    # HELPER: numpy reference for integer truncating modulo in the division tests.
    # Computes x minus the safe truncating quotient times y. Inputs are integer numpy arrays;
    # returns the remainder array.
    # Numpy crashes with a FPE for INT_MIN % -1.
    z = self.numpySafeTruncateDivInt(x, y)
    return x - z * y

  def intEdgeTestData(self, dtype):
    # [FUNCTION DOC]
    # HELPER: edge-case data generator for the division tests.
    # Input: dtype, a numpy integer dtype. Returns a pair of 4x4 arrays (numerators, divisors)
    # built from the minimum value, minimum plus one, -1, 1 and the maximum value of that dtype.
    # The pair (minimum, -1) is deliberately avoided in the numerators' row of divisor -1 by using
    # minimum plus one in one row, to avoid signed overflow in the operation under test.
    """Edge-case test data for integer types."""
    # INT_MIN/-1 will produce signed-integer overflow, so we instead test
    # (INT_MIN + 1) / -1.
    nums = np.array(
        [
            [np.iinfo(dtype).min, -1, 1, np.iinfo(dtype).max],
            [np.iinfo(dtype).min + 1, -1, 1, np.iinfo(dtype).max],
            [np.iinfo(dtype).min, -1, 1, np.iinfo(dtype).max],
            [np.iinfo(dtype).min, -1, 1, np.iinfo(dtype).max],
        ],
        dtype=dtype,
    )
    divs = np.array(
        [
            [
                np.iinfo(dtype).min,
                np.iinfo(dtype).min,
                np.iinfo(dtype).min,
                np.iinfo(dtype).min,
            ],
            [-1, -1, -1, -1],
            [1, 1, 1, 1],
            [
                np.iinfo(dtype).max,
                np.iinfo(dtype).max,
                np.iinfo(dtype).max,
                np.iinfo(dtype).max,
            ],
        ],
        dtype=dtype,
    )
    return nums, divs

  @test_util.disable_asan("Expected signed integer overflow.")
  @test_util.disable_ubsan("Expected signed integer overflow.")
  def testFloorDivModIntEdges(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: after computing floor modulo, the type assertion checks the floor-division
    # result again instead of the floor-modulo result, so the modulo result type is never checked.
    # For int32 and int64, builds edge-case numerators and divisors as weak tensors, asserts
    # floor_div and floormod match the numpy reference and the division result is a WeakTensor,
    # and asserts the identity quotient times divisor plus remainder equals the numerator
    # (broadcast) with a WeakTensor result. Disabled under address and undefined-behaviour
    # sanitizers because signed overflow is expected.
    for dtype in [np.int32, np.int64]:
      x, y = self.intEdgeTestData(dtype)
      x_weak, y_weak = _get_weak_tensor(x), _get_weak_tensor(y)
      tf_floor_div = math_ops.floor_div(x_weak, y_weak)
      np_floor_div = self.numpySafeFloorDivInt(x, y)
      self.assertIsInstance(tf_floor_div, WeakTensor)
      self.assertAllEqual(tf_floor_div, np_floor_div)

      tf_floor_mod = math_ops.floormod(x_weak, y_weak)
      np_floor_mod = self.numpySafeFloorModInt(x, y)
      self.assertIsInstance(tf_floor_div, WeakTensor)
      self.assertAllEqual(tf_floor_mod, np_floor_mod)
      z = math_ops.add(math_ops.multiply(tf_floor_div, y_weak), tf_floor_mod)
      # x = floor_div(x, y) * y + floor_mod(x, y)
      self.assertIsInstance(z, WeakTensor)
      self.assertAllEqual(z, np.broadcast_to(x, z.shape))

  @test_util.disable_asan("Expected signed integer overflow.")
  @test_util.disable_ubsan("Expected signed integer overflow.")
  def testTruncateDivModIntEdges(self):
    # [FUNCTION DOC]
    # For int32 and int64, builds edge-case numerators and divisors as weak tensors, asserts
    # truncatediv and truncatemod match the numpy references and are WeakTensor results, and
    # asserts quotient times divisor plus remainder equals the numerator (broadcast) with a
    # WeakTensor result. Disabled under address and undefined-behaviour sanitizers because signed
    # overflow is expected.
    for dtype in [np.int32, np.int64]:
      x, y = self.intEdgeTestData(dtype)
      x_weak, y_weak = _get_weak_tensor(x), _get_weak_tensor(y)
      tf_truncate_div = math_ops.truncatediv(x_weak, y_weak)
      np_truncate_div = self.numpySafeTruncateDivInt(x, y)
      self.assertIsInstance(tf_truncate_div, WeakTensor)
      self.assertAllEqual(tf_truncate_div, np_truncate_div)

      tf_truncate_mod = math_ops.truncatemod(x_weak, y_weak)
      np_truncate_mod = self.numpySafeTruncateModInt(x, y)
      self.assertIsInstance(tf_truncate_mod, WeakTensor)
      self.assertAllEqual(tf_truncate_mod, np_truncate_mod)
      z = math_ops.add(
          math_ops.multiply(tf_truncate_div, y_weak), tf_truncate_mod
      )
      self.assertIsInstance(z, WeakTensor)
      # x = truncatediv(x, y) * y + truncatemod(x, y)
      self.assertAllEqual(z, np.broadcast_to(x, z.shape))


@test_util.run_all_in_graph_and_eager_modes
class DivNoNanTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case (graph and eager modes) for math_ops.div_no_nan on weak tensors. The
  # class attribute lists supported dtypes: int32, int64, float32, float64, complex128.
  _SUPPORTED_DTYPES = [
      dtypes.int32,
      dtypes.int64,
      dtypes.float32,
      dtypes.float64,
      dtypes.complex128,
  ]

  @parameterized.parameters(*_SUPPORTED_DTYPES)
  def testBasic(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over supported dtypes. Builds column numerators and row divisors (unsigned,
    # signed integer or fractional ranges; the divisors must contain zero) as weak tensors, takes
    # the reference as true division with zero wherever the divisor is zero, and asserts
    # div_no_nan returns a WeakTensor close to the reference according to the dtype.
    if dtype.is_unsigned:
      nums = np.arange(0, 120, 3).reshape(40, 1)
      divs = np.arange(0, 48, 4).reshape(1, 12)
    elif dtype.is_integer:
      nums = np.arange(-120, 120, 3).reshape(80, 1)
      divs = np.arange(-48, 48, 4).reshape(1, 24)
    else:
      nums = np.arange(-10, 10, 0.25).reshape(80, 1)
      divs = np.arange(-3, 3, 0.25).reshape(1, 24)
    assert 0 in divs, "Bad test set-up"

    tf_nums = _get_weak_tensor(nums, dtype=dtype)
    tf_divs = _get_weak_tensor(divs, dtype=dtype)

    # Use tf versions for expected value to ensure inputs are identical
    np_nums = self.evaluate(tf_nums)
    np_divs = self.evaluate(tf_divs)
    np_result = np.true_divide(np_nums, np_divs)
    np_result[:, np_divs[0] == 0] = 0

    with test_util.use_gpu():
      tf_result = math_ops.div_no_nan(tf_nums, tf_divs)
      self.assertIsInstance(tf_result, WeakTensor)
      self.assertAllCloseAccordingToType(tf_result, np_result)

  @parameterized.product(
      type_x=_SUPPORTED_DTYPES + [float, int],
      type_y=_SUPPORTED_DTYPES + [float, int],
  )
  def testSameSupportedTypesAsDivide(self, type_x, type_y):
    # [FUNCTION DOC]
    # Parameterized over the product of supported dtypes plus python float and int for both
    # operands. Builds the value one in each kind and calls divide; if divide raises a type error,
    # asserts div_no_nan raises one too; otherwise asserts both results are WeakTensors with the
    # same dtype and equal values.
    def one(type_):
      # [FUNCTION DOC]
      # HELPER: builds a unit operand for the enclosing test. Input: type_, either the python
      # int type, the python float type, or a TensorFlow dtype. Returns python 1, python 1.0, or a
      # weak tensor holding 1 of that dtype respectively.
      if type_ is int:
        return 1
      elif type_ is float:
        return 1.0
      else:
        return _get_weak_tensor(1, dtype=type_)

    x = one(type_x)
    y = one(type_y)

    divide_raises = False
    try:
      divide_result = math_ops.divide(x, y)
    except TypeError:
      divide_raises = True

    if divide_raises:
      with self.assertRaises(TypeError):
        _ = math_ops.div_no_nan(x, y)
    else:
      divide_no_nan_result = math_ops.div_no_nan(x, y)
      self.assertIsInstance(divide_no_nan_result, WeakTensor)
      self.assertIsInstance(divide_result, WeakTensor)
      self.assertEqual(divide_no_nan_result.dtype, divide_result.dtype)
      self.assertAllEqual(divide_no_nan_result, divide_result)

  @parameterized.parameters(
      (dtypes.float32),
      (dtypes.float64),
      (dtypes.complex128),
  )
  def testSmall(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over float32, float64 and complex128. Divides weak zeros by tiny divisors whose
    # squared magnitude underflows, and asserts the result is exactly zero and a WeakTensor.
    # Choose values whose squared magnitude underflows to zero/subnormal.
    zero = _get_weak_tensor([0, 0, 0, 0], dtype=dtype)
    divs = _get_weak_tensor([1e-25, -1e-20, 1e-165, -1e-160], dtype=dtype)
    tf_result = math_ops.div_no_nan(zero, divs)

    # Results should always be exactly zero.
    self.assertAllEqual(tf_result, zero)
    self.assertIsInstance(tf_result, WeakTensor)

  @parameterized.parameters(
      (dtypes.float32),
      (dtypes.float64),
      (dtypes.complex128),
  )
  def testNonFiniteInNumerator(self, dtype):
    # [FUNCTION DOC]
    # Parameterized over float32, float64 and complex128. With numerators NaN, +inf and -inf,
    # asserts dividing by zero gives zeros and dividing by one leaves the numerators unchanged,
    # with WeakTensor results in both cases.
    nums = _get_weak_tensor([np.nan, np.inf, -np.inf], dtype=dtype)
    zeros = _get_weak_tensor([0, 0, 0], dtype=dtype)
    ones = _get_weak_tensor([1, 1, 1], dtype=dtype)
    with test_util.use_gpu():
      tf_result_zeros = math_ops.div_no_nan(nums, zeros)
      self.assertAllEqual([0, 0, 0], tf_result_zeros)
      self.assertIsInstance(tf_result_zeros, WeakTensor)
      tf_result_ones = math_ops.div_no_nan(nums, ones)
      self.assertAllEqual(nums / ones, tf_result_ones)
      self.assertIsInstance(tf_result_ones, WeakTensor)


if __name__ == "__main__":
  ops.set_dtype_conversion_mode("all")
  googletest.main()
