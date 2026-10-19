# [MODULE DOC]
# Tests for the low-level native eager execution entry points: the fast-path execute function
# (correctness against the regular ops, gradient tape recording, list-valued inputs, invalid
# inputs and attributes, very large output counts), the slow-path execute function, graph
# resource variable fallback, default dtype handling of op definitions, conversion from the array
# interface, hiding of the internal fallback exception, and large integer attributes. Many tests
# also check that no new tensors or garbage are created. Runs the test main entry point when
# executed directly.
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
"""Tests for low-level eager execution primitives."""

import sys
import traceback

import numpy as np

from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops


@test_util.with_eager_op_as_function
class Tests(test.TestCase):
  # [CLASS DOC]
  # Test case for the native eager execution primitives, decorated so ops also run as functions
  # when that mode is enabled. No shared fixtures.

  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastpathExecute_MatMulCorrectResponse(self):
    # [FUNCTION DOC]
    # Calls the fast-path executor for matrix multiply on 2x2 and 100x784 random inputs (second
    # with transpose of the right operand) and asserts results are close to the normal matmul.
    a_2_by_2 = random_ops.random_uniform((2, 2))
    b_2_by_2 = random_ops.random_uniform((2, 2))

    a_100_by_784 = random_ops.random_uniform((100, 784))
    b_100_by_784 = random_ops.random_uniform((100, 784))

    ctx = context.context()
    ctx.ensure_initialized()

    self.assertAllClose(
        math_ops.matmul(a_2_by_2, b_2_by_2),
        pywrap_tfe.TFE_Py_FastPathExecute(ctx, "MatMul", None,
                                          a_2_by_2, b_2_by_2, "transpose_a",
                                          False, "transpose_b", False))
    self.assertAllClose(
        math_ops.matmul(a_100_by_784, b_100_by_784, transpose_b=True),
        pywrap_tfe.TFE_Py_FastPathExecute(ctx, "MatMul", None,
                                          a_100_by_784, b_100_by_784,
                                          "transpose_a", False, "transpose_b",
                                          True))

  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastpathExecute_ResourceVariableMatMulCorrectResponse(self):
    # [FUNCTION DOC]
    # Runs fast-path matmul with a resource variable as both operands and with an equal constant
    # tensor, and asserts the two results are equal.
    ctx = context.context()
    ctx.ensure_initialized()

    a_2_by_2 = constant_op.constant(1.0, shape=[2, 2])
    m = resource_variable_ops.ResourceVariable(a_2_by_2)
    x = pywrap_tfe.TFE_Py_FastPathExecute(ctx, "MatMul", None, m,
                                          m, "transpose_a", False,
                                          "transpose_b", False)
    y = pywrap_tfe.TFE_Py_FastPathExecute(ctx, "MatMul", None,
                                          a_2_by_2, a_2_by_2, "transpose_a",
                                          False, "transpose_b", False)

    self.assertAllEqual(x, y)

  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastpathExecute_TapeWrite(self):
    # [FUNCTION DOC]
    # Runs fast-path matmul of a watched all-ones 2x2 tensor with itself under a persistent
    # gradient tape and asserts the gradient equals a 2x2 tensor of fours (the op was recorded).
    ctx = context.context()
    ctx.ensure_initialized()

    with backprop.GradientTape(persistent=True) as tape:
      a_2_by_2 = constant_op.constant(1.0, shape=[2, 2])
      tape.watch(a_2_by_2)
      z = pywrap_tfe.TFE_Py_FastPathExecute(ctx, "MatMul", None,
                                            a_2_by_2, a_2_by_2, "transpose_a",
                                            False, "transpose_b", False)
    dz_dy = tape.gradient(z, [a_2_by_2])[0]
    self.assertAllEqual(dz_dy.numpy(),
                        constant_op.constant(4.0, shape=[2, 2]).numpy())

  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastpathExecute_ResourceVariableTapeWrite(self):
    # [FUNCTION DOC]
    # Same as the tape-write test but with a watched resource variable as both operands; asserts
    # the gradient with respect to the variable equals a 2x2 tensor of fours.
    ctx = context.context()
    ctx.ensure_initialized()

    with backprop.GradientTape(persistent=True) as tape:
      a_2_by_2 = constant_op.constant(1.0, shape=[2, 2])
      m = resource_variable_ops.ResourceVariable(a_2_by_2)
      tape.watch(m)
      z = pywrap_tfe.TFE_Py_FastPathExecute(ctx, "MatMul", None, m,
                                            m, "transpose_a", False,
                                            "transpose_b", False)
    dz_dy = tape.gradient(z, [m])[0]
    self.assertAllEqual(dz_dy.numpy(),
                        constant_op.constant(4.0, shape=[2, 2]).numpy())

  # Tests homogeneous list op
  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastpathExecute_AddNCorrectResponse(self):
    # [FUNCTION DOC]
    # Fast-path executes add-n (homogeneous list input) on two random matrices and asserts the
    # result is close to the regular add-n.
    ctx = context.context()
    ctx.ensure_initialized()

    a_2_by_2 = random_ops.random_uniform((2, 2))
    b_2_by_2 = random_ops.random_uniform((2, 2))

    self.assertAllClose(
        math_ops.add_n([a_2_by_2, b_2_by_2]),
        pywrap_tfe.TFE_Py_FastPathExecute(ctx, "AddN", None,
                                          [a_2_by_2, b_2_by_2]))

  # Tests homogeneous list op
  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastpathExecute_AddNTapeWrite(self):
    # [FUNCTION DOC]
    # Under a persistent tape watching two tensors, runs add-n via the fast path and via the
    # regular API, and asserts the gradients with respect to the first tensor are equal.
    ctx = context.context()
    ctx.ensure_initialized()

    a_2_by_2 = random_ops.random_uniform((2, 2))
    b_2_by_2 = random_ops.random_uniform((2, 2))

    with backprop.GradientTape(persistent=True) as tape:
      tape.watch(a_2_by_2)
      tape.watch(b_2_by_2)
      z1 = pywrap_tfe.TFE_Py_FastPathExecute(ctx, "AddN", None,
                                             [a_2_by_2, b_2_by_2])
      z2 = math_ops.add_n([a_2_by_2, b_2_by_2])
    dz1_dy = tape.gradient(z1, [a_2_by_2])[0]
    dz2_dy = tape.gradient(z2, [a_2_by_2])[0]
    self.assertAllEqual(dz1_dy.numpy(), dz2_dy.numpy())

  # Tests heterogeneous list op
  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastpathExecute_IdentityNCorrectResponse(self):
    # [FUNCTION DOC]
    # Fast-path executes identity-n (heterogeneous list input) on two random matrices and asserts
    # the result is close to the regular identity-n.
    ctx = context.context()
    ctx.ensure_initialized()

    a_2_by_2 = random_ops.random_uniform((2, 2))
    b_2_by_2 = random_ops.random_uniform((2, 2))

    self.assertAllClose(
        array_ops.identity_n([a_2_by_2, b_2_by_2]),
        pywrap_tfe.TFE_Py_FastPathExecute(ctx, "IdentityN", None,
                                          [a_2_by_2, b_2_by_2]))

  # Tests heterogeneous list op
  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastpathExecute_IdentityNTapeWrite(self):
    # [FUNCTION DOC]
    # Under a persistent tape, runs identity-n via the fast path and the regular API, and asserts
    # the gradients of the first outputs with respect to the first input are equal.
    ctx = context.context()
    ctx.ensure_initialized()

    a_2_by_2 = random_ops.random_uniform((2, 2))
    b_2_by_2 = random_ops.random_uniform((2, 2))

    with backprop.GradientTape(persistent=True) as tape:
      tape.watch(a_2_by_2)
      tape.watch(b_2_by_2)
      z1 = pywrap_tfe.TFE_Py_FastPathExecute(ctx, "IdentityN",
                                             None, [a_2_by_2, b_2_by_2])
      z2 = array_ops.identity_n([a_2_by_2, b_2_by_2])
    dz1_dy = tape.gradient(z1[0], [a_2_by_2])[0]
    dz2_dy = tape.gradient(z2[0], [a_2_by_2])[0]
    self.assertAllEqual(dz1_dy.numpy(), dz2_dy.numpy())

  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastpathExecute_InvalidInputs(self):
    # [FUNCTION DOC]
    # Asserts eager mode, then calls the fast-path executor with bad arguments and checks the
    # errors: too few base parameters raises a value error mentioning at least 3 items; missing
    # inputs raises a value error (expected at least 4, got 3); a non-string op name raises a
    # type error.
    a_2_by_2 = random_ops.random_uniform((2, 2))
    ctx = context.context()
    ctx.ensure_initialized()

    assert ctx.executing_eagerly(
    ), "The prototype doesn't contain C code for graph construction"
    ctx_handle = ctx._handle  # pylint: disable=protected-access

    # Not enough base params
    with self.assertRaisesRegex(ValueError,
                                "at least 3 items in the input tuple"):
      pywrap_tfe.TFE_Py_FastPathExecute(ctx, "Identity")

    # Not enough inputs
    with self.assertRaisesRegex(ValueError, "Expected to be at least 4, was 3"):
      pywrap_tfe.TFE_Py_FastPathExecute(ctx, "Identity", None)

    # Bad type
    with self.assertRaisesRegex(TypeError, "expected a string for op_name"):
      pywrap_tfe.TFE_Py_FastPathExecute(ctx, ctx_handle, None,
                                        a_2_by_2)

  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastPathExecute_InvalidAttributes(self):
    # [FUNCTION DOC]
    # Fast-path executes split with an invalid (negative) split count and asserts the internal
    # fallback exception is raised.
    split_dim = constant_op.constant(0, dtype=dtypes.int32)
    value = constant_op.constant([0, 1, 2, 3], dtype=dtypes.float32)
    ctx = context.context()
    ctx.ensure_initialized()

    with self.assertRaises(core._FallbackException):
      pywrap_tfe.TFE_Py_FastPathExecute(ctx, "Split", None,
                                        split_dim, value, "num_split", -1)

  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testFastPathExecute_VeryLargeOutputs(self):
    # [FUNCTION DOC]
    # Fast-path executes split with an enormous split count (a trillion) and asserts a value error
    # saying the number of outputs is too big.
    split_dim = constant_op.constant(0, dtype=dtypes.int32)
    value = constant_op.constant([0, 1, 2, 3], dtype=dtypes.float32)
    ctx = context.context()
    ctx.ensure_initialized()

    with self.assertRaisesRegex(ValueError, "Number of outputs is too big"):
      pywrap_tfe.TFE_Py_FastPathExecute(ctx, "Split", None, split_dim, value,
                                        "num_split", 1000000000000)

  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testSlowPathExecute_VeryLargeOutputs(self):
    # [FUNCTION DOC]
    # With a plain list as input, the fast path with an enormous split count raises the fallback
    # exception. Then the slow-path executor with a tensor input and the same count is asserted
    # to raise a value error saying the number of outputs is too big.
    split_dim = constant_op.constant(0, dtype=dtypes.int32)
    value = [0, 1, 2, 3]
    ctx = context.context()
    ctx.ensure_initialized()

    with self.assertRaises(core._FallbackException):
      pywrap_tfe.TFE_Py_FastPathExecute(ctx, "Split", None, split_dim, value,
                                        "num_split", 1000000000000)

    value = constant_op.constant(value)
    attrs = ("num_split", 1000000000000, "T", value.dtype.as_datatype_enum)
    with self.assertRaisesRegex(ValueError, "Number of outputs is too big"):
      pywrap_tfe.TFE_Py_Execute(ctx._handle, None, "Split", [split_dim, value],
                                attrs, 1000000000000)

  @test_util.assert_no_new_tensors
  @test_util.assert_no_garbage_created
  def testInvalidNumOutputs(self):
    # [FUNCTION DOC]
    # Asserts that the split API with a negative or zero number of splits raises an error whose
    # message says the count attribute must be at least the minimum of 1 (or the negative number
    # attribute message, depending on the code path).
    with self.assertRaisesRegex(
        Exception, r"Value for number_attr\(\) -1 < 0 \[Op:Split\]|"
        r"Value for attr 'num_split' of -1 must be at least minimum 1"):
      array_ops.split(value=[1, 2, 3], num_or_size_splits=-1)

    with self.assertRaisesRegex(
        Exception,
        r"Value for attr 'num_split' of 0 must be at least minimum 1"):
      array_ops.split(value=[1, 2, 3], num_or_size_splits=0)

  def testIsFunction(self):
    # [FUNCTION DOC]
    # Asserts the context reports no function for an unknown name, then traces a function and
    # asserts the context reports the concrete function's name as existing.
    ctx = context.context()
    self.assertFalse(ctx.has_function("not_a_function"))

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Traced function returning the float one.
      return 1.

    self.assertTrue(ctx.has_function(f.get_concrete_function().name))

  def testEagerExecute_InvalidType(self):
    # [FUNCTION DOC]
    # Regression test for a reported issue. In a graph, stacks a resource variable (not a list)
    # and asserts a type error saying a list was expected for the values argument.
    # Test case for GitHub issue 26879.
    with ops.Graph().as_default():
      a_2_by_2 = constant_op.constant(1.0, shape=[2, 2])
      m = resource_variable_ops.ResourceVariable(a_2_by_2)
      with self.assertRaisesRegex(TypeError,
                                  "Expected list for 'values' argument"):
        _ = array_ops_stack.stack(m, axis=1)

  def testGraphResourceVariableRaisesFallback(self):
    # [FUNCTION DOC]
    # Creates a resource variable inside a graph, then in eager mode calls fast-path matmul on it
    # and asserts the internal fallback exception is raised.
    with ops.Graph().as_default():
      a_2_by_2 = constant_op.constant(1.0, shape=[2, 2])
      m = resource_variable_ops.ResourceVariable(a_2_by_2)
    ctx = context.context()
    ctx.ensure_initialized()
    with self.assertRaises(core._FallbackException):
      pywrap_tfe.TFE_Py_FastPathExecute(ctx, "MatMul", None, m, m,
                                        "transpose_a", False, "transpose_b",
                                        False)

  def testOpDefDefaultType(self):
    # [FUNCTION DOC]
    # Uses an op whose type attribute has a default. With a random unsigned 16-bit array, asserts
    # the fast path and slow (eager fallback) path give the same dtype, and that the graph path
    # matches. Asserts a traced function taking the array as an argument gives a different dtype
    # (arrays are converted to tensors before tracing), while a function capturing the array
    # matches the fast path.
    im = np.random.randint(  # pylint: disable=too-many-function-args
        low=0,
        high=65535,
        size=100,
        dtype=np.uint16).reshape(10, 10, 1)

    context.ensure_initialized()

    fastpath_dtype = test_ops.dtype_with_default_op(im).numpy()
    slowpath_dtype = test_ops.dtype_with_default_op_eager_fallback(
        im, None, context.context()).numpy()
    # Ensure the fastpath and slowpath eager paths work.
    self.assertEqual(fastpath_dtype, slowpath_dtype)

    with ops.Graph().as_default(), self.cached_session():
      graph_dtype_symbolic = test_ops.dtype_with_default_op(im)

      graph_dtype = self.evaluate(graph_dtype_symbolic)
    # Ensure the eager path matches the graph path.
    self.assertEqual(fastpath_dtype, graph_dtype)

    # Unfortunately, as of now, this doesn't work as expected on def_functions,
    # since we convert the numpy arrays to tensors pre-tracing (which won't get
    # overriddent by the default type).
    @def_function.function
    def func(im):
      # [FUNCTION DOC]
      # Traced function applying the default-type op to its argument and returning the result.
      return test_ops.dtype_with_default_op(im)

    function_dtype = func(im).numpy()
    self.assertNotEqual(fastpath_dtype, function_dtype)

    # Captures are OK, since they don't go through the conversion path.
    @def_function.function
    def func_captured():
      # [FUNCTION DOC]
      # Traced function applying the default-type op to the captured array and returning the
      # result.
      return test_ops.dtype_with_default_op(im)

    function_dtype = func_captured().numpy()
    self.assertEqual(fastpath_dtype, function_dtype)

  def testConvertFromArrayInterface(self):
    # [FUNCTION DOC]
    # Defines a class exposing the array interface, builds an eager tensor from an instance on the
    # context device, and asserts the tensor equals the underlying random array.
    context.ensure_initialized()
    ctx = context.context()

    class MyArrayClass(object):
      # [CLASS DOC]
      # HELPER: minimal array-like test object holding 16 random numbers and exposing them through
      # the NumPy array conversion protocol.

      def __init__(self):
        # [FUNCTION DOC]
        # Creates the object with an array of 16 random floats stored as an attribute.
        self.array = np.random.random(16)

      def __array__(self):
        # [FUNCTION DOC]
        # Returns the stored array for NumPy conversion.
        return self.array

    a = MyArrayClass()
    t = ops.EagerTensor(a, device=ctx.device_name, dtype=None)
    self.assertAllEqual(t, a)

    # TODO(b/147830189): Converting from EagerTensor should work.
    # _ = ops.EagerTensor(t, device=ctx.device_name, dtype=None)

    # TODO(b/147828820): Converting with tensors should work.
    # _ = ops.EagerTensor([[t]], device=ctx.device_name, dtype=None)

  def testFallbackErrorNotVisibleWhenFallbackMethodRaises(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the formatted exception text is only assigned inside the handler; if the matmul
    # call raised nothing, the later assertion would fail with an unbound-variable error rather
    # than a clear test failure. Also the inputs are lists built by repetition, making shapes
    # incompatible on purpose.
    # Calls matmul on incompatible shapes, catches the invalid-argument error, formats its full
    # traceback text, and asserts the text does not mention the internal fallback exception.
    ctx = context.context()
    ctx.ensure_initialized()

    try:
      math_ops.mat_mul([[1., 1.] * 2], [[1., 1.] * 3])
    except errors.InvalidArgumentError:
      etype, value, tb = sys.exc_info()
      full_exception_text = " ".join(
          traceback.format_exception(etype, value, tb))

    self.assertNotRegex(full_exception_text, "_FallbackException")

  def testIntAttrThatDoesNotFitIn32Bits(self):
    # [FUNCTION DOC]
    # Regression test: fast-path executes a random-integer op with an integer seed attribute of
    # 2 to the power 50 (beyond 32 bits). Passes if no exception is raised (no assertions).
    # Tests bug where int attributes >= 2**31 raised an exception on platforms
    # where sizeof(long) = 32 bits.
    ctx = context.context()
    ctx.ensure_initialized()
    shape = constant_op.constant([10])
    minval = constant_op.constant(0)
    maxval = constant_op.constant(10)
    seed = 2**50
    pywrap_tfe.TFE_Py_FastPathExecute(ctx, "RandomUniformInt", None,
                                      shape, minval, maxval,
                                      "seed", seed)


if __name__ == "__main__":
  test.main()
