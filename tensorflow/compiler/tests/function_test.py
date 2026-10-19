# [MODULE DOC]
# XLA tests that legacy graph-mode TensorFlow functions (defined with the function decorator)
# are executed correctly on the XLA test device: simple, nested, multiple return values and
# compile-time-constant arguments. One test is disabled. Runs the test main when executed.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Test cases for Tensorflow functions."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest


class FunctionTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case verifying execution of defined graph functions on the test device.
  # Stateless apart from base-class fixtures.

  def testFunction(self):
    # [FUNCTION DOC]
    # Defines a two-input function computing a plus twice b, calls it inside the test device
    # scope on two 2x2 float arrays, and checks the result against the same computation done
    # directly in numpy.
    """Executes a simple TensorFlow function."""

    def APlus2B(a, b):
      # [FUNCTION DOC]
      # HELPER: reference computation. Returns the first argument plus twice the second;
      # works on both arrays and tensors.
      return a + b * 2

    aval = np.array([4, 3, 2, 1]).reshape([2, 2]).astype(np.float32)
    bval = np.array([5, 6, 7, 8]).reshape([2, 2]).astype(np.float32)
    expected = APlus2B(aval, bval)

    with self.session():

      @function.Defun(dtypes.float32, dtypes.float32)
      def Foo(a, b):
        # [FUNCTION DOC]
        # Graph function of two float32 inputs; returns the plus-twice-b helper result.
        return APlus2B(a, b)

      a = constant_op.constant(aval, name="a")
      b = constant_op.constant(bval, name="b")
      with self.test_scope():
        call_f = Foo(a, b)
      result = self.evaluate(call_f)
    self.assertAllClose(result, expected, rtol=1e-3)

  def testNestedFunctions(self):
    # [FUNCTION DOC]
    # Same as the simple function test but the reference computation calls another helper that
    # doubles its input; checks the defined function's output matches the numpy result.
    """Executes two nested TensorFlow functions."""

    def TimesTwo(x):
      # [FUNCTION DOC]
      # HELPER: returns its input multiplied by two.
      return x * 2

    def APlus2B(a, b):
      # [FUNCTION DOC]
      # HELPER: reference computation; returns a plus the doubled b.
      return a + TimesTwo(b)

    aval = np.array([4, 3, 2, 1]).reshape([2, 2]).astype(np.float32)
    bval = np.array([4, 3, 2, 1]).reshape([2, 2]).astype(np.float32)
    expected = APlus2B(aval, bval)

    with self.session():

      @function.Defun(dtypes.float32, dtypes.float32)
      def Foo(a, b):
        # [FUNCTION DOC]
        # Graph function of two float32 inputs; returns the nested helper result.
        return APlus2B(a, b)

      a = constant_op.constant(aval, name="a")
      b = constant_op.constant(bval, name="b")
      with self.test_scope():
        call_g = Foo(a, b)
      result = self.evaluate(call_g)
    self.assertAllClose(result, expected, rtol=1e-3)

  def testFunctionMultipleRetvals(self):
    # [FUNCTION DOC]
    # Defines a graph function returning both the sum and difference of two 2x2 float arrays,
    # calls it on the test device and compares both outputs with numpy results.
    """Executes a function with multiple return values."""

    # This function will run on the XLA device
    def Func(a, b):
      # [FUNCTION DOC]
      # HELPER: reference computation; returns a pair of sum and difference of its inputs.
      return a + b, a - b

    aval = np.array([4, 3, 2, 1]).reshape([2, 2]).astype(np.float32)
    bval = np.array([5, 6, 7, 8]).reshape([2, 2]).astype(np.float32)
    expected = Func(aval, bval)

    with self.session():

      @function.Defun(dtypes.float32, dtypes.float32)
      def Foo(a, b):
        # [FUNCTION DOC]
        # Graph function of two float32 inputs; returns the pair produced by the helper.
        return Func(a, b)

      a = constant_op.constant(aval, name="a")
      b = constant_op.constant(bval, name="b")
      with self.test_scope():
        call_f = Foo(a, b)
      result = self.evaluate(call_f)
    self.assertAllClose(result, expected, rtol=1e-3)

  def testCompileTimeConstantsInDefun(self):
    # [FUNCTION DOC]
    # Defines a function that slices a float tensor using begin and size arguments, which XLA
    # needs as compile-time constants. Feeds placeholders (ones of shape 1x4x4x1, begin zeros,
    # size 1,2,2,1) and asserts the result is a 1x2x2x1 tensor of ones.
    """Tests that XLA handles compile-time constants in defuns."""
    with self.session() as sess:

      @function.Defun(dtypes.float32, dtypes.int32, dtypes.int32)
      def Foo(a, c, d):
        # [FUNCTION DOC]
        # Graph function taking a float tensor and two int vectors; returns the slice of the
        # tensor starting at the first vector with extent of the second.
        # c and d must be known at compile time
        x = array_ops.slice(a, c, d)
        return x

      a = array_ops.placeholder(dtypes.float32)
      c = array_ops.placeholder(dtypes.int32, shape=[4])
      d = array_ops.placeholder(dtypes.int32, shape=[4])
      with self.test_scope():
        call_f = Foo(a, c, d)
      result = sess.run(call_f, feed_dict={
          a: np.ones([1, 4, 4, 1]),
          c: [0, 0, 0, 0],
          d: [1, 2, 2, 1]})

    self.assertAllEqual(np.ones([1, 2, 2, 1]), result)

  # TODO(b/36139787): Re-enable this test when noinline works again.
  def DISABLED_testFunctionsNoInline(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: disabled (name prefix prevents discovery) pending a bug about no-inline
    # functions; it does not currently run.
    # Would define a no-inline doubling function called from another function, feed placeholders
    # on the test device and compare the output with a plus twice b in numpy.

    @function.Defun(dtypes.float32, noinline=True)
    def TimesTwo(x):
      # [FUNCTION DOC]
      # Graph function (not inlined) returning its float input doubled.
      return x * 2

    @function.Defun(dtypes.float32, dtypes.float32)
    def APlus2B(a, b):
      # [FUNCTION DOC]
      # Graph function returning the first input plus the doubled second input.
      return a + TimesTwo(b)

    aval = np.array([4, 3, 2, 1]).reshape([2, 2]).astype(np.float32)
    bval = np.array([4, 3, 2, 1]).reshape([2, 2]).astype(np.float32)
    expected = aval + bval * 2

    with self.session() as sess:
      with self.test_scope():
        a = array_ops.placeholder(dtypes.float32, name="a")
        b = array_ops.placeholder(dtypes.float32, name="b")
        call = APlus2B(a, b)
      result = sess.run(call, {a: aval, b: bval})
    self.assertAllClose(result, expected, rtol=1e-3)


if __name__ == "__main__":
  googletest.main()
