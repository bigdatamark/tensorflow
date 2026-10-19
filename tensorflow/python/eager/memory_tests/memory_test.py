# [MODULE DOC]
# Memory-leak tests for eager execution. Each leak test repeatedly runs a small workload
# (creating anonymous variables, tracing and calling a compiled function, building nested
# compiled functions with second-order gradients) through the shared leak-assertion helper and
# fails if process memory grows beyond a threshold; these tests are skipped when the optional
# memory profiler package is missing. One further test uses the python-object leak decorator to
# check that nested compiled functions are released and that first and second derivatives are
# correct. Running as a script starts the test main. No import-time side effects.
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
"""Tests for memory leaks in eager execution.

It is possible that this test suite will eventually become flaky due to taking
too long to run (since the tests iterate many times), but for now they are
helpful for finding memory leaks since not all PyObject leaks are found by
introspection (test_util decorators). Please be careful adding new tests here.
"""

from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.eager.memory_tests import memory_test_util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients as gradient_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.variables import Variable


class MemoryTest(test.TestCase):
  # [CLASS DOC]
  # Test case grouping eager memory-leak tests; no shared fixtures. Uses the leak assertion
  # helper from the memory test utility module.

  def testMemoryLeakAnonymousVariable(self):
    # [FUNCTION DOC]
    # Test: creating and immediately deleting an anonymous variable of zeros (32 by 100 floats)
    # 10000 times must not increase memory by 10 megabytes or more. Skipped if the memory
    # profiler is unavailable.
    if not memory_test_util.memory_profiler_is_available():
      self.skipTest("memory_profiler required to run this test")

    def f():
      # [FUNCTION DOC]
      # Workload: creates a variable from a zeros tensor and deletes the only local reference.
      inputs = Variable(array_ops.zeros([32, 100], dtypes.float32))
      del inputs

    memory_test_util.assert_no_leak(
        f, num_iters=10000, increase_threshold_absolute_mb=10)

  def testMemoryLeakInFunction(self):
    # [FUNCTION DOC]
    # Test: defining a compiled function and calling it once, repeated 1000 times, must not
    # increase memory by 30 megabytes or more. Skipped if the memory profiler is unavailable.
    if not memory_test_util.memory_profiler_is_available():
      self.skipTest("memory_profiler required to run this test")

    def f():
      # [FUNCTION DOC]
      # Workload: defines a fresh compiled function (forcing a new trace each time) and calls
      # it on the constant 42; returns nothing.

      @def_function.function
      def graph(x):
        # [FUNCTION DOC]
        # Compiled function computing x times x plus x.
        return x * x + x

      graph(constant_op.constant(42))

    memory_test_util.assert_no_leak(
        f, num_iters=1000, increase_threshold_absolute_mb=30)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testNestedFunctionsDeleted(self):
    # [FUNCTION DOC]
    # Test (decorated to fail if new Python objects survive the run, in eager mode): a compiled
    # function containing a nested compiled sine function is evaluated at 1 under two nested
    # gradient tapes. Asserts the value is about 0.8415, the first derivative about 0.5403 and
    # the second derivative about -0.8415.

    @def_function.function
    def f(x):
      # [FUNCTION DOC]
      # Outer compiled function: defines a nested compiled sine function on each trace and
      # returns its result for x.

      @def_function.function
      def my_sin(x):
        # [FUNCTION DOC]
        # Nested compiled function computing the sine of x.
        return math_ops.sin(x)

      return my_sin(x)

    x = constant_op.constant(1.)
    with backprop.GradientTape() as t1:
      t1.watch(x)
      with backprop.GradientTape() as t2:
        t2.watch(x)
        y = f(x)
      dy_dx = t2.gradient(y, x)
    dy2_dx2 = t1.gradient(dy_dx, x)

    self.assertAllClose(0.84147096, y.numpy())  # sin(1.)
    self.assertAllClose(0.54030230, dy_dx.numpy())  # cos(1.)
    self.assertAllClose(-0.84147096, dy2_dx2.numpy())  # -sin(1.)

  def testMemoryLeakInGlobalGradientRegistry(self):
    # [FUNCTION DOC]
    # Regression test for a past leak in the global gradient function registry. Repeats 300
    # times a workload that builds a compiled function containing a nested compiled cube
    # function, deletes the nested function's name, and takes second-order graph gradients; memory
    # must not grow by 50 megabytes or more. Skipped if the memory profiler is unavailable.
    # Past leak: b/139819011

    if not memory_test_util.memory_profiler_is_available():
      self.skipTest("memory_profiler required to run this test")

    def f():
      # [FUNCTION DOC]
      # Workload: defines and calls a compiled function (autograph off) on 1.5 and returns the
      # numeric value of its second-order gradient result.

      @def_function.function(autograph=False)
      def graph(x):
        # [FUNCTION DOC]
        # Outer compiled function: calls a nested cube function on x, deletes the nested
        # function's local name (the gradient must still work), and returns the gradient of
        # the gradient of the result with respect to x.

        @def_function.function(autograph=False)
        def cubed(a):
          # [FUNCTION DOC]
          # Nested compiled function computing a cubed.
          return a * a * a

        y = cubed(x)
        # To ensure deleting the function does not affect the gradient
        # computation.
        del cubed
        return gradient_ops.gradients(gradient_ops.gradients(y, x), x)

      return graph(constant_op.constant(1.5))[0].numpy()

    memory_test_util.assert_no_leak(
        f, num_iters=300, increase_threshold_absolute_mb=50)


if __name__ == "__main__":
  test.main()
