# [MODULE DOC]
# Test module for basic element-wise math ops on a GPU device: binary ops compared with CPU
# results, unary ops compared with numpy, broadcast-gradient arguments and division gradients,
# and a multi-threaded multi-session GPU consistency test. No import-time side effects; running
# as a script starts the test runner.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Functional tests for basic component wise operations using a GPU device."""

import itertools
import threading

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.gen_array_ops import broadcast_gradient_args
from tensorflow.python.platform import test


class GPUBinaryOpsTest(test.TestCase):
  # [CLASS DOC]
  # Test case comparing binary element-wise math ops (add, subtract, multiply, divide, floor
  # divide, power) computed on the GPU against the same ops on the CPU, for float32 and float64,
  # with and without broadcasting.

  def _compareGPU(self, x, y, np_func, tf_func):
    # [FUNCTION DOC]
    # HELPER: shared comparator for the binary-op tests.
    # KNOWN ISSUE: the np_func parameter is accepted but never used; the comparison is between
    # the CPU and GPU results of the TensorFlow function, not against numpy.
    # Converts x and y to tensors, applies tf_func in a default (GPU-capable) session and again
    # in a CPU-only session, evaluates both and asserts the results are close. Returns nothing.
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_gpu = self.evaluate(out)

    with self.cached_session(use_gpu=False):
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_cpu = self.evaluate(out)

    self.assertAllClose(tf_cpu, tf_gpu)

  def testFloatBasic(self):
    # [FUNCTION DOC]
    # Compares CPU and GPU for add, subtract, multiply, true divide, floor divide and power on
    # same-shaped float32 arrays of shape one by three by five (second operand offset by 0.1 for
    # the division cases to avoid dividing by zero).
    x = np.linspace(-5, 20, 15).reshape(1, 3, 5).astype(np.float32)  # pylint: disable=too-many-function-args
    y = np.linspace(20, -5, 15).reshape(1, 3, 5).astype(np.float32)  # pylint: disable=too-many-function-args
    self._compareGPU(x, y, np.add, math_ops.add)
    self._compareGPU(x, y, np.subtract, math_ops.subtract)
    self._compareGPU(x, y, np.multiply, math_ops.multiply)
    self._compareGPU(x, y + 0.1, np.true_divide, math_ops.truediv)
    self._compareGPU(x, y + 0.1, np.floor_divide, math_ops.floordiv)
    self._compareGPU(x, y, np.power, math_ops.pow)

  def testFloatWithBCast(self):
    # [FUNCTION DOC]
    # Compares CPU and GPU for add, subtract, multiply and true divide on float32 operands of
    # shapes three by five and two by three by five, exercising broadcasting.
    x = np.linspace(-5, 20, 15).reshape(3, 5).astype(np.float32)
    y = np.linspace(20, -5, 30).reshape(2, 3, 5).astype(np.float32)  # pylint: disable=too-many-function-args
    self._compareGPU(x, y, np.add, math_ops.add)
    self._compareGPU(x, y, np.subtract, math_ops.subtract)
    self._compareGPU(x, y, np.multiply, math_ops.multiply)
    self._compareGPU(x, y + 0.1, np.true_divide, math_ops.truediv)

  def testDoubleBasic(self):
    # [FUNCTION DOC]
    # Same-shape float64 CPU versus GPU comparison for add, subtract, multiply and true divide.
    # Unlike the float32 test, floor divide and power are not covered.
    x = np.linspace(-5, 20, 15).reshape(1, 3, 5).astype(np.float64)  # pylint: disable=too-many-function-args
    y = np.linspace(20, -5, 15).reshape(1, 3, 5).astype(np.float64)  # pylint: disable=too-many-function-args
    self._compareGPU(x, y, np.add, math_ops.add)
    self._compareGPU(x, y, np.subtract, math_ops.subtract)
    self._compareGPU(x, y, np.multiply, math_ops.multiply)
    self._compareGPU(x, y + 0.1, np.true_divide, math_ops.truediv)

  def testDoubleWithBCast(self):
    # [FUNCTION DOC]
    # Broadcasting float64 CPU versus GPU comparison (three by five with two by three by five)
    # for add, subtract, multiply and true divide.
    x = np.linspace(-5, 20, 15).reshape(3, 5).astype(np.float64)
    y = np.linspace(20, -5, 30).reshape(2, 3, 5).astype(np.float64)  # pylint: disable=too-many-function-args
    self._compareGPU(x, y, np.add, math_ops.add)
    self._compareGPU(x, y, np.subtract, math_ops.subtract)
    self._compareGPU(x, y, np.multiply, math_ops.multiply)
    self._compareGPU(x, y + 0.1, np.true_divide, math_ops.truediv)


class MathBuiltinUnaryTest(test.TestCase):
  # [CLASS DOC]
  # Test case verifying that unary math ops (trig, hyperbolic, exp/log, rounding, sqrt, etc.)
  # run on the GPU produce results matching numpy, plus a floor-divide check.

  def _compare(self, x, np_func, tf_func, use_gpu):
    # [FUNCTION DOC]
    # HELPER: applies the numpy function to x for the expected value, applies the TensorFlow
    # function to a tensor made from x inside a session (on GPU if use_gpu is true), and asserts
    # the results are close. Returns nothing. The session handle variable is unused.
    np_out = np_func(x)
    with self.cached_session(use_gpu=use_gpu) as sess:
      inx = ops.convert_to_tensor(x)
      ofunc = tf_func(inx)
      tf_out = self.evaluate(ofunc)
    self.assertAllClose(np_out, tf_out)

  def _inv(self, x):
    # [FUNCTION DOC]
    # HELPER: numpy-side reciprocal used to build the reference for reciprocal square root.
    # Returns one divided by x element-wise.
    return 1.0 / x

  def _rsqrt(self, x):
    # [FUNCTION DOC]
    # HELPER: numpy-side reference for reciprocal square root; returns the reciprocal of the
    # element-wise square root of x.
    return self._inv(np.sqrt(x))

  def _testDtype(self, dtype, use_gpu):
    # [FUNCTION DOC]
    # HELPER: builds a small array of the given numpy dtype with values from minus three
    # quarters to one half, plus a copy shifted up by two for ops needing inputs above one
    # (inverse hyperbolic cosine), and compares about twenty-two unary ops against numpy via the
    # comparison helper, on GPU when use_gpu is true. Returns nothing.
    data = (np.arange(-3, 3) / 4.).reshape([1, 3, 2]).astype(dtype)
    data_gt_1 = data + 2 # for x > 1
    self._compare(data, np.abs, math_ops.abs, use_gpu)
    self._compare(data, np.arccos, math_ops.acos, use_gpu)
    self._compare(data, np.arcsin, math_ops.asin, use_gpu)
    self._compare(data, np.arcsinh, math_ops.asinh, use_gpu)
    self._compare(data_gt_1, np.arccosh, math_ops.acosh, use_gpu)
    self._compare(data, np.arctan, math_ops.atan, use_gpu)
    self._compare(data, np.ceil, math_ops.ceil, use_gpu)
    self._compare(data, np.cos, math_ops.cos, use_gpu)
    self._compare(data, np.cosh, math_ops.cosh, use_gpu)
    self._compare(data, np.exp, math_ops.exp, use_gpu)
    self._compare(data, np.floor, math_ops.floor, use_gpu)
    self._compare(data, np.log, math_ops.log, use_gpu)
    self._compare(data, np.log1p, math_ops.log1p, use_gpu)
    self._compare(data, np.negative, math_ops.negative, use_gpu)
    self._compare(data, self._rsqrt, math_ops.rsqrt, use_gpu)
    self._compare(data, np.sin, math_ops.sin, use_gpu)
    self._compare(data, np.sinh, math_ops.sinh, use_gpu)
    self._compare(data, np.sqrt, math_ops.sqrt, use_gpu)
    self._compare(data, np.square, math_ops.square, use_gpu)
    self._compare(data, np.tan, math_ops.tan, use_gpu)
    self._compare(data, np.tanh, math_ops.tanh, use_gpu)
    self._compare(data, np.arctanh, math_ops.atanh, use_gpu)

  def testTypes(self):
    # [FUNCTION DOC]
    # Runs the unary-op comparison on the GPU for float32 only (the loop covers a single type).
    for dtype in [np.float32]:
      self._testDtype(dtype, use_gpu=True)

  def testFloorDivide(self):
    # [FUNCTION DOC]
    # Builds two identical float32 arrays, divides one by the other plus 0.1 using the division
    # operator, takes the floor with the floor op, and asserts the result is close to numpy's
    # floor divide.
    x = (1 + np.linspace(0, 5, np.prod([1, 3, 2]))).astype(np.float32).reshape(
        [1, 3, 2])
    y = (1 + np.linspace(0, 5, np.prod([1, 3, 2]))).astype(np.float32).reshape(
        [1, 3, 2])

    np_out = np.floor_divide(x, y + 0.1)

    with self.session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y + 0.1)
      ofunc = inx / iny
      out_func2 = math_ops.floor(ofunc)
      tf_out = self.evaluate(out_func2)

    self.assertAllClose(np_out, tf_out)


class BroadcastSimpleTest(test.TestCase):
  # [CLASS DOC]
  # Test case for broadcast-gradient reduction indices and for gradients and GPU results of
  # division ops under broadcasting-compatible shapes. Holds a per-dtype gradient tolerance.

  def _GetGradientArgs(self, xs, ys):
    # [FUNCTION DOC]
    # HELPER: evaluates the broadcast gradient args op for two shapes (lists of dimension sizes)
    # and returns the pair of reduction-index arrays for the first and second operand.
    return self.evaluate(broadcast_gradient_args(xs, ys))

  def testBroadcast(self):
    # [FUNCTION DOC]
    # Asks for gradient reduction indices for shapes two by three by five and one; asserts the
    # first has none and the second reduces over all three axes.
    r0, r1 = self._GetGradientArgs([2, 3, 5], [1])
    self.assertAllEqual(r0, [])
    self.assertAllEqual(r1, [0, 1, 2])

  _GRAD_TOL = {dtypes.float32: 1e-3}

  def _compareGpu(self, x, y, np_func, tf_func):
    # [FUNCTION DOC]
    # HELPER: computes the expected answer with the numpy function, evaluates the TensorFlow
    # function on tensors made from x and y in a default session, and asserts the values are
    # close and the static shape matches. Returns nothing. A TODO notes the gradient checker
    # does not yet work on GPU.
    np_ans = np_func(x, y)
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      iny = ops.convert_to_tensor(y)
      out = tf_func(inx, iny)
      tf_gpu = self.evaluate(out)
    self.assertAllClose(np_ans, tf_gpu)
    self.assertShapeEqual(np_ans, out)
    # TODO(zhifengc/ke): make gradient checker work on GPU.

  def testGradient(self):
    # [FUNCTION DOC]
    # Numerically checks the gradient of true division (scaled by 1.1) with respect to each
    # operand, using the v2 gradient checker, asserting the max error stays below the float32
    # tolerance. Then compares true divide and floor divide results against numpy via the GPU
    # comparison helper.
    x1 = (1 + np.linspace(0, 5, np.prod([1, 3, 2]))).astype(np.float32).reshape(
        [1, 3, 2])
    x2 = (1 + np.linspace(0, 5, np.prod([1, 3, 2]))).astype(np.float32).reshape(
        [1, 3, 2])

    def div_x1(x1):
      # [FUNCTION DOC]
      # HELPER: function of the first operand used by the gradient checker; divides it by the
      # enclosing second operand and scales by 1.1 cast to the input dtype.
      return math_ops.truediv(x1, x2) * math_ops.cast(1.1, dtype=x1.dtype)

    def div_x2(x2):
      # [FUNCTION DOC]
      # HELPER: function of the second operand used by the gradient checker; divides the
      # enclosing first operand by it and scales by 1.1 cast to the input dtype.
      return math_ops.truediv(x1, x2) * math_ops.cast(1.1, dtype=x2.dtype)

    with self.cached_session():
      err = gradient_checker_v2.max_error(*gradient_checker_v2.compute_gradient(
          div_x1, [x1]))
      self.assertLess(err, self._GRAD_TOL[dtypes.as_dtype(x1.dtype)])

      err = gradient_checker_v2.max_error(*gradient_checker_v2.compute_gradient(
          div_x2, [x2]))
      self.assertLess(err, self._GRAD_TOL[dtypes.as_dtype(x2.dtype)])

    self._compareGpu(x1, x2, np.true_divide, math_ops.truediv)
    self._compareGpu(x1, x2 + 0.1, np.floor_divide, math_ops.floordiv)


class GpuMultiSessionMemoryTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case checking that several graph-mode sessions running concurrently in threads on the
  # same GPU produce identical numerical results (no memory corruption between sessions).
  """Tests concurrent sessions executing on the same GPU."""

  def _run_session(self, session, results):
    # [FUNCTION DOC]
    # HELPER: thread body for the concurrency test. Using the given session (entered as a
    # context), creates a variable and two random matrices pinned to the first GPU, builds a
    # chain of multiplications and a transpose product, initializes variables, and runs the
    # final op up to 500 times. Adds the first element of each result to the given set and stops
    # early once the set holds more than one distinct value. Mutates the results set; returns
    # nothing. Sets the global random seed to 1 inside the thread.
    n_iterations = 500
    with session as s:
      data = variables.Variable(1.0)
      with ops.device('/device:GPU:0'):
        random_seed.set_random_seed(1)
        matrix1 = variables.Variable(
            random_ops.truncated_normal([1024, 1]), name='matrix1')
        matrix2 = variables.Variable(
            random_ops.truncated_normal([1, 1024]), name='matrix2')
        x1 = math_ops.multiply(data, matrix1, name='x1')
        x3 = math_ops.matmul(x1, math_ops.matmul(matrix2, matrix1))
        x4 = math_ops.matmul(array_ops.transpose(x3), x3, name='x4')
        s.run(variables.global_variables_initializer())

        for _ in range(n_iterations):
          value = s.run(x4)
          results.add(value.flat[0])
          if len(results) != 1:
            break

  @test_util.run_v1_only('b/126596827 needs graph mode in multiple threads')
  def testConcurrentSessions(self):
    # [FUNCTION DOC]
    # Graph-mode only test (skipped otherwise, bug reference about multi-threaded graph mode).
    # Starts four threads, each with its own graph and GPU session and its own result set, runs
    # the session helper in each, joins them, and asserts the union of all observed values has
    # exactly one element, i.e. every session got the same number.
    n_threads = 4
    threads = []
    results = []
    for _ in range(n_threads):
      session = self.session(graph=ops.Graph(), use_gpu=True)
      results.append(set())
      args = (session, results[-1])
      threads.append(threading.Thread(target=self._run_session, args=args))

    for thread in threads:
      thread.start()
    for thread in threads:
      thread.join()

    flat_results = set(itertools.chain(*results))
    self.assertEqual(1,
                     len(flat_results),
                     'Expected single value, got %r' % flat_results)


if __name__ == '__main__':
  test.main()
