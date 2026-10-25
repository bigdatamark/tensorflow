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
# [MODULE DOC]
# Tests for the generated low-level training ops: dense and sparse gradient descent, Adagrad,
# Ftrl and Adam updates, checked against numpy reference computations, plus a threaded
# regression test for a copy-on-read race. Runs via the standard test main entry point.
"""Tests for tensorflow.ops.gen_training_ops."""

import itertools
import threading

import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework.test_util import TensorFlowTestCase
# Import resource_variable_ops for the variables-to-tensor implicit conversion.
from tensorflow.python.ops import gen_training_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops  # pylint: disable=unused-import
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


class TrainingOpsTest(TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the raw generated training ops (gradient descent, Adagrad, Ftrl, sparse variants,
  # Adam and resource sparse Adagrad v2) compared against numpy reference formulas across
  # data types, devices and index types. Most helpers re-run setUp to reset state per case.

  def _toType(self, dtype):
    # [FUNCTION DOC]
    # HELPER: maps a numpy dtype (16/32/64-bit float, 32/64-bit int) to the framework dtype.
    # Any other input trips an assertion failure. Returns the framework dtype.
    if dtype == np.float16:
      return dtypes.float16
    elif dtype == np.float32:
      return dtypes.float32
    elif dtype == np.float64:
      return dtypes.float64
    elif dtype == np.int32:
      return dtypes.int32
    elif dtype == np.int64:
      return dtypes.int64
    else:
      assert False, (dtype)

  def _testTypes(self, x, alpha, delta, use_gpu=None):
    # [FUNCTION DOC]
    # HELPER: checks the gradient descent op for one input set.
    # Inputs: x initial values, alpha step size, delta gradient, use_gpu device choice. Resets
    # test state, creates a variable from x, runs the apply-gradient-descent op, and asserts the
    # output has the variable's shape and equals x minus alpha times delta (type-aware tolerance).
    self.setUp()
    with self.session(use_gpu=use_gpu):
      var = variable_v1.VariableV1(x)
      self.evaluate(variables.global_variables_initializer())
      self.assertAllCloseAccordingToType(x, self.evaluate(var))
      apply_sgd = gen_training_ops.apply_gradient_descent(var, alpha, delta)
      out = self.evaluate(apply_sgd)
      self.assertShapeEqual(out, apply_sgd)
      self.assertAllCloseAccordingToType(x - alpha * delta, out)

  @test_util.run_v1_only("ApplyGradientDescent op returns a ref, so it is not "
                         "supported in eager mode.")
  def testApplyGradientDescent(self):
    # [FUNCTION DOC]
    # Runs the gradient descent check over float16/32/64 on CPU and GPU with a 100-element
    # range input, step size two and gradient equal to the range (graph mode only because the
    # op returns a reference).
    for (dtype, use_gpu) in itertools.product(
        [np.float16, np.float32, np.float64], [False, True]):
      x = np.arange(100).astype(dtype)
      alpha = np.array(2.0).astype(dtype)
      delta = np.arange(100).astype(dtype)
      self._testTypes(x, alpha, delta, use_gpu)

  def _testTypesForAdagrad(self, x, y, lr, grad, use_gpu=None):
    # [FUNCTION DOC]
    # HELPER: checks the Adagrad op. Inputs: x variable values, y accumulator values, lr
    # learning rate, grad gradient, use_gpu device choice. After applying the op, asserts the
    # output equals x minus lr times grad divided by the square root of (accumulator plus
    # grad squared), and that the accumulator became accumulator plus grad squared.
    self.setUp()
    with self.session(use_gpu=use_gpu):
      var = variable_v1.VariableV1(x)
      accum = variable_v1.VariableV1(y)
      self.evaluate(variables.global_variables_initializer())

      self.assertAllCloseAccordingToType(x, self.evaluate(var))
      apply_adagrad = gen_training_ops.apply_adagrad(var, accum, lr, grad)
      out = self.evaluate(apply_adagrad)
      self.assertShapeEqual(out, apply_adagrad)
      self.assertAllCloseAccordingToType(x - lr * grad * (y + grad * grad)**
                                         (-0.5), out)
      self.assertAllCloseAccordingToType(y + grad * grad, self.evaluate(accum))

  def _testTypesForFtrl(self,
                        x,
                        y,
                        z,
                        lr,
                        grad,
                        use_gpu=None,
                        l1=0.0,
                        l2=0.0,
                        lr_power=-0.5):
    # [FUNCTION DOC]
    # HELPER: checks the Ftrl op. Inputs: x variable, y accumulator, z linear term, lr, grad,
    # use_gpu, and optional l1 and l2 regularization and lr_power (defaults zero, zero, -0.5).
    # Applies the op, then asserts the output shape, the accumulator update (accumulator plus
    # grad squared), the linear term and the new variable value against numpy formulas, with
    # looser tolerances for float16 and float32 and exact-style comparison for float64.
    self.setUp()
    with self.session(use_gpu=use_gpu):
      var = variable_v1.VariableV1(x)
      accum = variable_v1.VariableV1(y)
      linear = variable_v1.VariableV1(z)
      self.evaluate(variables.global_variables_initializer())

      self.assertAllCloseAccordingToType(x, self.evaluate(var))
      apply_ftrl = gen_training_ops.apply_ftrl(var, accum, linear, grad, lr, l1,
                                               l2, lr_power)
      out = self.evaluate(apply_ftrl)
      self.assertShapeEqual(out, apply_ftrl)
      accum_update = y + grad * grad
      linear_update = z + grad - (accum_update**(-lr_power) - y**
                                  (-lr_power)) / lr * x
      quadratic = 1.0 / (accum_update**(lr_power) * lr) + 2 * l2
      expected_out = np.array([(
          np.sign(linear_update[i]) * l1 - linear_update[i]) / (quadratic[i]) if
                               np.abs(linear_update[i]) > l1 else 0.0
                               for i in range(linear_update.size)])
      self.assertAllCloseAccordingToType(accum_update, self.evaluate(accum))
      if x.dtype == np.float16:
        # The calculations here really are not very precise in float16.
        self.assertAllClose(
            linear_update, self.evaluate(linear), rtol=2e-2, atol=2e-2)
        self.assertAllClose(expected_out, out, rtol=2e-2, atol=2e-2)
      elif x.dtype == np.float32:
        # The calculations here not sufficiently precise in float32.
        self.assertAllClose(
            linear_update, self.evaluate(linear), rtol=1e-5, atol=1e-5)
        self.assertAllClose(expected_out, out, rtol=1e-5, atol=1e-5)
      else:
        self.assertAllClose(linear_update, self.evaluate(linear))
        self.assertAllClose(expected_out, out)

  def _testTypesForFtrlMultiplyLinearByLr(self,
                                          x,
                                          y,
                                          z,
                                          lr,
                                          grad,
                                          use_gpu=None,
                                          l1=0.0,
                                          l2=0.0,
                                          lr_power=-0.5):
    # [FUNCTION DOC]
    # HELPER: same as the plain Ftrl check but with the option that multiplies the linear term
    # by the learning rate turned on. Inputs: x, y, z, lr, grad, use_gpu, optional l1, l2,
    # lr_power. Asserts shape, accumulator, linear term and output against the matching numpy
    # formulas (linear updated with grad times lr, thresholds scaled by lr), with
    # dtype-dependent tolerances.
    self.setUp()
    with self.session(use_gpu=use_gpu):
      var = variable_v1.VariableV1(x)
      accum = variable_v1.VariableV1(y)
      linear = variable_v1.VariableV1(z)
      self.evaluate(variables.global_variables_initializer())

      self.assertAllCloseAccordingToType(x, self.evaluate(var))
      apply_ftrl = (
          gen_training_ops.apply_ftrl(
              var,
              accum,
              linear,
              grad,
              lr,
              l1,
              l2,
              lr_power,
              multiply_linear_by_lr=True))
      out = self.evaluate(apply_ftrl)
      self.assertShapeEqual(out, apply_ftrl)
      accum_update = y + grad * grad
      linear_update = z + grad * lr - (accum_update**(-lr_power) - y**
                                       (-lr_power)) * x
      quadratic = accum_update**(-lr_power) + 2 * l2 * lr
      expected_out = np.array([
          (np.sign(linear_update[i]) * l1 * lr - linear_update[i]) /
          (quadratic[i]) if np.abs(linear_update[i]) > l1 * lr else 0.0
          for i in range(linear_update.size)
      ])
      self.assertAllCloseAccordingToType(accum_update, self.evaluate(accum))
      if x.dtype == np.float16:
        # The calculations here really are not very precise in float16.
        self.assertAllClose(
            linear_update, self.evaluate(linear), rtol=2e-2, atol=2e-2)
        self.assertAllClose(expected_out, out, rtol=2e-2, atol=2e-2)
      elif x.dtype == np.float32:
        # The calculations here not sufficiently precise in float32.
        self.assertAllClose(
            linear_update, self.evaluate(linear), rtol=1e-5, atol=1e-5)
        self.assertAllClose(expected_out, out, rtol=1e-5, atol=1e-5)
      else:
        self.assertAllClose(linear_update, self.evaluate(linear))
        self.assertAllClose(expected_out, out)

  @test_util.run_v1_only("ApplyAdagrad op returns a ref, so it is not "
                         "supported in eager mode.")
  def testApplyAdagrad(self):
    # [FUNCTION DOC]
    # Runs the Adagrad check for float16/32/64 on CPU and GPU with range inputs, accumulator
    # starting at one, learning rate two (graph mode only).
    for (dtype, use_gpu) in itertools.product(
        [np.float16, np.float32, np.float64], [False, True]):
      x = np.arange(100).astype(dtype)
      y = np.arange(1, 101).astype(dtype)
      lr = np.array(2.0).astype(dtype)
      grad = np.arange(100).astype(dtype)
      self._testTypesForAdagrad(x, y, lr, grad, use_gpu)

  @test_util.run_v1_only("ApplyFtrl op returns a ref, so it is not "
                         "supported in eager mode.")
  def testApplyFtrl(self):
    # [FUNCTION DOC]
    # Runs the Ftrl check for float16/32/64 on CPU with l1 three, l2 four, learning rate two
    # and range inputs (graph mode only).
    for dtype in [np.float16, np.float32, np.float64]:
      x = np.arange(100).astype(dtype)
      y = np.arange(1, 101).astype(dtype)
      z = np.arange(102, 202).astype(dtype)
      lr = np.array(2.0).astype(dtype)
      l1 = np.array(3.0).astype(dtype)
      l2 = np.array(4.0).astype(dtype)
      grad = np.arange(100).astype(dtype)
      self._testTypesForFtrl(x, y, z, lr, grad, use_gpu=False, l1=l1, l2=l2)

  @test_util.run_v1_only("ApplyFtrlMultiplyLinearByLr op returns a ref, so it "
                         "is not supported in eager mode.")
  def testApplyFtrlMultiplyLinearByLr(self):
    # [FUNCTION DOC]
    # Runs the Ftrl check with the multiply-linear-by-learning-rate option for float16/32/64
    # on CPU with the same inputs as the plain Ftrl test (graph mode only).
    for dtype in [np.float16, np.float32, np.float64]:
      x = np.arange(100).astype(dtype)
      y = np.arange(1, 101).astype(dtype)
      z = np.arange(102, 202).astype(dtype)
      lr = np.array(2.0).astype(dtype)
      l1 = np.array(3.0).astype(dtype)
      l2 = np.array(4.0).astype(dtype)
      grad = np.arange(100).astype(dtype)
      self._testTypesForFtrlMultiplyLinearByLr(
          x, y, z, lr, grad, use_gpu=False, l1=l1, l2=l2)

  def _testTypesForSparseAdagrad(self, x, y, lr, grad, indices, use_gpu):
    # [FUNCTION DOC]
    # HELPER: checks the sparse Adagrad op. Inputs: x variable rows, y accumulator rows, lr,
    # grad rows, indices of the rows to update, use_gpu. Applies the op and asserts the output
    # shape and, for each listed index, that the variable row and accumulator row match the
    # Adagrad formulas. Works with empty gradients and indices too.
    self.setUp()
    with self.session(use_gpu=use_gpu):
      var = variable_v1.VariableV1(x)
      accum = variable_v1.VariableV1(y)
      self.evaluate(variables.global_variables_initializer())

      self.assertAllCloseAccordingToType(x, self.evaluate(var))
      sparse_apply_adagrad = gen_training_ops.sparse_apply_adagrad(
          var, accum, lr, grad,
          constant_op.constant(indices, self._toType(indices.dtype)))
      out = self.evaluate(sparse_apply_adagrad)
      self.assertShapeEqual(out, sparse_apply_adagrad)

      for (i, index) in enumerate(indices):
        self.assertAllCloseAccordingToType(
            x[index] - lr * grad[i] * (y[index] + grad[i] * grad[i])**(-0.5),
            self.evaluate(var)[index])
        self.assertAllCloseAccordingToType(y[index] + grad[i] * grad[i],
                                           self.evaluate(accum)[index])

  def _testTypesForSparseFtrl(self,
                              x,
                              y,
                              z,
                              lr,
                              grad,
                              indices,
                              use_gpu,
                              l1=0.0,
                              l2=0.0,
                              lr_power=-0.5):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the expected-value check uses the simple Adagrad-style formula with lr_power
    # as the exponent and ignores l1, l2 and the linear term, so it only validates the
    # variable and accumulator rows for the default zero regularization cases used.
    # HELPER: checks the sparse Ftrl op. Inputs: x, y, z, lr, grad rows, indices, use_gpu,
    # optional l1, l2, lr_power. Applies the op and asserts shape and, for each index, the
    # variable and accumulator rows.
    self.setUp()
    with self.session(use_gpu=use_gpu):
      var = variable_v1.VariableV1(x)
      accum = variable_v1.VariableV1(y)
      linear = variable_v1.VariableV1(z)
      self.evaluate(variables.global_variables_initializer())

      self.assertAllCloseAccordingToType(x, self.evaluate(var))
      sparse_apply_ftrl = gen_training_ops.sparse_apply_ftrl(
          var,
          accum,
          linear,
          grad,
          constant_op.constant(indices, self._toType(indices.dtype)),
          lr,
          l1,
          l2,
          lr_power=lr_power)
      out = self.evaluate(sparse_apply_ftrl)
      self.assertShapeEqual(out, sparse_apply_ftrl)

      for (i, index) in enumerate(indices):
        self.assertAllCloseAccordingToType(
            x[index] - lr * grad[i] *
            (y[index] + grad[i] * grad[i])**(lr_power),
            self.evaluate(var)[index])
        self.assertAllCloseAccordingToType(y[index] + grad[i] * grad[i],
                                           self.evaluate(accum)[index])

  def _testTypesForSparseFtrlMultiplyLinearByLr(self,
                                                x,
                                                y,
                                                z,
                                                lr,
                                                grad,
                                                indices,
                                                l1=0.0,
                                                l2=0.0,
                                                lr_power=-0.5):
    # [FUNCTION DOC]
    # KNOWN ISSUE: like the plain sparse Ftrl check it ignores l1, l2 and the linear term in
    # its expected values.
    # HELPER: checks the sparse Ftrl op with multiply-linear-by-learning-rate enabled, always
    # on CPU. Inputs: x, y, z, lr, grad rows, indices, optional l1, l2, lr_power. Asserts shape
    # and per-index variable and accumulator rows.
    self.setUp()
    with self.session(use_gpu=False):
      var = variable_v1.VariableV1(x)
      accum = variable_v1.VariableV1(y)
      linear = variable_v1.VariableV1(z)
      self.evaluate(variables.global_variables_initializer())

      self.assertAllCloseAccordingToType(x, self.evaluate(var))
      sparse_apply_ftrl = (
          gen_training_ops.sparse_apply_ftrl(
              var,
              accum,
              linear,
              grad,
              constant_op.constant(indices, self._toType(indices.dtype)),
              lr,
              l1,
              l2,
              lr_power=lr_power,
              multiply_linear_by_lr=True))
      out = self.evaluate(sparse_apply_ftrl)
      self.assertShapeEqual(out, sparse_apply_ftrl)

      for (i, index) in enumerate(indices):
        self.assertAllCloseAccordingToType(
            x[index] - lr * grad[i] * (y[index] + grad[i] * grad[i])**
            (lr_power),
            self.evaluate(var)[index])
        self.assertAllCloseAccordingToType(y[index] + grad[i] * grad[i],
                                           self.evaluate(accum)[index])

  @test_util.run_v1_only("SparseApplyAdagrad op returns a ref, so it is not "
                         "supported in eager mode.")
  def testSparseApplyAdagrad(self):
    # [FUNCTION DOC]
    # Runs the sparse Adagrad check over float16/32/64, int32/int64 index types and CPU/GPU,
    # updating rows zero and two of a three-row variable, then repeats with empty gradients
    # and empty indices (graph mode only).
    for (dtype, index_type,
         use_gpu) in itertools.product([np.float16, np.float32, np.float64],
                                       [np.int32, np.int64], [False, True]):
      x_val = [np.arange(10), np.arange(10, 20), np.arange(20, 30)]
      y_val = [np.arange(1, 11), np.arange(11, 21), np.arange(21, 31)]
      x = np.array(x_val).astype(dtype)
      y = np.array(y_val).astype(dtype)
      lr = np.array(2.0).astype(dtype)
      grad_val = [np.arange(10), np.arange(10)]
      grad = np.array(grad_val).astype(dtype)
      indices = np.array([0, 2]).astype(index_type)
      self._testTypesForSparseAdagrad(x, y, lr, grad, indices, use_gpu)
      # Empty sparse gradients.
      empty_grad = np.zeros([0, 10], dtype=dtype)
      empty_indices = np.zeros([0], dtype=index_type)
      self._testTypesForSparseAdagrad(x, y, lr, empty_grad, empty_indices,
                                      use_gpu)

  @test_util.run_v1_only("SparseApplyAdagrad op returns a ref, so it is not "
                         "supported in eager mode.")
  def testSparseApplyAdagradDim1(self):
    # [FUNCTION DOC]
    # Sparse Adagrad check with single-column rows over the same dtype, index type and device
    # combinations (graph mode only).
    for (dtype, index_type,
         use_gpu) in itertools.product([np.float16, np.float32, np.float64],
                                       [np.int32, np.int64], [False, True]):
      x_val = [[1.0], [2.0], [3.0]]
      y_val = [[4.0], [5.0], [6.0]]
      x = np.array(x_val).astype(dtype)
      y = np.array(y_val).astype(dtype)
      lr = np.array(2.0).astype(dtype)
      grad_val = [[1.5], [2.5]]
      grad = np.array(grad_val).astype(dtype)
      indices = np.array([0, 2]).astype(index_type)
      self._testTypesForSparseAdagrad(x, y, lr, grad, indices, use_gpu)

  @test_util.run_v1_only("SparseApplyFtrl op returns a ref, so it is not "
                         "supported in eager mode.")
  def testSparseApplyFtrlDim1(self):
    # [FUNCTION DOC]
    # Sparse Ftrl check with single-column rows, zero initial variable and linear term, over
    # float types, index types and devices, also with empty gradients and indices (graph only).
    for (dtype, index_type,
         use_gpu) in itertools.product([np.float16, np.float32, np.float64],
                                       [np.int32, np.int64], [False, True]):
      x_val = [[0.0], [0.0], [0.0]]
      y_val = [[4.0], [5.0], [6.0]]
      z_val = [[0.0], [0.0], [0.0]]
      x = np.array(x_val).astype(dtype)
      y = np.array(y_val).astype(dtype)
      z = np.array(z_val).astype(dtype)
      lr = np.array(2.0).astype(dtype)
      grad_val = [[1.5], [2.5]]
      grad = np.array(grad_val).astype(dtype)
      indices = np.array([0, 2]).astype(index_type)
      self._testTypesForSparseFtrl(x, y, z, lr, grad, indices, use_gpu)
      # Empty sparse gradients.
      empty_grad = np.zeros([0, 1], dtype=dtype)
      empty_indices = np.zeros([0], dtype=index_type)
      self._testTypesForSparseFtrl(x, y, z, lr, empty_grad, empty_indices,
                                   use_gpu)

  @test_util.run_v1_only("SparseApplyFtrlMultiplyLinearByLr op returns a ref, "
                         "so it is not supported in eager mode.")
  def testSparseApplyFtrlMultiplyLinearByLrDim1(self):
    # [FUNCTION DOC]
    # Sparse Ftrl check with multiply-linear-by-learning-rate enabled, single-column rows, over
    # float types and index types on CPU (graph mode only).
    for (dtype,
         index_type) in itertools.product([np.float16, np.float32, np.float64],
                                          [np.int32, np.int64]):
      x_val = [[0.0], [0.0], [0.0]]
      y_val = [[4.0], [5.0], [6.0]]
      z_val = [[0.0], [0.0], [0.0]]
      x = np.array(x_val).astype(dtype)
      y = np.array(y_val).astype(dtype)
      z = np.array(z_val).astype(dtype)
      lr = np.array(2.0).astype(dtype)
      grad_val = [[1.5], [2.5]]
      grad = np.array(grad_val).astype(dtype)
      indices = np.array([0, 2]).astype(index_type)
      self._testTypesForSparseFtrlMultiplyLinearByLr(x, y, z, lr, grad, indices)

  @test_util.run_v1_only("ApplyAdam op returns a ref, so it is not "
                         "supported in eager mode.")
  def testApplyAdam(self):
    # [FUNCTION DOC]
    # Runs the Adam op check for float16/32/64 on CPU and GPU with range-valued variable,
    # first and second moments and gradient (graph mode only).
    for dtype, use_gpu in itertools.product(
        [np.float16, np.float32, np.float64], [False, True]):
      var = np.arange(100).astype(dtype)
      m = np.arange(1, 101).astype(dtype)
      v = np.arange(101, 201).astype(dtype)
      grad = np.arange(100).astype(dtype)
      self._testTypesForAdam(var, m, v, grad, use_gpu)

  def _testTypesForAdam(self, var, m, v, grad, use_gpu):
    # [FUNCTION DOC]
    # HELPER: checks the Adam op at step one. Inputs: var values, m first moment, v second
    # moment, grad, use_gpu. Uses beta1 0.9, beta2 0.999, learning rate 0.001, epsilon 1e-8,
    # computes the expected variable with the numpy helper, applies the op and asserts the
    # output shape and value.
    self.setUp()
    with self.session(use_gpu=use_gpu):
      var_t = variable_v1.VariableV1(var)
      m_t = variable_v1.VariableV1(m)
      v_t = variable_v1.VariableV1(v)

      t = 1
      beta1 = np.array(0.9, dtype=var.dtype)
      beta2 = np.array(0.999, dtype=var.dtype)
      beta1_power = beta1**t
      beta2_power = beta2**t
      lr = np.array(0.001, dtype=var.dtype)
      epsilon = np.array(1e-8, dtype=var.dtype)
      beta1_t = constant_op.constant(beta1, self._toType(var.dtype), [])
      beta2_t = constant_op.constant(beta2, self._toType(var.dtype), [])
      beta1_power_t = variable_v1.VariableV1(beta1_power)
      beta2_power_t = variable_v1.VariableV1(beta2_power)
      lr_t = constant_op.constant(lr, self._toType(var.dtype), [])
      epsilon_t = constant_op.constant(epsilon, self._toType(var.dtype), [])
      self.evaluate(variables.global_variables_initializer())

      self.assertAllCloseAccordingToType(var, self.evaluate(var_t))
      new_var, _, _ = self._adamUpdateNumpy(var, grad, t, m, v, lr, beta1,
                                            beta2, epsilon)
      apply_adam = gen_training_ops.apply_adam(var_t, m_t, v_t, beta1_power_t,
                                               beta2_power_t, lr_t, beta1_t,
                                               beta2_t, epsilon_t, grad)
      out = self.evaluate(apply_adam)
      self.assertShapeEqual(out, apply_adam)
      self.assertAllCloseAccordingToType(new_var, out)

  def _adamUpdateNumpy(self, param, g_t, t, m, v, alpha, beta1, beta2, epsilon):
    # [FUNCTION DOC]
    # HELPER: numpy reference for one Adam update. Inputs: param, gradient g_t, step t, moments
    # m and v, learning rate alpha, beta1, beta2, epsilon. Returns the updated parameter and
    # the updated first and second moments, using bias-corrected step size.
    alpha_t = alpha * np.sqrt(1 - beta2**t) / (1 - beta1**t)

    m_t = beta1 * m + (1 - beta1) * g_t
    v_t = beta2 * v + (1 - beta2) * g_t * g_t

    param_t = param - alpha_t * m_t / (np.sqrt(v_t) + epsilon)
    return param_t, m_t, v_t

  @test_util.run_v2_only
  def testResourceSparseApplyAdagradV2AndDisableCopyOnReadRace(self):
    # [FUNCTION DOC]
    # Regression test (eager/v2 only) for a deadlock between disabling copy-on-read and the
    # resource sparse Adagrad v2 op on the same variables. Builds two variables, defines two
    # compiled functions that loop a thousand times (one disabling copy-on-read on both
    # variables, one applying the sparse op), runs them in two threads at once and joins both.
    # Passing means no deadlock or error occurs.
    dtype = np.float32
    index_type = np.int32
    x_val = [np.arange(10), np.arange(10, 20), np.arange(20, 30)]
    y_val = [np.arange(1, 11), np.arange(11, 21), np.arange(21, 31)]
    x = np.array(x_val).astype(dtype)
    y = np.array(y_val).astype(dtype)
    lr = np.array(0.001, dtype=dtype)
    epsilon = np.array(1e-8, dtype=dtype)
    grad_val = [np.arange(10), np.arange(10)]
    grad = np.array(grad_val).astype(dtype)
    indices = np.array([0, 2]).astype(index_type)
    var = variables.Variable(x)
    accum = variables.Variable(y)
    num_iter = 1000
    self.evaluate(variables.global_variables_initializer())

    @def_function.function
    def fn_disable_copy_on_read():
      # [FUNCTION DOC]
      # Nested compiled function: loops over a range, each time disabling copy-on-read on both
      # variables' handles with the next addition ordered after them. Returns the loop sum.
      ret = constant_op.constant(0, dtypes.int32)
      for i in math_ops.range(num_iter):
        op1 = resource_variable_ops.disable_copy_on_read(var.handle)
        op2 = resource_variable_ops.disable_copy_on_read(accum.handle)
        with ops.control_dependencies([op1, op2]):
          ret += i
      return ret

    @def_function.function
    def fn_resource_sparse_apply_adagrad_v2():
      # [FUNCTION DOC]
      # Nested compiled function: loops over a range, each time applying the resource sparse
      # Adagrad v2 op to the variable and accumulator rows at the chosen indices, with the
      # addition ordered after it. Returns the loop sum.
      ret = constant_op.constant(0, dtypes.int32)
      for i in math_ops.range(num_iter):
        adagrad_op = gen_training_ops.resource_sparse_apply_adagrad_v2(
            var.handle, accum.handle, lr, epsilon, grad,
            constant_op.constant(indices, dtypes.int32))
        with ops.control_dependencies([adagrad_op]):
          ret += i
      return ret

    # Run two tf.functions simultaneously to make sure there is no race
    # condition between the two ops that caused deadlock before (b/270712679).
    thread1 = threading.Thread(
        target=lambda: self.evaluate(fn_disable_copy_on_read()))
    thread2 = threading.Thread(
        target=lambda: self.evaluate(fn_resource_sparse_apply_adagrad_v2()))
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()


if __name__ == '__main__':
  googletest.main()
