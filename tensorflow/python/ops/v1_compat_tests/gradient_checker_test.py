# [MODULE DOC]
# Tests for the v1 gradient checker (compute_gradient and compute_gradient_error) in graph mode.
# Registers two custom gradients ("BadGrad" returns a transposed, wrongly shaped gradient and
# "NaNGrad" returns not-a-number values) used to exercise failure paths, then checks numeric
# versus analytic gradients for add, gather, complex ops, empty tensors and a mini MNIST model.
# Importing registers those gradients globally; run as a script it starts the test runner.
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
"""Tests for tf.compat.v1.test.compute_gradient and tf.compute_gradient_error."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


@ops.RegisterGradient("BadGrad")
def _bad_grad(unused_op, grad):
  # [FUNCTION DOC]
  # HELPER: Gradient function registered under the name BadGrad for the empty-gradient failure test.
  # Ignores the op and returns the transpose of the incoming gradient, so an empty input of shape
  # (0, 3) receives a gradient of shape (3, 0), which the checker must reject. No side effects.
  """A gradient that returns the wrong shape."""
  return array_ops.transpose(grad)


@ops.RegisterGradient("NaNGrad")
def _nan_grad(unused_op, grad):
  # [FUNCTION DOC]
  # HELPER: Gradient function registered under the name NaNGrad for the NaN failure test.
  # Ignores the op and returns the incoming gradient multiplied by not-a-number, so every analytic
  # gradient entry becomes NaN. No side effects.
  """A gradient that returns NaN."""
  return np.nan * grad


class GradientCheckerTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the v1 gradient checker on small graphs: add, gather, complex multiply and
  # conjugate, empty inputs, and failure detection with custom bad or NaN gradients. Most tests are
  # decorated to run as deprecated v1 graph tests and use a session per test.

  @test_util.run_deprecated_v1
  def testAddSimple(self):
    # [FUNCTION DOC]
    # Scenario: float32 add of two constants of shape (2, 3), CPU-only session, fixed random seed.
    # Computes the gradient error with respect to the first input and asserts it is below 1e-4.
    np.random.seed(1)  # Fix seed to avoid flakiness
    with self.session(use_gpu=False):
      # a test case for Add operation
      size = (2, 3)
      x1 = constant_op.constant(2.0, shape=size, name="x1")
      x2 = constant_op.constant(3.0, shape=size, name="x2")
      y = math_ops.add(x1, x2, name="y")

      # checking gradients for x1
      error = gradient_checker.compute_gradient_error(x1, size, y, size)
    tf_logging.info("x1 error = %f", error)
    self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testAddSimpleGPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says GPU, but the body is identical to the CPU test except for the seed
    # and uses a default session without explicitly requesting a GPU, so a GPU is only used if it
    # is the default device.
    # Scenario: add of two float32 constants of shape (2, 3). Asserts that the gradient error for the
    # first input is below 1e-4.
    np.random.seed(2)  # Fix seed to avoid flakiness
    with self.session():
      # a test case for Add operation
      size = (2, 3)
      x1 = constant_op.constant(2.0, shape=size, name="x1")
      x2 = constant_op.constant(3.0, shape=size, name="x2")
      y = math_ops.add(x1, x2, name="y")

      # checking gradients for x1
      error = gradient_checker.compute_gradient_error(x1, size, y, size)
    tf_logging.info("x1 error = %f", error)
    self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testAddCustomized(self):
    # [FUNCTION DOC]
    # Scenario: float64 add of two constants of shape (2, 3). Checks the gradient of the second input
    # using a caller-supplied initial value (0..5 reshaped to 2 by 3) and a custom step of 0.01.
    # Asserts the gradient error is below 1e-10.
    np.random.seed(3)  # Fix seed to avoid flakiness
    with self.cached_session():
      # a test case for Add operation
      size = (2, 3)
      x1 = constant_op.constant(
          2.0, shape=size, dtype=dtypes.float64, name="x1")
      x2 = constant_op.constant(
          3.0, shape=size, dtype=dtypes.float64, name="x2")
      y = math_ops.add(x1, x2, name="y")

      # checkint gradients for x2 using a special init_value and delta
      x_init_value = np.asarray(np.arange(6, dtype=np.float64).reshape(2, 3))
      error = gradient_checker.compute_gradient_error(
          x2, size, y, size, x_init_value=x_init_value, delta=1e-2)
    tf_logging.info("x2 error = %f", error)
    self.assertLess(error, 1e-10)

  @test_util.run_deprecated_v1
  def testGather(self):
    # [FUNCTION DOC]
    # Scenario: gather rows 1 and 3 of a float64 parameter matrix of shape (4, 2), output shape
    # (2, 2). Asserts the gradient error with respect to the parameters is below 1e-4.
    np.random.seed(4)  # Fix seed to avoid flakiness
    with self.cached_session():
      p_shape = (4, 2)
      p_size = 8
      index_values = [1, 3]
      y_shape = [2, 2]
      params = constant_op.constant(
          np.arange(p_size).astype(np.float64), shape=p_shape, name="p")
      indices = constant_op.constant(index_values, name="i")
      y = array_ops.gather(params, indices, name="y")

      error = gradient_checker.compute_gradient_error(params, p_shape, y,
                                                      y_shape)
    tf_logging.info("gather error = %f", error)
    self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testNestedGather(self):
    # [FUNCTION DOC]
    # Scenario: two chained gathers on a float64 matrix of shape (8, 2) (rows 1,3,5,6 then rows 0,2
    # of the result), output shape (2, 2). Asserts the gradient error is below 1e-4.
    np.random.seed(5)  # Fix seed to avoid flakiness
    with self.cached_session():
      p_shape = (8, 2)
      p_size = 16
      index_values = [1, 3, 5, 6]
      index_values2 = [0, 2]
      y2_shape = [2, 2]

      params = constant_op.constant(
          np.arange(p_size).astype(np.float64), shape=p_shape, name="p")
      indices = constant_op.constant(index_values, name="i")
      y = array_ops.gather(params, indices, name="y")
      indices2 = constant_op.constant(index_values2, name="i2")
      y2 = array_ops.gather(y, indices2, name="y2")

      error = gradient_checker.compute_gradient_error(params, p_shape, y2,
                                                      y2_shape)
    tf_logging.info("nested gather error = %f", error)
    self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testComplexMul(self):
    # [FUNCTION DOC]
    # Scenario: scalar complex64 multiplication by a constant complex number. Computes the analytic
    # and numeric gradients, asserts the analytic one exactly equals the expected 2 by 2 real matrix
    # of the constant, the numeric one is close (relative tolerance 1e-4), and the error is below 3e-4.
    with self.cached_session():
      size = ()
      c = constant_op.constant(5 + 7j, dtype=dtypes.complex64)
      x = constant_op.constant(11 - 13j, dtype=dtypes.complex64)
      y = c * x
      analytical, numerical = gradient_checker.compute_gradient(x, size, y,
                                                                size)
      correct = np.array([[5, 7], [-7, 5]])
      self.assertAllEqual(correct, analytical)
      self.assertAllClose(correct, numerical, rtol=1e-4)
      self.assertLess(
          gradient_checker.compute_gradient_error(x, size, y, size), 3e-4)

  @test_util.run_deprecated_v1
  def testComplexConj(self):
    # [FUNCTION DOC]
    # Scenario: scalar complex64 conjugate. Asserts the analytic gradient exactly equals the 2 by 2
    # real matrix with entries 1, 0, 0, -1, the numeric one is close (relative tolerance 2e-5), and
    # the gradient error is below 2e-5.
    with self.cached_session():
      size = ()
      x = constant_op.constant(11 - 13j, dtype=dtypes.complex64)
      y = math_ops.conj(x)
      analytical, numerical = gradient_checker.compute_gradient(x, size, y,
                                                                size)
      correct = np.array([[1, 0], [0, -1]])
      self.assertAllEqual(correct, analytical)
      self.assertAllClose(correct, numerical, rtol=2e-5)
      self.assertLess(
          gradient_checker.compute_gradient_error(x, size, y, size), 2e-5)

  @test_util.run_deprecated_v1
  def testEmptySucceeds(self):
    # [FUNCTION DOC]
    # Scenario: identity of a float32 placeholder with an empty shape (0, 3). Asserts that both
    # returned gradient arrays have shape (0, 0) and that the computed gradient error is exactly 0.
    with self.cached_session():
      x = array_ops.placeholder(dtypes.float32)
      y = array_ops.identity(x)
      for grad in gradient_checker.compute_gradient(x, (0, 3), y, (0, 3)):
        self.assertEqual(grad.shape, (0, 0))
      error = gradient_checker.compute_gradient_error(x, (0, 3), y, (0, 3))
      self.assertEqual(error, 0)

  def testEmptyFails(self):
    # [FUNCTION DOC]
    # Scenario: identity op whose gradient is overridden with BadGrad (transposed gradient) on an
    # empty (0, 3) input in a fresh graph. Asserts both compute_gradient and compute_gradient_error
    # raise a value error saying the empty gradient has the wrong shape (expected (0, 3), got (3, 0)).
    with ops.Graph().as_default() as g:
      with self.session(graph=g):
        x = array_ops.placeholder(dtypes.float32)
        with g.gradient_override_map({"Identity": "BadGrad"}):
          y = array_ops.identity(x)
        bad = r"Empty gradient has wrong shape: expected \(0, 3\), got \(3, 0\)"
        with self.assertRaisesRegex(ValueError, bad):
          gradient_checker.compute_gradient(x, (0, 3), y, (0, 3))
        with self.assertRaisesRegex(ValueError, bad):
          gradient_checker.compute_gradient_error(x, (0, 3), y, (0, 3))

  def testNaNGradFails(self):
    # [FUNCTION DOC]
    # Scenario: identity op whose gradient is overridden with NaNGrad, scalar input. Computes the
    # gradient error, then asserts that the usual check of the error being less than 1.0 fails with
    # an assertion error, because NaN never compares as smaller. The compute call itself must not raise.
    with ops.Graph().as_default() as g:
      with self.session(graph=g):
        x = array_ops.placeholder(dtypes.float32)
        with g.gradient_override_map({"Identity": "NaNGrad"}):
          y = array_ops.identity(x)
          error = gradient_checker.compute_gradient_error(x, (), y, ())
          # Typical test would assert error < max_err, so assert this test would
          # raise AssertionError, since NaN is not < 1.0.
          error_msg = "(False|np.False_) is not true"
          with self.assertRaisesRegex(AssertionError, error_msg):
            self.assertTrue(error < 1.0)


class MiniMNISTTest(test.TestCase):
  # [CLASS DOC]
  # Test case that builds a tiny MNIST-style network (relu hidden layer plus softmax cross entropy
  # in float64) and checks the gradient of each of its five parameter tensors via the shared helper.

  # Gradient checker for MNIST.
  def _BuildAndTestMiniMNIST(self, param_index, tag):
    # [FUNCTION DOC]
    # HELPER: Shared implementation for the MiniMNIST gradient tests.
    # Inputs: param_index, an integer 0-4 selecting which tensor to check (input, hidden weight,
    # hidden bias, softmax weight, softmax bias); tag, a string used only in the log message.
    # Seeds numpy, builds random constant data (batch 3, 16 inputs, 32 features, 10 classes, labels
    # normalized per row), builds the network, and computes the gradient error of the cost with
    # respect to the selected tensor with step 1e-5. Logs the error and returns it as a float.
    # Fix seed to avoid occasional flakiness
    np.random.seed(6)

    # Hyperparameters
    batch = 3
    inputs = 16
    features = 32
    classes = 10

    # Define the parameters
    inp_data = np.random.random_sample(inputs * batch)
    hidden_weight_data = np.random.randn(inputs * features) / np.sqrt(inputs)
    hidden_bias_data = np.random.random_sample(features)
    sm_weight_data = np.random.randn(features * classes) / np.sqrt(features)
    sm_bias_data = np.random.random_sample(classes)

    # special care for labels since they need to be normalized per batch
    label_data = np.random.random(batch * classes).reshape((batch, classes))
    s = label_data.sum(axis=1)
    label_data /= s[:, None]

    with self.session():
      # We treat the inputs as "parameters" here
      inp = constant_op.constant(
          inp_data.tolist(),
          shape=[batch, inputs],
          dtype=dtypes.float64,
          name="inp")
      hidden_weight = constant_op.constant(
          hidden_weight_data.tolist(),
          shape=[inputs, features],
          dtype=dtypes.float64,
          name="hidden_weight")
      hidden_bias = constant_op.constant(
          hidden_bias_data.tolist(),
          shape=[features],
          dtype=dtypes.float64,
          name="hidden_bias")
      softmax_weight = constant_op.constant(
          sm_weight_data.tolist(),
          shape=[features, classes],
          dtype=dtypes.float64,
          name="softmax_weight")
      softmax_bias = constant_op.constant(
          sm_bias_data.tolist(),
          shape=[classes],
          dtype=dtypes.float64,
          name="softmax_bias")

      # List all the parameter so that we can test them one at a time
      all_params = [
          inp, hidden_weight, hidden_bias, softmax_weight, softmax_bias
      ]
      param_sizes = [
          [batch, inputs],  # inp
          [inputs, features],  # hidden_weight,
          [features],  # hidden_bias
          [features, classes],  # softmax_weight,
          [classes]
      ]  # softmax_bias

      # Now, Building MNIST
      features = nn_ops.relu(
          nn_ops.xw_plus_b(inp, hidden_weight, hidden_bias), name="features")
      logits = nn_ops.xw_plus_b(
          features, softmax_weight, softmax_bias, name="logits")
      labels = constant_op.constant(
          label_data.tolist(),
          shape=[batch, classes],
          dtype=dtypes.float64,
          name="labels")
      cost = nn_ops.softmax_cross_entropy_with_logits(
          labels=labels, logits=logits, name="cost")

      # Test the gradients.
      err = gradient_checker.compute_gradient_error(
          all_params[param_index],
          param_sizes[param_index],
          cost, [batch],
          delta=1e-5)

    tf_logging.info("Mini MNIST: %s gradient error = %g", tag, err)
    return err

  @test_util.run_deprecated_v1
  def testInputGradient(self):
    # [FUNCTION DOC]
    # Asserts the mini MNIST gradient error with respect to the input data is below 1e-8.
    self.assertLess(self._BuildAndTestMiniMNIST(0, "input"), 1e-8)

  @test_util.run_deprecated_v1
  def testHiddenWeightGradient(self):
    # [FUNCTION DOC]
    # Asserts the mini MNIST gradient error with respect to the hidden weights is below 1e-8.
    self.assertLess(self._BuildAndTestMiniMNIST(1, "hidden_weight"), 1e-8)

  @test_util.run_deprecated_v1
  def testHiddenBiasGradient(self):
    # [FUNCTION DOC]
    # Asserts the mini MNIST gradient error with respect to the hidden bias is below 1e-8.
    self.assertLess(self._BuildAndTestMiniMNIST(2, "hidden_bias"), 1e-8)

  @test_util.run_deprecated_v1
  def testSoftmaxWeightGradient(self):
    # [FUNCTION DOC]
    # Asserts the mini MNIST gradient error with respect to the softmax weights is below 1e-8.
    self.assertLess(self._BuildAndTestMiniMNIST(3, "softmax_weight"), 1e-8)

  @test_util.run_deprecated_v1
  def testSoftmaxBiasGradient(self):
    # [FUNCTION DOC]
    # Asserts the mini MNIST gradient error with respect to the softmax bias is below 1e-8.
    self.assertLess(self._BuildAndTestMiniMNIST(4, "softmax_bias"), 1e-8)


if __name__ == "__main__":
  test.main()
