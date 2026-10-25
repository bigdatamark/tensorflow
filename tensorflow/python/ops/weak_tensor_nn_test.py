# [MODULE DOC]
# Tests that miscellaneous nn ops (log softmax, relu, leaky relu, gelu, swish) accept weak tensors
# (WeakTensor) and return WeakTensor results with values matching numpy or analytic references,
# including gradient checks via the numerical gradient checker. Uses the shared weak tensor test
# utility to create inputs. Running as a script sets the dtype conversion mode to "all" before
# launching the test runner.
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
"""Tests for miscellaneous functionality in tensorflow.ops.nn on WeakTensor."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework import weak_tensor
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import weak_tensor_ops  # pylint: disable=unused-import
from tensorflow.python.ops import weak_tensor_test_util
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test as test_lib


_get_weak_tensor = weak_tensor_test_util.get_weak_tensor


@test_util.run_all_in_graph_and_eager_modes
class LogSoftmaxTest(test_lib.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case (graph and eager modes) for nn_ops.log_softmax_v2 on weak tensors:
  # values versus a numpy reference, axis handling, and gradients.

  def _log_softmax(self, x):
    # [FUNCTION DOC]
    # HELPER: numpy reference implementation of log softmax over axis 1 for the tests here.
    # Input: x, a rank-2 numpy array (asserts the rank is 2, otherwise an assertion error).
    # Subtracts the row maximum for stability and returns the array of log probabilities.
    assert len(x.shape) == 2
    m = x.max(1)[:, np.newaxis]
    u = x - m
    return u - np.log(np.sum(np.exp(u), 1, keepdims=True))

  def testLogSoftmax(self):
    # [FUNCTION DOC]
    # Applies log_softmax_v2 to a random 5x10 float32 weak tensor and asserts the result is a
    # WeakTensor close (tolerance 1e-3) to the numpy reference.
    x_shape = [5, 10]
    x_np = np.random.randn(*x_shape).astype(np.float32)
    y_np = self._log_softmax(x_np)
    x_wt = _get_weak_tensor(x_np)
    y_wt = nn_ops.log_softmax_v2(x_wt)
    y_tf_np = self.evaluate(y_wt)
    eps = 1e-3

    self.assertIsInstance(y_wt, weak_tensor.WeakTensor)
    self.assertAllClose(y_tf_np, y_np, eps)

  def testLogSoftmaxAxes(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the variable meant to test a different axis (named for a greater-than-rank
    # axis) also uses axis 0, so it duplicates the positive-axis case and no out-of-range axis is
    # actually tested.
    # Applies log_softmax_v2 to a 3x4 weak tensor with axis -2 and axis 0 (twice) and asserts the
    # negative-axis result equals the positive-axis result and the two axis-0 results agree.
    arr = _get_weak_tensor(np.linspace(0.0, 1, 12).reshape(3, 4))
    x_neg_axis = nn_ops.log_softmax_v2(arr, axis=-2)
    y_pos_axis = nn_ops.log_softmax_v2(arr, axis=0)
    z_gt_axis = nn_ops.log_softmax_v2(arr, axis=0)
    x_neg_axis_tf = self.evaluate(x_neg_axis)
    y_pos_axis_tf = self.evaluate(y_pos_axis)
    z_gt_axis_tf = self.evaluate(z_gt_axis)
    eps = 1e-3
    self.assertAllClose(x_neg_axis_tf, y_pos_axis_tf, eps)
    self.assertAllClose(y_pos_axis_tf, z_gt_axis_tf, eps)

  @parameterized.parameters(((5, 10),), ((2, 3, 4),))
  def testGradient(self, x_shape):
    # [FUNCTION DOC]
    # Parameterized over shapes (5, 10) and (2, 3, 4). Wraps random float64 data as a weak
    # tensor, computes theoretical and numerical gradients of log_softmax_v2 with the gradient
    # checker, and asserts they are close.
    x_np = np.random.randn(*x_shape).astype(np.float64)
    with self.cached_session():
      x_tf = _get_weak_tensor(x_np)
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          nn_ops.log_softmax_v2, [x_tf])
      self.assertAllClose(theoretical, numerical)


class ReluTest(test_lib.TestCase):
  # [CLASS DOC]
  # Test case for nn_ops.relu on weak tensors: values and NaN propagation.

  def test(self):
    # [FUNCTION DOC]
    # Seeds the random generator, applies relu to a random 3x4 float32 weak tensor, and asserts the
    # result is a WeakTensor exactly equal to the numpy elementwise maximum with zero.
    np.random.seed(1)  # Make it reproducible.
    x = np.random.randn(3, 4).astype(np.float32)
    x_wt = _get_weak_tensor(x)
    y_wt = nn_ops.relu(x_wt)
    y = np.maximum(x, 0.0)
    z = self.evaluate(y_wt)

    self.assertIsInstance(y_wt, weak_tensor.WeakTensor)
    self.assertAllEqual(y, z)

  @test_util.disable_xla(
      "This test relies on undefined behavior that XLA does not replicate")
  @test_util.run_deprecated_v1
  def testNaNs(self):
    # [FUNCTION DOC]
    # Disabled under XLA (relies on undefined behaviour) and run in v1 graph mode. For sizes 0 to
    # 17, applies relu to a weak tensor of NaNs on CPU only (a known GPU bug) and asserts every
    # output element is NaN.
    # Test that relu(nan) = nan for various sizes.
    for i in range(18):
      x = np.zeros(i) + np.nan
      # TODO(b/178335491): This is broken on GPU today.
      with self.cached_session(use_gpu=False):
        z = nn_ops.relu(_get_weak_tensor(x)).eval()
        self.assertTrue(np.isnan(z).all())


class LeakyReluTest(test_lib.TestCase):
  # [CLASS DOC]
  # Test case for nn_ops.leaky_relu on weak tensors: output type and range, values across dtypes
  # and op naming.

  def testRange(self):
    # [FUNCTION DOC]
    # Applies leaky_relu to a seeded random weak tensor of shape 3x4x4x3 with values in 0 to 1.
    # Asserts the shape is preserved, the output is a WeakTensor, outputs lie between 0 and 1, and
    # they equal the inputs (positive values pass unchanged).
    batch_size = 3
    height, width = 4, 4
    np.random.seed(1)  # Make it reproducible.
    inputs = np.random.uniform(size=(batch_size, height, width,
                                     3)).astype(np.float32)
    inputs = _get_weak_tensor(inputs)

    outputs = nn_ops.leaky_relu(inputs)
    self.assertEqual(inputs.shape, outputs.shape)
    self.assertIsInstance(outputs, weak_tensor.WeakTensor)

    inputs, outputs = self.evaluate([inputs, outputs])

    self.assertGreaterEqual(outputs.min(), 0.0)
    self.assertLessEqual(outputs.max(), 1.0)
    self.assertAllClose(inputs, outputs)

  @test_util.run_deprecated_v1
  def testValues(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the tolerance selection has a branch for float16, but float16 is not in the
    # tested dtype list, so that branch is dead and the tolerance is always 1e-6.
    # Runs in v1 graph mode. For int32, int64, float32 and float64, applies leaky_relu to the weak
    # tensor -2, -1, 0, 1, 2, asserts a WeakTensor result and values -0.4, -0.2, 0, 1, 2 (default
    # slope 0.2).
    for dtype in [np.int32, np.int64, np.float32, np.float64]:
      values = _get_weak_tensor(np.array([-2, -1, 0, 1, 2], dtype=dtype))
      outputs = nn_ops.leaky_relu(values)
      self.assertIsInstance(outputs, weak_tensor.WeakTensor)

      outputs = self.evaluate(outputs)

      tol = 2e-3 if dtype == np.float16 else 1e-6
      self.assertAllClose(
          outputs, [-0.4, -0.2, 0.0, 1.0, 2.0], rtol=tol, atol=tol)

  @test_util.run_deprecated_v1
  def testName(self):
    # [FUNCTION DOC]
    # Runs in v1 graph mode. Asserts leaky_relu on a float64 weak tensor returns a WeakTensor named
    # with the supplied name when one is given, and with the default operation name otherwise.
    values = _get_weak_tensor(np.array([-2, -1, 0, 1, 2], dtype=np.float64))
    outputs_with_name_set = nn_ops.leaky_relu(values, name="test_relu_op")
    self.assertEqual(outputs_with_name_set.name, "test_relu_op:0")
    self.assertIsInstance(outputs_with_name_set, weak_tensor.WeakTensor)

    outputs_without_name_set = nn_ops.leaky_relu(values)
    self.assertEqual(outputs_without_name_set.name, "LeakyRelu:0")
    self.assertIsInstance(outputs_without_name_set, weak_tensor.WeakTensor)


class GeluTest(test_lib.TestCase):
  # [CLASS DOC]
  # Test case for nn_ops.gelu on weak tensors, exact and approximate variants.

  def test(self):
    # [FUNCTION DOC]
    # Seeds the random generator, builds a random 3x4 float32 weak tensor, and compares nn_ops.gelu
    # with a reference for both the exact variant and the approximate (tanh) variant, asserting
    # WeakTensor results close to the references. The exact reference imports scipy lazily, so
    # scipy must be available.

    def gelu(x, approximate=False):
      # [FUNCTION DOC]
      # HELPER: numpy reference gelu for the enclosing test.
      # Inputs: x, a numpy array; approximate, whether to use the tanh approximation (default
      # false). Returns the approximate formula when requested, otherwise x times the standard
      # normal cumulative distribution computed with scipy (imported inside the function).
      if approximate:
        return 0.5 * x * (1.0 + np.tanh(
            np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))
      else:
        from scipy.stats import norm  # pylint: disable=g-import-not-at-top
        return x * norm.cdf(x)

    np.random.seed(1)  # Make it reproducible.
    x = np.random.randn(3, 4).astype(np.float32)
    x_wt = weak_tensor.WeakTensor(x)
    y = gelu(x)
    y_wt = nn_ops.gelu(x_wt)
    z = self.evaluate(y_wt)
    self.assertIsInstance(y_wt, weak_tensor.WeakTensor)
    self.assertAllClose(y, z)

    y = gelu(x, True)
    y_wt = nn_ops.gelu(x_wt, True)
    z = self.evaluate(y_wt)
    self.assertIsInstance(y_wt, weak_tensor.WeakTensor)
    self.assertAllClose(y, z)


@test_util.run_all_in_graph_and_eager_modes
class SwishTest(test_lib.TestCase):
  # [CLASS DOC]
  # Test case (graph and eager modes) for nn_impl.swish on weak tensors: values with and without
  # the beta parameter, and gradients.

  def testValues(self):
    # [FUNCTION DOC]
    # Applies swish to a float32 weak tensor spanning -7 to 7 and asserts a WeakTensor result close
    # to the input times its sigmoid.
    np_values = np.array(
        [np.linspace(-7.0, 0.0, 100),
         np.linspace(0.0, 7.0, 100)],
        dtype=np.float32)
    tf_values = _get_weak_tensor(np_values)
    actual_tf_outputs = nn_impl.swish(tf_values)
    self.assertIsInstance(actual_tf_outputs, weak_tensor.WeakTensor)
    expected_tf_outputs = tf_values * math_ops.sigmoid(tf_values)

    actual_outputs, expected_outputs = self.evaluate(
        [actual_tf_outputs, expected_tf_outputs])

    self.assertAllClose(actual_outputs, expected_outputs)

  def testValuesWithBeta(self):
    # [FUNCTION DOC]
    # Applies swish with beta 0.5 to a float32 weak tensor spanning -7 to 7 and asserts a
    # WeakTensor result close to the input times the sigmoid of half the input.
    np_values = np.array(
        [np.linspace(-7.0, 0.0, 100),
         np.linspace(0.0, 7.0, 100)],
        dtype=np.float32)
    tf_values = _get_weak_tensor(np_values)
    actual_tf_outputs = nn_impl.swish(tf_values, beta=0.5)
    self.assertIsInstance(actual_tf_outputs, weak_tensor.WeakTensor)
    expected_tf_outputs = tf_values * math_ops.sigmoid(0.5 * tf_values)

    actual_outputs, expected_outputs = self.evaluate(
        [actual_tf_outputs, expected_tf_outputs])

    self.assertAllClose(actual_outputs, expected_outputs)

  def testGradients(self):
    # [FUNCTION DOC]
    # Builds random weak float64 data (shape 5x3x4, scaled by 5) and asserts theoretical and
    # numerical gradients of swish agree using the gradient checker.
    shape = [5, 3, 4]
    sigma = 5
    input_values = np.random.randn(*shape) * sigma
    x_tf = _get_weak_tensor(input_values)
    with self.cached_session():
      def f(x):  # pylint: disable=invalid-name
        # [FUNCTION DOC]
        # HELPER: function under differentiation for the enclosing test.
        # Returns swish of x with the default beta.
        return nn_impl.swish(x)

      theoretical, numerical = gradient_checker_v2.compute_gradient(
          f, [x_tf])
      self.assertAllClose(theoretical, numerical)

  def testGradientsWithBeta(self):
    # [FUNCTION DOC]
    # Same as the gradient test but with swish using beta 0.5: asserts theoretical and numerical
    # gradients agree for random weak float64 data.
    shape = [5, 3, 4]
    sigma = 5
    input_values = np.random.randn(*shape) * sigma
    x_tf = _get_weak_tensor(input_values)
    with self.cached_session():
      def f(x):  # pylint: disable=invalid-name
        # [FUNCTION DOC]
        # HELPER: function under differentiation for the enclosing test.
        # Returns swish of x with beta 0.5.
        return nn_impl.swish(x, beta=0.5)

      theoretical, numerical = gradient_checker_v2.compute_gradient(
          f, [x_tf])
      self.assertAllClose(theoretical, numerical)

if __name__ == "__main__":
  ops.set_dtype_conversion_mode("all")
  test_lib.main()
