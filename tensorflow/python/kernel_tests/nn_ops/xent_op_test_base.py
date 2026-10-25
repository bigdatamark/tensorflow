# [MODULE DOC]
# HELPER: shared base test case for the softmax-cross-entropy-with-logits op, reused by the
# regular xent test module and the deterministic-ops test module.
# Defines XentOpTestBase with a NumPy reference implementation, helpers that run the op forward
# and backward, and the common value, broadcast, gradient, second-gradient, rank-3 and
# zero-dimension tests. Subclasses can override the single-class and labels-broadcast tests via the
# underscore-prefixed helper methods. Importing nn_grad's cross-entropy gradient registers the
# gradient function. The module has no script entry point.
# Copyright 2015-2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for SoftmaxCrossEntropyWithLogits op."""

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
# The following import is required to register the gradient function.
from tensorflow.python.ops.nn_grad import _SoftmaxCrossEntropyWithLogitsGrad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class XentOpTestBase(test.TestCase):
  # [CLASS DOC]
  # Base test case holding the shared tests for the softmax cross-entropy op: value checks for
  # several dtypes, shape mismatch, broadcasting of labels, first and second order gradient
  # accuracy, rank-3 inputs along every axis, and zero-sized input. Concrete subclasses reuse it
  # to run the same checks under different conditions (for example deterministic mode) and may
  # override the single-class and labels-broadcast tests. Stateless.

  def _opFwdBwd(self, labels, logits, axis=-1):
    # [FUNCTION DOC]
    # HELPER: runs the op under test forward and backward for the tests.
    # Inputs: labels (array-like), logits (array-like, converted to a tensor), and the class axis
    # (default last). Computes the softmax cross-entropy loss under a gradient tape that watches
    # the logits and returns a pair of the loss and its gradient with respect to the logits.
    """ Runs the op-under-test both forwards and backwards."""
    logits = ops.convert_to_tensor(logits)  # needed for the gradient tape
    with backprop.GradientTape() as tape:
      tape.watch(logits)
      loss = nn_ops.softmax_cross_entropy_with_logits(
          labels=labels, logits=logits, dim=axis)
    return loss, tape.gradient(loss, logits)

  def _npXent(self, labels, logits, dim=-1):
    # [FUNCTION DOC]
    # HELPER: NumPy reference for softmax cross-entropy.
    # Inputs: label and logit arrays of equal shape and the class axis (default last; minus one
    # means the last axis). Computes stable softmax probabilities, then returns a pair: the loss
    # (negative sum of labels times log of probabilities plus a tiny constant, reduced over the
    # axis) and the backprop value (probabilities minus labels). No side effects.
    if dim == -1:
      dim = len(logits.shape) - 1
    one_only_on_dim = list(logits.shape)
    one_only_on_dim[dim] = 1
    e = np.exp(logits - np.reshape(np.amax(logits, axis=dim), one_only_on_dim))
    probs = e / np.reshape(np.sum(e, axis=dim), one_only_on_dim)
    bp = (probs - labels)
    l = -np.sum(labels * np.log(probs + 1.0e-20), axis=dim)
    return l, bp

  # TODO(b/123860949): The values are constant folded for XLA, so placeholders
  # are needed.
  def _testXent2D(self,
                  np_labels,
                  np_logits,
                  with_placeholders=False,
                  expected_gradient=None):
    # [FUNCTION DOC]
    # HELPER: shared checker for rank-2 inputs.
    # Inputs: NumPy label and logit matrices, a flag (default false) to feed them through
    # placeholders (needed because XLA constant-folds values), and an optional expected gradient
    # that replaces the reference gradient. Runs the op forward and backward in a cached session
    # and asserts loss (with looser half-precision tolerance) and gradient match the reference
    # according to dtype. Returns nothing.
    np_loss, np_gradient = self._npXent(labels=np_labels, logits=np_logits)
    if expected_gradient is not None:
      np_gradient = expected_gradient
    with self.cached_session() as sess:
      if with_placeholders:
        logits_placeholder = array_ops.placeholder(np_logits.dtype)
        labels_placeholder = array_ops.placeholder(np_labels.dtype)
        loss, gradient = self._opFwdBwd(labels_placeholder, logits_placeholder)
        tf_loss, tf_gradient = sess.run([loss, gradient],
                                        feed_dict={
                                            labels_placeholder: np_labels,
                                            logits_placeholder: np_logits
                                        })
      else:
        loss, gradient = self._opFwdBwd(np_labels, np_logits)
        tf_loss, tf_gradient = self.evaluate([loss, gradient])
    self.assertAllCloseAccordingToType(np_loss, tf_loss, half_rtol=1e-2)
    self.assertAllCloseAccordingToType(np_gradient, tf_gradient)

  def _testXentND(self, np_labels, np_logits, dim=-1):
    # [FUNCTION DOC]
    # HELPER: shared checker for inputs of any rank.
    # Inputs: label and logit arrays and the class axis (default last). Computes the reference
    # loss, evaluates the public cross-entropy function with that axis, and asserts closeness
    # according to dtype. Only the loss is checked, not the gradient. Returns nothing.
    np_loss, _ = self._npXent(np_labels, np_logits, dim=dim)
    loss = nn_ops.softmax_cross_entropy_with_logits(
        labels=np_labels, logits=np_logits, dim=dim)
    tf_loss = self.evaluate(loss)
    self.assertAllCloseAccordingToType(np_loss, tf_loss)

  def _testSingleClass(self, expected_gradient=[[2.0], [1.0], [0.0], [0.0]]):
    # [FUNCTION DOC]
    # HELPER: single-class scenario with a parameterisable expected gradient, so subclasses can
    # override the test.
    # KNOWN ISSUE: the default for the expected gradient is a mutable list shared across calls
    # (harmless here because it is only read).
    # Input: expected gradient with respect to logits (default is the non-deterministic result).
    # For float16, float32 and bfloat16, runs the op forward and backward on four rows with one
    # class each (labels minus one, zero, one, one) and asserts every loss is zero and the gradient
    # equals the expected one.
    for dtype in np.float16, np.float32, dtypes.bfloat16.as_numpy_dtype:
      loss, gradient = self._opFwdBwd(
          labels=np.array([[-1.], [0.], [1.], [1.]]).astype(dtype),
          logits=np.array([[1.], [-1.], [0.], [1.]]).astype(dtype))
      self.assertAllClose([0.0, 0.0, 0.0, 0.0], loss)
      self.assertAllClose(expected_gradient, gradient)

  def testSingleClass(self):
    # [FUNCTION DOC]
    # Runs the single-class check with the default expected gradient. Kept as a thin wrapper so a
    # subclass can override just this test.
    """This method is structured to be easily overridden by a child class."""
    self._testSingleClass()

  def testNpXent(self):
    # [FUNCTION DOC]
    # Scenario: validates the NumPy reference on two batches of four logits (uniform logits with a
    # hard target, and one-to-four logits with a soft target). Asserts the reference gradients and
    # losses equal hand-computed constants within 1e-3.
    # We create 2 batches of logits for testing.
    # batch 0 is the boring uniform distribution: 1, 1, 1, 1, with target 3.
    # batch 1 has a bit of difference: 1, 2, 3, 4, with soft targets (1, 2).
    logits = [[1., 1., 1., 1.], [1., 2., 3., 4.]]
    labels = [[0., 0., 0., 1.], [0., .5, .5, 0.]]

    # For batch 0, we expect the uniform distribution: 0.25, 0.25, 0.25, 0.25
    # With a hard target 3, the gradient is [0.25, 0.25, 0.25, -0.75]
    # The loss for this batch is -log(0.25) = 1.386
    #
    # For batch 1, we have:
    # exp(0) = 1
    # exp(1) = 2.718
    # exp(2) = 7.389
    # exp(3) = 20.085
    # SUM = 31.192
    # So we have as probabilities:
    # exp(0) / SUM = 0.032
    # exp(1) / SUM = 0.087
    # exp(2) / SUM = 0.237
    # exp(3) / SUM = 0.644
    # With a soft target (1, 2), the gradient is
    # [0.032, 0.087 - 0.5 = -0.413, 0.237 - 0.5 = -0.263, 0.644]
    # The loss for this batch is [0.5 * -log(0.087), 0.5 * -log(0.237)]
    # = [1.3862, 1.9401]
    np_loss, np_gradient = self._npXent(np.array(labels), np.array(logits))
    self.assertAllClose(
        np.array([[0.25, 0.25, 0.25, -0.75], [0.0321, -0.4129, -0.2632,
                                              0.6439]]),
        np_gradient,
        rtol=1.e-3,
        atol=1.e-3)
    self.assertAllClose(
        np.array([1.3862, 1.9401]), np_loss, rtol=1.e-3, atol=1.e-3)

  # TODO(b/123860949): The values are constant folded for XLA, so placeholders
  # are needed.
  @test_util.run_deprecated_v1
  def _testLabelsBroadcast(self, uniform_labels_gradient):
    # [FUNCTION DOC]
    # HELPER: labels-broadcast scenario with a parameterisable expected gradient, so subclasses
    # can override the test.
    # KNOWN ISSUE: despite being a helper, it carries the deprecated-v1 (graph mode only)
    # decorator, which forces every caller into graph mode.
    # Input: expected logits gradient for the third case. Runs placeholder-based rank-2 checks in
    # float16 for (a) one label row against two logit rows, (b) a single-class case, and (c) three
    # label rows with a single class against three logit rows, using the supplied gradient for the
    # last case. Returns nothing.
    labels = np.array([[0., 0., 0., 1.]]).astype(np.float16)
    logits = np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float16)
    self._testXent2D(labels, logits, with_placeholders=True)
    labels = np.array([[1.]]).astype(np.float16)
    logits = np.array([[1.], [2.]]).astype(np.float16)
    self._testXent2D(labels, logits, with_placeholders=True)
    labels = np.array([[0.], [2.], [0.25]]).astype(np.float16)
    logits = np.array([[1., 1., 1., 1.], [1., 2., 3., 4.],
                       [1., 2., 3., 4.]]).astype(np.float16)
    self._testXent2D(
        labels,
        logits,
        with_placeholders=True,
        expected_gradient=uniform_labels_gradient)

  def testLabelsBroadcast(self):
    # [FUNCTION DOC]
    # Runs the labels-broadcast check with the non-deterministic expected gradient values. Kept as
    # a thin wrapper so a subclass can override just this test.
    """This method is structured to be easily overridden by a child class."""
    self._testLabelsBroadcast(uniform_labels_gradient=[[
        0.25, 0.25, 0.25, 0.25
    ], [-1.968, -1.913, -1.763, -1.355], [-0.218, -0.163, -0.013, 0.394]])

  @test_util.run_deprecated_v1
  def testShapeMismatch(self):
    # [FUNCTION DOC]
    # Scenario: labels of shape 2x3 with logits of shape 2x2. Expects a value error. Graph-mode
    # only (deprecated v1).
    with self.cached_session():
      with self.assertRaises(ValueError):
        self._opFwdBwd(
            labels=[[0., 1., 0.], [1., 0., 0.]], logits=[[0., 1.], [2., 3.]])

  def testHalf(self):
    # [FUNCTION DOC]
    # Scenario: rank-2 value and gradient check on a 2x4 float16 pair (hard and soft labels).
    labels = np.array([[0., 0., 0., 1.], [0., .5, .5, 0.]]).astype(np.float16)
    logits = np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float16)
    self._testXent2D(labels, logits)

  def testBfloat16(self):
    # [FUNCTION DOC]
    # Scenario: rank-2 value and gradient check on the same 2x4 data as the half test, in bfloat16.
    labels = np.array([[0., 0., 0., 1.],
                       [0., .5, .5, 0.]]).astype(dtypes.bfloat16.as_numpy_dtype)
    logits = np.array([[1., 1., 1., 1.],
                       [1., 2., 3., 4.]]).astype(dtypes.bfloat16.as_numpy_dtype)
    self._testXent2D(labels, logits)

  def testFloat(self):
    # [FUNCTION DOC]
    # Scenario: rank-2 value and gradient check on a 2x4 float32 pair.
    labels = np.array([[0., 0., 0., 1.], [0., .5, .5, 0.]]).astype(np.float32)
    logits = np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float32)
    self._testXent2D(labels, logits)

  def testDouble(self):
    # [FUNCTION DOC]
    # Scenario: rank-2 value and gradient check on a 2x4 float64 pair.
    labels = np.array([[0., 0., 0., 1.], [0., .5, .5, 0.]]).astype(np.float64)
    logits = np.array([[1., 1., 1., 1.], [1., 2., 3., 4.]]).astype(np.float64)
    self._testXent2D(labels, logits)

  @test_util.run_deprecated_v1
  def testGradient(self):
    # [FUNCTION DOC]
    # Scenario: float64 3x4 labels and logits; computes the numeric versus analytic gradient error
    # of the loss with respect to the logits and asserts it is below 5e-8. Also inspects the graph
    # and asserts no batch matrix-multiply op exists, i.e. the second derivative is not built when
    # only the first is requested. Graph-mode only (deprecated v1).
    with self.cached_session() as sess:
      labels = constant_op.constant(
          [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5],
          shape=[3, 4],
          dtype=dtypes.float64,
          name="labels")
      logits = constant_op.constant(
          [0.1, 0.2, 0.3, 0.4, 0.1, 0.4, 0.9, 1.6, 0.1, 0.8, 2.7, 6.4],
          shape=[3, 4],
          dtype=dtypes.float64,
          name="logits")
      x = nn_ops.softmax_cross_entropy_with_logits(
          labels=labels, logits=logits, name="xent")
      err = gradient_checker.compute_gradient_error(logits, [3, 4], x, [3])

      # Check that no extra computation gets performed. When only the first
      # derivative is requested, the second derivative must not be computed.
      # So when there is no second derivative, there is no `BatchMatMul` op
      # in the graph.
      op_names = [
          op.op_def.name for op in sess.graph.get_operations() if op.op_def
      ]
      self.assertNotIn("BatchMatMul", op_names)
      self.assertNotIn("BatchMatMulV2", op_names)

    self.assertLess(err, 5e-8)

  @test_util.run_deprecated_v1
  def testGradientLabelWithV2(self):
    # [FUNCTION DOC]
    # Scenario: float64 3x4 data using the version-2 cross-entropy function; checks the numeric
    # versus analytic gradient error with respect to the labels is below 5e-8. Graph-mode only
    # (deprecated v1).
    with self.cached_session():
      labels = constant_op.constant(
          [0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5],
          shape=[3, 4],
          dtype=dtypes.float64,
          name="labels")
      logits = constant_op.constant(
          [0.1, 0.2, 0.3, 0.4, 0.1, 0.4, 0.9, 1.6, 0.1, 0.8, 2.7, 6.4],
          shape=[3, 4],
          dtype=dtypes.float64,
          name="logits")
      x = nn_ops.softmax_cross_entropy_with_logits_v2(
          labels=labels, logits=logits, name="xent")
      err = gradient_checker.compute_gradient_error(labels, [3, 4], x, [3])

    self.assertLess(err, 5e-8)

  @test_util.run_deprecated_v1
  def testSecondGradient(self):
    # [FUNCTION DOC]
    # Scenario: float64 rank-1 data of 12 values (labels scaled to sum less than one); reduces the
    # loss by sum, takes its gradient with respect to the logits, and checks the numeric versus
    # analytic error of that gradient is below 5e-8. When op determinism is not enabled it also
    # asserts the graph contains a batch matrix-multiply op, showing the second derivative uses
    # it. Graph-mode only (deprecated v1).
    with self.cached_session() as sess:
      labels = constant_op.constant([
          0.0, 0.0, 1.0 / 3, 0.0, 1.0 / 3, 0.0, 0.0, 0.0, 0.0, 0.5 / 3, 0.0,
          0.5 / 3
      ],
                                    shape=[12],
                                    dtype=dtypes.float64,
                                    name="labels")
      logits = constant_op.constant(
          [0.1, 0.2, 0.3, 0.4, 0.1, 0.4, 0.9, 1.6, 0.1, 0.8, 2.7, 6.4],
          shape=[12],
          dtype=dtypes.float64,
          name="logits")
      x = nn_ops.softmax_cross_entropy_with_logits(
          labels=labels, logits=logits, name="xent")
      loss = math_ops.reduce_sum(x)

      gradients = gradients_impl.gradients(loss, [logits])[0]

      err = gradient_checker.compute_gradient_error(logits, [12], gradients,
                                                    [12])

      if not config.is_op_determinism_enabled():
        # Check how second derivative is calculated.
        # (it is equivalent to a `BatchMatMul` op being in the graph because of
        # the implementation in SoftmaxCrossEntropyWithLogitsGrad)
        op_names = [
            op.op_def.name for op in sess.graph.get_operations() if op.op_def
        ]
        self.assertIn("BatchMatMulV2", op_names)

    self.assertLess(err, 5e-8)

  def test3D(self):
    # [FUNCTION DOC]
    # Scenario: 3x2x4 float32 labels and logits; checks the loss against the reference for class
    # axes zero, one and minus one.
    labels = np.array([[[0., 0., 0., 1.], [0., 1., 0., 0.]],
                       [[0., 0.5, 0.5, 0.], [0.5, 0.5, 0., 0.]],
                       [[0., 1., 0., 0.], [0., 0., 1., 0.]]]).astype(np.float32)
    logits = np.array([[[1., 1., 1., 1.], [1., 2., 3., 4.]],
                       [[2., 3., 4., 5.], [6., 7., 8., 9.]],
                       [[5., 4., 3., 2.], [1., 2., 3., 4.]]]).astype(np.float32)
    self._testXentND(labels, logits, dim=0)
    self._testXentND(labels, logits, dim=1)
    self._testXentND(labels, logits, dim=-1)

  def testZeroDimension(self):
    # [FUNCTION DOC]
    # Scenario: zero-by-two-by-four float32 labels and logits; asserts the loss equals the
    # reference (an empty array).
    labels = np.zeros([0, 2, 4]).astype(np.float32)
    logits = np.zeros([0, 2, 4]).astype(np.float32)
    np_loss, _ = self._npXent(labels=labels, logits=logits)
    loss = nn_ops.softmax_cross_entropy_with_logits(
        labels=labels, logits=logits)
    tf_loss = self.evaluate(loss)
    self.assertAllEqual(np_loss, tf_loss)
