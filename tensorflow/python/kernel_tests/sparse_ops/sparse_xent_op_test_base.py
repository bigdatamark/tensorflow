# [MODULE DOC]
# HELPER: shared base-test module, imported by the regular and the deterministic xent test files.
# Defines the base test class holding the test cases for the sparse softmax cross entropy with
# logits function: a numpy reference implementation, forward and backward comparisons for several
# floating types and label types, invalid-label handling on CPU and GPU, shape and scalar error
# cases, empty and high-dimensional inputs, and first and second derivative checks. Some tests are
# split into an underscore-prefixed implementation taking expected values and a public wrapper
# so that subclasses (the deterministic variant) can override only the expectations. Importing it
# registers the cross entropy gradient functions through the nn_grad module. It has no script
# entry point and is not run on its own.
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
"""Tests for SparseSoftmaxCrossEntropyWithLogits op."""

import numpy as np

from tensorflow.python.eager import backprop as backprop_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import nn_ops
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class SparseXentOpTestBase(test.TestCase):
  # [CLASS DOC]
  # HELPER: shared base test case, meant to be inherited (or aliased) by concrete test modules.
  # Contains the test cases for the sparse softmax cross entropy with logits function, plus
  # overridable hook methods (underscore-prefixed implementations with expectations as inputs).
  # Holds no state. Subclasses may change the expected error messages and gradients.

  def _opFwdBwd(self, labels, logits):
    # [FUNCTION DOC]
    # HELPER: runs the function under test forward and backward.
    # Inputs: labels, class numbers (list or array of integers); logits, scores (list or array of
    # floating numbers with one more dimension than labels). Converts logits to a tensor, records
    # the version-2 sparse softmax cross entropy under a gradient tape that watches the logits.
    # Returns a pair: the loss tensor and its gradient with respect to the logits (the gradient of
    # the summed loss).
    """Runs the op-under-test both forwards and backwards"""
    logits = ops_lib.convert_to_tensor(logits)  # needed for the gradient tape
    with backprop_lib.GradientTape() as tape:
      tape.watch(logits)
      loss = nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
          labels=labels, logits=logits)
    return loss, tape.gradient(loss, logits)

  def _npXent(self, labels, logits):
    # [FUNCTION DOC]
    # HELPER: numpy reference implementation of softmax cross entropy and its gradient.
    # Inputs: labels, integer class numbers (any shape); logits, numpy array of scores whose last
    # dimension is the class dimension. Flattens both to batch by classes, computes a numerically
    # stable softmax (subtracting the row maximum), builds the one-hot label matrix, and
    # returns a pair: the per-example loss (negative log probability of the label, with a tiny
    # constant added to avoid taking the log of zero) as a flat vector, and the gradient matrix
    # (probabilities minus one-hot labels) with shape batch by classes. Raises an index error if a
    # label is out of range. No side effects.
    logits = np.reshape(logits, [-1, logits.shape[-1]])
    labels = np.reshape(labels, [-1])
    batch_dim = 0
    class_dim = 1
    batch_size = logits.shape[batch_dim]
    e = np.exp(logits -
               np.reshape(np.amax(logits, axis=class_dim), [batch_size, 1]))
    probs = e / np.reshape(np.sum(e, axis=class_dim), [batch_size, 1])
    labels_mat = np.zeros_like(probs).astype(probs.dtype)
    labels_mat[np.arange(batch_size), labels] = 1.0
    gradient = (probs - labels_mat)
    loss = -np.sum(labels_mat * np.log(probs + 1.0e-20), axis=1)
    return loss, gradient

  def _testXent(self, np_labels, np_logits):
    # [FUNCTION DOC]
    # HELPER: compares the op under test with the numpy reference for one input pair.
    # Inputs: np_labels and np_logits, numpy arrays of labels and scores. Computes the reference
    # loss and gradient, runs the op forward and backward, and asserts both are close using
    # tolerances chosen by the floating type. Returns nothing.
    np_loss, np_gradient = self._npXent(labels=np_labels, logits=np_logits)
    tf_loss, tf_gradient = self._opFwdBwd(labels=np_labels, logits=np_logits)
    self.assertAllCloseAccordingToType(np_loss, tf_loss)
    self.assertAllCloseAccordingToType(np_gradient, tf_gradient)

  def testSingleClass(self):
    # [FUNCTION DOC]
    # Scenario: for 32-bit and 64-bit integer labels, three examples with a single class (all
    # label zero) and logits one, minus one and zero.
    # Expected: the loss is zero for each example and the gradient is zero (a one-class softmax is
    # always one).
    for label_dtype in np.int32, np.int64:
      tf_loss, tf_gradient = self._opFwdBwd(
          labels=np.array([0, 0, 0]).astype(label_dtype),
          logits=np.array([[1.], [-1.], [0.]]).astype(np.float32))
      self.assertAllClose([0.0, 0.0, 0.0], tf_loss)
      self.assertAllClose([[0.0], [0.0], [0.0]], tf_gradient)

  @test_util.run_gpu_only
  def _testInvalidLabelGPU(self, invalid_label_gradient=np.nan):
    # [FUNCTION DOC]
    # HELPER: overridable implementation of the invalid-label check on GPU; the whole method is
    # decorated to run only when a GPU is available (the skip applies to any caller).
    # Input: invalid_label_gradient, the gradient value expected in rows with invalid labels
    # (default not-a-number; the deterministic variant passes zero).
    # Scenario: four examples with labels four, three, zero and minus one over four classes, where
    # labels four and minus one are out of range. Expected: losses of not-a-number, 1.3862, 3.4420
    # and not-a-number; the gradient rows of invalid labels are filled with the given value and the
    # valid rows hold the usual probabilities minus one-hot values, within a loose tolerance.
    labels = [4, 3, 0, -1]
    logits = [[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 2., 3., 4.],
              [1., 2., 3., 4.]]
    loss, gradient = self._opFwdBwd(labels=labels, logits=logits)
    self.assertAllClose([np.nan, 1.3862, 3.4420, np.nan],
                        loss,
                        rtol=1e-3,
                        atol=1e-3)
    self.assertAllClose(
        [[invalid_label_gradient] * 4, [0.25, 0.25, 0.25, -0.75],
         [-0.968, 0.087, 0.237, 0.6439], [invalid_label_gradient] * 4],
        gradient,
        rtol=1e-3,
        atol=1e-3)

  def testInvalidLabelGPU(self):
    # [FUNCTION DOC]
    # Public test hook that runs the GPU invalid-label check with default expectations (gradient
    # of not-a-number for invalid labels). Structured so that subclasses can override it with
    # different expectations. Skipped when no GPU is available.
    """This method is structured to be easily overridden by a child class."""
    self._testInvalidLabelGPU()

  @test_util.run_in_graph_and_eager_modes(use_gpu=False)
  @test_util.disable_xla("XLA cannot assert inside of a kernel.")
  def _testInvalidLabelCPU(self, expected_regex="Received a label value of"):
    # [FUNCTION DOC]
    # HELPER: overridable implementation of the invalid-label check on CPU; decorated to run in
    # graph and eager modes on CPU, and to be disabled under XLA (which cannot assert inside a
    # kernel).
    # Input: expected_regex, pattern the error message must match (default: a message about
    # receiving a label value out of range).
    # Scenario: four examples with labels four, three, zero and minus one over four classes.
    # Expected: an invalid-argument or unknown error whose message matches the pattern.
    labels = [4, 3, 0, -1]
    logits = [[1., 1., 1., 1.], [1., 1., 1., 1.], [1., 2., 3., 4.],
              [1., 2., 3., 4.]]
    with self.assertRaisesRegex(
        (errors_impl.InvalidArgumentError, errors_impl.UnknownError),
        expected_regex):
      self.evaluate(
          nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
              labels=labels, logits=logits))

  def testInvalidLabelCPU(self):
    # [FUNCTION DOC]
    # Public test hook that runs the CPU invalid-label check with the default expected message.
    # Structured so that subclasses can override it with a different expected message.
    """This method is structured to be easily overridden by a child class."""
    self._testInvalidLabelCPU()

  def testNpXent(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite living in an op test class, this test only checks the numpy reference
    # helper against hand-computed numbers; the TensorFlow op is not exercised.
    # Scenario: two examples over four classes: a uniform one with label three and the scores one
    # to four with label zero.
    # Expected: reference gradients (0.25, 0.25, 0.25, minus 0.75) and (minus 0.968, 0.087, 0.237,
    # 0.6439), and losses 1.3862 and 3.4420, within a loose tolerance.
    # We create 2 batches of logits for testing.
    # batch 0 is the boring uniform distribution: 1, 1, 1, 1, with target 3.
    # batch 1 has a bit of difference: 1, 2, 3, 4, with target 0.
    labels = [3, 0]
    logits = [[1., 1., 1., 1.], [1., 2., 3., 4.]]

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
    # With a hard 1, the gradient is [0.032 - 1.0 = -0.968, 0.087, 0.237, 0.644]
    # The loss for this batch is [1.0 * -log(0.25), 1.0 * -log(0.032)]
    # = [1.3862, 3.4420]
    np_loss, np_gradient = self._npXent(
        labels=np.array(labels), logits=np.array(logits))
    self.assertAllClose(
        np.array([[0.25, 0.25, 0.25, -0.75], [-0.968, 0.087, 0.237, 0.6439]]),
        np_gradient,
        rtol=1.e-3,
        atol=1.e-3)
    self.assertAllClose(
        np.array([1.3862, 3.4420]), np_loss, rtol=1.e-3, atol=1.e-3)

  def testShapeMismatch(self):
    # [FUNCTION DOC]
    # Scenario: call the function with labels of shape 1 by 2 and logits of shape 3 by 2 (label
    # rank equals logits rank, not one less).
    # Expected: a value error saying the labels rank must equal the logits rank minus one.
    with self.assertRaisesRegex(
        ValueError, "`labels.shape.rank` must equal `logits.shape.rank - 1`"):
      nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
          labels=[[0, 2]], logits=[[0., 1.], [2., 3.], [2., 3.]])

  def testScalar(self):
    # [FUNCTION DOC]
    # Scenario: call the function with a scalar label and a scalar logits value.
    # Expected: a value error saying logits cannot be a scalar.
    with self.assertRaisesRegex(ValueError, "`logits` cannot be a scalar"):
      nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
          labels=constant_op.constant(0), logits=constant_op.constant(1.0))

  def _testLabelsPlaceholderScalar(self, expected_error_message):
    # [FUNCTION DOC]
    # HELPER: overridable implementation of the scalar-labels-via-placeholder check.
    # Input: expected_error_message, pattern the op error message must match.
    # Scenario: in a fresh graph and session, labels are an integer placeholder of unknown shape
    # and logits are a constant 1 by 1 matrix; the loss is evaluated with a scalar label fed in.
    # Expected: an op error matching the pattern (the shape is only checked at run time).
    with ops_lib.Graph().as_default(), self.session():
      labels = array_ops.placeholder(np.int32)
      y = nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
          labels=labels, logits=[[7.]])
      with self.assertRaisesOpError(expected_error_message):
        y.eval(feed_dict={labels: 0})

  def testLabelsPlaceholderScalar(self):
    # [FUNCTION DOC]
    # Public test hook that runs the scalar-labels placeholder check expecting an error saying the
    # labels must be one-dimensional. Structured so subclasses can override the expected message.
    """This method is structured to be easily overridden by a child class."""
    self._testLabelsPlaceholderScalar(
        expected_error_message="labels must be 1-D")

  def testVector(self):
    # [FUNCTION DOC]
    # Scenario: call the function with a scalar label zero and a one-element logits vector.
    # Expected: a scalar loss of zero (a single class has probability one).
    loss = nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
        labels=constant_op.constant(0), logits=constant_op.constant([1.0]))
    self.assertAllClose(0.0, loss)

  def testFloat(self):
    # [FUNCTION DOC]
    # Scenario: for 32-bit and 64-bit integer labels, compare the op with the numpy reference on
    # two examples over four classes (labels three and zero) with single-precision logits.
    # Expected: loss and gradient close to the reference.
    for label_dtype in np.int32, np.int64:
      self._testXent(
          np_labels=np.array([3, 0]).astype(label_dtype),
          np_logits=np.array([[1., 1., 1., 1.], [1., 2., 3.,
                                                 4.]]).astype(np.float32))

  def testDouble(self):
    # [FUNCTION DOC]
    # Scenario: for both label integer types, compare the op with the numpy reference on two
    # examples over four classes with double-precision logits (labels zero and three, which
    # differ from the other type tests).
    # Expected: loss and gradient close to the reference.
    for label_dtype in np.int32, np.int64:
      self._testXent(
          np_labels=np.array([0, 3]).astype(label_dtype),
          np_logits=np.array([[1., 1., 1., 1.], [1., 2., 3.,
                                                 4.]]).astype(np.float64))

  def testHalf(self):
    # [FUNCTION DOC]
    # Scenario: for both label integer types, compare the op with the numpy reference on two
    # examples over four classes (labels three and zero) with half-precision logits.
    # Expected: loss and gradient close to the reference within half-precision tolerances.
    for label_dtype in np.int32, np.int64:
      self._testXent(
          np_labels=np.array([3, 0]).astype(label_dtype),
          np_logits=np.array([[1., 1., 1., 1.], [1., 2., 3.,
                                                 4.]]).astype(np.float16))

  def testBfloat16(self):
    # [FUNCTION DOC]
    # Scenario: for both label integer types, compare the op with the numpy reference on two
    # examples over four classes (labels three and zero) with bfloat16 logits.
    # Expected: loss and gradient close to the reference within bfloat16 tolerances.
    for label_dtype in np.int32, np.int64:
      self._testXent(
          np_labels=np.array([3, 0]).astype(label_dtype),
          np_logits=np.array([[1., 1., 1., 1.],
                              [1., 2., 3.,
                               4.]]).astype(dtypes.bfloat16.as_numpy_dtype))

  def testEmpty(self):
    # [FUNCTION DOC]
    # Scenario: compare the op with the numpy reference on an empty batch (zero labels and a
    # zero-by-three logits matrix).
    # Expected: empty loss and gradient that match the reference.
    self._testXent(
        np_labels=np.zeros((0,), dtype=np.int32), np_logits=np.zeros((0, 3)))

  @test_util.run_in_graph_and_eager_modes()
  def testGradient(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager modes): three examples over four classes with double-precision
    # logits; the first derivative of the loss is checked by the numeric gradient checker, with
    # labels captured separately because they are integers.
    # Expected: analytical and numerical Jacobians agree within 5e-8. In graph mode it also
    # asserts that no batched matrix multiply op exists in the graph, i.e. the second derivative
    # was not built when only the first was requested.
    with self.session() as sess:
      labels = constant_op.constant([3, 0, 1], name="labels")
      logits = constant_op.constant(
          [0.1, 0.2, 0.3, 0.4, 0.1, 0.4, 0.9, 1.6, 0.1, 0.8, 2.7, 6.4],
          shape=[3, 4],
          dtype=dtypes.float64,
          name="logits")

      def xent(logits):
        # [FUNCTION DOC]
        # HELPER: wraps the loss as a function of the logits only, so the gradient checker can
        # differentiate it. Input: logits tensor. Returns the per-example loss, using the labels
        # captured from the enclosing test.
        # gradient_checker_v2.computee_gradient doesn't take int32/int64.
        # labels must be of type int32/int64, so passing them separately here.
        return nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
            labels=labels, logits=logits, name="xent")

      analytical, numerical = gradient_checker_v2.compute_gradient(
          xent, [logits])

      if not context.executing_eagerly():
        # Check that no extra computation performed. When only first derivative
        # is requested, second derivative must not be computed. So when there is
        # no second derivative, there is no `BatchMatMul` op in the graph.
        op_names = [
            op.op_def.name for op in sess.graph.get_operations() if op.op_def
        ]
        self.assertNotIn("BatchMatMul", op_names)
        self.assertNotIn("BatchMatMulV2", op_names)

    tol = 5e-8
    self.assertAllClose(analytical, numerical, atol=tol, rtol=tol)

  @test_util.run_in_graph_and_eager_modes()
  def testSecondGradient(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager modes): three examples over four classes with double-precision
    # logits; the first derivative (computed under a gradient tape) is itself checked by the
    # numeric gradient checker, which validates the second derivative.
    # Expected: analytical and numerical Jacobians agree within 5e-8. In graph mode without op
    # determinism, it also asserts that a batched matrix multiply op appears in the graph (the
    # second derivative is built from it).
    with self.session() as sess:
      labels = constant_op.constant([3, 0, 1], name="labels")
      logits = constant_op.constant(
          [0.3, 0.4, 0.1, 1.2, 0.1, 1.9, 0.1, 0.7, 0.8, 0.2, 1.3, 1.3],
          shape=[3, 4],
          dtype=dtypes.float64,
          name="logits")

      def xent_grad(logits):
        # [FUNCTION DOC]
        # HELPER: first derivative of the loss as a function of the logits, for the checker.
        # Input: logits tensor. Watches it under a gradient tape, computes the loss using the
        # labels captured from the enclosing test, and returns the gradient of the loss with
        # respect to the logits (the first and only element of the gradient list).
        with backprop_lib.GradientTape() as tape:
          tape.watch(logits)
          return tape.gradient(
              nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
                  labels=labels, logits=logits, name="xent"), [logits])[0]

      analytical, numerical = gradient_checker_v2.compute_gradient(
          xent_grad, [logits])

      if (not context.executing_eagerly() and
          not config.is_op_determinism_enabled()):
        # Check that second derivative is calculated.
        # (it is equivalent to being `BatchMatMul` op in the graph because of
        # implementation of xentropy grad)
        op_names = [
            op.op_def.name for op in sess.graph.get_operations() if op.op_def
        ]
        self.assertIn("BatchMatMulV2", op_names)

    tol = 5e-8
    self.assertAllClose(analytical, numerical, atol=tol, rtol=tol)

  @test_util.run_in_graph_and_eager_modes()
  def _testHighDim(self, labels, logits):
    # [FUNCTION DOC]
    # HELPER: checks the op on inputs with more than one batch dimension; decorated to run in
    # graph and eager modes.
    # Inputs: labels, nested list of class numbers; logits, nested list of scores with one more
    # dimension. Computes the numpy reference (loss reshaped to the label shape), runs the op for
    # the loss and, under a gradient tape, for the gradient with respect to the logits (reshaped to
    # the reference's flattened shape), and asserts both are close to the reference by type.
    np_loss, np_gradient = self._npXent(
        labels=np.array(labels), logits=np.array(logits))
    # manually reshape loss
    np_loss = np.reshape(np_loss, np.array(labels).shape)
    tf_loss = nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
        labels=labels, logits=logits)
    with backprop_lib.GradientTape() as tape:
      logits = constant_op.constant(logits)
      tape.watch(logits)
      tf_gradient = tape.gradient(
          nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
              labels=labels, logits=logits), [logits])[0]
      tf_gradient = array_ops.reshape(tf_gradient, np_gradient.shape)

    self.assertAllCloseAccordingToType(np_loss, tf_loss)
    self.assertAllCloseAccordingToType(np_gradient, tf_gradient)

  def testHighDim(self):
    # [FUNCTION DOC]
    # Scenario: labels of shape 2 by 1 and logits of shape 2 by 1 by 4 (a batch of two with an
    # extra singleton dimension), checked by the high-dimension helper.
    # Expected: loss and gradient match the numpy reference.
    labels = [[3], [0]]
    logits = [[[1., 1., 1., 1.]], [[1., 2., 3., 4.]]]
    self._testHighDim(labels, logits)

  def testHighDim2(self):
    # [FUNCTION DOC]
    # Scenario: labels of shape 2 by 2 and logits of shape 2 by 2 by 4, checked by the
    # high-dimension helper.
    # Expected: loss and gradient match the numpy reference.
    labels = [[3, 2], [0, 3]]
    logits = [[[1., 1., 1., 1.], [2., 2., 2., 2.]],
              [[1., 2., 3., 4.], [5., 6., 7., 8.]]]
    self._testHighDim(labels, logits)

  def _testScalarHandling(self, expected_regex):
    # [FUNCTION DOC]
    # HELPER: overridable implementation of the check for labels that collapse to a scalar.
    # Input: expected_regex, pattern the invalid-argument error message must match.
    # Scenario: in a fresh graph on CPU, labels are a placeholder of shape unknown-by-1 that is
    # squeezed before use, logits a placeholder of shape unknown-by-3; the run feeds a single
    # example, so the squeezed labels become a scalar. Expected: an invalid-argument error
    # matching the pattern.
    with ops_lib.Graph().as_default(), self.session(use_gpu=False) as sess:
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  expected_regex):
        labels = array_ops.placeholder(dtypes.int32, shape=[None, 1])
        logits = array_ops.placeholder(dtypes.float32, shape=[None, 3])
        ce = nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
            labels=array_ops.squeeze(labels), logits=logits)
        labels_v2 = np.zeros((1, 1), dtype=np.int32)
        logits_v2 = np.random.randn(1, 3)
        sess.run([ce], feed_dict={labels: labels_v2, logits: logits_v2})

  def testScalarHandling(self):
    # [FUNCTION DOC]
    # Public test hook that runs the scalar-handling check expecting an error saying the labels
    # must be one-dimensional. Structured so subclasses can override the expected message.
    """This method is structured to be easily overridden by a child class."""
    self._testScalarHandling(expected_regex=".*labels must be 1-D.*")
