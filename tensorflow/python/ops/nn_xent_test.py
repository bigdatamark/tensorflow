# [MODULE DOC]
# Unit tests for the sigmoid cross entropy and weighted cross entropy loss functions of the
# neural-network ops. Each test class compares the TensorFlow results against a pure Python
# reference implementation (with probabilities clipped to a small epsilon), checks op naming,
# numeric gradients and shape validation. Not a helper module; no import-time side effects other
# than importing the gradient registrations.
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
"""Tests for cross entropy related functionality in tensorflow.ops.nn."""

import math

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_impl
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test

exp = math.exp
log = math.log


class SigmoidCrossEntropyWithLogitsTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the sigmoid cross entropy with logits loss: naming, forward values in float32 and
  # float16 on GPU/CPU settings, multi-dimensional input, gradient accuracy, gradient at zero
  # logits and shape mismatch errors. Shares helper methods that build inputs and reference losses.

  def _SigmoidCrossEntropyWithLogits(self, logits, targets):
    # [FUNCTION DOC]
    # HELPER: reference implementation for the sigmoid cross entropy tests.
    # Takes two equal-length Python lists (logits and targets, asserted equal in length) and
    # returns a list of per-element losses: the sigmoid prediction is clipped to the range
    # epsilon to one minus epsilon (epsilon 0.0001) and the standard binary cross entropy applied.
    assert len(logits) == len(targets)
    pred = [1 / (1 + exp(-x)) for x in logits]
    eps = 0.0001
    pred = [min(max(p, eps), 1 - eps) for p in pred]
    return [-z * log(y) - (1 - z) * log(1 - y) for y, z in zip(pred, targets)]

  def _Inputs(self, x=None, y=None, dtype=dtypes.float64, sizes=None):
    # [FUNCTION DOC]
    # HELPER: builds logits, targets and expected losses for the sigmoid tests.
    # Inputs: x and y are optional lists of logits and targets (defaults are fixed 8-element
    # lists, including extreme logits of -100 and 100), dtype (default float64), sizes (optional
    # shape, default a flat vector of the list length). Asserts x and y have equal length.
    # Returns constant logits tensor, constant targets tensor and a numpy array of expected
    # losses reshaped to the sizes.
    x = [-100, -2, -2, 0, 2, 2, 2, 100] if x is None else x
    y = [0, 0, 1, 0, 0, 1, 0.5, 1] if y is None else y
    assert len(x) == len(y)
    sizes = sizes if sizes else [len(x)]
    logits = constant_op.constant(x, shape=sizes, dtype=dtype, name="logits")
    targets = constant_op.constant(y, shape=sizes, dtype=dtype, name="targets")
    losses = np.array(self._SigmoidCrossEntropyWithLogits(x, y)).reshape(*sizes)
    return logits, targets, losses

  @test_util.run_deprecated_v1
  def testConstructionNamed(self):
    # [FUNCTION DOC]
    # Builds the sigmoid cross entropy loss with an explicit name in graph mode and asserts the
    # resulting op carries exactly that name.
    with self.cached_session():
      logits, targets, _ = self._Inputs()
      loss = nn_impl.sigmoid_cross_entropy_with_logits(
          labels=targets, logits=logits, name="mylogistic")
    self.assertEqual("mylogistic", loss.op.name)

  def testLogisticOutput(self):
    # [FUNCTION DOC]
    # For both GPU and CPU session settings and float32 and float16 inputs, evaluates the sigmoid
    # cross entropy on the default vector inputs and asserts it matches the reference losses
    # within an absolute tolerance of 0.001.
    for use_gpu in [True, False]:
      for dtype in [dtypes.float32, dtypes.float16]:
        with self.cached_session(use_gpu=use_gpu):
          logits, targets, losses = self._Inputs(dtype=dtype)
          loss = nn_impl.sigmoid_cross_entropy_with_logits(
              labels=targets, logits=logits)
          np_loss = np.array(losses).astype(np.float32)
          tf_loss = self.evaluate(loss)
        self.assertAllClose(np_loss, tf_loss, atol=0.001)

  def testLogisticOutputMultiDim(self):
    # [FUNCTION DOC]
    # Same as the plain output test but with inputs shaped 2x2x2; asserts the loss keeps the
    # shape and matches the reference within 0.001 for float32 and float16 on both device settings.
    for use_gpu in [True, False]:
      for dtype in [dtypes.float32, dtypes.float16]:
        with self.cached_session(use_gpu=use_gpu):
          logits, targets, losses = self._Inputs(dtype=dtype, sizes=[2, 2, 2])
          loss = nn_impl.sigmoid_cross_entropy_with_logits(
              labels=targets, logits=logits)
          np_loss = np.array(losses).astype(np.float32)
          tf_loss = self.evaluate(loss)
        self.assertAllClose(np_loss, tf_loss, atol=0.001)

  @test_util.run_deprecated_v1
  def testGradient(self):
    # [FUNCTION DOC]
    # Numerically checks the gradient of the sigmoid cross entropy with respect to logits on
    # 4x2 float64 inputs, prints the error, and asserts it is below 1e-7.
    sizes = [4, 2]
    with self.cached_session():
      logits, targets, _ = self._Inputs(sizes=sizes)
      loss = nn_impl.sigmoid_cross_entropy_with_logits(
          labels=targets, logits=logits)
      err = gradient_checker.compute_gradient_error(logits, sizes, loss, sizes)
    print("logistic loss gradient err = ", err)
    self.assertLess(err, 1e-7)

  @test_util.run_deprecated_v1
  def testGradientAtZero(self):
    # [FUNCTION DOC]
    # With zero logits and targets zero and one, computes the symbolic gradient of the loss with
    # respect to the logits and asserts it equals 0.5 and -0.5 (sigmoid minus target).
    with self.cached_session():
      logits = constant_op.constant([0.0, 0.0], dtype=dtypes.float64)
      targets = constant_op.constant([0.0, 1.0], dtype=dtypes.float64)
      loss = nn_impl.sigmoid_cross_entropy_with_logits(
          labels=targets, logits=logits)
      grads = gradients_impl.gradients(loss, logits)[0].eval()
    self.assertAllClose(grads, [0.5, -0.5])

  def testShapeError(self):
    # [FUNCTION DOC]
    # Passes labels of shape 3 and logits of shape 1x2 and asserts a value error whose message
    # says they must have the same shape.
    with self.assertRaisesRegex(ValueError, "must have the same shape"):
      nn_impl.sigmoid_cross_entropy_with_logits(labels=[1, 2, 3],
                                                logits=[[2, 1]])


class WeightedCrossEntropyTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the weighted cross entropy with logits loss (positive-class weight): naming, forward
  # values, multi-dimensional input, gradient accuracy and shape mismatch errors. Uses helper
  # methods to build inputs and a pure Python reference loss.

  def _WeightedCrossEntropy(self, logits, targets, pos_coeff):
    # [FUNCTION DOC]
    # HELPER: reference implementation for the weighted cross entropy tests.
    # Takes equal-length lists of logits and targets plus a positive-class coefficient; clips the
    # sigmoid prediction to epsilon (0.0001) to one minus epsilon and returns the list of losses
    # in which the positive term is multiplied by the coefficient.
    assert len(logits) == len(targets)
    pred = [1 / (1 + exp(-x)) for x in logits]
    eps = 0.0001
    pred = [min(max(p, eps), 1 - eps) for p in pred]
    return [
        -z * pos_coeff * log(y) - (1 - z) * log(1 - y)
        for y, z in zip(pred, targets)
    ]

  def _Inputs(self, x=None, y=None, q=3.0, dtype=dtypes.float64, sizes=None):
    # [FUNCTION DOC]
    # HELPER: builds logits, targets, positive weight and expected losses.
    # Inputs: x, y optional equal-length lists (defaults are fixed 8-element lists), q the positive
    # weight (default 3.0), dtype (default float64), sizes optional shape (default flat).
    # Returns constant logits, constant targets, the weight q, and a numpy array of expected
    # losses reshaped to the sizes.
    x = [-100, -2, -2, 0, 2, 2, 2, 100] if x is None else x
    y = [0, 0, 1, 0, 0, 1, 0.5, 1] if y is None else y
    assert len(x) == len(y)
    sizes = sizes if sizes else [len(x)]
    logits = constant_op.constant(x, shape=sizes, dtype=dtype, name="logits")
    targets = constant_op.constant(y, shape=sizes, dtype=dtype, name="targets")
    losses = np.array(self._WeightedCrossEntropy(x, y, q)).reshape(*sizes)
    return logits, targets, q, losses

  @test_util.run_deprecated_v1
  def testConstructionNamed(self):
    # [FUNCTION DOC]
    # Builds the weighted cross entropy with an explicit name in graph mode and asserts the
    # resulting op has that name.
    with self.cached_session():
      logits, targets, pos_weight, _ = self._Inputs()
      loss = nn_impl.weighted_cross_entropy_with_logits(
          targets=targets, logits=logits, pos_weight=pos_weight, name="mybce")
    self.assertEqual("mybce", loss.op.name)

  def testOutput(self):
    # [FUNCTION DOC]
    # For GPU and CPU session settings, evaluates the weighted loss on float32 default inputs
    # and asserts it matches the reference within an absolute tolerance of 0.001.
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu):
        logits, targets, pos_weight, losses = self._Inputs(dtype=dtypes.float32)
        loss = nn_impl.weighted_cross_entropy_with_logits(
            targets=targets, logits=logits, pos_weight=pos_weight)
        np_loss = np.array(losses).astype(np.float32)
        tf_loss = self.evaluate(loss)
      self.assertAllClose(np_loss, tf_loss, atol=0.001)

  def testOutputMultiDim(self):
    # [FUNCTION DOC]
    # Same as the output test but with 2x2x2 float32 inputs; asserts the result matches the
    # reference within 0.001 on both device settings.
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu):
        logits, targets, pos_weight, losses = self._Inputs(
            dtype=dtypes.float32, sizes=[2, 2, 2])
        loss = nn_impl.weighted_cross_entropy_with_logits(
            targets=targets, logits=logits, pos_weight=pos_weight)
        np_loss = np.array(losses).astype(np.float32)
        tf_loss = self.evaluate(loss)
      self.assertAllClose(np_loss, tf_loss, atol=0.001)

  @test_util.run_deprecated_v1
  def testGradient(self):
    # [FUNCTION DOC]
    # Numerically checks the gradient of the weighted loss with respect to logits on 4x2 float64
    # inputs, prints the error and asserts it is below 1e-7.
    sizes = [4, 2]
    with self.cached_session():
      logits, targets, pos_weight, _ = self._Inputs(sizes=sizes)
      loss = nn_impl.weighted_cross_entropy_with_logits(
          targets=targets, logits=logits, pos_weight=pos_weight)
      err = gradient_checker.compute_gradient_error(logits, sizes, loss, sizes)
    print("logistic loss gradient err = ", err)
    self.assertLess(err, 1e-7)

  def testShapeError(self):
    # [FUNCTION DOC]
    # Passes targets of shape 3 and logits of shape 1x2 with a positive weight and asserts a value
    # error saying they must have the same shape.
    with self.assertRaisesRegex(ValueError, "must have the same shape"):
      nn_impl.weighted_cross_entropy_with_logits(
          targets=[1, 2, 3], logits=[[2, 1]], pos_weight=2.0)


if __name__ == "__main__":
  test.main()
