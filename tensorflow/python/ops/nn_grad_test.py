# [MODULE DOC]
# Unit tests for the gradient functions registered in nn_grad.py. Each test builds a small
# neural-network op graph (softmax, relu6, conv2d, depthwise conv2d, elu, selu, swish), computes
# an analytic gradient (sometimes a second-order gradient) and compares it against a numerical
# estimate using the gradient checker utilities, asserting the maximum error stays small.
# Importing nn_grad here is only for its side effect of registering gradients. No import-time
# side effects beyond that; this is a test module, not a helper.
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
"""Tests for Python ops defined in nn_grad.py."""

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import nn_grad  # pylint: disable=unused-import
from tensorflow.python.ops import nn_impl
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class SoftmaxOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the gradient of the softmax op, in particular its second-order gradient.

  # This test is for bfloat16, but the type has a problem with compute_gradient.
  # TODO(penporn): Change the data type back to bfloat16 once b/157773623 is
  # fixed. (compute_gradient internally converts bfloat16 to float32 for
  # calculation anyway.)
  def testSoftmaxGradGradExtendType(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the comment above says this test is for bfloat16, but the data type used is
    # float32 (asserted inside the inner function), pending a TODO to switch back.
    # Scenario: defines an inner function that records a gradient tape, runs softmax on the
    # input and returns the first-order gradient. The numerical-vs-theoretical gradient of that
    # function (i.e. the softmax second-order gradient) is computed for a 2x4 float32 constant
    # and the maximum error is asserted to be below 1e-4. Takes no arguments, returns nothing.
    with self.cached_session():

      def f(x):
        # [FUNCTION DOC]
        # HELPER: function under test for the softmax second-order gradient check.
        # Takes a tensor x (asserted to be float32; otherwise an assertion error), watches it on a
        # gradient tape, applies softmax and returns the gradient of the softmax output with
        # respect to x. No other side effects.
        assert x.dtype == dtypes.float32
        with backprop.GradientTape() as tape:
          tape.watch(x)
          y = nn_ops.softmax(x)
        return tape.gradient(y, x)

      x = constant_op.constant([[-2, -1, 1, 3], [5, 7, 8, 9]],
                               dtype=dtypes.float32)
      error = gradient_checker_v2.max_error(
          *gradient_checker_v2.compute_gradient(f, [x]))
      self.assertLess(error, 1e-4)


class Relu6OpTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the gradient of the relu6 op (second-order gradient).

  @test_util.run_deprecated_v1
  def testRelu6GradGrad(self):
    # [FUNCTION DOC]
    # Scenario: builds relu6 of a 2x4 float32 constant, takes its gradient graph, then checks the
    # error between analytic and numerical derivative of that gradient with respect to the input,
    # starting from a chosen initial input value that straddles the clipping points 0 and 6.
    # Asserts the error is below 1e-4. Runs in graph mode (deprecated v1 decorator).
    inputs = constant_op.constant(
        [[-2, -1, 1, 3], [5, 7, 8, 9]], dtype=dtypes.float32)
    x_init_value = np.array([[-3.5, -1.5, 2, 4], [4.5, 7.5, 8.5, 11]])
    r = nn_ops.relu6(inputs)
    r_g = gradients_impl.gradients(r, inputs)[0]
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(
          inputs,
          inputs.get_shape().as_list(),
          r_g,
          r_g.get_shape().as_list(),
          x_init_value=x_init_value)
      self.assertLess(error, 1e-4)


class Conv2dOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering gradients of conv2d with respect to input and filter, and the gradient of
  # the backprop-filter computation. Shares a run_test helper that checks gradient error.

  def run_test(self, x, y):
    # [FUNCTION DOC]
    # HELPER: shared gradient-error check for the conv2d tests in this class.
    # Inputs: x, an input tensor, and y, an output tensor that depends on it. Computes the error
    # between analytic and numerical Jacobian of y with respect to x in a test session and asserts
    # it is below 1e-3. Returns nothing.
    with self.test_session():
      error = gradient_checker.compute_gradient_error(x,
                                                      x.get_shape().as_list(),
                                                      y,
                                                      y.get_shape().as_list())
      self.assertLess(error, 1e-3)

  @test_util.run_deprecated_v1
  def testConv2dGradWRTInput(self):
    # [FUNCTION DOC]
    # Scenario: conv2d of a placeholder input (1x4x4x3) with a constant 2x2x3x2 filter, stride 1,
    # SAME padding. Checks the gradient of the output with respect to the input via run_test.
    x = array_ops.placeholder(
        dtype=dtypes.float32, shape=[1, 4, 4, 3], name='input')
    f = constant_op.constant([0.5],
                             dtype=dtypes.float32,
                             shape=[2, 2, 3, 2],
                             name='filter')
    y = nn_ops.conv2d(x, f, [1, 1, 1, 1], 'SAME')
    self.run_test(x, y)

  @test_util.run_deprecated_v1
  def testConv2dGradWRTFilter(self):
    # [FUNCTION DOC]
    # Scenario: conv2d of a constant input (1x4x4x3) with a placeholder 2x2x3x2 filter, stride 1,
    # SAME padding. Checks the gradient of the output with respect to the filter via run_test.
    x = constant_op.constant([0.5],
                             dtype=dtypes.float32,
                             shape=[1, 4, 4, 3],
                             name='input')
    f = array_ops.placeholder(
        dtype=dtypes.float32, shape=[2, 2, 3, 2], name='filter')
    y = nn_ops.conv2d(x, f, [1, 1, 1, 1], 'SAME')
    self.run_test(f, y)

  @test_util.run_deprecated_v1
  def testConv2dBackpropFilterGrad(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says conv2d, but the graph is built with depthwise conv2d, so this
    # duplicates the depthwise backprop-filter test and does not exercise plain conv2d.
    # Scenario: depthwise convolution of a placeholder input with a constant filter; takes the
    # gradients of the output with respect to input and filter, then checks the error of each
    # gradient with respect to the filter and the input respectively via run_test.
    x = array_ops.placeholder(
        dtype=dtypes.float32, shape=[1, 4, 4, 3], name='input')
    f = constant_op.constant([0.5],
                             dtype=dtypes.float32,
                             shape=[2, 2, 3, 2],
                             name='filter')
    strides = [1, 1, 1, 1]
    padding = 'SAME'
    out = nn_impl.depthwise_conv2d(x, f, strides, padding)

    grad_wrt_input = gradients_impl.gradients(out, x)[0]
    self.run_test(f, grad_wrt_input)

    grad_wrt_filter = gradients_impl.gradients(out, f)[0]
    self.run_test(x, grad_wrt_filter)


class DepthwiseConv2dTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering gradients of depthwise conv2d with respect to input and filter, plus the
  # gradient of the backprop-filter computation. Shares a run_test helper.

  def run_test(self, x, y):
    # [FUNCTION DOC]
    # HELPER: shared gradient-error check for the depthwise conv2d tests in this class.
    # Inputs: x, an input tensor, and y, an output tensor depending on it. Computes the error
    # between analytic and numerical Jacobian in a test session and asserts it is below 1e-3.
    # Returns nothing.
    with self.test_session():
      error = gradient_checker.compute_gradient_error(x,
                                                      x.get_shape().as_list(),
                                                      y,
                                                      y.get_shape().as_list())
      self.assertLess(error, 1e-3)

  @test_util.run_deprecated_v1
  def testDepthwiseConv2dGradWRTInput(self):
    # [FUNCTION DOC]
    # Scenario: depthwise conv2d of a placeholder input (1x4x4x3) with a constant 2x2x3x2 filter,
    # stride 1, SAME padding; checks the gradient with respect to the input via run_test.
    x = array_ops.placeholder(
        dtype=dtypes.float32, shape=[1, 4, 4, 3], name='input')
    f = constant_op.constant([0.5],
                             dtype=dtypes.float32,
                             shape=[2, 2, 3, 2],
                             name='filter')
    strides = [1, 1, 1, 1]
    padding = 'SAME'
    y = nn_impl.depthwise_conv2d(x, f, strides, padding)
    self.run_test(x, y)

  @test_util.run_deprecated_v1
  def testDepthwiseConv2dGradWRTFilter(self):
    # [FUNCTION DOC]
    # Scenario: depthwise conv2d of a constant input with a placeholder filter, stride 1, SAME
    # padding; checks the gradient with respect to the filter via run_test.
    x = constant_op.constant([0.5],
                             dtype=dtypes.float32,
                             shape=[1, 4, 4, 3],
                             name='input')
    f = array_ops.placeholder(
        dtype=dtypes.float32, shape=[2, 2, 3, 2], name='filter')
    strides = [1, 1, 1, 1]
    padding = 'SAME'
    y = nn_impl.depthwise_conv2d(x, f, strides, padding)
    self.run_test(f, y)

  @test_util.run_deprecated_v1
  def testDepthwiseConv2dBackpropFilterGrad(self):
    # [FUNCTION DOC]
    # Scenario: depthwise conv2d of a placeholder input and constant filter; takes gradients of
    # the output with respect to the input and the filter, then checks each of those gradient
    # tensors for numerical accuracy (with respect to the filter and input respectively) via
    # run_test, exercising second-order gradients of the depthwise backprop ops.
    x = array_ops.placeholder(
        dtype=dtypes.float32, shape=[1, 4, 4, 3], name='input')
    f = constant_op.constant([0.5],
                             dtype=dtypes.float32,
                             shape=[2, 2, 3, 2],
                             name='filter')
    strides = [1, 1, 1, 1]
    padding = 'SAME'
    out = nn_impl.depthwise_conv2d(x, f, strides, padding)

    grad_wrt_input = gradients_impl.gradients(out, x)[0]
    self.run_test(f, grad_wrt_input)

    grad_wrt_filter = gradients_impl.gradients(out, f)[0]
    self.run_test(x, grad_wrt_filter)


class EluGradOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the second-order gradient of the elu op with respect to the incoming
  # gradient and with respect to the inputs.

  @test_util.run_deprecated_v1
  def testEluGradGradWRTgrad_ys(self):
    # [FUNCTION DOC]
    # Scenario: elu of a 2x4 float32 constant; its gradient is computed with a supplied upstream
    # gradient constant. Checks numerical error of that gradient with respect to the upstream
    # gradient and asserts it is below 1e-4. Graph mode.
    inputs = constant_op.constant(
        [[-2, -1, 1, 3], [5, 7, 8, 9]], dtype=dtypes.float32)
    dummy = constant_op.constant(
        [[3, 1, -1, -2], [9, 8, 7, 6]], dtype=dtypes.float32)

    elu = gen_nn_ops.elu(inputs)
    elu_grad = gradients_impl.gradients(elu, inputs, grad_ys=dummy)[0]
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(
          dummy,
          dummy.shape,
          elu_grad,
          elu_grad.shape)
      self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testEluGradGradWRTinputs(self):
    # [FUNCTION DOC]
    # Scenario: same elu graph as the upstream-gradient test, but checks the numerical error of
    # the gradient with respect to the inputs, asserting it is below 1e-4. Graph mode.
    inputs = constant_op.constant(
        [[-2, -1, 1, 3], [5, 7, 8, 9]], dtype=dtypes.float32)
    dummy = constant_op.constant(
        [[3, 1, -1, -2], [9, 8, 7, 6]], dtype=dtypes.float32)

    elu = gen_nn_ops.elu(inputs)
    elu_grad = gradients_impl.gradients(elu, inputs, grad_ys=dummy)[0]
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(
          inputs,
          inputs.shape,
          elu_grad,
          elu_grad.shape)
      self.assertLess(error, 1e-4)


class SeluGradOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the second-order gradient of the selu op with respect to the incoming
  # gradient and with respect to the inputs.

  @test_util.run_deprecated_v1
  def testSeluGradGradWRTgrad_ys(self):
    # [FUNCTION DOC]
    # Scenario: selu of a 2x4 float32 constant; its gradient is computed with a supplied upstream
    # gradient constant. Checks numerical error of that gradient with respect to the upstream
    # gradient and asserts it is below 1e-4. Graph mode.
    inputs = constant_op.constant(
        [[-2, -1, 1, 3], [5, 7, 8, 9]], dtype=dtypes.float32)
    dummy = constant_op.constant(
        [[3, 1, -1, -2], [9, 8, 7, 6]], dtype=dtypes.float32)

    selu = gen_nn_ops.selu(inputs)
    selu_grad = gradients_impl.gradients(selu, inputs, grad_ys=dummy)[0]
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(
          dummy,
          dummy.shape,
          selu_grad,
          selu_grad.shape)
      self.assertLess(error, 1e-4)

  @test_util.run_deprecated_v1
  def testSeluGradGradWRTinputs(self):
    # [FUNCTION DOC]
    # Scenario: same selu graph as the upstream-gradient test, but checks the numerical error of
    # the gradient with respect to the inputs, asserting it is below 1e-4. Graph mode.
    inputs = constant_op.constant(
        [[-2, -1, 1, 3], [5, 7, 8, 9]], dtype=dtypes.float32)
    dummy = constant_op.constant(
        [[3, 1, -1, -2], [9, 8, 7, 6]], dtype=dtypes.float32)

    selu = gen_nn_ops.selu(inputs)
    selu_grad = gradients_impl.gradients(selu, inputs, grad_ys=dummy)[0]
    with self.cached_session():
      error = gradient_checker.compute_gradient_error(
          inputs,
          inputs.shape,
          selu_grad,
          selu_grad.shape)
      self.assertLess(error, 1e-4)


class SwishGradOpTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the gradient of the swish activation, including its beta parameter.

  def testSwishGrad(self):
    # [FUNCTION DOC]
    # Scenario: swish applied to a 1x4 float32 feature constant with scalar beta 0.25. Computes
    # theoretical and numerical gradients with respect to both features and beta, and asserts the
    # maximum difference is below 1e-4. Runs in a cached session; returns nothing.
    features = constant_op.constant([[-2, -1, 1, 3]],
                                    dtype=dtypes.float32)
    beta = constant_op.constant(0.25, dtype=dtypes.float32)

    with self.cached_session():
      theoretical, numerical = gradient_checker_v2.compute_gradient(
          nn_impl.swish, [features, beta])
      error = gradient_checker_v2.max_error(theoretical, numerical)
      self.assertLess(error, 1e-4)


if __name__ == "__main__":
  test.main()
