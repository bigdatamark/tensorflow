# [MODULE DOC]
# Gradient definitions for random sampling ops. Registers, at import time, gradient functions
# for the random gamma op, the stateless random gamma ops (versions 2 and 3) and the stateless
# parameterized truncated normal op, all based on implicit reparameterization (Figurnov et al.,
# 2018). Gradients are produced only for the distribution parameters (shape, seed/key/counter
# and algorithm inputs get no gradient). Also holds small private helpers (leading-dimension
# padding, the shared gamma gradient routine and the normal distribution function).
# Not meant to be called directly; the gradient machinery looks the functions up by op type.
#
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
"""Gradients for operators defined in random_ops.py."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import math_ops


def add_leading_unit_dimensions(x, num_dimensions):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # HELPER: makes distribution parameters broadcastable against samples in the gamma gradients.
  # Inputs: x is a tensor; num_dimensions is a scalar integer tensor (or value) giving how many
  # dimensions of size one to put in front of x's shape. Returns x reshaped with that many
  # leading unit dimensions. No side effects other than adding graph ops.
  new_shape = array_ops.concat(
      [array_ops.ones([num_dimensions], dtype=dtypes.int32),
       array_ops.shape(x)], axis=0)
  return array_ops.reshape(x, new_shape)


@ops.RegisterGradient("RandomGamma")
def _RandomGammaGrad(op: ops.Operation, grad):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # HELPER: registered gradient function for the random gamma op (called by the autodiff engine).
  # KNOWN ISSUE: the inline comment says unit dimensions are appended, but the code puts them in
  # front (leading dimensions), which is what broadcasting against the sample requires.
  # Inputs: op is the forward operation (inputs: shape, alpha; output: sample) and grad is the
  # incoming gradient with respect to the sample. Computes the partial derivative of the sample
  # with respect to alpha by implicit differentiation, multiplies by grad and sums over the
  # leading sample dimensions, all under a control dependency on grad.
  # Returns a pair: nothing for the shape input and the gradient with respect to alpha.
  """Returns the gradient of a Gamma sample w.r.t. alpha.

  The gradient is computed using implicit differentiation
  (Figurnov et al., 2018).

  Args:
    op: A `RandomGamma` operation. We assume that the inputs to the operation
      are `shape` and `alpha` tensors, and the output is the `sample` tensor.
    grad: The incoming gradient `dloss / dsample` of the same shape as
      `op.outputs[0]`.

  Returns:
    A `Tensor` with derivatives `dloss / dalpha`.

  References:
    Implicit Reparameterization Gradients:
      [Figurnov et al., 2018]
      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients)
      ([pdf]
      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients.pdf))
  """
  shape = op.inputs[0]
  alpha = op.inputs[1]
  sample = op.outputs[0]

  with ops.control_dependencies([grad]):
    # Make the parameters alpha broadcastable with samples by appending
    # unit dimensions.
    num_sample_dimensions = array_ops.shape(shape)[0]
    alpha_broadcastable = add_leading_unit_dimensions(
        alpha, num_sample_dimensions)
    partial_a = gen_random_ops.random_gamma_grad(alpha_broadcastable, sample)

    # The first input is shape; the second input is alpha.
    return (None, math_ops.reduce_sum(
        grad * partial_a, axis=math_ops.range(num_sample_dimensions)))


@ops.RegisterGradient("StatelessRandomGammaV2")
def _StatelessRandomGammaV2Grad(op: ops.Operation, grad):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # HELPER: registered gradient function for the stateless random gamma op, version 2.
  # Inputs: op (forward operation with inputs shape, seed, alpha and output sample) and grad (the
  # incoming gradient with respect to the sample). Under a control dependency on grad, delegates
  # to the shared stateless gamma gradient routine. Returns a triple: nothing for shape, nothing
  # for seed, and the gradient with respect to alpha.
  """Returns the gradient of a Gamma sample w.r.t. alpha.

  The gradient is computed using implicit differentiation
  (Figurnov et al., 2018).

  Args:
    op: A `StatelessRandomGamma` operation. We assume that the inputs to the
      operation are `shape`, `seed` and `alpha` tensors, and the output is the
      `sample` tensor.
    grad: The incoming gradient `dloss / dsample` of the same shape as
      `op.outputs[0]`.

  Returns:
    A `Tensor` with derivatives `dloss / dalpha`.

  References:
    Implicit Reparameterization Gradients:
      [Figurnov et al., 2018]
      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients)
      ([pdf]
      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients.pdf))
  """
  shape = op.inputs[0]
  alpha = op.inputs[2]
  sample = op.outputs[0]

  with ops.control_dependencies([grad]):
    return (None, None, _StatelessGammaGradAlpha(shape, alpha, sample, grad))


@ops.RegisterGradient("StatelessRandomGammaV3")
def _StatelessRandomGammaV3Grad(op: ops.Operation, grad):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # HELPER: registered gradient function for the stateless random gamma op, version 3.
  # Inputs: op (forward operation with inputs shape, key, counter, algorithm, alpha and output
  # sample) and grad (incoming gradient with respect to the sample). Under a control dependency
  # on grad, delegates to the shared stateless gamma gradient routine using the fifth input as
  # alpha. Returns a five-element tuple with nothing for the first four inputs and the gradient
  # with respect to alpha last.
  """Returns the gradient of a Gamma sample w.r.t. alpha.

  The gradient is computed using implicit differentiation
  (Figurnov et al., 2018).

  Args:
    op: A `StatelessRandomGamma` operation. We assume that the inputs to the
      operation are `shape`, `key`, `counter`, `alg`, and `alpha` tensors, and
      the output is the `sample` tensor.
    grad: The incoming gradient `dloss / dsample` of the same shape as
      `op.outputs[0]`.

  Returns:
    A `Tensor` with derivatives `dloss / dalpha`.

  References:
    Implicit Reparameterization Gradients:
      [Figurnov et al., 2018]
      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients)
      ([pdf]
      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients.pdf))
  """
  shape = op.inputs[0]
  alpha = op.inputs[4]
  sample = op.outputs[0]

  with ops.control_dependencies([grad]):
    return (None, None, None, None,
            _StatelessGammaGradAlpha(shape, alpha, sample, grad))


def _StatelessGammaGradAlpha(shape, alpha, sample, grad):
  # [FUNCTION DOC]
  # HELPER: shared by the stateless gamma gradient functions.
  # Inputs: shape (requested sample shape, a vector), alpha (concentration tensor), sample (the
  # forward output) and grad (incoming gradient). The number of extra sample dimensions is the
  # length of shape minus the rank of alpha; alpha gets that many leading unit dimensions, the
  # gamma-gradient op gives the partial derivative of the sample with respect to alpha, and the
  # product with grad is summed over the extra leading dimensions.
  # Returns the gradient tensor with the shape of alpha.
  """Returns gradients of a gamma sampler wrt alpha."""
  # Note that the shape handling is slightly different for stateless_gamma,
  # in particular num_sample_dimensions is different.
  num_sample_dimensions = array_ops.shape(shape)[0] - array_ops.rank(alpha)
  # Make the parameters alpha broadcastable with samples by appending
  # unit dimensions.
  alpha_broadcastable = add_leading_unit_dimensions(alpha,
                                                    num_sample_dimensions)
  partial_a = gen_random_ops.random_gamma_grad(alpha_broadcastable, sample)

  # The first two inputs are shape, seed, third input is alpha.
  return math_ops.reduce_sum(
      grad * partial_a, axis=math_ops.range(num_sample_dimensions))


def _Ndtr(x):
  # [FUNCTION DOC]
  # HELPER: standard normal cumulative distribution function for the truncated normal gradient.
  # Input: x, a floating-point tensor. Computes the CDF through the error function, using the
  # complementary error function in the tails for numerical stability (switches on the magnitude
  # of x scaled by half the square root of two). Returns a tensor of the same shape and type as x.
  """Normal distribution function."""
  half_sqrt_2 = constant_op.constant(
      0.5 * np.sqrt(2.), dtype=x.dtype, name="half_sqrt_2")
  w = x * half_sqrt_2
  z = math_ops.abs(w)
  y = array_ops.where(
      z < half_sqrt_2,
      1. + math_ops.erf(w),
      array_ops.where(
          w > 0., 2. - math_ops.erfc(z), math_ops.erfc(z)))
  return 0.5 * y


@ops.RegisterGradient("StatelessParameterizedTruncatedNormal")
def _StatelessParameterizedTruncatedNormalGrad(op: ops.Operation, grad):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # HELPER: registered gradient function for the stateless parameterized truncated normal op.
  # KNOWN ISSUE: the docstring says it returns a list; it actually returns a six-element tuple.
  # Inputs: op (forward operation with inputs shape, seed, mean, stddev, minval, maxval and output
  # sample) and grad (incoming gradient with respect to the sample).
  # Standardizes the sample and bounds, computes the sample's CDF position within the truncation
  # interval (clipped to avoid log of zero), and derives the partial derivatives with respect to
  # mean, stddev, minval and maxval by implicit differentiation. Each is multiplied by grad,
  # summed over the extra leading dimensions introduced by the shape, then reduced over
  # broadcast dimensions and reshaped back to the corresponding parameter's shape.
  # Returns a tuple: nothing for shape and seed, then gradients for mean, stddev, minval, maxval.
  """Returns the gradient of a TruncatedNormal sample w.r.t. parameters.

  The gradient is computed using implicit differentiation
  (Figurnov et al., 2018).

  Args:
    op: A `StatelessParameterizedTruncatedNormal` operation. We assume that the
      inputs to the operation are `shape`, `seed`, `mean`, `stddev`, `minval`,
      and `maxval` tensors, and the output is the `sample` tensor.
    grad: The incoming gradient `dloss / dsample` of the same shape as
      `op.outputs[0]`.

  Returns:
    A list of `Tensor` with derivates with respect to each parameter.

  References:
    Implicit Reparameterization Gradients:
      [Figurnov et al., 2018]
      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients)
      ([pdf]
      (http://papers.nips.cc/paper/7326-implicit-reparameterization-gradients.pdf))
  """
  shape = op.inputs[0]
  mean = op.inputs[2]
  stddev = op.inputs[3]
  minval = op.inputs[4]
  maxval = op.inputs[5]
  sample = op.outputs[0]

  with ops.control_dependencies([grad]):
    minval_std = (minval - mean) / stddev
    maxval_std = (maxval - mean) / stddev
    sample_std = (sample - mean) / stddev

    cdf_sample = (_Ndtr(sample_std) - _Ndtr(minval_std)) / (
        _Ndtr(maxval_std) - _Ndtr(minval_std))

    # Clip to avoid zero argument for log_cdf expression
    tiny = np.finfo(mean.dtype.as_numpy_dtype).tiny
    eps = np.finfo(mean.dtype.as_numpy_dtype).eps
    cdf_sample = clip_ops.clip_by_value(cdf_sample, tiny, 1 - eps)

    dmaxval = math_ops.exp(0.5 * (sample_std ** 2 - maxval_std ** 2) +
                           math_ops.log(cdf_sample))
    dminval = math_ops.exp(0.5 * (sample_std ** 2 - minval_std ** 2) +
                           math_ops.log1p(-cdf_sample))
    dmean = array_ops.ones_like(sample_std)
    dstddev = sample_std

    # Reduce over extra dimensions caused by `shape`. We need to get the
    # difference in rank from shape vs. the broadcasted rank.

    mean_shape = array_ops.shape(mean)
    stddev_shape = array_ops.shape(stddev)
    minval_shape = array_ops.shape(minval)
    maxval_shape = array_ops.shape(maxval)

    broadcast_shape = array_ops.broadcast_dynamic_shape(
        mean_shape, stddev_shape)
    broadcast_shape = array_ops.broadcast_dynamic_shape(
        minval_shape, broadcast_shape)
    broadcast_shape = array_ops.broadcast_dynamic_shape(
        maxval_shape, broadcast_shape)
    extra_dims = math_ops.range(
        array_ops.size(shape) - array_ops.size(broadcast_shape))

    grad_mean = math_ops.reduce_sum(grad * dmean, axis=extra_dims)
    grad_stddev = math_ops.reduce_sum(grad * dstddev, axis=extra_dims)
    grad_minval = math_ops.reduce_sum(grad * dminval, axis=extra_dims)
    grad_maxval = math_ops.reduce_sum(grad * dmaxval, axis=extra_dims)

    _, rmean = gen_array_ops.broadcast_gradient_args(
        broadcast_shape, mean_shape)
    _, rstddev = gen_array_ops.broadcast_gradient_args(
        broadcast_shape, stddev_shape)
    _, rminval = gen_array_ops.broadcast_gradient_args(
        broadcast_shape, minval_shape)
    _, rmaxval = gen_array_ops.broadcast_gradient_args(
        broadcast_shape, maxval_shape)

    grad_mean = array_ops.reshape(
        math_ops.reduce_sum(grad_mean, axis=rmean, keepdims=True), mean_shape)

    grad_stddev = array_ops.reshape(
        math_ops.reduce_sum(grad_stddev, axis=rstddev, keepdims=True),
        stddev_shape)

    grad_minval = array_ops.reshape(
        math_ops.reduce_sum(grad_minval, axis=rminval, keepdims=True),
        minval_shape)

    grad_maxval = array_ops.reshape(
        math_ops.reduce_sum(grad_maxval, axis=rmaxval, keepdims=True),
        maxval_shape)

    # The first two inputs are shape.
    return (None, None, grad_mean, grad_stddev, grad_minval, grad_maxval)
