# [MODULE DOC]
# Unit tests for batch normalization and moment-related functions in the nn_impl module:
# batch normalization (original kernel, its re-implementation and the newer function), its
# gradients, sufficient statistics, normalize-moments, moments and weighted moments. Results
# are compared with numpy references. Importing the nn_grad module registers needed gradients.
# Runs through the test main entry point.
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
"""Tests for batch_norm related functionality in tensorflow.ops.nn."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl
import tensorflow.python.ops.nn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class BatchNormalizationTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the batch normalization functions in nn_impl: the deprecated global-normalization
  # op (version one), its Python re-implementation, and the newer batch normalization function
  # (version two), compared against a numpy reference and a plain-ops reference. Also covers
  # gradients via numerical gradient checking, parameters with kept dimensions, arbitrary shapes
  # and mixed precision. Provides numpy and op reference implementations as helper methods.

  def _npBatchNorm(self, x, m, v, beta, gamma, epsilon,
                   scale_after_normalization, shift_after_normalization):
    # [FUNCTION DOC]
    # HELPER: numpy reference implementation of batch normalization for the tests.
    # Normalizes x by subtracting mean m and dividing by the square root of variance v plus
    # epsilon; optionally multiplies by gamma (when scale flag is true) and adds beta (when shift
    # flag is true). Inputs are arrays plus the epsilon and two booleans. Returns the array.
    y = (x - m) / np.sqrt(v + epsilon)
    y = y * gamma if scale_after_normalization else y
    return y + beta if shift_after_normalization else y

  def _opsBatchNorm(self, x, m, v, beta, gamma, epsilon,
                    scale_after_normalization, shift_after_normalization):
    # [FUNCTION DOC]
    # HELPER: reference batch normalization built from elementary tensor ops (subtract, reciprocal
    # square root, multiply, add) so gradients can be compared with the fused implementations.
    # Multiplies by gamma only if the scale flag is true and adds beta only if the shift flag is
    # true. Returns the normalized tensor.
    y = (x - m) * math_ops.rsqrt(v + epsilon)
    if scale_after_normalization:
      y = gamma * y
    return y + beta if shift_after_normalization else y

  def _tfBatchNormV1(self, x, m, v, beta, gamma, epsilon,
                     scale_after_normalization):
    # [FUNCTION DOC]
    # HELPER: invokes the original (version one) global normalization kernel directly. First sets
    # the default graph producer version to eight so the deprecated op is permitted (side effect
    # on the default graph), then returns the op output. Takes no shift flag; beta is always added.
    """Original implementation."""
    test_util.set_producer_version(ops.get_default_graph(), 8)
    return gen_nn_ops._batch_norm_with_global_normalization(
        x, m, v, beta, gamma, epsilon, scale_after_normalization)

  def _tfBatchNormV1BW(self, x, m, v, beta, gamma, epsilon,
                       scale_after_normalization):
    # [FUNCTION DOC]
    # HELPER: calls the Python re-implementation of the version one global normalization
    # (backward-compatible wrapper in nn_impl) and returns the result tensor.
    """Re-implementation of the original kernel for backward compatibility."""
    return nn_impl.batch_norm_with_global_normalization(
        x, m, v, beta, gamma, epsilon, scale_after_normalization)

  def _tfBatchNormV2(self, x, m, v, beta, gamma, epsilon,
                     scale_after_normalization, shift_after_normalization):
    # [FUNCTION DOC]
    # HELPER: calls the newer batch normalization function in nn_impl, passing beta only when the
    # shift flag is true and gamma only when the scale flag is true (otherwise nothing/null),
    # and returns the result tensor.
    """New implementation."""
    return nn_impl.batch_normalization(x, m, v, beta if
                                       shift_after_normalization else None,
                                       gamma if scale_after_normalization else
                                       None, epsilon)

  @test_util.run_deprecated_v1
  def testBatchNorm(self):
    # [FUNCTION DOC]
    # Scenario: random float inputs of shape three by five by four by two with per-channel mean,
    # variance, beta and gamma, run on GPU and CPU settings and all four combinations of the
    # scale and shift flags. Builds the version two, version one re-implementation, version one
    # kernel and plain-op results and a numpy reference. Asserts the plain-op and version two
    # results match numpy and each other; when shifting is on (version one cannot disable it) it
    # also asserts both version one results match numpy and the plain-op result.
    x_shape = [3, 5, 4, 2]
    param_shape = [2]
    x_val = np.random.random_sample(x_shape).astype(np.float32)
    m_val = np.random.random_sample(param_shape).astype(np.float32)
    v_val = np.random.random_sample(param_shape).astype(np.float32)
    beta_val = np.random.random_sample(param_shape).astype(np.float32)
    gamma_val = np.random.random_sample(param_shape).astype(np.float32)
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu) as sess:
        x = constant_op.constant(x_val, name="x")
        m = constant_op.constant(m_val, name="m")
        v = constant_op.constant(v_val, name="v")
        beta = constant_op.constant(beta_val, name="beta")
        gamma = constant_op.constant(gamma_val, name="gamma")
        epsilon = 0.001
        for scale_after_normalization in [True, False]:
          for shift_after_normalization in [True, False]:
            bn2 = self._tfBatchNormV2(x, m, v, beta, gamma, epsilon,
                                      scale_after_normalization,
                                      shift_after_normalization)
            bn1bw = self._tfBatchNormV1BW(x, m, v, beta, gamma, epsilon,
                                          scale_after_normalization)
            bn1 = self._tfBatchNormV1(x, m, v, beta, gamma, epsilon,
                                      scale_after_normalization)
            on = self._opsBatchNorm(x, m, v, beta, gamma, epsilon,
                                    scale_after_normalization,
                                    shift_after_normalization)
            np_bn = self._npBatchNorm(x_val, m_val, v_val, beta_val, gamma_val,
                                      epsilon, scale_after_normalization,
                                      shift_after_normalization)
            tf_bn_v2, tf_bn_v1bw, tf_bn_v1, ops_bn = sess.run(
                [bn2, bn1bw, bn1, on])
            self.assertAllClose(np_bn, ops_bn, atol=0.00001)
            self.assertAllClose(np_bn, tf_bn_v2, atol=0.00001)
            self.assertAllClose(tf_bn_v2, ops_bn, atol=0.00001)
            # shift_after_normalization=False is not supported in v1.
            if shift_after_normalization:
              self.assertAllClose(np_bn, tf_bn_v1bw, atol=0.00001)
              self.assertAllClose(np_bn, tf_bn_v1, atol=0.00001)
              self.assertAllClose(tf_bn_v1, ops_bn, atol=0.00001)
              self.assertAllClose(tf_bn_v1bw, ops_bn, atol=0.00001)

  def _testBatchNormGradient(self,
                             param_index,
                             tag,
                             scale_after_normalization,
                             shift_after_normalization,
                             version,
                             err_tolerance=1e-11):
    # [FUNCTION DOC]
    # HELPER: numerical gradient check for one input of batch normalization.
    # KNOWN ISSUE: for an unknown version it prints a message and raises a value error with no
    # message text.
    # Inputs: index of the parameter to differentiate (0 x, 1 mean, 2 variance, 3 beta, 4 gamma),
    # a tag used for printing, the scale and shift flags, the implementation version (one or
    # two) and a tolerance (default very small). Builds double-precision random inputs with a
    # fixed numpy seed, creates the selected implementation, computes the max error between
    # analytic and numeric Jacobians for that parameter against the output, prints it and asserts
    # it is below the tolerance. Raises a value error for versions other than one or two.
    x_shape = [3, 5, 4, 5]
    param_shape = [5]
    np.random.seed(1)  # Make it reproducible.
    x_val = np.random.random_sample(x_shape).astype(np.float64)
    m_val = np.random.random_sample(param_shape).astype(np.float64)
    v_val = np.random.random_sample(param_shape).astype(np.float64)
    beta_val = np.random.random_sample(param_shape).astype(np.float64)
    gamma_val = np.random.random_sample(param_shape).astype(np.float64)
    with self.cached_session():
      x = constant_op.constant(x_val, name="x")
      m = constant_op.constant(m_val, name="m")
      v = constant_op.constant(v_val, name="v")
      beta = constant_op.constant(beta_val, name="beta")
      gamma = constant_op.constant(gamma_val, name="gamma")
      epsilon = 0.001
      if version == 1:
        output = self._tfBatchNormV1(x, m, v, beta, gamma, epsilon,
                                     scale_after_normalization)
      elif version == 2:
        output = self._tfBatchNormV2(x, m, v, beta, gamma, epsilon,
                                     scale_after_normalization,
                                     shift_after_normalization)
      else:
        print("Invalid version", version)
        raise ValueError()
      all_params = [x, m, v, beta, gamma]
      all_shapes = [x_shape, param_shape, param_shape, param_shape, param_shape]
      err = gradient_checker.compute_gradient_error(all_params[param_index],
                                                    all_shapes[param_index],
                                                    output, x_shape)
    print("Batch normalization v%d %s gradient %s scale and %s shift err = " %
          (version, tag, "with" if scale_after_normalization else "without",
           "with" if shift_after_normalization else "without"), err)
    self.assertLess(err, err_tolerance)

  def _testBatchNormGradientInAllNeedConfigs(self,
                                             param_index,
                                             tag,
                                             err_tolerance=1e-11):
    # [FUNCTION DOC]
    # HELPER: runs the single-parameter gradient check for every valid combination of the scale
    # and shift flags, using version one only when shifting is enabled (version one does not
    # support turning shifting off) and always version two. Inputs: parameter index, tag and
    # tolerance. Returns nothing; failures come from the underlying assertions.
    for scale_after_normalization in [True, False]:
      for shift_after_normalization in [True, False]:
        # shift_after_normalization=False is not supported in version 1.
        for v in ([1, 2] if shift_after_normalization else [2]):
          self._testBatchNormGradient(param_index, tag,
                                      scale_after_normalization,
                                      shift_after_normalization, v,
                                      err_tolerance)

  @test_util.run_deprecated_v1
  def testBatchNormInputGradient(self):
    # [FUNCTION DOC]
    # Checks the numerical gradient of the output with respect to the input x for all valid flag
    # and version combinations; asserts error under the default tight tolerance.
    self._testBatchNormGradientInAllNeedConfigs(0, "x")

  @test_util.run_deprecated_v1
  def testBatchNormMeanGradient(self):
    # [FUNCTION DOC]
    # Checks the numerical gradient with respect to the mean input for all valid flag and version
    # combinations; asserts error under the default tolerance.
    self._testBatchNormGradientInAllNeedConfigs(1, "mean")

  @test_util.run_deprecated_v1
  def testBatchNormVarianceGradient(self):
    # [FUNCTION DOC]
    # Checks the numerical gradient with respect to the variance input for all valid flag and
    # version combinations, with a looser tolerance (about one thousandth) because the dependence
    # on variance is more nonlinear.
    self._testBatchNormGradientInAllNeedConfigs(
        2, "variance", err_tolerance=1e-03)

  @test_util.run_deprecated_v1
  def testBatchNormBetaGradient(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the in-body comment says only scale-enabled cases are tested, but the loop runs
    # with scale both enabled and disabled.
    # Checks the numerical gradient with respect to beta for versions one and two, with scaling
    # both on and off and shifting always on (beta is unused otherwise); asserts error under the
    # default tolerance.
    # Since beta does not exist when scale_after_normalization=False, we only
    # test for scale_after_normalization=True.
    for scale_after_normalization in [True, False]:
      for v in [1, 2]:
        self._testBatchNormGradient(3, "beta", scale_after_normalization, True,
                                    v)

  @test_util.run_deprecated_v1
  def testBatchNormGammaGradient(self):
    # [FUNCTION DOC]
    # Checks the numerical gradient with respect to gamma. Version one is tested with scaling on
    # and off (shift on); version two is tested only with scaling on, shift on and off, because
    # without scaling gamma is unused and its gradient is null, which the checker cannot handle.
    # If scale_after_normalization is False, backprop for gamma in v1
    # will be 0. In version 2 of the API, if scale_after_normalization is False,
    # gamma is not used at all, and the gradient is None, which displeases the
    # gradient checker.
    for scale_after_normalization in [True, False]:
      self._testBatchNormGradient(4, "gamma", scale_after_normalization, True,
                                  1)
    for shift_after_normalization in [True, False]:
      self._testBatchNormGradient(4, "gamma", True, shift_after_normalization,
                                  2)

  @test_util.run_deprecated_v1
  def testBatchNormGradImpl(self):
    # [FUNCTION DOC]
    # Compares the dedicated batch norm gradient kernel against gradients of the plain-op
    # reference. For float inputs of shape seven by five by four by six (fixed seed), on CPU and
    # GPU settings and with scaling on and off, it sets producer version eight, calls the
    # deprecated gradient op, asserts its named outputs equal the tuple unpacking, computes
    # gradients of the reference by automatic differentiation with the same upstream gradient,
    # and asserts each corresponding gradient (input, mean, variance, beta and, when scaling,
    # gamma) is close within a tiny tolerance.
    x_shape = [7, 5, 4, 6]
    param_shape = [6]
    np.random.seed(1)  # Make it reproducible.
    x_val = np.random.random_sample(x_shape).astype(np.float32)
    m_val = np.random.random_sample(param_shape).astype(np.float32)
    v_val = np.random.random_sample(param_shape).astype(np.float32)
    beta_val = np.random.random_sample(param_shape).astype(np.float32)
    gamma_val = np.random.random_sample(param_shape).astype(np.float32)
    backprop_val = np.random.random_sample(x_shape).astype(np.float32)
    for use_gpu in [False, True]:
      with self.cached_session(use_gpu=use_gpu) as sess:
        x = constant_op.constant(x_val, name="x")
        m = constant_op.constant(m_val, name="m")
        v = constant_op.constant(v_val, name="v")
        beta = constant_op.constant(beta_val, name="beta")
        gamma = constant_op.constant(gamma_val, name="gamma")
        backprop = constant_op.constant(backprop_val, name="backprop")
        epsilon = 0.001
        for scale_after_normalization in [True, False]:
          # _batch_norm_with_global_normalization_grad is deprecated in v9
          test_util.set_producer_version(ops.get_default_graph(), 8)
          grad = gen_nn_ops.batch_norm_with_global_normalization_grad(
              x, m, v, gamma, backprop, epsilon, scale_after_normalization)
          dx, dm, dv, db, dg = grad
          self.assertEqual(grad.dx, dx)
          self.assertEqual(grad.dm, dm)
          self.assertEqual(grad.dv, dv)
          self.assertEqual(grad.db, db)
          self.assertEqual(grad.dg, dg)

          on = self._opsBatchNorm(x, m, v, beta, gamma, epsilon,
                                  scale_after_normalization, True)
          odx, odm, odv, odb, odg = gradients_impl.gradients(
              [on], [x, m, v, beta, gamma], [backprop])
          if scale_after_normalization:
            all_grads = self.evaluate(
                [dx, dm, dv, db, dg, odx, odm, odv, odb, odg])
            to_check = ["dx", "dm", "dv", "db", "dg"]
          else:
            all_grads = self.evaluate([dx, dm, dv, db, odx, odm, odv, odb])
            to_check = ["dx", "dm", "dv", "db"]
          for i, _ in enumerate(to_check):
            self.assertAllClose(
                all_grads[i + len(to_check)], all_grads[i], atol=0.000001)

  @test_util.run_deprecated_v1
  def testBatchNormKeepDims(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring talks about moments with the keep-dims option, but the test only
    # exercises batch normalization with reshaped parameters and never calls the moments
    # function. Also the parameter shape written in parentheses is a plain integer, not a tuple.
    # Verifies that mean, variance, beta and gamma given with shape one by one by one by depth
    # produce the same result as parameters of shape depth, for all scale and shift flag
    # combinations, on CPU and GPU settings. Asserts both outputs have the input shape and are
    # close to each other.
    """Test for tf.nn.moments(..., keep_dims=True / False).

    Make sure that parameters with shape (1, 1, 1, depth) yield the same
    result as parameters with shape (depth)
    """
    x_shape = (3, 5, 4, 2)
    param_shape = (2)
    keep_dims_param_shape = (1, 1, 1, 2)
    x_val = np.random.random_sample(x_shape).astype(np.float32)
    m_val = np.random.random_sample(param_shape).astype(np.float32)
    v_val = np.random.random_sample(param_shape).astype(np.float32)
    beta_val = np.random.random_sample(param_shape).astype(np.float32)
    gamma_val = np.random.random_sample(param_shape).astype(np.float32)
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu) as sess:
        x = constant_op.constant(x_val, name="x")
        m = constant_op.constant(m_val, name="m")
        v = constant_op.constant(v_val, name="v")
        beta = constant_op.constant(beta_val, name="beta")
        gamma = constant_op.constant(gamma_val, name="gamma")
        keep_dims_m = array_ops.reshape(
            m, keep_dims_param_shape, name="keep_dims_m")
        keep_dims_v = array_ops.reshape(
            v, keep_dims_param_shape, name="keep_dims_v")
        keep_dims_beta = array_ops.reshape(
            beta, keep_dims_param_shape, name="keep_dims_beta")
        keep_dims_gamma = array_ops.reshape(
            gamma, keep_dims_param_shape, name="keep_dims_gamma")
        epsilon = 0.001
        for scale_after_normalization in [True, False]:
          for shift_after_normalization in [True, False]:
            bn = self._tfBatchNormV2(x, m, v, beta, gamma, epsilon,
                                     scale_after_normalization,
                                     shift_after_normalization)
            keep_dims_bn = self._tfBatchNormV2(x, keep_dims_m, keep_dims_v,
                                               keep_dims_beta, keep_dims_gamma,
                                               epsilon,
                                               scale_after_normalization,
                                               shift_after_normalization)
            tf_batch_norm, keep_dims_tf_batch_norm = sess.run(
                [bn, keep_dims_bn])
            self.assertEqual(x_shape, tf_batch_norm.shape)
            self.assertEqual(x_shape, keep_dims_tf_batch_norm.shape)
            self.assertAllClose(
                tf_batch_norm, keep_dims_tf_batch_norm, atol=0.000001)

  def _testBatchNormArbitraryShapes(self, x_shape, param_shape, atol=0.0001,
                                    dtype=dtypes.float32,
                                    param_dtype=dtypes.float32):
    # [FUNCTION DOC]
    # HELPER: shared check of version two batch normalization for arbitrary shapes.
    # Inputs: shape of x, shape of the (broadcastable) parameters, absolute tolerance (default
    # one ten-thousandth), dtype of x and dtype of the parameters (both default to single
    # precision). Creates random constants, and for CPU and GPU settings and all flag
    # combinations compares the tensor result against the numpy reference, asserting the shapes
    # equal x_shape and values are close within the tolerance.
    numpy_dtype = dtype.as_numpy_dtype
    numpy_param_dtype = param_dtype.as_numpy_dtype
    x_val = np.random.random_sample(x_shape).astype(numpy_dtype)
    m_val = np.random.random_sample(param_shape).astype(numpy_param_dtype)
    v_val = np.random.random_sample(param_shape).astype(numpy_param_dtype)
    beta_val = np.random.random_sample(param_shape).astype(numpy_param_dtype)
    gamma_val = np.random.random_sample(param_shape).astype(numpy_param_dtype)
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu) as sess:
        x = constant_op.constant(x_val, name="x")
        m = constant_op.constant(m_val, name="m")
        v = constant_op.constant(v_val, name="v")
        beta = constant_op.constant(beta_val, name="beta")
        gamma = constant_op.constant(gamma_val, name="gamma")
        epsilon = 0.001
        for scale_after_normalization in [True, False]:
          for shift_after_normalization in [True, False]:
            bn = self._tfBatchNormV2(x, m, v, beta, gamma, epsilon,
                                     scale_after_normalization,
                                     shift_after_normalization)
            np_batch_norm = self._npBatchNorm(x_val, m_val, v_val, beta_val,
                                              gamma_val, epsilon,
                                              scale_after_normalization,
                                              shift_after_normalization)
            [tf_batch_norm] = self.evaluate([bn])
            self.assertEqual(x_shape, np_batch_norm.shape)
            self.assertEqual(x_shape, tf_batch_norm.shape)
            self.assertAllClose(np_batch_norm, tf_batch_norm, atol=atol)

  def testBatchNormArbitraryShapes(self):
    # [FUNCTION DOC]
    # Runs the arbitrary-shape check for two-dimensional inputs with the depth axis in either
    # position, a four-dimensional input with a depth axis in the second position, and a
    # five-dimensional input with two depth axes (looser tolerance).
    """Test for a variety of shapes and moments.

    Batch normalization is expected to work regardless of the position and
    dimensionality of the 'depth' axis/axes.
    """
    self._testBatchNormArbitraryShapes((3, 3), (1, 3))
    self._testBatchNormArbitraryShapes((3, 3), (3, 1))
    self._testBatchNormArbitraryShapes((3, 2, 4, 5), (1, 2, 1, 1))
    self._testBatchNormArbitraryShapes(
        (2, 3, 2, 4, 5), (1, 1, 1, 4, 5), atol=0.005)

  def testBatchNormMixedPrecision(self):
    # [FUNCTION DOC]
    # Runs the arbitrary-shape check with half-precision inputs and single-precision parameters
    # on a two-dimensional shape, with a loose tolerance; asserts the output matches numpy.
    self._testBatchNormArbitraryShapes((3, 3), (1, 3), dtype=dtypes.float16,
                                       param_dtype=dtypes.float32, atol=0.001)


class SufficientStatisticsTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the sufficient statistics function in nn_impl (count, shifted sum, shifted sum of
  # squares and the shift) against a numpy reference, for static and unknown input shapes, with
  # and without a shift and with the keep-dimensions option on and off.

  def _npSuffStats(self, x, axes, shift, keep_dims):
    # [FUNCTION DOC]
    # HELPER: numpy reference for sufficient statistics. Computes the sum and sum of squares of
    # x (minus the shift when one is given) over the given axes, optionally keeping reduced
    # dimensions, and the element count over those axes. Returns a tuple of count, shifted sum,
    # shifted sum of squares and the shift (converted to an array when dimensions are not kept).
    axis = tuple(axes)
    if shift is not None:
      m_ss = np.sum(x - shift, axis=axis, keepdims=keep_dims)
      v_ss = np.sum((x - shift) * (x - shift), axis=axis, keepdims=keep_dims)
    else:
      m_ss = np.sum(x, axis=axis, keepdims=keep_dims)
      v_ss = np.sum(x * x, axis=axis, keepdims=keep_dims)
    count = 1.0
    for d in range(x.ndim):
      if d in set(axes):
        count *= x.shape[d]
    if not keep_dims:
      shift = np.asarray(shift)
    return count, m_ss, v_ss, shift

  def _opSuffStats(self, x, axes, shift, keep_dims):
    # [FUNCTION DOC]
    # HELPER: thin wrapper that calls the library sufficient statistics function with the given
    # tensor, axes, shift and keep-dimensions flag and returns its four outputs.
    return nn_impl.sufficient_statistics(x, axes, shift, keep_dims)

  def _testSuffStats(self, x_shape, axes, shift, keep_dims, has_shape):
    # [FUNCTION DOC]
    # HELPER: shared check for one configuration. Builds a random single-precision input; for CPU
    # and GPU settings feeds it either as a constant with a static shape or through a placeholder
    # of unknown dimensions (when has_shape is false), obtains the four statistics and asserts
    # count, sum and sum of squares (and the shift, when a truthy shift is given) are close to
    # the numpy reference. Note a zero shift is treated as no shift for the shift output check.
    x_val = np.random.random_sample(x_shape).astype(np.float32)
    np_c, np_m, np_v, np_s = self._npSuffStats(x_val, axes, shift, keep_dims)
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu) as sess:
        if has_shape:
          x = constant_op.constant(x_val, name="x")
          x.set_shape(x_shape)
          op_c, op_m, op_v, op_s = self._opSuffStats(x, axes, shift, keep_dims)
          if shift:
            tf_c, tf_m, tf_v, tf_s = self.evaluate([op_c, op_m, op_v, op_s])
          else:
            tf_c, tf_m, tf_v = self.evaluate([op_c, op_m, op_v])
        else:
          x = array_ops.placeholder(
              dtype=dtypes.float32, shape=[None] * len(x_shape), name="x")
          op_c, op_m, op_v, op_s = self._opSuffStats(x, axes, shift, keep_dims)
          if shift:
            tf_c, tf_m, tf_v, tf_s = sess.run([op_c, op_m, op_v, op_s],
                                              feed_dict={x: x_val})
          else:
            tf_c, tf_m, tf_v = sess.run([op_c, op_m, op_v],
                                        feed_dict={x: x_val})
        self.assertAllClose(np_c, tf_c, atol=0.000001)
        self.assertAllClose(np_m, tf_m, atol=0.000001)
        self.assertAllClose(np_v, tf_v, atol=0.000001)
        if shift:
          self.assertAllClose(np_s, tf_s, atol=0.000001)

  @test_util.run_deprecated_v1
  def testSuffStats(self):
    # [FUNCTION DOC]
    # Runs the shared check over every combination of static or unknown shape, keep-dimensions
    # on or off, and no shift or a shift of one, for reductions over axis one or zero of a
    # two-by-three input and axes zero and two of a one-by-two-by-three input.
    for has_shape in [True, False]:
      for keep_dims in [True, False]:
        for shift in [None, 1.0]:
          self._testSuffStats([2, 3], [1], shift, keep_dims, has_shape)
          self._testSuffStats([2, 3], [0], shift, keep_dims, has_shape)
          self._testSuffStats([1, 2, 3], [0, 2], shift, keep_dims, has_shape)


class NormalizeMomentsTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the function that converts sufficient statistics into mean and variance,
  # compared against a numpy reference, with and without a shift.

  def _npNormalizeMoments(self, counts, mean_ss, variance_ss, shift):
    # [FUNCTION DOC]
    # HELPER: numpy reference. Mean is the mean sufficient statistic divided by the count (plus
    # the shift if provided); variance is the variance statistic divided by the count minus the
    # square of the unshifted mean. Returns a pair of mean and variance arrays.
    mean = mean_ss / counts
    variance = variance_ss / counts - mean * mean
    if shift is not None:
      mean += shift
    return mean, variance

  def _opNormalizeMoments(self, counts, mean_ss, variance_ss, shift):
    # [FUNCTION DOC]
    # HELPER: thin wrapper calling the library normalize-moments function and returning its
    # mean and variance tensors.
    return nn_impl.normalize_moments(counts, mean_ss, variance_ss, shift)

  def _testNormalizeMoments(self, shape, shift):
    # [FUNCTION DOC]
    # HELPER: shared check. Builds a count of one, random mean statistic and a squared random
    # variance statistic of the given shape, plus a random shift array if the shift flag is
    # truthy. For CPU and GPU settings it evaluates the library function and asserts mean and
    # variance are close to the numpy reference.
    counts = np.ones([1]).astype(np.float32)
    mean_ss = np.random.random_sample(shape).astype(np.float32)
    variance_ss = np.random.random_sample(shape).astype(np.float32)
    variance_ss *= variance_ss
    if shift:
      shift_v = np.random.random_sample(shape).astype(np.float32)
    else:
      shift_v = None
    npm, npv = self._npNormalizeMoments(counts, mean_ss, variance_ss, shift_v)
    for use_gpu in [True, False]:
      with self.cached_session(use_gpu=use_gpu) as sess:
        tf_counts = constant_op.constant(counts, name="counts")
        tf_mean_ss = constant_op.constant(mean_ss, name="mean_ss")
        tf_variance_ss = constant_op.constant(variance_ss, name="variance_ss")
        if shift:
          tf_shift_v = constant_op.constant(shift_v, name="shift")
        else:
          tf_shift_v = None
        opm, opv = self._opNormalizeMoments(tf_counts, tf_mean_ss,
                                            tf_variance_ss, tf_shift_v)
        tfm, tfv = self.evaluate([opm, opv])
        self.assertAllClose(npm, tfm, atol=0.000001)
        self.assertAllClose(npv, tfv, atol=0.000001)

  def testNormalizeMoments(self):
    # [FUNCTION DOC]
    # Runs the shared check for one-dimensional and two-dimensional shapes, each with no shift
    # and with a shift.
    for shift in [None, 4.0]:
      self._testNormalizeMoments([3], shift)
      self._testNormalizeMoments([2, 3], shift)


class MomentsTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the moments function (mean and variance over chosen axes) in nn_impl, with static
  # and unknown shapes, half and single precision, different axes and numerical gradient checks.
  # The moments call is routed through an overridable method so the weighted-moments subclass
  # can reuse all tests.

  def _unweighted_moments(self, x, axes, keep_dims=False, extra_out_grads=None):
    # [FUNCTION DOC]
    # HELPER: overridable hook that computes mean and variance of x over axes with the library
    # moments function and returns both tensors. The extra_out_grads list is ignored here; the
    # weighted subclass uses it to register extra tensors for gradient checks.
    # Method to compute moments of `x` wrt `axes`.
    #
    # This is exposed so WeightedMomentsTest can inherit the tests and
    # assertions from MomentsTest; the extra_out_grads argument allows
    # its inherited gradient tests to assert gradients against the
    # weights as well as the input values.

    return nn_impl.moments(x, axes, keep_dims=keep_dims)

  def RunMomentTestWithDynamicShape(self, shape, axes, keep_dims, dtype):
    # [FUNCTION DOC]
    # HELPER: shared check using a placeholder of unknown dimensions. Requires a four-dimensional
    # shape (asserted). Computes mean and variance of random normal data fed to the placeholder
    # and asserts they match numpy values (sum over axes divided by element count; variance as
    # mean of squares minus squared mean) within dtype-dependent tolerance. Inputs: shape, axes,
    # keep-dimensions flag and placeholder dtype.
    with self.cached_session():
      # shape = [batch, width, height, depth]
      assert len(shape) == 4

      x_numpy = np.random.normal(size=shape).astype(np.float32)
      x = array_ops.placeholder(dtype, shape=[None] * len(shape))

      mean, var = self._unweighted_moments(x, axes, keep_dims=keep_dims)

      num_elements = np.prod([shape[i] for i in axes])

      ax = tuple(axes)
      expected_mean = np.sum(x_numpy, axis=ax,
                             keepdims=keep_dims) / num_elements
      expected_mean_squared = np.multiply(expected_mean, expected_mean)
      expected_x_squared = np.sum(np.multiply(x_numpy, x_numpy),
                                  axis=ax,
                                  keepdims=keep_dims) / num_elements
      expected_variance = expected_x_squared - expected_mean_squared

      # Check that the moments are correct.
      self.assertAllCloseAccordingToType(
          expected_mean, mean.eval(feed_dict={x: x_numpy}))
      self.assertAllCloseAccordingToType(
          expected_variance, var.eval(feed_dict={x: x_numpy}))

  def RunMomentTest(self, shape, axes, keep_dims, dtype):
    # [FUNCTION DOC]
    # HELPER: shared check using a statically shaped input. Requires a four-dimensional shape
    # (asserted). Casts random normal data to the requested dtype, computes the moments, and
    # compares with numpy values computed in extended precision (to avoid cancellation error)
    # within dtype-dependent tolerance. Depends on the platform providing an extended-precision
    # float type.
    with self.cached_session():
      # shape = [batch, width, height, depth]
      assert len(shape) == 4

      x_numpy = np.random.normal(size=shape).astype(np.float32)
      x = math_ops.cast(constant_op.constant(x_numpy), dtype=dtype)

      # Compute the expected values at high precision since the method
      # is prone to catastrophic cancellation:
      x_numpy = x_numpy.astype(np.float128)

      mean, var = self._unweighted_moments(x, axes, keep_dims=keep_dims)

      num_elements = np.prod([shape[i] for i in axes])

      ax = tuple(axes)
      expected_mean = np.sum(x_numpy, axis=ax,
                             keepdims=keep_dims) / num_elements
      expected_mean_squared = np.multiply(expected_mean, expected_mean)
      expected_x_squared = np.sum(np.multiply(x_numpy, x_numpy),
                                  axis=ax,
                                  keepdims=keep_dims) / num_elements
      expected_variance = expected_x_squared - expected_mean_squared

      # Check that the moments are correct.
      self.assertAllCloseAccordingToType(expected_mean, self.evaluate(mean))
      self.assertAllCloseAccordingToType(expected_variance, self.evaluate(var))

  @test_util.run_deprecated_v1
  def testBasic(self):
    # [FUNCTION DOC]
    # Runs both moment checks (static and unknown shape) reducing over the batch axis only, for
    # keep-dimensions on and off and for single and half precision.
    for keep_dims in [False, True]:
      for dtype in [dtypes.float32, dtypes.float16]:
        self.RunMomentTest(
            shape=[2, 3, 5, 4], axes=[0], keep_dims=keep_dims, dtype=dtype)
        self.RunMomentTestWithDynamicShape(
            shape=[2, 3, 5, 4], axes=[0], keep_dims=keep_dims, dtype=dtype)

  @test_util.run_deprecated_v1
  def testGlobalNormalization(self):
    # [FUNCTION DOC]
    # Runs both moment checks reducing over batch, width and height (everything except depth),
    # for keep-dimensions on and off and for single and half precision.
    for keep_dims in [False, True]:
      for dtype in [dtypes.float32, dtypes.float16]:
        self.RunMomentTest(
            shape=[2, 3, 5, 4],
            axes=[0, 1, 2],
            keep_dims=keep_dims,
            dtype=dtype)
        self.RunMomentTestWithDynamicShape(
            shape=[2, 3, 5, 4],
            axes=[0, 1, 2],
            keep_dims=keep_dims,
            dtype=dtype)

  @test_util.run_deprecated_v1
  def testAxes(self):
    # [FUNCTION DOC]
    # Runs both moment checks reducing over width, height and depth (all but the batch axis),
    # for keep-dimensions on and off and for single and half precision.
    for keep_dims in [False, True]:
      for dtype in [dtypes.float32, dtypes.float16]:
        self.RunMomentTest(
            shape=[2, 3, 5, 4],
            axes=[1, 2, 3],
            keep_dims=keep_dims,
            dtype=dtype)
        self.RunMomentTestWithDynamicShape(
            shape=[2, 3, 5, 4],
            axes=[1, 2, 3],
            keep_dims=keep_dims,
            dtype=dtype)

  def _testGlobalGradient(self, from_y="mean"):
    # [FUNCTION DOC]
    # HELPER: numerical gradient check for the moments. KNOWN ISSUE: if from_y is neither mean
    # nor var, the output variable is never defined and a name error results.
    # Input: which output to differentiate, mean (default) or var. Uses a random double-precision
    # input of shape three by five by four by two reduced over the first three axes, gets the
    # moments through the overridable hook (which may append extra tensors, such as weights, to
    # the list of inputs to check), and for each input asserts the gradient error is below a tiny
    # tolerance, printing the error.
    with self.cached_session():
      x_shape = [3, 5, 4, 2]
      x_val = np.random.random_sample(x_shape).astype(np.float64)
      x = constant_op.constant(x_val)
      x.set_shape(x_shape)

      axes = [0, 1, 2]
      y_shape = [2]  # Depth of x

      inputs_to_compute_gradients_for = [x]

      out_mean, out_var = self._unweighted_moments(
          x, axes, extra_out_grads=inputs_to_compute_gradients_for)
      if from_y == "mean":
        y = out_mean
      elif from_y == "var":
        y = out_var

      for (i, v) in enumerate(inputs_to_compute_gradients_for):
        err = gradient_checker.compute_gradient_error(v,
                                                      v.get_shape().as_list(),
                                                      y, y_shape)
        print("Moments %s gradient err vs input %d = %g" % (from_y, i, err))
        self.assertLess(err, 1e-11)

  @test_util.run_deprecated_v1
  def testMeanGlobalGradient(self):
    # [FUNCTION DOC]
    # Runs the numerical gradient check on the mean output.
    self._testGlobalGradient(from_y="mean")

  @test_util.run_deprecated_v1
  def testVarGlobalGradient(self):
    # [FUNCTION DOC]
    # Runs the numerical gradient check on the variance output.
    self._testGlobalGradient(from_y="var")


class WeightedMomentsTest(MomentsTest):
  # [CLASS DOC]
  # Tests for the weighted moments function. Subclasses the moments tests and overrides the
  # moments hook to use constant unit weights, so every inherited test runs against the weighted
  # implementation; also overrides the shared moment checks to add runs with non-constant
  # weights of many broadcastable shapes, and adds a test for all-zero weights.
  """Tests for nn.weighted_moments.

  Note that this test inherits from MomentsTest, inheriting all its
  test methods!

  It modifies MomentsTest in two ways:

  a) By overriding _unweighted_moments, all the codepaths in
     MomentsTest are executed, but with calls to tf.nn.moments()
     replaced by calls to tf.nn.weighted_moments() with a constant
     weight of 1.

  b) By overriding RunMomentTest and RunMomentTestWithDynamicShape,
     this test adds multiple additional calls to
     RunWeightedMomentsTest() to exercise correctness with
     non-constant weights and varying broadcasting situations. (It
     also continues to call MomentsTest.Run(Weighted)?MomentsTest as
     well.)

  """

  def _unweighted_moments(self, x, axes, keep_dims=False, extra_out_grads=None):
    # [FUNCTION DOC]
    # HELPER: override of the moments hook. Creates a scalar constant weight of one with the
    # dtype of x, appends it to extra_out_grads when that list is given (so gradient checks also
    # cover the weights; mutates the list), and returns the weighted moments mean and variance.
    weights = constant_op.constant(1, dtype=x.dtype)
    if extra_out_grads is not None:
      # We want to assert gradients WRT weights as well as X!
      extra_out_grads.append(weights)
    return nn_impl.weighted_moments(x, axes, weights, keep_dims=keep_dims)

  def RunMomentTest(self, shape, axes, keep_dims, dtype, dynshapes=False):
    # [FUNCTION DOC]
    # HELPER: override of the shared moment check. Optional flag dynshapes (default false)
    # selects the parent's static-shape or unknown-shape check, then additionally runs the
    # weighted check with weights of the same shape as the input, with weights that span a single
    # axis (all others size one) for every axis, and with weights matching a trailing suffix of
    # the input shape of every length.
    if not dynshapes:
      super(WeightedMomentsTest, self).RunMomentTest(shape, axes, keep_dims,
                                                     dtype)
    else:
      super(WeightedMomentsTest, self).RunMomentTestWithDynamicShape(shape,
                                                                     axes,
                                                                     keep_dims,
                                                                     dtype)

    # 1:1 weights and inputs
    self.RunWeightedMomentTest(shape, shape, axes, keep_dims, dtype)

    # Various broadcasting combinations
    for idx in range(len(shape)):
      # try broadcasting weights in all positions
      weight_shape = [1] * len(shape)
      weight_shape[idx] = shape[idx]

      self.RunWeightedMomentTest(shape, weight_shape, axes, keep_dims, dtype)

      # Also try broadcasting with a suffix of length n
      weight_shape = shape[-(idx + 1):]
      self.RunWeightedMomentTest(
          shape, weight_shape, axes, keep_dims, dtype, dynshapes=dynshapes)

  def RunMomentTestWithDynamicShape(self, shape, axes, keep_dims, dtype):
    # [FUNCTION DOC]
    # HELPER: override that simply delegates to the weighted moment check with the dynamic shape
    # flag switched on.
    self.RunMomentTest(shape, axes, keep_dims, dtype, dynshapes=True)

  def RunWeightedMomentTest(self,
                            shape,
                            weights_shape,
                            axes,
                            keep_dims,
                            dtype,
                            dynshapes=False):
    # [FUNCTION DOC]
    # HELPER: core weighted-moments check. Inputs: input shape, weights shape, axes, keep
    # dimensions flag, dtype, and an optional flag to use unknown dimensions for placeholders.
    # Feeds random normal data and positive random weights (computed in extended precision for
    # the numpy reference), runs the weighted moments function and asserts mean and variance
    # match the weighted numpy mean and variance within dtype-dependent tolerance.
    with self.cached_session() as s:
      x_numpy = np.random.normal(size=shape).astype(np.float32)
      weights_numpy = np.absolute(  # weights must be positive
          np.random.normal(
              size=weights_shape, loc=1.0).astype(np.float32))

      # Expand the numpy version to higher precision
      x_numpy = x_numpy.astype(np.float128)
      weights_numpy = weights_numpy.astype(np.float128)

      x_shape = [None] * len(shape) if dynshapes else shape
      weights_shape = ([None] * len(weights_shape) if dynshapes else
                       weights_shape)

      x = array_ops.placeholder(dtype, shape=x_shape)
      weights = array_ops.placeholder(dtype, shape=weights_shape)

      mean, var = nn_impl.weighted_moments(
          x, axes, weights, keep_dims=keep_dims)

      ax = tuple(axes)

      def _np_weighted_sum(v):
        # [FUNCTION DOC]
        # HELPER: numpy helper that multiplies an array by the weights and sums over the axes,
        # honoring the keep-dimensions flag; returns the reduced array.
        return np.sum(weights_numpy * v, axis=ax, keepdims=keep_dims)

      weight_sum = _np_weighted_sum(np.ones_like(x_numpy))
      expected_mean = _np_weighted_sum(x_numpy) / weight_sum
      expected_mean_squared = np.multiply(expected_mean, expected_mean)
      expected_x_squared = (_np_weighted_sum(np.multiply(x_numpy, x_numpy)) /
                            weight_sum)
      expected_variance = expected_x_squared - expected_mean_squared

      mean_v, var_v = s.run([mean, var],
                            feed_dict={x: x_numpy,
                                       weights: weights_numpy})

      self.assertAllCloseAccordingToType(expected_mean, mean_v)
      self.assertAllCloseAccordingToType(expected_variance, var_v)

  def testAllZeroMasks(self):
    # [FUNCTION DOC]
    # Edge case: computes weighted moments of random data with weights that are all zero (shape
    # eight by three by one, reducing over the first two axes). Asserts both mean and variance
    # come out as zeros of shape four, i.e. division by a zero weight sum yields zeros not NaN.
    x = np.random.normal(size=[8, 3, 4]).astype(np.float32)
    weights = np.zeros(shape=[8, 3, 1]).astype(np.float32)
    axes = (0, 1)

    mean, var = nn_impl.weighted_moments(
        x, axes, weights, keep_dims=False)
    self.assertAllClose(mean, np.zeros(shape=[4]))
    self.assertAllClose(var, np.zeros(shape=[4]))

if __name__ == "__main__":
  test.main()
