# [MODULE DOC]
# Unit tests for the legacy (TF1 style) initializer classes and helper functions defined in
# tensorflow/python/ops/init_ops.py.
# The module defines a single test case class that checks, for many initializers, that the
# generated tensor has the requested shape and that its sample statistics (mean, standard
# deviation, maximum, minimum) are close to the theoretical values. It also contains two GPU-only
# tests about device placement of orthogonal initializers.
# It has no import-time side effects other than importing test utilities; running the file as a
# script starts the standard TensorFlow test runner.
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
"""Tests for initializers in init_ops."""

import numpy as np

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape as tensor_shape_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class InitializersTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the initializers in init_ops (random uniform/normal, truncated normal,
  # constant, zeros, ones, identity, orthogonal, LeCun, Glorot and He variants).
  # The whole class is decorated so every test runs in both graph mode and eager mode.
  # Most tests run each initializer on both a plain tuple shape and a tensor-shape object and
  # compare statistics through the shared _runner helper.

  def _runner(self,
              init,
              shape,
              target_mean=None,
              target_std=None,
              target_max=None,
              target_min=None):
    # [FUNCTION DOC]
    # HELPER: shared checker used by the statistics based tests in this class.
    # Calls the given initializer with the shape, evaluates the result, and asserts that the
    # output shape equals the requested shape. Then, for each target that is not null (mean,
    # standard deviation, maximum, minimum), asserts that the measured statistic of the output
    # differs from the target by less than an absolute tolerance of 0.03.
    # Parameters: init is a callable initializer, shape is a tuple or tensor-shape, the four
    # target values are optional numbers (skipped when null). Returns nothing; fails the test on
    # mismatch.
    output = self.evaluate(init(shape))
    self.assertEqual(output.shape, shape)
    lim = 3e-2
    if target_std is not None:
      self.assertGreater(lim, abs(output.std() - target_std))
    if target_mean is not None:
      self.assertGreater(lim, abs(output.mean() - target_mean))
    if target_max is not None:
      self.assertGreater(lim, abs(output.max() - target_max))
    if target_min is not None:
      self.assertGreater(lim, abs(output.min() - target_min))

  def test_uniform(self):
    # [FUNCTION DOC]
    # Scenario: random uniform initializer with bounds -1 and 1 and a fixed seed, on a three
    # dimensional shape given both as a tuple and as a tensor-shape object.
    # Asserts mean near 0, maximum near 1 and minimum near -1.
    shape = (9, 6, 99)
    with self.cached_session():
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        self._runner(
            init_ops.RandomUniform(minval=-1, maxval=1, seed=124),
            tensor_shape,
            target_mean=0.,
            target_max=1,
            target_min=-1)

  def test_normal(self):
    # [FUNCTION DOC]
    # Scenario: random normal initializer with mean 0, standard deviation 1 and a fixed seed, on
    # a three dimensional shape (tuple and tensor-shape forms).
    # Asserts mean near 0 and standard deviation near 1.
    shape = (8, 12, 99)
    with self.cached_session():
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        self._runner(
            init_ops.RandomNormal(mean=0, stddev=1, seed=153),
            tensor_shape,
            target_mean=0.,
            target_std=1)

  def test_truncated_normal(self):
    # [FUNCTION DOC]
    # Scenario: truncated normal initializer with mean 0, standard deviation 1 and a fixed seed,
    # on a three dimensional shape (tuple and tensor-shape forms).
    # Asserts mean near 0, maximum near 2 and minimum near -2 (truncation at two deviations).
    shape = (12, 99, 7)
    with self.cached_session():
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        self._runner(
            init_ops.TruncatedNormal(mean=0, stddev=1, seed=126),
            tensor_shape,
            target_mean=0.,
            target_max=2,
            target_min=-2)

  def test_constant(self):
    # [FUNCTION DOC]
    # Scenario: constant initializer with the scalar value 2 on a three dimensional shape
    # (tuple and tensor-shape forms).
    # Asserts that mean, maximum and minimum of the output are all near 2.
    shape = (5, 6, 4)
    with self.cached_session():
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        self._runner(
            init_ops.Constant(2),
            tensor_shape,
            target_mean=2,
            target_max=2,
            target_min=2)

  def test_lecun_uniform(self):
    # [FUNCTION DOC]
    # Scenario: LeCun uniform initializer with a fixed seed on a four dimensional shape.
    # The expected standard deviation is the square root of one over fan-in, with fan-in obtained
    # from the private fan computation helper. Asserts mean near 0 and that standard deviation.
    shape = (5, 6, 4, 2)
    with self.cached_session():
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        fan_in, _ = init_ops._compute_fans(tensor_shape)
        std = np.sqrt(1. / fan_in)
        self._runner(
            init_ops.lecun_uniform(seed=123),
            tensor_shape,
            target_mean=0.,
            target_std=std)

  def test_glorot_uniform_initializer(self):
    # [FUNCTION DOC]
    # Scenario: Glorot uniform initializer with a fixed seed on a four dimensional shape.
    # Expected standard deviation is the square root of two over the sum of fan-in and fan-out.
    # Asserts mean near 0 and that standard deviation.
    shape = (5, 6, 4, 2)
    with self.cached_session():
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        fan_in, fan_out = init_ops._compute_fans(tensor_shape)
        std = np.sqrt(2. / (fan_in + fan_out))
        self._runner(
            init_ops.glorot_uniform_initializer(seed=123),
            tensor_shape,
            target_mean=0.,
            target_std=std)

  def test_he_uniform(self):
    # [FUNCTION DOC]
    # Scenario: He uniform initializer with a fixed seed on a four dimensional shape.
    # Expected standard deviation is the square root of two over fan-in. Asserts mean near 0 and
    # that standard deviation.
    shape = (5, 6, 4, 2)
    with self.cached_session():
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        fan_in, _ = init_ops._compute_fans(tensor_shape)
        std = np.sqrt(2. / fan_in)
        self._runner(
            init_ops.he_uniform(seed=123),
            tensor_shape,
            target_mean=0.,
            target_std=std)

  def test_lecun_normal(self):
    # [FUNCTION DOC]
    # Scenario: LeCun normal initializer with a fixed seed on a four dimensional shape.
    # Expected standard deviation is the square root of one over fan-in. Asserts mean near 0 and
    # that standard deviation.
    shape = (5, 6, 4, 2)
    with self.cached_session():
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        fan_in, _ = init_ops._compute_fans(tensor_shape)
        std = np.sqrt(1. / fan_in)
        self._runner(
            init_ops.lecun_normal(seed=123),
            tensor_shape,
            target_mean=0.,
            target_std=std)

  def test_glorot_normal_initializer(self):
    # [FUNCTION DOC]
    # Scenario: Glorot normal initializer with a fixed seed on a four dimensional shape.
    # Expected standard deviation is the square root of two over the sum of fan-in and fan-out.
    # Asserts mean near 0 and that standard deviation.
    shape = (5, 6, 4, 2)
    with self.cached_session():
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        fan_in, fan_out = init_ops._compute_fans(tensor_shape)
        std = np.sqrt(2. / (fan_in + fan_out))
        self._runner(
            init_ops.glorot_normal_initializer(seed=123),
            tensor_shape,
            target_mean=0.,
            target_std=std)

  def test_he_normal(self):
    # [FUNCTION DOC]
    # Scenario: He normal initializer with a fixed seed on a four dimensional shape.
    # Expected standard deviation is the square root of two over fan-in. Asserts mean near 0 and
    # that standard deviation.
    shape = (5, 6, 4, 2)
    with self.cached_session():
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        fan_in, _ = init_ops._compute_fans(tensor_shape)
        std = np.sqrt(2. / fan_in)
        self._runner(
            init_ops.he_normal(seed=123),
            tensor_shape,
            target_mean=0.,
            target_std=std)

  def test_Orthogonal(self):
    # [FUNCTION DOC]
    # Scenario: orthogonal initializer with a fixed seed on a 20 by 20 shape (tuple and
    # tensor-shape forms). Asserts only the output shape and that the mean is near 0.
    shape = (20, 20)
    with self.cached_session():
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        self._runner(
            init_ops.Orthogonal(seed=123), tensor_shape, target_mean=0.)

  @test_util.run_gpu_only
  def testVariablePlacementWithOrthogonalInitializer(self):
    # [FUNCTION DOC]
    # GPU-only test (skipped when no GPU is available). In a fresh graph, creates two variables
    # on the first GPU device through the variable-scope getter: one using the orthogonal
    # initializer class and one using the random normal initializer class.
    # Then runs the global variables initializer in a session with soft placement disabled and
    # full tracing, so the test fails if the initializer ops cannot satisfy device/colocation
    # constraints. There are no explicit assertions; success means no placement error.
    with ops.Graph().as_default() as g:
      with ops.device('gpu:0'):
        variable_scope.get_variable(
            name='v', shape=[8, 2], initializer=init_ops.Orthogonal)
        variable_scope.get_variable(
            name='w', shape=[8, 2], initializer=init_ops.RandomNormal)
      run_metadata = config_pb2.RunMetadata()
      run_options = config_pb2.RunOptions(
          trace_level=config_pb2.RunOptions.FULL_TRACE)
      config = config_pb2.ConfigProto(
          allow_soft_placement=False, log_device_placement=True)

      # Note: allow_soft_placement=False will fail whenever we cannot satisfy
      # the colocation constraints.
      with session.Session(config=config, graph=g) as sess:
        sess.run(
            variables.global_variables_initializer(),
            options=run_options,
            run_metadata=run_metadata)

  @test_util.run_gpu_only
  def test_eager_orthogonal_gpu(self):
    # [FUNCTION DOC]
    # GPU-only test (skipped when no GPU is available). In eager mode, creates variables named
    # v and w through the variable-scope getter, with the orthogonal and random normal
    # initializer classes respectively. Asserts that each variable's handle device string
    # mentions a GPU.
    with context.eager_mode():
      v = variable_scope.get_variable(
          name='v', shape=[8, 2], initializer=init_ops.Orthogonal)
      w = variable_scope.get_variable(
          name='w', shape=[8, 2], initializer=init_ops.RandomNormal)
      self.assertTrue('GPU' in v.handle.device)
      self.assertTrue('GPU' in w.handle.device)

  def test_Identity(self):
    # [FUNCTION DOC]
    # Scenario 1: identity initializer on a three dimensional shape; asserts that a value error
    # is raised (only two dimensional shapes are accepted).
    # Scenario 2: on a 3 by 3 shape (tuple and tensor-shape forms); asserts mean near one over
    # the first dimension and maximum near 1, as expected for an identity matrix.
    with self.cached_session():
      shape = (3, 4, 5)
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        with self.assertRaises(ValueError):
          self._runner(
              init_ops.Identity(),
              tensor_shape,
              target_mean=1. / int(tensor_shape[0]),
              target_max=1.)

      shape = (3, 3)
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        self._runner(
            init_ops.Identity(),
            tensor_shape,
            target_mean=1. / int(tensor_shape[0]),
            target_max=1.)

  def test_Zeros(self):
    # [FUNCTION DOC]
    # Scenario: zeros initializer on a 4 by 5 shape (tuple and tensor-shape forms).
    # Asserts the output mean and maximum are both near 0.
    shape = (4, 5)
    with self.cached_session():
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        self._runner(
            init_ops.Zeros(), tensor_shape, target_mean=0., target_max=0.)

  def test_Ones(self):
    # [FUNCTION DOC]
    # Scenario: ones initializer on a 4 by 5 shape (tuple and tensor-shape forms).
    # Asserts the output mean and maximum are both near 1.
    shape = (4, 5)
    with self.cached_session():
      for tensor_shape in [shape, tensor_shape_lib.TensorShape(shape)]:
        self._runner(
            init_ops.Ones(), tensor_shape, target_mean=1., target_max=1.)


if __name__ == '__main__':
  test.main()
