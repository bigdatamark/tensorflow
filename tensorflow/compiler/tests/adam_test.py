# [MODULE DOC]
# Test module for the Adam training optimizer running on an XLA test device. It contains a
# reference implementation of one Adam step written with array math, and a test class that runs
# several optimizer steps and compares variables against that reference. It exposes no public
# API; running it as a script starts the test runner. No import-time side effects.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Adam."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adam


def adam_update_numpy(param,
                      g_t,
                      t,
                      m,
                      v,
                      alpha=0.001,
                      beta1=0.9,
                      beta2=0.999,
                      epsilon=1e-8):
  # [FUNCTION DOC]
  # HELPER: reference implementation used by the Adam tests to compute expected values.
  # Computes one Adam update with array math. Inputs: param (current parameter array), g_t
  # (gradient array), t (1-based step number), m and v (previous first and second moment
  # estimates), and optional alpha (learning rate, default 0.001), beta1 (0.9), beta2 (0.999)
  # and epsilon (1e-8). The step size is bias-corrected using the step number. Returns a tuple of
  # the new parameter, new first moment and new second moment. Pure function, no side effects.
  alpha_t = alpha * np.sqrt(1 - beta2**t) / (1 - beta1**t)

  m_t = beta1 * m + (1 - beta1) * g_t
  v_t = beta2 * v + (1 - beta2) * g_t * g_t

  param_t = param - alpha_t * m_t / (np.sqrt(v_t) + epsilon)
  return param_t, m_t, v_t


class AdamOptimizerTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the Adam optimizer. Each test trains two small resource variables with
  # gradients fed through placeholders for three steps and compares them, step by step, with the
  # numpy reference implementation, also checking the optimizer's beta power accumulators.
  # Float16 and bfloat16 are skipped because of excessive precision requirements.

  def testBasic(self):
    # [FUNCTION DOC]
    # For each float and complex type except half precision: enables resource variables in the
    # variable scope, creates two variables, runs three Adam steps with default hyperparameters
    # and asserts after each step that the beta power accumulators equal the expected powers and
    # that the variables match the reference computation.
    for dtype in self.float_types | self.complex_types:
      # TODO: test fails for float16 due to excessive precision requirements.
      if dtype in [np.float16, dtypes.bfloat16.as_numpy_dtype]:
        continue
      with self.session(), self.test_scope():
        variable_scope.get_variable_scope().set_use_resource(True)

        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype)

        var0 = resource_variable_ops.ResourceVariable(var0_np)
        var1 = resource_variable_ops.ResourceVariable(var1_np)
        grads0 = array_ops.placeholder(dtype)
        grads1 = array_ops.placeholder(dtype)
        opt = adam.AdamOptimizer()
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        beta1_power, beta2_power = opt._get_beta_accumulators()

        # Run 3 steps of Adam
        for t in range(1, 4):
          self.assertAllCloseAccordingToType(0.9**t, self.evaluate(beta1_power))
          self.assertAllCloseAccordingToType(0.999**t,
                                             self.evaluate(beta2_power))
          update.run(feed_dict={grads0: grads0_np, grads1: grads1_np})

          var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  def testTensorLearningRate(self):
    # [FUNCTION DOC]
    # Same as the basic test but the optimizer is built with the learning rate given as a
    # constant tensor (0.001) instead of a number. Asserts the beta powers and variable values
    # over three steps match the numpy reference.
    for dtype in self.float_types | self.complex_types:
      # TODO: test fails for float16 due to excessive precision requirements.
      if dtype in [np.float16, dtypes.bfloat16.as_numpy_dtype]:
        continue
      with self.session(), self.test_scope():
        variable_scope.get_variable_scope().set_use_resource(True)

        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype)

        var0 = resource_variable_ops.ResourceVariable(var0_np)
        var1 = resource_variable_ops.ResourceVariable(var1_np)
        grads0 = array_ops.placeholder(dtype)
        grads1 = array_ops.placeholder(dtype)
        opt = adam.AdamOptimizer(constant_op.constant(0.001))
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        beta1_power, beta2_power = opt._get_beta_accumulators()

        # Run 3 steps of Adam
        for t in range(1, 4):
          self.assertAllCloseAccordingToType(0.9**t, self.evaluate(beta1_power))
          self.assertAllCloseAccordingToType(0.999**t,
                                             self.evaluate(beta2_power))
          update.run(feed_dict={grads0: grads0_np, grads1: grads1_np})

          var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  def testSharing(self):
    # [FUNCTION DOC]
    # Applies one optimizer twice to the same variables, creating two update operations that
    # share slots and beta accumulators. Runs three steps, alternating between the two updates
    # (second update on odd steps, first on even steps), and asserts the beta powers and the
    # variables match a single-optimizer numpy reference.
    for dtype in self.float_types | self.complex_types:
      # TODO: test fails for float16 due to excessive precision requirements.
      if dtype in [np.float16, dtypes.bfloat16.as_numpy_dtype]:
        continue
      with self.session(), self.test_scope():
        variable_scope.get_variable_scope().set_use_resource(True)

        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype)

        var0 = resource_variable_ops.ResourceVariable(var0_np)
        var1 = resource_variable_ops.ResourceVariable(var1_np)
        grads0 = array_ops.placeholder(dtype)
        grads1 = array_ops.placeholder(dtype)
        opt = adam.AdamOptimizer()
        update1 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        update2 = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        self.evaluate(variables.global_variables_initializer())

        beta1_power, beta2_power = opt._get_beta_accumulators()

        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        # Run 3 steps of intertwined Adam1 and Adam2.
        for t in range(1, 4):
          self.assertAllCloseAccordingToType(0.9**t, self.evaluate(beta1_power))
          self.assertAllCloseAccordingToType(0.999**t,
                                             self.evaluate(beta2_power))
          if t % 2 == 0:
            update1.run(feed_dict={grads0: grads0_np, grads1: grads1_np})
          else:
            update2.run(feed_dict={grads0: grads0_np, grads1: grads1_np})

          var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))


if __name__ == "__main__":
  test.main()
