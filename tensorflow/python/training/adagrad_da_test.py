# [MODULE DOC]
# Functional tests for the AdagradDAOptimizer: numeric checks of one update step for
# no regularization (zero and non-zero initial variables, reference and resource variables),
# L1 only, and L1 plus L2, plus a sparse-gradient minimize test. Expected numbers come from the
# closed-form dual averaging update. Run via the TensorFlow test main.
#
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
"""Functional tests for AdagradDA operations."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adagrad_da


class AdagradDAOptimizerTest(test.TestCase):
  # [CLASS DOC]
  # Test case for AdagradDAOptimizer. Every test builds its own graph, a global step variable
  # and small vectors with fixed gradients, applies one optimizer step and compares with
  # precomputed expected values.

  def doTestAdagradDAwithoutRegularizationBasic1(self, use_resource=False):
    # [FUNCTION DOC]
    # HELPER: shared body for the no-regularization test. Input: use_resource, use resource
    # variables instead of reference ones (default false). For float64 and float32: two
    # zero-initialized variables, learning rate 3, initial squared accumulator 0.1, zero L1/L2.
    # Checks initial zeros, runs one step, then expects the known negative values from the
    # dual averaging formula (about -0.9045, -1.6036 and -0.0948, -0.1894).
    for dtype in [dtypes.float64, dtypes.float32]:
      with ops.Graph().as_default(), self.cached_session():
        global_step = variables.Variable(0, dtype=dtypes.int64)
        if use_resource:
          var0 = resource_variable_ops.ResourceVariable([0.0, 0.0], dtype=dtype)
          var1 = resource_variable_ops.ResourceVariable([0.0, 0.0], dtype=dtype)
        else:
          var0 = variables.Variable([0.0, 0.0], dtype=dtype)
          var1 = variables.Variable([0.0, 0.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)
        opt = adagrad_da.AdagradDAOptimizer(
            3.0,
            global_step,
            initial_gradient_squared_accumulator_value=0.1,
            l1_regularization_strength=0.0,
            l2_regularization_strength=0.0)
        update = opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]), global_step=global_step)
        self.evaluate(variables.global_variables_initializer())

        v0_val, v1_val = self.evaluate([var0, var1])
        self.assertAllClose([0.0, 0.0], v0_val)
        self.assertAllClose([0.0, 0.0], v1_val)

        # Run a step of AdagradDA
        update.run()

        v0_val, v1_val = self.evaluate([var0, var1])
        # Let g be the gradient accumulator, gg be the gradient squared
        # accumulator, T be the global step, lr be the learning rate,
        # and k the initial gradient squared accumulator value.
        # w = \dfrac{sign(-g)*lr*|g - l1*T|_{+}}{l2*T*lr + \sqrt{k+gg})}
        # For -0.1*3.0*(0.1 - 0)/(0 + sqrt(0.1 + 0.1*0.1)) = -0.904534
        # similarly for others.
        self.assertAllCloseAccordingToType(
            np.array([-0.904534, -1.603567]), v0_val)
        self.assertAllCloseAccordingToType(
            np.array([-0.094821, -0.189358]), v1_val)

  def testAdagradDAWithoutRegularizationBasic1(self):
    # [FUNCTION DOC]
    # Scenario: run the shared no-regularization check with reference variables. Expects the
    # helper's assertions to pass.
    self.doTestAdagradDAwithoutRegularizationBasic1()

  def testResourceAdagradDAWithoutRegularizationBasic1(self):
    # [FUNCTION DOC]
    # Scenario: same as the basic test but using resource variables. Expects the same values.
    self.doTestAdagradDAwithoutRegularizationBasic1(use_resource=True)

  @test_util.run_v1_only("loss needs to be callable in v2")
  def testMinimizeSparseResourceVariable(self):
    # [FUNCTION DOC]
    # Scenario (v1 only, since the loss must be callable in v2): a one-row resource variable is
    # read through an embedding lookup, multiplied by a constant and squared as the loss, then
    # minimized with AdagradDA (learning rate one) for one step with a resource global step.
    # Expects the initial value before the step and approximately minus one in each entry after
    # (relative tolerance one percent). Repeated for float32 and float64.
    for dtype in [dtypes.float32, dtypes.float64]:
      with self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([[1.0, 2.0]], dtype=dtype)
        global_step = resource_variable_ops.ResourceVariable(
            0, dtype=dtypes.int64)
        x = constant_op.constant([[4.0], [5.0]], dtype=dtype)
        pred = math_ops.matmul(embedding_ops.embedding_lookup([var0], [0]), x)
        loss = pred * pred
        sgd_op = adagrad_da.AdagradDAOptimizer(
            1.0, global_step).minimize(loss)
        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([[1.0, 2.0]], self.evaluate(var0))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType([[-1, -1]],
                                           self.evaluate(var0),
                                           rtol=0.01)

  def testAdagradDAwithoutRegularizationBasic2(self):
    # [FUNCTION DOC]
    # Scenario: variables start at non-zero values (one, two and four, three), no
    # regularization, one step. Expects the same final values as for zero-initialized variables,
    # since the dual averaging result does not depend on the starting value.
    for dtype in [dtypes.float64, dtypes.float32]:
      with ops.Graph().as_default(), self.cached_session():
        global_step = variables.Variable(0, dtype=dtypes.int64)
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([4.0, 3.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)

        opt = adagrad_da.AdagradDAOptimizer(
            3.0,
            global_step,
            initial_gradient_squared_accumulator_value=0.1,
            l1_regularization_strength=0.0,
            l2_regularization_strength=0.0)
        update = opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]), global_step=global_step)
        self.evaluate(variables.global_variables_initializer())

        v0_val, v1_val = self.evaluate([var0, var1])
        self.assertAllCloseAccordingToType([1.0, 2.0], v0_val)
        self.assertAllCloseAccordingToType([4.0, 3.0], v1_val)

        # Run a step of AdagradDA
        update.run()

        v0_val, v1_val = self.evaluate([var0, var1])
        self.assertAllCloseAccordingToType(
            np.array([-0.904534, -1.603567]), v0_val)
        self.assertAllCloseAccordingToType(
            np.array([-0.094821, -0.189358]), v1_val)

  def testAdagradDAWithL1(self):
    # [FUNCTION DOC]
    # Scenario: L1 strength 0.001, no L2, one step on non-zero starting variables. Expects the
    # slightly shrunken results (about -0.8955, -1.5956 and -0.0853, -0.1799) compared with the
    # unregularized case.
    for dtype in [dtypes.float64, dtypes.float32]:
      with ops.Graph().as_default(), self.cached_session():
        global_step = variables.Variable(0, dtype=dtypes.int64)
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([4.0, 3.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)

        opt = adagrad_da.AdagradDAOptimizer(
            3.0,
            global_step,
            initial_gradient_squared_accumulator_value=0.1,
            l1_regularization_strength=0.001,
            l2_regularization_strength=0.0)
        update = opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]), global_step=global_step)
        self.evaluate(variables.global_variables_initializer())

        v0_val, v1_val = self.evaluate([var0, var1])
        self.assertAllCloseAccordingToType([1.0, 2.0], v0_val)
        self.assertAllCloseAccordingToType([4.0, 3.0], v1_val)

        # Run a step of AdagradDA
        update.run()

        v0_val, v1_val = self.evaluate([var0, var1])
        self.assertAllCloseAccordingToType(
            np.array([-0.895489, -1.59555]), v0_val)
        self.assertAllCloseAccordingToType(
            np.array([-0.085339, -0.17989]), v1_val)

  def testAdagradDAWithL1_L2(self):
    # [FUNCTION DOC]
    # Scenario: L1 strength 0.001 and L2 strength 2.0, one step on non-zero starting variables.
    # Expects strongly shrunken results (about -0.0469, -0.0937 and -0.0043, -0.0090).
    for dtype in [dtypes.float64, dtypes.float32]:
      with ops.Graph().as_default(), self.cached_session():
        global_step = variables.Variable(0, dtype=dtypes.int64)
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([4.0, 3.0], dtype=dtype)
        grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
        grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)

        opt = adagrad_da.AdagradDAOptimizer(
            3.0,
            global_step,
            initial_gradient_squared_accumulator_value=0.1,
            l1_regularization_strength=0.001,
            l2_regularization_strength=2.0)
        update = opt.apply_gradients(
            zip([grads0, grads1], [var0, var1]), global_step=global_step)
        self.evaluate(variables.global_variables_initializer())

        v0_val, v1_val = self.evaluate([var0, var1])
        self.assertAllCloseAccordingToType([1.0, 2.0], v0_val)
        self.assertAllCloseAccordingToType([4.0, 3.0], v1_val)

        # Run a step of AdagradDA
        update.run()

        v0_val, v1_val = self.evaluate([var0, var1])
        self.assertAllCloseAccordingToType(
            np.array([-0.046907, -0.093659]), v0_val)
        self.assertAllCloseAccordingToType(
            np.array([-0.004275, -0.009023]), v1_val)


if __name__ == "__main__":
  test.main()
