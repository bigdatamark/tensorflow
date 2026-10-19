# [MODULE DOC]
# Test module for the AdagradDA (dual averaging Adagrad) training optimizer running on an XLA
# test device. It defines one test class that applies a single optimizer step to small float
# variables and compares the results with hand-computed expected values. It exposes no public
# API; running the file as a script starts the standard test runner. No import-time side effects.
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
"""Tests for AdagradDA optimizer."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adagrad_da


class AdagradDAOptimizerTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case covering the AdagradDA optimizer. Each test builds resource variables and
  # constant gradients on the XLA test device for every supported float type, applies one
  # optimizer step with a given pair of L1/L2 regularization strengths and checks the variables.
  # It holds no instance state of its own; it relies on the float type set from the base class.

  def testAdagradDAWithoutRegularizationBasic1(self):
    # [FUNCTION DOC]
    # Scenario: variables start at zero, learning rate 3.0, initial squared-gradient accumulator
    # 0.1, no L1 or L2 regularization. For each float type, one optimizer step is run with a
    # global step variable. Asserts that variables are zero before the step and equal the
    # precomputed negative values (about -0.9045, -1.6036 and -0.0948, -0.1894) afterwards.
    for dtype in self.float_types:
      with self.session(), self.test_scope():
        global_step = resource_variable_ops.ResourceVariable(
            0, dtype=dtypes.int64)
        var0 = resource_variable_ops.ResourceVariable([0.0, 0.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([0.0, 0.0], dtype=dtype)
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

        self.assertAllClose([0.0, 0.0], self.evaluate(var0))
        self.assertAllClose([0.0, 0.0], self.evaluate(var1))

        # Run a step of AdagradDA
        update.run()

        # Let g be the gradient accumulator, gg be the gradient squared
        # accumulator, T be the global step, lr be the learning rate,
        # and k the initial gradient squared accumulator value.
        # w = \dfrac{sign(-g)*lr*|g - l1*T|_{+}}{l2*T*lr + \sqrt{k+gg})}
        # For -0.1*3.0*(0.1 - 0)/(0 + sqrt(0.1 + 0.1*0.1)) = -0.904534
        # similarly for others.
        self.assertAllCloseAccordingToType(
            np.array([-0.904534, -1.603567]), self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            np.array([-0.094821, -0.189358]), self.evaluate(var1))

  def testAdagradDAwithoutRegularizationBasic2(self):
    # [FUNCTION DOC]
    # Same as the first basic test but variables start at non-zero values (1,2 and 4,3). Checks
    # initial values, runs one step without regularization and asserts the results are the same
    # as in the zero-start case, because the dual averaging result does not depend on the
    # initial variable values.
    for dtype in self.float_types:
      with self.session(), self.test_scope():
        global_step = resource_variable_ops.ResourceVariable(
            0, dtype=dtypes.int64)
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([4.0, 3.0], dtype=dtype)
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

        self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
        self.assertAllCloseAccordingToType([4.0, 3.0], self.evaluate(var1))

        # Run a step of AdagradDA
        update.run()

        self.assertAllCloseAccordingToType(
            np.array([-0.904534, -1.603567]), self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            np.array([-0.094821, -0.189358]), self.evaluate(var1))

  def testAdagradDAWithL1(self):
    # [FUNCTION DOC]
    # Runs one AdagradDA step with L1 regularization strength 0.001 and no L2, for each float
    # type. Asserts the variables move slightly less than in the unregularized case (about
    # -0.8955, -1.5956 and -0.0853, -0.1799) because the L1 term shrinks the accumulated gradient.
    for dtype in self.float_types:
      with self.session(), self.test_scope():
        global_step = resource_variable_ops.ResourceVariable(
            0, dtype=dtypes.int64)
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([4.0, 3.0], dtype=dtype)
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

        self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
        self.assertAllCloseAccordingToType([4.0, 3.0], self.evaluate(var1))

        # Run a step of AdagradDA
        update.run()

        self.assertAllCloseAccordingToType(
            np.array([-0.895489, -1.59555]), self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            np.array([-0.085339, -0.17989]), self.evaluate(var1))

  def testAdagradDAWithL1_L2(self):
    # [FUNCTION DOC]
    # Runs one AdagradDA step with L1 strength 0.001 and L2 strength 2.0, for each float type.
    # Asserts the strong L2 term damps the update heavily (about -0.0469, -0.0937 and
    # -0.0043, -0.0090) compared with the unregularized and L1-only cases.
    for dtype in self.float_types:
      with self.session(), self.test_scope():
        global_step = resource_variable_ops.ResourceVariable(
            0, dtype=dtypes.int64)
        var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype)
        var1 = resource_variable_ops.ResourceVariable([4.0, 3.0], dtype=dtype)
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

        self.assertAllCloseAccordingToType([1.0, 2.0], self.evaluate(var0))
        self.assertAllCloseAccordingToType([4.0, 3.0], self.evaluate(var1))

        # Run a step of AdagradDA
        update.run()

        self.assertAllCloseAccordingToType(
            np.array([-0.046907, -0.093659]), self.evaluate(var0))
        self.assertAllCloseAccordingToType(
            np.array([-0.004275, -0.009023]), self.evaluate(var1))


if __name__ == "__main__":
  test.main()
