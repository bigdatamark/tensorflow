# [MODULE DOC]
# Test for the Adadelta optimizer on XLA devices: compares the optimizer slot variables and
# parameter updates against a NumPy reference implementation of the Adadelta algorithm.
# Has no import-time side effects; run via the TensorFlow test main entry point.
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
"""Tests for Adadelta Optimizer."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adadelta


class AdadeltaOptimizerTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # Adadelta optimizer test case for the XLA test harness (inherits the XLA test case with
  # device scoping and the supported dtype sets).

  def testBasic(self):
    # [FUNCTION DOC]
    # Scenario: for every float and complex dtype supported on the device, and for several
    # gradient and learning rate values (fewer on CPU to avoid timeouts), two resource variables
    # (1,2 and 3,4) are created and a constant gradient is applied with an Adadelta optimizer
    # (rho 0.95, epsilon 1e-8). Checks that the optimizer has four variables named after the
    # parameters, that the slot names are accumulator and accumulator update, that slots have the
    # parameter shapes and are not trainable, and that initial values are unchanged. Then runs
    # four update steps, each time computing the expected accumulators, update and total
    # parameter change in NumPy, and checks slots and parameters against them (relative
    # tolerance 1e-5, dtype aware).
    # KNOWN ISSUE: the step loop is outside the inner learning rate loop, so the checks use only
    # the last learning rate and variables of each gradient iteration.
    num_updates = 4  # number of ADADELTA steps to perform
    if "CPU" in self.device:
      # To avoid timeout on CPU.
      all_grad = [0.2, 0.01]
      all_lr = [1.0, 0.1]
    else:
      all_grad = [0.2, 0.1, 0.01]
      all_lr = [1.0, 0.5, 0.1]

    for dtype in self.float_types | self.complex_types:
      with self.session(), self.test_scope():
        for grad in all_grad:
          for lr in all_lr:
            var0_init = [1.0, 2.0]
            var1_init = [3.0, 4.0]
            var0 = resource_variable_ops.ResourceVariable(
                var0_init, dtype=dtype)
            var1 = resource_variable_ops.ResourceVariable(
                var1_init, dtype=dtype)

            grads = constant_op.constant([grad, grad], dtype=dtype)

            accum = 0.0
            accum_update = 0.0

            # ADADELTA gradient optimizer
            rho = 0.95
            epsilon = 1e-8
            adadelta_opt = adadelta.AdadeltaOptimizer(
                learning_rate=lr, rho=rho, epsilon=epsilon)
            adadelta_update = adadelta_opt.apply_gradients(
                zip([grads, grads], [var0, var1]))
            self.evaluate(variables.global_variables_initializer())
            opt_vars = adadelta_opt.variables()
            self.assertStartsWith(opt_vars[0].name, var0._shared_name)
            self.assertStartsWith(opt_vars[1].name, var0._shared_name)
            self.assertStartsWith(opt_vars[2].name, var1._shared_name)
            self.assertStartsWith(opt_vars[3].name, var1._shared_name)
            self.assertEqual(4, len(opt_vars))
            # Assign slots
            slot = [None] * 2
            slot_update = [None] * 2
            self.assertEqual(["accum", "accum_update"],
                             adadelta_opt.get_slot_names())
            slot[0] = adadelta_opt.get_slot(var0, "accum")
            self.assertEqual(slot[0].get_shape(), var0.get_shape())
            self.assertNotIn(slot[0], variables.trainable_variables())

            slot_update[0] = adadelta_opt.get_slot(var0, "accum_update")
            self.assertEqual(slot_update[0].get_shape(), var0.get_shape())
            self.assertNotIn(slot_update[0], variables.trainable_variables())

            slot[1] = adadelta_opt.get_slot(var1, "accum")
            self.assertEqual(slot[1].get_shape(), var1.get_shape())
            self.assertNotIn(slot[1], variables.trainable_variables())

            slot_update[1] = adadelta_opt.get_slot(var1, "accum_update")
            self.assertEqual(slot_update[1].get_shape(), var1.get_shape())
            self.assertNotIn(slot_update[1], variables.trainable_variables())

            # Fetch params to validate initial values
            self.assertAllClose(var0_init, self.evaluate(var0))
            self.assertAllClose(var1_init, self.evaluate(var1))

          update = [None] * num_updates
          tot_update = 0
          for step in range(num_updates):
            # Run adadelta update for comparison
            self.evaluate(adadelta_update)

            # Perform initial update without previous accum values
            accum = accum * rho + (grad**2) * (1 - rho)
            update[step] = (
                np.sqrt(accum_update + epsilon) *
                (1. / np.sqrt(accum + epsilon)) * grad)
            accum_update = (
                accum_update * rho + (update[step]**2) * (1.0 - rho))
            tot_update += update[step] * lr

            # Check that the accumulators have been updated
            for slot_idx in range(2):
              self.assertAllCloseAccordingToType(
                  np.array([accum, accum], dtype=dtype),
                  self.evaluate(slot[slot_idx]),
                  rtol=1e-5)

              self.assertAllCloseAccordingToType(
                  np.array([accum_update, accum_update], dtype=dtype),
                  self.evaluate(slot_update[slot_idx]),
                  rtol=1e-5)

            # Check that the parameters have been updated
            self.assertAllCloseAccordingToType(
                np.array(
                    [var0_init[0] - tot_update, var0_init[1] - tot_update],
                    dtype=dtype),
                self.evaluate(var0),
                rtol=1e-5)

            self.assertAllCloseAccordingToType(
                np.array(
                    [var1_init[0] - tot_update, var1_init[1] - tot_update],
                    dtype=dtype),
                self.evaluate(var1),
                rtol=1e-5)


if __name__ == "__main__":
  test.main()
