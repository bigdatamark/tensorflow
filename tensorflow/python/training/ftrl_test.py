# [MODULE DOC]
# Functional tests for the FTRL-proximal optimizer of the training package. Each test builds
# small variables (dense, resource or sparse) in a fresh graph, applies the optimizer with
# various regularization settings (none, L1, L2, beta, L2 shrinkage) for several steps, and
# compares against precomputed values or against Adagrad / gradient descent, to which FTRL is
# equivalent in special configurations. No import-time registrations.
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
"""Functional tests for Ftrl operations."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adagrad
from tensorflow.python.training import ftrl
from tensorflow.python.training import gradient_descent


class FtrlOptimizerTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the FTRL optimizer: numerical results with different regularization
  # settings, dense and sparse gradients, reference and resource variables, half and 32-bit
  # float types, and equivalence to Adagrad and gradient descent. Has a shared helper that
  # applies a given optimizer for a few steps to zero-initialized variables.

  def doTestFtrlwithoutRegularization(self, use_resource=False):
    # [FUNCTION DOC]
    # HELPER: shared body for the plain and resource variants of the no-regularization test.
    # Input use_resource (default false) selects resource or ordinary variables. For half and
    # 32-bit float, two zero-initialized pairs of variables get constant gradients and FTRL with
    # learning rate 3.0 and accumulator 0.1 and no L1 or L2. Asserts the initial zeros, runs 3
    # steps and asserts the values match precomputed numbers (looser tolerance for half).
    # The v1 optimizers do not support eager execution
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32]:
        with self.cached_session():
          if use_resource:
            var0 = resource_variable_ops.ResourceVariable([0.0, 0.0],
                                                          dtype=dtype)
            var1 = resource_variable_ops.ResourceVariable([0.0, 0.0],
                                                          dtype=dtype)
          else:
            var0 = variables.Variable([0.0, 0.0], dtype=dtype)
            var1 = variables.Variable([0.0, 0.0], dtype=dtype)
          grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
          grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)
          opt = ftrl.FtrlOptimizer(
              3.0,
              initial_accumulator_value=0.1,
              l1_regularization_strength=0.0,
              l2_regularization_strength=0.0)
          update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
          self.evaluate(variables.global_variables_initializer())

          v0_val, v1_val = self.evaluate([var0, var1])
          self.assertAllClose([0.0, 0.0], v0_val)
          self.assertAllClose([0.0, 0.0], v1_val)

          # Run 3 steps FTRL
          for _ in range(3):
            update.run()

          v0_val, v1_val = self.evaluate([var0, var1])
          self.assertAllCloseAccordingToType(
              np.array([-2.60260963, -4.29698515]), v0_val, half_rtol=1e-2)
          self.assertAllCloseAccordingToType(
              np.array([-0.28432083, -0.56694895]), v1_val)

  def testFtrlWithoutRegularization(self):
    # [FUNCTION DOC]
    # Runs the shared no-regularization scenario with ordinary (reference) variables and checks
    # the values after 3 steps against expected numbers.
    self.doTestFtrlwithoutRegularization(use_resource=False)

  def testResourceFtrlWithoutRegularization(self):
    # [FUNCTION DOC]
    # Runs the shared no-regularization scenario with resource variables and checks the values
    # after 3 steps against expected numbers.
    self.doTestFtrlwithoutRegularization(use_resource=True)

  def testFtrlwithoutRegularization2(self):
    # [FUNCTION DOC]
    # Scenario: variables start at non-zero values (1,2 and 4,3), constant gradients, FTRL with
    # learning rate 3.0, accumulator 0.1 and no L1 or L2, for half and 32-bit float. Asserts the
    # initial values, runs 3 steps, and asserts the results equal precomputed numbers.
    # The v1 optimizers do not support eager execution
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32]:
        with self.cached_session():
          var0 = variables.Variable([1.0, 2.0], dtype=dtype)
          var1 = variables.Variable([4.0, 3.0], dtype=dtype)
          grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
          grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)

          opt = ftrl.FtrlOptimizer(
              3.0,
              initial_accumulator_value=0.1,
              l1_regularization_strength=0.0,
              l2_regularization_strength=0.0)
          update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
          self.evaluate(variables.global_variables_initializer())

          v0_val, v1_val = self.evaluate([var0, var1])
          self.assertAllCloseAccordingToType([1.0, 2.0], v0_val)
          self.assertAllCloseAccordingToType([4.0, 3.0], v1_val)

          # Run 3 steps FTRL
          for _ in range(3):
            update.run()
          v0_val, v1_val = self.evaluate([var0, var1])
          self.assertAllCloseAccordingToType(
              np.array([-2.55607247, -3.98729396]), v0_val)
          self.assertAllCloseAccordingToType(
              np.array([-0.28232238, -0.56096673]), v1_val)

  def testMinimizeSparseResourceVariable(self):
    # [FUNCTION DOC]
    # Scenario: a resource variable row is looked up by embedding lookup, multiplied by a
    # constant and squared to form a loss; FTRL with learning rate 1.0 minimizes it for one
    # step, for half, 32-bit and 64-bit floats. Asserts the initial value (1, 2) and that the
    # updated value is about (0, 1) within 0.01.
    # The v1 optimizers do not support eager execution
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
        with self.cached_session():
          var0 = resource_variable_ops.ResourceVariable([[1.0, 2.0]],
                                                        dtype=dtype)
          x = constant_op.constant([[4.0], [5.0]], dtype=dtype)
          pred = math_ops.matmul(embedding_ops.embedding_lookup([var0], [0]), x)
          loss = pred * pred
          sgd_op = ftrl.FtrlOptimizer(1.0).minimize(loss)
          self.evaluate(variables.global_variables_initializer())
          # Fetch params to validate initial values
          self.assertAllCloseAccordingToType([[1.0, 2.0]], self.evaluate(var0))
          # Run 1 step of sgd
          sgd_op.run()
          # Validate updated params
          self.assertAllCloseAccordingToType([[0, 1]],
                                             self.evaluate(var0),
                                             atol=0.01)

  def testFtrlWithL1(self):
    # [FUNCTION DOC]
    # Scenario: FTRL with learning rate 3.0, accumulator 0.1, L1 strength 0.001 and no L2 on two
    # small variables with constant gradients. Asserts initial values, runs 10 steps and asserts
    # the values match precomputed numbers for half and 32-bit float.
    # The v1 optimizers do not support eager execution
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32]:
        with self.cached_session():
          var0 = variables.Variable([1.0, 2.0], dtype=dtype)
          var1 = variables.Variable([4.0, 3.0], dtype=dtype)
          grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
          grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)

          opt = ftrl.FtrlOptimizer(
              3.0,
              initial_accumulator_value=0.1,
              l1_regularization_strength=0.001,
              l2_regularization_strength=0.0)
          update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
          self.evaluate(variables.global_variables_initializer())

          v0_val, v1_val = self.evaluate([var0, var1])
          self.assertAllCloseAccordingToType([1.0, 2.0], v0_val)
          self.assertAllCloseAccordingToType([4.0, 3.0], v1_val)

          # Run 10 steps FTRL
          for _ in range(10):
            update.run()
          v0_val, v1_val = self.evaluate([var0, var1])
          self.assertAllCloseAccordingToType(
              np.array([-7.66718769, -10.91273689]), v0_val)
          self.assertAllCloseAccordingToType(
              np.array([-0.93460727, -1.86147261]), v1_val)

  def testFtrlWithBeta(self):
    # [FUNCTION DOC]
    # Scenario: FTRL with learning rate 3.0, accumulator 0.1 and beta 0.1 (no L1 or L2).
    # Asserts initial values, runs 10 steps and asserts precomputed results for half and 32-bit
    # float.
    # The v1 optimizers do not support eager execution
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32]:
        with self.cached_session():
          var0 = variables.Variable([1.0, 2.0], dtype=dtype)
          var1 = variables.Variable([4.0, 3.0], dtype=dtype)
          grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
          grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)

          opt = ftrl.FtrlOptimizer(3.0, initial_accumulator_value=0.1, beta=0.1)
          update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
          self.evaluate(variables.global_variables_initializer())

          v0_val, v1_val = self.evaluate([var0, var1])
          self.assertAllCloseAccordingToType([1.0, 2.0], v0_val)
          self.assertAllCloseAccordingToType([4.0, 3.0], v1_val)

          # Run 10 steps FTRL
          for _ in range(10):
            update.run()
          v0_val, v1_val = self.evaluate([var0, var1])
          self.assertAllCloseAccordingToType(
              np.array([-6.096838, -9.162214]), v0_val)
          self.assertAllCloseAccordingToType(
              np.array([-0.717741, -1.425132]), v1_val)

  def testFtrlWithL2_Beta(self):
    # [FUNCTION DOC]
    # Scenario: FTRL with learning rate 3.0, accumulator 0.1, no L1, L2 strength 0.1 and beta
    # 0.1. Asserts initial values, runs 10 steps and asserts precomputed results for half and
    # 32-bit float.
    # The v1 optimizers do not support eager execution
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32]:
        with self.cached_session():
          var0 = variables.Variable([1.0, 2.0], dtype=dtype)
          var1 = variables.Variable([4.0, 3.0], dtype=dtype)
          grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
          grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)

          opt = ftrl.FtrlOptimizer(
              3.0,
              initial_accumulator_value=0.1,
              l1_regularization_strength=0.0,
              l2_regularization_strength=0.1,
              beta=0.1)
          update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
          self.evaluate(variables.global_variables_initializer())

          v0_val, v1_val = self.evaluate([var0, var1])
          self.assertAllCloseAccordingToType([1.0, 2.0], v0_val)
          self.assertAllCloseAccordingToType([4.0, 3.0], v1_val)

          # Run 10 steps FTRL
          for _ in range(10):
            update.run()
          v0_val, v1_val = self.evaluate([var0, var1])
          self.assertAllCloseAccordingToType(
              np.array([-2.735487, -4.704625]), v0_val)
          self.assertAllCloseAccordingToType(
              np.array([-0.294335, -0.586556]), v1_val)

  def testFtrlWithL1_L2(self):
    # [FUNCTION DOC]
    # Scenario: FTRL with learning rate 3.0, accumulator 0.1, L1 strength 0.001 and L2 strength
    # 2.0. Asserts initial values, runs 10 steps and asserts that the strong L2 keeps the values
    # small, matching precomputed numbers, for half and 32-bit float.
    # The v1 optimizers do not support eager execution
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32]:
        with self.cached_session():
          var0 = variables.Variable([1.0, 2.0], dtype=dtype)
          var1 = variables.Variable([4.0, 3.0], dtype=dtype)
          grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
          grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)

          opt = ftrl.FtrlOptimizer(
              3.0,
              initial_accumulator_value=0.1,
              l1_regularization_strength=0.001,
              l2_regularization_strength=2.0)
          update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
          self.evaluate(variables.global_variables_initializer())

          v0_val, v1_val = self.evaluate([var0, var1])
          self.assertAllCloseAccordingToType([1.0, 2.0], v0_val)
          self.assertAllCloseAccordingToType([4.0, 3.0], v1_val)

          # Run 10 steps FTRL
          for _ in range(10):
            update.run()

          v0_val, v1_val = self.evaluate([var0, var1])
          self.assertAllCloseAccordingToType(
              np.array([-0.24059935, -0.46829352]), v0_val)
          self.assertAllCloseAccordingToType(
              np.array([-0.02406147, -0.04830509]), v1_val)

  def testFtrlWithL1_L2_L2Shrinkage(self):
    # [FUNCTION DOC]
    # Scenario: same as the L1 plus L2 test but also with L2 shrinkage strength 0.1, which
    # selects the shrinkage version of the op. Asserts initial values, runs 10 steps and asserts
    # the results equal precomputed numbers (smaller magnitudes than without shrinkage).
    """Test the new FTRL op with support for l2 shrinkage.

    The addition of this parameter which places a constant pressure on weights
    towards the origin causes the gradient descent trajectory to differ. The
    weights will tend to have smaller magnitudes with this parameter set.
    """
    # The v1 optimizers do not support eager execution
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32]:
        with self.cached_session():
          var0 = variables.Variable([1.0, 2.0], dtype=dtype)
          var1 = variables.Variable([4.0, 3.0], dtype=dtype)
          grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
          grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)

          opt = ftrl.FtrlOptimizer(
              3.0,
              initial_accumulator_value=0.1,
              l1_regularization_strength=0.001,
              l2_regularization_strength=2.0,
              l2_shrinkage_regularization_strength=0.1)
          update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
          self.evaluate(variables.global_variables_initializer())

          v0_val, v1_val = self.evaluate([var0, var1])
          self.assertAllCloseAccordingToType([1.0, 2.0], v0_val)
          self.assertAllCloseAccordingToType([4.0, 3.0], v1_val)

          # Run 10 steps FTRL
          for _ in range(10):
            update.run()

          v0_val, v1_val = self.evaluate([var0, var1])
          self.assertAllCloseAccordingToType(
              np.array([-0.22578995, -0.44345796]), v0_val)
          self.assertAllCloseAccordingToType(
              np.array([-0.14378493, -0.13229476]), v1_val)

  def testFtrlWithL1_L2_L2ShrinkageSparse(self):
    # [FUNCTION DOC]
    # Scenario: the shrinkage configuration applied with sparse (indexed-slices) gradients, each
    # touching a single different row of a two-row variable. Asserts initial values, runs 10
    # steps and asserts only the touched rows changed, matching precomputed numbers.
    """Tests the new FTRL op with support for l2 shrinkage on sparse grads."""
    # The v1 optimizers do not support eager execution
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32]:
        with self.cached_session():
          var0 = variables.Variable([[1.0], [2.0]], dtype=dtype)
          var1 = variables.Variable([[4.0], [3.0]], dtype=dtype)
          grads0 = indexed_slices.IndexedSlices(
              constant_op.constant([0.1], shape=[1, 1], dtype=dtype),
              constant_op.constant([0]), constant_op.constant([2, 1]))
          grads1 = indexed_slices.IndexedSlices(
              constant_op.constant([0.02], shape=[1, 1], dtype=dtype),
              constant_op.constant([1]), constant_op.constant([2, 1]))

          opt = ftrl.FtrlOptimizer(
              3.0,
              initial_accumulator_value=0.1,
              l1_regularization_strength=0.001,
              l2_regularization_strength=2.0,
              l2_shrinkage_regularization_strength=0.1)
          update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
          self.evaluate(variables.global_variables_initializer())

          v0_val, v1_val = self.evaluate([var0, var1])
          self.assertAllCloseAccordingToType([[1.0], [2.0]], v0_val)
          self.assertAllCloseAccordingToType([[4.0], [3.0]], v1_val)

          # Run 10 steps FTRL
          for _ in range(10):
            update.run()

          v0_val, v1_val = self.evaluate([var0, var1])
          self.assertAllCloseAccordingToType([[-0.22578995], [2.]], v0_val)
          self.assertAllCloseAccordingToType([[4.], [-0.13229476]], v1_val)

  def testFtrlWithL2ShrinkageDoesNotChangeLrSchedule(self):
    # [FUNCTION DOC]
    # Scenario: two optimizers with identical settings except that one has L2 shrinkage 0.1, each
    # applied for 10 steps to equal variables with equal gradients. Asserts that the shrinkage
    # variable ends smaller in magnitude elementwise, and that the accumulator slots of both
    # optimizers are equal, i.e. shrinkage does not change accumulator updates.
    """Verifies that l2 shrinkage in FTRL does not change lr schedule."""
    # The v1 optimizers do not support eager execution
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32]:
        with self.cached_session():
          var0 = variables.Variable([1.0, 2.0], dtype=dtype)
          var1 = variables.Variable([1.0, 2.0], dtype=dtype)
          grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
          grads1 = constant_op.constant([0.1, 0.2], dtype=dtype)

          opt0 = ftrl.FtrlOptimizer(
              3.0,
              initial_accumulator_value=0.1,
              l1_regularization_strength=0.001,
              l2_regularization_strength=2.0,
              l2_shrinkage_regularization_strength=0.1)
          opt1 = ftrl.FtrlOptimizer(
              3.0,
              initial_accumulator_value=0.1,
              l1_regularization_strength=0.001,
              l2_regularization_strength=2.0)
          update0 = opt0.apply_gradients([(grads0, var0)])
          update1 = opt1.apply_gradients([(grads1, var1)])
          self.evaluate(variables.global_variables_initializer())

          v0_val, v1_val = self.evaluate([var0, var1])
          self.assertAllCloseAccordingToType([1.0, 2.0], v0_val)
          self.assertAllCloseAccordingToType([1.0, 2.0], v1_val)

          # Run 10 steps FTRL
          for _ in range(10):
            update0.run()
            update1.run()

          v0_val, v1_val = self.evaluate([var0, var1])
          # var0 is experiencing L2 shrinkage so it should be smaller than var1
          # in magnitude.
          self.assertTrue((v0_val**2 < v1_val**2).all())
          accum0 = list(self.evaluate(opt0._slots)["accum"].values())[0]
          accum1 = list(self.evaluate(opt1._slots)["accum"].values())[0]
          # L2 shrinkage should not change how we update grad accumulator.
          self.assertAllCloseAccordingToType(accum0, accum1)

  def applyOptimizer(self, opt, dtype, steps=5, is_sparse=False):
    # [FUNCTION DOC]
    # HELPER: shared by the equivalence tests.
    # Inputs: opt (optimizer to apply), dtype (variable dtype), steps (default 5), is_sparse
    # (default false; use indexed-slices gradients on two-row variables instead of dense ones).
    # Creates two zero-initialized variables with fixed gradients, applies the optimizer,
    # initializes variables, asserts they start at zero, runs the update for the given number of
    # steps and returns the two resulting variable values. Needs a default session.
    # KNOWN ISSUE: a local session variable obtained from the default session is never used.
    if is_sparse:
      var0 = variables.Variable([[0.0], [0.0]], dtype=dtype)
      var1 = variables.Variable([[0.0], [0.0]], dtype=dtype)
      grads0 = indexed_slices.IndexedSlices(
          constant_op.constant([0.1], shape=[1, 1], dtype=dtype),
          constant_op.constant([0]), constant_op.constant([2, 1]))
      grads1 = indexed_slices.IndexedSlices(
          constant_op.constant([0.02], shape=[1, 1], dtype=dtype),
          constant_op.constant([1]), constant_op.constant([2, 1]))
    else:
      var0 = variables.Variable([0.0, 0.0], dtype=dtype)
      var1 = variables.Variable([0.0, 0.0], dtype=dtype)
      grads0 = constant_op.constant([0.1, 0.2], dtype=dtype)
      grads1 = constant_op.constant([0.01, 0.02], dtype=dtype)

    update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
    self.evaluate(variables.global_variables_initializer())

    sess = ops.get_default_session()
    v0_val, v1_val = self.evaluate([var0, var1])
    if is_sparse:
      self.assertAllCloseAccordingToType([[0.0], [0.0]], v0_val)
      self.assertAllCloseAccordingToType([[0.0], [0.0]], v1_val)
    else:
      self.assertAllCloseAccordingToType([0.0, 0.0], v0_val)
      self.assertAllCloseAccordingToType([0.0, 0.0], v1_val)

    # Run Ftrl for a few steps
    for _ in range(steps):
      update.run()

    v0_val, v1_val = self.evaluate([var0, var1])
    return v0_val, v1_val

  # When variables are initialized with Zero, FTRL-Proximal has two properties:
  # 1. Without L1&L2 but with fixed learning rate, FTRL-Proximal is identical
  # with GradientDescent.
  # 2. Without L1&L2 but with adaptive learning rate, FTRL-Proximal is identical
  # with Adagrad.
  # So, basing on these two properties, we test if our implementation of
  # FTRL-Proximal performs same updates as Adagrad or GradientDescent.
  def testEquivAdagradwithoutRegularization(self):
    # [FUNCTION DOC]
    # Scenario: with zero-initialized variables, no L1 or L2 and learning rate power -0.5, FTRL
    # should behave like Adagrad. Applies both optimizers (rate 3.0, accumulator 0.1) with dense
    # gradients and asserts the resulting values agree, with looser tolerance for half.
    # The v1 optimizers do not support eager execution
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32]:
        with self.cached_session():
          val0, val1 = self.applyOptimizer(
              ftrl.FtrlOptimizer(
                  3.0,
                  # Adagrad learning rate
                  learning_rate_power=-0.5,
                  initial_accumulator_value=0.1,
                  l1_regularization_strength=0.0,
                  l2_regularization_strength=0.0),
              dtype)

        with self.cached_session():
          val2, val3 = self.applyOptimizer(
              adagrad.AdagradOptimizer(3.0, initial_accumulator_value=0.1),
              dtype)

        self.assertAllCloseAccordingToType(val0, val2, half_rtol=2e-3)
        self.assertAllCloseAccordingToType(val1, val3, half_rtol=2e-3)

  def testEquivSparseAdagradwithoutRegularization(self):
    # [FUNCTION DOC]
    # Same as the dense Adagrad equivalence test but with sparse gradients; asserts FTRL and
    # Adagrad produce equal values.
    # The v1 optimizers do not support eager execution
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32]:
        with self.cached_session():
          val0, val1 = self.applyOptimizer(
              ftrl.FtrlOptimizer(
                  3.0,
                  # Adagrad learning rate
                  learning_rate_power=-0.5,
                  initial_accumulator_value=0.1,
                  l1_regularization_strength=0.0,
                  l2_regularization_strength=0.0),
              dtype,
              is_sparse=True)

        with self.cached_session():
          val2, val3 = self.applyOptimizer(
              adagrad.AdagradOptimizer(3.0, initial_accumulator_value=0.1),
              dtype,
              is_sparse=True)

        self.assertAllCloseAccordingToType(val0, val2)
        self.assertAllCloseAccordingToType(val1, val3)

  def testEquivSparseGradientDescentwithoutRegularization(self):
    # [FUNCTION DOC]
    # Scenario: with zero-initialized variables, no L1 or L2 and a learning rate power of zero
    # (fixed rate), FTRL should match plain gradient descent. Applies both with sparse gradients
    # and rate 3.0 and asserts equal values.
    # The v1 optimizers do not support eager execution
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32]:
        with self.cached_session():
          val0, val1 = self.applyOptimizer(
              ftrl.FtrlOptimizer(
                  3.0,
                  # Fixed learning rate
                  learning_rate_power=-0.0,
                  initial_accumulator_value=0.1,
                  l1_regularization_strength=0.0,
                  l2_regularization_strength=0.0),
              dtype,
              is_sparse=True)

        with self.cached_session():
          val2, val3 = self.applyOptimizer(
              gradient_descent.GradientDescentOptimizer(3.0),
              dtype,
              is_sparse=True)

        self.assertAllCloseAccordingToType(val0, val2)
        self.assertAllCloseAccordingToType(val1, val3)

  def testEquivGradientDescentwithoutRegularization(self):
    # [FUNCTION DOC]
    # Same as the sparse gradient descent equivalence test but with dense gradients; asserts FTRL
    # with a fixed rate and gradient descent produce equal values.
    # The v1 optimizers do not support eager execution
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32]:
        with self.cached_session():
          val0, val1 = self.applyOptimizer(
              ftrl.FtrlOptimizer(
                  3.0,
                  # Fixed learning rate
                  learning_rate_power=-0.0,
                  initial_accumulator_value=0.1,
                  l1_regularization_strength=0.0,
                  l2_regularization_strength=0.0),
              dtype)

        with self.cached_session():
          val2, val3 = self.applyOptimizer(
              gradient_descent.GradientDescentOptimizer(3.0), dtype)

        self.assertAllCloseAccordingToType(val0, val2)
        self.assertAllCloseAccordingToType(val1, val3)


if __name__ == "__main__":
  test.main()
