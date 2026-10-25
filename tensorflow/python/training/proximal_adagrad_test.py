# [MODULE DOC]
# Functional tests for the v1 Proximal Adagrad optimizer. Covers dense updates with and without
# L1/L2 regularization against hard-coded expected numbers, a sparse resource-variable
# minimization, and equivalence with the plain Adagrad optimizer (dense and sparse) when both
# regularization strengths are zero. All tests build graph-mode sessions because the optimizer is
# only supported in the v1 graph API. Runs through the standard test main entry point.
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
"""Functional tests for Proximal Adagrad operations."""

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
from tensorflow.python.training import proximal_adagrad


class ProximalAdagradOptimizerTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the Proximal Adagrad optimizer: numeric results of dense updates under
  # different regularization settings, sparse resource-variable training, and agreement with the
  # plain Adagrad optimizer. Shared fixtures: the helper that applies an optimizer to two small
  # variables and returns their values, and the per-test default graph and cached session.

  def doTestProximalAdagradwithoutRegularization(self, use_resource=False):
    # [FUNCTION DOC]
    # HELPER: shared body for the plain and resource "without regularization" tests.
    # KNOWN ISSUE: the use_resource argument is ignored; both callers create ordinary variables, so
    # the resource variant does not actually exercise resource variables.
    # Scenario: two zero-initialized variables with fixed gradients, optimizer with learning rate
    # 3.0, initial accumulator 0.1 and zero L1/L2. Applies gradients, initializes, checks the
    # initial zeros, runs three update steps, then asserts the exact expected values for both
    # variables and that the optimizer owns exactly two slot variables whose names start with the
    # shared names of the two variables.
    # ProximalAdagradOptimizer is supported only in V1.
    with ops.Graph().as_default(), self.cached_session():
      var0 = variables.Variable([0.0, 0.0])
      var1 = variables.Variable([0.0, 0.0])
      grads0 = constant_op.constant([0.1, 0.2])
      grads1 = constant_op.constant([0.01, 0.02])
      opt = proximal_adagrad.ProximalAdagradOptimizer(
          3.0,
          initial_accumulator_value=0.1,
          l1_regularization_strength=0.0,
          l2_regularization_strength=0.0)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      self.evaluate(variables.global_variables_initializer())

      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllClose([0.0, 0.0], v0_val)
      self.assertAllClose([0.0, 0.0], v1_val)

      # Run 3 steps Proximal Adagrad.
      for _ in range(3):
        update.run()

      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllClose(np.array([-2.60260963, -4.29698515]), v0_val)
      self.assertAllClose(np.array([-0.28432083, -0.56694895]), v1_val)
      opt_vars = opt.variables()
      self.assertStartsWith(opt_vars[0].name, var0._shared_name)
      self.assertStartsWith(opt_vars[1].name, var1._shared_name)
      self.assertEqual(2, len(opt_vars))

  def testProximalAdagradwithoutRegularization(self):
    # [FUNCTION DOC]
    # Test: runs the shared without-regularization scenario with the resource flag off. Expects the
    # documented values after three steps and two optimizer slot variables.
    self.doTestProximalAdagradwithoutRegularization(use_resource=False)

  def testResourceProximalAdagradwithoutRegularization(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite the name, the shared scenario ignores the resource flag, so this
    # duplicates the non-resource test with ordinary variables.
    # Test: runs the shared without-regularization scenario with the resource flag on and expects
    # the same numeric results as the non-resource test.
    self.doTestProximalAdagradwithoutRegularization(use_resource=True)

  def testProximalAdagradwithoutRegularization2(self):
    # [FUNCTION DOC]
    # Test: variables start at nonzero values (1,2) and (4,3) with fixed small gradients; optimizer
    # uses learning rate 3.0, initial accumulator 0.1, no regularization. Checks the initial
    # values, runs three steps, and asserts the exact expected updated values of both variables.
    # ProximalAdagradOptimizer is supported only in V1.
    with ops.Graph().as_default(), self.cached_session():
      var0 = variables.Variable([1.0, 2.0])
      var1 = variables.Variable([4.0, 3.0])
      grads0 = constant_op.constant([0.1, 0.2])
      grads1 = constant_op.constant([0.01, 0.02])

      opt = proximal_adagrad.ProximalAdagradOptimizer(
          3.0,
          initial_accumulator_value=0.1,
          l1_regularization_strength=0.0,
          l2_regularization_strength=0.0)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      self.evaluate(variables.global_variables_initializer())

      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllClose([1.0, 2.0], v0_val)
      self.assertAllClose([4.0, 3.0], v1_val)

      # Run 3 steps Proximal Adagrad.
      for _ in range(3):
        update.run()
      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllClose(np.array([-1.60261, -2.296985]), v0_val)
      self.assertAllClose(np.array([3.715679, 2.433051]), v1_val)

  def testMinimizeSparseResourceVariable(self):
    # [FUNCTION DOC]
    # Test: for single and double precision floats, builds a resource variable holding one row,
    # reads it through an embedding lookup, multiplies by a constant column, squares the result as
    # the loss and minimizes it with learning rate 1.0. Asserts the initial value, runs one step,
    # and asserts the variable moved to roughly (0, 1) within a small absolute tolerance, which
    # exercises the sparse resource update path.
    for dtype in [dtypes.float32, dtypes.float64]:
      # ProximalAdagradOptimizer is supported only in V1.
      with ops.Graph().as_default(), self.cached_session():
        var0 = resource_variable_ops.ResourceVariable([[1.0, 2.0]], dtype=dtype)
        x = constant_op.constant([[4.0], [5.0]], dtype=dtype)
        pred = math_ops.matmul(embedding_ops.embedding_lookup([var0], [0]), x)
        loss = pred * pred
        sgd_op = proximal_adagrad.ProximalAdagradOptimizer(1.0).minimize(loss)
        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllCloseAccordingToType([[1.0, 2.0]], self.evaluate(var0))
        # Run 1 step of sgd
        sgd_op.run()
        # Validate updated params
        self.assertAllCloseAccordingToType([[0, 1]],
                                           self.evaluate(var0),
                                           atol=0.01)

  def testProximalAdagradWithL1(self):
    # [FUNCTION DOC]
    # Test: variables (1,2) and (4,3), fixed gradients, learning rate 3.0, initial accumulator 0.1,
    # L1 strength 0.001 and L2 zero. Verifies initial values, runs ten steps, and asserts the exact
    # expected values showing L1-regularized Adagrad results.
    # ProximalAdagradOptimizer is supported only in V1.
    with ops.Graph().as_default(), self.cached_session():
      var0 = variables.Variable([1.0, 2.0])
      var1 = variables.Variable([4.0, 3.0])
      grads0 = constant_op.constant([0.1, 0.2])
      grads1 = constant_op.constant([0.01, 0.02])

      opt = proximal_adagrad.ProximalAdagradOptimizer(
          3.0,
          initial_accumulator_value=0.1,
          l1_regularization_strength=0.001,
          l2_regularization_strength=0.0)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      self.evaluate(variables.global_variables_initializer())

      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllClose([1.0, 2.0], v0_val)
      self.assertAllClose([4.0, 3.0], v1_val)

      # Run 10 steps Proximal Adagrad
      for _ in range(10):
        update.run()
      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllClose(np.array([-6.663634, -9.190331]), v0_val)
      self.assertAllClose(np.array([2.959304, 1.029232]), v1_val)

  def testProximalAdagradWithL1_L2(self):
    # [FUNCTION DOC]
    # Test: same setup as the L1 test but with L1 strength 0.001 and L2 strength 2.0. Verifies the
    # initial values, runs ten steps, and asserts the variables shrink to values very close to zero
    # (about -0.05/-0.1 and -0.0045/-0.0095), showing the strong L2 shrinkage.
    # ProximalAdagradOptimizer is supported only in V1.
    with ops.Graph().as_default(), self.cached_session():
      var0 = variables.Variable([1.0, 2.0])
      var1 = variables.Variable([4.0, 3.0])
      grads0 = constant_op.constant([0.1, 0.2])
      grads1 = constant_op.constant([0.01, 0.02])

      opt = proximal_adagrad.ProximalAdagradOptimizer(
          3.0,
          initial_accumulator_value=0.1,
          l1_regularization_strength=0.001,
          l2_regularization_strength=2.0)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
      self.evaluate(variables.global_variables_initializer())

      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllClose([1.0, 2.0], v0_val)
      self.assertAllClose([4.0, 3.0], v1_val)

      # Run 10 steps Proximal Adagrad.
      for _ in range(10):
        update.run()

      v0_val, v1_val = self.evaluate([var0, var1])
      self.assertAllClose(np.array([-0.0495, -0.0995]), v0_val)
      self.assertAllClose(np.array([-0.0045, -0.0095]), v1_val)

  def applyOptimizer(self, opt, steps=5, is_sparse=False):
    # [FUNCTION DOC]
    # HELPER: runs a given optimizer on two small variables for the equivalence tests.
    # KNOWN ISSUE: a local variable holding the default session is assigned but never used.
    # Inputs: opt (optimizer instance), steps (number of update runs, default 5), is_sparse (when
    # true, variables are two-row columns and gradients are indexed slices touching one row each;
    # otherwise plain vectors with dense gradients). Applies gradients, initializes variables,
    # asserts the initial values, runs the update the given number of times in the default session
    # and returns the pair of final values of the two variables. Must be called inside a graph and
    # session.
    if is_sparse:
      var0 = variables.Variable([[1.0], [2.0]])
      var1 = variables.Variable([[3.0], [4.0]])
      grads0 = indexed_slices.IndexedSlices(
          constant_op.constant(
              [0.1], shape=[1, 1]),
          constant_op.constant([0]),
          constant_op.constant([2, 1]))
      grads1 = indexed_slices.IndexedSlices(
          constant_op.constant(
              [0.02], shape=[1, 1]),
          constant_op.constant([1]),
          constant_op.constant([2, 1]))
    else:
      var0 = variables.Variable([1.0, 2.0])
      var1 = variables.Variable([3.0, 4.0])
      grads0 = constant_op.constant([0.1, 0.2])
      grads1 = constant_op.constant([0.01, 0.02])

    update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
    self.evaluate(variables.global_variables_initializer())

    sess = ops.get_default_session()
    v0_val, v1_val = self.evaluate([var0, var1])
    if is_sparse:
      self.assertAllClose([[1.0], [2.0]], v0_val)
      self.assertAllClose([[3.0], [4.0]], v1_val)
    else:
      self.assertAllClose([1.0, 2.0], v0_val)
      self.assertAllClose([3.0, 4.0], v1_val)

    # Run ProximalAdagrad for a few steps
    for _ in range(steps):
      update.run()

    v0_val, v1_val = self.evaluate([var0, var1])
    return v0_val, v1_val

  def testEquivAdagradwithoutRegularization(self):
    # [FUNCTION DOC]
    # Test: applies Proximal Adagrad (learning rate 3.0, accumulator 0.1, zero L1/L2) and the plain
    # Adagrad optimizer with the same hyperparameters, each in a fresh graph and session, using
    # dense gradients. Asserts both variables end with the same values in both runs.
    # ProximalAdagradOptimizer is supported only in V1.
    with ops.Graph().as_default(), self.cached_session():
      val0, val1 = self.applyOptimizer(
          proximal_adagrad.ProximalAdagradOptimizer(
              3.0,
              initial_accumulator_value=0.1,
              l1_regularization_strength=0.0,
              l2_regularization_strength=0.0))

    with ops.Graph().as_default(), self.cached_session():
      val2, val3 = self.applyOptimizer(
          adagrad.AdagradOptimizer(
              3.0, initial_accumulator_value=0.1))

    self.assertAllClose(val0, val2)
    self.assertAllClose(val1, val3)

  def testEquivSparseAdagradwithoutRegularization(self):
    # [FUNCTION DOC]
    # Test: same as the dense equivalence test but with sparse (indexed slices) gradients. Asserts
    # Proximal Adagrad without regularization matches plain Adagrad on both variables.
    # ProximalAdagradOptimizer is supported only in V1.
    with ops.Graph().as_default(), self.cached_session():
      val0, val1 = self.applyOptimizer(
          proximal_adagrad.ProximalAdagradOptimizer(
              3.0,
              initial_accumulator_value=0.1,
              l1_regularization_strength=0.0,
              l2_regularization_strength=0.0),
          is_sparse=True)

    with ops.Graph().as_default(), self.cached_session():
      val2, val3 = self.applyOptimizer(
          adagrad.AdagradOptimizer(
              3.0, initial_accumulator_value=0.1),
          is_sparse=True)

    self.assertAllClose(val0, val2)
    self.assertAllClose(val1, val3)


if __name__ == "__main__":
  test.main()
