# [MODULE DOC]
# Unit tests for the legacy graph-style Adam optimizer (training.adam.AdamOptimizer).
# Contains a numpy reference implementation of one Adam update step and a single test case class
# that compares optimizer results against it for dense, sparse, tensor/callable learning rates,
# shared optimizers across update ops, multiple graphs/sessions, eager slot uniqueness and XLA
# sharding of variables and slots. No import-time side effects; runs via the test main entry point.
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
"""Tests for Adam."""

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import ref_variable
from tensorflow.python.ops import resource_variable_ops
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
  # HELPER: numpy reference implementation used by the Adam optimizer tests to compute expected values.
  # Performs one Adam step. Inputs: param (current parameter array), g_t (gradient array), t (1-based
  # step number), m and v (previous first and second moment estimates), and optional hyperparameters
  # alpha (learning rate), beta1, beta2 and epsilon. Computes the bias-corrected step size, updates
  # the moment estimates and the parameter. Returns a tuple of the new parameter, new first moment
  # and new second moment. No side effects and no exceptions raised by this code itself.
  alpha_t = alpha * np.sqrt(1 - beta2**t) / (1 - beta1**t)

  m_t = beta1 * m + (1 - beta1) * g_t
  v_t = beta2 * v + (1 - beta2) * g_t * g_t

  param_t = param - alpha_t * m_t / (np.sqrt(v_t) + epsilon)
  return param_t, m_t, v_t


class AdamOptimizerTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the legacy AdamOptimizer. Has no instance state of its own; each test builds
  # its own graph, variables and optimizer and compares the results of three optimizer steps with
  # the numpy reference update. Shared scenario drivers are doTestSparse and doTestBasic.

  def doTestSparse(self, use_resource=False):
    # [FUNCTION DOC]
    # HELPER: shared driver for the sparse-gradient tests (testSparse and testResourceSparse).
    # Input use_resource (default false) selects resource variables instead of reference variables.
    # For each of half, float32 and float64 it creates two variables, applies indexed-slices
    # gradients that cover every row, runs three Adam steps in graph mode and asserts the beta power
    # accumulators equal the beta values raised to the step number and the variables match the numpy
    # reference. Returns nothing; assertion failures signal errors.
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        if use_resource:
          var0 = resource_variable_ops.ResourceVariable(var0_np)
          var1 = resource_variable_ops.ResourceVariable(var1_np)
        else:
          var0 = ref_variable.RefVariable(var0_np)
          var1 = ref_variable.RefVariable(var1_np)
        grads0_np_indices = np.array([0, 1], dtype=np.int32)
        grads0 = indexed_slices.IndexedSlices(
            constant_op.constant(grads0_np),
            constant_op.constant(grads0_np_indices), constant_op.constant([2]))
        grads1_np_indices = np.array([0, 1], dtype=np.int32)
        grads1 = indexed_slices.IndexedSlices(
            constant_op.constant(grads1_np),
            constant_op.constant(grads1_np_indices), constant_op.constant([2]))
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
          update.run()

          var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  def testSparse(self):
    # [FUNCTION DOC]
    # Runs the sparse gradient scenario in a fresh graph using reference variables.
    with ops.Graph().as_default():
      self.doTestSparse(use_resource=False)

  def testResourceSparse(self):
    # [FUNCTION DOC]
    # Runs the sparse gradient scenario in a fresh graph using resource variables.
    with ops.Graph().as_default():
      self.doTestSparse(use_resource=True)

  def testSparseDevicePlacement(self):
    # [FUNCTION DOC]
    # For int32 and int64 index types, builds a loss from a gather on a variable and minimizes it
    # with Adam in graph mode, forcing GPU placement when a GPU exists. Asserts only that all
    # optimizer ops can be placed and run without error.
    with ops.Graph().as_default():
      for index_dtype in [dtypes.int32, dtypes.int64]:
        with self.cached_session(force_gpu=test.is_gpu_available()):
          # If a GPU is available, tests that all optimizer ops can be placed on
          # it (i.e. they have GPU kernels).
          var = variables.Variable([[1.0], [2.0]])
          indices = constant_op.constant([0, 1], dtype=index_dtype)
          gathered_sum = math_ops.reduce_sum(array_ops.gather(var, indices))
          optimizer = adam.AdamOptimizer(3.0)
          minimize_op = optimizer.minimize(gathered_sum)
          self.evaluate(variables.global_variables_initializer())
          minimize_op.run()

  def testGatherGradientWithBadIndicesPolicy(self):
    # [FUNCTION DOC]
    # Builds a gather-nd with the ignore policy for bad indices (one index is negative), then
    # minimizes the result with Adam and runs one step. Asserts only that the gradient flows
    # through the optimizer without raising an error.
    with ops.Graph().as_default():
      with self.cached_session(force_gpu=test.is_gpu_available()):
        var = variables.Variable([1.0, 2.0])
        indices = constant_op.constant([[1], [-1], [0]], dtype=dtypes.int32)
        out = array_ops.gather_nd(var,
                                  array_ops.expand_dims(indices, axis=-1),
                                  batch_dims=0,
                                  bad_indices_policy="IGNORE")
        optimizer = adam.AdamOptimizer(2.0, 0.0, 1.0)
        minimize_op = optimizer.minimize(out)
        self.evaluate(variables.global_variables_initializer())
        minimize_op.run()

  def testSparseRepeatedIndices(self):
    # [FUNCTION DOC]
    # For each float dtype, applies Adam to one variable with a sparse gradient that repeats an
    # index and to another with the pre-aggregated equivalent gradient. Asserts both variables are
    # equal initially and after each of three steps, i.e. repeated indices are summed.
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
        with self.cached_session():
          repeated_index_update_var = variables.Variable(
              [[1.0], [2.0]], dtype=dtype)
          aggregated_update_var = variables.Variable(
              [[1.0], [2.0]], dtype=dtype)
          grad_repeated_index = indexed_slices.IndexedSlices(
              constant_op.constant(
                  [0.1, 0.1], shape=[2, 1], dtype=dtype),
              constant_op.constant([1, 1]),
              constant_op.constant([2, 1]))
          grad_aggregated = indexed_slices.IndexedSlices(
              constant_op.constant(
                  [0.2], shape=[1, 1], dtype=dtype),
              constant_op.constant([1]),
              constant_op.constant([2, 1]))
          repeated_update = adam.AdamOptimizer().apply_gradients(
              [(grad_repeated_index, repeated_index_update_var)])
          aggregated_update = adam.AdamOptimizer().apply_gradients(
              [(grad_aggregated, aggregated_update_var)])
          self.evaluate(variables.global_variables_initializer())
          self.assertAllClose(aggregated_update_var,
                              self.evaluate(repeated_index_update_var))
          for _ in range(3):
            repeated_update.run()
            aggregated_update.run()
            self.assertAllClose(aggregated_update_var,
                                self.evaluate(repeated_index_update_var))

  def doTestBasic(self, use_resource=False, use_callable_params=False):
    # [FUNCTION DOC]
    # HELPER: shared driver for the dense-gradient tests (testBasic, testResourceBasic,
    # testBasicCallableParams).
    # Inputs: use_resource (default false) chooses resource over reference variables;
    # use_callable_params (default false) intends to pass hyperparameters as callables.
    # KNOWN ISSUE: only the learning rate is actually passed to the optimizer; the beta1, beta2 and
    # epsilon values (callable or not) are computed but never used.
    # Skips the test when running eagerly without resource variables. For each float dtype it
    # builds variables, applies gradients, checks the beta accumulators exist, are in the optimizer
    # variables, have the requested variable type and are not visible from another graph, then runs
    # three steps comparing with the numpy reference and, for resource variables, checks the slot
    # name. Returns nothing.
    if context.executing_eagerly() and not use_resource:
      self.skipTest(
          "Skipping test with use_resource=False and executing eagerly.")
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      with self.session(graph=ops.Graph()):
        # Initialize variables for numpy implementation.
        m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
        var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
        grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
        var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
        grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

        if use_resource:
          var0 = resource_variable_ops.ResourceVariable(
              var0_np, name="var0_%d" % i)
          var1 = resource_variable_ops.ResourceVariable(
              var1_np, name="var1_%d" % i)
        else:
          var0 = ref_variable.RefVariable(var0_np)
          var1 = ref_variable.RefVariable(var1_np)
        grads0 = constant_op.constant(grads0_np)
        grads1 = constant_op.constant(grads1_np)

        learning_rate = lambda: 0.001
        beta1 = lambda: 0.9
        beta2 = lambda: 0.999
        epsilon = lambda: 1e-8
        if not use_callable_params:
          learning_rate = learning_rate()
          beta1 = beta1()
          beta2 = beta2()
          epsilon = epsilon()

        opt = adam.AdamOptimizer(learning_rate=learning_rate)
        update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
        opt_variables = opt.variables()
        beta1_power, beta2_power = opt._get_beta_accumulators()
        self.assertTrue(beta1_power is not None)
        self.assertTrue(beta2_power is not None)
        self.assertIn(beta1_power, opt_variables)
        self.assertIn(beta2_power, opt_variables)
        # Ensure that non-slot variables are the same type as the requested
        # variables.
        self.assertEqual(
            use_resource,
            resource_variable_ops.is_resource_variable(beta1_power))
        self.assertEqual(
            use_resource,
            resource_variable_ops.is_resource_variable(beta2_power))

        if not context.executing_eagerly():
          with ops.Graph().as_default():
            # Shouldn't return non-slot variables from other graphs.
            self.assertEqual(0, len(opt.variables()))
          self.evaluate(variables.global_variables_initializer())
          # Fetch params to validate initial values
          self.assertAllClose([1.0, 2.0], self.evaluate(var0))
          self.assertAllClose([3.0, 4.0], self.evaluate(var1))

        beta1_power, beta2_power = opt._get_beta_accumulators()

        # Run 3 steps of Adam
        for t in range(1, 4):
          if not context.executing_eagerly():
            self.evaluate(update)
          elif t > 1:
            opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

          self.assertAllCloseAccordingToType(0.9**(t + 1),
                                             self.evaluate(beta1_power))
          self.assertAllCloseAccordingToType(0.999**(t + 1),
                                             self.evaluate(beta2_power))

          var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
          var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

          # Validate updated params
          self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
          self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))
          if use_resource:
            self.assertEqual("var0_%d/Adam:0" % (i,),
                             opt.get_slot(var=var0, name="m").name)

  def testBasic(self):
    # [FUNCTION DOC]
    # Runs the dense scenario with reference variables inside a cached session.
    with self.cached_session():
      self.doTestBasic(use_resource=False)

  @test_util.run_in_graph_and_eager_modes
  @test_util.disable_tfrt("b/168527439: invalid runtime fallback "
                          "resource variable reference on GPU.")
  def testResourceBasic(self):
    # [FUNCTION DOC]
    # Runs the dense scenario with resource variables in both graph and eager modes.
    self.doTestBasic(use_resource=True)

  @test_util.disable_tfrt("b/153089059: cannot create half tensor on GPU.")
  def testBasicCallableParams(self):
    # [FUNCTION DOC]
    # Runs the dense scenario in eager mode with resource variables and a callable learning rate.
    with context.eager_mode():
      self.doTestBasic(use_resource=True, use_callable_params=True)

  def testTensorLearningRate(self):
    # [FUNCTION DOC]
    # In graph mode, creates Adam with a constant tensor as learning rate, runs three steps for
    # each float dtype and asserts the beta powers and variable values match the numpy reference.
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
        with self.cached_session():
          # Initialize variables for numpy implementation.
          m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
          var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
          grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
          var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
          grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

          var0 = variables.Variable(var0_np)
          var1 = variables.Variable(var1_np)
          grads0 = constant_op.constant(grads0_np)
          grads1 = constant_op.constant(grads1_np)
          opt = adam.AdamOptimizer(constant_op.constant(0.001))
          update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))
          self.evaluate(variables.global_variables_initializer())

          # Fetch params to validate initial values
          self.assertAllClose([1.0, 2.0], self.evaluate(var0))
          self.assertAllClose([3.0, 4.0], self.evaluate(var1))

          beta1_power, beta2_power = opt._get_beta_accumulators()

          # Run 3 steps of Adam
          for t in range(1, 4):
            self.assertAllCloseAccordingToType(0.9**t,
                                               self.evaluate(beta1_power))
            self.assertAllCloseAccordingToType(0.999**t,
                                               self.evaluate(beta2_power))
            update.run()

            var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
            var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

            # Validate updated params
            self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
            self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  def testSharing(self):
    # [FUNCTION DOC]
    # Creates two update ops from one optimizer on the same variables and alternates running them
    # for three steps. Asserts the shared beta accumulators and variables follow the numpy
    # reference, showing slots and accumulators are shared rather than duplicated.
    with ops.Graph().as_default():
      for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
        with self.cached_session():
          # Initialize variables for numpy implementation.
          m0, v0, m1, v1 = 0.0, 0.0, 0.0, 0.0
          var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
          grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
          var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
          grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

          var0 = variables.Variable(var0_np)
          var1 = variables.Variable(var1_np)
          grads0 = constant_op.constant(grads0_np)
          grads1 = constant_op.constant(grads1_np)
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
            self.assertAllCloseAccordingToType(0.9**t,
                                               self.evaluate(beta1_power))
            self.assertAllCloseAccordingToType(0.999**t,
                                               self.evaluate(beta2_power))
            if t % 2 == 0:
              update1.run()
            else:
              update2.run()

            var0_np, m0, v0 = adam_update_numpy(var0_np, grads0_np, t, m0, v0)
            var1_np, m1, v1 = adam_update_numpy(var1_np, grads1_np, t, m1, v1)

            # Validate updated params
            self.assertAllCloseAccordingToType(var0_np, self.evaluate(var0))
            self.assertAllCloseAccordingToType(var1_np, self.evaluate(var1))

  @test_util.disable_tfrt("b/168527439: invalid runtime fallback "
                          "resource variable reference on GPU.")
  def testTwoSessions(self):
    # [FUNCTION DOC]
    # Reuses one optimizer first eagerly, then in two separate graphs with their own sessions.
    # Asserts nothing explicitly; passes if applying gradients in each context raises no error,
    # i.e. the optimizer keys its state by graph.
    optimizer = adam.AdamOptimizer()

    with context.eager_mode():
      var0 = variables.Variable(np.array([1.0, 2.0]), name="v0")
      grads0 = constant_op.constant(np.array([0.1, 0.1]))
      optimizer.apply_gradients([(grads0, var0)])

    g = ops.Graph()
    with g.as_default():
      with session.Session():
        var0 = variables.Variable(np.array([1.0, 2.0]), name="v0")
        grads0 = constant_op.constant(np.array([0.1, 0.1]))
        optimizer.apply_gradients([(grads0, var0)])

    gg = ops.Graph()
    with gg.as_default():
      with session.Session():
        var0 = variables.Variable(np.array([1.0, 2.0]), name="v0")
        grads0 = constant_op.constant(np.array([0.1, 0.1]))

        # If the optimizer saves any state not keyed by graph the following line
        # fails.
        optimizer.apply_gradients([(grads0, var0)])

  @test_util.disable_tfrt("b/168527439: invalid runtime fallback "
                          "resource variable reference on GPU.")
  def testSlotsUniqueEager(self):
    # [FUNCTION DOC]
    # In eager mode, minimizes a sum of two resource variables with Adam and asserts the optimizer
    # reports six distinct variables: two beta accumulators plus two slots for each variable.
    with context.eager_mode():
      v1 = resource_variable_ops.ResourceVariable(1.)
      v2 = resource_variable_ops.ResourceVariable(1.)
      opt = adam.AdamOptimizer(1.)
      opt.minimize(lambda: v1 + v2)
      # There should be two non-slot variables, and two unique slot variables
      # for v1 and v2 respectively.
      self.assertEqual(6, len({id(v) for v in opt.variables()}))

  @test_util.deprecated_graph_mode_only
  def testXlaSharding(self):
    # [FUNCTION DOC]
    # Graph-mode test: splits two resource variables with an XLA mesh split, applies one Adam
    # step, and asserts the beta accumulators carry no sharding while the variables and their
    # m and v slots do have a sharding annotation.
    dtype = dtypes.float32
    with self.session(graph=ops.Graph()):
      # Initialize variables for numpy implementation.
      var0_np = np.array([1.0, 2.0], dtype=dtype.as_numpy_dtype)
      grads0_np = np.array([0.1, 0.1], dtype=dtype.as_numpy_dtype)
      var1_np = np.array([3.0, 4.0], dtype=dtype.as_numpy_dtype)
      grads1_np = np.array([0.01, 0.01], dtype=dtype.as_numpy_dtype)

      var0 = resource_variable_ops.ResourceVariable(var0_np, name="var0")
      var1 = resource_variable_ops.ResourceVariable(var1_np, name="var1")
      var0, var1 = [
          xla_sharding.mesh_split(
              v, np.array([0, 1]), [0], use_sharding_op=False)
          for v in (var0, var1)
      ]
      grads0 = constant_op.constant(grads0_np)
      grads1 = constant_op.constant(grads1_np)

      learning_rate = lambda: 0.001

      opt = adam.AdamOptimizer(learning_rate=learning_rate)
      update = opt.apply_gradients(zip([grads0, grads1], [var0, var1]))

      self.evaluate(variables.global_variables_initializer())
      self.evaluate(update)
      # The beta accumulators are not sharded.
      beta1_power, beta2_power = opt._get_beta_accumulators()
      self.assertIsNone(xla_sharding.get_tensor_sharding(beta1_power))
      self.assertIsNone(xla_sharding.get_tensor_sharding(beta2_power))

      # Variables and slots are sharded.
      for v in (var0, var1):
        self.assertIsNotNone(xla_sharding.get_tensor_sharding(v))
        for slot_name in ("m", "v"):
          slot = opt.get_slot(v, slot_name)
          self.assertIsNotNone(xla_sharding.get_tensor_sharding(slot))


if __name__ == "__main__":
  test.main()
