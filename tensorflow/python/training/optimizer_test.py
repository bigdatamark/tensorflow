# [MODULE DOC]
# Functional tests for the base v1 training optimizer API, exercised through the gradient descent
# optimizer (and Adam for the slot test). Covers minimize in graph and eager modes, aggregation
# methods, precomputed gradient seeds, error cases when there are no variables or no gradients,
# gradients converted to variables, computing gradients with respect to plain tensors, registration
# in the training-op collection, variable constraints, and slot lookup under a mirrored
# distribution strategy. Runs through the standard test main entry point.
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
"""Functional test for optimizer."""

from tensorflow.python.distribute import cross_device_ops
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import adam
from tensorflow.python.training import gradient_descent


class OptimizerTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the generic optimizer behavior (minimize, compute and apply gradients, error
  # handling, collections, constraints, slots). No shared fixtures; each test creates its own
  # variables, usually looping over half, single and double precision floats, and uses plain
  # gradient descent with learning rate 3.0 for predictable numbers.

  @test_util.run_in_graph_and_eager_modes
  def testBasic(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): for each of three float types creates two resource variables
    # and a global step, defines a loss function of a weighted sum, minimizes it with gradient
    # descent (learning rate 3.0) given an explicit variable list, and after initialization checks
    # initial values, evaluates the resulting operation once, and asserts the variables moved by
    # minus learning rate times the coefficients (-14,-13 and -6,-5).
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      # Note that we name the variables uniquely here since the variables don't
      # seem to be getting deleted at the end of the loop.
      var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype,
                                                    name='a_%d' % i)
      var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype,
                                                    name='b_%d' % i)
      def loss():
        # [FUNCTION DOC]
        # HELPER: loss callback for minimize in this test.
        # Returns five times the first variable plus three times the second (an elementwise
        # tensor); uses the variables from the enclosing loop iteration. No side effects.
        return 5 * var0 + 3 * var1  # pylint: disable=cell-var-from-loop
      # Note that for eager execution, minimize expects a function instead of a
      # Tensor.
      global_step = resource_variable_ops.ResourceVariable(
          array_ops.zeros([], dtypes.int64), name='global_step_%d' % i)
      sgd_op = gradient_descent.GradientDescentOptimizer(3.0)

      self.evaluate(variables.global_variables_initializer())
      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], self.evaluate(var0))
      self.assertAllClose([3.0, 4.0], self.evaluate(var1))
      # Run 1 step of sgd through optimizer
      opt_op = sgd_op.minimize(loss, global_step, [var0, var1])
      self.evaluate(opt_op)
      # Validate updated params
      self.assertAllClose([-14., -13.], self.evaluate(var0))
      self.assertAllClose([-6., -5.], self.evaluate(var1))

  @test_util.run_deprecated_v1
  def testAggregationMethod(self):
    # [FUNCTION DOC]
    # Test (graph mode only): minimizes a weighted-sum cost of two variables with a global step
    # using the experimental accumulate-n gradient aggregation method for three float types. Runs
    # one step and asserts the same updated values as the basic test (-14,-13 and -6,-5), showing
    # the aggregation choice does not change results.
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        cost = 5 * var0 + 3 * var1
        global_step = variables.Variable(
            array_ops.zeros([], dtypes.int64), name='global_step')
        sgd_op = gradient_descent.GradientDescentOptimizer(3.0)
        opt_op = sgd_op.minimize(
            cost,
            global_step, [var0, var1],
            aggregation_method=gradients_util.AggregationMethod.
            EXPERIMENTAL_ACCUMULATE_N)

        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))
        # Run 1 step of sgd through optimizer
        opt_op.run()
        # Validate updated params
        self.assertAllClose([-14., -13.], self.evaluate(var0))
        self.assertAllClose([-6., -5.], self.evaluate(var1))

  @test_util.run_deprecated_v1
  def testPrecomputedGradient(self):
    # [FUNCTION DOC]
    # Test (graph mode only): minimizes a weighted-sum cost while providing an initial upstream
    # gradient constant (42, -42) for three float types. After one step asserts each variable
    # equals its start minus learning rate times coefficient times the supplied gradient.
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      with self.cached_session():
        var0 = variables.Variable([1.0, 2.0], dtype=dtype)
        var1 = variables.Variable([3.0, 4.0], dtype=dtype)
        cost = 5 * var0 + 3 * var1
        grad_loss = constant_op.constant([42, -42], dtype=dtype)
        global_step = variables.Variable(
            array_ops.zeros([], dtypes.int64), name='global_step')
        sgd_op = gradient_descent.GradientDescentOptimizer(3.0)
        opt_op = sgd_op.minimize(
            cost, global_step, [var0, var1], grad_loss=grad_loss)

        self.evaluate(variables.global_variables_initializer())
        # Fetch params to validate initial values
        self.assertAllClose([1.0, 2.0], self.evaluate(var0))
        self.assertAllClose([3.0, 4.0], self.evaluate(var1))
        # Run 1 step of sgd through optimizer
        opt_op.run()
        # Validate updated params
        self.assertAllClose([1.0 - 3 * 5 * 42.0, 2.0 - 3 * 5 * (-42.0)],
                            self.evaluate(var0))
        self.assertAllClose([3.0 - 3 * 3 * 42.0, 4.0 - 3 * 3 * (-42.0)],
                            self.evaluate(var1))

  @test_util.run_in_graph_and_eager_modes
  def testNoVariables(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): loss function creates only non-trainable variables. Calling
    # minimize without a variable list must raise a value error whose message matches "no ...
    # variables", checked for three float types.
    for dtype in [dtypes.half, dtypes.float32, dtypes.float64]:
      # pylint: disable=cell-var-from-loop
      def loss():
        # [FUNCTION DOC]
        # HELPER: loss callback for the no-variables test.
        # Creates two non-trainable resource variables each time it is called and returns five
        # times the first plus the second. Side effect: creates variables.
        var0 = resource_variable_ops.ResourceVariable(
            [1.0, 2.0], dtype=dtype, trainable=False, name='a')
        var1 = resource_variable_ops.ResourceVariable(
            [3.0, 4.0], dtype=dtype, trainable=False, name='b')
        return 5 * var0 + var1
      # pylint: enable=cell-var-from-loop
      sgd_op = gradient_descent.GradientDescentOptimizer(3.0)
      with self.assertRaisesRegex(ValueError, 'No.*variables'):
        sgd_op.minimize(loss)

  @test_util.run_in_graph_and_eager_modes
  def testNoGradients(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): the loss depends only on the first variable but minimize is
    # asked to train only the second. Expects a value error mentioning that no gradients were
    # provided, for three float types.
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      # Note that we name the variables uniquely here since the variables don't
      # seem to be getting deleted at the end of the loop.
      var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype,
                                                    name='a%d' % i)
      var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype,
                                                    name='b%d' % i)
      # pylint: disable=cell-var-from-loop
      def loss():
        # [FUNCTION DOC]
        # HELPER: loss callback for the no-gradients test.
        # Returns five times the first variable only, so the second variable gets no gradient.
        return 5 * var0
      # pylint: enable=cell-var-from-loop
      sgd_op = gradient_descent.GradientDescentOptimizer(3.0)
      with self.assertRaisesRegex(ValueError, 'No gradients'):
        # var1 has no gradient
        sgd_op.minimize(loss, var_list=[var1])

  @test_util.run_in_graph_and_eager_modes
  def testNoGradientsForAnyVariables_Minimize(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): the loss is a constant independent of the variables. Minimizing
    # over both variables must raise a value error saying no gradients were provided for any
    # variable, for three float types.
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      # Note that we name the variables uniquely here since the variables don't
      # seem to be getting deleted at the end of the loop.
      var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype,
                                                    name='a_%d' % i)
      var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype,
                                                    name='b_%d' % i)
      def loss():
        # [FUNCTION DOC]
        # HELPER: loss callback for the no-gradients-for-any-variable test.
        # Returns a constant scalar five that does not depend on any variable.
        return constant_op.constant(5.0)
      sgd_op = gradient_descent.GradientDescentOptimizer(3.0)
      with self.assertRaisesRegex(ValueError,
                                  'No gradients provided for any variable'):
        sgd_op.minimize(loss, var_list=[var0, var1])

  @test_util.run_in_graph_and_eager_modes
  def testNoGradientsForAnyVariables_ApplyGradients(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): applying gradients where every gradient is null must raise a
    # value error saying no gradients were provided for any variable, for three float types.
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      # Note that we name the variables uniquely here since the variables don't
      # seem to be getting deleted at the end of the loop.
      var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype,
                                                    name='a_%d' % i)
      var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype,
                                                    name='b_%d' % i)
      sgd_op = gradient_descent.GradientDescentOptimizer(3.0)
      with self.assertRaisesRegex(ValueError,
                                  'No gradients provided for any variable'):
        sgd_op.apply_gradients([(None, var0), (None, var1)])

  @test_util.run_in_graph_and_eager_modes
  def testGradientsAsVariables(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): computes gradients of a weighted-sum loss, copies each gradient
    # into a new resource variable by assignment, initializes and runs the copies, then applies
    # those variable-valued gradients with apply-gradients. Asserts initial values and that one
    # step gives the usual updated values (-14,-13 and -6,-5), for three float types.
    for i, dtype in enumerate([dtypes.half, dtypes.float32, dtypes.float64]):
      # Note that we name the variables uniquely here since the variables don't
      # seem to be getting deleted at the end of the loop.
      var0 = resource_variable_ops.ResourceVariable([1.0, 2.0], dtype=dtype,
                                                    name='a%d' % i)
      var1 = resource_variable_ops.ResourceVariable([3.0, 4.0], dtype=dtype,
                                                    name='b%d' % i)
      def loss():
        # [FUNCTION DOC]
        # HELPER: loss callback used to compute gradients in this test.
        # Returns five times the first variable plus three times the second.
        return 5 * var0 + 3 * var1  # pylint: disable=cell-var-from-loop
      sgd_op = gradient_descent.GradientDescentOptimizer(3.0)
      grads_and_vars = sgd_op.compute_gradients(loss, [var0, var1])
      # Convert gradients to tf.Variables
      converted_grads = [
          resource_variable_ops.ResourceVariable(array_ops.zeros([2], dtype),
                                                 name='c_%d_%d' % (i, j))
          for j, gv in enumerate(grads_and_vars)
      ]
      convert_ops = [
          state_ops.assign(converted_grads[j], gv[0])
          for j, gv in enumerate(grads_and_vars)
      ]

      self.evaluate(variables.global_variables_initializer())
      # Run convert_ops to achieve the gradients converting
      self.evaluate(convert_ops)
      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], self.evaluate(var0))
      self.assertAllClose([3.0, 4.0], self.evaluate(var1))

      # Run 1 step of sgd through optimizer
      converted_grads_and_vars = list(zip(converted_grads, [var0, var1]))
      opt_op = sgd_op.apply_gradients(converted_grads_and_vars)
      self.evaluate(opt_op)

      # Validate updated params
      self.assertAllClose([-14., -13.], self.evaluate(var0))
      self.assertAllClose([-6., -5.], self.evaluate(var1))

  @test_util.run_in_graph_and_eager_modes
  def testComputeGradientsWithTensors(self):
    # [FUNCTION DOC]
    # Test (graph and eager modes): computes gradients of the square of a plain tensor with respect
    # to that tensor. Asserts one pair is returned, the second element is the same tensor object,
    # and the gradient equals 2. Then asserts that applying those gradients raises a
    # not-implemented error because the target is not a variable.
    x = ops.convert_to_tensor(1.0)
    def f():
      # [FUNCTION DOC]
      # HELPER: loss callback for the tensor-gradient test.
      # Returns the square of the captured tensor.
      return x * x
    sgd_op = gradient_descent.GradientDescentOptimizer(3.0)
    grads_and_vars = sgd_op.compute_gradients(f, [x])
    self.assertEqual(1, len(grads_and_vars))
    grad, x_as_var = grads_and_vars[0]
    self.assertIs(x, x_as_var)
    self.assertEqual(2.0, self.evaluate(grad))

    with self.assertRaises(NotImplementedError):
      sgd_op.apply_gradients(grads_and_vars)

  @test_util.run_deprecated_v1
  def testTrainOp(self):
    # [FUNCTION DOC]
    # Test (graph mode only): minimizes a weighted-sum cost and asserts the returned operation is
    # registered in the graph's training-op collection.
    with self.cached_session():
      var0 = variables.Variable([1.0, 2.0])
      var1 = variables.Variable([3.0, 4.0])
      cost = 5 * var0 + 3 * var1
      global_step = variables.Variable(
          array_ops.zeros([], dtypes.int64), name='global_step')
      sgd_op = gradient_descent.GradientDescentOptimizer(3.0)
      opt_op = sgd_op.minimize(cost, global_step, [var0, var1])
      self.assertTrue(opt_op in ops.get_collection(ops.GraphKeys.TRAIN_OP))

  @test_util.run_deprecated_v1
  def testConstraint(self):
    # [FUNCTION DOC]
    # Test (graph mode only): two variables carry constraints that clip values to the range minus
    # 0.1 to 0 and 0 to 1 respectively. After one gradient descent step the unconstrained results
    # would be -14,-13 and -6,-5, so the test asserts the clipped values (-0.1,-0.1) and (0,0).
    constraint_01 = lambda x: clip_ops.clip_by_value(x, -0.1, 0.)
    constraint_0 = lambda x: clip_ops.clip_by_value(x, 0., 1.)
    with self.cached_session():
      var0 = variables.Variable([1.0, 2.0],
                                constraint=constraint_01)
      var1 = variables.Variable([3.0, 4.0],
                                constraint=constraint_0)
      cost = 5 * var0 + 3 * var1
      global_step = variables.Variable(
          array_ops.zeros([], dtypes.int64), name='global_step')
      sgd_op = gradient_descent.GradientDescentOptimizer(3.0)
      opt_op = sgd_op.minimize(cost, global_step, [var0, var1])

      self.evaluate(variables.global_variables_initializer())
      # Fetch params to validate initial values
      self.assertAllClose([1.0, 2.0], self.evaluate(var0))
      self.assertAllClose([3.0, 4.0], self.evaluate(var1))
      # Run 1 step of sgd through optimizer
      opt_op.run()
      # Validate updated params
      self.assertAllClose([-0.1, -0.1], self.evaluate(var0))
      self.assertAllClose([0., 0.], self.evaluate(var1))

  @test_util.run_deprecated_v1
  def testGetSlotUnderDistributedStrategy(self):
    # [FUNCTION DOC]
    # Test (graph mode only): builds a mirrored strategy over a CPU and a GPU device name with
    # hierarchical-copy all-reduce (no real GPU is needed in graph mode) and an Adam optimizer
    # because it creates slots. Runs a per-replica function that creates a variable, asserts it is
    # a distributed variable, applies one gradient, then asserts slot names exist and each slot is
    # non-null and also a distributed variable.
    # Only run this test in graph mode so we don't need actual GPU.
    ds = mirrored_strategy.MirroredStrategy(
        ['CPU:0', 'GPU:0'],
        cross_device_ops=cross_device_ops.HierarchicalCopyAllReduce())
    # We need an optimizer that creates slots.
    optimizer = adam.AdamOptimizer()

    def f():
      # [FUNCTION DOC]
      # HELPER: per-replica function run by the strategy in the slot test.
      # Creates a one-element variable, asserts it is distributed, applies a single gradient with
      # the Adam optimizer (which creates slot variables on first use), then asserts every slot
      # name resolves to a non-null distributed slot variable. Returns nothing.
      v = variables.Variable([1.0])
      self.assertTrue(distribute_utils.is_distributed_variable(v))
      # Slot variables are created in the first call to apply_gradients.
      optimizer.apply_gradients([(ops.convert_to_tensor([1.0]), v)])
      self.assertTrue(optimizer.get_slot_names())
      for name in optimizer.get_slot_names():
        slot = optimizer.get_slot(v, name)
        self.assertIsNotNone(slot)
        self.assertTrue(distribute_utils.is_distributed_variable(slot))

    ds.run(f)


if __name__ == '__main__':
  test.main()
