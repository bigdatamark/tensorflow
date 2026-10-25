# [MODULE DOC]
# Unit tests for the loss scaling optimizer wrapper. Covers fixed loss scale applied through
# minimize and through gradient computation, dynamic loss scale growth and shrinking on
# non-finite gradients, interaction with optimizer slot variables, checkpoint save and restore of
# the loss scale state, and rejection of a null loss scale. Most tests run both without a
# distribution strategy and under a mirrored strategy over CPU (and GPU when available).
# Also defines small shared helpers: a strategy factory, the parameterized test case list, a
# gradient getter and a factory of identity functions that assert on their incoming gradient.
# Running the file as a program starts the test runner.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for MixedPrecisionLossScaleOptimizer."""

import os

from absl.testing import parameterized
from tensorflow.python.checkpoint import checkpoint as trackable_utils
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import custom_gradient
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent
from tensorflow.python.training import momentum
from tensorflow.python.training.experimental import loss_scale as loss_scale_module
from tensorflow.python.training.experimental import loss_scale_optimizer

# If called outside any strategy.scope() calls, this will return the default
# strategy.
default_strategy_fn = distribute_lib.get_strategy


def create_mirrored_strategy():
  # [FUNCTION DOC]
  # HELPER: strategy factory used by the parameterized test cases.
  # Takes no inputs. Returns a mirrored distribution strategy over the CPU and, when at least one
  # GPU is visible, also the first GPU; otherwise over the CPU alone.
  if context.num_gpus() >= 1:
    return mirrored_strategy.MirroredStrategy(['cpu:0', 'gpu:0'])
  else:
    return mirrored_strategy.MirroredStrategy(['cpu:0'])


TESTCASES = ({
    'testcase_name': 'Base',
    'strategy_fn': default_strategy_fn
}, {
    'testcase_name': 'Distribute',
    'strategy_fn': create_mirrored_strategy
})


def get_gradients(opt, loss, params):
  # [FUNCTION DOC]
  # HELPER: shared test utility returning only the gradients.
  # Inputs: opt, an optimizer; loss, the value to minimize; params, list of variables.
  # Calls the optimizer's gradient computation and returns a tuple holding just the gradients
  # (dropping the variables).
  grads_and_vars = opt.compute_gradients(loss, params)
  grads, _ = zip(*grads_and_vars)
  return grads


def create_identity_with_grad_check_fn(expected_gradient, expected_dtype=None):
  # [FUNCTION DOC]
  # HELPER: shared test utility that builds a gradient-checking identity function.
  # Inputs: expected_gradient, a number or tensor the incoming gradient must equal;
  # expected_dtype, optional dtype the gradient must have (not checked when omitted).
  # Returns a plain (undecorated) identity function whose custom gradient asserts equality with
  # the expected gradient through an assert operation and then passes the gradient through.
  # The plain wrapper exists because Keras has trouble serializing decorated functions.
  """Returns a function that asserts it's gradient has a certain value.

  This serves as a hook to assert intermediate gradients have a certain value.
  This returns an identity function. The identity's gradient function is also
  the identity function, except it asserts that the gradient equals
  `expected_gradient` and has dtype `expected_dtype`.

  Args:
    expected_gradient: The gradient function asserts that the gradient is this
      value.
    expected_dtype: The gradient function asserts the gradient has this dtype.

  Returns:
    An identity function whose gradient function asserts the gradient has a
    certain value.
  """
  @custom_gradient.custom_gradient
  def _identity_with_grad_check(x):
    # [FUNCTION DOC]
    # HELPER: custom-gradient identity used by the gradient checking factory.
    # Input x, a tensor. Returns the identity of x paired with the gradient function defined
    # inside it, as the custom gradient mechanism requires.
    """Function that asserts it's gradient has a certain value."""
    x = array_ops.identity(x)
    def grad(dx):
      # [FUNCTION DOC]
      # HELPER: gradient function of the checking identity.
      # Input dx, the incoming gradient. If an expected dtype was given, a plain assertion fails
      # when the dtype differs. Converts the expected gradient to a tensor of dx's dtype and adds
      # an assert-equal operation (with a control dependency on the forward input so it does
      # not run when the input pipeline is exhausted). Returns dx passed through an identity that
      # depends on the assertion, so evaluating it triggers the check.
      """Gradient function that asserts the gradient has a certain value."""
      if expected_dtype:
        assert dx.dtype == expected_dtype, (
            'dx.dtype should be %s but is: %s' % (expected_dtype, dx.dtype))
      expected_tensor = tensor_conversion.convert_to_tensor_v2(
          expected_gradient, dtype=dx.dtype, name='expected_gradient'
      )
      # Control dependency is to ensure input is available. It's possible the
      # dataset will throw a StopIteration to indicate there is no more data, in
      # which case we don't want to run the assertion.
      with ops.control_dependencies([x]):
        assert_op = check_ops.assert_equal(dx, expected_tensor)
      with ops.control_dependencies([assert_op]):
        dx = array_ops.identity(dx)
      return dx
    return x, grad
  # Keras sometimes has trouble serializing Lambda layers with a decorated
  # function. So we define and return a non-decorated function.
  def identity_with_grad_check(x):
    # [FUNCTION DOC]
    # HELPER: undecorated forwarding wrapper returned by the factory.
    # Input x, a tensor; returns the result of the decorated checking identity applied to x.
    return _identity_with_grad_check(x)
  return identity_with_grad_check


class MixedPrecisionLossScaleOptimizerTest(test.TestCase,
                                           parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering the loss scaling optimizer wrapper: fixed and dynamic loss scales,
  # slot variables, checkpointing and argument validation. Most tests are parameterized over no
  # strategy and a mirrored strategy and run in both graph and eager modes.

  def _run_if_in_graph_mode(self, val):
    # [FUNCTION DOC]
    # HELPER: evaluation helper for the tests.
    # Input val, an operation or tensor. In graph mode evaluates it in the test session; in eager
    # mode does nothing because the work has already run. Returns nothing.
    # Running only in graph mode is useful, because optimizers sometimes return
    # a value that, in Graph mode, is runnable with self.evaluate. But in Eager
    # mode, the optimizer already does the computations and the return value
    # cannot be run.
    if not context.executing_eagerly():
      self.evaluate(val)

  def _run_fn_with_grad_check(self, strategy, var, opt, expected_grad):
    # [FUNCTION DOC]
    # HELPER: builds the per-replica step used by the loss scale tests.
    # Inputs: strategy (for its replica count), var (variable to train), opt (the optimizer under
    # test), expected_grad (the gradient the scaled loss must produce at the variable).
    # Returns a no-argument callable that minimizes a loss equal to the gradient-checking
    # identity of the variable divided by the replica count, over the single variable.
    grad_check_fn = create_identity_with_grad_check_fn(
        expected_grad)
    loss = lambda: grad_check_fn(var) / strategy.num_replicas_in_sync
    return lambda: opt.minimize(loss, var_list=[var])

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def testFixedLossScaleAppliedToLossWithMinimize(self, strategy_fn):
    # [FUNCTION DOC]
    # Scenario: wrapped gradient descent (rate 2) with a fixed loss scale of 10 on a variable
    # starting at 5, run under each strategy. The test requires the replica count to divide the
    # loss scale. Runs one minimize step with a gradient check expecting the scaled value divided
    # by the replica count. Asserts the variable ends at 3 (gradient 1 after unscaling).
    with strategy_fn().scope() as strategy:
      var = variables.Variable([5.0])
      opt = gradient_descent.GradientDescentOptimizer(2.0)
      loss_scale = 10.
      opt = loss_scale_optimizer.MixedPrecisionLossScaleOptimizer(
          opt, loss_scale)
      # We need num_replicas_in_sync to divide loss_scale, otherwise loss_scale
      # / strategy.num_replicas_in_sync will not be exact, which could lead to
      # assertion failures due to rounding issues.
      self.assertEqual(loss_scale % strategy.num_replicas_in_sync, 0)
      run_fn = self._run_fn_with_grad_check(
          strategy, var, opt, loss_scale / strategy.num_replicas_in_sync)
      run_op = strategy.experimental_run(run_fn)
      self.evaluate(variables.global_variables_initializer())
      self._run_if_in_graph_mode(run_op)
      # The loss is the identity of the variable. Therefore the gradient is 1,
      # and so the variable will be init_val - grad * lr == 5 - 1 * 2 == 3
      self.assertAllClose([3.], self.evaluate(var))

  @test_util.deprecated_graph_mode_only
  def testFixedLossScaleAppliedToLossWithGetGradients(self):
    # [FUNCTION DOC]
    # Scenario (graph mode only, no strategy): fixed loss scale 10 around gradient descent.
    # Builds a loss as the gradient-checking identity of a variable, requests gradients via the
    # wrapper, initializes variables and evaluates the gradients. The embedded assertion checks
    # the intermediate gradient equals the loss scale; the test fails if that assertion fails.
    var = variables.Variable([2.0])
    opt = gradient_descent.GradientDescentOptimizer(1.0)
    loss_scale = 10.
    opt = loss_scale_optimizer.MixedPrecisionLossScaleOptimizer(opt, loss_scale)
    grad_check_fn = create_identity_with_grad_check_fn(loss_scale)
    loss = grad_check_fn(var)
    run_op = get_gradients(opt, loss, [var])
    self.evaluate(variables.global_variables_initializer())
    # This will cause an assertion to run, as
    # create_identity_with_grad_check_fn added an assertion op.
    self.evaluate(run_op)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def testDynamicLossScale(self, strategy_fn):
    # [FUNCTION DOC]
    # Scenario: dynamic loss scale starting at 2, growing by a factor of 2 every step, around
    # gradient descent with rate 2, under each strategy. An expected-gradient variable is used by
    # the gradient check. After the first step asserts the variable is 3; then the expected
    # gradient is doubled (the scale has doubled) and a second step must leave the variable at 1.
    strategy = strategy_fn()
    learning_rate = 2.
    expected_gradient = resource_variable_ops.ResourceVariable(
        learning_rate / strategy.num_replicas_in_sync)
    with strategy.scope():
      var = variables.Variable([5.0])
      opt = gradient_descent.GradientDescentOptimizer(learning_rate)
      loss_scale = loss_scale_module.DynamicLossScale(
          initial_loss_scale=2, increment_period=1, multiplier=2)
      opt = loss_scale_optimizer.MixedPrecisionLossScaleOptimizer(
          opt, loss_scale)
      self.assertEqual(
          loss_scale.initial_loss_scale % strategy.num_replicas_in_sync, 0)

      run_fn = self._run_fn_with_grad_check(strategy, var, opt,
                                            expected_gradient)
      run_op = strategy.experimental_run(run_fn)
      self.evaluate(variables.global_variables_initializer())
      self._run_if_in_graph_mode(run_op)
      # The loss is the identity of the variable. Therefore the gradient is 1,
      # and so the variable will be init_val - grad * lr == 5 - 1 * 2 == 3
      self.assertAllClose([3.], self.evaluate(var))

      # Loss scale will be double, so the expected gradient is also doubled.
      self.evaluate(
          expected_gradient.assign(2 * learning_rate /
                                   strategy.num_replicas_in_sync))
      run_op = strategy.experimental_run(run_fn)
      self._run_if_in_graph_mode(run_op)
      # As before, the 2 is subtracted from the variable, making it's new value
      # 1.
      self.assertAllClose([1.], self.evaluate(var))

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def testDynamicUpdate(self, strategy_fn):
    # [FUNCTION DOC]
    # Scenario: dynamic loss scale (initial 2, period 1, multiplier 2) with gradient descent rate
    # 1 on two variables. A step with finite gradients must subtract 2 from each element and
    # double the loss scale to 4. A step whose loss is not-a-number must leave the variable
    # unchanged and halve the loss scale back to 2. Runs under each strategy.
    with strategy_fn().scope() as strategy:
      var = variables.Variable([1.0, 2.0])
      opt = gradient_descent.GradientDescentOptimizer(1.0)
      loss_scale = loss_scale_module.DynamicLossScale(
          initial_loss_scale=2, increment_period=1, multiplier=2)
      opt = loss_scale_optimizer.MixedPrecisionLossScaleOptimizer(
          opt, loss_scale)

      # Test optimizer with finite gradients
      loss = lambda: var * 2.0 / strategy.num_replicas_in_sync
      run_fn = lambda: opt.minimize(loss, var_list=[var])
      run_op = strategy.experimental_run(run_fn)
      self.evaluate(variables.global_variables_initializer())
      self._run_if_in_graph_mode(run_op)
      # Gradient is 2, so variable will have 2 subtracted from it
      self.assertAllClose([-1.0, 0.0], self.evaluate(var))
      # Loss scale has doubled from 2 to 4
      self.assertEqual(4., self.evaluate(opt._loss_scale()))

      # Test optimizer with NaN gradients
      loss = lambda: var * float('NaN')
      run_fn = lambda: opt.minimize(loss, var_list=[var])
      run_op = strategy.experimental_run(run_fn)
      self._run_if_in_graph_mode(run_op)
      # Variable should not change from before, due to NaN gradients.
      self.assertAllClose(self.evaluate(var), [-1.0, 0.0])
      # Loss scale should half due to NaN gradients.
      self.assertEqual(2., self.evaluate(opt._loss_scale()))

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def testDynamicLossScaleWithSlots(self, strategy_fn):
    # [FUNCTION DOC]
    # Scenario: momentum optimizer (which has slot variables) wrapped with a dynamic loss scale
    # (initial 2, period 1, multiplier 4). Runs two steps. Asserts the variable moves by 1 and
    # then by 2 more (momentum accumulation of the unscaled gradient) and that the loss scale
    # grows to 8 and then 32 after the respective steps.
    with strategy_fn().scope() as strategy:
      var = variables.Variable([1.0, 2.0])
      # An SGD optimizer with momentum has slot variables.
      opt = momentum.MomentumOptimizer(1.0, momentum=1.)
      initial_loss_scale = 2.
      loss_scale = loss_scale_module.DynamicLossScale(
          initial_loss_scale=initial_loss_scale,
          increment_period=1,
          multiplier=4)
      opt = loss_scale_optimizer.MixedPrecisionLossScaleOptimizer(
          opt, loss_scale)
      loss = lambda: var / strategy.num_replicas_in_sync
      run_fn = lambda: opt.minimize(loss, var_list=[var])
      run_op = strategy.experimental_run(run_fn)
      self.evaluate(variables.global_variables_initializer())
      self._run_if_in_graph_mode(run_op)
      # The momentum accumulator starts at 0 and the gradient is 1. The
      # accumulator is incremented by the gradient, so it is now 1. Then the
      # variable is subtracted by the accumulator, so the variable is subtracted
      # by 1.
      self.assertAllClose([0.0, 1.0], self.evaluate(var))
      self.assertEqual(self.evaluate(opt._loss_scale()), initial_loss_scale * 4)

      run_op = strategy.experimental_run(run_fn)
      self._run_if_in_graph_mode(run_op)
      # The momentum accumulator was 1 before this step and the gradient is 1.
      # The accumulator is incremented by the gradient, so it is now 2. Then the
      # variable is subtracted by the accumulator, so the variable is subtracted
      # by 2.
      self.assertAllClose([-2., -1.], self.evaluate(var))
      self.assertEqual(
          self.evaluate(opt._loss_scale()), initial_loss_scale * 16)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def testCheckpoint(self, strategy_fn):
    # [FUNCTION DOC]
    # Scenario: dynamic loss scale (initial 1, period 2, multiplier 2) around a momentum
    # optimizer. Skipped (returns early) for a mirrored strategy in graph mode, a known
    # unsupported case. Runs a step and checks loss scale 1 and one good step; saves a
    # checkpoint of the optimizer; runs another step and checks loss scale 2 with zero good steps;
    # restores the checkpoint, requires all values consumed, runs restore operations and asserts
    # the loss scale and good-step count are back to 1 and 1. Writes into the test temp dir.
    strategy = strategy_fn()
    if (isinstance(strategy, mirrored_strategy.MirroredStrategy) and
        not context.executing_eagerly()):
      # TODO(b/121381184): Enable running the test in this case.
      return

    with self.test_session(), strategy.scope():
      # Build and run a simple model.
      var = variables.Variable([2.0])
      loss_scale = loss_scale_module.DynamicLossScale(
          initial_loss_scale=1., increment_period=2., multiplier=2.)
      opt = momentum.MomentumOptimizer(1.0, momentum=1.)
      opt = loss_scale_optimizer.MixedPrecisionLossScaleOptimizer(
          opt, loss_scale)
      run_fn = lambda: opt.minimize(lambda: var + 1., var_list=[var])
      opt_op = strategy.experimental_run(run_fn)
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(opt_op)
      self.assertEqual(self.evaluate(loss_scale()), 1.)
      self.assertEqual(self.evaluate(loss_scale._num_good_steps), 1)

      # Save a checkpoint.
      checkpoint = trackable_utils.Checkpoint(optimizer=opt)
      prefix = os.path.join(self.get_temp_dir(), 'ckpt')
      save_path = checkpoint.save(prefix)

      # Run model again.
      self.evaluate(strategy.experimental_run(run_fn))
      self.assertEqual(self.evaluate(loss_scale()), 2.)
      self.assertEqual(self.evaluate(loss_scale._num_good_steps), 0)

      # Load checkpoint and ensure loss scale is back to it's original value.
      status = checkpoint.restore(save_path)
      status.assert_consumed()
      status.run_restore_ops()
      self.assertEqual(self.evaluate(loss_scale()), 1.)
      self.assertEqual(self.evaluate(loss_scale._num_good_steps), 1)

  def testPassingNoneToLossScale(self):
    # [FUNCTION DOC]
    # Scenario: constructing the wrapper around gradient descent with a null loss scale. Asserts
    # a value error is raised whose message says the loss scale cannot be none.
    opt = gradient_descent.GradientDescentOptimizer(1.0)
    with self.assertRaisesRegex(ValueError, r'loss_scale cannot be None'):
      loss_scale_optimizer.MixedPrecisionLossScaleOptimizer(opt, None)


if __name__ == '__main__':
  test.main()
