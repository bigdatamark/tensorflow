# [MODULE DOC]
# Unit tests for the mixed precision graph rewrite enable and disable functions. Covers wrapping
# an optimizer with the loss scale wrapper, error cases (non-optimizer input, already wrapped
# optimizer), registering custom wrapper classes with two or three arguments, whether the
# automatic mixed precision graph pass really runs (GPU only, graph and eager modes), and the
# warning logged when a session existed before the rewrite was enabled.
# The test class sets an environment variable so the graph pass runs on any GPU, and resets
# global state in its teardown. Running the file as a program starts the test runner.
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

import os

from absl.testing import parameterized

from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from tensorflow.python.training import gradient_descent as gradient_descent_v1
from tensorflow.python.training.experimental import loss_scale_optimizer as loss_scale_optimizer_v1
from tensorflow.python.training.experimental import mixed_precision
from tensorflow.python.training.experimental import mixed_precision_global_state


class MixedPrecisionTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the mixed precision graph rewrite functions. The class constant names the
  # environment variable that makes the graph pass ignore performance heuristics; setup sets it
  # and teardown restores it and disables the rewrite.

  IGNORE_PERF_VAR = 'TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE'

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup. Calls the parent setup, remembers the current value of the performance
    # ignoring environment variable (possibly unset) and sets it to one so the graph pass applies
    # on pre-Volta GPUs. Side effect: modifies the process environment.
    super(MixedPrecisionTest, self).setUp()
    # Enable the tests to be run on pre-Volta GPUs by telling the grappler pass
    # to ignore performance and always transform the graph.
    self._original_ignore_perf_value = os.getenv(self.IGNORE_PERF_VAR)
    os.environ[self.IGNORE_PERF_VAR] = '1'

  def tearDown(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: loss scale wrappers registered by the registration tests are never removed from
    # the module-level registry, so they leak into later tests in the same process.
    # Fixture teardown. Restores the environment variable to its saved value, or deletes it if it
    # was unset before. Disables the mixed precision graph rewrite (resetting the global option
    # and flag) and then calls the parent teardown.
    # Set the IGNORE_PERF_VAR variable back to it's original value.
    if self._original_ignore_perf_value is not None:
      os.environ[self.IGNORE_PERF_VAR] = self._original_ignore_perf_value
    else:
      del os.environ[self.IGNORE_PERF_VAR]

    mixed_precision.disable_mixed_precision_graph_rewrite_v1()
    super(MixedPrecisionTest, self).tearDown()

  @test_util.run_in_graph_and_eager_modes
  def test_wrap_optimizer(self):
    # [FUNCTION DOC]
    # Scenario: enable the rewrite on a gradient descent optimizer with loss scale 123. Asserts the
    # returned object is the loss scale wrapper class and its loss scale evaluates to 123.
    opt = gradient_descent_v1.GradientDescentOptimizer(1.0)
    opt = mixed_precision.enable_mixed_precision_graph_rewrite_v1(opt, 123.)
    self.assertIsInstance(
        opt, loss_scale_optimizer_v1.MixedPrecisionLossScaleOptimizer)
    self.assertEqual(self.evaluate(opt._loss_scale()), 123.)

  @test_util.run_in_graph_and_eager_modes
  def test_optimizer_errors(self):
    # [FUNCTION DOC]
    # Scenario: enabling with an invalid optimizer (the number one) must raise a value error whose
    # message says an optimizer instance was expected; enabling with an optimizer already wrapped
    # by the loss scale wrapper must raise a value error saying it must not already be wrapped.
    # After each failure asserts the automatic mixed precision option was not turned on.
    opt = 1
    expected_regex = ('"opt" must be an instance of a tf.train.Optimizer or '
                      'a tf.keras.optimizers.Optimizer, but got')
    with self.assertRaisesRegex(ValueError, expected_regex):
      mixed_precision.enable_mixed_precision_graph_rewrite_v1(opt)
    self.assertFalse(config.get_optimizer_experimental_options()
                     .get('auto_mixed_precision', False))

    opt = gradient_descent_v1.GradientDescentOptimizer(1.0)
    opt = loss_scale_optimizer_v1.MixedPrecisionLossScaleOptimizer(opt,
                                                                   'dynamic')
    with self.assertRaisesRegex(
        ValueError, '"opt" must not already be an instance of a '
        'MixedPrecisionLossScaleOptimizer.'):
      mixed_precision.enable_mixed_precision_graph_rewrite_v1(opt)
    self.assertFalse(config.get_optimizer_experimental_options()
                     .get('auto_mixed_precision', False))

  @test_util.run_in_graph_and_eager_modes()
  def test_register_loss_scale_wrapper_with_2_arguments(self):
    # [FUNCTION DOC]
    # Scenario: define a custom optimizer class and a wrapper class, register the wrapper class
    # alone (it serves as both factory and class), then enable the rewrite on an instance with
    # loss scale 123. Asserts the result is an instance of the wrapper and holds loss scale 123.
    class MyOptimizer:
      # [CLASS DOC]
      # HELPER: empty stand-in optimizer class used as the registration key in this test.
      pass

    class MyLossScaleOptimizer(MyOptimizer):
      # [CLASS DOC]
      # HELPER: stand-in loss scale wrapper that simply records the inner optimizer and the loss
      # scale it was constructed with.

      def __init__(self, inner_optimizer, loss_scale):
        # [FUNCTION DOC]
        # HELPER: stores the two arguments (inner optimizer and loss scale) as attributes.
        self.inner_optimizer = inner_optimizer
        self.loss_scale = loss_scale

    mixed_precision.register_loss_scale_wrapper(MyOptimizer,
                                                MyLossScaleOptimizer)
    opt = MyOptimizer()
    opt = mixed_precision.enable_mixed_precision_graph_rewrite_v1(opt, 123.)
    self.assertIsInstance(opt, MyLossScaleOptimizer)
    self.assertEqual(opt.loss_scale, 123.)

  @test_util.run_in_graph_and_eager_modes()
  def test_register_loss_scale_wrapper_with_3_arguments(self):
    # [FUNCTION DOC]
    # Scenario: like the two-argument registration test, but registers a separate factory function
    # together with the wrapper class. Enables the rewrite on an instance with loss scale 123 and
    # asserts the result is a wrapper instance with loss scale 123 and that the factory was called.
    class MyOptimizer:
      # [CLASS DOC]
      # HELPER: empty stand-in optimizer class used as the registration key in this test.
      pass

    class MyLossScaleOptimizer(MyOptimizer):
      # [CLASS DOC]
      # HELPER: stand-in loss scale wrapper that simply records the inner optimizer and the loss
      # scale it was constructed with.

      def __init__(self, inner_optimizer, loss_scale):
        # [FUNCTION DOC]
        # HELPER: stores the two arguments (inner optimizer and loss scale) as attributes.
        self.inner_optimizer = inner_optimizer
        self.loss_scale = loss_scale

    is_called = False

    def create_lso(inner_optimizer, loss_scale):
      # [FUNCTION DOC]
      # HELPER: factory registered in this test. Inputs: the optimizer to wrap and the loss scale.
      # Sets the enclosing test's called flag to true and returns a new stand-in wrapper.
      nonlocal is_called
      is_called = True
      return MyLossScaleOptimizer(inner_optimizer, loss_scale)

    mixed_precision.register_loss_scale_wrapper(MyOptimizer,
                                                create_lso,
                                                MyLossScaleOptimizer)
    opt = MyOptimizer()
    opt = mixed_precision.enable_mixed_precision_graph_rewrite_v1(opt, 123.)
    self.assertIsInstance(opt, MyLossScaleOptimizer)
    self.assertEqual(opt.loss_scale, 123.)
    self.assertTrue(is_called)

  @test_util.run_gpu_only
  @test_util.run_in_graph_and_eager_modes
  @test_util.disable_tfrt('Grappler rewrite doesn\'t apply to tfrt.')
  def test_grappler_pass_enabled(self):
    # [FUNCTION DOC]
    # Scenario (GPU only, skipped under the TFRT runtime): enable the rewrite, then compute a
    # matrix product of a scaled variable that overflows in float16 but not float32. With the
    # rewrite the result must be infinity (in eager mode only inside a compiled function; plain
    # eager calls give 2 to the power 20; in graph mode for sessions with default or user-given
    # configuration). After disabling the rewrite the result must be 2 to the power 20 again.
    opt = gradient_descent_v1.GradientDescentOptimizer(1.0)
    mixed_precision.enable_mixed_precision_graph_rewrite_v1(opt, 123.)

    var = variables.Variable([[1.0]])

    def overflow_in_float16():
      # [FUNCTION DOC]
      # HELPER: workload for the graph pass test. Takes no inputs; multiplies the variable by 1024,
      # multiplies the result by itself as a matrix product and returns it reshaped to a scalar.
      # The value is 2 to the power 20 in float32 but overflows to infinity in float16.
      out = var * 2 ** 10
      out = math_ops.matmul(out, out)
      return array_ops.reshape(out, ())

    if context.executing_eagerly():
      f = def_function.function(overflow_in_float16)
      self.assertEqual(f().numpy(), float('Inf'))
      # Outside a def_function.function, the grappler pass will not be applied.
      self.assertAlmostEqual(overflow_in_float16().numpy(), 2 ** 20)

      # Test disabling mixed precision.
      mixed_precision.disable_mixed_precision_graph_rewrite_v1()
      self.assertEqual(f().numpy(), 2 ** 20)
    else:
      with session.Session() as sess:
        out = overflow_in_float16()
        sess.run(var.initializer)
        self.assertEqual(sess.run(out), float('Inf'))

      # Test Session will enable the auto_mixed_precision grappler pass in a
      # ConfigProto passed by the user
      with session.Session(config=config_pb2.ConfigProto()) as sess:
        out = overflow_in_float16()
        sess.run(var.initializer)
        self.assertEqual(sess.run(out), float('Inf'))

      # Test disabling mixed precision.
      mixed_precision.disable_mixed_precision_graph_rewrite_v1()
      with session.Session() as sess:
        out = overflow_in_float16()
        sess.run(var.initializer)
        self.assertAlmostEqual(sess.run(out), 2 ** 20)

  @test.mock.patch.object(tf_logging, 'warn')
  def test_warn_if_session_already_exists(self, mock_warn):
    # [FUNCTION DOC]
    # Scenario: the warning logger is mocked. Resets the non mixed precision session flag, creates
    # a session, then enables the rewrite. Asserts the logger was called with the message saying
    # existing sessions will not be affected.
    # Set this to False, so Sessions created in previous tests do not trigger
    # the warning.
    mixed_precision_global_state.set_non_mixed_precision_session_created(False)

    with session.Session():
      mixed_precision.enable_mixed_precision_graph_rewrite_v1(
          gradient_descent_v1.GradientDescentOptimizer(1.0))
      mock_warn.assert_any_call(
          'You already have existing Sessions that do not use mixed precision. '
          'enable_mixed_precision_graph_rewrite() will not affect these '
          'Sessions.')

  @test.mock.patch.object(tf_logging, 'warn')
  def test_do_not_warn_if_session_does_not_already_exist(self, mock_warn):
    # [FUNCTION DOC]
    # Scenario: the warning logger is mocked. Resets the session flag, enables the rewrite first,
    # then creates a session. Asserts none of the logged warnings contains the existing sessions
    # message.
    # Set this to False, so Sessions created in previous tests do not trigger
    # the warning.
    mixed_precision_global_state.set_non_mixed_precision_session_created(False)

    mixed_precision.enable_mixed_precision_graph_rewrite_v1(
        gradient_descent_v1.GradientDescentOptimizer(1.0))
    with session.Session():
      # Make sure the "You already have existing Sessions" warning was not
      # issued, since the Session was only created after
      # enable_mixed_precision_graph_rewrite.
      for call_arg in mock_warn.call_args_list:
        msg = call_arg[0][0]
        self.assertNotIn('You already have existing Sessions that do not use '
                         'mixed precision', msg)


if __name__ == '__main__':
  test.main()
