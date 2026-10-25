# [MODULE DOC]
# Unit tests for the loss scale classes: the fixed loss scale and the dynamic loss scale, plus the
# getter that builds one from a specification. The fixed tests check the constant value, that
# updates always say to apply gradients, serialization, call result type and text form. The
# dynamic tests drive a loss scale through scripted sequences of finite and non-finite gradients
# (using a one-shot dataset iterator) and compare the resulting loss scale values with expected
# lists, with and without a mirrored distribution strategy, in graph and eager modes.
# Defines shared helpers: a strategy factory, the parameterized case list and an iterator maker.
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
"""Tests for LossScale classes.."""
from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training.experimental import loss_scale as loss_scale_module

# TODO(reedwm): Create test case using multiple graphs

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
    'testcase_name': 'base',
    'strategy_fn': default_strategy_fn
}, {
    'testcase_name': 'distribute',
    'strategy_fn': create_mirrored_strategy
})


class FixedLossScaleTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the fixed loss scale class: value, update behavior, serialization, call type
  # and text representation. No shared fixtures; each test builds its own loss scale.

  @test_util.run_in_graph_and_eager_modes
  def test_basic(self):
    # [FUNCTION DOC]
    # Scenario: fixed loss scale of 1000. Updating with a finite gradient and with a
    # not-a-number gradient must both return a plain boolean true (not a tensor, so callers need
    # no conditional) and leave the loss scale value at 1000.
    loss_scale_value = 1000
    loss_scale = loss_scale_module.FixedLossScale(loss_scale_value)

    update_op, should_apply = loss_scale.update([constant_op.constant(0.)])
    self.evaluate(update_op)
    # should_apply should be a bool instead of a tensor, so that a tf.cond does
    # not have to be built in the graph by the caller.
    self.assertIsInstance(should_apply, bool)
    self.assertTrue(should_apply)
    self.assertEqual(loss_scale_value, self.evaluate(loss_scale()))

    update_op, should_apply = loss_scale.update(
        [constant_op.constant(float('NaN'))])
    self.evaluate(update_op)
    self.assertIsInstance(should_apply, bool)
    self.assertTrue(should_apply)
    self.assertEqual(loss_scale_value, self.evaluate(loss_scale()))

  @test_util.run_in_graph_and_eager_modes
  def test_serialization(self):
    # [FUNCTION DOC]
    # Scenario: build a fixed loss scale from the number 123 via the getter, take its config,
    # rebuild it from the config and assert the rebuilt scale evaluates to 123.
    loss_scale = loss_scale_module.get(123)
    config = loss_scale.get_config()
    loss_scale = loss_scale_module.FixedLossScale.from_config(config)
    self.assertEqual(self.evaluate(loss_scale()), 123.)

  @test_util.run_in_graph_and_eager_modes
  def test_call_type(self):
    # [FUNCTION DOC]
    # Scenario: calling a fixed loss scale must return a tensor object.
    scalar = loss_scale_module.FixedLossScale(123)
    self.assertIsInstance(scalar(), tensor_lib.Tensor)

  @test_util.run_in_graph_and_eager_modes
  def test_repr(self):
    # [FUNCTION DOC]
    # Scenario: the text representation of a fixed loss scale of 123 must be the class name with
    # the value shown as a float (123.0) in parentheses.
    loss_scale = loss_scale_module.FixedLossScale(123)
    self.assertEqual(repr(loss_scale), 'FixedLossScale(123.0)')


def _get_example_iter(inputs):
  # [FUNCTION DOC]
  # HELPER: feeds the scripted finite/non-finite flags to the dynamic loss scale tests.
  # Input: inputs, a list of values. Builds a dataset with one element per value and returns a
  # one-shot iterator over it.
  dataset = dataset_ops.Dataset.from_tensor_slices(inputs)
  return dataset_ops.make_one_shot_iterator(dataset)


class DynamicLossScaleTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the dynamic loss scale class. Most tests are parameterized over no strategy and
  # a mirrored strategy and run in graph and eager modes. They share a helper that feeds scripted
  # sequences of finite or non-finite gradients and compares the loss scale after every step.

  def _get_tensor(self, is_finite):
    # [FUNCTION DOC]
    # HELPER: produces a gradient tensor for the scripted update tests.
    # Input is_finite, a boolean scalar tensor. Builds a conditional yielding 1 when true and
    # not-a-number when false. Without a distribution strategy returns it directly. With one,
    # calls a per-replica function so that only replica zero uses that value and other replicas
    # get 1; returns the per-replica result.
    tensor = cond.cond(is_finite, lambda: 1., lambda: float('NaN'))

    if not distribute_lib.has_strategy():
      return tensor

    def get():
      # [FUNCTION DOC]
      # HELPER: per-replica function of the gradient builder.
      # Takes no inputs; returns the captured tensor on the replica with id zero and the
      # number 1 on every other replica.
      rep_id = (
          distribute_lib.get_replica_context()
          .replica_id_in_sync_group)
      return cond.cond(
          math_ops.equal(rep_id, 0), lambda: tensor, lambda: 1.)

    distribution = distribute_lib.get_strategy()
    return distribution.extended.call_for_each_replica(get)

  def _test_helper(self,
                   inputs,
                   expected_outputs,
                   initial_loss_scale=1.,
                   increment_period=2,
                   multiplier=2):
    # [FUNCTION DOC]
    # HELPER: shared driver of the dynamic loss scale tests.
    # Inputs: inputs, list of booleans (true means the step has finite gradients); expected_outputs,
    # list of expected loss scale values after each step; initial_loss_scale (default 1),
    # increment_period (default 2) and multiplier (default 2) configure the dynamic loss scale.
    # Builds the loss scale, runs one update per input (an update op in graph mode, direct call in
    # eager mode) and inside each update asserts the returned should-apply flag equals the finite
    # flag. After each step records the loss scale value. Finally asserts the recorded list equals
    # the expected list. Returns nothing.
    loss_scale = loss_scale_module.DynamicLossScale(
        initial_loss_scale=initial_loss_scale,
        increment_period=increment_period,
        multiplier=multiplier)
    itr = _get_example_iter(inputs)

    def update():
      # [FUNCTION DOC]
      # HELPER: one scripted update step of the driver.
      # Takes no inputs. Reads the next flag from the iterator, builds the matching gradient,
      # updates the loss scale with it and creates an assert that the should-apply result equals
      # the flag. In eager mode returns nothing (the checks already ran). In graph mode returns
      # the update op passed through an identity that depends on the assert.
      is_finite = itr.get_next()
      grad = self._get_tensor(is_finite)
      update_op, should_apply_gradients = loss_scale.update([grad])
      assert_op = check_ops.assert_equal(should_apply_gradients, is_finite)
      if context.executing_eagerly():
        return
      with ops.control_dependencies([assert_op]):
        return array_ops.identity(update_op)

    actual_outputs = []

    if not context.executing_eagerly():
      update_op = update()
      self.evaluate(variables.global_variables_initializer())
    for _ in range(len(inputs)):
      if context.executing_eagerly():
        update()
      else:
        self.evaluate(update_op)
      actual_outputs.append(self.evaluate(loss_scale()))
    self.assertEqual(actual_outputs, expected_outputs)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_increase(self, strategy_fn):
    # [FUNCTION DOC]
    # Scenario: six finite steps with defaults (start 1, period 2, factor 2). Expects the loss
    # scale to double every second step: 1, 2, 2, 4, 4, 8.
    with strategy_fn().scope():
      inputs = [True] * 6
      expected_outputs = [1, 2, 2, 4, 4, 8]
      self._test_helper(inputs, expected_outputs)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_keep_increasing_until_capped(self, strategy_fn):
    # [FUNCTION DOC]
    # Scenario: start at a quarter of the largest 32-bit float and take six finite steps.
    # Expects one doubling, then growth that is capped at the largest 32-bit float and stays
    # there.
    with strategy_fn().scope():
      init_loss_scale = np.finfo(np.float32).max / 4
      max_float = np.finfo(np.float32).max

      inputs = [True] * 6
      # Output is capped the 2nd time it doubles.
      expected_outputs = [
          init_loss_scale, init_loss_scale * 2, init_loss_scale * 2, max_float,
          max_float, max_float
      ]

      self._test_helper(inputs, expected_outputs, init_loss_scale)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_decrease_every_step(self, strategy_fn):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the helper call is dedented out of the strategy scope block, so unlike the
    # sibling tests the scenario runs outside the strategy scope and the mirrored case does not
    # actually exercise the strategy.
    # Scenario: six non-finite steps starting at 1024. Expects the loss scale to halve each
    # step: 512, 256, 128, 64, 32, 16.
    with strategy_fn().scope():
      inputs = [False] * 6
      init_loss_scale = 1024
      expected_outputs = [512, 256, 128, 64, 32, 16]

    self._test_helper(inputs, expected_outputs, init_loss_scale)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_keep_decreasing_until_one(self, strategy_fn):
    # [FUNCTION DOC]
    # Scenario: six non-finite steps starting at 16. Expects halving down to 1 and then staying
    # at 1 (the loss scale never drops below one): 8, 4, 2, 1, 1, 1.
    with strategy_fn().scope():
      inputs = [False] * 6
      init_loss_scale = 16
      expected_outputs = [8, 4, 2, 1, 1, 1]

      self._test_helper(inputs, expected_outputs, init_loss_scale)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_nan_clear_good_step(self, strategy_fn):
    # [FUNCTION DOC]
    # Scenario: three finite steps, one non-finite step, one finite step (defaults). Expects
    # 1, 2, 2, 1, 1, showing a non-finite step halves the scale and resets the count of
    # consecutive good steps.
    with strategy_fn().scope():
      inputs = [True, True, True, False, True]
      expected_outputs = [1, 2, 2, 1, 1]
      self._test_helper(inputs, expected_outputs)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_trigger_loss_scale_update_each_step(self, strategy_fn):
    # [FUNCTION DOC]
    # Scenario: start 1, increment period 1; steps are three finite, one non-finite, two finite.
    # Expects 2, 4, 8, 4, 8, 16: growth on every good step and halving on the bad one.
    with strategy_fn().scope():
      init_loss_scale = 1
      increment_period = 1

      inputs = [True] * 3 + [False, True, True]
      expected_outputs = [2, 4, 8, 4, 8, 16]

      self._test_helper(inputs, expected_outputs, init_loss_scale,
                        increment_period)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_alternating_good_and_bad_gradients_trigger_each_step(
      self, strategy_fn):
    # [FUNCTION DOC]
    # Scenario: start 1, increment period 1, alternating finite and non-finite steps (nine in
    # all, ending finite). Expects the scale to alternate 2, 1, 2, 1, ... ending at 2.
    with strategy_fn().scope():
      init_loss_scale = 1
      increment_period = 1

      inputs = [True, False] * 4 + [True]
      expected_outputs = [2, 1, 2, 1, 2, 1, 2, 1, 2]
      self._test_helper(inputs, expected_outputs, init_loss_scale,
                        increment_period)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_alternating_good_and_bad_gradients_trigger_every_other_step(
      self, strategy_fn):
    # [FUNCTION DOC]
    # Scenario: start 32, increment period 2, alternating finite and non-finite steps (seven in
    # all). Because good steps never reach two in a row, expects only halvings:
    # 32, 16, 16, 8, 8, 4, 4.
    with strategy_fn().scope():
      init_loss_scale = 32
      increment_period = 2

      inputs = [True, False] * 3 + [True]
      expected_outputs = [32, 16, 16, 8, 8, 4, 4]
      self._test_helper(inputs, expected_outputs, init_loss_scale,
                        increment_period)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_nondefault_multiplier(self, strategy_fn):
    # [FUNCTION DOC]
    # Scenario: start 4 with multiplier 3 and steps good, good, bad, good, good. Expects
    # 4, 12, 4, 4, 12: growth multiplies by 3 and a bad step divides by 3 back to 4.
    with strategy_fn().scope():
      init_loss_scale = 4
      multiplier = 3
      inputs = [True, True, False, True, True]
      expected_outputs = [4, 12, 4, 4, 12]
      self._test_helper(
          inputs, expected_outputs, init_loss_scale, multiplier=multiplier)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_random_mix_good_and_bad_gradients(self, strategy_fn):
    # [FUNCTION DOC]
    # Scenario: start 4 with default period and factor and a fixed mixed sequence of eleven good
    # and bad steps. Expects 2, 2, 4, 4, 2, 2, 1, 1, 2, 2, 1.
    with strategy_fn().scope():
      init_loss_scale = 4
      inputs = [
          False, True, True, True, False, True, False, True, True, True, False
      ]
      expected_outputs = [2, 2, 4, 4, 2, 2, 1, 1, 2, 2, 1]
      self._test_helper(inputs, expected_outputs, init_loss_scale)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_single_tensor_gradient(self, strategy_fn):
    # [FUNCTION DOC]
    # Scenario: update a default dynamic loss scale with one finite tensor rather than a list.
    # Asserts the should-apply result evaluates to true.
    with strategy_fn().scope():
      loss_scale = loss_scale_module.DynamicLossScale()
      grad = constant_op.constant(4.0)
      _, should_apply = loss_scale.update(grad)
      self.assertTrue(self.evaluate(should_apply))

  @test_util.run_in_graph_and_eager_modes
  def test_serialization(self):
    # [FUNCTION DOC]
    # Scenario: dynamic loss scale (initial 1, period 2, multiplier 3) is turned into a config
    # and rebuilt from it. After initializing variables asserts the rebuilt scale evaluates to 1
    # and keeps increment period 2 and multiplier 3.
    loss_scale = loss_scale_module.DynamicLossScale(
        initial_loss_scale=1, increment_period=2, multiplier=3)
    config = loss_scale.get_config()
    loss_scale = loss_scale_module.DynamicLossScale.from_config(config)
    self.evaluate(variables.global_variables_initializer())
    self.assertEqual(self.evaluate(loss_scale()), 1)
    self.assertEqual(loss_scale.increment_period, 2)
    self.assertEqual(loss_scale.multiplier, 3)

  @test_util.run_in_graph_and_eager_modes
  def test_update_with_none_gradients(self):
    # [FUNCTION DOC]
    # Scenario: updating a default dynamic loss scale with a list holding only a null gradient.
    # Asserts nothing explicitly; passes if no exception is raised.
    loss_scale = loss_scale_module.DynamicLossScale()
    loss_scale.update([None])

  @test_util.run_in_graph_and_eager_modes
  def test_get(self):
    # [FUNCTION DOC]
    # Scenario: the getter called with the string "dynamic" must return a dynamic loss scale
    # whose initial value, increment period and multiplier equal those of a default-constructed
    # dynamic loss scale.
    scalar = loss_scale_module.get('dynamic')
    scalar2 = loss_scale_module.DynamicLossScale()
    self.assertEqual(scalar.initial_loss_scale, scalar2.initial_loss_scale)
    self.assertEqual(scalar.increment_period, scalar2.increment_period)
    self.assertEqual(scalar.multiplier, scalar2.multiplier)

  @test_util.run_in_graph_and_eager_modes
  def test_call_type(self):
    # [FUNCTION DOC]
    # Scenario: calling a default dynamic loss scale must return a tensor object.
    scalar = loss_scale_module.DynamicLossScale()
    self.assertIsInstance(scalar(), tensor_lib.Tensor)

  @parameterized.named_parameters(*TESTCASES)
  @test_util.run_in_graph_and_eager_modes
  def test_repr(self, strategy_fn):
    # [FUNCTION DOC]
    # Scenario: text form of a dynamic loss scale (initial 1, period 2, multiplier 3) under each
    # strategy. In eager mode it must also show the current loss scale and good step count;
    # in graph mode only the initial value, period and multiplier appear.
    with strategy_fn().scope():
      loss_scale = loss_scale_module.DynamicLossScale(
          initial_loss_scale=1, increment_period=2, multiplier=3)
      if context.executing_eagerly():
        self.assertEqual(repr(loss_scale),
                         'DynamicLossScale(current_loss_scale=1.0, '
                         'num_good_steps=0, initial_loss_scale=1.0, '
                         'increment_period=2, multiplier=3.0)')
      else:
        self.assertEqual(repr(loss_scale),
                         'DynamicLossScale(initial_loss_scale=1.0, '
                         'increment_period=2, multiplier=3.0)')


if __name__ == '__main__':
  test.main()
