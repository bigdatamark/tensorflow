# [MODULE DOC]
# Functional tests for the moving averages module: assign_moving_average (with and without
# zero-debias, naming of shadow variables, reuse), weighted_moving_average (float32 and
# bfloat16), and ExponentialMovingAverage (decay with/without num_updates, static and
# variable decay, zero-debias, control dependencies, eager use, average and restore names,
# variable scopes, device assignment, imported graphs, and copying of XLA sharding).
# Contains two test classes plus a small module-level helper; running the module as a script
# starts the test runner. No import-time side effects.
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
"""Functional test for moving_averages.py."""

import numpy as np

from tensorflow.python.compiler.xla.experimental import xla_sharding
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import moving_averages
from tensorflow.python.training import saver as saver_lib


class MovingAveragesTest(test.TestCase):
  # [CLASS DOC]
  # Tests for assign_moving_average and weighted_moving_average. Covers numeric results in
  # graph and eager modes, shadow variable naming across repeated calls and variable reuse, and
  # weighted averages for float32 and bfloat16 placeholders. No shared fixtures.

  @test_util.run_in_graph_and_eager_modes
  def testAssignMovingAverageWithoutZeroDebias(self):
    # [FUNCTION DOC]
    # Scenario: a two-element variable starting at 10 and 11, a constant value of 1 and 2, and
    # decay 0.25, with zero-debias turned off. Runs in both graph and eager modes (in graph mode
    # initializes variables and runs the returned op). Expects the variable to equal
    # old * decay + value * (1 - decay) elementwise afterwards, and the initial values before.
    var = variables.Variable([10.0, 11.0])
    val = constant_op.constant([1.0, 2.0], dtypes.float32)
    decay = 0.25
    if context.executing_eagerly():
      self.assertAllClose([10.0, 11.0], self.evaluate(var))
      assign = moving_averages.assign_moving_average(
          var, val, decay, zero_debias=False)
      self.assertAllClose(
          [10.0 * 0.25 + 1.0 * (1.0 - 0.25), 11.0 * 0.25 + 2.0 * (1.0 - 0.25)],
          self.evaluate(var))
    else:
      assign = moving_averages.assign_moving_average(
          var, val, decay, zero_debias=False)
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose([10.0, 11.0], self.evaluate(var))
      assign.op.run()
      self.assertAllClose(
          [10.0 * 0.25 + 1.0 * (1.0 - 0.25), 11.0 * 0.25 + 2.0 * (1.0 - 0.25)],
          self.evaluate(var))

  @test_util.run_in_graph_and_eager_modes
  def testAssignMovingAverage(self):
    # [FUNCTION DOC]
    # Scenario: a zero-initialized two-element variable, value 1 and 2, decay 0.25, default
    # zero-debias on, in graph and eager modes. Expects the variable to equal
    # value * (1 - decay) / (1 - decay), that is exactly the value, after the first update,
    # showing the debias removes the zero-initialization bias.
    var = variables.Variable([0.0, 0.0])
    val = constant_op.constant([1.0, 2.0], dtypes.float32)
    decay = 0.25
    if context.executing_eagerly():
      self.assertAllClose([0.0, 0.0], self.evaluate(var))
      assign = moving_averages.assign_moving_average(var, val, decay)
      self.assertAllClose(
          [1.0 * (1.0 - 0.25) / (1 - 0.25), 2.0 * (1.0 - 0.25) / (1 - 0.25)],
          self.evaluate(var))
    else:
      assign = moving_averages.assign_moving_average(var, val, decay)
      self.evaluate(variables.global_variables_initializer())
      self.assertAllClose([0.0, 0.0], self.evaluate(var))
      assign.op.run()
      self.assertAllClose(
          [1.0 * (1.0 - 0.25) / (1 - 0.25), 2.0 * (1.0 - 0.25) / (1 - 0.25)],
          self.evaluate(var))

  @test_util.deprecated_graph_mode_only
  def testAssignMovingAverageNewNamingMultipleCalls(self):
    # [FUNCTION DOC]
    # Graph-mode only. Inside nested variable scopes, creates a variable and calls
    # assign_moving_average twice on it. Expects the scope to contain exactly the original
    # variable plus two pairs of shadow variables (biased and local step), the second pair
    # carrying a numeric suffix, with the doubled scope prefix in their names.
    with variable_scope.variable_scope("scope1") as vs1:
      with variable_scope.variable_scope("scope2"):
        var = variables.Variable(1.0, name="Var")
        moving_averages.assign_moving_average(var, 0.0, 0.99)
        moving_averages.assign_moving_average(var, 0.0, 0.99)
    expected_names = ["scope1/scope2/Var:0",
                      "scope1/scope2/scope1/scope2/Var/biased:0",
                      "scope1/scope2/scope1/scope2/Var/local_step:0",
                      "scope1/scope2/scope1/scope2/Var/biased_1:0",
                      "scope1/scope2/scope1/scope2/Var/local_step_1:0"]
    actual_names = [v.name for v in vs1.global_variables()]
    self.assertSetEqual(set(expected_names), set(actual_names))

  @test_util.deprecated_graph_mode_only
  def testAssignMovingAverageNewNamingMultipleCallsWithReuse(self):
    # [FUNCTION DOC]
    # Graph-mode only. Calls assign_moving_average twice on a scoped variable, then re-enters the
    # scope with reuse enabled, fetches the variable again and calls it twice more. Asserts
    # nothing explicitly; passes if no error (such as a duplicate-variable error) is raised,
    # showing shadow variables are reused under a reusing scope.
    with variable_scope.variable_scope("scope1") as vs1:
      var = variable_scope.get_variable("Var", shape=[])
      moving_averages.assign_moving_average(var, 0.0, 0.99)
      moving_averages.assign_moving_average(var, 0.0, 0.99)
    with variable_scope.variable_scope(vs1, reuse=True):
      var = variable_scope.get_variable("Var", shape=[])
      moving_averages.assign_moving_average(var, 0.0, 0.99)
      moving_averages.assign_moving_average(var, 0.0, 0.99)

  @test_util.deprecated_graph_mode_only
  def testWeightedMovingAverage(self):
    # [FUNCTION DOC]
    # Graph-mode only. Builds a weighted moving average (decay 0.5) over float32 scalar
    # placeholders, evaluates it with two successive feeds (value 3 weight 4, then value 11
    # weight 22) and compares against numerator/denominator computed by hand from the decayed
    # sums of value*weight and weight.
    with self.cached_session() as sess:
      decay = 0.5
      weight = array_ops.placeholder(dtypes.float32, [])
      val = array_ops.placeholder(dtypes.float32, [])

      wma = moving_averages.weighted_moving_average(val, decay, weight)
      self.evaluate(variables.global_variables_initializer())

      # Get the first weighted moving average.
      val_1 = 3.0
      weight_1 = 4.0
      wma_array = sess.run(wma, feed_dict={val: val_1, weight: weight_1})
      numerator_1 = val_1 * weight_1 * (1.0 - decay)
      denominator_1 = weight_1 * (1.0 - decay)
      self.assertAllClose(numerator_1 / denominator_1, wma_array)

      # Get the second weighted moving average.
      val_2 = 11.0
      weight_2 = 22.0
      wma_array = sess.run(wma, feed_dict={val: val_2, weight: weight_2})
      numerator_2 = numerator_1 * decay + val_2 * weight_2 * (1.0 - decay)
      denominator_2 = denominator_1 * decay + weight_2 * (1.0 - decay)
      self.assertAllClose(numerator_2 / denominator_2, wma_array)

  @test_util.deprecated_graph_mode_only
  def testWeightedMovingAverageBfloat16(self):
    # [FUNCTION DOC]
    # Same scenario as the float32 weighted moving average test, but with bfloat16 placeholders.
    # The expected ratio is converted to bfloat16 before comparing with the evaluated result.
    with self.cached_session() as sess:
      decay = 0.5
      weight = array_ops.placeholder(dtypes.bfloat16, [])
      val = array_ops.placeholder(dtypes.bfloat16, [])

      wma = moving_averages.weighted_moving_average(val, decay, weight)
      self.evaluate(variables.global_variables_initializer())

      # Get the first weighted moving average.
      val_1 = 3.0
      weight_1 = 4.0
      wma_array = sess.run(wma, feed_dict={val: val_1, weight: weight_1})
      numerator_1 = val_1 * weight_1 * (1.0 - decay)
      denominator_1 = weight_1 * (1.0 - decay)
      self.assertAllClose(numerator_1 / denominator_1, wma_array)

      # Get the second weighted moving average.
      val_2 = 11.0
      weight_2 = 22.0
      wma_array = sess.run(wma, feed_dict={val: val_2, weight: weight_2})
      numerator_2 = numerator_1 * decay + val_2 * weight_2 * (1.0 - decay)
      denominator_2 = denominator_1 * decay + weight_2 * (1.0 - decay)
      self.assertAllClose(
          dtypes._np_bfloat16(numerator_2 / denominator_2), wma_array)


def _Repeat(value, dim):
  # [FUNCTION DOC]
  # HELPER: builds expected values for the exponential moving average tests.
  # Returns the value itself when dim is one, otherwise a list holding the value dim times.
  if dim == 1:
    return value
  return [value] * dim


class ExponentialMovingAverageTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the ExponentialMovingAverage class: numeric updates for scalar and vector
  # variables with and without num_updates, zero-debias and variable decay; control
  # dependencies; eager behavior; naming of averages and restore maps (also with scopes and
  # subsets); device placement; imported graphs; and XLA sharding propagation. Shared helpers
  # are the decay checker and the name-checking helpers.

  def _CheckDecay(self, ema, actual_decay, dim, dynamic_decay_value=None):
    # [FUNCTION DOC]
    # HELPER: shared checker for the average-variable tests.
    # Inputs: ema (the object under test), actual_decay (decay value expected to be effectively
    # applied), dim (1 for scalar variables, otherwise vector length), dynamic_decay_value
    # (optional; if given, assigned to the object's decay variable before use and again after
    # reinitialization). Creates two variables (10 and 30) and a plain tensor (their sum), applies
    # the object to all three, and asserts: moving-average collection holds just the two
    # variables, averages are not trainable, shadow names follow the naming convention, initial
    # averages equal the variables (zeros for the tensor), and after one and two updates averages
    # match the hand-computed decayed values (divided by the debias scale when zero-debias is on).

    def _Scale(dk, steps):
      # [FUNCTION DOC]
      # HELPER: expected debias divisor. Returns one minus decay to the power of steps when the
      # object has zero-debias enabled, otherwise one.
      if ema._zero_debias:
        return 1 - dk**steps
      else:
        return 1

    tens = _Repeat(10.0, dim)
    thirties = _Repeat(30.0, dim)
    var0 = variables.Variable(tens, name="v0")
    var1 = variables.Variable(thirties, name="v1")
    self.evaluate(variables.global_variables_initializer())
    # Note that tensor2 is not a Variable but just a plain Tensor resulting
    # from the sum operation.
    tensor2 = var0 + var1
    if dynamic_decay_value is not None:
      self.evaluate(ema._decay.assign(dynamic_decay_value))
    update = ema.apply([var0, var1, tensor2])
    avg0 = ema.average(var0)
    avg1 = ema.average(var1)
    avg2 = ema.average(tensor2)

    self.assertItemsEqual([var0, var1], variables.moving_average_variables())

    self.assertNotIn(avg0, variables.trainable_variables())
    self.assertNotIn(avg1, variables.trainable_variables())
    self.assertNotIn(avg2, variables.trainable_variables())
    self.evaluate(variables.global_variables_initializer())
    if dynamic_decay_value is not None:
      self.evaluate(ema._decay.assign(dynamic_decay_value))

    self.assertEqual("v0/ExponentialMovingAverage:0", avg0.name)
    self.assertEqual("v1/ExponentialMovingAverage:0", avg1.name)
    self.assertEqual("add/ExponentialMovingAverage:0", avg2.name)

    # Check initial values.
    self.assertAllClose(tens, self.evaluate(var0))
    self.assertAllClose(thirties, self.evaluate(var1))
    self.assertAllClose(_Repeat(10.0 + 30.0, dim), self.evaluate(tensor2))

    # Check that averages are initialized correctly.
    self.assertAllClose(tens, self.evaluate(avg0))
    self.assertAllClose(thirties, self.evaluate(avg1))
    # Note that averages of Tensor's initialize to zeros_like since no value
    # of the Tensor is known because the Op has not been run (yet).
    self.assertAllClose(_Repeat(0.0, dim), self.evaluate(avg2))

    # Update the averages and check.
    self.evaluate(update)
    dk = actual_decay

    expected = _Repeat(10.0 * dk + 10.0 * (1 - dk), dim)
    self.assertAllClose(expected, self.evaluate(avg0))
    expected = _Repeat(30.0 * dk + 30.0 * (1 - dk), dim)
    self.assertAllClose(expected, self.evaluate(avg1))
    expected = _Repeat(0.0 * dk + (10.0 + 30.0) * (1 - dk) / _Scale(dk, 1), dim)
    self.assertAllClose(expected, self.evaluate(avg2))

    # Again, update the averages and check.
    self.evaluate(update)
    expected = _Repeat((10.0 * dk + 10.0 * (1 - dk)) * dk + 10.0 * (1 - dk),
                       dim)
    self.assertAllClose(expected, self.evaluate(avg0))
    expected = _Repeat((30.0 * dk + 30.0 * (1 - dk)) * dk + 30.0 * (1 - dk),
                       dim)
    self.assertAllClose(expected, self.evaluate(avg1))
    expected = _Repeat(((0.0 * dk + (10.0 + 30.0) * (1 - dk)) * dk +
                        (10.0 + 30.0) * (1 - dk)) / _Scale(dk, 2), dim)
    self.assertAllClose(expected, self.evaluate(avg2))

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNoNumUpdates_Scalar(self):
    # [FUNCTION DOC]
    # Graph-mode only. Scalar variables, decay 0.25, no num_updates, no debias; runs the shared
    # decay checker expecting an effective decay of 0.25.
    ema = moving_averages.ExponentialMovingAverage(0.25)
    self._CheckDecay(ema, actual_decay=0.25, dim=1)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNoNumUpdates_Scalar_DynamicDecay(self):
    # [FUNCTION DOC]
    # Graph-mode only. Scalar case where the decay is a variable (initially 0.75) later set to
    # 0.25 by the checker; expects an effective decay of 0.25, with no num_updates or debias.
    decay_var = variables.Variable(0.75)
    ema = moving_averages.ExponentialMovingAverage(decay_var)
    self._CheckDecay(ema, actual_decay=0.25, dim=1, dynamic_decay_value=0.25)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNoNumUpdates_Scalar_Debias(self):
    # [FUNCTION DOC]
    # Graph-mode only. Scalar case with decay 0.25 and zero-debias enabled, no num_updates;
    # the checker expects debiased averages for the plain tensor input.
    ema = moving_averages.ExponentialMovingAverage(0.25, zero_debias=True)
    self._CheckDecay(ema, actual_decay=0.25, dim=1)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNoNumUpdates_Scalar_Debias_DynamicDecay(self):
    # [FUNCTION DOC]
    # Graph-mode only. Scalar case combining a variable decay (0.75, then set to 0.25) with
    # zero-debias; expects an effective decay of 0.25 and debiased results.
    decay_var = variables.Variable(0.75)
    ema = moving_averages.ExponentialMovingAverage(decay_var, zero_debias=True)
    self._CheckDecay(ema, actual_decay=0.25, dim=1, dynamic_decay_value=0.25)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNoNumUpdates_Vector(self):
    # [FUNCTION DOC]
    # Graph-mode only. Same as the scalar static-decay test but with five-element vectors.
    ema = moving_averages.ExponentialMovingAverage(0.25)
    self._CheckDecay(ema, actual_decay=0.25, dim=5)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNoNumUpdates_Vector_DynamicDecay(self):
    # [FUNCTION DOC]
    # Graph-mode only. Vector (length five) case with a variable decay set to 0.25 by the
    # checker, no num_updates, no debias.
    decay_var = variables.Variable(0.75)
    ema = moving_averages.ExponentialMovingAverage(decay_var)
    self._CheckDecay(ema, actual_decay=0.25, dim=5, dynamic_decay_value=0.25)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNoNumUpdates_Vector_Debias(self):
    # [FUNCTION DOC]
    # Graph-mode only. Vector case with decay 0.25 and zero-debias enabled, no num_updates.
    ema = moving_averages.ExponentialMovingAverage(0.25, zero_debias=True)
    self._CheckDecay(ema, actual_decay=0.25, dim=5)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNoNumUpdates_Vector_Debias_DynamicDecay(self):
    # [FUNCTION DOC]
    # Graph-mode only. Vector case with a variable decay (set to 0.25) and zero-debias enabled.
    decay_var = variables.Variable(0.75)
    ema = moving_averages.ExponentialMovingAverage(decay_var, zero_debias=True)
    self._CheckDecay(ema, actual_decay=0.25, dim=5, dynamic_decay_value=0.25)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNumUpdates_Scalar(self):
    # [FUNCTION DOC]
    # Graph-mode only. Scalar case with decay 0.25 and num_updates of 1, so the decay is capped
    # to (1+1)/(10+1), about 0.1818; the checker expects that effective decay.
    # With num_updates 1, the decay applied is 0.181818.
    ema = moving_averages.ExponentialMovingAverage(0.25, num_updates=1)
    self._CheckDecay(ema, actual_decay=0.181818, dim=1)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNumUpdates_Scalar_DynamicDecay(self):
    # [FUNCTION DOC]
    # Graph-mode only. Scalar case with a variable decay (set to 0.25) and num_updates of 1;
    # expects the capped effective decay of about 0.1818.
    decay_var = variables.Variable(0.75)
    # With num_updates 1, the decay applied is 0.181818.
    ema = moving_averages.ExponentialMovingAverage(decay_var, num_updates=1)
    self._CheckDecay(
        ema, actual_decay=0.181818, dim=1, dynamic_decay_value=0.25)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNumUpdates_Scalar_Debias(self):
    # [FUNCTION DOC]
    # Graph-mode only. Scalar case with decay 0.25, num_updates 1 and zero-debias; expects the
    # capped effective decay of about 0.1818 and debiased tensor averages.
    # With num_updates 1, the decay applied is 0.181818.
    ema = moving_averages.ExponentialMovingAverage(
        0.25, num_updates=1, zero_debias=True)
    self._CheckDecay(ema, actual_decay=0.181818, dim=1)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNumUpdates_Scalar_Debias_DynamicDecay(self):
    # [FUNCTION DOC]
    # Graph-mode only. Scalar case with variable decay, num_updates 1 and zero-debias; expects
    # the capped effective decay of about 0.1818.
    decay_var = variables.Variable(0.75)
    # With num_updates 1, the decay applied is 0.181818.
    ema = moving_averages.ExponentialMovingAverage(
        decay_var, num_updates=1, zero_debias=True)
    self._CheckDecay(
        ema, actual_decay=0.181818, dim=1, dynamic_decay_value=0.25)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNumUpdates_Vector(self):
    # [FUNCTION DOC]
    # Graph-mode only. Vector case with decay 0.25 and num_updates 1; expects the capped decay
    # of about 0.1818.
    # With num_updates 1, the decay applied is 0.181818.
    ema = moving_averages.ExponentialMovingAverage(0.25, num_updates=1)
    self._CheckDecay(ema, actual_decay=0.181818, dim=5)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNumUpdates_Vector_DynamicDecay(self):
    # [FUNCTION DOC]
    # Graph-mode only. Vector case with variable decay and num_updates 1; expects the capped
    # decay of about 0.1818.
    decay_var = variables.Variable(0.75)
    # With num_updates 1, the decay applied is 0.181818.
    ema = moving_averages.ExponentialMovingAverage(decay_var, num_updates=1)
    self._CheckDecay(
        ema, actual_decay=0.181818, dim=5, dynamic_decay_value=0.25)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNumUpdates_Vector_Debias(self):
    # [FUNCTION DOC]
    # Graph-mode only. Vector case with decay 0.25, num_updates 1 and zero-debias; expects the
    # capped decay of about 0.1818.
    # With num_updates 1, the decay applied is 0.181818.
    ema = moving_averages.ExponentialMovingAverage(
        0.25, num_updates=1, zero_debias=True)
    self._CheckDecay(ema, actual_decay=0.181818, dim=5)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNumUpdates_Vector_Debias_DynamicDecay(self):
    # [FUNCTION DOC]
    # Graph-mode only. Vector case with variable decay, num_updates 1 and zero-debias; expects
    # the capped decay of about 0.1818.
    decay_var = variables.Variable(0.75)
    # With num_updates 1, the decay applied is 0.181818.
    ema = moving_averages.ExponentialMovingAverage(
        decay_var, num_updates=1, zero_debias=True)
    self._CheckDecay(
        ema, actual_decay=0.181818, dim=5, dynamic_decay_value=0.25)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesWithControlDeps(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the assertion that the shadow variable's value op has no control inputs is
    # duplicated on two consecutive lines (harmless redundancy).
    # Graph-mode only. Applies an average to one variable inside a control dependency on an
    # increment of another variable. Expects the shadow variable's initializer and value op to
    # have no control inputs, so it can be initialized before the other variable; then, after
    # assigning 20 to the tracked variable and running the update op, the other variable has been
    # incremented to 1 and the average equals 17.5 (decay 0.25 from 10 toward 20).
    v0 = variables.Variable(0, name="v0")
    add_to_v0 = v0.assign_add(1)
    v1 = variables.Variable([10.0], name="v1")
    assign_to_v1 = v1.assign([20.0])
    ema = moving_averages.ExponentialMovingAverage(0.25)
    with ops.control_dependencies([add_to_v0]):
      ema_op = ema.apply([v1])
    # the moving average of v1 should not have any control inputs
    v1_avg = ema.average(v1)
    self.assertEqual([], v1_avg.initializer.control_inputs)
    self.assertEqual([], v1_avg.value().op.control_inputs)
    self.assertEqual([], v1_avg.value().op.control_inputs)
    # We should be able to initialize v1_avg before v0.
    self.evaluate(v1_avg.initializer)
    self.evaluate(v0.initializer)
    self.assertEqual([10.0], self.evaluate(v1_avg))
    # running ema_op should add to v0 (in addition to updating v1_avg)
    self.evaluate(assign_to_v1)
    self.evaluate(ema_op)
    self.assertEqual(1, self.evaluate(v0))
    self.assertEqual([17.5], self.evaluate(v1_avg))

  def testBasicEager(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite the name it has no eager-only decorator; it branches on whether
    # eager execution is active, so it runs in whatever mode the test environment uses.
    # Applies an average (decay 0.25, name "foo") to two variables (1 and 2), assigns 2 and 4,
    # applies again, and asserts the object's name, the average names (variable name plus foo)
    # and averages of 1.75 and 3.5.
    v0 = variables.Variable(1.0, name="v0")
    v1 = variables.Variable(2.0, name="v1")

    ema = moving_averages.ExponentialMovingAverage(0.25, name="foo")
    op = ema.apply([v0, v1])
    if not context.executing_eagerly():
      self.evaluate(variables.global_variables_initializer())
      self.evaluate(op)

    self.evaluate(v0.assign(2.0))
    self.evaluate(v1.assign(4.0))

    self.evaluate(ema.apply([v0, v1]))

    self.assertEqual("foo", ema.name)
    self.assertEqual("v0/foo", ema.average_name(v0))
    self.assertEqual("v1/foo", ema.average_name(v1))

    self.assertAllEqual(self.evaluate(ema.average(v0)), 1.75)
    self.assertAllEqual(self.evaluate(ema.average(v1)), 3.5)

  def averageVariablesNamesHelper(self, zero_debias):
    # [FUNCTION DOC]
    # HELPER: shared body for the average-name tests (not itself a test method).
    # Input: zero_debias boolean passed to the object. Creates variables v0, v1, a non-trainable
    # v2 and a tensor sum, checks average names before apply, applies to v0, v1 and the tensor,
    # and asserts the restore map keys are the three average names plus v2's name (plus the
    # biased and local-step shadow names of the tensor when debias is on). Also asserts each
    # shadow variable's op name equals the name predicted by the average-name method.
    v0 = variables.Variable(10.0, name="v0")
    v1 = variables.Variable(30.0, name="v1")
    # Add a non-trainable variable.
    v2 = variables.Variable(20.0, name="v2", trainable=False)
    tensor2 = v0 + v1
    ema = moving_averages.ExponentialMovingAverage(
        0.25, zero_debias=zero_debias, name="foo")
    self.assertEqual("foo", ema.name)
    self.assertEqual("v0/foo", ema.average_name(v0))
    self.assertEqual("v1/foo", ema.average_name(v1))
    self.assertEqual("add/foo", ema.average_name(tensor2))
    ema.apply([v0, v1, tensor2])
    vars_to_restore = ema.variables_to_restore()
    # vars_to_restore should contain the following:
    # {v0/foo : v0,
    #  v1/foo : v1,
    #  add/foo : add/foo,
    #  v2 : v2}
    expected_names = [
        ema.average_name(v0),
        ema.average_name(v1),
        ema.average_name(tensor2), v2.op.name
    ]
    if zero_debias:
      # vars_to_restore should also contain the following:
      #  {add/foo/biased: add/foo/biased,
      #  add/foo/local_step: add/foo/local_step}
      expected_names += [
          ema.average_name(tensor2) + "/biased",
          ema.average_name(tensor2) + "/local_step"
      ]
    self.assertEqual(sorted(expected_names), sorted(vars_to_restore.keys()))
    self.assertEqual(ema.average(v0).op.name, ema.average_name(v0))
    self.assertEqual(ema.average(v1).op.name, ema.average_name(v1))
    self.assertEqual(ema.average(tensor2).op.name, ema.average_name(tensor2))

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNames(self):
    # [FUNCTION DOC]
    # Graph-mode only. Runs the shared average-name helper with zero-debias enabled.
    self.averageVariablesNamesHelper(zero_debias=True)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNamesNoDebias(self):
    # [FUNCTION DOC]
    # Graph-mode only. Runs the shared average-name helper with zero-debias disabled.
    self.averageVariablesNamesHelper(zero_debias=False)

  @test_util.deprecated_graph_mode_only
  def averageVariablesNamesRespectScopeHelper(self, zero_debias):
    # [FUNCTION DOC]
    # HELPER: shared body for the scope-aware average-name tests.
    # KNOWN ISSUE: it carries the graph-mode-only test decorator although it is a helper (not
    # collected as a test) that takes an argument; the decorator is only effective when called.
    # Input: zero_debias boolean. Creates variables inside one scope and the average object
    # inside another scope, then asserts average names include the object's scope prefix, the
    # restore map contains the three average names plus the non-trainable variable's name (plus
    # prefixed biased and local-step names for the tensor when debias is on), and that shadow op
    # names equal the predicted average names.
    # See discussion on #2740.
    with variable_scope.variable_scope("scope1"):
      v0 = variables.Variable(10.0, name="v0")
      v1 = variables.Variable(30.0, name="v1")
      # Add a non-trainable variable.
      v2 = variables.Variable(20.0, name="v2", trainable=False)
      tensor2 = v0 + v1
    with variable_scope.variable_scope("scope2"):
      ema = moving_averages.ExponentialMovingAverage(
          0.25, zero_debias=zero_debias, name="foo")
      self.assertEqual("scope2/scope1/v0/foo", ema.average_name(v0))
      self.assertEqual("scope2/scope1/v1/foo", ema.average_name(v1))
      self.assertEqual("scope2/scope1/add/foo", ema.average_name(tensor2))
      ema.apply([v0, v1, tensor2])
      vars_to_restore = ema.variables_to_restore()
      # `vars_to_restore` should contain the following:
      # {scope2/scope1/v0/foo : v0,
      #  scope2/scope1/v1/foo : v1,
      #  scope2/scope1/add/foo : add/foo,
      #  scope1/v2 : v2}
      expected_names = [
          ema.average_name(v0),
          ema.average_name(v1),
          ema.average_name(tensor2), v2.op.name
      ]
      if zero_debias:
        # `vars_to_restore` should also contain the following:
        # {scope2/scope2/scope1/add/foo/biased: add/foo/biased,
        #  scope2/scope2/scope1/add/foo/local_step: add/foo/local_step}
        sc = "scope2/"
        expected_names += [
            sc + ema.average_name(tensor2) + "/biased",
            sc + ema.average_name(tensor2) + "/local_step"
        ]

      self.assertEqual(sorted(expected_names), sorted(vars_to_restore.keys()))
      self.assertEqual(ema.average(v0).op.name, ema.average_name(v0))
      self.assertEqual(ema.average(v1).op.name, ema.average_name(v1))
      self.assertEqual(ema.average(tensor2).op.name, ema.average_name(tensor2))

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNamesRespectScope(self):
    # [FUNCTION DOC]
    # Graph-mode only. Runs the scope-aware average-name helper with zero-debias enabled.
    self.averageVariablesNamesRespectScopeHelper(zero_debias=True)

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesNamesRespectScopeNoDebias(self):
    # [FUNCTION DOC]
    # Graph-mode only. Runs the scope-aware average-name helper with zero-debias disabled.
    self.averageVariablesNamesRespectScopeHelper(zero_debias=False)

  @test_util.deprecated_graph_mode_only
  def testSubsetAverageVariablesNames(self):
    # [FUNCTION DOC]
    # Graph-mode only. Asks for the restore map with an explicit subset (v0 and the tensor) of
    # variables to restore from averages, before applying. Expects keys to be the average names
    # of v0 and the tensor plus the plain names of v1 and the non-trainable v2. After applying to
    # all three, asserts shadow op names match the predicted average names.
    v0 = variables.Variable(10.0, name="v0")
    v1 = variables.Variable(30.0, name="v1")
    # Add a non-trainable variable.
    v2 = variables.Variable(20.0, name="v2", trainable=False)
    tensor2 = v0 + v1
    ema = moving_averages.ExponentialMovingAverage(0.25, name="foo_avg")
    self.assertEqual("v0/foo_avg", ema.average_name(v0))
    self.assertEqual("v1/foo_avg", ema.average_name(v1))
    self.assertEqual("add/foo_avg", ema.average_name(tensor2))
    vars_to_restore = ema.variables_to_restore([v0, tensor2])
    # vars_to_restore should contain the following:
    # {v0/foo_avg : v0,
    #  add/foo_avg : add
    #  v1 : v1,
    #  v2 : v2}
    self.assertEqual(
        sorted(vars_to_restore.keys()),
        sorted([
            ema.average_name(v0),
            ema.average_name(tensor2), v1.op.name, v2.op.name
        ]))
    ema.apply([v0, v1, tensor2])
    self.assertEqual(ema.average(v0).op.name, ema.average_name(v0))
    self.assertEqual(ema.average(v1).op.name, ema.average_name(v1))
    self.assertEqual(ema.average(tensor2).op.name, ema.average_name(tensor2))

  def testSubsetAverageVariablesNamesEager(self):
    # [FUNCTION DOC]
    # No mode decorator, so it runs in the default mode. Asks for the restore map with all three
    # variables listed (including a non-trainable one) and expects the three average names. Then
    # applies to two variables and asserts their shadow variable names (minus the output suffix)
    # equal the predicted average names.
    v0 = variables.Variable(10.0, name="v0")
    v1 = variables.Variable(30.0, name="v1")
    # Add a non-trainable variable.
    v2 = variables.Variable(20.0, name="v2", trainable=False)
    ema = moving_averages.ExponentialMovingAverage(0.25, name="foo_avg")
    self.assertEqual("v0/foo_avg", ema.average_name(v0))
    self.assertEqual("v1/foo_avg", ema.average_name(v1))
    vars_to_restore = ema.variables_to_restore([v0, v1, v2])
    self.assertAllEqual(
        sorted(vars_to_restore.keys()),
        sorted([
            ema.average_name(v0), ema.average_name(v1), ema.average_name(v2)
        ]))
    ema.apply([v0, v1])
    self.assertEqual(ema.average(v0).name[:-len(":0")], ema.average_name(v0))
    self.assertEqual(ema.average(v1).name[:-len(":0")], ema.average_name(v1))

  @test_util.deprecated_graph_mode_only
  def testAverageVariablesDeviceAssignment(self):
    # [FUNCTION DOC]
    # Graph-mode only. Places two variables (one an old-style reference variable) on different
    # job devices and applies the average under a third default device. Expects each variable's
    # shadow to sit on the variable's device, the old-style one to keep a colocation constraint
    # with it, and the shadow of the plain tensor to use the default device.
    with ops.device("/job:dev_v0"):
      v0 = variables.Variable(10.0, name="v0")
    with ops.device("/job:dev_v1"):
      v1 = gen_state_ops.variable(
          shape=[1],
          dtype=dtypes.float32,
          name="v1",
          container="",
          shared_name="")
      v1.set_shape([1])
    tensor2 = v0 + v1
    ema = moving_averages.ExponentialMovingAverage(0.25, name="foo_avg")
    with ops.device("/job:default"):
      ema.apply([v0, v1, tensor2])
    self.assertDeviceEqual("/job:dev_v0", ema.average(v0).device)
    self.assertDeviceEqual("/job:dev_v1", ema.average(v1).device)
    # However, the colocation property is maintained.
    self.assertEqual([b"loc:@v1"], ema.average(v1).op.colocation_groups())
    self.assertDeviceEqual("/job:default", ema.average(tensor2).device)

  def _ExportAndImportGraph(self, graph):
    # [FUNCTION DOC]
    # HELPER: round-trips a graph through meta-graph export and import.
    # Input: a graph. Exports it with all collection keys, imports the result into a fresh graph
    # (made the default only during the import) and returns the new graph.
    """Export and import graph into a new graph."""
    meta_graph = saver_lib.export_meta_graph(
        graph=graph, collection_list=graph.get_all_collection_keys())
    graph_copy = ops.Graph()
    with graph_copy.as_default():
      _ = saver_lib.import_meta_graph(meta_graph)
    return graph_copy

  @test_util.deprecated_graph_mode_only
  def testImportedGraphVariablesToRestore(self):
    # [FUNCTION DOC]
    # Graph-mode only. Builds a graph with one variable, copies it through export/import, and in
    # the copy asks the average object for its restore map. Expects exactly one entry, keyed
    # with the variable's average name, proving duplicate Python wrappers of the same imported
    # variable are not both added.
    g = ops.Graph()
    with g.as_default():
      variables.Variable(10.0, name="v")
    # Export and import the graph into a new graph.
    g_copy = self._ExportAndImportGraph(g)
    with g_copy.as_default():
      ema = moving_averages.ExponentialMovingAverage(0.25, name="foo_avg")
      vars_to_restore = ema.variables_to_restore()
      # There should only be one variable in vars_to_restore. This is important
      # to check because when importing from a GraphDef, TF makes duplicate
      # python Variable objects referring to the same underlying variable. We
      # need to be sure that two variables referring to the same variable don't
      # both get added to vars_to_restore.
      self.assertEqual(len(vars_to_restore), 1)
      self.assertIn("v/foo_avg", vars_to_restore)

  @test_util.deprecated_graph_mode_only
  def testCopyXlaSharding(self):
    # [FUNCTION DOC]
    # Graph-mode only. Creates a two-element variable with no sharding, applies a mesh split
    # sharding to it, initializes variables and applies the average. Expects the shadow variable
    # to carry the same XLA sharding as the source variable.
    ema = moving_averages.ExponentialMovingAverage(0.25, name="foo_avg")
    v = variables.Variable(_Repeat(10.0, 2), name="v")
    self.assertIsNone(xla_sharding.get_tensor_sharding(v))
    v = xla_sharding.mesh_split(v, np.array([0, 1]), [0], use_sharding_op=False)
    self.assertIsNotNone(xla_sharding.get_tensor_sharding(v))
    self.evaluate(variables.global_variables_initializer())
    ema.apply([v])
    avg = ema.average(v)
    self.assertEqual(
        xla_sharding.get_tensor_sharding(v),
        xla_sharding.get_tensor_sharding(avg))


if __name__ == "__main__":
  test.main()
