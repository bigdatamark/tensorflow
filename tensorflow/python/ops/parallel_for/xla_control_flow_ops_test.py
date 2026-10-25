# [MODULE DOC]
# Tests for pfor and vectorized_map combined with XLA (compile API, jit-compile functions, XLA jit
# scope), including while loops with loop-invariant conditions under control flow v2. Uses the shared
# parallel-for test case base. Runs under the standard test main when executed directly.
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
"""XLA tests for pfor."""
# pylint: disable=g-direct-tensorflow-import

from tensorflow.compiler.tf2xla.python import xla as xla_ops
from tensorflow.python.compiler.xla import jit
from tensorflow.python.compiler.xla import xla
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_v2_toggles
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.parallel_for import control_flow_ops as pfor_control_flow_ops
from tensorflow.python.ops.parallel_for.test_util import PForTestCase
from tensorflow.python.platform import test


@test_util.run_all_in_graph_and_eager_modes
class PForTest(PForTestCase):
  # [CLASS DOC]
  # Tests of vectorized maps and parallel loops (pfor) combined with XLA, run in both graph and eager
  # modes. Builds on the shared parallel-for test case base class; the constructor enables XLA devices.

  def __init__(self, method_name="runTest"):
    # [FUNCTION DOC]
    # Initializes the test case for the given test method name (default is the standard run-test name)
    # and enables XLA devices on the global TensorFlow context, a process-wide side effect.
    super(PForTest, self).__init__(method_name)
    context.context().enable_xla_devices()

  def test_xla_einsum(self):
    # [FUNCTION DOC]
    # Vectorizes ten iterations of XLA einsum ops over loop-invariant and loop-variant operands (one case
    # needs a transpose) and checks that the vectorized result equals the sequential loop result.
    num_loop = 10
    x_series = random_ops.random_uniform([num_loop, 9, 9])
    y_series = random_ops.random_uniform([num_loop, 9, 1])

    def loop_fn(i):
      # [FUNCTION DOC]
      # Per-iteration body for the einsum test: gathers invariant matrices (index zero) and iteration
      # dependent ones, applies five einsum combinations and returns the list of the five results.
      x = array_ops.gather(x_series, 0)  # invariant.
      y = array_ops.gather(y_series, 0)  # invariant.
      x_i = array_ops.gather(x_series, i)
      y_i = array_ops.gather(y_series, i)
      z1 = xla_ops.einsum(x_i, y, "ab,bc->ac")
      z2 = xla_ops.einsum(x, y_i, "ab,bc->ac")
      z3 = xla_ops.einsum(x, y, "ab,bc->ac")
      z4 = xla_ops.einsum(x_i, y_i, "ab,bc->ac")
      z5 = xla_ops.einsum(y_i, x_i, "cd,ce->de")  # Includes transpose.
      outputs = [z1, z2, z3, z4, z5]
      return outputs

    self._test_loop_fn(loop_fn, num_loop)

  def test_xla(self):
    # [FUNCTION DOC]
    # Compiles a vectorized map of a mean-reduction over a tensor of ones with shape 10 by 5 by 3 using
    # the XLA compile API and asserts the result equals ones of shape 10 by 1 by 3.

    def compute(x):
      # [FUNCTION DOC]
      # HELPER: mapped function for the test. Returns the mean of the input over its first axis, keeping
      # that axis with size one.
      return math_ops.reduce_mean(x, axis=0, keepdims=True)

    def vectorized_compute(x):
      # [FUNCTION DOC]
      # HELPER: wraps the vectorized map of the compute function over the leading dimension of x.
      # Returns the stacked per-slice results.
      return pfor_control_flow_ops.vectorized_map(compute, x)

    result = xla.compile(
        vectorized_compute, inputs=[array_ops.ones((10, 5, 3))])
    self.run_and_assert_equal(result, array_ops.ones((10, 1, 3)))

  def test_function_jit_compile(self):
    # [FUNCTION DOC]
    # Runs a vectorized map inside a function compiled with the jit-compile option on a tensor of ones
    # shaped 10 by 5 by 3 and asserts the result equals ones of shape 10 by 1 by 3.

    def compute(x):
      # [FUNCTION DOC]
      # HELPER: mapped function for the test. Returns the mean of the input over its first axis, keeping
      # that axis with size one.
      return math_ops.reduce_mean(x, axis=0, keepdims=True)

    @def_function.function(jit_compile=True)
    def vectorized_compute(x):
      # [FUNCTION DOC]
      # HELPER: jit-compiled function applying the vectorized map of the compute function to x and
      # returning the stacked results.
      return pfor_control_flow_ops.vectorized_map(compute, x)

    result = vectorized_compute(array_ops.ones((10, 5, 3)))
    self.run_and_assert_equal(result, array_ops.ones((10, 1, 3)))

  def test_xla_while_loop(self):
    # [FUNCTION DOC]
    # Uses a vectorized map inside a while loop that is compiled with XLA. Ten iterations each gather one
    # slice of a ones tensor shaped 10 by 5 by 3, map a mean over it and accumulate; asserts the final
    # value equals ones of shape 5 by 1 multiplied by ten.

    def compute(x):
      # [FUNCTION DOC]
      # HELPER: mapped function for the test. Returns the mean of the input over its first axis, keeping
      # that axis with size one.
      return math_ops.reduce_mean(x, axis=0, keepdims=True)

    def vectorized_compute(x, i):
      # [FUNCTION DOC]
      # HELPER: gathers slice i of x, vectorized-maps the mean over it and fixes the static shape of the
      # result to 5 by 1 so the while loop has consistent loop variable shapes. Returns that result.
      inp = array_ops.gather(x, i)
      output = pfor_control_flow_ops.vectorized_map(compute, inp)
      output.set_shape([5, 1])
      return output

    def while_compute(x):
      # [FUNCTION DOC]
      # HELPER: runs a while loop for ten iterations, starting from a zero counter and zeros of shape 5 by
      # 1, adding the vectorized computation for the current index each time. Returns the accumulated
      # tensor (the second loop variable).
      return while_loop.while_loop_v2(
          lambda i, _: i < 10,
          lambda i, y: (i + 1, y + vectorized_compute(x, i)),
          (0, array_ops.zeros([5, 1])))[1]

    result = xla.compile(while_compute, inputs=[array_ops.ones((10, 5, 3))])
    expected = array_ops.ones([5, 1]) * 10
    self.run_and_assert_equal(expected, result)

  def test_reduce_mean(self):
    # [FUNCTION DOC]
    # Checks the parallel-for configuration's reduce-mean inside a jit-compiled function: each of eight
    # iterations subtracts the cross-iteration mean from its row of a random 8 by 3 tensor. Asserts the
    # result equals the input minus its mean over the first axis.
    x = random_ops.random_uniform([8, 3])

    @def_function.function(jit_compile=True)
    def f():
      # [FUNCTION DOC]
      # HELPER: jit-compiled function building the pfor of the loop body over eight iterations and
      # returning its result.

      def loop_fn(i, pfor_config):
        # [FUNCTION DOC]
        # HELPER: per-iteration body taking the iteration index and the parallel-for configuration. Returns
        # the gathered row minus the mean of that row across all iterations.
        x_i = array_ops.gather(x, i)
        return x_i - pfor_config.reduce_mean(x_i)

      return pfor_control_flow_ops.pfor(loop_fn, 8)

    output = f()
    ans = x - math_ops.reduce_mean(x, axis=0)
    output_val, ans_val = self.evaluate([output, ans])
    self.assertAllClose(ans_val, output_val)


def _make_unstacked(cond, body, pfor_config):
  # [FUNCTION DOC]
  # HELPER: builds condition and body functions for a while loop whose condition is made loop
  # invariant for pfor. cond and body are the original condition and body callables, pfor_config is the
  # parallel-for configuration. Returns a pair (new condition, new body) operating on an extra leading
  # "not done" flag loop variable.

  def _cond(*args):
    # [FUNCTION DOC]
    # HELPER: new condition. Concatenates the not-done flags across iterations and returns true if any
    # iteration is still running.
    return math_ops.reduce_any(pfor_config.reduce_concat(args[0]))

  def _body(*args):
    # [FUNCTION DOC]
    # HELPER: new body. Updates the not-done flag by and-ing it with the original condition, runs the
    # original body, and keeps the new outputs only for iterations still not done (otherwise keeps the
    # old values). Returns a tuple of the flag followed by the selected values.
    not_done = args[0]
    args = args[1:]
    not_done = math_ops.logical_and(not_done, cond(*args))
    outputs = body(*args)
    return (not_done,) + tuple(
        array_ops.where_v2(not_done, x, y) for x, y in zip(outputs, args))

  return _cond, _body


@test_util.run_all_in_graph_and_eager_modes
class WhileV2Test(PForTestCase):
  # [CLASS DOC]
  # Tests of pfor over while loops under XLA with the version 2 control flow enabled, in both graph and
  # eager modes. The set-up and tear-down toggle control flow v2 and restore the previous setting.

  def setUp(self):
    # [FUNCTION DOC]
    # Records whether control flow v2 was enabled, enables it, then runs the base class set-up.
    # Side effect: changes the global control flow v2 setting for the test.
    self._enabled = control_flow_v2_toggles.control_flow_v2_enabled()
    control_flow_v2_toggles.enable_control_flow_v2()
    super(WhileV2Test, self).setUp()

  def tearDown(self):
    # [FUNCTION DOC]
    # Restores the previous control flow v2 setting (disables it if it was not enabled before) and then
    # runs the base class tear-down.
    if not self._enabled:
      control_flow_v2_toggles.disable_control_flow_v2()
    super(WhileV2Test, self).tearDown()

  def _test_loop_fn(self, loop_fn, iters, force_xla=False):
    # [FUNCTION DOC]
    # KNOWN ISSUE: overrides the base class helper with a different signature (no tolerance or fallback
    # arguments), so options of the base helper are not available in this class.
    # Runs loop_fn through pfor for iters iterations and compares to the same computation run under a
    # jit scope inside a function. With force_xla true (default false) it also compares against the
    # function compiled with the jit-compile option and with the XLA compile API. Returns nothing.

    def f():
      # [FUNCTION DOC]
      # HELPER: returns the pfor result of the loop body over the iteration count.
      return pfor_control_flow_ops.pfor(loop_fn, iters)

    @def_function.function
    def jit_f():
      # [FUNCTION DOC]
      # HELPER: function that evaluates the pfor function inside an experimental jit scope and returns its
      # result.
      with jit.experimental_jit_scope():
        return f()

    out = f()
    jit_out = jit_f()
    self.run_and_assert_equal(out, jit_out)
    # TODO(agarwal): The following may complain about uncompilable nodes. Hence
    # these are currently not enabled for all tests.
    if force_xla:
      out_exp_compile_f = def_function.function(jit_compile=True)(f)()
      self.run_and_assert_equal(out, out_exp_compile_f)
      out_xla_compile_f = xla.compile(f, inputs=[])
      self.run_and_assert_equal(out, out_xla_compile_f)

  def test_stateless_while(self):
    # [FUNCTION DOC]
    # Vectorizes a while loop whose trip count depends on the iteration (lengths 4, 0 and 2) and sums
    # gathered row elements; checks pfor against the sequential loop.
    x = random_ops.random_uniform([3, 5])
    lengths = constant_op.constant([4, 0, 2])

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: per-iteration body: gathers a row and its length, then runs a while loop adding row
      # elements up to that length. Returns the final loop variables (counter and sum).
      x_i = array_ops.gather(x, i)
      lengths_i = array_ops.gather(lengths, i)

      return while_loop.while_loop(
          lambda j, _: j < lengths_i,
          lambda j, t: (j + 1, t + array_ops.gather(x_i, j)),
          [0, 0.])

    self._test_loop_fn(loop_fn, 3)

  def test_while_with_variable(self):
    # [FUNCTION DOC]
    # Vectorizes a while loop that reads a resource variable. Skipped in graph mode (flaky with
    # sessions). Checks pfor against the sequential loop for three iterations.
    if not context.executing_eagerly():
      self.skipTest("Flaky with tf.Session")

    v = resource_variable_ops.ResourceVariable(5.)

    def loop_fn(_):
      # [FUNCTION DOC]
      # HELPER: per-iteration body ignoring its index: runs a four-step while loop adding the variable
      # value and returns the accumulated sum.
      _, output = while_loop.while_loop(
          lambda j, x: j < 4,
          lambda j, x: (j + 1, x + v),
          [0, 0.])
      return output

    self._test_loop_fn(loop_fn, 3)

  def test_while_unstacked_condition(self):
    # [FUNCTION DOC]
    # Vectorizes a while loop with a loop-invariant condition (four steps) whose body adds the iteration
    # index; checks pfor against the sequential loop, also forcing the XLA comparisons.

    def loop_fn(i):
      # [FUNCTION DOC]
      # HELPER: per-iteration body: runs a four-step while loop adding the iteration index. Returns the
      # final loop variables.
      return while_loop.while_loop(
          lambda j, x: j < 4,
          lambda j, x: (j + 1, x + i), [0, 0])

    self._test_loop_fn(loop_fn, 3, force_xla=True)

  def test_while_force_unstacked_condition(self):
    # [FUNCTION DOC]
    # Wraps the condition and body so the while condition becomes loop invariant for pfor, which lets XLA
    # compile without dynamic partitioning. Compares pfor against the sequential loop with the XLA
    # comparisons forced.
    # The while_loop in this setup is similar to the one in test_stateless_while
    # whose condition is loop variant. However here we wrap the cond and body of
    # the loop in a way that makes the while_loop condition pfor loop invariant.
    # This allows xla compilation to work since the vectorized code no longer
    # needs to perform dynamic partitioning of the inputs.
    x = random_ops.random_uniform([3, 5])
    lengths = constant_op.constant([4, 0, 2])

    def loop_fn(i, pfor_config):
      # [FUNCTION DOC]
      # HELPER: per-iteration body taking the index and pfor configuration; gathers a row and length,
      # builds the unstacked condition and body and runs the while loop from a true flag.
      x_i = array_ops.gather(x, i)
      lengths_i = array_ops.gather(lengths, i)

      def _cond(j, _):
        # [FUNCTION DOC]
        # HELPER: original loop condition: true while the counter is below this iteration's length.
        return j < lengths_i

      def _body(j, t):
        # [FUNCTION DOC]
        # HELPER: original loop body: increments the counter and adds the gathered row element to the sum.
        return (j + 1, t + array_ops.gather(x_i, j))

      cond, body = _make_unstacked(_cond, _body, pfor_config)
      return while_loop.while_loop(
          cond,
          body,
          [True, 0, 0.])

    self._test_loop_fn(loop_fn, 3, force_xla=True)


if __name__ == "__main__":
  test.main()
