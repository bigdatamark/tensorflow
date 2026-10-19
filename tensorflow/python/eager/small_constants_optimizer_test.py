# [MODULE DOC]
# Performance and error-handling tests for the "runtime constant optimization" (small constants
# optimizer) of traced functions, which folds small boolean host tensors into constants so that
# branches are resolved at run time. Contains a context manager helper that temporarily sets
# experimental optimizer options, and one test class that compares timings of optimized and
# unoptimized traced functions (with and without the graph optimizer) and checks error behavior for
# GPU-resident and multi-element boolean inputs. When run as a script, eager execution is enabled
# before starting the test runner.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

import contextlib
import timeit

from tensorflow.python.eager import context
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


@contextlib.contextmanager
def options(optimizer_options):
  # [FUNCTION DOC]
  # HELPER: context manager used by the tests to temporarily change optimizer settings.
  # Input: a mapping of experimental optimizer option names to values. On entry it saves the
  # current experimental optimizer options of the global eager context and installs the given
  # ones; it yields nothing; on exit (even after an exception) it restores the saved options.
  # Side effect: mutates global eager context state for the duration of the with-block.
  old_opts = context.context().get_optimizer_experimental_options()
  context.context().set_optimizer_experimental_options(optimizer_options)
  try:
    yield
  finally:
    context.context().set_optimizer_experimental_options(old_opts)


class FunctionTest(test.TestCase):
  # [CLASS DOC]
  # Tests that traced functions containing a thousand data-dependent branches run much faster when
  # constant folding (by the graph optimizer or by the runtime small-constants optimization) is
  # active. Timing based: each test asserts a minimum speedup ratio of the optimized variant.

  @test_util.run_v2_only
  def test_grappler_optimization(self):
    # [FUNCTION DOC]
    # Scenario (v2 only): trace a function that branches on a boolean tensor argument one thousand
    # times, and a second function that branches on a constant true tensor created inside it.
    # Checks both return the same result, times each (best of five repeats of one hundred calls),
    # and asserts the constant-branch version is more than three times faster.
    @polymorphic_function.function
    def brancher(inp):
      # [FUNCTION DOC]
      # Traced function: starts from one and adds one or two, one thousand times, choosing the
      # increment by the boolean input. Returns the final sum tensor.
      x = constant_op.constant(1)
      for _ in range(1000):
        if inp:
          x = x + constant_op.constant(1)
        else:
          x = x + constant_op.constant(2)
      return x

    @polymorphic_function.function
    def brancher_true():
      # [FUNCTION DOC]
      # Traced function: same one-thousand-step branching sum as the other, but the condition is a
      # constant true tensor built inside, so the branches can be folded away. Takes no input;
      # returns the final sum tensor.
      left = constant_op.constant(True)
      x = constant_op.constant(1)
      for _ in range(1000):
        if left:
          x = x + constant_op.constant(1)
        else:
          x = x + constant_op.constant(2)
      return x

    x = constant_op.constant(True)
    self.assertEqual(brancher(x), brancher_true())  # Trace each function once.

    benchmark = min(timeit.repeat(lambda: brancher(x), repeat=5, number=100))
    opt_benchmark = min(timeit.repeat(brancher_true, repeat=5, number=100))

    # Constant folded execution is usually 15 - 20 times faster. Here we check
    # for a 3x speedup to account for various machines the test might run on.
    self.assertLess(opt_benchmark * 3, benchmark)

  @test_util.run_v2_only
  def test_small_constants_optimization_with_grappler(self):
    # [FUNCTION DOC]
    # Scenario (v2 only): wrap one branching function twice, plainly and with the runtime constant
    # optimization attribute on. Feed a boolean tensor placed on the CPU, check both give equal
    # results, time both (best of five repeats of one hundred calls) and assert the optimized one
    # is more than twice as fast, with the normal graph optimizer enabled.
    def func(inp):
      # [FUNCTION DOC]
      # Undecorated function wrapped by the test: one thousand steps adding one or two to a running
      # sum depending on the boolean input; returns the final sum tensor.
      x = constant_op.constant(1)
      for _ in range(1000):
        if inp:
          x = x + constant_op.constant(1)
        else:
          x = x + constant_op.constant(2)
      return x

    brancher = polymorphic_function.function(func)
    brancher_opt = polymorphic_function.function(
        func, experimental_attributes={'runtime_constant_optimization': True}
    )

    # Trace each function once.
    with ops.device_v2('CPU'):
      x = constant_op.constant(True)
    self.assertEqual(brancher(x), brancher_opt(x))

    benchmark = min(timeit.repeat(lambda: brancher(x), repeat=5, number=100))
    opt_benchmark = min(
        timeit.repeat(lambda: brancher_opt(x), repeat=5, number=100)
    )

    # Constant folded execution is usually 15 - 20 times faster. Here we check
    # for a 2x speedup to account for various machines the test might run on.
    # Specially the kokoro machines seems to run much slower.
    self.assertLess(opt_benchmark * 2, benchmark)

  @test_util.run_v2_only
  @test_util.run_gpu_only
  def test_small_constants_optimization_disabled(self):
    # [FUNCTION DOC]
    # Scenario (v2 only, GPU only): a function with the runtime constant optimization attribute is
    # called with a boolean tensor in default placement (on the GPU). Asserts an invalid-argument
    # error saying the boolean tensor must be on the host when the optimization is enabled.
    @polymorphic_function.function(
        experimental_attributes={'runtime_constant_optimization': True}
    )
    def func(inp):
      # [FUNCTION DOC]
      # Traced identity function: returns its input unchanged.
      return inp

    x = constant_op.constant(True)
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        (
            'Expecting boolean tensor to be on host when'
            ' small_constants_optimizer is enabled.'
        ),
    ):
      func(x)

  @test_util.run_v2_only
  def test_small_constants_optimization_invalid_input(self):
    # [FUNCTION DOC]
    # Scenario (v2 only): call an identity function having the runtime constant optimization with
    # a two-element boolean tensor on the CPU. Asserts no crash occurs and the output equals the
    # input, i.e. non-scalar booleans are simply not optimized.
    @polymorphic_function.function(
        experimental_attributes={'runtime_constant_optimization': True}
    )
    def func(inp):
      # [FUNCTION DOC]
      # Traced identity function: returns its input unchanged.
      return inp

    with ops.device_v2('CPU'):
      x = constant_op.constant([True, True])
    # runtime_constant_optimization should not crash when the tf.function
    # is passed in a boolean tensor having > 1 element.
    self.assertAllEqual(func(x), x)

  @test_util.run_v2_only
  def test_small_constants_optimization_without_grappler(self):
    # [FUNCTION DOC]
    # Scenario (v2 only): like the with-grappler test, but the graph meta optimizer is disabled
    # through the options helper while timing, so only the runtime constant optimization can help.
    # Checks equal results for plain and optimized wrappers, then asserts the optimized one is
    # more than five times faster (best of five repeats of one hundred calls).
    def func(inp):
      # [FUNCTION DOC]
      # Undecorated function wrapped by the test: one thousand steps adding one or two to a running
      # sum depending on the boolean input; returns the final sum tensor.
      x = constant_op.constant(1)
      for _ in range(1000):
        if inp:
          x = x + constant_op.constant(1)
        else:
          x = x + constant_op.constant(2)
      return x

    brancher = polymorphic_function.function(func)
    brancher_opt = polymorphic_function.function(
        func, experimental_attributes={'runtime_constant_optimization': True}
    )

    # Trace each function once.
    with ops.device_v2('CPU'):
      x = constant_op.constant(True)
    self.assertEqual(brancher(x), brancher_opt(x))

    # Disable grappler and check that performance is still good with
    # small_constants_optimizer.
    with options({'disable_meta_optimizer': True}):
      benchmark = min(timeit.repeat(lambda: brancher(x), repeat=5, number=100))
      opt_benchmark = min(
          timeit.repeat(lambda: brancher_opt(x), repeat=5, number=100)
      )

    # Constant folded execution is usually 150x times faster (against a base
    # that has no grappler optimization). Here we check
    # for a 5x speedup to account for various machines the test might run on.
    # Specially the kokoro machines seems to run much slower.
    self.assertLess(opt_benchmark * 5, benchmark)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
