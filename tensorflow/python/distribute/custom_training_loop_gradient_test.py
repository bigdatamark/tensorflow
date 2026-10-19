# [MODULE DOC]
# Tests that gradient tapes work inside custom training loops run with distribution strategies.
# Defines a dataset helper, a mixin with a flattening assertion, and the GradientTapeTest class,
# which exercises gradients computed in functions run per replica across all strategies in eager
# mode. No import-time side effects beyond imports.

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
"""Tests for custom training loops."""

from absl.testing import parameterized

from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables


def get_dataset_from_tensor_slices(inp_array):
  # [FUNCTION DOC]
  # HELPER: dataset construction helper shared by the tests in this module.
  # Takes an array-like input and returns a dataset with one element per slice along the first
  # dimension. Uses the V2 dataset class, but falls back to the V1 dataset class when TF2
  # behavior is not enabled (workaround for a tracked bug).
  dataset = dataset_ops.DatasetV2.from_tensor_slices(inp_array)
  # TODO(b/138326910): Remove Dataset V1 version once bug resolved.
  if not tf2.enabled():
    dataset = dataset_ops.Dataset.from_tensor_slices(inp_array)
  return dataset


class AssertFlattenedMixin(object):
  # [CLASS DOC]
  # HELPER: mixin for test cases that supplies a flattening comparison assertion for per-replica
  # results. Expects to be combined with a test case class providing the assertion methods.
  """Mixin for specialized asserts."""

  def assert_equal_flattened(self, expected_results, actual_results):
    # [FUNCTION DOC]
    # HELPER: assertion helper for the distributed gradient tests.
    # Inputs: expected_results, a list of flat lists of numbers; actual_results, a list whose items
    # are sequences of per-replica tensors. Asserts equal lengths, then for each item concatenates
    # the numeric contents of the per-replica values in order and asserts equality with the
    # expected list. Returns nothing; fails the test on mismatch.
    """Asserts that flattened results are equal.

    Due to the number of replicas in the strategy, the output may have a
    different structure and needs to be flattened for comparison.

    Args:
      expected_results: The results expected as a result of a computation.
      actual_results: The actual results of a computation.
    """
    self.assertEqual(len(expected_results), len(actual_results))

    for i, expected_result in enumerate(expected_results):
      final_result = []
      actual_result = actual_results[i]
      for val in actual_result:
        final_result.extend(val.numpy())
      self.assertAllEqual(expected_result, final_result)


class GradientTapeTest(test.TestCase, parameterized.TestCase,
                       AssertFlattenedMixin):
  # [CLASS DOC]
  # Tests that gradient tapes behave correctly when used inside custom training loops that run
  # under each distribution strategy, in eager mode. Uses the flattening assertion mixin.

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def testStepInFunctionGradient(self, distribution):
    # [FUNCTION DOC]
    # For every strategy, runs a compiled train step per replica over a distributed dataset of
    # four values in batches of two. The step computes the gradient of the square of its input
    # with a tape. Asserts the flattened gradients equal twice the inputs: 10, 12, 14, 16.
    dataset = get_dataset_from_tensor_slices([5., 6., 7., 8.]).batch(2)

    @def_function.function
    def train_step(x):
      # [FUNCTION DOC]
      # Per-replica step: watches the input, computes its square via an inner function under a
      # gradient tape, and returns the gradient of the square with respect to the input.
      def computation(x):
        # [FUNCTION DOC]
        # Returns the elementwise square of the input tensor.
        return math_ops.square(x)
      with backprop.GradientTape() as tape:
        tape.watch(x)  # Manually watch non-variable tensors.
        y = computation(x)
      grads = tape.gradient(y, x)
      return grads

    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    results = []
    for x in dist_dataset:
      output = distribution.experimental_local_results(
          distribution.run(train_step, args=(x,)))
      results.append(output)
    self.assert_equal_flattened([[10., 12.], [14., 16.]], results)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"]
      ))
  def testRunInFunctionGradient(self, distribution):
    # [FUNCTION DOC]
    # Like the step-in-function test, but the strategy's run call itself is inside the compiled
    # function. For every strategy, asserts the flattened per-replica gradients of the square
    # equal 10, 12, 14, 16 across the two batches.
    dataset = get_dataset_from_tensor_slices([5., 6., 7., 8.]).batch(2)

    @def_function.function
    def run(x):
      # [FUNCTION DOC]
      # Compiled wrapper: runs the per-replica train step on the strategy with the given input
      # and returns the list of local per-replica results.
      def train_step(x):
        # [FUNCTION DOC]
        # Per-replica step: watches the input, squares it via computation under a gradient tape
        # and returns the gradient with respect to the input.
        def computation(x):
          # [FUNCTION DOC]
          # Returns the elementwise square of the input tensor.
          return math_ops.square(x)
        with backprop.GradientTape() as tape:
          tape.watch(x)  # Manually watch non-variable tensors.
          y = computation(x)
        grads = tape.gradient(y, x)
        return grads
      return distribution.experimental_local_results(
          distribution.run(train_step, args=(x,)))
    dist_dataset = distribution.experimental_distribute_dataset(dataset)
    results = []
    for x in dist_dataset:
      output = run(x)
      results.append(output)
    self.assert_equal_flattened([[10., 12.], [14., 16.]], results)

  @combinations.generate(
      combinations.combine(
          distribution=strategy_combinations.all_strategies,
          mode=["eager"],
          model_in_tf_function=[True, False]
      ))
  def testNestedFunction(self, distribution, model_in_tf_function):
    # [FUNCTION DOC]
    # For every strategy and for the model either plain or wrapped as a compiled function,
    # computes the gradient of a squared variable inside a replica step nested in a compiled
    # train step. Asserts there is one gradient per replica and none is null.
    def model(x):
      # [FUNCTION DOC]
      # Simple model: returns the input multiplied by itself.
      return x * x

    if model_in_tf_function:
      model = def_function.function(model)

    with distribution.scope():
      x = variables.Variable(1.0)

      @def_function.function
      def train_step():
        # [FUNCTION DOC]
        # Compiled step: runs the replica step on the strategy and returns the per-replica
        # gradients.
        def replica_step():
          # [FUNCTION DOC]
          # Computes the model output on the captured variable under a gradient tape and returns
          # the gradient of the output with respect to the variable.
          with backprop.GradientTape() as tape:
            y = model(x)
          return tape.gradient(y, x)
        return distribution.run(replica_step)

      grads = distribution.experimental_local_results(train_step())
      self.assertLen(grads, distribution.num_replicas_in_sync)
      self.assertTrue(all(g is not None for g in grads))


if __name__ == "__main__":
  test.main()
