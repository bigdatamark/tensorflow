# [MODULE DOC]
# Tests that distribution strategies handle batch normalization when the input batch is empty
# (zero examples) or has a dynamic batch size. Uses the one-device strategy in graph and eager
# modes with fused and unfused normalization. Running the file as a script starts the
# distributed test runner. No import-time side effects.
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
"""Test DistributionStrategy in the zero batch case."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import test_util
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.layers import normalization
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.losses import losses
from tensorflow.python.platform import test
from tensorflow.python.training import gradient_descent


class NormalizationTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for batch normalization layers under a distribution strategy with empty or
  # variable-sized batches; each test is parameterized over fused/unfused normalization.

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.one_device_strategy,
          ],
          mode=["graph"],
          fused=[True, False]))
  def testBNWithZeroBatchInputGraph(self, distribution, fused):
    # [FUNCTION DOC]
    # Test (graph mode): with the option to get the next element as optional enabled, builds
    # train and predict steps for a batch normalization layer with zero-sized inputs and runs
    # training for one hundred iterations. Expects the initial and final moving mean (zeros),
    # moving variance, scale (ones) and offset (zeros) to be unchanged, the training loss to be
    # zero each time, and the inference output to be empty.
    distribution.extended.experimental_enable_get_next_as_optional = True
    with distribution.scope(), self.cached_session() as sess:
      bn_list = []
      inputs = np.random.random((0, 4, 4, 3)) + 100
      targets = np.random.random((0, 4, 4, 3))
      inputs_placeholder = array_ops.placeholder(
          dtype=dtypes.float32, shape=[None, 4, 4, 3])
      targets_placeholder = array_ops.placeholder(
          dtype=dtypes.float32, shape=[None, 4, 4, 3])

      def step_fn(is_training, inputs, targets=None):
        # [FUNCTION DOC]
        # HELPER: per-replica step. Builds a batch normalization layer (recorded in the enclosing
        # list), applies it to the inputs; for inference returns the outputs, otherwise computes
        # mean squared error against the targets, minimizes it with gradient descent and returns
        # the loss after the training op.
        bn = normalization.BatchNormalization(
            axis=3, epsilon=1e-3, momentum=0.9, fused=fused)
        bn_list.append(bn)
        outputs = bn.apply(inputs, training=is_training)
        if not is_training:
          return outputs

        loss = losses.mean_squared_error(targets, outputs)
        optimizer = gradient_descent.GradientDescentOptimizer(0.01)
        train_op = optimizer.minimize(loss)
        with ops.control_dependencies([train_op]):
          return array_ops.identity(loss)

      train_op = distribution.extended.call_for_each_replica(
          step_fn, args=(True, inputs_placeholder, targets_placeholder))
      predict_op = distribution.extended.call_for_each_replica(
          step_fn, args=(False, inputs_placeholder))
      bn = bn_list[0]

      self.evaluate(variables.global_variables_initializer())

      # Check for initial statistics and weights.
      moving_mean, moving_var = self.evaluate(
          [bn.moving_mean, bn.moving_variance])
      self.assertAllEqual([0, 0, 0], moving_mean)
      self.assertAllEqual([1, 1, 1], moving_var)

      np_gamma, np_beta = self.evaluate([bn.gamma, bn.beta])
      self.assertAllEqual([1, 1, 1], np_gamma)
      self.assertAllEqual([0, 0, 0], np_beta)

      for _ in range(100):
        np_output, _, _ = sess.run([train_op] + bn.updates, {
            inputs_placeholder: inputs,
            targets_placeholder: targets
        })
        self.assertEqual(0.0, np_output)

      # Verify that the statistics and weights are not changed after training.
      moving_mean, moving_var = self.evaluate(
          [bn.moving_mean, bn.moving_variance])
      self.assertAllEqual([0, 0, 0], moving_mean)
      self.assertAllEqual([1, 1, 1], moving_var)

      np_gamma, np_beta = self.evaluate([bn.gamma, bn.beta])
      self.assertAllEqual([1, 1, 1], np_gamma)
      self.assertAllEqual([0, 0, 0], np_beta)

      # Test inference.
      np_output = sess.run(predict_op, {inputs_placeholder: inputs})
      self.assertEqual([], np_output.tolist())

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.one_device_strategy,
          ],
          mode=["eager"],
          fused=[True, False]))
  def testBNWithZeroBatchInput(self, distribution, fused):
    # [FUNCTION DOC]
    # Test (eager mode): same scenario as the graph version using compiled train and test steps
    # with a gradient tape and optimizer. Expects the loss to be zero for one hundred training
    # iterations, the layer statistics and weights to remain unchanged, and the inference output
    # to be an empty array of the right shape.
    distribution.extended.experimental_enable_get_next_as_optional = True
    with distribution.scope():
      inputs = np.random.random((0, 4, 4, 3)).astype(np.float32) + 100
      targets = np.random.random((0, 4, 4, 3)).astype(np.float32)
      bn = normalization.BatchNormalization(
          axis=3, epsilon=1e-3, momentum=0.9, fused=fused)
      optimizer = gradient_descent.GradientDescentOptimizer(0.01)

      @def_function.function
      def train_step():
        # [FUNCTION DOC]
        # HELPER: compiled training step that runs the inner step through the strategy.
        # Returns the per-replica loss.
        def step_fn(inputs, targets):
          # [FUNCTION DOC]
          # HELPER: computes the loss of the layer in training mode, takes gradients and applies
          # them with the optimizer; returns the loss.
          with backprop.GradientTape() as tape:
            outputs = bn.apply(inputs, training=True)
            loss = losses.mean_squared_error(targets, outputs)
          grads = tape.gradient(loss, bn.variables)
          optimizer.apply_gradients(zip(grads, bn.variables))
          return loss

        return distribution.run(step_fn, args=(inputs, targets))

      for _ in range(100):
        np_output = train_step().numpy()
        self.assertEqual(0.0, np_output)

      # Verify that the statistics and weights are not changed after training.
      self.assertAllEqual([0, 0, 0], bn.moving_mean.numpy())
      self.assertAllEqual([1, 1, 1], bn.moving_variance.numpy())
      self.assertAllEqual([1, 1, 1], bn.gamma.numpy())
      self.assertAllEqual([0, 0, 0], bn.beta.numpy())

      @def_function.function
      def test_step():
        # [FUNCTION DOC]
        # HELPER: compiled inference step that runs the inner step through the strategy and
        # returns the per-replica outputs.
        def step_fn(inputs):
          # [FUNCTION DOC]
          # HELPER: applies the layer to the inputs in inference mode and returns the outputs.
          outputs = bn.apply(inputs, training=False)
          return outputs

        return distribution.run(step_fn, args=(inputs,))

      # Test inference.
      self.assertAllEqual(np.zeros(shape=(0, 4, 4, 3), dtype=np.float32),
                          test_step().numpy())

  @combinations.generate(
      combinations.combine(
          distribution=[
              strategy_combinations.one_device_strategy,
          ],
          mode=["eager"],
          fused=[True, False]))
  def testBNWithDynamicBatchInputEager(self, distribution, fused):
    # [FUNCTION DOC]
    # Test (eager mode): distributes a dataset of eleven examples batched by ten without dropping
    # the remainder (so the batch size varies), and trains a layer for one hundred iterations.
    # Expects the moving mean, variance, scale and offset to differ from their initial values.
    # KNOWN ISSUE: the comparison builds arrays by shape from the initial-value lists rather than
    # arrays holding those values, so the not-equal assertions are weak.
    distribution.extended.experimental_enable_get_next_as_optional = True
    with distribution.scope():
      # Explicitly create dataset with drop_remainder=False.
      # This would make batch size unknown.
      inputs = np.random.random((11, 4, 4, 3)).astype(np.float32) + 100
      targets = np.random.random((11, 4, 4, 3)).astype(np.float32)
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets)).batch(
          10, drop_remainder=False).repeat()
      dataset_iterator = iter(
          distribution.experimental_distribute_dataset(dataset))

      bn = normalization.BatchNormalization(
          axis=-1, epsilon=1e-3, momentum=0.9, fused=fused)
      optimizer = gradient_descent.GradientDescentOptimizer(0.01)

      @def_function.function
      def train_step(iterator):
      # [FUNCTION DOC]
      # HELPER: compiled training step taking an iterator; runs the inner step on the next
      # element through the strategy.

        def step_fn(inputs):
          # [FUNCTION DOC]
          # HELPER: splits the input into features and targets, computes the training loss with a
          # gradient tape, applies the gradients with the optimizer and returns the loss.
          features, targets = inputs
          with backprop.GradientTape() as tape:
            outputs = bn(features, training=True)
            loss = losses.mean_squared_error(targets, outputs)

          grads = tape.gradient(loss, bn.variables)
          optimizer.apply_gradients(zip(grads, bn.variables))
          return loss

        return distribution.run(step_fn, args=(next(iterator),))

      for _ in range(100):
        train_step(dataset_iterator).numpy()

      # Verify that the statistics and weights are updated.
      self.assertNotAllEqual(np.ndarray([0, 0, 0]), bn.moving_mean.numpy())
      self.assertNotAllEqual(np.ndarray([1, 1, 1]), bn.moving_variance.numpy())
      self.assertNotAllEqual(np.ndarray([1, 1, 1]), bn.gamma.numpy())
      self.assertNotAllEqual(np.ndarray([0, 0, 0]), bn.beta.numpy())


if __name__ == "__main__":
  test_util.main()
