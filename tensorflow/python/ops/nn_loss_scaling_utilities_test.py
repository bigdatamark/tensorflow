# [MODULE DOC]
# Unit tests for the loss scaling helpers in the distributed neural-network implementation module
# (compute_average_loss and scale_regularization_loss). The tests cover explicit and default global
# batch sizes, sample weights of several shapes, empty batches, dtype preservation, invalid
# arguments, and the error raised when the helpers are called in a cross-replica context.
# Distributed cases run under a mirrored strategy with two CPU devices in graph and eager modes.
# Nothing is registered at import time; the module runs the test main when executed directly.
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
"""Tests for loss scaling utilities in tensorflow.ops.nn."""

from absl.testing import parameterized

from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_impl_distribute
from tensorflow.python.platform import test as test_lib


class LossUtilitiesTest(test_lib.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering the loss scaling utilities: average loss over a global batch size and
  # regularization loss scaling by the number of replicas. It has no shared fixtures; the
  # distributed tests receive a mirrored two-CPU strategy through the combinations decorator.

  def testComputeAverageLossGlobalBatchSize(self):
    # [FUNCTION DOC]
    # Checks that the summed per-example losses (five values totalling fifteen) are divided by an
    # explicit global batch size of ten, giving an average of exactly one and a half.
    # Takes no inputs, returns nothing, has no side effects.
    per_example_loss = [1, 2, 3, 4, 5]
    loss = nn_impl_distribute.compute_average_loss(
        per_example_loss, global_batch_size=10
    )
    self.assertEqual(self.evaluate(loss), 1.5)

  def testComputeAverageLossGlobalBatchSize_BatchSizeNonScalar(self):
    # [FUNCTION DOC]
    # Passes a list-valued (non-scalar) global batch size and asserts a value error whose message
    # says the global batch size must be scalar. Takes no inputs, returns nothing.
    per_example_loss = [1, 2, 3, 4, 5]
    with self.assertRaisesWithPredicateMatch(
        ValueError, "global_batch_size must be scalar"):
      nn_impl_distribute.compute_average_loss(
          per_example_loss, global_batch_size=[10]
      )

  def testComputeAverageLossGlobalBatchSize_BatchSizeFloat(self):
    # [FUNCTION DOC]
    # Passes a floating-point global batch size and asserts a type error whose message says the
    # global batch size must be an integer. Takes no inputs, returns nothing.
    per_example_loss = [1, 2, 3, 4, 5]
    with self.assertRaisesWithPredicateMatch(
        TypeError, "global_batch_size must be an int"):
      nn_impl_distribute.compute_average_loss(
          per_example_loss, global_batch_size=10.0
      )

  def testComputeAverageLossGlobalBatchSize_BatchSizeNegative(self):
    # [FUNCTION DOC]
    # Passes a negative global batch size and asserts an invalid-argument error whose message says
    # the global batch size must be non-negative. Takes no inputs, returns nothing.
    per_example_loss = [1, 2, 3, 4, 5]
    with self.assertRaisesWithPredicateMatch(
        errors_impl.InvalidArgumentError,
        "global_batch_size must be non-negative"):
      nn_impl_distribute.compute_average_loss(
          per_example_loss, global_batch_size=-1
      )

  def testComputeAverageLossGlobalBatchSize_BatchSizeZero(self):
    # [FUNCTION DOC]
    # Passes a global batch size of zero with non-empty losses and asserts the result evaluates to
    # exactly zero rather than raising a division error. Takes no inputs, returns nothing.
    per_example_loss = [1, 2, 3, 4, 5]
    loss = nn_impl_distribute.compute_average_loss(
        per_example_loss, global_batch_size=0
    )
    self.assertEqual(self.evaluate(loss), 0.0)

  @combinations.generate(
      combinations.combine(
          distribution=[strategy_combinations.mirrored_strategy_with_two_cpus],
          mode=["graph", "eager"],
      )
  )
  def testComputeAverageLossDefaultGlobalBatchSize(self, distribution):
    # [FUNCTION DOC]
    # Verifies the default global batch size. Without a strategy (one replica) the average of three
    # losses is their mean. Under the given two-replica strategy, each replica computes its own
    # average, and the summed reduction across replicas must equal the same mean.
    # Input: distribution, the strategy supplied by the combinations decorator. Returns nothing.
    # Without strategy - num replicas = 1
    per_example_loss = constant_op.constant([2.5, 6.2, 5.])
    loss = nn_impl_distribute.compute_average_loss(per_example_loss)
    self.assertAllClose(self.evaluate(loss), (2.5 + 6.2 + 5.) / 3)

    # With strategy - num replicas = 2
    with distribution.scope():
      per_replica_losses = distribution.run(
          nn_impl_distribute.compute_average_loss, args=(per_example_loss,)
      )
      loss = distribution.reduce("SUM", per_replica_losses, axis=None)
      self.assertAllClose(self.evaluate(loss), (2.5 + 6.2 + 5.) / 3)

  @combinations.generate(
      combinations.combine(
          distribution=[strategy_combinations.mirrored_strategy_with_two_cpus],
          mode=["graph", "eager"],
      )
  )
  def testComputeAverageLossDefaultGlobalBatchSizeEmptyBatch(self,
                                                             distribution):
    # [FUNCTION DOC]
    # Verifies that an empty per-example loss tensor yields zero (not a not-a-number value), both
    # without a strategy and when run per replica under the strategy and summed across replicas.
    # Input: distribution, the strategy supplied by the combinations decorator. Returns nothing.
    per_example_loss = constant_op.constant([], dtypes.float32)
    loss = nn_impl_distribute.compute_average_loss(per_example_loss)
    self.assertEqual(self.evaluate(loss), 0.0)

    with distribution.scope():
      per_replica_losses = distribution.run(
          nn_impl_distribute.compute_average_loss, args=(per_example_loss,)
      )
      loss = distribution.reduce("SUM", per_replica_losses, axis=None)
      self.assertAllClose(self.evaluate(loss), 0.0)

  @combinations.generate(
      combinations.combine(
          distribution=[strategy_combinations.mirrored_strategy_with_two_cpus],
          mode=["graph", "eager"],
      )
  )
  def testComputeAverageLossSampleWeights(self, distribution):
    # [FUNCTION DOC]
    # Under the strategy, checks weighted averaging for three sample weight shapes: a scalar
    # weight, one weight per example, and a per-time-step weight matrix. After summing the
    # per-replica results, each is compared with the hand-computed weighted sum divided by the
    # batch size. Input: distribution, the strategy from the decorator. Returns nothing.
    with distribution.scope():
      # Scalar sample weight
      per_replica_losses = distribution.run(
          nn_impl_distribute.compute_average_loss,
          args=([2.0, 4.0, 6.0],),
          kwargs={"sample_weight": 2},
      )
      loss = distribution.reduce("SUM", per_replica_losses, axis=None)
      self.assertAllClose(self.evaluate(loss), (2. + 4. + 6.) * 2. / 3)

      # Per example sample weight
      per_replica_losses = distribution.run(
          nn_impl_distribute.compute_average_loss,
          args=([2.0, 4.0, 6.0],),
          kwargs={"sample_weight": [0.3, 0.5, 0.2]},
      )
      loss = distribution.reduce("SUM", per_replica_losses, axis=None)
      self.assertAllClose(
          self.evaluate(loss), (2. * 0.3 + 4. * 0.5 + 6. * 0.2) / 3)

      # Time-step sample weight
      per_replica_losses = distribution.run(
          nn_impl_distribute.compute_average_loss,
          args=([[2.0, 0.5], [4.0, 1.0]],),
          kwargs={"sample_weight": [[0.3, 0.7], [0.2, 0.8]]},
      )
      loss = distribution.reduce("SUM", per_replica_losses, axis=None)
      self.assertAllClose(
          self.evaluate(loss), (2. * 0.3 + 0.5 * 0.7 + 4. * 0.2 + 1. * 0.8) / 2)

  @combinations.generate(
      combinations.combine(
          distribution=[strategy_combinations.mirrored_strategy_with_two_cpus],
          mode=["graph", "eager"],
      )
  )
  def testComputeAverageLossSampleWeightsEmptyBatch(self, distribution):
    # [FUNCTION DOC]
    # Under the strategy, runs the average loss on an empty loss tensor with a scalar weight and
    # with an empty per-example weight tensor, asserting the reduced result is zero in both cases.
    # Input: distribution, the strategy from the decorator. Returns nothing.
    empty_rank0 = constant_op.constant([], dtypes.float32)

    with distribution.scope():
      # Scalar sample weight
      per_replica_losses = distribution.run(
          nn_impl_distribute.compute_average_loss,
          args=(empty_rank0,),
          kwargs={"sample_weight": 2},
      )
      loss = distribution.reduce("SUM", per_replica_losses, axis=None)
      self.assertAllClose(self.evaluate(loss), 0.0)

      # Per example sample weight
      per_replica_losses = distribution.run(
          nn_impl_distribute.compute_average_loss,
          args=(empty_rank0,),
          kwargs={"sample_weight": empty_rank0},
      )
      loss = distribution.reduce("SUM", per_replica_losses, axis=None)
      self.assertAllClose(
          self.evaluate(loss), 0.0)

  def testComputeAverageLossInvalidSampleWeights(self):
    # [FUNCTION DOC]
    # Passes three losses with two sample weights (mismatched shapes) and an explicit global batch
    # size, and asserts an incompatible-shapes failure, accepted as either a value error (static
    # check) or an invalid-argument error (runtime check). Takes no inputs, returns nothing.
    with self.assertRaisesIncompatibleShapesError(
        (ValueError, errors_impl.InvalidArgumentError)):
      nn_impl_distribute.compute_average_loss(
          [2.5, 6.2, 5.0], sample_weight=[0.2, 0.8], global_batch_size=10
      )

  @combinations.generate(
      combinations.combine(
          distribution=[strategy_combinations.mirrored_strategy_with_two_cpus],
          mode=["graph", "eager"],
      )
  )
  def testComputeAverageLossDtype(self, distribution):
    # [FUNCTION DOC]
    # Feeds double-precision losses with a scalar integer weight through the strategy and asserts
    # the reduced loss keeps the double-precision type. Input: distribution, the strategy from the
    # decorator. Returns nothing.
    with distribution.scope():
      per_example_loss = constant_op.constant([2., 4., 6.],
                                              dtype=dtypes.float64)
      per_replica_losses = distribution.run(
          nn_impl_distribute.compute_average_loss,
          args=(per_example_loss,),
          kwargs={"sample_weight": 2},
      )
      loss = distribution.reduce("SUM", per_replica_losses, axis=None)
      self.assertEqual(loss.dtype, dtypes.float64)

  def testComputeAverageLossInvalidRank(self):
    # [FUNCTION DOC]
    # Passes a scalar (rank zero) loss and asserts a value error stating at least rank one is
    # expected. Then, in graph mode, uses a placeholder of unknown rank, builds the loss, and
    # asserts running it with a scalar feed raises an invalid-argument error with the same message
    # (the dynamic rank check). Takes no inputs, returns nothing; opens a cached session.
    per_example_loss = constant_op.constant(2.)

    # Static rank
    with self.assertRaisesRegex(
        ValueError, "Invalid value passed for `per_example_loss`. "
        "Expected a tensor with at least rank 1."):
      nn_impl_distribute.compute_average_loss(per_example_loss)

    with context.graph_mode():
      # Dynamic rank
      per_example_loss = array_ops.placeholder(dtype=dtypes.float32)
      loss = nn_impl_distribute.compute_average_loss(per_example_loss)

      with self.cached_session() as sess:
        with self.assertRaisesRegex(
            errors.InvalidArgumentError,
            "Invalid value passed for `per_example_loss`. "
            "Expected a tensor with at least rank 1."):
          sess.run(loss, {per_example_loss: 2})

  @combinations.generate(
      combinations.combine(
          distribution=[strategy_combinations.mirrored_strategy_with_two_cpus],
          mode=["graph", "eager"],
      )
  )
  def testComputeAverageLossInCrossReplicaContext(self, distribution):
    # [FUNCTION DOC]
    # Calls the average loss directly inside the strategy scope (cross-replica context) and asserts
    # a runtime error saying it must not be called there. Input: distribution, the strategy from
    # the decorator. Returns nothing.
    with distribution.scope():
      with self.assertRaisesRegex(
          RuntimeError,
          "You are calling `compute_average_loss` in cross replica context"):
        nn_impl_distribute.compute_average_loss([2, 3])

  @combinations.generate(
      combinations.combine(
          distribution=[strategy_combinations.mirrored_strategy_with_two_cpus],
          mode=["graph", "eager"],
      )
  )
  def testScaleRegularizationLoss(self, distribution):
    # [FUNCTION DOC]
    # Checks regularization loss scaling: without a strategy the summed regularization losses are
    # returned unchanged; under the two-replica strategy each replica scales by the replica count,
    # so the summed reduction equals the same total. Input: distribution, the strategy from the
    # decorator. Returns nothing.
    # Without strategy - num replicas = 1
    reg_losses = constant_op.constant([2.5, 6.2, 5.])
    loss = nn_impl_distribute.scale_regularization_loss(reg_losses)
    self.assertAllClose(self.evaluate(loss), (2.5 + 6.2 + 5.))

    # With strategy - num replicas = 2
    with distribution.scope():
      per_replica_losses = distribution.run(
          nn_impl_distribute.scale_regularization_loss, args=(reg_losses,)
      )
      loss = distribution.reduce("SUM", per_replica_losses, axis=None)
      self.assertAllClose(self.evaluate(loss), (2.5 + 6.2 + 5.))

  @combinations.generate(
      combinations.combine(
          distribution=[strategy_combinations.mirrored_strategy_with_two_cpus],
          mode=["graph", "eager"],
      )
  )
  def testScaleRegularizationLossInCrossReplicaContext(self, distribution):
    # [FUNCTION DOC]
    # Calls the regularization loss scaling function inside the strategy scope (cross-replica
    # context) and asserts a runtime error saying it cannot be called there. Input: distribution,
    # the strategy from the decorator. Returns nothing.
    with distribution.scope():
      with self.assertRaisesRegex(
          RuntimeError, "You are calling `scale_regularization_loss` in "
          "cross replica context"):
        nn_impl_distribute.scale_regularization_loss([2, 3])


if __name__ == "__main__":
  test_lib.main()
