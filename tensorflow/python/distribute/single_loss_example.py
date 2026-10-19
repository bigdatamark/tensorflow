# [MODULE DOC]
# Helper module with small example networks used by distribution strategy tests and examples: a
# single-loss step built on the step function classes, a minimal legacy minimize example, and a
# batch normalization example. Each builder returns the pieces (step or model function, dataset
# function, layer) that tests then run under a strategy. No import-time side effects.
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
"""A simple network to use in tests and examples."""

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import step_fn
from tensorflow.python.distribute import strategy_test_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.layers import core
from tensorflow.python.layers import normalization
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def single_loss_example(optimizer_fn, distribution, use_bias=False,
                        iterations_per_step=1):
  # [FUNCTION DOC]
  # HELPER: builds a tiny one-unit dense network example for tests. Inputs: optimizer_fn (callable
  # returning an optimizer), distribution (distribution strategy), use_bias (default false) and
  # iterations_per_step (default 1). Returns a pair: a standard single-loss step wired to the
  # dataset, loss, optimizer and strategy, and the dense layer (so tests can inspect kernels).
  # The loss is the squared difference between the layer output and one on constant input one.
  """Build a very simple network to use in tests and examples."""

  def dataset_fn():
    # [FUNCTION DOC]
    # HELPER: returns an endlessly repeating dataset of a single one-by-one tensor holding one.
    return dataset_ops.Dataset.from_tensors([[1.]]).repeat()

  optimizer = optimizer_fn()
  layer = core.Dense(1, use_bias=use_bias)

  def loss_fn(ctx, x):
    # [FUNCTION DOC]
    # HELPER: loss for the step. Ignores the multi-step context; applies the layer to x and
    # returns the square of (output minus one) as a scalar.
    del ctx
    y = array_ops.reshape(layer(x), []) - constant_op.constant(1.)
    return y * y

  single_loss_step = step_fn.StandardSingleLossStep(
      dataset_fn, loss_fn, optimizer, distribution, iterations_per_step)

  # Layer is returned for inspecting the kernels in tests.
  return single_loss_step, layer


def minimize_loss_example(optimizer, use_bias=False, use_callable_loss=True):
  # [FUNCTION DOC]
  # HELPER: builds a non-distribution-aware example for tests. Inputs: optimizer (an optimizer
  # instance), use_bias (default false) and use_callable_loss (default true; chooses between
  # passing the loss as a callable or an already computed value for older optimizers). Returns a
  # triple: model function (takes an input and returns the minimize op), dataset function and the
  # dense layer.
  """Example of non-distribution-aware legacy code."""

  def dataset_fn():
    # [FUNCTION DOC]
    # HELPER: returns a repeating dataset of one-by-one tensors holding one, batched with size one
    # dropping the remainder so shapes are fully defined (needed for TPU).
    dataset = dataset_ops.Dataset.from_tensors([[1.]]).repeat()
    # TODO(isaprykin): batch with drop_remainder causes shapes to be
    # fully defined for TPU.  Remove this when XLA supports dynamic shapes.
    return dataset.batch(1, drop_remainder=True)

  layer = core.Dense(1, use_bias=use_bias)

  def model_fn(x):
    # [FUNCTION DOC]
    # HELPER: model function for the example. Input: x, the batch of inputs. Defines a loss as the
    # squared difference between the layer output and one, then minimizes it: for v2-style
    # optimizers passes the callable loss and a callable listing the layer's trainable variables;
    # otherwise passes a callable loss or the evaluated loss depending on the outer flag.
    # Returns the minimize result.
    """A very simple model written by the user."""

    def loss_fn():
      # [FUNCTION DOC]
      # HELPER: loss of the model: square of (layer output on x minus one).
      y = array_ops.reshape(layer(x), []) - constant_op.constant(1.)
      return y * y

    if strategy_test_lib.is_optimizer_v2_instance(optimizer):
      return optimizer.minimize(loss_fn, lambda: layer.trainable_variables)
    elif use_callable_loss:
      return optimizer.minimize(loss_fn)
    else:
      return optimizer.minimize(loss_fn())

  return model_fn, dataset_fn, layer


def batchnorm_example(optimizer_fn,
                      batch_per_epoch=1,
                      momentum=0.9,
                      renorm=False,
                      update_ops_in_replica_mode=False):
  # [FUNCTION DOC]
  # HELPER: builds a batch normalization example for tests. Inputs: optimizer_fn (callable
  # returning an optimizer), batch_per_epoch (default 1), momentum (default 0.9), renorm (default
  # false; enables batch renormalization) and update_ops_in_replica_mode (default false; when
  # true the loss depends on the graph's update ops collection). Creates an unfused batch
  # normalization layer and a one-unit bias-free dense layer. Returns a triple: model function,
  # dataset function and the batch normalization layer.
  """Example of non-distribution-aware legacy code with batch normalization."""

  def dataset_fn():
    # [FUNCTION DOC]
    # HELPER: returns an endlessly repeating dataset of batch_per_epoch batches, each a sixteen by
    # eight float matrix whose values increase along both dimensions (offset by batch index).
    # input shape is [16, 8], input values are increasing in both dimensions.
    return dataset_ops.Dataset.from_tensor_slices(
        [[[float(x * 8 + y + z * 100)
           for y in range(8)]
          for x in range(16)]
         for z in range(batch_per_epoch)]).repeat()

  optimizer = optimizer_fn()
  batchnorm = normalization.BatchNormalization(
      renorm=renorm, momentum=momentum, fused=False)
  layer = core.Dense(1, use_bias=False)

  def model_fn(x):
    # [FUNCTION DOC]
    # HELPER: model function for the batch normalization example. Input: x, a batch. Minimizes the
    # loss using the callable loss form (with a variables callable for v2-style optimizers).
    # Returns the minimize result.
    """A model that uses batchnorm."""

    def loss_fn():
      # [FUNCTION DOC]
      # HELPER: applies batch normalization in training mode to x, then (optionally under control
      # dependencies on the update ops collection) returns the mean of the summed dense output
      # minus one.
      y = batchnorm(x, training=True)
      with ops.control_dependencies(
          ops.get_collection(ops.GraphKeys.UPDATE_OPS)
          if update_ops_in_replica_mode else []):
        loss = math_ops.reduce_mean(
            math_ops.reduce_sum(layer(y)) - constant_op.constant(1.))
      # `x` and `y` will be fetched by the gradient computation, but not `loss`.
      return loss

    if strategy_test_lib.is_optimizer_v2_instance(optimizer):
      return optimizer.minimize(loss_fn, lambda: layer.trainable_variables)

    # Callable loss.
    return optimizer.minimize(loss_fn)

  return model_fn, dataset_fn, batchnorm
