# [MODULE DOC]
# HELPER: internal helper module of the legacy (v1) Keras training engine for eager execution.
# Provides per-batch routines used when a model runs eagerly: computing the (masked,
# sample-weighted, regularized, distribution-scaled) loss for a batch, computing metric results,
# running a gradient step (with loss scaling support), and the public-to-engine train_on_batch
# and test_on_batch entry points returning dictionaries of losses and metrics.
# No import-time side effects beyond imports.
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
"""Keras training and evaluation routines for eager execution."""
# pylint: disable=protected-access

import numpy as np

from tensorflow.python.eager.backprop import GradientTape
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


def _eager_loss_fn(outputs, targets, loss_fn, output_name):
  # [FUNCTION DOC]
  # HELPER: single-output loss helper for the eager routines.
  # KNOWN ISSUE: appears unused in this module (loss computation is done inside _model_loss).
  # Computes one output's loss. Takes model outputs, targets, a loss callable and the
  # output name; calls the loss callable (targets first, then outputs) inside a name scope
  # derived from the output name and returns the loss value.
  with backend.name_scope(output_name + '_loss'):
    loss = loss_fn(targets, outputs)
  return loss


def _eager_metrics_fn(model, outputs, targets, sample_weights=None, masks=None):
  # [FUNCTION DOC]
  # HELPER: metric computation for train_on_batch and test_on_batch.
  # Flattens outputs and targets. If targets exist, pads the target list with nothing-values
  # where the model skipped targets, then asks the model to handle metrics (weighted and
  # unweighted, with skip-target masks) which updates metric state. Finally appends results of
  # metrics added separately (not part of compiled metric functions). Optional sample_weights
  # and masks are per-output lists. Returns a list of metric result values.
  """Calculates the metrics for each output of the given model.

  Args:
      model: The model on which metrics are being calculated.
      outputs: The outputs of the given model.
      targets: The predictions or targets of the given model.
      sample_weights: Optional list of sample weights for each output.
      masks: Optional list of masks for each output.

  Returns:
      Returns the metric results for each output of the model.
  """
  outputs = nest.flatten(outputs)
  targets = nest.flatten(targets)
  # Invoke all(weighted and unweighted) metrics.
  metric_results = []
  if targets:
    # Insert None values corresponding to the targets that need to be skipped
    # on the model.
    if len(model._targets) != len(targets):
      new_targets = [
          None if t is None else targets.pop(0) for t in model._targets
      ]
      targets = new_targets

    metric_results = model._handle_metrics(
        outputs,
        targets=targets,
        sample_weights=sample_weights,
        masks=masks,
        return_weighted_and_unweighted_metrics=True,
        skip_target_masks=model._prepare_skip_target_masks())

  # Add metric results from the `add_metric` metrics.
  metric_results.extend([
      m.result()
      for m in model.metrics
      if m not in model._compile_metric_functions
  ])
  return metric_results


def _model_loss(model,
                inputs,
                targets,
                output_loss_metrics=None,
                sample_weights=None,
                training=False):
  # [FUNCTION DOC]
  # HELPER: loss computation helper for the eager batch routines.
  # KNOWN ISSUE: output_loss_metrics defaults to nothing, but it is indexed for every output
  # when the model has more than one output, so omitting it then fails.
  # Core loss computation for the eager batch routines.
  # Runs the model on inputs (a single input is unwrapped; numpy values and numbers are
  # converted to tensors; the training flag is passed only if the model expects it), casts
  # targets and sample weights to matching float types, and gathers output masks. For each
  # loss function it combines mask and sample weights, computes the per-output loss (reduced
  # according to the loss reduction, with automatic reduction treated as sum over batch size;
  # custom losses are assumed to reduce themselves), records stateful per-output loss values
  # when there are several outputs, scales for distribution, and adds the weighted loss to the
  # total. Regularization losses are added at the end. Raises a value error if no loss exists
  # (message differs for training versus evaluation). Returns the model outputs list, the total
  # loss, the list of per-output loss metric values and the masks.
  """Calculates the loss for a given model.

  Args:
      model: The model on which metrics are being calculated.
      inputs: Either a dictionary of inputs to the model or a list of input
        arrays.
      targets: List of target arrays.
      output_loss_metrics: List of metrics that are used to aggregated output
        loss values.
      sample_weights: Optional list of sample weight arrays.
      training: Whether the model should be run in inference or training mode.

  Returns:
     Returns the model output, total loss, loss value calculated using the
     specified loss function and masks for each output. The total loss includes
     regularization losses and applies masking and sample weighting
     to the loss value.
  """
  # TODO(psv): Dedup code here with graph mode prepare_total_loss() fn.
  # Used to keep track of the total loss value (stateless).
  # eg., total_loss = loss_weight_1 * output_1_loss_fn(...) +
  #                   loss_weight_2 * output_2_loss_fn(...) +
  #                   layer losses.
  total_loss = 0
  kwargs = {}
  if model._expects_training_arg:
    kwargs['training'] = training
  if len(inputs) == 1 and not isinstance(inputs, dict):
    inputs = inputs[0]

  # Allow mixed `NumPy` and `EagerTensor` input here.
  if any(
      isinstance(input_t, (np.ndarray, float, int))
      for input_t in nest.flatten(inputs)):
    inputs = nest.map_structure(
        tensor_conversion.convert_to_tensor_v2_with_dispatch, inputs
    )

  outs = model(inputs, **kwargs)
  outs = nest.flatten(outs)

  if targets:
    targets = training_utils_v1.cast_if_floating_dtype_and_mismatch(
        targets, outs)
  # TODO(sallymatson/psv): check if we should do same mismatch fix for weights
  if sample_weights:
    new_sample_weights = []
    for val in sample_weights:
      if val is not None:
        new_sample_weights.append(training_utils_v1.cast_if_floating_dtype(
            tensor_conversion.convert_to_tensor_v2_with_dispatch(val)))
      else:
        new_sample_weights.append(None)
    sample_weights = new_sample_weights

  masks = [getattr(t, '_keras_mask', None) for t in outs]
  targets = nest.flatten(targets)

  # Used to keep track of individual output losses.
  output_losses = []

  with backend.name_scope('loss'):
    loss_fns = [
        loss_fn for loss_fn in model.loss_functions if loss_fn is not None
    ]
    custom_losses = model.losses  # Regularization losses

    if not loss_fns and not custom_losses:
      if training:
        raise ValueError('The model cannot be trained '
                         'because it has no loss to optimize.')
      else:
        raise ValueError('The model cannot be evaluated '
                         'because it has no loss to compute.')

    for i, loss_fn in enumerate(loss_fns):
      weights = sample_weights[i] if sample_weights else None
      mask = masks[i]
      with backend.name_scope(model.output_names[i] + '_loss'):
        if mask is not None:
          mask = math_ops.cast(mask, outs[i].dtype)
          # Update weights with mask.
          if weights is None:
            weights = mask
          else:
            # Update dimensions of weights to match with mask if possible.
            weights = math_ops.cast(weights, outs[i].dtype)
            mask, _, weights = (
                losses_utils.squeeze_or_expand_dimensions(
                    mask, sample_weight=weights))
            weights *= mask

        if hasattr(loss_fn, 'reduction'):
          per_sample_losses = loss_fn.call(targets[i], outs[i])
          weighted_losses = losses_utils.compute_weighted_loss(
              per_sample_losses,
              sample_weight=weights,
              reduction=losses_utils.ReductionV2.NONE)
          loss_reduction = loss_fn.reduction

          # `AUTO` loss reduction defaults to `SUM_OVER_BATCH_SIZE` for all
          # compile use cases.
          if loss_reduction == losses_utils.ReductionV2.AUTO:
            loss_reduction = losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE

          # Compute the stateless loss value.
          output_loss = losses_utils.reduce_weighted_loss(
              weighted_losses, reduction=loss_reduction)
        else:
          # Compute the stateless loss value for a custom loss class.
          # Here we assume that the class takes care of loss reduction
          # because if this class returns a vector value we cannot
          # differentiate between use case where a custom optimizer
          # expects a vector loss value vs unreduced per-sample loss value.
          output_loss = loss_fn(targets[i], outs[i], sample_weight=weights)
          loss_reduction = losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE

      # If the number of outputs is 1 then we don't append the loss metric
      # associated with each model output. When there are multiple outputs
      # associated with a model, each output's loss is calculated and returned
      # as part of the loss_metrics.
      if len(model.outputs) > 1:
        # Keep track of the stateful output loss result.
        output_losses.append(output_loss_metrics[i](output_loss))

      # Scale output loss for distribution. For custom losses we assume
      # reduction was mean.
      if loss_reduction == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE:
        output_loss = losses_utils.scale_loss_for_distribution(output_loss)
      total_loss += model._loss_weights_list[i] * output_loss

    # Add regularization losses
    if custom_losses:
      total_loss += losses_utils.scale_loss_for_distribution(
          math_ops.add_n(custom_losses))
  return outs, total_loss, output_losses, masks


def _process_single_batch(model,
                          inputs,
                          targets,
                          output_loss_metrics=None,
                          sample_weights=None,
                          training=False):
  # [FUNCTION DOC]
  # HELPER: single-batch loss and optional gradient step for train_on_batch.
  # Sets the eager learning phase (training or inference) and respects the model's compiled
  # trainable state, computes the loss under a gradient tape (scaling it when the optimizer is
  # a loss-scale optimizer). When training and trainable weights exist, applies gradients:
  # uses the model's own backward hook when present, otherwise computes gradients (unscaled for
  # loss-scale optimizers) and applies them with the optimizer, mutating the weights. With no
  # trainable weights logs a warning. Raises a value error (from the loss routine) when there
  # is no loss. Returns the model outputs, total loss, per-output losses and masks.
  """Calculate the loss and gradient for one input batch.

     The model weights are updated if training is set to True.

  Args:
      model: Model whose loss has to be calculated.
      inputs: List of input arrays.
      targets: List of target arrays.
      output_loss_metrics: List of metrics that are used to aggregated output
        loss values.
      sample_weights: Optional list of sample weight arrays.
      training: The boolean represents if the weights of the model are updated.
              'fit' methods will set this to True while 'evaluate' methods will
              set this to False.

  Returns:
      output of the model, total loss, the loss and the mask
      associated with each output.

  Raises:
      ValueError: If the model has no loss to optimize.
  """
  with backend.eager_learning_phase_scope(1 if training else 0), \
      training_utils.RespectCompiledTrainableState(model):
    with GradientTape() as tape:
      outs, total_loss, output_losses, masks = (
          _model_loss(
              model,
              inputs,
              targets,
              output_loss_metrics=output_loss_metrics,
              sample_weights=sample_weights,
              training=training))
      if isinstance(model.optimizer, loss_scale_optimizer.LossScaleOptimizer):
        scaled_total_loss = model.optimizer.get_scaled_loss(total_loss)
      else:
        scaled_total_loss = total_loss
    if training:
      trainable_weights = model.trainable_weights
      if trainable_weights:
        # TODO(tanzheny) b/132690565: Provide mechanism for user to override
        # model.train_on_batch.
        if hasattr(model, '_backwards'):
          model._backwards(tape, scaled_total_loss)
        else:
          grads = tape.gradient(scaled_total_loss, trainable_weights)
          if isinstance(model.optimizer,
                        loss_scale_optimizer.LossScaleOptimizer):
            grads = model.optimizer.get_unscaled_gradients(grads)
          model.optimizer.apply_gradients(zip(grads, trainable_weights))
      else:
        logging.warning('The list of trainable weights is empty. Make sure that'
                        ' you are not setting model.trainable to False before '
                        'compiling the model.')
    return outs, total_loss, output_losses, masks


def train_on_batch(model,
                   inputs,
                   targets,
                   sample_weights=None,
                   output_loss_metrics=None):
  # [FUNCTION DOC]
  # Runs one training step on a batch. Casts inputs to the model input dtypes, processes the
  # batch in training mode (updating weights), wraps single outputs in a list, computes metric
  # results and flattens the total loss. Takes model, inputs, targets, optional sample weights
  # and optional per-output loss metrics. Returns a mapping with the total loss list, the
  # per-output losses and the metric results. Raises a value error if the model has no loss.
  """Calculates the loss and gradient updates for one input batch.

  Args:
      model: Model whose loss has to be calculated.
      inputs: Input batch data.
      targets: Target batch data.
      sample_weights: Sample weight batch data.
      output_loss_metrics: List of metrics that are used to aggregated output
        loss values.

  Returns:
      Dict with three items:
        'total_loss': list with a single tensor for overall loss,
        'output_losses': list of tensors for loss corresponding to each of the
          model output. Could be a empty list when model has only one output.
        'metrics': list of tensors for metric specified.
  """
  inputs = training_utils_v1.cast_to_model_input_dtypes(inputs, model)
  outs, total_loss, output_losses, masks = (
      _process_single_batch(
          model,
          inputs,
          targets,
          sample_weights=sample_weights,
          training=True,
          output_loss_metrics=output_loss_metrics))
  if not isinstance(outs, list):
    outs = [outs]
  metrics_results = _eager_metrics_fn(
      model, outs, targets, sample_weights=sample_weights, masks=masks)
  total_loss = nest.flatten(total_loss)
  return {'total_loss': total_loss,
          'output_losses': output_losses,
          'metrics': metrics_results}


def test_on_batch(model,
                  inputs,
                  targets,
                  sample_weights=None,
                  output_loss_metrics=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says total_loss is a single tensor, but the code flattens it
  # and returns a list of tensors, as train_on_batch does.
  # Runs one evaluation step on a batch with no weight updates. Casts inputs to the model
  # input dtypes, computes the loss in inference mode (learning phase zero), wraps single
  # outputs in a list, computes metric results (which updates metric state) and flattens the
  # total loss. Takes model, inputs, targets, optional sample weights and optional per-output
  # loss metrics. Returns a mapping with the total loss list, per-output losses and metric
  # results. Raises a value error if the model has no loss to compute.
  """Calculates the loss for one input batch.

  Args:
      model: Model whose loss has to be calculated.
      inputs: Input batch data.
      targets: Target batch data.
      sample_weights: Sample weight batch data.
      output_loss_metrics: List of metrics that are used to aggregated output
        loss values.

  Returns:
      Dict with three items:
        'total_loss': single tensor for overall loss,
        'output_losses': list of tensors for loss corresponding to each of the
          model output. Could be a empty list when model has only one output.
        'metrics': list of tensors for metric specified.
  """
  inputs = training_utils_v1.cast_to_model_input_dtypes(inputs, model)

  with backend.eager_learning_phase_scope(0):
    outs, total_loss, output_losses, masks = (
        _model_loss(
            model,
            inputs,
            targets,
            sample_weights=sample_weights,
            training=False,
            output_loss_metrics=output_loss_metrics))
  if not isinstance(outs, list):
    outs = [outs]
  metrics_results = _eager_metrics_fn(
      model, outs, targets, sample_weights=sample_weights, masks=masks)
  total_loss = nest.flatten(total_loss)

  return {'total_loss': total_loss,
          'output_losses': output_losses,
          'metrics': metrics_results}
