# [MODULE DOC]
# Helper module for Keras Model.compile. Provides container classes that hold, normalize and
# apply user-supplied losses and metrics (LossesContainer, MetricsContainer, with a shared
# Container base), plus small free functions for pseudo output names, mapping dictionaries to
# output order, dtype/rank matching, Keras mask handling and naming custom callables.
# No import-time side effects.
#  Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for `Model.compile`."""

import copy

from tensorflow.python.distribute import distribute_lib
from tensorflow.python.keras import losses as losses_mod
from tensorflow.python.keras import metrics as metrics_mod
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest


class Container(object):
  # [CLASS DOC]
  # Base class for the losses and metrics containers. Stores the model output names and offers
  # shared logic to conform user structures (labels, weights, losses, metrics) to the model's
  # output structure and to broadcast a single loss or metric to all outputs. Subclasses must
  # provide the broadcast test and object copy hooks.
  """Base Container class."""

  def __init__(self, output_names=None):
    # [FUNCTION DOC]
    # Constructor. output_names is an optional list of output name strings; when none, names are
    # created later in build. Stores them. Returns nothing.
    self._output_names = output_names

  def build(self, y_pred):
    # [FUNCTION DOC]
    # If no output names are known, creates pseudo names (such as output_1) from the structure of
    # y_pred, the model predictions. Stores them on the container. Returns nothing.
    if self._output_names is None:
      # In Subclass API, output names like 'output_1' are used for
      # `Metric` names.
      self._output_names = create_pseudo_output_names(y_pred)

  def _conform_to_outputs(self, outputs, struct):
    # [FUNCTION DOC]
    # HELPER: shared by the losses and metrics containers.
    # Reshapes struct (nested labels, weights, losses or metrics) to match the structure of
    # outputs (model predictions): maps a dictionary to an output-ordered list using output names,
    # fills dictionary keys missing from struct with nothing, and repeats a single non-nested item
    # for every output. Returns the conformed structure. May raise a value error for unexpected
    # dictionary keys.
    """Convenience method to conform `struct` to `outputs` structure.

    Mappings performed:

    (1) Map a dict to a list of outputs, using the output names.
    (2) Fill missing keys in a dict w/ `None`s.
    (3) Map a single item to all outputs.

    Args:
      outputs: Model predictions.
      struct: Arbitrary nested structure (e.g. of labels, sample_weights,
        losses, or metrics).

    Returns:
      Mapping of `struct` to `outputs` structure.
    """
    struct = map_to_output_names(outputs, self._output_names, struct)
    struct = map_missing_dict_keys(outputs, struct)
    # Allow passing one object that applies to all outputs.
    if not nest.is_nested(struct) and nest.is_nested(outputs):
      struct = nest.map_structure(lambda _: struct, outputs)
    return struct

  def _maybe_broadcast_to_outputs(self, outputs, objects):
    # [FUNCTION DOC]
    # HELPER: shared by the containers' build step.
    # KNOWN ISSUE: the docstring places its description of what the method does after the Returns
    # section.
    # If the subclass says objects (losses or metrics) should apply to every output, returns a
    # structure shaped like outputs with the object at each position, copying it per output when
    # the model has more than one output; otherwise returns objects unchanged.
    """Determines if losses / metrics should be applied to all outputs.

    NOTE: This method should only be called for Metrics / Losses, not for
    y_true / sample_weight.

    Args:
      outputs: Model predictions.
      objects: Arbitrary nested structure (e.g. of losses or metrics)

    Returns:
      Arbitrary nested structure of objects, maybe copied to each output.

    Applies a Loss / Metric to all outputs.
    """
    if not self._should_broadcast(objects):
      return objects

    # When there is more than one Model output, this is needed to keep
    # each Metric / Loss separate. When there is only one Model output,
    # the user-supplied object should be used.
    should_copy_objects = len(nest.flatten(outputs)) > 1

    def _broadcast_fn():
      # [FUNCTION DOC]
      # HELPER: produces the object placed at each output; returns a per-output copy when the
      # model has several outputs, otherwise the original object. Takes no arguments.
      if should_copy_objects:
        return nest.map_structure(self._copy_object, objects)
      return objects

    return nest.map_structure(lambda _: _broadcast_fn(), outputs)

  def _should_broadcast(self, objects):
    # [FUNCTION DOC]
    # Abstract hook: subclasses decide whether objects should be applied to all outputs. Always
    # raises a not-implemented error here.
    raise NotImplementedError

  def _copy_object(self, obj):
    # [FUNCTION DOC]
    # Abstract hook: subclasses return an independent copy of obj. Always raises a
    # not-implemented error here.
    raise NotImplementedError


class LossesContainer(Container):
  # [CLASS DOC]
  # Holds the losses and loss weights passed to compile, converts them to loss objects at build
  # time, tracks a total-loss mean metric plus per-output loss metrics (multi-output only), and
  # is callable to compute the overall loss for a batch.
  """A container class for losses passed to `Model.compile`."""

  def __init__(self, losses, loss_weights=None, output_names=None):
    # [FUNCTION DOC]
    # Constructor. losses is any structure of loss names, functions or loss objects; loss_weights
    # is an optional matching structure of numbers; output_names is an optional list of strings.
    # Keeps the original values for recompilation and serialization, creates the total loss mean
    # metric named loss, and marks the container not built. Returns nothing.
    super(LossesContainer, self).__init__(output_names=output_names)

    # Keep user-supplied values untouched for recompiling and serialization.
    self._user_losses = losses
    self._user_loss_weights = loss_weights

    self._losses = losses
    self._loss_weights = loss_weights
    self._per_output_metrics = None  # Per-output losses become metrics.
    self._loss_metric = metrics_mod.Mean(name='loss')  # Total loss.
    self._built = False

  @property
  def metrics(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says per-output loss metrics, but the list also starts with the
    # total loss metric.
    # Property. Returns an empty list before build; afterwards the total loss metric followed by
    # the non-empty per-output loss metrics.
    """Per-output loss metrics."""
    if not self._built:
      return []
    per_output_metrics = [
        metric_obj for metric_obj in nest.flatten(self._per_output_metrics)
        if metric_obj is not None
    ]
    return [self._loss_metric] + per_output_metrics

  def build(self, y_pred):
    # [FUNCTION DOC]
    # One-time setup given y_pred, the model predictions. Builds output names, broadcasts and
    # conforms losses and loss weights to the output structure, converts losses to loss objects,
    # flattens both to lists, creates per-output metrics and marks the container built. Returns
    # nothing; invalid losses raise a value error.
    """One-time setup of loss objects."""
    super(LossesContainer, self).build(y_pred)

    self._losses = self._maybe_broadcast_to_outputs(y_pred, self._losses)
    self._losses = self._conform_to_outputs(y_pred, self._losses)
    self._losses = nest.map_structure(self._get_loss_object, self._losses)
    self._losses = nest.flatten(self._losses)

    self._loss_weights = self._maybe_broadcast_to_outputs(
        y_pred, self._loss_weights)
    self._loss_weights = self._conform_to_outputs(y_pred, self._loss_weights)
    self._loss_weights = nest.flatten(self._loss_weights)

    self._create_metrics()
    self._built = True

  @property
  def built(self):
    # [FUNCTION DOC]
    # Property returning whether build has been performed.
    return self._built

  def _create_metrics(self):
    # [FUNCTION DOC]
    # HELPER: used by build.
    # Creates per-output loss mean metrics named after each output with a loss suffix; for a
    # single-output model, or an output without a loss, the entry is nothing. Returns nothing.
    """Creates per-output loss metrics, but only for multi-output Models."""
    if len(self._output_names) == 1:
      self._per_output_metrics = [None]
    else:
      self._per_output_metrics = []
      for loss_obj, output_name in zip(self._losses, self._output_names):
        if loss_obj is None:
          self._per_output_metrics.append(None)
        else:
          self._per_output_metrics.append(
              metrics_mod.Mean(output_name + '_loss'))

  def __call__(self,
               y_true,
               y_pred,
               sample_weight=None,
               regularization_losses=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it returns a tuple of total loss and per-output list, but
    # the code returns only the total loss tensor.
    # Computes the total loss. Inputs: y_true and y_pred (structures of tensors), sample_weight
    # (optional structure) and regularization_losses (optional list of extra loss tensors). Builds
    # on first use. For each output with a label and loss, matches dtype and rank, applies any
    # Keras mask to weights, computes the loss, updates per-output and total loss metrics (with
    # distribution corrections for sum reduction), applies loss weights and scales for
    # distribution. Regularization losses are summed and added. Returns the summed loss, or a zero
    # scalar when there are no losses. Side effects: updates metric state.
    """Computes the overall loss.

    Args:
      y_true: An arbitrary structure of Tensors representing the ground truth.
      y_pred: An arbitrary structure of Tensors representing a Model's outputs.
      sample_weight: An arbitrary structure of Tensors representing the
        per-sample loss weights. If one Tensor is passed, it is used for all
        losses. If multiple Tensors are passed, the structure should match
        `y_pred`.
      regularization_losses: Additional losses to be added to the total loss.

    Returns:
      Tuple of `(total_loss, per_output_loss_list)`
    """
    y_true = self._conform_to_outputs(y_pred, y_true)
    sample_weight = self._conform_to_outputs(y_pred, sample_weight)

    if not self._built:
      self.build(y_pred)

    y_pred = nest.flatten(y_pred)
    y_true = nest.flatten(y_true)
    sample_weight = nest.flatten(sample_weight)

    loss_values = []  # Used for gradient calculation.
    loss_metric_values = []  # Used for loss metric calculation.
    batch_dim = None
    zip_args = (y_true, y_pred, sample_weight, self._losses, self._loss_weights,
                self._per_output_metrics)
    for y_t, y_p, sw, loss_obj, loss_weight, metric_obj in zip(*zip_args):
      if y_t is None or loss_obj is None:  # Ok to have no loss for an output.
        continue

      y_t, y_p, sw = match_dtype_and_rank(y_t, y_p, sw)
      sw = apply_mask(y_p, sw, get_mask(y_p))
      loss_value = loss_obj(y_t, y_p, sample_weight=sw)

      loss_metric_value = loss_value
      # Correct for the `Mean` loss metrics counting each replica as a batch.
      if loss_obj.reduction == losses_utils.ReductionV2.SUM:
        loss_metric_value *= distribute_lib.get_strategy().num_replicas_in_sync

      if batch_dim is None:
        if tf_utils.is_ragged(y_t):
          batch_dim = y_t.nrows()
        else:
          batch_dim = array_ops.shape(y_t)[0]

      if metric_obj is not None:
        metric_obj.update_state(loss_metric_value, sample_weight=batch_dim)

      if loss_weight is not None:
        loss_value *= loss_weight
        loss_metric_value *= loss_weight

      if (loss_obj.reduction == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE or
          loss_obj.reduction == losses_utils.ReductionV2.AUTO):
        loss_value = losses_utils.scale_loss_for_distribution(loss_value)

      loss_values.append(loss_value)
      loss_metric_values.append(loss_metric_value)

    if regularization_losses:
      regularization_losses = losses_utils.cast_losses_to_common_dtype(
          regularization_losses)
      reg_loss = math_ops.add_n(regularization_losses)
      loss_metric_values.append(reg_loss)
      loss_values.append(losses_utils.scale_loss_for_distribution(reg_loss))

    if loss_values:
      loss_metric_values = losses_utils.cast_losses_to_common_dtype(
          loss_metric_values)
      total_loss_metric_value = math_ops.add_n(loss_metric_values)
      self._loss_metric.update_state(
          total_loss_metric_value, sample_weight=batch_dim)

      loss_values = losses_utils.cast_losses_to_common_dtype(loss_values)
      total_loss = math_ops.add_n(loss_values)
      return total_loss
    else:
      # Ok for a model to have no compiled loss.
      return array_ops.zeros(shape=())

  def reset_state(self):
    # [FUNCTION DOC]
    # Resets the total loss metric and all per-output loss metrics. Does nothing before build.
    # Returns nothing.
    """Resets the state of loss metrics."""
    if not self._built:
      return
    metrics = [self._loss_metric] + nest.flatten(self._per_output_metrics)
    for metric_obj in metrics:
      if metric_obj is not None:
        metric_obj.reset_state()

  def _get_loss_object(self, loss):
    # [FUNCTION DOC]
    # HELPER: used by build.
    # Converts loss (nothing, name, function or loss object) into a loss object, wrapping plain
    # callables using their derived name, and flags it so sum-over-batch-size reduction is
    # allowed (this mutates the object). Returns nothing for none. Raises a value error if the
    # item is not callable-like and has no name.
    """Returns a `Loss` object.

    Converts the user-supplied loss to a `Loss` object. Also allows
    `SUM_OVER_BATCH_SIZE` reduction to be used for this loss.

    Args:
      loss: A string, function, or `Loss` object.

    Returns:
      A `Loss` object.
    """
    if loss is None:
      return None  # Ok to have no loss for an output.

    loss = losses_mod.get(loss)
    if not isinstance(loss, losses_mod.Loss):
      loss_name = get_custom_object_name(loss)
      if loss_name is None:
        raise ValueError('Loss should be a callable, found: {}'.format(loss))
      loss = losses_mod.LossFunctionWrapper(loss, name=loss_name)
    loss._allow_sum_over_batch_size = True  # pylint: disable=protected-access
    return loss

  def _should_broadcast(self, obj):
    # [FUNCTION DOC]
    # Returns true when obj is a single non-nested item, meaning it applies to all outputs.
    return not nest.is_nested(obj)

  def _copy_object(self, obj):
    # [FUNCTION DOC]
    # Returns obj itself; losses need no copying.
    return obj  # Losses don't need to be copied.


class MetricsContainer(Container):
  # [CLASS DOC]
  # Holds the metrics and weighted metrics passed to compile, converts them to metric objects at
  # build time, assigns unique names, keeps a backward-compatible flat ordering, and updates and
  # resets them per batch.
  """A container class for metrics passed to `Model.compile`."""

  def __init__(self, metrics=None, weighted_metrics=None, output_names=None,
               from_serialized=False):
    # [FUNCTION DOC]
    # Constructor. metrics and weighted_metrics are optional structures of metric names,
    # functions or objects (unweighted and weighted); output_names is an optional list of
    # strings; from_serialized (default false) skips metric renaming for loaded models. Keeps
    # the originals, marks the container not built. Returns nothing.
    """Initializes a container for metrics.

    Arguments:
      metrics: see the `metrics` argument from `tf.keras.Model.compile`.
      weighted_metrics: see the `weighted_metrics` argument from
        `tf.keras.Model.compile`.
      output_names: A list of strings of names of outputs for the model.
      from_serialized: Whether the model being compiled is from a serialized
        model.  Used to avoid redundantly applying pre-processing renaming
        steps.
    """
    super(MetricsContainer, self).__init__(output_names=output_names)

    # Keep user-supplied values untouched for recompiling and serialization.
    self._user_metrics = metrics
    self._user_weighted_metrics = weighted_metrics

    self._metrics = metrics
    self._weighted_metrics = weighted_metrics
    self._built = False

    self._from_serialized = from_serialized

  @property
  def metrics(self):
    # [FUNCTION DOC]
    # Property. Returns an empty list before build, afterwards the ordered flat list of all
    # metric objects (unweighted and weighted).
    """All metrics in this container."""
    if not self._built:
      return []
    return self._metrics_in_order

  @property
  def unweighted_metrics(self):
    # [FUNCTION DOC]
    # Property. Returns nothing before build, afterwards the flat list of per-output structures
    # of metrics that do not receive sample weights.
    """Metrics in this container that should not be passed `sample_weight`."""
    if not self._built:
      return None
    return nest.flatten(self._metrics)

  @property
  def weighted_metrics(self):
    # [FUNCTION DOC]
    # Property. Returns nothing before build, afterwards the flat list of per-output structures
    # of metrics that receive sample weights.
    """Metrics in this container that should be passed `sample_weight`."""
    if not self._built:
      return None
    return nest.flatten(self._weighted_metrics)

  def build(self, y_pred, y_true):
    # [FUNCTION DOC]
    # One-time setup given y_pred (predictions) and y_true (labels). Broadcasts and conforms
    # metrics and weighted metrics to the output structure, converts lists to tuples, converts
    # each entry to a metric object (using label and prediction shapes to pick accuracy or
    # crossentropy variants), flattens per output, sets unique names unless loaded from a
    # serialized model, builds the flat ordering and marks the container built. Returns nothing.
    """One-time setup of metric objects."""
    super(MetricsContainer, self).build(y_pred)

    self._metrics = self._maybe_broadcast_to_outputs(y_pred, self._metrics)
    self._metrics = self._conform_to_outputs(y_pred, self._metrics)

    self._weighted_metrics = self._maybe_broadcast_to_outputs(
        y_pred, self._weighted_metrics)
    self._weighted_metrics = self._conform_to_outputs(y_pred,
                                                      self._weighted_metrics)

    # Standardize on tuple since `tf.data` turns lists into `Tensor`s.
    y_pred = nest.list_to_tuple(y_pred)
    y_true = nest.list_to_tuple(y_true)
    self._metrics = nest.list_to_tuple(self._metrics)
    self._weighted_metrics = nest.list_to_tuple(self._weighted_metrics)

    # Convert to `Metric` objects, potentially disambiguating based on output
    # properties.
    self._metrics = nest.map_structure_up_to(y_pred, self._get_metric_objects,
                                             self._metrics, y_true, y_pred)
    self._weighted_metrics = nest.map_structure_up_to(y_pred,
                                                      self._get_metric_objects,
                                                      self._weighted_metrics,
                                                      y_true, y_pred)

    self._metrics = nest.flatten_up_to(y_pred, self._metrics, check_types=False)
    self._weighted_metrics = nest.flatten_up_to(
        y_pred, self._weighted_metrics, check_types=False)

    # Assumes metrics, weighted_metrics have been flattened up to outputs.
    #
    # If we are loading a model that has been already serialized, we do not
    # want to re-apply any pre-processing metric renaming steps.
    if not self._from_serialized:
      self._set_metric_names()
    self._create_ordered_metrics()
    self._built = True

  @property
  def built(self):
    # [FUNCTION DOC]
    # Property returning whether build has been performed.
    return self._built

  def _set_metric_names(self):
    # [FUNCTION DOC]
    # HELPER: used by build.
    # Gives each metric a unique name: prefixes the output name for multi-output models, and for
    # weighted metrics adds a weighted prefix when the name would collide. Mutates the metric
    # objects' names. Raises a value error if two metrics still share a name. Returns nothing.
    """Sets unique metric names."""
    # For multi-output models, prepend the output name to the metric name.
    # For weighted metrics, prepend "weighted_" if the name would be non-unique.
    # pylint: disable=protected-access
    metric_names = set()
    is_multi_output = len(self._output_names) > 1
    zip_args = (self._output_names, self._metrics, self._weighted_metrics)
    for output_name, output_metrics, weighted_output_metrics in zip(*zip_args):
      for m in output_metrics:
        if m is None:
          continue
        if is_multi_output:
          m._name = output_name + '_' + m._name
        if m._name in metric_names:
          raise ValueError('Found two metrics with the same name: {}'.format(
              m._name))
        metric_names.add(m._name)

      for wm in weighted_output_metrics:
        if wm is None:
          continue
        if is_multi_output:
          if output_name + '_' + wm._name in metric_names:
            wm._name = output_name + '_weighted_' + wm._name
          else:
            wm._name = output_name + '_' + wm._name
        elif wm._name in metric_names:
          wm._name = 'weighted_' + wm._name

        if wm._name in metric_names:
          raise ValueError('Found two metrics with the same name: {}'.format(
              wm._name))
        metric_names.add(wm._name)
    # pylint: enable=protected-access

  def _create_ordered_metrics(self):
    # [FUNCTION DOC]
    # HELPER: used by build.
    # Builds the flat list, per output unweighted metrics then weighted metrics, skipping
    # empty entries, stored for the metrics property. Returns nothing.
    """Cache the flat order needed when returning metrics, for backwards compat."""
    self._metrics_in_order = []
    for output_metrics, output_weighted_metrics in zip(self._metrics,
                                                       self._weighted_metrics):
      for m in nest.flatten(output_metrics):
        if m is not None:
          self._metrics_in_order.append(m)
      for wm in nest.flatten(output_weighted_metrics):
        if wm is not None:
          self._metrics_in_order.append(wm)

  def update_state(self, y_true, y_pred, sample_weight=None):
    # [FUNCTION DOC]
    # Updates all metrics with a batch. y_true and y_pred are structures of tensors,
    # sample_weight is an optional structure. Builds on first use. Per output, skips outputs with
    # no label or no metrics, matches dtype and rank, derives the mask from the prediction, and
    # updates unweighted metrics using the mask as weights and weighted metrics using the
    # sample weights combined with the mask. Returns nothing; mutates metric state.
    """Updates the state of per-output metrics."""
    y_true = self._conform_to_outputs(y_pred, y_true)
    sample_weight = self._conform_to_outputs(y_pred, sample_weight)

    if not self._built:
      self.build(y_pred, y_true)

    y_pred = nest.flatten(y_pred)
    y_true = nest.flatten(y_true) if y_true is not None else []
    sample_weight = nest.flatten(sample_weight)

    zip_args = (y_true, y_pred, sample_weight, self._metrics,
                self._weighted_metrics)
    for y_t, y_p, sw, metric_objs, weighted_metric_objs in zip(*zip_args):
      # Ok to have no metrics for an output.
      if (y_t is None or (all(m is None for m in metric_objs) and
                          all(wm is None for wm in weighted_metric_objs))):
        continue

      y_t, y_p, sw = match_dtype_and_rank(y_t, y_p, sw)
      mask = get_mask(y_p)
      sw = apply_mask(y_p, sw, mask)

      for metric_obj in metric_objs:
        if metric_obj is None:
          continue
        metric_obj.update_state(y_t, y_p, sample_weight=mask)

      for weighted_metric_obj in weighted_metric_objs:
        if weighted_metric_obj is None:
          continue
        weighted_metric_obj.update_state(y_t, y_p, sample_weight=sw)

  def reset_state(self):
    # [FUNCTION DOC]
    # Resets every metric object. Before build, resets only those metric objects the user passed
    # directly (strings and functions are ignored). Returns nothing.
    """Resets the state of all `Metric`s in this container."""
    if self._built:
      metrics = self._metrics_in_order
    else:
      # If the user supplied `Metric` objects directly, we should
      # reset those. This could also contain `str`s or `function`s
      # though.
      metrics = nest.flatten(self._user_metrics) + nest.flatten(
          self._user_weighted_metrics)

    for metric_obj in metrics:
      if isinstance(metric_obj, metrics_mod.Metric):
        metric_obj.reset_state()

  def _get_metric_objects(self, metrics, y_t, y_p):
    # [FUNCTION DOC]
    # HELPER: used by build.
    # Flattens the given metrics structure and converts each entry to a metric object using a
    # sample label y_t and output y_p. Returns the list of metric objects.
    """Convert user-supplied metrics to `Metric` objects."""
    metrics = nest.flatten(metrics)
    return [self._get_metric_object(m, y_t, y_p) for m in metrics]

  def _get_metric_object(self, metric, y_t, y_p):
    # [FUNCTION DOC]
    # HELPER: used by build.
    # Converts one user metric (nothing, string, function or metric object) to a metric object.
    # For the generic names accuracy, acc, crossentropy and ce it inspects the ranks and last
    # dimensions of y_t and y_p to choose binary, sparse categorical or categorical variants.
    # Loss objects are flagged to allow sum-over-batch-size. Plain callables are wrapped in a
    # mean-metric wrapper named after the string or the callable. Returns nothing for none.
    # Raises a value error for an unnamed non-callable.
    """Converts user-supplied metric to a `Metric` object.

    Args:
      metric: A string, function, or `Metric` object.
      y_t: Sample of label.
      y_p: Sample of output.

    Returns:
      A `Metric` object.
    """
    if metric is None:
      return None  # Ok to have no metric for an output.

    # Convenience feature for selecting b/t binary, categorical,
    # and sparse categorical.
    if str(metric).lower() not in ['accuracy', 'acc', 'crossentropy', 'ce']:
      metric_obj = metrics_mod.get(metric)
    else:
      y_t_rank = len(y_t.shape.as_list())
      y_p_rank = len(y_p.shape.as_list())
      y_t_last_dim = y_t.shape.as_list()[-1]
      y_p_last_dim = y_p.shape.as_list()[-1]

      is_binary = y_p_last_dim == 1
      is_sparse_categorical = (
          y_t_rank < y_p_rank or y_t_last_dim == 1 and y_p_last_dim > 1)

      if str(metric).lower() in ['accuracy', 'acc']:
        if is_binary:
          metric_obj = metrics_mod.binary_accuracy
        elif is_sparse_categorical:
          metric_obj = metrics_mod.sparse_categorical_accuracy
        else:
          metric_obj = metrics_mod.categorical_accuracy
      else:
        if is_binary:
          metric_obj = metrics_mod.binary_crossentropy
        elif is_sparse_categorical:
          metric_obj = metrics_mod.sparse_categorical_crossentropy
        else:
          metric_obj = metrics_mod.categorical_crossentropy

    if isinstance(metric_obj, losses_mod.Loss):
      metric_obj._allow_sum_over_batch_size = True  # pylint: disable=protected-access

    if not isinstance(metric_obj, metrics_mod.Metric):
      if isinstance(metric, str):
        metric_name = metric
      else:
        metric_name = get_custom_object_name(metric)
        if metric_name is None:
          raise ValueError(
              'Metric should be a callable, found: {}'.format(metric))

      metric_obj = metrics_mod.MeanMetricWrapper(metric_obj, name=metric_name)

    return metric_obj

  def _should_broadcast(self, obj):
    # [FUNCTION DOC]
    # Returns true for a single non-nested metric, or a flat list or tuple of metrics, meaning it
    # applies to every output.
    # e.g. 'mse'.
    if not nest.is_nested(obj):
      return True
    # e.g. ['mse'] or ['mse', 'mae'].
    return (isinstance(obj, (list, tuple)) and
            not any(nest.is_nested(o) for o in obj))

  def _copy_object(self, obj):
    # [FUNCTION DOC]
    # Returns a fresh metric rebuilt from the configuration of a metric object; any other value
    # (function, string or nothing) is returned as is.
    if isinstance(obj, metrics_mod.Metric):
      return obj.__class__.from_config(obj.get_config())
    return obj  # Can be a function or `None`.


def create_pseudo_output_names(outputs):
  # [FUNCTION DOC]
  # Returns a list of default output names (prefixed output_) for a subclassed model's outputs.
  """Create pseudo output names for a subclassed Model."""
  return _create_pseudo_names(outputs, prefix='output_')


def create_pseudo_input_names(inputs):
  # [FUNCTION DOC]
  # Returns a list of default input names (prefixed input_) for a subclassed model's inputs.
  """Create pseudo input names for a subclassed Model."""
  return _create_pseudo_names(inputs, prefix='input_')


def _create_pseudo_names(tensors, prefix):
  # [FUNCTION DOC]
  # HELPER: shared by the pseudo input and output name functions.
  # Walks the flat paths of the tensors structure and joins path parts with underscores (integer
  # indices shifted to start at one). Lists get the prefix prepended; a single tensor gets the
  # prefix plus one; dictionaries use their keys. Returns the list of names.
  """Creates pseudo {input | output} names for subclassed Models.

  Warning: this function should only be used to define default
  names for `Metrics` and `SavedModel`. No other use cases should
  rely on a `Model`'s input or output names.

  Example with dict:

  `{'a': [x1, x2], 'b': x3}` becomes:
  `['a_1', 'a_2', 'b']`

  Example with list:

  `[x, y]` becomes:
  `['output_1', 'output_2']`

  Args:
    tensors: `Model`'s outputs or inputs.
    prefix: 'output_' for outputs, 'input_' for inputs.

  Returns:
    Flattened list of pseudo names.
  """

  def one_index(ele):
    # [FUNCTION DOC]
    # HELPER: adds one to integer path elements so names start at one; returns other elements
    # unchanged.
    # Start with "output_1" instead of "output_0".
    if isinstance(ele, int):
      return ele + 1
    return ele

  flat_paths = list(nest.yield_flat_paths(tensors))
  flat_paths = nest.map_structure(one_index, flat_paths)
  names = []
  for path in flat_paths:
    if not path:
      name = prefix + '1'  # Single output.
    else:
      name = '_'.join(str(p) for p in path)
      if isinstance(path[0], int):
        name = prefix + name
    names.append(name)
  return names


def map_to_output_names(y_pred, output_names, struct):
  # [FUNCTION DOC]
  # Converts a dictionary struct to a list ordered by output_names when the predictions y_pred
  # are a single output or a flat list; otherwise returns struct unchanged. Missing names yield
  # nothing; a single-element result is unwrapped. Falls back to pseudo names when output_names
  # is empty. Does not mutate the input (works on a copy). Raises a value error if the dictionary
  # has keys not matching any output name.
  """Maps a dict to a list using `output_names` as keys.

  This is a convenience feature only. When a `Model`'s outputs
  are a list, you can specify per-output losses and metrics as
  a dict, where the keys are the output names. If you specify
  per-output losses and metrics via the same structure as the
  `Model`'s outputs (recommended), no mapping is performed.

  For the Functional API, the output names are the names of the
  last layer of each output. For the Subclass API, the output names
  are determined by `create_pseudo_output_names` (For example:
  `['output_1', 'output_2']` for a list of outputs).

  This mapping preserves backwards compatibility for `compile` and
  `fit`.

  Args:
    y_pred: Sample outputs of the Model, to determine if this convenience
      feature should be applied (`struct` is returned unmodified if `y_pred`
      isn't a flat list).
    output_names: List. The names of the outputs of the Model.
    struct: The structure to map.

  Returns:
    `struct` mapped to a list in same order as `output_names`.
  """
  single_output = not nest.is_nested(y_pred)
  outputs_are_flat_list = (not single_output and
                           isinstance(y_pred, (list, tuple)) and
                           not any(nest.is_nested(y_p) for y_p in y_pred))

  if (single_output or outputs_are_flat_list) and isinstance(struct, dict):
    output_names = output_names or create_pseudo_output_names(y_pred)
    struct = copy.copy(struct)
    new_struct = [struct.pop(name, None) for name in output_names]
    if struct:
      raise ValueError('Found unexpected keys that do not correspond '
                       'to any Model output: {}. Expected: {}'.format(
                           struct.keys(), output_names))
    if len(new_struct) == 1:
      return new_struct[0]
    return new_struct
  else:
    return struct


def map_missing_dict_keys(y_pred, struct):
  # [FUNCTION DOC]
  # When both y_pred and struct are dictionaries, adds each prediction key missing from struct
  # with a nothing value; this mutates struct in place. Returns struct (unchanged for other
  # types).
  """Replaces missing dict keys in `struct` with `None` placeholders."""
  if not isinstance(y_pred, dict) or not isinstance(struct, dict):
    return struct
  for k in y_pred.keys():
    if k not in struct:
      struct[k] = None
  return struct


def match_dtype_and_rank(y_t, y_p, sw):
  # [FUNCTION DOC]
  # Aligns labels y_t and optional sample weights sw with predictions y_p: a rank-one label or
  # weight against rank-two predictions gets a trailing dimension; labels are cast to the
  # prediction dtype when both are floating or both integer; weights are always cast to the
  # prediction dtype. Returns the tuple of adjusted labels, predictions and weights.
  """Match dtype and rank of predictions."""
  if y_t.shape.rank == 1 and y_p.shape.rank == 2:
    y_t = array_ops.expand_dims_v2(y_t, axis=-1)
  if sw is not None:
    if sw.shape.rank == 1 and y_p.shape.rank == 2:
      sw = array_ops.expand_dims_v2(sw, axis=-1)

  # Dtype.
  # This is required mainly for custom loss functions which do not take care
  # casting dtypes.
  if ((y_t.dtype.is_floating and y_p.dtype.is_floating) or
      (y_t.dtype.is_integer and y_p.dtype.is_integer)):
    y_t = math_ops.cast(y_t, y_p.dtype)

  if sw is not None:
    sw = math_ops.cast(sw, y_p.dtype)
  return y_t, y_p, sw


def get_mask(y_p):
  # [FUNCTION DOC]
  # Returns the Keras mask attached to tensor y_p, or nothing if absent.
  """Returns Keras mask from tensor."""
  return getattr(y_p, '_keras_mask', None)


def apply_mask(y_p, sw, mask):
  # [FUNCTION DOC]
  # Combines a mask with sample weights. With no mask, returns sw unchanged. Otherwise casts the
  # mask to the dtype of predictions y_p; if sw exists, aligns dimensions and multiplies them
  # (sw is multiplied in place when possible); if sw is nothing, the mask becomes the weights.
  # Returns the resulting weights.
  """Applies any mask on predictions to sample weights."""
  if mask is not None:
    mask = math_ops.cast(mask, y_p.dtype)
    if sw is not None:
      mask, _, sw = (
          losses_utils.squeeze_or_expand_dimensions(mask, sample_weight=sw))
      sw *= mask
    else:
      sw = mask
  return sw


def get_custom_object_name(obj):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the final fallback branch is unreachable in practice because every Python
  # object has a class attribute.
  # Returns a name for a custom loss or metric: its name attribute if present, else its function
  # name, else the snake-case form of its class name; otherwise nothing.
  """Returns the name to use for a custom loss or metric callable.

  Args:
    obj: Custom loss of metric callable

  Returns:
    Name to use, or `None` if the object was not recognized.
  """
  if hasattr(obj, 'name'):  # Accept `Loss` instance as `Metric`.
    return obj.name
  elif hasattr(obj, '__name__'):  # Function.
    return obj.__name__
  elif hasattr(obj, '__class__'):  # Class instance.
    return generic_utils.to_snake_case(obj.__class__.__name__)
  else:  # Unrecognized object.
    return None
