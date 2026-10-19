# [MODULE DOC]
# HELPER: internal helper module of the legacy (v1) Keras training engine for distribution
# strategies. Provides graph-mode custom loops for TPU strategies (fit, test and predict),
# step-function construction for running replicas, and training-loop classes that dispatch
# fit, evaluate and predict for a single worker or multiple workers (via the distribute
# coordinator). Non-TPU or eager cases fall through to the array-based loops in the sibling
# training_arrays_v1 module. Relies on the distributed utility helpers for cloning models,
# iterators and weight copying. No import-time side effects beyond imports.
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
"""Part of the Keras training engine related to distributed training."""
# pylint: disable=protected-access

import numpy as np
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import reduce_util as ds_reduce_util
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend
from tensorflow.python.keras import callbacks as cbks
from tensorflow.python.keras.distribute import distribute_coordinator_utils as dc
from tensorflow.python.keras.distribute import distributed_training_utils_v1 as dist_utils
from tensorflow.python.keras.engine import partial_batch_padding_handler as padding_util
from tensorflow.python.keras.engine import training_arrays_v1
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import tf_logging as logging


def _per_replica_execution_function(model, mode):
  # [FUNCTION DOC]
  # HELPER: executed inside each replica by the TPU loops. Builds the model's execution
  # function for the mode and returns a four-item tuple of its input tensors, output tensors,
  # update operation and session keyword arguments. Takes the (per-replica) model and mode.
  exec_func = model._make_execution_function(mode)
  return (exec_func.inputs, exec_func.outputs, exec_func.updates_op,
          exec_func.session_kwargs)


def _build_model(strategy, model, mode, inputs, targets=None):
  # [FUNCTION DOC]
  # HELPER: creates the distributed copy of the model for the TPU loops.
  # If the model was compiled with distribution (cloning), clones the model onto the replicas
  # using the given inputs and optional targets; otherwise builds the distributed network.
  # Takes strategy, model, mode, inputs, optional targets. Returns nothing; stores the
  # distributed model on the original model as a side effect.
  if model._compile_distribution:
    dist_utils.clone_model_on_replicas(
        model, strategy, mode, inputs=inputs, targets=targets)
  else:
    dist_utils._build_distributed_network(model, strategy, mode, inputs,
                                          targets)


def _make_train_step_fn(model, mode, strategy, output_labels):
  # [FUNCTION DOC]
  # HELPER: factory used by experimental_tpu_fit_loop. Takes the model, mode, strategy and the
  # list of output labels and returns a step function that the strategy runs repeatedly on an
  # iterator. No side effects until the returned function is invoked.
  """Create step fn.

  Args:
    model: a Keras Model instance.
    mode: One of ModeKeys.TRAIN/ModeKeys.TEST/ModeKeys.PREDICT.
    strategy: a `tf.distribute.Strategy` instance.
    output_labels: the output labels for the step function.

  Returns:
    A step function to run by `tf.distribute.Strategy`.
  """

  def _step_fn(ctx, inputs):
    # [FUNCTION DOC]
    # HELPER: per-step function returned by _make_train_step_fn. Takes the step context and one
    # batch (either a pair of inputs and targets, or just inputs). Dictionary inputs are
    # reordered into a list following the model's feed input names. Builds the distributed
    # model, calls the per-replica execution function on every replica, unwraps and combines
    # the results into one backend function, registers each output as a last-step output with
    # a sum reduction for loss and mean for everything else, and returns the update operation.
    # Feed dictionaries, session arguments, run options and metadata are ignored.
    """A step fn that returns update ops."""
    if isinstance(inputs, (tuple, list)) and len(inputs) == 2:
      inputs, targets = inputs
    else:
      targets = None

    # When input feature is a dictionary of tensors, dictionary is flattended
    # to an array and passed as a model input. This results in input mismatch
    # when model input layer names are not sorted in alphabetical order as
    # `nest.flatten()`sorts dictionary elements by keys. As so, transform input
    # tensors into an array and order it along `model._feed_input_names`.
    if isinstance(inputs, dict):
      inputs = [inputs[input_name] for input_name in model._feed_input_names]

    _build_model(strategy, model, mode, inputs, targets)

    (grouped_inputs, grouped_outputs, grouped_updates,
     grouped_session_args) = strategy.extended.call_for_each_replica(
         _per_replica_execution_function,
         args=(dist_utils.get_distributed_model(model, mode), mode))
    (all_inputs, all_outputs, all_updates,
     all_session_args) = dist_utils.unwrap_values(strategy, grouped_inputs,
                                                  grouped_outputs,
                                                  grouped_updates,
                                                  grouped_session_args)
    combined_fn = backend.function(
        all_inputs,
        all_outputs,
        updates=all_updates,
        name='distributed_' + str(mode) + '_function',
        **all_session_args)

    for label, output in zip(output_labels, combined_fn.outputs):
      if label == 'loss':
        reduce_op = ds_reduce_util.ReduceOp.SUM
      else:
        # We reduce all other metrics using mean for now. This is temporary
        # workaround until new metrics are in place.
        reduce_op = ds_reduce_util.ReduceOp.MEAN
      ctx.set_last_step_output(label, output, reduce_op)

    # TODO(priyag, sourabhbajaj): Ignoring these things from the combined_fn:
    # feed_dict, session kwargs, run options, run_metadata for now. These should
    # be handled appropriately
    return combined_fn.updates_op

  return _step_fn


def experimental_tpu_fit_loop(model,
                              dataset,
                              epochs=100,
                              verbose=1,
                              callbacks=None,
                              initial_epoch=0,
                              steps_per_epoch=None,
                              val_dataset=None,
                              validation_steps=None,
                              validation_freq=1):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it returns nothing, but the function returns the model
  # training history object.
  # KNOWN ISSUE: the out-of-data warning formats the message with the steps value and then
  # multiplies the resulting text by the epoch count instead of multiplying the numbers, so
  # the message is wrong (and repeated). Also steps_per_epoch defaults to nothing but is used
  # arithmetically, so omitting it fails; the strategy scope is not exited if an error occurs.
  # Graph-mode custom training loop for TPU strategies. Takes the model, a dataset yielding
  # inputs and targets, epochs, verbosity, callbacks, initial epoch (may be overridden by a
  # checkpoint), steps per epoch, an optional validation dataset, validation steps and
  # validation frequency. Creates a steps-per-run variable, an iterator and a strategy scope,
  # builds a multi-step run on the iterator, then per epoch runs chunks of steps (adjusting the
  # variable for the remainder), fires callbacks, optionally copies weights back and runs the
  # test loop for validation, and stops early on stop-training or when data runs out.
  # Returns the history; side effects: session runs, weight copies, logging, callbacks.
  """Fit loop for training with TPU tf.distribute.Strategy.

  Args:
      model: Keras Model instance.
      dataset: Dataset that returns inputs and targets
      epochs: Number of times to iterate over the data
      verbose: Integer, Verbosity mode, 0, 1 or 2
      callbacks: List of callbacks to be called during training
      initial_epoch: Epoch at which to start training
          (useful for resuming a previous training run)
      steps_per_epoch: Total number of steps (batches of samples)
          before declaring one epoch finished and starting the
          next epoch. Ignored with the default value of `None`.
      val_dataset: Dataset for validation data.
      validation_steps: Number of steps to run validation for
          (only if doing validation from data tensors).
          Ignored with the default value of `None`.
      validation_freq: Only relevant if validation data is provided. Integer or
          `collections.abc.Container` instance (e.g. list, tuple, etc.). If an
          integer, specifies how many training epochs to run before a new
          validation run is performed, e.g. `validation_freq=2` runs
          validation every 2 epochs. If a Container, specifies the epochs on
          which to run validation, e.g. `validation_freq=[1, 2, 10]` runs
          validation at the end of the 1st, 2nd, and 10th epochs.

  Returns:
      Returns `None`.

  Raises:
      ValueError: in case of invalid arguments.
  """
  mode = ModeKeys.TRAIN

  current_strategy = model._distribution_strategy
  iteration_value = min(steps_per_epoch,
                        current_strategy.extended.steps_per_run)
  steps_per_run = backend.variable(
      value=iteration_value,
      dtype='int32',
      name='steps_per_run')

  # TODO(fchollet): add support for `steps_per_epoch=None` in TPU loops.
  iterator = dist_utils.get_iterator(dataset, current_strategy)

  scope = dist_utils.distributed_scope(
      strategy=current_strategy, learning_phase=1)
  scope.__enter__()

  out_labels = model.metrics_names or []

  step_fn = _make_train_step_fn(model, ModeKeys.TRAIN, current_strategy,
                                out_labels)

  # Add initial dummy values for loss and other metric tensors.
  initial_loop_values = {}
  initial_loop_values['loss'] = constant_op.constant(1e7)
  for m in model._get_training_eval_metrics():
    tensor = m.result()
    initial_loop_values[m.name] = array_ops.zeros(tensor.shape, tensor.dtype)

  ctx = current_strategy.extended.experimental_run_steps_on_iterator(
      step_fn, iterator, iterations=steps_per_run,
      initial_loop_values=initial_loop_values)
  train_op = ctx.run_op
  output_tensors = ctx.last_step_outputs

  do_validation = bool(validation_steps)

  if model._compile_distribution:
    dist_utils._copy_weights_to_distributed_model(model, mode)

  callbacks = cbks.configure_callbacks(
      callbacks,
      model,
      do_validation=do_validation,
      epochs=epochs,
      steps_per_epoch=steps_per_epoch,
      verbose=verbose,
      count_mode='steps',
      mode=mode)

  # Calculate the steps each time on the device.
  steps_to_run = ([current_strategy.extended.steps_per_run] *
                  (steps_per_epoch //
                   current_strategy.extended.steps_per_run))
  if steps_per_epoch % current_strategy.extended.steps_per_run:
    steps_to_run.append(
        steps_per_epoch % current_strategy.extended.steps_per_run)
  target_steps = len(steps_to_run)

  callbacks._call_begin_hook(mode)

  initial_epoch = model._maybe_load_initial_epoch_from_ckpt(initial_epoch, mode)

  for epoch in range(initial_epoch, epochs):
    dist_utils._reset_metrics(model)
    callbacks.on_epoch_begin(epoch)
    epoch_logs = {}
    step_index = 0
    prev_step_count = None
    current_step = 0
    while current_step < target_steps:
      step_count = steps_to_run[current_step]
      batch_logs = {'batch': step_index, 'size': 1, 'num_steps': step_count}
      callbacks._call_batch_hook(mode, 'begin', step_index, batch_logs)
      if prev_step_count is None or step_count != prev_step_count:
        backend.get_session().run(steps_per_run.assign(step_count))
        prev_step_count = step_count
      try:
        _, outputs = backend.batch_get_value([train_op, output_tensors])
      except errors.OutOfRangeError:
        logging.warning('Your dataset iterator ran out of data; '
                        'interrupting training. Make sure that your dataset '
                        'can generate at least `steps_per_epoch * epochs` '
                        'batches (in this case, %d batches).' %
                        steps_per_epoch * epochs)
        break

      batch_logs.update(outputs)
      callbacks._call_batch_hook(mode, 'end', step_index, batch_logs)
      step_index = step_index + step_count
      current_step += 1

      if callbacks.model.stop_training:
        break

    if (do_validation and
        training_utils_v1.should_run_validation(validation_freq, epoch)):
      logging.info('Running validation at fit epoch: %s', epoch)

      if model._compile_distribution:
        # Since we create a new clone from the original model we need to copy
        # the weights back to the original model before we can run validation.
        dist_utils._copy_weights_to_original_model(model, ModeKeys.TRAIN)

      val_outs = experimental_tpu_test_loop(  # pylint: disable=undefined-variable
          model,
          val_dataset,
          steps=validation_steps,
          verbose=verbose,
          callbacks=callbacks)
      if not isinstance(val_outs, list):
        val_outs = [val_outs]
      # Same labels assumed.
      for label, val_out in zip(out_labels, val_outs):
        epoch_logs['val_' + label] = val_out

    callbacks.on_epoch_end(epoch, epoch_logs)
    if callbacks.model.stop_training:
      break
  model._successful_loop_finish = True
  callbacks._call_end_hook(mode)

  if model._compile_distribution:
    # Copy the weights back from the replicated model to the original model.
    dist_utils._copy_weights_to_original_model(model, ModeKeys.TRAIN)
  scope.__exit__(None, None, None)
  return model.history


def experimental_tpu_test_loop(model,
                               dataset,
                               verbose=0,
                               steps=None,
                               callbacks=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: a check that the outputs list is non-empty is written so it is always true,
  # so loss is always divided by the step count; the division is by the (possibly shortened)
  # target step count, and the strategy scope is not exited when an exception is raised.
  # Graph-mode custom evaluation loop for TPU strategies. Takes the model, dataset, verbosity,
  # number of steps (required) and callbacks. Builds a per-replica test step, reduces outputs
  # (sum for loss, mean for others), copies weights to the distributed model, resets metrics,
  # then runs step by step with callbacks and an optional progress bar. Loss is accumulated and
  # averaged over steps while other metrics keep the latest value. Raises a value error when
  # steps is not given. Returns a single value if there is one metric, otherwise a list.
  """Test loop for evaluating with TPU tf.distribute.Strategy.

  Args:
      model: Keras Model instance.
      dataset: Dataset for input data.
      verbose: Integer, Verbosity mode 0 or 1.
      steps: Total number of steps (batches of samples)
          before declaring predictions finished.
          Ignored with the default value of `None`.
      callbacks: List of callbacks to be called during training

  Returns:
      Scalar loss (if the model has a single output and no metrics)
      or list of scalars (if the model has multiple outputs
      and/or metrics). The attribute `model.metrics_names` will give you
      the display labels for the outputs.
  """
  mode = ModeKeys.TEST
  current_strategy = model._distribution_strategy
  iterator = dist_utils.get_iterator(dataset, current_strategy)

  scope = dist_utils.distributed_scope(
      strategy=current_strategy, learning_phase=0)
  scope.__enter__()

  out_labels = model.metrics_names

  def _test_step_fn(inputs):
    # [FUNCTION DOC]
    # HELPER: per-replica test step for experimental_tpu_test_loop. Takes one batch (inputs
    # and targets pair, or just inputs), builds the distributed model through a merge call in
    # the replica context, runs the per-replica execution function, and returns the output
    # tensors with a control dependency on the update operation.
    """A fn that returns output of single test step."""
    if isinstance(inputs, (tuple, list)) and len(inputs) == 2:
      inputs, targets = inputs
    else:
      targets = None

    (distribute_lib.get_replica_context().merge_call(
        _build_model, args=(model, mode, inputs, targets)))

    (_, outputs, updates, _) = _per_replica_execution_function(
        dist_utils.get_distributed_model(model, mode), mode)
    with ops.control_dependencies([updates]):
      return [array_ops.identity(out) for out in outputs]

  test_input_data = iterator.get_next()
  per_replica_outputs = current_strategy.run(
      _test_step_fn, args=(test_input_data,))
  output_tensors = {}
  for label, output in zip(out_labels, per_replica_outputs):
    if label == 'loss':
      reduce_op = ds_reduce_util.ReduceOp.SUM
    else:
      # We reduce all other metrics using mean for now. This is temporary
      # workaround until new metrics are in place.
      reduce_op = ds_reduce_util.ReduceOp.MEAN
    output_tensors[label] = current_strategy.reduce(reduce_op, output,
                                                    axis=None)
  test_op = control_flow_ops.group(list(output_tensors.values()))

  if verbose >= 1:
    progbar = Progbar(target=steps)

  if model._compile_distribution:
    dist_utils._copy_weights_to_distributed_model(model, mode)

  dist_utils._reset_metrics(model)

  callbacks = cbks.configure_callbacks(
      callbacks,
      model,
      do_validation=False,
      epochs=1,
      steps_per_epoch=steps,
      verbose=verbose,
      count_mode='steps',
      mode=ModeKeys.TEST)
  callbacks._call_begin_hook(mode)

  outs = [0.] * len(model.metrics_names)
  if steps is not None:
    target_steps = steps
  else:
    raise ValueError('Number of steps could not be inferred from the data, '
                     'please pass the steps argument.')

  current_step = 0
  while current_step < target_steps:
    batch_logs = {'batch': current_step, 'size': 1}
    callbacks._call_batch_hook(mode, 'begin', current_step, batch_logs)
    try:
      _, batch_outs = backend.batch_get_value([test_op, output_tensors])
    except errors.OutOfRangeError:
      warning_msg = (
          'Make sure that your dataset can generate at least '
          '`steps` batches (in this case, {} batches).'.format(steps))

      logging.warning('Your dataset iterator ran out of data; '
                      'interrupting evaluation. ' + warning_msg)
      target_steps = current_step
      break
    for i, label in enumerate(model.metrics_names):
      if i == 0:
        # Loss is stateless metrics.
        outs[i] += batch_outs[label]
      else:
        # For all stateful metrics, the aggregation is handled by mirrored vars.
        outs[i] = batch_outs[label]

    batch_logs = cbks.make_logs(model, batch_logs, outs, mode)
    callbacks._call_batch_hook(mode, 'end', current_step, batch_logs)
    if verbose == 1:
      progbar.update(current_step + 1)
    current_step += 1

  if verbose >= 1:
    # Progress bar finishes at the end.
    progbar.update(target_steps)
  callbacks._call_end_hook(mode)

  scope.__exit__(None, None, None)
  if len(outs) >= 0:
    outs[0] /= (target_steps)

  if len(outs) == 1:
    return outs[0]
  return outs


def experimental_tpu_predict_loop(model,
                                  dataset,
                                  verbose=0,
                                  steps=None,
                                  callbacks=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the strategy scope is entered by hand and not exited if an error (such as
  # the missing-steps error) is raised afterwards.
  # Graph-mode custom prediction loop for TPU strategies. Takes the model, dataset, verbosity,
  # number of steps (required) and callbacks. If the dataset shape is not fully defined, pads
  # partial batches (via a partial-batch padding handler), unbatches and rebatches with
  # remainder dropped to get static shapes, preserving prefetch. Builds a per-replica predict
  # step, copies weights, resets metrics, then runs steps collecting per-replica outputs per
  # model output; stops with a warning if data runs out. Raises a value error if steps is not
  # given. Returns one concatenated array for a single output or a list of arrays, with padding
  # removed when a padding handler was used.
  """Predict loop for predicting with TPU tf.distribute.Strategy.

  Args:
      model: Keras Model instance.
      dataset: Dataset for input data.
      verbose: Integer, Verbosity mode 0 or 1.
      steps: Total number of steps (batches of samples)
          before declaring `_predict_loop` finished.
          Ignored with the default value of `None`.
      callbacks: List of callbacks to be called during training

  Returns:
      Array of predictions (if the model has a single output)
      or list of arrays of predictions
      (if the model has multiple outputs).
  """
  mode = ModeKeys.PREDICT
  dataset_fully_shaped = dist_utils.is_dataset_shape_fully_defined(dataset)
  padding_handler = None
  if not dataset_fully_shaped:
    # TODO(hongjunchoi): Investigate whether operations from
    # PartialBatchPaddingHandler are unnecessarily pruned out
    # during graph optimization.
    padding_handler = padding_util.PartialBatchPaddingHandler(
        model._feed_output_shapes)
    batch_size, _, prefetch_buffer = input_lib._get_dataset_attributes(dataset)
    padding_handler.padded_batch_size = batch_size
    padding_handler.padding_mask = dataset.reduce(padding_handler.padding_mask,
                                                  padding_handler.update_mask)

    dataset = dataset.map(padding_handler.pad_batch)
    dataset = dataset.unbatch()
    # Upon this point, it is guaranteed that the dataset does not
    # have partial batches. Thus, we set `drop_remainder=True` to
    # get static shape information about the elements in the dataset.
    dataset = dataset.batch(batch_size, drop_remainder=True)

    if prefetch_buffer is not None:
      dataset = dataset.prefetch(prefetch_buffer)

  current_strategy = model._distribution_strategy
  iterator = dist_utils.get_iterator(dataset, current_strategy)

  scope = dist_utils.distributed_scope(
      strategy=current_strategy, learning_phase=0)
  scope.__enter__()

  def _predict_step_fn(inputs):
    # [FUNCTION DOC]
    # HELPER: per-replica prediction step for experimental_tpu_predict_loop. Takes one batch of
    # inputs, builds the distributed model through a merge call in the replica context, runs the
    # per-replica execution function, and returns the outputs with a control dependency on the
    # update operation.
    """A fn that returns output of single prediction step."""

    (distribute_lib.get_replica_context().merge_call(
        _build_model, args=(model, mode, inputs)))

    (_, outputs, updates, _) = _per_replica_execution_function(
        dist_utils.get_distributed_model(model, mode), mode)

    with ops.control_dependencies([updates]):
      return [array_ops.identity(out) for out in outputs]

  # TODO(hongjunchoi): When numpy array is passed as an input to `predict()`
  # use numpy arrays directly to avoid cumulating unnecessary input pipeline
  # ops.
  predict_input_data = iterator.get_next()
  per_replica_outputs = current_strategy.run(
      _predict_step_fn, args=(predict_input_data,))
  output_tensors = dist_utils.flatten_per_replica_values(
      current_strategy, per_replica_outputs)

  if verbose >= 1:
    progbar = Progbar(target=steps)

  if model._compile_distribution:
    dist_utils._copy_weights_to_distributed_model(model, mode)

  dist_utils._reset_metrics(model)

  callbacks = cbks.configure_callbacks(
      callbacks,
      model,
      do_validation=False,
      epochs=1,
      steps_per_epoch=steps,
      verbose=verbose,
      count_mode='steps',
      mode=mode)
  callbacks._call_begin_hook(mode)

  # Since we do not know how many samples we will see, we cannot pre-allocate
  # the returned Numpy arrays. Instead, we store one array per batch seen
  # and concatenate them upon returning.
  num_model_outputs = len(model.output_names)
  unconcatenated_outs = [[] for _ in range(num_model_outputs)]
  if steps is not None:
    target_steps = steps
  else:
    raise ValueError('Number of steps could not be inferred from the data, '
                     'please pass the steps argument.')

  current_step = 0
  while current_step < target_steps:
    batch_logs = {'batch': current_step, 'size': 1}
    callbacks._call_batch_hook(mode, 'begin', current_step, batch_logs)
    try:
      predict_ops = control_flow_ops.group(output_tensors)
      _, batch_outs = backend.batch_get_value([predict_ops, output_tensors])

    except errors.OutOfRangeError:
      warning_msg = (
          'Make sure that your dataset can generate at least '
          '`steps` batches (in this case, {} batches).'.format(steps))

      logging.warning('Your dataset iterator ran out of data; '
                      'interrupting evaluation. ' + warning_msg)
      break

    # TODO(priyag): maybe need to unwrap the outputs first for MirroredStrategy.
    for i in range(num_model_outputs):
      output_start_index = i * current_strategy.num_replicas_in_sync
      output_end_index = (
          output_start_index + current_strategy.num_replicas_in_sync)
      single_model_output = batch_outs[output_start_index:output_end_index]
      unconcatenated_outs[i].extend(single_model_output)

    batch_logs = cbks.make_logs(model, batch_logs, batch_outs, mode)
    callbacks._call_batch_hook(mode, 'end', current_step, batch_logs)
    if verbose == 1:
      progbar.update(current_step + 1)
    current_step += 1

  if verbose >= 1:
    # Progress bar finishes at the end.
    progbar.update(current_step)

  callbacks._call_end_hook(mode)

  scope.__exit__(None, None, None)

  if len(unconcatenated_outs) == 1:
    prediction_result = np.concatenate(unconcatenated_outs[0], axis=0)
  else:
    prediction_result = [
        np.concatenate(out, axis=0) for out in unconcatenated_outs
    ]

  if padding_handler:
    prediction_result = padding_handler.apply_mask(prediction_result)

  return prediction_result


class DistributionSingleWorkerTrainingLoop(training_utils_v1.TrainingLoop):
  # [CLASS DOC]
  # Stateless training loop for a model with a distribution strategy on a single worker.
  # Overrides fit, evaluate and predict: each validates inputs, computes batch and step sizes,
  # standardizes data into datasets, and either uses the TPU custom graph loops or the array
  # based loops from training_arrays_v1.
  """Training loop for distribution strategy with single worker."""

  def fit(self,
          model,
          x=None,
          y=None,
          batch_size=None,
          epochs=1,
          verbose=1,
          callbacks=None,
          validation_split=0.,
          validation_data=None,
          shuffle=True,
          class_weight=None,
          sample_weight=None,
          initial_epoch=0,
          steps_per_epoch=None,
          validation_steps=None,
          validation_freq=1,
          **kwargs):
    # [FUNCTION DOC]
    # Fit for distribution strategies. Validates callbacks against the optimizer and the
    # inputs, resolves batch size and steps per epoch, standardizes data into a dataset (and
    # again under the strategy scope when not distributing by cloning), and prepares an
    # optional validation dataset from validation_data. Raises a value error when
    # validation_split is given without validation data (unsupported), or when a TPU strategy
    # cannot infer the number of steps. Uses the TPU custom loop in graph mode, otherwise the
    # array fit loop. Returns the training history. Extra keyword arguments are ignored.
    """Fit loop for Distribution Strategies."""
    dist_utils.validate_callbacks(input_callbacks=callbacks,
                                  optimizer=model.optimizer)
    dist_utils.validate_inputs(x, y)

    batch_size, steps_per_epoch = dist_utils.process_batch_and_step_size(
        model._distribution_strategy,
        x,
        batch_size,
        steps_per_epoch,
        ModeKeys.TRAIN,
        validation_split=validation_split)
    batch_size = model._validate_or_infer_batch_size(
        batch_size, steps_per_epoch, x)
    dataset = model._distribution_standardize_user_data(
        x, y,
        sample_weight=sample_weight,
        class_weight=class_weight,
        batch_size=batch_size,
        validation_split=validation_split,
        shuffle=shuffle,
        epochs=epochs)
    if not dist_utils.is_distributing_by_cloning(model):
      with model._distribution_strategy.scope():
        (dataset, _, _) = model._standardize_user_data(
            dataset,
            sample_weight=sample_weight,
            class_weight=class_weight,
            batch_size=batch_size,
            validation_split=validation_split,
            shuffle=shuffle)

    val_dataset = None
    if validation_data:
      val_x, val_y, val_sample_weights = (
          training_utils_v1.unpack_validation_data(validation_data))
      dist_utils.validate_inputs(val_x, val_y)
      _, validation_steps = dist_utils.process_batch_and_step_size(
          model._distribution_strategy, val_x, batch_size, validation_steps,
          ModeKeys.TEST)

      val_dataset = model._distribution_standardize_user_data(
          val_x, val_y,
          sample_weight=val_sample_weights,
          class_weight=None,
          batch_size=batch_size,
          validation_split=validation_split,
          shuffle=shuffle,
          allow_partial_batch=True)
    elif validation_split:
      raise ValueError('validation_split argument is not supported with '
                       'distribution strategies.')

    if backend.is_tpu_strategy(model._distribution_strategy):
      steps_per_epoch = training_utils_v1.infer_steps_for_dataset(
          model, dataset, steps_per_epoch, epochs, steps_name='steps_per_epoch')
      if steps_per_epoch is None:
        raise ValueError('Number of steps could not be inferred from the data, '
                         'please pass the steps_per_epoch argument.')

      if not context.executing_eagerly():
        # Run TPU training in a custom loop in graph mode.
        return experimental_tpu_fit_loop(
            model,
            dataset,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            val_dataset=val_dataset,
            initial_epoch=initial_epoch,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps,
            validation_freq=validation_freq)

    return training_arrays_v1.fit_loop(
        model,
        dataset,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        val_inputs=val_dataset,
        shuffle=shuffle,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        validation_freq=validation_freq,
        steps_name='steps_per_epoch')

  def evaluate(self,
               model,
               x=None,
               y=None,
               batch_size=None,
               verbose=1,
               sample_weight=None,
               steps=None,
               callbacks=None,
               **kwargs):
    # [FUNCTION DOC]
    # Evaluate for distribution strategies. Validates inputs, resolves batch size and steps,
    # standardizes data into a dataset allowing a partial final batch. For TPU strategies infers
    # steps (raising a value error if impossible) and in graph mode uses the TPU test loop;
    # otherwise uses the array test loop. Returns the evaluation metrics. Extra keyword
    # arguments are ignored.
    """Evaluate loop for Distribution Strategies."""
    dist_utils.validate_inputs(x, y)
    batch_size, steps = dist_utils.process_batch_and_step_size(
        model._distribution_strategy, x, batch_size, steps, ModeKeys.TEST)
    batch_size = model._validate_or_infer_batch_size(batch_size, steps, x)
    dataset = model._distribution_standardize_user_data(
        x, y,
        sample_weight=sample_weight,
        batch_size=batch_size,
        allow_partial_batch=True)

    if backend.is_tpu_strategy(model._distribution_strategy):
      steps = training_utils_v1.infer_steps_for_dataset(
          model, dataset, steps, steps_name='steps')
      if steps is None:
        raise ValueError('Number of steps could not be inferred from the data, '
                         'please pass the steps argument.')

      if not context.executing_eagerly():
        # Run TPU evaluation in a custom loop in graph mode.
        return experimental_tpu_test_loop(
            model, dataset, verbose=verbose, steps=steps, callbacks=callbacks)

    return training_arrays_v1.test_loop(
        model,
        inputs=dataset,
        batch_size=batch_size,
        verbose=verbose,
        steps=steps,
        callbacks=callbacks)

  def predict(self,
              model,
              x,
              batch_size=None,
              verbose=0,
              steps=None,
              callbacks=None,
              **kwargs):
    # [FUNCTION DOC]
    # Predict for distribution strategies. Validates inputs, resolves batch size and steps,
    # standardizes data into a dataset allowing a partial final batch. For TPU strategies infers
    # steps (raising a value error if impossible) and in graph mode uses the TPU predict loop;
    # otherwise uses the array predict loop. Returns the predictions. Extra keyword arguments
    # are ignored.
    """Predict loop for Distribution Strategies."""
    dist_utils.validate_inputs(x=x, y=None)
    batch_size, steps = dist_utils.process_batch_and_step_size(
        model._distribution_strategy, x, batch_size, steps, ModeKeys.PREDICT)
    batch_size = model._validate_or_infer_batch_size(batch_size, steps, x)
    dataset = model._distribution_standardize_user_data(
        x,
        batch_size=batch_size,
        allow_partial_batch=True)
    if backend.is_tpu_strategy(model._distribution_strategy):
      steps = training_utils_v1.infer_steps_for_dataset(
          model, dataset, steps, steps_name='steps')
      if steps is None:
        raise ValueError('Number of steps could not be inferred from the data, '
                         'please pass the steps argument.')
      if not context.executing_eagerly():
        return experimental_tpu_predict_loop(
            model, dataset, verbose=verbose, steps=steps, callbacks=callbacks)
    return training_arrays_v1.predict_loop(
        model,
        dataset,
        batch_size=batch_size,
        verbose=verbose,
        steps=steps,
        callbacks=callbacks)


def _train_with_multi_worker(method):
  # [FUNCTION DOC]
  # HELPER: used by DistributionMultiWorkerTrainingLoop. Takes a single-worker loop method and
  # returns a wrapper that runs it under the distribute coordinator on every worker.
  """Decorator that handles multi worker training with distribution strategy."""

  def wrapper(model, **kwargs):
    # [FUNCTION DOC]
    # HELPER: wrapper returned by _train_with_multi_worker. Takes the model and keyword
    # arguments, and runs a per-worker function through the distribute coordinator using the
    # model's strategy; returns the coordinator's result. The wrapper accepts only keyword
    # arguments besides the model.
    def _worker_fn(_):
      # [FUNCTION DOC]
      # HELPER: per-worker function. Ignores its argument, removes the callbacks from the
      # shared keyword arguments, filters them to those valid for distributed workers, stores
      # the filtered callbacks back (mutating the keyword mapping) and calls the wrapped method.
      # Returns the method result.
      callbacks = kwargs.pop('callbacks', None)
      filtered_callbacks = dist_utils.filter_distributed_callbacks(
          callbacks, model)
      kwargs['callbacks'] = filtered_callbacks
      return method(model, **kwargs)

    return dc.run_distribute_coordinator(
        _worker_fn,
        model._distribution_strategy)

  return wrapper


class DistributionMultiWorkerTrainingLoop(training_utils_v1.TrainingLoop):
  # [CLASS DOC]
  # Training loop for distribution strategies across multiple workers. Holds a single-worker
  # loop and delegates to it: fit and evaluate run through the distribute coordinator on each
  # worker, while predict still uses the single-worker implementation directly.
  """Training loop for distribution strategy with multiple worker."""

  def __init__(self, single_worker_loop):
    # [FUNCTION DOC]
    # Stores the single-worker training loop (the object whose methods are run on each
    # worker). Returns nothing.
    self._single_worker_loop = single_worker_loop

  def fit(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Runs the single-worker fit through the multi-worker wrapper; accepts the model and
    # keyword arguments and returns the fit result of the coordinator run.
    return _train_with_multi_worker(self._single_worker_loop.fit)(
        *args, **kwargs)

  def evaluate(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Runs the single-worker evaluate through the multi-worker wrapper; accepts the model and
    # keyword arguments and returns the evaluation result of the coordinator run.
    return _train_with_multi_worker(self._single_worker_loop.evaluate)(
        *args, **kwargs)

  def predict(self, *args, **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: unlike fit and evaluate, this does not use the multi-worker wrapper; it
    # simply forwards all arguments to the single-worker predict.
    # Currently predict is still using the single worker implementation.
    return self._single_worker_loop.predict(*args, **kwargs)
