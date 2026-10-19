# [MODULE DOC]
# Legacy (tf.python.keras) callbacks module. Defines the Callback base class, the CallbackList
# container that fans lifecycle events out to many callbacks, and the built-in callbacks: loggers
# (BaseLogger, ProgbarLogger, History, CSVLogger, RemoteMonitor, LambdaCallback), training
# control (ModelCheckpoint, BackupAndRestore, EarlyStopping, TerminateOnNaN, LearningRateScheduler,
# ReduceLROnPlateau) and TensorBoard integration (TensorBoard, including profiling support).
# Also contains helpers used by the training loops to build and configure a callback list.
# Import-time side effects: tries to import the optional requests package and falls back to a
# null value when it is missing; no other global registration is performed here.
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
# pylint: disable=g-import-not-at-top
# pylint: disable=g-classes-have-attributes
"""Callbacks: utilities called at certain points during model training."""

import collections
import copy
import csv
import json
import os
import re
import sys
import time

import numpy as np

from tensorflow.core.framework import summary_pb2
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.checkpoint import checkpoint_options as checkpoint_options_lib
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.distribute import collective_all_reduce_strategy
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import mirrored_strategy
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.keras import backend
from tensorflow.python.keras.distribute import distributed_file_utils
from tensorflow.python.keras.distribute import worker_training_state
from tensorflow.python.keras.optimizer_v2 import learning_rate_schedule
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.data_utils import Sequence
from tensorflow.python.keras.utils.generic_utils import Progbar
from tensorflow.python.keras.utils.io_utils import path_to_string
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import profiler_v2 as profiler
from tensorflow.python.saved_model import save_options as save_options_lib
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls

try:
  import requests
except ImportError:
  requests = None


# Note: `configure_callbacks` is only used in TF1.
def configure_callbacks(callbacks,
                        model,
                        do_validation=False,
                        batch_size=None,
                        epochs=None,
                        steps_per_epoch=None,
                        samples=None,
                        verbose=1,
                        count_mode='steps',
                        mode=ModeKeys.TRAIN):
  # [FUNCTION DOC]
  # HELPER: glue used by the legacy (TF1-style) training loops to build the callback container.
  # Wraps the user callbacks into a CallbackList (returned unchanged if it already is one). In
  # train mode it creates a fresh History object, stores it on the model, and surrounds the
  # user callbacks with a BaseLogger in front and the history behind, plus a ProgbarLogger when
  # verbose is truthy (using count_mode). Then it attaches the model returned by the model's
  # callback-model accessor, pushes parameters via set_callback_parameters, and resets the
  # model's stop-training flag to false. Inputs: callbacks (list or nothing), model, optional
  # do_validation, batch_size, epochs, steps_per_epoch, samples, verbose, count_mode ('steps' or
  # 'samples') and mode (train/test/predict key). Returns the configured CallbackList.
  # Side effects: mutates the model (history, stop_training).
  """Configures callbacks for use in various training loops.

  Args:
      callbacks: List of Callbacks.
      model: Model being trained.
      do_validation: Whether or not validation loop will be run.
      batch_size: Number of samples per batch.
      epochs: Number of epoch to train.
      steps_per_epoch: Number of batches to run per training epoch.
      samples: Number of training samples.
      verbose: int, 0 or 1. Keras logging verbosity to pass to ProgbarLogger.
      count_mode: One of 'steps' or 'samples'. Per-batch or per-sample count.
      mode: String. One of ModeKeys.TRAIN, ModeKeys.TEST, or ModeKeys.PREDICT.
        Which loop mode to configure callbacks for.

  Returns:
      Instance of CallbackList used to control all Callbacks.
  """
  # Check if callbacks have already been configured.
  if isinstance(callbacks, CallbackList):
    return callbacks

  if not callbacks:
    callbacks = []

  # Add additional callbacks during training.
  if mode == ModeKeys.TRAIN:
    model.history = History()
    callbacks = [BaseLogger()] + (callbacks or []) + [model.history]
    if verbose:
      callbacks.append(ProgbarLogger(count_mode))
  callback_list = CallbackList(callbacks)

  # Set callback model
  callback_model = model._get_callback_model()  # pylint: disable=protected-access
  callback_list.set_model(callback_model)

  set_callback_parameters(
      callback_list,
      model,
      do_validation=do_validation,
      batch_size=batch_size,
      epochs=epochs,
      steps_per_epoch=steps_per_epoch,
      samples=samples,
      verbose=verbose,
      mode=mode)

  callback_list.model.stop_training = False
  return callback_list


def set_callback_parameters(callback_list,
                            model,
                            do_validation=False,
                            batch_size=None,
                            epochs=None,
                            steps_per_epoch=None,
                            samples=None,
                            verbose=1,
                            mode=ModeKeys.TRAIN):
  # [FUNCTION DOC]
  # HELPER: internal glue for configure_callbacks that pushes run parameters into a CallbackList.
  # Marks the metric names (without the leading loss entry) as stateful metrics on every
  # BaseLogger and ProgbarLogger in the list. Builds the metric name list (empty in predict mode;
  # otherwise a copy of the model's metric names, plus val_-prefixed copies when do_validation is
  # true) and calls set_params on the list with a mapping holding batch size, epochs, steps,
  # samples, verbose, do_validation and metrics. Inputs: callback_list, model, optional
  # do_validation, batch_size, epochs, steps_per_epoch, samples, verbose, mode. Returns nothing;
  # mutates the callbacks in the list.
  """Sets callback parameters.

  Args:
      callback_list: CallbackList instance.
      model: Model being trained.
      do_validation: Whether or not validation loop will be run.
      batch_size: Number of samples per batch.
      epochs: Number of epoch to train.
      steps_per_epoch: Number of batches to run per training epoch.
      samples: Number of training samples.
      verbose: int, 0 or 1. Keras logging verbosity to pass to ProgbarLogger.
      mode: String. One of ModeKeys.TRAIN, ModeKeys.TEST, or ModeKeys.PREDICT.
        Which loop mode to configure callbacks for.
  """
  metric_names = model.metrics_names
  for cbk in callback_list:
    if isinstance(cbk, (BaseLogger, ProgbarLogger)):
      cbk.stateful_metrics = metric_names[1:]  # Exclude `loss`

  # Set callback parameters
  callback_metrics = []
  # When we have deferred build scenario with iterator input, we will compile
  # when we standardize first batch of data.
  if mode != ModeKeys.PREDICT:
    callback_metrics = copy.copy(metric_names)
    if do_validation:
      callback_metrics += ['val_' + n for n in metric_names]
  callback_params = {
      'batch_size': batch_size,
      'epochs': epochs,
      'steps': steps_per_epoch,
      'samples': samples,
      'verbose': verbose,
      'do_validation': do_validation,
      'metrics': callback_metrics,
  }
  callback_list.set_params(callback_params)


def _is_generator_like(data):
  # [FUNCTION DOC]
  # HELPER: private predicate for the training-data handling code.
  # Returns true when the data object has a next-style iteration method (Python 3 or 2 flavor) or
  # is an instance of the Keras Sequence class or a dataset iterator type; false otherwise.
  # Takes one arbitrary object, has no side effects and raises nothing.
  """Checks if data is a generator, Sequence, or Iterator."""
  return (hasattr(data, '__next__') or hasattr(data, 'next') or isinstance(
      data, (Sequence, iterator_ops.Iterator, iterator_ops.IteratorBase)))


def make_logs(model, logs, outputs, mode, prefix=''):
  # [FUNCTION DOC]
  # HELPER: used by the training loops to build the logs mapping handed to batch-end hooks.
  # In train or test mode, when the model has metric names, each metric name (with the optional
  # prefix prepended) is stored in the logs mapping with the matching entry of outputs, pairing
  # them in order. Otherwise (predict mode, or no metric names) the whole outputs value is stored
  # under the key 'outputs'. Mutates and returns the same logs mapping. Raises nothing itself.
  """Computes logs for sending to `on_batch_end` methods."""
  metric_names = model.metrics_names
  if mode in {ModeKeys.TRAIN, ModeKeys.TEST} and metric_names:
    for label, output in zip(metric_names, outputs):
      logs[prefix + label] = output
  else:
    logs['outputs'] = outputs
  return logs


class CallbackList:
  # [CLASS DOC]
  # Container that holds an ordered list of Callback objects and forwards each lifecycle event
  # (epoch, train/test/predict begin and end, batch begin and end) to all of them. State: the
  # callbacks list, optional progress-bar and history callbacks, the shared params and model,
  # cached flags saying whether any callback implements batch-level hooks or supports raw
  # tensor logs, and bookkeeping used to warn when callback hooks are slow compared to batch time.
  # Created by the training loops (or users writing custom loops); it is iterable over the
  # callbacks it holds.
  """Container abstracting a list of callbacks."""

  def __init__(self,
               callbacks=None,
               add_history=False,
               add_progbar=False,
               model=None,
               **params):
    # [FUNCTION DOC]
    # Builds the container. callbacks is an optional (possibly nested) list that is flattened;
    # add_history and add_progbar request a History or ProgbarLogger callback when none is
    # present; model, if given, is attached to every callback; extra keyword params, if given, are
    # passed to every callback as parameters. Computes whether tensor logs are supported by all
    # callbacks, which batch hooks (train, test, predict) are implemented by any callback, and sets
    # up the slow-hook timing check (only enabled when a callback class is not defined in this
    # module; it looks at the first five batches). Raises a value error if batch-level callbacks
    # are used with a parameter-server strategy (via the strategy check).
    """Container for `Callback` instances.

    This object wraps a list of `Callback` instances, making it possible
    to call them all at once via a single endpoint
    (e.g. `callback_list.on_epoch_end(...)`).

    Args:
      callbacks: List of `Callback` instances.
      add_history: Whether a `History` callback should be added, if one does not
        already exist in the `callbacks` list.
      add_progbar: Whether a `ProgbarLogger` callback should be added, if one
        does not already exist in the `callbacks` list.
      model: The `Model` these callbacks are used with.
      **params: If provided, parameters will be passed to each `Callback` via
        `Callback.set_params`.
    """
    self.callbacks = nest.flatten(callbacks) if callbacks else []
    self._add_default_callbacks(add_history, add_progbar)

    if model:
      self.set_model(model)
    if params:
      self.set_params(params)

    # Performance optimization: determines if batch hooks need to be called.
    # pylint: disable=protected-access
    self._supports_tf_logs = all(
        getattr(cb, '_supports_tf_logs', False) for cb in self.callbacks)
    self._batch_hooks_support_tf_logs = all(
        getattr(cb, '_supports_tf_logs', False)
        for cb in self.callbacks
        if cb._implements_train_batch_hooks() or cb
        ._implements_test_batch_hooks() or cb._implements_predict_batch_hooks())

    self._should_call_train_batch_hooks = any(
        cb._implements_train_batch_hooks() for cb in self.callbacks)
    self._should_call_test_batch_hooks = any(
        cb._implements_test_batch_hooks() for cb in self.callbacks)
    self._should_call_predict_batch_hooks = any(
        cb._implements_predict_batch_hooks() for cb in self.callbacks)
    # pylint: enable=protected-access

    self._disallow_batch_hooks_in_ps_strategy()

    # Performance check: Check batch hooks for slowness compared to batch time.
    # Only run check for custom callbacks (i.e. not present in this file).
    self._check_timing = any(
        cbk.__class__.__name__ not in globals() for cbk in self.callbacks)
    self._num_batches_for_timing_check = 5
    self._hook_times = {}
    self._batch_start_time = None
    self._batch_times = []

  def _add_default_callbacks(self, add_history, add_progbar):
    # [FUNCTION DOC]
    # HELPER: internal setup step of the constructor.
    # Scans the callbacks for an existing ProgbarLogger and History and remembers them. If a
    # progress bar was requested and none exists, creates one counting steps and inserts it at the
    # front; if a history was requested and none exists, creates one and appends it. Returns
    # nothing; mutates the callbacks list and the progbar/history attributes.
    """Adds `Callback`s that are always present."""
    self._progbar = None
    self._history = None

    for cb in self.callbacks:
      if isinstance(cb, ProgbarLogger):
        self._progbar = cb
      elif isinstance(cb, History):
        self._history = cb

    if self._progbar is None and add_progbar:
      self._progbar = ProgbarLogger(count_mode='steps')
      self.callbacks.insert(0, self._progbar)

    if self._history is None and add_history:
      self._history = History()
      self.callbacks.append(self._history)

  def _process_logs(self, logs, is_batch_hook=False):
    # [FUNCTION DOC]
    # HELPER: normalizes logs before they are passed to callbacks.
    # Returns an empty mapping when logs is nothing. Returns logs untouched when every callback
    # supports raw tensor logs, or when this is a batch hook (is_batch_hook true) and all callbacks
    # that implement batch hooks support them. Otherwise converts tensors in logs to numpy values
    # or Python scalars (this can force a synchronization with remote or async execution).
    """Turns tensors into numpy arrays or Python scalars if necessary."""
    if logs is None:
      return {}
    if self._supports_tf_logs:
      return logs
    if is_batch_hook and self._batch_hooks_support_tf_logs:
      return logs
    return tf_utils.sync_to_numpy_or_python_type(logs)

  def append(self, callback):
    # [FUNCTION DOC]
    # Adds one callback object to the end of the list. Does not update the cached batch-hook
    # flags, so a callback appended later that implements batch hooks may not be reflected in them.
    self.callbacks.append(callback)

  def set_params(self, params):
    # [FUNCTION DOC]
    # Stores the params mapping (training parameters such as batch size, epochs, steps, verbosity)
    # on the container and forwards it to every callback by calling their set_params.
    self.params = params
    for callback in self.callbacks:
      callback.set_params(params)

  def set_model(self, model):
    # [FUNCTION DOC]
    # Stores the model on the container and attaches it to every callback. If a history callback
    # is present, also stores it on the model as the model's history attribute (side effect).
    self.model = model
    if self._history:
      model.history = self._history
    for callback in self.callbacks:
      callback.set_model(model)

  def _call_batch_hook(self, mode, hook, batch, logs=None):
    # [FUNCTION DOC]
    # HELPER: dispatcher for all batch begin and end methods.
    # Does nothing when there are no callbacks. Otherwise, depending on hook ('begin' or 'end'),
    # delegates to the begin or end batch helper with the mode, batch index and optional logs.
    # Raises a value error for any other hook string.
    """Helper function for all batch_{begin | end} methods."""
    if not self.callbacks:
      return

    if hook == 'begin':
      self._call_batch_begin_hook(mode, batch, logs)
    elif hook == 'end':
      self._call_batch_end_hook(mode, batch, logs)
    else:
      raise ValueError('Unrecognized hook: {}'.format(hook))

  def _call_batch_begin_hook(self, mode, batch, logs):
    # [FUNCTION DOC]
    # HELPER: runs the on_<mode>_batch_begin hook of every callback via the shared hook helper.
    # When the slow-hook timing check is active, also records the wall-clock batch start time.
    # Inputs: mode name, batch index, logs mapping. Returns nothing.
    """Helper function for `on_*_batch_begin` methods."""
    hook_name = 'on_{mode}_batch_begin'.format(mode=mode)
    self._call_batch_hook_helper(hook_name, batch, logs)

    if self._check_timing:
      self._batch_start_time = time.time()

  def _call_batch_end_hook(self, mode, batch, logs):
    # [FUNCTION DOC]
    # HELPER: internal batch-end dispatcher of the callback container.
    # KNOWN ISSUE: the average begin-hook time is looked up in the recorded hook times without a
    # guard; if no begin-hook timing was recorded for the mode (for example the begin hook never
    # ran while timing was on) a missing-key error would be raised.
    # Runs the on_<mode>_batch_end hook of every callback and performs the slow-hook check.
    # When timing is enabled and the batch index is at least one, records the elapsed time since
    # the batch began. After calling the hooks, once enough batch times were collected (five), it
    # compares the average begin-hook and end-hook times to the average batch time and logs a
    # warning for each hook that is slower than the batch itself, then disables further checking
    # and clears the timing state. Inputs: mode name, batch index, logs mapping. Returns nothing.
    """Helper function for `on_*_batch_end` methods."""
    hook_name = 'on_{mode}_batch_end'.format(mode=mode)

    if self._check_timing and batch >= 1:
      batch_time = time.time() - self._batch_start_time
      self._batch_times.append(batch_time)

    self._call_batch_hook_helper(hook_name, batch, logs)

    if len(self._batch_times) >= self._num_batches_for_timing_check:
      end_hook_name = hook_name
      begin_hook_name = 'on_{mode}_batch_begin'.format(mode=mode)
      avg_batch_time = sum(self._batch_times) / len(self._batch_times)
      avg_end_hook_time = sum(self._hook_times[end_hook_name]) / len(
          self._hook_times[end_hook_name])
      avg_begin_hook_time = sum(self._hook_times[begin_hook_name]) / len(
          self._hook_times[begin_hook_name])

      threshold_time = 1.0 * avg_batch_time
      warning_msg = ('Callback method `{hook}` is slow compared to '
                     'the batch time (batch time: {batch_time:.4f}s vs '
                     '`{hook}` time: {hook_time:.4f}s). Check your callbacks.')
      if avg_begin_hook_time > threshold_time:
        logging.warning(warning_msg.format(
            hook=begin_hook_name,
            batch_time=avg_batch_time,
            hook_time=avg_begin_hook_time))
      if avg_end_hook_time > threshold_time:
        logging.warning(warning_msg.format(
            hook=end_hook_name,
            batch_time=avg_batch_time,
            hook_time=avg_end_hook_time))
      self._check_timing = False
      self._batch_start_time = None
      self._batch_times = []
      self._hook_times = {}

  def _call_batch_hook_helper(self, hook_name, batch, logs):
    # [FUNCTION DOC]
    # HELPER: shared core of the batch begin and end dispatchers.
    # Converts the logs through the log-processing step (batch-hook mode), then invokes the named
    # hook method on each callback in order with the batch index and logs. When timing is enabled
    # it measures the total time spent in this call and appends it to the list of times recorded
    # under the hook name. Inputs: hook_name (method name string), batch index, logs mapping.
    """Helper function for `on_*_batch_*` methods."""
    if self._check_timing:
      start_time = time.time()

    logs = self._process_logs(logs, is_batch_hook=True)
    for callback in self.callbacks:
      hook = getattr(callback, hook_name)
      hook(batch, logs)

    if self._check_timing:
      if hook_name not in self._hook_times:
        self._hook_times[hook_name] = []
      self._hook_times[hook_name].append(time.time() - start_time)

  def _call_begin_hook(self, mode):
    # [FUNCTION DOC]
    # HELPER: calls the container's train-begin, test-begin or predict-begin method according to the
    # mode key (any mode other than train or test is treated as predict). Returns nothing.
    """Helper function for on_{train|test|predict}_begin methods."""
    if mode == ModeKeys.TRAIN:
      self.on_train_begin()
    elif mode == ModeKeys.TEST:
      self.on_test_begin()
    else:
      self.on_predict_begin()

  def _call_end_hook(self, mode):
    # [FUNCTION DOC]
    # HELPER: calls the container's train-end, test-end or predict-end method according to the mode
    # key (any mode other than train or test is treated as predict). Returns nothing.
    """Helper function for on_{train|test|predict}_end methods."""
    if mode == ModeKeys.TRAIN:
      self.on_train_end()
    elif mode == ModeKeys.TEST:
      self.on_test_end()
    else:
      self.on_predict_end()

  def on_batch_begin(self, batch, logs=None):
    # [FUNCTION DOC]
    # Backwards-compatible alias of the train batch begin event: if any callback implements
    # train batch hooks, forwards the batch index and optional logs to each callback's batch-begin
    # hook in train mode; otherwise does nothing.
    if self._should_call_train_batch_hooks:
      self._call_batch_hook(ModeKeys.TRAIN, 'begin', batch, logs=logs)

  def on_batch_end(self, batch, logs=None):
    # [FUNCTION DOC]
    # Backwards-compatible alias of the train batch end event: if any callback implements train
    # batch hooks, forwards the batch index and optional logs to each callback's batch-end hook in
    # train mode; otherwise does nothing.
    if self._should_call_train_batch_hooks:
      self._call_batch_hook(ModeKeys.TRAIN, 'end', batch, logs=logs)

  def on_epoch_begin(self, epoch, logs=None):
    # [FUNCTION DOC]
    # Notifies every callback that an epoch is starting. epoch is the integer epoch index; logs is
    # an optional mapping (currently unused data). Logs are normalized first (nothing becomes an
    # empty mapping; tensors may be converted to numpy values). Train mode only. Returns nothing.
    """Calls the `on_epoch_begin` methods of its callbacks.

    This function should only be called during TRAIN mode.

    Args:
        epoch: Integer, index of epoch.
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
    logs = self._process_logs(logs)
    for callback in self.callbacks:
      callback.on_epoch_begin(epoch, logs)

  def on_epoch_end(self, epoch, logs=None):
    # [FUNCTION DOC]
    # Notifies every callback that an epoch finished. epoch is the integer epoch index; logs is an
    # optional mapping of epoch metric results (validation keys prefixed with val_), normalized
    # to numpy values/scalars when needed before being passed on. Train mode only. Returns nothing.
    """Calls the `on_epoch_end` methods of its callbacks.

    This function should only be called during TRAIN mode.

    Args:
        epoch: Integer, index of epoch.
        logs: Dict, metric results for this training epoch, and for the
          validation epoch if validation is performed. Validation result keys
          are prefixed with `val_`.
    """
    logs = self._process_logs(logs)
    for callback in self.callbacks:
      callback.on_epoch_end(epoch, logs)

  def on_train_batch_begin(self, batch, logs=None):
    # [FUNCTION DOC]
    # Forwards the start of a training batch (integer batch index, optional logs mapping) to each
    # callback's train-batch-begin hook, but only when at least one callback implements train batch
    # hooks (otherwise skipped for speed). Returns nothing.
    """Calls the `on_train_batch_begin` methods of its callbacks.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict, contains the return value of `model.train_step`. Typically,
          the values of the `Model`'s metrics are returned.  Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
    """
    if self._should_call_train_batch_hooks:
      self._call_batch_hook(ModeKeys.TRAIN, 'begin', batch, logs=logs)

  def on_train_batch_end(self, batch, logs=None):
    # [FUNCTION DOC]
    # Forwards the end of a training batch (integer batch index, optional logs with aggregated
    # metrics) to each callback's train-batch-end hook, only when some callback implements train
    # batch hooks. Returns nothing.
    """Calls the `on_train_batch_end` methods of its callbacks.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Aggregated metric results up until this batch.
    """
    if self._should_call_train_batch_hooks:
      self._call_batch_hook(ModeKeys.TRAIN, 'end', batch, logs=logs)

  def on_test_batch_begin(self, batch, logs=None):
    # [FUNCTION DOC]
    # Forwards the start of an evaluation batch (batch index, optional logs) to each callback's
    # test-batch-begin hook, only when some callback implements test batch hooks. Returns nothing.
    """Calls the `on_test_batch_begin` methods of its callbacks.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict, contains the return value of `model.test_step`. Typically,
          the values of the `Model`'s metrics are returned.  Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
    """
    if self._should_call_test_batch_hooks:
      self._call_batch_hook(ModeKeys.TEST, 'begin', batch, logs=logs)

  def on_test_batch_end(self, batch, logs=None):
    # [FUNCTION DOC]
    # Forwards the end of an evaluation batch (batch index, optional logs with aggregated metrics)
    # to each callback's test-batch-end hook, only when some callback implements test batch hooks.
    # Returns nothing.
    """Calls the `on_test_batch_end` methods of its callbacks.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Aggregated metric results up until this batch.
    """
    if self._should_call_test_batch_hooks:
      self._call_batch_hook(ModeKeys.TEST, 'end', batch, logs=logs)

  def on_predict_batch_begin(self, batch, logs=None):
    # [FUNCTION DOC]
    # Forwards the start of a prediction batch (batch index, optional logs) to each callback's
    # predict-batch-begin hook, only when some callback implements predict batch hooks.
    # Returns nothing.
    """Calls the `on_predict_batch_begin` methods of its callbacks.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict, contains the return value of `model.predict_step`,
          it typically returns a dict with a key 'outputs' containing
          the model's outputs.
    """
    if self._should_call_predict_batch_hooks:
      self._call_batch_hook(ModeKeys.PREDICT, 'begin', batch, logs=logs)

  def on_predict_batch_end(self, batch, logs=None):
    # [FUNCTION DOC]
    # Forwards the end of a prediction batch (batch index, optional logs) to each callback's
    # predict-batch-end hook, only when some callback implements predict batch hooks.
    # Returns nothing.
    """Calls the `on_predict_batch_end` methods of its callbacks.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Aggregated metric results up until this batch.
    """
    if self._should_call_predict_batch_hooks:
      self._call_batch_hook(ModeKeys.PREDICT, 'end', batch, logs=logs)

  def on_train_begin(self, logs=None):
    # [FUNCTION DOC]
    # Notifies every callback that training is starting. logs is an optional mapping (currently
    # carries no data); it is normalized first (nothing becomes an empty mapping). Returns nothing.
    """Calls the `on_train_begin` methods of its callbacks.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
    logs = self._process_logs(logs)
    for callback in self.callbacks:
      callback.on_train_begin(logs)

  def on_train_end(self, logs=None):
    # [FUNCTION DOC]
    # Notifies every callback that training is over. logs is an optional mapping (normalized first).
    # Returns nothing.
    """Calls the `on_train_end` methods of its callbacks.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
    logs = self._process_logs(logs)
    for callback in self.callbacks:
      callback.on_train_end(logs)

  def on_test_begin(self, logs=None):
    # [FUNCTION DOC]
    # Notifies every callback that evaluation is starting. logs is an optional mapping (normalized
    # first). Returns nothing.
    """Calls the `on_test_begin` methods of its callbacks.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
    logs = self._process_logs(logs)
    for callback in self.callbacks:
      callback.on_test_begin(logs)

  def on_test_end(self, logs=None):
    # [FUNCTION DOC]
    # Notifies every callback that evaluation finished. logs is an optional mapping (normalized
    # first). Returns nothing.
    """Calls the `on_test_end` methods of its callbacks.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
    logs = self._process_logs(logs)
    for callback in self.callbacks:
      callback.on_test_end(logs)

  def on_predict_begin(self, logs=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring has a mismatched quote around the method name (single quote
    # opening, backtick closing); this is only a typo and does not affect behavior.
    # Notifies every callback that prediction is starting. logs is an optional mapping
    # (normalized first). Returns nothing.
    """Calls the 'on_predict_begin` methods of its callbacks.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
    logs = self._process_logs(logs)
    for callback in self.callbacks:
      callback.on_predict_begin(logs)

  def on_predict_end(self, logs=None):
    # [FUNCTION DOC]
    # Notifies every callback that prediction finished. logs is an optional mapping (normalized
    # first). Returns nothing.
    """Calls the `on_predict_end` methods of its callbacks.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """
    logs = self._process_logs(logs)
    for callback in self.callbacks:
      callback.on_predict_end(logs)

  def __iter__(self):
    # [FUNCTION DOC]
    # Returns an iterator over the contained callbacks, in order.
    return iter(self.callbacks)

  def _disallow_batch_hooks_in_ps_strategy(self):
    # [FUNCTION DOC]
    # HELPER: validation step run by the constructor.
    # Looks at the current distribution strategy; if it uses a coordinator (parameter-server
    # style), collects every callback that implements any batch-level hook and does not declare
    # support for raw remote tensor logs. Raises a value error listing those callbacks if any.
    # Does nothing for other strategies.
    """Error out if batch-level callbacks are passed with PSStrategy."""
    # pylint: disable=protected-access
    strategy = distribute_lib.get_strategy()
    if strategy._should_use_with_coordinator:
      unsupported_callbacks = []
      for cb in self.callbacks:
        # These Callbacks can accept RemoteValues directly.
        if getattr(cb, '_supports_tf_logs', False):
          continue
        if (cb._implements_train_batch_hooks() or
            cb._implements_test_batch_hooks() or
            cb._implements_predict_batch_hooks()):
          unsupported_callbacks.append(cb)
      if unsupported_callbacks:
        raise ValueError('Batch-level `Callback`s are not supported with '
                         '`ParameterServerStrategy`. Found unsupported '
                         'callbacks: {}'.format(unsupported_callbacks))
    # pylint: enable=protected-access


class Callback:
  # [CLASS DOC]
  # Abstract base class for all Keras callbacks. Defines no-op hooks for epoch, train, test and
  # predict begin/end and for batch begin/end that subclasses override. Holds the model and the
  # params mapping assigned by the training loop or CallbackList, a validation_data slot, and
  # private flags: whether it only runs on the chief worker in multi-worker training and whether
  # it can receive raw (unconverted) tensor logs. Subclass instances are passed to fit, evaluate
  # or predict, or grouped into a CallbackList for custom loops.
  """Abstract base class used to build new callbacks.

  Callbacks can be passed to keras methods such as `fit`, `evaluate`, and
  `predict` in order to hook into the various stages of the model training and
  inference lifecycle.

  To create a custom callback, subclass `keras.callbacks.Callback` and override
  the method associated with the stage of interest. See
  https://www.tensorflow.org/guide/keras/custom_callback for more information.

  Example:

  >>> training_finished = False
  >>> class MyCallback(tf.keras.callbacks.Callback):
  ...   def on_train_end(self, logs=None):
  ...     global training_finished
  ...     training_finished = True
  >>> model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
  >>> model.compile(loss='mean_squared_error')
  >>> model.fit(tf.constant([[1.0]]), tf.constant([[1.0]]),
  ...           callbacks=[MyCallback()])
  >>> assert training_finished == True

  If you want to use `Callback` objects in a custom training loop:

  1. You should pack all your callbacks into a single `callbacks.CallbackList`
     so they can all be called together.
  2. You will need to manually call all the `on_*` methods at the appropriate
     locations in your loop. Like this:

     ```
     callbacks =  tf.keras.callbacks.CallbackList([...])
     callbacks.append(...)

     callbacks.on_train_begin(...)
     for epoch in range(EPOCHS):
       callbacks.on_epoch_begin(epoch)
       for i, data in dataset.enumerate():
         callbacks.on_train_batch_begin(i)
         batch_logs = model.train_step(data)
         callbacks.on_train_batch_end(i, batch_logs)
       epoch_logs = ...
       callbacks.on_epoch_end(epoch, epoch_logs)
     final_logs=...
     callbacks.on_train_end(final_logs)
     ```

  Attributes:
      params: Dict. Training parameters (eg. verbosity, batch size, number of
        epochs...).
      model: Instance of `keras.models.Model`. Reference of the model being
        trained.

  The `logs` dictionary that callback methods
  take as argument will contain keys for quantities relevant to
  the current batch or epoch (see method-specific docstrings).
  """

  def __init__(self):
    # [FUNCTION DOC]
    # Initializes the callback state: no validation data, no model, the chief-worker-only flag
    # unset (nothing), and the flag saying raw tensor logs are supported set to false. Takes no
    # arguments and returns nothing. Subclasses should call it first in their own constructor.
    self.validation_data = None  # pylint: disable=g-missing-from-attributes
    self.model = None
    # Whether this Callback should only run on the chief worker in a
    # Multi-Worker setting.
    # TODO(omalleyt): Make this attr public once solution is stable.
    self._chief_worker_only = None
    self._supports_tf_logs = False

  def set_params(self, params):
    # [FUNCTION DOC]
    # Stores the training parameters mapping (batch size, epochs, steps, verbosity, metrics...)
    # on the callback. Called by the training loop or the callback container. Returns nothing.
    self.params = params

  def set_model(self, model):
    # [FUNCTION DOC]
    # Stores the model being trained on the callback. Called by the training loop or the callback
    # container. Returns nothing.
    self.model = model

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_batch_begin(self, batch, logs=None):
    # [FUNCTION DOC]
    # Default no-op hook (marked as default so the container can detect it is not overridden);
    # backwards-compatible alias of the train batch begin event. Takes a batch index and an
    # optional logs mapping; does nothing.
    """A backwards compatibility alias for `on_train_batch_begin`."""

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_batch_end(self, batch, logs=None):
    # [FUNCTION DOC]
    # Default no-op hook (marked as default so the container can detect it is not overridden);
    # backwards-compatible alias of the train batch end event. Takes a batch index and an optional
    # logs mapping; does nothing.
    """A backwards compatibility alias for `on_train_batch_end`."""

  @doc_controls.for_subclass_implementers
  def on_epoch_begin(self, epoch, logs=None):
    # [FUNCTION DOC]
    # No-op hook called at the start of each epoch (train mode only). epoch is the integer epoch
    # index; logs is an optional mapping with no data currently. Subclasses override it.
    """Called at the start of an epoch.

    Subclasses should override for any actions to run. This function should only
    be called during TRAIN mode.

    Args:
        epoch: Integer, index of epoch.
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """

  @doc_controls.for_subclass_implementers
  def on_epoch_end(self, epoch, logs=None):
    # [FUNCTION DOC]
    # No-op hook called at the end of each epoch (train mode only). epoch is the integer epoch
    # index; logs is an optional mapping with the epoch's metric results (and val_-prefixed
    # validation results when validation ran). Subclasses override it.
    """Called at the end of an epoch.

    Subclasses should override for any actions to run. This function should only
    be called during TRAIN mode.

    Args:
        epoch: Integer, index of epoch.
        logs: Dict, metric results for this training epoch, and for the
          validation epoch if validation is performed. Validation result keys
          are prefixed with `val_`. For training epoch, the values of the
         `Model`'s metrics are returned. Example : `{'loss': 0.2, 'accuracy':
           0.7}`.
    """

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_train_batch_begin(self, batch, logs=None):
    # [FUNCTION DOC]
    # Default hook called before each training batch (every N batches when steps per execution is
    # N). Takes the batch index and an optional logs mapping. The default implementation simply
    # forwards to the legacy batch-begin hook for backwards compatibility. Returns nothing.
    """Called at the beginning of a training batch in `fit` methods.

    Subclasses should override for any actions to run.

    Note that if the `steps_per_execution` argument to `compile` in
    `tf.keras.Model` is set to `N`, this method will only be called every `N`
    batches.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict, contains the return value of `model.train_step`. Typically,
          the values of the `Model`'s metrics are returned.  Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
    """
    # For backwards compatibility.
    self.on_batch_begin(batch, logs=logs)

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_train_batch_end(self, batch, logs=None):
    # [FUNCTION DOC]
    # Default hook called after each training batch (every N batches when steps per execution is
    # N). Takes the batch index and a logs mapping of aggregated metrics. The default
    # implementation forwards to the legacy batch-end hook for backwards compatibility.
    # Returns nothing.
    """Called at the end of a training batch in `fit` methods.

    Subclasses should override for any actions to run.

    Note that if the `steps_per_execution` argument to `compile` in
    `tf.keras.Model` is set to `N`, this method will only be called every `N`
    batches.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Aggregated metric results up until this batch.
    """
    # For backwards compatibility.
    self.on_batch_end(batch, logs=logs)

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_test_batch_begin(self, batch, logs=None):
    # [FUNCTION DOC]
    # Default no-op hook (marked as default so overriding can be detected) called before each batch
    # of evaluation, including validation batches during fit. Takes the batch index and an
    # optional logs mapping. Returns nothing.
    """Called at the beginning of a batch in `evaluate` methods.

    Also called at the beginning of a validation batch in the `fit`
    methods, if validation data is provided.

    Subclasses should override for any actions to run.

    Note that if the `steps_per_execution` argument to `compile` in
    `tf.keras.Model` is set to `N`, this method will only be called every `N`
    batches.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict, contains the return value of `model.test_step`. Typically,
          the values of the `Model`'s metrics are returned.  Example:
          `{'loss': 0.2, 'accuracy': 0.7}`.
    """

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_test_batch_end(self, batch, logs=None):
    # [FUNCTION DOC]
    # Default no-op hook (marked as default) called after each batch of evaluation, including
    # validation batches during fit. Takes the batch index and a logs mapping of aggregated
    # metrics. Returns nothing.
    """Called at the end of a batch in `evaluate` methods.

    Also called at the end of a validation batch in the `fit`
    methods, if validation data is provided.

    Subclasses should override for any actions to run.

    Note that if the `steps_per_execution` argument to `compile` in
    `tf.keras.Model` is set to `N`, this method will only be called every `N`
    batches.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Aggregated metric results up until this batch.
    """

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_predict_batch_begin(self, batch, logs=None):
    # [FUNCTION DOC]
    # Default no-op hook (marked as default) called before each prediction batch. Takes the batch
    # index and an optional logs mapping. Returns nothing.
    """Called at the beginning of a batch in `predict` methods.

    Subclasses should override for any actions to run.

    Note that if the `steps_per_execution` argument to `compile` in
    `tf.keras.Model` is set to `N`, this method will only be called every `N`
    batches.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict, contains the return value of `model.predict_step`,
          it typically returns a dict with a key 'outputs' containing
          the model's outputs.
    """

  @doc_controls.for_subclass_implementers
  @generic_utils.default
  def on_predict_batch_end(self, batch, logs=None):
    # [FUNCTION DOC]
    # Default no-op hook (marked as default) called after each prediction batch. Takes the batch
    # index and a logs mapping (typically holding the model outputs). Returns nothing.
    """Called at the end of a batch in `predict` methods.

    Subclasses should override for any actions to run.

    Note that if the `steps_per_execution` argument to `compile` in
    `tf.keras.Model` is set to `N`, this method will only be called every `N`
    batches.

    Args:
        batch: Integer, index of batch within the current epoch.
        logs: Dict. Aggregated metric results up until this batch.
    """

  @doc_controls.for_subclass_implementers
  def on_train_begin(self, logs=None):
    # [FUNCTION DOC]
    # No-op hook called once when training starts. logs is an optional mapping (currently empty).
    # Subclasses override it. Returns nothing.
    """Called at the beginning of training.

    Subclasses should override for any actions to run.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """

  @doc_controls.for_subclass_implementers
  def on_train_end(self, logs=None):
    # [FUNCTION DOC]
    # No-op hook called once when training ends. logs is an optional mapping (currently the output
    # of the last epoch-end call). Subclasses override it. Returns nothing.
    """Called at the end of training.

    Subclasses should override for any actions to run.

    Args:
        logs: Dict. Currently the output of the last call to `on_epoch_end()`
          is passed to this argument for this method but that may change in
          the future.
    """

  @doc_controls.for_subclass_implementers
  def on_test_begin(self, logs=None):
    # [FUNCTION DOC]
    # No-op hook called when evaluation or validation starts. logs is an optional mapping
    # (currently empty). Subclasses override it. Returns nothing.
    """Called at the beginning of evaluation or validation.

    Subclasses should override for any actions to run.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """

  @doc_controls.for_subclass_implementers
  def on_test_end(self, logs=None):
    # [FUNCTION DOC]
    # No-op hook called when evaluation or validation ends. logs is an optional mapping (currently
    # the output of the last test batch end call). Subclasses override it. Returns nothing.
    """Called at the end of evaluation or validation.

    Subclasses should override for any actions to run.

    Args:
        logs: Dict. Currently the output of the last call to
          `on_test_batch_end()` is passed to this argument for this method
          but that may change in the future.
    """

  @doc_controls.for_subclass_implementers
  def on_predict_begin(self, logs=None):
    # [FUNCTION DOC]
    # No-op hook called when prediction starts. logs is an optional mapping (currently empty).
    # Subclasses override it. Returns nothing.
    """Called at the beginning of prediction.

    Subclasses should override for any actions to run.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """

  @doc_controls.for_subclass_implementers
  def on_predict_end(self, logs=None):
    # [FUNCTION DOC]
    # No-op hook called when prediction ends. logs is an optional mapping (currently empty).
    # Subclasses override it. Returns nothing.
    """Called at the end of prediction.

    Subclasses should override for any actions to run.

    Args:
        logs: Dict. Currently no data is passed to this argument for this method
          but that may change in the future.
    """

  def _implements_train_batch_hooks(self):
    # [FUNCTION DOC]
    # HELPER: lets the callback container decide whether per-batch train hooks must be called.
    # Returns true if any of the four train-batch hooks (legacy and new begin/end) has been
    # overridden, i.e. is no longer the default implementation; false otherwise.
    """Determines if this Callback should be called for each train batch."""
    return (not generic_utils.is_default(self.on_batch_begin) or
            not generic_utils.is_default(self.on_batch_end) or
            not generic_utils.is_default(self.on_train_batch_begin) or
            not generic_utils.is_default(self.on_train_batch_end))

  def _implements_test_batch_hooks(self):
    # [FUNCTION DOC]
    # HELPER: lets the callback container decide whether per-batch test hooks must be called.
    # Returns true if either test-batch hook was overridden (is not the default); false otherwise.
    """Determines if this Callback should be called for each test batch."""
    return (not generic_utils.is_default(self.on_test_batch_begin) or
            not generic_utils.is_default(self.on_test_batch_end))

  def _implements_predict_batch_hooks(self):
    # [FUNCTION DOC]
    # HELPER: lets the callback container decide whether per-batch predict hooks must be called.
    # Returns true if either predict-batch hook was overridden (is not the default); false
    # otherwise.
    """Determines if this Callback should be called for each predict batch."""
    return (not generic_utils.is_default(self.on_predict_batch_begin) or
            not generic_utils.is_default(self.on_predict_batch_end))


class BaseLogger(Callback):
  # [CLASS DOC]
  # Callback automatically added to every model in train mode. During each epoch it accumulates
  # batch-size-weighted sums of the batch metrics (stateful metrics are just overwritten with the
  # latest value) and at epoch end writes the epoch averages into the logs mapping so later
  # callbacks see them. State: the set of stateful metric names, the seen-sample count and the
  # per-metric totals.
  """Callback that accumulates epoch averages of metrics.

  This callback is automatically applied to every Keras model.

  Args:
      stateful_metrics: Iterable of string names of metrics that
          should *not* be averaged over an epoch.
          Metrics in this list will be logged as-is in `on_epoch_end`.
          All others will be averaged in `on_epoch_end`.
  """

  def __init__(self, stateful_metrics=None):
    # [FUNCTION DOC]
    # Calls the base constructor and stores the optional iterable of metric names that must not
    # be averaged, converted to a set (empty set when nothing is given). Returns nothing.
    super(BaseLogger, self).__init__()
    self.stateful_metrics = set(stateful_metrics or [])

  def on_epoch_begin(self, epoch, logs=None):
    # [FUNCTION DOC]
    # Resets the seen-sample counter to zero and clears the per-metric totals at the start of each
    # epoch. Arguments epoch and logs are unused. Returns nothing.
    self.seen = 0
    self.totals = {}

  def on_batch_end(self, batch, logs=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: every logged key is accumulated, including the size and step-count entries
    # themselves (no key is excluded), and when the batch size is absent it defaults to zero so
    # non-stateful metrics are weighted by zero and the epoch average would divide by zero.
    # Accumulates the batch's metrics. Reads the batch size (default zero) and number of steps
    # (default one) from the logs, adds their product to the seen count, and for every logged
    # value either stores it as is (stateful metrics) or adds the value times batch size to the
    # running total. The batch index is unused; nothing is returned.
    logs = logs or {}
    batch_size = logs.get('size', 0)
    # In case of distribution strategy we can potentially run multiple steps
    # at the same time, we should account for that in the `seen` calculation.
    num_steps = logs.get('num_steps', 1)
    self.seen += batch_size * num_steps

    for k, v in logs.items():
      if k in self.stateful_metrics:
        self.totals[k] = v
      else:
        if k in self.totals:
          self.totals[k] += v * batch_size
        else:
          self.totals[k] = v * batch_size

  def on_epoch_end(self, epoch, logs=None):
    # [FUNCTION DOC]
    # Writes the epoch results into the logs mapping (mutated in place so later callbacks see
    # them): for each metric name listed in the run params that has accumulated totals, stores the
    # raw total for stateful metrics or the total divided by the seen sample count otherwise.
    # Does nothing when logs is nothing. Epoch index unused. Returns nothing. Fails if the params
    # were never set or no samples were seen (division by zero).
    if logs is not None:
      for k in self.params['metrics']:
        if k in self.totals:
          # Make value available to next callbacks.
          if k in self.stateful_metrics:
            logs[k] = self.totals[k]
          else:
            logs[k] = self.totals[k] / self.seen


class TerminateOnNaN(Callback):
  # [CLASS DOC]
  # Callback that watches the loss after every batch and requests the training loop to stop
  # (by setting the model's stop-training flag) when the loss is not finite. Declares that it can
  # receive raw tensor logs and converts the loss itself.
  """Callback that terminates training when a NaN loss is encountered.
  """

  def __init__(self):
    # [FUNCTION DOC]
    # Calls the base constructor and marks the callback as able to receive raw (unconverted)
    # tensor logs. Takes no arguments, returns nothing.
    super(TerminateOnNaN, self).__init__()
    self._supports_tf_logs = True

  def on_batch_end(self, batch, logs=None):
    # [FUNCTION DOC]
    # Looks up the loss in the logs (nothing logged means nothing happens), converts it to a
    # number, and if it is NaN or infinite prints a message naming the batch to standard output
    # and sets the model's stop-training flag to true. Returns nothing.
    logs = logs or {}
    loss = logs.get('loss')
    if loss is not None:
      loss = tf_utils.sync_to_numpy_or_python_type(loss)
      if np.isnan(loss) or np.isinf(loss):
        print('Batch %d: Invalid loss, terminating training' % (batch))
        self.model.stop_training = True


class ProgbarLogger(Callback):
  # [CLASS DOC]
  # Callback that renders a console progress bar with metric values for fit, evaluate and
  # predict. State: whether progress counts steps or samples, the target total (taken from the
  # params or inferred after the first epoch), seen count, verbosity, epoch count, the progress
  # bar object, references to the model's step counters used to infer the target, a flag that
  # batch hooks are needed, and a flag that it is running inside fit (so validation stays silent).
  # Prints to standard output through the progress bar utility.
  """Callback that prints metrics to stdout.

  Args:
      count_mode: One of `"steps"` or `"samples"`.
          Whether the progress bar should
          count samples seen or steps (batches) seen.
      stateful_metrics: Iterable of string names of metrics that
          should *not* be averaged over an epoch.
          Metrics in this list will be logged as-is.
          All others will be averaged over time (e.g. loss, etc).
          If not provided, defaults to the `Model`'s metrics.

  Raises:
      ValueError: In case of invalid `count_mode`.
  """

  def __init__(self, count_mode='samples', stateful_metrics=None):
    # [FUNCTION DOC]
    # Sets up the progress-bar callback. count_mode must be 'samples' (default) or 'steps' and
    # chooses whether progress counts samples or steps; any other value raises a value error.
    # stateful_metrics is an optional iterable of metric names not to average (stored as a set,
    # empty by default). Initializes seen count, progress bar, target, verbosity (1), epochs (1),
    # step counter references and the in-fit flag; declares support for raw tensor logs.
    super(ProgbarLogger, self).__init__()
    self._supports_tf_logs = True
    if count_mode == 'samples':
      self.use_steps = False
    elif count_mode == 'steps':
      self.use_steps = True
    else:
      raise ValueError('Unknown `count_mode`: ' + str(count_mode))
    # Defaults to all Model's metrics except for loss.
    self.stateful_metrics = set(stateful_metrics) if stateful_metrics else set()

    self.seen = 0
    self.progbar = None
    self.target = None
    self.verbose = 1
    self.epochs = 1

    self._train_step, self._test_step, self._predict_step = None, None, None
    self._call_batch_hooks = True

    self._called_in_fit = False

  def set_params(self, params):
    # [FUNCTION DOC]
    # Reads verbosity and epoch count from the params mapping (both required; a missing key raises
    # a lookup error) and picks the progress target: the steps entry when counting steps, the
    # samples entry when counting samples, otherwise nothing (inferred after the first epoch).
    # Batch hooks are enabled only for verbosity one. When the target is unknown it grabs the
    # model's train/test/predict step counters, and falls back to always calling batch hooks if
    # the model lacks them. Returns nothing.
    self.verbose = params['verbose']
    self.epochs = params['epochs']
    if self.use_steps and 'steps' in params:
      self.target = params['steps']
    elif not self.use_steps and 'samples' in params:
      self.target = params['samples']
    else:
      self.target = None  # Will be inferred at the end of the first epoch.

    self._call_batch_hooks = self.verbose == 1
    if self.target is None:
      try:
        self._train_step = self.model._train_counter  # pylint: disable=protected-access
        self._test_step = self.model._test_counter  # pylint: disable=protected-access
        self._predict_step = self.model._predict_counter  # pylint: disable=protected-access
      except AttributeError:
        self._call_batch_hooks = True

  def on_train_begin(self, logs=None):
    # [FUNCTION DOC]
    # Marks that the logger is running inside fit so that validation (test) phases stay silent.
    # Returns nothing.
    # When this logger is called inside `fit`, validation is silent.
    self._called_in_fit = True

  def on_test_begin(self, logs=None):
    # [FUNCTION DOC]
    # When not running inside fit (standalone evaluate), resets the progress state and creates the
    # progress bar. Does nothing during validation inside fit. Returns nothing.
    if not self._called_in_fit:
      self._reset_progbar()
      self._maybe_init_progbar()

  def on_predict_begin(self, logs=None):
    # [FUNCTION DOC]
    # Resets the progress state and creates the progress bar at the start of prediction.
    # Returns nothing.
    self._reset_progbar()
    self._maybe_init_progbar()

  def on_epoch_begin(self, epoch, logs=None):
    # [FUNCTION DOC]
    # Resets the progress state, creates a new progress bar and, when verbose and there is more
    # than one epoch, prints a header line with the one-based epoch number and total epochs to
    # standard output. Returns nothing.
    self._reset_progbar()
    self._maybe_init_progbar()
    if self.verbose and self.epochs > 1:
      print('Epoch %d/%d' % (epoch + 1, self.epochs))

  def on_train_batch_end(self, batch, logs=None):
    # [FUNCTION DOC]
    # Updates the progress bar after a training batch using the batch index and logs.
    # Returns nothing.
    self._batch_update_progbar(batch, logs)

  def on_test_batch_end(self, batch, logs=None):
    # [FUNCTION DOC]
    # Updates the progress bar after an evaluation batch, unless running inside fit (validation is
    # silent then). Takes the batch index and logs. Returns nothing.
    if not self._called_in_fit:
      self._batch_update_progbar(batch, logs)

  def on_predict_batch_end(self, batch, logs=None):
    # [FUNCTION DOC]
    # Updates the progress bar after a prediction batch, deliberately passing no logs so the
    # prediction outputs are not displayed. Returns nothing.
    # Don't pass prediction results.
    self._batch_update_progbar(batch, None)

  def on_epoch_end(self, epoch, logs=None):
    # [FUNCTION DOC]
    # Completes the progress bar for the epoch with the epoch logs, using the model's train step
    # counter to infer the target when it was unknown. Returns nothing.
    self._finalize_progbar(logs, self._train_step)

  def on_test_end(self, logs=None):
    # [FUNCTION DOC]
    # Completes the progress bar at the end of a standalone evaluation (using the test step
    # counter to infer the target); does nothing inside fit. Returns nothing.
    if not self._called_in_fit:
      self._finalize_progbar(logs, self._test_step)

  def on_predict_end(self, logs=None):
    # [FUNCTION DOC]
    # Completes the progress bar at the end of prediction (using the predict step counter to infer
    # the target). Returns nothing.
    self._finalize_progbar(logs, self._predict_step)

  def _reset_progbar(self):
    # [FUNCTION DOC]
    # HELPER: clears the seen counter to zero and drops the current progress bar object so that a
    # fresh one is created next time. Returns nothing.
    self.seen = 0
    self.progbar = None

  def _maybe_init_progbar(self):
    # [FUNCTION DOC]
    # HELPER: lazily prepares the progress bar. Converts the stateful metrics to a set, adds the
    # names of all of the model's current metrics (if a model is attached), creates the progress
    # bar (with the target, verbosity, stateful metrics and a step or sample unit name) if it does
    # not exist, and pushes the updated stateful metric names into the bar. Returns nothing.
    """Instantiate a `Progbar` if not yet, and update the stateful metrics."""
    # TODO(rchao): Legacy TF1 code path may use list for
    # `self.stateful_metrics`. Remove "cast to set" when TF1 support is dropped.
    self.stateful_metrics = set(self.stateful_metrics)

    if self.model:
      # Update the existing stateful metrics as `self.model.metrics` may contain
      # updated metrics after `MetricsContainer` is built in the first train
      # step.
      self.stateful_metrics = self.stateful_metrics.union(
          set(m.name for m in self.model.metrics))

    if self.progbar is None:
      self.progbar = Progbar(
          target=self.target,
          verbose=self.verbose,
          stateful_metrics=self.stateful_metrics,
          unit_name='step' if self.use_steps else 'sample')

    self.progbar._update_stateful_metrics(self.stateful_metrics)  # pylint: disable=protected-access

  def _implements_train_batch_hooks(self):
    # [FUNCTION DOC]
    # Overrides the base check: reports whether train batch hooks are needed, which is the
    # call-batch-hooks flag (on only for verbosity one or when step counters are unavailable).
    return self._call_batch_hooks

  def _implements_test_batch_hooks(self):
    # [FUNCTION DOC]
    # Overrides the base check: reports whether test batch hooks are needed, which is the
    # call-batch-hooks flag.
    return self._call_batch_hooks

  def _implements_predict_batch_hooks(self):
    # [FUNCTION DOC]
    # Overrides the base check: reports whether predict batch hooks are needed, which is the
    # call-batch-hooks flag.
    return self._call_batch_hooks

  def _batch_update_progbar(self, batch, logs=None):
    # [FUNCTION DOC]
    # HELPER: refreshes the progress bar after a batch. Ensures the bar exists. When counting
    # steps, the seen count becomes the one-based batch index; when counting samples (old graph
    # path only) it removes size, step-count and batch entries from a copy of the logs and adds
    # their product to the seen count. Only when verbosity is one, converts the logs to numpy or
    # Python values (which blocks on async execution) and updates the bar without finalizing it.
    # Inputs: batch index, optional logs mapping. Returns nothing.
    """Updates the progbar."""
    logs = logs or {}
    self._maybe_init_progbar()
    if self.use_steps:
      self.seen = batch + 1  # One-indexed.
    else:
      # v1 path only.
      logs = copy.copy(logs)
      batch_size = logs.pop('size', 0)
      num_steps = logs.pop('num_steps', 1)
      logs.pop('batch', None)
      add_seen = num_steps * batch_size
      self.seen += add_seen

    if self.verbose == 1:
      # Only block async when verbose = 1.
      logs = tf_utils.sync_to_numpy_or_python_type(logs)
      self.progbar.update(self.seen, list(logs.items()), finalize=False)

  def _finalize_progbar(self, logs, counter):
    # [FUNCTION DOC]
    # HELPER: draws the final state of the progress bar. Converts logs (nothing becomes empty) to
    # plain values. If the target total was unknown, derives it from the model's step counter
    # (read as a number, multiplied by the logged size when counting samples) or else from the
    # seen count, and sets it on the bar. Then updates the bar to the target with finalization.
    # Inputs: logs mapping or nothing, counter (step counter variable or nothing). Returns nothing.
    logs = tf_utils.sync_to_numpy_or_python_type(logs or {})
    if self.target is None:
      if counter is not None:
        counter = counter.numpy()
        if not self.use_steps:
          counter *= logs.get('size', 1)
      self.target = counter or self.seen
      self.progbar.target = self.target
    self.progbar.update(self.target, list(logs.items()), finalize=True)


class History(Callback):
  # [CLASS DOC]
  # Callback automatically applied to every model; it records per-epoch logs. State: a mapping
  # from metric name to the list of per-epoch values (history), and the list of epoch indices
  # (epoch, created when training begins). The instance is what fit returns to the caller, and it
  # also keeps the params assigned by the training loop.
  """Callback that records events into a `History` object.

  This callback is automatically applied to
  every Keras model. The `History` object
  gets returned by the `fit` method of models.

  Example:

  >>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
  >>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
  >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
  ...                     epochs=10, verbose=1)
  >>> print(history.params)
  {'verbose': 1, 'epochs': 10, 'steps': 1}
  >>> # check the keys of history object
  >>> print(history.history.keys())
  dict_keys(['loss'])

  """

  def __init__(self):
    # [FUNCTION DOC]
    # Calls the base constructor and creates the empty mapping that will hold metric histories.
    # Takes no arguments, returns nothing.
    super(History, self).__init__()
    self.history = {}

  def on_train_begin(self, logs=None):
    # [FUNCTION DOC]
    # Starts a new list of recorded epoch indices when training begins. The recorded metric
    # history is not cleared, so repeated fit calls with the same object keep appending.
    # Returns nothing.
    self.epoch = []

  def on_epoch_end(self, epoch, logs=None):
    # [FUNCTION DOC]
    # Records the epoch index and appends every logged metric value to that metric's list in the
    # history mapping (creating the list on first use). Finally sets this object as the model's
    # history attribute so the model holds the latest state. Returns nothing.
    logs = logs or {}
    self.epoch.append(epoch)
    for k, v in logs.items():
      self.history.setdefault(k, []).append(v)

    # Set the history attribute on the model after the epoch ends. This will
    # make sure that the state which is set is the latest one.
    self.model.history = self


class ModelCheckpoint(Callback):
  # [CLASS DOC]
  # Callback that saves the whole model or only its weights during fit, either at the end of
  # every epoch or after a fixed number of batches, optionally only when the monitored metric
  # improves. State: the file path template (may contain epoch and log-key placeholders), monitored
  # metric name, best value so far and the comparison rule, verbosity, save options, counters of
  # epochs and batches since the last save, and deprecated settings (period, load-on-restart).
  # In multi-worker training non-chief workers save to temporary locations that are cleaned up.
  """Callback to save the Keras model or model weights at some frequency.

  `ModelCheckpoint` callback is used in conjunction with training using
  `model.fit()` to save a model or weights (in a checkpoint file) at some
  interval, so the model or weights can be loaded later to continue the training
  from the state saved.

  A few options this callback provides include:

  - Whether to only keep the model that has achieved the "best performance" so
    far, or whether to save the model at the end of every epoch regardless of
    performance.
  - Definition of 'best'; which quantity to monitor and whether it should be
    maximized or minimized.
  - The frequency it should save at. Currently, the callback supports saving at
    the end of every epoch, or after a fixed number of training batches.
  - Whether only weights are saved, or the whole model is saved.

  Note: If you get `WARNING:tensorflow:Can save best model only with <name>
  available, skipping` see the description of the `monitor` argument for
  details on how to get this right.

  Example:

  ```python
  model.compile(loss=..., optimizer=...,
                metrics=['accuracy'])

  EPOCHS = 10
  checkpoint_filepath = '/tmp/checkpoint'
  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
      filepath=checkpoint_filepath,
      save_weights_only=True,
      monitor='val_accuracy',
      mode='max',
      save_best_only=True)

  # Model weights are saved at the end of every epoch, if it's the best seen
  # so far.
  model.fit(epochs=EPOCHS, callbacks=[model_checkpoint_callback])

  # The model weights (that are considered the best) are loaded into the model.
  model.load_weights(checkpoint_filepath)
  ```

  Args:
      filepath: string or `PathLike`, path to save the model file. e.g.
        filepath = os.path.join(working_dir, 'ckpt', file_name). `filepath`
        can contain named formatting options, which will be filled the value of
        `epoch` and keys in `logs` (passed in `on_epoch_end`). For example: if
        `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`, then the model
        checkpoints will be saved with the epoch number and the validation loss
        in the filename. The directory of the filepath should not be reused by
        any other callbacks to avoid conflicts.
      monitor: The metric name to monitor. Typically the metrics are set by the
        `Model.compile` method. Note:

        * Prefix the name with `"val_`" to monitor validation metrics.
        * Use `"loss"` or "`val_loss`" to monitor the model's total loss.
        * If you specify metrics as strings, like `"accuracy"`, pass the same
          string (with or without the `"val_"` prefix).
        * If you pass `metrics.Metric` objects, `monitor` should be set to
          `metric.name`
        * If you're not sure about the metric names you can check the contents
          of the `history.history` dictionary returned by
          `history = model.fit()`
        * Multi-output models set additional prefixes on the metric names.

      verbose: verbosity mode, 0 or 1.
      save_best_only: if `save_best_only=True`, it only saves when the model
        is considered the "best" and the latest best model according to the
        quantity monitored will not be overwritten. If `filepath` doesn't
        contain formatting options like `{epoch}` then `filepath` will be
        overwritten by each new better model.
      mode: one of {'auto', 'min', 'max'}. If `save_best_only=True`, the
        decision to overwrite the current save file is made based on either
        the maximization or the minimization of the monitored quantity.
        For `val_acc`, this should be `max`, for `val_loss` this should be
        `min`, etc. In `auto` mode, the mode is set to `max` if the quantities
        monitored are 'acc' or start with 'fmeasure' and are set to `min` for
        the rest of the quantities.
      save_weights_only: if True, then only the model's weights will be saved
        (`model.save_weights(filepath)`), else the full model is saved
        (`model.save(filepath)`).
      save_freq: `'epoch'` or integer. When using `'epoch'`, the callback saves
        the model after each epoch. When using integer, the callback saves the
        model at end of this many batches. If the `Model` is compiled with
        `steps_per_execution=N`, then the saving criteria will be
        checked every Nth batch. Note that if the saving isn't aligned to
        epochs, the monitored metric may potentially be less reliable (it
        could reflect as little as 1 batch, since the metrics get reset every
        epoch). Defaults to `'epoch'`.
      options: Optional `tf.train.CheckpointOptions` object if
        `save_weights_only` is true or optional `tf.saved_model.SaveOptions`
        object if `save_weights_only` is false.
      **kwargs: Additional arguments for backwards compatibility. Possible key
        is `period`.
  """

  def __init__(self,
               filepath,
               monitor='val_loss',
               verbose=0,
               save_best_only=False,
               save_weights_only=False,
               mode='auto',
               save_freq='epoch',
               options=None,
               **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the type error message raised for invalid save options when saving the full
    # model is missing a space between "must be" and "either" (the two string pieces are joined
    # without a separator).
    # Configures the checkpoint callback. filepath (string or path-like object) is the target,
    # possibly with epoch and log-key placeholders; monitor is the metric name (default
    # validation loss); verbose is 0 or 1; save_best_only saves only on improvement;
    # save_weights_only saves just weights; mode is auto, min or max (an unknown mode logs a
    # warning and falls back to auto; auto picks max for names containing acc or starting with
    # fmeasure, min otherwise); save_freq is the string epoch or an integer batch count (anything
    # else raises a value error); options must be checkpoint options when saving weights only or
    # saved-model save options otherwise, else a type error is raised. Optional keyword arguments
    # load_weights_on_restart and period are deprecated and each logs a warning. Initializes the
    # best value and comparison function, counters, declares raw tensor log support, and sets the
    # chief-only flag false so all workers restore at train begin.
    super(ModelCheckpoint, self).__init__()
    self._supports_tf_logs = True
    self.monitor = monitor
    self.verbose = verbose
    self.filepath = path_to_string(filepath)
    self.save_best_only = save_best_only
    self.save_weights_only = save_weights_only
    self.save_freq = save_freq
    self.epochs_since_last_save = 0
    self._batches_seen_since_last_saving = 0
    self._last_batch_seen = 0

    if save_weights_only:
      if options is None or isinstance(
          options, checkpoint_options_lib.CheckpointOptions):
        self._options = options or checkpoint_options_lib.CheckpointOptions()
      else:
        raise TypeError('If save_weights_only is True, then `options` must be '
                        'either None or a tf.train.CheckpointOptions')
    else:
      if options is None or isinstance(options, save_options_lib.SaveOptions):
        self._options = options or save_options_lib.SaveOptions()
      else:
        raise TypeError('If save_weights_only is False, then `options` must be'
                        'either None or a tf.saved_model.SaveOptions')

    # Deprecated field `load_weights_on_restart` is for loading the checkpoint
    # file from `filepath` at the start of `model.fit()`
    # TODO(rchao): Remove the arg during next breaking release.
    if 'load_weights_on_restart' in kwargs:
      self.load_weights_on_restart = kwargs['load_weights_on_restart']
      logging.warning('`load_weights_on_restart` argument is deprecated. '
                      'Please use `model.load_weights()` for loading weights '
                      'before the start of `model.fit()`.')
    else:
      self.load_weights_on_restart = False

    # Deprecated field `period` is for the number of epochs between which
    # the model is saved.
    if 'period' in kwargs:
      self.period = kwargs['period']
      logging.warning('`period` argument is deprecated. Please use `save_freq` '
                      'to specify the frequency in number of batches seen.')
    else:
      self.period = 1

    if mode not in ['auto', 'min', 'max']:
      logging.warning('ModelCheckpoint mode %s is unknown, '
                      'fallback to auto mode.', mode)
      mode = 'auto'

    if mode == 'min':
      self.monitor_op = np.less
      self.best = np.inf
    elif mode == 'max':
      self.monitor_op = np.greater
      self.best = -np.inf
    else:
      if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
        self.monitor_op = np.greater
        self.best = -np.inf
      else:
        self.monitor_op = np.less
        self.best = np.inf

    if self.save_freq != 'epoch' and not isinstance(self.save_freq, int):
      raise ValueError('Unrecognized save_freq: {}'.format(self.save_freq))

    # Only the chief worker writes model checkpoints, but all workers
    # restore checkpoint at on_train_begin().
    self._chief_worker_only = False

  def on_train_begin(self, logs=None):
    # [FUNCTION DOC]
    # Only does something when the deprecated load-weights-on-restart option is on: finds the most
    # recently modified existing file matching the path pattern and loads its weights into the
    # model. Raises a value error if loading fails with an I/O or value error. Returns nothing.
    if self.load_weights_on_restart:
      filepath_to_load = (
          self._get_most_recently_modified_file_matching_pattern(self.filepath))
      if (filepath_to_load is not None and
          self._checkpoint_exists(filepath_to_load)):
        try:
          # `filepath` may contain placeholders such as `{epoch:02d}`, and
          # thus it attempts to load the most recently modified file with file
          # name matching the pattern.
          self.model.load_weights(filepath_to_load)
        except (IOError, ValueError) as e:
          raise ValueError('Error loading file from {}. Reason: {}'.format(
              filepath_to_load, e))

  def _implements_train_batch_hooks(self):
    # [FUNCTION DOC]
    # Overrides the base check: batch hooks are needed only when saving by batch count, i.e. when
    # the save frequency is not the epoch setting.
    # Only call batch hooks when saving on batch
    return self.save_freq != 'epoch'

  def on_train_batch_end(self, batch, logs=None):
    # [FUNCTION DOC]
    # After a training batch, saves the model (for the current epoch, with the given logs) if the
    # batch-count save frequency has been reached. Returns nothing.
    if self._should_save_on_batch(batch):
      self._save_model(epoch=self._current_epoch, logs=logs)

  def on_epoch_begin(self, epoch, logs=None):
    # [FUNCTION DOC]
    # Remembers the current epoch index so batch-level saves can use it. Returns nothing.
    self._current_epoch = epoch

  def on_epoch_end(self, epoch, logs=None):
    # [FUNCTION DOC]
    # Increments the count of epochs since the last save and, when saving per epoch, saves the
    # model with the epoch index and logs (subject to the period and best-only rules).
    # Returns nothing.
    self.epochs_since_last_save += 1
    # pylint: disable=protected-access
    if self.save_freq == 'epoch':
      self._save_model(epoch=epoch, logs=logs)

  def _should_save_on_batch(self, batch):
    # [FUNCTION DOC]
    # HELPER: batch-level save decision. Returns false when saving per epoch. Otherwise updates
    # the running count of batches seen since the last save (a batch index not larger than the
    # last seen one means a new epoch started, so it counts from the start; this also copes with
    # several steps per execution), remembers the batch index, and returns true (resetting the
    # count) once the count reaches the save frequency; else false.
    """Handles batch-level saving logic, supports steps_per_execution."""
    if self.save_freq == 'epoch':
      return False

    if batch <= self._last_batch_seen:  # New epoch.
      add_batches = batch + 1  # batches are zero-indexed.
    else:
      add_batches = batch - self._last_batch_seen
    self._batches_seen_since_last_saving += add_batches
    self._last_batch_seen = batch

    if self._batches_seen_since_last_saving >= self.save_freq:
      self._batches_seen_since_last_saving = 0
      return True
    return False

  def _save_model(self, epoch, logs):
    # [FUNCTION DOC]
    # HELPER: performs the actual checkpoint write. Proceeds only when the save frequency is a
    # batch count or enough epochs (the period) have passed; then converts logs to plain values,
    # resets the epoch counter and computes the target path from the epoch and logs. With
    # save-best-only it reads the monitored value from logs (logs a warning and skips if missing),
    # and if it beats the best so far records it, optionally prints a message, and saves weights or
    # the whole model with the configured options (overwriting); otherwise optionally prints a
    # no-improvement message. Without save-best-only it always saves. Afterwards removes any
    # temporary directory used by non-chief workers. Raises an I/O error when the path is an
    # existing directory, and re-raises other I/O errors. Side effects: writes files, prints.
    """Saves the model.

    Args:
        epoch: the epoch this iteration is in.
        logs: the `logs` dict passed in to `on_batch_end` or `on_epoch_end`.
    """
    logs = logs or {}

    if isinstance(self.save_freq,
                  int) or self.epochs_since_last_save >= self.period:
      # Block only when saving interval is reached.
      logs = tf_utils.sync_to_numpy_or_python_type(logs)
      self.epochs_since_last_save = 0
      filepath = self._get_file_path(epoch, logs)

      try:
        if self.save_best_only:
          current = logs.get(self.monitor)
          if current is None:
            logging.warning('Can save best model only with %s available, '
                            'skipping.', self.monitor)
          else:
            if self.monitor_op(current, self.best):
              if self.verbose > 0:
                print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                      ' saving model to %s' % (epoch + 1, self.monitor,
                                               self.best, current, filepath))
              self.best = current
              if self.save_weights_only:
                self.model.save_weights(
                    filepath, overwrite=True, options=self._options)
              else:
                self.model.save(filepath, overwrite=True, options=self._options)
            else:
              if self.verbose > 0:
                print('\nEpoch %05d: %s did not improve from %0.5f' %
                      (epoch + 1, self.monitor, self.best))
        else:
          if self.verbose > 0:
            print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
          if self.save_weights_only:
            self.model.save_weights(
                filepath, overwrite=True, options=self._options)
          else:
            self.model.save(filepath, overwrite=True, options=self._options)

        self._maybe_remove_file()
      except IsADirectoryError as e:  # h5py 3.x
        raise IOError('Please specify a non-directory filepath for '
                      'ModelCheckpoint. Filepath used is an existing '
                      'directory: {}'.format(filepath))
      except IOError as e:  # h5py 2.x
        # `e.errno` appears to be `None` so checking the content of `e.args[0]`.
        if 'is a directory' in str(e.args[0]).lower():
          raise IOError('Please specify a non-directory filepath for '
                        'ModelCheckpoint. Filepath used is an existing '
                        'directory: {}'.format(filepath))
        # Re-throw the error for any other causes.
        raise e

  def _get_file_path(self, epoch, logs):
    # [FUNCTION DOC]
    # HELPER: builds the path to write. Fills the path template with the one-based epoch number
    # and the entries of logs; a placeholder with no matching log key raises a key error with a
    # descriptive message. Then maps the path to the location this worker should write (a
    # temporary location on non-chief workers), remembers it as the write path and returns it.
    """Returns the file path for checkpoint."""
    # pylint: disable=protected-access
    try:
      # `filepath` may contain placeholders such as `{epoch:02d}` and
      # `{mape:.2f}`. A mismatch between logged metrics and the path's
      # placeholders can cause formatting to fail.
      file_path = self.filepath.format(epoch=epoch + 1, **logs)
    except KeyError as e:
      raise KeyError('Failed to format this callback filepath: "{}". '
                     'Reason: {}'.format(self.filepath, e))
    self._write_filepath = distributed_file_utils.write_filepath(
        file_path, self.model.distribute_strategy)
    return self._write_filepath

  def _maybe_remove_file(self):
    # [FUNCTION DOC]
    # HELPER: after a save, deletes the temporary checkpoint directory that non-chief workers
    # wrote in multi-worker training (does nothing for the chief). Uses the last write path and
    # the model's distribution strategy. Returns nothing.
    # Remove the checkpoint directory in multi-worker training where this worker
    # should not checkpoint. It is a dummy directory previously saved for sync
    # distributed training.
    distributed_file_utils.remove_temp_dir_with_filepath(
        self._write_filepath, self.model.distribute_strategy)

  def _checkpoint_exists(self, filepath):
    # [FUNCTION DOC]
    # HELPER: tells whether a checkpoint exists at the path. For a path ending in the h5
    # extension it checks that file; otherwise true if the path itself (saved-model directory) or
    # the path with the index suffix (weights-only checkpoint) exists. Returns a boolean.
    """Returns whether the checkpoint `filepath` refers to exists."""
    if filepath.endswith('.h5'):
      return file_io.file_exists_v2(filepath)
    tf_saved_model_exists = file_io.file_exists_v2(filepath)
    tf_weights_only_checkpoint_exists = file_io.file_exists_v2(
        filepath + '.index')
    return tf_saved_model_exists or tf_weights_only_checkpoint_exists

  def _get_most_recently_modified_file_matching_pattern(self, pattern):
    # [FUNCTION DOC]
    # HELPER: finds the checkpoint to restore. Turns the file name part of the pattern into a
    # regular expression (placeholders become wildcards). First tries the latest checkpoint
    # recorded in the directory and returns it if its name matches. Otherwise lists the directory
    # (if it exists), finds the matching file with the latest modification time, breaking ties
    # (or the no-match-with-time case) by the lexicographically largest path, and returns it.
    # Returns nothing/null when no file matches. Input: path pattern string.
    """Returns the most recently modified filepath matching pattern.

    Pattern may contain python formatting placeholder. If
    `tf.train.latest_checkpoint()` does not return None, use that; otherwise,
    check for most recently modified one that matches the pattern.

    In the rare case where there are more than one pattern-matching file having
    the same modified time that is most recent among all, return the filepath
    that is largest (by `>` operator, lexicographically using the numeric
    equivalents). This provides a tie-breaker when multiple files are most
    recent. Note that a larger `filepath` can sometimes indicate a later time of
    modification (for instance, when epoch/batch is used as formatting option),
    but not necessarily (when accuracy or loss is used). The tie-breaker is
    put in the logic as best effort to return the most recent, and to avoid
    undeterministic result.

    Modified time of a file is obtained with `os.path.getmtime()`.

    This utility function is best demonstrated via an example:

    ```python
    file_pattern = 'f.batch{batch:02d}epoch{epoch:02d}.h5'
    test_dir = self.get_temp_dir()
    path_pattern = os.path.join(test_dir, file_pattern)
    file_paths = [
        os.path.join(test_dir, file_name) for file_name in
        ['f.batch03epoch02.h5', 'f.batch02epoch02.h5', 'f.batch01epoch01.h5']
    ]
    for file_path in file_paths:
      # Write something to each of the files
    self.assertEqual(
        _get_most_recently_modified_file_matching_pattern(path_pattern),
        file_paths[-1])
    ```

    Args:
        pattern: The file pattern that may optionally contain python placeholder
            such as `{epoch:02d}`.

    Returns:
        The most recently modified file's full filepath matching `pattern`. If
        `pattern` does not contain any placeholder, this returns the filepath
        that
        exactly matches `pattern`. Returns `None` if no match is found.
    """
    dir_name = os.path.dirname(pattern)
    base_name = os.path.basename(pattern)
    base_name_regex = '^' + re.sub(r'{.*}', r'.*', base_name) + '$'

    # If tf.train.latest_checkpoint tells us there exists a latest checkpoint,
    # use that as it is more robust than `os.path.getmtime()`.
    latest_tf_checkpoint = checkpoint_management.latest_checkpoint(dir_name)
    if latest_tf_checkpoint is not None and re.match(
        base_name_regex, os.path.basename(latest_tf_checkpoint)):
      return latest_tf_checkpoint

    latest_mod_time = 0
    file_path_with_latest_mod_time = None
    n_file_with_latest_mod_time = 0
    file_path_with_largest_file_name = None

    if file_io.file_exists_v2(dir_name):
      for file_name in os.listdir(dir_name):
        # Only consider if `file_name` matches the pattern.
        if re.match(base_name_regex, file_name):
          file_path = os.path.join(dir_name, file_name)
          mod_time = os.path.getmtime(file_path)
          if (file_path_with_largest_file_name is None or
              file_path > file_path_with_largest_file_name):
            file_path_with_largest_file_name = file_path
          if mod_time > latest_mod_time:
            latest_mod_time = mod_time
            file_path_with_latest_mod_time = file_path
            # In the case a file with later modified time is found, reset
            # the counter for the number of files with latest modified time.
            n_file_with_latest_mod_time = 1
          elif mod_time == latest_mod_time:
            # In the case a file has modified time tied with the most recent,
            # increment the counter for the number of files with latest modified
            # time by 1.
            n_file_with_latest_mod_time += 1

    if n_file_with_latest_mod_time == 1:
      # Return the sole file that has most recent modified time.
      return file_path_with_latest_mod_time
    else:
      # If there are more than one file having latest modified time, return
      # the file path with the largest file name.
      return file_path_with_largest_file_name


class BackupAndRestore(Callback):
  # [CLASS DOC]
  # Fault-tolerance callback: at train begin it creates a worker training-state object on the
  # model for the given backup directory and restores any previous backup; after each epoch it
  # backs up the epoch and state; at train end it deletes the backup and the state. Works in eager
  # mode only and for a fixed set of distribution strategies. Not chief-only: every worker restores.
  """Callback to back up and restore the training state.

  `BackupAndRestore` callback is intended to recover from interruptions that
  happened in the middle of a model.fit execution by backing up the
  training states in a temporary checkpoint file (based on TF CheckpointManager)
  at the end of each epoch. If training restarted before completion, the
  training state and model are restored to the most recently saved state at the
  beginning of a new model.fit() run.
  Note that user is responsible to bring jobs back up.
  This callback is important for the backup and restore mechanism for fault
  tolerance purpose. And the model to be restored from an previous checkpoint is
  expected to be the same as the one used to back up. If user changes arguments
  passed to compile or fit, the checkpoint saved for fault tolerance can become
  invalid.

  Note:
  1. This callback is not compatible with disabling eager execution.
  2. A checkpoint is saved at the end of each epoch, when restoring we'll redo
  any partial work from an unfinished epoch in which the training got restarted
  (so the work done before a interruption doesn't affect the final model state).
  3. This works for both single worker and multi-worker mode, only
  MirroredStrategy and MultiWorkerMirroredStrategy are supported for now.

  Example:

  >>> class InterruptingCallback(tf.keras.callbacks.Callback):
  ...   def on_epoch_begin(self, epoch, logs=None):
  ...     if epoch == 4:
  ...       raise RuntimeError('Interrupting!')
  >>> callback = tf.keras.callbacks.experimental.BackupAndRestore(
  ... backup_dir="/tmp/backup")
  >>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
  >>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
  >>> try:
  ...   model.fit(np.arange(100).reshape(5, 20), np.zeros(5), epochs=10,
  ...             batch_size=1, callbacks=[callback, InterruptingCallback()],
  ...             verbose=0)
  ... except:
  ...   pass
  >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5), epochs=10,
  ...             batch_size=1, callbacks=[callback], verbose=0)
  >>> # Only 6 more epochs are run, since first training got interrupted at
  >>> # zero-indexed epoch 4, second training will continue from 4 to 9.
  >>> len(history.history['loss'])
  6

  Args:
      backup_dir: String, path to store the checkpoint. e.g. backup_dir =
        os.path.join(working_dir, 'backup') This is the directory in which the
        system stores temporary files to recover the model from jobs terminated
        unexpectedly. The directory cannot be reused elsewhere to store other
        files, e.g. by BackupAndRestore callback of another training, or by
        another callback (ModelCheckpoint) of the same training.
  """

  def __init__(self, backup_dir):
    # [FUNCTION DOC]
    # Stores the backup directory (string path), declares raw tensor log support, records the
    # tuple of supported strategy classes (mirrored, collective all-reduce, TPU and parameter
    # server variants) and marks the callback as not chief-only. Raises a value error when not
    # executing eagerly: one message if inside a compiled function, another (suggesting the model
    # checkpoint callback) in legacy graph mode.
    super(BackupAndRestore, self).__init__()
    self.backup_dir = backup_dir
    self._supports_tf_logs = True
    self._supported_strategies = (
        mirrored_strategy.MirroredStrategy,
        collective_all_reduce_strategy.CollectiveAllReduceStrategy,
        tpu_strategy.TPUStrategy, tpu_strategy.TPUStrategyV2,
        parameter_server_strategy_v2.ParameterServerStrategyV2)

    if not context.executing_eagerly():
      if ops.inside_function():
        raise ValueError('This Callback\'s method contains Python state and '
                         'should be called outside of `tf.function`s.')
      else:  # Legacy graph mode:
        raise ValueError(
            'BackupAndRestore only supports eager mode. In graph '
            'mode, consider using ModelCheckpoint to manually save '
            'and restore weights with `model.load_weights()` and by '
            'providing `initial_epoch` in `model.fit()` for fault tolerance.')

    # Only the chief worker writes model checkpoints, but all workers
    # restore checkpoint at on_train_begin().
    self._chief_worker_only = False

  def on_train_begin(self, logs=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the error message lists only a few supported strategies (empty, mirrored,
    # multi-worker mirrored and TPU) while the supported tuple also includes the parameter-server
    # strategy.
    # Verifies the model's distribution strategy is one of the supported ones (raises a
    # not-implemented error otherwise), creates the worker training state for the backup
    # directory, stores it on both the model and the callback, and restores any existing backup
    # (which can set the initial epoch). Returns nothing.
    # TrainingState is used to manage the training state needed for
    # failure-recovery of a worker in training.
    # pylint: disable=protected-access

    if self.model._distribution_strategy and not isinstance(
        self.model.distribute_strategy, self._supported_strategies):
      raise NotImplementedError(
          '%s is not supported yet. '
          'Currently BackupAndRestore callback only supports empty strategy, '
          'MirroredStrategy, MultiWorkerMirroredStrategy and TPUStrategy.' %
          type(self.model.distribute_strategy).__name__)
    self.model._training_state = (
        worker_training_state.WorkerTrainingState(self.model, self.backup_dir))
    self._training_state = self.model._training_state
    self._training_state.restore()

  def on_train_end(self, logs=None):
    # [FUNCTION DOC]
    # Deletes the backup checkpoint written for worker recovery and removes the training state
    # object from both the callback and the model. Returns nothing.
    # pylint: disable=protected-access
    # On exit of training, delete the training state backup file that was saved
    # for the purpose of worker recovery.
    self._training_state.delete_backup()

    # Clean up the training state.
    del self._training_state
    del self.model._training_state

  def on_epoch_end(self, epoch, logs=None):
    # [FUNCTION DOC]
    # Writes a backup of the model state and the finished epoch index via the training state so
    # training can resume after an interruption. Returns nothing.
    # Back up the model and current epoch for possible future recovery.
    self._training_state.back_up(epoch)


class EarlyStopping(Callback):
  # [CLASS DOC]
  # Callback that stops training when a monitored metric has not improved for a number of
  # epochs. State: monitored name, patience, minimum delta (sign adjusted for direction),
  # comparison function, optional baseline, wait counter, best value, stopped epoch and optionally
  # the best weights to restore when stopping.
  """Stop training when a monitored metric has stopped improving.

  Assuming the goal of a training is to minimize the loss. With this, the
  metric to be monitored would be `'loss'`, and mode would be `'min'`. A
  `model.fit()` training loop will check at end of every epoch whether
  the loss is no longer decreasing, considering the `min_delta` and
  `patience` if applicable. Once it's found no longer decreasing,
  `model.stop_training` is marked True and the training terminates.

  The quantity to be monitored needs to be available in `logs` dict.
  To make it so, pass the loss or metrics at `model.compile()`.

  Args:
    monitor: Quantity to be monitored.
    min_delta: Minimum change in the monitored quantity
        to qualify as an improvement, i.e. an absolute
        change of less than min_delta, will count as no
        improvement.
    patience: Number of epochs with no improvement
        after which training will be stopped.
    verbose: verbosity mode.
    mode: One of `{"auto", "min", "max"}`. In `min` mode,
        training will stop when the quantity
        monitored has stopped decreasing; in `"max"`
        mode it will stop when the quantity
        monitored has stopped increasing; in `"auto"`
        mode, the direction is automatically inferred
        from the name of the monitored quantity.
    baseline: Baseline value for the monitored quantity.
        Training will stop if the model doesn't show improvement over the
        baseline.
    restore_best_weights: Whether to restore model weights from
        the epoch with the best value of the monitored quantity.
        If False, the model weights obtained at the last step of
        training are used. An epoch will be restored regardless
        of the performance relative to the `baseline`. If no epoch
        improves on `baseline`, training will run for `patience`
        epochs and restore weights from the best epoch in that set.

  Example:

  >>> callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
  >>> # This callback will stop the training when there is no improvement in
  >>> # the loss for three consecutive epochs.
  >>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
  >>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
  >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
  ...                     epochs=10, batch_size=1, callbacks=[callback],
  ...                     verbose=0)
  >>> len(history.history['loss'])  # Only 4 epochs are run.
  4
  """

  def __init__(self,
               monitor='val_loss',
               min_delta=0,
               patience=0,
               verbose=0,
               mode='auto',
               baseline=None,
               restore_best_weights=False):
    # [FUNCTION DOC]
    # Configures early stopping. monitor is the metric name (default validation loss); min_delta
    # is the minimum change counting as improvement (its absolute value is stored, then signed
    # by direction); patience is the number of epochs without improvement tolerated; verbose is
    # the verbosity; mode is auto, min or max (unknown values log a warning and use auto, which
    # means max when the name contains acc, min otherwise); baseline is an optional value the
    # metric must beat; restore_best_weights chooses whether to restore the best epoch's weights.
    # Returns nothing.
    super(EarlyStopping, self).__init__()

    self.monitor = monitor
    self.patience = patience
    self.verbose = verbose
    self.baseline = baseline
    self.min_delta = abs(min_delta)
    self.wait = 0
    self.stopped_epoch = 0
    self.restore_best_weights = restore_best_weights
    self.best_weights = None

    if mode not in ['auto', 'min', 'max']:
      logging.warning('EarlyStopping mode %s is unknown, '
                      'fallback to auto mode.', mode)
      mode = 'auto'

    if mode == 'min':
      self.monitor_op = np.less
    elif mode == 'max':
      self.monitor_op = np.greater
    else:
      if 'acc' in self.monitor:
        self.monitor_op = np.greater
      else:
        self.monitor_op = np.less

    if self.monitor_op == np.greater:
      self.min_delta *= 1
    else:
      self.min_delta *= -1

  def on_train_begin(self, logs=None):
    # [FUNCTION DOC]
    # Resets the wait counter, stopped epoch and stored best weights, and sets the best value to
    # the worst possible (positive infinity for minimizing, negative infinity otherwise) so the
    # callback instance can be reused across fit calls. Returns nothing.
    # Allow instances to be re-used
    self.wait = 0
    self.stopped_epoch = 0
    self.best = np.inf if self.monitor_op == np.less else -np.inf
    self.best_weights = None

  def on_epoch_end(self, epoch, logs=None):
    # [FUNCTION DOC]
    # Reads the monitored value from the epoch logs (returns silently if unavailable). Increments
    # the wait counter; if the value improves on the best, records it (and the model weights when
    # restoring is on) and resets the wait only if it also beats the baseline (when one is set).
    # When the wait reaches the patience, remembers the epoch, sets the model's stop-training flag
    # and, if requested and weights were saved, restores the best weights (printing a note when
    # verbose). Returns nothing.
    current = self.get_monitor_value(logs)
    if current is None:
      return
    if self.restore_best_weights and self.best_weights is None:
      # Restore the weights after first epoch if no progress is ever made.
      self.best_weights = self.model.get_weights()

    self.wait += 1
    if self._is_improvement(current, self.best):
      self.best = current
      if self.restore_best_weights:
        self.best_weights = self.model.get_weights()
      # Only restart wait if we beat both the baseline and our previous best.
      if self.baseline is None or self._is_improvement(current, self.baseline):
        self.wait = 0

    if self.wait >= self.patience:
      self.stopped_epoch = epoch
      self.model.stop_training = True
      if self.restore_best_weights and self.best_weights is not None:
        if self.verbose > 0:
          print('Restoring model weights from the end of the best epoch.')
        self.model.set_weights(self.best_weights)

  def on_train_end(self, logs=None):
    # [FUNCTION DOC]
    # If training was stopped early and verbosity is on, prints a message with the one-based
    # stopping epoch to standard output. Returns nothing.
    if self.stopped_epoch > 0 and self.verbose > 0:
      print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

  def get_monitor_value(self, logs):
    # [FUNCTION DOC]
    # Returns the monitored metric's value from the logs mapping (nothing counts as empty). If it
    # is missing, logs a warning listing the available metric names and returns nothing/null.
    logs = logs or {}
    monitor_value = logs.get(self.monitor)
    if monitor_value is None:
      logging.warning('Early stopping conditioned on metric `%s` '
                      'which is not available. Available metrics are: %s',
                      self.monitor, ','.join(list(logs.keys())))
    return monitor_value

  def _is_improvement(self, monitor_value, reference_value):
    # [FUNCTION DOC]
    # HELPER: returns whether the monitored value beats the reference value by at least the
    # minimum delta in the monitored direction (the delta is already signed for the direction).
    return self.monitor_op(monitor_value - self.min_delta, reference_value)


class RemoteMonitor(Callback):
  # [CLASS DOC]
  # Callback that posts each epoch's logs, together with the epoch number, to a remote HTTP
  # server (root plus path), either as a JSON body or as a form field. Needs the optional
  # requests library. State: root URL, path, form field name, optional headers and the
  # send-as-JSON flag.
  """Callback used to stream events to a server.

  Requires the `requests` library.
  Events are sent to `root + '/publish/epoch/end/'` by default. Calls are
  HTTP POST, with a `data` argument which is a
  JSON-encoded dictionary of event data.
  If `send_as_json=True`, the content type of the request will be
  `"application/json"`.
  Otherwise the serialized JSON will be sent within a form.

  Args:
    root: String; root url of the target server.
    path: String; path relative to `root` to which the events will be sent.
    field: String; JSON field under which the data will be stored.
        The field is used only if the payload is sent within a form
        (i.e. send_as_json is set to False).
    headers: Dictionary; optional custom HTTP headers.
    send_as_json: Boolean; whether the request should be
        sent as `"application/json"`.
  """

  def __init__(self,
               root='http://localhost:9000',
               path='/publish/epoch/end/',
               field='data',
               headers=None,
               send_as_json=False):
    # [FUNCTION DOC]
    # Stores the server settings: root URL (default local port 9000), path (default epoch-end
    # publish path), form field name for the payload (default data), optional mapping of HTTP
    # headers, and whether to send as JSON (default false, i.e. form). Returns nothing.
    super(RemoteMonitor, self).__init__()

    self.root = root
    self.path = path
    self.field = field
    self.headers = headers
    self.send_as_json = send_as_json

  def on_epoch_end(self, epoch, logs=None):
    # [FUNCTION DOC]
    # Sends the epoch event to the server. Raises an import error if the requests library is not
    # installed. Builds a payload from the epoch index and the logs, unwrapping numpy values into
    # plain scalars, then does an HTTP POST (JSON body or form field holding serialized JSON,
    # with the configured headers). A network failure is caught and only logged as a warning.
    # Side effect: network request. Returns nothing.
    if requests is None:
      raise ImportError('RemoteMonitor requires the `requests` library.')
    logs = logs or {}
    send = {}
    send['epoch'] = epoch
    for k, v in logs.items():
      # np.ndarray and np.generic are not scalar types
      # therefore we must unwrap their scalar values and
      # pass to the json-serializable dict 'send'
      if isinstance(v, (np.ndarray, np.generic)):
        send[k] = v.item()
      else:
        send[k] = v
    try:
      if self.send_as_json:
        requests.post(self.root + self.path, json=send, headers=self.headers)
      else:
        requests.post(
            self.root + self.path, {self.field: json.dumps(send)},
            headers=self.headers)
    except requests.exceptions.RequestException:
      logging.warning('Warning: could not reach RemoteMonitor '
                      'root server at ' + str(self.root))


class LearningRateScheduler(Callback):
  # [CLASS DOC]
  # Callback that updates the optimizer's learning rate at the start of every epoch using a
  # user-supplied schedule function, and adds the current learning rate to the epoch logs.
  # State: the schedule function and the verbosity.
  """Learning rate scheduler.

  At the beginning of every epoch, this callback gets the updated learning rate
  value from `schedule` function provided at `__init__`, with the current epoch
  and current learning rate, and applies the updated learning rate
  on the optimizer.

  Args:
    schedule: a function that takes an epoch index (integer, indexed from 0)
        and current learning rate (float) as inputs and returns a new
        learning rate as output (float).
    verbose: int. 0: quiet, 1: update messages.

  Example:

  >>> # This function keeps the initial learning rate for the first ten epochs
  >>> # and decreases it exponentially after that.
  >>> def scheduler(epoch, lr):
  ...   if epoch < 10:
  ...     return lr
  ...   else:
  ...     return lr * tf.math.exp(-0.1)
  >>>
  >>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
  >>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
  >>> round(model.optimizer.lr.numpy(), 5)
  0.01

  >>> callback = tf.keras.callbacks.LearningRateScheduler(scheduler)
  >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
  ...                     epochs=15, callbacks=[callback], verbose=0)
  >>> round(model.optimizer.lr.numpy(), 5)
  0.00607

  """

  def __init__(self, schedule, verbose=0):
    # [FUNCTION DOC]
    # Stores the schedule function (takes an epoch index and optionally the current learning rate
    # and returns the new rate) and the verbosity (0 quiet, 1 prints updates). Returns nothing.
    super(LearningRateScheduler, self).__init__()
    self.schedule = schedule
    self.verbose = verbose

  def on_epoch_begin(self, epoch, logs=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the fallback catches any type error, including one raised inside a new-style
    # schedule, so such errors are masked and the schedule is called again with fewer arguments.
    # Applies the schedule at the start of an epoch. Raises a value error if the optimizer has no
    # learning-rate attribute. Reads the current rate and calls the schedule with epoch and rate,
    # falling back to calling it with the epoch only if that raises a type error (old style).
    # Raises a value error if the result is not a float or float tensor. Writes the new rate into
    # the optimizer and, when verbose, prints a message. Returns nothing.
    if not hasattr(self.model.optimizer, 'lr'):
      raise ValueError('Optimizer must have a "lr" attribute.')
    try:  # new API
      lr = float(backend.get_value(self.model.optimizer.lr))
      lr = self.schedule(epoch, lr)
    except TypeError:  # Support for old API for backward compatibility
      lr = self.schedule(epoch)
    if not isinstance(lr, (tensor_lib.Tensor, float, np.float32, np.float64)):
      raise ValueError('The output of the "schedule" function '
                       'should be float.')
    if isinstance(lr, tensor_lib.Tensor) and not lr.dtype.is_floating:
      raise ValueError('The dtype of Tensor should be float')
    backend.set_value(self.model.optimizer.lr, backend.get_value(lr))
    if self.verbose > 0:
      print('\nEpoch %05d: LearningRateScheduler setting learning '
            'rate to %s.' % (epoch + 1, lr))

  def on_epoch_end(self, epoch, logs=None):
    # [FUNCTION DOC]
    # Adds the optimizer's current learning rate to the epoch logs under the key lr (mutating the
    # given mapping; a fresh mapping is used if none was given, so then it is not visible to the
    # caller). Returns nothing.
    logs = logs or {}
    logs['lr'] = backend.get_value(self.model.optimizer.lr)


def keras_model_summary(name, data, step=None):
  # [FUNCTION DOC]
  # HELPER: used by the TensorBoard callback to write a conceptual model graph summary.
  # Builds summary metadata with the fixed graph-keras-model plugin name and content marker of
  # version one, serializes the model (data) to a JSON string, and if serialization fails of any
  # reason logs a warning and returns false. Otherwise, inside a summary scope named by name, it
  # creates a string constant on the CPU and writes it as a summary at the given step (when step
  # is omitted the global default step is used and must be set). Returns the result of the
  # summary write: true on success, false when no default writer is active. Raises a value error
  # (from the summary machinery) when a writer exists but no step is available.
  """Writes a Keras model as JSON to as a Summary.

  Writing the Keras model configuration allows the TensorBoard graph plugin to
  render a conceptual graph, as opposed to graph of ops. In case the model fails
  to serialize as JSON, it ignores and returns False.

  Args:
    name: A name for this summary. The summary tag used for TensorBoard will be
      this name prefixed by any active name scopes.
    data: A Keras Model to write.
    step: Explicit `int64`-castable monotonic step value for this summary. If
      omitted, this defaults to `tf.summary.experimental.get_step()`, which must
      not be None.

  Returns:
    True on success, or False if no summary was written because no default
    summary writer was available.

  Raises:
    ValueError: if a default writer exists, but no step was provided and
      `tf.summary.experimental.get_step()` is None.
  """
  summary_metadata = summary_pb2.SummaryMetadata()
  # Hard coding a plugin name. Please refer to go/tb-plugin-name-hardcode for
  # the rationale.
  summary_metadata.plugin_data.plugin_name = 'graph_keras_model'
  # version number = 1
  summary_metadata.plugin_data.content = b'1'

  try:
    json_string = data.to_json()
  except Exception as exc:  # pylint: disable=broad-except
    # An exception should not break a model code.
    logging.warning('Model failed to serialize as JSON. Ignoring... %s', exc)
    return False

  with summary_ops_v2.summary_scope(name, 'graph_keras_model',
                                    [data, step]) as (tag, _):
    with ops.device('cpu:0'):
      tensor = constant_op.constant(json_string, dtype=dtypes.string)
    return summary_ops_v2.write(
        tag=tag, tensor=tensor, step=step, metadata=summary_metadata)


class TensorBoard(Callback, version_utils.TensorBoardVersionSelector):
  # [CLASS DOC]
  # Callback that writes TensorBoard event data during fit and evaluate: epoch (and optional
  # batch-level) scalar metrics for train and validation runs, the model graph and train function
  # graph, weight histograms and images, embedding projector configuration and checkpoints,
  # steps-per-second timing and sampled profiler traces. State: the log directory, write options
  # and update frequency, lazily created train and validation file writers, the profiling batch
  # range and tracing/profiler flags, global batch counters and timing accumulators, and a stack
  # of saved default-summary contexts that is restored when training or evaluation ends. Also
  # inherits a version selector base so the right implementation is chosen between old and new
  # TensorBoard callback versions. In multi-worker runs only the chief writes to the real
  # directory; other workers write to temporary directories that are removed at the end.
  # pylint: disable=line-too-long
  """Enable visualizations for TensorBoard.

  TensorBoard is a visualization tool provided with TensorFlow.

  This callback logs events for TensorBoard, including:

  * Metrics summary plots
  * Training graph visualization
  * Activation histograms
  * Sampled profiling

  When used in `Model.evaluate`, in addition to epoch summaries, there will be
  a summary that records evaluation metrics vs `Model.optimizer.iterations`
  written. The metric names will be prepended with `evaluation`, with
  `Model.optimizer.iterations` being the step in the visualized TensorBoard.

  If you have installed TensorFlow with pip, you should be able
  to launch TensorBoard from the command line:

  ```
  tensorboard --logdir=path_to_your_logs
  ```

  You can find more information about TensorBoard
  [here](https://www.tensorflow.org/get_started/summaries_and_tensorboard).

  Args:
      log_dir: the path of the directory where to save the log files to be
        parsed by TensorBoard. e.g. log_dir = os.path.join(working_dir, 'logs')
        This directory should not be reused by any other callbacks.
      histogram_freq: frequency (in epochs) at which to compute activation and
        weight histograms for the layers of the model. If set to 0, histograms
        won't be computed. Validation data (or split) must be specified for
        histogram visualizations.
      write_graph: whether to visualize the graph in TensorBoard. The log file
        can become quite large when write_graph is set to True.
      write_images: whether to write model weights to visualize as image in
        TensorBoard.
      write_steps_per_second: whether to log the training steps per second into
        Tensorboard. This supports both epoch and batch frequency logging.
      update_freq: `'batch'` or `'epoch'` or integer. When using `'batch'`,
        writes the losses and metrics to TensorBoard after each batch. The same
        applies for `'epoch'`. If using an integer, let's say `1000`, the
        callback will write the metrics and losses to TensorBoard every 1000
        batches. Note that writing too frequently to TensorBoard can slow down
        your training.
      profile_batch: Profile the batch(es) to sample compute characteristics.
        profile_batch must be a non-negative integer or a tuple of integers.
        A pair of positive integers signify a range of batches to profile.
        By default, it will profile the second batch. Set profile_batch=0
        to disable profiling.
      embeddings_freq: frequency (in epochs) at which embedding layers will be
        visualized. If set to 0, embeddings won't be visualized.
      embeddings_metadata: Dictionary which maps embedding layer names to the
        filename of a file in which to save metadata for the embedding layer.
        In case the same metadata file is to be
        used for all embedding layers, a single filename can be passed.

  Examples:

  Basic usage:

  ```python
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
  model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])
  # Then run the tensorboard command to view the visualizations.
  ```

  Custom batch-level summaries in a subclassed Model:

  ```python
  class MyModel(tf.keras.Model):

    def build(self, _):
      self.dense = tf.keras.layers.Dense(10)

    def call(self, x):
      outputs = self.dense(x)
      tf.summary.histogram('outputs', outputs)
      return outputs

  model = MyModel()
  model.compile('sgd', 'mse')

  # Make sure to set `update_freq=N` to log a batch-level summary every N batches.
  # In addition to any `tf.summary` contained in `Model.call`, metrics added in
  # `Model.compile` will be logged every N batches.
  tb_callback = tf.keras.callbacks.TensorBoard('./logs', update_freq=1)
  model.fit(x_train, y_train, callbacks=[tb_callback])
  ```

  Custom batch-level summaries in a Functional API Model:

  ```python
  def my_summary(x):
    tf.summary.histogram('x', x)
    return x

  inputs = tf.keras.Input(10)
  x = tf.keras.layers.Dense(10)(inputs)
  outputs = tf.keras.layers.Lambda(my_summary)(x)
  model = tf.keras.Model(inputs, outputs)
  model.compile('sgd', 'mse')

  # Make sure to set `update_freq=N` to log a batch-level summary every N batches.
  # In addition to any `tf.summary` contained in `Model.call`, metrics added in
  # `Model.compile` will be logged every N batches.
  tb_callback = tf.keras.callbacks.TensorBoard('./logs', update_freq=1)
  model.fit(x_train, y_train, callbacks=[tb_callback])
  ```

  Profiling:

  ```python
  # Profile a single batch, e.g. the 5th batch.
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir='./logs', profile_batch=5)
  model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])

  # Profile a range of batches, e.g. from 10 to 20.
  tensorboard_callback = tf.keras.callbacks.TensorBoard(
      log_dir='./logs', profile_batch=(10,20))
  model.fit(x_train, y_train, epochs=2, callbacks=[tensorboard_callback])
  ```
  """

  # pylint: enable=line-too-long

  def __init__(self,
               log_dir='logs',
               histogram_freq=0,
               write_graph=True,
               write_images=False,
               write_steps_per_second=False,
               update_freq='epoch',
               profile_batch=2,
               embeddings_freq=0,
               embeddings_metadata=None,
               **kwargs):
    # [FUNCTION DOC]
    # Configures the callback. log_dir (string or path-like, default logs) is where event files
    # go; histogram_freq is the epoch interval for weight histograms (0 disables); write_graph
    # and write_images toggle graph and weight-image output; write_steps_per_second toggles
    # steps-per-second logging; update_freq is batch, epoch or an integer batch count (batch is
    # stored as one); profile_batch is a non-negative integer or a pair of positive integers
    # (default the second batch, zero disables profiling); embeddings_freq is the epoch interval
    # for embedding visualization (0 disables); embeddings_metadata maps embedding layer names to
    # metadata file names or is a single file name. Extra keyword arguments are only accepted for
    # legacy names, which are warned about and ignored; any other raises a value error.
    # Initializes counters, the empty writer cache and the saved summary state stack; declares raw
    # tensor log support. Invalid profile_batch values raise a value error.
    super(TensorBoard, self).__init__()
    self._supports_tf_logs = True
    self._validate_kwargs(kwargs)

    self.log_dir = path_to_string(log_dir)
    self.histogram_freq = histogram_freq
    self.write_graph = write_graph
    self.write_images = write_images
    self.write_steps_per_second = write_steps_per_second
    self.update_freq = 1 if update_freq == 'batch' else update_freq
    self.embeddings_freq = embeddings_freq
    self.embeddings_metadata = embeddings_metadata
    self._init_profile_batch(profile_batch)
    self._global_train_batch = 0
    self._previous_epoch_iterations = 0
    self._train_accumulated_time = 0
    self._batch_start_time = 0

    # Lazily initialized in order to avoid creating event files when
    # not needed.
    self._writers = {}

    # Used to restore any existing `SummaryWriter` after training ends.
    self._prev_summary_state = []

  def _validate_kwargs(self, kwargs):
    # [FUNCTION DOC]
    # HELPER: checks the extra constructor arguments. Logs a warning for each recognized legacy
    # option that was supplied with a truthy value (gradient writing, batch size, embedding layer
    # names, embedding data) saying it is ignored or unsupported, then raises a value error if
    # any key outside those four legacy names is present. Input: mapping of extra keyword
    # arguments. Returns nothing.
    """Handle arguments were supported in V1."""
    if kwargs.get('write_grads', False):
      logging.warning('`write_grads` will be ignored in TensorFlow 2.0 '
                      'for the `TensorBoard` Callback.')
    if kwargs.get('batch_size', False):
      logging.warning('`batch_size` is no longer needed in the '
                      '`TensorBoard` Callback and will be ignored '
                      'in TensorFlow 2.0.')
    if kwargs.get('embeddings_layer_names', False):
      logging.warning('`embeddings_layer_names` is not supported in '
                      'TensorFlow 2.0. Instead, all `Embedding` layers '
                      'will be visualized.')
    if kwargs.get('embeddings_data', False):
      logging.warning('`embeddings_data` is not supported in TensorFlow '
                      '2.0. Instead, all `Embedding` variables will be '
                      'visualized.')

    unrecognized_kwargs = set(kwargs.keys()) - {
        'write_grads', 'embeddings_layer_names', 'embeddings_data', 'batch_size'
    }

    # Only allow kwargs that were supported in V1.
    if unrecognized_kwargs:
      raise ValueError('Unrecognized arguments in `TensorBoard` '
                       'Callback: ' + str(unrecognized_kwargs))

  def set_model(self, model):
    # [FUNCTION DOC]
    # Attaches the model and prepares output locations: computes the log directory to write to
    # (a temporary one on non-chief workers), the train and validation sub-directories and the
    # model's train and test step counters, and clears any cached writers. If graph writing is on,
    # writes the model graph summary and flags that the train function graph should be written
    # after the first training batch. If embedding visualization is on, writes the projector
    # configuration file. Side effects: creates writers and files. Returns nothing.
    """Sets Keras model and writes graph if specified."""
    self.model = model
    self._log_write_dir = self._get_log_write_dir()

    self._train_dir = os.path.join(self._log_write_dir, 'train')
    self._train_step = self.model._train_counter  # pylint: disable=protected-access

    self._val_dir = os.path.join(self._log_write_dir, 'validation')
    self._val_step = self.model._test_counter  # pylint: disable=protected-access

    self._writers = {}  # Resets writers.

    self._should_write_train_graph = False
    if self.write_graph:
      self._write_keras_model_summary()
      self._should_write_train_graph = True
    if self.embeddings_freq:
      self._configure_embeddings()

  @property
  def _train_writer(self):
    # [FUNCTION DOC]
    # HELPER: property returning the file writer for training summaries, creating it lazily in the
    # train directory the first time (so no event file is made unless needed).
    if 'train' not in self._writers:
      self._writers['train'] = summary_ops_v2.create_file_writer_v2(
          self._train_dir)
    return self._writers['train']

  @property
  def _val_writer(self):
    # [FUNCTION DOC]
    # HELPER: property returning the file writer for validation summaries, creating it lazily in
    # the validation directory the first time.
    if 'val' not in self._writers:
      self._writers['val'] = summary_ops_v2.create_file_writer_v2(self._val_dir)
    return self._writers['val']

  def _get_log_write_dir(self):
    # [FUNCTION DOC]
    # HELPER: returns the directory this worker should write logs to: the configured log directory
    # on the chief, or a temporary directory on other workers in multi-worker training, decided
    # from the model's distribution strategy.
    """For multi-worker, only chief should write, others write to '/tmp'."""
    return distributed_file_utils.write_dirpath(self.log_dir,
                                                self.model.distribute_strategy)

  def _delete_tmp_write_dir(self):
    # [FUNCTION DOC]
    # HELPER: removes the temporary log directory used by non-chief workers in multi-worker
    # training (no effect for the chief). Returns nothing.
    """Deletes tmp write directories for multi-worker."""
    distributed_file_utils.remove_temp_dirpath(self.log_dir,
                                               self.model.distribute_strategy)

  def _write_keras_model_train_graph(self):
    # [FUNCTION DOC]
    # HELPER: with the train writer as default and recording forced on, writes the graph of the
    # model's compiled train function as a graph summary, but only when that function is a
    # compiled function object exposing a function spec. Returns nothing.
    """Writes Keras model train_function graph to TensorBoard."""
    with self._train_writer.as_default():
      with summary_ops_v2.record_if(True):
        train_fn = self.model.train_tf_function
        # If the train_function is a `tf.function`, we can write out a graph
        if hasattr(train_fn, 'function_spec'):
          summary_ops_v2.graph(train_fn._concrete_stateful_fn.graph)  # pylint: disable=protected-access

  def _write_keras_model_summary(self):
    # [FUNCTION DOC]
    # HELPER: writes the conceptual model graph summary at step zero through the train writer, but
    # only for graph-network models (functional) or models whose class is named Sequential;
    # other (subclassed) models are skipped. Returns nothing.
    """Writes Keras graph network summary to TensorBoard."""
    with self._train_writer.as_default():
      with summary_ops_v2.record_if(True):
        summary_writable = (
            self.model._is_graph_network or  # pylint: disable=protected-access
            self.model.__class__.__name__ == 'Sequential')  # pylint: disable=protected-access
        if summary_writable:
          keras_model_summary('keras', self.model, step=0)

  def _configure_embeddings(self):
    # [FUNCTION DOC]
    # HELPER: writes the embedding projector configuration. For every embedding layer of the model
    # it adds an entry with a fixed tensor name (assuming the embedding is the first layer with
    # weights) and, when metadata was given, the metadata file (a single string applies to all
    # layers; a mapping is looked up by layer name and the used entries are removed from it). Raises
    # a value error if the mapping still has names that matched no embedding layer. Writes the
    # configuration as text into a projector config file in the log directory. Returns nothing.
    """Configure the Projector for embeddings."""
    # TODO(omalleyt): Add integration tests.
    from google.protobuf import text_format
    from tensorflow.python.keras.layers import embeddings
    from tensorflow.python.keras.protobuf import projector_config_pb2

    config = projector_config_pb2.ProjectorConfig()
    for layer in self.model.layers:
      if isinstance(layer, embeddings.Embedding):
        embedding = config.embeddings.add()
        # Embeddings are always the first layer, so this naming should be
        # consistent in any keras models checkpoints.
        name = 'layer_with_weights-0/embeddings/.ATTRIBUTES/VARIABLE_VALUE'
        embedding.tensor_name = name

        if self.embeddings_metadata is not None:
          if isinstance(self.embeddings_metadata, str):
            embedding.metadata_path = self.embeddings_metadata
          else:
            if layer.name in self.embeddings_metadata.keys():
              embedding.metadata_path = self.embeddings_metadata.pop(layer.name)

    if self.embeddings_metadata and not isinstance(self.embeddings_metadata,
                                                   str):
      raise ValueError('Unrecognized `Embedding` layer names passed to '
                       '`keras.callbacks.TensorBoard` `embeddings_metadata` '
                       'argument: ' + str(self.embeddings_metadata.keys()))

    config_pbtxt = text_format.MessageToString(config)
    path = os.path.join(self._log_write_dir, 'projector_config.pbtxt')
    with gfile.Open(path, 'w') as f:
      f.write(config_pbtxt)

  def _push_writer(self, writer, step):
    # [FUNCTION DOC]
    # HELPER: makes a writer the default for custom batch-level summaries during training or
    # evaluation. Does nothing when updates are per epoch. Otherwise enters two contexts, saving
    # them on a stack for later: the writer as default at the current step value, and a recording
    # condition that is true every update-frequency steps. Inputs: file writer, step variable.
    # Returns nothing.
    """Sets the default writer for custom batch-level summaries."""
    if self.update_freq == 'epoch':
      return

    should_record = lambda: math_ops.equal(step % self.update_freq, 0)
    # TODO(b/151339474): Fix deadlock when not using .value() here.
    summary_context = (writer.as_default(step.value()),
                       summary_ops_v2.record_if(should_record))
    self._prev_summary_state.append(summary_context)
    summary_context[0].__enter__()
    summary_context[1].__enter__()

  def _pop_writer(self):
    # [FUNCTION DOC]
    # HELPER: undoes the matching push of a default writer. Does nothing when updates are per
    # epoch; otherwise pops the saved pair of contexts and exits them in reverse order, restoring
    # the previous default writer and recording condition. Returns nothing.
    """Pops the current writer."""
    if self.update_freq == 'epoch':
      return

    # See _push_writer for the content of the previous_context, which is pair
    # of context.
    previous_context = self._prev_summary_state.pop()
    previous_context[1].__exit__(*sys.exc_info())
    previous_context[0].__exit__(*sys.exc_info())

  def _close_writers(self):
    # [FUNCTION DOC]
    # HELPER: closes every file writer that was created (train and validation). Returns nothing.
    for writer in self._writers.values():
      writer.close()

  def _init_profile_batch(self, profile_batch):
    # [FUNCTION DOC]
    # HELPER: parses and validates the profiling setting and sets up profiling state. Accepts an
    # integer, a two-element tuple or list, or a legacy comma-separated string (converted to
    # integers); an integer means start and stop are the same batch. Raises a value error for any
    # other shape, a negative start, or a stop before the start. Initializes the profiler-started
    # and tracing flags; if profiling starts after the first batch, starts and immediately stops
    # the profiler once without saving as a warm-up. Tracing is enabled unless both start and stop
    # are zero. Sets the start and stop batch attributes. Returns nothing.
    """Validate profile_batch value and set the range of batches to profile.
    Sets values of _start_batch and _stop_batch attributes,
    specifying the start and stop batch to profile.
    Setting `profile_batch=0` disables profiling.

    Args:
      profile_batch: The range of batches to profile. Should be a non-negative
        integer or a comma separated string of pair of positive integers. A pair
        of positive integers signify a range of batches to profile.

    Raises:
      ValueError: If profile_batch is not an integer or a comma separated pair
                  of positive integers.

    """
    profile_batch_error_message = (
        'profile_batch must be a non-negative integer or 2-tuple of positive '
        'integers. A pair of positive integers signifies a range of batches '
        'to profile. Found: {}'.format(profile_batch))

    # Support legacy way of specifying "start,stop" or "start" as str.
    if isinstance(profile_batch, str):
      profile_batch = str(profile_batch).split(',')
      profile_batch = nest.map_structure(int, profile_batch)

    if isinstance(profile_batch, int):
      self._start_batch = profile_batch
      self._stop_batch = profile_batch
    elif isinstance(profile_batch, (tuple, list)) and len(profile_batch) == 2:
      self._start_batch, self._stop_batch = profile_batch
    else:
      raise ValueError(profile_batch_error_message)

    if self._start_batch < 0 or self._stop_batch < self._start_batch:
      raise ValueError(profile_batch_error_message)

    # True when the profiler was successfully started by this callback.
    # We track the status here to make sure callbacks do not interfere with
    # each other. The callback will only stop the profiler it started.
    self._profiler_started = False
    if self._start_batch > 0:
      # Warm up and improve the profiling accuracy.
      self._start_profiler(logdir='')
      self._stop_profiler(save=False)
    # True when a trace is running.
    self._is_tracing = False

    # Setting `profile_batch=0` disables profiling.
    self._should_trace = not (self._start_batch == 0 and self._stop_batch == 0)

  def on_train_begin(self, logs=None):
    # [FUNCTION DOC]
    # Resets the global batch counter, the previous-epoch iteration count and the accumulated
    # training time, then makes the train writer the default for batch-level summaries.
    # Returns nothing.
    self._global_train_batch = 0
    self._previous_epoch_iterations = 0
    self._train_accumulated_time = 0
    self._push_writer(self._train_writer, self._train_step)

  def on_train_end(self, logs=None):
    # [FUNCTION DOC]
    # Restores the previous default writer, stops and exports any trace still running, closes all
    # writers and deletes the temporary log directory used by non-chief workers. Returns nothing.
    self._pop_writer()

    if self._is_tracing:
      self._stop_trace()

    self._close_writers()
    self._delete_tmp_write_dir()

  def on_test_begin(self, logs=None):
    # [FUNCTION DOC]
    # Makes the validation writer the default for batch-level summaries during evaluation.
    # Returns nothing.
    self._push_writer(self._val_writer, self._val_step)

  def on_test_end(self, logs=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: logs is iterated without a check for nothing, so a missing logs value raises an
    # error when the model has an optimizer with an iteration counter.
    # When the model has an optimizer with an iterations counter, writes every evaluation metric
    # in logs as a scalar named with an evaluation prefix and a vs-iterations suffix, using the
    # optimizer iteration count as the step, via the validation writer. Then restores the previous
    # default writer. Returns nothing.
    if self.model.optimizer and hasattr(self.model.optimizer, 'iterations'):
      with summary_ops_v2.record_if(True), self._val_writer.as_default():
        for name, value in logs.items():
          summary_ops_v2.scalar(
              'evaluation_' + name + '_vs_iterations',
              value,
              step=self.model.optimizer.iterations.read_value())
    self._pop_writer()

  def _implements_train_batch_hooks(self):
    # [FUNCTION DOC]
    # Overrides the base check: batch hooks are needed only when tracing is enabled or steps per
    # second logging is on.
    # Only call batch hooks when tracing or write_steps_per_second are enabled
    return self._should_trace or self.write_steps_per_second

  def on_train_batch_begin(self, batch, logs=None):
    # [FUNCTION DOC]
    # Increments the global training batch counter, records the batch start time when steps per
    # second logging is on, and starts a trace and profiler when tracing is enabled and the
    # counter equals the configured start batch. Returns nothing.
    self._global_train_batch += 1
    if self.write_steps_per_second:
      self._batch_start_time = time.time()
    if not self._should_trace:
      return

    if self._global_train_batch == self._start_batch:
      self._start_trace()

  def on_train_batch_end(self, batch, logs=None):
    # [FUNCTION DOC]
    # After the first training batch writes the train function graph once. When steps per second
    # logging is on, measures the batch duration, adds it to the accumulated time and writes a
    # batch steps-per-second scalar. When tracing is active and the global batch counter has
    # reached the stop batch, stops and exports the trace. Returns nothing.
    if self._should_write_train_graph:
      self._write_keras_model_train_graph()
      self._should_write_train_graph = False
    if self.write_steps_per_second:
      batch_run_time = time.time() - self._batch_start_time
      self._train_accumulated_time += batch_run_time
      summary_ops_v2.scalar(
          'batch_steps_per_second', 1. / batch_run_time, step=self._train_step)
    if not self._should_trace:
      return

    if self._is_tracing and self._global_train_batch >= self._stop_batch:
      self._stop_trace()

  def on_epoch_begin(self, epoch, logs=None):
    # [FUNCTION DOC]
    # When steps per second logging is on, remembers the optimizer iteration count at the epoch
    # start and zeroes the accumulated batch time so epoch-level steps per second can be computed.
    # Returns nothing.
    # Keeps track of epoch for profiling.
    if self.write_steps_per_second:
      self._previous_epoch_iterations = self.model.optimizer.iterations.numpy()
      self._train_accumulated_time = 0

  def on_epoch_end(self, epoch, logs=None):
    # [FUNCTION DOC]
    # Writes the epoch metrics as scalar summaries, then, if the histogram frequency divides the
    # epoch index, writes weight histograms (and images if enabled), and if the embeddings
    # frequency divides the epoch index, saves an embeddings checkpoint. Returns nothing.
    """Runs metrics and histogram summaries at epoch end."""
    self._log_epoch_metrics(epoch, logs)

    if self.histogram_freq and epoch % self.histogram_freq == 0:
      self._log_weights(epoch)

    if self.embeddings_freq and epoch % self.embeddings_freq == 0:
      self._log_embeddings(epoch)

  def _start_trace(self):
    # [FUNCTION DOC]
    # HELPER: turns on graph tracing (without the summary-level profiler), starts the profiler
    # writing into the train directory and marks that a trace is running. Returns nothing.
    summary_ops_v2.trace_on(graph=True, profiler=False)
    self._start_profiler(logdir=self._train_dir)
    self._is_tracing = True

  def _stop_trace(self, batch=None):
    # [FUNCTION DOC]
    # HELPER: exports the collected trace graph through the train writer under a name and step
    # based on the batch number (default: the configured stop batch), stops the profiler (saving
    # its results) and marks tracing as finished. Returns nothing.
    """Logs the trace graph to TensorBoard."""
    if batch is None:
      batch = self._stop_batch
    with self._train_writer.as_default():
      with summary_ops_v2.record_if(True):
        # TODO(b/126388999): Remove step info in the summary name.
        summary_ops_v2.trace_export(name='batch_%d' % batch, step=batch)
    self._stop_profiler()
    self._is_tracing = False

  def _collect_learning_rate(self, logs):
    # [FUNCTION DOC]
    # HELPER: if the optimizer's learning rate is a schedule object, evaluates it at the current
    # optimizer iteration count and stores it in the given logs mapping under the learning-rate
    # key (mutating it). Returns the same logs mapping.
    lr_schedule = getattr(self.model.optimizer, 'lr', None)
    if isinstance(lr_schedule, learning_rate_schedule.LearningRateSchedule):
      logs['learning_rate'] = lr_schedule(self.model.optimizer.iterations)
    return logs

  def _compute_steps_per_second(self):
    # [FUNCTION DOC]
    # HELPER: returns the number of optimizer iterations since the epoch started divided by the
    # accumulated batch time. Raises a division error if no batch time was accumulated.
    current_iteration = self.model.optimizer.iterations.numpy()
    steps_per_second = ((current_iteration - self._previous_epoch_iterations) /
                        (self._train_accumulated_time))
    return steps_per_second

  def _log_epoch_metrics(self, epoch, logs):
    # [FUNCTION DOC]
    # HELPER: writes epoch metrics as scalar summaries named with an epoch prefix. Does nothing
    # when logs is empty. Splits logs into training entries and validation entries (keys starting
    # with the validation prefix), adds the learning rate when a schedule is used and steps per
    # second when enabled to the training entries, and writes training scalars with the train
    # writer and validation scalars (prefix stripped) with the validation writer, at the epoch as
    # step. Inputs: epoch (integer step), logs mapping. Returns nothing.
    """Writes epoch metrics out as scalar summaries.

    Args:
        epoch: Int. The global step to use for TensorBoard.
        logs: Dict. Keys are scalar summary names, values are scalars.
    """
    if not logs:
      return

    train_logs = {k: v for k, v in logs.items() if not k.startswith('val_')}
    val_logs = {k: v for k, v in logs.items() if k.startswith('val_')}
    train_logs = self._collect_learning_rate(train_logs)
    if self.write_steps_per_second:
      train_logs['steps_per_second'] = self._compute_steps_per_second()

    with summary_ops_v2.record_if(True):
      if train_logs:
        with self._train_writer.as_default():
          for name, value in train_logs.items():
            summary_ops_v2.scalar('epoch_' + name, value, step=epoch)
      if val_logs:
        with self._val_writer.as_default():
          for name, value in val_logs.items():
            name = name[4:]  # Remove 'val_' prefix.
            summary_ops_v2.scalar('epoch_' + name, value, step=epoch)

  def _log_weights(self, epoch):
    # [FUNCTION DOC]
    # HELPER: writes a histogram for every weight of every model layer (name with colons replaced
    # by underscores) at the epoch step through the train writer, plus an image of each weight when
    # image writing is on, then flushes the writer. Returns nothing.
    """Logs the weights of the Model to TensorBoard."""
    with self._train_writer.as_default():
      with summary_ops_v2.record_if(True):
        for layer in self.model.layers:
          for weight in layer.weights:
            weight_name = weight.name.replace(':', '_')
            summary_ops_v2.histogram(weight_name, weight, step=epoch)
            if self.write_images:
              self._log_weight_as_image(weight, weight_name, epoch)
        self._train_writer.flush()

  def _log_weight_as_image(self, weight, weight_name, epoch):
    # [FUNCTION DOC]
    # HELPER: reshapes a weight tensor into an image batch and writes it as an image summary at
    # the epoch step. Singleton dimensions are squeezed; a vector becomes a one-row image; a
    # matrix is transposed when taller than wide and shown as one image; a three-dimensional
    # weight (conv kernel) is moved to a channels-first arrangement when the backend is channels
    # last so each kernel is its own image. Only written when the result has four dimensions and
    # one, three or four channels; other shapes are skipped silently. Returns nothing.
    """Logs a weight as a TensorBoard image."""
    w_img = array_ops.squeeze(weight)
    shape = backend.int_shape(w_img)
    if len(shape) == 1:  # Bias case
      w_img = array_ops.reshape(w_img, [1, shape[0], 1, 1])
    elif len(shape) == 2:  # Dense layer kernel case
      if shape[0] > shape[1]:
        w_img = array_ops.transpose(w_img)
        shape = backend.int_shape(w_img)
      w_img = array_ops.reshape(w_img, [1, shape[0], shape[1], 1])
    elif len(shape) == 3:  # ConvNet case
      if backend.image_data_format() == 'channels_last':
        # Switch to channels_first to display every kernel as a separate
        # image.
        w_img = array_ops.transpose(w_img, perm=[2, 0, 1])
        shape = backend.int_shape(w_img)
      w_img = array_ops.reshape(w_img, [shape[0], shape[1], shape[2], 1])

    shape = backend.int_shape(w_img)
    # Not possible to handle 3D convnets etc.
    if len(shape) == 4 and shape[-1] in [1, 3, 4]:
      summary_ops_v2.image(weight_name, w_img, step=epoch)

  def _log_embeddings(self, epoch):
    # [FUNCTION DOC]
    # HELPER: saves the model weights as a checkpoint named with the epoch number inside the train
    # sub-directory of the log directory, for the embedding projector. Returns nothing.
    embeddings_ckpt = os.path.join(self._log_write_dir, 'train',
                                   'keras_embedding.ckpt-{}'.format(epoch))
    self.model.save_weights(embeddings_ckpt)

  def _start_profiler(self, logdir):
    # [FUNCTION DOC]
    # HELPER: starts the profiler writing to the given directory unless this callback already
    # started it. Marks the profiler as started on success; if the profiler is already running
    # elsewhere, logs an error instead of raising. Returns nothing.
    """Starts the profiler if currently inactive.

    Args:
      logdir: Directory where profiler results will be saved.
    """
    if self._profiler_started:
      return
    try:
      profiler.start(logdir=logdir)
      self._profiler_started = True
    except errors.AlreadyExistsError as e:
      # Profiler errors should not be fatal.
      logging.error('Failed to start profiler: %s', e.message)

  def _stop_profiler(self, save=True):
    # [FUNCTION DOC]
    # HELPER: stops the profiler only if this callback started it, saving results when save is
    # true (default). If the profiler is unavailable, logs an error instead of raising. Always
    # clears the started flag afterwards. Returns nothing.
    """Stops the profiler if currently active.

    Args:
      save: Whether to save the profiler results to TensorBoard.
    """
    if not self._profiler_started:
      return
    try:
      profiler.stop(save=save)
    except errors.UnavailableError as e:
      # Profiler errors should not be fatal.
      logging.error('Failed to stop profiler: %s', e.message)
    finally:
      self._profiler_started = False


class ReduceLROnPlateau(Callback):
  # [CLASS DOC]
  # Callback that lowers the optimizer learning rate by a multiplicative factor when a monitored
  # metric stops improving for a number of epochs, with an optional cooldown period and a lower
  # bound. State: monitored name, factor, minimum learning rate, minimum delta, patience,
  # cooldown settings and counters, the wait counter, the best value so far and the comparison
  # function derived from the mode.
  """Reduce learning rate when a metric has stopped improving.

  Models often benefit from reducing the learning rate by a factor
  of 2-10 once learning stagnates. This callback monitors a
  quantity and if no improvement is seen for a 'patience' number
  of epochs, the learning rate is reduced.

  Example:

  ```python
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                patience=5, min_lr=0.001)
  model.fit(X_train, Y_train, callbacks=[reduce_lr])
  ```

  Args:
      monitor: quantity to be monitored.
      factor: factor by which the learning rate will be reduced.
        `new_lr = lr * factor`.
      patience: number of epochs with no improvement after which learning rate
        will be reduced.
      verbose: int. 0: quiet, 1: update messages.
      mode: one of `{'auto', 'min', 'max'}`. In `'min'` mode,
        the learning rate will be reduced when the
        quantity monitored has stopped decreasing; in `'max'` mode it will be
        reduced when the quantity monitored has stopped increasing; in `'auto'`
        mode, the direction is automatically inferred from the name of the
        monitored quantity.
      min_delta: threshold for measuring the new optimum, to only focus on
        significant changes.
      cooldown: number of epochs to wait before resuming normal operation after
        lr has been reduced.
      min_lr: lower bound on the learning rate.
  """

  def __init__(self,
               monitor='val_loss',
               factor=0.1,
               patience=10,
               verbose=0,
               mode='auto',
               min_delta=1e-4,
               cooldown=0,
               min_lr=0,
               **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: unrecognized extra keyword arguments (other than the deprecated delta name) are
    # silently ignored rather than rejected.
    # Configures the callback. monitor is the metric name (default validation loss); factor is the
    # multiplier applied to the learning rate and must be below one (a value of one or more raises
    # a value error); patience is the epochs without improvement before reducing; verbose is 0 or
    # 1; mode is auto, min or max; min_delta is the threshold for a significant change; cooldown is
    # the epochs to wait after a reduction; min_lr is the lower bound of the rate. A deprecated
    # keyword argument for the delta overrides min_delta with a warning. Initializes counters and
    # calls the reset step. Returns nothing.
    super(ReduceLROnPlateau, self).__init__()

    self.monitor = monitor
    if factor >= 1.0:
      raise ValueError('ReduceLROnPlateau ' 'does not support a factor >= 1.0.')
    if 'epsilon' in kwargs:
      min_delta = kwargs.pop('epsilon')
      logging.warning('`epsilon` argument is deprecated and '
                      'will be removed, use `min_delta` instead.')
    self.factor = factor
    self.min_lr = min_lr
    self.min_delta = min_delta
    self.patience = patience
    self.verbose = verbose
    self.cooldown = cooldown
    self.cooldown_counter = 0  # Cooldown counter.
    self.wait = 0
    self.best = 0
    self.mode = mode
    self.monitor_op = None
    self._reset()

  def _reset(self):
    # [FUNCTION DOC]
    # HELPER: internal reset step of the plateau callback (constructor and train begin).
    # KNOWN ISSUE: the docstring only mentions the wait and cooldown counters, but this also
    # validates the mode (unknown values log a warning and fall back to auto), builds the
    # comparison function and resets the best value.
    # Chooses the comparison rule: for min mode, or auto mode when the monitored name does not
    # contain acc, an improvement is a decrease by more than the minimum delta and the best starts
    # at positive infinity; otherwise an increase by more than the delta with the best starting at
    # negative infinity. Resets cooldown counter and wait counter to zero. Returns nothing.
    """Resets wait counter and cooldown counter.
    """
    if self.mode not in ['auto', 'min', 'max']:
      logging.warning('Learning rate reduction mode %s is unknown, '
                      'fallback to auto mode.', self.mode)
      self.mode = 'auto'
    if (self.mode == 'min' or
        (self.mode == 'auto' and 'acc' not in self.monitor)):
      self.monitor_op = lambda a, b: np.less(a, b - self.min_delta)
      self.best = np.inf
    else:
      self.monitor_op = lambda a, b: np.greater(a, b + self.min_delta)
      self.best = -np.inf
    self.cooldown_counter = 0
    self.wait = 0

  def on_train_begin(self, logs=None):
    # [FUNCTION DOC]
    # Resets the counters, comparison rule and best value so the callback can be reused across fit
    # calls. Returns nothing.
    self._reset()

  def on_epoch_end(self, epoch, logs=None):
    # [FUNCTION DOC]
    # Adds the current learning rate to the epoch logs under the lr key, then reads the monitored
    # metric (logging a warning listing available metrics if it is missing). Otherwise: during
    # cooldown decrements the cooldown counter and zeroes the wait; if the value improves, stores
    # it and zeroes the wait; else, when not in cooldown, increments the wait and once it reaches
    # the patience and the current rate is above the minimum, multiplies the rate by the factor
    # (not going below the minimum), sets it on the optimizer, optionally prints a message,
    # starts the cooldown and zeroes the wait. Returns nothing.
    logs = logs or {}
    logs['lr'] = backend.get_value(self.model.optimizer.lr)
    current = logs.get(self.monitor)
    if current is None:
      logging.warning('Learning rate reduction is conditioned on metric `%s` '
                      'which is not available. Available metrics are: %s',
                      self.monitor, ','.join(list(logs.keys())))

    else:
      if self.in_cooldown():
        self.cooldown_counter -= 1
        self.wait = 0

      if self.monitor_op(current, self.best):
        self.best = current
        self.wait = 0
      elif not self.in_cooldown():
        self.wait += 1
        if self.wait >= self.patience:
          old_lr = backend.get_value(self.model.optimizer.lr)
          if old_lr > np.float32(self.min_lr):
            new_lr = old_lr * self.factor
            new_lr = max(new_lr, self.min_lr)
            backend.set_value(self.model.optimizer.lr, new_lr)
            if self.verbose > 0:
              print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                    'rate to %s.' % (epoch + 1, new_lr))
            self.cooldown_counter = self.cooldown
            self.wait = 0

  def in_cooldown(self):
    # [FUNCTION DOC]
    # Returns true while the cooldown counter is still above zero, i.e. the callback is waiting
    # after a recent learning rate reduction.
    return self.cooldown_counter > 0


class CSVLogger(Callback):
  # [CLASS DOC]
  # Callback that writes one row per epoch (epoch number plus all logged values, sorted by key)
  # to a CSV file, with a header row, using a configurable separator and either appending to or
  # overwriting an existing file. State: separator, file name, append flag, the open file, the
  # CSV writer, the sorted column keys and whether a header must still be written.
  """Callback that streams epoch results to a CSV file.

  Supports all values that can be represented as a string,
  including 1D iterables such as `np.ndarray`.

  Example:

  ```python
  csv_logger = CSVLogger('training.log')
  model.fit(X_train, Y_train, callbacks=[csv_logger])
  ```

  Args:
      filename: Filename of the CSV file, e.g. `'run/log.csv'`.
      separator: String used to separate elements in the CSV file.
      append: Boolean. True: append if file exists (useful for continuing
          training). False: overwrite existing file.
  """

  def __init__(self, filename, separator=',', append=False):
    # [FUNCTION DOC]
    # Stores the settings: file name (string or path-like), element separator (default comma) and
    # whether to append (default false: overwrite). Initializes the writer and keys to nothing and
    # marks that a header is needed. Calls the base constructor last. Returns nothing.
    self.sep = separator
    self.filename = path_to_string(filename)
    self.append = append
    self.writer = None
    self.keys = None
    self.append_header = True
    super(CSVLogger, self).__init__()

  def on_train_begin(self, logs=None):
    # [FUNCTION DOC]
    # Opens the CSV file: in append mode, if the file already exists with content, the header is
    # not written again (the first line is checked), and the file is opened for appending;
    # otherwise it is opened for writing (truncating). Side effect: opens a file handle that is
    # closed at train end. Returns nothing.
    if self.append:
      if file_io.file_exists_v2(self.filename):
        with gfile.GFile(self.filename, 'r') as f:
          self.append_header = not bool(len(f.readline()))
      mode = 'a'
    else:
      mode = 'w'
    self.csv_file = gfile.GFile(self.filename, mode)

  def on_epoch_end(self, epoch, logs=None):
    # [FUNCTION DOC]
    # Writes one CSV row for the epoch. On first call fixes the column keys as the sorted log
    # keys. If the model has been told to stop training, missing keys are filled with the text NA
    # so the last row stays parseable. On first call also creates the dictionary-style CSV writer
    # (with the epoch column first and the configured separator) and writes the header when
    # needed. Each value is converted by a helper (strings kept, non-scalar iterables rendered as
    # a quoted bracketed list, others unchanged). Flushes the file after the row. Returns nothing.
    logs = logs or {}

    def handle_value(k):
      # [FUNCTION DOC]
      # HELPER: formats one logged value for the CSV. Strings and zero-dimensional numpy arrays
      # and other non-iterables are returned unchanged; any other iterable (list, array) is turned
      # into a quoted, bracketed, comma-and-space separated text of its elements.
      is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
      if isinstance(k, str):
        return k
      elif isinstance(k, collections.abc.Iterable) and not is_zero_dim_ndarray:
        return '"[%s]"' % (', '.join(map(str, k)))
      else:
        return k

    if self.keys is None:
      self.keys = sorted(logs.keys())

    if self.model.stop_training:
      # We set NA so that csv parsers do not fail for this last epoch.
      logs = dict((k, logs[k]) if k in logs else (k, 'NA') for k in self.keys)

    if not self.writer:

      class CustomDialect(csv.excel):
        # [CLASS DOC]
        # HELPER: local CSV dialect derived from the standard spreadsheet dialect whose only
        # change is the field delimiter, set to the callback's configured separator.
        delimiter = self.sep

      fieldnames = ['epoch'] + self.keys

      self.writer = csv.DictWriter(
          self.csv_file,
          fieldnames=fieldnames,
          dialect=CustomDialect)
      if self.append_header:
        self.writer.writeheader()

    row_dict = collections.OrderedDict({'epoch': epoch})
    row_dict.update((key, handle_value(logs[key])) for key in self.keys)
    self.writer.writerow(row_dict)
    self.csv_file.flush()

  def on_train_end(self, logs=None):
    # [FUNCTION DOC]
    # Closes the CSV file opened at train begin and clears the writer so a new one is created if
    # the callback is reused. Returns nothing.
    self.csv_file.close()
    self.writer = None


class LambdaCallback(Callback):
  # [CLASS DOC]
  # Callback built from plain user functions. Each of the six supported hooks (epoch begin/end,
  # batch begin/end, train begin/end) is replaced on the instance by the supplied function, or by
  # a function doing nothing when none is given. Extra keyword arguments become attributes of the
  # instance. The functions must accept positional arguments matching the hook (epoch or batch
  # index and logs, or just logs for the train hooks).
  r"""Callback for creating simple, custom callbacks on-the-fly.

  This callback is constructed with anonymous functions that will be called
  at the appropriate time (during `Model.{fit | evaluate | predict}`).
  Note that the callbacks expects positional arguments, as:

  - `on_epoch_begin` and `on_epoch_end` expect two positional arguments:
    `epoch`, `logs`
  - `on_batch_begin` and `on_batch_end` expect two positional arguments:
    `batch`, `logs`
  - `on_train_begin` and `on_train_end` expect one positional argument:
    `logs`

  Args:
      on_epoch_begin: called at the beginning of every epoch.
      on_epoch_end: called at the end of every epoch.
      on_batch_begin: called at the beginning of every batch.
      on_batch_end: called at the end of every batch.
      on_train_begin: called at the beginning of model training.
      on_train_end: called at the end of model training.

  Example:

  ```python
  # Print the batch number at the beginning of every batch.
  batch_print_callback = LambdaCallback(
      on_batch_begin=lambda batch,logs: print(batch))

  # Stream the epoch loss to a file in JSON format. The file content
  # is not well-formed JSON but rather has a JSON object per line.
  import json
  json_log = open('loss_log.json', mode='wt', buffering=1)
  json_logging_callback = LambdaCallback(
      on_epoch_end=lambda epoch, logs: json_log.write(
          json.dumps({'epoch': epoch, 'loss': logs['loss']}) + '\n'),
      on_train_end=lambda logs: json_log.close()
  )

  # Terminate some processes after having finished model training.
  processes = ...
  cleanup_callback = LambdaCallback(
      on_train_end=lambda logs: [
          p.terminate() for p in processes if p.is_alive()])

  model.fit(...,
            callbacks=[batch_print_callback,
                       json_logging_callback,
                       cleanup_callback])
  ```
  """

  def __init__(self,
               on_epoch_begin=None,
               on_epoch_end=None,
               on_batch_begin=None,
               on_batch_end=None,
               on_train_begin=None,
               on_train_end=None,
               **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the instance attributes for the hooks are replaced with the supplied (or
    # do-nothing) functions, so the callback container's check for overridden batch hooks sees them
    # as always implemented, and the extra keyword arguments are copied straight into the instance
    # attributes without any validation (they could overwrite existing attributes).
    # Sets up the callback: calls the base constructor, copies extra keyword arguments into the
    # instance attributes, then assigns each of the six optional hook functions (each defaults to
    # nothing, in which case a function that does nothing is used). Returns nothing.
    super(LambdaCallback, self).__init__()
    self.__dict__.update(kwargs)
    if on_epoch_begin is not None:
      self.on_epoch_begin = on_epoch_begin
    else:
      self.on_epoch_begin = lambda epoch, logs: None
    if on_epoch_end is not None:
      self.on_epoch_end = on_epoch_end
    else:
      self.on_epoch_end = lambda epoch, logs: None
    if on_batch_begin is not None:
      self.on_batch_begin = on_batch_begin
    else:
      self.on_batch_begin = lambda batch, logs: None
    if on_batch_end is not None:
      self.on_batch_end = on_batch_end
    else:
      self.on_batch_end = lambda batch, logs: None
    if on_train_begin is not None:
      self.on_train_begin = on_train_begin
    else:
      self.on_train_begin = lambda logs: None
    if on_train_end is not None:
      self.on_train_end = on_train_end
    else:
      self.on_train_end = lambda logs: None
