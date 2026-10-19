# [MODULE DOC]
# Legacy (v1, graph-mode capable) implementation of the Keras Model training engine.
# It defines a Model subclass of the newer training Model that keeps the old compile/fit/evaluate/
# predict machinery: per-output training endpoints, graph-built train/test/predict functions
# created lazily, Session-style keyword arguments passed to compile, and several training loops
# (array, generator, dataset/iterator, distribution-strategy) chosen from the input type.
# It also defines the training-endpoint helper classes that bundle one model output with its loss,
# target, sample weights and metrics, a callback model wrapper used with distribution strategies,
# and module-level helpers for scipy sparse conversion, symbolic-tensor checks, metric discovery
# from layers and constant-value extraction.
# Import-time side effects: tries to import the optional scipy sparse check and falls back
# to nothing when scipy is missing; no registrations or flags are created.
# This is the V1 counterpart of the main training module, used when the model was built under
# the old graph semantics.
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
"""V1 Training-related part of the Keras engine."""

import collections
import warnings

import numpy as np

from tensorflow.python import tf2
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import parameter_server_strategy
from tensorflow.python.distribute import parameter_server_strategy_v2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.keras import backend
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.distribute import distributed_training_utils
from tensorflow.python.keras.distribute import distributed_training_utils_v1
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import training as training_lib
from tensorflow.python.keras.engine import training_arrays_v1
from tensorflow.python.keras.engine import training_distributed_v1
from tensorflow.python.keras.engine import training_eager_v1
from tensorflow.python.keras.engine import training_generator_v1
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.engine import training_utils_v1
from tensorflow.python.keras.mixed_precision import loss_scale_optimizer
from tensorflow.python.keras.mixed_precision import policy
from tensorflow.python.keras.optimizer_v2 import optimizer_v2
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import model_serialization
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils.mode_keys import ModeKeys
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import data as data_types
from tensorflow.python.util import nest

try:
  from scipy.sparse import issparse  # pylint: disable=g-import-not-at-top
except ImportError:
  issparse = None


class Model(training_lib.Model):
  # [CLASS DOC]
  # Legacy v1 Keras Model that groups layers into an object with training and inference features.
  # It extends the current training Model and overrides compile, fit, evaluate, predict, the
  # on-batch methods and weight handling to use the old graph-based engine. State kept on an
  # instance includes the distribution strategy (explicit or compile-time), the run-eagerly flag,
  # a flag saying whether the experimental tf.function based loop is used, the Session keyword
  # arguments given to compile, the list of training endpoints (one per output), loss functions,
  # total loss, per-output metric information and lazily built train/test/predict functions.
  # Instances are created like any Keras model (functional API or subclassing) and then compiled
  # before training. Training dispatches to a loop object chosen from the input data type.
  """`Model` groups layers into an object with training and inference features.

  There are two ways to instantiate a `Model`:

  1 - With the "functional API", where you start from `Input`,
  you chain layer calls to specify the model's forward pass,
  and finally you create your model from inputs and outputs:

  ```python
  import tensorflow as tf

  inputs = tf.keras.Input(shape=(3,))
  x = tf.keras.layers.Dense(4, activation=tf.nn.relu)(inputs)
  outputs = tf.keras.layers.Dense(5, activation=tf.nn.softmax)(x)
  model = tf.keras.Model(inputs=inputs, outputs=outputs)
  ```

  2 - By subclassing the `Model` class: in that case, you should define your
  layers in `__init__` and you should implement the model's forward pass
  in `call`.

  ```python
  import tensorflow as tf

  class MyModel(tf.keras.Model):

    def __init__(self):
      super(MyModel, self).__init__()
      self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
      self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

    def call(self, inputs):
      x = self.dense1(inputs)
      return self.dense2(x)

  model = MyModel()
  ```

  If you subclass `Model`, you can optionally have
  a `training` argument (boolean) in `call`, which you can use to specify
  a different behavior in training and inference:

  ```python
  import tensorflow as tf

  class MyModel(tf.keras.Model):

    def __init__(self):
      super(MyModel, self).__init__()
      self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
      self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
      self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, inputs, training=False):
      x = self.dense1(inputs)
      if training:
        x = self.dropout(x, training=training)
      return self.dense2(x)

  model = MyModel()
  ```
  """

  def __init__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Forwards all positional and keyword arguments to the parent Model constructor,
    # then initializes the v1-specific state: no distribution strategy, no compile-time strategy
    # (the current strategy is recorded as compile-time strategy when running eagerly outside
    # functions inside a strategy scope), the deprecated compile-distribution flag off, no explicit
    # run-eagerly choice, the flag for the tf.function based training loop (on only when eager
    # execution is enabled outside functions), and a flag recording that v1 compile has not
    # been called. Returns nothing.
    super(Model, self).__init__(*args, **kwargs)
    # initializing _distribution_strategy here since it is possible to call
    # predict on a model without compiling it.
    self._distribution_strategy = None
    self._compile_time_distribution_strategy = None
    if (ops.executing_eagerly_outside_functions() and
        distribute_lib.has_strategy()):
      self._set_strategy(
          distribute_lib.get_strategy())

    # This flag is used to track if the user is using the deprecated path of
    # passing distribution strategy to compile rather than creating the model
    # under distribution strategy scope.
    self._compile_distribution = False

    self._run_eagerly = None
    self._experimental_run_tf_function = (
        ops.executing_eagerly_outside_functions())

    self._v1_compile_was_called = False

  def _init_batch_counters(self):
    # [FUNCTION DOC]
    # HELPER: Override used by the parent Model constructor to set up counters.
    # Intentionally does nothing: the v1 graph-mode engine must not create batch counters.
    # Takes no inputs, returns nothing, no side effects.
    pass  # Batch counters should not be created in legacy graph mode.

  @trackable.no_automatic_dependency_tracking
  def _set_strategy(self, strategy):
    # [FUNCTION DOC]
    # HELPER: Records the distribution strategy active when the model was created.
    # Stores the given strategy object as the compile-time distribution strategy of the model.
    # Automatic dependency tracking is disabled for this method so the assignment does not
    # create a checkpoint dependency. Returns nothing.
    self._compile_time_distribution_strategy = strategy

  def get_weights(self):
    # [FUNCTION DOC]
    # Returns all model weights as a flat list of Numpy arrays, taken through the base layer
    # implementation. If a distribution strategy (the model's own or the compile-time one) is
    # set, the read happens inside that strategy's scope so that distributed variables are
    # read correctly. No arguments, no exceptions of its own, no mutation.
    """Retrieves the weights of the model.

    Returns:
        A flat list of Numpy arrays.
    """
    strategy = (self._distribution_strategy or
                self._compile_time_distribution_strategy)
    if strategy:
      with strategy.scope():
        return base_layer.Layer.get_weights(self)
    return base_layer.Layer.get_weights(self)

  def load_weights(self, filepath, by_name=False, skip_mismatch=False):
    # [FUNCTION DOC]
    # Loads weights from a TensorFlow checkpoint prefix or an HDF5 file by delegating to the
    # parent implementation. Inputs: filepath (string path or checkpoint prefix), by_name (load by
    # layer name instead of topological order, default off), skip_mismatch (skip layers whose
    # weights mismatch, only valid with by_name, default off). Before delegating, if the model uses
    # a TPU strategy with more than one step per run and the path is not an HDF5 file, raises a
    # value error because that combination is unsupported. Returns whatever the parent returns
    # (a restore status for checkpoint format, nothing for HDF5). Side effect: assigns weights.
    """Loads all layer weights, either from a TensorFlow or an HDF5 weight file.

    If `by_name` is False weights are loaded based on the network's
    topology. This means the architecture should be the same as when the weights
    were saved.  Note that layers that don't have weights are not taken into
    account in the topological ordering, so adding or removing layers is fine as
    long as they don't have weights.

    If `by_name` is True, weights are loaded into layers only if they share the
    same name. This is useful for fine-tuning or transfer-learning models where
    some of the layers have changed.

    Only topological loading (`by_name=False`) is supported when loading weights
    from the TensorFlow format. Note that topological loading differs slightly
    between TensorFlow and HDF5 formats for user-defined classes inheriting from
    `tf.keras.Model`: HDF5 loads based on a flattened list of weights, while the
    TensorFlow format loads based on the object-local names of attributes to
    which layers are assigned in the `Model`'s constructor.

    Args:
        filepath: String, path to the weights file to load. For weight files in
            TensorFlow format, this is the file prefix (the same as was passed
            to `save_weights`).
        by_name: Boolean, whether to load weights by name or by topological
            order. Only topological loading is supported for weight files in
            TensorFlow format.
        skip_mismatch: Boolean, whether to skip loading of layers where there is
            a mismatch in the number of weights, or a mismatch in the shape of
            the weight (only valid when `by_name=True`).

    Returns:
        When loading a weight file in TensorFlow format, returns the same status
        object as `tf.train.Checkpoint.restore`. When graph building, restore
        ops are run automatically as soon as the network is built (on first call
        for user-defined classes inheriting from `Model`, immediately if it is
        already built).

        When loading weights in HDF5 format, returns `None`.

    Raises:
        ImportError: If h5py is not available and the weight file is in HDF5
            format.
        ValueError: If `skip_mismatch` is set to `True` when `by_name` is
          `False`.
    """
    if backend.is_tpu_strategy(self._distribution_strategy):
      if (self._distribution_strategy.extended.steps_per_run > 1 and
          (not saving_utils.is_hdf5_filepath(filepath))):  # pylint: disable=protected-access
        raise ValueError('Load weights is not yet supported with TPUStrategy '
                         'with steps_per_run greater than 1.')
    return super(Model, self).load_weights(filepath, by_name, skip_mismatch)

  @trackable.no_automatic_dependency_tracking
  def compile(self,
              optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None,
              sample_weight_mode=None,
              weighted_metrics=None,
              target_tensors=None,
              distribute=None,
              **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring says only a value error is raised, but the code also raises a type
    # error for unknown keyword arguments and not-implemented errors for parameter-server
    # strategies. The docstring also says loss_weights, when a tensor, maps output names to
    # coefficients, which should say dictionary. Some value-error messages are built from
    # several separate arguments instead of one formatted string.
    # Configures the model for training (v1 graph/eager hybrid compile). Inputs: optimizer (name or
    # optimizer instance, or a list of them), loss (name, callable, loss object, or list/mapping
    # per output), metrics, loss_weights, sample_weight_mode, weighted_metrics, target_tensors
    # (own target tensors instead of placeholders), distribute (deprecated explicit strategy),
    # and extra keyword arguments: the recognized ones are run_eagerly, the tf.function loop flag,
    # a serialized-model flag, a legacy cloning flag (ignored) and the Session arguments
    # feed_dict, fetches, options and run_metadata; any other keyword raises a type error.
    # Session arguments disable the tf.function loop and are rejected when running eagerly.
    # Raises value errors for v1 optimizers with eager execution, the distribute argument under
    # TF2 or the tf.function loop, target tensors with eager running, and variables not created
    # in the strategy scope; raises not-implemented errors for parameter server strategies.
    # Side effects: sets optimizer, loss, loss weights, metric lists, empties the training
    # endpoint list and distributed caches, clears eager losses, may create a distributed session,
    # builds loss functions, endpoints, metrics and loss graphs and resets train/test/predict
    # functions to be built lazily. If the model is not yet built (unknown inputs/outputs) it
    # returns early and compilation completes at the first call on data. Returns nothing.
    """Configures the model for training.

    Args:
        optimizer: String (name of optimizer) or optimizer instance.
            See `tf.keras.optimizers`.
        loss: String (name of objective function), objective function or
            `tf.keras.losses.Loss` instance. See `tf.keras.losses`. An objective
            function is any callable with the signature
            `scalar_loss = fn(y_true, y_pred)`. If the model has multiple
            outputs, you can use a different loss on each output by passing a
            dictionary or a list of losses. The loss value that will be
            minimized by the model will then be the sum of all individual
            losses.
        metrics: List of metrics to be evaluated by the model during training
            and testing. Typically you will use `metrics=['accuracy']`.
            To specify different metrics for different outputs of a
            multi-output model, you could also pass a dictionary, such as
            `metrics={'output_a': 'accuracy', 'output_b': ['accuracy', 'mse']}`.
            You can also pass a list (len = len(outputs)) of lists of metrics
            such as `metrics=[['accuracy'], ['accuracy', 'mse']]` or
            `metrics=['accuracy', ['accuracy', 'mse']]`.
        loss_weights: Optional list or dictionary specifying scalar
            coefficients (Python floats) to weight the loss contributions
            of different model outputs.
            The loss value that will be minimized by the model
            will then be the *weighted sum* of all individual losses,
            weighted by the `loss_weights` coefficients.
            If a list, it is expected to have a 1:1 mapping
            to the model's outputs. If a tensor, it is expected to map
            output names (strings) to scalar coefficients.
        sample_weight_mode: If you need to do timestep-wise
            sample weighting (2D weights), set this to `"temporal"`.
            `None` defaults to sample-wise weights (1D).
            If the model has multiple outputs, you can use a different
            `sample_weight_mode` on each output by passing a
            dictionary or a list of modes.
        weighted_metrics: List of metrics to be evaluated and weighted
            by sample_weight or class_weight during training and testing.
        target_tensors: By default, Keras will create placeholders for the
            model's target, which will be fed with the target data during
            training. If instead you would like to use your own
            target tensors (in turn, Keras will not expect external
            Numpy data for these targets at training time), you
            can specify them via the `target_tensors` argument. It can be
            a single tensor (for a single-output model), a list of tensors,
            or a dict mapping output names to target tensors.
        distribute: NOT SUPPORTED IN TF 2.0, please create and compile the
            model under distribution strategy scope instead of passing it to
            compile.
        **kwargs: Any additional arguments.

    Raises:
        ValueError: In case of invalid arguments for
            `optimizer`, `loss`, `metrics` or `sample_weight_mode`.
    """
    self._assert_built_as_v1()
    self._run_eagerly = kwargs.pop('run_eagerly', None)
    self._experimental_run_tf_function = kwargs.pop(
        'experimental_run_tf_function', True)
    self._v1_compile_was_called = True

    # Prepare Session arguments (legacy).
    kwargs.pop('cloning', None)  # Legacy DistStrat argument, never used.
    self._from_serialized = kwargs.pop('from_serialized', False)
    allowed_kwargs = {'feed_dict', 'fetches', 'options', 'run_metadata'}
    unknown_kwargs = set(kwargs.keys()) - allowed_kwargs
    if unknown_kwargs:
      raise TypeError(
          'Invalid keyword argument(s) in `compile`: %s' % (unknown_kwargs,))
    self._function_kwargs = kwargs
    if self._function_kwargs:
      self._experimental_run_tf_function = False
      if self.run_eagerly:
        raise ValueError(
            'Session keyword arguments are not supported '
            'when `run_eagerly=True`. You passed the following '
            'Session arguments: %s' % (self._function_kwargs,))

    self._set_optimizer(optimizer)
    is_any_keras_optimizer_v1 = any(
        (isinstance(opt, optimizer_v1.Optimizer)
         and not isinstance(opt, optimizer_v1.TFOptimizer)
        ) for opt in nest.flatten(self.optimizer))

    if is_any_keras_optimizer_v1 and ops.executing_eagerly_outside_functions():
      raise ValueError('`tf.compat.v1.keras` Optimizer (', optimizer, ') is '
                       'not supported when eager execution is enabled. Use a '
                       '`tf.keras` Optimizer instead, or disable eager '
                       'execution.')

    if ((target_tensors is not None)
        or not ops.executing_eagerly_outside_functions()):
      # Fallback out of things that aren't supported with v2 loops
      self._experimental_run_tf_function = False

    if distribute is not None:
      if tf2.enabled() or self._experimental_run_tf_function:
        raise ValueError(
            'Distribute argument in compile is not available in TF 2.0 please '
            'create the model under the distribution strategy scope.')
      logging.warning('Distribute argument in compile is deprecated please '
                      'create the model under the distribution strategy scope.')
      self._distribution_strategy = distribute
      self._compile_distribution = True
    else:
      if distribute_lib.has_strategy():
        # When the user builds the model in the DS scope and cross replica
        # context we want distribution strategy to be set but when building the
        # replica copies of the models internally we should not be compiling
        # with distribution strategy and use the default compilation path.
        if distribute_lib.in_cross_replica_context():
          self._distribution_strategy = (
              distribute_lib.get_strategy())

    if isinstance(self._distribution_strategy,
                  parameter_server_strategy.ParameterServerStrategyV1):
      raise NotImplementedError(
          '`tf.compat.v1.distribute.experimental.ParameterServerStrategy` '
          'currently only works with the deprecated tf.Estimator API')

    if isinstance(self._distribution_strategy,
                  parameter_server_strategy_v2.ParameterServerStrategyV2):
      raise NotImplementedError(
          '`tf.distribute.experimental.ParameterServerStrategy` is only '
          'supported in TF2.')

    if not self._experimental_run_tf_function:
      self._validate_compile_param_for_distribution_strategy(self.run_eagerly,
                                                             sample_weight_mode,
                                                             target_tensors,
                                                             weighted_metrics)
    # We've disabled automatic dependency tracking for this method, but do want
    # to add a checkpoint dependency on the optimizer if it's trackable.
    if isinstance(self.optimizer, trackable.Trackable):
      self._track_trackable(
          self.optimizer, name='optimizer', overwrite=True)
    self.loss = loss or {}
    self.loss_weights = loss_weights
    self.sample_weight_mode = sample_weight_mode
    self._compile_metrics = metrics or []
    self._compile_weighted_metrics = weighted_metrics
    if self.run_eagerly and target_tensors is not None:
      raise ValueError(
          'target_tensors argument is not supported when '
          'running a model eagerly.')

    # _training_endpoints contains a list of _TrainingEndpoint object, which has
    # all the model output/target/loss and related metadata.
    self._training_endpoints = []

    # Used to freeze the behavior of the Model once `compile` has been called.
    self._compiled_trainable_state = self._get_trainable_state()

    # Set tf.distribute.Strategy specific parameters.
    self._distributed_model_cache = {}
    self._distributed_function_cache = {}

    # Clear any `_eager_losses` that was added.
    self._clear_losses()

    if (not context.executing_eagerly() and
        self._distribution_strategy is not None):
      # Ensures a Session is created and configured correctly for Distribution
      # Strategy.
      backend.configure_and_create_distributed_session(
          self._distribution_strategy)
    # Initialize model metric attributes.
    self._init_metric_attributes()
    if not self.built or not self.inputs or not self.outputs:
      # Model is not compilable because it does not know its number of inputs
      # and outputs, nor their shapes and names. We will compile after the first
      # time the model gets called on training data.
      return
    self._is_compiled = True

    # Prepare list of loss functions, same size of model outputs.
    self.loss_functions = training_utils_v1.prepare_loss_functions(
        self.loss, self.output_names)

    target_tensors = self._process_target_tensor_for_compile(target_tensors)

    for o, n, l, t in zip(self.outputs, self.output_names,
                          self.loss_functions, target_tensors):
      endpoint = _TrainingEndpoint(o, n, l)
      endpoint.create_training_target(t, run_eagerly=self.run_eagerly)
      self._training_endpoints.append(endpoint)

    # Prepare list loss weights, same size of model outputs.
    training_utils_v1.prepare_loss_weights(self._training_endpoints,
                                           loss_weights)

    # Initialization for Eager mode execution.
    if self.run_eagerly:
      self._compile_eagerly(metrics, weighted_metrics, sample_weight_mode)
      return

    with backend.get_graph().as_default():
      # Save all metric attributes per output of the model.
      self._cache_output_metric_attributes(metrics, weighted_metrics)

      # Set metric attributes on model.
      self._set_metric_attributes()

      # Invoke metric functions (unweighted) for all the outputs.
      self._handle_metrics(
          self.outputs,
          targets=self._targets,
          skip_target_masks=self._prepare_skip_target_masks(),
          masks=self._prepare_output_masks())

      # Prepare sample weight modes. List with the same length as model outputs.
      training_utils_v1.prepare_sample_weight_modes(
          self._training_endpoints, sample_weight_mode)

      # Creates the model loss and weighted metrics sub-graphs.
      self._compile_weights_loss_and_weighted_metrics()

      # Functions for train, test and predict will
      # be compiled lazily when required.
      # This saves time when the user is not using all functions.
      self.train_function = None
      self.test_function = None
      self.predict_function = None

      # Collected trainable weights, sorted in topological order.
      self._collected_trainable_weights = self.trainable_weights

      # Validate all variables were correctly created in distribution scope.
      if self._distribution_strategy and not self._compile_distribution:
        for v in self.variables:
          strategy = self._distribution_strategy
          if not strategy.extended.variable_created_in_scope(v):
            raise ValueError(
                'Variable (%s) was not created in the distribution strategy '
                'scope of (%s). It is most likely due to not all layers or '
                'the model or optimizer being created outside the distribution '
                'strategy scope. Try to make sure your code looks similar '
                'to the following.\n'
                'with strategy.scope():\n'
                '  model=_create_model()\n'
                '  model.compile(...)'% (v, strategy))

  @trackable.no_automatic_dependency_tracking
  def _init_distributed_function_cache_if_not_compiled(self):
    # [FUNCTION DOC]
    # HELPER: Lazy initializer for the distributed function cache.
    # Creates an empty mapping for the distributed function cache if the attribute does not exist
    # yet (for example when predict is used on an uncompiled model). Returns nothing.
    if not hasattr(self, '_distributed_function_cache'):
      self._distributed_function_cache = {}

  @property
  def metrics(self):
    # [FUNCTION DOC]
    # Property returning the list of all metric objects of the model: the metrics created by
    # compile (only if compiled), metrics added with the add-metric API, and metrics of the
    # direct sublayers. If the model was created as a v2 model and converted (v1 compile marker
    # missing) it falls back to the parent implementation. Returns a new list each time.
    """Returns the model's metrics added using `compile`, `add_metric` APIs."""
    metrics = []
    if self._is_compiled:
      if not hasattr(self, '_v1_compile_was_called'):
        # See b/155687393 for more details, the model is created as a v2
        # instance but converted to v1. Fallback to use base Model to retrieve
        # the metrics.
        return super(Model, self).metrics
      metrics += self._compile_metric_functions
    metrics.extend(self._metrics)
    metrics.extend(
        _get_metrics_from_layers(
            list(self._flatten_layers(include_self=False, recursive=False))))
    return metrics

  @property
  def metrics_names(self):
    # [FUNCTION DOC]
    # Property returning the display names of all outputs as a list of strings: always starts with
    # the overall loss name, then (when compiled and there are several outputs) the per-output loss
    # names for outputs that are not skipped, then the names of all metrics. Falls back to the
    # parent implementation for converted v2 models lacking the v1 compile marker.
    """Returns the model's display labels for all outputs."""

    # This property includes all output names including `loss` and per-output
    # losses for backward compatibility.
    metrics_names = ['loss']
    if self._is_compiled:
      if not hasattr(self, '_v1_compile_was_called'):
        # See b/155687393 for more details, the model is created as a v2
        # instance but converted to v1. Fallback to use base Model to retrieve
        # the metrics name
        return super(Model, self).metrics_names

      # Add output loss metric names to the metric names list.
      if len(self._training_endpoints) > 1:
        metrics_names.extend([
            e.loss_name()
            for e in self._training_endpoints
            if not e.should_skip_target()
        ])

    # Add all metric names.
    metrics_names += [m.name for m in self.metrics]
    return metrics_names

  @property
  def run_eagerly(self):
    # [FUNCTION DOC]
    # Property getter that decides whether the model runs step by step eagerly. Raises a value
    # error if eager running was requested but eager execution is not enabled. For models that are
    # not dynamic it returns the explicit setting, or, when none was set, the global run-functions-
    # eagerly configuration. For dynamic models (layers that need eager execution) it raises a
    # value error when eager execution is off or when eager running was explicitly disabled, and
    # otherwise returns true. Returns a boolean.
    """Settable attribute indicating whether the model should run eagerly.

    Running eagerly means that your model will be run step by step,
    like Python code. Your model might run slower, but it should become easier
    for you to debug it by stepping into individual layer calls.

    By default, we will attempt to compile your model to a static graph to
    deliver the best execution performance.

    Returns:
      Boolean, whether the model should run eagerly.
    """
    if self._run_eagerly is True and not context.executing_eagerly():
      raise ValueError('You can only set `run_eagerly=True` if eager execution '
                       'is enabled.')
    if not self.dynamic:
      if self._run_eagerly is None:
        # Respect `tf.config.run_functions_eagerly` unless
        # `run_eagerly` was explicitly passed to `compile`.
        return def_function.functions_run_eagerly()
      else:
        return self._run_eagerly
    else:
      if not context.executing_eagerly():
        raise ValueError('Your model contains layers that can only be '
                         'successfully run in eager execution (layers '
                         'constructed with `dynamic=True`). '
                         'You must enable eager execution with '
                         '`tf.enable_eager_execution()`.')
      if self._run_eagerly is False:
        # TODO(fchollet): consider using py_func to enable this.
        raise ValueError('Your model contains layers that can only be '
                         'successfully run in eager execution (layers '
                         'constructed with `dynamic=True`). '
                         'You cannot set `run_eagerly=False`.')
      return context.executing_eagerly()

  @run_eagerly.setter
  def run_eagerly(self, value):
    # [FUNCTION DOC]
    # Property setter: stores the given value (true, false or nothing for "use the global default")
    # as the explicit run-eagerly choice. No validation happens here; validation is deferred to
    # the getter. Returns nothing.
    self._run_eagerly = value

  def _select_training_loop(self, inputs):
    # [FUNCTION DOC]
    # HELPER: Dispatcher used by fit, evaluate and predict to choose the loop implementation.
    # KNOWN ISSUE: iterator inputs are rejected up front, so the later check for eager datasets
    # or iterators can in practice only match datasets; its iterator half is unreachable.
    # Input: the user data. Raises a value error if the data is a manually created dataset
    # iterator object (users must pass the dataset itself). Returns a loop object: with a
    # distribution strategy, the single-worker distributed loop (wrapped in the multi-worker loop
    # in multi-worker mode); otherwise the generator/sequence loop for generators and sequences,
    # the eager dataset/iterator loop for eager datasets; otherwise the generator-like loop when
    # running eagerly, else the array-like (graph) loop.
    """Select training loop for fit/eval/predict based on the inputs."""
    # TODO(kaftan) or TODO(scottzhu): This check should eventually be nicely
    #  integrated into the data adapters in the v2 loop. We can't do this yet
    #  because we currently have to fall back for unhandled data types.
    if isinstance(inputs, (iterator_ops.Iterator,
                           iterator_ops.IteratorBase)):
      raise ValueError('For performance reasons Keras `fit`, `evaluate` and'
                       '`predict` accept tf.data `Datasets` as input but not '
                       'iterators that have been manually generated from '
                       'Datasets by users. Please directly pass in the '
                       'original `Dataset` object instead of passing in '
                       '`iter(dataset)`.')

    # Case 1: distribution strategy.
    if self._distribution_strategy:
      if self._in_multi_worker_mode():
        return training_distributed_v1.DistributionMultiWorkerTrainingLoop(
            training_distributed_v1.DistributionSingleWorkerTrainingLoop())
      else:
        return training_distributed_v1.DistributionSingleWorkerTrainingLoop()

    # Case 2: generator-like. Input is Python generator, or Sequence object,
    # or a non-distributed Dataset or iterator in eager execution.
    if data_utils.is_generator_or_sequence(inputs):
      return training_generator_v1.GeneratorOrSequenceTrainingLoop()
    if training_utils_v1.is_eager_dataset_or_iterator(inputs):
      return training_generator_v1.EagerDatasetOrIteratorTrainingLoop()

    # Case 3: Symbolic tensors or Numpy array-like.
    # This includes Datasets and iterators in graph mode (since they
    # generate symbolic tensors).
    if self.run_eagerly:
      return training_generator_v1.GeneratorLikeTrainingLoop()
    else:
      return training_arrays_v1.ArrayLikeTrainingLoop()

  def fit(self,
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
          max_queue_size=10,
          workers=1,
          use_multiprocessing=False,
          **kwargs):
    # [FUNCTION DOC]
    # Trains the model for a fixed number of epochs by delegating to the loop selected for the
    # input type. Inputs: x and y (training data and targets in any supported form), batch_size,
    # epochs, verbose, callbacks, validation_split, validation_data, shuffle, class_weight,
    # sample_weight, initial_epoch, steps_per_epoch, validation_steps, validation_freq and the
    # generator options max_queue_size, workers, use_multiprocessing. The legacy keyword for epoch
    # count is accepted (with a warning) and any other extra keyword raises a type error.
    # Requires a model built as v1 and compiled (otherwise an error is raised by the assertion
    # helpers); the call arguments of the model's forward method are checked. Returns the history
    # object produced by the loop. Side effects: trains weights, runs callbacks, logs warnings.
    """Trains the model for a fixed number of epochs (iterations on a dataset).

    Args:
        x: Input data. It could be:
          - A Numpy array (or array-like), or a list of arrays
            (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
            (in case the model has multiple inputs).
          - A dict mapping input names to the corresponding array/tensors,
            if the model has named inputs.
          - A `tf.data` dataset. Should return a tuple
            of either `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
          - A generator or `keras.utils.Sequence` returning `(inputs, targets)`
            or `(inputs, targets, sample weights)`.
        y: Target data. Like the input data `x`,
          it could be either Numpy array(s) or TensorFlow tensor(s).
          It should be consistent with `x` (you cannot have Numpy inputs and
          tensor targets, or inversely). If `x` is a dataset, generator,
          or `keras.utils.Sequence` instance, `y` should
          not be specified (since targets will be obtained from `x`).
        batch_size: Integer or `None`.
            Number of samples per gradient update.
            If unspecified, `batch_size` will default to 32.
            Do not specify the `batch_size` if your data is in the
            form of symbolic tensors, datasets,
            generators, or `keras.utils.Sequence` instances (since they generate
            batches).
        epochs: Integer. Number of epochs to train the model.
            An epoch is an iteration over the entire `x` and `y`
            data provided.
            Note that in conjunction with `initial_epoch`,
            `epochs` is to be understood as "final epoch".
            The model is not trained for a number of iterations
            given by `epochs`, but merely until the epoch
            of index `epochs` is reached.
        verbose: 0, 1, or 2. Verbosity mode.
            0 = silent, 1 = progress bar, 2 = one line per epoch.
            Note that the progress bar is not particularly useful when
            logged to a file, so verbose=2 is recommended when not running
            interactively (eg, in a production environment).
        callbacks: List of `keras.callbacks.Callback` instances.
            List of callbacks to apply during training.
            See `tf.keras.callbacks`.
        validation_split: Float between 0 and 1.
            Fraction of the training data to be used as validation data.
            The model will set apart this fraction of the training data,
            will not train on it, and will evaluate
            the loss and any model metrics
            on this data at the end of each epoch.
            The validation data is selected from the last samples
            in the `x` and `y` data provided, before shuffling. This argument is
            not supported when `x` is a dataset, generator or
           `keras.utils.Sequence` instance.
        validation_data: Data on which to evaluate
            the loss and any model metrics at the end of each epoch.
            The model will not be trained on this data.
            `validation_data` will override `validation_split`.
            `validation_data` could be:
              - tuple `(x_val, y_val)` of Numpy arrays or tensors
              - tuple `(x_val, y_val, val_sample_weights)` of Numpy arrays
              - dataset
            For the first two cases, `batch_size` must be provided.
            For the last case, `validation_steps` could be provided.
        shuffle: Boolean (whether to shuffle the training data
            before each epoch) or str (for 'batch').
            'batch' is a special option for dealing with the
            limitations of HDF5 data; it shuffles in batch-sized chunks.
            Has no effect when `steps_per_epoch` is not `None`.
        class_weight: Optional dictionary mapping class indices (integers)
            to a weight (float) value, used for weighting the loss function
            (during training only).
            This can be useful to tell the model to
            "pay more attention" to samples from
            an under-represented class.
        sample_weight: Optional Numpy array of weights for
            the training samples, used for weighting the loss function
            (during training only). You can either pass a flat (1D)
            Numpy array with the same length as the input samples
            (1:1 mapping between weights and samples),
            or in the case of temporal data,
            you can pass a 2D array with shape
            `(samples, sequence_length)`,
            to apply a different weight to every timestep of every sample.
            In this case you should make sure to specify
            `sample_weight_mode="temporal"` in `compile()`. This argument is not
            supported when `x` is a dataset, generator, or
           `keras.utils.Sequence` instance, instead provide the sample_weights
            as the third element of `x`.
        initial_epoch: Integer.
            Epoch at which to start training
            (useful for resuming a previous training run).
        steps_per_epoch: Integer or `None`.
            Total number of steps (batches of samples)
            before declaring one epoch finished and starting the
            next epoch. When training with input tensors such as
            TensorFlow data tensors, the default `None` is equal to
            the number of samples in your dataset divided by
            the batch size, or 1 if that cannot be determined. If x is a
            `tf.data` dataset, and 'steps_per_epoch'
            is None, the epoch will run until the input dataset is exhausted.
            This argument is not supported with array inputs.
        validation_steps: Only relevant if `validation_data` is provided and
            is a `tf.data` dataset. Total number of steps (batches of
            samples) to draw before stopping when performing validation
            at the end of every epoch. If 'validation_steps' is None, validation
            will run until the `validation_data` dataset is exhausted. In the
            case of a infinite dataset, it will run into a infinite loop.
            If 'validation_steps' is specified and only part of the dataset
            will be consumed, the evaluation will start from the beginning of
            the dataset at each epoch. This ensures that the same validation
            samples are used every time.
        validation_freq: Only relevant if validation data is provided. Integer
            or `collections.abc.Container` instance (e.g. list, tuple, etc.).
            If an integer, specifies how many training epochs to run before a
            new validation run is performed, e.g. `validation_freq=2` runs
            validation every 2 epochs. If a Container, specifies the epochs on
            which to run validation, e.g. `validation_freq=[1, 2, 10]` runs
            validation at the end of the 1st, 2nd, and 10th epochs.
        max_queue_size: Integer. Used for generator or `keras.utils.Sequence`
            input only. Maximum size for the generator queue.
            If unspecified, `max_queue_size` will default to 10.
        workers: Integer. Used for generator or `keras.utils.Sequence` input
            only. Maximum number of processes to spin up
            when using process-based threading. If unspecified, `workers`
            will default to 1. If 0, will execute the generator on the main
            thread.
        use_multiprocessing: Boolean. Used for generator or
            `keras.utils.Sequence` input only. If `True`, use process-based
            threading. If unspecified, `use_multiprocessing` will default to
            `False`. Note that because this implementation relies on
            multiprocessing, you should not pass non-picklable arguments to
            the generator as they can't be passed easily to children processes.
        **kwargs: Used for backwards compatibility.

    Returns:
        A `History` object. Its `History.history` attribute is
        a record of training loss values and metrics values
        at successive epochs, as well as validation loss values
        and validation metrics values (if applicable).

    Raises:
        RuntimeError: If the model was never compiled.
        ValueError: In case of mismatch between the provided input data
            and what the model expects.
    """
    self._assert_built_as_v1()
    # Legacy support
    if 'nb_epoch' in kwargs:
      logging.warning(
          'The `nb_epoch` argument in `fit` has been renamed `epochs`.')
      epochs = kwargs.pop('nb_epoch')
    if kwargs:
      raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))
    self._assert_compile_was_called()
    self._check_call_args('fit')

    func = self._select_training_loop(x)
    return func.fit(
        self,
        x=x,
        y=y,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        validation_split=validation_split,
        validation_data=validation_data,
        shuffle=shuffle,
        class_weight=class_weight,
        sample_weight=sample_weight,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        validation_freq=validation_freq,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing)

  def evaluate(self,
               x=None,
               y=None,
               batch_size=None,
               verbose=1,
               sample_weight=None,
               steps=None,
               callbacks=None,
               max_queue_size=10,
               workers=1,
               use_multiprocessing=False):
    # [FUNCTION DOC]
    # Evaluates the model in test mode by delegating to the loop selected for the input type.
    # Inputs: x and y (data and targets), batch_size, verbose, sample_weight, steps, callbacks and
    # the generator options max_queue_size, workers, use_multiprocessing. Requires a model built
    # as v1 and compiled; checks the model's forward-call arguments. Returns the scalar loss or a
    # list of loss and metric values, as produced by the loop. Raises value errors for invalid
    # arguments (via the helpers and loops). No weights are changed; metrics state is updated.
    """Returns the loss value & metrics values for the model in test mode.

    Computation is done in batches (see the `batch_size` arg.)

    Args:
        x: Input data. It could be:
          - A Numpy array (or array-like), or a list of arrays
            (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
            (in case the model has multiple inputs).
          - A dict mapping input names to the corresponding array/tensors,
            if the model has named inputs.
          - A `tf.data` dataset.
          - A generator or `keras.utils.Sequence` instance.
        y: Target data. Like the input data `x`,
          it could be either Numpy array(s) or TensorFlow tensor(s).
          It should be consistent with `x` (you cannot have Numpy inputs and
          tensor targets, or inversely).
          If `x` is a dataset, generator or
          `keras.utils.Sequence` instance, `y` should not be specified (since
          targets will be obtained from the iterator/dataset).
        batch_size: Integer or `None`.
            Number of samples per batch of computation.
            If unspecified, `batch_size` will default to 32.
            Do not specify the `batch_size` if your data is in the
            form of symbolic tensors, dataset,
            generators, or `keras.utils.Sequence` instances (since they generate
            batches).
        verbose: 0 or 1. Verbosity mode.
            0 = silent, 1 = progress bar.
        sample_weight: Optional Numpy array of weights for
            the test samples, used for weighting the loss function.
            You can either pass a flat (1D)
            Numpy array with the same length as the input samples
            (1:1 mapping between weights and samples),
            or in the case of temporal data,
            you can pass a 2D array with shape
            `(samples, sequence_length)`,
            to apply a different weight to every timestep of every sample.
            In this case you should make sure to specify
            `sample_weight_mode="temporal"` in `compile()`. This argument is not
            supported when `x` is a dataset, instead pass
            sample weights as the third element of `x`.
        steps: Integer or `None`.
            Total number of steps (batches of samples)
            before declaring the evaluation round finished.
            Ignored with the default value of `None`.
            If x is a `tf.data` dataset and `steps` is
            None, 'evaluate' will run until the dataset is exhausted.
            This argument is not supported with array inputs.
        callbacks: List of `keras.callbacks.Callback` instances.
            List of callbacks to apply during evaluation.
            See [callbacks](/api_docs/python/tf/keras/callbacks).
        max_queue_size: Integer. Used for generator or `keras.utils.Sequence`
            input only. Maximum size for the generator queue.
            If unspecified, `max_queue_size` will default to 10.
        workers: Integer. Used for generator or `keras.utils.Sequence` input
            only. Maximum number of processes to spin up when using
            process-based threading. If unspecified, `workers` will default
            to 1. If 0, will execute the generator on the main thread.
        use_multiprocessing: Boolean. Used for generator or
            `keras.utils.Sequence` input only. If `True`, use process-based
            threading. If unspecified, `use_multiprocessing` will default to
            `False`. Note that because this implementation relies on
            multiprocessing, you should not pass non-picklable arguments to
            the generator as they can't be passed easily to children processes.

    Returns:
        Scalar test loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.

    Raises:
        ValueError: in case of invalid arguments.
    """
    self._assert_built_as_v1()
    self._assert_compile_was_called()
    self._check_call_args('evaluate')

    func = self._select_training_loop(x)
    return func.evaluate(
        self,
        x=x,
        y=y,
        batch_size=batch_size,
        verbose=verbose,
        sample_weight=sample_weight,
        steps=steps,
        callbacks=callbacks,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing)

  def predict(self,
              x,
              batch_size=None,
              verbose=0,
              steps=None,
              callbacks=None,
              max_queue_size=10,
              workers=1,
              use_multiprocessing=False):
    # [FUNCTION DOC]
    # Generates predictions for the input samples in batches by delegating to the loop selected
    # for the input type. Inputs: x (data in any supported form, required), batch_size, verbose,
    # steps, callbacks and generator options max_queue_size, workers, use_multiprocessing.
    # Requires a model built as v1 but, unlike fit and evaluate, does not require compilation.
    # Checks the model's forward-call arguments. Returns Numpy array(s) of predictions. Raises
    # value errors on input mismatch (via the loops).
    """Generates output predictions for the input samples.

    Computation is done in batches (see the `batch_size` arg.)

    Args:
        x: Input samples. It could be:
          - A Numpy array (or array-like), or a list of arrays
            (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
            (in case the model has multiple inputs).
          - A `tf.data` dataset.
          - A generator or `keras.utils.Sequence` instance.
        batch_size: Integer or `None`.
            Number of samples per batch of computation.
            If unspecified, `batch_size` will default to 32.
            Do not specify the `batch_size` if your data is in the
            form of symbolic tensors, dataset,
            generators, or `keras.utils.Sequence` instances (since they generate
            batches).
        verbose: Verbosity mode, 0 or 1.
        steps: Total number of steps (batches of samples)
            before declaring the prediction round finished.
            Ignored with the default value of `None`. If x is a `tf.data`
            dataset and `steps` is None, `predict` will
            run until the input dataset is exhausted.
        callbacks: List of `keras.callbacks.Callback` instances.
            List of callbacks to apply during prediction.
            See [callbacks](/api_docs/python/tf/keras/callbacks).
        max_queue_size: Integer. Used for generator or `keras.utils.Sequence`
            input only. Maximum size for the generator queue.
            If unspecified, `max_queue_size` will default to 10.
        workers: Integer. Used for generator or `keras.utils.Sequence` input
            only. Maximum number of processes to spin up when using
            process-based threading. If unspecified, `workers` will default
            to 1. If 0, will execute the generator on the main thread.
        use_multiprocessing: Boolean. Used for generator or
            `keras.utils.Sequence` input only. If `True`, use process-based
            threading. If unspecified, `use_multiprocessing` will default to
            `False`. Note that because this implementation relies on
            multiprocessing, you should not pass non-picklable arguments to
            the generator as they can't be passed easily to children processes.


    Returns:
        Numpy array(s) of predictions.

    Raises:
        ValueError: In case of mismatch between the provided
            input data and the model's expectations,
            or in case a stateful model receives a number of samples
            that is not a multiple of the batch size.
    """
    self._assert_built_as_v1()
    self._check_call_args('predict')

    func = self._select_training_loop(x)
    return func.predict(
        self,
        x=x,
        batch_size=batch_size,
        verbose=verbose,
        steps=steps,
        callbacks=callbacks,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing)

  def reset_metrics(self):
    # [FUNCTION DOC]
    # Resets the state of every training/evaluation metric of the model to its initial value and,
    # when a distribution strategy is in use, also resets the metrics of the distributed (cloned)
    # models. Takes no inputs and returns nothing; mutates metric state.
    """Resets the state of metrics."""
    metrics = self._get_training_eval_metrics()
    for m in metrics:
      m.reset_state()

    # Reset metrics on all the distributed (cloned) models.
    if self._distribution_strategy:
      distributed_training_utils_v1._reset_metrics(self)  # pylint: disable=protected-access

  def train_on_batch(self,
                     x,
                     y=None,
                     sample_weight=None,
                     class_weight=None,
                     reset_metrics=True):
    # [FUNCTION DOC]
    # KNOWN ISSUE: unlike fit it does not verify the model was built as v1.
    # Runs one gradient update on one batch. Inputs: x (data), y (targets, optional when x is a
    # dataset), sample_weight, class_weight, reset_metrics (default true: returned metrics refer to
    # this batch only). Requires compilation; checks forward-call arguments. Raises a
    # not-implemented error when called in the cross-replica context of a distribution strategy.
    # Standardizes the data, then either runs the eager training step (when running eagerly or in
    # a replica of a strategy) or feeds the graph train function (building it and the sample
    # weight modes on demand, adding the learning-phase value when it is symbolic). Resets metrics
    # afterwards if requested. Returns the single value if only one output value exists, otherwise
    # a list of values (loss followed by metrics). Side effect: updates weights.
    """Runs a single gradient update on a single batch of data.

    Args:
        x: Input data. It could be:
          - A Numpy array (or array-like), or a list of arrays
              (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
              (in case the model has multiple inputs).
          - A dict mapping input names to the corresponding array/tensors,
              if the model has named inputs.
          - A `tf.data` dataset.
        y: Target data. Like the input data `x`, it could be either Numpy
          array(s) or TensorFlow tensor(s). It should be consistent with `x`
          (you cannot have Numpy inputs and tensor targets, or inversely). If
          `x` is a dataset, `y` should not be specified
          (since targets will be obtained from the iterator).
        sample_weight: Optional array of the same length as x, containing
          weights to apply to the model's loss for each sample. In the case of
          temporal data, you can pass a 2D array with shape (samples,
          sequence_length), to apply a different weight to every timestep of
          every sample. In this case you should make sure to specify
          sample_weight_mode="temporal" in compile(). This argument is not
          supported when `x` is a dataset.
        class_weight: Optional dictionary mapping class indices (integers) to a
          weight (float) to apply to the model's loss for the samples from this
          class during training. This can be useful to tell the model to "pay
          more attention" to samples from an under-represented class.
        reset_metrics: If `True`, the metrics returned will be only for this
          batch. If `False`, the metrics will be statefully accumulated across
          batches.

    Returns:
        Scalar training loss
        (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.

    Raises:
      ValueError: In case of invalid user-provided arguments.
    """
    self._assert_compile_was_called()
    self._check_call_args('train_on_batch')

    # If at this point we are in the replica context, then it is okay to execute
    # the Eager code path.  The expected way to get here is to call `fit` that
    # calls `train_on_batch` on each replica.
    if (self._distribution_strategy and
        distribute_lib.in_cross_replica_context()):
      raise NotImplementedError('`train_on_batch` is not supported for models '
                                'distributed with tf.distribute.Strategy.')
    # Validate and standardize user data.
    x, y, sample_weights = self._standardize_user_data(
        x, y, sample_weight=sample_weight, class_weight=class_weight,
        extract_tensors_from_dataset=True)

    # If `self._distribution_strategy` is True, then we are in a replica context
    # at this point because of the check above.  `train_on_batch` is being run
    # for each replica by `self._distribution_strategy` and the same code path
    # as Eager is expected to be taken.
    if self.run_eagerly or self._distribution_strategy:
      output_dict = training_eager_v1.train_on_batch(
          self,
          x,
          y,
          sample_weights=sample_weights,
          output_loss_metrics=self._output_loss_metrics)
      outputs = (output_dict['total_loss'] + output_dict['output_losses']
                 + output_dict['metrics'])
      outputs = [_non_none_constant_value(v) for v in outputs]  # pylint: disable=protected-access
    else:
      x = training_utils_v1.ModelInputs(x).as_list()
      ins = x + list(y or []) + list(sample_weights or [])

      if not isinstance(backend.symbolic_learning_phase(), int):
        ins += [True]  # Add learning phase value.

      self._update_sample_weight_modes(sample_weights=sample_weights)
      self._make_train_function()
      outputs = self.train_function(ins)  # pylint: disable=not-callable

    if reset_metrics:
      self.reset_metrics()

    if len(outputs) == 1:
      return outputs[0]
    return outputs

  def test_on_batch(self, x, y=None, sample_weight=None, reset_metrics=True):
    # [FUNCTION DOC]
    # Evaluates the model on one batch without updating weights. Inputs: x (data), y (targets,
    # optional for datasets), sample_weight, reset_metrics (default true). Requires compilation;
    # checks forward-call arguments; raises a not-implemented error in the cross-replica context
    # of a distribution strategy. Standardizes data, then runs the eager test step (eager or
    # strategy replica) or the lazily built graph test function. Optionally resets metrics.
    # Returns a single value when only one output value exists, otherwise a list (loss followed by
    # metrics). Raises value errors for invalid arguments (via data standardization).
    """Test the model on a single batch of samples.

    Args:
        x: Input data. It could be:
          - A Numpy array (or array-like), or a list of arrays
            (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
            (in case the model has multiple inputs).
          - A dict mapping input names to the corresponding array/tensors,
            if the model has named inputs.
          - A `tf.data` dataset.
        y: Target data. Like the input data `x`,
          it could be either Numpy array(s) or TensorFlow tensor(s).
          It should be consistent with `x` (you cannot have Numpy inputs and
          tensor targets, or inversely). If `x` is a dataset `y` should
          not be specified (since targets will be obtained from the iterator).
        sample_weight: Optional array of the same length as x, containing
            weights to apply to the model's loss for each sample.
            In the case of temporal data, you can pass a 2D array
            with shape (samples, sequence_length),
            to apply a different weight to every timestep of every sample.
            In this case you should make sure to specify
            sample_weight_mode="temporal" in compile(). This argument is not
            supported when `x` is a dataset.
        reset_metrics: If `True`, the metrics returned will be only for this
          batch. If `False`, the metrics will be statefully accumulated across
          batches.

    Returns:
        Scalar test loss (if the model has a single output and no metrics)
        or list of scalars (if the model has multiple outputs
        and/or metrics). The attribute `model.metrics_names` will give you
        the display labels for the scalar outputs.

    Raises:
        ValueError: In case of invalid user-provided arguments.
    """
    self._assert_compile_was_called()
    self._check_call_args('test_on_batch')

    if (self._distribution_strategy and
        distribute_lib.in_cross_replica_context()):
      raise NotImplementedError('`test_on_batch` is not supported for models '
                                'distributed with tf.distribute.Strategy.')
    # Validate and standardize user data.
    x, y, sample_weights = self._standardize_user_data(
        x, y, sample_weight=sample_weight, extract_tensors_from_dataset=True)

    # If `self._distribution_strategy` is True, then we are in a replica context
    # at this point.
    if self.run_eagerly or self._distribution_strategy:
      output_dict = training_eager_v1.test_on_batch(
          self,
          x,
          y,
          sample_weights=sample_weights,
          output_loss_metrics=self._output_loss_metrics)
      outputs = (output_dict['total_loss'] + output_dict['output_losses']
                 + output_dict['metrics'])
      outputs = [_non_none_constant_value(v) for v in outputs]  # pylint: disable=protected-access
    else:
      x = training_utils_v1.ModelInputs(x).as_list()
      inputs = x + list(y or []) + list(sample_weights or [])

      self._update_sample_weight_modes(sample_weights=sample_weights)
      self._make_test_function()
      outputs = self.test_function(inputs)  # pylint: disable=not-callable

    if reset_metrics:
      self.reset_metrics()

    if len(outputs) == 1:
      return outputs[0]
    return outputs

  def predict_on_batch(self, x):
    # [FUNCTION DOC]
    # Returns predictions for a single batch. Input: x (arrays, tensors or dataset). Checks the
    # forward-call arguments, raises a not-implemented error in the cross-replica context of a
    # distribution strategy, and standardizes the data. When running eagerly or in a strategy
    # replica, casts floating inputs, unwraps a one-element sequence and calls the model directly;
    # otherwise builds the graph predict function lazily and runs it. Returns a single output
    # or a list of outputs. Does not require compilation.
    """Returns predictions for a single batch of samples.

    Args:
        x: Input data. It could be:
          - A Numpy array (or array-like), or a list of arrays
            (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
            (in case the model has multiple inputs).
          - A `tf.data` dataset.

    Returns:
        Numpy array(s) of predictions.

    Raises:
        ValueError: In case of mismatch between given number of inputs and
          expectations of the model.
    """
    self._check_call_args('predict_on_batch')

    if (self._distribution_strategy and
        distribute_lib.in_cross_replica_context()):
      raise NotImplementedError(
          '`predict_on_batch` is not supported for models distributed with'
          ' tf.distribute.Strategy.')
    # Validate and standardize user data.
    inputs, _, _ = self._standardize_user_data(
        x, extract_tensors_from_dataset=True)
    # If `self._distribution_strategy` is True, then we are in a replica context
    # at this point.
    if self.run_eagerly or self._distribution_strategy:
      inputs = training_utils_v1.cast_if_floating_dtype(inputs)
      if isinstance(inputs, collections.abc.Sequence):
        # Unwrap lists with only one input, as we do when training on batch
        if len(inputs) == 1:
          inputs = inputs[0]

      return self(inputs)  # pylint: disable=not-callable

    self._make_predict_function()
    outputs = self.predict_function(inputs)

    if len(outputs) == 1:
      return outputs[0]
    return outputs

  def fit_generator(self,
                    generator,
                    steps_per_epoch=None,
                    epochs=1,
                    verbose=1,
                    callbacks=None,
                    validation_data=None,
                    validation_steps=None,
                    validation_freq=1,
                    class_weight=None,
                    max_queue_size=10,
                    workers=1,
                    use_multiprocessing=False,
                    shuffle=True,
                    initial_epoch=0):
    # [FUNCTION DOC]
    # Deprecated wrapper kept for backward compatibility. Emits a warning, then forwards the
    # generator and all training options (steps per epoch, epochs, verbosity, callbacks,
    # validation data and steps and frequency, class weight, queue size, workers, multiprocessing,
    # shuffle, initial epoch) to fit. Returns the history object from fit.
    """Fits the model on data yielded batch-by-batch by a Python generator.

    DEPRECATED:
      `Model.fit` now supports generators, so there is no longer any need to use
      this endpoint.
    """
    warnings.warn('`model.fit_generator` is deprecated and '
                  'will be removed in a future version. '
                  'Please use `Model.fit`, which supports generators.')
    return self.fit(
        generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=verbose,
        callbacks=callbacks,
        validation_data=validation_data,
        validation_steps=validation_steps,
        validation_freq=validation_freq,
        class_weight=class_weight,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        shuffle=shuffle,
        initial_epoch=initial_epoch)

  def evaluate_generator(self,
                         generator,
                         steps=None,
                         callbacks=None,
                         max_queue_size=10,
                         workers=1,
                         use_multiprocessing=False,
                         verbose=0):
    # [FUNCTION DOC]
    # Deprecated wrapper. Emits a warning, checks the model's forward-call arguments (raising a
    # value error if call has extra positional arguments), then forwards the generator and the
    # options (steps, callbacks, queue size, workers, multiprocessing, verbosity) to evaluate.
    # Returns the evaluation result.
    """Evaluates the model on a data generator.

    DEPRECATED:
      `Model.evaluate` now supports generators, so there is no longer any need
      to use this endpoint.
    """
    warnings.warn('`Model.evaluate_generator` is deprecated and '
                  'will be removed in a future version. '
                  'Please use `Model.evaluate`, which supports generators.')
    self._check_call_args('evaluate_generator')

    return self.evaluate(
        generator,
        steps=steps,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        verbose=verbose,
        callbacks=callbacks)

  def predict_generator(self,
                        generator,
                        steps=None,
                        callbacks=None,
                        max_queue_size=10,
                        workers=1,
                        use_multiprocessing=False,
                        verbose=0):
    # [FUNCTION DOC]
    # Deprecated wrapper. Emits a warning, then forwards the generator and the options (steps,
    # callbacks, queue size, workers, multiprocessing, verbosity) to predict. Returns the
    # predictions.
    """Generates predictions for the input samples from a data generator.

    DEPRECATED:
      `Model.predict` now supports generators, so there is no longer any need
      to use this endpoint.
    """
    warnings.warn('`Model.predict_generator` is deprecated and '
                  'will be removed in a future version. '
                  'Please use `Model.predict`, which supports generators.')
    return self.predict(
        generator,
        steps=steps,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
        verbose=verbose,
        callbacks=callbacks)

  def _check_call_args(self, method_name):
    # [FUNCTION DOC]
    # HELPER: Validation used by fit, evaluate, predict and the on-batch methods.
    # Input: method_name, the public method being called (used in the error message). Inspects the
    # argument specification of the model's forward method, drops default-valued arguments and the
    # training argument, and raises a value error if more than the receiver and one input
    # remain as positional arguments. Returns nothing.
    """Check that `call` has only one positional arg."""
    # Always allow first arg, regardless of arg name.
    fullargspec = self._call_full_argspec
    if fullargspec.defaults:
      positional_args = fullargspec.args[:-len(fullargspec.defaults)]
    else:
      positional_args = fullargspec.args
    if 'training' in positional_args:
      positional_args.remove('training')

    # self and first arg can be positional.
    if len(positional_args) > 2:
      extra_args = positional_args[2:]
      raise ValueError(
          'Models passed to `' + method_name + '` can only have `training` '
          'and the first argument in `call` as positional arguments, '
          'found: ' + str(extra_args) + '.')

  def _set_optimizer(self, optimizer):
    # [FUNCTION DOC]
    # HELPER: Used by compile to resolve and wrap the optimizer.
    # KNOWN ISSUE: the two value-error messages use format placeholders but pass the values as
    # separate exception arguments (the first one is not even a tuple), so the placeholders are
    # never filled; building the first message would actually fail with a formatting error
    # because two placeholders receive a single value.
    # Input: optimizer, a name, an optimizer instance, or a list/tuple of them. Resolves each into
    # an optimizer object and stores it as the model optimizer. Based on the dtype policy (a v1
    # policy with a loss scale, or the mixed float16 policy using dynamic scaling) wraps it in a
    # loss-scale optimizer when not already one. Raises value errors if a loss scale is needed
    # but a list of optimizers was given, or the optimizer is not a v2 optimizer instance.
    # Returns nothing; sets the optimizer attribute.
    """Sets self.optimizer.

    Sets self.optimizer to `optimizer`, potentially wrapping it with a
    LossScaleOptimizer.

    Args:
      optimizer: The optimizer(s) to assign to self.optimizer.
    """
    if isinstance(optimizer, (list, tuple)):
      self.optimizer = [optimizers.get(opt) for opt in optimizer]
    else:
      self.optimizer = optimizers.get(optimizer)

    if isinstance(self._dtype_policy, policy.PolicyV1):
      loss_scale = self._dtype_policy.loss_scale
    elif self._dtype_policy.name == 'mixed_float16':
      loss_scale = 'dynamic'
    else:
      loss_scale = None

    if (loss_scale is not None and
        not isinstance(self.optimizer,
                       loss_scale_optimizer.LossScaleOptimizer)):
      if isinstance(self.optimizer, list):
        raise ValueError('When a dtype policy with a loss scale is used, you '
                         'can only pass a single optimizer. Using policy %s '
                         'and got optimizers: %s' %
                         self._dtype_policy, self.optimizer)
      if not isinstance(self.optimizer, optimizer_v2.OptimizerV2):
        raise ValueError('"optimizer" must be an instance of '
                         'tf.keras.optimizers.Optimizer when a dype policy '
                         'with a loss scale  used, but got: %s. Using policy: '
                         '%s' %
                         (self.optimizer, self._dtype_policy))
      if loss_scale == 'dynamic':
        self.optimizer = loss_scale_optimizer.LossScaleOptimizer(self.optimizer)
      else:
        self.optimizer = loss_scale_optimizer.LossScaleOptimizerV1(
            self.optimizer, loss_scale)

  def _prepare_validation_data(self, validation_data, batch_size,
                               validation_steps):
    # [FUNCTION DOC]
    # HELPER: Used by training loops to prepare validation data.
    # Inputs: validation_data (tuple of inputs, targets and optional sample weights, or a
    # dataset), batch_size, validation_steps. Unpacks the validation data and returns the result of
    # standardizing it (inputs, targets, sample weights) using the validation steps name for
    # error messages.
    """Unpack and check the validation data."""
    val_x, val_y, val_sample_weights = training_utils_v1.unpack_validation_data(
        validation_data)
    return self._standardize_user_data(
        val_x,
        val_y,
        sample_weight=val_sample_weights,
        batch_size=batch_size,
        steps=validation_steps,
        steps_name='validation_steps')

  def _validate_compile_param_for_distribution_strategy(
      self, run_eagerly, sample_weight_mode, target_tensors, weighted_metrics):
    # [FUNCTION DOC]
    # HELPER: Argument validation used by compile when the tf.function loop is not used.
    # If a distribution strategy is set, rejects unsupported compile arguments: raises a
    # not-implemented error for sample weight mode or weighted metrics, and value errors for
    # target tensors, eager running, or cloning-based distribution with a model whose inputs and
    # outputs are not yet known. Does nothing without a strategy. Returns nothing.
    # Validate that arguments passed by the user to `compile` are supported by
    # tf.distribute.Strategy.
    if self._distribution_strategy:
      if sample_weight_mode:
        raise NotImplementedError('sample_weight_mode is not supported with '
                                  'tf.distribute.Strategy.')
      if weighted_metrics:
        raise NotImplementedError('weighted_metrics is not supported with '
                                  'tf.distribute.Strategy.')
      if target_tensors:
        raise ValueError('target_tensors is not supported with '
                         'tf.distribute.Strategy.')

      if run_eagerly:
        raise ValueError(
            'We currently do not support enabling `run_eagerly` with '
            'distribution strategy.')

      if (distributed_training_utils_v1.is_distributing_by_cloning(self) and
          (not self.built or not self.inputs or not self.outputs)):
        raise ValueError(
            'We currently do not support distribution strategy with a '
            '`Sequential` model that is created without `input_shape`/'
            '`input_dim` set in its first layer or a subclassed model.')

  def _process_target_tensor_for_compile(self, target_tensors):
    # [FUNCTION DOC]
    # HELPER: Normalizes the target_tensors argument of compile.
    # Input: target_tensors, nothing, a single tensor, a list, or a mapping of output names to
    # tensors. Returns a list with one entry per model output (entries may be nothing). When
    # running eagerly, or when the input is nothing or an empty list, returns all-nothing entries.
    # A list must match the number of outputs (value error otherwise); a mapping must only use
    # known output names (value error otherwise) and is ordered by output name with missing
    # names as nothing; a single tensor becomes a one-element list; anything else raises a type
    # error.
    if self.run_eagerly:
      # target tensor is not supported with run_eagerly. Create a list with None
      # as placeholder for each output.
      return [None for _ in self.output_names]

    if target_tensors is not None and not (isinstance(target_tensors, list) and
                                           target_tensors == []):  # pylint: disable=g-explicit-bool-comparison
      if isinstance(target_tensors, list):
        if len(target_tensors) != len(self.outputs):
          raise ValueError(
              'When passing a list as `target_tensors`, '
              'it should have one entry per model output. '
              'The model has %s outputs, but you passed target_tensors=%s' %
              (len(self.outputs), target_tensors))
      elif isinstance(target_tensors, dict):
        unexpected_target_tensor_names = set(target_tensors.keys()).difference(
            self.output_names)
        if unexpected_target_tensor_names:
          raise ValueError(
              'Unknown entry in `target_tensors` dictionary: "{name}". '
              'Only expected the following keys: {keys}'.format(
                  name=unexpected_target_tensor_names,
                  keys=str(self.output_names)))
        tmp_target_tensors = []
        for name in self.output_names:
          tmp_target_tensors.append(target_tensors.get(name, None))
        target_tensors = tmp_target_tensors
      elif tensor_util.is_tf_type(target_tensors):
        target_tensors = [target_tensors]
      else:
        raise TypeError('Expected `target_tensors` to be a list or tuple or '
                        'dict or a single tensor, but got:', target_tensors)
    else:
      # In case target tensor is empty or None, create a list with Nones
      # that has same length as self.output_names. With that, the None check of
      # target tensor can be skipped downstream.
      target_tensors = [None for _ in self.output_names]
    return target_tensors

  def _compile_eagerly(self, metrics, weighted_metrics, sample_weight_mode):
    # [FUNCTION DOC]
    # HELPER: Eager-mode tail of compile.
    # Inputs: metrics, weighted_metrics and sample_weight_mode as given to compile. Prepares the
    # sample weight modes and sample weights of the training endpoints, caches per-output metric
    # names and functions, clears the total loss (not graph-built when eager), sets the metric
    # attributes on the model and records the trainable weights. Returns nothing.
    # Prepare sample weight modes. List with the same length as model outputs.
    training_utils_v1.prepare_sample_weight_modes(
        self._training_endpoints, sample_weight_mode)
    # Prepare sample weights.
    self._prepare_sample_weights()
    # Save all metric attributes per output of the model.
    self._cache_output_metric_attributes(metrics, weighted_metrics)
    self.total_loss = None
    # Set metric attributes on model.
    self._set_metric_attributes()

    self._collected_trainable_weights = self.trainable_weights

  def _update_sample_weight_modes(self, sample_weights=None):
    # [FUNCTION DOC]
    # HELPER: Keeps endpoint sample-weight modes in step with the data being fed.
    # Input: sample_weights, a list aligned with the outputs or nothing. Does nothing if the model
    # is not compiled. If any sample weight is given, every endpoint without a mode gets the
    # sample-wise mode (existing modes such as temporal are kept); if none is given, every
    # endpoint mode is reset to nothing. Mutates the endpoints; returns nothing.
    """Updates sample weight modes based on training/eval inputs.

    Sample weight placeholders will be created for all or no outputs
    based on whether sample_weight is provided for any output.

    If model contains `_sample_weight_modes` we check if the input
    `sample_weights` corresponds to the sample weight modes.
      1. Set sample weight mode to be 'temporal' for output i, if `compile`
        sample_weight_mode was set to `temporal` and sample weight inputs
        are given for one or more outputs.
      2. Set sample weight mode to be 'samplewise' for output i, if `compile`
        sample_weight_mode was not set and sample weight inputs are given for
        one or more outputs.
      3. Reset sample weight mode to None for output i if sample weight mode
        was set but there is no sample weight input.

    Args:
      sample_weights: List of sample weights of the same length as model outputs
        or None.
    """
    if not self._is_compiled:
      return
    if sample_weights and any(s is not None for s in sample_weights):
      for endpoint in self._training_endpoints:
        endpoint.sample_weight_mode = (
            endpoint.sample_weight_mode or 'samplewise')
    else:
      for endpoint in self._training_endpoints:
        endpoint.sample_weight_mode = None

  def _recompile_weights_loss_and_weighted_metrics(self):
    # [FUNCTION DOC]
    # HELPER: Rebuilds loss and weighted metrics when sample-weight placeholders changed.
    # Returns false immediately if the model is not compiled. Otherwise, if any training endpoint
    # reports that its sample weights no longer match its mode, recompiles the loss and weighted
    # metric graphs. Returns a boolean telling whether recompilation happened.
    if not self._is_compiled:
      return False
    recompile = any(
        e.sample_weights_mismatch() for e in self._training_endpoints)

    if recompile:
      self._compile_weights_loss_and_weighted_metrics()
    return recompile

  @trackable.no_automatic_dependency_tracking
  def _compile_weights_loss_and_weighted_metrics(self, sample_weights=None):
    # [FUNCTION DOC]
    # HELPER: Graph-building step of compile (and of recompilation).
    # Input: sample_weights, optional list of tensors (one per output) to use instead of
    # placeholders. Inside the backend graph: updates the sample weight modes when tensors are
    # given, prepares sample weights, computes the output masks, builds the weighted metrics and
    # stores the total loss tensor on the model. Returns nothing; sets sample weights and total
    # loss. Dependency tracking is disabled for this method.
    """Compiles the model loss and weighted metric sub-graphs.

    This may be used to set graph tensors as sample weights (instead of creating
    placeholders).

    Args:
      sample_weights: List of tensors to use as the sample weights. Must be the
        same length as the number of outputs. If left as `None`, placeholders
        are used instead.
    """
    with backend.get_graph().as_default():
      if sample_weights is not None:
        self._update_sample_weight_modes(sample_weights)
      self._prepare_sample_weights(sample_weights)

      masks = self._prepare_output_masks()

      # Compute weighted metrics.
      self._handle_metrics(
          self.outputs,
          targets=self._targets,
          skip_target_masks=self._prepare_skip_target_masks(),
          sample_weights=self.sample_weights,
          masks=masks,
          return_weighted_metrics=True)

      # Compute total loss.
      # Used to keep track of the total loss value (stateless).
      # eg., total_loss = loss_weight_1 * output_1_loss_fn(...) +
      #                   loss_weight_2 * output_2_loss_fn(...) +
      #                   layer losses.
      self.total_loss = self._prepare_total_loss(masks)

  def _prepare_skip_target_masks(self):
    # [FUNCTION DOC]
    # HELPER: Computes which outputs have no loss function.
    # Returns a list of booleans, one per loss function, true where the loss function is nothing
    # (such outputs are skipped in loss computation and target feeding). No side effects.
    """Boolean mask for whether the target in the output list should be skipped.

    If the loss function corresponding to a model output is None, then this
    output will be skipped during total loss calculation and feed targets
    preparation.

    Returns:
      A boolean list for whether the corresponding target in the output list
      should be skipped during loss calculation.
    """
    return [l is None for l in self.loss_functions]

  def _prepare_output_masks(self):
    # [FUNCTION DOC]
    # HELPER: Collects the Keras mask attached to each model output.
    # Returns a list with one entry per output: its mask tensor or nothing if it has none.
    """Returns masks corresponding to model outputs."""
    return [getattr(x, '_keras_mask', None) for x in self.outputs]

  def _prepare_total_loss(self, masks):
    # [FUNCTION DOC]
    # HELPER: Builds the total loss graph tensor used by compile.
    # KNOWN ISSUE: the docstring says it returns a list of loss weights of python floats, but it
    # actually returns the total loss tensor (sum of weighted per-output losses and layer losses,
    # or the number zero when there are no losses).
    # Input: masks, list of output masks (entries may be nothing). Raises a type error when running
    # eagerly. For every non-skipped endpoint: combines the mask into the sample weight, computes
    # the loss with the loss object's reduction (or, for custom callables, assuming mean
    # reduction), records a stateful per-output loss metric when there are several outputs, scales
    # for distribution when the reduction is sum-over-batch-size, and multiplies by the loss weight.
    # Raises a value error if there is no loss to optimize at all. Then adds regularization and
    # other layer losses (scaled for distribution), casts to a common dtype and sums.
    # Returns the total loss.
    """Computes total loss from loss functions.

    Args:
        masks: List of mask values corresponding to each model output.

    Returns:
        A list of loss weights of python floats.

    Raises:
        TypeError: If model run_eagerly is True.
    """
    if self.run_eagerly:
      raise TypeError('total loss can not be computed when compiled with '
                      'run_eagerly = True.')
    loss_list = []
    with backend.name_scope('loss'):
      for endpoint, mask in zip(self._training_endpoints, masks):
        if endpoint.should_skip_target():
          continue
        y_true = endpoint.training_target.target
        y_pred = endpoint.output
        loss_fn = endpoint.loss_fn
        loss_weight = endpoint.loss_weight
        loss_name = endpoint.loss_name()
        sample_weight = endpoint.sample_weight

        with backend.name_scope(loss_name):
          if mask is not None:
            mask = math_ops.cast(mask, y_pred.dtype)
            # Update weights with mask.
            if sample_weight is None:
              sample_weight = mask
            else:
              # Update dimensions of weights to match with mask if possible.
              mask, _, sample_weight = (
                  losses_utils.squeeze_or_expand_dimensions(
                      mask, sample_weight=sample_weight))
              sample_weight *= mask

          if hasattr(loss_fn, 'reduction'):
            per_sample_losses = loss_fn.call(y_true, y_pred)
            weighted_losses = losses_utils.compute_weighted_loss(
                per_sample_losses,
                sample_weight=sample_weight,
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
            output_loss = loss_fn(y_true, y_pred, sample_weight=sample_weight)
            loss_reduction = losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE

        if len(self.outputs) > 1:
          # Keep track of stateful result tensor for the loss.
          endpoint.output_loss_metric(output_loss)

        # Scale output loss for distribution. For custom losses we assume
        # reduction was mean.
        if loss_reduction == losses_utils.ReductionV2.SUM_OVER_BATCH_SIZE:
          output_loss = losses_utils.scale_loss_for_distribution(output_loss)

        loss_list.append(loss_weight * output_loss)
      if not loss_list and not self.losses:
        raise ValueError('The model cannot be compiled '
                         'because it has no loss to optimize.')

      # Add regularization penalties and other layer-specific losses.
      custom_losses = self.get_losses_for(None) + self.get_losses_for(
          self.inputs)
      if custom_losses:
        total_custom_loss = math_ops.add_n(
            losses_utils.cast_losses_to_common_dtype(custom_losses))
        loss_list.append(
            losses_utils.scale_loss_for_distribution(total_custom_loss))

      loss_list = losses_utils.cast_losses_to_common_dtype(loss_list)
      if loss_list:
        total_loss = math_ops.add_n(loss_list)
      else:
        total_loss = 0.
    return total_loss

  def _get_callback_model(self):
    # [FUNCTION DOC]
    # HELPER: Used by callbacks to find the model they should save/load.
    # Returns the replicated callback model if one was made for distributed training, else an
    # existing callback model attribute if set, else the model itself.
    """Returns the Callback Model for this Model."""

    if hasattr(self, '_replicated_model') and self._replicated_model:
      # When using training_distributed, we set the callback model
      # to an instance of the `DistributedModel` that we create in
      # the `compile` call. The `DistributedModel` is initialized
      # with the first replicated model. We need to set the callback
      # model to a DistributedModel to allow us to override saving
      # and loading weights when we checkpoint the model during training.
      return self._replicated_model
    if hasattr(self, 'callback_model') and self.callback_model:
      return self.callback_model
    return self

  @trackable.no_automatic_dependency_tracking
  def _make_callback_model(self, grouped_model):
    # [FUNCTION DOC]
    # HELPER: Creates the callback model used with distribution strategies.
    # Input: grouped_model, the per-replica grouped model. Unwraps it to its first replica, wraps
    # that in a distributed callback model and links it back to this model as the original.
    # Side effect: sets the replicated-model attribute (dependency tracking disabled). Returns
    # nothing.
    first_replicated_model = self._distribution_strategy.unwrap(
        grouped_model)[0]
    # We initialize the callback model with the first replicated model.
    self._replicated_model = DistributedCallbackModel(first_replicated_model)
    self._replicated_model.set_original_model(self)

  def _validate_or_infer_batch_size(self, batch_size, steps, x):
    # [FUNCTION DOC]
    # HELPER: Used by data standardization to validate or infer the batch size.
    # KNOWN ISSUE: the docstring promises the validated batch size is returned, but for dataset,
    # sequence and generator inputs it returns nothing (after raising a value error if a batch
    # size was specified for such inputs).
    # Inputs: batch_size (integer or nothing), steps (integer or nothing), x (the data). If the
    # first layer has a static batch size: checks batch_size is divisible by the number of replicas
    # (when the strategy splits global batches) and equals the static size per replica, checks
    # the dataset or iterator batch size likewise, and, when steps is nothing, infers the batch
    # size from the static one times the replica count. All mismatches raise value errors.
    # Without batch size or steps it falls back to the legacy default of thirty-two.
    # Returns the resulting batch size.
    """Validates that the `batch_size` provided is consistent with InputLayer.

    It's possible that the user specified a static batch size in their
    InputLayer. If so, this method checks the provided `batch_size` and `x`
    arguments are consistent with this static batch size. Also, if
    `batch_size` is `None`, this method will attempt to infer the batch size
    from the static batch size of the InputLayer. Lastly, ValueError will be
    raised if `x` is a tf.data.Dataset and `batch_size` is specified as we
    expect users to provide batched datasets.

    Args:
      batch_size: The batch_size provided as an argument to
        fit/evaluate/predict.
      steps: The steps provided as an argument to fit/evaluate/predict.
      x: The data passed as `x` to fit/evaluate/predict.

    Returns:
      The validated batch_size, auto-inferred from the first layer if not
      provided.
    """
    if (isinstance(x, (data_types.DatasetV1,
                       data_types.DatasetV2,
                       data_utils.Sequence)) or
        tf_inspect.isgenerator(x)):
      if batch_size is not None:
        raise ValueError(
            'The `batch_size` argument must not be specified for the given '
            'input type. Received input: {}, batch_size: {}'.format(
                x, batch_size))
      return

    # Avoids the override in Sequential.layers which filters Input layers.
    # (Which are often the very layers that we're after.)
    layers = self._flatten_layers(include_self=False, recursive=False)
    first_layer = next(layers, None)
    if first_layer:
      # The per-replica static batch size.
      static_batch_size = training_utils.get_static_batch_size(first_layer)
      if static_batch_size is not None:

        # Determine number of times the user-supplied batch size will be split.
        if (self._distribution_strategy and
            distributed_training_utils.global_batch_size_supported(
                self._distribution_strategy)):
          num_splits_for_ds = self._distribution_strategy.num_replicas_in_sync
        else:
          num_splits_for_ds = 1

        # Check `batch_size` argument is consistent with InputLayer.
        if batch_size is not None:
          if batch_size % num_splits_for_ds != 0:
            raise ValueError('The `batch_size` argument ({}) must be divisible '
                             'the by number of replicas ({})'.format(
                                 batch_size, num_splits_for_ds))
          per_replica_batch_size = batch_size // num_splits_for_ds

          if per_replica_batch_size != static_batch_size:
            raise ValueError('The `batch_size` argument value {} is '
                             'incompatible with the specified batch size of '
                             'your Input Layer: {}'.format(
                                 per_replica_batch_size, static_batch_size))

        # Check Dataset/Iterator batch size is consistent with InputLayer.
        if isinstance(x, (data_types.DatasetV2, iterator_ops.Iterator,
                          iterator_ops.IteratorBase)):
          ds_batch_size = tensor_shape.Dimension(
              nest.flatten(dataset_ops.get_legacy_output_shapes(x))[0][0]).value
          if ds_batch_size is not None:
            if ds_batch_size % num_splits_for_ds != 0:
              raise ValueError(
                  'The batch output shape of your `Dataset` {} '
                  'cannot be divisible by number of replicas {}'.format(
                      ds_batch_size, num_splits_for_ds))

            ds_per_replica_batch_size = ds_batch_size // num_splits_for_ds
            if ds_per_replica_batch_size != static_batch_size:
              raise ValueError('The batch output shape of your `Dataset` is '
                               '{}, which is incompatible with the specified '
                               'batch size of your Input Layer: {}'.format(
                                   ds_per_replica_batch_size,
                                   static_batch_size))

        # Set inferred batch size from the InputLayer.
        if steps is None:
          batch_size = static_batch_size * num_splits_for_ds

    if batch_size is None and steps is None:
      # Backwards compatibility
      batch_size = 32
    return batch_size

  def _prepare_sample_weights(self, sample_weights=None):
    # [FUNCTION DOC]
    # HELPER: Creates or assigns sample weight tensors for each training endpoint.
    # Input: sample_weights, optional list with one entry per output (tensors or nothing); when
    # omitted every entry is nothing, so placeholders are created as the endpoint mode requires.
    # Raises a value error if the list length differs from the number of endpoints. Calls each
    # endpoint to populate its sample weight. Returns nothing.
    """Sets sample weight attribute on the model."""
    # List with the same length as model outputs.
    if sample_weights is not None:
      if len(sample_weights) != len(self._training_endpoints):
        raise ValueError('Provided sample weights must have same length as the '
                         'number of outputs. Expected: {}, got: {}.'.format(
                             len(self._training_endpoints),
                             len(sample_weights)))
    else:
      sample_weights = [None] * len(self._training_endpoints)
    for endpoint, weight in zip(self._training_endpoints, sample_weights):
      endpoint.populate_sample_weight(weight, endpoint.sample_weight_mode)

  def _cache_output_metric_attributes(self, metrics, weighted_metrics):
    # [FUNCTION DOC]
    # HELPER: Prepares per-output metric information during compile.
    # Inputs: metrics and weighted_metrics as given to compile (list or mapping per output).
    # Gathers each output's static shape (nothing when unknown), then stores on the model the
    # per-output metric info and the per-output weighted metric info computed by the metric
    # collection helper, using output names, loss functions and the serialized-model flag.
    # Returns nothing.
    """Caches metric name and function attributes for every model output."""
    output_shapes = []
    for output in self.outputs:
      if output is None or output.shape.rank is None:
        output_shapes.append(None)
      else:
        output_shapes.append(output.shape.as_list())
    self._per_output_metrics = training_utils_v1.collect_per_output_metric_info(
        metrics, self.output_names, output_shapes, self.loss_functions,
        from_serialized=self._from_serialized)
    self._per_output_weighted_metrics = (
        training_utils_v1.collect_per_output_metric_info(
            weighted_metrics,
            self.output_names,
            output_shapes,
            self.loss_functions,
            from_serialized=self._from_serialized,
            is_weighted=True))

  def _add_unique_metric_name(self, metric_name, metric_fn, output_index):
    # [FUNCTION DOC]
    # HELPER: Naming utility used when registering compile metrics.
    # Inputs: metric_name (the user-facing name such as accuracy), metric_fn (the metric object or
    # stateless metric function), output_index (index of the output it applies to). For
    # multi-output models prefixes the output name, unless the metric object was loaded from an
    # already serialized model (where the prefix is already present). Then, while the name already
    # appears in the model's metric names, appends a counter suffix starting at one. Returns the
    # unique name string. No mutation.
    """Makes the metric name unique.

      If there are multiple outputs for which the metrics are calculated, the
      metric names have to be made unique by appending an integer.

    Args:
      metric_name: Metric name that corresponds to the metric specified by the
          user. For example: 'acc'.
      metric_fn: The Metric object.
      output_index: The index of the model output for which the metric name is
        being added.

    Returns:
      string, name of the model's unique metric name
    """
    # For multi-output models, prepend the output names to the metric name.
    if len(self.output_names) > 1:
      # If we're loading from an already-serialized model, we've already
      # prepended the output name, and we don't want to do it again.
      #
      # Alternatively, we may be receiving a stateless metric (e.g. the string
      # "accuracy") rather than a `Metric` object, in which case we want to
      # prepend the output name even if we are loading a serialized model.
      if not getattr(metric_fn, '_from_serialized', False):
        metric_name = '%s_%s' % (self.output_names[output_index], metric_name)

    j = 1
    base_metric_name = metric_name
    while metric_name in self.metrics_names:
      metric_name = '%s_%d' % (base_metric_name, j)
      j += 1

    return metric_name

  def _init_metric_attributes(self):
    # [FUNCTION DOC]
    # HELPER: Called by compile to reset metric bookkeeping.
    # Sets the list of stateful compile metric functions (used for resetting metric state during
    # training and evaluation) to an empty list. Returns nothing.
    """Initialized model metric attributes."""
    # List of stateful metric functions. Used for resetting metric state during
    # training/eval.
    self._compile_metric_functions = []

  def _set_per_output_metric_attributes(self, metrics_dict, output_index):
    # [FUNCTION DOC]
    # HELPER: Registers the metrics of one output on the model.
    # Inputs: metrics_dict (ordered mapping of metric names to metric objects), output_index.
    # For each metric computes a unique name, renames the metric object to it, and appends the
    # object to the model's list of compile metric functions. Returns a new ordered mapping keyed
    # by the unique names. Side effects: mutates the metric objects' names and the model list.
    """Sets the metric attributes on the model for the given output.

    Args:
      metrics_dict: A dict with metric names as keys and metric fns as values.
      output_index: The index of the model output for which the metric
        attributes are added.

    Returns:
      Metrics dict updated with unique metric names as keys.
    """
    updated_metrics_dict = collections.OrderedDict()
    for metric_name, metric_fn in metrics_dict.items():
      metric_name = self._add_unique_metric_name(
          metric_name, metric_fn, output_index)

      # Update the name on the metric class to be the unique generated name.
      metric_fn._name = metric_name  # pylint: disable=protected-access
      updated_metrics_dict[metric_name] = metric_fn
      # Keep track of metric name and function.
      self._compile_metric_functions.append(metric_fn)
    return updated_metrics_dict

  def _set_metric_attributes(self):
    # [FUNCTION DOC]
    # HELPER: Called by compile (graph and eager paths) after metric info was collected.
    # For every training endpoint not skipped, registers its unweighted and weighted metrics with
    # unique names; skipped endpoints keep their entries unchanged. When there are several
    # endpoints, also creates a running-mean metric for each non-skipped output loss. Finally
    # replaces the per-output metric and weighted-metric lists on the model with the updated
    # ones. Returns nothing.
    """Sets the metric attributes on the model for all the model outputs."""
    updated_per_output_metrics = []
    updated_per_output_weighted_metrics = []
    for i, endpoint in enumerate(self._training_endpoints):
      if endpoint.should_skip_target():
        updated_per_output_metrics.append(self._per_output_metrics[i])
        updated_per_output_weighted_metrics.append(
            self._per_output_weighted_metrics[i])
        continue
      updated_per_output_metrics.append(
          self._set_per_output_metric_attributes(self._per_output_metrics[i],
                                                 i))
      updated_per_output_weighted_metrics.append(
          self._set_per_output_metric_attributes(
              self._per_output_weighted_metrics[i], i))

    # Create a metric wrapper for each output loss. This computes mean of an
    # output loss across mini-batches (irrespective of how we reduce within a
    # batch).
    if len(self._training_endpoints) > 1:
      for endpoint in self._training_endpoints:
        if not endpoint.should_skip_target():
          endpoint.output_loss_metric = metrics_module.Mean(
              name=endpoint.loss_name())

    self._per_output_metrics = updated_per_output_metrics
    self._per_output_weighted_metrics = updated_per_output_weighted_metrics

  def _handle_per_output_metrics(self,
                                 metrics_dict,
                                 y_true,
                                 y_pred,
                                 mask,
                                 weights=None):
    # [FUNCTION DOC]
    # HELPER: Per-output worker of the metric handling step.
    # Inputs: metrics_dict (names to metric objects), y_true (targets), y_pred (predictions), mask
    # (output mask or nothing), weights (optional sample weights). Calls each metric inside a name
    # scope of its own name via the metric-calling helper. Returns a list of metric result
    # tensors in the mapping order.
    """Calls metric functions for a single output.

    Args:
      metrics_dict: A dict with metric names as keys and metric fns as values.
      y_true: Target output.
      y_pred: Predicted output.
      mask: Computed mask value for the current output.
      weights: Weights to be applied on the current output.

    Returns:
      A list of metric result tensors.
    """
    metric_results = []
    for metric_name, metric_fn in metrics_dict.items():
      with backend.name_scope(metric_name):
        metric_result = training_utils_v1.call_metric_function(
            metric_fn, y_true, y_pred, weights=weights, mask=mask)
        metric_results.append(metric_result)
    return metric_results

  def _handle_metrics(self,
                      outputs,
                      targets=None,
                      skip_target_masks=None,
                      sample_weights=None,
                      masks=None,
                      return_weighted_metrics=False,
                      return_weighted_and_unweighted_metrics=False):
    # [FUNCTION DOC]
    # HELPER: Builds the metric computation tensors for all outputs during compile or recompile.
    # Inputs: outputs (predictions), targets, skip_target_masks (booleans for outputs to ignore;
    # defaults to none skipped), sample_weights, masks, and two flags choosing whether weighted
    # metrics only, or both weighted and unweighted metrics, are computed (default: unweighted
    # only). For each non-skipped output, calls its unweighted and/or weighted metrics. Returns a
    # flat list of metric result tensors. Runs inside a metrics name scope.
    """Handles calling metric functions.

    Args:
      outputs: List of outputs (predictions).
      targets: List of targets.
      skip_target_masks: Optional. List of boolean for whether the corresponding
        target should be ignored or not.
      sample_weights: Optional list of sample weight arrays.
      masks: List of computed output mask values.
      return_weighted_metrics: Flag that indicates whether weighted metrics
        should be computed instead of unweighted metrics. This flag is ignored
        when `return_weighted_and_unweighted_metrics` is enabled.
      return_weighted_and_unweighted_metrics: Flag that is used to indicate
        whether both weighted and unweighted metrics should be computed. When
        this is not enabled, we use `return_weighted_metrics` param to indicate
        whether weighted or unweighted metrics should be returned.

    Returns:
      A list of metric result tensors.
    """
    # TODO(scottzhu): Update this to use the new training_endpoints. Currently
    # the eager and graph logic is bit different.
    skip_target_masks = skip_target_masks or [False] * len(outputs)
    metric_results = []
    with backend.name_scope('metrics'):
      # Invoke all metrics added using `compile`.
      for i in range(len(outputs)):
        if skip_target_masks[i]:
          continue
        output = outputs[i] if outputs else None
        target = targets[i] if targets else None
        output_mask = masks[i] if masks else None

        if (return_weighted_and_unweighted_metrics or
            not return_weighted_metrics):
          metric_results.extend(
              self._handle_per_output_metrics(self._per_output_metrics[i],
                                              target, output, output_mask))
        if return_weighted_and_unweighted_metrics or return_weighted_metrics:
          metric_results.extend(
              self._handle_per_output_metrics(
                  self._per_output_weighted_metrics[i],
                  target,
                  output,
                  output_mask,
                  weights=sample_weights[i] if sample_weights else None))
    return metric_results

  def _check_trainable_weights_consistency(self):
    # [FUNCTION DOC]
    # HELPER: Sanity check run before building the train function.
    # Does nothing if the collected trainable weights were never recorded. Otherwise, if the
    # current trainable weight count differs from the collected count (the trainable flag changed
    # after compile), logs a warning once. Returns nothing.
    """Check trainable weights count consistency.

    This will raise a warning if `trainable_weights` and
    `_collected_trainable_weights` are inconsistent (i.e. have different
    number of parameters).
    Inconsistency will typically arise when one modifies `model.trainable`
    without calling `model.compile` again.
    """
    if not hasattr(self, '_collected_trainable_weights'):
      return

    if len(self.trainable_weights) != len(self._collected_trainable_weights):
      logging.log_first_n(
          logging.WARN, 'Discrepancy between trainable weights and collected'
          ' trainable weights, did you set `model.trainable`'
          ' without calling `model.compile` after ?', 1)

  def _make_train_function(self):
    # [FUNCTION DOC]
    # HELPER: Lazily builds the graph training function used by train_on_batch and the array loop.
    # Recompiles loss and weighted metrics if sample weights changed, checks trainable weight
    # consistency, and raises a value error if the optimizer is a list. If no train function
    # exists yet or recompilation happened, temporarily restores the trainable state captured at
    # compile time, builds optimizer updates for the collected trainable weights plus layer
    # updates, and creates a backend function returning the total loss and metric tensors,
    # feeding inputs, targets, sample weights and (if symbolic) the learning phase. Stores it as
    # the train function and then restores the current trainable state. Returns nothing.
    has_recompiled = self._recompile_weights_loss_and_weighted_metrics()
    self._check_trainable_weights_consistency()
    if isinstance(self.optimizer, list):
      raise ValueError('The `optimizer` in `compile` should be a single '
                       'optimizer.')
    # If we have re-compiled the loss/weighted metric sub-graphs then create
    # train function even if one exists already. This is because
    # `_feed_sample_weights` list has been updated on re-compile.
    if getattr(self, 'train_function', None) is None or has_recompiled:
      # Restore the compiled trainable state.
      current_trainable_state = self._get_trainable_state()
      self._set_trainable_state(self._compiled_trainable_state)

      inputs = (self._feed_inputs +
                self._feed_targets +
                self._feed_sample_weights)
      if not isinstance(backend.symbolic_learning_phase(), int):
        inputs += [backend.symbolic_learning_phase()]

      with backend.get_graph().as_default():
        with backend.name_scope('training'):
          # Training updates
          updates = self.optimizer.get_updates(
              params=self._collected_trainable_weights, loss=self.total_loss)
          # Unconditional updates
          updates += self.get_updates_for(None)
          # Conditional updates relevant to this model
          updates += self.get_updates_for(self.inputs)

        metrics = self._get_training_eval_metrics()
        metrics_tensors = [
            m._call_result for m in metrics if hasattr(m, '_call_result')  # pylint: disable=protected-access
        ]

      with backend.name_scope('training'):
        # Gets loss and metrics. Updates weights at each call.
        fn = backend.function(
            inputs, [self.total_loss] + metrics_tensors,
            updates=updates,
            name='train_function',
            **self._function_kwargs)
        setattr(self, 'train_function', fn)

      # Restore the current trainable state
      self._set_trainable_state(current_trainable_state)

  def _make_test_function(self):
    # [FUNCTION DOC]
    # HELPER: Lazily builds the graph evaluation function.
    # Recompiles loss and weighted metrics if needed; if no test function exists or recompilation
    # happened, creates a backend function that takes inputs, targets and sample weights and
    # returns the total loss and metric tensors, applying only network state updates (no weight
    # updates). Stores it as the test function. Returns nothing.
    has_recompiled = self._recompile_weights_loss_and_weighted_metrics()
    # If we have re-compiled the loss/weighted metric sub-graphs then create
    # test function even if one exists already. This is because
    # `_feed_sample_weights` list has been updated on re-compile.
    if getattr(self, 'test_function', None) is None or has_recompiled:
      inputs = (self._feed_inputs +
                self._feed_targets +
                self._feed_sample_weights)

      with backend.get_graph().as_default():
        metrics = self._get_training_eval_metrics()
        metrics_tensors = [
            m._call_result for m in metrics if hasattr(m, '_call_result')  # pylint: disable=protected-access
        ]

      with backend.name_scope('evaluation'):
        updates = self.state_updates
        # Return loss and metrics, no gradient updates.
        # Does update the network states.
        fn = backend.function(
            inputs, [self.total_loss] + metrics_tensors,
            updates=updates,
            name='test_function',
            **self._function_kwargs)
        setattr(self, 'test_function', fn)

  def _make_predict_function(self):
    # [FUNCTION DOC]
    # HELPER: Lazily builds the graph prediction function.
    # Creates the predict function attribute if missing; if it is nothing, builds a backend
    # function from the feed inputs to the model outputs, applying network state updates only and
    # using any stored Session keyword arguments, inside a predict name scope. Returns nothing.
    if not hasattr(self, 'predict_function'):
      self.predict_function = None
    if self.predict_function is None:
      inputs = self._feed_inputs
      # Gets network outputs. Does not update weights.
      # Does update the network states.
      kwargs = getattr(self, '_function_kwargs', {})
      with backend.name_scope(ModeKeys.PREDICT):
        self.predict_function = backend.function(
            inputs,
            self.outputs,
            updates=self.state_updates,
            name='predict_function',
            **kwargs)

  def _make_execution_function(self, mode):
    # [FUNCTION DOC]
    # HELPER: Mode-based dispatcher used by the training loops.
    # KNOWN ISSUE: for a mode that is not train, test or predict it silently returns nothing
    # instead of raising an error.
    # Input: mode, one of the train, test or predict mode keys. Builds the corresponding graph
    # function lazily and returns it.
    if mode == ModeKeys.TRAIN:
      self._make_train_function()
      return self.train_function
    if mode == ModeKeys.TEST:
      self._make_test_function()
      return self.test_function
    if mode == ModeKeys.PREDICT:
      self._make_predict_function()
      return self.predict_function

  def _distribution_standardize_user_data(self,
                                          x,
                                          y=None,
                                          sample_weight=None,
                                          class_weight=None,
                                          batch_size=None,
                                          validation_split=0,
                                          shuffle=False,
                                          epochs=1,
                                          allow_partial_batch=False):
    # [FUNCTION DOC]
    # HELPER: Data preparation used by the distribution-strategy training loops.
    # KNOWN ISSUE: the docstring says a runtime error is raised if the model was never compiled,
    # but this method never raises that; it does raise not-implemented errors for class weights and
    # for sample weights on TPU, which are undocumented. Shuffling numpy data with no batch size
    # would fail because the buffer size multiplies the batch size.
    # Inputs: x (numpy data or dataset), y, sample_weight, class_weight (unsupported), batch_size,
    # validation_split, shuffle, epochs (repeat count for numpy data), allow_partial_batch.
    # Verifies dataset shuffling when requested. Inside the strategy scope: for numpy input,
    # converts lists to tuples, makes a strategy dataset (using a session in graph mode),
    # optionally shuffles with a buffer of at least 1024 or eight batches, repeats for epochs,
    # drops the remainder batch when static shapes are required or on TPU with evenly divisible
    # data, and batches it; for dataset input validates it. Returns the dataset.
    """Runs validation checks on input and target data passed by the user.

    This is called when using tf.distribute.Strategy to train, evaluate or serve
    the model.

    Args:
      x: Input data. A numpy array or `tf.data` dataset.
      y: Target data. A numpy array or None if x is a `tf.data` dataset.
      sample_weight: An optional sample-weight array passed by the user to
        weight the importance of each sample in `x`.
      class_weight: An optional class-weight array by the user to
        weight the importance of samples in `x` based on the class they belong
        to, as conveyed by `y`.
      batch_size: Integer batch size. If provided, it is used to run additional
        validation checks on stateful models.
      validation_split: Float between 0 and 1.
        Fraction of the training data to be used as validation data.
      shuffle: Boolean whether to shuffle the training data before each epoch.
      epochs: Integer epochs. If > 1, repeat the numpy training data epochs
        times when converting to training dataset.
      allow_partial_batch: Boolean whether to enforce that all batches have the
        same size.

    Returns:
      Dataset instance.

    Raises:
      ValueError: In case of invalid user-provided data.
      RuntimeError: If the model was never compiled.
    """
    if class_weight:
      raise NotImplementedError('`class_weight` is currently not supported '
                                'when using tf.distribute.Strategy.')

    if (sample_weight is not None and sample_weight.all() and
        backend.is_tpu_strategy(self._distribution_strategy)):
      raise NotImplementedError('`sample_weight` is currently not supported '
                                'when using TPUStrategy.')

    # Validates `steps` and `shuffle` arguments right at the beginning
    # since we use it to construct the dataset object.
    # TODO(anjalisridhar): Remove this check once we refactor the
    # _standardize_user_data code path. This check is already present elsewhere
    # in the codebase.
    if isinstance(x, data_types.DatasetV2):
      if shuffle:
        training_utils_v1.verify_dataset_shuffled(x)

    strategy = self._distribution_strategy
    with strategy.scope():
      # We should be sure to call get_session() inside the strategy.scope()
      # so the strategy can affect the session options.
      if ops.executing_eagerly_outside_functions():
        session = None
      else:
        session = backend.get_session()

      first_x_value = nest.flatten(x)[0]
      if isinstance(first_x_value, np.ndarray):
        x = training_utils.list_to_tuple(x)
        if y is not None:
          y = training_utils.list_to_tuple(y)
          if sample_weight is not None:
            sample_weight = training_utils.list_to_tuple(sample_weight)
            in_tuple = (x, y, sample_weight)
          else:
            in_tuple = (x, y)
        else:
          in_tuple = x

        ds = strategy.extended.experimental_make_numpy_dataset(in_tuple,
                                                               session=session)
        if shuffle:
          # We want a buffer size that is larger than the batch size provided by
          # the user and provides sufficient randomness. Note that larger
          # numbers introduce more memory usage based on the size of each
          # sample.
          ds = ds.shuffle(max(1024, batch_size * 8))
        if epochs > 1:
          ds = ds.repeat(epochs)

        # We need to use the drop_remainder argument to get a known static
        # input shape which is required for TPUs.
        drop_remainder = (not allow_partial_batch and
                          strategy.extended.experimental_require_static_shapes)

        # TODO(b/131720208): We still drop remainder here if number of examples
        # is divisible by batch size, as sometimes dynamic padder will time out
        # with keras.metrics.CategoricalAccuracy() metric.
        if backend.is_tpu_strategy(strategy) and not drop_remainder:
          dataset_size = first_x_value.shape[0]
          if dataset_size % batch_size == 0:
            drop_remainder = True

        x = ds.batch(batch_size, drop_remainder=drop_remainder)
      else:
        assert isinstance(x, data_types.DatasetV2)
        training_utils_v1.validate_dataset_input(x, y, sample_weight,
                                                 validation_split)
    return x

  def _standardize_user_data(self,
                             x,
                             y=None,
                             sample_weight=None,
                             class_weight=None,
                             batch_size=None,
                             check_steps=False,
                             steps_name='steps',
                             steps=None,
                             validation_split=0,
                             shuffle=False,
                             extract_tensors_from_dataset=False):
    # [FUNCTION DOC]
    # HELPER: Central input normalization used by all training, evaluation and prediction paths.
    # KNOWN ISSUE: the docstring lists a runtime error for a never-compiled model, but this method
    # raises none itself; compile is only triggered on the fly when an optimizer exists. When
    # inputs are symbolic and the model was just built and compiled, it returns empty lists and
    # nothing for the sample weights, not lists as the docstring suggests.
    # Inputs: x, y, sample_weight, class_weight, batch_size, check_steps (validate steps),
    # steps_name, steps, validation_split, shuffle, extract_tensors_from_dataset. Steps: validate
    # dataset or iterator input (and shuffling), optionally extract tensors; check steps; build
    # the model on the fly when it has no inputs; compile on the fly when not compiled but an
    # optimizer is set; short-circuit for symbolic graph inputs; otherwise standardize the
    # tensors. Returns a triple of inputs, targets and sample weights. Side effects: may set model
    # inputs, outputs and compile state.
    """Runs validation checks on input and target data passed by the user.

    Also standardizes the data to lists of arrays, in order.

    Also builds and compiles the model on the fly if it is a subclassed model
    that has never been called before (and thus has no inputs/outputs).

    This is a purely internal method, subject to refactoring at any time.

    Args:
      x: Input data. It could be:
        - A Numpy array (or array-like), or a list of arrays
          (in case the model has multiple inputs).
        - A TensorFlow tensor, or a list of tensors
          (in case the model has multiple inputs).
        - A dict mapping input names to the corresponding array/tensors,
          if the model has named inputs.
        - A `tf.data` dataset.
      y: Target data. Like the input data `x`,
        it could be either Numpy array(s) or TensorFlow tensor(s).
        It should be consistent with `x` (you cannot have Numpy inputs and
        tensor targets, or inversely). If `x` is a dataset, `y` should not be
        specified (since targets will be obtained from the iterator).
      sample_weight: An optional sample-weight array passed by the user to
        weight the importance of each sample in `x`.
      class_weight: An optional class-weight array by the user to
        weight the importance of samples in `x` based on the class they belong
        to, as conveyed by `y`. If both `sample_weight` and `class_weight` are
        provided, the weights are multiplied.
      batch_size: Integer batch size. If provided, it is used to run additional
        validation checks on stateful models.
      check_steps: boolean, True if we want to check for validity of `steps` and
        False, otherwise. For example, when we are standardizing one batch of
        data for train_on_batch/predict_on_batch/test_on_batch APIs, `steps`
        value is not required and we should not check for its validity in these
        cases.
      steps_name: The public API's parameter name for `steps`.
      steps: Integer or `None`. Total number of steps (batches of samples) to
        execute.
      validation_split: Float between 0 and 1.
        Fraction of the training data to be used as validation data.
      shuffle: Boolean whether to shuffle the training data before each epoch.
      extract_tensors_from_dataset: Boolean. When `x` is a dataset instance,
        this indicates whether to extract actual tensors from the dataset or
        instead output the dataset instance itself.
        Set to True when calling from `train_on_batch`/etc.

    Returns:
      A tuple of 3: inputs (arrays or dicts, depending on whether `x` was a dict
      or not), target arrays, sample-weight arrays.
      If the model's input and targets are symbolic, these lists are empty
      (since the model takes no user-provided data, instead the data comes
      from the symbolic inputs/targets).

    Raises:
      ValueError: In case of invalid user-provided data.
      RuntimeError: If the model was never compiled.
    """
    if isinstance(x, (data_types.DatasetV1, data_types.DatasetV2)):
      # Graph mode dataset. We'll pass the dataset as-is (unless
      # `extract_tensors_from_dataset` is True, in which case we extract
      # the tensors from the dataset and we output them.
      training_utils_v1.validate_dataset_input(x, y, sample_weight,
                                               validation_split)
      if shuffle:
        training_utils_v1.verify_dataset_shuffled(x)

      is_dataset = True
      if extract_tensors_from_dataset:
        # We do this for `train_on_batch`/etc.
        x, y, sample_weight = training_utils_v1.extract_tensors_from_dataset(x)
    elif isinstance(x, iterator_ops.Iterator):
      # Graph mode iterator. We extract the symbolic tensors.
      training_utils_v1.validate_dataset_input(x, y, sample_weight,
                                               validation_split)
      iterator = x
      x, y, sample_weight = training_utils_v1.unpack_iterator_input(iterator)
      is_dataset = True
    else:
      is_dataset = False

    # Validates `steps` argument based on x's type.
    if check_steps:
      training_utils_v1.check_steps_argument(x, steps, steps_name)

    # First, we build the model on the fly if necessary.
    if not self.inputs:
      all_inputs, y_input, dict_inputs = self._build_model_with_inputs(x, y)
      is_build_called = True
    else:
      all_inputs = []
      # Whether this is a subclassed model that expects dictionary inputs
      # rather than list inputs (e.g. FeatureColumn-based models).
      dict_inputs = isinstance(self.inputs, dict)
      is_build_called = False
      y_input = y

    # Second, we compile the model on the fly if necessary, mostly for subclass
    # models.
    is_compile_called = False
    if not self._is_compiled and self.optimizer:
      self._compile_from_inputs(all_inputs, y_input, x, y)
      is_compile_called = True

    # In graph mode, if we had just set inputs and targets as symbolic tensors
    # by invoking build and compile on the model respectively, we do not have to
    # feed anything to the model. Model already has input and target data as
    # part of the graph.
    # Note: in this case, `any` and `all` are equivalent since we disallow
    # mixed symbolic/value inputs.

    # self.run_eagerly is not free to compute, so we want to reuse the value.
    run_eagerly = self.run_eagerly

    if (not run_eagerly and is_build_called and is_compile_called and
        not is_dataset  and any(_is_symbolic_tensor(v) for v in all_inputs)):
      return [], [], None

    return self._standardize_tensors(
        x, y, sample_weight,
        run_eagerly=run_eagerly,
        dict_inputs=dict_inputs,
        is_dataset=is_dataset,
        class_weight=class_weight,
        batch_size=batch_size)

  def _standardize_tensors(self, x, y, sample_weight, run_eagerly, dict_inputs,
                           is_dataset, class_weight=None, batch_size=None):
    # [FUNCTION DOC]
    # HELPER: Second half of standardizing user data, shared by the eager and graph paths.
    # Inputs: x, y, sample_weight, run_eagerly flag, dict_inputs flag (inputs are a mapping),
    # is_dataset flag, class_weight, batch_size. Chooses feed names and shapes by mode (no shape
    # validation when eager or for subclassed graphs), standardizes inputs to lists of arrays
    # (not for datasets), converts scipy sparse values, checks the input structure against the
    # model inputs, and if targets are given standardizes targets, sample weights and class
    # weights, checks array lengths and loss/target compatibility (not under a strategy), and
    # handles partial sample weights. For stateful models raises a value error when the number
    # of samples is not a multiple of the batch size. Returns a triple of inputs (a mapping when
    # dict inputs were used), targets (empty list if no targets) and sample weights (nothing if no
    # targets).
    if run_eagerly:
      # In eager mode, do not do shape validation
      # since the network has no input nodes (placeholders) to be fed.
      feed_input_names = self.input_names
      feed_input_shapes = None
    elif not self._is_graph_network:
      # Case: symbolic-mode subclassed network. Do not do shape validation.
      feed_input_names = self._feed_input_names
      feed_input_shapes = None
    else:
      # Case: symbolic-mode graph network.
      # In this case, we run extensive shape validation checks.
      feed_input_names = self._feed_input_names
      feed_input_shapes = self._feed_input_shapes

    # Standardize the inputs.
    if not isinstance(x, (data_types.DatasetV1, data_types.DatasetV2)):
      # TODO(fchollet): run static checks with dataset output shape(s).
      x = training_utils_v1.standardize_input_data(
          x,
          feed_input_names,
          feed_input_shapes,
          check_batch_axis=False,  # Don't enforce the batch size.
          exception_prefix='input')

    # Get typespecs for the input data and sanitize it if necessary.
    # TODO(momernick): This should be capable of doing full input validation
    # at all times - validate that this is so and refactor the standardization
    # code.
    if isinstance(x, data_types.DatasetV2):
      x_shapes = dataset_ops.get_structure(x)
      if isinstance(x_shapes, tuple):
        # If the output of a Dataset is a tuple, we assume it's either of the
        # form (x_data, y_data) or (x_data, y_data, sample_weights). In either
        # case, we only care about x_data here.
        x_shapes = x_shapes[0]
    else:
      flat_inputs = nest.flatten(x, expand_composites=False)
      flat_expected_inputs = nest.flatten(self.inputs, expand_composites=False)
      converted_x = []
      for (a, b) in zip(flat_inputs, flat_expected_inputs):
        converted_x.append(_convert_scipy_sparse_tensor(a, b))
      x = nest.pack_sequence_as(x, converted_x, expand_composites=False)

      def _type_spec_from_value(value):
        # [FUNCTION DOC]
        # HELPER: Nested utility for the structure check in tensor standardization.
        # Input: value, a composite tensor, an array-like with shape and dtype, or other value.
        # Returns its type specification: the stored one for extension types, a tensor spec built
        # from shape and dtype for array-likes (without converting data to a tensor), otherwise
        # whatever the generic type-spec inference returns.
        """Grab type_spec without converting array-likes to tensors."""
        if tf_utils.is_extension_type(value):
          return value._type_spec  # pylint: disable=protected-access
        # Get a TensorSpec for array-like data without
        # converting the data to a Tensor
        if hasattr(value, 'shape') and hasattr(value, 'dtype'):
          return tensor_spec.TensorSpec(value.shape, value.dtype)
        else:
          return type_spec.type_spec_from_value(value)

      x_shapes = nest.map_structure(_type_spec_from_value, x)

    flat_inputs = nest.flatten(x_shapes, expand_composites=False)
    flat_expected_inputs = nest.flatten(self.inputs, expand_composites=False)
    for (a, b) in zip(flat_inputs, flat_expected_inputs):
      nest.assert_same_structure(a, b, expand_composites=True)

    if y is not None:
      # Prepare self._sample_weight_modes. List with the same length as
      # model outputs.
      training_utils_v1.prepare_sample_weight_modes(self._training_endpoints,
                                                    self.sample_weight_mode)
      feed_output_names = self._feed_output_names
      feed_sample_weight_modes = self._sample_weight_modes
      if not self._is_graph_network:
        feed_output_shapes = None
      else:
        feed_output_shapes = self._feed_output_shapes

      # Standardize the outputs.
      y = training_utils_v1.standardize_input_data(
          y,
          feed_output_names,
          # Don't enforce target shapes to match output shapes.
          # Precise checks will be run in `check_loss_and_target_compatibility`.
          shapes=None,
          check_batch_axis=False,  # Don't enforce the batch size.
          exception_prefix='target')

      # Generate sample-wise weight values given the `sample_weight` and
      # `class_weight` arguments.
      sample_weights = training_utils_v1.standardize_sample_weights(
          sample_weight, feed_output_names)
      class_weights = training_utils_v1.standardize_class_weights(
          class_weight, feed_output_names)

      sample_weights = [
          training_utils_v1.standardize_weights(ref, sw, cw, mode)
          for (ref, sw, cw, mode) in zip(y, sample_weights, class_weights,
                                         feed_sample_weight_modes)
      ]
      # Check that all arrays have the same length.
      if not self._distribution_strategy:
        training_utils_v1.check_array_lengths(x, y, sample_weights)
        if self._is_graph_network and not run_eagerly:
          # Additional checks to avoid users mistakenly using improper loss fns.
          training_utils_v1.check_loss_and_target_compatibility(
              y, self._feed_loss_fns, feed_output_shapes)

      sample_weights, _, _ = training_utils.handle_partial_sample_weights(
          y, sample_weights, feed_sample_weight_modes, check_all_flat=True)
    else:
      y = []
      sample_weights = None

    if self.stateful and batch_size and not is_dataset:
      # Check that for stateful networks, number of samples is a multiple
      # of the static batch size.
      if x[0].shape[0] % batch_size != 0:
        raise ValueError('In a stateful network, '
                         'you should only pass inputs with '
                         'a number of samples that can be '
                         'divided by the batch size. Found: ' +
                         str(x[0].shape[0]) + ' samples')

    # If dictionary inputs were provided, we return a dictionary as well.
    if dict_inputs and not isinstance(x, (data_types.DatasetV1,
                                          data_types.DatasetV2)):
      x = dict(zip(feed_input_names, x))
    return x, y, sample_weights

  def _build_model_with_inputs(self, inputs, targets):
    # [FUNCTION DOC]
    # HELPER: On-the-fly model building for subclassed models without known inputs.
    # Inputs: inputs (array, list, mapping, dataset, or iterator) and targets. For datasets, first
    # extracts the tensors. Validates the input types, flattens inputs into a list (mappings are
    # ordered by sorted key), and raises a value error if any input is a composite or sparse
    # value that was not declared explicitly. Then derives the spec to build with: tensor specs
    # for dataset-like inputs (casting floating dtypes to the model dtype if no inputs set yet),
    # floating-cast values if tensors are present, else inputs as is. Calls the model input setter
    # (side effect: sets model inputs and outputs). Returns the flat list of inputs, the targets
    # (possibly extracted from the dataset) and a flag telling whether inputs were a mapping.
    """Build the model (set model inputs/outputs), mainly for subclass model."""
    processed_inputs = []
    is_dict_inputs = False
    orig_inputs = inputs
    # We need to use `inputs` to set the model inputs.
    # If input data is a dataset iterator in graph mode or if it is an eager
    # iterator and only one batch of samples is required, we fetch the data
    # tensors from the iterator and then standardize them.
    if isinstance(inputs, (data_types.DatasetV1, data_types.DatasetV2)):
      inputs, targets, _ = training_utils_v1.extract_tensors_from_dataset(
          inputs)
    # We type-check that `inputs` and `targets` are either single arrays
    # or lists of arrays, and extract a flat list of inputs from the passed
    # structure.
    training_utils_v1.validate_input_types(inputs, orig_inputs)

    if isinstance(inputs, (list, tuple)):
      processed_inputs += list(inputs)
    elif isinstance(inputs, dict):
      is_dict_inputs = True
      keys = sorted(inputs.keys())
      processed_inputs = [inputs[k] for k in keys]
    else:
      processed_inputs.append(inputs)
    # Now that we have a flat set of inputs, we make sure that none of them
    # are CompositeTensors or CompositeTensorValues of any type (or scipy
    # sparse arrays, which we treat as SparseTensor values). We cannot safely
    # infer input data from an arbitrary composite tensor, so we don't try -
    # users should explicitly add composite tensor inputs to their subclassed
    # models.
    for input_tensor in processed_inputs:
      if training_utils_v1.is_composite_or_composite_value(input_tensor):
        # TODO(b/132691975): Document subclass-model CT input handling.
        raise ValueError(
            'All SparseTensor and RaggedTensor inputs must be explicitly '
            'declared using a keras.Input() with sparse=True or ragged=True. '
            'We found an undeclared input %s. For Sequential models, please '
            'add a keras.Input() as your first Layer. For subclassed models, '
            'please call self._set_inputs() on your input set, which you can '
            'create using keras.Input() for each input to your model.' %
            (input_tensor,))
    # Build the model using the retrieved inputs (value or symbolic).
    # If values are generated from a dataset, then in symbolic-mode
    # placeholders will be created to match the value shapes.
    if isinstance(orig_inputs, (data_types.DatasetV1, data_types.DatasetV2,
                                iterator_ops.Iterator)):
      if not self.inputs:
        # For subclassed models, a robust input spec is not available so we
        # must cast to the model dtype.
        inputs = training_utils_v1.cast_if_floating_dtype(inputs, self.dtype)

      def create_tensor_spec(t):
        # [FUNCTION DOC]
        # HELPER: Nested mapper used to turn each tensor of the input structure into a spec.
        # Input: t, a tensor or array with shape and dtype. Returns a tensor specification with
        # the same shape and dtype.
        return tensor_spec.TensorSpec(t.shape, t.dtype)

      cast_inputs = nest.map_structure(create_tensor_spec, inputs)
    elif training_utils_v1.has_tensors(inputs):
      cast_inputs = training_utils_v1.cast_if_floating_dtype(inputs)
    else:
      cast_inputs = inputs
    self._set_inputs(cast_inputs)
    return processed_inputs, targets, is_dict_inputs

  def _compile_from_inputs(self, all_inputs, target, orig_inputs, orig_target):
    # [FUNCTION DOC]
    # HELPER: Compiles a subclassed model on the fly after its inputs became known.
    # Inputs: all_inputs (flat list of inputs, extended in place with the targets), target (the
    # targets), orig_inputs and orig_target (the original user values, used for type checks and
    # error messages). Casts tensor targets when dtypes mismatch outputs, validates target types,
    # and raises a value error when inputs and targets mix numpy arrays and tensors. Computes
    # target tensors from symbolic targets (none for datasets or eager mode). Finally calls compile
    # again with the previously stored optimizer, loss, metrics, weights, sample weight mode and
    # run settings. Returns nothing.
    if target is not None:
      # We need to use `y` to set the model targets.
      if training_utils_v1.has_tensors(target):
        target = training_utils_v1.cast_if_floating_dtype_and_mismatch(
            target, self.outputs)
      training_utils_v1.validate_input_types(
          target, orig_target, allow_dict=False, field_name='target')
      if isinstance(target, (list, tuple)):
        all_inputs += list(target)
      else:
        all_inputs.append(target)
    # Type check that all inputs are *either* value *or* symbolic.
    # TODO(fchollet): this check could be removed in Eager mode?
    if any(tensor_util.is_tf_type(v) for v in all_inputs):
      if not all(tensor_util.is_tf_type(v) for v in all_inputs):
        raise ValueError('Do not pass inputs that mix Numpy arrays and '
                         'TensorFlow tensors. '
                         'You passed: x=' + str(orig_inputs) +
                         '; y=' + str(orig_target))
    is_dataset = isinstance(orig_inputs, (data_types.DatasetV1,
                                          data_types.DatasetV2,
                                          iterator_ops.Iterator))
    if is_dataset or context.executing_eagerly():
      target_tensors = None
    else:
      # Handle target tensors if any passed.
      if target is not None:
        if not isinstance(target, (list, tuple)):
          target = [target]
        target_tensors = [v for v in target if _is_symbolic_tensor(v)]
      else:
        target_tensors = None

    self.compile(
        optimizer=self.optimizer,
        loss=self.loss,
        metrics=self._compile_metrics,
        weighted_metrics=self._compile_weighted_metrics,
        loss_weights=self.loss_weights,
        target_tensors=target_tensors,
        sample_weight_mode=self.sample_weight_mode,
        run_eagerly=self.run_eagerly,
        experimental_run_tf_function=self._experimental_run_tf_function)

  # TODO(omalleyt): Consider changing to a more descriptive function name.
  def _set_inputs(self, inputs, outputs=None, training=None):
    # [FUNCTION DOC]
    # Sets the model's input and output specs from the first data seen (for subclassed models).
    # Inputs: inputs (array, list, placeholders, data tensors or specs), outputs (optional tensors;
    # if nothing they are obtained by calling the model on the inputs), training (optional
    # boolean for symbolic mode; in graph mode with nothing the Keras learning phase is used, and
    # it is passed only if the model's call accepts a training argument). If calling the model
    # raises a not-implemented error (dynamic model without output shape computation) the
    # outputs stay nothing. Raises a value error from the input setter if inputs are already set
    # or dict inputs go to a Sequential model without a feature layer. Side effects: records the
    # save spec, sets input and output attributes. Returns nothing.
    """Set model's input and output specs based on the input data received.

    This is to be used for Model subclasses, which do not know at instantiation
    time what their inputs look like.

    Args:
      inputs: Single array, or list of arrays. The arrays could be placeholders,
        Numpy arrays, data tensors, or TensorSpecs.
        - if placeholders: the model is built on top of these placeholders,
          and we expect Numpy data to be fed for them when calling `fit`/etc.
        - if Numpy data or TensorShapes: we create placeholders matching the
          TensorShapes or shapes of the Numpy arrays. We expect Numpy data to be
          fed for these placeholders when calling `fit`/etc.
        - if data tensors: the model is built on top of these tensors.
          We do not expect any Numpy data to be provided when calling `fit`/etc.
      outputs: None, a data tensor, or a list of tensors. If None, the
        outputs will be determined by invoking `self.call()`, otherwise the
        provided value will be used.
      training: Boolean or None. Only relevant in symbolic mode. Specifies
        whether to build the model's graph in inference mode (False), training
        mode (True), or using the Keras learning phase (None).
    Raises:
      ValueError: If dict inputs are passed to a Sequential Model where the
        first layer isn't FeatureLayer.
    """
    self._set_save_spec(inputs)
    inputs = self._set_input_attrs(inputs)

    if outputs is None:
      kwargs = {}
      if self._expects_training_arg:
        # In V2 mode, feeding `training=None` is not allowed because any value
        # explicitly passed by the user is respected, even `None`.`
        if training is None and not ops.executing_eagerly_outside_functions():
          training = backend.learning_phase()
        if training is not None:
          kwargs['training'] = training
      try:
        outputs = self(inputs, **kwargs)
      except NotImplementedError:
        # This Model or a submodel is dynamic and hasn't overridden
        # `compute_output_shape`.
        outputs = None

    self._set_output_attrs(outputs)

  @trackable.no_automatic_dependency_tracking
  def _set_input_attrs(self, inputs):
    # [FUNCTION DOC]
    # HELPER: Input half of the input/output setter.
    # Input: inputs (tensor, shape, mapping or array-like). Raises a value error if model inputs
    # are already set. For an unbuilt Sequential model records the build input shape (batch axis
    # unknown, rest taken from the tensor, shape or array; mappings need a feature layer first,
    # else value error). Casts inputs to the compute dtype, then creates symbolic inputs (reusing
    # given tensors or making placeholders) and stores them with the input names, and records which
    # inputs are placeholders (their names, tensors and shapes) as the feed inputs. Dependency
    # tracking is disabled. Returns the symbolic inputs.
    """Sets attributes related to the inputs of the Model."""
    if self.inputs:
      raise ValueError('Model inputs are already set.')

    if self.__class__.__name__ == 'Sequential' and not self.built:
      if tensor_util.is_tf_type(inputs):
        input_shape = (None,) + tuple(inputs.shape.as_list()[1:])
      elif isinstance(inputs, tensor_shape.TensorShape):
        input_shape = (None,) + tuple(inputs.as_list()[1:])
      elif isinstance(inputs, dict):
        # We assert that the first layer is a FeatureLayer.
        if not training_utils_v1.is_feature_layer(self.layers[0]):
          raise ValueError('Passing a dictionary input to a Sequential Model '
                           'which doesn\'t have FeatureLayer as the first layer'
                           ' is an error.')
        input_shape = (None,)
      else:
        input_shape = (None,) + tuple(inputs.shape[1:])
      self._build_input_shape = input_shape

    # Cast inputs to the compute dtype. This is primarily used
    # when saving to determine the correct dtype in the input signature.
    inputs = self._maybe_cast_inputs(inputs)

    # On-the-fly setting of symbolic model inputs (either by using the tensor
    # provided, or by creating a placeholder if Numpy data was provided).
    model_inputs = training_utils_v1.ModelInputs(inputs)
    inputs = model_inputs.get_symbolic_inputs()
    self.inputs = model_inputs.get_symbolic_inputs(return_single_as_list=True)
    self.input_names = model_inputs.get_input_names()

    self._feed_inputs = []
    self._feed_input_names = []
    self._feed_input_shapes = []

    for k, v in model_inputs.as_dict():
      if backend.is_placeholder(v):
        self._feed_input_names.append(k)
        self._feed_inputs.append(v)
        self._feed_input_shapes.append(backend.int_shape(v))

    return inputs

  @trackable.no_automatic_dependency_tracking
  def _set_output_attrs(self, outputs):
    # [FUNCTION DOC]
    # HELPER: Output half of the input/output setter.
    # Input: outputs (a tensor or nested structure of tensors, or nothing). Flattens them in the
    # standard order (the data adapter depends on that order), stores them as the model outputs,
    # generates default output names and marks the model as built. Dependency tracking is
    # disabled. Returns nothing.
    """Sets attributes related to the outputs of the Model."""
    # NOTE(taylorrobie): This convention cannot be changed without updating the
    #                    data adapter since it assumes nest.flatten ordering.
    outputs = nest.flatten(outputs)
    self.outputs = outputs
    self.output_names = training_utils_v1.generic_output_names(outputs)
    # TODO(scottzhu): Should we cleanup the self._training_endpoints here?
    self.built = True

  @property
  def _targets(self):
    # [FUNCTION DOC]
    # HELPER: Read-only property used when building metrics and loss.
    # Returns the list of target tensors of the training endpoints that have a training target
    # (feedable or not).
    """The output target tensors for the model."""
    return [
        e.training_target.target
        for e in self._training_endpoints
        if e.has_training_target()
    ]

  @property
  def _feed_targets(self):
    # [FUNCTION DOC]
    # HELPER: Read-only property used to build the graph functions.
    # Returns the target tensors that must be fed with data (endpoints with a feedable target).
    return [
        e.training_target.target
        for e in self._training_endpoints
        if e.has_feedable_training_target()
    ]

  @property
  def _feed_output_names(self):
    # [FUNCTION DOC]
    # HELPER: Read-only property used when standardizing targets.
    # Returns the output names of the endpoints whose target must be fed with data.
    return [
        e.output_name
        for e in self._training_endpoints
        if e.has_feedable_training_target()
    ]

  @property
  def _feed_output_shapes(self):
    # [FUNCTION DOC]
    # HELPER: Read-only property used when validating targets.
    # Returns the expected target shapes (entries may be nothing for custom losses) of the
    # endpoints whose target must be fed with data.
    return [
        e.feed_output_shape
        for e in self._training_endpoints
        if e.has_feedable_training_target()
    ]

  @property
  def _feed_loss_fns(self):
    # [FUNCTION DOC]
    # HELPER: Read-only property used for loss/target compatibility checks.
    # Returns the loss functions of the endpoints whose target must be fed with data.
    return [
        e.loss_fn
        for e in self._training_endpoints
        if e.has_feedable_training_target()
    ]

  @property
  def _loss_weights_list(self):
    # [FUNCTION DOC]
    # HELPER: Read-only property returning the loss weight of every training endpoint as a list,
    # in output order.
    return [e.loss_weight for e in self._training_endpoints]

  @property
  def _output_loss_metrics(self):
    # [FUNCTION DOC]
    # HELPER: Read-only property used by the eager steps and metric reporting.
    # Returns the list of per-output loss metric objects that exist (only created for
    # multi-output models), or nothing if the model has no training endpoints attribute yet.
    if hasattr(self, '_training_endpoints'):
      return [
          e.output_loss_metric
          for e in self._training_endpoints
          if e.output_loss_metric is not None
      ]
    return None

  @property
  def sample_weights(self):
    # [FUNCTION DOC]
    # Read-only property returning the sample weight tensor of each training endpoint in output
    # order (entries are nothing for outputs without sample weights).
    return [e.sample_weight for e in self._training_endpoints]

  @property
  def _sample_weight_modes(self):
    # [FUNCTION DOC]
    # HELPER: Read-only property returning the sample weight mode (temporal, sample-wise or
    # nothing) of each training endpoint in output order.
    return [e.sample_weight_mode for e in self._training_endpoints]

  @property
  def _feed_sample_weights(self):
    # [FUNCTION DOC]
    # HELPER: Read-only property used to build the graph functions.
    # Returns the sample weight tensors that exist (endpoints whose sample weight is not nothing).
    return [e.sample_weight for e in self._training_endpoints
            if e.sample_weight is not None]

  def _maybe_load_initial_epoch_from_ckpt(self, initial_epoch, mode):
    # [FUNCTION DOC]
    # HELPER: Used by training loops for multi-worker fault tolerance.
    # Inputs: initial_epoch (the epoch passed to fit) and mode (the running mode). If a training
    # state object exists, asks it for the epoch to resume from after worker recovery; otherwise
    # returns the given initial epoch unchanged. Returns an integer epoch.
    """Maybe load initial epoch from ckpt considering possible worker recovery.

    Refer to tensorflow/python/keras/distribute/worker_training_state.py
    for more information.

    Args:
      initial_epoch: The original initial_epoch user passes in in `fit()`.
      mode: The mode for running `model.fit()`.

    Returns:
      If the training is recovering from previous failure under multi-worker
      training setting, return the epoch the training is supposed to continue
      at. Otherwise, return the `initial_epoch` the user passes in.
    """
    if self._training_state is not None:
      return self._training_state.maybe_load_initial_epoch_from_ckpt(
          initial_epoch, mode)
    return initial_epoch

  def _get_training_eval_metrics(self):
    # [FUNCTION DOC]
    # HELPER: Collects the metrics reported and reset during training and evaluation.
    # Returns a new list: the per-output loss metrics followed by all model metrics (compile
    # metrics, weighted metrics and metrics added through the add-metric API).
    """Returns all the metrics that are to be reported.

    This includes the output loss metrics, compile metrics/weighted metrics,
    add_metric metrics.
    """
    metrics = []
    metrics.extend(getattr(self, '_output_loss_metrics', None) or [])
    metrics.extend(getattr(self, 'metrics', None) or [])
    return metrics

  def _assert_compile_was_called(self):
    # [FUNCTION DOC]
    # HELPER: Guard used by fit, evaluate and the on-batch methods.
    # Raises a runtime error telling the user to compile the model first if compile has not been
    # called (this is different from the model being fully compiled). Returns nothing.
    # Checks whether `compile` has been called. If it has been called,
    # then the optimizer is set. This is different from whether the
    # model is compiled
    # (i.e. whether the model is built and its inputs/outputs are set).
    if not self._compile_was_called:
      raise RuntimeError('You must compile your model before '
                         'training/testing. '
                         'Use `model.compile(optimizer, loss)`.')

  def _in_multi_worker_mode(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it returns whether the model is in multi-worker mode, but
    # with no strategy it returns the falsy strategy value (nothing) rather than a boolean.
    # Uses the model's distribution strategy, or else the strategy of the current scope, and asks
    # its extended object whether it runs in multi-worker mode. Returns that answer, or a falsy
    # value when no strategy exists. No side effects.
    """Method to infer if this `Model` is working in multi-worker settings.

    Multi-worker training refers to the setup where the training is
    distributed across multiple workers, as opposed to the case where
    only a local process performs the training. This function is
    used to infer for example whether or not a distribute coordinator
    should be run, and thus TensorFlow servers should be started for
    communication with other servers in the cluster, or whether or not
    saving/restoring checkpoints is relevant for preemption fault tolerance.

    Experimental. Signature and implementation are subject to change.

    Returns:
      Whether this model indicates it's working in multi-worker settings.
    """
    strategy = self._distribution_strategy

    # Otherwise, use the strategy whose scope this is in.
    if not strategy and distribute_lib.has_strategy():
      strategy = distribute_lib.get_strategy()
    return strategy and strategy.extended._in_multi_worker_mode()  # pylint: disable=protected-access

  @property
  def _trackable_saved_model_saver(self):
    # [FUNCTION DOC]
    # HELPER: Property used by the SavedModel machinery.
    # Returns a new model saver object that serializes this model for the SavedModel format.
    return model_serialization.ModelSavedModelSaver(self)

  def _get_compile_args(self, user_metrics=True):
    # [FUNCTION DOC]
    # HELPER: Used when saving or cloning to recover the arguments given to compile.
    # Input: user_metrics, accepted for interface compatibility and ignored. Raises a runtime
    # error if compile was never called. Returns a mapping with the loss, compile metrics, loss
    # weights, sample weight mode and weighted metrics.
    del user_metrics
    self._assert_compile_was_called()
    kwargs = {
        'loss': self.loss,
        'metrics': self._compile_metrics,
        'loss_weights': self.loss_weights,
        'sample_weight_mode': self.sample_weight_mode,
        'weighted_metrics': self._compile_weighted_metrics,
    }
    return kwargs

  @property
  def _compile_was_called(self):
    # [FUNCTION DOC]
    # HELPER: Read-only property overriding the parent's notion of whether compile ran.
    # Returns the flag recording whether the v1 compile method has been called.
    return self._v1_compile_was_called


class DistributedCallbackModel(Model):
  # [CLASS DOC]
  # HELPER: Wrapper model handed to callbacks when training with a distribution strategy.
  # It is built from the first replicated model (copying only its optimizer) and remembers the
  # original model. Saving and weight loading are redirected so callbacks such as checkpointing
  # operate on the original model and then synchronize the replicas. Reading attributes not set
  # on the wrapper logs a warning because they may not be populated correctly.
  """Model that is used for callbacks with tf.distribute.Strategy."""

  def __init__(self, model):
    # [FUNCTION DOC]
    # Constructor. Input: model, the first replicated model. Calls the parent constructor with no
    # arguments and copies only the optimizer from the given model. Returns nothing.
    super(DistributedCallbackModel, self).__init__()
    self.optimizer = model.optimizer

  def set_original_model(self, orig_model):
    # [FUNCTION DOC]
    # Records the original (non-replicated) model that save and load operations are forwarded to.
    # Input: orig_model, the user's model. Returns nothing; sets an attribute.
    self._original_model = orig_model

  def save_weights(self, filepath, overwrite=True, save_format=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: it forwards to an attribute named replicated model on this same wrapper object,
    # which is never set on it (only the original model is stored), so the call goes through the
    # warning attribute fallback and is likely to fail rather than save anything.
    # Intended to save weights through the replicated model. Inputs: filepath, overwrite
    # (default true), save_format (optional format choice). Returns nothing.
    self._replicated_model.save_weights(filepath, overwrite=overwrite,
                                        save_format=save_format)

  def save(self, filepath, overwrite=True, include_optimizer=True):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the overwrite and include_optimizer arguments are ignored; the original model is
    # always saved with overwrite enabled and without the optimizer.
    # Saves the distributed weights through the original model. Input: filepath. Copies this
    # wrapper's weights into the original model, then saves the original model to the path.
    # Side effects: mutates the original model's weights and writes a file. Returns nothing.
    # save weights from the distributed model to the original model
    distributed_model_weights = self.get_weights()
    self._original_model.set_weights(distributed_model_weights)
    # TODO(anjalisridhar): Do we need to save the original model here?
    # Saving the first replicated model works as well.
    self._original_model.save(filepath, overwrite=True, include_optimizer=False)

  def load_weights(self, filepath, by_name=False):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the by_name argument is ignored (loading is always topological), and unlike the
    # parent method there is no skip-mismatch parameter.
    # Loads weights into the original model from filepath, then copies the original model's
    # weights to each replica of the distribution strategy. Returns nothing.
    self._original_model.load_weights(filepath, by_name=False)
    # Copy the weights from the original model to each of the replicated models.
    orig_model_weights = self._original_model.get_weights()
    distributed_training_utils_v1.set_weights(
        self._original_model._distribution_strategy, self,  # pylint: disable=protected-access
        orig_model_weights)

  def __getattr__(self, item):
    # [FUNCTION DOC]
    # Fallback attribute lookup, called only when normal lookup fails. Input: item, the attribute
    # name. Logs a warning that the attribute may not have been set correctly unless it is one of
    # the two internal names for setattr tracking and layers, then defers to the parent fallback
    # and returns its result (which may raise an attribute error).
    # Allowed attributes of the model that can be accessed by the user
    # during a callback.
    if item not in ('_setattr_tracking', '_layers'):
      logging.warning('You are accessing attribute ' + item + ' of the '
                      'DistributedCallbackModel that may not have been set '
                      'correctly.')
    return super(DistributedCallbackModel, self).__getattr__(item)


class _TrainingEndpoint(object):
  # [CLASS DOC]
  # HELPER: Internal per-output record used by the v1 Model.
  # Bundles one model output tensor with its name, loss function, loss weight, training target
  # (a training-target container), per-output loss metric, sample weight tensor and sample weight
  # mode, so these can be read from one place. Created by compile, one per output; the target and
  # sample weights are populated later. Provides predicates (skip target, feedable target),
  # naming, expected feed shape and sample-weight placeholder creation.
  """A container for the training output/target and related entities.

  In the case of model with multiple outputs, there is a one-to-one mapping
  between model output (y_pred), model target (y_true), loss, metrics etc.
  By unifying these entities into one class, different entity can access
  information between each other, rather than currently access different list of
  attributes of the model.
  """

  def __init__(self,
               output,
               output_name,
               loss_fn,
               loss_weight=None,
               training_target=None,
               output_loss_metric=None,
               sample_weight=None,
               sample_weight_mode=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: output (model output tensor), output_name (unique name), loss_fn (loss
    # function, nothing means the output has no loss), and optional loss_weight, training_target,
    # output_loss_metric, sample_weight and sample_weight_mode (temporal, sample-wise or
    # nothing). Simply stores each into private attributes. Returns nothing.
    """Initialize the _TrainingEndpoint.

    Note that the output and output_name should be stable as long as the model
    structure doesn't change. The training_target suppose to be mutable since
    the information is provided via `compile()`

    Args:
      output: the output tensor of the model.
      output_name: the unique name of the output tensor.
      loss_fn: the loss function for the output tensor.
      loss_weight: float, the weights for the loss.
      training_target: the _TrainingTarget for the model.
      output_loss_metric: the metric object for the loss function.
      sample_weight: the weights for how a sample is weighted during metric and
        loss calculation. Could be None.
      sample_weight_mode: string, 'temporal', 'samplewise' or None. The mode for
        how the sample_weight is populated.
    """
    self._output = output
    self._output_name = output_name
    self._loss_fn = loss_fn
    self._loss_weight = loss_weight
    self._training_target = training_target
    self._output_loss_metric = output_loss_metric
    self._sample_weight = sample_weight
    self._sample_weight_mode = sample_weight_mode

  @property
  def output(self):
    # [FUNCTION DOC]
    # Read-only property: the model output tensor of this endpoint.
    return self._output

  @property
  def output_name(self):
    # [FUNCTION DOC]
    # Read-only property: the unique name of this endpoint's output.
    return self._output_name

  @property
  def shape(self):
    # [FUNCTION DOC]
    # Read-only property: the static shape of the output tensor as a tuple of integers (unknown
    # dimensions are nothing), computed on each access.
    return backend.int_shape(self.output)

  @property
  def loss_fn(self):
    # [FUNCTION DOC]
    # Read-only property: the loss function (or loss object) for this output, or nothing if the
    # output has no loss.
    return self._loss_fn

  @property
  def loss_weight(self):
    # [FUNCTION DOC]
    # Property getter: the scalar weight applied to this output's loss (may be nothing until
    # loss weights have been prepared).
    return self._loss_weight

  @loss_weight.setter
  def loss_weight(self, value):
    # [FUNCTION DOC]
    # Property setter: replaces the loss weight with the given value. Returns nothing.
    self._loss_weight = value

  @property
  def training_target(self):
    # [FUNCTION DOC]
    # Property getter: the training-target container of this endpoint, or nothing if not yet
    # created.
    return self._training_target

  @training_target.setter
  def training_target(self, value):
    # [FUNCTION DOC]
    # Property setter: replaces the training-target container. Returns nothing.
    self._training_target = value

  def create_training_target(self, target, run_eagerly=False):
    # [FUNCTION DOC]
    # Creates and stores the training-target container for this endpoint. Inputs: target (user
    # supplied target tensor or nothing), run_eagerly (default false). Raises a value error if a
    # target was already created. When running eagerly the target is ignored and a feedable
    # empty target is stored. If the output has no loss an empty non-feedable target is stored.
    # Otherwise a real tensor given by the user (not a placeholder) is non-feedable and its
    # weights are skipped; a placeholder or missing target is feedable, and when missing a
    # placeholder is created named after the output, with the rank of the output, sparsity of the
    # output and a dtype taken from the loss's label dtype table (default: output dtype).
    # Returns nothing.
    """Create training_target instance and update the self.training_target.

    Note that the input target should just be a tensor or None, and
    corresponding training target will be created based on the output and
    loss_fn.

    Args:
      target: the target tensor for the current output. Could be None.
      run_eagerly: boolean, whether the model is in run_eagerly mode.

    Raises:
      ValueError if the training_target field for the current instance has
      already been populated.
    """
    if self.has_training_target():
      raise ValueError('The training_target field for the _TrainingEndpoint '
                       'instance has already been populated')
    if run_eagerly:
      # When run_eagerly, the target tensor is ignored, and the None placeholder
      # is created instead.
      self.training_target = _TrainingTarget(
          None, feedable=True, skip_target_weights=False)
      return

    if self.should_skip_target():
      self.training_target = _TrainingTarget(None)
    else:
      if target is not None and not backend.is_placeholder(target):
        feedable = False
        skip_target_weights = True
      else:
        feedable = True
        skip_target_weights = False

      if target is None:
        target_dtype = losses.LABEL_DTYPES_FOR_LOSSES.get(
            self.loss_fn, backend.dtype(self.output))

        target = backend.placeholder(
            ndim=len(self.shape),
            name=self.output_name + '_target',
            sparse=backend.is_sparse(self.output),
            dtype=target_dtype)

      self.training_target = _TrainingTarget(
          target,
          feedable=feedable,
          skip_target_weights=skip_target_weights)

  @property
  def output_loss_metric(self):
    # [FUNCTION DOC]
    # Property getter: the metric object tracking this output's loss, or nothing.
    return self._output_loss_metric

  @output_loss_metric.setter
  def output_loss_metric(self, value):
    # [FUNCTION DOC]
    # Property setter: replaces the output loss metric object. Returns nothing.
    self._output_loss_metric = value

  @property
  def sample_weight(self):
    # [FUNCTION DOC]
    # Property getter: the sample weight tensor for this output, or nothing.
    return self._sample_weight

  @sample_weight.setter
  def sample_weight(self, value):
    # [FUNCTION DOC]
    # Property setter: replaces the sample weight tensor. Returns nothing.
    self._sample_weight = value

  @property
  def sample_weight_mode(self):
    # [FUNCTION DOC]
    # Property getter: the sample weight mode (temporal, sample-wise or nothing).
    return self._sample_weight_mode

  @sample_weight_mode.setter
  def sample_weight_mode(self, value):
    # [FUNCTION DOC]
    # Property setter: replaces the sample weight mode. Returns nothing.
    self._sample_weight_mode = value

  def should_skip_target(self):
    # [FUNCTION DOC]
    # Returns true when this output has no loss function, meaning its target and loss are
    # skipped entirely.
    return self._loss_fn is None

  def should_skip_target_weights(self):
    # [FUNCTION DOC]
    # Returns true when sample weights are not needed for this output: the target is skipped,
    # there is no training target, or the training target says to skip weights.
    return (self.should_skip_target() or self.training_target is None or
            self.training_target.skip_target_weights)

  def has_training_target(self):
    # [FUNCTION DOC]
    # Returns true when a training-target container has been created for this endpoint.
    return self.training_target is not None

  def has_feedable_training_target(self):
    # [FUNCTION DOC]
    # Returns true when the output has a loss, has a training target, and that target must be fed
    # with data by the user (not supplied as a fixed tensor).
    return (not self.should_skip_target() and
            self.training_target is not None and self.training_target.feedable)

  def loss_name(self):
    # [FUNCTION DOC]
    # Returns the display name of this output's loss (the output name with a loss suffix), or
    # nothing when the output has no loss function.
    if self._loss_fn is not None:
      return self._output_name + '_loss'
    return None

  @property
  def feed_output_shape(self):
    # [FUNCTION DOC]
    # Property giving the shape the fed target is expected to have, used for validation.
    # Returns nothing if the target is not feedable. For sparse categorical cross-entropy losses
    # the class axis is collapsed to one (position depends on the image data format). For
    # user-defined losses (not a loss object, or a wrapped function missing from the losses
    # module) returns nothing, making no assumption. Otherwise returns the output shape.
    """The output shape for the feedable target."""
    if not self.has_feedable_training_target():
      return None

    if ((isinstance(self.loss_fn, losses.LossFunctionWrapper) and
         self.loss_fn.fn == losses.sparse_categorical_crossentropy)) or (
             isinstance(self.loss_fn, losses.SparseCategoricalCrossentropy)):
      if backend.image_data_format() == 'channels_first':
        return (self.shape[0], 1) + self.shape[2:]
      else:
        return self.shape[:-1] + (1,)
    elif (not isinstance(self.loss_fn, losses.Loss) or
          (isinstance(self.loss_fn, losses.LossFunctionWrapper) and
           (getattr(losses, self.loss_fn.fn.__name__, None) is None))):
      # If the given loss is not an instance of the `Loss` class (custom
      # class) or if the loss function that is wrapped is not in the
      # `losses` module, then it is a user-defined loss and we make no
      # assumptions about it.
      return None
    else:
      return self.shape

  def sample_weights_mismatch(self):
    # [FUNCTION DOC]
    # Returns true when the sample weight mode and the presence of a sample weight tensor
    # disagree (a mode without a tensor, or a tensor without a mode), signalling that the
    # dependent graph parts must be rebuilt.
    """Check if the sample weight and the mode match or not."""
    # If there is a mismatch between sample weight mode and the placeholders
    # created, then recompile the sub-graphs that depend on sample weights.
    return (
        (self.sample_weight_mode is not None and self.sample_weight is None) or
        (self.sample_weight_mode is None and self.sample_weight is not None))

  def populate_sample_weight(self, sample_weight, sample_weight_mode):
    # [FUNCTION DOC]
    # Sets this endpoint's sample weight. Inputs: sample_weight (tensor or nothing) and
    # sample_weight_mode (temporal, sample-wise or nothing). With no tensor and either weights
    # skipped, no mode, or eager execution, the stored weight becomes nothing. Otherwise the mode
    # must be temporal or sample-wise (an assertion failure otherwise). A given tensor must be
    # shape compatible with two unknown dimensions (temporal) or one (sample-wise), else a value
    # error is raised, and is stored. Without a tensor, a placeholder with default value one is
    # created and named after the output. Returns nothing.
    """Populate the sample weight and based on the sample weight mode."""
    if (sample_weight is None and
        (self.should_skip_target_weights() or sample_weight_mode is None or
         context.executing_eagerly())):
      self._sample_weight = None
      return

    assert sample_weight_mode in ['temporal', 'samplewise']
    if sample_weight_mode == 'temporal':
      default_value = [[1.]]
      shape = [None, None]
    else:
      # sample_weight_mode == 'samplewise'
      default_value = [1.]
      shape = [None]

    if sample_weight is not None:
      if not sample_weight.shape.is_compatible_with(shape):
        raise ValueError('Received sample weight with shape {}. Expected shape '
                         '{}.'.format(sample_weight.shape, shape))
      self._sample_weight = sample_weight
    else:
      self._sample_weight = array_ops.placeholder_with_default(
          constant_op.constant(default_value, dtype=backend.floatx()),
          shape=shape,
          name=self.output_name + '_sample_weights')


class _TrainingTarget(object):
  # [CLASS DOC]
  # HELPER: Internal immutable record used by the training endpoint class.
  # Holds a target tensor (or nothing when the output has no loss), whether the target is feedable
  # (data must be supplied during training) and whether sample weights are skipped for it.
  # Created by the endpoint when compile builds targets; exposes read-only properties.
  """Container for a target tensor (y_true) and its metadata (shape, loss...).

  Args:
    target: A target tensor for the model. It may be `None` if the
      output is excluded from loss computation. It is still kept as None
      since each output of the model should have a corresponding target. If
      the target is None, the rest of the attributes will be None as well.
    feedable: Boolean, whether the target is feedable (requires data to be
      passed in `fit` or `train_on_batch`), or not (model compiled with
      `target_tensors` argument).
    skip_target_weights: Boolean, whether the target should be skipped during
      weights calculation.
  """

  def __init__(self, target, feedable=False, skip_target_weights=True):
    # [FUNCTION DOC]
    # Constructor. Inputs: target (tensor or nothing), feedable (default false), and
    # skip_target_weights (default true). Stores them in private attributes. Returns nothing.
    self._target = target
    self._feedable = feedable
    self._skip_target_weights = skip_target_weights

  @property
  def target(self):
    # [FUNCTION DOC]
    # Read-only property: the target tensor, or nothing if the output has no loss.
    return self._target

  @property
  def feedable(self):
    # [FUNCTION DOC]
    # Read-only property: true when data for this target must be supplied during training.
    return self._feedable

  @property
  def skip_target_weights(self):
    # [FUNCTION DOC]
    # Read-only property: true when sample weights should not be computed for this target.
    return self._skip_target_weights


def _is_symbolic_tensor(x):
  # [FUNCTION DOC]
  # HELPER: Small predicate used when standardizing data and compiling on the fly.
  # Input: x, any value. Returns true if it is a TensorFlow tensor type (symbolic or eager
  # tensor), false otherwise.
  return tensor_util.is_tf_type(x)


def _convert_scipy_sparse_tensor(value, expected_input):
  # [FUNCTION DOC]
  # HELPER: Input conversion used when standardizing tensors.
  # KNOWN ISSUE: the error message advises calling a densify method but its quoting is broken (a
  # backtick is left unclosed in the message text); behavior is otherwise as documented.
  # Inputs: value (any input value, possibly a scipy sparse matrix) and expected_input (the
  # model input placeholder it will feed). If scipy is available and the value is a scipy sparse
  # matrix: converts it to a sparse tensor when the expected input is sparse; otherwise raises a
  # value error in TF2-style (eager outside functions) mode, or returns a dense array in graph
  # mode. Any other value is returned unchanged.
  """Handle scipy sparse tensor conversions.

  This method takes a value 'value' and returns the proper conversion. If
  value is a scipy sparse tensor and the expected input is a dense tensor,
  we densify 'value'. If value is a scipy sparse tensor and the expected input
  is a TF SparseTensor, we convert 'value' to a SparseTensor. If 'value' is
  not a scipy sparse tensor, or scipy is not imported, we pass it through
  unchanged.

  Args:
    value: An object that may be a scipy sparse tensor
    expected_input: The expected input placeholder.

  Returns:
    The possibly-converted 'value'.
  """
  if issparse is not None and issparse(value):
    if backend.is_sparse(expected_input):
      sparse_coo = value.tocoo()
      row, col = sparse_coo.row, sparse_coo.col
      data, shape = sparse_coo.data, sparse_coo.shape
      indices = np.concatenate((np.expand_dims(row, 1), np.expand_dims(col, 1)),
                               1)
      return sparse_tensor.SparseTensor(indices, data, shape)
    else:
      if ops.executing_eagerly_outside_functions():
        # In TF2 we do not silently densify sparse matrices.
        raise ValueError('A SciPy sparse matrix was passed to a model '
                         'that expects dense inputs. Please densify your '
                         'inputs first, such as by calling `x.toarray().')
      return value.toarray()
  else:
    return value


def _get_metrics_from_layers(layers):
  # [FUNCTION DOC]
  # HELPER: Used by the model metrics property to gather metrics from sublayers.
  # Input: layers, a list of layers. Filters out empty layer containers, then for each layer that
  # is itself a v1 Model collects its added (non-compile) metrics and recurses into its layers;
  # for other layers collects their metrics. Returns a flat list of metric objects.
  """Returns list of metrics from the given layers.

  This will not include the `compile` metrics of a model layer.

  Args:
    layers: List of layers.

  Returns:
    List of metrics.
  """
  metrics = []
  layers = layer_utils.filter_empty_layer_containers(layers)
  for layer in layers:
    if isinstance(layer, Model):
      # We cannot call 'metrics' on the model because we do not want to
      # include the metrics that were added in compile API of a nested model.
      metrics.extend(layer._metrics)  # pylint: disable=protected-access
      metrics.extend(_get_metrics_from_layers(layer.layers))
    else:
      metrics.extend(layer.metrics)
  return metrics


def _non_none_constant_value(v):
  # [FUNCTION DOC]
  # HELPER: Used by the on-batch methods to turn eager step outputs into plain values.
  # Input: v, a tensor or value. Returns its statically known constant value if one can be
  # computed, otherwise returns the input unchanged.
  constant_value = tensor_util.constant_value(v)
  return constant_value if constant_value is not None else v
