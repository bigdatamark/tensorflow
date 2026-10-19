# [MODULE DOC]
# HELPER: shared utilities for Keras model saving, used by the HDF5 and SavedModel code paths.
# Provides: extraction of compile metrics, computation of a model's input signature, tracing a
# model call into a function for export, assembly of model metadata (version, backend, model and
# training configuration) for saving, overwrite confirmation, rebuilding compile arguments from a
# saved training configuration (with nested serialization helpers), building compiled loss and
# metrics, and an HDF5 file name check. No import-time side effects.
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
"""Utils related to keras model saving."""

import collections
import copy
import os

from tensorflow.python.eager import def_function
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import losses
from tensorflow.python.keras import optimizer_v1
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.io_utils import ask_to_proceed_with_overwrite
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest


def extract_model_metrics(model):
  # [FUNCTION DOC]
  # HELPER: gathers compile-time metrics for converting a model to a SavedModel.
  # KNOWN ISSUE: the test looks at one private attribute (the compiled metrics container) but the
  # returned mapping is built from a different private attribute (the compile metric functions),
  # so the two can disagree; metrics added via the add-metric API are deliberately excluded.
  # Input: a Keras model. Returns a mapping from metric names to metric objects, or nothing if the
  # model has no compiled metrics.
  """Convert metrics from a Keras model `compile` API to dictionary.

  This is used for converting Keras models to SavedModels.

  Args:
    model: A `tf.keras.Model` object.

  Returns:
    Dictionary mapping metric names to metric instances. May return `None` if
    the model does not contain any metrics.
  """
  if getattr(model, '_compile_metrics', None):
    # We are not using model.metrics here because we want to exclude the metrics
    # added using `add_metric` API.
    return {m.name: m for m in model._compile_metric_functions}  # pylint: disable=protected-access
  return None


def model_input_signature(model, keep_original_batch_size=False):
  # [FUNCTION DOC]
  # Computes the input signature of a model for saving. Inputs: model and keep_original_batch_size
  # (default false: the batch dimension is made unknown). Reads the model's save specification,
  # makes names consistent (clearing all names if only some specs have names), and returns a list
  # with a single element: the specification itself if it is already a one-element sequence,
  # otherwise the specification wrapped in a one-element list. Returns nothing (null) if the model
  # has no save specification. Does not include the training argument.
  """Inspect model to get its input signature.

  The model's input signature is a list with a single (possibly-nested) object.
  This is due to the Keras-enforced restriction that tensor inputs must be
  passed in as the first argument.

  For example, a model with input {'feature1': <Tensor>, 'feature2': <Tensor>}
  will have input signature: [{'feature1': TensorSpec, 'feature2': TensorSpec}]

  Args:
    model: Keras Model object.
    keep_original_batch_size: A boolean indicating whether we want to keep using
      the original batch size or set it to None. Default is `False`, which means
      that the batch dim of the returned input signature will always be set to
      `None`.

  Returns:
    A list containing either a single TensorSpec or an object with nested
    TensorSpecs. This list does not contain the `training` argument.
  """
  input_specs = model._get_save_spec(dynamic_batch=not keep_original_batch_size)  # pylint: disable=protected-access
  if input_specs is None:
    return None
  input_specs = _enforce_names_consistency(input_specs)
  # Return a list with a single element as the model's input signature.
  if isinstance(input_specs,
                collections.abc.Sequence) and len(input_specs) == 1:
    # Note that the isinstance check filters out single-element dictionaries,
    # which should also be wrapped as a single-element list.
    return input_specs
  else:
    return [input_specs]


def raise_model_input_error(model):
  # [FUNCTION DOC]
  # HELPER: error helper for saving. Always raises a value error explaining that the model cannot
  # be saved because its input shapes are not set, and how to set them (run fit or predict, or build
  # with an input shape). Input: the model, used in the message. Never returns.
  raise ValueError(
      'Model {} cannot be saved because the input shapes have not been '
      'set. Usually, input shapes are automatically determined from calling'
      ' `.fit()` or `.predict()`. To manually set the shapes, call '
      '`model.build(input_shape)`.'.format(model))


def trace_model_call(model, input_signature=None):
  # [FUNCTION DOC]
  # Wraps a model's call in a traced function with a fixed input signature, for exporting.
  # Inputs: model and optional input_signature (list of tensor specs). If not given, uses the
  # signature of the model's call when that is already a function, else infers it from the model.
  # Raises a value error (via the input-error helper) if no signature can be determined.
  # Returns a function that runs the model in inference mode within a saving call context and
  # returns a flat mapping from output names to output tensors.
  """Trace the model call to create a tf.function for exporting a Keras model.

  Args:
    model: A Keras model.
    input_signature: optional, a list of tf.TensorSpec objects specifying the
      inputs to the model.

  Returns:
    A tf.function wrapping the model's call function with input signatures set.

  Raises:
    ValueError: if input signature cannot be inferred from the model.
  """
  if input_signature is None:
    if isinstance(model.call, def_function.Function):
      input_signature = model.call.input_signature

  if input_signature is None:
    input_signature = model_input_signature(model)

  if input_signature is None:
    raise_model_input_error(model)

  @def_function.function(input_signature=input_signature)
  def _wrapped_model(*args):
    # [FUNCTION DOC]
    # HELPER: the traced function returned by the model-call tracing helper.
    # Inputs: the positional input tensors (one tensor is passed through as-is, several are
    # collected into a list). Calls the model with training off inside a saving call context, then
    # flattens the outputs; output names come from the model, or are generated pseudo names for
    # subclassed models. Returns a mapping from output names to output tensors.
    """A concrete tf.function that wraps the model's call function."""
    # When given a single input, Keras models will call the model on the tensor
    # rather than a list consisting of the single tensor.
    inputs = args[0] if len(input_signature) == 1 else list(args)

    with base_layer_utils.call_context().enter(
        model, inputs=inputs, build_graph=False, training=False, saving=True):
      outputs = model(inputs, training=False)

    # Outputs always has to be a flat dict.
    output_names = model.output_names  # Functional Model.
    if output_names is None:  # Subclassed Model.
      from tensorflow.python.keras.engine import compile_utils  # pylint: disable=g-import-not-at-top
      output_names = compile_utils.create_pseudo_output_names(outputs)
    outputs = nest.flatten(outputs)
    return {name: output for name, output in zip(output_names, outputs)}

  return _wrapped_model


def model_metadata(model, include_optimizer=True, require_config=True):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the training configuration is stored in the metadata before the check that
  # rejects optimizers restored from a SavedModel, so the error occurs after partial work; and
  # when require_config is false and the model has no configuration, the config entry is silently
  # left out of the model configuration.
  # Builds the metadata mapping saved with a model. Inputs: model, include_optimizer (default
  # true) and require_config (default true: re-raise the not-implemented error if the model cannot
  # produce a configuration). Returns a mapping with the Keras version, backend name and model
  # configuration (class name and config); when an optimizer is present and requested and the model
  # was compiled, also a training configuration with the serialized compile arguments and the
  # optimizer class name and configuration. A wrapped TF optimizer only logs a warning. Raises a
  # not-implemented error for an optimizer restored from a SavedModel.
  """Returns a dictionary containing the model metadata."""
  from tensorflow.python.keras import __version__ as keras_version  # pylint: disable=g-import-not-at-top
  from tensorflow.python.keras.optimizer_v2 import optimizer_v2  # pylint: disable=g-import-not-at-top

  model_config = {'class_name': model.__class__.__name__}
  try:
    model_config['config'] = model.get_config()
  except NotImplementedError as e:
    if require_config:
      raise e

  metadata = dict(
      keras_version=str(keras_version),
      backend=K.backend(),
      model_config=model_config)
  if model.optimizer and include_optimizer:
    if isinstance(model.optimizer, optimizer_v1.TFOptimizer):
      logging.warning(
          'TensorFlow optimizers do not '
          'make it possible to access '
          'optimizer attributes or optimizer state '
          'after instantiation. '
          'As a result, we cannot save the optimizer '
          'as part of the model save file. '
          'You will have to compile your model again after loading it. '
          'Prefer using a Keras optimizer instead '
          '(see keras.io/optimizers).')
    elif model._compile_was_called:  # pylint: disable=protected-access
      training_config = model._get_compile_args(user_metrics=False)  # pylint: disable=protected-access
      training_config.pop('optimizer', None)  # Handled separately.
      metadata['training_config'] = _serialize_nested_config(training_config)
      if isinstance(model.optimizer, optimizer_v2.RestoredOptimizer):
        raise NotImplementedError(
            'As of now, Optimizers loaded from SavedModel cannot be saved. '
            'If you\'re calling `model.save` or `tf.keras.models.save_model`,'
            ' please set the `include_optimizer` option to `False`. For '
            '`tf.saved_model.save`, delete the optimizer from the model.')
      else:
        optimizer_config = {
            'class_name':
                generic_utils.get_registered_name(model.optimizer.__class__),
            'config':
                model.optimizer.get_config()
        }
      metadata['training_config']['optimizer_config'] = optimizer_config
  return metadata


def should_overwrite(filepath, overwrite):
  # [FUNCTION DOC]
  # Decides whether saving may replace the target. Inputs: filepath and overwrite flag. If
  # overwrite is false and a file exists at the path, asks the user interactively and returns the
  # answer; otherwise returns true.
  """Returns whether the filepath should be overwritten."""
  # If file exists and should not be overwritten.
  if not overwrite and os.path.isfile(filepath):
    return ask_to_proceed_with_overwrite(filepath)
  return True


def compile_args_from_training_config(training_config, custom_objects=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the sample weight mode is looked up with an attribute test on a mapping, which is
  # never true for a dictionary, so the sample weight mode is always nothing regardless of what was
  # saved.
  # Rebuilds the arguments for compiling a model from a saved training configuration. Inputs:
  # training_config (mapping with optimizer configuration, optionally loss, metrics, weighted
  # metrics, and loss weights) and custom_objects (optional mapping, default empty, made available
  # during deserialization). Deserializes the optimizer, losses, metrics and weighted metrics
  # (nested structures supported). Raises a key error if the optimizer configuration or loss weights
  # key is missing. Returns a mapping with optimizer, loss, metrics, weighted metrics, loss weights
  # and sample weight mode.
  """Return model.compile arguments from training config."""
  if custom_objects is None:
    custom_objects = {}

  with generic_utils.CustomObjectScope(custom_objects):
    optimizer_config = training_config['optimizer_config']
    optimizer = optimizers.deserialize(optimizer_config)

    # Recover losses.
    loss = None
    loss_config = training_config.get('loss', None)
    if loss_config is not None:
      loss = _deserialize_nested_config(losses.deserialize, loss_config)

    # Recover metrics.
    metrics = None
    metrics_config = training_config.get('metrics', None)
    if metrics_config is not None:
      metrics = _deserialize_nested_config(_deserialize_metric, metrics_config)

    # Recover weighted metrics.
    weighted_metrics = None
    weighted_metrics_config = training_config.get('weighted_metrics', None)
    if weighted_metrics_config is not None:
      weighted_metrics = _deserialize_nested_config(_deserialize_metric,
                                                    weighted_metrics_config)

    sample_weight_mode = training_config['sample_weight_mode'] if hasattr(
        training_config, 'sample_weight_mode') else None
    loss_weights = training_config['loss_weights']

  return dict(
      optimizer=optimizer,
      loss=loss,
      metrics=metrics,
      weighted_metrics=weighted_metrics,
      loss_weights=loss_weights,
      sample_weight_mode=sample_weight_mode)


def _deserialize_nested_config(deserialize_fn, config):
  # [FUNCTION DOC]
  # HELPER: recursive deserializer used when restoring compile arguments.
  # Inputs: a deserialization function and a configuration (nothing, a serialized object mapping
  # with a class name, a string, a mapping of such, or a list/tuple of such). Returns nothing for
  # nothing, applies the function to single objects, rebuilds mappings value by value, and returns
  # a list for sequences. Raises a value error for anything else.
  """Deserializes arbitrary Keras `config` using `deserialize_fn`."""

  def _is_single_object(obj):
    # [FUNCTION DOC]
    # HELPER: classification step of the nested deserializer. Returns true for a serialized object
    # mapping (has a class name) or a string (serialized function or name), false otherwise.
    if isinstance(obj, dict) and 'class_name' in obj:
      return True  # Serialized Keras object.
    if isinstance(obj, str):
      return True  # Serialized function or string.
    return False

  if config is None:
    return None
  if _is_single_object(config):
    return deserialize_fn(config)
  elif isinstance(config, dict):
    return {
        k: _deserialize_nested_config(deserialize_fn, v)
        for k, v in config.items()
    }
  elif isinstance(config, (tuple, list)):
    return [_deserialize_nested_config(deserialize_fn, obj) for obj in config]

  raise ValueError('Saved configuration not understood.')


def _serialize_nested_config(config):
  # [FUNCTION DOC]
  # HELPER: recursive serializer used when building the training configuration.
  # Input: a nested structure. Returns a structure of the same shape in which every callable
  # element is replaced by its serialized Keras form and other elements are kept as they are.
  """Serialized a nested structure of Keras objects."""

  def _serialize_fn(obj):
    # [FUNCTION DOC]
    # HELPER: per-element step of the nested serializer. Returns the serialized Keras form of a
    # callable, or the element unchanged if it is not callable.
    if callable(obj):
      return generic_utils.serialize_keras_object(obj)
    return obj

  return nest.map_structure(_serialize_fn, config)


def _deserialize_metric(metric_config):
  # [FUNCTION DOC]
  # HELPER: metric deserializer used when restoring compile arguments.
  # Input: a metric configuration. The special strings for accuracy and cross-entropy (and their
  # short forms) are returned unchanged because compile resolves them from the output shape;
  # anything else is deserialized into a metric object (imported lazily).
  """Deserialize metrics, leaving special strings untouched."""
  from tensorflow.python.keras import metrics as metrics_module  # pylint:disable=g-import-not-at-top
  if metric_config in ['accuracy', 'acc', 'crossentropy', 'ce']:
    # Do not deserialize accuracy and cross-entropy strings as we have special
    # case handling for these in compile, based on model output shape.
    return metric_config
  return metrics_module.deserialize(metric_config)


def _enforce_names_consistency(specs):
  # [FUNCTION DOC]
  # HELPER: normalizes input specifications for model_input_signature.
  # Input: a nested structure of tensor specs. If some specs have names and others do not, returns
  # a copy in which all names are cleared; otherwise returns the input unchanged.
  """Enforces that either all specs have names or none do."""

  def _has_name(spec):
    # [FUNCTION DOC]
    # HELPER: returns true if the spec has a name attribute that is not nothing.
    return hasattr(spec, 'name') and spec.name is not None

  def _clear_name(spec):
    # [FUNCTION DOC]
    # HELPER: returns a deep copy of the spec with its private name cleared (when it has one).
    spec = copy.deepcopy(spec)
    if hasattr(spec, 'name'):
      spec._name = None  # pylint:disable=protected-access
    return spec

  flat_specs = nest.flatten(specs)
  name_inconsistency = (
      any(_has_name(s) for s in flat_specs) and
      not all(_has_name(s) for s in flat_specs))

  if name_inconsistency:
    specs = nest.map_structure(_clear_name, specs)
  return specs


def try_build_compiled_arguments(model):
  # [FUNCTION DOC]
  # KNOWN ISSUE: it catches every exception (bare handler) and the warning text claims a loaded
  # model was compiled, even when called from the save path.
  # Builds the compiled loss and compiled metrics of a model if they are not yet built. Input: a
  # model. Does nothing for legacy v1 models or models without outputs. Any failure while building
  # is swallowed and only logs a warning. Side effect: builds the model's loss and metric
  # containers. Returns nothing.
  if (not version_utils.is_v1_layer_or_model(model) and
      model.outputs is not None):
    try:
      if not model.compiled_loss.built:
        model.compiled_loss.build(model.outputs)
      if not model.compiled_metrics.built:
        model.compiled_metrics.build(model.outputs, model.outputs)
    except:  # pylint: disable=bare-except
      logging.warning(
          'Compiled the loaded model, but the compiled metrics have yet to '
          'be built. `model.compile_metrics` will be empty until you train '
          'or evaluate the model.')


def is_hdf5_filepath(filepath):
  # [FUNCTION DOC]
  # HELPER: file-name check used to pick the HDF5 format. Input: a path string. Returns true if it
  # ends with the extension for HDF5 (.h5 or .hdf5) or the Keras (.keras) extension; the check is
  # case sensitive and looks only at the name, not the file contents.
  return (filepath.endswith('.h5') or filepath.endswith('.keras') or
          filepath.endswith('.hdf5'))
