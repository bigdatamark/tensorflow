# [MODULE DOC]
# Implementation of Keras SavedModel serialization for layers, models and metrics. It builds the
# extra trackable objects (variable lists, wrapped regularization losses, metrics) and the traced
# tf.function wrappers (layer call, call returning conditional losses, call returning all losses,
# activity regularizer) that get attached to a serialized layer. To do this it temporarily swaps
# the call functions and loss lists of a layer and its children, traces everything with a shared
# input signature inside a tracing scope, then restores the original attributes.
# Exposes: should_skip_serialization, wrap_layer_objects, wrap_layer_functions,
# default_save_signature, tracing_scope and the LayerCallCollection / LayerCall helper classes.
# Import-time side effects: creates lazy loaders for engine modules (to avoid circular imports) and
# a thread-local tracing context holding the tracing flag and trace queue.
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
"""Keras SavedModel serialization.

TODO (kathywu): Move to layer_serialization.py. Some model-specific logic should
go to model_serialization.py.
"""

import functools
import threading
import weakref

from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.mixed_precision import autocast_variable
from tensorflow.python.keras.saving import saving_utils
from tensorflow.python.keras.saving.saved_model import constants
from tensorflow.python.keras.saving.saved_model import load as keras_load
from tensorflow.python.keras.saving.saved_model import serialized_attributes
from tensorflow.python.keras.saving.saved_model import utils
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import data_structures
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator


# To avoid circular dependencies between keras/engine and keras/saving,
# code in keras/saving must delay imports.

# TODO(b/134426265): Switch back to single-quotes to match the rest of the file
# once the issue with copybara is fixed.
# pylint:disable=g-inconsistent-quotes
base_layer = LazyLoader(
    "base_layer", globals(),
    "tensorflow.python.keras.engine.base_layer")
metrics = LazyLoader("metrics", globals(),
                     "tensorflow.python.keras.metrics")
input_layer = LazyLoader(
    "input_layer", globals(),
    "tensorflow.python.keras.engine.input_layer")
training_lib = LazyLoader(
    "training_lib", globals(),
    "tensorflow.python.keras.engine.training")
sequential_lib = LazyLoader(
    "sequential_lib", globals(),
    "tensorflow.python.keras.engine.sequential")
# pylint:enable=g-inconsistent-quotes


def should_skip_serialization(layer):
  # [FUNCTION DOC]
  # Decides whether full serialization (extra objects and traced functions) must be skipped.
  # Input: layer, a Keras layer or model. A model counts as having inputs set when it is a Model
  # with a recorded saved-model input spec. Returns true when the layer is not built and has no
  # such spec; side effect: logs a warning naming the layer in that case. Otherwise returns false.
  """Skip serializing extra objects and functions if layer inputs aren't set."""
  saved_model_input_spec_set = (isinstance(layer, training_lib.Model) and
                                layer._saved_model_inputs_spec is not None)  # pylint: disable=protected-access
  if not layer.built and not saved_model_input_spec_set:
    logging.warning('Skipping full serialization of Keras layer {}, because '
                    'it is not built.'.format(layer))
    return True
  return False


def wrap_layer_objects(layer, serialization_cache):
  # [FUNCTION DOC]
  # Collects the extra trackable objects to attach to a serialized layer. Inputs: layer, a Keras
  # layer; serialization_cache, a mapping shared across the whole serialization pass (a
  # 'keras_losses' entry is created in it and used to cache wrapped loss functions so a loss shared
  # by several layers is wrapped once). Gathers callable losses of the layer and all its
  # sublayers, wraps each as a function, and returns a mapping with variables, trainable and
  # non-trainable variables, nested layers, metrics, all regularization losses, this layer's own
  # regularization losses and a mapping from metric name to metric, all wrapped as trackable data
  # structures. Side effect: mutates the cache.
  """Returns extra trackable objects to attach to the serialized layer.

  Args:
    layer: Keras Layer object.
    serialization_cache: Dictionary shared between all objects during
      serialization.

  Returns:
    A dictionary containing all checkpointable objects from a
    SerializedAttributes object. See LayerAttributes and ModelAttributes for
    entire list of objects
  """
  # Wrap all regularization losses as tf.functions.
  # First, generate list of all regularization losses in this layer and
  # sublayers.
  all_losses = layer._callable_losses[:]  # pylint: disable=protected-access
  for child_layer in utils.list_all_layers(layer):
    all_losses.extend(child_layer._callable_losses)  # pylint: disable=protected-access
  # Next, wrap all loss functions as tf.functions. Use the serialization cache
  # to store already-wrapped functions.
  keras_loss_cache = serialization_cache.setdefault('keras_losses', {})
  wrapped_loss_functions = []
  for loss_fn in all_losses:
    if loss_fn in keras_loss_cache:
      wrapped_loss_functions.append(keras_loss_cache[loss_fn])
    else:
      wrapped_loss = _wrap_unconditional_loss(loss_fn, len(keras_loss_cache))
      keras_loss_cache[loss_fn] = wrapped_loss
      wrapped_loss_functions.append(wrapped_loss)
  wrapped_layer_losses = [keras_loss_cache[fn]
                          for fn in layer._callable_losses[:]]  # pylint: disable=protected-access

  layer_metrics = data_structures.wrap_or_unwrap(
      {m.name: m for m in layer._metrics})  # pylint: disable=protected-access
  return dict(
      variables=data_structures.wrap_or_unwrap(layer.variables),
      trainable_variables=data_structures.wrap_or_unwrap(
          layer.trainable_variables),
      non_trainable_variables=data_structures.wrap_or_unwrap(
          layer.non_trainable_variables),
      layers=data_structures.wrap_or_unwrap(utils.list_all_layers(layer)),
      metrics=data_structures.wrap_or_unwrap(layer.metrics),
      regularization_losses=data_structures.wrap_or_unwrap(
          wrapped_loss_functions),
      layer_regularization_losses=data_structures.wrap_or_unwrap(
          wrapped_layer_losses),
      layer_metrics=layer_metrics)
  # pylint: disable=protected-access


def wrap_layer_functions(layer, serialization_cache):
  # [FUNCTION DOC]
  # Builds the traced functions to serialize for a layer. Inputs: layer, a Keras layer or model;
  # serialization_cache, the shared serialization mapping. If the layer is a revived (loaded)
  # layer that is not a Sequential model, simply returns the already-saved functions from its
  # keras api object (missing ones are null). Otherwise it temporarily replaces child layer call
  # functions and resets losses, creates a call collection, and wraps the call-with-conditional-
  # losses function and the plain call function (and, when an activity regularizer exists, the
  # regularizer function and the call-returning-all-losses function; without one the latter is the
  # conditional-losses function and the regularizer entry is null). Then, inside a tracing scope
  # and a saving call context, it traces every function that has an input signature to obtain a
  # concrete function, and finally restores the original child functions and losses.
  # Returns a mapping from function names to the wrapped functions.
  """Returns dict of wrapped layer call function and losses in tf.functions.

  Args:
    layer: Keras Layer object.
    serialization_cache: Dictionary shared between all objects during
      serialization.

  Returns:
    A dictionary containing all keras tf.functions to serialize. See
    LayerAttributes and ModelAttributes for the list of all attributes.
  """
  # Since Sequential models may be modified in place using model.add() or
  # model.pop(), don't use saved functions.
  if (isinstance(layer, keras_load.RevivedLayer) and
      not isinstance(layer, sequential_lib.Sequential)):
    return {fn_name: getattr(layer.keras_api, fn_name, None)
            for fn_name in serialized_attributes.LayerAttributes.all_functions}

  # Reset the losses of the layer and its children. The call function in each
  # child layer is replaced with tf.functions.
  original_fns = _replace_child_layer_functions(layer, serialization_cache)
  original_losses = _reset_layer_losses(layer)

  # Wrap all the layer call and activity regularizer functions.

  # Use LayerCallCollection to ensure that all layer call functions (__call__,
  # call with losses) are traced with the same inputs.
  call_collection = LayerCallCollection(layer)
  call_fn_with_losses = call_collection.add_function(
      _wrap_call_and_conditional_losses(layer),
      '{}_layer_call_and_return_conditional_losses'.format(layer.name),
      # If any of this layer's child layers use the training arg, the traced
      # call functions of this layer will have a training keyword argument. If
      # the original layer does not expect the training arg, then it will have
      # to be removed (by setting `match_layer_training_arg`).
      match_layer_training_arg=True)
  call_fn = call_collection.add_function(
      _extract_outputs_from_fn(layer, call_fn_with_losses),
      '{}_layer_call_fn'.format(layer.name),
      # Since `call_fn` wraps call_fn_with_losses and not the original call
      # function, `match_layer_training_arg` should be set to False.
      match_layer_training_arg=False)

  fns = {'call_and_return_conditional_losses': call_fn_with_losses,
         '__call__': call_fn}

  if layer._activity_regularizer is not None:  # pylint: disable=protected-access
    fns['activity_regularizer_fn'] = _wrap_activity_regularizer(layer)
    fns['call_and_return_all_conditional_losses'] = (
        call_collection.add_function(
            _append_activity_regularizer_loss(
                layer, call_fn_with_losses, fns['activity_regularizer_fn']),
            '{}_layer_call_and_return_all_conditional_losses'.format(
                layer.name),
            match_layer_training_arg=False))
  else:
    fns['activity_regularizer_fn'] = None
    fns['call_and_return_all_conditional_losses'] = call_fn_with_losses

  # Manually trigger traces before restoring the overwritten functions. The
  # functions are traced within the layer call context to ensure that layer
  # functions (e.g. add_loss) behave as though running in graph mode.
  with tracing_scope():
    call_collection.trace_with_input_signature()
    with base_layer_utils.call_context().enter(
        layer, inputs=None, build_graph=True, training=None, saving=True):
      for fn in fns.values():
        if fn is not None and fn.input_signature is not None:
          if isinstance(fn, LayerCall):
            fn = fn.wrapped_call
          fn.get_concrete_function()

  # Restore overwritten functions and losses
  _restore_child_layer_functions(original_fns)
  _restore_layer_losses(original_losses)

  return fns


def default_save_signature(layer):
  # [FUNCTION DOC]
  # Produces the default serving function for a layer or model. Input: layer. Temporarily clears
  # the losses of the layer and its sublayers, traces the model call through the saving utilities,
  # forces creation of a concrete function, restores the losses, and returns the traced function.
  original_losses = _reset_layer_losses(layer)
  fn = saving_utils.trace_model_call(layer)
  fn.get_concrete_function()
  _restore_layer_losses(original_losses)
  return fn


def _replace_child_layer_functions(layer, serialization_cache):
  # [FUNCTION DOC]
  # HELPER: step of wrap_layer_functions that swaps child layer functions for traced ones.
  # Inputs: layer, a Keras layer; serialization_cache, the shared serialization mapping. For every
  # child layer (input layers are skipped) it finds the serialized functions, either from the cache
  # or by asking the child's saver for its serialized attributes; children with no functions yet
  # (circular dependency or undefined inputs) are skipped. Metrics get their call, result and
  # update-state replaced, other layers get their call and activity regularizer replaced, with
  # automatic dependency tracking disabled. Returns a mapping from child layer to its original
  # attributes so they can be restored. Side effect: mutates the child layers.
  """Replaces functions in the children layers with wrapped tf.functions.

  This step allows functions from parent layers to reference the wrapped
  functions from their children layers instead of retracing the ops.

  This function also resets all losses stored in the layer. These are stored in
  the returned dictionary. Use `_restore_child_layer_functions` to restore
  the original attributes.

  Args:
    layer: Keras Layer object.
    serialization_cache: Dictionary shared between all objects during
      serialization.

  Returns:
    Dictionary mapping layer objects -> original functions and losses:
      { Child layer 1: {
          'losses': Original losses,
          'call': Original call function
          '_activity_regularizer': Original activity regularizer},
        Child layer 2: ...
      }
  """
  # pylint: disable=protected-access
  original_fns = {}

  def replace_layer_functions(child_layer, serialized_fns):
    # [FUNCTION DOC]
    # HELPER: nested helper of _replace_child_layer_functions for ordinary layers.
    # Records the child's current call and activity regularizer in the enclosing mapping, then sets
    # the regularizer to the serialized one (ignored if the layer forbids setting it) and the call
    # to a wrapper around the serialized call-with-conditional-losses function (training default
    # false). Returns nothing.
    """Replaces layer call and activity regularizer with wrapped functions."""
    original_fns[child_layer] = {
        'call': child_layer.call,
        '_activity_regularizer': child_layer._activity_regularizer
    }
    with utils.no_automatic_dependency_tracking_scope(child_layer):
      try:
        child_layer._activity_regularizer = serialized_fns.get(
            'activity_regularizer_fn')
      except AttributeError:
        # Some layers have an unsettable activity regularizer.
        pass
      child_layer.call = utils.use_wrapped_call(
          child_layer,
          serialized_fns['call_and_return_conditional_losses'],
          default_training_value=False)

  def replace_metric_functions(child_layer, serialized_fns):
    # [FUNCTION DOC]
    # HELPER: nested helper of _replace_child_layer_functions for metric objects.
    # Records the metric's current call, result and update-state methods in the enclosing mapping,
    # then replaces them with the serialized versions (looked up by name; a missing entry raises a
    # missing-key error). Returns nothing.
    """Replaces metric functions with wrapped functions."""
    original_fns[child_layer] = {
        '__call__': child_layer.__call__,
        'result': child_layer.result,
        'update_state': child_layer.update_state
    }
    with utils.no_automatic_dependency_tracking_scope(child_layer):
      child_layer.__call__ = serialized_fns['__call__']
      child_layer.result = serialized_fns['result']
      child_layer.update_state = serialized_fns['update_state']

  for child_layer in utils.list_all_layers(layer):
    if isinstance(child_layer, input_layer.InputLayer):
      continue

    if child_layer not in serialization_cache[constants.KERAS_CACHE_KEY]:
      serialized_functions = (
          child_layer._trackable_saved_model_saver._get_serialized_attributes(
              serialization_cache).functions)
    else:
      serialized_functions = (
          serialization_cache[constants.KERAS_CACHE_KEY][child_layer].functions)
    if not serialized_functions:
      # This indicates either:
      #   - circular dependency, which means the current layer's functions
      #     should be wrapped first.
      #   - Child layer's inputs are not defined, so its functions have not been
      #     wrapped. In this case, no replacement is necessary so move on to the
      #     next child.
      continue

    if isinstance(child_layer, metrics.Metric):
      replace_metric_functions(child_layer, serialized_functions)
    else:
      replace_layer_functions(child_layer, serialized_functions)

  return original_fns
  # pylint: enable=protected-access


def _restore_child_layer_functions(original_fns):
  # [FUNCTION DOC]
  # HELPER: undoes _replace_child_layer_functions. Input: mapping from child layer to a mapping of
  # attribute names to original values. Sets each attribute back with dependency tracking
  # disabled; attributes that cannot be set (such as the activity regularizer) are silently
  # ignored. Returns nothing.
  """Restores attributes replaced with `_replace_child_layer_functions`."""
  for child_layer, fns in original_fns.items():
    with utils.no_automatic_dependency_tracking_scope(child_layer):
      for fn_name, fn in fns.items():
        try:
          setattr(child_layer, fn_name, fn)  # pylint: disable=protected-access
        except AttributeError:
          pass  # In the case of _activity_regularizer, setting the attribute
          # may be disallowed.


# pylint: disable=protected-access
def _reset_layer_losses(parent_layer):
  # [FUNCTION DOC]
  # HELPER: saves and clears loss state during tracing. Input: parent_layer. For the layer and all
  # its sublayers, copies the loss list and eager loss list into a mapping keyed by layer, then
  # sets both lists to empty (tracking disabled). Returns that mapping of saved losses.
  """Resets losses of layer and its sublayers, and returns original losses."""
  losses_dict = {}
  for layer in utils.list_all_layers_and_sublayers(parent_layer):
    losses_dict[layer] = {'losses': layer._losses[:],
                          'eager_losses': layer._eager_losses[:]}
    with utils.no_automatic_dependency_tracking_scope(layer):
      layer._losses = []
      layer._eager_losses = []
  return losses_dict


def _restore_layer_losses(losses_dict):
  # [FUNCTION DOC]
  # HELPER: inverse of _reset_layer_losses. Input: mapping from layer to its saved loss and eager
  # loss lists. Reassigns those lists on each layer (tracking disabled). Returns nothing.
  for layer in losses_dict:
    with utils.no_automatic_dependency_tracking_scope(layer):
      layer._losses = losses_dict[layer]['losses']
      layer._eager_losses = losses_dict[layer]['eager_losses']
# pylint: enable=protected-access


class LayerTracingContext(threading.local):
  # [CLASS DOC]
  # HELPER: thread-local holder of tracing state for saving. Each thread sees its own flag saying
  # whether extra call traces should be queued, and its own queue of pending traces. One
  # module-level instance is used by the tracing scope and the call collection.

  def __init__(self):
    # [FUNCTION DOC]
    # Initializes per-thread state: call tracing disabled and an empty trace queue. Takes no
    # arguments and returns nothing; runs again in every thread that touches the instance.
    super(LayerTracingContext, self).__init__()
    self.enable_call_tracing = False
    self.trace_queue = []

_thread_local_data = LayerTracingContext()


@tf_contextlib.contextmanager
def tracing_scope():
  # [FUNCTION DOC]
  # Context manager that turns on queued tracing for the current thread. On entry it remembers the
  # previous flag and queue, enables tracing and starts a fresh queue. On exit (even after an
  # error) it drains the queue, calling get_concrete_function on each queued function with its
  # recorded arguments, inside a learning-phase scope when a training value was recorded, then
  # restores the previous queue and flag. Takes no arguments; yields nothing.
  """Enables tracing scope."""
  # This enables the LayerCallCollection's tracing mechanism to trace all call
  # functions in the collection.
  previous_value = _thread_local_data.enable_call_tracing
  previous_queue = _thread_local_data.trace_queue
  try:
    _thread_local_data.enable_call_tracing = True
    _thread_local_data.trace_queue = []
    yield
  finally:
    # Run traces from the queue.
    while _thread_local_data.trace_queue:
      fn, args, kwargs, training = _thread_local_data.trace_queue.pop()
      if training is not None:
        with K.deprecated_internal_learning_phase_scope(training):
          fn.get_concrete_function(*args, **kwargs)
      else:
        fn.get_concrete_function(*args, **kwargs)
    _thread_local_data.trace_queue = previous_queue
    _thread_local_data.enable_call_tracing = previous_value


def add_trace_to_queue(fn, args, kwargs, training=None):
  # [FUNCTION DOC]
  # Queues a deferred trace of a function. Inputs: fn, a traceable function; args, list of
  # positional arguments; kwargs, mapping of keyword arguments; training, optional training flag
  # (default none). Only when tracing is enabled, appends a tuple with copies of args and kwargs
  # to the thread-local queue; otherwise does nothing. Returns nothing.
  if tracing_enabled():
    _thread_local_data.trace_queue.append(
        (fn, args[:], kwargs.copy(), training))


def tracing_enabled():
  # [FUNCTION DOC]
  # Returns the current thread's flag saying whether extra traces should be queued (true only
  # inside a tracing scope). No arguments, no side effects.
  """Whether to add extra traces to the queue."""
  return _thread_local_data.enable_call_tracing


class LayerCallCollection(object):
  # [CLASS DOC]
  # Groups the wrapped call functions of one layer so they are all traced with identical inputs.
  # Holds the layer, its call method, whether it expects a training argument and its index, whether
  # the call has keyword arguments (which rules out a fixed input signature), the inferred input
  # signature, the name of the input argument, and a weak mapping from function name to the traced
  # function. Created by wrap_layer_functions; functions are added with add_function.
  """Groups wrapped layer call functions.

  This is used to ensure that all layer call functions are traced with the same
  inputs-
    - call
    - call_and_return_conditional_losses
    - call_and_return_all_conditional_losses
  """

  def __init__(self, layer):
    # [FUNCTION DOC]
    # Inspects the layer to set up the collection. Input: layer, a Keras layer or model. Stores the
    # layer, the underlying call method (unwrapping a tf.function), whether the layer call uses a
    # boolean training argument and where it sits, whether the call signature has defaults, keyword
    # only arguments or variable keywords (so no input signature can be used), the generated input
    # signature, an empty weak mapping of functions, and the name of the first call argument
    # (default 'inputs' when none).
    self.layer = layer

    self.layer_call_method = _get_layer_call_method(layer)
    self._expects_training_arg = utils.layer_uses_training_bool(layer)
    self._training_arg_index = utils.get_training_arg_index(
        self.layer_call_method)

    # If the layer call function has kwargs, then the traced function cannot
    # have an input signature.
    arg_spec = tf_inspect.getfullargspec(self.layer_call_method)
    self._has_kwargs = bool(self._expects_training_arg or
                            arg_spec.defaults or
                            arg_spec.kwonlyargs or
                            arg_spec.varkw)

    self._input_signature = self._generate_input_signature(layer)
    self._functions = weakref.WeakValueDictionary()

    # Get the input argument name from the args.
    args = arg_spec.args
    if tf_inspect.ismethod(self.layer_call_method):
      args = args[1:]
    self._input_arg_name = args[0] if args else 'inputs'

  def _generate_input_signature(self, layer):
    # [FUNCTION DOC]
    # HELPER: infers the input signature of a layer call. Input: layer. Uses, in order, the input
    # signature of an existing tf.function call, the model input signature for models, or tensor
    # specs derived from the layer input spec (only if the layer allows using it as a call
    # signature; any spec with fully unknown shape becomes null). Returns a list of possibly nested
    # tensor specs, or null when nothing can be inferred.
    """Inspects layer object and returns the inferred input signature.

    Args:
      layer: Layer object.

    Returns:
      List of possibly nested TensorSpecs of the layer call function inputs.
      The list does not contain the `training` argument.
    """
    if (isinstance(layer.call, def_function.Function) and
        layer.call.input_signature is not None):
      return layer.call.input_signature
    elif isinstance(layer, training_lib.Model):
      return saving_utils.model_input_signature(layer)
    elif (layer.input_spec is not None and
          layer._use_input_spec_as_call_signature):  # pylint: disable=protected-access

      def to_tensor_spec_or_none(x):
        # [FUNCTION DOC]
        # HELPER: nested converter. Input: one input spec element. Converts it to a tensor spec
        # using the layer compute dtype and returns it, or returns null when the spec shape is
        # completely unknown, so that separate functions are generated per inferred signature.
        spec = input_spec.to_tensor_spec(x, layer._compute_dtype)  # pylint: disable=protected-access
        # If the shape is too general (e.g. multiple dimensions are allowed),
        # return None so that separate functions can be generated for each
        # inferred input signature.
        # TODO(b/134962016): currently partial signatures are not supported.
        if spec.shape == tensor_shape.TensorShape(None):
          return None
        return spec
      input_signature = [nest.map_structure(
          to_tensor_spec_or_none, layer.input_spec)]

      return input_signature
    else:
      return None

  def add_trace(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Queues traces of every function in the collection with the same arguments. Inputs: the
    # positional and keyword arguments the original call received. Arguments are copied. If the
    # layer call uses a training argument, each function is queued twice, once with training true
    # and once with false (the argument is set in the copied arguments); otherwise once. Queuing
    # only happens while tracing is enabled. Returns nothing.
    """Traces all functions with the same args and kwargs.

    Args:
      *args: Positional args passed to the original function.
      **kwargs: Keyword args passed to the original function.
    """
    args = list(args)
    kwargs = kwargs.copy()

    for fn in self._functions.values():
      # TODO(kathywu): Replace arguments with broader shapes defined in the
      # input signature.
      if self._expects_training_arg:
        def trace_with_training(value, fn=fn):
          # [FUNCTION DOC]
          # HELPER: nested helper. Inputs: value, the training flag to use; fn, the function to
          # trace (bound at definition time). Writes the flag into the shared argument list or
          # mapping at the training argument position, then queues a trace of fn. Returns nothing.
          utils.set_training_arg(value, self._training_arg_index, args, kwargs)
          add_trace_to_queue(fn, args, kwargs, value)

        trace_with_training(True)
        trace_with_training(False)
      else:
        add_trace_to_queue(fn, args, kwargs)

  @property
  def fn_input_signature(self):
    # [FUNCTION DOC]
    # Property giving the input signature usable for wrapped functions. Returns null if the call
    # has keyword arguments or if any part of the inferred signature is unknown; otherwise returns
    # the inferred signature list.
    """Returns input signature for the wrapped layer call function."""
    if self._has_kwargs:
      # Input signatures may only describe tensor arguments and kwargs are not
      # supported.
      return None
    if None in nest.flatten(self._input_signature):
      # TODO(b/134962016): If input signature cannot be partially defined.
      return None
    return self._input_signature

  def training_arg_was_passed(self, args, kwargs):
    # [FUNCTION DOC]
    # Tells whether a training value was supplied. Inputs: args (positional arguments), kwargs
    # (keyword arguments) of a call. If the layer itself has no training argument but the traced
    # function adds one, checks that the value at the training position is not null; otherwise
    # asks the layer whether its training argument was passed. Returns a boolean.
    if not self.layer._expects_training_arg and self._expects_training_arg:  # pylint: disable=protected-access
      return (utils.get_training_arg(self._training_arg_index, args, kwargs)
              is not None)
    else:
      return self.layer._call_arg_was_passed(  # pylint: disable=protected-access
          'training', args, kwargs, inputs_in_args=True)

  def get_training_arg_value(self, args, kwargs):
    # [FUNCTION DOC]
    # Reads the training value from a call. Inputs: args and kwargs of the call. Uses the added
    # training position when only the traced function has such an argument, otherwise asks the
    # layer for the value of its training argument. Returns the value (possibly null).
    if not self.layer._expects_training_arg and self._expects_training_arg:  # pylint: disable=protected-access
      return utils.get_training_arg(self._training_arg_index, args, kwargs)
    else:
      return self.layer._get_call_arg_value(  # pylint: disable=protected-access
          'training', args, kwargs, inputs_in_args=True)

  def get_input_arg_value(self, args, kwargs):
    # [FUNCTION DOC]
    # Reads the value of the layer's main input argument from a call. Inputs: args and kwargs of
    # the call; the lookup uses the recorded input argument name and assumes inputs may be
    # positional. Returns the input value.
    return self.layer._get_call_arg_value(  # pylint: disable=protected-access
        self._input_arg_name, args, kwargs, inputs_in_args=True)

  def _maybe_wrap_with_training_arg(self, call_fn, match_layer_training_arg):
    # [FUNCTION DOC]
    # HELPER: adds a training argument to a function when child layers need one. Inputs: call_fn,
    # the function to wrap; match_layer_training_arg, boolean; when true the added training value
    # is removed before calling call_fn. If the layer lacks a training argument but its children
    # use one, builds an argument spec with an extra trailing training argument defaulting to
    # false, updates the stored training argument index (side effect), and returns a decorated
    # wrapper using that spec. Otherwise returns call_fn unchanged.
    """Wraps call function with added training argument if necessary."""
    if not self.layer._expects_training_arg and self._expects_training_arg:  # pylint: disable=protected-access
      # Add training arg to wrapper function.
      arg_spec = tf_inspect.getfullargspec(call_fn)
      args = arg_spec.args + ['training']
      defaults = list(arg_spec.defaults or [])
      defaults.append(False)
      new_arg_spec = tf_inspect.FullArgSpec(
          args=args,
          varargs=arg_spec.varargs,
          varkw=arg_spec.varkw,
          defaults=defaults,
          kwonlyargs=arg_spec.kwonlyargs,
          kwonlydefaults=arg_spec.kwonlydefaults,
          annotations=arg_spec.annotations)

      # Set new training arg index
      self._training_arg_index = len(args) - 1
      if tf_inspect.ismethod(call_fn):
        self._training_arg_index -= 1

      def wrap_with_training_arg(*args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: nested wrapper. Accepts any arguments; when matching the layer training
        # argument, removes the training value from copies of the arguments, then calls the
        # original function and returns its result.
        if match_layer_training_arg:
          # Remove the training value, since the original call_fn does not
          # expect a training arg. Instead, the training value will be
          # propagated using the call context created in LayerCall.
          args = list(args)
          kwargs = kwargs.copy()
          utils.remove_training_arg(self._training_arg_index, args, kwargs)
        return call_fn(*args, **kwargs)

      return tf_decorator.make_decorator(
          target=call_fn,
          decorator_func=wrap_with_training_arg,
          decorator_argspec=new_arg_spec)

    return call_fn

  def add_function(self, call_fn, name, match_layer_training_arg):
    # [FUNCTION DOC]
    # Registers a call function in the collection. Inputs: call_fn, a python function; name, the
    # function name; match_layer_training_arg, boolean passed to the training-argument wrapping.
    # Creates a LayerCall with the collection's input signature, stores its underlying traced
    # function (weakly) under name, and returns the LayerCall.
    """Adds a layer call function to the collection.

    Args:
      call_fn: a python function
      name: Name of call function
      match_layer_training_arg: If True, removes the `training` from the
        function arguments when calling `call_fn`.

    Returns:
      LayerCall (tf.function)
    """
    fn = LayerCall(
        self,
        self._maybe_wrap_with_training_arg(call_fn, match_layer_training_arg),
        name,
        input_signature=self.fn_input_signature)
    self._functions[name] = fn.wrapped_call
    return fn

  def trace_with_input_signature(self):
    # [FUNCTION DOC]
    # Queues traces using the inferred input signature, but only when that signature is fully
    # defined and the call has keyword arguments (in which case tf.functions could not be given the
    # signature directly). Takes no arguments, returns nothing.
    """Trace with the layer/models inferred input signature if possible."""
    if (None not in nest.flatten(self._input_signature) and self._has_kwargs):
      # Manually add traces for layers that have keyword arguments and have
      # a fully defined input signature.
      self.add_trace(*self._input_signature)


def _filtered_inputs(inputs):
  # [FUNCTION DOC]
  # HELPER: flattens a nested structure and returns a list of only its tensor or variable leaves.
  return list(filter(tf_utils.is_tensor_or_variable, nest.flatten(inputs)))


def layer_call_wrapper(call_collection, method, name):
  # [FUNCTION DOC]
  # Wraps a method so it runs in a saving call context with losses preserved. Inputs:
  # call_collection, the LayerCallCollection (supplies the layer and training-argument access);
  # method, the function to wrap; name, the name given to the result. The wrapper resets layer
  # losses, enters the call context with the tensor inputs, training value (if passed) and saving
  # flag, enables auto-cast variables for the layer compute dtype, calls method, restores losses
  # and returns the method result. Returns the wrapper decorated to look like method and renamed to
  # name (so traced functions are not all called "call").
  """Ensures layer losses are kept the same, and runs method in call context."""

  # Create wrapper that deals with losses and call context.
  def wrapper(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: the actual wrapper created by layer_call_wrapper. Accepts the call arguments,
    # determines the training value if one was passed, runs the method inside a saving call
    # context with auto-cast variables, keeps the layer losses unchanged, and returns the result.
    """Calls method within call context."""
    layer = call_collection.layer
    training = None
    inputs = _filtered_inputs([args, kwargs])
    # pylint: disable=protected-access
    if (args or kwargs) and call_collection.training_arg_was_passed(
        args, kwargs):
      training = call_collection.get_training_arg_value(args, kwargs)
    # pylint: enable=protected-access
    original_losses = _reset_layer_losses(layer)
    with base_layer_utils.call_context().enter(
        layer, inputs=inputs, build_graph=False, training=training,
        saving=True):
      with autocast_variable.enable_auto_cast_variables(
          layer._compute_dtype_object):  # pylint: disable=protected-access
        ret = method(*args, **kwargs)
    _restore_layer_losses(original_losses)
    return ret

  # Rename to `name`, since tf.function doesn't have a name argument. Without
  # this, all functions returned by this method will be named "call", which
  # would be a nightmare to debug.
  fn = tf_decorator.make_decorator(target=method, decorator_func=wrapper)
  fn.__name__ = name
  return fn


class LayerCall(object):
  # [CLASS DOC]
  # Callable wrapper around a traced layer call function. Holds the owning collection, the input
  # signature, the underlying tf.function (wrapped_call) and the original layer call method.
  # Whenever it is called or asked for a concrete function while tracing is enabled, it first
  # queues traces of its sibling functions so all are traced with the same arguments.
  """Function that triggers traces of other functions in the same collection."""

  def __init__(self, call_collection, call_fn, name, input_signature):
    # [FUNCTION DOC]
    # Builds the wrapper. Inputs: call_collection, the LayerCallCollection; call_fn, the python
    # call function; name, its name; input_signature, optional input signature (may be null).
    # Creates the tf.function from call_fn wrapped by layer_call_wrapper and records the layer's
    # original call method.
    """Initializes a LayerCall object.

    Args:
      call_collection: a LayerCallCollection, which contains the other layer
        call functions (e.g. call_with_conditional_losses, call). These
        functions should be traced with the same arguments.
      call_fn: A call function.
      name: Name of the call function.
      input_signature: Input signature of call_fn (can be None).
    """
    self.call_collection = call_collection
    self.input_signature = input_signature
    self.wrapped_call = def_function.function(
        layer_call_wrapper(call_collection, call_fn, name),
        input_signature=input_signature)
    self.original_layer_call = call_collection.layer_call_method

  def _maybe_trace(self, args, kwargs):
    # [FUNCTION DOC]
    # HELPER: when tracing is enabled, asks the collection to queue traces of all sibling functions
    # with the given positional args and keyword args. Returns nothing.
    # Trigger traces of other call functions + extra training-arg traces.
    if tracing_enabled():
      self.call_collection.add_trace(*args, **kwargs)

  def __call__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Calls the wrapped tf.function with the given arguments after queuing sibling traces (when
    # tracing is enabled). Returns the function result.
    self._maybe_trace(args, kwargs)
    return self.wrapped_call(*args, **kwargs)

  def get_concrete_function(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Gets the concrete function of the wrapped tf.function for the given arguments, after queuing
    # sibling traces (when tracing is enabled). Returns the concrete function.
    self._maybe_trace(args, kwargs)
    return self.wrapped_call.get_concrete_function(*args, **kwargs)


def _wrap_call_and_conditional_losses(layer):
  # [FUNCTION DOC]
  # HELPER: builds the function returning outputs plus conditional losses. Input: layer. Returns
  # a decorated function that calls the layer's call method and pairs the output with the
  # conditional losses (via the layer's losses-for-inputs query for v1 layers or models, otherwise
  # all layer losses lacking the unconditional marker). Activity regularizer loss is excluded.
  """Wraps call function that returns a tuple of (outputs, losses).

  The losses returned are conditional on the inputs passed to the call function.
  Unconditional losses (e.g. weight regularizeration) are wrapped separately.

  Args:
    layer: a Keras layer object

  Returns:
    python call function that returns outputs and conditional losses -- excludes
    activity regularizer
  """
  # Create function that generates both outputs and losses
  layer_call = _get_layer_call_method(layer)
  def call_and_return_conditional_losses(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: nested function. Calls the layer with the given arguments and returns a pair of the
    # call output and the list of conditional losses.
    """Returns layer (call_output, conditional losses) tuple."""
    call_output = layer_call(*args, **kwargs)
    if version_utils.is_v1_layer_or_model(layer):
      conditional_losses = layer.get_losses_for(
          _filtered_inputs([args, kwargs]))
    else:
      conditional_losses = [
          l for l in layer.losses if not hasattr(l, '_unconditional_loss')
      ]
    return call_output, conditional_losses

  return _create_call_fn_decorator(layer, call_and_return_conditional_losses)


def _extract_outputs_from_fn(layer, call_and_return_conditional_losses):
  # [FUNCTION DOC]
  # HELPER: builds a call function that returns only the outputs. Inputs: layer;
  # call_and_return_conditional_losses, a function returning outputs and losses. For a revived
  # layer returns its saved keras api call function directly (the second input is then unused).
  # Otherwise returns a decorated function that calls the second input and keeps only element zero.
  """Returns a function that returns only call function outputs."""
  if isinstance(layer, keras_load.RevivedLayer):
    return layer.keras_api.__call__  # pylint: disable=protected-access
  def call(inputs, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: nested function. Calls the outputs-and-losses function and returns only the outputs.
    return call_and_return_conditional_losses(inputs, *args, **kwargs)[0]
  return _create_call_fn_decorator(layer, call)


def _append_activity_regularizer_loss(
    layer, call_fn_with_losses, activity_regularizer_fn):
  # [FUNCTION DOC]
  # HELPER: builds a call function returning outputs and all losses. Inputs: layer;
  # call_fn_with_losses, function returning outputs and conditional losses; activity_regularizer_fn,
  # function mapping outputs to a loss. Returns a decorated function that calls the first, appends
  # the regularizer loss of the outputs to the loss list (mutating it) and returns both.
  """Appends activity regularizer loss to losses returned by the wrapped fn."""
  def fn(inputs, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: nested function implementing the appending behavior described above.
    outputs, losses = call_fn_with_losses(inputs, *args, **kwargs)
    losses.append(activity_regularizer_fn(outputs))
    return outputs, losses
  return _create_call_fn_decorator(layer, fn)


def _create_call_fn_decorator(layer, wrapped_call):
  # [FUNCTION DOC]
  # HELPER: makes a wrapped function look like the layer's call method. Inputs: layer;
  # wrapped_call, the replacement implementation. Adds a training argument if needed (default
  # false) and returns the function decorated with the original call as target and the resulting
  # argument spec, so signatures and names are preserved.
  call_fn = _get_layer_call_method(layer)
  fn, arg_spec = utils.maybe_add_training_arg(
      call_fn, wrapped_call, layer._expects_training_arg,  # pylint: disable=protected-access
      default_training_value=False)
  return tf_decorator.make_decorator(
      target=call_fn,
      decorator_func=fn,
      decorator_argspec=arg_spec)


def _wrap_unconditional_loss(loss_fn, index):
  # [FUNCTION DOC]
  # HELPER: turns a callable loss into a serializable function.
  # KNOWN ISSUE: for a partial application only its first bound positional argument is used,
  # assumed to be the real loss function; any other bound arguments are discarded, and an empty
  # argument list would raise an index error.
  # Inputs: loss_fn, a callable or a partial application; index, an integer used to name the
  # function. Returns the function itself if already a tf.function, otherwise a new tf.function
  # with an empty input signature named by index.
  """Wraps callable/unconditional loss, returning a serializable function."""
  # Extract original loss function from partial function
  fn = loss_fn.args[0] if isinstance(loss_fn, functools.partial) else loss_fn
  if isinstance(fn, def_function.Function):
    return fn
  else:
    return def_function.Function(
        fn, 'loss_fn_{}'.format(index), input_signature=[])


def _wrap_activity_regularizer(layer):
  # [FUNCTION DOC]
  # HELPER: makes the layer's activity regularizer serializable. Input: layer with a non-null
  # regularizer. Returns it unchanged if already a tf.function, else a new tf.function named after
  # the layer with a single input signature of unknown shape and the layer compute dtype (or the
  # default float type when none).
  """Wraps the activity regularizer."""
  # pylint: disable=protected-access
  if isinstance(layer._activity_regularizer, def_function.Function):
    return layer._activity_regularizer
  return def_function.Function(
      layer._activity_regularizer,
      '{}_activity_regularizer'.format(layer.name),
      input_signature=[
          tensor_spec.TensorSpec(None, layer._compute_dtype or K.floatx())
      ])
  # pylint: enable=protected-access


def _get_layer_call_method(layer):
  # [FUNCTION DOC]
  # HELPER: returns the plain python call method of a layer, unwrapping its tf.function if the call
  # has already been converted into one; otherwise returns the call method itself.
  if isinstance(layer.call, (def_function.Function)):
    return layer.call.python_function
  return layer.call
