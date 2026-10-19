# [MODULE DOC]
# HELPER: utility module shared by the Keras SavedModel saving and loading code (save_impl,
# load and the layer savers). Provides helpers to wrap traced call functions so they add losses
# and handle the training argument, to inspect where the training argument sits in a call
# signature and read/write/remove it in argument lists, to enumerate layers and sublayers, a
# thread-local option controlling whether layer traces are saved, and a context manager that turns
# off automatic dependency tracking on attribute assignment. Import-time side effects: creates a
# lazy loader for the training engine module and a thread-local save options object.
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
"""Utility functions shared between SavedModel saving/loading implementations."""

import itertools
import threading
import types

from tensorflow.python.eager import context
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils.generic_utils import LazyLoader
from tensorflow.python.util import tf_decorator


# pylint:disable=g-inconsistent-quotes
training_lib = LazyLoader(
    "training_lib", globals(),
    "tensorflow.python.keras.engine.training")
# pylint:enable=g-inconsistent-quotes


def use_wrapped_call(layer, call_fn, default_training_value=None,
                     return_method=False):
  # [FUNCTION DOC]
  # HELPER: wraps a traced call function so a restored or saved layer can call it like its own
  # call method. Inputs: layer, the Keras layer; call_fn, a function (or LayerCall object, in which
  # case its original layer call is used for argument inspection) returning outputs and a list of
  # losses; default_training_value, default for the training keyword (none means use the backend
  # learning phase); return_method, when true the result is bound to the layer as a method and the
  # leading self argument is dropped before calling. The returned function calls call_fn, adds the
  # returned losses to the layer as input-dependent losses (side effect), and, when executing
  # eagerly, sets a placeholder eager loss list on every child layer (a temporary hack for revived
  # layers). It returns only the outputs. Returns the decorated function or bound method.
  """Creates fn that adds the losses returned by call_fn & returns the outputs.

  Args:
    layer: A Keras layer object
    call_fn: tf.function that takes layer inputs (and possibly a training arg),
      and returns a tuple of (outputs, list of losses).
    default_training_value: Default value of the training kwarg. If `None`, the
      default is `K.learning_phase()`.
    return_method: Whether to return a method bound to the layer.

  Returns:
    function that calls call_fn and returns the outputs. Losses returned by
    call_fn are added to the layer losses.
  """
  expects_training_arg = layer_uses_training_bool(layer)
  if hasattr(call_fn, 'original_layer_call'):  # call_fn is a LayerCall object
    original_call = call_fn.original_layer_call
    # In Python 3, callable objects are not compatible with inspect.getargspec
    call_fn = call_fn.__call__
  else:
    original_call = call_fn
  fn, arg_spec = maybe_add_training_arg(
      original_call, call_fn, expects_training_arg, default_training_value)

  def return_outputs_and_add_losses(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: nested wrapper. Drops the first argument when a bound method is wanted, calls the
    # wrapped function to get outputs and losses, adds the losses to the layer, and in eager mode
    # replaces eager losses of all sublayers with the revived-loss placeholder. Returns outputs.
    """Returns the outputs from the layer call function, and adds the losses."""
    if return_method:
      args = args[1:]

    outputs, losses = fn(*args, **kwargs)
    layer.add_loss(losses, inputs=True)

    # TODO(kathywu): This is a temporary hack. When a network of layers is
    # revived from SavedModel, only the top-level layer will have losses. This
    # causes issues in eager mode because the child layers may have graph losses
    # (thus model.losses returns a mix of Eager and graph tensors). To fix this,
    # whenever eager losses are added to one layer, add eager losses to all
    # child layers. This causes `.losses` to only return eager losses.
    # pylint: disable=protected-access
    if context.executing_eagerly():
      for i in layer._flatten_layers():
        if i is not layer:
          i._eager_losses = [base_layer_utils.REVIVED_LOSS_PLACEHOLDER]
    # pylint: enable=protected-access
    return outputs

  decorated = tf_decorator.make_decorator(
      target=call_fn,
      decorator_func=return_outputs_and_add_losses,
      decorator_argspec=arg_spec)

  if return_method:
    return types.MethodType(decorated, layer)
  else:
    return decorated


def layer_uses_training_bool(layer):
  # [FUNCTION DOC]
  # Tells whether a layer or any descendant takes a training argument. Input: layer. True right
  # away if the layer expects it; otherwise walks all sublayers (each visited once) and returns
  # true on the first one expecting it. A sublayer lacking the attribute altogether is treated as
  # expecting it (the default is true). Returns a boolean.
  """Returns whether this layer or any of its children uses the training arg."""
  if layer._expects_training_arg:  # pylint: disable=protected-access
    return True
  visited = {layer}
  to_visit = list_all_layers(layer)
  while to_visit:
    layer = to_visit.pop()
    if layer in visited:
      continue
    if getattr(layer, '_expects_training_arg', True):
      return True
    visited.add(layer)
    to_visit.extend(list_all_layers(layer))
  return False


def list_all_layers(obj):
  # [FUNCTION DOC]
  # HELPER: returns the direct child layers of obj as a list. For a Model, returns its layers
  # property (which, for Sequential, omits the input layer); otherwise the non-recursive flattened
  # sublayers excluding obj itself.
  if isinstance(obj, training_lib.Model):
    # Handle special case of Sequential, which doesn't return
    # the `Input` layer.
    return obj.layers
  else:
    return list(obj._flatten_layers(include_self=False, recursive=False))  # pylint: disable=protected-access


def list_all_layers_and_sublayers(obj):
  # [FUNCTION DOC]
  # HELPER: recursively collects obj and every nested sublayer into a set (using the direct
  # children from list_all_layers at each level). Returns the set.
  s = set([obj])
  s.update(itertools.chain.from_iterable(
      list_all_layers_and_sublayers(layer) for layer in list_all_layers(obj)))
  return s


def maybe_add_training_arg(
    original_call, wrapped_call, expects_training_arg, default_training_value):
  # [FUNCTION DOC]
  # HELPER: gives a wrapped call function a training argument when the layer expects one.
  # KNOWN ISSUE: the default is chosen with a truthiness test, so a default_training_value of
  # false is treated like none and the backend learning phase is used instead, contrary to what
  # the docstring implies. Also the argument-spec lists of the original call (such as keyword-only
  # names and defaults) are appended to in place rather than copied.
  # Inputs: original_call, the layer call used to inspect the signature; wrapped_call, the
  # function to run; expects_training_arg, boolean; default_training_value, default flag.
  # If no training argument is expected returns the wrapped call and null. Otherwise builds a
  # wrapper that finds the training value in the call arguments (or falls back to the default or
  # learning phase), then runs wrapped_call through a smart conditional once with training true
  # and once with false (so both branches are covered in graph mode), and a new argument spec
  # in which training is added as keyword-only with the default, or its none default replaced.
  # Returns a pair of the wrapper and the argument spec.
  """Decorate call and optionally adds training argument.

  If a layer expects a training argument, this function ensures that 'training'
  is present in the layer args or kwonly args, with the default training value.

  Args:
    original_call: Original call function.
    wrapped_call: Wrapped call function.
    expects_training_arg: Whether to include 'training' argument.
    default_training_value: Default value of the training kwarg to include in
      the arg spec. If `None`, the default is `K.learning_phase()`.

  Returns:
    Tuple of (
      function that calls `wrapped_call` and sets the training arg,
      Argspec of returned function or `None` if the argspec is unchanged)
  """
  if not expects_training_arg:
    return wrapped_call, None
  def wrap_with_training_arg(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: nested wrapper created by maybe_add_training_arg. Determines the training value from
    # the call arguments (default or learning phase when absent) and calls the wrapped function
    # under a conditional on that value, returning its result.
    """Wrap the `wrapped_call` function, and set training argument."""
    training_arg_index = get_training_arg_index(original_call)
    training = get_training_arg(training_arg_index, args, kwargs)
    if training is None:
      training = default_training_value or K.learning_phase()

    args = list(args)
    kwargs = kwargs.copy()

    def replace_training_and_call(training):
      # [FUNCTION DOC]
      # HELPER: nested function. Input: training, a boolean. Writes it into the copied arguments
      # and calls the wrapped function, returning the result.
      set_training_arg(training, training_arg_index, args, kwargs)
      return wrapped_call(*args, **kwargs)

    return control_flow_util.smart_cond(
        training, lambda: replace_training_and_call(True),
        lambda: replace_training_and_call(False))

  # Create arg spec for decorated function. If 'training' is not defined in the
  # args of the original arg spec, then add it to kwonlyargs.
  arg_spec = tf_inspect.getfullargspec(original_call)
  defaults = list(arg_spec.defaults) if arg_spec.defaults is not None else []

  kwonlyargs = arg_spec.kwonlyargs
  kwonlydefaults = arg_spec.kwonlydefaults or {}
  # Add training arg if it does not exist, or set the default training value.
  if 'training' not in arg_spec.args:
    kwonlyargs.append('training')
    kwonlydefaults['training'] = default_training_value
  else:
    index = arg_spec.args.index('training')
    training_default_index = len(arg_spec.args) - index
    if (arg_spec.defaults and
        len(arg_spec.defaults) >= training_default_index and
        defaults[-training_default_index] is None):
      defaults[-training_default_index] = default_training_value

  decorator_argspec = tf_inspect.FullArgSpec(
      args=arg_spec.args,
      varargs=arg_spec.varargs,
      varkw=arg_spec.varkw,
      defaults=defaults,
      kwonlyargs=kwonlyargs,
      kwonlydefaults=kwonlydefaults,
      annotations=arg_spec.annotations)
  return wrap_with_training_arg, decorator_argspec


def get_training_arg_index(call_fn):
  # [FUNCTION DOC]
  # Finds where the training argument sits in a call function's signature. Input: call_fn.
  # With variable positional arguments, training must be a keyword: returns minus one if it is
  # keyword-only or the function takes variable keywords, else null. Otherwise returns its position
  # among the positional names (self skipped for bound methods), minus one when only keyword-able,
  # or null when the signature has no training argument.
  """Returns the index of 'training' in the layer call function arguments.

  Args:
    call_fn: Call function.

  Returns:
    - n: index of 'training' in the call function arguments.
    - -1: if 'training' is not found in the arguments, but layer.call accepts
          variable keyword arguments
    - None: if layer doesn't expect a training argument.
  """
  argspec = tf_inspect.getfullargspec(call_fn)
  if argspec.varargs:
    # When there are variable args, training must be a keyword arg.
    if 'training' in argspec.kwonlyargs or argspec.varkw:
      return -1
    return None
  else:
    # Try to find 'training' in the list of args or kwargs.
    arg_list = argspec.args
    if tf_inspect.ismethod(call_fn):
      arg_list = arg_list[1:]

    if 'training' in arg_list:
      return arg_list.index('training')
    elif 'training' in argspec.kwonlyargs or argspec.varkw:
      return -1
    return None


def set_training_arg(training, index, args, kwargs):
  # [FUNCTION DOC]
  # Stores a training value into call arguments. Inputs: training, the value; index, position
  # from get_training_arg_index; args, mutable list of positional arguments; kwargs, mutable
  # mapping of keyword arguments. If the index is null, negative or beyond the positional list,
  # puts the value in the keyword mapping, otherwise overwrites the positional entry. Mutates the
  # inputs and returns the pair of args and kwargs.
  if index is None or index < 0 or len(args) <= index:  # index is invalid
    kwargs['training'] = training
  else:
    args[index] = training
  return args, kwargs


def get_training_arg(index, args, kwargs):
  # [FUNCTION DOC]
  # Reads the training value from call arguments. Inputs: index (from get_training_arg_index),
  # args (positional list), kwargs (keyword mapping). Uses the positional entry when the index is
  # valid, else the keyword value, or null when absent. Returns the value.
  if index is None or index < 0 or len(args) <= index:  # index is invalid
    return kwargs.get('training', None)
  else:
    return args[index]


def remove_training_arg(index, args, kwargs):
  # [FUNCTION DOC]
  # Removes the training value from call arguments, in place. Inputs: index, args (mutable
  # positional list), kwargs (mutable keyword mapping). Pops the positional entry when the index is
  # valid, else drops the keyword entry if present. Returns nothing.
  if index is None or index < 0 or len(args) <= index:  # index is invalid
    kwargs.pop('training', None)
  else:
    args.pop(index)


class SaveOptionsContext(threading.local):
  # [CLASS DOC]
  # HELPER: thread-local holder of Keras save options. Currently one flag, whether layer call
  # traces should be saved, default true. A single module-level instance is shared.

  def __init__(self):
    # [FUNCTION DOC]
    # Initializes the per-thread state with the save-traces flag set to true. Returns nothing.
    super(SaveOptionsContext, self).__init__()
    self.save_traces = True


_save_options_context = SaveOptionsContext()


@tf_contextlib.contextmanager
def keras_option_scope(save_traces):
  # [FUNCTION DOC]
  # Context manager that temporarily sets the thread-local save-traces option. Input: save_traces,
  # boolean. Sets the flag on entry and restores the previous value on exit, even after an error.
  # Yields nothing.
  previous_value = _save_options_context.save_traces
  try:
    _save_options_context.save_traces = save_traces
    yield
  finally:
    _save_options_context.save_traces = previous_value


def should_save_traces():
  # [FUNCTION DOC]
  # Returns the current thread's save-traces option (true by default, changed by
  # keras_option_scope). No arguments, no side effects.
  """Whether to trace layer functions-can be disabled in the save_traces arg."""
  return _save_options_context.save_traces


@tf_contextlib.contextmanager
def no_automatic_dependency_tracking_scope(obj):
  # [FUNCTION DOC]
  # Context manager disabling automatic dependency tracking on attribute assignment. Input: obj,
  # a trackable object. Remembers its current tracking flag (default true if missing), turns the
  # flag off for the duration, and restores the remembered value on exit even after an error.
  # Yields nothing.
  """A context that disables automatic dependency tracking when assigning attrs.

  Objects that inherit from Autotrackable automatically creates dependencies
  to trackable objects through attribute assignments, and wraps data structures
  (lists or dicts) with trackable classes. This scope may be used to temporarily
  disable this behavior. This works similar to the decorator
  `no_automatic_dependency_tracking`.

  Example usage:
  ```
  model = tf.keras.Model()
  model.arr1 = []  # Creates a ListWrapper object
  with no_automatic_dependency_tracking_scope(model):
    model.arr2 = []  # Creates a regular, untracked python list
  ```

  Args:
    obj: A trackable object.

  Yields:
    a scope in which the object doesn't track dependencies.
  """
  previous_value = getattr(obj, '_setattr_tracking', True)
  obj._setattr_tracking = False  # pylint: disable=protected-access
  try:
    yield
  finally:
    obj._setattr_tracking = previous_value  # pylint: disable=protected-access
