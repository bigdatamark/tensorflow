# [MODULE DOC]
# HELPER: helper module for the TensorFlow Lite converter's handling of Keras models.
# Holds copies of a few Keras utilities (save spec lookup, input signature inference, pseudo
# output names, tracing a model call into a graph function) so TensorFlow Lite does not depend on
# Keras directly; Keras-specific pieces are reached through the dependency inversion helpers.
# No import-time side effects.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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

"""Keras functions required by TensorFlow Lite.

The functions defined in this library have been copied over from Keras in order
to remove the dependency from TensorFlow Lite to Keras. The functions which
could not be copied over are accessed using the dependency inversion principle.
(for details, refer to tensorflow/python/util/keras_deps.py).
"""

import copy

from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_spec
from tensorflow.python.util import keras_deps
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc


def _enforce_names_consistency(specs):
  # [FUNCTION DOC]
  # HELPER: used by the model input signature code to normalize spec names.
  # Input: specs, a possibly nested structure of tensor specs. Flattens it; if some but not all
  # have names, replaces the structure with copies in which every name is cleared. Returns the
  # (possibly rewritten) structure; otherwise returns the input unchanged. No mutation of input.
  """Enforces that either all specs have names or none do."""

  def _has_name(spec):
    # [FUNCTION DOC]
    # HELPER: true when the spec has a name attribute that is not null.
    return hasattr(spec, 'name') and spec.name is not None

  def _clear_name(spec):
    # [FUNCTION DOC]
    # HELPER: returns a deep copy of the spec with its private name field set to null, if the spec
    # has a name attribute; the original is not modified.
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


def get_save_spec(model):
  # [FUNCTION DOC]
  # Input: model, a subclassed Keras model. Reads its recorded build shapes mapping (if absent or
  # empty returns nothing/null). Raises a value error if that mapping lacks an input shape entry
  # (KNOWN ISSUE: the error message has an unfilled placeholder and is never formatted with the
  # model). The batch dimension is replaced by an unknown size. For a tuple shape returns one
  # tensor spec; for a mapping of shapes returns a mapping of named tensor specs; for a list of
  # shapes returns a list of specs; all use the model's input dtype. Any other shape type
  # implicitly returns nothing.
  """Returns the save spec of the subclassing keras model."""
  shapes_dict = getattr(model, '_build_shapes_dict', None)
  if not shapes_dict:
    return None

  if 'input_shape' not in shapes_dict:
    raise ValueError(
        'Model {} cannot be saved because the input shapes have not been set.'
    )

  input_shape = shapes_dict['input_shape']
  if isinstance(input_shape, tuple):
    shape = input_shape
    shape = (None,) + shape[1:]
    return tensor_spec.TensorSpec(
        shape=shape, dtype=model.input_dtype
    )
  elif isinstance(input_shape, dict):
    specs = {}
    for key, shape in input_shape.items():
      shape = (None,) + shape[1:]
      specs[key] = tensor_spec.TensorSpec(
          shape=shape, dtype=model.input_dtype, name=key
      )
    return specs
  elif isinstance(input_shape, list):
    specs = []
    for shape in input_shape:
      shape = (None,) + shape[1:]
      specs.append(tensor_spec.TensorSpec(shape=shape, dtype=model.input_dtype))
    return specs


def model_input_signature(model, keep_original_batch_size=False):
  # [FUNCTION DOC]
  # Inputs: model, a Keras model; keep_original_batch_size, optional boolean (default false) which
  # keeps the real batch size instead of making it dynamic. Gets the model's save spec (using the
  # public method if present, otherwise the private one), returns nothing/null if there is none,
  # takes the first positional argument spec, makes names consistent, and returns it as a list
  # with one element (a single-element sequence is returned as is, anything else is wrapped).
  # Output: list of specs or null. No exceptions of its own.
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
  if hasattr(model, 'save_spec'):
    input_specs = model.save_spec(dynamic_batch=not keep_original_batch_size)
    if input_specs is None:
      return None
    # The model's save spec returns (args, kwargs). Extract the first input arg
    # to use as the input spec.
    # TODO(b/188105669): Add support for multiple tensor arguments.
    input_specs = input_specs[0][0]
  else:
    input_specs = model._get_save_spec(  # pylint: disable=protected-access
        dynamic_batch=not keep_original_batch_size)
    if input_specs is None:
      return None
  input_specs = _enforce_names_consistency(input_specs)
  # Return a list with a single element as the model's input signature.
  if isinstance(input_specs,
                collections_abc.Sequence) and len(input_specs) == 1:
    # Note that the isinstance check filters out single-element dictionaries,
    # which should also be wrapped as a single-element list.
    return input_specs
  else:
    return [input_specs]


def raise_model_input_error(model):
  # [FUNCTION DOC]
  # HELPER: shared error raiser for the model tracing code.
  # Input: model, only used in the message. Always raises a value error saying the model cannot be
  # saved because input shapes were not set and suggesting to call fit, predict or build.
  raise ValueError(
      'Model {} cannot be saved because the input shapes have not been '
      'set. Usually, input shapes are automatically determined from calling'
      ' `.fit()` or `.predict()`. To manually set the shapes, call '
      '`model.build(input_shape)`.'.format(model))


def _create_pseudo_names(tensors, prefix):
  # [FUNCTION DOC]
  # HELPER: builds default names for outputs of subclassed models.
  # Inputs: tensors, a nested structure of model inputs or outputs; prefix, text such as output_
  # or input_. Flattens the structure into paths, makes integer indices one-based, and builds one
  # name per leaf: a lone unnested tensor gets the prefix plus 1; other paths are joined with
  # underscores, with the prefix added when the path starts with an integer. Returns the list.
  """Creates pseudo {input | output} names for subclassed Models.

  Warning: this function should only be used to define default
  names for `Metics` and `SavedModel`. No other use cases should
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
    # HELPER: converts a zero-based integer path element into one-based; other values unchanged.
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


def create_pseudo_output_names(outputs):
  # [FUNCTION DOC]
  # Input: outputs, the nested outputs of a subclassed model. Returns the list of pseudo output
  # names built with the output_ prefix.
  """Create pseudo output names for a subclassed Model."""
  return _create_pseudo_names(outputs, prefix='output_')


def trace_model_call(model, input_signature=None):
  # [FUNCTION DOC]
  # Inputs: model, a Keras model; input_signature, optional list of tensor specs. If not given,
  # uses the signature of the model's call if that is already a graph function, else infers one
  # from the model; raises a value error (via the shared error helper) if none can be found.
  # Returns a graph function (autograph disabled) with that input signature wrapping the model call
  # in inference mode. Side effects: none until the returned function is traced or called.
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

  @def_function.function(input_signature=input_signature, autograph=False)
  def _wrapped_model(*args):
    # [FUNCTION DOC]
    # HELPER: the traced wrapper returned by the model tracing function.
    # Takes the positional tensors; with a single-input signature the single tensor is passed to
    # the model, otherwise a list. Enters the Keras call context in saving mode with training
    # false and no graph building, calls the model with training off, and returns its outputs.
    """A concrete tf.function that wraps the model's call function."""
    # When given a single input, Keras models will call the model on the tensor
    # rather than a list consisting of the single tensor.
    inputs = args[0] if len(input_signature) == 1 else list(args)

    with keras_deps.get_call_context_function()().enter(
        model,
        inputs=inputs,
        build_graph=False,
        call_context_args={'training': False},
        saving=True,
    ):
      outputs = model(inputs, training=False)

    return outputs

  return _wrapped_model
