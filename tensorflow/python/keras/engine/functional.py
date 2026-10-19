# [MODULE DOC]
# Implements the functional model class (a model defined as a graph of layers) together with the
# graph analysis, serialization (config building and reconstruction) and module-wrapper helpers
# it relies on. Used by the Model constructor, Sequential and model cloning/saving.
# No import-time side effects other than class definitions.
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
# pylint: disable=protected-access
"""A `Network` is way to compose layers: the topological form of a `Model`."""

import collections
import copy
import itertools
import warnings

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.engine import input_layer as input_layer_module
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.keras.engine import node as node_module
from tensorflow.python.keras.engine import training as training_lib
from tensorflow.python.keras.engine import training_utils
from tensorflow.python.keras.saving.saved_model import network_serialization
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.trackable import base as trackable
from tensorflow.python.util import nest
from tensorflow.tools.docs import doc_controls


# pylint: disable=g-classes-have-attributes
class Functional(training_lib.Model):
  # [CLASS DOC]
  # A model defined as a static directed acyclic graph of layers, built from symbolic input
  # tensors and output tensors (each possibly a nested structure). It extends the base model
  # class with graph-network behavior: it records the input and output layers and their
  # coordinates, the graph nodes grouped by depth, tracked layers, input and output names, and
  # caches for output tensors, masks and shapes. Calling it replays the layers in depth order.
  # It also supports configuration/serialization, reconstruction from configuration, input
  # specs, and layer lookup by name or index. Sequential models reuse it by reinitializing the
  # graph network. Instances are usually produced by the Model constructor class swap.
  """A `Functional` model is a `Model` defined as a directed graph of layers.

  Three types of `Model` exist: subclassed `Model`, `Functional` model,
  and `Sequential` (a special case of `Functional`).
  In general, more Keras features are supported with `Functional`
  than with subclassed `Model`s, specifically:

  - Model cloning (`keras.models.clone`)
  - Serialization (`model.get_config()/from_config`, `model.to_json()`
  - Whole-model saving (`model.save()`)

  A `Functional` model can be instantiated by passing two arguments to
  `__init__`. The first argument is the `keras.Input` Tensors that represent
  the inputs to the model. The second argument specifies the output
  tensors that represent the outputs of this model. Both arguments can be a
  nested structure of tensors.

  Example:

  ```
  inputs = {'x1': keras.Input(shape=(10,)), 'x2': keras.Input(shape=(1,))}
  t = keras.layers.Dense(1, activation='relu')(inputs['x1'])
  outputs = keras.layers.Add()([t, inputs['x2'])
  model = keras.Model(inputs, outputs)
  ```

  A `Functional` model constructed using the Functional API can also include raw
  TensorFlow functions, with the exception of functions that create Variables
  or assign ops.

  Example:

  ```
  inputs = keras.Input(shape=(10,))
  x = keras.layers.Dense(1)(inputs)
  outputs = tf.nn.relu(x)
  model = keras.Model(inputs, outputs)
  ```

  Args:
    inputs: List of input tensors (must be created via `tf.keras.Input()`).
    outputs: List of output tensors.
    name: String, optional. Name of the model.
    trainable: Boolean, optional. If the model's variables should be trainable.
  """

  # See tf.Module for the usage of this property.
  # The key of _layer_call_argspecs is a layer. tf.Module._flatten will fail to
  # flatten the key since it is trying to convert Trackable/Layer to a string.
  _TF_MODULE_IGNORED_PROPERTIES = frozenset(itertools.chain(
      ('_layer_call_argspecs', '_compiled_trainable_state',
       '_output_mask_cache', '_output_tensor_cache', '_output_shape_cache'),
      training_lib.Model._TF_MODULE_IGNORED_PROPERTIES
  ))

  @trackable.no_automatic_dependency_tracking
  def __init__(self, inputs, outputs, name=None, trainable=True,
               **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: inputs (symbolic input tensor, or list/mapping/nested structure of
    # them, made by the Input function), outputs (tensor or nested structure of output tensors),
    # name (optional string), trainable (flag, default true), and extra keyword arguments of
    # which only the internal flag that skips initialization is consumed.
    # Behavior: if the skip flag is set it returns immediately (the Model class swaps classes and
    # would otherwise initialize twice). Otherwise it validates that no other keywords remain,
    # initializes the base model and builds the graph network from inputs and outputs.
    # Raises a type or value error for unknown keyword arguments; graph errors from the build.
    # Dependency tracking is disabled for attribute assignments during construction.
    # This is used by the Model class, since we have some logic to swap the
    # class in the __new__ method, which will lead to __init__ get invoked
    # twice. Using the skip_init to skip one of the invocation of __init__ to
    # avoid any side effects
    skip_init = kwargs.pop('skip_init', False)
    if skip_init:
      return
    generic_utils.validate_kwargs(kwargs, {})
    super(Functional, self).__init__(name=name, trainable=trainable)
    self._init_graph_network(inputs, outputs)

  @trackable.no_automatic_dependency_tracking
  def _init_graph_network(self, inputs, outputs):
    # [FUNCTION DOC]
    # HELPER: graph-network initialization shared by the constructor and by Sequential when
    # layers are added or removed.
    # KNOWN ISSUE: input layers are validated with plain assertions on node and tensor index
    # (stripped under optimized runs) instead of raising proper errors.
    # Inputs: inputs and outputs, each a tensor or nested structure of tensors carrying layer
    # history. A one-element list is unwrapped to its element. Returns nothing.
    # Behavior: stores nested and flat inputs and outputs; decides whether a mapping of names to
    # input values may be used when calling (only for flat structures); in graph mode, creates
    # missing layer history for outputs; validates graph inputs and outputs; marks the model
    # built, always expecting training and mask arguments, no autocast, remembers the input
    # shapes; resets the output tensor, mask and shape caches; records input and output layers
    # and coordinates; maps the graph to get nodes, nodes by depth and unique layers (tracked as
    # sub-trackables, with their call signatures stored); computes input names and the feed
    # input names, shapes and tensors for layers that are placeholders; computes tensor usage
    # counts, the save spec, and asserts no legacy layers are present.
    # Side effects: sets many attributes on the model. Raises value errors on invalid graphs.
    # This method is needed for Sequential to reinitialize graph network when
    # layer is added or removed.
    self._is_graph_network = True

    # Normalize and set self.inputs, self.outputs.
    if isinstance(inputs, list) and len(nest.flatten(inputs)) == 1:
      inputs = inputs[0]
    if isinstance(outputs, list) and len(nest.flatten(outputs)) == 1:
      outputs = outputs[0]
    self._nested_inputs = inputs
    self._nested_outputs = outputs
    self.inputs = nest.flatten(inputs)
    self.outputs = nest.flatten(outputs)

    # Models constructed with a single Tensor or list of Tensors can
    # be called with a dict, where the keys of the dict are the names
    # of the `Input` objects. Extra keys are ignored with warning.
    if not nest.is_nested(self._nested_inputs):
      self._enable_dict_to_input_mapping = True
    elif (isinstance(self._nested_inputs, (list, tuple)) and
          not any(nest.is_nested(t) for t in self._nested_inputs)):
      self._enable_dict_to_input_mapping = True
    elif (isinstance(self._nested_inputs, dict) and
          not any(nest.is_nested(t) for t in self._nested_inputs.values())):
      self._enable_dict_to_input_mapping = True
    else:
      self._enable_dict_to_input_mapping = False

    if not ops.executing_eagerly_outside_functions():
      if any(not hasattr(tensor, '_keras_history') for tensor in self.outputs):
        base_layer_utils.create_keras_history(self._nested_outputs)

    self._validate_graph_inputs_and_outputs()

    # A Network does not create weights of its own, thus it is already
    # built.
    self.built = True
    self._build_input_shape = nest.map_structure(lambda x: x.shape, inputs)
    self._compute_output_and_mask_jointly = True
    # `_expects_training_arg` is True since the `training` argument is always
    # present in the signature of the `call` method of a graph network.
    self._expects_training_arg = True
    self._expects_mask_arg = True
    # A graph network does not autocast inputs, as its layers will cast them
    # instead.
    self._autocast = False

    self._input_layers = []
    self._output_layers = []
    self._input_coordinates = []
    self._output_coordinates = []

    # This is for performance optimization when calling the Network on new
    # inputs. Every time the Network is called on a set on input tensors,
    # we compute the output tensors, output masks and output shapes in one pass,
    # then cache them here. When any of these outputs is queried later, we
    # retrieve it from there instead of recomputing it.
    self._output_mask_cache = {}
    self._output_tensor_cache = {}
    self._output_shape_cache = {}

    # Build self._output_layers:
    for x in self.outputs:
      layer, node_index, tensor_index = x._keras_history  # pylint: disable=protected-access
      self._output_layers.append(layer)
      self._output_coordinates.append((layer, node_index, tensor_index))

    # Build self._input_layers:
    for x in self.inputs:
      layer, node_index, tensor_index = x._keras_history  # pylint: disable=protected-access
      # It's supposed to be an input layer, so only one node
      # and one tensor output.
      assert node_index == 0
      assert tensor_index == 0
      self._input_layers.append(layer)
      self._input_coordinates.append((layer, node_index, tensor_index))

    # Keep track of the network's nodes and layers.
    nodes, nodes_by_depth, layers, _ = _map_graph_network(
        self.inputs, self.outputs)
    self._network_nodes = nodes
    self._nodes_by_depth = nodes_by_depth
    self._self_tracked_trackables = layers
    self._layer_call_argspecs = {}
    for layer in self._self_tracked_trackables:
      self._layer_call_argspecs[layer] = tf_inspect.getfullargspec(layer.call)

    # Build self.input_names and self.output_names.
    self._set_output_names()
    self.input_names = []
    self._feed_input_names = []
    self._feed_inputs = []
    self._feed_input_shapes = []
    for layer in self._input_layers:
      self.input_names.append(layer.name)
      if layer.is_placeholder:
        self._feed_input_names.append(layer.name)
        # Use batch_input_shape here because non-eager composite tensors may not
        # have a shape attribute that's meaningful (sparse, for instance, has
        # a tensor that's non-constant and needs to be fed). This means that
        # input layers that create placeholders will need to have the
        # batch_input_shape attr to allow for input shape validation.
        self._feed_input_shapes.append(layer._batch_input_shape)
        self._feed_inputs.append(layer.input)

    self._compute_tensor_usage_count()
    self._set_save_spec(self._nested_inputs)
    tf_utils.assert_no_legacy_layers(self.layers)

  @property
  def input(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring claims it raises errors in eager mode or when there are no
    # inbound nodes, but it simply returns the stored nested input structure and never raises.
    # Property returning the model's input tensor or nested structure exactly as given at
    # construction. No side effects.
    """Retrieves the input tensor(s) of a layer.

    Only applicable if the layer has exactly one input,
    i.e. if it is connected to one incoming layer.

    Returns:
        Input tensor or list of input tensors.

    Raises:
      RuntimeError: If called in Eager mode.
      AttributeError: If no inbound nodes are found.
    """
    return self._nested_inputs

  @property
  def input_shape(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring mentions attribute and eager-mode errors, and a single-input
    # restriction, but the code just maps the backend integer-shape function over the input
    # structure and returns a matching structure; it adds no such checks.
    # Property returning the static shape (tuple of ints or unknown entries) of every input,
    # in the same structure as the inputs. No side effects.
    """Retrieves the input shape(s) of a layer.

    Only applicable if the layer has exactly one input,
    i.e. if it is connected to one incoming layer, or if all inputs
    have the same shape.

    Returns:
        Input shape, as an integer shape tuple
        (or list of shape tuples, one tuple per input tensor).

    Raises:
        AttributeError: if the layer has no defined input_shape.
        RuntimeError: if called in Eager mode.
    """
    return nest.map_structure(backend.int_shape, self.input)

  @property
  def input_spec(self):
    # [FUNCTION DOC]
    # Property computing the input specification used to validate inputs when the model is
    # called. If a manual spec was assigned it is returned as is. If the nested inputs are a
    # list, tuple or mapping whose length differs from the flat input count (a nested structure),
    # nothing is returned because no safe check is possible. For a mapping of inputs it returns
    # one spec per key in sorted key order, each with the input's shape without the batch
    # dimension, last-axis squeeze allowed and the key as name. Otherwise it returns one spec per
    # input, named after the input's layer. No side effects.
    if hasattr(self, '_manual_input_spec'):
      return self._manual_input_spec
    if (isinstance(self._nested_inputs, (dict, list, tuple)) and
        len(self._nested_inputs) != len(self.inputs)):
      # Case where we have a nested structure.
      # In such a case we can't safely run any checks.
      return None
    if isinstance(self._nested_inputs, dict):
      # Case where `_nested_inputs` is a plain dict of Inputs.
      names = sorted(self._nested_inputs.keys())
      return [input_spec.InputSpec(
          shape=shape_with_no_batch_size(self._nested_inputs[name]),
          allow_last_axis_squeeze=True, name=name) for name in names]
    else:
      # Single input, or list / tuple of inputs.
      # The data may be passed as a dict keyed by input name.
      return [input_spec.InputSpec(
          shape=shape_with_no_batch_size(x), allow_last_axis_squeeze=True,
          name=x._keras_history.layer.name) for x in self.inputs]

  @input_spec.setter
  def input_spec(self, value):
    # [FUNCTION DOC]
    # Property setter that overrides the computed input spec with the given value (any spec
    # object, structure of them, or nothing); the getter will then return it unchanged.
    self._manual_input_spec = value

  @property
  def output(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring lists connection and eager-mode errors, but the code only returns
    # the stored nested outputs and never raises.
    # Property returning the model's output tensor or nested structure as given at construction.
    """Retrieves the output tensor(s) of a layer.

    Only applicable if the layer has exactly one output,
    i.e. if it is connected to one incoming layer.

    Returns:
      Output tensor or list of output tensors.

    Raises:
      AttributeError: if the layer is connected to more than one incoming
        layers.
      RuntimeError: if called in Eager mode.
    """
    return self._nested_outputs

  @property
  def output_shape(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring mentions attribute and eager-mode errors and a same-shape
    # requirement, but the code only maps the backend integer-shape function over the outputs.
    # Property returning the static shape of every output, in the same structure as the outputs.
    """Retrieves the output shape(s) of a layer.

    Only applicable if the layer has one output,
    or if all outputs have the same shape.

    Returns:
        Output shape, as an integer shape tuple
        (or list of shape tuples, one tuple per output tensor).

    Raises:
        AttributeError: if the layer has no defined output shape.
        RuntimeError: if called in Eager mode.
    """
    return nest.map_structure(backend.int_shape, self.output)

  def _set_output_names(self):
    # [FUNCTION DOC]
    # HELPER: naming glue used while initializing the graph network.
    # Gives each output a unique name derived from its layer name (made unique when an output
    # layer appears several times, by appending a numeric suffix). Stores the list on the model
    # as its output names. Takes no inputs, returns nothing.
    """Assigns unique names to the Network's outputs.

    Output layers with multiple output tensors would otherwise lead to duplicate
    names in self.output_names.
    """
    uniquified = []
    output_names = set()
    prefix_count = {}
    for layer in self._output_layers:
      proposal = layer.name
      while proposal in output_names:
        existing_count = prefix_count.get(layer.name, 1)
        proposal = '{}_{}'.format(layer.name, existing_count)
        prefix_count[layer.name] = existing_count + 1
      output_names.add(proposal)
      uniquified.append(proposal)
    self.output_names = uniquified

  @property
  def _layer_checkpoint_dependencies(self):
    # [FUNCTION DOC]
    # HELPER: checkpoint dependency bookkeeping for this model's layers.
    # Property returning an ordered mapping from dependency names to layers. Every layer gets a
    # position-based name; layers that currently have weights additionally get a name based on a
    # separate counter of weighted layers, so inserting weightless layers does not break old
    # checkpoints. A layer whose weights cannot yet be read (not built) is treated as weightless.
    """Dictionary of layer dependencies to be included in the checkpoint."""
    weight_layer_index = 0

    dependencies = collections.OrderedDict()
    for layer_index, layer in enumerate(self.layers):
      try:
        if layer.weights:
          # Keep a separate index for layers which have weights. This allows
          # users to insert Layers without weights anywhere in the network
          # without breaking checkpoints.
          dependencies['layer_with_weights-%d' % weight_layer_index] = layer
          weight_layer_index += 1
      except ValueError:
        # The layer might have weights, but may not be built yet. We just treat
        # it as layer without weight.
        pass

      # Even if it doesn't have weights, we should still track everything in
      # case it has/will have Trackable dependencies.
      dependencies['layer-%d' % layer_index] = layer
    return dependencies

  def _trackable_children(self,
                          save_type=trackable.SaveType.CHECKPOINT,
                          **kwargs):
    # [FUNCTION DOC]
    # Override returning the trackable children for saving: the layer checkpoint dependencies,
    # updated with whatever the base model reports. Inputs: save_type (checkpoint by default) and
    # extra keyword arguments forwarded to the base. Entries from the base win on a name clash.
    dependencies = self._layer_checkpoint_dependencies
    dependencies.update(
        super(Functional, self)._trackable_children(save_type, **kwargs))
    return dependencies

  def _lookup_dependency(self, name):
    # [FUNCTION DOC]
    # Finds a checkpoint dependency by name. Input: dependency name string. Returns the matching
    # layer if the name is one of the layer dependencies, otherwise whatever the base
    # implementation returns (possibly nothing).
    layer_dependencies = self._layer_checkpoint_dependencies
    if name in layer_dependencies:
      return layer_dependencies[name]
    return super(Functional, self)._lookup_dependency(name)

  def _handle_deferred_layer_dependencies(self, layers):
    # [FUNCTION DOC]
    # HELPER: checkpoint restore glue for layers added after initialization.
    # Input: an iterable of layers. For each one that is currently a layer dependency, it asks
    # the base class to resolve any deferred (pending restore) dependency registered under that
    # dependency name. Returns nothing.
    """Handles layer checkpoint dependencies that are added after init."""
    layer_checkpoint_dependencies = self._layer_checkpoint_dependencies
    layer_to_name = {v: k for k, v in layer_checkpoint_dependencies.items()}
    for layer in layers:
      if layer in layer_to_name:
        self._handle_deferred_dependencies(name=layer_to_name[layer],
                                           trackable=layer)

  @property
  def _should_compute_mask(self):
    # [FUNCTION DOC]
    # Property that always reports true: a functional model always computes masks.
    return True

  def compute_mask(self, inputs, mask):
    # [FUNCTION DOC]
    # KNOWN ISSUE: per the existing TODO, calling this alone re-runs every layer, duplicating
    # updates and losses in graph mode.
    # Inputs: inputs (tensor or nested structure) and mask (matching masks or nothing). Runs the
    # whole internal graph and returns, in the output structure, each output's mask attribute
    # (nothing where absent).
    # TODO(omalleyt): b/123540974 This function is not really safe to call
    # by itself because it will duplicate any updates and losses in graph
    # mode by `call`ing the Layers again.
    output_tensors = self._run_internal_graph(inputs, mask=mask)
    return nest.map_structure(lambda t: getattr(t, '_keras_mask', None),
                              output_tensors)

  @doc_controls.do_not_doc_inheritable
  def call(self, inputs, training=None, mask=None):
    # [FUNCTION DOC]
    # Forward pass: reapplies every layer of the graph to the new inputs. Inputs: inputs (tensor
    # or nested structure, or a mapping keyed by input names), training (optional boolean or
    # boolean scalar tensor), mask (optional mask or masks). Returns the outputs in the same
    # structure as the model outputs. Side effects: calls the layers (which may create ops,
    # updates and losses).
    """Calls the model on new inputs.

    In this case `call` just reapplies
    all ops in the graph to the new inputs
    (e.g. build a new computational graph from the provided inputs).

    Args:
        inputs: A tensor or list of tensors.
        training: Boolean or boolean scalar tensor, indicating whether to run
          the `Network` in training mode or inference mode.
        mask: A mask or list of masks. A mask can be
            either a tensor or None (no mask).

    Returns:
        A tensor if there is a single output, or
        a list of tensors if there are more than one outputs.
    """
    return self._run_internal_graph(
        inputs, training=training, mask=mask)

  def compute_output_shape(self, input_shape):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the output shapes are only computed inside the branch taken when the graph has
    # more than one depth level; for a graph of only input layers the final return refers to an
    # unassigned name and fails. Also, if converting shapes to the cache key raises (unknown
    # dimensions), the later cache store uses an unassigned key and fails.
    # Computes output shapes without running data. Input: shape or nested structure of shapes
    # (tuples or tensor shapes), one per model input. Returns shapes (as tensor shapes) in the
    # output structure. Raises a value error if the number of input shapes differs from the
    # number of input layers. Side effect: stores the result in the output shape cache keyed by
    # the input shapes. Layers are visited by decreasing depth, each layer's own shape function
    # is called on the shapes flowing from its inputs.
    # Convert any shapes in tuple format to TensorShapes.
    input_shape = tf_utils.convert_shapes(input_shape, to_tuples=False)

    if len(nest.flatten(input_shape)) != len(nest.flatten(self._input_layers)):
      raise ValueError('Invalid input_shape argument ' + str(input_shape) +
                       ': model has ' + str(len(self._input_layers)) +
                       ' tensor inputs.')

    # Use the tuple of TensorShape as the cache key, since tuple is hashable
    # and can be used as hash key.
    try:
      cache_key = tuple(tf_utils.convert_shapes(input_shape, to_tuples=True))
      if cache_key in self._output_shape_cache:
        # Cache hit. Return shapes as TensorShapes.
        return self._output_shape_cache[cache_key]
    except ValueError:
      # In case there are unknown TensorShape, eg for sparse tensor input,
      # We skip the caching since the shape is unknown.
      pass

    layers_to_output_shapes = {}
    for layer, shape in zip(self._input_layers, nest.flatten(input_shape)):
      # It's an input layer: then `compute_output_shape` is identity,
      # and there is only one node and one tensor..
      shape_key = layer.name + '_0_0'
      layers_to_output_shapes[shape_key] = shape

    depth_keys = list(self._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    # Iterate over nodes, by depth level.
    if len(depth_keys) > 1:
      for depth in depth_keys:
        nodes = self._nodes_by_depth[depth]
        for node in nodes:
          layer = node.layer
          if layer in self._input_layers:
            # We've already covered the input layers
            # a few lines above.
            continue
          # Get the input shapes for the first argument of the node
          layer_input_shapes = []
          layer_inputs = node.call_args[0]
          for layer_input in nest.flatten(layer_inputs):
            kh = layer_input._keras_history
            input_layer_key = kh.layer.name + '_%s_%s' % (kh.node_index,
                                                          kh.tensor_index)
            layer_input_shapes.append(layers_to_output_shapes[input_layer_key])
          layer_input_shapes = nest.pack_sequence_as(layer_inputs,
                                                     layer_input_shapes)
          # Layers expect shapes to be tuples for `compute_output_shape`.
          layer_input_shapes = tf_utils.convert_shapes(
              layer_input_shapes, to_tuples=True)
          layer_output_shapes = layer.compute_output_shape(layer_input_shapes)
          # Convert back to TensorShapes.
          layer_output_shapes = tf_utils.convert_shapes(
              layer_output_shapes, to_tuples=False)

          node_index = layer._inbound_nodes.index(node)  # pylint: disable=protected-access
          for j, shape in enumerate(nest.flatten(layer_output_shapes)):
            shape_key = layer.name + '_%s_%s' % (node_index, j)
            layers_to_output_shapes[shape_key] = shape

      # Read final output shapes from layers_to_output_shapes.
      output_shapes = []
      for i in range(len(self._output_layers)):
        layer, node_index, tensor_index = self._output_coordinates[i]
        shape_key = layer.name + '_%s_%s' % (node_index, tensor_index)
        output_shapes.append(layers_to_output_shapes[shape_key])
      output_shapes = nest.pack_sequence_as(self._nested_outputs, output_shapes)
      # Store in cache.
      self._output_shape_cache[cache_key] = output_shapes

    # Return shapes as TensorShapes.
    return output_shapes

  def _init_set_name(self, name, zero_based=True):
    # [FUNCTION DOC]
    # Override that sets the model name. Inputs: name (string, may be empty or nothing) and
    # zero_based (flag for the numbering of generated names, default true). A non-empty name is
    # used as given. Otherwise a unique name is generated from the class name in snake case,
    # with the base Functional class shown as "model". Stores the name on the instance.
    if not name:
      cls_name = self.__class__.__name__
      if self.__class__ == Functional:
        # Hide the functional class name from user, since its not a public
        # visible class. Use "Model" instead,
        cls_name = 'Model'
      self._name = backend.unique_object_name(
          generic_utils.to_snake_case(cls_name),
          zero_based=zero_based)
    else:
      self._name = name

  def _run_internal_graph(self, inputs, training=None, mask=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the training argument is accepted but never used or forwarded to the layers
    # here, contrary to what callers of the forward pass assume; training handling is left to
    # the layers' own call context. Also the final check on computable outputs is an assertion.
    # Core graph executor. Inputs: inputs (tensor, nested structure or mapping by input names),
    # training (unused), mask (optional matching masks). Flattens inputs to the reference input
    # order, attaches masks to them, conforms each to its reference input (shape and dtype),
    # then walks nodes from the greatest depth down, skipping input nodes and nodes whose inputs
    # are not yet available, calling each node's layer on mapped arguments and storing outputs
    # with a remaining-use count. Finally collects each model output and packs them in the output
    # structure, which is returned. Side effects: sets the mask attribute on input tensors and
    # invokes the layers. Raises an assertion error if an output could not be computed.
    """Computes output tensors for new inputs.
 output tensors for new inputs.

    # Note:
        - Can be run on non-Keras tensors.

    Args:
        inputs: Tensor or nested structure of Tensors.
        training: Boolean learning phase.
        mask: (Optional) Tensor or nested structure of Tensors.

    Returns:
        output_tensors
    """
    inputs = self._flatten_to_reference_inputs(inputs)
    if mask is None:
      masks = [None] * len(inputs)
    else:
      masks = self._flatten_to_reference_inputs(mask)
    for input_t, mask in zip(inputs, masks):
      input_t._keras_mask = mask

    # Dictionary mapping reference tensors to computed tensors.
    tensor_dict = {}
    tensor_usage_count = self._tensor_usage_count
    for x, y in zip(self.inputs, inputs):
      y = self._conform_to_reference_input(y, ref_input=x)
      x_id = str(id(x))
      tensor_dict[x_id] = [y] * tensor_usage_count[x_id]

    nodes_by_depth = self._nodes_by_depth
    depth_keys = list(nodes_by_depth.keys())
    depth_keys.sort(reverse=True)

    for depth in depth_keys:
      nodes = nodes_by_depth[depth]
      for node in nodes:
        if node.is_input:
          continue  # Input tensors already exist.

        if any(t_id not in tensor_dict for t_id in node.flat_input_ids):
          continue  # Node is not computable, try skipping.

        args, kwargs = node.map_arguments(tensor_dict)
        outputs = node.layer(*args, **kwargs)

        # Update tensor_dict.
        for x_id, y in zip(node.flat_output_ids, nest.flatten(outputs)):
          tensor_dict[x_id] = [y] * tensor_usage_count[x_id]

    output_tensors = []
    for x in self.outputs:
      x_id = str(id(x))
      assert x_id in tensor_dict, 'Could not compute output ' + str(x)
      output_tensors.append(tensor_dict[x_id].pop())

    return nest.pack_sequence_as(self._nested_outputs, output_tensors)

  def _flatten_to_reference_inputs(self, tensors):
    # [FUNCTION DOC]
    # HELPER: input ordering for the graph executor.
    # Input: a tensor, nested structure, or mapping. If it is a mapping and mapping-by-name is
    # enabled, values are picked in the order of the model's input names (mapping keys in sorted
    # order when the model was built from a mapping, otherwise layer names), warning if the
    # mapping has more entries than inputs; if some name is missing it falls back to flattening
    # the mapping in its own order. Otherwise it simply flattens. Returns a flat list.
    """Maps `tensors` to their respective `keras.Input`."""
    if self._enable_dict_to_input_mapping and isinstance(tensors, dict):
      ref_inputs = self._nested_inputs
      if not nest.is_nested(ref_inputs):
        ref_inputs = [self._nested_inputs]
      if isinstance(ref_inputs, dict):
        # In the case that the graph is constructed with dict input tensors,
        # We will use the original dict key to map with the keys in the input
        # data. Note that the model.inputs is using nest.flatten to process the
        # input tensors, which means the dict input tensors are ordered by their
        # keys.
        ref_input_names = sorted(ref_inputs.keys())
      else:
        ref_input_names = [inp._keras_history.layer.name for inp in ref_inputs]

      # Raise an warning if there are more input data comparing to input tensor
      if len(tensors) > len(ref_input_names):
        warnings.warn(
            'Input dict contained keys {} which did not match any model input. '
            'They will be ignored by the model.'.format(
                [n for n in tensors.keys() if n not in ref_input_names])
            )

      try:
        # Flatten in the order `Input`s were passed during Model construction.
        return [tensors[n] for n in ref_input_names]
      except KeyError:
        # TODO(b/151582614)
        return nest.flatten(tensors)

    # Otherwise both self.inputs and tensors will already be in same order.
    return nest.flatten(tensors)

  def _conform_to_reference_input(self, tensor, ref_input):
    # [FUNCTION DOC]
    # HELPER: input adaptation for the graph executor.
    # Inputs: tensor (a dense tensor, extension-type value, or other) and ref_input (the model's
    # reference input). For dense tensors: squeezes a trailing size-1 axis, or adds one, when
    # ranks differ by one that way; restores layer history; in graph mode merges the static shape
    # with the reference (logging a warning if incompatible); casts to the reference dtype. For
    # extension types it casts to the reference dtype unless that is the variant type. Other
    # values pass through. Returns the adapted value.
    """Set shape and dtype based on `keras.Input`s."""
    if isinstance(tensor, tensor_lib.Tensor):
      # Allow (None,) and (None, 1) Tensors to be passed interchangeably. Use
      # the shape specified by the `keras.Input`.
      t_shape = tensor.shape
      t_rank = t_shape.rank
      ref_shape = ref_input.shape
      ref_rank = ref_shape.rank
      keras_history = getattr(tensor, '_keras_history', None)
      if t_rank is not None and ref_rank is not None:
        # Should squeeze last dimension.
        # True if tensor is (BATCH, ..., 1) and reference is (BATCH, ...).
        if (t_rank == ref_rank + 1 and t_shape[-1] == 1):
          tensor = array_ops.squeeze_v2(tensor, axis=-1)
        # Should expand last_dimension.
        # True if tensor is (BATCH, ...) and reference is (BATCH, ..., 1).
        elif (t_rank == ref_rank - 1 and ref_shape[-1] == 1):
          tensor = array_ops.expand_dims_v2(tensor, axis=-1)
      if keras_history is not None:  # Restore keras history.
        tensor._keras_history = keras_history

      # Add shape hints to Tensors that may have None shape dims but have shapes
      # defined by the `keras.Input` (not applicable in eager mode).
      if not context.executing_eagerly():
        try:
          tensor.set_shape(tensor.shape.merge_with(ref_input.shape))
        except ValueError:
          logging.warning(
              'Model was constructed with shape {} for input {}, but it was '
              'called on an input with incompatible shape {}.'.format(
                  ref_input.shape, ref_input, tensor.shape))

      # Dtype casting.
      tensor = math_ops.cast(tensor, dtype=ref_input.dtype)
    elif tf_utils.is_extension_type(tensor):
      # Dtype casting (If the extension type has a non-variant dtype and
      # supports being cast)
      ref_input_dtype = getattr(ref_input, 'dtype', None)
      if ref_input_dtype is not None and ref_input_dtype != dtypes.variant:
        tensor = math_ops.cast(tensor, dtype=ref_input_dtype)

    return tensor

  def get_config(self):
    # [FUNCTION DOC]
    # Returns a deep copy of the serializable network configuration (name, layer configs,
    # input and output layer references) built by the module-level helper. No inputs.
    return copy.deepcopy(get_network_config(self))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    # [FUNCTION DOC]
    # Class method rebuilding a model from a configuration mapping. Inputs: config (mapping as
    # produced by get_config) and custom_objects (optional mapping from names to custom classes
    # or functions for deserialization). Within a shared-object loading scope it reconstructs the
    # layers and input and output tensors, creates the model with the config's name, then
    # reconnects ancillary layers not reachable from the graph. Returns the model. Raises value
    # errors for malformed configs.
    """Instantiates a Model from its config (output of `get_config()`).

    Args:
        config: Model config dictionary.
        custom_objects: Optional dictionary mapping names
            (strings) to custom classes or functions to be
            considered during deserialization.

    Returns:
        A model instance.

    Raises:
        ValueError: In case of improperly formatted config dict.
    """
    with generic_utils.SharedObjectLoadingScope():
      input_tensors, output_tensors, created_layers = reconstruct_from_config(
          config, custom_objects)
      model = cls(inputs=input_tensors, outputs=output_tensors,
                  name=config.get('name'))
      connect_ancillary_layers(model, created_layers)
      return model

  def _validate_graph_inputs_and_outputs(self):
    # [FUNCTION DOC]
    # HELPER: validation step of graph network initialization.
    # Checks that inputs are unique (value error if repeated), carry layer history (value error
    # if they do not come from the Input function), warns if an input is the output of a non-input
    # layer, raises a value error if static batch sizes of input layers disagree, and raises a
    # value error if an output lacks layer history. Returns nothing.
    """Validates the inputs and outputs of a Graph Network."""
    # Check for redundancy in inputs.
    if len({id(i) for i in self.inputs}) != len(self.inputs):
      raise ValueError('The list of inputs passed to the model '
                       'is redundant. '
                       'All inputs should only appear once.'
                       ' Found: ' + str(self.inputs))

    for x in self.inputs:
      # Check that x has appropriate `_keras_history` metadata.
      if not hasattr(x, '_keras_history'):
        cls_name = self.__class__.__name__
        raise ValueError('Input tensors to a ' + cls_name + ' ' +
                         'must come from `tf.keras.Input`. '
                         'Received: ' + str(x) +
                         ' (missing previous layer metadata).')
      # Check that x is an input tensor.
      # pylint: disable=protected-access
      layer = x._keras_history.layer
      if len(layer._inbound_nodes) > 1 or (
          layer._inbound_nodes and not layer._inbound_nodes[0].is_input):
        cls_name = self.__class__.__name__
        logging.warning(cls_name + ' model inputs must come from '
                        '`tf.keras.Input` (thus holding past layer metadata), '
                        'they cannot be the output of '
                        'a previous non-Input layer. '
                        'Here, a tensor specified as '
                        'input to "' + self.name + '" was not an Input tensor, '
                        'it was generated by layer ' + layer.name + '.\n'
                        'Note that input tensors are '
                        'instantiated via `tensor = tf.keras.Input(shape)`.\n'
                        'The tensor that caused the issue was: ' + str(x.name))

    # Check compatibility of batch sizes of Input Layers.
    input_batch_sizes = [
        training_utils.get_static_batch_size(x._keras_history.layer)
        for x in self.inputs
    ]
    consistent_batch_size = None
    for batch_size in input_batch_sizes:
      if batch_size is not None:
        if (consistent_batch_size is not None and
            batch_size != consistent_batch_size):
          raise ValueError('The specified batch sizes of the Input Layers'
                           ' are incompatible. Found batch sizes: {}'.format(
                               input_batch_sizes))
        consistent_batch_size = batch_size

    for x in self.outputs:
      if not hasattr(x, '_keras_history'):
        cls_name = self.__class__.__name__
        raise ValueError('Output tensors of a ' + cls_name + ' model must be '
                         'the output of a TensorFlow `Layer` '
                         '(thus holding past layer metadata). Found: ' + str(x))

  def _insert_layers(self, layers, relevant_nodes=None):
    # [FUNCTION DOC]
    # Adds layers (and their graph nodes) to an already built graph network, so they take part in
    # the forward pass and configuration but are not added to the outputs.
    # Inputs: layers (nested structure of layers, each reachable from this model's inputs) and
    # relevant_nodes (optional list of nodes to consider; defaults to all inbound nodes of the
    # layers). Behavior: rejects legacy layers, assigns each new node a depth one level shallower
    # than its shallowest input, deferring nodes whose inputs are unprocessed, registers new node
    # keys and nodes by depth, appends unseen layers to the tracked layers with their call
    # signatures, resolves deferred checkpoint dependencies, and recomputes tensor usage counts.
    # Raises a value error if layers cannot be placed (missing dependencies, detected after a
    # large iteration count). Side effects: mutates the model's node and layer bookkeeping.
    """Inserts Layers into the Network after Network creation.

    This is only valid for Keras Graph Networks.  Layers added via this function
    will be included in the `call` computation and `get_config` of this Network.
    They will not be added to the Network's outputs.


    Args:
      layers: Arbitrary nested structure of Layers. Layers must be reachable
        from one or more of the `keras.Input` Tensors that correspond to this
        Network's inputs.
      relevant_nodes: Nodes from the Layers that should be considered part of
        this Network. If `None`, all Nodes will be considered part of this
        Network.

    Raises:
      ValueError: If the layers depend on `Input`s not found in this Model.
    """
    layers = nest.flatten(layers)
    tf_utils.assert_no_legacy_layers(layers)
    node_to_depth = {}
    for depth, nodes in self._nodes_by_depth.items():
      node_to_depth.update({node: depth for node in nodes})
    # The nodes of these Layers that are relevant to this Network. If not
    # provided, assume all Nodes are relevant
    if not relevant_nodes:
      relevant_nodes = nest.flatten([layer._inbound_nodes for layer in layers])
    network_nodes = set(relevant_nodes + list(node_to_depth.keys()))

    def _get_min_depth(node):
      # [FUNCTION DOC]
      # HELPER: depth computation for the layer insertion routine.
      # Input: a node. Returns the depth for the node: one less than the minimum depth among its
      # inbound nodes already placed (ignoring inbound nodes outside the network), or nothing if
      # some relevant inbound node has not been placed yet. No side effects.
      """Gets the minimum depth at which node can be computed."""
      min_depth = 0
      for layer, node_id, _, _ in node.iterate_inbound():
        inbound_node = layer._inbound_nodes[node_id]
        if inbound_node in node_to_depth:
          min_depth = min(min_depth, node_to_depth[inbound_node])
        elif inbound_node not in network_nodes:
          continue
        else:
          # Previous relevant nodes haven't been processed yet.
          return None
      # New node is one shallower than its shallowest input.
      return min_depth - 1

    # Insert nodes into `_nodes_by_depth` and other node attrs.
    unprocessed_nodes = copy.copy(relevant_nodes)
    i = 0
    while unprocessed_nodes:
      i += 1
      # Do a sanity check. This can occur if `Input`s from outside this Model
      # are being relied on.
      if i > 10000:
        raise ValueError('Layers could not be added due to missing '
                         'dependencies.')

      node = unprocessed_nodes.pop(0)
      depth = _get_min_depth(node)
      if depth is None:  # Defer until inbound nodes are processed.
        unprocessed_nodes.append(node)
        continue
      node_key = _make_node_key(node.layer.name,
                                node.layer._inbound_nodes.index(node))
      if node_key not in self._network_nodes:
        node_to_depth[node] = depth
        self._network_nodes.add(node_key)
        self._nodes_by_depth[depth].append(node)

    # Insert layers and update other layer attrs.
    layer_set = set(self._self_tracked_trackables)
    deferred_layers = []
    for layer in layers:
      if layer not in layer_set:
        self._self_tracked_trackables.append(layer)
        deferred_layers.append(layer)
        self._layer_call_argspecs[layer] = tf_inspect.getfullargspec(layer.call)
        layer_set.add(layer)
    self._handle_deferred_layer_dependencies(deferred_layers)

    self._compute_tensor_usage_count()

  def _compute_tensor_usage_count(self):
    # [FUNCTION DOC]
    # HELPER: memory optimization bookkeeping for the graph executor.
    # Counts, for every tensor, how many times it is consumed by computable nodes (skipping the
    # deepest depth level, which holds the inputs) plus one for each time it is a model output,
    # and stores the counter on the model. Only nodes whose inputs are all available are
    # counted. Takes no inputs and returns nothing.
    """Compute the #. of tensor usages for all the output tensors of layers.

    The computed tensor usage count is saved as `self._tensor_usage_count`. This
    is later used for saving memory in eager computation by releasing
    no-longer-needed tensors as early as possible.
    """
    tensor_usage_count = collections.Counter()
    available_tensors = set(str(id(tensor)) for tensor in self.inputs)

    depth_keys = list(self._nodes_by_depth.keys())
    depth_keys.sort(reverse=True)
    depth_keys = depth_keys[1:]

    for depth in depth_keys:
      for node in self._nodes_by_depth[depth]:
        input_tensors = {
            str(id(tensor)) for tensor in nest.flatten(node.keras_inputs)
        }
        if input_tensors.issubset(available_tensors):
          for tensor in nest.flatten(node.keras_inputs):
            tensor_usage_count[str(id(tensor))] += 1

          for output_tensor in nest.flatten(node.outputs):
            available_tensors.add(str(id(output_tensor)))

    for tensor in self.outputs:
      tensor_usage_count[str(id(tensor))] += 1

    self._tensor_usage_count = tensor_usage_count

  def _assert_weights_created(self):
    # [FUNCTION DOC]
    # Override that intentionally does nothing: a functional model has its weights created at
    # construction, so the base class check is unnecessary.
    # Override the implementation in Model.
    # The Functional model should always have weight created already.
    return

  def _graph_network_add_loss(self, symbolic_loss):
    # [FUNCTION DOC]
    # HELPER: support for adding a loss defined on symbolic tensors to a built graph network.
    # Input: symbolic_loss (symbolic tensor). Maps the sub-graph from the model inputs to the
    # loss, wraps the loss in a new loss-adding layer (conditional on nothing) so it is keyed on
    # inputs as distribution strategies require, and inserts the new nodes and layers into the
    # model. Returns nothing; mutates the model.
    new_nodes, new_layers = _map_subgraph_network(self.inputs, [symbolic_loss])
    # Losses must be keyed on inputs no matter what in order to be supported in
    # DistributionStrategy.
    add_loss_layer = base_layer.AddLoss(
        unconditional=False, dtype=symbolic_loss.dtype)
    add_loss_layer(symbolic_loss)
    new_nodes.extend(add_loss_layer.inbound_nodes)
    new_layers.append(add_loss_layer)
    self._insert_layers(new_layers, new_nodes)

  def _graph_network_add_metric(self, value, aggregation, name):
    # [FUNCTION DOC]
    # HELPER: support for adding a metric defined on symbolic tensors to a built graph network.
    # Inputs: value (symbolic tensor), aggregation (how values are aggregated across batches) and
    # name (metric name). Maps the sub-graph from model inputs to the value, wraps it in a new
    # metric-adding layer, and inserts the new nodes and layers into the model. Returns nothing.
    new_nodes, new_layers = _map_subgraph_network(self.inputs, [value])
    add_metric_layer = base_layer.AddMetric(
        aggregation, name, dtype=value.dtype)
    add_metric_layer(value)
    new_nodes.extend(add_metric_layer.inbound_nodes)
    new_layers.append(add_metric_layer)
    self._insert_layers(new_layers, new_nodes)

  @property
  def _trackable_saved_model_saver(self):
    # [FUNCTION DOC]
    # HELPER: saved-model serialization glue. Property returning the network-specific saver
    # object wrapping this model.
    return network_serialization.NetworkSavedModelSaver(self)

  def _get_save_spec(self, dynamic_batch=True):
    # [FUNCTION DOC]
    # Override producing the input signature used for saving. Input: dynamic_batch (flag,
    # default true, whether to make the batch dimension unknown). If the model has an explicit
    # input shape (the default when the attribute is missing) the flag is forced off so the batch
    # size set by the input layer is kept. Returns what the base class computes.
    if getattr(self, '_has_explicit_input_shape', True):
      # Functional models and Sequential models that have an explicit input
      # shape should use the batch size set by the input layer.
      dynamic_batch = False
    return super(Functional, self)._get_save_spec(dynamic_batch)


def _make_node_key(layer_name, node_index):
  # [FUNCTION DOC]
  # HELPER: builds the string key identifying a node in serialization and node bookkeeping.
  # Inputs: layer_name (string) and node_index (integer). Returns the name joined with a fixed
  # marker and the index as text.
  return layer_name + '_ib-' + str(node_index)


def _map_graph_network(inputs, outputs):
  # [FUNCTION DOC]
  # HELPER: topology analysis used by graph network initialization.
  # Inputs: inputs and outputs (lists of symbolic tensors with layer history). Topologically
  # sorts nodes from the outputs, assigns each node and layer a depth (the greatest distance to
  # an output, with shared layers taking their deepest depth), adds input layers not connected to
  # any output at depth zero, orders layers by depth then traversal order, and verifies that every
  # node input is computable from the given inputs and that layer names are unique.
  # Returns four values: the set of node keys, a mapping of depth to nodes, the ordered list of
  # layers and a mapping of depth to layers. Raises a value error for a disconnected graph or a
  # repeated layer name. No other side effects.
  """Validates a network's topology and gather its layers and nodes.

  Args:
    inputs: List of input tensors.
    outputs: List of outputs tensors.

  Returns:
    A tuple `(nodes, nodes_by_depth, layers, layers_by_depth)`.
    - nodes: list of Node instances.
    - nodes_by_depth: dict mapping ints (depth) to lists of node instances.
    - layers: list of Layer instances.
    - layers_by_depth: dict mapping ints (depth) to lists of layer instances.

  Raises:
    ValueError: In case the network is not valid (e.g. disconnected graph).
  """
  # "depth" is number of layers between output Node and the Node.
  # Nodes are ordered from inputs -> outputs.
  nodes_in_decreasing_depth, layer_indices = _build_map(outputs)
  network_nodes = {
      _make_node_key(node.layer.name, node.layer._inbound_nodes.index(node))
      for node in nodes_in_decreasing_depth
  }

  nodes_depths = {}  # dict {node: depth value}
  layers_depths = {}  # dict {layer: depth value}

  for node in reversed(nodes_in_decreasing_depth):
    # If the depth is not set, the node has no outbound nodes (depth 0).
    depth = nodes_depths.setdefault(node, 0)

    # Update the depth of the corresponding layer
    previous_depth = layers_depths.get(node.layer, 0)
    # If we've seen this layer before at a higher depth,
    # we should use that depth instead of the node depth.
    # This is necessary for shared layers that have inputs at different
    # depth levels in the graph.
    depth = max(depth, previous_depth)
    layers_depths[node.layer] = depth
    nodes_depths[node] = depth

    # Update the depth of inbound nodes.
    # The "depth" of a node is the max of the depths
    # of all nodes it is connected to + 1.
    for node_dep in node.parent_nodes:
      previous_depth = nodes_depths.get(node_dep, 0)
      nodes_depths[node_dep] = max(depth + 1, previous_depth)

  # Handle inputs that are not connected to outputs.
  # We do not error out here because the inputs may be used to compute losses
  # and metrics.
  for input_t in inputs:
    input_layer = input_t._keras_history[0]
    if input_layer not in layers_depths:
      layers_depths[input_layer] = 0
      layer_indices[input_layer] = -1
      nodes_depths[input_layer._inbound_nodes[0]] = 0
      network_nodes.add(_make_node_key(input_layer.name, 0))

  # Build a dict {depth: list of nodes with this depth}
  nodes_by_depth = collections.defaultdict(list)
  for node, depth in nodes_depths.items():
    nodes_by_depth[depth].append(node)

  # Build a dict {depth: list of layers with this depth}
  layers_by_depth = collections.defaultdict(list)
  for layer, depth in layers_depths.items():
    layers_by_depth[depth].append(layer)

  # Get sorted list of layer depths.
  depth_keys = list(layers_by_depth.keys())
  depth_keys.sort(reverse=True)

  # Set self.layers ordered by depth.
  layers = []
  for depth in depth_keys:
    layers_for_depth = layers_by_depth[depth]
    # Network.layers needs to have a deterministic order:
    # here we order them by traversal order.
    layers_for_depth.sort(key=lambda x: layer_indices[x])
    layers.extend(layers_for_depth)

  # Get sorted list of node depths.
  depth_keys = list(nodes_by_depth.keys())
  depth_keys.sort(reverse=True)

  # Check that all tensors required are computable.
  # computable_tensors: all tensors in the graph
  # that can be computed from the inputs provided.
  computable_tensors = set()
  for x in inputs:
    computable_tensors.add(id(x))

  layers_with_complete_input = []  # To provide a better error msg.
  for depth in depth_keys:
    for node in nodes_by_depth[depth]:
      layer = node.layer
      if layer and not node.is_input:
        for x in nest.flatten(node.keras_inputs):
          if id(x) not in computable_tensors:
            raise ValueError('Graph disconnected: '
                             'cannot obtain value for tensor ' + str(x) +
                             ' at layer "' + layer.name + '". '
                             'The following previous layers '
                             'were accessed without issue: ' +
                             str(layers_with_complete_input))
        for x in nest.flatten(node.outputs):
          computable_tensors.add(id(x))
        layers_with_complete_input.append(layer.name)

  # Ensure name unicity, which will be crucial for serialization
  # (since serialized nodes refer to layers by their name).
  all_names = [layer.name for layer in layers]
  for name in all_names:
    if all_names.count(name) != 1:
      raise ValueError('The name "' + name + '" is used ' +
                       str(all_names.count(name)) + ' times in the model. '
                       'All layer names should be unique.')
  return network_nodes, nodes_by_depth, layers, layers_by_depth


def _build_map(outputs):
  # [FUNCTION DOC]
  # HELPER: traversal step of the topology analysis.
  # Input: output tensors (any nested structure). Runs a depth-first walk over layer history of
  # each flattened output through the recursive helper. Returns a list of nodes ordered from
  # inputs toward outputs and a mapping from each layer to the index of its first visit.
  """This method topologically sorts nodes in order from inputs to outputs.

  It uses a depth-first search to topologically sort nodes that appear in the
  _keras_history connectivity metadata of `outputs`.

  Args:
    outputs: the output tensors whose _keras_history metadata should be walked.
    This may be an arbitrary nested structure.

  Returns:
    A tuple like (ordered_nodes, layer_to_first_traversal_index)
    ordered_nodes: list of nodes appearing in the keras history, topologically
      sorted from original inputs to the `outputs`.
      (If outputs have different sets of ancestors, the inputs to one output
      may appear after a different output).
    layer_to_first_traversal_index:
      A dict mapping layer to the traversal index in the DFS where it is
      seen. Note: if a layer is shared by several nodes, the dict will only
      store the index corresponding to the *first* time the layer seen.
  """
  finished_nodes = set()
  nodes_in_progress = set()
  nodes_in_decreasing_depth = []  # nodes from inputs -> outputs.
  layer_indices = {}  # layer -> in traversal order.
  for output in nest.flatten(outputs):
    _build_map_helper(output, finished_nodes, nodes_in_progress,
                      nodes_in_decreasing_depth, layer_indices)
  return nodes_in_decreasing_depth, layer_indices


def _build_map_helper(tensor, finished_nodes, nodes_in_progress,
                      nodes_in_decreasing_depth, layer_indices):
  # [FUNCTION DOC]
  # HELPER: recursive depth-first step for the topological sort.
  # KNOWN ISSUE: the name says decreasing depth, but the list is filled from inputs to outputs
  # (increasing distance from inputs).
  # Inputs: a tensor with layer history, the set of finished nodes, the set of nodes in progress,
  # the output list of nodes and the mapping of layer to first traversal index; the last three
  # are mutated. Visits the tensor's node unless done, records the layer's first index, recurses
  # into the node's inputs (non-input nodes), then appends the node. Raises a value error if a
  # cycle is found. Returns nothing.
  """Recursive helper for `_build_map`."""
  layer, node_index, _ = tensor._keras_history  # pylint: disable=protected-access
  node = layer._inbound_nodes[node_index]  # pylint: disable=protected-access

  # Don't repeat work for shared subgraphs
  if node in finished_nodes:
    return

  # Prevent cycles.
  if node in nodes_in_progress:
    raise ValueError('The tensor ' + str(tensor) + ' at layer "' + layer.name +
                     '" is part of a cycle.')

  # Store the traversal order for layer sorting.
  if layer not in layer_indices:
    layer_indices[layer] = len(layer_indices)

  # Propagate to all previous tensors connected to this node.
  nodes_in_progress.add(node)
  if not node.is_input:
    for tensor in node.keras_inputs:
      _build_map_helper(tensor, finished_nodes, nodes_in_progress,
                        nodes_in_decreasing_depth, layer_indices)

  finished_nodes.add(node)
  nodes_in_progress.remove(node)
  nodes_in_decreasing_depth.append(node)


def _map_subgraph_network(inputs, outputs):
  # [FUNCTION DOC]
  # HELPER: sub-graph discovery for adding losses and metrics.
  # Inputs: inputs and outputs (lists of tensors). In graph mode it first creates layer history
  # for the outputs. Then maps the topology between inputs and outputs. Returns a flat list of
  # all nodes and the list of layers. Raises value errors if the topology is invalid.
  """Returns the nodes and layers in the topology from `inputs` to `outputs`.

  Args:
    inputs: List of input tensors.
    outputs: List of output tensors.

  Returns:
    A tuple of List{Node] and List[Layer].
  """
  if not ops.executing_eagerly_outside_functions():
    base_layer_utils.create_keras_history(outputs)
  # Keep only nodes and layers in the topology between inputs and outputs.
  _, nodes_by_depth, layers, _ = _map_graph_network(inputs, outputs)
  return nest.flatten([nodes for nodes in nodes_by_depth.values()]), layers


def _should_skip_first_node(layer):
  # [FUNCTION DOC]
  # HELPER: serialization rule. Input: a layer. Returns true if the layer is a functional model
  # whose first node (linking its input to its output, created when it was built) should be left
  # out of saved configs and ignored on load. If the layer has tracked sub-layers, it also needs
  # the first of them to be an input layer (this excludes Sequential models without an input
  # shape); a functional model without tracked layers always qualifies. Other layers give false.
  """Returns True if the first layer node should not be saved or loaded."""
  # Networks that are constructed with an Input layer/shape start with a
  # pre-existing node linking their input to output. This node is excluded from
  # the network config.
  if layer._self_tracked_trackables:
    return (isinstance(layer, Functional) and
            # Filter out Sequential models without an input shape.
            isinstance(layer._self_tracked_trackables[0],
                       input_layer_module.InputLayer))
  else:
    return isinstance(layer, Functional)


def connect_ancillary_layers(model, created_layers):
  # [FUNCTION DOC]
  # Adds to a model the layers rebuilt from a config that are not part of its graph (for example
  # those created by adding losses). Inputs: model (a functional model) and created_layers
  # (mapping from names to layers). Finds layers not already in the model; if any, collects the
  # relevant nodes of all created layers (skipping the first node where the skip rule applies)
  # and inserts the ancillary layers into the model. Returns the same model; mutates it.
  """Adds layers that are not connected to the outputs to the model."""
  # Layers not connected to outputs, such as those added in `add_loss`.
  ancillary_layers = [
      layer for layer in created_layers.values() if layer not in model.layers
  ]
  if ancillary_layers:
    relevant_nodes = nest.flatten([
        layer.inbound_nodes[1:]
        if _should_skip_first_node(layer) else layer.inbound_nodes
        for layer in created_layers.values()
    ])
    model._insert_layers(ancillary_layers, relevant_nodes)
  return model


def reconstruct_from_config(config, custom_objects=None, created_layers=None):
  # [FUNCTION DOC]
  # Rebuilds the layers and symbolic tensors of a graph network from its configuration.
  # Inputs: config (mapping with layer configs, input layers and output layers), custom_objects
  # (optional mapping from names to custom classes or functions) and created_layers (optional
  # ordered mapping from names to already-created layers, extended in place).
  # Behavior: deserializes every layer not yet created, queues each layer's inbound node data,
  # then repeatedly processes queued nodes in layer order, calling layers on their inputs, until
  # all are processed (a node waits until its inbound nodes exist), keeping a map from config node
  # indices to actual node indices. Finally collects input and output tensors in the config's
  # structure. Returns the input tensors, output tensors and the mapping of created layers.
  # Raises value errors for malformed node data. Side effect: calls layers, creating nodes.
  # KNOWN ISSUE: a malformed config with unsatisfiable dependencies makes the processing loop
  # run forever (the code's own comment warns of this).
  """Reconstructs graph from config object.

  Args:
    config: Dictionary returned from Network.get_config()
    custom_objects: Optional dictionary mapping names (strings) to custom
      classes or functions to be considered during deserialization.
    created_layers: Optional dictionary mapping names to Layer objects. Any
      layer not in this dictionary will be created and added to the dict.
      This function will add new nodes to all layers (excluding InputLayers),
      instead of re-using pre-existing nodes in the layers.

  Returns:
    Tuple of (input tensors, output tensors, dictionary of created layers)
  """
  # Layer instances created during the graph reconstruction process.
  created_layers = created_layers or collections.OrderedDict()

  # Maps input data (tuple of inbound layer name, node index) from the config
  # to node indices in the newly generated model. The node indices may be
  # different if the layers have already been called previously.
  node_index_map = {}
  node_count_by_layer = {}

  # Dictionary mapping layer instances to
  # node data that specifies a layer call.
  # It acts as a queue that maintains any unprocessed
  # layer call until it becomes possible to process it
  # (i.e. until the input tensors to the call all exist).
  unprocessed_nodes = {}

  def add_unprocessed_node(layer, node_data):
    # [FUNCTION DOC]
    # HELPER: queue management for reconstruct_from_config. Appends the node data to the pending
    # list for the layer, creating the list if needed. Returns nothing.
    if layer not in unprocessed_nodes:
      unprocessed_nodes[layer] = [node_data]
    else:
      unprocessed_nodes[layer].append(node_data)

  def get_node_index(layer, config_node_index):
    # [FUNCTION DOC]
    # HELPER: node index translation for reconstruct_from_config. Inputs: a layer and the node
    # index recorded in the config. Returns zero for input layers, otherwise the actual index
    # from the map of processed nodes, or nothing if that node is not processed yet.
    """Returns node index in layer (might differ from config_node_index)."""
    if isinstance(layer, input_layer_module.InputLayer):
      return 0
    return node_index_map.get((layer.name, config_node_index), None)

  def _deserialize_keras_tensors(kwargs, layer_map):
    # [FUNCTION DOC]
    # HELPER: keyword argument restoration for reconstruct_from_config.
    # Inputs: kwargs (serialized call keyword arguments, nested) and layer_map (mapping from
    # layer names to created layers). Wraps inner node data, then replaces every encoded tensor
    # reference with the actual tensor. Returns the restored structure. Raises an index error
    # (used as a signal to retry later) if a referenced node is not processed yet.
    """Deserializes Keras Tensors passed to `call`.."""

    def _deserialize_keras_tensor(t):
      # [FUNCTION DOC]
      # HELPER: single value restoration. If the value is an encoded tensor reference (layer name,
      # node index, tensor index), looks up the layer and the real node index and returns that
      # node's output tensor; raises an index error if the node is not yet processed. Any other
      # value is returned unchanged.
      """Deserializes a single Keras Tensor passed to `call`."""
      if isinstance(t, tf_utils.ListWrapper):
        t = t.as_list()
        layer_name = t[0]
        node_index = t[1]
        tensor_index = t[2]

        layer = layer_map[layer_name]
        new_node_index = get_node_index(layer, node_index)
        if new_node_index is None:
          # The inbound node may not have been processed yet,
          # (This can happen e.g. if it depends on a different set
          # of inputs than those that have been processed already).
          # raise an IndexError so that the current node puts itself
          # back on the unprocessed queue.
          # Caution: This may lead to infinite loops for malformed
          # network configurations! (or when there is a bug in
          # the network config loading code).
          raise IndexError
        node = layer._inbound_nodes[new_node_index]
        return nest.flatten(node.outputs)[tensor_index]
      return t

    kwargs = tf_utils.convert_inner_node_data(kwargs, wrap=True)
    return nest.map_structure(_deserialize_keras_tensor, kwargs)

  def process_node(layer, node_data):
    # [FUNCTION DOC]
    # HELPER: node reconstruction for reconstruct_from_config.
    # KNOWN ISSUE: the final check that the assembled inputs are not null can never fail, and
    # the keyword arguments of only the last inbound entry are used for the call.
    # Inputs: layer and node_data (nested structure of encoded inbound references). For each
    # reference it reads the inbound layer, node and tensor index plus optional call keyword
    # arguments (a malformed length raises a value error). If an inbound node or a keyword tensor
    # is not yet available, the node is re-queued and nothing else happens. Constants are passed
    # through. Otherwise the layer is called on the assembled inputs, creating a node, and the
    # node index map and per-layer node count are updated. Returns nothing.
    """Deserialize a node.

    Args:
        layer: layer instance.
        node_data: Nested structure of `ListWrapper`.

    Raises:
        ValueError: In case of improperly formatted `node_data`.
    """
    input_tensors = []
    for input_data in nest.flatten(node_data):
      input_data = input_data.as_list()
      inbound_layer_name = input_data[0]
      inbound_node_index = input_data[1]
      inbound_tensor_index = input_data[2]
      if len(input_data) == 3:
        kwargs = {}
      elif len(input_data) == 4:
        kwargs = input_data[3]
        try:
          kwargs = _deserialize_keras_tensors(kwargs, created_layers)
        except IndexError:
          # Happens if keras tensors in kwargs are still unprocessed
          add_unprocessed_node(layer, node_data)
          return
      else:
        raise ValueError('Improperly formatted model config.')

      if inbound_layer_name != node_module._CONSTANT_VALUE:
        inbound_layer = created_layers[inbound_layer_name]
        inbound_node_index = get_node_index(inbound_layer, inbound_node_index)

        if inbound_node_index is None:
          add_unprocessed_node(layer, node_data)
          return
        inbound_node = inbound_layer._inbound_nodes[inbound_node_index]
        input_tensors.append(
            nest.flatten(inbound_node.outputs)[inbound_tensor_index])
      else:
        # We received a constant w/ no Keras history attached
        input_tensors.append(inbound_tensor_index)
    input_tensors = nest.pack_sequence_as(node_data, input_tensors)
    # Call layer on its inputs, thus creating the node
    # and building the layer if needed.
    if input_tensors is not None:
      if not layer._preserve_input_structure_in_config:
        input_tensors = (
            base_layer_utils.unnest_if_single_tensor(input_tensors))
      output_tensors = layer(input_tensors, **kwargs)

      # Update node index map.
      output_index = nest.flatten(output_tensors)[0]._keras_history.node_index
      node_index_map[(layer.name, node_count_by_layer[layer])] = output_index
      node_count_by_layer[layer] += 1

  def process_layer(layer_data):
    # [FUNCTION DOC]
    # HELPER: layer creation step of reconstruct_from_config.
    # Input: a layer config mapping with its name and inbound node data. Reuses the created layer
    # of that name or deserializes a new one (using custom objects) and records it. Initializes
    # its node count (one if the first node is to be skipped), converts inbound node data and
    # queues each entry for later processing. Does not call the layer. Returns nothing.
    """Deserializes a layer, then call it on appropriate inputs.

    Args:
        layer_data: layer config dict.

    Raises:
        ValueError: In case of improperly formatted `layer_data` dict.
    """
    layer_name = layer_data['name']

    if layer_name in created_layers:
      layer = created_layers[layer_name]
    else:
      # Instantiate layer.
      from tensorflow.python.keras.layers import deserialize as deserialize_layer  # pylint: disable=g-import-not-at-top

      layer = deserialize_layer(layer_data, custom_objects=custom_objects)
      created_layers[layer_name] = layer

    node_count_by_layer[layer] = int(_should_skip_first_node(layer))

    # Gather layer inputs and convert to `ListWrapper` objects.
    inbound_nodes_data = layer_data['inbound_nodes']
    inbound_nodes_data = tf_utils.convert_inner_node_data(
        inbound_nodes_data, wrap=True)
    for node_data in inbound_nodes_data:
      # We don't process nodes (i.e. make layer calls)
      # on the fly because the inbound node may not yet exist,
      # in case of layer shared at different topological depths
      # (e.g. a model such as A(B(A(B(x)))))
      add_unprocessed_node(layer, node_data)

  # First, we create all layers and enqueue nodes to be processed
  for layer_data in config['layers']:
    process_layer(layer_data)
  # Then we process nodes in order of layer depth.
  # Nodes that cannot yet be processed (if the inbound node
  # does not yet exist) are re-enqueued, and the process
  # is repeated until all nodes are processed.
  while unprocessed_nodes:
    for layer_data in config['layers']:
      layer = created_layers[layer_data['name']]
      if layer in unprocessed_nodes:
        for node_data in unprocessed_nodes.pop(layer):
          process_node(layer, node_data)

  input_tensors = []
  output_tensors = []

  input_layers = tf_utils.convert_inner_node_data(
      config['input_layers'], wrap=True)
  for layer_data in nest.flatten(input_layers):
    layer_name, node_index, tensor_index = layer_data.as_list()
    assert layer_name in created_layers
    layer = created_layers[layer_name]
    node_index = get_node_index(layer, node_index)
    layer_output_tensors = layer._inbound_nodes[node_index].output_tensors
    input_tensors.append(nest.flatten(layer_output_tensors)[tensor_index])

  output_layers = tf_utils.convert_inner_node_data(
      config['output_layers'], wrap=True)
  for layer_data in nest.flatten(output_layers):
    layer_name, node_index, tensor_index = layer_data.as_list()
    assert layer_name in created_layers
    layer = created_layers[layer_name]
    node_index = get_node_index(layer, node_index)
    layer_output_tensors = layer._inbound_nodes[node_index].output_tensors
    output_tensors.append(nest.flatten(layer_output_tensors)[tensor_index])

  input_tensors = nest.pack_sequence_as(input_layers, input_tensors)
  output_tensors = nest.pack_sequence_as(output_layers, output_tensors)
  return input_tensors, output_tensors, created_layers


def get_network_config(network, serialize_layer_fn=None):
  # [FUNCTION DOC]
  # Builds the serializable configuration of a graph network.
  # Inputs: network (a functional model) and serialize_layer_fn (optional function turning a layer
  # into a config mapping; defaults to the generic Keras object serializer).
  # Behavior: renumbers each layer's nodes so only nodes belonging to the network are kept (the
  # first node is reserved when the skip rule applies); within a shared-object saving scope, for
  # each layer in order serializes the layer, sets its name and attaches its serialized inbound
  # nodes (input nodes excluded); then records the input and output layers as lists of layer
  # name, new node index and tensor index in the same structure as the model inputs/outputs
  # (single values are wrapped in a list), skipping coordinates outside the network.
  # Returns a mapping with name, layers, input layers and output layers. No other side effects.
  """Builds the config, which consists of the node graph and serialized layers.

  Args:
    network: A Network object.
    serialize_layer_fn: Function used to serialize layers.

  Returns:
    Config dictionary.
  """
  serialize_layer_fn = (
      serialize_layer_fn or generic_utils.serialize_keras_object)
  config = {
      'name': network.name,
  }
  node_conversion_map = {}
  for layer in network.layers:
    kept_nodes = 1 if _should_skip_first_node(layer) else 0
    for original_node_index, node in enumerate(layer._inbound_nodes):
      node_key = _make_node_key(layer.name, original_node_index)
      if node_key in network._network_nodes:
        node_conversion_map[node_key] = kept_nodes
        kept_nodes += 1
  layer_configs = []

  with generic_utils.SharedObjectSavingScope():
    for layer in network.layers:  # From the earliest layers on.
      filtered_inbound_nodes = []
      for original_node_index, node in enumerate(layer._inbound_nodes):
        node_key = _make_node_key(layer.name, original_node_index)
        if node_key in network._network_nodes and not node.is_input:
          # The node is relevant to the model:
          # add to filtered_inbound_nodes.
          node_data = node.serialize(_make_node_key, node_conversion_map)
          filtered_inbound_nodes.append(node_data)

      layer_config = serialize_layer_fn(layer)
      layer_config['name'] = layer.name
      layer_config['inbound_nodes'] = filtered_inbound_nodes
      layer_configs.append(layer_config)
    config['layers'] = layer_configs

  # Gather info about inputs and outputs.
  model_inputs = []
  for i in range(len(network._input_layers)):
    layer, node_index, tensor_index = network._input_coordinates[i]
    node_key = _make_node_key(layer.name, node_index)
    if node_key not in network._network_nodes:
      continue
    new_node_index = node_conversion_map[node_key]
    model_inputs.append(
        tf_utils.ListWrapper([layer.name, new_node_index, tensor_index]))
  model_inputs = nest.pack_sequence_as(network._nested_inputs, model_inputs)
  # Preserve external Keras compat for Models with single input.
  if not nest.is_nested(model_inputs):
    model_inputs = [model_inputs]
  model_inputs = tf_utils.convert_inner_node_data(model_inputs)
  config['input_layers'] = model_inputs

  model_outputs = []
  for i in range(len(network._output_layers)):
    layer, node_index, tensor_index = network._output_coordinates[i]
    node_key = _make_node_key(layer.name, node_index)
    if node_key not in network._network_nodes:
      continue
    new_node_index = node_conversion_map[node_key]
    model_outputs.append(
        tf_utils.ListWrapper([layer.name, new_node_index, tensor_index]))
  model_outputs = nest.pack_sequence_as(network._nested_outputs, model_outputs)
  # Preserve external Keras compat for Models with single output.
  if not nest.is_nested(model_outputs):
    model_outputs = [model_outputs]
  model_outputs = tf_utils.convert_inner_node_data(model_outputs)
  config['output_layers'] = model_outputs
  return config


def shape_with_no_batch_size(x):
  # [FUNCTION DOC]
  # HELPER: shape normalization for building input specs.
  # Input: a tensor-like object with a static shape. Returns nothing if the rank is unknown,
  # otherwise the shape as a list of dimensions with the first (batch) dimension replaced by
  # unknown (a scalar shape yields an empty list).
  if x.shape.rank is None:
    return None
  shape = x.shape.as_list()
  if shape:
    shape[0] = None
  return shape


class ModuleWrapper(base_layer.Layer):
  # [CLASS DOC]
  # Layer adapter that lets a plain tf.Module (or any callable object) be used inside the
  # Functional and Sequential APIs. Holds the wrapped module, the name of the method used as the
  # forward pass, and flags saying whether that method takes training and mask arguments.
  """Wrapper for `tf.Module`s to support the Functional and Sequential API."""

  def __init__(self, module, method_name=None, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: module (the module object to wrap), method_name (optional string
    # naming the method used as the forward pass; defaults to the call-operator method if the
    # object is callable, else a method named call) and extra keyword arguments passed to the base
    # layer. Raises a value error if no usable method exists. Stores the module and method name,
    # and inspects the method's signature to record whether it accepts training and mask
    # arguments (true also if it takes arbitrary keywords).
    """Initializes the wrapper Layer for this module.

    Args:
      module: The `tf.Module` instance to be wrapped.
      method_name: (Optional) str. The name of the method to use as the forward
        pass of the module. If not set, defaults to '__call__' if defined, or
        'call'.
      **kwargs: Additional keywrod arguments. See `tf.keras.layers.Layer`.

    Raises:
      ValueError: If `method` is not defined on `module`.
    """
    super(ModuleWrapper, self).__init__(**kwargs)
    if method_name is None:
      if hasattr(module, '__call__'):
        method_name = '__call__'
      elif hasattr(module, 'call'):
        method_name = 'call'
    if method_name is None or not hasattr(module, method_name):
      raise ValueError('{} is not defined on object {}'.format(
          method_name, module))

    self._module = module
    self._method_name = method_name

    # Check if module.__call__ has a `training` arg or accepts `**kwargs`.
    method = getattr(module, method_name)
    method_arg_spec = tf_inspect.getfullargspec(method)
    self._expects_training_arg = ('training' in method_arg_spec.args or
                                  method_arg_spec.varkw is not None)
    self._expects_mask_arg = ('mask' in method_arg_spec.args or
                              method_arg_spec.varkw is not None)

  def call(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Forward pass: calls the wrapped module's chosen method with the given positional and
    # keyword arguments. The training and mask keyword arguments are dropped first if the module
    # method does not accept them. Returns whatever the module method returns.
    if 'training' in kwargs and not self._expects_training_arg:
      kwargs.pop('training')
    if 'mask' in kwargs and not self._expects_mask_arg:
      kwargs.pop('mask')
    return getattr(self._module, self._method_name)(*args, **kwargs)
