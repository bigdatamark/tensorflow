# [MODULE DOC]
# Defines Node, the record of one call of a Keras layer on some inputs (connectivity between
# layers in a Functional graph), KerasHistory (named tuple attached to symbolic tensors saying
# which layer call produced them) and the is_keras_tensor predicate. Used by base layer and
# functional model construction, serialization of configs, and replaying graphs.
# No import-time side effects beyond defining a module constant for the constant-value marker.
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
# pylint: disable=g-classes-have-attributes
"""Contains the `Node` class."""

import collections
import copy
import json
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.saving.saved_model import json_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.util import nest

_CONSTANT_VALUE = '_CONSTANT_VALUE'


class Node:
  # [CLASS DOC]
  # One edge-set of the layer graph: records a single call of a layer, with copies of the call
  # arguments, the outputs, which arguments are symbolic Keras tensors, and cached id lists. On
  # construction it registers itself in the layer's inbound node list and in the outbound node
  # lists of the producing layers, and attaches layer history to the output tensors. Offers
  # navigation (parents, inbound iteration), argument substitution during graph replay, config
  # serialization, and several legacy compatibility properties that only look at the first argument.
  """A `Node` describes the connectivity between two layers.

  Each time a layer is connected to some new input,
  a node is added to `layer._inbound_nodes`.
  Each time the output of a layer is used by another layer,
  a node is added to `layer._outbound_nodes`.

  Args:
      layer: The Layer for the Layer.__call__ this node represents.
      call_args: The positional arguments the Layer was called with.
      call_kwargs: The keyword arguments the Layer was called with.
      outputs: The outputs of the Layer.__call__
  """

  def __init__(self,
               layer,
               call_args=None,
               call_kwargs=None,
               outputs=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: layer, the layer that was called; call_args, positional arguments of the
    # call (default empty list); call_kwargs, keyword arguments (default empty mapping); outputs,
    # the layer outputs, possibly a nested structure (default empty list). A node is an input node
    # when both argument containers are empty. Arguments and outputs are structurally copied (not
    # deep copied) so later user changes do not affect the node. Caches the flattened arguments,
    # and a fast-path flag for a single positional tensor. Outside eager mode (TF1 graphs) it
    # creates Keras history for raw tensors that need it. Records which flattened arguments are
    # Keras tensors, by identity string and position. Side effects: appends itself to the layer's
    # inbound nodes and to each producing layer's outbound nodes, and sets layer history (layer,
    # node index, output position) on each flattened output. Caches input and output id strings.
    call_args = [] if call_args is None else call_args
    call_kwargs = {} if call_kwargs is None else call_kwargs
    outputs = [] if outputs is None else outputs

    self.layer = layer
    self.is_input = not call_args and not call_kwargs

    # These arguments are user-provided. Copy the structures here so that
    # future user modifications do not affect the node's metadata.
    # We copy using map_structure rather than python's shallow or deep copy,
    # because the args can be data structures (so shallow copy is
    # insufficient), but individual values might not support copy.copy
    # or be too expensive to deep copy.
    call_args = nest.map_structure(lambda t: t, call_args)
    call_kwargs = nest.map_structure(lambda t: t, call_kwargs)
    self.outputs = nest.map_structure(lambda t: t, outputs)
    self.call_args = call_args
    self.call_kwargs = call_kwargs

    # Cached for performance.
    self._flat_arguments = nest.flatten((self.call_args, self.call_kwargs))
    # Used to avoid expensive `nest` operations in the most common case.
    self._single_positional_tensor_passed = (not self.call_kwargs and len(
        self.call_args) == 1 and tensor_util.is_tf_type(self.call_args[0]))

    if not ops.executing_eagerly_outside_functions():
      # Create TensorFlowOpLayers if needed (in TF1)
      for obj in self._flat_arguments:
        if (isinstance(obj, tensor_lib.Tensor) and
            base_layer_utils.needs_keras_history(
                obj, ignore_call_context=True)):
          base_layer_utils.create_keras_history(obj)

    self._keras_inputs = []
    self._keras_inputs_ids_and_indices = []
    for i, ele in enumerate(self._flat_arguments):
      if is_keras_tensor(ele):
        self._keras_inputs.append(ele)
        kt_id = str(id(ele))
        kt_index = i
        self._keras_inputs_ids_and_indices.append((kt_id, kt_index))

    # Wire up Node to Layers.
    self.layer._inbound_nodes.append(self)
    for kt in self.keras_inputs:
      inbound_layer = kt._keras_history.layer
      if inbound_layer is not None:  # `None` for `Input` tensors.
        inbound_layer._outbound_nodes.append(self)

    # Set metadata on outputs.
    node_index = len(self.layer._inbound_nodes) - 1
    for i, tensor in enumerate(nest.flatten(outputs)):
      tensor._keras_history = KerasHistory(
          layer=layer, node_index=node_index, tensor_index=i)

    # Cached for performance.
    self.flat_input_ids = [str(id(t)) for t in self._keras_inputs]
    self.flat_output_ids = [str(id(t)) for t in nest.flatten(self.outputs)]

  @property
  def keras_inputs(self):
    # [FUNCTION DOC]
    # Property returning the list of symbolic Keras tensors among the call arguments (in flattened
    # order), computed at construction time.
    """Tensors input to this node that can be traced back to a `keras.Input`."""
    return self._keras_inputs

  @property
  def parent_nodes(self):
    # [FUNCTION DOC]
    # Property returning the list of nodes that produced this node's symbolic inputs: for each
    # Keras input with a producing layer, the node at the recorded index in that layer's inbound
    # nodes. Inputs without a layer (graph input tensors) are skipped; duplicates are kept.
    """Returns all the `Node`s whose output this node immediately depends on."""
    node_deps = []
    for kt in self.keras_inputs:
      layer = kt._keras_history.layer
      node_index = kt._keras_history.node_index
      if layer is not None:  # `None` for `Input` tensors.
        node_deps.append(layer._inbound_nodes[node_index])
    return node_deps

  def iterate_inbound(self):
    # [FUNCTION DOC]
    # Generator over this node's symbolic inputs. For each one yields a four-item tuple: the
    # producing layer (nothing for graph inputs), the producing node index, the output position
    # within that call, and the tensor itself. No side effects.
    """Yields tuples representing the data inbound from other nodes.

    Yields:
      tuples like: (inbound_layer, node_index, tensor_index, tensor).
    """
    for kt in self.keras_inputs:
      keras_history = kt._keras_history
      layer = keras_history.layer
      node_index = keras_history.node_index
      tensor_index = keras_history.tensor_index
      yield layer, node_index, tensor_index, kt

  def map_arguments(self, tensor_dict):
    # [FUNCTION DOC]
    # Rebuilds the call arguments with real computed tensors substituted for the symbolic ones.
    # Input: tensor_dict, mapping from the identity string of a symbolic tensor to a list of
    # computed tensors. For each symbolic argument it removes (pops) one computed tensor from the
    # matching list, so the lists are mutated and each entry can be consumed once per use. Has a
    # fast path for a single positional tensor. Returns a pair: positional arguments (tuple in the
    # fast path, otherwise the original structure type) and keyword arguments mapping. Raises a
    # key error if an id is missing and an index error if a list is empty.
    """Maps Keras Tensors to computed Tensors using `tensor_dict`."""
    if self._single_positional_tensor_passed:
      # Performance optimization for most common case.
      kt_id, _ = self._keras_inputs_ids_and_indices[0]
      return (tensor_dict[kt_id].pop(),), {}
    else:
      flat_arguments = copy.copy(self._flat_arguments)
      for kt_id, kt_index in self._keras_inputs_ids_and_indices:
        flat_arguments[kt_index] = tensor_dict[kt_id].pop()

      args, kwargs = nest.pack_sequence_as((self.call_args, self.call_kwargs),
                                           flat_arguments)
      return args, kwargs

  def serialize(self, make_node_key, node_conversion_map):
    # [FUNCTION DOC]
    # Produces the saved-config form of this node for a Functional model. Inputs: make_node_key,
    # callback taking a layer name and node index and returning a key string; node_conversion_map,
    # mapping from such keys to renumbered node indices (missing keys default to index zero). The
    # first call argument is split out; all other positional arguments are merged with keyword
    # arguments into one keyword mapping, with symbolic tensors replaced by layer name, node index
    # and tensor index lists, arrays and eager-evaluated tensors turned into lists. Raises a type
    # error naming the layer and argument types if the keyword arguments are not JSON-serializable.
    # Each element of the first argument becomes a four-item record (layer name, node index,
    # tensor index, keyword arguments) or, for constants, a marker record with index minus one. The
    # result is wrapped in list wrappers, wrapped in an extra list when not nested and the layer
    # does not preserve input structure, then converted to the inner-node-data format. Returns it.
    # The same keyword mapping object is shared by every first-argument record.
    """Serializes `Node` for Functional API's `get_config`."""
    # Serialization still special-cases first argument.
    args, kwargs = self.call_args, self.call_kwargs
    inputs, args, kwargs = self.layer._split_out_first_arg(args, kwargs)

    # Treat everything other than first argument as a kwarg.
    arguments = dict(zip(self.layer._call_fn_args[1:], args))
    arguments.update(kwargs)
    kwargs = arguments

    def _serialize_keras_tensor(t):
      # [FUNCTION DOC]
      # HELPER: serialization helper for Node.serialize. Converts one call argument: a symbolic
      # tensor becomes a list of layer name, converted node index and tensor index; a numpy array
      # becomes a nested list; a tensor becomes its evaluated value as a list (requires it to be
      # evaluable); anything else is returned unchanged.
      """Serializes a single Tensor passed to `call`."""
      if hasattr(t, '_keras_history'):
        kh = t._keras_history
        node_index = kh.node_index
        node_key = make_node_key(kh.layer.name, node_index)
        new_node_index = node_conversion_map.get(node_key, 0)
        return [kh.layer.name, new_node_index, kh.tensor_index]

      if isinstance(t, np.ndarray):
        return t.tolist()

      if isinstance(t, tensor_lib.Tensor):
        return backend.get_value(t).tolist()

      return t

    kwargs = nest.map_structure(_serialize_keras_tensor, kwargs)
    try:
      json.dumps(kwargs, default=json_utils.get_json_type)
    except TypeError:
      kwarg_types = nest.map_structure(type, kwargs)
      raise TypeError('Layer ' + self.layer.name +
                      ' was passed non-JSON-serializable arguments. ' +
                      'Arguments had types: ' +
                      str(kwarg_types) + '. They cannot be serialized out '
                      'when saving the model.')

    # `kwargs` is added to each Tensor in the first arg. This should be
    # changed in a future version of the serialization format.
    def serialize_first_arg_tensor(t):
      # [FUNCTION DOC]
      # HELPER: serialization helper for Node.serialize. Converts one element of the first call
      # argument into a list wrapper holding a four-item record: for symbolic tensors layer name,
      # converted node index, tensor index and the shared keyword arguments; for constants a
      # constant marker, minus one, the serialized value and the keyword arguments.
      if is_keras_tensor(t):
        kh = t._keras_history
        node_index = kh.node_index
        node_key = make_node_key(kh.layer.name, node_index)
        new_node_index = node_conversion_map.get(node_key, 0)
        data = [kh.layer.name, new_node_index, kh.tensor_index, kwargs]
      else:
        # If an element in the first call argument did not originate as a
        # keras tensor and is a constant value, we save it using the format
        # ['_CONSTANT_VALUE', -1, serializaed_tensor_or_python_constant]
        # (potentially including serialized kwargs in an optional 4th argument
        data = [_CONSTANT_VALUE, -1, _serialize_keras_tensor(t), kwargs]
      return tf_utils.ListWrapper(data)

    data = nest.map_structure(serialize_first_arg_tensor, inputs)
    if (not nest.is_nested(data) and
        not self.layer._preserve_input_structure_in_config):
      data = [data]
    data = tf_utils.convert_inner_node_data(data)
    return data

  #############################################################
  # Properties for Backwards compatibility.
  # These only check the first input argument
  # As nodes are internal, they may be removed in the future.
  #############################################################

  @property
  def input_tensors(self):
    # [FUNCTION DOC]
    # Legacy property. For an input node returns a one-item list holding the outputs; otherwise
    # returns the first positional call argument.
    if self.is_input:
      return [self.outputs]  # Used in `Layer.input`.
    return self.call_args[0]

  @property
  def output_tensors(self):
    # [FUNCTION DOC]
    # Legacy property. For an input node returns a one-item list holding the outputs; otherwise
    # returns the outputs as stored.
    if self.is_input:
      return [self.outputs]  # Used in `Layer.input`.
    return self.outputs

  @property
  def input_shapes(self):
    # [FUNCTION DOC]
    # Legacy property. Maps the backend shape query over the input tensors and returns the shapes
    # in the same structure; if there is exactly one entry and this is not an input node, returns
    # that single shape directly instead of a one-item list.
    input_shapes = nest.map_structure(backend.int_shape, self.input_tensors)
    if len(input_shapes) == 1 and not self.is_input:
      return input_shapes[0]
    return input_shapes

  @property
  def output_shapes(self):
    # [FUNCTION DOC]
    # Legacy property returning the backend-reported shapes of the output tensors, in the same
    # structure as the outputs.
    return nest.map_structure(backend.int_shape, self.output_tensors)

  @property
  def outbound_layer(self):
    # [FUNCTION DOC]
    # Legacy property returning the layer this node belongs to (the layer that was called).
    return self.layer

  @property
  def inbound_layers(self):
    # [FUNCTION DOC]
    # Legacy property. Returns an empty list for input nodes; otherwise, in the structure of the
    # first call argument, the layer that produced each tensor (read from its layer history).
    # Raises an attribute error if an element has no history.
    if self.is_input:
      return []
    inbound_layers = nest.map_structure(lambda t: t._keras_history.layer,
                                        self.call_args[0])
    return inbound_layers


class KerasHistory(
    collections.namedtuple('KerasHistory',
                           ['layer', 'node_index', 'tensor_index'])):
  # [CLASS DOC]
  # Immutable three-field record (layer, node index, tensor index) attached to symbolic tensors to
  # remember which call of which layer produced them and at which output position. Extends a named
  # tuple with empty slots to keep it lightweight.
  """Tracks the Layer call that created a Tensor, for Keras Graph Networks.

  During construction of Keras Graph Networks, this metadata is added to
  each Tensor produced as the output of a Layer, starting with an
  `InputLayer`. This allows Keras to track how each Tensor was produced, and
  this information is later retraced by the `keras.engine.Network` class to
  reconstruct the Keras Graph Network.

  Attributes:
    layer: The Layer that produced the Tensor.
    node_index: The specific call to the Layer that produced this Tensor. Layers
      can be called multiple times in order to share weights. A new node is
      created every time a Layer is called.
    tensor_index: The output index for this Tensor. Always zero if the Layer
      that produced this Tensor only has one output. Nested structures of
      Tensors are deterministically assigned an index via `nest.flatten`.
  """
  # Added to maintain memory and performance characteristics of `namedtuple`
  # while subclassing.
  __slots__ = ()


def is_keras_tensor(obj):
  # [FUNCTION DOC]
  # Predicate: true if the given object carries layer history, which is how a symbolic Keras
  # tensor is recognised. Input: obj, anything. Returns a boolean. No side effects.
  return hasattr(obj, '_keras_history')
