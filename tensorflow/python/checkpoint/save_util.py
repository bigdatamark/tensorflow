# [MODULE DOC]
# Object-based (TF2) checkpoint serialization core. Walks an object graph view, builds the
# TrackableObjectGraph proto, and collects the tensors to write, split into three kinds:
# trackables that serialize themselves to tensors, Python-state trackables (serialized as strings,
# fed at run time in graph mode), and trackables handled by registered savers. Legacy
# saveable-object trackables are bridged through the v1 save utilities. Exposes
# serialize_graph_view as the entry point; everything else is private. No import-time side effects.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Extracts tensors for checkpointing while updating a TrackableObjectGraph.

The tensors are extracted from `Trackable._serialize_to_tensors`.
"""
import collections

from typing import Any, Callable, List, Optional, Tuple, Mapping, Union, Dict

from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import graph_view as graph_view_lib
from tensorflow.python.checkpoint import save_util_v1
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.checkpoint import util
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import base
from tensorflow.python.trackable import python_state
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training.saving import saveable_object as saveable_object_lib
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.types import core
from tensorflow.python.util import object_identity

# Attributes for each Trackable in the checkpointed object graph.
_TrackableData = collections.namedtuple(
    "_TrackableData",
    [
        # A trackable in the root Trackable object graph.
        "trackable",
        # The index at which the Trackable appears in TrackableObjectGraph.nodes.
        "node_id",
        # The BFS-generated path from the root object / used to generate readable
        # checkpoint keys.
        "object_name",
        # A list of ObjectReference for each child connected to this Trackable.
        "children_proto",
        # A list of SlotVariableReference to save to the object (only valid for
        # Optimizer objects).
        "slot_variable_proto",
        # The object to save to checkpoint. Usually this is the same as `trackable`,
        # but can differ when the caller wants to specify a different object to
        # save. For example, when saving checkpoints asynchronously, variables are
        # copied to the CPU. `object_to_save` is set as the copied variable.
        "object_to_save",
    ],
)


def _split_trackables(
    trackable_data: List[_TrackableData]
) -> Tuple[List[_TrackableData], List[_TrackableData],
           Dict[str, List[_TrackableData]]]:
  # [FUNCTION DOC]
  # HELPER: classifies trackables for serialize_graph_view.
  # Takes a list of trackable-data records. Returns three things: a list of ordinary tensor
  # trackables, a list of Python-state trackables, and a mapping from registered saver name to
  # the list of records using it. Python-state takes precedence over a registered saver; anything
  # else goes to the tensor list. Looks up the saver name on the object to save; no side effects.
  """Splits Trackables into 3 categories (tensor/pystate/registered)."""
  tensor_trackables = []
  pystate_trackables = []
  registered_trackables = collections.defaultdict(list)

  for td in trackable_data:
    saver_name = registration.get_registered_saver_name(td.object_to_save)
    if isinstance(td.object_to_save, python_state.PythonState):
      pystate_trackables.append(td)
    elif saver_name:
      registered_trackables[saver_name].append(td)
    else:
      tensor_trackables.append(td)

  return tensor_trackables, pystate_trackables, registered_trackables


def _gather_trackable_data(
    graph_view: graph_view_lib.ObjectGraphView,
    object_map: Mapping[base.Trackable, base.Trackable]
) -> Tuple[List[_TrackableData], Dict[base.Trackable, int]]:
  # [FUNCTION DOC]
  # HELPER: builds the per-object records for serialize_graph_view.
  # Inputs: graph_view (object graph view to traverse) and object_map (mapping from original
  # trackables to replacement objects to save, may be nothing). Traverses breadth first, assigns
  # node ids and path-based object names, serializes slot variables, and builds child references.
  # Returns a list of records (one per trackable, in node order) and a mapping from trackable to
  # node id. No side effects beyond reading the graph.
  """Returns a list of generated TrackableData based on the ObjectGraphView."""
  trackable_objects, node_paths = graph_view.breadth_first_traversal()
  object_names = object_identity.ObjectIdentityDictionary()
  for obj, path in node_paths.items():
    object_names[obj] = trackable_utils.object_path_to_string(path)
  node_ids = object_identity.ObjectIdentityDictionary()
  for node_id, node in enumerate(trackable_objects):
    node_ids[node] = node_id
  slot_variables = util.serialize_slot_variables(
      trackable_objects=trackable_objects,
      node_ids=node_ids,
      object_names=object_names)
  trackable_data = []
  for trackable in trackable_objects:
    children_proto = []
    for child in graph_view.list_children(trackable):
      children_proto.append(
          trackable_object_graph_pb2.TrackableObjectGraph.TrackableObject
          .ObjectReference(node_id=node_ids[child.ref],
                           local_name=child.name))

    trackable_data.append(_TrackableData(
        trackable,
        node_id=node_ids[trackable],
        object_name=object_names[trackable],
        children_proto=children_proto,
        slot_variable_proto=slot_variables.get(trackable, []),
        object_to_save=util.get_mapped_trackable(trackable, object_map)))
  return trackable_data, node_ids


def _fill_object_graph_proto(
    trackable_data: List[_TrackableData]
) -> trackable_object_graph_pb2.TrackableObjectGraph:
  # [FUNCTION DOC]
  # HELPER: creates the object graph proto skeleton for serialize_graph_view.
  # Input: list of trackable-data records, whose node ids must equal their list positions
  # (asserted). Returns a new TrackableObjectGraph proto with one node per record carrying its
  # slot variable references and child references. Nothing is mutated.
  """Name non-slot `Trackable`s and add them to `object_graph_proto`."""
  object_graph_proto = trackable_object_graph_pb2.TrackableObjectGraph()
  for checkpoint_id, td in enumerate(trackable_data):
    assert td.node_id == checkpoint_id
    object_graph_proto.nodes.add(
        slot_variables=td.slot_variable_proto,
        children=td.children_proto)
  return object_graph_proto


def _get_and_write_tensors_to_serialize(
    tensor_trackables: List[_TrackableData],
    node_ids: Dict[base.Trackable, int],
    call_with_mapped_captures: Union[Callable[..., Any], None],
    cache: Union[Dict[base.Trackable, any], None],
    object_graph_proto: trackable_object_graph_pb2.TrackableObjectGraph
) -> Dict[base.Trackable, Any]:
  # [FUNCTION DOC]
  # HELPER: collects tensors to save for serialize_graph_view.
  # Inputs: tensor_trackables (records to serialize), node_ids (mapping trackable to node id),
  # call_with_mapped_captures (optional callback for calling saved functions with captured
  # values mapped), cache (optional mapping used to reuse earlier results, may be nothing), and
  # object_graph_proto (proto to update). For each record: reuses the cached entry if present
  # (merging cached attributes into the proto); otherwise uses the legacy saveable path when the
  # object lacks tensor serialization or has a compat saveable name, else the direct path.
  # Returns a mapping from trackable to a mapping of checkpoint key to tensor. Side effects:
  # adds attribute entries to the proto and populates the cache.
  """Creates dictionary of tensors to checkpoint, and updates the proto."""
  # Maps trackable to the a dictionary of tensors, which maps
  # checkpoint key (-> slice_spec) -> tensor.
  serialized_tensors = object_identity.ObjectIdentityDictionary()

  for td in tensor_trackables:
    if cache is not None and td.object_to_save in cache:
      trackable, tensor_dict, object_proto = cache[td.object_to_save]
      serialized_tensors[trackable] = tensor_dict
      object_graph_proto.nodes[td.node_id].attributes.MergeFrom(object_proto)
      continue

    legacy_name = saveable_compat.get_saveable_name(td.object_to_save) or ""

    if (not saveable_object_util.trackable_has_serialize_to_tensor(
        td.object_to_save) or
        legacy_name):
      # Use the legacy code path for objects that are using SaveableObjects
      # or the compat saveable name decorator.
      trackable, tensor_dict = _get_tensors_from_legacy_saveable(
          td, node_ids, call_with_mapped_captures, object_graph_proto)
    else:
      tensor_dict = _get_tensors_from_trackable(
          td, call_with_mapped_captures, object_graph_proto)
      trackable = td.object_to_save
    serialized_tensors[trackable] = tensor_dict

    if cache is not None and td.object_to_save not in cache:
      cache[td.object_to_save] = (
          trackable, tensor_dict,
          object_graph_proto.nodes[td.node_id].attributes)

  return serialized_tensors


def _get_tensors_from_legacy_saveable(
    trackable_data: _TrackableData,
    node_ids: Dict[base.Trackable, int],
    call_with_mapped_captures: Callable[..., Any],
    object_graph_proto: trackable_object_graph_pb2.TrackableObjectGraph
) -> Tuple[base.Trackable, Dict[str, Any]]:
  # [FUNCTION DOC]
  # HELPER: legacy-saveable branch of the tensor collection.
  # Inputs: one trackable-data record, node_ids, call_with_mapped_captures callback, and the
  # object graph proto. Uses the v1 save utilities to create saveable objects for the single
  # object, wraps them in a saveable compatibility converter, and asks it for its tensors.
  # Returns the converter (used as the trackable) and its mapping of checkpoint key to tensor.
  # Side effect: the v1 utilities update the proto.
  """Gets tensors to serialize from a Trackable with legacy SaveableObjects."""
  # Call `save_util_v1` methods to create legacy SaveableObjects and update the
  # proto.
  object_names = object_identity.ObjectIdentityDictionary()
  object_names[trackable_data.trackable] = trackable_data.object_name
  object_map = object_identity.ObjectIdentityDictionary()
  object_map[trackable_data.trackable] = trackable_data.object_to_save

  checkpoint_factory_map, _ = save_util_v1.get_checkpoint_factories_and_keys(
      object_names, object_map)
  named_saveable_objects, _ = (
      save_util_v1.generate_saveable_objects(
          checkpoint_factory_map,
          object_graph_proto,
          node_ids,
          object_map,
          call_with_mapped_captures,
          saveables_cache=None))
  trackable = (
      saveable_object_util.SaveableCompatibilityConverter(
          trackable_data.object_to_save, named_saveable_objects))
  return trackable, trackable._serialize_to_tensors()  # pylint: disable=protected-access


def _get_tensors_from_trackable(
    trackable_data: _TrackableData,
    call_with_mapped_captures: Union[Callable[..., Any], None],
    object_graph_proto: trackable_object_graph_pb2.TrackableObjectGraph
) -> Dict[str, Any]:
  # [FUNCTION DOC]
  # HELPER: direct tensor-serialization branch of the tensor collection.
  # Inputs: one trackable-data record, an optional call_with_mapped_captures callback (used when
  # the serialize function is a concrete function), and the object graph proto (may be nothing).
  # Calls the object's tensor serialization, builds a checkpoint key for each returned name
  # (escaping the local name), and for save-spec values renames them to the key with an empty
  # slice spec. If a proto is given, adds an attribute entry per tensor. Returns a mapping of
  # checkpoint key to tensor.
  """Gets tensors to serialize from a Trackable."""
  trackable = trackable_data.object_to_save
  save_fn = trackable._serialize_to_tensors  # pylint: disable=protected-access

  if (call_with_mapped_captures and
      isinstance(save_fn, core.ConcreteFunction)):
    ret_tensor_dict = call_with_mapped_captures(save_fn, [])
  else:
    ret_tensor_dict = save_fn()

  # Create checkpoint keys for each entry in the returned tensor dict, and
  # write each entry to the object proto.
  tensor_dict = {}
  for tensor_name, maybe_tensor in ret_tensor_dict.items():
    local_name = trackable_utils.escape_local_name(tensor_name)
    checkpoint_key = trackable_utils.checkpoint_key(trackable_data.object_name,
                                                    local_name)
    tensor_dict[checkpoint_key] = maybe_tensor

    # TODO(b/261786493): Delete this when DCheckpoint is removed.
    if isinstance(maybe_tensor, saveable_object_lib.SaveSpec):
      maybe_tensor.name = checkpoint_key
      maybe_tensor.slice_spec = ""

    if object_graph_proto is not None:
      object_graph_proto.nodes[trackable_data.node_id].attributes.add(
          name=local_name,
          checkpoint_key=checkpoint_key,
          full_name=util.get_full_name(trackable))

  return tensor_dict


def _get_and_write_pystate_feed_additions(
    pystate_trackables: List[_TrackableData],
    cache: Union[Dict[base.Trackable, Any], None],
    object_graph_proto=None
) -> Tuple[Dict[base.Trackable, Any], Dict[base.Trackable, Any]]:
  # [FUNCTION DOC]
  # HELPER: Python-state handling for graph-mode saving in serialize_graph_view.
  # Inputs: pystate_trackables (records), cache (mapping used to keep placeholder string tensors
  # across saves; it must not be nothing here because it is queried), object_graph_proto (proto to
  # update; despite the default of nothing it is always dereferenced). For each record, reuses or
  # creates a constant string tensor on the CPU as a placeholder, serializes the Python state
  # under an init scope, and records the value to feed. Adds an attribute entry to the proto.
  # Returns the mapping trackable to key-to-placeholder, and the mapping placeholder to value.
  """Gets feed additions needed for checkpointing Python State."""
  serialized_tensors = object_identity.ObjectIdentityDictionary()
  # Maps tensor placeholders to python values.
  feed_additions = {}

  for td in pystate_trackables:
    trackable = td.object_to_save
    checkpoint_key = trackable_utils.checkpoint_key(td.object_name,
                                                    python_state.PYTHON_STATE)
    if trackable in cache:
      save_string = cache[td.object_to_save][python_state.PYTHON_STATE]
    else:
      with ops.device("/cpu:0"):
        save_string = constant_op.constant("", dtype=dtypes.string)
        cache[trackable] = {python_state.PYTHON_STATE: save_string}

    with ops.init_scope():
      value = trackable.serialize()
    feed_additions[save_string] = value
    serialized_tensors[trackable] = {checkpoint_key: save_string}

    object_graph_proto.nodes[td.node_id].attributes.add(
        name=python_state.PYTHON_STATE,
        checkpoint_key=checkpoint_key,
        full_name=util.get_full_name(trackable))

  return serialized_tensors, feed_additions


def _get_and_write_registered_savers(
    registered_trackables: Dict[str, List[_TrackableData]],
    object_graph_proto: trackable_object_graph_pb2.TrackableObjectGraph
) -> Dict[str, Dict[str, base.Trackable]]:
  # [FUNCTION DOC]
  # HELPER: registered-saver handling for serialize_graph_view.
  # Inputs: mapping from saver name to records, and the object graph proto. Returns a mapping from
  # saver name to a mapping of object name to the object to save. Side effect: sets the registered
  # saver name and object name on each record's proto node.
  """Generates dictionary of registered savers and updates the proto."""
  registered_savers = collections.defaultdict(dict)
  for saver_name, trackables in registered_trackables.items():
    for td in trackables:
      registered_savers[saver_name][td.object_name] = td.object_to_save

      object_proto = object_graph_proto.nodes[td.node_id]
      object_proto.registered_saver.name = saver_name
      object_proto.registered_saver.object_name = td.object_name

  return registered_savers


def serialize_graph_view(
    graph_view: graph_view_lib.ObjectGraphView,
    object_map: Optional[Mapping[base.Trackable, base.Trackable]] = None,
    call_with_mapped_captures: Optional[Callable[..., Any]] = None,
    cache: Optional[Dict[base.Trackable, Any]] = None) -> ...:
  # [FUNCTION DOC]
  # Main entry point. Inputs: graph_view (object graph view of the root), optional object_map
  # (trackable to replacement object), optional call_with_mapped_captures callback, and optional
  # cache (present in TF1 graph mode, nothing in eager mode). Gathers records, builds the proto,
  # collects tensors and registered savers, and handles Python-state objects either directly
  # (no cache) or via placeholder tensors plus feed additions (cache). Finally marks which proto
  # nodes have checkpoint values. Returns a 4-tuple: serialized tensors per trackable, feed
  # additions (nothing in eager mode), registered savers, and the object graph proto.
  # Side effect: updates the cache when given. The return annotation is a bare ellipsis.
  """Gathers serialization objects, and creates a TrackableObjectGraph proto."""
  # There are 3 types of checkpoint serialization types supported:
  # 1. Trackables that override `Trackable._serialize_to_tensor()`.
  # 2. PythonState: A special type of Trackable that serializes a Python string.
  # 3. Registered Trackable Savers: For objects that need to define advanced
  #    checkpointing operations not supported by (1) or (2).
  trackable_data, node_ids = _gather_trackable_data(graph_view, object_map)
  tensor_trackables, pystate_trackables, registered_trackables = (
      _split_trackables(trackable_data))

  object_graph_proto = _fill_object_graph_proto(trackable_data)

  serialized_tensors = _get_and_write_tensors_to_serialize(
      tensor_trackables,
      node_ids,
      call_with_mapped_captures,
      cache,
      object_graph_proto)
  registered_savers = _get_and_write_registered_savers(
      registered_trackables, object_graph_proto)

  # PythonState trackables must be treated differently depending on if the
  # checkpoint is being saved in TF1 graph mode (`cache` exists) or
  # eager mode (`cache` is None).
  if cache is None:
    # When the tensor cache is None, get the serialized tensors directly.
    feed_additions = None
    serialized_tensors.update(_get_and_write_tensors_to_serialize(
        pystate_trackables,
        node_ids,
        call_with_mapped_captures,
        cache,
        object_graph_proto))
  else:
    # Python state is not automatically updated within a TF session so these
    # values must be passed to sess.run(feed_additions=...).
    new_serialized_tensors, feed_additions = (
        _get_and_write_pystate_feed_additions(pystate_trackables,
                                              cache,
                                              object_graph_proto))
    serialized_tensors.update(new_serialized_tensors)

  # Gather all trackables that have checkpoint values or descendants with
  # checkpoint values, and add that info to the proto.
  util.add_checkpoint_values_check(object_graph_proto)
  return (serialized_tensors, feed_additions, registered_savers,
          object_graph_proto)
