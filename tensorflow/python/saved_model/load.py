# [MODULE DOC]
# Imports a trackable object from a SavedModel (the TF2 loading path). Provides the public load
# and partial-load functions, a TF2-detection helper, and the internal Loader class that rebuilds
# the object graph, functions, variables and checkpoint state, plus a wrapper for concrete
# functions under distribution strategies. TF1-style SavedModels are delegated to the v1-in-v2
# loader. Also records load metrics and fingerprint information. Import-time side effects: exports
# the public load APIs and defines the built-in registration table and the metrics label.
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
"""Import a trackable object from a SavedModel."""

import collections
import functools
import os
import sys

from absl import logging

from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.function.capture import restore_captures
from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import graph_view
from tensorflow.python.checkpoint import restore
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import values_util
from tensorflow.python.eager import context
from tensorflow.python.eager import function
from tensorflow.python.eager.polymorphic_function import saved_model_utils as function_saved_model_utils
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import fingerprinting
from tensorflow.python.saved_model import fingerprinting_utils
from tensorflow.python.saved_model import function_deserialization
from tensorflow.python.saved_model import load_options
from tensorflow.python.saved_model import load_v1_in_v2
from tensorflow.python.saved_model import loader_impl
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import registration
from tensorflow.python.saved_model import revived_types
from tensorflow.python.saved_model import utils_impl as saved_model_utils
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base
from tensorflow.python.trackable import data_structures
from tensorflow.python.trackable import resource
from tensorflow.python.trackable import trackable_utils
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export

# API label for SavedModel metrics.
_LOAD_V2_LABEL = "load_v2"
# Built-in registrations use the "oneof kind" field in the SavedObject proto,
# instead of "registered_name" field. The "kind" field has almost the same
# functionality as the registered_name, but only contains built-in TensorFlow
# types (like variable, functions, assets).
_BUILT_IN_REGISTRATIONS = {
    "asset": asset.Asset,
    "resource": resource.RestoredResource,
    "constant": function_saved_model_utils.TrackableConstant}


def _unused_handle():
  # [FUNCTION DOC]
  # HELPER: used by the distribution-aware function wrapper when loading under a strategy.
  # Creates a resource-typed placeholder that must never be executed. An assertion operation that
  # always fails (with a message about running cross-replica graphs in a replica context) is
  # made a control dependency of the placeholder. When building a function graph in graph mode,
  # the graph is marked as unsaveable with an explanatory message. Returns the placeholder.
  """Returns a placeholder as a handle that is not supposed to be accessed."""
  error_message = ("Trying to access a placeholder that is not supposed to be "
                   "executed. This means you are executing a graph generated "
                   "from the cross-replica context in an in-replica context.")
  save_error_message = (
      "It seems that you are trying to save a "
      "tf.types.experimental.ConcreteFunction that involves a distributed "
      "model, and the model contains parts that are loaded form a SavedModel. "
      "It's not supported to save such tf.types.experimental.ConcreteFunction. "
      "Try saving a tf.function with input_signature instead, and file a bug if"
      " there are still issues.")

  assert_op = control_flow_assert.Assert(
      array_ops.placeholder_with_default(False, shape=()), [error_message])
  if (not context.executing_eagerly()
     ) and ops.get_default_graph().building_function:
    ops.get_default_graph().mark_as_unsaveable(save_error_message)

  with ops.control_dependencies([assert_op]):
    return array_ops.placeholder(dtype=dtypes.resource)


class _WrapperFunction(function.ConcreteFunction):
  # [CLASS DOC]
  # HELPER: wrapper used by the loader so restored concrete functions behave correctly under
  # distribution strategies. It shallow-copies the state of an existing concrete function and
  # overrides the flat call so that distributed-variable captures are resolved to resource handles
  # in replica context (or when saving a non-distributed model), and to never-executed
  # placeholders in cross-replica context. Created via the function-library loading step.
  """A class wraps a concrete function to handle different distributed contexts.

  The reason for wrapping a concrete function is because the _captured_inputs
  fields used for in-replica context and cross-replica context are different.
  When `load()` is called from within a tf.distribute.strategy scope, the
  captured inputs are distributed variables. When using these distributed
  variables during calling the function, we need different approaches when it is
  in-replica and when it is not in-replica. When it is in replica, naturally we
  should use the corresponding component of the distributed variable; when it is
  not in-replica, calling the function should mean that it is constructing a
  graph that is not actually going to be used. A typical use case is when
  constructing a functional model. In this case, return a placeholder with a
  control dependency to ensure that is never accessed.
  """

  def __init__(self, concrete_function):
    # [FUNCTION DOC]
    # Copies all instance attributes of the given concrete function into this wrapper (shallow
    # copy; the parent constructor is not called). Returns nothing.
    # Shallow copy the concrete_function
    self.__dict__.update(vars(concrete_function))

  def _call_flat(self, args, captured_inputs):
    # [FUNCTION DOC]
    # Calls the function with flat tensor args and the list of captured inputs. Distributed
    # variables among the captured inputs are replaced by their handle when running in a replica
    # context or while saving a non-distributed version; in cross-replica context they are replaced
    # by a placeholder that fails if executed. Other captures are unchanged. Returns whatever the
    # parent flat call returns.

    def get_handle(x):
      # [FUNCTION DOC]
      # HELPER: returns the resource handle of a distributed variable, or x unchanged otherwise.
      return x.handle if distribute_utils.is_distributed_variable(x) else x

    def get_unused_handle(x):
      # [FUNCTION DOC]
      # HELPER: returns a never-executed placeholder for a distributed variable, else x unchanged.
      return _unused_handle() if distribute_utils.is_distributed_variable(x)   \
          else x

    if (distribute_lib.get_replica_context() is not None or
        values_util.is_saving_non_distributed()):
      # If we're in the replica context or are saving a non-distributed version
      # of the model, we resolve the captured variables to the corresponding
      # resource handle. In both situation we call var.handle, but it has
      # different behavior. In the replica context, var.handle resolves the
      # replica local variable handle if the variable is replicated. When saving
      # a non-distributed version of the model, var.handle resolves to the
      # primary variable handle, since we only save one copy of a replicated
      # variable.
      captured_inputs = list(map(get_handle, captured_inputs))
    else:  # cross-replica context
      captured_inputs = list(map(get_unused_handle, captured_inputs))
    return super()._call_flat(args, captured_inputs)


class Loader(object):
  # [CLASS DOC]
  # HELPER: internal engine behind loading an object-based SavedModel. Given the object graph and
  # saved model protos, it deserializes concrete functions, optionally filters which nodes to
  # load, orders nodes by dependency, recreates every object (variables, functions, assets,
  # resources, constants, user objects, registered types), wires the edges between them, restores
  # checkpoint values and set-ups function captures. State: the proto, export directory, options,
  # restored concrete functions and their aliases, node filters and the path-to-id map, the
  # created nodes with their attribute setters, and the ordered node ids. Everything is done in the
  # constructor; callers then read the result with the get method.
  """Helper class to load an object-based SavedModel."""

  def __init__(self, object_graph_proto, saved_model_proto, export_dir,
               ckpt_options, save_options, filters):
    # [FUNCTION DOC]
    # Performs the whole load. Inputs: object_graph_proto (saved object graph), saved_model_proto
    # (provides the first meta graph: asset file defs, node attributes, function library, function
    # aliases), export_dir (path of the SavedModel), ckpt_options (checkpoint options),
    # save_options (the load options object; despite the name it is the load options),
    # filters (null to load all nodes, else a list of dotted node paths or a mapping from paths
    # to existing objects or object/setter pairs). Steps: load the function library with the
    # distribution wrapper; optionally build the alias mapping (warning for unknown functions);
    # resolve filters into node ids; order nodes; load nodes and edges; restore the checkpoint
    # unless skipping is requested; initialize capturable resources (adding the initializer to the
    # table-initializer collection in graph mode). Raises errors from the called helpers.
    # Side effects: logging, graph collections, mutation of passed-in filter objects.
    meta_graph = saved_model_proto.meta_graphs[0]
    self._asset_file_def = meta_graph.asset_file_def
    self._operation_attributes = {
        node.name: node.attr for node in meta_graph.graph_def.node}
    self._proto = object_graph_proto
    self._export_dir = export_dir
    self._concrete_functions = (
        function_deserialization.load_function_def_library(
            library=meta_graph.graph_def.library,
            saved_object_graph=self._proto,
            wrapper_function=_WrapperFunction))
    # Store a set of all concrete functions that have been set up with
    # captures.
    self._restored_concrete_functions = set()
    self._checkpoint_options = ckpt_options
    self._save_options = save_options

    # Metagraph has a mapping from FunctionDef name to aliases
    self._concrete_function_aliases = meta_graph.meta_info_def.function_aliases
    self.function_aliases = {}
    if self._save_options.experimental_load_function_aliases:
      # Create a mapping from aliases to polymorphic restored functions or lists
      # of concrete functions. This mapping can later be used with SaveOptions
      # when re-saving the loaded object to a SavedModel. We start with a
      # mapping from aliases to lists of concrete functions. Later in
      # _recreate_function, on a entry by entry basis, we replace lists with
      # polymorphic restored functions if the concrete function associated with
      # a restored function is identical to a list of concrete functions in an
      # entry.
      concrete_func_list_by_alias = collections.defaultdict(list)
      for concrete_func_name, alias in self._concrete_function_aliases.items():
        if concrete_func_name not in self._concrete_functions:
          logging.warn(
              (
                  "ConcreteFunction `%s` is listed in function alias but it"
                  " is not found."
              ),
              concrete_func_name,
          )
          continue
        concrete_function = self._concrete_functions[concrete_func_name]
        concrete_func_list_by_alias[alias].append(concrete_function)
      self.function_aliases = dict(concrete_func_list_by_alias)

    self._pretty_printer = checkpoint.ObjectGraphProtoPrettyPrinter(self._proto)

    # Stores user-defined node_filters argument.
    self._node_filters = filters
    # Stores map of string paths to integers.
    self._node_path_to_id = self._convert_node_paths_to_ints()
    self._loaded_nodes = {}
    if isinstance(filters, dict):
      # If node_filters is a dict, then the values may contain already created
      # trackable objects. In this case, create a dictionary mapping node IDs to
      # the already created nodes. This dict will be updated in
      # `_retrieve_all_filtered_nodes` with tracked children.
      for node_path, node in filters.items():
        if isinstance(node, tuple):
          self._loaded_nodes[self._node_path_to_id[node_path]] = node
        else:
          self._loaded_nodes[self._node_path_to_id[node_path]] = (node, setattr)

    # Get a list of all integer node ids to load, or None if all nodes should be
    # loaded. This list includes ids of child nodes.
    self._filtered_nodes = self._retrieve_all_filtered_nodes()

    # Order all nodes or filtered nodes using the dependencies.
    self._ordered_node_ids = self._generate_ordered_node_ids()

    self._load_all()

    if not save_options.experimental_skip_checkpoint:
      self._restore_checkpoint()
    for node in self._nodes:
      if isinstance(node, resource.CapturableResource):
        init_op = node._initialize()  # pylint: disable=protected-access
        if not context.executing_eagerly():
          ops.add_to_collection(ops.GraphKeys.TABLE_INITIALIZERS, init_op)

  def _convert_node_paths_to_ints(self):
    # [FUNCTION DOC]
    # Translates the dotted string paths in the node filters into integer node ids by walking the
    # object graph from the root. Returns null if there are no filters, otherwise a mapping from
    # path to id. Raises a value error if a path does not start with the root name or a child is
    # missing, and a type error if a filter key is not a string.
    """Maps all string node paths in node_filters to the int node ids."""
    if self._node_filters is None:
      return None
    path_to_int = {}
    for node_id in self._node_filters:
      int_node_id = None
      if isinstance(node_id, str):
        node_path = node_id.split(".")
        if node_path[0] != "root":
          raise ValueError(
              "When passing string identifiers to node_filters, the first name"
              f" must be root. Received {node_path[0]}.")
        int_node_id = 0
        for n, name in enumerate(node_path[1:]):
          int_node_id = self._find_node_child(
              int_node_id, name, ".".join(node_path[:n+2]))
        path_to_int[node_id] = int_node_id
      else:
        raise TypeError("Elements in node_filters must be strings.")
    return path_to_int

  def _retrieve_all_filtered_nodes(self):
    # [FUNCTION DOC]
    # Breadth-first traversal of the object graph starting from the filtered paths, collecting ids
    # of all nodes (including descendants) that must be loaded. For pre-created objects supplied in
    # the filters, their existing child dependencies are registered as already loaded (with a
    # do-nothing setter for existing data structures). Mutates the path-to-id map and loaded nodes.
    # Returns null when no filter is set or the root is included (load everything); otherwise the
    # set of node ids. Raises a type error if a supplied object is not trackable.
    """Traverses through the object graph to get the IDs of all nodes to load.

    As a side-effect, if node_filters is a dictionary that contains already-
    created objects, then the children tracked by those objects will be
    added to node_filters.

    Returns:
      List of all nodes to load, or None if all nodes should be loaded.

    """
    if self._node_filters is None:
      return None  # All nodes should be loaded.

    all_filtered_nodes = set()
    nodes_to_visit = list(self._node_filters)

    while nodes_to_visit:
      node_path = nodes_to_visit.pop(0)
      node_id = self._node_path_to_id[node_path]
      if node_id in all_filtered_nodes:
        continue
      all_filtered_nodes.add(node_id)

      node, setter = self._loaded_nodes.get(node_id, (None, None))
      if node is not None:
        if not isinstance(node, base.Trackable):
          raise TypeError(
              "Error when processing dictionary values passed to nodes_to_load."
              f"Object at {node_path} is expected to be a checkpointable (i.e. "
              "'trackable') TensorFlow object (e.g. tf.Variable, tf.Module or "
              "Keras layer).")
        node._maybe_initialize_trackable()  # pylint: disable=protected-access

      for reference in self._proto.nodes[node_id].children:
        child_object, _ = self._loaded_nodes.get(
            reference.node_id, (None, None))

        # See if node already tracks the child reference, in which case add the
        # child to the loaded_nodes dict.
        if child_object is None and node is not None:
          child_object = node._lookup_dependency(reference.local_name)  # pylint: disable=protected-access
          if isinstance(child_object, data_structures.TrackableDataStructure):
            # Make setattr a noop to avoid overwriting already existing data
            # structures.
            setter = lambda *args: None

            self._loaded_nodes[reference.node_id] = (child_object, setter)

        child_path = "{}.{}".format(node_path, reference.local_name)
        self._node_path_to_id[child_path] = reference.node_id
        nodes_to_visit.append(child_path)

    if 0 in all_filtered_nodes:
      return None
    return all_filtered_nodes

  def _find_node_child(self, node_id, child_name, path):
    # [FUNCTION DOC]
    # HELPER: path resolution. Returns the node id of the child with the given local name under
    # node_id; path is used only for the error message. Raises a value error if absent.
    for reference in self._proto.nodes[node_id].children:
      if reference.local_name == child_name:
        return reference.node_id
    raise ValueError(f"Unable to find node {path}.")

  def _load_all(self):
    # [FUNCTION DOC]
    # Recreates all nodes, connects their edges, sets up captures for concrete functions outside
    # the object graph, and restores checkpoint save/restore functions on nodes. Returns nothing.
    """Loads all nodes and functions from the SavedModel and their edges."""
    self._load_nodes()
    self._load_edges()

    # Set up concrete functions that aren't part of the object graph
    # (e.g. gradient functions)
    self._setup_remaining_functions()
    self._load_checkpoint_save_and_restore_functions()

  def _load_checkpoint_save_and_restore_functions(self):
    # [FUNCTION DOC]
    # For every loaded node, restores its checkpoint functions. If the only saveable object is the
    # serialize-to-tensors entry, the node's serialize and restore-from-tensors functions are set
    # from the referenced loaded functions. Otherwise legacy saveable object factories are
    # recreated from named pairs of save and restore functions (sharing one lazily created
    # session holder). Mutates private attributes of nodes. Returns nothing.
    """Restores the checkpoint-related save/restore functions to all nodes."""
    temp_session = [None]
    for node_id, proto in self._iter_all_nodes():
      node = self.get(node_id)
      if proto.saveable_objects.keys() == {
          trackable_utils.SERIALIZE_TO_TENSORS_NAME}:
        # Restore Trackable serialize- and restore-from-tensor functions.
        assert len(proto.saveable_objects) == 1
        saveable_object_proto = next(iter(proto.saveable_objects.values()))
        save_fn_id = saveable_object_proto.save_function
        restore_fn_id = saveable_object_proto.restore_function
        node._serialize_to_tensors = self.get(save_fn_id)  # pylint: disable=protected-access
        node._restore_from_tensors = self.get(restore_fn_id)  # pylint: disable=protected-access
      else:
        # Restore legacy SaveableObject functions.
        saveable_fn_by_name = {}
        for name, saveable_object_proto in proto.saveable_objects.items():
          save_fn_id = saveable_object_proto.save_function
          restore_fn_id = saveable_object_proto.restore_function
          saveable_fn_by_name[name] = (self.get(save_fn_id),
                                       self.get(restore_fn_id))

        node._self_saveable_object_factories = (  # pylint: disable=protected-access
            saveable_object_util.recreate_saveable_objects(saveable_fn_by_name,
                                                           temp_session))

  def _load_edges(self):
    # [FUNCTION DOC]
    # Adds attribute edges from each node to its children. If a filter excluded the root, creates
    # a dummy root and intermediate base user objects along each filtered path and attaches the
    # loaded nodes, for checkpoint compatibility. Returns nothing; mutates created objects.
    """Adds edges from objects to other objects and functions."""
    for node_id, object_proto in self._iter_all_nodes():
      self._add_object_graph_edges(object_proto, node_id)

    # If root object isn't loaded, then create edges from the root for
    # checkpoint compatibility.
    if self._filtered_nodes is not None and 0 not in self._filtered_nodes:
      root = self.get(0)
      for node_path in self._node_filters:
        loaded_node = self._nodes[self._node_path_to_id[node_path]]
        path = node_path.split(".")
        current_node = root
        for name in path[1:-1]:
          if not hasattr(current_node, name):
            setattr(current_node, name, self._recreate_base_user_object()[0])
          current_node = getattr(current_node, name)
        if not hasattr(current_node, path[-1]):
          setattr(current_node, path[-1], loaded_node)

  def _add_object_graph_edges(self, proto, node_id):
    # [FUNCTION DOC]
    # Sets each child of the node (named in proto's children) as an attribute of the node using
    # the node's setter. If a child is named for the call operator and the object is not callable,
    # installs a call method on the object's type (which affects the whole class, not just the
    # instance, despite the per-instance comment). Returns nothing.
    # KNOWN ISSUE: the inline comment says the call hook is added per instance, but the code sets
    # it on the object's type.
    """Adds edges from an object to its children."""
    obj = self._nodes[node_id]
    setter = self._node_setters[node_id]

    for reference in proto.children:
      setter(obj, reference.local_name, self._nodes[reference.node_id])
      # Note: if an object has an attribute `__call__` add a class method
      # that allows `obj()` syntax to work. This is done per-instance to
      # allow `callable` to be used to find out if an object is callable.
      if reference.local_name == "__call__" and not callable(obj):
        setattr(type(obj), "__call__", _call_attribute)

  def _setup_remaining_functions(self):
    # [FUNCTION DOC]
    # Sets up captures for every saved concrete function (in sorted name order) that has not yet
    # been set up, e.g. gradient functions not reachable from the object graph. Returns nothing.
    concrete_function_names = sorted(self._proto.concrete_functions.keys())
    for name in concrete_function_names:
      if name in self._restored_concrete_functions:
        continue
      self._setup_function_captures(name, self._nodes)

  def _setup_function_captures(self, concrete_function_name, nodes):
    # [FUNCTION DOC]
    # Restores the captured inputs of one concrete function once (idempotent, tracked in the
    # restored set). The function's bound input node ids are mapped through nodes (mapping from
    # node id to object) and passed to the capture restoration utility. Returns nothing.
    """Setup captures and variables in a restored function."""
    if concrete_function_name in self._restored_concrete_functions:
      return
    self._restored_concrete_functions.add(concrete_function_name)
    concrete_function = self._concrete_functions[concrete_function_name]
    proto = self._proto.concrete_functions[concrete_function_name]
    inputs = [nodes[node_id] for node_id in proto.bound_inputs]
    restore_captures.restore_captures(concrete_function, inputs)

  def _initialize_loaded_nodes(self):
    # [FUNCTION DOC]
    # HELPER: splits the already-loaded (user supplied) nodes into two new mappings from node id
    # to object and from node id to setter. Returns both.
    nodes = {}
    node_setters = {}
    for node_id, (node, setter) in self._loaded_nodes.items():
      nodes[node_id] = node
      node_setters[node_id] = setter
    return nodes, node_setters

  def _get_node_dependencies(self, proto):
    # [FUNCTION DOC]
    # Collects what must exist before a saved object can be created. Input: a saved object proto.
    # Returns a mapping with its declared dependencies by name, plus (for functions and bare
    # concrete functions) the bound input node ids of their concrete functions keyed by the id
    # itself, plus (for resources) the resource creator child under its name. No side effects.
    """Returns a dictionary of all dependencies of an object.

    Args:
      proto: A SavedObject proto.

    Returns:
      Dict mapping string dependency name *or* int node id to the node id.
      The int node id key is used for mapping function captures.
    """
    dependencies = {ref.local_name: ref.node_id for ref in proto.dependencies}
    kind = proto.WhichOneof("kind")
    if kind == "function":
      concrete_functions = proto.function.concrete_functions
      for fn_name in concrete_functions:
        for bound_input in self._proto.concrete_functions[fn_name].bound_inputs:
          dependencies[bound_input] = bound_input
    elif kind == "bare_concrete_function":
      fn_name = proto.bare_concrete_function.concrete_function_name
      for bound_input in self._proto.concrete_functions[fn_name].bound_inputs:
        dependencies[bound_input] = bound_input
    elif kind == "resource":
      # Make sure that the resource creator is listed as a dependency.
      for child in proto.children:
        if child.local_name == "_create_resource":
          dependencies["_create_resource"] = child.node_id
    return dependencies

  def _generate_ordered_node_ids(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the cycle error message concatenates string pieces without separating spaces
    # ("dependenciesin", "pleasefile"), so the text reads incorrectly.
    # Computes node ids (all, or the filtered subset) in an order where dependencies come first.
    # Builds a dependency map from each node's dependencies (skipped for already-created nodes),
    # and adds slot-variable ordering: a slot variable depends on its optimizer, its original
    # variable, and the previous slot of the same optimizer. Returns the ordered list. Raises a
    # value error if a dependency is outside the filter, or if a dependency cycle exists.
    """Orders the node ids so that dependencies appear first."""
    if self._filtered_nodes is None:
      unordered_ids = range(len(self._proto.nodes))
    else:
      unordered_ids = list(self._filtered_nodes)

    # Maps node ids -> list of dependencies (ids of other nodes that must be
    # loaded before it).
    dependency_map = collections.defaultdict(list)
    for node_id in unordered_ids:
      deps = dependency_map[node_id]
      if self._loaded_nodes.get(node_id) is not None:
        # Deps are only used if the node has not been created.
        continue
      proto = self._proto.nodes[node_id]
      for dep in set(self._get_node_dependencies(proto).values()):
        deps.append(dep)
        if self._filtered_nodes is not None and dep not in self._filtered_nodes:
          raise ValueError(
              "Unable to partially load SavedModel since the specified filter "
              "does not include all required objects for loading (e.g. "
              "variables used in functions or deserialization dependencies). "
              "Please include this path in the filter: "
              f"{self._pretty_printer.node_names[dep]}")

      # Add optimizer slot variable to dependency map.
      prev_slot = None
      for slot_variable_proto in proto.slot_variables:
        slot_variable_node_id = slot_variable_proto.slot_variable_node_id
        # The optimizer and original variable must be created before the slot
        # variable, since the slot variable is generated using the Optimizer's
        # add_slot API.
        slot_deps = dependency_map[slot_variable_node_id]
        slot_deps.append(node_id)
        slot_deps.append(slot_variable_proto.original_variable_node_id)

        if prev_slot is not None:
          # Add previous slot to deps so that the optimizer slot variables are
          # added in order. The ordering is needed because the slot name and
          # variable are both added to ordered lists, which are exposed to the
          # user via `Optimizer.get_slot_names()` and `Optimizer.weights`.
          # TODO(kathywu): Maybe enforce some sort of deterministic ordering in
          # `order_by_dependency` to avoid doing this?
          slot_deps.append(prev_slot)
        prev_slot = slot_variable_node_id
    try:
      return list(trackable_utils.order_by_dependency(dependency_map))
    except trackable_utils.CyclicDependencyError:
      # This should not happen since there is already a validation for cycles
      # when saving, but raise an error just in case.
      raise ValueError("Encountered a cycle in the deserialization dependencies"
                       "in the SavedModel. This is extremely unexpected, please"
                       "file a bug and make sure you are not manually modifying"
                       " the SavedModel.")

  def _iter_all_nodes(self):
    # [FUNCTION DOC]
    # Generator yielding pairs of node id and saved object proto for the ordered node ids.
    for node_id in self._ordered_node_ids:
      yield node_id, self._proto.nodes[node_id]

  def _load_nodes(self):
    # [FUNCTION DOC]
    # Creates the object for every saved node in dependency order. Pre-supplied nodes are kept;
    # optimizer slot variables are created through the optimizer's slot-adding method (using the
    # original variable and slot name); other nodes are recreated from their proto. If the root
    # was not loaded, a dummy base user object stands in as root. Sets the node list (indexed by
    # node id, null for unloaded) and the node setters mapping. Returns nothing.
    """Load all saved objects."""
    # `nodes` maps from node ids to recreated objects
    # `node_setters` maps from node ids to setter functions
    # (same signature as setattr) for setting children.
    nodes, node_setters = self._initialize_loaded_nodes()

    # Figure out which objects are slot variables. These objects are created
    # with Optimizer.add_slot rather than _recreate_variable.
    # Maps slot node id -> optimizer node id, SlotVariableReference proto
    slot_variable_node_ids = {}

    for node_id, proto in self._iter_all_nodes():
      for slot_variable_proto in proto.slot_variables:
        slot_variable_node_id = slot_variable_proto.slot_variable_node_id
        slot_variable_node_ids[slot_variable_node_id] = (node_id,
                                                         slot_variable_proto)

    # Re-create everything.
    for node_id, proto in self._iter_all_nodes():
      if nodes.get(node_id) is not None:
        continue
      elif node_id in slot_variable_node_ids:
        # Use the public Optimizer interface when creating slot variables.
        optimizer_node_id, slot_variable_proto = slot_variable_node_ids[node_id]
        optimizer_object = nodes[optimizer_node_id]
        optimized_variable = nodes[
            slot_variable_proto.original_variable_node_id]
        slot_variable = optimizer_object.add_slot(
            var=optimized_variable,
            slot_name=slot_variable_proto.slot_name)
        nodes[slot_variable_proto.slot_variable_node_id] = slot_variable
        node_setters[slot_variable_proto.slot_variable_node_id] = setattr
      else:
        node, setter = self._recreate(proto, node_id, nodes)
        nodes[node_id] = node
        node_setters[node_id] = setter

    # If root object is not loaded, add a dummy root object for checkpoint
    # compatibility.
    if 0 not in nodes:
      nodes[0] = self._recreate_base_user_object()[0]

    self._nodes = [nodes.get(node_id)
                   for node_id in range(len(self._proto.nodes))]
    self._node_setters = node_setters

  def _restore_checkpoint(self):
    # [FUNCTION DOC]
    # Restores variable values from the SavedModel's variables checkpoint into the loaded object
    # graph rooted at node 0, with the checkpoint file prefix constant created on the CPU. With
    # partial checkpoints allowed, expects partial matches and asserts the match is non-trivial;
    # otherwise asserts that all existing objects matched. In graph mode it additionally wires the
    # restore operations into variable initializers or the table-initializer collection. Raises a
    # not-implemented error in graph mode for objects using registered savers or whose state
    # cannot be restored. Returns nothing.
    """Load state from checkpoint into the deserialized objects."""
    variables_path = path_helpers.get_variables_path(self._export_dir)
    # TODO(b/205010730): Clean use of private methods of TrackableSaver.
    # pylint: disable=protected-access
    saver = checkpoint.TrackableSaver(graph_view.ObjectGraphView(self.get(0)))
    with ops.device("CPU"):
      saver._file_prefix_placeholder = constant_op.constant(variables_path)
    if self._save_options.allow_partial_checkpoint:
      load_status = saver.restore(variables_path,
                                  self._checkpoint_options).expect_partial()
      load_status.assert_nontrivial_match()
    else:
      load_status = saver.restore(variables_path, self._checkpoint_options)
      load_status.assert_existing_objects_matched()
    ckpt = load_status._checkpoint

    if not context.executing_eagerly():
      reader = py_checkpoint_reader.NewCheckpointReader(variables_path)

      # When running in eager mode, the `restore` call above has already run and
      # restored the state of trackables, and calling `position.restore_ops()`
      # would re-run the restore. In graph mode, that will return a cached list
      # of ops that must run to restore the object on that position. We have to
      # wire them in the initializers of the objects so that they get
      # initialized properly when using common practices (e.g. the ones used by
      # ManagedSession) without further user action.
      for object_id, obj in dict(ckpt.object_by_proto_id).items():
        position = restore.CheckpointPosition(checkpoint=ckpt,
                                              proto_id=object_id)
        registered_saver = position.get_registered_saver_name()
        if registered_saver:
          raise NotImplementedError(
              "Loading a SavedModel that uses registered checkpoint saver is "
              f"not supported in graph mode. The loaded object {obj} uses the "
              f"saver registered with the name {registered_saver}.")

        restore_ops = position.restore_ops(reader)
        if restore_ops:
          if resource_variable_ops.is_resource_variable(obj):
            if len(restore_ops) == 1:
              obj._initializer_op = restore_ops[0]
            else:
              obj._initializer_op = control_flow_ops.group(*restore_ops)
          elif (isinstance(obj, lookup_ops.LookupInterface) or
                isinstance(obj, resource.CapturableResource)):
            # We don't need to check for eager execution here, since this code
            # path should only be taken if we are restoring in graph mode.
            ops.add_to_collection(ops.GraphKeys.TABLE_INITIALIZERS, restore_ops)
          else:
            raise NotImplementedError(
                f"Unable to restore state of object {obj} from the checkpoint.")

  def adjust_debug_info_func_names(self, debug_info):
    # [FUNCTION DOC]
    # Builds a new graph debug info message from debug_info: the file list is copied and each trace
    # key (node name, at sign, function name) is rewritten so the function part becomes the name of
    # the restored concrete function's definition signature, or empty if the function is not among
    # the restored ones. Name-to-trace-id entries are not copied. Returns the new message.
    # Raises a value error if a key does not contain exactly one at sign.
    """Rewrite func names in the debug info by using the concrete func names."""
    output_debug_info = graph_debug_info_pb2.GraphDebugInfo()
    output_debug_info.files[:] = debug_info.files
    # TODO: b/292007261 - Read name_to_trace_id as well as traces
    for key in debug_info.traces:
      node, func = key.split("@")
      new_func = ""
      if func in self._concrete_functions:
        new_func = self._concrete_functions[func].function_def.signature.name
      output_debug_info.traces[node + "@" + new_func].CopyFrom(
          debug_info.traces[key])
    return output_debug_info

  def get(self, node_id):
    # [FUNCTION DOC]
    # Returns the loaded object for a node, given either an integer node id or a dotted node path
    # string (resolved through the path-to-id map). Raises a key error for unknown paths, an index
    # error for out-of-range ids. Returns null for nodes that were filtered out.
    if isinstance(node_id, str):
      node_id = self._node_path_to_id[node_id]
    return self._nodes[node_id]

  def _recreate(self, proto, node_id, nodes):
    # [FUNCTION DOC]
    # Creates the Python object for one saved object proto. Looks up a registered class by the
    # proto's registered name, falling back to built-in kinds (asset, resource, constant). If a
    # class is found, it deserializes the object from the proto using already created dependency
    # objects (taken from nodes), export directory, asset file defs and operation attributes, and
    # picks a child setter (the class's trackable-child adder if the object is trackable, else the
    # plain attribute setter). Otherwise uses the default per-kind creation. Returns the object and
    # its setter. Raises errors from deserialization or for unknown kinds.
    """Creates a Python object from a SavedObject protocol buffer.

    Args:
      proto: a SavedObject proto
      node_id: int, the index of this object in the SavedObjectGraph node list.
      nodes: dict mapping int node_ids -> created objects.

    Returns:
      The recreated object, and the set-attribute function for reconnecting
      the trackable children.
    """
    registered_class = registration.get_registered_class(proto.registered_name)
    if registered_class is None:
      registered_class = _BUILT_IN_REGISTRATIONS.get(proto.WhichOneof("kind"))

    dependencies = {}
    for key, dep_node_id in self._get_node_dependencies(proto).items():
      dependencies[key] = nodes[dep_node_id]

    if registered_class:
      obj = registered_class._deserialize_from_proto(  # pylint: disable=protected-access
          proto=proto.serialized_user_proto,
          object_proto=proto,
          dependencies=dependencies,
          export_dir=self._export_dir,
          asset_file_def=self._asset_file_def,
          operation_attributes=self._operation_attributes)
      if isinstance(obj, base.Trackable):
        setter = type(obj)._add_trackable_child  # pylint: disable=protected-access
      else:
        # Returned object may be non-Trackable (e.g. when restoring captures).
        setter = setattr
      return obj, setter
    else:
      return self._recreate_default(proto, node_id, dependencies)

  def _recreate_default(self, proto, node_id, deps):
    # [FUNCTION DOC]
    # Default creation by object kind for objects without a registered class: user object,
    # function, bare concrete function, variable, or captured tensor. deps is the mapping of
    # dependency objects. Returns the object and its setter. Raises a value error listing the
    # supported kinds when the kind is unknown.
    """Creates a Python object from a SavedObject protocol buffer."""
    factory = {
        "user_object": (
            lambda: self._recreate_user_object(proto.user_object, node_id)),
        "function": lambda: self._recreate_function(proto.function, deps),
        "bare_concrete_function": functools.partial(
            self._recreate_bare_concrete_function,
            proto=proto.bare_concrete_function, dependencies=deps),
        "variable": lambda: self._recreate_variable(proto.variable),
        "captured_tensor": functools.partial(
            self._get_tensor_from_fn, proto.captured_tensor),
    }
    kind = proto.WhichOneof("kind")
    if kind not in factory:
      raise ValueError(f"Unknown SavedObject type: {kind}. Expected one of "
                       f"{list(factory.keys())}.")
    return factory[kind]()

  def _recreate_user_object(self, proto, node_id):
    # [FUNCTION DOC]
    # Instantiates a user object from a saved user-object proto. For optimizers, first tries to
    # import the Keras optimizer modules (tf_keras legacy or v2 optimizers, then keras ones) so that
    # the optimizer type is registered; raises an import error if none can be imported. Then looks
    # up a revived type by identifier; if none is found, returns a generic base user object
    # instead. Returns the (object, setter) pair as given by the callee. Side effect: imports.
    """Instantiates a SavedUserObject."""
    if proto.identifier == "optimizer":
      # Make sure that the Keras optimizers module is imported. This is needed
      # to be able to load the "optimizer" object (OptimizerV2), which has
      # special logic around adding slot variables with `add_slot` in this file.
      try:
        import tf_keras  # pylint: disable=g-import-not-at-top,unused-import
        try:
          import tf_keras.optimizers.legacy as _  # pylint: disable=g-import-not-at-top
        except ImportError:
          try:
            import tf_keras.optimizers.optimizer_v2 as _  # pylint: disable=g-import-not-at-top
          except ImportError as e:
            raise ImportError(
                "Error when importing Keras. Unable to load SavedModel that "
                "contains an optimizer without the Keras module.") from e
      except ImportError:
        try:
          import keras.optimizers.legacy as _  # pylint: disable=g-import-not-at-top
        except ImportError:
          try:
            import keras.optimizers.optimizer_v2 as _  # pylint: disable=g-import-not-at-top
          except ImportError as e:
            raise ImportError(
                "Error when importing Keras. Unable to load SavedModel that "
                "contains an optimizer without the Keras module.") from e
    looked_up = revived_types.deserialize(proto)
    if looked_up is None:
      return self._recreate_base_user_object(proto, node_id)
    return looked_up

  def _recreate_base_user_object(self, proto=None, node_id=None):
    # [FUNCTION DOC]
    # Creates a generic trackable placeholder object for a user object that has no registered type.
    # The proto and node_id arguments are accepted for interface compatibility and ignored. A fresh
    # class is defined per call so each instance can be made individually callable. Returns the new
    # instance together with the plain attribute setter.
    del proto, node_id
    # Note: each user object has its own class. This allows making each one
    # individually callable by adding a `__call__` method to the classes of
    # the objects instances that have a `__call__` property.

    class _UserObject(autotrackable.AutoTrackable):
      # [CLASS DOC]
      # HELPER: empty automatically tracked object type, created freshly for each placeholder user
      # object so that a call method can be attached to one object's class without affecting others.
      pass

    return _UserObject(), setattr

  def _recreate_function(self, proto, dependencies):
    # [FUNCTION DOC]
    # Recreates a polymorphic function from a saved function proto using the function
    # deserialization module, then sets up captures for each of its concrete functions using the
    # dependency mapping. If function aliases are being loaded and all its concrete functions have
    # an alias, then the alias entry (a list of concrete functions) is replaced by the restored
    # function when the sets of function names match; otherwise a warning is logged. Returns the
    # function and the plain attribute setter. Raises an assertion error if the alias entry is not
    # a list.
    fn = function_deserialization.recreate_function(
        proto, self._concrete_functions)
    for name in proto.concrete_functions:
      self._setup_function_captures(name, dependencies)

    # If the list of concrete functions associated with this polymorphic
    # restored function is identical to a list of concrete functions found in
    # the function alias mapping, we replace the latter with this restored
    # function. Also see comments in the __init__ method.
    if self._save_options.experimental_load_function_aliases:
      if proto.concrete_functions and all(
          name in self._concrete_function_aliases
          for name in proto.concrete_functions
      ):
        alias = self._concrete_function_aliases[
            next(iter(proto.concrete_functions))
        ]
        aliased = self.function_aliases.get(alias)
        assert isinstance(aliased, list)
        # Note that we cannot compare f.name below with proto.concrete_functions
        # because the former is new name for the restored ConcreteFunction
        # object while the latter is the old name in the original proto.
        if set(f.name for f in aliased) == set(
            f.name for f in fn._list_all_concrete_functions()  # pylint: disable=protected-access
        ):
          self.function_aliases[alias] = fn
        else:
          logging.warn(
              (
                  "Not aliasing '%s' to polymorphic restored function because"
                  " of mismatched concrete functions: %s vs %s"
              ),
              alias,
              set(f.name for f in aliased),
              set(f.name for f in fn._list_all_concrete_functions()),  # pylint: disable=protected-access
          )

    return fn, setattr

  def _recreate_bare_concrete_function(self, proto, dependencies):
    # [FUNCTION DOC]
    # Recreates a bare concrete function from its proto: sets it up for calling, restores its
    # captures from dependencies, and returns the function with the plain attribute setter.
    fn = function_deserialization.setup_bare_concrete_function(
        proto, self._concrete_functions)
    self._setup_function_captures(proto.concrete_function_name, dependencies)
    return fn, setattr

  def _recreate_variable(self, proto):
    # [FUNCTION DOC]
    # Creates an uninitialized variable from a saved variable proto (name, shape, dtype,
    # trainability, synchronization, aggregation, device). The synchronization, aggregation and
    # trainable settings are validated first. Creation happens under a low-priority variable
    # creator that makes uninitialized variables so a distribution strategy's creator can take
    # precedence; the saved device is used when the variable policy saves devices, soft placement
    # is on and a device was saved. Returns the variable and the plain attribute setter. Values are
    # restored later from the checkpoint.
    name = proto.name if proto.name else None
    if name is not None:
      dbg_name = name
    else:
      dbg_name = "<variable loaded from saved model>"
    synchronization, aggregation, trainable = (
        variables.validate_synchronization_aggregation_trainable(
            proto.synchronization, proto.aggregation, proto.trainable,
            name=dbg_name))

    def uninitialized_variable_creator(next_creator, **kwargs):
      # [FUNCTION DOC]
      # HELPER: variable creator hook for _recreate_variable. Ignores the next creator and returns
      # an uninitialized resource variable built from the keyword arguments.
      """A variable creator that creates uninitialized variables."""
      del next_creator
      return resource_variable_ops.UninitializedVariable(**kwargs)

    # Create a variable_creator_scope that creates uninitialized variables with
    # a lower priority such that a potential distributed variable_creator_scope
    # can take precedence.
    with ops.get_default_graph()._variable_creator_scope(  # pylint: disable=protected-access
        uninitialized_variable_creator,
        priority=50):
      saved_device = proto.device
      load_with_device = (
          self._save_options.experimental_variable_policy
          ._save_variable_devices() and config.get_soft_device_placement() and
          saved_device)
      if load_with_device:
        with ops.device(saved_device):
          return variables.Variable(
              shape=proto.shape,
              dtype=proto.dtype,
              name=name,
              trainable=trainable,
              synchronization=synchronization,
              aggregation=aggregation), setattr
      else:
        return variables.Variable(
            shape=proto.shape,
            dtype=proto.dtype,
            name=name,
            trainable=trainable,
            synchronization=synchronization,
            aggregation=aggregation), setattr

  def _get_tensor_from_fn(self, proto):
    # [FUNCTION DOC]
    # Finds the captured tensor named in the proto inside the graph of the referenced restored
    # concrete function. Returns the tensor and the plain attribute setter. Raises a key error if
    # the function or tensor name is unknown.
    outer_graph = self._concrete_functions[proto.concrete_function].graph
    captured_tensor = outer_graph.get_tensor_by_name(proto.name)
    return captured_tensor, setattr


def _call_attribute(instance, *args, **kwargs):
  # [FUNCTION DOC]
  # HELPER: installed as the call method on loaded objects that have a call attribute but are not
  # otherwise callable. Forwards all arguments to the instance's call attribute and returns its
  # result.
  return instance.__call__(*args, **kwargs)


@tf_export("saved_model.load", v1=["saved_model.load_v2"])
def load(export_dir, tags=None, options=None):
  # [FUNCTION DOC]
  # Public entry point (exported as the saved-model load function) that loads a complete
  # SavedModel. export_dir is the directory (string or path-like); tags is an optional tag or
  # sequence of tags selecting the meta graph; options is an optional load options object.
  # Delegates to the partial loader with no filters and returns the root trackable object, which
  # has a signatures attribute. Raises a value error if tags do not match; other errors propagate
  # from the loader. Side effects: reads files, updates metrics and logs.
  """Load a SavedModel from `export_dir`.

  Signatures associated with the SavedModel are available as functions:

  ```python
  imported = tf.saved_model.load(path)
  f = imported.signatures["serving_default"]
  print(f(x=tf.constant([[1.]])))
  ```

  Objects exported with `tf.saved_model.save` additionally have trackable
  objects and functions assigned to attributes:

  ```python
  exported = tf.train.Checkpoint(v=tf.Variable(3.))
  exported.f = tf.function(
      lambda x: exported.v * x,
      input_signature=[tf.TensorSpec(shape=None, dtype=tf.float32)])
  tf.saved_model.save(exported, path)
  imported = tf.saved_model.load(path)
  assert 3. == imported.v.numpy()
  assert 6. == imported.f(x=tf.constant(2.)).numpy()
  ```

  _Loading Keras models_

  Keras models are trackable, so they can be saved to SavedModel. The object
  returned by `tf.saved_model.load` is not a Keras object (i.e. doesn't have
  `.fit`, `.predict`, etc. methods). A few attributes and functions are still
  available: `.variables`, `.trainable_variables` and `.__call__`.

  ```python
  model = tf.keras.Model(...)
  tf.saved_model.save(model, path)
  imported = tf.saved_model.load(path)
  outputs = imported(inputs)
  ```

  Use `tf.keras.models.load_model` to restore the Keras model.

  _Importing SavedModels from TensorFlow 1.x_

  1.x SavedModels APIs have a flat graph instead of `tf.function` objects.
  These SavedModels will be loaded with the following attributes:

  * `.signatures`: A dictionary mapping signature names to functions.
  * `.prune(feeds, fetches) `: A method which allows you to extract
    functions for new subgraphs. This is equivalent to importing the SavedModel
    and naming feeds and fetches in a Session from TensorFlow 1.x.

    ```python
    imported = tf.saved_model.load(path_to_v1_saved_model)
    pruned = imported.prune("x:0", "out:0")
    pruned(tf.ones([]))
    ```

    See `tf.compat.v1.wrap_function` for details.
  * `.variables`: A list of imported variables.
  * `.graph`: The whole imported graph.
  * `.restore(save_path)`: A function that restores variables from a checkpoint
    saved from `tf.compat.v1.Saver`.

  _Consuming SavedModels asynchronously_

  When consuming SavedModels asynchronously (the producer is a separate
  process), the SavedModel directory will appear before all files have been
  written, and `tf.saved_model.load` will fail if pointed at an incomplete
  SavedModel. Rather than checking for the directory, check for
  "saved_model_dir/saved_model.pb". This file is written atomically as the last
  `tf.saved_model.save` file operation.

  Args:
    export_dir: The SavedModel directory to load from.
    tags: A tag or sequence of tags identifying the MetaGraph to load. Optional
      if the SavedModel contains a single MetaGraph, as for those exported from
      `tf.saved_model.save`.
    options: `tf.saved_model.LoadOptions` object that specifies options for
      loading.

  Returns:
    A trackable object with a `signatures` attribute mapping from signature
    keys to functions. If the SavedModel was exported by `tf.saved_model.save`,
    it also points to trackable objects, functions, debug info which it has been
    saved.

  Raises:
    ValueError: If `tags` don't match a MetaGraph in the SavedModel.
  """
  if isinstance(export_dir, os.PathLike):
    export_dir = os.fspath(export_dir)
  result = load_partial(export_dir, None, tags, options)["root"]
  return result


@tf_export("__internal__.saved_model.load_partial", v1=[])
def load_partial(export_dir, filters, tags=None, options=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the existing docstring examples are malformed (unterminated code fence, missing
  # prompts).
  # Core loader behind the load functions. Inputs: export_dir, filters (null/empty to load all, or
  # a list or mapping of dotted node paths, optionally mapping to existing objects), tags
  # (optional, a single tag or sequence or set), options (optional load options; defaults made).
  # Parses the SavedModel and its debug info. For a TF2 object-graph model (exactly one meta graph
  # with an object graph): records metrics, byte-swaps tensor content on big-endian hosts,
  # validates tags (value error on mismatch), builds a Loader inside an init scope (a not-found
  # error is converted into a file-not-found error with a hint about the io device option), then
  # attaches adjusted debug info, TensorFlow version and git version to the root, and the function
  # aliases if requested (value error if the root already has that attribute). For TF1 models, node
  # filters are rejected with a value error and the v1-in-v2 loader is used. Afterwards it records
  # the read path, reads the fingerprint (logging and recording metrics on missing/bad files) and
  # sets the path/singleprint metric. Returns a mapping from filter paths to loaded objects when
  # filters are given for a TF2 model, otherwise a mapping with the single key for the root.
  """Partially load a SavedModel (saved from V2).

  Similar to `tf.saved_model.load`, but with an additional argument that
  lets you specify which nodes to load.
  `tf.saved_model.load_partial(export_dir, ["root"])` and
  `tf.saved_model.load(export_dir)` are equivalent.

  Note: This only works for SavedModels saved with TensorFlow V2 from
  `tf.saved_model.save` or Keras.

  In Tensorflow V2, SavedModel stores the **object graph** of the saved object.
  The graph contains nodes (`tf.Module`, `tf.Variable`, `tf.function`, Keras
  layers, etc.) and edges that are the name of the attributes connecting the
  objects.

  *Example 1*

  ```
  model = tf.Module()
  model.child_layer = tf.Module()
  model.child_layer.v = tf.Variable(5.)
  tf.saved_model.save(model, '/tmp/model')
  loaded = tf.__internal__.saved_model.load_partial(
  ...   '/tmp/model',
  ...   ['root.child_layer', 'root.child_layer.v'])
  loaded['root.child_layer'].v.numpy()
  5.
  loaded['root.child_layer'].v is loaded['root.child_layer.v']
  True

  *Example 2*
  model = tf.Module()
  model.child_layer = tf.Module()
  model.child_layer.v = tf.Variable(5.)
  >>>
  tf.saved_model.save(model, '/tmp/model')
  # Create a variable
  new_variable = tf.Variable(0.)
  loaded = tf.__internal__.saved_model.load_partial(
  ...   '/tmp/model',
  ...   {'root.child_layer': None, 'root.child_layer.v': new_variable})
  loaded['root.child_layer'].v.numpy()
  5.
  new_variable.numpy()
  5.
  ```

  **Loading under different distribution strategies**
  You can load different parts of the model under different distribution
  strategies. Note that this is very experimental so use with care.

  ```
  model = tf.Module()
  model.layer_1 = tf.Module()
  model.layer_1.v = tf.Variable(5.)
  model.layer_2 = tf.Module()
  model.layer_2.v = tf.Variable(7.)
  tf.saved_model.save(model, '/tmp/model')
  # Load with no strategy
  loaded = tf.__internal__.saved_model.load_partial(
  ...   '/tmp/model',
  ...   ['root.layer_1'])
  loaded['root.layer_1'].v
  <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=5.0>
  strategy = tf.distribute.MirroredStrategy()
  with strategy.scope():
  ...   loaded2 = tf.__internal__.saved_model.load_partial(
  ...     '/tmp/model',
  ...     ['root.layer_2'])
  loaded2['root.layer_2'].v
  MirroredVariable:{
      0: <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=7.0>
  }
  ```

  Args:
    export_dir: The SavedModel directory to load from.
    filters: A list or dictionary where each element or key is a string
      path to nodes that should be loaded. Node paths consist of all the child
      attribute names to reach that node in the form: `root.{attribute_name}`.
      The loader will load all of the specified nodes and their recursive
      descendants. When this option is defined, the loader will return a
      dictionary mapping the node paths to the loaded objects.
    tags: A tag or sequence of tags identifying the MetaGraph to load. Optional
      if the SavedModel contains a single MetaGraph, as for those exported from
      `tf.saved_model.save`.
    options: `tf.saved_model.LoadOptions` object that specifies options for
      loading.

  Returns:
    A dictionary mapping node paths from the filter to loaded objects.
  """
  options = options or load_options.LoadOptions()
  if tags is not None and not isinstance(tags, set):
    # Supports e.g. tags=SERVING and tags=[SERVING]. Sets aren't considered
    # sequences for nest.flatten, so we put those through as-is.
    tags = nest.flatten(tags)
  saved_model_proto, debug_info = (
      loader_impl.parse_saved_model_with_debug_info(export_dir))

  loader = None
  if (len(saved_model_proto.meta_graphs) == 1 and
      saved_model_proto.meta_graphs[0].HasField("object_graph_def")):
    metrics.IncrementReadApi(_LOAD_V2_LABEL)
    meta_graph_def = saved_model_proto.meta_graphs[0]
    # tensor_content field contains raw bytes in litle endian format
    # which causes problems when loaded on big-endian systems
    # requiring byteswap
    if sys.byteorder == "big":
      saved_model_utils.swap_function_tensor_content(meta_graph_def, "little",
                                                     "big")
    if (tags is not None
        and set(tags) != set(meta_graph_def.meta_info_def.tags)):
      raise ValueError(
          f"Got an incompatible argument to `tags`: {tags}. The SavedModel at "
          f"{export_dir} has one MetaGraph with tags "
          f"{meta_graph_def.meta_info_def.tags}. You may omit the argument, "
          "pass 'None', or pass matching tags.")
    object_graph_proto = meta_graph_def.object_graph_def

    ckpt_options = checkpoint_options.CheckpointOptions(
        experimental_io_device=options.experimental_io_device)
    with ops.init_scope():
      try:
        loader = Loader(object_graph_proto, saved_model_proto, export_dir,
                        ckpt_options, options, filters)
      except errors.NotFoundError as err:
        raise FileNotFoundError(
            str(err) + "\n You may be trying to load on a different device "
            "from the computational device. Consider setting the "
            "`experimental_io_device` option in `tf.saved_model.LoadOptions` "
            "to the io_device such as '/job:localhost'.")
      root = loader.get(0)
      root.graph_debug_info = loader.adjust_debug_info_func_names(debug_info)
    root.tensorflow_version = meta_graph_def.meta_info_def.tensorflow_version
    root.tensorflow_git_version = (
        meta_graph_def.meta_info_def.tensorflow_git_version)
    metrics.IncrementRead(write_version="2")

    if options.experimental_load_function_aliases:
      if hasattr(root, "function_aliases"):
        raise ValueError(
            "Could not load with experimental_load_function_aliases option"
            " because the top-level object already has an attributed with name"
            " 'function_aliases'"
        )
      root.function_aliases = loader.function_aliases
  else:
    if filters:
      raise ValueError("SavedModels saved from Tensorflow 1.x) cannot be "
                       "loaded with node filters.")
    with ops.init_scope():
      root = load_v1_in_v2.load(
          export_dir, tags, options.experimental_skip_checkpoint
      )
      root.graph_debug_info = debug_info
  # For privacy concerns, please see the note in
  #  tensorflow/cc/saved_model/metrics.h
  metrics.SetReadPath(saved_model_path=str(export_dir))

  # Read and log SavedModel checksum, if it is nonzero.
  try:
    fingerprint = fingerprinting.read_fingerprint(export_dir)
  except FileNotFoundError:
    metrics.SetFoundFingerprintOnLoad(found_status=metrics.kFingerprintNotFound)
    logging.info(
        "Fingerprint not found. Saved model loading will continue.")
    singleprint = ""
  except RuntimeError:
    metrics.SetFoundFingerprintOnLoad(found_status=metrics.kFingerprintError)
    logging.exception(
        "Fingerprint was found, but there was an error when reading the proto. "
        "Saved model loading will continue.")
    singleprint = ""
  else:
    metrics.SetFoundFingerprintOnLoad(found_status=metrics.kFingerprintFound)
    metrics.SetReadFingerprint(
        fingerprint=fingerprinting_utils.to_proto(
            fingerprint).SerializeToString())
    singleprint = fingerprint.singleprint()

  try:
    metrics.SetReadPathAndSingleprint(path=export_dir, singleprint=singleprint)
  except metrics.MetricException:
    logging.info("path_and_singleprint metric could not be logged. "
                 "Saved model loading will continue.")

  if filters and loader is not None:
    return {node_id: loader.get(node_id) for node_id in filters}
  else:
    return {"root": root}


def is_tf2_saved_model(export_dir):
  # [FUNCTION DOC]
  # KNOWN ISSUE: if the fingerprint is readable but its saved object graph hash is zero, the
  # function falls through to the TF1 answer without parsing the meta graph, which can misreport
  # a TF2 model written without that hash.
  # Tells whether the SavedModel in export_dir (directory string) uses TF2 semantics. First reads
  # the fingerprint and answers true if the saved object graph hash is nonzero. If reading fails
  # for any reason, parses the SavedModel proto and answers true when it has one meta graph with
  # an object graph. Otherwise returns false. Logs the decision. Errors from parsing propagate.
  """Identifies if an exported SavedModel is a TF2 SavedModel.

  There are differences in SavedModel semantics between TF1 and TF2 that are
  documented here:
  https://www.tensorflow.org/guide/migrate/saved_model#savedmodel. This helper
  util function serves to distinguish the TF1 vs TF2 semantics used when
  exporting SavedModels.

  Args:
    export_dir: The SavedModel directory to load from.

  Returns:
    True if TF2 SavedModel semantics are used, False if TF1 SavedModel semantics
    are used.
  """
  # Try reading the fingerprint first before parsing the SavedModel proto
  try:
    fingerprint = fingerprinting.read_fingerprint(export_dir)
    if fingerprint.saved_object_graph_hash != 0:
      logging.info("SavedModel at %s is a TF2 SavedModel", export_dir)
      return True
  except Exception:  # pylint: disable=broad-exception-caught
    logging.info(
        "Failed to read fingerprint from SavedModel. Parsing MetaGraph ..."
    )
    saved_model_proto = loader_impl.parse_saved_model(export_dir)
    if len(
        saved_model_proto.meta_graphs
    ) == 1 and saved_model_proto.meta_graphs[0].HasField("object_graph_def"):
      logging.info("SavedModel at %s is a TF2 SavedModel", export_dir)
      return True

  logging.info("SavedModel at %s is a TF1 SavedModel", export_dir)
  return False
