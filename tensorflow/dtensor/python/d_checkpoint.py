# [MODULE DOC]
# Deprecated DTensor checkpointing support. Provides a DTensor-aware saver (_DSaver) that writes
# and reads sharded tensors through DTensor save/restore ops, a restore coordinator that uses it, a
# trackable saver subclass (DTrackableSaver) and the public DTensorCheckpoint class (a checkpoint
# bound to a mesh), exported under the experimental dtensor namespace and marked deprecated in favor
# of the standard checkpoint class, which now supports DTensors directly. Builds on the generic
# checkpoint machinery, overriding private hooks of its classes. Registers the export and
# deprecation wrapper at import time.

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
"""DTensor Checkpoint.

Note that this module contains deprecated functionality, and the DTensor related
checkpoint has been integrated with tf.train.Checkpoint. It can be used out of
the box to save and restore dtensors.
"""

from typing import Dict, List, Optional
import weakref

from tensorflow.core.protobuf import trackable_object_graph_pb2

from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import gen_dtensor_ops
from tensorflow.dtensor.python import layout
from tensorflow.dtensor.python import save_restore
from tensorflow.python.checkpoint import checkpoint as util
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.checkpoint import graph_view as graph_view_lib
from tensorflow.python.checkpoint import restore as restore_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.trackable import base
from tensorflow.python.trackable import data_structures
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import deprecation
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


class _DSaver:  # pylint: disable=protected-access
  # [CLASS DOC]
  # HELPER: internal saver used by the DTensor checkpoint classes.
  # Saves and restores a list of saveable objects on the DTensor device for a given mesh. Holds the
  # saveable objects and the mesh. Created by the trackable saver at save time and by the restore
  # coordinator at restore time.
  """A single device saver that places tensors on DTensor Device."""

  def __init__(self, mesh: layout.Mesh,
               saveable_objects: List[saveable_object.SaveableObject]):
    # [FUNCTION DOC]
    # Constructor. Stores the mesh (the DTensor mesh to save from or restore onto) and the list of
    # saveable objects to handle. Returns nothing.
    self._saveable_objects = saveable_objects
    self._mesh = mesh

  def save(
      self,
      file_prefix: str,
      options: Optional[checkpoint_options.CheckpointOptions] = None
  ) -> Optional[ops.Operation]:
    # [FUNCTION DOC]
    # Saves all saveable objects to a checkpoint with the given prefix using the sharded DTensor
    # save helper. Inputs: file_prefix (string or scalar string tensor); options (optional
    # checkpoint options; ignored except that setting an IO device raises a value error).
    # Gathers name, slice spec and tensor from every spec whose tensor is present (specs with no
    # tensor are skipped). A tensor whose spec device is not the DTensor device (for example a small
    # counter on CPU) is packed as a replicated DTensor over the host mesh. Returns the operation
    # from the sharded save, or nothing when eager (per the underlying helper).
    """Saves the saveable objects to a checkpoint with `file_prefix`.

    Also query the generated shards from the distributed DTensor SaveV2 ops and
    do a MergeV2 on those. Each op here is backed by a global_barrier to avoid
    racing from multiple clients.

    Args:
      file_prefix: A string or scalar string Tensor containing the prefix to
        save under.
      options: Optional `CheckpointOptions` object. This is unused in DTensor.

    Returns:
      An `Operation`, or None when executing eagerly.
    """
    if options is not None and options.experimental_io_device is not None:
      raise ValueError(
          "Specified experimental_io_device in DTensor checkpoint is not supported."
      )
    del options
    tensor_names = []
    tensors = []
    tensor_slices = []
    for saveable in self._saveable_objects:
      for spec in saveable.specs:
        tensor = spec.tensor
        # A tensor value of `None` indicates that this SaveableObject gets
        # recorded in the object graph, but that no value is saved in the
        # checkpoint.
        if tensor is not None:
          if api.device_name() != spec.device:
            # Some small tensors are placed on CPU0 from save manager and
            # broadcasted to DTensor mesh, e,g., SaveCounter.
            tensor = api.pack([tensor] *
                              self._mesh.host_mesh().num_local_devices(),
                              layout.Layout.replicated(
                                  self._mesh.host_mesh(),
                                  rank=tensor.shape.rank))
          tensor_names.append(spec.name)
          tensors.append(tensor)
          tensor_slices.append(spec.slice_spec)
    return save_restore.sharded_save(self._mesh, file_prefix, tensor_names,
                                     tensor_slices, tensors)

  def restore(
      self,
      file_prefix: str,
      options: Optional[checkpoint_options.CheckpointOptions] = None
  ) -> Dict[str, ops.Operation]:
    # [FUNCTION DOC]
    # Restores all saveable objects from a checkpoint with the given prefix using the DTensor
    # restore op. Inputs: file_prefix (string or scalar string tensor); options (optional; setting an
    # IO device raises a value error, otherwise ignored).
    # For each spec, records name, slice spec, dtype, layout and global shape: DTensor save specs
    # supply their own layout and shape; ordinary save specs fall back to a replicated layout on the
    # host mesh with the shape of the current tensor (so variables must already exist). Runs the
    # DTensor restore op on the DTensor device, regroups results per saveable, and calls each
    # saveable's restore. Returns a mapping from saveable names to restore operations. Fails
    # (unpacking error) if there are no specs at all.
    """Restore the saveable objects from a checkpoint with `file_prefix`.

    Args:
      file_prefix: A string or scalar string Tensor containing the prefix for
        files to read from.
      options: Optional `CheckpointOptions` object. This is unused in DTensor.

    Returns:
      A dictionary mapping from SaveableObject names to restore operations.
    """
    if options is not None and options.experimental_io_device is not None:
      raise ValueError(
          "Specified experimental_io_device in DTensor checkpoint is not "
          "supported.")
    del options
    restore_specs = []
    tensor_structure = []
    for saveable in self._saveable_objects:
      saveable_tensor_structure = []
      tensor_structure.append(saveable_tensor_structure)
      # DTensor change 1 : Gather shapes and layout from original saveable
      # specs.
      # Note that this relies on the fact that the variables are already
      # initialized -- which isn't the behavior we want eventually.
      # TODO(b/159035705): Handle the variable initialization in restore.
      for spec in saveable.specs:
        saveable_tensor_structure.append(spec.name)
        if isinstance(spec, d_variable.DSaveSpec):
          restore_specs.append((spec.name, spec.slice_spec, spec.dtype,
                                spec.layout, spec.global_shape))
        # Fall back to replicated layouts for non-DTensor saves that constructs
        # normal SaveSpec.
        elif isinstance(spec, saveable_object.SaveSpec):
          restore_specs.append(
              (spec.name, spec.slice_spec, spec.dtype,
               layout.Layout.replicated(self._mesh.host_mesh(),
                                        spec.tensor.shape.rank).to_string(),
               spec.tensor.shape.as_list()))
    tensor_names, tensor_slices, tensor_dtypes, layouts, global_shapes = zip(
        *restore_specs)
    with ops.device(api.device_name()):
      # DTensor change 2 : Run on customized DTensor RestoreV2 op rather than
      # stock TF io_ops.RestoreV2.
      restored_tensors = gen_dtensor_ops.d_tensor_restore_v2(
          prefix=file_prefix,
          tensor_names=tensor_names,
          shape_and_slices=tensor_slices,
          input_shapes=global_shapes,
          input_layouts=layouts,
          dtypes=tensor_dtypes)
    structured_restored_tensors = nest.pack_sequence_as(tensor_structure,
                                                        restored_tensors)
    restore_ops = {}
    for saveable, restored_tensors in zip(self._saveable_objects,
                                          structured_restored_tensors):
      restore_ops[saveable.name] = saveable.restore(
          restored_tensors, restored_shapes=None)
    return restore_ops


class _DCheckpointRestoreCoordinator(util._CheckpointRestoreCoordinator):  # pylint: disable=protected-access
  # [CLASS DOC]
  # HELPER: internal restore coordinator for DTensor checkpoints.
  # Subclass of the generic object-based checkpoint restore coordinator that remembers the mesh and
  # overrides restoring of saveables so that the DTensor saver is used. Created by the trackable
  # saver during restore.
  """Holds the status of an object-based checkpoint load."""

  def __init__(self, mesh: layout.Mesh, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Passes all keyword arguments to the parent coordinator and stores the mesh.
    # Returns nothing.
    super().__init__(**kwargs)
    self._mesh = mesh

  def restore_saveables(self,
                        tensor_saveables: Dict[str,
                                               saveable_object.SaveableObject],
                        python_positions: List[restore_lib.CheckpointPosition],
                        registered_savers: Optional[Dict[str, Dict[
                            str, base.Trackable]]] = None,
                        reader: py_checkpoint_reader.NewCheckpointReader = None
                        ) -> Optional[List[ops.Operation]]:
    # [FUNCTION DOC]
    # Runs or builds restore operations for saveable objects, overriding the parent to restore via
    # the DTensor saver. Inputs: tensor_saveables (mapping from names to saveable objects for
    # tensors); python_positions (checkpoint positions of Python-state trackables, restored eagerly
    # by reading their value from the checkpoint and deserializing); registered_savers (ignored;
    # not supported); reader (optional checkpoint reader, created lazily from the save path).
    # Raises an assertion error if validating/slicing changes the set of saveable keys. In graph
    # mode, caches the new restore operations by name (asserting a name is not restored twice) and
    # appends them to the result. Returns the list of restore operations (empty when eager).
    """Run or build restore operations for SaveableObjects.

    Args:
      tensor_saveables: `SaveableObject`s which correspond to Tensors.
      python_positions: `CheckpointPosition`s which correspond to `PythonState`
        Trackables bound to the checkpoint.
      registered_savers: a dict mapping saver names-> object name -> Trackable.
        This argument is not implemented for DTensorCheckpoint.
      reader: A CheckpointReader. Creates one lazily if None.

    Returns:
      When graph building, a list of restore operations, either cached or newly
      created, to restore `tensor_saveables`.
    """
    del registered_savers

    restore_ops = []
    # Eagerly run restorations for Python state.
    if python_positions:
      # Lazily create the NewCheckpointReader, since this requires file access
      # and we may not have any Python saveables.
      if reader is None:
        reader = py_checkpoint_reader.NewCheckpointReader(self.save_path_string)
      for position in python_positions:
        key = position.object_proto.attributes[0].checkpoint_key
        position.trackable.deserialize(reader.get_tensor(key))

    # If we have new SaveableObjects, extract and cache restore ops.
    if tensor_saveables:
      validated_saveables = saveable_object_util.validate_and_slice_inputs(
          tensor_saveables)
      validated_names = set(saveable.name for saveable in validated_saveables)
      if set(tensor_saveables.keys()) != validated_names:
        raise AssertionError(
            ("Saveable keys changed when validating. Got back %s, was "
             "expecting %s") % (tensor_saveables.keys(), validated_names))
      # DTensor change: Use _DSaver that does restore on DTensor with
      # customized DTensorRestoreV2 op.
      new_restore_ops = _DSaver(self._mesh, validated_saveables).restore(
          self.save_path_tensor, self.options)
      if not context.executing_eagerly():
        for name, restore_op in sorted(new_restore_ops.items()):
          restore_ops.append(restore_op)
          assert name not in self.restore_ops_by_name
          self.restore_ops_by_name[name] = restore_op
    return restore_ops


class DTrackableSaver(util.TrackableSaver):
  # [CLASS DOC]
  # Trackable saver for DTensor checkpoints. Extends the generic trackable saver so that saveable
  # gathering, save-op construction and restore use the DTensor saver and a mesh. Holds the mesh in
  # addition to the parent's state. Used by DTensorCheckpoint.
  # KNOWN ISSUE: docstring mentions a single-device saver, but the code uses the DTensor saver.
  """A DTensor trackable saver that uses _SingleDeviceSaver."""

  def __init__(self, mesh: layout.Mesh, graph_view):
    # [FUNCTION DOC]
    # Constructor. Initializes the parent saver with the graph view (view of the object graph to
    # save/restore) and stores the mesh. Returns nothing.
    super(DTrackableSaver, self).__init__(graph_view)
    self._mesh = mesh

  def _gather_saveables(self, object_graph_tensor=None):
    # [FUNCTION DOC]
    # HELPER: collects saveable objects for saving, since the base class does not return them.
    # Input: object_graph_tensor (optional tensor that carries the serialized object graph). Gathers
    # serialized tensors; reuses the cached saveables when present, otherwise removes the object
    # graph entry from the serialized tensors (taking the tensor from it) and converts the rest to
    # saveables. Flattens all saveables and appends a no-restore saveable for the object graph.
    # Returns a four-tuple: list of saveable objects, graph proto, feed additions, registered savers.
    # Since the base Checkpoint class does not return SaveableObjects, re-use
    # the saveables cache or generate new Saveables.
    (serialized_tensors, feed_additions, registered_savers,
     graph_proto) = self._gather_serialized_tensors(object_graph_tensor)

    saveables_dict = self._saveables_cache
    if saveables_dict is None:
      # Get and remove object graph tensor from `serialized_tensors`, because
      # the function `serialized_tensors_to_saveable_cache` isn't equipped
      # to handle it.
      object_graph_tensor = serialized_tensors.pop(
          None)[base.OBJECT_GRAPH_PROTO_KEY]
      saveables_dict = (
          saveable_object_util.serialized_tensors_to_saveable_cache(
              serialized_tensors))
    named_saveable_objects = []
    for saveable_by_name in saveables_dict.values():
      for saveables in saveable_by_name.values():
        named_saveable_objects.extend(saveables)
    named_saveable_objects.append(
        base.NoRestoreSaveable(
            tensor=object_graph_tensor,
            name=base.OBJECT_GRAPH_PROTO_KEY))
    return (named_saveable_objects, graph_proto, feed_additions,
            registered_savers)

  def _save_cached_when_graph_building(self,
                                       file_prefix,
                                       object_graph_tensor,
                                       options,
                                       update_ckpt_state=False):
    # [FUNCTION DOC]
    # Creates or reuses the save operation, overriding the parent's private method to use the
    # DTensor saver. Inputs: file_prefix (checkpoint file prefix); object_graph_tensor (tensor fed
    # the current object graph); options (checkpoint options); update_ckpt_state (boolean, default
    # false, accepted but unused; async checkpointing is not supported).
    # Gathers saveables; if the object graph changed, or when eager or inside a function, builds a
    # new DTensor saver, saves, and caches an identity of the file prefix on CPU that depends on the
    # save op, recording the graph proto. Returns a two-element tuple: the cached save-operation
    # tensor and the feed additions.
    """Create or retrieve save ops, overrides parents's private method.

    Args:
      file_prefix: The prefix for saved checkpoint files.
      object_graph_tensor: A `Tensor` to which the current object graph will be
        fed.
      options: `CheckpointOptions` object.
      update_ckpt_state: Optional bool flag. Indiciate whether the internal
        checkpoint state needs to be updated. This is used for async checkpoint,
        which DTrackableSaver currently does not support.
    TODO(chienchunh): Implement async checkpoint for DTrackableSaver.

    Returns:
      A two-element tuple with a filename tensor and a feed_dict of tensors to
      feed when running it (if graph building). The feed dict contains the
      current object graph and any Python state to be saved in the
      checkpoint. When executing eagerly only the first argument is meaningful.
    """
    (named_saveable_objects, graph_proto, feed_additions,
     unused_registered_savers) = self._gather_saveables(
         object_graph_tensor=object_graph_tensor)
    if (self._last_save_object_graph != graph_proto
        # When executing eagerly, we need to re-create SaveableObjects each time
        # save() is called so they pick up new Tensors passed to their
        # constructors. That means the Saver needs to be copied with a new
        # var_list.
        or context.executing_eagerly() or ops.inside_function()):
      # This is needed to avoid MultiDeviceSaver creating unnecessary MergeV2
      # ops in DTensor. It is an issue when saving TPU Variables on host CPU
      # mesh given our limited expressiveness in API and hard-coded logic in
      # broadcasting -- for a small constant Tensor with no extra information,
      # we place it on the first registered mesh(A.K.A. default mesh).
      saver = _DSaver(self._mesh, named_saveable_objects)
      save_op = saver.save(file_prefix, options=options)
      with ops.device("/cpu:0"):
        with ops.control_dependencies([save_op]):
          self._cached_save_operation = array_ops.identity(file_prefix)
      self._last_save_object_graph = graph_proto
    return self._cached_save_operation, feed_additions

  # TODO(b/180466245): Use proper mesh placement semantic.
  def restore(self, save_path, options=None):
    # [FUNCTION DOC]
    # Restores a training checkpoint, placing file prefixes on the host mesh. Inputs: save_path
    # (checkpoint path; if nothing, returns an initialization-only status); options (optional
    # checkpoint options, default created). If the checkpoint has no object graph, falls back to
    # name-based restore: registers a name-based coordinator on each existing trackable (eager only)
    # and returns a name-based status. Otherwise packs the path as a replicated DTensor on the host
    # mesh (a placeholder with a feed dictionary in graph mode), parses the object graph, restores
    # from the root using the DTensor restore coordinator, and also restores attached dependencies
    # (skipping the root and objects already restored). Returns a load status object.
    """Restore a training checkpoint with host mesh placement."""
    options = options or checkpoint_options.CheckpointOptions()
    if save_path is None:
      return util.InitializationOnlyStatus(self._graph_view, ops.uid())
    reader = py_checkpoint_reader.NewCheckpointReader(save_path)
    graph_building = not context.executing_eagerly()
    if graph_building:
      dtype_map = None
    else:
      dtype_map = reader.get_variable_to_dtype_map()
    try:
      object_graph_string = reader.get_tensor(base.OBJECT_GRAPH_PROTO_KEY)
    except errors_impl.NotFoundError:
      # The object graph proto does not exist in this checkpoint. Try the
      # name-based compatibility mode.
      restore_coordinator = util._NameBasedRestoreCoordinator(  # pylint: disable=protected-access
          save_path=save_path,
          dtype_map=dtype_map)
      if not graph_building:
        for existing_trackable in self._graph_view.list_objects():
          # pylint: disable=protected-access
          existing_trackable._maybe_initialize_trackable()
          existing_trackable._name_based_restores.add(restore_coordinator)
          existing_trackable._name_based_attribute_restore(restore_coordinator)
          # pylint: enable=protected-access
      return util.NameBasedSaverStatus(
          restore_coordinator, graph_view=self._graph_view)

    if graph_building:
      if self._file_prefix_placeholder is None:
        # DTensor change: provide a hint for mesh broadcasting to put the input
        # onto the host mesh.
        self._file_prefix_placeholder = api.pack(
            [constant_op.constant("model")] * self._mesh.num_local_devices(),
            layout.Layout.replicated(self._mesh.host_mesh(), rank=0))
      file_prefix_tensor = self._file_prefix_placeholder
      file_prefix_feed_dict = {self._file_prefix_placeholder: save_path}
    else:
      # DTensor change: provide a hint for mesh broadcasting to put the input
      # onto the host mesh.
      file_prefix_tensor = api.pack(
          [constant_op.constant(save_path)] * self._mesh.num_local_devices(),
          layout.Layout.replicated(self._mesh.host_mesh(), rank=0))
      file_prefix_feed_dict = None
    object_graph_proto = (trackable_object_graph_pb2.TrackableObjectGraph())
    object_graph_proto.ParseFromString(object_graph_string)
    # DTensor Change: Hook the proper DSaver in restore.
    checkpoint = _DCheckpointRestoreCoordinator(
        mesh=self._mesh,
        object_graph_proto=object_graph_proto,
        save_path=save_path,
        save_path_tensor=file_prefix_tensor,
        reader=reader,
        restore_op_cache=self._restore_op_cache,
        graph_view=self._graph_view,
        options=options,
        saveables_cache=self._saveables_cache)
    restore_lib.CheckpointPosition(
        checkpoint=checkpoint, proto_id=0).restore(self._graph_view.root)

    # Attached dependencies are not attached to the root, so should be restored
    # separately.
    if self._graph_view.attached_dependencies:
      for ref in self._graph_view.attached_dependencies:
        if ref.name == "root":
          # Root dependency is automatically added to attached dependencies --
          # this can be ignored since it maps back to the root object.
          continue
        proto_id = None
        # Find proto ID of attached dependency (if it is in the proto).
        for proto_ref in object_graph_proto.nodes[0].children:
          if proto_ref.local_name == ref.name:
            proto_id = proto_ref.node_id
            break

        if proto_id in checkpoint.object_by_proto_id:
          # Object has already been restored. This can happen when there's an
          # indirect connection from the attached object to the root.
          continue

        restore_lib.CheckpointPosition(
            checkpoint=checkpoint, proto_id=proto_id).restore(ref.ref)

    load_status = util.CheckpointLoadStatus(
        checkpoint,
        graph_view=self._graph_view,
        feed_dict=file_prefix_feed_dict,
        options=options)
    return load_status


@deprecation.deprecated(
    date=None,
    instructions="Please use tf.train.Checkpoint instead of DTensorCheckpoint. "
    "DTensor is integrated with tf.train.Checkpoint and it can be "
    "used out of the box to save and restore dtensors.")
@tf_export("experimental.dtensor.DTensorCheckpoint", v1=[])
class DTensorCheckpoint(util.Checkpoint):
  # [CLASS DOC]
  # Deprecated checkpoint class for DTensor (use the standard checkpoint class instead). Extends the
  # generic checkpoint with a mesh and replaces its saver with the DTensor trackable saver. Holds
  # the mesh and a lazily created save counter. Exported publicly under the experimental dtensor
  # namespace.
  """Manages saving/restoring trackable values to disk, for DTensor."""

  def __init__(self, mesh: layout.Mesh, root=None, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: mesh (DTensor mesh); root (optional trackable to checkpoint; when given
    # it becomes the saver root and its save counter is reused); keyword arguments (named trackable
    # objects to track). Calls the parent constructor, then sets each keyword argument as an
    # attribute in sorted name order, validating each is trackable. When a root is given, attaches
    # them as dependencies, raising a value error if the root already has a different child of that
    # name. Finally replaces the saver with the DTensor trackable saver for the mesh. Returns nothing.
    super(DTensorCheckpoint, self).__init__(root=root, **kwargs)
    self._mesh = mesh

    saver_root = self
    attached_dependencies = None
    self._save_counter = None  # Created lazily for restore-on-create.
    self._save_assign_op = None

    if root:
      util._assert_trackable(root, "root")
      saver_root = root
      attached_dependencies = []

      # All keyword arguments (including root itself) are set as children
      # of root.
      kwargs["root"] = root
      root._maybe_initialize_trackable()

      self._save_counter = data_structures.NoDependency(
          root._lookup_dependency("save_counter"))
      self._root = data_structures.NoDependency(root)

    for k, v in sorted(kwargs.items(), key=lambda item: item[0]):
      setattr(self, k, v)

      # Call getattr instead of directly using v because setattr converts
      # v to a Trackable data structure when v is a list/dict/tuple.
      converted_v = getattr(self, k)
      util._assert_trackable(converted_v, k)

      if root:
        # Make sure that root doesn't already have dependencies with these names
        attached_dependencies = attached_dependencies or []
        child = root._lookup_dependency(k)
        if child is None:
          attached_dependencies.append(base.TrackableReference(k, converted_v))
        elif child != converted_v:
          raise ValueError(
              "Cannot create a Checkpoint with keyword argument {name} if "
              "root.{name} already exists.".format(name=k))
    # DTensor Change:
    # Override the parents saver with DTrackableSaver with _SingleDeviceSaver.
    self._saver = DTrackableSaver(
        mesh,
        graph_view_lib.ObjectGraphView(
            weakref.ref(saver_root),
            attached_dependencies=attached_dependencies))
