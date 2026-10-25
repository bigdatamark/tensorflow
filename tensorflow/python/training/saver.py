# [MODULE DOC]
# Legacy (graph-mode oriented) checkpoint saving and restoring of variables. Exposes the
# saver builder classes (base and bulk-restore variants) that construct the save and restore
# graph operations, the Saver class that drives saving, restoring and checkpoint bookkeeping
# (keeping the newest N checkpoints, time-based retention, checkpoint state file), and the
# module-level helpers to import and export meta graphs and to build a Saver from an
# object-based checkpoint. Several checkpoint-management functions are re-exported here as
# aliases for old users. Import-time side effects: defines aliases to checkpoint management
# functions, a module-level lock and timestamp used for checkpoint write metrics, and
# registers public symbols through the export decorator. Symbols are deprecated in favor of
# the trackable and saving packages.
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

# pylint: disable=invalid-name
"""Save and restore variables.

Symbols in this file are deprecated. See replacements in
tensorflow/python/training/trackable and tensorflow/python/training/saving.
"""
import collections
import glob
import os.path
import threading
import time

import numpy as np
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.core.protobuf import trackable_object_graph_pb2
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.client import session
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training import py_checkpoint_reader
from tensorflow.python.training import training_util
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export

# TODO(allenl): Remove these aliases once all users are migrated off.
get_checkpoint_state = checkpoint_management.get_checkpoint_state
update_checkpoint_state = checkpoint_management.update_checkpoint_state
generate_checkpoint_state_proto = (
    checkpoint_management.generate_checkpoint_state_proto)
latest_checkpoint = checkpoint_management.latest_checkpoint
checkpoint_exists = checkpoint_management.checkpoint_exists
get_checkpoint_mtimes = checkpoint_management.get_checkpoint_mtimes
remove_checkpoint = checkpoint_management.remove_checkpoint

# Captures the timestamp of the first Saver object instantiation or end of a
# save operation. Can be accessed by multiple Saver instances.
_END_TIME_OF_LAST_WRITE = None
_END_TIME_OF_LAST_WRITE_LOCK = threading.Lock()

# API label for cell name used in checkpoint metrics.
_SAVER_LABEL = "saver_v1"


def _get_duration_microseconds(start_time_seconds, end_time_seconds):
  # [FUNCTION DOC]
  # HELPER: converts timestamps into durations for the checkpoint write/read metrics.
  # Takes a start time and an end time, both in seconds (numbers). Returns the elapsed time as
  # a whole number of microseconds, rounded. If the end is earlier than the start (clock
  # skew), returns zero rather than a negative value. No side effects.
  if end_time_seconds < start_time_seconds:
    # Avoid returning negative value in case of clock skew.
    return 0
  return round((end_time_seconds - start_time_seconds) * 1000000)


def _get_checkpoint_size(prefix):
  # [FUNCTION DOC]
  # HELPER: computes checkpoint size for the write metrics reported by Saver.save.
  # Input is a checkpoint path prefix (string). Finds every file on the local file system
  # whose name starts with the prefix (the index file and all sharded data files) and sums
  # their sizes, obtained from the native file-system metrics helper. Returns the total in
  # bytes (zero when nothing matches). Note the match is purely by string prefix, so unrelated
  # files sharing the prefix are also counted. Only reads file metadata.
  """Calculates filesize of checkpoint based on prefix."""
  size = 0
  # Gather all files beginning with prefix (.index plus sharded data files).
  files = glob.glob("{}*".format(prefix))
  for file in files:
    # Use TensorFlow's C++ FileSystem API.
    size += metrics.CalculateFileSize(file)
  return size


class BaseSaverBuilder:
  # [CLASS DOC]
  # Base builder that constructs the graph operations for saving and restoring variables
  # (save, restore, sharded variants, and assembling a saver definition message). Created with
  # the checkpoint format version to write (default is the version 2 format). Holds only that
  # version as state. Subclasses may override the save, restore and bulk restore hooks to
  # generate different operations. Also exposes aliases of saveable object types for old code.
  """Base class for Savers.

  Can be extended to create different Ops.
  """

  SaveSpec = saveable_object.SaveSpec
  SaveableObject = saveable_object.SaveableObject

  # Aliases for code which was moved but still has lots of users.
  VariableSaveable = saveable_object_util.ReferenceVariableSaveable
  ResourceVariableSaveable = saveable_object_util.ResourceVariableSaveable

  def __init__(self, write_version=saver_pb2.SaverDef.V2):
    # [FUNCTION DOC]
    # Constructor. Takes the checkpoint format version to write (an enum value of the saver
    # definition message, defaulting to the version 2 format) and stores it for later use by
    # the save and sharded-save methods. Returns nothing; no other side effects and no
    # validation of the value.
    self._write_version = write_version

  def save_op(self, filename_tensor, saveables):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the unexpected-version error message joins a string with the version value
    # directly; since the version is an integer enum, building the message itself fails with a
    # type error instead of the documented runtime error.
    # Creates the single save operation for a list of saveable objects. Inputs: a string tensor
    # (a file name for the version 1 format, but a checkpoint path prefix for version 2) and a
    # list of saveable objects. Collects the name, tensor and slice specification of every spec
    # of every saveable, then emits either the legacy save op (version 1) or the version 2 save
    # op. Returns the created operation. Intended to be overridden by subclasses. Raises a
    # runtime error if the configured write version is neither supported value (see above).
    # Side effect: adds ops to the current graph (or executes them eagerly).
    """Create an Op to save 'saveables'.

    This is intended to be overridden by subclasses that want to generate
    different Ops.

    Args:
      filename_tensor: String Tensor.
      saveables: A list of BaseSaverBuilder.SaveableObject objects.

    Returns:
      An Operation that save the variables.

    Raises:
      RuntimeError: (implementation detail) if "self._write_version" is an
        unexpected value.
    """
    # pylint: disable=protected-access
    tensor_names = []
    tensors = []
    tensor_slices = []
    for saveable in saveables:
      for spec in saveable.specs:
        tensor_names.append(spec.name)
        tensors.append(spec.tensor)
        tensor_slices.append(spec.slice_spec)
    if self._write_version == saver_pb2.SaverDef.V1:
      return io_ops._save(
          filename=filename_tensor,
          tensor_names=tensor_names,
          tensors=tensors,
          tensor_slices=tensor_slices)
    elif self._write_version == saver_pb2.SaverDef.V2:
      # "filename_tensor" is interpreted *NOT AS A FILENAME*, but as a prefix
      # of a V2 checkpoint: e.g. "/fs/train/ckpt-<step>/tmp/worker<i>-<step>".
      return io_ops.save_v2(filename_tensor, tensor_names, tensor_slices,
                            tensors)
    else:
      raise RuntimeError("Unexpected write_version: " + self._write_version)

  def bulk_restore(self, filename_tensor, saveables, preferred_shard,
                   restore_sequentially):
    # [FUNCTION DOC]
    # Restores every tensor of every saveable by calling the per-saveable restore hook one
    # saveable at a time. Inputs: a string tensor with the checkpoint path, a list of saveable
    # objects, the shard to open first (passed through to the hook) and a sequential flag that
    # is ignored here. Each restore is placed on the saveable's device, forced to the CPU of
    # that device when one is set, otherwise with no device constraint. Returns a flat list of
    # the restored tensors in saveable and spec order. Subclasses may restore in one call.
    # Side effect: adds restore ops to the graph.
    """Restore all tensors contained in saveables.

    By default, this issues separate calls to `restore_op` for each saveable.
    Subclasses may override to load multiple saveables in a single call.

    Args:
      filename_tensor: String Tensor.
      saveables: List of BaseSaverBuilder.SaveableObject objects.
      preferred_shard: Int.  Shard to open first when loading a sharded file.
      restore_sequentially: Unused.  Bool.  If true, each restore is sequential.

    Returns:
      A list of Tensors resulting from reading 'saveable' from
        'filename'.

    """
    del restore_sequentially
    all_tensors = []
    for saveable in saveables:
      if saveable.device:
        device = saveable_object_util.set_cpu0(saveable.device)
      else:
        device = None
      with ops.device(device):
        all_tensors.extend(
            self.restore_op(filename_tensor, saveable, preferred_shard))
    return all_tensors

  # pylint: disable=unused-argument
  def restore_op(self, filename_tensor, saveable, preferred_shard):
    # [FUNCTION DOC]
    # Creates the restore operations for one saveable. Inputs: a string tensor with the
    # checkpoint path, one saveable object, and a preferred shard number that this default
    # implementation ignores. For each spec of the saveable it issues a version 2 restore of
    # that single named tensor with its slice specification and data type. Returns a list of
    # restored tensors, one per spec. Intended to be overridden. Side effect: adds ops to the
    # graph.
    """Create ops to restore 'saveable'.

    This is intended to be overridden by subclasses that want to generate
    different Ops.

    Args:
      filename_tensor: String Tensor.
      saveable: A BaseSaverBuilder.SaveableObject object.
      preferred_shard: Int.  Shard to open first when loading a sharded file.

    Returns:
      A list of Tensors resulting from reading 'saveable' from
        'filename'.
    """
    # pylint: disable=protected-access
    tensors = []
    for spec in saveable.specs:
      tensors.append(
          io_ops.restore_v2(filename_tensor, [spec.name], [spec.slice_spec],
                            [spec.dtype])[0])

    return tensors

  # pylint: enable=unused-argument

  def sharded_filename(self, filename_tensor, shard, num_shards):
    # [FUNCTION DOC]
    # Builds the per-shard file name. Inputs: a string tensor with the base name or prefix,
    # the shard index (integer) and the total number of shards (integer or integer tensor).
    # Returns a string tensor produced by the native sharded-file-name op, which appends shard
    # information to the base name. Side effect: adds an op to the graph.
    """Append sharding information to a filename.

    Args:
      filename_tensor: A string tensor.
      shard: Integer.  The shard for the filename.
      num_shards: An int Tensor for the number of shards.

    Returns:
      A string tensor.
    """
    return gen_io_ops.sharded_filename(filename_tensor, shard, num_shards)

  def _AddSaveOps(self, filename_tensor, saveables):
    # [FUNCTION DOC]
    # HELPER: used by the build methods to create the save op for one shard or the whole set.
    # Inputs: a string tensor with the file name or prefix, and a list of saveable objects that
    # live on the same shard. Creates the save operation and returns the file name tensor
    # wrapped so that it only becomes available after the save has run. Side effect: adds ops
    # to the graph.
    """Add ops to save variables that are on the same shard.

    Args:
      filename_tensor: String Tensor.
      saveables: A list of SaveableObject objects.

    Returns:
      A tensor with the filename used to save.
    """
    save = self.save_op(filename_tensor, saveables)
    return control_flow_ops.with_dependencies([save], filename_tensor)

  def _AddShardedSaveOpsForV2(self, checkpoint_prefix, per_device):
    # [FUNCTION DOC]
    # HELPER: sharded save path for the version 2 format, used by the sharded save builder.
    # Inputs: a scalar string tensor holding the checkpoint prefix (not a real file name) and
    # a list of pairs of device name and saveable list, one pair per shard, as produced by the
    # device grouping helper. Computes a temporary prefix by appending a suffix that is ".part"
    # for prefixes starting with an S3 scheme, and a temporary directory part path otherwise
    # (evaluated on the CPU at run time). For each shard, builds a sharded temporary file name
    # on that device's CPU and a save op. After all shard saves finish, a merge step on the
    # last shard's device merges the metadata of the shards into the final prefix and deletes
    # the temporary directories. Returns an op that, when evaluated, yields only the original
    # prefix, without any shard suffix. Side effects: adds ops to the graph; at run time writes
    # files and removes the temporary directory. If there are no shards the last device is
    # missing, so an empty device list is not supported.
    """Add ops to save the params per shard, for the V2 format.

    Note that the sharded save procedure for the V2 format is different from
    V1: there is a special "merge" step that merges the small metadata produced
    from each device.

    Args:
      checkpoint_prefix: scalar String Tensor.  Interpreted *NOT AS A FILENAME*,
        but as a prefix of a V2 checkpoint;
      per_device: A list of (device, BaseSaverBuilder.VarToSave) pairs, as
        returned by _GroupByDevices().

    Returns:
      An op to save the variables, which, when evaluated, returns the prefix
        "<user-fed prefix>" only and does not include the sharded spec suffix.
    """
    # IMPLEMENTATION DETAILS: most clients should skip.
    #
    # Suffix for any well-formed "checkpoint_prefix", when sharded.
    # Transformations:
    # * Users pass in "save_path" in save() and restore().  Say "myckpt".
    # * checkpoint_prefix gets fed <save_path><_SHARDED_SUFFIX>.
    # * If checkpoint_prefix is a S3 bucket path ".part" is appended to it
    # * Otherwise _temp/part is appended which is normalized relative to the OS
    # Example:
    #   During runtime, a temporary directory is first created, which contains
    #   files
    #
    #     <train dir>/myckpt_temp/
    #        part-?????-of-?????{.index, .data-00000-of-00001}
    #
    #   Before .save() finishes, they will be (hopefully, atomically) renamed to
    #
    #     <train dir>/
    #        myckpt{.index, .data-?????-of-?????}
    #
    #   Filesystems with eventual consistency (such as S3), don't need a
    #   temporary location. Using a temporary directory in those cases might
    #   cause situations where files are not available during copy.
    #
    # Users only need to interact with the user-specified prefix, which is
    # "<train dir>/myckpt" in this case.  Save() and Restore() work with the
    # prefix directly, instead of any physical pathname.  (On failure and
    # subsequent restore, an outdated and orphaned temporary directory can be
    # safely removed.)
    with ops.device("CPU"):
      _SHARDED_SUFFIX = array_ops.where(
          string_ops.regex_full_match(checkpoint_prefix, "^s3://.*"),
          constant_op.constant(".part"),
          constant_op.constant(os.path.normpath("_temp/part")))
      tmp_checkpoint_prefix = string_ops.string_join(
          [checkpoint_prefix, _SHARDED_SUFFIX])

    num_shards = len(per_device)
    sharded_saves = []
    sharded_prefixes = []
    num_shards_tensor = constant_op.constant(num_shards, name="num_shards")
    last_device = None
    for shard, (device, saveables) in enumerate(per_device):
      last_device = device
      with ops.device(saveable_object_util.set_cpu0(device)):
        sharded_filename = self.sharded_filename(tmp_checkpoint_prefix, shard,
                                                 num_shards_tensor)
        sharded_prefixes.append(sharded_filename)
        sharded_saves.append(self._AddSaveOps(sharded_filename, saveables))

    with ops.control_dependencies([x.op for x in sharded_saves]):
      # Co-locates the merge step with the last device.
      with ops.device(saveable_object_util.set_cpu0(last_device)):
        # V2 format write path consists of a metadata merge step.  Once merged,
        # attempts to delete the temporary directory, "<user-fed prefix>_temp".
        merge_step = gen_io_ops.merge_v2_checkpoints(
            sharded_prefixes, checkpoint_prefix, delete_old_dirs=True)
        with ops.control_dependencies([merge_step]):
          # Returns the prefix "<user-fed prefix>" only.  DOES NOT include the
          # sharded spec suffix.
          return array_ops.identity(checkpoint_prefix)

  def _AddShardedSaveOps(self, filename_tensor, per_device):
    # [FUNCTION DOC]
    # HELPER: dispatches sharded saving by checkpoint format version.
    # Inputs: a scalar string tensor (name or prefix) and a list of pairs of device and
    # saveables, one per shard. For the version 2 format it delegates to the version 2 sharded
    # helper. For version 1 it creates one save op per shard on that shard's device, using a
    # sharded file name, and returns the sharded file specification op that runs only after all
    # shard saves. Side effect: adds ops to the graph.
    """Add ops to save the params per shard.

    Args:
      filename_tensor: a scalar String Tensor.
      per_device: A list of (device, BaseSaverBuilder.SaveableObject) pairs, as
        returned by _GroupByDevices().

    Returns:
      An op to save the variables.
    """
    if self._write_version == saver_pb2.SaverDef.V2:
      return self._AddShardedSaveOpsForV2(filename_tensor, per_device)

    num_shards = len(per_device)
    sharded_saves = []
    num_shards_tensor = constant_op.constant(num_shards, name="num_shards")
    for shard, (device, saveables) in enumerate(per_device):
      with ops.device(device):
        sharded_filename = self.sharded_filename(filename_tensor, shard,
                                                 num_shards_tensor)
        sharded_saves.append(self._AddSaveOps(sharded_filename, saveables))
    # Return the sharded name for the save path.
    with ops.control_dependencies([x.op for x in sharded_saves]):
      return gen_io_ops.sharded_filespec(filename_tensor, num_shards_tensor)

  def _AddRestoreOps(self,
                     filename_tensor,
                     saveables,
                     restore_sequentially,
                     reshape,
                     preferred_shard=-1,
                     name="restore_all"):
    # [FUNCTION DOC]
    # HELPER: creates the restore half of the saver graph for a set of saveables.
    # Inputs: tensor with the checkpoint path, list of saveable objects, a flag to restore
    # sequentially, a flag to let the restore reshape loaded values to the variables' shapes,
    # the preferred shard (default minus one, meaning none) and the name of the returned op
    # (default restore_all). Restores all tensors through the bulk restore hook, then slices
    # the flat result per saveable and asks each saveable to build its assignment, passing the
    # target shapes when reshaping (the static shape if fully defined, otherwise a dynamic
    # shape tensor). Returns a grouped no-op that depends on all assignments. Side effect:
    # adds ops to the graph.
    """Add operations to restore saveables.

    Args:
      filename_tensor: Tensor for the path of the file to load.
      saveables: A list of SaveableObject objects.
      restore_sequentially: True if we want to restore variables sequentially
        within a shard.
      reshape: True if we want to reshape loaded tensors to the shape of the
        corresponding variable.
      preferred_shard: Shard to open first when loading a sharded file.
      name: Name for the returned op.

    Returns:
      An Operation that restores the variables.
    """
    all_tensors = self.bulk_restore(filename_tensor, saveables, preferred_shard,
                                    restore_sequentially)

    assign_ops = []
    idx = 0
    # Load and optionally reshape on the CPU, as string tensors are not
    # available on the GPU.
    # TODO(touts): Re-enable restore on GPU when we can support annotating
    # string tensors as "HostMemory" inputs.
    for saveable in saveables:
      shapes = None
      if reshape:
        # Compute the shapes, let the restore op decide if and how to do
        # the reshape.
        shapes = []
        for spec in saveable.specs:
          v = spec.tensor
          shape = v.get_shape()
          if not shape.is_fully_defined():
            shape = array_ops.shape(v)
          shapes.append(shape)
      saveable_tensors = all_tensors[idx:idx + len(saveable.specs)]
      idx += len(saveable.specs)
      assign_ops.append(saveable.restore(saveable_tensors, shapes))

    # Create a Noop that has control dependencies from all the updates.
    return control_flow_ops.group(*assign_ops, name=name)

  def _AddShardedRestoreOps(self, filename_tensor, per_device,
                            restore_sequentially, reshape):
    # [FUNCTION DOC]
    # HELPER: sharded restore for the builder. Inputs: checkpoint path tensor, list of pairs of
    # device and saveables (one per shard), the sequential flag and the reshape flag. For each
    # shard, on that shard's device, builds a restore group named restore_shard with that
    # shard as the preferred one. Returns one grouped op named restore_all depending on every
    # shard restore. Side effect: adds ops to the graph.
    """Add Ops to restore variables from multiple devices.

    Args:
      filename_tensor: Tensor for the path of the file to load.
      per_device: A list of (device, SaveableObject) pairs, as returned by
        _GroupByDevices().
      restore_sequentially: True if we want to restore variables sequentially
        within a shard.
      reshape: True if we want to reshape loaded tensors to the shape of the
        corresponding variable.

    Returns:
      An Operation that restores the variables.
    """
    sharded_restores = []
    for shard, (device, saveables) in enumerate(per_device):
      with ops.device(device):
        sharded_restores.append(
            self._AddRestoreOps(
                filename_tensor,
                saveables,
                restore_sequentially,
                reshape,
                preferred_shard=shard,
                name="restore_shard"))
    return control_flow_ops.group(*sharded_restores, name="restore_all")

  def _GroupByDevices(self, saveables):
    # [FUNCTION DOC]
    # HELPER: partitions saveables by device for the sharded save and restore.
    # Input: a list of saveable objects. For each, computes the canonical device names of all
    # its specs; they must agree. Returns a list of pairs of canonical device name and list of
    # saveables, sorted by device name. Raises a value error naming the saveable when its specs
    # sit on different devices. Note the docstring mentions a TODO about duplicate devices:
    # unspecified devices all collapse into one group. No side effects.
    """Group Variable tensor slices per device.

    TODO(touts): Make sure that all the devices found are on different
    job/replica/task/cpu|gpu.  It would be bad if 2 were on the same device.
    It can happen if the devices are unspecified.

    Args:
      saveables: A list of BaseSaverBuilder.SaveableObject objects.

    Returns:
      A list of tuples: (device_name, BaseSaverBuilder.SaveableObject) tuples.
      The list is sorted by ascending device_name.

    Raises:
      ValueError: If the tensors of a saveable are on different devices.
    """
    per_device = collections.defaultdict(lambda: [])
    for saveable in saveables:
      canonical_device = set(
          pydev.canonical_name(spec.device) for spec in saveable.specs)
      if len(canonical_device) != 1:
        raise ValueError("All tensors of a saveable object must be "
                         "on the same device: %s" % saveable.name)
      per_device[canonical_device.pop()].append(saveable)
    return sorted(per_device.items(), key=lambda t: t[0])

  def build(self,
            names_to_saveables,
            reshape=False,
            sharded=False,
            max_to_keep=5,
            keep_checkpoint_every_n_hours=10000.0,
            name=None,
            restore_sequentially=False,
            filename="model"):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring lists a type error for non-dictionary input, but the internal
    # builder converts a list into a dictionary instead of raising; uniqueness errors come from
    # the validation helper.
    # Public entry point that builds the save and restore graph (or runs save and restore in
    # eager mode) and returns a saver definition message. Takes a mapping from checkpoint names
    # to variables or saveable objects, the reshape flag, the sharded flag, the maximum number
    # of checkpoints to keep (none or zero means keep all files), the retention interval in
    # hours (default 10000), an optional name scope, the sequential restore flag and the file
    # name (default model). Simply forwards all arguments to the internal builder with both
    # save and restore enabled and returns its result.
    """Builds save/restore graph nodes or runs save/restore in eager mode.

    Args:
      names_to_saveables: A dictionary mapping name to a Variable or
        SaveableObject. Each name will be associated with the corresponding
        variable in the checkpoint.
      reshape: If True, allow restoring parameters from a checkpoint that where
        the parameters have a different shape.  This is only needed when you try
        to restore from a Dist-Belief checkpoint, and only some times.
      sharded: If True, shard the checkpoints, one per device that has Variable
        nodes.
      max_to_keep: Maximum number of checkpoints to keep.  As new checkpoints
        are created, old ones are deleted.  If None or 0, no checkpoints are
        deleted from the filesystem but only the last one is kept in the
        `checkpoint` file.  Presently the number is only roughly enforced.  For
        example in case of restarts more than max_to_keep checkpoints may be
        kept.
      keep_checkpoint_every_n_hours: How often checkpoints should be kept.
        Defaults to 10,000 hours.
      name: String.  Optional name to use as a prefix when adding operations.
      restore_sequentially: A Bool, which if true, causes restore of different
        variables to happen sequentially within each device.
      filename: If known at graph construction time, filename used for variable
        loading/saving. If None, then the default name "model" will be used.

    Returns:
      A SaverDef proto.

    Raises:
      TypeError: If 'names_to_saveables' is not a dictionary mapping string
        keys to variable Tensors.
      ValueError: If any of the keys or values in 'names_to_saveables' is not
        unique.
    """
    return self._build_internal(
        names_to_saveables=names_to_saveables,
        reshape=reshape,
        sharded=sharded,
        max_to_keep=max_to_keep,
        keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
        name=name,
        restore_sequentially=restore_sequentially,
        filename=filename)

  def _build_internal(self,
                      names_to_saveables,
                      reshape=False,
                      sharded=False,
                      max_to_keep=5,
                      keep_checkpoint_every_n_hours=10000.0,
                      name=None,
                      restore_sequentially=False,
                      filename="model",
                      build_save=True,
                      build_restore=True):
    # [FUNCTION DOC]
    # HELPER: shared implementation behind the public build method, with switches to build only
    # the save half or only the restore half (allowed only in eager mode).
    # Inputs: mapping from names to variables or saveables (a list is converted to a mapping),
    # reshape, sharded, maximum checkpoints to keep (none becomes zero), retention hours, name
    # scope, sequential restore, file name (default model; a falsy value falls back to model),
    # and the two build switches. Validates and slices the inputs into saveables, then under a
    # name scope creates a filename placeholder with default value (and a second one named
    # Const for backward compatibility), and the save and restore ops, sharded per device if
    # requested. In eager mode returns a saver definition whose tensor names are the evaluated
    # values (the restore op name is empty). In graph mode first makes sure that every
    # partitioned variable stored in any graph collection has its concat op (creating it when
    # its operation is missing), then returns a saver definition with the names of the created
    # tensors and ops.
    # Raises a value error if only one half is requested in graph mode, plus errors from the
    # input validation. Side effects: adds ops to the default graph.
    # KNOWN ISSUE: the commented-out assertion on the restore op name was disabled; also if
    # build_save is false in graph mode the save tensor is never defined (guarded by the first
    # check).
    """build() with option to only perform save and restore."""
    if not context.executing_eagerly() and (not build_save or
                                            not build_restore):
      raise ValueError("save and restore operations need to be built together "
                       " when eager execution is not enabled.")

    if not isinstance(names_to_saveables, dict):
      names_to_saveables = saveable_object_util.op_list_to_dict(
          names_to_saveables)
    saveables = saveable_object_util.validate_and_slice_inputs(
        names_to_saveables)
    if max_to_keep is None:
      max_to_keep = 0

    with ops.name_scope(name, "save",
                        [saveable.op for saveable in saveables]) as name:
      # Add a placeholder string tensor for the filename.
      filename_tensor = array_ops.placeholder_with_default(
          filename or "model", shape=(), name="filename")
      # Keep the name "Const" for backwards compatibility.
      filename_tensor = array_ops.placeholder_with_default(
          filename_tensor, shape=(), name="Const")

      # Add the save ops.
      if sharded:
        per_device = self._GroupByDevices(saveables)
        if build_save:
          save_tensor = self._AddShardedSaveOps(filename_tensor, per_device)
        if build_restore:
          restore_op = self._AddShardedRestoreOps(filename_tensor, per_device,
                                                  restore_sequentially, reshape)
      else:
        if build_save:
          save_tensor = self._AddSaveOps(filename_tensor, saveables)
        if build_restore:
          restore_op = self._AddRestoreOps(filename_tensor, saveables,
                                           restore_sequentially, reshape)

    # In the following use case, it's possible to have restore_ops be called
    # something else:
    # - Build inference graph and export a meta_graph.
    # - Import the inference meta_graph
    # - Extend the inference graph to a train graph.
    # - Export a new meta_graph.
    # Now the second restore_op will be called "restore_all_1".
    # As such, comment out the assert for now until we know whether supporting
    # such usage model makes sense.
    #
    # assert restore_op.name.endswith("restore_all"), restore_op.name
    if context.executing_eagerly():
      # Store the tensor values to the tensor_names.
      save_tensor_name = save_tensor.numpy() if build_save else ""
      return saver_pb2.SaverDef(
          filename_tensor_name=filename_tensor.numpy(),
          save_tensor_name=save_tensor_name,
          restore_op_name="",
          max_to_keep=max_to_keep,
          sharded=sharded,
          keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
          version=self._write_version)
    else:
      graph = ops.get_default_graph()
      # Do some sanity checking on collections containing
      # PartitionedVariables. If a saved collection has a PartitionedVariable,
      # the GraphDef needs to include concat ops to get the value (or there'll
      # be a lookup error on load).
      check_collection_list = graph.get_all_collection_keys()
      for collection_type in check_collection_list:
        for element in graph.get_collection(collection_type):
          if isinstance(element, variables.PartitionedVariable):
            try:
              graph.get_operation_by_name(element.name)
            except KeyError:
              # Create a concat op for this PartitionedVariable. The user may
              # not need it, but we'll try looking it up on MetaGraph restore
              # since it's in a collection.
              element.as_tensor()
      return saver_pb2.SaverDef(
          filename_tensor_name=filename_tensor.name,
          save_tensor_name=save_tensor.name,
          restore_op_name=restore_op.name,
          max_to_keep=max_to_keep,
          sharded=sharded,
          keep_checkpoint_every_n_hours=keep_checkpoint_every_n_hours,
          version=self._write_version)


class BulkSaverBuilder(BaseSaverBuilder):
  # [CLASS DOC]
  # Saver builder variant that restores all saveables with one combined restore operation
  # instead of one per saveable. Inherits everything else from the base builder, overriding only
  # the bulk restore hook. No extra state.
  """SaverBuilder with support for bulk restoring multiple saveables."""

  def bulk_restore(self, filename_tensor, saveables, preferred_shard,
                   restore_sequentially):
    # [FUNCTION DOC]
    # Overrides the base behavior to restore everything in one call. Inputs: checkpoint path
    # tensor, list of saveables, preferred shard and sequential flag (the last two are ignored;
    # the single restore is internally sequential). Gathers the name, slice specification and
    # data type of every spec of every saveable, then on the first CPU issues a single
    # version 2 restore for all of them. Returns the list of restored tensors in spec order.
    # Fails on unpacking if there are no specs at all. Side effect: adds one op to the graph.

    # Ignored: bulk restore is internally sequential.
    del restore_sequentially
    restore_specs = []
    for saveable in saveables:
      for spec in saveable.specs:
        restore_specs.append((spec.name, spec.slice_spec, spec.dtype))

    names, slices, dtypes = zip(*restore_specs)
    # Load all tensors onto CPU 0 for compatibility with existing code.
    with ops.device("cpu:0"):
      return io_ops.restore_v2(filename_tensor, names, slices, dtypes)


def _get_saver_or_default():
  # [FUNCTION DOC]
  # HELPER: used by the training scaffold and checkpoint hooks to share one Saver per graph.
  # KNOWN ISSUE: the check that the created saver is not null is always true since the
  # constructor never returns null.
  # Looks in the savers collection of the default graph. Returns the single saver found. If
  # none exists, creates a sharded Saver that tolerates having no variables, adds it to the
  # collection and returns it. Raises a runtime error if more than one saver is registered.
  # Side effect: may add to the graph collection.
  """Returns the saver from SAVERS collection, or creates a default one.

  This method is used by other members of the training module, such as
  `Scaffold`, or `CheckpointSaverHook`.

  Returns:
    `Saver`.

  Raises:
    RuntimeError: If the SAVERS collection already has more than one items.
  """
  collection_key = ops.GraphKeys.SAVERS
  savers = ops.get_collection(collection_key)
  if savers:
    if len(savers) > 1:
      raise RuntimeError(
          "More than one item in collection {}. "
          "Please indicate which one to use by passing it to the constructor."
          .format(collection_key))
    return savers[0]
  saver = Saver(sharded=True, allow_empty=True)
  if saver is not None:
    ops.add_to_collection(collection_key, saver)
  return saver


@tf_export(v1=["train.Saver"])
class Saver:
  # [CLASS DOC]
  # Legacy name-based checkpoint saver, exported as the version 1 train Saver. Adds save and
  # restore ops for a set of variables or saveable objects to the graph and offers methods to
  # run them in a session, to manage a rolling list of recent checkpoints (deleting old ones
  # beyond a maximum, optionally keeping one per N hours), to maintain the checkpoint state
  # file, and to export or convert the saver definition message. State: the variable list, the
  # builder, the saver definition message, flags (reshape, sharded, sequential restore, built,
  # empty, relative paths, step padding), the checkpoint format version, the list of recent
  # checkpoints as pairs of path and timestamp, the list pending deletion, the next
  # keep-forever time, and an optional second saver used for object-based checkpoints. Created
  # directly by user code, or from a saver definition. Deprecated in favor of the object-based
  # checkpoint API; works only partially in eager mode.
  # pylint: disable=line-too-long
  """Saves and restores variables.

  @compatibility(TF2)
  `tf.compat.v1.train.Saver` is not supported for saving and restoring
  checkpoints in TF2. Please switch to `tf.train.Checkpoint` or
  `tf.keras.Model.save_weights`, which perform a more robust [object-based
  saving](https://www.tensorflow.org/guide/checkpoint#loading_mechanics).

  ### How to Rewrite Checkpoints

  Please rewrite your checkpoints immediately using the object-based checkpoint
  APIs.

  You can load a name-based checkpoint written by `tf.compat.v1.train.Saver`
  using `tf.train.Checkpoint.restore` or `tf.keras.Model.load_weights`. However,
  you may have to change the names of the variables in your model to match the
  variable names in the name-based checkpoint, which can be viewed with
  `tf.train.list_variables(path)`.

  Another option is to create an `assignment_map` that maps the name of the
  variables in the name-based checkpoint to the variables in your model, eg:
  ```
  {
      'sequential/dense/bias': model.variables[0],
      'sequential/dense/kernel': model.variables[1]
  }
  ```
  and use `tf.compat.v1.train.init_from_checkpoint(path, assignment_map)` to
  restore the name-based checkpoint.

  After restoring, re-encode your checkpoint
  using `tf.train.Checkpoint.save` or `tf.keras.Model.save_weights`.

  See the [Checkpoint compatibility](
  https://www.tensorflow.org/guide/migrate#checkpoint_compatibility)
  section of the migration guide for more details.


  ### Checkpoint Management in TF2

  Use `tf.train.CheckpointManager` to manage checkpoints in TF2.
  `tf.train.CheckpointManager` offers equivalent `keep_checkpoint_every_n_hours`
  and `max_to_keep` parameters.

  To recover the latest checkpoint,

  ```
  checkpoint = tf.train.Checkpoint(model)
  manager = tf.train.CheckpointManager(checkpoint)
  status = checkpoint.restore(manager.latest_checkpoint)
  ```

  `tf.train.CheckpointManager` also writes a [`CheckpointState` proto]
  (https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/training/checkpoint_state.proto)
  which contains the timestamp when each checkpoint was created.

  ### Writing `MetaGraphDef`s in TF2

  To replace, `tf.compat.v1.train.Saver.save(write_meta_graph=True)`, use
  `tf.saved_model.save` to write the `MetaGraphDef` (which is contained in
  `saved_model.pb`).

  @end_compatibility

  See [Variables](https://tensorflow.org/guide/variables)
  for an overview of variables, saving and restoring.

  The `Saver` class adds ops to save and restore variables to and from
  *checkpoints*.  It also provides convenience methods to run these ops.

  Checkpoints are binary files in a proprietary format which map variable names
  to tensor values.  The best way to examine the contents of a checkpoint is to
  load it using a `Saver`.

  Savers can automatically number checkpoint filenames with a provided counter.
  This lets you keep multiple checkpoints at different steps while training a
  model.  For example you can number the checkpoint filenames with the training
  step number.  To avoid filling up disks, savers manage checkpoint files
  automatically. For example, they can keep only the N most recent files, or
  one checkpoint for every N hours of training.

  You number checkpoint filenames by passing a value to the optional
  `global_step` argument to `save()`:

  ```python
  saver.save(sess, 'my-model', global_step=0) ==> filename: 'my-model-0'
  ...
  saver.save(sess, 'my-model', global_step=1000) ==> filename: 'my-model-1000'
  ```

  Additionally, optional arguments to the `Saver()` constructor let you control
  the proliferation of checkpoint files on disk:

  * `max_to_keep` indicates the maximum number of recent checkpoint files to
    keep.  As new files are created, older files are deleted.   If None or 0,
    no checkpoints are deleted from the filesystem but only the last one is
    kept in the `checkpoint` file.  Defaults to 5 (that is, the 5 most recent
    checkpoint files are kept.)

  * `keep_checkpoint_every_n_hours`: In addition to keeping the most recent
    `max_to_keep` checkpoint files, you might want to keep one checkpoint file
    for every N hours of training.  This can be useful if you want to later
    analyze how a model progressed during a long training session.  For
    example, passing `keep_checkpoint_every_n_hours=2` ensures that you keep
    one checkpoint file for every 2 hours of training.  The default value of
    10,000 hours effectively disables the feature.

  Note that you still have to call the `save()` method to save the model.
  Passing these arguments to the constructor will not save variables
  automatically for you.

  A training program that saves regularly looks like:

  ```python
  ...
  # Create a saver.
  saver = tf.compat.v1.train.Saver(...variables...)
  # Launch the graph and train, saving the model every 1,000 steps.
  sess = tf.compat.v1.Session()
  for step in range(1000000):
      sess.run(..training_op..)
      if step % 1000 == 0:
          # Append the step number to the checkpoint name:
          saver.save(sess, 'my-model', global_step=step)
  ```

  In addition to checkpoint files, savers keep a protocol buffer on disk with
  the list of recent checkpoints. This is used to manage numbered checkpoint
  files and by `latest_checkpoint()`, which makes it easy to discover the path
  to the most recent checkpoint. That protocol buffer is stored in a file named
  'checkpoint' next to the checkpoint files.

  If you create several savers, you can specify a different filename for the
  protocol buffer file in the call to `save()`.
  """

  # pylint: enable=line-too-long

  def __init__(self,
               var_list=None,
               reshape=False,
               sharded=False,
               max_to_keep=5,
               keep_checkpoint_every_n_hours=10000.0,
               name=None,
               restore_sequentially=False,
               saver_def=None,
               builder=None,
               defer_build=False,
               allow_empty=False,
               write_version=saver_pb2.SaverDef.V2,
               pad_step_number=False,
               save_relative_paths=False,
               filename=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says a type error is raised for an invalid variable list, but
    # this constructor never raises it itself; it only comes (if at all) from the builder.
    # Also the docstring says a saver with no variables raises unless empty is allowed, which
    # holds only in graph mode when the build is not deferred.
    # Constructor. Inputs: var_list (list of variables or saveables, or mapping from names to
    # them; none means all saveable objects in the graph), reshape, sharded, maximum
    # checkpoints to keep (default 5), retention hours (default 10000), name scope, sequential
    # restore flag, an existing saver definition message to reuse, a builder (default: the
    # bulk builder), defer-build flag, allow-empty flag, write format version (default 2),
    # step-number padding flag, relative-paths flag, and an optional file name.
    # On the first Saver created, records the current time in the module-level last-write
    # timestamp under its lock. Raises a value error if building is deferred while a variable
    # list is given, and a runtime error in eager mode when no variable list is given (eager
    # mode also logs a deprecation warning and computes the next keep-forever time instead of
    # building). In graph mode, unless deferred, builds the save and restore ops immediately.
    # If a saver definition is present it is validated and its version overrides the given
    # write version. Side effects: adds ops to the graph, logs, sets module-level state.
    """Creates a `Saver`.

    The constructor adds ops to save and restore variables.

    `var_list` specifies the variables that will be saved and restored. It can
    be passed as a `dict` or a list:

    * A `dict` of names to variables: The keys are the names that will be
      used to save or restore the variables in the checkpoint files.
    * A list of variables: The variables will be keyed with their op name in
      the checkpoint files.

    For example:

    ```python
    v1 = tf.Variable(..., name='v1')
    v2 = tf.Variable(..., name='v2')

    # Pass the variables as a dict:
    saver = tf.compat.v1.train.Saver({'v1': v1, 'v2': v2})

    # Or pass them as a list.
    saver = tf.compat.v1.train.Saver([v1, v2])
    # Passing a list is equivalent to passing a dict with the variable op names
    # as keys:
    saver = tf.compat.v1.train.Saver({v.op.name: v for v in [v1, v2]})
    ```

    Note: the newer `AutoTrackable` API is not supported by `Saver`. In this
    case, the `tf.train.Checkpoint` class should be used.

    The optional `reshape` argument, if `True`, allows restoring a variable from
    a save file where the variable had a different shape, but the same number
    of elements and type.  This is useful if you have reshaped a variable and
    want to reload it from an older checkpoint.

    The optional `sharded` argument, if `True`, instructs the saver to shard
    checkpoints per device.

    Args:
      var_list: A list of `Variable`/`SaveableObject`, or a dictionary mapping
        names to `SaveableObject`s. If `None`, defaults to the list of all
        saveable objects.
      reshape: If `True`, allows restoring parameters from a checkpoint where
        the variables have a different shape.
      sharded: If `True`, shard the checkpoints, one per device.
      max_to_keep: Maximum number of recent checkpoints to keep. Defaults to 5.
      keep_checkpoint_every_n_hours: How often to keep checkpoints. Defaults to
        10,000 hours.
      name: String.  Optional name to use as a prefix when adding operations.
      restore_sequentially: A `Bool`, which if true, causes restore of different
        variables to happen sequentially within each device.  This can lower
        memory usage when restoring very large models.
      saver_def: Optional `SaverDef` proto to use instead of running the
        builder. This is only useful for specialty code that wants to recreate a
        `Saver` object for a previously built `Graph` that had a `Saver`. The
        `saver_def` proto should be the one returned by the `as_saver_def()`
        call of the `Saver` that was created for that `Graph`.
      builder: Optional `SaverBuilder` to use if a `saver_def` was not provided.
        Defaults to `BulkSaverBuilder()`.
      defer_build: If `True`, defer adding the save and restore ops to the
        `build()` call. In that case `build()` should be called before
        finalizing the graph or using the saver.
      allow_empty: If `False` (default) raise an error if there are no variables
        in the graph. Otherwise, construct the saver anyway and make it a no-op.
      write_version: controls what format to use when saving checkpoints.  It
        also affects certain filepath matching logic.  The V2 format is the
        recommended choice: it is much more optimized than V1 in terms of memory
        required and latency incurred during restore.  Regardless of this flag,
        the Saver is able to restore from both V2 and V1 checkpoints.
      pad_step_number: if True, pads the global step number in the checkpoint
        filepaths to some fixed width (8 by default).  This is turned off by
        default.
      save_relative_paths: If `True`, will write relative paths to the
        checkpoint state file. This is needed if the user wants to copy the
        checkpoint directory and reload from the copied directory.
      filename: If known at graph construction time, filename used for variable
        loading/saving.

    Raises:
      TypeError: If `var_list` is invalid.
      ValueError: If any of the keys or values in `var_list` are not unique.
      RuntimeError: If eager execution is enabled and`var_list` does not specify
        a list of variables to save.

    @compatibility(eager)
    When eager execution is enabled, `var_list` must specify a `list` or `dict`
    of variables to save. Otherwise, a `RuntimeError` will be raised.

    Although Saver works in some cases when executing eagerly, it is
    fragile. Please switch to `tf.train.Checkpoint` or
    `tf.keras.Model.save_weights`, which perform a more robust object-based
    saving. These APIs will load checkpoints written by `Saver`.
    @end_compatibility
    """
    global _END_TIME_OF_LAST_WRITE
    with _END_TIME_OF_LAST_WRITE_LOCK:
      if _END_TIME_OF_LAST_WRITE is None:
        _END_TIME_OF_LAST_WRITE = time.time()

    if defer_build and var_list:
      raise ValueError(
          "If `var_list` is provided then build cannot be deferred. "
          "Either set defer_build=False or var_list=None.")
    if context.executing_eagerly():
      logging.warning(
          "Saver is deprecated, please switch to tf.train.Checkpoint or "
          "tf.keras.Model.save_weights for training checkpoints. When "
          "executing eagerly variables do not necessarily have unique names, "
          "and so the variable.name-based lookups Saver performs are "
          "error-prone.")
      if var_list is None:
        raise RuntimeError(
            "When eager execution is enabled, `var_list` must specify a list "
            "or dict of variables to save")
    self._var_list = var_list
    self._reshape = reshape
    self._sharded = sharded
    self._max_to_keep = max_to_keep
    self._keep_checkpoint_every_n_hours = keep_checkpoint_every_n_hours
    self._name = name
    self._restore_sequentially = restore_sequentially
    self.saver_def = saver_def
    self._builder = builder
    self._is_built = False
    self._allow_empty = allow_empty
    self._is_empty = None
    self._write_version = write_version
    self._pad_step_number = pad_step_number
    self._filename = filename
    self._last_checkpoints = []
    self._checkpoints_to_be_deleted = []
    if context.executing_eagerly():
      self._next_checkpoint_time = (
          time.time() + self._keep_checkpoint_every_n_hours * 3600)
    elif not defer_build:
      self.build()
    if self.saver_def:
      self._check_saver_def()
      self._write_version = self.saver_def.version
    self._save_relative_paths = save_relative_paths
    # For compatibility with object-based checkpoints, we may build a second
    # Saver to read the renamed keys.
    self._object_restore_saver = None

  def build(self):
    # [FUNCTION DOC]
    # Builds the save and restore ops now, for savers created with a deferred build. Takes no
    # inputs and returns nothing. Raises a runtime error in eager mode. Otherwise runs the
    # internal build using the stored file name with both save and restore enabled; building
    # twice is harmless since an already built saver returns early. Side effect: adds ops to
    # the graph and sets the saver definition.
    if context.executing_eagerly():
      raise RuntimeError("Use save/restore instead of build in eager mode.")
    self._build(self._filename, build_save=True, build_restore=True)

  def _build_eager(self, checkpoint_path, build_save, build_restore):
    # [FUNCTION DOC]
    # HELPER: eager-mode entry used by save and restore to run the build with a concrete
    # checkpoint path and the chosen halves. Simply forwards the path and the two switches to
    # the internal build and returns nothing.
    self._build(
        checkpoint_path, build_save=build_save, build_restore=build_restore)

  def _build(self, checkpoint_path, build_save, build_restore):
    # [FUNCTION DOC]
    # HELPER: core builder of the saver definition. Inputs: file name to use (may be none), and
    # switches for building the save and restore halves. In graph mode returns immediately if
    # already built, otherwise marks it built. If there is no saver definition (or in eager
    # mode) it creates the default bulk builder if none was given, defaults the variable list
    # to all saveable objects, and handles an empty list by either marking the saver empty and
    # returning (when allowed) or raising a value error saying there are no variables to save.
    # Otherwise it asks the builder to produce the saver definition. If a definition was
    # supplied together with a name, it prefixes the three tensor/op names in it with that
    # name scope (the name doubles as an import scope). Finally validates the definition and,
    # in graph mode, computes the next keep-forever time. Returns nothing; sets state.
    """Builds saver_def."""
    if not context.executing_eagerly():
      if self._is_built:
        return
      self._is_built = True

    if not self.saver_def or context.executing_eagerly():
      if self._builder is None:
        self._builder = BulkSaverBuilder(self._write_version)

      if self._var_list is None:
        # pylint: disable=protected-access
        self._var_list = variables._all_saveable_objects()
      if not self._var_list:
        if self._allow_empty:
          self._is_empty = True
          return
        else:
          raise ValueError("No variables to save")
      self._is_empty = False

      self.saver_def = self._builder._build_internal(  # pylint: disable=protected-access
          self._var_list,
          reshape=self._reshape,
          sharded=self._sharded,
          max_to_keep=self._max_to_keep,
          keep_checkpoint_every_n_hours=self._keep_checkpoint_every_n_hours,
          name=self._name,
          restore_sequentially=self._restore_sequentially,
          filename=checkpoint_path,
          build_save=build_save,
          build_restore=build_restore)
    elif self.saver_def and self._name:
      # Since self._name is used as a name_scope by builder(), we are
      # overloading the use of this field to represent the "import_scope" as
      # well.
      self.saver_def.filename_tensor_name = ops.prepend_name_scope(
          self.saver_def.filename_tensor_name, self._name)
      self.saver_def.save_tensor_name = ops.prepend_name_scope(
          self.saver_def.save_tensor_name, self._name)
      self.saver_def.restore_op_name = ops.prepend_name_scope(
          self.saver_def.restore_op_name, self._name)

    self._check_saver_def()
    if not context.executing_eagerly():
      # Updates next checkpoint time.
      # Set in __init__ when executing eagerly.
      self._next_checkpoint_time = (
          time.time() + self.saver_def.keep_checkpoint_every_n_hours * 3600)

  def _check_saver_def(self):
    # [FUNCTION DOC]
    # HELPER: validates the saver definition. Raises a value error if it is not a saver
    # definition message; in graph mode also raises a value error if the save tensor name or
    # the restore op name is empty. Returns nothing; no side effects.
    if not isinstance(self.saver_def, saver_pb2.SaverDef):
      raise ValueError("saver_def must be a saver_pb2.SaverDef: %s" %
                       self.saver_def)
    if not context.executing_eagerly():
      if not self.saver_def.save_tensor_name:
        raise ValueError("saver_def must specify the save_tensor_name: %s" %
                         str(self.saver_def))
      if not self.saver_def.restore_op_name:
        raise ValueError("saver_def must specify the restore_op_name: %s" %
                         str(self.saver_def))

  def _CheckpointFilename(self, p):
    # [FUNCTION DOC]
    # HELPER: extracts the path from a recorded checkpoint entry. Input is a pair of file name
    # and timestamp; returns the file name. No side effects.
    """Returns the checkpoint filename given a `(filename, time)` pair.

    Args:
      p: (filename, time) pair.

    Returns:
      Checkpoint file name.
    """
    name, _ = p
    return name

  def _RecordLastCheckpoint(self, latest_save_path):
    # [FUNCTION DOC]
    # HELPER: bookkeeping after a successful save. Input: the path just written. Does nothing
    # if the maximum to keep is zero. Otherwise removes any older entry with the same path,
    # appends the path with the current time, and if the list exceeds the maximum moves the
    # oldest entry to the list of checkpoints pending deletion. Returns nothing; mutates
    # saver state.
    """Manages the list of the latest checkpoints."""
    if not self.saver_def.max_to_keep:
      return
    # Remove first from list if the same name was used before.
    for p in self._last_checkpoints[:]:
      if latest_save_path == self._CheckpointFilename(p):
        self._last_checkpoints.remove(p)

    # Append new path to list
    self._last_checkpoints.append((latest_save_path, time.time()))

    # If more than max_to_keep, remove oldest.
    if len(self._last_checkpoints) > self.saver_def.max_to_keep:
      self._checkpoints_to_be_deleted.append(self._last_checkpoints.pop(0))

  def _MaybeDeleteOldCheckpoints(self, meta_graph_suffix="meta"):
    # [FUNCTION DOC]
    # HELPER: retention policy applied after each save. Input: suffix of the meta graph file
    # (default meta). Takes at most one entry from the pending-deletion list (the oldest). If
    # its timestamp is later than the next keep-forever time, it is kept and that time is
    # advanced by the retention interval; otherwise the checkpoint files, including the meta
    # graph file, are removed through the checkpoint management helper. Any failure while
    # deleting is logged as a warning and ignored. Returns nothing. Note only one entry is
    # processed per call. Side effects: deletes files, logs.
    """Deletes old checkpoints if necessary.

    `self._checkpoints_to_be_deleted` is going to contain checkpoints that are
    over `max_to_keep`.  They are going to be deleted.  If
    `keep_checkpoint_every_n_hours` was specified, keep an additional checkpoint
    every `N` hours. For example, if `N` is 0.5, an additional checkpoint is
    kept for every 0.5 hours of training; if `N` is 10, an additional
    checkpoint is kept for every 10 hours of training.

    Args:
      meta_graph_suffix: Suffix for `MetaGraphDef` file. Defaults to 'meta'.
    """
    if self._checkpoints_to_be_deleted:
      p = self._checkpoints_to_be_deleted.pop(0)
      # Do not delete the file if we keep_checkpoint_every_n_hours is set and we
      # have reached N hours of training.
      should_keep = p[1] > self._next_checkpoint_time
      if should_keep:
        self._next_checkpoint_time += (
            self.saver_def.keep_checkpoint_every_n_hours * 3600)
        return

      # Otherwise delete the files.
      try:
        checkpoint_management.remove_checkpoint(
            self._CheckpointFilename(p), self.saver_def.version,
            meta_graph_suffix)
      except Exception as e:  # pylint: disable=broad-except
        logging.warning("Ignoring: %s", str(e))

  def as_saver_def(self):
    # [FUNCTION DOC]
    # Returns the saver definition message of this saver (the stored object itself, not a
    # copy); it may be none for a deferred or empty saver. No side effects.
    """Generates a `SaverDef` representation of this saver.

    Returns:
      A `SaverDef` proto.
    """
    return self.saver_def

  def to_proto(self, export_scope=None):
    # [FUNCTION DOC]
    # Converts the saver to a saver definition message for meta graph export. Input: optional
    # name scope to strip. Without a scope returns the stored definition itself. With a scope,
    # returns nothing if any of the filename tensor, save tensor or restore op names does not
    # start with that scope; otherwise returns a copy with the scope removed from those three
    # names. The stored definition is not modified.
    """Converts this `Saver` to a `SaverDef` protocol buffer.

    Args:
      export_scope: Optional `string`. Name scope to remove.

    Returns:
      A `SaverDef` protocol buffer.
    """
    if export_scope is None:
      return self.saver_def

    if not (self.saver_def.filename_tensor_name.startswith(export_scope) and
            self.saver_def.save_tensor_name.startswith(export_scope) and
            self.saver_def.restore_op_name.startswith(export_scope)):
      return None

    saver_def = saver_pb2.SaverDef()
    saver_def.CopyFrom(self.saver_def)
    saver_def.filename_tensor_name = ops.strip_name_scope(
        saver_def.filename_tensor_name, export_scope)
    saver_def.save_tensor_name = ops.strip_name_scope(
        saver_def.save_tensor_name, export_scope)
    saver_def.restore_op_name = ops.strip_name_scope(saver_def.restore_op_name,
                                                     export_scope)
    return saver_def

  @staticmethod
  def from_proto(saver_def, import_scope=None):
    # [FUNCTION DOC]
    # Static factory used when importing a meta graph. Inputs: a saver definition message and
    # an optional import name scope. Returns a new Saver built around that definition, with the
    # scope passed as the name (which prefixes the definition's names). Side effects: those of
    # the constructor, including validation errors for an invalid definition.
    """Returns a `Saver` object created from `saver_def`.

    Args:
      saver_def: a `SaverDef` protocol buffer.
      import_scope: Optional `string`. Name scope to use.

    Returns:
      A `Saver` built from saver_def.
    """
    return Saver(saver_def=saver_def, name=import_scope)

  @property
  def last_checkpoints(self):
    # [FUNCTION DOC]
    # Read-only property. Returns a new list of the paths of the checkpoints that have not yet
    # been deleted, ordered from oldest to newest, usable as restore inputs. No side effects.
    """List of not-yet-deleted checkpoint filenames.

    You can pass any of the returned values to `restore()`.

    Returns:
      A list of checkpoint filenames, sorted from oldest to newest.
    """
    return list(self._CheckpointFilename(p) for p in self._last_checkpoints)

  def set_last_checkpoints(self, last_checkpoints):
    # [FUNCTION DOC]
    # Deprecated setter. Input: list of checkpoint paths. Replaces the recorded list with
    # entries that all carry an infinite timestamp so they are never deleted by the
    # keep-forever logic. Raises an assertion error if the input is not a list. Returns nothing.
    """DEPRECATED: Use set_last_checkpoints_with_time.

    Sets the list of old checkpoint filenames.

    Args:
      last_checkpoints: A list of checkpoint filenames.

    Raises:
      AssertionError: If last_checkpoints is not a list.
    """
    assert isinstance(last_checkpoints, list)
    # We use a timestamp of +inf so that this checkpoint will never be
    # deleted.  This is both safe and backwards compatible to a previous
    # version of the code which used s[1] as the "timestamp".
    self._last_checkpoints = [(s, np.inf) for s in last_checkpoints]

  def set_last_checkpoints_with_time(self, last_checkpoints_with_time):
    # [FUNCTION DOC]
    # Setter for the recorded checkpoint list. Input: list of pairs of path and timestamp. Stores
    # that list directly (no copy). Raises an assertion error if it is not a list. Returns
    # nothing.
    """Sets the list of old checkpoint filenames and timestamps.

    Args:
      last_checkpoints_with_time: A list of tuples of checkpoint filenames and
        timestamps.

    Raises:
      AssertionError: If last_checkpoints_with_time is not a list.
    """
    assert isinstance(last_checkpoints_with_time, list)
    self._last_checkpoints = last_checkpoints_with_time

  def recover_last_checkpoints(self, checkpoint_paths):
    # [FUNCTION DOC]
    # Rebuilds the recorded checkpoint list after a crash. Input: list of checkpoint paths. For
    # each, looks up the modification time on disk; paths that no longer exist (not-found
    # error) or yield no time are skipped. The surviving pairs of path and modification time
    # replace the recorded list. Returns nothing. Side effect: reads file metadata.
    """Recovers the internal saver state after a crash.

    This method is useful for recovering the "self._last_checkpoints" state.

    Globs for the checkpoints pointed to by `checkpoint_paths`.  If the files
    exist, use their mtime as the checkpoint timestamp.

    Args:
      checkpoint_paths: a list of checkpoint paths.
    """
    checkpoints_with_mtimes = []
    for checkpoint_path in checkpoint_paths:
      try:
        mtime = checkpoint_management.get_checkpoint_mtimes([checkpoint_path])
      except errors.NotFoundError:
        # It's fine if some other thread/process is deleting some older
        # checkpoint concurrently.
        continue
      if mtime:
        checkpoints_with_mtimes.append((checkpoint_path, mtime[0]))
    self.set_last_checkpoints_with_time(checkpoints_with_mtimes)

  def save(self,
           sess,
           save_path,
           global_step=None,
           latest_filename=None,
           meta_graph_suffix="meta",
           write_meta_graph=True,
           write_state=True,
           strip_default_attrs=False,
           save_debug_info=False):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says a type error is raised if the session is not a Session,
    # but this check is skipped in eager mode. Also, in graph mode with an empty saver and
    # write_meta_graph true, the meta graph is still exported although nothing was saved.
    # Runs the save op and manages checkpoint bookkeeping. Inputs: session (graph mode),
    # save path prefix, optional global step (integer, tensor or tensor name; appended to the
    # path, zero-padded to eight digits if padding is enabled), optional name of the checkpoint
    # state file (default checkpoint), meta graph suffix (default meta), flags to write the
    # meta graph (default true) and the checkpoint state (default true), strip default
    # attributes, and save debug info. Warns when using the deprecated version 1 format.
    # Raises a runtime error if the ops have not been built, a value error if the state file
    # name has path components or collides with the save path (unless sharded), a type error
    # for a non-session in graph mode, and a value error if the parent directory is missing
    # when the file system reports not-found or failed-precondition (other errors of those
    # kinds are re-raised). Unless the saver is empty it runs the save (feeding the file name),
    # records the checkpoint, rewrites the state file and may delete old checkpoints. Then it
    # records write duration and saved-training-time metrics (updating the module-level
    # last-write time under its lock), writes the meta graph in graph mode, and records the
    # checkpoint size metric. Returns the path prefix actually used, or nothing if the saver
    # is empty. Side effects: files, metrics, logging, saver state.
    # pylint: disable=line-too-long
    """Saves variables.

    This method runs the ops added by the constructor for saving variables.
    It requires a session in which the graph was launched.  The variables to
    save must also have been initialized.

    The method returns the path prefix of the newly created checkpoint files.
    This string can be passed directly to a call to `restore()`.

    Args:
      sess: A Session to use to save the variables.
      save_path: String.  Prefix of filenames created for the checkpoint.
      global_step: If provided the global step number is appended to `save_path`
        to create the checkpoint filenames. The optional argument can be a
        `Tensor`, a `Tensor` name or an integer.
      latest_filename: Optional name for the protocol buffer file that will
        contains the list of most recent checkpoints.  That file, kept in the
        same directory as the checkpoint files, is automatically managed by the
        saver to keep track of recent checkpoints.  Defaults to 'checkpoint'.
      meta_graph_suffix: Suffix for `MetaGraphDef` file. Defaults to 'meta'.
      write_meta_graph: `Boolean` indicating whether or not to write the meta
        graph file.
      write_state: `Boolean` indicating whether or not to write the
        `CheckpointStateProto`.
      strip_default_attrs: Boolean. If `True`, default-valued attributes will be
        removed from the NodeDefs. For a detailed guide, see [Stripping
        Default-Valued
        Attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).
      save_debug_info: If `True`, save the GraphDebugInfo to a separate file,
        which in the same directory of save_path and with `_debug` added before
        the file extension. This is only enabled when `write_meta_graph` is
        `True`

    Returns:
      A string: path prefix used for the checkpoint files.  If the saver is
        sharded, this string ends with: '-?????-of-nnnnn' where 'nnnnn'
        is the number of shards created.
      If the saver is empty, returns None.

    Raises:
      TypeError: If `sess` is not a `Session`.
      ValueError: If `latest_filename` contains path components, or if it
        collides with `save_path`.
      RuntimeError: If save and restore ops weren't built.
    """
    # pylint: enable=line-too-long
    start_time = time.time()
    if not self._is_built and not context.executing_eagerly():
      raise RuntimeError(
          "`build()` should be called before save if defer_build==True")
    if latest_filename is None:
      latest_filename = "checkpoint"
    if self._write_version != saver_pb2.SaverDef.V2:
      logging.warning("*******************************************************")
      logging.warning("TensorFlow's V1 checkpoint format has been deprecated.")
      logging.warning("Consider switching to the more efficient V2 format:")
      logging.warning("   `tf.train.Saver(write_version=tf.train.SaverDef.V2)`")
      logging.warning("now on by default.")
      logging.warning("*******************************************************")

    if os.path.split(latest_filename)[0]:
      raise ValueError("'latest_filename' must not contain path components")

    save_path = compat.as_str(save_path)
    if global_step is not None:
      if not isinstance(global_step, compat.integral_types):
        global_step = training_util.global_step(sess, global_step)
      checkpoint_file = "%s-%d" % (save_path, global_step)
      if self._pad_step_number:
        # Zero-pads the step numbers, so that they are sorted when listed.
        checkpoint_file = "%s-%s" % (save_path, "{:08d}".format(global_step))
    else:
      checkpoint_file = save_path
      if os.path.basename(save_path) == latest_filename and not self._sharded:
        # Guard against collision between data file and checkpoint state file.
        raise ValueError(
            "'latest_filename' collides with 'save_path': '%s' and '%s'" %
            (latest_filename, save_path))

    if (not context.executing_eagerly() and
        not isinstance(sess, session.SessionInterface)):
      raise TypeError("'sess' must be a Session; %s" % sess)

    save_path_parent = os.path.dirname(save_path)
    if not self._is_empty:
      try:
        if context.executing_eagerly():
          self._build_eager(
              checkpoint_file, build_save=True, build_restore=False)
          model_checkpoint_path = self.saver_def.save_tensor_name
        else:
          model_checkpoint_path = sess.run(
              self.saver_def.save_tensor_name,
              {self.saver_def.filename_tensor_name: checkpoint_file})

        model_checkpoint_path = compat.as_str(model_checkpoint_path)
        if write_state:
          self._RecordLastCheckpoint(model_checkpoint_path)
          checkpoint_management.update_checkpoint_state_internal(
              save_dir=save_path_parent,
              model_checkpoint_path=model_checkpoint_path,
              all_model_checkpoint_paths=self.last_checkpoints,
              latest_filename=latest_filename,
              save_relative_paths=self._save_relative_paths)
          self._MaybeDeleteOldCheckpoints(meta_graph_suffix=meta_graph_suffix)
      except (errors.FailedPreconditionError, errors.NotFoundError) as exc:
        if not gfile.IsDirectory(save_path_parent):
          exc = ValueError(
              "Parent directory of {} doesn't exist, can't save.".format(
                  save_path))
        raise exc

    end_time = time.time()
    metrics.AddCheckpointWriteDuration(
        api_label=_SAVER_LABEL,
        microseconds=_get_duration_microseconds(start_time, end_time))
    global _END_TIME_OF_LAST_WRITE
    with _END_TIME_OF_LAST_WRITE_LOCK:
      metrics.AddTrainingTimeSaved(
          api_label=_SAVER_LABEL,
          microseconds=_get_duration_microseconds(_END_TIME_OF_LAST_WRITE,
                                                  end_time))
      _END_TIME_OF_LAST_WRITE = end_time

    if write_meta_graph:
      meta_graph_filename = checkpoint_management.meta_graph_filename(
          checkpoint_file, meta_graph_suffix=meta_graph_suffix)
      if not context.executing_eagerly():
        with sess.graph.as_default():
          self.export_meta_graph(
              meta_graph_filename,
              strip_default_attrs=strip_default_attrs,
              save_debug_info=save_debug_info)

    if self._is_empty:
      return None
    else:
      metrics.RecordCheckpointSize(
          api_label=_SAVER_LABEL,
          filesize=_get_checkpoint_size(model_checkpoint_path))
      return model_checkpoint_path

  def export_meta_graph(self,
                        filename=None,
                        collection_list=None,
                        as_text=False,
                        export_scope=None,
                        clear_devices=False,
                        clear_extraneous_savers=False,
                        strip_default_attrs=False,
                        save_debug_info=False):
    # [FUNCTION DOC]
    # Exports the default graph together with this saver's definition as a meta graph message.
    # Inputs: optional output file name with path, list of collection keys to include, flag to
    # write text instead of binary, optional name scope to strip, flags to clear device fields
    # and to remove saver information not belonging to this saver, strip default attributes,
    # and save debug info to a separate file. Serializes the default graph (with shapes) and
    # delegates to the module-level export function. Returns the meta graph message; writes the
    # file if a name is given. Errors (graph over 2 GB, eager mode) come from the delegate.
    # pylint: disable=line-too-long
    """Writes `MetaGraphDef` to save_path/filename.

    Args:
      filename: Optional meta_graph filename including the path.
      collection_list: List of string keys to collect.
      as_text: If `True`, writes the meta_graph as an ASCII proto.
      export_scope: Optional `string`. Name scope to remove.
      clear_devices: Whether or not to clear the device field for an `Operation`
        or `Tensor` during export.
      clear_extraneous_savers: Remove any Saver-related information from the
        graph (both Save/Restore ops and SaverDefs) that are not associated with
        this Saver.
      strip_default_attrs: Boolean. If `True`, default-valued attributes will be
        removed from the NodeDefs. For a detailed guide, see [Stripping
        Default-Valued
        Attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).
      save_debug_info: If `True`, save the GraphDebugInfo to a separate file,
        which in the same directory of filename and with `_debug` added before
        the file extension.

    Returns:
      A `MetaGraphDef` proto.
    """
    # pylint: enable=line-too-long
    return export_meta_graph(
        filename=filename,
        graph_def=ops.get_default_graph().as_graph_def(
            add_shapes=True, use_pybind11_proto=True
        ),
        saver_def=self.saver_def,
        collection_list=collection_list,
        as_text=as_text,
        export_scope=export_scope,
        clear_devices=clear_devices,
        clear_extraneous_savers=clear_extraneous_savers,
        strip_default_attrs=strip_default_attrs,
        save_debug_info=save_debug_info,
    )

  def restore(self, sess, save_path):
    # [FUNCTION DOC]
    # Restores variables from a checkpoint. Inputs: a session (unused in eager mode, where it
    # may be none) and the checkpoint path prefix. Returns nothing at once if the saver is
    # empty. Raises a value error if the path is none or does not name an existing checkpoint.
    # Otherwise logs the path and runs the restore op, feeding the path (in eager mode it
    # builds and runs the restore half instead). If the op reports not-found: if the file is an
    # object-based checkpoint, logs a warning, builds (or reuses the cached) helper saver that
    # maps variable names to the checkpoint's keys, and restores through it; if it is not,
    # re-raises the same kind of error with an explanatory message about a missing variable
    # name or graph key. An invalid-argument error is re-raised with a message about a
    # graph/checkpoint mismatch. On success records the read duration metric. Side effects:
    # assigns variable values, logging, metrics, may create ops and set the cached helper saver.
    """Restores previously saved variables.

    This method runs the ops added by the constructor for restoring variables.
    It requires a session in which the graph was launched.  The variables to
    restore do not have to have been initialized, as restoring is itself a way
    to initialize variables.

    The `save_path` argument is typically a value previously returned from a
    `save()` call, or a call to `latest_checkpoint()`.

    Args:
      sess: A `Session` to use to restore the parameters. None in eager mode.
      save_path: Path where parameters were previously saved.

    Raises:
      ValueError: If save_path is None or not a valid checkpoint.
    """
    start_time = time.time()
    if self._is_empty:
      return
    if save_path is None:
      raise ValueError("Can't load save_path when it is None.")

    checkpoint_prefix = compat.as_text(save_path)
    if not checkpoint_management.checkpoint_exists_internal(checkpoint_prefix):
      raise ValueError("The passed save_path is not a valid checkpoint: " +
                       checkpoint_prefix)

    logging.info("Restoring parameters from %s", checkpoint_prefix)
    try:
      if context.executing_eagerly():
        self._build_eager(save_path, build_save=False, build_restore=True)
      else:
        sess.run(self.saver_def.restore_op_name,
                 {self.saver_def.filename_tensor_name: save_path})
    except errors.NotFoundError as err:
      # There are three common conditions that might cause this error:
      # 0. The file is missing. We ignore here, as this is checked above.
      # 1. This is an object-based checkpoint trying name-based loading.
      # 2. The graph has been altered and a variable or other name is missing.

      # 1. The checkpoint would not be loaded successfully as is. Try to parse
      # it as an object-based checkpoint.
      try:
        names_to_keys = object_graph_key_mapping(save_path)
      except errors.NotFoundError:
        # 2. This is not an object-based checkpoint, which likely means there
        # is a graph mismatch. Re-raise the original error with
        # a helpful message (b/110263146)
        raise _wrap_restore_error_with_msg(
            err, "a Variable name or other graph key that is missing")

      # This is an object-based checkpoint. We'll print a warning and then do
      # the restore.
      logging.warning(
          "Restoring an object-based checkpoint using a name-based saver. This "
          "may be somewhat fragile, and will re-build the Saver. Instead, "
          "consider loading object-based checkpoints using "
          "tf.train.Checkpoint().")
      self._object_restore_saver = saver_from_object_based_checkpoint(
          checkpoint_path=save_path,
          var_list=self._var_list,
          builder=self._builder,
          names_to_keys=names_to_keys,
          cached_saver=self._object_restore_saver)
      self._object_restore_saver.restore(sess=sess, save_path=save_path)
    except errors.InvalidArgumentError as err:
      # There is a mismatch between the graph and the checkpoint being loaded.
      # We add a more reasonable error message here to help users (b/110263146)
      raise _wrap_restore_error_with_msg(
          err, "a mismatch between the current graph and the graph")
    metrics.AddCheckpointReadDuration(
        api_label=_SAVER_LABEL,
        microseconds=_get_duration_microseconds(start_time, time.time()))

  @staticmethod
  def _add_collection_def(meta_graph_def, key, export_scope=None):
    # [FUNCTION DOC]
    # HELPER: static wrapper kept for meta graph export code. Inputs: a meta graph message, a
    # collection key (standard graph key or user string) and an optional name scope to strip.
    # Delegates to the meta graph module to add that collection to the message. Mutates the
    # message; returns nothing.
    """Adds a collection to MetaGraphDef protocol buffer.

    Args:
      meta_graph_def: MetaGraphDef protocol buffer.
      key: One of the GraphKeys or user-defined string.
      export_scope: Optional `string`. Name scope to remove.
    """
    meta_graph.add_collection_def(
        meta_graph_def, key, export_scope=export_scope)


@tf_export(v1=["train.import_meta_graph"])
def import_meta_graph(meta_graph_or_file,
                      clear_devices=False,
                      import_scope=None,
                      **kwargs):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring examples use old top-level names (session, import function,
  # global variable initializer) that no longer exist at the top level; they only work under
  # the compatibility version 1 namespace.
  # Imports a saved meta graph into the current default graph. Inputs: a meta graph message or
  # the path of a file containing one, a flag to clear device fields, an optional import name
  # scope (only used when importing from a message) and extra keyword arguments forwarded to
  # the importer. Adds the graph nodes and collections, then returns a Saver built from the
  # stored saver definition, or a default Saver if only variables exist, or nothing if there
  # is nothing to restore. Raises a runtime error in eager mode. Side effect: modifies the
  # default graph.
  """Recreates a Graph saved in a `MetaGraphDef` proto.

  This function takes a `MetaGraphDef` protocol buffer as input. If
  the argument is a file containing a `MetaGraphDef` protocol buffer ,
  it constructs a protocol buffer from the file content. The function
  then adds all the nodes from the `graph_def` field to the
  current graph, recreates all the collections, and returns a saver
  constructed from the `saver_def` field.

  In combination with `export_meta_graph()`, this function can be used to

  * Serialize a graph along with other Python objects such as `QueueRunner`,
    `Variable` into a `MetaGraphDef`.

  * Restart training from a saved graph and checkpoints.

  * Run inference from a saved graph and checkpoints.

  ```Python
  ...
  # Create a saver.
  saver = tf.compat.v1.train.Saver(...variables...)
  # Remember the training_op we want to run by adding it to a collection.
  tf.compat.v1.add_to_collection('train_op', train_op)
  sess = tf.compat.v1.Session()
  for step in range(1000000):
      sess.run(train_op)
      if step % 1000 == 0:
          # Saves checkpoint, which by default also exports a meta_graph
          # named 'my-model-global_step.meta'.
          saver.save(sess, 'my-model', global_step=step)
  ```

  Later we can continue training from this saved `meta_graph` without building
  the model from scratch.

  ```Python
  with tf.Session() as sess:
    new_saver =
    tf.train.import_meta_graph('my-save-dir/my-model-10000.meta')
    new_saver.restore(sess, 'my-save-dir/my-model-10000')
    # tf.get_collection() returns a list. In this example we only want
    # the first one.
    train_op = tf.get_collection('train_op')[0]
    for step in range(1000000):
      sess.run(train_op)
  ```

  NOTE: Restarting training from saved `meta_graph` only works if the
  device assignments have not changed.

  Example:
  Variables, placeholders, and independent operations can also be stored, as
  shown in the following example.

  ```Python
  # Saving contents and operations.
  v1 = tf.placeholder(tf.float32, name="v1")
  v2 = tf.placeholder(tf.float32, name="v2")
  v3 = tf.math.multiply(v1, v2)
  vx = tf.Variable(10.0, name="vx")
  v4 = tf.add(v3, vx, name="v4")
  saver = tf.train.Saver([vx])
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())
  sess.run(vx.assign(tf.add(vx, vx)))
  result = sess.run(v4, feed_dict={v1:12.0, v2:3.3})
  print(result)
  saver.save(sess, "./model_ex1")
  ```

  Later this model can be restored and contents loaded.

  ```Python
  # Restoring variables and running operations.
  saver = tf.train.import_meta_graph("./model_ex1.meta")
  sess = tf.Session()
  saver.restore(sess, "./model_ex1")
  result = sess.run("v4:0", feed_dict={"v1:0": 12.0, "v2:0": 3.3})
  print(result)
  ```

  Args:
    meta_graph_or_file: `MetaGraphDef` protocol buffer or filename (including
      the path) containing a `MetaGraphDef`.
    clear_devices: Whether or not to clear the device field for an `Operation`
      or `Tensor` during import.
    import_scope: Optional `string`. Name scope to add. Only used when
      initializing from protocol buffer.
    **kwargs: Optional keyed arguments.

  Returns:
    A saver constructed from `saver_def` in `MetaGraphDef` or None.

    A None value is returned if no variables exist in the `MetaGraphDef`
    (i.e., there are no variables to restore).

  Raises:
    RuntimeError: If called with eager execution enabled.

  @compatibility(eager)
  Exporting/importing meta graphs is not supported. No graph exists when eager
  execution is enabled.
  @end_compatibility
  """  # pylint: disable=g-doc-exception
  return _import_meta_graph_with_return_elements(meta_graph_or_file,
                                                 clear_devices, import_scope,
                                                 **kwargs)[0]


def _import_meta_graph_with_return_elements(meta_graph_or_file,
                                            clear_devices=False,
                                            import_scope=None,
                                            return_elements=None,
                                            **kwargs):
  # [FUNCTION DOC]
  # HELPER: shared implementation of the meta graph import, also used by callers that need
  # specific graph elements. Inputs: meta graph message or file path, clear-devices flag,
  # import scope, optional list of names of elements to return, and extra keyword arguments
  # forwarded to the scoped importer. Raises a runtime error in eager mode. Reads the file if
  # a path is given, imports into the default graph, then creates a saver from the imported
  # variables. Returns a pair: the saver (possibly none) and the list of requested elements.
  """Import MetaGraph, and return both a saver and returned elements."""
  if context.executing_eagerly():
    raise RuntimeError("Exporting/importing meta graphs is not supported when "
                       "eager execution is enabled. No graph exists when eager "
                       "execution is enabled.")
  if not isinstance(meta_graph_or_file, meta_graph_pb2.MetaGraphDef):
    meta_graph_def = meta_graph.read_meta_graph_file(meta_graph_or_file)
  else:
    meta_graph_def = meta_graph_or_file

  imported_vars, imported_return_elements = (
      meta_graph.import_scoped_meta_graph_with_return_elements(
          meta_graph_def,
          clear_devices=clear_devices,
          import_scope=import_scope,
          return_elements=return_elements,
          **kwargs))

  saver = _create_saver_from_imported_meta_graph(meta_graph_def, import_scope,
                                                 imported_vars)
  return saver, imported_return_elements


def _create_saver_from_imported_meta_graph(meta_graph_def, import_scope,
                                           imported_vars):
  # [FUNCTION DOC]
  # HELPER: used by the meta graph import to produce the matching saver. Inputs: the meta
  # graph message, the import scope and the mapping from names to imported variables. If the
  # message has a saver definition, the scope is inferred from the first imported variable
  # (its full name minus its original key) when any exist, and a Saver is built from the
  # definition with that scope as its name. Without a definition, returns a default Saver if
  # saveable objects exist in the scope, otherwise logs a message and returns nothing.
  """Return a saver for restoring variable values to an imported MetaGraph."""
  if meta_graph_def.HasField("saver_def"):
    # Infer the scope that is prepended by `import_scoped_meta_graph`.
    scope = import_scope
    var_names = list(imported_vars.keys())
    if var_names:
      sample_key = var_names[0]
      sample_var = imported_vars[sample_key]
      scope = sample_var.name[:-len(sample_key)]

    return Saver(saver_def=meta_graph_def.saver_def, name=scope)
  else:
    if variables._all_saveable_objects(scope=import_scope):  # pylint: disable=protected-access
      # Return the default saver instance for all graph variables.
      return Saver()
    else:
      # If no graph variables exist, then a Saver cannot be constructed.
      logging.info("Saver not created because there are no variables in the"
                   " graph to restore")
      return None


@tf_export(v1=["train.export_meta_graph"])
def export_meta_graph(filename=None,
                      meta_info_def=None,
                      graph_def=None,
                      saver_def=None,
                      collection_list=None,
                      as_text=False,
                      graph=None,
                      export_scope=None,
                      clear_devices=False,
                      clear_extraneous_savers=False,
                      strip_default_attrs=False,
                      save_debug_info=False,
                      **kwargs):
  # [FUNCTION DOC]
  # Exports a graph, saver definition and collections as a meta graph message. Inputs: optional
  # output file name, meta info message, graph definition, saver definition, list of collection
  # keys, text flag, graph (default is the default graph), export scope, flags to clear
  # devices, remove unrelated saver information, strip default attributes and save debug
  # info, plus extra keyword arguments forwarded to the scoped exporter. Raises a runtime
  # error in eager mode unless both a graph definition and a graph are supplied; a graph over
  # 2 GB raises a value error in the delegate. Returns the meta graph message and writes it
  # to the file when a name is given.
  # pylint: disable=line-too-long
  """Returns `MetaGraphDef` proto.

  Optionally writes it to filename.

  This function exports the graph, saver, and collection objects into
  `MetaGraphDef` protocol buffer with the intention of it being imported
  at a later time or location to restart training, run inference, or be
  a subgraph.

  Args:
    filename: Optional filename including the path for writing the generated
      `MetaGraphDef` protocol buffer.
    meta_info_def: `MetaInfoDef` protocol buffer.
    graph_def: `GraphDef` protocol buffer.
    saver_def: `SaverDef` protocol buffer.
    collection_list: List of string keys to collect.
    as_text: If `True`, writes the `MetaGraphDef` as an ASCII proto.
    graph: The `Graph` to export. If `None`, use the default graph.
    export_scope: Optional `string`. Name scope under which to extract the
      subgraph. The scope name will be striped from the node definitions for
      easy import later into new name scopes. If `None`, the whole graph is
      exported. graph_def and export_scope cannot both be specified.
    clear_devices: Whether or not to clear the device field for an `Operation`
      or `Tensor` during export.
    clear_extraneous_savers: Remove any Saver-related information from the graph
      (both Save/Restore ops and SaverDefs) that are not associated with the
      provided SaverDef.
    strip_default_attrs: Boolean. If `True`, default-valued attributes will be
      removed from the NodeDefs. For a detailed guide, see [Stripping
      Default-Valued
      Attributes](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md#stripping-default-valued-attributes).
    save_debug_info: If `True`, save the GraphDebugInfo to a separate file,
      which in the same directory of filename and with `_debug` added before the
      file extend.
    **kwargs: Optional keyed arguments.

  Returns:
    A `MetaGraphDef` proto.

  Raises:
    ValueError: When the `GraphDef` is larger than 2GB.
    RuntimeError: If called with eager execution enabled.

  @compatibility(eager)
  Exporting/importing meta graphs is not supported unless both `graph_def` and
  `graph` are provided. No graph exists when eager execution is enabled.
  @end_compatibility
  """
  # pylint: enable=line-too-long
  if context.executing_eagerly() and not (graph_def is not None and
                                          graph is not None):
    raise RuntimeError("Exporting/importing meta graphs is not supported when "
                       "eager execution is enabled. No graph exists when eager "
                       "execution is enabled.")
  meta_graph_def, _ = meta_graph.export_scoped_meta_graph(
      filename=filename,
      meta_info_def=meta_info_def,
      graph_def=graph_def,
      saver_def=saver_def,
      collection_list=collection_list,
      as_text=as_text,
      graph=graph,
      export_scope=export_scope,
      clear_devices=clear_devices,
      clear_extraneous_savers=clear_extraneous_savers,
      strip_default_attrs=strip_default_attrs,
      save_debug_info=save_debug_info,
      **kwargs)
  return meta_graph_def


def _wrap_restore_error_with_msg(err, extra_verbiage):
  # [FUNCTION DOC]
  # HELPER: used by the restore method to make restore failures clearer. Inputs: a TensorFlow
  # error and a phrase describing the probable cause. Builds a message saying the restore
  # failed likely because of that cause and embeds the original message. Returns a new error
  # of the same class carrying the original node definition and op plus the new message
  # (the caller raises it). No other side effects.
  err_msg = ("Restoring from checkpoint failed. This is most likely "
             "due to {} from the checkpoint. Please ensure that you "
             "have not altered the graph expected based on the checkpoint. "
             "Original error:\n\n{}").format(extra_verbiage, err.message)
  return err.__class__(err.node_def, err.op, err_msg)


ops.register_proto_function(
    ops.GraphKeys.SAVERS,
    proto_type=saver_pb2.SaverDef,
    to_proto=Saver.to_proto,
    from_proto=Saver.from_proto)


def object_graph_key_mapping(checkpoint_path):
  # [FUNCTION DOC]
  # Reads an object-based checkpoint and returns a mapping from each variable's full name to its
  # checkpoint key. Input: the checkpoint path (string). Opens a checkpoint reader, loads the
  # serialized object graph tensor, parses it, and collects the full name and key of every
  # attribute of every node. Raises a not-found error if the checkpoint lacks the object graph
  # tensor (i.e. is not object-based). Side effect: reads the checkpoint file.
  """Return name to key mappings from the checkpoint.

  Args:
    checkpoint_path: string, path to object-based checkpoint

  Returns:
    Dictionary mapping tensor names to checkpoint keys.
  """
  reader = py_checkpoint_reader.NewCheckpointReader(checkpoint_path)
  object_graph_string = reader.get_tensor(trackable.OBJECT_GRAPH_PROTO_KEY)
  object_graph_proto = (trackable_object_graph_pb2.TrackableObjectGraph())
  object_graph_proto.ParseFromString(object_graph_string)
  names_to_keys = {}
  for node in object_graph_proto.nodes:
    for attribute in node.attributes:
      names_to_keys[attribute.full_name] = attribute.checkpoint_key
  return names_to_keys


def saver_from_object_based_checkpoint(checkpoint_path,
                                       var_list=None,
                                       builder=None,
                                       names_to_keys=None,
                                       cached_saver=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring lists a value error and a not-found error but, when a cached
  # saver is given, the saveable specs are still renamed in place, mutating the current
  # graph's saveables even though the cached saver is what is returned.
  # Builds a Saver that reads a name-based request from an object-based checkpoint. Inputs:
  # checkpoint path, optional variable list (default: all saveable objects), optional builder
  # (default: bulk builder, otherwise unused), optional mapping from names to checkpoint keys
  # (default: read from the checkpoint) and an optional previously built saver to reuse.
  # Raises a value error if the mapping must be read but the checkpoint is not object-based,
  # and a not-found error listing missing, extra and matching names if any current variable
  # name is absent from the mapping. Otherwise renames each spec to its checkpoint key and
  # returns the cached saver if given, else a new Saver over the saveables. Side effects:
  # mutates spec names; creating a Saver adds graph ops.
  """Return a `Saver` which reads from an object-based checkpoint.

  This function validates that all variables in the variables list are remapped
  in the object-based checkpoint (or `names_to_keys` dict if provided). A
  saver will be created with the list of remapped variables.

  The `cached_saver` argument allows the user to pass in a previously created
  saver, so multiple `saver.restore()` calls don't pollute the graph when graph
  building. This assumes that keys are consistent, meaning that the
    1) `checkpoint_path` checkpoint, and
    2) checkpoint used to create the `cached_saver`
  are the same type of object-based checkpoint. If this argument is set, this
  function will simply validate that all variables have been remapped by the
  checkpoint at `checkpoint_path`.

  Note that in general, `tf.train.Checkpoint` should be used to restore/save an
  object-based checkpoint.

  Args:
    checkpoint_path: string, path to object-based checkpoint
    var_list: list of `Variables` that appear in the checkpoint. If `None`,
      `var_list` will be set to all saveable objects.
    builder: a `BaseSaverBuilder` instance. If `None`, a new `BulkSaverBuilder`
      will be created.
    names_to_keys: dict mapping string tensor names to checkpoint keys. If
      `None`, this dict will be generated from the checkpoint file.
    cached_saver: Cached `Saver` object with remapped variables.

  Returns:
    `Saver` with remapped variables for reading from an object-based checkpoint.

  Raises:
    ValueError if the checkpoint provided is not an object-based checkpoint.
    NotFoundError: If one of the variables in `var_list` can not be found in the
      checkpoint. This could mean the checkpoint or `names_to_keys` mapping is
      missing the variable.
  """
  if names_to_keys is None:
    try:
      names_to_keys = object_graph_key_mapping(checkpoint_path)
    except errors.NotFoundError:
      raise ValueError("Checkpoint in %s not an object-based checkpoint." %
                       checkpoint_path)
  if var_list is None:
    var_list = variables._all_saveable_objects()  # pylint: disable=protected-access
  if builder is None:
    builder = BulkSaverBuilder()

  if not isinstance(var_list, dict):
    var_list = saveable_object_util.op_list_to_dict(var_list)
  saveables = saveable_object_util.validate_and_slice_inputs(var_list)
  current_names = set()
  for saveable in saveables:
    for spec in saveable.specs:
      current_names.add(spec.name)
  previous_names = set(names_to_keys.keys())
  missing_names = current_names - previous_names
  if missing_names:
    extra_names = previous_names - current_names
    intersecting_names = previous_names.intersection(current_names)
    raise errors.NotFoundError(
        None,
        None,
        message=(
            "\n\nExisting variables not in the checkpoint: %s\n\n"
            "Variables names when this checkpoint was written which don't "
            "exist now: %s\n\n"
            "(%d variable name(s) did match)\n\n"
            "Could not find some variables in the checkpoint (see names "
            "above). Saver was attempting to load an object-based checkpoint "
            "(saved using tf.train.Checkpoint or tf.keras.Model.save_weights) "
            "using variable names. If the checkpoint was written with eager "
            "execution enabled, it's possible that variable names have "
            "changed (for example missing a '_1' suffix). It's also "
            "possible that there are new variables which did not exist "
            "when the checkpoint was written. You can construct a "
            "Saver(var_list=...) with only the variables which previously "
            "existed, and if variable names have changed you may need to "
            "make this a dictionary with the old names as keys.") %
                (", ".join(sorted(missing_names)), ", ".join(
                    sorted(extra_names)), len(intersecting_names)))
  for saveable in saveables:
    for spec in saveable.specs:
      spec.name = names_to_keys[spec.name]
  if cached_saver is None:
    return Saver(saveables)
  return cached_saver
