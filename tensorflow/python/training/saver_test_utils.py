# [MODULE DOC]
# HELPER: Small shared test-support module for the checkpointing/saver tests.
# Defines CheckpointedOp, a thin wrapper around a mutable string-to-float hash table resource
# that carries a hand-written saveable object so tests can exercise custom (non-variable)
# checkpointing through the Saver. Importing it has no side effects beyond the imports; the
# class registers a saveable in the graph collection only when an instance is built in graph mode.
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
# =============================================================================
"""Utility classes for testing checkpointing."""

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops as ops_lib
from tensorflow.python.ops import gen_lookup_ops
from tensorflow.python.training import saver as saver_module


class CheckpointedOp:
  # [CLASS DOC]
  # HELPER: Test utility for checkpoint/saver tests.
  # Wraps a mutable hash table resource (string keys, float32 values) and exposes insert, lookup,
  # keys and values operations plus a saveable object describing how to save and restore the
  # table contents. In graph mode the saveable is created once at construction and added to the
  # graph's saveable-objects collection; in eager mode a new saveable is built on each access.
  # Holds the table resource handle and the op name. Contains the nested CustomSaveable class.
  """Op with a custom checkpointing implementation.

  Defined as part of the test because the MutableHashTable Python code is
  currently in contrib.
  """

  # pylint: disable=protected-access
  def __init__(self, name, table_ref=None):
    # [FUNCTION DOC]
    # Builds the wrapper. Inputs: name (string) used for the table op and the saveable; table_ref
    # (optional) an existing table resource handle to reuse. If table_ref is absent, creates a new
    # mutable hash table with string keys and float32 values. Stores the handle and name. When
    # not executing eagerly, creates a custom saveable and adds it to the graph's
    # saveable-objects collection (side effect on the default graph). Returns nothing.
    if table_ref is None:
      self.table_ref = gen_lookup_ops.mutable_hash_table_v2(
          key_dtype=dtypes.string, value_dtype=dtypes.float32, name=name)
    else:
      self.table_ref = table_ref
    self._name = name
    if not context.executing_eagerly():
      self._saveable = CheckpointedOp.CustomSaveable(self, name)
      ops_lib.add_to_collection(ops_lib.GraphKeys.SAVEABLE_OBJECTS,
                                self._saveable)

  @property
  def name(self):
    # [FUNCTION DOC]
    # Read-only property returning the name string given at construction.
    return self._name

  @property
  def saveable(self):
    # [FUNCTION DOC]
    # Read-only property returning the saveable object for this table. In eager mode a fresh
    # custom saveable is built on every access; in graph mode the one created at construction is
    # returned.
    if context.executing_eagerly():
      return CheckpointedOp.CustomSaveable(self, self.name)
    else:
      return self._saveable

  def insert(self, keys, values):
    # [FUNCTION DOC]
    # Inserts entries into the table. Inputs: keys (string tensor) and values (float32 tensor) of
    # matching size. Returns the insert operation (executed immediately in eager mode, an op to
    # run in graph mode). Mutates the table resource.
    return gen_lookup_ops.lookup_table_insert_v2(self.table_ref, keys, values)

  def lookup(self, keys, default):
    # [FUNCTION DOC]
    # Looks up values in the table. Inputs: keys (string tensor) and default (scalar float32
    # tensor returned for missing keys). Returns a float32 tensor of looked-up values. Does not
    # modify the table.
    return gen_lookup_ops.lookup_table_find_v2(self.table_ref, keys, default)

  def keys(self):
    # [FUNCTION DOC]
    # Returns a string tensor holding all keys currently in the table (first element of the table
    # export). No inputs, no side effects on the table.
    return self._export()[0]

  def values(self):
    # [FUNCTION DOC]
    # Returns a float32 tensor holding all values currently in the table, in the same order as
    # keys (second element of the table export). No inputs, no side effects on the table.
    return self._export()[1]

  def _export(self):
    # [FUNCTION DOC]
    # HELPER: Shared by keys, values and the custom saveable to read the table contents.
    # Exports the table as a pair of tensors: string keys and float32 values. Returns that pair;
    # no side effects on the table.
    return gen_lookup_ops.lookup_table_export_v2(self.table_ref, dtypes.string,
                                                 dtypes.float32)

  class CustomSaveable(saver_module.BaseSaverBuilder.SaveableObject):
    # [CLASS DOC]
    # HELPER: Test utility describing how the Saver saves and restores a CheckpointedOp.
    # Subclass of the saver builder's saveable object. Saves two entries (keys and values) taken
    # from the table export, named after the op with keys and values suffixes, and restores them
    # by importing both tensors back into the table.
    """A custom saveable for CheckpointedOp."""

    def __init__(self, table, name):
      # [FUNCTION DOC]
      # Inputs: table (the CheckpointedOp being saved) and name (string base name). Exports the
      # table, builds two save specs (keys named name plus keys suffix, values named name plus
      # values suffix, both with empty slice spec), and initializes the base saveable with the
      # table, the specs and the name. Returns nothing.
      tensors = table._export()
      specs = [
          saver_module.BaseSaverBuilder.SaveSpec(tensors[0], "",
                                                 name + "-keys"),
          saver_module.BaseSaverBuilder.SaveSpec(tensors[1], "",
                                                 name + "-values")
      ]
      super(CheckpointedOp.CustomSaveable, self).__init__(table, specs, name)

    def restore(self, restore_tensors, shapes):
      # [FUNCTION DOC]
      # Restores the table from a checkpoint. Inputs: restore_tensors (list of two tensors: keys
      # then values, as read from the checkpoint) and shapes (ignored). Imports the tensors into
      # the table resource, replacing its contents, and returns the import operation.
      return gen_lookup_ops.lookup_table_import_v2(
          self.op.table_ref, restore_tensors[0], restore_tensors[1])
  # pylint: enable=protected-access
