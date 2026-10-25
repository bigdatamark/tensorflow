# [MODULE DOC]
# Defines the Asset trackable, a wrapper around a file path that a SavedModel copies into its
# assets directory so the model stays hermetic. It is exported publicly as the SavedModel Asset
# class. Import-time side effect: registers a tensor conversion function so an Asset can be used
# anywhere a string tensor is accepted (it converts to its path tensor).
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Asset-type Trackable object."""
import os

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.trackable import base
from tensorflow.python.util.tf_export import tf_export


@tf_export("saved_model.Asset")
class Asset(base.Trackable):
  # [CLASS DOC]
  # Trackable object representing a file to include in a SavedModel. Holds the file path as a
  # scalar string tensor in a single attribute (the asset path). When a trackable graph containing
  # an Asset is exported, the file is copied into the SavedModel assets directory; on load, an
  # Asset is rebuilt pointing at the copied file. Created directly by users from a path (string
  # or path-like object) or by the SavedModel loader. Inherits from the Trackable base class.
  """Represents a file asset to hermetically include in a SavedModel.

  A SavedModel can include arbitrary files, called assets, that are needed
  for its use. For example a vocabulary file used initialize a lookup table.

  When a trackable object is exported via `tf.saved_model.save()`, all the
  `Asset`s reachable from it are copied into the SavedModel assets directory.
  Upon loading, the assets and the serialized functions that depend on them
  will refer to the correct filepaths inside the SavedModel directory.

  Example:

  ```
  filename = tf.saved_model.Asset("file.txt")

  @tf.function(input_signature=[])
  def func():
    return tf.io.read_file(filename)

  trackable_obj = tf.train.Checkpoint()
  trackable_obj.func = func
  trackable_obj.filename = filename
  tf.saved_model.save(trackable_obj, "/tmp/saved_model")

  # The created SavedModel is hermetic, it does not depend on
  # the original file and can be moved to another path.
  tf.io.gfile.remove("file.txt")
  tf.io.gfile.rename("/tmp/saved_model", "/tmp/new_location")

  reloaded_obj = tf.saved_model.load("/tmp/new_location")
  print(reloaded_obj.func())
  ```

  Attributes:
    asset_path: A path, or a 0-D `tf.string` tensor with path to the asset.
  """

  def __init__(self, path):
    # [FUNCTION DOC]
    # Constructor. Input: path, either a string, a path-like object (converted to a string) or a
    # string tensor naming the asset file. Converts the path to a scalar string tensor named
    # asset path, created under an init scope on the CPU device so it is not captured by function
    # graphs and does not end up in serialized function bodies. Stores the tensor on the instance.
    # Returns nothing; it does not check that the file exists.
    """Record the full path to the asset."""
    if isinstance(path, os.PathLike):
      path = os.fspath(path)
    # The init_scope prevents functions from capturing `path` in an
    # initialization graph, since it is transient and should not end up in a
    # serialized function body.
    with ops.init_scope(), ops.device("CPU"):
      self._path = ops.convert_to_tensor(
          path, dtype=dtypes.string, name="asset_path")

  @property
  def asset_path(self):
    # [FUNCTION DOC]
    # Read-only property returning the stored scalar string tensor holding the asset path. No
    # inputs, no side effects, no exceptions.
    """Fetch the current asset path."""
    return self._path

  @classmethod
  def _deserialize_from_proto(cls, object_proto, export_dir, asset_file_def,
                              **unused_kwargs):
    # [FUNCTION DOC]
    # HELPER: used by the SavedModel loader to rebuild an Asset from its serialized form.
    # Class method. Inputs: object_proto (the serialized trackable object description holding an
    # asset entry with an index into the asset file definitions), export_dir (the SavedModel
    # directory), asset_file_def (list of asset file definitions with filenames); extra keyword
    # arguments are ignored. Joins the assets directory of export_dir with the filename at the
    # referenced index and creates an instance of the class from it. In graph (non-eager) mode it
    # also adds the asset path tensor to the graph's asset file path collection (side effect).
    # Returns the new asset.
    proto = object_proto.asset
    filename = file_io.join(
        path_helpers.get_assets_dir(export_dir),
        asset_file_def[proto.asset_file_def_index].filename)
    asset = cls(filename)
    if not context.executing_eagerly():
      ops.add_to_collection(ops.GraphKeys.ASSET_FILEPATHS, asset.asset_path)
    return asset

  def _add_trackable_child(self, name, value):
    # [FUNCTION DOC]
    # HELPER: hook called by the checkpoint/SavedModel loading machinery to attach a restored child.
    # Inputs: name (attribute name) and value (the child trackable). Simply sets the attribute on
    # this object. Returns nothing.
    setattr(self, name, value)

  def _export_to_saved_model_graph(self, tensor_map, **unused_kwargs):
    # [FUNCTION DOC]
    # HELPER: hook used during SavedModel export. Input: tensor_map, a mutable mapping from
    # original tensors to their replacements in the exported graph (extra keyword arguments are
    # ignored). Creates a string placeholder with the shape of the asset path and a resource
    # variable initialized from it, then records in tensor_map that the asset path tensor maps to
    # that variable (mutating the mapping and creating graph objects). Returns a list containing
    # the asset path tensor.
    # TODO(b/205008097): Instead of mapping 1-1 between trackable asset
    # and asset in the graph def consider deduping the assets that
    # point to the same file.
    asset_path_initializer = array_ops.placeholder(
        shape=self.asset_path.shape,
        dtype=dtypes.string,
        name="asset_path_initializer")
    asset_variable = resource_variable_ops.ResourceVariable(
        asset_path_initializer)

    tensor_map[self.asset_path] = asset_variable
    return [self.asset_path]


tensor_conversion_registry.register_tensor_conversion_function(
    Asset, lambda asset, **kw: ops.convert_to_tensor(asset.asset_path, **kw))
