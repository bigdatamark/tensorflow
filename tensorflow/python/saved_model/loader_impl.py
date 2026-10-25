# [MODULE DOC]
# Implementation of session-based (graph mode) SavedModel loading. Provides functions to parse
# the SavedModel file (binary or text), read the debug info, find asset tensors, init and train
# ops, check whether a directory looks like a SavedModel, the deprecated v1 load function, and the
# loader class used by it (and subclassed by the eager v1 loader). Import-time effect: registers
# public API exports and defines a metrics label. Public facade modules re-export parts of it.
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
"""Loader implementation for SavedModel with hermetic, language-neutral exports.
"""

import os
import sys

from google.protobuf import message
from google.protobuf import text_format

from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.framework import ops
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.platform import tf_logging
from tensorflow.python.saved_model import constants
from tensorflow.python.saved_model import path_helpers
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import utils_impl as saved_model_utils
# Placeholder for protosplitter merger import.
from tensorflow.python.saved_model.pywrap_saved_model import metrics
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export

# API label for SavedModel metrics.
_LOADER_LABEL = "loader"


def parse_saved_model_with_debug_info(export_dir):
  # [FUNCTION DOC]
  # Reads the SavedModel proto from export_dir (via parse_saved_model) and the optional graph debug
  # info file from the debug subdirectory. A missing debug info file yields an empty debug info
  # message. Returns a pair: the SavedModel proto and the GraphDebugInfo proto. Raises an I/O
  # error if the SavedModel is missing or unparsable, or if the debug info file exists but cannot
  # be parsed (that error does not chain the original decode error).
  """Reads the savedmodel as well as the graph debug info.

  Args:
    export_dir: Directory containing the SavedModel and GraphDebugInfo files.

  Returns:
    `SavedModel` and `GraphDebugInfo` protocol buffers.

  Raises:
    IOError: If the saved model file does not exist, or cannot be successfully
    parsed. Missing graph debug info file is fine.
  """
  saved_model = parse_saved_model(export_dir)

  debug_info_path = file_io.join(
      path_helpers.get_debug_dir(export_dir),
      constants.DEBUG_INFO_FILENAME_PB)
  debug_info = graph_debug_info_pb2.GraphDebugInfo()
  if file_io.file_exists(debug_info_path):
    with file_io.FileIO(debug_info_path, "rb") as debug_file:
      try:
        debug_info.ParseFromString(debug_file.read())
      except message.DecodeError as e:
        raise IOError(f"Cannot parse file {debug_info_path}: {e}.")

  return (saved_model, debug_info)


@tf_export("__internal__.saved_model.parse_saved_model", v1=[])
def parse_saved_model(export_dir):
  # [FUNCTION DOC]
  # KNOWN ISSUE: a path for the cpb-format file is built but never used, so a directory holding
  # only that format raises the does-not-exist error here, although the directory-check functions
  # in this module treat that file as a SavedModel. The error message also lists only the pbtxt
  # and pb names.
  # Reads the SavedModel proto from export_dir (string or path-like). Prefers the binary file if
  # it exists, otherwise the text format file. Returns the parsed SavedModel proto. Raises an I/O
  # error if neither file exists or the file content cannot be parsed. Reads files only.
  """Reads the savedmodel.pb or savedmodel.pbtxt file containing `SavedModel`.

  Args:
    export_dir: String or Pathlike, path to the directory containing the
    SavedModel file.

  Returns:
    A `SavedModel` protocol buffer.

  Raises:
    IOError: If the file does not exist, or cannot be successfully parsed.
  """
  # Build the path to the SavedModel in pbtxt format.
  path_to_pbtxt = file_io.join(
      compat.as_bytes(compat.path_to_str(export_dir)),
      compat.as_bytes(constants.SAVED_MODEL_FILENAME_PBTXT))
  # Build the path to the SavedModel in pb format.
  path_to_pb = file_io.join(
      compat.as_bytes(compat.path_to_str(export_dir)),
      compat.as_bytes(constants.SAVED_MODEL_FILENAME_PB))
  # Build the path to the SavedModel in cpb format.
  path_to_cpb = file_io.join(
      compat.as_bytes(compat.path_to_str(export_dir)),
      compat.as_bytes(constants.SAVED_MODEL_FILENAME_CPB))

  # Parse the SavedModel protocol buffer.
  saved_model = saved_model_pb2.SavedModel()
  if file_io.file_exists(path_to_pb):
    with file_io.FileIO(path_to_pb, "rb") as f:
      file_content = f.read()
    try:
      saved_model.ParseFromString(file_content)
    except message.DecodeError as e:
      raise IOError(f"Cannot parse file {path_to_pb}: {str(e)}.") from e
  elif file_io.file_exists(path_to_pbtxt):
    with file_io.FileIO(path_to_pbtxt, "rb") as f:
      file_content = f.read()
    try:
      text_format.Parse(file_content.decode("utf-8"), saved_model)
    except text_format.ParseError as e:
      raise IOError(f"Cannot parse file {path_to_pbtxt}: {str(e)}.") from e
  else:
    raise IOError(
        f"SavedModel file does not exist at: {export_dir}{os.path.sep}"
        f"{{{constants.SAVED_MODEL_FILENAME_PBTXT}|"
        f"{constants.SAVED_MODEL_FILENAME_PB}}}")
  return saved_model


def get_asset_tensors(export_dir, meta_graph_def_to_load, import_scope=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring promises absolute asset paths, but the value is just the assets
  # directory joined under export_dir, so it is absolute only if export_dir is absolute.
  # Collects asset file definitions from the MetaGraph, preferring the asset file definition list
  # and falling back to the assets collection (unpacking the stored messages). Inputs: export_dir
  # (SavedModel directory), the MetaGraph definition, and optional import_scope (string prefix
  # added with a slash to tensor names). Returns a mapping from asset tensor name to the asset
  # file path (bytes joined path). No side effects.
  """Gets the asset tensors, if defined in the meta graph def to load.

  Args:
    export_dir: Directory where the SavedModel is located.
    meta_graph_def_to_load: The meta graph def from the SavedModel to be loaded.
    import_scope: Optional `string` -- if specified, prepend this followed by
        '/' to all returned asset tensor names.

  Returns:
    A dictionary of asset tensors, keyed by the name of the asset tensor. The
    value in the map corresponds to the absolute path of the asset file.
  """
  # Collection-def that may contain the assets key.
  collection_def = meta_graph_def_to_load.collection_def

  asset_tensor_dict = {}
  asset_protos = []

  if meta_graph_def_to_load.asset_file_def:
    asset_protos = meta_graph_def_to_load.asset_file_def
  elif constants.ASSETS_KEY in collection_def:
    assets_any_proto = collection_def[constants.ASSETS_KEY].any_list.value
    for asset_any_proto in assets_any_proto:
      asset_proto = meta_graph_pb2.AssetFileDef()
      asset_any_proto.Unpack(asset_proto)
      asset_protos.append(asset_proto)

  # Location of the assets for SavedModel.
  assets_directory = file_io.join(
      compat.as_bytes(export_dir), compat.as_bytes(constants.ASSETS_DIRECTORY))
  # Process each asset and add it to the asset tensor dictionary.
  for asset_proto in asset_protos:
    tensor_name = asset_proto.tensor_info.name
    if import_scope:
      tensor_name = "%s/%s" % (import_scope, tensor_name)
    asset_tensor_dict[tensor_name] = file_io.join(
        compat.as_bytes(assets_directory),
        compat.as_bytes(asset_proto.filename))

  return asset_tensor_dict


def _get_main_op_tensor(
    meta_graph_def_to_load, init_op_key=constants.MAIN_OP_KEY):
  # [FUNCTION DOC]
  # HELPER: init op lookup for get_init_op and the collection-based op helper.
  # KNOWN ISSUE: the op is fetched from the default graph's collection of the same name, not from
  # the node list in the MetaGraph, so it assumes the graph was already imported into the default
  # graph, and import scopes are not applied.
  # Looks up the collection named init_op_key (default: the main op key) in the MetaGraph. If
  # present it must hold exactly one node, otherwise a runtime error is raised; returns the first
  # item of the default graph's collection with that key. Returns nothing if the key is absent.
  """Gets the main op tensor, if one exists.

  Args:
    meta_graph_def_to_load: The meta graph def from the SavedModel to be loaded.
    init_op_key: name of the collection to check; should be one of MAIN_OP_KEY
      or the deprecated LEGACY_INIT_OP_KEY

  Returns:
    The main op tensor, if it exists and `None` otherwise.

  Raises:
    RuntimeError: If the collection def corresponding to the main op key has
        other than exactly one tensor.
  """
  collection_def = meta_graph_def_to_load.collection_def
  init_op = None
  if init_op_key in collection_def:
    init_op_list = collection_def[init_op_key].node_list.value
    if len(init_op_list) != 1:
      raise RuntimeError("Expected exactly one SavedModel init op. "
                         f"Found {len(init_op_list)}: {init_op_list}.")
    init_op = ops.get_collection(init_op_key)[0]
  return init_op


def _get_op_from_collection(meta_graph_def, op_key):
  # [FUNCTION DOC]
  # HELPER: thin alias for the main-op lookup used by get_init_op and get_train_op.
  # Returns the single op stored in the named collection, or nothing if absent; see the lookup
  # function for errors and side effects.
  return _get_main_op_tensor(meta_graph_def, op_key)


def _get_op_from_signature_def(meta_graph_def, op_signature_key, import_scope):
  # [FUNCTION DOC]
  # HELPER: signature-based op lookup for get_init_op and get_train_op.
  # If the MetaGraph has a signature definition under op_signature_key, returns the op loaded from
  # it (honoring import_scope, a string prefix or nothing); otherwise returns nothing.
  """Retrieve op stored in the imported meta graph's signature def."""
  if op_signature_key in meta_graph_def.signature_def:
    return signature_def_utils.load_op_from_signature_def(
        meta_graph_def.signature_def[op_signature_key], op_signature_key,
        import_scope)
  else:
    return None


def get_init_op(meta_graph_def, import_scope=None):
  # [FUNCTION DOC]
  # Finds the initialization op of a loaded MetaGraph. Tries, in order, the init op signature
  # definition, the main op collection and the legacy init op collection, returning the first
  # one found, or nothing. import_scope is an optional name prefix used for the signature lookup
  # only. Raises a runtime error if a collection holds other than exactly one node.
  return (_get_op_from_signature_def(
      meta_graph_def, constants.INIT_OP_SIGNATURE_KEY, import_scope) or
          _get_op_from_collection(meta_graph_def, constants.MAIN_OP_KEY) or
          _get_op_from_collection(meta_graph_def, constants.LEGACY_INIT_OP_KEY))


def get_train_op(meta_graph_def, import_scope=None):
  # [FUNCTION DOC]
  # Finds the training op of a loaded MetaGraph: first from the train op signature definition,
  # and if absent from the train op collection. import_scope is an optional name prefix for the
  # signature lookup. Returns the op or nothing. Raises a runtime error if the collection holds
  # other than exactly one node.
  train_op = _get_op_from_signature_def(
      meta_graph_def, constants.TRAIN_OP_SIGNATURE_KEY, import_scope)
  if train_op is None:
    train_op = _get_op_from_collection(meta_graph_def, constants.TRAIN_OP_KEY)
  return train_op


@tf_export(v1=[
    "saved_model.contains_saved_model",
    "saved_model.maybe_saved_model_directory",
    "saved_model.loader.maybe_saved_model_directory"
])
@deprecation.deprecated_endpoints(
    "saved_model.loader.maybe_saved_model_directory")
def maybe_saved_model_directory(export_dir):
  # [FUNCTION DOC]
  # Cheap check for SavedModel presence. Input: export_dir, a string path (not a path-like
  # object). Returns true if any of the text, binary or cpb format SavedModel files exists in the
  # directory, false otherwise. Does not parse the file, so true does not guarantee loadability.
  # Deprecated endpoint in the v1 API.
  """Checks whether the provided export directory could contain a SavedModel.

  Note that the method does not load any data by itself. If the method returns
  `false`, the export directory definitely does not contain a SavedModel. If the
  method returns `true`, the export directory may contain a SavedModel but
  provides no guarantee that it can be loaded.

  Args:
    export_dir: Absolute string path to possible export location. For example,
                '/my/foo/model'.

  Returns:
    True if the export directory contains SavedModel files, False otherwise.
  """
  txt_path = file_io.join(export_dir, constants.SAVED_MODEL_FILENAME_PBTXT)
  pb_path = file_io.join(export_dir, constants.SAVED_MODEL_FILENAME_PB)
  cpb_path = file_io.join(export_dir, constants.SAVED_MODEL_FILENAME_CPB)
  return (
      file_io.file_exists(txt_path)
      or file_io.file_exists(pb_path)
      or file_io.file_exists(cpb_path)
  )


@tf_export("saved_model.contains_saved_model", v1=[])
def contains_saved_model(export_dir):
  # [FUNCTION DOC]
  # Public v2 wrapper around the directory check. Accepts a string or path-like export_dir
  # (converted to a string) and returns true if a SavedModel file exists there, false otherwise.
  # Does not verify that the model can be loaded.
  """Checks whether the provided export directory could contain a SavedModel.

  Note that the method does not load any data by itself. If the method returns
  `false`, the export directory definitely does not contain a SavedModel. If the
  method returns `true`, the export directory may contain a SavedModel but
  provides no guarantee that it can be loaded.

  Args:
    export_dir: Absolute path to possible export location. For example,
                '/my/foo/model'.

  Returns:
    True if the export directory contains SavedModel files, False otherwise.
  """
  if isinstance(export_dir, os.PathLike):
    export_dir = os.fspath(export_dir)
  return maybe_saved_model_directory(export_dir)


@tf_export(v1=["saved_model.load", "saved_model.loader.load"])
@deprecation.deprecated(
    None,
    "Use `tf.saved_model.load` instead.")
def load(sess, tags, export_dir, import_scope=None, **saver_kwargs):
  # [FUNCTION DOC]
  # Deprecated v1 function: loads the MetaGraph with the given tags (set or list of strings) from
  # export_dir into the graph of session sess, restores variables and runs init ops. Optional
  # import_scope prefixes tensor names; extra keyword arguments go to the saver import. Creates a
  # loader object and delegates to its load method. Returns the loaded MetaGraph definition.
  # Raises a runtime error if no MetaGraph matches the tags; file errors propagate.
  """Loads the model from a SavedModel as specified by tags.

  Args:
    sess: The TensorFlow session to restore the variables.
    tags: Set of string tags to identify the required MetaGraphDef. These should
        correspond to the tags used when saving the variables using the
        SavedModel `save()` API.
    export_dir: Directory in which the SavedModel protocol buffer and variables
        to be loaded are located.
    import_scope: Optional `string` -- if specified, prepend this string
        followed by '/' to all loaded tensor names. This scope is applied to
        tensor instances loaded into the passed session, but it is *not* written
        through to the static `MetaGraphDef` protocol buffer that is returned.
    **saver_kwargs: Optional keyword arguments passed through to Saver.

  Returns:
    The `MetaGraphDef` protocol buffer loaded in the provided session. This
    can be used to further extract signature-defs, collection-defs, etc.

  Raises:
    RuntimeError: MetaGraphDef associated with the tags cannot be found.

  @compatibility(TF2)

  `tf.compat.v1.saved_model.load` or `tf.compat.v1.saved_model.loader.load` is
  not compatible with eager execution. Please use `tf.saved_model.load` instead
  to load your model. You can refer to the [SavedModel guide]
  (https://www.tensorflow.org/guide/saved_model) for more information as well as
  "Importing SavedModels from TensorFlow 1.x" in the [`tf.saved_model.load`]
  (https://www.tensorflow.org/api_docs/python/tf/saved_model/load) docstring.

  #### How to Map Arguments

  | TF1 Arg Name          | TF2 Arg Name    | Note                       |
  | :-------------------- | :-------------- | :------------------------- |
  | `sess`                | Not supported   | -                          |
  | `tags`                | `tags`          | -                          |
  | `export_dir`          | `export_dir`    | -                          |
  | `import_scope`        | Not supported   | Name scopes are not needed.
  :                       :                 : By default, variables are  :
  :                       :                 : associated with the loaded :
  :                       :                 : object and function names  :
  :                       :                 : are deduped.               :
  | `saver_kwargs`        | Not supported   | -                          |

  #### Before & After Usage Example

  Before:

  ```
  with tf.compat.v1.Session(graph=tf.Graph()) as sess:
    tf.compat.v1.saved_model.loader.load(sess, ["foo-tag"], export_dir)
  ```

  After:

  ```
  model = tf.saved_model.load(export_dir, tags=["foo-tag"])
  ```
  @end_compatibility
  """
  loader = SavedModelLoader(export_dir)
  return loader.load(sess, tags, import_scope, **saver_kwargs)


class SavedModelLoader(object):
  # [CLASS DOC]
  # Loader for a SavedModel in a directory. Parses the SavedModel proto on construction and keeps
  # the export directory, the variables checkpoint path and the parsed proto (exposed through
  # read-only properties). Provides steps to pick a MetaGraph by tags, import its graph, restore
  # variables and run init ops, plus a combined load. Subclassed by the eager v1 loader.
  """Load graphs and restore variable values from a `SavedModel`."""

  def __init__(self, export_dir):
    # [FUNCTION DOC]
    # Constructor. Input: export_dir, the SavedModel directory. Stores it, computes the
    # variables checkpoint path, and parses the SavedModel file. Raises an I/O error if the
    # SavedModel file is missing or unparsable. Returns nothing.
    """Creates a `SavedModelLoader`.

    Args:
      export_dir: Directory in which the SavedModel protocol buffer and
        variables to be loaded are located.
    """
    self._export_dir = export_dir
    self._variables_path = path_helpers.get_variables_path(export_dir)
    self._saved_model = parse_saved_model(export_dir)

  @property
  def export_dir(self):
    # [FUNCTION DOC]
    # Read-only property returning the SavedModel directory given at construction.
    """Directory containing the SavedModel."""
    return self._export_dir

  @property
  def variables_path(self):
    # [FUNCTION DOC]
    # Read-only property returning the path prefix of the variables checkpoint files.
    """Path to variable checkpoint files."""
    return self._variables_path

  @property
  def saved_model(self):
    # [FUNCTION DOC]
    # Read-only property returning the parsed SavedModel proto held by the loader.
    """SavedModel object parsed from the export directory."""
    return self._saved_model

  def get_meta_graph_def_from_tags(self, tags):
    # [FUNCTION DOC]
    # Finds the MetaGraph whose tag set equals the set of the given tags (list or set of strings,
    # order and duplicates ignored); the first match is returned. Raises a runtime error listing
    # the available tag sets if none matches. No side effects.
    """Return MetaGraphDef with the exact specified tags.

    Args:
      tags: A list or set of string tags that identify the MetaGraphDef.

    Returns:
      MetaGraphDef with the same tags.

    Raises:
      RuntimeError: if no metagraphs were found with the associated tags.
    """
    found_match = False
    meta_graph_def_to_load = None
    available_tags = []
    for meta_graph_def in self._saved_model.meta_graphs:
      available_tags.append(set(meta_graph_def.meta_info_def.tags))
      if set(meta_graph_def.meta_info_def.tags) == set(tags):
        meta_graph_def_to_load = meta_graph_def
        found_match = True
        break

    if not found_match:
      raise RuntimeError(
          f"MetaGraphDef associated with tags {str(tags).strip('[]')} "
          "could not be found in SavedModel, with available tags "
          f"'{available_tags}'. To inspect available tag-sets in"
          " the SavedModel, please use the SavedModel CLI: `saved_model_cli`.")
    return meta_graph_def_to_load

  def load_graph(self, graph, tags, import_scope=None, **saver_kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it returns a tuple of saver and list of imported elements;
    # that is what the underlying import returns, but the MetaGraph is modified in place on
    # big-endian hosts, and the documented import_scope remark about the returned definition does
    # not apply since no definition is returned.
    # Imports the MetaGraph selected by tags into the given graph, passing import_scope and extra
    # keyword arguments to the import. On big-endian machines first byte-swaps function tensor
    # contents of the MetaGraph (mutating it). Returns the saver defined by the MetaGraph and the
    # list of returned elements. Raises a runtime error if no MetaGraph matches the tags.
    """Load ops and nodes from SavedModel MetaGraph into graph.

    Args:
      graph: tf.Graph object.
      tags: a set of string tags identifying a MetaGraphDef.
      import_scope: Optional `string` -- if specified, prepend this string
        followed by '/' to all loaded tensor names. This scope is applied to
        tensor instances loaded into the passed session, but it is *not* written
        through to the static `MetaGraphDef` protocol buffer that is returned.
      **saver_kwargs: keyword arguments to pass to tf.train.import_meta_graph.

    Returns:
      A tuple of
        * Saver defined by the MetaGraph, which can be used to restore the
          variable values.
        * List of `Operation`/`Tensor` objects returned from
          `tf.import_graph_def` (may be `None`).
    """
    meta_graph_def = self.get_meta_graph_def_from_tags(tags)
    if sys.byteorder == "big":
      saved_model_utils.swap_function_tensor_content(meta_graph_def, "little",
                                                     "big")
    with graph.as_default():
      return tf_saver._import_meta_graph_with_return_elements(  # pylint: disable=protected-access
          meta_graph_def, import_scope=import_scope, **saver_kwargs)

  def restore_variables(self, sess, saver, import_scope=None):
    # [FUNCTION DOC]
    # Restores variable values from the SavedModel checkpoint into session sess. If saver is
    # nothing and the graph (within import_scope) has no saveable variables, logs that there is
    # nothing to restore. If saver is a saver object, restores from the variables path. Otherwise
    # raises a value error saying a saver is required. Side effect: assigns variable values in
    # the session.
    """Restore SavedModel variable values into the session.

    Args:
      sess: tf.compat.v1.Session to restore variable values.
      saver: a tf.compat.v1.train.Saver object. Can be None if there are no
        variables in graph. This may be the saver returned by the load_graph()
        function, or a default `tf.compat.v1.train.Saver()`.
      import_scope: Optional `string` -- if specified, prepend this string
        followed by '/' to all loaded tensor names. This scope is applied to
        tensor instances loaded into the passed session, but it is *not* written
        through to the static `MetaGraphDef` protocol buffer that is returned.

    Raises:
      ValueError: if no saver was passed to the saver argument, and there are
        variables in the graph.
    """
    with sess.graph.as_default():
      if (saver is None and
          not variables._all_saveable_objects(scope=import_scope)):  # pylint: disable=protected-access
        tf_logging.info("The specified SavedModel has no variables; no "
                        "checkpoints were restored.")
      elif isinstance(saver, tf_saver.Saver):
        saver.restore(sess, self._variables_path)
      else:
        raise ValueError(
            "No tf.train.Saver object was passed to the function "
            "`SavedModelLoader.restore_variables`. Since there are variables in"
            " the graph, a saver is required.")

  def run_init_ops(self, sess, tags, import_scope=None):
    # [FUNCTION DOC]
    # Runs the initialization op of the MetaGraph selected by tags in session sess, feeding the
    # asset file paths to their asset tensors. Does nothing if the MetaGraph has no init op.
    # import_scope is an optional name prefix. Raises a runtime error if no MetaGraph matches.
    # Returns nothing.
    """Run initialization ops defined in the `MetaGraphDef`.

    Args:
      sess: tf.compat.v1.Session to restore variable values.
      tags: a set of string tags identifying a MetaGraphDef.
      import_scope: Optional `string` -- if specified, prepend this string
        followed by '/' to all loaded tensor names. This scope is applied to
        tensor instances loaded into the passed session, but it is *not* written
        through to the static `MetaGraphDef` protocol buffer that is returned.
    """
    meta_graph_def = self.get_meta_graph_def_from_tags(tags)
    with sess.graph.as_default():
      # Get asset tensors, if any.
      asset_tensors_dictionary = get_asset_tensors(
          self._export_dir, meta_graph_def, import_scope=import_scope)

      init_op = get_init_op(meta_graph_def, import_scope)
      if init_op is not None:
        sess.run(fetches=[init_op], feed_dict=asset_tensors_dictionary)

  def load(self, sess, tags, import_scope=None, **saver_kwargs):
    # [FUNCTION DOC]
    # Full session-based load. Re-parses the SavedModel file (only to decide metrics), records a
    # read-API metric, then within the session graph imports the MetaGraph for tags, restores
    # variables and runs init ops. Records a read metric tagged version 2 if there is a single
    # MetaGraph with an object graph, else version 1. Returns the loaded MetaGraph definition.
    # Inputs: sess, tags, optional import_scope and saver keyword arguments. Errors propagate.
    """Load the MetaGraphDef graph and restore variable values into the session.

    Args:
      sess: tf.compat.v1.Session to restore variable values.
      tags: a set of string tags identifying a MetaGraphDef.
      import_scope: Optional `string` -- if specified, prepend this string
        followed by '/' to all loaded tensor names. This scope is applied to
        tensor instances loaded into the passed session, but it is *not* written
        through to the static `MetaGraphDef` protocol buffer that is returned.
      **saver_kwargs: keyword arguments to pass to tf.train.import_meta_graph.

    Returns:
      `MetagraphDef` proto of the graph that was loaded.
    """
    saved_model_proto = parse_saved_model(self._export_dir)
    metrics.IncrementReadApi(_LOADER_LABEL)

    with sess.graph.as_default():
      saver, _ = self.load_graph(sess.graph, tags, import_scope,
                                 **saver_kwargs)
      self.restore_variables(sess, saver, import_scope)
      self.run_init_ops(sess, tags, import_scope)
    meta_graph_def = self.get_meta_graph_def_from_tags(tags)

    if (len(saved_model_proto.meta_graphs) == 1 and
        saved_model_proto.meta_graphs[0].HasField("object_graph_def")):
      metrics.IncrementRead(write_version="2")
    else:
      metrics.IncrementRead(write_version="1")

    return meta_graph_def
