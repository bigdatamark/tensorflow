# [MODULE DOC]
# Core of the tf.data input pipeline Python API. Defines the abstract base dataset classes
# (DatasetV2 for TF2 behavior, DatasetV1 / DatasetV1Adapter for the legacy compat.v1 behavior),
# the dataset type specification (DatasetSpec), the numpy iterator wrapper, and a set of
# module-level helpers (structure/legacy-type accessors, variant conversion, rewrites,
# resource-resolver for automatic control dependencies, rejection-resampling math).
# Most dataset transformations are thin methods that lazily import a dedicated *_op module
# (e.g. map, batch, shuffle, cache) to avoid circular imports and delegate construction there.
# Exposes the AUTOTUNE, INFINITE and UNKNOWN constants and the module-level name Dataset, which
# points at DatasetV2 when TF2 behavior is enabled and DatasetV1 otherwise (it is switched by a
# callback registered with the v2 compat layer).
# Import-time side effects: exports constants through the public API export machinery, marks the
# ReduceDataset op as not differentiable, registers a TypeSpec codec for DatasetSpec, registers
# the callback that toggles Dataset between V1 and V2, registers an automatic-control-dependencies
# resource resolver for tf.data ops, creates lazy loaders for prefetch/shuffle/batch/repeat ops,
# and installs autograph overrides for datasets.
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
"""Python wrappers for Datasets."""

import abc
import functools
import queue
import threading
from typing import Union
import warnings

import numpy as np

from tensorflow.core.framework import dataset_metadata_pb2
from tensorflow.core.framework import dataset_options_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import struct_pb2
from tensorflow.python import tf2
from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_autograph
from tensorflow.python.data.ops import debug_mode
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.data.util import traverse
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import wrap_function
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import auto_control_deps_utils as acd_utils
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import none_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed as core_random_seed
from tensorflow.python.framework import sparse_tensor as sparse_tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_io_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import base as tracking_base
from tensorflow.python.trackable import resource as resource_lib
from tensorflow.python.types import data as data_types
from tensorflow.python.types import trace
from tensorflow.python.util import deprecation
from tensorflow.python.util import lazy_loader
from tensorflow.python.util import nest as tf_nest
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export


# Symbols forwarded for legacy access through dataset_ops.py. These forwarded
# symbols can be removed once all internal uses are updated.
StructuredFunctionWrapper = structured_function.StructuredFunctionWrapper

# TODO(b/240947712): Clean up the circular dependencies.
# Loaded lazily due to a circular dependency (dataset_ops ->
# prefetch_op -> dataset_ops).
prefetch_op = lazy_loader.LazyLoader(
    "prefetch_op", globals(),
    "tensorflow.python.data.ops.prefetch_op")
# Loaded lazily due to a circular dependency (dataset_ops ->
# shuffle_op -> dataset_ops).
shuffle_op = lazy_loader.LazyLoader(
    "shuffle_op", globals(),
    "tensorflow.python.data.ops.shuffle_op")


ops.NotDifferentiable("ReduceDataset")

# A constant that can be used to enable auto-tuning.
AUTOTUNE = -1
tf_export("data.AUTOTUNE").export_constant(__name__, "AUTOTUNE")
# TODO(b/168128531): Deprecate and remove this symbol.
tf_export("data.experimental.AUTOTUNE").export_constant(__name__, "AUTOTUNE")

# Constants representing infinite and unknown cardinalities.
INFINITE = -1
UNKNOWN = -2
COMPRESSION_GZIP = "GZIP"
COMPRESSION_SNAPPY = "NONE"
DATASET_SPEC_FILENAME = "dataset_spec.pb"
tf_export("data.INFINITE_CARDINALITY").export_constant(__name__, "INFINITE")
tf_export("data.UNKNOWN_CARDINALITY").export_constant(__name__, "UNKNOWN")


def _validate_and_encode(name):
  # [FUNCTION DOC]
  # HELPER: validates and serializes user-supplied dataset names for dataset metadata.
  # Checks that name is a valid identifier (letters, digits, underscores, not starting with a
  # digit, no spaces) and returns it encoded as UTF-8 bytes. Raises a value error with an
  # explanatory message otherwise.
  if not name.isidentifier():
    raise ValueError("Invalid `name`. The argument `name` needs to be a valid "
                     "identifier. Value is considered a valid identifier if it "
                     "only contains alphanumeric characters (a-z), (A-Z), and "
                     "(0-9), or underscores (_). A valid identifier cannot "
                     "start with a number, or contain any spaces.")
  return name.encode("utf-8")


def get_type(value):
  # [FUNCTION DOC]
  # Returns the Python type of a value. If the value is a type specification, returns the type of
  # values that specification describes; otherwise returns the plain type of the value.
  """Returns the type of `value` if it is a TypeSpec."""

  if isinstance(value, type_spec.TypeSpec):
    return value.value_type()
  else:
    return type(value)


@tf_export("data.Dataset", v1=[])
class DatasetV2(
    collections_abc.Iterable,
    tracking_base.Trackable,
    composite_tensor.CompositeTensor,
    data_types.DatasetV2,
    metaclass=abc.ABCMeta):
  # [CLASS DOC]
  # Public tf.data.Dataset in TF2: an abstract, iterable, trackable and composite-tensor representation
  # of a potentially large sequence of elements (nested structures of tensors and other components).
  # Each instance wraps a variant tensor that is the handle of the underlying dataset op, remembers
  # the graph it was created in, and holds a merged, immutable options object combined from all of
  # its input datasets. Concrete subclasses (sources and transformations) create their variant tensor
  # and pass it to the constructor, and must implement the list of input datasets and the element
  # type specification. Most transformation methods delegate to the dedicated op modules and return
  # a new dataset. Instances are iterated in eager mode or inside a function.
  """Represents a potentially large set of elements.

  The `tf.data.Dataset` API supports writing descriptive and efficient input
  pipelines. `Dataset` usage follows a common pattern:

  1. Create a source dataset from your input data.
  2. Apply dataset transformations to preprocess the data.
  3. Iterate over the dataset and process the elements.

  Iteration happens in a streaming fashion, so the full dataset does not need to
  fit into memory.

  Source Datasets:

  The simplest way to create a dataset is to create it from a python `list`:

  >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
  >>> for element in dataset:
  ...   print(element)
  tf.Tensor(1, shape=(), dtype=int32)
  tf.Tensor(2, shape=(), dtype=int32)
  tf.Tensor(3, shape=(), dtype=int32)

  To process lines from files, use `tf.data.TextLineDataset`:

  >>> dataset = tf.data.TextLineDataset(["file1.txt", "file2.txt"])

  To process records written in the `TFRecord` format, use `TFRecordDataset`:

  >>> dataset = tf.data.TFRecordDataset(["file1.tfrecords", "file2.tfrecords"])

  To create a dataset of all files matching a pattern, use
  `tf.data.Dataset.list_files`:

  ```python
  dataset = tf.data.Dataset.list_files("/path/*.txt")
  ```

  See `tf.data.FixedLengthRecordDataset` and `tf.data.Dataset.from_generator`
  for more ways to create datasets.

  Transformations:

  Once you have a dataset, you can apply transformations to prepare the data for
  your model:

  >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
  >>> dataset = dataset.map(lambda x: x*2)
  >>> [a.item() for a in dataset.as_numpy_iterator()]
  [2, 4, 6]

  Common Terms:

  **Element**: A single output from calling `next()` on a dataset iterator.
    Elements may be nested structures containing multiple components. For
    example, the element `(1, (3, "apple"))` has one tuple nested in another
    tuple. The components are `1`, `3`, and `"apple"`.

  **Component**: The leaf in the nested structure of an element.

  Supported types:

  Elements can be nested structures of tuples, named tuples, and dictionaries.
  Note that Python lists are *not* treated as nested structures of components.
  Instead, lists are converted to tensors and treated as components. For
  example, the element `(1, [1, 2, 3])` has only two components; the tensor `1`
  and the tensor `[1, 2, 3]`. Element components can be of any type
  representable by `tf.TypeSpec`, including `tf.Tensor`, `tf.data.Dataset`,
  `tf.sparse.SparseTensor`, `tf.RaggedTensor`, and `tf.TensorArray`.

  ```python
  a = 1 # Integer element
  b = 2.0 # Float element
  c = (1, 2) # Tuple element with 2 components
  d = {"a": (2, 2), "b": 3} # Dict element with 3 components
  Point = collections.namedtuple("Point", ["x", "y"])
  e = Point(1, 2) # Named tuple
  f = tf.data.Dataset.range(10) # Dataset element
  ```

  For more information,
  read [this guide](https://www.tensorflow.org/guide/data).
  """

  def __init__(self, variant_tensor):
    # [FUNCTION DOC]
    # Constructor. Takes the variant tensor (the dataset handle) produced by the subclass. Stores it
    # and the current default graph, then builds the options of this dataset by merging the options of
    # every input dataset (reading the wrapped dataset of legacy V1 inputs, skipping V1 sources that
    # have no wrapped dataset) and finally freezes the options as immutable.
    # Raises a type error if any input is neither a V1 nor V2 dataset, or if a wrapped V1 dataset is
    # not a V2 dataset.
    """Creates a DatasetV2 object.

    This is a difference between DatasetV1 and DatasetV2. DatasetV1 does not
    take anything in its constructor whereas in the DatasetV2, we expect
    subclasses to create a variant_tensor and pass it in to the super() call.

    Args:
      variant_tensor: A DT_VARIANT tensor that represents the dataset.
    """
    self._variant_tensor_attr = variant_tensor
    self._graph_attr = ops.get_default_graph()

    # Initialize the options for this dataset and its inputs.
    self._options_attr = options_lib.Options()
    for input_dataset in self._inputs():
      input_options = None
      if isinstance(input_dataset, data_types.DatasetV1):
        # If the V1 dataset does not have the `_dataset` attribute, we assume it
        # is a dataset source and hence does not have options. Otherwise, we
        # grab the options of `_dataset` object
        if hasattr(input_dataset, "_dataset"):
          if not isinstance(input_dataset._dataset, data_types.DatasetV2):
            raise TypeError(
                f"Each input of dataset {type(self)} should be a subclass of "
                f"`tf.data.Dataset` but encountered "
                f"{type(input_dataset._dataset)}.")
          input_options = input_dataset._dataset._options_attr
      elif isinstance(input_dataset, data_types.DatasetV2):
        input_options = input_dataset._options_attr
      else:
        raise TypeError(
            f"Each input of dataset {type(self)} should be a subclass of "
            f"`tf.data.Dataset` but encountered {type(input_dataset)}.")
      if input_options is not None:
        self._options_attr = self._options_attr.merge(input_options)
    self._options_attr._set_mutable(False)  # pylint: disable=protected-access

  @property
  def _variant_tensor(self):
    # [FUNCTION DOC]
    # Read-only property returning the variant tensor that represents this dataset.
    return self._variant_tensor_attr

  @_variant_tensor.setter
  def _variant_tensor(self, _):
    # [FUNCTION DOC]
    # Setter that always raises a value error: the variant tensor cannot be reassigned.
    raise ValueError("The `_variant_tensor` property cannot be modified.")

  @deprecation.deprecated_args(None, "Use external_state_policy instead",
                               "allow_stateful")
  def _as_serialized_graph(
      self,
      allow_stateful=None,
      strip_device_assignment=None,
      external_state_policy=options_lib.ExternalStatePolicy.WARN):
    # [FUNCTION DOC]
    # Produces a serialized graph definition of this dataset as a scalar string tensor.
    # Inputs: allow_stateful (deprecated, optional; only used when no external state policy is given),
    # strip_device_assignment (optional flag to strip job/task device placement), external_state_policy
    # (enumeration controlling how external state is handled; defaults to warn).
    # If a policy is given (the default) the policy-aware op is used and allow_stateful is ignored;
    # otherwise the older op is used, with or without device stripping. Returns the string tensor.
    """Produces serialized graph representation of the dataset.

    Args:
      allow_stateful: If true, we allow stateful ops to be present in the graph
        def. In that case, the state in these ops would be thrown away.
      strip_device_assignment: If true, non-local (i.e. job and task) device
        assignment is stripped from ops in the serialized graph.
      external_state_policy: The ExternalStatePolicy enum that determines how we
        handle input pipelines that depend on external state. By default, its
        set to WARN.

    Returns:
      A scalar `tf.Tensor` of `tf.string` type, representing this dataset as a
      serialized graph.
    """
    if external_state_policy:
      policy = external_state_policy.value
      return gen_dataset_ops.dataset_to_graph_v2(
          self._variant_tensor,
          external_state_policy=policy,
          strip_device_assignment=strip_device_assignment)
    if strip_device_assignment:
      return gen_dataset_ops.dataset_to_graph(
          self._variant_tensor,
          allow_stateful=allow_stateful,
          strip_device_assignment=strip_device_assignment)
    return gen_dataset_ops.dataset_to_graph(
        self._variant_tensor, allow_stateful=allow_stateful)

  def _maybe_track_assets(self, graph_def):
    # [FUNCTION DOC]
    # HELPER: used when tracing a dataset into a saveable function to track files as assets.
    # Scans the nodes of a serialized graph definition for file-identity nodes, evaluates their
    # constant string values eagerly on the CPU, and tracks each file path as an asset object on this
    # dataset (overwriting earlier tracked children). Returns a mapping from the asset constant node
    # name to the list of tracked asset objects; an empty mapping if there are no such nodes.
    """Finds and tracks nodes in `graph_def` that refer to asset files.

    Args:
      graph_def: Serialized graph representation of this dataset.

    Returns:
      A dictionary mapping the node name of an asset constant to a tracked
      `asset.Asset` object.
    """
    asset_tracker = {}
    for node in graph_def.node:
      if node.name.startswith("FileIdentity"):
        asset_tracker[node.input[0]] = None

    if not asset_tracker:
      return {}

    for node in graph_def.node:
      if node.name in asset_tracker:
        tensor_proto = node.attr["value"].tensor
        with context.eager_mode(), ops.device("CPU"):
          node_value = gen_parsing_ops.parse_tensor(
              tensor_proto.SerializeToString(), dtypes.string).numpy()
        asset_tracker[node.name] = ([
            self._track_trackable(asset.Asset(n),
                                  name=node.name + "_" + str(i), overwrite=True)
            for i, n in enumerate(node_value)
        ])
    return asset_tracker

  def _trackable_children(self,
                          save_type=tracking_base.SaveType.CHECKPOINT,
                          **kwargs):
    # [FUNCTION DOC]
    # Trackable override. For anything other than a saved-model save, returns no children. For a
    # saved-model save it traces a no-input function that recreates the dataset variant (forcing the
    # tracing so assets get tracked), and adds a variant-tracker child wrapping the variant tensor and
    # that creator function in addition to the inherited children. Returns the mapping of children.
    if save_type != tracking_base.SaveType.SAVEDMODEL:
      return {}

    # _trace_variant_creation only works when executing eagerly, so we don't
    # want to run it in the object initialization.
    @def_function.function(input_signature=[], autograph=False)
    def _creator():
      # [FUNCTION DOC]
      # HELPER: zero-argument function (no autograph) that re-creates this dataset's variant tensor by
      # running the traced variant-creation function; used as the resource creator for saving.
      resource = self._trace_variant_creation()()  # pylint: disable=protected-access
      return resource
    _creator.get_concrete_function()  # Trigger asset tracking

    children = super(DatasetV2, self)._trackable_children(save_type, **kwargs)
    children["_variant_tracker"] = _VariantTracker(self._variant_tensor,
                                                   _creator)
    return children

  def _trace_variant_creation(self):
    # [FUNCTION DOC]
    # Traces and returns a zero-argument concrete function that outputs a variant tensor for this
    # dataset, by serializing the dataset graph (failing on external state), locating its single
    # return node, mapping any file nodes to tracked assets when building inside a function, wrapping
    # the graph definition as a function and adding all functions used by this dataset to its graph.
    # Raises not-implemented if the dataset was not created eagerly, and an assertion error if the
    # graph does not have exactly one return value. Only works in eager mode.
    """Traces a function which outputs a variant `tf.Tensor` for this dataset.

    Note that creating this function involves evaluating an op, and is currently
    only supported when executing eagerly.

    Returns:
      A zero-argument `ConcreteFunction` which outputs a variant `tf.Tensor`.
    """
    variant = self._variant_tensor
    if not isinstance(variant, ops.EagerTensor):
      raise NotImplementedError(
          "Constructing a tf.function that reproduces a given dataset is only "
          "supported for datasets created eagerly. Please file a feature "
          "request if this is important to you.")
    with context.eager_mode(), ops.device("CPU"):
      # pylint: disable=protected-access
      graph_def = graph_pb2.GraphDef().FromString(
          self._as_serialized_graph(external_state_policy=options_lib
                                    .ExternalStatePolicy.FAIL).numpy())
    output_node_names = []
    for node in graph_def.node:
      if node.op == "_Retval":
        output_node_names = node.input

    if len(output_node_names) != 1:
      raise AssertionError(
          f"Dataset graph is expected to only have one return value but found "
          f"{len(output_node_names)} return values: {output_node_names}.")

    output_node_name = output_node_names[0]

    file_path_nodes = {}
    # When building a tf.function, track files as `saved_model.Asset`s.
    if ops.get_default_graph().building_function:
      asset_tracker = self._maybe_track_assets(graph_def)
      for key in asset_tracker:
        assets_list = [
            array_ops.expand_dims(asset.asset_path, axis=0)
            for asset in asset_tracker[key]
        ]
        file_path_nodes[key] = array_ops.concat(assets_list, axis=0)

    # Add functions used in this Dataset to the function's graph, since they
    # need to follow it around (and for example be added to a SavedModel which
    # references the dataset).
    variant_function = wrap_function.function_from_graph_def(
        graph_def,
        inputs=[],
        outputs=output_node_name + ":0",
        captures=file_path_nodes)
    for used_function in self._functions():
      used_function.function.add_to_graph(variant_function.graph)
    return variant_function

  @abc.abstractmethod
  def _inputs(self):
    # [FUNCTION DOC]
    # Abstract method that subclasses implement to return the list of input datasets. The base
    # version raises a not-implemented error.
    """Returns a list of the input datasets of the dataset."""

    raise NotImplementedError(f"{type(self)}._inputs()")

  @property
  def _graph(self):
    # [FUNCTION DOC]
    # Read-only property returning the graph this dataset was created in.
    return self._graph_attr

  @_graph.setter
  def _graph(self, _):
    # [FUNCTION DOC]
    # Setter that always raises a value error: the graph cannot be reassigned.
    raise ValueError("The `_graph` property cannot be modified.")

  # TODO(jsimsa): Change this to be the transitive closure of functions used
  # by this dataset and its inputs.
  def _functions(self) -> list[StructuredFunctionWrapper]:
    # [FUNCTION DOC]
    # Returns the list of structured function wrappers used by this dataset. The base version returns
    # an empty list; subclasses with user functions override it. (A TODO notes this should become the
    # transitive closure over inputs.)
    """Returns a list of functions associated with this dataset.

    Returns:
      A list of `StructuredFunctionWrapper` objects.
    """
    return []

  def _options(self):
    # [FUNCTION DOC]
    # Returns the serialized options tensor of this dataset, obtained from the dataset op.
    """Returns the options tensor for this dataset."""
    # pylint: disable=protected-access
    return gen_dataset_ops.get_options(self._variant_tensor)

  @classmethod
  def _options_tensor_to_options(cls, serialized_options):
    # [FUNCTION DOC]
    # HELPER: class method converting a serialized options tensor into an options object.
    # Creates default options and, if the tensor has a statically known value, parses it as an
    # options message and loads it into the object. Returns the options object (default options when
    # the value is not statically known, as in graph mode).
    """Converts options tensor to tf.data.Options object."""
    options = options_lib.Options()
    if tensor_util.constant_value(serialized_options) is not None:
      pb = dataset_options_pb2.Options.FromString(tensor_util.constant_value(
          serialized_options))
      options._from_proto(pb)  # pylint: disable=protected-access
    return options

  def options(self):
    # [FUNCTION DOC]
    # Returns the options of this dataset and its inputs. In eager mode the options are read from the
    # dataset op, converted to an options object and made immutable. In graph mode a warning is
    # emitted and the legacy options computed at construction time are returned, which may not reflect
    # the real values.
    """Returns the options for this dataset and its inputs.

    Returns:
      A `tf.data.Options` object representing the dataset options.
    """
    if context.executing_eagerly():
      options = self._options_tensor_to_options(self._options())
      options._set_mutable(False)  # pylint: disable=protected-access
      return options
    warnings.warn("To make it possible to preserve tf.data options across "
                  "serialization boundaries, their implementation has moved to "
                  "be part of the TensorFlow graph. As a consequence, the "
                  "options value is in general no longer known at graph "
                  "construction time. Invoking this method in graph mode "
                  "retains the legacy behavior of the original implementation, "
                  "but note that the returned value might not reflect the "
                  "actual value of the options.")
    return self._options_attr

  def _apply_debug_options(self):
    # [FUNCTION DOC]
    # HELPER: used before executing a dataset to honor debug mode.
    # When debug mode is on, returns a new dataset with autotuning, filter parallelization,
    # map-and-batch fusion and map parallelization disabled (to avoid parallelism and asynchrony);
    # otherwise returns this dataset unchanged.
    if debug_mode.DEBUG_MODE:
      # Disable autotuning and static optimizations that could introduce
      # parallelism or asynchrony.
      options = options_lib.Options()
      options.autotune.enabled = False
      options.experimental_optimization.filter_parallelization = False
      options.experimental_optimization.map_and_batch_fusion = False
      options.experimental_optimization.map_parallelization = False
      dataset = _OptionsDataset(self, options)
    else:
      dataset = self

    return dataset

  def __iter__(self) -> iterator_ops.OwnedIterator:
    # [FUNCTION DOC]
    # Creates an iterator over the elements (Python iteration protocol). Allowed in eager mode or
    # inside a function; the iterator is placed on the same device as the dataset variant tensor.
    # Returns an owned iterator. Raises a runtime error in graph mode outside a function.
    """Creates an iterator for elements of this dataset.

    The returned iterator implements the Python Iterator protocol.

    Returns:
      An `tf.data.Iterator` for the elements of this dataset.

    Raises:
      RuntimeError: If not inside of tf.function and not executing eagerly.
    """
    if context.executing_eagerly() or ops.inside_function():
      with ops.colocate_with(self._variant_tensor):
        return iterator_ops.OwnedIterator(self)
    else:
      raise RuntimeError("`tf.data.Dataset` only supports Python-style "
                         "iteration in eager mode or within tf.function.")

  def __bool__(self):
    # [FUNCTION DOC]
    # Always returns true so that a dataset is truthy even though a length method is defined.
    return True  # Required as __len__ is defined

  __nonzero__ = __bool__  # Python 2 backward compatibility

  def __len__(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says a runtime error is raised and that an integer is returned, but
    # the code raises type errors (not runtime errors) and returns the eager scalar tensor produced
    # by the cardinality method, not a Python integer.
    # Returns the length of the dataset in eager mode. Raises a type error in graph mode, or if the
    # cardinality is infinite or unknown.
    """Returns the length of the dataset if it is known and finite.

    This method requires that you are running in eager mode, and that the
    length of the dataset is known and non-infinite. When the length may be
    unknown or infinite, or if you are running in graph mode, use
    `tf.data.Dataset.cardinality` instead.

    Returns:
      An integer representing the length of the dataset.

    Raises:
      RuntimeError: If the dataset length is unknown or infinite, or if eager
        execution is not enabled.
    """
    if not context.executing_eagerly():
      raise TypeError("`tf.data.Dataset` only supports `len` in eager mode. "
                      "Use `tf.data.Dataset.cardinality()` instead.")
    length = self.cardinality()
    if length.numpy() == INFINITE:
      raise TypeError("The dataset is infinite.")
    if length.numpy() == UNKNOWN:
      raise TypeError("The dataset length is unknown.")
    return length

  @abc.abstractproperty
  def element_spec(self):
    # [FUNCTION DOC]
    # Abstract property (subclasses must provide it). Returns a nested structure of type
    # specifications matching the structure of one dataset element. The base version raises a
    # not-implemented error.
    """The type specification of an element of this dataset.

    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    >>> dataset.element_spec
    TensorSpec(shape=(), dtype=tf.int32, name=None)

    For more information,
    read [this guide](https://www.tensorflow.org/guide/data#dataset_structure).

    Returns:
      A (nested) structure of `tf.TypeSpec` objects matching the structure of an
      element of this dataset and specifying the type of individual components.
    """
    raise NotImplementedError(f"{type(self)}.element_spec()")

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a short text such as the concrete class name and the element spec. For a V1 adapter the
    # wrapped dataset's class name is used.
    type_ = type(self._dataset if isinstance(self, DatasetV1Adapter) else self)
    return f"<{type_.__name__} element_spec={self.element_spec}>"

  def __debug_string__(self):
    # [FUNCTION DOC]
    # Returns a multi-line debugging string showing this dataset and its transitive inputs, one line
    # per dataset (its representation), prefixed with dashes proportional to its depth, found with a
    # stack-based traversal. The format may change without warning.
    """Returns a string showing the type of the dataset and its inputs.

    This string is intended only for debugging purposes, and may change without
    warning.
    """
    lines = []
    to_process = [(self, 0)]  # Stack of (dataset, depth) pairs.
    while to_process:
      dataset, depth = to_process.pop()
      lines.append("-"*2*depth + repr(dataset))
      to_process.extend([(ds, depth+1) for ds in dataset._inputs()])  # pylint: disable=protected-access
    return "\n".join(lines)

  def as_numpy_iterator(self):
    # [FUNCTION DOC]
    # Returns a numpy iterator over the dataset elements with tensors converted to numpy arrays,
    # preserving nesting. Only works in eager mode and only when all components are tensors, ragged
    # tensors, sparse tensors or none-values. Raises a runtime error when not eager and a type error
    # for unsupported component types.
    """Returns an iterator which converts all elements of the dataset to numpy.

    Use `as_numpy_iterator` to inspect the content of your dataset. To see
    element shapes and types, print dataset elements directly instead of using
    `as_numpy_iterator`.

    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    >>> for element in dataset:
    ...   print(element)
    tf.Tensor(1, shape=(), dtype=int32)
    tf.Tensor(2, shape=(), dtype=int32)
    tf.Tensor(3, shape=(), dtype=int32)

    This method requires that you are running in eager mode and the dataset's
    element_spec contains only `TensorSpec` components.

    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    >>> for element in dataset.as_numpy_iterator():
    ...   print(element)
    1
    2
    3

    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    >>> [a.item() for a in dataset.as_numpy_iterator()]
    [1, 2, 3]

    `as_numpy_iterator()` will preserve the nested structure of dataset
    elements.

    >>> dataset = tf.data.Dataset.from_tensor_slices({'a': ([1, 2], [3, 4]),
    ...                                               'b': [5, 6]})
    >>> list(dataset.as_numpy_iterator()) == [{'a': (1, 3), 'b': 5},
    ...                                       {'a': (2, 4), 'b': 6}]
    True

    Returns:
      An iterable over the elements of the dataset, with their tensors converted
      to numpy arrays.

    Raises:
      TypeError: if an element contains a non-`Tensor` value.
      RuntimeError: if eager execution is not enabled.
    """
    if not context.executing_eagerly():
      raise RuntimeError("`tf.data.Dataset.as_numpy_iterator()` is only "
                         "supported in eager mode.")
    for component_spec in nest.flatten(self.element_spec):
      if not isinstance(
          component_spec,
          (tensor_spec.TensorSpec, ragged_tensor.RaggedTensorSpec,
           sparse_tensor_lib.SparseTensorSpec, none_tensor.NoneTensorSpec)):
        raise TypeError(
            f"`tf.data.Dataset.as_numpy_iterator()` is not supported for "
            f"datasets that produce values of type {component_spec.value_type}")

    return NumpyIterator(self)

  @property
  def _flat_shapes(self):
    # [FUNCTION DOC]
    # HELPER: property giving the flat list of tensor shapes for the tensor representation of an
    # element, derived from the element spec.
    """Returns a list `tf.TensorShapes`s for the element tensor representation.

    Returns:
      A list `tf.TensorShapes`s for the element tensor representation.
    """
    return structure.get_flat_tensor_shapes(self.element_spec)

  @property
  def _flat_types(self):
    # [FUNCTION DOC]
    # HELPER: property giving the flat list of tensor data types for the tensor representation of an
    # element, derived from the element spec.
    """Returns a list `tf.DType`s for the element tensor representation.

    Returns:
      A list `tf.DType`s for the element tensor representation.
    """
    return structure.get_flat_tensor_types(self.element_spec)

  @property
  def _flat_structure(self):
    # [FUNCTION DOC]
    # HELPER: property returning a mapping with the output shapes and output types attributes that
    # most dataset op constructors expect, so subclasses can pass them as keyword arguments.
    """Helper for setting `output_shapes` and `output_types` attrs of an op.

    Most dataset op constructors expect `output_shapes` and `output_types`
    arguments that represent the flattened structure of an element. This helper
    function generates these attrs as a keyword argument dictionary, allowing
    `Dataset._variant_tensor` implementations to pass `**self._flat_structure`
    to the op constructor.

    Returns:
      A dictionary of keyword arguments that can be passed to a dataset op
      constructor.
    """
    return {
        "output_shapes": self._flat_shapes,
        "output_types": self._flat_types,
    }

  @property
  def _metadata(self):
    # [FUNCTION DOC]
    # HELPER: property building a dataset metadata message, setting the validated name if the dataset
    # has a name. Relies on subclasses setting the name attribute.
    """Helper for generating dataset metadata."""
    metadata = dataset_metadata_pb2.Metadata()
    if self._name:
      metadata.name = _validate_and_encode(self._name)
    return metadata

  @property
  def _common_args(self):
    # [FUNCTION DOC]
    # HELPER: property returning a mapping of arguments common to most dataset ops: serialized
    # metadata, flat output shapes and flat output types.
    """Helper for generating arguments that are common across most dataset ops.

    Most dataset op constructors expect `output_shapes` and `output_types`
    arguments that represent the flattened structure of an element, as well as a
    `metadata` argument for additional metadata such as user-defined dataset
    name. This helper function generates common attributes as a keyword argument
    dictionary, allowing `Dataset._variant_tensor` implementations to pass
    `**self._common_args` to the op constructor.

    Returns:
      A dictionary of keyword arguments that can be passed to a dataset op
      constructor.
    """
    return {
        "metadata": self._metadata.SerializeToString(),
        "output_shapes": self._flat_shapes,
        "output_types": self._flat_types,
    }

  @property
  def _type_spec(self):
    # [FUNCTION DOC]
    # Composite-tensor hook: returns the dataset type specification built from this dataset's element
    # spec.
    return DatasetSpec(self.element_spec)

  @staticmethod
  def from_tensors(tensors, name=None) -> "DatasetV2":
    # [FUNCTION DOC]
    # Static factory: creates a dataset with a single element made of the given tensors (not sliced).
    # Inputs: tensors (any supported element structure), name (optional operation name).
    # Lazily imports the from-tensors op module and delegates to it. Returns the new dataset.
    """Creates a `Dataset` with a single element, comprising the given tensors.

    `from_tensors` produces a dataset containing only a single element. To slice
    the input tensor into multiple elements, use `from_tensor_slices` instead.

    >>> dataset = tf.data.Dataset.from_tensors([1, 2, 3])
    >>> list(dataset.as_numpy_iterator())
    [array([1, 2, 3], dtype=int32)]
    >>> dataset = tf.data.Dataset.from_tensors(([1, 2, 3], 'A'))
    >>> list(dataset.as_numpy_iterator())
    [(array([1, 2, 3], dtype=int32), b'A')]

    >>> # You can use `from_tensors` to produce a dataset which repeats
    >>> # the same example many times.
    >>> example = tf.constant([1,2,3])
    >>> dataset = tf.data.Dataset.from_tensors(example).repeat(2)
    >>> list(dataset.as_numpy_iterator())
    [array([1, 2, 3], dtype=int32), array([1, 2, 3], dtype=int32)]

    Note that if `tensors` contains a NumPy array, and eager execution is not
    enabled, the values will be embedded in the graph as one or more
    `tf.constant` operations. For large datasets (> 1 GB), this can waste
    memory and run into byte limits of graph serialization. If `tensors`
    contains one or more large NumPy arrays, consider the alternative described
    in [this
    guide](https://tensorflow.org/guide/data#consuming_numpy_arrays).

    Args:
      tensors: A dataset "element". Supported values are documented
        [here](https://www.tensorflow.org/guide/data#dataset_structure).
      name: (Optional.) A name for the tf.data operation.

    Returns:
      Dataset: A `Dataset`.
    """
    # Loaded lazily due to a circular dependency (dataset_ops ->
    # from_tensors_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import from_tensors_op
    return from_tensors_op._from_tensors(tensors, name)
    # pylint: enable=g-import-not-at-top,protected-access

  @staticmethod
  def from_tensor_slices(tensors, name=None) -> "DatasetV2":
    # [FUNCTION DOC]
    # Static factory: creates a dataset whose elements are slices of the given tensors along their
    # first dimension, preserving structure; all components need the same first dimension size.
    # Inputs: tensors (element structure), name (optional). Delegates to the lazily imported
    # from-tensor-slices op module. Returns the new dataset.
    """Creates a `Dataset` whose elements are slices of the given tensors.

    The given tensors are sliced along their first dimension. This operation
    preserves the structure of the input tensors, removing the first dimension
    of each tensor and using it as the dataset dimension. All input tensors
    must have the same size in their first dimensions.

    >>> # Slicing a 1D tensor produces scalar tensor elements.
    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    >>> [a.item() for a in dataset.as_numpy_iterator()]
    [1, 2, 3]

    >>> # Slicing a 2D tensor produces 1D tensor elements.
    >>> dataset = tf.data.Dataset.from_tensor_slices([[1, 2], [3, 4]])
    >>> list(dataset.as_numpy_iterator())
    [array([1, 2], dtype=int32), array([3, 4], dtype=int32)]

    >>> # Slicing a tuple of 1D tensors produces tuple elements containing
    >>> # scalar tensors.
    >>> dataset = tf.data.Dataset.from_tensor_slices(([1, 2], [3, 4], [5, 6]))
    >>> [(n0.item(), n1.item(), n2.item()) for n0, n1, n2 in
    ...        dataset.as_numpy_iterator()]
    [(1, 3, 5), (2, 4, 6)]

    >>> # Dictionary structure is also preserved.
    >>> dataset = tf.data.Dataset.from_tensor_slices({"a": [1, 2], "b": [3, 4]})
    >>> list(dataset.as_numpy_iterator()) == [{'a': 1, 'b': 3},
    ...                                       {'a': 2, 'b': 4}]
    True

    >>> # Two tensors can be combined into one Dataset object.
    >>> features = tf.constant([[1, 3], [2, 1], [3, 3]]) # ==> 3x2 tensor
    >>> labels = tf.constant(['A', 'B', 'A']) # ==> 3x1 tensor
    >>> dataset = Dataset.from_tensor_slices((features, labels))
    >>> # Both the features and the labels tensors can be converted
    >>> # to a Dataset object separately and combined after.
    >>> features_dataset = Dataset.from_tensor_slices(features)
    >>> labels_dataset = Dataset.from_tensor_slices(labels)
    >>> dataset = Dataset.zip((features_dataset, labels_dataset))
    >>> # A batched feature and label set can be converted to a Dataset
    >>> # in similar fashion.
    >>> batched_features = tf.constant([[[1, 3], [2, 3]],
    ...                                 [[2, 1], [1, 2]],
    ...                                 [[3, 3], [3, 2]]], shape=(3, 2, 2))
    >>> batched_labels = tf.constant([['A', 'A'],
    ...                               ['B', 'B'],
    ...                               ['A', 'B']], shape=(3, 2, 1))
    >>> dataset = Dataset.from_tensor_slices((batched_features, batched_labels))
    >>> for element in dataset.as_numpy_iterator():
    ...   print(element)
    (array([[1, 3],
           [2, 3]], dtype=int32), array([[b'A'],
           [b'A']], dtype=object))
    (array([[2, 1],
           [1, 2]], dtype=int32), array([[b'B'],
           [b'B']], dtype=object))
    (array([[3, 3],
           [3, 2]], dtype=int32), array([[b'A'],
           [b'B']], dtype=object))

    Note that if `tensors` contains a NumPy array, and eager execution is not
    enabled, the values will be embedded in the graph as one or more
    `tf.constant` operations. For large datasets (> 1 GB), this can waste
    memory and run into byte limits of graph serialization. If `tensors`
    contains one or more large NumPy arrays, consider the alternative described
    in [this guide](
    https://tensorflow.org/guide/data#consuming_numpy_arrays).

    Args:
      tensors: A dataset element, whose components have the same first
        dimension. Supported values are documented
        [here](https://www.tensorflow.org/guide/data#dataset_structure).
      name: (Optional.) A name for the tf.data operation.

    Returns:
      Dataset: A `Dataset`.
    """
    # Loaded lazily due to a circular dependency (dataset_ops ->
    # from_tensor_slices_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import from_tensor_slices_op
    return from_tensor_slices_op._from_tensor_slices(tensors, name)
    # pylint: enable=g-import-not-at-top,protected-access

  class _GeneratorState:
    # [CLASS DOC]
    # HELPER: internal bookkeeping for the generator-based dataset.
    # Tracks outstanding iterators created from a Python generator callable, since a generator may be
    # iterated several times at once (repeat or parallel use). Holds the generator, a lock-protected
    # counter of iterator ids, the arguments saved for each id and the live iterators by id.
    """Stores outstanding iterators created from a Python generator.

    This class keeps track of potentially multiple iterators that may have
    been created from a generator, e.g. in the case that the dataset is
    repeated, or nested within a parallel computation.
    """

    def __init__(self, generator):
      # [FUNCTION DOC]
      # Stores the generator callable and initializes the lock, the next-id counter (zero), and empty
      # mappings for arguments and iterators.
      self._generator = generator
      self._lock = threading.Lock()
      self._next_id = 0  # GUARDED_BY(self._lock)
      self._args = {}
      self._iterators = {}

    def _normalize_id(self, iterator_id):
      # [FUNCTION DOC]
      # HELPER: converts an iterator id that may be a numpy array (as in debug mode) into a plain
      # scalar so it can be used as a mapping key; other values are returned unchanged.
      # In debug mode, iterator ids may be eagerly-generated np.arrays instead
      # of Tensors. We convert them to scalars to make them hashable.
      if isinstance(iterator_id, np.ndarray):
        return iterator_id.item()
      return iterator_id

    def get_next_id(self, *args):
      # [FUNCTION DOC]
      # Allocates a new iterator id under the lock, remembers the given arguments for it, and returns
      # the id as a 64-bit integer numpy array (explicitly 64-bit because implicit casting on Windows
      # would make it 32-bit).
      with self._lock:
        ret = self._next_id
        self._next_id += 1
      self._args[ret] = args
      # NOTE(mrry): Explicitly create an array of `np.int64` because implicit
      # casting in `py_func()` will create an array of `np.int32` on Windows,
      # leading to a runtime error.
      return np.array(ret, dtype=np.int64)

    def get_iterator(self, iterator_id):
      # [FUNCTION DOC]
      # Returns the iterator for the given id. On first request it pops the stored arguments for that
      # id, calls the generator with them, stores and returns the iterator; later calls return the
      # stored iterator. Raises a key error if the id was never allocated (or its arguments were used).
      iterator_id = self._normalize_id(iterator_id)
      try:
        return self._iterators[iterator_id]
      except KeyError:
        iterator = iter(self._generator(*self._args.pop(iterator_id)))
        self._iterators[iterator_id] = iterator
        return iterator

    def iterator_completed(self, iterator_id):
      # [FUNCTION DOC]
      # Removes the finished iterator for the given id. Raises a key error if there is no such iterator.
      del self._iterators[self._normalize_id(iterator_id)]

  @staticmethod
  @deprecation.deprecated_args(None, "Use output_signature instead",
                               "output_types", "output_shapes")
  def from_generator(
      generator,
      output_types=None,
      output_shapes=None,
      args=None,
      output_signature=None,
      name=None,
  ) -> "DatasetV2":
    # [FUNCTION DOC]
    # Static factory: creates a dataset whose elements are produced by a generator callable.
    # Inputs: generator (callable returning an iterable), output_types and output_shapes (deprecated,
    # optional), args (optional tensors passed to the generator), output_signature (optional nested
    # type specifications, the recommended way), name (optional). Delegates to the lazily imported
    # generator op module (which uses python-function ops, so it must run in the same process).
    # Returns the new dataset.
    """Creates a `Dataset` whose elements are generated by `generator`.

    Note: The current implementation of `Dataset.from_generator()` uses
    `tf.numpy_function` and inherits the same constraints. In particular, it
    requires the dataset and iterator related operations to be placed
    on a device in the same process as the Python program that called
    `Dataset.from_generator()`. In particular, using `from_generator` will
    preclude the use of tf.data service for scaling out dataset processing.
    The body of `generator` will not be serialized in a `GraphDef`, and you
    should not use this method if you need to serialize your model and restore
    it in a different environment.

    The `generator` argument must be a callable object that returns
    an object that supports the `iter()` protocol (e.g. a generator function).

    The elements generated by `generator` must be compatible with either the
    given `output_signature` argument or with the given `output_types` and
    (optionally) `output_shapes` arguments, whichever was specified.

    The recommended way to call `from_generator` is to use the
    `output_signature` argument. In this case the output will be assumed to
    consist of objects with the classes, shapes and types defined by
    `tf.TypeSpec` objects from `output_signature` argument:

    >>> def gen():
    ...   ragged_tensor = tf.ragged.constant([[1, 2], [3]])
    ...   yield 42, ragged_tensor
    >>>
    >>> dataset = tf.data.Dataset.from_generator(
    ...      gen,
    ...      output_signature=(
    ...          tf.TensorSpec(shape=(), dtype=tf.int32),
    ...          tf.RaggedTensorSpec(shape=(2, None), dtype=tf.int32)))
    >>>
    >>> list(dataset.take(1))
    [(<tf.Tensor: shape=(), dtype=int32, numpy=42>,
    <tf.RaggedTensor [[1, 2], [3]]>)]

    There is also a deprecated way to call `from_generator` by either with
    `output_types` argument alone or together with `output_shapes` argument.
    In this case the output of the function will be assumed to consist of
    `tf.Tensor` objects with the types defined by `output_types` and with the
    shapes which are either unknown or defined by `output_shapes`.

    Note: If `generator` depends on mutable global variables or other external
    state, be aware that the runtime may invoke `generator` multiple times
    (in order to support repeating the `Dataset`) and at any time
    between the call to `Dataset.from_generator()` and the production of the
    first element from the generator. Mutating global variables or external
    state can cause undefined behavior, and we recommend that you explicitly
    cache any external state in `generator` before calling
    `Dataset.from_generator()`.

    Note: While the `output_signature` parameter makes it possible to yield
    `Dataset` elements, the scope of `Dataset.from_generator()` should be
    limited to logic that cannot be expressed through tf.data operations. Using
    tf.data operations within the generator function is an anti-pattern and may
    result in incremental memory growth.

    Args:
      generator: A callable object that returns an object that supports the
        `iter()` protocol. If `args` is not specified, `generator` must take no
        arguments; otherwise it must take as many arguments as there are values
        in `args`.
      output_types: (Optional.) A (nested) structure of `tf.DType` objects
        corresponding to each component of an element yielded by `generator`.
      output_shapes: (Optional.) A (nested) structure of `tf.TensorShape`
        objects corresponding to each component of an element yielded by
        `generator`.
      args: (Optional.) A tuple of `tf.Tensor` objects that will be evaluated
        and passed to `generator` as NumPy-array arguments.
      output_signature: (Optional.) A (nested) structure of `tf.TypeSpec`
        objects corresponding to each component of an element yielded by
        `generator`.
      name: (Optional.) A name for the tf.data operations used by
        `from_generator`.

    Returns:
      Dataset: A `Dataset`.
    """
    # Loaded lazily due to a circular dependency (dataset_ops ->
    # from_generator_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import from_generator_op
    return from_generator_op._from_generator(generator, output_types,
                                             output_shapes, args,
                                             output_signature, name)
    # pylint: enable=g-import-not-at-top,protected-access

  @staticmethod
  def range(*args, **kwargs) -> "DatasetV2":
    # [FUNCTION DOC]
    # Static factory: creates a dataset of a range of integers like Python's range. Positional
    # arguments give stop, start and stop, or start, stop and step; keyword arguments may give the
    # output type (default 64-bit integer) and name. Delegates to the lazily imported range op module
    # and returns the dataset. That module raises a value error when no positional arguments given.
    """Creates a `Dataset` of a step-separated range of values.

    >>> ds = Dataset.range(5)
    >>> [a.item() for a in ds.as_numpy_iterator()]
    [0, 1, 2, 3, 4]
    >>> ds = Dataset.range(2, 5)
    >>> [a.item() for a in ds.as_numpy_iterator()]
    [2, 3, 4]
    >>> ds = Dataset.range(1, 5, 2)
    >>> [a.item() for a in ds.as_numpy_iterator()]
    [1, 3]
    >>> ds = Dataset.range(1, 5, -2)
    >>> [a.item() for a in ds.as_numpy_iterator()]
    []
    >>> ds = Dataset.range(5, 1)
    >>> [a.item() for a in ds.as_numpy_iterator()]
    []
    >>> ds = Dataset.range(5, 1, -2)
    >>> [a.item() for a in ds.as_numpy_iterator()]
    [5, 3]
    >>> ds = Dataset.range(2, 5, output_type=tf.int32)
    >>> [a.item() for a in ds.as_numpy_iterator()]
    [2, 3, 4]
    >>> ds = Dataset.range(1, 5, 2, output_type=tf.float32)
    >>> [a.item() for a in ds.as_numpy_iterator()]
    [1.0, 3.0]

    Args:
      *args: follows the same semantics as python's range.
        len(args) == 1 -> start = 0, stop = args[0], step = 1.
        len(args) == 2 -> start = args[0], stop = args[1], step = 1.
        len(args) == 3 -> start = args[0], stop = args[1], step = args[2].
      **kwargs:
        - output_type: Its expected dtype. (Optional, default: `tf.int64`).
        - name: (Optional.) A name for the tf.data operation.

    Returns:
      Dataset: A `RangeDataset`.

    Raises:
      ValueError: if len(args) == 0.
    """
    # Loaded lazily due to a circular dependency (dataset_ops -> range_op ->
    # -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import range_op
    return range_op._range(*args, **kwargs)
    # pylint: enable=g-import-not-at-top,protected-access

  @staticmethod
  def zip(*args, datasets=None, name=None) -> "DatasetV2":
    # [FUNCTION DOC]
    # Static factory: zips datasets (or nested structures of datasets) element-wise; stops at the
    # shortest. Inputs: positional datasets or structures, or the legacy keyword datasets (cannot be
    # combined with positional ones), and name (optional). Raises a type error if nothing is given or
    # both forms are given. A single positional argument is used as the structure; several are
    # grouped as a tuple. Returns the zipped dataset.
    """Creates a `Dataset` by zipping together the given datasets.

    This method has similar semantics to the built-in `zip()` function
    in Python, with the main difference being that the `datasets`
    argument can be a (nested) structure of `Dataset` objects. The supported
    nesting mechanisms are documented
    [here] (https://www.tensorflow.org/guide/data#dataset_structure).

    >>> # The datasets or nested structure of datasets `*args` argument
    >>> # determines the structure of elements in the resulting dataset.
    >>> a = tf.data.Dataset.range(1, 4)  # ==> [ 1, 2, 3 ]
    >>> b = tf.data.Dataset.range(4, 7)  # ==> [ 4, 5, 6 ]
    >>> ds = tf.data.Dataset.zip(a, b)
    >>> [(i.item(), j.item()) for i, j in ds.as_numpy_iterator()]
    [(1, 4), (2, 5), (3, 6)]
    >>> ds = tf.data.Dataset.zip(b, a)
    >>> [(i.item(), j.item()) for i, j in ds.as_numpy_iterator()]
    [(4, 1), (5, 2), (6, 3)]
    >>>
    >>> # The `datasets` argument may contain an arbitrary number of datasets.
    >>> c = tf.data.Dataset.range(7, 13).batch(2)  # ==> [ [7, 8],
    ...                                            #       [9, 10],
    ...                                            #       [11, 12] ]
    >>> ds = tf.data.Dataset.zip(a, b, c)
    >>> for i, j, k in ds.as_numpy_iterator():
    ...   print(i.item(), j.item(), k)
    1 4 [7 8]
    2 5 [ 9 10]
    3 6 [11 12]
    >>>
    >>> # The number of elements in the resulting dataset is the same as
    >>> # the size of the smallest dataset in `datasets`.
    >>> d = tf.data.Dataset.range(13, 15)  # ==> [ 13, 14 ]
    >>> ds = tf.data.Dataset.zip(a, d)
    >>> [(i.item(), j.item()) for i, j in ds.as_numpy_iterator()]
    [(1, 13), (2, 14)]

    Args:
      *args: Datasets or nested structures of datasets to zip together. This
        can't be set if `datasets` is set.
      datasets: A (nested) structure of datasets. This can't be set if `*args`
        is set. Note that this exists only for backwards compatibility and it is
        preferred to use *args.
      name: (Optional.) A name for the tf.data operation.

    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    # Loaded lazily due to a circular dependency (dataset_ops -> zip_op ->
    # dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import zip_op

    if not args and datasets is None:
      raise TypeError("Must pass at least one dataset to `zip`.")
    if args and datasets is not None:
      raise TypeError("Both `*args` and `datasets` cannot be set.")
    if len(args) == 1:
      datasets = args[0]
    elif len(args) > 1:
      datasets = args
    return zip_op._zip(datasets, name)
    # pylint: enable=g-import-not-at-top,protected-access

  def concatenate(self, dataset, name=None) -> "DatasetV2":
    # [FUNCTION DOC]
    # Returns a dataset with the elements of this dataset followed by those of the given dataset,
    # which must have a compatible element spec (otherwise the delegate raises a type error).
    # Inputs: dataset to append, name (optional).
    """Creates a `Dataset` by concatenating the given dataset with this dataset.

    >>> a = tf.data.Dataset.range(1, 4)  # ==> [ 1, 2, 3 ]
    >>> b = tf.data.Dataset.range(4, 8)  # ==> [ 4, 5, 6, 7 ]
    >>> ds = a.concatenate(b)
    >>> [a.item() for a in ds.as_numpy_iterator()]
    [1, 2, 3, 4, 5, 6, 7]
    >>> # The input dataset and dataset to be concatenated should have
    >>> # compatible element specs.
    >>> c = tf.data.Dataset.zip((a, b))
    >>> a.concatenate(c)
    Traceback (most recent call last):
    TypeError: Two datasets to concatenate have different types
    <dtype: 'int64'> and (tf.int64, tf.int64)
    >>> d = tf.data.Dataset.from_tensor_slices(["a", "b", "c"])
    >>> a.concatenate(d)
    Traceback (most recent call last):
    TypeError: Two datasets to concatenate have different types
    <dtype: 'int64'> and <dtype: 'string'>

    Args:
      dataset: `Dataset` to be concatenated.
      name: (Optional.) A name for the tf.data operation.

    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    # Loaded lazily due to a circular dependency (dataset_ops ->
    # concatenate_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import concatenate_op
    return concatenate_op._concatenate(self, dataset, name)
    # pylint: enable=g-import-not-at-top,protected-access

  @staticmethod
  def counter(start=0, step=1, dtype=dtypes.int64, name=None) -> "DatasetV2":
    # [FUNCTION DOC]
    # Static factory: creates an unbounded dataset counting from start (default 0) in steps of step
    # (default 1) with element type dtype (default 64-bit integer); name optional. Delegates to the
    # lazily imported counter op module and returns the dataset.
    """Creates a `Dataset` that counts from `start` in steps of size `step`.

    Unlike `tf.data.Dataset.range`, which stops at some ending number,
    `tf.data.Dataset.counter` produces elements indefinitely.

    >>> dataset = tf.data.experimental.Counter().take(5)
    >>> [a.item() for a in dataset.as_numpy_iterator()]
    [0, 1, 2, 3, 4]
    >>> dataset.element_spec
    TensorSpec(shape=(), dtype=tf.int64, name=None)
    >>> dataset = tf.data.experimental.Counter(dtype=tf.int32)
    >>> dataset.element_spec
    TensorSpec(shape=(), dtype=tf.int32, name=None)
    >>> dataset = tf.data.experimental.Counter(start=2).take(5)
    >>> [a.item() for a in dataset.as_numpy_iterator()]
    [2, 3, 4, 5, 6]
    >>> dataset = tf.data.experimental.Counter(start=2, step=5).take(5)
    >>> [a.item() for a in dataset.as_numpy_iterator()]
    [2, 7, 12, 17, 22]
    >>> dataset = tf.data.experimental.Counter(start=10, step=-1).take(5)
    >>> [a.item() for a in dataset.as_numpy_iterator()]
    [10, 9, 8, 7, 6]

    Args:
      start: (Optional.) The starting value for the counter. Defaults to 0.
      step: (Optional.) The step size for the counter. Defaults to 1.
      dtype: (Optional.) The data type for counter elements. Defaults to
        `tf.int64`.
      name: (Optional.) A name for the tf.data operation.

    Returns:
      A `Dataset` of scalar `dtype` elements.
    """
    # Loaded lazily due to a circular dependency (dataset_ops -> counter_op
    # -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import counter_op
    return counter_op._counter(start, step, dtype, name=name)
    # pylint: enable=g-import-not-at-top,protected-access

  def fingerprint(self):
    # [FUNCTION DOC]
    # Returns a scalar unsigned 64-bit tensor fingerprint of this dataset computed from its graph.
    # Equal fingerprints guarantee identical elements (given deterministic data); different
    # fingerprints do not guarantee different elements.
    """Computes the fingerprint of this `Dataset`.

    If two datasets have the same fingerprint, it is guaranteed that they
    would produce identical elements as long as the content of the upstream
    input files does not change and they produce data deterministically.

    However, two datasets producing identical values does not always mean they
    would have the same fingerprint due to different graph constructs.

    In other words, if two datasets have different fingerprints, they could
    still produce identical values.

    Returns:
      A scalar `tf.Tensor` of type `tf.uint64`.
    """
    return gen_dataset_ops.dataset_fingerprint(self._variant_tensor)

  def rebatch(self, batch_size, drop_remainder=False, name=None) -> "DatasetV2":
    # [FUNCTION DOC]
    # Returns a dataset that re-batches elements of this batched dataset to new batch sizes.
    # Inputs: batch_size (scalar, or a vector whose entries are cycled through), drop_remainder
    # (optional flag, default false, to drop a last short batch), name (optional). Delegates to the
    # lazily imported rebatch op module.
    """Creates a `Dataset` that rebatches the elements from this dataset.

    `rebatch(N)` is functionally equivalent to `unbatch().batch(N)`, but is
    more efficient, performing one copy instead of two.

    >>> ds = tf.data.Dataset.range(6)
    >>> ds = ds.batch(2)
    >>> ds = ds.rebatch(3)
    >>> list(ds.as_numpy_iterator())
    [array([0, 1, 2]), array([3, 4, 5])]

    >>> ds = tf.data.Dataset.range(7)
    >>> ds = ds.batch(4)
    >>> ds = ds.rebatch(3)
    >>> list(ds.as_numpy_iterator())
    [array([0, 1, 2]), array([3, 4, 5]), array([6])]

    >>> ds = tf.data.Dataset.range(7)
    >>> ds = ds.batch(2)
    >>> ds = ds.rebatch(3, drop_remainder=True)
    >>> list(ds.as_numpy_iterator())
    [array([0, 1, 2]), array([3, 4, 5])]

    If the `batch_size` argument is a list, `rebatch` cycles through the list
    to determine the size of each batch.

    >>> ds = tf.data.Dataset.range(8)
    >>> ds = ds.batch(4)
    >>> ds = ds.rebatch([2, 1, 1])
    >>> list(ds.as_numpy_iterator())
    [array([0, 1]), array([2]), array([3]), array([4, 5]), array([6]),
    array([7])]

    Args:
      batch_size: A `tf.int64` scalar or vector, representing the size of
        batches to produce. If this argument is a vector, these values are
        cycled through in round robin fashion.
      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
        whether the last batch should be dropped in the case it has fewer than
        `batch_size[cycle_index]` elements; the default behavior is not to drop
        the smaller batch.
      name: (Optional.) A name for the tf.data operation.

    Returns:
      A `Dataset` of scalar `dtype` elements.
    """
    # Loaded lazily due to a circular dependency (dataset_ops -> rebatch_op ->
    # rebatch_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import rebatch_op
    return rebatch_op._rebatch(self, batch_size, drop_remainder, name=name)
    # pylint: enable=g-import-not-at-top,protected-access

  def prefetch(self, buffer_size, name=None) -> "DatasetV2":
    # [FUNCTION DOC]
    # Returns a dataset that prefetches elements while earlier ones are processed. Input buffer_size
    # is the maximum number of buffered elements (the autotune constant lets the buffer be tuned
    # dynamically); name optional. Delegates to the lazily loaded prefetch op module.
    """Creates a `Dataset` that prefetches elements from this dataset.

    Most dataset input pipelines should end with a call to `prefetch`. This
    allows later elements to be prepared while the current element is being
    processed. This often improves latency and throughput, at the cost of
    using additional memory to store prefetched elements.

    Note: Like other `Dataset` methods, prefetch operates on the
    elements of the input dataset. It has no concept of examples vs. batches.
    `examples.prefetch(2)` will prefetch two elements (2 examples),
    while `examples.batch(20).prefetch(2)` will prefetch 2 elements
    (2 batches, of 20 examples each).

    >>> dataset = tf.data.Dataset.range(3)
    >>> dataset = dataset.prefetch(2)
    >>> [a.item() for a in dataset.as_numpy_iterator()]
    [0, 1, 2]

    Args:
      buffer_size: A `tf.int64` scalar `tf.Tensor`, representing the maximum
        number of elements that will be buffered when prefetching. If the value
        `tf.data.AUTOTUNE` is used, then the buffer size is dynamically tuned.
      name: Optional. A name for the tf.data transformation.

    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    return prefetch_op._prefetch(  # pylint: disable=protected-access
        self, buffer_size, name=name)

  @staticmethod
  def list_files(
      file_pattern, shuffle=None, seed=None, name=None
  ) -> "DatasetV2":
    # [FUNCTION DOC]
    # Static factory: creates a dataset of file names matching one or more glob patterns.
    # Inputs: file_pattern (string, list or string tensor), shuffle (optional; unset means true),
    # seed (optional random seed), name (optional). Matches the files, asserts at run time that at
    # least one file matched (failing with a message naming the patterns otherwise), builds a
    # dataset of the names (wrapped in a V1 adapter if the default dataset class is V1), and if
    # shuffling shuffles it with a buffer size equal to the number of files (at least one).
    # Returns the dataset.
    """A dataset of all files matching one or more glob patterns.

    The `file_pattern` argument should be a small number of glob patterns.
    If your filenames have already been globbed, use
    `Dataset.from_tensor_slices(filenames)` instead, as re-globbing every
    filename with `list_files` may result in poor performance with remote
    storage systems.

    Note: The default behavior of this method is to return filenames in
    a non-deterministic random shuffled order. Pass a `seed` or `shuffle=False`
    to get results in a deterministic order.

    Example:
      If we had the following files on our filesystem:

        - /path/to/dir/a.txt
        - /path/to/dir/b.py
        - /path/to/dir/c.py

      If we pass "/path/to/dir/*.py" as the directory, the dataset
      would produce:

        - /path/to/dir/b.py
        - /path/to/dir/c.py

    Args:
      file_pattern: A string, a list of strings, or a `tf.Tensor` of string type
        (scalar or vector), representing the filename glob (i.e. shell wildcard)
        pattern(s) that will be matched.
      shuffle: (Optional.) If `True`, the file names will be shuffled randomly.
        Defaults to `True`.
      seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random
        seed that will be used to create the distribution. See
        `tf.random.set_seed` for behavior.
      name: Optional. A name for the tf.data operations used by `list_files`.

    Returns:
     Dataset: A `Dataset` of strings corresponding to file names.
    """
    with ops.name_scope("list_files"):
      if shuffle is None:
        shuffle = True
      file_pattern = ops.convert_to_tensor(
          file_pattern, dtype=dtypes.string, name="file_pattern")
      matching_files = gen_io_ops.matching_files(file_pattern)

      # Raise an exception if `file_pattern` does not match any files.
      condition = math_ops.greater(array_ops.shape(matching_files)[0], 0,
                                   name="match_not_empty")

      message = math_ops.add(
          "No files matched pattern: ",
          string_ops.reduce_join(file_pattern, separator=", "), name="message")

      assert_not_empty = control_flow_assert.Assert(
          condition, [message], summarize=1, name="assert_not_empty")
      with ops.control_dependencies([assert_not_empty]):
        matching_files = array_ops.identity(matching_files)

      # TODO(b/240947712): Remove lazy import after this method is factored out.
      # Loaded lazily due to a circular dependency (dataset_ops ->
      # from_tensor_slices_op -> dataset_ops).
      # pylint: disable=g-import-not-at-top,protected-access
      from tensorflow.python.data.ops import from_tensor_slices_op
      dataset = from_tensor_slices_op._TensorSliceDataset(
          matching_files, is_files=True, name=name)
      # pylint: enable=g-import-not-at-top,protected-access
      if issubclass(Dataset, DatasetV1):
        dataset = DatasetV1Adapter(dataset)
      if shuffle:
        # NOTE(mrry): The shuffle buffer size must be greater than zero, but the
        # list of files might be empty.
        buffer_size = math_ops.maximum(
            array_ops.shape(matching_files, out_type=dtypes.int64)[0], 1)
        dataset = dataset.shuffle(buffer_size, seed=seed, name=name)
      return dataset

  def repeat(self, count=None, name=None) -> "DatasetV2":
    # [FUNCTION DOC]
    # Returns a dataset that repeats this dataset count times; count unset or minus one repeats
    # forever. Name optional. Delegates to the lazily imported repeat op module.
    """Repeats this dataset so each original value is seen `count` times.

    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    >>> dataset = dataset.repeat(3)
    >>> [a.item() for a in dataset.as_numpy_iterator()]
    [1, 2, 3, 1, 2, 3, 1, 2, 3]

    Note: If the input dataset depends on global state (e.g. a random number
    generator) or its output is non-deterministic (e.g. because of upstream
    `shuffle`), then different repetitions may produce different elements.

    Args:
      count: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
        number of times the dataset should be repeated. The default behavior (if
        `count` is `None` or `-1`) is for the dataset be repeated indefinitely.
      name: (Optional.) A name for the tf.data operation.

    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    # Loaded lazily due to a circular dependency (dataset_ops -> repeat_op ->
    # dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access,redefined-outer-name
    from tensorflow.python.data.ops import repeat_op
    return repeat_op._repeat(self, count, name)
    # pylint: enable=g-import-not-at-top,protected-access,redefined-outer-name

  def enumerate(self, start=0, name=None) -> "DatasetV2":
    # [FUNCTION DOC]
    # Returns a dataset of pairs of an index counter and the original element. Inputs: start (first
    # index, default 0), name (optional). Builds a range dataset from start up to the largest 64-bit
    # integer, applies the replicate-on-split rewrite so that each split is enumerated independently,
    # and zips it with this dataset.
    """Enumerates the elements of this dataset.

    It is similar to python's `enumerate`.

    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    >>> dataset = dataset.enumerate(start=5)
    >>> for pos, element in dataset.as_numpy_iterator():
    ...   print(tuple((pos.item(), element.item())))
    (5, 1)
    (6, 2)
    (7, 3)

    >>> # The (nested) structure of the input dataset determines the
    >>> # structure of elements in the resulting dataset.
    >>> dataset = tf.data.Dataset.from_tensor_slices([(7, 8), (9, 10)])
    >>> dataset = dataset.enumerate()
    >>> for pos, element in dataset.as_numpy_iterator():
    ...   print(tuple((pos.item(), element)))
    (0, array([7, 8], dtype=int32))
    (1, array([ 9, 10], dtype=int32))

    Args:
      start: A `tf.int64` scalar `tf.Tensor`, representing the start value for
        enumeration.
      name: Optional. A name for the tf.data operations used by `enumerate`.

    Returns:
      A new `Dataset` with the transformation applied as described above.
    """

    max_value = np.iinfo(dtypes.int64.as_numpy_dtype).max
    range_dataset = Dataset.range(start, max_value, name=name)
    # Replicate the range component so that each split is enumerated
    # independently. This avoids the need for prohibitively expensive
    # cross-split coordination.
    range_dataset = apply_rewrite(range_dataset, "replicate_on_split")
    return Dataset.zip((range_dataset, self), name=name)

  def shuffle(
      self, buffer_size, seed=None, reshuffle_each_iteration=True, name=None
  ) -> "DatasetV2":
    # [FUNCTION DOC]
    # Returns a dataset that randomly shuffles elements using a buffer of buffer_size elements.
    # Inputs: buffer_size (number of elements to sample from; use the full cardinality for a uniform
    # shuffle), seed (optional), reshuffle_each_iteration (optional, default true: use a different
    # order each epoch), name (optional). Delegates to the lazily loaded shuffle op module.
    """Randomly shuffles the elements of this dataset.

    This dataset fills a buffer with `buffer_size` elements, then randomly
    samples elements from this buffer, replacing the selected elements with new
    elements. For perfect shuffling, a buffer size greater than or equal to the
    full size of the dataset is required.

    For instance, if your dataset contains 10,000 elements but `buffer_size` is
    set to 1,000, then `shuffle` will initially select a random element from
    only the first 1,000 elements in the buffer. Once an element is selected,
    its space in the buffer is replaced by the next (i.e. 1,001-st) element,
    maintaining the 1,000 element buffer.

    `reshuffle_each_iteration` controls whether the shuffle order should be
    different for each epoch. However you should avoid using
    `shuffle(reshuffle_each_iteration=True)`, then `take` and `skip` to split
    a dataset into training and test sets, which would lead to data leakage (as
    the entire dataset would be re-shuffled then re-split after each epoch).
    Please use the `tf.keras.utils.split_dataset` method instead. In TF 1.X,
    the idiomatic way to create epochs was through the `repeat` transformation:

    ```python
    dataset = tf.data.Dataset.range(3)
    dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
    dataset = dataset.repeat(2)
    # [1, 0, 2, 1, 2, 0]

    dataset = tf.data.Dataset.range(3)
    dataset = dataset.shuffle(3, reshuffle_each_iteration=False)
    dataset = dataset.repeat(2)
    # [1, 0, 2, 1, 0, 2]
    ```

    In TF 2.0, `tf.data.Dataset` objects are Python iterables which makes it
    possible to also create epochs through Python iteration:

    ```python
    dataset = tf.data.Dataset.range(3)
    dataset = dataset.shuffle(3, reshuffle_each_iteration=True)
    list(dataset.as_numpy_iterator())
    # [1, 0, 2]
    list(dataset.as_numpy_iterator())
    # [1, 2, 0]
    ```

    ```python
    dataset = tf.data.Dataset.range(3)
    dataset = dataset.shuffle(3, reshuffle_each_iteration=False)
    list(dataset.as_numpy_iterator())
    # [1, 0, 2]
    list(dataset.as_numpy_iterator())
    # [1, 0, 2]
    ```

    #### Fully shuffling all the data

    To shuffle an entire dataset, set `buffer_size=dataset.cardinality()`. This
    is equivalent to setting the `buffer_size` equal to the number of elements
    in the dataset, resulting in uniform shuffle.

    Note: `shuffle(dataset.cardinality())` loads the full dataset into memory so
    that it can be shuffled. This will cause a memory overflow (OOM) error if
    the dataset is too large, so full-shuffle should only be used for datasets
    that are known to fit in the memory, such as datasets of filenames or other
    small datasets.

    ```python
    dataset = tf.data.Dataset.range(20)
    dataset = dataset.shuffle(dataset.cardinality())
    # [18, 4, 9, 2, 17, 8, 5, 10, 0, 6, 16, 3, 19, 7, 14, 11, 15, 13, 12, 1]
    ```

    Args:
      buffer_size: An int or `tf.int64` scalar `tf.Tensor`, representing the
        number of elements from this dataset from which the new dataset will
        sample. To uniformly shuffle the entire dataset, use
        `buffer_size=dataset.cardinality()`.
      seed: (Optional.) An int or `tf.int64` scalar `tf.Tensor`, representing
        the random seed that will be used to create the distribution. See
        `tf.random.set_seed` for behavior.
      reshuffle_each_iteration: (Optional.) A boolean, which if true indicates
        that the dataset should be pseudorandomly reshuffled each time it is
        iterated over. (Defaults to `True`.)
      name: (Optional.) A name for the tf.data operation.

    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    return shuffle_op._shuffle(  # pylint: disable=protected-access
        self, buffer_size, seed, reshuffle_each_iteration, name=name)

  def cache(self, filename="", name=None) -> "DatasetV2":
    # [FUNCTION DOC]
    # Returns a dataset that caches elements on first full iteration, in memory (default empty
    # filename) or in files at the given filename prefix; later iterations read from the cache.
    # Name optional. Delegates to the lazily imported cache op module.
    """Caches the elements in this dataset.

    The first time the dataset is iterated over, its elements will be cached
    either in the specified file or in memory. Subsequent iterations will
    use the cached data.

    Note: To guarantee that the cache gets finalized, the input dataset must be
    iterated through in its entirety, until it raises StopIteration. Otherwise,
    subsequent iterations may not use cached data.

    >>> dataset = tf.data.Dataset.range(5)
    >>> dataset = dataset.map(lambda x: x**2)
    >>> dataset = dataset.cache()
    >>> # The first time reading through the data will generate the data using
    >>> # `range` and `map`.
    >>> [a.item() for a in dataset.as_numpy_iterator()]
    [0, 1, 4, 9, 16]
    >>> # Subsequent iterations read from the cache.
    >>> [a.item() for a in dataset.as_numpy_iterator()]
    [0, 1, 4, 9, 16]

    When caching to a file, the cached data will persist across runs. Even the
    first iteration through the data will read from the cache file. Changing
    the input pipeline before the call to `.cache()` will have no effect until
    the cache file is removed or the filename is changed.

    ```python
    dataset = tf.data.Dataset.range(5)
    dataset = dataset.cache("/path/to/file")
    list(dataset.as_numpy_iterator())
    # [0, 1, 2, 3, 4]
    dataset = tf.data.Dataset.range(10)
    dataset = dataset.cache("/path/to/file")  # Same file!
    list(dataset.as_numpy_iterator())
    # [0, 1, 2, 3, 4]
    ```

    Note: `cache` will produce exactly the same elements during each iteration
    through the dataset. If you wish to randomize the iteration order, make sure
    to call `shuffle` *after* calling `cache`.

    Args:
      filename: A `tf.string` scalar `tf.Tensor`, representing the name of a
        directory on the filesystem to use for caching elements in this Dataset.
        If a filename is not provided, the dataset will be cached in memory.
      name: (Optional.) A name for the tf.data operation.

    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    # Loaded lazily due to a circular dependency (dataset_ops -> cache_op ->
    # -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import cache_op
    return cache_op._cache(self, filename, name)
    # pylint: enable=g-import-not-at-top,protected-access

  def take(self, count, name=None) -> "DatasetV2":
    # [FUNCTION DOC]
    # Returns a dataset with at most count elements from the start of this dataset (minus one or a
    # count larger than the dataset gives everything). Name optional. Delegates to the take op module.
    """Creates a `Dataset` with at most `count` elements from this dataset.

    >>> dataset = tf.data.Dataset.range(10)
    >>> dataset = dataset.take(3)
    >>> [a.item() for a in dataset.as_numpy_iterator()]
    [0, 1, 2]

    Args:
      count: A `tf.int64` scalar `tf.Tensor`, representing the number of
        elements of this dataset that should be taken to form the new dataset.
        If `count` is -1, or if `count` is greater than the size of this
        dataset, the new dataset will contain all elements of this dataset.
      name: (Optional.) A name for the tf.data operation.

    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    # Loaded lazily due to a circular dependency (dataset_ops ->
    # take_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import take_op
    return take_op._take(self, count, name=name)
    # pylint: enable=g-import-not-at-top,protected-access

  def skip(self, count, name=None) -> "DatasetV2":
    # [FUNCTION DOC]
    # Returns a dataset that skips the first count elements (minus one skips everything; a count
    # larger than the dataset yields an empty dataset). Name optional. Delegates to the skip op module.
    """Creates a `Dataset` that skips `count` elements from this dataset.

    >>> dataset = tf.data.Dataset.range(10)
    >>> dataset = dataset.skip(7)
    >>> [a.item() for a in dataset.as_numpy_iterator()]
    [7, 8, 9]

    Args:
      count: A `tf.int64` scalar `tf.Tensor`, representing the number of
        elements of this dataset that should be skipped to form the new dataset.
        If `count` is greater than the size of this dataset, the new dataset
        will contain no elements.  If `count` is -1, skips the entire dataset.
      name: (Optional.) A name for the tf.data operation.

    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    # Loaded lazily due to a circular dependency (dataset_ops ->
    # skip_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import skip_op
    return skip_op._skip(self, count, name)
    # pylint: enable=g-import-not-at-top,protected-access

  def shard(self, num_shards, index, name=None) -> "DatasetV2":
    # [FUNCTION DOC]
    # Returns a dataset containing only elements whose position modulo num_shards equals index, for
    # distributed reading. Inputs: num_shards, index, name (optional). Delegates to the shard op
    # module, which can raise an invalid-argument error for illegal values (checked best effort).
    """Creates a `Dataset` that includes only 1/`num_shards` of this dataset.

    `shard` is deterministic. The Dataset produced by `A.shard(n, i)` will
    contain all elements of A whose index mod n = i.

    >>> A = tf.data.Dataset.range(10)
    >>> B = A.shard(num_shards=3, index=0)
    >>> [a.item() for a in B.as_numpy_iterator()]
    [0, 3, 6, 9]
    >>> C = A.shard(num_shards=3, index=1)
    >>> [a.item() for a in C.as_numpy_iterator()]
    [1, 4, 7]
    >>> D = A.shard(num_shards=3, index=2)
    >>> [a.item() for a in D.as_numpy_iterator()]
    [2, 5, 8]

    This dataset operator is very useful when running distributed training, as
    it allows each worker to read a unique subset.

    When reading a single input file, you can shard elements as follows:

    ```python
    d = tf.data.TFRecordDataset(input_file)
    d = d.shard(num_workers, worker_index)
    d = d.repeat(num_epochs)
    d = d.shuffle(shuffle_buffer_size)
    d = d.map(parser_fn, num_parallel_calls=num_map_threads)
    ```

    Important caveats:

    - Be sure to shard before you use any randomizing operator (such as
      shuffle).
    - Generally it is best if the shard operator is used early in the dataset
      pipeline. For example, when reading from a set of TFRecord files, shard
      before converting the dataset to input samples. This avoids reading every
      file on every worker. The following is an example of an efficient
      sharding strategy within a complete pipeline:

    ```python
    d = Dataset.list_files(pattern, shuffle=False)
    d = d.shard(num_workers, worker_index)
    d = d.repeat(num_epochs)
    d = d.shuffle(shuffle_buffer_size)
    d = d.interleave(tf.data.TFRecordDataset,
                     cycle_length=num_readers, block_length=1)
    d = d.map(parser_fn, num_parallel_calls=num_map_threads)
    ```

    Args:
      num_shards: A `tf.int64` scalar `tf.Tensor`, representing the number of
        shards operating in parallel.
      index: A `tf.int64` scalar `tf.Tensor`, representing the worker index.
      name: (Optional.) A name for the tf.data operation.

    Returns:
      A new `Dataset` with the transformation applied as described above.

    Raises:
      InvalidArgumentError: if `num_shards` or `index` are illegal values.

        Note: error checking is done on a best-effort basis, and errors aren't
        guaranteed to be caught upon dataset creation. (e.g. providing in a
        placeholder tensor bypasses the early checking, and will instead result
        in an error during a session.run call.)
    """
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import shard_op
    return shard_op._shard(self, num_shards, index, name=name)
    # pylint: enable=g-import-not-at-top,protected-access

  def save(self,
           path,
           compression=None,
           shard_func=None,
           checkpoint_args=None):
    # [FUNCTION DOC]
    # Saves the dataset contents to disk in shards and returns nothing (the save op module does the
    # work). Inputs: path (directory), compression (optional), shard_func (optional function that
    # chooses the shard of an element; default is round-robin), checkpoint_args (optional mapping
    # enabling checkpointing while saving). Side effect: writes files under the path.
    """Saves the content of the given dataset.

      Example usage:

      >>> import tempfile
      >>> path = os.path.join(tempfile.gettempdir(), "saved_data")
      >>> # Save a dataset
      >>> dataset = tf.data.Dataset.range(2)
      >>> dataset.save(path)
      >>> new_dataset = tf.data.Dataset.load(path)
      >>> for elem in new_dataset:
      ...   print(elem)
      tf.Tensor(0, shape=(), dtype=int64)
      tf.Tensor(1, shape=(), dtype=int64)

      The saved dataset is saved in multiple file "shards". By default, the
      dataset output is divided to shards in a round-robin fashion but custom
      sharding can be specified via the `shard_func` function. For example, you
      can save the dataset to using a single shard as follows:

      ```python
      dataset = make_dataset()
      def custom_shard_func(element):
        return np.int64(0)
      dataset.save(
          path="/path/to/data", ..., shard_func=custom_shard_func)
      ```

      To enable checkpointing, pass in `checkpoint_args` to the `save` method
      as follows:

      ```python
      dataset = tf.data.Dataset.range(100)
      save_dir = "..."
      checkpoint_prefix = "..."
      step_counter = tf.Variable(0, trainable=False)
      checkpoint_args = {
        "checkpoint_interval": 50,
        "step_counter": step_counter,
        "directory": checkpoint_prefix,
        "max_to_keep": 20,
      }
      dataset.save(dataset, save_dir, checkpoint_args=checkpoint_args)
      ```

      NOTE: The directory layout and file format used for saving the dataset is
      considered an implementation detail and may change. For this reason,
      datasets saved through `tf.data.Dataset.save` should only be consumed
      through `tf.data.Dataset.load`, which is guaranteed to be
      backwards compatible.

    Args:
     path: Required. A directory to use for saving the dataset.
     compression: Optional. The algorithm to use to compress data when writing
          it. Supported options are `GZIP` and `NONE`. Defaults to `NONE`.
     shard_func: Optional. A function to control the mapping of dataset
          elements to file shards. The function is expected to map elements of
          the input dataset to int64 shard IDs. If present, the function will be
          traced and executed as graph computation.
     checkpoint_args: Optional args for checkpointing which will be passed into
          the `tf.train.CheckpointManager`. If `checkpoint_args` are not
          specified, then checkpointing will not be performed. The `save()`
          implementation creates a `tf.train.Checkpoint` object internally, so
          users should not set the `checkpoint` argument in `checkpoint_args`.

    Returns:
      An operation which when executed performs the save. When writing
      checkpoints, returns None. The return value is useful in unit tests.

    Raises:
      ValueError if `checkpoint` is passed into `checkpoint_args`.
    """
    # Loaded lazily due to a circular dependency (dataset_ops -> save_op ->
    # dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import save_op
    return save_op._save(self, path, compression, shard_func, checkpoint_args)
    # pylint: enable=g-import-not-at-top,protected-access

  @staticmethod
  def load(
      path, element_spec=None, compression=None, reader_func=None, wait=False,
  ) -> "DatasetV2":
    # [FUNCTION DOC]
    # Static factory: loads a dataset previously saved to disk. Inputs: path, element_spec (optional,
    # otherwise read from saved metadata), compression (optional), reader_func (optional function
    # that controls how shards are read and combined), wait (optional flag to wait for the saved
    # data to be complete). Delegates to the lazily imported load op module. Returns the dataset.
    """Loads a previously saved dataset.

    Example usage:

    >>> import tempfile
    >>> path = os.path.join(tempfile.gettempdir(), "saved_data")
    >>> # Save a dataset
    >>> dataset = tf.data.Dataset.range(2)
    >>> tf.data.Dataset.save(dataset, path)
    >>> new_dataset = tf.data.Dataset.load(path)
    >>> for elem in new_dataset:
    ...   print(elem)
    tf.Tensor(0, shape=(), dtype=int64)
    tf.Tensor(1, shape=(), dtype=int64)


    If the default option of sharding the saved dataset was used, the element
    order of the saved dataset will be preserved when loading it.

    The `reader_func` argument can be used to specify a custom order in which
    elements should be loaded from the individual shards. The `reader_func` is
    expected to take a single argument -- a dataset of datasets, each containing
    elements of one of the shards -- and return a dataset of elements. For
    example, the order of shards can be shuffled when loading them as follows:

    ```python
    def custom_reader_func(datasets):
      datasets = datasets.shuffle(NUM_SHARDS)
      return datasets.interleave(lambda x: x, num_parallel_calls=AUTOTUNE)

    dataset = tf.data.Dataset.load(
        path="/path/to/data", ..., reader_func=custom_reader_func)
    ```

    Args:
      path: Required. A path pointing to a previously saved dataset.
      element_spec: Optional. A nested structure of `tf.TypeSpec` objects
        matching the structure of an element of the saved dataset and specifying
        the type of individual element components. If not provided, the nested
        structure of `tf.TypeSpec` saved with the saved dataset is used. Note
        that this argument is required in graph mode.
      compression: Optional. The algorithm to use to decompress the data when
        reading it. Supported options are `GZIP` and `NONE`. Defaults to `NONE`.
      reader_func: Optional. A function to control how to read data from shards.
        If present, the function will be traced and executed as graph
        computation.
      wait: If `True`, for snapshots written with `distributed_save`, it reads
        the snapshot while it is being written. For snapshots written with
        regular `save`, it waits for the snapshot until it's finished. The
        default is `False` for backward compatibility. Users of
        `distributed_save` are recommended to set it to `True`.

    Returns:
      A `tf.data.Dataset` instance.

    Raises:
      FileNotFoundError: If `element_spec` is not specified and the saved nested
        structure of `tf.TypeSpec` can not be located with the saved dataset.
      ValueError: If `element_spec` is not specified and the method is executed
        in graph mode.
    """
    # Loaded lazily due to a circular dependency (dataset_ops -> load_op ->
    # dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import load_op
    return load_op._load(
        path=path,
        element_spec=element_spec,
        compression=compression,
        reader_func=reader_func,
        wait=wait)
    # pylint: enable=g-import-not-at-top,protected-access

  def batch(
      self,
      batch_size,
      drop_remainder=False,
      num_parallel_calls=None,
      deterministic=None,
      name=None,
  ) -> "DatasetV2":
    # [FUNCTION DOC]
    # Returns a dataset that combines consecutive elements into batches. Inputs: batch_size,
    # drop_remainder (optional, default false), num_parallel_calls (optional), deterministic
    # (optional ordering control for parallel batching), name (optional). Delegates to the batch op
    # module.
    """Combines consecutive elements of this dataset into batches.

    >>> dataset = tf.data.Dataset.range(8)
    >>> dataset = dataset.batch(3)
    >>> list(dataset.as_numpy_iterator())
    [array([0, 1, 2]), array([3, 4, 5]), array([6, 7])]

    >>> dataset = tf.data.Dataset.range(8)
    >>> dataset = dataset.batch(3, drop_remainder=True)
    >>> list(dataset.as_numpy_iterator())
    [array([0, 1, 2]), array([3, 4, 5])]

    The components of the resulting element will have an additional outer
    dimension, which will be `batch_size` (or `N % batch_size` for the last
    element if `batch_size` does not divide the number of input elements `N`
    evenly and `drop_remainder` is `False`). If your program depends on the
    batches having the same outer dimension, you should set the `drop_remainder`
    argument to `True` to prevent the smaller batch from being produced.

    Note: If your program requires data to have a statically known shape (e.g.,
    when using XLA), you should use `drop_remainder=True`. Without
    `drop_remainder=True` the shape of the output dataset will have an unknown
    leading dimension due to the possibility of a smaller final batch.

    Args:
      batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        consecutive elements of this dataset to combine in a single batch.
      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
        whether the last batch should be dropped in the case it has fewer than
        `batch_size` elements; the default behavior is not to drop the smaller
        batch.
      num_parallel_calls: (Optional.) A `tf.int64` scalar `tf.Tensor`,
        representing the number of batches to compute asynchronously in
        parallel.
        If not specified, batches will be computed sequentially. If the value
        `tf.data.AUTOTUNE` is used, then the number of parallel
        calls is set dynamically based on available resources.
      deterministic: (Optional.) When `num_parallel_calls` is specified, if this
        boolean is specified (`True` or `False`), it controls the order in which
        the transformation produces elements. If set to `False`, the
        transformation is allowed to yield elements out of order to trade
        determinism for performance. If not specified, the
        `tf.data.Options.deterministic` option (`True` by default) controls the
        behavior.
      name: (Optional.) A name for the tf.data operation.

    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    # Loaded lazily due to a circular dependency (dataset_ops -> batch_op ->
    # dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access,redefined-outer-name
    from tensorflow.python.data.ops import batch_op
    return batch_op._batch(self, batch_size, drop_remainder, num_parallel_calls,
                           deterministic, name)
    # pylint: enable=g-import-not-at-top,protected-access,redefined-outer-name

  def padded_batch(
      self,
      batch_size,
      padded_shapes=None,
      padding_values=None,
      drop_remainder=False,
      name=None,
  ) -> "DatasetV2":
    # [FUNCTION DOC]
    # Returns a dataset that batches consecutive elements, padding variable-shaped components to a
    # common shape. Inputs: batch_size, padded_shapes (optional target shapes), padding_values
    # (optional pad values per component), drop_remainder (optional, default false), name
    # (optional). Delegates to the padded-batch op module.
    """Combines consecutive elements of this dataset into padded batches.

    This transformation combines multiple consecutive elements of the input
    dataset into a single element.

    Like `tf.data.Dataset.batch`, the components of the resulting element will
    have an additional outer dimension, which will be `batch_size` (or
    `N % batch_size` for the last element if `batch_size` does not divide the
    number of input elements `N` evenly and `drop_remainder` is `False`). If
    your program depends on the batches having the same outer dimension, you
    should set the `drop_remainder` argument to `True` to prevent the smaller
    batch from being produced.

    Unlike `tf.data.Dataset.batch`, the input elements to be batched may have
    different shapes, and this transformation will pad each component to the
    respective shape in `padded_shapes`. The `padded_shapes` argument
    determines the resulting shape for each dimension of each component in an
    output element:

    * If the dimension is a constant, the component will be padded out to that
      length in that dimension.
    * If the dimension is unknown, the component will be padded out to the
      maximum length of all elements in that dimension.

    >>> A = (tf.data.Dataset
    ...      .range(1, 5, output_type=tf.int32)
    ...      .map(lambda x: tf.fill([x], x)))
    >>> # Pad to the smallest per-batch size that fits all elements.
    >>> B = A.padded_batch(2)
    >>> for element in B.as_numpy_iterator():
    ...   print(element)
    [[1 0]
     [2 2]]
    [[3 3 3 0]
     [4 4 4 4]]
    >>> # Pad to a fixed size.
    >>> C = A.padded_batch(2, padded_shapes=5)
    >>> for element in C.as_numpy_iterator():
    ...   print(element)
    [[1 0 0 0 0]
     [2 2 0 0 0]]
    [[3 3 3 0 0]
     [4 4 4 4 0]]
    >>> # Pad with a custom value.
    >>> D = A.padded_batch(2, padded_shapes=5, padding_values=-1)
    >>> for element in D.as_numpy_iterator():
    ...   print(element)
    [[ 1 -1 -1 -1 -1]
     [ 2  2 -1 -1 -1]]
    [[ 3  3  3 -1 -1]
     [ 4  4  4  4 -1]]
    >>> # Components of nested elements can be padded independently.
    >>> elements = [([1, 2, 3], [10]),
    ...             ([4, 5], [11, 12])]
    >>> dataset = tf.data.Dataset.from_generator(
    ...     lambda: iter(elements), (tf.int32, tf.int32))
    >>> # Pad the first component of the tuple to length 4, and the second
    >>> # component to the smallest size that fits.
    >>> dataset = dataset.padded_batch(2,
    ...     padded_shapes=([4], [None]),
    ...     padding_values=(-1, 100))
    >>> list(dataset.as_numpy_iterator())
    [(array([[ 1,  2,  3, -1], [ 4,  5, -1, -1]], dtype=int32),
      array([[ 10, 100], [ 11,  12]], dtype=int32))]
    >>> # Pad with a single value and multiple components.
    >>> E = tf.data.Dataset.zip((A, A)).padded_batch(2, padding_values=-1)
    >>> for element in E.as_numpy_iterator():
    ...   print(element)
    (array([[ 1, -1],
           [ 2,  2]], dtype=int32), array([[ 1, -1],
           [ 2,  2]], dtype=int32))
    (array([[ 3,  3,  3, -1],
           [ 4,  4,  4,  4]], dtype=int32), array([[ 3,  3,  3, -1],
           [ 4,  4,  4,  4]], dtype=int32))

    See also `tf.data.experimental.dense_to_sparse_batch`, which combines
    elements that may have different shapes into a `tf.sparse.SparseTensor`.

    Args:
      batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        consecutive elements of this dataset to combine in a single batch.
      padded_shapes: (Optional.) A (nested) structure of `tf.TensorShape` or
        `tf.int64` vector tensor-like objects representing the shape to which
        the respective component of each input element should be padded prior
        to batching. Any unknown dimensions will be padded to the maximum size
        of that dimension in each batch. If unset, all dimensions of all
        components are padded to the maximum size in the batch. `padded_shapes`
        must be set if any component has an unknown rank.
      padding_values: (Optional.) A (nested) structure of scalar-shaped
        `tf.Tensor`, representing the padding values to use for the respective
        components. None represents that the (nested) structure should be padded
        with default values.  Defaults are `0` for numeric types and the empty
        string for string types. The `padding_values` should have the same
        (nested) structure as the input dataset. If `padding_values` is a single
        element and the input dataset has multiple components, then the same
        `padding_values` will be used to pad every component of the dataset.
        If `padding_values` is a scalar, then its value will be broadcasted
        to match the shape of each component.
      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
        whether the last batch should be dropped in the case it has fewer than
        `batch_size` elements; the default behavior is not to drop the smaller
        batch.
      name: (Optional.) A name for the tf.data operation.

    Returns:
      A new `Dataset` with the transformation applied as described above.

    Raises:
      ValueError: If a component has an unknown rank, and the `padded_shapes`
        argument is not set.
      TypeError: If a component is of an unsupported type. The list of supported
        types is documented in
        https://www.tensorflow.org/guide/data#dataset_structure.
    """
    # Loaded lazily due to a circular dependency (dataset_ops ->
    # padded_batch_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import padded_batch_op
    return padded_batch_op._padded_batch(self, batch_size, padded_shapes,
                                         padding_values, drop_remainder, name)
    # pylint: enable=g-import-not-at-top,protected-access

  def ragged_batch(
      self,
      batch_size,
      drop_remainder=False,
      row_splits_dtype=dtypes.int64,
      name=None,
  ) -> "DatasetV2":
    # [FUNCTION DOC]
    # Returns a dataset that batches consecutive elements into ragged tensors (no padding).
    # Inputs: batch_size, drop_remainder (optional, default false), row_splits_dtype (default 64-bit
    # integer), name (optional). Delegates to the ragged-batch op module.
    """Combines consecutive elements of this dataset into `tf.RaggedTensor`s.

    Like `tf.data.Dataset.batch`, the components of the resulting element will
    have an additional outer dimension, which will be `batch_size` (or
    `N % batch_size` for the last element if `batch_size` does not divide the
    number of input elements `N` evenly and `drop_remainder` is `False`). If
    your program depends on the batches having the same outer dimension, you
    should set the `drop_remainder` argument to `True` to prevent the smaller
    batch from being produced.

    Unlike `tf.data.Dataset.batch`, the input elements to be batched may have
    different shapes:

    *  If an input element is a `tf.Tensor` whose static `tf.TensorShape` is
    fully defined, then it is batched as normal.
    *  If an input element is a `tf.Tensor` whose static `tf.TensorShape`
    contains one or more axes with unknown size (i.e., `shape[i]=None`), then
    the output will contain a `tf.RaggedTensor` that is ragged up to any of such
    dimensions.
    *  If an input element is a `tf.RaggedTensor` or any other type, then it is
    batched as normal.

    Example:

    >>> dataset = tf.data.Dataset.range(6)
    >>> dataset = dataset.map(lambda x: tf.range(x))
    >>> dataset.element_spec.shape
    TensorShape([None])
    >>> dataset = dataset.ragged_batch(2)
    >>> for batch in dataset:
    ...   print(batch)
    <tf.RaggedTensor [[], [0]]>
    <tf.RaggedTensor [[0, 1], [0, 1, 2]]>
    <tf.RaggedTensor [[0, 1, 2, 3], [0, 1, 2, 3, 4]]>

    Args:
      batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        consecutive elements of this dataset to combine in a single batch.
      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
        whether the last batch should be dropped in the case it has fewer than
        `batch_size` elements; the default behavior is not to drop the smaller
        batch.
      row_splits_dtype: The dtype that should be used for the `row_splits` of
        any new ragged tensors.  Existing `tf.RaggedTensor` elements do not have
        their row_splits dtype changed.
      name: (Optional.) A string indicating a name for the `tf.data` operation.

    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    # Loaded lazily due to a circular dependency (dataset_ops ->
    # ragged_batch_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import ragged_batch_op
    return ragged_batch_op._ragged_batch(self, batch_size, drop_remainder,
                                         row_splits_dtype, name)
    # pylint: enable=g-import-not-at-top,protected-access

  def sparse_batch(self, batch_size, row_shape, name=None) -> "DatasetV2":
    # [FUNCTION DOC]
    # Returns a dataset that combines consecutive elements into sparse tensors. Inputs: batch_size,
    # row_shape (dense shape of each element), name (optional). Delegates to the sparse-batch op
    # module.
    """Combines consecutive elements into `tf.sparse.SparseTensor`s.

    Like `Dataset.padded_batch()`, this transformation combines multiple
    consecutive elements of the dataset, which might have different
    shapes, into a single element. The resulting element has three
    components (`indices`, `values`, and `dense_shape`), which
    comprise a `tf.sparse.SparseTensor` that represents the same data. The
    `row_shape` represents the dense shape of each row in the
    resulting `tf.sparse.SparseTensor`, to which the effective batch size is
    prepended. For example:

    ```python
    # NOTE: The following examples use `{ ... }` to represent the
    # contents of a dataset.
    a = { ['a', 'b', 'c'], ['a', 'b'], ['a', 'b', 'c', 'd'] }

    a.apply(tf.data.experimental.dense_to_sparse_batch(
        batch_size=2, row_shape=[6])) ==
    {
        ([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]],  # indices
         ['a', 'b', 'c', 'a', 'b'],                 # values
         [2, 6]),                                   # dense_shape
        ([[0, 0], [0, 1], [0, 2], [0, 3]],
         ['a', 'b', 'c', 'd'],
         [1, 6])
    }
    ```

    Args:
      batch_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        consecutive elements of this dataset to combine in a single batch.
      row_shape: A `tf.TensorShape` or `tf.int64` vector tensor-like object
        representing the equivalent dense shape of a row in the resulting
        `tf.sparse.SparseTensor`. Each element of this dataset must have the
        same rank as `row_shape`, and must have size less than or equal to
        `row_shape` in each dimension.
      name: (Optional.) A string indicating a name for the `tf.data` operation.

    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    # Loaded lazily due to a circular dependency (dataset_ops ->
    # sparse_batch_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import sparse_batch_op
    return sparse_batch_op._sparse_batch(self, batch_size, row_shape, name)
    # pylint: disable=g-import-not-at-top,protected-access

  def map(
      self,
      map_func,
      num_parallel_calls=None,
      deterministic=None,
      synchronous=None,
      use_unbounded_threadpool=False,
      name=None,
  ) -> "DatasetV2":
    # [FUNCTION DOC]
    # Returns a dataset applying a function to each element. Inputs: map_func (function from element
    # components to new element), num_parallel_calls (optional; autotune allowed), deterministic
    # (optional ordering control), synchronous (optional flag forcing synchronous execution),
    # use_unbounded_threadpool (optional, default false), name (optional). Delegates to the V2 map
    # implementation in the map op module.
    """Maps `map_func` across the elements of this dataset.

    This transformation applies `map_func` to each element of this dataset, and
    returns a new dataset containing the transformed elements, in the same
    order as they appeared in the input. `map_func` can be used to change both
    the values and the structure of a dataset's elements. Supported structure
    constructs are documented
    [here](https://www.tensorflow.org/guide/data#dataset_structure).

    For example, `map` can be used for adding 1 to each element, or projecting a
    subset of element components.

    >>> dataset = Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]
    >>> dataset = dataset.map(lambda x: x + 1)
    >>> [a.item() for a in dataset.as_numpy_iterator()]
    [2, 3, 4, 5, 6]

    The input signature of `map_func` is determined by the structure of each
    element in this dataset.

    >>> dataset = Dataset.range(5)
    >>> # `map_func` takes a single argument of type `tf.Tensor` with the same
    >>> # shape and dtype.
    >>> result = dataset.map(lambda x: x + 1)

    >>> # Each element is a tuple containing two `tf.Tensor` objects.
    >>> elements = [(1, "foo"), (2, "bar"), (3, "baz")]
    >>> dataset = tf.data.Dataset.from_generator(
    ...     lambda: elements, (tf.int32, tf.string))
    >>> # `map_func` takes two arguments of type `tf.Tensor`. This function
    >>> # projects out just the first component.
    >>> result = dataset.map(lambda x_int, y_str: x_int)
    >>> [a.item() for a in result.as_numpy_iterator()]
    [1, 2, 3]

    >>> # Each element is a dictionary mapping strings to `tf.Tensor` objects.
    >>> elements =  ([{"a": 1, "b": "foo"},
    ...               {"a": 2, "b": "bar"},
    ...               {"a": 3, "b": "baz"}])
    >>> dataset = tf.data.Dataset.from_generator(
    ...     lambda: elements, {"a": tf.int32, "b": tf.string})
    >>> # `map_func` takes a single argument of type `dict` with the same keys
    >>> # as the elements.
    >>> result = dataset.map(lambda d: str(d["a"]) + d["b"])

    The value or values returned by `map_func` determine the structure of each
    element in the returned dataset.

    >>> dataset = tf.data.Dataset.range(3)
    >>> # `map_func` returns two `tf.Tensor` objects.
    >>> def g(x):
    ...   return tf.constant(37.0), tf.constant(["Foo", "Bar", "Baz"])
    >>> result = dataset.map(g)
    >>> result.element_spec
    (TensorSpec(shape=(), dtype=tf.float32, name=None), TensorSpec(shape=(3,), \
dtype=tf.string, name=None))
    >>> # Python primitives, lists, and NumPy arrays are implicitly converted to
    >>> # `tf.Tensor`.
    >>> def h(x):
    ...   return 37.0, ["Foo", "Bar"], np.array([1.0, 2.0], dtype=np.float64)
    >>> result = dataset.map(h)
    >>> result.element_spec
    (TensorSpec(shape=(), dtype=tf.float32, name=None), TensorSpec(shape=(2,), \
dtype=tf.string, name=None), TensorSpec(shape=(2,), dtype=tf.float64, \
name=None))
    >>> # `map_func` can return nested structures.
    >>> def i(x):
    ...   return (37.0, [42, 16]), "foo"
    >>> result = dataset.map(i)
    >>> result.element_spec
    ((TensorSpec(shape=(), dtype=tf.float32, name=None),
      TensorSpec(shape=(2,), dtype=tf.int32, name=None)),
     TensorSpec(shape=(), dtype=tf.string, name=None))

    `map_func` can accept as arguments and return any type of dataset element.

    Note that irrespective of the context in which `map_func` is defined (eager
    vs. graph), tf.data traces the function and executes it as a graph. To use
    Python code inside of the function you have a few options:

    1) Rely on AutoGraph to convert Python code into an equivalent graph
    computation. The downside of this approach is that AutoGraph can convert
    some but not all Python code.

    2) Use `tf.py_function`, which allows you to write arbitrary Python code but
    will generally result in worse performance than 1). For example:

    >>> d = tf.data.Dataset.from_tensor_slices(['hello', 'world'])
    >>> # transform a string tensor to upper case string using a Python function
    >>> def upper_case_fn(t: tf.Tensor):
    ...   return t.numpy().decode('utf-8').upper()
    >>> d = d.map(lambda x: tf.py_function(func=upper_case_fn,
    ...           inp=[x], Tout=tf.string))
    >>> list(d.as_numpy_iterator())
    [b'HELLO', b'WORLD']

    3) Use `tf.numpy_function`, which also allows you to write arbitrary
    Python code. Note that `tf.py_function` accepts `tf.Tensor` whereas
    `tf.numpy_function` accepts numpy arrays and returns only numpy arrays.
    For example:

    >>> d = tf.data.Dataset.from_tensor_slices(['hello', 'world'])
    >>> def upper_case_fn(t: np.ndarray):
    ...   return t.decode('utf-8').upper()
    >>> d = d.map(lambda x: tf.numpy_function(func=upper_case_fn,
    ...           inp=[x], Tout=tf.string))
    >>> list(d.as_numpy_iterator())
    [b'HELLO', b'WORLD']

    Note that the use of `tf.numpy_function` and `tf.py_function`
    in general precludes the possibility of executing user-defined
    transformations in parallel (because of Python GIL).

    Performance can often be improved by setting `num_parallel_calls` so that
    `map` will use multiple threads to process elements. If deterministic order
    isn't required, it can also improve performance to set
    `deterministic=False`.

    >>> dataset = Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]
    >>> dataset = dataset.map(lambda x: x + 1,
    ...     num_parallel_calls=tf.data.AUTOTUNE,
    ...     deterministic=False)

    The order of elements yielded by this transformation is deterministic if
    `deterministic=True`. If `map_func` contains stateful operations and
    `num_parallel_calls > 1`, the order in which that state is accessed is
    undefined, so the values of output elements may not be deterministic
    regardless of the `deterministic` flag value.

    Args:
      map_func: A function mapping a dataset element to another dataset element.
      num_parallel_calls: (Optional.) A `tf.int64` scalar `tf.Tensor`,
        representing the number elements to process asynchronously in parallel.
        If the value `tf.data.AUTOTUNE` is used, then the number of parallel
        calls is set dynamically based on available CPU. If not specified, the
        `tf.data.Options.experimental_optimization.map_parallelization` option
        (`True` by default) controls whether the map will run as with
        `tf.data.AUTOTUNE` or run sequentially.
      deterministic: (Optional.) When `num_parallel_calls` is specified, if this
        boolean is specified (`True` or `False`), it controls the order in which
        the transformation produces elements. If set to `False`, the
        transformation is allowed to yield elements out of order to trade
        determinism for performance. If not specified, the
        `tf.data.Options.deterministic` option (`True` by default) controls the
        behavior.
      synchronous: (Optional.) Whether to force the map transformation to run
        synchronously. This only matters when
        `options.experimental_optimization.map_parallelization=True`. That
        option would normally change the map to run with
        `num_parallel_calls=tf.data.AUTOTUNE`, but if `synchronous=True` is
        specified, the map will not be parallelized at all. This is useful for
        saving memory, since even setting `num_parallel_calls=1` will cause one
        batch to be buffered, while with `synchronous=True` the map
        transformation doesn't buffer anything.
      use_unbounded_threadpool: (Optional.) By default, map functions run in a
        limited threadpool based on the number of cores on the machine. This
        efficient for CPU-heavy processing, but if the map function performs IO
        it is better to use an unbounded threadpool by setting it to `True`. It
        is `False` by default.
      name: (Optional.) A name for the tf.data operation.

    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    # Loaded lazily due to a circular dependency (dataset_ops -> map_op ->
    # dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import map_op

    return map_op._map_v2(
        self,
        map_func,
        num_parallel_calls=num_parallel_calls,
        deterministic=deterministic,
        synchronous=synchronous,
        use_unbounded_threadpool=use_unbounded_threadpool,
        name=name,
    )
    # pylint: enable=g-import-not-at-top,protected-access

  def flat_map(self, map_func, name=None) -> "DatasetV2":
    # [FUNCTION DOC]
    # Returns a dataset obtained by mapping a function that returns a dataset over this dataset and
    # flattening the results sequentially. Inputs: map_func, name (optional).
    """Maps `map_func` across this dataset and flattens the result.

    The type signature is:

    ```
    def flat_map(
      self: Dataset[T],
      map_func: Callable[[T], Dataset[S]]
    ) -> Dataset[S]
    ```

    Use `flat_map` if you want to make sure that the order of your dataset
    stays the same. For example, to flatten a dataset of batches into a
    dataset of their elements:

    >>> dataset = tf.data.Dataset.from_tensor_slices(
    ...     [[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> dataset = dataset.flat_map(tf.data.Dataset.from_tensor_slices)
    >>> [a.item() for a in dataset.as_numpy_iterator()]
    [1, 2, 3, 4, 5, 6, 7, 8, 9]

    `tf.data.Dataset.interleave()` is a generalization of `flat_map`, since
    `flat_map` produces the same output as
    `tf.data.Dataset.interleave(cycle_length=1)`

    Args:
      map_func: A function mapping a dataset element to a dataset.
      name: (Optional.) A name for the tf.data operation.

    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    # Loaded lazily due to a circular dependency (dataset_ops -> flat_map_op ->
    # dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import flat_map_op
    return flat_map_op._flat_map(self, map_func, name=name)
    # pylint: enable=g-import-not-at-top,protected-access

  def ignore_errors(self, log_warning=False, name=None) -> "DatasetV2":
    # [FUNCTION DOC]
    # Returns a dataset that silently drops elements whose production raises errors (such as from a
    # prior map). Inputs: log_warning (optional, default false, log a warning for each drop), name.
    """Drops elements that cause errors.

    >>> dataset = tf.data.Dataset.from_tensor_slices([1., 2., 0., 4.])
    >>> dataset = dataset.map(lambda x: tf.debugging.check_numerics(1. / x, ""))
    >>> list(dataset.as_numpy_iterator())
    Traceback (most recent call last):
    ...
    InvalidArgumentError: ... Tensor had Inf values
    >>> dataset = dataset.ignore_errors()
    >>> list(dataset.as_numpy_iterator())
    [1.0, 0.5, 0.25]

    Args:
      log_warning: (Optional.) A bool indicating whether or not ignored errors
        should be logged to stderr. Defaults to `False`.
      name: (Optional.) A string indicating a name for the `tf.data` operation.

    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    # Loaded lazily due to a circular dependency (dataset_ops ->
    # ignore_errors_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import ignore_errors_op
    return ignore_errors_op._ignore_errors(self, log_warning, name)
    # pylint: enable=g-import-not-at-top,protected-access

  def interleave(
      self,
      map_func,
      cycle_length=None,
      block_length=None,
      num_parallel_calls=None,
      deterministic=None,
      name=None,
  ) -> "DatasetV2":
    # [FUNCTION DOC]
    # Returns a dataset that maps a function returning datasets over this dataset and interleaves
    # their elements. Inputs: map_func, cycle_length and block_length (optional counts of concurrently
    # processed inputs and consecutive elements taken from each), num_parallel_calls (optional),
    # deterministic (optional), name (optional).
    """Maps `map_func` across this dataset, and interleaves the results.

    The type signature is:

    ```
    def interleave(
      self: Dataset[T],
      map_func: Callable[[T], Dataset[S]]
    ) -> Dataset[S]
    ```

    For example, you can use `Dataset.interleave()` to process many input files
    concurrently:

    >>> # Preprocess 4 files concurrently, and interleave blocks of 16 records
    >>> # from each file.
    >>> filenames = ["/var/data/file1.txt", "/var/data/file2.txt",
    ...              "/var/data/file3.txt", "/var/data/file4.txt"]
    >>> dataset = tf.data.Dataset.from_tensor_slices(filenames)
    >>> def parse_fn(filename):
    ...   return tf.data.Dataset.range(10)
    >>> dataset = dataset.interleave(lambda x:
    ...     tf.data.TextLineDataset(x).map(parse_fn, num_parallel_calls=1),
    ...     cycle_length=4, block_length=16)

    The `cycle_length` and `block_length` arguments control the order in which
    elements are produced. `cycle_length` controls the number of input elements
    that are processed concurrently. If you set `cycle_length` to 1, this
    transformation will handle one input element at a time, and will produce
    identical results to `tf.data.Dataset.flat_map`. In general,
    this transformation will apply `map_func` to `cycle_length` input elements,
    open iterators on the returned `Dataset` objects, and cycle through them
    producing `block_length` consecutive elements from each iterator, and
    consuming the next input element each time it reaches the end of an
    iterator.

    For example:

    >>> dataset = Dataset.range(1, 6)  # ==> [ 1, 2, 3, 4, 5 ]
    >>> # NOTE: New lines indicate "block" boundaries.
    >>> dataset = dataset.interleave(
    ...     lambda x: Dataset.from_tensors(x).repeat(6),
    ...     cycle_length=2, block_length=4)
    >>> [a.item() for a in dataset.as_numpy_iterator()]
    [1, 1, 1, 1,
     2, 2, 2, 2,
     1, 1,
     2, 2,
     3, 3, 3, 3,
     4, 4, 4, 4,
     3, 3,
     4, 4,
     5, 5, 5, 5,
     5, 5]

    Note: The order of elements yielded by this transformation is
    deterministic, as long as `map_func` is a pure function and
    `deterministic=True`. If `map_func` contains any stateful operations, the
    order in which that state is accessed is undefined.

    Performance can often be improved by setting `num_parallel_calls` so that
    `interleave` will use multiple threads to fetch elements. If determinism
    isn't required, it can also improve performance to set
    `deterministic=False`.

    >>> filenames = ["/var/data/file1.txt", "/var/data/file2.txt",
    ...              "/var/data/file3.txt", "/var/data/file4.txt"]
    >>> dataset = tf.data.Dataset.from_tensor_slices(filenames)
    >>> dataset = dataset.interleave(lambda x: tf.data.TFRecordDataset(x),
    ...     cycle_length=4, num_parallel_calls=tf.data.AUTOTUNE,
    ...     deterministic=False)

    Args:
      map_func: A function that takes a dataset element and returns a
        `tf.data.Dataset`.
      cycle_length: (Optional.) The number of input elements that will be
        processed concurrently. If not set, the tf.data runtime decides what it
        should be based on available CPU. If `num_parallel_calls` is set to
        `tf.data.AUTOTUNE`, the `cycle_length` argument identifies
        the maximum degree of parallelism.
      block_length: (Optional.) The number of consecutive elements to produce
        from each input element before cycling to another input element. If not
        set, defaults to 1.
      num_parallel_calls: (Optional.) If specified, the implementation creates a
        threadpool, which is used to fetch inputs from cycle elements
        asynchronously and in parallel. The default behavior is to fetch inputs
        from cycle elements synchronously with no parallelism. If the value
        `tf.data.AUTOTUNE` is used, then the number of parallel
        calls is set dynamically based on available CPU.
      deterministic: (Optional.) When `num_parallel_calls` is specified, if this
        boolean is specified (`True` or `False`), it controls the order in which
        the transformation produces elements. If set to `False`, the
        transformation is allowed to yield elements out of order to trade
        determinism for performance. If not specified, the
        `tf.data.Options.deterministic` option (`True` by default) controls the
        behavior.
      name: (Optional.) A name for the tf.data operation.

    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    # Loaded lazily due to a circular dependency (
    # dataset_ops -> interleave_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import interleave_op
    return interleave_op._interleave(self, map_func, cycle_length, block_length,
                                     num_parallel_calls, deterministic, name)
    # pylint: enable=g-import-not-at-top,protected-access

  def filter(self, predicate, name=None) -> "DatasetV2":
    # [FUNCTION DOC]
    # Returns a dataset containing only elements for which the predicate function returns true.
    # Inputs: predicate (function returning a scalar boolean), name (optional).
    """Filters this dataset according to `predicate`.

    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    >>> dataset = dataset.filter(lambda x: x < 3)
    >>> [a.item() for a in dataset.as_numpy_iterator()]
    [1, 2]
    >>> # `tf.math.equal(x, y)` is required for equality comparison
    >>> def filter_fn(x):
    ...   return tf.math.equal(x, 1)
    >>> dataset = dataset.filter(filter_fn)
    >>> [a.item() for a in dataset.as_numpy_iterator()]
    [1]

    Args:
      predicate: A function mapping a dataset element to a boolean.
      name: (Optional.) A name for the tf.data operation.

    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    # Loaded lazily due to a circular dependency (dataset_ops -> filter_op ->
    # dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import filter_op
    return filter_op._filter(self, predicate, name)
    # pylint: enable=g-import-not-at-top,protected-access

  def apply(self, transformation_func) -> "DatasetV2":
    # [FUNCTION DOC]
    # Applies a transformation function (dataset to dataset) to this dataset and returns the result.
    # Raises a type error if the function does not return a dataset. Side effect: sets the input
    # datasets attribute on the result to this dataset.
    """Applies a transformation function to this dataset.

    `apply` enables chaining of custom `Dataset` transformations, which are
    represented as functions that take one `Dataset` argument and return a
    transformed `Dataset`.

    >>> dataset = tf.data.Dataset.range(100)
    >>> def dataset_fn(ds):
    ...   return ds.filter(lambda x: x < 5)
    >>> dataset = dataset.apply(dataset_fn)
    >>> [a.item() for a in dataset.as_numpy_iterator()]
    [0, 1, 2, 3, 4]

    Args:
      transformation_func: A function that takes one `Dataset` argument and
        returns a `Dataset`.

    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    dataset = transformation_func(self)
    if not isinstance(dataset, data_types.DatasetV2):
      raise TypeError(
          f"`transformation_func` must return a `tf.data.Dataset` object. "
          f"Got {type(dataset)}.")
    dataset._input_datasets = [self]  # pylint: disable=protected-access
    return dataset

  def window(
      self, size, shift=None, stride=1, drop_remainder=False, name=None
  ) -> "DatasetV2":
    # [FUNCTION DOC]
    # Returns a dataset of windows, each itself a dataset (or structure of datasets). Inputs: size
    # (elements per window), shift (optional, defaults to size), stride (default 1), drop_remainder
    # (optional, default false), name (optional). Delegates to the window op module.
    """Returns a dataset of "windows".

    Each "window" is a dataset that contains a subset of elements of the
    input dataset. These are finite datasets of size `size` (or possibly fewer
    if there are not enough input elements to fill the window and
    `drop_remainder` evaluates to `False`).

    For example:

    >>> dataset = tf.data.Dataset.range(7).window(3)
    >>> for window in dataset:
    ...   print(window)
    <...Dataset element_spec=TensorSpec(shape=(), dtype=tf.int64, name=None)>
    <...Dataset element_spec=TensorSpec(shape=(), dtype=tf.int64, name=None)>
    <...Dataset element_spec=TensorSpec(shape=(), dtype=tf.int64, name=None)>

    Since windows are datasets, they can be iterated over:

    >>> for window in dataset:
    ...   print([a.item() for a in window.as_numpy_iterator()])
    [0, 1, 2]
    [3, 4, 5]
    [6]

    #### Shift

    The `shift` argument determines the number of input elements to shift
    between the start of each window. If windows and elements are both numbered
    starting at 0, the first element in window `k` will be element `k * shift`
    of the input dataset. In particular, the first element of the first window
    will always be the first element of the input dataset.

    >>> dataset = tf.data.Dataset.range(7).window(3, shift=1,
    ...                                           drop_remainder=True)
    >>> for window in dataset:
    ...   print([a.item() for a in window.as_numpy_iterator()])
    [0, 1, 2]
    [1, 2, 3]
    [2, 3, 4]
    [3, 4, 5]
    [4, 5, 6]

    #### Stride

    The `stride` argument determines the stride between input elements within a
    window.

    >>> dataset = tf.data.Dataset.range(7).window(3, shift=1, stride=2,
    ...                                           drop_remainder=True)
    >>> for window in dataset:
    ...   print([a.item() for a in window.as_numpy_iterator()])
    [0, 2, 4]
    [1, 3, 5]
    [2, 4, 6]

    #### Nested elements

    When the `window` transformation is applied to a dataset whose elements are
    nested structures, it produces a dataset where the elements have the same
    nested structure but each leaf is replaced by a window. In other words,
    the nesting is applied outside of the windows as opposed inside of them.

    The type signature is:

    ```
    def window(
        self: Dataset[Nest[T]], ...
    ) -> Dataset[Nest[Dataset[T]]]
    ```

    Applying `window` to a `Dataset` of tuples gives a tuple of windows:

    >>> dataset = tf.data.Dataset.from_tensor_slices(([1, 2, 3, 4, 5],
    ...                                               [6, 7, 8, 9, 10]))
    >>> dataset = dataset.window(2)
    >>> windows = next(iter(dataset))
    >>> windows
    (<...Dataset element_spec=TensorSpec(shape=(), dtype=tf.int32, name=None)>,
     <...Dataset element_spec=TensorSpec(shape=(), dtype=tf.int32, name=None)>)

    >>> def to_numpy(ds):
    ...   return [a.item() for a in ds.as_numpy_iterator()]
    >>>
    >>> for windows in dataset:
    ...   print(to_numpy(windows[0]), to_numpy(windows[1]))
    [1, 2] [6, 7]
    [3, 4] [8, 9]
    [5] [10]

    Applying `window` to a `Dataset` of dictionaries gives a dictionary of
    `Datasets`:

    >>> dataset = tf.data.Dataset.from_tensor_slices({'a': [1, 2, 3],
    ...                                               'b': [4, 5, 6],
    ...                                               'c': [7, 8, 9]})
    >>> dataset = dataset.window(2)
    >>> def to_numpy(ds):
    ...   return [a.item() for a in ds.as_numpy_iterator()]
    >>>
    >>> for windows in dataset:
    ...   print(tf.nest.map_structure(to_numpy, windows))
    {'a': [1, 2], 'b': [4, 5], 'c': [7, 8]}
    {'a': [3], 'b': [6], 'c': [9]}

    #### Flatten a dataset of windows

    The `Dataset.flat_map` and `Dataset.interleave` methods can be used to
    flatten a dataset of windows into a single dataset.

    The argument to `flat_map` is a function that takes an element from the
    dataset and returns a `Dataset`. `flat_map` chains together the resulting
    datasets sequentially.

    For example, to turn each window into a dense tensor:

    >>> dataset = tf.data.Dataset.range(7).window(3, shift=1,
    ...                                           drop_remainder=True)
    >>> batched = dataset.flat_map(lambda x:x.batch(3))
    >>> for batch in batched:
    ...   print(batch.numpy())
    [0 1 2]
    [1 2 3]
    [2 3 4]
    [3 4 5]
    [4 5 6]

    Args:
      size: A `tf.int64` scalar `tf.Tensor`, representing the number of elements
        of the input dataset to combine into a window. Must be positive.
      shift: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
        number of input elements by which the window moves in each iteration.
        Defaults to `size`. Must be positive.
      stride: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the
        stride of the input elements in the sliding window. Must be positive.
        The default value of 1 means "retain every input element".
      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
        whether the last windows should be dropped if their size is smaller than
        `size`.
      name: (Optional.) A name for the tf.data operation.

    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    # Loaded lazily due to a circular dependency (dataset_ops -> window_op ->
    # dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import window_op
    return window_op._window(self, size, shift, stride, drop_remainder, name)
    # pylint: enable=g-import-not-at-top,protected-access

  def reduce(self, initial_state, reduce_func, name=None):
    # [FUNCTION DOC]
    # Reduces the dataset to a single element by folding with a function. Inputs: initial_state
    # (nested structure), reduce_func (function of state and element returning new state), name
    # (optional). Traces reduce_func repeatedly until the state shapes reach a fixed point (weakening
    # shapes to the most specific compatible shape each pass), validating that the returned classes
    # and types match the initial state; raises a type error on a mismatch. Adds the function to the
    # default graph, runs the reduce dataset op on the (debug-adjusted) dataset and returns the final
    # state with the original structure.
    """Reduces the input dataset to a single element.

    The transformation calls `reduce_func` successively on every element of
    the input dataset until the dataset is exhausted, aggregating information in
    its internal state. The `initial_state` argument is used for the initial
    state and the final state is returned as the result.

    >>> tf.data.Dataset.range(5).reduce(np.int64(0), lambda x, _: x +
    ...   1).numpy().item()
    5
    >>> tf.data.Dataset.range(5).reduce(np.int64(0), lambda x, y: x +
    ...   y).numpy().item()
    10

    Args:
      initial_state: An element representing the initial state of the
        transformation.
      reduce_func: A function that maps `(old_state, input_element)` to
        `new_state`. It must take two arguments and return a new element The
        structure of `new_state` must match the structure of `initial_state`.
      name: (Optional.) A name for the tf.data operation.

    Returns:
      A dataset element corresponding to the final state of the transformation.
    """

    with ops.name_scope("initial_state"):
      initial_state = structure.normalize_element(initial_state)
    state_structure = structure.type_spec_from_value(initial_state)

    # Iteratively rerun the reduce function until reaching a fixed point on
    # `state_structure`.
    need_to_rerun = True
    while need_to_rerun:

      wrapped_func = structured_function.StructuredFunctionWrapper(
          reduce_func,
          "reduce()",
          input_structure=(state_structure, self.element_spec),
          add_to_graph=False)

      # Extract and validate class information from the returned values.
      output_classes = wrapped_func.output_classes
      state_classes = nest.map_structure(
          lambda component_spec: component_spec._to_legacy_output_classes(),  # pylint: disable=protected-access
          state_structure)
      for new_state_class, state_class in zip(
          nest.flatten(output_classes), nest.flatten(state_classes)):
        if not issubclass(new_state_class, state_class):
          raise TypeError(
              f"The element classes for the new state must match the initial "
              f"state. Expected {state_classes} but got "
              f"{wrapped_func.output_classes}.")

      # Extract and validate type information from the returned values.
      output_types = wrapped_func.output_types
      state_types = nest.map_structure(
          lambda component_spec: component_spec._to_legacy_output_types(),  # pylint: disable=protected-access
          state_structure)
      for new_state_type, state_type in zip(
          nest.flatten(output_types), nest.flatten(state_types)):
        if new_state_type != state_type:
          raise TypeError(
              f"The element types for the new state must match the initial "
              f"state. Expected {state_types} but got "
              f"{wrapped_func.output_types}.")

      # Extract shape information from the returned values.
      output_shapes = wrapped_func.output_shapes
      state_shapes = nest.map_structure(
          lambda component_spec: component_spec._to_legacy_output_shapes(),  # pylint: disable=protected-access
          state_structure)
      flat_state_shapes = nest.flatten(state_shapes)
      flat_new_state_shapes = nest.flatten(output_shapes)
      weakened_state_shapes = [
          original.most_specific_compatible_shape(new)
          for original, new in zip(flat_state_shapes, flat_new_state_shapes)
      ]

      need_to_rerun = False
      for original_shape, weakened_shape in zip(flat_state_shapes,
                                                weakened_state_shapes):
        if original_shape.ndims is not None and (
            weakened_shape.ndims is None or
            original_shape.as_list() != weakened_shape.as_list()):
          need_to_rerun = True
          break

      if need_to_rerun:
        # TODO(b/110122868): Support a "most specific compatible structure"
        # method for combining structures, to avoid using legacy structures
        # here.
        state_structure = structure.convert_legacy_structure(
            state_types,
            nest.pack_sequence_as(state_shapes, weakened_state_shapes),
            state_classes)

    reduce_func = wrapped_func.function
    reduce_func.add_to_graph(ops.get_default_graph())

    dataset = self._apply_debug_options()

    # pylint: disable=protected-access
    metadata = dataset_metadata_pb2.Metadata()
    if name:
      metadata.name = _validate_and_encode(name)
    return structure.from_compatible_tensor_list(
        state_structure,
        gen_dataset_ops.reduce_dataset(
            dataset._variant_tensor,
            structure.to_tensor_list(state_structure, initial_state),
            reduce_func.captured_inputs,
            f=reduce_func,
            output_shapes=structure.get_flat_tensor_shapes(state_structure),
            output_types=structure.get_flat_tensor_types(state_structure),
            metadata=metadata.SerializeToString()))

  def get_single_element(self, name=None):
    # [FUNCTION DOC]
    # Returns the single element of a dataset that must contain exactly one element, running the
    # dataset-to-single-element op (the op errors if the dataset is empty or has more elements).
    # Input: name (optional). Returns the element in the structure of the element spec.
    """Returns the single element of the `dataset`.

    The function enables you to use a `tf.data.Dataset` in a stateless
    "tensor-in tensor-out" expression, without creating an iterator.
    This facilitates the ease of data transformation on tensors using the
    optimized `tf.data.Dataset` abstraction on top of them.

    For example, lets consider a `preprocessing_fn` which would take as an
    input the raw features and returns the processed feature along with
    it's label.

    ```python
    def preprocessing_fn(raw_feature):
      # ... the raw_feature is preprocessed as per the use-case
      return feature

    raw_features = ...  # input batch of BATCH_SIZE elements.
    dataset = (tf.data.Dataset.from_tensor_slices(raw_features)
              .map(preprocessing_fn, num_parallel_calls=BATCH_SIZE)
              .batch(BATCH_SIZE))

    processed_features = dataset.get_single_element()
    ```

    In the above example, the `raw_features` tensor of length=BATCH_SIZE
    was converted to a `tf.data.Dataset`. Next, each of the `raw_feature` was
    mapped using the `preprocessing_fn` and the processed features were
    grouped into a single batch. The final `dataset` contains only one element
    which is a batch of all the processed features.

    NOTE: The `dataset` should contain only one element.

    Now, instead of creating an iterator for the `dataset` and retrieving the
    batch of features, the `tf.data.get_single_element()` function is used
    to skip the iterator creation process and directly output the batch of
    features.

    This can be particularly useful when your tensor transformations are
    expressed as `tf.data.Dataset` operations, and you want to use those
    transformations while serving your model.

    #### Keras

    ```python

    model = ... # A pre-built or custom model

    class PreprocessingModel(tf.keras.Model):
      def __init__(self, model):
        super().__init__(self)
        self.model = model

      @tf.function(input_signature=[...])
      def serving_fn(self, data):
        ds = tf.data.Dataset.from_tensor_slices(data)
        ds = ds.map(preprocessing_fn, num_parallel_calls=BATCH_SIZE)
        ds = ds.batch(batch_size=BATCH_SIZE)
        return tf.argmax(self.model(ds.get_single_element()), axis=-1)

    preprocessing_model = PreprocessingModel(model)
    your_exported_model_dir = ... # save the model to this path.
    tf.saved_model.save(preprocessing_model, your_exported_model_dir,
                  signatures={'serving_default': preprocessing_model.serving_fn}
                  )
    ```

    Args:
      name: (Optional.) A name for the tf.data operation.

    Returns:
      A nested structure of `tf.Tensor` objects, corresponding to the single
      element of `dataset`.

    Raises:
      InvalidArgumentError: (at runtime) if `dataset` does not contain exactly
        one element.
    """

    metadata = dataset_metadata_pb2.Metadata()
    if name:
      metadata.name = _validate_and_encode(name)
    return structure.from_compatible_tensor_list(
        self.element_spec,
        gen_dataset_ops.dataset_to_single_element(
            self._variant_tensor,
            metadata=metadata.SerializeToString(),
            **self._flat_structure))  # pylint: disable=protected-access

  def unbatch(self, name=None) -> "DatasetV2":
    # [FUNCTION DOC]
    # Returns a dataset that splits each element along its first dimension into multiple elements.
    # Name optional. Delegates to the unbatch op module.
    """Splits elements of a dataset into multiple elements.

    For example, if elements of the dataset are shaped `[B, a0, a1, ...]`,
    where `B` may vary for each input element, then for each element in the
    dataset, the unbatched dataset will contain `B` consecutive elements
    of shape `[a0, a1, ...]`.

    >>> elements = [ [1, 2, 3], [1, 2], [1, 2, 3, 4] ]
    >>> dataset = tf.data.Dataset.from_generator(lambda: elements, tf.int64)
    >>> dataset = dataset.unbatch()
    >>> [a.item() for a in dataset.as_numpy_iterator()]
    [1, 2, 3, 1, 2, 1, 2, 3, 4]

    Note: `unbatch` requires a data copy to slice up the batched tensor into
    smaller, unbatched tensors. When optimizing performance, try to avoid
    unnecessary usage of `unbatch`.

    Args:
      name: (Optional.) A name for the tf.data operation.

    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    # Loaded lazily due to a circular dependency (
    # dataset_ops -> unbatch_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import unbatch_op
    return unbatch_op._unbatch(self, name=name)
    # pylint: enable=g-import-not-at-top,protected-access

  def with_options(self, options, name=None) -> "DatasetV2":
    # [FUNCTION DOC]
    # Returns a new dataset with the given options merged in, as an identity options dataset.
    # Inputs: options (options object), name (optional). The original dataset is unchanged.
    """Returns a new `tf.data.Dataset` with the given options set.

    The options are "global" in the sense they apply to the entire dataset.
    If options are set multiple times, they are merged as long as different
    options do not use different non-default values.

    >>> ds = tf.data.Dataset.range(5)
    >>> ds = ds.interleave(lambda x: tf.data.Dataset.range(5),
    ...                    cycle_length=3,
    ...                    num_parallel_calls=3)
    >>> options = tf.data.Options()
    >>> # This will make the interleave order non-deterministic.
    >>> options.deterministic = False
    >>> ds = ds.with_options(options)

    Args:
      options: A `tf.data.Options` that identifies the options the use.
      name: (Optional.) A name for the tf.data operation.

    Returns:
      A new `Dataset` with the transformation applied as described above.

    Raises:
      ValueError: when an option is set more than once to a non-default value
    """
    return _OptionsDataset(self, options, name=name)

  def cardinality(self):
    # [FUNCTION DOC]
    # Returns a scalar tensor with the number of elements if known, or special constants for infinite
    # and unknown cardinality.
    """Returns the cardinality of the dataset, if known.

    `cardinality` may return `tf.data.INFINITE_CARDINALITY` if the dataset
    contains an infinite number of elements or `tf.data.UNKNOWN_CARDINALITY` if
    the analysis fails to determine the number of elements in the dataset
    (e.g. when the dataset source is a file).

    >>> dataset = tf.data.Dataset.range(42)
    >>> print(dataset.cardinality().numpy())
    42
    >>> dataset = dataset.repeat()
    >>> cardinality = dataset.cardinality()
    >>> print((cardinality == tf.data.INFINITE_CARDINALITY).numpy())
    True
    >>> dataset = dataset.filter(lambda x: True)
    >>> cardinality = dataset.cardinality()
    >>> print((cardinality == tf.data.UNKNOWN_CARDINALITY).numpy())
    True

    Returns:
      A scalar `tf.int64` `Tensor` representing the cardinality of the dataset.
      If the cardinality is infinite or unknown, `cardinality` returns the
      named constants `tf.data.INFINITE_CARDINALITY` and
      `tf.data.UNKNOWN_CARDINALITY` respectively.
    """
    return gen_dataset_ops.dataset_cardinality(self._variant_tensor)

  def group_by_window(
      self,
      key_func,
      reduce_func,
      window_size=None,
      window_size_func=None,
      name=None,
  ) -> "DatasetV2":
    # [FUNCTION DOC]
    # Returns a dataset that groups elements by key and reduces each window of a key. Inputs:
    # key_func (element to 64-bit key), reduce_func (key and window dataset to dataset), window_size
    # or window_size_func (fixed size, or function of key; give one), name (optional).
    """Groups windows of elements by key and reduces them.

    This transformation maps each consecutive element in a dataset to a key
    using `key_func` and groups the elements by key. It then applies
    `reduce_func` to at most `window_size_func(key)` elements matching the same
    key. All except the final window for each key will contain
    `window_size_func(key)` elements; the final window may be smaller.

    You may provide either a constant `window_size` or a window size determined
    by the key through `window_size_func`.

    >>> dataset = tf.data.Dataset.range(10)
    >>> window_size = 5
    >>> key_func = lambda x: x%2
    >>> reduce_func = lambda key, dataset: dataset.batch(window_size)
    >>> dataset = dataset.group_by_window(
    ...           key_func=key_func,
    ...           reduce_func=reduce_func,
    ...           window_size=window_size)
    >>> for elem in dataset.as_numpy_iterator():
    ...   print(elem)
    [0 2 4 6 8]
    [1 3 5 7 9]

    Args:
      key_func: A function mapping a nested structure of tensors (having shapes
        and types defined by `self.output_shapes` and `self.output_types`) to a
        scalar `tf.int64` tensor.
      reduce_func: A function mapping a key and a dataset of up to `window_size`
        consecutive elements matching that key to another dataset.
      window_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
        consecutive elements matching the same key to combine in a single batch,
        which will be passed to `reduce_func`. Mutually exclusive with
        `window_size_func`.
      window_size_func: A function mapping a key to a `tf.int64` scalar
        `tf.Tensor`, representing the number of consecutive elements matching
        the same key to combine in a single batch, which will be passed to
        `reduce_func`. Mutually exclusive with `window_size`.
      name: (Optional.) A name for the tf.data operation.

    Returns:
      A new `Dataset` with the transformation applied as described above.

    Raises:
      ValueError: if neither or both of {`window_size`, `window_size_func`} are
        passed.
    """
    # Loaded lazily due to a circular dependency (
    # dataset_ops -> group_by_window_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import group_by_window_op
    return group_by_window_op._group_by_window(
        self, key_func, reduce_func, window_size, window_size_func, name=name)
    # pylint: enable=g-import-not-at-top,protected-access

  def bucket_by_sequence_length(
      self,
      element_length_func,
      bucket_boundaries,
      bucket_batch_sizes,
      padded_shapes=None,
      padding_values=None,
      pad_to_bucket_boundary=False,
      no_padding=False,
      drop_remainder=False,
      name=None,
  ) -> "DatasetV2":
    # [FUNCTION DOC]
    # Returns a dataset that buckets elements by length and batches each bucket. Inputs:
    # element_length_func (element to length), bucket_boundaries (increasing length cutoffs),
    # bucket_batch_sizes (batch size per bucket; must be one more than the number of boundaries, else
    # a value error is raised), padded_shapes and padding_values (optional), pad_to_bucket_boundary
    # (optional, pad to boundary minus one), no_padding (optional, plain batching), drop_remainder,
    # name. Implemented with grouping by a bucket id key and a window size from the bucket batch
    # sizes, then (padded) batching each window.
    """A transformation that buckets elements in a `Dataset` by length.

    Elements of the `Dataset` are grouped together by length and then are padded
    and batched.

    This is useful for sequence tasks in which the elements have variable
    length. Grouping together elements that have similar lengths reduces the
    total fraction of padding in a batch which increases training step
    efficiency.

    Below is an example to bucketize the input data to the 3 buckets
    "[0, 3), [3, 5), [5, inf)" based on sequence length, with batch size 2.

    >>> elements = [
    ...   [0], [1, 2, 3, 4], [5, 6, 7],
    ...   [7, 8, 9, 10, 11], [13, 14, 15, 16, 19, 20], [21, 22]]
    >>> dataset = tf.data.Dataset.from_generator(
    ...     lambda: elements, tf.int64, output_shapes=[None])
    >>> dataset = dataset.bucket_by_sequence_length(
    ...         element_length_func=lambda elem: tf.shape(elem)[0],
    ...         bucket_boundaries=[3, 5],
    ...         bucket_batch_sizes=[2, 2, 2])
    >>> for elem in dataset.as_numpy_iterator():
    ...   print(elem)
    [[1 2 3 4]
    [5 6 7 0]]
    [[ 7  8  9 10 11  0]
    [13 14 15 16 19 20]]
    [[ 0  0]
    [21 22]]

    Args:
      element_length_func: function from element in `Dataset` to `tf.int32`,
        determines the length of the element, which will determine the bucket it
        goes into.
      bucket_boundaries: `list<int>`, upper length boundaries of the buckets.
      bucket_batch_sizes: `list<int>`, batch size per bucket. Length should be
        `len(bucket_boundaries) + 1`.
      padded_shapes: Nested structure of `tf.TensorShape` to pass to
        `tf.data.Dataset.padded_batch`. If not provided, will use
        `dataset.output_shapes`, which will result in variable length dimensions
        being padded out to the maximum length in each batch.
      padding_values: Values to pad with, passed to
        `tf.data.Dataset.padded_batch`. Defaults to padding with 0.
      pad_to_bucket_boundary: bool, if `False`, will pad dimensions with unknown
        size to maximum length in batch. If `True`, will pad dimensions with
        unknown size to bucket boundary minus 1 (i.e., the maximum length in
        each bucket), and caller must ensure that the source `Dataset` does not
        contain any elements with length longer than `max(bucket_boundaries)`.
      no_padding: `bool`, indicates whether to pad the batch features (features
        need to be either of type `tf.sparse.SparseTensor` or of same shape).
      drop_remainder: (Optional.) A `tf.bool` scalar `tf.Tensor`, representing
        whether the last batch should be dropped in the case it has fewer than
        `batch_size` elements; the default behavior is not to drop the smaller
        batch.
      name: (Optional.) A name for the tf.data operation.

    Returns:
      A new `Dataset` with the transformation applied as described above.

    Raises:
      ValueError: if `len(bucket_batch_sizes) != len(bucket_boundaries) + 1`.
    """
    if len(bucket_batch_sizes) != (len(bucket_boundaries) + 1):
      raise ValueError(
          f"`len(bucket_batch_sizes)` must equal `len(bucket_boundaries) + 1` "
          f"but `len(bucket_batch_sizes)={len(bucket_batch_sizes)}` and "
          f"`len(bucket_boundaries)={len(bucket_boundaries)}`.")

    batch_sizes = constant_op.constant(bucket_batch_sizes, dtype=dtypes.int64)

    def element_to_bucket_id(*args):
      # [FUNCTION DOC]
      # HELPER: key function for bucketing. Computes the element length and returns the 64-bit index
      # of the bucket whose boundaries contain that length (lower bound inclusive, upper exclusive).
      """Return int64 id of the length bucket for this element."""
      seq_length = element_length_func(*args)

      boundaries = list(bucket_boundaries)
      buckets_min = [np.iinfo(np.int32).min] + boundaries
      buckets_max = boundaries + [np.iinfo(np.int32).max]
      conditions_c = math_ops.logical_and(
          math_ops.less_equal(buckets_min, seq_length),
          math_ops.less(seq_length, buckets_max))
      bucket_id = math_ops.reduce_min(array_ops.where(conditions_c))

      return bucket_id

    def window_size_fn(bucket_id):
      # [FUNCTION DOC]
      # HELPER: returns the batch size configured for the given bucket id, used as the window size.
      # The window size is set to the batch size for this bucket
      window_size = batch_sizes[bucket_id]
      return window_size

    def make_padded_shapes(shapes, none_filler=None):
      # [FUNCTION DOC]
      # HELPER: converts a structure of shapes into a structure of lists where unknown dimensions are
      # replaced by the filler value (nothing by default); other dimensions are kept.
      padded = []
      for shape in nest.flatten(shapes):
        shape = tensor_shape.TensorShape(shape)
        shape = [
            none_filler if tensor_shape.dimension_value(d) is None else d
            for d in shape
        ]
        padded.append(shape)
      return nest.pack_sequence_as(shapes, padded)

    def batching_fn(bucket_id, grouped_dataset):
      # [FUNCTION DOC]
      # HELPER: reduce function that batches one bucket's window dataset. With no padding it uses
      # plain batching. Otherwise, if padding to the bucket boundary, asserts the bucket is not the last
      # one and uses the boundary minus one as the filler for unknown dimensions; then it uses padded
      # batching with the given or inferred shapes and padding values.
      """Batch elements in dataset."""
      batch_size = window_size_fn(bucket_id)
      if no_padding:
        return grouped_dataset.batch(
            batch_size, drop_remainder=drop_remainder, name=name)
      none_filler = None
      if pad_to_bucket_boundary:
        err_msg = ("When pad_to_bucket_boundary=True, elements must have "
                   "length < max(bucket_boundaries).")
        check = check_ops.assert_less(
            bucket_id,
            constant_op.constant(
                len(bucket_batch_sizes) - 1, dtype=dtypes.int64),
            message=err_msg)
        with ops.control_dependencies([check]):
          boundaries = constant_op.constant(
              bucket_boundaries, dtype=dtypes.int64)
          bucket_boundary = boundaries[bucket_id]
          none_filler = bucket_boundary - 1
      input_shapes = get_legacy_output_shapes(grouped_dataset)
      shapes = make_padded_shapes(
          padded_shapes or input_shapes, none_filler=none_filler)
      return grouped_dataset.padded_batch(
          batch_size,
          shapes,
          padding_values,
          drop_remainder=drop_remainder,
          name=name)

    return self.group_by_window(
        key_func=element_to_bucket_id,
        reduce_func=batching_fn,
        window_size_func=window_size_fn,
        name=name)

  @staticmethod
  def random(
      seed=None, rerandomize_each_iteration=None, name=None
  ) -> "DatasetV2":
    # [FUNCTION DOC]
    # Static factory: creates an unbounded dataset of pseudorandom 64-bit integers. Inputs: seed
    # (optional), rerandomize_each_iteration (optional flag to change the stream each epoch), name.
    # Delegates to the random op module.
    """Creates a `Dataset` of pseudorandom values.

    The dataset generates a sequence of uniformly distributed integer values.

    `rerandomize_each_iteration` controls whether the sequence of random number
    generated should be re-randomized for each epoch. The default value is False
    where the dataset generates the same sequence of random numbers for each
    epoch.

    >>> ds1 = tf.data.Dataset.random(seed=4).take(10)
    >>> ds2 = tf.data.Dataset.random(seed=4).take(10)
    >>> print(list(ds1.as_numpy_iterator())==list(ds2.as_numpy_iterator()))
    True

    >>> ds3 = tf.data.Dataset.random(seed=4).take(10)
    >>> ds3_first_epoch = list(ds3.as_numpy_iterator())
    >>> ds3_second_epoch = list(ds3.as_numpy_iterator())
    >>> print(ds3_first_epoch == ds3_second_epoch)
    True

    >>> ds4 = tf.data.Dataset.random(
    ...     seed=4, rerandomize_each_iteration=True).take(10)
    >>> ds4_first_epoch = list(ds4.as_numpy_iterator())
    >>> ds4_second_epoch = list(ds4.as_numpy_iterator())
    >>> print(ds4_first_epoch == ds4_second_epoch)
    False

    Args:
      seed: (Optional) If specified, the dataset produces a deterministic
        sequence of values.
      rerandomize_each_iteration: (Optional) If set to False, the dataset
      generates the same sequence of random numbers for each epoch. If set to
      True, it generates a different deterministic sequence of random numbers
      for each epoch. It is defaulted to False if left unspecified.
      name: (Optional.) A name for the tf.data operation.

    Returns:
      Dataset: A `Dataset`.
    """
    # Loaded lazily due to a circular dependency (
    # dataset_ops -> random_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import random_op
    return random_op._random(
        seed=seed,
        rerandomize_each_iteration=rerandomize_each_iteration,
        name=name)
    # pylint: enable=g-import-not-at-top,protected-access

  def snapshot(
      self,
      path,
      compression="AUTO",
      reader_func=None,
      shard_func=None,
      name=None,
  ) -> "DatasetV2":
    # [FUNCTION DOC]
    # Persists the output of this dataset to disk and reads it back on later runs. Inputs: path,
    # compression (default automatic), reader_func and shard_func (optional functions), name.
    # Delegates to the snapshot op module.
    """API to persist the output of the input dataset.

    The snapshot API allows users to transparently persist the output of their
    preprocessing pipeline to disk, and materialize the pre-processed data on a
    different training run.

    This API enables repeated preprocessing steps to be consolidated, and allows
    re-use of already processed data, trading off disk storage and network
    bandwidth for freeing up more valuable CPU resources and accelerator compute
    time.

    https://github.com/tensorflow/community/blob/master/rfcs/20200107-tf-data-snapshot.md
    has detailed design documentation of this feature.

    Users can specify various options to control the behavior of snapshot,
    including how snapshots are read from and written to by passing in
    user-defined functions to the `reader_func` and `shard_func` parameters.

    `shard_func` is a user specified function that maps input elements to
    snapshot shards.

    Users may want to specify this function to control how snapshot files should
    be written to disk. Below is an example of how a potential `shard_func`
    could be written.

    ```python
    dataset = ...
    dataset = dataset.enumerate()
    dataset = dataset.snapshot("/path/to/snapshot/dir",
        shard_func=lambda x, y: x % NUM_SHARDS, ...)
    dataset = dataset.map(lambda x, y: y)
    ```

    `reader_func` is a user specified function that accepts a single argument:
    (1) a Dataset of Datasets, each representing a "split" of elements of the
    original dataset. The cardinality of the input dataset matches the
    number of the shards specified in the `shard_func` (see above). The function
    should return a Dataset of elements of the original dataset.

    Users may want specify this function to control how snapshot files should be
    read from disk, including the amount of shuffling and parallelism.

    Here is an example of a standard reader function a user can define. This
    function enables both dataset shuffling and parallel reading of datasets:

    ```python
    def user_reader_func(datasets):
      # shuffle the datasets splits
      datasets = datasets.shuffle(NUM_CORES)
      # read datasets in parallel and interleave their elements
      return datasets.interleave(lambda x: x, num_parallel_calls=AUTOTUNE)

    dataset = dataset.snapshot("/path/to/snapshot/dir",
        reader_func=user_reader_func)
    ```

    By default, snapshot parallelizes reads by the number of cores available on
    the system, but will not attempt to shuffle the data.

    Args:
      path: Required. A directory to use for storing / loading the snapshot to /
        from.
      compression: Optional. The type of compression to apply to the snapshot
        written to disk. Supported options are `GZIP`, `SNAPPY`, `AUTO` or None.
        Defaults to `AUTO`, which attempts to pick an appropriate compression
        algorithm for the dataset.
      reader_func: Optional. A function to control how to read data from
        snapshot shards.
      shard_func: Optional. A function to control how to shard data when writing
        a snapshot.
      name: (Optional.) A name for the tf.data operation.

    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    # Loaded lazily due to a circular dependency (
    # dataset_ops -> snapshot_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import snapshot_op
    return snapshot_op._snapshot(
        self, path, compression, reader_func, shard_func, name=name)
    # pylint: enable=g-import-not-at-top,protected-access

  def scan(self, initial_state, scan_func, name=None) -> "DatasetV2":
    # [FUNCTION DOC]
    # Returns a dataset that applies a stateful function across elements, like a reduce that emits
    # each intermediate result. Inputs: initial_state, scan_func (function of state and element
    # returning new state and output), name (optional).
    """A transformation that scans a function across an input dataset.

    This transformation is a stateful relative of `tf.data.Dataset.map`.
    In addition to mapping `scan_func` across the elements of the input dataset,
    `scan()` accumulates one or more state tensors, whose initial values are
    `initial_state`.

    >>> dataset = tf.data.Dataset.range(10)
    >>> initial_state = tf.constant(0, dtype=tf.int64)
    >>> scan_func = lambda state, i: (state + i, state + i)
    >>> dataset = dataset.scan(initial_state=initial_state, scan_func=scan_func)
    >>> [a.item() for a in dataset.as_numpy_iterator()]
    [0, 1, 3, 6, 10, 15, 21, 28, 36, 45]

    Args:
      initial_state: A nested structure of tensors, representing the initial
        state of the accumulator.
      scan_func: A function that maps `(old_state, input_element)` to
        `(new_state, output_element)`. It must take two arguments and return a
        pair of nested structures of tensors. The `new_state` must match the
        structure of `initial_state`.
      name: (Optional.) A name for the tf.data operation.

    Returns:
      A new `Dataset` with the transformation applied as described above.
    """

    # Loaded lazily due to a circular dependency (dataset_ops ->
    # scan_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import scan_op
    return scan_op._scan(self, initial_state, scan_func, name=name)
    # pylint: enable=g-import-not-at-top,protected-access

  def take_while(self, predicate, name=None) -> "DatasetV2":
    # [FUNCTION DOC]
    # Returns a dataset that yields elements until the predicate function first returns false.
    # Inputs: predicate, name (optional).
    """A transformation that stops dataset iteration based on a `predicate`.

    >>> dataset = tf.data.Dataset.range(10)
    >>> dataset = dataset.take_while(lambda x: x < 5)
    >>> [a.item() for a in dataset.as_numpy_iterator()]
    [0, 1, 2, 3, 4]

    Args:
      predicate: A function that maps a nested structure of tensors (having
        shapes and types defined by `self.output_shapes` and
        `self.output_types`) to a scalar `tf.bool` tensor.
      name: (Optional.) A name for the tf.data operation.

    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    # Loaded lazily due to a circular dependency (
    # dataset_ops -> take_while_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import take_while_op
    return take_while_op._take_while(self, predicate, name=name)
    # pylint: enable=g-import-not-at-top,protected-access

  def unique(self, name=None) -> "DatasetV2":
    # [FUNCTION DOC]
    # Returns a dataset with duplicate elements removed (supports int and string components).
    # Name optional.
    """A transformation that discards duplicate elements of a `Dataset`.

    Use this transformation to produce a dataset that contains one instance of
    each unique element in the input. For example:

    >>> dataset = tf.data.Dataset.from_tensor_slices([1, 37, 2, 37, 2, 1])
    >>> dataset = dataset.unique()
    >>> sorted([a.item() for a in dataset.as_numpy_iterator()])
    [1, 2, 37]

    Note: This transformation only supports datasets which fit into memory
    and have elements of either `tf.int32`, `tf.int64` or `tf.string` type.

    Args:
      name: (Optional.) A name for the tf.data operation.

    Returns:
      A new `Dataset` with the transformation applied as described above.
    """
    # Loaded lazily due to a circular dependency (dataset_ops -> unique_op ->
    # dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import unique_op
    return unique_op._unique(self, name)
    # pylint: enable=g-import-not-at-top,protected-access

  def rejection_resample(
      self, class_func, target_dist, initial_dist=None, seed=None, name=None
  ) -> "DatasetV2":
    # [FUNCTION DOC]
    # Returns a dataset of pairs of class and element resampled so classes follow a target
    # distribution. Inputs: class_func (element to integer class), target_dist, initial_dist
    # (optional; estimated online from the data if not given), seed, name. Builds datasets of
    # acceptance probabilities (with mixing) and the initial distribution, filters this dataset by
    # a random draw against the acceptance probability, prefetches, and then: if the original data
    # already follows the target (static probability one) it only attaches classes; if zero it
    # returns only the filtered data; otherwise it mixes both with sampling weights.
    """Resamples elements to reach a target distribution.

    Note: This implementation can reject **or repeat** elements in order to
    reach the `target_dist`. So, in some cases, the output `Dataset` may be
    larger than the input `Dataset`.

    >>> initial_dist = [0.6, 0.4]
    >>> n = 1000
    >>> elems = np.random.choice(len(initial_dist), size=n, p=initial_dist)
    >>> dataset = tf.data.Dataset.from_tensor_slices(elems)
    >>> zero, one = np.bincount(list(dataset.as_numpy_iterator())) / n

    Following from `initial_dist`, `zero` is ~0.6 and `one` is ~0.4.

    >>> target_dist = [0.5, 0.5]
    >>> dataset = dataset.rejection_resample(
    ...    class_func=lambda x: x,
    ...    target_dist=target_dist,
    ...    initial_dist=initial_dist)
    >>> dataset = dataset.map(lambda class_func_result, data: data)
    >>> zero, one = np.bincount(list(dataset.as_numpy_iterator())) / n

    Following from `target_dist`, `zero` is ~0.5 and `one` is ~0.5.

    Args:
      class_func: A function mapping an element of the input dataset to a scalar
        `tf.int32` tensor. Values should be in `[0, num_classes)`.
      target_dist: A floating point type tensor, shaped `[num_classes]`.
      initial_dist: (Optional.)  A floating point type tensor, shaped
        `[num_classes]`.  If not provided, the true class distribution is
        estimated live in a streaming fashion.
      seed: (Optional.) Python integer seed for the resampler.
      name: (Optional.) A name for the tf.data operation.

    Returns:
      A new `Dataset` with the transformation applied as described above.
    """

    # TODO(b/245793127): Consider switching back to the 'v1' implementation.

    target_dist_t = ops.convert_to_tensor(target_dist, name="target_dist")
    target_dist_t = math_ops.cast(target_dist_t, dtypes.float32)

    # Get initial distribution.
    if initial_dist is not None:
      initial_dist_t = ops.convert_to_tensor(initial_dist, name="initial_dist")
      initial_dist_t = math_ops.cast(initial_dist_t, dtypes.float32)
      acceptance_dist, prob_of_original = (
          _calculate_acceptance_probs_with_mixing(initial_dist_t,
                                                  target_dist_t))
      initial_dist_ds = DatasetV2.from_tensors(
          initial_dist_t, name=name).repeat(name=name)
      acceptance_dist_ds = DatasetV2.from_tensors(
          acceptance_dist, name=name).repeat(name=name)
      prob_of_original_ds = DatasetV2.from_tensors(
          prob_of_original, name=name).repeat(name=name)
    else:
      initial_dist_ds = _estimate_initial_dist_ds(
          target_dist_t, self.map(class_func, name=name), name=name)
      acceptance_and_original_prob_ds = initial_dist_ds.map(
          lambda initial: _calculate_acceptance_probs_with_mixing(  # pylint: disable=g-long-lambda
              initial, target_dist_t),
          name=name)
      acceptance_dist_ds = acceptance_and_original_prob_ds.map(
          lambda accept_prob, _: accept_prob, name=name)
      prob_of_original_ds = acceptance_and_original_prob_ds.map(
          lambda _, prob_original: prob_original, name=name)
    filtered_ds = _filter_ds(self, acceptance_dist_ds, initial_dist_ds,
                             class_func, seed)
    # Prefetch filtered dataset for speed.
    filtered_ds = filtered_ds.prefetch(3, name=name)

    prob_original_static = _get_prob_original_static(
        initial_dist_t, target_dist_t) if initial_dist is not None else None

    def add_class_value(*x):
      # [FUNCTION DOC]
      # HELPER: map function pairing an element with its class (computed by the class function); a
      # single-component element is unwrapped before pairing.
      if len(x) == 1:
        return class_func(*x), x[0]
      else:
        return class_func(*x), x

    if prob_original_static == 1:
      return self.map(add_class_value, name=name)
    elif prob_original_static == 0:
      return filtered_ds
    else:
      return Dataset.sample_from_datasets(
          [self.map(add_class_value), filtered_ds],
          weights=prob_of_original_ds.map(lambda prob: [(prob, 1.0 - prob)]),
          seed=seed,
          stop_on_empty_dataset=True)

  @staticmethod
  def sample_from_datasets(
      datasets,
      weights=None,
      seed=None,
      stop_on_empty_dataset=False,
      rerandomize_each_iteration=None,
  ) -> "DatasetV2":
    # [FUNCTION DOC]
    # Static factory: samples elements at random from several datasets. Inputs: datasets, weights
    # (optional probabilities or a dataset of them), seed, stop_on_empty_dataset (optional),
    # rerandomize_each_iteration (optional). Delegates to the sample-from-datasets op module.
    """Samples elements at random from the datasets in `datasets`.

    Creates a dataset by interleaving elements of `datasets` with `weight[i]`
    probability of picking an element from dataset `i`. Sampling is done without
    replacement. For example, suppose we have 2 datasets:

    ```python
    dataset1 = tf.data.Dataset.range(0, 3)
    dataset2 = tf.data.Dataset.range(100, 103)
    ```

    Suppose that we sample from these 2 datasets with the following weights:

    ```python
    sample_dataset = tf.data.Dataset.sample_from_datasets(
        [dataset1, dataset2], weights=[0.5, 0.5])
    ```

    One possible outcome of elements in sample_dataset is:

    ```
    print(list(sample_dataset.as_numpy_iterator()))
    # [100, 0, 1, 101, 2, 102]
    ```

    Args:
      datasets: A non-empty list of `tf.data.Dataset` objects with compatible
        structure.
      weights: (Optional.) A list or Tensor of `len(datasets)` floating-point
        values where `weights[i]` represents the probability to sample from
        `datasets[i]`, or a `tf.data.Dataset` object where each element is such
        a list. Defaults to a uniform distribution across `datasets`.
      seed: (Optional.) A `tf.int64` scalar `tf.Tensor`, representing the random
        seed that will be used to create the distribution. See
        `tf.random.set_seed` for behavior.
      stop_on_empty_dataset: If `True`, sampling stops if it encounters an empty
        dataset. If `False`, it continues sampling and skips any empty datasets.
        It is recommended to set it to `True`. Otherwise, the distribution of
        samples starts off as the user intends, but may change as input datasets
        become empty. This can be difficult to detect since the dataset starts
        off looking correct. Default to `False` for backward compatibility.
      rerandomize_each_iteration: An optional `bool`. The boolean argument
      controls whether the sequence of random numbers used to determine which
      dataset to sample from will be rerandomized each epoch. That is, it
      determinies whether datasets will be sampled in the same order across
      different epochs (the default behavior) or not.

    Returns:
      A dataset that interleaves elements from `datasets` at random, according
      to `weights` if provided, otherwise with uniform probability.

    Raises:
      TypeError: If the `datasets` or `weights` arguments have the wrong type.
      ValueError:
        - If `datasets` is empty, or
        - If `weights` is specified and does not match the length of `datasets`.
    """
    # Loaded lazily due to a circular dependency
    # (dataset_ops -> sample_from_datasets_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import sample_from_datasets_op
    return sample_from_datasets_op._sample_from_datasets(  # pylint: disable=protected-access
        datasets,
        weights,
        seed,
        stop_on_empty_dataset,
        rerandomize_each_iteration,
    )
    # pylint: enable=g-import-not-at-top,protected-access

  @staticmethod
  def choose_from_datasets(
      datasets, choice_dataset, stop_on_empty_dataset=True
  ) -> "DatasetV2":
    # [FUNCTION DOC]
    # Static factory: builds a dataset that picks elements from the given datasets using a dataset
    # of integer choices. Inputs: datasets, choice_dataset, stop_on_empty_dataset (default true).
    """Creates a dataset that deterministically chooses elements from `datasets`.

    For example, given the following datasets:

    ```python
    datasets = [tf.data.Dataset.from_tensors("foo").repeat(),
                tf.data.Dataset.from_tensors("bar").repeat(),
                tf.data.Dataset.from_tensors("baz").repeat()]

    # Define a dataset containing `[0, 1, 2, 0, 1, 2, 0, 1, 2]`.
    choice_dataset = tf.data.Dataset.range(3).repeat(3)

    result = tf.data.Dataset.choose_from_datasets(datasets, choice_dataset)
    ```

    The elements of `result` will be:

    ```
    "foo", "bar", "baz", "foo", "bar", "baz", "foo", "bar", "baz"
    ```

    Args:
      datasets: A non-empty list of `tf.data.Dataset` objects with compatible
        structure.
      choice_dataset: A `tf.data.Dataset` of scalar `tf.int64` tensors between
        `0` and `len(datasets) - 1`.
      stop_on_empty_dataset: If `True`, selection stops if it encounters an
        empty dataset. If `False`, it skips empty datasets. It is recommended to
        set it to `True`. Otherwise, the selected elements start off as the user
        intends, but may change as input datasets become empty. This can be
        difficult to detect since the dataset starts off looking correct.
        Defaults to `True`.

    Returns:
      A new `Dataset` with the transformation applied as described above.

    Raises:
      TypeError: If `datasets` or `choice_dataset` has the wrong type.
      ValueError: If `datasets` is empty.
    """
    # Loaded lazily due to a circular dependency
    # (dataset_ops -> choose_from_datasets_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import choose_from_datasets_op
    return choose_from_datasets_op._choose_from_datasets(
        datasets, choice_dataset, stop_on_empty_dataset)
    # pylint: enable=g-import-not-at-top,protected-access


@tf_export(v1=["data.Dataset"])
class DatasetV1(DatasetV2, data_types.DatasetV1):
  # [CLASS DOC]
  # Legacy (tf.compat.v1) Dataset class. Extends DatasetV2 with graph-mode iterator creation
  # methods and legacy output class/shape/type properties, and wraps every transformation result
  # in a V1 adapter. Subclasses implement a method returning the variant tensor, which the
  # constructor calls to obtain the handle.
  """Represents a potentially large set of elements.

  A `Dataset` can be used to represent an input pipeline as a
  collection of elements and a "logical plan" of transformations that act on
  those elements.
  """

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor with no arguments. Calls the subclass method that creates the variant tensor, with
    # helpful attribute errors if that method is missing or accessed attributes that are defined after
    # the base constructor call, then initializes the V2 base class with the variant tensor.
    try:
      variant_tensor = self._as_variant_tensor()
    except AttributeError as e:
      if "_as_variant_tensor" in str(e):
        raise AttributeError("Please use `_variant_tensor` instead of "
                             "`_as_variant_tensor()` to obtain the variant "
                             "associated with a dataset.")
      raise AttributeError("{}: A likely cause of this error is that the super "
                           "call for this dataset is not the last line of the "
                           "`__init__` method. The base class invokes the "
                           "`_as_variant_tensor()` method in its constructor "
                           "and if that method uses attributes defined in the "
                           "`__init__` method, those attributes need to be "
                           "defined before the super call.".format(e))
    super(DatasetV1, self).__init__(variant_tensor)

  @abc.abstractmethod
  def _as_variant_tensor(self):
    # [FUNCTION DOC]
    # Abstract method for subclasses to return the scalar variant tensor of the dataset. The base
    # version raises a not-implemented error.
    """Creates a scalar `tf.Tensor` of `tf.variant` representing this dataset.

    Returns:
      A scalar `tf.Tensor` of `tf.variant` type, which represents this dataset.
    """
    raise NotImplementedError(f"{type(self)}.as_variant_tensor()")

  @deprecation.deprecated(
      None, "This is a deprecated API that should only be used in TF 1 graph "
      "mode and legacy TF 2 graph mode available through `tf.compat.v1`. In "
      "all other situations -- namely, eager mode and inside `tf.function` -- "
      "you can consume dataset elements using `for elem in dataset: ...` or "
      "by explicitly creating iterator via `iterator = iter(dataset)` and "
      "fetching its elements via `values = next(iterator)`. Furthermore, "
      "this API is not available in TF 2. During the transition from TF 1 "
      "to TF 2 you can use `tf.compat.v1.data.make_one_shot_iterator(dataset)` "
      "to create a TF 1 graph mode style iterator for a dataset created "
      "through TF 2 APIs. Note that this should be a transient state of your "
      "code base as there are in general no guarantees about the "
      "interoperability of TF 1 and TF 2 code.")
  def make_one_shot_iterator(
      self,
  ) -> Union[iterator_ops.Iterator, iterator_ops.OwnedIterator]:
    # [FUNCTION DOC]
    # Deprecated TF1 API creating a one-shot iterator. Delegates to the internal method. Returns the
    # iterator.
    """Creates an iterator for elements of this dataset.

    Note: The returned iterator will be initialized automatically.
    A "one-shot" iterator does not currently support re-initialization. For
    that see `make_initializable_iterator`.

    Example:

    ```python
    # Building graph ...
    dataset = ...
    next_value = dataset.make_one_shot_iterator().get_next()

    # ... from within a session ...
    try:
      while True:
        value = sess.run(next_value)
        ...
    except tf.errors.OutOfRangeError:
        pass
    ```

    Returns:
      An `tf.data.Iterator` for elements of this dataset.
    """
    return self._make_one_shot_iterator()

  def _make_one_shot_iterator(
      self,
  ) -> Union[iterator_ops.Iterator, iterator_ops.OwnedIterator]:  # pylint: disable=missing-docstring
    # [FUNCTION DOC]
    # Creates a one-shot iterator. In eager mode returns an owned iterator. In graph mode checks all
    # datasets share the current graph, captures the stateless dataset state by value in a defined
    # function (applying the graph-level and op-level seeds), adds it to the graph and returns an
    # iterator over the one-shot iterator op. Raises a value error with an explanatory message if the
    # dataset captures stateful objects, suggesting an initializable iterator.
    if context.executing_eagerly():
      with ops.colocate_with(self._variant_tensor):
        return iterator_ops.OwnedIterator(self)

    _ensure_same_dataset_graph(self)
    # Some ops (e.g. dataset ops) are marked as stateful but are still safe to
    # to capture by value. We must allowlist these ops so that the capturing
    # logic captures the ops instead of raising an exception.
    allowlisted_stateful_ops = traverse.obtain_capture_by_value_ops(self)
    graph_level_seed, op_level_seed = core_random_seed.get_seed(None)

    # NOTE(mrry): We capture by value here to ensure that `_make_dataset()` is
    # a 0-argument function.
    @function.Defun(
        capture_by_value=True,
        allowlisted_stateful_ops=allowlisted_stateful_ops)
    def _make_dataset():
      # [FUNCTION DOC]
      # HELPER: zero-argument factory function that sets the local random seed from the captured seeds
      # and returns the (debug-adjusted) dataset's variant tensor; used by the one-shot iterator op.
      """Factory function for a dataset."""
      # NOTE(mrry): `Defun` does not capture the graph-level seed from the
      # enclosing graph, so if a graph-level seed is present we set the local
      # graph seed based on a combination of the graph- and op-level seeds.
      if graph_level_seed is not None:
        assert op_level_seed is not None
        core_random_seed.set_random_seed(
            (graph_level_seed + 87654321 * op_level_seed) % (2 ** 63 - 1))

      dataset = self._apply_debug_options()
      return dataset._variant_tensor  # pylint: disable=protected-access

    try:
      _make_dataset.add_to_graph(ops.get_default_graph())
    except ValueError as err:
      if "Cannot capture a stateful node" in str(err):
        raise ValueError(
            "{}: A likely cause of this error is that the dataset for which "
            "you are calling `make_one_shot_iterator()` captures a stateful "
            "object, such as a `tf.Variable` or `tf.lookup.StaticHashTable`, "
            "which is not supported. Use `make_initializable_iterator()` "
            "instead.".format(err)) from None
      else:
        raise

    with ops.colocate_with(self._variant_tensor):
      # pylint: disable=protected-access
      return iterator_ops.Iterator(
          gen_dataset_ops.one_shot_iterator(
              dataset_factory=_make_dataset, **self._flat_structure), None,
          get_legacy_output_types(self), get_legacy_output_shapes(self),
          get_legacy_output_classes(self))

  @deprecation.deprecated(
      None, "This is a deprecated API that should only be used in TF 1 graph "
      "mode and legacy TF 2 graph mode available through `tf.compat.v1`. "
      "In all other situations -- namely, eager mode and inside `tf.function` "
      "-- you can consume dataset elements using `for elem in dataset: ...` "
      "or by explicitly creating iterator via `iterator = iter(dataset)` "
      "and fetching its elements via `values = next(iterator)`. "
      "Furthermore, this API is not available in TF 2. During the transition "
      "from TF 1 to TF 2 you can use "
      "`tf.compat.v1.data.make_initializable_iterator(dataset)` to create a TF "
      "1 graph mode style iterator for a dataset created through TF 2 APIs. "
      "Note that this should be a transient state of your code base as there "
      "are in general no guarantees about the interoperability of TF 1 and TF "
      "2 code.")
  def make_initializable_iterator(
      self, shared_name=None
  ) -> iterator_ops.Iterator:
    # [FUNCTION DOC]
    # Deprecated TF1 API creating an initializable iterator, with optional shared_name. Delegates to
    # the internal method and returns the iterator.
    """Creates an iterator for elements of this dataset.

    Note: The returned iterator will be in an uninitialized state,
    and you must run the `iterator.initializer` operation before using it:

    ```python
    # Building graph ...
    dataset = ...
    iterator = dataset.make_initializable_iterator()
    next_value = iterator.get_next()  # This is a Tensor.

    # ... from within a session ...
    sess.run(iterator.initializer)
    try:
      while True:
        value = sess.run(next_value)
        ...
    except tf.errors.OutOfRangeError:
        pass
    ```

    Args:
      shared_name: (Optional.) If non-empty, the returned iterator will be
        shared under the given name across multiple sessions that share the same
        devices (e.g. when using a remote server).

    Returns:
      A `tf.data.Iterator` for elements of this dataset.

    Raises:
      RuntimeError: If eager execution is enabled.
    """
    return self._make_initializable_iterator(shared_name)

  def _make_initializable_iterator(
      self, shared_name=None
  ) -> iterator_ops.Iterator:  # pylint: disable=missing-docstring
    # [FUNCTION DOC]
    # Creates an initializable iterator in graph mode. Raises a runtime error in eager mode. Checks
    # the dataset graph, creates an iterator resource (shared_name defaults to empty) and an
    # initializer op, and returns an iterator wrapping them with the legacy types, shapes, classes.
    if context.executing_eagerly():
      raise RuntimeError("`make_initializable_iterator()` is not supported in "
                         "eager mode. Use Python-style iteration instead.")
    _ensure_same_dataset_graph(self)
    dataset = self._apply_debug_options()
    if shared_name is None:
      shared_name = ""

    with ops.colocate_with(self._variant_tensor):
      iterator_resource = gen_dataset_ops.iterator_v2(
          container="", shared_name=shared_name, **self._flat_structure)

      initializer = gen_dataset_ops.make_iterator(
          dataset._variant_tensor,  # pylint: disable=protected-access
          iterator_resource)

      # pylint: disable=protected-access
      return iterator_ops.Iterator(iterator_resource, initializer,
                                   get_legacy_output_types(dataset),
                                   get_legacy_output_shapes(dataset),
                                   get_legacy_output_classes(dataset))

  @property
  @deprecation.deprecated(
      None, "Use `tf.compat.v1.data.get_output_classes(dataset)`.")
  def output_classes(self):
    # [FUNCTION DOC]
    # Deprecated property returning the legacy output classes of each element component.
    """Returns the class of each component of an element of this dataset.

    Returns:
      A (nested) structure of Python `type` objects corresponding to each
      component of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_classes(),  # pylint: disable=protected-access
        self.element_spec)

  @property
  @deprecation.deprecated(
      None, "Use `tf.compat.v1.data.get_output_shapes(dataset)`.")
  def output_shapes(self):
    # [FUNCTION DOC]
    # Deprecated property returning the legacy output shapes of each element component.
    """Returns the shape of each component of an element of this dataset.

    Returns:
      A (nested) structure of `tf.TensorShape` objects corresponding to each
      component of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_shapes(),  # pylint: disable=protected-access
        self.element_spec)

  @property
  @deprecation.deprecated(
      None, "Use `tf.compat.v1.data.get_output_types(dataset)`.")
  def output_types(self):
    # [FUNCTION DOC]
    # Deprecated property returning the legacy output types of each element component.
    """Returns the type of each component of an element of this dataset.

    Returns:
      A (nested) structure of `tf.DType` objects corresponding to each component
      of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_types(),  # pylint: disable=protected-access
        self.element_spec)

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property override that builds the element spec from the legacy output types, shapes and
    # classes (pending removal once all datasets implement the structure directly).
    # TODO(b/110122868): Remove this override once all `Dataset` instances
    # implement `element_structure`.
    return structure.convert_legacy_structure(
        self.output_types, self.output_shapes, self.output_classes)

  @staticmethod
  @functools.wraps(DatasetV2.from_tensors)
  def from_tensors(tensors, name=None):
    # [FUNCTION DOC]
    # V1 wrapper of the same-named V2 factory; returns the result wrapped in a V1 adapter.
    return DatasetV1Adapter(DatasetV2.from_tensors(tensors, name=name))

  @staticmethod
  @functools.wraps(DatasetV2.from_tensor_slices)
  def from_tensor_slices(tensors, name=None):
    # [FUNCTION DOC]
    # V1 wrapper of the same-named V2 factory; returns the result wrapped in a V1 adapter.
    return DatasetV1Adapter(DatasetV2.from_tensor_slices(tensors, name=name))

  @staticmethod
  @deprecation.deprecated(None, "Use `tf.data.Dataset.from_tensor_slices()`.")
  def from_sparse_tensor_slices(sparse_tensor):
    # [FUNCTION DOC]
    # Deprecated static factory creating a dataset with one element per slice of a sparse tensor.
    # Delegates to the lazily imported sparse-slices op module.
    """Splits each rank-N `tf.sparse.SparseTensor` in this dataset row-wise.

    Args:
      sparse_tensor: A `tf.sparse.SparseTensor`.

    Returns:
      Dataset: A `Dataset` of rank-(N-1) sparse tensors.
    """
    # Loaded lazily due to a circular dependency (dataset_ops ->
    # from_sparse_tensor_slices_op -> dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import from_sparse_tensor_slices_op
    return from_sparse_tensor_slices_op._from_sparse_tensor_slices(
        sparse_tensor)
    # pylint: enable=g-import-not-at-top,protected-access

  @staticmethod
  @functools.wraps(DatasetV2.from_generator)
  @deprecation.deprecated_args(None, "Use output_signature instead",
                               "output_types", "output_shapes")
  def from_generator(generator,
                     output_types=None,
                     output_shapes=None,
                     args=None,
                     output_signature=None,
                     name=None):
    # [FUNCTION DOC]
    # V1 wrapper of the V2 generator factory with deprecation warnings silenced for the inner call
    # (the decorator already warns); returns the result wrapped in a V1 adapter. Same inputs.
    # Calling DatasetV2.from_generator with output_shapes or output_types is
    # deprecated, but this is already checked by the decorator on this function.
    with deprecation.silence():
      return DatasetV1Adapter(
          DatasetV2.from_generator(
              generator,
              output_types,
              output_shapes,
              args,
              output_signature,
              name=name))

  @staticmethod
  @functools.wraps(DatasetV2.range)
  def range(*args, **kwargs):
    # [FUNCTION DOC]
    # V1 wrapper of the V2 range factory; returns the result wrapped in a V1 adapter.
    return DatasetV1Adapter(DatasetV2.range(*args, **kwargs))

  @staticmethod
  @functools.wraps(DatasetV2.zip)
  def zip(*args, datasets=None, name=None):
    # [FUNCTION DOC]
    # V1 wrapper of the V2 zip factory; returns the result wrapped in a V1 adapter.
    return DatasetV1Adapter(DatasetV2.zip(*args, datasets=datasets, name=name))

  @functools.wraps(DatasetV2.concatenate)
  def concatenate(self, dataset, name=None):
    # [FUNCTION DOC]
    # V1 wrapper of the V2 method; returns the result wrapped in a V1 adapter.
    return DatasetV1Adapter(
        super(DatasetV1, self).concatenate(dataset, name=name))

  @functools.wraps(DatasetV2.prefetch)
  def prefetch(self, buffer_size, name=None):
    # [FUNCTION DOC]
    # V1 wrapper of the V2 method; returns the result wrapped in a V1 adapter.
    return DatasetV1Adapter(
        super(DatasetV1, self).prefetch(buffer_size, name=name))

  @staticmethod
  @functools.wraps(DatasetV2.list_files)
  def list_files(file_pattern, shuffle=None, seed=None, name=None):
    # [FUNCTION DOC]
    # V1 wrapper of the V2 list-files factory; returns the result wrapped in a V1 adapter.
    return DatasetV1Adapter(
        DatasetV2.list_files(file_pattern, shuffle, seed, name=name))

  @functools.wraps(DatasetV2.repeat)
  def repeat(self, count=None, name=None):
    # [FUNCTION DOC]
    # V1 wrapper of the V2 method; returns the result wrapped in a V1 adapter.
    return DatasetV1Adapter(super(DatasetV1, self).repeat(count, name=name))

  @functools.wraps(DatasetV2.shuffle)
  def shuffle(self,
              buffer_size,
              seed=None,
              reshuffle_each_iteration=None,
              name=None):
    # [FUNCTION DOC]
    # V1 wrapper of the V2 method; returns the result wrapped in a V1 adapter. Note the default for
    # reshuffle_each_iteration is unset here (the V2 default is true) and the value is passed on as is.
    return DatasetV1Adapter(
        super(DatasetV1, self).shuffle(
            buffer_size, seed, reshuffle_each_iteration, name=name))

  @functools.wraps(DatasetV2.cache)
  def cache(self, filename="", name=None):
    # [FUNCTION DOC]
    # V1 wrapper of the V2 method; returns the result wrapped in a V1 adapter.
    return DatasetV1Adapter(super(DatasetV1, self).cache(filename, name=name))

  @functools.wraps(DatasetV2.take)
  def take(self, count, name=None):
    # [FUNCTION DOC]
    # V1 wrapper of the V2 method; returns the result wrapped in a V1 adapter.
    return DatasetV1Adapter(super(DatasetV1, self).take(count, name=name))

  @functools.wraps(DatasetV2.skip)
  def skip(self, count, name=None):
    # [FUNCTION DOC]
    # V1 wrapper of the V2 method; returns the result wrapped in a V1 adapter.
    return DatasetV1Adapter(super(DatasetV1, self).skip(count, name=name))

  @functools.wraps(DatasetV2.shard)
  def shard(self, num_shards, index, name=None):
    # [FUNCTION DOC]
    # V1 wrapper of the V2 method; returns the result wrapped in a V1 adapter.
    return DatasetV1Adapter(
        super(DatasetV1, self).shard(num_shards, index, name=name))

  @functools.wraps(DatasetV2.batch)
  def batch(self,
            batch_size,
            drop_remainder=False,
            num_parallel_calls=None,
            deterministic=None,
            name=None):
    # [FUNCTION DOC]
    # V1 wrapper of the V2 method; returns the result wrapped in a V1 adapter.
    return DatasetV1Adapter(
        super(DatasetV1, self).batch(
            batch_size,
            drop_remainder,
            num_parallel_calls,
            deterministic,
            name=name))

  @functools.wraps(DatasetV2.padded_batch)
  def padded_batch(self,
                   batch_size,
                   padded_shapes=None,
                   padding_values=None,
                   drop_remainder=False,
                   name=None):
    # [FUNCTION DOC]
    # V1 wrapper of the V2 method; returns the result wrapped in a V1 adapter.
    return DatasetV1Adapter(
        super(DatasetV1, self).padded_batch(
            batch_size,
            padded_shapes,
            padding_values,
            drop_remainder,
            name=name))

  @functools.wraps(DatasetV2.map)
  def map(
      self,
      map_func,
      num_parallel_calls=None,
      deterministic=None,
      synchronous=None,
      use_unbounded_threadpool=False,
      name=None,
  ):
    # [FUNCTION DOC]
    # KNOWN ISSUE: accepts a name argument (like the V2 version) but does not pass it on to the V1 map
    # implementation, so the name is silently ignored.
    # V1 map: delegates to the V1 map implementation in the map op module with the parallelism,
    # determinism, synchronous and unbounded-threadpool options; returns the resulting dataset.
    # Loaded lazily due to a circular dependency (dataset_ops -> map_op ->
    # dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import map_op

    return map_op._map_v1(
        self,
        map_func,
        num_parallel_calls=num_parallel_calls,
        deterministic=deterministic,
        synchronous=synchronous,
        use_unbounded_threadpool=use_unbounded_threadpool,
    )
    # pylint: enable=g-import-not-at-top,protected-access

  @deprecation.deprecated(None, "Use `tf.data.Dataset.map()")
  def map_with_legacy_function(
      self,
      map_func,
      num_parallel_calls=None,
      deterministic=None,
      use_unbounded_threadpool=False,
  ) -> "DatasetV1Adapter":
    # [FUNCTION DOC]
    # Deprecated: maps a function over elements using the legacy function wrapper. Inputs: map_func,
    # num_parallel_calls, deterministic, use_unbounded_threadpool (optional). Delegates to the legacy
    # map implementation in the map op module and returns the dataset.
    """Maps `map_func` across the elements of this dataset.

    Note: This is an escape hatch for existing uses of `map` that do not work
    with V2 functions. New uses are strongly discouraged and existing uses
    should migrate to `map` as this method will be removed in V2.

    Args:
      map_func: A function mapping a (nested) structure of tensors (having
        shapes and types defined by `self.output_shapes` and
        `self.output_types`) to another (nested) structure of tensors.
      num_parallel_calls: (Optional.) A `tf.int32` scalar `tf.Tensor`,
        representing the number elements to process asynchronously in parallel.
        If not specified, elements will be processed sequentially. If the value
        `tf.data.AUTOTUNE` is used, then the number of parallel calls is set
        dynamically based on available CPU.
      deterministic: (Optional.) When `num_parallel_calls` is specified, this
        boolean controls the order in which the transformation produces
        elements. If set to `False`, the transformation is allowed to yield
        elements out of order to trade determinism for performance. If not
        specified, the `tf.data.Options.deterministic` option (`True` by
        default) controls the behavior.
      use_unbounded_threadpool: (Optional.) By default, map functions run in a
        limited threadpool based on the number of cores on the machine. This
        efficient for CPU-heavy processing, but if the map function performs IO
        it is better to use an unbounded threadpool by setting it to `True`. It
        is `False` by default.

    Returns:
      Dataset: A `Dataset`.
    """
    # Loaded lazily due to a circular dependency (dataset_ops -> map_op ->
    # dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import map_op
    return map_op._map_v1_with_legacy_function(
        self,
        map_func,
        num_parallel_calls=num_parallel_calls,
        deterministic=deterministic)
    # pylint: enable=g-import-not-at-top,protected-access

  @functools.wraps(DatasetV2.flat_map)
  def flat_map(self, map_func, name=None) -> "DatasetV1Adapter":
    # [FUNCTION DOC]
    # V1 wrapper of the V2 method; returns the result wrapped in a V1 adapter.
    return DatasetV1Adapter(
        super(DatasetV1, self).flat_map(map_func, name=name))

  @functools.wraps(DatasetV2.interleave)
  def interleave(
      self,
      map_func,
      cycle_length=None,
      block_length=None,
      num_parallel_calls=None,
      deterministic=None,
      name=None,
  ) -> "DatasetV1Adapter":
    # [FUNCTION DOC]
    # V1 wrapper of the V2 method; returns the result wrapped in a V1 adapter.
    return DatasetV1Adapter(
        super(DatasetV1, self).interleave(
            map_func,
            cycle_length,
            block_length,
            num_parallel_calls,
            deterministic,
            name=name))

  @functools.wraps(DatasetV2.filter)
  def filter(self, predicate, name=None) -> "DatasetV1Adapter":
    # [FUNCTION DOC]
    # V1 wrapper of the V2 method; returns the result wrapped in a V1 adapter.
    return DatasetV1Adapter(super(DatasetV1, self).filter(predicate, name=name))

  @deprecation.deprecated(None, "Use `tf.data.Dataset.filter()")
  def filter_with_legacy_function(self, predicate) -> "DatasetV2":
    # [FUNCTION DOC]
    # Deprecated: filters with a predicate using the legacy function wrapper, by building the filter
    # dataset directly in the filter op module. Returns that dataset (not wrapped in a V1 adapter).
    """Filters this dataset according to `predicate`.

    Note: This is an escape hatch for existing uses of `filter` that do not work
    with V2 functions. New uses are strongly discouraged and existing uses
    should migrate to `filter` as this method will be removed in V2.

    Args:
      predicate: A function mapping a (nested) structure of tensors (having
        shapes and types defined by `self.output_shapes` and
        `self.output_types`) to a scalar `tf.bool` tensor.

    Returns:
      Dataset: The `Dataset` containing the elements of this dataset for which
          `predicate` is `True`.
    """
    # Loaded lazily due to a circular dependency (dataset_ops -> filter_op ->
    # dataset_ops).
    # pylint: disable=g-import-not-at-top,protected-access
    from tensorflow.python.data.ops import filter_op
    return filter_op._FilterDataset(self, predicate, use_legacy_function=True)
    # pylint: enable=g-import-not-at-top,protected-access

  @functools.wraps(DatasetV2.apply)
  def apply(self, transformation_func) -> "DatasetV1Adapter":
    # [FUNCTION DOC]
    # V1 wrapper of the V2 method; returns the result wrapped in a V1 adapter.
    return DatasetV1Adapter(super(DatasetV1, self).apply(transformation_func))

  @functools.wraps(DatasetV2.window)
  def window(
      self, size, shift=None, stride=1, drop_remainder=False, name=None
  ) -> "DatasetV1Adapter":
    # [FUNCTION DOC]
    # V1 wrapper of the V2 method; returns the result wrapped in a V1 adapter.
    return DatasetV1Adapter(
        super(DatasetV1,
              self).window(size, shift, stride, drop_remainder, name=name))

  @functools.wraps(DatasetV2.unbatch)
  def unbatch(self, name=None) -> "DatasetV1Adapter":
    # [FUNCTION DOC]
    # V1 wrapper of the V2 method; returns the result wrapped in a V1 adapter.
    return DatasetV1Adapter(super(DatasetV1, self).unbatch(name=name))

  @functools.wraps(DatasetV2.with_options)
  def with_options(self, options, name=None) -> "DatasetV1Adapter":
    # [FUNCTION DOC]
    # V1 wrapper of the V2 method; returns the result wrapped in a V1 adapter.
    return DatasetV1Adapter(
        super(DatasetV1, self).with_options(options, name=name))


if tf2.enabled():
  Dataset = DatasetV2
else:
  Dataset = DatasetV1


def _tf2_callback():
  # [FUNCTION DOC]
  # HELPER: callback registered with the v2 compat layer; when TF2 behavior is toggled it rebinds
  # the module-level Dataset name to DatasetV2 (enabled) or DatasetV1 (disabled). Takes no
  # arguments, returns nothing, mutates the global.
  global Dataset
  if tf2.enabled():
    Dataset = DatasetV2
  else:
    Dataset = DatasetV1


v2_compat.register_data_v2_callback(_tf2_callback)


class DatasetV1Adapter(DatasetV1):
  # [CLASS DOC]
  # Wraps a V2 dataset so it exposes the V1 dataset API. Holds the wrapped dataset and forwards
  # the variant tensor, inputs, functions, options, element spec and iteration to it.
  """Wraps a V2 `Dataset` object in the `tf.compat.v1.data.Dataset` API."""

  def __init__(self, dataset: DatasetV2):
    # [FUNCTION DOC]
    # Stores the wrapped V2 dataset, then runs the V1 constructor (which obtains the variant tensor
    # through the variant-tensor method below).
    self._dataset = dataset
    super(DatasetV1Adapter, self).__init__()

  def _as_variant_tensor(self):
    # [FUNCTION DOC]
    # Returns the variant tensor of the wrapped dataset.
    return self._dataset._variant_tensor  # pylint: disable=protected-access

  def _inputs(self):
    # [FUNCTION DOC]
    # Returns the input datasets of the wrapped dataset.
    return self._dataset._inputs()  # pylint: disable=protected-access

  def _functions(self) -> list[StructuredFunctionWrapper]:
    # [FUNCTION DOC]
    # Returns the structured functions of the wrapped dataset.
    return self._dataset._functions()  # pylint: disable=protected-access

  def options(self):
    # [FUNCTION DOC]
    # Returns the options of the wrapped dataset.
    return self._dataset.options()

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the element spec of the wrapped dataset.
    return self._dataset.element_spec  # pylint: disable=protected-access

  def __iter__(self):
    # [FUNCTION DOC]
    # Returns an iterator over the wrapped dataset.
    return iter(self._dataset)


def _ensure_same_dataset_graph(dataset):
  # [FUNCTION DOC]
  # HELPER: used by the V1 iterator constructors. Walks the dataset and its transitive inputs
  # breadth-first and raises a value error if any was created in a graph different from the current
  # default graph. Returns nothing.
  """Walks the dataset graph to ensure all datasets come from the same graph."""
  # pylint: disable=protected-access
  current_graph = ops.get_default_graph()
  bfs_q = queue.Queue()
  bfs_q.put(dataset)
  visited = []
  while not bfs_q.empty():
    ds = bfs_q.get()
    visited.append(ds)
    ds_graph = ds._graph
    if current_graph != ds_graph:
      raise ValueError(
          f"The graph {current_graph} of the iterator is different from the "
          f"graph {ds_graph} the dataset: {ds._variant_tensor} was created in. "
          f"Make sure that the dataset is created in the same graph as the "
          f"iterator.")
    for input_ds in ds._inputs():
      if input_ds not in visited:
        bfs_q.put(input_ds)


@tf_export(v1=["data.make_one_shot_iterator"])
def make_one_shot_iterator(
    dataset: DatasetV1,
) -> Union[iterator_ops.Iterator, iterator_ops.OwnedIterator]:
  # [FUNCTION DOC]
  # Compat.v1 module-level function: creates a one-shot iterator for a dataset. Uses the dataset's
  # own method when it exists (some datasets override it), else wraps it in a V1 adapter first.
  # Returns the iterator.
  """Creates an iterator for elements of `dataset`.

  Note: The returned iterator will be initialized automatically.
  A "one-shot" iterator does not support re-initialization.

  Args:
    dataset: A `tf.data.Dataset`.

  Returns:
    A `tf.data.Iterator` for elements of `dataset`.

  @compatibility(TF2)
  This is a legacy API for consuming dataset elements and should only be used
  during transition from TF 1 to TF 2. Note that using this API should be
  a transient state of your code base as there are in general no guarantees
  about the interoperability of TF 1 and TF 2 code.

  In TF 2 datasets are Python iterables which means you can consume their
  elements using `for elem in dataset: ...` or by explicitly creating iterator
  via `iterator = iter(dataset)` and fetching its elements via
  `values = next(iterator)`.
  @end_compatibility
  """
  try:
    # Call the defined `_make_one_shot_iterator()` if there is one, because some
    # datasets (e.g. for prefetching) override its behavior.
    return dataset._make_one_shot_iterator()  # pylint: disable=protected-access
  except AttributeError:
    return DatasetV1Adapter(dataset)._make_one_shot_iterator()  # pylint: disable=protected-access


@tf_export(v1=["data.make_initializable_iterator"])
def make_initializable_iterator(
    dataset: DatasetV1, shared_name=None
) -> iterator_ops.Iterator:
  # [FUNCTION DOC]
  # Compat.v1 module-level function: creates an initializable iterator for a dataset, with optional
  # shared_name. Uses the dataset's own method if available, else wraps it in a V1 adapter.
  # Returns the iterator, which must be initialized before use.
  """Creates an iterator for elements of `dataset`.

  Note: The returned iterator will be in an uninitialized state,
  and you must run the `iterator.initializer` operation before using it:

  ```python
  dataset = ...
  iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
  # ...
  sess.run(iterator.initializer)
  ```

  Args:
    dataset: A `tf.data.Dataset`.
    shared_name: (Optional.) If non-empty, the returned iterator will be shared
      under the given name across multiple sessions that share the same devices
      (e.g. when using a remote server).

  Returns:
    A `tf.data.Iterator` for elements of `dataset`.

  Raises:
    RuntimeError: If eager execution is enabled.

  @compatibility(TF2)
  This is a legacy API for consuming dataset elements and should only be used
  during transition from TF 1 to TF 2. Note that using this API should be
  a transient state of your code base as there are in general no guarantees
  about the interoperability of TF 1 and TF 2 code.

  In TF 2 datasets are Python iterables which means you can consume their
  elements using `for elem in dataset: ...` or by explicitly creating iterator
  via `iterator = iter(dataset)` and fetching its elements via
  `values = next(iterator)`.
  @end_compatibility
  """
  try:
    # Call the defined `_make_initializable_iterator()` if there is one, because
    # some datasets (e.g. for prefetching) override its behavior.
    return dataset._make_initializable_iterator(shared_name)  # pylint: disable=protected-access
  except AttributeError:
    return DatasetV1Adapter(dataset)._make_initializable_iterator(shared_name)  # pylint: disable=protected-access


@tf_export("data.experimental.get_structure")
def get_structure(dataset_or_iterator):
  # [FUNCTION DOC]
  # Returns the element spec (nested type specification) of a dataset or iterator. Raises a type
  # error if the argument has no element spec.
  """Returns the type signature for elements of the input dataset / iterator.

  For example, to get the structure of a `tf.data.Dataset`:

  >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
  >>> tf.data.experimental.get_structure(dataset)
  TensorSpec(shape=(), dtype=tf.int32, name=None)

  >>> dataset = tf.data.experimental.from_list([(1, 'a'), (2, 'b'), (3, 'c')])
  >>> tf.data.experimental.get_structure(dataset)
  (TensorSpec(shape=(), dtype=tf.int32, name=None),
   TensorSpec(shape=(), dtype=tf.string, name=None))

  To get the structure of an `tf.data.Iterator`:

  >>> dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3])
  >>> tf.data.experimental.get_structure(iter(dataset))
  TensorSpec(shape=(), dtype=tf.int32, name=None)

  Args:
    dataset_or_iterator: A `tf.data.Dataset` or an `tf.data.Iterator`.

  Returns:
    A (nested) structure of `tf.TypeSpec` objects matching the structure of an
    element of `dataset_or_iterator` and specifying the type of individual
    components.

  Raises:
    TypeError: If input is not a `tf.data.Dataset` or an `tf.data.Iterator`
      object.
  """
  try:
    return dataset_or_iterator.element_spec  # pylint: disable=protected-access
  except AttributeError:
    raise TypeError(f"Invalid `dataset_or_iterator`. `dataset_or_iterator` "
                    f"must be a `tf.data.Dataset` or tf.data.Iterator object, "
                    f"but got {type(dataset_or_iterator)}.")


@tf_export(v1=["data.get_output_classes"])
def get_legacy_output_classes(dataset_or_iterator):
  # [FUNCTION DOC]
  # Compat.v1 function: returns the legacy output classes for each component of a dataset or
  # iterator element, mapped over its structure.
  """Returns the output classes for elements of the input dataset / iterator.

  Args:
    dataset_or_iterator: A `tf.data.Dataset` or `tf.data.Iterator`.

  Returns:
    A (nested) structure of Python `type` objects matching the structure of the
    dataset / iterator elements and specifying the class of the individual
    components.

  @compatibility(TF2)
  This is a legacy API for inspecting the type signature of dataset elements. In
  TF 2, you should use the `tf.data.Dataset.element_spec` attribute instead.
  @end_compatibility
  """
  return nest.map_structure(
      lambda component_spec: component_spec._to_legacy_output_classes(),  # pylint: disable=protected-access
      get_structure(dataset_or_iterator))


@tf_export(v1=["data.get_output_shapes"])
def get_legacy_output_shapes(dataset_or_iterator):
  # [FUNCTION DOC]
  # Compat.v1 function: returns the legacy output shapes for each component of a dataset or
  # iterator element, mapped over its structure.
  """Returns the output shapes for elements of the input dataset / iterator.

  Args:
    dataset_or_iterator: A `tf.data.Dataset` or `tf.data.Iterator`.

  Returns:
    A (nested) structure of `tf.TensorShape` objects matching the structure of
    the dataset / iterator elements and specifying the shape of the individual
    components.

  @compatibility(TF2)
  This is a legacy API for inspecting the type signature of dataset elements. In
  TF 2, you should use the `tf.data.Dataset.element_spec` attribute instead.
  @end_compatibility
  """
  return nest.map_structure(
      lambda component_spec: component_spec._to_legacy_output_shapes(),  # pylint: disable=protected-access
      get_structure(dataset_or_iterator))


@tf_export(v1=["data.get_output_types"])
def get_legacy_output_types(dataset_or_iterator):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it returns the output shapes, but the code returns the legacy
  # output types.
  # Compat.v1 function: returns the legacy output types for each component of a dataset or
  # iterator element, mapped over its structure.
  """Returns the output shapes for elements of the input dataset / iterator.

  Args:
    dataset_or_iterator: A `tf.data.Dataset` or `tf.data.Iterator`.

  Returns:
    A (nested) structure of `tf.DType` objects matching the structure of
    dataset / iterator elements and specifying the shape of the individual
    components.

  @compatibility(TF2)
  This is a legacy API for inspecting the type signature of dataset elements. In
  TF 2, you should use the `tf.data.Dataset.element_spec` attribute instead.
  @end_compatibility
  """
  return nest.map_structure(
      lambda component_spec: component_spec._to_legacy_output_types(),  # pylint: disable=protected-access
      get_structure(dataset_or_iterator))


class DatasetSource(DatasetV2):
  # [CLASS DOC]
  # Abstract base for datasets that have no input datasets (data sources).
  """Abstract class representing a dataset with no inputs."""

  def _inputs(self):
    # [FUNCTION DOC]
    # Returns an empty list: sources have no inputs.
    return []


class UnaryDataset(DatasetV2):
  # [CLASS DOC]
  # Abstract base for datasets with exactly one input dataset. Holds the input in an attribute.
  """Abstract class representing a dataset with one input."""

  def __init__(self, input_dataset: DatasetV2, variant_tensor):
    # [FUNCTION DOC]
    # Stores the input dataset and initializes the base class with the given variant tensor.
    self._input_dataset = input_dataset
    super(UnaryDataset, self).__init__(variant_tensor)

  def _inputs(self):
    # [FUNCTION DOC]
    # Returns a list with the single input dataset.
    return [self._input_dataset]


class UnaryUnchangedStructureDataset(UnaryDataset):
  # [CLASS DOC]
  # Unary dataset whose output has the same structure as its input, so its element spec is taken
  # from the input.
  """Represents a unary dataset with the same input and output structure."""

  def __init__(self, input_dataset: DatasetV2, variant_tensor):
    # [FUNCTION DOC]
    # Stores the input dataset and initializes the unary base class with the input and variant tensor.
    self._input_dataset = input_dataset
    super(UnaryUnchangedStructureDataset, self).__init__(
        input_dataset, variant_tensor)

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the element spec of the input dataset.
    return self._input_dataset.element_spec


class _VariantDataset(DatasetV2):
  # [CLASS DOC]
  # HELPER: dataset wrapper around a variant-typed tensor (for example a function argument) plus a
  # known element spec. It has no input datasets; used when rebuilding datasets from variants.
  """A Dataset wrapper around a `tf.variant`-typed function argument."""

  def __init__(self, dataset_variant, element_spec):
    # [FUNCTION DOC]
    # Stores the element spec and initializes the base class with the dataset variant tensor.
    self._element_spec = element_spec
    super(_VariantDataset, self).__init__(dataset_variant)

  def _inputs(self):
    # [FUNCTION DOC]
    # Returns an empty list: the dataset is opaque, so it reports no inputs.
    return []

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the stored element spec.
    return self._element_spec


class _NestedVariant(composite_tensor.CompositeTensor):
  # [CLASS DOC]
  # HELPER: composite tensor holding a batch (non-scalar shape) of dataset variants together with
  # the element spec and the dataset shape; its type spec is a dataset spec with that shape.

  def __init__(self, variant_tensor, element_spec, dataset_shape):
    # [FUNCTION DOC]
    # Stores the variant tensor, element spec and dataset shape.
    self._variant_tensor = variant_tensor
    self._element_spec = element_spec
    self._dataset_shape = dataset_shape

  @property
  def _type_spec(self):
    # [FUNCTION DOC]
    # Property returning a dataset spec built from the element spec and dataset shape.
    return DatasetSpec(self._element_spec, self._dataset_shape)


@tf_export("data.experimental.from_variant")
def from_variant(variant, structure):
  # [FUNCTION DOC]
  # Constructs a dataset from a scalar variant tensor and a nested element spec (the structure)
  # and returns it as an opaque variant-backed dataset.
  """Constructs a dataset from the given variant and (nested) structure.

  Args:
    variant: A scalar `tf.variant` tensor representing a dataset.
    structure: A (nested) structure of `tf.TypeSpec` objects representing the
      structure of each element in the dataset.

  Returns:
    A `tf.data.Dataset` instance.
  """
  return _VariantDataset(variant, structure)  # pylint: disable=protected-access


@tf_export("data.experimental.to_variant")
def to_variant(dataset: DatasetV2):
  # [FUNCTION DOC]
  # Returns the variant tensor that represents the given dataset.
  """Returns a variant representing the given dataset.

  Args:
    dataset: A `tf.data.Dataset`.

  Returns:
    A scalar `tf.variant` tensor representing the given dataset.
  """
  return dataset._variant_tensor  # pylint: disable=protected-access


@tf_export(
    "data.DatasetSpec",
    v1=["data.DatasetSpec", "data.experimental.DatasetStructure"])
class DatasetSpec(type_spec.BatchableTypeSpec):
  # [CLASS DOC]
  # Batchable type specification for datasets. Holds the element spec and the dataset shape
  # (empty for a single dataset, otherwise the shape of a batch of datasets). Implements subtype
  # and supertype checks, component conversion to and from variant tensors, batching and
  # unbatching, and equality. Registered with the type spec codec for saving.
  """Type specification for `tf.data.Dataset`.

  See `tf.TypeSpec` for more information about TensorFlow type specifications.

  >>> dataset = tf.data.Dataset.range(3)
  >>> tf.data.DatasetSpec.from_value(dataset)
  DatasetSpec(TensorSpec(shape=(), dtype=tf.int64, name=None), TensorShape([]))
  """

  __slots__ = ["_element_spec", "_dataset_shape"]

  def __init__(self, element_spec, dataset_shape=()):
    # [FUNCTION DOC]
    # Stores the element spec and the dataset shape (default scalar), converting the shape to a
    # tensor shape object.
    self._element_spec = element_spec
    self._dataset_shape = tensor_shape.as_shape(dataset_shape)

  @property
  def value_type(self):
    # [FUNCTION DOC]
    # Property returning the current default dataset class.
    return Dataset

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the inner element spec.
    """The inner element spec."""
    return self._element_spec

  def is_subtype_of(self, other):
    # [FUNCTION DOC]
    # Returns true if this spec is a subtype of the other: same spec class, same element structure
    # with each pair of components being subtype-or-equal, and the dataset shape a subtype of the
    # other's. Returns false on any mismatch or structure error.
    """See base class."""
    if type(self) is not type(other):
      return False

    # TODO(b/220385675): _element_spec should always be a TypeSpec.
    try:
      tf_nest.assert_same_structure(self.element_spec, other.element_spec)
    except (TypeError, ValueError):
      return False

    self_elements = tf_nest.flatten(self.element_spec)
    other_elements = tf_nest.flatten(other.element_spec)

    def is_subtype_or_equal(a, b):
      # [FUNCTION DOC]
      # HELPER: component comparison: uses the subtype check for trace types, otherwise equality.
      if isinstance(a, trace.TraceType):
        return a.is_subtype_of(b)
      else:
        return a == b

    for self_element, other_element in zip(self_elements, other_elements):
      if not is_subtype_or_equal(self_element, other_element):
        return False

    return self._dataset_shape.is_subtype_of(other._dataset_shape)  # pylint: disable=protected-access

  def most_specific_common_supertype(self, others):
    # [FUNCTION DOC]
    # Returns the most specific common supertype of this spec and a list of others, or nothing if one
    # does not exist (different classes, different structure, no common supertype for some component
    # or for the dataset shape). Components are compared one by one; the result keeps the structure.
    """See base class."""
    if not all(type(self) is type(other) for other in others):
      return None

    try:
      for other in others:
        tf_nest.assert_same_structure(self.element_spec, other.element_spec)
    except (TypeError, ValueError):
      return None

    self_components = tf_nest.flatten(self.element_spec)
    others_components = [
        tf_nest.flatten(other.element_spec) for other in others
    ]
    common_components = [None] * len(self_components)

    def common_supertype_or_equal(a, bs):
      # [FUNCTION DOC]
      # HELPER: for a component and a list of others, returns the common supertype for trace types,
      # otherwise the component itself if all are equal, else nothing.
      if isinstance(a, trace.TraceType):
        return a.most_specific_common_supertype(bs)
      else:
        return a if all(a == b for b in bs) else None

    for i, self_component in enumerate(self_components):
      common_components[i] = common_supertype_or_equal(
          self_component,
          [other_components[i] for other_components in others_components])
      if self_component is not None and common_components[i] is None:
        return None
    common_element_spec = tf_nest.pack_sequence_as(self._element_spec,
                                                   common_components)

    common_dataset_shape = self._dataset_shape.most_specific_common_supertype(
        [other._dataset_shape for other in others])  # pylint: disable=protected-access
    if common_dataset_shape is None:
      return None

    return DatasetSpec(common_element_spec, common_dataset_shape)

  # TODO(b/220385675): Once _element_spec is guaranteed to be TypeSpec, the
  # following functions do not need to be overloaded: is_subtype_of,
  # most_specific_common_supertype, __hash__ and __eq__
  def _serialize(self):
    # [FUNCTION DOC]
    # Returns a tuple of the element spec and dataset shape that identifies this spec.
    return (self._element_spec, self._dataset_shape)

  @property
  def _component_specs(self):
    # [FUNCTION DOC]
    # Property returning the single tensor spec of variant type with the dataset shape.
    return tensor_spec.TensorSpec(self._dataset_shape, dtypes.variant)

  def _to_components(self, value):
    # [FUNCTION DOC]
    # Returns the variant tensor of the given dataset value.
    return value._variant_tensor  # pylint: disable=protected-access

  def _from_components(self, components):
    # [FUNCTION DOC]
    # Rebuilds a value from the variant tensor: a variant-backed dataset when the dataset shape is
    # scalar, otherwise a nested-variant composite.
    # pylint: disable=protected-access
    if self._dataset_shape.ndims == 0:
      return _VariantDataset(components, self._element_spec)
    else:
      return _NestedVariant(components, self._element_spec, self._dataset_shape)

  def _to_tensor_list(self, value):
    # [FUNCTION DOC]
    # Returns a one-item list with the tensor converted from the variant tensors of the value (mapped
    # over a possibly nested structure).
    return [
        ops.convert_to_tensor(
            tf_nest.map_structure(lambda x: x._variant_tensor, value))  # pylint: disable=protected-access
    ]

  @staticmethod
  def from_value(value):
    # [FUNCTION DOC]
    # Static method: creates a dataset spec for a given dataset, using its element spec.
    """Creates a `DatasetSpec` for the given `tf.data.Dataset` value."""
    return DatasetSpec(value.element_spec)  # pylint: disable=protected-access

  def _batch(self, batch_size):
    # [FUNCTION DOC]
    # Returns a spec whose dataset shape is the batch size prepended to this shape.
    return DatasetSpec(
        self._element_spec,
        tensor_shape.TensorShape([batch_size]).concatenate(self._dataset_shape))

  def _unbatch(self):
    # [FUNCTION DOC]
    # Returns a spec with the leading dimension of the dataset shape removed. Raises a value error
    # when the dataset shape is a scalar.
    if self._dataset_shape.ndims == 0:
      raise ValueError("Slicing dataset elements is not supported for rank 0.")
    return DatasetSpec(self._element_spec, self._dataset_shape[1:])

  def _to_batched_tensor_list(self, value):
    # [FUNCTION DOC]
    # Same as converting to a tensor list, but raises a value error for a scalar dataset shape
    # (nothing to slice).
    if self._dataset_shape.ndims == 0:
      raise ValueError("Slicing dataset elements is not supported for rank 0.")
    return self._to_tensor_list(value)

  def _to_legacy_output_types(self):
    # [FUNCTION DOC]
    # Legacy hook: returns this spec itself.
    return self

  def _to_legacy_output_shapes(self):
    # [FUNCTION DOC]
    # Legacy hook: returns this spec itself.
    return self

  def _to_legacy_output_classes(self):
    # [FUNCTION DOC]
    # Legacy hook: returns this spec itself.
    return self

  def __hash__(self):
    # [FUNCTION DOC]
    # Returns a constant hash (the hash of the class) for every instance, because the element spec
    # may contain unhashable values. Equal specs therefore hash equally, but all collide.
    # TODO(b/220385675): attributes can be dicts and hence unhashable.
    return hash(DatasetSpec)

  def __eq__(self, other):
    # [FUNCTION DOC]
    # Returns true when the other is a dataset spec with equal element spec and dataset shape.
    return (isinstance(other, DatasetSpec) and
            self._element_spec == other._element_spec and
            self._dataset_shape == other._dataset_shape)


nested_structure_coder.register_codec(
    nested_structure_coder.BuiltInTypeSpecCodec(
        DatasetSpec, struct_pb2.TypeSpecProto.DATA_DATASET_SPEC
    )
)


@tf_export("data.NumpyIterator")
class NumpyIterator(tracking_base.Trackable):
  # [CLASS DOC]
  # Iterator that wraps a dataset iterator and yields elements converted to numpy values
  # (read-only arrays). Trackable, delegating save/restore of state to the wrapped iterator. Created
  # by the as-numpy-iterator method of datasets.
  """Iterator over a dataset with elements converted to numpy."""

  __slots__ = ["_iterator"]

  def __init__(self, dataset):
    # [FUNCTION DOC]
    # Creates the underlying iterator over the dataset and remembers the dataset.
    self._iterator = iter(dataset)
    self._dataset = dataset

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a text showing the wrapped iterator.
    return f"NumpyIterator(iterator={self._iterator})"

  def __iter__(self):
    # [FUNCTION DOC]
    # Returns the iterator itself.
    return self

  def __next__(self):
    # [FUNCTION DOC]
    # Returns the next element with each component converted to numpy (nested structure kept). Raises
    # the stop-iteration error when exhausted. Numpy arrays are marked read-only since they share the
    # tensor buffer; none components stay none.

    def to_numpy(x):
      # [FUNCTION DOC]
      # HELPER: converts one component to numpy using its numpy method (private one preferred),
      # returns none for none, and sets arrays as non-writable.
      if hasattr(x, "_numpy"):
        numpy = x._numpy()  # pylint: disable=protected-access
      elif x is not None:
        numpy = x.numpy()
      else:
        return None
      if isinstance(numpy, np.ndarray):
        # `numpy` shares the same underlying buffer as the `x` Tensor.
        # Tensors are expected to be immutable, so we disable writes.
        numpy.setflags(write=False)
      return numpy

    return nest.map_structure(to_numpy, next(self._iterator))

  def next(self):
    # [FUNCTION DOC]
    # Python-2-style alias; returns the next element.
    return self.__next__()

  # override
  def _serialize_to_tensors(self):
    # [FUNCTION DOC]
    # Override: delegates to the wrapped iterator and returns its serialized tensors.
    # pylint: disable=protected-access
    return self._iterator._serialize_to_tensors()

  # override
  def _restore_from_tensors(self, restored_tensors):
    # [FUNCTION DOC]
    # Override: delegates restoring from the given tensors to the wrapped iterator.
    # pylint: disable=protected-access
    return self._iterator._restore_from_tensors(restored_tensors)

  # override
  def _copy_trackable_to_cpu(self, object_map):
    # [FUNCTION DOC]
    # Override: copies this iterator's state into a mapping of copies. If this object is not in the
    # mapping yet it adds a fresh numpy iterator over the same dataset, then serializes the state and
    # restores it into the copy. Mutates the given object map.
    if self not in object_map:
      # If self is not populated in object_map yet, instantiate the copy
      object_map[self] = NumpyIterator(self._dataset)

    # Copy values from `self` to copy of `self`
    serialized = self._serialize_to_tensors()
    object_map[self]._restore_from_tensors(serialized)  # pylint: disable=protected-access

  # TODO(b/284309865): Remove once `_save` is no longer used anywhere.
  def _save(self):
    # [FUNCTION DOC]
    # Deprecated private alias for the save method.
    # pylint: disable=protected-access
    return self.save()

  def save(self):
    # [FUNCTION DOC]
    # Saves the state of the underlying iterator and returns the saved state.
    # pylint: disable=protected-access
    return self._iterator._save()

  # TODO(b/284309865): Remove once `_restore` is no longer used anywhere.
  def _restore(self, state):
    # [FUNCTION DOC]
    # Deprecated private alias for the restore method.
    return self.restore(state)

  def restore(self, state):
    # [FUNCTION DOC]
    # Restores the underlying iterator from the given saved state and returns its result.
    # pylint: disable=protected-access
    return self._iterator._restore(state)


# TODO(b/284309865): Remove once `_NumpyIterator` is no longer used anywhere.
_NumpyIterator = NumpyIterator


class _VariantTracker(resource_lib.CapturableResource):
  # [CLASS DOC]
  # HELPER: capturable resource that lets a function capturing a dataset be exported in a saved
  # model. Associates the dataset variant tensor with a function that can recreate it.
  """Allows export of functions capturing a Dataset in SavedModels.

  When saving a SavedModel, `tf.saved_model.save` traverses the object
  graph. Since Datasets reference _VariantTracker objects, that traversal will
  find a _VariantTracker for each Dataset and so know how to save and restore
  functions which reference the Dataset's variant Tensor.
  """

  def __init__(self, variant_tensor, resource_creator):
    # [FUNCTION DOC]
    # Records the variant tensor as the resource handle and the creator function as the way to
    # recreate it; initializes the base on the CPU device. Raises a type error if the creator is not
    # already a traced function.
    """Record that `variant_tensor` is associated with `resource_creator`.

    Args:
      variant_tensor: The variant-dtype Tensor associated with the Dataset. This
        Tensor will be a captured input to functions which use the Dataset, and
        is used by saving code to identify the corresponding _VariantTracker.
      resource_creator: A zero-argument function which creates a new
        variant-dtype Tensor. This function will be included in SavedModels and
        run to re-create the Dataset's variant Tensor on restore.
    """
    super(_VariantTracker, self).__init__(device="CPU")
    self._resource_handle = variant_tensor
    if not isinstance(resource_creator, def_function.Function):
      # Internal validation -- _VariantTracker assumes that resource creator is
      # already a tf.function.
      raise TypeError("Resource creator should already be a tf.function.")
    self._create_resource = resource_creator

  def _trackable_children(self,
                          save_type=tracking_base.SaveType.CHECKPOINT,
                          **kwargs):
    # [FUNCTION DOC]
    # For non-saved-model saves returns no children; otherwise returns the base children with the
    # resource-creation entry replaced by the stored creator function.
    if save_type != tracking_base.SaveType.SAVEDMODEL:
      return {}

    children = super(_VariantTracker,
                     self)._trackable_children(save_type, **kwargs)
    # Overwrite the _create_resource function, since `self._create_resource`
    # is already a tf.function.
    children["_create_resource"] = self._create_resource
    return children


# TODO(b/254291122): Remove.
# Loaded lazily due to a circular dependency (dataset_ops ->
# batch_op -> dataset_ops).
batch_op = lazy_loader.LazyLoader(
    "batch_op", globals(),
    "tensorflow.python.data.ops.batch_op")
BatchDataset = batch_op._BatchDataset  # pylint: disable=protected-access
PrefetchDataset = prefetch_op._PrefetchDataset  # pylint: disable=protected-access
ShuffleDataset = shuffle_op._ShuffleDataset  # pylint: disable=protected-access


# TODO(b/254291122): Remove.
# Loaded lazily due to a circular dependency (dataset_ops ->
# repeat_op -> dataset_ops).
repeat_op = lazy_loader.LazyLoader(
    "repeat_op", globals(),
    "tensorflow.python.data.ops.repeat_op")
RepeatDataset = repeat_op._RepeatDataset  # pylint: disable=protected-access


class _OptionsDataset(UnaryUnchangedStructureDataset):
  # [CLASS DOC]
  # HELPER: identity dataset that stores options. Used by the with-options method and debug mode.
  """An identity `Dataset` that stores options."""

  def __init__(self, input_dataset, options, name=None):
    # [FUNCTION DOC]
    # Creates an options dataset op over the input dataset carrying the serialized options. Inputs:
    # input_dataset, options object, name (optional). Afterwards merges the given options into the
    # options inherited from the input (or just uses them if none), and freezes them.
    # pylint: disable=protected-access
    self._input_dataset = input_dataset
    options_pb = dataset_options_pb2.Options()
    options_pb.CopyFrom(options._to_proto())
    self._name = name
    with ops.colocate_with(input_dataset._variant_tensor):
      variant_tensor = gen_dataset_ops.options_dataset(
          input_dataset._variant_tensor, options_pb.SerializeToString(),
          **self._common_args)
    super(_OptionsDataset, self).__init__(input_dataset, variant_tensor)

    if self._options_attr:
      self._options_attr._set_mutable(True)
      self._options_attr = self._options_attr.merge(options)
    else:
      self._options_attr = options
    self._options_attr._set_mutable(False)


def normalize_to_dense(dataset: Dataset):
  # [FUNCTION DOC]
  # Converts non-tensor components (sparse, ragged and so on) of a dataset into dense batched
  # tensor lists via a map function and then reapplies the original element spec through a
  # restructured dataset. Returns the new dataset.
  """Normalizes non-tensor components in a dataset to dense representations.

  This is necessary for dataset transformations that slice along the batch
  dimension and are oblivious to non-tensors, e.g. `unbatch`, `rebatch`.

  Args:
    dataset: Dataset to normalize.

  Returns:
    A dataset whose sparse and ragged tensors have been normalized to their
    dense representations.
  """

  # NOTE(mrry): This leads to a somewhat inefficient re-encoding step for all
  # non-tensor components.
  #
  # TODO(mrry): Consider optimizing this if it turns out to be a bottleneck.
  if structured_function._should_unpack(dataset.element_spec):  # pylint: disable=protected-access

    def normalize(*args):
      # [FUNCTION DOC]
      # HELPER: map function (multiple-argument form) converting the element into a batched tensor list.
      return structure.to_batched_tensor_list(dataset.element_spec, tuple(args))
  else:
    def normalize(arg):
      # [FUNCTION DOC]
      # HELPER: map function (single-argument form) converting the element into a batched tensor list.
      return structure.to_batched_tensor_list(dataset.element_spec, arg)

  normalized_dataset = dataset.map(normalize)

  # NOTE(mrry): Our `map()` has lost information about the structure of
  # non-tensor components, so re-apply the structure of the original dataset.
  return _RestructuredDataset(normalized_dataset, dataset.element_spec)


class _RestructuredDataset(UnaryDataset):
  # [CLASS DOC]
  # HELPER: internal unary dataset that reuses its input's variant tensor but reports a different
  # element spec.
  """An internal helper for changing the element spec of a dataset."""

  def __init__(self, dataset, element_spec):
    # [FUNCTION DOC]
    # Stores the input dataset and new element spec, reuses the input's variant tensor and
    # initializes the base class.
    self._input_dataset = dataset
    self._element_spec = element_spec

    variant_tensor = self._input_dataset._variant_tensor  # pylint: disable=protected-access
    super(_RestructuredDataset, self).__init__(dataset, variant_tensor)

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the replacement element spec.
    return self._element_spec


def _get_prob_original_static(initial_dist_t, target_dist_t):
  # [FUNCTION DOC]
  # HELPER: for rejection resampling. Returns the smallest ratio of target to initial probability
  # when both distributions are statically known, else nothing.
  """Returns the static probability of sampling from the original.

  `tensor_util.constant_value(prob_of_original)` returns `None` if it encounters
  an Op that it isn't defined for. We have some custom logic to avoid this.

  Args:
    initial_dist_t: A tensor of the initial distribution.
    target_dist_t: A tensor of the target distribution.

  Returns:
    The probability of sampling from the original distribution as a constant,
    if it is a constant, or `None`.
  """
  init_static = tensor_util.constant_value(initial_dist_t)
  target_static = tensor_util.constant_value(target_dist_t)

  if init_static is None or target_static is None:
    return None
  else:
    return np.min(target_static / init_static)


def _filter_ds(dataset,
               acceptance_dist_ds,
               initial_dist_ds,
               class_func,
               seed,
               name=None) -> DatasetV2:
  # [FUNCTION DOC]
  # HELPER: for rejection resampling. Zips the acceptance distribution with the initial one
  # (printing a warning, limited times, when over half the examples would be rejected), pairs each
  # element with its class and acceptance probability, keeps elements by comparing a random uniform
  # draw (with the seed) against the probability, and returns a dataset of class and element pairs.
  """Filters a dataset based on per-class acceptance probabilities.

  Args:
    dataset: The dataset to be filtered.
    acceptance_dist_ds: A dataset of acceptance probabilities.
    initial_dist_ds: A dataset of the initial probability distribution, given or
      estimated.
    class_func: A function mapping an element of the input dataset to a scalar
      `tf.int32` tensor. Values should be in `[0, num_classes)`.
    seed: (Optional.) Python integer seed for the resampler.
    name: (Optional.) A name for the tf.data operation.

  Returns:
    A dataset of (class value, data) after filtering.
  """

  def maybe_warn_on_large_rejection(accept_dist, initial_dist):
    # [FUNCTION DOC]
    # HELPER: computes the proportion of rejected examples; returns the acceptance distribution
    # unchanged if it is under one half, else the same values passed through a print op that logs a
    # warning (first ten times).
    proportion_rejected = math_ops.reduce_sum((1 - accept_dist) * initial_dist)
    return cond.cond(
        math_ops.less(proportion_rejected, .5),
        lambda: accept_dist,
        lambda: logging_ops.Print(  # pylint: disable=g-long-lambda
            accept_dist, [proportion_rejected, initial_dist, accept_dist],
            message="Proportion of examples rejected by sampler is high: ",
            summarize=100,
            first_n=10))

  acceptance_dist_ds = (
      DatasetV2.zip((acceptance_dist_ds, initial_dist_ds),
                    name=name).map(maybe_warn_on_large_rejection, name=name))

  def _gather_and_copy(acceptance_prob, data):
    # [FUNCTION DOC]
    # HELPER: map function returning the class value of the element, the acceptance probability for
    # that class, and the element itself.
    if isinstance(data, tuple):
      class_val = class_func(*data)
    else:
      class_val = class_func(data)
    return class_val, array_ops.gather(acceptance_prob, class_val), data

  current_probabilities_and_class_and_data_ds = DatasetV2.zip(
      (acceptance_dist_ds, dataset), name=name).map(
          _gather_and_copy, name=name)

  def _reject(unused_class_val, p, unused_data):
    # [FUNCTION DOC]
    # HELPER: filter predicate returning true when a uniform random number is below the probability.
    return random_ops.random_uniform([], seed=seed, dtype=p.dtype) < p

  filtered_ds = current_probabilities_and_class_and_data_ds.filter(
      _reject, name=name)
  return filtered_ds.map(
      lambda class_value, _, data: (class_value, data), name=name)


# pylint: disable=missing-function-docstring
def _estimate_initial_dist_ds(target_dist_t,
                              class_values_ds,
                              dist_estimation_batch_size=32,
                              smoothing_constant=10,
                              name=None):
  # [FUNCTION DOC]
  # HELPER: for rejection resampling with unknown initial distribution. Batches the class values,
  # scans over batches with running per-class counts (initialized with a smoothing constant),
  # tiles the current estimated distribution to the batch size and unbatches, giving a dataset of
  # the distribution estimate per element. Inputs: target distribution, class value dataset,
  # estimation batch size (default 32), smoothing constant (default 10), name.
  num_classes = (target_dist_t.shape[0] or array_ops.shape(target_dist_t)[0])
  initial_examples_per_class_seen = array_ops.fill([num_classes],
                                                   np.int64(smoothing_constant))

  def update_estimate_and_tile(num_examples_per_class_seen, c):
    # [FUNCTION DOC]
    # HELPER: scan function updating class counts with a batch of classes and returning the new
    # counts and the estimated distribution repeated for every batch row.
    updated_examples_per_class_seen, dist = _estimate_data_distribution(
        c, num_examples_per_class_seen)
    tiled_dist = array_ops.tile(
        array_ops.expand_dims(dist, 0), [dist_estimation_batch_size, 1])
    return updated_examples_per_class_seen, tiled_dist

  initial_dist_ds = (
      class_values_ds.batch(dist_estimation_batch_size, name=name).scan(
          initial_examples_per_class_seen, update_estimate_and_tile,
          name=name).unbatch(name=name))

  return initial_dist_ds


def _get_target_to_initial_ratio(initial_probs, target_probs):
  # [FUNCTION DOC]
  # HELPER: returns target probabilities divided by initial probabilities, adding the smallest
  # representable positive number to the denominator to avoid division by zero.
  # Add tiny to initial_probs to avoid divide by zero.
  denom = (initial_probs + np.finfo(initial_probs.dtype.as_numpy_dtype).tiny)
  return target_probs / denom


def _estimate_data_distribution(c, num_examples_per_class_seen):
  # [FUNCTION DOC]
  # HELPER: adds the one-hot counts of a batch of class labels to the running per-class counts and
  # returns the new counts and the 32-bit float probability estimate (counts over their total).
  """Estimate data distribution as labels are seen.

  Args:
    c: The class labels.  Type `int32`, shape `[batch_size]`.
    num_examples_per_class_seen: Type `int64`, shape `[num_classes]`, containing
      counts.

  Returns:
    num_examples_per_lass_seen: Updated counts.  Type `int64`, shape
      `[num_classes]`.
    dist: The updated distribution.  Type `float32`, shape `[num_classes]`.
  """
  num_classes = num_examples_per_class_seen.get_shape()[0]
  # Update the class-count based on what labels are seen in batch.
  num_examples_per_class_seen = math_ops.add(
      num_examples_per_class_seen,
      math_ops.reduce_sum(
          array_ops.one_hot(c, num_classes, dtype=dtypes.int64), 0))
  init_prob_estimate = math_ops.truediv(
      num_examples_per_class_seen,
      math_ops.reduce_sum(num_examples_per_class_seen))
  dist = math_ops.cast(init_prob_estimate, dtypes.float32)
  return num_examples_per_class_seen, dist


def _calculate_acceptance_probs_with_mixing(initial_probs, target_probs):
  # [FUNCTION DOC]
  # HELPER: computes, from the initial and target class distributions, per-class acceptance
  # probabilities and the mixing probability of sampling from the original distribution. Uses the
  # ratio target over initial; the mixing value is the smallest ratio and the acceptance is the
  # ratio shifted by it and normalized by the range. Returns both. (Division by zero if all ratios
  # are equal.)
  """Calculates the acceptance probabilities and mixing ratio.

  In this case, we assume that we can *either* sample from the original data
  distribution with probability `m`, or sample from a reshaped distribution
  that comes from rejection sampling on the original distribution. This
  rejection sampling is done on a per-class basis, with `a_i` representing the
  probability of accepting data from class `i`.

  This method is based on solving the following analysis for the reshaped
  distribution:

  Let F be the probability of a rejection (on any example).
  Let p_i be the proportion of examples in the data in class i (init_probs)
  Let a_i is the rate the rejection sampler should *accept* class i
  Let t_i is the target proportion in the minibatches for class i (target_probs)

  ```
  F = sum_i(p_i * (1-a_i))
    = 1 - sum_i(p_i * a_i)     using sum_i(p_i) = 1
  ```

  An example with class `i` will be accepted if `k` rejections occur, then an
  example with class `i` is seen by the rejector, and it is accepted. This can
  be written as follows:

  ```
  t_i = sum_k=0^inf(F^k * p_i * a_i)
      = p_i * a_j / (1 - F)    using geometric series identity, since 0 <= F < 1
      = p_i * a_i / sum_j(p_j * a_j)        using F from above
  ```

  Note that the following constraints hold:
  ```
  0 <= p_i <= 1, sum_i(p_i) = 1
  0 <= a_i <= 1
  0 <= t_i <= 1, sum_i(t_i) = 1
  ```

  A solution for a_i in terms of the other variables is the following:
    ```a_i = (t_i / p_i) / max_i[t_i / p_i]```

  If we try to minimize the amount of data rejected, we get the following:

  M_max = max_i [ t_i / p_i ]
  M_min = min_i [ t_i / p_i ]

  The desired probability of accepting data if it comes from class `i`:

  a_i = (t_i/p_i - m) / (M_max - m)

  The desired probability of pulling a data element from the original dataset,
  rather than the filtered one:

  m = M_min

  Args:
    initial_probs: A Tensor of the initial probability distribution, given or
      estimated.
    target_probs: A Tensor of the corresponding classes.

  Returns:
    (A 1D Tensor with the per-class acceptance probabilities, the desired
    probability of pull from the original distribution.)
  """
  ratio_l = _get_target_to_initial_ratio(initial_probs, target_probs)
  max_ratio = math_ops.reduce_max(ratio_l)
  min_ratio = math_ops.reduce_min(ratio_l)

  # Target prob to sample from original distribution.
  m = min_ratio

  # TODO(joelshor): Simplify fraction, if possible.
  a_i = (ratio_l - m) / (max_ratio - m)
  return a_i, m


def apply_rewrite(dataset, rewrite):
  # [FUNCTION DOC]
  # Applies a named graph rewrite to a dataset using the rewrite dataset op and returns a
  # variant-backed dataset with the same element spec.
  """Applies a rewrite to a dataset.

  Args:
    dataset: A dataset object.
    rewrite: Name of the rewrite. Currently supports "replicate_on_split" that
      distributes small datasets across workers when using the tf.data service
      dynamic sharding mode and combining multiple datasets (for instance,
      with `zip`, `choose_from_datasets`, etc).

  Returns:
    The rewritten dataset.
  """
  # pylint: disable=protected-access
  return _VariantDataset(
      gen_dataset_ops.rewrite_dataset(dataset._variant_tensor, rewrite,
                                      **dataset._flat_structure),
      dataset.element_spec)


def _collect_resource_inputs(op):
  # [FUNCTION DOC]
  # HELPER: for the automatic control dependency resolver. Walks from an op through its variant
  # (dataset) inputs and collects resource inputs. Returns lists of read and written resources;
  # currently all are reported as writes.
  """Collects resource inputs for the given ops (and its variant inputs)."""

  def _process(op_queue, seen_ops):
    # [FUNCTION DOC]
    # HELPER: pops the next op from the queue (skipping seen ones), records it as seen, gets its
    # resource reads and writes, queues its variant-typed input ops, and returns the reads and writes.
    """Processes the next element of the op queue.

    Args:
      op_queue: Queue of Dataset operations to process.
      seen_ops: Already processed set of Operations.

    Returns:
      A 2-tuple containing sets of resource handles. The first tuple entry
      contains read-only handles and the second entry contains read-write
      handles.
    """

    reads = []
    writes = []
    op = op_queue.pop()
    if op in seen_ops:
      return reads, writes
    seen_ops.add(op)
    # TODO(b/150139257): All resource inputs are in writes right now since we
    # have not updated the functional ops to set the special attribute that ACD
    # uses to figure out which of the op's inputs are read-only.
    reads, writes = acd_utils.get_read_write_resource_inputs(op)
    # Conservatively assume that any variant inputs are datasets.
    op_queue.extend(t.op for t in op.inputs if t.dtype == dtypes.variant)
    return reads, writes

  op_queue = [op]
  seen_ops = set()
  all_reads = []
  all_writes = []
  while op_queue:
    reads, writes = _process(op_queue, seen_ops)
    all_reads.extend(reads)
    all_writes.extend(writes)

  return all_reads, all_writes


@auto_control_deps.register_acd_resource_resolver
def _resource_resolver(op, resource_reads, resource_writes):
  # [FUNCTION DOC]
  # Registered automatic-control-dependencies hook. For dataset-consuming ops (single element,
  # to TFRecord, reduce) it adds the resources used by their dataset graph to the read and write
  # sets; for iterator get-next ops it does the same through the unique make-iterator op consuming
  # the iterator resource. Mutates the two sets and returns true if any resource was added.
  """Updates resource inputs for tf.data ops with indirect dependencies."""

  updated = False
  if op.type in [
      "DatasetToSingleElement", "DatasetToTFRecord", "ReduceDataset"
  ]:
    reads, writes = _collect_resource_inputs(op)
    for inp in reads:
      if inp not in resource_reads:
        updated = True
        resource_reads.add(inp)
    for inp in writes:
      if inp not in resource_writes:
        updated = True
        resource_writes.add(inp)

  if op.type in [
      "IteratorGetNext", "IteratorGetNextSync", "IteratorGetNextAsOptional"
  ]:
    iterator_resource = op.inputs[0]
    make_iterator_ops = [
        op for op in iterator_resource.consumers() if op.type == "MakeIterator"
    ]

    if len(make_iterator_ops) == 1:
      reads, writes = _collect_resource_inputs(make_iterator_ops[0])
      for inp in reads:
        if inp not in resource_reads:
          updated = True
          resource_reads.add(inp)
      for inp in writes:
        if inp not in resource_writes:
          updated = True
          resource_writes.add(inp)

  return updated


dataset_autograph.register_overrides()
