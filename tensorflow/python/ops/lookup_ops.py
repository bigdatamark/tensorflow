# [MODULE DOC]
# Python front end for TensorFlow lookup tables (the tf.lookup namespace and the older v1 table
# helpers). It defines the abstract LookupInterface and the initializable table base class, the
# immutable StaticHashTable (plus its v1 variants and the legacy HashTable alias), table
# initializers (from key and value tensors, or from text files with selectable key and value
# columns), hasher specifications for out-of-vocabulary bucketing, the wrapper tables that map
# unknown keys to hash buckets (IdTableWithHashBuckets, StaticVocabularyTable), the mutable
# mutable table classes (MutableHashTable and DenseHashTable) and
# convenience constructors such as index_table_from_file and index_to_string_table_from_tensor.
# Tables wrap resource handles created by the generated lookup kernels and are trackable, so they
# can be saved in checkpoints and SavedModels.
# Import-time side effects: imports the lookup gradient registrations, re-exports every symbol of
# the generated lookup op module through a wildcard import, and registers exports with the public
# API decorators and a SavedModel serialization registration for StaticHashTable.
# Not a helper module; it is a public API module.
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
"""Lookup operations."""
# pylint: disable=g-bad-name
import collections
import functools
import uuid

from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_lookup_ops
# Ensure lookup gradients are registered
from tensorflow.python.ops import lookup_grad  # pylint: disable=unused-import
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_lookup_ops import *
from tensorflow.python.saved_model import registration
from tensorflow.python.trackable import asset
# pylint: enable=wildcard-import
from tensorflow.python.trackable import base as trackable_base
from tensorflow.python.trackable import resource
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.types import internal
from tensorflow.python.util import compat as compat_util
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["initialize_all_tables"])
@deprecated(None, "Use `tf.tables_initializer` instead.")
def initialize_all_tables(name="init_all_tables"):
  # [FUNCTION DOC]
  # Deprecated v1 alias of the tables initializer function. Takes an optional operation name
  # (default "init_all_tables") and returns whatever the tables initializer returns: a grouped
  # operation initializing every table registered in the default graph, or a no-op if none.
  # Side effects: only the deprecation warning emitted by the decorator, plus graph op creation.
  """Returns an Op that initializes all tables of the default graph.

  Args:
    name: Optional name for the initialization op.

  Returns:
    An Op that initializes all tables.  Note that if there are
    not tables the returned Op is a NoOp.
  """
  return tables_initializer(name)


@tf_export(v1=["initializers.tables_initializer", "tables_initializer"])
def tables_initializer(name="init_all_tables"):
  # [FUNCTION DOC]
  # Collects every op stored in the default graph's table-initializer collection. If there are
  # any, returns a single grouped operation (named by the optional name argument, default
  # "init_all_tables") that runs them all; otherwise returns a no-op with that name.
  # Reads graph collection state only; creates one graph op; raises nothing itself.
  """Returns an Op that initializes all tables of the default graph.

  Args:
    name: Optional name for the initialization op.

  Returns:
    An Op that initializes all tables.  Note that if there are
    not tables the returned Op is a NoOp.

  @compatibility(TF2)
  `tf.compat.v1.tables_initializer` is no longer needed with eager execution and
  `tf.function`. In TF2, when creating an initializable table like a
  `tf.lookup.StaticHashTable`, the table will automatically be initialized on
  creation.

  #### Before & After Usage Example

  Before:

  >>> with tf.compat.v1.Session():
  ...   init = tf.compat.v1.lookup.KeyValueTensorInitializer(['a', 'b'], [1, 2])
  ...   table = tf.compat.v1.lookup.StaticHashTable(init, default_value=-1)
  ...   tf.compat.v1.tables_initializer().run()
  ...   result = table.lookup(tf.constant(['a', 'c'])).eval()
  >>> result
  array([ 1, -1], dtype=int32)

  After:

  >>> init = tf.lookup.KeyValueTensorInitializer(['a', 'b'], [1, 2])
  >>> table = tf.lookup.StaticHashTable(init, default_value=-1)
  >>> table.lookup(tf.constant(['a', 'c'])).numpy()
  array([ 1, -1], dtype=int32)

  @end_compatibility
  """
  initializers = ops.get_collection(ops.GraphKeys.TABLE_INITIALIZERS)
  if initializers:
    return control_flow_ops.group(*initializers, name=name)
  return control_flow_ops.no_op(name=name)


def check_table_dtypes(table, key_dtype, value_dtype):
  # [FUNCTION DOC]
  # HELPER: validation helper used by table initializers before importing data into a table.
  # Compares the base dtype of key_dtype and value_dtype against the table's own key and value
  # dtypes. Returns nothing. Raises a type error naming expected and received dtypes if the key
  # dtype mismatches, or if the value dtype mismatches (key is checked first). No side effects.
  """Check that the given key_dtype and value_dtype matches the table dtypes.

  Args:
    table: The table to check types against to.
    key_dtype: The key data type to check.
    value_dtype: The value data type to check.

  Raises:
    TypeError: when 'key_dtype' or 'value_dtype' doesn't match the table data
      types.
  """
  if key_dtype.base_dtype != table.key_dtype:
    raise TypeError(f"Invalid key dtype for table, expected {table.key_dtype} "
                    f"but got {key_dtype}.")
  if value_dtype.base_dtype != table.value_dtype:
    raise TypeError("Invalid value dtype for table, expected "
                    f"{table.value_dtype} but got {value_dtype}.")


class LookupInterface(resource.TrackableResource):
  # [CLASS DOC]
  # Abstract base for all lookup tables. A trackable resource that records the key and value
  # dtypes and declares the table API (name, size, lookup, subscript access). Concrete subclasses
  # supply the resource creation and the operations. Instances are not created directly.
  """Represent a lookup table that persists across different steps."""

  def __init__(self, key_dtype, value_dtype):
    # [FUNCTION DOC]
    # Stores the key and value dtypes (each converted to a canonical dtype object from anything
    # dtype-like) and then runs the trackable-resource base constructor. Both arguments required.
    # Raises a type error if either is not convertible to a dtype. Returns nothing.
    """Construct a lookup table interface.

    Args:
      key_dtype: The table key type.
      value_dtype: The table value type.
    """
    self._key_dtype = dtypes.as_dtype(key_dtype)
    self._value_dtype = dtypes.as_dtype(value_dtype)
    super(LookupInterface, self).__init__()

  def _create_resource(self):
    # [FUNCTION DOC]
    # HELPER: abstract hook for subclasses that create the underlying table resource handle.
    # Always raises a not-implemented error in this base class.
    raise NotImplementedError

  @property
  def key_dtype(self):
    # [FUNCTION DOC]
    # Read-only property returning the table's key dtype as a dtype object.
    """The table key dtype."""
    return self._key_dtype

  @property
  def value_dtype(self):
    # [FUNCTION DOC]
    # Read-only property returning the table's value dtype as a dtype object.
    """The table value dtype."""
    return self._value_dtype

  @property
  def name(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the code returns the NotImplementedError class instead of raising it, so
    # accessing the property on the base class silently yields an exception type, not a name.
    # Subclasses override this property to return the real table name.
    """The name of the table."""
    return NotImplementedError

  def size(self, name=None):
    # [FUNCTION DOC]
    # Abstract method: would return the number of entries in the table. Takes an optional op
    # name. Always raises a not-implemented error here; subclasses override it.
    """Compute the number of elements in this table."""
    raise NotImplementedError

  def lookup(self, keys, name=None):
    # [FUNCTION DOC]
    # Abstract method: would return the values for the given keys. Takes the keys and an
    # optional op name. Always raises a not-implemented error here; subclasses override it.
    """Looks up `keys` in a table, outputs the corresponding values."""
    raise NotImplementedError

  def __getitem__(self, keys):
    # [FUNCTION DOC]
    # Subscript syntax sugar: calls the lookup method with the given keys (no op name) and
    # returns its result. Raises whatever lookup raises.
    """Looks up `keys` in a table, outputs the corresponding values."""
    return self.lookup(keys)


class InitializableLookupTableBase(LookupInterface):
  # [CLASS DOC]
  # Base class for read-only lookup tables that are filled once by a table initializer. Holds the
  # initializer, a scalar default tensor returned for missing keys, the resource handle and the
  # initialization op. Provides size and lookup (supporting dense, sparse and ragged keys) on top
  # of the generated lookup kernels. Subclasses must implement resource creation.
  """Initializable lookup table interface.

  An initializable lookup tables persist across different steps.
  """

  def __init__(self, default_value, initializer):
    # [FUNCTION DOC]
    # Builds the table: takes the default value (anything convertible to a scalar tensor of the
    # initializer's value dtype) and the initializer (supplies key and value dtypes and the
    # initialization op). Tracks the initializer if it is trackable, creates the resource handle
    # inside the init scope, and builds the initialization op (also inside the init scope when
    # in graph mode within a control-flow context, otherwise directly).
    # Raises a value error if the default value is not a scalar. Returns nothing.
    # Side effects: creates graph or eager ops and the table resource, and may register the
    # initialization op in the graph's table-initializer collection through the initializer.
    """Construct a table object from a table reference.

    If requires a table initializer object (subclass of `TableInitializerBase`).
    It provides the table key and value types, as well as the op to initialize
    the table. The caller is responsible to execute the initialization op.

    Args:
      default_value: The value to use if a key is missing in the table.
      initializer: The table initializer to use.
    """
    super(InitializableLookupTableBase, self).__init__(initializer.key_dtype,
                                                       initializer.value_dtype)
    self._default_value = ops.convert_to_tensor(
        default_value, dtype=self._value_dtype)
    self._default_value.get_shape().merge_with(tensor_shape.TensorShape([]))
    if isinstance(initializer, trackable_base.Trackable):
      self._initializer = self._track_trackable(initializer, "_initializer")
    with ops.init_scope():
      self._resource_handle = self._create_resource()
    if (not context.executing_eagerly() and
        ops.get_default_graph()._get_control_flow_context() is not None):  # pylint: disable=protected-access
      with ops.init_scope():
        self._init_op = self._initialize()
    else:
      self._init_op = self._initialize()

  def _initialize(self):
    # [FUNCTION DOC]
    # HELPER: asks the stored initializer to build the table's initialization op for this table
    # and returns that op (nothing else is done).
    return self._initializer.initialize(self)

  @property
  def default_value(self):
    # [FUNCTION DOC]
    # Read-only property returning the scalar tensor used for keys absent from the table.
    """The default value of the table."""
    return self._default_value

  def size(self, name=None):
    # [FUNCTION DOC]
    # Returns a scalar int64 tensor with the number of entries, using the table-size kernel on
    # the resource handle inside a name scope derived from the optional op name and table name.
    """Compute the number of elements in this table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A scalar tensor containing the number of elements in this table.
    """
    with ops.name_scope(name, "%s_Size" % self.name, [self.resource_handle]):
      return gen_lookup_ops.lookup_table_size_v2(self.resource_handle)

  def lookup(self, keys, name=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says a type error is raised when default_value mismatches the
    # table types, but only the keys dtype is checked here (default value was already converted
    # to the value dtype at construction).
    # Looks up keys (dense tensor, sparse tensor or ragged tensor; the flat values are used for
    # sparse and ragged) and returns the stored values, or the default value for missing keys.
    # The result has the same structure as the input: sparse in gives sparse (same indices and
    # dense shape), ragged in gives ragged (same row partition), dense in gives dense with the
    # keys' static shape. Optional name sets the op name scope.
    # Raises a type error if the keys dtype differs from the table key dtype.
    """Looks up `keys` in a table, outputs the corresponding values.

    The `default_value` is used for keys not present in the table.

    Args:
      keys: Keys to look up. May be either a `SparseTensor` or dense `Tensor`.
      name: A name for the operation (optional).

    Returns:
      A `SparseTensor` if keys are sparse, a `RaggedTensor` if keys are ragged,
      otherwise a dense `Tensor`.

    Raises:
      TypeError: when `keys` or `default_value` doesn't match the table data
        types.
    """
    key_tensor = keys
    # TODO(b/296302236): Remove RaggedTensor check by adding ragged
    # dispatching.
    if isinstance(keys, (sparse_tensor.SparseTensor, internal.RaggedTensor)):
      key_tensor = keys.values

    if keys.dtype.base_dtype != self._key_dtype:
      raise TypeError(f"Dtype of argument `keys` must be {self._key_dtype}, "
                      f"received: {keys.dtype}")

    with ops.name_scope(
        name, "%s_Lookup" % self.name,
        (self.resource_handle, key_tensor, self._default_value)):
      values = gen_lookup_ops.lookup_table_find_v2(self.resource_handle,
                                                   key_tensor,
                                                   self._default_value)

    values.set_shape(key_tensor.get_shape())
    if isinstance(keys, sparse_tensor.SparseTensor):
      return sparse_tensor.SparseTensor(keys.indices, values, keys.dense_shape)
    # TODO(b/296302236): Remove RaggedTensor check by adding ragged
    # dispatching.
    elif isinstance(keys, internal.RaggedTensor):
      return keys.with_values(values)
    else:
      return values


class InitializableLookupTableBaseV1(InitializableLookupTableBase):
  # [CLASS DOC]
  # Graph-mode (v1) flavor of the initializable table base: adds the initializer property that
  # exposes the initialization op so v1 sessions can run it. No other changes.

  @property
  def initializer(self):
    # [FUNCTION DOC]
    # Read-only property returning the op that initializes the table (to be run in a v1 session).
    return self._init_op


@registration.register_tf_serializable(
    predicate=lambda obj: isinstance(obj, StaticHashTable))
@tf_export("lookup.StaticHashTable", v1=[])
class StaticHashTable(InitializableLookupTableBase):
  # [CLASS DOC]
  # Immutable hash table: keys and values come from an initializer and cannot change after
  # initialization. Holds the initializer, default value, anonymity flag, shared name used to
  # share the underlying resource between kernels, the op name and the value shape. Supports
  # export of all contents, SavedModel registration (serialized as nothing; restored as a
  # restored resource) and trackable child attachment. Created with an initializer and default.
  """A generic hash table that is immutable once initialized.

  Example usage:

  >>> keys_tensor = tf.constant(['a', 'b', 'c'])
  >>> vals_tensor = tf.constant([7, 8, 9])
  >>> input_tensor = tf.constant(['a', 'f'])
  >>> table = tf.lookup.StaticHashTable(
  ...     tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor),
  ...     default_value=-1)
  >>> table.lookup(input_tensor).numpy()
  array([ 7, -1], dtype=int32)

  Or for more pythonic code:

  >>> table[input_tensor].numpy()
  array([ 7, -1], dtype=int32)

  The result of a lookup operation has the same shape as the argument:

  >>> input_tensor = tf.constant([['a', 'b'], ['c', 'd']])
  >>> table[input_tensor].numpy()
  array([[ 7,  8],
         [ 9, -1]], dtype=int32)


  """

  def __init__(self,
               initializer,
               default_value,
               name=None,
               experimental_is_anonymous=False):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring claims a HashTable object is returned, but a constructor
    # returns nothing; and the table is initialized during construction, not left non-initialized.
    # Stores the initializer and default value, and the anonymous flag. If not anonymous, takes
    # the initializer's shared name, generating a unique one from a random UUID if it is empty,
    # so the resource can be shared across kernels. Sets the op name (default "hash_table"),
    # then runs the base constructor (which converts the default to a scalar tensor, creates the
    # resource and builds the initialization op) and records the default value's shape.
    # Inputs: initializer, default_value, optional name, optional anonymous flag (default false).
    # Raises a value error if the default value is not a scalar. Side effects: creates ops and
    # a table resource.
    """Creates a non-initialized `HashTable` object.

    Creates a table, the type of its keys and values are specified by the
    initializer.
    Before using the table you will have to initialize it. After initialization
    the table will be immutable.

    Args:
      initializer: The table initializer to use. See `HashTable` kernel for
        supported key and value types.
      default_value: The value to use if a key is missing in the table.
      name: A name for the operation (optional).
      experimental_is_anonymous: Whether to use anonymous mode for the
        table (default is False). In anonymous mode, the table
        resource can only be accessed via a resource handle. It can't
        be looked up by a name. When all resource handles pointing to
        that resource are gone, the resource will be deleted
        automatically.

    Returns:
      A `HashTable` object.
    """
    self._initializer = initializer
    self._default_value = default_value
    self._is_anonymous = experimental_is_anonymous
    if not self._is_anonymous:
      self._shared_name = self._initializer._shared_name  # pylint: disable=protected-access
      if not self._shared_name:
        # Force using a shared name so that StaticHashTable resources can be
        # shared across different kernels. If no "shared_name" is set and
        # "use_node_name_sharing" is False, then each kernel gets its own local
        # resource.
        self._shared_name = "hash_table_%s" % (str(uuid.uuid4()),)
    self._name = name or "hash_table"
    self._table_name = None
    super(StaticHashTable, self).__init__(default_value, initializer)
    self._value_shape = self._default_value.get_shape()

  def _create_resource(self):
    # [FUNCTION DOC]
    # HELPER: creates the hash table resource for this table. Uses the anonymous-table kernel
    # when anonymous, otherwise the named kernel with the shared name; both use the initializer's
    # key and value dtypes and the table op name. In graph mode remembers the last component of
    # the created op's name as the table name; in eager mode the table name is cleared.
    # Returns the resource handle.
    if self._is_anonymous:
      table_ref = gen_lookup_ops.anonymous_hash_table(
          key_dtype=self._initializer.key_dtype,
          value_dtype=self._initializer.value_dtype,
          name=self._name)
    else:
      table_ref = gen_lookup_ops.hash_table_v2(
          shared_name=self._shared_name,
          key_dtype=self._initializer.key_dtype,
          value_dtype=self._initializer.value_dtype,
          name=self._name)
    if context.executing_eagerly():
      self._table_name = None
    else:
      self._table_name = table_ref.op.name.split("/")[-1]
    return table_ref

  @property
  def name(self):
    # [FUNCTION DOC]
    # Read-only property: the table's graph op name (last path component), or nothing in eager
    # mode.
    return self._table_name

  def export(self, name=None):
    # [FUNCTION DOC]
    # Exports the whole table contents using the export kernel. Optional name for the op scope.
    # Returns a pair (keys tensor, values tensor); the values tensor's static shape is set to the
    # keys' shape extended with the per-entry value shape. No exceptions raised by this code.
    """Returns tensors of all keys and values in the table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A pair of tensors with the first tensor containing all keys and the
        second tensors containing all values in the table.
    """
    with ops.name_scope(name, "%s_Export" % self.name, [self.resource_handle]):
      exported_keys, exported_values = gen_lookup_ops.lookup_table_export_v2(
          self.resource_handle, self._key_dtype, self._value_dtype)

    exported_values.set_shape(exported_keys.get_shape().concatenate(
        self._value_shape))
    return exported_keys, exported_values

  def _serialize_to_proto(self, **unused_kwargs):
    # [FUNCTION DOC]
    # HELPER: SavedModel registration hook. Ignores all keyword arguments and returns nothing,
    # meaning no extra proto data is stored for the table.
    return None

  def _add_trackable_child(self, name, value):
    # [FUNCTION DOC]
    # HELPER: SavedModel loading hook. Sets the attribute with the given name to the given value
    # and, if the value is trackable, also registers it as a tracked dependency under that name.
    # Returns nothing.
    setattr(self, name, value)
    if isinstance(value, trackable_base.Trackable):
      self._track_trackable(value, name)  # pylint:disable=protected-access

  @classmethod
  def _deserialize_from_proto(cls, **kwargs):
    # [FUNCTION DOC]
    # HELPER: SavedModel loading hook. Defines a local restored-resource subclass whose resource
    # type is "RestoredStaticHashTable" and delegates to its deserialization with the given
    # keyword arguments, returning the restored object.

    class _RestoredStaticHashTable(resource.RestoredResource):  # pylint: disable=protected-access
      # [CLASS DOC]
      # HELPER: local restored-resource type used when loading a saved StaticHashTable; it only
      # overrides the reported resource type name.

      @classmethod
      def _resource_type(cls):
        # [FUNCTION DOC]
        # HELPER: returns the constant resource type name used by the loader.
        return "RestoredStaticHashTable"

    return _RestoredStaticHashTable._deserialize_from_proto(**kwargs)  # pylint: disable=protected-access


@tf_export(v1=["lookup.StaticHashTable"])
class StaticHashTableV1(StaticHashTable):
  # [CLASS DOC]
  # Graph-mode (v1) variant of the static hash table, exported under the v1 lookup namespace.
  # Adds the initializer property exposing the initialization op for session-based running.
  """A generic hash table that is immutable once initialized.

  When running in graph mode, you must evaluate the tensor returned by
  `tf.tables_initializer()` before evaluating the tensor returned by
  this class's `lookup()` method. Example usage in graph mode:

  ```python
  keys_tensor = tf.constant([1, 2])
  vals_tensor = tf.constant([3, 4])
  input_tensor = tf.constant([1, 5])
  table = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1)
  out = table.lookup(input_tensor)
  with tf.Session() as sess:
      sess.run(tf.tables_initializer())
      print(sess.run(out))
  ```

  Note that in graph mode if you set `experimental_is_anonymous` to
  `True`, you should only call `Session.run` once, otherwise each
  `Session.run` will create (and destroy) a new table unrelated to
  each other, leading to errors such as "Table not initialized".
  You can do so like this:

  ```python
  keys_tensor = tf.constant([1, 2])
  vals_tensor = tf.constant([3, 4])
  input_tensor = tf.constant([1, 5])
  table = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1,
      experimental_is_anonymous=True)
  with tf.control_dependencies([tf.tables_initializer()]):
    out = table.lookup(input_tensor)
  with tf.Session() as sess:
    print(sess.run(out))
  ```

  In eager mode, no special code is needed to initialize the table.
  Example usage in eager mode:

  ```python
  tf.enable_eager_execution()
  keys_tensor = tf.constant([1, 2])
  vals_tensor = tf.constant([3, 4])
  input_tensor = tf.constant([1, 5])
  table = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor), -1)
  print(table.lookup(input_tensor))
  ```
  """

  @property
  def initializer(self):
    # [FUNCTION DOC]
    # Read-only property returning the op that initializes the table (for v1 sessions).
    return self._init_op


# For backwards compatibility. This will be removed in TF 2.0.
class HashTable(StaticHashTableV1):
  # [CLASS DOC]
  # HELPER: backwards-compatibility alias class for the v1 static hash table; adds the legacy
  # init property. Not exported publicly here.

  @property
  def init(self):
    # [FUNCTION DOC]
    # HELPER: legacy accessor; returns the initializer property, i.e. the initialization op.
    return self.initializer


class TableInitializerBase(trackable_base.Trackable):
  # [CLASS DOC]
  # Abstract base for table initializers. Stores key and value dtypes, exposes them, declares
  # the initialize method that subclasses implement to build the initialization op, and
  # computes a default shared name (unique per call in eager mode, empty in graph mode).
  """Base class for lookup table initializers."""

  def __init__(self, key_dtype, value_dtype):
    # [FUNCTION DOC]
    # Records the key and value dtypes after converting each to a canonical dtype object.
    # Raises a type error if either is not dtype-like. Returns nothing.
    """Construct a table initializer object.

    Args:
      key_dtype: Type of the table keys.
      value_dtype: Type of the table values.
    """
    self._key_dtype = dtypes.as_dtype(key_dtype)
    self._value_dtype = dtypes.as_dtype(value_dtype)

  @property
  def key_dtype(self):
    # [FUNCTION DOC]
    # Read-only property: the key dtype the target table must have.
    """The expected table key dtype."""
    return self._key_dtype

  @property
  def value_dtype(self):
    # [FUNCTION DOC]
    # Read-only property: the value dtype the target table must have.
    """The expected table value dtype."""
    return self._value_dtype

  def initialize(self, table):
    # [FUNCTION DOC]
    # Abstract method: subclasses build and return the op that fills the given table. Always
    # raises a not-implemented error here.
    """Returns the table initialization op."""
    raise NotImplementedError

  @property
  def _shared_name(self):
    # [FUNCTION DOC]
    # HELPER: supplies the shared name for the table resource. Empty string in graph mode; in
    # eager mode a unique id string, so separate tables do not accidentally share a resource.
    """Returns a shared name to be used by the table."""
    shared_name = ""
    if context.executing_eagerly():
      # Ensure a unique name when eager execution is enabled to avoid spurious
      # sharing issues.
      # TODO(rohanj): Use context.anonymous_name() instead.
      shared_name += str(ops.uid())
    return shared_name


@tf_export("lookup.KeyValueTensorInitializer")
class KeyValueTensorInitializer(TableInitializerBase):
  # [CLASS DOC]
  # Initializer that fills a table from a keys tensor and a values tensor of equal leading size.
  # Holds the converted key and value tensors and a name (made unique in eager mode). The
  # table's dtypes are those of the tensors.
  """Table initializers given `keys` and `values` tensors.

  >>> keys_tensor = tf.constant(['a', 'b', 'c'])
  >>> vals_tensor = tf.constant([7, 8, 9])
  >>> input_tensor = tf.constant(['a', 'f'])
  >>> init = tf.lookup.KeyValueTensorInitializer(keys_tensor, vals_tensor)
  >>> table = tf.lookup.StaticHashTable(
  ...     init,
  ...     default_value=-1)
  >>> table.lookup(input_tensor).numpy()
  array([ 7, -1], dtype=int32)

  """

  def __init__(self, keys, values, key_dtype=None, value_dtype=None, name=None):
    # [FUNCTION DOC]
    # Converts keys and values (tensors or python arrays) to tensors, honoring the optional
    # key_dtype and value_dtype hints. In graph mode inside a control-flow context the
    # conversion runs in the init scope so the tensors live outside the loop or conditional.
    # Stores the name (default "key_value_init"; a unique id is appended in eager mode) and then
    # calls the base constructor with the resulting tensors' dtypes. Returns nothing.
    # Raises whatever tensor conversion raises for incompatible input.
    """Constructs a table initializer object based on keys and values tensors.

    Args:
      keys: The tensor for the keys.
      values: The tensor for the values.
      key_dtype: The `keys` data type. Used when `keys` is a python array.
      value_dtype: The `values` data type. Used when `values` is a python array.
      name: A name for the operation (optional).
    """
    if (not context.executing_eagerly() and
        ops.get_default_graph()._get_control_flow_context() is not None):  # pylint: disable=protected-access
      with ops.init_scope():
        self._keys = ops.convert_to_tensor(keys, dtype=key_dtype, name="keys")
        self._values = ops.convert_to_tensor(
            values, dtype=value_dtype, name="values")
    else:
      self._keys = ops.convert_to_tensor(keys, dtype=key_dtype, name="keys")
      self._values = ops.convert_to_tensor(
          values, dtype=value_dtype, name="values")
    self._name = name if name is not None else "key_value_init"
    if context.executing_eagerly():
      # Ensure a unique name when eager execution is enabled to avoid spurious
      # sharing issues.
      # TODO(rohanj): Use context.anonymous_name() instead.
      self._name += str(ops.uid())

    super(KeyValueTensorInitializer, self).__init__(self._keys.dtype,
                                                    self._values.dtype)

  def initialize(self, table):
    # [FUNCTION DOC]
    # Checks that the stored keys and values dtypes match the table, then builds the bulk-import
    # op that loads the keys and values into the table's resource inside a name scope. Adds the
    # op to the graph's table-initializer collection and returns it.
    # Raises a type error on dtype mismatch.
    """Initializes the given `table` with `keys` and `values` tensors.

    Args:
      table: The table to initialize.

    Returns:
      The operation that initializes the table.

    Raises:
      TypeError: when the keys and values data types do not match the table
      key and value data types.
    """
    check_table_dtypes(table, self._keys.dtype, self._values.dtype)
    with ops.name_scope(
        self._name, values=(table.resource_handle, self._keys, self._values)):
      init_op = gen_lookup_ops.lookup_table_import_v2(table.resource_handle,
                                                      self._keys, self._values)
    ops.add_to_collection(ops.GraphKeys.TABLE_INITIALIZERS, init_op)
    return init_op


@tf_export("lookup.TextFileIndex")
class TextFileIndex:
  # [CLASS DOC]
  # Constants namespace for text-file initializers: WHOLE_LINE (minus two) selects the entire
  # line text, LINE_NUMBER (minus one) selects the zero-based line number; non-negative values
  # select a delimited column. Never instantiated; holds only class-level constants.
  """The key and value content to get from each line.

  This class defines the key and value used for `tf.lookup.TextFileInitializer`.

  The key and value content to get from each line is specified either
  by the following, or a value `>=0`.
  * `TextFileIndex.LINE_NUMBER` means use the line number starting from zero,
    expects data type int64.
  * `TextFileIndex.WHOLE_LINE` means use the whole line content, expects data
    type string.

  A value `>=0` means use the index (starting at zero) of the split line based
      on `delimiter`.
  """
  WHOLE_LINE = -2
  LINE_NUMBER = -1


@tf_export("lookup.TextFileInitializer")
class TextFileInitializer(TableInitializerBase):
  # [CLASS DOC]
  # Initializer that fills a table from a text file, one entry per line. Key and value come
  # from a column index, the line number or the whole line. Holds the file name (tracked as an
  # asset so it is bundled on export), key/value indices, optional vocabulary size, delimiter,
  # name and a value offset. Computes a deterministic shared name from these settings.
  r"""Table initializers from a text file.

  This initializer assigns one entry in the table for each line in the file.

  The key and value type of the table to initialize is given by `key_dtype` and
  `value_dtype`.

  The key and value content to get from each line is specified by
  the `key_index` and `value_index`.

  * `TextFileIndex.LINE_NUMBER` means use the line number starting from zero,
    expects data type int64.
  * `TextFileIndex.WHOLE_LINE` means use the whole line content, expects data
    type string.
  * A value `>=0` means use the index (starting at zero) of the split line based
      on `delimiter`.

  For example if we have a file with the following content:

  >>> import tempfile
  >>> f = tempfile.NamedTemporaryFile(delete=False)
  >>> content='\n'.join(["emerson 10", "lake 20", "palmer 30",])
  >>> f.file.write(content.encode('utf-8'))
  >>> f.file.close()

  The following snippet initializes a table with the first column as keys and
  second column as values:

  * `emerson -> 10`
  * `lake -> 20`
  * `palmer -> 30`

  >>> init= tf.lookup.TextFileInitializer(
  ...    filename=f.name,
  ...    key_dtype=tf.string, key_index=0,
  ...    value_dtype=tf.int64, value_index=1,
  ...    delimiter=" ")
  >>> table = tf.lookup.StaticHashTable(init, default_value=-1)
  >>> table.lookup(tf.constant(['palmer','lake','tarkus'])).numpy()

  Similarly to initialize the whole line as keys and the line number as values.

  * `emerson 10 -> 0`
  * `lake 20 -> 1`
  * `palmer 30 -> 2`

  >>> init = tf.lookup.TextFileInitializer(
  ...   filename=f.name,
  ...   key_dtype=tf.string, key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
  ...   value_dtype=tf.int64, value_index=tf.lookup.TextFileIndex.LINE_NUMBER)
  >>> table = tf.lookup.StaticHashTable(init, -1)
  >>> print(table.lookup(tf.constant('palmer 30')).numpy())
  2
  """

  def __init__(self,
               filename,
               key_dtype,
               key_index,
               value_dtype,
               value_index,
               vocab_size=None,
               delimiter="\t",
               name=None,
               value_index_offset=0):
    # [FUNCTION DOC]
    # Validates and stores the settings. Inputs: filename (string or scalar string tensor, must
    # be non-empty unless a tensor), key and value dtypes, key_index and value_index (an index
    # of at least minus two: whole line, line number, or column), optional vocab_size (must be
    # positive if given), delimiter (default tab), op name, and value_index_offset (added to
    # extracted values, default zero). Wraps the file name as a tracked asset.
    # Raises a value error for an empty file name, indices below minus two, a line-number index
    # with a non-int64 dtype, a whole-line index with a dtype that is neither integer nor
    # string, or a non-positive vocab size. Returns nothing.
    """Constructs a table initializer object to populate from a text file.

    It generates one key-value pair per line. The type of table key and
    value are specified by `key_dtype` and `value_dtype`, respectively.
    Similarly the content of the key and value are specified by the key_index
    and value_index.

    - TextFileIndex.LINE_NUMBER means use the line number starting from zero,
      expects data type int64.
    - TextFileIndex.WHOLE_LINE means use the whole line content, expects data
      type string or int64.
    - A value >=0 means use the index (starting at zero) of the split line based
      on `delimiter`.

    Args:
      filename: The filename of the text file to be used for initialization. The
        path must be accessible from wherever the graph is initialized (eg.
        trainer or eval workers). The filename may be a scalar `Tensor`.
      key_dtype: The `key` data type.
      key_index: the index that represents information of a line to get the
        table 'key' values from.
      value_dtype: The `value` data type.
      value_index: the index that represents information of a line to get the
        table 'value' values from.'
      vocab_size: The number of elements in the file, if known.
      delimiter: The delimiter to separate fields in a line.
      name: A name for the operation (optional).
      value_index_offset: A number to add to all indices extracted from the file
        This is useful for cases where a user would like to reserve one or more
        low index values for control characters. For instance, if you would
        like to ensure that no vocabulary item is mapped to index 0 (so you can
        reserve 0 for a masking value), you can set value_index_offset to 1;
        this will mean that the first vocabulary element is mapped to 1
        instead of 0.

    Raises:
      ValueError: when the filename is empty, or when the table key and value
      data types do not match the expected data types.
    """
    if not isinstance(filename, tensor_lib.Tensor) and not filename:
      raise ValueError("`filename` argument required for tf.lookup.TextFileInitializer")

    self._filename_arg = filename
    key_dtype = dtypes.as_dtype(key_dtype)
    value_dtype = dtypes.as_dtype(value_dtype)

    if key_index < -2:
      raise ValueError(f"`key_index` should be >= -2, received: {key_index}.")

    if key_index == TextFileIndex.LINE_NUMBER and key_dtype != dtypes.int64:
      raise ValueError("`key_dtype` must be int64 if `key_index` is "
                       f"{TextFileIndex.LINE_NUMBER}, received: {key_dtype}")
    if ((key_index == TextFileIndex.WHOLE_LINE) and
        (not key_dtype.is_integer) and (key_dtype != dtypes.string)):
      raise ValueError(
          "`key_dtype` should be either integer or string for `key_index` "
          f"{TextFileIndex.WHOLE_LINE}, received: {key_dtype}")
    if value_index < -2:
      raise ValueError("`value_index` should be >= -2, received: "
                       f"{value_index}")

    if value_index == TextFileIndex.LINE_NUMBER and value_dtype != dtypes.int64:
      raise ValueError("`value_dtype` must be int64 for `value_index` "
                       f"{TextFileIndex.LINE_NUMBER}, received: {value_dtype}")
    if ((value_index == TextFileIndex.WHOLE_LINE) and
        (not value_dtype.is_integer) and (value_dtype != dtypes.string)):
      raise ValueError(
          "`value_dtype` should be either integer or string for `value_index` "
          f"{TextFileIndex.WHOLE_LINE}, received: {value_dtype}")

    if (vocab_size is not None) and (vocab_size <= 0):
      raise ValueError(f"`vocab_size` should be > 0, received: {vocab_size}")

    self._key_index = key_index
    self._value_index = value_index
    self._vocab_size = vocab_size
    self._delimiter = delimiter
    self._name = name
    self._filename = self._track_trackable(
        asset.Asset(filename), "_filename")
    self._offset = value_index_offset

    super(TextFileInitializer, self).__init__(key_dtype, value_dtype)

  def initialize(self, table):
    # [FUNCTION DOC]
    # Checks the table dtypes against this initializer, converts the tracked file name to a
    # string tensor, and builds the op that loads the table from the text file with the stored
    # key index, value index, vocab size (minus one if unknown), delimiter and offset.
    # Adds the op to the table-initializer collection and, in graph mode with a constant file
    # name, adds the file name to the asset-file-path collection. Returns the op.
    # Raises a type error on dtype mismatch.
    """Initializes the table from a text file.

    Args:
      table: The table to be initialized.

    Returns:
      The operation that initializes the table.

    Raises:
      TypeError: when the keys and values data types do not match the table
      key and value data types.
    """
    check_table_dtypes(table, self.key_dtype, self.value_dtype)
    with ops.name_scope(self._name, "text_file_init", (table.resource_handle,)):
      filename = ops.convert_to_tensor(
          self._filename, dtypes.string, name="asset_filepath")
      init_op = gen_lookup_ops.initialize_table_from_text_file_v2(
          table.resource_handle, filename, self._key_index, self._value_index,
          -1 if self._vocab_size is None else self._vocab_size, self._delimiter,
          self._offset)
    ops.add_to_collection(ops.GraphKeys.TABLE_INITIALIZERS, init_op)
    # If the filename tensor is anything other than a string constant (e.g.,
    # if it is a placeholder) then it does not make sense to track it as an
    # asset.
    if not context.executing_eagerly() and constant_op.is_constant(filename):
      ops.add_to_collection(ops.GraphKeys.ASSET_FILEPATHS, filename)
    return init_op

  @property
  def _shared_name(self):
    # [FUNCTION DOC]
    # HELPER: shared-name provider used when the table resource is created.
    # KNOWN ISSUE: overrides the base shared-name property but never makes it unique in eager
    # mode; identical settings always share one resource.
    # Builds a deterministic shared name from the original file name argument, the
    # vocab size (only when set), key index, value index and offset (only when non-zero), so
    # tables built from identical settings share a resource. Returns the name string.
    if self._vocab_size:
      # Keep the shared_name:
      # <table_type>_<filename>_<vocab_size>_<key_index>_<value_index>_<offset>
      if self._offset:
        shared_name = "hash_table_%s_%d_%s_%s_%s" % (
            self._filename_arg, self._vocab_size, self._key_index,
            self._value_index, self._offset)
      else:
        shared_name = "hash_table_%s_%d_%s_%s" % (
            self._filename_arg, self._vocab_size, self._key_index,
            self._value_index)
    else:
      # Keep the shared_name
      # <table_type>_<filename>_<key_index>_<value_index>_<offset>
      if self._offset:
        shared_name = "hash_table_%s_%s_%s_%s" % (
            self._filename_arg, self._key_index, self._value_index,
            self._offset)
      else:
        shared_name = "hash_table_%s_%s_%s" % (
            self._filename_arg, self._key_index, self._value_index)

    return shared_name


class TextFileStringTableInitializer(TextFileInitializer):
  # [CLASS DOC]
  # Convenience text-file initializer for id-to-string tables: keys are int64 (default: line
  # number) and values are strings (default: whole line).
  """Table initializer for `int64` IDs to string tables from a text file."""

  def __init__(self,
               filename,
               key_column_index=TextFileIndex.LINE_NUMBER,
               value_column_index=TextFileIndex.WHOLE_LINE,
               vocab_size=None,
               delimiter="\t",
               name="text_file_string_table_init"):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says a type error is raised for an empty file name or wrong
    # dtypes, but the parent constructor actually raises value errors.
    # Delegates to the text-file initializer with key dtype int64 and value dtype string, passing
    # file name, key column index (default line number), value column index (default whole line),
    # optional vocab size, delimiter (default tab) and op name. Returns nothing.
    """Constructs an initializer for an id-to-string table from a text file.

    It populates a table that its key and value types are int64 and string,
    respectively. It generates one key-value pair per line.
    The content of the key and value are specified by `key_column_index`
    and `value_column_index`.

    - TextFileIndex.LINE_NUMBER means use the line number starting from zero,
      expects data type int64.
    - TextFileIndex.WHOLE_LINE means use the whole line content, expects data
      type string or int64.
    - A value >=0 means use the index (starting at zero) of the split line based
      on `delimiter`.

    Args:
      filename: The filename of the text file to be used for initialization. The
        path must be accessible from wherever the graph is initialized (eg.
        trainer or eval workers). The filename may be a scalar `Tensor`.
      key_column_index: The column index from the text file to get the keys
        from. The default is to use the line number, starting from zero.
      value_column_index: The column index from the text file to get the values
        from. The default is to use the whole line content.
      vocab_size: The number of elements in the file, if known.
      delimiter: The delimiter to separate fields in a line.
      name: Optional name for the op.

    Raises:
      TypeError: when the filename is empty, or when the table key and value
      data types do not match the expected data types.
    """
    super(TextFileStringTableInitializer, self).__init__(
        filename,
        dtypes.int64,
        key_column_index,
        dtypes.string,
        value_column_index,
        vocab_size=vocab_size,
        delimiter=delimiter,
        name=name)


class TextFileIdTableInitializer(TextFileInitializer):
  # [CLASS DOC]
  # Convenience text-file initializer for string-to-id tables: keys default to the whole line
  # (dtype configurable, string by default) and values are int64 line numbers by default.
  """Table initializer for string to `int64` IDs tables from a text file."""

  def __init__(self,
               filename,
               key_column_index=TextFileIndex.WHOLE_LINE,
               value_column_index=TextFileIndex.LINE_NUMBER,
               vocab_size=None,
               delimiter="\t",
               name="text_file_id_table_init",
               key_dtype=dtypes.string):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says a type error is raised for an empty file name or wrong
    # dtypes, but the parent constructor actually raises value errors.
    # Delegates to the text-file initializer with the given key dtype (default string) and value
    # dtype int64. Inputs: file name, key column index (default whole line), value column index
    # (default line number), optional vocab size, delimiter (default tab), op name. Returns
    # nothing.
    """Constructs an initializer for an string-to-id table from a text file.

    It populates a table that its key and value types are string and int64,
    respectively. It generates one key-value pair per line.
    The content of the key and value are specified by the key_index
    and value_index.

    - TextFileIndex.LINE_NUMBER means use the line number starting from zero,
      expects data type int64.
    - TextFileIndex.WHOLE_LINE means use the whole line content, expects data
      type string.
    - A value >=0 means use the index (starting at zero) of the split line based
      on `delimiter`.

    Args:
      filename: The filename of the text file to be used for initialization. The
        path must be accessible from wherever the graph is initialized (eg.
        trainer or eval workers). The filename may be a scalar `Tensor`.
      key_column_index: The column index from the text file to get the `key`
        values from. The default is to use the whole line content.
      value_column_index: The column index from the text file to get the `value`
        values from. The default is to use the line number, starting from zero.
      vocab_size: The number of elements in the file, if known.
      delimiter: The delimiter to separate fields in a line.
      name: Optional name for the op.
      key_dtype: The `key` data type.

    Raises:
      TypeError: when the filename is empty, or when the table key and value
      data types do not match the expected data types.
    """
    super(TextFileIdTableInitializer, self).__init__(
        filename,
        key_dtype,
        key_column_index,
        dtypes.int64,
        value_column_index,
        vocab_size=vocab_size,
        delimiter=delimiter,
        name=name)


class HasherSpec(collections.namedtuple("HasherSpec", ["hasher", "key"])):
  # [CLASS DOC]
  # Immutable pair describing the hash function for out-of-vocabulary buckets: the hasher name
  # ("fasthash", "stronghash" or "legacy") and an optional key (used only by the strong hash).
  # Declares no extra slots.
  """A structure for the spec of the hashing function to use for hash buckets.

  `hasher` is the name of the hashing function to use (eg. "fasthash",
  "stronghash").
  `key` is optional and specify the key to use for the hash function if
  supported, currently only used by a strong hash.

  Fields:
    hasher: The hasher name to use.
    key: The key to be used by the hashing function, if required.
  """
  __slots__ = ()


FastHashSpec = HasherSpec("fasthash", None)  # pylint: disable=invalid-name


class StrongHashSpec(HasherSpec):
  # [CLASS DOC]
  # Hasher spec for the keyed strong hash: the hasher name is fixed to "stronghash" and the key
  # must be a pair of integers. Construction validates the key.
  """A structure to specify a key of the strong keyed hash spec.

  The strong hash requires a `key`, which is a list of 2 unsigned integer
  numbers. These should be non-zero; random numbers generated from random.org
  would be a fine choice.

  Fields:
    key: The key to be used by the keyed hashing function.
  """
  __slots__ = ()

  def __new__(cls, key):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the class text says the key numbers must be unsigned, but the check accepts
    # any integer types including negative values.
    # Creates the spec. Input: key, a sequence of exactly two integers. Raises a value error if
    # its length is not two, and a type error if either element is not an integral type.
    # Returns a new spec with hasher name "stronghash" and that key.
    if len(key) != 2:
      raise ValueError(f"`key` must have size 2, received {len(key)}")

    if not isinstance(key[0], compat_util.integral_types) or not isinstance(
        key[1], compat_util.integral_types):
      raise TypeError("Invalid key %s. Must be unsigned integer values." % key)

    return super(cls, StrongHashSpec).__new__(cls, "stronghash", key)


def _as_string(tensor):
  # [FUNCTION DOC]
  # HELPER: converts keys to strings before hashing into buckets. Returns the tensor unchanged
  # if its base dtype is string, otherwise returns the string-converted tensor from the
  # as-string op.
  if dtypes.string == tensor.dtype.base_dtype:
    return tensor
  return string_ops.as_string(tensor)


class IdTableWithHashBuckets(LookupInterface):
  # [CLASS DOC]
  # Wrapper that gives a string-or-int64 to int64 id table out-of-vocabulary handling: unknown
  # keys are hashed into a configurable number of extra buckets whose ids follow the base
  # table's size. Holds the optional wrapped table, bucket count, hasher spec and table name.
  # With no wrapped table every key is bucketed. Delegates resource creation, initialization and
  # handle to the wrapped table.
  r"""String to Id table wrapper that assigns out-of-vocabulary keys to buckets.

  For example, if an instance of `IdTableWithHashBuckets` is initialized with a
  string-to-id table that maps:

  * `emerson -> 0`
  * `lake -> 1`
  * `palmer -> 2`

  The `IdTableWithHashBuckets` object will performs the following mapping:

  * `emerson -> 0`
  * `lake -> 1`
  * `palmer -> 2`
  * `<other term> -> bucket_id`, where bucket_id will be between `3` and
  `3 + num_oov_buckets - 1`, calculated by:
  `hash(<term>) % num_oov_buckets + vocab_size`

  If input_tensor is `["emerson", "lake", "palmer", "king", "crimson"]`,
  the lookup result is `[0, 1, 2, 4, 7]`.

  If `table` is None, only out-of-vocabulary buckets are used.

  Example usage:

  ```python
  num_oov_buckets = 3
  input_tensor = tf.constant(["emerson", "lake", "palmer", "king", "crimnson"])
  table = tf.IdTableWithHashBuckets(
      tf.StaticHashTable(
          tf.lookup.TextFileInitializer(
              filename,
              key_dtype=tf.string,
              key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
              value_dtype=tf.int64,
              value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
              delimiter="\t"),
          default_value),
      num_oov_buckets)
  out = table.lookup(input_tensor).
  table.init.run()
  print(out.eval())
  ```

  The hash function used for generating out-of-vocabulary buckets ID is handled
  by `hasher_spec`.
  """

  def __init__(self,
               table,
               num_oov_buckets,
               hasher_spec=FastHashSpec,
               name=None,
               key_dtype=None):
    # [FUNCTION DOC]
    # Validates and stores the configuration. Inputs: table (optional wrapped table, key dtype
    # int64 or string and value dtype int64), num_oov_buckets, hasher_spec (default fast hash),
    # name, key_dtype (defaults to the table's key dtype, or string without a table).
    # Trailing slashes are stripped from the name; the stored table name is its last path part.
    # Raises a type error for unsupported table key dtype, integer/non-integer mismatch between
    # the lookup and table key dtypes, non-int64 table values, a key dtype that is neither
    # integer nor string, or a hasher spec of the wrong type. Raises a value error if there is
    # no table and bucket count is not positive. Returns nothing.
    """Construct a `IdTableWithHashBuckets` object.

    Args:
      table: Table that maps `tf.string` or `tf.int64` keys to `tf.int64` ids.
      num_oov_buckets: Number of buckets to use for out-of-vocabulary keys.
      hasher_spec: A `HasherSpec` to specify the hash function to use for
        assignation of out-of-vocabulary buckets  (optional).
      name: A name for the operation (optional).
      key_dtype: Data type of keys passed to `lookup`. Defaults to
        `table.key_dtype` if `table` is specified, otherwise `tf.string`. Must
        be string or integer, and must be castable to `table.key_dtype`.

    Raises:
      ValueError: when `table` in None and `num_oov_buckets` is not positive.
      TypeError: when `hasher_spec` is invalid.
    """
    # If a name ends with a '/' it is a "name scope", remove all trailing '/'
    # characters to use as table name.
    if name:
      name = name.rstrip("/")
    if table:
      if key_dtype is None:
        key_dtype = table.key_dtype
      supported_table_key_dtypes = (dtypes.int64, dtypes.string)
      if table.key_dtype not in supported_table_key_dtypes:
        raise TypeError("Invalid `key_dtype`, expected one of "
                        f"{supported_table_key_dtypes}, received {key_dtype}.")
      if table.key_dtype.is_integer != key_dtype.is_integer:
        raise TypeError("Invalid `key dtype`, expected %s but got %s." %
                        ("integer" if key_dtype.is_integer else "non-integer",
                         table.key_dtype))
      if table.value_dtype != dtypes.int64:
        raise TypeError("Invalid `value_dtype`: expected int64 but got %s." %
                        (table.value_dtype))
      self._table = table
      name = name or self._table.name
    else:
      if num_oov_buckets <= 0:
        raise ValueError("`oov_buckets` must be > 0 if no `table` is supplied.")
      key_dtype = dtypes.string if key_dtype is None else key_dtype
      self._table = None
      name = name or "hash_bucket"
    if (not key_dtype.is_integer) and (dtypes.string != key_dtype):
      raise TypeError("Invalid `key_dtype`, expected integer or string, got "
                      f"{key_dtype}.")
    self._num_oov_buckets = num_oov_buckets

    if not isinstance(hasher_spec, HasherSpec):
      raise TypeError("`hasher_spec` must be of type HasherSpec, got "
                      f"{type(hasher_spec)}.")
    self._hasher_spec = hasher_spec
    if name:
      self._table_name = name.split("/")[-1]
    else:
      self._table_name = None
    super(IdTableWithHashBuckets, self).__init__(key_dtype, dtypes.int64)

  def _create_resource(self):
    # [FUNCTION DOC]
    # HELPER: delegates to the wrapped table's resource creation; returns nothing if there is
    # no wrapped table.
    if self._table is not None:
      return self._table._create_resource()  # pylint: disable=protected-access
    return None

  def _initialize(self):
    # [FUNCTION DOC]
    # HELPER: returns the wrapped table's initialization op, or a no-op named init when there is
    # no wrapped table.
    if self._table is not None:
      return self._table._initialize()  # pylint: disable=protected-access
    with ops.name_scope(None, "init"):
      return control_flow_ops.no_op()

  @property
  def initializer(self):
    # [FUNCTION DOC]
    # Read-only property: the wrapped table's existing initialization op, or a new no-op when
    # there is no wrapped table.
    if self._table is not None:
      return self._table._init_op  # pylint: disable=protected-access
    with ops.name_scope(None, "init"):
      return control_flow_ops.no_op()

  @property
  @deprecated("2018-12-15", "Use `initializer` instead.")
  def init(self):
    # [FUNCTION DOC]
    # Deprecated alias of the initializer property; returns the same op and emits a deprecation
    # warning.
    return self.initializer

  @property
  def resource_handle(self):
    # [FUNCTION DOC]
    # Read-only property: the wrapped table's resource handle, or nothing without a table.
    if self._table is not None:
      return self._table.resource_handle
    return None

  @property
  def name(self):
    # [FUNCTION DOC]
    # Read-only property: the table name (last path component), possibly nothing.
    return self._table_name

  def size(self, name=None):
    # [FUNCTION DOC]
    # Returns an int64 tensor: the wrapped table's size (zero without a table) plus the number
    # of out-of-vocabulary buckets. Optional op name.
    """Compute the number of elements in this table."""
    with ops.name_scope(name, "%s_Size" % self.name):
      if self._table:
        tsize = self._table.size()
      else:
        tsize = ops.convert_to_tensor(0, dtype=dtypes.int64)
      return tsize + self._num_oov_buckets

  def _get_string_to_hash_bucket_fn(self, hasher_spec):
    # [FUNCTION DOC]
    # HELPER: maps a hasher spec to the string-hashing function to use. Fast hash gives the fast
    # bucket op, legacy gives the legacy op, strong hash gives the strong op bound to the spec's
    # key. Raises a type error if the argument is not a hasher spec, and a value error for an
    # unknown hasher name. Returns the callable.
    """Returns the string_to_hash_bucket op to use based on `hasher_spec`."""
    if not isinstance(hasher_spec, HasherSpec):
      raise TypeError("`hasher_spec` must be of type HasherSpec, got "
                      f"{type(hasher_spec)}.")
    if hasher_spec.hasher == "fasthash":
      return string_ops.string_to_hash_bucket_fast
    if hasher_spec.hasher == "legacy":
      return string_ops.string_to_hash_bucket
    if hasher_spec.hasher == "stronghash":
      return functools.partial(
          string_ops.string_to_hash_bucket_strong, key=hasher_spec.key)
    raise ValueError(
        f"Found unknown hasher {hasher_spec.hasher} in `hasher_spec`")

  def lookup(self, keys, name=None):
    # [FUNCTION DOC]
    # Maps keys to ids. Keys may be dense, sparse or ragged (flat values used). Keys are cast
    # to int64 when the wrapped table has int64 keys. With zero buckets, returns the wrapped
    # table's lookup result. Otherwise hashes the string form of every key into a bucket using
    # the configured hasher, and, if a wrapped table exists, uses the table id when the key is
    # present (id differs from the table default) and the bucket id offset by the table size
    # otherwise. Output keeps the input structure (sparse, ragged or dense).
    # Raises a type error if the keys dtype differs from the table's lookup key dtype or the
    # hasher spec is invalid. Optional op name.
    """Looks up `keys` in the table, outputs the corresponding values.

    It assigns out-of-vocabulary keys to buckets based in their hashes.

    Args:
      keys: Keys to look up. May be either a `SparseTensor` or dense `Tensor`.
      name: Optional name for the op.

    Returns:
      A `SparseTensor` if keys are sparse, a `RaggedTensor` if keys are ragged,
      otherwise a dense `Tensor`.

    Raises:
      TypeError: when `keys` doesn't match the table key data type.
    """
    if keys.dtype.base_dtype != self._key_dtype:
      raise TypeError(f"Dtype of argument `keys` must be {self._key_dtype}, "
                      f"received: {keys.dtype}")
    values = keys
    # TODO(b/296302236): Remove RaggedTensor check by adding ragged
    # dispatching.
    if isinstance(keys, (sparse_tensor.SparseTensor, internal.RaggedTensor)):
      values = keys.values
    if self._table and (self._table.key_dtype.base_dtype == dtypes.int64):
      values = math_ops.cast(values, dtypes.int64)

    if self._num_oov_buckets == 0:
      ids = self._table.lookup(values, name=name)
    else:
      # TODO(yleon): Consider moving this functionality to its own kernel.
      with ops.name_scope(name, "%s_Lookup" % self.name):
        str_to_hash_bucket = self._get_string_to_hash_bucket_fn(
            self._hasher_spec)
        buckets = str_to_hash_bucket(
            _as_string(values),
            num_buckets=self._num_oov_buckets,
            name="hash_bucket")
        if self._table:
          ids = self._table.lookup(values)
          buckets = math_ops.add(buckets, self._table.size())
          is_id_non_default = math_ops.not_equal(ids, self._table.default_value)
          ids = array_ops.where_v2(is_id_non_default, ids, buckets)
        else:
          ids = buckets
    if isinstance(keys, sparse_tensor.SparseTensor):
      return sparse_tensor.SparseTensor(keys.indices, ids, keys.dense_shape)
    # TODO(b/296302236): Remove RaggedTensor check by adding ragged
    # dispatching.
    elif isinstance(keys, internal.RaggedTensor):
      return keys.with_values(ids)
    return ids


@tf_export("lookup.StaticVocabularyTable", v1=[])
class StaticVocabularyTable(LookupInterface):
  # [CLASS DOC]
  # Immutable string-or-int64 to int64 id table with out-of-vocabulary handling: keys found in
  # the vocabulary (loaded by an initializer into an internal hash table with default minus one)
  # get their id; other keys are hashed with the fast fingerprint hash into a fixed number of
  # buckets whose ids start after the vocabulary size. Without an initializer all keys are
  # bucketed. Holds the internal table, bucket count and table name; delegates resource
  # creation, initialization and handle to the internal table.
  r"""String to Id table that assigns out-of-vocabulary keys to hash buckets.

  For example, if an instance of `StaticVocabularyTable` is initialized with a
  string-to-id initializer that maps:

  >>> init = tf.lookup.KeyValueTensorInitializer(
  ...     keys=tf.constant(['emerson', 'lake', 'palmer']),
  ...     values=tf.constant([0, 1, 2], dtype=tf.int64))
  >>> table = tf.lookup.StaticVocabularyTable(
  ...    init,
  ...    num_oov_buckets=5)

  The `Vocabulary` object will performs the following mapping:

  * `emerson -> 0`
  * `lake -> 1`
  * `palmer -> 2`
  * `<other term> -> bucket_id`, where `bucket_id` will be between `3` and
  `3 + num_oov_buckets - 1 = 7`, calculated by:
  `hash(<term>) % num_oov_buckets + vocab_size`

  If input_tensor is:

  >>> input_tensor = tf.constant(["emerson", "lake", "palmer",
  ...                             "king", "crimson"])
  >>> table[input_tensor].numpy()
  array([0, 1, 2, 6, 7])

  If `initializer` is None, only out-of-vocabulary buckets are used.

  Example usage:

  >>> num_oov_buckets = 3
  >>> vocab = ["emerson", "lake", "palmer", "crimnson"]
  >>> import tempfile
  >>> f = tempfile.NamedTemporaryFile(delete=False)
  >>> f.write('\n'.join(vocab).encode('utf-8'))
  >>> f.close()

  >>> init = tf.lookup.TextFileInitializer(
  ...     f.name,
  ...     key_dtype=tf.string, key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
  ...     value_dtype=tf.int64, value_index=tf.lookup.TextFileIndex.LINE_NUMBER)
  >>> table = tf.lookup.StaticVocabularyTable(init, num_oov_buckets)
  >>> table.lookup(tf.constant(["palmer", "crimnson" , "king",
  ...                           "tarkus", "black", "moon"])).numpy()
  array([2, 3, 5, 6, 6, 4])

  The hash function used for generating out-of-vocabulary buckets ID is
  Fingerprint64.

  Note that the out-of-vocabulary bucket IDs always range from the table `size`
  up to `size + num_oov_buckets - 1` regardless of the table values, which could
  cause unexpected collisions:

  >>> init = tf.lookup.KeyValueTensorInitializer(
  ...     keys=tf.constant(["emerson", "lake", "palmer"]),
  ...     values=tf.constant([1, 2, 3], dtype=tf.int64))
  >>> table = tf.lookup.StaticVocabularyTable(
  ...     init,
  ...     num_oov_buckets=1)
  >>> input_tensor = tf.constant(["emerson", "lake", "palmer", "king"])
  >>> table[input_tensor].numpy()
  array([1, 2, 3, 3])
  """

  def __init__(self,
               initializer,
               num_oov_buckets,
               lookup_key_dtype=None,
               name=None,
               experimental_is_anonymous=False):
    # [FUNCTION DOC]
    # Validates arguments and builds the internal hash table. Inputs: initializer (optional
    # table initializer with int64 or string keys and int64 values), num_oov_buckets (must be
    # positive), lookup_key_dtype (defaults to the initializer key dtype, or string without an
    # initializer), name (trailing slashes stripped), and the anonymous flag (default false).
    # Raises a value error if the bucket count is not positive; raises a type error for an
    # unsupported initializer key dtype, integer/non-integer mismatch, non-int64 initializer
    # values, or a lookup key dtype that is neither integer nor string.
    # Side effects: tracks the initializer if trackable, creates a hash table with default minus
    # one (which creates a resource and an initialization op). Returns nothing.
    """Construct a `StaticVocabularyTable` object.

    Args:
      initializer: A `TableInitializerBase` object that contains the data used
        to initialize the table. If None, then we only use out-of-vocab buckets.
      num_oov_buckets: Number of buckets to use for out-of-vocabulary keys. Must
        be greater than zero. If out-of-vocab buckets are not required, use
        `StaticHashTable` instead.
      lookup_key_dtype: Data type of keys passed to `lookup`. Defaults to
        `initializer.key_dtype` if `initializer` is specified, otherwise
        `tf.string`. Must be string or integer, and must be castable to
        `initializer.key_dtype`.
      name: A name for the operation (optional).
      experimental_is_anonymous: Whether to use anonymous mode for the
        table (default is False). In anonymous mode, the table
        resource can only be accessed via a resource handle. It can't
        be looked up by a name. When all resource handles pointing to
        that resource are gone, the resource will be deleted
        automatically.

    Raises:
      ValueError: when `num_oov_buckets` is not positive.
      TypeError: when lookup_key_dtype or initializer.key_dtype are not
        integer or string. Also when initializer.value_dtype != int64.
    """
    if num_oov_buckets <= 0:
      raise ValueError("`num_oov_buckets` must be > 0; use StaticHashTable.")
    # If a name ends with a '/' it is a "name scope", remove all trailing '/'
    # characters to use as table name.
    if name:
      name = name.rstrip("/")
    if initializer:
      if lookup_key_dtype is None:
        lookup_key_dtype = initializer.key_dtype
      supported_table_key_dtypes = (dtypes.int64, dtypes.string)
      if initializer.key_dtype not in supported_table_key_dtypes:
        raise TypeError("Invalid `key_dtype`, expected one of %s, but got %s." %
                        (supported_table_key_dtypes, initializer.key_dtype))
      if initializer.key_dtype.is_integer != lookup_key_dtype.is_integer:
        raise TypeError(
            "Invalid `key_dtype`, expected %s but got %s." %
            ("integer" if lookup_key_dtype.is_integer else "non-integer",
             initializer.key_dtype))
      if initializer.value_dtype != dtypes.int64:
        raise TypeError("Invalid `value_dtype`, expected %s but got %s." %
                        (dtypes.int64, initializer.value_dtype))
      if isinstance(initializer, trackable_base.Trackable):
        self._initializer = self._track_trackable(initializer, "_initializer")
      self._table = HashTable(
          initializer,
          default_value=-1,
          experimental_is_anonymous=experimental_is_anonymous)
      name = name or self._table.name
    else:
      lookup_key_dtype = dtypes.string
      self._table = None
      name = name or "hash_bucket"
    if (not lookup_key_dtype.is_integer) and (dtypes.string !=
                                              lookup_key_dtype):
      raise TypeError("Invalid `key_dtype`, expected integer or string, got "
                      f"{lookup_key_dtype}")
    self._num_oov_buckets = num_oov_buckets

    self._table_name = None
    if name is not None:
      self._table_name = name.split("/")[-1]
    super(StaticVocabularyTable, self).__init__(lookup_key_dtype, dtypes.int64)

  def _create_resource(self):
    # [FUNCTION DOC]
    # HELPER: delegates to the internal table's resource creation; nothing if no internal table.
    if self._table is not None:
      return self._table._create_resource()  # pylint: disable=protected-access
    return None

  def _initialize(self):
    # [FUNCTION DOC]
    # HELPER: returns the internal table's initialization op, or a new no-op named init when
    # there is no internal table.
    if self._table is not None:
      return self._table._initialize()  # pylint: disable=protected-access
    with ops.name_scope(None, "init"):
      return control_flow_ops.no_op()

  @property
  def resource_handle(self):
    # [FUNCTION DOC]
    # Read-only property: the internal table's resource handle, or nothing without a table.
    if self._table is not None:
      return self._table.resource_handle
    return None

  @property
  def name(self):
    # [FUNCTION DOC]
    # Read-only property: the table name (last path component), possibly nothing.
    return self._table_name

  def size(self, name=None):
    # [FUNCTION DOC]
    # Returns an int64 tensor: the internal table's size (zero without a table) plus the number
    # of out-of-vocabulary buckets. Optional op name.
    """Compute the number of elements in this table."""
    with ops.name_scope(name, "%s_Size" % self.name):
      if self._table:
        tsize = self._table.size()
      else:
        tsize = ops.convert_to_tensor(0, dtype=dtypes.int64)
      return tsize + self._num_oov_buckets

  def lookup(self, keys, name=None):
    # [FUNCTION DOC]
    # Maps keys (dense, sparse or ragged; flat values used) to int64 ids. Keys are cast to int64
    # when the internal table has int64 keys. Every key is hashed (string form) into a bucket
    # with the fast hash; if an internal table exists, keys present in it get their stored id
    # and the rest get the bucket id offset by the table size; without a table the bucket id is
    # the result. Output keeps the input structure. Raises a type error if the keys dtype
    # differs from the table's lookup key dtype. Optional op name.
    """Looks up `keys` in the table, outputs the corresponding values.

    It assigns out-of-vocabulary keys to buckets based in their hashes.

    Args:
      keys: Keys to look up. May be either a `SparseTensor` or dense `Tensor`.
      name: Optional name for the op.

    Returns:
      A `SparseTensor` if keys are sparse, a `RaggedTensor` if keys are ragged,
      otherwise a dense `Tensor`.

    Raises:
      TypeError: when `keys` doesn't match the table key data type.
    """
    if keys.dtype.base_dtype != self._key_dtype:
      raise TypeError(f"Dtype of argument `keys` must be {self._key_dtype}, "
                      f"received: {keys.dtype}")
    values = keys
    # TODO(b/296302236): Remove RaggedTensor check by adding ragged
    # dispatching.
    if isinstance(keys, (sparse_tensor.SparseTensor, internal.RaggedTensor)):
      values = keys.values
    if self._table and (self._table.key_dtype.base_dtype == dtypes.int64):
      values = math_ops.cast(values, dtypes.int64)

    # TODO(yleon): Consider moving this functionality to its own kernel.
    with ops.name_scope(name, "%s_Lookup" % self.name):
      buckets = string_ops.string_to_hash_bucket_fast(
          _as_string(values),
          num_buckets=self._num_oov_buckets,
          name="hash_bucket")
      if self._table:
        ids = self._table.lookup(values)
        buckets = math_ops.add(buckets, self._table.size())
        is_id_non_default = math_ops.not_equal(ids, self._table.default_value)
        ids = array_ops.where_v2(is_id_non_default, ids, buckets)
      else:
        ids = buckets
    if isinstance(keys, sparse_tensor.SparseTensor):
      return sparse_tensor.SparseTensor(keys.indices, ids, keys.dense_shape)
    # TODO(b/296302236): Remove RaggedTensor check by adding ragged
    # dispatching.
    elif isinstance(keys, internal.RaggedTensor):
      return keys.with_values(ids)
    return ids


@tf_export(v1=["lookup.StaticVocabularyTable"])
class StaticVocabularyTableV1(StaticVocabularyTable):
  # [CLASS DOC]
  # Graph-mode (v1) variant of the static vocabulary table: adds the initializer property
  # exposing the initialization op for session-based running.

  @property
  def initializer(self):
    # [FUNCTION DOC]
    # Read-only property: the internal table's initialization op, or a new no-op when there is
    # no internal table.
    if self._table is not None:
      return self._table._init_op  # pylint: disable=protected-access
    with ops.name_scope(None, "init"):
      return control_flow_ops.no_op()


def index_table_from_file(vocabulary_file=None,
                          num_oov_buckets=0,
                          vocab_size=None,
                          default_value=-1,
                          hasher_spec=FastHashSpec,
                          key_dtype=dtypes.string,
                          name=None,
                          key_column_index=TextFileIndex.WHOLE_LINE,
                          value_column_index=TextFileIndex.LINE_NUMBER,
                          delimiter="\t"):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring tells the caller to run the table initializer in a session, but
  # in eager mode the table is already initialized at construction; and key_dtype must be a
  # dtype object (its integer property is read directly), not any dtype-like value.
  # Builds a file-based vocabulary lookup table. Inputs: vocabulary_file (required, non-empty
  # path or constant string tensor), num_oov_buckets (non-negative, default zero), vocab_size
  # (optional, must be at least one), default_value (id for unknown keys when there are no
  # buckets, default minus one), hasher_spec, key_dtype (integer or string, default string),
  # name, key and value column indices and delimiter.
  # Creates a static hash table fed by a text-file id initializer (integer keys are read as
  # int64) inside name scopes. If the bucket count is positive, wraps it in an out-of-vocabulary
  # bucket table. Returns the resulting table.
  # Raises a value error for a missing or empty vocabulary file, negative bucket count or
  # non-positive vocab size, and a type error for a key dtype that is neither integer nor string.
  """Returns a lookup table that converts a string tensor into int64 IDs.

  This operation constructs a lookup table to convert tensor of strings into
  int64 IDs. The mapping can be initialized from a vocabulary file specified in
  `vocabulary_file`, where the whole line is the key and the zero-based line
  number is the ID.

  Any lookup of an out-of-vocabulary token will return a bucket ID based on its
  hash if `num_oov_buckets` is greater than zero. Otherwise it is assigned the
  `default_value`.
  The bucket ID range is
  `[vocabulary size, vocabulary size + num_oov_buckets - 1]`.

  The underlying table must be initialized by calling
  `session.run(tf.compat.v1.tables_initializer())` or
  `session.run(table.init())` once.

  To specify multi-column vocabulary files, use key_column_index and
  value_column_index and delimiter.

  - TextFileIndex.LINE_NUMBER means use the line number starting from zero,
    expects data type int64.
  - TextFileIndex.WHOLE_LINE means use the whole line content, expects data
    type string.
  - A value >=0 means use the index (starting at zero) of the split line based
    on `delimiter`.

  Sample Usages:

  If we have a vocabulary file "test.txt" with the following content:

  ```
  emerson
  lake
  palmer
  ```

  ```python
  features = tf.constant(["emerson", "lake", "and", "palmer"])
  table = tf.lookup.index_table_from_file(
      vocabulary_file="test.txt", num_oov_buckets=1)
  ids = table.lookup(features)
  ...
  tf.compat.v1.tables_initializer().run()

  ids.eval()  ==> [0, 1, 3, 2]  # where 3 is the out-of-vocabulary bucket
  ```

  Args:
    vocabulary_file: The vocabulary filename, may be a constant scalar `Tensor`.
    num_oov_buckets: The number of out-of-vocabulary buckets.
    vocab_size: Number of the elements in the vocabulary, if known.
    default_value: The value to use for out-of-vocabulary feature values.
      Defaults to -1.
    hasher_spec: A `HasherSpec` to specify the hash function to use for
      assignation of out-of-vocabulary buckets.
    key_dtype: The `key` data type.
    name: A name for this op (optional).
    key_column_index: The column index from the text file to get the `key`
      values from. The default is to use the whole line content.
    value_column_index: The column index from the text file to get the `value`
      values from. The default is to use the line number, starting from zero.
    delimiter: The delimiter to separate fields in a line.

  Returns:
    The lookup table to map a `key_dtype` `Tensor` to index `int64` `Tensor`.

  Raises:
    ValueError: If `vocabulary_file` is not set.
    ValueError: If `num_oov_buckets` is negative or `vocab_size` is not greater
      than zero.
  """
  if vocabulary_file is None or (isinstance(vocabulary_file, str) and
                                 not vocabulary_file):
    raise ValueError(
        "`vocabulary_file` must be specified and must not be empty.")
  if num_oov_buckets < 0:
    raise ValueError(
        "num_oov_buckets must be greater or equal than 0, got %d." %
        num_oov_buckets)
  if vocab_size is not None and vocab_size < 1:
    vocab_file_value = vocabulary_file
    if isinstance(vocabulary_file, tensor_lib.Tensor):
      vocab_file_value = tensor_util.constant_value(vocabulary_file) or "?"
    raise ValueError("`vocab_size` must be greater than 0, got %d for "
                     "vocabulary_file: %s." % (vocab_size, vocab_file_value))
  if (not key_dtype.is_integer) and (dtypes.string != key_dtype.base_dtype):
    raise TypeError("Dtype for `keys` should be either integer or string.")

  with ops.name_scope(name, "string_to_index"):
    table = None
    with ops.name_scope(None, "hash_table"):
      init = TextFileIdTableInitializer(
          vocabulary_file,
          vocab_size=vocab_size,
          key_dtype=dtypes.int64 if key_dtype.is_integer else key_dtype,
          name="table_init",
          key_column_index=key_column_index,
          value_column_index=value_column_index,
          delimiter=delimiter)

      table = StaticHashTableV1(init, default_value)
    if num_oov_buckets:
      table = IdTableWithHashBuckets(
          table,
          num_oov_buckets=num_oov_buckets,
          hasher_spec=hasher_spec,
          key_dtype=key_dtype)

    return table


def index_table_from_tensor(vocabulary_list,
                            num_oov_buckets=0,
                            default_value=-1,
                            hasher_spec=FastHashSpec,
                            dtype=dtypes.string,
                            name=None):
  # [FUNCTION DOC]
  # Builds a vocabulary lookup table from an in-memory list or tensor of keys, the id being the
  # position in the list. Inputs: vocabulary_list (required, one-dimensional, convertible to a
  # tensor), num_oov_buckets (non-negative, default zero), default_value (default minus one),
  # hasher_spec, dtype of lookup keys (integer or string, default string) and name.
  # Integer keys are cast to int64. Creates a static hash table from a key-value tensor
  # initializer (ids from a range cast to int64) and, if the bucket count is positive, wraps it
  # in an out-of-vocabulary bucket table. Returns the table.
  # Raises a value error if the list is missing, the bucket count is negative or the list
  # dtype disagrees with the requested dtype; raises a type error if dtype is neither integer
  # nor string.
  """Returns a lookup table that converts a string tensor into int64 IDs.

  This operation constructs a lookup table to convert tensor of strings into
  int64 IDs. The mapping can be initialized from a string `vocabulary_list` 1-D
  tensor where each element is a key and corresponding index within the tensor
  is the value.

  Any lookup of an out-of-vocabulary token will return a bucket ID based on its
  hash if `num_oov_buckets` is greater than zero. Otherwise it is assigned the
  `default_value`. The bucket ID range is
  `[vocabulary list size, vocabulary list size + num_oov_buckets - 1]`.

  The underlying table must be initialized by calling
  `session.run(tf.compat.v1.tables_initializer())` or
  `session.run(table.init())` once.

  Elements in `vocabulary_list` cannot have duplicates, otherwise when executing
  the table initializer op, it will throw a `FailedPreconditionError`.

  Sample Usages:

  ```python
  vocabulary_list = tf.constant(["emerson", "lake", "palmer"])
  table = tf.lookup.index_table_from_tensor(
      vocabulary_list=vocabulary_list, num_oov_buckets=1, default_value=-1)
  features = tf.constant(["emerson", "lake", "and", "palmer"])
  ids = table.lookup(features)
  ...
  tf.compat.v1.tables_initializer().run()

  ids.eval()  ==> [0, 1, 4, 2]
  ```

  Args:
    vocabulary_list: A 1-D `Tensor` that specifies the mapping of keys to
      indices. The type of this object must be castable to `dtype`.
    num_oov_buckets: The number of out-of-vocabulary buckets.
    default_value: The value to use for out-of-vocabulary feature values.
      Defaults to -1.
    hasher_spec: A `HasherSpec` to specify the hash function to use for
      assignment of out-of-vocabulary buckets.
    dtype: The type of values passed to `lookup`. Only string and integers are
      supported.
    name: A name for this op (optional).

  Returns:
    The lookup table to map an input `Tensor` to index `int64` `Tensor`.

  Raises:
    ValueError: If `vocabulary_list` is invalid.
    ValueError: If `num_oov_buckets` is negative.
  """
  if vocabulary_list is None:
    raise ValueError("`vocabulary_list` must be specified.")

  if num_oov_buckets < 0:
    raise ValueError(
        "`num_oov_buckets` must be greater or equal than 0, got %d." %
        num_oov_buckets)

  if (not dtype.is_integer) and (dtypes.string != dtype.base_dtype):
    raise TypeError("`dtype` must either be integer or string.")

  with ops.name_scope(name, "string_to_index"):
    keys = ops.convert_to_tensor(vocabulary_list)
    if keys.dtype.is_integer != dtype.is_integer:
      raise ValueError(
          "Invalid `dtype`: Expected %s, got %s." %
          ("integer" if dtype.is_integer else "non-integer", keys.dtype))
    if (not dtype.is_integer) and (keys.dtype.base_dtype != dtype):
      raise ValueError("Invalid `dtype`: Expected %s, got %s." %
                       (dtype, keys.dtype))
    num_elements = array_ops.size(keys)
    values = math_ops.cast(math_ops.range(num_elements), dtypes.int64)

    with ops.name_scope(None, "hash_table"):
      table_keys = math_ops.cast(
          keys, dtypes.int64) if keys.dtype.is_integer else keys
      init = KeyValueTensorInitializer(
          table_keys,
          values,
          table_keys.dtype.base_dtype,
          dtypes.int64,
          name="table_init")
      table = StaticHashTableV1(init, default_value)
    if num_oov_buckets:
      table = IdTableWithHashBuckets(
          table,
          num_oov_buckets=num_oov_buckets,
          hasher_spec=hasher_spec,
          key_dtype=dtype)
    return table


def index_to_string_table_from_file(vocabulary_file,
                                    vocab_size=None,
                                    default_value="UNK",
                                    name=None,
                                    key_column_index=TextFileIndex.LINE_NUMBER,
                                    value_column_index=TextFileIndex.WHOLE_LINE,
                                    delimiter="\t"):
  # [FUNCTION DOC]
  # Builds a table mapping int64 indices to strings from a vocabulary file. Inputs:
  # vocabulary_file (required, non-empty), vocab_size (optional, at least one), default_value
  # (string for unknown indices, default "UNK"), name, key and value column indices (defaults:
  # line number and whole line) and delimiter. Creates a text-file id-to-string initializer
  # and returns a static hash table over it. Raises a value error for an empty file or invalid
  # vocab size.
  """Returns a lookup table that maps a `Tensor` of indices into strings.

  This operation constructs a lookup table to map int64 indices into string
  values. The table is initialized from a vocabulary file specified in
  `vocabulary_file`, where the whole line is the value and the
  zero-based line number is the index.

  Any input which does not have a corresponding index in the vocabulary file
  (an out-of-vocabulary entry) is assigned the `default_value`

  The underlying table must be initialized by calling
  `session.run(tf.compat.v1.tables_initializer())` or
  `session.run(table.init())` once.

  To specify multi-column vocabulary files, use key_column_index and
  value_column_index and delimiter.

  - TextFileIndex.LINE_NUMBER means use the line number starting from zero,
    expects data type int64.
  - TextFileIndex.WHOLE_LINE means use the whole line content, expects data
    type string.
  - A value >=0 means use the index (starting at zero) of the split line based
    on `delimiter`.

  Sample Usages:

  If we have a vocabulary file "test.txt" with the following content:

  ```
  emerson
  lake
  palmer
  ```

  ```python
  indices = tf.constant([1, 5], tf.int64)
  table = tf.lookup.index_to_string_table_from_file(
      vocabulary_file="test.txt", default_value="UNKNOWN")
  values = table.lookup(indices)
  ...
  tf.compat.v1.tables_initializer().run()

  values.eval() ==> ["lake", "UNKNOWN"]
  ```

  Args:
    vocabulary_file: The vocabulary filename, may be a constant scalar `Tensor`.
    vocab_size: Number of the elements in the vocabulary, if known.
    default_value: The value to use for out-of-vocabulary indices.
    name: A name for this op (optional).
    key_column_index: The column index from the text file to get the `key`
      values from. The default is to use the line number, starting from zero.
    value_column_index: The column index from the text file to get the `value`
      values from. The default is to use the whole line content.
    delimiter: The delimiter to separate fields in a line.

  Returns:
    The lookup table to map a string values associated to a given index `int64`
    `Tensors`.

  Raises:
    ValueError: when `vocabulary_file` is empty.
    ValueError: when `vocab_size` is invalid.
  """
  if vocabulary_file is None or (isinstance(vocabulary_file, str) and
                                 not vocabulary_file):
    raise ValueError(
        "`vocabulary_file` must be specified and must not be empty.")

  if vocab_size is not None and vocab_size < 1:
    raise ValueError(f"`vocab_size` must be greater than 0, got {vocab_size}.")

  with ops.name_scope(name, "index_to_string"):
    init = TextFileStringTableInitializer(
        vocabulary_file,
        vocab_size=vocab_size,
        name="table_init",
        key_column_index=key_column_index,
        value_column_index=value_column_index,
        delimiter=delimiter)

    # TODO(yleon): Use a more efficient structure.
    return StaticHashTableV1(init, default_value)


def index_to_string_table_from_tensor(vocabulary_list,
                                      default_value="UNK",
                                      name=None):
  # [FUNCTION DOC]
  # Builds a table mapping int64 indices to strings from an in-memory string list. Inputs:
  # vocabulary_list (required, one-dimensional strings), default_value (default "UNK") and name.
  # Converts the list to a string tensor, uses its positions (as int64) as keys, and returns a
  # static hash table fed by a key-value tensor initializer. Raises a value error if the list is
  # missing.
  """Returns a lookup table that maps a `Tensor` of indices into strings.

  This operation constructs a lookup table to map int64 indices into string
  values. The mapping is initialized from a string `vocabulary_list` 1-D
  `Tensor` where each element is a value and the corresponding index within the
  tensor is the key.

  Any input which does not have a corresponding index in 'vocabulary_list'
  (an out-of-vocabulary entry) is assigned the `default_value`

  The underlying table must be initialized by calling
  `session.run(tf.compat.v1.tables_initializer())` or
  `session.run(table.init())` once.

  Elements in `vocabulary_list` cannot have duplicates, otherwise when executing
  the table initializer op, it will throw a `FailedPreconditionError`.

  Sample Usages:

  ```python
  vocabulary_list = tf.constant(["emerson", "lake", "palmer"])
  indices = tf.constant([1, 5], tf.int64)
  table = tf.lookup.index_to_string_table_from_tensor(
      vocabulary_list, default_value="UNKNOWN")
  values = table.lookup(indices)
  ...
  tf.compat.v1.tables_initializer().run()

  values.eval() ==> ["lake", "UNKNOWN"]
  ```

  Args:
    vocabulary_list: A 1-D string `Tensor` that specifies the strings to map
      from indices.
    default_value: The value to use for out-of-vocabulary indices.
    name: A name for this op (optional).

  Returns:
    The lookup table to map a string values associated to a given index `int64`
    `Tensors`.

  Raises:
    ValueError: when `vocabulary_list` is not set.
  """

  if vocabulary_list is None:
    raise ValueError("`vocabulary_list` argument must be specified.")

  with ops.name_scope(name, "index_to_string"):
    vocabulary_list = ops.convert_to_tensor(vocabulary_list, dtypes.string)
    num_elements = array_ops.size(vocabulary_list)
    keys = math_ops.cast(math_ops.range(num_elements), dtypes.int64)

    init = KeyValueTensorInitializer(
        keys, vocabulary_list, dtypes.int64, dtypes.string, name="table_init")
    # TODO(yleon): Use a more efficient structure.
    return StaticHashTableV1(init, default_value)


@tf_export("lookup.experimental.MutableHashTable")
@saveable_compat.legacy_saveable_name("table")
class MutableHashTable(LookupInterface):
  # [CLASS DOC]
  # Mutable hash table: entries can be inserted, removed, looked up and exported at any time;
  # it is not initialized from an initializer. Holds the default value tensor (scalar or
  # per-entry vector), key and value dtypes, name, checkpoint flag, anonymity flag and optional
  # shared name. Supports checkpointing through trackable tensor serialization, plus a nested
  # saveable for legacy checkpoints, and can copy itself to CPU for checkpoint saving.
  """A generic mutable hash table implementation.

  Data can be inserted by calling the `insert` method and removed by calling the
  `remove` method. It does not support initialization via the init method.

  `MutableHashTable` requires additional memory during checkpointing and restore
  operations to create temporary key and value tensors.

  Example usage:

  >>> table = tf.lookup.experimental.MutableHashTable(key_dtype=tf.string,
  ...                                                 value_dtype=tf.int64,
  ...                                                 default_value=-1)
  >>> keys_tensor = tf.constant(['a', 'b', 'c'])
  >>> vals_tensor = tf.constant([7, 8, 9], dtype=tf.int64)
  >>> input_tensor = tf.constant(['a', 'f'])
  >>> table.insert(keys_tensor, vals_tensor)
  >>> table.lookup(input_tensor).numpy()
  array([ 7, -1])
  >>> table.remove(tf.constant(['c']))
  >>> table.lookup(keys_tensor).numpy()
  array([ 7, 8, -1])
  >>> sorted(table.export()[0].numpy())
  [b'a', b'b']
  >>> [a.item() for a in sorted(table.export()[1].numpy())]
  [7, 8]
  """

  def __init__(self,
               key_dtype,
               value_dtype,
               default_value,
               name="MutableHashTable",
               checkpoint=True,
               experimental_is_anonymous=False):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it returns a table object and raises a value error when
    # checkpoint is true and no name is given; a constructor returns nothing and no such check
    # exists in the code (the name has a default).
    # Creates an empty mutable table. Inputs: key and value dtypes, default_value (converted to a
    # tensor of the value dtype; its shape decides scalar or vector values), name (default
    # "MutableHashTable"), checkpoint flag (default true) and anonymous flag (default false).
    # Not anonymous: in eager mode a unique shared name is generated (which can leak memory per
    # the source comment); otherwise none, so node-name sharing is used when checkpointing.
    # Side effects: creates the table resource, and if checkpointing builds a saveable object
    # and, in graph mode, adds it to the graph's saveable-objects collection.
    """Creates an empty `MutableHashTable` object.

    Creates a table, the type of its keys and values are specified by key_dtype
    and value_dtype, respectively.

    Args:
      key_dtype: the type of the key tensors.
      value_dtype: the type of the value tensors.
      default_value: The value to use if a key is missing in the table.
      name: A name for the operation (optional).
      checkpoint: if True, the contents of the table are saved to and restored
        from checkpoints. If `shared_name` is empty for a checkpointed table, it
        is shared using the table node name.
      experimental_is_anonymous: Whether to use anonymous mode for the
        table (default is False). In anonymous mode, the table
        resource can only be accessed via a resource handle. It can't
        be looked up by a name. When all resource handles pointing to
        that resource are gone, the resource will be deleted
        automatically.

    Returns:
      A `MutableHashTable` object.

    Raises:
      ValueError: If checkpoint is True and no name was specified.
    """
    self._default_value = ops.convert_to_tensor(
        default_value, dtype=value_dtype)
    self._value_shape = self._default_value.get_shape()
    self._checkpoint = checkpoint
    self._key_dtype = key_dtype
    self._value_dtype = value_dtype
    self._name = name
    self._is_anonymous = experimental_is_anonymous
    if not self._is_anonymous:
      self._shared_name = None
      if context.executing_eagerly():
        # TODO(allenl): This will leak memory due to kernel caching by
        # the shared_name attribute value (but is better than the
        # alternative of sharing everything by default when executing
        # eagerly; hopefully creating tables in a loop is uncommon).
        self._shared_name = "table_%d" % (ops.uid(),)
    super(MutableHashTable, self).__init__(key_dtype, value_dtype)
    self._resource_handle = self._create_resource()
    if checkpoint:
      saveable = MutableHashTable._Saveable(self, name)
      if not context.executing_eagerly():
        ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable)

  def _create_resource(self):
    # [FUNCTION DOC]
    # HELPER: creates the mutable table resource. Chooses the anonymous or named kernel, and the
    # scalar-value or vector-value (value shape from the default value) variant, based on the
    # anonymity flag and the default value's rank. Named tables use node-name sharing when
    # checkpointing without an explicit shared name. Records the table name (last component of
    # the op name in graph mode, nothing in eager mode) and returns the resource handle.
    if self._is_anonymous:
      if self._default_value.get_shape().ndims == 0:
        table_ref = gen_lookup_ops.anonymous_mutable_hash_table(
            key_dtype=self._key_dtype,
            value_dtype=self._value_dtype,
            name=self._name)
      else:
        table_ref = gen_lookup_ops.anonymous_mutable_hash_table_of_tensors(
            key_dtype=self._key_dtype,
            value_dtype=self._value_dtype,
            value_shape=self._default_value.get_shape(),
            name=self._name)
    else:
      # The table must be shared if checkpointing is requested for multi-worker
      # training to work correctly. Use the node name if no shared_name has been
      # explicitly specified.
      use_node_name_sharing = self._checkpoint and self._shared_name is None
      if self._default_value.get_shape().ndims == 0:
        table_ref = gen_lookup_ops.mutable_hash_table_v2(
            shared_name=self._shared_name,
            use_node_name_sharing=use_node_name_sharing,
            key_dtype=self._key_dtype,
            value_dtype=self._value_dtype,
            name=self._name)
      else:
        table_ref = gen_lookup_ops.mutable_hash_table_of_tensors_v2(
            shared_name=self._shared_name,
            use_node_name_sharing=use_node_name_sharing,
            key_dtype=self._key_dtype,
            value_dtype=self._value_dtype,
            value_shape=self._default_value.get_shape(),
            name=self._name)

    if context.executing_eagerly():
      self._table_name = None
    else:
      self._table_name = table_ref.op.name.split("/")[-1]
    return table_ref

  @property
  def name(self):
    # [FUNCTION DOC]
    # Read-only property: the table's op name (last path component), nothing in eager mode.
    return self._table_name

  def size(self, name=None):
    # [FUNCTION DOC]
    # Returns a scalar int64 tensor with the number of entries, computed on the device holding
    # the table resource. Optional op name.
    """Compute the number of elements in this table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A scalar tensor containing the number of elements in this table.
    """
    with ops.name_scope(name, "%s_Size" % self.name, [self.resource_handle]):
      with ops.colocate_with(self.resource_handle):
        return gen_lookup_ops.lookup_table_size_v2(self.resource_handle)

  def remove(self, keys, name=None):
    # [FUNCTION DOC]
    # Deletes the given keys (a tensor of any shape with the table's key dtype) and their
    # values; absent keys are ignored. Optional op name. Returns the removal operation (or
    # nothing useful in eager mode). Raises a type error if the keys dtype differs from the
    # table's key dtype. Mutates table state when run.
    """Removes `keys` and its associated values from the table.

    If a key is not present in the table, it is silently ignored.

    Args:
      keys: Keys to remove. Can be a tensor of any shape. Must match the table's
        key type.
      name: A name for the operation (optional).

    Returns:
      The created Operation.

    Raises:
      TypeError: when `keys` do not match the table data types.
    """
    if keys.dtype != self._key_dtype:
      raise TypeError(f"Dtype of argument `keys` must be {self._key_dtype}, "
                      f"received: {keys.dtype}")

    with ops.name_scope(name, "%s_lookup_table_remove" % self.name,
                        (self.resource_handle, keys, self._default_value)):
      op = gen_lookup_ops.lookup_table_remove_v2(self.resource_handle, keys)

    return op

  def lookup(self, keys, dynamic_default_values=None, name=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says a type error is raised for mismatched key types, but the
    # keys are converted to the key dtype so the error comes from tensor conversion.
    # Looks up keys (anything convertible to the key dtype, any shape) and returns values with
    # the keys' shape plus the value shape. Missing keys get dynamic_default_values if provided
    # (matching the default's shape or the result's shape, giving per-key defaults), else the
    # table's default value. Runs on the device holding the table. Optional op name.
    """Looks up `keys` in a table, outputs the corresponding values.

    The `default_value` is used for keys not present in the table.

    Args:
      keys: Keys to look up. Can be a tensor of any shape. Must match the
        table's key_dtype.
      dynamic_default_values: The values to use if a key is missing in the
        table. If None (by default), the `table.default_value` will be used.
        Shape of `dynamic_default_values` must be same with
        `table.default_value` or the lookup result tensor.
        In the latter case, each key will have a different default value.

        For example:

          ```python
          keys = [0, 1, 3]
          dynamic_default_values = [[1, 3, 4], [2, 3, 9], [8, 3, 0]]

          # The key '0' will use [1, 3, 4] as default value.
          # The key '1' will use [2, 3, 9] as default value.
          # The key '3' will use [8, 3, 0] as default value.
          ```

      name: A name for the operation (optional).

    Returns:
      A tensor containing the values in the same shape as `keys` using the
        table's value type.

    Raises:
      TypeError: when `keys` do not match the table data types.
    """
    with ops.name_scope(name, "%s_lookup_table_find" % self.name,
                        (self.resource_handle, keys, self._default_value)):
      keys = ops.convert_to_tensor(keys, dtype=self._key_dtype, name="keys")
      with ops.colocate_with(self.resource_handle):
        values = gen_lookup_ops.lookup_table_find_v2(
            self.resource_handle, keys, dynamic_default_values
            if dynamic_default_values is not None else self._default_value)
    return values

  def insert(self, keys, values, name=None):
    # [FUNCTION DOC]
    # Inserts or overwrites entries. Inputs: keys and values (convertible to the table's key and
    # value dtypes; values must match the keys' shape plus value shape) and an optional op name.
    # Returns the insertion operation. Conversion errors surface as type errors. Mutates the
    # table when run.
    """Associates `keys` with `values`.

    Args:
      keys: Keys to insert. Can be a tensor of any shape. Must match the table's
        key type.
      values: Values to be associated with keys. Must be a tensor of the same
        shape as `keys` and match the table's value type.
      name: A name for the operation (optional).

    Returns:
      The created Operation.

    Raises:
      TypeError: when `keys` or `values` doesn't match the table data
        types.
    """
    with ops.name_scope(name, "%s_lookup_table_insert" % self.name,
                        [self.resource_handle, keys, values]):
      keys = ops.convert_to_tensor(keys, self._key_dtype, name="keys")
      values = ops.convert_to_tensor(values, self._value_dtype, name="values")
      with ops.colocate_with(self.resource_handle):
        # pylint: disable=protected-access
        op = gen_lookup_ops.lookup_table_insert_v2(self.resource_handle, keys,
                                                   values)
    return op

  def export(self, name=None):
    # [FUNCTION DOC]
    # Exports every entry using the export kernel on the table's device. Optional op name.
    # Returns a pair (all keys tensor, all values tensor). No static shape is set on the values.
    """Returns tensors of all keys and values in the table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A pair of tensors with the first tensor containing all keys and the
        second tensors containing all values in the table.
    """
    with ops.name_scope(name, "%s_lookup_table_export_values" % self.name,
                        [self.resource_handle]):
      with ops.colocate_with(self.resource_handle):
        exported_keys, exported_values = gen_lookup_ops.lookup_table_export_v2(
            self.resource_handle, self._key_dtype, self._value_dtype)
    return exported_keys, exported_values

  def _serialize_to_tensors(self):
    # [FUNCTION DOC]
    # HELPER: checkpoint hook. Exports the table and returns a mapping from the names for keys
    # and for values to the exported key and value tensors.
    """Implements checkpointing protocols for `Trackable`."""
    tensors = self.export()
    return {"-keys": tensors[0], "-values": tensors[1]}

  def _restore_from_tensors(self, restored_tensors):
    # [FUNCTION DOC]
    # HELPER: checkpoint hook. Takes a mapping holding restored keys and values tensors and
    # bulk-imports them into the table on its device; returns the import operation.
    """Implements checkpointing protocols for `Trackable`."""
    with ops.name_scope("%s_table_restore" % self._name):
      with ops.colocate_with(self.resource_handle):
        return gen_lookup_ops.lookup_table_import_v2(
            self.resource_handle,
            restored_tensors["-keys"],
            restored_tensors["-values"])

  def _copy_trackable_to_cpu(self, object_map):
    # [FUNCTION DOC]
    # HELPER: checkpoint hook. Given a mapping from original objects to their copies, creates a
    # new table with the same configuration for this object if none exists yet, then copies
    # this table's contents into that copy. Mutates the mapping. Returns nothing.
    """Implements checkpointing protocols for `Trackable`."""
    if self not in object_map:
      # If self is not already populated in object map, instantiate the copy
      object_map[self] = MutableHashTable(
          self._key_dtype,
          self._value_dtype,
          self._default_value,
          self._name,
          self._checkpoint,
          self._is_anonymous
      )

    # Copy values from `self` to copy of `self`
    serialized = self._serialize_to_tensors()
    object_map[self]._restore_from_tensors(serialized)  # pylint: disable=protected-access

    # This class is needed for `MutableHashTable(checkpoint=True)`.
  class _Saveable(BaseSaverBuilder.SaveableObject):
    # [CLASS DOC]
    # HELPER: legacy saveable object used when a checkpointed mutable table is saved or restored
    # through the old saver; it saves the exported keys and values as two named specs.
    # KNOWN ISSUE: the docstring below names DenseHashTable, but this class belongs to
    # MutableHashTable (copy and paste error).
    """SaveableObject implementation for DenseHashTable."""

    def __init__(self, table, name, table_name=None):
      # [FUNCTION DOC]
      # HELPER: builds two save specs from the table's exported keys and values, named with the
      # given name plus keys and values suffixes, remembers the table name (defaults to the
      # name) and calls the saveable base constructor. Exports the table, creating ops.
      tensors = table.export()
      specs = [
          BaseSaverBuilder.SaveSpec(tensors[0], "", name + "-keys"),
          BaseSaverBuilder.SaveSpec(tensors[1], "", name + "-values")
      ]
      self.table_name = table_name or name
      # pylint: disable=protected-access
      super(MutableHashTable._Saveable, self).__init__(table, specs, name)

    def restore(self, restored_tensors, restored_shapes):
      # [FUNCTION DOC]
      # HELPER: restores the table from a pair (keys, values) of restored tensors by bulk
      # import on the table's device. The shapes argument is ignored. Returns the import op.
      del restored_shapes  # unused
      # pylint: disable=protected-access
      with ops.name_scope("%s_table_restore" % self.table_name):
        with ops.colocate_with(self.op.resource_handle):
          return gen_lookup_ops.lookup_table_import_v2(self.op.resource_handle,
                                                       restored_tensors[0],
                                                       restored_tensors[1])


@tf_export("lookup.experimental.DenseHashTable")
@saveable_compat.legacy_saveable_name("table")
class DenseHashTable(LookupInterface):
  # [CLASS DOC]
  # Mutable open-addressing hash table with faster operations and larger memory use than the
  # mutable hash table. Needs two reserved keys (empty and deleted) that may never be inserted,
  # removed or looked up. Holds default value, dtypes, reserved keys, initial bucket count, name,
  # checkpoint and anonymity flags. Supports insert (or assign), erase, remove, lookup, size,
  # export, checkpoint hooks and a nested saveable for legacy checkpoints.
  """A mutable hash table with faster lookups and higher memory usage.

  Data can be inserted by calling the `insert` method and removed by calling the
  `remove` method. It does not support initialization via the init method.

  Compared to `MutableHashTable`, `DenseHashTable` offers generally faster
  `insert`, `remove` and `lookup` operations, in exchange for a higher overall
  memory footprint.

  It uses "open addressing" with quadratic reprobing to resolve collisions. This
  requires specifying two keys in the key space, `empty_key` and `deleted_key`,
  that can never inserted into the table.

  Unlike `MutableHashTable`, `DenseHashTable` does not require additional memory
  for temporary tensors created during checkpointing and restore operations.

  Example usage:

  >>> table = tf.lookup.experimental.DenseHashTable(
  ...     key_dtype=tf.string,
  ...     value_dtype=tf.int64,
  ...     default_value=-1,
  ...     empty_key='',
  ...     deleted_key='$')
  >>> keys = tf.constant(['a', 'b', 'c'])
  >>> values = tf.constant([0, 1, 2], dtype=tf.int64)
  >>> table.insert(keys, values)
  >>> table.remove(tf.constant(['c']))
  >>> table.lookup(tf.constant(['a', 'b', 'c','d'])).numpy()
  array([ 0,  1, -1, -1])
  """

  # TODO(andreasst): consider extracting common code with MutableHashTable into
  # a common superclass.
  def __init__(self,
               key_dtype,
               value_dtype,
               default_value,
               empty_key,
               deleted_key,
               initial_num_buckets=None,
               name="MutableDenseHashTable",
               checkpoint=True,
               experimental_is_anonymous=False):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it returns a table object and raises a value error when
    # checkpoint is true and no name is given; a constructor returns nothing and no such check
    # exists in the code (the name has a default).
    # Creates an empty dense table. Inputs: key and value dtypes, default_value (converted to a
    # tensor of the value dtype), empty_key and deleted_key (reserved keys, converted to the key
    # dtype when the resource is created), initial_num_buckets (optional; the kernel default is
    # large), name (default "MutableDenseHashTable"), checkpoint flag (default true) and anonymous
    # flag (default false). Not anonymous: in eager mode a unique shared name is generated (can
    # leak memory per the source comment). Side effects: creates the table resource and, if
    # checkpointing, a saveable object added to the saveable-objects collection in graph mode.
    """Creates an empty `DenseHashTable` object.

    Creates a table, the type of its keys and values are specified by key_dtype
    and value_dtype, respectively.

    Args:
      key_dtype: the type of the key tensors.
      value_dtype: the type of the value tensors.
      default_value: The value to use if a key is missing in the table.
      empty_key: the key to use to represent empty buckets internally. Must not
        be used in insert, remove or lookup operations.
      deleted_key: the key to use to represent deleted buckets internally. Must
        not be used in insert, remove or lookup operations and be different from
        the empty_key.
      initial_num_buckets: the initial number of buckets (optional,
        default to 2^17=131072). Note that the default value is
        relatively large (~1MB), so if you are going to create many
        tables (likely the case when `experimental_is_anonymous` is
        `True`), you should set `initial_num_buckets` to a smaller
        value to reduce memory usage.
      name: A name for the operation (optional).
      checkpoint: if True, the contents of the table are saved to and restored
        from checkpoints. If `shared_name` is empty for a checkpointed table, it
        is shared using the table node name.
      experimental_is_anonymous: Whether to use anonymous mode for the
        table (default is False). In anonymous mode, the table
        resource can only be accessed via a resource handle. It can't
        be looked up by a name. When all resource handles pointing to
        that resource are gone, the resource will be deleted
        automatically.

    Returns:
      A `DenseHashTable` object.

    Raises:
      ValueError: If checkpoint is True and no name was specified.
    """
    self._default_value = ops.convert_to_tensor(
        default_value, dtype=value_dtype, name="default_value")
    self._key_dtype = key_dtype
    self._value_dtype = value_dtype
    # TODO(b/201578996): Pick a good default for initial_num_buckets
    #   other than 2^17.
    self._initial_num_buckets = initial_num_buckets
    self._value_shape = self._default_value.get_shape()
    self._checkpoint = checkpoint
    self._name = name
    self._empty_key = empty_key
    self._deleted_key = deleted_key
    self._is_anonymous = experimental_is_anonymous
    if not self._is_anonymous:
      self._shared_name = None
      if context.executing_eagerly():
        # TODO(allenl): This will leak memory due to kernel caching by
        # the shared_name attribute value (but is better than the
        # alternative of sharing everything by default when executing
        # eagerly; hopefully creating tables in a loop is uncommon).
        self._shared_name = "table_%d" % (ops.uid(),)
    super(DenseHashTable, self).__init__(key_dtype, value_dtype)
    self._resource_handle = self._create_resource()
    if checkpoint:
      saveable = DenseHashTable._Saveable(self, name)
      if not context.executing_eagerly():
        ops.add_to_collection(ops.GraphKeys.SAVEABLE_OBJECTS, saveable)

  def _create_resource(self):
    # [FUNCTION DOC]
    # HELPER: converts the empty and deleted keys to tensors of the key dtype and creates the
    # dense table resource using the anonymous or named kernel (named tables use node-name
    # sharing when checkpointing without a shared name). Records the table name (last component
    # of the op name in graph mode, nothing in eager mode). Returns the resource handle.
    empty_key = ops.convert_to_tensor(
        self._empty_key, dtype=self._key_dtype, name="empty_key")
    deleted_key = ops.convert_to_tensor(
        self._deleted_key, dtype=self._key_dtype, name="deleted_key")
    if self._is_anonymous:
      table_ref = gen_lookup_ops.anonymous_mutable_dense_hash_table(
          empty_key=empty_key,
          deleted_key=deleted_key,
          value_dtype=self._value_dtype,
          value_shape=self._value_shape,
          initial_num_buckets=self._initial_num_buckets,
          name=self._name)
    else:
      # The table must be shared if checkpointing is requested for multi-worker
      # training to work correctly. Use the node name if no shared_name has been
      # explicitly specified.
      use_node_name_sharing = self._checkpoint and self._shared_name is None
      table_ref = gen_lookup_ops.mutable_dense_hash_table_v2(
          empty_key=empty_key,
          deleted_key=deleted_key,
          shared_name=self._shared_name,
          use_node_name_sharing=use_node_name_sharing,
          value_dtype=self._value_dtype,
          value_shape=self._value_shape,
          initial_num_buckets=self._initial_num_buckets,
          name=self._name)
    if context.executing_eagerly():
      self._table_name = None
    else:
      self._table_name = table_ref.op.name.split("/")[-1]
    return table_ref

  @property
  def name(self):
    # [FUNCTION DOC]
    # Read-only property: the table's op name (last path component), nothing in eager mode.
    return self._table_name

  def size(self, name=None):
    # [FUNCTION DOC]
    # Returns a scalar int64 tensor with the number of entries, computed on the device holding
    # the table resource. Optional op name.
    """Compute the number of elements in this table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A scalar tensor containing the number of elements in this table.
    """
    with ops.name_scope(name, "%s_Size" % self.name, [self.resource_handle]):
      with ops.colocate_with(self.resource_handle):
        return gen_lookup_ops.lookup_table_size_v2(self.resource_handle)

  def lookup(self, keys, name=None):
    # [FUNCTION DOC]
    # Looks up keys (anything convertible to the key dtype, any shape) and returns values with
    # the keys' shape plus the value shape; missing keys get the table default value. Runs on the
    # table's device. Optional op name. Conversion problems surface as type errors.
    """Looks up `keys` in a table, outputs the corresponding values.

    The `default_value` is used for keys not present in the table.

    Args:
      keys: Keys to look up. Can be a tensor of any shape. Must match the
        table's key_dtype.
      name: A name for the operation (optional).

    Returns:
      A tensor containing the values in the same shape as `keys` using the
        table's value type.

    Raises:
      TypeError: when `keys` do not match the table data types.
    """
    with ops.name_scope(name, "%s_lookup_table_find" % self.name,
                        [self.resource_handle, keys]):
      keys = ops.convert_to_tensor(keys, dtype=self._key_dtype, name="keys")
      with ops.colocate_with(self.resource_handle):
        values = gen_lookup_ops.lookup_table_find_v2(self.resource_handle, keys,
                                                     self._default_value)

    return values

  def insert_or_assign(self, keys, values, name=None):
    # [FUNCTION DOC]
    # Inserts or overwrites entries. Inputs: keys and values (convertible to the table dtypes;
    # values must match keys' shape plus value shape) and an optional op name. Returns the
    # insertion operation. Conversion errors surface as type errors. Mutates the table when run.
    """Associates `keys` with `values`.

    Args:
      keys: Keys to insert. Can be a tensor of any shape. Must match the table's
        key type.
      values: Values to be associated with keys. Must be a tensor of the same
        shape as `keys` and match the table's value type.
      name: A name for the operation (optional).

    Returns:
      The created Operation.

    Raises:
      TypeError: when `keys` or `values` doesn't match the table data
        types.
    """
    with ops.name_scope(name, "%s_lookup_table_insert" % self.name,
                        [self.resource_handle, keys, values]):
      keys = ops.convert_to_tensor(keys, dtype=self._key_dtype, name="keys")
      values = ops.convert_to_tensor(
          values, dtype=self._value_dtype, name="values")
      with ops.colocate_with(self.resource_handle):
        op = gen_lookup_ops.lookup_table_insert_v2(self.resource_handle, keys,
                                                   values)
      return op

  def insert(self, keys, values, name=None):
    # [FUNCTION DOC]
    # Alias of insert_or_assign: passes keys, values and the optional name through and returns
    # the insertion operation.
    """Associates `keys` with `values`.

    Args:
      keys: Keys to insert. Can be a tensor of any shape. Must match the table's
        key type.
      values: Values to be associated with keys. Must be a tensor of the same
        shape as `keys` and match the table's value type.
      name: A name for the operation (optional).

    Returns:
      The created Operation.

    Raises:
      TypeError: when `keys` or `values` doesn't match the table data
        types.
    """
    return self.insert_or_assign(keys, values, name)

  def erase(self, keys, name=None):
    # [FUNCTION DOC]
    # Deletes the given keys (tensor with the table's key dtype) and their values; absent keys
    # are ignored. Optional op name. Returns the removal operation. Raises a type error if the
    # keys dtype differs from the table's key dtype. Mutates the table when run. Unlike lookup,
    # keys are not converted, so a plain python list is not accepted.
    """Removes `keys` and its associated values from the table.

    If a key is not present in the table, it is silently ignored.

    Args:
      keys: Keys to remove. Can be a tensor of any shape. Must match the table's
        key type.
      name: A name for the operation (optional).

    Returns:
      The created Operation.

    Raises:
      TypeError: when `keys` do not match the table data types.
    """
    if keys.dtype != self._key_dtype:
      raise TypeError("Signature mismatch. Keys must be dtype %s, got %s." %
                      (self._key_dtype, keys.dtype))

    with ops.name_scope(name, "%s_lookup_table_remove" % self.name,
                        (self.resource_handle, keys, self._default_value)):
      # pylint: disable=protected-access
      op = gen_lookup_ops.lookup_table_remove_v2(self.resource_handle, keys)

    return op

  def remove(self, keys, name=None):
    # [FUNCTION DOC]
    # Alias of erase: passes keys and the optional name through and returns the removal
    # operation; raises a type error on key dtype mismatch.
    """Removes `keys` and its associated values from the table.

    If a key is not present in the table, it is silently ignored.

    Args:
      keys: Keys to remove. Can be a tensor of any shape. Must match the table's
        key type.
      name: A name for the operation (optional).

    Returns:
      The created Operation.

    Raises:
      TypeError: when `keys` do not match the table data types.
    """
    return self.erase(keys, name)

  def export(self, name=None):
    # [FUNCTION DOC]
    # Exports every entry using the export kernel on the table's device. Optional op name.
    # Returns a pair (all keys tensor, all values tensor).
    """Returns tensors of all keys and values in the table.

    Args:
      name: A name for the operation (optional).

    Returns:
      A pair of tensors with the first tensor containing all keys and the
        second tensors containing all values in the table.
    """
    with ops.name_scope(name, "%s_lookup_table_export_values" % self.name,
                        [self.resource_handle]):
      with ops.colocate_with(self.resource_handle):
        exported_keys, exported_values = gen_lookup_ops.lookup_table_export_v2(
            self.resource_handle, self._key_dtype, self._value_dtype)

    return exported_keys, exported_values

  def _serialize_to_tensors(self):
    # [FUNCTION DOC]
    # HELPER: checkpoint hook. Exports the table and returns a mapping from the names for keys
    # and for values to the exported key and value tensors.
    """Implements checkpointing interface in `Trackable`."""
    tensors = self.export()
    return {"-keys": tensors[0], "-values": tensors[1]}

  def _restore_from_tensors(self, restored_tensors):
    # [FUNCTION DOC]
    # HELPER: checkpoint hook. Takes a mapping holding restored keys and values tensors and
    # bulk-imports them into the table on its device; returns the import operation.
    """Implements checkpointing interface in `Trackable`."""
    with ops.name_scope("%s_table_restore" % self._name):
      with ops.colocate_with(self.resource_handle):
        return gen_lookup_ops.lookup_table_import_v2(
            self.resource_handle,
            restored_tensors["-keys"],
            restored_tensors["-values"])

  def _copy_trackable_to_cpu(self, object_map):
    # [FUNCTION DOC]
    # HELPER: checkpoint hook. Given a mapping from original objects to their copies, creates a
    # new table with the same configuration for this object if none exists yet, then copies
    # this table's contents into that copy. Mutates the mapping. Returns nothing.
    """Implements checkpointing protocols for `Trackable`."""
    if self not in object_map:
      # If self is not already populated in object map, instantiate the copy
      object_map[self] = DenseHashTable(
          self._key_dtype,
          self._value_dtype,
          self._default_value,
          self._empty_key,
          self._deleted_key,
          self._initial_num_buckets,
          self._name,
          self._checkpoint,
          self._is_anonymous
      )

    # Copy values from `self` to copy of `self`
    serialized = self._serialize_to_tensors()
    object_map[self]._restore_from_tensors(serialized)  # pylint: disable=protected-access

  # This class is needed for `DenseHashTable(checkpoint=True)`.
  class _Saveable(BaseSaverBuilder.SaveableObject):
    # [CLASS DOC]
    # HELPER: legacy saveable object used when a checkpointed dense table is saved or restored
    # through the old saver; it saves the exported keys and values as two named specs.
    """SaveableObject implementation for DenseHashTable."""

    def __init__(self, table, name, table_name=None):
      # [FUNCTION DOC]
      # HELPER: builds two save specs from the table's exported keys and values, named with the
      # given name plus keys and values suffixes, remembers the table name (defaults to the
      # name) and calls the saveable base constructor. Exports the table, creating ops.
      tensors = table.export()
      specs = [
          BaseSaverBuilder.SaveSpec(tensors[0], "", name + "-keys"),
          BaseSaverBuilder.SaveSpec(tensors[1], "", name + "-values")
      ]
      self.table_name = table_name or name
      # pylint: disable=protected-access
      super(DenseHashTable._Saveable, self).__init__(table, specs, name)

    def restore(self, restored_tensors, restored_shapes):
      # [FUNCTION DOC]
      # HELPER: restores the table from a pair (keys, values) of restored tensors by bulk
      # import on the table's device. The shapes argument is ignored. Returns the import op.
      del restored_shapes  # unused
      # pylint: disable=protected-access
      with ops.name_scope("%s_table_restore" % self.table_name):
        with ops.colocate_with(self.op.resource_handle):
          return gen_lookup_ops.lookup_table_import_v2(self.op.resource_handle,
                                                       restored_tensors[0],
                                                       restored_tensors[1])
