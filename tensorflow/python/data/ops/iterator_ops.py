# [MODULE DOC]
# Python wrappers for tf.data iterators. Contains the graph-mode (v1) Iterator class built from
# an iterator resource tensor, the eager/function-friendly OwnedIterator and its multi-device
# variant, a type spec describing iterators, checkpoint saveable support, and the helper that
# decides whether the device stack is empty. Also defines the call-count thresholds and warning
# messages used to detect get-next misuse, and the graph collection name for iterators.
# Import-time side effects: registers a type spec, the v1 iterator saveable name, public API
# exports, and the autograph overrides for iterators (registered at the end of the module).
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
"""Python wrappers for Iterators."""

import abc
import threading
import warnings

from tensorflow.core.protobuf import struct_pb2
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.checkpoint import saveable_compat
from tensorflow.python.data.ops import iterator_autograph
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.util import nest
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_utils
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_string_ops
from tensorflow.python.saved_model import nested_structure_coder
from tensorflow.python.trackable import base as trackable
from tensorflow.python.training.saver import BaseSaverBuilder
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export


# NOTE(mrry): It is legitimate to call `Iterator.get_next()` multiple
# times, e.g. when you are distributing different elements to multiple
# devices in a single step. However, a common pitfall arises when
# users call `Iterator.get_next()` in each iteration of their training
# loop. `Iterator.get_next()` adds ops to the graph, and executing
# each op allocates resources (including threads); as a consequence,
# invoking it in every iteration of a training loop causes slowdown
# and eventual resource exhaustion. To guard against this outcome, we
# log a warning when the number of uses crosses a threshold of suspicion.
GET_NEXT_CALL_WARNING_THRESHOLD = 32

GET_NEXT_CALL_WARNING_MESSAGE = (
    "An unusually high number of `Iterator.get_next()` calls was detected. "
    "This often indicates that `Iterator.get_next()` is being called inside "
    "a training loop, which will cause gradual slowdown and eventual resource "
    "exhaustion. If this is the case, restructure your code to call "
    "`next_element = iterator.get_next()` once outside the loop, and use "
    "`next_element` as the input to some computation that is invoked inside "
    "the loop.")

# NOTE(jsimsa): Threshold used as a heuristic to check for infinite loop during
# tf.function tracing.
GET_NEXT_CALL_ERROR_THRESHOLD = 32

GET_NEXT_CALL_ERROR_MESSAGE = (
    "An unusually high number of `tf.data.Iterator.get_next()` calls was "
    "detected. This suggests that the `for elem in dataset: ...` idiom is used "
    "within tf.function with AutoGraph disabled. This idiom is only supported "
    "when AutoGraph is enabled.")

# Collection of all IteratorResources in the `Graph`.
GLOBAL_ITERATORS = "iterators"


def _device_stack_is_empty():
  # [FUNCTION DOC]
  # HELPER: tells whether any device placement scope is active, used when creating iterators.
  # Takes no inputs. In eager mode returns true when no device name is set on the eager context;
  # in graph mode returns true when the default graph has an empty device function stack.
  # Returns a boolean; no side effects.
  if context.executing_eagerly():
    return context.context().device_name is None
  # pylint: disable=protected-access
  device_stack = ops.get_default_graph()._device_functions_outer_to_inner
  # pylint: enable=protected-access
  return not bool(device_stack)


@saveable_compat.legacy_saveable_name("ITERATOR")
@tf_export(v1=["data.Iterator"])
class Iterator(trackable.Trackable):
  # [CLASS DOC]
  # Graph-mode (v1) iterator over a dataset, exported as the v1 data Iterator API.
  # Wraps an iterator resource tensor plus an optional initializer operation and the element
  # spec derived from legacy output types, shapes and classes. Also keeps the string handle and a
  # counter of get-next calls used to warn about misuse. Normally created through dataset
  # methods or the from-structure and from-string-handle constructors. Registered as trackable
  # with the saveable name ITERATOR.
  """Represents the state of iterating through a `Dataset`."""

  def __init__(self, iterator_resource, initializer, output_types,
               output_shapes, output_classes):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says a TypeError is raised for missing structure arguments, but
    # the code raises a ValueError.
    # Inputs: iterator_resource (scalar resource tensor); initializer (operation initializing the
    # iterator, or none); output_types, output_shapes, output_classes (nested structures, all
    # required). Raises ValueError if any of the three is none. Builds the element spec and
    # flat shapes/types, creates the string handle tensor, resets the get-next counter, and adds
    # the resource to the global iterators graph collection (side effect).
    """Creates a new iterator from the given iterator resource.

    Note: Most users will not call this initializer directly, and will
    instead use `Dataset.make_initializable_iterator()` or
    `Dataset.make_one_shot_iterator()`.

    Args:
      iterator_resource: A `tf.resource` scalar `tf.Tensor` representing the
        iterator.
      initializer: A `tf.Operation` that should be run to initialize this
        iterator.
      output_types: A (nested) structure of `tf.DType` objects corresponding to
        each component of an element of this iterator.
      output_shapes: A (nested) structure of `tf.TensorShape` objects
        corresponding to each component of an element of this iterator.
      output_classes: A (nested) structure of Python `type` objects
        corresponding to each component of an element of this iterator.

    Raises:
      TypeError: If `output_types`, `output_shapes`, or `output_classes` is not
        specified.
    """
    self._iterator_resource = iterator_resource
    self._initializer = initializer

    if (output_types is None or output_shapes is None
        or output_classes is None):
      raise ValueError(
          "All of `output_types`, `output_shapes`, and `output_classes` "
          "must be specified to create an iterator. Got "
          f"`output_types` = {output_types!r}, "
          f"`output_shapes` = {output_shapes!r}, "
          f"`output_classes` = {output_classes!r}.")
    self._element_spec = structure.convert_legacy_structure(
        output_types, output_shapes, output_classes)
    self._flat_tensor_shapes = structure.get_flat_tensor_shapes(
        self._element_spec)
    self._flat_tensor_types = structure.get_flat_tensor_types(
        self._element_spec)

    self._string_handle = gen_dataset_ops.iterator_to_string_handle(
        self._iterator_resource)
    self._get_next_call_count = 0
    ops.add_to_collection(GLOBAL_ITERATORS, self._iterator_resource)

  @staticmethod
  def from_structure(output_types,
                     output_shapes=None,
                     shared_name=None,
                     output_classes=None):
    # [FUNCTION DOC]
    # Static constructor for an uninitialized iterator not bound to any dataset.
    # Inputs: output_types (nested dtypes, required); output_shapes (optional, defaults to unknown
    # shapes); shared_name (optional string, default empty, to share across sessions);
    # output_classes (optional, defaults to tensor class for each component). Raises an error if
    # the structures of types and shapes differ. Creates an iterator resource op and returns an
    # Iterator with no initializer.
    """Creates a new, uninitialized `Iterator` with the given structure.

    This iterator-constructing method can be used to create an iterator that
    is reusable with many different datasets.

    The returned iterator is not bound to a particular dataset, and it has
    no `initializer`. To initialize the iterator, run the operation returned by
    `Iterator.make_initializer(dataset)`.

    The following is an example

    ```python
    iterator = Iterator.from_structure(tf.int64, tf.TensorShape([]))

    dataset_range = Dataset.range(10)
    range_initializer = iterator.make_initializer(dataset_range)

    dataset_evens = dataset_range.filter(lambda x: x % 2 == 0)
    evens_initializer = iterator.make_initializer(dataset_evens)

    # Define a model based on the iterator; in this example, the model_fn
    # is expected to take scalar tf.int64 Tensors as input (see
    # the definition of 'iterator' above).
    prediction, loss = model_fn(iterator.get_next())

    # Train for `num_epochs`, where for each epoch, we first iterate over
    # dataset_range, and then iterate over dataset_evens.
    for _ in range(num_epochs):
      # Initialize the iterator to `dataset_range`
      sess.run(range_initializer)
      while True:
        try:
          pred, loss_val = sess.run([prediction, loss])
        except tf.errors.OutOfRangeError:
          break

      # Initialize the iterator to `dataset_evens`
      sess.run(evens_initializer)
      while True:
        try:
          pred, loss_val = sess.run([prediction, loss])
        except tf.errors.OutOfRangeError:
          break
    ```

    Args:
      output_types: A (nested) structure of `tf.DType` objects corresponding to
        each component of an element of this dataset.
      output_shapes: (Optional.) A (nested) structure of `tf.TensorShape`
        objects corresponding to each component of an element of this dataset.
        If omitted, each component will have an unconstrainted shape.
      shared_name: (Optional.) If non-empty, this iterator will be shared under
        the given name across multiple sessions that share the same devices
        (e.g. when using a remote server).
      output_classes: (Optional.) A (nested) structure of Python `type` objects
        corresponding to each component of an element of this iterator. If
        omitted, each component is assumed to be of type `tf.Tensor`.

    Returns:
      An `Iterator`.

    Raises:
      TypeError: If the structures of `output_shapes` and `output_types` are
        not the same.
    """
    output_types = nest.map_structure(dtypes.as_dtype, output_types)
    if output_shapes is None:
      output_shapes = nest.map_structure(
          lambda _: tensor_shape.TensorShape(None), output_types)
    else:
      output_shapes = nest.map_structure_up_to(output_types,
                                               tensor_shape.as_shape,
                                               output_shapes)
    if output_classes is None:
      output_classes = nest.map_structure(lambda _: tensor.Tensor, output_types)
    nest.assert_same_structure(output_types, output_shapes)
    output_structure = structure.convert_legacy_structure(
        output_types, output_shapes, output_classes)
    if shared_name is None:
      shared_name = ""
    iterator_resource = gen_dataset_ops.iterator_v2(
        container="",
        shared_name=shared_name,
        output_types=structure.get_flat_tensor_types(output_structure),
        output_shapes=structure.get_flat_tensor_shapes(
            output_structure))
    return Iterator(iterator_resource, None, output_types, output_shapes,
                    output_classes)

  @staticmethod
  def from_string_handle(string_handle,
                         output_types,
                         output_shapes=None,
                         output_classes=None):
    # [FUNCTION DOC]
    # Static constructor for a feedable iterator based on a string handle.
    # Inputs: string_handle (scalar string tensor or convertible); output_types (required nested
    # dtypes); output_shapes (optional, default unknown); output_classes (optional, default
    # tensor). Raises an error if types and shapes structures differ. Creates an iterator
    # resource from the handle and returns an Iterator without an initializer.
    """Creates a new, uninitialized `Iterator` based on the given handle.

    This method allows you to define a "feedable" iterator where you can choose
    between concrete iterators by feeding a value in a `tf.Session.run` call.
    In that case, `string_handle` would be a `tf.compat.v1.placeholder`, and you
    would
    feed it with the value of `tf.data.Iterator.string_handle` in each step.

    For example, if you had two iterators that marked the current position in
    a training dataset and a test dataset, you could choose which to use in
    each step as follows:

    ```python
    train_iterator = tf.data.Dataset(...).make_one_shot_iterator()
    train_iterator_handle = sess.run(train_iterator.string_handle())

    test_iterator = tf.data.Dataset(...).make_one_shot_iterator()
    test_iterator_handle = sess.run(test_iterator.string_handle())

    handle = tf.compat.v1.placeholder(tf.string, shape=[])
    iterator = tf.data.Iterator.from_string_handle(
        handle, train_iterator.output_types)

    next_element = iterator.get_next()
    loss = f(next_element)

    train_loss = sess.run(loss, feed_dict={handle: train_iterator_handle})
    test_loss = sess.run(loss, feed_dict={handle: test_iterator_handle})
    ```

    Args:
      string_handle: A scalar `tf.Tensor` of type `tf.string` that evaluates to
        a handle produced by the `Iterator.string_handle()` method.
      output_types: A (nested) structure of `tf.DType` objects corresponding to
        each component of an element of this dataset.
      output_shapes: (Optional.) A (nested) structure of `tf.TensorShape`
        objects corresponding to each component of an element of this dataset.
        If omitted, each component will have an unconstrainted shape.
      output_classes: (Optional.) A (nested) structure of Python `type` objects
        corresponding to each component of an element of this iterator. If
        omitted, each component is assumed to be of type `tf.Tensor`.

    Returns:
      An `Iterator`.
    """
    output_types = nest.map_structure(dtypes.as_dtype, output_types)
    if output_shapes is None:
      output_shapes = nest.map_structure(
          lambda _: tensor_shape.TensorShape(None), output_types)
    else:
      output_shapes = nest.map_structure_up_to(output_types,
                                               tensor_shape.as_shape,
                                               output_shapes)
    if output_classes is None:
      output_classes = nest.map_structure(lambda _: tensor.Tensor, output_types)
    nest.assert_same_structure(output_types, output_shapes)
    output_structure = structure.convert_legacy_structure(
        output_types, output_shapes, output_classes)
    string_handle = ops.convert_to_tensor(string_handle, dtype=dtypes.string)
    iterator_resource = gen_dataset_ops.iterator_from_string_handle_v2(
        string_handle,
        output_types=structure.get_flat_tensor_types(output_structure),
        output_shapes=structure.get_flat_tensor_shapes(output_structure))
    return Iterator(iterator_resource, None, output_types, output_shapes,
                    output_classes)

  @property
  def initializer(self):
    # [FUNCTION DOC]
    # Property returning the initializer operation. Raises a ValueError when the iterator has
    # none (for example one-shot iterators).
    """A `tf.Operation` that should be run to initialize this iterator.

    Returns:
      A `tf.Operation` that should be run to initialize this iterator

    Raises:
      ValueError: If this iterator initializes itself automatically.
    """
    if self._initializer is not None:
      return self._initializer
    else:
      # TODO(mrry): Consider whether one-shot iterators should have
      # initializers that simply reset their state to the beginning.
      raise ValueError(
          "The iterator does not have an initializer. This means it was likely "
          "created using `tf.data.Dataset.make_one_shot_iterator()`. For an "
          "initializable iterator, use "
          "`tf.data.Dataset.make_initializable_iterator()` instead.")

  def make_initializer(self, dataset, name=None):
    # [FUNCTION DOC]
    # Inputs: dataset (dataset whose element spec must match this iterator); name (optional
    # operation name). Compares the legacy output types, shapes and classes of the dataset with
    # the iterator's: structure mismatch raises an error from the nest check, and a class, dtype
    # or incompatible shape mismatch raises a TypeError. Returns an operation, created under a
    # colocation constraint with the iterator resource, that initializes the iterator on the
    # dataset.
    """Returns a `tf.Operation` that initializes this iterator on `dataset`.

    Args:
      dataset: A `Dataset` whose `element_spec` if compatible with this
        iterator.
      name: (Optional.) A name for the created operation.

    Returns:
      A `tf.Operation` that can be run to initialize this iterator on the given
      `dataset`.

    Raises:
      TypeError: If `dataset` and this iterator do not have a compatible
        `element_spec`.
    """
    with ops.name_scope(name, "make_initializer") as name:
      # NOTE(mrry): Cannot depend on `dataset_ops.get_legacy_output*()` due
      # to that creating a circular dependency.
      # pylint: disable=protected-access
      dataset_output_types = nest.map_structure(
          lambda component_spec: component_spec._to_legacy_output_types(),
          dataset.element_spec)
      dataset_output_shapes = nest.map_structure(
          lambda component_spec: component_spec._to_legacy_output_shapes(),
          dataset.element_spec)
      dataset_output_classes = nest.map_structure(
          lambda component_spec: component_spec._to_legacy_output_classes(),
          dataset.element_spec)
      # pylint: enable=protected-access

      nest.assert_same_structure(self.output_types, dataset_output_types)
      nest.assert_same_structure(self.output_shapes, dataset_output_shapes)
      for iterator_class, dataset_class in zip(
          nest.flatten(self.output_classes),
          nest.flatten(dataset_output_classes)):
        if iterator_class is not dataset_class:
          raise TypeError(
              f"Expected output classes {self.output_classes!r} but got "
              f"dataset with output classes {dataset_output_classes!r}.")
      for iterator_dtype, dataset_dtype in zip(
          nest.flatten(self.output_types), nest.flatten(dataset_output_types)):
        if iterator_dtype != dataset_dtype:
          raise TypeError(
              f"Expected output types {self.output_types!r} but got dataset "
              f"with output types {dataset_output_types!r}.")
      for iterator_shape, dataset_shape in zip(
          nest.flatten(self.output_shapes), nest.flatten(
              dataset_output_shapes)):
        if not iterator_shape.is_compatible_with(dataset_shape):
          raise TypeError(
              f"Expected output shapes compatible with {self.output_shapes!r} "
              f"but got dataset with output shapes {dataset_output_shapes!r}.")

    # TODO(b/169442955): Investigate the need for this colocation constraint.
    with ops.colocate_with(self._iterator_resource):
      # pylint: disable=protected-access
      return gen_dataset_ops.make_iterator(
          dataset._variant_tensor, self._iterator_resource, name=name)

  def get_next(self, name=None):
    # [FUNCTION DOC]
    # Input: name (optional operation name). Increments the get-next call counter and issues a
    # warning once it exceeds the threshold. Adds a get-next op (colocated with the iterator
    # resource) and returns the result packed into the element structure. Execution of the op
    # raises an out-of-range error when the iterator is exhausted.
    """Returns the next element.

    In graph mode, you should typically call this method *once* and use its
    result as the input to another computation. A typical loop will then call
    `tf.Session.run` on the result of that computation. The loop will terminate
    when the `Iterator.get_next()` operation raises
    `tf.errors.OutOfRangeError`. The following skeleton shows how to use
    this method when building a training loop:

    ```python
    dataset = ...  # A `tf.data.Dataset` object.
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()

    # Build a TensorFlow graph that does something with each element.
    loss = model_function(next_element)
    optimizer = ...  # A `tf.compat.v1.train.Optimizer` object.
    train_op = optimizer.minimize(loss)

    with tf.compat.v1.Session() as sess:
      try:
        while True:
          sess.run(train_op)
      except tf.errors.OutOfRangeError:
        pass
    ```

    NOTE: It is legitimate to call `Iterator.get_next()` multiple times, e.g.
    when you are distributing different elements to multiple devices in a single
    step. However, a common pitfall arises when users call `Iterator.get_next()`
    in each iteration of their training loop. `Iterator.get_next()` adds ops to
    the graph, and executing each op allocates resources (including threads); as
    a consequence, invoking it in every iteration of a training loop causes
    slowdown and eventual resource exhaustion. To guard against this outcome, we
    log a warning when the number of uses crosses a fixed threshold of
    suspiciousness.

    Args:
      name: (Optional.) A name for the created operation.

    Returns:
      A (nested) structure of values matching `tf.data.Iterator.element_spec`.
    """
    self._get_next_call_count += 1
    if self._get_next_call_count > GET_NEXT_CALL_WARNING_THRESHOLD:
      warnings.warn(GET_NEXT_CALL_WARNING_MESSAGE)

    # TODO(b/169442955): Investigate the need for this colocation constraint.
    with ops.colocate_with(self._iterator_resource):
      # pylint: disable=protected-access
      flat_ret = gen_dataset_ops.iterator_get_next(
          self._iterator_resource,
          output_types=self._flat_tensor_types,
          output_shapes=self._flat_tensor_shapes,
          name=name)
      return structure.from_tensor_list(self._element_spec, flat_ret)

  def get_next_as_optional(self):
    # [FUNCTION DOC]
    # Returns an optional value wrapping the next element (empty when the iterator is exhausted,
    # instead of raising an error). Takes no inputs; the op is colocated with the iterator
    # resource. Note that it does not count towards the get-next warning counter.
    # TODO(b/169442955): Investigate the need for this colocation constraint.
    with ops.colocate_with(self._iterator_resource):
      # pylint: disable=protected-access
      return optional_ops._OptionalImpl(
          gen_dataset_ops.iterator_get_next_as_optional(
              self._iterator_resource,
              output_types=structure.get_flat_tensor_types(self.element_spec),
              output_shapes=structure.get_flat_tensor_shapes(
                  self.element_spec)), self.element_spec)

  def string_handle(self, name=None):
    # [FUNCTION DOC]
    # Input: name (optional). Returns a scalar string tensor identifying this iterator: the
    # cached handle created at construction when no name is given, otherwise a new op with that
    # name.
    """Returns a string-valued `tf.Tensor` that represents this iterator.

    Args:
      name: (Optional.) A name for the created operation.

    Returns:
      A scalar `tf.Tensor` of type `tf.string`.
    """
    if name is None:
      return self._string_handle
    else:
      return gen_dataset_ops.iterator_to_string_handle(
          self._iterator_resource, name=name)

  @property
  @deprecation.deprecated(
      None, "Use `tf.compat.v1.data.get_output_classes(iterator)`.")
  def output_classes(self):
    # [FUNCTION DOC]
    # Deprecated property. Returns the legacy class of each element component, derived from the
    # element spec.
    """Returns the class of each component of an element of this iterator.

    The expected values are `tf.Tensor` and `tf.sparse.SparseTensor`.

    Returns:
      A (nested) structure of Python `type` objects corresponding to each
      component of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_classes(),  # pylint: disable=protected-access
        self._element_spec)

  @property
  @deprecation.deprecated(
      None, "Use `tf.compat.v1.data.get_output_shapes(iterator)`.")
  def output_shapes(self):
    # [FUNCTION DOC]
    # Deprecated property. Returns the legacy shape of each element component, derived from the
    # element spec.
    """Returns the shape of each component of an element of this iterator.

    Returns:
      A (nested) structure of `tf.TensorShape` objects corresponding to each
      component of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_shapes(),  # pylint: disable=protected-access
        self._element_spec)

  @property
  @deprecation.deprecated(
      None, "Use `tf.compat.v1.data.get_output_types(iterator)`.")
  def output_types(self):
    # [FUNCTION DOC]
    # Deprecated property. Returns the legacy dtype of each element component, derived from the
    # element spec.
    """Returns the type of each component of an element of this iterator.

    Returns:
      A (nested) structure of `tf.DType` objects corresponding to each component
      of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_types(),  # pylint: disable=protected-access
        self._element_spec)

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the nested type spec of one element, computed at construction.
    """The type specification of an element of this iterator.

    For more information,
    read [this guide](https://www.tensorflow.org/guide/data#dataset_structure).

    Returns:
      A (nested) structure of `tf.TypeSpec` objects matching the structure of an
      element of this iterator and specifying the type of individual components.
    """

    return self._element_spec

  # override
  def _serialize_to_tensors(self):
    # [FUNCTION DOC]
    # Checkpoint support: serializes the iterator state using the fail-on-external-state policy.
    # Returns a mapping from the name for state to the serialized iterator variant tensor.
    serialized_iterator = gen_dataset_ops.serialize_iterator(
        self._iterator_resource,
        options_lib.ExternalStatePolicy.FAIL.value)
    return {"_STATE": serialized_iterator}

  # override
  def _restore_from_tensors(self, restored_tensors):
    # [FUNCTION DOC]
    # Checkpoint support: input is a mapping with the state entry holding the serialized
    # iterator. Deserializes it into the iterator resource (colocated) and returns a list with
    # the resulting op.
    with ops.colocate_with(self._iterator_resource):
      return [gen_dataset_ops.deserialize_iterator(
          self._iterator_resource, restored_tensors["_STATE"])]


_uid_counter = 0
_uid_lock = threading.Lock()


def _generate_shared_name(prefix):
  # [FUNCTION DOC]
  # HELPER: produces unique resource names. Input: prefix (string). Under a lock, increments a
  # module-global counter and returns the prefix followed by the counter value as text.
  # Side effect: advances the global counter. Appears unused within this module.
  with _uid_lock:
    global _uid_counter
    uid = _uid_counter
    _uid_counter += 1
  return "{}{}".format(prefix, uid)


@tf_export("data.Iterator", v1=[])
class IteratorBase(
    collections_abc.Iterator,
    trackable.Trackable,
    composite_tensor.CompositeTensor,
    metaclass=abc.ABCMeta):
  # [CLASS DOC]
  # Abstract base class of the public (v2) data Iterator API. It is a Python iterator, trackable
  # and composite tensor. Declares the abstract element spec property and the get-next and
  # get-next-as-optional methods that concrete iterators (the owned iterator) must implement.
  """Represents an iterator of a `tf.data.Dataset`.

  `tf.data.Iterator` is the primary mechanism for enumerating elements of a
  `tf.data.Dataset`. It supports the Python Iterator protocol, which means
  it can be iterated over using a for-loop:

  >>> dataset = tf.data.Dataset.range(2)
  >>> for element in dataset:
  ...   print(element)
  tf.Tensor(0, shape=(), dtype=int64)
  tf.Tensor(1, shape=(), dtype=int64)

  or by fetching individual elements explicitly via `get_next()`:

  >>> dataset = tf.data.Dataset.range(2)
  >>> iterator = iter(dataset)
  >>> print(iterator.get_next())
  tf.Tensor(0, shape=(), dtype=int64)
  >>> print(iterator.get_next())
  tf.Tensor(1, shape=(), dtype=int64)

  In addition, non-raising iteration is supported via `get_next_as_optional()`,
  which returns the next element (if available) wrapped in a
  `tf.experimental.Optional`.

  >>> dataset = tf.data.Dataset.from_tensors(42)
  >>> iterator = iter(dataset)
  >>> optional = iterator.get_next_as_optional()
  >>> print(optional.has_value())
  tf.Tensor(True, shape=(), dtype=bool)
  >>> optional = iterator.get_next_as_optional()
  >>> print(optional.has_value())
  tf.Tensor(False, shape=(), dtype=bool)
  """

  @abc.abstractproperty
  def element_spec(self):
    # [FUNCTION DOC]
    # Abstract property: subclasses return the nested type spec of an element. The base version
    # raises a not-implemented error.
    """The type specification of an element of this iterator.

    >>> dataset = tf.data.Dataset.from_tensors(42)
    >>> iterator = iter(dataset)
    >>> iterator.element_spec
    tf.TensorSpec(shape=(), dtype=tf.int32, name=None)

    For more information,
    read [this guide](https://www.tensorflow.org/guide/data#dataset_structure).

    Returns:
      A (nested) structure of `tf.TypeSpec` objects matching the structure of an
      element of this iterator, specifying the type of individual components.
    """
    raise NotImplementedError("Iterator.element_spec")

  @abc.abstractmethod
  def get_next(self):
    # [FUNCTION DOC]
    # Abstract method: subclasses return the next element and raise an out-of-range error at
    # the end. The base version raises a not-implemented error.
    """Returns the next element.

    >>> dataset = tf.data.Dataset.from_tensors(42)
    >>> iterator = iter(dataset)
    >>> print(iterator.get_next())
    tf.Tensor(42, shape=(), dtype=int32)

    Returns:
      A (nested) structure of values matching `tf.data.Iterator.element_spec`.

    Raises:
      `tf.errors.OutOfRangeError`: If the end of the iterator has been reached.
    """
    raise NotImplementedError("Iterator.get_next()")

  @abc.abstractmethod
  def get_next_as_optional(self):
    # [FUNCTION DOC]
    # Abstract method: subclasses return the next element wrapped in an optional that has no
    # value at the end. The base version raises a not-implemented error.
    """Returns the next element wrapped in `tf.experimental.Optional`.

    If the iterator has reached the end of the sequence, the returned
    `tf.experimental.Optional` will have no value.

    >>> dataset = tf.data.Dataset.from_tensors(42)
    >>> iterator = iter(dataset)
    >>> optional = iterator.get_next_as_optional()
    >>> print(optional.has_value())
    tf.Tensor(True, shape=(), dtype=bool)
    >>> print(optional.get_value())
    tf.Tensor(42, shape=(), dtype=int32)
    >>> optional = iterator.get_next_as_optional()
    >>> print(optional.has_value())
    tf.Tensor(False, shape=(), dtype=bool)

    Returns:
      A `tf.experimental.Optional` object representing the next element.
    """
    raise NotImplementedError("Iterator.get_next_as_optional()")


@saveable_compat.legacy_saveable_name("ITERATOR")
class OwnedIterator(IteratorBase):
  # [CLASS DOC]
  # Public (v2) iterator that owns its iterator resource; the resource lives as long as the
  # Python object, making it suitable for eager mode and tf.function. Created from a dataset
  # (creating an anonymous iterator resource and keeping a reference to the dataset) or from
  # existing resource components plus an element spec (used when rebuilding from its composite
  # tensor form). Holds the element spec, flat output types and shapes, the resource and a
  # counter used to detect infinite tracing loops. Implements the Python iterator protocol,
  # checkpoint serialization and copying to host memory.
  """An iterator producing tf.Tensor objects from a tf.data.Dataset.

  The iterator resource  created through `OwnedIterator` is owned by the Python
  object and the life time of the underlying resource is tied to the life time
  of the `OwnedIterator` object. This makes `OwnedIterator` appropriate for use
  in eager mode and inside of tf.functions.
  """

  def __init__(self, dataset=None, components=None, element_spec=None):
    # [FUNCTION DOC]
    # Inputs: dataset (optional dataset); components (optional tuple holding the iterator
    # resource); element_spec (optional nested type spec). Exactly one construction mode: with a
    # dataset, creates a new iterator from it; without, both components and element_spec must be
    # given. Raises ValueError if a dataset is missing and either of the others is missing, or if
    # a dataset is given together with components or element_spec. Resets the get-next counter.
    """Creates a new iterator from the given dataset.

    If `dataset` is not specified, the iterator will be created from the given
    tensor components and element structure. In particular, the alternative for
    constructing the iterator is used when the iterator is reconstructed from
    it `CompositeTensor` representation.

    Args:
      dataset: A `tf.data.Dataset` object.
      components: Tensor components to construct the iterator from.
      element_spec: A (nested) structure of `TypeSpec` objects that
        represents the type specification of elements of the iterator.

    Raises:
      ValueError: If `dataset` is not provided and either `components` or
        `element_spec` is not provided. Or `dataset` is provided and either
        `components` and `element_spec` is provided.
    """
    super(OwnedIterator, self).__init__()

    if dataset is None:
      if (components is None or element_spec is None):
        raise ValueError(
            "When `dataset` is not provided, both `components` and "
            "`element_spec` must be specified.")
      # pylint: disable=protected-access
      self._element_spec = element_spec
      self._flat_output_types = structure.get_flat_tensor_types(
          self._element_spec)
      self._flat_output_shapes = structure.get_flat_tensor_shapes(
          self._element_spec)
      self._components = components
      self._iterator_resource, = components
    else:
      if (components is not None or element_spec is not None):
        raise ValueError(
            "When `dataset` is provided, `element_spec` and `components` must "
            "not be specified.")
      self._create_iterator(dataset)

    self._get_next_call_count = 0

  def _create_iterator(self, dataset):
    # [FUNCTION DOC]
    # HELPER: builds the iterator resource for the constructor. Input: dataset. Applies debug
    # options to the dataset and stores a reference so it stays alive while the iterator is in
    # use (needed e.g. for generator-based datasets). Records element spec and flat types and
    # shapes, creates an anonymous iterator resource colocated with the dataset variant, in graph
    # mode attaches full type information (asserting it matches the number of flat outputs),
    # and finally creates the make-iterator op linking dataset and resource.
    # pylint: disable=protected-access
    dataset = dataset._apply_debug_options()

    # Store dataset reference to ensure that dataset is alive when this iterator
    # is being used. For example, `tf.data.Dataset.from_generator` registers
    # a few py_funcs that are needed in `self._next_internal`.  If the dataset
    # is deleted, this iterator crashes on `self.__next__(...)` call.
    self._dataset = dataset

    ds_variant = dataset._variant_tensor
    self._element_spec = dataset.element_spec
    self._flat_output_types = structure.get_flat_tensor_types(
        self._element_spec)
    self._flat_output_shapes = structure.get_flat_tensor_shapes(
        self._element_spec)
    with ops.colocate_with(ds_variant):
      self._iterator_resource = (
          gen_dataset_ops.anonymous_iterator_v3(
              output_types=self._flat_output_types,
              output_shapes=self._flat_output_shapes))
      if not context.executing_eagerly():
        # Add full type information to the graph so host memory types inside
        # variants stay on CPU, e.g, ragged string tensors.
        # TODO(b/224776031) Remove this when AnonymousIterateV3 can use
        # (reverse) type inference and all other ops that are needed to
        # provide type information to the AnonymousIterateV3 also support
        # type inference (esp. cross-function type inference) instead of
        # setting the full type information manually.
        fulltype = type_utils.iterator_full_type_from_spec(
            self._element_spec)
        # fulltype is PRODUCT[ITERATOR[PRODUCT[...]]]
        assert len(fulltype.args[0].args[0].args) == len(
            self._flat_output_types)
        self._iterator_resource.op.experimental_set_type(fulltype)
      gen_dataset_ops.make_iterator(ds_variant, self._iterator_resource)

  def __iter__(self):
    # [FUNCTION DOC]
    # Iterator protocol: returns the iterator itself.
    return self

  def next(self):  # For Python 2 compatibility
    # [FUNCTION DOC]
    # Legacy alias of the next-element method; returns the next element or raises StopIteration.
    return self.__next__()

  def _next_internal(self):
    # [FUNCTION DOC]
    # HELPER: fetches the next element. In graph mode with AutoGraph disabled, counts calls and
    # raises a ValueError once the error threshold is exceeded (suspected infinite loop while
    # tracing). In graph mode returns a get-next op result (colocated) as the element structure;
    # in eager mode runs the op synchronously and converts the result using the element spec
    # (fast path for non-nested specs). Raises an out-of-range error at the end of the data.
    autograph_status = autograph_ctx.control_status_ctx().status
    autograph_disabled = autograph_status == autograph_ctx.Status.DISABLED
    if not context.executing_eagerly() and autograph_disabled:
      self._get_next_call_count += 1
      if self._get_next_call_count > GET_NEXT_CALL_ERROR_THRESHOLD:
        raise ValueError(GET_NEXT_CALL_ERROR_MESSAGE)

    if not context.executing_eagerly():
      # TODO(b/169442955): Investigate the need for this colocation constraint.
      with ops.colocate_with(self._iterator_resource):
        ret = gen_dataset_ops.iterator_get_next(
            self._iterator_resource,
            output_types=self._flat_output_types,
            output_shapes=self._flat_output_shapes)
        return structure.from_compatible_tensor_list(self._element_spec, ret)

    # TODO(b/77291417): This runs in sync mode as iterators use an error status
    # to communicate that there is no more data to iterate over.
    with context.execution_mode(context.SYNC):
      ret = gen_dataset_ops.iterator_get_next(
          self._iterator_resource,
          output_types=self._flat_output_types,
          output_shapes=self._flat_output_shapes)

      try:
        # Fast path for the case `self._structure` is not a nested structure.
        return self._element_spec._from_compatible_tensor_list(ret)  # pylint: disable=protected-access
      except AttributeError:
        return structure.from_compatible_tensor_list(self._element_spec, ret)

  def _save(self):
    # [FUNCTION DOC]
    # Serializes the iterator state to a single string tensor. Uses the dataset's external state
    # policy option if set. Splits the serialized variant into slices, serializes each (to avoid
    # the 2GB proto limit), base64-encodes them and joins with commas. Returns that string.
    external_state_policy = None
    if (
        self._dataset
        and self._dataset.options().experimental_external_state_policy
    ):
      external_state_policy = (
          self._dataset.options().experimental_external_state_policy.value
      )
    state_variant = gen_dataset_ops.serialize_iterator(
        self._iterator_resource, external_state_policy
    )
    # Serialize each slice of the state_variant separately, to avoid hitting the
    # 2GB proto serialization limit.
    state = array_ops_stack.unstack(state_variant)
    state = [parsing_ops.serialize_tensor(x) for x in state]
    state = array_ops_stack.stack(state)

    state = string_ops.encode_base64(state)
    state = string_ops.string_join(state, separator=",")
    return state

  def _restore(self, state):
    # [FUNCTION DOC]
    # Inverse of the save method. Input: state, the comma-joined base64 string. Splits, decodes,
    # parses each slice into a variant, stacks them and deserializes into the iterator resource.
    # Returns the resulting op.
    state = ragged_string_ops.string_split_v2(state, sep=",")
    state = string_ops.decode_base64(state)

    state = array_ops_stack.unstack(state)
    state = [parsing_ops.parse_tensor(x, dtypes.variant) for x in state]
    state_variant = array_ops_stack.stack(state)
    return gen_dataset_ops.deserialize_iterator(
        self._iterator_resource, state_variant
    )

  @property
  def _type_spec(self):
    # [FUNCTION DOC]
    # Property returning the iterator type spec built from this iterator's element spec.
    return IteratorSpec(self.element_spec)

  def __next__(self):
    # [FUNCTION DOC]
    # Iterator protocol: returns the next element, converting the out-of-range error at the end
    # of the data into StopIteration.
    try:
      return self._next_internal()
    except errors.OutOfRangeError:
      raise StopIteration

  @property
  @deprecation.deprecated(
      None, "Use `tf.compat.v1.data.get_output_classes(iterator)`.")
  def output_classes(self):
    # [FUNCTION DOC]
    # Deprecated property. Returns the legacy class of each element component, derived from the
    # element spec.
    """Returns the class of each component of an element of this iterator.

    The expected values are `tf.Tensor` and `tf.sparse.SparseTensor`.

    Returns:
      A (nested) structure of Python `type` objects corresponding to each
      component of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_classes(),  # pylint: disable=protected-access
        self._element_spec)

  @property
  @deprecation.deprecated(
      None, "Use `tf.compat.v1.data.get_output_shapes(iterator)`.")
  def output_shapes(self):
    # [FUNCTION DOC]
    # Deprecated property. Returns the legacy shape of each element component, derived from the
    # element spec.
    """Returns the shape of each component of an element of this iterator.

    Returns:
      A (nested) structure of `tf.TensorShape` objects corresponding to each
      component of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_shapes(),  # pylint: disable=protected-access
        self._element_spec)

  @property
  @deprecation.deprecated(
      None, "Use `tf.compat.v1.data.get_output_types(iterator)`.")
  def output_types(self):
    # [FUNCTION DOC]
    # Deprecated property. Returns the legacy dtype of each element component, derived from the
    # element spec.
    """Returns the type of each component of an element of this iterator.

    Returns:
      A (nested) structure of `tf.DType` objects corresponding to each component
      of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_types(),  # pylint: disable=protected-access
        self._element_spec)

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the nested type spec of one element, set at construction.
    return self._element_spec

  def get_next(self):
    # [FUNCTION DOC]
    # Returns the next element by delegating to the internal next logic. Raises an out-of-range
    # error (not StopIteration) at the end of the data.
    return self._next_internal()

  def get_next_as_optional(self):
    # [FUNCTION DOC]
    # Returns an optional value wrapping the next element; it has no value when the iterator is
    # exhausted instead of raising. Takes no inputs; the op is colocated with the iterator
    # resource.
    # TODO(b/169442955): Investigate the need for this colocation constraint.
    with ops.colocate_with(self._iterator_resource):
      # pylint: disable=protected-access
      return optional_ops._OptionalImpl(
          gen_dataset_ops.iterator_get_next_as_optional(
              self._iterator_resource,
              output_types=structure.get_flat_tensor_types(self.element_spec),
              output_shapes=structure.get_flat_tensor_shapes(
                  self.element_spec)), self.element_spec)

  def _serialize_to_tensors(self):
    # [FUNCTION DOC]
    # Checkpoint support: serializes the iterator, using the dataset's external state policy if
    # set, otherwise the fail policy. Returns a mapping from the state name to the serialized
    # variant tensor. (Reads the dataset reference, which is none for iterators rebuilt from
    # components, so those use the fail policy.)
    serialized_iterator = None
    if (self._dataset and
        self._dataset.options().experimental_external_state_policy):
      serialized_iterator = gen_dataset_ops.serialize_iterator(
          self._iterator_resource,
          self._dataset.options().experimental_external_state_policy.value)
    else:
      serialized_iterator = gen_dataset_ops.serialize_iterator(
          self._iterator_resource,
          options_lib.ExternalStatePolicy.FAIL.value)
    return {"_STATE": serialized_iterator}

  def _restore_from_tensors(self, restored_tensors):
    # [FUNCTION DOC]
    # Checkpoint support: input is a mapping with the state entry. Deserializes it into the
    # iterator resource (colocated) and returns a list with the resulting op.
    with ops.colocate_with(self._iterator_resource):
      return [gen_dataset_ops.deserialize_iterator(
          self._iterator_resource, restored_tensors["_STATE"])]

  def _copy_trackable_to_cpu(self, object_map):
    # [FUNCTION DOC]
    # Checkpointing hook. Input: object_map, a mapping from trackable objects to their copies,
    # mutated in place. If this iterator has no copy yet, creates a new owned iterator (from the
    # same dataset, or from the same components and element spec when no dataset exists). Then
    # serializes this iterator's state and restores it into the copy. Returns nothing.
    """Implements checkpointing protocols for `Trackable`."""
    # Generate values to copy over
    if self not in object_map:
      # If self is not populated in object_map yet, instantiate the copy
      if self._dataset is None:
        object_map[self] = OwnedIterator(components=self._components,
                                         element_spec=self._element_spec)
      else:
        object_map[self] = OwnedIterator(dataset=self._dataset)

    # Copy values from `self` to copy of `self`
    serialized = self._serialize_to_tensors()
    object_map[self]._restore_from_tensors(serialized)  # pylint: disable=protected-access

  def __tf_tracing_type__(self, _):
    # [FUNCTION DOC]
    # Tracing protocol hook; the argument (a tracing context) is ignored. Returns the iterator
    # type spec used as the tf.function tracing type.
    return self._type_spec


@tf_export("data.IteratorSpec", v1=[])
class IteratorSpec(type_spec.TypeSpec):
  # [CLASS DOC]
  # Type specification for data iterators, exported publicly. Holds one attribute, the nested
  # element spec of the iterator. Its value type is the owned iterator, and it has a single
  # resource-tensor component, so iterators can be passed to tf.function with an input signature.
  """Type specification for `tf.data.Iterator`.

  For instance, `tf.data.IteratorSpec` can be used to define a tf.function that
  takes `tf.data.Iterator` as an input argument:

  >>> @tf.function(input_signature=[tf.data.IteratorSpec(
  ...   tf.TensorSpec(shape=(), dtype=tf.int32, name=None))])
  ... def square(iterator):
  ...   x = iterator.get_next()
  ...   return x * x
  >>> dataset = tf.data.Dataset.from_tensors(5)
  >>> iterator = iter(dataset)
  >>> print(square(iterator))
  tf.Tensor(25, shape=(), dtype=int32)

  Attributes:
    element_spec: A (nested) structure of `tf.TypeSpec` objects that represents
      the type specification of the iterator elements.
  """

  __slots__ = ["_element_spec"]

  def __init__(self, element_spec):
    # [FUNCTION DOC]
    # Input: element_spec, the nested type spec of iterator elements. Stores it; returns nothing.
    self._element_spec = element_spec

  @property
  def value_type(self):
    # [FUNCTION DOC]
    # Property returning the class of values described by this spec: the owned iterator class.
    return OwnedIterator

  def _serialize(self):
    # [FUNCTION DOC]
    # Returns a one-item tuple holding the element spec, used to serialize this type spec.
    return (self._element_spec,)

  @property
  def _component_specs(self):
    # [FUNCTION DOC]
    # Property returning a one-item tuple: a scalar resource tensor spec (the iterator resource).
    return (tensor.TensorSpec([], dtypes.resource),)

  def _to_components(self, value):
    # [FUNCTION DOC]
    # Input: value, an owned iterator. Returns a one-item tuple with its iterator resource.
    return (value._iterator_resource,)  # pylint: disable=protected-access

  def _from_components(self, components):
    # [FUNCTION DOC]
    # Input: components, the one-item tuple with the iterator resource. Returns a new owned
    # iterator rebuilt from the components and this spec's element spec (no dataset).
    return OwnedIterator(
        dataset=None,
        components=components,
        element_spec=self._element_spec)

  @staticmethod
  def from_value(value):
    # [FUNCTION DOC]
    # Static constructor. Input: value, any object with an element spec (an iterator). Returns
    # an iterator spec with that element spec.
    return IteratorSpec(value.element_spec)  # pylint: disable=protected-access


# TODO(b/71645805): Expose trackable stateful objects from dataset.
class _IteratorSaveable(BaseSaverBuilder.SaveableObject):
  # [CLASS DOC]
  # HELPER: legacy saveable object used to save and restore iterator state with the saver.
  # Holds the iterator resource and a single save spec for the serialized state.
  """SaveableObject for saving/restoring iterator state."""

  def __init__(
      self,
      iterator_resource,
      name,
      external_state_policy=options_lib.ExternalStatePolicy.FAIL):
    # [FUNCTION DOC]
    # Inputs: iterator_resource (iterator resource tensor); name (string prefix for the save
    # spec); external_state_policy (optional enum, default fail, controlling how external state
    # is handled). Serializes the iterator immediately, builds a save spec named with a state
    # suffix on the iterator's device, and initializes the base class. Returns nothing.
    serialized_iterator = gen_dataset_ops.serialize_iterator(
        iterator_resource, external_state_policy=external_state_policy.value)
    specs = [
        BaseSaverBuilder.SaveSpec(
            serialized_iterator,
            "",
            name + "_STATE",
            device=iterator_resource.device)
    ]
    super(_IteratorSaveable, self).__init__(iterator_resource, specs, name)

  def restore(self, restored_tensors, restored_shapes):
    # [FUNCTION DOC]
    # Inputs: restored_tensors (list whose first entry is the serialized state); restored_shapes
    # (unused). Deserializes the state into the iterator resource (colocated) and returns the op.
    with ops.colocate_with(self.op):
      return gen_dataset_ops.deserialize_iterator(self.op, restored_tensors[0])


nested_structure_coder.register_codec(
    nested_structure_coder.BuiltInTypeSpecCodec(
        IteratorSpec, struct_pb2.TypeSpecProto.DATA_ITERATOR_SPEC
    )
)


@deprecation.deprecated(
    None, "Use `tf.data.Iterator.get_next_as_optional()` instead.")
@tf_export("data.experimental.get_next_as_optional")
def get_next_as_optional(iterator):
  # [FUNCTION DOC]
  # Deprecated public API (data.experimental.get_next_as_optional). Input: iterator, a data
  # iterator. Returns the optional produced by the iterator's own get-next-as-optional method:
  # it holds the next element, or no value at the end. Errors come from the iterator.
  """Returns a `tf.experimental.Optional` with the next element of the iterator.

  If the iterator has reached the end of the sequence, the returned
  `tf.experimental.Optional` will have no value.

  Args:
    iterator: A `tf.data.Iterator`.

  Returns:
    A `tf.experimental.Optional` object which either contains the next element
    of the iterator (if it exists) or no value.
  """
  return iterator.get_next_as_optional()


iterator_autograph.register_overrides()
