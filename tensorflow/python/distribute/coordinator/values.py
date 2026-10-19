# [MODULE DOC]
# Value classes used by the parameter server cluster coordinator. Provides the concrete remote
# value implementation (and a variant that never caches), the public container for per-worker
# values with its type specification, classes that turn a dataset or a dataset function into a
# per-worker dataset whose iterators are created on every worker, helpers to serialize a dataset
# to a graph definition and rebuild it on a worker, and the per-worker distributed iterator type.
# The public per-worker values class is registered for API export at import time. Not a pure
# helper module; the coordinator module builds on it.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Important value classes relevant to `ClusterCoordinator`.

This is currently under development and the API is subject to change.
"""

import threading

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops.options import ExternalStatePolicy
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute.coordinator import remote_value
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import function as tf_function
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import type_spec as type_spec_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_dataset_ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest
from tensorflow.python.util.tf_export import tf_export


# TODO(yuefengz): create an implementation for resource RemoteValue which needs
# to remember the closure object while a normal RemoteValue doesn't.
class RemoteValueImpl(remote_value.RemoteValue):
  # [CLASS DOC]
  # Concrete remote value produced when a closure is scheduled. Holds the originating closure, a
  # type specification, the raw per-worker values (possibly remote tensors), a cached local copy
  # guarded by a lock, an error, a status (not ready, aborted or ready) and a threading event that
  # signals the status became available. Consumers block on the event; the coordinator sets
  # values, errors or the aborted state.
  """Implementation of `RemoteValue`."""

  def __init__(self, closure, type_spec):  # pylint: disable=super-init-not-called
    # [FUNCTION DOC]
    # Creates a not-ready remote value. Parameters: closure (the closure that produces the value)
    # and type_spec (used to trace functions that take this value as input). Initializes empty
    # values, error and local cache, a lock for the local copy and an unset status event.
    # Does not call the parent constructor. Returns nothing.
    """Initializes a `RemoteValueImpl`.

    Args:
      closure: The closure from which the `RemoteValue` is created.
      type_spec: The type spec for this `RemoteValue` which is used to trace
        functions that take this `RemoteValue` as input.
    """
    self._closure = closure
    self._type_spec = type_spec
    self._values = None
    self._has_fetched_to_local = False
    self._has_fetched_to_local_lock = threading.Lock()
    self._fetched_tensors = None
    self._error = None
    self._status_available_event = threading.Event()
    self._status = remote_value.RemoteValueStatus.NOT_READY

  def _set_aborted(self, error):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the inline comment says the event is cleared, but the code sets it, which wakes
    # waiting threads so they can see the aborted status and raise.
    # HELPER: marks this value aborted (producing closure aborted by a task failure, retryable).
    # Records the given error, drops any values, and sets the status event to wake waiters.
    self._status = remote_value.RemoteValueStatus.ABORTED
    self._values = None
    self._error = error

    # Wake up any waiting thread and clear the event.
    self._status_available_event.set()

  def _rebuild_on(self, worker):
    # [FUNCTION DOC]
    # HELPER: re-executes the producing closure on the given worker to rebuild this value.
    # Clears the status event first so readers block again, then asks the closure to execute on
    # the worker. Returns nothing. A TODO notes inputs are not rebuilt.
    self._status_available_event.clear()
    # TODO(yuefengz): we may need to rebuild its inputs as well.
    self._closure.execute_on(worker)

  def _set_values(self, tensors):
    # [FUNCTION DOC]
    # HELPER: marks the value ready with the given result (a tensor or structure of tensors, or
    # nothing for functions without output), clears any error and wakes waiters. Returns nothing.
    self._status = remote_value.RemoteValueStatus.READY
    self._values = tensors
    self._error = None
    self._status_available_event.set()

  def _set_error(self, error):
    # [FUNCTION DOC]
    # HELPER: marks the value ready but failed (non-retryable error). Stores the error, drops
    # values and wakes waiters. Returns nothing.
    self._status = remote_value.RemoteValueStatus.READY
    self._values = None
    self._error = error
    self._status_available_event.set()

  def _get_values(self):
    # [FUNCTION DOC]
    # HELPER: blocks until the status is available, then returns the raw stored values (possibly
    # nothing).
    self._status_available_event.wait()
    return self._values

  def _get_error(self):
    # [FUNCTION DOC]
    # HELPER: blocks until the status is available, then returns the stored error or nothing.
    self._status_available_event.wait()
    return self._error

  def _wait_and_maybe_error(self):
    # [FUNCTION DOC]
    # HELPER: blocks until the status is available. Raises a cancelled error if the value was
    # aborted (the function must be rescheduled), or re-raises the stored error if there is one.
    # Returns nothing when the value is fine.
    self._status_available_event.wait()
    if self._status is remote_value.RemoteValueStatus.ABORTED:
      raise errors.CancelledError(
          None, None,
          "The corresponding function is aborted. Please reschedule the "
          "function.")
    if self._error is not None:
      raise self._error

  def fetch(self):
    # [FUNCTION DOC]
    # Waits for the value (raising the same errors as get) and returns it converted to numpy data:
    # every element of the result structure that has a numpy conversion is converted, other
    # elements are returned as they are. Returns the converted structure.
    # TODO(rchao): Discuss the possibility of letting users perform `numpy`
    # themselves at API graduation.
    return nest.map_structure(
        lambda x: x.numpy() if hasattr(x, "numpy") else x, self.get())

  def _copy_to_local(self):
    # [FUNCTION DOC]
    # HELPER: copies the stored remote tensors to the coordinator.
    # Returns nothing (the null value) if the function produced no value; otherwise maps over the
    # structure, copying each tensor with an identity operation placed on the coordinator job (the
    # job name of the current server definition). Distributed iterators are returned unchanged.
    def copy_tensor(composite_tensor_obj):
      # [FUNCTION DOC]
      # HELPER: copies one element to the coordinator. A distributed iterator is returned as it
      # is; anything else gets an identity operation under a device scope for the coordinator job.
      """Copy a remote tensor to local (coordinator)."""
      if isinstance(composite_tensor_obj, input_lib.DistributedIterator):
        # A DistributedIterator cannot be copied to local; users should not
        # access that anyway.
        return composite_tensor_obj

      with ops.device("/job:%s" % context.get_server_def().job_name):
        # Copying to local (the coordinator) with `tf.device`.
        return array_ops.identity(composite_tensor_obj)

    fetched_result = None
    if self._values is not None:
      # When `self._values` is `None`, it indicates the associated function
      # does not have a return value.
      fetched_result = nest.map_structure(copy_tensor, self._values)
    return fetched_result

  def get(self):
    # [FUNCTION DOC]
    # Waits for the value (raising on abort or error), copies it to the coordinator once under a
    # lock, caches the local copy, and returns it (tensors or a structure of tensors, or nothing
    # if there is no output).
    self._wait_and_maybe_error()

    with self._has_fetched_to_local_lock:
      if not self._has_fetched_to_local:
        self._fetched_tensors = self._copy_to_local()
        self._has_fetched_to_local = True

    return self._fetched_tensors


class RemoteVariable(RemoteValueImpl):
  # [CLASS DOC]
  # Remote value for a mutable per-worker variable. Same as the base implementation except that
  # reads are never cached, so each read returns the current variable contents.
  """A RemoteValue that represents a mutable per-worker variable."""

  def get(self):
    # [FUNCTION DOC]
    # Waits for the value (raising on abort or error) and returns a fresh local copy each call,
    # bypassing the cache so the up-to-date value is read.
    """Retrieve value with no caching to ensure we get the up-to-date value."""
    self._wait_and_maybe_error()
    return self._copy_to_local()


@tf_export("distribute.experimental.coordinator.PerWorkerValues",
           "distribute.coordinator.PerWorkerValue", v1=[])
class PerWorkerValues(composite_tensor.CompositeTensor):
  # [CLASS DOC]
  # Public container holding one remote value per worker. When passed as an argument of a
  # scheduled function, each worker receives its own value. Currently only created by iterating a
  # per-worker dataset. Holds the tuple of remote values; participates in tracing as a composite
  # tensor through its type specification. Exported in the public distribute API.
  """A container that holds a list of values, one value per worker.

  `tf.distribute.experimental.coordinator.PerWorkerValues` contains a collection
  of values, where each of the values is located on its corresponding worker,
  and upon being used as one of the `args` or `kwargs` of
  `tf.distribute.experimental.coordinator.ClusterCoordinator.schedule()`, the
  value specific to a worker will be passed into the function being executed at
  that corresponding worker.

  Currently, the only supported path to create an object of
  `tf.distribute.experimental.coordinator.PerWorkerValues` is through calling
  `iter` on a `ClusterCoordinator.create_per_worker_dataset`-returned
  distributed dataset instance. The mechanism to create a custom
  `tf.distribute.experimental.coordinator.PerWorkerValues` is not yet supported.
  """

  def __init__(self, values):
    # [FUNCTION DOC]
    # Stores the given collection of values as a tuple. Raises an assertion error (not a type
    # error) if any element is not a remote value. Returns nothing.
    for v in values:
      if not isinstance(v, remote_value.RemoteValue):
        raise AssertionError(
            "`PerWorkerValues` should only take `RemoteValue`s.")
    self._values = tuple(values)

  @property
  def _type_spec(self):
    # [FUNCTION DOC]
    # HELPER: type specification used for tracing. Returns a per-worker values type spec built
    # from the type specification of the first remote value and the concrete class of this object.
    return PerWorkerValuesTypeSpec(
        self._values[0]._type_spec,  # pylint: disable=protected-access
        type(self))


class PerWorkerValuesTypeSpec(type_spec_lib.TypeSpec):
  # [CLASS DOC]
  # Type specification for per-worker values. Supports only tracing a function with such a value
  # as input. Holds the specification of a single value and the concrete container class. Most of
  # the standard type spec protocol is simplified or unimplemented.
  """TypeSpec for PerWorkerValues.

  It only support tracing a function using a PerWorkerValues.
  """

  def __init__(self, value_spec, descendant_type):
    # [FUNCTION DOC]
    # Stores the specification of one per-worker value (must be a non-empty, truthy object,
    # otherwise an assertion error is raised) and the container class that this spec describes.
    assert value_spec
    self._value_spec = value_spec
    self._descendant_type = descendant_type

  def _serialize(self):
    # [FUNCTION DOC]
    # HELPER: returns the serialized form of this spec, a one element tuple holding the value spec.
    return (self._value_spec,)

  @property
  def value_type(self):
    # [FUNCTION DOC]
    # Returns the container class described by this spec.
    return self._descendant_type

  def most_specific_common_supertype(self, others):
    # [FUNCTION DOC]
    # KNOWN ISSUE: always raises, so this spec cannot be combined or relaxed with other specs.
    # Takes a list of other specs and raises a not-implemented error.
    raise NotImplementedError(
        "most_specific_common_supertype is not implemented")

  @property
  def _component_specs(self):
    # [FUNCTION DOC]
    # HELPER: returns the stored value spec as the component specs (a single spec, not a
    # per-worker structure).
    return self._value_spec

  def _to_components(self, value):
    # [FUNCTION DOC]
    # KNOWN ISSUE: ignores the given value and returns the stored value spec rather than
    # components extracted from the value.
    # HELPER: part of the type spec protocol; returns the stored value spec.
    return self._value_spec

  def _from_components(self, value):
    # [FUNCTION DOC]
    # HELPER: part of the type spec protocol; returns the given components unchanged.
    return value


class PerWorkerDatasetFromDatasetFunction(object):
  # [CLASS DOC]
  # Per-worker dataset created from a dataset function. The function is turned into a concrete
  # function at construction (variable creation inside is forbidden). Holds that function, the
  # cluster coordinator and a cached element spec (initially nothing). Iterating it creates one
  # iterator resource on each worker and returns a per-worker distributed iterator.
  """Represents worker-distributed datasets created from dataset function."""

  def __init__(self, dataset_fn, coordinator):
    # [FUNCTION DOC]
    # Parameters: dataset_fn (a function returning a dataset; may be a traced function wrapper, an
    # already concrete function or a plain callable) and coordinator (cluster coordinator used to
    # create dataset resources). Converts the function to a concrete function under a variable
    # creator scope that raises a value error when the function tries to create variables.
    # Stores the function, the coordinator and an empty element spec. Returns nothing.
    """Makes an iterable from datasets created by the given function.

    Args:
      dataset_fn: A function that returns a `Dataset`.
      coordinator: a `ClusterCoordinator` object, used to create dataset
        resources.
    """

    def disallow_variable_creation(next_creator, **kwargs):
      # [FUNCTION DOC]
      # HELPER: variable creator hook that always raises a value error saying variables cannot be
      # created in the dataset function. Parameters are the next creator and keyword arguments
      # (unused).
      raise ValueError("Creating variables in `dataset_fn` is not allowed.")

    if isinstance(dataset_fn, def_function.Function):
      with variable_scope.variable_creator_scope(disallow_variable_creation):
        dataset_fn = dataset_fn.get_concrete_function()
    elif not isinstance(dataset_fn, tf_function.ConcreteFunction):
      with variable_scope.variable_creator_scope(disallow_variable_creation):
        dataset_fn = def_function.function(dataset_fn).get_concrete_function()
    self._dataset_fn = dataset_fn
    self._coordinator = coordinator
    self._element_spec = None

  def build(self):
    # [FUNCTION DOC]
    # Triggers dataset creation on every worker without creating an iterator. Returns a per-worker
    # values object whose remote values refer to the built dataset on each worker; their type
    # specs are overwritten with the output spec of the dataset function.
    """Trigger dataset creation on workers without creating an iterator.

    Returns:
      A PerWorkerValues object containing a tuple of RemoteValues, themselves
      containing the built Dataset for each worker
    """
    def _create_per_worker_dataset():
      # [FUNCTION DOC]
      # HELPER: closure run on each worker; calls the dataset function and returns the dataset.
      dataset = self._dataset_fn()
      return dataset

    # pylint: disable=protected-access
    per_worker_dataset = self._coordinator._create_per_worker_resources(
        _create_per_worker_dataset)
    # hack type_spec of RemoteValues
    dataset_fn_output_type_spec = self._dataset_fn.structured_outputs._type_spec
    for dataset_remote_value in per_worker_dataset._values:
      dataset_remote_value._type_spec = dataset_fn_output_type_spec
    return per_worker_dataset

  def __iter__(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: a comment claims the resource is created only once per object (using an object
    # id), but no such deduplication is visible here; each call creates new per-worker resources.
    # Creates an iterator on every worker. Raises a runtime error when not executing eagerly or
    # when called inside a function being built. Sets each remote iterator's type spec from the
    # strategy and the dataset function output. Returns a per-worker distributed iterator.
    # We would like users to create iterators outside `tf.function`s so that we
    # can track them.
    if (not context.executing_eagerly() or
        ops.get_default_graph().building_function):
      raise RuntimeError(
          "__iter__() is not supported inside of tf.function or in graph mode.")

    def _create_per_worker_iterator():
      # [FUNCTION DOC]
      # HELPER: closure run on each worker; calls the dataset function and returns an iterator over
      # the dataset.
      dataset = self._dataset_fn()
      return iter(dataset)

    # If PerWorkerDatasetFromDatasetFunction.__iter__ is called multiple
    # times, for the same object it should only create and register resource
    # once. Using object id to distinguish different iterator resources.
    per_worker_iterator = self._coordinator._create_per_worker_resources(
        _create_per_worker_iterator)

    # Setting type_spec of each RemoteValue so that functions taking these
    # RemoteValues as inputs can be traced.
    for iterator_remote_value in per_worker_iterator._values:
      iterator_remote_value._type_spec = (
          input_lib.get_iterator_spec_from_dataset(
              self._coordinator.strategy, self._dataset_fn.structured_outputs))

    return PerWorkerDistributedIterator(per_worker_iterator._values)

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Returns the element type specification of the dataset produced by the dataset function.
    # Raises a not-implemented error if the stored function is not a concrete function.
    """The type specification of an element of this dataset.

    This property is subject to change without notice.
    """
    if not isinstance(self._dataset_fn, tf_function.ConcreteFunction):
      raise NotImplementedError(
          "`element_spec` is not supported when the `dataset_fn` is not "
          "a `ConcreteFunction`.")
    return self._dataset_fn.structured_outputs.element_spec


def serialize_dataset_to_graph(dataset):
  # [FUNCTION DOC]
  # Serializes a dataset into a graph definition. Applies the dataset's debug options, converts it
  # with the external-state policy set to warn and device assignments stripped, and returns the
  # graph definition. Parameter: a dataset object.
  dataset = dataset._apply_debug_options()  # pylint: disable=protected-access
  graph_def = gen_dataset_ops.dataset_to_graph_v2(
      dataset._variant_tensor,  # pylint: disable=protected-access
      external_state_policy=ExternalStatePolicy.WARN.value,
      strip_device_assignment=True)
  return graph_def


class _RemoteDataset(dataset_ops.DatasetSource):
  # [CLASS DOC]
  # HELPER: dataset source rebuilt from a serialized graph definition, used on workers to
  # recreate a dataset serialized on the coordinator. Holds the element specification.
  """Creates a dataset given a graph def."""

  def __init__(self, graph_def, element_spec):
    # [FUNCTION DOC]
    # Stores the element spec, creates the dataset variant from the graph definition, and
    # initializes the dataset source base class with it. Returns nothing.
    self._elem_spec = element_spec
    variant_tensor = ged_ops.dataset_from_graph(graph_def)
    super(_RemoteDataset, self).__init__(variant_tensor)

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Returns the element specification given at construction.
    return self._elem_spec


def deserialize_dataset_from_graph(graph_def, element_spec):
  # [FUNCTION DOC]
  # HELPER: builds and returns a dataset from a serialized graph definition and an element spec.
  return _RemoteDataset(graph_def, element_spec)


class PerWorkerDatasetFromDataset(PerWorkerDatasetFromDatasetFunction):
  # [CLASS DOC]
  # Per-worker dataset created from an existing dataset (plain, distributed, or distributed from
  # a function) by wrapping it in a generated dataset function. Otherwise behaves as its parent.
  """Represents worker-distributed datasets created from a dataset."""

  def __init__(self, dataset, coordinator):
    # [FUNCTION DOC]
    # Parameters: dataset (a plain dataset, a distributed dataset, or a distributed dataset built
    # from a function) and coordinator (cluster coordinator). Builds a dataset function that, for a
    # distributed dataset, serializes the original dataset and rebuilds it in place of the
    # original; for a function-based distributed dataset, builds it; for a plain dataset,
    # deserializes its serialized graph. Raises a value error for any other type. Then calls the
    # parent constructor with the generated function.
    """Makes an iterable from datasets created by the given dataset.

    It creates a dataset_fn which deserializes a dataset from a graph under the
    hood.

    Args:
      dataset: A tf.data.Dataset, a DistributedDataset or a
        DistributedDatasetsFromFunction
      coordinator: a `ClusterCoordinator` object, used to create dataset
        resources.
    """
    if isinstance(dataset, input_lib.DistributedDataset):
      original_dataset = dataset._original_dataset
      serialized = serialize_dataset_to_graph(original_dataset)

      def dataset_fn():
        # [FUNCTION DOC]
        # HELPER: deserializes the original dataset from its graph, rebuilds the distributed
        # dataset to replace its source with it, and returns the distributed dataset.
        deserialized = deserialize_dataset_from_graph(
            serialized, original_dataset.element_spec)
        dataset.build(dataset_to_replace=deserialized)
        return dataset
    elif isinstance(dataset, input_lib.DistributedDatasetsFromFunction):
      def dataset_fn():
        # [FUNCTION DOC]
        # HELPER: builds the function-based distributed dataset and returns it.
        dataset.build()
        return dataset
    elif isinstance(dataset, dataset_ops.Dataset):
      serialized = serialize_dataset_to_graph(dataset)

      def dataset_fn():
        # [FUNCTION DOC]
        # HELPER: deserializes the captured serialized graph into a dataset with the original
        # element spec and returns it.
        return deserialize_dataset_from_graph(serialized, dataset.element_spec)
    else:
      raise ValueError("Unexpected dataset type!")

    super(PerWorkerDatasetFromDataset, self).__init__(dataset_fn, coordinator)


def get_per_worker_dataset(dataset_or_dataset_fn, coordinator):
  # [FUNCTION DOC]
  # Factory. If the first argument is callable it is treated as a dataset function and a
  # per-worker dataset from function is returned; otherwise it is treated as a dataset and a
  # per-worker dataset from dataset is returned. The coordinator is passed through.
  """Returns a per-worker dataset from a dataset or a dataset function."""
  if callable(dataset_or_dataset_fn):
    return PerWorkerDatasetFromDatasetFunction(dataset_or_dataset_fn,
                                               coordinator)
  else:
    return PerWorkerDatasetFromDataset(dataset_or_dataset_fn, coordinator)


class PerWorkerDistributedIterator(PerWorkerValues):
  # [CLASS DOC]
  # Per-worker values container acting as the iterator over a per-worker dataset. Hands out
  # per-worker remote iterators to scheduled functions; direct iteration on the coordinator is
  # unsupported.
  """Distributed iterator for `ClusterCoordinator`."""

  def __next__(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: always ends in a not-implemented error because it delegates to the next-element
    # method, which is unsupported.
    # Python iteration hook; delegates to the next-element method.
    return self.get_next()

  def get_next(self, name=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring claims it returns the next input for all replicas, but it always
    # raises a not-implemented error, and the message names a different class (an async
    # distributed iterator).
    # Parameter name is optional and unused.
    """Returns the next input from the iterator for all replicas."""
    raise NotImplementedError("Iterating over an `AsyncDistributedIterator` "
                              "is not supported right now.")
