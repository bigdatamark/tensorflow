# [MODULE DOC]
# Distributed input machinery for tf.distribute strategies. Defines the classes that turn a
# dataset (or a function producing datasets) into per-worker datasets and iterators whose
# elements are grouped across replicas: InputWorkers (mapping from input worker devices to the
# compute devices they feed), the type specifications (for tf.function tracing and composite
# tensor support) of distributed iterators and datasets, DistributedIterator,
# DistributedDataset, DistributedDatasetsFromFunction, the per-worker iterator classes, and
# MultiStepContext (collects outputs when running several steps at once). It also holds
# private helpers for partial-batch handling (dummy values, optionals, cardinality checks),
# rebatching, and building per-replica values.
# Import-time side effects: creates two monitoring sampler metrics that record how long it takes
# to initialize distributed datasets (one for datasets, one for datasets from a function), and
# registers the distributed iterator and distributed dataset classes with the autograph
# enumerate registry so that using enumerate on them inside a tf.function raises an error.
# This is mostly an internal module; strategies and distribute_lib use it to build the objects
# returned by the public distribution-strategy dataset APIs.
#
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
"""Various classes representing distributed inputs."""

import functools
import sys
import time

import six

from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.data.experimental.ops import batching
from tensorflow.python.data.experimental.ops import cardinality as cardinality_lib
from tensorflow.python.data.experimental.ops import distribute
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_ops
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import values
from tensorflow.python.distribute.distribute_lib import InputReplicationMode
from tensorflow.python.eager import context
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import device as tf_device
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.types import distribute as distribute_types
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc


_distributed_dataset_initialization_time_milliseconds = monitoring.Sampler(
    "/tensorflow/api/distribution_strategy/"
    "distributed_dataset_initialization_time_milliseconds",
    monitoring.ExponentialBuckets(scale=1, growth_factor=2, bucket_count=26),
    "Track the time (in milliseconds) to initialize distributed datasets.",
    "strategy", "workers")

_distributed_dataset_from_function_initialization_time_milliseconds = (
    monitoring.Sampler(
        "/tensorflow/api/distribution_strategy/"
        "distributed_dataset_from_function_initialization_time_milliseconds",
        monitoring.ExponentialBuckets(
            scale=1, growth_factor=2, bucket_count=26),
        "Track the time (in milliseconds) to initialize distributed datasets "
        "from function.",
        "strategy", "workers"))


def get_iterator_spec_from_dataset(strategy, dataset):
  # [FUNCTION DOC]
  # Builds the type specification that iterating over the given dataset would produce.
  # Inputs: strategy is the distribution strategy; dataset is a dataset object (or the
  # structured output of a dataset function). If the dataset's own type spec is a distributed
  # dataset spec (or distributed-from-function spec), the result is a distributed iterator spec
  # built from the strategy's input workers, the dataset's element spec and cardinality, with
  # partial-batch handling (get-next-as-optional) enabled and no options. Otherwise a plain
  # iterator spec for the element spec is returned, after logging a warning when the strategy
  # has GPUs allocated per worker (suggesting the use of the distributed dataset APIs).
  # Returns the iterator type spec. Raises nothing itself; logs a warning as a side effect.
  """Returns an iterator spec from dataset function.

  This function constructs type spec for iterator obtained from
  iter(dataset).

  Args:
    strategy: a `tf.distribute.Strategy` object, used to run all-reduce to
        handle last partial batch.
    dataset: A tf.data.Dataset instance. If using a function that returns a
      tf.data.Dataset instance, pass dataset_fn.structured_outputs.

  Returns:
    A type_spec for iterator for dataset instance.

  """
  # pylint: disable=protected-access
  output_element_spec = dataset.element_spec
  if isinstance(dataset._type_spec,
                (DistributedDatasetSpec,
                 DistributedDatasetsFromFunctionSpec)):
    iterator_type_spec = DistributedIteratorSpec(
        strategy.extended._input_workers_with_options(),
        output_element_spec,
        strategy.extended._container_strategy(),
        options=None,
        cardinality=dataset.cardinality,
        enable_get_next_as_optional=True)
  else:
    if strategy.extended._num_gpus_per_worker:
      logging.warning(
          f"{strategy.extended._num_gpus_per_worker} GPUs "
          "are allocated per worker. Please use DistributedDataset by "
          "calling strategy.experimental_distribute_dataset or strategy."
          "distribute_datasets_from_function to make best use of GPU "
          "resources"
      )
    iterator_type_spec = iterator_ops.IteratorSpec(output_element_spec)
  return iterator_type_spec
  # pylint: enable=protected-access


class InputWorkers(object):
  # [CLASS DOC]
  # Describes which devices feed which: a list of input (worker) devices, each paired with the
  # tuple of compute devices that it feeds data to. Holds the original pairs, the tuple of input
  # worker devices, whether devices are fully canonicalized, and the canonicalized compute
  # devices per worker. Created by strategies and passed to distributed datasets and iterators.
  """A 1-to-many mapping from input worker devices to compute devices."""

  # TODO(ishark): Remove option canonicalize_devices and make all the callers
  # pass canonicalized or raw device strings as relevant from strategy.
  def __init__(self,
               worker_device_pairs,
               canonicalize_devices=True):
    # [FUNCTION DOC]
    # Stores the worker/device pairs and precomputes the canonical device names.
    # Inputs: worker_device_pairs is a sequence of pairs of an input device string and a tuple of
    # compute device strings fed by it; canonicalize_devices (default true) chooses full
    # canonicalization of the compute devices, otherwise job and task are stripped out.
    # Returns nothing; sets the input worker devices tuple and the fed devices tuple of tuples.
    """Initialize an `InputWorkers` object.

    Args:
      worker_device_pairs: A sequence of pairs: `(input device, a tuple of
        compute devices fed by that input device)`.
      canonicalize_devices: Whether to canonicalize devices for workers fully or
        partially. If False, it will partially canonicalize devices by removing
        job and task.
    """
    self._worker_device_pairs = worker_device_pairs
    self._input_worker_devices = tuple(d for d, _ in self._worker_device_pairs)
    self._canonicalize_devices = canonicalize_devices
    if canonicalize_devices:
      self._fed_devices = tuple(
          tuple(device_util.canonicalize(d)
                for d in f)
          for _, f in self._worker_device_pairs)
    else:
      self._fed_devices = tuple(
          tuple(device_util.canonicalize_without_job_and_task(d)
                for d in f)
          for _, f in self._worker_device_pairs)

  @property
  def num_workers(self):
    # [FUNCTION DOC]
    # Property returning the number of input workers (length of the input device list).
    return len(self._input_worker_devices)

  @property
  def worker_devices(self):
    # [FUNCTION DOC]
    # Property returning the tuple of input worker device strings, in worker order.
    return self._input_worker_devices

  def compute_devices_for_worker(self, worker_index):
    # [FUNCTION DOC]
    # Returns the tuple of canonicalized compute devices fed by the input worker at the given
    # integer index. Raises an index error when the index is out of range.
    return self._fed_devices[worker_index]

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a multi-line debug string: the class name followed by one line per worker showing
    # its index, input device and the compute devices it feeds.
    devices = self.worker_devices
    debug_repr = ",\n".join("  %d %s: %s" %
                            (i, devices[i], self._fed_devices[i])
                            for i in range(len(devices)))
    return "%s:{\n%s}" % (self.__class__.__name__, debug_repr)

  def serialize(self):
    # [FUNCTION DOC]
    # Returns a hashable-style description of this object: a two-item tuple of the original
    # worker/device pairs and the canonicalize-devices flag. Used to compare type specs.
    return (self._worker_device_pairs, self._canonicalize_devices)

  def deserialize(self, serialized):
    # [FUNCTION DOC]
    # KNOWN ISSUE: does not invert serialize. It passes the whole serialized two-item tuple as
    # the worker/device pairs argument (so the pairs and the flag are not unpacked), and it is
    # an instance method that ignores the instance.
    # Builds and returns a new InputWorkers from the given serialized value.
    return InputWorkers(serialized)


def _calculate_replicas_with_values(strategy, input_workers, optional_list):
  # [FUNCTION DOC]
  # HELPER: partial-batch handling for distributed iterators.
  # Counts how many replicas actually produced a value. For each input worker, places work on
  # that device and sums, over its compute devices, whether the optional has a value (as 64-bit
  # integers); then sums over workers. In multi-worker mode the client count is all-reduced with
  # a sum across workers through the strategy. Inputs: strategy, input_workers (InputWorkers),
  # optional_list (list per worker of lists of optionals). Returns a scalar integer tensor.
  """Computes the number of replicas that have values.

  Args:
    strategy: the `tf.distribute.Strategy`.
    input_workers: the `InputWorkers`.
    optional_list: a list of lists `tf.experimental.Optional`. The values from
      each compute device grouped by the input device.

  Returns:
    A scalar Tensor.
  """
  worker_has_values = []
  for worker, optionals in zip(input_workers.worker_devices, optional_list):
    with ops.device(worker):
      device_has_values = [
          math_ops.cast(v.has_value(), dtypes.int64) for v in optionals
      ]
      worker_has_values.append(
          math_ops.reduce_sum(device_has_values, keepdims=True))
  client_has_values = math_ops.reduce_sum(worker_has_values, keepdims=True)
  if strategy.extended._in_multi_worker_mode():  # pylint: disable=protected-access
    global_has_values = strategy.reduce(
        reduce_util.ReduceOp.SUM, client_has_values, axis=None)
    return array_ops.reshape(global_has_values, [])
  else:
    return array_ops.reshape(client_has_values, [])


def _is_statically_shaped(element_spec):
  # [FUNCTION DOC]
  # HELPER: decides whether partial-batch handling is needed for an iterator.
  # Flattens the element spec and returns false if any component is not fully static: for sparse
  # and ragged specs only the first (batch) dimension is checked (non-scalar and unknown batch
  # size means not static); for other specs every flat tensor spec must have a fully defined
  # shape. Returns true otherwise. Input is a nested structure of type specs.
  """Test if an iterator output is statically shaped.

  For sparse and ragged tensors this only tests the batch dimension.

  Args:
    element_spec: a nest structure of `tf.TypeSpec`. The element spec of the
      dataset of the iterator.

  Returns:
    True if the shape is static, false otherwise.
  """

  for spec in nest.flatten(element_spec):
    if isinstance(
        spec, (sparse_tensor.SparseTensorSpec, ragged_tensor.RaggedTensorSpec)):
      # For sparse or ragged tensor, we should only check the first
      # dimension in order to get_next_as_optional. This is because
      # when these tensors get batched by dataset only the batch dimension
      # is set.
      if spec.shape.rank > 0 and spec.shape.as_list()[0] is None:
        return False
    else:
      for component in spec._flat_tensor_specs:  # pylint: disable=protected-access
        if not component.shape.is_fully_defined():
          return False
  return True


class DistributedIteratorBase(collections_abc.Iterator,
                              distribute_types.DistributedIteratorInterface):
  # [CLASS DOC]
  # Common implementation of a distributed input iterator. Holds one per-worker iterator for each
  # input worker, the InputWorkers description, the strategy, the dataset cardinality, a flag
  # saying whether partial batches are handled through optionals, and an optional replica order.
  # Gets the next element for all replicas, producing per-replica values. Subclassed by
  # DistributedIterator.
  """Common implementation for all input iterators."""

  # pylint: disable=super-init-not-called
  def __init__(
      self,
      input_workers,
      iterators,
      strategy,
      cardinality,
      enable_get_next_as_optional,
      replica_order=None,
  ):
    # [FUNCTION DOC]
    # Stores the constructor arguments: input_workers (InputWorkers), iterators (list of per-worker
    # iterators), strategy, cardinality of the dataset, enable_get_next_as_optional (whether
    # partial batches are handled), and replica_order (optional list giving the order in which
    # replicas are rearranged, default none). Asserts input_workers is an InputWorkers and raises
    # a value error if it has no workers. Returns nothing.
    assert isinstance(input_workers, InputWorkers)
    if not input_workers.worker_devices:
      raise ValueError("Should have at least one worker for input iterator.")

    self._iterators = iterators
    self._input_workers = input_workers
    self._strategy = strategy
    self._cardinality = cardinality
    self._enable_get_next_as_optional = enable_get_next_as_optional
    self._replica_order = replica_order

  def next(self):
    # [FUNCTION DOC]
    # Python-2 style alias: returns the next element by delegating to the next-item protocol method.
    return self.__next__()

  def __next__(self):
    # [FUNCTION DOC]
    # Returns the next per-replica element from get_next. Converts the out-of-range error raised
    # at the end of the dataset into the standard stop-iteration signal.
    try:
      return self.get_next()
    except errors.OutOfRangeError:
      raise StopIteration

  def __iter__(self):
    # [FUNCTION DOC]
    # Returns the iterator itself, as required by the iterator protocol.
    return self

  def get_next_as_optional(self):
    # [FUNCTION DOC]
    # Returns an optional holding the next per-replica value, or an empty optional when no
    # replica has data left. Always performs partial-batch handling (unlike get_next), except
    # that for an infinite dataset it wraps the value from the no-partial-batch path, and for an
    # empty dataset (cardinality zero) in single-client mode it returns an empty optional
    # immediately. Otherwise it gets an optional list from every worker's iterator, counts the
    # replicas that have values, and conditionally builds either an optional of the value (with
    # dummy values substituted for replicas that ran out, reordered by replica order when set)
    # or an empty optional. Takes no arguments; builds graph/eager ops placed on worker devices.
    # Ideally get_next_as_optional() should be consistent with get_next(), but
    # we used to always do partial batch handling in get_next_as_optional(). We
    # are keeping this behavior for now until we understantd the impact.

    # Skip partial batch handling when the dataset is infinite or empty, as
    # there won't be any partial batches in those cases. This gives the user
    # more static shapes as it avoids the tf.cond. Note that for empty datasets,
    # we can only skip in single client mode, as the dataset can be non-empty on
    # other workers.
    if self._cardinality == cardinality_lib.INFINITE:
      return optional_ops.Optional.from_value(
          self._get_next_no_partial_batch_handling())
    if (self._cardinality == 0 and
        not self._strategy.extended._in_multi_worker_mode()):  # pylint: disable=protected-access
      return optional_ops.Optional.empty(self._element_spec)

    optional_list = []
    for i, worker in enumerate(self._input_workers.worker_devices):
      with ops.device(worker):
        optional_list.append(self._iterators[i].get_next_as_optional_list())

    def _create_optional_with_dummy():
      # [FUNCTION DOC]
      # HELPER: branch of the conditional in get_next_as_optional used when at least one replica
      # has data. Collects values (dummy empty values for exhausted replicas), reorders them by
      # replica order if one is set, groups them into a per-replica value and wraps that in an
      # optional. Takes no arguments.
      value_list = _get_value_or_dummy(
          self._input_workers, optional_list, produce_dummy=True)

      if self._replica_order is not None:
        value_list = self._reorder_replicas(value_list)

      per_replica = _create_per_replica(value_list, self._strategy)
      return optional_ops.Optional.from_value(per_replica)

    def _create_empty_optional():
      # [FUNCTION DOC]
      # HELPER: branch of the conditional in get_next_as_optional used when no replica has data;
      # returns an empty optional with this iterator's element spec.
      return optional_ops.Optional.empty(self._element_spec)

    num_replicas_with_values = _calculate_replicas_with_values(
        self._strategy, self._input_workers, optional_list)

    return tf_cond.cond(
        num_replicas_with_values > 0,
        _create_optional_with_dummy,
        _create_empty_optional,
        strict=True)

  def get_next(self, name=None):
    # [FUNCTION DOC]
    # Returns the next input for all replicas, as a per-replica structure.
    # Input: name, optional string used to name ops (only used when partial-batch handling is
    # disabled). Raises a value error if called from inside a replica context (it must be called
    # from outside the replica function). When partial-batch handling is off, delegates to the
    # no-partial-batch path. Otherwise gets optionals from every worker, counts replicas with
    # values and conditionally returns the value (with dummy values for exhausted replicas,
    # reordered if a replica order is set) or, when no replica has data, triggers the underlying
    # get-next to raise the end-of-sequence error.
    """Returns the next input from the iterator for all replicas."""
    with distribute_lib.enter_or_assert_strategy(
        self._strategy):
      if distribute_lib.get_replica_context() is not None:
        raise ValueError("next(iterator) should be called from outside of "
                         "replica_fn. e.g. strategy.run(replica_fn, "
                         "args=(next(iterator),))")

    if not self._enable_get_next_as_optional:
      return self._get_next_no_partial_batch_handling(name)

    optional_list = []
    for i, worker in enumerate(self._input_workers.worker_devices):
      with ops.device(worker):
        optional_list.append(self._iterators[i].get_next_as_optional_list())
    num_replicas_with_values = _calculate_replicas_with_values(
        self._strategy, self._input_workers, optional_list)

    def _value_or_dummy():
      # [FUNCTION DOC]
      # HELPER: branch of the conditional in get_next used when some replica has data. Gets
      # values (dummy empty values for exhausted replicas), applies the replica order if set,
      # and returns the grouped per-replica value. Takes no arguments.
      value_list = _get_value_or_dummy(
          self._input_workers, optional_list, produce_dummy=True)

      if self._replica_order is not None:
        value_list = self._reorder_replicas(value_list)

      return _create_per_replica(value_list, self._strategy)

    def _eof():
      # [FUNCTION DOC]
      # HELPER: branch of the conditional in get_next used when no replica has data. Calls the
      # no-partial-batch path purely so the underlying iterator raises its end-of-sequence error.
      # Optional.get_value raises InvalidArgumentError when there's no value,
      # so we need to call GetNext to raise EOFError.
      return self._get_next_no_partial_batch_handling()

    return tf_cond.cond(
        num_replicas_with_values > 0, _value_or_dummy, _eof, strict=True)

  def _get_next_no_partial_batch_handling(self, name=None):
    # [FUNCTION DOC]
    # HELPER: fast path of get_next without optionals. For each worker, runs the worker's
    # iterator get-next-as-list on the worker device (naming ops with the job and task of the
    # worker when a name is given) and concatenates results into one flat list of per-replica
    # values, reorders them if a replica order is set, and groups them into a per-replica value.
    # Raises the end-of-sequence error if any device has no data. Input: optional name string.
    replicas = []
    for i, worker in enumerate(self._input_workers.worker_devices):
      if name is not None:
        d = tf_device.DeviceSpec.from_string(worker)
        new_name = "%s_%s_%d" % (name, d.job, d.task)
      else:
        new_name = None
      with ops.device(worker):
        # Make `replicas` a flat list of values across all replicas.
        replicas.extend(self._iterators[i].get_next_as_list(new_name))

    if self._replica_order is not None:
      replicas = self._reorder_replicas(replicas)

    return _create_per_replica(replicas, self._strategy)

  def _reorder_replicas(self, replicas):
    # [FUNCTION DOC]
    # HELPER: rearranges a flat list of per-replica values so that position i of the result is
    # the entry at the index given by the i-th item of the replica order. Asserts that the replica
    # order and the list have the same length (assertion error with a message otherwise).
    # Returns the new list.
    assert len(self._replica_order) == len(
        replicas
    ), "replica order size ({}) != replicas size ({})!".format(
        len(self._replica_order), len(replicas)
    )
    return [replicas[i] for i in self._replica_order]


class DistributedDatasetAndIteratorSpec(type_spec.TypeSpec):
  # [CLASS DOC]
  # Shared base type specification for distributed datasets, distributed datasets from a
  # function, and distributed iterators. Holds the input workers, element spec, strategy,
  # dataset cardinality, partial-batch flag, input options, canonicalize-devices flag and replica
  # order. Supports subtype checks and most-specific-supertype computation. Deserialization is
  # intentionally unsupported because the strategy cannot be serialized.
  """Common Type specification for `DistributedDataset and DistributedDatasetsFromFunction."""

  __slots__ = [
      "_input_workers", "_element_spec", "_strategy", "_cardinality",
      "_enable_get_next_as_optional", "_options", "_canonicalize_devices"
  ]

  def __init__(
      self,
      input_workers,
      element_spec,
      strategy,
      options,
      cardinality=cardinality_lib.UNKNOWN,
      enable_get_next_as_optional=None,
      replica_order=None,
  ):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the replica order attribute is assigned but is not listed in the declared
    # slots, and it is not part of the serialized form nor passed on by the supertype and
    # tensor-rank-only/without-names copies, so it is dropped by those operations.
    # Stores the input workers (an InputWorkers object), element spec, strategy, options,
    # cardinality (default unknown), partial-batch flag (default none) and replica order
    # (default none). The canonicalize-devices flag is taken from the strategy (default true,
    # also true when there is no strategy). Raises a not-implemented error if input_workers is a
    # plain tuple, which is what a deserialization attempt would pass.
    # We don't want to allow deserialization of this class because we don't
    # serialize the strategy object. Currently the only places where
    # _deserialize is called is when we save/restore using SavedModels.
    if isinstance(input_workers, tuple):
      raise NotImplementedError("DistributedIteratorSpec does not have support "
                                "for deserialization.")
    else:
      self._input_workers = input_workers
      self._element_spec = element_spec
      self._strategy = strategy
      self._cardinality = cardinality
      self._enable_get_next_as_optional = enable_get_next_as_optional
      self._options = options
      if self._strategy:
        self._canonicalize_devices = getattr(self._strategy,
                                             "_canonicalize_devices", True)
      else:
        self._canonicalize_devices = True
      self._replica_order = replica_order

  def _serialize(self):
    # [FUNCTION DOC]
    # Returns a tuple used for spec comparison: the serialized input workers, the element spec,
    # and the identity numbers of the strategy and the options (because those cannot be truly
    # serialized).
    # We cannot serialize the strategy object so we convert it to an id that we
    # can use for comparison.
    return (self._input_workers.serialize(), self._element_spec,
            id(self._strategy), id(self._options))

  def _deserialize(self):
    # [FUNCTION DOC]
    # Always raises a value error stating deserialization is unsupported for this spec type.
    raise ValueError(
        f"Deserialization is currently unsupported for {type(self)}.")

  def sanity_check_type(self, other):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring claims it returns the most specific compatible type spec, but it
    # returns nothing; it only validates.
    # Validates that other can be compared with this spec. Raises a value error if the two specs
    # have different Python types, if their serialized input workers differ, or if they do not
    # use the very same strategy object. Returns nothing.
    """Returns the most specific TypeSpec compatible with `self` and `other`.

    Args:
      other: A `TypeSpec`.

    Raises:
      ValueError: If there is no TypeSpec that is compatible with both `self`
        and `other`.
    """
    # pylint: disable=protected-access
    if type(self) is not type(other):
      raise ValueError("No TypeSpec is compatible with both %s and %s" %
                       (self, other))
    if self._input_workers.serialize() != other._input_workers.serialize():
      raise ValueError("_input_workers is not compatible with both %s "
                       "and %s" % (self, other))
    if self._strategy is not other._strategy:
      raise ValueError("tf.distribute strategy is not compatible with both %s "
                       "and %s" % (self, other))

  def is_subtype_of(self, other):
    # [FUNCTION DOC]
    # Returns true if this spec is a subtype of other: the sanity check passes, both element specs
    # have the same nested structure, and every flattened element of this spec is a subtype of
    # the matching element of other. Returns false when the sanity check or the structure check
    # raises a type or value error.
    """Returns True if `self` is subtype of `other`.

    Args:
      other: A `TypeSpec`.
    """
    try:
      self.sanity_check_type(other)
      nest.assert_same_structure(self._element_spec, other._element_spec)  # pylint: disable=protected-access
    except (TypeError, ValueError):
      return False

    self_elements = nest.flatten(self._element_spec)
    other_elements = nest.flatten(other._element_spec)  # pylint: disable=protected-access

    return all(
        self_element.is_subtype_of(other_element)
        for (self_element, other_element) in zip(self_elements, other_elements))

  def most_specific_common_supertype(self, others):
    # [FUNCTION DOC]
    # Computes the most specific spec that is a supertype of this spec and every spec in others
    # (a sequence of specs). Returns none if any other fails the sanity check or has a different
    # structure, or if some element has no common supertype. Otherwise returns a new spec of the
    # same class with the combined element spec and this spec's workers, strategy, options,
    # cardinality and partial-batch flag (the replica order is not carried over).
    """Returns the most specific supertype of `self` and `others`.

    Args:
      others: A Sequence of `TypeSpec`.

    Returns `None` if a supertype does not exist.
    """
    try:
      for other in others:
        self.sanity_check_type(other)
        nest.assert_same_structure(self._element_spec, other._element_spec)  # pylint: disable=protected-access
    except (TypeError, ValueError):
      return None

    self_elements = nest.flatten(self._element_spec)
    others_elements = [nest.flatten(other._element_spec) for other in others]  # pylint: disable=protected-access
    common_elements = [None] * len(self_elements)

    for i, self_element in enumerate(self_elements):
      common_elements[i] = self_element.most_specific_common_supertype(
          [other_elements[i] for other_elements in others_elements])
      if common_elements[i] is None:
        return None
    common_element_spec = nest.pack_sequence_as(self._element_spec,
                                                common_elements)
    return type(self)(
        self._input_workers,
        common_element_spec,
        self._strategy,
        self._options,
        cardinality=self._cardinality,
        enable_get_next_as_optional=self._enable_get_next_as_optional)

  def _with_tensor_ranks_only(self):
    # [FUNCTION DOC]
    # Returns a new spec of the same class whose element spec keeps only tensor ranks (shapes
    # relaxed to unknown dimensions), with the other fields copied (replica order dropped).
    element_spec = nest.map_structure(
        lambda s: s._with_tensor_ranks_only(),  # pylint: disable=protected-access
        self._element_spec)
    return type(self)(
        self._input_workers,
        element_spec,
        self._strategy,
        self._options,
        cardinality=self._cardinality,
        enable_get_next_as_optional=self._enable_get_next_as_optional)

  # TODO(b/206014848): Remove once names are not used.
  def _without_tensor_names(self):
    # [FUNCTION DOC]
    # Returns a new spec of the same class whose element spec has tensor names removed, with the
    # other fields copied (replica order dropped). Temporary until names are no longer used.
    element_spec = nest.map_structure(
        lambda s: s._without_tensor_names(),  # pylint: disable=protected-access
        self._element_spec)
    return type(self)(
        self._input_workers,
        element_spec,
        self._strategy,
        self._options,
        cardinality=self._cardinality,
        enable_get_next_as_optional=self._enable_get_next_as_optional)


class DistributedIteratorSpec(DistributedDatasetAndIteratorSpec):
  # [CLASS DOC]
  # Type specification of a DistributedIterator. Its components are one single-worker iterator
  # spec per input worker. Used when a distributed iterator crosses tf.function boundaries.
  """Type specification for `DistributedIterator`."""

  @property
  def value_type(self):
    # [FUNCTION DOC]
    # Property returning the class of values this spec describes (DistributedIterator).
    return DistributedIterator

  @property
  def _component_specs(self):
    # [FUNCTION DOC]
    # Returns a list with one single-worker dataset-iterator spec per input worker, built from the
    # worker's input device, its compute devices, the element spec with per-replica specs replaced
    # by the worker's own piece, the options and the canonicalize-devices flag.
    specs = []
    worker_device_pairs = self._input_workers._worker_device_pairs  # pylint: disable=protected-access

    for i, (input_device, compute_devices) in enumerate(worker_device_pairs):
      element_spec = nest.map_structure(
          functools.partial(_replace_per_replica_spec, i=i), self._element_spec)
      specs.append(
          _SingleWorkerDatasetIteratorSpec(input_device, compute_devices,
                                           element_spec, self._options,
                                           self._canonicalize_devices))
    return specs

  def _to_components(self, value):
    # [FUNCTION DOC]
    # Returns the list of per-worker iterators of the given distributed iterator value.
    return value._iterators  # pylint: disable=protected-access

  def _from_components(self, components):
    # [FUNCTION DOC]
    # Rebuilds a DistributedIterator from per-worker iterator components, using this spec's
    # input workers, element spec, strategy, cardinality, partial-batch flag, options and
    # replica order. No iterators argument is given, so the components are used directly.
    return DistributedIterator(
        input_workers=self._input_workers,
        iterators=None,
        components=components,
        element_spec=self._element_spec,
        strategy=self._strategy,
        cardinality=self._cardinality,
        enable_get_next_as_optional=self._enable_get_next_as_optional,
        options=self._options,
        replica_order=self._replica_order,
    )

  @staticmethod
  def from_value(value):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the value's replica order is not copied into the new spec.
    # Static constructor: builds a spec from a DistributedIterator value by copying its input
    # workers, element spec, strategy, options, cardinality and partial-batch flag.
    # pylint: disable=protected-access
    return DistributedIteratorSpec(
        value._input_workers,
        value._element_spec,
        value._strategy,
        value._options,
        cardinality=value._cardinality,
        enable_get_next_as_optional=value._enable_get_next_as_optional)


class DistributedIterator(DistributedIteratorBase,
                          composite_tensor.CompositeTensor):
  # [CLASS DOC]
  # Iterator over a distributed dataset; the object returned by iterating a DistributedDataset or
  # DistributedDatasetsFromFunction. It is a composite tensor so it can be passed into
  # tf.function; its components are the per-worker iterators. Can be created either from live
  # iterators or from components plus an element spec (used when rebuilt from its spec).
  """Input Iterator for a distributed dataset."""

  def __init__(
      self,
      input_workers=None,
      iterators=None,
      strategy=None,
      components=None,
      element_spec=None,
      cardinality=cardinality_lib.UNKNOWN,
      enable_get_next_as_optional=False,
      options=None,
      replica_order=None,
  ):
    # [FUNCTION DOC]
    # Creates the iterator in one of two modes. Inputs: input_workers (required), iterators (list
    # of per-worker iterators), strategy, components and element_spec (used together instead of
    # iterators when rebuilding from a spec), cardinality (default unknown),
    # enable_get_next_as_optional (default false), options (input options) and replica_order.
    # Raises a value error if input_workers is missing, if iterators is absent and components or
    # element_spec is missing, or if iterators is given together with both components and
    # element_spec. With components, fields are set directly; otherwise the base constructor is used.
    if input_workers is None:
      raise ValueError("`input_workers` should be "
                       "provided.")

    error_message = ("Either `input_workers` or "
                     "both `components` and `element_spec` need to be "
                     "provided.")
    self._options = options

    if iterators is None:
      if (components is None or element_spec is None):
        raise ValueError(error_message)
      self._element_spec = element_spec
      self._input_workers = input_workers
      self._iterators = components
      self._strategy = strategy
      self._cardinality = cardinality
      self._enable_get_next_as_optional = enable_get_next_as_optional
      self._replica_order = replica_order
    else:
      if (components is not None and element_spec is not None):
        raise ValueError(error_message)

      super(DistributedIterator, self).__init__(
          input_workers,
          iterators,
          strategy,
          cardinality,
          enable_get_next_as_optional,
          replica_order,
      )

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the element spec of this iterator. When partial-batch handling is on and
    # the strategy runs in multi-worker mode, batch dimensions are made dynamic (unknown) because
    # empty batches may be produced; otherwise the stored element spec is returned unchanged.
    # When partial batch handling is enabled, always set the batch dimension to
    # None, otherwise we just follow element_spec of the underlying dataset
    # (whose batch dimension may also be None). This is because with partial
    # batching handling we could always produce empty batches.
    if (self._enable_get_next_as_optional and
        self._strategy.extended._in_multi_worker_mode()):  # pylint: disable=protected-access
      return nest.map_structure(
          _rebatch_as_dynamic, self._element_spec, expand_composites=False)
    return self._element_spec

  @property
  def _type_spec(self):
    # [FUNCTION DOC]
    # Returns the DistributedIteratorSpec for this iterator, built from the stored (not the
    # dynamic-batch) element spec plus workers, strategy, options, cardinality, partial-batch
    # flag and replica order, so it matches the underlying iterators' specs.
    # Note that we use actual element_spec instead of the rebatched-as-dynamic
    # one to create DistributedIteratorSpec, to be consistent with the
    # underlying iterators' specs.
    return DistributedIteratorSpec(
        self._input_workers,
        self._element_spec,
        self._strategy,
        self._options,
        self._cardinality,
        self._enable_get_next_as_optional,
        self._replica_order,
    )


class _IterableInput(collections_abc.Iterable,
                     distribute_types.DistributedDatasetInterface):
  # [CLASS DOC]
  # HELPER: private base class of DistributedDataset and DistributedDatasetsFromFunction. Holds
  # the InputWorkers and provides a reduce implementation on top of the iterator. Subclasses must
  # implement iteration.
  """Base class for iterable inputs for distribution strategies."""

  # pylint: disable=super-init-not-called
  def __init__(self, input_workers):
    # [FUNCTION DOC]
    # Stores the input workers; asserts that it is an InputWorkers. Returns nothing.
    assert isinstance(input_workers, InputWorkers)
    self._input_workers = input_workers

  def __iter__(self):
    # [FUNCTION DOC]
    # Abstract: always raises a not-implemented error; descendants must provide iteration.
    raise NotImplementedError("must be implemented in descendants")

  def reduce(self, initial_state, reduce_fn):
    # [FUNCTION DOC]
    # Folds reduce_fn over all elements of the input using a while loop. Inputs: initial_state
    # (starting state structure) and reduce_fn (callback taking the state and an element and
    # returning the new state). Creates an iterator, then loops (one parallel iteration at a
    # time) while the next optional has a value, getting the next optional each step.
    # Returns the final state structure.
    """Execute a `reduce_fn` over all the elements of the input."""
    iterator = iter(self)
    optional_data = iterator.get_next_as_optional()

    def cond(optional_data, state):
      # [FUNCTION DOC]
      # HELPER: while-loop condition for reduce; returns whether the optional holds a value. The
      # state argument is ignored.
      del state  # Unused.
      return optional_data.has_value()

    def loop_body(optional_data, state):
      # [FUNCTION DOC]
      # HELPER: while-loop body for reduce; applies the reduce callback to the state and the held
      # value, fetches the next optional, and returns the new optional and state as a pair.
      """Executes `reduce_fn` in a loop till the dataset is empty."""
      state = reduce_fn(state, optional_data.get_value())
      optional_data = iterator.get_next_as_optional()
      return optional_data, state

    optional_data, final_state = while_loop.while_loop(
        cond,
        loop_body, [optional_data, initial_state],
        parallel_iterations=1,
        return_same_structure=True)
    return final_state


class DistributedDatasetSpec(DistributedDatasetAndIteratorSpec):
  # [CLASS DOC]
  # Type specification of a DistributedDataset. Its components are one plain dataset spec per
  # input worker (the cloned per-worker datasets).
  """Type specification for `DistributedDataset."""

  @property
  def value_type(self):
    # [FUNCTION DOC]
    # Property returning the class of values this spec describes (DistributedDataset).
    return DistributedDataset

  @property
  def _component_specs(self):
    # [FUNCTION DOC]
    # Returns a list with one dataset spec per input worker, each for the element spec with the
    # per-replica specs replaced by that worker's piece.
    specs = []
    worker_device_pairs = self._input_workers._worker_device_pairs  # pylint: disable=protected-access

    for i, _ in enumerate(worker_device_pairs):
      element_spec = nest.map_structure(
          functools.partial(_replace_per_replica_spec, i=i), self._element_spec)
      specs.append(dataset_ops.DatasetSpec(element_spec))
    return specs

  def _to_components(self, value):
    # [FUNCTION DOC]
    # Returns the list of cloned per-worker datasets of the given distributed dataset value.
    return value._cloned_datasets  # pylint: disable=protected-access

  def _from_components(self, components):
    # [FUNCTION DOC]
    # Rebuilds a DistributedDataset from per-worker dataset components using this spec's input
    # workers, strategy, element spec, partial-batch flag, options and replica order.
    return DistributedDataset(
        input_workers=self._input_workers,
        strategy=self._strategy,
        components=components,
        element_spec=self._element_spec,
        enable_get_next_as_optional=self._enable_get_next_as_optional,
        options=self._options,
        replica_order=self._replica_order,
    )

  @staticmethod
  def from_value(value):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the value's replica order is not copied into the new spec.
    # Static constructor: builds a spec from a DistributedDataset value by copying its input
    # workers, element spec, strategy, options and partial-batch flag.
    # pylint: disable=protected-access
    return DistributedDatasetSpec(
        value._input_workers,
        value._element_spec,
        value._strategy,
        value._options,
        enable_get_next_as_optional=value._enable_get_next_as_optional)
    # pylint: enable=protected-access


class DistributedDataset(_IterableInput, composite_tensor.CompositeTensor):
  # [CLASS DOC]
  # A dataset distributed across the input workers of a strategy. From one source dataset it
  # creates one cloned, sharded (and optionally rebatched) dataset per worker, computes the
  # cardinality, whether partial-batch handling is needed, and the distributed element spec.
  # Iterating it yields a DistributedIterator. It is a composite tensor whose components are the
  # per-worker datasets. It can be built lazily (build flag) for parameter server training.
  """Distributed dataset that supports prefetching to multiple devices."""

  def __init__(
      self,
      input_workers,
      strategy,
      dataset=None,
      num_replicas_in_sync=None,
      input_context=None,
      components=None,
      element_spec=None,
      enable_get_next_as_optional=None,
      build=True,
      options=None,
      replica_order=None,
  ):
    # [FUNCTION DOC]
    # Creates the distributed dataset either from a source dataset or from existing per-worker
    # components. Inputs are described in the docstring; in short: exactly one of dataset or
    # components is required; num_replicas_in_sync and input_context affect rebatching and
    # sharding; element_spec and enable_get_next_as_optional are required with components; build
    # (default true) says whether to build the cloned datasets now; options and replica_order are
    # stored. Raises a value error if input workers or strategy are missing, if both or neither of
    # dataset and components are given, if components are used with build false, if
    # enable_get_next_as_optional is missing with components, or if the given element spec does not
    # match the distributed spec computed from the first component. With a dataset it stores the
    # original dataset and builds immediately when requested; with components it sets cloned
    # datasets, cardinality, element spec and marks itself built.
    """Distribute the dataset on all workers.

    If `num_replicas_in_sync` is not None, we split each batch of the dataset
    into `num_replicas_in_sync` smaller batches, to be distributed among that
    worker's replicas, so that the batch size for a global step (across all
    workers and replicas) is as expected.

    Args:
      input_workers: an `InputWorkers` object.
      strategy: a `tf.distribute.Strategy` object, used to run all-reduce to
        handle last partial batch.
      dataset: `tf.data.Dataset` that will be used as the input source. Either
        dataset or components field should be passed when constructing
        DistributedDataset. Use this when constructing DistributedDataset from a
        new `tf.data.Dataset`. Use components when constructing using
        DistributedDatasetSpec.
      num_replicas_in_sync: Optional integer. If this is not None, the value is
        used to decide how to rebatch datasets into smaller batches so that the
        total batch size for each step (across all workers and replicas) adds up
        to `dataset`'s batch size.
      input_context: `InputContext` for sharding. Only pass this in for between
        graph multi-worker cases where there is only one `input_worker`. In
        these cases, we will shard based on the `input_pipeline_id` and
        `num_input_pipelines` in the `InputContext`.
      components: datasets when DistributedDataset is constructed from
        DistributedDatasetSpec. Either field dataset or components should be
        passed.
      element_spec: element spec for DistributedDataset when constructing from
        DistributedDatasetSpec. This will be used to set the element_spec for
        DistributedDataset and verified against element_spec from components.
      enable_get_next_as_optional: this is required when components is passed
        instead of dataset.
      build: whether to build underlying datasets when this object is created.
        This is only useful for `ParameterServerStrategy` now.
      options: `tf.distribute.InputOptions` used to control options on how this
        dataset is distributed.
      replica_order: the order of the replicas, which will be used to reorder
        the iterators to match the device order.
    """
    super(DistributedDataset, self).__init__(input_workers=input_workers)
    if input_workers is None or strategy is None:
      raise ValueError("input_workers and strategy are required arguments")
    if dataset is not None and components is not None:
      raise ValueError("Only one of dataset or components should be present")
    if dataset is None and components is None:
      raise ValueError("At least one of dataset or components should be passed")

    self._input_workers = input_workers
    self._strategy = strategy
    self._options = options
    self._input_context = input_context
    self._num_replicas_in_sync = num_replicas_in_sync
    self._replica_order = replica_order

    if dataset is not None:
      self._original_dataset = dataset
      self._built = False
      if build:
        self.build()
    else:
      if not build:
        raise ValueError(
            "When constructing DistributedDataset with components, build "
            "should not be False. This is an internal error. Please file a "
            "bug.")
      if enable_get_next_as_optional is None:
        raise ValueError(
            "When constructing DistributedDataset with components, " +
            "enable_get_next_as_optional should also be passed")
      self._cloned_datasets = components
      self._cardinality = _cardinality(self._cloned_datasets[0])
      self._enable_get_next_as_optional = enable_get_next_as_optional

      assert element_spec is not None
      if element_spec != _create_distributed_tensor_spec(
          self._strategy, self._cloned_datasets[0].element_spec):
        raise ValueError("Mismatched element_spec from the passed components")
      self._element_spec = element_spec

      self._built = True

  def build(self, dataset_to_replace=None):
    # [FUNCTION DOC]
    # Builds the per-worker datasets. Input: dataset_to_replace, optional dataset used instead of
    # the original one. Asserts the object has not been built yet. Computes the cardinality and
    # the partial-batch flag, clones and shards the dataset for each worker, and in eager mode
    # waits for async work and records the elapsed initialization time (milliseconds) in the
    # monitoring sampler labeled by strategy class name and worker count. Then computes the
    # distributed element spec and marks the object built. Returns nothing.
    assert not self._built
    dataset = dataset_to_replace or self._original_dataset
    self._cardinality = _cardinality(dataset)
    self._enable_get_next_as_optional = _enable_get_next_as_optional(
        self._strategy, dataset, self._cardinality)
    distribute_start_time_ns = time.time_ns()
    self._create_cloned_datasets_from_dataset(dataset, self._input_context,
                                              self._input_workers,
                                              self._strategy,
                                              self._num_replicas_in_sync)
    if context.executing_eagerly():
      # Records the time to initialize the distributed dataset.
      context.async_wait()
      distribute_duration_ms = (time.time_ns() -
                                distribute_start_time_ns) // 1_000_000
      _distributed_dataset_initialization_time_milliseconds.get_cell(
          self._strategy.__class__.__name__,
          str(self._input_workers.num_workers)).add(distribute_duration_ms)
    self._element_spec = _create_distributed_tensor_spec(
        self._strategy, self._cloned_datasets[0].element_spec)
    self._built = True

  def auto_shard(self, num_shards, shard_ix):
    # [FUNCTION DOC]
    # Returns a new DistributedDataset in which every per-worker cloned dataset is further
    # auto-sharded into num_shards pieces, keeping piece number shard_ix (both integers). The
    # sharding op is colocated with each dataset. Asserts that the number of cloned datasets
    # equals the number of input workers. The result is built from components using the same
    # element spec, options and partial-batch flag.
    assert (
        len(self._cloned_datasets) == len(self._input_workers.worker_devices)
    ), (
        f"datasets: {len(self._cloned_datasets)}, "
        f"input workers: {len(self._input_workers.worker_devices)}"
    )
    sharded_datasets = []
    for i in range(len(self._input_workers.worker_devices)):
      with ops.colocate_with(self._cloned_datasets[i]._variant_tensor):  # pylint:disable=protected-access
        sharded_datasets.append(
            input_ops.auto_shard_dataset(
                self._cloned_datasets[i], num_shards, shard_ix,
                self._num_replicas_in_sync
                ))
    return DistributedDataset(
        self._input_workers,
        self._strategy,
        components=sharded_datasets,
        element_spec=self._element_spec,
        options=self._options,
        enable_get_next_as_optional=self._enable_get_next_as_optional)

  @property
  def cardinality(self):
    # [FUNCTION DOC]
    # Property returning the dataset cardinality computed at build time. Raises a value error if
    # the dataset has not been built yet.
    if not self._built:
      raise ValueError(
          "Cannot get the cardinality of a dataset that is not built")
    return self._cardinality

  def _create_cloned_datasets_from_dataset(self, dataset, input_context,
                                           input_workers, strategy,
                                           num_replicas_in_sync):
    # [FUNCTION DOC]
    # HELPER: used by build. Fills the list of cloned datasets, one per worker. Inputs: dataset,
    # input_context (optional; when given, a single between-graph worker shards by the
    # context's pipeline id), input_workers, strategy (unused here) and num_replicas_in_sync
    # (when above one, a rebatch function splits each batch among a worker's replicas). With an
    # input context it asserts exactly one worker, rebatches and auto-shards the dataset.
    # Otherwise it replicates the dataset to the worker devices and, on each worker device,
    # rebatches (if needed) and auto-shards it by worker index. Sets the cloned datasets
    # attribute; returns nothing.
    # We clone and shard the dataset on each worker. The current setup tries to
    # shard the dataset by files if possible so that each worker sees a
    # different subset of files. If that is not possible, will attempt to shard
    # the final input such that each worker will run the entire preprocessing
    # pipeline and only receive its own shard of the dataset.

    # Additionally, we rebatch the dataset on each worker into
    # `num_replicas_in_sync` smaller batches to be distributed among that
    # worker's replicas, so that the batch size for a global step (across all
    # workers and replicas) adds up to the original dataset's batch size.
    if num_replicas_in_sync is not None and num_replicas_in_sync > 1:
      num_workers = input_context.num_input_pipelines if input_context else len(
          input_workers.worker_devices)
      rebatch_fn = self._make_rebatch_fn(dataset, num_workers,
                                         num_replicas_in_sync)
    else:
      rebatch_fn = None
    self._cloned_datasets = []
    if input_context:
      # Between-graph where we rely on the input_context for sharding
      assert input_workers.num_workers == 1
      if rebatch_fn is not None:
        dataset = rebatch_fn(dataset, input_context.input_pipeline_id)
      dataset = input_ops.auto_shard_dataset(dataset,
                                             input_context.num_input_pipelines,
                                             input_context.input_pipeline_id,
                                             num_replicas_in_sync)
      self._cloned_datasets.append(dataset)
    else:
      replicated_ds = distribute.replicate(dataset,
                                           input_workers.worker_devices)
      for i, worker in enumerate(input_workers.worker_devices):
        with ops.device(worker):
          cloned_dataset = replicated_ds[worker]
          if rebatch_fn is not None:
            cloned_dataset = rebatch_fn(cloned_dataset, i)
          cloned_dataset = input_ops.auto_shard_dataset(
              cloned_dataset, len(input_workers.worker_devices), i,
              num_replicas_in_sync)
          self._cloned_datasets.append(cloned_dataset)

  def _make_rebatch_fn(self, dataset, num_workers, num_replicas_in_sync):
    # [FUNCTION DOC]
    # HELPER: used when cloning datasets. Returns a callback taking a dataset and a worker index
    # that splits its batches among that worker's replicas. Inputs: dataset (used to compute the
    # global batch size), num_workers and num_replicas_in_sync (integers). Raises a value error if
    # the replica count is not divisible by the worker count. The returned callback conditionally
    # uses the modern rebatch (when the batch size is known) or the legacy rebatch, then prefetches;
    # it converts an invalid-argument error about a missing batch into a value error telling the
    # user to call batch on the dataset.
    """Returns a callable that rebatches the input dataset.

    Args:
      dataset: A `tf.data.Dataset` representing the dataset to be distributed.
      num_workers: An integer representing the number of workers to distribute
        `dataset` among.
      num_replicas_in_sync: An integer representing the number of replicas in
        sync across all workers.
    """
    if num_replicas_in_sync % num_workers:
      raise ValueError(
          "tf.distribute expects every worker to have the same number of "
          "replicas. However, encountered `num_replicas_in_sync` ({}) that "
          "cannot be divided by `num_workers` ({})".format(
              num_replicas_in_sync, num_workers))

    num_replicas_per_worker = num_replicas_in_sync // num_workers
    with ops.colocate_with(dataset._variant_tensor):  # pylint: disable=protected-access
      batch_size = distribute.compute_batch_size(dataset)

    def rebatch_fn(dataset, worker_index):
      # [FUNCTION DOC]
      # HELPER: the rebatching callback returned by _make_rebatch_fn. Takes a dataset and the
      # worker index and returns the rebatched, prefetched dataset (see the enclosing function).
      try:

        def apply_rebatch():
          # [FUNCTION DOC]
          # HELPER: true-branch of the conditional in rebatch_fn. Computes per-replica batch sizes
          # for this worker and returns the dataset rebatched to them, prefetching one batch per
          # replica of the worker.
          batch_sizes = distribute.batch_sizes_for_worker(
              batch_size, num_workers, num_replicas_per_worker, worker_index)
          return dataset.rebatch(batch_sizes).prefetch(num_replicas_per_worker)

        # pylint: disable=protected-access
        def apply_legacy_rebatch():
          # [FUNCTION DOC]
          # HELPER: false-branch of the conditional in rebatch_fn, used when the batch size is not
          # statically known (equals minus one). Returns the dataset passed through the legacy
          # rebatch dataset for the total replica count, then prefetched.
          return distribute._LegacyRebatchDataset(
              dataset, num_replicas_in_sync).prefetch(num_replicas_per_worker)

        with ops.colocate_with(dataset._variant_tensor):
          return tf_cond.cond(
              math_ops.not_equal(batch_size, -1),
              true_fn=apply_rebatch,
              false_fn=apply_legacy_rebatch)
      except errors.InvalidArgumentError as e:
        if "without encountering a batch" in str(e):
          six.reraise(
              ValueError,
              ValueError(
                  "Call the `batch` method on the input Dataset in order to be "
                  "able to split your input across {} replicas.\n Please see "
                  "the tf.distribute.Strategy guide. {}".format(
                      num_replicas_in_sync, e)),
              sys.exc_info()[2])
        else:
          raise

    return rebatch_fn

  def __iter__(self):
    # [FUNCTION DOC]
    # Creates and returns a DistributedIterator over the cloned datasets, with one owned
    # per-worker iterator each. Raises a runtime error if called neither eagerly nor while
    # tracing a function, and a value error if the dataset was not built (it must go through the
    # cluster coordinator's per-worker dataset creation). The iterator gets this dataset's
    # element spec, cardinality, partial-batch flag, options and replica order. In eager mode it
    # waits for asynchronous work so the underlying iterators are initialized.
    if not (context.executing_eagerly() or
            ops.get_default_graph().building_function):
      raise RuntimeError("__iter__() is only supported inside of tf.function "
                         "or when eager execution is enabled.")
    if not self._built:
      raise ValueError("To use this dataset, you need to pass this dataset to "
                       "ClusterCoordinator.create_per_worker_dataset.")

    canonicalize_devices = getattr(self._strategy, "_canonicalize_devices",
                                   True)

    worker_iterators = _create_iterators_per_worker(
        self._cloned_datasets,
        self._input_workers,
        options=self._options,
        canonicalize_devices=canonicalize_devices)
    iterator = DistributedIterator(
        self._input_workers,
        worker_iterators,
        self._strategy,
        cardinality=self._cardinality,
        enable_get_next_as_optional=self._enable_get_next_as_optional,
        options=self._options,
        replica_order=self._replica_order,
    )
    iterator._element_spec = self._element_spec  # pylint: disable=protected-access

    # When async eager is enabled, sometimes the iterator may not finish
    # initialization before passing to a multi device function, add a sync point
    # here to make sure all underlying iterators are initialized.
    if context.executing_eagerly():
      context.async_wait()

    return iterator

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the element spec of this dataset; in multi-worker mode with partial-batch
    # handling the batch dimensions are made dynamic, otherwise the stored spec is returned.
    """The type specification of an element of this dataset."""
    # When partial batch handling is enabled, always set the batch dimension to
    # None, otherwise we just follow element_spec of the underlying dataset
    # (whose batch dimension may also be None). This is because with partial
    # batching handling we could always produce empty batches.
    if (self._enable_get_next_as_optional and
        self._strategy.extended._in_multi_worker_mode()):  # pylint: disable=protected-access
      return nest.map_structure(
          _rebatch_as_dynamic, self._element_spec, expand_composites=False)
    return self._element_spec

  @property
  def _type_spec(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the replica order is not included in the returned spec.
    # Returns the DistributedDatasetSpec for this dataset (workers, stored element spec,
    # strategy, options and partial-batch flag).
    return DistributedDatasetSpec(
        self._input_workers,
        self._element_spec,
        self._strategy,
        self._options,
        enable_get_next_as_optional=self._enable_get_next_as_optional)


class DistributedDatasetsFromFunctionSpec(DistributedDatasetAndIteratorSpec):
  # [CLASS DOC]
  # Type specification of a DistributedDatasetsFromFunction. Its components are one plain dataset
  # spec per input worker.
  """Type specification for `DistributedDatasetsFromFunction."""

  @property
  def value_type(self):
    # [FUNCTION DOC]
    # Property returning the class of values this spec describes (DistributedDatasetsFromFunction).
    return DistributedDatasetsFromFunction

  @property
  def _component_specs(self):
    # [FUNCTION DOC]
    # Returns a list with one dataset spec per input worker, each for the element spec with the
    # per-replica specs replaced by that worker's piece.
    specs = []
    worker_device_pairs = self._input_workers._worker_device_pairs  # pylint: disable=protected-access

    for i, _ in enumerate(worker_device_pairs):
      element_spec = nest.map_structure(
          functools.partial(_replace_per_replica_spec, i=i), self._element_spec)
      specs.append(dataset_ops.DatasetSpec(element_spec))
    return specs

  def _to_components(self, value):
    # [FUNCTION DOC]
    # Returns the list of per-worker datasets held by the given value.
    return value._datasets  # pylint: disable=protected-access

  def _from_components(self, components):
    # [FUNCTION DOC]
    # Rebuilds a DistributedDatasetsFromFunction from per-worker dataset components, using this
    # spec's input workers, strategy, element spec and options.
    return DistributedDatasetsFromFunction(
        input_workers=self._input_workers,
        strategy=self._strategy,
        components=components,
        element_spec=self._element_spec,
        options=self._options)

  @staticmethod
  def from_value(value):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the value's replica order is not copied into the new spec.
    # Static constructor: builds a spec from a DistributedDatasetsFromFunction value by copying
    # its input workers, element spec, strategy and options.
    # pylint: disable=protected-access
    return DistributedDatasetsFromFunctionSpec(
        input_workers=value._input_workers,
        element_spec=value._element_spec,
        strategy=value._strategy,
        options=value._options)


# TODO(priyag): Add other replication modes.
class DistributedDatasetsFromFunction(_IterableInput,
                                      composite_tensor.CompositeTensor):
  # [CLASS DOC]
  # Distributed inputs created by calling a user dataset function once per input worker with that
  # worker's input context (so the function does its own sharding and batching). Holds one
  # dataset per worker, the element spec, cardinality, partial-batch flag, options and replica
  # order. Iterating yields a DistributedIterator. It is a composite tensor whose components are
  # the per-worker datasets, and can be built lazily.
  """Inputs created from dataset function."""

  def __init__(
      self,
      input_workers,
      strategy,
      input_contexts=None,
      dataset_fn=None,
      options=None,
      components=None,
      element_spec=None,
      build=True,
      replica_order=None,
  ):
    # [FUNCTION DOC]
    # Creates the object from a dataset function or from existing per-worker components. Exactly
    # one of dataset_fn and components is required. With dataset_fn, input_contexts must have one
    # context per input worker (a value error is raised otherwise); the contexts, function and
    # replica count are stored and the datasets are built immediately when build (default true)
    # is set. With components, element_spec is required (value error otherwise), build must not be
    # false (value error), and the datasets, cardinality and partial-batch flag are set directly
    # and the object is marked built. Raises a value error if both or neither of dataset_fn and
    # components are given. Also stores input workers, strategy, options and replica order.
    """Makes an iterable from datasets created by the given function.

    Args:
      input_workers: an `InputWorkers` object.
      strategy: a `tf.distribute.Strategy` object, used to run all-reduce to
        handle last partial batch.
      input_contexts: A list of `InputContext` instances to be passed to call(s)
        to `dataset_fn`. Length and order should match worker order in
        `worker_device_pairs`.
      dataset_fn: A function that returns a `Dataset` given an `InputContext`.
        Either dataset_fn or components should be passed to construct
        DistributedDatasetsFromFunction. Use this when constructing
        DistributedDataset using a function. Use components when constructing
        using DistributedDatasetsFromFunctionSpec.
      options: `tf.distribute.InputOptions` used to control options on how this
        dataset is distributed.
      components: datasets when DistributedDatasetsFromFunction is constructed
        from DistributedDatasetsFromFunctionSpec. Only one of dataset or
        components should be passed.
      element_spec: element spec for DistributedDataset when constructing from
        DistributedDatasetSpec. This will be used to set the element_spec for
        DistributedDatasetsFromFunctionSpec and verified against element_spec
        from components.
      build: whether to build underlying datasets when this object is created.
        This is only useful for `ParameterServerStrategy` now.
      replica_order: the order of the replicas, which will be used to reorder
        the iterators to match the device order.
    """
    super(DistributedDatasetsFromFunction, self).__init__(
        input_workers=input_workers)
    self._input_workers = input_workers
    self._strategy = strategy
    self._options = options
    self._replica_order = replica_order
    if dataset_fn is not None and components is not None:
      raise ValueError("Only one of dataset_fn or components should be set")
    if dataset_fn is None and components is None:
      raise ValueError("At least one of dataset_fn or components should be set")

    if dataset_fn is not None:
      if input_workers.num_workers != len(input_contexts):
        raise ValueError(
            "Number of input workers (%d) is not same as number of "
            "input_contexts (%d)" %
            (input_workers.num_workers, len(input_contexts)))
      self._input_contexts = input_contexts
      self._num_replicas_in_sync = self._input_contexts[0].num_replicas_in_sync
      self._dataset_fn = dataset_fn
      self._built = False
      if build:
        self.build()
    else:
      if element_spec is None:
        raise ValueError(
            "element_spec should also be passed when passing components")
      if not build:
        raise ValueError(
            "When constructing DistributedDatasetFromFunction with components, "
            "build should not be False. This is an internal error. Please file "
            "a bug.")
      self._element_spec = element_spec
      self._datasets = components
      self._num_replicas_in_sync = None
      self._built = True
      self._cardinality = _cardinality(self._datasets[0])
      self._enable_get_next_as_optional = _enable_get_next_as_optional(
          self._strategy, self._datasets[0], self._cardinality)

  def build(self):
    # [FUNCTION DOC]
    # Builds the per-worker datasets by calling the dataset function with each worker's input
    # context on that worker's device. Asserts the object is not yet built. In eager mode waits
    # for async work and records the elapsed time (milliseconds) in the from-function monitoring
    # sampler. Then sets the distributed element spec, cardinality and partial-batch flag and
    # marks the object built. Returns nothing.
    assert not self._built
    distribute_start_time_ns = time.time_ns()
    self._datasets, element_spec = (
        _create_datasets_from_function_with_input_context(
            self._input_contexts, self._input_workers, self._dataset_fn))
    if context.executing_eagerly():
      # Records the time to initialize the distributed dataset.
      context.async_wait()
      distribute_duration_ms = (time.time_ns() -
                                distribute_start_time_ns) // 1_000_000
      _distributed_dataset_from_function_initialization_time_milliseconds.get_cell(
          self._strategy.__class__.__name__,
          str(self._input_workers.num_workers)).add(distribute_duration_ms)

    self._element_spec = _create_distributed_tensor_spec(
        self._strategy, element_spec)
    self._cardinality = _cardinality(self._datasets[0])
    self._enable_get_next_as_optional = _enable_get_next_as_optional(
        self._strategy, self._datasets[0], self._cardinality)
    self._built = True

  def auto_shard(self, num_shards, shard_ix):
    # [FUNCTION DOC]
    # Returns a new DistributedDatasetsFromFunction whose per-worker datasets are each
    # auto-sharded into num_shards pieces keeping piece shard_ix (integers). Asserts that the
    # number of datasets equals the number of input workers. The result is built from components
    # with the same element spec and options.
    assert (
        len(self._datasets) == len(self._input_workers.worker_devices)
    ), (
        f"datasets: {len(self._datasets)}, "
        f"input workers: {len(self._input_workers.worker_devices)}"
    )
    sharded_datasets = []
    for i in range(len(self._input_workers.worker_devices)):
      with ops.colocate_with(self._datasets[i]._variant_tensor):  # pylint: disable=protected-access
        sharded_datasets.append(
            input_ops.auto_shard_dataset(
                self._datasets[i], num_shards, shard_ix,
                self._num_replicas_in_sync
            )
        )
    return DistributedDatasetsFromFunction(self._input_workers, self._strategy,
                                           components=sharded_datasets,
                                           element_spec=self._element_spec,
                                           options=self._options)

  @property
  def cardinality(self):
    # [FUNCTION DOC]
    # Property returning the dataset cardinality computed at build time. Raises a value error if
    # the object has not been built yet.
    if not self._built:
      raise ValueError(
          "Cannot get the cardinality of a dataset that is not built")
    return self._cardinality

  def __iter__(self):
    # [FUNCTION DOC]
    # Creates and returns a DistributedIterator over the per-worker datasets. Raises a runtime
    # error if not executing eagerly (outside functions) and not building a function, and a value
    # error if the object was not built. The iterator gets this object's element spec,
    # cardinality, partial-batch flag, options and replica order; in eager mode it waits for
    # async work so the underlying iterators are initialized.
    if not (ops.executing_eagerly_outside_functions() or
            ops.get_default_graph().building_function):
      raise RuntimeError("__iter__() is only supported inside of tf.function "
                         "or when eager execution is enabled.")

    if not self._built:
      raise ValueError("You need to use this dataset in "
                       "ClusterCoordinator.create_per_worker_dataset.")

    canonicalize_devices = getattr(self._strategy, "_canonicalize_devices",
                                   True)

    iterators = _create_iterators_per_worker(
        self._datasets,
        self._input_workers,
        options=self._options,
        canonicalize_devices=canonicalize_devices)
    iterator = DistributedIterator(
        input_workers=self._input_workers,
        iterators=iterators,
        strategy=self._strategy,
        cardinality=self._cardinality,
        enable_get_next_as_optional=self._enable_get_next_as_optional,
        options=self._options,
        replica_order=self._replica_order,
    )
    iterator._element_spec = self._element_spec  # pylint: disable=protected-access

    # When async eager is enabled, sometimes the iterator may not finish
    # initialization before passing to a multi device function, add a sync
    # point here to make sure all underlying iterators are initialized.
    if context.executing_eagerly():
      context.async_wait()

    return iterator

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the element spec; in multi-worker mode with partial-batch handling the
    # batch dimensions are made dynamic, otherwise the stored spec is returned.
    """The type specification of an element of this dataset."""
    # When partial batch handling is enabled, always set the batch dimension to
    # None, otherwise we just follow element_spec of the underlying dataset
    # (whose batch dimension may also be None). This is because with partial
    # batching handling we could always produce empty batches.
    if (self._enable_get_next_as_optional and
        self._strategy.extended._in_multi_worker_mode()):  # pylint: disable=protected-access
      return nest.map_structure(
          _rebatch_as_dynamic, self._element_spec, expand_composites=False)
    return self._element_spec

  @property
  def _type_spec(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the replica order is not included in the returned spec.
    # Returns the DistributedDatasetsFromFunctionSpec (workers, element spec, strategy, options).
    return DistributedDatasetsFromFunctionSpec(self._input_workers,
                                               self._element_spec,
                                               self._strategy, self._options)


def _dummy_tensor_fn(value_structure):
  # [FUNCTION DOC]
  # HELPER: partial-batch handling. Maps a nested structure of type specs to a structure of
  # empty placeholder values (zero-size batch dimension) used for replicas that ran out of data.
  # Input: value_structure, nested structure of specs. Returns the matching nested structure.
  """A function to create dummy tensors from `value_structure`."""

  def create_dummy_tensor(spec):
    # [FUNCTION DOC]
    # HELPER: creates one placeholder value for a single type spec. If the spec offers its own
    # empty-value creator, that is used. Otherwise computes the feature shape (for ragged specs
    # the ragged dimensions are spliced out), replaces unknown dimensions with zero and sets the
    # presumed batch (first) dimension to zero when the shape is static or the spec is ragged.
    # Sparse specs give an empty sparse tensor; ragged specs give an empty ragged tensor with
    # zero-size splits; others give a zeros tensor of that shape and dtype.
    """Create a dummy tensor with possible batch dimensions set to 0."""
    if hasattr(spec, "_create_empty_value"):
      # Type spec may overwrite default dummy values behavior by declaring the
      # `_create_empty_value(self)` method. This method must return a value
      # compatible with the type spec with batch dimensions set to 0 or fail if
      # such a value does not exist. This allows a composite tensor to customize
      # dummy values creation as, in general, its dummy value is not composed
      # from dummy components (e.g. `row_splits` tensor of a RaggedTensor is
      # never allowed to be empty). See b/183969859 for more discussions.
      # TODO(b/186079336): reconsider CompositeTensor support.
      return spec._create_empty_value()  # pylint: disable=protected-access

    if isinstance(spec, ragged_tensor.RaggedTensorSpec):
      # Splice out the ragged dimensions.
      # pylint: disable=protected-access
      feature_shape = spec._shape[:1].concatenate(
          spec._shape[(1 + spec._ragged_rank):])
      feature_type = spec._dtype
      # pylint: enable=protected-access
    else:
      feature_shape = spec.shape
      feature_type = spec.dtype
    # Ideally we should set the batch dimension to 0, however as in
    # DistributionStrategy we don't know the batch dimension, we try to
    # guess it as much as possible. If the feature has unknown dimensions, we
    # will set them to 0. If the feature shape is already static, we guess the
    # first dimension as batch dimension and set it to 0.
    dims = ([dim if dim is not None else 0 for dim in feature_shape.as_list()]
            if feature_shape else [])
    if dims and (isinstance(spec, ragged_tensor.RaggedTensorSpec) or
                 feature_shape.is_fully_defined()):
      dims[0] = tensor_shape.Dimension(0)

    if isinstance(spec, sparse_tensor.SparseTensorSpec):
      return sparse_tensor.SparseTensor(
          values=array_ops.zeros(0, feature_type),
          indices=array_ops.zeros((0, len(dims)), dtypes.int64),
          dense_shape=dims)

    # Create the dummy tensor.
    dummy_tensor = array_ops.zeros(tensor_shape.TensorShape(dims), feature_type)
    if isinstance(spec, ragged_tensor.RaggedTensorSpec):
      # Reinsert the ragged dimensions with size 0.
      # pylint: disable=protected-access
      row_splits = array_ops.zeros(1, spec._row_splits_dtype)
      dummy_tensor = ragged_tensor.RaggedTensor.from_nested_row_splits(
          dummy_tensor, (row_splits,) * spec._ragged_rank, validate=False)
      # pylint: enable=protected-access
    return dummy_tensor

  return nest.map_structure(create_dummy_tensor, value_structure)


def _get_value_or_dummy(input_workers, optional_list, produce_dummy):
  # [FUNCTION DOC]
  # HELPER: partial-batch handling. Flattens the optionals into a list of values, placing ops on
  # each worker and compute device. Inputs: input_workers, optional_list (per worker lists of
  # optionals) and produce_dummy (boolean). With produce_dummy, each entry conditionally gives the
  # optional's value or a dummy empty value when it has none; otherwise it always reads the value
  # (which fails for an empty optional). Returns the flat list of values.
  """Returns the value of the optionals or dummy values.

  Args:
    input_workers: the `InputWorkers`.
    optional_list: a list of lists `tf.experimental.Optional`. The values from
      each compute device grouped by the input device.
    produce_dummy: a bool. Whether to produce dummy tensors when the optional
      doesn't have a value.

  Returns:
    A flatten list of Tensors.

  """
  value_list = []
  for i, worker in enumerate(input_workers.worker_devices):
    with ops.device(worker):
      devices = input_workers.compute_devices_for_worker(i)
      for j, device in enumerate(devices):
        with ops.device(device):
          if produce_dummy:
            # pylint: disable=cell-var-from-loop
            value_list.append(
                tf_cond.cond(
                    optional_list[i][j].has_value(),
                    lambda: optional_list[i][j].get_value(),  # pylint: disable=unnecessary-lambda
                    lambda: _dummy_tensor_fn(optional_list[i][j].element_spec),
                    strict=True,
                ))
            # pylint: enable=cell-var-from-loop
          else:
            value_list.append(optional_list[i][j].get_value())
  return value_list


class _SingleWorkerDatasetIteratorBase(object):
  # [CLASS DOC]
  # HELPER: private base class for an iterator over a single dataset on one worker, feeding that
  # worker's compute devices. Holds the dataset, worker device, devices, element spec and
  # options, and delegates to an iterator created by the subclass's make-iterator method.
  """Iterator for a single `tf.data.Dataset`."""

  def __init__(self, dataset, worker, devices, options=None):
    # [FUNCTION DOC]
    # Stores the dataset, worker device, compute devices, the dataset's element spec and options
    # (optional input options, default none), then calls the subclass hook to create the
    # underlying iterator. Returns nothing.
    """Create iterator for the `dataset` to fetch data to worker's `devices` .

    A `MultiDeviceIterator`  or `OwnedMultiDeviceIterator` is used to prefetch
    input to the devices on the given worker.

    Args:
      dataset: A `tf.data.Dataset` instance.
      worker: Worker on which ops should be created.
      devices: Distribute data from `dataset` to these devices.
      options: options.
    """
    self._dataset = dataset
    self._worker = worker
    self._devices = devices
    self._element_spec = dataset.element_spec
    self._options = options
    self._make_iterator()

  def _make_iterator(self):
    # [FUNCTION DOC]
    # Abstract hook: always raises a not-implemented error; subclasses create the iterator.
    raise NotImplementedError("must be implemented in descendants")

  def _format_data_list_with_options(self, data_list):
    # [FUNCTION DOC]
    # HELPER: normalizes iterator output to a list. When the replication mode is per-replica and
    # fetching to the device is disabled, the iterator returns data without the enclosing list,
    # so the data is wrapped in a one-item list; otherwise it is returned unchanged.
    """Change the data in to a list type if required.

    The OwnedMultiDeviceIterator returns the list data type,
    while the PER_REPLICA iterator (when used with prefetch disabled)
    returns without the enclosed list. This is to fix the inconsistency.
    Args:
      data_list: data_list
    Returns:
      list
    """
    if (self._options and self._options.experimental_replication_mode ==
        InputReplicationMode.PER_REPLICA and
        not self._options.experimental_fetch_to_device):
      return [data_list]
    else:
      return data_list

  def get_next(self, device, name=None):
    # [FUNCTION DOC]
    # Returns the next element for one compute device, with ops placed on the worker device.
    # Inputs: device (compute device string, only used with the multi-device iterator) and name
    # (ignored). With a multi-device iterator the element for that device is fetched; otherwise
    # the plain iterator's next element is returned.
    """Get next element for the given device."""
    del name
    with ops.device(self._worker):
      if _should_use_multi_device_iterator(self._options):
        return self._iterator.get_next(device)
      else:
        return self._iterator.get_next()

  def get_next_as_list(self, name=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says this is faster than get_next_as_list (itself); it presumably
    # means get_next_as_optional_list.
    # Returns the next elements from the underlying iterator as a list, one per device, with ops
    # placed on the worker device (result normalized by the options-based formatting). Input:
    # name is ignored. Raises the end-of-sequence error if any device has no data.
    """Get next element from the underlying iterator.

    Runs the iterator get_next() within a device scope. Since this doesn't use
    get_next_as_optional(), it is considerably faster than get_next_as_list(),
    but it raises EOFError if any of the device doesn't get any data.

    Args:
      name: not used.

    Returns:
      A list consisting of the next data from each device.
    """
    del name
    with ops.device(self._worker):
      return self._format_data_list_with_options(self._iterator.get_next())

  def get_next_as_optional_list(self):
    # [FUNCTION DOC]
    # Returns the underlying iterator's next-as-optional result (one optional per device) as a
    # list, with ops placed on the worker device. Does not raise at the end of data; the
    # optionals are empty instead.
    with ops.device(self._worker):
      return self._format_data_list_with_options(
          self._iterator.get_next_as_optional())


class _SingleWorkerDatasetIteratorSpec(type_spec.TypeSpec):
  # [CLASS DOC]
  # HELPER: type specification of a single-worker owned dataset iterator. Holds the worker
  # device, compute devices (canonicalized), element spec, options and canonicalize flag. Its
  # component is one iterator spec: a multi-device iterator spec or a plain iterator spec.
  """Type specification for `_SingleWorkerOwnedDatasetIterator`."""

  __slots__ = [
      "_worker", "_devices", "_element_spec", "_options",
      "_canonicalize_devices"
  ]

  def __init__(self, worker, devices, element_spec, options,
               canonicalize_devices=True):
    # [FUNCTION DOC]
    # Stores the worker device, the compute devices (fully canonicalized, or with job and task
    # removed when canonicalize_devices is false; default true), the element spec, and the
    # options (a default input-options object is used when none is given so serialization works).
    self._worker = worker
    if canonicalize_devices:
      self._devices = tuple(device_util.canonicalize(d) for d in devices)
    else:
      self._devices = tuple(
          device_util.canonicalize_without_job_and_task(d) for d in devices)
    self._element_spec = element_spec
    # `self._options` intentionally made not `None` for proper serialization.
    self._options = (options if options is not None else
                     distribute_lib.InputOptions())
    self._canonicalize_devices = canonicalize_devices

  @property
  def value_type(self):
    # [FUNCTION DOC]
    # Property returning the class of values this spec describes (the owned single-worker iterator).
    return _SingleWorkerOwnedDatasetIterator

  def _serialize(self):
    # [FUNCTION DOC]
    # Returns a tuple of worker, devices, element spec, options and the canonicalize flag.
    return (self._worker, self._devices, self._element_spec, self._options,
            self._canonicalize_devices)

  def _get_multi_device_iterator_spec(self, specs):
    # [FUNCTION DOC]
    # HELPER: for _component_specs. Appends to the given list a multi-device iterator spec for
    # this spec's devices and element spec. The source device is the host device of the worker
    # canonicalized against the current device scope. Mutates the list; returns nothing.
    device_scope = device_util.canonicalize(self._worker, device_util.current())
    host_device = device_util.get_host_for_device(device_scope)
    # source_device while creating iterator governs the worker device in
    # iterator spec.
    worker = host_device
    specs.append(
        multi_device_iterator_ops.MultiDeviceIteratorSpec(
            self._devices, worker, element_spec=self._element_spec))

  @property
  def _component_specs(self):
    # [FUNCTION DOC]
    # Returns a one-item list: a multi-device iterator spec when the options call for a
    # multi-device iterator, else a plain iterator spec for the element spec.
    specs = []
    if _should_use_multi_device_iterator(self._options):
      self._get_multi_device_iterator_spec(specs)
    else:
      specs.append(iterator_ops.IteratorSpec(element_spec=self._element_spec))
    return specs

  def _to_components(self, value):
    # [FUNCTION DOC]
    # Returns a one-item list holding the value's underlying iterator.
    return [value._iterator]  # pylint: disable=protected-access

  def _from_components(self, components):
    # [FUNCTION DOC]
    # Rebuilds an owned single-worker iterator from its components (no dataset), using this
    # spec's worker, devices, element spec, options and canonicalize flag.
    return _SingleWorkerOwnedDatasetIterator(
        dataset=None,
        worker=self._worker,
        devices=self._devices,
        components=components,
        element_spec=self._element_spec,
        options=self._options,
        canonicalize_devices=self._canonicalize_devices)

  @staticmethod
  def from_value(value):
    # [FUNCTION DOC]
    # Static constructor: builds a spec from an owned single-worker iterator by copying its
    # worker, devices, element spec, options and canonicalize flag.
    # pylint: disable=protected-access
    return _SingleWorkerDatasetIteratorSpec(value._worker, value._devices,
                                            value._element_spec, value._options,
                                            value._canonicalize_devices)


class _SingleWorkerOwnedDatasetIterator(_SingleWorkerDatasetIteratorBase,
                                        composite_tensor.CompositeTensor):
  # [CLASS DOC]
  # HELPER: per-worker iterator used inside DistributedIterator. Owns its underlying iterator
  # (an owned multi-device iterator or a plain dataset iterator), so the resource is released
  # when the Python object goes away. A composite tensor, created either from a dataset or from
  # components plus an element spec (when rebuilt from its spec).
  """Iterator for a DistributedDataset instance."""

  def __init__(self,
               dataset=None,
               worker=None,
               devices=None,
               components=None,
               element_spec=None,
               options=None,
               canonicalize_devices=None):
    # [FUNCTION DOC]
    # Creates the iterator from a dataset or from components. Raises a value error if worker or
    # devices are missing, if no dataset is given and components or element spec is missing, or
    # if a dataset is given together with components or an element spec. With components, the
    # element spec, worker, devices and the first component (the iterator) are stored directly;
    # with a dataset the base constructor runs, which calls the make-iterator method. Also stores
    # options and the canonicalize flag.
    """Create iterator for the `dataset` to fetch data to worker's `devices` .

    `OwnedMultiDeviceIterator` is used to prefetch input to the devices on the
    given worker. The lifetime of this iterator is tied to the encompassing
    python object. Once we go out of scope of the python object or return from
    a tf.function the underlying iterator resource is deleted.

    Args:
      dataset: A `tf.data.Dataset` instance.
      worker: Worker on which ops should be created.
      devices: Distribute data from `dataset` to these devices.
      components: Tensor components to construct the
        _SingleWorkerOwnedDatasetIterator from.
      element_spec: A nested structure of `TypeSpec` objects that represents the
      type specification of elements of the iterator.
      options: `tf.distribute.InputOptions` used to control options on how this
      dataset is distributed.
      canonicalize_devices: Whether to canonicalize devices for workers fully or
      partially. If False, it will partially canonicalize devices by removing
      job and task.
    """
    if worker is None or devices is None:
      raise ValueError("Both `worker` and `devices` should be provided")

    error_message = ("Either `dataset` or both `components` and `element_spec` "
                     "need to be provided.")

    self._options = options
    self._canonicalize_devices = canonicalize_devices
    if dataset is None:
      if (components is None or element_spec is None):
        raise ValueError(error_message)
      self._element_spec = element_spec
      self._worker = worker
      self._devices = devices
      self._iterator = components[0]
    else:
      if (components is not None or element_spec is not None):
        raise ValueError(error_message)
      super(_SingleWorkerOwnedDatasetIterator,
            self).__init__(dataset, worker, devices, self._options)

  def _create_owned_multi_device_iterator(self):
    # [FUNCTION DOC]
    # HELPER: for _make_iterator. Creates an owned multi-device iterator over the dataset and
    # compute devices and stores it as the underlying iterator. The source device is the host of
    # the worker canonicalized against the current device scope, except inside a function where
    # the worker device itself is used (the function is traced on the coordinator but runs on the
    # worker). When options are set, the per-replica buffer size option sets the maximum buffer
    # and prefetch buffer sizes. Returns nothing.
    # If the worker devices are already canonicalized, canonicalizing again
    # would have no impact.
    # For strategies running on remote workers such as PS Strategy, the device
    # scope will be derived from current worker, if used under init_scope().
    if not ops.inside_function():
      device_scope = device_util.canonicalize(self._worker,
                                              device_util.current())
      host_device = device_util.get_host_for_device(device_scope)
    else:
      # In general, iterators should not be created within tf.functions. For
      # exact visitation guarantee solutions for parameter server training,
      # however, we do create iterators within the tf.functions that are
      # dispatched to workers. In these cases, the traced device must match the
      # runtime device. Since tracing occurs on the chief, we do not want to use
      # the current device scope, which would be the chief, but rather use the
      # relative worker device scope explicitly.
      device_scope, host_device = self._worker, self._worker
    with ops.device(device_scope):
      if self._options is not None:
        self._iterator = multi_device_iterator_ops.OwnedMultiDeviceIterator(
            self._dataset,
            self._devices,
            source_device=host_device,
            max_buffer_size=self._options
            .experimental_per_replica_buffer_size,
            prefetch_buffer_size=self._options
            .experimental_per_replica_buffer_size)
      else:
        self._iterator = multi_device_iterator_ops.OwnedMultiDeviceIterator(
            self._dataset, self._devices, source_device=host_device)

  def _make_iterator(self):
    # [FUNCTION DOC]
    # Creates the underlying iterator. Raises a value error if no worker device is set. Uses an
    # owned multi-device iterator when the options call for it, otherwise a plain iterator over
    # the dataset created on the worker device. Stores it as the iterator; returns nothing.
    """Make appropriate iterator on the dataset."""
    if not self._worker:
      raise ValueError("Worker device must be specified when creating an "
                       "owned iterator.")
    if _should_use_multi_device_iterator(self._options):
      self._create_owned_multi_device_iterator()
    else:
      with ops.device(self._worker):
        self._iterator = iter(self._dataset)

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the element spec stored for this iterator.
    return self._element_spec

  @property
  def _type_spec(self):
    # [FUNCTION DOC]
    # Returns the single-worker iterator spec built from this iterator's worker, devices, element
    # spec, options and canonicalize flag.
    return _SingleWorkerDatasetIteratorSpec(self._worker, self._devices,
                                            self._element_spec, self._options,
                                            self._canonicalize_devices)

  @property
  def output_classes(self):
    # [FUNCTION DOC]
    # Property returning, for each component of the element spec, its legacy output class
    # (nested structure of Python types).
    """Returns the class of each component of an element of this iterator.

    The expected values are `tf.Tensor` and `tf.SparseTensor`.

    Returns:
      A nested structure of Python `type` objects corresponding to each
      component of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_classes(),  # pylint: disable=protected-access
        self._element_spec)

  @property
  def output_shapes(self):
    # [FUNCTION DOC]
    # Property returning, for each component of the element spec, its legacy output shape
    # (nested structure of tensor shapes).
    """Returns the shape of each component of an element of this iterator.

    Returns:
      A nested structure of `tf.TensorShape` objects corresponding to each
      component of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_shapes(),  # pylint: disable=protected-access
        self._element_spec)

  @property
  def output_types(self):
    # [FUNCTION DOC]
    # Property returning, for each component of the element spec, its legacy output data type
    # (nested structure of dtypes).
    """Returns the type of each component of an element of this iterator.

    Returns:
      A nested structure of `tf.DType` objects corresponding to each component
      of an element of this dataset.
    """
    return nest.map_structure(
        lambda component_spec: component_spec._to_legacy_output_types(),  # pylint: disable=protected-access
        self._element_spec)


def _create_iterators_per_worker(worker_datasets,
                                 input_workers,
                                 options=None,
                                 canonicalize_devices=False):
  # [FUNCTION DOC]
  # HELPER: creates one owned single-worker iterator per input worker. Inputs: worker_datasets
  # (list of datasets, one per worker), input_workers (InputWorkers), options (optional input
  # options) and canonicalize_devices (default false). Asserts matching counts and types. Each
  # iterator is created with ops on its worker device, fed to that worker's compute devices.
  # Returns the list of iterators.
  """Create a multidevice iterator on each of the workers."""
  assert isinstance(input_workers, InputWorkers)
  assert len(worker_datasets) == len(input_workers.worker_devices)
  iterators = []
  for i, worker in enumerate(input_workers.worker_devices):
    with ops.device(worker):
      worker_devices = input_workers.compute_devices_for_worker(i)
      iterator = _SingleWorkerOwnedDatasetIterator(
          dataset=worker_datasets[i],
          worker=worker,
          devices=worker_devices,
          options=options,
          canonicalize_devices=canonicalize_devices)
      iterators.append(iterator)
  return iterators


def _create_datasets_from_function_with_input_context(input_contexts,
                                                      input_workers,
                                                      dataset_fn):
  # [FUNCTION DOC]
  # HELPER: for datasets-from-function. Calls the dataset function once per input context, each
  # call under the matching worker's device. Inputs: input_contexts (list), input_workers and
  # dataset_fn (callback taking an input context and returning a dataset). Returns a pair: the
  # list of datasets and the element spec of the last created dataset (assumed identical).
  """Create device datasets per worker given a dataset function."""
  datasets = []
  for i, ctx in enumerate(input_contexts):
    worker = input_workers.worker_devices[i]
    with ops.device(worker):
      dataset = dataset_fn(ctx)
      datasets.append(dataset)
  return datasets, dataset.element_spec


# TODO(sourabhbajaj): Remove this in lieu of distributed datasets
def _get_batched_dataset(d):
  # [FUNCTION DOC]
  # HELPER: walks back a dataset pipeline to find the batching dataset. Unwraps the version-1
  # adapter; returns d if it is a batch or map-and-batch dataset; for prefetch and options
  # datasets recurses into the input dataset. Raises a value error if batching is not the last
  # operation (optionally followed by prefetch).
  """Get the batched dataset from `d`."""
  # pylint: disable=protected-access
  if isinstance(d, dataset_ops.DatasetV1Adapter):
    d = d._dataset

  if isinstance(d, (dataset_ops.BatchDataset, batching._MapAndBatchDataset)):
    return d
  elif isinstance(d, (dataset_ops.PrefetchDataset,
                      dataset_ops._OptionsDataset)):
    return _get_batched_dataset(d._input_dataset)

  raise ValueError(
      "Unable to get batched dataset from the input dataset. `batch` "
      "`map_and_batch` need to be the last operations on the dataset. "
      "The batch operations can be followed by a prefetch.")


def _get_batched_dataset_attributes(d):
  # [FUNCTION DOC]
  # HELPER: reads the batch size and drop-remainder setting from a batch or map-and-batch
  # dataset (asserting that type). Tensor values are converted to constants when possible.
  # Returns a pair of batch size and drop remainder.
  """Get `batch_size`, `drop_remainder` of dataset."""
  # pylint: disable=protected-access
  assert isinstance(d,
                    (dataset_ops.BatchDataset, batching._MapAndBatchDataset))
  if isinstance(d, dataset_ops.BatchDataset):
    batch_size = d._batch_size
    drop_remainder = d._drop_remainder
  elif isinstance(d, batching._MapAndBatchDataset):
    batch_size = d._batch_size_t
    drop_remainder = d._drop_remainder_t
  # pylint: enable=protected-access

  if tensor_util.is_tf_type(batch_size):
    batch_size = tensor_util.constant_value(batch_size)

  if tensor_util.is_tf_type(drop_remainder):
    drop_remainder = tensor_util.constant_value(drop_remainder)

  return batch_size, drop_remainder


# TODO(sourabhbajaj): Remove this in lieu of distributed datasets
def _get_dataset_attributes(dataset):
  # [FUNCTION DOC]
  # HELPER: finds the batching dataset behind the given dataset and returns a triple of batch
  # size, drop remainder and prefetch buffer size. The prefetch buffer is read from the given
  # dataset itself (through the version-1 adapter if present) and is none if it is not a
  # prefetch dataset. Raises a value error if no batching dataset can be found.
  """Get the underlying attributes from the dataset object."""
  # pylint: disable=protected-access

  # First, get batch_size and drop_remainder from the dataset. We need
  # to walk back the dataset creation process and find the batched version in
  # order to get the attributes.
  batched_dataset = _get_batched_dataset(dataset)
  batch_size, drop_remainder = _get_batched_dataset_attributes(batched_dataset)

  # Second, prefetch buffer should be get from the original dataset.
  prefetch_buffer = None
  if isinstance(dataset, dataset_ops.PrefetchDataset):
    prefetch_buffer = dataset._buffer_size
  elif (isinstance(dataset, dataset_ops.DatasetV1Adapter)
        and isinstance(dataset._dataset, dataset_ops.PrefetchDataset)):
    prefetch_buffer = dataset._dataset._buffer_size

  return batch_size, drop_remainder, prefetch_buffer


def _should_use_multi_device_iterator(options):
  # [FUNCTION DOC]
  # HELPER: returns true when no options are given, or when the replication mode is per-worker,
  # or when it is per-replica with fetch-to-device enabled; false otherwise.
  """Determine whether to use multi_device_iterator_ops."""
  if (options is None or
      options.experimental_replication_mode == InputReplicationMode.PER_WORKER
      or
      (options.experimental_replication_mode == InputReplicationMode.PER_REPLICA
       and options.experimental_fetch_to_device)):
    return True
  return False


class MultiStepContext(object):
  # [CLASS DOC]
  # Context object that captures outputs while running several steps at a time. It holds a
  # mapping of last-step outputs, a mapping from output names to the reduce operation used (or
  # none), and a mapping of non-tensor outputs. Created by the run-steps-on-iterator machinery
  # and passed to the user's step function.
  """A context object that can be used to capture things when running steps.

  This context object is useful when running multiple steps at a time using the
  `experimental_run_steps_on_iterator` API. For e.g. it allows the user's step
  function to specify which outputs to emit at what frequency. Currently it
  supports capturing output from the last step, as well as capturing non tensor
  outputs.  In the future it will be augmented to support other use cases such
  as output each N steps.
  """

  def __init__(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it returns a context object, but a constructor returns nothing.
    # Initializes three empty mappings: last step outputs, their reduce operations, and
    # non-tensor outputs.
    """Initialize an output context.

    Returns:
      A context object.
    """
    self._last_step_outputs = {}
    self._last_step_outputs_reduce_ops = {}
    self._non_tensor_outputs = {}

  @property
  def last_step_outputs(self):
    # [FUNCTION DOC]
    # Property returning the mapping from output names to values captured for the last step
    # (reduced values when a reduce operation was given).
    """A dictionary consisting of outputs to be captured on last step.

    Keys in the dictionary are names of tensors to be captured, as specified
    when `set_last_step_output` is called.
    Values in the dictionary are the tensors themselves. If
    `set_last_step_output` was called with a `reduce_op` for this output,
    then the value is the reduced value.

    Returns:
      A dictionary with last step outputs.
    """
    return self._last_step_outputs

  def _set_last_step_outputs(self, outputs):
    # [FUNCTION DOC]
    # Replaces the whole last-step-outputs mapping with the given one. Raises a value error if
    # the argument is not a dictionary. Returns nothing.
    """Replace the entire dictionary of last step outputs."""
    if not isinstance(outputs, dict):
      raise ValueError("Need a dictionary to set last_step_outputs.")
    self._last_step_outputs = outputs

  def set_last_step_output(self, name, output, reduce_op=None):
    # [FUNCTION DOC]
    # Records an output to be returned from the last step. Inputs: name (string key), output
    # (tensor or per-replica structure) and reduce_op (optional reduction; required in a replica
    # context). In a cross-replica context it stores the reduce op and either the raw output (no
    # reduce op) or the output reduced by the current strategy over all replicas. In a replica
    # context it asserts a reduce op is given and uses a merge call that reduces the value and
    # records the reduce op once. Mutates the two mappings; returns nothing.
    """Set `output` with `name` to be outputted from the last step.

    Args:
      name: String, name to identify the output. Doesn't need to match tensor
        name.
      output: The tensors that should be outputted with `name`. See below for
        actual types supported.
      reduce_op: Reduction method to use to reduce outputs from multiple
        replicas. Required if `set_last_step_output` is called in a replica
        context. Optional in cross_replica_context.
        When present, the outputs from all the replicas are reduced using the
        current distribution strategy's `reduce` method. Hence, the type of
        `output` must be what's supported by the corresponding `reduce` method.
        For e.g. if using MirroredStrategy and reduction is set, output
        must be a `PerReplica` value.
        The reduce method is also recorded in a dictionary
        `_last_step_outputs_reduce_ops` for later interpreting of the
        outputs as already reduced or not.
    """
    if distribute_lib.in_cross_replica_context():
      self._last_step_outputs_reduce_ops[name] = reduce_op
      if reduce_op is None:
        self._last_step_outputs[name] = output
      else:
        distribution = distribute_lib.get_strategy()
        self._last_step_outputs[name] = distribution.reduce(reduce_op, output,
                                                            axis=None)
    else:
      assert reduce_op is not None
      def merge_fn(distribution, value):
        # [FUNCTION DOC]
        # HELPER: merge-call callback of set_last_step_output. Reduces the value over all replicas
        # with the strategy and stores it under the output name, and records the reduce op there
        # (once, since replicas share this context).
        self._last_step_outputs[name] = distribution.reduce(reduce_op, value,
                                                            axis=None)
        # Setting this inside the `merge_fn` because all replicas share the same
        # context object, so it's more robust to set it only once (even if all
        # the replicas are trying to set the same value).
        self._last_step_outputs_reduce_ops[name] = reduce_op

      distribute_lib.get_replica_context().merge_call(
          merge_fn, args=(output,))

  @property
  def non_tensor_outputs(self):
    # [FUNCTION DOC]
    # Property returning the mapping of captured non-tensor outputs.
    """A dictionary consisting of any non tensor outputs to be captured."""
    return self._non_tensor_outputs

  def set_non_tensor_output(self, name, output):
    # [FUNCTION DOC]
    # Records a non-tensor output under the given name. In a cross-replica context the output is
    # stored as is; in a replica context a merge call stores the list of per-replica local
    # results. Mutates the non-tensor outputs mapping; returns nothing.
    """Set `output` with `name` to be captured as a non tensor output."""
    if distribute_lib.in_cross_replica_context():
      self._non_tensor_outputs[name] = output
    else:
      def merge_fn(distribution, value):
        # [FUNCTION DOC]
        # HELPER: merge-call callback of set_non_tensor_output. Stores the list of local
        # per-replica results of the value under the output name (no reduction is applied).
        # NOTE(priyag): For non tensor outputs, we simply return all the values
        # in a list as reduction doesn't make sense on non tensors.
        self._non_tensor_outputs[name] = (
            distribution.experimental_local_results(value))
      distribute_lib.get_replica_context().merge_call(
          merge_fn, args=(output,))


def _create_distributed_tensor_spec(strategy, tensor_spec):
  # [FUNCTION DOC]
  # HELPER: converts element specs into the specs of the values a strategy yields. Inputs: the
  # strategy and a nested structure of specs. When the strategy does not always wrap values
  # (single device, not multi-worker), the structure is returned unchanged; otherwise each spec
  # is replaced by a per-replica spec repeating it once per worker device of the strategy.
  """Create a `tf.TypeSpec` for a given strategy and input `tensor_spec`.

  Args:
    strategy: The given `tf.distribute` strategy.
    tensor_spec: `tf.TensorSpec` of a given value. The batch dimension of the
      shape should be None if you have partial batches.

  Returns:
    A `tf.TypeSpec` that matches the values produced by a given strategy. This
    can be a `tf.TensorSpec` or a `PerRelicaSpec`.
  """
  num_replicas = len(strategy.extended.worker_devices)

  # For one device strategy that is not MultiWorkerMirroredStrategy,  return the
  # tensor_spec as is, since we don't wrap the output with PerReplica in this
  # case.
  # TODO(b/166464552): remove after we always wrap for all strategies.
  if not _always_wrap(strategy):
    return tensor_spec

  # For other cases we assume the input to tf.function is a per replica type.
  def _get_value_per_replica(tensor_spec_per_input):
    # [FUNCTION DOC]
    # HELPER: builds a per-replica spec that repeats the given spec once for every replica.
    value_specs = [tensor_spec_per_input for _ in range(num_replicas)]
    return values.PerReplicaSpec(*value_specs)

  return nest.map_structure(_get_value_per_replica, tensor_spec)


def _replace_per_replica_spec(spec, i):
  # [FUNCTION DOC]
  # HELPER: for splitting specs per worker. If spec is a per-replica spec, returns its i-th
  # value spec; otherwise returns the spec unchanged.
  """If `spec` is a `PerReplicaSpec`, then return its `i`th value_spec."""
  if isinstance(spec, values.PerReplicaSpec):
    return spec._value_specs[i]  # pylint: disable=protected-access
  else:
    return spec


def _cardinality(dataset):
  # [FUNCTION DOC]
  # HELPER: returns the dataset cardinality as a number when executing eagerly (computed on the
  # dataset's device); outside eager mode returns the unknown-cardinality constant.
  """Returns the cardinality of the dataset."""
  if context.executing_eagerly():
    with ops.device(dataset._variant_tensor.device):  # pylint: disable=protected-access
      return dataset.cardinality().numpy()
  return cardinality_lib.UNKNOWN


def _enable_get_next_as_optional(strategy, dataset, cardinality):
  # [FUNCTION DOC]
  # HELPER: decides whether to use partial-batch handling (get-next-as-optional). Returns false
  # if the strategy's partial-batch flag (new or older experimental name; default false) is off,
  # or if the dataset is infinite. Otherwise returns true when the element spec is not fully
  # statically shaped or the strategy runs in multi-worker mode.
  """Returns whether to enable using partial batch handling."""
  # TODO(b/133073708): we currently need a flag to control the usage because
  # there is a performance difference between get_next() and
  # get_next_as_optional(). And we only enable get_next_as_optional when the
  # output shapes are not static.
  #
  # TODO(rxsang): We want to always enable the get_next_as_optional behavior
  # when user passed input_fn instead of dataset.
  if not getattr(
      strategy.extended, "enable_partial_batch_handling",
      getattr(strategy.extended, "experimental_enable_get_next_as_optional",
              False)):
    return False

  # If the dataset is infinite, we don't need to enable last partial batch
  # support. Note that we can only evaluate the cardinality of the dataset in
  # eager.
  if cardinality == cardinality_lib.INFINITE:
    return False

  return not _is_statically_shaped(
      dataset.element_spec) or strategy.extended._in_multi_worker_mode()  # pylint: disable=protected-access


def _create_per_replica(value_list, strategy):
  # [FUNCTION DOC]
  # HELPER: groups a list of per-replica values (one per replica) into a per-replica structure
  # through the regroup utility, forcing wrapping when the strategy always wraps values.
  # Returns the structure.
  """Creates a PerReplica.

  For strategies other than OneDeviceStrategy, it creates a PerReplica whose
  type spec is set to the element spec of the dataset. This helps avoid
  retracing for partial batches. Retracing is problematic for multi client when
  different client retraces different time, since retracing changes the
  collective keys in the tf.function, and causes mismatches among clients.

  For single client strategies, this simply calls distribute_utils.regroup().

  Args:
    value_list: a list of values, one for each replica.
    strategy: the `tf.distribute.Strategy`.

  Returns:
    a structure of PerReplica.

  """
  # TODO(b/166464552): always wrap for all one device strategies as well.
  always_wrap = _always_wrap(strategy)
  per_replicas = distribute_utils.regroup(value_list, always_wrap=always_wrap)
  return per_replicas


def _always_wrap(strategy):
  # [FUNCTION DOC]
  # HELPER: returns true if the strategy is in multi-worker mode or has more than one worker
  # device, meaning values must always be wrapped as distributed values.
  """Returns whether to always wrap the values in a DistributedValues."""
  return strategy.extended._in_multi_worker_mode() or len(  # pylint: disable=protected-access
      strategy.extended.worker_devices) > 1


def _rebatch_as_dynamic(per_replica_spec):
  # [FUNCTION DOC]
  # HELPER: for element_spec properties. Returns a per-replica spec whose component specs have a
  # dynamic (unknown) batch dimension. Asserts the input is a per-replica spec.
  """Rebatch the spec to have a dynamic batch dimension."""
  assert isinstance(per_replica_spec, values.PerReplicaSpec), per_replica_spec

  # pylint: disable=protected-access
  def _rebatch(spec):
    # [FUNCTION DOC]
    # HELPER: unbatches the spec and batches it again with an unknown batch size; if the spec
    # cannot be unbatched (value error) it is returned unchanged.
    # Rebatch if possible.
    try:
      return spec._unbatch()._batch(None)
    except ValueError:
      pass
    return spec

  return values.PerReplicaSpec(
      *nest.map_structure(_rebatch, per_replica_spec._value_specs))
  # pylint: enable=protected-access


def _ag_enumerate_not_implemented(s, unused_start):
  # [FUNCTION DOC]
  # HELPER: autograph enumerate override registered for distributed iterators and datasets.
  # Always raises a not-implemented error saying enumerate is unsupported on this type inside a
  # tf.function and advising a for loop with a separate counter. The start argument is unused.
  msg = (
      f"enumerate not supported with {s.__class__.__name__} types within "
      "tf.functions. Use a for loop over the dataset and keep a separate "
      "counter instead."
  )
  raise NotImplementedError(msg)


py_builtins.enumerate_registry.register(
    DistributedIterator, _ag_enumerate_not_implemented
)
py_builtins.enumerate_registry.register(
    DistributedDataset, _ag_enumerate_not_implemented
)
