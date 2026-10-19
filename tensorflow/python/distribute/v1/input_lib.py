# [MODULE DOC]
# Graph-mode (v1) distributed input classes for tf.distribute. Subclasses the generic distributed
# dataset and iterator classes to add v1-style one-shot and initializable iterators, iterator
# initializer ops, legacy output class/shape/type properties, iterators created from an input
# function, and per-worker single-device-group iterators built on multi-device iterators or plain
# callables. Contains a private helper that creates the per-worker iterators. No import-time side
# effects.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.data.experimental.ops import cardinality as cardinality_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import multi_device_iterator_ops
from tensorflow.python.data.ops import optional_ops
from tensorflow.python.distribute import input_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.types import data as data_types
from tensorflow.python.util.deprecation import deprecated


class DistributedDatasetV1(input_lib.DistributedDataset):
  # [CLASS DOC]
  # Distributed dataset for v1 that prefetches to multiple devices. Extends the generic
  # distributed dataset with deprecated-style one-shot and initializable iterator factories and an
  # iteration method that works in eager mode or inside a function. Creating an iterator builds one
  # multi-device iterator per worker over the cloned datasets.
  """Distributed dataset that supports prefetching to multiple devices."""

  def __init__(self,
               dataset,
               input_workers,
               strategy,
               num_replicas_in_sync=None,
               input_context=None,
               options=None):
    # [FUNCTION DOC]
    # Constructor. Takes the source dataset, the input workers description, the strategy, and
    # optionally the replica count to split batches across, an input context for sharding, and
    # distribution options. Stores the input workers and forwards everything to the parent
    # constructor (which clones and shards the datasets).
    self._input_workers = input_workers
    super(DistributedDatasetV1, self).__init__(
        input_workers,
        strategy,
        dataset,
        num_replicas_in_sync=num_replicas_in_sync,
        input_context=input_context,
        options=options)

  def make_one_shot_iterator(self):
    # [FUNCTION DOC]
    # Public (documented as deprecated) wrapper that returns a single-use iterator by delegating to
    # the private one-shot factory; raises a value error in graph mode.
    """Get a one time use iterator for DistributedDatasetV1.

    Note: This API is deprecated. Please use `for ... in dataset:` to iterate
    over the dataset or `iter` to create an iterator.

    Returns:
      A DistributedIteratorV1 instance.
    """
    return self._make_one_shot_iterator()

  def _make_one_shot_iterator(self):
    # [FUNCTION DOC]
    # Returns a new iterator only when executing eagerly; in graph mode raises a value error
    # telling the caller to use the initializable iterator instead.
    """Get an iterator for DistributedDatasetV1."""
    # Graph mode with one shot iterator is disabled because we have to call
    # `initialize` on the iterator which is only required if we are using a
    # tf.distribute strategy.
    if not context.executing_eagerly():
      raise ValueError("Cannot create a one shot iterator. Please use "
                       "`make_initializable_iterator()` instead.")
    return self._get_iterator()

  def make_initializable_iterator(self):
    # [FUNCTION DOC]
    # Public (documented as deprecated) wrapper that returns an initializable iterator by
    # delegating to the private factory; raises a value error in eager mode.
    """Get an initializable iterator for DistributedDatasetV1.

    Note: This API is deprecated. Please use
    `tf.compat.v1.data.make_initializable_iterator(dataset)` to create an
    initializable iterator.

    Returns:
      A DistributedIteratorV1 instance.
    """
    return self._make_initializable_iterator()

  def _make_initializable_iterator(self, shared_name=None):  # pylint: disable=unused-argument
    # [FUNCTION DOC]
    # Returns a new iterator in graph mode (whose initializer must be run); raises a value error in
    # eager mode, where iterators are already initialized. The optional shared name is ignored.
    """Get an initializable iterator for DistributedDatasetV1."""
    # Eager mode generates already initialized iterators. Hence we cannot create
    # an initializable iterator.
    if context.executing_eagerly():
      raise ValueError("Cannot create initializable iterator in Eager mode. "
                       "Please use `iter()` instead.")
    return self._get_iterator()

  def _get_iterator(self):
    # [FUNCTION DOC]
    # HELPER: shared iterator construction for the iterator factories and iteration method.
    # Creates per-worker iterators over the cloned datasets, wraps them in a v1 distributed
    # iterator along with the cardinality of the first cloned dataset, the strategy and the
    # optional-get-next flag, copies this dataset's element spec onto it, and, in eager mode, waits
    # for async execution so all underlying iterators are initialized. Returns the iterator.
    worker_iterators = _create_iterators_per_worker(self._cloned_datasets,
                                                    self._input_workers,
                                                    self._options)
    cardinality = input_lib._cardinality(self._cloned_datasets[0])  # pylint: disable=protected-access
    iterator = DistributedIteratorV1(self._input_workers, worker_iterators,
                                     self._strategy, cardinality,
                                     self._enable_get_next_as_optional)
    iterator._element_spec = self.element_spec  # pylint: disable=protected-access

    # When async eager is enabled, sometimes the iterator may not finish
    # initialization before passing to a multi device function, add a sync point
    # here to make sure all underlying iterators are initialized.
    if context.executing_eagerly():
      context.async_wait()

    return iterator

  # pylint: disable=non-iterator-returned
  def __iter__(self):
    # [FUNCTION DOC]
    # Returns a new distributed iterator when eager execution is enabled outside functions or
    # when building a function; otherwise raises a runtime error (graph-mode iteration must use
    # an initializable iterator).
    if (ops.executing_eagerly_outside_functions() or
        ops.get_default_graph().building_function):
      return self._get_iterator()

    raise RuntimeError("__iter__() is only supported inside of tf.function "
                       "or when eager execution is enabled.")

  # pylint: enable=non-iterator-returned


class DistributedDatasetsFromFunctionV1(
    input_lib.DistributedDatasetsFromFunction):
  # [CLASS DOC]
  # Distributed inputs created from a dataset function, for v1. Adds the v1 initializable and
  # one-shot iterator factories and an iteration method on top of the generic parent, using the
  # per-worker datasets the parent already built.
  """Inputs created from dataset function."""

  def _make_initializable_iterator(self, shared_name=None):
    # [FUNCTION DOC]
    # Returns a new iterator in graph mode; raises a value error in eager mode. The optional
    # shared name is deleted and unused.
    """Get an initializable iterator for DistributedDatasetsFromFunctionV1."""
    del shared_name  # Unused
    # Eager mode generates already initialized iterators. Hence we cannot create
    # an initializable iterator.
    if context.executing_eagerly():
      raise ValueError("Cannot create initializable iterator in Eager mode. "
                       "Please use `iter()` instead.")
    return self._get_iterator()

  def _make_one_shot_iterator(self):
    # [FUNCTION DOC]
    # Returns a new iterator only when executing eagerly; in graph mode raises a value error.
    """Get an iterator for iterating over DistributedDatasetsFromFunctionV1."""
    # Graph mode with one shot iterator is disabled because we have to call
    # `initialize` on the iterator which is only required if we are using a
    # tf.distribute strategy.
    if not context.executing_eagerly():
      raise ValueError("Cannot create a one shot iterator. Please use "
                       "`make_initializable_iterator()` instead.")
    return self._get_iterator()

  def _get_iterator(self):
    # [FUNCTION DOC]
    # HELPER: shared iterator construction. Creates per-worker iterators over the stored datasets,
    # wraps them in a v1 distributed iterator with the first dataset's cardinality, the strategy
    # and the optional-get-next flag, sets the stored element spec on it and, in eager mode, waits
    # for async execution to finish initialization. Returns the iterator.
    iterators = _create_iterators_per_worker(self._datasets,
                                             self._input_workers, self._options)
    cardinality = input_lib._cardinality(self._datasets[0])  # pylint: disable=protected-access
    iterator = DistributedIteratorV1(self._input_workers, iterators,
                                     self._strategy, cardinality,
                                     self._enable_get_next_as_optional)
    iterator._element_spec = self._element_spec  # pylint: disable=protected-access

    # When async eager is enabled, sometimes the iterator may not finish
    # initialization before passing to a multi device function, add a sync point
    # here to make sure all underlying iterators are initialized.
    if context.executing_eagerly():
      context.async_wait()

    return iterator

  # pylint: disable=non-iterator-returned
  def __iter__(self):
    # [FUNCTION DOC]
    # Returns a new distributed iterator when eager execution is enabled outside functions or
    # when building a function; otherwise raises a runtime error.
    if (ops.executing_eagerly_outside_functions() or
        ops.get_default_graph().building_function):
      return self._get_iterator()

    raise RuntimeError("__iter__() is only supported inside of tf.function "
                       "or when eager execution is enabled.")

  # pylint: enable=non-iterator-returned


class DistributedIteratorV1(input_lib.DistributedIteratorBase):
  # [CLASS DOC]
  # Input iterator for a distributed dataset in v1. Holds a list of per-worker iterators (from the
  # base class) and adds an initializer (group of all per-worker initializer ops), a deprecated
  # initialize method, legacy output class/shape/type properties taken from the first iterator, a
  # lookup of the iterator for a given worker device, and the element spec.
  """Input Iterator for a distributed dataset."""

  # We need a private initializer method for re-initializing multidevice
  # iterators when used with Keras training loops. If we don't reinitialize the
  # iterator we run into memory leak issues (b/123315763).
  @property
  def _initializer(self):
    # [FUNCTION DOC]
    # Property. Collects the initializer ops of every per-worker iterator and returns a single
    # grouped op that runs them all (used to re-initialize iterators, e.g. for Keras loops).
    init_ops = []
    for it in self._iterators:
      init_ops.extend(it.initialize())
    return control_flow_ops.group(init_ops)

  @deprecated(None, "Use the iterator's `initializer` property instead.")
  def initialize(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it returns a list of initializer ops, but it returns the
    # single grouped op from the private initializer property. Also the deprecation notice points
    # to the public initializer property, which itself calls this deprecated method.
    # Returns the grouped initializer op for all per-worker iterators.
    """Initialize underlying iterators.

    Returns:
      A list of any initializer ops that should be run.
    """
    return self._initializer

  @property
  def initializer(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it returns a list of ops, but it returns whatever the
    # deprecated initialize method returns (a single grouped op), and calling it triggers the
    # deprecation warning pointing back to this property.
    # Property returning the iterator's initializer op.
    """Returns a list of ops that initialize the iterator."""
    return self.initialize()

  # TODO(priyag): Remove when we switch to using `MultiDeviceIterator` for TPUs.
  @property
  def output_classes(self):
    # [FUNCTION DOC]
    # Property: legacy output classes, taken from the first per-worker iterator.
    return self._iterators[0].output_classes

  # TODO(priyag): Remove when we switch to using `MultiDeviceIterator` for TPUs.
  @property
  def output_shapes(self):
    # [FUNCTION DOC]
    # Property: legacy output shapes, taken from the first per-worker iterator.
    return self._iterators[0].output_shapes

  # TODO(priyag): Remove when we switch to using `MultiDeviceIterator` for TPUs.
  @property
  def output_types(self):
    # [FUNCTION DOC]
    # Property: legacy output types, taken from the first per-worker iterator.
    return self._iterators[0].output_types

  # TODO(priyag): Remove when we switch to using `MultiDeviceIterator` for TPUs.
  def get_iterator(self, worker):
    # [FUNCTION DOC]
    # Returns the per-worker iterator whose worker device equals the given device name, or nothing
    # if no worker matches.
    for i, w in enumerate(self._input_workers.worker_devices):
      if worker == w:
        return self._iterators[i]
    return None

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the stored element type specification (set when the iterator is created).
    """The type specification of an element of this iterator."""
    return self._element_spec


class DatasetIterator(DistributedIteratorV1):
  # [CLASS DOC]
  # Iterator created directly from an input dataset: wraps the dataset in a v1 distributed
  # dataset, creates per-worker iterators over its cloned datasets and initializes the parent
  # iterator with them, copying the element spec.
  """Iterator created from input dataset."""

  def __init__(self,
               dataset,
               input_workers,
               strategy,
               num_replicas_in_sync=None,
               input_context=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: dataset, input workers, strategy, optional replica count and optional
    # input context. Builds a v1 distributed dataset, creates per-worker iterators from its
    # cloned datasets (without distribution options), then calls the parent constructor with the
    # dataset's cardinality and optional-get-next flag and stores its element spec.
    """Make an iterator for the dataset on given devices.

    If `num_replicas_in_sync` is not None, we split each batch of the dataset
    into `num_replicas_in_sync` smaller batches, to be distributed among that
    worker's replicas, so that the batch size for a global step (across all
    workers and replicas) is as expected.

    Args:
      dataset: `tf.data.Dataset` that will be used as the input source.
      input_workers: an `InputWorkers` object.
      strategy: a `tf.distribute.Strategy` object, used to run all-reduce to
        handle last partial batch.
      num_replicas_in_sync: Optional integer. If this is not None, the value is
        used to decide how to rebatch datasets into smaller batches so that the
        total batch size for each step (across all workers and replicas) adds up
        to `dataset`'s batch size.
      input_context: `InputContext` for sharding. Only pass this in for between
        graph multi-worker cases where there is only one `input_worker`. In
        these cases, we will shard based on the `input_pipeline_id` and
        `num_input_pipelines` in the `InputContext`.
    """
    dist_dataset = DistributedDatasetV1(
        dataset,
        input_workers,
        strategy,
        num_replicas_in_sync=num_replicas_in_sync,
        input_context=input_context)
    # pylint: disable=protected-access
    worker_iterators = _create_iterators_per_worker(
        dist_dataset._cloned_datasets, input_workers)
    super(DatasetIterator,
          self).__init__(input_workers, worker_iterators, strategy,
                         dist_dataset.cardinality,
                         dist_dataset._enable_get_next_as_optional)
    self._element_spec = dist_dataset.element_spec
    # pylint: enable=protected-access


class InputFunctionIterator(DistributedIteratorV1):
  # [CLASS DOC]
  # Iterator created from a user input function, called once per worker with that worker's input
  # context. Each result (a dataset or a callable) becomes a single-worker iterator. Optional
  # get-next is disabled and cardinality is unknown.
  """Iterator created from input function."""

  def __init__(self, input_fn, input_workers, input_contexts, strategy):
    # [FUNCTION DOC]
    # Constructor. Inputs: input function, input workers, a list of input contexts (one per
    # worker, same order) and the strategy. Asserts the workers are the expected type and raises
    # a value error if the number of workers differs from the number of contexts. For each worker,
    # under its device, calls the input function with its context; a dataset result gets a
    # dataset iterator, a callable result a callable iterator, anything else raises a value error.
    # Then calls the parent constructor with unknown cardinality and no optional get-next.
    """Make an iterator for input provided via an input function.

    Currently implements PER_WORKER mode, in which the `input_fn` is called
    once on each worker.

    TODO(priyag): Add other replication modes.

    Args:
      input_fn: Input function that returns a `tf.data.Dataset` object.
      input_workers: an `InputWorkers` object.
      input_contexts: A list of `InputContext` instances to be passed to call(s)
        to `input_fn`. Length and order should match worker order in
        `worker_device_pairs`.
      strategy: a `tf.distribute.Strategy` object, used to run all-reduce to
        handle last partial batch.
    """
    assert isinstance(input_workers, input_lib.InputWorkers)
    if input_workers.num_workers != len(input_contexts):
      raise ValueError("Number of input workers (%d) is not same as number of "
                       "input_contexts (%d)" %
                       (input_workers.num_workers, len(input_contexts)))

    iterators = []
    for i, ctx in enumerate(input_contexts):
      worker = input_workers.worker_devices[i]
      with ops.device(worker):
        result = input_fn(ctx)
        devices = input_workers.compute_devices_for_worker(i)
        if isinstance(result, data_types.DatasetV2):
          iterator = _SingleWorkerDatasetIterator(result, worker, devices)
        elif callable(result):
          iterator = _SingleWorkerCallableIterator(result, worker, devices)
        else:
          raise ValueError(
              "input_fn must return a tf.data.Dataset or a callable.")
        iterators.append(iterator)

    super(InputFunctionIterator, self).__init__(
        input_workers,
        iterators,
        strategy,
        cardinality=cardinality_lib.UNKNOWN,
        enable_get_next_as_optional=False)
    self._enable_get_next_as_optional = False


class _SingleWorkerDatasetIterator(input_lib._SingleWorkerDatasetIteratorBase):  # pylint: disable=protected-access
  # [CLASS DOC]
  # HELPER: per-worker iterator used by the distributed iterators. Builds a multi-device iterator
  # over the worker's dataset and compute devices (optionally with per-replica buffer sizes from
  # the options), and exposes initialization and legacy output class/shape/type properties.
  """Iterator for a single DistributedDatasetV1 instance."""

  def _make_iterator(self):
    # [FUNCTION DOC]
    # Under the worker's device, creates the multi-device iterator over the dataset and devices
    # and stores it. If options are given, uses the per-replica buffer size option for both the
    # maximum and prefetch buffer sizes; otherwise uses defaults.
    """Make appropriate iterator on the dataset."""
    with ops.device(self._worker):
      if self._options is not None:
        self._iterator = multi_device_iterator_ops.MultiDeviceIterator(
            self._dataset,
            self._devices,
            max_buffer_size=self._options.experimental_per_replica_buffer_size,
            prefetch_buffer_size=self._options
            .experimental_per_replica_buffer_size)
      else:
        self._iterator = multi_device_iterator_ops.MultiDeviceIterator(
            self._dataset,
            self._devices,
        )

  def initialize(self):
    # [FUNCTION DOC]
    # In eager mode (outside functions) resets the underlying iterator and returns an empty list;
    # otherwise returns a list holding the iterator's initializer op.
    """Initialize underlying iterator.

    In eager execution, this simply recreates the underlying iterator.
    In graph execution, it returns the initializer ops for the underlying
    iterator.

    Returns:
      A list of any initializer ops that should be run.
    """
    if ops.executing_eagerly_outside_functions():
      self._iterator._eager_reset()  # pylint: disable=protected-access
      return []
    else:
      return [self._iterator.initializer]

  @property
  def output_classes(self):
    # [FUNCTION DOC]
    # Property: legacy output classes of the underlying iterator.
    return dataset_ops.get_legacy_output_classes(self._iterator)

  @property
  def output_shapes(self):
    # [FUNCTION DOC]
    # Property: legacy output shapes of the underlying iterator.
    return dataset_ops.get_legacy_output_shapes(self._iterator)

  @property
  def output_types(self):
    # [FUNCTION DOC]
    # Property: legacy output types of the underlying iterator.
    return dataset_ops.get_legacy_output_types(self._iterator)


class _SingleWorkerCallableIterator(object):
  # [CLASS DOC]
  # HELPER: per-worker iterator backed by a callable that returns a tensor (or structure) each
  # time it is called. Holds the callable, the worker device and the list of replica devices.
  # Offers the same get-next style methods as dataset iterators; every call just invokes the
  # callable under the worker device.
  """Iterator for a single tensor-returning callable."""

  def __init__(self, fn, worker, devices):
    # [FUNCTION DOC]
    # Stores the callable, the worker device name and the list of compute device names.
    self._fn = fn
    self._worker = worker
    self._devices = devices

  def get_next(self, device, name=None):
    # [FUNCTION DOC]
    # Calls the callable once under the worker device and returns its result. The device and
    # optional name arguments are ignored.
    """Get next element for the given device from the callable."""
    del device, name
    with ops.device(self._worker):
      return self._fn()

  def get_next_as_list(self, name=None):
    # [FUNCTION DOC]
    # Calls the callable once per replica device under the worker device and returns the list of
    # results. The optional name is ignored.
    """Get next element from the callable."""
    del name
    with ops.device(self._worker):
      data_list = [self._fn() for _ in self._devices]
      return data_list

  def get_next_as_optional_list(self):
    # [FUNCTION DOC]
    # Calls the callable once per replica device under the worker device, wraps each result in an
    # always-present optional value, and returns the list.
    with ops.device(self._worker):
      data_list = [
          optional_ops.Optional.from_value(self._fn()) for _ in self._devices
      ]
      return data_list

  def initialize(self):
    # [FUNCTION DOC]
    # Nothing to initialize for a callable; returns an empty list (a code comment questions
    # whether it should raise instead).
    # TODO(petebu) Should this throw an exception instead?
    return []


def _create_iterators_per_worker(worker_datasets, input_workers, options=None):
  # [FUNCTION DOC]
  # HELPER: builds the per-worker iterators for the distributed datasets and iterators.
  # Inputs: a list of datasets (one per worker), the input workers description (asserted to be the
  # expected type and to match the dataset count) and optional distribution options. For each
  # worker, under its device, creates a single-worker multi-device dataset iterator over the
  # worker's compute devices. Returns the list of iterators.
  """Create a multidevice iterator on each of the workers."""
  assert isinstance(input_workers, input_lib.InputWorkers)
  assert len(worker_datasets) == len(input_workers.worker_devices)
  iterators = []
  for i, worker in enumerate(input_workers.worker_devices):
    with ops.device(worker):
      worker_devices = input_workers.compute_devices_for_worker(i)
      iterator = _SingleWorkerDatasetIterator(
          worker_datasets[i],  # pylint: disable=protected-access
          worker,
          worker_devices,
          options)
      iterators.append(iterator)
  return iterators
