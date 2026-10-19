# [MODULE DOC]
# Defines the base classes for Keras preprocessing layers: an abstract PreprocessingLayer whose
# state is computed ahead of training through an adapt step, a CombinerPreprocessingLayer that
# delegates the state computation to a Combiner object, and the abstract Combiner interface that
# describes a shardable computation using accumulators. Also exposes a helper that converts
# tensor-like values to plain lists and a private guard against calling adapt inside a
# traced function. No registrations or global state are created at import time; it only imports
# backend, data adapter and base layer modules from the same package.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Contains the base ProcessingLayer and a subclass that uses Combiners."""

import abc
import collections

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import data_adapter
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.keras.utils import version_utils
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.trackable import base as trackable


class PreprocessingLayer(Layer, metaclass=abc.ABCMeta):
  """Base class for Preprocessing Layers.

  **Don't use this class directly: it's an abstract base class!** You may
  be looking for one of the many built-in
  [preprocessing layers](https://keras.io/guides/preprocessing_layers/)
  instead.

  Preprocessing layers are layers whose state gets computed before model
  training starts. They do not get updated during training.
  Most preprocessing layers implement an `adapt()` method for state computation.

  The `PreprocessingLayer` class is the base class you would subclass to
  implement your own preprocessing layers.

  Attributes:
    streaming: Whether a layer can be adapted multiple times without resetting
      the state of the layer.
  """
  # [CLASS DOC]
  # Abstract base class for Keras layers whose state (for example statistics or vocabularies) is
  # computed from data before training via adapt, and is not updated by gradient training.
  # Holds flags for streaming, compiled and adapted status, a cached adapt function and a
  # steps-per-execution variable. Subclasses must implement the state update, reset and
  # (optionally) merge and finalize hooks. Uses an abstract-base-class metaclass and requires
  # restoring from configuration when deserialized.
  _must_restore_from_config = True

  def __init__(self, streaming=True, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Takes streaming (boolean, default true: whether adapt may be called repeatedly
    # without resetting state) and keyword arguments forwarded to the base layer. Marks the layer
    # as not compiled and not adapted, wraps the (subclass) reset_state method on the instance so
    # that resetting also clears the adapted flag, and initializes the cached adapt function to
    # nothing. Returns nothing.
    super(PreprocessingLayer, self).__init__(**kwargs)
    self._streaming = streaming
    self._is_compiled = False
    self._is_adapted = False

    # Sets `is_adapted=False` when `reset_state` is called.
    self._reset_state_impl = self.reset_state
    self.reset_state = self._reset_state_wrapper

    self._adapt_function = None

  @property
  def streaming(self):
    # [FUNCTION DOC]
    # Read-only property returning the boolean streaming flag given at construction.
    """Whether `adapt` can be called twice without resetting the state."""
    return self._streaming

  @property
  def is_adapted(self):
    # [FUNCTION DOC]
    # Read-only property returning true once adapt has completed and no reset has happened since.
    """Whether the layer has been fit to data already."""
    return self._is_adapted

  def update_state(self, data):
    # [FUNCTION DOC]
    # Abstract hook: subclasses fold one mini-batch of inputs (data) into the layer's statistics.
    # This base version always raises a not-implemented error. Returns nothing.
    """Accumulates statistics for the preprocessing layer.

    Arguments:
      data: A mini-batch of inputs to the layer.
    """
    raise NotImplementedError

  def reset_state(self):  # pylint: disable=method-hidden
    # [FUNCTION DOC]
    # Abstract hook: subclasses clear the accumulated statistics. This base version always raises
    # a not-implemented error. Note that the constructor replaces this attribute on each instance
    # with a wrapper that also clears the adapted flag.
    """Resets the statistics of the preprocessing layer."""
    raise NotImplementedError

  def merge_state(self, layers):
    # [FUNCTION DOC]
    # Abstract hook: subclasses merge the statistics of the given list of other preprocessing
    # layers into this layer. This base version always raises a not-implemented error.
    """Merge the statistics of multiple preprocessing layers.

    This layer will contain the merged state.

    Arguments:
      layers: Layers whose statistics should be merge with the statistics of
        this layer.
    """
    raise NotImplementedError

  def finalize_state(self):
    # [FUNCTION DOC]
    # Optional hook called at the end of adapt (and after restoring state). The base version does
    # nothing and returns nothing; subclasses override it for one-time post-processing of state.
    """Finalize the statistics for the preprocessing layer.

    This method is called at the end of `adapt` or after restoring a serialized
    preprocessing layer's state. This method handles any one-time operations
    that should occur on the layer's state before `Layer.__call__`.
    """
    pass

  def make_adapt_function(self):
    # [FUNCTION DOC]
    # Builds (or returns the cached) function that runs one adapt step, or several when the
    # steps-per-execution value is greater than one. A step takes the next batch from a dataset
    # iterator, builds the layer from that batch if needed, and calls update_state. Unless the
    # layer runs eagerly, the function is wrapped in a traced graph function. The result is stored
    # in the cache attribute and returned. Requires that compile has set the steps-per-execution
    # variable and eager flag, otherwise an attribute error occurs.
    """Creates a function to execute one step of `adapt`.

    This method can be overridden to support custom adapt logic.
    This method is called by `PreprocessingLayer.adapt`.

    Typically, this method directly controls `tf.function` settings,
    and delegates the actual state update logic to
    `PreprocessingLayer.update_state`.

    This function is cached the first time `PreprocessingLayer.adapt`
    is called. The cache is cleared whenever `PreprocessingLayer.compile`
    is called.

    Returns:
      Function. The function created by this method should accept a
      `tf.data.Iterator`, retrieve a batch, and update the state of the
      layer.
    """
    if self._adapt_function is not None:
      return self._adapt_function

    def adapt_step(iterator):
      # [FUNCTION DOC]
      # HELPER: single-step body used by make_adapt_function.
      # Pulls the next batch from the given iterator, builds the layer from its shape if not yet
      # built, and updates the layer state with it. Returns nothing; raises the iterator's
      # end-of-data error when it is exhausted.
      data = next(iterator)
      self._adapt_maybe_build(data)
      self.update_state(data)

    if self._steps_per_execution.numpy().item() == 1:
      adapt_fn = adapt_step
    else:

      def adapt_fn(iterator):
        # [FUNCTION DOC]
        # HELPER: multi-step body used by make_adapt_function.
        # Runs the single adapt step repeatedly, once per steps-per-execution count, on the same
        # iterator. Returns nothing.
        for _ in math_ops.range(self._steps_per_execution):
          adapt_step(iterator)

    if not self._run_eagerly:
      adapt_fn = def_function.function(adapt_fn)

    self._adapt_function = adapt_fn
    return self._adapt_function

  def compile(self, run_eagerly=None, steps_per_execution=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says run_eagerly defaults to false, but when it is None the code
    # uses the layer's dynamic flag instead; the docstring also indents steps_per_execution as if
    # it belonged to run_eagerly.
    # Configures adapt. steps_per_execution (integer, optional, default one) is stored in a
    # non-tracked integer variable; run_eagerly (boolean, optional) decides whether adapt steps
    # skip graph tracing. Marks the layer as compiled. Side effects: creates a variable and sets
    # the eager flag. Returns nothing. Does not clear the cached adapt function despite what
    # make_adapt_function's docstring implies.
    """Configures the layer for `adapt`.

    Arguments:
      run_eagerly: Bool. Defaults to `False`. If `True`, this `Model`'s logic
        will not be wrapped in a `tf.function`. Recommended to leave this as
        `None` unless your `Model` cannot be run inside a `tf.function`.
        steps_per_execution: Int. Defaults to 1. The number of batches to run
          during each `tf.function` call. Running multiple batches inside a
          single `tf.function` call can greatly improve performance on TPUs or
          small models with a large Python overhead.
    """
    if steps_per_execution is None:
      steps_per_execution = 1
    self._configure_steps_per_execution(steps_per_execution)

    if run_eagerly is None:
      run_eagerly = self.dynamic
    self._run_eagerly = run_eagerly

    self._is_compiled = True

  def adapt(self, data, batch_size=None, steps=None, reset_state=True):
    # [FUNCTION DOC]
    # Fits the layer state to data. Inputs: data (dataset, generator, sequence or array),
    # batch_size (optional integer), steps (optional integer number of batches), reset_state
    # (boolean, default true: clear existing state first when already built). Raises a runtime
    # error if called inside a traced function or when not running in TF2 behavior, and a
    # value error if the layer is non-streaming, already adapted, and reset_state is false.
    # Compiles with defaults if needed, iterates once over the data with a data handler running
    # the cached adapt function, waits for async work when required, then calls finalize_state
    # and marks the layer adapted. Returns nothing.
    """Fits the state of the preprocessing layer to the data being passed.

    After calling `adapt` on a layer, a preprocessing layer's state will not
    update during training. In order to make preprocessing layers efficient in
    any distribution context, they are kept constant with respect to any
    compiled `tf.Graph`s that call the layer. This does not affect the layer use
    when adapting each layer only once, but if you adapt a layer multiple times
    you will need to take care to re-compile any compiled functions as follows:

     * If you are adding a preprocessing layer to a `keras.Model`, you need to
       call `model.compile` after each subsequent call to `adapt`.
     * If you are calling a preprocessing layer inside `tf.data.Dataset.map`,
       you should call `map` again on the input `tf.data.Dataset` after each
       `adapt`.
     * If you are using a `tf.function` directly which calls a preprocessing
       layer, you need to call `tf.function` again on your callable after
       each subsequent call to `adapt`.

    `tf.keras.Model` example with multiple adapts:

    >>> layer = tf.keras.layers.experimental.preprocessing.Normalization(
    ...     axis=None)
    >>> layer.adapt([0, 2])
    >>> model = tf.keras.Sequential(layer)
    >>> model.predict([0, 1, 2])
    array([-1.,  0.,  1.], dtype=float32)
    >>> layer.adapt([-1, 1])
    >>> model.compile() # This is needed to re-compile model.predict!
    >>> model.predict([0, 1, 2])
    array([0., 1., 2.], dtype=float32)

    `tf.data.Dataset` example with multiple adapts:

    >>> layer = tf.keras.layers.experimental.preprocessing.Normalization(
    ...     axis=None)
    >>> layer.adapt([0, 2])
    >>> input_ds = tf.data.Dataset.range(3)
    >>> normalized_ds = input_ds.map(layer)
    >>> list(normalized_ds.as_numpy_iterator())
    [array([-1.], dtype=float32),
     array([0.], dtype=float32),
     array([1.], dtype=float32)]
    >>> layer.adapt([-1, 1])
    >>> normalized_ds = input_ds.map(layer) # Re-map over the input dataset.
    >>> list(normalized_ds.as_numpy_iterator())
    [array([0.], dtype=float32),
     array([1.], dtype=float32),
     array([2.], dtype=float32)]

    Arguments:
        data: The data to train on. It can be passed either as a tf.data
          Dataset, or as a numpy array.
        batch_size: Integer or `None`.
            Number of samples per state update.
            If unspecified, `batch_size` will default to 32.
            Do not specify the `batch_size` if your data is in the
            form of datasets, generators, or `keras.utils.Sequence` instances
            (since they generate batches).
        steps: Integer or `None`.
            Total number of steps (batches of samples)
            When training with input tensors such as
            TensorFlow data tensors, the default `None` is equal to
            the number of samples in your dataset divided by
            the batch size, or 1 if that cannot be determined. If x is a
            `tf.data` dataset, and 'steps' is None, the epoch will run until
            the input dataset is exhausted. When passing an infinitely
            repeating dataset, you must specify the `steps` argument. This
            argument is not supported with array inputs.
        reset_state: Optional argument specifying whether to clear the state of
          the layer at the start of the call to `adapt`, or whether to start
          from the existing state. This argument may not be relevant to all
          preprocessing layers: a subclass of PreprocessingLayer may choose to
          throw if 'reset_state' is set to False.
    """
    _disallow_inside_tf_function('adapt')
    if not version_utils.should_use_v2():
      raise RuntimeError('`adapt` is only supported in tensorflow v2.')  # pylint: disable=g-doc-exception
    if not self.streaming and self._is_adapted and not reset_state:
      raise ValueError('{} does not supporting calling `adapt` twice without '
                       'resetting the state.'.format(self.__class__.__name__))
    if not self._is_compiled:
      self.compile()  # Compile with defaults.
    if self.built and reset_state:
      self.reset_state()
    data_handler = data_adapter.DataHandler(
        data,
        batch_size=batch_size,
        steps_per_epoch=steps,
        epochs=1,
        steps_per_execution=self._steps_per_execution,
        distribute=False)
    self._adapt_function = self.make_adapt_function()
    for _, iterator in data_handler.enumerate_epochs():
      with data_handler.catch_stop_iteration():
        for _ in data_handler.steps():
          self._adapt_function(iterator)
          if data_handler.should_sync:
            context.async_wait()
    self.finalize_state()
    self._is_adapted = True

  def _reset_state_wrapper(self):
    # [FUNCTION DOC]
    # HELPER: instance-level replacement for reset_state installed by the constructor.
    # Calls the original reset_state saved at construction, then marks the layer as not adapted.
    # Returns nothing.
    """Calls `reset_state` and sets `adapted` to `False`."""
    self._reset_state_impl()
    self._is_adapted = False

  @trackable.no_automatic_dependency_tracking
  def _configure_steps_per_execution(self, steps_per_execution):
    # [FUNCTION DOC]
    # HELPER: used by compile.
    # Creates a 64-bit integer variable holding the number of steps per execution (aggregated by
    # taking the first replica) and stores it on the layer without automatic dependency tracking.
    # Returns nothing.
    self._steps_per_execution = variables.Variable(
        steps_per_execution,
        dtype='int64',
        aggregation=variables.VariableAggregationV2.ONLY_FIRST_REPLICA)

  # TODO(omalleyt): Unify this logic with `Layer._maybe_build`.
  def _adapt_maybe_build(self, data):
    # [FUNCTION DOC]
    # HELPER: used by the adapt step to build the layer lazily.
    # If the layer is not yet built, infers the shape of the given batch (unknown if the batch has
    # no shape attribute), records an all-unknown-dimension batch input shape if none was set,
    # builds the layer with the shape and marks it built. Does nothing if already built. Returns
    # nothing.
    if not self.built:
      try:
        # If this is a Numpy array or tensor, we can get shape from .shape.
        # If not, an attribute error will be thrown.
        data_shape = data.shape
        data_shape_nones = tuple([None] * len(data.shape))
      except AttributeError:
        # The input has an unknown number of dimensions.
        data_shape = None
        data_shape_nones = None

      # TODO (b/159261555): move this to base layer build.
      batch_input_shape = getattr(self, '_batch_input_shape', None)
      if batch_input_shape is None:
        # Set the number of dimensions.
        self._batch_input_shape = data_shape_nones
      self.build(data_shape)
      self.built = True


# TODO(omalleyt): This class will be gradually replaced.
class CombinerPreprocessingLayer(PreprocessingLayer):
  # [CLASS DOC]
  # Preprocessing layer whose adapt state is computed by a Combiner object. Keeps an ordered
  # mapping of named state variables (non-trainable weights), the combiner, and the in-progress
  # accumulator used during adapt. Subclasses define how accumulators map to state variables.
  """Base class for PreprocessingLayers that do computation using a Combiner.

  This class provides several helper methods to make creating a
  PreprocessingLayer easier. It assumes that the core of your computation will
  be done via a Combiner object. Subclassing this class to create a
  PreprocessingLayer allows your layer to be compatible with distributed
  computation.

  This class is compatible with Tensorflow 2.0+.
  """

  def __init__(self, combiner, **kwargs):
    # [FUNCTION DOC]
    # Constructor. combiner is the Combiner object performing the computation; other keyword
    # arguments go to the parent constructor. Creates an empty ordered mapping of state variables
    # and clears the adapt accumulator. Returns nothing.
    super(CombinerPreprocessingLayer, self).__init__(**kwargs)
    self.state_variables = collections.OrderedDict()
    self._combiner = combiner
    self._adapt_accumulator = None

  def reset_state(self):  # pylint: disable=method-hidden
    # [FUNCTION DOC]
    # Discards the in-progress accumulator. Does not modify the stored state variables. Returns
    # nothing.
    self._adapt_accumulator = None

  @trackable.no_automatic_dependency_tracking
  def update_state(self, data):
    # [FUNCTION DOC]
    # Folds one batch (data) into the accumulator using the combiner's compute step. If there is
    # no accumulator yet, starts from the one restored from the state variables (or nothing when
    # not adapted). Stores the new accumulator on the layer. Returns nothing.
    if self._adapt_accumulator is None:
      self._adapt_accumulator = self._get_accumulator()
    self._adapt_accumulator = self._combiner.compute(data,
                                                     self._adapt_accumulator)

  def merge_state(self, layers):
    # [FUNCTION DOC]
    # Merges this layer's state with that of the given list of other combiner layers: collects
    # each layer's accumulator, merges them with the combiner, and writes the merged result into
    # this layer's state variables. Returns nothing. Layers that are not adapted contribute no
    # accumulator (nothing), which the combiner must tolerate.
    accumulators = ([self._get_accumulator()] +
                    [l._get_accumulator() for l in layers])  # pylint: disable=protected-access
    merged_accumulator = self._combiner.merge(accumulators)
    self._set_accumulator(merged_accumulator)

  def finalize_state(self):
    # [FUNCTION DOC]
    # If an accumulator was built during adapt, extracts it into the state variables and clears
    # it; otherwise does nothing. Returns nothing.
    if self._adapt_accumulator is not None:
      self._set_accumulator(self._adapt_accumulator)

  def compile(self, run_eagerly=None, steps_per_execution=None):
    # [FUNCTION DOC]
    # Same as the parent compile except that run_eagerly defaults to true (not the dynamic flag)
    # when not given, since combiner computations run in plain Python. steps_per_execution is
    # passed through unchanged. Returns nothing.
    # TODO(omalleyt): Remove this once sublayers are switched to new APIs.
    if run_eagerly is None:
      run_eagerly = True
    super(CombinerPreprocessingLayer, self).compile(
        run_eagerly=run_eagerly, steps_per_execution=steps_per_execution)

  def adapt(self, data, batch_size=None, steps=None, reset_state=True):
    # [FUNCTION DOC]
    # Same inputs as the parent adapt. When reset_state is false, first seeds the accumulator from
    # the current state variable values so new data adds to existing state; then delegates to the
    # parent adapt. Returns nothing.
    if not reset_state:
      self._adapt_accumulator = self._combiner.restore(self._restore_updates())
    super(CombinerPreprocessingLayer, self).adapt(
        data, batch_size=batch_size, steps=steps, reset_state=reset_state)

  def _add_state_variable(self,
                          name,
                          shape,
                          dtype,
                          initializer=None,
                          partitioner=None,
                          use_resource=None,
                          **kwargs):
    # [FUNCTION DOC]
    # HELPER: for subclasses to declare adapt state.
    # Creates a non-trainable weight with the given name, shape, dtype, optional initializer,
    # partitioner and use_resource flag (extra keyword arguments pass to weight creation), with no
    # regularizer or constraint, and records it in the state variable mapping under its name.
    # Returns the weight. An existing entry with the same name is silently overwritten.
    """Add a variable that can hold state which is updated during adapt().

    Args:
      name: Variable name.
      shape: Variable shape. Defaults to scalar if unspecified.
      dtype: The type of the variable. Defaults to `self.dtype` or `float32`.
      initializer: initializer instance (callable).
      partitioner: Partitioner to be passed to the `Trackable` API.
      use_resource: Whether to use `ResourceVariable`
      **kwargs: Additional keyword arguments. Accepted values are `getter` and
        `collections`.

    Returns:
      The created variable.
    """
    weight = self.add_weight(
        name=name,
        shape=shape,
        dtype=dtype,
        initializer=initializer,
        regularizer=None,
        trainable=False,
        constraint=None,
        partitioner=partitioner,
        use_resource=use_resource,
        **kwargs)
    # TODO(momernick): Do not allow collisions here.
    self.state_variables[name] = weight
    return weight

  def _restore_updates(self):
    # [FUNCTION DOC]
    # HELPER: reads state variables for accumulator restoration.
    # Returns a mapping from each state variable name to its current value as a numpy array.
    """Recreates a dict of updates from the layer's weights."""
    data_dict = {}
    for name, var in self.state_variables.items():
      data_dict[name] = var.numpy()
    return data_dict

  def _get_accumulator(self):
    # [FUNCTION DOC]
    # HELPER: current accumulator access.
    # Returns an accumulator rebuilt from the state variables if the layer has been adapted,
    # otherwise nothing.
    if self._is_adapted:
      return self._combiner.restore(self._restore_updates())
    else:
      return None

  def _set_accumulator(self, accumulator):
    # [FUNCTION DOC]
    # HELPER: writes an accumulator into the layer state.
    # Extracts the accumulator into a mapping of values via the combiner, assigns them to the
    # state variables, and clears the in-progress adapt accumulator. Returns nothing; raises a
    # runtime error if the layer is not built.
    updates = self._combiner.extract(accumulator)
    self._set_state_variables(updates)
    self._adapt_accumulator = None  # Reset accumulator from adapt.

  def _set_state_variables(self, updates):
    # [FUNCTION DOC]
    # HELPER: assigns values to state variables.
    # KNOWN ISSUE: the docstring's error text refers to a method named set_processing_state, which
    # does not exist; the actual message names this method.
    # Takes a mapping from state variable names to values and assigns each to the matching
    # variable, under an init scope so it works from inside graph functions. Raises a runtime error
    # if the layer is not built; an unknown name causes a key error. Returns nothing.
    """Directly update the internal state of this Layer.

    This method expects a string-keyed dict of {state_variable_name: state}. The
    precise nature of the state, and the names associated, are describe by
    the subclasses of CombinerPreprocessingLayer.

    Args:
      updates: A string keyed dict of weights to update.

    Raises:
      RuntimeError: if 'build()' was not called before 'set_processing_state'.
    """
    # TODO(momernick): Do we need to do any more input sanitization?
    if not self.built:
      raise RuntimeError('_set_state_variables() must be called after build().')

    with ops.init_scope():
      for var_name, value in updates.items():
        self.state_variables[var_name].assign(value)


def convert_to_list(values, sparse_default_value=None):
  """Convert a TensorLike, CompositeTensor, or ndarray into a Python list."""
  if tf_utils.is_ragged(values):
    # There is a corner case when dealing with ragged tensors: if you get an
    # actual RaggedTensor (not a RaggedTensorValue) passed in non-eager mode,
    # you can't call to_list() on it without evaluating it first. However,
    # because we don't yet fully support composite tensors across Keras,
    # backend.get_value() won't evaluate the tensor.
    # TODO(momernick): Get Keras to recognize composite tensors as Tensors
    # and then replace this with a call to backend.get_value.
    if (isinstance(values, ragged_tensor.RaggedTensor) and
        not context.executing_eagerly()):
      values = backend.get_session(values).run(values)
    values = values.to_list()

  if isinstance(values,
                (sparse_tensor.SparseTensor, sparse_tensor.SparseTensorValue)):
    if sparse_default_value is None:
      if dtypes.as_dtype(values.values.dtype) == dtypes.string:
        sparse_default_value = ''
      else:
        sparse_default_value = -1
    dense_tensor = sparse_ops.sparse_tensor_to_dense(
        values, default_value=sparse_default_value)
    values = backend.get_value(dense_tensor)

  if isinstance(values, tensor.Tensor):
    values = backend.get_value(values)

  # We may get passed a ndarray or the code above may give us a ndarray.
  # In either case, we want to force it into a standard python list.
  if isinstance(values, np.ndarray):
    values = values.tolist()

  return values


# TODO(omalleyt): This class will be gradually replaced.
class Combiner(object):
  """Functional object that defines a shardable computation.

  This object defines functions required to create and manipulate data objects.
  These data objects, referred to below as 'accumulators', are computation-
  specific and may be implemented alongside concrete subclasses of Combiner
  (if necessary - some computations may be simple enough that standard Python
  types can be used as accumulators).

  The intent for this class is that by describing computations in this way, we
  can arbitrarily shard a dataset, perform computations on a subset, and then
  merge the computation into a final result. This enables distributed
  computation.

  The combiner itself does not own any state - all computational state is owned
  by the accumulator objects. This is so that we can have an arbitrary number of
  Combiners (thus sharding the computation N ways) without risking any change
  to the underlying computation. These accumulator objects are uniquely
  associated with each Combiner; a Combiner defines what the accumulator object
  should be and will only work with accumulators of that type.
  """
  __metaclass__ = abc.ABCMeta

  def __repr__(self):
    return '<{}>'.format(self.__class__.__name__)

  @abc.abstractmethod
  def compute(self, batch_values, accumulator=None):
    """Compute a step in this computation, returning a new accumulator.

    This method computes a step of the computation described by this Combiner.
    If an accumulator is passed, the data in that accumulator is also used; so
    compute(batch_values) results in f(batch_values), while
    compute(batch_values, accumulator) results in
    merge(f(batch_values), accumulator).

    Args:
      batch_values: A list of ndarrays representing the values of the inputs for
        this step of the computation.
      accumulator: the current accumulator. Can be None.

    Returns:
      An accumulator that includes the passed batch of inputs.
    """
    pass

  @abc.abstractmethod
  def merge(self, accumulators):
    """Merge several accumulators to a single accumulator.

    This method takes the partial values in several accumulators and combines
    them into a single accumulator. This computation must not be order-specific
    (that is, merge([a, b]) must return the same result as merge([b, a]).

    Args:
      accumulators: the accumulators to merge, as a list.

    Returns:
      A merged accumulator.
    """
    pass

  @abc.abstractmethod
  def extract(self, accumulator):
    """Convert an accumulator into a dict of output values.

    Args:
      accumulator: The accumulator to convert.

    Returns:
      A dict of ndarrays representing the data in this accumulator.
    """
    pass

  @abc.abstractmethod
  def restore(self, output):
    """Create an accumulator based on 'output'.

    This method creates a new accumulator with identical internal state to the
    one used to create the data in 'output'. This means that if you do

    output_data = combiner.extract(accumulator_1)
    accumulator_2 = combiner.restore(output_data)

    then accumulator_1 and accumulator_2 will have identical internal state, and
    computations using either of them will be equivalent.

    Args:
      output: The data output from a previous computation. Should be in the same
        form as provided by 'extract_output'.

    Returns:
      A new accumulator.
    """
    pass

  @abc.abstractmethod
  def serialize(self, accumulator):
    """Serialize an accumulator for a remote call.

    This function serializes an accumulator to be sent to a remote process.

    Args:
      accumulator: The accumulator to serialize.

    Returns:
      A byte string representing the passed accumulator.
    """
    pass

  @abc.abstractmethod
  def deserialize(self, encoded_accumulator):
    """Deserialize an accumulator received from 'serialize()'.

    This function deserializes an accumulator serialized by 'serialize()'.

    Args:
      encoded_accumulator: A byte string representing an accumulator.

    Returns:
      The accumulator represented by the passed byte_string.
    """
    pass


def _disallow_inside_tf_function(method_name):
  """Disallow calling a method inside a `tf.function`."""
  if ops.inside_function():
    error_msg = (
        'Detected a call to `PreprocessingLayer.{method_name}` inside a '
        '`tf.function`. `PreprocessingLayer.{method_name} is a high-level '
        'endpoint that manages its own `tf.function`. Please move the call '
        'to `PreprocessingLayer.{method_name}` outside of all enclosing '
        '`tf.function`s. Note that you can call a `PreprocessingLayer` '
        'directly on `Tensor`s inside a `tf.function` like: `layer(x)`, '
        'or update its state like: `layer.update_state(x)`.').format(
            method_name=method_name)
    raise RuntimeError(error_msg)
