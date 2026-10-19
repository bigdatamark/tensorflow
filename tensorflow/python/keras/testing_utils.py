# [MODULE DOC]
# HELPER: shared test-support module used by Keras unit tests across the package.
# Provides: synthetic classification data generation; a generic layer test routine that checks
# construction, weights, shape and dtype inference, serialization and training of a layer;
# thread-local scopes selecting the model type, eager execution and saved-model format under
# test; small MLP model factories in sequential, functional and subclassed styles; decorators
# that run a test across all model types, saved-model formats and execution modes; optimizer
# and layer-name helpers; and a few model-building utilities.
# Import-time side effects: creates a thread-local store and initializes four of its fields
# (model type, eager flag, saved model format, save arguments) to nothing for the importing
# thread only; imports the optimizer, layers and model modules.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for unit-testing Keras."""

import collections
import contextlib
import functools
import itertools
import threading

import numpy as np

from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.optimizer_v2 import adadelta as adadelta_v2
from tensorflow.python.keras.optimizer_v2 import adagrad as adagrad_v2
from tensorflow.python.keras.optimizer_v2 import adam as adam_v2
from tensorflow.python.keras.optimizer_v2 import adamax as adamax_v2
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_v2
from tensorflow.python.keras.optimizer_v2 import nadam as nadam_v2
from tensorflow.python.keras.optimizer_v2 import rmsprop as rmsprop_v2
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.util import tf_decorator


def string_test(actual, expected):
  # [FUNCTION DOC]
  # HELPER: equality comparator used by layer_test for string-typed outputs.
  # Asserts that two array-likes are exactly equal element by element. Inputs: actual and
  # expected arrays. Returns nothing; raises an assertion error on mismatch.
  np.testing.assert_array_equal(actual, expected)


def numeric_test(actual, expected):
  # [FUNCTION DOC]
  # HELPER: tolerance comparator used by layer_test for numeric outputs.
  # Asserts that two arrays are element-wise close, with relative tolerance 0.001 and absolute
  # tolerance 0.000001. Inputs: actual and expected arrays. Returns nothing; raises an
  # assertion error on mismatch.
  np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-6)


def get_test_data(train_samples,
                  test_samples,
                  input_shape,
                  num_classes,
                  random_seed=None):
  # [FUNCTION DOC]
  # HELPER: synthetic data generator for Keras tests.
  # Creates a labeled dataset: each class has a random template of the input shape (scaled by
  # twice the number of classes), labels are drawn uniformly at random, and each sample is its
  # class template plus unit-variance Gaussian noise (float32). Inputs: train_samples and
  # test_samples (integer counts), input_shape (tuple of integers), num_classes (integer) and
  # random_seed (optional integer; if given, seeds the global numpy generator, which is a side
  # effect). Returns a pair of pairs: (train inputs, train labels) and (test inputs, test
  # labels), the first train_samples samples going to training.
  """Generates test data to train a model on.

  Args:
    train_samples: Integer, how many training samples to generate.
    test_samples: Integer, how many test samples to generate.
    input_shape: Tuple of integers, shape of the inputs.
    num_classes: Integer, number of classes for the data and targets.
    random_seed: Integer, random seed used by numpy to generate data.

  Returns:
    A tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
  """
  if random_seed is not None:
    np.random.seed(random_seed)
  num_sample = train_samples + test_samples
  templates = 2 * num_classes * np.random.random((num_classes,) + input_shape)
  y = np.random.randint(0, num_classes, size=(num_sample,))
  x = np.zeros((num_sample,) + input_shape, dtype=np.float32)
  for i in range(num_sample):
    x[i] = templates[y[i]] + np.random.normal(loc=0, scale=1., size=input_shape)
  return ((x[:train_samples], y[:train_samples]),
          (x[train_samples:], y[train_samples:]))


@test_util.disable_cudnn_autotune
def layer_test(layer_cls,
               kwargs=None,
               input_shape=None,
               input_dtype=None,
               input_data=None,
               expected_output=None,
               expected_output_dtype=None,
               expected_output_shape=None,
               validate_training=True,
               adapt_data=None,
               custom_objects=None,
               test_harness=None,
               supports_masking=None):
  # [FUNCTION DOC]
  # HELPER: generic layer test routine shared by Keras layer tests.
  # KNOWN ISSUE: the check for a weights argument in the layer constructor tests membership of
  # a string in the argument-specification tuple (whose items are lists and values, never a
  # string), so it is effectively never true and the rebuild-from-weights step is dead code.
  # KNOWN ISSUE: the docstring says only the missing input shape raises an error, but mismatched
  # shapes, dtypes or masking support raise assertion errors, and the comment about rebuilding
  # the model before training is cut off mid-sentence.
  # Exercises one layer with a single input and output. Inputs: layer_cls (layer class),
  # kwargs (optional mapping of constructor arguments), input_shape (tuple, may contain nothing
  # entries replaced by random sizes), input_dtype (default float32 when data is generated),
  # input_data (optional array; generated randomly if absent), expected_output (optional array
  # to compare against), expected_output_dtype (default input dtype), expected_output_shape
  # (optional), validate_training (whether to compile and run a training step), adapt_data
  # (optional data for adapting preprocessing layers), custom_objects (optional mapping used
  # when rebuilding from configuration), test_harness (optional test case whose assertion
  # methods are used) and supports_masking (optional expected value of the layer's masking
  # property). Steps: instantiate; check masking flag; adapt; get and set weights; build a
  # functional model and check output dtype and declared shape; check computed output shape and
  # signature against actual predictions; compare to expected output; round-trip through
  # configuration and weights; train one batch (honoring the thread-local eager setting); then
  # rebuild the layer as the first layer of a sequential model and repeat the shape, output and
  # serialization checks. Returns the final prediction array. Raises a value error if neither
  # input data nor input shape is given, and assertion errors on any mismatch. Side effects:
  # uses the global numpy random generator and mutates the kwargs mapping.
  """Test routine for a layer with a single input and single output.

  Args:
    layer_cls: Layer class object.
    kwargs: Optional dictionary of keyword arguments for instantiating the
      layer.
    input_shape: Input shape tuple.
    input_dtype: Data type of the input data.
    input_data: Numpy array of input data.
    expected_output: Numpy array of the expected output.
    expected_output_dtype: Data type expected for the output.
    expected_output_shape: Shape tuple for the expected shape of the output.
    validate_training: Whether to attempt to validate training on this layer.
      This might be set to False for non-differentiable layers that output
      string or integer values.
    adapt_data: Optional data for an 'adapt' call. If None, adapt() will not
      be tested for this layer. This is only relevant for PreprocessingLayers.
    custom_objects: Optional dictionary mapping name strings to custom objects
      in the layer class. This is helpful for testing custom layers.
    test_harness: The Tensorflow test, if any, that this function is being
      called in.
    supports_masking: Optional boolean to check the `supports_masking` property
      of the layer. If None, the check will not be performed.

  Returns:
    The output data (Numpy array) returned by the layer, for additional
    checks to be done by the calling code.

  Raises:
    ValueError: if `input_shape is None`.
  """
  if input_data is None:
    if input_shape is None:
      raise ValueError('input_shape is None')
    if not input_dtype:
      input_dtype = 'float32'
    input_data_shape = list(input_shape)
    for i, e in enumerate(input_data_shape):
      if e is None:
        input_data_shape[i] = np.random.randint(1, 4)
    input_data = 10 * np.random.random(input_data_shape)
    if input_dtype[:5] == 'float':
      input_data -= 0.5
    input_data = input_data.astype(input_dtype)
  elif input_shape is None:
    input_shape = input_data.shape
  if input_dtype is None:
    input_dtype = input_data.dtype
  if expected_output_dtype is None:
    expected_output_dtype = input_dtype

  if dtypes.as_dtype(expected_output_dtype) == dtypes.string:
    if test_harness:
      assert_equal = test_harness.assertAllEqual
    else:
      assert_equal = string_test
  else:
    if test_harness:
      assert_equal = test_harness.assertAllClose
    else:
      assert_equal = numeric_test

  # instantiation
  kwargs = kwargs or {}
  layer = layer_cls(**kwargs)

  if (supports_masking is not None
      and layer.supports_masking != supports_masking):
    raise AssertionError(
        'When testing layer %s, the `supports_masking` property is %r'
        'but expected to be %r.\nFull kwargs: %s' %
        (layer_cls.__name__, layer.supports_masking, supports_masking, kwargs))

  # Test adapt, if data was passed.
  if adapt_data is not None:
    layer.adapt(adapt_data)

  # test get_weights , set_weights at layer level
  weights = layer.get_weights()
  layer.set_weights(weights)

  # test and instantiation from weights
  if 'weights' in tf_inspect.getargspec(layer_cls.__init__):
    kwargs['weights'] = weights
    layer = layer_cls(**kwargs)

  # test in functional API
  x = layers.Input(shape=input_shape[1:], dtype=input_dtype)
  y = layer(x)
  if backend.dtype(y) != expected_output_dtype:
    raise AssertionError('When testing layer %s, for input %s, found output '
                         'dtype=%s but expected to find %s.\nFull kwargs: %s' %
                         (layer_cls.__name__, x, backend.dtype(y),
                          expected_output_dtype, kwargs))

  def assert_shapes_equal(expected, actual):
    # [FUNCTION DOC]
    # HELPER: nested shape comparator for layer_test.
    # Compares two shapes dimension by dimension; dimension objects are converted to plain
    # values first and an unknown expected dimension matches anything. Raises an assertion
    # error (naming the layer, input and keyword arguments from the enclosing scope) if the
    # ranks differ or any known dimension differs. Returns nothing.
    """Asserts that the output shape from the layer matches the actual shape."""
    if len(expected) != len(actual):
      raise AssertionError(
          'When testing layer %s, for input %s, found output_shape='
          '%s but expected to find %s.\nFull kwargs: %s' %
          (layer_cls.__name__, x, actual, expected, kwargs))

    for expected_dim, actual_dim in zip(expected, actual):
      if isinstance(expected_dim, tensor_shape.Dimension):
        expected_dim = expected_dim.value
      if isinstance(actual_dim, tensor_shape.Dimension):
        actual_dim = actual_dim.value
      if expected_dim is not None and expected_dim != actual_dim:
        raise AssertionError(
            'When testing layer %s, for input %s, found output_shape='
            '%s but expected to find %s.\nFull kwargs: %s' %
            (layer_cls.__name__, x, actual, expected, kwargs))

  if expected_output_shape is not None:
    assert_shapes_equal(tensor_shape.TensorShape(expected_output_shape),
                        y.shape)

  # check shape inference
  model = models.Model(x, y)
  computed_output_shape = tuple(
      layer.compute_output_shape(
          tensor_shape.TensorShape(input_shape)).as_list())
  computed_output_signature = layer.compute_output_signature(
      tensor_spec.TensorSpec(shape=input_shape, dtype=input_dtype))
  actual_output = model.predict(input_data)
  actual_output_shape = actual_output.shape
  assert_shapes_equal(computed_output_shape, actual_output_shape)
  assert_shapes_equal(computed_output_signature.shape, actual_output_shape)
  if computed_output_signature.dtype != actual_output.dtype:
    raise AssertionError(
        'When testing layer %s, for input %s, found output_dtype='
        '%s but expected to find %s.\nFull kwargs: %s' %
        (layer_cls.__name__, x, actual_output.dtype,
         computed_output_signature.dtype, kwargs))
  if expected_output is not None:
    assert_equal(actual_output, expected_output)

  # test serialization, weight setting at model level
  model_config = model.get_config()
  recovered_model = models.Model.from_config(model_config, custom_objects)
  if model.weights:
    weights = model.get_weights()
    recovered_model.set_weights(weights)
    output = recovered_model.predict(input_data)
    assert_equal(output, actual_output)

  # test training mode (e.g. useful for dropout tests)
  # Rebuild the model to avoid the graph being reused between predict() and
  # See b/120160788 for more details. This should be mitigated after 2.0.
  layer_weights = layer.get_weights()  # Get the layer weights BEFORE training.
  if validate_training:
    model = models.Model(x, layer(x))
    if _thread_local_data.run_eagerly is not None:
      model.compile(
          'rmsprop',
          'mse',
          weighted_metrics=['acc'],
          run_eagerly=should_run_eagerly())
    else:
      model.compile('rmsprop', 'mse', weighted_metrics=['acc'])
    model.train_on_batch(input_data, actual_output)

  # test as first layer in Sequential API
  layer_config = layer.get_config()
  layer_config['batch_input_shape'] = input_shape
  layer = layer.__class__.from_config(layer_config)

  # Test adapt, if data was passed.
  if adapt_data is not None:
    layer.adapt(adapt_data)

  model = models.Sequential()
  model.add(layers.Input(shape=input_shape[1:], dtype=input_dtype))
  model.add(layer)

  layer.set_weights(layer_weights)
  actual_output = model.predict(input_data)
  actual_output_shape = actual_output.shape
  for expected_dim, actual_dim in zip(computed_output_shape,
                                      actual_output_shape):
    if expected_dim is not None:
      if expected_dim != actual_dim:
        raise AssertionError(
            'When testing layer %s **after deserialization**, '
            'for input %s, found output_shape='
            '%s but expected to find inferred shape %s.\nFull kwargs: %s' %
            (layer_cls.__name__,
             x,
             actual_output_shape,
             computed_output_shape,
             kwargs))
  if expected_output is not None:
    assert_equal(actual_output, expected_output)

  # test serialization, weight setting at model level
  model_config = model.get_config()
  recovered_model = models.Sequential.from_config(model_config, custom_objects)
  if model.weights:
    weights = model.get_weights()
    recovered_model.set_weights(weights)
    output = recovered_model.predict(input_data)
    assert_equal(output, actual_output)

  # for further checks in the caller function
  return actual_output


_thread_local_data = threading.local()
_thread_local_data.model_type = None
_thread_local_data.run_eagerly = None
_thread_local_data.saved_model_format = None
_thread_local_data.save_kwargs = None


@tf_contextlib.contextmanager
def model_type_scope(value):
  # [FUNCTION DOC]
  # HELPER: scope selecting the model type for tests.
  # Context manager: stores value as the current thread's model type, yields value, and in all
  # cases restores the previous model type on exit. Input: value (a model type name such as
  # sequential, functional or subclass). Side effect: thread-local state.
  """Provides a scope within which the model type to test is equal to `value`.

  The model type gets restored to its original value upon exiting the scope.

  Args:
     value: model type value

  Yields:
    The provided value.
  """
  previous_value = _thread_local_data.model_type
  try:
    _thread_local_data.model_type = value
    yield value
  finally:
    # Restore model type to initial value.
    _thread_local_data.model_type = previous_value


@tf_contextlib.contextmanager
def run_eagerly_scope(value):
  # [FUNCTION DOC]
  # HELPER: scope selecting whether test models run eagerly.
  # KNOWN ISSUE: the comment in the restore step says it restores the model type, but it
  # restores the eager flag.
  # Context manager: stores value (boolean) as the current thread's eager flag, yields it, and
  # restores the previous flag on exit even on error. Side effect: thread-local state.
  """Provides a scope within which we compile models to run eagerly or not.

  The boolean gets restored to its original value upon exiting the scope.

  Args:
     value: Bool specifying if we should run models eagerly in the active test.
     Should be True or False.

  Yields:
    The provided value.
  """
  previous_value = _thread_local_data.run_eagerly
  try:
    _thread_local_data.run_eagerly = value
    yield value
  finally:
    # Restore model type to initial value.
    _thread_local_data.run_eagerly = previous_value


def should_run_eagerly():
  # [FUNCTION DOC]
  # HELPER: query for model-compilation code in tests.
  # Returns true only if the thread's eager flag is true and execution is currently eager.
  # Raises a value error if no flag has been set (outside an eager scope or the decorator that
  # sets it). Reads thread-local state and the execution context; no mutation.
  """Returns whether the models we are testing should be run eagerly."""
  if _thread_local_data.run_eagerly is None:
    raise ValueError('Cannot call `should_run_eagerly()` outside of a '
                     '`run_eagerly_scope()` or `run_all_keras_modes` '
                     'decorator.')

  return _thread_local_data.run_eagerly and context.executing_eagerly()


@tf_contextlib.contextmanager
def saved_model_format_scope(value, **kwargs):
  # [FUNCTION DOC]
  # HELPER: scope selecting the saved-model format for tests.
  # KNOWN ISSUE: the docstring claims the provided value is yielded, but the code yields nothing.
  # Context manager: stores value (format name) and kwargs (extra arguments for the save call)
  # as the thread's current format and save arguments, yields nothing, and restores both
  # previous values on exit. Side effect: thread-local state.
  """Provides a scope within which the savde model format to test is `value`.

  The saved model format gets restored to its original value upon exiting the
  scope.

  Args:
     value: saved model format value
     **kwargs: optional kwargs to pass to the save function.

  Yields:
    The provided value.
  """
  previous_format = _thread_local_data.saved_model_format
  previous_kwargs = _thread_local_data.save_kwargs
  try:
    _thread_local_data.saved_model_format = value
    _thread_local_data.save_kwargs = kwargs
    yield
  finally:
    # Restore saved model format to initial value.
    _thread_local_data.saved_model_format = previous_format
    _thread_local_data.save_kwargs = previous_kwargs


def get_save_format():
  # [FUNCTION DOC]
  # HELPER: returns the saved-model format of the current scope.
  # Raises a value error if none is set (outside the scope or the decorator that sets it).
  # Reads thread-local state; takes no inputs.
  if _thread_local_data.saved_model_format is None:
    raise ValueError(
        'Cannot call `get_save_format()` outside of a '
        '`saved_model_format_scope()` or `run_with_all_saved_model_formats` '
        'decorator.')
  return _thread_local_data.saved_model_format


def get_save_kwargs():
  # [FUNCTION DOC]
  # HELPER: returns the extra save arguments of the current scope.
  # Returns the mapping stored by the scope (an empty mapping if it was empty). Raises a value
  # error if no scope is active, i.e. the stored value is nothing. Reads thread-local state.
  if _thread_local_data.save_kwargs is None:
    raise ValueError(
        'Cannot call `get_save_kwargs()` outside of a '
        '`saved_model_format_scope()` or `run_with_all_saved_model_formats` '
        'decorator.')
  return _thread_local_data.save_kwargs or {}


def get_model_type():
  # [FUNCTION DOC]
  # HELPER: returns the model type name of the current scope. Raises a value error if none is
  # set (outside the scope or the decorator that sets it). Reads thread-local state.
  """Gets the model type that should be tested."""
  if _thread_local_data.model_type is None:
    raise ValueError('Cannot call `get_model_type()` outside of a '
                     '`model_type_scope()` or `run_with_all_model_types` '
                     'decorator.')

  return _thread_local_data.model_type


def get_small_sequential_mlp(num_hidden, num_classes, input_dim=None):
  # [FUNCTION DOC]
  # HELPER: small model factory for tests.
  # Builds an uncompiled sequential model: a hidden dense layer with rectified-linear activation
  # (num_hidden units, given the input dimension if input_dim is set and nonzero) followed by
  # an output dense layer of num_classes units with sigmoid activation if there is one class,
  # otherwise softmax. Returns the model.
  model = models.Sequential()
  if input_dim:
    model.add(layers.Dense(num_hidden, activation='relu', input_dim=input_dim))
  else:
    model.add(layers.Dense(num_hidden, activation='relu'))
  activation = 'sigmoid' if num_classes == 1 else 'softmax'
  model.add(layers.Dense(num_classes, activation=activation))
  return model


def get_small_functional_mlp(num_hidden, num_classes, input_dim):
  # [FUNCTION DOC]
  # HELPER: small model factory for tests.
  # Builds the same hidden-plus-output dense network as the sequential factory, using the
  # functional style with an input of the given dimension. Output activation is sigmoid for one
  # class, else softmax. Returns the uncompiled functional model.
  inputs = layers.Input(shape=(input_dim,))
  outputs = layers.Dense(num_hidden, activation='relu')(inputs)
  activation = 'sigmoid' if num_classes == 1 else 'softmax'
  outputs = layers.Dense(num_classes, activation=activation)(outputs)
  return models.Model(inputs, outputs)


class SmallSubclassMLP(models.Model):
  # [CLASS DOC]
  # HELPER: small subclassed model used by tests.
  # A two-layer dense network (hidden layer then output layer) with optional dropout and batch
  # normalization between them. Holds the two dense layers and the optional extra layers as
  # attributes. Extends the Keras model class and is fixed to the model name test_model.
  """A subclass model based small MLP."""

  def __init__(self,
               num_hidden,
               num_classes,
               use_bn=False,
               use_dp=False,
               **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: num_hidden (hidden units), num_classes (output units; one class gives
    # sigmoid output, otherwise softmax), use_bn (add batch normalization, default false),
    # use_dp (add dropout with rate 0.5, default false) and extra keyword arguments forwarded to
    # the parent constructor. Creates the layers and stores the flags. The model name is forced
    # to test_model, so a name passed in the keyword arguments would cause a duplicate-argument
    # error. Returns nothing.
    super(SmallSubclassMLP, self).__init__(name='test_model', **kwargs)
    self.use_bn = use_bn
    self.use_dp = use_dp

    self.layer_a = layers.Dense(num_hidden, activation='relu')
    activation = 'sigmoid' if num_classes == 1 else 'softmax'
    self.layer_b = layers.Dense(num_classes, activation=activation)
    if self.use_dp:
      self.dp = layers.Dropout(0.5)
    if self.use_bn:
      self.bn = layers.BatchNormalization(axis=-1)

  def call(self, inputs, **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: extra keyword arguments such as a training flag are swallowed, so dropout and
    # batch normalization rely on the ambient learning phase instead of an explicit argument.
    # Forward pass: applies the hidden layer, then dropout if enabled, then batch normalization
    # if enabled, then the output layer, and returns the result. Extra keyword arguments (such
    # as a training flag) are accepted but not passed on to the sublayers.
    x = self.layer_a(inputs)
    if self.use_dp:
      x = self.dp(x)
    if self.use_bn:
      x = self.bn(x)
    return self.layer_b(x)


class _SmallSubclassMLPCustomBuild(models.Model):
  # [CLASS DOC]
  # HELPER: small subclassed model for tests of models that create their layers in a custom
  # build method. Stores the sizes at construction; the two dense layers are created lazily
  # when the model is built from an input shape.
  """A subclass model small MLP that uses a custom build method."""

  def __init__(self, num_hidden, num_classes):
    # [FUNCTION DOC]
    # Constructor. Inputs: num_hidden and num_classes (integers). Calls the parent constructor,
    # sets the two layer attributes to nothing and stores the sizes. Returns nothing.
    super(_SmallSubclassMLPCustomBuild, self).__init__()
    self.layer_a = None
    self.layer_b = None
    self.num_hidden = num_hidden
    self.num_classes = num_classes

  def build(self, input_shape):
    # [FUNCTION DOC]
    # Custom build hook. Creates the hidden dense layer (rectified-linear) and the output dense
    # layer (sigmoid for one class, else softmax). The input shape argument is ignored. Returns
    # nothing; sets the layer attributes.
    self.layer_a = layers.Dense(self.num_hidden, activation='relu')
    activation = 'sigmoid' if self.num_classes == 1 else 'softmax'
    self.layer_b = layers.Dense(self.num_classes, activation=activation)

  def call(self, inputs, **kwargs):
    # [FUNCTION DOC]
    # Forward pass: applies the hidden layer then the output layer and returns the result.
    # Extra keyword arguments are accepted and ignored.
    x = self.layer_a(inputs)
    return self.layer_b(x)


def get_small_subclass_mlp(num_hidden, num_classes):
  # [FUNCTION DOC]
  # HELPER: small model factory for tests. Returns a new subclassed two-layer network with the
  # given hidden size and class count (no batch normalization or dropout).
  return SmallSubclassMLP(num_hidden, num_classes)


def get_small_subclass_mlp_with_custom_build(num_hidden, num_classes):
  # [FUNCTION DOC]
  # HELPER: small model factory for tests. Returns a new subclassed two-layer network whose
  # layers are created lazily in a custom build step, with the given hidden size and classes.
  return _SmallSubclassMLPCustomBuild(num_hidden, num_classes)


def get_small_mlp(num_hidden, num_classes, input_dim):
  # [FUNCTION DOC]
  # HELPER: model factory dispatching on the current scope's model type.
  # Inputs: num_hidden, num_classes and input_dim (integers; input_dim is used only by the
  # sequential and functional types). Returns a small network of the type named subclass,
  # subclass_custom_build, sequential or functional. Raises a value error for an unknown type,
  # and also if no model type scope is active.
  """Get a small mlp of the model type specified by `get_model_type`."""
  model_type = get_model_type()
  if model_type == 'subclass':
    return get_small_subclass_mlp(num_hidden, num_classes)
  if model_type == 'subclass_custom_build':
    return get_small_subclass_mlp_with_custom_build(num_hidden, num_classes)
  if model_type == 'sequential':
    return get_small_sequential_mlp(num_hidden, num_classes, input_dim)
  if model_type == 'functional':
    return get_small_functional_mlp(num_hidden, num_classes, input_dim)
  raise ValueError('Unknown model type {}'.format(model_type))


class _SubclassModel(models.Model):
  # [CLASS DOC]
  # HELPER: subclassed model wrapping a list of layers applied in order, used when tests need
  # a subclass version of a sequential stack. Each layer becomes a numbered attribute (for
  # example layer0); the layer count is stored. Optionally sets inputs from a given tensor.
  """A Keras subclass model."""

  def __init__(self, model_layers, *args, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: model_layers (list of layers), extra positional and keyword arguments
    # forwarded to the parent constructor, except an optional input_tensor keyword (needed for
    # ragged or sparse inputs) which is removed first. Registers each layer as a numbered
    # attribute, records the layer count, and if an input tensor was given, sets the model
    # inputs from it. Returns nothing.
    """Instantiate a model.

    Args:
      model_layers: a list of layers to be added to the model.
      *args: Model's args
      **kwargs: Model's keyword args, at most one of input_tensor -> the input
        tensor required for ragged/sparse input.
    """

    inputs = kwargs.pop('input_tensor', None)
    super(_SubclassModel, self).__init__(*args, **kwargs)
    # Note that clone and build doesn't support lists of layers in subclassed
    # models. Adding each layer directly here.
    for i, layer in enumerate(model_layers):
      setattr(self, self._layer_name_for_i(i), layer)

    self.num_layers = len(model_layers)

    if inputs is not None:
      self._set_inputs(inputs)

  def _layer_name_for_i(self, i):
    # [FUNCTION DOC]
    # HELPER: attribute-name generator. Returns the string layer followed by the index i, used
    # to store and look up the i-th layer of the model.
    return 'layer{}'.format(i)

  def call(self, inputs, **kwargs):
    # [FUNCTION DOC]
    # Forward pass: feeds the inputs through each stored layer in index order and returns the
    # final result. Extra keyword arguments are accepted and ignored.
    x = inputs
    for i in range(self.num_layers):
      layer = getattr(self, self._layer_name_for_i(i))
      x = layer(x)
    return x


class _SubclassModelCustomBuild(models.Model):
  # [CLASS DOC]
  # HELPER: subclassed model whose layers come from a user-supplied generator function and are
  # created lazily in the build step. Holds the generator and, after building, the layer list.
  """A Keras subclass model that uses a custom build method."""

  def __init__(self, layer_generating_func, *args, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Inputs: layer_generating_func (callable with no arguments returning an
    # iterable of layers), plus extra arguments forwarded to the parent constructor. Stores the
    # callable and sets the layer list to nothing. Returns nothing.
    super(_SubclassModelCustomBuild, self).__init__(*args, **kwargs)
    self.all_layers = None
    self._layer_generating_func = layer_generating_func

  def build(self, input_shape):
    # [FUNCTION DOC]
    # Custom build hook. Calls the stored generator function, collects the layers it produces
    # into a list and stores it as the model's layer list. The input shape is ignored.
    # Returns nothing.
    model_layers = []
    for layer in self._layer_generating_func():
      model_layers.append(layer)
    self.all_layers = model_layers

  def call(self, inputs, **kwargs):
    # [FUNCTION DOC]
    # Forward pass: feeds the inputs through every layer in the built layer list in order and
    # returns the result. Fails if the model has not been built. Extra keyword arguments ignored.
    x = inputs
    for layer in self.all_layers:
      x = layer(x)
    return x


def get_model_from_layers(model_layers,
                          input_shape=None,
                          input_dtype=None,
                          name=None,
                          input_ragged=None,
                          input_sparse=None,
                          model_type=None):
  # [FUNCTION DOC]
  # HELPER: model factory used by tests to build the same layer stack in different styles.
  # Inputs: model_layers (list of layers), input_shape (optional shape without batch
  # dimension), input_dtype, name (optional model name), input_ragged and input_sparse
  # (optional flags for ragged or sparse inputs), and model_type (one of subclass,
  # subclass_custom_build, sequential, functional; when nothing, the current scope's type is
  # used). Subclass: wraps the layers, creating an explicit input only for ragged or sparse
  # data. Custom-build subclass: layers are supplied lazily. Sequential: adds an input layer if
  # a shape is given, then each layer. Functional: requires an input shape, chains the layers
  # on an input. Returns the model. Raises a value error for a functional model without an
  # input shape, or an unknown model type (or no scope when model_type is nothing).
  """Builds a model from a sequence of layers.

  Args:
    model_layers: The layers used to build the network.
    input_shape: Shape tuple of the input or 'TensorShape' instance.
    input_dtype: Datatype of the input.
    name: Name for the model.
    input_ragged: Boolean, whether the input data is a ragged tensor.
    input_sparse: Boolean, whether the input data is a sparse tensor.
    model_type: One of "subclass", "subclass_custom_build", "sequential", or
      "functional". When None, defaults to `get_model_type`.

  Returns:
    A Keras model.
  """
  if model_type is None:
    model_type = get_model_type()
  if model_type == 'subclass':
    inputs = None
    if input_ragged or input_sparse:
      inputs = layers.Input(
          shape=input_shape,
          dtype=input_dtype,
          ragged=input_ragged,
          sparse=input_sparse)
    return _SubclassModel(model_layers, name=name, input_tensor=inputs)

  if model_type == 'subclass_custom_build':
    layer_generating_func = lambda: model_layers
    return _SubclassModelCustomBuild(layer_generating_func, name=name)

  if model_type == 'sequential':
    model = models.Sequential(name=name)
    if input_shape:
      model.add(
          layers.InputLayer(
              input_shape=input_shape,
              dtype=input_dtype,
              ragged=input_ragged,
              sparse=input_sparse))
    for layer in model_layers:
      model.add(layer)
    return model

  if model_type == 'functional':
    if not input_shape:
      raise ValueError('Cannot create a functional model from layers with no '
                       'input shape.')
    inputs = layers.Input(
        shape=input_shape,
        dtype=input_dtype,
        ragged=input_ragged,
        sparse=input_sparse)
    outputs = inputs
    for layer in model_layers:
      outputs = layer(outputs)
    return models.Model(inputs, outputs, name=name)

  raise ValueError('Unknown model type {}'.format(model_type))


class Bias(layers.Layer):
  # [CLASS DOC]
  # HELPER: tiny test layer that adds one trainable scalar bias (initialized to zero) to its
  # input. Holds the bias variable created at build time.

  def build(self, input_shape):
    # [FUNCTION DOC]
    # Creates the single-element bias variable, initialized to zeros, and stores it. The input
    # shape is ignored. Returns nothing.
    self.bias = self.add_variable('bias', (1,), initializer='zeros')

  def call(self, inputs):
    # [FUNCTION DOC]
    # Returns the inputs with the scalar bias added (broadcast). No other side effects.
    return inputs + self.bias


class _MultiIOSubclassModel(models.Model):
  # [CLASS DOC]
  # HELPER: subclassed model with two layer branches, an optional shared input stack and an
  # optional shared output stack, built by get_multi_io_model. Holds the four layer lists.
  """Multi IO Keras subclass model."""

  def __init__(self, branch_a, branch_b, shared_input_branch=None,
               shared_output_branch=None, name=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: branch_a and branch_b (lists of layers), shared_input_branch and
    # shared_output_branch (optional lists of layers, default nothing) and name (optional model
    # name). Stores the lists. Returns nothing.
    super(_MultiIOSubclassModel, self).__init__(name=name)
    self._shared_input_branch = shared_input_branch
    self._branch_a = branch_a
    self._branch_b = branch_b
    self._shared_output_branch = shared_output_branch

  def call(self, inputs, **kwargs):
    # [FUNCTION DOC]
    # Forward pass. If a shared input stack exists, passes the single input through it and feeds
    # the result to both branches; otherwise splits the inputs into two parts (from a mapping
    # with keys input_1 and input_2, or from a two-element sequence). Applies branch a and
    # branch b layers in order, forms a list of the two results, then, if a shared output stack
    # exists, passes that list through it. Returns the list of two outputs, or the output of the
    # shared stack. Extra keyword arguments are ignored.
    if self._shared_input_branch:
      for layer in self._shared_input_branch:
        inputs = layer(inputs)
      a = inputs
      b = inputs
    elif isinstance(inputs, dict):
      a = inputs['input_1']
      b = inputs['input_2']
    else:
      a, b = inputs

    for layer in self._branch_a:
      a = layer(a)
    for layer in self._branch_b:
      b = layer(b)
    outs = [a, b]

    if self._shared_output_branch:
      for layer in self._shared_output_branch:
        outs = layer(outs)

    return outs


class _MultiIOSubclassModelCustomBuild(models.Model):
  # [CLASS DOC]
  # HELPER: like the multi-input/multi-output subclass model, but the layer lists come from
  # callables evaluated lazily in the build step. Holds the four callables and, after building,
  # the four layer lists.
  """Multi IO Keras subclass model that uses a custom build method."""

  def __init__(self, branch_a_func, branch_b_func,
               shared_input_branch_func=None,
               shared_output_branch_func=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: shared branch callables default to nothing but build calls them without a
    # null check, so the defaults are unusable.
    # Constructor. Inputs: branch_a_func and branch_b_func (callables returning layer lists),
    # shared_input_branch_func and shared_output_branch_func (optional callables, default
    # nothing; note that build calls them unconditionally, so omitting them makes build fail).
    # Stores the callables and sets all four layer lists to nothing. Returns nothing.
    super(_MultiIOSubclassModelCustomBuild, self).__init__()
    self._shared_input_branch_func = shared_input_branch_func
    self._branch_a_func = branch_a_func
    self._branch_b_func = branch_b_func
    self._shared_output_branch_func = shared_output_branch_func

    self._shared_input_branch = None
    self._branch_a = None
    self._branch_b = None
    self._shared_output_branch = None

  def build(self, input_shape):
    # [FUNCTION DOC]
    # Custom build hook. Calls the shared-input callable (calling it twice) and stores its
    # result if non-empty, calls the callables for branch a and branch b, and stores the shared
    # output list if its callable returns something non-empty. Input shape is ignored. Returns
    # nothing. Raises a type error if a shared callable is nothing.
    if self._shared_input_branch_func():
      self._shared_input_branch = self._shared_input_branch_func()
    self._branch_a = self._branch_a_func()
    self._branch_b = self._branch_b_func()

    if self._shared_output_branch_func():
      self._shared_output_branch = self._shared_output_branch_func()

  def call(self, inputs, **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: unlike the non-lazy variant, a mapping of named inputs is not supported here;
    # inputs must be a two-element sequence (or a single input when a shared stack exists), and
    # the results are a tuple rather than a list.
    # Forward pass. With a shared input stack, passes the single input through it and feeds both
    # branches; otherwise unpacks two inputs. Applies each branch, groups the results in a pair,
    # and passes the pair through the shared output stack if one exists. Returns the pair or the
    # stack output. Extra keyword arguments are ignored.
    if self._shared_input_branch:
      for layer in self._shared_input_branch:
        inputs = layer(inputs)
      a = inputs
      b = inputs
    else:
      a, b = inputs

    for layer in self._branch_a:
      a = layer(a)
    for layer in self._branch_b:
      b = layer(b)
    outs = a, b

    if self._shared_output_branch:
      for layer in self._shared_output_branch:
        outs = layer(outs)

    return outs


def get_multi_io_model(
    branch_a,
    branch_b,
    shared_input_branch=None,
    shared_output_branch=None):
  # [FUNCTION DOC]
  # HELPER: model factory for multi-input / multi-output tests.
  # KNOWN ISSUE: the docstring example for the shared-input case misspells the first branch
  # variable name, and the sequential model type is not supported.
  # Inputs: branch_a and branch_b (sequences of layers; each starts with an input layer unless a
  # shared input branch is given), shared_input_branch (optional sequence whose first item is
  # the input layer for the whole model) and shared_output_branch (optional sequence merging the
  # two branch outputs). The leading input layers are split off, then a model of the current
  # scope's type is built: a subclass model, a lazily built subclass model, or a functional model
  # (inputs chained through the shared input stack if present, both branches, then the shared
  # output stack if present). Returns the model. Raises a value error for the sequential type
  # or an unknown type (or when no model type scope is active).
  """Builds a multi-io model that contains two branches.

  The produced model will be of the type specified by `get_model_type`.

  To build a two-input, two-output model:
    Specify a list of layers for branch a and branch b, but do not specify any
    shared input branch or shared output branch. The resulting model will apply
    each branch to a different input, to produce two outputs.

    The first value in branch_a must be the Keras 'Input' layer for branch a,
    and the first value in branch_b must be the Keras 'Input' layer for
    branch b.

    example usage:
    ```
    branch_a = [Input(shape=(2,), name='a'), Dense(), Dense()]
    branch_b = [Input(shape=(3,), name='b'), Dense(), Dense()]

    model = get_multi_io_model(branch_a, branch_b)
    ```

  To build a two-input, one-output model:
    Specify a list of layers for branch a and branch b, and specify a
    shared output branch. The resulting model will apply
    each branch to a different input. It will then apply the shared output
    branch to a tuple containing the intermediate outputs of each branch,
    to produce a single output. The first layer in the shared_output_branch
    must be able to merge a tuple of two tensors.

    The first value in branch_a must be the Keras 'Input' layer for branch a,
    and the first value in branch_b must be the Keras 'Input' layer for
    branch b.

    example usage:
    ```
    input_branch_a = [Input(shape=(2,), name='a'), Dense(), Dense()]
    input_branch_b = [Input(shape=(3,), name='b'), Dense(), Dense()]
    shared_output_branch = [Concatenate(), Dense(), Dense()]

    model = get_multi_io_model(input_branch_a, input_branch_b,
                               shared_output_branch=shared_output_branch)
    ```
  To build a one-input, two-output model:
    Specify a list of layers for branch a and branch b, and specify a
    shared input branch. The resulting model will take one input, and apply
    the shared input branch to it. It will then respectively apply each branch
    to that intermediate result in parallel, to produce two outputs.

    The first value in the shared_input_branch must be the Keras 'Input' layer
    for the whole model. Branch a and branch b should not contain any Input
    layers.

    example usage:
    ```
    shared_input_branch = [Input(shape=(2,), name='in'), Dense(), Dense()]
    output_branch_a = [Dense(), Dense()]
    output_branch_b = [Dense(), Dense()]


    model = get_multi_io_model(output__branch_a, output_branch_b,
                               shared_input_branch=shared_input_branch)
    ```

  Args:
    branch_a: A sequence of layers for branch a of the model.
    branch_b: A sequence of layers for branch b of the model.
    shared_input_branch: An optional sequence of layers to apply to a single
      input, before applying both branches to that intermediate result. If set,
      the model will take only one input instead of two. Defaults to None.
    shared_output_branch: An optional sequence of layers to merge the
      intermediate results produced by branch a and branch b. If set,
      the model will produce only one output instead of two. Defaults to None.

  Returns:
    A multi-io model of the type specified by `get_model_type`, specified
    by the different branches.
  """
  # Extract the functional inputs from the layer lists
  if shared_input_branch:
    inputs = shared_input_branch[0]
    shared_input_branch = shared_input_branch[1:]
  else:
    inputs = branch_a[0], branch_b[0]
    branch_a = branch_a[1:]
    branch_b = branch_b[1:]

  model_type = get_model_type()
  if model_type == 'subclass':
    return _MultiIOSubclassModel(branch_a, branch_b, shared_input_branch,
                                 shared_output_branch)

  if model_type == 'subclass_custom_build':
    return _MultiIOSubclassModelCustomBuild((lambda: branch_a),
                                            (lambda: branch_b),
                                            (lambda: shared_input_branch),
                                            (lambda: shared_output_branch))

  if model_type == 'sequential':
    raise ValueError('Cannot use `get_multi_io_model` to construct '
                     'sequential models')

  if model_type == 'functional':
    if shared_input_branch:
      a_and_b = inputs
      for layer in shared_input_branch:
        a_and_b = layer(a_and_b)
      a = a_and_b
      b = a_and_b
    else:
      a, b = inputs

    for layer in branch_a:
      a = layer(a)
    for layer in branch_b:
      b = layer(b)
    outputs = a, b

    if shared_output_branch:
      for layer in shared_output_branch:
        outputs = layer(outputs)

    return models.Model(inputs, outputs)

  raise ValueError('Unknown model type {}'.format(model_type))


_V2_OPTIMIZER_MAP = {
    'adadelta': adadelta_v2.Adadelta,
    'adagrad': adagrad_v2.Adagrad,
    'adam': adam_v2.Adam,
    'adamax': adamax_v2.Adamax,
    'nadam': nadam_v2.Nadam,
    'rmsprop': rmsprop_v2.RMSprop,
    'sgd': gradient_descent_v2.SGD
}


def get_v2_optimizer(name, **kwargs):
  # [FUNCTION DOC]
  # HELPER: optimizer lookup for tests.
  # KNOWN ISSUE: a missing-key error raised inside the optimizer constructor itself would also be
  # reported as an unknown name, since the whole construction sits inside the lookup guard.
  # Looks up the name in the module's mapping of v2 optimizer names (adadelta, adagrad, adam,
  # adamax, nadam, rmsprop, sgd) and constructs the optimizer with the given keyword arguments.
  # Returns the new optimizer. Raises a value error listing valid names if the name is unknown.
  """Get the v2 optimizer requested.

  This is only necessary until v2 are the default, as we are testing in Eager,
  and Eager + v1 optimizers fail tests. When we are in v2, the strings alone
  should be sufficient, and this mapping can theoretically be removed.

  Args:
    name: string name of Keras v2 optimizer.
    **kwargs: any kwargs to pass to the optimizer constructor.

  Returns:
    Initialized Keras v2 optimizer.

  Raises:
    ValueError: if an unknown name was passed.
  """
  try:
    return _V2_OPTIMIZER_MAP[name](**kwargs)
  except KeyError:
    raise ValueError(
        'Could not find requested v2 optimizer: {}\nValid choices: {}'.format(
            name, list(_V2_OPTIMIZER_MAP.keys())))


def get_expected_metric_variable_names(var_names, name_suffix=''):
  # [FUNCTION DOC]
  # HELPER: expected-name builder for metric tests.
  # Input: var_names (list of base variable names) and name_suffix (string appended in graph
  # mode, default empty). If the version-2 behavior is enabled or execution is eager, returns
  # each name followed by the zero output index suffix; otherwise returns each name with the
  # suffix and then the index suffix (graph mode makes names unique). Returns a list of strings.
  """Returns expected metric variable names given names and prefix/suffix."""
  if tf2.enabled() or context.executing_eagerly():
    # In V1 eager mode and V2 variable names are not made unique.
    return [n + ':0' for n in var_names]
  # In V1 graph mode variable names are made unique using a suffix.
  return [n + name_suffix + ':0' for n in var_names]


def enable_v2_dtype_behavior(fn):
  # [FUNCTION DOC]
  # HELPER: test decorator. Returns a version of the test function fn that runs with the layer
  # version-2 dtype behavior turned on, restoring the previous setting afterwards.
  """Decorator for enabling the layer V2 dtype behavior on a test."""
  return _set_v2_dtype_behavior(fn, True)


def disable_v2_dtype_behavior(fn):
  # [FUNCTION DOC]
  # HELPER: test decorator. Returns a version of the test function fn that runs with the layer
  # version-2 dtype behavior turned off, restoring the previous setting afterwards.
  """Decorator for disabling the layer V2 dtype behavior on a test."""
  return _set_v2_dtype_behavior(fn, False)


def _set_v2_dtype_behavior(fn, enabled):
  # [FUNCTION DOC]
  # HELPER: shared implementation of the enable and disable dtype-behavior decorators.
  # Inputs: fn (function to wrap) and enabled (boolean). Returns a wrapper, registered via the
  # decorator utility so the original function's metadata and signature are preserved, that
  # sets the global flag in the base layer utilities to enabled, calls fn with all arguments,
  # returns its result, and always restores the previous flag value.
  """Returns version of 'fn' that runs with v2 dtype behavior on or off."""
  @functools.wraps(fn)
  def wrapper(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: nested wrapper. Saves the global dtype-behavior flag, sets it to the enabled value
    # captured from the outer scope, calls the wrapped function with all arguments and returns
    # its result; the saved flag is restored in all cases, including on error.
    v2_dtype_behavior = base_layer_utils.V2_DTYPE_BEHAVIOR
    base_layer_utils.V2_DTYPE_BEHAVIOR = enabled
    try:
      return fn(*args, **kwargs)
    finally:
      base_layer_utils.V2_DTYPE_BEHAVIOR = v2_dtype_behavior

  return tf_decorator.make_decorator(fn, wrapper)


@contextlib.contextmanager
def device(should_use_gpu):
  # [FUNCTION DOC]
  # HELPER: device-placement context manager for tests.
  # Input: should_use_gpu (boolean). Runs the body under the first GPU device if requested and a
  # GPU is available, otherwise under the first CPU device. Yields nothing. Side effect:
  # temporarily changes the default device scope.
  """Uses gpu when requested and available."""
  if should_use_gpu and test_util.is_gpu_available():
    dev = '/device:GPU:0'
  else:
    dev = '/device:CPU:0'
  with ops.device(dev):
    yield


@contextlib.contextmanager
def use_gpu():
  # [FUNCTION DOC]
  # HELPER: context manager that runs the body on the GPU when one is available, otherwise on
  # the CPU, by delegating to the device helper with GPU requested. Yields nothing.
  """Uses gpu when requested and available."""
  with device(should_use_gpu=True):
    yield


def for_all_test_methods(decorator, *args, **kwargs):
  # [FUNCTION DOC]
  # HELPER: converts a method-level test decorator into a class decorator.
  # Inputs: decorator (a decorator factory), and positional and keyword arguments passed to the
  # factory for each method. Returns a class decorator that, for every callable attribute whose
  # name starts with test (except test_session), replaces it on the class with the decorated
  # version, then returns the class. Side effect: mutates the decorated class.
  """Generate class-level decorator from given method-level decorator.

  It is expected for the given decorator to take some arguments and return
  a method that is then called on the test method to produce a decorated
  method.

  Args:
    decorator: The decorator to apply.
    *args: Positional arguments
    **kwargs: Keyword arguments
  Returns: Function that will decorate a given classes test methods with the
    decorator.
  """

  def all_test_methods_impl(cls):
    # [FUNCTION DOC]
    # HELPER: the class decorator returned by for_all_test_methods. Replaces each test method
    # of the class (callable, name starting with test, not test_session) with the method
    # wrapped by the configured decorator, and returns the same class (mutated in place).
    """Apply decorator to all test methods in class."""
    for name in dir(cls):
      value = getattr(cls, name)
      if callable(value) and name.startswith('test') and (name !=
                                                          'test_session'):
        setattr(cls, name, decorator(*args, **kwargs)(value))
    return cls

  return all_test_methods_impl


# The description is just for documentation purposes.
def run_without_tensor_float_32(description):  # pylint: disable=unused-argument
  # [FUNCTION DOC]
  # HELPER: decorator factory for tests. Input: description (documentation-only string, unused).
  # Returns a decorator that makes a test method run with TensorFloat-32 execution disabled,
  # restoring the previous setting afterwards.
  """Execute test with TensorFloat-32 disabled.

  While almost every real-world deep learning model runs fine with
  TensorFloat-32, many tests use assertAllClose or similar methods.
  TensorFloat-32 matmuls typically will cause such methods to fail with the
  default tolerances.

  Args:
    description: A description used for documentation purposes, describing why
      the test requires TensorFloat-32 to be disabled.

  Returns:
    Decorator which runs a test with TensorFloat-32 disabled.
  """

  def decorator(f):
    # [FUNCTION DOC]
    # HELPER: the decorator returned by the TensorFloat-32 factory. Input: f, a test method.
    # Returns a wrapper (preserving f's metadata) that disables TensorFloat-32 execution around
    # f.

    @functools.wraps(f)
    def decorated(self, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: nested wrapper for the TensorFloat-32 decorator.
      # KNOWN ISSUE: the wrapper does not return the wrapped test's return value (it returns
      # nothing).
      # Records whether TensorFloat-32 execution is enabled, disables it,
      # calls the test method with all arguments, and restores the recorded setting in all cases.
      # Side effect: toggles the global TensorFloat-32 setting during the call.
      allowed = config.tensor_float_32_execution_enabled()
      try:
        config.enable_tensor_float_32_execution(False)
        f(self, *args, **kwargs)
      finally:
        config.enable_tensor_float_32_execution(allowed)

    return decorated

  return decorator


# The description is just for documentation purposes.
def run_all_without_tensor_float_32(description):  # pylint: disable=unused-argument
  # [FUNCTION DOC]
  # HELPER: class decorator factory for tests. Input: description (documentation-only string).
  # Returns a class decorator applying the TensorFloat-32-disabling decorator to every test
  # method of a class.
  """Execute all tests in a class with TensorFloat-32 disabled."""
  return for_all_test_methods(run_without_tensor_float_32, description)


def run_v2_only(func=None):
  # [FUNCTION DOC]
  # HELPER: decorator for tests of version-2-only functionality.
  # Input: func (optional test method). With a function it returns the decorated function;
  # without, it returns the decorator. The decorated test is skipped when version-2 behavior is
  # not enabled. Raises a value error (at decoration time) if applied to a class.
  """Execute the decorated test only if running in v2 mode.

  This function is intended to be applied to tests that exercise v2 only
  functionality. If the test is run in v1 mode it will simply be skipped.

  See go/tf-test-decorator-cheatsheet for the decorators to use in different
  v1/v2/eager/graph combinations.

  Args:
    func: function to be annotated. If `func` is None, this method returns a
      decorator the can be applied to a function. If `func` is not None this
      returns the decorator applied to `func`.

  Returns:
    Returns a decorator that will conditionally skip the decorated test method.
  """

  def decorator(f):
    # [FUNCTION DOC]
    # HELPER: the decorator built by run_v2_only.
    # KNOWN ISSUE: the wrapper is not given f's name or metadata (no functools wrapping), so
    # test discovery by name may be affected.
    # Input: f, a test method. Raises a value error
    # if f is a class. Returns a wrapper that skips the test unless version-2 behavior is
    # enabled and otherwise returns f's result.
    if tf_inspect.isclass(f):
      raise ValueError('`run_v2_only` only supports test methods.')

    def decorated(self, *args, **kwargs):
      # [FUNCTION DOC]
      # HELPER: nested wrapper. Skips the test (via the test case's skip method) when version-2
      # behavior is not enabled; otherwise calls the original test method with the same
      # arguments and returns its result.
      if not tf2.enabled():
        self.skipTest('Test is only compatible with v2')

      return f(self, *args, **kwargs)

    return decorated

  if func is not None:
    return decorator(func)

  return decorator


def generate_combinations_with_testcase_name(**kwargs):
  # [FUNCTION DOC]
  # HELPER: parameterized-test helper.
  # KNOWN ISSUE: a single value that is itself a list is treated as multiple possibilities, and
  # the docstring mentions a call to a combine function that this code does not make.
  # Input: keyword arguments, each either a list of possible values or a single value (wrapped
  # into a one-item list). Keys are processed in sorted order and the Cartesian product of
  # values is built, each combination as an ordered mapping. Each gets an extra entry named
  # testcase_name made from a prefix plus key and value text with non-alphanumeric characters
  # removed. Returns the list of ordered mappings. No side effects.
  """Generate combinations based on its keyword arguments using combine().

  This function calls combine() and appends a testcase name to the list of
  dictionaries returned. The 'testcase_name' key is a required for named
  parameterized tests.

  Args:
    **kwargs: keyword arguments of form `option=[possibilities, ...]` or
      `option=the_only_possibility`.

  Returns:
    a list of dictionaries for each combination. Keys in the dictionaries are
    the keyword argument names.  Each key has one value - one of the
    corresponding keyword argument values.
  """
  sort_by_key = lambda k: k[0]
  combinations = []
  for key, values in sorted(kwargs.items(), key=sort_by_key):
    if not isinstance(values, list):
      values = [values]
    combinations.append([(key, value) for value in values])

  combinations = [collections.OrderedDict(result)
                  for result in itertools.product(*combinations)]
  named_combinations = []
  for combination in combinations:
    assert isinstance(combination, collections.OrderedDict)
    name = ''.join([
        '_{}_{}'.format(''.join(filter(str.isalnum, key)),
                        ''.join(filter(str.isalnum, str(value))))
        for key, value in combination.items()
    ])
    named_combinations.append(
        collections.OrderedDict(
            list(combination.items()) +
            [('testcase_name', '_test{}'.format(name))]))

  return named_combinations
