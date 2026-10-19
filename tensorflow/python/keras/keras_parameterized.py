# [MODULE DOC]
# Helper module for unit-testing Keras. Provides a TestCase base class (absl parameterized plus
# TensorFlow test case that clears the Keras session after each test) and decorators that run a
# test method, or every test method of a class, once per Keras saved-model format, weight format,
# model type (functional, subclass, sequential) or execution mode (v1 graph session, v2 eager,
# v2 function), built on absl named parameters and the scope helpers in testing_utils. Import
# time side effect: sets the environment variable that disables HDF5 plugin loading and tries to
# import h5py, leaving it as nothing if unavailable.
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
import functools
import itertools
import os
import unittest

from absl.testing import parameterized

from tensorflow.python import keras
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras import testing_utils
from tensorflow.python.platform import test
from tensorflow.python.util import nest

try:
  # Disable loading HDF5 plugins from a default path and prevent ZDI-CAN-25480.
  # Importing h5py prior to importing tensorflow will restore the old behavior.
  os.environ['HDF5_PLUGIN_PATH'] = 'disable'
  import h5py  # pylint:disable=g-import-not-at-top
except ImportError:
  h5py = None


class TestCase(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Base test case for Keras tests, combining the TensorFlow test case with absl parameterized
  # test support. Its only addition is that the Keras global session state is cleared after every
  # test.

  def tearDown(self):
    # [FUNCTION DOC]
    # Clears the global Keras backend session (releasing graphs, layer name counters and so on)
    # and then runs the parent tear-down. Takes no inputs and returns nothing.
    keras.backend.clear_session()
    super(TestCase, self).tearDown()


def run_with_all_saved_model_formats(
    test_or_class=None,
    exclude_formats=None):
  # [FUNCTION DOC]
  # Decorator factory that runs a test once for each Keras saved-model format (h5, tf and tf with
  # no traces) by generating absl named parameters. Inputs: test_or_class, an optional test
  # method, list of cases or test class to decorate (when nothing, a decorator is returned);
  # exclude_formats, a collection or single format name to skip. Returns the decorated object or
  # a decorator. If h5py is not installed, h5 is added to the exclusions.
  # KNOWN ISSUE: exclude_formats defaults to nothing, so appending to it raises an attribute
  # error when h5py is missing and no exclusions were given; it also mutates a caller-supplied
  # list and appends a nested list. The docstring says formats are hdf5, h5, keras, tensorflow
  # and tf, and that ImportError is raised, but the code only knows h5, tf and tf_no_traces and
  # raises no import error.
  """Execute the decorated test with all Keras saved model formats).

  This decorator is intended to be applied either to individual test methods in
  a `keras_parameterized.TestCase` class, or directly to a test class that
  extends it. Doing so will cause the contents of the individual test
  method (or all test methods in the class) to be executed multiple times - once
  for each Keras saved model format.

  The Keras saved model formats include:
  1. HDF5: 'h5'
  2. SavedModel: 'tf'

  Note: if stacking this decorator with absl.testing's parameterized decorators,
  those should be at the bottom of the stack.

  Various methods in `testing_utils` to get file path for saved models will
  auto-generate a string of the two saved model formats. This allows unittests
  to confirm the equivalence between the two Keras saved model formats.

  For example, consider the following unittest:

  ```python
  class MyTests(testing_utils.KerasTestCase):

    @testing_utils.run_with_all_saved_model_formats
    def test_foo(self):
      save_format = testing_utils.get_save_format()
      saved_model_dir = '/tmp/saved_model/'
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(3,)))
      model.add(keras.layers.Dense(3))
      model.compile(loss='mse', optimizer='sgd', metrics=['acc'])

      keras.models.save_model(model, saved_model_dir, save_format=save_format)
      model = keras.models.load_model(saved_model_dir)

  if __name__ == "__main__":
    tf.test.main()
  ```

  This test tries to save the model into the formats of 'hdf5', 'h5', 'keras',
  'tensorflow', and 'tf'.

  We can also annotate the whole class if we want this to apply to all tests in
  the class:
  ```python
  @testing_utils.run_with_all_saved_model_formats
  class MyTests(testing_utils.KerasTestCase):

    def test_foo(self):
      save_format = testing_utils.get_save_format()
      saved_model_dir = '/tmp/saved_model/'
      model = keras.models.Sequential()
      model.add(keras.layers.Dense(2, input_shape=(3,)))
      model.add(keras.layers.Dense(3))
      model.compile(loss='mse', optimizer='sgd', metrics=['acc'])

      keras.models.save_model(model, saved_model_dir, save_format=save_format)
      model = tf.keras.models.load_model(saved_model_dir)

  if __name__ == "__main__":
    tf.test.main()
  ```

  Args:
    test_or_class: test method or class to be annotated. If None,
      this method returns a decorator that can be applied to a test method or
      test class. If it is not None this returns the decorator applied to the
      test or class.
    exclude_formats: A collection of Keras saved model formats to not run.
      (May also be a single format not wrapped in a collection).
      Defaults to None.

  Returns:
    Returns a decorator that will run the decorated test method multiple times:
    once for each desired Keras saved model format.

  Raises:
    ImportError: If abseil parameterized is not installed or not included as
      a target dependency.
  """
  # Exclude h5 save format if H5py isn't available.
  if h5py is None:
    exclude_formats.append(['h5'])
  saved_model_formats = ['h5', 'tf', 'tf_no_traces']
  params = [('_%s' % saved_format, saved_format)
            for saved_format in saved_model_formats
            if saved_format not in nest.flatten(exclude_formats)]

  def single_method_decorator(f):
    # [FUNCTION DOC]
    # HELPER: builds the parameterized version of one test method. Wraps f with absl named
    # parameters for each selected saved-model format and returns the wrapper.
    """Decorator that constructs the test cases."""
    # Use named_parameters so it can be individually run from the command line
    @parameterized.named_parameters(*params)
    @functools.wraps(f)
    def decorated(self, saved_format, *args, **kwargs):
      # [FUNCTION DOC]
      # Runs the wrapped test under the saved-model format given by saved_format (h5, tf or
      # tf_no_traces) by delegating to the matching helper. Raises a value error for any other
      # format. Extra arguments are passed through to the test; returns nothing.
      """A run of a single test case w/ the specified model type."""
      if saved_format == 'h5':
        _test_h5_saved_model_format(f, self, *args, **kwargs)
      elif saved_format == 'tf':
        _test_tf_saved_model_format(f, self, *args, **kwargs)
      elif saved_format == 'tf_no_traces':
        _test_tf_saved_model_format_no_traces(f, self, *args, **kwargs)
      else:
        raise ValueError('Unknown model type: %s' % (saved_format,))
    return decorated

  return _test_or_class_decorator(test_or_class, single_method_decorator)


def _test_h5_saved_model_format(f, test_or_class, *args, **kwargs):
  # [FUNCTION DOC]
  # HELPER: runs the test function f with the test instance and extra arguments inside the h5
  # saved-model-format scope of testing_utils. Returns nothing.
  with testing_utils.saved_model_format_scope('h5'):
    f(test_or_class, *args, **kwargs)


def _test_tf_saved_model_format(f, test_or_class, *args, **kwargs):
  # [FUNCTION DOC]
  # HELPER: runs the test function f with the test instance and extra arguments inside the tf
  # (SavedModel) format scope of testing_utils. Returns nothing.
  with testing_utils.saved_model_format_scope('tf'):
    f(test_or_class, *args, **kwargs)


def _test_tf_saved_model_format_no_traces(f, test_or_class, *args, **kwargs):
  # [FUNCTION DOC]
  # HELPER: runs the test function f inside the tf saved-model format scope with saving of
  # function traces turned off. Returns nothing.
  with testing_utils.saved_model_format_scope('tf', save_traces=False):
    f(test_or_class, *args, **kwargs)


def run_with_all_weight_formats(test_or_class=None, exclude_formats=None):
  # [FUNCTION DOC]
  # Decorator factory like the saved-model-format one, but for weight saving formats: it always
  # adds the tf no-traces format to the exclusions, since that applies only to saving whole
  # models. Inputs: test_or_class (optional test, cases or class) and exclude_formats (optional
  # list; defaults to a new empty list). Returns the decorated object or a decorator.
  # KNOWN ISSUE: appends to the caller's list in place, and fails if a single string is passed
  # as exclude_formats.
  """Runs all tests with the supported formats for saving weights."""
  exclude_formats = exclude_formats or []
  exclude_formats.append('tf_no_traces')  # Only applies to saving models
  return run_with_all_saved_model_formats(test_or_class, exclude_formats)


# TODO(kaftan): Possibly enable 'subclass_custom_build' when tests begin to pass
# it. Or perhaps make 'subclass' always use a custom build method.
def run_with_all_model_types(
    test_or_class=None,
    exclude_models=None):
  # [FUNCTION DOC]
  # Decorator factory that runs a test once for each Keras model type (functional, subclass,
  # sequential) using absl named parameters. Inputs: test_or_class, an optional test method,
  # list of cases or class (nothing means a decorator is returned); exclude_models, a collection
  # or single model type to skip. Returns the decorated object or a decorator.
  # KNOWN ISSUE: the docstring says ImportError can be raised, but the code never raises it.
  """Execute the decorated test with all Keras model types.

  This decorator is intended to be applied either to individual test methods in
  a `keras_parameterized.TestCase` class, or directly to a test class that
  extends it. Doing so will cause the contents of the individual test
  method (or all test methods in the class) to be executed multiple times - once
  for each Keras model type.

  The Keras model types are: ['functional', 'subclass', 'sequential']

  Note: if stacking this decorator with absl.testing's parameterized decorators,
  those should be at the bottom of the stack.

  Various methods in `testing_utils` to get models will auto-generate a model
  of the currently active Keras model type. This allows unittests to confirm
  the equivalence between different Keras models.

  For example, consider the following unittest:

  ```python
  class MyTests(testing_utils.KerasTestCase):

    @testing_utils.run_with_all_model_types(
      exclude_models = ['sequential'])
    def test_foo(self):
      model = testing_utils.get_small_mlp(1, 4, input_dim=3)
      optimizer = RMSPropOptimizer(learning_rate=0.001)
      loss = 'mse'
      metrics = ['mae']
      model.compile(optimizer, loss, metrics=metrics)

      inputs = np.zeros((10, 3))
      targets = np.zeros((10, 4))
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)

      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)

  if __name__ == "__main__":
    tf.test.main()
  ```

  This test tries building a small mlp as both a functional model and as a
  subclass model.

  We can also annotate the whole class if we want this to apply to all tests in
  the class:
  ```python
  @testing_utils.run_with_all_model_types(exclude_models = ['sequential'])
  class MyTests(testing_utils.KerasTestCase):

    def test_foo(self):
      model = testing_utils.get_small_mlp(1, 4, input_dim=3)
      optimizer = RMSPropOptimizer(learning_rate=0.001)
      loss = 'mse'
      metrics = ['mae']
      model.compile(optimizer, loss, metrics=metrics)

      inputs = np.zeros((10, 3))
      targets = np.zeros((10, 4))
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)

      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)

  if __name__ == "__main__":
    tf.test.main()
  ```


  Args:
    test_or_class: test method or class to be annotated. If None,
      this method returns a decorator that can be applied to a test method or
      test class. If it is not None this returns the decorator applied to the
      test or class.
    exclude_models: A collection of Keras model types to not run.
      (May also be a single model type not wrapped in a collection).
      Defaults to None.

  Returns:
    Returns a decorator that will run the decorated test method multiple times:
    once for each desired Keras model type.

  Raises:
    ImportError: If abseil parameterized is not installed or not included as
      a target dependency.
  """
  model_types = ['functional', 'subclass', 'sequential']
  params = [('_%s' % model, model) for model in model_types
            if model not in nest.flatten(exclude_models)]

  def single_method_decorator(f):
    # [FUNCTION DOC]
    # HELPER: builds the parameterized version of one test method, wrapping f with named
    # parameters for each selected model type. Returns the wrapper.
    """Decorator that constructs the test cases."""
    # Use named_parameters so it can be individually run from the command line
    @parameterized.named_parameters(*params)
    @functools.wraps(f)
    def decorated(self, model_type, *args, **kwargs):
      # [FUNCTION DOC]
      # Runs the wrapped test under the model type named by model_type (functional, subclass or
      # sequential) via the matching helper. Raises a value error for an unknown type. Extra
      # arguments pass through to the test; returns nothing.
      """A run of a single test case w/ the specified model type."""
      if model_type == 'functional':
        _test_functional_model_type(f, self, *args, **kwargs)
      elif model_type == 'subclass':
        _test_subclass_model_type(f, self, *args, **kwargs)
      elif model_type == 'sequential':
        _test_sequential_model_type(f, self, *args, **kwargs)
      else:
        raise ValueError('Unknown model type: %s' % (model_type,))
    return decorated

  return _test_or_class_decorator(test_or_class, single_method_decorator)


def _test_functional_model_type(f, test_or_class, *args, **kwargs):
  # [FUNCTION DOC]
  # HELPER: runs test function f inside the functional model-type scope of testing_utils.
  # Returns nothing.
  with testing_utils.model_type_scope('functional'):
    f(test_or_class, *args, **kwargs)


def _test_subclass_model_type(f, test_or_class, *args, **kwargs):
  # [FUNCTION DOC]
  # HELPER: runs test function f inside the subclass model-type scope of testing_utils.
  # Returns nothing.
  with testing_utils.model_type_scope('subclass'):
    f(test_or_class, *args, **kwargs)


def _test_sequential_model_type(f, test_or_class, *args, **kwargs):
  # [FUNCTION DOC]
  # HELPER: runs test function f inside the sequential model-type scope of testing_utils.
  # Returns nothing.
  with testing_utils.model_type_scope('sequential'):
    f(test_or_class, *args, **kwargs)


def run_all_keras_modes(test_or_class=None,
                        config=None,
                        always_skip_v1=False,
                        always_skip_eager=False,
                        **kwargs):
  # [FUNCTION DOC]
  # Decorator factory that runs a test in each Keras execution mode: v2 function (eager context,
  # run-eagerly false), v2 eager (run-eagerly true) and, unless TF2 behavior is enabled or
  # always_skip_v1 is set, a v1 graph session. Inputs: test_or_class (optional test, cases or
  # class), config (optional session configuration for the v1 mode), always_skip_v1 and
  # always_skip_eager (booleans, default false), and extra keyword arguments which are not
  # allowed. Raises a value error if any extra keyword argument is given. Returns the decorated
  # object or a decorator.
  # KNOWN ISSUE: the docstring says it raises ImportError, which never happens. The decorated
  # wrapper returns, rather than raises, a value error for an unknown mode (unreachable in
  # practice).
  """Execute the decorated test with all keras execution modes.

  This decorator is intended to be applied either to individual test methods in
  a `keras_parameterized.TestCase` class, or directly to a test class that
  extends it. Doing so will cause the contents of the individual test
  method (or all test methods in the class) to be executed multiple times -
  once executing in legacy graph mode, once running eagerly and with
  `should_run_eagerly` returning True, and once running eagerly with
  `should_run_eagerly` returning False.

  If Tensorflow v2 behavior is enabled, legacy graph mode will be skipped, and
  the test will only run twice.

  Note: if stacking this decorator with absl.testing's parameterized decorators,
  those should be at the bottom of the stack.

  For example, consider the following unittest:

  ```python
  class MyTests(testing_utils.KerasTestCase):

    @testing_utils.run_all_keras_modes
    def test_foo(self):
      model = testing_utils.get_small_functional_mlp(1, 4, input_dim=3)
      optimizer = RMSPropOptimizer(learning_rate=0.001)
      loss = 'mse'
      metrics = ['mae']
      model.compile(
          optimizer, loss, metrics=metrics,
          run_eagerly=testing_utils.should_run_eagerly())

      inputs = np.zeros((10, 3))
      targets = np.zeros((10, 4))
      dataset = dataset_ops.Dataset.from_tensor_slices((inputs, targets))
      dataset = dataset.repeat(100)
      dataset = dataset.batch(10)

      model.fit(dataset, epochs=1, steps_per_epoch=2, verbose=1)

  if __name__ == "__main__":
    tf.test.main()
  ```

  This test will try compiling & fitting the small functional mlp using all
  three Keras execution modes.

  Args:
    test_or_class: test method or class to be annotated. If None,
      this method returns a decorator that can be applied to a test method or
      test class. If it is not None this returns the decorator applied to the
      test or class.
    config: An optional config_pb2.ConfigProto to use to configure the
      session when executing graphs.
    always_skip_v1: If True, does not try running the legacy graph mode even
      when Tensorflow v2 behavior is not enabled.
    always_skip_eager: If True, does not execute the decorated test
      with eager execution modes.
    **kwargs: Additional kwargs for configuring tests for
     in-progress Keras behaviors/ refactorings that we haven't fully
     rolled out yet

  Returns:
    Returns a decorator that will run the decorated test method multiple times.

  Raises:
    ImportError: If abseil parameterized is not installed or not included as
      a target dependency.
  """
  if kwargs:
    raise ValueError('Unrecognized keyword args: {}'.format(kwargs))

  params = [('_v2_function', 'v2_function')]
  if not always_skip_eager:
    params.append(('_v2_eager', 'v2_eager'))
  if not (always_skip_v1 or tf2.enabled()):
    params.append(('_v1_session', 'v1_session'))

  def single_method_decorator(f):
    # [FUNCTION DOC]
    # HELPER: builds the parameterized version of one test method, wrapping f with named
    # parameters for each selected execution mode. Returns the wrapper.
    """Decorator that constructs the test cases."""

    # Use named_parameters so it can be individually run from the command line
    @parameterized.named_parameters(*params)
    @functools.wraps(f)
    def decorated(self, run_mode, *args, **kwargs):
      # [FUNCTION DOC]
      # Runs the wrapped test in the mode named by run_mode (v1_session, v2_eager or
      # v2_function) via the matching helper. KNOWN ISSUE: for an unknown mode it returns a value
      # error object instead of raising it, so the failure is silent. Returns nothing otherwise.
      """A run of a single test case w/ specified run mode."""
      if run_mode == 'v1_session':
        _v1_session_test(f, self, config, *args, **kwargs)
      elif run_mode == 'v2_eager':
        _v2_eager_test(f, self, *args, **kwargs)
      elif run_mode == 'v2_function':
        _v2_function_test(f, self, *args, **kwargs)
      else:
        return ValueError('Unknown run mode %s' % run_mode)

    return decorated

  return _test_or_class_decorator(test_or_class, single_method_decorator)


def _v1_session_test(f, test_or_class, config, *args, **kwargs):
  # [FUNCTION DOC]
  # HELPER: runs test function f in legacy graph mode: inside the default graph, with
  # run-eagerly off, and inside the test case's session created with the given config. Returns
  # nothing.
  with ops.get_default_graph().as_default():
    with testing_utils.run_eagerly_scope(False):
      with test_or_class.test_session(config=config):
        f(test_or_class, *args, **kwargs)


def _v2_eager_test(f, test_or_class, *args, **kwargs):
  # [FUNCTION DOC]
  # HELPER: runs test function f in an eager context with the run-eagerly scope set to true.
  # Returns nothing.
  with context.eager_mode():
    with testing_utils.run_eagerly_scope(True):
      f(test_or_class, *args, **kwargs)


def _v2_function_test(f, test_or_class, *args, **kwargs):
  # [FUNCTION DOC]
  # HELPER: runs test function f in an eager context with the run-eagerly scope set to false, so
  # models are compiled into functions. Returns nothing.
  with context.eager_mode():
    with testing_utils.run_eagerly_scope(False):
      f(test_or_class, *args, **kwargs)


def _test_or_class_decorator(test_or_class, single_method_decorator):
  # [FUNCTION DOC]
  # HELPER: shared glue for the decorators in this module. Inputs: test_or_class, a test method,
  # an iterable of already parameterized cases, a test class, or nothing; single_method_decorator,
  # a function that decorates one method. A class has the decorator applied to each test method
  # (mutating the class) and a copied class is returned; an iterable has the decorator applied to
  # each element with the results flattened; a single method is decorated directly. When
  # test_or_class is nothing, the inner decorating function is returned for later use.
  """Decorate a test or class with a decorator intended for one method.

  If the test_or_class is a class:
    This will apply the decorator to all test methods in the class.

  If the test_or_class is an iterable of already-parameterized test cases:
    This will apply the decorator to all the cases, and then flatten the
    resulting cross-product of test cases. This allows stacking the Keras
    parameterized decorators w/ each other, and to apply them to test methods
    that have already been marked with an absl parameterized decorator.

  Otherwise, treat the obj as a single method and apply the decorator directly.

  Args:
    test_or_class: A test method (that may have already been decorated with a
      parameterized decorator, or a test class that extends
      keras_parameterized.TestCase
    single_method_decorator:
      A parameterized decorator intended for a single test method.
  Returns:
    The decorated result.
  """
  def _decorate_test_or_class(obj):
    # [FUNCTION DOC]
    # HELPER: applies the single-method decorator to obj, which may be an iterable of cases, a
    # class, or a single method, as described for the enclosing function. For a class it replaces
    # each test method in place and returns a freshly created copy of the class.
    # KNOWN ISSUE: strings or other iterables are also treated as case collections, and the class
    # is both mutated and copied.
    if isinstance(obj, collections.abc.Iterable):
      return itertools.chain.from_iterable(
          single_method_decorator(method) for method in obj)
    if isinstance(obj, type):
      cls = obj
      for name, value in cls.__dict__.copy().items():
        if callable(value) and name.startswith(
            unittest.TestLoader.testMethodPrefix):
          setattr(cls, name, single_method_decorator(value))

      cls = type(cls).__new__(type(cls), cls.__name__, cls.__bases__,
                              cls.__dict__.copy())
      return cls

    return single_method_decorator(obj)

  if test_or_class is not None:
    return _decorate_test_or_class(test_or_class)

  return _decorate_test_or_class
