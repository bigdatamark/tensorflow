# [MODULE DOC]
# Unit tests for the legacy (v1) feature column implementation in the feature_column module, plus
# a few cases that use the newer v2 shared embedding column factory. Each test class covers one
# column type (numeric, bucketized, hashed categorical, crossed, vocabulary file, vocabulary list,
# identity categorical, indicator, embedding, shared embedding, weighted categorical) or one
# consumer (lazy feature builder, linear model function, the Keras-style linear model class, the
# input layer class and function, parse example spec creation, feature transformation ordering).
# Tests check construction-time validation errors, parse specs, deep copying, id transformation,
# dense and sparse tensor extraction and numeric results of linear models and input layers. Most
# tests build a fresh graph and evaluate through a v1 session, because the v1 columns rely on
# variable scopes and collections. The module defines small shared helpers (a session initializer,
# accessors for linear model variables, a wrapper that builds the Keras-style linear model, and a
# sparse tensor value comparator). It has no import-time side effects beyond imports and runs the
# standard test main when executed as a script.
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
"""Tests for feature_column."""

import collections
import copy

from absl.testing import parameterized
import numpy as np

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.feature_column import feature_column as fc
from tensorflow.python.feature_column import feature_column_v2 as fc_new
from tensorflow.python.feature_column.feature_column import _CategoricalColumn
from tensorflow.python.feature_column.feature_column import _DenseColumn
from tensorflow.python.feature_column.feature_column import _FeatureColumn
from tensorflow.python.feature_column.feature_column import _LazyBuilder
from tensorflow.python.feature_column.feature_column import _LinearModel
from tensorflow.python.feature_column.feature_column import _transform_features
from tensorflow.python.feature_column.feature_column import InputLayer
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import partitioned_variables
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.platform import test


def _initialized_session(config=None):
  # [FUNCTION DOC]
  # HELPER: shared by almost every graph-mode test in this module to evaluate tensors.
  # Creates a v1 session, optionally using the given session configuration (default nothing, meaning
  # default settings), then runs the global variable initializer and the lookup table initializer in
  # the default graph so variables and vocabulary tables are ready. Returns the new session, which
  # callers typically use as a context manager. Side effects: creates a session bound to the current
  # default graph and mutates variable and table state by initializing it.
  sess = session.Session(config=config)
  sess.run(variables_lib.global_variables_initializer())
  sess.run(lookup_ops.tables_initializer())
  return sess


class LazyColumnTest(test.TestCase):
  # [CLASS DOC]
  # Tests the lazy feature builder that transforms raw feature inputs into column outputs on demand
  # and caches the result. Covers caching of transform results, no pollution of the caller supplied
  # features mapping, and error handling for unknown keys, unsupported columns and wrong key types.
  # Several tests define tiny local feature column subclasses as fixtures.

  def test_transformations_called_once(self):
    # [FUNCTION DOC]
    # Scenario: a column whose transform counts how many times it runs. A lazy builder is created over
    # a small mapping with one feature, the column is requested twice, and the test asserts the
    # transform ran zero times before the first request and exactly once after both requests, proving
    # the result is cached.

    class TransformCounter(_FeatureColumn):
      # [CLASS DOC]
      # Local fixture column derived from the legacy feature column base class. It counts how many times
      # its transform is invoked and returns the raw feature named a. Used only by the caching test.

      def __init__(self):
        # [FUNCTION DOC]
        # Constructor of the fixture column; initializes the transform call counter to zero. Takes no
        # arguments and returns nothing.
        self.num_transform = 0

      @property
      def name(self):
        # [FUNCTION DOC]
        # Property returning the fixed column name used by the fixture. Takes no arguments; returns a
        # string.
        return 'TransformCounter'

      def _transform_feature(self, cache):
        # [FUNCTION DOC]
        # Transform hook of the fixture: increments the call counter, then returns the raw feature named a
        # fetched from the given cache (the lazy builder). Returns the fetched feature; side effect is the
        # counter increment.
        self.num_transform += 1  # Count transform calls.
        return cache.get('a')

      @property
      def _parse_example_spec(self):
        # [FUNCTION DOC]
        # Property required by the column interface; the fixture does nothing and returns nothing, since
        # parsing is never exercised in this test.
        pass

    builder = _LazyBuilder(features={'a': [[2], [3.]]})
    column = TransformCounter()
    self.assertEqual(0, column.num_transform)
    builder.get(column)
    self.assertEqual(1, column.num_transform)
    builder.get(column)
    self.assertEqual(1, column.num_transform)

  def test_returns_transform_output(self):
    # [FUNCTION DOC]
    # Scenario: a column whose transform returns a constant string. Asserts that requesting the column
    # from the lazy builder returns that string, and that requesting it a second time returns the same
    # value.

    class Transformer(_FeatureColumn):
      # [CLASS DOC]
      # Local fixture column whose transform always returns a fixed string. Used to check that the
      # builder returns transform output unchanged.

      @property
      def name(self):
        # [FUNCTION DOC]
        # Property returning the fixed fixture column name; no arguments, returns a string.
        return 'Transformer'

      def _transform_feature(self, cache):
        # [FUNCTION DOC]
        # Transform hook of the fixture; ignores the given cache and returns a constant string.
        return 'Output'

      @property
      def _parse_example_spec(self):
        # [FUNCTION DOC]
        # Property required by the column interface; the fixture returns nothing because parsing is not
        # exercised.
        pass

    builder = _LazyBuilder(features={'a': [[2], [3.]]})
    column = Transformer()
    self.assertEqual('Output', builder.get(column))
    self.assertEqual('Output', builder.get(column))

  def test_does_not_pollute_given_features_dict(self):
    # [FUNCTION DOC]
    # Scenario: builds a lazy builder over a features mapping holding only the key a, requests a
    # fixture column, and asserts that the original mapping still contains only the key a, i.e. the
    # builder caches into its own storage rather than mutating the caller input.

    class Transformer(_FeatureColumn):
      # [CLASS DOC]
      # Local fixture column that returns a constant string from its transform; used to trigger a cache
      # write without touching the input mapping.

      @property
      def name(self):
        # [FUNCTION DOC]
        # Property returning the fixed fixture column name; returns a string.
        return 'Transformer'

      def _transform_feature(self, cache):
        # [FUNCTION DOC]
        # Transform hook of the fixture; ignores the cache and returns a constant string.
        return 'Output'

      @property
      def _parse_example_spec(self):
        # [FUNCTION DOC]
        # Property required by the column interface; returns nothing as parsing is not exercised.
        pass

    features = {'a': [[2], [3.]]}
    builder = _LazyBuilder(features=features)
    builder.get(Transformer())
    self.assertEqual(['a'], list(features.keys()))

  def test_error_if_feature_is_not_found(self):
    # [FUNCTION DOC]
    # Scenario: requesting a missing string key, both as a plain string and as a text string, from a
    # lazy builder that only knows the key a. Asserts that a value error saying the key is not in the
    # features dictionary is raised each time.
    builder = _LazyBuilder(features={'a': [[2], [3.]]})
    with self.assertRaisesRegex(ValueError,
                                'bbb is not in features dictionary'):
      builder.get('bbb')
    with self.assertRaisesRegex(ValueError,
                                'bbb is not in features dictionary'):
      builder.get(u'bbb')

  def test_not_supported_feature_column(self):
    # [FUNCTION DOC]
    # Scenario: a fixture column whose transform returns nothing. Asserts that asking the lazy builder
    # for it raises a value error saying the column is not supported, because transforms must return a
    # non-null result.

    class NotAProperColumn(_FeatureColumn):
      # [CLASS DOC]
      # Local fixture column whose transform returns nothing, which the builder must reject.

      @property
      def name(self):
        # [FUNCTION DOC]
        # Property returning the fixed fixture column name; returns a string.
        return 'NotAProperColumn'

      def _transform_feature(self, cache):
        # [FUNCTION DOC]
        # Transform hook of the fixture that deliberately returns nothing (a null result), which is the
        # invalid behavior under test.
        # It should return not None.
        pass

      @property
      def _parse_example_spec(self):
        # [FUNCTION DOC]
        # Property required by the column interface; returns nothing as parsing is not exercised.
        pass

    builder = _LazyBuilder(features={'a': [[2], [3.]]})
    with self.assertRaisesRegex(ValueError,
                                'NotAProperColumn is not supported'):
      builder.get(NotAProperColumn())

  def test_key_should_be_string_or_feature_colum(self):
    # [FUNCTION DOC]
    # Scenario: requesting an object that is neither a string nor a feature column from the lazy
    # builder. Asserts that a type error stating the key must be a string or a feature column is
    # raised.

    class NotAFeatureColumn(object):
      # [CLASS DOC]
      # Empty local fixture class, a plain object that is not a feature column, used as an invalid key.
      pass

    builder = _LazyBuilder(features={'a': [[2], [3.]]})
    with self.assertRaisesRegex(
        TypeError, '"key" must be either a "str" or "_FeatureColumn".'):
      builder.get(NotAFeatureColumn())

  def test_expand_dim_rank_1_sparse_tensor_empty_batch(self):
    # [FUNCTION DOC]
    # Scenario: the builder is given a rank one sparse tensor with zero elements and a batch of zero.
    # Fetching it through the builder should expand it to rank two; the test asserts the dense shape
    # becomes zero by one and the indices array has shape zero by two. Evaluated inside a cached test
    # session.
    # empty 1-D sparse tensor:
    builder = _LazyBuilder(features={'a': sparse_tensor.SparseTensor(
        indices=np.reshape(np.array([], dtype=np.int64), (0, 1)),
        dense_shape=[0],
        values=np.array([]))})
    with self.cached_session():
      spv = builder.get('a')
      self.assertAllEqual(np.array([0, 1], dtype=np.int64), spv.dense_shape)
      self.assertAllEqual(
          np.reshape(np.array([], dtype=np.int64), (0, 2)), spv.indices)


class NumericColumnTest(test.TestCase):
  # [CLASS DOC]
  # Tests the legacy numeric column: defaults, argument validation, parse specification, normalizer
  # function handling, dense tensor output, deep copy and use in the linear model function and the
  # Keras-style linear model.

  def test_defaults(self):
    # [FUNCTION DOC]
    # Creates a numeric column with only a key and asserts the defaults: name, variable scope name and
    # key equal the key, shape is a single element, no default value, float32 dtype and no normalizer
    # function.
    a = fc._numeric_column('aaa')
    self.assertEqual('aaa', a.key)
    self.assertEqual('aaa', a.name)
    self.assertEqual('aaa', a._var_scope_name)
    self.assertEqual((1,), a.shape)
    self.assertIsNone(a.default_value)
    self.assertEqual(dtypes.float32, a.dtype)
    self.assertIsNone(a.normalizer_fn)

  def test_key_should_be_string(self):
    # [FUNCTION DOC]
    # Asserts that passing a non-string key (a tuple) raises a value error saying the key must be a
    # string.
    with self.assertRaisesRegex(ValueError, 'key must be a string.'):
      fc._numeric_column(key=('aaa',))

  def test_shape_saved_as_tuple(self):
    # [FUNCTION DOC]
    # Asserts that a shape given as a list is stored on the column as a tuple of integers.
    a = fc._numeric_column('aaa', shape=[1, 2], default_value=[[3, 2.]])
    self.assertEqual((1, 2), a.shape)

  def test_default_value_saved_as_tuple(self):
    # [FUNCTION DOC]
    # Asserts that a scalar default value and a nested list default value are both stored as nested
    # tuples of floats on the column.
    a = fc._numeric_column('aaa', default_value=4.)
    self.assertEqual((4.,), a.default_value)
    a = fc._numeric_column('aaa', shape=[1, 2], default_value=[[3, 2.]])
    self.assertEqual(((3., 2.),), a.default_value)

  def test_shape_and_default_value_compatibility(self):
    # [FUNCTION DOC]
    # Asserts that a default value whose shape is compatible with the column shape is accepted, while
    # default values with a mismatching shape raise a value error mentioning the shape of
    # default_value. Exercises one-dimensional and two-dimensional shapes.
    fc._numeric_column('aaa', shape=[2], default_value=[1, 2.])
    with self.assertRaisesRegex(ValueError, 'The shape of default_value'):
      fc._numeric_column('aaa', shape=[2], default_value=[1, 2, 3.])
    fc._numeric_column(
        'aaa', shape=[3, 2], default_value=[[2, 3], [1, 2], [2, 3.]])
    with self.assertRaisesRegex(ValueError, 'The shape of default_value'):
      fc._numeric_column(
          'aaa', shape=[3, 1], default_value=[[2, 3], [1, 2], [2, 3.]])
    with self.assertRaisesRegex(ValueError, 'The shape of default_value'):
      fc._numeric_column(
          'aaa', shape=[3, 3], default_value=[[2, 3], [1, 2], [2, 3.]])

  def test_default_value_type_check(self):
    # [FUNCTION DOC]
    # Asserts that default values compatible with the dtype (floats for float32, integers for int32)
    # are accepted, while a float default for an integer dtype and a string default raise a type error
    # saying the default value must be compatible with the dtype.
    fc._numeric_column(
        'aaa', shape=[2], default_value=[1, 2.], dtype=dtypes.float32)
    fc._numeric_column(
        'aaa', shape=[2], default_value=[1, 2], dtype=dtypes.int32)
    with self.assertRaisesRegex(TypeError, 'must be compatible with dtype'):
      fc._numeric_column(
          'aaa', shape=[2], default_value=[1, 2.], dtype=dtypes.int32)
    with self.assertRaisesRegex(TypeError,
                                'default_value must be compatible with dtype'):
      fc._numeric_column('aaa', default_value=['string'])

  def test_shape_must_be_positive_integer(self):
    # [FUNCTION DOC]
    # Asserts that a non-integer shape dimension raises a type error and a zero shape dimension raises
    # a value error saying dimensions must be greater than zero.
    with self.assertRaisesRegex(TypeError, 'shape dimensions must be integer'):
      fc._numeric_column(
          'aaa', shape=[
              1.0,
          ])

    with self.assertRaisesRegex(ValueError,
                                'shape dimensions must be greater than 0'):
      fc._numeric_column(
          'aaa', shape=[
              0,
          ])

  def test_dtype_is_convertible_to_float(self):
    # [FUNCTION DOC]
    # Asserts that creating a numeric column with a string dtype raises a value error saying the dtype
    # must be convertible to float.
    with self.assertRaisesRegex(ValueError,
                                'dtype must be convertible to float'):
      fc._numeric_column('aaa', dtype=dtypes.string)

  def test_scalar_default_value_fills_the_shape(self):
    # [FUNCTION DOC]
    # Asserts that a scalar default value is broadcast to fill the whole two-by-three column shape and
    # is stored as nested tuples.
    a = fc._numeric_column('aaa', shape=[2, 3], default_value=2.)
    self.assertEqual(((2., 2., 2.), (2., 2., 2.)), a.default_value)

  def test_parse_spec(self):
    # [FUNCTION DOC]
    # Asserts that the parse example spec of a numeric column with a two-by-three int32 shape is a
    # mapping from the key to a fixed length feature with that shape and dtype.
    a = fc._numeric_column('aaa', shape=[2, 3], dtype=dtypes.int32)
    self.assertEqual({
        'aaa': parsing_ops.FixedLenFeature((2, 3), dtype=dtypes.int32)
    }, a._parse_example_spec)

  def test_parse_example_no_default_value(self):
    # [FUNCTION DOC]
    # Scenario: serializes one example holding two price values, parses it using the spec derived from
    # a numeric column of shape two with no default, and asserts the parsed price equals the stored
    # values.
    price = fc._numeric_column('price', shape=[2])
    data = example_pb2.Example(features=feature_pb2.Features(
        feature={
            'price':
                feature_pb2.Feature(float_list=feature_pb2.FloatList(
                    value=[20., 110.]))
        }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec([price]))
    self.assertIn('price', features)
    with self.cached_session():
      self.assertAllEqual([[20., 110.]], features['price'])

  def test_parse_example_with_default_value(self):
    # [FUNCTION DOC]
    # Scenario: parses two serialized examples, one holding price values and one lacking the price
    # feature, using a numeric column with default value eleven. Asserts the first row holds the
    # stored values and the second row is filled with the default.
    price = fc._numeric_column('price', shape=[2], default_value=11.)
    data = example_pb2.Example(features=feature_pb2.Features(
        feature={
            'price':
                feature_pb2.Feature(float_list=feature_pb2.FloatList(
                    value=[20., 110.]))
        }))
    no_data = example_pb2.Example(features=feature_pb2.Features(
        feature={
            'something_else':
                feature_pb2.Feature(float_list=feature_pb2.FloatList(
                    value=[20., 110.]))
        }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString(),
                    no_data.SerializeToString()],
        features=fc.make_parse_example_spec([price]))
    self.assertIn('price', features)
    self.assertAllEqual([[20., 110.], [11., 11.]], features['price'])

  def test_normalizer_fn_must_be_callable(self):
    # [FUNCTION DOC]
    # Asserts that passing a non-callable normalizer function raises a type error saying it must be
    # callable.
    with self.assertRaisesRegex(TypeError, 'must be a callable'):
      fc._numeric_column('price', normalizer_fn='NotACallable')

  def test_normalizer_fn_transform_feature(self):
    # [FUNCTION DOC]
    # Scenario: a numeric column with a normalizer that adds two. Transforms a small price feature
    # through the module transform function and asserts every value is increased by two.

    def _increment_two(input_tensor):
      # [FUNCTION DOC]
      # Local normalizer used by the test; takes a tensor and returns it with two added to every
      # element.
      return input_tensor + 2.

    price = fc._numeric_column('price', shape=[2], normalizer_fn=_increment_two)
    output = _transform_features({'price': [[1., 2.], [5., 6.]]}, [price])
    with self.cached_session():
      self.assertAllEqual([[3., 4.], [7., 8.]], output[price])

  def test_get_dense_tensor(self):
    # [FUNCTION DOC]
    # Scenario: a numeric column with a normalizer adding two. Asserts that the dense tensor obtained
    # directly from the column equals the tensor fetched from the lazy builder, both being the
    # normalized values.

    def _increment_two(input_tensor):
      # [FUNCTION DOC]
      # Local normalizer used by the test; takes a tensor and returns it with two added to every
      # element.
      return input_tensor + 2.

    price = fc._numeric_column('price', shape=[2], normalizer_fn=_increment_two)
    builder = _LazyBuilder({'price': [[1., 2.], [5., 6.]]})
    self.assertAllClose(builder.get(price), price._get_dense_tensor(builder))

  def test_sparse_tensor_not_supported(self):
    # [FUNCTION DOC]
    # Asserts that transforming a sparse tensor input with a numeric column raises a value error
    # saying the input must be a dense tensor.
    price = fc._numeric_column('price')
    builder = _LazyBuilder({
        'price':
            sparse_tensor.SparseTensor(
                indices=[[0, 0]], values=[0.3], dense_shape=[1, 1])
    })
    with self.assertRaisesRegex(ValueError, 'must be a Tensor'):
      price._transform_feature(builder)

  def test_deep_copy(self):
    # [FUNCTION DOC]
    # Deep-copies a numeric column with a shape and nested default value and asserts the copy keeps
    # the name, shape and default value.
    a = fc._numeric_column('aaa', shape=[1, 2], default_value=[[3., 2.]])
    a_copy = copy.deepcopy(a)
    self.assertEqual(a_copy.name, 'aaa')
    self.assertEqual(a_copy.shape, (1, 2))
    self.assertEqual(a_copy.default_value, ((3., 2.),))

  def test_numpy_default_value(self):
    # [FUNCTION DOC]
    # Asserts that a numpy array default value is converted to nested tuples of floats on the column.
    a = fc._numeric_column(
        'aaa', shape=[1, 2], default_value=np.array([[3., 2.]]))
    self.assertEqual(a.default_value, ((3., 2.),))

  def test_linear_model(self):
    # [FUNCTION DOC]
    # Scenario: graph with one numeric price column and two examples run through the linear model
    # function. Asserts bias and weight start at zero and predictions are zero, then assigns the
    # weight ten and asserts predictions equal ten times the price.
    price = fc._numeric_column('price')
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      predictions = fc.linear_model(features, [price])
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        self.assertAllClose([[0.]], self.evaluate(price_var))
        self.assertAllClose([[0.], [0.]], self.evaluate(predictions))
        sess.run(price_var.assign([[10.]]))
        self.assertAllClose([[10.], [50.]], self.evaluate(predictions))

  def test_keras_linear_model(self):
    # [FUNCTION DOC]
    # Same scenario as the linear model test but built through the Keras-style linear model helper:
    # initial zeros, then weight ten gives predictions ten and fifty.
    price = fc._numeric_column('price')
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      predictions = get_keras_linear_model_predictions(features, [price])
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        self.assertAllClose([[0.]], self.evaluate(price_var))
        self.assertAllClose([[0.], [0.]], self.evaluate(predictions))
        sess.run(price_var.assign([[10.]]))
        self.assertAllClose([[10.], [50.]], self.evaluate(predictions))


class BucketizedColumnTest(test.TestCase):
  # [CLASS DOC]
  # Tests the legacy bucketized column: constructor validation, name and shape properties, parsing,
  # transformation into bucket ids, dense one-hot and sparse tensor outputs, deep copy and linear
  # model behavior for source shapes of one and two.

  def test_invalid_source_column_type(self):
    # [FUNCTION DOC]
    # Asserts that creating a bucketized column from a non-numeric source column (a hashed categorical
    # column) raises a value error saying the source must come from a numeric column.
    a = fc._categorical_column_with_hash_bucket('aaa', hash_bucket_size=10)
    with self.assertRaisesRegex(
        ValueError,
        'source_column must be a column generated with numeric_column'):
      fc._bucketized_column(a, boundaries=[0, 1])

  def test_invalid_source_column_shape(self):
    # [FUNCTION DOC]
    # Asserts that a source numeric column with a two-dimensional shape raises a value error saying
    # the source must be a one-dimensional column.
    a = fc._numeric_column('aaa', shape=[2, 3])
    with self.assertRaisesRegex(ValueError,
                                'source_column must be one-dimensional column'):
      fc._bucketized_column(a, boundaries=[0, 1])

  def test_invalid_boundaries(self):
    # [FUNCTION DOC]
    # Asserts that boundaries that are missing, not a list, unsorted or containing duplicates raise a
    # value error saying boundaries must be a sorted list.
    a = fc._numeric_column('aaa')
    with self.assertRaisesRegex(ValueError, 'boundaries must be a sorted list'):
      fc._bucketized_column(a, boundaries=None)
    with self.assertRaisesRegex(ValueError, 'boundaries must be a sorted list'):
      fc._bucketized_column(a, boundaries=1.)
    with self.assertRaisesRegex(ValueError, 'boundaries must be a sorted list'):
      fc._bucketized_column(a, boundaries=[1, 0])
    with self.assertRaisesRegex(ValueError, 'boundaries must be a sorted list'):
      fc._bucketized_column(a, boundaries=[1, 1])

  def test_name(self):
    # [FUNCTION DOC]
    # Asserts the bucketized column name is the source name with a bucketized suffix.
    a = fc._numeric_column('aaa', dtype=dtypes.int32)
    b = fc._bucketized_column(a, boundaries=[0, 1])
    self.assertEqual('aaa_bucketized', b.name)

  def test_var_scope_name(self):
    # [FUNCTION DOC]
    # Asserts the variable scope name equals the source name with a bucketized suffix.
    a = fc._numeric_column('aaa', dtype=dtypes.int32)
    b = fc._bucketized_column(a, boundaries=[0, 1])
    self.assertEqual('aaa_bucketized', b._var_scope_name)

  def test_parse_spec(self):
    # [FUNCTION DOC]
    # Asserts that the parse spec of a bucketized column is the same fixed length feature spec as its
    # source numeric column.
    a = fc._numeric_column('aaa', shape=[2], dtype=dtypes.int32)
    b = fc._bucketized_column(a, boundaries=[0, 1])
    self.assertEqual({
        'aaa': parsing_ops.FixedLenFeature((2,), dtype=dtypes.int32)
    }, b._parse_example_spec)

  def test_variable_shape(self):
    # [FUNCTION DOC]
    # Asserts that a source of shape two with two boundaries (three buckets) yields a variable shape
    # of two by three.
    a = fc._numeric_column('aaa', shape=[2], dtype=dtypes.int32)
    b = fc._bucketized_column(a, boundaries=[0, 1])
    # Column 'aaa` has shape [2] times three buckets -> variable_shape=[2, 3].
    self.assertAllEqual((2, 3), b._variable_shape)

  def test_num_buckets(self):
    # [FUNCTION DOC]
    # Asserts the total number of buckets equals the source width times the bucket count, six for
    # width two and three buckets.
    a = fc._numeric_column('aaa', shape=[2], dtype=dtypes.int32)
    b = fc._bucketized_column(a, boundaries=[0, 1])
    # Column 'aaa` has shape [2] times three buckets -> num_buckets=6.
    self.assertEqual(6, b._num_buckets)

  def test_parse_example(self):
    # [FUNCTION DOC]
    # Parses a serialized example containing two price values with a spec created from a bucketized
    # column and asserts the raw price values come back, since parsing is done on the source feature.
    price = fc._numeric_column('price', shape=[2])
    bucketized_price = fc._bucketized_column(price, boundaries=[0, 50])
    data = example_pb2.Example(features=feature_pb2.Features(
        feature={
            'price':
                feature_pb2.Feature(float_list=feature_pb2.FloatList(
                    value=[20., 110.]))
        }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec([bucketized_price]))
    self.assertIn('price', features)
    self.assertAllEqual([[20., 110.]], features['price'])

  def test_transform_feature(self):
    # [FUNCTION DOC]
    # Transforms a two-wide price feature with boundaries zero, two, four and six, and asserts the
    # output is the bucket index of every value.
    price = fc._numeric_column('price', shape=[2])
    bucketized_price = fc._bucketized_column(price, boundaries=[0, 2, 4, 6])
    transformed_tensor = _transform_features({'price': [[-1., 1.], [5., 6.]]},
                                             [bucketized_price])
    self.assertAllClose([[0, 1], [3, 4]], transformed_tensor[bucketized_price])

  def test_get_dense_tensor_one_input_value(self):
    # [FUNCTION DOC]
    # Scenario: one price value per example. In a fresh graph, asserts the dense tensor from the
    # bucketized column is the expected one-hot encoding with five buckets.
    """Tests _get_dense_tensor() for input with shape=[1]."""
    price = fc._numeric_column('price', shape=[1])
    bucketized_price = fc._bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      builder = _LazyBuilder({'price': [[-1.], [1.], [5.], [6.]]})
      with _initialized_session():
        bucketized_price_tensor = bucketized_price._get_dense_tensor(builder)
        self.assertAllClose(
            # One-hot tensor.
            [[[1., 0., 0., 0., 0.]], [[0., 1., 0., 0., 0.]],
             [[0., 0., 0., 1., 0.]], [[0., 0., 0., 0., 1.]]],
            self.evaluate(bucketized_price_tensor))

  def test_get_dense_tensor_two_input_values(self):
    # [FUNCTION DOC]
    # Scenario: two price values per example. Asserts the dense tensor is the one-hot encoding per
    # value, giving a three-dimensional result with a row per example and a one-hot vector per input
    # value.
    """Tests _get_dense_tensor() for input with shape=[2]."""
    price = fc._numeric_column('price', shape=[2])
    bucketized_price = fc._bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      builder = _LazyBuilder({'price': [[-1., 1.], [5., 6.]]})
      with _initialized_session():
        bucketized_price_tensor = bucketized_price._get_dense_tensor(builder)
        self.assertAllClose(
            # One-hot tensor.
            [[[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.]],
             [[0., 0., 0., 1., 0.], [0., 0., 0., 0., 1.]]],
            self.evaluate(bucketized_price_tensor))

  def test_get_sparse_tensors_one_input_value(self):
    # [FUNCTION DOC]
    # Scenario: one price value per example. Asserts no weight tensor is returned and the id tensor
    # has the expected indices, bucket ids and dense shape of four by one.
    """Tests _get_sparse_tensors() for input with shape=[1]."""
    price = fc._numeric_column('price', shape=[1])
    bucketized_price = fc._bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      builder = _LazyBuilder({'price': [[-1.], [1.], [5.], [6.]]})
      with _initialized_session() as sess:
        id_weight_pair = bucketized_price._get_sparse_tensors(builder)
        self.assertIsNone(id_weight_pair.weight_tensor)
        id_tensor_value = sess.run(id_weight_pair.id_tensor)
        self.assertAllEqual(
            [[0, 0], [1, 0], [2, 0], [3, 0]], id_tensor_value.indices)
        self.assertAllEqual([0, 1, 3, 4], id_tensor_value.values)
        self.assertAllEqual([4, 1], id_tensor_value.dense_shape)

  def test_get_sparse_tensors_two_input_values(self):
    # [FUNCTION DOC]
    # Scenario: two price values per example. Asserts no weight tensor is returned and the id tensor
    # has bucket ids offset by the bucket count for the second input column, with a dense shape of two
    # by two.
    """Tests _get_sparse_tensors() for input with shape=[2]."""
    price = fc._numeric_column('price', shape=[2])
    bucketized_price = fc._bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      builder = _LazyBuilder({'price': [[-1., 1.], [5., 6.]]})
      with _initialized_session() as sess:
        id_weight_pair = bucketized_price._get_sparse_tensors(builder)
        self.assertIsNone(id_weight_pair.weight_tensor)
        id_tensor_value = sess.run(id_weight_pair.id_tensor)
        self.assertAllEqual(
            [[0, 0], [0, 1], [1, 0], [1, 1]], id_tensor_value.indices)
        # Values 0-4 correspond to the first column of the input price.
        # Values 5-9 correspond to the second column of the input price.
        self.assertAllEqual([0, 6, 3, 9], id_tensor_value.values)
        self.assertAllEqual([2, 2], id_tensor_value.dense_shape)

  def test_sparse_tensor_input_not_supported(self):
    # [FUNCTION DOC]
    # Asserts that transforming a sparse tensor input with a bucketized column raises a value error
    # saying the input must be a dense tensor.
    price = fc._numeric_column('price')
    bucketized_price = fc._bucketized_column(price, boundaries=[0, 1])
    builder = _LazyBuilder({
        'price':
            sparse_tensor.SparseTensor(
                indices=[[0, 0]], values=[0.3], dense_shape=[1, 1])
    })
    with self.assertRaisesRegex(ValueError, 'must be a Tensor'):
      bucketized_price._transform_feature(builder)

  def test_deep_copy(self):
    # [FUNCTION DOC]
    # Deep-copies a bucketized column and asserts the copy keeps its name, variable shape and
    # boundaries.
    a = fc._numeric_column('aaa', shape=[2])
    a_bucketized = fc._bucketized_column(a, boundaries=[0, 1])
    a_bucketized_copy = copy.deepcopy(a_bucketized)
    self.assertEqual(a_bucketized_copy.name, 'aaa_bucketized')
    self.assertAllEqual(a_bucketized_copy._variable_shape, (2, 3))
    self.assertEqual(a_bucketized_copy.boundaries, (0, 1))

  def test_linear_model_one_input_value(self):
    # [FUNCTION DOC]
    # Scenario: bucketized price with one value per example in the linear model function. Checks zero
    # initial bias, weights per bucket and predictions, then assigns weights and bias and asserts each
    # prediction equals the weight of the example bucket plus the bias.
    """Tests linear_model() for input with shape=[1]."""
    price = fc._numeric_column('price', shape=[1])
    bucketized_price = fc._bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      features = {'price': [[-1.], [1.], [5.], [6.]]}
      predictions = fc.linear_model(features, [bucketized_price])
      bias = get_linear_model_bias()
      bucketized_price_var = get_linear_model_column_var(bucketized_price)
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        # One weight variable per bucket, all initialized to zero.
        self.assertAllClose([[0.], [0.], [0.], [0.], [0.]],
                            self.evaluate(bucketized_price_var))
        self.assertAllClose([[0.], [0.], [0.], [0.]],
                            self.evaluate(predictions))
        sess.run(bucketized_price_var.assign(
            [[10.], [20.], [30.], [40.], [50.]]))
        # price -1. is in the 0th bucket, whose weight is 10.
        # price 1. is in the 1st bucket, whose weight is 20.
        # price 5. is in the 3rd bucket, whose weight is 40.
        # price 6. is in the 4th bucket, whose weight is 50.
        self.assertAllClose([[10.], [20.], [40.], [50.]],
                            self.evaluate(predictions))
        sess.run(bias.assign([1.]))
        self.assertAllClose([[11.], [21.], [41.], [51.]],
                            self.evaluate(predictions))

  def test_linear_model_two_input_values(self):
    # [FUNCTION DOC]
    # Scenario: bucketized price with two values per example in the linear model function. Asserts ten
    # zero weights initially, then after assigning weights asserts each prediction is the sum of the
    # weights of the two buckets hit, and that adding a bias shifts each prediction.
    """Tests linear_model() for input with shape=[2]."""
    price = fc._numeric_column('price', shape=[2])
    bucketized_price = fc._bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      features = {'price': [[-1., 1.], [5., 6.]]}
      predictions = fc.linear_model(features, [bucketized_price])
      bias = get_linear_model_bias()
      bucketized_price_var = get_linear_model_column_var(bucketized_price)
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        # One weight per bucket per input column, all initialized to zero.
        self.assertAllClose(
            [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]],
            self.evaluate(bucketized_price_var))
        self.assertAllClose([[0.], [0.]], self.evaluate(predictions))
        sess.run(bucketized_price_var.assign(
            [[10.], [20.], [30.], [40.], [50.],
             [60.], [70.], [80.], [90.], [100.]]))
        # 1st example:
        #   price -1. is in the 0th bucket, whose weight is 10.
        #   price 1. is in the 6th bucket, whose weight is 70.
        # 2nd example:
        #   price 5. is in the 3rd bucket, whose weight is 40.
        #   price 6. is in the 9th bucket, whose weight is 100.
        self.assertAllClose([[80.], [140.]], self.evaluate(predictions))
        sess.run(bias.assign([1.]))
        self.assertAllClose([[81.], [141.]], self.evaluate(predictions))

  def test_keras_linear_model_one_input_value(self):
    # [FUNCTION DOC]
    # Same as the one-value linear model test but through the Keras-style linear model helper; asserts
    # weights, bucket lookups and bias in the predictions.
    """Tests _LinearModel for input with shape=[1]."""
    price = fc._numeric_column('price', shape=[1])
    bucketized_price = fc._bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      features = {'price': [[-1.], [1.], [5.], [6.]]}
      predictions = get_keras_linear_model_predictions(features,
                                                       [bucketized_price])
      bias = get_linear_model_bias()
      bucketized_price_var = get_linear_model_column_var(bucketized_price)
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        # One weight variable per bucket, all initialized to zero.
        self.assertAllClose([[0.], [0.], [0.], [0.], [0.]],
                            self.evaluate(bucketized_price_var))
        self.assertAllClose([[0.], [0.], [0.], [0.]],
                            self.evaluate(predictions))
        sess.run(
            bucketized_price_var.assign([[10.], [20.], [30.], [40.], [50.]]))
        # price -1. is in the 0th bucket, whose weight is 10.
        # price 1. is in the 1st bucket, whose weight is 20.
        # price 5. is in the 3rd bucket, whose weight is 40.
        # price 6. is in the 4th bucket, whose weight is 50.
        self.assertAllClose([[10.], [20.], [40.], [50.]],
                            self.evaluate(predictions))
        sess.run(bias.assign([1.]))
        self.assertAllClose([[11.], [21.], [41.], [51.]],
                            self.evaluate(predictions))

  def test_keras_linear_model_two_input_values(self):
    # [FUNCTION DOC]
    # Same as the two-value linear model test but through the Keras-style linear model helper; asserts
    # per-example sums of the selected bucket weights, then the bias shift.
    """Tests _LinearModel for input with shape=[2]."""
    price = fc._numeric_column('price', shape=[2])
    bucketized_price = fc._bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      features = {'price': [[-1., 1.], [5., 6.]]}
      predictions = get_keras_linear_model_predictions(features,
                                                       [bucketized_price])
      bias = get_linear_model_bias()
      bucketized_price_var = get_linear_model_column_var(bucketized_price)
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        # One weight per bucket per input column, all initialized to zero.
        self.assertAllClose(
            [[0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.], [0.]],
            self.evaluate(bucketized_price_var))
        self.assertAllClose([[0.], [0.]], self.evaluate(predictions))
        sess.run(
            bucketized_price_var.assign([[10.], [20.], [30.], [40.], [50.],
                                         [60.], [70.], [80.], [90.], [100.]]))
        # 1st example:
        #   price -1. is in the 0th bucket, whose weight is 10.
        #   price 1. is in the 6th bucket, whose weight is 70.
        # 2nd example:
        #   price 5. is in the 3rd bucket, whose weight is 40.
        #   price 6. is in the 9th bucket, whose weight is 100.
        self.assertAllClose([[80.], [140.]], self.evaluate(predictions))
        sess.run(bias.assign([1.]))
        self.assertAllClose([[81.], [141.]], self.evaluate(predictions))


class HashedCategoricalColumnTest(test.TestCase):
  # [CLASS DOC]
  # Tests the legacy hashed categorical column: defaults, validation, deep copy, parsing, hashing of
  # string and integer inputs into buckets, tensor extraction, weight collection behavior and linear
  # model results.

  def test_defaults(self):
    # [FUNCTION DOC]
    # Creates a hashed column with ten buckets and asserts name, variable scope name, key, bucket size
    # and the default string dtype.
    a = fc._categorical_column_with_hash_bucket('aaa', 10)
    self.assertEqual('aaa', a.name)
    self.assertEqual('aaa', a._var_scope_name)
    self.assertEqual('aaa', a.key)
    self.assertEqual(10, a.hash_bucket_size)
    self.assertEqual(dtypes.string, a.dtype)

  def test_key_should_be_string(self):
    # [FUNCTION DOC]
    # Asserts that a non-string key raises a value error saying the key must be a string.
    with self.assertRaisesRegex(ValueError, 'key must be a string.'):
      fc._categorical_column_with_hash_bucket(('key',), 10)

  def test_bucket_size_should_be_given(self):
    # [FUNCTION DOC]
    # Asserts that a missing bucket size raises a value error saying hash bucket size must be set.
    with self.assertRaisesRegex(ValueError, 'hash_bucket_size must be set.'):
      fc._categorical_column_with_hash_bucket('aaa', None)

  def test_bucket_size_should_be_positive(self):
    # [FUNCTION DOC]
    # Asserts that a bucket size of zero raises a value error saying it must be at least one.
    with self.assertRaisesRegex(ValueError,
                                'hash_bucket_size must be at least 1'):
      fc._categorical_column_with_hash_bucket('aaa', 0)

  def test_dtype_should_be_string_or_integer(self):
    # [FUNCTION DOC]
    # Asserts that string and int32 dtypes are accepted while a float dtype raises a value error
    # saying the dtype must be string or integer.
    fc._categorical_column_with_hash_bucket('aaa', 10, dtype=dtypes.string)
    fc._categorical_column_with_hash_bucket('aaa', 10, dtype=dtypes.int32)
    with self.assertRaisesRegex(ValueError, 'dtype must be string or integer'):
      fc._categorical_column_with_hash_bucket('aaa', 10, dtype=dtypes.float32)

  def test_deep_copy(self):
    # [FUNCTION DOC]
    # Asserts that both the original and a deep copy of a hashed column keep name, bucket size, number
    # of buckets and string dtype.
    original = fc._categorical_column_with_hash_bucket('aaa', 10)
    for column in (original, copy.deepcopy(original)):
      self.assertEqual('aaa', column.name)
      self.assertEqual(10, column.hash_bucket_size)
      self.assertEqual(10, column._num_buckets)
      self.assertEqual(dtypes.string, column.dtype)

  def test_parse_spec_string(self):
    # [FUNCTION DOC]
    # Asserts that the parse spec of a default hashed column is a variable length string feature under
    # its key.
    a = fc._categorical_column_with_hash_bucket('aaa', 10)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.string)
    }, a._parse_example_spec)

  def test_parse_spec_int(self):
    # [FUNCTION DOC]
    # Asserts that the parse spec of an int32 hashed column is a variable length int32 feature under
    # its key.
    a = fc._categorical_column_with_hash_bucket('aaa', 10, dtype=dtypes.int32)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int32)
    }, a._parse_example_spec)

  def test_parse_example(self):
    # [FUNCTION DOC]
    # Parses a serialized example holding two byte strings with the spec of a hashed column and
    # asserts the parsed sparse tensor holds the original strings with expected indices and dense
    # shape.
    a = fc._categorical_column_with_hash_bucket('aaa', 10)
    data = example_pb2.Example(features=feature_pb2.Features(
        feature={
            'aaa':
                feature_pb2.Feature(bytes_list=feature_pb2.BytesList(
                    value=[b'omar', b'stringer']))
        }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec([a]))
    self.assertIn('aaa', features)
    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [0, 1]],
            values=np.array([b'omar', b'stringer'], dtype=np.object_),
            dense_shape=[1, 2]), self.evaluate(features['aaa']))

  def test_strings_should_be_hashed(self):
    # [FUNCTION DOC]
    # Transforms a sparse string tensor into bucket ids and asserts the exact hashed values (which
    # would break if the hash function changed), int64 dtype and unchanged indices and dense shape.
    hashed_sparse = fc._categorical_column_with_hash_bucket('wire', 10)
    wire_tensor = sparse_tensor.SparseTensor(
        values=['omar', 'stringer', 'marlo'],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    outputs = _transform_features({'wire': wire_tensor}, [hashed_sparse])
    output = outputs[hashed_sparse]
    # Check exact hashed output. If hashing changes this test will break.
    expected_values = [6, 4, 1]
    with self.cached_session():
      self.assertEqual(dtypes.int64, output.values.dtype)
      self.assertAllEqual(expected_values, output.values)
      self.assertAllEqual(wire_tensor.indices, output.indices)
      self.assertAllEqual(wire_tensor.dense_shape, output.dense_shape)

  def test_tensor_dtype_should_be_string_or_integer(self):
    # [FUNCTION DOC]
    # Builds columns of string and int types and sparse inputs of string, integer and float values.
    # Asserts string and integer inputs are accepted by the lazy builder and a float input raises a
    # value error saying the dtype must be string or integer.
    string_fc = fc._categorical_column_with_hash_bucket(
        'a_string', 10, dtype=dtypes.string)
    int_fc = fc._categorical_column_with_hash_bucket(
        'a_int', 10, dtype=dtypes.int32)
    float_fc = fc._categorical_column_with_hash_bucket(
        'a_float', 10, dtype=dtypes.string)
    int_tensor = sparse_tensor.SparseTensor(
        values=[101],
        indices=[[0, 0]],
        dense_shape=[1, 1])
    string_tensor = sparse_tensor.SparseTensor(
        values=['101'],
        indices=[[0, 0]],
        dense_shape=[1, 1])
    float_tensor = sparse_tensor.SparseTensor(
        values=[101.],
        indices=[[0, 0]],
        dense_shape=[1, 1])
    builder = _LazyBuilder({
        'a_int': int_tensor,
        'a_string': string_tensor,
        'a_float': float_tensor
    })
    builder.get(string_fc)
    builder.get(int_fc)
    with self.assertRaisesRegex(ValueError, 'dtype must be string or integer'):
      builder.get(float_fc)

  def test_dtype_should_match_with_tensor(self):
    # [FUNCTION DOC]
    # Asserts that feeding string values to an int64 hashed column raises a value error saying the
    # dtype must be compatible.
    hashed_sparse = fc._categorical_column_with_hash_bucket(
        'wire', 10, dtype=dtypes.int64)
    wire_tensor = sparse_tensor.SparseTensor(
        values=['omar'], indices=[[0, 0]], dense_shape=[1, 1])
    builder = _LazyBuilder({'wire': wire_tensor})
    with self.assertRaisesRegex(ValueError, 'dtype must be compatible'):
      builder.get(hashed_sparse)

  def test_ints_should_be_hashed(self):
    # [FUNCTION DOC]
    # Hashes integer ids with an int64 hashed column and asserts the exact expected bucket values.
    hashed_sparse = fc._categorical_column_with_hash_bucket(
        'wire', 10, dtype=dtypes.int64)
    wire_tensor = sparse_tensor.SparseTensor(
        values=[101, 201, 301],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    builder = _LazyBuilder({'wire': wire_tensor})
    output = builder.get(hashed_sparse)
    # Check exact hashed output. If hashing changes this test will break.
    expected_values = [3, 7, 5]
    with self.cached_session():
      self.assertAllEqual(expected_values, output.values)

  def test_int32_64_is_compatible(self):
    # [FUNCTION DOC]
    # Asserts that an int32 input tensor is accepted by an int64 hashed column and hashes to the same
    # expected values.
    hashed_sparse = fc._categorical_column_with_hash_bucket(
        'wire', 10, dtype=dtypes.int64)
    wire_tensor = sparse_tensor.SparseTensor(
        values=constant_op.constant([101, 201, 301], dtype=dtypes.int32),
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    builder = _LazyBuilder({'wire': wire_tensor})
    output = builder.get(hashed_sparse)
    # Check exact hashed output. If hashing changes this test will break.
    expected_values = [3, 7, 5]
    self.assertAllEqual(expected_values, output.values)

  def test_get_sparse_tensors(self):
    # [FUNCTION DOC]
    # Asserts that the sparse tensors method returns no weight tensor and an id tensor equal to the
    # lazy builder output for the same column.
    hashed_sparse = fc._categorical_column_with_hash_bucket('wire', 10)
    builder = _LazyBuilder({
        'wire':
            sparse_tensor.SparseTensor(
                values=['omar', 'stringer', 'marlo'],
                indices=[[0, 0], [1, 0], [1, 1]],
                dense_shape=[2, 2])
    })
    id_weight_pair = hashed_sparse._get_sparse_tensors(builder)
    self.assertIsNone(id_weight_pair.weight_tensor)
    self.assertEqual(builder.get(hashed_sparse), id_weight_pair.id_tensor)

  def test_get_sparse_tensors_weight_collections(self):
    # [FUNCTION DOC]
    # Calls the sparse tensors method with a named weight collection and asserts that no variables are
    # added either to the global variables collection or the named collection.
    column = fc._categorical_column_with_hash_bucket('aaa', 10)
    inputs = sparse_tensor.SparseTensor(
        values=['omar', 'stringer', 'marlo'],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    column._get_sparse_tensors(
        _LazyBuilder({
            'aaa': inputs
        }), weight_collections=('my_weights',))

    self.assertCountEqual([],
                          ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES))
    self.assertCountEqual([], ops.get_collection('my_weights'))

  def test_get_sparse_tensors_dense_input(self):
    # [FUNCTION DOC]
    # Feeds a dense nested tuple of strings including an empty string and asserts no weight tensor is
    # returned and the id tensor equals the lazy builder output.
    hashed_sparse = fc._categorical_column_with_hash_bucket('wire', 10)
    builder = _LazyBuilder({'wire': (('omar', ''), ('stringer', 'marlo'))})
    id_weight_pair = hashed_sparse._get_sparse_tensors(builder)
    self.assertIsNone(id_weight_pair.weight_tensor)
    self.assertEqual(builder.get(hashed_sparse), id_weight_pair.id_tensor)

  def test_linear_model(self):
    # [FUNCTION DOC]
    # Scenario: four-bucket hashed column in the linear model function. Asserts zero bias, weights and
    # predictions, then assigns weights one to four and asserts each prediction is the sum of the
    # weights at the hashed buckets of the example strings.
    wire_column = fc._categorical_column_with_hash_bucket('wire', 4)
    self.assertEqual(4, wire_column._num_buckets)
    with ops.Graph().as_default():
      predictions = fc.linear_model({
          wire_column.name: sparse_tensor.SparseTensorValue(
              indices=((0, 0), (1, 0), (1, 1)),
              values=('marlo', 'skywalker', 'omar'),
              dense_shape=(2, 2))
      }, (wire_column,))
      bias = get_linear_model_bias()
      wire_var = get_linear_model_column_var(wire_column)
      with _initialized_session():
        self.assertAllClose((0.,), self.evaluate(bias))
        self.assertAllClose(((0.,), (0.,), (0.,), (0.,)),
                            self.evaluate(wire_var))
        self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
        self.evaluate(wire_var.assign(((1.,), (2.,), (3.,), (4.,))))
        # 'marlo' -> 3: wire_var[3] = 4
        # 'skywalker' -> 2, 'omar' -> 2: wire_var[2] + wire_var[2] = 3+3 = 6
        self.assertAllClose(((4.,), (6.,)), self.evaluate(predictions))

  def test_keras_linear_model(self):
    # [FUNCTION DOC]
    # Same scenario as the linear model test but built through the Keras-style linear model helper.
    wire_column = fc._categorical_column_with_hash_bucket('wire', 4)
    self.assertEqual(4, wire_column._num_buckets)
    with ops.Graph().as_default():
      predictions = get_keras_linear_model_predictions({
          wire_column.name:
              sparse_tensor.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=('marlo', 'skywalker', 'omar'),
                  dense_shape=(2, 2))
      }, (wire_column,))
      bias = get_linear_model_bias()
      wire_var = get_linear_model_column_var(wire_column)
      with _initialized_session():
        self.assertAllClose((0.,), self.evaluate(bias))
        self.assertAllClose(((0.,), (0.,), (0.,), (0.,)),
                            self.evaluate(wire_var))
        self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
        self.evaluate(wire_var.assign(((1.,), (2.,), (3.,), (4.,))))
        # 'marlo' -> 3: wire_var[3] = 4
        # 'skywalker' -> 2, 'omar' -> 2: wire_var[2] + wire_var[2] = 3+3 = 6
        self.assertAllClose(((4.,), (6.,)), self.evaluate(predictions))


class CrossedColumnTest(test.TestCase):
  # [CLASS DOC]
  # Tests the legacy crossed column: key validation, bucket size validation, naming (alphabetical,
  # nested crossings), parse spec, deep copy, transformation to hashed cross ids, sparse tensor
  # extraction and linear model use including the rejection of weighted inputs.

  def test_keys_empty(self):
    # [FUNCTION DOC]
    # Asserts that an empty key list raises a value error saying keys must be a list with more than
    # one element.
    with self.assertRaisesRegex(ValueError,
                                'keys must be a list with length > 1'):
      fc._crossed_column([], 10)

  def test_keys_length_one(self):
    # [FUNCTION DOC]
    # Asserts that a key list with a single element raises the same value error.
    with self.assertRaisesRegex(ValueError,
                                'keys must be a list with length > 1'):
      fc._crossed_column(['a'], 10)

  def test_key_type_unsupported(self):
    # [FUNCTION DOC]
    # Asserts that a numeric column key raises a value error for an unsupported key type, and that a
    # hashed categorical column key raises a value error saying it is not supported.
    with self.assertRaisesRegex(ValueError, 'Unsupported key type'):
      fc._crossed_column(['a', fc._numeric_column('c')], 10)

    with self.assertRaisesRegex(
        ValueError, 'categorical_column_with_hash_bucket is not supported'):
      fc._crossed_column(
          ['a', fc._categorical_column_with_hash_bucket('c', 10)], 10)

  def test_hash_bucket_size_negative(self):
    # [FUNCTION DOC]
    # Asserts that a negative hash bucket size raises a value error saying it must be greater than
    # one.
    with self.assertRaisesRegex(ValueError, 'hash_bucket_size must be > 1'):
      fc._crossed_column(['a', 'c'], -1)

  def test_hash_bucket_size_zero(self):
    # [FUNCTION DOC]
    # Asserts that a zero hash bucket size raises the same value error.
    with self.assertRaisesRegex(ValueError, 'hash_bucket_size must be > 1'):
      fc._crossed_column(['a', 'c'], 0)

  def test_hash_bucket_size_none(self):
    # [FUNCTION DOC]
    # Asserts that a missing hash bucket size raises the same value error.
    with self.assertRaisesRegex(ValueError, 'hash_bucket_size must be > 1'):
      fc._crossed_column(['a', 'c'], None)

  def test_name(self):
    # [FUNCTION DOC]
    # Crosses a bucketized column, a string key and another crossed column, and asserts the combined
    # name joins the sorted leaf key names with a cross separator.
    a = fc._numeric_column('a', dtype=dtypes.int32)
    b = fc._bucketized_column(a, boundaries=[0, 1])
    crossed1 = fc._crossed_column(['d1', 'd2'], 10)

    crossed2 = fc._crossed_column([b, 'c', crossed1], 10)
    self.assertEqual('a_bucketized_X_c_X_d1_X_d2', crossed2.name)

  def test_name_ordered_alphabetically(self):
    # [FUNCTION DOC]
    # Asserts that the crossed column name does not depend on the order of the crossed inputs, using a
    # different input order than the previous test.
    """Tests that the name does not depend on the order of given columns."""
    a = fc._numeric_column('a', dtype=dtypes.int32)
    b = fc._bucketized_column(a, boundaries=[0, 1])
    crossed1 = fc._crossed_column(['d1', 'd2'], 10)

    crossed2 = fc._crossed_column([crossed1, 'c', b], 10)
    self.assertEqual('a_bucketized_X_c_X_d1_X_d2', crossed2.name)

  def test_name_leaf_keys_ordered_alphabetically(self):
    # [FUNCTION DOC]
    # Asserts that leaf keys of nested crossed columns are also sorted alphabetically in the final
    # name.
    """Tests that the name does not depend on the order of given columns."""
    a = fc._numeric_column('a', dtype=dtypes.int32)
    b = fc._bucketized_column(a, boundaries=[0, 1])
    crossed1 = fc._crossed_column(['d2', 'c'], 10)

    crossed2 = fc._crossed_column([crossed1, 'd1', b], 10)
    self.assertEqual('a_bucketized_X_c_X_d1_X_d2', crossed2.name)

  def test_var_scope_name(self):
    # [FUNCTION DOC]
    # Asserts that the variable scope name of a nested crossed column equals its combined sorted name.
    a = fc._numeric_column('a', dtype=dtypes.int32)
    b = fc._bucketized_column(a, boundaries=[0, 1])
    crossed1 = fc._crossed_column(['d1', 'd2'], 10)

    crossed2 = fc._crossed_column([b, 'c', crossed1], 10)
    self.assertEqual('a_bucketized_X_c_X_d1_X_d2', crossed2._var_scope_name)

  def test_parse_spec(self):
    # [FUNCTION DOC]
    # Asserts the parse spec of a cross of a bucketized numeric column and a string key combines a
    # fixed length int32 feature for the numeric source and a variable length string feature for the
    # plain key.
    a = fc._numeric_column('a', shape=[2], dtype=dtypes.int32)
    b = fc._bucketized_column(a, boundaries=[0, 1])
    crossed = fc._crossed_column([b, 'c'], 10)
    self.assertEqual({
        'a': parsing_ops.FixedLenFeature((2,), dtype=dtypes.int32),
        'c': parsing_ops.VarLenFeature(dtypes.string),
    }, crossed._parse_example_spec)

  def test_num_buckets(self):
    # [FUNCTION DOC]
    # Asserts that the number of buckets of a crossed column equals the requested hash bucket size.
    a = fc._numeric_column('a', shape=[2], dtype=dtypes.int32)
    b = fc._bucketized_column(a, boundaries=[0, 1])
    crossed = fc._crossed_column([b, 'c'], 15)
    self.assertEqual(15, crossed._num_buckets)

  def test_deep_copy(self):
    # [FUNCTION DOC]
    # Deep-copies a nested crossed column with a hash key and asserts the copy keeps its name, hash
    # bucket size and hash key.
    a = fc._numeric_column('a', dtype=dtypes.int32)
    b = fc._bucketized_column(a, boundaries=[0, 1])
    crossed1 = fc._crossed_column(['d1', 'd2'], 10)
    crossed2 = fc._crossed_column([b, 'c', crossed1], 15, hash_key=5)
    crossed2_copy = copy.deepcopy(crossed2)
    self.assertEqual('a_bucketized_X_c_X_d1_X_d2', crossed2_copy.name,)
    self.assertEqual(15, crossed2_copy.hash_bucket_size)
    self.assertEqual(5, crossed2_copy.hash_key)

  def test_parse_example(self):
    # [FUNCTION DOC]
    # Parses a serialized example containing a price and a wire feature using the spec of a cross of
    # the bucketized price and wire, and asserts both raw features are parsed with expected values,
    # indices and dense shape.
    price = fc._numeric_column('price', shape=[2])
    bucketized_price = fc._bucketized_column(price, boundaries=[0, 50])
    price_cross_wire = fc._crossed_column([bucketized_price, 'wire'], 10)
    data = example_pb2.Example(features=feature_pb2.Features(
        feature={
            'price':
                feature_pb2.Feature(float_list=feature_pb2.FloatList(
                    value=[20., 110.])),
            'wire':
                feature_pb2.Feature(bytes_list=feature_pb2.BytesList(
                    value=[b'omar', b'stringer'])),
        }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec([price_cross_wire]))
    self.assertIn('price', features)
    self.assertIn('wire', features)
    with self.cached_session():
      self.assertAllEqual([[20., 110.]], features['price'])
      wire_sparse = features['wire']
      self.assertAllEqual([[0, 0], [0, 1]], wire_sparse.indices)
      # Use byte constants to pass the open-source test.
      self.assertAllEqual([b'omar', b'stringer'], wire_sparse.values)
      self.assertAllEqual([1, 2], wire_sparse.dense_shape)

  def test_transform_feature(self):
    # [FUNCTION DOC]
    # Transforms dense price and sparse wire features through a crossed column and asserts the
    # expected output indices, that every hashed value lies within the bucket range, and the dense
    # shape.
    price = fc._numeric_column('price', shape=[2])
    bucketized_price = fc._bucketized_column(price, boundaries=[0, 50])
    hash_bucket_size = 10
    price_cross_wire = fc._crossed_column([bucketized_price, 'wire'],
                                          hash_bucket_size)
    features = {
        'price': constant_op.constant([[1., 2.], [5., 6.]]),
        'wire': sparse_tensor.SparseTensor(
            values=['omar', 'stringer', 'marlo'],
            indices=[[0, 0], [1, 0], [1, 1]],
            dense_shape=[2, 2]),
    }
    outputs = _transform_features(features, [price_cross_wire])
    output = outputs[price_cross_wire]
    output_val = self.evaluate(output)
    self.assertAllEqual([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]],
                        output_val.indices)
    for val in output_val.values:
      self.assertIn(val, list(range(hash_bucket_size)))
    self.assertAllEqual([2, 4], output_val.dense_shape)

  def test_get_sparse_tensors(self):
    # [FUNCTION DOC]
    # Scenario: a nested cross of a bucketized column, a plain key and another cross with several
    # sparse inputs. Evaluates the id tensor in a graph and asserts the exact indices, the exact
    # hashed values (breaks if hashing changes) and the dense shape.
    a = fc._numeric_column('a', dtype=dtypes.int32, shape=(2,))
    b = fc._bucketized_column(a, boundaries=(0, 1))
    crossed1 = fc._crossed_column(['d1', 'd2'], 10)
    crossed2 = fc._crossed_column([b, 'c', crossed1], 15, hash_key=5)
    with ops.Graph().as_default():
      builder = _LazyBuilder({
          'a':
              constant_op.constant(((-1., .5), (.5, 1.))),
          'c':
              sparse_tensor.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=['cA', 'cB', 'cC'],
                  dense_shape=(2, 2)),
          'd1':
              sparse_tensor.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=['d1A', 'd1B', 'd1C'],
                  dense_shape=(2, 2)),
          'd2':
              sparse_tensor.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=['d2A', 'd2B', 'd2C'],
                  dense_shape=(2, 2)),
      })
      id_weight_pair = crossed2._get_sparse_tensors(builder)
      with _initialized_session():
        id_tensor_eval = id_weight_pair.id_tensor.eval()
        self.assertAllEqual(
            ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
             (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13),
             (1, 14), (1, 15)),
            id_tensor_eval.indices)
        # Check exact hashed output. If hashing changes this test will break.
        # All values are within [0, hash_bucket_size).
        expected_values = (
            6, 14, 0, 13, 8, 8, 10, 12, 2, 0, 1, 9, 8, 12, 2, 0, 10, 11)
        self.assertAllEqual(expected_values, id_tensor_eval.values)
        self.assertAllEqual((2, 16), id_tensor_eval.dense_shape)

  def test_get_sparse_tensors_simple(self):
    # [FUNCTION DOC]
    # Simpler version of the previous test using one cross of a bucketized column and one string key;
    # asserts exact indices, hashed values and dense shape.
    """Same as test_get_sparse_tensors, but with simpler values."""
    a = fc._numeric_column('a', dtype=dtypes.int32, shape=(2,))
    b = fc._bucketized_column(a, boundaries=(0, 1))
    crossed = fc._crossed_column([b, 'c'], hash_bucket_size=5, hash_key=5)
    with ops.Graph().as_default():
      builder = _LazyBuilder({
          'a':
              constant_op.constant(((-1., .5), (.5, 1.))),
          'c':
              sparse_tensor.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=['cA', 'cB', 'cC'],
                  dense_shape=(2, 2)),
      })
      id_weight_pair = crossed._get_sparse_tensors(builder)
      with _initialized_session():
        id_tensor_eval = id_weight_pair.id_tensor.eval()
        self.assertAllEqual(
            ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (1, 3)),
            id_tensor_eval.indices)
        # Check exact hashed output. If hashing changes this test will break.
        # All values are within [0, hash_bucket_size).
        expected_values = (1, 0, 1, 3, 4, 2)
        self.assertAllEqual(expected_values, id_tensor_eval.values)
        self.assertAllEqual((2, 4), id_tensor_eval.dense_shape)

  def test_linear_model(self):
    # [FUNCTION DOC]
    # Scenario: the simple crossed column in the linear model function. Asserts zero initial state,
    # then assigns weights one to five and asserts predictions are the sums of the weights of the
    # crossed ids, then adds a bias and checks the shift.
    """Tests linear_model.

    Uses data from test_get_sparse_tensors_simple.
    """
    a = fc._numeric_column('a', dtype=dtypes.int32, shape=(2,))
    b = fc._bucketized_column(a, boundaries=(0, 1))
    crossed = fc._crossed_column([b, 'c'], hash_bucket_size=5, hash_key=5)
    with ops.Graph().as_default():
      predictions = fc.linear_model({
          'a': constant_op.constant(((-1., .5), (.5, 1.))),
          'c': sparse_tensor.SparseTensor(
              indices=((0, 0), (1, 0), (1, 1)),
              values=['cA', 'cB', 'cC'],
              dense_shape=(2, 2)),
      }, (crossed,))
      bias = get_linear_model_bias()
      crossed_var = get_linear_model_column_var(crossed)
      with _initialized_session():
        self.assertAllClose((0.,), self.evaluate(bias))
        self.assertAllClose(((0.,), (0.,), (0.,), (0.,), (0.,)),
                            self.evaluate(crossed_var))
        self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
        self.evaluate(crossed_var.assign(((1.,), (2.,), (3.,), (4.,), (5.,))))
        # Expected ids after cross = (1, 0, 1, 3, 4, 2)
        self.assertAllClose(((3.,), (14.,)), self.evaluate(predictions))
        self.evaluate(bias.assign((.1,)))
        self.assertAllClose(((3.1,), (14.1,)), self.evaluate(predictions))

  def test_linear_model_with_weights(self):
    # [FUNCTION DOC]
    # Scenario: a cross containing a fixture column that provides both ids and weights. Asserts that
    # the linear model function raises a value error saying a crossed column does not support a weight
    # tensor.
    class _TestColumnWithWeights(_CategoricalColumn):
      # [CLASS DOC]
      # Local fixture categorical column with five buckets that produces sparse ids together with a
      # separate sparse weights feature; used to trigger the weight rejection.
      """Produces sparse IDs and sparse weights."""

      @property
      def name(self):
        # [FUNCTION DOC]
        # Property returning the fixed fixture column name; returns a string.
        return 'test_column'

      @property
      def _parse_example_spec(self):
        # [FUNCTION DOC]
        # Property returning a mapping with a variable length int32 feature for the column name and a
        # variable length float32 feature for the weights key derived from the name.
        return {
            self.name: parsing_ops.VarLenFeature(dtypes.int32),
            '{}_weights'.format(self.name): parsing_ops.VarLenFeature(
                dtypes.float32),
            }

      @property
      def _num_buckets(self):
        # [FUNCTION DOC]
        # Property returning the constant bucket count five.
        return 5

      def _transform_feature(self, inputs):
        # [FUNCTION DOC]
        # Transform hook returning the pair of the ids feature and the weights feature fetched from the
        # given inputs.
        return (inputs.get(self.name),
                inputs.get('{}_weights'.format(self.name)))

      def _get_sparse_tensors(self, inputs, weight_collections=None,
                              trainable=None):
        # [FUNCTION DOC]
        # Returns an id and weight pair built from the transformed pair of ids and weights. Parameters
        # weight_collections and trainable are accepted but ignored; no variables are created.
        """Populates both id_tensor and weight_tensor."""
        ids_and_weights = inputs.get(self)
        return _CategoricalColumn.IdWeightPair(
            id_tensor=ids_and_weights[0], weight_tensor=ids_and_weights[1])

    t = _TestColumnWithWeights()
    crossed = fc._crossed_column([t, 'c'], hash_bucket_size=5, hash_key=5)
    with ops.Graph().as_default():
      with self.assertRaisesRegex(
          ValueError,
          'crossed_column does not support weight_tensor.*{}'.format(t.name)):
        fc.linear_model({
            t.name: sparse_tensor.SparseTensor(
                indices=((0, 0), (1, 0), (1, 1)),
                values=[0, 1, 2],
                dense_shape=(2, 2)),
            '{}_weights'.format(t.name): sparse_tensor.SparseTensor(
                indices=((0, 0), (1, 0), (1, 1)),
                values=[1., 10., 2.],
                dense_shape=(2, 2)),
            'c': sparse_tensor.SparseTensor(
                indices=((0, 0), (1, 0), (1, 1)),
                values=['cA', 'cB', 'cC'],
                dense_shape=(2, 2)),
        }, (crossed,))

  def test_keras_linear_model(self):
    # [FUNCTION DOC]
    # Same as the crossed linear model test but built through the Keras-style linear model helper.
    """Tests _LinearModel.

    Uses data from test_get_sparse_tensors_simple.
    """
    a = fc._numeric_column('a', dtype=dtypes.int32, shape=(2,))
    b = fc._bucketized_column(a, boundaries=(0, 1))
    crossed = fc._crossed_column([b, 'c'], hash_bucket_size=5, hash_key=5)
    with ops.Graph().as_default():
      predictions = get_keras_linear_model_predictions({
          'a':
              constant_op.constant(((-1., .5), (.5, 1.))),
          'c':
              sparse_tensor.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=['cA', 'cB', 'cC'],
                  dense_shape=(2, 2)),
      }, (crossed,))
      bias = get_linear_model_bias()
      crossed_var = get_linear_model_column_var(crossed)
      with _initialized_session():
        self.assertAllClose((0.,), self.evaluate(bias))
        self.assertAllClose(((0.,), (0.,), (0.,), (0.,), (0.,)),
                            self.evaluate(crossed_var))
        self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
        self.evaluate(crossed_var.assign(((1.,), (2.,), (3.,), (4.,), (5.,))))
        # Expected ids after cross = (1, 0, 1, 3, 4, 2)
        self.assertAllClose(((3.,), (14.,)), self.evaluate(predictions))
        self.evaluate(bias.assign((.1,)))
        self.assertAllClose(((3.1,), (14.1,)), self.evaluate(predictions))

  def test_keras_linear_model_with_weights(self):
    # [FUNCTION DOC]
    # Same as the weights rejection test but through the Keras-style linear model helper: asserts a
    # value error saying a crossed column does not support a weight tensor.

    class _TestColumnWithWeights(_CategoricalColumn):
      # [CLASS DOC]
      # Local fixture categorical column with five buckets producing sparse ids and sparse weights, used
      # to trigger the weight rejection.
      """Produces sparse IDs and sparse weights."""

      @property
      def name(self):
        # [FUNCTION DOC]
        # Property returning the fixed fixture column name; returns a string.
        return 'test_column'

      @property
      def _parse_example_spec(self):
        # [FUNCTION DOC]
        # Property returning variable length feature specs for the ids (int32) and for the weights
        # (float32, keyed by the name plus a weights suffix).
        return {
            self.name:
                parsing_ops.VarLenFeature(dtypes.int32),
            '{}_weights'.format(self.name):
                parsing_ops.VarLenFeature(dtypes.float32),
        }

      @property
      def _num_buckets(self):
        # [FUNCTION DOC]
        # Property returning the constant bucket count five.
        return 5

      def _transform_feature(self, inputs):
        # [FUNCTION DOC]
        # Transform hook returning the pair of the ids feature and the weights feature fetched from the
        # given inputs.
        return (inputs.get(self.name),
                inputs.get('{}_weights'.format(self.name)))

      def _get_sparse_tensors(self,
                              inputs,
                              weight_collections=None,
                              trainable=None):
        # [FUNCTION DOC]
        # Returns an id and weight pair built from the transformed pair. Parameters weight_collections and
        # trainable are accepted but ignored.
        """Populates both id_tensor and weight_tensor."""
        ids_and_weights = inputs.get(self)
        return _CategoricalColumn.IdWeightPair(
            id_tensor=ids_and_weights[0], weight_tensor=ids_and_weights[1])

    t = _TestColumnWithWeights()
    crossed = fc._crossed_column([t, 'c'], hash_bucket_size=5, hash_key=5)
    with ops.Graph().as_default():
      with self.assertRaisesRegex(
          ValueError,
          'crossed_column does not support weight_tensor.*{}'.format(t.name)):
        get_keras_linear_model_predictions({
            t.name:
                sparse_tensor.SparseTensor(
                    indices=((0, 0), (1, 0), (1, 1)),
                    values=[0, 1, 2],
                    dense_shape=(2, 2)),
            '{}_weights'.format(t.name):
                sparse_tensor.SparseTensor(
                    indices=((0, 0), (1, 0), (1, 1)),
                    values=[1., 10., 2.],
                    dense_shape=(2, 2)),
            'c':
                sparse_tensor.SparseTensor(
                    indices=((0, 0), (1, 0), (1, 1)),
                    values=['cA', 'cB', 'cC'],
                    dense_shape=(2, 2)),
        }, (crossed,))


def get_linear_model_bias(name='linear_model'):
  # [FUNCTION DOC]
  # HELPER: shared test utility used by the linear model tests to inspect the bias variable.
  # Looks up the bias weights variable created by a linear model inside the variable scope with the
  # given name (default linear_model) by re-entering the scope in reuse mode. Returns that variable.
  # Raises an error if the variable does not exist.
  with variable_scope.variable_scope(name, reuse=True):
    return variable_scope.get_variable('bias_weights')


def get_linear_model_column_var(column, name='linear_model'):
  # [FUNCTION DOC]
  # HELPER: shared test utility used by the linear model tests to inspect per-column weights.
  # Finds the weight variable of the given feature column in the linear model with the given scope
  # name (default linear_model) by querying the global variables collection with a name prefix made
  # of the scope and the column name. Returns the first matching variable. Raises an index error if
  # none exists.
  return ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES,
                            name + '/' + column.name)[0]


def get_keras_linear_model_predictions(features,
                                       feature_columns,
                                       units=1,
                                       sparse_combiner='sum',
                                       weight_collections=None,
                                       trainable=True,
                                       cols_to_vars=None):
  # [FUNCTION DOC]
  # HELPER: wrapper letting the Keras-style linear model tests share assertions with the function
  # based tests.
  # Builds the Keras-style linear model class from the feature columns, number of output units
  # (default one), sparse combiner (default sum), optional weight collections, and trainable flag
  # (default true) under the name linear_model, and calls it on the features. If a mapping is passed
  # as cols_to_vars, it is updated with the model column to variables mapping. Returns the
  # prediction tensor.
  keras_linear_model = _LinearModel(
      feature_columns,
      units,
      sparse_combiner,
      weight_collections,
      trainable,
      name='linear_model')
  retval = keras_linear_model(features)  # pylint: disable=not-callable
  if cols_to_vars is not None:
    cols_to_vars.update(keras_linear_model.cols_to_vars())
  return retval


class LinearModelTest(test.TestCase):
  # [CLASS DOC]
  # Tests the legacy linear model function end to end: argument validation, bias and weight creation
  # for dense and sparse columns, multiple outputs, combiners, reshaping, variable collections,
  # trainability, column ordering, batch size checks, one dimensional and unknown rank inputs and
  # repeated model creation.

  def test_raises_if_empty_feature_columns(self):
    # [FUNCTION DOC]
    # Asserts that an empty feature column list raises a value error saying the columns must not be
    # empty.
    with self.assertRaisesRegex(ValueError,
                                'feature_columns must not be empty'):
      fc.linear_model(features={}, feature_columns=[])

  def test_should_be_feature_column(self):
    # [FUNCTION DOC]
    # Asserts that passing a string instead of a feature column raises a value error saying it must be
    # a feature column.
    with self.assertRaisesRegex(ValueError, 'must be a _FeatureColumn'):
      fc.linear_model(features={'a': [[0]]}, feature_columns='NotSupported')

  def test_should_be_dense_or_categorical_column(self):
    # [FUNCTION DOC]
    # Scenario: a fixture column that is neither dense nor categorical. Asserts the linear model
    # function raises a value error saying a dense or categorical column is required.

    class NotSupportedColumn(_FeatureColumn):
      # [CLASS DOC]
      # Local fixture feature column that is neither dense nor categorical, used to trigger the type
      # error path.

      @property
      def name(self):
        # [FUNCTION DOC]
        # Property returning the fixed fixture name; returns a string.
        return 'NotSupportedColumn'

      def _transform_feature(self, cache):
        # [FUNCTION DOC]
        # Transform hook of the fixture; does nothing and returns nothing.
        pass

      @property
      def _parse_example_spec(self):
        # [FUNCTION DOC]
        # Property required by the interface; returns nothing.
        pass

    with self.assertRaisesRegex(
        ValueError, 'must be either a _DenseColumn or _CategoricalColumn'):
      fc.linear_model(
          features={'a': [[0]]}, feature_columns=[NotSupportedColumn()])

  def test_does_not_support_dict_columns(self):
    # [FUNCTION DOC]
    # Asserts that passing a mapping instead of an iterable of columns raises a value error saying a
    # dict was found.
    with self.assertRaisesRegex(
        ValueError, 'Expected feature_columns to be iterable, found dict.'):
      fc.linear_model(
          features={'a': [[0]]}, feature_columns={'a': fc._numeric_column('a')})

  def test_raises_if_duplicate_name(self):
    # [FUNCTION DOC]
    # Asserts that two numeric columns with the same name raise a value error about a duplicate
    # feature column name.
    with self.assertRaisesRegex(
        ValueError, 'Duplicate feature column name found for columns'):
      fc.linear_model(
          features={'a': [[0]]},
          feature_columns=[fc._numeric_column('a'),
                           fc._numeric_column('a')])

  def test_dense_bias(self):
    # [FUNCTION DOC]
    # Scenario: one numeric price column. Asserts the bias starts at zero, then assigns weight ten and
    # bias five and asserts predictions are the price times ten plus five.
    price = fc._numeric_column('price')
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      predictions = fc.linear_model(features, [price])
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        sess.run(price_var.assign([[10.]]))
        sess.run(bias.assign([5.]))
        self.assertAllClose([[15.], [55.]], self.evaluate(predictions))

  def test_sparse_bias(self):
    # [FUNCTION DOC]
    # Scenario: a four-bucket hashed column. Asserts zero initial bias and weights, then assigns
    # weights and a bias and asserts predictions are the sum of the weights at the hashed ids plus the
    # bias.
    wire_cast = fc._categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default():
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar', 'stringer', 'marlo'],  # hashed to = [2, 0, 3]
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {'wire_cast': wire_tensor}
      predictions = fc.linear_model(features, [wire_cast])
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        self.assertAllClose([[0.], [0.], [0.], [0.]],
                            self.evaluate(wire_cast_var))
        sess.run(wire_cast_var.assign([[10.], [100.], [1000.], [10000.]]))
        sess.run(bias.assign([5.]))
        self.assertAllClose([[1005.], [10015.]], self.evaluate(predictions))

  def test_dense_and_sparse_bias(self):
    # [FUNCTION DOC]
    # Scenario: one hashed and one numeric column together. After assigning weights and the bias,
    # asserts predictions combine the sparse and dense contributions and a single bias.
    wire_cast = fc._categorical_column_with_hash_bucket('wire_cast', 4)
    price = fc._numeric_column('price')
    with ops.Graph().as_default():
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar', 'stringer', 'marlo'],  # hashed to = [2, 0, 3]
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {'wire_cast': wire_tensor, 'price': [[1.], [5.]]}
      predictions = fc.linear_model(features, [wire_cast, price])
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        sess.run(wire_cast_var.assign([[10.], [100.], [1000.], [10000.]]))
        sess.run(bias.assign([5.]))
        sess.run(price_var.assign([[10.]]))
        self.assertAllClose([[1015.], [10065.]], self.evaluate(predictions))

  def test_dense_and_sparse_column(self):
    # [FUNCTION DOC]
    # Scenario: a fixture column that is both dense and categorical. Asserts the linear model uses the
    # sparse path (its dense methods raise errors if called) and that predictions equal the sum of the
    # selected weights plus the bias.
    """When the column is both dense and sparse, uses sparse tensors."""

    class _DenseAndSparseColumn(_DenseColumn, _CategoricalColumn):
      # [CLASS DOC]
      # Local fixture column that is both a dense and a categorical column. Its dense accessors raise
      # errors, so any use of the dense path makes the test fail; the sparse path returns fixed ids.

      @property
      def name(self):
        # [FUNCTION DOC]
        # Property returning the fixed fixture name; returns a string.
        return 'dense_and_sparse_column'

      @property
      def _parse_example_spec(self):
        # [FUNCTION DOC]
        # Property returning a variable length feature spec keyed by the column name with the column
        # dtype.
        return {self.name: parsing_ops.VarLenFeature(self.dtype)}

      def _transform_feature(self, inputs):
        # [FUNCTION DOC]
        # Transform hook returning the raw feature with the column name taken from the given inputs.
        return inputs.get(self.name)

      @property
      def _variable_shape(self):
        # [FUNCTION DOC]
        # Dense interface property; always raises a value error saying it should not be used.
        raise ValueError('Should not use this method.')

      def _get_dense_tensor(self, inputs, weight_collections=None,
                            trainable=None):
        # [FUNCTION DOC]
        # Dense interface method with inputs, weight collections and trainable parameters; always raises a
        # value error saying it should not be used.
        raise ValueError('Should not use this method.')

      @property
      def _num_buckets(self):
        # [FUNCTION DOC]
        # Property returning the constant bucket count four.
        return 4

      def _get_sparse_tensors(self, inputs, weight_collections=None,
                              trainable=None):
        # [FUNCTION DOC]
        # Returns an id and weight pair whose id tensor is a fixed sparse tensor with ids two, zero and
        # three and no weights. The input parameters are ignored.
        sp_tensor = sparse_tensor.SparseTensor(
            indices=[[0, 0], [1, 0], [1, 1]],
            values=[2, 0, 3],
            dense_shape=[2, 2])
        return _CategoricalColumn.IdWeightPair(sp_tensor, None)

    dense_and_sparse_column = _DenseAndSparseColumn()
    with ops.Graph().as_default():
      sp_tensor = sparse_tensor.SparseTensor(
          values=['omar', 'stringer', 'marlo'],
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {dense_and_sparse_column.name: sp_tensor}
      predictions = fc.linear_model(features, [dense_and_sparse_column])
      bias = get_linear_model_bias()
      dense_and_sparse_column_var = get_linear_model_column_var(
          dense_and_sparse_column)
      with _initialized_session() as sess:
        sess.run(dense_and_sparse_column_var.assign(
            [[10.], [100.], [1000.], [10000.]]))
        sess.run(bias.assign([5.]))
        self.assertAllClose([[1005.], [10015.]], self.evaluate(predictions))

  def test_dense_multi_output(self):
    # [FUNCTION DOC]
    # Scenario: numeric column with three output units. Asserts bias has three zeros and the weight is
    # one by three, then assigns values and asserts the three-wide predictions.
    price = fc._numeric_column('price')
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      predictions = fc.linear_model(features, [price], units=3)
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        self.assertAllClose(np.zeros((3,)), self.evaluate(bias))
        self.assertAllClose(np.zeros((1, 3)), self.evaluate(price_var))
        sess.run(price_var.assign([[10., 100., 1000.]]))
        sess.run(bias.assign([5., 6., 7.]))
        self.assertAllClose([[15., 106., 1007.], [55., 506., 5007.]],
                            self.evaluate(predictions))

  def test_sparse_multi_output(self):
    # [FUNCTION DOC]
    # Scenario: four-bucket hashed column with three output units. Asserts shapes of the bias and
    # weights then, after assigning, asserts each prediction row sums the selected weight rows plus
    # the bias.
    wire_cast = fc._categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default():
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar', 'stringer', 'marlo'],  # hashed to = [2, 0, 3]
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {'wire_cast': wire_tensor}
      predictions = fc.linear_model(features, [wire_cast], units=3)
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      with _initialized_session() as sess:
        self.assertAllClose(np.zeros((3,)), self.evaluate(bias))
        self.assertAllClose(np.zeros((4, 3)), self.evaluate(wire_cast_var))
        sess.run(
            wire_cast_var.assign([[10., 11., 12.], [100., 110., 120.], [
                1000., 1100., 1200.
            ], [10000., 11000., 12000.]]))
        sess.run(bias.assign([5., 6., 7.]))
        self.assertAllClose([[1005., 1106., 1207.], [10015., 11017., 12019.]],
                            self.evaluate(predictions))

  def test_dense_multi_dimension(self):
    # [FUNCTION DOC]
    # Scenario: numeric column of shape two. Asserts a two by one weight, then after assigning asserts
    # the predictions are dot products of each feature row with the weights.
    price = fc._numeric_column('price', shape=2)
    with ops.Graph().as_default():
      features = {'price': [[1., 2.], [5., 6.]]}
      predictions = fc.linear_model(features, [price])
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        self.assertAllClose([[0.], [0.]], self.evaluate(price_var))
        sess.run(price_var.assign([[10.], [100.]]))
        self.assertAllClose([[210.], [650.]], self.evaluate(predictions))

  def test_sparse_multi_rank(self):
    # [FUNCTION DOC]
    # Scenario: a rank three sparse placeholder fed with a value. Asserts predictions are zero with
    # zero weights, then after assigning weights asserts predictions sum the weights of the hashed ids
    # over the inner dimensions of each example.
    wire_cast = fc._categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default():
      wire_tensor = array_ops.sparse_placeholder(dtypes.string)
      wire_value = sparse_tensor.SparseTensorValue(
          values=['omar', 'stringer', 'marlo', 'omar'],  # hashed = [2, 0, 3, 2]
          indices=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 1]],
          dense_shape=[2, 2, 2])
      features = {'wire_cast': wire_tensor}
      predictions = fc.linear_model(features, [wire_cast])
      wire_cast_var = get_linear_model_column_var(wire_cast)
      with _initialized_session() as sess:
        self.assertAllClose(np.zeros((4, 1)), self.evaluate(wire_cast_var))
        self.assertAllClose(
            np.zeros((2, 1)),
            predictions.eval(feed_dict={wire_tensor: wire_value}))
        sess.run(wire_cast_var.assign([[10.], [100.], [1000.], [10000.]]))
        self.assertAllClose(
            [[1010.], [11000.]],
            predictions.eval(feed_dict={wire_tensor: wire_value}))

  def test_sparse_combiner(self):
    # [FUNCTION DOC]
    # Scenario: hashed column with the mean combiner. Asserts that an example with two ids gets the
    # average of the two weights plus the bias while a single id example gets its weight.
    wire_cast = fc._categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default():
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar', 'stringer', 'marlo'],  # hashed to = [2, 0, 3]
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {'wire_cast': wire_tensor}
      predictions = fc.linear_model(
          features, [wire_cast], sparse_combiner='mean')
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      with _initialized_session() as sess:
        sess.run(wire_cast_var.assign([[10.], [100.], [1000.], [10000.]]))
        sess.run(bias.assign([5.]))
        self.assertAllClose([[1005.], [5010.]], self.evaluate(predictions))

  def test_sparse_combiner_with_negative_weights(self):
    # [FUNCTION DOC]
    # Scenario: a weighted categorical column with a negative weight and the sum combiner. Asserts the
    # prediction uses the signed weighted sum plus the bias.
    wire_cast = fc._categorical_column_with_hash_bucket('wire_cast', 4)
    wire_cast_weights = fc._weighted_categorical_column(wire_cast, 'weights')

    with ops.Graph().as_default():
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar', 'stringer', 'marlo'],  # hashed to = [2, 0, 3]
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {
          'wire_cast': wire_tensor,
          'weights': constant_op.constant([[1., 1., -1.0]])
      }
      predictions = fc.linear_model(
          features, [wire_cast_weights], sparse_combiner='sum')
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      with _initialized_session() as sess:
        sess.run(wire_cast_var.assign([[10.], [100.], [1000.], [10000.]]))
        sess.run(bias.assign([5.]))
        self.assertAllClose([[1005.], [-9985.]], self.evaluate(predictions))

  def test_dense_multi_dimension_multi_output(self):
    # [FUNCTION DOC]
    # Scenario: numeric column of shape two with three output units. Asserts shapes of bias and
    # weights and predictions computed after assigning weights and bias.
    price = fc._numeric_column('price', shape=2)
    with ops.Graph().as_default():
      features = {'price': [[1., 2.], [5., 6.]]}
      predictions = fc.linear_model(features, [price], units=3)
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        self.assertAllClose(np.zeros((3,)), self.evaluate(bias))
        self.assertAllClose(np.zeros((2, 3)), self.evaluate(price_var))
        sess.run(price_var.assign([[1., 2., 3.], [10., 100., 1000.]]))
        sess.run(bias.assign([2., 3., 4.]))
        self.assertAllClose([[23., 205., 2007.], [67., 613., 6019.]],
                            self.evaluate(predictions))

  def test_raises_if_shape_mismatch(self):
    # [FUNCTION DOC]
    # Asserts that feeding one value per example to a column declared with two values raises an error
    # mentioning that a tensor with two elements cannot be reshaped to two by two.
    price = fc._numeric_column('price', shape=2)
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      with self.assertRaisesRegex(
          Exception,
          r'Cannot reshape a tensor with 2 elements to shape \[2,2\]'):
        fc.linear_model(features, [price])

  def test_dense_reshaping(self):
    # [FUNCTION DOC]
    # Scenario: numeric column of shape one by two with a three-dimensional input. Asserts the input
    # is flattened so a two by one weight yields the expected predictions.
    price = fc._numeric_column('price', shape=[1, 2])
    with ops.Graph().as_default():
      features = {'price': [[[1., 2.]], [[5., 6.]]]}
      predictions = fc.linear_model(features, [price])
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        self.assertAllClose([[0.], [0.]], self.evaluate(price_var))
        self.assertAllClose([[0.], [0.]], self.evaluate(predictions))
        sess.run(price_var.assign([[10.], [100.]]))
        self.assertAllClose([[210.], [650.]], self.evaluate(predictions))

  def test_dense_multi_column(self):
    # [FUNCTION DOC]
    # Scenario: two numeric columns of different widths. Asserts separate weights per column and that
    # predictions sum the contributions of both columns plus the bias.
    price1 = fc._numeric_column('price1', shape=2)
    price2 = fc._numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': [[1., 2.], [5., 6.]],
          'price2': [[3.], [4.]]
      }
      predictions = fc.linear_model(features, [price1, price2])
      bias = get_linear_model_bias()
      price1_var = get_linear_model_column_var(price1)
      price2_var = get_linear_model_column_var(price2)
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        self.assertAllClose([[0.], [0.]], self.evaluate(price1_var))
        self.assertAllClose([[0.]], self.evaluate(price2_var))
        self.assertAllClose([[0.], [0.]], self.evaluate(predictions))
        sess.run(price1_var.assign([[10.], [100.]]))
        sess.run(price2_var.assign([[1000.]]))
        sess.run(bias.assign([7.]))
        self.assertAllClose([[3217.], [4657.]], self.evaluate(predictions))

  def test_fills_cols_to_vars(self):
    # [FUNCTION DOC]
    # Scenario: two numeric columns and a mapping passed as the columns to variables argument. Asserts
    # the mapping holds the bias variable under the bias key and each column weight variable as a
    # single element list.
    price1 = fc._numeric_column('price1', shape=2)
    price2 = fc._numeric_column('price2')
    with ops.Graph().as_default():
      features = {'price1': [[1., 2.], [5., 6.]], 'price2': [[3.], [4.]]}
      cols_to_vars = {}
      fc.linear_model(features, [price1, price2], cols_to_vars=cols_to_vars)
      bias = get_linear_model_bias()
      price1_var = get_linear_model_column_var(price1)
      price2_var = get_linear_model_column_var(price2)
      self.assertEqual(cols_to_vars['bias'], [bias])
      self.assertEqual(cols_to_vars[price1], [price1_var])
      self.assertEqual(cols_to_vars[price2], [price2_var])

  def test_fills_cols_to_vars_partitioned_variables(self):
    # [FUNCTION DOC]
    # Scenario: variables created under a fixed-size partitioner with two shards. Asserts the bias is
    # a single variable while each column weight is split into the expected shard shapes, with the
    # second shard of the three-wide column holding the remainder.
    price1 = fc._numeric_column('price1', shape=2)
    price2 = fc._numeric_column('price2', shape=3)
    with ops.Graph().as_default():
      features = {
          'price1': [[1., 2.], [6., 7.]],
          'price2': [[3., 4., 5.], [8., 9., 10.]]
      }
      cols_to_vars = {}
      with variable_scope.variable_scope(
          'linear',
          partitioner=partitioned_variables.fixed_size_partitioner(2, axis=0)):
        fc.linear_model(features, [price1, price2], cols_to_vars=cols_to_vars)
      with _initialized_session():
        self.assertEqual([0.], cols_to_vars['bias'][0].eval())
        # Partitioning shards the [2, 1] price1 var into 2 [1, 1] Variables.
        self.assertAllEqual([[0.]], cols_to_vars[price1][0])
        self.assertAllEqual([[0.]], cols_to_vars[price1][1])
        # Partitioning shards the [3, 1] price2 var into a [2, 1] Variable and
        # a [1, 1] Variable.
        self.assertAllEqual([[0.], [0.]], cols_to_vars[price2][0])
        self.assertAllEqual([[0.]], cols_to_vars[price2][1])

  def test_fills_cols_to_output_tensors(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test name and enclosing class suggest the linear model function is exercised,
    # but the body calls the input layer function and checks its output-tensor mapping, so it
    # duplicates input layer coverage rather than testing the linear model.
    # Scenario: a numeric column, a bucketized column and an embedding column are passed to the input
    # layer function with a mapping for columns to output tensors. Asserts the mapping holds exactly
    # the given columns and that its values are the tensors that were concatenated into the input
    # layer output.
    # Provide three _DenseColumn's to input_layer: a _NumericColumn, a
    # _BucketizedColumn, and an _EmbeddingColumn.  Only the _EmbeddingColumn
    # creates a Variable.
    apple_numeric_column = fc._numeric_column('apple_numeric_column')
    banana_dense_feature = fc._numeric_column('banana_dense_feature')
    banana_dense_feature_bucketized = fc._bucketized_column(
        banana_dense_feature, boundaries=[0.])
    cherry_sparse_column = fc._categorical_column_with_hash_bucket(
        'cherry_sparse_feature', hash_bucket_size=5)
    dragonfruit_embedding_column = fc._embedding_column(
        cherry_sparse_column, dimension=10)
    with ops.Graph().as_default():
      features = {
          'apple_numeric_column': [[3.], [4.]],
          'banana_dense_feature': [[-1.], [4.]],
          'cherry_sparse_feature': [['a'], ['x']],
      }
      cols_to_output_tensors = {}
      all_cols = [
          apple_numeric_column, banana_dense_feature_bucketized,
          dragonfruit_embedding_column
      ]
      input_layer = fc.input_layer(
          features, all_cols, cols_to_output_tensors=cols_to_output_tensors)

      # We check the mapping by checking that we have the right keys,
      # and that the values (output_tensors) were indeed the ones used to
      # form the input layer.
      self.assertCountEqual(all_cols, cols_to_output_tensors.keys())
      input_layer_inputs = [tensor for tensor in input_layer.op.inputs[:-1]]
      output_tensors = [tensor for tensor in cols_to_output_tensors.values()]
      self.assertCountEqual(input_layer_inputs, output_tensors)

  def test_dense_collection(self):
    # [FUNCTION DOC]
    # Asserts that when a weight collection name is passed, both the bias and the numeric column
    # weight are added to that collection.
    price = fc._numeric_column('price')
    with ops.Graph().as_default() as g:
      features = {'price': [[1.], [5.]]}
      fc.linear_model(features, [price], weight_collections=['my-vars'])
      my_vars = g.get_collection('my-vars')
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      self.assertIn(bias, my_vars)
      self.assertIn(price_var, my_vars)

  def test_sparse_collection(self):
    # [FUNCTION DOC]
    # Asserts that when a weight collection name is passed, both the bias and the hashed column weight
    # are added to that collection.
    wire_cast = fc._categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default() as g:
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar'], indices=[[0, 0]], dense_shape=[1, 1])
      features = {'wire_cast': wire_tensor}
      fc.linear_model(
          features, [wire_cast], weight_collections=['my-vars'])
      my_vars = g.get_collection('my-vars')
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      self.assertIn(bias, my_vars)
      self.assertIn(wire_cast_var, my_vars)

  def test_dense_trainable_default(self):
    # [FUNCTION DOC]
    # Asserts that by default the bias and numeric column weights appear in the trainable variables
    # collection.
    price = fc._numeric_column('price')
    with ops.Graph().as_default() as g:
      features = {'price': [[1.], [5.]]}
      fc.linear_model(features, [price])
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      trainable_vars = g.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      self.assertIn(bias, trainable_vars)
      self.assertIn(price_var, trainable_vars)

  def test_sparse_trainable_default(self):
    # [FUNCTION DOC]
    # Asserts that by default the bias and hashed column weights appear in the trainable variables
    # collection.
    wire_cast = fc._categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default() as g:
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar'], indices=[[0, 0]], dense_shape=[1, 1])
      features = {'wire_cast': wire_tensor}
      fc.linear_model(features, [wire_cast])
      trainable_vars = g.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      self.assertIn(bias, trainable_vars)
      self.assertIn(wire_cast_var, trainable_vars)

  def test_dense_trainable_false(self):
    # [FUNCTION DOC]
    # Asserts that with trainable set to false, the trainable variables collection stays empty for a
    # numeric column.
    price = fc._numeric_column('price')
    with ops.Graph().as_default() as g:
      features = {'price': [[1.], [5.]]}
      fc.linear_model(features, [price], trainable=False)
      trainable_vars = g.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      self.assertEqual([], trainable_vars)

  def test_sparse_trainable_false(self):
    # [FUNCTION DOC]
    # Asserts that with trainable set to false, the trainable variables collection stays empty for a
    # hashed column.
    wire_cast = fc._categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default() as g:
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar'], indices=[[0, 0]], dense_shape=[1, 1])
      features = {'wire_cast': wire_tensor}
      fc.linear_model(features, [wire_cast], trainable=False)
      trainable_vars = g.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      self.assertEqual([], trainable_vars)

  def test_column_order(self):
    # [FUNCTION DOC]
    # Builds the linear model twice in separate graphs with the same columns passed in different
    # orders and asserts that variables in the named collection are always ordered by column name
    # (price a, price b, wire cast), independent of input order.
    price_a = fc._numeric_column('price_a')
    price_b = fc._numeric_column('price_b')
    wire_cast = fc._categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default() as g:
      features = {
          'price_a': [[1.]],
          'price_b': [[3.]],
          'wire_cast':
              sparse_tensor.SparseTensor(
                  values=['omar'], indices=[[0, 0]], dense_shape=[1, 1])
      }
      fc.linear_model(
          features, [price_a, wire_cast, price_b],
          weight_collections=['my-vars'])
      my_vars = g.get_collection('my-vars')
      self.assertIn('price_a', my_vars[0].name)
      self.assertIn('price_b', my_vars[1].name)
      self.assertIn('wire_cast', my_vars[2].name)

    with ops.Graph().as_default() as g:
      features = {
          'price_a': [[1.]],
          'price_b': [[3.]],
          'wire_cast':
              sparse_tensor.SparseTensor(
                  values=['omar'], indices=[[0, 0]], dense_shape=[1, 1])
      }
      fc.linear_model(
          features, [wire_cast, price_b, price_a],
          weight_collections=['my-vars'])
      my_vars = g.get_collection('my-vars')
      self.assertIn('price_a', my_vars[0].name)
      self.assertIn('price_b', my_vars[1].name)
      self.assertIn('wire_cast', my_vars[2].name)

  def test_static_batch_size_mismatch(self):
    # [FUNCTION DOC]
    # Two numeric features with statically known and different batch sizes (three and two). Asserts
    # the linear model function raises a value error saying the batch size of each feature must be the
    # same.
    price1 = fc._numeric_column('price1')
    price2 = fc._numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': [[1.], [5.], [7.]],  # batchsize = 3
          'price2': [[3.], [4.]]  # batchsize = 2
      }
    with self.assertRaisesRegex(
        ValueError,
        r'Batch size \(first dimension\) of each feature must be same.'):
      fc.linear_model(features, [price1, price2])

  def test_subset_of_static_batch_size_mismatch(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the inline comment labeling the placeholder feature as having batch size three is
    # misleading, because that placeholder has an unknown shape; only the two static features have
    # known batch sizes.
    # Three numeric features where one is an unknown-shape placeholder and two have statically
    # different batch sizes. Asserts a value error about mismatching batch sizes is raised even though
    # only a subset of features have static sizes.
    price1 = fc._numeric_column('price1')
    price2 = fc._numeric_column('price2')
    price3 = fc._numeric_column('price3')
    with ops.Graph().as_default():
      features = {
          'price1': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 3
          'price2': [[3.], [4.]],  # batchsize = 2
          'price3': [[3.], [4.], [5.]]  # batchsize = 3
      }
      with self.assertRaisesRegex(
          ValueError,
          r'Batch size \(first dimension\) of each feature must be same.'):  # pylint: disable=anomalous-backslash-in-string
        fc.linear_model(features, [price1, price2, price3])

  def test_runtime_batch_size_mismatch(self):
    # [FUNCTION DOC]
    # One placeholder feature and one static feature of batch size two. Builds the model, then runs it
    # feeding three rows for the placeholder and asserts a runtime op error saying the tensors must
    # have the same size and shape.
    price1 = fc._numeric_column('price1')
    price2 = fc._numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 3
          'price2': [[3.], [4.]]  # batchsize = 2
      }
      predictions = fc.linear_model(features, [price1, price2])
      with _initialized_session() as sess:
        with self.assertRaisesRegex(errors.OpError,
                                    'must have the same size and shape'):
          sess.run(
              predictions, feed_dict={features['price1']: [[1.], [5.], [7.]]})

  def test_runtime_batch_size_matches(self):
    # [FUNCTION DOC]
    # Two placeholder features fed at run time with matching batch sizes; asserts the model runs
    # without error.
    price1 = fc._numeric_column('price1')
    price2 = fc._numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 2
          'price2': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 2
      }
      predictions = fc.linear_model(features, [price1, price2])
      with _initialized_session() as sess:
        sess.run(
            predictions,
            feed_dict={
                features['price1']: [[1.], [5.]],
                features['price2']: [[1.], [5.]],
            })

  @test_util.run_deprecated_v1
  # FeatureColumnV1 uses variable scope etc. which is TF1.
  def test_with_1d_sparse_tensor(self):
    # [FUNCTION DOC]
    # Scenario (graph mode, decorated to run the deprecated v1 way): a bucketized price from a rank
    # one dense tensor and a vocabulary list column from a rank one sparse tensor. Asserts the inputs
    # have rank one, then assigns weights and bias and asserts the predictions combine bucket and
    # vocabulary weights.
    price = fc._numeric_column('price')
    price_buckets = fc._bucketized_column(
        price, boundaries=[
            0.,
            10.,
            100.,
        ])
    body_style = fc._categorical_column_with_vocabulary_list(
        'body-style', vocabulary_list=['hardtop', 'wagon', 'sedan'])

    # Provides 1-dim tensor and dense tensor.
    features = {
        'price': constant_op.constant([-1., 12.,]),
        'body-style': sparse_tensor.SparseTensor(
            indices=((0,), (1,)),
            values=('sedan', 'hardtop'),
            dense_shape=(2,)),
    }
    self.assertEqual(1, features['price'].shape.ndims)
    self.assertEqual(1, features['body-style'].dense_shape.get_shape()[0])

    net = fc.linear_model(features, [price_buckets, body_style])
    with _initialized_session() as sess:
      bias = get_linear_model_bias()
      price_buckets_var = get_linear_model_column_var(price_buckets)
      body_style_var = get_linear_model_column_var(body_style)

      sess.run(price_buckets_var.assign([[10.], [100.], [1000.], [10000.]]))
      sess.run(body_style_var.assign([[-10.], [-100.], [-1000.]]))
      sess.run(bias.assign([5.]))

      self.assertAllClose([[10 - 1000 + 5.], [1000 - 10 + 5.]],
                          self.evaluate(net))

  @test_util.run_deprecated_v1
  # Placeholders are TF1. Replacing with tf.function not feasible because of V1
  # variable creation.
  def test_with_1d_unknown_shape_sparse_tensor(self):
    # [FUNCTION DOC]
    # Scenario (v1 graph mode): placeholders of unknown rank for a numeric price, a sparse body style
    # and a dense country feature. Asserts the shapes are unknown, builds the model, assigns weights
    # and bias and checks predictions after feeding rank one data.
    price = fc._numeric_column('price')
    price_buckets = fc._bucketized_column(
        price, boundaries=[
            0.,
            10.,
            100.,
        ])
    body_style = fc._categorical_column_with_vocabulary_list(
        'body-style', vocabulary_list=['hardtop', 'wagon', 'sedan'])
    country = fc._categorical_column_with_vocabulary_list(
        'country', vocabulary_list=['US', 'JP', 'CA'])

    # Provides 1-dim tensor and dense tensor.
    features = {
        'price': array_ops.placeholder(dtypes.float32),
        'body-style': array_ops.sparse_placeholder(dtypes.string),
        'country': array_ops.placeholder(dtypes.string),
    }
    self.assertIsNone(features['price'].shape.ndims)
    self.assertIsNone(features['body-style'].get_shape().ndims)

    price_data = np.array([-1., 12.])
    body_style_data = sparse_tensor.SparseTensorValue(
        indices=((0,), (1,)),
        values=('sedan', 'hardtop'),
        dense_shape=(2,))
    country_data = np.array(['US', 'CA'])

    net = fc.linear_model(features, [price_buckets, body_style, country])
    bias = get_linear_model_bias()
    price_buckets_var = get_linear_model_column_var(price_buckets)
    body_style_var = get_linear_model_column_var(body_style)
    with _initialized_session() as sess:
      sess.run(price_buckets_var.assign([[10.], [100.], [1000.], [10000.]]))
      sess.run(body_style_var.assign([[-10.], [-100.], [-1000.]]))
      sess.run(bias.assign([5.]))

      self.assertAllClose([[10 - 1000 + 5.], [1000 - 10 + 5.]],
                          sess.run(
                              net,
                              feed_dict={
                                  features['price']: price_data,
                                  features['body-style']: body_style_data,
                                  features['country']: country_data
                              }))

  @test_util.run_deprecated_v1
  # Placeholders are TF1. Replacing with tf.function not feasible because of V1
  # variable creation.
  def test_with_rank_0_feature(self):
    # [FUNCTION DOC]
    # Scenario (v1 graph mode): a scalar numeric feature. Asserts a value error saying the feature
    # cannot have rank zero for a statically known rank, and for a placeholder of unknown rank asserts
    # a runtime error with the same message when a scalar is fed.
    price = fc._numeric_column('price')
    features = {
        'price': constant_op.constant(0),
    }
    self.assertEqual(0, features['price'].shape.ndims)

    # Static rank 0 should fail
    with self.assertRaisesRegex(ValueError, 'Feature .* cannot have rank 0'):
      fc.linear_model(features, [price])

    # Dynamic rank 0 should fail
    features = {
        'price': array_ops.placeholder(dtypes.float32),
    }
    net = fc.linear_model(features, [price])
    self.assertEqual(1, net.shape[1])
    with _initialized_session() as sess:
      with self.assertRaisesOpError('Feature .* cannot have rank 0'):
        sess.run(net, feed_dict={features['price']: np.array(1)})

  def test_multiple_linear_models(self):
    # [FUNCTION DOC]
    # Builds two linear models in the same graph on different features. Asserts the second model gets
    # variables in a uniquified scope, that each has its own bias and weight, and that each predicts
    # according to its own assigned values.
    price = fc._numeric_column('price')
    with ops.Graph().as_default():
      features1 = {'price': [[1.], [5.]]}
      features2 = {'price': [[2.], [10.]]}
      predictions1 = fc.linear_model(features1, [price])
      predictions2 = fc.linear_model(features2, [price])
      bias1 = get_linear_model_bias(name='linear_model')
      bias2 = get_linear_model_bias(name='linear_model_1')
      price_var1 = get_linear_model_column_var(price, name='linear_model')
      price_var2 = get_linear_model_column_var(price, name='linear_model_1')
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias1))
        sess.run(price_var1.assign([[10.]]))
        sess.run(bias1.assign([5.]))
        self.assertAllClose([[15.], [55.]], self.evaluate(predictions1))
        self.assertAllClose([0.], self.evaluate(bias2))
        sess.run(price_var2.assign([[10.]]))
        sess.run(bias2.assign([5.]))
        self.assertAllClose([[25.], [105.]], self.evaluate(predictions2))


class _LinearModelTest(test.TestCase):
  # [CLASS DOC]
  # Tests the Keras-style linear model class through a helper wrapper. Mirrors the tests of the
  # linear model function: validation, bias and weights, multi output, combiners, reshaping,
  # variable collections, trainability, column ordering, batch size checks and rank handling.

  def test_raises_if_empty_feature_columns(self):
    # [FUNCTION DOC]
    # Asserts that an empty feature column list raises a value error saying the columns must not be
    # empty.
    with self.assertRaisesRegex(ValueError,
                                'feature_columns must not be empty'):
      get_keras_linear_model_predictions(features={}, feature_columns=[])

  def test_should_be_feature_column(self):
    # [FUNCTION DOC]
    # Asserts that passing a string instead of feature columns raises a value error saying it must be
    # a feature column.
    with self.assertRaisesRegex(ValueError, 'must be a _FeatureColumn'):
      get_keras_linear_model_predictions(
          features={'a': [[0]]}, feature_columns='NotSupported')

  def test_should_be_dense_or_categorical_column(self):
    # [FUNCTION DOC]
    # Scenario: a fixture column that is neither dense nor categorical passed to the Keras-style
    # linear model; asserts a value error saying a dense or categorical column is required.

    class NotSupportedColumn(_FeatureColumn):
      # [CLASS DOC]
      # Local fixture feature column that is neither dense nor categorical.

      @property
      def name(self):
        # [FUNCTION DOC]
        # Property returning the fixed fixture name; returns a string.
        return 'NotSupportedColumn'

      def _transform_feature(self, cache):
        # [FUNCTION DOC]
        # Transform hook of the fixture; does nothing and returns nothing.
        pass

      @property
      def _parse_example_spec(self):
        # [FUNCTION DOC]
        # Property required by the interface; returns nothing.
        pass

    with self.assertRaisesRegex(
        ValueError, 'must be either a _DenseColumn or _CategoricalColumn'):
      get_keras_linear_model_predictions(
          features={'a': [[0]]}, feature_columns=[NotSupportedColumn()])

  def test_does_not_support_dict_columns(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: this test calls the linear model function instead of the Keras-style linear model
    # wrapper used by the other tests in this class, so it duplicates the function based test and does
    # not test the Keras-style class.
    with self.assertRaisesRegex(
        ValueError, 'Expected feature_columns to be iterable, found dict.'):
      fc.linear_model(
          features={'a': [[0]]}, feature_columns={'a': fc._numeric_column('a')})

  def test_raises_if_duplicate_name(self):
    # [FUNCTION DOC]
    # Asserts that two numeric columns with the same name passed to the Keras-style linear model raise
    # a value error about duplicate names.
    with self.assertRaisesRegex(
        ValueError, 'Duplicate feature column name found for columns'):
      get_keras_linear_model_predictions(
          features={'a': [[0]]},
          feature_columns=[fc._numeric_column('a'),
                           fc._numeric_column('a')])

  def test_dense_bias(self):
    # [FUNCTION DOC]
    # Numeric price column: asserts zero bias, then assigned weight and bias produce the expected
    # predictions.
    price = fc._numeric_column('price')
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      predictions = get_keras_linear_model_predictions(features, [price])
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        sess.run(price_var.assign([[10.]]))
        sess.run(bias.assign([5.]))
        self.assertAllClose([[15.], [55.]], self.evaluate(predictions))

  def test_sparse_bias(self):
    # [FUNCTION DOC]
    # Four-bucket hashed column: asserts zero initial state and that predictions equal selected
    # weights plus bias after assignment.
    wire_cast = fc._categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default():
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar', 'stringer', 'marlo'],  # hashed to = [2, 0, 3]
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {'wire_cast': wire_tensor}
      predictions = get_keras_linear_model_predictions(features, [wire_cast])
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        self.assertAllClose([[0.], [0.], [0.], [0.]],
                            self.evaluate(wire_cast_var))
        sess.run(wire_cast_var.assign([[10.], [100.], [1000.], [10000.]]))
        sess.run(bias.assign([5.]))
        self.assertAllClose([[1005.], [10015.]], self.evaluate(predictions))

  def test_dense_and_sparse_bias(self):
    # [FUNCTION DOC]
    # One hashed and one numeric column together: asserts combined predictions after assigning weights
    # and bias.
    wire_cast = fc._categorical_column_with_hash_bucket('wire_cast', 4)
    price = fc._numeric_column('price')
    with ops.Graph().as_default():
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar', 'stringer', 'marlo'],  # hashed to = [2, 0, 3]
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {'wire_cast': wire_tensor, 'price': [[1.], [5.]]}
      predictions = get_keras_linear_model_predictions(features,
                                                       [wire_cast, price])
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        sess.run(wire_cast_var.assign([[10.], [100.], [1000.], [10000.]]))
        sess.run(bias.assign([5.]))
        sess.run(price_var.assign([[10.]]))
        self.assertAllClose([[1015.], [10065.]], self.evaluate(predictions))

  def test_dense_and_sparse_column(self):
    # [FUNCTION DOC]
    # Fixture column that is both dense and categorical: asserts only the sparse path is used and
    # predictions equal the selected weights plus bias.
    """When the column is both dense and sparse, uses sparse tensors."""

    class _DenseAndSparseColumn(_DenseColumn, _CategoricalColumn):
      # [CLASS DOC]
      # Local fixture column that is both dense and categorical; dense accessors raise errors, the
      # sparse accessor returns fixed ids.

      @property
      def name(self):
        # [FUNCTION DOC]
        # Property returning the fixed fixture name; returns a string.
        return 'dense_and_sparse_column'

      @property
      def _parse_example_spec(self):
        # [FUNCTION DOC]
        # Property returning a variable length feature spec keyed by the column name with the column
        # dtype.
        return {self.name: parsing_ops.VarLenFeature(self.dtype)}

      def _transform_feature(self, inputs):
        # [FUNCTION DOC]
        # Transform hook returning the raw feature for the column name from the given inputs.
        return inputs.get(self.name)

      @property
      def _variable_shape(self):
        # [FUNCTION DOC]
        # Dense interface property; always raises a value error saying it should not be used.
        raise ValueError('Should not use this method.')

      def _get_dense_tensor(self,
                            inputs,
                            weight_collections=None,
                            trainable=None):
        # [FUNCTION DOC]
        # Dense interface method; always raises a value error saying it should not be used. Parameters are
        # ignored.
        raise ValueError('Should not use this method.')

      @property
      def _num_buckets(self):
        # [FUNCTION DOC]
        # Property returning the constant bucket count four.
        return 4

      def _get_sparse_tensors(self,
                              inputs,
                              weight_collections=None,
                              trainable=None):
        # [FUNCTION DOC]
        # Returns an id and weight pair with a fixed sparse id tensor (ids two, zero, three) and no
        # weights; parameters are ignored.
        sp_tensor = sparse_tensor.SparseTensor(
            indices=[[0, 0], [1, 0], [1, 1]],
            values=[2, 0, 3],
            dense_shape=[2, 2])
        return _CategoricalColumn.IdWeightPair(sp_tensor, None)

    dense_and_sparse_column = _DenseAndSparseColumn()
    with ops.Graph().as_default():
      sp_tensor = sparse_tensor.SparseTensor(
          values=['omar', 'stringer', 'marlo'],
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {dense_and_sparse_column.name: sp_tensor}
      predictions = get_keras_linear_model_predictions(
          features, [dense_and_sparse_column])
      bias = get_linear_model_bias()
      dense_and_sparse_column_var = get_linear_model_column_var(
          dense_and_sparse_column)
      with _initialized_session() as sess:
        sess.run(
            dense_and_sparse_column_var.assign([[10.], [100.], [1000.],
                                                [10000.]]))
        sess.run(bias.assign([5.]))
        self.assertAllClose([[1005.], [10015.]], self.evaluate(predictions))

  def test_dense_multi_output(self):
    # [FUNCTION DOC]
    # Numeric column with three output units: asserts shapes and the three-wide predictions after
    # assignment.
    price = fc._numeric_column('price')
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      predictions = get_keras_linear_model_predictions(
          features, [price], units=3)
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        self.assertAllClose(np.zeros((3,)), self.evaluate(bias))
        self.assertAllClose(np.zeros((1, 3)), self.evaluate(price_var))
        sess.run(price_var.assign([[10., 100., 1000.]]))
        sess.run(bias.assign([5., 6., 7.]))
        self.assertAllClose([[15., 106., 1007.], [55., 506., 5007.]],
                            self.evaluate(predictions))

  def test_sparse_multi_output(self):
    # [FUNCTION DOC]
    # Hashed column with three output units: asserts shapes and that each prediction row sums selected
    # weight rows plus bias.
    wire_cast = fc._categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default():
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar', 'stringer', 'marlo'],  # hashed to = [2, 0, 3]
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {'wire_cast': wire_tensor}
      predictions = get_keras_linear_model_predictions(
          features, [wire_cast], units=3)
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      with _initialized_session() as sess:
        self.assertAllClose(np.zeros((3,)), self.evaluate(bias))
        self.assertAllClose(np.zeros((4, 3)), self.evaluate(wire_cast_var))
        sess.run(
            wire_cast_var.assign([[10., 11., 12.], [100., 110., 120.],
                                  [1000., 1100.,
                                   1200.], [10000., 11000., 12000.]]))
        sess.run(bias.assign([5., 6., 7.]))
        self.assertAllClose([[1005., 1106., 1207.], [10015., 11017., 12019.]],
                            self.evaluate(predictions))

  def test_dense_multi_dimension(self):
    # [FUNCTION DOC]
    # Numeric column of shape two: asserts a two by one weight and dot product predictions after
    # assignment.
    price = fc._numeric_column('price', shape=2)
    with ops.Graph().as_default():
      features = {'price': [[1., 2.], [5., 6.]]}
      predictions = get_keras_linear_model_predictions(features, [price])
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        self.assertAllClose([[0.], [0.]], self.evaluate(price_var))
        sess.run(price_var.assign([[10.], [100.]]))
        self.assertAllClose([[210.], [650.]], self.evaluate(predictions))

  def test_sparse_multi_rank(self):
    # [FUNCTION DOC]
    # Rank three sparse placeholder input: asserts zero predictions initially and sums of hashed id
    # weights after assigning.
    wire_cast = fc._categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default():
      wire_tensor = array_ops.sparse_placeholder(dtypes.string)
      wire_value = sparse_tensor.SparseTensorValue(
          values=['omar', 'stringer', 'marlo', 'omar'],  # hashed = [2, 0, 3, 2]
          indices=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 1]],
          dense_shape=[2, 2, 2])
      features = {'wire_cast': wire_tensor}
      predictions = get_keras_linear_model_predictions(features, [wire_cast])
      wire_cast_var = get_linear_model_column_var(wire_cast)
      with _initialized_session() as sess:
        self.assertAllClose(np.zeros((4, 1)), self.evaluate(wire_cast_var))
        self.assertAllClose(
            np.zeros((2, 1)),
            predictions.eval(feed_dict={wire_tensor: wire_value}))
        sess.run(wire_cast_var.assign([[10.], [100.], [1000.], [10000.]]))
        self.assertAllClose(
            [[1010.], [11000.]],
            predictions.eval(feed_dict={wire_tensor: wire_value}))

  def test_sparse_combiner(self):
    # [FUNCTION DOC]
    # Hashed column with the mean combiner: asserts the average of weights for a multi id example plus
    # the bias.
    wire_cast = fc._categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default():
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar', 'stringer', 'marlo'],  # hashed to = [2, 0, 3]
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {'wire_cast': wire_tensor}
      predictions = get_keras_linear_model_predictions(
          features, [wire_cast], sparse_combiner='mean')
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      with _initialized_session() as sess:
        sess.run(wire_cast_var.assign([[10.], [100.], [1000.], [10000.]]))
        sess.run(bias.assign([5.]))
        self.assertAllClose([[1005.], [5010.]], self.evaluate(predictions))

  def test_dense_multi_dimension_multi_output(self):
    # [FUNCTION DOC]
    # Numeric column of shape two with three output units: asserts shapes and predictions after
    # assigning weights and bias.
    price = fc._numeric_column('price', shape=2)
    with ops.Graph().as_default():
      features = {'price': [[1., 2.], [5., 6.]]}
      predictions = get_keras_linear_model_predictions(
          features, [price], units=3)
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        self.assertAllClose(np.zeros((3,)), self.evaluate(bias))
        self.assertAllClose(np.zeros((2, 3)), self.evaluate(price_var))
        sess.run(price_var.assign([[1., 2., 3.], [10., 100., 1000.]]))
        sess.run(bias.assign([2., 3., 4.]))
        self.assertAllClose([[23., 205., 2007.], [67., 613., 6019.]],
                            self.evaluate(predictions))

  def test_raises_if_shape_mismatch(self):
    # [FUNCTION DOC]
    # Asserts that an input with one value per example for a column of shape two raises an error about
    # reshaping two elements to two by two.
    price = fc._numeric_column('price', shape=2)
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      with self.assertRaisesRegex(
          Exception,
          r'Cannot reshape a tensor with 2 elements to shape \[2,2\]'):
        get_keras_linear_model_predictions(features, [price])

  def test_dense_reshaping(self):
    # [FUNCTION DOC]
    # Numeric column of shape one by two with three-dimensional input: asserts flattening and expected
    # predictions.
    price = fc._numeric_column('price', shape=[1, 2])
    with ops.Graph().as_default():
      features = {'price': [[[1., 2.]], [[5., 6.]]]}
      predictions = get_keras_linear_model_predictions(features, [price])
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        self.assertAllClose([[0.], [0.]], self.evaluate(price_var))
        self.assertAllClose([[0.], [0.]], self.evaluate(predictions))
        sess.run(price_var.assign([[10.], [100.]]))
        self.assertAllClose([[210.], [650.]], self.evaluate(predictions))

  def test_dense_multi_column(self):
    # [FUNCTION DOC]
    # Two numeric columns: asserts separate weights and combined predictions plus bias.
    price1 = fc._numeric_column('price1', shape=2)
    price2 = fc._numeric_column('price2')
    with ops.Graph().as_default():
      features = {'price1': [[1., 2.], [5., 6.]], 'price2': [[3.], [4.]]}
      predictions = get_keras_linear_model_predictions(features,
                                                       [price1, price2])
      bias = get_linear_model_bias()
      price1_var = get_linear_model_column_var(price1)
      price2_var = get_linear_model_column_var(price2)
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        self.assertAllClose([[0.], [0.]], self.evaluate(price1_var))
        self.assertAllClose([[0.]], self.evaluate(price2_var))
        self.assertAllClose([[0.], [0.]], self.evaluate(predictions))
        sess.run(price1_var.assign([[10.], [100.]]))
        sess.run(price2_var.assign([[1000.]]))
        sess.run(bias.assign([7.]))
        self.assertAllClose([[3217.], [4657.]], self.evaluate(predictions))

  def test_fills_cols_to_vars(self):
    # [FUNCTION DOC]
    # Asserts the mapping passed for columns to variables is filled with the bias variable and each
    # column weight variable.
    price1 = fc._numeric_column('price1', shape=2)
    price2 = fc._numeric_column('price2')
    with ops.Graph().as_default():
      features = {'price1': [[1., 2.], [5., 6.]], 'price2': [[3.], [4.]]}
      cols_to_vars = {}
      get_keras_linear_model_predictions(
          features, [price1, price2], cols_to_vars=cols_to_vars)
      bias = get_linear_model_bias()
      price1_var = get_linear_model_column_var(price1)
      price2_var = get_linear_model_column_var(price2)
      self.assertEqual(cols_to_vars['bias'], [bias])
      self.assertEqual(cols_to_vars[price1], [price1_var])
      self.assertEqual(cols_to_vars[price2], [price2_var])

  def test_fills_cols_to_vars_partitioned_variables(self):
    # [FUNCTION DOC]
    # Under a two-shard fixed-size partitioner, asserts the bias is a single variable and the column
    # weights are split into the expected shard shapes.
    price1 = fc._numeric_column('price1', shape=2)
    price2 = fc._numeric_column('price2', shape=3)
    with ops.Graph().as_default():
      features = {
          'price1': [[1., 2.], [6., 7.]],
          'price2': [[3., 4., 5.], [8., 9., 10.]]
      }
      cols_to_vars = {}
      with variable_scope.variable_scope(
          'linear',
          partitioner=partitioned_variables.fixed_size_partitioner(2, axis=0)):
        get_keras_linear_model_predictions(
            features, [price1, price2], cols_to_vars=cols_to_vars)
      with _initialized_session():
        self.assertEqual([0.], cols_to_vars['bias'][0].eval())
        # Partitioning shards the [2, 1] price1 var into 2 [1, 1] Variables.
        self.assertAllEqual([[0.]], cols_to_vars[price1][0])
        self.assertAllEqual([[0.]], cols_to_vars[price1][1])
        # Partitioning shards the [3, 1] price2 var into a [2, 1] Variable and
        # a [1, 1] Variable.
        self.assertAllEqual([[0.], [0.]], cols_to_vars[price2][0])
        self.assertAllEqual([[0.]], cols_to_vars[price2][1])

  def test_dense_collection(self):
    # [FUNCTION DOC]
    # Asserts the bias and numeric weights are added to the given weight collection.
    price = fc._numeric_column('price')
    with ops.Graph().as_default() as g:
      features = {'price': [[1.], [5.]]}
      get_keras_linear_model_predictions(
          features, [price], weight_collections=['my-vars'])
      my_vars = g.get_collection('my-vars')
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      self.assertIn(bias, my_vars)
      self.assertIn(price_var, my_vars)

  def test_sparse_collection(self):
    # [FUNCTION DOC]
    # Asserts the bias and hashed column weights are added to the given weight collection.
    wire_cast = fc._categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default() as g:
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar'], indices=[[0, 0]], dense_shape=[1, 1])
      features = {'wire_cast': wire_tensor}
      get_keras_linear_model_predictions(
          features, [wire_cast], weight_collections=['my-vars'])
      my_vars = g.get_collection('my-vars')
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      self.assertIn(bias, my_vars)
      self.assertIn(wire_cast_var, my_vars)

  def test_dense_trainable_default(self):
    # [FUNCTION DOC]
    # Asserts the bias and numeric weights are trainable by default.
    price = fc._numeric_column('price')
    with ops.Graph().as_default() as g:
      features = {'price': [[1.], [5.]]}
      get_keras_linear_model_predictions(features, [price])
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      trainable_vars = g.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      self.assertIn(bias, trainable_vars)
      self.assertIn(price_var, trainable_vars)

  def test_sparse_trainable_default(self):
    # [FUNCTION DOC]
    # Asserts the bias and hashed column weights are trainable by default.
    wire_cast = fc._categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default() as g:
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar'], indices=[[0, 0]], dense_shape=[1, 1])
      features = {'wire_cast': wire_tensor}
      get_keras_linear_model_predictions(features, [wire_cast])
      trainable_vars = g.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      self.assertIn(bias, trainable_vars)
      self.assertIn(wire_cast_var, trainable_vars)

  def test_dense_trainable_false(self):
    # [FUNCTION DOC]
    # Asserts no trainable variables exist when trainable is false for a numeric column.
    price = fc._numeric_column('price')
    with ops.Graph().as_default() as g:
      features = {'price': [[1.], [5.]]}
      get_keras_linear_model_predictions(features, [price], trainable=False)
      trainable_vars = g.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      self.assertEqual([], trainable_vars)

  def test_sparse_trainable_false(self):
    # [FUNCTION DOC]
    # Asserts no trainable variables exist when trainable is false for a hashed column.
    wire_cast = fc._categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default() as g:
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar'], indices=[[0, 0]], dense_shape=[1, 1])
      features = {'wire_cast': wire_tensor}
      get_keras_linear_model_predictions(features, [wire_cast], trainable=False)
      trainable_vars = g.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      self.assertEqual([], trainable_vars)

  def test_column_order(self):
    # [FUNCTION DOC]
    # Builds the model twice with differently ordered columns and asserts variables in the weight
    # collection are always ordered by column name.
    price_a = fc._numeric_column('price_a')
    price_b = fc._numeric_column('price_b')
    wire_cast = fc._categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default() as g:
      features = {
          'price_a': [[1.]],
          'price_b': [[3.]],
          'wire_cast':
              sparse_tensor.SparseTensor(
                  values=['omar'], indices=[[0, 0]], dense_shape=[1, 1])
      }
      get_keras_linear_model_predictions(
          features, [price_a, wire_cast, price_b],
          weight_collections=['my-vars'])
      my_vars = g.get_collection('my-vars')
      self.assertIn('price_a', my_vars[0].name)
      self.assertIn('price_b', my_vars[1].name)
      self.assertIn('wire_cast', my_vars[2].name)

    with ops.Graph().as_default() as g:
      features = {
          'price_a': [[1.]],
          'price_b': [[3.]],
          'wire_cast':
              sparse_tensor.SparseTensor(
                  values=['omar'], indices=[[0, 0]], dense_shape=[1, 1])
      }
      get_keras_linear_model_predictions(
          features, [wire_cast, price_b, price_a],
          weight_collections=['my-vars'])
      my_vars = g.get_collection('my-vars')
      self.assertIn('price_a', my_vars[0].name)
      self.assertIn('price_b', my_vars[1].name)
      self.assertIn('wire_cast', my_vars[2].name)

  def test_static_batch_size_mismatch(self):
    # [FUNCTION DOC]
    # Two numeric features with static batch sizes three and two; asserts a value error saying batch
    # sizes must be the same.
    price1 = fc._numeric_column('price1')
    price2 = fc._numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': [[1.], [5.], [7.]],  # batchsize = 3
          'price2': [[3.], [4.]]  # batchsize = 2
      }
    with self.assertRaisesRegex(
        ValueError,
        r'Batch size \(first dimension\) of each feature must be same.'):  # pylint: disable=anomalous-backslash-in-string
      get_keras_linear_model_predictions(features, [price1, price2])

  def test_subset_of_static_batch_size_mismatch(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the inline comment labeling the placeholder feature as having batch size three is
    # misleading, because that placeholder has an unknown shape.
    # One unknown-shape placeholder and two statically mismatched features; asserts a value error
    # about mismatching batch sizes.
    price1 = fc._numeric_column('price1')
    price2 = fc._numeric_column('price2')
    price3 = fc._numeric_column('price3')
    with ops.Graph().as_default():
      features = {
          'price1': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 3
          'price2': [[3.], [4.]],  # batchsize = 2
          'price3': [[3.], [4.], [5.]]  # batchsize = 3
      }
      with self.assertRaisesRegex(
          ValueError,
          r'Batch size \(first dimension\) of each feature must be same.'):  # pylint: disable=anomalous-backslash-in-string
        get_keras_linear_model_predictions(features, [price1, price2, price3])

  def test_runtime_batch_size_mismatch(self):
    # [FUNCTION DOC]
    # Placeholder fed with three rows against a static feature of two rows; asserts a runtime op error
    # saying the tensors must have the same size and shape.
    price1 = fc._numeric_column('price1')
    price2 = fc._numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 3
          'price2': [[3.], [4.]]  # batchsize = 2
      }
      predictions = get_keras_linear_model_predictions(features,
                                                       [price1, price2])
      with _initialized_session() as sess:
        with self.assertRaisesRegex(errors.OpError,
                                    'must have the same size and shape'):
          sess.run(
              predictions, feed_dict={features['price1']: [[1.], [5.], [7.]]})

  def test_runtime_batch_size_matches(self):
    # [FUNCTION DOC]
    # Two placeholders fed with matching batch sizes; asserts the model runs without error.
    price1 = fc._numeric_column('price1')
    price2 = fc._numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 2
          'price2': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 2
      }
      predictions = get_keras_linear_model_predictions(features,
                                                       [price1, price2])
      with _initialized_session() as sess:
        sess.run(
            predictions,
            feed_dict={
                features['price1']: [[1.], [5.]],
                features['price2']: [[1.], [5.]],
            })

  @test_util.run_deprecated_v1
  # FeatureColumnV1 uses variable scope etc. which is TF1.
  def test_with_1d_sparse_tensor(self):
    # [FUNCTION DOC]
    # Scenario (v1 graph mode): rank one dense price feature bucketized, and a rank one sparse
    # vocabulary list feature. Asserts predictions combine bucket and vocabulary weights plus bias
    # after assignment.
    price = fc._numeric_column('price')
    price_buckets = fc._bucketized_column(
        price, boundaries=[
            0.,
            10.,
            100.,
        ])
    body_style = fc._categorical_column_with_vocabulary_list(
        'body-style', vocabulary_list=['hardtop', 'wagon', 'sedan'])

    # Provides 1-dim tensor and dense tensor.
    features = {
        'price':
            constant_op.constant([
                -1.,
                12.,
            ]),
        'body-style':
            sparse_tensor.SparseTensor(
                indices=((0,), (1,)),
                values=('sedan', 'hardtop'),
                dense_shape=(2,)),
    }
    self.assertEqual(1, features['price'].shape.ndims)
    self.assertEqual(1, features['body-style'].dense_shape.get_shape()[0])

    net = get_keras_linear_model_predictions(features,
                                             [price_buckets, body_style])
    with _initialized_session() as sess:
      bias = get_linear_model_bias()
      price_buckets_var = get_linear_model_column_var(price_buckets)
      body_style_var = get_linear_model_column_var(body_style)

      sess.run(price_buckets_var.assign([[10.], [100.], [1000.], [10000.]]))
      sess.run(body_style_var.assign([[-10.], [-100.], [-1000.]]))
      sess.run(bias.assign([5.]))

      self.assertAllClose([[10 - 1000 + 5.], [1000 - 10 + 5.]],
                          self.evaluate(net))

  @test_util.run_deprecated_v1
  # Placeholders are TF1. Replacing with tf.function not feasible because of V1
  # variable creation.
  def test_with_1d_unknown_shape_sparse_tensor(self):
    # [FUNCTION DOC]
    # Scenario (v1 graph mode): unknown-shape placeholders for price, body style and country; asserts
    # predictions after feeding rank one data and assigning weights and bias.
    price = fc._numeric_column('price')
    price_buckets = fc._bucketized_column(
        price, boundaries=[
            0.,
            10.,
            100.,
        ])
    body_style = fc._categorical_column_with_vocabulary_list(
        'body-style', vocabulary_list=['hardtop', 'wagon', 'sedan'])
    country = fc._categorical_column_with_vocabulary_list(
        'country', vocabulary_list=['US', 'JP', 'CA'])

    # Provides 1-dim tensor and dense tensor.
    features = {
        'price': array_ops.placeholder(dtypes.float32),
        'body-style': array_ops.sparse_placeholder(dtypes.string),
        'country': array_ops.placeholder(dtypes.string),
    }
    self.assertIsNone(features['price'].shape.ndims)
    self.assertIsNone(features['body-style'].get_shape().ndims)

    price_data = np.array([-1., 12.])
    body_style_data = sparse_tensor.SparseTensorValue(
        indices=((0,), (1,)), values=('sedan', 'hardtop'), dense_shape=(2,))
    country_data = np.array(['US', 'CA'])

    net = get_keras_linear_model_predictions(
        features, [price_buckets, body_style, country])
    bias = get_linear_model_bias()
    price_buckets_var = get_linear_model_column_var(price_buckets)
    body_style_var = get_linear_model_column_var(body_style)
    with _initialized_session() as sess:
      sess.run(price_buckets_var.assign([[10.], [100.], [1000.], [10000.]]))
      sess.run(body_style_var.assign([[-10.], [-100.], [-1000.]]))
      sess.run(bias.assign([5.]))

      self.assertAllClose([[10 - 1000 + 5.], [1000 - 10 + 5.]],
                          sess.run(
                              net,
                              feed_dict={
                                  features['price']: price_data,
                                  features['body-style']: body_style_data,
                                  features['country']: country_data
                              }))

  @test_util.run_deprecated_v1
  # Placeholders are TF1. Replacing with tf.function not feasible because of V1
  # variable creation.
  def test_with_rank_0_feature(self):
    # [FUNCTION DOC]
    # Scenario (v1 graph mode): asserts a scalar feature raises a value error about rank zero
    # statically, and a runtime error with the same message when a scalar is fed to a placeholder.
    price = fc._numeric_column('price')
    features = {
        'price': constant_op.constant(0),
    }
    self.assertEqual(0, features['price'].shape.ndims)

    # Static rank 0 should fail
    with self.assertRaisesRegex(ValueError, 'Feature .* cannot have rank 0'):
      get_keras_linear_model_predictions(features, [price])

    # Dynamic rank 0 should fail
    features = {
        'price': array_ops.placeholder(dtypes.float32),
    }
    net = get_keras_linear_model_predictions(features, [price])
    self.assertEqual(1, net.shape[1])
    with _initialized_session() as sess:
      with self.assertRaisesOpError('Feature .* cannot have rank 0'):
        sess.run(net, feed_dict={features['price']: np.array(1)})


class InputLayerTest(test.TestCase):
  # [CLASS DOC]
  # Tests the input layer class (a Keras-style layer that concatenates dense columns): retrieving
  # inputs, variable reuse across calls in eager mode and gradient flow to embedding weights.

  @test_util.run_in_graph_and_eager_modes
  def test_retrieving_input(self):
    # [FUNCTION DOC]
    # Runs in both graph and eager modes. Builds an input layer over a numeric column, calls it on a
    # one-value feature and asserts the output is a one by one matrix holding that value.
    features = {'a': [0.]}
    input_layer = InputLayer(fc._numeric_column('a'))
    inputs = self.evaluate(input_layer(features))
    self.assertAllClose([[0.]], inputs)

  def test_reuses_variables(self):
    # [FUNCTION DOC]
    # Scenario (eager mode): an embedding column over an identity column with a fixed initializer.
    # Asserts the layer output equals the embedding rows for the ids, that exactly one variable is
    # created, and that calling the layer again does not create more variables and keeps the same
    # variable object.
    with context.eager_mode():
      sparse_input = sparse_tensor.SparseTensor(
          indices=((0, 0), (1, 0), (2, 0)),
          values=(0, 1, 2),
          dense_shape=(3, 3))

      # Create feature columns (categorical and embedding).
      categorical_column = fc._categorical_column_with_identity(
          key='a', num_buckets=3)
      embedding_dimension = 2
      def _embedding_column_initializer(shape, dtype, partition_info):
        # [FUNCTION DOC]
        # HELPER: fixture initializer for the embedding table in this test.
        # Initializer that ignores the requested shape, dtype and partition information and returns a
        # fixed three by two table of embedding values.
        del shape  # unused
        del dtype  # unused
        del partition_info  # unused
        embedding_values = (
            (1, 0),  # id 0
            (0, 1),  # id 1
            (1, 1))  # id 2
        return embedding_values

      embedding_column = fc._embedding_column(
          categorical_column,
          dimension=embedding_dimension,
          initializer=_embedding_column_initializer)

      input_layer = InputLayer([embedding_column])
      features = {'a': sparse_input}

      inputs = input_layer(features)
      variables = input_layer.variables

      # Sanity check: test that the inputs are correct.
      self.assertAllEqual([[1, 0], [0, 1], [1, 1]], inputs)

      # Check that only one variable was created.
      self.assertEqual(1, len(variables))

      # Check that invoking input_layer on the same features does not create
      # additional variables
      _ = input_layer(features)
      self.assertEqual(1, len(variables))
      self.assertIs(variables[0], input_layer.variables[0])

  def test_feature_column_input_layer_gradient(self):
    # [FUNCTION DOC]
    # Scenario (eager mode): an embedding column input layer wrapped in a function that doubles the
    # output. Asserts the output is correct, then uses implicit gradients and asserts the gradient is
    # an indexed slices object with ids zero to two and every gradient value equal to two.
    with context.eager_mode():
      sparse_input = sparse_tensor.SparseTensor(
          indices=((0, 0), (1, 0), (2, 0)),
          values=(0, 1, 2),
          dense_shape=(3, 3))

      # Create feature columns (categorical and embedding).
      categorical_column = fc._categorical_column_with_identity(
          key='a', num_buckets=3)
      embedding_dimension = 2

      def _embedding_column_initializer(shape, dtype, partition_info):
        # [FUNCTION DOC]
        # HELPER: fixture initializer for the embedding table in this test.
        # Initializer that ignores the requested shape, dtype and partition information and returns a
        # fixed three by two table of embedding values.
        del shape  # unused
        del dtype  # unused
        del partition_info  # unused
        embedding_values = (
            (1, 0),  # id 0
            (0, 1),  # id 1
            (1, 1))  # id 2
        return embedding_values

      embedding_column = fc._embedding_column(
          categorical_column,
          dimension=embedding_dimension,
          initializer=_embedding_column_initializer)

      input_layer = InputLayer([embedding_column])
      features = {'a': sparse_input}

      def scale_matrix():
        # [FUNCTION DOC]
        # HELPER: function whose gradient is taken by the test.
        # Calls the input layer on the features of the enclosing test and returns the result multiplied by
        # two. Takes no arguments.
        matrix = input_layer(features)
        return 2 * matrix

      # Sanity check: Verify that scale_matrix returns the correct output.
      self.assertAllEqual([[2, 0], [0, 2], [2, 2]], scale_matrix())

      # Check that the returned gradient is correct.
      grad_function = backprop.implicit_grad(scale_matrix)
      grads_and_vars = grad_function()
      indexed_slice = grads_and_vars[0][0]
      gradient = grads_and_vars[0][0].values

      self.assertAllEqual([0, 1, 2], indexed_slice.indices)
      self.assertAllEqual([[2, 2], [2, 2], [2, 2]], gradient)


class FunctionalInputLayerTest(test.TestCase):
  # [CLASS DOC]
  # Tests the input layer function (the functional counterpart of the input layer class):
  # validation, bare and generator column arguments, shape handling, columns to variables mapping
  # (including partitioned and shared embeddings), column ordering, batch size checks, repeated
  # calls creating separate variables, and one dimensional, unknown-shape and rank zero inputs.

  def test_raises_if_empty_feature_columns(self):
    # [FUNCTION DOC]
    # Asserts that an empty feature column list raises a value error saying the columns must not be
    # empty.
    with self.assertRaisesRegex(ValueError,
                                'feature_columns must not be empty'):
      fc.input_layer(features={}, feature_columns=[])

  def test_should_be_dense_column(self):
    # [FUNCTION DOC]
    # Asserts that passing a categorical column (not a dense column) raises a value error saying it
    # must be a dense column.
    with self.assertRaisesRegex(ValueError, 'must be a _DenseColumn'):
      fc.input_layer(
          features={'a': [[0]]},
          feature_columns=[
              fc._categorical_column_with_hash_bucket('wire_cast', 4)
          ])

  def test_does_not_support_dict_columns(self):
    # [FUNCTION DOC]
    # Asserts that passing a mapping of columns instead of an iterable raises a value error saying a
    # dict was found.
    with self.assertRaisesRegex(
        ValueError, 'Expected feature_columns to be iterable, found dict.'):
      fc.input_layer(
          features={'a': [[0]]}, feature_columns={'a': fc._numeric_column('a')})

  def test_bare_column(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the features variable is assigned twice in a chained assignment, which is redundant
    # but harmless.
    # Passes a single numeric column (not wrapped in a list) to the input layer function and asserts
    # the output is the one by one matrix holding the feature value.
    with ops.Graph().as_default():
      features = features = {'a': [0.]}
      net = fc.input_layer(features, fc._numeric_column('a'))
      with _initialized_session():
        self.assertAllClose([[0.]], self.evaluate(net))

  def test_column_generator(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the features variable is assigned twice in a chained assignment, which is redundant
    # but harmless.
    # Passes a generator of numeric columns to the input layer function and asserts the output
    # concatenates both feature values in order.
    with ops.Graph().as_default():
      features = features = {'a': [0.], 'b': [1.]}
      columns = (fc._numeric_column(key) for key in features)
      net = fc.input_layer(features, columns)
      with _initialized_session():
        self.assertAllClose([[0., 1.]], self.evaluate(net))

  def test_raises_if_duplicate_name(self):
    # [FUNCTION DOC]
    # Asserts that two numeric columns with the same name raise a value error about a duplicate
    # feature column name.
    with self.assertRaisesRegex(
        ValueError, 'Duplicate feature column name found for columns'):
      fc.input_layer(
          features={'a': [[0]]},
          feature_columns=[fc._numeric_column('a'),
                           fc._numeric_column('a')])

  def test_one_column(self):
    # [FUNCTION DOC]
    # One numeric column with two examples; asserts the output equals the input values.
    price = fc._numeric_column('price')
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      net = fc.input_layer(features, [price])
      with _initialized_session():
        self.assertAllClose([[1.], [5.]], self.evaluate(net))

  def test_multi_dimension(self):
    # [FUNCTION DOC]
    # One numeric column of shape two; asserts the output equals the two-wide input.
    price = fc._numeric_column('price', shape=2)
    with ops.Graph().as_default():
      features = {'price': [[1., 2.], [5., 6.]]}
      net = fc.input_layer(features, [price])
      with _initialized_session():
        self.assertAllClose([[1., 2.], [5., 6.]], self.evaluate(net))

  def test_raises_if_shape_mismatch(self):
    # [FUNCTION DOC]
    # Asserts that a feature with one value per example fed to a column of shape two raises an error
    # about reshaping two elements to two by two.
    price = fc._numeric_column('price', shape=2)
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      with self.assertRaisesRegex(
          Exception,
          r'Cannot reshape a tensor with 2 elements to shape \[2,2\]'):
        fc.input_layer(features, [price])

  def test_reshaping(self):
    # [FUNCTION DOC]
    # Numeric column of shape one by two with a three-dimensional input; asserts the output is
    # flattened to two columns.
    price = fc._numeric_column('price', shape=[1, 2])
    with ops.Graph().as_default():
      features = {'price': [[[1., 2.]], [[5., 6.]]]}
      net = fc.input_layer(features, [price])
      with _initialized_session():
        self.assertAllClose([[1., 2.], [5., 6.]], self.evaluate(net))

  def test_multi_column(self):
    # [FUNCTION DOC]
    # Two numeric columns of widths two and one; asserts the output concatenates them in column name
    # order into three columns.
    price1 = fc._numeric_column('price1', shape=2)
    price2 = fc._numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': [[1., 2.], [5., 6.]],
          'price2': [[3.], [4.]]
      }
      net = fc.input_layer(features, [price1, price2])
      with _initialized_session():
        self.assertAllClose([[1., 2., 3.], [5., 6., 4.]], self.evaluate(net))

  def test_fills_cols_to_vars(self):
    # [FUNCTION DOC]
    # Scenario: a numeric, a bucketized and an embedding column with a columns to variables mapping.
    # Asserts the mapping has exactly those columns, the first two map to no variables and the
    # embedding maps to one variable of shape five by ten.
    # Provide three _DenseColumn's to input_layer: a _NumericColumn, a
    # _BucketizedColumn, and an _EmbeddingColumn.  Only the _EmbeddingColumn
    # creates a Variable.
    price1 = fc._numeric_column('price1')
    dense_feature = fc._numeric_column('dense_feature')
    dense_feature_bucketized = fc._bucketized_column(
        dense_feature, boundaries=[0.])
    some_sparse_column = fc._categorical_column_with_hash_bucket(
        'sparse_feature', hash_bucket_size=5)
    some_embedding_column = fc._embedding_column(
        some_sparse_column, dimension=10)
    with ops.Graph().as_default():
      features = {
          'price1': [[3.], [4.]],
          'dense_feature': [[-1.], [4.]],
          'sparse_feature': [['a'], ['x']],
      }
      cols_to_vars = {}
      all_cols = [price1, dense_feature_bucketized, some_embedding_column]
      fc.input_layer(features, all_cols, cols_to_vars=cols_to_vars)
      self.assertCountEqual(list(cols_to_vars.keys()), all_cols)
      self.assertEqual(0, len(cols_to_vars[price1]))
      self.assertEqual(0, len(cols_to_vars[dense_feature_bucketized]))
      self.assertEqual(1, len(cols_to_vars[some_embedding_column]))
      self.assertIsInstance(cols_to_vars[some_embedding_column][0],
                            variables_lib.Variable)
      self.assertAllEqual(cols_to_vars[some_embedding_column][0].shape, [5, 10])

  def test_fills_cols_to_vars_shared_embedding(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test deliberately asserts what its own comment calls a bug of the legacy
    # implementation, namely that the second shared embedding column maps to zero variables while the
    # first maps to the shared one.
    # Scenario: numeric, bucketized, embedding and two shared embedding columns passed with a columns
    # to variables mapping. Asserts the key set, no variables for numeric and bucketized columns, one
    # variable for the plain embedding, one shared variable of shape three by two for the first shared
    # column and zero for the second.
    # Provide 5 DenseColumn's to input_layer: a NumericColumn, a
    # BucketizedColumn, an EmbeddingColumn, two SharedEmbeddingColumns. The
    # EmbeddingColumn creates a Variable and the two SharedEmbeddingColumns
    # shared one variable.
    with ops.Graph().as_default():
      price1 = fc._numeric_column('price1')
      dense_feature = fc._numeric_column('dense_feature')
      dense_feature_bucketized = fc._bucketized_column(
          dense_feature, boundaries=[0.])
      some_sparse_column = fc._categorical_column_with_hash_bucket(
          'sparse_feature', hash_bucket_size=5)
      some_embedding_column = fc._embedding_column(
          some_sparse_column, dimension=10)
      categorical_column_a = fc._categorical_column_with_identity(
          key='aaa', num_buckets=3)
      categorical_column_b = fc._categorical_column_with_identity(
          key='bbb', num_buckets=3)
      shared_embedding_a, shared_embedding_b = fc_new.shared_embedding_columns(
          [categorical_column_a, categorical_column_b], dimension=2)
      features = {
          'price1': [[3.], [4.]],
          'dense_feature': [[-1.], [4.]],
          'sparse_feature': [['a'], ['x']],
          'aaa':
              sparse_tensor.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(0, 1, 0),
                  dense_shape=(2, 2)),
          'bbb':
              sparse_tensor.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(1, 2, 1),
                  dense_shape=(2, 2)),
      }
      cols_to_vars = {}
      all_cols = [
          price1, dense_feature_bucketized, some_embedding_column,
          shared_embedding_a, shared_embedding_b
      ]
      fc.input_layer(features, all_cols, cols_to_vars=cols_to_vars)
      self.assertCountEqual(list(cols_to_vars.keys()), all_cols)
      self.assertEqual(0, len(cols_to_vars[price1]))
      self.assertEqual(0, len(cols_to_vars[dense_feature_bucketized]))
      self.assertEqual(1, len(cols_to_vars[some_embedding_column]))
      self.assertEqual(1, len(cols_to_vars[shared_embedding_a]))
      # This is a bug in the current implementation and should be fixed in the
      # new one.
      self.assertEqual(0, len(cols_to_vars[shared_embedding_b]))
      self.assertIsInstance(cols_to_vars[some_embedding_column][0],
                            variables_lib.Variable)
      self.assertAllEqual(cols_to_vars[some_embedding_column][0].shape, [5, 10])
      self.assertIsInstance(cols_to_vars[shared_embedding_a][0],
                            variables_lib.Variable)
      self.assertAllEqual(cols_to_vars[shared_embedding_a][0].shape, [3, 2])

  def test_fills_cols_to_vars_partitioned_variables(self):
    # [FUNCTION DOC]
    # Scenario: an embedding column built under a three-shard fixed-size partitioner. Asserts the
    # columns to variables mapping lists three shards for the embedding with the expected first shard
    # name and shard shapes (two, two and one rows by ten).
    price1 = fc._numeric_column('price1')
    dense_feature = fc._numeric_column('dense_feature')
    dense_feature_bucketized = fc._bucketized_column(
        dense_feature, boundaries=[0.])
    some_sparse_column = fc._categorical_column_with_hash_bucket(
        'sparse_feature', hash_bucket_size=5)
    some_embedding_column = fc._embedding_column(
        some_sparse_column, dimension=10)
    with ops.Graph().as_default():
      features = {
          'price1': [[3.], [4.]],
          'dense_feature': [[-1.], [4.]],
          'sparse_feature': [['a'], ['x']],
      }
      cols_to_vars = {}
      all_cols = [price1, dense_feature_bucketized, some_embedding_column]
      with variable_scope.variable_scope(
          'input_from_feature_columns',
          partitioner=partitioned_variables.fixed_size_partitioner(3, axis=0)):
        fc.input_layer(features, all_cols, cols_to_vars=cols_to_vars)
      self.assertCountEqual(list(cols_to_vars.keys()), all_cols)
      self.assertEqual(0, len(cols_to_vars[price1]))
      self.assertEqual(0, len(cols_to_vars[dense_feature_bucketized]))
      self.assertEqual(3, len(cols_to_vars[some_embedding_column]))
      self.assertEqual(
          'input_from_feature_columns/input_layer/sparse_feature_embedding/'
          'embedding_weights/part_0:0',
          cols_to_vars[some_embedding_column][0].name)
      self.assertAllEqual(cols_to_vars[some_embedding_column][0].shape, [2, 10])
      self.assertAllEqual(cols_to_vars[some_embedding_column][1].shape, [2, 10])
      self.assertAllEqual(cols_to_vars[some_embedding_column][2].shape, [1, 10])

  def test_column_order(self):
    # [FUNCTION DOC]
    # Asserts that the input layer output is the same regardless of the order the numeric columns are
    # given, since columns are ordered by name.
    price_a = fc._numeric_column('price_a')
    price_b = fc._numeric_column('price_b')
    with ops.Graph().as_default():
      features = {
          'price_a': [[1.]],
          'price_b': [[3.]],
      }
      net1 = fc.input_layer(features, [price_a, price_b])
      net2 = fc.input_layer(features, [price_b, price_a])
      with _initialized_session():
        self.assertAllClose([[1., 3.]], self.evaluate(net1))
        self.assertAllClose([[1., 3.]], self.evaluate(net2))

  def test_fails_for_categorical_column(self):
    # [FUNCTION DOC]
    # Asserts that a categorical identity column raises an error saying it must be a dense column.
    animal = fc._categorical_column_with_identity('animal', num_buckets=4)
    with ops.Graph().as_default():
      features = {
          'animal':
              sparse_tensor.SparseTensor(
                  indices=[[0, 0], [0, 1]], values=[1, 2], dense_shape=[1, 2])
      }
      with self.assertRaisesRegex(Exception, 'must be a _DenseColumn'):
        fc.input_layer(features, [animal])

  def test_static_batch_size_mismatch(self):
    # [FUNCTION DOC]
    # Two features with static batch sizes three and two; asserts a value error saying batch sizes
    # must be the same.
    price1 = fc._numeric_column('price1')
    price2 = fc._numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': [[1.], [5.], [7.]],  # batchsize = 3
          'price2': [[3.], [4.]]  # batchsize = 2
      }
      with self.assertRaisesRegex(
          ValueError,
          r'Batch size \(first dimension\) of each feature must be same.'):  # pylint: disable=anomalous-backslash-in-string
        fc.input_layer(features, [price1, price2])

  def test_subset_of_static_batch_size_mismatch(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the inline comment labeling the placeholder feature as having batch size three is
    # misleading, because that placeholder has an unknown shape.
    # One unknown-shape placeholder and two statically mismatched features; asserts a value error
    # about mismatching batch sizes.
    price1 = fc._numeric_column('price1')
    price2 = fc._numeric_column('price2')
    price3 = fc._numeric_column('price3')
    with ops.Graph().as_default():
      features = {
          'price1': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 3
          'price2': [[3.], [4.]],  # batchsize = 2
          'price3': [[3.], [4.], [5.]]  # batchsize = 3
      }
      with self.assertRaisesRegex(
          ValueError,
          r'Batch size \(first dimension\) of each feature must be same.'):  # pylint: disable=anomalous-backslash-in-string
        fc.input_layer(features, [price1, price2, price3])

  def test_runtime_batch_size_mismatch(self):
    # [FUNCTION DOC]
    # Placeholder fed with three rows against a static two-row feature; asserts a runtime op error
    # saying dimension zero of both shapes must be equal.
    price1 = fc._numeric_column('price1')
    price2 = fc._numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 3
          'price2': [[3.], [4.]]  # batchsize = 2
      }
      net = fc.input_layer(features, [price1, price2])
      with _initialized_session() as sess:
        with self.assertRaisesRegex(errors.OpError,
                                    'Dimension 0 in both shapes must be equal'):
          sess.run(net, feed_dict={features['price1']: [[1.], [5.], [7.]]})

  def test_runtime_batch_size_matches(self):
    # [FUNCTION DOC]
    # Two placeholders fed with matching batch sizes; asserts the layer runs without error.
    price1 = fc._numeric_column('price1')
    price2 = fc._numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 2
          'price2': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 2
      }
      net = fc.input_layer(features, [price1, price2])
      with _initialized_session() as sess:
        sess.run(
            net,
            feed_dict={
                features['price1']: [[1.], [5.]],
                features['price2']: [[1.], [5.]],
            })

  def test_multiple_layers_with_same_embedding_column(self):
    # [FUNCTION DOC]
    # Calls the input layer function twice with the same embedding column in one graph and asserts two
    # separate embedding variables are created, with names under the input layer scope and its
    # uniquified second scope.
    some_sparse_column = fc._categorical_column_with_hash_bucket(
        'sparse_feature', hash_bucket_size=5)
    some_embedding_column = fc._embedding_column(
        some_sparse_column, dimension=10)

    with ops.Graph().as_default():
      features = {
          'sparse_feature': [['a'], ['x']],
      }
      all_cols = [some_embedding_column]
      fc.input_layer(features, all_cols)
      fc.input_layer(features, all_cols)
      # Make sure that 2 variables get created in this case.
      self.assertEqual(2, len(
          ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)))
      expected_var_names = [
          'input_layer/sparse_feature_embedding/embedding_weights:0',
          'input_layer_1/sparse_feature_embedding/embedding_weights:0'
      ]
      self.assertCountEqual(
          expected_var_names,
          [v.name for v in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)])

  def test_multiple_layers_with_same_shared_embedding_column(self):
    # [FUNCTION DOC]
    # Calls the input layer function twice with the same pair of shared embedding columns in one graph
    # and asserts only one shared variable is created.
    with ops.Graph().as_default():
      categorical_column_a = fc._categorical_column_with_identity(
          key='aaa', num_buckets=3)
      categorical_column_b = fc._categorical_column_with_identity(
          key='bbb', num_buckets=3)
      embedding_dimension = 2
      embedding_column_b, embedding_column_a = fc_new.shared_embedding_columns(
          [categorical_column_b, categorical_column_a],
          dimension=embedding_dimension)

      features = {
          'aaa':
              sparse_tensor.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(0, 1, 0),
                  dense_shape=(2, 2)),
          'bbb':
              sparse_tensor.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(1, 2, 1),
                  dense_shape=(2, 2)),
      }
      all_cols = [embedding_column_a, embedding_column_b]
      fc.input_layer(features, all_cols)
      fc.input_layer(features, all_cols)
      # Make sure that only 1 variable gets created in this case.
      self.assertEqual(1, len(
          ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)))
      self.assertCountEqual(
          ['input_layer/aaa_bbb_shared_embedding/embedding_weights:0'],
          [v.name for v in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)])

  def test_multiple_layers_with_same_shared_embedding_column_diff_graphs(self):
    # [FUNCTION DOC]
    # Uses the same shared embedding columns in two different graphs and asserts each graph creates
    # exactly one shared variable, with the expected name in the second graph.
    with ops.Graph().as_default():
      categorical_column_a = fc._categorical_column_with_identity(
          key='aaa', num_buckets=3)
      categorical_column_b = fc._categorical_column_with_identity(
          key='bbb', num_buckets=3)
      embedding_dimension = 2
      embedding_column_b, embedding_column_a = fc_new.shared_embedding_columns(
          [categorical_column_b, categorical_column_a],
          dimension=embedding_dimension)
      all_cols = [embedding_column_a, embedding_column_b]

      features = {
          'aaa':
              sparse_tensor.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(0, 1, 0),
                  dense_shape=(2, 2)),
          'bbb':
              sparse_tensor.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(1, 2, 1),
                  dense_shape=(2, 2)),
      }
      fc.input_layer(features, all_cols)
      # Make sure that only 1 variable gets created in this case.
      self.assertEqual(1, len(
          ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)))

    with ops.Graph().as_default():
      features1 = {
          'aaa':
              sparse_tensor.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(0, 1, 0),
                  dense_shape=(2, 2)),
          'bbb':
              sparse_tensor.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(1, 2, 1),
                  dense_shape=(2, 2)),
      }

      fc.input_layer(features1, all_cols)
      # Make sure that only 1 variable gets created in this case.
      self.assertEqual(1, len(
          ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)))
      self.assertCountEqual(
          ['input_layer/aaa_bbb_shared_embedding/embedding_weights:0'],
          [v.name for v in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)])

  def test_with_1d_sparse_tensor(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the comment about row layout names the embedded column first and calls it embedded
    # body style, but the output order is actually by column name (one-hot body style, then embedded
    # country, then price) and the embedding belongs to the country column.
    # Scenario: price as a rank one dense tensor, body style as a rank one sparse tensor with an
    # indicator column, and country as a rank one dense string tensor with an embedding column.
    # Asserts the inputs are rank one, the output width equals one plus three plus five, and the exact
    # output rows.
    with ops.Graph().as_default():
      embedding_values = (
          (1., 2., 3., 4., 5.),  # id 0
          (6., 7., 8., 9., 10.),  # id 1
          (11., 12., 13., 14., 15.)  # id 2
      )
      def _initializer(shape, dtype, partition_info):
        # [FUNCTION DOC]
        # HELPER: fixture initializer for the embedding table in this test.
        # Embedding initializer that ignores shape, dtype and partition information and returns a fixed
        # three by five table.
        del shape, dtype, partition_info
        return embedding_values

      # price has 1 dimension in input_layer
      price = fc._numeric_column('price')

      # one_hot_body_style has 3 dims in input_layer.
      body_style = fc._categorical_column_with_vocabulary_list(
          'body-style', vocabulary_list=['hardtop', 'wagon', 'sedan'])
      one_hot_body_style = fc._indicator_column(body_style)

      # embedded_body_style has 5 dims in input_layer.
      country = fc._categorical_column_with_vocabulary_list(
          'country', vocabulary_list=['US', 'JP', 'CA'])
      embedded_country = fc._embedding_column(
          country, dimension=5, initializer=_initializer)

      # Provides 1-dim tensor and dense tensor.
      features = {
          'price': constant_op.constant([11., 12.,]),
          'body-style': sparse_tensor.SparseTensor(
              indices=((0,), (1,)),
              values=('sedan', 'hardtop'),
              dense_shape=(2,)),
          # This is dense tensor for the categorical_column.
          'country': constant_op.constant(['CA', 'US']),
      }
      self.assertEqual(1, features['price'].shape.ndims)
      self.assertEqual(1, features['body-style'].dense_shape.get_shape()[0])
      self.assertEqual(1, features['country'].shape.ndims)

      net = fc.input_layer(features,
                           [price, one_hot_body_style, embedded_country])
      self.assertEqual(1 + 3 + 5, net.shape[1])
      with _initialized_session():

        # Each row is formed by concatenating `embedded_body_style`,
        # `one_hot_body_style`, and `price` in order.
        self.assertAllEqual(
            [[0., 0., 1., 11., 12., 13., 14., 15., 11.],
             [1., 0., 0., 1., 2., 3., 4., 5., 12.]],
            self.evaluate(net))

  @test_util.run_deprecated_v1
  # Placeholders are TF1. Replacing with tf.function not feasible because of V1
  # variable creation.
  def test_with_1d_unknown_shape_sparse_tensor(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the comments claim the embedded column has five dimensions and call it embedded
    # body style, but the embedding dimension is two and the embedded column is the country column;
    # the output order is also by column name rather than as the comment states.
    # Scenario (v1 graph mode): unknown-shape placeholders for price, sparse body style and dense
    # country with an indicator and an embedding column. Asserts shapes are unknown, the output width
    # is one plus three plus two, and the exact output after feeding rank one price and body style
    # data and a rank two country batch.
    embedding_values = (
        (1., 2.),  # id 0
        (6., 7.),  # id 1
        (11., 12.)  # id 2
    )
    def _initializer(shape, dtype, partition_info):
      # [FUNCTION DOC]
      # HELPER: fixture initializer for the embedding table in this test.
      # Embedding initializer that ignores shape, dtype and partition information and returns a fixed
      # three by two table.
      del shape, dtype, partition_info
      return embedding_values

    # price has 1 dimension in input_layer
    price = fc._numeric_column('price')

    # one_hot_body_style has 3 dims in input_layer.
    body_style = fc._categorical_column_with_vocabulary_list(
        'body-style', vocabulary_list=['hardtop', 'wagon', 'sedan'])
    one_hot_body_style = fc._indicator_column(body_style)

    # embedded_body_style has 5 dims in input_layer.
    country = fc._categorical_column_with_vocabulary_list(
        'country', vocabulary_list=['US', 'JP', 'CA'])
    embedded_country = fc._embedding_column(
        country, dimension=2, initializer=_initializer)

    # Provides 1-dim tensor and dense tensor.
    features = {
        'price': array_ops.placeholder(dtypes.float32),
        'body-style': array_ops.sparse_placeholder(dtypes.string),
        # This is dense tensor for the categorical_column.
        'country': array_ops.placeholder(dtypes.string),
    }
    self.assertIsNone(features['price'].shape.ndims)
    self.assertIsNone(features['body-style'].get_shape().ndims)
    self.assertIsNone(features['country'].shape.ndims)

    price_data = np.array([11., 12.])
    body_style_data = sparse_tensor.SparseTensorValue(
        indices=((0,), (1,)),
        values=('sedan', 'hardtop'),
        dense_shape=(2,))
    country_data = np.array([['US'], ['CA']])

    net = fc.input_layer(features,
                         [price, one_hot_body_style, embedded_country])
    self.assertEqual(1 + 3 + 2, net.shape[1])
    with _initialized_session() as sess:

      # Each row is formed by concatenating `embedded_body_style`,
      # `one_hot_body_style`, and `price` in order.
      self.assertAllEqual(
          [[0., 0., 1., 1., 2., 11.], [1., 0., 0., 11., 12., 12.]],
          sess.run(
              net,
              feed_dict={
                  features['price']: price_data,
                  features['body-style']: body_style_data,
                  features['country']: country_data
              }))

  @test_util.run_deprecated_v1
  # Placeholders are TF1. Replacing with tf.function not feasible because of V1
  # variable creation.
  def test_with_rank_0_feature(self):
    # [FUNCTION DOC]
    # Scenario (v1 graph mode): a scalar numeric feature. Asserts a value error saying the feature
    # cannot have rank zero for a static rank, and a runtime error with the same message when a scalar
    # is fed to an unknown-rank placeholder.
    # price has 1 dimension in input_layer
    price = fc._numeric_column('price')
    features = {
        'price': constant_op.constant(0),
    }
    self.assertEqual(0, features['price'].shape.ndims)

    # Static rank 0 should fail
    with self.assertRaisesRegex(ValueError, 'Feature .* cannot have rank 0'):
      fc.input_layer(features, [price])

    # Dynamic rank 0 should fail
    features = {
        'price': array_ops.placeholder(dtypes.float32),
    }
    net = fc.input_layer(features, [price])
    self.assertEqual(1, net.shape[1])
    with _initialized_session() as sess:
      with self.assertRaisesOpError('Feature .* cannot have rank 0'):
        sess.run(net, feed_dict={features['price']: np.array(1)})


class MakeParseExampleSpecTest(test.TestCase):
  # [CLASS DOC]
  # Tests the function that merges the parse specifications of a collection of feature columns into
  # one mapping. Defines a small fixture column class whose parse spec is supplied at construction.

  class _TestFeatureColumn(_FeatureColumn,
                           collections.namedtuple('_TestFeatureColumn',
                                                  ['parse_spec'])):
    # [CLASS DOC]
    # Fixture feature column built as a named tuple with one field holding a parse specification
    # mapping; its example spec property returns that mapping so tests can control what each column
    # contributes.

    @property
    def _parse_example_spec(self):
      # [FUNCTION DOC]
      # Property returning the parse spec mapping stored in the named tuple field.
      return self.parse_spec

  def test_no_feature_columns(self):
    # [FUNCTION DOC]
    # Asserts that an empty list of columns yields an empty mapping.
    actual = fc.make_parse_example_spec([])
    self.assertDictEqual({}, actual)

  def test_invalid_type(self):
    # [FUNCTION DOC]
    # Asserts that a collection containing a non-column item (a string) raises a value error saying
    # all columns must be feature column instances and naming the invalid item.
    key1 = 'key1'
    parse_spec1 = parsing_ops.FixedLenFeature(
        shape=(2,), dtype=dtypes.float32, default_value=0.)
    with self.assertRaisesRegex(
        ValueError,
        'All feature_columns must be _FeatureColumn instances.*invalid_column'):
      fc.make_parse_example_spec(
          (self._TestFeatureColumn({key1: parse_spec1}), 'invalid_column'))  # pylint: disable=abstract-class-instantiated

  def test_one_feature_column(self):
    # [FUNCTION DOC]
    # Asserts that one column with a fixed length feature spec yields a mapping with exactly that
    # spec.
    key1 = 'key1'
    parse_spec1 = parsing_ops.FixedLenFeature(
        shape=(2,), dtype=dtypes.float32, default_value=0.)
    actual = fc.make_parse_example_spec(
        (self._TestFeatureColumn({key1: parse_spec1}),))  # pylint: disable=abstract-class-instantiated
    self.assertDictEqual({key1: parse_spec1}, actual)

  def test_two_feature_columns(self):
    # [FUNCTION DOC]
    # Asserts that two columns with different keys yield a mapping containing both specs.
    key1 = 'key1'
    parse_spec1 = parsing_ops.FixedLenFeature(
        shape=(2,), dtype=dtypes.float32, default_value=0.)
    key2 = 'key2'
    parse_spec2 = parsing_ops.VarLenFeature(dtype=dtypes.string)
    actual = fc.make_parse_example_spec((
        self._TestFeatureColumn({key1: parse_spec1}),  # pylint: disable=abstract-class-instantiated
        self._TestFeatureColumn({key2: parse_spec2})))  # pylint: disable=abstract-class-instantiated
    self.assertDictEqual({key1: parse_spec1, key2: parse_spec2}, actual)

  def test_equal_keys_different_parse_spec(self):
    # [FUNCTION DOC]
    # Asserts that two columns using the same key with different specs raise a value error saying the
    # columns contain different parse specs for that key.
    key1 = 'key1'
    parse_spec1 = parsing_ops.FixedLenFeature(
        shape=(2,), dtype=dtypes.float32, default_value=0.)
    parse_spec2 = parsing_ops.VarLenFeature(dtype=dtypes.string)
    with self.assertRaisesRegex(
        ValueError,
        'feature_columns contain different parse_spec for key key1'):
      fc.make_parse_example_spec((
          self._TestFeatureColumn({key1: parse_spec1}),  # pylint: disable=abstract-class-instantiated
          self._TestFeatureColumn({key1: parse_spec2})))  # pylint: disable=abstract-class-instantiated

  def test_equal_keys_equal_parse_spec(self):
    # [FUNCTION DOC]
    # Asserts that two columns using the same key with identical specs are merged without error into a
    # single entry.
    key1 = 'key1'
    parse_spec1 = parsing_ops.FixedLenFeature(
        shape=(2,), dtype=dtypes.float32, default_value=0.)
    actual = fc.make_parse_example_spec((
        self._TestFeatureColumn({key1: parse_spec1}),  # pylint: disable=abstract-class-instantiated
        self._TestFeatureColumn({key1: parse_spec1})))  # pylint: disable=abstract-class-instantiated
    self.assertDictEqual({key1: parse_spec1}, actual)

  def test_multiple_features_dict(self):
    # [FUNCTION DOC]
    # Asserts that a column whose parse spec mapping has several keys contributes all of them to the
    # merged result.
    """parse_spc for one column is a dict with length > 1."""
    key1 = 'key1'
    parse_spec1 = parsing_ops.FixedLenFeature(
        shape=(2,), dtype=dtypes.float32, default_value=0.)
    key2 = 'key2'
    parse_spec2 = parsing_ops.VarLenFeature(dtype=dtypes.string)
    key3 = 'key3'
    parse_spec3 = parsing_ops.VarLenFeature(dtype=dtypes.int32)
    # pylint: disable=abstract-class-instantiated
    actual = fc.make_parse_example_spec((
        self._TestFeatureColumn({key1: parse_spec1}),
        self._TestFeatureColumn({
            key2: parse_spec2,
            key3: parse_spec3
        })))
    self.assertDictEqual(
        {key1: parse_spec1, key2: parse_spec2, key3: parse_spec3}, actual)


def _assert_sparse_tensor_value(test_case, expected, actual):
  # [FUNCTION DOC]
  # HELPER: shared assertion helper for the sparse tensor comparison in several test classes.
  # Takes a test case, an expected sparse tensor value and an actual sparse tensor value. Asserts
  # the actual indices and dense shape are int64, that expected and actual indices, values and dense
  # shape are equal, and that the value dtypes match. Returns nothing; failures surface as assertion
  # errors on the given test case.
  test_case.assertEqual(np.int64, np.array(actual.indices).dtype)
  test_case.assertAllEqual(expected.indices, actual.indices)

  test_case.assertEqual(
      np.array(expected.values).dtype, np.array(actual.values).dtype)
  test_case.assertAllEqual(expected.values, actual.values)

  test_case.assertEqual(np.int64, np.array(actual.dense_shape).dtype)
  test_case.assertAllEqual(expected.dense_shape, actual.dense_shape)


class VocabularyFileCategoricalColumnTest(test.TestCase):
  # [CLASS DOC]
  # Tests the legacy vocabulary file categorical column: defaults, constructor arguments, deep copy,
  # argument validation, file errors at table initialization, parsing, id lookup for string and
  # integer vocabularies with default values and out-of-vocabulary buckets, and linear model use.

  def setUp(self):
    # [FUNCTION DOC]
    # Calls the base setup, then records the paths of two vocabulary test data files (warriors jersey
    # numbers with five integer entries, and wire character names with three string entries) together
    # with their sizes, as instance attributes for use by the tests.
    super(VocabularyFileCategoricalColumnTest, self).setUp()

    # Contains ints, Golden State Warriors jersey numbers: 30, 35, 11, 23, 22
    self._warriors_vocabulary_file_name = test.test_src_dir_path(
        'python/feature_column/testdata/warriors_vocabulary.txt')
    self._warriors_vocabulary_size = 5

    # Contains strings, character names from 'The Wire': omar, stringer, marlo
    self._wire_vocabulary_file_name = test.test_src_dir_path(
        'python/feature_column/testdata/wire_vocabulary.txt')
    self._wire_vocabulary_size = 3

  def test_defaults(self):
    # [FUNCTION DOC]
    # Asserts defaults of a vocabulary file column: name, scope name, key, bucket count of three and a
    # variable length string parse spec.
    column = fc._categorical_column_with_vocabulary_file(
        key='aaa', vocabulary_file='path_to_file', vocabulary_size=3)
    self.assertEqual('aaa', column.name)
    self.assertEqual('aaa', column._var_scope_name)
    self.assertEqual('aaa', column.key)
    self.assertEqual(3, column._num_buckets)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.string)
    }, column._parse_example_spec)

  def test_key_should_be_string(self):
    # [FUNCTION DOC]
    # Asserts that a non-string key raises a value error saying the key must be a string.
    with self.assertRaisesRegex(ValueError, 'key must be a string.'):
      fc._categorical_column_with_vocabulary_file(
          key=('aaa',), vocabulary_file='path_to_file', vocabulary_size=3)

  def test_all_constructor_args(self):
    # [FUNCTION DOC]
    # Creates a column with extra out-of-vocabulary buckets and an int32 dtype; asserts the bucket
    # count is vocabulary size plus oov buckets and the parse spec uses int32.
    column = fc._categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file='path_to_file',
        vocabulary_size=3,
        num_oov_buckets=4,
        dtype=dtypes.int32)
    self.assertEqual(7, column._num_buckets)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int32)
    }, column._parse_example_spec)

  def test_deep_copy(self):
    # [FUNCTION DOC]
    # Asserts that both original and deep-copied columns keep name, bucket count and parse spec.
    original = fc._categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file='path_to_file',
        vocabulary_size=3,
        num_oov_buckets=4,
        dtype=dtypes.int32)
    for column in (original, copy.deepcopy(original)):
      self.assertEqual('aaa', column.name)
      self.assertEqual(7, column._num_buckets)
      self.assertEqual({
          'aaa': parsing_ops.VarLenFeature(dtypes.int32)
      }, column._parse_example_spec)

  def test_vocabulary_file_none(self):
    # [FUNCTION DOC]
    # Asserts that a missing vocabulary file raises a value error saying it is missing.
    with self.assertRaisesRegex(ValueError, 'Missing vocabulary_file'):
      fc._categorical_column_with_vocabulary_file(
          key='aaa', vocabulary_file=None, vocabulary_size=3)

  def test_vocabulary_file_empty_string(self):
    # [FUNCTION DOC]
    # Asserts that an empty vocabulary file name raises the same missing file value error.
    with self.assertRaisesRegex(ValueError, 'Missing vocabulary_file'):
      fc._categorical_column_with_vocabulary_file(
          key='aaa', vocabulary_file='', vocabulary_size=3)

  def test_invalid_vocabulary_file(self):
    # [FUNCTION DOC]
    # Creates a column for a nonexistent file and builds its sparse tensors; asserts that running the
    # table initializer raises an op error mentioning the missing file name.
    with ops.Graph().as_default():
      column = fc._categorical_column_with_vocabulary_file(
          key='aaa', vocabulary_file='file_does_not_exist', vocabulary_size=10)
      inputs = sparse_tensor.SparseTensorValue(
          indices=((0, 0), (1, 0), (1, 1)),
          values=('marlo', 'skywalker', 'omar'),
          dense_shape=(2, 2))
      column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
      with self.assertRaisesRegex(errors.OpError, 'file_does_not_exist'):
        with self.cached_session():
          lookup_ops.tables_initializer().run()

  def test_invalid_vocabulary_size(self):
    # [FUNCTION DOC]
    # Asserts that vocabulary sizes of minus one and zero raise a value error about an invalid
    # vocabulary size.
    with self.assertRaisesRegex(ValueError, 'Invalid vocabulary_size'):
      fc._categorical_column_with_vocabulary_file(
          key='aaa',
          vocabulary_file=self._wire_vocabulary_file_name,
          vocabulary_size=-1)
    with self.assertRaisesRegex(ValueError, 'Invalid vocabulary_size'):
      fc._categorical_column_with_vocabulary_file(
          key='aaa',
          vocabulary_file=self._wire_vocabulary_file_name,
          vocabulary_size=0)

  def test_too_large_vocabulary_size(self):
    # [FUNCTION DOC]
    # Creates a column with a vocabulary size larger than the file; asserts the table initializer
    # raises an op error about an invalid vocabulary size.
    with ops.Graph().as_default():
      column = fc._categorical_column_with_vocabulary_file(
          key='aaa',
          vocabulary_file=self._wire_vocabulary_file_name,
          vocabulary_size=self._wire_vocabulary_size + 1)
      inputs = sparse_tensor.SparseTensorValue(
          indices=((0, 0), (1, 0), (1, 1)),
          values=('marlo', 'skywalker', 'omar'),
          dense_shape=(2, 2))
      column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
      with self.assertRaisesRegex(errors.OpError, 'Invalid vocab_size'):
        with self.cached_session():
          lookup_ops.tables_initializer().run()

  def test_invalid_num_oov_buckets(self):
    # [FUNCTION DOC]
    # Asserts that a negative number of oov buckets raises a value error.
    with self.assertRaisesRegex(ValueError, 'Invalid num_oov_buckets'):
      fc._categorical_column_with_vocabulary_file(
          key='aaa',
          vocabulary_file='path',
          vocabulary_size=3,
          num_oov_buckets=-1)

  def test_invalid_dtype(self):
    # [FUNCTION DOC]
    # Asserts that a float64 dtype raises a value error saying the dtype must be string or integer.
    with self.assertRaisesRegex(ValueError, 'dtype must be string or integer'):
      fc._categorical_column_with_vocabulary_file(
          key='aaa',
          vocabulary_file='path',
          vocabulary_size=3,
          dtype=dtypes.float64)

  def test_invalid_buckets_and_default_value(self):
    # [FUNCTION DOC]
    # Asserts that specifying both a positive number of oov buckets and a default value raises a value
    # error.
    with self.assertRaisesRegex(ValueError,
                                'both num_oov_buckets and default_value'):
      fc._categorical_column_with_vocabulary_file(
          key='aaa',
          vocabulary_file=self._wire_vocabulary_file_name,
          vocabulary_size=self._wire_vocabulary_size,
          num_oov_buckets=100,
          default_value=2)

  def test_invalid_input_dtype_int32(self):
    # [FUNCTION DOC]
    # Asserts that integer input fed to a string vocabulary column raises a value error saying the
    # dtype must be compatible.
    column = fc._categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size,
        dtype=dtypes.string)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=(12, 24, 36),
        dense_shape=(2, 2))
    with self.assertRaisesRegex(ValueError, 'dtype must be compatible'):
      column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))

  def test_invalid_input_dtype_string(self):
    # [FUNCTION DOC]
    # Asserts that string input fed to an int32 vocabulary column raises a value error saying the
    # dtype must be compatible.
    column = fc._categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._warriors_vocabulary_file_name,
        vocabulary_size=self._warriors_vocabulary_size,
        dtype=dtypes.int32)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('omar', 'stringer', 'marlo'),
        dense_shape=(2, 2))
    with self.assertRaisesRegex(ValueError, 'dtype must be compatible'):
      column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))

  def test_parse_example(self):
    # [FUNCTION DOC]
    # Parses a serialized example of two byte strings using the spec of a vocabulary file column;
    # asserts the parsed sparse tensor has the original strings.
    a = fc._categorical_column_with_vocabulary_file(
        key='aaa', vocabulary_file='path_to_file', vocabulary_size=3)
    data = example_pb2.Example(features=feature_pb2.Features(
        feature={
            'aaa':
                feature_pb2.Feature(bytes_list=feature_pb2.BytesList(
                    value=[b'omar', b'stringer']))
        }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec([a]))
    self.assertIn('aaa', features)
    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [0, 1]],
            values=np.array([b'omar', b'stringer'], dtype=np.object_),
            dense_shape=[1, 2]), self.evaluate(features['aaa']))

  def test_get_sparse_tensors(self):
    # [FUNCTION DOC]
    # Looks up string inputs in the wire vocabulary; asserts no weight tensor and ids equal to the
    # vocabulary positions with minus one for the unknown word.
    with ops.Graph().as_default():
      column = fc._categorical_column_with_vocabulary_file(
          key='aaa',
          vocabulary_file=self._wire_vocabulary_file_name,
          vocabulary_size=self._wire_vocabulary_size)
      inputs = sparse_tensor.SparseTensorValue(
          indices=((0, 0), (1, 0), (1, 1)),
          values=('marlo', 'skywalker', 'omar'),
          dense_shape=(2, 2))
      id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
      self.assertIsNone(id_weight_pair.weight_tensor)
      with _initialized_session():
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=inputs.indices,
                values=np.array((2, -1, 0), dtype=np.int64),
                dense_shape=inputs.dense_shape),
            id_weight_pair.id_tensor.eval())

  def test_get_sparse_tensors_none_vocabulary_size(self):
    # [FUNCTION DOC]
    # Same lookup without giving a vocabulary size, so the whole file is used; asserts the same ids.
    with ops.Graph().as_default():
      column = fc._categorical_column_with_vocabulary_file(
          key='aaa', vocabulary_file=self._wire_vocabulary_file_name)
      inputs = sparse_tensor.SparseTensorValue(
          indices=((0, 0), (1, 0), (1, 1)),
          values=('marlo', 'skywalker', 'omar'),
          dense_shape=(2, 2))
      id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
      self.assertIsNone(id_weight_pair.weight_tensor)
      with _initialized_session():
        _assert_sparse_tensor_value(self,
                                    sparse_tensor.SparseTensorValue(
                                        indices=inputs.indices,
                                        values=np.array(
                                            (2, -1, 0), dtype=np.int64),
                                        dense_shape=inputs.dense_shape),
                                    id_weight_pair.id_tensor.eval())

  def test_transform_feature(self):
    # [FUNCTION DOC]
    # Transforms string inputs through the module transform function and asserts ids equal vocabulary
    # positions with minus one for unknown words.
    with ops.Graph().as_default():
      column = fc._categorical_column_with_vocabulary_file(
          key='aaa',
          vocabulary_file=self._wire_vocabulary_file_name,
          vocabulary_size=self._wire_vocabulary_size)
      inputs = sparse_tensor.SparseTensorValue(
          indices=((0, 0), (1, 0), (1, 1)),
          values=('marlo', 'skywalker', 'omar'),
          dense_shape=(2, 2))
      id_tensor = _transform_features({'aaa': inputs}, [column])[column]
      with _initialized_session():
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=inputs.indices,
                values=np.array((2, -1, 0), dtype=np.int64),
                dense_shape=inputs.dense_shape), self.evaluate(id_tensor))

  def test_get_sparse_tensors_weight_collections(self):
    # [FUNCTION DOC]
    # Asserts that requesting sparse tensors with a named weight collection creates no variables in
    # the global variables collection or the named collection.
    column = fc._categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size)
    inputs = sparse_tensor.SparseTensor(
        values=['omar', 'stringer', 'marlo'],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    column._get_sparse_tensors(
        _LazyBuilder({
            'aaa': inputs
        }), weight_collections=('my_weights',))

    self.assertCountEqual([],
                          ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES))
    self.assertCountEqual([], ops.get_collection('my_weights'))

  def test_get_sparse_tensors_dense_input(self):
    # [FUNCTION DOC]
    # Feeds a dense nested tuple of strings with an empty string and asserts the resulting sparse ids
    # skip the empty entry and map unknown words to minus one.
    with ops.Graph().as_default():
      column = fc._categorical_column_with_vocabulary_file(
          key='aaa',
          vocabulary_file=self._wire_vocabulary_file_name,
          vocabulary_size=self._wire_vocabulary_size)
      id_weight_pair = column._get_sparse_tensors(
          _LazyBuilder({
              'aaa': (('marlo', ''), ('skywalker', 'omar'))
          }))
      self.assertIsNone(id_weight_pair.weight_tensor)
      with _initialized_session():
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=((0, 0), (1, 0), (1, 1)),
                values=np.array((2, -1, 0), dtype=np.int64),
                dense_shape=(2, 2)),
            id_weight_pair.id_tensor.eval())

  def test_get_sparse_tensors_default_value_in_vocabulary(self):
    # [FUNCTION DOC]
    # Uses a default value inside the vocabulary range; asserts unknown words are mapped to that
    # default id.
    with ops.Graph().as_default():
      column = fc._categorical_column_with_vocabulary_file(
          key='aaa',
          vocabulary_file=self._wire_vocabulary_file_name,
          vocabulary_size=self._wire_vocabulary_size,
          default_value=2)
      inputs = sparse_tensor.SparseTensorValue(
          indices=((0, 0), (1, 0), (1, 1)),
          values=('marlo', 'skywalker', 'omar'),
          dense_shape=(2, 2))
      id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
      self.assertIsNone(id_weight_pair.weight_tensor)
      with _initialized_session():
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=inputs.indices,
                values=np.array((2, 2, 0), dtype=np.int64),
                dense_shape=inputs.dense_shape),
            id_weight_pair.id_tensor.eval())

  def test_get_sparse_tensors_with_oov_buckets(self):
    # [FUNCTION DOC]
    # Uses one hundred oov buckets; asserts known words map to their positions and the unknown words
    # are hashed to ids at or beyond the vocabulary size.
    with ops.Graph().as_default():
      column = fc._categorical_column_with_vocabulary_file(
          key='aaa',
          vocabulary_file=self._wire_vocabulary_file_name,
          vocabulary_size=self._wire_vocabulary_size,
          num_oov_buckets=100)
      inputs = sparse_tensor.SparseTensorValue(
          indices=((0, 0), (1, 0), (1, 1), (1, 2)),
          values=('marlo', 'skywalker', 'omar', 'heisenberg'),
          dense_shape=(2, 3))
      id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
      self.assertIsNone(id_weight_pair.weight_tensor)
      with _initialized_session():
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=inputs.indices,
                values=np.array((2, 33, 0, 62), dtype=np.int64),
                dense_shape=inputs.dense_shape),
            id_weight_pair.id_tensor.eval())

  def test_get_sparse_tensors_small_vocabulary_size(self):
    # [FUNCTION DOC]
    # Sets the vocabulary size to one less than the file length; asserts the last word of the file
    # becomes unknown and maps to minus one.
    with ops.Graph().as_default():
      # 'marlo' is the last entry in our vocabulary file, so be setting
      # `vocabulary_size` to 1 less than number of entries in file, we take
      # 'marlo' out of the vocabulary.
      column = fc._categorical_column_with_vocabulary_file(
          key='aaa',
          vocabulary_file=self._wire_vocabulary_file_name,
          vocabulary_size=self._wire_vocabulary_size - 1)
      inputs = sparse_tensor.SparseTensorValue(
          indices=((0, 0), (1, 0), (1, 1)),
          values=('marlo', 'skywalker', 'omar'),
          dense_shape=(2, 2))
      id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
      self.assertIsNone(id_weight_pair.weight_tensor)
      with _initialized_session():
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=inputs.indices,
                values=np.array((-1, -1, 0), dtype=np.int64),
                dense_shape=inputs.dense_shape),
            id_weight_pair.id_tensor.eval())

  def test_get_sparse_tensors_int32(self):
    # [FUNCTION DOC]
    # Looks up int32 jersey numbers in the integer vocabulary; asserts ids equal positions with minus
    # one for a missing number.
    with ops.Graph().as_default():
      column = fc._categorical_column_with_vocabulary_file(
          key='aaa',
          vocabulary_file=self._warriors_vocabulary_file_name,
          vocabulary_size=self._warriors_vocabulary_size,
          dtype=dtypes.int32)
      inputs = sparse_tensor.SparseTensorValue(
          indices=((0, 0), (1, 0), (1, 1), (2, 2)),
          values=(11, 100, 30, 22),
          dense_shape=(3, 3))
      id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
      self.assertIsNone(id_weight_pair.weight_tensor)
      with _initialized_session():
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=inputs.indices,
                values=np.array((2, -1, 0, 4), dtype=np.int64),
                dense_shape=inputs.dense_shape),
            id_weight_pair.id_tensor.eval())

  def test_get_sparse_tensors_int32_dense_input(self):
    # [FUNCTION DOC]
    # Feeds a dense integer matrix with minus one as the missing marker and a default value of minus
    # one hundred; asserts the missing marker entries are dropped and unknown numbers map to the
    # default id.
    with ops.Graph().as_default():
      default_value = -100
      column = fc._categorical_column_with_vocabulary_file(
          key='aaa',
          vocabulary_file=self._warriors_vocabulary_file_name,
          vocabulary_size=self._warriors_vocabulary_size,
          dtype=dtypes.int32,
          default_value=default_value)
      id_weight_pair = column._get_sparse_tensors(
          _LazyBuilder({
              'aaa': ((11, -1, -1), (100, 30, -1), (-1, -1, 22))
          }))
      self.assertIsNone(id_weight_pair.weight_tensor)
      with _initialized_session():
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=((0, 0), (1, 0), (1, 1), (2, 2)),
                values=np.array((2, default_value, 0, 4), dtype=np.int64),
                dense_shape=(3, 3)),
            id_weight_pair.id_tensor.eval())

  def test_get_sparse_tensors_int32_with_oov_buckets(self):
    # [FUNCTION DOC]
    # Looks up integers with one hundred oov buckets; asserts known numbers map to positions and
    # unknown ones are hashed into the oov range.
    with ops.Graph().as_default():
      column = fc._categorical_column_with_vocabulary_file(
          key='aaa',
          vocabulary_file=self._warriors_vocabulary_file_name,
          vocabulary_size=self._warriors_vocabulary_size,
          dtype=dtypes.int32,
          num_oov_buckets=100)
      inputs = sparse_tensor.SparseTensorValue(
          indices=((0, 0), (1, 0), (1, 1), (2, 2)),
          values=(11, 100, 30, 22),
          dense_shape=(3, 3))
      id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
      self.assertIsNone(id_weight_pair.weight_tensor)
      with _initialized_session():
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=inputs.indices,
                values=np.array((2, 60, 0, 4), dtype=np.int64),
                dense_shape=inputs.dense_shape),
            id_weight_pair.id_tensor.eval())

  def test_linear_model(self):
    # [FUNCTION DOC]
    # Four-bucket (one oov) vocabulary file column in the linear model function; asserts zero initial
    # state and predictions equal selected weights after assigning, including the oov bucket.
    wire_column = fc._categorical_column_with_vocabulary_file(
        key='wire',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size,
        num_oov_buckets=1)
    self.assertEqual(4, wire_column._num_buckets)
    with ops.Graph().as_default():
      predictions = fc.linear_model({
          wire_column.name: sparse_tensor.SparseTensorValue(
              indices=((0, 0), (1, 0), (1, 1)),
              values=('marlo', 'skywalker', 'omar'),
              dense_shape=(2, 2))
      }, (wire_column,))
      bias = get_linear_model_bias()
      wire_var = get_linear_model_column_var(wire_column)
      with _initialized_session():
        self.assertAllClose((0.,), self.evaluate(bias))
        self.assertAllClose(((0.,), (0.,), (0.,), (0.,)),
                            self.evaluate(wire_var))
        self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
        wire_var.assign(((1.,), (2.,), (3.,), (4.,))).eval()
        # 'marlo' -> 2: wire_var[2] = 3
        # 'skywalker' -> 3, 'omar' -> 0: wire_var[3] + wire_var[0] = 4+1 = 5
        self.assertAllClose(((3.,), (5.,)), self.evaluate(predictions))

  def test_keras_linear_model(self):
    # [FUNCTION DOC]
    # Same as the linear model test but through the Keras-style linear model helper.
    wire_column = fc._categorical_column_with_vocabulary_file(
        key='wire',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size,
        num_oov_buckets=1)
    self.assertEqual(4, wire_column._num_buckets)
    with ops.Graph().as_default():
      predictions = get_keras_linear_model_predictions({
          wire_column.name:
              sparse_tensor.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=('marlo', 'skywalker', 'omar'),
                  dense_shape=(2, 2))
      }, (wire_column,))
      bias = get_linear_model_bias()
      wire_var = get_linear_model_column_var(wire_column)
      with _initialized_session():
        self.assertAllClose((0.,), self.evaluate(bias))
        self.assertAllClose(((0.,), (0.,), (0.,), (0.,)),
                            self.evaluate(wire_var))
        self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
        wire_var.assign(((1.,), (2.,), (3.,), (4.,))).eval()
        # 'marlo' -> 2: wire_var[2] = 3
        # 'skywalker' -> 3, 'omar' -> 0: wire_var[3] + wire_var[0] = 4+1 = 5
        self.assertAllClose(((3.,), (5.,)), self.evaluate(predictions))


class VocabularyListCategoricalColumnTest(test.TestCase):
  # [CLASS DOC]
  # Tests the legacy vocabulary list categorical column: defaults for string and integer lists,
  # validation, parsing, id lookup with default values and oov buckets, integer array vocabularies
  # and linear model use.

  def test_defaults_string(self):
    # [FUNCTION DOC]
    # Asserts defaults for a string list: name, key, scope name, three buckets and string parse spec.
    column = fc._categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    self.assertEqual('aaa', column.name)
    self.assertEqual('aaa', column.key)
    self.assertEqual('aaa', column._var_scope_name)
    self.assertEqual(3, column._num_buckets)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.string)
    }, column._parse_example_spec)

  def test_key_should_be_string(self):
    # [FUNCTION DOC]
    # Asserts that a non-string key raises a value error saying the key must be a string.
    with self.assertRaisesRegex(ValueError, 'key must be a string.'):
      fc._categorical_column_with_vocabulary_list(
          key=('aaa',), vocabulary_list=('omar', 'stringer', 'marlo'))

  def test_defaults_int(self):
    # [FUNCTION DOC]
    # Asserts defaults for an integer list: name, key, scope name, three buckets and an int64 parse
    # spec.
    column = fc._categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=(12, 24, 36))
    self.assertEqual('aaa', column.name)
    self.assertEqual('aaa', column.key)
    self.assertEqual('aaa', column._var_scope_name)
    self.assertEqual(3, column._num_buckets)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int64)
    }, column._parse_example_spec)

  def test_all_constructor_args(self):
    # [FUNCTION DOC]
    # Creates a column with an int32 dtype and a default value; asserts three buckets and an int32
    # parse spec.
    column = fc._categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=(12, 24, 36),
        dtype=dtypes.int32,
        default_value=-99)
    self.assertEqual(3, column._num_buckets)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int32)
    }, column._parse_example_spec)

  def test_deep_copy(self):
    # [FUNCTION DOC]
    # Asserts that original and deep-copied columns keep name, bucket count and parse spec.
    original = fc._categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=(12, 24, 36), dtype=dtypes.int32)
    for column in (original, copy.deepcopy(original)):
      self.assertEqual('aaa', column.name)
      self.assertEqual(3, column._num_buckets)
      self.assertEqual({
          'aaa': parsing_ops.VarLenFeature(dtypes.int32)
      }, column._parse_example_spec)

  def test_invalid_dtype(self):
    # [FUNCTION DOC]
    # Asserts that a float dtype raises a value error saying the dtype must be string or integer.
    with self.assertRaisesRegex(ValueError, 'dtype must be string or integer'):
      fc._categorical_column_with_vocabulary_list(
          key='aaa',
          vocabulary_list=('omar', 'stringer', 'marlo'),
          dtype=dtypes.float32)

  def test_invalid_mapping_dtype(self):
    # [FUNCTION DOC]
    # Asserts that a vocabulary of floats raises a value error saying the vocabulary dtype must be
    # string or integer.
    with self.assertRaisesRegex(ValueError,
                                r'vocabulary dtype must be string or integer'):
      fc._categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=(12., 24., 36.))

  def test_mismatched_int_dtype(self):
    # [FUNCTION DOC]
    # Asserts that a string vocabulary with an int32 dtype raises a value error saying the dtypes do
    # not match.
    with self.assertRaisesRegex(ValueError,
                                r'dtype.*and vocabulary dtype.*do not match'):
      fc._categorical_column_with_vocabulary_list(
          key='aaa',
          vocabulary_list=('omar', 'stringer', 'marlo'),
          dtype=dtypes.int32)

  def test_mismatched_string_dtype(self):
    # [FUNCTION DOC]
    # Asserts that an integer vocabulary with a string dtype raises a value error saying the dtypes do
    # not match.
    with self.assertRaisesRegex(ValueError,
                                r'dtype.*and vocabulary dtype.*do not match'):
      fc._categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=(12, 24, 36), dtype=dtypes.string)

  def test_none_mapping(self):
    # [FUNCTION DOC]
    # Asserts that a missing vocabulary list raises a value error saying it must be non-empty.
    with self.assertRaisesRegex(ValueError,
                                r'vocabulary_list.*must be non-empty'):
      fc._categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=None)

  def test_empty_mapping(self):
    # [FUNCTION DOC]
    # Asserts that an empty vocabulary list raises the same must be non-empty value error.
    with self.assertRaisesRegex(ValueError,
                                r'vocabulary_list.*must be non-empty'):
      fc._categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=tuple([]))

  def test_duplicate_mapping(self):
    # [FUNCTION DOC]
    # Asserts that a vocabulary with a repeated entry raises a value error about duplicate keys.
    with self.assertRaisesRegex(ValueError, 'Duplicate keys'):
      fc._categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=(12, 24, 12))

  def test_invalid_num_oov_buckets(self):
    # [FUNCTION DOC]
    # Asserts that a negative oov bucket count raises a value error.
    with self.assertRaisesRegex(ValueError, 'Invalid num_oov_buckets'):
      fc._categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=(12, 24, 36), num_oov_buckets=-1)

  def test_invalid_buckets_and_default_value(self):
    # [FUNCTION DOC]
    # Asserts that giving both oov buckets and a default value raises a value error.
    with self.assertRaisesRegex(ValueError,
                                'both num_oov_buckets and default_value'):
      fc._categorical_column_with_vocabulary_list(
          key='aaa',
          vocabulary_list=(12, 24, 36),
          num_oov_buckets=100,
          default_value=2)

  def test_invalid_input_dtype_int32(self):
    # [FUNCTION DOC]
    # Asserts that integer inputs to a string vocabulary column raise a value error saying the dtype
    # must be compatible.
    column = fc._categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=(12, 24, 36),
        dense_shape=(2, 2))
    with self.assertRaisesRegex(ValueError, 'dtype must be compatible'):
      column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))

  def test_invalid_input_dtype_string(self):
    # [FUNCTION DOC]
    # Asserts that string inputs to an integer vocabulary column raise a value error saying the dtype
    # must be compatible.
    column = fc._categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=(12, 24, 36))
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('omar', 'stringer', 'marlo'),
        dense_shape=(2, 2))
    with self.assertRaisesRegex(ValueError, 'dtype must be compatible'):
      column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))

  def test_parse_example_string(self):
    # [FUNCTION DOC]
    # Parses a serialized example of byte strings with the column parse spec and asserts the parsed
    # sparse strings.
    a = fc._categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    data = example_pb2.Example(features=feature_pb2.Features(
        feature={
            'aaa':
                feature_pb2.Feature(bytes_list=feature_pb2.BytesList(
                    value=[b'omar', b'stringer']))
        }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec([a]))
    self.assertIn('aaa', features)
    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [0, 1]],
            values=np.array([b'omar', b'stringer'], dtype=np.object_),
            dense_shape=[1, 2]), self.evaluate(features['aaa']))

  def test_parse_example_int(self):
    # [FUNCTION DOC]
    # Parses a serialized example of int64 values with the column parse spec and asserts the parsed
    # sparse integers.
    a = fc._categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=(11, 21, 31))
    data = example_pb2.Example(features=feature_pb2.Features(
        feature={
            'aaa':
                feature_pb2.Feature(int64_list=feature_pb2.Int64List(
                    value=[11, 21]))
        }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec([a]))
    self.assertIn('aaa', features)
    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [0, 1]], values=[11, 21], dense_shape=[1, 2]),
        self.evaluate(features['aaa']))

  def test_get_sparse_tensors(self):
    # [FUNCTION DOC]
    # Looks up strings in the vocabulary list; asserts no weights and ids equal to positions with
    # minus one for the unknown word.
    with ops.Graph().as_default():
      column = fc._categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
      inputs = sparse_tensor.SparseTensorValue(
          indices=((0, 0), (1, 0), (1, 1)),
          values=('marlo', 'skywalker', 'omar'),
          dense_shape=(2, 2))
      id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
      self.assertIsNone(id_weight_pair.weight_tensor)
      with _initialized_session():
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=inputs.indices,
                values=np.array((2, -1, 0), dtype=np.int64),
                dense_shape=inputs.dense_shape),
            id_weight_pair.id_tensor.eval())

  def test_transform_feature(self):
    # [FUNCTION DOC]
    # Transforms strings via the module transform function and asserts position ids with minus one for
    # the unknown word.
    with ops.Graph().as_default():
      column = fc._categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
      inputs = sparse_tensor.SparseTensorValue(
          indices=((0, 0), (1, 0), (1, 1)),
          values=('marlo', 'skywalker', 'omar'),
          dense_shape=(2, 2))
      id_tensor = _transform_features({'aaa': inputs}, [column])[column]
      with _initialized_session():
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=inputs.indices,
                values=np.array((2, -1, 0), dtype=np.int64),
                dense_shape=inputs.dense_shape), self.evaluate(id_tensor))

  def test_get_sparse_tensors_weight_collections(self):
    # [FUNCTION DOC]
    # Asserts that requesting sparse tensors with a weight collection name creates no variables in any
    # collection.
    column = fc._categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    inputs = sparse_tensor.SparseTensor(
        values=['omar', 'stringer', 'marlo'],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    column._get_sparse_tensors(
        _LazyBuilder({
            'aaa': inputs
        }), weight_collections=('my_weights',))

    self.assertCountEqual([],
                          ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES))
    self.assertCountEqual([], ops.get_collection('my_weights'))

  def test_get_sparse_tensors_dense_input(self):
    # [FUNCTION DOC]
    # Feeds a dense nested tuple of strings with an empty string; asserts the empty entry is dropped
    # and unknown words map to minus one.
    with ops.Graph().as_default():
      column = fc._categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
      id_weight_pair = column._get_sparse_tensors(
          _LazyBuilder({'aaa': (('marlo', ''), ('skywalker', 'omar'))}))
      self.assertIsNone(id_weight_pair.weight_tensor)
      with _initialized_session():
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=((0, 0), (1, 0), (1, 1)),
                values=np.array((2, -1, 0), dtype=np.int64),
                dense_shape=(2, 2)), id_weight_pair.id_tensor.eval())

  def test_get_sparse_tensors_default_value_in_vocabulary(self):
    # [FUNCTION DOC]
    # Uses a default id within the vocabulary; asserts unknown words map to it.
    with ops.Graph().as_default():
      column = fc._categorical_column_with_vocabulary_list(
          key='aaa',
          vocabulary_list=('omar', 'stringer', 'marlo'),
          default_value=2)
      inputs = sparse_tensor.SparseTensorValue(
          indices=((0, 0), (1, 0), (1, 1)),
          values=('marlo', 'skywalker', 'omar'),
          dense_shape=(2, 2))
      id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
      self.assertIsNone(id_weight_pair.weight_tensor)
      with _initialized_session():
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=inputs.indices,
                values=np.array((2, 2, 0), dtype=np.int64),
                dense_shape=inputs.dense_shape),
            id_weight_pair.id_tensor.eval())

  def test_get_sparse_tensors_with_oov_buckets(self):
    # [FUNCTION DOC]
    # Uses one hundred oov buckets; asserts known words map to positions and unknown words are hashed
    # into the oov id range.
    with ops.Graph().as_default():
      column = fc._categorical_column_with_vocabulary_list(
          key='aaa',
          vocabulary_list=('omar', 'stringer', 'marlo'),
          num_oov_buckets=100)
      inputs = sparse_tensor.SparseTensorValue(
          indices=((0, 0), (1, 0), (1, 1), (1, 2)),
          values=('marlo', 'skywalker', 'omar', 'heisenberg'),
          dense_shape=(2, 3))
      id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
      self.assertIsNone(id_weight_pair.weight_tensor)
      with _initialized_session():
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=inputs.indices,
                values=np.array((2, 33, 0, 62), dtype=np.int64),
                dense_shape=inputs.dense_shape),
            id_weight_pair.id_tensor.eval())

  def test_get_sparse_tensors_int32(self):
    # [FUNCTION DOC]
    # Uses an int32 numpy array vocabulary; asserts ids equal positions with minus one for a missing
    # number.
    with ops.Graph().as_default():
      column = fc._categorical_column_with_vocabulary_list(
          key='aaa',
          vocabulary_list=np.array((30, 35, 11, 23, 22), dtype=np.int32),
          dtype=dtypes.int32)
      inputs = sparse_tensor.SparseTensorValue(
          indices=((0, 0), (1, 0), (1, 1), (2, 2)),
          values=np.array((11, 100, 30, 22), dtype=np.int32),
          dense_shape=(3, 3))
      id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
      self.assertIsNone(id_weight_pair.weight_tensor)
      with _initialized_session():
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=inputs.indices,
                values=np.array((2, -1, 0, 4), dtype=np.int64),
                dense_shape=inputs.dense_shape),
            id_weight_pair.id_tensor.eval())

  def test_get_sparse_tensors_int32_dense_input(self):
    # [FUNCTION DOC]
    # Feeds a dense int32 matrix using minus one as the missing marker and a default id of minus one
    # hundred; asserts missing entries are dropped and unknown numbers map to the default id.
    with ops.Graph().as_default():
      default_value = -100
      column = fc._categorical_column_with_vocabulary_list(
          key='aaa',
          vocabulary_list=np.array((30, 35, 11, 23, 22), dtype=np.int32),
          dtype=dtypes.int32,
          default_value=default_value)
      id_weight_pair = column._get_sparse_tensors(
          _LazyBuilder({
              'aaa':
                  np.array(((11, -1, -1), (100, 30, -1), (-1, -1, 22)),
                           dtype=np.int32)
          }))
      self.assertIsNone(id_weight_pair.weight_tensor)
      with _initialized_session():
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=((0, 0), (1, 0), (1, 1), (2, 2)),
                values=np.array((2, default_value, 0, 4), dtype=np.int64),
                dense_shape=(3, 3)), id_weight_pair.id_tensor.eval())

  def test_get_sparse_tensors_int32_with_oov_buckets(self):
    # [FUNCTION DOC]
    # Uses an int32 vocabulary with one hundred oov buckets; asserts known numbers map to positions
    # and unknown ones are hashed into the oov range.
    with ops.Graph().as_default():
      column = fc._categorical_column_with_vocabulary_list(
          key='aaa',
          vocabulary_list=np.array((30, 35, 11, 23, 22), dtype=np.int32),
          dtype=dtypes.int32,
          num_oov_buckets=100)
      inputs = sparse_tensor.SparseTensorValue(
          indices=((0, 0), (1, 0), (1, 1), (2, 2)),
          values=(11, 100, 30, 22),
          dense_shape=(3, 3))
      id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
      self.assertIsNone(id_weight_pair.weight_tensor)
      with _initialized_session():
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=inputs.indices,
                values=np.array((2, 60, 0, 4), dtype=np.int64),
                dense_shape=inputs.dense_shape),
            id_weight_pair.id_tensor.eval())

  def test_linear_model(self):
    # [FUNCTION DOC]
    # Four-bucket vocabulary list column in the linear model function; asserts zero initial state and
    # predictions equal selected weights after assigning, including the oov bucket.
    wire_column = fc._categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=('omar', 'stringer', 'marlo'),
        num_oov_buckets=1)
    self.assertEqual(4, wire_column._num_buckets)
    with ops.Graph().as_default():
      predictions = fc.linear_model({
          wire_column.name: sparse_tensor.SparseTensorValue(
              indices=((0, 0), (1, 0), (1, 1)),
              values=('marlo', 'skywalker', 'omar'),
              dense_shape=(2, 2))
      }, (wire_column,))
      bias = get_linear_model_bias()
      wire_var = get_linear_model_column_var(wire_column)
      with _initialized_session():
        self.assertAllClose((0.,), self.evaluate(bias))
        self.assertAllClose(((0.,), (0.,), (0.,), (0.,)),
                            self.evaluate(wire_var))
        self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
        wire_var.assign(((1.,), (2.,), (3.,), (4.,))).eval()
        # 'marlo' -> 2: wire_var[2] = 3
        # 'skywalker' -> 3, 'omar' -> 0: wire_var[3] + wire_var[0] = 4+1 = 5
        self.assertAllClose(((3.,), (5.,)), self.evaluate(predictions))

  def test_keras_linear_model(self):
    # [FUNCTION DOC]
    # Same as the linear model test but through the Keras-style linear model helper.
    wire_column = fc._categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=('omar', 'stringer', 'marlo'),
        num_oov_buckets=1)
    self.assertEqual(4, wire_column._num_buckets)
    with ops.Graph().as_default():
      predictions = get_keras_linear_model_predictions({
          wire_column.name:
              sparse_tensor.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=('marlo', 'skywalker', 'omar'),
                  dense_shape=(2, 2))
      }, (wire_column,))
      bias = get_linear_model_bias()
      wire_var = get_linear_model_column_var(wire_column)
      with _initialized_session():
        self.assertAllClose((0.,), self.evaluate(bias))
        self.assertAllClose(((0.,), (0.,), (0.,), (0.,)),
                            self.evaluate(wire_var))
        self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
        wire_var.assign(((1.,), (2.,), (3.,), (4.,))).eval()
        # 'marlo' -> 2: wire_var[2] = 3
        # 'skywalker' -> 3, 'omar' -> 0: wire_var[3] + wire_var[0] = 4+1 = 5
        self.assertAllClose(((3.,), (5.,)), self.evaluate(predictions))


class IdentityCategoricalColumnTest(test.TestCase):
  # [CLASS DOC]
  # Tests the legacy identity categorical column: constructor, validation of bucket count and
  # default value, input dtype check, parsing, id extraction with defaults for out-of-range values,
  # use in an embedding lookup with out-of-range inputs, and linear model use.

  def test_constructor(self):
    # [FUNCTION DOC]
    # Asserts name, key, scope name, bucket count and the variable length int64 parse spec of an
    # identity column.
    column = fc._categorical_column_with_identity(key='aaa', num_buckets=3)
    self.assertEqual('aaa', column.name)
    self.assertEqual('aaa', column.key)
    self.assertEqual('aaa', column._var_scope_name)
    self.assertEqual(3, column._num_buckets)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int64)
    }, column._parse_example_spec)

  def test_key_should_be_string(self):
    # [FUNCTION DOC]
    # Asserts that a non-string key raises a value error saying the key must be a string.
    with self.assertRaisesRegex(ValueError, 'key must be a string.'):
      fc._categorical_column_with_identity(key=('aaa',), num_buckets=3)

  def test_deep_copy(self):
    # [FUNCTION DOC]
    # Asserts that original and deep-copied columns keep name, bucket count and parse spec.
    original = fc._categorical_column_with_identity(key='aaa', num_buckets=3)
    for column in (original, copy.deepcopy(original)):
      self.assertEqual('aaa', column.name)
      self.assertEqual(3, column._num_buckets)
      self.assertEqual({
          'aaa': parsing_ops.VarLenFeature(dtypes.int64)
      }, column._parse_example_spec)

  def test_invalid_num_buckets_zero(self):
    # [FUNCTION DOC]
    # Asserts that zero buckets raises a value error saying the bucket count is below one.
    with self.assertRaisesRegex(ValueError, 'num_buckets 0 < 1'):
      fc._categorical_column_with_identity(key='aaa', num_buckets=0)

  def test_invalid_num_buckets_negative(self):
    # [FUNCTION DOC]
    # Asserts that a negative bucket count raises a value error saying it is below one.
    with self.assertRaisesRegex(ValueError, 'num_buckets -1 < 1'):
      fc._categorical_column_with_identity(key='aaa', num_buckets=-1)

  def test_invalid_default_value_too_small(self):
    # [FUNCTION DOC]
    # Asserts that a default value of minus one raises a value error saying it is not in range.
    with self.assertRaisesRegex(ValueError, 'default_value -1 not in range'):
      fc._categorical_column_with_identity(
          key='aaa', num_buckets=3, default_value=-1)

  def test_invalid_default_value_too_big(self):
    # [FUNCTION DOC]
    # Asserts that a default value equal to the bucket count raises a value error saying it is not in
    # range.
    with self.assertRaisesRegex(ValueError, 'default_value 3 not in range'):
      fc._categorical_column_with_identity(
          key='aaa', num_buckets=3, default_value=3)

  def test_invalid_input_dtype(self):
    # [FUNCTION DOC]
    # Asserts that string inputs raise a value error saying the input is not an integer.
    column = fc._categorical_column_with_identity(key='aaa', num_buckets=3)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('omar', 'stringer', 'marlo'),
        dense_shape=(2, 2))
    with self.assertRaisesRegex(ValueError, 'Invalid input, not integer'):
      column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))

  def test_parse_example(self):
    # [FUNCTION DOC]
    # Parses a serialized example of two int64 values with the column spec and asserts the parsed
    # sparse ids.
    a = fc._categorical_column_with_identity(key='aaa', num_buckets=30)
    data = example_pb2.Example(features=feature_pb2.Features(
        feature={
            'aaa':
                feature_pb2.Feature(int64_list=feature_pb2.Int64List(
                    value=[11, 21]))
        }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec([a]))
    self.assertIn('aaa', features)
    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [0, 1]],
            values=np.array([11, 21], dtype=np.int64),
            dense_shape=[1, 2]), self.evaluate(features['aaa']))

  def test_get_sparse_tensors(self):
    # [FUNCTION DOC]
    # Asserts that in-range integer ids pass through unchanged as an int64 sparse tensor with no
    # weights.
    with ops.Graph().as_default():
      column = fc._categorical_column_with_identity(key='aaa', num_buckets=3)
      inputs = sparse_tensor.SparseTensorValue(
          indices=((0, 0), (1, 0), (1, 1)),
          values=(0, 1, 0),
          dense_shape=(2, 2))
      id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
      self.assertIsNone(id_weight_pair.weight_tensor)
      with _initialized_session():
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=inputs.indices,
                values=np.array((0, 1, 0), dtype=np.int64),
                dense_shape=inputs.dense_shape),
            id_weight_pair.id_tensor.eval())

  def test_transform_feature(self):
    # [FUNCTION DOC]
    # Transforms in-range ids via the module transform function and asserts they are unchanged and
    # int64.
    with ops.Graph().as_default():
      column = fc._categorical_column_with_identity(key='aaa', num_buckets=3)
      inputs = sparse_tensor.SparseTensorValue(
          indices=((0, 0), (1, 0), (1, 1)),
          values=(0, 1, 0),
          dense_shape=(2, 2))
      id_tensor = _transform_features({'aaa': inputs}, [column])[column]
      with _initialized_session():
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=inputs.indices,
                values=np.array((0, 1, 0), dtype=np.int64),
                dense_shape=inputs.dense_shape), self.evaluate(id_tensor))

  def test_get_sparse_tensors_weight_collections(self):
    # [FUNCTION DOC]
    # Asserts that requesting sparse tensors with a weight collection name creates no variables in any
    # collection.
    column = fc._categorical_column_with_identity(key='aaa', num_buckets=3)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=(0, 1, 0),
        dense_shape=(2, 2))
    column._get_sparse_tensors(
        _LazyBuilder({
            'aaa': inputs
        }), weight_collections=('my_weights',))

    self.assertCountEqual([],
                          ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES))
    self.assertCountEqual([], ops.get_collection('my_weights'))

  def test_get_sparse_tensors_dense_input(self):
    # [FUNCTION DOC]
    # Feeds a dense matrix where minus one marks missing entries; asserts the missing entry is dropped
    # from the resulting sparse ids.
    with ops.Graph().as_default():
      column = fc._categorical_column_with_identity(key='aaa', num_buckets=3)
      id_weight_pair = column._get_sparse_tensors(
          _LazyBuilder({
              'aaa': ((0, -1), (1, 0))
          }))
      self.assertIsNone(id_weight_pair.weight_tensor)
      with _initialized_session():
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=((0, 0), (1, 0), (1, 1)),
                values=np.array((0, 1, 0), dtype=np.int64),
                dense_shape=(2, 2)),
            id_weight_pair.id_tensor.eval())

  def test_get_sparse_tensors_with_inputs_too_big(self):
    # [FUNCTION DOC]
    # Feeds an id equal to the bucket count into an embedding column over the identity column; asserts
    # that evaluating the embedding lookup raises an op error saying the index is outside the valid
    # range.
    with ops.Graph().as_default():
      # Inputs.
      vocabulary_size = 2
      sparse_input = sparse_tensor.SparseTensorValue(
          indices=((0, 0), (1, 0), (1, 1)), values=(2, 1, 0),
          dense_shape=(2, 2))

      # Embedding variable.
      embedding_dimension = 2
      embedding_values = (
          (1., 2.),  # id 0
          (3., 5.),  # id 1
      )

      def _initializer(shape, dtype, partition_info=None):
        # [FUNCTION DOC]
        # HELPER: fixture initializer for the embedding table in this test.
        # Embedding initializer that ignores shape, dtype and partition information and returns a fixed
        # two by two table.
        del shape, dtype, partition_info
        return embedding_values

      # Build columns.
      categorical_column = fc._categorical_column_with_identity(
          key='aaa', num_buckets=vocabulary_size)
      embedding_column = fc._embedding_column(
          categorical_column,
          dimension=embedding_dimension,
          initializer=_initializer)

      # Provide sparse input and get dense result.
      embedding_lookup = embedding_column._get_dense_tensor(
          _LazyBuilder({'aaa': sparse_input}))

      with _initialized_session():
        with self.assertRaisesRegex(errors.OpError,
                                    r'indices\[0\] .* 2 .* \[0, 2\)'):
          self.evaluate(embedding_lookup)

  def test_get_sparse_tensors_with_inputs_too_small(self):
    # [FUNCTION DOC]
    # Feeds negative ids (which are dropped) into an embedding lookup; asserts the lookup result only
    # reflects the valid ids and no error is raised. Note the input contains a duplicated index entry.
    with ops.Graph().as_default():
      # Inputs.
      vocabulary_size = 2
      sparse_input = sparse_tensor.SparseTensorValue(
          indices=((0, 0), (0, 0), (1, 1), (1, 2)),
          values=(-9, 0, -6, 1),
          dense_shape=(2, 4))

      # Embedding variable.
      embedding_dimension = 2
      embedding_values = (
          (1., 2.),  # id 0
          (3., 5.),  # id 1
      )

      def _initializer(shape, dtype, partition_info=None):
        # [FUNCTION DOC]
        # HELPER: fixture initializer for the embedding table in this test.
        # Embedding initializer that ignores shape, dtype and partition information and returns a fixed
        # two by two table.
        del shape, dtype, partition_info
        return embedding_values

      # Build columns.
      categorical_column = fc._categorical_column_with_identity(
          key='aaa', num_buckets=vocabulary_size)
      embedding_column = fc._embedding_column(
          categorical_column,
          dimension=embedding_dimension,
          initializer=_initializer)

      # Provide sparse input and get dense result.
      embedding_lookup = embedding_column._get_dense_tensor(
          _LazyBuilder({'aaa': sparse_input}))
      expected_lookups = ((1., 2.), (3., 5))
      with _initialized_session():
        self.assertAllEqual(expected_lookups, self.evaluate(embedding_lookup))

  def test_get_sparse_tensors_with_default_value(self):
    # [FUNCTION DOC]
    # Uses a default value of three; asserts negative and too large ids are replaced by the default
    # while valid ids are kept.
    with ops.Graph().as_default():
      column = fc._categorical_column_with_identity(
          key='aaa', num_buckets=4, default_value=3)
      inputs = sparse_tensor.SparseTensorValue(
          indices=((0, 0), (1, 0), (1, 1)),
          values=(1, -1, 99),
          dense_shape=(2, 2))
      id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
      self.assertIsNone(id_weight_pair.weight_tensor)
      with _initialized_session():
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=inputs.indices,
                values=np.array((1, 3, 3), dtype=np.int64),
                dense_shape=inputs.dense_shape),
            id_weight_pair.id_tensor.eval())

  def test_get_sparse_tensors_with_default_value_and_placeholder_inputs(self):
    # [FUNCTION DOC]
    # Same as the default value test, but indices, values and shape are placeholders fed at run time;
    # asserts the same replacement of out-of-range ids by the default.
    with ops.Graph().as_default():
      column = fc._categorical_column_with_identity(
          key='aaa', num_buckets=4, default_value=3)
      input_indices = array_ops.placeholder(dtype=dtypes.int64)
      input_values = array_ops.placeholder(dtype=dtypes.int32)
      input_shape = array_ops.placeholder(dtype=dtypes.int64)
      inputs = sparse_tensor.SparseTensorValue(
          indices=input_indices,
          values=input_values,
          dense_shape=input_shape)
      id_weight_pair = column._get_sparse_tensors(_LazyBuilder({'aaa': inputs}))
      self.assertIsNone(id_weight_pair.weight_tensor)
      with _initialized_session():
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=np.array(((0, 0), (1, 0), (1, 1)), dtype=np.int64),
                values=np.array((1, 3, 3), dtype=np.int64),
                dense_shape=np.array((2, 2), dtype=np.int64)),
            id_weight_pair.id_tensor.eval(feed_dict={
                input_indices: ((0, 0), (1, 0), (1, 1)),
                input_values: (1, -1, 99),
                input_shape: (2, 2),
            }))

  def test_linear_model(self):
    # [FUNCTION DOC]
    # Three-bucket identity column in the linear model function; asserts zero initial state and sums
    # of selected weights after assigning.
    column = fc._categorical_column_with_identity(key='aaa', num_buckets=3)
    self.assertEqual(3, column._num_buckets)
    with ops.Graph().as_default():
      predictions = fc.linear_model({
          column.name: sparse_tensor.SparseTensorValue(
              indices=((0, 0), (1, 0), (1, 1)),
              values=(0, 2, 1),
              dense_shape=(2, 2))
      }, (column,))
      bias = get_linear_model_bias()
      weight_var = get_linear_model_column_var(column)
      with _initialized_session():
        self.assertAllClose((0.,), self.evaluate(bias))
        self.assertAllClose(((0.,), (0.,), (0.,)), self.evaluate(weight_var))
        self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
        weight_var.assign(((1.,), (2.,), (3.,))).eval()
        # weight_var[0] = 1
        # weight_var[2] + weight_var[1] = 3+2 = 5
        self.assertAllClose(((1.,), (5.,)), self.evaluate(predictions))

  def test_keras_linear_model(self):
    # [FUNCTION DOC]
    # Same as the linear model test but through the Keras-style linear model helper.
    column = fc._categorical_column_with_identity(key='aaa', num_buckets=3)
    self.assertEqual(3, column._num_buckets)
    with ops.Graph().as_default():
      predictions = get_keras_linear_model_predictions({
          column.name:
              sparse_tensor.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(0, 2, 1),
                  dense_shape=(2, 2))
      }, (column,))
      bias = get_linear_model_bias()
      weight_var = get_linear_model_column_var(column)
      with _initialized_session():
        self.assertAllClose((0.,), self.evaluate(bias))
        self.assertAllClose(((0.,), (0.,), (0.,)), self.evaluate(weight_var))
        self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
        weight_var.assign(((1.,), (2.,), (3.,))).eval()
        # weight_var[0] = 1
        # weight_var[2] + weight_var[1] = 3+2 = 5
        self.assertAllClose(((1.,), (5.,)), self.evaluate(predictions))


class TransformFeaturesTest(test.TestCase):
  # [CLASS DOC]
  # Tests the module level transform features function. The comment notes that individual transform
  # tests live in the column test classes; this class only covers a multi column case and the
  # evaluation order of columns.

  # All transform tests are distributed in column test.
  # Here we only test multi column case and naming
  def transform_multi_column(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the method name does not start with the test prefix, so the test runner never
    # collects or runs it; the multi column transform check is effectively dead code.
    # Intended scenario: transform a bucketized price column and a hashed string column together and
    # assert each output tensor name contains its column name and the values equal the expected bucket
    # ids and hashes.
    bucketized_price = fc._bucketized_column(
        fc._numeric_column('price'), boundaries=[0, 2, 4, 6])
    hashed_sparse = fc._categorical_column_with_hash_bucket('wire', 10)
    with ops.Graph().as_default():
      features = {
          'price': [[-1.], [5.]],
          'wire':
              sparse_tensor.SparseTensor(
                  values=['omar', 'stringer', 'marlo'],
                  indices=[[0, 0], [1, 0], [1, 1]],
                  dense_shape=[2, 2])
      }
      transformed = _transform_features(features,
                                        [bucketized_price, hashed_sparse])
      with _initialized_session():
        self.assertIn(bucketized_price.name, transformed[bucketized_price].name)
        self.assertAllEqual([[0], [3]], transformed[bucketized_price])
        self.assertIn(hashed_sparse.name, transformed[hashed_sparse].name)
        self.assertAllEqual([6, 4, 1], transformed[hashed_sparse].values)

  def test_column_order(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring about a column being both dense and sparse was copied from another
    # test and does not describe this test, which checks transform call order.
    # Scenario: two fixture columns that record the order in which their transform is called.
    # Transforms the columns in one order and then in the reverse order, resetting the counter in
    # between. In both runs it asserts the first column has call order zero and the second has call
    # order one, i.e. the transform order follows the sorted column order rather than the order of the
    # given list.
    """When the column is both dense and sparse, uses sparse tensors."""

    class _LoggerColumn(_FeatureColumn):
      # [CLASS DOC]
      # Local fixture feature column that records the order in which its transform hook is called, using
      # a counter shared via a mapping in the enclosing test.

      def __init__(self, name):
        # [FUNCTION DOC]
        # Constructor storing the column name given as the only argument. Returns nothing.
        self._name = name

      @property
      def name(self):
        # [FUNCTION DOC]
        # Property returning the stored column name.
        return self._name

      def _transform_feature(self, inputs):
        # [FUNCTION DOC]
        # Transform hook that ignores its inputs, records the current counter value as the call order of
        # this column, increments the shared counter and returns a constant string.
        del inputs
        self.call_order = call_logger['count']
        call_logger['count'] += 1
        return 'Anything'

      @property
      def _parse_example_spec(self):
        # [FUNCTION DOC]
        # Property required by the interface; returns nothing.
        pass

    with ops.Graph().as_default():
      column1 = _LoggerColumn('1')
      column2 = _LoggerColumn('2')
      call_logger = {'count': 0}
      _transform_features({}, [column1, column2])
      self.assertEqual(0, column1.call_order)
      self.assertEqual(1, column2.call_order)

      call_logger = {'count': 0}
      _transform_features({}, [column2, column1])
      self.assertEqual(0, column1.call_order)
      self.assertEqual(1, column2.call_order)


class IndicatorColumnTest(test.TestCase):
  # [CLASS DOC]
  # Tests the legacy indicator column that converts a categorical column into multi-hot dense
  # vectors: naming and shape, 1D and 2D inputs, multi-hot counts, deep copy, parsing,
  # transformation including weighted categorical columns and unknown values, and use in linear
  # models and the input layer.

  def test_indicator_column(self):
    # [FUNCTION DOC]
    # Asserts for two hashed columns that the indicator column exposes the wrapped column, has a name
    # and scope name with an indicator suffix and a variable shape of one by the bucket count.
    a = fc._categorical_column_with_hash_bucket('a', 4)
    indicator_a = fc._indicator_column(a)
    self.assertEqual(indicator_a.categorical_column.name, 'a')
    self.assertEqual(indicator_a.name, 'a_indicator')
    self.assertEqual(indicator_a._var_scope_name, 'a_indicator')
    self.assertEqual(indicator_a._variable_shape, [1, 4])

    b = fc._categorical_column_with_hash_bucket('b', hash_bucket_size=100)
    indicator_b = fc._indicator_column(b)
    self.assertEqual(indicator_b.categorical_column.name, 'b')
    self.assertEqual(indicator_b.name, 'b_indicator')
    self.assertEqual(indicator_b._var_scope_name, 'b_indicator')
    self.assertEqual(indicator_b._variable_shape, [1, 100])

  def test_1D_shape_succeeds(self):
    # [FUNCTION DOC]
    # Feeds a rank one list of repeated strings; asserts the indicator output is one-hot rows with the
    # hashed bucket set.
    animal = fc._indicator_column(
        fc._categorical_column_with_hash_bucket('animal', 4))
    builder = _LazyBuilder({'animal': ['fox', 'fox']})
    output = builder.get(animal)
    with self.cached_session():
      self.assertAllEqual([[0., 0., 1., 0.], [0., 0., 1., 0.]],
                          self.evaluate(output))

  def test_2D_shape_succeeds(self):
    # [FUNCTION DOC]
    # Feeds a rank two sparse tensor of strings; asserts the same one-hot rows as the one-dimensional
    # case.
    # TODO(ispir/cassandrax): Switch to categorical_column_with_keys when ready.
    animal = fc._indicator_column(
        fc._categorical_column_with_hash_bucket('animal', 4))
    builder = _LazyBuilder({
        'animal':
            sparse_tensor.SparseTensor(
                indices=[[0, 0], [1, 0]],
                values=['fox', 'fox'],
                dense_shape=[2, 1])
    })
    output = builder.get(animal)
    with self.cached_session():
      self.assertAllEqual([[0., 0., 1., 0.], [0., 0., 1., 0.]],
                          self.evaluate(output))

  def test_multi_hot(self):
    # [FUNCTION DOC]
    # Feeds a repeated id into an identity column indicator and asserts the count in that position is
    # two, showing counts rather than binary values.
    animal = fc._indicator_column(
        fc._categorical_column_with_identity('animal', num_buckets=4))

    builder = _LazyBuilder({
        'animal':
            sparse_tensor.SparseTensor(
                indices=[[0, 0], [0, 1]], values=[1, 1], dense_shape=[1, 2])
    })
    output = builder.get(animal)
    with self.cached_session():
      self.assertAllEqual([[0., 2., 0., 0.]], self.evaluate(output))

  def test_multi_hot2(self):
    # [FUNCTION DOC]
    # Feeds two different ids and asserts a multi-hot row with ones at those positions.
    animal = fc._indicator_column(
        fc._categorical_column_with_identity('animal', num_buckets=4))
    builder = _LazyBuilder({
        'animal':
            sparse_tensor.SparseTensor(
                indices=[[0, 0], [0, 1]], values=[1, 2], dense_shape=[1, 2])
    })
    output = builder.get(animal)
    with self.cached_session():
      self.assertAllEqual([[0., 1., 1., 0.]], self.evaluate(output))

  def test_deep_copy(self):
    # [FUNCTION DOC]
    # Deep-copies an indicator column and asserts the copy keeps the wrapped column name, the
    # indicator name and the variable shape.
    a = fc._categorical_column_with_hash_bucket('a', 4)
    column = fc._indicator_column(a)
    column_copy = copy.deepcopy(column)
    self.assertEqual(column_copy.categorical_column.name, 'a')
    self.assertEqual(column.name, 'a_indicator')
    self.assertEqual(column._variable_shape, [1, 4])

  def test_parse_example(self):
    # [FUNCTION DOC]
    # Parses a serialized example using the spec of an indicator column over a vocabulary column and
    # asserts the parsed sparse strings.
    a = fc._categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    a_indicator = fc._indicator_column(a)
    data = example_pb2.Example(features=feature_pb2.Features(
        feature={
            'aaa':
                feature_pb2.Feature(bytes_list=feature_pb2.BytesList(
                    value=[b'omar', b'stringer']))
        }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec([a_indicator]))
    self.assertIn('aaa', features)
    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [0, 1]],
            values=np.array([b'omar', b'stringer'], dtype=np.object_),
            dense_shape=[1, 2]), self.evaluate(features['aaa']))

  def test_transform(self):
    # [FUNCTION DOC]
    # Transforms string inputs through a vocabulary list indicator column and asserts the exact
    # multi-hot rows, with unknown words ignored.
    with ops.Graph().as_default():
      a = fc._categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
      a_indicator = fc._indicator_column(a)
      features = {
          'aaa': sparse_tensor.SparseTensorValue(
              indices=((0, 0), (1, 0), (1, 1)),
              values=('marlo', 'skywalker', 'omar'),
              dense_shape=(2, 2))
      }
      indicator_tensor = _transform_features(features,
                                             [a_indicator])[a_indicator]
      with _initialized_session():
        self.assertAllEqual([[0, 0, 1], [1, 0, 0]],
                            self.evaluate(indicator_tensor))

  def test_transform_with_weighted_column(self):
    # [FUNCTION DOC]
    # Regression scenario for a reported issue: an indicator over a weighted categorical column with
    # repeated ids; asserts the indicator output sums weights per id.
    with ops.Graph().as_default():
      # Github issue 12557
      ids = fc._categorical_column_with_vocabulary_list(
          key='ids', vocabulary_list=('a', 'b', 'c'))
      weights = fc._weighted_categorical_column(ids, 'weights')
      indicator = fc._indicator_column(weights)
      features = {
          'ids': constant_op.constant([['c', 'b', 'a', 'c']]),
          'weights': constant_op.constant([[2., 4., 6., 1.]])
      }
      indicator_tensor = _transform_features(features, [indicator])[indicator]
      with _initialized_session():
        self.assertAllEqual([[6., 4., 3.]], self.evaluate(indicator_tensor))

  def test_transform_with_missing_value_in_weighted_column(self):
    # [FUNCTION DOC]
    # Regression scenario for a reported issue: an unknown id in a weighted column; asserts the
    # unknown entry contributes nothing while others keep their weights.
    with ops.Graph().as_default():
      # Github issue 12583
      ids = fc._categorical_column_with_vocabulary_list(
          key='ids', vocabulary_list=('a', 'b', 'c'))
      weights = fc._weighted_categorical_column(ids, 'weights')
      indicator = fc._indicator_column(weights)
      features = {
          'ids': constant_op.constant([['c', 'b', 'unknown']]),
          'weights': constant_op.constant([[2., 4., 6.]])
      }
      indicator_tensor = _transform_features(features, [indicator])[indicator]
      with _initialized_session():
        self.assertAllEqual([[0., 4., 2.]], self.evaluate(indicator_tensor))

  def test_transform_with_missing_value_in_categorical_column(self):
    # [FUNCTION DOC]
    # Regression scenario for a reported issue: an unknown id in a plain categorical column; asserts
    # the output has ones only for the known ids.
    with ops.Graph().as_default():
      # Github issue 12583
      ids = fc._categorical_column_with_vocabulary_list(
          key='ids', vocabulary_list=('a', 'b', 'c'))
      indicator = fc._indicator_column(ids)
      features = {
          'ids': constant_op.constant([['c', 'b', 'unknown']]),
      }
      indicator_tensor = _transform_features(features, [indicator])[indicator]
      with _initialized_session():
        self.assertAllEqual([[0., 1., 1.]], self.evaluate(indicator_tensor))

  def test_linear_model(self):
    # [FUNCTION DOC]
    # Indicator over an identity column in the linear model function; asserts zero weights initially
    # and that predictions sum the weights of the active ids after assigning.
    animal = fc._indicator_column(
        fc._categorical_column_with_identity('animal', num_buckets=4))
    with ops.Graph().as_default():
      features = {
          'animal':
              sparse_tensor.SparseTensor(
                  indices=[[0, 0], [0, 1]], values=[1, 2], dense_shape=[1, 2])
      }

      predictions = fc.linear_model(features, [animal])
      weight_var = get_linear_model_column_var(animal)
      with _initialized_session():
        # All should be zero-initialized.
        self.assertAllClose([[0.], [0.], [0.], [0.]], self.evaluate(weight_var))
        self.assertAllClose([[0.]], self.evaluate(predictions))
        weight_var.assign([[1.], [2.], [3.], [4.]]).eval()
        self.assertAllClose([[2. + 3.]], self.evaluate(predictions))

  def test_keras_linear_model(self):
    # [FUNCTION DOC]
    # Same as the linear model test but through the Keras-style linear model helper.
    animal = fc._indicator_column(
        fc._categorical_column_with_identity('animal', num_buckets=4))
    with ops.Graph().as_default():
      features = {
          'animal':
              sparse_tensor.SparseTensor(
                  indices=[[0, 0], [0, 1]], values=[1, 2], dense_shape=[1, 2])
      }

      predictions = get_keras_linear_model_predictions(features, [animal])
      weight_var = get_linear_model_column_var(animal)
      with _initialized_session():
        # All should be zero-initialized.
        self.assertAllClose([[0.], [0.], [0.], [0.]], self.evaluate(weight_var))
        self.assertAllClose([[0.]], self.evaluate(predictions))
        weight_var.assign([[1.], [2.], [3.], [4.]]).eval()
        self.assertAllClose([[2. + 3.]], self.evaluate(predictions))

  def test_input_layer(self):
    # [FUNCTION DOC]
    # Passes an indicator column to the input layer function and asserts the output is the multi-hot
    # row.
    animal = fc._indicator_column(
        fc._categorical_column_with_identity('animal', num_buckets=4))
    with ops.Graph().as_default():
      features = {
          'animal':
              sparse_tensor.SparseTensor(
                  indices=[[0, 0], [0, 1]], values=[1, 2], dense_shape=[1, 2])
      }
      net = fc.input_layer(features, [animal])
      with _initialized_session():
        self.assertAllClose([[0., 1., 1., 0.]], self.evaluate(net))


class EmbeddingColumnTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests the legacy embedding column (also with parameterized cases): defaults and constructor
  # arguments, deep copy, validation, parsing, transformation, dense tensor lookup with and without
  # safe lookup and partitioned variables, three dimensional inputs, collections, placeholder
  # inputs, checkpoint restore, and use in linear models and the input layer.

  def test_defaults(self):
    # [FUNCTION DOC]
    # Asserts default attributes of an embedding column: wrapped column, dimension, mean combiner, no
    # checkpoint settings, no max norm, trainable, embedding suffix name and scope name, variable
    # shape equal to the dimension and the parse spec of the wrapped column.
    categorical_column = fc._categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_dimension = 2
    embedding_column = fc._embedding_column(
        categorical_column, dimension=embedding_dimension)
    self.assertIs(categorical_column, embedding_column.categorical_column)
    self.assertEqual(embedding_dimension, embedding_column.dimension)
    self.assertEqual('mean', embedding_column.combiner)
    self.assertIsNone(embedding_column.ckpt_to_load_from)
    self.assertIsNone(embedding_column.tensor_name_in_ckpt)
    self.assertIsNone(embedding_column.max_norm)
    self.assertTrue(embedding_column.trainable)
    self.assertEqual('aaa_embedding', embedding_column.name)
    self.assertEqual('aaa_embedding', embedding_column._var_scope_name)
    self.assertEqual(
        (embedding_dimension,), embedding_column._variable_shape)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int64)
    }, embedding_column._parse_example_spec)

  def test_all_constructor_args(self):
    # [FUNCTION DOC]
    # Creates an embedding column with every optional argument set (combiner, initializer, checkpoint
    # path and tensor name, max norm, not trainable) and asserts all are exposed as attributes.
    categorical_column = fc._categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_dimension = 2
    embedding_column = fc._embedding_column(
        categorical_column,
        dimension=embedding_dimension,
        combiner='my_combiner',
        initializer=lambda: 'my_initializer',
        ckpt_to_load_from='my_ckpt',
        tensor_name_in_ckpt='my_ckpt_tensor',
        max_norm=42.,
        trainable=False)
    self.assertIs(categorical_column, embedding_column.categorical_column)
    self.assertEqual(embedding_dimension, embedding_column.dimension)
    self.assertEqual('my_combiner', embedding_column.combiner)
    self.assertEqual('my_ckpt', embedding_column.ckpt_to_load_from)
    self.assertEqual('my_ckpt_tensor', embedding_column.tensor_name_in_ckpt)
    self.assertEqual(42., embedding_column.max_norm)
    self.assertFalse(embedding_column.trainable)
    self.assertEqual('aaa_embedding', embedding_column.name)
    self.assertEqual('aaa_embedding', embedding_column._var_scope_name)
    self.assertEqual(
        (embedding_dimension,), embedding_column._variable_shape)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int64)
    }, embedding_column._parse_example_spec)

  def test_deep_copy(self):
    # [FUNCTION DOC]
    # Deep-copies an embedding column with all arguments set and asserts both copies expose the same
    # wrapped column and attributes.
    categorical_column = fc._categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_dimension = 2
    original = fc._embedding_column(
        categorical_column,
        dimension=embedding_dimension,
        combiner='my_combiner',
        initializer=lambda: 'my_initializer',
        ckpt_to_load_from='my_ckpt',
        tensor_name_in_ckpt='my_ckpt_tensor',
        max_norm=42.,
        trainable=False)
    for embedding_column in (original, copy.deepcopy(original)):
      self.assertEqual('aaa', embedding_column.categorical_column.name)
      self.assertEqual(3, embedding_column.categorical_column._num_buckets)
      self.assertEqual({
          'aaa': parsing_ops.VarLenFeature(dtypes.int64)
      }, embedding_column.categorical_column._parse_example_spec)

      self.assertEqual(embedding_dimension, embedding_column.dimension)
      self.assertEqual('my_combiner', embedding_column.combiner)
      self.assertEqual('my_ckpt', embedding_column.ckpt_to_load_from)
      self.assertEqual('my_ckpt_tensor', embedding_column.tensor_name_in_ckpt)
      self.assertEqual(42., embedding_column.max_norm)
      self.assertFalse(embedding_column.trainable)
      self.assertEqual('aaa_embedding', embedding_column.name)
      self.assertEqual(
          (embedding_dimension,), embedding_column._variable_shape)
      self.assertEqual({
          'aaa': parsing_ops.VarLenFeature(dtypes.int64)
      }, embedding_column._parse_example_spec)

  def test_invalid_initializer(self):
    # [FUNCTION DOC]
    # Asserts that a non-callable initializer raises a value error saying the initializer must be
    # callable.
    categorical_column = fc._categorical_column_with_identity(
        key='aaa', num_buckets=3)
    with self.assertRaisesRegex(ValueError, 'initializer must be callable'):
      fc._embedding_column(
          categorical_column, dimension=2, initializer='not_fn')

  def test_parse_example(self):
    # [FUNCTION DOC]
    # Parses a serialized example with the spec of an embedding column over a vocabulary column and
    # asserts the parsed sparse strings.
    a = fc._categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    a_embedded = fc._embedding_column(a, dimension=2)
    data = example_pb2.Example(features=feature_pb2.Features(
        feature={
            'aaa':
                feature_pb2.Feature(bytes_list=feature_pb2.BytesList(
                    value=[b'omar', b'stringer']))
        }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec([a_embedded]))
    self.assertIn('aaa', features)
    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [0, 1]],
            values=np.array([b'omar', b'stringer'], dtype=np.object_),
            dense_shape=[1, 2]), self.evaluate(features['aaa']))

  def test_transform_feature(self):
    # [FUNCTION DOC]
    # Transforms an identity column and its embedding column together and asserts both transformed
    # outputs hold the same sparse ids.
    with ops.Graph().as_default():
      a = fc._categorical_column_with_identity(key='aaa', num_buckets=3)
      a_embedded = fc._embedding_column(a, dimension=2)
      features = {
          'aaa': sparse_tensor.SparseTensor(
              indices=((0, 0), (1, 0), (1, 1)),
              values=(0, 1, 0),
              dense_shape=(2, 2))
      }
      outputs = _transform_features(features, [a, a_embedded])
      output_a = outputs[a]
      output_embedded = outputs[a_embedded]
      with _initialized_session():
        _assert_sparse_tensor_value(self, self.evaluate(output_a),
                                    self.evaluate(output_embedded))

  @parameterized.named_parameters(
      {
          'testcase_name': 'use_safe_embedding_lookup',
          'use_safe_embedding_lookup': True,
          'partition_variables': False,
      }, {
          'testcase_name': 'dont_use_safe_embedding_lookup',
          'use_safe_embedding_lookup': False,
          'partition_variables': False,
      }, {
          'testcase_name': 'use_safe_embedding_lookup_partitioned',
          'use_safe_embedding_lookup': True,
          'partition_variables': True,
      }, {
          'testcase_name': 'dont_use_safe_embedding_lookup_partitioned',
          'use_safe_embedding_lookup': False,
          'partition_variables': True,
      })

  def test_get_dense_tensor(self, use_safe_embedding_lookup,
                            partition_variables):
    # [FUNCTION DOC]
    # Parameterized over using safe embedding lookup (on or off) and partitioning variables into two
    # shards (on or off). Builds an embedding column with an asserting initializer, looks up ids with
    # an empty example, and asserts variable names, variable contents, expected mean combined
    # embeddings (zero for the empty example) and that the fill empty rows operation exists in the
    # graph only when safe lookup is used. Parameters: whether to use safe embedding lookup and
    # whether to partition variables.
    with ops.Graph().as_default():
      # Inputs.
      vocabulary_size = 4
      sparse_input = sparse_tensor.SparseTensorValue(
          # example 0, ids [2]
          # example 1, ids [0, 1]
          # example 2, ids []
          # example 3, ids [1]
          indices=((0, 0), (1, 0), (1, 4), (3, 0)),
          values=(2, 0, 1, 1),
          dense_shape=(4, 5))

      # Embedding variable.
      embedding_dimension = 2
      embedding_values = (
          (1., 2.),  # id 0
          (3., 5.),  # id 1
          (7., 11.),  # id 2
          (9., 13.)  # id 3
      )

      def _initializer(shape, dtype, partition_info=None):
        # [FUNCTION DOC]
        # HELPER: fixture initializer that also validates how the variable is created.
        # Embedding initializer that asserts the dtype is float32. In the partitioned case it asserts
        # partition information is present with the full shape and the shard shape is half the rows, and
        # returns the matching slice of the fixed table; otherwise it asserts the full shape and no
        # partition information and returns the whole fixed four by two table.
        self.assertEqual(dtypes.float32, dtype)
        if partition_variables:
          assert partition_info is not None
          self.assertEqual([vocabulary_size, embedding_dimension],
                           partition_info.full_shape)
          self.assertAllEqual((2, embedding_dimension), shape)
          return array_ops.slice(
              embedding_values, partition_info.var_offset, shape
          )
        else:
          self.assertAllEqual((vocabulary_size, embedding_dimension), shape)
          self.assertIsNone(partition_info)
          return embedding_values

      # Expected lookup result, using combiner='mean'.
      expected_lookups = (
          # example 0, ids [2], embedding = [7, 11]
          (7., 11.),
          # example 1, ids [0, 1], embedding = mean([1, 2] + [3, 5]) = [2, 3.5]
          (2., 3.5),
          # example 2, ids [], embedding = [0, 0]
          (0., 0.),
          # example 3, ids [1], embedding = [3, 5]
          (3., 5.),
      )

      # Build columns.
      categorical_column = fc._categorical_column_with_identity(
          key='aaa', num_buckets=vocabulary_size)
      partitioner = None
      if partition_variables:
        partitioner = partitioned_variables.fixed_size_partitioner(2, axis=0)
      with variable_scope.variable_scope('vars', partitioner=partitioner):
        embedding_column = fc._embedding_column(
            categorical_column,
            dimension=embedding_dimension,
            initializer=_initializer,
            use_safe_embedding_lookup=use_safe_embedding_lookup)

        # Provide sparse input and get dense result.
        embedding_lookup = embedding_column._get_dense_tensor(
            _LazyBuilder({'aaa': sparse_input}))

      # Assert expected embedding variable and lookups.
      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      if partition_variables:
        self.assertCountEqual(('vars/embedding_weights/part_0:0',
                               'vars/embedding_weights/part_1:0'),
                              tuple([v.name for v in global_vars]))
      else:
        self.assertCountEqual(('vars/embedding_weights:0',),
                              tuple([v.name for v in global_vars]))
      for v in global_vars:
        self.assertIsInstance(v, variables_lib.Variable)
      with _initialized_session():
        if partition_variables:
          self.assertAllEqual(
              embedding_values, array_ops.concat(global_vars, axis=0)
          )
        else:
          self.assertAllEqual(embedding_values, global_vars[0])
        self.assertAllEqual(expected_lookups, self.evaluate(embedding_lookup))

      if use_safe_embedding_lookup:
        self.assertIn(
            'SparseFillEmptyRows',
            [x.type for x in ops.get_default_graph().get_operations()])
      else:
        self.assertNotIn(
            'SparseFillEmptyRows',
            [x.type for x in ops.get_default_graph().get_operations()])

  def test_get_dense_tensor_3d(self):
    # [FUNCTION DOC]
    # Looks up a rank three sparse input with an asserting initializer; asserts one variable and that
    # the result is a rank three tensor holding per inner row mean combined embeddings, with zero rows
    # for empty entries.
    with ops.Graph().as_default():
      # Inputs.
      vocabulary_size = 4
      sparse_input = sparse_tensor.SparseTensorValue(
          # example 0, ids [2]
          # example 1, ids [0, 1]
          # example 2, ids []
          # example 3, ids [1]
          indices=((0, 0, 0), (1, 1, 0), (1, 1, 4), (3, 0, 0), (3, 1, 2)),
          values=(2, 0, 1, 1, 2),
          dense_shape=(4, 2, 5))

      # Embedding variable.
      embedding_dimension = 3
      embedding_values = (
          (1., 2., 4.),  # id 0
          (3., 5., 1.),  # id 1
          (7., 11., 2.),  # id 2
          (2., 7., 12.)  # id 3
      )

      def _initializer(shape, dtype, partition_info):
        # [FUNCTION DOC]
        # HELPER: fixture initializer that also validates how the variable is created.
        # Embedding initializer asserting the expected full shape, float32 dtype and no partition
        # information; returns a fixed four by three table.
        self.assertAllEqual((vocabulary_size, embedding_dimension), shape)
        self.assertEqual(dtypes.float32, dtype)
        self.assertIsNone(partition_info)
        return embedding_values

      # Expected lookup result, using combiner='mean'.
      expected_lookups = (
          # example 0, ids [[2], []], embedding = [[7, 11, 2], [0, 0, 0]]
          ((7., 11., 2.), (0., 0., 0.)),
          # example 1, ids [[], [0, 1]], embedding
          # = mean([[], [1, 2, 4] + [3, 5, 1]]) = [[0, 0, 0], [2, 3.5, 2.5]]
          ((0., 0., 0.), (2., 3.5, 2.5)),
          # example 2, ids [[], []], embedding = [[0, 0, 0], [0, 0, 0]]
          ((0., 0., 0.), (0., 0., 0.)),
          # example 3, ids [[1], [2]], embedding = [[3, 5, 1], [7, 11, 2]]
          ((3., 5., 1.), (7., 11., 2.)),
      )

      # Build columns.
      categorical_column = fc._categorical_column_with_identity(
          key='aaa', num_buckets=vocabulary_size)
      embedding_column = fc._embedding_column(
          categorical_column,
          dimension=embedding_dimension,
          initializer=_initializer)

      # Provide sparse input and get dense result.
      embedding_lookup = embedding_column._get_dense_tensor(
          _LazyBuilder({'aaa': sparse_input}))

      # Assert expected embedding variable and lookups.
      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertCountEqual(('embedding_weights:0',),
                            tuple([v.name for v in global_vars]))
      with _initialized_session():
        self.assertAllEqual(embedding_values, global_vars[0])
        self.assertAllEqual(expected_lookups, self.evaluate(embedding_lookup))

  def test_get_dense_tensor_weight_collections(self):
    # [FUNCTION DOC]
    # Requests the dense tensor with a named weight collection and asserts the embedding variable is
    # in both the global variables collection and the named collection.
    with ops.Graph().as_default():
      sparse_input = sparse_tensor.SparseTensorValue(
          # example 0, ids [2]
          # example 1, ids [0, 1]
          # example 2, ids []
          # example 3, ids [1]
          indices=((0, 0), (1, 0), (1, 4), (3, 0)),
          values=(2, 0, 1, 1),
          dense_shape=(4, 5))

      # Build columns.
      categorical_column = fc._categorical_column_with_identity(
          key='aaa', num_buckets=3)
      embedding_column = fc._embedding_column(categorical_column, dimension=2)

      # Provide sparse input and get dense result.
      embedding_column._get_dense_tensor(
          _LazyBuilder({'aaa': sparse_input}), weight_collections=('my_vars',))

      # Assert expected embedding variable and lookups.
      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertCountEqual(('embedding_weights:0',),
                            tuple([v.name for v in global_vars]))
      my_vars = ops.get_collection('my_vars')
      self.assertCountEqual(('embedding_weights:0',),
                            tuple([v.name for v in my_vars]))

  @test_util.run_deprecated_v1
  # Placeholders are TF1. Replacing with tf.function not feasible because of V1
  # variable creation.
  def test_get_dense_tensor_placeholder_inputs(self):
    # [FUNCTION DOC]
    # Scenario (v1 graph mode): sparse index, value and shape placeholders fed at run time. Asserts
    # the single embedding variable and that the evaluated lookup equals the expected mean combined
    # embeddings.
    # Inputs.
    vocabulary_size = 3
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        # example 2, ids []
        # example 3, ids [1]
        indices=((0, 0), (1, 0), (1, 4), (3, 0)),
        values=(2, 0, 1, 1),
        dense_shape=(4, 5))

    # Embedding variable.
    embedding_dimension = 2
    embedding_values = (
        (1., 2.),  # id 0
        (3., 5.),  # id 1
        (7., 11.)  # id 2
    )
    def _initializer(shape, dtype, partition_info):
      # [FUNCTION DOC]
      # HELPER: fixture initializer that also validates how the variable is created.
      # Embedding initializer asserting expected shape, float32 dtype and no partition information;
      # returns a fixed three by two table.
      self.assertAllEqual((vocabulary_size, embedding_dimension), shape)
      self.assertEqual(dtypes.float32, dtype)
      self.assertIsNone(partition_info)
      return embedding_values

    # Expected lookup result, using combiner='mean'.
    expected_lookups = (
        # example 0, ids [2], embedding = [7, 11]
        (7., 11.),
        # example 1, ids [0, 1], embedding = mean([1, 2] + [3, 5]) = [2, 3.5]
        (2., 3.5),
        # example 2, ids [], embedding = [0, 0]
        (0., 0.),
        # example 3, ids [1], embedding = [3, 5]
        (3., 5.),
    )

    # Build columns.
    categorical_column = fc._categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc._embedding_column(
        categorical_column,
        dimension=embedding_dimension,
        initializer=_initializer)

    # Provide sparse input and get dense result.
    input_indices = array_ops.placeholder(dtype=dtypes.int64)
    input_values = array_ops.placeholder(dtype=dtypes.int64)
    input_shape = array_ops.placeholder(dtype=dtypes.int64)
    embedding_lookup = embedding_column._get_dense_tensor(
        _LazyBuilder({
            'aaa':
                sparse_tensor.SparseTensorValue(
                    indices=input_indices,
                    values=input_values,
                    dense_shape=input_shape)
        }))

    # Assert expected embedding variable and lookups.
    global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
    self.assertCountEqual(('embedding_weights:0',),
                          tuple([v.name for v in global_vars]))
    with _initialized_session():
      self.assertAllEqual(embedding_values, global_vars[0])
      self.assertAllEqual(expected_lookups, embedding_lookup.eval(
          feed_dict={
              input_indices: sparse_input.indices,
              input_values: sparse_input.values,
              input_shape: sparse_input.dense_shape,
          }))

  def test_get_dense_tensor_restore_from_ckpt(self):
    # [FUNCTION DOC]
    # Creates an embedding column that loads its initial values from a checkpoint file in the test
    # data directory; asserts the variable holds the checkpoint values and the lookup returns the
    # expected mean combined embeddings.
    with ops.Graph().as_default():
      # Inputs.
      vocabulary_size = 3
      sparse_input = sparse_tensor.SparseTensorValue(
          # example 0, ids [2]
          # example 1, ids [0, 1]
          # example 2, ids []
          # example 3, ids [1]
          indices=((0, 0), (1, 0), (1, 4), (3, 0)),
          values=(2, 0, 1, 1),
          dense_shape=(4, 5))

      # Embedding variable. The checkpoint file contains _embedding_values.
      embedding_dimension = 2
      embedding_values = (
          (1., 2.),  # id 0
          (3., 5.),  # id 1
          (7., 11.)  # id 2
      )
      ckpt_path = test.test_src_dir_path(
          'python/feature_column/testdata/embedding.ckpt')
      ckpt_tensor = 'my_embedding'

      # Expected lookup result, using combiner='mean'.
      expected_lookups = (
          # example 0, ids [2], embedding = [7, 11]
          (7., 11.),
          # example 1, ids [0, 1], embedding = mean([1, 2] + [3, 5]) = [2, 3.5]
          (2., 3.5),
          # example 2, ids [], embedding = [0, 0]
          (0., 0.),
          # example 3, ids [1], embedding = [3, 5]
          (3., 5.),
      )

      # Build columns.
      categorical_column = fc._categorical_column_with_identity(
          key='aaa', num_buckets=vocabulary_size)
      embedding_column = fc._embedding_column(
          categorical_column,
          dimension=embedding_dimension,
          ckpt_to_load_from=ckpt_path,
          tensor_name_in_ckpt=ckpt_tensor)

      # Provide sparse input and get dense result.
      embedding_lookup = embedding_column._get_dense_tensor(
          _LazyBuilder({'aaa': sparse_input}))

      # Assert expected embedding variable and lookups.
      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertCountEqual(('embedding_weights:0',),
                            tuple([v.name for v in global_vars]))
      with _initialized_session():
        self.assertAllEqual(embedding_values, global_vars[0])
        self.assertAllEqual(expected_lookups, self.evaluate(embedding_lookup))

  def test_linear_model(self):
    # [FUNCTION DOC]
    # Embedding column in the linear model function with an all-zero embedding initializer. Asserts
    # the exact set of created and trainable variable names (bias, linear weights, embedding weights),
    # zero predictions initially, and after assigning embedding and linear weights asserts the
    # predictions equal the dot products of the mean embeddings with the linear weights.
    # Inputs.
    batch_size = 4
    vocabulary_size = 3
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        # example 2, ids []
        # example 3, ids [1]
        indices=((0, 0), (1, 0), (1, 4), (3, 0)),
        values=(2, 0, 1, 1),
        dense_shape=(batch_size, 5))

    # Embedding variable.
    embedding_dimension = 2
    embedding_shape = (vocabulary_size, embedding_dimension)
    zeros_embedding_values = np.zeros(embedding_shape)
    def _initializer(shape, dtype, partition_info):
      # [FUNCTION DOC]
      # HELPER: fixture initializer that also validates how the variable is created.
      # Embedding initializer asserting expected shape, float32 dtype and no partition information;
      # returns a table of zeros.
      self.assertAllEqual(embedding_shape, shape)
      self.assertEqual(dtypes.float32, dtype)
      self.assertIsNone(partition_info)
      return zeros_embedding_values

    # Build columns.
    categorical_column = fc._categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc._embedding_column(
        categorical_column,
        dimension=embedding_dimension,
        initializer=_initializer)

    with ops.Graph().as_default():
      predictions = fc.linear_model({
          categorical_column.name: sparse_input
      }, (embedding_column,))
      expected_var_names = (
          'linear_model/bias_weights:0',
          'linear_model/aaa_embedding/weights:0',
          'linear_model/aaa_embedding/embedding_weights:0',
      )
      self.assertCountEqual(
          expected_var_names,
          [v.name for v in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)])
      trainable_vars = {
          v.name: v for v in ops.get_collection(
              ops.GraphKeys.TRAINABLE_VARIABLES)
      }
      self.assertCountEqual(expected_var_names, trainable_vars.keys())
      bias = trainable_vars['linear_model/bias_weights:0']
      embedding_weights = trainable_vars[
          'linear_model/aaa_embedding/embedding_weights:0']
      linear_weights = trainable_vars[
          'linear_model/aaa_embedding/weights:0']
      with _initialized_session():
        # Predictions with all zero weights.
        self.assertAllClose(np.zeros((1,)), self.evaluate(bias))
        self.assertAllClose(zeros_embedding_values,
                            self.evaluate(embedding_weights))
        self.assertAllClose(
            np.zeros((embedding_dimension, 1)), self.evaluate(linear_weights))
        self.assertAllClose(
            np.zeros((batch_size, 1)), self.evaluate(predictions))

        # Predictions with all non-zero weights.
        embedding_weights.assign((
            (1., 2.),  # id 0
            (3., 5.),  # id 1
            (7., 11.)  # id 2
        )).eval()
        linear_weights.assign(((4.,), (6.,))).eval()
        # example 0, ids [2], embedding[0] = [7, 11]
        # example 1, ids [0, 1], embedding[1] = mean([1, 2] + [3, 5]) = [2, 3.5]
        # example 2, ids [], embedding[2] = [0, 0]
        # example 3, ids [1], embedding[3] = [3, 5]
        # sum(embeddings * linear_weights)
        # = [4*7 + 6*11, 4*2 + 6*3.5, 4*0 + 6*0, 4*3 + 6*5] = [94, 29, 0, 42]
        self.assertAllClose(((94.,), (29.,), (0.,), (42.,)),
                            self.evaluate(predictions))

  def test_keras_linear_model(self):
    # [FUNCTION DOC]
    # Same as the linear model embedding test but through the Keras-style linear model helper.
    # Inputs.
    batch_size = 4
    vocabulary_size = 3
    sparse_input = sparse_tensor.SparseTensorValue(
        # example 0, ids [2]
        # example 1, ids [0, 1]
        # example 2, ids []
        # example 3, ids [1]
        indices=((0, 0), (1, 0), (1, 4), (3, 0)),
        values=(2, 0, 1, 1),
        dense_shape=(batch_size, 5))

    # Embedding variable.
    embedding_dimension = 2
    embedding_shape = (vocabulary_size, embedding_dimension)
    zeros_embedding_values = np.zeros(embedding_shape)

    def _initializer(shape, dtype, partition_info):
      # [FUNCTION DOC]
      # HELPER: fixture initializer that also validates how the variable is created.
      # Embedding initializer asserting expected shape, float32 dtype and no partition information;
      # returns a table of zeros.
      self.assertAllEqual(embedding_shape, shape)
      self.assertEqual(dtypes.float32, dtype)
      self.assertIsNone(partition_info)
      return zeros_embedding_values

    # Build columns.
    categorical_column = fc._categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc._embedding_column(
        categorical_column,
        dimension=embedding_dimension,
        initializer=_initializer)

    with ops.Graph().as_default():
      predictions = get_keras_linear_model_predictions({
          categorical_column.name: sparse_input
      }, (embedding_column,))
      expected_var_names = (
          'linear_model/bias_weights:0',
          'linear_model/aaa_embedding/weights:0',
          'linear_model/aaa_embedding/embedding_weights:0',
      )
      self.assertCountEqual(
          expected_var_names,
          [v.name for v in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)])
      trainable_vars = {
          v.name: v
          for v in ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      }
      self.assertCountEqual(expected_var_names, trainable_vars.keys())
      bias = trainable_vars['linear_model/bias_weights:0']
      embedding_weights = trainable_vars[
          'linear_model/aaa_embedding/embedding_weights:0']
      linear_weights = trainable_vars['linear_model/aaa_embedding/weights:0']
      with _initialized_session():
        # Predictions with all zero weights.
        self.assertAllClose(np.zeros((1,)), self.evaluate(bias))
        self.assertAllClose(zeros_embedding_values,
                            self.evaluate(embedding_weights))
        self.assertAllClose(
            np.zeros((embedding_dimension, 1)), self.evaluate(linear_weights))
        self.assertAllClose(
            np.zeros((batch_size, 1)), self.evaluate(predictions))

        # Predictions with all non-zero weights.
        embedding_weights.assign((
            (1., 2.),  # id 0
            (3., 5.),  # id 1
            (7., 11.)  # id 2
        )).eval()
        linear_weights.assign(((4.,), (6.,))).eval()
        # example 0, ids [2], embedding[0] = [7, 11]
        # example 1, ids [0, 1], embedding[1] = mean([1, 2] + [3, 5]) = [2, 3.5]
        # example 2, ids [], embedding[2] = [0, 0]
        # example 3, ids [1], embedding[3] = [3, 5]
        # sum(embeddings * linear_weights)
        # = [4*7 + 6*11, 4*2 + 6*3.5, 4*0 + 6*0, 4*3 + 6*5] = [94, 29, 0, 42]
        self.assertAllClose(((94.,), (29.,), (0.,), (42.,)),
                            self.evaluate(predictions))

  def test_input_layer(self):
    # [FUNCTION DOC]
    # Embedding column through the input layer function: asserts the single embedding variable name
    # under the input layer scope, that it is trainable, its values, and that the output equals the
    # expected mean combined embeddings.
    with ops.Graph().as_default():
      # Inputs.
      vocabulary_size = 3
      sparse_input = sparse_tensor.SparseTensorValue(
          # example 0, ids [2]
          # example 1, ids [0, 1]
          # example 2, ids []
          # example 3, ids [1]
          indices=((0, 0), (1, 0), (1, 4), (3, 0)),
          values=(2, 0, 1, 1),
          dense_shape=(4, 5))

      # Embedding variable.
      embedding_dimension = 2
      embedding_values = (
          (1., 2.),  # id 0
          (3., 5.),  # id 1
          (7., 11.)  # id 2
      )

      def _initializer(shape, dtype, partition_info):
        # [FUNCTION DOC]
        # HELPER: fixture initializer that also validates how the variable is created.
        # Embedding initializer asserting expected shape, float32 dtype and no partition information;
        # returns a fixed three by two table.
        self.assertAllEqual((vocabulary_size, embedding_dimension), shape)
        self.assertEqual(dtypes.float32, dtype)
        self.assertIsNone(partition_info)
        return embedding_values

      # Expected lookup result, using combiner='mean'.
      expected_lookups = (
          # example 0, ids [2], embedding = [7, 11]
          (7., 11.),
          # example 1, ids [0, 1], embedding = mean([1, 2] + [3, 5]) = [2, 3.5]
          (2., 3.5),
          # example 2, ids [], embedding = [0, 0]
          (0., 0.),
          # example 3, ids [1], embedding = [3, 5]
          (3., 5.),
      )

      # Build columns.
      categorical_column = fc._categorical_column_with_identity(
          key='aaa', num_buckets=vocabulary_size)
      embedding_column = fc._embedding_column(
          categorical_column,
          dimension=embedding_dimension,
          initializer=_initializer)

      # Provide sparse input and get dense result.
      input_layer = fc.input_layer({'aaa': sparse_input}, (embedding_column,))

      # Assert expected embedding variable and lookups.
      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertCountEqual(('input_layer/aaa_embedding/embedding_weights:0',),
                            tuple([v.name for v in global_vars]))
      trainable_vars = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      self.assertCountEqual(('input_layer/aaa_embedding/embedding_weights:0',),
                            tuple([v.name for v in trainable_vars]))
      with _initialized_session():
        self.assertAllEqual(embedding_values, trainable_vars[0])
        self.assertAllEqual(expected_lookups, self.evaluate(input_layer))

  def test_input_layer_not_trainable(self):
    # [FUNCTION DOC]
    # Same as the input layer test but with a non trainable embedding column; asserts the variable
    # exists in the global collection but the trainable collection is empty.
    with ops.Graph().as_default():
      # Inputs.
      vocabulary_size = 3
      sparse_input = sparse_tensor.SparseTensorValue(
          # example 0, ids [2]
          # example 1, ids [0, 1]
          # example 2, ids []
          # example 3, ids [1]
          indices=((0, 0), (1, 0), (1, 4), (3, 0)),
          values=(2, 0, 1, 1),
          dense_shape=(4, 5))

      # Embedding variable.
      embedding_dimension = 2
      embedding_values = (
          (1., 2.),  # id 0
          (3., 5.),  # id 1
          (7., 11.)  # id 2
      )

      def _initializer(shape, dtype, partition_info):
        # [FUNCTION DOC]
        # HELPER: fixture initializer that also validates how the variable is created.
        # Embedding initializer asserting expected shape, float32 dtype and no partition information;
        # returns a fixed three by two table.
        self.assertAllEqual((vocabulary_size, embedding_dimension), shape)
        self.assertEqual(dtypes.float32, dtype)
        self.assertIsNone(partition_info)
        return embedding_values

      # Expected lookup result, using combiner='mean'.
      expected_lookups = (
          # example 0, ids [2], embedding = [7, 11]
          (7., 11.),
          # example 1, ids [0, 1], embedding = mean([1, 2] + [3, 5]) = [2, 3.5]
          (2., 3.5),
          # example 2, ids [], embedding = [0, 0]
          (0., 0.),
          # example 3, ids [1], embedding = [3, 5]
          (3., 5.),
      )

      # Build columns.
      categorical_column = fc._categorical_column_with_identity(
          key='aaa', num_buckets=vocabulary_size)
      embedding_column = fc._embedding_column(
          categorical_column,
          dimension=embedding_dimension,
          initializer=_initializer,
          trainable=False)

      # Provide sparse input and get dense result.
      input_layer = fc.input_layer({'aaa': sparse_input}, (embedding_column,))

      # Assert expected embedding variable and lookups.
      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertCountEqual(('input_layer/aaa_embedding/embedding_weights:0',),
                            tuple([v.name for v in global_vars]))
      self.assertCountEqual([],
                            ops.get_collection(
                                ops.GraphKeys.TRAINABLE_VARIABLES))
      with _initialized_session():
        self.assertAllEqual(embedding_values, global_vars[0])
        self.assertAllEqual(expected_lookups, self.evaluate(input_layer))


class SharedEmbeddingColumnTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests shared embedding columns created by the newer shared embedding columns factory but
  # consumed by the legacy machinery: defaults and constructor arguments, deep copy, validation,
  # parsing, transformation, dense tensor lookup (parameterized), weight collections, placeholder
  # inputs, linear models and the input layer, including sharing of one variable between columns.

  def test_defaults(self):
    # [FUNCTION DOC]
    # Creates two shared embedding columns over identity columns and asserts defaults: wrapped
    # columns, dimension, mean combiner, no checkpoint settings, collection name built from the sorted
    # keys, per column names, shared scope name, variable shape and parse specs.
    with ops.Graph().as_default():
      categorical_column_a = fc._categorical_column_with_identity(
          key='aaa', num_buckets=3)
      categorical_column_b = fc._categorical_column_with_identity(
          key='bbb', num_buckets=3)
      embedding_dimension = 2
      embedding_column_b, embedding_column_a = fc_new.shared_embedding_columns(
          [categorical_column_b, categorical_column_a],
          dimension=embedding_dimension)
      self.assertIs(categorical_column_a, embedding_column_a.categorical_column)
      self.assertIs(categorical_column_b, embedding_column_b.categorical_column)
      self.assertEqual(embedding_dimension, embedding_column_a.dimension)
      self.assertEqual(embedding_dimension, embedding_column_b.dimension)
      self.assertEqual('mean', embedding_column_a.combiner)
      self.assertEqual('mean', embedding_column_b.combiner)
      self.assertIsNone(embedding_column_a.ckpt_to_load_from)
      self.assertIsNone(embedding_column_b.ckpt_to_load_from)
      self.assertEqual('aaa_bbb_shared_embedding',
                       embedding_column_a.shared_embedding_collection_name)
      self.assertEqual('aaa_bbb_shared_embedding',
                       embedding_column_b.shared_embedding_collection_name)
      self.assertIsNone(embedding_column_a.tensor_name_in_ckpt)
      self.assertIsNone(embedding_column_b.tensor_name_in_ckpt)
      self.assertIsNone(embedding_column_a.max_norm)
      self.assertIsNone(embedding_column_b.max_norm)
      self.assertTrue(embedding_column_a.trainable)
      self.assertTrue(embedding_column_b.trainable)
      self.assertEqual('aaa_shared_embedding', embedding_column_a.name)
      self.assertEqual('bbb_shared_embedding', embedding_column_b.name)
      self.assertEqual('aaa_bbb_shared_embedding',
                       embedding_column_a._var_scope_name)
      self.assertEqual('aaa_bbb_shared_embedding',
                       embedding_column_b._var_scope_name)
      self.assertEqual((embedding_dimension,),
                       embedding_column_a._variable_shape)
      self.assertEqual((embedding_dimension,),
                       embedding_column_b._variable_shape)
      self.assertEqual({'aaa': parsing_ops.VarLenFeature(dtypes.int64)},
                       embedding_column_a._parse_example_spec)
      self.assertEqual({'bbb': parsing_ops.VarLenFeature(dtypes.int64)},
                       embedding_column_b._parse_example_spec)

  def test_all_constructor_args(self):
    # [FUNCTION DOC]
    # Creates shared embedding columns with every option set (combiner, initializer, collection name,
    # checkpoint settings, max norm, not trainable) and asserts every attribute is shared by both
    # columns and the scope name equals the custom collection name.
    with ops.Graph().as_default():
      categorical_column_a = fc._categorical_column_with_identity(
          key='aaa', num_buckets=3)
      categorical_column_b = fc._categorical_column_with_identity(
          key='bbb', num_buckets=3)
      embedding_dimension = 2
      embedding_column_a, embedding_column_b = fc_new.shared_embedding_columns(
          [categorical_column_a, categorical_column_b],
          dimension=embedding_dimension,
          combiner='my_combiner',
          initializer=lambda: 'my_initializer',
          shared_embedding_collection_name='shared_embedding_collection_name',
          ckpt_to_load_from='my_ckpt',
          tensor_name_in_ckpt='my_ckpt_tensor',
          max_norm=42.,
          trainable=False)
      self.assertIs(categorical_column_a, embedding_column_a.categorical_column)
      self.assertIs(categorical_column_b, embedding_column_b.categorical_column)
      self.assertEqual(embedding_dimension, embedding_column_a.dimension)
      self.assertEqual(embedding_dimension, embedding_column_b.dimension)
      self.assertEqual('my_combiner', embedding_column_a.combiner)
      self.assertEqual('my_combiner', embedding_column_b.combiner)
      self.assertEqual('shared_embedding_collection_name',
                       embedding_column_a.shared_embedding_collection_name)
      self.assertEqual('shared_embedding_collection_name',
                       embedding_column_b.shared_embedding_collection_name)
      self.assertEqual('my_ckpt', embedding_column_a.ckpt_to_load_from)
      self.assertEqual('my_ckpt', embedding_column_b.ckpt_to_load_from)
      self.assertEqual('my_ckpt_tensor', embedding_column_a.tensor_name_in_ckpt)
      self.assertEqual('my_ckpt_tensor', embedding_column_b.tensor_name_in_ckpt)
      self.assertEqual(42., embedding_column_a.max_norm)
      self.assertEqual(42., embedding_column_b.max_norm)
      self.assertFalse(embedding_column_a.trainable)
      self.assertFalse(embedding_column_b.trainable)
      self.assertEqual('aaa_shared_embedding', embedding_column_a.name)
      self.assertEqual('bbb_shared_embedding', embedding_column_b.name)
      self.assertEqual('shared_embedding_collection_name',
                       embedding_column_a._var_scope_name)
      self.assertEqual('shared_embedding_collection_name',
                       embedding_column_b._var_scope_name)
      self.assertEqual(
          (embedding_dimension,), embedding_column_a._variable_shape)
      self.assertEqual((embedding_dimension,),
                       embedding_column_b._variable_shape)
      self.assertEqual({
          'aaa': parsing_ops.VarLenFeature(dtypes.int64)
      }, embedding_column_a._parse_example_spec)
      self.assertEqual({'bbb': parsing_ops.VarLenFeature(dtypes.int64)},
                       embedding_column_b._parse_example_spec)

  def test_deep_copy(self):
    # [FUNCTION DOC]
    # Deep-copies a shared embedding column created with all options and asserts original and copy
    # keep the wrapped column and all attributes.
    with ops.Graph().as_default():
      categorical_column_a = fc._categorical_column_with_identity(
          key='aaa', num_buckets=3)
      categorical_column_b = fc._categorical_column_with_identity(
          key='bbb', num_buckets=3)
      embedding_dimension = 2
      original_a, _ = fc_new.shared_embedding_columns(
          [categorical_column_a, categorical_column_b],
          dimension=embedding_dimension,
          combiner='my_combiner',
          initializer=lambda: 'my_initializer',
          shared_embedding_collection_name='shared_embedding_collection_name',
          ckpt_to_load_from='my_ckpt',
          tensor_name_in_ckpt='my_ckpt_tensor',
          max_norm=42.,
          trainable=False)
      for embedding_column_a in (original_a, copy.deepcopy(original_a)):
        self.assertEqual('aaa', embedding_column_a.categorical_column.name)
        self.assertEqual(3, embedding_column_a.categorical_column._num_buckets)
        self.assertEqual(
            {'aaa': parsing_ops.VarLenFeature(dtypes.int64)},
            embedding_column_a.categorical_column._parse_example_spec)

        self.assertEqual(embedding_dimension, embedding_column_a.dimension)
        self.assertEqual('my_combiner', embedding_column_a.combiner)
        self.assertEqual('shared_embedding_collection_name',
                         embedding_column_a.shared_embedding_collection_name)
        self.assertEqual('my_ckpt', embedding_column_a.ckpt_to_load_from)
        self.assertEqual('my_ckpt_tensor',
                         embedding_column_a.tensor_name_in_ckpt)
        self.assertEqual(42., embedding_column_a.max_norm)
        self.assertFalse(embedding_column_a.trainable)
        self.assertEqual('aaa_shared_embedding', embedding_column_a.name)
        self.assertEqual((embedding_dimension,),
                         embedding_column_a._variable_shape)
        self.assertEqual({'aaa': parsing_ops.VarLenFeature(dtypes.int64)},
                         embedding_column_a._parse_example_spec)

  def test_invalid_initializer(self):
    # [FUNCTION DOC]
    # Asserts that a non-callable initializer raises a value error saying it must be callable.
    with ops.Graph().as_default():
      categorical_column_a = fc._categorical_column_with_identity(
          key='aaa', num_buckets=3)
      categorical_column_b = fc._categorical_column_with_identity(
          key='bbb', num_buckets=3)
      with self.assertRaisesRegex(ValueError, 'initializer must be callable'):
        fc_new.shared_embedding_columns(
            [categorical_column_a, categorical_column_b],
            dimension=2,
            initializer='not_fn')

  def test_incompatible_column_type(self):
    # [FUNCTION DOC]
    # Asserts that mixing identity and hashed categorical columns raises a value error saying all
    # categorical columns must have the same type and naming both types.
    with ops.Graph().as_default():
      categorical_column_a = fc._categorical_column_with_identity(
          key='aaa', num_buckets=3)
      categorical_column_b = fc._categorical_column_with_identity(
          key='bbb', num_buckets=3)
      categorical_column_c = fc._categorical_column_with_hash_bucket(
          key='ccc', hash_bucket_size=3)
      with self.assertRaisesRegex(
          ValueError, 'all categorical_columns must have the same type.*'
          '_IdentityCategoricalColumn.*_HashedCategoricalColumn'):
        fc_new.shared_embedding_columns(
            [categorical_column_a, categorical_column_b, categorical_column_c],
            dimension=2)

  def test_weighted_categorical_column_ok(self):
    # [FUNCTION DOC]
    # Asserts that weighted categorical columns can be shared with plain ones in any combination
    # without raising an error.
    with ops.Graph().as_default():
      categorical_column_a = fc._categorical_column_with_identity(
          key='aaa', num_buckets=3)
      weighted_categorical_column_a = fc._weighted_categorical_column(
          categorical_column_a, weight_feature_key='aaa_weights')
      categorical_column_b = fc._categorical_column_with_identity(
          key='bbb', num_buckets=3)
      weighted_categorical_column_b = fc._weighted_categorical_column(
          categorical_column_b, weight_feature_key='bbb_weights')
      fc_new.shared_embedding_columns(
          [weighted_categorical_column_a, categorical_column_b], dimension=2)
      fc_new.shared_embedding_columns(
          [categorical_column_a, weighted_categorical_column_b], dimension=2)
      fc_new.shared_embedding_columns(
          [weighted_categorical_column_a, weighted_categorical_column_b],
          dimension=2)

  def test_parse_example(self):
    # [FUNCTION DOC]
    # Parses a serialized example containing two features using the specs of two shared embedding
    # columns; asserts both parsed sparse string tensors.
    with ops.Graph().as_default():
      a = fc._categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
      b = fc._categorical_column_with_vocabulary_list(
          key='bbb', vocabulary_list=('omar', 'stringer', 'marlo'))
      a_embedded, b_embedded = fc_new.shared_embedding_columns([a, b],
                                                               dimension=2)
      data = example_pb2.Example(
          features=feature_pb2.Features(
              feature={
                  'aaa':
                      feature_pb2.Feature(
                          bytes_list=feature_pb2.BytesList(
                              value=[b'omar', b'stringer'])),
                  'bbb':
                      feature_pb2.Feature(
                          bytes_list=feature_pb2.BytesList(
                              value=[b'stringer', b'marlo'])),
              }))
      features = parsing_ops.parse_example(
          serialized=[data.SerializeToString()],
          features=fc.make_parse_example_spec([a_embedded, b_embedded]))
      self.assertIn('aaa', features)
      self.assertIn('bbb', features)
      with self.cached_session():
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=[[0, 0], [0, 1]],
                values=np.array([b'omar', b'stringer'], dtype=np.object_),
                dense_shape=[1, 2]), features['aaa'].eval())
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=[[0, 0], [0, 1]],
                values=np.array([b'stringer', b'marlo'], dtype=np.object_),
                dense_shape=[1, 2]), features['bbb'].eval())

  def test_transform_feature(self):
    # [FUNCTION DOC]
    # Transforms two identity columns and their shared embedding columns; asserts each embedding
    # column output equals its wrapped column ids.
    with ops.Graph().as_default():
      a = fc._categorical_column_with_identity(key='aaa', num_buckets=3)
      b = fc._categorical_column_with_identity(key='bbb', num_buckets=3)
      a_embedded, b_embedded = fc_new.shared_embedding_columns([a, b],
                                                               dimension=2)
      features = {
          'aaa':
              sparse_tensor.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(0, 1, 0),
                  dense_shape=(2, 2)),
          'bbb':
              sparse_tensor.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(1, 2, 1),
                  dense_shape=(2, 2)),
      }
      outputs = _transform_features(features, [a, a_embedded, b, b_embedded])
      output_a = outputs[a]
      output_a_embedded = outputs[a_embedded]
      output_b = outputs[b]
      output_b_embedded = outputs[b_embedded]
      with _initialized_session():
        _assert_sparse_tensor_value(self, self.evaluate(output_a),
                                    self.evaluate(output_a_embedded))
        _assert_sparse_tensor_value(self, self.evaluate(output_b),
                                    self.evaluate(output_b_embedded))

  @parameterized.named_parameters(
      {
          'testcase_name': 'use_safe_embedding_lookup',
          'use_safe_embedding_lookup': True,
          'partition_variables': False,
      }, {
          'testcase_name': 'dont_use_safe_embedding_lookup',
          'use_safe_embedding_lookup': False,
          'partition_variables': False,
      }, {
          'testcase_name': 'use_safe_embedding_lookup_partitioned',
          'use_safe_embedding_lookup': True,
          'partition_variables': True,
      }, {
          'testcase_name': 'dont_use_safe_embedding_lookup_partitioned',
          'use_safe_embedding_lookup': False,
          'partition_variables': True,
      })

  def test_get_dense_tensor(self, use_safe_embedding_lookup,
                            partition_variables):
    # [FUNCTION DOC]
    # Parameterized over safe embedding lookup and variable partitioning. Looks up dense matrix inputs
    # (minus one meaning missing) through two shared columns; asserts a single shared variable (or two
    # shards), its contents, the expected mean combined lookups (the second column expected result has
    # the empty row only when safe lookup is used), and the presence of the fill empty rows operation
    # only with safe lookup.
    with ops.Graph().as_default():
      # Inputs.
      vocabulary_size = 4
      # -1 values are ignored.
      input_a = np.array([
          [2, -1, -1],  # example 0, ids [2]
          [0, 1, -1]
      ])  # example 1, ids [0, 1]
      input_b = np.array([
          [0, -1, -1],  # example 0, ids [0]
          [-1, -1, -1]
      ])  # example 1, ids []
      input_features = {'aaa': input_a, 'bbb': input_b}

      # Embedding variable.
      embedding_dimension = 2
      embedding_values = (
          (1., 2.),  # id 0
          (3., 5.),  # id 1
          (7., 11.),  # id 2
          (9., 13.)  # id 3
      )

      def _initializer(shape, dtype, partition_info=None):
        # [FUNCTION DOC]
        # HELPER: fixture initializer that also validates how the variable is created.
        # Embedding initializer asserting float32 dtype; in the partitioned case asserts the partition
        # full shape and shard shape and returns the matching slice of the fixed table, otherwise asserts
        # the full shape and no partition information and returns the whole fixed four by two table.
        self.assertEqual(dtypes.float32, dtype)
        if partition_variables:
          assert partition_info is not None
          self.assertEqual([vocabulary_size, embedding_dimension],
                           partition_info.full_shape)
          self.assertAllEqual((2, embedding_dimension), shape)
          return array_ops.slice(
              embedding_values, partition_info.var_offset, shape
          )
        else:
          self.assertAllEqual((vocabulary_size, embedding_dimension), shape)
          self.assertIsNone(partition_info)
          return embedding_values

      # Expected lookup result, using combiner='mean'.
      expected_lookups_a = (
          # example 0:
          (7., 11.),  # ids [2], embedding = [7, 11]
          # example 1:
          (2., 3.5),  # ids [0, 1], embedding = mean([1, 2] + [3, 5]) = [2, 3.5]
      )
      if use_safe_embedding_lookup:
        expected_lookups_b = (
            # example 0:
            (1., 2.),  # ids [0], embedding = [1, 2]
            # example 1:
            (0., 0.),  # ids [], embedding = [0, 0]
        )
      else:
        expected_lookups_b = (
            # example 0:
            (1., 2.),  # ids [0], embedding = [1, 2]
        )

      # Build columns.
      categorical_column_a = fc._categorical_column_with_identity(
          key='aaa', num_buckets=vocabulary_size)
      categorical_column_b = fc._categorical_column_with_identity(
          key='bbb', num_buckets=vocabulary_size)

      partitioner = None
      if partition_variables:
        partitioner = partitioned_variables.fixed_size_partitioner(2, axis=0)

      with variable_scope.variable_scope('vars', partitioner=partitioner):
        embedding_column_a, embedding_column_b = (
            fc_new.shared_embedding_columns(
                [categorical_column_a, categorical_column_b],
                dimension=embedding_dimension,
                initializer=_initializer,
                use_safe_embedding_lookup=use_safe_embedding_lookup))
        # Provide sparse input and get dense result.
        embedding_lookup_a = embedding_column_a._get_dense_tensor(
            _LazyBuilder(input_features))
        embedding_lookup_b = embedding_column_b._get_dense_tensor(
            _LazyBuilder(input_features))
      # Assert expected embedding variable and lookups.
      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      if partition_variables:
        self.assertCountEqual(('vars/embedding_weights/part_0:0',
                               'vars/embedding_weights/part_1:0'),
                              tuple([v.name for v in global_vars]))
        embedding_var = array_ops.concat(global_vars, axis=0)
      else:
        self.assertCountEqual(('vars/embedding_weights:0',),
                              tuple([v.name for v in global_vars]))
        embedding_var = global_vars[0]

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllEqual(embedding_values, self.evaluate(embedding_var))
      self.assertAllEqual(expected_lookups_a, self.evaluate(embedding_lookup_a))
      self.assertAllEqual(expected_lookups_b, self.evaluate(embedding_lookup_b))

      if use_safe_embedding_lookup:
        self.assertIn(
            'SparseFillEmptyRows',
            [x.type for x in ops.get_default_graph().get_operations()])
      else:
        self.assertNotIn(
            'SparseFillEmptyRows',
            [x.type for x in ops.get_default_graph().get_operations()])

  def test_get_dense_tensor_weight_collections(self):
    # [FUNCTION DOC]
    # Calls the input layer function with a named weight collection and shared embedding columns;
    # asserts the single shared variable is in both the global collection and the named collection.
    with ops.Graph().as_default():
      # Inputs.
      vocabulary_size = 3
      # -1 values are ignored.
      input_a = np.array([
          [2, -1, -1],  # example 0, ids [2]
          [0, 1, -1]
      ])  # example 1, ids [0, 1]
      input_b = np.array([
          [0, -1, -1],  # example 0, ids [0]
          [-1, -1, -1]
      ])  # example 1, ids []
      input_features = {'aaa': input_a, 'bbb': input_b}

      # Embedding variable.
      embedding_dimension = 2
      embedding_values = (
          (1., 2.),  # id 0
          (3., 5.),  # id 1
          (7., 11.)  # id 2
      )

      def _initializer(shape, dtype, partition_info):
        # [FUNCTION DOC]
        # HELPER: fixture initializer that also validates how the variable is created.
        # Embedding initializer asserting expected shape, float32 dtype and no partition information;
        # returns a fixed three by two table.
        self.assertAllEqual((vocabulary_size, embedding_dimension), shape)
        self.assertEqual(dtypes.float32, dtype)
        self.assertIsNone(partition_info)
        return embedding_values

      # Build columns.
      categorical_column_a = fc._categorical_column_with_identity(
          key='aaa', num_buckets=vocabulary_size)
      categorical_column_b = fc._categorical_column_with_identity(
          key='bbb', num_buckets=vocabulary_size)
      embedding_column_a, embedding_column_b = fc_new.shared_embedding_columns(
          [categorical_column_a, categorical_column_b],
          dimension=embedding_dimension,
          initializer=_initializer)

      fc.input_layer(
          input_features, [embedding_column_a, embedding_column_b],
          weight_collections=('my_vars',))

      # Assert expected embedding variable and lookups.
      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertCountEqual(
          ('input_layer/aaa_bbb_shared_embedding/embedding_weights:0',),
          tuple(v.name for v in global_vars))
      my_vars = ops.get_collection('my_vars')
      self.assertCountEqual(
          ('input_layer/aaa_bbb_shared_embedding/embedding_weights:0',),
          tuple(v.name for v in my_vars))

  @test_util.run_deprecated_v1
  # Placeholders are TF1. Replacing with tf.function not feasible because of V1
  # variable creation.
  def test_get_dense_tensor_placeholder_inputs(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test only runs the two lookups with fed placeholder data and asserts nothing
    # about the results, so it only checks that graph construction and execution do not fail.
    # Scenario (v1 graph mode): dense integer placeholders with a declared rank for two shared
    # columns. Builds the dense lookups, initializes variables and runs both lookups with fed arrays.
    # Inputs.
    vocabulary_size = 3
    # -1 values are ignored.
    input_a = np.array(
        [[2, -1, -1],  # example 0, ids [2]
         [0, 1, -1]])  # example 1, ids [0, 1]
    input_b = np.array(
        [[0, -1, -1],  # example 0, ids [0]
         [-1, -1, -1]])  # example 1, ids []
    # Specify shape, because dense input must have rank specified.
    input_a_placeholder = array_ops.placeholder(
        dtype=dtypes.int64, shape=[None, 3])
    input_b_placeholder = array_ops.placeholder(
        dtype=dtypes.int64, shape=[None, 3])
    input_features = {
        'aaa': input_a_placeholder,
        'bbb': input_b_placeholder,
    }
    feed_dict = {
        input_a_placeholder: input_a,
        input_b_placeholder: input_b,
    }

    # Embedding variable.
    embedding_dimension = 2
    embedding_values = (
        (1., 2.),  # id 0
        (3., 5.),  # id 1
        (7., 11.)  # id 2
    )
    def _initializer(shape, dtype, partition_info):
      # [FUNCTION DOC]
      # HELPER: fixture initializer that also validates how the variable is created.
      # Embedding initializer asserting expected shape, float32 dtype and no partition information;
      # returns a fixed three by two table.
      self.assertAllEqual((vocabulary_size, embedding_dimension), shape)
      self.assertEqual(dtypes.float32, dtype)
      self.assertIsNone(partition_info)
      return embedding_values

    # Build columns.
    categorical_column_a = fc._categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    categorical_column_b = fc._categorical_column_with_identity(
        key='bbb', num_buckets=vocabulary_size)
    embedding_column_a, embedding_column_b = fc_new.shared_embedding_columns(
        [categorical_column_a, categorical_column_b],
        dimension=embedding_dimension,
        initializer=_initializer)

    # Provide sparse input and get dense result.
    embedding_lookup_a = embedding_column_a._get_dense_tensor(
        _LazyBuilder(input_features))
    embedding_lookup_b = embedding_column_b._get_dense_tensor(
        _LazyBuilder(input_features))

    with _initialized_session() as sess:
      sess.run([embedding_lookup_a, embedding_lookup_b], feed_dict=feed_dict)

  def test_linear_model(self):
    # [FUNCTION DOC]
    # Two shared embedding columns in the linear model function with a zero initialized embedding.
    # Asserts the set of created variable names (one shared embedding, separate linear weights per
    # column, the second in a uniquified scope), zero predictions initially, and after assigning
    # values asserts the per example sum of both columns dot products.
    with ops.Graph().as_default():
      # Inputs.
      batch_size = 2
      vocabulary_size = 3
      # -1 values are ignored.
      input_a = np.array([
          [2, -1, -1],  # example 0, ids [2]
          [0, 1, -1]
      ])  # example 1, ids [0, 1]
      input_b = np.array([
          [0, -1, -1],  # example 0, ids [0]
          [-1, -1, -1]
      ])  # example 1, ids []

      # Embedding variable.
      embedding_dimension = 2
      embedding_shape = (vocabulary_size, embedding_dimension)
      zeros_embedding_values = np.zeros(embedding_shape)

      def _initializer(shape, dtype, partition_info):
        # [FUNCTION DOC]
        # HELPER: fixture initializer that also validates how the variable is created.
        # Embedding initializer asserting expected shape, float32 dtype and no partition information;
        # returns a table of zeros.
        self.assertAllEqual(embedding_shape, shape)
        self.assertEqual(dtypes.float32, dtype)
        self.assertIsNone(partition_info)
        return zeros_embedding_values

      # Build columns.
      categorical_column_a = fc._categorical_column_with_identity(
          key='aaa', num_buckets=vocabulary_size)
      categorical_column_b = fc._categorical_column_with_identity(
          key='bbb', num_buckets=vocabulary_size)
      embedding_column_a, embedding_column_b = fc_new.shared_embedding_columns(
          [categorical_column_a, categorical_column_b],
          dimension=embedding_dimension,
          initializer=_initializer)

      predictions = fc.linear_model({
          categorical_column_a.name: input_a,
          categorical_column_b.name: input_b,
      }, (embedding_column_a, embedding_column_b))
      # Linear weights do not follow the column name. But this is a rare use
      # case, and fixing it would add too much complexity to the code.
      expected_var_names = (
          'linear_model/bias_weights:0',
          'linear_model/aaa_bbb_shared_embedding/weights:0',
          'linear_model/aaa_bbb_shared_embedding/embedding_weights:0',
          'linear_model/aaa_bbb_shared_embedding_1/weights:0',
      )
      self.assertCountEqual(
          expected_var_names,
          [v.name for v in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)])
      trainable_vars = {
          v.name: v for v in ops.get_collection(
              ops.GraphKeys.TRAINABLE_VARIABLES)
      }
      self.assertCountEqual(expected_var_names, trainable_vars.keys())
      bias = trainable_vars['linear_model/bias_weights:0']
      embedding_weights = trainable_vars[
          'linear_model/aaa_bbb_shared_embedding/embedding_weights:0']
      linear_weights_a = trainable_vars[
          'linear_model/aaa_bbb_shared_embedding/weights:0']
      linear_weights_b = trainable_vars[
          'linear_model/aaa_bbb_shared_embedding_1/weights:0']
      with _initialized_session():
        # Predictions with all zero weights.
        self.assertAllClose(np.zeros((1,)), self.evaluate(bias))
        self.assertAllClose(zeros_embedding_values,
                            self.evaluate(embedding_weights))
        self.assertAllClose(
            np.zeros((embedding_dimension, 1)), self.evaluate(linear_weights_a))
        self.assertAllClose(
            np.zeros((embedding_dimension, 1)), self.evaluate(linear_weights_b))
        self.assertAllClose(
            np.zeros((batch_size, 1)), self.evaluate(predictions))

        # Predictions with all non-zero weights.
        embedding_weights.assign((
            (1., 2.),  # id 0
            (3., 5.),  # id 1
            (7., 11.)  # id 2
        )).eval()
        linear_weights_a.assign(((4.,), (6.,))).eval()
        # example 0, ids [2], embedding[0] = [7, 11]
        # example 1, ids [0, 1], embedding[1] = mean([1, 2] + [3, 5]) = [2, 3.5]
        # sum(embeddings * linear_weights)
        # = [4*7 + 6*11, 4*2 + 6*3.5] = [94, 29]
        linear_weights_b.assign(((3.,), (5.,))).eval()
        # example 0, ids [0], embedding[0] = [1, 2]
        # example 1, ids [], embedding[1] = 0, 0]
        # sum(embeddings * linear_weights)
        # = [3*1 + 5*2, 3*0 +5*0] = [13, 0]
        self.assertAllClose([[94. + 13.], [29.]], self.evaluate(predictions))

  def test_keras_linear_model(self):
    # [FUNCTION DOC]
    # Same as the linear model shared embedding test but through the Keras-style linear model helper.
    with ops.Graph().as_default():
      # Inputs.
      batch_size = 2
      vocabulary_size = 3
      # -1 values are ignored.
      input_a = np.array([
          [2, -1, -1],  # example 0, ids [2]
          [0, 1, -1]
      ])  # example 1, ids [0, 1]
      input_b = np.array([
          [0, -1, -1],  # example 0, ids [0]
          [-1, -1, -1]
      ])  # example 1, ids []

      # Embedding variable.
      embedding_dimension = 2
      embedding_shape = (vocabulary_size, embedding_dimension)
      zeros_embedding_values = np.zeros(embedding_shape)

      def _initializer(shape, dtype, partition_info):
        # [FUNCTION DOC]
        # HELPER: fixture initializer that also validates how the variable is created.
        # Embedding initializer asserting expected shape, float32 dtype and no partition information;
        # returns a table of zeros.
        self.assertAllEqual(embedding_shape, shape)
        self.assertEqual(dtypes.float32, dtype)
        self.assertIsNone(partition_info)
        return zeros_embedding_values

      # Build columns.
      categorical_column_a = fc._categorical_column_with_identity(
          key='aaa', num_buckets=vocabulary_size)
      categorical_column_b = fc._categorical_column_with_identity(
          key='bbb', num_buckets=vocabulary_size)
      embedding_column_a, embedding_column_b = fc_new.shared_embedding_columns(
          [categorical_column_a, categorical_column_b],
          dimension=embedding_dimension,
          initializer=_initializer)

      predictions = get_keras_linear_model_predictions({
          categorical_column_a.name: input_a,
          categorical_column_b.name: input_b,
      }, (embedding_column_a, embedding_column_b))
      # Linear weights do not follow the column name. But this is a rare use
      # case, and fixing it would add too much complexity to the code.
      expected_var_names = (
          'linear_model/bias_weights:0',
          'linear_model/aaa_bbb_shared_embedding/weights:0',
          'linear_model/aaa_bbb_shared_embedding/embedding_weights:0',
          'linear_model/aaa_bbb_shared_embedding_1/weights:0',
      )
      self.assertCountEqual(
          expected_var_names,
          [v.name for v in ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)])
      trainable_vars = {
          v.name: v
          for v in ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      }
      self.assertCountEqual(expected_var_names, trainable_vars.keys())
      bias = trainable_vars['linear_model/bias_weights:0']
      embedding_weights = trainable_vars[
          'linear_model/aaa_bbb_shared_embedding/embedding_weights:0']
      linear_weights_a = trainable_vars[
          'linear_model/aaa_bbb_shared_embedding/weights:0']
      linear_weights_b = trainable_vars[
          'linear_model/aaa_bbb_shared_embedding_1/weights:0']
      with _initialized_session():
        # Predictions with all zero weights.
        self.assertAllClose(np.zeros((1,)), self.evaluate(bias))
        self.assertAllClose(zeros_embedding_values,
                            self.evaluate(embedding_weights))
        self.assertAllClose(
            np.zeros((embedding_dimension, 1)), self.evaluate(linear_weights_a))
        self.assertAllClose(
            np.zeros((embedding_dimension, 1)), self.evaluate(linear_weights_b))
        self.assertAllClose(
            np.zeros((batch_size, 1)), self.evaluate(predictions))

        # Predictions with all non-zero weights.
        embedding_weights.assign((
            (1., 2.),  # id 0
            (3., 5.),  # id 1
            (7., 11.)  # id 2
        )).eval()
        linear_weights_a.assign(((4.,), (6.,))).eval()
        # example 0, ids [2], embedding[0] = [7, 11]
        # example 1, ids [0, 1], embedding[1] = mean([1, 2] + [3, 5]) = [2, 3.5]
        # sum(embeddings * linear_weights)
        # = [4*7 + 6*11, 4*2 + 6*3.5] = [94, 29]
        linear_weights_b.assign(((3.,), (5.,))).eval()
        # example 0, ids [0], embedding[0] = [1, 2]
        # example 1, ids [], embedding[1] = 0, 0]
        # sum(embeddings * linear_weights)
        # = [3*1 + 5*2, 3*0 +5*0] = [13, 0]
        self.assertAllClose([[94. + 13.], [29.]], self.evaluate(predictions))

  def _test_input_layer(self, trainable=True):
    # [FUNCTION DOC]
    # HELPER: shared implementation for the trainable and non trainable input layer tests below.
    # Builds two shared embedding columns with the given trainable flag (default true) and runs the
    # input layer function passing the columns in reverse order. Asserts exactly one shared embedding
    # variable exists, that it is in the trainable collection only when the flag is true, and that the
    # output rows concatenate the per column mean embeddings in column name order.
    with ops.Graph().as_default():
      # Inputs.
      vocabulary_size = 3
      sparse_input_a = sparse_tensor.SparseTensorValue(
          # example 0, ids [2]
          # example 1, ids [0, 1]
          indices=((0, 0), (1, 0), (1, 4)),
          values=(2, 0, 1),
          dense_shape=(2, 5))
      sparse_input_b = sparse_tensor.SparseTensorValue(
          # example 0, ids [0]
          # example 1, ids []
          indices=((0, 0),),
          values=(0,),
          dense_shape=(2, 5))

      # Embedding variable.
      embedding_dimension = 2
      embedding_values = (
          (1., 2.),  # id 0
          (3., 5.),  # id 1
          (7., 11.)  # id 2
      )

      def _initializer(shape, dtype, partition_info):
        # [FUNCTION DOC]
        # HELPER: fixture initializer that also validates how the variable is created.
        # Embedding initializer asserting expected shape, float32 dtype and no partition information;
        # returns a fixed three by two table.
        self.assertAllEqual((vocabulary_size, embedding_dimension), shape)
        self.assertEqual(dtypes.float32, dtype)
        self.assertIsNone(partition_info)
        return embedding_values

      # Expected lookup result, using combiner='mean'.
      expected_lookups = (
          # example 0:
          # A ids [2], embedding = [7, 11]
          # B ids [0], embedding = [1, 2]
          (7., 11., 1., 2.),
          # example 1:
          # A ids [0, 1], embedding = mean([1, 2] + [3, 5]) = [2, 3.5]
          # B ids [], embedding = [0, 0]
          (2., 3.5, 0., 0.),
      )

      # Build columns.
      categorical_column_a = fc._categorical_column_with_identity(
          key='aaa', num_buckets=vocabulary_size)
      categorical_column_b = fc._categorical_column_with_identity(
          key='bbb', num_buckets=vocabulary_size)
      embedding_column_a, embedding_column_b = fc_new.shared_embedding_columns(
          [categorical_column_a, categorical_column_b],
          dimension=embedding_dimension,
          initializer=_initializer,
          trainable=trainable)

      # Provide sparse input and get dense result.
      input_layer = fc.input_layer(
          features={
              'aaa': sparse_input_a,
              'bbb': sparse_input_b
          },
          feature_columns=(embedding_column_b, embedding_column_a))

      # Assert expected embedding variable and lookups.
      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertCountEqual(
          ['input_layer/aaa_bbb_shared_embedding/embedding_weights:0'],
          tuple([v.name for v in global_vars]))
      trainable_vars = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      if trainable:
        self.assertCountEqual(
            ['input_layer/aaa_bbb_shared_embedding/embedding_weights:0'],
            tuple([v.name for v in trainable_vars]))
      else:
        self.assertCountEqual([], tuple([v.name for v in trainable_vars]))
      shared_embedding_vars = global_vars
      with _initialized_session():
        self.assertAllEqual(embedding_values, shared_embedding_vars[0])
        self.assertAllEqual(expected_lookups, self.evaluate(input_layer))

  def test_input_layer(self):
    # [FUNCTION DOC]
    # Runs the shared input layer scenario with the default trainable setting.
    self._test_input_layer()

  def test_input_layer_no_trainable(self):
    # [FUNCTION DOC]
    # Runs the shared input layer scenario with trainable set to false.
    self._test_input_layer(trainable=False)


class WeightedCategoricalColumnTest(test.TestCase):
  # [CLASS DOC]
  # Tests the legacy weighted categorical column: defaults, deep copy, validation of dtype and
  # inputs, name collision, missing weights, parsing, transformation of sparse and dense ids and
  # weights, and use in the Keras-style linear model and the linear model function including shape
  # mismatch cases.

  def test_defaults(self):
    # [FUNCTION DOC]
    # Asserts name (ids weighted by values), scope name, bucket count and a parse spec with an int64
    # variable length id feature and a float32 variable length weight feature.
    column = fc._weighted_categorical_column(
        categorical_column=fc._categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    self.assertEqual('ids_weighted_by_values', column.name)
    self.assertEqual('ids_weighted_by_values', column._var_scope_name)
    self.assertEqual(3, column._num_buckets)
    self.assertEqual({
        'ids': parsing_ops.VarLenFeature(dtypes.int64),
        'values': parsing_ops.VarLenFeature(dtypes.float32)
    }, column._parse_example_spec)

  def test_deep_copy(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it tests deep copy of a hashed categorical column, but the test
    # actually copies a weighted categorical column built over an identity column.
    # Asserts that original and deep-copied weighted columns keep name, bucket count and parse spec.
    """Tests deepcopy of categorical_column_with_hash_bucket."""
    original = fc._weighted_categorical_column(
        categorical_column=fc._categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    for column in (original, copy.deepcopy(original)):
      self.assertEqual('ids_weighted_by_values', column.name)
      self.assertEqual(3, column._num_buckets)
      self.assertEqual({
          'ids': parsing_ops.VarLenFeature(dtypes.int64),
          'values': parsing_ops.VarLenFeature(dtypes.float32)
      }, column._parse_example_spec)

  def test_invalid_dtype_none(self):
    # [FUNCTION DOC]
    # Asserts that a missing weight dtype raises a value error saying it is not convertible to float.
    with self.assertRaisesRegex(ValueError, 'is not convertible to float'):
      fc._weighted_categorical_column(
          categorical_column=fc._categorical_column_with_identity(
              key='ids', num_buckets=3),
          weight_feature_key='values',
          dtype=None)

  def test_invalid_dtype_string(self):
    # [FUNCTION DOC]
    # Asserts that a string weight dtype raises a value error saying it is not convertible to float.
    with self.assertRaisesRegex(ValueError, 'is not convertible to float'):
      fc._weighted_categorical_column(
          categorical_column=fc._categorical_column_with_identity(
              key='ids', num_buckets=3),
          weight_feature_key='values',
          dtype=dtypes.string)

  def test_invalid_input_dtype(self):
    # [FUNCTION DOC]
    # Feeds string values as the weights; asserts the transform raises a value error about a bad
    # dtype.
    column = fc._weighted_categorical_column(
        categorical_column=fc._categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    strings = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('omar', 'stringer', 'marlo'),
        dense_shape=(2, 2))
    with self.assertRaisesRegex(ValueError, 'Bad dtype'):
      _transform_features({'ids': strings, 'values': strings}, (column,))

  def test_column_name_collision(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the parse spec is a property but the test calls it like a function; the expected
    # value error is raised while evaluating the property, so the extra call is never reached and is
    # misleading.
    # Asserts that using the same key for the ids and the weights raises a value error saying the
    # parse config already exists.
    with self.assertRaisesRegex(ValueError, r'Parse config.*already exists'):
      fc._weighted_categorical_column(
          categorical_column=fc._categorical_column_with_identity(
              key='aaa', num_buckets=3),
          weight_feature_key='aaa')._parse_example_spec()

  def test_missing_weights(self):
    # [FUNCTION DOC]
    # Transforms features lacking the weight feature; asserts a value error saying the weight key is
    # not in the features dictionary.
    column = fc._weighted_categorical_column(
        categorical_column=fc._categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('omar', 'stringer', 'marlo'),
        dense_shape=(2, 2))
    with self.assertRaisesRegex(ValueError,
                                'values is not in features dictionary'):
      _transform_features({'ids': inputs}, (column,))

  def test_parse_example(self):
    # [FUNCTION DOC]
    # Parses a serialized example with ids and weights using the column spec; asserts both parsed
    # sparse tensors (strings and float32 weights).
    a = fc._categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    a_weighted = fc._weighted_categorical_column(
        a, weight_feature_key='weights')
    data = example_pb2.Example(features=feature_pb2.Features(
        feature={
            'aaa':
                feature_pb2.Feature(bytes_list=feature_pb2.BytesList(
                    value=[b'omar', b'stringer'])),
            'weights':
                feature_pb2.Feature(float_list=feature_pb2.FloatList(
                    value=[1., 10.]))
        }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec([a_weighted]))
    self.assertIn('aaa', features)
    self.assertIn('weights', features)
    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [0, 1]],
            values=np.array([b'omar', b'stringer'], dtype=np.object_),
            dense_shape=[1, 2]), self.evaluate(features['aaa']))
    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [0, 1]],
            values=np.array([1., 10.], dtype=np.float32),
            dense_shape=[1, 2]), self.evaluate(features['weights']))

  def test_transform_features(self):
    # [FUNCTION DOC]
    # Transforms sparse ids and sparse weights; asserts the output id tensor and weight tensor keep
    # indices and shape with int64 ids and float32 weights.
    with ops.Graph().as_default():
      column = fc._weighted_categorical_column(
          categorical_column=fc._categorical_column_with_identity(
              key='ids', num_buckets=3),
          weight_feature_key='values')
      inputs = sparse_tensor.SparseTensorValue(
          indices=((0, 0), (1, 0), (1, 1)),
          values=(0, 1, 0),
          dense_shape=(2, 2))
      weights = sparse_tensor.SparseTensorValue(
          indices=((0, 0), (1, 0), (1, 1)),
          values=(0.5, 1.0, 0.1),
          dense_shape=(2, 2))
      id_tensor, weight_tensor = _transform_features({
          'ids': inputs,
          'values': weights,
      }, (column,))[column]
      with _initialized_session():
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=inputs.indices,
                values=np.array(inputs.values, dtype=np.int64),
                dense_shape=inputs.dense_shape), self.evaluate(id_tensor))
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=weights.indices,
                values=np.array(weights.values, dtype=np.float32),
                dense_shape=weights.dense_shape), self.evaluate(weight_tensor))

  def test_transform_features_dense_input(self):
    # [FUNCTION DOC]
    # Feeds dense ids with minus one as missing and sparse weights; asserts the ids become a sparse
    # tensor without the missing entry and the weights are kept.
    with ops.Graph().as_default():
      column = fc._weighted_categorical_column(
          categorical_column=fc._categorical_column_with_identity(
              key='ids', num_buckets=3),
          weight_feature_key='values')
      weights = sparse_tensor.SparseTensorValue(
          indices=((0, 0), (1, 0), (1, 1)),
          values=(0.5, 1.0, 0.1),
          dense_shape=(2, 2))
      id_tensor, weight_tensor = _transform_features({
          'ids': ((0, -1), (1, 0)),
          'values': weights,
      }, (column,))[column]
      with _initialized_session():
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=((0, 0), (1, 0), (1, 1)),
                values=np.array((0, 1, 0), dtype=np.int64),
                dense_shape=(2, 2)), self.evaluate(id_tensor))
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=weights.indices,
                values=np.array(weights.values, dtype=np.float32),
                dense_shape=weights.dense_shape), self.evaluate(weight_tensor))

  def test_transform_features_dense_weights(self):
    # [FUNCTION DOC]
    # Feeds sparse ids and dense weights; asserts the dense weights are converted to a sparse tensor
    # aligned with the ids, dropping zero weights.
    with ops.Graph().as_default():
      column = fc._weighted_categorical_column(
          categorical_column=fc._categorical_column_with_identity(
              key='ids', num_buckets=3),
          weight_feature_key='values')
      inputs = sparse_tensor.SparseTensorValue(
          indices=((0, 0), (1, 0), (1, 1)),
          values=(2, 1, 0),
          dense_shape=(2, 2))
      id_tensor, weight_tensor = _transform_features({
          'ids': inputs,
          'values': ((.5, 0.), (1., .1)),
      }, (column,))[column]
      with _initialized_session():
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=inputs.indices,
                values=np.array(inputs.values, dtype=np.int64),
                dense_shape=inputs.dense_shape), self.evaluate(id_tensor))
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=((0, 0), (1, 0), (1, 1)),
                values=np.array((.5, 1., .1), dtype=np.float32),
                dense_shape=(2, 2)), self.evaluate(weight_tensor))

  def test_keras_linear_model(self):
    # [FUNCTION DOC]
    # Weighted column through the Keras-style linear model helper; asserts zero initial state, then
    # after assigning weights asserts predictions equal weighted sums of the selected weights.
    column = fc._weighted_categorical_column(
        categorical_column=fc._categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    with ops.Graph().as_default():
      predictions = get_keras_linear_model_predictions({
          'ids':
              sparse_tensor.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(0, 2, 1),
                  dense_shape=(2, 2)),
          'values':
              sparse_tensor.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(.5, 1., .1),
                  dense_shape=(2, 2))
      }, (column,))
      bias = get_linear_model_bias()
      weight_var = get_linear_model_column_var(column)
      with _initialized_session():
        self.assertAllClose((0.,), self.evaluate(bias))
        self.assertAllClose(((0.,), (0.,), (0.,)), self.evaluate(weight_var))
        self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
        weight_var.assign(((1.,), (2.,), (3.,))).eval()
        # weight_var[0] * weights[0, 0] = 1 * .5 = .5
        # weight_var[2] * weights[1, 0] + weight_var[1] * weights[1, 1]
        # = 3*1 + 2*.1 = 3+.2 = 3.2
        self.assertAllClose(((.5,), (3.2,)), self.evaluate(predictions))

  def test_keras_linear_model_mismatched_shape(self):
    # [FUNCTION DOC]
    # Feeds weights with a different sparse structure than the ids; asserts a value error saying
    # dimensions are not compatible.
    column = fc._weighted_categorical_column(
        categorical_column=fc._categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    with ops.Graph().as_default():
      with self.assertRaisesRegex(ValueError,
                                  r'Dimensions.*are not compatible'):
        get_keras_linear_model_predictions({
            'ids':
                sparse_tensor.SparseTensorValue(
                    indices=((0, 0), (1, 0), (1, 1)),
                    values=(0, 2, 1),
                    dense_shape=(2, 2)),
            'values':
                sparse_tensor.SparseTensorValue(
                    indices=((0, 0), (0, 1), (1, 0), (1, 1)),
                    values=(.5, 11., 1., .1),
                    dense_shape=(2, 2))
        }, (column,))

  def test_keras_linear_model_mismatched_dense_values(self):
    # [FUNCTION DOC]
    # Feeds dense weights with fewer entries than ids and the mean combiner, with the constant folding
    # optimizer turned off (it changes the error message across devices); asserts a runtime op error
    # about incompatible shapes.
    column = fc._weighted_categorical_column(
        categorical_column=fc._categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    with ops.Graph().as_default():
      predictions = get_keras_linear_model_predictions(
          {
              'ids':
                  sparse_tensor.SparseTensorValue(
                      indices=((0, 0), (1, 0), (1, 1)),
                      values=(0, 2, 1),
                      dense_shape=(2, 2)),
              'values': ((.5,), (1.,))
          }, (column,),
          sparse_combiner='mean')
      # Disabling the constant folding optimizer here since it changes the
      # error message differently on CPU and GPU.
      config = config_pb2.ConfigProto()
      config.graph_options.rewrite_options.constant_folding = (
          rewriter_config_pb2.RewriterConfig.OFF)
      with _initialized_session(config):
        with self.assertRaisesRegex(errors.OpError, 'Incompatible shapes'):
          self.evaluate(predictions)

  def test_keras_linear_model_mismatched_dense_shape(self):
    # [FUNCTION DOC]
    # Feeds dense weights with more rows than the ids; asserts the model still evaluates and the
    # predictions equal weighted sums, showing extra weight rows are ignored.
    column = fc._weighted_categorical_column(
        categorical_column=fc._categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    with ops.Graph().as_default():
      predictions = get_keras_linear_model_predictions({
          'ids':
              sparse_tensor.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(0, 2, 1),
                  dense_shape=(2, 2)),
          'values': ((.5,), (1.,), (.1,))
      }, (column,))
      bias = get_linear_model_bias()
      weight_var = get_linear_model_column_var(column)
      with _initialized_session():
        self.assertAllClose((0.,), self.evaluate(bias))
        self.assertAllClose(((0.,), (0.,), (0.,)), self.evaluate(weight_var))
        self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
        weight_var.assign(((1.,), (2.,), (3.,))).eval()
        # weight_var[0] * weights[0, 0] = 1 * .5 = .5
        # weight_var[2] * weights[1, 0] + weight_var[1] * weights[1, 1]
        # = 3*1 + 2*.1 = 3+.2 = 3.2
        self.assertAllClose(((.5,), (3.2,)), self.evaluate(predictions))

  def test_linear_model(self):
    # [FUNCTION DOC]
    # Same scenario as the Keras-style test but via the linear model function: weighted sums of
    # selected weights after assignment.
    column = fc._weighted_categorical_column(
        categorical_column=fc._categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    with ops.Graph().as_default():
      predictions = fc.linear_model({
          'ids': sparse_tensor.SparseTensorValue(
              indices=((0, 0), (1, 0), (1, 1)),
              values=(0, 2, 1),
              dense_shape=(2, 2)),
          'values': sparse_tensor.SparseTensorValue(
              indices=((0, 0), (1, 0), (1, 1)),
              values=(.5, 1., .1),
              dense_shape=(2, 2))
      }, (column,))
      bias = get_linear_model_bias()
      weight_var = get_linear_model_column_var(column)
      with _initialized_session():
        self.assertAllClose((0.,), self.evaluate(bias))
        self.assertAllClose(((0.,), (0.,), (0.,)), self.evaluate(weight_var))
        self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
        weight_var.assign(((1.,), (2.,), (3.,))).eval()
        # weight_var[0] * weights[0, 0] = 1 * .5 = .5
        # weight_var[2] * weights[1, 0] + weight_var[1] * weights[1, 1]
        # = 3*1 + 2*.1 = 3+.2 = 3.2
        self.assertAllClose(((.5,), (3.2,)), self.evaluate(predictions))

  def test_linear_model_mismatched_shape(self):
    # [FUNCTION DOC]
    # Weights with a different sparse structure than the ids passed to the linear model function;
    # asserts a value error saying dimensions are not compatible.
    column = fc._weighted_categorical_column(
        categorical_column=fc._categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    with ops.Graph().as_default():
      with self.assertRaisesRegex(ValueError,
                                  r'Dimensions.*are not compatible'):
        fc.linear_model({
            'ids': sparse_tensor.SparseTensorValue(
                indices=((0, 0), (1, 0), (1, 1)),
                values=(0, 2, 1),
                dense_shape=(2, 2)),
            'values': sparse_tensor.SparseTensorValue(
                indices=((0, 0), (0, 1), (1, 0), (1, 1)),
                values=(.5, 11., 1., .1),
                dense_shape=(2, 2))
        }, (column,))

  def test_linear_model_mismatched_dense_values(self):
    # [FUNCTION DOC]
    # Dense weights with fewer entries than ids and the mean combiner, constant folding disabled;
    # asserts a runtime op error about incompatible shapes.
    column = fc._weighted_categorical_column(
        categorical_column=fc._categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    with ops.Graph().as_default():
      predictions = fc.linear_model(
          {
              'ids':
                  sparse_tensor.SparseTensorValue(
                      indices=((0, 0), (1, 0), (1, 1)),
                      values=(0, 2, 1),
                      dense_shape=(2, 2)),
              'values': ((.5,), (1.,))
          }, (column,),
          sparse_combiner='mean')
      # Disabling the constant folding optimizer here since it changes the
      # error message differently on CPU and GPU.
      config = config_pb2.ConfigProto()
      config.graph_options.rewrite_options.constant_folding = (
          rewriter_config_pb2.RewriterConfig.OFF)
      with _initialized_session(config):
        with self.assertRaisesRegex(errors.OpError, 'Incompatible shapes'):
          self.evaluate(predictions)

  def test_linear_model_mismatched_dense_shape(self):
    # [FUNCTION DOC]
    # Dense weights with more rows than the ids; asserts the linear model function still evaluates and
    # predictions equal the weighted sums.
    column = fc._weighted_categorical_column(
        categorical_column=fc._categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    with ops.Graph().as_default():
      predictions = fc.linear_model({
          'ids': sparse_tensor.SparseTensorValue(
              indices=((0, 0), (1, 0), (1, 1)),
              values=(0, 2, 1),
              dense_shape=(2, 2)),
          'values': ((.5,), (1.,), (.1,))
      }, (column,))
      bias = get_linear_model_bias()
      weight_var = get_linear_model_column_var(column)
      with _initialized_session():
        self.assertAllClose((0.,), self.evaluate(bias))
        self.assertAllClose(((0.,), (0.,), (0.,)), self.evaluate(weight_var))
        self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
        weight_var.assign(((1.,), (2.,), (3.,))).eval()
        # weight_var[0] * weights[0, 0] = 1 * .5 = .5
        # weight_var[2] * weights[1, 0] + weight_var[1] * weights[1, 1]
        # = 3*1 + 2*.1 = 3+.2 = 3.2
        self.assertAllClose(((.5,), (3.2,)), self.evaluate(predictions))

  # TODO(ptucker): Add test with embedding of weighted categorical.

if __name__ == '__main__':
  test.main()
