# [MODULE DOC]
# Unit tests for the version-2 feature column library (feature_column_v2) together with its
# interaction with the legacy (version-1) feature column library and the old linear model and
# input layer builders.
# The module defines a few small shared helpers (an initialized-session factory, accessors for
# linear model variables, a dummy base feature column, a recording state manager and a sparse
# tensor comparison helper) followed by one test case class per column kind: lazy transformation
# cache, numeric, bucketized, hashed categorical, crossed, vocabulary file (plain and gzip
# tfrecord variants), vocabulary list, identity, indicator, embedding, shared embedding and
# weighted categorical columns, plus tests for the legacy linear model, input layer, parse-spec
# creation and multi-column transformation.
# Typical checks cover constructor validation errors, default attribute values, deep copying,
# parse specifications, feature transformation results, dense and sparse tensor retrieval, linear
# model predictions with hand-computed weights, and configuration serialization round trips.
# Importing the module has no side effects beyond imports; running it as a script starts the
# standard TensorFlow test runner. This is a test module, not a library module.
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
from tensorflow.python.feature_column import feature_column as fc_old
from tensorflow.python.feature_column import feature_column_v2 as fc
from tensorflow.python.feature_column import feature_column_v2_types
from tensorflow.python.feature_column import serialization
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
  # HELPER: Shared test utility that gives tests a ready-to-use graph-mode session.
  # Creates a new session using the optional configuration protocol buffer (default none), runs
  # the global variable initializer and the lookup table initializer in it, and returns the
  # session so tests can use it as a context manager.
  # Side effects: opens a session on the current default graph and executes initialization
  # operations; the caller is responsible for closing it. Raises whatever errors the initializers
  # raise.
  sess = session.Session(config=config)
  sess.run(variables_lib.global_variables_initializer())
  sess.run(lookup_ops.tables_initializer())
  return sess


def get_linear_model_bias(name='linear_model'):
  # [FUNCTION DOC]
  # HELPER: Shared test utility for inspecting variables created by the legacy linear model
  # builder.
  # Looks up the bias variable named bias_weights inside the variable scope with the given name
  # (default linear_model), re-entering that scope in reuse mode, and returns the variable.
  # Raises an error if the scope or variable does not exist yet, so it must be called after the
  # linear model has been built in the current graph.
  with variable_scope.variable_scope(name, reuse=True):
    return variable_scope.get_variable('bias_weights')


def get_linear_model_column_var(column, name='linear_model'):
  # [FUNCTION DOC]
  # HELPER: Shared test utility for inspecting variables created by the legacy linear model
  # builder.
  # Takes a feature column and an optional model scope name (default linear_model) and returns the
  # first global variable in the graph whose name is the scope name followed by a slash and the
  # column name, i.e. the weights variable for that column.
  # Raises an index error if no such variable exists, so it must be called after the linear model
  # has been built.
  return ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES,
                            name + '/' + column.name)[0]


class BaseFeatureColumnForTests(feature_column_v2_types.FeatureColumn):
  # [CLASS DOC]
  # HELPER: Test-only base class that removes boilerplate when tests define their own feature
  # column subclasses.
  # Derives from the version-2 abstract feature column type and supplies stub implementations of
  # the abstract members for parents, from_config and get_config, each of which raises a value
  # error stating the method should not be used.
  # Test subclasses override only the members they need (name, transform, parse spec, etc.); no
  # state is held and no registration happens.
  """A base FeatureColumn useful to avoid boiler-plate in tests.

  Provides dummy implementations for abstract methods that raise ValueError in
  order to avoid re-defining all abstract methods for each test sub-class.
  """

  @property
  def parents(self):
    # [FUNCTION DOC]
    # HELPER: Stub of the abstract parents property used by test feature columns.
    # Property that always raises a value error saying the method should not be used; it never
    # returns a value.
    raise ValueError('Should not use this method.')

  @classmethod
  def from_config(cls, config, custom_objects=None, columns_by_name=None):
    # [FUNCTION DOC]
    # HELPER: Stub of the abstract class-level deserialization constructor used by test feature
    # columns.
    # Class method taking a configuration mapping, an optional custom-objects mapping and an
    # optional mapping of already-built columns by name; ignores all of them and always raises a
    # value error saying it should not be used.
    raise ValueError('Should not use this method.')

  def get_config(self):
    # [FUNCTION DOC]
    # HELPER: Stub of the abstract serialization method used by test feature columns.
    # Always raises a value error saying the method should not be used; it never returns a
    # configuration.
    raise ValueError('Should not use this method.')


class SortableFeatureColumnTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the ordering behavior of feature columns when they are compared with each
  # other or with strings. Contains a single test and no shared fixtures.

  def test_sort_columns_by_string_representation(self):
    # [FUNCTION DOC]
    # Scenario: builds two version-2 numeric columns and one legacy numeric column, places them in
    # a list together with plain strings, and sorts the reversed list.
    # Asserts that sorting yields the expected order (strings sort around the columns by their
    # text representation), that a version-2 column compares less than another version-2 column
    # and less than a legacy column, that the reverse comparison is false, and that comparison
    # with strings works in both directions.
    # These should be sorted lexicographically based on their string
    # representations. For FeatureColumns, this looks like
    # '<__main__.FeatureColumn object at ...>'.

    a = fc.numeric_column('first')  # '<__main__.NumericColumn ...>'
    b = fc.numeric_column('second')  # '<__main__.NumericColumn ...>'
    c = fc_old._numeric_column('third')  # '<__main__._NumericColumn ...>'

    sorted_sequence = ['0', a, b, c, 'd']
    reversed_sequence = sorted_sequence[::-1]
    self.assertAllEqual(sorted(reversed_sequence), sorted_sequence)

    # pylint: disable=g-generic-assert
    self.assertTrue(a < b)  # V2 < V2 feature columns.
    self.assertTrue(a < c)  # V2 < V1 feature columns.
    self.assertFalse(c < a)  # V1 < V2 feature columns.
    self.assertTrue('0' < a)  # string < V2 feature column.
    self.assertTrue(a < 'd')  # V2 feature column < string.
    # pylint: enable=g-generic-assert


class LazyColumnTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the feature transformation cache: lazy evaluation, caching, error handling
  # for unknown keys or unsupported column types, and expansion of empty rank-1 sparse tensors.
  # Each test defines small local feature column subclasses on top of the dummy base column.

  def test_transformations_called_once(self):
    # [FUNCTION DOC]
    # Scenario: defines a column that counts how many times its transform method is invoked and
    # returns the cached feature named a.
    # Creates a transformation cache over a small feature mapping, asks the cache for the column
    # twice, and asserts the counter is zero before and exactly one after both requests, proving
    # the transformation result is cached.

    class TransformCounter(BaseFeatureColumnForTests):
      # [CLASS DOC]
      # HELPER: Local test fixture column used by the lazy transformation caching test.
      # Feature column subclass that counts transform calls in an instance counter starting at
      # zero and reports itself as a version-2 column named TransformCounter. Its transform looks
      # up the feature named a through the cache and its parse spec is empty.

      def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Constructor of the local counting test column.
        # Calls the parent constructor and initializes the transform call counter to zero. Takes
        # no arguments and returns nothing.
        super(TransformCounter, self).__init__()
        self.num_transform = 0

      @property
      def _is_v2_column(self):
        # [FUNCTION DOC]
        # HELPER: Marker property on a local test column.
        # Property returning true so the column is treated as a version-2 feature column.
        return True

      @property
      def name(self):
        # [FUNCTION DOC]
        # HELPER: Name property of a local test column.
        # Property returning the fixed string TransformCounter used as the column name.
        return 'TransformCounter'

      def transform_feature(self, transformation_cache, state_manager):
        # [FUNCTION DOC]
        # HELPER: Transform hook of the local counting test column.
        # Increments the call counter, then returns the cached value of the input feature named a,
        # obtained from the given transformation cache and state manager. Side effect: mutates the
        # counter.
        self.num_transform += 1  # Count transform calls.
        return transformation_cache.get('a', state_manager)

      @property
      def parse_example_spec(self):
        # [FUNCTION DOC]
        # HELPER: Parse spec property of a local test column.
        # Property that does nothing and returns the null value; the test never parses examples
        # with this column.
        pass

    transformation_cache = fc.FeatureTransformationCache(
        features={'a': [[2], [3.]]})
    column = TransformCounter()
    self.assertEqual(0, column.num_transform)
    transformation_cache.get(column, None)
    self.assertEqual(1, column.num_transform)
    transformation_cache.get(column, None)
    self.assertEqual(1, column.num_transform)

  def test_returns_transform_output(self):
    # [FUNCTION DOC]
    # Scenario: defines a column whose transform returns the fixed string Output.
    # Asks a transformation cache for that column twice and asserts both results equal that
    # string, showing the cache returns whatever the column transform produced.

    class Transformer(BaseFeatureColumnForTests):
      # [CLASS DOC]
      # HELPER: Local test fixture column used by transformation cache tests.
      # Feature column subclass named Transformer, flagged as version 2, whose transform returns
      # the fixed string Output and whose parse spec is empty.

      @property
      def _is_v2_column(self):
        # [FUNCTION DOC]
        # HELPER: Marker property on a local test column.
        # Property returning true so the column is treated as a version-2 feature column.
        return True

      @property
      def name(self):
        # [FUNCTION DOC]
        # HELPER: Name property of a local test column.
        # Property returning the fixed string Transformer.
        return 'Transformer'

      def transform_feature(self, transformation_cache, state_manager):
        # [FUNCTION DOC]
        # HELPER: Transform hook of a local test column.
        # Ignores the cache and state manager and returns the fixed string Output.
        return 'Output'

      @property
      def parse_example_spec(self):
        # [FUNCTION DOC]
        # HELPER: Parse spec property of a local test column.
        # Property that does nothing and returns the null value.
        pass

    transformation_cache = fc.FeatureTransformationCache(
        features={'a': [[2], [3.]]})
    column = Transformer()
    self.assertEqual('Output', transformation_cache.get(column, None))
    self.assertEqual('Output', transformation_cache.get(column, None))

  def test_does_not_pollute_given_features_dict(self):
    # [FUNCTION DOC]
    # Scenario: creates a transformation cache from a feature mapping holding only the key a and
    # requests the transformed value of a local column that returns a constant string.
    # Asserts that afterwards the original mapping still contains only the key a, i.e. the cache
    # does not insert transformed results into the caller-supplied mapping.

    class Transformer(BaseFeatureColumnForTests):
      # [CLASS DOC]
      # HELPER: Local test fixture column used by transformation cache tests.
      # Feature column subclass named Transformer, flagged as version 2, whose transform returns
      # the fixed string Output and whose parse spec is empty.

      @property
      def _is_v2_column(self):
        # [FUNCTION DOC]
        # HELPER: Marker property on a local test column.
        # Property returning true so the column is treated as a version-2 feature column.
        return True

      @property
      def name(self):
        # [FUNCTION DOC]
        # HELPER: Name property of a local test column.
        # Property returning the fixed string Transformer.
        return 'Transformer'

      def transform_feature(self, transformation_cache, state_manager):
        # [FUNCTION DOC]
        # HELPER: Transform hook of a local test column.
        # Ignores the cache and state manager and returns the fixed string Output.
        return 'Output'

      @property
      def parse_example_spec(self):
        # [FUNCTION DOC]
        # HELPER: Parse spec property of a local test column.
        # Property that does nothing and returns the null value.
        pass

    features = {'a': [[2], [3.]]}
    transformation_cache = fc.FeatureTransformationCache(features=features)
    transformation_cache.get(Transformer(), None)
    self.assertEqual(['a'], list(features.keys()))

  def test_error_if_feature_is_not_found(self):
    # [FUNCTION DOC]
    # Scenario: transformation cache built over a mapping with only the key a.
    # Asks the cache for a missing string key, both as a byte-native string and as a unicode
    # string, and asserts each request raises a value error saying the key is not in the features
    # dictionary.
    transformation_cache = fc.FeatureTransformationCache(
        features={'a': [[2], [3.]]})
    with self.assertRaisesRegex(ValueError,
                                'bbb is not in features dictionary'):
      transformation_cache.get('bbb', None)
    with self.assertRaisesRegex(ValueError,
                                'bbb is not in features dictionary'):
      transformation_cache.get(u'bbb', None)

  def test_not_supported_feature_column(self):
    # [FUNCTION DOC]
    # Scenario: defines a column whose transform returns nothing (the null value) and asks the
    # cache for it.
    # Asserts a value error is raised stating the column is not supported, because a
    # transformation must produce a non-null result.

    class NotAProperColumn(BaseFeatureColumnForTests):
      # [CLASS DOC]
      # HELPER: Local test fixture column used to trigger the unsupported-column error.
      # Feature column subclass named NotAProperColumn, flagged as version 2, whose transform
      # returns nothing and whose parse spec is empty.

      @property
      def _is_v2_column(self):
        # [FUNCTION DOC]
        # HELPER: Marker property on a local test column.
        # Property returning true so the column is treated as a version-2 feature column.
        return True

      @property
      def name(self):
        # [FUNCTION DOC]
        # HELPER: Name property of a local test column.
        # Property returning the fixed string NotAProperColumn.
        return 'NotAProperColumn'

      def transform_feature(self, transformation_cache, state_manager):
        # [FUNCTION DOC]
        # HELPER: Transform hook of a local test column.
        # Deliberately does nothing and returns the null value, which the cache must reject as an
        # invalid transformation result.
        # It should return not None.
        pass

      @property
      def parse_example_spec(self):
        # [FUNCTION DOC]
        # HELPER: Parse spec property of a local test column.
        # Property that does nothing and returns the null value.
        pass

    transformation_cache = fc.FeatureTransformationCache(
        features={'a': [[2], [3.]]})
    with self.assertRaisesRegex(ValueError,
                                'NotAProperColumn is not supported'):
      transformation_cache.get(NotAProperColumn(), None)

  def test_key_should_be_string_or_feature_colum(self):
    # [FUNCTION DOC]
    # Scenario: asks a transformation cache for a key that is an instance of a plain unrelated
    # class.
    # Asserts a type error is raised saying the key must be either a string or a feature column.
    # The test name has a typo (colum) but the behavior tested is as described.

    class NotAFeatureColumn(object):
      # [CLASS DOC]
      # HELPER: Local test fixture used to provide an invalid key type.
      # Plain empty class that is neither a string nor a feature column; used only to trigger the
      # key type validation error.
      pass

    transformation_cache = fc.FeatureTransformationCache(
        features={'a': [[2], [3.]]})
    with self.assertRaisesRegex(
        TypeError, '"key" must be either a "str" or "FeatureColumn".'):
      transformation_cache.get(NotAFeatureColumn(), None)

  def test_expand_dim_rank_1_sparse_tensor_empty_batch(self):
    # [FUNCTION DOC]
    # Scenario: transformation cache over a rank-1 sparse tensor that is empty (no indices, dense
    # shape of zero).
    # Requests the feature, evaluates it, and asserts the cache expanded it to rank 2: the dense
    # shape becomes zero by one and the indices array has shape zero by two. Verifies empty-batch
    # handling of dimension expansion.
    # empty 1-D sparse tensor:
    transformation_cache = fc.FeatureTransformationCache(
        features={
            'a':
                sparse_tensor.SparseTensor(
                    indices=np.reshape(np.array([], dtype=np.int64), (0, 1)),
                    dense_shape=[0],
                    values=np.array([]))
        })

    spv = self.evaluate(transformation_cache.get('a', None))
    self.assertAllEqual(np.array([0, 1], dtype=np.int64), spv.dense_shape)
    self.assertAllEqual(
        np.reshape(np.array([], dtype=np.int64), (0, 2)), spv.indices)


class NumericColumnTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the numeric column: constructor validation, default values, shape and
  # default-value normalization, parsing, normalizer function use, dense tensor retrieval, deep
  # copy, legacy linear model use and configuration serialization.

  def test_defaults(self):
    # [FUNCTION DOC]
    # Creates a numeric column with only a key and asserts the defaults: name equals key, shape is
    # a one-element tuple of one, default value is nothing, dtype is 32-bit float, no normalizer
    # function, and the column is flagged version 2.
    a = fc.numeric_column('aaa')
    self.assertEqual('aaa', a.key)
    self.assertEqual('aaa', a.name)
    self.assertEqual((1,), a.shape)
    self.assertIsNone(a.default_value)
    self.assertEqual(dtypes.float32, a.dtype)
    self.assertIsNone(a.normalizer_fn)
    self.assertTrue(a._is_v2_column)

  def test_key_should_be_string(self):
    # [FUNCTION DOC]
    # Asserts that creating a numeric column with a tuple as key raises a value error saying the
    # key must be a string.
    with self.assertRaisesRegex(ValueError, 'key must be a string.'):
      fc.numeric_column(key=('aaa',))

  def test_shape_saved_as_tuple(self):
    # [FUNCTION DOC]
    # Creates a numeric column with a list shape of one by two and asserts the stored shape is a
    # tuple with the same dimensions.
    a = fc.numeric_column('aaa', shape=[1, 2], default_value=[[3, 2.]])
    self.assertEqual((1, 2), a.shape)

  def test_default_value_saved_as_tuple(self):
    # [FUNCTION DOC]
    # Creates numeric columns with a scalar default and a nested-list default and asserts the
    # stored default values are converted to a tuple of floats and a nested tuple matching the
    # requested shape.
    a = fc.numeric_column('aaa', default_value=4.)
    self.assertEqual((4.,), a.default_value)
    a = fc.numeric_column('aaa', shape=[1, 2], default_value=[[3, 2.]])
    self.assertEqual(((3., 2.),), a.default_value)

  def test_shape_and_default_value_compatibility(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: After the first failing column creation inside the first error-checking block,
    # the remaining statements of that block (building a three-by-two column and asserting its
    # default) can never run because the exception is raised first, so that part of the test is
    # dead code.
    # Scenario: checks that default values must match the declared shape. A compatible
    # one-dimensional pair is accepted and stored as a tuple; an incompatible length of three for
    # shape two raises a value error mentioning the shape of default_value.
    # Two further mismatches (shape three by one and three by three against a three-by-two
    # default) are asserted to raise the same kind of error.
    a = fc.numeric_column('aaa', shape=[2], default_value=[1, 2.])
    self.assertEqual((1, 2.), a.default_value)
    with self.assertRaisesRegex(ValueError, 'The shape of default_value'):
      fc.numeric_column('aaa', shape=[2], default_value=[1, 2, 3.])
      a = fc.numeric_column(
          'aaa', shape=[3, 2], default_value=[[2, 3], [1, 2], [2, 3.]])
      self.assertEqual(((2, 3), (1, 2), (2, 3.)), a.default_value)
    with self.assertRaisesRegex(ValueError, 'The shape of default_value'):
      fc.numeric_column(
          'aaa', shape=[3, 1], default_value=[[2, 3], [1, 2], [2, 3.]])
    with self.assertRaisesRegex(ValueError, 'The shape of default_value'):
      fc.numeric_column(
          'aaa', shape=[3, 3], default_value=[[2, 3], [1, 2], [2, 3.]])

  def test_default_value_type_check(self):
    # [FUNCTION DOC]
    # Asserts dtype checks on default values: a float default with float32 and an integer default
    # with int32 are accepted, while a float default with int32 raises a type error about
    # compatibility with the dtype, and a string default raises a type error saying default_value
    # must be compatible with dtype.
    fc.numeric_column(
        'aaa', shape=[2], default_value=[1, 2.], dtype=dtypes.float32)
    fc.numeric_column(
        'aaa', shape=[2], default_value=[1, 2], dtype=dtypes.int32)
    with self.assertRaisesRegex(TypeError, 'must be compatible with dtype'):
      fc.numeric_column(
          'aaa', shape=[2], default_value=[1, 2.], dtype=dtypes.int32)
    with self.assertRaisesRegex(TypeError,
                                'default_value must be compatible with dtype'):
      fc.numeric_column('aaa', default_value=['string'])

  def test_shape_must_be_positive_integer(self):
    # [FUNCTION DOC]
    # Asserts that a non-integer shape dimension raises a type error saying dimensions must be
    # integer, and a zero dimension raises a value error saying dimensions must be greater than
    # zero.
    with self.assertRaisesRegex(TypeError, 'shape dimensions must be integer'):
      fc.numeric_column(
          'aaa', shape=[
              1.0,
          ])

    with self.assertRaisesRegex(ValueError,
                                'shape dimensions must be greater than 0'):
      fc.numeric_column(
          'aaa', shape=[
              0,
          ])

  def test_dtype_is_convertible_to_float(self):
    # [FUNCTION DOC]
    # Asserts that creating a numeric column with string dtype raises a value error saying the
    # dtype must be convertible to float.
    with self.assertRaisesRegex(ValueError,
                                'dtype must be convertible to float'):
      fc.numeric_column('aaa', dtype=dtypes.string)

  def test_scalar_default_value_fills_the_shape(self):
    # [FUNCTION DOC]
    # Creates a numeric column with shape two by three and a scalar default of two and asserts the
    # stored default is a nested tuple of that shape filled with the scalar.
    a = fc.numeric_column('aaa', shape=[2, 3], default_value=2.)
    self.assertEqual(((2., 2., 2.), (2., 2., 2.)), a.default_value)

  def test_parse_spec(self):
    # [FUNCTION DOC]
    # Creates an int32 numeric column of shape two by three and asserts its parse example spec
    # maps the key to a fixed-length feature with that shape and dtype.
    a = fc.numeric_column('aaa', shape=[2, 3], dtype=dtypes.int32)
    self.assertEqual({
        'aaa': parsing_ops.FixedLenFeature((2, 3), dtype=dtypes.int32)
    }, a.parse_example_spec)

  def test_parse_example_no_default_value(self):
    # [FUNCTION DOC]
    # Scenario: serialized example with a two-element float feature named price; the column has no
    # default.
    # Parses the example using the spec made from the column and asserts the key is present and
    # the parsed value equals the two floats.
    price = fc.numeric_column('price', shape=[2])
    data = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'price':
                    feature_pb2.Feature(
                        float_list=feature_pb2.FloatList(value=[20., 110.]))
            }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec_v2([price]))
    self.assertIn('price', features)

    self.assertAllEqual([[20., 110.]], self.evaluate(features['price']))

  def test_parse_example_with_default_value(self):
    # [FUNCTION DOC]
    # Scenario: two serialized examples, one containing price and one containing an unrelated
    # feature; the column has a default of eleven.
    # Parses both with the generated spec and asserts the first row holds the real values while
    # the second row is filled with the default value for both elements.
    price = fc.numeric_column('price', shape=[2], default_value=11.)
    data = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'price':
                    feature_pb2.Feature(
                        float_list=feature_pb2.FloatList(value=[20., 110.]))
            }))
    no_data = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'something_else':
                    feature_pb2.Feature(
                        float_list=feature_pb2.FloatList(value=[20., 110.]))
            }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString(),
                    no_data.SerializeToString()],
        features=fc.make_parse_example_spec_v2([price]))
    self.assertIn('price', features)

    self.assertAllEqual([[20., 110.], [11., 11.]],
                        self.evaluate(features['price']))

  def test_normalizer_fn_must_be_callable(self):
    # [FUNCTION DOC]
    # Asserts that passing a non-callable string as the normalizer function raises a type error
    # saying it must be a callable.
    with self.assertRaisesRegex(TypeError, 'must be a callable'):
      fc.numeric_column('price', normalizer_fn='NotACallable')

  def test_normalizer_fn_transform_feature(self):
    # [FUNCTION DOC]
    # Scenario: numeric column with a normalizer that adds two to its input.
    # Transforms a small feature mapping through the version-2 transform helper and asserts each
    # output element equals the input plus two.

    def _increment_two(input_tensor):
      # [FUNCTION DOC]
      # HELPER: Local normalizer function used by the numeric column test.
      # Takes an input tensor and returns it with two added to every element.
      return input_tensor + 2.

    price = fc.numeric_column('price', shape=[2], normalizer_fn=_increment_two)
    output = fc._transform_features_v2({
        'price': [[1., 2.], [5., 6.]]
    }, [price], None)

    self.assertAllEqual([[3., 4.], [7., 8.]], self.evaluate(output[price]))

  def test_get_dense_tensor(self):
    # [FUNCTION DOC]
    # Scenario: numeric column with the add-two normalizer and a transformation cache over a two
    # by two feature.
    # Asserts that the dense tensor obtained from the column equals the cached transformed feature
    # for that column.

    def _increment_two(input_tensor):
      # [FUNCTION DOC]
      # HELPER: Local normalizer function used by the numeric column test.
      # Takes an input tensor and returns it with two added to every element.
      return input_tensor + 2.

    price = fc.numeric_column('price', shape=[2], normalizer_fn=_increment_two)
    transformation_cache = fc.FeatureTransformationCache({
        'price': [[1., 2.], [5., 6.]]
    })
    self.assertAllEqual(
        transformation_cache.get(price, None),
        price.get_dense_tensor(transformation_cache, None))

  def test_sparse_tensor_not_supported(self):
    # [FUNCTION DOC]
    # Scenario: numeric column whose feature value in the cache is a sparse tensor.
    # Asserts that calling the column transform raises a value error saying the input must be a
    # dense tensor.
    price = fc.numeric_column('price')
    transformation_cache = fc.FeatureTransformationCache({
        'price':
            sparse_tensor.SparseTensor(
                indices=[[0, 0]], values=[0.3], dense_shape=[1, 1])
    })
    with self.assertRaisesRegex(ValueError, 'must be a Tensor'):
      price.transform_feature(transformation_cache, None)

  def test_deep_copy(self):
    # [FUNCTION DOC]
    # Creates a numeric column with a non-trivial shape and default, makes a deep copy, and
    # asserts the copy keeps the name, shape and default value.
    a = fc.numeric_column('aaa', shape=[1, 2], default_value=[[3., 2.]])
    a_copy = copy.deepcopy(a)
    self.assertEqual(a_copy.name, 'aaa')
    self.assertEqual(a_copy.shape, (1, 2))
    self.assertEqual(a_copy.default_value, ((3., 2.),))

  def test_numpy_default_value(self):
    # [FUNCTION DOC]
    # Creates a numeric column whose default value is a numpy array and asserts the stored default
    # is converted to the equivalent nested tuple of floats.
    a = fc.numeric_column(
        'aaa', shape=[1, 2], default_value=np.array([[3., 2.]]))
    self.assertEqual(a.default_value, ((3., 2.),))

  def test_old_linear_model(self):
    # [FUNCTION DOC]
    # Scenario: version-2 numeric column used in the legacy linear model builder inside a fresh
    # graph.
    # Verifies the bias and the column weight start at zero and predictions are zero, then assigns
    # a weight of ten and asserts predictions equal ten times each input value (ten and fifty).
    price = fc.numeric_column('price')
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      predictions = fc_old.linear_model(features, [price])
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        self.assertAllClose([[0.]], self.evaluate(price_var))
        self.assertAllClose([[0.], [0.]], self.evaluate(predictions))
        sess.run(price_var.assign([[10.]]))
        self.assertAllClose([[10.], [50.]], self.evaluate(predictions))

  def test_serialization(self):
    # [FUNCTION DOC]
    # Scenario: numeric column with a custom normalizer function.
    # Asserts its parents are just the key, that its configuration mapping contains key, shape,
    # default value, dtype name and the normalizer function name, and that rebuilding from
    # configuration with the function supplied via the custom-objects mapping yields an equal
    # column. Also round-trips through the generic serialize and deserialize utilities.

    def _increment_two(input_tensor):
      # [FUNCTION DOC]
      # HELPER: Local normalizer function used by the numeric column test.
      # Takes an input tensor and returns it with two added to every element.
      return input_tensor + 2.

    price = fc.numeric_column('price', normalizer_fn=_increment_two)
    self.assertEqual(['price'], price.parents)

    config = price.get_config()
    self.assertEqual({
        'key': 'price',
        'shape': (1,),
        'default_value': None,
        'dtype': 'float32',
        'normalizer_fn': '_increment_two'
    }, config)

    new_col = fc.NumericColumn.from_config(
        config, custom_objects={'_increment_two': _increment_two})
    self.assertEqual(price, new_col)
    self.assertEqual(new_col.shape, (1,))

    # Also test round trip through feature column serialization utils.
    new_col = serialization.deserialize_feature_column(
        serialization.serialize_feature_column(price),
        custom_objects={'_increment_two': _increment_two})
    self.assertEqual(price, new_col)


class BucketizedColumnTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the bucketized column: argument validation, naming, version flags, parse spec,
  # shape and bucket counts, transformation to bucket indices, dense one-hot and sparse id
  # retrieval, rejection of sparse input, deep copy, legacy linear model behavior and
  # serialization.

  def test_invalid_source_column_type(self):
    # [FUNCTION DOC]
    # Asserts that bucketizing a hashed categorical column raises a value error saying the source
    # column must be generated with a numeric column.
    a = fc.categorical_column_with_hash_bucket('aaa', hash_bucket_size=10)
    with self.assertRaisesRegex(
        ValueError,
        'source_column must be a column generated with numeric_column'):
      fc.bucketized_column(a, boundaries=[0, 1])

  def test_invalid_source_column_shape(self):
    # [FUNCTION DOC]
    # Asserts that bucketizing a numeric column of shape two by three raises a value error saying
    # the source must be a one-dimensional column.
    a = fc.numeric_column('aaa', shape=[2, 3])
    with self.assertRaisesRegex(ValueError,
                                'source_column must be one-dimensional column'):
      fc.bucketized_column(a, boundaries=[0, 1])

  def test_invalid_boundaries(self):
    # [FUNCTION DOC]
    # Asserts that boundaries that are missing, a scalar, descending or containing duplicates each
    # raise a value error (empty boundaries message for none, sorted-list message for the others).
    a = fc.numeric_column('aaa')
    with self.assertRaisesRegex(ValueError, 'boundaries must not be empty'):
      fc.bucketized_column(a, boundaries=None)
    with self.assertRaisesRegex(ValueError, 'boundaries must be a sorted list'):
      fc.bucketized_column(a, boundaries=1.)
    with self.assertRaisesRegex(ValueError, 'boundaries must be a sorted list'):
      fc.bucketized_column(a, boundaries=[1, 0])
    with self.assertRaisesRegex(ValueError, 'boundaries must be a sorted list'):
      fc.bucketized_column(a, boundaries=[1, 1])

  def test_name(self):
    # [FUNCTION DOC]
    # Builds a bucketized column over a version-2 int32 numeric column and asserts it is flagged
    # version 2 and its name is the source name followed by the bucketized suffix.
    a = fc.numeric_column('aaa', dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    self.assertTrue(b._is_v2_column)
    self.assertEqual('aaa_bucketized', b.name)

  def test_is_v2_column_old_numeric(self):
    # [FUNCTION DOC]
    # Builds a bucketized column over a legacy numeric column and asserts it is not flagged
    # version 2 while still having the bucketized-suffix name.
    a = fc_old._numeric_column('aaa', dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    self.assertFalse(b._is_v2_column)
    self.assertEqual('aaa_bucketized', b.name)

  def test_parse_spec(self):
    # [FUNCTION DOC]
    # Asserts the bucketized column's parse spec is the source numeric column's fixed-length
    # feature spec with shape two and int32 dtype.
    a = fc.numeric_column('aaa', shape=[2], dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    self.assertEqual({
        'aaa': parsing_ops.FixedLenFeature((2,), dtype=dtypes.int32)
    }, b.parse_example_spec)

  def test_variable_shape(self):
    # [FUNCTION DOC]
    # Asserts that a source of shape two with two boundaries (three buckets) gives a variable
    # shape of two by three.
    a = fc.numeric_column('aaa', shape=[2], dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    # Column 'aaa` has shape [2] times three buckets -> variable_shape=[2, 3].
    self.assertAllEqual((2, 3), b.variable_shape)

  def test_num_buckets(self):
    # [FUNCTION DOC]
    # Asserts that a source of shape two with two boundaries gives six buckets in total (three per
    # input element).
    a = fc.numeric_column('aaa', shape=[2], dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    # Column 'aaa` has shape [2] times three buckets -> num_buckets=6.
    self.assertEqual(6, b.num_buckets)

  def test_parse_example(self):
    # [FUNCTION DOC]
    # Parses a serialized example with a two-element float price feature using the spec derived
    # from the bucketized column and asserts the raw values are parsed unchanged under the source
    # key.
    price = fc.numeric_column('price', shape=[2])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 50])
    data = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'price':
                    feature_pb2.Feature(
                        float_list=feature_pb2.FloatList(value=[20., 110.]))
            }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec_v2([bucketized_price]))
    self.assertIn('price', features)

    self.assertAllEqual([[20., 110.]], self.evaluate(features['price']))

  def test_transform_feature(self):
    # [FUNCTION DOC]
    # Scenario: bucketized price column with boundaries zero, two, four and six, inside a fresh
    # graph.
    # Transforms a two by two feature and asserts the output holds bucket indices for each value
    # (for example negative one falls in bucket zero and six in bucket four).
    price = fc.numeric_column('price', shape=[2])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      transformed_tensor = fc._transform_features_v2({
          'price': [[-1., 1.], [5., 6.]]
      }, [bucketized_price], None)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllEqual([[0, 1], [3, 4]],
                          self.evaluate(transformed_tensor[bucketized_price]))

  def test_get_dense_tensor_one_input_value(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring says the private underscore method is tested, but the test calls
    # the public get_dense_tensor method of the column.
    # Scenario: one-element price per example and four examples; boundaries zero, two, four, six.
    # Calls the column's dense tensor method through a transformation cache and asserts the result
    # is a one-hot encoding over five buckets with shape of four examples by one by five.
    """Tests _get_dense_tensor() for input with shape=[1]."""
    price = fc.numeric_column('price', shape=[1])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      transformation_cache = fc.FeatureTransformationCache({
          'price': [[-1.], [1.], [5.], [6.]]
      })

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      bucketized_price_tensor = bucketized_price.get_dense_tensor(
          transformation_cache, None)
      self.assertAllClose(
          # One-hot tensor.
          [[[1., 0., 0., 0., 0.]], [[0., 1., 0., 0., 0.]],
           [[0., 0., 0., 1., 0.]], [[0., 0., 0., 0., 1.]]],
          self.evaluate(bucketized_price_tensor))

  def test_get_dense_tensor_two_input_values(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring says the private underscore method is tested, but the test calls
    # the public get_dense_tensor method of the column.
    # Scenario: two-element price per example and two examples.
    # Asserts the dense tensor is a one-hot encoding per element, shaped examples by elements by
    # buckets.
    """Tests _get_dense_tensor() for input with shape=[2]."""
    price = fc.numeric_column('price', shape=[2])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      transformation_cache = fc.FeatureTransformationCache({
          'price': [[-1., 1.], [5., 6.]]
      })

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      bucketized_price_tensor = bucketized_price.get_dense_tensor(
          transformation_cache, None)
      self.assertAllClose(
          # One-hot tensor.
          [[[1., 0., 0., 0., 0.], [0., 1., 0., 0., 0.]],
           [[0., 0., 0., 1., 0.], [0., 0., 0., 0., 1.]]],
          self.evaluate(bucketized_price_tensor))

  def test_get_sparse_tensors_one_input_value(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring says the private underscore method is tested, but the test calls
    # the public get_sparse_tensors method of the column.
    # Scenario: one-element price per example, four examples, inside a graph with an initialized
    # session.
    # Asserts there is no weight tensor and the id tensor has one id per example with the expected
    # indices, bucket id values and dense shape of four by one.
    """Tests _get_sparse_tensors() for input with shape=[1]."""
    price = fc.numeric_column('price', shape=[1])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      transformation_cache = fc.FeatureTransformationCache({
          'price': [[-1.], [1.], [5.], [6.]]
      })
      with _initialized_session() as sess:
        id_weight_pair = bucketized_price.get_sparse_tensors(
            transformation_cache, None)
        self.assertIsNone(id_weight_pair.weight_tensor)
        id_tensor_value = sess.run(id_weight_pair.id_tensor)
        self.assertAllEqual([[0, 0], [1, 0], [2, 0], [3, 0]],
                            id_tensor_value.indices)
        self.assertAllEqual([0, 1, 3, 4], id_tensor_value.values)
        self.assertAllEqual([4, 1], id_tensor_value.dense_shape)

  def test_get_sparse_tensors_two_input_values(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring says the private underscore method is tested, but the test calls
    # the public get_sparse_tensors method of the column.
    # Scenario: two-element price per example, two examples.
    # Asserts no weight tensor and that ids are offset per input element (first element ids zero
    # to four, second element ids five to nine), giving the expected indices, values and dense
    # shape of two by two.
    """Tests _get_sparse_tensors() for input with shape=[2]."""
    price = fc.numeric_column('price', shape=[2])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      transformation_cache = fc.FeatureTransformationCache({
          'price': [[-1., 1.], [5., 6.]]
      })
      with _initialized_session() as sess:
        id_weight_pair = bucketized_price.get_sparse_tensors(
            transformation_cache, None)
        self.assertIsNone(id_weight_pair.weight_tensor)
        id_tensor_value = sess.run(id_weight_pair.id_tensor)
        self.assertAllEqual([[0, 0], [0, 1], [1, 0], [1, 1]],
                            id_tensor_value.indices)
        # Values 0-4 correspond to the first column of the input price.
        # Values 5-9 correspond to the second column of the input price.
        self.assertAllEqual([0, 6, 3, 9], id_tensor_value.values)
        self.assertAllEqual([2, 2], id_tensor_value.dense_shape)

  def test_sparse_tensor_input_not_supported(self):
    # [FUNCTION DOC]
    # Asserts that transforming a bucketized column whose feature is a sparse tensor raises a
    # value error saying the input must be a dense tensor.
    price = fc.numeric_column('price')
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 1])
    transformation_cache = fc.FeatureTransformationCache({
        'price':
            sparse_tensor.SparseTensor(
                indices=[[0, 0]], values=[0.3], dense_shape=[1, 1])
    })
    with self.assertRaisesRegex(ValueError, 'must be a Tensor'):
      bucketized_price.transform_feature(transformation_cache, None)

  def test_deep_copy(self):
    # [FUNCTION DOC]
    # Deep copies a bucketized column and asserts the copy keeps its name, variable shape of two
    # by three and boundaries tuple.
    a = fc.numeric_column('aaa', shape=[2])
    a_bucketized = fc.bucketized_column(a, boundaries=[0, 1])
    a_bucketized_copy = copy.deepcopy(a_bucketized)
    self.assertEqual(a_bucketized_copy.name, 'aaa_bucketized')
    self.assertAllEqual(a_bucketized_copy.variable_shape, (2, 3))
    self.assertEqual(a_bucketized_copy.boundaries, (0, 1))

  def test_old_linear_model_one_input_value(self):
    # [FUNCTION DOC]
    # Scenario: legacy linear model over a bucketized one-element price column.
    # Verifies bias and five bucket weights start at zero with zero predictions, assigns weights
    # ten through fifty and asserts predictions equal the weight of each example's bucket, then
    # sets the bias to one and asserts predictions each increase by one.
    """Tests linear_model() for input with shape=[1]."""
    price = fc.numeric_column('price', shape=[1])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      features = {'price': [[-1.], [1.], [5.], [6.]]}
      predictions = fc_old.linear_model(features, [bucketized_price])
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

  def test_old_linear_model_two_input_values(self):
    # [FUNCTION DOC]
    # Scenario: legacy linear model over a bucketized two-element price column.
    # Verifies ten zero weights initially, assigns weights ten to one hundred and asserts each
    # prediction is the sum of the two selected bucket weights, then adds a bias of one and checks
    # the shifted predictions.
    """Tests linear_model() for input with shape=[2]."""
    price = fc.numeric_column('price', shape=[2])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      features = {'price': [[-1., 1.], [5., 6.]]}
      predictions = fc_old.linear_model(features, [bucketized_price])
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

  def test_old_linear_model_one_input_value_old_numeric(self):
    # [FUNCTION DOC]
    # Same scenario as the one-input linear model test, but the bucketized column is built over a
    # legacy numeric column; asserts the same weight initialization, predictions after assigning
    # weights and the effect of the bias.
    """Tests linear_model() for input with shape=[1]."""
    price = fc_old._numeric_column('price', shape=[1])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 2, 4, 6])
    with ops.Graph().as_default():
      features = {'price': [[-1.], [1.], [5.], [6.]]}
      predictions = fc_old.linear_model(features, [bucketized_price])
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

  def test_serialization(self):
    # [FUNCTION DOC]
    # Asserts the bucketized column's parents are its source column, that the configuration holds
    # the nested source column configuration and boundaries, and that rebuilding from
    # configuration gives an equal column with a fresh source column object.
    # Also rebuilds using a mapping from column names to existing columns and asserts the original
    # source column object is reused.
    price = fc.numeric_column('price', shape=[2])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 2, 4, 6])
    self.assertEqual([price], bucketized_price.parents)

    config = bucketized_price.get_config()
    self.assertEqual({
        'source_column': {
            'class_name': 'NumericColumn',
            'config': {
                'key': 'price',
                'shape': (2,),
                'default_value': None,
                'dtype': 'float32',
                'normalizer_fn': None
            }
        },
        'boundaries': (0, 2, 4, 6)
    }, config)

    new_bucketized_price = fc.BucketizedColumn.from_config(config)
    self.assertEqual(bucketized_price, new_bucketized_price)
    self.assertIsNot(price, new_bucketized_price.source_column)

    new_bucketized_price = fc.BucketizedColumn.from_config(
        config,
        columns_by_name={
            serialization._column_name_with_class_name(price): price
        })
    self.assertEqual(bucketized_price, new_bucketized_price)
    self.assertIs(price, new_bucketized_price.source_column)


class HashedCategoricalColumnTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the hash-bucket categorical column: constructor validation, defaults, deep copy,
  # parse spec for string and integer dtypes, parsing, hashing of string and integer inputs with
  # exact expected bucket values, sparse tensor retrieval, legacy linear model use and
  # serialization.

  def test_defaults(self):
    # [FUNCTION DOC]
    # Creates a hashed column with bucket size ten and asserts the default name and key, bucket
    # size, string dtype and the version-2 flag.
    a = fc.categorical_column_with_hash_bucket('aaa', 10)
    self.assertEqual('aaa', a.name)
    self.assertEqual('aaa', a.key)
    self.assertEqual(10, a.hash_bucket_size)
    self.assertEqual(dtypes.string, a.dtype)
    self.assertTrue(a._is_v2_column)

  def test_key_should_be_string(self):
    # [FUNCTION DOC]
    # Asserts a tuple key raises a value error saying the key must be a string.
    with self.assertRaisesRegex(ValueError, 'key must be a string.'):
      fc.categorical_column_with_hash_bucket(('key',), 10)

  def test_bucket_size_should_be_given(self):
    # [FUNCTION DOC]
    # Asserts a missing hash bucket size raises a value error saying it must be set.
    with self.assertRaisesRegex(ValueError, 'hash_bucket_size must be set.'):
      fc.categorical_column_with_hash_bucket('aaa', None)

  def test_bucket_size_should_be_positive(self):
    # [FUNCTION DOC]
    # Asserts a bucket size of zero raises a value error saying it must be at least one.
    with self.assertRaisesRegex(ValueError,
                                'hash_bucket_size must be at least 1'):
      fc.categorical_column_with_hash_bucket('aaa', 0)

  def test_dtype_should_be_string_or_integer(self):
    # [FUNCTION DOC]
    # Asserts string and int32 dtypes are accepted and stored, while float32 raises a value error
    # saying the dtype must be string or integer.
    a = fc.categorical_column_with_hash_bucket('aaa', 10, dtype=dtypes.string)
    b = fc.categorical_column_with_hash_bucket('aaa', 10, dtype=dtypes.int32)
    self.assertEqual(dtypes.string, a.dtype)
    self.assertEqual(dtypes.int32, b.dtype)

    with self.assertRaisesRegex(ValueError, 'dtype must be string or integer'):
      fc.categorical_column_with_hash_bucket('aaa', 10, dtype=dtypes.float32)

  def test_deep_copy(self):
    # [FUNCTION DOC]
    # Checks both an original column and its deep copy retain name, bucket size, number of buckets
    # (equal to bucket size) and dtype.
    original = fc.categorical_column_with_hash_bucket('aaa', 10)
    for column in (original, copy.deepcopy(original)):
      self.assertEqual('aaa', column.name)
      self.assertEqual(10, column.hash_bucket_size)
      self.assertEqual(10, column.num_buckets)
      self.assertEqual(dtypes.string, column.dtype)

  def test_parse_spec_string(self):
    # [FUNCTION DOC]
    # Asserts the parse spec of a default hashed column is a variable-length feature of string
    # dtype under the key.
    a = fc.categorical_column_with_hash_bucket('aaa', 10)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.string)
    }, a.parse_example_spec)

  def test_parse_spec_int(self):
    # [FUNCTION DOC]
    # Asserts the parse spec of an int32 hashed column is a variable-length feature of int32 dtype
    # under the key.
    a = fc.categorical_column_with_hash_bucket('aaa', 10, dtype=dtypes.int32)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int32)
    }, a.parse_example_spec)

  def test_parse_example(self):
    # [FUNCTION DOC]
    # Parses a serialized example holding two byte strings and asserts the resulting sparse tensor
    # has the expected indices, string values and dense shape, using the module's sparse
    # comparison helper.
    a = fc.categorical_column_with_hash_bucket('aaa', 10)
    data = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'aaa':
                    feature_pb2.Feature(
                        bytes_list=feature_pb2.BytesList(
                            value=[b'omar', b'stringer']))
            }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec_v2([a]))
    self.assertIn('aaa', features)

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [0, 1]],
            values=np.array([b'omar', b'stringer'], dtype=np.object_),
            dense_shape=[1, 2]), self.evaluate(features['aaa']))

  def test_strings_should_be_hashed(self):
    # [FUNCTION DOC]
    # Scenario: sparse string input fed through the transform helper to a hashed column of ten
    # buckets.
    # Asserts the output values are 64-bit integers with exact expected bucket ids (six, four,
    # one), while indices and dense shape are unchanged. Fails if the hash function ever changes.
    hashed_sparse = fc.categorical_column_with_hash_bucket('wire', 10)
    wire_tensor = sparse_tensor.SparseTensor(
        values=['omar', 'stringer', 'marlo'],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    outputs = fc._transform_features_v2({
        'wire': wire_tensor
    }, [hashed_sparse], None)
    output = outputs[hashed_sparse]
    # Check exact hashed output. If hashing changes this test will break.
    expected_values = [6, 4, 1]

    self.assertEqual(dtypes.int64, output.values.dtype)
    self.assertAllEqual(expected_values, self.evaluate(output.values))
    self.assertAllEqual(
        self.evaluate(wire_tensor.indices), self.evaluate(output.indices))
    self.assertAllEqual(
        self.evaluate(wire_tensor.dense_shape),
        self.evaluate(output.dense_shape))

  def test_tensor_dtype_should_be_string_or_integer(self):
    # [FUNCTION DOC]
    # Scenario: three columns and three tensors (string, int and float values) in one cache.
    # Asserts the string and int column lookups succeed while the one whose tensor holds floats
    # raises a value error saying dtype must be string or integer.
    string_fc = fc.categorical_column_with_hash_bucket(
        'a_string', 10, dtype=dtypes.string)
    int_fc = fc.categorical_column_with_hash_bucket(
        'a_int', 10, dtype=dtypes.int32)
    float_fc = fc.categorical_column_with_hash_bucket(
        'a_float', 10, dtype=dtypes.string)
    int_tensor = sparse_tensor.SparseTensor(
        values=[101], indices=[[0, 0]], dense_shape=[1, 1])
    string_tensor = sparse_tensor.SparseTensor(
        values=['101'], indices=[[0, 0]], dense_shape=[1, 1])
    float_tensor = sparse_tensor.SparseTensor(
        values=[101.], indices=[[0, 0]], dense_shape=[1, 1])
    transformation_cache = fc.FeatureTransformationCache({
        'a_int': int_tensor,
        'a_string': string_tensor,
        'a_float': float_tensor
    })
    transformation_cache.get(string_fc, None)
    transformation_cache.get(int_fc, None)
    with self.assertRaisesRegex(ValueError, 'dtype must be string or integer'):
      transformation_cache.get(float_fc, None)

  def test_dtype_should_match_with_tensor(self):
    # [FUNCTION DOC]
    # Asserts that a hashed column declared as int64 given a string tensor raises a value error
    # saying the dtype must be compatible.
    hashed_sparse = fc.categorical_column_with_hash_bucket(
        'wire', 10, dtype=dtypes.int64)
    wire_tensor = sparse_tensor.SparseTensor(
        values=['omar'], indices=[[0, 0]], dense_shape=[1, 1])
    transformation_cache = fc.FeatureTransformationCache({'wire': wire_tensor})
    with self.assertRaisesRegex(ValueError, 'dtype must be compatible'):
      transformation_cache.get(hashed_sparse, None)

  def test_ints_should_be_hashed(self):
    # [FUNCTION DOC]
    # Hashes an int64 sparse tensor through a transformation cache and asserts the exact expected
    # bucket ids (three, seven, five).
    hashed_sparse = fc.categorical_column_with_hash_bucket(
        'wire', 10, dtype=dtypes.int64)
    wire_tensor = sparse_tensor.SparseTensor(
        values=[101, 201, 301],
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    transformation_cache = fc.FeatureTransformationCache({'wire': wire_tensor})
    output = transformation_cache.get(hashed_sparse, None)
    # Check exact hashed output. If hashing changes this test will break.
    expected_values = [3, 7, 5]

    self.assertAllEqual(expected_values, self.evaluate(output.values))

  def test_int32_64_is_compatible(self):
    # [FUNCTION DOC]
    # Same as the integer hashing test but the tensor holds int32 values while the column is
    # int64; asserts no error and the same expected bucket ids, showing the two integer widths are
    # compatible.
    hashed_sparse = fc.categorical_column_with_hash_bucket(
        'wire', 10, dtype=dtypes.int64)
    wire_tensor = sparse_tensor.SparseTensor(
        values=constant_op.constant([101, 201, 301], dtype=dtypes.int32),
        indices=[[0, 0], [1, 0], [1, 1]],
        dense_shape=[2, 2])
    transformation_cache = fc.FeatureTransformationCache({'wire': wire_tensor})
    output = transformation_cache.get(hashed_sparse, None)
    # Check exact hashed output. If hashing changes this test will break.
    expected_values = [3, 7, 5]

    self.assertAllEqual(expected_values, self.evaluate(output.values))

  def test_get_sparse_tensors(self):
    # [FUNCTION DOC]
    # Asserts get_sparse_tensors on a hashed column returns a pair whose weight tensor is nothing
    # and whose id tensor equals the cache's transformed feature.
    hashed_sparse = fc.categorical_column_with_hash_bucket('wire', 10)
    transformation_cache = fc.FeatureTransformationCache({
        'wire':
            sparse_tensor.SparseTensor(
                values=['omar', 'stringer', 'marlo'],
                indices=[[0, 0], [1, 0], [1, 1]],
                dense_shape=[2, 2])
    })
    id_weight_pair = hashed_sparse.get_sparse_tensors(transformation_cache,
                                                      None)
    self.assertIsNone(id_weight_pair.weight_tensor)
    self.assertEqual(
        transformation_cache.get(hashed_sparse, None), id_weight_pair.id_tensor)

  def test_get_sparse_tensors_dense_input(self):
    # [FUNCTION DOC]
    # Same as above but the input feature is a dense nested tuple of strings including an empty
    # string; asserts no weight tensor and that the id tensor equals the transformed value.
    hashed_sparse = fc.categorical_column_with_hash_bucket('wire', 10)
    transformation_cache = fc.FeatureTransformationCache({
        'wire': (('omar', ''), ('stringer', 'marlo'))
    })
    id_weight_pair = hashed_sparse.get_sparse_tensors(transformation_cache,
                                                      None)
    self.assertIsNone(id_weight_pair.weight_tensor)
    self.assertEqual(
        transformation_cache.get(hashed_sparse, None), id_weight_pair.id_tensor)

  def test_old_linear_model(self):
    # [FUNCTION DOC]
    # Scenario: legacy linear model over a four-bucket hashed column with string sparse input.
    # Verifies zero bias, weights and predictions, assigns weights one through four and asserts
    # predictions equal the hand-computed sums for the hashed ids of each example (four and six).
    wire_column = fc.categorical_column_with_hash_bucket('wire', 4)
    self.assertEqual(4, wire_column.num_buckets)
    with ops.Graph().as_default():
      predictions = fc_old.linear_model({
          wire_column.name:
              sparse_tensor.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=('marlo', 'skywalker', 'omar'),
                  dense_shape=(2, 2))
      }, (wire_column,))
      bias = get_linear_model_bias()
      wire_var = get_linear_model_column_var(wire_column)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose((0.,), self.evaluate(bias))
      self.assertAllClose(((0.,), (0.,), (0.,), (0.,)), self.evaluate(wire_var))
      self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
      self.evaluate(wire_var.assign(((1.,), (2.,), (3.,), (4.,))))
      # 'marlo' -> 3: wire_var[3] = 4
      # 'skywalker' -> 2, 'omar' -> 2: wire_var[2] + wire_var[2] = 3+3 = 6
      self.assertAllClose(((4.,), (6.,)), self.evaluate(predictions))

  def test_serialization(self):
    # [FUNCTION DOC]
    # Asserts the hashed column's parents are its key, its configuration holds key, bucket size
    # and dtype name, and rebuilding from configuration gives an equal column.
    wire_column = fc.categorical_column_with_hash_bucket('wire', 4)
    self.assertEqual(['wire'], wire_column.parents)

    config = wire_column.get_config()
    self.assertEqual({
        'key': 'wire',
        'hash_bucket_size': 4,
        'dtype': 'string'
    }, config)

    self.assertEqual(wire_column,
                     fc.HashedCategoricalColumn.from_config(config))


class CrossedColumnTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the crossed column: key validation, hash bucket size validation, naming
  # (alphabetical and flattened), version flag, parse spec, bucket count, deep copy, parsing,
  # transformation, hashed id outputs with exact expected values, legacy linear model use and
  # serialization.

  def test_keys_empty(self):
    # [FUNCTION DOC]
    # Asserts that an empty key list raises a value error saying keys must be a list with length
    # greater than one.
    with self.assertRaisesRegex(ValueError,
                                'keys must be a list with length > 1'):
      fc.crossed_column([], 10)

  def test_keys_length_one(self):
    # [FUNCTION DOC]
    # Asserts that a single key raises the same keys-length value error.
    with self.assertRaisesRegex(ValueError,
                                'keys must be a list with length > 1'):
      fc.crossed_column(['a'], 10)

  def test_key_type_unsupported(self):
    # [FUNCTION DOC]
    # Asserts that a numeric column key raises a value error about an unsupported key type, and a
    # hashed categorical column key raises a value error saying hash bucket categorical columns
    # are not supported as keys.
    with self.assertRaisesRegex(ValueError, 'Unsupported key type'):
      fc.crossed_column(['a', fc.numeric_column('c')], 10)

    with self.assertRaisesRegex(
        ValueError, 'categorical_column_with_hash_bucket is not supported'):
      fc.crossed_column(
          ['a', fc.categorical_column_with_hash_bucket('c', 10)], 10)

  def test_hash_bucket_size_negative(self):
    # [FUNCTION DOC]
    # Asserts a negative hash bucket size raises a value error saying it must be greater than one.
    with self.assertRaisesRegex(ValueError, 'hash_bucket_size must be > 1'):
      fc.crossed_column(['a', 'c'], -1)

  def test_hash_bucket_size_zero(self):
    # [FUNCTION DOC]
    # Asserts a zero hash bucket size raises the same value error.
    with self.assertRaisesRegex(ValueError, 'hash_bucket_size must be > 1'):
      fc.crossed_column(['a', 'c'], 0)

  def test_hash_bucket_size_none(self):
    # [FUNCTION DOC]
    # Asserts a missing hash bucket size raises the same value error.
    with self.assertRaisesRegex(ValueError, 'hash_bucket_size must be > 1'):
      fc.crossed_column(['a', 'c'], None)

  def test_name(self):
    # [FUNCTION DOC]
    # Builds a nested cross of a bucketized numeric column, a string key and another crossed
    # column and asserts the name joins all leaf names alphabetically with a cross separator, and
    # that the column is flagged version 2.
    a = fc.numeric_column('a', dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    crossed1 = fc.crossed_column(['d1', 'd2'], 10)
    self.assertTrue(crossed1._is_v2_column)

    crossed2 = fc.crossed_column([b, 'c', crossed1], 10)
    self.assertTrue(crossed2._is_v2_column)
    self.assertEqual('a_bucketized_X_c_X_d1_X_d2', crossed2.name)

  def test_is_v2_column(self):
    # [FUNCTION DOC]
    # Same nesting but with a legacy numeric source; asserts the inner cross is version 2 while
    # the outer cross is not, and the name is unchanged.
    a = fc_old._numeric_column('a', dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    crossed1 = fc.crossed_column(['d1', 'd2'], 10)
    self.assertTrue(crossed1._is_v2_column)

    crossed2 = fc.crossed_column([b, 'c', crossed1], 10)
    self.assertFalse(crossed2._is_v2_column)
    self.assertEqual('a_bucketized_X_c_X_d1_X_d2', crossed2.name)

  def test_name_ordered_alphabetically(self):
    # [FUNCTION DOC]
    # Asserts the crossed column's name does not depend on the order of the supplied keys.
    """Tests that the name does not depend on the order of given columns."""
    a = fc.numeric_column('a', dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    crossed1 = fc.crossed_column(['d1', 'd2'], 10)

    crossed2 = fc.crossed_column([crossed1, 'c', b], 10)
    self.assertEqual('a_bucketized_X_c_X_d1_X_d2', crossed2.name)

  def test_name_leaf_keys_ordered_alphabetically(self):
    # [FUNCTION DOC]
    # Asserts that leaf keys from nested crosses are flattened and sorted alphabetically in the
    # name regardless of nesting and input order.
    """Tests that the name does not depend on the order of given columns."""
    a = fc.numeric_column('a', dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    crossed1 = fc.crossed_column(['d2', 'c'], 10)

    crossed2 = fc.crossed_column([crossed1, 'd1', b], 10)
    self.assertEqual('a_bucketized_X_c_X_d1_X_d2', crossed2.name)

  def test_parse_spec(self):
    # [FUNCTION DOC]
    # Asserts the parse spec combines the bucketized column's fixed-length integer spec and a
    # variable-length string spec for the plain string key.
    a = fc.numeric_column('a', shape=[2], dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    crossed = fc.crossed_column([b, 'c'], 10)
    self.assertEqual({
        'a': parsing_ops.FixedLenFeature((2,), dtype=dtypes.int32),
        'c': parsing_ops.VarLenFeature(dtypes.string),
    }, crossed.parse_example_spec)

  def test_num_buckets(self):
    # [FUNCTION DOC]
    # Asserts the number of buckets of a crossed column equals the hash bucket size passed in
    # (fifteen).
    a = fc.numeric_column('a', shape=[2], dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    crossed = fc.crossed_column([b, 'c'], 15)
    self.assertEqual(15, crossed.num_buckets)

  def test_deep_copy(self):
    # [FUNCTION DOC]
    # Deep copies a crossed column with a hash key and asserts the copy keeps its name, hash
    # bucket size and hash key.
    a = fc.numeric_column('a', dtype=dtypes.int32)
    b = fc.bucketized_column(a, boundaries=[0, 1])
    crossed1 = fc.crossed_column(['d1', 'd2'], 10)
    crossed2 = fc.crossed_column([b, 'c', crossed1], 15, hash_key=5)
    crossed2_copy = copy.deepcopy(crossed2)
    self.assertEqual(
        'a_bucketized_X_c_X_d1_X_d2',
        crossed2_copy.name,
    )
    self.assertEqual(15, crossed2_copy.hash_bucket_size)
    self.assertEqual(5, crossed2_copy.hash_key)

  def test_parse_example(self):
    # [FUNCTION DOC]
    # Parses a serialized example containing price and wire features using the spec from a crossed
    # column and asserts both keys are present with the expected dense floats and sparse indices,
    # string values and dense shape.
    price = fc.numeric_column('price', shape=[2])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 50])
    price_cross_wire = fc.crossed_column([bucketized_price, 'wire'], 10)
    data = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'price':
                    feature_pb2.Feature(
                        float_list=feature_pb2.FloatList(value=[20., 110.])),
                'wire':
                    feature_pb2.Feature(
                        bytes_list=feature_pb2.BytesList(
                            value=[b'omar', b'stringer'])),
            }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec_v2([price_cross_wire]))
    self.assertIn('price', features)
    self.assertIn('wire', features)

    self.assertAllEqual([[20., 110.]], self.evaluate(features['price']))
    wire_sparse = features['wire']
    self.assertAllEqual([[0, 0], [0, 1]], self.evaluate(wire_sparse.indices))
    # Use byte constants to pass the open-source test.
    self.assertAllEqual([b'omar', b'stringer'],
                        self.evaluate(wire_sparse.values))
    self.assertAllEqual([1, 2], self.evaluate(wire_sparse.dense_shape))

  def test_transform_feature(self):
    # [FUNCTION DOC]
    # Scenario: cross of a bucketized price column and a sparse string wire column, bucket size
    # ten.
    # Transforms the features, evaluates the resulting sparse tensor and asserts the expected
    # indices (every combination per example), that every hashed value lies in the allowed range,
    # and the dense shape of two by four.
    price = fc.numeric_column('price', shape=[2])
    bucketized_price = fc.bucketized_column(price, boundaries=[0, 50])
    hash_bucket_size = 10
    price_cross_wire = fc.crossed_column([bucketized_price, 'wire'],
                                         hash_bucket_size)
    features = {
        'price':
            constant_op.constant([[1., 2.], [5., 6.]]),
        'wire':
            sparse_tensor.SparseTensor(
                values=['omar', 'stringer', 'marlo'],
                indices=[[0, 0], [1, 0], [1, 1]],
                dense_shape=[2, 2]),
    }
    outputs = fc._transform_features_v2(features, [price_cross_wire], None)
    output = outputs[price_cross_wire]
    output_val = self.evaluate(output)
    self.assertAllEqual([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]],
                        output_val.indices)
    for val in output_val.values:
      self.assertIn(val, list(range(hash_bucket_size)))
    self.assertAllEqual([2, 4], output_val.dense_shape)

  def test_get_sparse_tensors(self):
    # [FUNCTION DOC]
    # Scenario: nested cross of a bucketized int column, a string key and an inner cross, with
    # hash bucket size fifteen and hash key five.
    # Evaluates the id tensor from a cache and asserts the exact index list, the exact hashed ids
    # (which would change if the hash function changed) and the dense shape of two by sixteen.
    # Test is graph mode.
    a = fc.numeric_column('a', dtype=dtypes.int32, shape=(2,))
    b = fc.bucketized_column(a, boundaries=(0, 1))
    crossed1 = fc.crossed_column(['d1', 'd2'], 10)
    crossed2 = fc.crossed_column([b, 'c', crossed1], 15, hash_key=5)
    with ops.Graph().as_default():
      transformation_cache = fc.FeatureTransformationCache({
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
      id_weight_pair = crossed2.get_sparse_tensors(transformation_cache, None)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      id_tensor_eval = self.evaluate(id_weight_pair.id_tensor)
      self.assertAllEqual(
          ((0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
           (1, 6), (1, 7), (1, 8), (1, 9), (1, 10), (1, 11), (1, 12), (1, 13),
           (1, 14), (1, 15)), id_tensor_eval.indices)
      # Check exact hashed output. If hashing changes this test will break.
      # All values are within [0, hash_bucket_size).
      expected_values = (6, 14, 0, 13, 8, 8, 10, 12, 2, 0, 1, 9, 8, 12, 2, 0,
                         10, 11)
      self.assertAllEqual(expected_values, id_tensor_eval.values)
      self.assertAllEqual((2, 16), id_tensor_eval.dense_shape)

  def test_get_sparse_tensors_simple(self):
    # [FUNCTION DOC]
    # Simpler version of the previous test with one bucketized column and one string key, bucket
    # size five and hash key five; asserts the exact indices, ids (one, zero, one, three, four,
    # two) and dense shape of two by four.
    """Same as test_get_sparse_tensors, but with simpler values."""
    a = fc.numeric_column('a', dtype=dtypes.int32, shape=(2,))
    b = fc.bucketized_column(a, boundaries=(0, 1))
    crossed = fc.crossed_column([b, 'c'], hash_bucket_size=5, hash_key=5)
    with ops.Graph().as_default():
      transformation_cache = fc.FeatureTransformationCache({
          'a':
              constant_op.constant(((-1., .5), (.5, 1.))),
          'c':
              sparse_tensor.SparseTensor(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=['cA', 'cB', 'cC'],
                  dense_shape=(2, 2)),
      })
      id_weight_pair = crossed.get_sparse_tensors(transformation_cache, None)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      id_tensor_eval = self.evaluate(id_weight_pair.id_tensor)
      self.assertAllEqual(((0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (1, 3)),
                          id_tensor_eval.indices)
      # Check exact hashed output. If hashing changes this test will break.
      # All values are within [0, hash_bucket_size).
      expected_values = (1, 0, 1, 3, 4, 2)
      self.assertAllEqual(expected_values, id_tensor_eval.values)
      self.assertAllEqual((2, 4), id_tensor_eval.dense_shape)

  def test_old_linear_model(self):
    # [FUNCTION DOC]
    # Scenario: legacy linear model over the crossed column using the data of the simple sparse
    # test.
    # Verifies zero initialization, assigns weights one through five and asserts predictions are
    # the sums of the weights for the crossed ids of each example (three and fourteen), then adds
    # a bias of one tenth and checks the shift.
    """Tests linear_model.

    Uses data from test_get_sparse_tensors_simple.
    """
    a = fc.numeric_column('a', dtype=dtypes.int32, shape=(2,))
    b = fc.bucketized_column(a, boundaries=(0, 1))
    crossed = fc.crossed_column([b, 'c'], hash_bucket_size=5, hash_key=5)
    with ops.Graph().as_default():
      predictions = fc_old.linear_model({
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
      with _initialized_session() as sess:
        self.assertAllClose((0.,), self.evaluate(bias))
        self.assertAllClose(((0.,), (0.,), (0.,), (0.,), (0.,)),
                            self.evaluate(crossed_var))
        self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
        sess.run(crossed_var.assign(((1.,), (2.,), (3.,), (4.,), (5.,))))
        # Expected ids after cross = (1, 0, 1, 3, 4, 2)
        self.assertAllClose(((3.,), (14.,)), self.evaluate(predictions))
        sess.run(bias.assign((.1,)))
        self.assertAllClose(((3.1,), (14.1,)), self.evaluate(predictions))

  def test_old_linear_model_with_weights(self):
    # [FUNCTION DOC]
    # Scenario: a local test column that produces both ids and weights is crossed with a string
    # key and passed to the legacy linear model.
    # Asserts a value error is raised saying the crossed column does not support a weight tensor
    # and naming the offending column.

    class _TestColumnWithWeights(BaseFeatureColumnForTests,
                                 fc.CategoricalColumn,
                                 fc_old._CategoricalColumn):
      # [CLASS DOC]
      # HELPER: Local test fixture column that supplies both sparse ids and sparse weights.
      # Subclass of the dummy base column, the version-2 categorical column and the legacy
      # categorical column. It is flagged version 2, named test_column, has five buckets, declares
      # a parse spec with an id feature and a weights feature, and its legacy transform returns
      # the id and weight features. Version-2 transform and sparse tensor entry points raise
      # errors because only the legacy path is exercised.
      """Produces sparse IDs and sparse weights."""

      @property
      def _is_v2_column(self):
        # [FUNCTION DOC]
        # HELPER: Marker property on a local test column.
        # Property returning true so the column is treated as a version-2 feature column.
        return True

      @property
      def name(self):
        # [FUNCTION DOC]
        # HELPER: Name property of a local test column.
        # Property returning the fixed name test_column.
        return 'test_column'

      @property
      def parse_example_spec(self):
        # [FUNCTION DOC]
        # HELPER: Parse spec property of a local test column.
        # Property returning a mapping from the column name to a variable-length int32 feature and
        # from the name plus a weights suffix to a variable-length float feature.
        return {
            self.name:
                parsing_ops.VarLenFeature(dtypes.int32),
            '{}_weights'.format(self.name):
                parsing_ops.VarLenFeature(dtypes.float32),
        }

      @property
      def _parse_example_spec(self):
        # [FUNCTION DOC]
        # HELPER: Legacy alias property of a local test column.
        # Property returning the same mapping as the version-2 parse spec.
        return self.parse_example_spec

      @property
      def num_buckets(self):
        # [FUNCTION DOC]
        # HELPER: Bucket count property of a local test column.
        # Property returning the fixed value five.
        return 5

      @property
      def _num_buckets(self):
        # [FUNCTION DOC]
        # HELPER: Legacy alias property of a local test column.
        # Property returning the same value as the version-2 bucket count.
        return self.num_buckets

      def transform_feature(self, transformation_cache, state_manager):
        # [FUNCTION DOC]
        # HELPER: Version-2 transform stub of a local test column.
        # Always raises a value error saying it should not be called, because the legacy path is
        # used.
        raise ValueError('Should not be called.')

      def _transform_feature(self, inputs):
        # [FUNCTION DOC]
        # HELPER: Legacy transform hook of a local test column.
        # Takes the legacy input cache and returns a pair made of the id feature and the weights
        # feature looked up by the column name and the name with a weights suffix.
        return (inputs.get(self.name),
                inputs.get('{}_weights'.format(self.name)))

      def get_sparse_tensors(self, transformation_cache, state_manager):
        # [FUNCTION DOC]
        # HELPER: Version-2 sparse tensor stub of a local test column.
        # Always raises a value error saying it should not be called.
        raise ValueError('Should not be called.')

      def _get_sparse_tensors(self,
                              inputs,
                              weight_collections=None,
                              trainable=None):
        # [FUNCTION DOC]
        # HELPER: Legacy sparse tensor hook of a local test column.
        # Takes the legacy input cache plus optional weight collections and trainable flag (both
        # unused), fetches the cached id and weight pair for this column and returns an id and
        # weight pair carrying both tensors. The docstring accurately states it populates both.
        """Populates both id_tensor and weight_tensor."""
        ids_and_weights = inputs.get(self)
        return fc.CategoricalColumn.IdWeightPair(
            id_tensor=ids_and_weights[0], weight_tensor=ids_and_weights[1])

    t = _TestColumnWithWeights()
    crossed = fc.crossed_column([t, 'c'], hash_bucket_size=5, hash_key=5)
    with ops.Graph().as_default():
      with self.assertRaisesRegex(
          ValueError,
          'crossed_column does not support weight_tensor.*{}'.format(t.name)):
        fc_old.linear_model({
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

  def test_old_linear_model_old_numeric(self):
    # [FUNCTION DOC]
    # Same scenario as the version-2 crossed linear model test but the bucketized source is built
    # over a legacy numeric column; asserts the same zero initialization, predictions after
    # assigning weights (three and fourteen) and the bias effect.
    """Tests linear_model.

    Uses data from test_get_sparse_tensors_simple.
    """
    a = fc_old._numeric_column('a', dtype=dtypes.int32, shape=(2,))
    b = fc.bucketized_column(a, boundaries=(0, 1))
    crossed = fc.crossed_column([b, 'c'], hash_bucket_size=5, hash_key=5)
    with ops.Graph().as_default():
      predictions = fc_old.linear_model({
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
      with _initialized_session() as sess:
        self.assertAllClose((0.,), self.evaluate(bias))
        self.assertAllClose(((0.,), (0.,), (0.,), (0.,), (0.,)),
                            self.evaluate(crossed_var))
        self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
        sess.run(crossed_var.assign(((1.,), (2.,), (3.,), (4.,), (5.,))))
        # Expected ids after cross = (1, 0, 1, 3, 4, 2)
        self.assertAllClose(((3.,), (14.,)), self.evaluate(predictions))
        sess.run(bias.assign((.1,)))
        self.assertAllClose(((3.1,), (14.1,)), self.evaluate(predictions))

  def test_serialization(self):
    # [FUNCTION DOC]
    # Asserts the crossed column's parents are its keys, its configuration contains the hash
    # bucket size, hash key and nested key configurations (with the plain string kept as is), and
    # that rebuilding gives an equal column with fresh nested column objects.
    # Rebuilding with a mapping from names to existing columns reuses the original bucketized
    # column object.
    a = fc.numeric_column('a', dtype=dtypes.int32, shape=(2,))
    b = fc.bucketized_column(a, boundaries=(0, 1))
    crossed = fc.crossed_column([b, 'c'], hash_bucket_size=5, hash_key=5)

    self.assertEqual([b, 'c'], crossed.parents)

    config = crossed.get_config()
    self.assertEqual({
        'hash_bucket_size':
            5,
        'hash_key':
            5,
        'keys': ({
            'config': {
                'boundaries': (0, 1),
                'source_column': {
                    'config': {
                        'dtype': 'int32',
                        'default_value': None,
                        'key': 'a',
                        'normalizer_fn': None,
                        'shape': (2,)
                    },
                    'class_name': 'NumericColumn'
                }
            },
            'class_name': 'BucketizedColumn'
        }, 'c')
    }, config)

    new_crossed = fc.CrossedColumn.from_config(config)
    self.assertEqual(crossed, new_crossed)
    self.assertIsNot(b, new_crossed.keys[0])

    new_crossed = fc.CrossedColumn.from_config(
        config,
        columns_by_name={serialization._column_name_with_class_name(b): b})
    self.assertEqual(crossed, new_crossed)
    self.assertIs(b, new_crossed.keys[0])


class OldLinearModelTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the legacy linear model builder when used with version-2 feature columns:
  # argument validation, bias and weights behavior for dense, sparse and mixed columns, multiple
  # outputs, shapes, combiners, variable collections, trainable flags, partitioned variables,
  # ordering, batch size checks, one-dimensional and rank-zero inputs, repeated models and mixing
  # of version-1 and version-2 shared embedding columns.

  def test_raises_if_empty_feature_columns(self):
    # [FUNCTION DOC]
    # Asserts building a linear model with an empty column list raises a value error saying the
    # columns must not be empty.
    with self.assertRaisesRegex(ValueError,
                                'feature_columns must not be empty'):
      fc_old.linear_model(features={}, feature_columns=[])

  def test_should_be_feature_column(self):
    # [FUNCTION DOC]
    # Asserts passing a string instead of columns raises a value error saying it must be a feature
    # column.
    with self.assertRaisesRegex(ValueError, 'must be a _FeatureColumn'):
      fc_old.linear_model(features={'a': [[0]]}, feature_columns='NotSupported')

  def test_should_be_dense_or_categorical_column(self):
    # [FUNCTION DOC]
    # Scenario: a local column that is neither dense nor categorical is passed to the linear
    # model.
    # Asserts a value error saying each column must be either a dense or a categorical column.

    class NotSupportedColumn(
        BaseFeatureColumnForTests,
        feature_column_v2_types.FeatureColumn,
        fc_old._FeatureColumn,
    ):
      # [CLASS DOC]
      # HELPER: Local test fixture column that is a feature column but neither dense nor
      # categorical.
      # Inherits from the dummy base column, the version-2 feature column and the legacy feature
      # column. Reports itself as version 2 with fixed name NotSupportedColumn; its transform
      # hooks and parse specs do nothing.

      @property
      def _is_v2_column(self):
        # [FUNCTION DOC]
        # HELPER: Marker property on a local test column.
        # Property returning true so the column is treated as a version-2 feature column.
        return True

      @property
      def name(self):
        # [FUNCTION DOC]
        # HELPER: Name property of a local test column.
        # Property returning the fixed string NotSupportedColumn.
        return 'NotSupportedColumn'

      def transform_feature(self, transformation_cache, state_manager):
        # [FUNCTION DOC]
        # HELPER: Version-2 transform hook of a local test column.
        # Does nothing and returns the null value.
        pass

      def _transform_feature(self, inputs):
        # [FUNCTION DOC]
        # HELPER: Legacy transform hook of a local test column.
        # Does nothing and returns the null value.
        pass

      @property
      def parse_example_spec(self):
        # [FUNCTION DOC]
        # HELPER: Parse spec property of a local test column.
        # Property that does nothing and returns the null value.
        pass

      @property
      def _parse_example_spec(self):
        # [FUNCTION DOC]
        # HELPER: Legacy parse spec property of a local test column.
        # Property that does nothing and returns the null value.
        pass

    with self.assertRaisesRegex(
        ValueError, 'must be either a _DenseColumn or _CategoricalColumn'):
      fc_old.linear_model(
          features={'a': [[0]]}, feature_columns=[NotSupportedColumn()])

  def test_does_not_support_dict_columns(self):
    # [FUNCTION DOC]
    # Asserts that passing a mapping rather than an iterable of columns raises a value error
    # saying an iterable was expected but a dict was found.
    with self.assertRaisesRegex(
        ValueError, 'Expected feature_columns to be iterable, found dict.'):
      fc_old.linear_model(
          features={'a': [[0]]}, feature_columns={'a': fc.numeric_column('a')})

  def test_raises_if_duplicate_name(self):
    # [FUNCTION DOC]
    # Asserts that two columns with the same name raise a value error about a duplicate column
    # name.
    with self.assertRaisesRegex(
        ValueError, 'Duplicate feature column name found for columns'):
      fc_old.linear_model(
          features={'a': [[0]]},
          feature_columns=[fc.numeric_column('a'),
                           fc.numeric_column('a')])

  def test_dense_bias(self):
    # [FUNCTION DOC]
    # Legacy linear model with one numeric column: asserts the bias starts at zero, then after
    # assigning a weight of ten and bias of five the predictions equal weight times input plus
    # bias.
    price = fc.numeric_column('price')
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      predictions = fc_old.linear_model(features, [price])
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        self.assertAllClose([0.], self.evaluate(bias))
        sess.run(price_var.assign([[10.]]))
        sess.run(bias.assign([5.]))
        self.assertAllClose([[15.], [55.]], self.evaluate(predictions))

  def test_sparse_bias(self):
    # [FUNCTION DOC]
    # Legacy linear model with a hashed categorical column: asserts zero bias and four zero
    # weights, then assigns weights and bias and asserts predictions are the sums of weights for
    # the hashed ids of each example plus bias.
    wire_cast = fc.categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default():
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar', 'stringer', 'marlo'],  # hashed to = [2, 0, 3]
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {'wire_cast': wire_tensor}
      predictions = fc_old.linear_model(features, [wire_cast])
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
    # Legacy linear model with both a hashed categorical and a numeric column: after assigning
    # weights and bias asserts each prediction equals the sum of both contributions plus the
    # single shared bias.
    wire_cast = fc.categorical_column_with_hash_bucket('wire_cast', 4)
    price = fc.numeric_column('price')
    with ops.Graph().as_default():
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar', 'stringer', 'marlo'],  # hashed to = [2, 0, 3]
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {'wire_cast': wire_tensor, 'price': [[1.], [5.]]}
      predictions = fc_old.linear_model(features, [wire_cast, price])
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
    # Scenario: a local column that is both dense and categorical is passed to the linear model.
    # Asserts the model uses the sparse (categorical) path: with weights and bias assigned,
    # predictions match the sparse id lookup result. The docstring is accurate.
    """When the column is both dense and sparse, uses sparse tensors."""

    class _DenseAndSparseColumn(BaseFeatureColumnForTests, fc.DenseColumn,
                                fc.CategoricalColumn, fc_old._DenseColumn,
                                fc_old._CategoricalColumn):
      # [CLASS DOC]
      # HELPER: Local test fixture column that is both a dense and a categorical column.
      # Combines the dummy base column with the version-2 and legacy dense and categorical column
      # types. Named dense_and_sparse_column, version 2, four buckets, with a variable-length
      # parse spec. Only the legacy sparse path is functional; other version-2 entry points raise
      # errors.

      @property
      def _is_v2_column(self):
        # [FUNCTION DOC]
        # HELPER: Marker property on a local test column.
        # Property returning true so the column is treated as a version-2 feature column.
        return True

      @property
      def name(self):
        # [FUNCTION DOC]
        # HELPER: Name property of a local test column.
        # Property returning the fixed string dense_and_sparse_column.
        return 'dense_and_sparse_column'

      @property
      def parse_example_spec(self):
        # [FUNCTION DOC]
        # HELPER: Parse spec property of a local test column.
        # Property returning a mapping from the column name to a variable-length feature with the
        # column's dtype.
        return {self.name: parsing_ops.VarLenFeature(self.dtype)}

      @property
      def _parse_example_spec(self):
        # [FUNCTION DOC]
        # HELPER: Legacy parse spec property of a local test column.
        # Property returning the version-2 parse spec.
        return self.parse_example_spec

      def transform_feature(self, transformation_cache, state_manager):
        # [FUNCTION DOC]
        # HELPER: Version-2 transform stub of a local test column.
        # Always raises a value error saying it should not be used.
        raise ValueError('Should not use this method.')

      def _transform_feature(self, inputs):
        # [FUNCTION DOC]
        # HELPER: Legacy transform hook of a local test column.
        # Returns the input feature stored under the column name in the legacy input cache.
        return inputs.get(self.name)

      @property
      def variable_shape(self):
        # [FUNCTION DOC]
        # HELPER: Variable shape property of a local test column.
        # KNOWN ISSUE: The property returns itself (a self-referential lookup of the same property
        # name), which would recurse forever if it were ever accessed; the test never accesses it,
        # so it is effectively dead code.
        # Intended to report the dense variable shape of the column; as written it simply re-reads
        # the same property.
        return self.variable_shape

      @property
      def _variable_shape(self):
        # [FUNCTION DOC]
        # HELPER: Legacy alias property of a local test column.
        # Property returning the version-2 variable shape (which itself is self-referential, see
        # that property).
        return self.variable_shape

      def get_dense_tensor(self, transformation_cache, state_manager):
        # [FUNCTION DOC]
        # HELPER: Version-2 dense tensor stub of a local test column.
        # Always raises a value error saying it should not be used.
        raise ValueError('Should not use this method.')

      def _get_dense_tensor(self, inputs):
        # [FUNCTION DOC]
        # HELPER: Legacy dense tensor stub of a local test column.
        # Always raises a value error saying it should not be used, which proves the linear model
        # did not take the dense path.
        raise ValueError('Should not use this method.')

      @property
      def num_buckets(self):
        # [FUNCTION DOC]
        # HELPER: Bucket count property of a local test column.
        # Property returning the fixed value four.
        return 4

      @property
      def _num_buckets(self):
        # [FUNCTION DOC]
        # HELPER: Legacy alias property of a local test column.
        # Property returning the same bucket count.
        return self.num_buckets

      def get_sparse_tensors(self, transformation_cache, state_manager):
        # [FUNCTION DOC]
        # HELPER: Version-2 sparse tensor stub of a local test column.
        # Always raises a value error saying it should not be used.
        raise ValueError('Should not use this method.')

      def _get_sparse_tensors(self,
                              inputs,
                              weight_collections=None,
                              trainable=None):
        # [FUNCTION DOC]
        # HELPER: Legacy sparse tensor hook of a local test column.
        # Ignores its input cache and optional weight collections and trainable arguments, and
        # returns an id and weight pair whose id tensor is a fixed two by two sparse tensor with
        # values two, zero and three and whose weight tensor is nothing.
        sp_tensor = sparse_tensor.SparseTensor(
            indices=[[0, 0], [1, 0], [1, 1]],
            values=[2, 0, 3],
            dense_shape=[2, 2])
        return fc.CategoricalColumn.IdWeightPair(sp_tensor, None)

    dense_and_sparse_column = _DenseAndSparseColumn()
    with ops.Graph().as_default():
      sp_tensor = sparse_tensor.SparseTensor(
          values=['omar', 'stringer', 'marlo'],
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {dense_and_sparse_column.name: sp_tensor}
      predictions = fc_old.linear_model(features, [dense_and_sparse_column])
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
    # Legacy linear model with a numeric column and three output units: asserts bias and weights
    # have zero values of the right shape, then after assigning weights and biases asserts each of
    # the three outputs matches the hand-computed values.
    price = fc.numeric_column('price')
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      predictions = fc_old.linear_model(features, [price], units=3)
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
    # Same as above for a hashed categorical column with three units: asserts zero initialization
    # shapes (bias of three, weights four by three) and the predictions after assigning values.
    wire_cast = fc.categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default():
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar', 'stringer', 'marlo'],  # hashed to = [2, 0, 3]
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {'wire_cast': wire_tensor}
      predictions = fc_old.linear_model(features, [wire_cast], units=3)
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      with _initialized_session() as sess:
        self.assertAllClose(np.zeros((3,)), self.evaluate(bias))
        self.assertAllClose(np.zeros((4, 3)), self.evaluate(wire_cast_var))
        sess.run(
            wire_cast_var.assign([[10., 11., 12.], [100., 110., 120.],
                                  [1000., 1100., 1200.],
                                  [10000., 11000., 12000.]]))
        sess.run(bias.assign([5., 6., 7.]))
        self.assertAllClose([[1005., 1106., 1207.], [10015., 11017., 12019.]],
                            self.evaluate(predictions))

  def test_dense_multi_dimension(self):
    # [FUNCTION DOC]
    # Numeric column of shape two: asserts a weight variable of two rows and that predictions
    # equal the dot product of inputs and weights.
    price = fc.numeric_column('price', shape=2)
    with ops.Graph().as_default():
      features = {'price': [[1., 2.], [5., 6.]]}
      predictions = fc_old.linear_model(features, [price])
      price_var = get_linear_model_column_var(price)
      with _initialized_session() as sess:
        self.assertAllClose([[0.], [0.]], self.evaluate(price_var))
        sess.run(price_var.assign([[10.], [100.]]))
        self.assertAllClose([[210.], [650.]], self.evaluate(predictions))

  def test_sparse_multi_rank(self):
    # [FUNCTION DOC]
    # Scenario: a string sparse placeholder of rank three fed through a hashed column in the
    # legacy linear model.
    # Asserts weight shape and zero predictions, then after assigning weights asserts the
    # predictions reduce over the extra dimensions as expected when fed with a sparse value.
    wire_cast = fc.categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default():
      wire_tensor = array_ops.sparse_placeholder(dtypes.string)
      wire_value = sparse_tensor.SparseTensorValue(
          values=['omar', 'stringer', 'marlo', 'omar'],  # hashed = [2, 0, 3, 2]
          indices=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 1]],
          dense_shape=[2, 2, 2])
      features = {'wire_cast': wire_tensor}
      predictions = fc_old.linear_model(features, [wire_cast])
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
    # Legacy linear model with mean combiner over a hashed column: asserts that when an example
    # has several ids the prediction is the mean of the corresponding weights plus bias.
    wire_cast = fc.categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default():
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar', 'stringer', 'marlo'],  # hashed to = [2, 0, 3]
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {'wire_cast': wire_tensor}
      predictions = fc_old.linear_model(
          features, [wire_cast], sparse_combiner='mean')
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      with _initialized_session() as sess:
        sess.run(wire_cast_var.assign([[10.], [100.], [1000.], [10000.]]))
        sess.run(bias.assign([5.]))
        self.assertAllClose([[1005.], [5010.]], self.evaluate(predictions))

  def test_sparse_combiner_with_negative_weights(self):
    # [FUNCTION DOC]
    # Weighted categorical column with a negative weight and the sum combiner: asserts predictions
    # use weight-scaled sums including the negative contribution plus bias.
    wire_cast = fc.categorical_column_with_hash_bucket('wire_cast', 4)
    wire_cast_weights = fc.weighted_categorical_column(wire_cast, 'weights')

    with ops.Graph().as_default():
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar', 'stringer', 'marlo'],  # hashed to = [2, 0, 3]
          indices=[[0, 0], [1, 0], [1, 1]],
          dense_shape=[2, 2])
      features = {
          'wire_cast': wire_tensor,
          'weights': constant_op.constant([[1., 1., -1.0]])
      }
      predictions = fc_old.linear_model(
          features, [wire_cast_weights], sparse_combiner='sum')
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      with _initialized_session() as sess:
        sess.run(wire_cast_var.assign([[10.], [100.], [1000.], [10000.]]))
        sess.run(bias.assign([5.]))
        self.assertAllClose([[1005.], [-9985.]], self.evaluate(predictions))

  def test_dense_multi_dimension_multi_output(self):
    # [FUNCTION DOC]
    # Numeric column of shape two with three units: asserts zero initial variable shapes and the
    # predictions after assigning a weight matrix and bias.
    price = fc.numeric_column('price', shape=2)
    with ops.Graph().as_default():
      features = {'price': [[1., 2.], [5., 6.]]}
      predictions = fc_old.linear_model(features, [price], units=3)
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
    # Asserts that feeding one-element rows to a numeric column declared with two elements raises
    # an error saying a tensor with two elements cannot be reshaped to shape two by two.
    price = fc.numeric_column('price', shape=2)
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      with self.assertRaisesRegex(
          Exception,
          r'Cannot reshape a tensor with 2 elements to shape \[2,2\]'):
        fc_old.linear_model(features, [price])

  def test_dense_reshaping(self):
    # [FUNCTION DOC]
    # Numeric column of shape one by two with matching three-dimensional input: asserts the model
    # flattens the input, has two weights, and predictions equal the weighted sums.
    price = fc.numeric_column('price', shape=[1, 2])
    with ops.Graph().as_default():
      features = {'price': [[[1., 2.]], [[5., 6.]]]}
      predictions = fc_old.linear_model(features, [price])
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
    # Two numeric columns of different widths: asserts each gets its own weight variable of the
    # right shape and predictions are the sum of both weighted contributions plus bias.
    price1 = fc.numeric_column('price1', shape=2)
    price2 = fc.numeric_column('price2')
    with ops.Graph().as_default():
      features = {'price1': [[1., 2.], [5., 6.]], 'price2': [[3.], [4.]]}
      predictions = fc_old.linear_model(features, [price1, price2])
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
    # Passes an empty mapping as the column-to-variables output argument and asserts it is filled
    # with the bias list under a bias key and the weight variable list for each column.
    price1 = fc.numeric_column('price1', shape=2)
    price2 = fc.numeric_column('price2')
    with ops.Graph().as_default():
      features = {'price1': [[1., 2.], [5., 6.]], 'price2': [[3.], [4.]]}
      cols_to_vars = {}
      fc_old.linear_model(features, [price1, price2], cols_to_vars=cols_to_vars)
      bias = get_linear_model_bias()
      price1_var = get_linear_model_column_var(price1)
      price2_var = get_linear_model_column_var(price2)
      self.assertEqual(cols_to_vars['bias'], [bias])
      self.assertEqual(cols_to_vars[price1], [price1_var])
      self.assertEqual(cols_to_vars[price2], [price2_var])

  def test_fills_cols_to_vars_partitioned_variables(self):
    # [FUNCTION DOC]
    # Builds the model inside a variable scope with a fixed-size partitioner and asserts the
    # mapping lists the bias plus the partitioned shards of each column's weights with the
    # expected shard shapes.
    price1 = fc.numeric_column('price1', shape=2)
    price2 = fc.numeric_column('price2', shape=3)
    with ops.Graph().as_default():
      features = {
          'price1': [[1., 2.], [6., 7.]],
          'price2': [[3., 4., 5.], [8., 9., 10.]]
      }
      cols_to_vars = {}
      with variable_scope.variable_scope(
          'linear',
          partitioner=partitioned_variables.fixed_size_partitioner(2, axis=0)):
        fc_old.linear_model(
            features, [price1, price2], cols_to_vars=cols_to_vars)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertEqual([0.], self.evaluate(cols_to_vars['bias'][0]))
      # Partitioning shards the [2, 1] price1 var into 2 [1, 1] Variables.
      self.assertAllEqual([[0.]], self.evaluate(cols_to_vars[price1][0]))
      self.assertAllEqual([[0.]], self.evaluate(cols_to_vars[price1][1]))
      # Partitioning shards the [3, 1] price2 var into a [2, 1] Variable and
      # a [1, 1] Variable.
      self.assertAllEqual([[0.], [0.]], self.evaluate(cols_to_vars[price2][0]))
      self.assertAllEqual([[0.]], self.evaluate(cols_to_vars[price2][1]))

  def test_fills_cols_to_output_tensors(self):
    # [FUNCTION DOC]
    # Uses the legacy input layer (not the linear model) with a numeric, a bucketized and an
    # embedding column and an output mapping argument. Asserts the mapping has exactly those
    # columns as keys and its tensors are the very inputs used to build the layer's concatenation.
    # Provide three _DenseColumn's to input_layer: a _NumericColumn, a
    # _BucketizedColumn, and an _EmbeddingColumn.  Only the _EmbeddingColumn
    # creates a Variable.
    apple_numeric_column = fc.numeric_column('apple_numeric_column')
    banana_dense_feature = fc.numeric_column('banana_dense_feature')
    banana_dense_feature_bucketized = fc.bucketized_column(
        banana_dense_feature, boundaries=[0.])
    cherry_sparse_column = fc.categorical_column_with_hash_bucket(
        'cherry_sparse_feature', hash_bucket_size=5)
    dragonfruit_embedding_column = fc.embedding_column(
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
      input_layer = fc_old.input_layer(
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
    # Asserts that with a weight collections argument the bias and the column weights are added to
    # the named graph collection.
    price = fc.numeric_column('price')
    with ops.Graph().as_default() as g:
      features = {'price': [[1.], [5.]]}
      fc_old.linear_model(features, [price], weight_collections=['my-vars'])
      my_vars = g.get_collection('my-vars')
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      self.assertIn(bias, my_vars)
      self.assertIn(price_var, my_vars)

  def test_sparse_collection(self):
    # [FUNCTION DOC]
    # Same as the dense collection test but for a hashed categorical column.
    wire_cast = fc.categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default() as g:
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar'], indices=[[0, 0]], dense_shape=[1, 1])
      features = {'wire_cast': wire_tensor}
      fc_old.linear_model(features, [wire_cast], weight_collections=['my-vars'])
      my_vars = g.get_collection('my-vars')
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      self.assertIn(bias, my_vars)
      self.assertIn(wire_cast_var, my_vars)

  def test_dense_trainable_default(self):
    # [FUNCTION DOC]
    # Asserts that by default the bias and numeric weights are in the trainable variables
    # collection.
    price = fc.numeric_column('price')
    with ops.Graph().as_default() as g:
      features = {'price': [[1.], [5.]]}
      fc_old.linear_model(features, [price])
      bias = get_linear_model_bias()
      price_var = get_linear_model_column_var(price)
      trainable_vars = g.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      self.assertIn(bias, trainable_vars)
      self.assertIn(price_var, trainable_vars)

  def test_sparse_trainable_default(self):
    # [FUNCTION DOC]
    # Asserts that by default the bias and hashed column weights are trainable.
    wire_cast = fc.categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default() as g:
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar'], indices=[[0, 0]], dense_shape=[1, 1])
      features = {'wire_cast': wire_tensor}
      fc_old.linear_model(features, [wire_cast])
      trainable_vars = g.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      bias = get_linear_model_bias()
      wire_cast_var = get_linear_model_column_var(wire_cast)
      self.assertIn(bias, trainable_vars)
      self.assertIn(wire_cast_var, trainable_vars)

  def test_dense_trainable_false(self):
    # [FUNCTION DOC]
    # Asserts that passing trainable as false leaves the trainable variables collection empty for
    # a numeric column model.
    price = fc.numeric_column('price')
    with ops.Graph().as_default() as g:
      features = {'price': [[1.], [5.]]}
      fc_old.linear_model(features, [price], trainable=False)
      trainable_vars = g.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      self.assertEqual([], trainable_vars)

  def test_sparse_trainable_false(self):
    # [FUNCTION DOC]
    # Asserts the same for a hashed categorical column model.
    wire_cast = fc.categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default() as g:
      wire_tensor = sparse_tensor.SparseTensor(
          values=['omar'], indices=[[0, 0]], dense_shape=[1, 1])
      features = {'wire_cast': wire_tensor}
      fc_old.linear_model(features, [wire_cast], trainable=False)
      trainable_vars = g.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      self.assertEqual([], trainable_vars)

  def test_column_order(self):
    # [FUNCTION DOC]
    # Builds the model twice with different column orders and a named collection; asserts the
    # variables are always added in alphabetical column order (price a, price b, wire cast),
    # showing the order is independent of input order.
    price_a = fc.numeric_column('price_a')
    price_b = fc.numeric_column('price_b')
    wire_cast = fc.categorical_column_with_hash_bucket('wire_cast', 4)
    with ops.Graph().as_default() as g:
      features = {
          'price_a': [[1.]],
          'price_b': [[3.]],
          'wire_cast':
              sparse_tensor.SparseTensor(
                  values=['omar'], indices=[[0, 0]], dense_shape=[1, 1])
      }
      fc_old.linear_model(
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
      fc_old.linear_model(
          features, [wire_cast, price_b, price_a],
          weight_collections=['my-vars'])
      my_vars = g.get_collection('my-vars')
      self.assertIn('price_a', my_vars[0].name)
      self.assertIn('price_b', my_vars[1].name)
      self.assertIn('wire_cast', my_vars[2].name)

  def test_static_batch_size_mismatch(self):
    # [FUNCTION DOC]
    # Scenario: two features with static batch sizes three and two.
    # Asserts that building the model raises a value error saying the batch size of each feature
    # must be the same. Note that the assertion block sits outside the graph context used to build
    # features.
    price1 = fc.numeric_column('price1')
    price2 = fc.numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': [[1.], [5.], [7.]],  # batchsize = 3
          'price2': [[3.], [4.]]  # batchsize = 2
      }
    with self.assertRaisesRegex(
        ValueError,
        r'Batch size \(first dimension\) of each feature must be same.'):  # pylint: disable=anomalous-backslash-in-string
      fc_old.linear_model(features, [price1, price2])

  def test_subset_of_static_batch_size_mismatch(self):
    # [FUNCTION DOC]
    # Mix of an unknown-shape placeholder and two static features with batch sizes two and three;
    # asserts the model still raises the batch size value error at build time.
    price1 = fc.numeric_column('price1')
    price2 = fc.numeric_column('price2')
    price3 = fc.numeric_column('price3')
    with ops.Graph().as_default():
      features = {
          'price1': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 3
          'price2': [[3.], [4.]],  # batchsize = 2
          'price3': [[3.], [4.], [5.]]  # batchsize = 3
      }
      with self.assertRaisesRegex(
          ValueError,
          r'Batch size \(first dimension\) of each feature must be same.'):  # pylint: disable=anomalous-backslash-in-string
        fc_old.linear_model(features, [price1, price2, price3])

  def test_runtime_batch_size_mismatch(self):
    # [FUNCTION DOC]
    # Scenario: one placeholder feature and one static feature of batch two.
    # Runs the predictions with a batch of three for the placeholder and asserts an op error
    # saying the values must have the same size and shape.
    price1 = fc.numeric_column('price1')
    price2 = fc.numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 3
          'price2': [[3.], [4.]]  # batchsize = 2
      }
      predictions = fc_old.linear_model(features, [price1, price2])
      with _initialized_session() as sess:
        with self.assertRaisesRegex(errors.OpError,
                                    'must have the same size and shape'):
          sess.run(
              predictions, feed_dict={features['price1']: [[1.], [5.], [7.]]})

  def test_runtime_batch_size_matches(self):
    # [FUNCTION DOC]
    # Same scenario but both placeholders are fed matching batches; asserts the run completes
    # without error.
    price1 = fc.numeric_column('price1')
    price2 = fc.numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 2
          'price2': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 2
      }
      predictions = fc_old.linear_model(features, [price1, price2])
      with _initialized_session() as sess:
        sess.run(
            predictions,
            feed_dict={
                features['price1']: [[1.], [5.]],
                features['price2']: [[1.], [5.]],
            })

  def test_with_1d_sparse_tensor(self):
    # [FUNCTION DOC]
    # Scenario: rank-one dense price (bucketized) and rank-one sparse body style feature in a
    # legacy graph.
    # Asserts the input ranks are one, then after assigning bucket and vocabulary weights and bias
    # asserts the predictions are the sums of the selected weights. Runs inside a legacy graph
    # because it uses variable scope access.
    # This test does direct variable scope access + manipulations
    # that require entering a legacy graph
    with ops.Graph().as_default():
      price = fc.numeric_column('price')
      price_buckets = fc.bucketized_column(
          price, boundaries=[
              0.,
              10.,
              100.,
          ])
      body_style = fc.categorical_column_with_vocabulary_list(
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

      net = fc_old.linear_model(features, [price_buckets, body_style])
      with _initialized_session() as sess:
        bias = get_linear_model_bias()
        price_buckets_var = get_linear_model_column_var(price_buckets)
        body_style_var = get_linear_model_column_var(body_style)

        sess.run(price_buckets_var.assign([[10.], [100.], [1000.], [10000.]]))
        sess.run(body_style_var.assign([[-10.], [-100.], [-1000.]]))
        sess.run(bias.assign([5.]))

        self.assertAllClose([[10 - 1000 + 5.], [1000 - 10 + 5.]],
                            self.evaluate(net))

  def test_with_1d_unknown_shape_sparse_tensor(self):
    # [FUNCTION DOC]
    # Same idea with unknown-rank placeholders for price, body style and a dense country feature;
    # feeds data at run time and asserts the same predictions, proving rank is resolved
    # dynamically.
    # This test needs to access variables bia variable scope & needs to be
    # run inside of a legacy graph
    with ops.Graph().as_default():
      price = fc.numeric_column('price')
      price_buckets = fc.bucketized_column(
          price, boundaries=[
              0.,
              10.,
              100.,
          ])
      body_style = fc.categorical_column_with_vocabulary_list(
          'body-style', vocabulary_list=['hardtop', 'wagon', 'sedan'])
      country = fc.categorical_column_with_vocabulary_list(
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

      net = fc_old.linear_model(features, [price_buckets, body_style, country])
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

  def test_with_rank_0_feature(self):
    # [FUNCTION DOC]
    # Asserts a statically rank-zero feature raises a value error saying the feature cannot have
    # rank zero, and a dynamically rank-zero placeholder raises an op error with the same message
    # when run.
    price = fc.numeric_column('price')
    features = {
        'price': constant_op.constant(0),
    }
    self.assertEqual(0, features['price'].shape.ndims)

    # Static rank 0 should fail
    with self.assertRaisesRegex(ValueError, 'Feature .* cannot have rank 0'):
      fc_old.linear_model(features, [price])

    # This test needs to construct graph placeholders
    # w/ dynamic rank 0, so we enter a graph
    with ops.Graph().as_default():
      # Dynamic rank 0 should fail
      features = {
          'price': array_ops.placeholder(dtypes.float32),
      }
      net = fc_old.linear_model(features, [price])
      self.assertEqual(1, net.shape[1])
      with _initialized_session() as sess:
        with self.assertRaisesOpError('Feature .* cannot have rank 0'):
          sess.run(net, feed_dict={features['price']: np.array(1)})

  def test_multiple_linear_models(self):
    # [FUNCTION DOC]
    # Builds two linear models on the same column in one graph and asserts the second gets a
    # separate scope (suffixed name) with its own bias and weights; assigns values separately and
    # asserts independent predictions.
    price = fc.numeric_column('price')
    with ops.Graph().as_default():
      features1 = {'price': [[1.], [5.]]}
      features2 = {'price': [[2.], [10.]]}
      predictions1 = fc_old.linear_model(features1, [price])
      predictions2 = fc_old.linear_model(features2, [price])
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

  def test_linear_model_v1_shared_embedding_all_other_v2(self):
    # [FUNCTION DOC]
    # Graph-only scenario: numeric, hashed-embedding and legacy shared embedding columns built on
    # version-2 categorical inputs are all passed to the legacy linear model.
    # Asserts model construction succeeds and the bias is zero after initialization.
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      price = fc.numeric_column('price')  # v2
      some_sparse_column = fc.categorical_column_with_hash_bucket(
          'sparse_feature', hash_bucket_size=5)  # v2
      some_embedding_column = fc.embedding_column(
          some_sparse_column, dimension=10)  # v2
      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=3)  # v2
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=3)  # v2
      shared_embedding_a, shared_embedding_b = fc.shared_embedding_columns(
          [categorical_column_a, categorical_column_b], dimension=2)  # v1
      all_cols = [
          price, some_embedding_column, shared_embedding_a, shared_embedding_b
      ]

      features = {
          'price': [[3.], [4.]],
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
      fc_old.linear_model(features, all_cols)
      bias = get_linear_model_bias()

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([0.], self.evaluate(bias))

  def test_linear_model_v1_shared_embedding_with_v2_cat_all_other_v2(self):
    # [FUNCTION DOC]
    # Same as the previous test with version-2 categorical columns feeding the legacy shared
    # embeddings; asserts construction succeeds and the bias is zero. Nearly identical to the
    # previous test.
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      price = fc.numeric_column('price')  # v2
      some_sparse_column = fc.categorical_column_with_hash_bucket(
          'sparse_feature', hash_bucket_size=5)  # v2
      some_embedding_column = fc.embedding_column(
          some_sparse_column, dimension=10)  # v2
      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=3)  # v2
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=3)  # v2
      shared_embedding_a, shared_embedding_b = fc.shared_embedding_columns(
          [categorical_column_a, categorical_column_b], dimension=2)  # v1
      all_cols = [
          price, some_embedding_column, shared_embedding_a, shared_embedding_b
      ]

      features = {
          'price': [[3.], [4.]],
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
      fc_old.linear_model(features, all_cols)
      bias = get_linear_model_bias()

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([0.], self.evaluate(bias))

  def test_linear_model_v1_v2_mix(self):
    # [FUNCTION DOC]
    # Another mix of columns marked as version one and two feeding the legacy shared embedding
    # columns; asserts construction succeeds and the bias is zero. The body is effectively the
    # same as the previous two tests; only the comments differ.
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      price = fc.numeric_column('price')  # v2
      some_sparse_column = fc.categorical_column_with_hash_bucket(
          'sparse_feature', hash_bucket_size=5)  # v1
      some_embedding_column = fc.embedding_column(
          some_sparse_column, dimension=10)  # v1
      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=3)  # v2
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=3)  # v2
      shared_embedding_a, shared_embedding_b = fc.shared_embedding_columns(
          [categorical_column_a, categorical_column_b], dimension=2)  # v1
      all_cols = [
          price, some_embedding_column, shared_embedding_a, shared_embedding_b
      ]

      features = {
          'price': [[3.], [4.]],
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
      fc_old.linear_model(features, all_cols)
      bias = get_linear_model_bias()

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([0.], self.evaluate(bias))

  def test_linear_model_v2_shared_embedding_all_other_v1(self):
    # [FUNCTION DOC]
    # Uses version-2 shared embedding columns with the legacy linear model and asserts a value
    # error saying shared embedding columns are not supported.
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      price = fc.numeric_column('price')  # v1
      some_sparse_column = fc.categorical_column_with_hash_bucket(
          'sparse_feature', hash_bucket_size=5)  # v1
      some_embedding_column = fc.embedding_column(
          some_sparse_column, dimension=10)  # v1
      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=3)  # v2
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=3)  # v2
      shared_embedding_a, shared_embedding_b = fc.shared_embedding_columns_v2(
          [categorical_column_a, categorical_column_b], dimension=2)  # v2
      all_cols = [
          price, some_embedding_column, shared_embedding_a, shared_embedding_b
      ]

      features = {
          'price': [[3.], [4.]],
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
      with self.assertRaisesRegex(ValueError,
                                  'SharedEmbeddingColumns are not supported'):
        fc_old.linear_model(features, all_cols)


class InputLayerTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the legacy InputLayer object used with version-2 columns: retrieving inputs,
  # variable reuse across calls, and gradients through an embedding column.

  @test_util.run_in_graph_and_eager_modes
  def test_retrieving_input(self):
    # [FUNCTION DOC]
    # Runs in both graph and eager modes. Builds an input layer over one numeric column, calls it
    # on a small feature mapping and asserts the output equals the feature value reshaped to a
    # column.
    features = {'a': [0.]}
    input_layer = fc_old.InputLayer(fc.numeric_column('a'))
    inputs = self.evaluate(input_layer(features))
    self.assertAllClose([[0.]], inputs)

  def test_reuses_variables(self):
    # [FUNCTION DOC]
    # Eager-mode scenario: embedding column over an identity categorical column with a fixed
    # initializer.
    # Calls the input layer, asserts the embedding output matches the initializer rows, that
    # exactly one variable was created, and that calling again on the same features creates no new
    # variable and returns the same variable object.
    with context.eager_mode():
      sparse_input = sparse_tensor.SparseTensor(
          indices=((0, 0), (1, 0), (2, 0)),
          values=(0, 1, 2),
          dense_shape=(3, 3))

      # Create feature columns (categorical and embedding).
      categorical_column = fc.categorical_column_with_identity(
          key='a', num_buckets=3)
      embedding_dimension = 2

      def _embedding_column_initializer(shape, dtype, partition_info=None):
        # [FUNCTION DOC]
        # HELPER: Local initializer for the embedding variable in the reuse test.
        # Ignores the shape, dtype and partition information and returns a fixed three by two
        # table of embedding values.
        del shape  # unused
        del dtype  # unused
        del partition_info  # unused
        embedding_values = (
            (1, 0),  # id 0
            (0, 1),  # id 1
            (1, 1))  # id 2
        return embedding_values

      embedding_column = fc.embedding_column(
          categorical_column,
          dimension=embedding_dimension,
          initializer=_embedding_column_initializer)

      input_layer = fc_old.InputLayer([embedding_column])
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
    # Eager-mode scenario: same setup as the reuse test, with a function that scales the layer
    # output by two.
    # Asserts the output values, then computes the implicit gradient and asserts the result is an
    # indexed-slices object with ids zero, one, two and every gradient element equal to two.
    with context.eager_mode():
      sparse_input = sparse_tensor.SparseTensor(
          indices=((0, 0), (1, 0), (2, 0)),
          values=(0, 1, 2),
          dense_shape=(3, 3))

      # Create feature columns (categorical and embedding).
      categorical_column = fc.categorical_column_with_identity(
          key='a', num_buckets=3)
      embedding_dimension = 2

      def _embedding_column_initializer(shape, dtype, partition_info=None):
        # [FUNCTION DOC]
        # HELPER: Local initializer for the embedding variable in the gradient test.
        # Ignores the shape, dtype and partition information and returns a fixed three by two
        # table of embedding values.
        del shape  # unused
        del dtype  # unused
        del partition_info  # unused
        embedding_values = (
            (1, 0),  # id 0
            (0, 1),  # id 1
            (1, 1))  # id 2
        return embedding_values

      embedding_column = fc.embedding_column(
          categorical_column,
          dimension=embedding_dimension,
          initializer=_embedding_column_initializer)

      input_layer = fc_old.InputLayer([embedding_column])
      features = {'a': sparse_input}

      def scale_matrix():
        # [FUNCTION DOC]
        # HELPER: Local function whose gradient the test computes.
        # Calls the input layer on the captured features and returns the result multiplied by two.
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
  # Test case for the legacy functional input_layer builder used with version-2 columns:
  # validation, ordering, dense columns of different shapes, variable mapping, partitioning, batch
  # size checks, repeated layers, one-dimensional and rank-zero inputs.

  def test_raises_if_empty_feature_columns(self):
    # [FUNCTION DOC]
    # Asserts that an empty column list raises a value error saying the columns must not be empty.
    with self.assertRaisesRegex(ValueError,
                                'feature_columns must not be empty'):
      fc_old.input_layer(features={}, feature_columns=[])

  def test_should_be_dense_column(self):
    # [FUNCTION DOC]
    # Asserts that passing a categorical column raises a value error saying a dense column is
    # required.
    with self.assertRaisesRegex(ValueError, 'must be a _DenseColumn'):
      fc_old.input_layer(
          features={'a': [[0]]},
          feature_columns=[
              fc.categorical_column_with_hash_bucket('wire_cast', 4)
          ])

  def test_does_not_support_dict_columns(self):
    # [FUNCTION DOC]
    # Asserts a mapping of columns raises a value error saying an iterable was expected.
    with self.assertRaisesRegex(
        ValueError, 'Expected feature_columns to be iterable, found dict.'):
      fc_old.input_layer(
          features={'a': [[0]]}, feature_columns={'a': fc.numeric_column('a')})

  def test_bare_column(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The features mapping is assigned twice in a chained assignment, which is
    # redundant but harmless.
    # Passes a single column (not in a list) to the input layer and asserts the result is the
    # feature value as a one-column matrix.
    with ops.Graph().as_default():
      features = features = {'a': [0.]}
      net = fc_old.input_layer(features, fc.numeric_column('a'))

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([[0.]], self.evaluate(net))

  def test_column_generator(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The features mapping is assigned twice in a chained assignment, which is
    # redundant but harmless.
    # Passes a generator of numeric columns and asserts the input layer consumes it and
    # concatenates the features into one row.
    with ops.Graph().as_default():
      features = features = {'a': [0.], 'b': [1.]}
      columns = (fc.numeric_column(key) for key in features)
      net = fc_old.input_layer(features, columns)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([[0., 1.]], self.evaluate(net))

  def test_raises_if_duplicate_name(self):
    # [FUNCTION DOC]
    # Asserts duplicate column names raise a value error about duplicate names.
    with self.assertRaisesRegex(
        ValueError, 'Duplicate feature column name found for columns'):
      fc_old.input_layer(
          features={'a': [[0]]},
          feature_columns=[fc.numeric_column('a'),
                           fc.numeric_column('a')])

  def test_one_column(self):
    # [FUNCTION DOC]
    # Single numeric column: asserts the output equals the input values.
    price = fc.numeric_column('price')
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      net = fc_old.input_layer(features, [price])

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([[1.], [5.]], self.evaluate(net))

  def test_multi_dimension(self):
    # [FUNCTION DOC]
    # Numeric column of shape two: asserts the output equals the two-wide input.
    price = fc.numeric_column('price', shape=2)
    with ops.Graph().as_default():
      features = {'price': [[1., 2.], [5., 6.]]}
      net = fc_old.input_layer(features, [price])

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([[1., 2.], [5., 6.]], self.evaluate(net))

  def test_raises_if_shape_mismatch(self):
    # [FUNCTION DOC]
    # Asserts one-element rows for a width-two column raise an error saying the tensor cannot be
    # reshaped to shape two by two.
    price = fc.numeric_column('price', shape=2)
    with ops.Graph().as_default():
      features = {'price': [[1.], [5.]]}
      with self.assertRaisesRegex(
          Exception,
          r'Cannot reshape a tensor with 2 elements to shape \[2,2\]'):
        fc_old.input_layer(features, [price])

  def test_reshaping(self):
    # [FUNCTION DOC]
    # Numeric column of shape one by two with three-dimensional input: asserts the output is
    # flattened to two columns.
    price = fc.numeric_column('price', shape=[1, 2])
    with ops.Graph().as_default():
      features = {'price': [[[1., 2.]], [[5., 6.]]]}
      net = fc_old.input_layer(features, [price])

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([[1., 2.], [5., 6.]], self.evaluate(net))

  def test_multi_column(self):
    # [FUNCTION DOC]
    # Two numeric columns: asserts the output concatenates the columns in order into rows of three
    # values.
    price1 = fc.numeric_column('price1', shape=2)
    price2 = fc.numeric_column('price2')
    with ops.Graph().as_default():
      features = {'price1': [[1., 2.], [5., 6.]], 'price2': [[3.], [4.]]}
      net = fc_old.input_layer(features, [price1, price2])

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([[1., 2., 3.], [5., 6., 4.]], self.evaluate(net))

  def test_fills_cols_to_vars(self):
    # [FUNCTION DOC]
    # Uses numeric, bucketized and embedding columns with a column-to-variables output mapping;
    # asserts all three are keys, only the embedding column has a variable, and that variable has
    # the expected shape and variable type.
    # Provide three _DenseColumn's to input_layer: a _NumericColumn, a
    # _BucketizedColumn, and an _EmbeddingColumn.  Only the _EmbeddingColumn
    # creates a Variable.
    price1 = fc.numeric_column('price1')
    dense_feature = fc.numeric_column('dense_feature')
    dense_feature_bucketized = fc.bucketized_column(
        dense_feature, boundaries=[0.])
    some_sparse_column = fc.categorical_column_with_hash_bucket(
        'sparse_feature', hash_bucket_size=5)
    some_embedding_column = fc.embedding_column(
        some_sparse_column, dimension=10)
    with ops.Graph().as_default():
      features = {
          'price1': [[3.], [4.]],
          'dense_feature': [[-1.], [4.]],
          'sparse_feature': [['a'], ['x']],
      }
      cols_to_vars = {}
      all_cols = [price1, dense_feature_bucketized, some_embedding_column]
      fc_old.input_layer(features, all_cols, cols_to_vars=cols_to_vars)
      self.assertCountEqual(list(cols_to_vars.keys()), all_cols)
      self.assertEqual(0, len(cols_to_vars[price1]))
      self.assertEqual(0, len(cols_to_vars[dense_feature_bucketized]))
      self.assertEqual(1, len(cols_to_vars[some_embedding_column]))
      self.assertIsInstance(cols_to_vars[some_embedding_column][0],
                            variables_lib.Variable)
      self.assertAllEqual(cols_to_vars[some_embedding_column][0].shape, [5, 10])

  def test_fills_cols_to_vars_shared_embedding(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The test documents a known defect in the current implementation: for two
    # columns sharing one embedding, only the first shared column lists the variable and the
    # second lists none (the code comment says this should be fixed in the new implementation).
    # Graph-only scenario with a numeric, bucketized, embedding and two legacy shared embedding
    # columns. Asserts the mapping keys, the variable counts per column and the shapes of the
    # embedding and shared embedding variables.
    # Provide 5 DenseColumn's to input_layer: a NumericColumn, a
    # BucketizedColumn, an EmbeddingColumn, two SharedEmbeddingColumns. The
    # EmbeddingColumn creates a Variable and the two SharedEmbeddingColumns
    # shared one variable.
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      price1 = fc.numeric_column('price1')
      dense_feature = fc.numeric_column('dense_feature')
      dense_feature_bucketized = fc.bucketized_column(
          dense_feature, boundaries=[0.])
      some_sparse_column = fc.categorical_column_with_hash_bucket(
          'sparse_feature', hash_bucket_size=5)
      some_embedding_column = fc.embedding_column(
          some_sparse_column, dimension=10)
      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=3)
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=3)
      shared_embedding_a, shared_embedding_b = fc.shared_embedding_columns(
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
      fc_old.input_layer(features, all_cols, cols_to_vars=cols_to_vars)
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
    # Builds the input layer in a variable scope with a three-way partitioner and asserts the
    # embedding column maps to three shards with the expected names and shapes while the other
    # columns map to none.
    price1 = fc.numeric_column('price1')
    dense_feature = fc.numeric_column('dense_feature')
    dense_feature_bucketized = fc.bucketized_column(
        dense_feature, boundaries=[0.])
    some_sparse_column = fc.categorical_column_with_hash_bucket(
        'sparse_feature', hash_bucket_size=5)
    some_embedding_column = fc.embedding_column(
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
        fc_old.input_layer(features, all_cols, cols_to_vars=cols_to_vars)
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
    # Builds two layers with columns in different orders and asserts both outputs are equal,
    # showing columns are ordered by name and not by argument order.
    price_a = fc.numeric_column('price_a')
    price_b = fc.numeric_column('price_b')
    with ops.Graph().as_default():
      features = {
          'price_a': [[1.]],
          'price_b': [[3.]],
      }
      net1 = fc_old.input_layer(features, [price_a, price_b])
      net2 = fc_old.input_layer(features, [price_b, price_a])

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([[1., 3.]], self.evaluate(net1))
      self.assertAllClose([[1., 3.]], self.evaluate(net2))

  def test_fails_for_categorical_column(self):
    # [FUNCTION DOC]
    # Asserts that giving a categorical column to the input layer fails with an error saying a
    # dense column is required.
    animal = fc.categorical_column_with_identity('animal', num_buckets=4)
    with ops.Graph().as_default():
      features = {
          'animal':
              sparse_tensor.SparseTensor(
                  indices=[[0, 0], [0, 1]], values=[1, 2], dense_shape=[1, 2])
      }
      with self.assertRaisesRegex(Exception, 'must be a _DenseColumn'):
        fc_old.input_layer(features, [animal])

  def test_static_batch_size_mismatch(self):
    # [FUNCTION DOC]
    # Asserts features with static batch sizes three and two raise a value error saying all
    # features must share the batch size.
    price1 = fc.numeric_column('price1')
    price2 = fc.numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': [[1.], [5.], [7.]],  # batchsize = 3
          'price2': [[3.], [4.]]  # batchsize = 2
      }
      with self.assertRaisesRegex(
          ValueError,
          r'Batch size \(first dimension\) of each feature must be same.'):  # pylint: disable=anomalous-backslash-in-string
        fc_old.input_layer(features, [price1, price2])

  def test_subset_of_static_batch_size_mismatch(self):
    # [FUNCTION DOC]
    # Mixture of an unknown-shape placeholder and static features of sizes two and three; asserts
    # the same batch size value error at build time.
    price1 = fc.numeric_column('price1')
    price2 = fc.numeric_column('price2')
    price3 = fc.numeric_column('price3')
    with ops.Graph().as_default():
      features = {
          'price1': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 3
          'price2': [[3.], [4.]],  # batchsize = 2
          'price3': [[3.], [4.], [5.]]  # batchsize = 3
      }
      with self.assertRaisesRegex(
          ValueError,
          r'Batch size \(first dimension\) of each feature must be same.'):  # pylint: disable=anomalous-backslash-in-string
        fc_old.input_layer(features, [price1, price2, price3])

  def test_runtime_batch_size_mismatch(self):
    # [FUNCTION DOC]
    # Runs the layer feeding a placeholder with batch three against a static batch of two and
    # asserts an op error saying dimension zero of both shapes must be equal.
    price1 = fc.numeric_column('price1')
    price2 = fc.numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 3
          'price2': [[3.], [4.]]  # batchsize = 2
      }
      net = fc_old.input_layer(features, [price1, price2])
      with _initialized_session() as sess:
        with self.assertRaisesRegex(errors.OpError,
                                    'Dimension 0 in both shapes must be equal'):
          sess.run(net, feed_dict={features['price1']: [[1.], [5.], [7.]]})

  def test_runtime_batch_size_matches(self):
    # [FUNCTION DOC]
    # Feeds matching batches to two placeholders and asserts the run succeeds.
    price1 = fc.numeric_column('price1')
    price2 = fc.numeric_column('price2')
    with ops.Graph().as_default():
      features = {
          'price1': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 2
          'price2': array_ops.placeholder(dtype=dtypes.int64),  # batchsize = 2
      }
      net = fc_old.input_layer(features, [price1, price2])
      with _initialized_session() as sess:
        sess.run(
            net,
            feed_dict={
                features['price1']: [[1.], [5.]],
                features['price2']: [[1.], [5.]],
            })

  def test_multiple_layers_with_same_embedding_column(self):
    # [FUNCTION DOC]
    # Builds the input layer twice on the same embedding column in one graph and asserts two
    # separate embedding variables exist with names under input_layer and input_layer_1.
    some_sparse_column = fc.categorical_column_with_hash_bucket(
        'sparse_feature', hash_bucket_size=5)
    some_embedding_column = fc.embedding_column(
        some_sparse_column, dimension=10)

    with ops.Graph().as_default():
      features = {
          'sparse_feature': [['a'], ['x']],
      }
      all_cols = [some_embedding_column]
      fc_old.input_layer(features, all_cols)
      fc_old.input_layer(features, all_cols)
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

  def test_with_1d_sparse_tensor(self):
    # [FUNCTION DOC]
    # Scenario: numeric price, indicator body style and embedded country columns, all with
    # rank-one inputs (sparse for body style, dense strings for country).
    # Only runs the numeric assertions when executing eagerly (variables are only initialized
    # there): asserts the output width and the exact concatenated rows of indicator, embedding and
    # price values.
    embedding_values = (
        (1., 2., 3., 4., 5.),  # id 0
        (6., 7., 8., 9., 10.),  # id 1
        (11., 12., 13., 14., 15.)  # id 2
    )

    def _initializer(shape, dtype, partition_info=None):
      # [FUNCTION DOC]
      # HELPER: Local embedding initializer used by the test.
      # Ignores shape, dtype and partition information and returns a fixed three by five table of
      # embedding values.
      del shape, dtype, partition_info
      return embedding_values

    # price has 1 dimension in input_layer
    price = fc.numeric_column('price')

    # one_hot_body_style has 3 dims in input_layer.
    body_style = fc.categorical_column_with_vocabulary_list(
        'body-style', vocabulary_list=['hardtop', 'wagon', 'sedan'])
    one_hot_body_style = fc.indicator_column(body_style)

    # embedded_body_style has 5 dims in input_layer.
    country = fc.categorical_column_with_vocabulary_list(
        'country', vocabulary_list=['US', 'JP', 'CA'])
    embedded_country = fc.embedding_column(
        country, dimension=5, initializer=_initializer)

    # Provides 1-dim tensor and dense tensor.
    features = {
        'price':
            constant_op.constant([
                11.,
                12.,
            ]),
        'body-style':
            sparse_tensor.SparseTensor(
                indices=((0,), (1,)),
                values=('sedan', 'hardtop'),
                dense_shape=(2,)),
        # This is dense tensor for the categorical_column.
        'country':
            constant_op.constant(['CA', 'US']),
    }
    self.assertEqual(1, features['price'].shape.ndims)
    self.assertEqual(1, features['body-style'].dense_shape.get_shape()[0])
    self.assertEqual(1, features['country'].shape.ndims)

    if context.executing_eagerly():
      # The variables will only be initialized in TF2
      net = fc_old.input_layer(features,
                               [price, one_hot_body_style, embedded_country])
      self.assertEqual(1 + 3 + 5, net.shape[1])

      # Each row is formed by concatenating `embedded_body_style`,
      # `one_hot_body_style`, and `price` in order.
      self.assertAllEqual([[0., 0., 1., 11., 12., 13., 14., 15., 11.],
                           [1., 0., 0., 1., 2., 3., 4., 5., 12.]],
                          self.evaluate(net))

  def test_with_1d_unknown_shape_sparse_tensor(self):
    # [FUNCTION DOC]
    # Graph-only scenario using unknown-shape placeholders for price, body style and country, an
    # indicator column and a two-wide embedding. Feeds data and asserts the output width and exact
    # concatenated rows.
    # This test needs to construct graph placeholders
    # w/ unknown shapes, so we enter a graph
    with ops.Graph().as_default():
      embedding_values = (
          (1., 2.),  # id 0
          (6., 7.),  # id 1
          (11., 12.)  # id 2
      )

      def _initializer(shape, dtype, partition_info=None):
        # [FUNCTION DOC]
        # HELPER: Local embedding initializer used by the test.
        # Ignores shape, dtype and partition information and returns a fixed three by two table of
        # embedding values.
        del shape, dtype, partition_info
        return embedding_values

      # price has 1 dimension in input_layer
      price = fc.numeric_column('price')

      # one_hot_body_style has 3 dims in input_layer.
      body_style = fc.categorical_column_with_vocabulary_list(
          'body-style', vocabulary_list=['hardtop', 'wagon', 'sedan'])
      one_hot_body_style = fc.indicator_column(body_style)

      # embedded_body_style has 5 dims in input_layer.
      country = fc.categorical_column_with_vocabulary_list(
          'country', vocabulary_list=['US', 'JP', 'CA'])
      embedded_country = fc.embedding_column(
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
          indices=((0,), (1,)), values=('sedan', 'hardtop'), dense_shape=(2,))
      country_data = np.array([['US'], ['CA']])

      net = fc_old.input_layer(features,
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

  def test_with_rank_0_feature(self):
    # [FUNCTION DOC]
    # Asserts a statically rank-zero feature raises a value error saying it cannot have rank zero,
    # and a dynamically rank-zero placeholder raises an op error with the same message when the
    # graph is run.
    # price has 1 dimension in input_layer
    price = fc.numeric_column('price')
    features = {
        'price': constant_op.constant(0),
    }
    self.assertEqual(0, features['price'].shape.ndims)

    # Static rank 0 should fail
    with self.assertRaisesRegex(ValueError, 'Feature .* cannot have rank 0'):
      fc_old.input_layer(features, [price])

    # This test needs to construct graph placeholders
    # w/ dynamic rank 0, so we enter a graph
    with ops.Graph().as_default():
      # Dynamic rank 0 should fail
      features = {
          'price': array_ops.placeholder(dtypes.float32),
      }
      net = fc_old.input_layer(features, [price])
      self.assertEqual(1, net.shape[1])
      with _initialized_session() as sess:
        with self.assertRaisesOpError('Feature .* cannot have rank 0'):
          sess.run(net, feed_dict={features['price']: np.array(1)})


class MakeParseExampleSpecTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the function that merges parse specs from a list of feature columns. Uses a
  # nested namedtuple-based test column whose parse spec is supplied at construction.

  class _TestFeatureColumn(BaseFeatureColumnForTests,
                           collections.namedtuple('_TestFeatureColumn',
                                                  ('parse_spec'))):
    # [CLASS DOC]
    # HELPER: Local test column holding a caller-supplied parse spec.
    # Nested class combining the dummy base column with a one-field named tuple whose single field
    # is the parse spec. It is flagged version 2, named _TestFeatureColumn, and its parse example
    # spec simply returns the stored field.

    @property
    def _is_v2_column(self):
      # [FUNCTION DOC]
      # HELPER: Marker property on a local test column.
      # Property returning true so the column is treated as a version-2 feature column.
      return True

    @property
    def name(self):
      # [FUNCTION DOC]
      # HELPER: Name property of a local test column.
      # Property returning the fixed string _TestFeatureColumn.
      return '_TestFeatureColumn'

    def transform_feature(self, transformation_cache, state_manager):
      # [FUNCTION DOC]
      # HELPER: Version-2 transform hook of a local test column.
      # Does nothing and returns the null value.
      pass

    def _transform_feature(self, inputs):
      # [FUNCTION DOC]
      # HELPER: Legacy transform hook of a local test column.
      # Does nothing and returns the null value.
      pass

    @property
    def parse_example_spec(self):
      # [FUNCTION DOC]
      # HELPER: Parse spec property of a local test column.
      # Property returning the parse spec mapping stored in the tuple field.
      return self.parse_spec

    @property
    def _parse_example_spec(self):
      # [FUNCTION DOC]
      # HELPER: Legacy parse spec property of a local test column.
      # Property returning the same stored parse spec mapping.
      return self.parse_spec

  def test_no_feature_columns(self):
    # [FUNCTION DOC]
    # Asserts the merged spec of an empty column list is an empty mapping.
    actual = fc.make_parse_example_spec_v2([])
    self.assertDictEqual({}, actual)

  def test_invalid_type(self):
    # [FUNCTION DOC]
    # Asserts that a list containing a string instead of a feature column raises a value error
    # saying all items must be feature columns and naming the invalid one.
    key1 = 'key1'
    parse_spec1 = parsing_ops.FixedLenFeature(
        shape=(2,), dtype=dtypes.float32, default_value=0.)
    with self.assertRaisesRegex(
        ValueError,
        'All feature_columns must be FeatureColumn instances.*invalid_column'):
      fc.make_parse_example_spec_v2((self._TestFeatureColumn({
          key1: parse_spec1
      }), 'invalid_column'))

  def test_one_feature_column(self):
    # [FUNCTION DOC]
    # Asserts the merged spec of one column equals that column's spec.
    key1 = 'key1'
    parse_spec1 = parsing_ops.FixedLenFeature(
        shape=(2,), dtype=dtypes.float32, default_value=0.)
    actual = fc.make_parse_example_spec_v2((self._TestFeatureColumn({
        key1: parse_spec1
    }),))
    self.assertDictEqual({key1: parse_spec1}, actual)

  def test_two_feature_columns(self):
    # [FUNCTION DOC]
    # Asserts the merged spec of two columns with different keys contains both entries.
    key1 = 'key1'
    parse_spec1 = parsing_ops.FixedLenFeature(
        shape=(2,), dtype=dtypes.float32, default_value=0.)
    key2 = 'key2'
    parse_spec2 = parsing_ops.VarLenFeature(dtype=dtypes.string)
    actual = fc.make_parse_example_spec_v2((self._TestFeatureColumn({
        key1: parse_spec1
    }), self._TestFeatureColumn({
        key2: parse_spec2
    })))
    self.assertDictEqual({key1: parse_spec1, key2: parse_spec2}, actual)

  def test_equal_keys_different_parse_spec(self):
    # [FUNCTION DOC]
    # Asserts two columns using the same key with different specs raise a value error saying the
    # columns contain different parse specs for that key.
    key1 = 'key1'
    parse_spec1 = parsing_ops.FixedLenFeature(
        shape=(2,), dtype=dtypes.float32, default_value=0.)
    parse_spec2 = parsing_ops.VarLenFeature(dtype=dtypes.string)
    with self.assertRaisesRegex(
        ValueError,
        'feature_columns contain different parse_spec for key key1'):
      fc.make_parse_example_spec_v2((self._TestFeatureColumn({
          key1: parse_spec1
      }), self._TestFeatureColumn({
          key1: parse_spec2
      })))

  def test_equal_keys_equal_parse_spec(self):
    # [FUNCTION DOC]
    # Asserts two columns using the same key with identical specs merge into a single entry
    # without error.
    key1 = 'key1'
    parse_spec1 = parsing_ops.FixedLenFeature(
        shape=(2,), dtype=dtypes.float32, default_value=0.)
    actual = fc.make_parse_example_spec_v2((self._TestFeatureColumn({
        key1: parse_spec1
    }), self._TestFeatureColumn({
        key1: parse_spec1
    })))
    self.assertDictEqual({key1: parse_spec1}, actual)

  def test_multiple_features_dict(self):
    # [FUNCTION DOC]
    # One column contributes a spec with two keys and another contributes one; asserts the merged
    # mapping has all three entries. The docstring has a typo (parse_spc) but describes this
    # scenario.
    """parse_spc for one column is a dict with length > 1."""
    key1 = 'key1'
    parse_spec1 = parsing_ops.FixedLenFeature(
        shape=(2,), dtype=dtypes.float32, default_value=0.)
    key2 = 'key2'
    parse_spec2 = parsing_ops.VarLenFeature(dtype=dtypes.string)
    key3 = 'key3'
    parse_spec3 = parsing_ops.VarLenFeature(dtype=dtypes.int32)
    actual = fc.make_parse_example_spec_v2((self._TestFeatureColumn({
        key1: parse_spec1
    }), self._TestFeatureColumn({
        key2: parse_spec2,
        key3: parse_spec3
    })))
    self.assertDictEqual({
        key1: parse_spec1,
        key2: parse_spec2,
        key3: parse_spec3
    }, actual)


def _assert_sparse_tensor_value(test_case, expected, actual):
  # [FUNCTION DOC]
  # HELPER: Shared test utility for comparing sparse tensor values with exact dtype checking.
  # Takes a test case, an expected sparse tensor value and an actual one. Asserts the actual
  # indices and dense shape are 64-bit integer arrays, that the indices and dense shapes equal the
  # expected ones, and that the value dtypes and contents match exactly.
  # Raises assertion failures through the test case; returns nothing.
  test_case.assertEqual(np.int64, np.array(actual.indices).dtype)
  test_case.assertAllEqual(expected.indices, actual.indices)

  test_case.assertEqual(
      np.array(expected.values).dtype,
      np.array(actual.values).dtype)
  test_case.assertAllEqual(expected.values, actual.values)

  test_case.assertEqual(np.int64, np.array(actual.dense_shape).dtype)
  test_case.assertAllEqual(expected.dense_shape, actual.dense_shape)


class VocabularyFileCategoricalColumnTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the vocabulary-file categorical column in plain text format. Class attributes
  # hold the file format (none) and the expected error for an oversized vocabulary size; a
  # subclass overrides them for gzip tfrecord files. Covers defaults, validation, deep copy,
  # parsing, lookups with out-of-vocabulary buckets and default values, integer vocabularies,
  # legacy linear model use and serialization.

  _FILE_FORMAT = None
  _VOCABULARY_SIZE_ERROR = (errors.OpError, 'Invalid vocab_size')

  def setUp(self):
    # [FUNCTION DOC]
    # HELPER: Fixture setup for the vocabulary-file tests.
    # Calls the parent setup, then stores the paths to the test data vocabulary files (warriors
    # jersey numbers, wire character names and a unicode vocabulary) and the sizes of the first
    # two (five and three) on the instance. Side effect: resolves paths under the test source
    # directory.
    super(VocabularyFileCategoricalColumnTest, self).setUp()

    # Contains ints, Golden State Warriors jersey numbers: 30, 35, 11, 23, 22
    self._warriors_vocabulary_file_name = test.test_src_dir_path(
        'python/feature_column/testdata/warriors_vocabulary.txt')
    self._warriors_vocabulary_size = 5

    # Contains strings, character names from 'The Wire': omar, stringer, marlo
    self._wire_vocabulary_file_name = test.test_src_dir_path(
        'python/feature_column/testdata/wire_vocabulary.txt')
    self._wire_vocabulary_size = 3

    # Contains unicode characters.
    self._unicode_vocabulary_file_name = test.test_src_dir_path(
        'python/feature_column/testdata/unicode_vocabulary')

  def test_defaults(self):
    # [FUNCTION DOC]
    # Creates a vocabulary file column with a file path and size three and asserts name, key,
    # bucket count, string parse spec and version flag.
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa', vocabulary_file='path_to_file', vocabulary_size=3)
    self.assertEqual('aaa', column.name)
    self.assertEqual('aaa', column.key)
    self.assertEqual(3, column.num_buckets)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.string)
    }, column.parse_example_spec)
    self.assertTrue(column._is_v2_column)

  def test_defaults_unicode(self):
    # [FUNCTION DOC]
    # Creates a column without a size from the unicode vocabulary and asserts the bucket count is
    # one hundred sixty-five (evaluated if it is a tensor), the parse spec and the version flag.
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa', vocabulary_file=self._unicode_vocabulary_file_name)
    self.assertEqual('aaa', column.name)
    self.assertEqual('aaa', column.key)
    if isinstance(column.num_buckets, (int, np.integer)):
      self.assertEqual(165, column.num_buckets)
    else:
      self.assertEqual(165, self.evaluate(column.num_buckets))
    self.assertEqual({'aaa': parsing_ops.VarLenFeature(dtypes.string)},
                     column.parse_example_spec)
    self.assertTrue(column._is_v2_column)

  def test_key_should_be_string(self):
    # [FUNCTION DOC]
    # Asserts a tuple key raises a value error saying the key must be a string.
    with self.assertRaisesRegex(ValueError, 'key must be a string.'):
      fc.categorical_column_with_vocabulary_file(
          key=('aaa',), vocabulary_file='path_to_file', vocabulary_size=3)

  def test_all_constructor_args(self):
    # [FUNCTION DOC]
    # Creates a column with extra out-of-vocabulary buckets and int32 dtype; asserts the total
    # bucket count is vocabulary size plus buckets and the parse spec has int32 dtype.
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file='path_to_file',
        vocabulary_size=3,
        num_oov_buckets=4,
        dtype=dtypes.int32)
    self.assertEqual(7, column.num_buckets)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int32)
    }, column.parse_example_spec)

  def test_deep_copy(self):
    # [FUNCTION DOC]
    # Asserts an original and its deep copy both report the same name, bucket count and int32
    # parse spec.
    original = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file='path_to_file',
        vocabulary_size=3,
        num_oov_buckets=4,
        dtype=dtypes.int32)
    for column in (original, copy.deepcopy(original)):
      self.assertEqual('aaa', column.name)
      self.assertEqual(7, column.num_buckets)
      self.assertEqual({
          'aaa': parsing_ops.VarLenFeature(dtypes.int32)
      }, column.parse_example_spec)

  def test_vocabulary_file_none(self):
    # [FUNCTION DOC]
    # Asserts a missing vocabulary file raises a value error saying the file is missing.
    with self.assertRaisesRegex(ValueError, 'Missing vocabulary_file'):
      fc.categorical_column_with_vocabulary_file(
          key='aaa', vocabulary_file=None, vocabulary_size=3)

  def test_vocabulary_file_empty_string(self):
    # [FUNCTION DOC]
    # Asserts an empty file path raises the same missing-file value error.
    with self.assertRaisesRegex(ValueError, 'Missing vocabulary_file'):
      fc.categorical_column_with_vocabulary_file(
          key='aaa', vocabulary_file='', vocabulary_size=3)

  def test_invalid_vocabulary_file(self):
    # [FUNCTION DOC]
    # Asserts that looking up ids with a nonexistent file raises an op error naming the file.
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa', vocabulary_file='file_does_not_exist', vocabulary_size=10)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('marlo', 'skywalker', 'omar'),
        dense_shape=(2, 2))
    with self.assertRaisesRegex(errors.OpError, 'file_does_not_exist'):
      column.get_sparse_tensors(
          fc.FeatureTransformationCache({
              'aaa': inputs
          }), None)
      self.evaluate(lookup_ops.tables_initializer())

  def test_invalid_vocabulary_size(self):
    # [FUNCTION DOC]
    # Asserts negative and zero vocabulary sizes raise a value error about an invalid size.
    with self.assertRaisesRegex(ValueError, 'Invalid vocabulary_size'):
      fc.categorical_column_with_vocabulary_file(
          key='aaa',
          vocabulary_file=self._wire_vocabulary_file_name,
          vocabulary_size=-1)
    with self.assertRaisesRegex(ValueError, 'Invalid vocabulary_size'):
      fc.categorical_column_with_vocabulary_file(
          key='aaa',
          vocabulary_file=self._wire_vocabulary_file_name,
          vocabulary_size=0)

  def test_too_large_vocabulary_size(self):
    # [FUNCTION DOC]
    # Creates a column whose declared size exceeds the real file length and asserts the lookup
    # raises the error stored in the class attribute (an invalid vocabulary size op error for text
    # files; a different error for gzip tfrecord files).
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size + 1)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('marlo', 'skywalker', 'omar'),
        dense_shape=(2, 2))
    with self.assertRaisesRegex(*self._VOCABULARY_SIZE_ERROR):
      column.get_sparse_tensors(
          fc.FeatureTransformationCache({
              'aaa': inputs
          }), None)
      self.evaluate(lookup_ops.tables_initializer())

  def test_invalid_num_oov_buckets(self):
    # [FUNCTION DOC]
    # Asserts a negative number of out-of-vocabulary buckets raises a value error.
    with self.assertRaisesRegex(ValueError, 'Invalid num_oov_buckets'):
      fc.categorical_column_with_vocabulary_file(
          key='aaa',
          vocabulary_file='path',
          vocabulary_size=3,
          num_oov_buckets=-1)

  def test_invalid_dtype(self):
    # [FUNCTION DOC]
    # Asserts float64 dtype raises a value error saying dtype must be string or integer.
    with self.assertRaisesRegex(ValueError, 'dtype must be string or integer'):
      fc.categorical_column_with_vocabulary_file(
          key='aaa',
          vocabulary_file='path',
          vocabulary_size=3,
          dtype=dtypes.float64)

  def test_invalid_buckets_and_default_value(self):
    # [FUNCTION DOC]
    # Asserts setting both a positive number of out-of-vocabulary buckets and a default value
    # raises a value error.
    with self.assertRaisesRegex(ValueError,
                                'both num_oov_buckets and default_value'):
      fc.categorical_column_with_vocabulary_file(
          key='aaa',
          vocabulary_file=self._wire_vocabulary_file_name,
          vocabulary_size=self._wire_vocabulary_size,
          num_oov_buckets=100,
          default_value=2)

  def test_invalid_input_dtype_int32(self):
    # [FUNCTION DOC]
    # Asserts a string column given integer input raises a value error saying the dtype must be
    # compatible.
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size,
        dtype=dtypes.string)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=(12, 24, 36),
        dense_shape=(2, 2))
    with self.assertRaisesRegex(ValueError, 'dtype must be compatible'):
      column.get_sparse_tensors(
          fc.FeatureTransformationCache({
              'aaa': inputs
          }), None)

  def test_invalid_input_dtype_string(self):
    # [FUNCTION DOC]
    # Asserts an integer column given string input raises a value error saying the dtype must be
    # compatible.
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._warriors_vocabulary_file_name,
        vocabulary_size=self._warriors_vocabulary_size,
        dtype=dtypes.int32)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('omar', 'stringer', 'marlo'),
        dense_shape=(2, 2))
    with self.assertRaisesRegex(ValueError, 'dtype must be compatible'):
      column.get_sparse_tensors(
          fc.FeatureTransformationCache({
              'aaa': inputs
          }), None)

  def test_parse_example(self):
    # [FUNCTION DOC]
    # Parses a serialized example with two byte strings using the column's spec and asserts the
    # sparse result via the comparison helper.
    a = fc.categorical_column_with_vocabulary_file(
        key='aaa', vocabulary_file='path_to_file', vocabulary_size=3)
    data = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'aaa':
                    feature_pb2.Feature(
                        bytes_list=feature_pb2.BytesList(
                            value=[b'omar', b'stringer']))
            }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec_v2([a]))
    self.assertIn('aaa', features)

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [0, 1]],
            values=np.array([b'omar', b'stringer'], dtype=np.object_),
            dense_shape=[1, 2]), self.evaluate(features['aaa']))

  def test_get_sparse_tensors(self):
    # [FUNCTION DOC]
    # Looks up string sparse input in the wire vocabulary; asserts no weights and ids two, minus
    # one for the unknown word, and zero for the known words.
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('marlo', 'skywalker', 'omar'),
        dense_shape=(2, 2))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': inputs
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((2, -1, 0), dtype=np.int64),
            dense_shape=inputs.dense_shape),
        self.evaluate(id_weight_pair.id_tensor))

  def test_get_sparse_tensors_none_vocabulary_size(self):
    # [FUNCTION DOC]
    # Same as above but the vocabulary size is not given (inferred from the file); asserts the
    # same ids.
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa', vocabulary_file=self._wire_vocabulary_file_name)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('marlo', 'skywalker', 'omar'),
        dense_shape=(2, 2))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': inputs
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((2, -1, 0), dtype=np.int64),
            dense_shape=inputs.dense_shape),
        self.evaluate(id_weight_pair.id_tensor))

  def test_transform_feature(self):
    # [FUNCTION DOC]
    # Transforms the same input through the version-2 transform helper and asserts the same id
    # tensor.
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('marlo', 'skywalker', 'omar'),
        dense_shape=(2, 2))
    id_tensor = fc._transform_features_v2({
        'aaa': inputs
    }, [column], None)[column]

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((2, -1, 0), dtype=np.int64),
            dense_shape=inputs.dense_shape), self.evaluate(id_tensor))

  def test_get_sparse_tensors_dense_input(self):
    # [FUNCTION DOC]
    # Same lookup with dense nested-tuple string input containing an empty string; asserts the ids
    # are returned as a sparse tensor skipping the empty entry.
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size)
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': (('marlo', ''), ('skywalker', 'omar'))
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=((0, 0), (1, 0), (1, 1)),
            values=np.array((2, -1, 0), dtype=np.int64),
            dense_shape=(2, 2)), self.evaluate(id_weight_pair.id_tensor))

  def test_get_sparse_tensors_default_value_in_vocabulary(self):
    # [FUNCTION DOC]
    # Column with a default value of two; asserts the unknown word maps to two instead of minus
    # one.
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size,
        default_value=2)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('marlo', 'skywalker', 'omar'),
        dense_shape=(2, 2))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': inputs
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((2, 2, 0), dtype=np.int64),
            dense_shape=inputs.dense_shape),
        self.evaluate(id_weight_pair.id_tensor))

  def test_get_sparse_tensors_with_oov_buckets(self):
    # [FUNCTION DOC]
    # Column with one hundred out-of-vocabulary buckets; asserts unknown words hash into bucket
    # ids above the vocabulary range (thirty-three and sixty-two for the two unknown words) and
    # known words keep their ids.
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size,
        num_oov_buckets=100)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1), (1, 2)),
        values=('marlo', 'skywalker', 'omar', 'heisenberg'),
        dense_shape=(2, 3))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': inputs
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((2, 33, 0, 62), dtype=np.int64),
            dense_shape=inputs.dense_shape),
        self.evaluate(id_weight_pair.id_tensor))

  def test_get_sparse_tensors_small_vocabulary_size(self):
    # [FUNCTION DOC]
    # Sets the declared size one below the file length so the last word is excluded; asserts that
    # word and the unknown word map to minus one while the first word maps to zero.
    # 'marlo' is the last entry in our vocabulary file, so be setting
    # `vocabulary_size` to 1 less than number of entries in file, we take
    # 'marlo' out of the vocabulary.
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size - 1)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('marlo', 'skywalker', 'omar'),
        dense_shape=(2, 2))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': inputs
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((-1, -1, 0), dtype=np.int64),
            dense_shape=inputs.dense_shape),
        self.evaluate(id_weight_pair.id_tensor))

  def test_get_sparse_tensors_int32(self):
    # [FUNCTION DOC]
    # Integer vocabulary of jersey numbers with int32 dtype; asserts known numbers map to their
    # positions and the unknown number to minus one.
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._warriors_vocabulary_file_name,
        vocabulary_size=self._warriors_vocabulary_size,
        dtype=dtypes.int32)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1), (2, 2)),
        values=(11, 100, 30, 22),
        dense_shape=(3, 3))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': inputs
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((2, -1, 0, 4), dtype=np.int64),
            dense_shape=inputs.dense_shape),
        self.evaluate(id_weight_pair.id_tensor))

  def test_get_sparse_tensors_int32_dense_input(self):
    # [FUNCTION DOC]
    # Dense integer input with a custom default value; asserts the ids use positions for known
    # numbers and the default for unknown ones, skipping the padding entries.
    default_value = -100
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._warriors_vocabulary_file_name,
        vocabulary_size=self._warriors_vocabulary_size,
        dtype=dtypes.int32,
        default_value=default_value)
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': ((11, -1, -1), (100, 30, -1), (-1, -1, 22))
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=((0, 0), (1, 0), (1, 1), (2, 2)),
            values=np.array((2, default_value, 0, 4), dtype=np.int64),
            dense_shape=(3, 3)), self.evaluate(id_weight_pair.id_tensor))

  def test_get_sparse_tensors_int32_with_oov_buckets(self):
    # [FUNCTION DOC]
    # Integer vocabulary with one hundred out-of-vocabulary buckets; asserts the unknown number
    # hashes to a bucket id above the vocabulary range.
    column = fc.categorical_column_with_vocabulary_file(
        key='aaa',
        vocabulary_file=self._warriors_vocabulary_file_name,
        vocabulary_size=self._warriors_vocabulary_size,
        dtype=dtypes.int32,
        num_oov_buckets=100)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1), (2, 2)),
        values=(11, 100, 30, 22),
        dense_shape=(3, 3))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': inputs
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((2, 60, 0, 4), dtype=np.int64),
            dense_shape=inputs.dense_shape),
        self.evaluate(id_weight_pair.id_tensor))

  def test_old_linear_model(self):
    # [FUNCTION DOC]
    # Legacy linear model over a column with one extra out-of-vocabulary bucket: asserts four zero
    # weights initially, then after assigning weights asserts predictions equal the sums for each
    # example's ids (three and five).
    wire_column = fc.categorical_column_with_vocabulary_file(
        key='wire',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size,
        num_oov_buckets=1)
    self.assertEqual(4, wire_column.num_buckets)
    with ops.Graph().as_default():
      predictions = fc_old.linear_model({
          wire_column.name:
              sparse_tensor.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=('marlo', 'skywalker', 'omar'),
                  dense_shape=(2, 2))
      }, (wire_column,))
      bias = get_linear_model_bias()
      wire_var = get_linear_model_column_var(wire_column)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose((0.,), self.evaluate(bias))
      self.assertAllClose(((0.,), (0.,), (0.,), (0.,)), self.evaluate(wire_var))
      self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
      self.evaluate(wire_var.assign(((1.,), (2.,), (3.,), (4.,))))
      # 'marlo' -> 2: wire_var[2] = 3
      # 'skywalker' -> 3, 'omar' -> 0: wire_var[3] + wire_var[0] = 4+1 = 5
      self.assertAllClose(((3.,), (5.,)), self.evaluate(predictions))

  def test_serialization(self):
    # [FUNCTION DOC]
    # Asserts the column's parents are its key, its configuration includes default value, dtype,
    # key, bucket count, file, size and the file format class attribute, and rebuilding from
    # configuration gives an equal column.
    wire_column = fc.categorical_column_with_vocabulary_file(
        key='wire',
        vocabulary_file=self._wire_vocabulary_file_name,
        vocabulary_size=self._wire_vocabulary_size,
        num_oov_buckets=1)

    self.assertEqual(['wire'], wire_column.parents)

    config = wire_column.get_config()
    self.assertEqual(
        {
            'default_value': -1,
            'dtype': 'string',
            'key': 'wire',
            'num_oov_buckets': 1,
            'vocabulary_file': self._wire_vocabulary_file_name,
            'vocabulary_size': 3,
            'file_format': self._FILE_FORMAT,
        }, config)

    self.assertEqual(wire_column,
                     fc.VocabularyFileCategoricalColumn.from_config(config))


class VocabularyTfrecordGzipFileCategoricalColumnTest(
    VocabularyFileCategoricalColumnTest):
  # [CLASS DOC]
  # Subclass of the vocabulary-file test case that reruns every inherited test with vocabularies
  # stored as gzip-compressed tfrecord files. Overrides the file format class attribute to the
  # tfrecord gzip value and the expected too-large-size error to a failed-precondition error about
  # the dataset element count.

  _FILE_FORMAT = 'tfrecord_gzip'
  _VOCABULARY_SIZE_ERROR = (errors.FailedPreconditionError,
                            'Input dataset was expected to contain 4 elements')

  def setUp(self):
    # [FUNCTION DOC]
    # HELPER: Fixture setup that redirects the vocabulary-file tests to tfrecord gzip data.
    # Calls the parent setup and then replaces the stored file paths with the gzip tfrecord
    # versions of the warriors, wire and unicode vocabulary files; sizes remain five and three.
    super(VocabularyTfrecordGzipFileCategoricalColumnTest, self).setUp()

    # Contains ints, Golden State Warriors jersey numbers: 30, 35, 11, 23, 22
    self._warriors_vocabulary_file_name = test.test_src_dir_path(
        'python/feature_column/testdata/warriors_vocabulary.tfrecord.gz')
    self._warriors_vocabulary_size = 5

    # Contains strings, character names from 'The Wire': omar, stringer, marlo
    self._wire_vocabulary_file_name = test.test_src_dir_path(
        'python/feature_column/testdata/wire_vocabulary.tfrecord.gz')
    self._wire_vocabulary_size = 3

    # Contains unicode characters.
    self._unicode_vocabulary_file_name = test.test_src_dir_path(
        'python/feature_column/testdata/unicode_vocabulary.tfrecord.gz')


class VocabularyListCategoricalColumnTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the in-memory vocabulary list categorical column: defaults for string and
  # integer lists, argument validation, deep copy, parsing for string and integer data, lookups
  # with default values and out-of-vocabulary buckets, int32 handling, legacy linear model and
  # serialization.

  def test_defaults_string(self):
    # [FUNCTION DOC]
    # Creates a column from a string list and asserts name, key, bucket count of three, string
    # parse spec and version flag.
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    self.assertEqual('aaa', column.name)
    self.assertEqual('aaa', column.key)
    self.assertEqual(3, column.num_buckets)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.string)
    }, column.parse_example_spec)
    self.assertTrue(column._is_v2_column)

  def test_key_should_be_string(self):
    # [FUNCTION DOC]
    # Asserts a tuple key raises a value error saying the key must be a string.
    with self.assertRaisesRegex(ValueError, 'key must be a string.'):
      fc.categorical_column_with_vocabulary_list(
          key=('aaa',), vocabulary_list=('omar', 'stringer', 'marlo'))

  def test_defaults_int(self):
    # [FUNCTION DOC]
    # Creates a column from an integer list and asserts the parse spec uses 64-bit integers.
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=(12, 24, 36))
    self.assertEqual('aaa', column.name)
    self.assertEqual('aaa', column.key)
    self.assertEqual(3, column.num_buckets)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int64)
    }, column.parse_example_spec)

  def test_all_constructor_args(self):
    # [FUNCTION DOC]
    # Creates a column with explicit int32 dtype and a default value; asserts the bucket count and
    # the int32 parse spec.
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=(12, 24, 36),
        dtype=dtypes.int32,
        default_value=-99)
    self.assertEqual(3, column.num_buckets)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int32)
    }, column.parse_example_spec)

  def test_deep_copy(self):
    # [FUNCTION DOC]
    # Asserts both the original and a deep copy report the same name, bucket count and int32 parse
    # spec.
    original = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=(12, 24, 36), dtype=dtypes.int32)
    for column in (original, copy.deepcopy(original)):
      self.assertEqual('aaa', column.name)
      self.assertEqual(3, column.num_buckets)
      self.assertEqual({
          'aaa': parsing_ops.VarLenFeature(dtypes.int32)
      }, column.parse_example_spec)

  def test_invalid_dtype(self):
    # [FUNCTION DOC]
    # Asserts a float dtype raises a value error saying dtype must be string or integer.
    with self.assertRaisesRegex(ValueError, 'dtype must be string or integer'):
      fc.categorical_column_with_vocabulary_list(
          key='aaa',
          vocabulary_list=('omar', 'stringer', 'marlo'),
          dtype=dtypes.float32)

  def test_invalid_mapping_dtype(self):
    # [FUNCTION DOC]
    # Asserts a vocabulary of floats raises a value error saying the vocabulary dtype must be
    # string or integer.
    with self.assertRaisesRegex(ValueError,
                                r'vocabulary dtype must be string or integer'):
      fc.categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=(12., 24., 36.))

  def test_mismatched_int_dtype(self):
    # [FUNCTION DOC]
    # Asserts a string vocabulary with an integer dtype raises a value error saying the dtypes do
    # not match.
    with self.assertRaisesRegex(ValueError,
                                r'dtype.*and vocabulary dtype.*do not match'):
      fc.categorical_column_with_vocabulary_list(
          key='aaa',
          vocabulary_list=('omar', 'stringer', 'marlo'),
          dtype=dtypes.int32)

  def test_mismatched_string_dtype(self):
    # [FUNCTION DOC]
    # Asserts an integer vocabulary with a string dtype raises the same mismatch error.
    with self.assertRaisesRegex(ValueError,
                                r'dtype.*and vocabulary dtype.*do not match'):
      fc.categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=(12, 24, 36), dtype=dtypes.string)

  def test_none_mapping(self):
    # [FUNCTION DOC]
    # Asserts a missing vocabulary list raises a value error saying it must be non-empty.
    with self.assertRaisesRegex(ValueError,
                                r'vocabulary_list.*must be non-empty'):
      fc.categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=None)

  def test_empty_mapping(self):
    # [FUNCTION DOC]
    # Asserts an empty vocabulary list raises the non-empty value error.
    with self.assertRaisesRegex(ValueError,
                                r'vocabulary_list.*must be non-empty'):
      fc.categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=tuple([]))

  def test_duplicate_mapping(self):
    # [FUNCTION DOC]
    # Asserts a vocabulary with repeated entries raises a value error about duplicate keys.
    with self.assertRaisesRegex(ValueError, 'Duplicate keys'):
      fc.categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=(12, 24, 12))

  def test_invalid_num_oov_buckets(self):
    # [FUNCTION DOC]
    # Asserts a negative bucket count raises a value error about invalid out-of-vocabulary
    # buckets.
    with self.assertRaisesRegex(ValueError, 'Invalid num_oov_buckets'):
      fc.categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=(12, 24, 36), num_oov_buckets=-1)

  def test_invalid_buckets_and_default_value(self):
    # [FUNCTION DOC]
    # Asserts using both positive out-of-vocabulary buckets and a default value raises a value
    # error.
    with self.assertRaisesRegex(ValueError,
                                'both num_oov_buckets and default_value'):
      fc.categorical_column_with_vocabulary_list(
          key='aaa',
          vocabulary_list=(12, 24, 36),
          num_oov_buckets=100,
          default_value=2)

  def test_invalid_input_dtype_int32(self):
    # [FUNCTION DOC]
    # Asserts integer input given to a string vocabulary column raises a value error saying the
    # dtype must be compatible.
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=(12, 24, 36),
        dense_shape=(2, 2))
    with self.assertRaisesRegex(ValueError, 'dtype must be compatible'):
      column.get_sparse_tensors(
          fc.FeatureTransformationCache({
              'aaa': inputs
          }), None)

  def test_invalid_input_dtype_string(self):
    # [FUNCTION DOC]
    # Asserts string input given to an integer vocabulary column raises the same compatibility
    # error.
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=(12, 24, 36))
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('omar', 'stringer', 'marlo'),
        dense_shape=(2, 2))
    with self.assertRaisesRegex(ValueError, 'dtype must be compatible'):
      column.get_sparse_tensors(
          fc.FeatureTransformationCache({
              'aaa': inputs
          }), None)

  def test_parse_example_string(self):
    # [FUNCTION DOC]
    # Parses a serialized example with two byte strings and asserts the sparse result with the
    # comparison helper.
    a = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    data = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'aaa':
                    feature_pb2.Feature(
                        bytes_list=feature_pb2.BytesList(
                            value=[b'omar', b'stringer']))
            }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec_v2([a]))
    self.assertIn('aaa', features)

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [0, 1]],
            values=np.array([b'omar', b'stringer'], dtype=np.object_),
            dense_shape=[1, 2]), self.evaluate(features['aaa']))

  def test_parse_example_int(self):
    # [FUNCTION DOC]
    # Parses a serialized example with two integers and asserts the sparse result.
    a = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=(11, 21, 31))
    data = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'aaa':
                    feature_pb2.Feature(
                        int64_list=feature_pb2.Int64List(value=[11, 21]))
            }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec_v2([a]))
    self.assertIn('aaa', features)

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [0, 1]], values=[11, 21], dense_shape=[1, 2]),
        self.evaluate(features['aaa']))

  def test_get_sparse_tensors(self):
    # [FUNCTION DOC]
    # Looks up string sparse input; asserts no weights and ids two, minus one for unknown and
    # zero.
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('marlo', 'skywalker', 'omar'),
        dense_shape=(2, 2))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': inputs
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((2, -1, 0), dtype=np.int64),
            dense_shape=inputs.dense_shape),
        self.evaluate(id_weight_pair.id_tensor))

  def test_transform_feature(self):
    # [FUNCTION DOC]
    # Same lookup via the version-2 transform helper; asserts the same id tensor.
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('marlo', 'skywalker', 'omar'),
        dense_shape=(2, 2))
    id_tensor = fc._transform_features_v2({
        'aaa': inputs
    }, [column], None)[column]

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((2, -1, 0), dtype=np.int64),
            dense_shape=inputs.dense_shape), self.evaluate(id_tensor))

  def test_get_sparse_tensors_dense_input(self):
    # [FUNCTION DOC]
    # Dense nested-tuple input with an empty string; asserts the ids come back as a sparse tensor
    # without the empty entry.
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': (('marlo', ''), ('skywalker', 'omar'))
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=((0, 0), (1, 0), (1, 1)),
            values=np.array((2, -1, 0), dtype=np.int64),
            dense_shape=(2, 2)), self.evaluate(id_weight_pair.id_tensor))

  def test_get_sparse_tensors_default_value_in_vocabulary(self):
    # [FUNCTION DOC]
    # With a default value of two, asserts the unknown word maps to two.
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=('omar', 'stringer', 'marlo'),
        default_value=2)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('marlo', 'skywalker', 'omar'),
        dense_shape=(2, 2))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': inputs
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((2, 2, 0), dtype=np.int64),
            dense_shape=inputs.dense_shape),
        self.evaluate(id_weight_pair.id_tensor))

  def test_get_sparse_tensors_with_oov_buckets(self):
    # [FUNCTION DOC]
    # With one hundred out-of-vocabulary buckets, asserts the unknown words hash to ids
    # thirty-three and sixty-two.
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=('omar', 'stringer', 'marlo'),
        num_oov_buckets=100)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1), (1, 2)),
        values=('marlo', 'skywalker', 'omar', 'heisenberg'),
        dense_shape=(2, 3))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': inputs
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((2, 33, 0, 62), dtype=np.int64),
            dense_shape=inputs.dense_shape),
        self.evaluate(id_weight_pair.id_tensor))

  def test_get_sparse_tensors_int32(self):
    # [FUNCTION DOC]
    # Integer vocabulary as an int32 array; asserts known numbers map to positions and unknown to
    # minus one.
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=np.array((30, 35, 11, 23, 22), dtype=np.int32),
        dtype=dtypes.int32)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1), (2, 2)),
        values=np.array((11, 100, 30, 22), dtype=np.int32),
        dense_shape=(3, 3))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': inputs
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((2, -1, 0, 4), dtype=np.int64),
            dense_shape=inputs.dense_shape),
        self.evaluate(id_weight_pair.id_tensor))

  def test_get_sparse_tensors_int32_dense_input(self):
    # [FUNCTION DOC]
    # Dense int32 input with a custom default; asserts ids for known numbers and the default for
    # unknown ones.
    default_value = -100
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=np.array((30, 35, 11, 23, 22), dtype=np.int32),
        dtype=dtypes.int32,
        default_value=default_value)
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa':
                np.array(((11, -1, -1), (100, 30, -1), (-1, -1, 22)),
                         dtype=np.int32)
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=((0, 0), (1, 0), (1, 1), (2, 2)),
            values=np.array((2, default_value, 0, 4), dtype=np.int64),
            dense_shape=(3, 3)), self.evaluate(id_weight_pair.id_tensor))

  def test_get_sparse_tensors_int32_with_oov_buckets(self):
    # [FUNCTION DOC]
    # With one hundred out-of-vocabulary buckets, asserts the unknown integer hashes to a bucket
    # above the vocabulary range.
    column = fc.categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=np.array((30, 35, 11, 23, 22), dtype=np.int32),
        dtype=dtypes.int32,
        num_oov_buckets=100)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1), (2, 2)),
        values=(11, 100, 30, 22),
        dense_shape=(3, 3))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': inputs
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((2, 60, 0, 4), dtype=np.int64),
            dense_shape=inputs.dense_shape),
        self.evaluate(id_weight_pair.id_tensor))

  def test_old_linear_model(self):
    # [FUNCTION DOC]
    # Legacy linear model with one extra bucket: asserts zero initialization, then predictions of
    # three and five after assigning weights.
    wire_column = fc.categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=('omar', 'stringer', 'marlo'),
        num_oov_buckets=1)
    self.assertEqual(4, wire_column.num_buckets)
    with ops.Graph().as_default():
      predictions = fc_old.linear_model({
          wire_column.name:
              sparse_tensor.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=('marlo', 'skywalker', 'omar'),
                  dense_shape=(2, 2))
      }, (wire_column,))
      bias = get_linear_model_bias()
      wire_var = get_linear_model_column_var(wire_column)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose((0.,), self.evaluate(bias))
      self.assertAllClose(((0.,), (0.,), (0.,), (0.,)), self.evaluate(wire_var))
      self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
      self.evaluate(wire_var.assign(((1.,), (2.,), (3.,), (4.,))))
      # 'marlo' -> 2: wire_var[2] = 3
      # 'skywalker' -> 3, 'omar' -> 0: wire_var[3] + wire_var[0] = 4+1 = 5
      self.assertAllClose(((3.,), (5.,)), self.evaluate(predictions))

  def test_serialization(self):
    # [FUNCTION DOC]
    # Asserts the parents, the configuration (default value, dtype, key, bucket count, vocabulary
    # list) and that rebuilding from the configuration gives an equal column.
    wire_column = fc.categorical_column_with_vocabulary_list(
        key='aaa',
        vocabulary_list=('omar', 'stringer', 'marlo'),
        num_oov_buckets=1)

    self.assertEqual(['aaa'], wire_column.parents)

    config = wire_column.get_config()
    self.assertEqual({
        'default_value': -1,
        'dtype': 'string',
        'key': 'aaa',
        'num_oov_buckets': 1,
        'vocabulary_list': ('omar', 'stringer', 'marlo')
    }, config)

    self.assertEqual(wire_column,
                     fc.VocabularyListCategoricalColumn.from_config(config))


class IdentityCategoricalColumnTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the identity categorical column: constructor and validation, deep copy, dtype
  # checks, parsing, transformation, handling of out-of-range ids with and without a default
  # value, graph placeholder inputs, legacy linear model and serialization.

  def test_constructor(self):
    # [FUNCTION DOC]
    # Creates an identity column with three buckets and asserts name, key, bucket count, 64-bit
    # integer parse spec and version flag.
    column = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
    self.assertEqual('aaa', column.name)
    self.assertEqual('aaa', column.key)
    self.assertEqual(3, column.num_buckets)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int64)
    }, column.parse_example_spec)
    self.assertTrue(column._is_v2_column)

  def test_key_should_be_string(self):
    # [FUNCTION DOC]
    # Asserts a tuple key raises a value error saying the key must be a string.
    with self.assertRaisesRegex(ValueError, 'key must be a string.'):
      fc.categorical_column_with_identity(key=('aaa',), num_buckets=3)

  def test_deep_copy(self):
    # [FUNCTION DOC]
    # Asserts the original and its deep copy have the same name, bucket count and parse spec.
    original = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
    for column in (original, copy.deepcopy(original)):
      self.assertEqual('aaa', column.name)
      self.assertEqual(3, column.num_buckets)
      self.assertEqual({
          'aaa': parsing_ops.VarLenFeature(dtypes.int64)
      }, column.parse_example_spec)

  def test_invalid_num_buckets_zero(self):
    # [FUNCTION DOC]
    # Asserts zero buckets raises a value error saying the bucket count is less than one.
    with self.assertRaisesRegex(ValueError, 'num_buckets 0 < 1'):
      fc.categorical_column_with_identity(key='aaa', num_buckets=0)

  def test_invalid_num_buckets_negative(self):
    # [FUNCTION DOC]
    # Asserts a negative bucket count raises the same style of value error.
    with self.assertRaisesRegex(ValueError, 'num_buckets -1 < 1'):
      fc.categorical_column_with_identity(key='aaa', num_buckets=-1)

  def test_invalid_default_value_too_small(self):
    # [FUNCTION DOC]
    # Asserts a default value of minus one raises a value error saying it is not in range.
    with self.assertRaisesRegex(ValueError, 'default_value -1 not in range'):
      fc.categorical_column_with_identity(
          key='aaa', num_buckets=3, default_value=-1)

  def test_invalid_default_value_too_big(self):
    # [FUNCTION DOC]
    # Asserts a default value equal to the number of buckets raises a value error saying it is not
    # in range.
    with self.assertRaisesRegex(ValueError, 'default_value 3 not in range'):
      fc.categorical_column_with_identity(
          key='aaa', num_buckets=3, default_value=3)

  def test_invalid_input_dtype(self):
    # [FUNCTION DOC]
    # Asserts string sparse input raises a value error saying the input is not integer.
    column = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('omar', 'stringer', 'marlo'),
        dense_shape=(2, 2))
    with self.assertRaisesRegex(ValueError, 'Invalid input, not integer'):
      column.get_sparse_tensors(
          fc.FeatureTransformationCache({
              'aaa': inputs
          }), None)

  def test_parse_example(self):
    # [FUNCTION DOC]
    # Parses a serialized example with two int64 values and asserts the sparse result.
    a = fc.categorical_column_with_identity(key='aaa', num_buckets=30)
    data = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'aaa':
                    feature_pb2.Feature(
                        int64_list=feature_pb2.Int64List(value=[11, 21]))
            }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec_v2([a]))
    self.assertIn('aaa', features)

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [0, 1]],
            values=np.array([11, 21], dtype=np.int64),
            dense_shape=[1, 2]), self.evaluate(features['aaa']))

  def test_get_sparse_tensors(self):
    # [FUNCTION DOC]
    # Asserts the ids of sparse integer input are returned unchanged and no weights are present.
    column = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)), values=(0, 1, 0), dense_shape=(2, 2))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': inputs
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((0, 1, 0), dtype=np.int64),
            dense_shape=inputs.dense_shape),
        self.evaluate(id_weight_pair.id_tensor))

  def test_transform_feature(self):
    # [FUNCTION DOC]
    # Asserts transformation via the version-2 helper returns the same ids.
    column = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)), values=(0, 1, 0), dense_shape=(2, 2))
    id_tensor = fc._transform_features_v2({
        'aaa': inputs
    }, [column], None)[column]

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((0, 1, 0), dtype=np.int64),
            dense_shape=inputs.dense_shape), self.evaluate(id_tensor))

  def test_get_sparse_tensors_dense_input(self):
    # [FUNCTION DOC]
    # Dense input with a negative padding entry; asserts the result is a sparse tensor holding
    # only the non-negative entries.
    column = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': ((0, -1), (1, 0))
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=((0, 0), (1, 0), (1, 1)),
            values=np.array((0, 1, 0), dtype=np.int64),
            dense_shape=(2, 2)), self.evaluate(id_weight_pair.id_tensor))

  def _test_get_sparse_tensors_with_inputs_too_small(self):
    # [FUNCTION DOC]
    # HELPER: Shared body for the too-small-input tests, run with and without control flow version
    # two.
    # Builds an embedding column over an identity column with two buckets and sparse input
    # containing negative ids, creates its state with the recording state manager and gets the
    # dense tensor. Asserts that the lookups equal the first and second embedding rows, i.e.
    # negative ids are ignored and the valid ones are looked up.
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
      # HELPER: Local embedding initializer used by the test.
      # Ignores its shape, dtype and partition information arguments and returns a fixed two by
      # two table of embedding values.
      del shape, dtype, partition_info
      return embedding_values

    # Build columns.
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc.embedding_column(
        categorical_column,
        dimension=embedding_dimension,
        initializer=_initializer)
    state_manager = _TestStateManager()
    embedding_column.create_state(state_manager)

    # Provide sparse input and get dense result.
    embedding_lookup = embedding_column.get_dense_tensor(
        fc.FeatureTransformationCache({'aaa': sparse_input}), state_manager)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())
    expected_lookups = ((1., 2.), (3., 5))
    self.assertAllEqual(expected_lookups, self.evaluate(embedding_lookup))

  def test_get_sparse_tensors_with_inputs_too_small(self):
    # [FUNCTION DOC]
    # Runs the shared too-small-input scenario with the default control flow implementation.
    self._test_get_sparse_tensors_with_inputs_too_small()

  @test_util.enable_control_flow_v2
  def test_get_sparse_tensors_with_inputs_too_small_v2(self):
    # [FUNCTION DOC]
    # Runs the shared too-small-input scenario with control flow version two enabled by a
    # decorator.
    self._test_get_sparse_tensors_with_inputs_too_small()

  def _test_get_sparse_tensors_with_inputs_too_big(self):
    # [FUNCTION DOC]
    # HELPER: Shared body for the too-big-input tests, run with and without control flow version
    # two.
    # Builds an embedding column over an identity column with two buckets and sparse input
    # containing an id equal to the bucket count. Asserts that evaluating the dense tensor raises
    # an op error stating the index is outside the valid range.
    # Inputs.
    vocabulary_size = 2
    sparse_input = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0)), values=(2, 0), dense_shape=(4, 5))

    # Embedding variable.
    embedding_dimension = 2
    embedding_values = (
        (1., 2.),  # id 0
        (3., 5.),  # id 1
    )

    def _initializer(shape, dtype, partition_info=None):
      # [FUNCTION DOC]
      # HELPER: Local embedding initializer used by the test.
      # Ignores its arguments and returns a fixed two by two table of embedding values.
      del shape, dtype, partition_info
      return embedding_values

    # Build columns.
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc.embedding_column(
        categorical_column,
        dimension=embedding_dimension,
        initializer=_initializer)
    state_manager = _TestStateManager()
    embedding_column.create_state(state_manager)

    with self.assertRaisesRegex(errors.OpError,
                                r'indices\[0\].*\[0, 2\)'):
      # Provide sparse input and get dense result.
      embedding_lookup = embedding_column.get_dense_tensor(
          fc.FeatureTransformationCache({'aaa': sparse_input}), state_manager)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.evaluate(embedding_lookup)

  def test_get_sparse_tensors_with_inputs_too_big(self):
    # [FUNCTION DOC]
    # Runs the shared too-big-input scenario with the default control flow implementation.
    self._test_get_sparse_tensors_with_inputs_too_big()

  @test_util.enable_control_flow_v2
  def test_get_sparse_tensors_with_inputs_too_big_v2(self):
    # [FUNCTION DOC]
    # Runs the shared too-big-input scenario with control flow version two enabled by a decorator.
    self._test_get_sparse_tensors_with_inputs_too_big()

  def test_get_sparse_tensors_with_default_value(self):
    # [FUNCTION DOC]
    # Column with default value three and four buckets; asserts negative and too-large ids are
    # replaced by the default while valid ids are kept.
    column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=4, default_value=3)
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=(1, -1, 99),
        dense_shape=(2, 2))
    id_weight_pair = column.get_sparse_tensors(
        fc.FeatureTransformationCache({
            'aaa': inputs
        }), None)
    self.assertIsNone(id_weight_pair.weight_tensor)

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=inputs.indices,
            values=np.array((1, 3, 3), dtype=np.int64),
            dense_shape=inputs.dense_shape),
        self.evaluate(id_weight_pair.id_tensor))

  def test_get_sparse_tensors_with_default_value_and_placeholder_inputs(self):
    # [FUNCTION DOC]
    # Graph-only scenario using placeholders for the indices, values and shape of the sparse
    # input; feeds data at run time and asserts out-of-range ids are mapped to the default value.
    # This test needs to run in a graph because it is explicitly testing
    # graph placeholders
    with ops.Graph().as_default():
      column = fc.categorical_column_with_identity(
          key='aaa', num_buckets=4, default_value=3)
      input_indices = array_ops.placeholder(dtype=dtypes.int64)
      input_values = array_ops.placeholder(dtype=dtypes.int32)
      input_shape = array_ops.placeholder(dtype=dtypes.int64)
      inputs = sparse_tensor.SparseTensorValue(
          indices=input_indices, values=input_values, dense_shape=input_shape)
      id_weight_pair = column.get_sparse_tensors(
          fc.FeatureTransformationCache({
              'aaa': inputs
          }), None)
      self.assertIsNone(id_weight_pair.weight_tensor)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())
      with _initialized_session():
        _assert_sparse_tensor_value(
            self,
            sparse_tensor.SparseTensorValue(
                indices=np.array(((0, 0), (1, 0), (1, 1)), dtype=np.int64),
                values=np.array((1, 3, 3), dtype=np.int64),
                dense_shape=np.array((2, 2), dtype=np.int64)),
            id_weight_pair.id_tensor.eval(
                feed_dict={
                    input_indices: ((0, 0), (1, 0), (1, 1)),
                    input_values: (1, -1, 99),
                    input_shape: (2, 2),
                }))

  def test_old_linear_model(self):
    # [FUNCTION DOC]
    # Legacy linear model over an identity column: asserts zero weights, then predictions equal to
    # the weights of the ids of each example (one and five) after assignment.
    column = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
    self.assertEqual(3, column.num_buckets)
    with ops.Graph().as_default():
      predictions = fc_old.linear_model({
          column.name:
              sparse_tensor.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(0, 2, 1),
                  dense_shape=(2, 2))
      }, (column,))
      bias = get_linear_model_bias()
      weight_var = get_linear_model_column_var(column)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose((0.,), self.evaluate(bias))
      self.assertAllClose(((0.,), (0.,), (0.,)), self.evaluate(weight_var))
      self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
      self.evaluate(weight_var.assign(((1.,), (2.,), (3.,))))
      # weight_var[0] = 1
      # weight_var[2] + weight_var[1] = 3+2 = 5
      self.assertAllClose(((1.,), (5.,)), self.evaluate(predictions))

  def test_serialization(self):
    # [FUNCTION DOC]
    # Asserts the parents, the configuration (default value, key, bucket count) and that
    # rebuilding gives an equal column.
    column = fc.categorical_column_with_identity(key='aaa', num_buckets=3)

    self.assertEqual(['aaa'], column.parents)

    config = column.get_config()
    self.assertEqual({
        'default_value': None,
        'key': 'aaa',
        'number_buckets': 3
    }, config)

    self.assertEqual(column, fc.IdentityCategoricalColumn.from_config(config))


class TransformFeaturesTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the multi-column transformation helper; most transformation behavior is tested
  # in the per-column test classes.

  # All transform tests are distributed in column test.
  # Here we only test multi column case and naming
  def transform_multi_column(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The method name does not begin with the test prefix, so the test runner never
    # collects or runs it; it is dead test code.
    # Intended scenario: transforms a bucketized price column and a hashed wire column together in
    # a graph, then asserts the output names contain the column names, the bucket ids are as
    # expected, and the hashed ids are the expected values.
    bucketized_price = fc.bucketized_column(
        fc.numeric_column('price'), boundaries=[0, 2, 4, 6])
    hashed_sparse = fc.categorical_column_with_hash_bucket('wire', 10)
    with ops.Graph().as_default():
      features = {
          'price': [[-1.], [5.]],
          'wire':
              sparse_tensor.SparseTensor(
                  values=['omar', 'stringer', 'marlo'],
                  indices=[[0, 0], [1, 0], [1, 1]],
                  dense_shape=[2, 2])
      }
      transformed = fc._transform_features_v2(
          features, [bucketized_price, hashed_sparse], None)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertIn(bucketized_price.name, transformed[bucketized_price].name)
      self.assertAllEqual([[0], [3]],
                          self.evaluate(transformed[bucketized_price]))
      self.assertIn(hashed_sparse.name, transformed[hashed_sparse].name)
      self.assertAllEqual([6, 4, 1],
                          self.evaluate(transformed[hashed_sparse].values))

  def test_column_order(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring (when the column is both dense and sparse, sparse tensors are
    # used) was copied from another test and does not describe this test, which checks
    # transformation call ordering.
    # Defines a local column that records the order in which transforms are called, transforms two
    # such columns in both orders and asserts the call order follows the order in the list given
    # to the helper.
    """When the column is both dense and sparse, uses sparse tensors."""

    class _LoggerColumn(BaseFeatureColumnForTests):
      # [CLASS DOC]
      # HELPER: Local fixture column that records transformation order.
      # Feature column subclass constructed with a name. It is version 2 and its transform stores
      # the current counter of a shared call logger on itself as its call order, increments the
      # counter, and returns a fixed string.

      def __init__(self, name):
        # [FUNCTION DOC]
        # HELPER: Constructor of the local logging test column.
        # Takes the column name, calls the parent constructor and stores the name.
        super(_LoggerColumn, self).__init__()
        self._name = name

      @property
      def _is_v2_column(self):
        # [FUNCTION DOC]
        # HELPER: Marker property on a local test column.
        # Property returning true so the column is treated as a version-2 feature column.
        return True

      @property
      def name(self):
        # [FUNCTION DOC]
        # HELPER: Name property of a local test column.
        # Property returning the name given at construction.
        return self._name

      def transform_feature(self, transformation_cache, state_manager):
        # [FUNCTION DOC]
        # HELPER: Transform hook of the local logging test column.
        # Records the current value of the shared counter mapping as the call order, increments
        # that counter and returns the fixed string Anything. Side effect: mutates the shared
        # counter and the instance.
        self.call_order = call_logger['count']
        call_logger['count'] += 1
        return 'Anything'

      @property
      def parse_example_spec(self):
        # [FUNCTION DOC]
        # HELPER: Parse spec property of a local test column.
        # Property that does nothing and returns the null value.
        pass

    with ops.Graph().as_default():
      column1 = _LoggerColumn('1')
      column2 = _LoggerColumn('2')
      call_logger = {'count': 0}
      fc._transform_features_v2({}, [column1, column2], None)
      self.assertEqual(0, column1.call_order)
      self.assertEqual(1, column2.call_order)

      call_logger = {'count': 0}
      fc._transform_features_v2({}, [column2, column1], None)
      self.assertEqual(0, column1.call_order)
      self.assertEqual(1, column2.call_order)


class IndicatorColumnTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the indicator (multi-hot) column: construction, shape handling of dense and
  # sparse categorical inputs, multi-hot counts, deep copy, parsing, transformation including
  # weighted columns and missing values, legacy linear model and input layer use, and
  # serialization.

  def test_indicator_column(self):
    # [FUNCTION DOC]
    # Asserts an indicator over a version-2 hashed column has the right name suffix, variable
    # shape and version flag, and an indicator over a legacy hashed column reports not version 2.
    a = fc.categorical_column_with_hash_bucket('a', 4)
    indicator_a = fc.indicator_column(a)
    self.assertEqual(indicator_a.categorical_column.name, 'a')
    self.assertEqual(indicator_a.name, 'a_indicator')
    self.assertEqual(indicator_a.variable_shape, [1, 4])
    self.assertTrue(indicator_a._is_v2_column)

    b = fc_old._categorical_column_with_hash_bucket('b', hash_bucket_size=100)
    indicator_b = fc.indicator_column(b)
    self.assertEqual(indicator_b.categorical_column.name, 'b')
    self.assertEqual(indicator_b.name, 'b_indicator')
    self.assertEqual(indicator_b.variable_shape, [1, 100])
    self.assertFalse(indicator_b._is_v2_column)

  def test_not_categorical_input(self):
    # [FUNCTION DOC]
    # Asserts building an indicator from a plain string raises a value error about an unsupported
    # input type.
    with self.assertRaisesRegex(ValueError, 'Unsupported input type.'):
      fc.indicator_column('aaa')

  def test_1D_shape_succeeds(self):
    # [FUNCTION DOC]
    # Hashed animal column with rank-one dense string input; asserts the transformed output is a
    # one-hot matrix per example.
    animal = fc.indicator_column(
        fc.categorical_column_with_hash_bucket('animal', 4))
    transformation_cache = fc.FeatureTransformationCache({
        'animal': ['fox', 'fox']
    })
    output = transformation_cache.get(animal, None)

    self.assertAllEqual([[0., 0., 1., 0.], [0., 0., 1., 0.]],
                        self.evaluate(output))

  def test_2D_shape_succeeds(self):
    # [FUNCTION DOC]
    # Same with a rank-two sparse input; asserts the same one-hot rows. Contains a to-do note
    # about switching to a keyed categorical column.
    # TODO(ispir/cassandrax): Switch to categorical_column_with_keys when ready.
    animal = fc.indicator_column(
        fc.categorical_column_with_hash_bucket('animal', 4))
    transformation_cache = fc.FeatureTransformationCache({
        'animal':
            sparse_tensor.SparseTensor(
                indices=[[0, 0], [1, 0]],
                values=['fox', 'fox'],
                dense_shape=[2, 1])
    })
    output = transformation_cache.get(animal, None)

    self.assertAllEqual([[0., 0., 1., 0.], [0., 0., 1., 0.]],
                        self.evaluate(output))

  def test_multi_hot(self):
    # [FUNCTION DOC]
    # Identity column with repeated ids in one example; asserts the count for the repeated id is
    # two (counts rather than binary values).
    animal = fc.indicator_column(
        fc.categorical_column_with_identity('animal', num_buckets=4))

    transformation_cache = fc.FeatureTransformationCache({
        'animal':
            sparse_tensor.SparseTensor(
                indices=[[0, 0], [0, 1]], values=[1, 1], dense_shape=[1, 2])
    })
    output = transformation_cache.get(animal, None)

    self.assertAllEqual([[0., 2., 0., 0.]], self.evaluate(output))

  def test_multi_hot2(self):
    # [FUNCTION DOC]
    # Identity column with two different ids in one example; asserts both positions are one.
    animal = fc.indicator_column(
        fc.categorical_column_with_identity('animal', num_buckets=4))
    transformation_cache = fc.FeatureTransformationCache({
        'animal':
            sparse_tensor.SparseTensor(
                indices=[[0, 0], [0, 1]], values=[1, 2], dense_shape=[1, 2])
    })
    output = transformation_cache.get(animal, None)

    self.assertAllEqual([[0., 1., 1., 0.]], self.evaluate(output))

  def test_deep_copy(self):
    # [FUNCTION DOC]
    # Deep copies an indicator column and asserts the copy retains the categorical column name,
    # the indicator name and variable shape.
    a = fc.categorical_column_with_hash_bucket('a', 4)
    column = fc.indicator_column(a)
    column_copy = copy.deepcopy(column)
    self.assertEqual(column_copy.categorical_column.name, 'a')
    self.assertEqual(column.name, 'a_indicator')
    self.assertEqual(column.variable_shape, [1, 4])

  def test_parse_example(self):
    # [FUNCTION DOC]
    # Parses a serialized example using the spec made from an indicator over a vocabulary list
    # column and asserts the underlying sparse string feature.
    a = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    a_indicator = fc.indicator_column(a)
    data = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'aaa':
                    feature_pb2.Feature(
                        bytes_list=feature_pb2.BytesList(
                            value=[b'omar', b'stringer']))
            }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec_v2([a_indicator]))
    self.assertIn('aaa', features)

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [0, 1]],
            values=np.array([b'omar', b'stringer'], dtype=np.object_),
            dense_shape=[1, 2]), self.evaluate(features['aaa']))

  def test_transform(self):
    # [FUNCTION DOC]
    # Transforms sparse string input through an indicator over a vocabulary list; asserts the
    # expected multi-hot rows.
    a = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    a_indicator = fc.indicator_column(a)
    features = {
        'aaa':
            sparse_tensor.SparseTensorValue(
                indices=((0, 0), (1, 0), (1, 1)),
                values=('marlo', 'skywalker', 'omar'),
                dense_shape=(2, 2))
    }
    indicator_tensor = fc._transform_features_v2(features, [a_indicator],
                                                 None)[a_indicator]

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    self.assertAllEqual([[0, 0, 1], [1, 0, 0]], self.evaluate(indicator_tensor))

  def test_transform_with_weighted_column(self):
    # [FUNCTION DOC]
    # Regression test for a reported issue: indicator over a weighted categorical column; asserts
    # the output contains summed weights per id.
    # Github issue 12557
    ids = fc.categorical_column_with_vocabulary_list(
        key='ids', vocabulary_list=('a', 'b', 'c'))
    weights = fc.weighted_categorical_column(ids, 'weights')
    indicator = fc.indicator_column(weights)
    features = {
        'ids': constant_op.constant([['c', 'b', 'a', 'c']]),
        'weights': constant_op.constant([[2., 4., 6., 1.]])
    }
    indicator_tensor = fc._transform_features_v2(features, [indicator],
                                                 None)[indicator]

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    self.assertAllEqual([[6., 4., 3.]], self.evaluate(indicator_tensor))

  def test_transform_with_missing_value_in_weighted_column(self):
    # [FUNCTION DOC]
    # Regression test for a reported issue: an unknown id in a weighted column; asserts that
    # unknown entry is dropped while the known weights are placed in their slots.
    # Github issue 12583
    ids = fc.categorical_column_with_vocabulary_list(
        key='ids', vocabulary_list=('a', 'b', 'c'))
    weights = fc.weighted_categorical_column(ids, 'weights')
    indicator = fc.indicator_column(weights)
    features = {
        'ids': constant_op.constant([['c', 'b', 'unknown']]),
        'weights': constant_op.constant([[2., 4., 6.]])
    }
    indicator_tensor = fc._transform_features_v2(features, [indicator],
                                                 None)[indicator]

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    self.assertAllEqual([[0., 4., 2.]], self.evaluate(indicator_tensor))

  def test_transform_with_missing_value_in_categorical_column(self):
    # [FUNCTION DOC]
    # Regression test for a reported issue: an unknown id in a plain categorical column; asserts
    # the output has ones for known ids and the unknown entry does not contribute.
    # Github issue 12583
    ids = fc.categorical_column_with_vocabulary_list(
        key='ids', vocabulary_list=('a', 'b', 'c'))
    indicator = fc.indicator_column(ids)
    features = {
        'ids': constant_op.constant([['c', 'b', 'unknown']]),
    }
    indicator_tensor = fc._transform_features_v2(features, [indicator],
                                                 None)[indicator]

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    self.assertAllEqual([[0., 1., 1.]], self.evaluate(indicator_tensor))

  def test_old_linear_model(self):
    # [FUNCTION DOC]
    # Legacy linear model over an indicator of an identity column: asserts zero weights and
    # predictions, then after assigning weights asserts the prediction equals the sum of the
    # weights of the two active ids.
    animal = fc.indicator_column(
        fc.categorical_column_with_identity('animal', num_buckets=4))
    with ops.Graph().as_default():
      features = {
          'animal':
              sparse_tensor.SparseTensor(
                  indices=[[0, 0], [0, 1]], values=[1, 2], dense_shape=[1, 2])
      }

      predictions = fc_old.linear_model(features, [animal])
      weight_var = get_linear_model_column_var(animal)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      # All should be zero-initialized.
      self.assertAllClose([[0.], [0.], [0.], [0.]], self.evaluate(weight_var))
      self.assertAllClose([[0.]], self.evaluate(predictions))
      self.evaluate(weight_var.assign([[1.], [2.], [3.], [4.]]))
      self.assertAllClose([[2. + 3.]], self.evaluate(predictions))

  def test_old_linear_model_old_categorical(self):
    # [FUNCTION DOC]
    # Same as above but the indicator is built over a legacy identity column.
    animal = fc.indicator_column(
        fc_old._categorical_column_with_identity('animal', num_buckets=4))
    with ops.Graph().as_default():
      features = {
          'animal':
              sparse_tensor.SparseTensor(
                  indices=[[0, 0], [0, 1]], values=[1, 2], dense_shape=[1, 2])
      }

      predictions = fc_old.linear_model(features, [animal])
      weight_var = get_linear_model_column_var(animal)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      # All should be zero-initialized.
      self.assertAllClose([[0.], [0.], [0.], [0.]], self.evaluate(weight_var))
      self.assertAllClose([[0.]], self.evaluate(predictions))
      self.evaluate(weight_var.assign([[1.], [2.], [3.], [4.]]))
      self.assertAllClose([[2. + 3.]], self.evaluate(predictions))

  def test_input_layer(self):
    # [FUNCTION DOC]
    # Legacy input layer over an indicator column: asserts the output is the expected multi-hot
    # row.
    animal = fc.indicator_column(
        fc.categorical_column_with_identity('animal', num_buckets=4))
    with ops.Graph().as_default():
      features = {
          'animal':
              sparse_tensor.SparseTensor(
                  indices=[[0, 0], [0, 1]], values=[1, 2], dense_shape=[1, 2])
      }
      net = fc_old.input_layer(features, [animal])

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([[0., 1., 1., 0.]], self.evaluate(net))

  def test_input_layer_old_categorical(self):
    # [FUNCTION DOC]
    # Same as above with a legacy categorical source column.
    animal = fc.indicator_column(
        fc_old._categorical_column_with_identity('animal', num_buckets=4))
    with ops.Graph().as_default():
      features = {
          'animal':
              sparse_tensor.SparseTensor(
                  indices=[[0, 0], [0, 1]], values=[1, 2], dense_shape=[1, 2])
      }
      net = fc_old.input_layer(features, [animal])

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose([[0., 1., 1., 0.]], self.evaluate(net))

  def test_serialization(self):
    # [FUNCTION DOC]
    # Asserts the parents are the categorical column, the configuration contains the nested
    # categorical column configuration, and rebuilding gives an equal column with a fresh
    # categorical column unless the existing one is supplied by name.
    parent = fc.categorical_column_with_identity('animal', num_buckets=4)
    animal = fc.indicator_column(parent)

    self.assertEqual([parent], animal.parents)

    config = animal.get_config()
    self.assertEqual({
        'categorical_column': {
            'class_name': 'IdentityCategoricalColumn',
            'config': {
                'key': 'animal',
                'default_value': None,
                'number_buckets': 4
            }
        }
    }, config)

    new_animal = fc.IndicatorColumn.from_config(config)
    self.assertEqual(animal, new_animal)
    self.assertIsNot(parent, new_animal.categorical_column)

    new_animal = fc.IndicatorColumn.from_config(
        config,
        columns_by_name={
            serialization._column_name_with_class_name(parent): parent
        })
    self.assertEqual(animal, new_animal)
    self.assertIs(parent, new_animal.categorical_column)


class _TestStateManager(fc.StateManager):
  # [CLASS DOC]
  # HELPER: Test-only state manager used by embedding column tests.
  # Subclass of the feature column state manager that stores variables per feature column and name
  # in a nested mapping and creates them with variable scope getters. Holds a trainable flag that
  # is combined with each variable's own trainable request, so the whole manager can disable
  # training.

  def __init__(self, trainable=True):
    # [FUNCTION DOC]
    # HELPER: Constructor of the recording state manager.
    # Takes an optional trainable flag (default true), creates an empty mapping from feature
    # columns to their variable mappings and stores the flag. Returns nothing.
    # Dict of feature_column to a dict of variables.
    self._all_variables = {}
    self._trainable = trainable

  def create_variable(self,
                      feature_column,
                      name,
                      shape,
                      dtype=None,
                      trainable=True,
                      use_resource=True,
                      initializer=None):
    # [FUNCTION DOC]
    # HELPER: State manager hook used by columns to create their variables in tests.
    # Takes a feature column, a variable name, shape, optional dtype, trainable flag (default
    # true), use-resource flag (default true) and initializer. Returns the already stored variable
    # when the column and name were seen before; otherwise creates a variable via the variable
    # scope getter (trainable only if both the manager and the request allow it), stores it and
    # returns it.
    # Side effect: may create a variable in the current graph or eager context.
    if feature_column not in self._all_variables:
      self._all_variables[feature_column] = {}
    var_dict = self._all_variables[feature_column]
    if name in var_dict:
      return var_dict[name]
    else:
      var = variable_scope.get_variable(
          name=name,
          shape=shape,
          dtype=dtype,
          trainable=self._trainable and trainable,
          use_resource=use_resource,
          initializer=initializer)
      var_dict[name] = var
      return var

  def get_variable(self, feature_column, name):
    # [FUNCTION DOC]
    # HELPER: State manager hook used by columns to read their variables in tests.
    # Takes a feature column and a name and returns the stored variable. Raises a value error
    # saying the feature column is not recognized if the column is unknown, and another value
    # error saying the variable could not be found if the name is unknown for that column.
    if feature_column not in self._all_variables:
      raise ValueError('Do not recognize FeatureColumn.')
    if name in self._all_variables[feature_column]:
      return self._all_variables[feature_column][name]
    raise ValueError('Could not find variable.')


class EmbeddingColumnTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the embedding column (parameterized test base): defaults, constructor arguments,
  # version flag, deep copy, initializer validation, parsing, transformation, dense tensor lookups
  # with the mean combiner (2D and 3D inputs, placeholders, legacy categorical input, restore from
  # checkpoint), legacy input layer and linear model behavior, and serialization with default and
  # custom initializers.

  def test_defaults(self):
    # [FUNCTION DOC]
    # Creates an embedding column over an identity column and asserts all defaults: categorical
    # column, dimension, mean combiner, no checkpoint info, no max norm, trainable, name with
    # embedding suffix, variable shape, parse spec and version flag.
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_dimension = 2
    embedding_column = fc.embedding_column(
        categorical_column, dimension=embedding_dimension)
    self.assertIs(categorical_column, embedding_column.categorical_column)
    self.assertEqual(embedding_dimension, embedding_column.dimension)
    self.assertEqual('mean', embedding_column.combiner)
    self.assertIsNone(embedding_column.ckpt_to_load_from)
    self.assertIsNone(embedding_column.tensor_name_in_ckpt)
    self.assertIsNone(embedding_column.max_norm)
    self.assertTrue(embedding_column.trainable)
    self.assertEqual('aaa_embedding', embedding_column.name)
    self.assertEqual((embedding_dimension,), embedding_column.variable_shape)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int64)
    }, embedding_column.parse_example_spec)
    self.assertTrue(embedding_column._is_v2_column)

  def test_is_v2_column(self):
    # [FUNCTION DOC]
    # Asserts an embedding column over a legacy categorical column is not flagged version 2.
    categorical_column = fc_old._categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_dimension = 2
    embedding_column = fc.embedding_column(
        categorical_column, dimension=embedding_dimension)
    self.assertFalse(embedding_column._is_v2_column)

  def test_all_constructor_args(self):
    # [FUNCTION DOC]
    # Creates an embedding column with every optional argument (combiner, initializer, checkpoint
    # path and tensor name, max norm, trainable false) and asserts each is exposed unchanged on
    # the column.
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_dimension = 2
    embedding_column = fc.embedding_column(
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
    self.assertEqual((embedding_dimension,), embedding_column.variable_shape)
    self.assertEqual({
        'aaa': parsing_ops.VarLenFeature(dtypes.int64)
    }, embedding_column.parse_example_spec)

  def test_deep_copy(self):
    # [FUNCTION DOC]
    # Deep copies such a fully configured column and asserts both the original and the copy keep
    # all attributes, including the nested categorical column details.
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_dimension = 2
    original = fc.embedding_column(
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
      self.assertEqual(3, embedding_column.categorical_column.num_buckets)
      self.assertEqual({
          'aaa': parsing_ops.VarLenFeature(dtypes.int64)
      }, embedding_column.categorical_column.parse_example_spec)

      self.assertEqual(embedding_dimension, embedding_column.dimension)
      self.assertEqual('my_combiner', embedding_column.combiner)
      self.assertEqual('my_ckpt', embedding_column.ckpt_to_load_from)
      self.assertEqual('my_ckpt_tensor', embedding_column.tensor_name_in_ckpt)
      self.assertEqual(42., embedding_column.max_norm)
      self.assertFalse(embedding_column.trainable)
      self.assertEqual('aaa_embedding', embedding_column.name)
      self.assertEqual((embedding_dimension,), embedding_column.variable_shape)
      self.assertEqual({
          'aaa': parsing_ops.VarLenFeature(dtypes.int64)
      }, embedding_column.parse_example_spec)

  def test_invalid_initializer(self):
    # [FUNCTION DOC]
    # Asserts a non-callable initializer raises a value error saying the initializer must be
    # callable.
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    with self.assertRaisesRegex(ValueError, 'initializer must be callable'):
      fc.embedding_column(categorical_column, dimension=2, initializer='not_fn')

  def test_parse_example(self):
    # [FUNCTION DOC]
    # Parses a serialized example using the spec of an embedding over a vocabulary list column and
    # asserts the sparse string result.
    a = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    a_embedded = fc.embedding_column(a, dimension=2)
    data = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'aaa':
                    feature_pb2.Feature(
                        bytes_list=feature_pb2.BytesList(
                            value=[b'omar', b'stringer']))
            }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec_v2([a_embedded]))
    self.assertIn('aaa', features)

    _assert_sparse_tensor_value(
        self,
        sparse_tensor.SparseTensorValue(
            indices=[[0, 0], [0, 1]],
            values=np.array([b'omar', b'stringer'], dtype=np.object_),
            dense_shape=[1, 2]), self.evaluate(features['aaa']))

  def test_transform_feature(self):
    # [FUNCTION DOC]
    # Asserts that the embedding column's transform output equals its categorical column's
    # transform output (the embedding is applied later at lookup).
    a = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
    a_embedded = fc.embedding_column(a, dimension=2)
    features = {
        'aaa':
            sparse_tensor.SparseTensor(
                indices=((0, 0), (1, 0), (1, 1)),
                values=(0, 1, 0),
                dense_shape=(2, 2))
    }
    outputs = fc._transform_features_v2(features, [a, a_embedded], None)
    output_a = outputs[a]
    output_embedded = outputs[a_embedded]

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

    _assert_sparse_tensor_value(self, self.evaluate(output_a),
                                self.evaluate(output_embedded))

  def test_get_dense_tensor(self):
    # [FUNCTION DOC]
    # Scenario: four examples of ids with an empty example, a three-id vocabulary and fixed
    # embedding values.
    # Creates the column state through the recording state manager, gets the dense tensor, and
    # asserts in graph mode that the only global variable is the embedding weights holding the
    # fixed values; always asserts the mean-combined lookups (empty examples give zeros).
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

    def _initializer(shape, dtype, partition_info=None):
      # [FUNCTION DOC]
      # HELPER: Local embedding initializer used by the test.
      # Asserts the requested shape is vocabulary size by dimension, the dtype is float32 and
      # there is no partition info, then returns the fixed embedding table.
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
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc.embedding_column(
        categorical_column,
        dimension=embedding_dimension,
        initializer=_initializer)
    state_manager = _TestStateManager()
    embedding_column.create_state(state_manager)

    # Provide sparse input and get dense result.
    embedding_lookup = embedding_column.get_dense_tensor(
        fc.FeatureTransformationCache({
            'aaa': sparse_input
        }), state_manager)

    # Assert expected embedding variable and lookups.
    if not context.executing_eagerly():
      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertCountEqual(('embedding_weights:0',),
                            tuple([v.name for v in global_vars]))

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllEqual(embedding_values, self.evaluate(global_vars[0]))
    self.assertAllEqual(expected_lookups, self.evaluate(embedding_lookup))

  def test_get_dense_tensor_old_categorical(self):
    # [FUNCTION DOC]
    # Same as the previous test but with a legacy categorical column and the legacy lookup entry
    # points using a legacy lazy input builder; asserts the same variable and lookups.
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

    def _initializer(shape, dtype, partition_info=None):
      # [FUNCTION DOC]
      # HELPER: Local embedding initializer used by the test.
      # Asserts the shape, float32 dtype and absence of partition info, then returns the fixed
      # embedding table.
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
    categorical_column = fc_old._categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc.embedding_column(
        categorical_column,
        dimension=embedding_dimension,
        initializer=_initializer)

    # Provide sparse input and get dense result.
    embedding_lookup = embedding_column._get_dense_tensor(
        fc_old._LazyBuilder({
            'aaa': sparse_input
        }))

    # Assert expected embedding variable and lookups.
    if not context.executing_eagerly():
      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertCountEqual(('embedding_weights:0',),
                            tuple([v.name for v in global_vars]))

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllEqual(embedding_values, self.evaluate(global_vars[0]))
    self.assertAllEqual(expected_lookups, self.evaluate(embedding_lookup))

  def test_get_dense_tensor_3d(self):
    # [FUNCTION DOC]
    # Rank-three sparse input with four ids; asserts the lookups produce a three-dimensional
    # result with mean combination over the last dimension and zeros for empty cells, and the
    # embedding variable holds the expected values.
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

    def _initializer(shape, dtype, partition_info=None):
      # [FUNCTION DOC]
      # HELPER: Local embedding initializer used by the test.
      # Asserts the shape, float32 dtype and absence of partition info, then returns the fixed
      # four by three table.
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
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc.embedding_column(
        categorical_column,
        dimension=embedding_dimension,
        initializer=_initializer)
    state_manager = _TestStateManager()
    embedding_column.create_state(state_manager)

    # Provide sparse input and get dense result.
    embedding_lookup = embedding_column.get_dense_tensor(
        fc.FeatureTransformationCache({
            'aaa': sparse_input
        }), state_manager)

    # Assert expected embedding variable and lookups.
    if not context.executing_eagerly():
      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertCountEqual(('embedding_weights:0',),
                            tuple([v.name for v in global_vars]))

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllEqual(embedding_values, self.evaluate(global_vars[0]))
    self.assertAllEqual(expected_lookups, self.evaluate(embedding_lookup))

  def test_get_dense_tensor_placeholder_inputs(self):
    # [FUNCTION DOC]
    # Graph-only scenario where indices, values and shape of the sparse input are placeholders;
    # asserts the embedding variable and the lookups after feeding the data.
    # This test explicitly checks graph placeholders, so we need to
    # be in a graph
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

      def _initializer(shape, dtype, partition_info=None):
        # [FUNCTION DOC]
        # HELPER: Local embedding initializer used by the test.
        # Asserts the shape, float32 dtype and absence of partition info, then returns the fixed
        # embedding table.
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
      categorical_column = fc.categorical_column_with_identity(
          key='aaa', num_buckets=vocabulary_size)
      embedding_column = fc.embedding_column(
          categorical_column,
          dimension=embedding_dimension,
          initializer=_initializer)
      state_manager = _TestStateManager()
      embedding_column.create_state(state_manager)

      # Provide sparse input and get dense result.
      input_indices = array_ops.placeholder(dtype=dtypes.int64)
      input_values = array_ops.placeholder(dtype=dtypes.int64)
      input_shape = array_ops.placeholder(dtype=dtypes.int64)
      embedding_lookup = embedding_column.get_dense_tensor(
          fc.FeatureTransformationCache({
              'aaa':
                  sparse_tensor.SparseTensorValue(
                      indices=input_indices,
                      values=input_values,
                      dense_shape=input_shape)
          }), state_manager)

      # Assert expected embedding variable and lookups.
      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertCountEqual(('embedding_weights:0',),
                            tuple([v.name for v in global_vars]))

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())
      with _initialized_session():
        self.assertAllEqual(embedding_values, self.evaluate(global_vars[0]))
        self.assertAllEqual(
            expected_lookups,
            embedding_lookup.eval(
                feed_dict={
                    input_indices: sparse_input.indices,
                    input_values: sparse_input.values,
                    input_shape: sparse_input.dense_shape,
                }))

  def test_get_dense_tensor_restore_from_ckpt(self):
    # [FUNCTION DOC]
    # Embedding column configured to load its weights from a checkpoint file and tensor name in
    # the test data directory; asserts the variable and the lookups equal the values stored in the
    # checkpoint.
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
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc.embedding_column(
        categorical_column,
        dimension=embedding_dimension,
        ckpt_to_load_from=ckpt_path,
        tensor_name_in_ckpt=ckpt_tensor)
    state_manager = _TestStateManager()
    embedding_column.create_state(state_manager)

    # Provide sparse input and get dense result.
    embedding_lookup = embedding_column.get_dense_tensor(
        fc.FeatureTransformationCache({
            'aaa': sparse_input
        }), state_manager)

    # Assert expected embedding variable and lookups.
    if not context.executing_eagerly():
      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertCountEqual(('embedding_weights:0',),
                            tuple([v.name for v in global_vars]))

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllEqual(embedding_values, self.evaluate(global_vars[0]))
    self.assertAllEqual(expected_lookups, self.evaluate(embedding_lookup))

  def test_input_layer(self):
    # [FUNCTION DOC]
    # Legacy input layer over an embedding column; in graph mode asserts the variable names and
    # that the variable is trainable, always asserts the lookups equal the expected mean-combined
    # values.
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

    def _initializer(shape, dtype, partition_info=None):
      # [FUNCTION DOC]
      # HELPER: Local embedding initializer used by the test.
      # Asserts the shape, float32 dtype and absence of partition info, then returns the fixed
      # embedding table.
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
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc.embedding_column(
        categorical_column,
        dimension=embedding_dimension,
        initializer=_initializer)

    # Provide sparse input and get dense result.
    feature_layer = fc_old.input_layer({
        'aaa': sparse_input
    }, (embedding_column,))

    if not context.executing_eagerly():
      # Assert expected embedding variable and lookups.
      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertCountEqual(('input_layer/aaa_embedding/embedding_weights:0',),
                            tuple([v.name for v in global_vars]))
      trainable_vars = ops.get_collection(ops.GraphKeys.TRAINABLE_VARIABLES)
      self.assertCountEqual(('input_layer/aaa_embedding/embedding_weights:0',),
                            tuple([v.name for v in trainable_vars]))

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllEqual(embedding_values, self.evaluate(trainable_vars[0]))
    self.assertAllEqual(expected_lookups, self.evaluate(feature_layer))

  def test_old_linear_model(self):
    # [FUNCTION DOC]
    # Graph-mode legacy linear model over an embedding column with a zero initializer: asserts the
    # three variable names (bias, linear weights, embedding weights) are trainable and zero, then
    # after assigning values asserts predictions equal hand-computed weighted sums of mean
    # embeddings.
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

    def _initializer(shape, dtype, partition_info=None):
      # [FUNCTION DOC]
      # HELPER: Local embedding initializer used by the test.
      # Asserts the shape and float32 dtype, requires no partition info and returns zeros of the
      # embedding shape.
      self.assertAllEqual(embedding_shape, shape)
      self.assertEqual(dtypes.float32, dtype)
      self.assertIsNone(partition_info)
      return zeros_embedding_values

    # Build columns.
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc.embedding_column(
        categorical_column,
        dimension=embedding_dimension,
        initializer=_initializer)

    with ops.Graph().as_default():
      predictions = fc_old.linear_model({
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

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      # Predictions with all zero weights.
      self.assertAllClose(np.zeros((1,)), self.evaluate(bias))
      self.assertAllClose(zeros_embedding_values,
                          self.evaluate(embedding_weights))
      self.assertAllClose(
          np.zeros((embedding_dimension, 1)), self.evaluate(linear_weights))
      self.assertAllClose(np.zeros((batch_size, 1)), self.evaluate(predictions))

      # Predictions with all non-zero weights.
      self.evaluate(
          embedding_weights.assign((
              (1., 2.),  # id 0
              (3., 5.),  # id 1
              (7., 11.)  # id 2
          )))
      self.evaluate(linear_weights.assign(((4.,), (6.,))))
      # example 0, ids [2], embedding[0] = [7, 11]
      # example 1, ids [0, 1], embedding[1] = mean([1, 2] + [3, 5]) = [2, 3.5]
      # example 2, ids [], embedding[2] = [0, 0]
      # example 3, ids [1], embedding[3] = [3, 5]
      # sum(embeddings * linear_weights)
      # = [4*7 + 6*11, 4*2 + 6*3.5, 4*0 + 6*0, 4*3 + 6*5] = [94, 29, 0, 42]
      self.assertAllClose(((94.,), (29.,), (0.,), (42.,)),
                          self.evaluate(predictions))

  def test_old_linear_model_old_categorical(self):
    # [FUNCTION DOC]
    # Same as the previous test using a legacy categorical column under the embedding column.
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

    def _initializer(shape, dtype, partition_info=None):
      # [FUNCTION DOC]
      # HELPER: Local embedding initializer used by the test.
      # Asserts the shape and float32 dtype, requires no partition info and returns zeros of the
      # embedding shape.
      self.assertAllEqual(embedding_shape, shape)
      self.assertEqual(dtypes.float32, dtype)
      self.assertIsNone(partition_info)
      return zeros_embedding_values

    # Build columns.
    categorical_column = fc_old._categorical_column_with_identity(
        key='aaa', num_buckets=vocabulary_size)
    embedding_column = fc.embedding_column(
        categorical_column,
        dimension=embedding_dimension,
        initializer=_initializer)

    with ops.Graph().as_default():
      predictions = fc_old.linear_model({
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

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      # Predictions with all zero weights.
      self.assertAllClose(np.zeros((1,)), self.evaluate(bias))
      self.assertAllClose(zeros_embedding_values,
                          self.evaluate(embedding_weights))
      self.assertAllClose(
          np.zeros((embedding_dimension, 1)), self.evaluate(linear_weights))
      self.assertAllClose(np.zeros((batch_size, 1)), self.evaluate(predictions))

      # Predictions with all non-zero weights.
      self.evaluate(
          embedding_weights.assign((
              (1., 2.),  # id 0
              (3., 5.),  # id 1
              (7., 11.)  # id 2
          )))
      self.evaluate(linear_weights.assign(((4.,), (6.,))))
      # example 0, ids [2], embedding[0] = [7, 11]
      # example 1, ids [0, 1], embedding[1] = mean([1, 2] + [3, 5]) = [2, 3.5]
      # example 2, ids [], embedding[2] = [0, 0]
      # example 3, ids [1], embedding[3] = [3, 5]
      # sum(embeddings * linear_weights)
      # = [4*7 + 6*11, 4*2 + 6*3.5, 4*0 + 6*0, 4*3 + 6*5] = [94, 29, 0, 42]
      self.assertAllClose(((94.,), (29.,), (0.,), (42.,)),
                          self.evaluate(predictions))

  def test_serialization_with_default_initializer(self):
    # [FUNCTION DOC]
    # Asserts the column's parents, that the configuration contains the nested categorical column,
    # dimension, combiner, safe-lookup flag and a truncated-normal initializer description, and
    # that rebuilding (with and without an existing column mapping) gives a column with the same
    # configuration.

    # Build columns.
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_column = fc.embedding_column(categorical_column, dimension=2)

    self.assertEqual([categorical_column], embedding_column.parents)

    config = embedding_column.get_config()
    self.assertEqual(
        {
            'categorical_column': {
                'class_name': 'IdentityCategoricalColumn',
                'config': {
                    'number_buckets': 3,
                    'key': 'aaa',
                    'default_value': None
                }
            },
            'ckpt_to_load_from': None,
            'combiner': 'mean',
            'dimension': 2,
            'initializer': {
                'class_name': 'TruncatedNormal',
                'config': {
                    'dtype': 'float32',
                    'stddev': 0.7071067811865475,
                    'seed': None,
                    'mean': 0.0
                }
            },
            'max_norm': None,
            'tensor_name_in_ckpt': None,
            'trainable': True,
            'use_safe_embedding_lookup': True
        }, config)

    new_embedding_column = fc.EmbeddingColumn.from_config(
        config, custom_objects=None)
    self.assertEqual(embedding_column.get_config(),
                     new_embedding_column.get_config())
    self.assertIsNot(categorical_column,
                     new_embedding_column.categorical_column)

    new_embedding_column = fc.EmbeddingColumn.from_config(
        config,
        custom_objects=None,
        columns_by_name={
            serialization._column_name_with_class_name(categorical_column):
                categorical_column
        })
    self.assertEqual(embedding_column.get_config(),
                     new_embedding_column.get_config())
    self.assertIs(categorical_column, new_embedding_column.categorical_column)

  def test_serialization_with_custom_initializer(self):
    # [FUNCTION DOC]
    # Same with a custom initializer function: configuration stores its name, rebuilding uses the
    # custom-objects mapping to supply it. Also removes the safe-lookup flag from the
    # configuration to simulate old configurations and asserts rebuilding still works.

    def _initializer(shape, dtype, partition_info=None):
      # [FUNCTION DOC]
      # HELPER: Local initializer used by the serialization test.
      # Ignores its arguments and returns a value error object (it returns rather than raises it);
      # it is never called by the test.
      del shape, dtype, partition_info
      return ValueError('Not expected to be called')

    # Build columns.
    categorical_column = fc.categorical_column_with_identity(
        key='aaa', num_buckets=3)
    embedding_column = fc.embedding_column(
        categorical_column, dimension=2, initializer=_initializer)

    self.assertEqual([categorical_column], embedding_column.parents)

    config = embedding_column.get_config()
    self.assertEqual(
        {
            'categorical_column': {
                'class_name': 'IdentityCategoricalColumn',
                'config': {
                    'number_buckets': 3,
                    'key': 'aaa',
                    'default_value': None
                }
            },
            'ckpt_to_load_from': None,
            'combiner': 'mean',
            'dimension': 2,
            'initializer': '_initializer',
            'max_norm': None,
            'tensor_name_in_ckpt': None,
            'trainable': True,
            'use_safe_embedding_lookup': True
        }, config)

    custom_objects = {
        '_initializer': _initializer,
    }

    # use_safe_embedding_lookup might not be populated for legacy reasons.
    del config['use_safe_embedding_lookup']

    new_embedding_column = fc.EmbeddingColumn.from_config(
        config, custom_objects=custom_objects)
    self.assertEqual(embedding_column, new_embedding_column)
    self.assertIsNot(categorical_column,
                     new_embedding_column.categorical_column)

    new_embedding_column = fc.EmbeddingColumn.from_config(
        config,
        custom_objects=custom_objects,
        columns_by_name={
            serialization._column_name_with_class_name(categorical_column):
                categorical_column
        })
    self.assertEqual(embedding_column, new_embedding_column)
    self.assertIs(categorical_column, new_embedding_column.categorical_column)


class SharedEmbeddingColumnTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for version-2 shared embedding columns (graph mode only): defaults, constructor
  # arguments, deep copy, validation, parsing, transformation, dense lookups with and without safe
  # lookup and with partitioned variables, placeholder inputs and the parents in serialization.

  def test_defaults(self):
    # [FUNCTION DOC]
    # Creates shared embedding columns over two identity columns and asserts categorical columns,
    # absence of max norm, names with the shared-embedding suffix, variable shapes and parse specs
    # for each.
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=3)
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=3)
      embedding_dimension = 2
      embedding_column_b, embedding_column_a = fc.shared_embedding_columns_v2(
          [categorical_column_b, categorical_column_a],
          dimension=embedding_dimension)
      self.assertIs(categorical_column_a, embedding_column_a.categorical_column)
      self.assertIs(categorical_column_b, embedding_column_b.categorical_column)
      self.assertIsNone(embedding_column_a.max_norm)
      self.assertIsNone(embedding_column_b.max_norm)
      self.assertEqual('aaa_shared_embedding', embedding_column_a.name)
      self.assertEqual('bbb_shared_embedding', embedding_column_b.name)
      self.assertEqual((embedding_dimension,),
                       embedding_column_a.variable_shape)
      self.assertEqual((embedding_dimension,),
                       embedding_column_b.variable_shape)
      self.assertEqual({
          'aaa': parsing_ops.VarLenFeature(dtypes.int64)
      }, embedding_column_a.parse_example_spec)
      self.assertEqual({
          'bbb': parsing_ops.VarLenFeature(dtypes.int64)
      }, embedding_column_b.parse_example_spec)

  def test_all_constructor_args(self):
    # [FUNCTION DOC]
    # Creates shared embedding columns with every optional argument and asserts the max norm,
    # names, variable shapes and parse specs.
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=3)
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=3)
      embedding_dimension = 2
      embedding_column_a, embedding_column_b = fc.shared_embedding_columns_v2(
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
      self.assertEqual(42., embedding_column_a.max_norm)
      self.assertEqual(42., embedding_column_b.max_norm)
      self.assertEqual('aaa_shared_embedding', embedding_column_a.name)
      self.assertEqual('bbb_shared_embedding', embedding_column_b.name)
      self.assertEqual((embedding_dimension,),
                       embedding_column_a.variable_shape)
      self.assertEqual((embedding_dimension,),
                       embedding_column_b.variable_shape)
      self.assertEqual({
          'aaa': parsing_ops.VarLenFeature(dtypes.int64)
      }, embedding_column_a.parse_example_spec)
      self.assertEqual({
          'bbb': parsing_ops.VarLenFeature(dtypes.int64)
      }, embedding_column_b.parse_example_spec)

  def test_deep_copy(self):
    # [FUNCTION DOC]
    # Deep copies the first of two shared columns and asserts the original and copy keep the
    # categorical column details, max norm, name, shape and parse spec.
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=3)
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=3)
      embedding_dimension = 2
      original_a, _ = fc.shared_embedding_columns_v2(
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
        self.assertEqual(3, embedding_column_a.categorical_column.num_buckets)
        self.assertEqual({
            'aaa': parsing_ops.VarLenFeature(dtypes.int64)
        }, embedding_column_a.categorical_column.parse_example_spec)

        self.assertEqual(42., embedding_column_a.max_norm)
        self.assertEqual('aaa_shared_embedding', embedding_column_a.name)
        self.assertEqual((embedding_dimension,),
                         embedding_column_a.variable_shape)
        self.assertEqual({
            'aaa': parsing_ops.VarLenFeature(dtypes.int64)
        }, embedding_column_a.parse_example_spec)

  def test_invalid_initializer(self):
    # [FUNCTION DOC]
    # Asserts a non-callable initializer raises a value error saying the initializer must be
    # callable.
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=3)
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=3)
      with self.assertRaisesRegex(ValueError, 'initializer must be callable'):
        fc.shared_embedding_columns_v2(
            [categorical_column_a, categorical_column_b],
            dimension=2,
            initializer='not_fn')

  def test_incompatible_column_type(self):
    # [FUNCTION DOC]
    # Asserts mixing identity and hashed categorical columns raises a value error saying all
    # categorical columns must have the same type.
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=3)
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=3)
      categorical_column_c = fc.categorical_column_with_hash_bucket(
          key='ccc', hash_bucket_size=3)
      with self.assertRaisesRegex(
          ValueError, 'all categorical_columns must have the same type.*'
          'IdentityCategoricalColumn.*HashedCategoricalColumn'):
        fc.shared_embedding_columns_v2(
            [categorical_column_a, categorical_column_b, categorical_column_c],
            dimension=2)

  def test_weighted_categorical_column_ok(self):
    # [FUNCTION DOC]
    # Asserts that weighted categorical columns (alone or mixed with plain ones) are accepted
    # without error.
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=3)
      weighted_categorical_column_a = fc.weighted_categorical_column(
          categorical_column_a, weight_feature_key='aaa_weights')
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=3)
      weighted_categorical_column_b = fc.weighted_categorical_column(
          categorical_column_b, weight_feature_key='bbb_weights')
      fc.shared_embedding_columns_v2(
          [weighted_categorical_column_a, categorical_column_b], dimension=2)
      fc.shared_embedding_columns_v2(
          [categorical_column_a, weighted_categorical_column_b], dimension=2)
      fc.shared_embedding_columns_v2(
          [weighted_categorical_column_a, weighted_categorical_column_b],
          dimension=2)

  def test_parse_example(self):
    # [FUNCTION DOC]
    # Parses a serialized example with two vocabulary list features and asserts each is returned
    # as the expected sparse string tensor.
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      a = fc.categorical_column_with_vocabulary_list(
          key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
      b = fc.categorical_column_with_vocabulary_list(
          key='bbb', vocabulary_list=('omar', 'stringer', 'marlo'))
      a_embedded, b_embedded = fc.shared_embedding_columns_v2([a, b],
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
          features=fc.make_parse_example_spec_v2([a_embedded, b_embedded]))
      self.assertIn('aaa', features)
      self.assertIn('bbb', features)

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
              values=np.array([b'stringer', b'marlo'], dtype=np.object_),
              dense_shape=[1, 2]), self.evaluate(features['bbb']))

  def test_transform_feature(self):
    # [FUNCTION DOC]
    # Asserts the transformation of each shared embedding column equals the transformation of its
    # own categorical column.
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      a = fc.categorical_column_with_identity(key='aaa', num_buckets=3)
      b = fc.categorical_column_with_identity(key='bbb', num_buckets=3)
      a_embedded, b_embedded = fc.shared_embedding_columns_v2([a, b],
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
      outputs = fc._transform_features_v2(features,
                                          [a, a_embedded, b, b_embedded], None)
      output_a = outputs[a]
      output_a_embedded = outputs[a_embedded]
      output_b = outputs[b]
      output_b_embedded = outputs[b_embedded]

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

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
    # Parameterized by whether safe embedding lookup is used and whether variables are
    # partitioned. Dense integer inputs with padding of minus one, a shared embedding table over
    # four ids.
    # Asserts the variable names (single or two partitions), the table contents, the lookup
    # results for both columns (rows with no ids are zero only when safe lookup is used, otherwise
    # they are dropped), and that the graph contains the fill-empty-rows operation only when safe
    # lookup is enabled.
    # SharedEmbeddingColumns are graph-only
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
        # HELPER: Local embedding initializer used by the test.
        # Asserts float32 dtype; when partitioned requires partition info with the full shape and
        # slices the fixed table for the partition's offset, otherwise requires the full shape and
        # no partition info and returns the whole fixed table.
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
      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=vocabulary_size)
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=vocabulary_size)

      partitioner = None
      if partition_variables:
        partitioner = partitioned_variables.fixed_size_partitioner(2, axis=0)

      with variable_scope.variable_scope('vars', partitioner=partitioner):
        embedding_column_a, embedding_column_b = fc.shared_embedding_columns_v2(
            [categorical_column_a, categorical_column_b],
            dimension=embedding_dimension,
            initializer=_initializer,
            use_safe_embedding_lookup=use_safe_embedding_lookup)
        # Provide sparse input and get dense result.
        embedding_lookup_a = embedding_column_a.get_dense_tensor(
            fc.FeatureTransformationCache(input_features), None)
        embedding_lookup_b = embedding_column_b.get_dense_tensor(
            fc.FeatureTransformationCache(input_features), None)

      # Assert expected embedding variable and lookups.
      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      if partition_variables:
        self.assertCountEqual(('vars/aaa_bbb_shared_embedding/part_0:0',
                               'vars/aaa_bbb_shared_embedding/part_1:0'),
                              tuple([v.name for v in global_vars]))
        embedding_var = array_ops.concat(global_vars, axis=0)
      else:
        self.assertCountEqual(('vars/aaa_bbb_shared_embedding:0',),
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

  @parameterized.named_parameters(
      {
          'testcase_name': 'use_safe_embedding_lookup',
          'use_safe_embedding_lookup': True
      }, {
          'testcase_name': 'dont_use_safe_embedding_lookup',
          'use_safe_embedding_lookup': False
      })
  def test_get_dense_tensor_valid(self, use_safe_embedding_lookup):
    # [FUNCTION DOC]
    # Parameterized by safe lookup use. Dense inputs with only valid ids in some rows; asserts the
    # shared variable name, its contents, both columns' mean-combined lookups, and presence or
    # absence of the fill-empty-rows operation.
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      # Inputs.
      vocabulary_size = 3
      # -1 values are ignored.
      input_a = np.array([
          [2, 1],  # example 0, ids [2, 1]
          [0, -1]
      ])  # example 1, ids [0]
      input_b = np.array([
          [1, -1],  # example 0, ids [1]
          [1, 2]
      ])  # example 1, ids [1, 2]
      input_features = {'aaa': input_a, 'bbb': input_b}

      # Embedding variable.
      embedding_dimension = 2
      embedding_values = (
          (1., 2.),  # id 0
          (3., 5.),  # id 1
          (7., 11.)  # id 2
      )

      def _initializer(shape, dtype, partition_info=None):
        # [FUNCTION DOC]
        # HELPER: Local embedding initializer used by the test.
        # Asserts the shape, float32 dtype and absence of partition info, then returns the fixed
        # embedding table.
        self.assertAllEqual((vocabulary_size, embedding_dimension), shape)
        self.assertEqual(dtypes.float32, dtype)
        self.assertIsNone(partition_info)
        return embedding_values

      # Expected lookup result, using combiner='mean'.
      expected_lookups_a = (
          # example 0:
          (5., 8.),  # ids [2, 1], embedding =  mean([3, 5] + [7, 11]) = [5, 8]
          # example 1:
          (1., 2),  # ids [0], embedding = [1, 2]
      )
      expected_lookups_b = (
          # example 0:
          (3., 5.),  # ids [1], embedding = [3, 5]
          # example 1:
          (5., 8.),  # ids [1, 2], embedding = mean([3, 5] + [7, 11]) = [5, 8]
      )

      # Build columns.
      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=vocabulary_size)
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=vocabulary_size)
      embedding_column_a, embedding_column_b = fc.shared_embedding_columns_v2(
          [categorical_column_a, categorical_column_b],
          dimension=embedding_dimension,
          initializer=_initializer,
          use_safe_embedding_lookup=use_safe_embedding_lookup)

      # Provide sparse input and get dense result.
      embedding_lookup_a = embedding_column_a.get_dense_tensor(
          fc.FeatureTransformationCache(input_features), None)
      embedding_lookup_b = embedding_column_b.get_dense_tensor(
          fc.FeatureTransformationCache(input_features), None)

      # Assert expected embedding variable and lookups.
      global_vars = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
      self.assertCountEqual(('aaa_bbb_shared_embedding:0',),
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

  @parameterized.named_parameters(
      {
          'testcase_name': 'use_safe_embedding_lookup',
          'use_safe_embedding_lookup': True
      }, {
          'testcase_name': 'dont_use_safe_embedding_lookup',
          'use_safe_embedding_lookup': False
      })
  def test_get_dense_tensor_placeholder_inputs(self, use_safe_embedding_lookup):
    # [FUNCTION DOC]
    # Parameterized by safe lookup use. Dense inputs are placeholders with a fixed second
    # dimension; builds the lookups, asserts the presence or absence of the fill-empty-rows
    # operation, then runs both lookups with fed data inside an initialized session (no value
    # assertions on the result).
    # SharedEmbeddingColumns are graph-only
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

      def _initializer(shape, dtype, partition_info=None):
        # [FUNCTION DOC]
        # HELPER: Local embedding initializer used by the test.
        # Asserts the shape, float32 dtype and absence of partition info, then returns the fixed
        # embedding table.
        self.assertAllEqual((vocabulary_size, embedding_dimension), shape)
        self.assertEqual(dtypes.float32, dtype)
        self.assertIsNone(partition_info)
        return embedding_values

      # Build columns.
      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=vocabulary_size)
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=vocabulary_size)
      embedding_column_a, embedding_column_b = fc.shared_embedding_columns_v2(
          [categorical_column_a, categorical_column_b],
          dimension=embedding_dimension,
          initializer=_initializer,
          use_safe_embedding_lookup=use_safe_embedding_lookup)

      # Provide sparse input and get dense result.
      embedding_lookup_a = embedding_column_a.get_dense_tensor(
          fc.FeatureTransformationCache(input_features), None)
      embedding_lookup_b = embedding_column_b.get_dense_tensor(
          fc.FeatureTransformationCache(input_features), None)
      if use_safe_embedding_lookup:
        self.assertIn(
            'SparseFillEmptyRows',
            [x.type for x in ops.get_default_graph().get_operations()])
      else:
        self.assertNotIn(
            'SparseFillEmptyRows',
            [x.type for x in ops.get_default_graph().get_operations()])

      with _initialized_session() as sess:
        sess.run([embedding_lookup_a, embedding_lookup_b], feed_dict=feed_dict)

  def test_serialization(self):
    # [FUNCTION DOC]
    # Creates two shared embedding columns with a custom initializer and asserts each column's
    # parents are its own categorical column. Configuration round trip is not tested (a to-do note
    # says it is not yet implemented).
    # SharedEmbeddingColumns are graph-only
    with ops.Graph().as_default():
      def _initializer(shape, dtype, partition_info=None):
        # [FUNCTION DOC]
        # HELPER: Local initializer used by the serialization test.
        # Ignores its arguments and returns a value error object (returns rather than raises);
        # never invoked in the test.
        del shape, dtype, partition_info
        return ValueError('Not expected to be called')

      categorical_column_a = fc.categorical_column_with_identity(
          key='aaa', num_buckets=3)
      categorical_column_b = fc.categorical_column_with_identity(
          key='bbb', num_buckets=3)
      embedding_column_a, embedding_column_b = fc.shared_embedding_columns_v2(
          [categorical_column_a, categorical_column_b],
          dimension=2,
          initializer=_initializer)

      self.assertEqual([categorical_column_a], embedding_column_a.parents)
      self.assertEqual([categorical_column_b], embedding_column_b.parents)
      # TODO(rohanj): Add tests for (from|get)_config once implemented


class WeightedCategoricalColumnTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the weighted categorical column: defaults, version flag, deep copy, dtype and
  # name validation, missing weights, parsing, transformation with sparse and dense inputs and
  # weights, legacy linear model behavior including shape mismatches, and serialization.

  def test_defaults(self):
    # [FUNCTION DOC]
    # Creates a weighted column over an identity column and asserts name (ids weighted by values),
    # bucket count, parse spec containing both keys with int64 and float32 specs, and version
    # flag.
    column = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    self.assertEqual('ids_weighted_by_values', column.name)
    self.assertEqual(3, column.num_buckets)
    self.assertEqual({
        'ids': parsing_ops.VarLenFeature(dtypes.int64),
        'values': parsing_ops.VarLenFeature(dtypes.float32)
    }, column.parse_example_spec)
    self.assertTrue(column._is_v2_column)

  def test_is_v2_column(self):
    # [FUNCTION DOC]
    # Asserts a weighted column over a legacy categorical column is not flagged version 2.
    column = fc.weighted_categorical_column(
        categorical_column=fc_old._categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    self.assertFalse(column._is_v2_column)

  def test_deep_copy(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The docstring says it tests deep copy of the hash bucket categorical column,
    # but the test actually builds and copies a weighted categorical column over an identity
    # column.
    # Asserts the original and its deep copy keep name, bucket count and parse spec.
    """Tests deepcopy of categorical_column_with_hash_bucket."""
    original = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    for column in (original, copy.deepcopy(original)):
      self.assertEqual('ids_weighted_by_values', column.name)
      self.assertEqual(3, column.num_buckets)
      self.assertEqual({
          'ids': parsing_ops.VarLenFeature(dtypes.int64),
          'values': parsing_ops.VarLenFeature(dtypes.float32)
      }, column.parse_example_spec)

  def test_invalid_dtype_none(self):
    # [FUNCTION DOC]
    # Asserts a missing weight dtype raises a value error saying it is not convertible to float.
    with self.assertRaisesRegex(ValueError, 'is not convertible to float'):
      fc.weighted_categorical_column(
          categorical_column=fc.categorical_column_with_identity(
              key='ids', num_buckets=3),
          weight_feature_key='values',
          dtype=None)

  def test_invalid_dtype_string(self):
    # [FUNCTION DOC]
    # Asserts a string weight dtype raises the same error.
    with self.assertRaisesRegex(ValueError, 'is not convertible to float'):
      fc.weighted_categorical_column(
          categorical_column=fc.categorical_column_with_identity(
              key='ids', num_buckets=3),
          weight_feature_key='values',
          dtype=dtypes.string)

  def test_invalid_input_dtype(self):
    # [FUNCTION DOC]
    # Asserts transforming string inputs for both ids and weights raises a value error about a bad
    # dtype.
    column = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    strings = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('omar', 'stringer', 'marlo'),
        dense_shape=(2, 2))
    with self.assertRaisesRegex(ValueError, 'Bad dtype'):
      fc._transform_features_v2({
          'ids': strings,
          'values': strings
      }, (column,), None)

  def test_column_name_collision(self):
    # [FUNCTION DOC]
    # Asserts that using the same key for the categorical column and weights raises a value error
    # that a parse config already exists, when the parse spec is accessed.
    with self.assertRaisesRegex(ValueError, r'Parse config.*already exists'):
      fc.weighted_categorical_column(
          categorical_column=fc.categorical_column_with_identity(
              key='aaa', num_buckets=3),
          weight_feature_key='aaa').parse_example_spec()

  def test_missing_weights(self):
    # [FUNCTION DOC]
    # Asserts transforming without the weights feature raises a value error saying the weight key
    # is not in the features dictionary.
    column = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=('omar', 'stringer', 'marlo'),
        dense_shape=(2, 2))
    with self.assertRaisesRegex(ValueError,
                                'values is not in features dictionary'):
      fc._transform_features_v2({'ids': inputs}, (column,), None)

  def test_parse_example(self):
    # [FUNCTION DOC]
    # Parses a serialized example with ids and weights and asserts both sparse results with the
    # comparison helper.
    a = fc.categorical_column_with_vocabulary_list(
        key='aaa', vocabulary_list=('omar', 'stringer', 'marlo'))
    a_weighted = fc.weighted_categorical_column(a, weight_feature_key='weights')
    data = example_pb2.Example(
        features=feature_pb2.Features(
            feature={
                'aaa':
                    feature_pb2.Feature(
                        bytes_list=feature_pb2.BytesList(
                            value=[b'omar', b'stringer'])),
                'weights':
                    feature_pb2.Feature(
                        float_list=feature_pb2.FloatList(value=[1., 10.]))
            }))
    features = parsing_ops.parse_example(
        serialized=[data.SerializeToString()],
        features=fc.make_parse_example_spec_v2([a_weighted]))
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
    # Transforms sparse ids and sparse weights and asserts the output pair holds the ids as int64
    # and the weights as float32 with matching indices and shapes.
    column = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)), values=(0, 1, 0), dense_shape=(2, 2))
    weights = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=(0.5, 1.0, 0.1),
        dense_shape=(2, 2))
    id_tensor, weight_tensor = fc._transform_features_v2({
        'ids': inputs,
        'values': weights,
    }, (column,), None)[column]

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

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
    # Dense ids with a padding entry and sparse weights; asserts ids are converted to a sparse
    # tensor and weights are unchanged.
    column = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    weights = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)),
        values=(0.5, 1.0, 0.1),
        dense_shape=(2, 2))
    id_tensor, weight_tensor = fc._transform_features_v2({
        'ids': ((0, -1), (1, 0)),
        'values': weights,
    }, (column,), None)[column]

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

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
    # Sparse ids with dense weights; asserts the weights are converted to a sparse tensor aligned
    # with the ids.
    column = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    inputs = sparse_tensor.SparseTensorValue(
        indices=((0, 0), (1, 0), (1, 1)), values=(2, 1, 0), dense_shape=(2, 2))
    id_tensor, weight_tensor = fc._transform_features_v2({
        'ids': inputs,
        'values': ((.5, 0.), (1., .1)),
    }, (column,), None)[column]

    self.evaluate(variables_lib.global_variables_initializer())
    self.evaluate(lookup_ops.tables_initializer())

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

  def test_old_linear_model(self):
    # [FUNCTION DOC]
    # Legacy linear model with weighted ids: asserts zero initialization and, after assigning
    # weights, predictions equal weight-scaled sums (half and three point two).
    column = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    with ops.Graph().as_default():
      predictions = fc_old.linear_model({
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

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose((0.,), self.evaluate(bias))
      self.assertAllClose(((0.,), (0.,), (0.,)), self.evaluate(weight_var))
      self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
      self.evaluate(weight_var.assign(((1.,), (2.,), (3.,))))
      # weight_var[0] * weights[0, 0] = 1 * .5 = .5
      # weight_var[2] * weights[1, 0] + weight_var[1] * weights[1, 1]
      # = 3*1 + 2*.1 = 3+.2 = 3.2
      self.assertAllClose(((.5,), (3.2,)), self.evaluate(predictions))

  def test_old_linear_model_mismatched_shape(self):
    # [FUNCTION DOC]
    # Asserts weights with a different sparsity pattern than the ids raise a value error saying
    # the dimensions are not compatible.
    column = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    with ops.Graph().as_default():
      with self.assertRaisesRegex(ValueError,
                                  r'Dimensions.*are not compatible'):
        fc_old.linear_model({
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

  def test_old_linear_model_mismatched_dense_values(self):
    # [FUNCTION DOC]
    # Dense weights with the wrong width and the mean combiner; with constant folding disabled in
    # the session configuration (to get a stable message), asserts an op error saying shapes are
    # incompatible when evaluating.
    column = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    with ops.Graph().as_default():
      predictions = fc_old.linear_model({
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

  def test_old_linear_model_mismatched_dense_shape(self):
    # [FUNCTION DOC]
    # Dense weights with too many rows for the ids; asserts the model still builds and predictions
    # equal the expected weighted sums, i.e. the extra weight row is tolerated.
    column = fc.weighted_categorical_column(
        categorical_column=fc.categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    with ops.Graph().as_default():
      predictions = fc_old.linear_model({
          'ids':
              sparse_tensor.SparseTensorValue(
                  indices=((0, 0), (1, 0), (1, 1)),
                  values=(0, 2, 1),
                  dense_shape=(2, 2)),
          'values': ((.5,), (1.,), (.1,))
      }, (column,))
      bias = get_linear_model_bias()
      weight_var = get_linear_model_column_var(column)

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose((0.,), self.evaluate(bias))
      self.assertAllClose(((0.,), (0.,), (0.,)), self.evaluate(weight_var))
      self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
      self.evaluate(weight_var.assign(((1.,), (2.,), (3.,))))
      # weight_var[0] * weights[0, 0] = 1 * .5 = .5
      # weight_var[2] * weights[1, 0] + weight_var[1] * weights[1, 1]
      # = 3*1 + 2*.1 = 3+.2 = 3.2
      self.assertAllClose(((.5,), (3.2,)), self.evaluate(predictions))

  def test_old_linear_model_old_categorical(self):
    # [FUNCTION DOC]
    # Same as the basic weighted linear model test with a legacy categorical column under the
    # weighted column; asserts the same weights and predictions.
    column = fc.weighted_categorical_column(
        categorical_column=fc_old._categorical_column_with_identity(
            key='ids', num_buckets=3),
        weight_feature_key='values')
    with ops.Graph().as_default():
      predictions = fc_old.linear_model({
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

      self.evaluate(variables_lib.global_variables_initializer())
      self.evaluate(lookup_ops.tables_initializer())

      self.assertAllClose((0.,), self.evaluate(bias))
      self.assertAllClose(((0.,), (0.,), (0.,)), self.evaluate(weight_var))
      self.assertAllClose(((0.,), (0.,)), self.evaluate(predictions))
      self.evaluate(weight_var.assign(((1.,), (2.,), (3.,))))
      # weight_var[0] * weights[0, 0] = 1 * .5 = .5
      # weight_var[2] * weights[1, 0] + weight_var[1] * weights[1, 1]
      # = 3*1 + 2*.1 = 3+.2 = 3.2
      self.assertAllClose(((.5,), (3.2,)), self.evaluate(predictions))

  # TODO(ptucker): Add test with embedding of weighted categorical.

  def test_serialization(self):
    # [FUNCTION DOC]
    # Asserts the parents are the categorical column and the weight key, the configuration
    # contains the nested categorical column, dtype and weight key, and rebuilding gives an equal
    # column; also rebuilds with an existing column mapping and asserts the same categorical
    # column object is reused.
    categorical_column = fc.categorical_column_with_identity(
        key='ids', num_buckets=3)
    column = fc.weighted_categorical_column(
        categorical_column=categorical_column, weight_feature_key='weight')

    self.assertEqual([categorical_column, 'weight'], column.parents)

    config = column.get_config()
    self.assertEqual({
        'categorical_column': {
            'config': {
                'key': 'ids',
                'number_buckets': 3,
                'default_value': None
            },
            'class_name': 'IdentityCategoricalColumn'
        },
        'dtype': 'float32',
        'weight_feature_key': 'weight'
    }, config)

    self.assertEqual(column, fc.WeightedCategoricalColumn.from_config(config))

    new_column = fc.WeightedCategoricalColumn.from_config(
        config,
        columns_by_name={
            serialization._column_name_with_class_name(categorical_column):
                categorical_column
        })
    self.assertEqual(column, new_column)
    self.assertIs(categorical_column, new_column.categorical_column)

if __name__ == '__main__':
  test.main()
