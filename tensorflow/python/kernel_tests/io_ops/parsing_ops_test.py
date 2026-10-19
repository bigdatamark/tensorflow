# [MODULE DOC]
# Unit tests for the example-parsing and raw-decoding operations in the parsing ops module:
# parse_example, parse_single_example, parse_sequence_example, parse_single_sequence_example,
# decode_raw (both versions), decode_json_example and parse_tensor.
# The file defines a few module-level helpers (short aliases that build Example, Feature, Features,
# FeatureList, FeatureLists and SequenceExample protocol buffer messages, a flatten helper and a
# comparison helper) and six test classes, one per operation family. Most classes run every test in
# both graph mode and eager mode.
# Importing the module has no side effects beyond creating the aliases; running it as a script
# starts the standard TensorFlow test runner.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.parsing_ops."""

import copy
import itertools

import numpy as np
import sys

from google.protobuf import json_format

from tensorflow.core.example import example_pb2
from tensorflow.core.example import feature_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops.ragged import ragged_concat_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging

# Helpers for creating Example objects
example = example_pb2.Example
feature = feature_pb2.Feature
features = lambda d: feature_pb2.Features(feature=d)
bytes_feature = lambda v: feature(bytes_list=feature_pb2.BytesList(value=v))
int64_feature = lambda v: feature(int64_list=feature_pb2.Int64List(value=v))
float_feature = lambda v: feature(float_list=feature_pb2.FloatList(value=v))
# Helpers for creating SequenceExample objects
feature_list = lambda l: feature_pb2.FeatureList(feature=l)
feature_lists = lambda d: feature_pb2.FeatureLists(feature_list=d)
sequence_example = example_pb2.SequenceExample


def flatten(list_of_lists):
  # [FUNCTION DOC]
  # HELPER: Small utility shared by tests for combining nested sequences.
  # Takes a sequence of sequences (list_of_lists) and returns a lazy iterator that yields the items of
  # each inner sequence in order, removing exactly one level of nesting. Does not copy the input and
  # has no side effects.
  """Flatten one level of nesting."""
  return itertools.chain.from_iterable(list_of_lists)


def _compare_output_to_expected(tester, actual, expected):
  # [FUNCTION DOC]
  # HELPER: Shared assertion helper used by the parse-example, parse-single-example and
  # HELPER: parse-sequence-example test classes to check parser output.
  # Compares a mapping of actual parser outputs (actual) against a mapping of expected values
  # (expected) using the test case (tester) for assertions. First asserts that both mappings have the
  # same set of keys. Then, for every key, logs the key and compares: a sparse tensor output must be
  # matched by a tuple of exactly three items (indices, values, dense shape) which are each compared
  # for equality with the corresponding sparse tensor component; any other output (dense or ragged)
  # is compared directly for equality with the expected value.
  # Raises test assertion failures on any mismatch; returns nothing.
  tester.assertEqual(set(actual.keys()), set(expected.keys()))
  for k, v in actual.items():
    expected_v = expected[k]
    tf_logging.info("Comparing key: %s", k)
    if isinstance(v, sparse_tensor.SparseTensor):
      tester.assertTrue(isinstance(expected_v, tuple))
      tester.assertLen(expected_v, 3)
      tester.assertAllEqual(v.indices, expected_v[0])
      tester.assertAllEqual(v.values, expected_v[1])
      tester.assertAllEqual(v.dense_shape, expected_v[2])
    else:
      tester.assertAllEqual(v, expected_v)


@test_util.run_all_in_graph_and_eager_modes
class ParseExampleTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the batch example parser. Covers sparse (variable-length) features, sparse features
  # built from index and value keys (including reuse of keys, three-dimensional indices and unsorted
  # indices), dense fixed-length features with and without defaults, variable-length dense features
  # (sequence features with padding), ragged features with every partition kind (row splits, lengths,
  # starts, limits, value row ids, uniform row length and nested partitions), and the expected error
  # messages for malformed input. Decorated so that each test runs in both graph and eager mode.
  # The shared fixture is the internal comparison method, which runs the parser, checks the outputs
  # and checks the static shapes.

  def _test(self, kwargs, expected_values=None, expected_err=None):
    # [FUNCTION DOC]
    # HELPER: Shared driver used by every test in this class.
    # Runs the batch example parser with the keyword arguments given in kwargs (a mapping that holds
    # the serialized examples, the feature specification and optionally example names).
    # If expected_err (a pair of an error type and a message pattern) is given: in graph mode the
    # parse is evaluated and must raise that error type with a matching message; in eager mode it must
    # raise some exception (any type, the message is not checked). The method then returns without any
    # shape checks.
    # Otherwise the output is compared against expected_values (a mapping from feature name to either a
    # dense value, a ragged value or an indices/values/shape tuple for sparse features). It then checks
    # shapes: a fixed-length feature with a known shape must have the batch size followed by that shape
    # (the batch size is taken from the serialized input, evaluating it if it is a tensor); a
    # variable-length feature must have rank-two indices, rank-one values and rank-one dense shape
    # (compatible with unknown sizes in eager mode, exactly unknown sizes in graph mode).
    if expected_err:
      if not context.executing_eagerly():
        with self.assertRaisesWithPredicateMatch(expected_err[0],
                                                 expected_err[1]):
          self.evaluate(parsing_ops.parse_example(**kwargs))
      else:
        with self.assertRaises(Exception):
          parsing_ops.parse_example(**kwargs)
      return
    else:
      out = parsing_ops.parse_example(**kwargs)
      _compare_output_to_expected(self, out, expected_values)

    # Check shapes; if serialized is a Tensor we need its size to
    # properly check.
    serialized = kwargs["serialized"]
    batch_size = (
        self.evaluate(serialized).size
        if isinstance(serialized, tensor_lib.Tensor)
        else np.asarray(serialized).size
    )
    for k, f in kwargs["features"].items():
      if isinstance(f, parsing_ops.FixedLenFeature) and f.shape is not None:
        self.assertEqual(tuple(out[k].shape.as_list()), (batch_size,) + f.shape)
      elif isinstance(f, parsing_ops.VarLenFeature):
        if context.executing_eagerly():
          out[k].indices.shape.assert_is_compatible_with([None, 2])
          out[k].values.shape.assert_is_compatible_with([None])
          out[k].dense_shape.shape.assert_is_compatible_with([2])
        else:
          self.assertEqual(out[k].indices.shape.as_list(), [None, 2])
          self.assertEqual(out[k].values.shape.as_list(), [None])
          self.assertEqual(out[k].dense_shape.shape.as_list(), [2])

  def testEmptySerializedWithAllDefaults(self):
    # [FUNCTION DOC]
    # Scenario: two empty serialized examples are parsed with one variable-length feature and three
    # fixed-length features that all have default values (including a feature whose name contains a
    # colon). Expects an empty sparse result with batch size two and every dense feature filled by its
    # default value repeated for each example.
    sparse_name = "st_a"
    a_name = "a"
    b_name = "b"
    c_name = "c:has_a_tricky_name"
    a_default = [0, 42, 0]
    b_default = np.random.rand(3, 3).astype(bytes)
    c_default = np.random.rand(2).astype(np.float32)

    expected_st_a = (  # indices, values, shape
        np.empty((0, 2), dtype=np.int64),  # indices
        np.empty((0,), dtype=np.int64),  # sp_a is DT_INT64
        np.array([2, 0], dtype=np.int64))  # batch == 2, max_elems = 0

    expected_output = {
        sparse_name: expected_st_a,
        a_name: np.array(2 * [[a_default]]),
        b_name: np.array(2 * [b_default]),
        c_name: np.array(2 * [c_default]),
    }

    self._test(
        {
            "example_names": np.empty((0,), dtype=bytes),
            "serialized": ops.convert_to_tensor(["", ""]),
            "features": {
                sparse_name:
                    parsing_ops.VarLenFeature(dtypes.int64),
                a_name:
                    parsing_ops.FixedLenFeature(
                        (1, 3), dtypes.int64, default_value=a_default),
                b_name:
                    parsing_ops.FixedLenFeature(
                        (3, 3), dtypes.string, default_value=b_default),
                c_name:
                    parsing_ops.FixedLenFeature(
                        (2,), dtypes.float32, default_value=c_default),
            }
        }, expected_output)

  def testEmptySerializedWithoutDefaultsShouldFail(self):
    # [FUNCTION DOC]
    # Scenario: a fixed-length feature without a default value is requested while the input lacks it.
    # First case: the key is present but its feature is empty. Second case: both serialized examples are
    # empty. In both cases an op error is expected whose message says the named feature of float type
    # is required, reported for the first example name.
    input_features = {
        "st_a":
            parsing_ops.VarLenFeature(dtypes.int64),
        "a":
            parsing_ops.FixedLenFeature((1, 3),
                                        dtypes.int64,
                                        default_value=[0, 42, 0]),
        "b":
            parsing_ops.FixedLenFeature(
                (3, 3),
                dtypes.string,
                default_value=np.random.rand(3, 3).astype(bytes)),
        # Feature "c" is missing a default, this gap will cause failure.
        "c":
            parsing_ops.FixedLenFeature((2,), dtype=dtypes.float32),
    }

    # Edge case where the key is there but the feature value is empty
    original = example(features=features({"c": feature()}))
    self._test(
        {
            "example_names": ["in1"],
            "serialized": [original.SerializeToString()],
            "features": input_features,
        },
        expected_err=(
            errors_impl.OpError,
            "Name: in1, Feature: c \\(data type: float\\) is required"))

    # Standard case of missing key and value.
    self._test(
        {
            "example_names": ["in1", "in2"],
            "serialized": ["", ""],
            "features": input_features,
        },
        expected_err=(
            errors_impl.OpError,
            "Name: in1, Feature: c \\(data type: float\\) is required"))

  def testDenseNotMatchingShapeShouldFail(self):
    # [FUNCTION DOC]
    # Scenario: two examples hold float lists of length three and two, but the fixed-length feature
    # declares a one-by-three shape. Expects an op error naming the failing example, the key and the
    # index of the offending example, complaining about the number of float values.
    original = [
        example(features=features({
            "a": float_feature([1, 1, 3]),
        })),
        example(features=features({
            "a": float_feature([-1, -1]),
        }))
    ]

    names = ["passing", "failing"]
    serialized = [m.SerializeToString() for m in original]

    self._test(
        {
            "example_names": names,
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
                "a": parsing_ops.FixedLenFeature((1, 3), dtypes.float32)
            }
        },
        expected_err=(errors_impl.OpError,
                      "Name: failing, Key: a, Index: 1.  Number of float val"))

  def testDenseDefaultNoShapeShouldFail(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The name says the default is missing, but the test actually passes a fixed-length
    # KNOWN ISSUE: feature with no shape (and no default); the failure being tested is the missing shape.
    # Scenario: a fixed-length feature is declared with an undefined shape. Expects a value error that
    # says the shape is missing for feature a.
    original = [
        example(features=features({
            "a": float_feature([1, 1, 3]),
        })),
    ]

    serialized = [m.SerializeToString() for m in original]

    self._test(
        {
            "example_names": ["failing"],
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
                "a": parsing_ops.FixedLenFeature(None, dtypes.float32)
            }
        },
        expected_err=(ValueError, "Missing shape for feature a"))

  def testSerializedContainingSparse(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: Comments in the expected values say the batch size is two, but there are four
    # KNOWN ISSUE: examples and the expected dense shapes use four rows.
    # Scenario: four examples where the sparse float feature is present, empty, absent or combined with
    # a string sparse feature whose entry is an empty feature. Expects, per feature, the indices,
    # values and dense shape of sparse results (rows are examples, columns are positions within the
    # list, the column count is the longest list).
    original = [
        example(features=features({"st_c": float_feature([3, 4])})),
        example(
            features=features({
                "st_c": float_feature([]),  # empty float list
            })),
        example(
            features=features({
                "st_d": feature(),  # feature with nothing in it
            })),
        example(
            features=features({
                "st_c": float_feature([1, 2, -1]),
                "st_d": bytes_feature([b"hi"])
            }))
    ]

    serialized = [m.SerializeToString() for m in original]

    expected_st_c = (  # indices, values, shape
        np.array([[0, 0], [0, 1], [3, 0], [3, 1], [3, 2]], dtype=np.int64),
        np.array([3.0, 4.0, 1.0, 2.0, -1.0], dtype=np.float32),
        np.array([4, 3], dtype=np.int64))  # batch == 2, max_elems = 3

    expected_st_d = (  # indices, values, shape
        np.array([[3, 0]], dtype=np.int64), np.array(["hi"], dtype=bytes),
        np.array([4, 1], dtype=np.int64))  # batch == 2, max_elems = 1

    expected_output = {
        "st_c": expected_st_c,
        "st_d": expected_st_d,
    }

    self._test(
        {
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
                "st_c": parsing_ops.VarLenFeature(dtypes.float32),
                "st_d": parsing_ops.VarLenFeature(dtypes.string)
            }
        }, expected_output)

  def testSerializedContainingSparseFeature(self):
    # [FUNCTION DOC]
    # Scenario: a sparse feature built from an index key and a value key with declared size thirteen.
    # Examples include empty lists, a value feature with nothing in it with a missing index feature, and
    # unsorted indices. Expects indices sorted per row, values reordered accordingly, and dense shape
    # of batch size by thirteen.
    original = [
        example(
            features=features({
                "val": float_feature([3, 4]),
                "idx": int64_feature([5, 10])
            })),
        example(
            features=features({
                "val": float_feature([]),  # empty float list
                "idx": int64_feature([])
            })),
        example(
            features=features({
                "val": feature(),  # feature with nothing in it
                # missing idx feature
            })),
        example(
            features=features({
                "val": float_feature([1, 2, -1]),
                "idx":
                    int64_feature([0, 9, 3])  # unsorted
            }))
    ]

    serialized = [m.SerializeToString() for m in original]

    expected_sp = (  # indices, values, shape
        np.array([[0, 5], [0, 10], [3, 0], [3, 3], [3, 9]], dtype=np.int64),
        np.array([3.0, 4.0, 1.0, -1.0, 2.0], dtype=np.float32),
        np.array([4, 13], dtype=np.int64))  # batch == 4, max_elems = 13

    expected_output = {
        "sp": expected_sp,
    }

    self._test(
        {
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
                "sp":
                    parsing_ops.SparseFeature(["idx"], "val", dtypes.float32,
                                              [13])
            }
        }, expected_output)

  def testSerializedContainingSparseFeatureReuse(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The comment on the second expected sparse result claims the maximum number of
    # KNOWN ISSUE: elements is thirteen, but that feature has size seven and its dense shape says seven.
    # Scenario: two sparse features share the same index key but use different value keys and sizes
    # (thirteen, and seven with the already-sorted flag set). Expects both results to be built from the
    # same indices with their own values and sizes.
    original = [
        example(
            features=features({
                "val1": float_feature([3, 4]),
                "val2": float_feature([5, 6]),
                "idx": int64_feature([5, 10])
            })),
        example(
            features=features({
                "val1": float_feature([]),  # empty float list
                "idx": int64_feature([])
            })),
    ]

    serialized = [m.SerializeToString() for m in original]

    expected_sp1 = (  # indices, values, shape
        np.array([[0, 5], [0, 10]],
                 dtype=np.int64), np.array([3.0, 4.0], dtype=np.float32),
        np.array([2, 13], dtype=np.int64))  # batch == 2, max_elems = 13

    expected_sp2 = (  # indices, values, shape
        np.array([[0, 5], [0, 10]],
                 dtype=np.int64), np.array([5.0, 6.0], dtype=np.float32),
        np.array([2, 7], dtype=np.int64))  # batch == 2, max_elems = 13

    expected_output = {
        "sp1": expected_sp1,
        "sp2": expected_sp2,
    }

    self._test(
        {
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
                "sp1":
                    parsing_ops.SparseFeature("idx", "val1", dtypes.float32,
                                              13),
                "sp2":
                    parsing_ops.SparseFeature(
                        "idx",
                        "val2",
                        dtypes.float32,
                        size=7,
                        already_sorted=True)
            }
        }, expected_output)

  def testSerializedContaining3DSparseFeature(self):
    # [FUNCTION DOC]
    # Scenario: a sparse feature with two index keys and a three-dimensional size (thirteen by three).
    # Examples include empty, missing and unsorted indices. Expects three-column indices ordered by
    # row and then by index, matching values, and dense shape of batch size by thirteen by three.
    original = [
        example(
            features=features({
                "val": float_feature([3, 4]),
                "idx0": int64_feature([5, 10]),
                "idx1": int64_feature([0, 2]),
            })),
        example(
            features=features({
                "val": float_feature([]),  # empty float list
                "idx0": int64_feature([]),
                "idx1": int64_feature([]),
            })),
        example(
            features=features({
                "val": feature(),  # feature with nothing in it
                # missing idx feature
            })),
        example(
            features=features({
                "val": float_feature([1, 2, -1]),
                "idx0": int64_feature([0, 9, 3]),  # unsorted
                "idx1": int64_feature([1, 0, 2]),
            }))
    ]

    serialized = [m.SerializeToString() for m in original]

    expected_sp = (
        # indices
        np.array([[0, 5, 0], [0, 10, 2], [3, 0, 1], [3, 3, 2], [3, 9, 0]],
                 dtype=np.int64),
        # values
        np.array([3.0, 4.0, 1.0, -1.0, 2.0], dtype=np.float32),
        # shape batch == 4, max_elems = 13
        np.array([4, 13, 3], dtype=np.int64))

    expected_output = {
        "sp": expected_sp,
    }

    self._test(
        {
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
                "sp":
                    parsing_ops.SparseFeature(["idx0", "idx1"], "val",
                                              dtypes.float32, [13, 3])
            }
        }, expected_output)

  def testSerializedContainingDense(self):
    # [FUNCTION DOC]
    # Scenario: two examples each containing a float feature (two values) and a string feature (one
    # value; one name contains special characters). Fixed-length features with shapes (one, two, one)
    # and (one, one, one, one) and no defaults are requested. Expects dense outputs reshaped to the batch
    # size followed by these shapes.
    aname = "a"
    bname = "b*has+a:tricky_name"
    original = [
        example(
            features=features({
                aname: float_feature([1, 1]),
                bname: bytes_feature([b"b0_str"]),
            })),
        example(
            features=features({
                aname: float_feature([-1, -1]),
                bname: bytes_feature([b""]),
            }))
    ]

    serialized = [m.SerializeToString() for m in original]

    # pylint: disable=too-many-function-args
    expected_output = {
        aname:
            np.array([[1, 1], [-1, -1]], dtype=np.float32).reshape(2, 1, 2, 1),
        bname:
            np.array(["b0_str", ""], dtype=bytes).reshape(2, 1, 1, 1, 1),
    }
    # pylint: enable=too-many-function-args

    # No defaults, values required
    self._test(
        {
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
                aname:
                    parsing_ops.FixedLenFeature(
                        (1, 2, 1), dtype=dtypes.float32),
                bname:
                    parsing_ops.FixedLenFeature(
                        (1, 1, 1, 1), dtype=dtypes.string),
            }
        }, expected_output)

  # This test is identical as the previous one except
  # for the creation of 'serialized'.
  def testSerializedContainingDenseWithConcat(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: A TODO inside the test notes that a feature appearing twice in the same serialized
    # KNOWN ISSUE: example should become an error in future; the test currently relies on the concatenated
    # KNOWN ISSUE: messages being merged with the last occurrence winning.
    # Same scenario as the previous dense test, except each serialized item is the concatenation of two
    # serialized example messages (so some features appear in both parts, which the parser merges).
    # Expects the same dense outputs, using the later values where a feature is repeated.
    aname = "a"
    bname = "b*has+a:tricky_name"
    # TODO(lew): Feature appearing twice should be an error in future.
    original = [
        (example(features=features({
            aname: float_feature([10, 10]),
        })),
         example(
             features=features({
                 aname: float_feature([1, 1]),
                 bname: bytes_feature([b"b0_str"]),
             }))),
        (
            example(features=features({
                bname: bytes_feature([b"b100"]),
            })),
            example(
                features=features({
                    aname: float_feature([-1, -1]),
                    bname: bytes_feature([b"b1"]),
                })),
        ),
    ]

    serialized = [
        m.SerializeToString() + n.SerializeToString() for (m, n) in original
    ]

    # pylint: disable=too-many-function-args
    expected_output = {
        aname:
            np.array([[1, 1], [-1, -1]], dtype=np.float32).reshape(2, 1, 2, 1),
        bname:
            np.array(["b0_str", "b1"], dtype=bytes).reshape(2, 1, 1, 1, 1),
    }
    # pylint: enable=too-many-function-args

    # No defaults, values required
    self._test(
        {
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
                aname:
                    parsing_ops.FixedLenFeature(
                        (1, 2, 1), dtype=dtypes.float32),
                bname:
                    parsing_ops.FixedLenFeature(
                        (1, 1, 1, 1), dtype=dtypes.string),
            }
        }, expected_output)

  def testSerializedContainingDenseScalar(self):
    # [FUNCTION DOC]
    # Scenario: a fixed-length feature with shape of one and default value of minus one; one example has
    # the value and the other is empty. Expects a two-by-one column with the value and the default.
    original = [
        example(features=features({
            "a": float_feature([1]),
        })),
        example(features=features({}))
    ]

    serialized = [m.SerializeToString() for m in original]

    expected_output = {
        "a":
            np.array([[1], [-1]], dtype=np.float32)  # 2x1 (column vector)
    }

    self._test(
        {
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
                "a":
                    parsing_ops.FixedLenFeature(
                        (1,), dtype=dtypes.float32, default_value=-1),
            }
        }, expected_output)

  def testSerializedContainingDenseWithDefaults(self):
    # [FUNCTION DOC]
    # Scenario: fixed-length float and string features with defaults, over three examples where the
    # features are present, absent or present but empty. Expects defaults to be used wherever a value
    # is missing or empty and the provided values otherwise, in the requested shapes.
    original = [
        example(features=features({
            "a": float_feature([1, 1]),
        })),
        example(features=features({
            "b": bytes_feature([b"b1"]),
        })),
        example(features=features({"b": feature()})),
    ]

    serialized = [m.SerializeToString() for m in original]

    # pylint: disable=too-many-function-args
    expected_output = {
        "a":
            np.array(
                [[1, 1], [3, -3], [3, -3]],
                dtype=np.float32).reshape(3, 1, 2, 1),
        "b":
            np.array(
                ["tmp_str", "b1", "tmp_str"],
                dtype=bytes).reshape(3, 1, 1, 1, 1),
    }
    # pylint: enable=too-many-function-args

    self._test(
        {
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
                "a":
                    parsing_ops.FixedLenFeature((1, 2, 1),
                                                dtype=dtypes.float32,
                                                default_value=[3.0, -3.0]),
                "b":
                    parsing_ops.FixedLenFeature((1, 1, 1, 1),
                                                dtype=dtypes.string,
                                                default_value="tmp_str"),
            }
        }, expected_output)

  def testSerializedContainingSparseAndSparseFeatureAndDenseWithNoDefault(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The comment on the expected sparse feature says batch is four, but only two examples
    # KNOWN ISSUE: are used and the dense shape says two.
    # Scenario: one request mixes a variable-length feature that is absent from the input, a sparse
    # feature built from index and value keys, two dense features filled from defaults and one dense
    # feature without default that is present in the input. Passes example names. Expects an empty
    # sparse result, the sparse feature results, defaults repeated for each example and the provided
    # values for the required feature.
    expected_st_a = (  # indices, values, shape
        np.empty((0, 2), dtype=np.int64),  # indices
        np.empty((0,), dtype=np.int64),  # sp_a is DT_INT64
        np.array([2, 0], dtype=np.int64))  # batch == 2, max_elems = 0
    expected_sp = (  # indices, values, shape
        np.array([[0, 0], [0, 3], [1, 7]],
                 dtype=np.int64), np.array(["a", "b", "c"], dtype="|S"),
        np.array([2, 13], dtype=np.int64))  # batch == 4, max_elems = 13

    original = [
        example(
            features=features({
                "c": float_feature([3, 4]),
                "val": bytes_feature([b"a", b"b"]),
                "idx": int64_feature([0, 3])
            })),
        example(
            features=features({
                "c": float_feature([1, 2]),
                "val": bytes_feature([b"c"]),
                "idx": int64_feature([7])
            }))
    ]

    names = ["in1", "in2"]
    serialized = [m.SerializeToString() for m in original]

    a_default = [1, 2, 3]
    b_default = np.random.rand(3, 3).astype(bytes)
    expected_output = {
        "st_a": expected_st_a,
        "sp": expected_sp,
        "a": np.array(2 * [[a_default]]),
        "b": np.array(2 * [b_default]),
        "c": np.array([[3, 4], [1, 2]], dtype=np.float32),
    }

    self._test(
        {
            "example_names": names,
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
                "st_a":
                    parsing_ops.VarLenFeature(dtypes.int64),
                "sp":
                    parsing_ops.SparseFeature("idx", "val", dtypes.string, 13),
                "a":
                    parsing_ops.FixedLenFeature(
                        (1, 3), dtypes.int64, default_value=a_default),
                "b":
                    parsing_ops.FixedLenFeature(
                        (3, 3), dtypes.string, default_value=b_default),
                # Feature "c" must be provided, since it has no default_value.
                "c":
                    parsing_ops.FixedLenFeature((2,), dtypes.float32),
            }
        },
        expected_output)

  def testSerializedContainingSparseAndSparseFeatureWithReuse(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: Comments on the expected values say batch is four, but two examples are used.
    # Scenario: the index key is requested both as a variable-length feature and as the index of a
    # sparse feature built from index and value keys. Expects both results from the same input: the
    # raw index values per row, and the value list reordered by the sorted index.
    expected_idx = (  # indices, values, shape
        np.array([[0, 0], [0, 1], [1, 0], [1, 1]],
                 dtype=np.int64), np.array([0, 3, 7, 1]),
        np.array([2, 2], dtype=np.int64))  # batch == 4, max_elems = 2

    expected_sp = (  # indices, values, shape
        np.array([[0, 0], [0, 3], [1, 1], [1, 7]],
                 dtype=np.int64), np.array(["a", "b", "d", "c"], dtype="|S"),
        np.array([2, 13], dtype=np.int64))  # batch == 4, max_elems = 13

    original = [
        example(
            features=features({
                "val": bytes_feature([b"a", b"b"]),
                "idx": int64_feature([0, 3])
            })),
        example(
            features=features({
                "val": bytes_feature([b"c", b"d"]),
                "idx": int64_feature([7, 1])
            }))
    ]

    names = ["in1", "in2"]
    serialized = [m.SerializeToString() for m in original]

    expected_output = {
        "idx": expected_idx,
        "sp": expected_sp,
    }

    self._test(
        {
            "example_names": names,
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
                "idx":
                    parsing_ops.VarLenFeature(dtypes.int64),
                "sp":
                    parsing_ops.SparseFeature(["idx"], "val", dtypes.string,
                                              [13]),
            }
        }, expected_output)

  def _testSerializedContainingVarLenDenseLargerBatch(self, batch_size):
    # [FUNCTION DOC]
    # HELPER: Parameterised body for the larger-batch variable-length dense test below; not itself a test.
    # Takes batch_size, the number of examples. Builds one integer and one list-of-two string feature per
    # example, then randomly (using the global random generator) drops the second, or both, string
    # entries in about a quarter of the examples after the first. Parses with variable-length sequence
    # features that allow missing values with default minus one and the text default. Expects the integer
    # column of shape batch size by one and a string matrix padded with the default value. Guards against
    # a historical bug where output copying assumed each parse buffer held a single example.
    # During parsing, data read from the serialized proto is stored in buffers.
    # For small batch sizes, a buffer will contain one minibatch entry.
    # For larger batch sizes, a buffer may contain several minibatch
    # entries.  This test identified a bug where the code that copied
    # data out of the buffers and into the output tensors assumed each
    # buffer only contained one minibatch entry.  The bug has since been fixed.
    truth_int = [i for i in range(batch_size)]
    truth_str = [[("foo%d" % i).encode(), ("bar%d" % i).encode()]
                 for i in range(batch_size)]

    expected_str = copy.deepcopy(truth_str)

    # Delete some intermediate entries.  (Skip the first entry, to ensure that
    # we have at least one entry with length 2, to get the expected padding.)
    for i in range(1, batch_size):
      col = 1
      if np.random.rand() < 0.25:
        # w.p. 25%, drop out the second entry
        expected_str[i][col] = b"default"
        col -= 1
        truth_str[i].pop()
      if np.random.rand() < 0.25:
        # w.p. 25%, drop out the second entry (possibly again)
        expected_str[i][col] = b"default"
        truth_str[i].pop()

    expected_output = {
        # Batch size batch_size, 1 time step.
        "a": np.array(truth_int, dtype=np.int64).reshape(batch_size, 1),
        # Batch size batch_size, 2 time steps.
        "b": np.array(expected_str, dtype="|S").reshape(batch_size, 2),
    }

    original = [
        example(
            features=features({
                "a": int64_feature([truth_int[i]]),
                "b": bytes_feature(truth_str[i])
            })) for i in range(batch_size)
    ]

    serialized = [m.SerializeToString() for m in original]

    self._test(
        {
            "serialized":
                ops.convert_to_tensor(serialized, dtype=dtypes.string),
            "features": {
                "a":
                    parsing_ops.FixedLenSequenceFeature(
                        shape=(),
                        dtype=dtypes.int64,
                        allow_missing=True,
                        default_value=-1),
                "b":
                    parsing_ops.FixedLenSequenceFeature(
                        shape=[],
                        dtype=dtypes.string,
                        allow_missing=True,
                        default_value="default"),
            }
        }, expected_output)

  def testSerializedContainingVarLenDenseLargerBatch(self):
    # [FUNCTION DOC]
    # Test entry point: seeds the numpy random generator (side effect on global random state) and runs
    # the larger-batch helper for batch sizes one, ten, twenty, one hundred and two hundred fifty-six.
    np.random.seed(3456)
    for batch_size in (1, 10, 20, 100, 256):
      self._testSerializedContainingVarLenDenseLargerBatch(batch_size)

  def testSerializedContainingVarLenDense(self):
    # [FUNCTION DOC]
    # Scenario: four examples with float lists, string lists and integer lists of varying lengths (or
    # missing) parsed with fixed-length sequence features that allow missing values. Expects results padded
    # to the longest example, a zero-width result for a feature absent everywhere, and that a custom default
    # changes the padding value. Also checks error cases: the number of values not being a multiple of
    # the inner shape (op error), a default of an empty list (value error about reshaping), a fixed-length
    # feature with unknown first dimension (value error suggesting the sequence feature), unknown
    # inner dimensions (value error) and a sequence feature with the allow-missing flag false (value error
    # about it being unsupported).
    aname = "a"
    bname = "b"
    cname = "c"
    dname = "d"
    example_names = ["in1", "in2", "in3", "in4"]
    original = [
        example(features=features({
            cname: int64_feature([2]),
        })),
        example(
            features=features({
                aname: float_feature([1, 1]),
                bname: bytes_feature([b"b0_str", b"b1_str"]),
            })),
        example(
            features=features({
                aname: float_feature([-1, -1, 2, 2]),
                bname: bytes_feature([b"b1"]),
            })),
        example(
            features=features({
                aname: float_feature([]),
                cname: int64_feature([3]),
            })),
    ]

    serialized = [m.SerializeToString() for m in original]

    # pylint: disable=too-many-function-args
    expected_output = {
        aname:
            np.array(
                [
                    [0, 0, 0, 0],
                    [1, 1, 0, 0],
                    [-1, -1, 2, 2],
                    [0, 0, 0, 0],
                ],
                dtype=np.float32).reshape(4, 2, 2, 1),
        bname:
            np.array(
                [["", ""], ["b0_str", "b1_str"], ["b1", ""], ["", ""]],
                dtype=bytes).reshape(4, 2, 1, 1, 1),
        cname:
            np.array([2, 0, 0, 3], dtype=np.int64).reshape(4, 1),
        dname:
            np.empty(shape=(4, 0), dtype=bytes),
    }
    # pylint: enable=too-many-function-args

    self._test(
        {
            "example_names": example_names,
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
                aname:
                    parsing_ops.FixedLenSequenceFeature(
                        (2, 1), dtype=dtypes.float32, allow_missing=True),
                bname:
                    parsing_ops.FixedLenSequenceFeature(
                        (1, 1, 1), dtype=dtypes.string, allow_missing=True),
                cname:
                    parsing_ops.FixedLenSequenceFeature(
                        shape=[], dtype=dtypes.int64, allow_missing=True),
                dname:
                    parsing_ops.FixedLenSequenceFeature(
                        shape=[], dtype=dtypes.string, allow_missing=True),
            }
        }, expected_output)

    # Test with padding values.
    expected_output_custom_padding = dict(expected_output)
    # pylint: disable=too-many-function-args
    expected_output_custom_padding[aname] = np.array(
        [
            [-2, -2, -2, -2],
            [1, 1, -2, -2],
            [-1, -1, 2, 2],
            [-2, -2, -2, -2],
        ],
        dtype=np.float32).reshape(4, 2, 2, 1)
    # pylint: enable=too-many-function-args

    self._test(
        {
            "example_names": example_names,
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
                aname:
                    parsing_ops.FixedLenSequenceFeature((2, 1),
                                                        dtype=dtypes.float32,
                                                        allow_missing=True,
                                                        default_value=-2.0),
                bname:
                    parsing_ops.FixedLenSequenceFeature(
                        (1, 1, 1), dtype=dtypes.string, allow_missing=True),
                cname:
                    parsing_ops.FixedLenSequenceFeature(
                        shape=[], dtype=dtypes.int64, allow_missing=True),
                dname:
                    parsing_ops.FixedLenSequenceFeature(
                        shape=[], dtype=dtypes.string, allow_missing=True),
            }
        }, expected_output_custom_padding)

    # Change number of required values so the inputs are not a
    # multiple of this size.
    self._test(
        {
            "example_names": example_names,
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
                aname:
                    parsing_ops.FixedLenSequenceFeature(
                        (2, 1), dtype=dtypes.float32, allow_missing=True),
                bname:
                    parsing_ops.FixedLenSequenceFeature(
                        (2, 1, 1), dtype=dtypes.string, allow_missing=True),
            }
        },
        expected_err=(
            errors_impl.OpError, "Name: in3, Key: b, Index: 2.  "
            "Number of bytes values is not a multiple of stride length."))

    self._test(
        {
            "example_names": example_names,
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
                aname:
                    parsing_ops.FixedLenSequenceFeature((2, 1),
                                                        dtype=dtypes.float32,
                                                        allow_missing=True,
                                                        default_value=[]),
                bname:
                    parsing_ops.FixedLenSequenceFeature(
                        (2, 1, 1), dtype=dtypes.string, allow_missing=True),
            }
        },
        expected_err=(ValueError,
                      "Cannot reshape a tensor with 0 elements to shape"))

    self._test(
        {
            "example_names": example_names,
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
                aname:
                    parsing_ops.FixedLenFeature(
                        (None, 2, 1), dtype=dtypes.float32),
                bname:
                    parsing_ops.FixedLenSequenceFeature(
                        (2, 1, 1), dtype=dtypes.string, allow_missing=True),
            }
        },
        expected_err=(ValueError,
                      "First dimension of shape for feature a unknown. "
                      "Consider using FixedLenSequenceFeature."))

    self._test(
        {
            "example_names": example_names,
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
                cname:
                    parsing_ops.FixedLenFeature(
                        (1, None), dtype=dtypes.int64, default_value=[[1]]),
            }
        },
        expected_err=(ValueError,
                      "All dimensions of shape for feature c need to be known "
                      r"but received \(1, None\)."))

    self._test(
        {
            "example_names": example_names,
            "serialized": ops.convert_to_tensor(serialized),
            "features": {
                aname:
                    parsing_ops.FixedLenSequenceFeature(
                        (2, 1), dtype=dtypes.float32, allow_missing=True),
                bname:
                    parsing_ops.FixedLenSequenceFeature(
                        (1, 1, 1), dtype=dtypes.string, allow_missing=True),
                cname:
                    parsing_ops.FixedLenSequenceFeature(
                        shape=[], dtype=dtypes.int64, allow_missing=False),
                dname:
                    parsing_ops.FixedLenSequenceFeature(
                        shape=[], dtype=dtypes.string, allow_missing=True),
            }
        },
        expected_err=(ValueError,
                      "Unsupported: FixedLenSequenceFeature requires "
                      "allow_missing to be True."))

  def testSerializedContainingRaggedFeatureWithNoPartitions(self):
    # [FUNCTION DOC]
    # Scenario: ragged features with no partitions (one float, one string with explicit row split
    # type) over four examples including empty and absent ones. Expects ragged results with one ragged
    # dimension and the requested row splits dtype. Then repeats with the batch repeated sixty-four
    # times (so a parse minibatch holds several examples) and expects the concatenated expected value.
    original = [
        example(features=features({"rt_c": float_feature([3, 4])})),
        example(
            features=features({
                "rt_c": float_feature([]),  # empty float list
            })),
        example(
            features=features({
                "rt_d": feature(),  # feature with nothing in it
            })),
        example(
            features=features({
                "rt_c": float_feature([1, 2, -1]),
                "rt_d": bytes_feature([b"hi"])
            }))
    ]
    serialized = [m.SerializeToString() for m in original]

    test_features = {
        "rt_c":
            parsing_ops.RaggedFeature(dtype=dtypes.float32),
        "rt_d":
            parsing_ops.RaggedFeature(
                dtype=dtypes.string, row_splits_dtype=dtypes.int64)
    }

    expected_rt_c = ragged_factory_ops.constant(
        [[3.0, 4.0], [], [], [1.0, 2.0, -1.0]],
        dtype=dtypes.float32,
        row_splits_dtype=dtypes.int32)
    expected_rt_d = ragged_factory_ops.constant([[], [], [], [b"hi"]])

    expected_output = {
        "rt_c": expected_rt_c,
        "rt_d": expected_rt_d,
    }

    self._test(
        {
            "serialized": ops.convert_to_tensor(serialized),
            "features": test_features
        }, expected_output)

    # Test with a large enough batch to ensure that the minibatch size is >1.
    batch_serialized = serialized * 64
    self.assertEqual(expected_rt_c.row_splits.dtype, np.int32)
    batch_expected_out = {
        "rt_c": ragged_concat_ops.concat([expected_rt_c] * 64, axis=0),
        "rt_d": ragged_concat_ops.concat([expected_rt_d] * 64, axis=0)
    }
    self.assertEqual(batch_expected_out["rt_c"].row_splits.dtype, dtypes.int32)
    self._test(
        {
            "serialized": ops.convert_to_tensor(batch_serialized),
            "features": test_features
        }, batch_expected_out)

  def testSerializedContainingRaggedFeature(self):
    # [FUNCTION DOC]
    # Scenario: ragged features defined by the same value key with each partition kind (row splits,
    # row lengths, row starts, row limits, value row ids), plus two with uniform row length (one
    # followed by row splits). Four examples, some empty or with empty features. Expects every
    # non-uniform kind to produce the same ragged tensor, and the uniform ones to give their own ragged
    # tensors with matching row splits dtype.
    original = [
        example(
            features=features({
                # rt = [[3], [4, 5, 6]]
                "rt_values": float_feature([3, 4, 5, 6]),
                "rt_splits": int64_feature([0, 1, 4]),
                "rt_lengths": int64_feature([1, 3]),
                "rt_starts": int64_feature([0, 1]),
                "rt_limits": int64_feature([1, 4]),
                "rt_rowids": int64_feature([0, 1, 1, 1]),
            })),
        example(
            features=features({
                # rt = []
                "rt_values": float_feature([]),
                "rt_splits": int64_feature([0]),
                "rt_lengths": int64_feature([]),
                "rt_starts": int64_feature([]),
                "rt_limits": int64_feature([]),
                "rt_rowids": int64_feature([]),
            })),
        example(
            features=features({
                # rt = []
                "rt_values": feature(),  # feature with nothing in it
                "rt_splits": int64_feature([0]),
                "rt_lengths": feature(),
                "rt_starts": feature(),
                "rt_limits": feature(),
                "rt_rowids": feature(),
            })),
        example(
            features=features({
                # rt = [[1.0, 2.0, -1.0], [], [8.0, 9.0], [5.0]]
                "rt_values": float_feature([1, 2, -1, 8, 9, 5]),
                "rt_splits": int64_feature([0, 3, 3, 5, 6]),
                "rt_lengths": int64_feature([3, 0, 2, 1]),
                "rt_starts": int64_feature([0, 3, 3, 5]),
                "rt_limits": int64_feature([3, 3, 5, 6]),
                "rt_rowids": int64_feature([0, 0, 0, 2, 2, 3]),
            }))
    ]
    serialized = ops.convert_to_tensor(
        [m.SerializeToString() for m in original])

    test_features = {
        "rt1":
            parsing_ops.RaggedFeature(
                value_key="rt_values",
                partitions=[parsing_ops.RaggedFeature.RowSplits("rt_splits")],
                dtype=dtypes.float32),
        "rt2":
            parsing_ops.RaggedFeature(
                value_key="rt_values",
                partitions=[parsing_ops.RaggedFeature.RowLengths("rt_lengths")],
                dtype=dtypes.float32),
        "rt3":
            parsing_ops.RaggedFeature(
                value_key="rt_values",
                partitions=[parsing_ops.RaggedFeature.RowStarts("rt_starts")],
                dtype=dtypes.float32),
        "rt4":
            parsing_ops.RaggedFeature(
                value_key="rt_values",
                partitions=[parsing_ops.RaggedFeature.RowLimits("rt_limits")],
                dtype=dtypes.float32),
        "rt5":
            parsing_ops.RaggedFeature(
                value_key="rt_values",
                partitions=[parsing_ops.RaggedFeature.ValueRowIds("rt_rowids")],
                dtype=dtypes.float32),
        "uniform1":
            parsing_ops.RaggedFeature(
                value_key="rt_values",
                partitions=[parsing_ops.RaggedFeature.UniformRowLength(2)],
                dtype=dtypes.float32),
        "uniform2":
            parsing_ops.RaggedFeature(
                value_key="rt_values",
                partitions=[
                    parsing_ops.RaggedFeature.UniformRowLength(2),
                    parsing_ops.RaggedFeature.RowSplits("rt_splits")
                ],
                dtype=dtypes.float32),
    }

    expected_rt = ragged_factory_ops.constant(
        [[[3], [4, 5, 6]], [], [], [[1, 2, -1], [], [8, 9], [5]]],
        dtype=dtypes.float32,
        row_splits_dtype=dtypes.int32)

    expected_uniform1 = ragged_factory_ops.constant(
        [[[3, 4], [5, 6]], [], [], [[1, 2], [-1, 8], [9, 5]]],
        ragged_rank=1,
        dtype=dtypes.float32,
        row_splits_dtype=dtypes.int32)

    expected_uniform2 = ragged_factory_ops.constant(
        [[[[3], [4, 5, 6]]], [], [], [[[1, 2, -1], []], [[8, 9], [5]]]],
        dtype=dtypes.float32,
        row_splits_dtype=dtypes.int32)

    expected_output = {
        "rt1": expected_rt,
        "rt2": expected_rt,
        "rt3": expected_rt,
        "rt4": expected_rt,
        "rt5": expected_rt,
        "uniform1": expected_uniform1,
        "uniform2": expected_uniform2,
    }

    self._test({
        "serialized": serialized,
        "features": test_features
    }, expected_output)

  def testSerializedContainingNestedRaggedFeature(self):
    # [FUNCTION DOC]
    # Scenario: one ragged feature with three partitions (uniform row length, row lengths and row splits,
    # with sixty-four-bit row splits) over two examples. Expects a ragged tensor with the nested structure
    # described in the comments in the test.
    """Test RaggedFeature with 3 partitions."""
    original = [
        # rt shape: [(batch), 2, None, None]
        example(
            features=features({
                # rt = [[[[1]], [[2, 3], [4]]], [[], [[5, 6, 7]]]]
                "rt_values": float_feature([1, 2, 3, 4, 5, 6, 7]),
                "lengths_axis2": int64_feature([1, 2, 0, 1]),
                "lengths_axis3": int64_feature([1, 2, 1, 3]),
                "splits_axis3": int64_feature([0, 1, 3, 4, 7]),
            })),
        example(
            features=features({
                # rt = [[[[1, 2, 3], [4]], [[5], [6], [7, 8]]]]
                "rt_values": float_feature([1, 2, 3, 4, 5, 6, 7, 8]),
                "lengths_axis2": int64_feature([2, 3]),
                "lengths_axis3": int64_feature([3, 1, 1, 1, 2]),
                "splits_axis3": int64_feature([0, 3, 4, 5, 6, 8]),
            }))
    ]
    serialized = ops.convert_to_tensor(
        [m.SerializeToString() for m in original])

    test_features = {
        "rt1":
            parsing_ops.RaggedFeature(
                value_key="rt_values",
                partitions=[
                    parsing_ops.RaggedFeature.UniformRowLength(2),
                    parsing_ops.RaggedFeature.RowLengths("lengths_axis2"),
                    parsing_ops.RaggedFeature.RowSplits("splits_axis3"),
                ],
                dtype=dtypes.float32,
                row_splits_dtype=dtypes.int64,
            ),
    }

    expected_rt = ragged_factory_ops.constant(
        [[[[[1]], [[2, 3], [4]]], [[], [[5, 6, 7]]]],
         [[[[1, 2, 3], [4]], [[5], [6], [7, 8]]]]],
        dtype=dtypes.float32,
        row_splits_dtype=dtypes.int64)

    expected_output = {
        "rt1": expected_rt,
    }

    self._test({
        "serialized": serialized,
        "features": test_features
    }, expected_output)


@test_util.run_all_in_graph_and_eager_modes
class ParseSingleExampleTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the single-example parser. Covers a combination of sparse, sparse-from-keys, dense,
  # sequence-style and ragged features in a single serialized example. Decorated to run in both graph
  # and eager mode. Shared fixture is the internal comparison method.

  def _test(self, kwargs, expected_values=None, expected_err=None):
    # [FUNCTION DOC]
    # HELPER: Shared driver used by every test in this class.
    # KNOWN ISSUE: There is no early return after the error branch, so with expected_err given the shape
    # KNOWN ISSUE: checks below would still run and use a result variable that was never assigned. No
    # KNOWN ISSUE: current test passes expected_err, so this path is dead.
    # Runs the single example parser with kwargs (mapping holding the serialized example, feature
    # specification and optionally example name). If expected_err (pair of error type and message pattern)
    # is given, the parse is evaluated and must raise that error; otherwise the output is compared with
    # expected_values (mapping from feature name to value or sparse tuple).
    # Afterwards checks shapes: fixed-length features with a known shape must have exactly that shape;
    # variable-length features must have rank-one indices (two by one in eager mode, unknown rows in
    # graph mode), unknown-length values (two in eager mode) and a dense shape of length one.
    if expected_err:
      with self.assertRaisesWithPredicateMatch(expected_err[0],
                                               expected_err[1]):
        self.evaluate(parsing_ops.parse_single_example(**kwargs))
    else:
      out = parsing_ops.parse_single_example(**kwargs)
      _compare_output_to_expected(self, out, expected_values)

    # Check shapes.
    for k, f in kwargs["features"].items():
      if isinstance(f, parsing_ops.FixedLenFeature) and f.shape is not None:
        self.assertEqual(
            tuple(out[k].get_shape()), tensor_shape.as_shape(f.shape))
      elif isinstance(f, parsing_ops.VarLenFeature):
        if context.executing_eagerly():
          self.assertEqual(tuple(out[k].indices.shape.as_list()), (2, 1))
          self.assertEqual(tuple(out[k].values.shape.as_list()), (2,))
          self.assertEqual(tuple(out[k].dense_shape.shape.as_list()), (1,))
        else:
          self.assertEqual(tuple(out[k].indices.shape.as_list()), (None, 1))
          self.assertEqual(tuple(out[k].values.shape.as_list()), (None,))
          self.assertEqual(tuple(out[k].dense_shape.shape.as_list()), (1,))

  def testSingleExampleWithSparseAndSparseFeatureAndDense(self):
    # [FUNCTION DOC]
    # Scenario: one example with dense, sparse and sparse-from-keys inputs. Parses with default values
    # for two dense features, a required dense feature and a sequence-style feature allowing missing
    # values. Expects the sparse tuples, defaults for missing features and provided values otherwise. It
    # is run twice, with and without an example name, because without a name a different code path is
    # used.
    original = example(
        features=features({
            "c": float_feature([3, 4]),
            "d": float_feature([0.0, 1.0]),
            "val": bytes_feature([b"a", b"b"]),
            "idx": int64_feature([0, 3]),
            "st_a": float_feature([3.0, 4.0])
        }))

    serialized = original.SerializeToString()

    a_default = [1, 2, 3]
    b_default = np.random.rand(3, 3).astype(bytes)
    test_features = {
        "st_a":
            parsing_ops.VarLenFeature(dtypes.float32),
        "sp":
            parsing_ops.SparseFeature(["idx"], "val", dtypes.string, [13]),
        "a":
            parsing_ops.FixedLenFeature((1, 3),
                                        dtypes.int64,
                                        default_value=a_default),
        "b":
            parsing_ops.FixedLenFeature((3, 3),
                                        dtypes.string,
                                        default_value=b_default),
        # Feature "c" must be provided, since it has no default_value.
        "c":
            parsing_ops.FixedLenFeature(2, dtypes.float32),
        "d":
            parsing_ops.FixedLenSequenceFeature([],
                                                dtypes.float32,
                                                allow_missing=True)
    }

    expected_st_a = (
        np.array([[0], [1]], dtype=np.int64),  # indices
        np.array([3.0, 4.0], dtype=np.float32),  # values
        np.array([2], dtype=np.int64))  # shape: max_values = 2

    expected_sp = (  # indices, values, shape
        np.array([[0], [3]], dtype=np.int64), np.array(["a", "b"], dtype="|S"),
        np.array([13], dtype=np.int64))  # max_values = 13

    expected_output = {
        "st_a": expected_st_a,
        "sp": expected_sp,
        "a": [a_default],
        "b": b_default,
        "c": np.array([3, 4], dtype=np.float32),
        "d": np.array([0.0, 1.0], dtype=np.float32),
    }

    self._test(
        {
            "example_names": ops.convert_to_tensor("in1"),
            "serialized": ops.convert_to_tensor(serialized),
            "features": test_features,
        }, expected_output)

    # Note: if example_names is None, then a different code-path gets used.
    self._test(
        {
            "serialized": ops.convert_to_tensor(serialized),
            "features": test_features,
        }, expected_output)

  def testSingleExampleWithAllFeatureTypes(self):
    # [FUNCTION DOC]
    # Scenario: one example containing fixed-length, sparse and ragged inputs. Ragged features cover a
    # one-dimensional feature, two-dimensional features for each partition kind, uniform row length,
    # and three-dimensional ones (with two row split partitions or a uniform row length first). Expects
    # sparse tuples, defaults, dense values and the matching ragged or dense tensors (the uniform
    # two-dimensional result is expected as a dense tensor).
    original = example(
        features=features({
            # FixLen features
            "c": float_feature([3, 4]),
            "d": float_feature([0.0, 1.0]),
            # Sparse features
            "val": bytes_feature([b"a", b"b"]),  # for sp
            "idx": int64_feature([0, 3]),  # for sp
            "st_a": float_feature([3.0, 4.0]),
            # Ragged features
            "rt_1d": float_feature([3.0, 4.0]),
            "rt_values": float_feature([5, 6, 7]),  # for rt_2d
            "rt_splits": int64_feature([0, 1, 1, 3]),  # for rt_2d
            "rt_lengths": int64_feature([1, 0, 2]),  # for rt_2d
            "rt_starts": int64_feature([0, 1, 1]),  # for rt_2d
            "rt_limits": int64_feature([1, 1, 3]),  # for rt_2d
            "rt_rowids": int64_feature([0, 2, 2]),  # for rt_2d
            "rt_splits2": int64_feature([0, 2, 3]),  # for rt_3d
        }))
    serialized = original.SerializeToString()

    a_default = [1, 2, 3]
    b_default = np.random.rand(3, 3).astype(bytes)
    test_features = {
        "st_a":
            parsing_ops.VarLenFeature(dtypes.float32),
        "sp":
            parsing_ops.SparseFeature(["idx"], "val", dtypes.string, [13]),
        "a":
            parsing_ops.FixedLenFeature((1, 3),
                                        dtypes.int64,
                                        default_value=a_default),
        "b":
            parsing_ops.FixedLenFeature((3, 3),
                                        dtypes.string,
                                        default_value=b_default),
        # Feature "c" must be provided, since it has no default_value.
        "c":
            parsing_ops.FixedLenFeature(2, dtypes.float32),
        "d":
            parsing_ops.FixedLenSequenceFeature([],
                                                dtypes.float32,
                                                allow_missing=True),
        "rt_1d":
            parsing_ops.RaggedFeature(dtypes.float32),
        "rt_2d_with_splits":
            parsing_ops.RaggedFeature(
                value_key="rt_values",
                partitions=[parsing_ops.RaggedFeature.RowSplits("rt_splits")],
                dtype=dtypes.float32),
        "rt_2d_with_lengths":
            parsing_ops.RaggedFeature(
                value_key="rt_values",
                partitions=[parsing_ops.RaggedFeature.RowLengths("rt_lengths")],
                dtype=dtypes.float32),
        "rt_2d_with_starts":
            parsing_ops.RaggedFeature(
                value_key="rt_values",
                partitions=[parsing_ops.RaggedFeature.RowStarts("rt_starts")],
                dtype=dtypes.float32),
        "rt_2d_with_limits":
            parsing_ops.RaggedFeature(
                value_key="rt_values",
                partitions=[parsing_ops.RaggedFeature.RowLimits("rt_limits")],
                dtype=dtypes.float32),
        "rt_2d_with_rowids":
            parsing_ops.RaggedFeature(
                value_key="rt_values",
                partitions=[parsing_ops.RaggedFeature.ValueRowIds("rt_rowids")],
                dtype=dtypes.float32),
        "rt_2d_with_uniform_row_length":
            parsing_ops.RaggedFeature(
                value_key="rt_values",
                partitions=[parsing_ops.RaggedFeature.UniformRowLength(1)],
                dtype=dtypes.float32),
        "rt_3d":
            parsing_ops.RaggedFeature(
                value_key="rt_values",
                partitions=[
                    parsing_ops.RaggedFeature.RowSplits("rt_splits2"),
                    parsing_ops.RaggedFeature.RowSplits("rt_splits")
                ],
                dtype=dtypes.float32),
        "rt_3d_with_uniform_row_length":
            parsing_ops.RaggedFeature(
                value_key="rt_values",
                partitions=[
                    parsing_ops.RaggedFeature.UniformRowLength(1),
                    parsing_ops.RaggedFeature.RowSplits("rt_splits")
                ],
                dtype=dtypes.float32),
    }

    expected_st_a = (
        np.array([[0], [1]], dtype=np.int64),  # indices
        np.array([3.0, 4.0], dtype=np.float32),  # values
        np.array([2], dtype=np.int64))  # shape: max_values = 2

    expected_sp = (  # indices, values, shape
        np.array([[0], [3]], dtype=np.int64), np.array(["a", "b"], dtype="|S"),
        np.array([13], dtype=np.int64))  # max_values = 13

    expected_rt_1d = constant_op.constant([3, 4], dtypes.float32)

    expected_rt_2d = ragged_factory_ops.constant([[5], [], [6, 7]],
                                                 dtype=dtypes.float32)

    expected_rt_2d_uniform = constant_op.constant([[5], [6], [7]],
                                                  dtype=dtypes.float32)

    expected_rt_3d = ragged_factory_ops.constant([[[5], []], [[6, 7]]],
                                                 dtype=dtypes.float32)

    expected_rt_3d_with_uniform = (
        ragged_tensor.RaggedTensor.from_uniform_row_length(
            expected_rt_2d, uniform_row_length=1))

    expected_output = {
        "st_a": expected_st_a,
        "sp": expected_sp,
        "a": [a_default],
        "b": b_default,
        "c": np.array([3, 4], dtype=np.float32),
        "d": np.array([0.0, 1.0], dtype=np.float32),
        "rt_1d": expected_rt_1d,
        "rt_2d_with_splits": expected_rt_2d,
        "rt_2d_with_lengths": expected_rt_2d,
        "rt_2d_with_starts": expected_rt_2d,
        "rt_2d_with_limits": expected_rt_2d,
        "rt_2d_with_rowids": expected_rt_2d,
        "rt_2d_with_uniform_row_length": expected_rt_2d_uniform,
        "rt_3d": expected_rt_3d,
        "rt_3d_with_uniform_row_length": expected_rt_3d_with_uniform,
    }

    self._test(
        {
            "example_names": ops.convert_to_tensor("in1"),
            "serialized": ops.convert_to_tensor(serialized),
            "features": test_features,
        }, expected_output)


@test_util.run_all_in_graph_and_eager_modes
class ParseSequenceExampleTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the sequence example parsers, both the batch version and the single-example version.
  # Most tests go through a helper that runs both. Covers context features, feature lists (dense with
  # padding, sparse and ragged), error messages for inconsistent types and shapes, batches with
  # padding and returned lengths, and nested ragged partitions. Decorated to run in graph and eager
  # mode.

  def testCreateSequenceExample(self):
    # [FUNCTION DOC]
    # Smoke test: builds a sequence example holding one context feature and two feature lists and
    # serializes it. Asserts nothing besides that serialization does not raise an error.
    value = sequence_example(
        context=features({
            "global_feature": float_feature([1, 2, 3]),
        }),
        feature_lists=feature_lists({
            "repeated_feature_2_frames":
                feature_list([
                    bytes_feature([b"a", b"b", b"c"]),
                    bytes_feature([b"a", b"d", b"e"])
                ]),
            "repeated_feature_3_frames":
                feature_list([
                    int64_feature([3, 4, 5, 6, 7]),
                    int64_feature([-1, 0, 0, 0, 0]),
                    int64_feature([1, 2, 3, 4, 5])
                ])
        }))
    value.SerializeToString()  # Smoke test

  def _test(self,
            kwargs,
            expected_context_values=None,
            expected_feat_list_values=None,
            expected_length_values=None,
            expected_err=None,
            batch=False):
    # [FUNCTION DOC]
    # HELPER: Shared driver used by the tests in this class (single or batch parsing).
    # KNOWN ISSUE: After an expected error the shape checks still run when context features are in
    # KNOWN ISSUE: kwargs, and they would use a context output variable that was never assigned; no test
    # KNOWN ISSUE: combines an error with context features, so this is latent.
    # Takes kwargs (mapping of arguments to the parser), expected context values, expected feature list
    # values and expected lengths (mappings from feature name to values; each defaults to empty),
    # expected_err (pair of error type and message pattern) and batch (true uses the batch parser,
    # false the single parser, in which case lengths are empty).
    # With expected_err given, the parse must raise it (evaluated in graph mode). Otherwise the three
    # output groups are compared against the expectations.
    # When context features are given, checks shapes: fixed-length features must match their declared
    # shape (dropping the leading batch dimension in batch mode); variable-length features must be rank
    # two in batch mode and rank one otherwise, with unknown sizes.
    expected_context_values = expected_context_values or {}
    expected_feat_list_values = expected_feat_list_values or {}
    expected_length_values = expected_length_values or {}

    if expected_err:
      with self.assertRaisesWithPredicateMatch(expected_err[0],
                                               expected_err[1]):
        if batch:
          self.evaluate(parsing_ops.parse_sequence_example(**kwargs))
        else:
          self.evaluate(parsing_ops.parse_single_sequence_example(**kwargs))
    else:
      if batch:
        (context_out, feat_list_out,
         lengths_out) = parsing_ops.parse_sequence_example(**kwargs)
      else:
        (context_out,
         feat_list_out) = parsing_ops.parse_single_sequence_example(**kwargs)
        lengths_out = {}

      # Check values.
      _compare_output_to_expected(self, context_out, expected_context_values)
      _compare_output_to_expected(self, feat_list_out,
                                  expected_feat_list_values)
      _compare_output_to_expected(self, lengths_out, expected_length_values)

    # Check shapes; if serialized is a Tensor we need its size to
    # properly check.
    if "context_features" in kwargs:
      for k, f in kwargs["context_features"].items():
        if isinstance(f, parsing_ops.FixedLenFeature) and f.shape is not None:
          if batch:
            self.assertEqual(tuple(context_out[k].shape.as_list()[1:]), f.shape)
          else:
            self.assertEqual(tuple(context_out[k].shape.as_list()), f.shape)
        elif isinstance(f, parsing_ops.VarLenFeature) and batch:
          if context.executing_eagerly():
            context_out[k].indices.shape.assert_is_compatible_with([None, 2])
            context_out[k].values.shape.assert_is_compatible_with([None])
            context_out[k].dense_shape.shape.assert_is_compatible_with([2])
          else:
            self.assertEqual(context_out[k].indices.shape.as_list(), [None, 2])
            self.assertEqual(context_out[k].values.shape.as_list(), [None])
            self.assertEqual(context_out[k].dense_shape.shape.as_list(), [2])
        elif isinstance(f, parsing_ops.VarLenFeature) and not batch:
          if context.executing_eagerly():
            context_out[k].indices.shape.assert_is_compatible_with([None, 1])
            context_out[k].values.shape.assert_is_compatible_with([None])
            context_out[k].dense_shape.shape.assert_is_compatible_with([1])
          else:
            self.assertEqual(context_out[k].indices.shape.as_list(), [None, 1])
            self.assertEqual(context_out[k].values.shape.as_list(), [None])
            self.assertEqual(context_out[k].dense_shape.shape.as_list(), [1])

  def _testBoth(self,
                kwargs,
                expected_context_values=None,
                expected_feat_list_values=None,
                expected_err=None):
    # [FUNCTION DOC]
    # HELPER: Shared driver that runs one scenario against both the single and batch parsers.
    # KNOWN ISSUE: It modifies the caller-supplied kwargs mapping in place (removes the serialized and
    # KNOWN ISSUE: example name entries and replaces them), so kwargs must not be reused afterwards.
    # First runs the single-example parser with the given kwargs, expected context values, expected
    # feature list values and optional expected error. Then converts the input into a batch of one: wraps
    # the serialized value in a list, converts the optional example name to a list of names (or none), and
    # adds a batch dimension to every expected value (dense and ragged values get a new leading axis,
    # sparse tuples get a zero index column and a batch size of one in the shape). For fixed-length
    # sequence features it also records expected lengths from the first dimension. Finally runs the batch parser.
    # Test using tf.io.parse_single_sequence_example
    self._test(
        kwargs,
        expected_context_values=expected_context_values,
        expected_feat_list_values=expected_feat_list_values,
        expected_err=expected_err,
        batch=False)

    # Convert the input to a batch of size 1, and test using
    # tf.parse_sequence_example.

    # Some replacements are needed for the batch version.
    kwargs["serialized"] = [kwargs.pop("serialized")]
    kwargs["example_names"] = [kwargs.pop("example_name")
                              ] if "example_name" in kwargs else None

    # Add a batch dimension to expected output
    if expected_context_values:
      new_values = {}
      for k in expected_context_values:
        v = expected_context_values[k]
        if isinstance(kwargs["context_features"][k],
                      (parsing_ops.FixedLenFeature, parsing_ops.RaggedFeature)):
          new_values[k] = np.expand_dims(v, axis=0)
        else:
          # Sparse tensor.
          new_values[k] = (np.insert(v[0], 0, 0,
                                     axis=1), v[1], np.insert(v[2], 0, 1))
      expected_context_values = new_values

    expected_length_values = {}
    if expected_feat_list_values:
      new_values = {}
      for k in expected_feat_list_values:
        v = expected_feat_list_values[k]
        if isinstance(kwargs["sequence_features"][k],
                      parsing_ops.FixedLenSequenceFeature):
          expected_length_values[k] = [np.shape(v)[0]]
          new_values[k] = np.expand_dims(v, axis=0)
        elif isinstance(kwargs["sequence_features"][k],
                        parsing_ops.RaggedFeature):
          new_values[k] = np.expand_dims(v, axis=0)
        else:
          # Sparse tensor.
          new_values[k] = (np.insert(v[0], 0, 0,
                                     axis=1), v[1], np.insert(v[2], 0, 1))
      expected_feat_list_values = new_values

    self._test(
        kwargs,
        expected_context_values=expected_context_values,
        expected_feat_list_values=expected_feat_list_values,
        expected_length_values=expected_length_values,
        expected_err=expected_err,
        batch=True)

  def testSequenceExampleWithSparseAndDenseContext(self):
    # [FUNCTION DOC]
    # Scenario: a sequence example with only context features; variable-length context feature, two
    # dense context features with defaults and one required dense context feature. Expects sparse tuple,
    # defaults and provided value, with and without batching.
    original = sequence_example(
        context=features({
            "c": float_feature([3, 4]),
            "st_a": float_feature([3.0, 4.0])
        }))

    serialized = original.SerializeToString()

    expected_st_a = (
        np.array([[0], [1]], dtype=np.int64),  # indices
        np.array([3.0, 4.0], dtype=np.float32),  # values
        np.array([2], dtype=np.int64))  # shape: num_features = 2

    a_default = [[1, 2, 3]]
    b_default = np.random.rand(3, 3).astype(bytes)
    expected_context_output = {
        "st_a": expected_st_a,
        "a": a_default,
        "b": b_default,
        "c": np.array([3, 4], dtype=np.float32),
    }

    self._testBoth(
        {
            "example_name": "in1",
            "serialized": ops.convert_to_tensor(serialized),
            "context_features": {
                "st_a":
                    parsing_ops.VarLenFeature(dtypes.float32),
                "a":
                    parsing_ops.FixedLenFeature(
                        (1, 3), dtypes.int64, default_value=a_default),
                "b":
                    parsing_ops.FixedLenFeature(
                        (3, 3), dtypes.string, default_value=b_default),
                # Feature "c" must be provided, since it has no default_value.
                "c":
                    parsing_ops.FixedLenFeature((2,), dtypes.float32),
            }
        },
        expected_context_values=expected_context_output)

  def testSequenceExampleWithMultipleSizeFeatureLists(self):
    # [FUNCTION DOC]
    # Scenario: feature lists of different numbers of steps and shapes: integer lists of four steps
    # with one-by-three, a single step of two-by-two strings, two steps of two floats and one missing
    # list allowed to be missing. Expects dense time-major outputs of those shapes, and an empty
    # result for the missing-allowed list.
    original = sequence_example(
        feature_lists=feature_lists({
            "a":
                feature_list([
                    int64_feature([-1, 0, 1]),
                    int64_feature([2, 3, 4]),
                    int64_feature([5, 6, 7]),
                    int64_feature([8, 9, 10]),
                ]),
            "b":
                feature_list([bytes_feature([b"r00", b"r01", b"r10", b"r11"])]),
            "c":
                feature_list([float_feature([3, 4]),
                              float_feature([-1, 2])]),
        }))

    serialized = original.SerializeToString()

    expected_feature_list_output = {
        "a":
            np.array(
                [  # outer dimension is time.
                    [[-1, 0, 1]],  # inside are 1x3 matrices
                    [[2, 3, 4]],
                    [[5, 6, 7]],
                    [[8, 9, 10]]
                ],
                dtype=np.int64),
        "b":
            np.array(
                [  # outer dimension is time, inside are 2x2 matrices
                    [[b"r00", b"r01"], [b"r10", b"r11"]]
                ],
                dtype=bytes),
        "c":
            np.array(
                [  # outer dimension is time, inside are 2-vectors
                    [3, 4], [-1, 2]
                ],
                dtype=np.float32),
        "d":
            np.empty(shape=(0, 5), dtype=np.float32),  # empty_allowed_missing
    }

    self._testBoth(
        {
            "example_name": "in1",
            "serialized": ops.convert_to_tensor(serialized),
            "sequence_features": {
                "a":
                    parsing_ops.FixedLenSequenceFeature((1, 3), dtypes.int64),
                "b":
                    parsing_ops.FixedLenSequenceFeature((2, 2), dtypes.string),
                "c":
                    parsing_ops.FixedLenSequenceFeature(2, dtypes.float32),
                "d":
                    parsing_ops.FixedLenSequenceFeature(
                        (5,), dtypes.float32, allow_missing=True),
            }
        },
        expected_feat_list_values=expected_feature_list_output)

  def testSequenceExampleWithoutDebugName(self):
    # [FUNCTION DOC]
    # Scenario: feature lists parsed as sparse and dense features when no example name is provided.
    # Expects sparse tuples for sparse lists (empty entries produce no indices, absent list gives an empty
    # result with zero dense shape) and a dense integer matrix for the dense list.
    original = sequence_example(
        feature_lists=feature_lists({
            "a":
                feature_list([int64_feature([3, 4]),
                              int64_feature([1, 0])]),
            "st_a":
                feature_list([
                    float_feature([3.0, 4.0]),
                    float_feature([5.0]),
                    float_feature([])
                ]),
            "st_b":
                feature_list([
                    bytes_feature([b"a"]),
                    bytes_feature([]),
                    bytes_feature([]),
                    bytes_feature([b"b", b"c"])
                ])
        }))

    serialized = original.SerializeToString()

    expected_st_a = (
        np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64),  # indices
        np.array([3.0, 4.0, 5.0], dtype=np.float32),  # values
        np.array([3, 2], dtype=np.int64))  # shape: num_time = 3, max_feat = 2

    expected_st_b = (
        np.array([[0, 0], [3, 0], [3, 1]], dtype=np.int64),  # indices
        np.array(["a", "b", "c"], dtype="|S"),  # values
        np.array([4, 2], dtype=np.int64))  # shape: num_time = 4, max_feat = 2

    expected_st_c = (
        np.empty((0, 2), dtype=np.int64),  # indices
        np.empty((0,), dtype=np.int64),  # values
        np.array([0, 0], dtype=np.int64))  # shape: num_time = 0, max_feat = 0

    expected_feature_list_output = {
        "a": np.array([[3, 4], [1, 0]], dtype=np.int64),
        "st_a": expected_st_a,
        "st_b": expected_st_b,
        "st_c": expected_st_c,
    }

    self._testBoth(
        {
            "serialized": ops.convert_to_tensor(serialized),
            "sequence_features": {
                "st_a": parsing_ops.VarLenFeature(dtypes.float32),
                "st_b": parsing_ops.VarLenFeature(dtypes.string),
                "st_c": parsing_ops.VarLenFeature(dtypes.int64),
                "a": parsing_ops.FixedLenSequenceFeature((2,), dtypes.int64),
            }
        },
        expected_feat_list_values=expected_feature_list_output)

  def testSequenceExampleWithSparseAndDenseFeatureLists(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: Nearly identical to the previous test; the only difference is that an example name
    # KNOWN ISSUE: is provided, so it mostly duplicates its coverage.
    # Scenario: same feature lists as the test without a debug name, but with an example name. Expects
    # the same sparse and dense results.
    original = sequence_example(
        feature_lists=feature_lists({
            "a":
                feature_list([int64_feature([3, 4]),
                              int64_feature([1, 0])]),
            "st_a":
                feature_list([
                    float_feature([3.0, 4.0]),
                    float_feature([5.0]),
                    float_feature([])
                ]),
            "st_b":
                feature_list([
                    bytes_feature([b"a"]),
                    bytes_feature([]),
                    bytes_feature([]),
                    bytes_feature([b"b", b"c"])
                ])
        }))

    serialized = original.SerializeToString()

    expected_st_a = (
        np.array([[0, 0], [0, 1], [1, 0]], dtype=np.int64),  # indices
        np.array([3.0, 4.0, 5.0], dtype=np.float32),  # values
        np.array([3, 2], dtype=np.int64))  # shape: num_time = 3, max_feat = 2

    expected_st_b = (
        np.array([[0, 0], [3, 0], [3, 1]], dtype=np.int64),  # indices
        np.array(["a", "b", "c"], dtype="|S"),  # values
        np.array([4, 2], dtype=np.int64))  # shape: num_time = 4, max_feat = 2

    expected_st_c = (
        np.empty((0, 2), dtype=np.int64),  # indices
        np.empty((0,), dtype=np.int64),  # values
        np.array([0, 0], dtype=np.int64))  # shape: num_time = 0, max_feat = 0

    expected_feature_list_output = {
        "a": np.array([[3, 4], [1, 0]], dtype=np.int64),
        "st_a": expected_st_a,
        "st_b": expected_st_b,
        "st_c": expected_st_c,
    }

    self._testBoth(
        {
            "example_name": "in1",
            "serialized": ops.convert_to_tensor(serialized),
            "sequence_features": {
                "st_a": parsing_ops.VarLenFeature(dtypes.float32),
                "st_b": parsing_ops.VarLenFeature(dtypes.string),
                "st_c": parsing_ops.VarLenFeature(dtypes.int64),
                "a": parsing_ops.FixedLenSequenceFeature((2,), dtypes.int64),
            }
        },
        expected_feat_list_values=expected_feature_list_output)

  def testSequenceExampleWithEmptyFeatureInFeatureLists(self):
    # [FUNCTION DOC]
    # Scenario: a sparse feature list in which the middle step is an empty feature. Expects that step to
    # produce no sparse entries but still count as a time step (three rows, widest step two).
    original = sequence_example(
        feature_lists=feature_lists({
            "st_a":
                feature_list([
                    float_feature([3.0, 4.0]),
                    feature(),
                    float_feature([5.0]),
                ]),
        }))

    serialized = original.SerializeToString()

    expected_st_a = (
        np.array([[0, 0], [0, 1], [2, 0]], dtype=np.int64),  # indices
        np.array([3.0, 4.0, 5.0], dtype=np.float32),  # values
        np.array([3, 2], dtype=np.int64))  # shape: num_time = 3, max_feat = 2

    expected_feature_list_output = {
        "st_a": expected_st_a,
    }

    self._testBoth(
        {
            "example_name": "in1",
            "serialized": ops.convert_to_tensor(serialized),
            "sequence_features": {
                "st_a": parsing_ops.VarLenFeature(dtypes.float32),
            }
        },
        expected_feat_list_values=expected_feature_list_output)

  def testSequenceExampleListWithInconsistentDataFails(self):
    # [FUNCTION DOC]
    # Scenario: a feature list mixing an integer step and a float step parsed as integer dense. Expects
    # an op error that the data types do not match for the second step.
    original = sequence_example(
        feature_lists=feature_lists({
            "a": feature_list([int64_feature([-1, 0]),
                               float_feature([2, 3])])
        }))

    serialized = original.SerializeToString()

    self._testBoth(
        {
            "example_name": "in1",
            "serialized": ops.convert_to_tensor(serialized),
            "sequence_features": {
                "a": parsing_ops.FixedLenSequenceFeature((2,), dtypes.int64)
            }
        },
        expected_err=(errors_impl.OpError, "Feature list: a, Index: 1."
                      "  Data types don't match. Expected type: int64"))

  def testSequenceExampleListWithWrongDataTypeFails(self):
    # [FUNCTION DOC]
    # Scenario: a feature list of a single float step parsed as integer dense. Expects an op error
    # reporting data types do not match at index zero.
    original = sequence_example(
        feature_lists=feature_lists(
            {"a": feature_list([float_feature([2, 3])])}))

    serialized = original.SerializeToString()

    self._testBoth(
        {
            "example_name": "in1",
            "serialized": ops.convert_to_tensor(serialized),
            "sequence_features": {
                "a": parsing_ops.FixedLenSequenceFeature((2,), dtypes.int64)
            }
        },
        expected_err=(errors_impl.OpError,
                      "Feature list: a, Index: 0.  Data types don't match."
                      " Expected type: int64"))

  def testSequenceExampleListWithWrongSparseDataTypeFails(self):
    # [FUNCTION DOC]
    # Scenario: a feature list with two integer steps followed by a float step, parsed as integer dense
    # with an example name. Expects an op error naming the example and index two for mismatched types.
    original = sequence_example(
        feature_lists=feature_lists({
            "a":
                feature_list([
                    int64_feature([3, 4]),
                    int64_feature([1, 2]),
                    float_feature([2.0, 3.0])
                ])
        }))

    serialized = original.SerializeToString()

    self._testBoth(
        {
            "example_name": "in1",
            "serialized": ops.convert_to_tensor(serialized),
            "sequence_features": {
                "a": parsing_ops.FixedLenSequenceFeature((2,), dtypes.int64)
            }
        },
        expected_err=(errors_impl.OpError,
                      "Name: in1, Feature list: a, Index: 2."
                      "  Data types don't match. Expected type: int64"))

  def testSequenceExampleListWithWrongShapeFails(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The message pattern accepts two alternatives (one from the single-example kernel
    # KNOWN ISSUE: and one from the fast batch kernel), so the test does not say which applies to which mode.
    # Scenario: steps of two and three values but a fixed shape of two. Expects an op error about the
    # number of values differing from the expected shape, in either message form.
    original = sequence_example(
        feature_lists=feature_lists({
            "a":
                feature_list([int64_feature([2, 3]),
                              int64_feature([2, 3, 4])]),
        }))

    serialized = original.SerializeToString()

    self._testBoth(
        {
            "example_name": "in1",
            "serialized": ops.convert_to_tensor(serialized),
            "sequence_features": {
                "a": parsing_ops.FixedLenSequenceFeature((2,), dtypes.int64)
            }
        },
        expected_err=(
            errors_impl.OpError,
            # message from ParseSingleExample.
            r"Name: in1, Key: a, Index: 1."
            r"  Number of int64 values != expected."
            r"  values size: 3 but output shape: \[2\]"
            # or message from FastParseSequenceExample
            r"|Feature list 'a' has an unexpected number of values.  "
            r"Total values size: 5 is not consistent with output "
            r"shape: \[\?,2\]"))

  def testSequenceExampleListWithWrongShapeFails2(self):
    # [FUNCTION DOC]
    # Scenario: like the previous test but with steps of one and three values, so a bad shape can only
    # be detected by examining individual rows in the fast batch parser. Expects an op error naming the
    # first step, with the word int64 optional in the message.
    # This exercises a different code path for FastParseSequenceExample than
    # testSequenceExampleListWithWrongShapeFails (in that test, we can tell that
    # the shape is bad based on the total number of values; in this test, we
    # can't tell the shape is bad until we look at individual rows.)
    original = sequence_example(
        feature_lists=feature_lists({
            "a": feature_list([int64_feature([2]),
                               int64_feature([2, 3, 4])]),
        }))

    serialized = original.SerializeToString()

    self._testBoth(
        {
            "example_name": "in1",
            "serialized": ops.convert_to_tensor(serialized),
            "sequence_features": {
                "a": parsing_ops.FixedLenSequenceFeature((2,), dtypes.int64)
            }
        },
        expected_err=(errors_impl.OpError, r"Name: in1, Key: a, Index: 0."
                      r"  Number of (int64 )?values != expected."
                      r"  values size: 1 but output shape: \[2\]"))

  def testSequenceExampleWithMissingFeatureListFails(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: A comment refers to adding a default entry for the feature list, which is not how
    # KNOWN ISSUE: the current API works; the message instead points to the missing-assumed-empty or
    # KNOWN ISSUE: defaults arguments.
    # Scenario: a sequence example with no feature lists while a required dense feature list is
    # requested. Expects an op error saying the feature list is required but could not be found.
    original = sequence_example(feature_lists=feature_lists({}))

    # Test fails because we didn't add:
    #  feature_list_dense_defaults = {"a": None}
    self._testBoth(
        {
            "example_name": "in1",
            "serialized": ops.convert_to_tensor(original.SerializeToString()),
            "sequence_features": {
                "a": parsing_ops.FixedLenSequenceFeature((2,), dtypes.int64)
            }
        },
        expected_err=(
            errors_impl.OpError,
            "Name: in1, Feature list 'a' is required but could not be found."
            "  Did you mean to include it in"
            " feature_list_dense_missing_assumed_empty or"
            " feature_list_dense_defaults?"))

  def testSequenceExampleBatch(self):
    # [FUNCTION DOC]
    # Scenario: two sequence examples of different lengths parsed together with the batch parser,
    # using a context feature with a default and three feature lists (one required, one allowed missing,
    # one allowed missing and absent everywhere). Expects results padded with zeros over time, defaults
    # for missing context values, an empty result for the absent list and returned lengths per example.
    first = sequence_example(
        feature_lists=feature_lists({
            "a":
                feature_list([
                    int64_feature([-1, 0, 1]),
                    int64_feature([2, 3, 4]),
                    int64_feature([5, 6, 7]),
                    int64_feature([8, 9, 10]),
                ])
        }))
    second = sequence_example(
        context=features({"c": float_feature([7])}),
        feature_lists=feature_lists({
            "a": feature_list([
                int64_feature([21, 2, 11]),
            ]),
            "b": feature_list([
                int64_feature([5]),
            ]),
        }))

    serialized = [first.SerializeToString(), second.SerializeToString()]

    expected_context_output = {
        "c": np.array([-1, 7], dtype=np.float32),
    }
    expected_feature_list_output = {
        "a":
            np.array(
                [  # outermost dimension is example id
                    [  # middle dimension is time.
                        [[-1, 0, 1]],  # inside are 1x3 matrices
                        [[2, 3, 4]],
                        [[5, 6, 7]],
                        [[8, 9, 10]]
                    ],
                    [  # middle dimension is time.
                        [[21, 2, 11]],  # inside are 1x3 matrices
                        [[0, 0, 0]],  # additional entries are padded with 0
                        [[0, 0, 0]],
                        [[0, 0, 0]]
                    ]
                ],
                dtype=np.int64),
        "b":
            np.array([[0], [5]], dtype=np.int64),
        "d":
            np.empty(shape=(2, 0, 5), dtype=np.float32),  # allowed_missing
    }

    self._test(
        {
            "example_names": ops.convert_to_tensor(["in1", "in2"]),
            "serialized": ops.convert_to_tensor(serialized),
            "context_features": {
                "c":
                    parsing_ops.FixedLenFeature(
                        (), dtypes.float32, default_value=-1),
            },
            "sequence_features": {
                "a":
                    parsing_ops.FixedLenSequenceFeature((1, 3), dtypes.int64),
                "b":
                    parsing_ops.FixedLenSequenceFeature(
                        (), dtypes.int64, allow_missing=True),
                "d":
                    parsing_ops.FixedLenSequenceFeature(
                        (5,), dtypes.float32, allow_missing=True),
            }
        },
        expected_context_values=expected_context_output,
        expected_feat_list_values=expected_feature_list_output,
        expected_length_values={
            "a": [4, 1],
            "b": [0, 1],
            "d": [0, 0]
        },
        batch=True)

  def testSerializedContainingRaggedFeatureWithNoPartitions(self):
    # [FUNCTION DOC]
    # Scenario: ragged features without partitions for both context and sequence features over four
    # sequence examples (some with empty lists or empty features). Expects ragged results with the
    # requested row split types for the batch, then checks the single-example parse of the first example,
    # then repeats the batch parse with the data repeated sixty-four times.
    original = [
        sequence_example(
            context=features({"a": float_feature([3, 4])}),
            feature_lists=feature_lists({
                "b": feature_list([float_feature([5]),
                                   float_feature([3])]),
                "c": feature_list([int64_feature([6, 7, 8, 9])])
            })),
        sequence_example(
            context=features({"a": float_feature([9])}),
            feature_lists=feature_lists({
                "b": feature_list([]),
                "c": feature_list([int64_feature([]),
                                   int64_feature([1, 2, 3])])
            })),
        sequence_example(
            feature_lists=feature_lists({
                "b":
                    feature_list([
                        float_feature([1]),
                        float_feature([1, 2]),
                        float_feature([1, 2, 3])
                    ])
            })),
        sequence_example(
            context=features({"a": feature()}),
            feature_lists=feature_lists({
                "b": feature_list([feature()]),
                "c": feature_list([int64_feature([3, 3, 3])])
            }))
    ]
    serialized = [m.SerializeToString() for m in original]

    context_features = {"a": parsing_ops.RaggedFeature(dtype=dtypes.float32)}
    sequence_features = {
        "b":
            parsing_ops.RaggedFeature(dtype=dtypes.float32),
        "c":
            parsing_ops.RaggedFeature(
                dtype=dtypes.int64, row_splits_dtype=dtypes.int64)
    }

    expected_a = ragged_factory_ops.constant([[3, 4], [9], [], []],
                                             dtype=dtypes.float32,
                                             row_splits_dtype=dtypes.int32)
    expected_b = ragged_factory_ops.constant(
        [[[5], [3]], [], [[1], [1, 2], [1, 2, 3]], [[]]],
        dtype=dtypes.float32,
        row_splits_dtype=dtypes.int32)
    expected_c = ragged_factory_ops.constant(
        [[[6, 7, 8, 9]], [[], [1, 2, 3]], [], [[3, 3, 3]]],
        dtype=dtypes.int64,
        row_splits_dtype=dtypes.int64)

    expected_context_output = dict(a=expected_a)
    expected_feature_list_output = dict(b=expected_b, c=expected_c)

    self._test(
        {
            "serialized": ops.convert_to_tensor(serialized),
            "context_features": context_features,
            "sequence_features": sequence_features,
        },
        expected_context_output,
        expected_feature_list_output,
        batch=True)

    self._test(
        {
            "serialized": ops.convert_to_tensor(serialized)[0],
            "context_features": context_features,
            "sequence_features": sequence_features,
        },
        expected_context_values={"a": [3, 4]},
        expected_feat_list_values={
            "b": [[5], [3]],
            "c": [[6, 7, 8, 9]]
        },
        batch=False)

    # Test with a larger batch of examples.
    batch_serialized = serialized * 64
    batch_context_expected_out = {
        "a": ragged_concat_ops.concat([expected_a] * 64, axis=0)
    }
    batch_feature_list_expected_out = {
        "b": ragged_concat_ops.concat([expected_b] * 64, axis=0),
        "c": ragged_concat_ops.concat([expected_c] * 64, axis=0)
    }
    self._test(
        {
            "serialized": ops.convert_to_tensor(batch_serialized),
            "context_features": context_features,
            "sequence_features": sequence_features,
        },
        batch_context_expected_out,
        batch_feature_list_expected_out,
        batch=True)

  def testSerializedContainingNestedRaggedFeature(self):
    # [FUNCTION DOC]
    # Scenario: ragged features with nested partitions. A context feature uses uniform row length, row
    # lengths and row splits; sequence features use row splits or uniform row length of one with values in
    # feature lists. Three examples, the second is empty. Expects the batch parse to equal the expected
    # ragged tensors, and the single parse of the first example to equal their first rows.
    """Test RaggedFeatures with nested partitions."""
    original = [
        # rt shape: [(batch), 2, None, None]
        sequence_example(
            context=features({
                # a[0] = [[[[1]], [[2, 3], [4]]], [[], [[5, 6, 7]]]]
                "a_values": float_feature([1, 2, 3, 4, 5, 6, 7]),
                "a_lengths_axis2": int64_feature([1, 2, 0, 1]),
                "a_lengths_axis3": int64_feature([1, 2, 1, 3]),
                "a_splits_axis3": int64_feature([0, 1, 3, 4, 7])
            }),
            feature_lists=feature_lists({
                # b[0] = [[[1], [2, 3, 4]], [[2, 4], [6]]]
                "b_values":
                    feature_list(
                        [float_feature([1, 2, 3, 4]),
                         float_feature([2, 4, 6])]),
                "b_splits":
                    feature_list(
                        [int64_feature([0, 1, 4]),
                         int64_feature([0, 2, 3])]),
            })),
        sequence_example(
            # a[1] = []
            # b[1] = []
        ),
        sequence_example(
            context=features({
                # a[2] = [[[[1, 2, 3], [4]], [[5], [6], [7, 8]]]]
                "a_values": float_feature([1, 2, 3, 4, 5, 6, 7, 8]),
                "a_lengths_axis2": int64_feature([2, 3]),
                "a_lengths_axis3": int64_feature([3, 1, 1, 1, 2]),
                "a_splits_axis3": int64_feature([0, 3, 4, 5, 6, 8])
            }),
            feature_lists=feature_lists({
                # b[2] = [[[9], [8, 7, 6], [5]], [[4, 3, 2, 1]], [[0]]]
                "b_values":
                    feature_list([
                        float_feature([9, 8, 7, 6, 5]),
                        float_feature([4, 3, 2, 1]),
                        float_feature([0])
                    ]),
                "b_splits":
                    feature_list([
                        int64_feature([0, 1, 4, 5]),
                        int64_feature([0, 4]),
                        int64_feature([0, 1])
                    ])
            }))
    ]
    serialized = [m.SerializeToString() for m in original]

    context_features = {
        "a":
            parsing_ops.RaggedFeature(
                value_key="a_values",
                partitions=[
                    parsing_ops.RaggedFeature.UniformRowLength(2),
                    parsing_ops.RaggedFeature.RowLengths("a_lengths_axis2"),
                    parsing_ops.RaggedFeature.RowSplits("a_splits_axis3"),
                ],
                dtype=dtypes.float32,
                row_splits_dtype=dtypes.int64,
            )
    }
    sequence_features = {
        "b":
            parsing_ops.RaggedFeature(
                value_key="b_values",
                dtype=dtypes.float32,
                partitions=[parsing_ops.RaggedFeature.RowSplits("b_splits")]),
        "c":
            parsing_ops.RaggedFeature(
                value_key="b_values",
                dtype=dtypes.float32,
                partitions=[parsing_ops.RaggedFeature.UniformRowLength(1)]),
    }

    expected_context = {
        "a":
            ragged_factory_ops.constant(
                [[[[[1]], [[2, 3], [4]]], [[], [[5, 6, 7]]]], [],
                 [[[[1, 2, 3], [4]], [[5], [6], [7, 8]]]]],
                dtype=dtypes.float32,
                row_splits_dtype=dtypes.int64)
    }
    expected_feature_list = {
        "b":
            ragged_factory_ops.constant(
                [[[[1], [2, 3, 4]], [[2, 4], [6]]], [],
                 [[[9], [8, 7, 6], [5]], [[4, 3, 2, 1]], [[0]]]],
                dtype=dtypes.float32,
                row_splits_dtype=dtypes.int32),
        "c":
            ragged_factory_ops.constant(
                [[[[1], [2], [3], [4]], [[2], [4], [6]]], [],
                 [[[9], [8], [7], [6], [5]], [[4], [3], [2], [1]], [[0]]]],
                ragged_rank=2,
                dtype=dtypes.float32,
                row_splits_dtype=dtypes.int32),
    }

    self._test(
        dict(
            serialized=ops.convert_to_tensor(serialized),
            context_features=context_features,
            sequence_features=sequence_features),
        expected_context,
        expected_feature_list,
        batch=True)

    self._test(
        dict(
            serialized=ops.convert_to_tensor(serialized)[0],
            context_features=context_features,
            sequence_features=sequence_features),
        {"a": expected_context["a"][0]}, {
            "b": expected_feature_list["b"][0],
            "c": expected_feature_list["c"][0]
        },
        batch=False)

  def testSerializedContainingMisalignedNestedRaggedFeature(self):
    # [FUNCTION DOC]
    # Scenario: a feature list with two value steps but only one row-splits step, with validation
    # requested. Expects an error (invalid argument or value error depending on mode) whose message
    # says values and partitions are not aligned, or the tensor is not a valid ragged tensor, or shapes
    # are incompatible.
    """FeatureList with 2 value tensors but only one splits tensor."""
    original = sequence_example(
        feature_lists=feature_lists({
            "b_values":
                feature_list(
                    [float_feature([1, 2, 3, 4]),
                     float_feature([2, 4, 6])]),
            "b_splits":
                feature_list([int64_feature([0, 1, 4])]),
        }))
    sequence_features = {
        "b":
            parsing_ops.RaggedFeature(
                value_key="b_values",
                dtype=dtypes.float32,
                partitions=[parsing_ops.RaggedFeature.RowSplits("b_splits")],
                validate=True)
    }
    self._testBoth(
        dict(
            serialized=ops.convert_to_tensor(original.SerializeToString()),
            sequence_features=sequence_features),
        expected_err=(
            (errors_impl.InvalidArgumentError, ValueError),
            # Message for batch=true:
            "Feature b: values and partitions are not aligned"
            # Message for batch=false in graph mode:
            "|.* do not form a valid RaggedTensor"
            # Message for batch=false in eager mode:
            "|Incompatible shapes|required broadcastable shapes"))


@test_util.run_all_in_graph_and_eager_modes
class DecodeRawTest(test.TestCase):
  # [CLASS DOC]
  # Tests for raw byte decoding: the first version (which needs all strings of equal length) and the
  # second version (which can pad or truncate to a fixed length and choose endianness). Provides
  # helpers to decode and to compute expected byte codes. Decorated to run in graph and eager mode.

  def _decode_v1(self, words):
    # [FUNCTION DOC]
    # HELPER: Runs the first version of raw decoding for the decode-raw tests.
    # Takes words, a list of strings. Converts it to a string constant, decodes to unsigned eight-bit
    # integers inside a cached session and returns the evaluated numpy result. Raises whatever errors
    # the op raises.
    with self.cached_session():
      examples = np.array(words)
      example_tensor = constant_op.constant(
          examples, shape=examples.shape, dtype=dtypes.string)
      byte_tensor = parsing_ops.decode_raw_v1(example_tensor, dtypes.uint8)
      return self.evaluate(byte_tensor)

  def _decode_v2(
      self, words, fixed_length=None, dtype=dtypes.uint8, little_endian=True
  ):
    # [FUNCTION DOC]
    # HELPER: Runs the second version of raw decoding for the decode-raw tests.
    # Takes words (list of strings), fixed_length (optional length in bytes; default none, meaning
    # use the input length), dtype (output element type, default unsigned eight-bit integer) and
    # little_endian (default true). Decodes inside a cached session and returns the evaluated result.
    # Raises whatever errors the op raises.
    with self.cached_session():
      examples = np.array(words)
      byte_tensor = parsing_ops.decode_raw(
          examples,
          dtype,
          little_endian=little_endian,
          fixed_length=fixed_length,
      )
      return self.evaluate(byte_tensor)

  def _ordinalize(self, words, fixed_length=None):
    # [FUNCTION DOC]
    # HELPER: Computes expected output for the decode-raw tests.
    # Takes words (list of strings) and fixed_length (optional; default is the length of the first word).
    # Returns a two-dimensional array with one row per word holding character codes, padded with zeros
    # (or truncated) to the fixed length.
    outputs = []
    if fixed_length is None:
      fixed_length = len(words[0])

    for word in words:
      output = []
      for i in range(fixed_length):
        if i < len(word):
          output.append(ord(word[i]))
        else:
          output.append(0)
      outputs.append(output)
    return np.array(outputs)

  def testDecodeRawV1EqualLength(self):
    # [FUNCTION DOC]
    # Scenario: two strings of equal length decoded with the first version. Expects shape and values
    # to equal the character codes.
    words = ["string1", "string2"]

    observed = self._decode_v1(words)
    expected = self._ordinalize(words)

    self.assertAllEqual(expected.shape, observed.shape)
    self.assertAllEqual(expected, observed)

  def testDecodeRawV2FallbackEqualLength(self):
    # [FUNCTION DOC]
    # Scenario: two strings of equal length decoded with the second version without a fixed length.
    # Expects the same result as the first version.
    words = ["string1", "string2"]

    observed = self._decode_v2(words)
    expected = self._ordinalize(words)

    self.assertAllEqual(expected.shape, observed.shape)
    self.assertAllEqual(expected, observed)

  def testDecodeRawV1VariableLength(self):
    # [FUNCTION DOC]
    # Scenario: strings of different lengths decoded with the first version. Expects an invalid-argument
    # error.
    words = ["string", "longer_string"]
    with self.assertRaises(errors_impl.InvalidArgumentError):
      self._decode_v1(words)

  def testDecodeRawV2FallbackVariableLength(self):
    # [FUNCTION DOC]
    # Scenario: strings of different lengths decoded with the second version and no fixed length.
    # Expects an invalid-argument error.
    words = ["string", "longer_string"]
    with self.assertRaises(errors_impl.InvalidArgumentError):
      self._decode_v2(words)

  def testDecodeRawV2VariableLength(self):
    # [FUNCTION DOC]
    # Scenario: strings of different lengths decoded with the second version with a fixed length of
    # eight. Expects zero padding to eight bytes per row, matching the computed character codes.
    words = ["string", "longer_string"]

    observed = self._decode_v2(words, fixed_length=8)
    expected = self._ordinalize(words, fixed_length=8)

    self.assertAllEqual(expected.shape, observed.shape)
    self.assertAllEqual(expected, observed)

  def testDecodeRawInvalidFixedLengthSize(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: The call that raises is the decode helper itself, so the evaluate wrapper around it
    # KNOWN ISSUE: is unnecessary (the helper already returns a numpy array).
    # Scenario: fixed length of seven with a four-byte element type. Expects a value error or
    # invalid-argument error saying the length must be a multiple of the element size (message depends on
    # whether shape inference or the kernel catches it).
    input_bytes = ["1"]
    # Different error messages depending on shape inference vs kernel.
    with self.assertRaisesRegex(
        (ValueError, errors_impl.InvalidArgumentError),
        "must be a multiple of|evenly divisible by",
    ):
      self.evaluate(
          self._decode_v2(input_bytes, fixed_length=7, dtype=dtypes.float32)
      )

  def testDecodeRawExtendedInputBytesLittleEndian(self):
    # [FUNCTION DOC]
    # Scenario: a two-byte input decoded as thirty-two-bit integers with fixed length eight, little
    # endian. Expects internal zero padding to element size so the first element is the bytes as a
    # little-endian number, the second is zero.
    # Input bytes properly padded internally to at least dtype size.
    input_bytes = ["\x01\x23"]
    observed = self._decode_v2(
        input_bytes, fixed_length=8, dtype=dtypes.int32, little_endian=True
    )
    expected = np.array([[0x00002301, 0]], dtype=np.int32)

    self.assertAllEqual(expected.shape, observed.shape)
    self.assertAllEqual(expected, observed)

  def testDecodeRawExtendedInputBytesBigEndian(self):
    # [FUNCTION DOC]
    # Scenario: same as the little-endian test but big endian with a byte string input. Expects the
    # two bytes to be the high-order bytes of the first element, the second element zero.
    # Input bytes properly padded internally to at least dtype size.
    input_bytes = [b"\x01\x23"]
    observed = self._decode_v2(
        input_bytes, fixed_length=8, dtype=dtypes.int32, little_endian=False
    )
    expected = np.array([[0x01230000, 0]], dtype=np.int32)

    self.assertAllEqual(expected.shape, observed.shape)
    self.assertAllEqual(expected, observed)


@test_util.run_all_in_graph_and_eager_modes
class DecodeJSONExampleTest(test.TestCase):
  # [CLASS DOC]
  # Tests for converting JSON text of example messages to binary serialized messages. Uses a
  # round trip helper. Decorated to run in graph and eager mode.

  def _testRoundTrip(self, examples):
    # [FUNCTION DOC]
    # HELPER: Shared round-trip check for the JSON decoding tests.
    # Takes examples (an arbitrarily nested list or a single example message, possibly empty). Converts
    # the messages to JSON text in a string tensor of the same shape, decodes with the decode-JSON op,
    # evaluates, and compares: for non-scalar input checks that shapes match and every decoded message
    # equals the original; for scalar input compares the single decoded message.
    examples = np.array(examples, dtype=np.object_)

    json_tensor = constant_op.constant(
        [json_format.MessageToJson(m) for m in examples.flatten()],
        shape=examples.shape,
        dtype=dtypes.string)
    binary_tensor = parsing_ops.decode_json_example(json_tensor)
    binary_val = self.evaluate(binary_tensor)

    if examples.shape:
      self.assertShapeEqual(binary_val, json_tensor)
      for input_example, output_binary in zip(
          np.array(examples).flatten(), binary_val.flatten()):
        output_example = example_pb2.Example()
        output_example.ParseFromString(output_binary)
        self.assertProtoEquals(input_example, output_example)
    else:
      output_example = example_pb2.Example()
      output_example.ParseFromString(binary_val)
      self.assertProtoEquals(examples.item(), output_example)

  def testEmptyTensor(self):
    # [FUNCTION DOC]
    # Scenario: an empty list and a list of empty lists. Expects the round trip to succeed.
    self._testRoundTrip([])
    self._testRoundTrip([[], [], []])

  def testEmptyExamples(self):
    # [FUNCTION DOC]
    # Scenario: three empty example messages. Expects the round trip to reproduce them.
    self._testRoundTrip([example(), example(), example()])

  def testDenseFeaturesScalar(self):
    # [FUNCTION DOC]
    # Scenario: one example (scalar input) with a float feature. Expects an identical message back.
    self._testRoundTrip(
        example(features=features({"a": float_feature([1, 1, 3])})))

  def testDenseFeaturesVector(self):
    # [FUNCTION DOC]
    # Scenario: a list of two examples with float features. Expects identical messages back.
    self._testRoundTrip([
        example(features=features({"a": float_feature([1, 1, 3])})),
        example(features=features({"a": float_feature([-1, -1, 2])})),
    ])

  def testDenseFeaturesMatrix(self):
    # [FUNCTION DOC]
    # Scenario: a two-by-one nested list of examples. Expects shape preserved and identical messages.
    self._testRoundTrip([
        [example(features=features({"a": float_feature([1, 1, 3])}))],
        [example(features=features({"a": float_feature([-1, -1, 2])}))],
    ])

  def testSparseFeatures(self):
    # [FUNCTION DOC]
    # Scenario: examples with present, empty, missing-value and combined float and bytes features.
    # Expects identical messages back.
    self._testRoundTrip([
        example(features=features({"st_c": float_feature([3, 4])})),
        example(features=features({"st_c": float_feature([])})),
        example(features=features({"st_d": feature()})),
        example(
            features=features({
                "st_c": float_feature([1, 2, -1]),
                "st_d": bytes_feature([b"hi"])
            })),
    ])

  def testSerializedContainingBytes(self):
    # [FUNCTION DOC]
    # Scenario: examples with float and bytes features, one feature name containing special characters.
    # Expects identical messages back.
    aname = "a"
    bname = "b*has+a:tricky_name"
    self._testRoundTrip([
        example(
            features=features({
                aname: float_feature([1, 1]),
                bname: bytes_feature([b"b0_str"])
            })),
        example(
            features=features({
                aname: float_feature([-1, -1]),
                bname: bytes_feature([b"b1"])
            })),
    ])

  def testInvalidSyntax(self):
    # [FUNCTION DOC]
    # Scenario: malformed JSON text. In eager mode the call itself must raise an invalid-argument error
    # saying parsing JSON failed; in graph mode the error must appear when the result is evaluated.
    json_tensor = constant_op.constant(["{]"])
    if context.executing_eagerly():
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  "Error while parsing JSON"):
        parsing_ops.decode_json_example(json_tensor)
    else:
      binary_tensor = parsing_ops.decode_json_example(json_tensor)
      with self.assertRaisesOpError("Error while parsing JSON"):
        self.evaluate(binary_tensor)


class ParseTensorOpTest(test.TestCase):
  # [CLASS DOC]
  # Tests for parsing a serialized tensor proto into a tensor. Not decorated for both modes; each test
  # is marked to run only as a deprecated version one test and uses a cached session with placeholder
  # feeds.

  @test_util.run_deprecated_v1
  def testToFloat32(self):
    # [FUNCTION DOC]
    # Scenario: a random float tensor is turned into a proto (byte swapped on big-endian hosts),
    # serialized, fed through a placeholder, and parsed as float32. Expects the result to equal the original.
    with self.cached_session():
      expected = np.random.rand(3, 4, 5).astype(np.float32)
      if sys.byteorder == "big":
        tensor_proto = tensor_util.make_tensor_proto(expected.byteswap())
      else:
        tensor_proto = tensor_util.make_tensor_proto(expected)

      serialized = array_ops.placeholder(dtypes.string)
      tensor = parsing_ops.parse_tensor(serialized, dtypes.float32)

      result = tensor.eval(
          feed_dict={serialized: tensor_proto.SerializeToString()})

      self.assertAllEqual(expected, result)

  @test_util.run_deprecated_v1
  def testToUint8(self):
    # [FUNCTION DOC]
    # Scenario: same as the float test but with an unsigned eight-bit tensor. Expects equality.
    with self.cached_session():
      expected = np.random.rand(3, 4, 5).astype(np.uint8)
      tensor_proto = tensor_util.make_tensor_proto(expected)

      serialized = array_ops.placeholder(dtypes.string)
      tensor = parsing_ops.parse_tensor(serialized, dtypes.uint8)

      result = tensor.eval(
          feed_dict={serialized: tensor_proto.SerializeToString()})

      self.assertAllEqual(expected, result)

  @test_util.run_deprecated_v1
  def testTypeMismatch(self):
    # [FUNCTION DOC]
    # Scenario: a proto holding eight-bit data parsed with a sixteen-bit type. Expects an op error that
    # the type of the parsed tensor does not match the requested type.
    with self.cached_session():
      expected = np.random.rand(3, 4, 5).astype(np.uint8)
      tensor_proto = tensor_util.make_tensor_proto(expected)

      serialized = array_ops.placeholder(dtypes.string)
      tensor = parsing_ops.parse_tensor(serialized, dtypes.uint16)

      with self.assertRaisesOpError(
          r"Type mismatch between parsed tensor \(uint8\) and dtype "
          r"\(uint16\)"):
        tensor.eval(feed_dict={serialized: tensor_proto.SerializeToString()})

  @test_util.run_deprecated_v1
  def testInvalidInput(self):
    # [FUNCTION DOC]
    # Scenario: feeds a string that is not a tensor proto, expecting an op error that it could not be
    # parsed (quoting it in base64), then feeds a vector of one string, expecting an op error that the
    # input must be a scalar.
    with self.cached_session():
      serialized = array_ops.placeholder(dtypes.string)
      tensor = parsing_ops.parse_tensor(serialized, dtypes.uint16)

      with self.assertRaisesOpError(
          "Could not parse `serialized` as TensorProto, base64: Ym9ndXM="):
        tensor.eval(feed_dict={serialized: "bogus"})

      with self.assertRaisesOpError(
          r"Expected `serialized` to be a scalar, got shape: \[1\]"):
        tensor.eval(feed_dict={serialized: ["bogus"]})


if __name__ == "__main__":
  test.main()
