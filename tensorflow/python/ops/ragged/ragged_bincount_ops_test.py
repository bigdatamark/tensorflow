# [MODULE DOC]
# KNOWN ISSUE: the license header contains a corrupted word ("maxlengthations" instead of
# "limitations"); comment-only typo.
# Unit tests for the bincount and sparse bincount ops when given ragged tensor inputs.
# Covers dense bincount (counts, binary output, weights, minlength/maxlength clipping and
# padding, axis none/0/-1, comparison with numpy), the sparse bincount variant (indices,
# values and dense shape for many option combinations) and its failure modes for
# mismatched weights types or row splits. Has no import-time side effects beyond defining
# test classes and two small helpers; run through the standard test main entry point.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
# maxlengthations under the License.
# ==============================================================================
"""Tests for bincount ops with RaggedTensor inputs."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import bincount_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test


def _ragged_factory(x):
  # [FUNCTION DOC]
  # HELPER: builds deferred test inputs for the parameterized bincount tests.
  # Takes a nested list of numbers and returns a function with no arguments that, when
  # called, creates a ragged tensor from that list. Deferring creation keeps tensor
  # construction out of test-parameter definition time.
  return lambda: ragged_factory_ops.constant(x)


def _adjust_expected_rank1(x, minlength, maxlength):
  # [FUNCTION DOC]
  # HELPER: computes the expected bincount output for the length options in tests.
  # Inputs: x (list of counts), minlength and maxlength (optional integers or nothing).
  # If x is shorter than minlength it is padded with zeros at the end; if longer than
  # maxlength it is truncated. Returns the adjusted list; the input list is not modified.
  """Trim or pad an expected result based on minlength and maxlength."""
  n = len(x)
  if (minlength is not None) and (n < minlength):
    x = x + [0] * (minlength - n)
  if (maxlength is not None) and (n > maxlength):
    x = x[:maxlength]
  return x


def _adjust_expected_rank2(x, minlength, maxlength):
  # [FUNCTION DOC]
  # HELPER: two-dimensional version of the expected-output length adjustment.
  # Applies the one-dimensional padding/truncation to each row of x (a list of lists)
  # using minlength and maxlength, and returns the new list of adjusted rows.
  return [_adjust_expected_rank1(i, minlength, maxlength) for i in x]


class TestDenseBincount(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests the dense bincount API with ragged tensor inputs, parameterized over integer
  # dtypes, length options, weights, binary output and axis. Has no shared fixtures;
  # each test builds its own ragged inputs and compares against hand-written expectations
  # or numpy results.

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_ragged_input_count(self, dtype):
    # [FUNCTION DOC]
    # Scenario: a ragged tensor (built with the given dtype) with empty rows and rows of
    # small integers. Action: bincount
    # per row (axis -1). Expects a dense matrix of per-row counts, zero rows for empty rows.
    x = ragged_factory_ops.constant([[], [], [3, 0, 1], [], [5, 0, 4, 4]],
                                    dtype)
    # pyformat: disable
    expected_output = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 2, 1]]
    # pyformat: enable
    self.assertAllEqual(expected_output,
                        self.evaluate(bincount_ops.bincount(arr=x, axis=-1)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_ragged_input_binary(self, dtype):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the dtype parameter is never used; the input is built without a dtype,
    # so the int32 and int64 variants run identical code.
    # Scenario: ragged rows including empty rows and a repeated value. Action: bincount per
    # row with binary output. Expects 1 wherever a value appears, never counts above 1.
    x = ragged_factory_ops.constant([[], [], [3, 0, 1], [], [5, 0, 4, 4]])
    # pyformat: disable
    expected_output = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 1]]
    # pyformat: enable
    self.assertAllEqual(
        expected_output,
        self.evaluate(
            bincount_ops.bincount(arr=x, axis=-1, binary_output=True)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_ragged_input_count_with_weights(self, dtype):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the dtype parameter is never used; the input is built without a dtype,
    # so both parameterized variants run identical code.
    # Scenario: ragged values with a ragged float weights tensor of identical shape.
    # Action: weighted bincount per row. Expects the sum of weights per value in each row,
    # compared with approximate float equality.
    x = ragged_factory_ops.constant([[], [], [3, 0, 1], [], [5, 0, 4, 4]])
    weights = ragged_factory_ops.constant([[], [], [.1, .2, .3], [],
                                           [.2, .5, .6, .3]])
    # pyformat: disable
    expected_output = [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [.2, .3, 0, .1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [.5, 0, 0, 0, .9, .2]]
    # pyformat: enable
    self.assertAllClose(
        expected_output,
        self.evaluate(bincount_ops.bincount(arr=x, weights=weights, axis=-1)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_ragged_input_count_np(self, dtype):
    # [FUNCTION DOC]
    # Scenario: a random dense integer matrix (fixed seed, 128 rows, 27 columns, values
    # below 1000) of the given dtype converted to a ragged tensor. Action: per-row bincount
    # with minlength 1000. Expects equality with numpy's per-row bincount stacked together.
    np.random.seed(42)
    num_rows = 128
    num_cols = 27
    size = 1000
    inp = np.random.randint(0, size, (num_rows, num_cols), dtype=dtype)
    np_out = np.reshape(
        np.concatenate(
            [np.bincount(inp[j, :], minlength=size) for j in range(num_rows)],
            axis=0), (num_rows, size))
    x = ragged_tensor.RaggedTensor.from_tensor(inp)
    self.assertAllEqual(
        np_out,
        self.evaluate(bincount_ops.bincount(arr=x, minlength=size, axis=-1)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_ragged_input_count_np_with_weights(self, dtype):
    # [FUNCTION DOC]
    # KNOWN ISSUE: float weights are compared with exact equality rather than approximate
    # equality, which can be fragile.
    # Scenario: random integer matrix and random float weights (fixed seed), both converted
    # to ragged tensors. Action: weighted per-row bincount with minlength 1000. Expects the
    # result to match numpy's weighted per-row bincount.
    np.random.seed(42)
    num_rows = 128
    num_cols = 27
    size = 1000
    inp = np.random.randint(0, size, (num_rows, num_cols), dtype=dtype)
    np_weight = np.random.random((num_rows, num_cols))
    np_out = np.reshape(
        np.concatenate([
            np.bincount(inp[j, :], weights=np_weight[j, :], minlength=size)
            for j in range(num_rows)
        ],
                       axis=0), (num_rows, size))
    x = ragged_tensor.RaggedTensor.from_tensor(inp)
    weights = ragged_tensor.RaggedTensor.from_tensor(np_weight)
    self.assertAllEqual(
        np_out,
        self.evaluate(
            bincount_ops.bincount(
                arr=x, weights=weights, minlength=size, axis=-1)))

  @parameterized.product(
      (
          dict(
              tid="_r2",
              x_factory=_ragged_factory([[], [1], [2, 2], [3, 3, 3]]),
              expected=[0, 1, 2, 3],  # no implied zeros
          ),
          dict(
              tid="_r3",
              x_factory=_ragged_factory([[[], [1]], [[2, 2], [3, 3, 3]]]),
              expected=[0, 1, 2, 3],  # no implied zeros
          ),
      ),
      (
          dict(minlength=None, maxlength=None),
          dict(minlength=3, maxlength=None),
          dict(minlength=5, maxlength=None),
          dict(minlength=None, maxlength=3),
          dict(minlength=None, maxlength=5),
          dict(minlength=2, maxlength=3),
          dict(minlength=3, maxlength=5),
          dict(minlength=5, maxlength=10),
          dict(minlength=None, maxlength=0),
      ),
  )
  def test_default(self, x_factory, minlength, maxlength, expected, tid=None):
    # [FUNCTION DOC]
    # Parameterized over a rank-2 and a rank-3 ragged input and many minlength/maxlength
    # combinations (tid only labels the case). Action: bincount with default axis and with
    # axis 0. Expects the same flattened count vector for both, padded or truncated per the
    # length options.
    x = x_factory()
    expected = _adjust_expected_rank1(expected, minlength, maxlength)
    self.assertAllEqual(
        expected,
        self.evaluate(
            bincount_ops.bincount(x, minlength=minlength, maxlength=maxlength)
        ),
    )
    self.assertAllEqual(
        expected,
        self.evaluate(
            bincount_ops.bincount(
                x, minlength=minlength, maxlength=maxlength, axis=0
            )
        ),
    )

  @parameterized.product(
      (
          dict(
              tid="_r2",
              x_factory=_ragged_factory([[], [1], [2, 2], [3, 3, 3]]),
              # no implied zeros
              expected=[[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0], [0, 0, 0, 3]],
          ),
      ),
      (
          dict(minlength=None, maxlength=None),
          dict(minlength=3, maxlength=None),
          dict(minlength=5, maxlength=None),
          dict(minlength=None, maxlength=3),
          dict(minlength=None, maxlength=5),
          dict(minlength=2, maxlength=3),
          dict(minlength=3, maxlength=5),
          dict(minlength=5, maxlength=10),
          dict(minlength=None, maxlength=0),
      ),
  )
  def test_axis_neg_1(self, tid, x_factory, minlength, maxlength, expected):
    # [FUNCTION DOC]
    # Parameterized over a rank-2 ragged input and length option combinations. Action:
    # bincount with axis -1. Expects a matrix with one count row per input row, each row
    # padded or truncated according to minlength and maxlength.
    x = x_factory()
    expected = _adjust_expected_rank2(expected, minlength, maxlength)
    self.assertAllEqual(
        expected,
        self.evaluate(
            bincount_ops.bincount(
                x, minlength=minlength, maxlength=maxlength, axis=-1
            )
        ),
    )

  @parameterized.product(
      (
          dict(
              tid="_r2",
              x_factory=_ragged_factory([[], [1], [2, 2], [3, 3, 3]]),
              weights_factory=_ragged_factory([[], [1], [2, 3], [4, 5, 6]]),
              axis=None,
              expected=[0, 1, 5, 15],  # no implied zeros
          ),
          dict(
              tid="_r3",
              x_factory=_ragged_factory([[[], [1]], [[2, 2], [3, 3, 3]]]),
              weights_factory=_ragged_factory([[[], [1]], [[2, 3], [4, 5, 6]]]),
              expected=[0, 1, 5, 15],  # no implied zeros
              axis=None,
          ),
          dict(
              tid="_r2_axis_neg_1",
              x_factory=_ragged_factory([[], [1], [2, 2], [3, 3, 3]]),
              weights_factory=_ragged_factory([[], [1], [2, 3], [4, 5, 6]]),
              # no implied zeros
              expected=[
                  [0, 0, 0, 0],
                  [0, 1, 0, 0],
                  [0, 0, 5, 0],
                  [0, 0, 0, 15],
              ],
              axis=-1,
          ),
      ),
      (
          dict(minlength=None, maxlength=None),
          dict(minlength=3, maxlength=None),
          dict(minlength=5, maxlength=None),
          dict(minlength=None, maxlength=3),
          dict(minlength=None, maxlength=5),
          dict(minlength=2, maxlength=3),
          dict(minlength=3, maxlength=5),
          dict(minlength=5, maxlength=10),
          dict(minlength=None, maxlength=0),
      ),
  )
  def test_weights(
      self,
      tid,
      x_factory,
      weights_factory,
      minlength,
      maxlength,
      expected,
      axis,
  ):
    # [FUNCTION DOC]
    # Parameterized over ragged inputs of rank 2 and 3, matching ragged weights, axis
    # (none or -1) and length options. Skipped on GPU without XLA because the dense
    # bincount GPU kernel lacks weights support. Action: weighted bincount. Expects summed
    # weights per value, as a vector (axis none) or per-row matrix (axis -1), adjusted by
    # minlength and maxlength.
    device_set = set([d.device_type for d in tf_config.list_physical_devices()])
    if "GPU" in device_set and not test_util.is_xla_enabled():
      self.skipTest(
          "b/263004039 The DenseBincount GPU kernel does not support weights."
          " unsorted_segment_sum should be used instead on GPU."
      )
    x = x_factory()
    weights = weights_factory()
    if axis == -1:
      expected = _adjust_expected_rank2(expected, minlength, maxlength)
    else:
      expected = _adjust_expected_rank1(expected, minlength, maxlength)
    self.assertAllEqual(
        expected,
        self.evaluate(
            bincount_ops.bincount(
                x,
                weights=weights,
                minlength=minlength,
                maxlength=maxlength,
                axis=axis,
            )
        ),
    )

  @parameterized.product(
      (
          dict(
              tid="_r2",
              x_factory=_ragged_factory([[], [1], [2, 2], [3, 3, 3]]),
              expected=[0, 1, 1, 1],  # no implied zeros
              axis=None,
          ),
          dict(
              tid="_r3",
              x_factory=_ragged_factory([[[], [1]], [[2, 2], [3, 3, 3]]]),
              expected=[0, 1, 1, 1],  # no implied zeros
              axis=None,
          ),
          dict(
              tid="_r2_axis_neg_1",
              x_factory=_ragged_factory([[], [1], [2, 2], [3, 3, 3]]),
              # no implied zeros
              expected=[[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
              axis=-1,
          ),
      ),
      (
          dict(minlength=None, maxlength=None),
          dict(minlength=3, maxlength=None),
          dict(minlength=5, maxlength=None),
          dict(minlength=None, maxlength=3),
          dict(minlength=None, maxlength=5),
          dict(minlength=2, maxlength=3),
          dict(minlength=3, maxlength=5),
          dict(minlength=5, maxlength=10),
          dict(minlength=None, maxlength=0),
      ),
  )
  def test_binary_output(
      self,
      tid,
      x_factory,
      minlength,
      maxlength,
      expected,
      axis=None,
  ):
    # [FUNCTION DOC]
    # Parameterized over ragged inputs of rank 2 and 3, axis (none or -1) and length
    # options. Action: bincount with binary output enabled. Expects 1 for every value
    # present (vector for axis none, per-row matrix for axis -1), adjusted by minlength
    # and maxlength.
    x = x_factory()
    if axis == -1:
      expected = _adjust_expected_rank2(expected, minlength, maxlength)
    else:
      expected = _adjust_expected_rank1(expected, minlength, maxlength)
    self.assertAllEqual(
        expected,
        self.evaluate(
            bincount_ops.bincount(
                x,
                minlength=minlength,
                maxlength=maxlength,
                binary_output=True,
                axis=axis,
            )
        ),
    )


class TestSparseCount(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests the sparse bincount API with ragged inputs through a single table-driven test
  # covering max/min length, binary output, weights, one-dimensional input, flattening
  # over all axes and very large values.
  @parameterized.named_parameters(
      {
          "testcase_name": "_no_maxlength",
          "x": [[], [], [3, 0, 1], [], [5, 0, 4, 4]],
          "expected_indices": [[2, 0], [2, 1], [2, 3], [4, 0], [4, 4], [4, 5]],
          "expected_values": [1, 1, 1, 1, 2, 1],
          "expected_shape": [5, 6],
      }, {
          "testcase_name": "_maxlength",
          "x": [[], [], [3, 0, 1], [7], [5, 0, 4, 4]],
          "maxlength": 7,
          "expected_indices": [[2, 0], [2, 1], [2, 3], [4, 0], [4, 4], [4, 5]],
          "expected_values": [1, 1, 1, 1, 2, 1],
          "expected_shape": [5, 7],
      }, {
          "testcase_name": "_maxlength_zero",
          "x": [[], [], [3, 0, 1], [7], [5, 0, 4, 4]],
          "maxlength": 0,
          "expected_indices": np.empty([0, 2], dtype=np.int64),
          "expected_values": [],
          "expected_shape": [5, 0],
      }, {
          "testcase_name": "_minlength",
          "x": [[], [], [3, 0, 1], [7], [5, 0, 4, 4]],
          "minlength": 9,
          "expected_indices": [[2, 0], [2, 1], [2, 3], [3, 7], [4, 0], [4, 4],
                               [4, 5]],
          "expected_values": [1, 1, 1, 1, 1, 2, 1],
          "expected_shape": [5, 9],
      }, {
          "testcase_name": "_minlength_larger_values",
          "x": [[], [], [3, 0, 1], [7], [5, 0, 4, 4]],
          "minlength": 3,
          "expected_indices": [[2, 0], [2, 1], [2, 3], [3, 7], [4, 0], [4, 4],
                               [4, 5]],
          "expected_values": [1, 1, 1, 1, 1, 2, 1],
          "expected_shape": [5, 8],
      }, {
          "testcase_name": "_no_maxlength_binary",
          "x": [[], [], [3, 0, 1], [], [5, 0, 4, 4]],
          "expected_indices": [[2, 0], [2, 1], [2, 3], [4, 0], [4, 4], [4, 5]],
          "expected_values": [1, 1, 1, 1, 1, 1],
          "expected_shape": [5, 6],
          "binary_output": True,
      }, {
          "testcase_name": "_maxlength_binary",
          "x": [[], [], [3, 0, 1], [7], [5, 0, 4, 4]],
          "maxlength": 7,
          "expected_indices": [[2, 0], [2, 1], [2, 3], [4, 0], [4, 4], [4, 5]],
          "expected_values": [1, 1, 1, 1, 1, 1],
          "expected_shape": [5, 7],
          "binary_output": True,
      }, {
          "testcase_name": "_minlength_binary",
          "x": [[], [], [3, 0, 1], [7], [5, 0, 4, 4]],
          "minlength": 9,
          "expected_indices": [[2, 0], [2, 1], [2, 3], [3, 7], [4, 0], [4, 4],
                               [4, 5]],
          "expected_values": [1, 1, 1, 1, 1, 1, 1],
          "expected_shape": [5, 9],
          "binary_output": True,
      }, {
          "testcase_name": "_minlength_larger_values_binary",
          "x": [[], [], [3, 0, 1], [7], [5, 0, 4, 4]],
          "minlength": 3,
          "binary_output": True,
          "expected_indices": [[2, 0], [2, 1], [2, 3], [3, 7], [4, 0], [4, 4],
                               [4, 5]],
          "expected_values": [1, 1, 1, 1, 1, 1, 1],
          "expected_shape": [5, 8],
      }, {
          "testcase_name": "_no_maxlength_weights",
          "x": [[], [], [3, 0, 1], [], [5, 0, 4, 4]],
          "expected_indices": [[2, 0], [2, 1], [2, 3], [4, 0], [4, 4], [4, 5]],
          "expected_values": [0.5, 2, 6, 0.25, 8, 10],
          "expected_shape": [5, 6],
          "weights": [[], [], [6, 0.5, 2], [], [10, 0.25, 5, 3]],
      }, {
          "testcase_name": "_maxlength_weights",
          "x": [[], [], [3, 0, 1], [7], [5, 0, 4, 4]],
          "maxlength": 7,
          "expected_indices": [[2, 0], [2, 1], [2, 3], [4, 0], [4, 4], [4, 5]],
          "expected_values": [0.5, 2, 6, 0.25, 8, 10],
          "expected_shape": [5, 7],
          "weights": [[], [], [6, 0.5, 2], [14], [10, 0.25, 5, 3]],
      }, {
          "testcase_name": "_minlength_weights",
          "x": [[], [], [3, 0, 1], [7], [5, 0, 4, 4]],
          "minlength": 9,
          "expected_indices": [[2, 0], [2, 1], [2, 3], [3, 7], [4, 0], [4, 4],
                               [4, 5]],
          "expected_values": [0.5, 2, 6, 14, 0.25, 8, 10],
          "expected_shape": [5, 9],
          "weights": [[], [], [6, 0.5, 2], [14], [10, 0.25, 5, 3]],
      }, {
          "testcase_name": "_minlength_larger_values_weights",
          "x": [[], [], [3, 0, 1], [7], [5, 0, 4, 4]],
          "minlength": 3,
          "expected_indices": [[2, 0], [2, 1], [2, 3], [3, 7], [4, 0], [4, 4],
                               [4, 5]],
          "expected_values": [0.5, 2, 6, 14, 0.25, 8, 10],
          "expected_shape": [5, 8],
          "weights": [[], [], [6, 0.5, 2], [14], [10, 0.25, 5, 3]],
      }, {
          "testcase_name": "_1d",
          "x": [3, 0, 1, 1],
          "expected_indices": [[0], [1], [3]],
          "expected_values": [1, 2, 1],
          "expected_shape": [4],
      }, {
          "testcase_name": "_all_axes",
          "x": [[], [], [3, 0, 1], [], [5, 0, 4, 4]],
          "expected_indices": [[0], [1], [3], [4], [5]],
          "expected_values": [2, 1, 1, 2, 1],
          "expected_shape": [6],
          "axis": None,
      }, {
          "testcase_name": "_large_inputs",
          "x": [[1941591354222760687, 1748591354222760687],
                [1941591354222760687, 1241591354229760689, 1241591354229760687]
               ],
          "expected_indices": [[1241591354229760687], [1241591354229760689],
                               [1748591354222760687], [1941591354222760687]],
          "expected_values": [1, 1, 1, 2],
          "expected_shape": [1941591354222760687 + 1],
          "axis": None
      })
  def test_ragged_input(self,
                        x,
                        expected_indices,
                        expected_values,
                        expected_shape,
                        maxlength=None,
                        minlength=None,
                        binary_output=False,
                        weights=None,
                        axis=-1):
    # [FUNCTION DOC]
    # Table-driven test. Inputs: x (nested list of integers), the expected sparse indices,
    # values and dense shape, and optional maxlength, minlength, binary_output, weights
    # (nested list) and axis (default -1). Builds ragged tensors from x and weights, runs
    # sparse bincount, and asserts the output indices, values and dense shape equal the
    # expectations. Note that the one-dimensional case passes a plain ragged constant.
    x_ragged = ragged_factory_ops.constant(x)
    w = ragged_factory_ops.constant(weights) if weights is not None else None
    y = sparse_ops.sparse_bincount(
        x_ragged,
        weights=w,
        minlength=minlength,
        maxlength=maxlength,
        binary_output=binary_output,
        axis=axis)
    self.assertAllEqual(expected_indices, y.indices)
    self.assertAllEqual(expected_values, y.values)
    self.assertAllEqual(expected_shape, y.dense_shape)


class TestSparseCountFailureModes(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests that sparse bincount rejects incompatible combinations of values and weights
  # (different tensor kinds or mismatching row splits) with the expected errors.
  def test_dense_input_ragged_weights_fails(self):
    # [FUNCTION DOC]
    # Scenario: dense integer matrix values with ragged weights. Action: sparse bincount
    # on axis -1. Expects a value error saying weights must be a plain tensor.
    x = np.array([[3, 2, 1], [5, 4, 4]], dtype=np.int32)
    weights = ragged_factory_ops.constant([[6, 0.5, 2], [14], [10, 0.25, 5, 3]])
    with self.assertRaisesRegex(ValueError, "must be a tf.Tensor"):
      self.evaluate(sparse_ops.sparse_bincount(x, weights=weights, axis=-1))

  def test_sparse_input_ragged_weights_fails(self):
    # [FUNCTION DOC]
    # Scenario: sparse tensor values with ragged weights. Action: sparse bincount on axis
    # -1. Expects a value error saying weights must be a sparse tensor.
    x = sparse_ops.from_dense(
        np.array([[3, 0, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4]], dtype=np.int32))
    weights = ragged_factory_ops.constant([[6, 0.5, 2], [14], [10, 0.25, 5, 3]])
    with self.assertRaisesRegex(ValueError, "must be a SparseTensor"):
      self.evaluate(sparse_ops.sparse_bincount(x, weights=weights, axis=-1))

  def test_ragged_input_dense_weights_fails(self):
    # [FUNCTION DOC]
    # Scenario: ragged values with dense array weights. Action: sparse bincount on axis -1.
    # Expects a value error saying weights must be a ragged tensor.
    x = ragged_factory_ops.constant([[6, 1, 2], [14], [10, 1, 5, 3]])
    weights = np.array([[3, 2, 1], [5, 4, 4]], dtype=np.int32)
    with self.assertRaisesRegex(ValueError, "must be a RaggedTensor"):
      self.evaluate(sparse_ops.sparse_bincount(x, weights=weights, axis=-1))

  def test_ragged_input_sparse_weights_fails(self):
    # [FUNCTION DOC]
    # Scenario: ragged values with sparse tensor weights. Action: sparse bincount on axis
    # -1. Expects a value error saying weights must be a ragged tensor.
    x = ragged_factory_ops.constant([[6, 1, 2], [14], [10, 1, 5, 3]])
    weights = sparse_ops.from_dense(
        np.array([[3, 0, 1, 0], [0, 0, 0, 0], [5, 0, 4, 4]], dtype=np.int32))
    with self.assertRaisesRegex(ValueError, "must be a RaggedTensor"):
      self.evaluate(sparse_ops.sparse_bincount(x, weights=weights, axis=-1))

  def test_ragged_input_different_shape_fails(self):
    # [FUNCTION DOC]
    # Scenario: ragged values and ragged weights whose row lengths differ. Action: sparse
    # bincount on axis -1. Expects an invalid-argument error stating the row splits must
    # be the same.
    x = ragged_factory_ops.constant([[6, 1, 2], [14], [10, 1, 5, 3]])
    weights = ragged_factory_ops.constant([[6, 0.5, 2], [], [10, 0.25, 5, 3]])
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "must have the same row splits"):
      self.evaluate(sparse_ops.sparse_bincount(x, weights=weights, axis=-1))

if __name__ == "__main__":
  test.main()
