# [MODULE DOC]
# Tests for the bincount operation and the raw count kernels. One group (dense bincount tests)
# checks the public bincount function against reference results for sparse inputs (with and
# without weights, binary output, flattening or per-row counting) and for dense inputs of rank 1
# to 3 across many combinations of minimum and maximum output lengths, plus weighted and binary
# modes. Two further groups call the low-level sparse and ragged count kernels with malformed
# arguments and assert the specific error messages (heap out-of-bounds regression and input
# validation). Includes two small helpers that adjust expected results for length bounds. No
# import-time side effects; running as a script starts the standard test runner.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for bincount ops."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import bincount_ops
from tensorflow.python.ops import gen_count_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


def _adjust_expected_rank1(x, minlength, maxlength):
  # [FUNCTION DOC]
  # HELPER: computes expected results for the bincount tests.
  # Inputs: x, a Python list of expected counts; minlength and maxlength, optional bounds (nothing
  # means no bound). Pads the list with zeros at the end up to minlength if it is shorter, and
  # truncates it to maxlength if longer (the maximum is applied after padding, so it wins when the
  # bounds conflict). Returns the adjusted list; the input list is not modified.
  """Trim or pad an expected result based on minlength and maxlength."""
  n = len(x)
  if (minlength is not None) and (n < minlength):
    x = x + [0] * (minlength - n)
  if (maxlength is not None) and (n > maxlength):
    x = x[:maxlength]
  return x


def _adjust_expected_rank2(x, minlength, maxlength):
  # [FUNCTION DOC]
  # HELPER: row-wise version of the rank-one expectation adjuster for the bincount tests.
  # Inputs: x, a list of rows of expected counts; minlength and maxlength, optional bounds.
  # Returns a new list in which every row is padded or truncated as the rank-one helper does.
  return [_adjust_expected_rank1(i, minlength, maxlength) for i in x]


class TestDenseBincount(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for the public bincount function: sparse inputs (counting, weights,
  # binary output, whole-tensor and per-row reductions) and dense inputs of various ranks with a
  # grid of minimum and maximum length settings, including weighted and binary variants. Has no
  # shared fixtures; parameters come from the parameterized decorators.

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_sparse_input_all_count(self, dtype):
    # [FUNCTION DOC]
    # Test (parameterized over 32-bit and 64-bit integer dtype): builds a sparse 4096 by 1 tensor
    # with 128 random values below 1000, counts them with axis 0 and a minimum length of 1000,
    # and asserts the result equals the reference count (implicit zeros are not counted).
    np.random.seed(42)
    num_rows = 4096
    size = 1000
    n_elems = 128
    inp_indices = np.random.randint(0, num_rows, (n_elems, 1))
    inp_indices = np.concatenate([inp_indices, np.zeros((n_elems, 1))], axis=1)
    inp_vals = np.random.randint(0, size, (n_elems,), dtype=dtype)
    sparse_inp = sparse_tensor.SparseTensor(inp_indices, inp_vals,
                                            [num_rows, 1])

    # Note that the default for sparse tensors is to not count implicit zeros.
    np_out = np.bincount(inp_vals, minlength=size)
    self.assertAllEqual(
        np_out,
        self.evaluate(
            bincount_ops.bincount(sparse_inp, axis=0, minlength=size)
        ),
    )

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_sparse_input_all_count_with_weights(self, dtype):
    # [FUNCTION DOC]
    # Test (parameterized over integer dtype): a sparse input of 128 random values (the last one
    # forced to the maximum so the output has exactly 1000 bins) with a sparse tensor of random
    # float weights at the same positions; asserts the weighted bincount over axis 0 equals the
    # reference weighted count.
    np.random.seed(42)
    num_rows = 4096
    size = 1000
    n_elems = 128
    inp_indices = np.random.randint(0, num_rows, (n_elems, 1))
    inp_indices = np.concatenate([inp_indices, np.zeros((n_elems, 1))], axis=1)
    inp_vals = np.random.randint(0, size, (n_elems-1,), dtype=dtype)
    # Add an element with value `size-1` to input so bincount output has `size`
    # elements.
    inp_vals = np.concatenate([inp_vals, [size-1]], axis=0)
    sparse_inp = sparse_tensor.SparseTensor(inp_indices, inp_vals,
                                            [num_rows, 1])
    weight_vals = np.random.random((n_elems,))
    sparse_weights = sparse_tensor.SparseTensor(inp_indices, weight_vals,
                                                [num_rows, 1])

    np_out = np.bincount(inp_vals, minlength=size, weights=weight_vals)
    self.assertAllEqual(
        np_out,
        self.evaluate(bincount_ops.bincount(
            sparse_inp, sparse_weights, axis=0)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_sparse_input_all_binary(self, dtype):
    # [FUNCTION DOC]
    # Test (parameterized over integer dtype): sparse input of 128 random values below 10 (so
    # every value appears); asserts binary-output bincount yields a vector of ten ones.
    np.random.seed(42)
    num_rows = 4096
    size = 10
    n_elems = 128
    inp_indices = np.random.randint(0, num_rows, (n_elems, 1))
    inp_indices = np.concatenate([inp_indices, np.zeros((n_elems, 1))], axis=1)
    inp_vals = np.random.randint(0, size, (n_elems,), dtype=dtype)
    sparse_inp = sparse_tensor.SparseTensor(inp_indices, inp_vals,
                                            [num_rows, 1])

    np_out = np.ones((size,))
    self.assertAllEqual(
        np_out,
        self.evaluate(bincount_ops.bincount(sparse_inp, binary_output=True)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_sparse_input_col_reduce_count(self, dtype):
    # [FUNCTION DOC]
    # Test (parameterized over integer dtype): a random 128 by 27 matrix of values below 100 is
    # converted to a sparse tensor on the CPU (shifted by one so zeros survive the conversion,
    # then shifted back); asserts bincount with axis -1 gives per-row counts matching the
    # reference computed with a per-row count.
    num_rows = 128
    num_cols = 27
    size = 100
    np.random.seed(42)
    inp = np.random.randint(0, size, (num_rows, num_cols), dtype=dtype)
    np_out = np.reshape(
        np.concatenate(
            [np.bincount(inp[j, :], minlength=size) for j in range(num_rows)],
            axis=0), (num_rows, size))
    # from_dense will filter out 0s.
    inp = inp + 1
    # from_dense will cause OOM in GPU.
    with ops.device("/CPU:0"):
      inp_sparse = sparse_ops.from_dense(inp)
      inp_sparse = sparse_tensor.SparseTensor(inp_sparse.indices,
                                              inp_sparse.values - 1,
                                              inp_sparse.dense_shape)
    self.assertAllEqual(
        np_out, self.evaluate(bincount_ops.bincount(arr=inp_sparse, axis=-1)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_sparse_input_col_reduce_binary(self, dtype):
    # [FUNCTION DOC]
    # Test (parameterized over integer dtype): same setup as the per-row count test but with
    # binary output; asserts each row of the result holds one where the value occurs in the row
    # and zero elsewhere.
    num_rows = 128
    num_cols = 27
    size = 100
    np.random.seed(42)
    inp = np.random.randint(0, size, (num_rows, num_cols), dtype=dtype)
    np_out = np.reshape(
        np.concatenate([
            np.where(np.bincount(inp[j, :], minlength=size) > 0, 1, 0)
            for j in range(num_rows)
        ],
                       axis=0), (num_rows, size))
    # from_dense will filter out 0s.
    inp = inp + 1
    # from_dense will cause OOM in GPU.
    with ops.device("/CPU:0"):
      inp_sparse = sparse_ops.from_dense(inp)
      inp_sparse = sparse_tensor.SparseTensor(inp_sparse.indices,
                                              inp_sparse.values - 1,
                                              inp_sparse.dense_shape)
    self.assertAllEqual(
        np_out,
        self.evaluate(
            bincount_ops.bincount(arr=inp_sparse, axis=-1, binary_output=True)))

  @parameterized.product(
      (
          dict(
              tid="_d1",
              x=[1, 2, 2, 3, 3, 3],
              expected=[0, 1, 2, 3],
          ),
          dict(
              tid="_d2",
              x=[[0, 0, 0], [0, 1, 0], [2, 0, 2], [3, 3, 3]],
              expected=[6, 1, 2, 3],
          ),
          dict(
              tid="_d3",
              x=[[[0, 0, 0], [0, 1, 0]], [[2, 0, 2], [3, 3, 3]]],
              expected=[6, 1, 2, 3],
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
  def test_default(
      self,
      x,
      minlength,
      maxlength,
      expected,
      tid=None,
  ):
    # [FUNCTION DOC]
    # Test (parameterized over rank 1, 2 and 3 inputs crossed with nine minimum/maximum length
    # settings; the test id parameter only labels the case): the expected counts are adjusted
    # for the length bounds, then asserts bincount matches both with the default axis and with
    # axis 0 (full flattening).
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
              tid="_d2",
              x=[[0, 0, 0], [0, 1, 0], [2, 0, 2], [3, 3, 3]],
              expected=[[3, 0, 0, 0], [2, 1, 0, 0], [1, 0, 2, 0], [0, 0, 0, 3]],
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
  def test_axis_neg_1(
      self, tid, x, minlength, maxlength, expected
  ):
    # [FUNCTION DOC]
    # Test (parameterized over one 4 by 3 matrix crossed with the nine length settings): asserts
    # bincount with axis -1 yields per-row counts, with each row padded or truncated according to
    # the minimum and maximum length.
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
              tid="_d1",
              x=[1, 2, 2, 3, 3, 3],
              weights=[1, 2, 3, 4, 5, 6],
              axis=None,
              expected=[0, 1, 5, 15],
          ),
          dict(
              tid="_d2",
              x=[[0, 0, 0], [0, 1, 0], [2, 0, 2], [3, 3, 3]],
              weights=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
              axis=None,
              expected=[24, 5, 16, 33],
          ),
          dict(
              tid="_d3",
              x=[[[0, 0, 0], [0, 1, 0]], [[2, 0, 2], [3, 3, 3]]],
              weights=[[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]],
              axis=None,
              expected=[24, 5, 16, 33],
          ),
          dict(
              tid="_d2_axis_neg_1",
              x=[[0, 0, 0], [0, 1, 0], [2, 0, 2], [3, 3, 3]],
              weights=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
              axis=-1,
              expected=[
                  [6, 0, 0, 0],
                  [10, 5, 0, 0],
                  [8, 0, 16, 0],
                  [0, 0, 0, 33],
              ],
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
      x,
      weights,
      minlength,
      maxlength,
      expected,
      axis=None,
  ):
    # [FUNCTION DOC]
    # Test (parameterized over several inputs with weights, with and without axis -1, crossed
    # with the nine length settings): skipped on GPU without XLA because the dense bincount GPU
    # kernel does not support weights. Otherwise asserts weighted bincount equals the expected
    # sums, adjusted per row for axis -1 or as a vector otherwise.
    device_set = set([d.device_type for d in tf_config.list_physical_devices()])
    if "GPU" in device_set and not test_util.is_xla_enabled():
      self.skipTest(
          "b/263004039 The DenseBincount GPU kernel does not support weights."
          " unsorted_segment_sum should be used instead on GPU."
      )
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
              tid="_d1",
              x=[1, 2, 2, 3, 3, 3],
              expected=[0, 1, 1, 1],
              axis=None,
          ),
          dict(
              tid="_d2",
              x=[[0, 0, 0], [0, 1, 0], [2, 0, 2], [3, 3, 3]],
              expected=[1, 1, 1, 1],
              axis=None,
          ),
          dict(
              tid="_d3",
              x=[[[0, 0, 0], [0, 1, 0]], [[2, 0, 2], [3, 3, 3]]],
              expected=[1, 1, 1, 1],
              axis=None,
          ),
          dict(
              tid="_d2_axis_neg_1",
              x=[[0, 0, 0], [0, 1, 0], [2, 0, 2], [3, 3, 3]],
              expected=[[1, 0, 0, 0], [1, 1, 0, 0], [1, 0, 1, 0], [0, 0, 0, 1]],
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
      x,
      minlength,
      maxlength,
      expected,
      axis=None,
  ):
    # [FUNCTION DOC]
    # Test (parameterized over several inputs, with and without axis -1, crossed with the nine
    # length settings): asserts bincount with binary output yields ones for present values and
    # zeros elsewhere, adjusted per row for axis -1 or as a vector otherwise.
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


class RawOpsHeapOobTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case holding a security regression test for the raw sparse count kernel; graph-mode
  # (version 1) only. No shared fixtures.

  @test_util.run_v1_only("Test security error")
  def testSparseCountSparseOutputBadIndicesShapeTooSmall(self):
    # [FUNCTION DOC]
    # Test (version 1 only, security regression): calls the raw sparse count kernel with a rank-1
    # indices argument where rank 2 is required, and asserts it fails with a value error saying
    # the shape must be rank 2 but is rank 1, instead of reading out of bounds.
    indices = [1]
    values = [[1]]
    weights = []
    dense_shape = [10]
    with self.assertRaisesRegex(ValueError,
                                "Shape must be rank 2 but is rank 1 for"):
      self.evaluate(
          gen_count_ops.SparseCountSparseOutput(
              indices=indices,
              values=values,
              dense_shape=dense_shape,
              weights=weights,
              binary_output=True))


@test_util.run_all_in_graph_and_eager_modes
@test_util.disable_tfrt
class RawOpsTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for argument validation of the raw sparse count and ragged count kernels, run in
  # both graph and eager modes (and excluded from the TFRT runtime). Each test passes malformed
  # indices, weights, values or splits and asserts an invalid-argument error with a specific
  # message, except one test that checks a valid ragged call.

  def testSparseCountSparseOutputBadIndicesShape(self):
    # [FUNCTION DOC]
    # Test: gives the sparse count kernel a three-dimensional indices tensor and asserts an
    # invalid-argument error saying the indices must be a 2-dimensional tensor.
    indices = [[[0], [0]], [[0], [1]], [[1], [0]], [[1], [2]]]
    values = [1, 1, 1, 10]
    weights = [1, 2, 4, 6]
    dense_shape = [2, 3]
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Input indices must be a 2-dimensional tensor"):
      self.evaluate(
          gen_count_ops.SparseCountSparseOutput(
              indices=indices,
              values=values,
              dense_shape=dense_shape,
              weights=weights,
              binary_output=False))

  def testSparseCountSparseOutputBadWeightsShape(self):
    # [FUNCTION DOC]
    # Test: gives the sparse count kernel four values but only three weights and asserts an
    # invalid-argument error saying weights and values must have the same shape.
    indices = [[0, 0], [0, 1], [1, 0], [1, 2]]
    values = [1, 1, 1, 10]
    weights = [1, 2, 4]
    dense_shape = [2, 3]
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Weights and values must have the same shape"):
      self.evaluate(
          gen_count_ops.SparseCountSparseOutput(
              indices=indices,
              values=values,
              dense_shape=dense_shape,
              weights=weights,
              binary_output=False))

  def testSparseCountSparseOutputBadNumberOfValues(self):
    # [FUNCTION DOC]
    # Test: gives the sparse count kernel three index rows but four values and asserts an
    # invalid-argument error saying the number of values must match the first dimension of the
    # indices.
    indices = [[0, 0], [0, 1], [1, 0]]
    values = [1, 1, 1, 10]
    weights = [1, 2, 4, 6]
    dense_shape = [2, 3]
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Number of values must match first dimension of indices"):
      self.evaluate(
          gen_count_ops.SparseCountSparseOutput(
              indices=indices,
              values=values,
              dense_shape=dense_shape,
              weights=weights,
              binary_output=False))

  def testSparseCountSparseOutputNegativeValue(self):
    # [FUNCTION DOC]
    # Test: gives the sparse count kernel a negative value and asserts an invalid-argument error
    # saying input values must all be non-negative.
    indices = [[0, 0], [0, 1], [1, 0], [1, 2]]
    values = [1, 1, -1, 10]
    dense_shape = [2, 3]
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Input values must all be non-negative"):
      self.evaluate(
          gen_count_ops.SparseCountSparseOutput(
              indices=indices,
              values=values,
              dense_shape=dense_shape,
              binary_output=False))

  def testRaggedCountSparseOutput(self):
    # [FUNCTION DOC]
    # Test: valid call of the ragged count kernel with splits 0, 4, 7, seven values and weights;
    # asserts the sparse output has the expected indices (per row and value), summed weights
    # (7, 3, 5, 7, 6) and dense shape of 2 rows by 11 columns.
    splits = [0, 4, 7]
    values = [1, 1, 2, 1, 2, 10, 5]
    weights = [1, 2, 3, 4, 5, 6, 7]
    output_indices, output_values, output_shape = self.evaluate(
        gen_count_ops.RaggedCountSparseOutput(
            splits=splits, values=values, weights=weights, binary_output=False))
    self.assertAllEqual([[0, 1], [0, 2], [1, 2], [1, 5], [1, 10]],
                        output_indices)
    self.assertAllEqual([7, 3, 5, 7, 6], output_values)
    self.assertAllEqual([2, 11], output_shape)

  def testRaggedCountSparseOutputBadWeightsShape(self):
    # [FUNCTION DOC]
    # Test: ragged count kernel with seven values but six weights; asserts an invalid-argument
    # error saying weights and values must have the same shape.
    splits = [0, 4, 7]
    values = [1, 1, 2, 1, 2, 10, 5]
    weights = [1, 2, 3, 4, 5, 6]
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Weights and values must have the same shape"):
      self.evaluate(
          gen_count_ops.RaggedCountSparseOutput(
              splits=splits,
              values=values,
              weights=weights,
              binary_output=False))

  def testRaggedCountSparseOutputEmptySplits(self):
    # [FUNCTION DOC]
    # Test: ragged count kernel with an empty splits list; asserts an invalid-argument error
    # saying at least two elements must be provided for the splits argument.
    splits = []
    values = [1, 1, 2, 1, 2, 10, 5]
    weights = [1, 2, 3, 4, 5, 6, 7]
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Must provide at least 2 elements for the splits argument"):
      self.evaluate(
          gen_count_ops.RaggedCountSparseOutput(
              splits=splits,
              values=values,
              weights=weights,
              binary_output=False))

  def testRaggedCountSparseOutputBadSplitsStart(self):
    # [FUNCTION DOC]
    # Test: ragged count kernel with splits beginning at 1 instead of 0; asserts an
    # invalid-argument error saying splits must start with 0.
    splits = [1, 7]
    values = [1, 1, 2, 1, 2, 10, 5]
    weights = [1, 2, 3, 4, 5, 6, 7]
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Splits must start with 0"):
      self.evaluate(
          gen_count_ops.RaggedCountSparseOutput(
              splits=splits,
              values=values,
              weights=weights,
              binary_output=False))

  def testRaggedCountSparseOutputBadSplitsEnd(self):
    # [FUNCTION DOC]
    # Test: ragged count kernel with splits ending at 5 although there are seven values; asserts
    # an invalid-argument error saying splits must end with the number of values.
    splits = [0, 5]
    values = [1, 1, 2, 1, 2, 10, 5]
    weights = [1, 2, 3, 4, 5, 6, 7]
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Splits must end with the number of values"):
      self.evaluate(
          gen_count_ops.RaggedCountSparseOutput(
              splits=splits,
              values=values,
              weights=weights,
              binary_output=False))

  def testRaggedCountSparseOutputNegativeValue(self):
    # [FUNCTION DOC]
    # Test: ragged count kernel with a negative value among the inputs (no weights); asserts an
    # invalid-argument error saying input values must all be non-negative.
    splits = [0, 4, 7]
    values = [1, 1, 2, 1, -2, 10, 5]
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Input values must all be non-negative"):
      self.evaluate(
          gen_count_ops.RaggedCountSparseOutput(
              splits=splits, values=values, binary_output=False))


if __name__ == "__main__":
  test.main()
