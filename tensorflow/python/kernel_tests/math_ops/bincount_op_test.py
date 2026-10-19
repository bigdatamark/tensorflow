# [MODULE DOC]
# Test module for the bincount function and the raw dense, sparse and ragged bincount kernels.
# Four test classes check counts, weights, binary output, length limits, shape inference, GPU
# code paths, determinism errors and invalid arguments against numpy references. No import-time
# side effects; running as a script starts the test runner.
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
"""Tests for bincount_ops.bincount."""
from absl.testing import parameterized
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bincount_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest


class BincountTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the public bincount function (counting occurrences of non-negative integers,
  # optionally weighted, with minimum and maximum output length). Covers empty input, basic
  # values, maximum length, random weighted and unweighted data against numpy, determinism
  # errors on GPU, zero weights, negative inputs, shape inference and invalid inputs.

  def test_empty(self):
    # [FUNCTION DOC]
    # Calls bincount on empty input with various minimum lengths (5, 1, 0) and asserts
    # zero-filled or empty outputs of those lengths; checks the requested output dtype (32- and
    # 64-bit float) is honored; and checks binary output on an empty typed constant gives zeros.
    with self.session():
      self.assertAllEqual(
          self.evaluate(bincount_ops.bincount([], minlength=5)),
          [0, 0, 0, 0, 0])
      self.assertAllEqual(
          self.evaluate(bincount_ops.bincount([], minlength=1)), [0])
      self.assertAllEqual(
          self.evaluate(bincount_ops.bincount([], minlength=0)), [])
      self.assertEqual(
          self.evaluate(
              bincount_ops.bincount([], minlength=0, dtype=np.float32)).dtype,
          np.float32)
      self.assertEqual(
          self.evaluate(
              bincount_ops.bincount([], minlength=3, dtype=np.float64)).dtype,
          np.float64)
      self.assertAllEqual(
          self.evaluate(
              bincount_ops.bincount(
                  constant_op.constant([], shape=[0], dtype=np.int32),
                  minlength=5,
                  binary_output=True,
              )
          ),
          [0, 0, 0, 0, 0],
      )

  def test_values(self):
    # [FUNCTION DOC]
    # Checks bincount counts for small lists, a list built up with repeats, lists containing
    # zero, empty input, a single element and a large range of ten thousand distinct values,
    # comparing against explicit expected count vectors.
    with self.session():
      self.assertAllEqual(
          self.evaluate(bincount_ops.bincount([1, 1, 1, 2, 2, 3])),
          [0, 3, 2, 1])
      arr = [1, 1, 2, 1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5]
      self.assertAllEqual(
          self.evaluate(bincount_ops.bincount(arr)), [0, 5, 4, 3, 2, 1])
      arr += [0, 0, 0, 0, 0, 0]
      self.assertAllEqual(
          self.evaluate(bincount_ops.bincount(arr)), [6, 5, 4, 3, 2, 1])

      self.assertAllEqual(self.evaluate(bincount_ops.bincount([])), [])
      self.assertAllEqual(self.evaluate(bincount_ops.bincount([0, 0, 0])), [3])
      self.assertAllEqual(
          self.evaluate(bincount_ops.bincount([5])), [0, 0, 0, 0, 0, 1])
      self.assertAllEqual(
          self.evaluate(bincount_ops.bincount(np.arange(10000))),
          np.ones(10000))

  def test_maxlength(self):
    # [FUNCTION DOC]
    # Verifies the maximum length option truncates the output: values at or beyond the limit are
    # dropped (giving zeros or fewer bins), and empty input still gives an empty result.
    with self.session():
      self.assertAllEqual(
          self.evaluate(bincount_ops.bincount([5], maxlength=3)), [0, 0, 0])
      self.assertAllEqual(
          self.evaluate(bincount_ops.bincount([1], maxlength=3)), [0, 1])
      self.assertAllEqual(
          self.evaluate(bincount_ops.bincount([], maxlength=3)), [])

  def test_random_with_weights(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the dtype loop variable only selects the kind of weights (integer or random
    # float); the weights are never converted to that dtype.
    # Seeds numpy (global state) and for each of four dtypes draws ten thousand random values
    # below one thousand with integer or floating weights, and asserts weighted bincount is
    # close to numpy's weighted bincount.
    num_samples = 10000
    with self.session():
      np.random.seed(42)
      for dtype in [dtypes.int32, dtypes.int64, dtypes.float32, dtypes.float64]:
        arr = np.random.randint(0, 1000, num_samples)
        if dtype == dtypes.int32 or dtype == dtypes.int64:
          weights = np.random.randint(-100, 100, num_samples)
        else:
          weights = np.random.random(num_samples)
        self.assertAllClose(
            self.evaluate(bincount_ops.bincount(arr, weights)),
            np.bincount(arr, weights))

  def test_random_without_weights(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the loop over two dtypes builds unit weights of each dtype only for numpy's
    # reference, and the weights are never passed to bincount, so the loop iterations are
    # effectively identical.
    # Seeds numpy (global state), draws ten thousand random values below one thousand, and
    # asserts bincount with no weights is close to numpy's bincount with unit weights.
    num_samples = 10000
    with self.session():
      np.random.seed(42)
      for dtype in [np.int32, np.float32]:
        arr = np.random.randint(0, 1000, num_samples)
        weights = np.ones(num_samples).astype(dtype)
        self.assertAllClose(
            self.evaluate(bincount_ops.bincount(arr, None)),
            np.bincount(arr, weights))

  @test_util.run_gpu_only
  @test_util.disable_xla("Bincount is deterministic with XLA")
  def test_bincount_determinism_error(self):
    # [FUNCTION DOC]
    # GPU-only test (disabled under XLA). With deterministic ops enabled, bincount on one
    # dimensional input and on a two-dimensional input (last axis) must raise an unimplemented
    # error saying determinism is not supported in the GPU implementation of the bincount kernel.
    arr = np.random.randint(0, 1000, size=1000)
    with test_util.deterministic_ops(), self.assertRaisesRegex(
        errors_impl.UnimplementedError,
        "Determinism is not yet supported in GPU implementation of "
        "(Dense)?Bincount.",
    ):
      self.evaluate(bincount_ops.bincount(arr, None, axis=None))
    arr = np.random.randint(0, 1000, size=(100, 100))
    with test_util.deterministic_ops(), self.assertRaisesRegex(
        errors_impl.UnimplementedError,
        "Determinism is not yet supported in GPU implementation of "
        "(Dense)?Bincount."):
      self.evaluate(bincount_ops.bincount(arr, None, axis=-1))

  def test_zero_weights(self):
    # [FUNCTION DOC]
    # Calls bincount on a range of one thousand values with all-zero weights and asserts the
    # result is a thousand zeros.
    with self.session():
      self.assertAllEqual(
          self.evaluate(bincount_ops.bincount(np.arange(1000), np.zeros(1000))),
          np.zeros(1000))

  @test_util.disable_xla("This is not raised on XLA CPU")
  def test_negative(self):
    # [FUNCTION DOC]
    # Pinned to the CPU (the segment-sum error is only reported there; disabled under XLA).
    # Asserts bincount on a list containing a negative value, and the raw dense bincount kernel
    # on a matrix containing a negative value, raise a value or invalid-argument error saying
    # values must be non-negative.
    # unsorted_segment_sum will only report InvalidArgumentError on CPU
    with self.cached_session(), ops.device("/CPU:0"):
      with self.assertRaisesRegex(
          (ValueError, errors.InvalidArgumentError), "must be non-negative"
      ):
        self.evaluate(bincount_ops.bincount([1, 2, 3, -1, 6, 8]))
      with self.assertRaisesRegex(
          (ValueError, errors.InvalidArgumentError), "must be non-negative"
      ):
        self.evaluate(
            gen_math_ops.dense_bincount(
                input=[[1, 1, 3], [0, -1, 2]], weights=[], size=4
            )
        )

  @test_util.run_in_graph_and_eager_modes
  def test_shape_function(self):
    # [FUNCTION DOC]
    # Tests shape inference of the raw bincount kernel in graph and eager modes: a non-scalar
    # size raises an error about rank, a negative size raises a non-negative error, a constant
    # size gives a known static output shape, and a placeholder size gives an unknown dimension.
    # size must be scalar.
    with self.assertRaisesRegex(
        (ValueError, errors.InvalidArgumentError),
        "(?s)Shape must be rank 0 but is rank 1.*Bincount"):
      gen_math_ops.bincount([1, 2, 3, 1, 6, 8], [1], [])
    # size must be positive.
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "must be non-negative"):
      gen_math_ops.bincount([1, 2, 3, 1, 6, 8], -5, [])
    # if size is a constant then the shape is known.
    v1 = gen_math_ops.bincount([1, 2, 3, 1, 6, 8], 5, [])
    self.assertAllEqual(v1.get_shape().as_list(), [5])
    # if size is a placeholder then the shape is unknown.
    with ops.Graph().as_default():
      s = array_ops.placeholder(dtype=dtypes.int32)
      v2 = gen_math_ops.bincount([1, 2, 3, 1, 6, 8], s, [])
      self.assertAllEqual(v2.get_shape().as_list(), [None])

  @test_util.run_in_graph_and_eager_modes
  def test_invalid_inputs(self):
    # [FUNCTION DOC]
    # Fuzz-style regression test (graph and eager): feeds the dense bincount kernel a random
    # integer matrix with negative values, a random scalar size and a random scalar weight with
    # binary output on, and asserts an invalid-argument error is raised.
    binary_output = True
    inp = random_ops.random_uniform(
        shape=[10, 10],
        minval=-10000,
        maxval=10000,
        dtype=dtypes.int32,
        seed=-2460)
    size = random_ops.random_uniform(
        shape=[], minval=-10000, maxval=10000, dtype=dtypes.int32, seed=-10000)
    weights = random_ops.random_uniform(
        shape=[],
        minval=-10000,
        maxval=10000,
        dtype=dtypes.float32,
        seed=-10000)
    with self.assertRaises(errors.InvalidArgumentError):
      self.evaluate(
          gen_math_ops.dense_bincount(
              input=inp,
              size=size,
              weights=weights,
              binary_output=binary_output))


class BincountOpTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for the raw dense bincount kernel (generated op) on the GPU, covering
  # one-dimensional and per-row two-dimensional inputs, with or without weights and binary
  # output, shared-memory versus global-memory code paths, and invalid rank or size arguments.
  # Private helpers build numpy references.

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_bincount_all_count(self, dtype):
    # [FUNCTION DOC]
    # For 32- and 64-bit integer inputs, draws 4096 random values below 1000 (numpy seeded to
    # 42, global state) and asserts the dense bincount kernel (run on GPU when available) equals
    # numpy's bincount with a minimum length of 1000.
    np.random.seed(42)
    size = 1000
    inp = np.random.randint(0, size, (4096), dtype=dtype)
    np_out = np.bincount(inp, minlength=size)
    with test_util.use_gpu():
      self.assertAllEqual(
          np_out,
          self.evaluate(
              gen_math_ops.dense_bincount(input=inp, weights=[], size=size)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_bincount_all_count_with_weights(self, dtype):
    # [FUNCTION DOC]
    # Like the unweighted test but with 4096 random floating weights; asserts the kernel output
    # (GPU when available) exactly equals numpy's weighted bincount (exact comparison, not
    # approximate). Numpy seeded to 42.
    np.random.seed(42)
    size = 1000
    inp = np.random.randint(0, size, (4096,), dtype=dtype)
    np_weight = np.random.random((4096,))
    np_out = np.bincount(inp, minlength=size, weights=np_weight)
    with test_util.use_gpu():
      self.assertAllEqual(
          np_out,
          self.evaluate(
              gen_math_ops.dense_bincount(
                  input=inp, weights=np_weight, size=size)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_bincount_all_binary(self, dtype):
    # [FUNCTION DOC]
    # With binary output on, 4096 random values below ten and no weights, asserts the kernel
    # returns a vector of ten ones (every bin is present). Numpy seeded to 42.
    np.random.seed(42)
    size = 10
    inp = np.random.randint(0, size, (4096), dtype=dtype)
    np_out = np.ones((size,))
    with test_util.use_gpu():
      self.assertAllEqual(
          np_out,
          self.evaluate(
              gen_math_ops.dense_bincount(
                  input=inp, weights=[], size=size, binary_output=True)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_bincount_all_binary_with_weights(self, dtype):
    # [FUNCTION DOC]
    # Binary output with random weights supplied: asserts the output is still a vector of ten
    # ones, i.e. weights are ignored in binary mode. Numpy seeded to 42.
    np.random.seed(42)
    size = 10
    inp = np.random.randint(0, size, (4096,), dtype=dtype)
    np_weight = np.random.random((4096,))
    np_out = np.ones((size,))
    with test_util.use_gpu():
      self.assertAllEqual(
          np_out,
          self.evaluate(
              gen_math_ops.dense_bincount(
                  input=inp, weights=np_weight, size=size, binary_output=True)))

  def _test_bincount_col_count(self, num_rows, num_cols, size, dtype):
    # [FUNCTION DOC]
    # HELPER: for the per-row (column reduction) tests. Seeds numpy to 42, draws a random integer
    # matrix with the given rows, columns, value range and dtype, builds the numpy reference by
    # counting each row, and asserts the dense bincount kernel on the matrix (GPU when
    # available) matches it exactly. Returns nothing.
    np.random.seed(42)
    inp = np.random.randint(0, size, (num_rows, num_cols), dtype=dtype)
    np_out = np.reshape(
        np.concatenate(
            [np.bincount(inp[j, :], minlength=size) for j in range(num_rows)],
            axis=0), (num_rows, size))
    with test_util.use_gpu():
      self.assertAllEqual(
          np_out,
          self.evaluate(
              gen_math_ops.dense_bincount(input=inp, weights=[], size=size)))

  def _test_bincount_col_binary(self, num_rows, num_cols, size, dtype):
    # [FUNCTION DOC]
    # HELPER: like the per-row count helper but with binary output; the reference marks each bin
    # as one if present in the row and zero otherwise. Seeds numpy to 42; asserts exact equality
    # with the kernel's output.
    np.random.seed(42)
    inp = np.random.randint(0, size, (num_rows, num_cols), dtype=dtype)
    np_out = np.reshape(
        np.concatenate([
            np.where(np.bincount(inp[j, :], minlength=size) > 0, 1, 0)
            for j in range(num_rows)
        ],
                       axis=0), (num_rows, size))
    with test_util.use_gpu():
      self.assertAllEqual(
          np_out,
          self.evaluate(
              gen_math_ops.dense_bincount(
                  input=inp, weights=[], size=size, binary_output=True)))

  def _test_bincount_col_count_with_weights(self, num_rows, num_cols, size,
                                            dtype):
    # [FUNCTION DOC]
    # HELPER: per-row weighted count check. Seeds numpy to 42, draws a random integer matrix and
    # same-shaped random weights, builds the weighted per-row numpy reference, and asserts the
    # kernel output matches (approximately when the reference is floating point, otherwise
    # exactly).
    np.random.seed(42)
    inp = np.random.randint(0, size, (num_rows, num_cols), dtype=dtype)
    np_weight = np.random.random((num_rows, num_cols))
    np_out = np.reshape(
        np.concatenate([
            np.bincount(inp[j, :], weights=np_weight[j, :], minlength=size)
            for j in range(num_rows)
        ],
                       axis=0), (num_rows, size))
    with test_util.use_gpu():
      evaluated = self.evaluate(
          gen_math_ops.dense_bincount(input=inp, weights=np_weight, size=size))
      if np_out.dtype in (np.float32, np.float64):
        self.assertAllClose(np_out, evaluated)
      else:
        self.assertAllEqual(np_out, evaluated)

  def test_col_reduce_basic(self):
    # [FUNCTION DOC]
    # Runs the dense kernel on a two-by-three matrix with size four and no weights and asserts
    # the per-row counts are the expected two-by-four array.
    with test_util.use_gpu():
      v = self.evaluate(
          gen_math_ops.dense_bincount(
              input=[[1, 2, 3], [0, 3, 2]], weights=[], size=4))
    expected_out = [[0., 1., 1., 1.], [1., 0., 1., 1.]]
    self.assertAllEqual(expected_out, v)

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_col_reduce_shared_memory(self, dtype):
    # [FUNCTION DOC]
    # Per-row count check with 128 rows, 27 columns and 10 bins (small enough for the GPU
    # shared-memory path) for 32- and 64-bit integer inputs.
    # num_rows * num_bins less than half of max shared memory.
    num_rows = 128
    num_cols = 27
    size = 10
    self._test_bincount_col_count(num_rows, num_cols, size, dtype)

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_col_reduce_global_memory(self, dtype):
    # [FUNCTION DOC]
    # Per-row count check with 128 rows, 27 columns and 1024 bins (large enough to need the GPU
    # global-memory path) for 32- and 64-bit integer inputs.
    # num_rows * num_bins more than half of max shared memory.
    num_rows = 128
    num_cols = 27
    size = 1024
    self._test_bincount_col_count(num_rows, num_cols, size, dtype)

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_col_reduce_shared_memory_with_weights(self, dtype):
    # [FUNCTION DOC]
    # Weighted per-row count check with 128 rows, 27 columns and 100 bins (shared-memory path)
    # for 32- and 64-bit integer inputs.
    # num_rows * num_bins less than half of max shared memory.
    num_rows = 128
    num_cols = 27
    size = 100
    self._test_bincount_col_count_with_weights(num_rows, num_cols, size, dtype)

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_col_reduce_global_memory_with_weights(self, dtype):
    # [FUNCTION DOC]
    # Weighted per-row count check with 128 rows, 27 columns and 1024 bins (global-memory path)
    # for 32- and 64-bit integer inputs.
    # num_rows * num_bins more than half of max shared memory.
    num_rows = 128
    num_cols = 27
    size = 1024
    self._test_bincount_col_count_with_weights(num_rows, num_cols, size, dtype)

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_col_reduce_binary(self, dtype):
    # [FUNCTION DOC]
    # Binary-output per-row check with 128 rows, 7 columns and 10 bins for 32- and 64-bit
    # integer inputs.
    num_rows = 128
    num_cols = 7
    size = 10
    self._test_bincount_col_binary(num_rows, num_cols, size, dtype)

  def test_invalid_rank(self):
    # [FUNCTION DOC]
    # A rank-three input to the dense kernel must raise a value or invalid-argument error saying
    # at most rank two is supported.
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "at most rank 2"):
      with test_util.use_gpu():
        self.evaluate(
            gen_math_ops.dense_bincount(
                input=[[[1, 2, 3], [0, 3, 2]]], weights=[], size=10))

  @test_util.run_in_graph_and_eager_modes
  def test_size_is_not_scalar(self):  # b/206619828
    # [FUNCTION DOC]
    # Regression test for an internal bug: a size argument of rank one passed to the dense
    # kernel must raise an error saying the shape must be rank zero (graph and eager modes).
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "Shape must be rank 0 but is rank 1"):
      self.evaluate(
          gen_math_ops.dense_bincount(
              input=[0], size=[1, 1], weights=[3], binary_output=False))


class SparseBincountOpTest(test_util.TensorFlowTestCase,
                           parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for the raw sparse bincount kernel, which takes indices, values,
  # dense shape, size and optional weights; tests compare outputs against numpy for 32- and
  # 64-bit integer values, with and without weights, binary output and several shapes, plus
  # invalid-argument cases.

  @parameterized.parameters([
      {
          "dtype": np.int32,
      },
      {
          "dtype": np.int64,
      },
  ])
  def test_sparse_bincount_all_count(self, dtype):
    # [FUNCTION DOC]
    # Builds 128 random sparse entries in a one-dimensional dense shape of 4096 (numpy seeded to
    # 42) and asserts the sparse kernel without weights equals numpy's bincount with minimum
    # length 1000.
    np.random.seed(42)
    num_rows = 4096
    size = 1000
    n_elems = 128
    inp_indices = np.random.randint(0, num_rows, (n_elems, 1))
    inp_vals = np.random.randint(0, size, (n_elems,), dtype=dtype)

    np_out = np.bincount(inp_vals, minlength=size)
    self.assertAllEqual(
        np_out,
        self.evaluate(
            gen_math_ops.sparse_bincount(
                indices=inp_indices,
                values=inp_vals,
                dense_shape=[num_rows],
                size=size,
                weights=[])))

  @parameterized.parameters([
      {
          "dtype": np.int32,
      },
      {
          "dtype": np.int64,
      },
  ])
  def test_sparse_bincount_all_count_with_weights(self, dtype):
    # [FUNCTION DOC]
    # Same as the unweighted sparse test but with random floating weights per entry; asserts the
    # kernel equals numpy's weighted bincount exactly. Numpy seeded to 42.
    np.random.seed(42)
    num_rows = 4096
    size = 1000
    n_elems = 128
    inp_indices = np.random.randint(0, num_rows, (n_elems, 1))
    inp_vals = np.random.randint(0, size, (n_elems,), dtype=dtype)
    inp_weight = np.random.random((n_elems,))

    np_out = np.bincount(inp_vals, minlength=size, weights=inp_weight)
    self.assertAllEqual(
        np_out,
        self.evaluate(
            gen_math_ops.sparse_bincount(
                indices=inp_indices,
                values=inp_vals,
                dense_shape=[num_rows],
                size=size,
                weights=inp_weight)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_sparse_bincount_all_binary(self, dtype):
    # [FUNCTION DOC]
    # Binary output with 4096 random sparse entries across 128 rows and values below ten, no
    # weights: asserts the sparse kernel returns a vector of ten ones. Numpy seeded to 42.
    np.random.seed(42)
    num_rows = 128
    size = 10
    n_elems = 4096
    inp_indices = np.random.randint(0, num_rows, (n_elems, 1))
    inp_vals = np.random.randint(0, size, (n_elems,), dtype=dtype)

    np_out = np.ones((size,))
    self.assertAllEqual(
        np_out,
        self.evaluate(
            gen_math_ops.sparse_bincount(
                indices=inp_indices,
                values=inp_vals,
                dense_shape=[num_rows],
                size=size,
                weights=[],
                binary_output=True)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_sparse_bincount_all_binary_weights(self, dtype):
    # [FUNCTION DOC]
    # Same as the binary sparse test but with random weights supplied; asserts the output is
    # still a vector of ten ones because binary mode ignores weights. Numpy seeded to 42.
    np.random.seed(42)
    num_rows = 128
    size = 10
    n_elems = 4096
    inp_indices = np.random.randint(0, num_rows, (n_elems, 1))
    inp_vals = np.random.randint(0, size, (n_elems,), dtype=dtype)
    inp_weight = np.random.random((n_elems,))

    np_out = np.ones((size,))
    self.assertAllEqual(
        np_out,
        self.evaluate(
            gen_math_ops.sparse_bincount(
                indices=inp_indices,
                values=inp_vals,
                dense_shape=[num_rows],
                size=size,
                weights=inp_weight,
                binary_output=True)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_sparse_bincount_col_reduce_count(self, dtype):
    # [FUNCTION DOC]
    # Builds a random 128 by 27 integer matrix, computes per-row numpy counts as the reference,
    # converts the matrix (shifted up by one so zeros are not dropped) to sparse form on the CPU
    # to avoid GPU memory exhaustion, then runs the sparse kernel with values shifted back and
    # asserts exact equality with the reference. Numpy seeded to 42.
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
    self.assertAllEqual(
        np_out,
        self.evaluate(
            gen_math_ops.sparse_bincount(
                indices=inp_sparse.indices,
                values=inp_sparse.values - 1,
                dense_shape=inp_sparse.dense_shape,
                size=size,
                weights=[])))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_sparse_bincount_col_reduce_binary(self, dtype):
    # [FUNCTION DOC]
    # Binary-output version of the per-row sparse test: reference marks each bin present (one)
    # or absent (zero) per row; the matrix is shifted by one, made sparse on the CPU, and the
    # sparse kernel is run with binary output on shifted-back values. Asserts exact equality.
    # Numpy seeded to 42.
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
    self.assertAllEqual(
        np_out,
        self.evaluate(
            gen_math_ops.sparse_bincount(
                indices=inp_sparse.indices,
                values=inp_sparse.values - 1,
                dense_shape=inp_sparse.dense_shape,
                size=size,
                weights=[],
                binary_output=True)))

  @parameterized.parameters([
      {
          "values": [0, 1, 2, 2],
          "axis": 0,
          "binary": False,
          "expect": [1, 1, 2],
      },
      {
          "values": [2, 1, 2, 2],
          "axis": 0,
          "binary": False,
          "expect": [0, 1, 3],
      },
      {
          "values": [0, 1, 2, 2],
          "axis": 0,
          "binary": True,
          "expect": [1, 1, 1],
      },
      {
          "values": [2, 1, 2, 2],
          "axis": 0,
          "binary": True,
          "expect": [0, 1, 1],
      },
      {
          "values": [0, 1, 2, 2],
          "axis": -1,
          "binary": False,
          "expect": [[0, 0, 0], [1, 1, 0], [0, 0, 2], [0, 0, 0]],
      },
      {
          "values": [2, 1, 2, 2],
          "axis": -1,
          "binary": False,
          "expect": [[0, 0, 0], [0, 1, 1], [0, 0, 2], [0, 0, 0]],
      },
      {
          "values": [0, 1, 2, 2],
          "axis": -1,
          "binary": True,
          "expect": [[0, 0, 0], [1, 1, 0], [0, 0, 1], [0, 0, 0]],
      },
      {
          "values": [2, 1, 2, 2],
          "axis": -1,
          "binary": True,
          "expect": [[0, 0, 0], [0, 1, 1], [0, 0, 1], [0, 0, 0]],
      },
  ])
  def test_sparse_bincount_implicit_zeros(
      self, values, axis, binary, expect
  ):
    # [FUNCTION DOC]
    # Parameterized over a list of four values, an axis (0 for one-dimensional sparse input,
    # negative one for two-dimensional), a binary flag and the expected counts. Builds indices
    # and dense shape (rows four and columns five for axis negative one; length twenty
    # otherwise) with most entries implicit, runs the sparse kernel with size three and no
    # weights, and asserts the output equals the expectation (implicit entries are not counted
    # as zeros).
    if axis == -1:
      indices = [[1, 2], [1, 4], [2, 2], [2, 4]]
      dense_shape = [4, 5]
    else:
      indices = [[7], [9], [12], [14]]
      dense_shape = [20]

    self.assertAllEqual(
        expect,
        self.evaluate(
            gen_math_ops.sparse_bincount(
                indices=indices,
                values=values,
                dense_shape=dense_shape,
                size=3,
                weights=[],
                binary_output=binary)))

  @test_util.run_in_graph_and_eager_modes
  def test_size_is_not_scalar(self):  # b/206619828
    # [FUNCTION DOC]
    # Regression test for an internal bug: a size argument of rank one passed to the sparse
    # kernel must raise an error saying the shape must be rank zero (graph and eager modes).
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "Shape must be rank 0 but is rank 1"):
      self.evaluate(
          gen_math_ops.sparse_bincount(
              indices=[[0], [1]],
              values=[0, 0],
              dense_shape=[1, 1],
              size=[1, 1],
              weights=[0, 0],
              binary_output=False))

  def test_sparse_bincount_input_validation(self):
    # [FUNCTION DOC]
    # Builds random sparse input (numpy seeded to 42), forces one index to a negative number,
    # and asserts the sparse kernel raises a value or invalid-argument error mentioning an out
    # of bounds index.
    np.random.seed(42)
    num_rows = 128
    size = 1000
    n_elems = 4096
    inp_indices = np.random.randint(0, num_rows, (n_elems, 1))
    inp_vals = np.random.randint(0, size, (n_elems,))

    # Insert negative index.
    inp_indices[10, 0] = -2

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "out of bounds"):
      self.evaluate(
          gen_math_ops.sparse_bincount(
              indices=inp_indices,
              values=inp_vals,
              dense_shape=[num_rows],
              size=size,
              weights=[]))


class RaggedBincountOpTest(test_util.TensorFlowTestCase,
                           parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized test case for the raw ragged bincount kernel, which takes row splits, values,
  # optional weights and size and returns one count row per ragged row. Compares against explicit
  # expectations and numpy for counts, binary output and weights, plus invalid-argument cases.

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_ragged_bincount_count(self, dtype):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the dtype parameter is accepted but never used, so both parameterized runs
    # are identical.
    # Uses a five-row ragged constant (some rows empty) with size six and no weights and asserts
    # the kernel gives the expected per-row counts.
    x = ragged_factory_ops.constant([[], [], [3, 0, 1], [], [5, 0, 4, 4]])
    expected_output = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0,
                                            0], [1, 1, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 2, 1]]
    self.assertAllEqual(
        expected_output,
        self.evaluate(
            gen_math_ops.ragged_bincount(
                splits=x.row_splits, values=x.values, weights=[], size=6)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_ragged_bincount_binary(self, dtype):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the dtype parameter is accepted but never used, so both parameterized runs
    # are identical.
    # Same ragged input as the count test with binary output on; asserts each bin present in a
    # row is one (the repeated value in the last row gives one, not two).
    x = ragged_factory_ops.constant([[], [], [3, 0, 1], [], [5, 0, 4, 4]])
    expected_output = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0,
                                            0], [1, 1, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 1, 1]]
    self.assertAllEqual(
        expected_output,
        self.evaluate(
            gen_math_ops.ragged_bincount(
                splits=x.row_splits,
                values=x.values,
                weights=[],
                size=6,
                binary_output=True)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_ragged_bincount_count_with_weights(self, dtype):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the dtype parameter is accepted but never used, so both parameterized runs
    # are identical.
    # Same ragged input with a matching ragged tensor of floating weights; asserts the kernel
    # sums weights per bin per row, approximately equal to the expected matrix.
    x = ragged_factory_ops.constant([[], [], [3, 0, 1], [], [5, 0, 4, 4]])
    weights = ragged_factory_ops.constant([[], [], [.1, .2, .3], [],
                                           [.2, .5, .6, .3]])
    expected_output = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0],
                       [.2, .3, 0, .1, 0, 0], [0, 0, 0, 0, 0, 0],
                       [.5, 0, 0, 0, .9, .2]]
    self.assertAllClose(
        expected_output,
        self.evaluate(
            gen_math_ops.ragged_bincount(
                splits=x.row_splits,
                values=x.values,
                weights=weights.values,
                size=6)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_ragged_bincount_count_np(self, dtype):
    # [FUNCTION DOC]
    # Builds a random 128 by 27 integer matrix (numpy seeded to 42), converts it to a ragged
    # tensor, runs the ragged kernel with size 1000 and no weights, and asserts exact equality
    # with per-row numpy bincounts.
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
        self.evaluate(
            gen_math_ops.ragged_bincount(
                splits=x.row_splits, values=x.values, weights=[], size=size)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_ragged_bincount_count_np_with_weights(self, dtype):
    # [FUNCTION DOC]
    # Same as the numpy count test but with random floating weights of the same matrix shape;
    # asserts exact (not approximate) equality with weighted per-row numpy bincounts. Numpy
    # seeded to 42.
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
    self.assertAllEqual(
        np_out,
        self.evaluate(
            gen_math_ops.ragged_bincount(
                splits=x.row_splits,
                values=x.values,
                weights=np_weight,
                size=size)))

  @parameterized.parameters([{
      "dtype": np.int32,
  }, {
      "dtype": np.int64,
  }])
  def test_ragged_bincount_binary_np_with_weights(self, dtype):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite the name, no weights are generated or passed to the kernel; it tests
    # binary output without weights.
    # Random 128 by 27 integer matrix converted to ragged form; with binary output on asserts
    # equality with the per-row present/absent numpy reference. Numpy seeded to 42.
    np.random.seed(42)
    num_rows = 128
    num_cols = 27
    size = 1000
    inp = np.random.randint(0, size, (num_rows, num_cols), dtype=dtype)
    np_out = np.reshape(
        np.concatenate([
            np.where(np.bincount(inp[j, :], minlength=size) > 0, 1, 0)
            for j in range(num_rows)
        ],
                       axis=0), (num_rows, size))
    x = ragged_tensor.RaggedTensor.from_tensor(inp)
    self.assertAllEqual(
        np_out,
        self.evaluate(
            gen_math_ops.ragged_bincount(
                splits=x.row_splits,
                values=x.values,
                weights=[],
                size=size,
                binary_output=True)))

  @test_util.run_in_graph_and_eager_modes
  def test_size_is_not_scalar(self):  # b/206619828
    # [FUNCTION DOC]
    # Regression test for an internal bug: a size argument of rank one passed to the ragged
    # kernel must raise an error saying the shape must be rank zero (graph and eager modes).
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "Shape must be rank 0 but is rank 1"):
      self.evaluate(
          gen_math_ops.ragged_bincount(
              splits=[0, 0, 1],
              values=[1],
              size=[1, 1],
              weights=[0, 0, 0],
              binary_output=False,
              name=None))

  @test_util.run_in_graph_and_eager_modes
  def test_splits_empty(self):  # b/238450914
    # [FUNCTION DOC]
    # Regression test for an internal bug: an empty row-splits input to the ragged kernel must
    # raise a value or invalid-argument error saying splits must be non-empty (graph and eager).
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                "Splits must be non-empty"):
      self.evaluate(
          gen_math_ops.ragged_bincount(
              splits=[],  # Invalid splits
              values=[1],
              size=1,
              weights=[1],
              binary_output=False,
              name=None))

if __name__ == "__main__":
  googletest.main()
