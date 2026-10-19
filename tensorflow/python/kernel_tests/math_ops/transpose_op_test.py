# [MODULE DOC]
# Functional tests for the array transpose operation. A single test case compares the TensorFlow
# transpose (values, shapes and gradients, on CPU and on GPU when available) with NumPy for many
# dtypes, ranks, permutations, conjugation, large GPU-specific shapes, singleton dimensions,
# dynamic shapes in compiled functions, empty and scalar tensors, negative permutation entries
# and error cases. Not a helper module; no import-time side effects.
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
"""Functional tests for Transpose op."""

import itertools

import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.platform import test


class TransposeTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the transpose operation. Provides comparison helpers against NumPy (CPU, GPU,
  # both) and many scenario tests built on them. No per-test fixtures.

  def _np_transpose(self, x, perm):
    # [FUNCTION DOC]
    # HELPER: reference transpose using NumPy. Copies the input array and returns it with axes
    # reordered by the permutation (list or array of axis indices). No side effects.
    ret = np.copy(x)
    ret = ret.transpose(perm)
    return ret

  def _compareCpu(self, x, p, conjugate=False):
    # [FUNCTION DOC]
    # HELPER: checks transpose on CPU against NumPy. Inputs: array x, permutation p (or nothing,
    # meaning reverse the axes), and a conjugate flag (default false) that also conjugates the
    # reference. Asserts result shape and values equal the reference, and for single/double
    # precision real and complex types also runs a numerical-versus-theoretical gradient check
    # (looser tolerance for single precision). Returns the TensorFlow result and the theoretical
    # Jacobian (nothing for other dtypes).
    if p is None:
      rank = x.ndim
      perm = (rank - 1) - np.arange(rank)
    else:
      perm = p
    np_ans = self._np_transpose(x, perm)
    if conjugate:
      np_ans = np.conj(np_ans)
    with self.cached_session(use_gpu=False):
      inx = ops.convert_to_tensor(x)
      y = array_ops.transpose(inx, p, conjugate=conjugate)
      tf_ans = self.evaluate(y)
      self.assertShapeEqual(np_ans, y)
      self.assertAllEqual(np_ans, tf_ans)

      jacob_t = None
      # Gradient check on CPU.
      if x.dtype in [np.float32, np.complex64]:
        jacob_t, jacob_n = gradient_checker_v2.compute_gradient(
            lambda x: array_ops.transpose(x, p, conjugate=conjugate), [inx])
        self.assertAllClose(jacob_t, jacob_n, 1e-3, 1e-3)
      elif x.dtype in [np.float64, np.complex128]:
        jacob_t, jacob_n = gradient_checker_v2.compute_gradient(
            lambda x: array_ops.transpose(x, p, conjugate=conjugate), [inx])
        self.assertAllClose(jacob_t, jacob_n, 1e-6, 1e-6)

      return tf_ans, jacob_t

  def _compareGpu(self, x, p, conjugate=False):
    # [FUNCTION DOC]
    # HELPER: like the CPU comparison but runs in the default cached session (which allows GPU
    # placement). Inputs: array x, permutation p (or nothing for reversed axes), conjugate flag.
    # Asserts values and shape against NumPy; runs a gradient check only for single and double
    # real types. Returns the TensorFlow result and theoretical Jacobian (nothing otherwise).
    # KNOWN ISSUE: the session is created without explicitly requesting a GPU, so on machines
    # without a GPU this silently exercises the CPU kernel instead of failing or skipping.
    if p is None:
      rank = x.ndim
      perm = (rank - 1) - np.arange(rank)
    else:
      perm = p
    np_ans = self._np_transpose(x, perm)
    if conjugate:
      np_ans = np.conj(np_ans)
    with self.cached_session():
      inx = ops.convert_to_tensor(x)
      y = array_ops.transpose(inx, p, conjugate=conjugate)
      tf_ans = self.evaluate(y)

      self.assertAllEqual(np_ans, tf_ans)
      self.assertShapeEqual(np_ans, y)

      jacob_t = None
      # Gradient check on GPU.
      if x.dtype == np.float32:
        jacob_t, jacob_n = gradient_checker_v2.compute_gradient(
            lambda x: array_ops.transpose(x, p, conjugate=conjugate), [inx])
        self.assertAllClose(jacob_t, jacob_n, 1e-3, 1e-3)
      elif x.dtype == np.float64:
        jacob_t, jacob_n = gradient_checker_v2.compute_gradient(
            lambda x: array_ops.transpose(x, p, conjugate=conjugate), [inx])
        self.assertAllClose(jacob_t, jacob_n, 1e-6, 1e-6)

      return tf_ans, jacob_t

  def _compare(self, x, use_gpu=False):
    # [FUNCTION DOC]
    # HELPER: compares transpose against NumPy for two randomly chosen permutations of the input
    # axes, plus the default reversed-axis case. For complex dtypes both plain and conjugate
    # variants are tested. Inputs: array x and a flag (default false) to also run the GPU
    # comparison. Returns nothing; uses global random state.
    n = np.ndim(x)
    # generate all permutations of [0, 1, ... n-1] in random order.
    all_perm = np.random.permutation(
        [p for p in itertools.permutations(range(n))]).astype(np.int32)
    cs = [False, True] if x.dtype in [np.complex64, np.complex128] else [False]
    for c in cs:
      for p in all_perm[:2]:
        self._compareCpu(x, p, conjugate=c)
        if use_gpu:
          self._compareGpu(x, p, conjugate=c)
    # Test with an empty permutation
    for c in cs:
      self._compareCpu(x, None, conjugate=c)
      if use_gpu:
        self._compareGpu(x, None, conjugate=c)

  def _compare_cpu_gpu(self, x):
    # [FUNCTION DOC]
    # HELPER: for two permutations, runs both the CPU and GPU comparisons and asserts that their
    # results and gradients agree within a precision-dependent tolerance (single and double
    # precision only), also requiring that both gradients exist. Input: array x. Rank-one or
    # lower input uses the default permutation. Returns nothing.
    # KNOWN ISSUE: the comment says permutations are chosen in random order, but the code takes
    # the first two permutations in lexicographic order and only shuffles within each; for rank
    # one, the gradient assertions fail only if no gradient is produced.
    n = np.ndim(x)
    # generate all permutation of [0, 1, ... n-1] in random order,
    # choose the first two.
    perms = itertools.permutations(range(n))
    for _ in range(2):
      p = np.random.permutation(next(perms)).astype(np.int32) if n > 1 else None
      tf_a_cpu, tf_g_cpu = self._compareCpu(x, p)
      tf_a_gpu, tf_g_gpu = self._compareGpu(x, p)
      assert tf_g_cpu is not None
      assert tf_g_gpu is not None
      if x.dtype == np.float32:
        self.assertAllClose(tf_a_cpu, tf_a_gpu, 1e-3, 1e-3)
        self.assertAllClose(tf_g_cpu, tf_g_gpu, 1e-3, 1e-3)
      elif x.dtype == np.float64:
        self.assertAllClose(tf_a_cpu, tf_a_gpu, 1e-6, 1e-6)
        self.assertAllClose(tf_g_cpu, tf_g_gpu, 1e-6, 1e-6)

  def _testBoth(self, x):
    # [FUNCTION DOC]
    # HELPER: runs the permutation comparison for input x once without and once with GPU
    # comparison. Returns nothing.
    self._compare(x, use_gpu=False)
    self._compare(x, use_gpu=True)

  def testRank1(self):
    # [FUNCTION DOC]
    # Test: transposing a two-element vector with the identity permutation on CPU matches NumPy.
    self._compareCpu(np.arange(0., 2), [0])

  def test1D(self):
    # [FUNCTION DOC]
    # Test: a five-dimensional array with all dimensions of size one except one of size two is
    # compared without and with GPU comparison. Returns nothing.
    vector = np.arange(0, 2).reshape((1, 1, 1, 2, 1))
    self._compare(vector, use_gpu=False)
    self._compare(vector, use_gpu=True)

  def test5DGPU(self):
    # [FUNCTION DOC]
    # Test: for large five-dimensional shapes and several permutations, and for int8, half,
    # single, double, complex128 and bfloat16 data, transposes sequential-valued arrays and
    # checks values and shape against NumPy (each combination as a sub-test). Returns early
    # (passing silently, not marked skipped) when no CUDA GPU is available.
    # If no GPU available, skip the test
    if not test.is_gpu_available(cuda_only=True):
      return
    large_shapes = [[4, 10, 10, 10, 3], [4, 10, 10, 10, 8], [4, 10, 10, 10, 13],
                    [4, 3, 10, 10, 10], [4, 8, 10, 10, 10], [4, 13, 10, 10,
                                                             10]] * 3
    perms = [[0, 4, 1, 2, 3]] * 3 + [[0, 2, 3, 4, 1]] * 3 + [[
        4, 1, 2, 3, 0
    ]] * 6 + [[1, 2, 3, 4, 0]] * 6

    datatypes = [
        np.int8, np.float16, np.float32, np.float64, np.complex128,
        dtypes.bfloat16.as_numpy_dtype
    ]
    for datatype in datatypes:
      for input_shape, perm in zip(large_shapes, perms):
        with self.subTest(
            datatype=datatype, input_shape=input_shape, perm=perm):
          total_size = np.prod(input_shape)
          inp = np.arange(
              1, total_size + 1, dtype=datatype).reshape(input_shape)
          np_ans = self._np_transpose(inp, perm)
          with self.cached_session():
            inx = ops.convert_to_tensor(inp)
            y = array_ops.transpose(inx, perm)
            tf_ans = self.evaluate(y)
          self.assertAllEqual(np_ans, tf_ans)
          self.assertShapeEqual(np_ans, y)

  def test4DGPU(self):
    # [FUNCTION DOC]
    # Test: four-dimensional single-precision arrays with several large shapes and permutations,
    # then a long list of shapes taken from Inception-style convolutions with a channels-last to
    # channels-first permutation; checks values and shape against NumPy per sub-test. Returns
    # early (silently passing) when no CUDA GPU is available.
    # If no GPU available, skip the test
    if not test.is_gpu_available(cuda_only=True):
      return
    large_shapes = [[4, 10, 10, 3], [4, 10, 10, 8], [4, 10, 10, 13],
                    [4, 3, 10, 10], [4, 8, 10, 10], [4, 13, 10, 10]] * 3
    perms = [[0, 3, 1, 2]] * 3 + [[0, 2, 3, 1]] * 3 + [[3, 1, 2, 0]] * 6 + [[
        1, 2, 3, 0
    ]] * 3 + [[2, 3, 0, 1]] * 3

    for input_shape, perm in zip(large_shapes, perms):
      with self.subTest(input_shape=input_shape, perm=perm):
        total_size = np.prod(input_shape)
        inp = np.arange(
            1, total_size + 1, dtype=np.float32).reshape(input_shape)
        np_ans = self._np_transpose(inp, perm)
        with self.cached_session():
          inx = ops.convert_to_tensor(inp)
          y = array_ops.transpose(inx, perm)
          tf_ans = self.evaluate(y)
        self.assertAllEqual(np_ans, tf_ans)
        self.assertShapeEqual(np_ans, y)

    # shapes related to Inception (taken from conv_ops_test.py)
    inception_shapes = [[4, 5, 5, 124], [4, 8, 8, 38], [4, 8, 8, 38], [
        4, 8, 8, 204
    ], [4, 8, 8, 44], [4, 8, 8, 204], [4, 8, 8, 204], [4, 8, 8, 204], [
        4, 8, 8, 176
    ], [4, 8, 8, 176], [4, 8, 8, 176], [4, 8, 8, 176], [4, 17, 17, 19], [
        4, 17, 17, 19
    ], [4, 17, 17, 124], [4, 17, 17, 12], [4, 17, 17, 124], [4, 17, 17, 22], [
        4, 17, 17, 19
    ], [4, 17, 17, 19], [4, 17, 17, 121], [4, 17, 17, 121], [4, 17, 17, 22], [
        4, 17, 17, 19
    ], [4, 17, 17, 19], [4, 17, 17, 115], [4, 17, 17, 115], [4, 17, 17, 19], [
        4, 17, 17, 16
    ], [4, 17, 17, 115], [4, 17, 17, 102], [4, 17, 17, 12], [4, 17, 17, 102], [
        4, 17, 17, 12
    ], [4, 17, 17, 102], [4, 17, 17, 12], [4, 17, 17, 76], [4, 17, 17, 12], [
        4, 17, 17, 12
    ], [4, 17, 17, 76], [4, 17, 17, 76], [4, 35, 35, 9], [4, 35, 35, 28], [
        4, 35, 35, 6
    ], [4, 35, 35, 28], [4, 35, 35, 25], [4, 35, 35, 4], [4, 35, 35, 25],
                        [4, 35, 35, 9], [4, 35, 35, 19], [4, 35, 35, 19],
                        [4, 35, 35, 19], [4, 73, 73, 6], [4, 73, 73,
                                                          6], [4, 147, 147, 2]]
    for input_shape in inception_shapes:
      with self.subTest(input_shape=input_shape):
        perm = [0, 3, 1, 2]
        total_size = np.prod(input_shape)
        inp = np.arange(
            1, total_size + 1, dtype=np.float32).reshape(input_shape)
        np_ans = self._np_transpose(inp, perm)
        with self.cached_session():
          inx = ops.convert_to_tensor(inp)
          y = array_ops.transpose(inx, perm)
          tf_ans = self.evaluate(y)
        self.assertAllEqual(np_ans, tf_ans)
        self.assertShapeEqual(np_ans, y)

  def test3DGPU(self):
    # [FUNCTION DOC]
    # Test: three-dimensional arrays of int8, half, single, double and complex128 types with
    # shapes having one dimension of size 1000 and one small dimension, under several
    # permutations; checks values and shape against NumPy per sub-test. Returns early (silently
    # passing) when no CUDA GPU is available.
    # If no GPU available, skip the test
    if not test.is_gpu_available(cuda_only=True):
      return

    datatypes = [np.int8, np.float16, np.float32, np.float64, np.complex128]
    large_shapes = [[4, 1000, 3], [4, 1000, 8], [4, 1000, 13], [4, 3, 1000],
                    [4, 8, 1000], [4, 13, 1000]] * 3
    perms = [[0, 2, 1]] * 6 + [[2, 1, 0]] * 6 + [[1, 2, 0]] * 3 + [[2, 0, 1]
                                                                  ] * 3
    for datatype in datatypes:
      for input_shape, perm in zip(large_shapes, perms):
        with self.subTest(
            datatype=datatype, input_shape=input_shape, perm=perm):
          total_size = np.prod(input_shape)
          inp = np.arange(
              1, total_size + 1, dtype=datatype).reshape(input_shape)
          np_ans = self._np_transpose(inp, perm)
          with self.cached_session():
            inx = ops.convert_to_tensor(inp)
            y = array_ops.transpose(inx, perm)
            tf_ans = self.evaluate(y)
          self.assertAllEqual(np_ans, tf_ans)
          self.assertShapeEqual(np_ans, y)

  def testLargeSizeGPU(self):
    # [FUNCTION DOC]
    # Test: very large three-dimensional single-precision arrays (millions of elements) with the
    # swap-last-two-axes permutation; checks values and shape against NumPy per sub-test.
    # Returns early (silently passing) when no CUDA GPU is available.
    # If no GPU available, skip the test
    if not test.is_gpu_available(cuda_only=True):
      return

    large_shapes = [[1000000, 31, 3], [3, 1000000, 31], [3, 31, 1000000],
                    [10000, 310, 3], [3, 10000, 310], [3, 310, 10000],
                    [2, 1000, 1000], [1000, 2, 1000], [1000, 1000, 2]]
    perms = [[0, 2, 1]] * 9

    for input_shape, perm in zip(large_shapes, perms):
      with self.subTest(input_shape=input_shape, perm=perm):
        total_size = np.prod(input_shape)
        inp = np.arange(
            1, total_size + 1, dtype=np.float32).reshape(input_shape)
        np_ans = self._np_transpose(inp, perm)
        with self.cached_session():
          inx = ops.convert_to_tensor(inp)
          y = array_ops.transpose(inx, perm)
          tf_ans = self.evaluate(y)
        self.assertAllEqual(np_ans, tf_ans)
        self.assertShapeEqual(np_ans, y)

  def testRandomizedSmallDimLargeSizeGPU(self):
    # [FUNCTION DOC]
    # Test: draws ten random three-dimensional shapes with a randomly sized first dimension
    # (about 40 percent small, 40 percent medium, 20 percent large) and one small and one large
    # other dimension in random order (total about 500 thousand elements), fills them with random
    # small integers, transposes with the swap-last-two-axes permutation and checks values and
    # shape against NumPy, clearing the cached session after each shape to free memory. Uses
    # global random state. Returns early (silently passing) when no CUDA GPU is available.
    # KNOWN ISSUE: the comment claims a 50 percent chance for the second dimension to be the
    # small one; the code does choose each order with equal chance, but ten samples are drawn
    # via integer truncation (4+4+2), which is only correct because the sample count is ten.
    # If no GPU available, skip the test
    if not test.is_gpu_available(cuda_only=True):
      return

    # Draw 10 random shapes with large dimension sizes.
    # 40% prob to generate dim[0] size within [1, 2047]
    # 40% prob to generate dim[0] size within [2048, 4095]
    # 20% prob to generate dim[0] size within [4096, 100000]
    # 50% prob to use dim[1] as the small dim (<16)
    num_samples = 10
    total_size = 500000
    small_size_limit = 2048
    large_size_limit = 95905
    small_size_percentage = 0.4
    medium_size_percentage = 0.4
    large_size_percentage = 0.2
    perms = [[0, 2, 1]] * num_samples
    dim_zero_sizes = []
    dim_zero_sizes += list(
        np.random.randint(
            small_size_limit, size=int(small_size_percentage * num_samples)) +
        1)
    dim_zero_sizes += list(
        np.random.randint(
            small_size_limit, size=int(medium_size_percentage * num_samples)) +
        small_size_limit)
    dim_zero_sizes += list(
        np.random.randint(
            large_size_limit, size=int(large_size_percentage * num_samples)) +
        small_size_limit * 2)
    input_shapes = []
    small_dim_limit = 16
    for dim_zero_size in dim_zero_sizes:
      small_dim_size = np.random.randint(small_dim_limit - 1) + 1
      large_dim_size = int(
          total_size / dim_zero_size / small_dim_size) + small_dim_limit
      input_shapes += ([[dim_zero_size, small_dim_size, large_dim_size]]
                       if np.random.randint(2) else
                       [[dim_zero_size, large_dim_size, small_dim_size]])

    for input_shape, perm in zip(input_shapes, perms):
      # generate input data with random ints from 0 to 9.
      with self.subTest(input_shape=input_shape, perm=perm):
        inp = np.random.randint(10, size=input_shape)
        np_ans = self._np_transpose(inp, perm)
        with self.cached_session():
          inx = ops.convert_to_tensor(inp)
          y = array_ops.transpose(inx, perm)
          tf_ans = self.evaluate(y)
        self.assertAllEqual(np_ans, tf_ans)
        self.assertShapeEqual(np_ans, y)
        self._ClearCachedSession()

  def testNop(self):
    # [FUNCTION DOC]
    # Test: identity permutation on a three by two single-precision array on CPU equals NumPy.
    self._compareCpu(np.arange(0, 6).reshape([3, 2]).astype(np.float32), [0, 1])

  def testSimple(self):
    # [FUNCTION DOC]
    # Test: swapping the two axes of a two by four single-precision array on CPU equals NumPy.
    self._compareCpu(
        np.arange(0, 8).reshape([2, 4]).astype(np.float32),
        np.array([1, 0]).astype(np.int32))

  def testPermType(self):
    # [FUNCTION DOC]
    # Test: the permutation passed as a constant tensor of 64-bit and 32-bit integer dtype must
    # give the same result as NumPy and the right shape (each dtype a sub-test). Returns nothing.
    for perm_dtype in [np.int64, np.int32]:
      with self.subTest(perm_dtype=perm_dtype):
        x = np.arange(0, 8).reshape([2, 4]).astype(np.float32)
        p = np.array([1, 0]).astype(perm_dtype)
        np_ans = np.copy(x).transpose(p)
        with self.cached_session():
          inx = ops.convert_to_tensor(x)
          inp = constant_op.constant(p)
          y = array_ops.transpose(inx, inp)
          tf_ans = self.evaluate(y)
          self.assertShapeEqual(np_ans, y)
          self.assertAllEqual(np_ans, tf_ans)

  def testHalf(self):
    # [FUNCTION DOC]
    # Test: half-precision arrays of rank two, four and eight go through the CPU permutation
    # comparison. Returns nothing.
    self._compare(np.arange(0, 21).reshape([3, 7]).astype(np.float16))
    self._compare(np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.float16))
    self._compare(
        np.arange(0, 16).reshape([1, 2, 1, 2, 1, 2, 1, 2]).astype(np.float16))

  def testBfloat16(self):
    # [FUNCTION DOC]
    # Test: bfloat16 arrays of rank two, four and eight go through the CPU permutation
    # comparison. Returns nothing.
    self._compare(
        np.arange(0, 21).reshape([3, 7]).astype(dtypes.bfloat16.as_numpy_dtype))
    self._compare(
        np.arange(0, 210).reshape([2, 3, 5,
                                   7]).astype(dtypes.bfloat16.as_numpy_dtype))
    self._compare(
        np.arange(0, 16).reshape([1, 2, 1, 2, 1, 2, 1,
                                  2]).astype(dtypes.bfloat16.as_numpy_dtype))

  def testFloat(self):
    # [FUNCTION DOC]
    # Test: single-precision arrays of rank two, four and eight; CPU and GPU results and
    # gradients must agree. Returns nothing.
    self._compare_cpu_gpu(np.arange(0, 21).reshape([3, 7]).astype(np.float32))
    self._compare_cpu_gpu(
        np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.float32))
    self._compare_cpu_gpu(
        np.arange(0, 16).reshape([1, 2, 1, 2, 1, 2, 1, 2]).astype(np.float32))

  def testDouble(self):
    # [FUNCTION DOC]
    # Test: double-precision arrays of rank two, four and eight; CPU and GPU results and
    # gradients must agree. Returns nothing.
    self._compare_cpu_gpu(np.arange(0, 21).reshape([3, 7]).astype(np.float64))
    self._compare_cpu_gpu(
        np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.float64))
    self._compare_cpu_gpu(
        np.arange(0, 16).reshape([1, 2, 1, 2, 1, 2, 1, 2]).astype(np.float64))

  def testComplex64(self):
    # [FUNCTION DOC]
    # Test: single-precision complex data from scalar up to rank six, with and without
    # conjugation, compared on CPU and GPU against NumPy. Returns nothing.
    self._testBoth(np.array(1 + 2j).astype(np.complex64))
    self._testBoth((1 + 2j) * np.arange(0, 21).astype(np.complex64))
    self._testBoth(
        (1 + 2j) * np.arange(0, 21).reshape([3, 7]).astype(np.complex64))
    self._testBoth(
        (1 + 2j) * np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.complex64))
    self._testBoth(
        (1 + 2j) *
        np.arange(0, 1260).reshape([2, 3, 5, 7, 2, 3]).astype(np.complex64))

  def testComplex128(self):
    # [FUNCTION DOC]
    # Test: double-precision complex data from scalar up to rank six, with and without
    # conjugation, compared on CPU and GPU against NumPy. Returns nothing.
    self._testBoth(np.array(1 + 2j).astype(np.complex128))
    self._testBoth((1 + 2j) * np.arange(0, 21).astype(np.complex128))
    self._testBoth(
        (1 + 2j) * np.arange(0, 21).reshape([3, 7]).astype(np.complex128))
    self._testBoth(
        (1 + 2j) *
        np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.complex128))
    self._testBoth(
        (1 + 2j) *
        np.arange(0, 1260).reshape([2, 3, 5, 7, 2, 3]).astype(np.complex128))

  def testInt8(self):
    # [FUNCTION DOC]
    # Test: 8-bit integer arrays of rank two, four and six compared on CPU and GPU. Returns
    # nothing.
    self._testBoth(np.arange(0, 21).reshape([3, 7]).astype(np.int8))
    self._testBoth(np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.int8))
    self._testBoth(
        np.arange(0, 1260).reshape([2, 3, 5, 7, 2, 3]).astype(np.int8))

  def testInt16(self):
    # [FUNCTION DOC]
    # Test: 16-bit integer arrays of rank two, four and six compared on CPU and GPU. Returns
    # nothing.
    self._testBoth(np.arange(0, 21).reshape([3, 7]).astype(np.int16))
    self._testBoth(np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.int16))
    self._testBoth(
        np.arange(0, 1260).reshape([2, 3, 5, 7, 2, 3]).astype(np.int16))

  def testInt32(self):
    # [FUNCTION DOC]
    # Test: 32-bit integer arrays of rank two, four and six compared on CPU and GPU. Returns
    # nothing.
    self._testBoth(np.arange(0, 21).reshape([3, 7]).astype(np.int32))
    self._testBoth(np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.int32))
    self._testBoth(
        np.arange(0, 1260).reshape([2, 3, 5, 7, 2, 3]).astype(np.int32))

  def testInt64(self):
    # [FUNCTION DOC]
    # Test: 64-bit integer arrays of rank two, four and six compared on CPU and GPU. Returns
    # nothing.
    self._testBoth(np.arange(0, 21).reshape([3, 7]).astype(np.int64))
    self._testBoth(np.arange(0, 210).reshape([2, 3, 5, 7]).astype(np.int64))
    self._testBoth(
        np.arange(0, 1260).reshape([2, 3, 5, 7, 2, 3]).astype(np.int64))

  def testTranspose2DAuto(self):
    # [FUNCTION DOC]
    # Test: transposing a nested list of two rows by three columns without a permutation, with
    # and without GPU use, must produce the three by two matrix (each setting a sub-test).
    x_np = [[1, 2, 3], [4, 5, 6]]
    for use_gpu in [False, True]:
      with self.subTest(use_gpu=use_gpu):
        with self.cached_session(use_gpu=use_gpu):
          x_tf = array_ops.transpose(x_np)
          self.assertAllEqual(x_tf, [[1, 4], [2, 5], [3, 6]])

  def testSingletonDims(self):
    # [FUNCTION DOC]
    # Test: shapes containing size-one dimensions (which allow a reshape-based fast path in the
    # kernel) are compared on CPU and GPU against NumPy, per shape as a sub-test. The test only
    # checks result correctness, not that memory copies are avoided. Returns nothing.
    # A singleton dimension is a dimension i with shape[i] == 1. Such dimensions
    # can be collapsed and expanded using reshape without changing the
    # underlying data storage. If all non-singleton dimensions remain in
    # ascending order, the shuffled singletons will be transposed by a reshape,
    # saving a memory allocation & copy. Since this gets a special code-path in
    # transpose_op.cc, we test that the codepath is exercised and the results
    # are as expected; we do not test that we save the memory allocation and
    # copy here.
    for shape in [[2, 1, 2], [2, 1, 2, 1, 1, 2], [1, 2, 2, 1, 1, 1],
                  [1, 1, 1, 2, 2, 2], [2, 2, 1, 1, 1]]:
      with self.subTest(shape=shape):
        self._compare_cpu_gpu(
            np.arange(np.prod(shape)).reshape(shape).astype(np.float32))

  def testTransposeShapes(self):
    # [FUNCTION DOC]
    # Test: static shape inference of transpose for scalar, vector, matrix and rank-three
    # constants, with default and explicit permutations; asserts the inferred dimension lists
    # (default reverses the dimensions). Returns nothing.
    self.assertEqual([],
                     array_ops.transpose(
                         constant_op.constant(1, dtype=dtypes.int32,
                                              shape=[])).get_shape().dims)
    self.assertEqual([100],
                     array_ops.transpose(
                         constant_op.constant(
                             1, dtype=dtypes.int32,
                             shape=[100])).get_shape().dims)
    self.assertEqual([37, 100],
                     array_ops.transpose(
                         constant_op.constant(
                             1, dtype=dtypes.int32,
                             shape=[100, 37])).get_shape().dims)
    self.assertEqual([100, 37],
                     array_ops.transpose(
                         constant_op.constant(
                             1, dtype=dtypes.int32, shape=[100, 37]),
                         [0, 1]).get_shape().dims)
    self.assertEqual([15, 37, 100],
                     array_ops.transpose(
                         constant_op.constant(
                             1, dtype=dtypes.int32,
                             shape=[100, 37, 15])).get_shape().dims)
    self.assertEqual([15, 100, 37],
                     array_ops.transpose(
                         constant_op.constant(
                             1, dtype=dtypes.int32, shape=[100, 37, 15]),
                         [2, 0, 1]).get_shape().dims)

  def testTransposeDynamicShapes(self):
    # [FUNCTION DOC]
    # Test: defines two compiled functions whose input signatures have unknown rank (one with a
    # permutation input too). Inside each, the traced result must have unknown shape; outside,
    # transposing a two by three integer matrix gives the three by two matrix, and an identity
    # permutation returns the input. Returns nothing.
    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=None, dtype=dtypes.int32),
    ])
    def transpose(x):
      # [FUNCTION DOC]
      # Nested compiled function: transposes the unknown-rank integer input x with the default
      # permutation, asserts at trace time that the result shape is fully unknown, and returns
      # the transposed tensor.
      y = array_ops.transpose(x)
      self.assertEqual(y.shape, tensor_shape.TensorShape(None))
      return y

    x = constant_op.constant([[1, 2, 3], [4, 5, 6]])  # Shape (2, 3)
    expected_transpose = constant_op.constant([[1, 4], [2, 5],
                                               [3, 6]])  # Shape (3, 2)
    self.assertAllEqual(expected_transpose, transpose(x))

    @def_function.function(input_signature=[
        tensor_spec.TensorSpec(shape=None, dtype=dtypes.int32),
        tensor_spec.TensorSpec(shape=None, dtype=dtypes.int32),
    ])
    def transpose_with_perm(x, perm):
      # [FUNCTION DOC]
      # Nested compiled function: transposes integer input x with the permutation tensor perm
      # (both unknown rank), asserts at trace time that the result shape is fully unknown, and
      # returns the transposed tensor.
      y = array_ops.transpose(x, perm)
      self.assertEqual(y.shape, tensor_shape.TensorShape(None))
      return y

    self.assertAllEqual(x, transpose_with_perm(x, [0, 1]))

  def testNullTensor(self):
    # [FUNCTION DOC]
    # Test: a float tensor of shape one by four by zero (no elements) transposed with a
    # permutation moving the last two axes has shape one by zero by four. Returns nothing.
    with self.cached_session():
      x = constant_op.constant([], dtype=dtypes.float32, shape=[1, 4, 0])
      xt = array_ops.transpose(x, [0, 2, 1])
      self.assertAllEqual(xt.shape, (1, 0, 4))

  def testScalar(self):
    # [FUNCTION DOC]
    # Test: transposing a scalar float tensor returns an equal scalar. Returns nothing.
    with self.cached_session():
      x = constant_op.constant(42, dtype=dtypes.float32, shape=[])
      xt = array_ops.transpose(x)
      self.assertAllEqual(xt, x)

  def _testError(self, x, p, err):
    # [FUNCTION DOC]
    # HELPER: for testError. Evaluates transpose of x with permutation p in a cached session and
    # asserts that an op error whose message matches the pattern err is raised. Returns nothing.
    with self.cached_session():
      with self.assertRaisesOpError(err):
        self.evaluate(array_ops.transpose(x, p))

  def testError(self):
    # [FUNCTION DOC]
    # Test: invalid permutations on a rank-three array must fail: a rank-two permutation raises
    # a value or invalid-argument error saying it must be rank one; an out-of-range index raises
    # the same kinds of error mentioning range; a repeated index (missing axis two) raises an op
    # error at evaluation. Returns nothing.
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                r"must be rank 1"):
      array_ops.transpose(
          np.arange(0., 30).reshape([2, 3, 5]), [[0, 1], [2, 3]])
    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                r"3 is out of range"):
      array_ops.transpose(np.arange(0., 30).reshape([2, 3, 5]), [0, 1, 3])
    self._testError(
        np.arange(0., 30).reshape([2, 3, 5]), [0, 1, 1], "2 is missing")

  def testNegativePerm(self):
    # [FUNCTION DOC]
    # Test: a permutation with negative entries on a 100 by 37 by 15 constant is accepted and
    # infers a static shape of 15 by 100 by 37, so negative indices count from the end.
    self.assertEqual(
        [15, 100, 37],
        array_ops.transpose(
            constant_op.constant(1, dtype=dtypes.int32, shape=[100, 37, 15]),
            [-1, -3, -2],
        )
        .get_shape()
        .dims,
    )


if __name__ == "__main__":
  test.main()
