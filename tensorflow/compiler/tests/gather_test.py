# [MODULE DOC]
# Tests and microbenchmarks for the gather op on the XLA test device. The test class covers
# scalar and vector indices, 2-D params along several axes, 64-bit indices, higher ranks with
# empty or scalar index shapes, differently shaped indices and precision. The benchmark class
# measures slice-like and general gathers on a large tensor with and without the XLA JIT.
# Defines a module-level alias to the command-line flags object. Runs the test main when
# executed directly.
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
"""Functional tests for XLA Gather Op."""

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import flags
from tensorflow.python.platform import test

FLAGS = flags.FLAGS


class GatherTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case for the gather op across all supported TensorFlow types. Stateless; uses a
  # helper to prepare numpy parameters for each type.

  def _buildParams(self, data, dtype):
    # [FUNCTION DOC]
    # HELPER: converts a numpy array to the numpy equivalent of the given TensorFlow type.
    # For complex types, adds ten times the data as an imaginary part so wrong elements are
    # distinguishable. Returns the converted array.
    data = data.astype(dtype.as_numpy_dtype)
    # For complex types, adds an index-dependent imaginary component so we can
    # tell we got the right value.
    if dtype.is_complex:
      return data + 10j * data
    return data

  def testScalar1D(self):
    # [FUNCTION DOC]
    # For every type, gathers from a 1-D array using a scalar index, a one-element list and a
    # list of several indices; compares with numpy indexing.
    with self.session() as session, self.test_scope():
      data = np.array([0, 1, 2, 3, 7, 5])
      for dtype in self.all_tf_types:
        for indices in 4, [4], [1, 2, 2, 4, 5]:
          params_np = self._buildParams(data, dtype)
          params = array_ops.placeholder(dtype=dtype)
          indices_tf = constant_op.constant(indices)
          gather_t = array_ops.gather(params, indices_tf)
          gather_val = session.run(gather_t, feed_dict={params: params_np})
          np_val = constant_op.constant(params_np[indices])
          self.assertAllEqual(np_val, gather_val)

  def testScalar2D(self):
    # [FUNCTION DOC]
    # For every type and axes zero, one and minus one, gathers with scalar index two from a 5x3
    # array and compares with numpy take.
    with self.session() as session, self.test_scope():
      data = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11],
                       [12, 13, 14]])
      for dtype in self.all_tf_types:
        for axis in 0, 1, -1:
          params_np = self._buildParams(data, dtype)
          params = array_ops.placeholder(dtype=dtype)
          indices = constant_op.constant(2)
          gather_t = array_ops.gather(params, indices, axis=axis)
          gather_val = session.run(gather_t, feed_dict={params: params_np})
          expected = constant_op.constant(
              np.take(params_np, 2, axis=axis), dtype)
          self.assertAllEqual(expected, gather_val)

  def testSimpleTwoD32(self):
    # [FUNCTION DOC]
    # For every type and axes zero, one and minus one, gathers with indices zero, one, zero, two
    # (in bounds for every axis) from a 5x3 array and compares with numpy take.
    with self.session() as session, self.test_scope():
      data = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11],
                       [12, 13, 14]])
      for dtype in self.all_tf_types:
        for axis in 0, 1, -1:
          params_np = self._buildParams(data, dtype)
          params = array_ops.placeholder(dtype=dtype)
          # The indices must be in bounds for any axis.
          indices = constant_op.constant([0, 1, 0, 2])
          gather_t = array_ops.gather(params, indices, axis=axis)
          gather_val = session.run(gather_t, feed_dict={params: params_np})
          expected = constant_op.constant(
              np.take(params_np, [0, 1, 0, 2], axis=axis), dtype)
          self.assertAllEqual(expected, gather_val)

  def testSimpleTwoD32_Int64Indices(self):
    # [FUNCTION DOC]
    # Returns immediately (passes) if the device does not support 64-bit integers. Otherwise
    # same as the simple 2-D test but indices are fed through a 64-bit integer placeholder.
    if np.int64 not in self.int_types:
      return

    with self.session() as session, self.test_scope():
      data = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11],
                       [12, 13, 14]])
      # The indices must be in bounds for any axis.
      indices_np = np.array([0, 1, 0, 2])
      for dtype in self.all_tf_types:
        for axis in 0, 1, -1:
          params_np = self._buildParams(data, dtype)
          params = array_ops.placeholder(dtype=dtype)
          indices = array_ops.placeholder(dtype=dtypes.int64)
          gather_t = array_ops.gather(params, indices, axis=axis)
          gather_val = session.run(
              gather_t, feed_dict={
                  params: params_np,
                  indices: indices_np
              })
          expected = constant_op.constant(
              np.take(params_np, [0, 1, 0, 2], axis=axis), dtype)
          self.assertAllEqual(expected, gather_val)

  def testHigherRank(self):
    # [FUNCTION DOC]
    # Uses random 4-D params and random in-range indices of shapes scalar, empty, two by zero
    # and two by three, for all types and six axes (including negative); compares with numpy
    # take. Uses unseeded random data.
    """Check that scalar and empty indices shapes work as well."""
    shape = (2, 1, 3, 2)
    for indices_shape in (), (0,), (2, 0), (2, 3):
      for dtype in self.all_tf_types:
        for axis in 0, 1, 2, 3, -1, -2:
          params = self._buildParams(np.random.randn(*shape), dtype)
          indices = np.random.randint(shape[axis], size=indices_shape)
          with self.session() as sess, self.test_scope():
            tf_params = array_ops.placeholder(dtype=dtype)
            tf_indices = constant_op.constant(indices, dtype=dtypes.int32)
            gather = array_ops.gather(tf_params, tf_indices, axis=axis)
            gather_value = sess.run(gather, feed_dict={tf_params: params})
            gather_np = constant_op.constant(
                np.take(params, indices, axis=axis), dtype)
            self.assertAllEqual(gather_np, gather_value)

  def testIndicesWithDifferentDimensions(self):
    # [FUNCTION DOC]
    # For each numeric type, gathers from a 3-element vector with scalar, 1-D and 2-D indices
    # fed at run time; checks the result rank follows the index rank (7, [7], [[7]]).
    with self.session():
      for dtype in self.numeric_tf_types:
        params = array_ops.placeholder(dtype=dtype)
        indices = array_ops.placeholder(dtype=np.int32)
        with self.test_scope():
          gather = array_ops.gather(params, indices)
        self.assertAllEqual(
            7, gather.eval(feed_dict={params: [4, 7, 2], indices: 1}))
        self.assertAllEqual(
            [7], gather.eval(feed_dict={params: [4, 7, 2], indices: [1]}))
        self.assertAllEqual(
            [[7]], gather.eval(feed_dict={params: [4, 7, 2], indices: [[1]]}))

  def testGatherPrecision(self):
    # [FUNCTION DOC]
    # Gathers rows of a float32 array containing a value slightly above two and fractional
    # values; asserts exact equality with numpy indexing, i.e. gather must not lose precision.
    with self.session() as session, self.test_scope():
      data = np.array([[0, 0, 0, 0], [0, 2 * (1 + np.exp2(-8)), 0, 0],
                       [0, 0, 0, 0], [0.015789, 0.0985, 0.55789, 0.3842]])
      indices = np.array([1, 2, 3, 1])
      dtype = dtypes.float32
      params_np = self._buildParams(data, dtype)
      params = array_ops.placeholder(dtype=dtype)
      indices_tf = constant_op.constant(indices)
      gather_t = array_ops.gather(params, indices_tf)
      gather_val = session.run(gather_t, feed_dict={params: params_np})
      np_val = params_np[indices]
      self.assertAllEqual(np_val, gather_val)


class GatherBenchmark(test.Benchmark):
  # [CLASS DOC]
  # Microbenchmark suite for the gather op on a large five-dimensional tensor, for axes 0, 1
  # and 4, with a one-element (slice-like) index list and a longer index list, each with and
  # without the XLA JIT. Benchmark entry points are the public methods; private ones build
  # and run them.
  """Microbenchmarks for the gather op."""

  def _benchmarkGather(self, name, axis, gather_indices, use_xla_jit):
    # [FUNCTION DOC]
    # HELPER: shared benchmark driver.
    # Inputs: a name prefix, the gather axis, a list of integer indices, and whether to enable
    # the XLA JIT. Defines a builder that creates a large float zero variable, an index
    # variable and a gather node, and hands it to the XLA benchmark runner on the CPU device.
    # Returns nothing; results are reported by the runner.

    def BuilderFn():
      # [FUNCTION DOC]
      # HELPER: graph builder passed to the benchmark runner. Creates the input and index
      # variables and the gather output; returns a pair of a benchmark label (name and axis)
      # and a list containing the gather output.
      inputs = variables.Variable(
          array_ops.zeros([100, 100, 10, 100, 50], dtype=dtypes.float32),
          dtype=dtypes.float32,
          name='input')
      indices = variables.Variable(
          gather_indices, dtype=dtypes.int32, name='indices')
      gather_t = array_ops.gather(inputs, indices, axis=axis)
      return '%s.axis%d' % (name, axis), [gather_t]

    xla_test.Benchmark(self, BuilderFn, use_xla_jit=use_xla_jit, device='cpu')

  def _benchmarkSliceGather(self, axis, use_xla_jit):
    # [FUNCTION DOC]
    # HELPER: benchmarks a single-index gather (equivalent to a dynamic slice) for the given
    # axis and JIT setting.
    """Benchmarks a gather op that's really a dynamic slice."""
    self._benchmarkGather('slice_gather', axis, [1], use_xla_jit)

  def _benchmarkNontrivialGather(self, axis, use_xla_jit):
    # [FUNCTION DOC]
    # HELPER: benchmarks a gather with a sixteen-element index list (nine, one, zero, two
    # repeated four times) for the given axis and JIT setting.
    self._benchmarkGather('nontrivial_gather', axis, [9, 1, 0, 2] * 4,
                          use_xla_jit)

  def benchmarkSliceGatherAxis0(self):
    # [FUNCTION DOC]
    # Benchmark: slice-like gather along axis 0 without the XLA JIT.
    self._benchmarkSliceGather(axis=0, use_xla_jit=False)

  def benchmarkSliceGatherAxis0XLA(self):
    # [FUNCTION DOC]
    # Benchmark: slice-like gather along axis 0 with the XLA JIT.
    self._benchmarkSliceGather(axis=0, use_xla_jit=True)

  def benchmarkSliceGatherAxis1(self):
    # [FUNCTION DOC]
    # Benchmark: slice-like gather along axis 1 without the XLA JIT.
    self._benchmarkSliceGather(axis=1, use_xla_jit=False)

  def benchmarkSliceGatherAxis1XLA(self):
    # [FUNCTION DOC]
    # Benchmark: slice-like gather along axis 1 with the XLA JIT.
    self._benchmarkSliceGather(axis=1, use_xla_jit=True)

  def benchmarkSliceGatherAxis4(self):
    # [FUNCTION DOC]
    # Benchmark: slice-like gather along axis 4 without the XLA JIT.
    self._benchmarkSliceGather(axis=4, use_xla_jit=False)

  def benchmarkSliceGatherAxis4XLA(self):
    # [FUNCTION DOC]
    # Benchmark: slice-like gather along axis 4 with the XLA JIT.
    self._benchmarkSliceGather(axis=4, use_xla_jit=True)

  def benchmarkNontrivialGatherAxis0(self):
    # [FUNCTION DOC]
    # Benchmark: multi-index gather along axis 0 without the XLA JIT.
    self._benchmarkNontrivialGather(axis=0, use_xla_jit=False)

  def benchmarkNontrivialGatherAxis0XLA(self):
    # [FUNCTION DOC]
    # Benchmark: multi-index gather along axis 0 with the XLA JIT.
    self._benchmarkNontrivialGather(axis=0, use_xla_jit=True)

  def benchmarkNontrivialGatherAxis1(self):
    # [FUNCTION DOC]
    # Benchmark: multi-index gather along axis 1 without the XLA JIT.
    self._benchmarkNontrivialGather(axis=1, use_xla_jit=False)

  def benchmarkNontrivialGatherAxis1XLA(self):
    # [FUNCTION DOC]
    # Benchmark: multi-index gather along axis 1 with the XLA JIT.
    self._benchmarkNontrivialGather(axis=1, use_xla_jit=True)

  def benchmarkNontrivialGatherAxis4(self):
    # [FUNCTION DOC]
    # Benchmark: multi-index gather along axis 4 without the XLA JIT.
    self._benchmarkNontrivialGather(axis=4, use_xla_jit=False)

  def benchmarkNontrivialGatherAxis4XLA(self):
    # [FUNCTION DOC]
    # Benchmark: multi-index gather along axis 4 with the XLA JIT.
    self._benchmarkNontrivialGather(axis=4, use_xla_jit=True)


if __name__ == '__main__':
  test.main()
