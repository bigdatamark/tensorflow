# [MODULE DOC]
# Tests and benchmarks for the matrix band part operation, which keeps a band of diagonals of the
# innermost two dimensions of a (possibly batched) matrix and zeroes the rest. Test cases are
# generated dynamically: factory functions build test bodies for many element types, batch shapes
# and matrix shapes, and a registration helper attaches them to otherwise empty test classes. A
# forward-value test class compares against numpy triangular masks, a gradient test class checks
# numerical gradient error, and a benchmark class times the op on CPU and GPU. Test generation and
# running only happen when the file is executed as a script; importing it has no side effects.
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

import numpy as np

from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gradient_checker
from tensorflow.python.ops import variables
from tensorflow.python.platform import benchmark
from tensorflow.python.platform import test as test_lib


def _AddTest(test, op_name, testcase_name, fn):
  # [FUNCTION DOC]
  # HELPER: registers dynamically generated test bodies on the test classes in this module.
  # Attaches the callable fn to the test class as a method named from the prefix "test", the op
  # name and the test case name joined by underscores. Inputs: test is the test class, op_name and
  # testcase_name are strings, fn is the test body taking the test instance. Returns nothing;
  # mutates the class. Raises a runtime error if a method with that name already exists.
  test_name = "_".join(["test", op_name, testcase_name])
  if hasattr(test, test_name):
    raise RuntimeError("Test %s defined more than once" % test_name)
  setattr(test, test_name, fn)


class MatrixBandPartTest(test_lib.TestCase):
  # [CLASS DOC]
  # Test case for the forward matrix band part op. Initially empty; test methods are attached at
  # run time by the registration helper using bodies from the forward test factory, one per
  # combination of dtype, batch shape and matrix shape.
  pass  # Filled in below


def _GetMatrixBandPartTest(dtype_, batch_shape_, shape_):
  # [FUNCTION DOC]
  # HELPER: factory producing forward-value test bodies for MatrixBandPartTest.
  # Inputs: dtype_ is the numpy element type, batch_shape_ is a tuple of leading batch dimensions
  # (possibly empty), shape_ is the (rows, columns) tuple. Returns the test method closure; no
  # side effects.

  @test_util.run_v1_only("b/120545219")
  def Test(self):
    # [FUNCTION DOC]
    # Test body (graph mode only). Builds a matrix of ones tiled over the batch shape, then for
    # each lower bound and upper bound among -1, 0, 1 and the last row/column count minus one,
    # builds the expected band with numpy upper/lower triangle masks (negative bound means keep
    # everything on that side). For both 32-bit and 64-bit integer bound dtypes it runs the band
    # part op on CPU and asserts the result equals the expected array exactly.
    mat = np.ones(shape_).astype(dtype_)
    batch_mat = np.tile(mat, batch_shape_ + (1, 1))
    for lower in -1, 0, 1, shape_[-2] - 1:
      for upper in -1, 0, 1, shape_[-1] - 1:
        band_np = mat
        if lower >= 0:
          band_np = np.triu(band_np, -lower)
        if upper >= 0:
          band_np = np.tril(band_np, upper)
        if batch_shape_ != ():
          band_np = np.tile(band_np, batch_shape_ + (1, 1))
        for index_dtype in [dtypes_lib.int32, dtypes_lib.int64]:
          with self.cached_session(use_gpu=False):
            band = array_ops.matrix_band_part(
                batch_mat,
                constant_op.constant(lower, index_dtype),
                constant_op.constant(upper, index_dtype))
            self.assertAllEqual(band_np, self.evaluate(band))

  return Test


class MatrixBandPartGradTest(test_lib.TestCase):
  # [CLASS DOC]
  # Test case for the gradient of the matrix band part op. Initially empty; test methods are
  # attached at run time from the gradient test factory for float32 and float64 shapes.
  pass  # Filled in below


def _GetMatrixBandPartGradTest(dtype_, batch_shape_, shape_):
  # [FUNCTION DOC]
  # HELPER: factory producing gradient test bodies for MatrixBandPartGradTest.
  # Inputs: dtype_ element type, batch_shape_ tuple of batch dimensions, shape_ (rows, columns)
  # tuple. Returns the test method closure; no side effects.

  @test_util.run_v1_only("b/120545219")
  def Test(self):
    # [FUNCTION DOC]
    # Test body (graph mode only). Creates a random constant of batch plus matrix shape and, for
    # each combination of lower and upper bounds, applies the band part op and compares analytic
    # and numeric gradients with the gradient checker, asserting the error is below 1e-4.
    shape = batch_shape_ + shape_
    x = constant_op.constant(np.random.rand(*shape), dtype=dtype_)
    with self.session(use_gpu=False):
      for lower in -1, 0, 1, shape_[-2] - 1:
        for upper in -1, 0, 1, shape_[-1] - 1:
          y = array_ops.matrix_band_part(x, lower, upper)
          error = gradient_checker.compute_gradient_error(
              x, x.get_shape().as_list(), y, y.get_shape().as_list())
          self.assertLess(error, 1e-4)

  return Test


class MatrixBandPartBenchmark(test_lib.Benchmark):
  # [CLASS DOC]
  # Benchmark for the matrix band part op over a fixed list of batched matrix shapes (class
  # attribute), run on CPU and, when available, GPU.

  shapes = [
      (10, 16, 16),
      (10, 101, 101),
      (10, 256, 256),
      (10, 1000, 1000),
      (10, 1024, 1024),
      (10, 2048, 2048),
      (10, 10, 4, 4),
      (10, 10, 10, 10),
      (10, 10, 16, 16),
      (10, 10, 101, 101),
      (10, 10, 256, 256),
      (10, 10, 1000, 1000),
      (10, 10, 1024, 1024),
      (10, 10, 2048, 2048),
  ]

  def benchmarkMatrixBandPartOp(self):
    # [FUNCTION DOC]
    # For every shape and every band limit pair (full, lower-only, upper-only, width two), builds
    # a fresh graph with a variable of ones, applies the band part op, initializes variables and
    # runs the op benchmark (at least 10 iterations) on the CPU, then again on the GPU only if a
    # GPU is available. Takes no inputs, returns nothing; reports benchmark results with names
    # encoding device, shape and limits.
    for shape_ in self.shapes:
      for limits in (-1, -1), (-1, 0), (0, -1), (2, 2):
        with ops.Graph().as_default(), \
            session.Session(config=benchmark.benchmark_config()) as sess, \
            ops.device("/cpu:0"):
          matrix = variables.Variable(array_ops.ones(shape_))
          band = array_ops.matrix_band_part(matrix, limits[0], limits[1])
          self.evaluate(variables.global_variables_initializer())
          self.run_op_benchmark(
              sess,
              control_flow_ops.group(band),
              min_iters=10,
              name="matrix_band_part_cpu_{shape}_{limits}".format(
                  shape=shape_, limits=limits))

        if test_lib.is_gpu_available(True):
          with ops.Graph().as_default(), \
              session.Session(config=benchmark.benchmark_config()) as sess, \
              ops.device("/gpu:0"):
            matrix = variables.Variable(array_ops.ones(shape_))
            band = array_ops.matrix_band_part(matrix, limits[0], limits[1])
            self.evaluate(variables.global_variables_initializer())
            self.run_op_benchmark(
                sess,
                control_flow_ops.group(band),
                min_iters=10,
                name="matrix_band_part_gpu_{shape}_{limits}".format(
                    shape=shape_, limits=limits))


if __name__ == "__main__":
  dtypes = (np.bool_, np.int32, np.int64, np.float16,
            dtypes_lib.bfloat16.as_numpy_dtype, np.float32, np.float64,
            np.complex64, np.complex128)
  for dtype in dtypes:
    for batch_shape in ((), (2,), (1, 3, 2)):
      for rows in 1, 2, 7, 23:
        for cols in 1, 2, 7, 23:
          shape = (rows, cols)
          name = "%s_%s" % (dtype.__name__,
                            "_".join(map(str, batch_shape + shape)))
          _AddTest(MatrixBandPartTest, "MatrixBandPart", name,
                   _GetMatrixBandPartTest(dtype, batch_shape, shape))

  for dtype in (np.float32, np.float64):
    for batch_shape in ((), (2,)):
      for rows in 1, 2, 7:
        for cols in 1, 2, 7:
          shape = (rows, cols)
          name = "%s_%s" % (dtype.__name__,
                            "_".join(map(str, batch_shape + shape)))
          _AddTest(MatrixBandPartGradTest, "MatrixBandPartGrad", name,
                   _GetMatrixBandPartGradTest(dtype, batch_shape, shape))

  test_lib.main()
