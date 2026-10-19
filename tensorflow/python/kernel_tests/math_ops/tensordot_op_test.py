# [MODULE DOC]
# Tests for the tensordot operation in the math ops module. Contains a small hand-written test
# case (invalid shapes, invalid axes, valid axes, partial shape inference) plus a factory that
# builds randomized comparison tests against the NumPy implementation for several dtypes, tensor
# ranks, numbers of contracted dimensions, and static versus placeholder (dynamic) shapes.
# KNOWN ISSUE: the generated randomized tests are only attached to the test class inside the
# main-program block, so they do not exist when the module is merely imported by a test runner.
# Not a helper module; no import-time side effects beyond defining a constant for max dimension.
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
"""Tests for tensorflow.ops.math_ops.matmul."""

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test as test_lib

_MAXDIM = 5


def _add_test(test, test_name, fn):
  # [FUNCTION DOC]
  # HELPER: registers dynamically generated test functions on the tensordot test class.
  # Attaches the given function to the given test class under a name made by prefixing the
  # supplied name with the test prefix. Inputs: test class, base name string, function.
  # Raises a runtime error if an attribute with that name already exists. Mutates the class;
  # returns nothing.
  test_name = "_".join(["test", test_name])
  if hasattr(test, test_name):
    raise RuntimeError("Test %s defined more than once" % test_name)
  setattr(test, test_name, fn)


class TensordotTest(test_lib.TestCase):
  # [CLASS DOC]
  # Test case for the tensordot operation. Holds hand-written tests for error handling and shape
  # inference; additional randomized tests are attached to it dynamically from the main block
  # through the registration helper. No shared fixtures.

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def test_invalid_shape(self):
    # [FUNCTION DOC]
    # Test: contracts a 2x2 matrix with a 3x2 matrix over incompatible axes. Expects a value
    # error or invalid-argument error for the statically known shapes. In graph mode it also
    # feeds the same data through placeholders and expects an op error reporting a matrix size
    # incompatibility. Runs in both graph and eager modes; returns nothing.
    a = [[1, 2], [3, 4]]
    b = [[1, 2], [3, 4], [5, 6]]
    a_axes = [1]
    b_axes = [0]
    # Invalid static shapes.
    with self.assertRaises((ValueError, errors_impl.InvalidArgumentError)):
      math_ops.tensordot(a, b, (a_axes, b_axes))

    # Invalid dynamic shapes.
    if context.executing_eagerly():
      return
    with self.cached_session() as sess:
      with self.assertRaisesOpError(
          r"Matrix size-incompatible: In\[0\]: \[2,2\], In\[1\]: \[3,2\]"):
        a_ph = array_ops.placeholder(dtypes.float32)
        b_ph = array_ops.placeholder(dtypes.float32)
        axes_ph = array_ops.placeholder(dtypes.int32)
        output = math_ops.tensordot(a_ph, b_ph, axes_ph)
        _ = sess.run(
            [output], feed_dict={
                a_ph: a,
                b_ph: b,
                axes_ph: (a_axes, b_axes)
            })

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def test_invalid_axes(self):
    # [FUNCTION DOC]
    # Test: calls tensordot on two 2x2 matrices with several malformed static axes (out-of-range,
    # wrong nesting, mismatched lengths) and expects a value error; an out-of-range axis index
    # in a nested list expects an index error. In graph mode only, repeats with placeholder
    # inputs and a placeholder axes tensor and expects an invalid-argument error at run time
    # (scalar axes tensors are unsupported). Returns nothing.
    a = [[1, 2], [3, 4]]
    b = [[1, 2], [3, 4]]
    # Invalid static axes.
    for axes_value in -1, 3, [1], [[1]], [[1], [0, 1]]:
      with self.assertRaises(ValueError):
        math_ops.tensordot(a, b, axes_value)

    with self.assertRaises(IndexError):
      math_ops.tensordot(a, b, [[0], [7]])

    if context.executing_eagerly():
      return
    # Invalid dynamic axes.
    a_ph = array_ops.placeholder(dtypes.float32)
    b_ph = array_ops.placeholder(dtypes.float32)
    axes_ph = array_ops.placeholder(dtypes.int32)
    output = math_ops.tensordot(a_ph, b_ph, axes_ph)
    # Note: We don't support scalar Tensor values for axes.
    for axes_value in 1, [1], [0, 1], [[1]], [[0, 1]], [[0], [7]]:
      with self.cached_session() as sess:
        with self.assertRaises(errors_impl.InvalidArgumentError):
          _ = sess.run(
              [output], feed_dict={
                  a_ph: a,
                  b_ph: b,
                  axes_ph: axes_value
              })

  # Test case for https://github.com/tensorflow/tensorflow/issues/11950
  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def test_valid_axis(self):
    # [FUNCTION DOC]
    # Regression test for a reported issue about valid axes specifications. For several axes
    # forms (list, nested lists, empty axes, scalar zero) it computes tensordot of a 3x3 ones
    # tensor with a 1x1x3 tensor in both NumPy and TensorFlow, and asserts that shapes and values
    # are equal. Returns nothing.
    for axes_value in [1, 2], [[1], [2]], [[], []], 0:
      np_a = np.ones((3, 3))
      np_b = np.array([2, 3, 1])[None, None]
      np_ans = np.tensordot(np_a, np_b, axes_value)

      tf_a = array_ops.ones((3, 3), dtype=dtypes.float32)
      tf_b = constant_op.constant([2, 3, 1], dtype=dtypes.float32)[None, None]
      tf_ans = math_ops.tensordot(tf_a, tf_b, axes_value)

      self.assertAllEqual(tf_ans.shape, np_ans.shape)
      self.assertAllEqual(self.evaluate(tf_ans), np_ans)

  @test_util.run_v1_only("Shape inference test")
  def test_partial_shape_inference(self):
    # [FUNCTION DOC]
    # Test (graph-mode only): with placeholders of unknown rank, the tensordot result must have
    # unknown rank. After setting partial static shapes (unknown by 2 against 2 by 3, then 2 by 2
    # against 2 by unknown), the result must have rank two with the known dimensions inferred and
    # unknown dimensions left unknown. Uses scalar and list axes forms. Returns nothing.
    for axes in ([1], [0]), 1:
      a = array_ops.placeholder(dtypes.float32)
      b = array_ops.placeholder(dtypes.float32)
      output = math_ops.tensordot(a, b, axes)
      self.assertEqual(output.get_shape().ndims, None)
      a.set_shape([None, 2])
      b.set_shape([2, 3])
      output = math_ops.tensordot(a, b, axes)
      output_shape = output.get_shape()
      self.assertEqual(output_shape.ndims, 2)
      output_shape = output_shape.as_list()
      self.assertEqual(output_shape[0], None)
      self.assertEqual(output_shape[1], 3)
      a = array_ops.placeholder(dtypes.float32)
      b = array_ops.placeholder(dtypes.float32)
      a.set_shape([2, 2])
      b.set_shape([2, None])
      output = math_ops.tensordot(a, b, axes)
      output_shape = output.get_shape()
      self.assertEqual(output_shape.ndims, 2)
      output_shape = output_shape.as_list()
      self.assertEqual(output_shape[0], 2)
      self.assertEqual(output_shape[1], None)


def _get_tensordot_tests(dtype_, rank_a_, rank_b_, num_dims_, dynamic_shape_):
  # [FUNCTION DOC]
  # HELPER: factory producing randomized tensordot test functions for the test class.
  # Inputs: element dtype, rank of first operand, rank of second operand, number of contracted
  # dimensions, and a flag selecting placeholder (dynamic shape) versus constant inputs.
  # Returns a list of two test functions (general axes and scalar axes), each taking a test
  # instance, closed over these parameters. No side effects when called.

  # Select a random subset of size m from [0, 1, ..., n-1].
  def _random_subset(m, n):
    # [FUNCTION DOC]
    # HELPER: picks m distinct indices out of the first n integers in random order, returned as
    # a 32-bit integer array. Asserts (assertion error) that m does not exceed n.
    assert m <= n
    return (np.random.permutation(n)[:m]).astype(np.int32)

  def _generate_random_tensors_and_dims():
    # [FUNCTION DOC]
    # HELPER: builds two random arrays of the enclosing dtype and ranks whose sizes agree along
    # randomly chosen contracted dimensions. Returns the first array, the second array, and the
    # two lists of contracted dimension indices. Uses the global random state.
    # KNOWN ISSUE: relies on a NumPy random-integer routine that is deprecated and removed in
    # recent NumPy releases.
    a_shape = np.random.random_integers(1, _MAXDIM, rank_a_)
    b_shape = np.random.random_integers(1, _MAXDIM, rank_b_)
    shared_shape = np.random.random_integers(1, _MAXDIM, num_dims_)
    a_dims = _random_subset(num_dims_, rank_a_)
    b_dims = _random_subset(num_dims_, rank_b_)
    for i in range(num_dims_):
      a_shape[a_dims[i]] = shared_shape[i]
      b_shape[b_dims[i]] = shared_shape[i]
    a = np.random.uniform(
        low=-1.0, high=1.0,
        size=np.prod(a_shape)).reshape(a_shape).astype(dtype_)
    b = np.random.uniform(
        low=-1.0, high=1.0,
        size=np.prod(b_shape)).reshape(b_shape).astype(dtype_)
    return a, b, a_dims, b_dims

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  @test_util.run_without_tensor_float_32("Tests tensordot, which calls matmul")
  def test_tensordot(self):
    # [FUNCTION DOC]
    # Test (generated): for a bounded number of trials, builds random operands, computes tensordot
    # in NumPy as reference and in TensorFlow (via placeholders with fed axes when dynamic shapes
    # are requested, else directly), and asserts closeness within a dtype-dependent tolerance
    # (loose for half precision, tighter for single, very tight otherwise) and equal shapes.
    # Skipped in eager mode when using placeholders. Returns nothing.
    if dynamic_shape_ and context.executing_eagerly():
      self.skipTest("Placeholders not support in eager mode")
    num_trials = min(30, num_dims_ * num_dims_)
    if dtype_ == np.float16:
      tol = 0.05
    elif dtype_ == np.float32 or dtype_ == np.complex64:
      tol = 1e-5
    else:
      tol = 1e-12
    for _ in range(num_trials):
      a_np, b_np, a_dims_np, b_dims_np = _generate_random_tensors_and_dims()
      np_ans = np.tensordot(a_np, b_np, axes=(a_dims_np, b_dims_np))
      with self.cached_session() as sess:
        if dynamic_shape_:
          a = array_ops.placeholder(dtype_)
          b = array_ops.placeholder(dtype_)
          axes = array_ops.placeholder(dtypes.int32)
          c = math_ops.tensordot(a, b, axes)
          tf_ans = sess.run(
              c, feed_dict={
                  a: a_np,
                  b: b_np,
                  axes: (a_dims_np, b_dims_np)
              })
        else:
          tf_ans = math_ops.tensordot(a_np, b_np, (a_dims_np, b_dims_np))
      self.assertAllClose(tf_ans, np_ans, rtol=tol, atol=tol)
      self.assertAllEqual(tf_ans.shape, np_ans.shape)

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  @test_util.run_without_tensor_float_32("Tests tensordot, which calls matmul")
  def test_tensordot_scalar_axes(self):
    # [FUNCTION DOC]
    # Test (generated): uses two random arrays whose every dimension has size five, and for
    # scalar axes values zero, one and (when rank exceeds two) the last dimension count, compares
    # tensordot against NumPy for values (within a dtype-dependent tolerance) and shapes. Skipped
    # in eager mode with placeholders, and skipped when the number of dimensions is zero.
    # Returns nothing.
    if dynamic_shape_ and context.executing_eagerly():
      self.skipTest("Placeholders not support in eager mode")
    if num_dims_ < 1:
      self.skipTest("Not a test")
    if dtype_ == np.float16:
      tol = 0.05
    elif dtype_ == np.float32 or dtype_ == np.complex64:
      tol = 1e-5
    else:
      tol = 1e-12
    shape = [5] * num_dims_
    a_np = np.random.uniform(
        low=-1.0, high=1.0, size=np.prod(shape)).reshape(shape).astype(dtype_)
    b_np = np.random.uniform(
        low=-1.0, high=1.0, size=np.prod(shape)).reshape(shape).astype(dtype_)
    all_axes = [0, 1]
    if a_np.ndim > 2:
      all_axes.append(a_np.ndim - 1)
    for axes in all_axes:
      np_ans = np.tensordot(a_np, b_np, axes=axes)
      with self.cached_session() as sess:
        if dynamic_shape_:
          a = array_ops.placeholder(dtype_)
          b = array_ops.placeholder(dtype_)
          c = math_ops.tensordot(a, b, axes=axes)
          tf_ans = sess.run(c, feed_dict={a: a_np, b: b_np})
        else:
          tf_ans = math_ops.tensordot(a_np, b_np, axes=axes)
      self.assertAllClose(tf_ans, np_ans, rtol=tol, atol=tol)
      self.assertAllEqual(tf_ans.shape, np_ans.shape)

  return [test_tensordot, test_tensordot_scalar_axes]


if __name__ == "__main__":
  dtypes_to_test = [
      np.float16, np.float32, np.float64, np.complex64, np.complex128
  ]
  for dtype in dtypes_to_test:
    for rank_a in 1, 2, 4, 5:
      for rank_b in 1, 2, 4, 5:
        for num_dims in range(0, min(rank_a, rank_b) + 1):
          # TF2 does not support placeholders under eager so we skip it
          for dynamic_shape in set([False, True]):
            for testcase in _get_tensordot_tests(dtype, rank_a, rank_b,
                                                 num_dims, dynamic_shape):
              name = "%s_%s_%s_%s_%s_%s" % (testcase.__name__, dtype.__name__,
                                            rank_a, rank_b, num_dims,
                                            dynamic_shape)
              _add_test(TensordotTest, name, testcase)
  test_lib.main()
