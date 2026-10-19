# [MODULE DOC]
# KNOWN ISSUE: the generated tests are only attached to the test class when the file is run as a
# script; if the module is merely imported (for example by a different test runner) only the two
# error-checking tests exist.
# Unit tests for linalg_ops.norm (the generic vector or matrix norm). Contains a small helper that
# registers dynamically generated test methods, a test class with two error-handling tests, and a
# factory that builds one comparison test (against numpy's norm) per combination of dtype, shape,
# order, axis, keep-dimensions flag and static-or-dynamic shape. The combinations are generated
# and registered in the script entry point, which then starts the test runner.
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
"""Tests for tensorflow.ops.tf.norm."""

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.platform import test as test_lib


def _AddTest(test, test_name, fn):
  # [FUNCTION DOC]
  # HELPER: registers a dynamically generated test method on a test class. Inputs: the test class,
  # a name suffix and the test function. The attribute name is the suffix prefixed with the word
  # test. Raises a runtime error if a method of that name already exists; otherwise sets the
  # attribute on the class (mutating it) and returns nothing.
  test_name = "_".join(["test", test_name])
  if hasattr(test, test_name):
    raise RuntimeError("Test %s defined more than once" % test_name)
  setattr(test, test_name, fn)


class NormOpTest(test_lib.TestCase):
  # [CLASS DOC]
  # Test case for the norm op. Defines two static error-checking tests; many more comparison
  # tests are attached to it dynamically by the script entry point via the registration helper.

  @test_util.run_v1_only("b/120545219")
  def testBadOrder(self):
    # [FUNCTION DOC]
    # Scenario: a 2x2 matrix is passed with unsupported norm orders. With no axis or a single
    # axis, the orders fro, -7, -1.1 and 0 must raise a value error saying the order must be a
    # supported vector norm; with a two-element axis, the orders foo, -7, -1.1 and 1.1 must raise
    # a value error saying it must be a supported matrix norm. Runs only under TF1 behavior.
    matrix = [[0., 1.], [2., 3.]]
    for ord_ in "fro", -7, -1.1, 0:
      with self.assertRaisesRegex(ValueError,
                                  "'ord' must be a supported vector norm"):
        linalg_ops.norm(matrix, ord=ord_)

    for ord_ in "fro", -7, -1.1, 0:
      with self.assertRaisesRegex(ValueError,
                                  "'ord' must be a supported vector norm"):
        linalg_ops.norm(matrix, ord=ord_, axis=-1)

    for ord_ in "foo", -7, -1.1, 1.1:
      with self.assertRaisesRegex(ValueError,
                                  "'ord' must be a supported matrix norm"):
        linalg_ops.norm(matrix, ord=ord_, axis=[-2, -1])

  @test_util.run_v1_only("b/120545219")
  def testInvalidAxis(self):
    # [FUNCTION DOC]
    # Scenario: a 2x2 matrix is passed with invalid axis values (empty list, three entries, nested
    # lists, a non-integer, or repeated entries). Expects a value error stating the axis must be
    # none, an integer, or a tuple of two unique integers. Runs only under TF1 behavior.
    matrix = [[0., 1.], [2., 3.]]
    for axis_ in [], [1, 2, 3], [[1]], [[1], [2]], [3.1415], [1, 1]:
      error_prefix = ("'axis' must be None, an integer, or a tuple of 2 unique "
                      "integers")
      with self.assertRaisesRegex(ValueError, error_prefix):
        linalg_ops.norm(matrix, axis=axis_)


def _GetNormOpTest(dtype_, shape_, ord_, axis_, keep_dims_, use_static_shape_):
  # [FUNCTION DOC]
  # HELPER: factory producing one test method for the norm op. Inputs: numpy dtype, full shape
  # list, norm order, axis (none, integer or pair), keep-dimensions flag, and whether to feed the
  # input as a constant with static shape (true) or through a placeholder (false). Returns the
  # test function, which closes over these parameters; nothing is registered here.

  def _CompareNorm(self, matrix):
    # [FUNCTION DOC]
    # HELPER: nested check for the generated test. Computes the norm of the given numpy matrix
    # with numpy and with the TensorFlow op (as a constant, or fed to a placeholder in a session)
    # using the captured order, axis and keep-dimensions settings, then asserts they are close
    # (tolerance 1e-5). Returns nothing.
    np_norm = np.linalg.norm(matrix, ord=ord_, axis=axis_, keepdims=keep_dims_)
    with self.cached_session() as sess:
      if use_static_shape_:
        tf_matrix = constant_op.constant(matrix)
        tf_norm = linalg_ops.norm(
            tf_matrix, ord=ord_, axis=axis_, keepdims=keep_dims_)
        tf_norm_val = self.evaluate(tf_norm)
      else:
        tf_matrix = array_ops.placeholder(dtype_)
        tf_norm = linalg_ops.norm(
            tf_matrix, ord=ord_, axis=axis_, keepdims=keep_dims_)
        tf_norm_val = sess.run(tf_norm, feed_dict={tf_matrix: matrix})
    self.assertAllClose(np_norm, tf_norm_val, rtol=1e-5, atol=1e-5)

  @test_util.run_v1_only("b/120545219")
  def Test(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the second skip condition reads the module-level variable named shape (set by
    # the loops in the script entry point), not the captured shape parameter; it only works
    # because tests are generated there, and the skip tests the global's rank, not this test's.
    # Generated test body. Skips unsupported combinations: the fro order for vector norms, a
    # non-integer real order for matrix norms, the euclidean order, and no axis on inputs above
    # rank two. Otherwise builds a random normal matrix of the captured shape and dtype (adding a
    # random imaginary part for complex dtypes; unseeded) and compares numpy's and TensorFlow's
    # norms. Runs only under TF1 behavior.
    is_matrix_norm = (isinstance(axis_, tuple) or
                      isinstance(axis_, list)) and len(axis_) == 2
    is_fancy_p_norm = np.isreal(ord_) and np.floor(ord_) != ord_
    if ((not is_matrix_norm and ord_ == "fro") or
        (is_matrix_norm and is_fancy_p_norm)):
      self.skipTest("Not supported by neither numpy.linalg.norm nor tf.norm")
    if ord_ == "euclidean" or (axis_ is None and len(shape) > 2):
      self.skipTest("Not supported by numpy.linalg.norm")
    matrix = np.random.randn(*shape_).astype(dtype_)
    if dtype_ in (np.complex64, np.complex128):
      matrix += 1j * np.random.randn(*shape_).astype(dtype_)
    _CompareNorm(self, matrix)

  return Test

# pylint: disable=redefined-builtin
if __name__ == "__main__":
  for use_static_shape in False, True:
    for dtype in np.float32, np.float64, np.complex64, np.complex128:
      for rows in 2, 5:
        for cols in 2, 5:
          for batch in [], [2], [2, 3]:
            shape = batch + [rows, cols]
            for ord in "euclidean", "fro", 0.5, 1, 2, np.inf:
              for axis in [
                  None, (-2, -1), (-1, -2), -len(shape), 0, len(shape) - 1
              ]:
                for keep_dims in False, True:
                  name = "%s_%s_ord_%s_axis_%s_%s_%s" % (
                      dtype.__name__, "_".join(map(str, shape)), ord, axis,
                      keep_dims, use_static_shape)
                  _AddTest(NormOpTest, "Norm_" + name,
                           _GetNormOpTest(dtype, shape, ord, axis, keep_dims,
                                          use_static_shape))

  test_lib.main()
