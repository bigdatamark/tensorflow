# [MODULE DOC]
# Test module for the permutation linear operator in the linalg package (the operator that
# permutes rows according to a vector of integer indices). Defines one test class derived from
# the shared square linear-operator test base class, which builds an operator from a random
# shuffle of the index range and compares it with the explicitly constructed permutation matrix,
# plus tests for input validation and for the dense form of fixed 4 by 4 permutations. Defines a
# module-level alias for the tape-safety skip options helper from the shared test utilities (it
# is not used in this file). When run as a script it registers the generated base-class tests
# and starts the test runner.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.linalg import linalg as linalg_lib
from tensorflow.python.ops.linalg import linear_operator_permutation as permutation
from tensorflow.python.ops.linalg import linear_operator_test_util
from tensorflow.python.platform import test

linalg = linalg_lib
CheckTapeSafeSkipOptions = linear_operator_test_util.CheckTapeSafeSkipOptions


@test_util.run_all_in_graph_and_eager_modes
class LinearOperatorPermutationTest(
    linear_operator_test_util.SquareLinearOperatorDerivedClassTest):
  # [CLASS DOC]
  # Test case for the permutation linear operator. Inherits generic square-operator checks from
  # the shared base class, supplies the shapes to test, skips the Cholesky and eigenvalue tests
  # (permutation matrices are almost never positive definite) and provides the operator/matrix
  # factory. Adds validation tests and a dense-form test. Runs in graph and eager modes;
  # TensorFloat-32 is disabled per test and restored afterwards.
  """Most tests done in the base class LinearOperatorDerivedClassTest."""

  def tearDown(self):
    # [FUNCTION DOC]
    # Test fixture teardown: restores the global TensorFloat-32 execution setting saved in
    # setUp. Returns nothing.
    config.enable_tensor_float_32_execution(self.tf32_keep_)

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture setup: saves the current TensorFloat-32 setting on the instance and disables
    # TensorFloat-32 globally. Returns nothing.
    self.tf32_keep_ = config.tensor_float_32_execution_enabled()
    config.enable_tensor_float_32_execution(False)

  @staticmethod
  def operator_shapes_infos():
    # [FUNCTION DOC]
    # Hook for the base test class: returns the list of operator shape descriptions to test, all
    # square: 1 by 1, batched 1 by 3 by 3, batched 3 by 4 by 4 and 2 by 1 by 4 by 4.
    shape_info = linear_operator_test_util.OperatorShapesInfo
    return [
        shape_info((1, 1)),
        shape_info((1, 3, 3)),
        shape_info((3, 4, 4)),
        shape_info((2, 1, 4, 4))]

  @staticmethod
  def skip_these_tests():
    # [FUNCTION DOC]
    # Hook for the base test class: returns the names of generated tests to skip, here the
    # Cholesky and symmetric eigenvalue tests, because a permutation operator is almost never
    # positive definite.
    # This linear operator is almost never positive definite.
    return ["cholesky", "eigvalsh"]

  def operator_and_matrix(
      self, build_info, dtype, use_placeholder,
      ensure_self_adjoint_and_pd=False):
    # [FUNCTION DOC]
    # Factory hook used by the base class. Inputs: build_info (its shape is the full
    # batch-plus-matrix shape), dtype, use_placeholder (feed the permutation through a
    # placeholder of unknown shape) and ensure_self_adjoint_and_pd (accepted but ignored, even
    # though a random permutation is generally not self-adjoint or positive definite; that is why
    # the related tests are skipped). Creates the index range for the matrix size, repeats it for
    # every batch member and randomly shuffles it, then builds the permutation operator from the
    # shuffled indices and the dense reference matrix whose entry at row i, column j is one
    # exactly when column j equals the index chosen for row i, cast to the requested dtype.
    # Returns (operator, dense matrix). Note that the shuffle acts on the whole tensor along its
    # first axis, so for batched shapes it may shuffle across batch members rather than within
    # each permutation; the reference matrix is built from the same shuffled values, so the
    # comparison stays consistent either way.
    shape = list(build_info.shape)
    perm = math_ops.range(0, shape[-1])
    perm = array_ops.broadcast_to(perm, shape[:-1])
    perm = random_ops.random_shuffle(perm)

    if use_placeholder:
      perm = array_ops.placeholder_with_default(
          perm, shape=None)

    operator = permutation.LinearOperatorPermutation(
        perm, dtype=dtype)
    matrix = math_ops.cast(
        math_ops.equal(
            math_ops.range(0, shape[-1]),
            perm[..., array_ops.newaxis]),
        dtype)
    return operator, matrix

  def test_permutation_raises(self):
    # [FUNCTION DOC]
    # Test: invalid permutation arguments are rejected by the constructor. A scalar integer
    # raises a value error saying it must have at least 1 dimension; a list of floats raises a
    # type error saying it must have an integer dtype; a list with a negative entry (not a valid
    # set of unique indices) raises a value error saying it must be a vector of unique integers.
    perm = constant_op.constant(0, dtype=dtypes.int32)
    with self.assertRaisesRegex(ValueError, "must have at least 1 dimension"):
      permutation.LinearOperatorPermutation(perm)
    perm = [0., 1., 2.]
    with self.assertRaisesRegex(TypeError, "must be integer dtype"):
      permutation.LinearOperatorPermutation(perm)
    perm = [-1, 2, 3]
    with self.assertRaisesRegex(ValueError,
                                "must be a vector of unique integers"):
      permutation.LinearOperatorPermutation(perm)

  def test_to_dense_4x4(self):
    # [FUNCTION DOC]
    # Test: the dense form of the operator for three fixed permutations of four elements. The
    # identity permutation gives the 4 by 4 identity matrix; the permutation (1, 0, 3, 2) gives
    # the matrix swapping adjacent pairs; the permutation (3, 2, 0, 1) gives the matrix with ones
    # at the corresponding row/column positions. Each is asserted close to the hand-written
    # matrix.
    perm = [0, 1, 2, 3]
    self.assertAllClose(
        permutation.LinearOperatorPermutation(perm).to_dense(),
        linalg_ops.eye(4))
    perm = [1, 0, 3, 2]
    self.assertAllClose(
        permutation.LinearOperatorPermutation(perm).to_dense(),
        [[0., 1, 0, 0], [1., 0, 0, 0], [0., 0, 0, 1], [0., 0, 1, 0]])
    perm = [3, 2, 0, 1]
    self.assertAllClose(
        permutation.LinearOperatorPermutation(perm).to_dense(),
        [[0., 0, 0, 1], [0., 0, 1, 0], [1., 0, 0, 0], [0., 1, 0, 0]])


if __name__ == "__main__":
  linear_operator_test_util.add_tests(LinearOperatorPermutationTest)
  test.main()
