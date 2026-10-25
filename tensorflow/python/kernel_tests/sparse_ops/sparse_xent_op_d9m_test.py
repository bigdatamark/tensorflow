# [MODULE DOC]
# Tests of the deterministic (op determinism enabled) behavior of the sparse softmax cross entropy
# with logits op. One test class checks that the raw GPU kernel raises an unimplemented error under
# determinism. A second class inherits the shared base test cases of the regular op (so ordinary
# functionality is also verified on the deterministic path), adds checks that forward and backward
# results are bit-reproducible, and overrides tests whose expected invalid-label and scalar-label
# behavior differs because the deterministic path is built from a gather operation. Importing the
# module registers the op's gradient function. As a script, it enables op determinism globally
# before starting the TensorFlow test runner.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for deterministic functionality of SparseSoftmaxCrossEntropyWithLogits op."""

import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.kernel_tests.sparse_ops import sparse_xent_op_test_base
# The following import is required to register the gradient function.
from tensorflow.python.ops.nn_grad import _SparseSoftmaxCrossEntropyWithLogitsGrad  # pylint: disable=unused-import
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import test


class SparseXentOpDeterminismExceptionsTest(test.TestCase):
  # [CLASS DOC]
  # Test case verifying that calling the raw sparse softmax cross entropy kernel on GPU with op
  # determinism enabled raises an unimplemented error. Holds no state.
  """Test d9m-unimplemented exceptions from SparseSoftmaxXentWithLogitsOp.

  Test that tf.errors.UnimplementedError is thrown, as
  appropriate, by the GPU code-paths through SparseSoftmaxXentWithLogitsOp when
  deterministic ops are enabled.

  This test assumes that sparse_xent_op_test.py runs equivalent test cases
  when deterministic ops are not enabled and will therefore detect erroneous
  exception throwing in those cases.
  """

  @test_util.run_gpu_only
  @test_util.run_in_graph_and_eager_modes
  def testExceptionThrowing(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the expected message contains the misspelling "determinsim", which must match
    # the message produced by the kernel.
    # Scenario: GPU-only, graph and eager modes. For half and single precision features and 32-bit
    # and 64-bit integer labels, call the raw generated op on a 2x2 feature matrix with labels one
    # and zero, with GPU placement forced, then evaluate.
    # Expected: an unimplemented error saying the GPU implementation is not deterministic and that
    # the Python API uses an alternative deterministic path. Relies on determinism being enabled
    # by the script entry point.
    with self.session(), test_util.force_gpu():
      for features_dtype in [dtypes.float16, dtypes.float32]:
        for labels_dtype in [dtypes.int32, dtypes.int64]:
          features = constant_op.constant([[0.3, 0.5], [0.2, 0.6]],
                                          dtype=features_dtype)
          labels = constant_op.constant([1, 0], dtype=labels_dtype)
          with self.assertRaisesRegex(
              errors_impl.UnimplementedError,
              "The GPU implementation of SparseSoftmaxCrossEntropyWithLogits " +
              "that would have been executed is not deterministic. Note that " +
              "the Python API uses an alternative, deterministic, " +
              "GPU-accelerated path when determinsim is enabled."):
            result = gen_nn_ops.sparse_softmax_cross_entropy_with_logits(
                features=features, labels=labels)
            self.evaluate(result)


class SparseXentOpDeterministicTest(
    sparse_xent_op_test_base.SparseXentOpTestBase):
  # [CLASS DOC]
  # Test case for reproducibility of the sparse softmax cross entropy with logits function. It
  # inherits all test cases of the shared base class, so they run with the deterministic code path
  # selected, and adds reproducibility tests plus overrides for cases whose error behavior differs
  # on the deterministic path. Holds no state beyond the base class fixtures.
  """Test that SparseSoftmaxCrossEntropyWithLogits operates reproducibly.

  Inheriting from sparse_xent_op_test_base.SparseXentOpTestBase ensures that
  regular op functionality is correct when the deterministic code-path is
  selected.

  Note that because nn_ops.sparse_softmax_cross_entropy_with_logits_v2 calls
  nn_ops.sparse_softmax_cross_entropy_with_logits directly, the focus of
  testing is on the former in order to test both.
  """

  def _randomInts(self, shape, high, dtype):
    # [FUNCTION DOC]
    # HELPER: creates a constant tensor of random integers. Inputs: shape, the output shape;
    # high, exclusive upper bound (lower bound is zero); dtype, numpy integer type. Returns the
    # tensor. Uses numpy's global random state.
    return constant_op.constant(
        np.random.randint(low=0, high=high, size=shape).astype(dtype))

  def _randomFloats(self, shape, dtype):
    # [FUNCTION DOC]
    # HELPER: creates a constant tensor of random floating values uniformly drawn between minus
    # one and one. Inputs: shape, the output shape; dtype, numpy floating type. Returns the
    # tensor. Uses numpy's global random state.
    return constant_op.constant(
        (2 * np.random.random_sample(shape) - 1).astype(dtype))

  def _generateInputs(self, labels_dtype, logits_dtype, seed):
    # [FUNCTION DOC]
    # HELPER: generates a reproducible labels and logits pair. Inputs: labels_dtype and
    # logits_dtype, numpy types; seed, used to seed numpy randomness (a global side effect).
    # Uses a batch of 1024 and 1000 classes; the label shape is the plain batch size rather than
    # a one-element tuple, which still works. Returns the labels tensor (random class numbers) and
    # the logits tensor (batch by classes).
    batch_size = 1024
    classes_count = 1000
    np.random.seed(seed)
    labels_shape = (batch_size)
    labels = self._randomInts(
        labels_shape, high=classes_count, dtype=labels_dtype)
    logits_shape = (batch_size, classes_count)
    logits = self._randomFloats(logits_shape, logits_dtype)
    return labels, logits

  @test_util.run_in_graph_and_eager_modes
  def testForward(self):
    # [FUNCTION DOC]
    # Scenario: for half, single, double and bfloat16 logits, 32-bit and 64-bit labels, and five
    # seeds each, generate inputs and compute the version-2 sparse softmax cross entropy twice.
    # Graph and eager modes.
    # Expected: the two results are exactly equal (the forward pass is reproducible).
    with self.cached_session():
      for logits_dtype in [np.float16, np.float32, np.float64, \
          dtypes.bfloat16.as_numpy_dtype]:
        for labels_dtype in [np.int32, np.int64]:
          for trial in range(5):
            seed = 123 + trial
            labels, logits = self._generateInputs(
                labels_dtype, logits_dtype, seed=seed)
            result_a = nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
                labels=labels, logits=logits)
            result_b = nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
                labels=labels, logits=logits)
            self.assertAllEqual(result_a, result_b)

  @test_util.run_in_graph_and_eager_modes
  def testBackward(self):
    # [FUNCTION DOC]
    # Scenario: for each logits type (half, single, double, bfloat16) and label type, generate
    # inputs once, then for five seeds compute gradients of the loss with respect to the logits
    # twice, using random upstream gradients derived from the seed. Graph and eager modes.
    # Expected: the two gradient results for the same seed are exactly equal (the backward pass is
    # reproducible).
    with self.cached_session():
      for logits_dtype in [np.float16, np.float32, np.float64, \
          dtypes.bfloat16.as_numpy_dtype]:
        for labels_dtype in [np.int32, np.int64]:
          labels, logits = self._generateInputs(
              labels_dtype, logits_dtype, seed=456)
          output_shape = labels.shape[0]

          def gradients(seed):
            # [FUNCTION DOC]
            # HELPER: computes the gradient of the weighted loss with respect to the logits.
            # Input: seed, used to seed numpy randomness (global side effect) before drawing
            # random upstream gradients with the shape of the loss. Records the loss under a
            # persistent gradient tape watching the logits, multiplies the loss by the upstream
            # values, and returns the gradient tensor of that product with respect to the logits.
            np.random.seed(seed)
            upstream_gradients = self._randomFloats(output_shape, logits_dtype)
            with backprop.GradientTape(persistent=True) as tape:
              tape.watch(logits)
              op_output = nn_ops.sparse_softmax_cross_entropy_with_logits_v2(
                  labels=labels, logits=logits)
              gradient_injector_output = op_output * upstream_gradients
            return tape.gradient(gradient_injector_output, logits)

          for trial in range(5):
            seed = 456 + trial
            result_a = gradients(seed=seed)
            result_b = gradients(seed=seed)
            self.assertAllEqual(result_a, result_b)

  # Modifications to the parent class
  # (sparse_xent_op_test_base.SparseXentOpTestBase) follow

  def testInvalidLabelGPU(self):
    # [FUNCTION DOC]
    # Override of the base class test. Runs the shared invalid-label GPU check with the expected
    # gradient for invalid labels set to zero: on the deterministic path the forward result is
    # not-a-number for invalid labels but the backward gradient is zero (the regular kernel gives
    # not-a-number in both directions).
    """Modified test for invalid labels on GPU.

    When running on GPU, the pre-existing, nondeterministic implementation
    produces NaN (in both the forward and backward directions) for results
    associated with invalid labels (less than zero or greater than the number of
    classes minus one). However, while the deterministic implementation also
    produces NaN in the forward direction, it produces zeros in the backward
    direction.
    """
    self._testInvalidLabelGPU(invalid_label_gradient=0.0)

  def testInvalidLabelCPU(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the expected message pattern is a normal string with backslash-bracket
    # sequences, which are invalid string escapes (should be a raw string).
    # Override of the base class test. Runs the shared invalid-label CPU check, expecting the
    # error message from the gather operation (index 0 equal to 4 is not in the range zero to 4)
    # instead of the custom invalid-label message of the regular kernel.
    """Modified test for invalid labels on CPU.

    When running on CPU, the pre-existing, nondeterministic implementation
    throws a custom exception when any of the label values are invalid (less
    than zero or greater than the number of classes minus one). However, in the
    deterministic implementation, tf.gather throws an exception instead.
    """
    self._testInvalidLabelCPU(
        expected_regex="indices\[0\] = 4 is not in \[0, 4\)")

  def testLabelsPlaceholderScalar(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the expected message pattern is a normal string with backslash-bracket
    # sequences, which are invalid string escapes (should be a raw string).
    # Override of the base class test. Runs the shared check for labels given through a placeholder
    # of unknown shape and fed a scalar, expecting the gather operation's error saying the batch
    # dimensions must be in the range zero to zero but got one, since the deterministic path has
    # no dedicated kernel that could detect this case.
    """Test exception-throwing for non-statically-shaped, zero-rank labels.

    The deterministic implementation cannot check for this case because it does
    not have a specific implementation of SparseSoftmaxXentWithLogitsOp.
    Instead tf.gather, which is used to create the deterministic implementation,
    throws an error.
    """
    self._testLabelsPlaceholderScalar(
        expected_error_message="Expected batch_dims in the range \[0, 0\], " +
        "but got 1")

  def testScalarHandling(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring is a copy of the one on the placeholder-scalar test and does not
    # describe this test's own scenario name; the expected pattern is also a normal string with
    # invalid backslash escapes (should be a raw string).
    # Override of the base class test. Runs the shared scalar-handling check, expecting the gather
    # operation's error saying the batch dimensions must be in the range zero to zero but got one
    # (followed by any text), instead of the regular kernel's scalar-labels error.
    """Test exception-throwing for non-statically-shaped, zero-rank labels.

    The deterministic implementation cannot check for this case because it does
    not have a specific implementation of SparseSoftmaxXentWithLogitsOp.
    Instead tf.gather, which is used to create the deterministic implementation,
    throws an error.
    """
    self._testScalarHandling(
        expected_regex="Expected batch_dims in the range \[0, 0\], but got 1.*")


if __name__ == "__main__":
  # TODO(reedwm): Merge this test with sparse_xent_op_test.py.
  config.enable_op_determinism()
  test.main()
