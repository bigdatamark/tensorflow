# [MODULE DOC]
# Unit tests for the CTC (connectionist temporal classification) loss ops in the ctc_ops module:
# the v1-style loss called through the v2 API, ctc_loss_v2, the dense-label variants, the
# GPU-oriented ctc_loss_v3, and internal helpers (collapse_repeated, dense_labels_to_sparse,
# ctc_unique_labels, state summing and state-to-output-label conversion, and the scan helper).
# Also checks determinism of forward and backward passes. Defines small helper functions plus
# four test case classes. No import-time side effects; running as a script starts the test runner.
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
"""Tests for tensorflow.ctc_ops.ctc_loss_op."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import ctc_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.platform import test


def SimpleSparseTensorFrom(x):
  # [FUNCTION DOC]
  # HELPER: builds label sparse tensors for the CTC loss tests.
  # Input x is a list of per-batch lists of integer labels. Returns a sparse tensor with
  # 64-bit indices (batch, time), 32-bit values and a dense shape of batch count by the longest
  # label length. Raises an error (from the maximum over empty indices) if x has no labels at all.
  # KNOWN ISSUE: the docstring says it returns the indices and values, but the function actually
  # returns one sparse tensor object built from indices, values and shape.
  """Create a very simple SparseTensor with dimensions (batch, time).

  Args:
    x: a list of lists of type int

  Returns:
    x_ix and x_val, the indices and values of the SparseTensor<2>.
  """
  x_ix = []
  x_val = []
  for batch_i, batch in enumerate(x):
    for time, val in enumerate(batch):
      x_ix.append([batch_i, time])
      x_val.append(val)
  x_shape = [len(x), np.asarray(x_ix).max(0)[1] + 1]
  x_ix = constant_op.constant(x_ix, dtypes.int64)
  x_val = constant_op.constant(x_val, dtypes.int32)
  x_shape = constant_op.constant(x_shape, dtypes.int64)

  return sparse_tensor.SparseTensor(x_ix, x_val, x_shape)


def _ctc_loss_v2(labels, inputs, sequence_length,
                 preprocess_collapse_repeated=False,
                 ctc_merge_repeated=True,
                 ignore_longer_outputs_than_inputs=False,
                 time_major=True):
  # [FUNCTION DOC]
  # HELPER: adapter letting tests written for the old v1 loss signature call the v2 loss.
  # Inputs: labels (sparse label tensor), inputs (logits), sequence_length (per-batch logit
  # lengths), the three legacy flags preprocess_collapse_repeated, ctc_merge_repeated and
  # ignore_longer_outputs_than_inputs, and time_major (whether logits are time by batch).
  # The legacy flags are only asserted to hold their defaults (false, true, false), so a
  # different value fails an assertion. Calls the v2 loss with no label length and blank index
  # of minus one, and returns the per-batch loss tensor.
  """Call ctc_loss_v2 with v1 args."""
  assert not preprocess_collapse_repeated
  assert ctc_merge_repeated
  assert not ignore_longer_outputs_than_inputs
  return ctc_ops.ctc_loss_v2(
      labels=labels,
      logits=inputs,
      logit_length=sequence_length,
      label_length=None,
      blank_index=-1,
      logits_time_major=time_major)


class CTCLossTest(test.TestCase):
  # [CLASS DOC]
  # Tests the CTC loss and its gradient (via the v1-argument adapter around the v2 loss) against
  # reference values, plus time-major handling, second-gradient rejection and empty batches.
  # Shares one helper that computes loss and gradient and compares with truth.

  def _testCTCLoss(self,
                   inputs,
                   seq_lens,
                   labels,
                   loss_truth,
                   grad_truth,
                   expected_err_re=None):
    # [FUNCTION DOC]
    # HELPER: shared checker for the CTC loss tests in this class.
    # Inputs: inputs (time by batch by depth log-probability array), seq_lens (per-batch
    # lengths), labels (sparse label tensor), loss_truth and grad_truth (expected loss and
    # gradient arrays), expected_err_re (optional regular expression of an expected op error).
    # Asserts the input and expected gradient have equal leading size, computes the loss through
    # the v2 adapter and its gradient with respect to the inputs in a CPU-only session, asserts
    # static shapes match the truth, then either asserts values match within 1e-6 or, if an
    # error pattern was given, asserts evaluation raises a matching op error. Returns nothing.
    self.assertEqual(len(inputs), len(grad_truth))

    inputs_t = constant_op.constant(inputs)

    with self.cached_session(use_gpu=False) as sess:
      loss = _ctc_loss_v2(
          inputs=inputs_t, labels=labels, sequence_length=seq_lens)
      grad = gradients_impl.gradients(loss, [inputs_t])[0]

      self.assertShapeEqual(loss_truth, loss)
      self.assertShapeEqual(grad_truth, grad)

      if expected_err_re is None:
        (tf_loss, tf_grad) = self.evaluate([loss, grad])
        self.assertAllClose(tf_loss, loss_truth, atol=1e-6)
        self.assertAllClose(tf_grad, grad_truth, atol=1e-6)
      else:
        with self.assertRaisesOpError(expected_err_re):
          self.evaluate([loss, grad])

  @test_util.run_v1_only("b/120545219")
  def testBasic(self):
    # [FUNCTION DOC]
    # Scenario: two batch entries (label sequences of length 5 and 4, depth 6, 7 time steps of
    # which the last two are not-a-number padding beyond the sequence length of 5). Inputs are
    # log probabilities and truth values for loss and gradient come from a reference
    # implementation by Alex Graves (listed in the comments). Expects the loss and gradient to
    # match, with zero gradient on padded steps. Graph mode only (v1).
    """Test two batch entries."""
    # Input and ground truth from Alex Graves' implementation.
    #
    #### Batch entry 0 #####
    # targets: 0 1 2 1 0
    # outputs:
    # 0 0.633766 0.221185 0.0917319 0.0129757 0.0142857 0.0260553
    # 1 0.111121 0.588392 0.278779 0.0055756 0.00569609 0.010436
    # 2 0.0357786 0.633813 0.321418 0.00249248 0.00272882 0.0037688
    # 3 0.0663296 0.643849 0.280111 0.00283995 0.0035545 0.00331533
    # 4 0.458235 0.396634 0.123377 0.00648837 0.00903441 0.00623107
    # alpha:
    # 0 -3.64753 -0.456075 -inf -inf -inf -inf -inf -inf -inf -inf -inf
    # 1 -inf -inf -inf -0.986437 -inf -inf -inf -inf -inf -inf -inf
    # 2 -inf -inf -inf -inf -inf -2.12145 -inf -inf -inf -inf -inf
    # 3 -inf -inf -inf -inf -inf -inf -inf -2.56174 -inf -inf -inf
    # 4 -inf -inf -inf -inf -inf -inf -inf -inf -inf -3.34211 -inf
    # beta:
    # 0 -inf -2.88604 -inf -inf -inf -inf -inf -inf -inf -inf -inf
    # 1 -inf -inf -inf -2.35568 -inf -inf -inf -inf -inf -inf -inf
    # 2 -inf -inf -inf -inf -inf -1.22066 -inf -inf -inf -inf -inf
    # 3 -inf -inf -inf -inf -inf -inf -inf -0.780373 -inf -inf -inf
    # 4 -inf -inf -inf -inf -inf -inf -inf -inf -inf 0 0
    # prob: -3.34211
    # outputDerivs:
    # 0 -0.366234 0.221185 0.0917319 0.0129757 0.0142857 0.0260553
    # 1 0.111121 -0.411608 0.278779 0.0055756 0.00569609 0.010436
    # 2 0.0357786 0.633813 -0.678582 0.00249248 0.00272882 0.0037688
    # 3 0.0663296 -0.356151 0.280111 0.00283995 0.0035545 0.00331533
    # 4 -0.541765 0.396634 0.123377 0.00648837 0.00903441 0.00623107
    #
    #### Batch entry 1 #####
    #
    # targets: 0 1 1 0
    # outputs:
    # 0 0.30176 0.28562 0.0831517 0.0862751 0.0816851 0.161508
    # 1 0.24082 0.397533 0.0557226 0.0546814 0.0557528 0.19549
    # 2 0.230246 0.450868 0.0389607 0.038309 0.0391602 0.202456
    # 3 0.280884 0.429522 0.0326593 0.0339046 0.0326856 0.190345
    # 4 0.423286 0.315517 0.0338439 0.0393744 0.0339315 0.154046
    # alpha:
    # 0 -1.8232 -1.19812 -inf -inf -inf -inf -inf -inf -inf
    # 1 -inf -2.19315 -2.83037 -2.1206 -inf -inf -inf -inf -inf
    # 2 -inf -inf -inf -2.03268 -3.71783 -inf -inf -inf -inf
    # 3 -inf -inf -inf -inf -inf -4.56292 -inf -inf -inf
    # 4 -inf -inf -inf -inf -inf -inf -inf -5.42262 -inf
    # beta:
    # 0 -inf -4.2245 -inf -inf -inf -inf -inf -inf -inf
    # 1 -inf -inf -inf -3.30202 -inf -inf -inf -inf -inf
    # 2 -inf -inf -inf -inf -1.70479 -0.856738 -inf -inf -inf
    # 3 -inf -inf -inf -inf -inf -0.859706 -0.859706 -0.549337 -inf
    # 4 -inf -inf -inf -inf -inf -inf -inf 0 0
    # prob: -5.42262
    # outputDerivs:
    # 0 -0.69824 0.28562 0.0831517 0.0862751 0.0816851 0.161508
    # 1 0.24082 -0.602467 0.0557226 0.0546814 0.0557528 0.19549
    # 2 0.230246 0.450868 0.0389607 0.038309 0.0391602 -0.797544
    # 3 0.280884 -0.570478 0.0326593 0.0339046 0.0326856 0.190345
    # 4 -0.576714 0.315517 0.0338439 0.0393744 0.0339315 0.154046

    # max_time_steps == 7
    depth = 6

    # seq_len_0 == 5
    targets_0 = [0, 1, 2, 1, 0]
    loss_log_prob_0 = -3.34211
    # dimensions are time x depth
    input_prob_matrix_0 = np.asarray(
        [[0.633766, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
         [0.111121, 0.588392, 0.278779, 0.0055756, 0.00569609, 0.010436],
         [0.0357786, 0.633813, 0.321418, 0.00249248, 0.00272882, 0.0037688],
         [0.0663296, 0.643849, 0.280111, 0.00283995, 0.0035545, 0.00331533],
         [0.458235, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107]],
        dtype=np.float32)
    input_log_prob_matrix_0 = np.log(input_prob_matrix_0)
    gradient_log_prob_0 = np.asarray(
        [[-0.366234, 0.221185, 0.0917319, 0.0129757, 0.0142857, 0.0260553],
         [0.111121, -0.411608, 0.278779, 0.0055756, 0.00569609, 0.010436],
         [0.0357786, 0.633813, -0.678582, 0.00249248, 0.00272882, 0.0037688],
         [0.0663296, -0.356151, 0.280111, 0.00283995, 0.0035545, 0.00331533],
         [-0.541765, 0.396634, 0.123377, 0.00648837, 0.00903441, 0.00623107]],
        dtype=np.float32)

    # seq_len_1 == 5
    targets_1 = [0, 1, 1, 0]
    loss_log_prob_1 = -5.42262
    # dimensions are time x depth

    input_prob_matrix_1 = np.asarray(
        [[0.30176, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508],
         [0.24082, 0.397533, 0.0557226, 0.0546814, 0.0557528, 0.19549],
         [0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, 0.202456],
         [0.280884, 0.429522, 0.0326593, 0.0339046, 0.0326856, 0.190345],
         [0.423286, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046]],
        dtype=np.float32)
    input_log_prob_matrix_1 = np.log(input_prob_matrix_1)
    gradient_log_prob_1 = np.asarray(
        [[-0.69824, 0.28562, 0.0831517, 0.0862751, 0.0816851, 0.161508],
         [0.24082, -0.602467, 0.0557226, 0.0546814, 0.0557528, 0.19549],
         [0.230246, 0.450868, 0.0389607, 0.038309, 0.0391602, -0.797544],
         [0.280884, -0.570478, 0.0326593, 0.0339046, 0.0326856, 0.190345],
         [-0.576714, 0.315517, 0.0338439, 0.0393744, 0.0339315, 0.154046]],
        dtype=np.float32)

    # len max_time_steps array of 2 x depth matrices
    inputs = [
        np.vstack(
            [input_log_prob_matrix_0[t, :], input_log_prob_matrix_1[t, :]])
        for t in range(5)
    ] + 2 * [np.nan * np.ones((2, depth), np.float32)]

    # convert inputs into [max_time x batch_size x depth tensor] Tensor
    inputs = np.asarray(inputs, dtype=np.float32)

    # len batch_size array of label vectors
    labels = SimpleSparseTensorFrom([targets_0, targets_1])

    # batch_size length vector of sequence_lengths
    seq_lens = np.array([5, 5], dtype=np.int32)

    # output: batch_size length vector of negative log probabilities
    loss_truth = np.array([-loss_log_prob_0, -loss_log_prob_1], np.float32)

    # output: len max_time_steps array of 2 x depth matrices
    grad_truth = [
        np.vstack([gradient_log_prob_0[t, :], gradient_log_prob_1[t, :]])
        for t in range(5)
    ] + 2 * [np.zeros((2, depth), np.float32)]

    # convert grad_truth into [max_time x batch_size x depth] Tensor
    grad_truth = np.asarray(grad_truth, dtype=np.float32)

    self._testCTCLoss(inputs, seq_lens, labels, loss_truth, grad_truth)

  def test_time_major(self):
    # [FUNCTION DOC]
    # Scenario: random small logits computed once in time-major layout and once transposed to
    # batch-major with the time-major flag turned off. Expects both losses to be exactly equal.
    """Testing time_major param.


    testing if transposing and setting time_major=False will result in the same
    loss
    """
    # [max_time x batch_size x depth tensor]
    inputs = np.random.randn(2, 2, 3).astype(np.float32)
    labels = SimpleSparseTensorFrom([[0, 1], [1, 0]])
    seq_lens = np.array([2, 2], dtype=np.int32)

    inputs_t = constant_op.constant(inputs)

    # Transposing tensor to [batch_size x max_time x depth tensor]
    inputs_t_transposed = constant_op.constant(inputs.transpose(1, 0, 2))

    with self.session(use_gpu=False) as sess:
      loss = _ctc_loss_v2(
          inputs=inputs_t, labels=labels, sequence_length=seq_lens)
      loss_transposed = _ctc_loss_v2(
          inputs=inputs_t_transposed,
          labels=labels,
          sequence_length=seq_lens,
          time_major=False)

      (tf_loss, tf_loss_transposed) = self.evaluate([loss, loss_transposed])
      self.assertAllEqual(tf_loss, tf_loss_transposed)

  @test_util.run_v1_only("b/120545219")
  def testInvalidSecondGradient(self):
    # [FUNCTION DOC]
    # Scenario: computes the loss on random logits, then requests a Hessian-vector product (a
    # second derivative). Expects a lookup error mentioning that the gradient is explicitly
    # disabled, because the CTC loss gradient does not support a second derivative. Graph mode only.
    inputs = np.random.randn(2, 2, 3).astype(np.float32)
    inputs_t = constant_op.constant(inputs)
    labels = SimpleSparseTensorFrom([[0, 1], [1, 0]])
    seq_lens = np.array([2, 2], dtype=np.int32)
    v = [1.0]

    with self.session(use_gpu=False):
      loss = _ctc_loss_v2(
          inputs=inputs_t, labels=labels, sequence_length=seq_lens)
      # Taking this second gradient should fail, since it is not
      # yet supported.
      with self.assertRaisesRegex(LookupError, "explicitly disabled"):
        _ = gradients_impl._hessian_vector_product(loss, [inputs_t], v)

  @test_util.run_v1_only("b/120545219")
  def testEmptyBatch(self):
    # [FUNCTION DOC]
    # Scenario: logits, sequence lengths and sparse labels all with zero batch entries.
    # Expects running the loss to raise an invalid-argument error saying the batch size must
    # not be zero. Graph mode only.
    inputs = constant_op.constant([], dtype=dtypes.float32, shape=(1, 0, 2))
    sequence_lengths = constant_op.constant([], dtype=dtypes.int32)
    labels = sparse_tensor.SparseTensor(
        indices=constant_op.constant([], shape=(0, 2), dtype=dtypes.int64),
        values=constant_op.constant([], shape=(0,), dtype=dtypes.int32),
        dense_shape=[5, 5])

    with self.session(use_gpu=False) as sess:
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "batch_size must not be 0"):
        sess.run(_ctc_loss_v2(labels, inputs, sequence_lengths))


class CTCLossTestV2(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests the v2 CTC loss API and the dense-label loss variant against the older loss op, plus
  # utility ops: repeated-label collapsing, dense to sparse label conversion, unique label
  # extraction, state summing, state to output-label conversion and the scan helper. Supports
  # parameterized tests.

  @test_util.run_in_graph_and_eager_modes
  def testCtcLossV2(self):
    # [FUNCTION DOC]
    # Scenario: random dense labels and logits with a fixed seed (batch 8, 6 labels, 12 frames).
    # Computes a reference loss and gradient with dense labels, then the loss with the same
    # labels converted to sparse form and blank index 0. In graph mode asserts loss and gradient
    # agree with the reference (tolerance about 2e-6); in eager mode the comparison is skipped.
    # Runs in both graph and eager modes.
    random_seed.set_random_seed(5)

    batch_size = 8
    num_labels = 6
    max_label_length = 5
    num_frames = 12

    labels = random_ops.random_uniform(
        [batch_size, max_label_length], minval=1, maxval=num_labels,
        dtype=dtypes.int64)
    logits = random_ops.random_uniform([num_frames, batch_size, num_labels])

    label_length = random_ops.random_uniform(
        [batch_size], minval=2, maxval=max_label_length, dtype=dtypes.int64)
    label_mask = array_ops.sequence_mask(
        label_length, maxlen=max_label_length, dtype=label_length.dtype)
    labels *= label_mask
    logit_length = [num_frames] * batch_size

    with backprop.GradientTape() as t:
      t.watch(logits)
      ref_loss = ctc_ops.ctc_loss_v2(
          labels=labels,
          logits=logits,
          label_length=label_length,
          logit_length=logit_length)
    ref_grad = t.gradient(ref_loss, [logits])

    sparse_labels = ctc_ops.dense_labels_to_sparse(labels, label_length)

    def assert_same_loss_and_grads(loss):
      # [FUNCTION DOC]
      # HELPER: nested comparison used by the enclosing test.
      # Takes a loss tensor. Does nothing in eager mode. In graph mode evaluates it against the
      # reference loss and compares its gradient with respect to the logits with the reference
      # gradient (relative and absolute tolerance 2e-6). Returns nothing.
      if context.executing_eagerly():
        return
      with self.cached_session():
        self.assertAllClose(*self.evaluate([loss, ref_loss]))
        grad = gradients_impl.gradients(loss, [logits])
        self.assertAllClose(
            *self.evaluate([grad, ref_grad]), rtol=2e-06, atol=2e-06)

    assert_same_loss_and_grads(
        ctc_ops.ctc_loss_v2(
            labels=sparse_labels,
            logits=logits,
            label_length=label_length,
            logit_length=logit_length,
            blank_index=0))

  @test_util.run_v1_only("b/120545219")
  def testCtcLossDenseIsSameAsCtcLoss(self):
    # [FUNCTION DOC]
    # Scenario: random dense labels (blank class 0), variable logit lengths, on GPU if available
    # else CPU. Computes the dense CTC loss and gradient, and the classic loss on labels shifted
    # down by one with the blank column moved to the end. Repeats 32 times, asserting loss and
    # gradient agree (tolerance 4e-6 on gradients). Graph mode only.
    with ops.device("/GPU:0" if test.is_gpu_available() else "/CPU:0"):
      random_seed.set_random_seed(5)

      batch_size = 8
      num_labels = 6
      label_length = 5
      minimum_logits_length = 10
      num_frames = minimum_logits_length + batch_size
      logits = random_ops.random_uniform([num_frames, batch_size, num_labels])
      labels = random_ops.random_uniform(
          [batch_size, label_length], minval=1, maxval=num_labels,
          dtype=dtypes.int64)

      label_lengths = random_ops.random_uniform(
          [batch_size], minval=2, maxval=label_length, dtype=dtypes.int64)
      label_mask = array_ops.sequence_mask(
          label_lengths, maxlen=label_length, dtype=label_lengths.dtype)
      labels *= label_mask

      logit_lengths = math_ops.range(batch_size) + minimum_logits_length

      ctc_loss = ctc_ops.ctc_loss_dense(
          labels=labels,
          logits=logits,
          label_length=label_lengths,
          logit_length=logit_lengths)
      ctc_loss_grads = gradients_impl.gradients(ctc_loss, [logits])[0]

      # Shift labels down by one (move blank from 0 to num_labels -1)
      tf_ctc_loss_labels = math_ops.cast(labels, dtypes.int32) - 1
      tf_nn_ctc_logits = array_ops.concat([
          logits[:, :, 1:],
          logits[:, :, 0:1],
      ], axis=2)

      tf_ctc_loss_labels = ctc_ops.dense_labels_to_sparse(
          tf_ctc_loss_labels, label_lengths)

      tf_nn_ctc_loss = ctc_ops.ctc_loss(
          labels=tf_ctc_loss_labels,
          inputs=tf_nn_ctc_logits,
          sequence_length=logit_lengths,
          time_major=True)
      tf_nn_ctc_grads = gradients_impl.gradients(tf_nn_ctc_loss, [logits])[0]

      with self.cached_session() as sess:
        for _ in range(32):
          self.assertAllClose(*self.evaluate([ctc_loss, tf_nn_ctc_loss]))
          self.assertAllClose(
              *self.evaluate([ctc_loss_grads, tf_nn_ctc_grads]),
              rtol=4e-06,
              atol=4e-06)

  @test_util.run_v1_only("b/120545219")
  def testCtcLossDenseUniqueFastPathIsSameAsCtcLoss(self):
    # [FUNCTION DOC]
    # Scenario: like the dense-versus-classic comparison but passes precomputed unique labels to
    # the dense loss (the fast path) and uses a fixed logit length equal to the frame count.
    # Repeats 32 times, asserting loss and gradient match the classic loss (tolerance 2e-6).
    # Graph mode only.
    random_seed.set_random_seed(5)

    batch_size = 8
    num_labels = 6
    label_length = 5
    num_frames = 12
    logits = random_ops.random_uniform([num_frames, batch_size, num_labels])
    labels = random_ops.random_uniform(
        [batch_size, label_length], minval=1, maxval=num_labels,
        dtype=dtypes.int64)

    label_lengths = random_ops.random_uniform(
        [batch_size], minval=2, maxval=label_length, dtype=dtypes.int64)
    label_mask = array_ops.sequence_mask(
        label_lengths, maxlen=label_length, dtype=label_lengths.dtype)
    labels *= label_mask

    logit_lengths = [num_frames] * batch_size

    ctc_loss = ctc_ops.ctc_loss_dense(
        labels=labels,
        logits=logits,
        label_length=label_lengths,
        logit_length=logit_lengths,
        unique=ctc_ops.ctc_unique_labels(labels))
    ctc_loss_grads = gradients_impl.gradients(ctc_loss, [logits])[0]

    # Shift labels down by one (move blank from 0 to num_labels -1)
    tf_ctc_loss_labels = math_ops.cast(labels, dtypes.int32) - 1
    tf_nn_ctc_logits = array_ops.concat([
        logits[:, :, 1:],
        logits[:, :, 0:1],
    ], axis=2)

    tf_ctc_loss_labels = ctc_ops.dense_labels_to_sparse(
        tf_ctc_loss_labels, label_lengths)

    tf_nn_ctc_loss = ctc_ops.ctc_loss(
        labels=tf_ctc_loss_labels,
        inputs=tf_nn_ctc_logits,
        sequence_length=logit_lengths,
        time_major=True)
    tf_nn_ctc_grads = gradients_impl.gradients(tf_nn_ctc_loss, [logits])[0]

    with self.cached_session():
      for _ in range(32):
        self.assertAllClose(*self.evaluate([ctc_loss, tf_nn_ctc_loss]))
        self.assertAllClose(
            *self.evaluate([ctc_loss_grads, tf_nn_ctc_grads]),
            rtol=2e-06,
            atol=2e-06)

  @test_util.run_v1_only("b/120545219")
  def testCtcLossDenseUniqueFastPathWithBlankIndexIsSameAsCtcLoss(self):
    # [FUNCTION DOC]
    # Scenario: the unique-labels fast path of the dense loss with the blank class moved to
    # index 2 (logits and labels shifted accordingly), compared with the classic loss on the
    # unshifted data. Repeats 32 times, asserting loss and gradient match (tolerance 2e-6).
    # Graph mode only.
    random_seed.set_random_seed(5)

    batch_size = 8
    num_labels = 6
    label_length = 5
    num_frames = 12
    logits = random_ops.random_uniform([num_frames, batch_size, num_labels])
    labels = random_ops.random_uniform([batch_size, label_length],
                                       minval=0,
                                       maxval=num_labels - 1,
                                       dtype=dtypes.int64)

    label_lengths = random_ops.random_uniform([batch_size],
                                              minval=2,
                                              maxval=label_length,
                                              dtype=dtypes.int64)
    label_mask = array_ops.sequence_mask(
        label_lengths, maxlen=label_length, dtype=label_lengths.dtype)
    labels *= label_mask

    logit_lengths = [num_frames] * batch_size

    tf_ctc_loss_labels = math_ops.cast(labels, dtypes.int32)
    tf_ctc_loss_labels = ctc_ops.dense_labels_to_sparse(tf_ctc_loss_labels,
                                                        label_lengths)

    tf_nn_ctc_loss = ctc_ops.ctc_loss(
        labels=tf_ctc_loss_labels,
        inputs=logits,
        sequence_length=logit_lengths,
        time_major=True)
    tf_nn_ctc_grads = gradients_impl.gradients(tf_nn_ctc_loss, [logits])[0]

    # Shift the blank logits/labels to be somewhere in the middle.
    blank_index = 2
    shifted_logits = array_ops.concat([
        logits[:, :, :blank_index],
        logits[:, :, -1:],
        logits[:, :, blank_index:-1],
    ],
                                      axis=2)
    shifted_labels = array_ops.where_v2(labels < blank_index, labels,
                                        labels + 1)

    ctc_loss = ctc_ops.ctc_loss_dense(
        labels=shifted_labels,
        logits=shifted_logits,
        label_length=label_lengths,
        logit_length=logit_lengths,
        blank_index=blank_index,
        unique=ctc_ops.ctc_unique_labels(shifted_labels))
    ctc_loss_grads = gradients_impl.gradients(ctc_loss, [logits])[0]

    with self.cached_session() as sess:
      for _ in range(32):
        self.assertAllClose(*self.evaluate([ctc_loss, tf_nn_ctc_loss]))
        self.assertAllClose(
            *self.evaluate([ctc_loss_grads, tf_nn_ctc_grads]),
            rtol=2e-06,
            atol=2e-06)

  @test_util.run_v1_only("b/120545219")
  def testCtcLossDenseWithBlankIndexIsSameAsCtcLoss(self):
    # [FUNCTION DOC]
    # Scenario: the dense loss without the unique-labels fast path and with the blank class at
    # index 2 (logits and labels shifted), compared with the classic loss on unshifted data.
    # Repeats 32 times, asserting loss and gradient match (tolerance 2e-6). Graph mode only.
    random_seed.set_random_seed(5)

    batch_size = 8
    num_labels = 6
    label_length = 5
    num_frames = 12
    logits = random_ops.random_uniform([num_frames, batch_size, num_labels])
    labels = random_ops.random_uniform(
        [batch_size, label_length], minval=0, maxval=num_labels-1,
        dtype=dtypes.int64)

    label_lengths = random_ops.random_uniform(
        [batch_size], minval=2, maxval=label_length, dtype=dtypes.int64)
    label_mask = array_ops.sequence_mask(
        label_lengths, maxlen=label_length, dtype=label_lengths.dtype)
    labels *= label_mask

    logit_lengths = [num_frames] * batch_size

    tf_ctc_loss_labels = math_ops.cast(labels, dtypes.int32)
    tf_ctc_loss_labels = ctc_ops.dense_labels_to_sparse(
        tf_ctc_loss_labels, label_lengths)

    tf_nn_ctc_loss = ctc_ops.ctc_loss(
        labels=tf_ctc_loss_labels,
        inputs=logits,
        sequence_length=logit_lengths,
        time_major=True)
    tf_nn_ctc_grads = gradients_impl.gradients(tf_nn_ctc_loss, [logits])[0]

    # Shift the blank logits/labels to be somewhere in the middle.
    blank_index = 2
    shifted_logits = array_ops.concat([
        logits[:, :, :blank_index],
        logits[:, :, -1:],
        logits[:, :, blank_index:-1],
    ], axis=2)
    shifted_labels = array_ops.where_v2(labels < blank_index, labels,
                                        labels + 1)

    ctc_loss = ctc_ops.ctc_loss_dense(
        labels=shifted_labels,
        logits=shifted_logits,
        label_length=label_lengths,
        logit_length=logit_lengths,
        blank_index=blank_index)
    ctc_loss_grads = gradients_impl.gradients(ctc_loss, [logits])[0]

    with self.cached_session() as sess:
      for _ in range(32):
        self.assertAllClose(*self.evaluate([ctc_loss, tf_nn_ctc_loss]))
        self.assertAllClose(
            *self.evaluate([ctc_loss_grads, tf_nn_ctc_grads]),
            rtol=2e-06,
            atol=2e-06)

  @test_util.run_v1_only("b/120545219")
  def testCtcLossDenseWithNegativeBlankIndexIsSameAsCtcLoss(self):
    # [FUNCTION DOC]
    # Scenario: the dense loss with a blank index of minus one (last class) on GPU if available
    # else CPU, compared with the classic loss on the same logits and labels. Repeats 32 times,
    # asserting loss and gradient match (tolerance 2e-6). Graph mode only.
    with ops.device("/GPU:0" if test.is_gpu_available() else "/CPU:0"):
      random_seed.set_random_seed(5)

      batch_size = 8
      num_labels = 6
      label_length = 5
      num_frames = 12
      logits = random_ops.random_uniform([num_frames, batch_size, num_labels])
      labels = random_ops.random_uniform(
          [batch_size, label_length], minval=0, maxval=num_labels-1,
          dtype=dtypes.int64)

      label_lengths = random_ops.random_uniform(
          [batch_size], minval=2, maxval=label_length, dtype=dtypes.int64)
      label_mask = array_ops.sequence_mask(
          label_lengths, maxlen=label_length, dtype=label_lengths.dtype)
      labels *= label_mask

      logit_lengths = [num_frames] * batch_size

      ctc_loss = ctc_ops.ctc_loss_dense(
          labels=labels,
          logits=logits,
          label_length=label_lengths,
          logit_length=logit_lengths,
          blank_index=-1)
      ctc_loss_grads = gradients_impl.gradients(ctc_loss, [logits])[0]

      tf_ctc_loss_labels = math_ops.cast(labels, dtypes.int32)
      tf_ctc_loss_labels = ctc_ops.dense_labels_to_sparse(
          tf_ctc_loss_labels, label_lengths)

      tf_nn_ctc_loss = ctc_ops.ctc_loss(
          labels=tf_ctc_loss_labels,
          inputs=logits,
          sequence_length=logit_lengths,
          time_major=True)
      tf_nn_ctc_grads = gradients_impl.gradients(tf_nn_ctc_loss, [logits])[0]

      with self.cached_session() as sess:
        for _ in range(32):
          self.assertAllClose(*self.evaluate([ctc_loss, tf_nn_ctc_loss]))
          self.assertAllClose(
              *self.evaluate([ctc_loss_grads, tf_nn_ctc_grads]),
              rtol=2e-06,
              atol=2e-06)

  @parameterized.parameters((False, 0), (True, 0), (False, -1), (True, -1))
  def testCtcLossDenseWithUndefinedStaticDimensions(self, unique, blank_index):
    # [FUNCTION DOC]
    # Parameterized over whether unique labels are used and blank index 0 or minus one.
    # Scenario: traces a compiled function wrapping the dense loss with unknown batch size and
    # frame count, then calls the concrete function with real random data (batch 8, 12 frames).
    # Only asserts that tracing and execution succeed without error; no values are compared.
    random_seed.set_random_seed(5)

    # Trace without a batch size and number of frames
    batch_size = None
    num_labels = 6
    label_length = 5
    num_frames = None

    @def_function.function
    def func(labels, logits, label_lengths, logit_lengths):
      # [FUNCTION DOC]
      # HELPER: nested compiled function traced by the enclosing test.
      # Takes dense labels, logits, label lengths and logit lengths; optionally computes unique
      # labels (depending on the test parameter) and returns the dense CTC loss using the
      # test's blank index.
      unique_labels = ctc_ops.ctc_unique_labels(labels) if unique else None
      return ctc_ops.ctc_loss_dense(
          labels=labels,
          logits=logits,
          label_length=label_lengths,
          logit_length=logit_lengths,
          unique=unique_labels,
          blank_index=blank_index)

    labels_spec = tensor_spec.TensorSpec([batch_size, label_length],
                                         dtypes.int64)
    logits_spec = tensor_spec.TensorSpec([num_frames, batch_size, num_labels],
                                         dtypes.float32)
    label_lengths_spec = tensor_spec.TensorSpec([batch_size], dtypes.int64)
    logit_lengths_spec = tensor_spec.TensorSpec([batch_size], dtypes.int64)

    f = func.get_concrete_function(
        labels_spec, logits_spec, label_lengths_spec, logit_lengths_spec)

    # Execute with a defined batch size and number of frames
    batch_size = 8
    num_frames = 12

    logits = random_ops.random_uniform([num_frames, batch_size, num_labels])
    labels = random_ops.random_uniform(
        [batch_size, label_length], minval=1, maxval=num_labels,
        dtype=dtypes.int64)

    label_lengths = random_ops.random_uniform(
        [batch_size], minval=2, maxval=label_length, dtype=dtypes.int64)
    label_mask = array_ops.sequence_mask(
        label_lengths, maxlen=label_length, dtype=label_lengths.dtype)
    labels *= label_mask

    logit_lengths = constant_op.constant(
        [num_frames] * batch_size, dtype=dtypes.int64)

    f(labels, logits, label_lengths, logit_lengths)

  def testCollapseRepeated(self):
    # [FUNCTION DOC]
    # Scenario: three padded label rows with runs of repeated labels and sequence lengths 4, 5, 5.
    # Expects repeats collapsed, new lengths 2, 3, 4 and output rows padded with zeros.
    collapsed, new_seq_lengths = ctc_ops.collapse_repeated(
        labels=[[1, 3, 3, 3, 0],
                [1, 4, 4, 4, 0],
                [4, 2, 2, 9, 4]],
        seq_length=[4, 5, 5])
    self.assertAllEqual(new_seq_lengths, [2, 3, 4])
    self.assertAllEqual(
        collapsed,
        [[1, 3, 0, 0],
         [1, 4, 0, 0],
         [4, 2, 9, 4]])

  def testCollapseRepeatedPreservesDtypes(self):
    # [FUNCTION DOC]
    # Scenario: same data as the basic collapse test but with 64-bit integer labels and lengths.
    # Expects both outputs to stay 64-bit and have the same collapsed values and lengths.
    collapsed, new_seq_lengths = ctc_ops.collapse_repeated(
        labels=constant_op.constant(
            [[1, 3, 3, 3, 0],
             [1, 4, 4, 4, 0],
             [4, 2, 2, 9, 4]],
            dtype=dtypes.int64),
        seq_length=constant_op.constant([4, 5, 5], dtype=dtypes.int64))
    self.assertEqual(new_seq_lengths.dtype, dtypes.int64)
    self.assertEqual(collapsed.dtype, dtypes.int64)
    self.assertAllEqual(new_seq_lengths, [2, 3, 4])
    self.assertAllEqual(
        collapsed,
        [[1, 3, 0, 0],
         [1, 4, 0, 0],
         [4, 2, 9, 4]])

  def testCollapseRepeatedExtraPadding(self):
    # [FUNCTION DOC]
    # Scenario: label rows with extra trailing content beyond the sequence length (including
    # non-zero values). Expects that content to be ignored: same collapsed result and lengths as
    # the basic case.
    collapsed, new_seq_lengths = ctc_ops.collapse_repeated(
        labels=[[1, 3, 3, 3, 0, 0, 0],
                [1, 4, 4, 4, 0, 1, 2],
                [4, 2, 2, 9, 4, 0, 0]],
        seq_length=[4, 5, 5])
    self.assertAllEqual(new_seq_lengths, [2, 3, 4])
    self.assertAllEqual(
        collapsed,
        [[1, 3, 0, 0],
         [1, 4, 0, 0],
         [4, 2, 9, 4]])

  def testCollapseRepeatedFrontRepeats(self):
    # [FUNCTION DOC]
    # Scenario: rows starting with repeated labels and sequence lengths 5, 4, 3. Expects
    # collapsed lengths 2, 2, 1 and zero-padded rows, checking runs at the start are handled.
    collapsed, new_seq_lengths = ctc_ops.collapse_repeated(
        labels=[[1, 1, 1, 2, 2],
                [1, 1, 1, 2, 2],
                [1, 1, 1, 2, 2]],
        seq_length=[5, 4, 3])
    self.assertAllEqual(new_seq_lengths, [2, 2, 1])
    self.assertAllEqual(
        collapsed,
        [[1, 2],
         [1, 2],
         [1, 0]])

  def testCollapseRepeatedAllLabelsTheSame(self):
    # [FUNCTION DOC]
    # Scenario: every row contains only one repeated label with lengths 4, 5, 1. Expects every
    # row to collapse to a single label, lengths 1, 1, 1, and an output of width one.
    collapsed, new_seq_lengths = ctc_ops.collapse_repeated(
        labels=[[1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1]],
        seq_length=[4, 5, 1])
    self.assertAllEqual(new_seq_lengths, [1, 1, 1])
    self.assertAllEqual(
        collapsed,
        [[1],
         [1],
         [1]])

  def testDenseSequencesToSparse(self):
    # [FUNCTION DOC]
    # Scenario: converts dense padded labels with lengths to sparse labels and back to dense.
    # Expects the round trip to reproduce the original, and that a version with extra padding
    # columns converts to the same dense result as the unpadded one.
    labels = [[1, 3, 3, 3, 0],
              [1, 4, 4, 4, 0],
              [4, 2, 2, 9, 4]]
    length = [4, 5, 5]
    sparse = ctc_ops.dense_labels_to_sparse(labels, length)
    new_dense = sparse_ops.sparse_tensor_to_dense(sparse)

    self.assertAllEqual(labels, new_dense)

    padded_labels = [[1, 3, 3, 3, 0, 0, 0, 0],
                     [1, 4, 4, 4, 0, 0, 0, 0],
                     [4, 2, 2, 9, 4, 0, 0, 0]]
    length = [4, 5, 5]
    sparse = ctc_ops.dense_labels_to_sparse(padded_labels, length)
    padded_dense = sparse_ops.sparse_tensor_to_dense(sparse)

    self.assertAllEqual(padded_dense, new_dense)

  def testUnique(self):
    # [FUNCTION DOC]
    # Scenario: two label rows with repeated values. Expects the unique-labels op to return, per
    # row, the distinct labels padded with zeros and an index matrix mapping each position to its
    # position in the unique list.
    labels = [
        [3, 4, 4, 3],
        [1, 1, 1, 0],
    ]
    unique, idx = ctc_ops.ctc_unique_labels(labels)
    self.assertAllEqual([
        [3, 4, 0, 0],
        [1, 0, 0, 0],
    ], unique)
    self.assertAllEqual([
        [0, 1, 1, 0],
        [0, 0, 0, 1],
    ], idx)

  def testSumStates(self):
    # [FUNCTION DOC]
    # Scenario: log-space states for two frames and two batch entries with an index matrix
    # grouping states that share a label. Expects the internal state-summing helper, after
    # exponentiation, to give per-group sums of the original values, padded with zeros.
    idx = [
        [0, 1, 0, 1],
        [0, 0, 0, 1],
    ]
    states = math_ops.log([
        [[1.0, 2.0, 3.0, 4.0],
         [5.0, 6.0, 7.0, 8.0]],
        [[0.1, 0.2, 0.3, 0.4],
         [0.5, 0.6, 0.7, 0.8]],
    ])
    sum_of_states = math_ops.exp(ctc_ops._sum_states(idx, states))
    self.assertAllClose([
        [[4.0, 6.0, 0.0, 0.0],
         [18.0, 8.0, 0.0, 0.0]],
        [[0.4, 0.6, 0.0, 0.0],
         [1.8, 0.8, 0.0, 0.0]]
    ], sum_of_states)

  def testStateToOlabel(self):
    # [FUNCTION DOC]
    # Scenario: three frames, two batch entries, ten states each (label states then blank
    # states), 8 output labels. Expects the internal state to output-label helper, after
    # exponentiation, to give the blank output as the sum of blank states and each label output
    # as the sum of the states carrying that label (the label column is shown without the blank).
    labels = [
        [3, 4, 3, 4],
        [1, 1, 1, 0],
    ]
    num_labels = 8

    # 3 frames, 2 batch, 10 states (5 label, 5 blank).
    states = [
        [[0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20],
         [0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30]],
        [[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
         [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]],
        [[11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
         [21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0]],
    ]
    labels = ops.convert_to_tensor(labels)
    states = math_ops.log(states)
    olabel = ctc_ops._state_to_olabel(labels, num_labels, states)
    olabel = math_ops.exp(olabel)
    blank = olabel[:, :, 0]
    self.assertAllClose(blank, [
        [0.16 + 0.17 + 0.18 + 0.19 + 0.20,
         0.26 + 0.27 + 0.28 + 0.29 + 0.30],
        [1.6 + 1.7 + 1.8 + 1.9 + 2.0,
         2.6 + 2.7 + 2.8 + 2.9 + 3.0],
        [16.0 + 17.0 + 18.0 + 19.0 + 20.0,
         26.0 + 27.0 + 28.0 + 29.0 + 30.0]
    ])
    self.assertAllClose(olabel[:, :, 1:], [
        [[0.0, 0.0, 0.12 + 0.14, 0.13 + 0.15, 0.0, 0.0, 0.0],
         [0.22 + 0.23 + 0.24, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 1.2 + 1.4, 1.3 + 1.5, 0.0, 0.0, 0.0],
         [2.2 + 2.3 + 2.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 12.0 + 14.0, 13.0 + 15.0, 0.0, 0.0, 0.0],
         [22.0 + 23.0 + 24.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    ])

  def testStateToOlabelUnique(self):
    # [FUNCTION DOC]
    # Scenario: same data and expectations as the plain state to output-label test, but using
    # the variant that takes precomputed unique labels. Expects identical blank and label sums.
    labels = [
        [3, 4, 3, 4],
        [1, 1, 1, 0],
    ]
    num_labels = 8

    # 3 frames, 2 batch, 10 states (5 label, 5 blank).
    states = [
        [[0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20],
         [0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.30]],
        [[1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0],
         [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0]],
        [[11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0],
         [21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0]],
    ]
    labels = ops.convert_to_tensor(labels)
    states = math_ops.log(states)
    olabel = ctc_ops._state_to_olabel_unique(
        labels, num_labels, states, ctc_ops.ctc_unique_labels(labels))
    olabel = math_ops.exp(olabel)
    blank = olabel[:, :, 0]
    self.assertAllClose(blank, [
        [0.16 + 0.17 + 0.18 + 0.19 + 0.20,
         0.26 + 0.27 + 0.28 + 0.29 + 0.30],
        [1.6 + 1.7 + 1.8 + 1.9 + 2.0,
         2.6 + 2.7 + 2.8 + 2.9 + 3.0],
        [16.0 + 17.0 + 18.0 + 19.0 + 20.0,
         26.0 + 27.0 + 28.0 + 29.0 + 30.0]])
    self.assertAllClose(olabel[:, :, 1:], [
        [[0.0, 0.0, 0.12 + 0.14, 0.13 + 0.15, 0.0, 0.0, 0.0],
         [0.22 + 0.23 + 0.24, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 1.2 + 1.4, 1.3 + 1.5, 0.0, 0.0, 0.0],
         [2.2 + 2.3 + 2.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        [[0.0, 0.0, 12.0 + 14.0, 13.0 + 15.0, 0.0, 0.0, 0.0],
         [22.0 + 23.0 + 24.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
    ])

  def testStateToOlabelUniqueSinglePath(self):
    # [FUNCTION DOC]
    # Scenario: label lengths equal frame lengths so only one valid path exists per sequence,
    # states are one-hot (a zero row for a sequence that has ended). Expects the unique-label
    # conversion to give zero blank probability and the one-hot label outputs for the single path.
    labels = [
        [3, 4, 3],
        [1, 0, 0],
    ]
    num_labels = 8

    # 3 frames, 2 batch, 8 states (4 label, 4 blank).
    #
    # There is only single valid path for each sequence because the frame
    # lengths and the label lengths are the same.
    states = [[[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
              [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
              [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
    labels = ops.convert_to_tensor(labels)
    states = math_ops.log(states)
    olabel = ctc_ops._state_to_olabel_unique(labels, num_labels, states,
                                             ctc_ops.ctc_unique_labels(labels))
    olabel = math_ops.exp(olabel)
    blank = olabel[:, :, 0]

    self.assertAllClose(blank, [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    self.assertAllClose(olabel[:, :, 1:],
                        [
                            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                             [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                            [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                            [[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                        ])

  @test_util.run_deprecated_v1
  def testScan(self):
    # [FUNCTION DOC]
    # Scenario: the internal scan helper with an addition function over a vector of three
    # numbers and an initial value, on GPU if available else CPU. Checks the default (exclusive,
    # forward) output, inclusive output with the initial value included, reverse scanning,
    # reverse plus inclusive, and a scan over rows of a matrix with a vector initial value.
    # Graph mode only (deprecated v1).
    with ops.device("/GPU:0" if test.is_gpu_available() else "/CPU:0"):
      out = ctc_ops._scan(
          lambda accum, elem: accum + elem,
          constant_op.constant([1.0, 2.0, 3.0]), 23.0)
      self.assertAllEqual([24.0, 26.0, 29.0], out)

      out = ctc_ops._scan(
          lambda a, e: a + e,
          constant_op.constant([1.0, 2.0, 3.0]), 23.0,
          inclusive=True)
      self.assertAllEqual([23.0, 24.0, 26.0, 29.0], out)

      out = ctc_ops._scan(
          lambda a, e: a + e,
          constant_op.constant([1.0, 2.0, 3.0]), 23.0,
          reverse=True)
      self.assertAllEqual([29.0, 28.0, 26.0], out)

      out = ctc_ops._scan(
          lambda a, e: a + e,
          constant_op.constant([1.0, 2.0, 3.0]), 23.0,
          reverse=True,
          inclusive=True)
      self.assertAllEqual([29.0, 28.0, 26.0, 23.0], out)

      out = ctc_ops._scan(
          lambda a, e: a + e,
          constant_op.constant([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]),
          constant_op.constant([23.0, 24.0]))
      self.assertAllEqual([[23.0, 25.0], [25.0, 28.0], [29.0, 33.0]], out)

  @test_util.run_deprecated_v1
  def testScanCapturesVariables(self):
    # [FUNCTION DOC]
    # Scenario: the scan function uses a random scalar tensor defined outside it (a captured
    # value). Expects the scan result to equal the accumulation using that same scalar, checked
    # by evaluating both in one session run so the random value is identical. Graph mode only.
    with self.cached_session() as sess:
      x = random_ops.random_uniform([])
      fn = lambda accum, elem: accum + x * elem
      out = ctc_ops._scan(fn, constant_op.constant([0.0, 1.0, 2.0]), 23.0)
      self.assertAllClose(*sess.run([
          [23.0 + x * 0.0, 23.0 + x * 1.0, 23.0 + x * 3.0], out
      ]))

  @test_util.run_deprecated_v1
  def testScanMultipleAccumulators(self):
    # [FUNCTION DOC]
    # Scenario: scan with a pair of accumulators (a running sum and a running product of
    # vectors) over three elements. Expects the two outputs to be the running sums and running
    # element-wise products. Graph mode only.
    with ops.device("/GPU:0" if test.is_gpu_available() else "/CPU:0"):
      def fn(accum, elem):
        # [FUNCTION DOC]
        # HELPER: nested scan step function for the enclosing test.
        # Takes a pair accumulator and one element; returns a new pair whose first part adds the
        # element and whose second part is multiplied by it.
        accum_a, accum_b = accum
        return accum_a + elem, accum_b * elem
      out = ctc_ops._scan(
          fn, constant_op.constant([1.0, 2.0, 3.0]),
          (23.0, constant_op.constant([1.0, 2.0])))
      a, b = out
      self.assertAllEqual([24.0, 26.0, 29.0], a)
      self.assertAllEqual([[1.0, 2.0], [2.0, 4.0], [6.0, 12.0]], b)

  @test_util.run_deprecated_v1
  def testScanMultipleElements(self):
    # [FUNCTION DOC]
    # Scenario: scan whose elements are a pair of tensors (a vector and a matrix) with an
    # initial vector value. Expects the running accumulation of the products of the paired
    # elements. Graph mode only.
    with ops.device("/GPU:0" if test.is_gpu_available() else "/CPU:0"):
      def fn(accum, elem):
        # [FUNCTION DOC]
        # HELPER: nested scan step function for the enclosing test.
        # Takes an accumulator and a pair element; returns the accumulator plus the product of
        # the two parts of the element.
        elem_a, elem_b = elem
        return accum + (elem_a * elem_b)
      elems_a = constant_op.constant([1.0, 2.0, 3.0])
      elems_b = constant_op.constant([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
      out = ctc_ops._scan(
          fn, (elems_a, elems_b),
          initial=constant_op.constant([0.0, 0.0]))
      self.assertAllEqual(
          [[1.0, 2.0], [5.0, 8.0], [14.0, 20.0]], out)


def _ctc_loss_v3(labels, logits, label_length, logit_length, use_gpu,
                 sparse=True):
  # [FUNCTION DOC]
  # HELPER: runs the v3 CTC loss and its gradient for the V3 and deterministic tests.
  # Inputs: labels (dense label matrix), logits (tensor), label_length and logit_length
  # (per-batch lengths), use_gpu (selects device placement), sparse (default true: convert
  # labels to sparse form first). Uses blank index 0. Records the gradient with respect to the
  # logits. Returns a pair of loss and gradient.
  with test_util.device(use_gpu=use_gpu):
    if sparse:
      labels = ctc_ops.dense_labels_to_sparse(labels, label_length)
    with backprop.GradientTape() as t:
      t.watch(logits)
      ref_loss = ctc_ops.ctc_loss_v3(
          labels=labels,
          logits=logits,
          label_length=label_length,
          logit_length=logit_length,
          blank_index=0)
    ref_grad = t.gradient(ref_loss, logits)
    return ref_loss, ref_grad


@test_util.run_all_in_graph_and_eager_modes
class CTCLossTestV3(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests the v3 CTC loss (GPU-capable implementation): agreement with the CPU path, half
  # precision output, list-typed logits and algorithm fallback for very large label counts.
  # All tests run in both graph and eager modes; several are skipped without a GPU.

  @parameterized.parameters([False, True])
  @test_util.run_v2_only
  def testCtcLossV3(self, run_tf_func):
    # [FUNCTION DOC]
    # Parameterized by whether the helper is wrapped in a compiled function. Skips unless a GPU
    # is available and eager execution is active. Scenario: random dense labels and logits;
    # computes loss and gradient on CPU with dense labels and on GPU with sparse labels, and
    # asserts they agree (loss 1e-6, gradient 2e-6). Version 2 behavior only.
    """Testing GPU CTC loss.


    testing if GPU CTC loss will generate same result with CPU version
    """
    if not test.is_gpu_available():
      self.skipTest("Need GPU for testing.")
    if not context.executing_eagerly():
      self.skipTest("Need eager execution for testing.")
    random_seed.set_random_seed(5)

    batch_size = 8
    num_labels = 6
    max_label_length = 5
    num_frames = 12

    labels = random_ops.random_uniform([batch_size, max_label_length],
                                       minval=1,
                                       maxval=num_labels,
                                       dtype=dtypes.int64)
    logits = random_ops.random_uniform([num_frames, batch_size, num_labels])

    label_length = random_ops.random_uniform([batch_size],
                                             minval=2,
                                             maxval=max_label_length,
                                             dtype=dtypes.int64)
    label_mask = array_ops.sequence_mask(
        label_length, maxlen=max_label_length, dtype=label_length.dtype)
    labels *= label_mask
    logit_length = [num_frames] * batch_size

    if run_tf_func:
      ctc_loss = def_function.function(_ctc_loss_v3)
    else:
      ctc_loss = _ctc_loss_v3

    ref_loss, ref_grad = ctc_loss(labels, logits, label_length, logit_length,
                                  False)
    loss, grad = ctc_loss(labels, logits, label_length, logit_length, True)

    self.assertAllClose(loss, ref_loss, atol=1e-6)
    self.assertAllClose(grad, ref_grad, atol=2e-6)

  @parameterized.parameters([False, True])
  def testCtcLossFp16(self, sparse_labels):
    # [FUNCTION DOC]
    # Parameterized by whether labels are sparse. Scenario: random half-precision logits; computes
    # the v3 loss and gradient on GPU in half precision and in single precision on the cast
    # logits. Asserts the half-precision outputs are half precision and match the single
    # precision results cast to half precision.
    batch_size = 8
    num_labels = 6
    max_label_length = 5
    num_frames = 12

    labels = np.random.randint(1, num_labels, [batch_size, max_label_length])
    labels = ops.convert_to_tensor(labels, dtypes.int64)
    fp16_logits = np.random.uniform(size=[num_frames, batch_size, num_labels])
    fp16_logits = ops.convert_to_tensor(fp16_logits, dtypes.float16)
    label_length = np.random.randint(2, max_label_length, [batch_size])
    label_length = ops.convert_to_tensor(label_length, dtypes.int64)

    label_mask = array_ops.sequence_mask(
        label_length, maxlen=max_label_length, dtype=label_length.dtype)
    labels *= label_mask
    logit_length = [num_frames] * batch_size

    fp16_loss, fp16_grad = _ctc_loss_v3(
        labels, fp16_logits, label_length, logit_length, use_gpu=True,
        sparse=sparse_labels)
    fp32_loss, fp32_grad = _ctc_loss_v3(
        labels, math_ops.cast(fp16_logits, dtypes.float32), label_length,
        logit_length, use_gpu=True, sparse=sparse_labels)

    self.assertEqual(fp16_loss.dtype, dtypes.float16)
    self.assertEqual(fp16_grad.dtype, dtypes.float16)
    self.assertAllClose(
        self.evaluate(fp16_loss),
        self.evaluate(math_ops.cast(fp32_loss, dtypes.float16))
    )
    self.assertAllClose(
        self.evaluate(fp16_grad),
        self.evaluate(math_ops.cast(fp32_grad, dtypes.float16))
    )

  @parameterized.parameters([False, True])
  def testCtcLossWithListLogits(self, sparse_labels):
    # [FUNCTION DOC]
    # Parameterized by whether labels are sparse. Scenario: the v3 loss called once with logits
    # given as a nested list and once as a tensor. Asserts both losses are equal.
    batch_size = 8
    num_labels = 6
    max_label_length = 5
    num_frames = 12

    labels = np.random.randint(1, num_labels, [batch_size, max_label_length])
    labels = ops.convert_to_tensor(labels, dtypes.int64)
    logits = np.random.uniform(size=[num_frames, batch_size, num_labels])
    label_length = np.random.randint(2, max_label_length, [batch_size])
    label_length = ops.convert_to_tensor(label_length, dtypes.int64)

    label_mask = array_ops.sequence_mask(
        label_length, maxlen=max_label_length, dtype=label_length.dtype)
    labels *= label_mask
    logit_length = [num_frames] * batch_size
    if sparse_labels:
      labels = ctc_ops.dense_labels_to_sparse(labels, label_length)

    list_loss = ctc_ops.ctc_loss_v3(
        labels=labels,
        logits=logits.tolist(),
        label_length=label_length,
        logit_length=logit_length,
        blank_index=0)
    tensor_loss = ctc_ops.ctc_loss_v3(
        labels=labels,
        logits=ops.convert_to_tensor(logits, dtypes.float32),
        label_length=label_length,
        logit_length=logit_length,
        blank_index=0)

    self.assertAllClose(self.evaluate(list_loss), self.evaluate(tensor_loss))

  @test_util.run_v2_only
  def testCtcLossAlgorithmFallback(self):
    # [FUNCTION DOC]
    # Skips unless a GPU is available and eager execution is active. Scenario: a very large
    # label count (11777) with one frame and a tiny batch, which should force the GPU loss to
    # fall back to another algorithm. Asserts loss and gradient on GPU match the CPU results
    # (1e-6 and 2e-6). Version 2 behavior only.
    """Test if GPU CTC loss can fallback to the correct algorithm."""
    if not test.is_gpu_available():
      self.skipTest("Need GPU for testing.")
    if not context.executing_eagerly():
      self.skipTest("Need eager execution for testing.")
    random_seed.set_random_seed(5)

    batch_size = 1
    num_labels = 11777
    max_label_length = 2
    num_frames = 1

    labels = random_ops.random_uniform([batch_size, max_label_length],
                                       minval=1,
                                       maxval=num_labels,
                                       dtype=dtypes.int64)
    logits = random_ops.random_uniform([num_frames, batch_size, num_labels])

    label_length = random_ops.random_uniform([batch_size],
                                             minval=1,
                                             maxval=max_label_length,
                                             dtype=dtypes.int64)
    logit_length = [num_frames] * batch_size

    loss, grad = _ctc_loss_v3(labels, logits, label_length, logit_length, True)
    ref_loss, ref_grad = _ctc_loss_v3(labels, logits, label_length,
                                      logit_length, False)

    self.assertAllClose(loss, ref_loss, atol=1e-6)
    self.assertAllClose(grad, ref_grad, atol=2e-6)


@test_util.run_all_in_graph_and_eager_modes
class CTCLossDeterministicTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests that with deterministic ops enabled the v3 CTC loss and its gradient are bit-for-bit
  # reproducible across repeated runs with the same seed. Provides helpers that generate random
  # inputs and run forward and backward passes. Runs in graph and eager modes.

  def _randomFloats(self, shape):
    # [FUNCTION DOC]
    # HELPER: creates test data. Takes a shape and returns a single-precision constant tensor of
    # random values uniformly in minus one to one, drawn from the global numpy random state.
    x = (2 * np.random.random_sample(shape) - 1)
    return constant_op.constant(x, dtype=dtypes.float32)

  def _genInputParams(self,
                      num_classes=10,
                      batch_size=32,
                      max_label_sequence_length=50,
                      num_frames=100,
                      logits_time_major=True,
                      sparse_labels=True):
    # [FUNCTION DOC]
    # HELPER: generates random inputs for the CTC loss. Inputs (all optional): num_classes
    # including blank (10), batch_size (32), max_label_sequence_length (50), num_frames (100,
    # must be at least the label length, asserted), logits_time_major (true: time by batch
    # layout, else batch by time), sparse_labels (true: labels converted to sparse form).
    # Labels are random from 1 up to num_classes, zero-padded beyond random lengths of at least 1.
    # Uses the global numpy random state. Returns labels, logits, label lengths and logit
    # lengths (all frames used).
    assert num_frames >= max_label_sequence_length

    labels_shape = (batch_size, max_label_sequence_length)
    # Zero-pad the labels. Zero is the default blank index in the TF2 API.
    # num_classes includes the blank class
    unmasked_labels = np.random.randint(
        1, num_classes, size=labels_shape, dtype=np.int32)
    labels_lengths = np.random.randint(
        1, high=max_label_sequence_length, size=batch_size, dtype=np.int32)
    labels_masks = (np.arange(max_label_sequence_length) <
                    labels_lengths.reshape(batch_size, 1)).astype(np.int32)
    labels = unmasked_labels * labels_masks
    if sparse_labels:
      labels = ctc_ops.dense_labels_to_sparse(labels, labels_lengths)

    if logits_time_major:
      logits_shape = (num_frames, batch_size, num_classes)
    else:
      logits_shape = (batch_size, num_frames, num_classes)
    logits = self._randomFloats(logits_shape)

    labels_lengths = constant_op.constant(labels_lengths)

    logits_lengths = [num_frames] * batch_size
    logits_lengths = constant_op.constant(logits_lengths)

    return labels, logits, labels_lengths, logits_lengths

  def _forwardAndBackward(self, sparse_labels, logits_time_major, seed):
    # [FUNCTION DOC]
    # HELPER: one forward and backward pass for the determinism test.
    # Seeds numpy with the given seed, generates inputs (sparse or dense labels, time-major or
    # not), computes the v3 loss with blank index 0, and backpropagates random upstream
    # gradients (one per batch entry) to the logits. Returns the loss and the logits gradient.
    np.random.seed(seed)
    params = self._genInputParams(
        logits_time_major=logits_time_major, sparse_labels=sparse_labels)
    labels, logits, labels_lengths, logits_lengths = params
    output_shape = (labels_lengths.shape[0],)
    upstream_gradients = self._randomFloats(output_shape)
    with backprop.GradientTape() as tape:
      tape.watch(logits)
      loss = ctc_ops.ctc_loss_v3(
          labels,
          logits,
          labels_lengths,
          logits_lengths,
          logits_time_major=logits_time_major,
          blank_index=0)
      gradient_injector_output = loss * upstream_gradients
    return loss, tape.gradient(gradient_injector_output, logits)

  @parameterized.parameters(  # parameterized.product not yet available
      (False, False), (False, True), (True, False), (True, True))
  def testForwardAndBackward(self, sparse_labels, logits_time_major):
    # [FUNCTION DOC]
    # Parameterized over all four combinations of sparse labels and time-major logits.
    # Scenario: with deterministic ops enabled, for two seeds runs forward and backward twice
    # with identical inputs. Asserts the two losses are exactly equal and the two gradients are
    # exactly equal.
    with test_util.deterministic_ops():
      for seed in range(2):
        loss_a, gradient_a = self._forwardAndBackward(sparse_labels,
                                                      logits_time_major, seed)
        loss_b, gradient_b = self._forwardAndBackward(sparse_labels,
                                                      logits_time_major, seed)
        loss_a, loss_b, gradient_a, gradient_b = self.evaluate(
            (loss_a, loss_b, gradient_a, gradient_b))
        self.assertAllEqual(loss_a, loss_b, "Loss mismatch")
        self.assertAllEqual(gradient_a, gradient_b, "Gradient mismatch")


if __name__ == "__main__":
  test.main()
