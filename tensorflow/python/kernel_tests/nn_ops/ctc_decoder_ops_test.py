# [MODULE DOC]
# Unit tests for the CTC (connectionist temporal classification) decoders in the ctc_ops module:
# the greedy best-path decoder and the beam search decoder. Defines two small list helpers and
# one test case class that feeds hand-built log-probability matrices to the decoders and compares
# the resulting sparse tensors and log probabilities with known truth values.
# No import-time side effects; running the file as a script starts the test runner.
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
"""Tests for tensorflow.ctc_ops.ctc_decoder_ops."""

import itertools

import numpy as np

from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import ctc_ops
from tensorflow.python.platform import test


def grouper(iterable, n, fillvalue=None):
  # [FUNCTION DOC]
  # HELPER: small utility used by the decoder tests to regroup flat session outputs.
  # Splits an iterable into consecutive tuples of length n; the last tuple is padded with
  # fillvalue (default: nothing/null value) if the input length is not a multiple of n.
  # Returns a lazy iterator of tuples. No side effects.
  """Collect data into fixed-length chunks or blocks."""
  # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
  args = [iter(iterable)] * n
  return itertools.zip_longest(fillvalue=fillvalue, *args)


def flatten(list_of_lists):
  # [FUNCTION DOC]
  # HELPER: small utility used by the decoder tests to flatten per-path sparse tensor parts.
  # Takes an iterable of iterables and returns a lazy iterator over the items of each inner
  # iterable in order, removing exactly one level of nesting. No side effects.
  """Flatten one level of nesting."""
  return itertools.chain.from_iterable(list_of_lists)


class CTCGreedyDecoderTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering both the CTC greedy decoder and the CTC beam search decoder (despite the
  # name, it also holds the beam search test). Shares one helper that runs a decoder and checks
  # results. Created and run by the test runner; holds no instance state.

  def _testCTCDecoder(self,
                      decoder,
                      inputs,
                      seq_lens,
                      log_prob_truth,
                      decode_truth,
                      expected_err_re=None,
                      **decoder_args):
    # [FUNCTION DOC]
    # HELPER: shared checker for the decoder tests in this class.
    # Inputs: decoder (the decoding function under test), inputs (list over time of batch by
    # depth arrays), seq_lens (per-batch sequence lengths), log_prob_truth (expected log
    # probabilities), decode_truth (list of expected index/value/shape triples, one per decoded
    # path), expected_err_re (optional regular expression of an expected op error), plus extra
    # keyword arguments forwarded to the decoder (e.g. blank index, beam width, top paths).
    # Stacks the inputs into one tensor, runs the decoder in a CPU-only cached session and, when
    # no error is expected, asserts the number of paths, the indices, values and dense shape of
    # each sparse result, their static shapes, and the log probabilities (tolerance 1e-6).
    # If an error pattern is given, asserts that running the session raises a matching op error.
    # Returns nothing.
    inputs_t = [ops.convert_to_tensor(x) for x in inputs]
    # convert inputs_t into a [max_time x batch_size x depth] tensor
    # from a len time python list of [batch_size x depth] tensors
    inputs_t = array_ops_stack.stack(inputs_t)

    with self.cached_session(use_gpu=False) as sess:
      decoded_list, log_probability = decoder(
          inputs_t, sequence_length=seq_lens, **decoder_args)
      decoded_unwrapped = list(
          flatten([(st.indices, st.values, st.dense_shape) for st in
                   decoded_list]))

      if expected_err_re is None:
        outputs = sess.run(decoded_unwrapped + [log_probability])

        # Group outputs into (ix, vals, shape) tuples
        output_sparse_tensors = list(grouper(outputs[:-1], 3))

        output_log_probability = outputs[-1]

        # Check the number of decoded outputs (top_paths) match
        self.assertEqual(len(output_sparse_tensors), len(decode_truth))

        # For each SparseTensor tuple, compare (ix, vals, shape)
        for out_st, truth_st, tf_st in zip(output_sparse_tensors, decode_truth,
                                           decoded_list):
          self.assertAllEqual(out_st[0], truth_st[0])  # ix
          self.assertAllEqual(out_st[1], truth_st[1])  # vals
          self.assertAllEqual(out_st[2], truth_st[2])  # shape
          # Compare the shapes of the components with the truth. The
          # `None` elements are not known statically.
          self.assertEqual([None, truth_st[0].shape[1]],
                           tf_st.indices.get_shape().as_list())
          self.assertEqual([None], tf_st.values.get_shape().as_list())
          self.assertShapeEqual(truth_st[2], tf_st.dense_shape)

        # Make sure decoded probabilities match
        self.assertAllClose(output_log_probability, log_prob_truth, atol=1e-6)
      else:
        with self.assertRaisesOpError(expected_err_re):
          sess.run(decoded_unwrapped + [log_probability])

  @test_util.run_deprecated_v1
  def testCTCGreedyDecoder(self):
    # [FUNCTION DOC]
    # Scenario: greedy best-path decoding of a batch of two sequences (lengths 4 and 5, depth 4,
    # six time steps, trailing steps ignored). Expects one decoded path with the known indices,
    # values and shape, and known negative log probabilities. Runs it with the default blank
    # index, then with the blank class moved to position 2 given as a positive index and as a
    # negative index; all three must give the same truth. Runs in graph mode (deprecated v1).
    """Test two batch entries - best path decoder."""
    max_time_steps = 6
    # depth == 4
    seq_len_0 = 4
    input_prob_matrix_0 = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],  # t=0
            [0.0, 0.0, 0.4, 0.6],  # t=1
            [0.0, 0.0, 0.4, 0.6],  # t=2
            [0.0, 0.9, 0.1, 0.0],  # t=3
            [0.0, 0.0, 0.0, 0.0],  # t=4 (ignored)
            [0.0, 0.0, 0.0, 0.0]
        ],  # t=5 (ignored)
        dtype=np.float32)
    input_log_prob_matrix_0 = np.log(input_prob_matrix_0)

    seq_len_1 = 5
    # dimensions are time x depth
    input_prob_matrix_1 = np.asarray(
        [
            [0.1, 0.9, 0.0, 0.0],  # t=0
            [0.0, 0.9, 0.1, 0.0],  # t=1
            [0.0, 0.0, 0.1, 0.9],  # t=2
            [0.0, 0.9, 0.1, 0.1],  # t=3
            [0.9, 0.1, 0.0, 0.0],  # t=4
            [0.0, 0.0, 0.0, 0.0]  # t=5 (ignored)
        ],
        dtype=np.float32)
    input_log_prob_matrix_1 = np.log(input_prob_matrix_1)

    # len max_time_steps array of batch_size x depth matrices
    inputs = np.array([
        np.vstack(
            [input_log_prob_matrix_0[t, :], input_log_prob_matrix_1[t, :]])
        for t in range(max_time_steps)
    ])

    # batch_size length vector of sequence_lengths
    seq_lens = np.array([seq_len_0, seq_len_1], dtype=np.int32)

    # batch_size length vector of negative log probabilities
    log_prob_truth = np.array([
        np.sum(-np.log([1.0, 0.6, 0.6, 0.9])),
        np.sum(-np.log([0.9, 0.9, 0.9, 0.9, 0.9]))
    ], np.float32)[:, np.newaxis]

    # decode_truth: one SparseTensor (ix, vals, shape)
    decode_truth = [
        (
            np.array(
                [
                    [0, 0],  # batch 0, 2 outputs
                    [0, 1],
                    [1, 0],  # batch 1, 3 outputs
                    [1, 1],
                    [1, 2]
                ],
                dtype=np.int64),
            np.array(
                [
                    0,  # batch 0, 2 values
                    1,
                    1,  # batch 1, 3 values
                    1,
                    0
                ],
                dtype=np.int64),
            # shape is batch x max_decoded_length
            np.array([2, 3], dtype=np.int64)),
    ]

    # Test without defining blank_index
    self._testCTCDecoder(ctc_ops.ctc_greedy_decoder, inputs, seq_lens,
                         log_prob_truth, decode_truth)

    # Shift blank_index to be somewhere in the middle of inputs
    blank_index = 2
    inputs = np.concatenate(
        (inputs[:, :, :blank_index], inputs[:, :, -1:], inputs[:, :,
                                                               blank_index:-1]),
        axis=2)

    # Test positive value in blank_index
    self._testCTCDecoder(
        ctc_ops.ctc_greedy_decoder,
        inputs,
        seq_lens,
        log_prob_truth,
        decode_truth,
        blank_index=2)

    # Test negative value in blank_index
    self._testCTCDecoder(
        ctc_ops.ctc_greedy_decoder,
        inputs,
        seq_lens,
        log_prob_truth,
        decode_truth,
        blank_index=-2)

  @test_util.run_deprecated_v1
  def testCTCDecoderBeamSearch(self):
    # [FUNCTION DOC]
    # Scenario: beam search decoding of one batch entry (length 5 of 8 padded steps, depth 6)
    # with beam width 2 and top paths 2. Expects two decoded paths (two labels and one label)
    # and their known log probabilities. Then asks for 3 top paths with beam width 2 and expects
    # an invalid-argument error saying more paths were requested than the beam width allows.
    # Runs in graph mode (deprecated v1).
    """Test one batch, two beams - hibernating beam search."""
    # max_time_steps == 8
    depth = 6

    seq_len_0 = 5
    input_prob_matrix_0 = np.asarray(
        [
            [0.30999, 0.309938, 0.0679938, 0.0673362, 0.0708352, 0.173908],
            [0.215136, 0.439699, 0.0370931, 0.0393967, 0.0381581, 0.230517],
            [0.199959, 0.489485, 0.0233221, 0.0251417, 0.0233289, 0.238763],
            [0.279611, 0.452966, 0.0204795, 0.0209126, 0.0194803, 0.20655],
            [0.51286, 0.288951, 0.0243026, 0.0220788, 0.0219297, 0.129878],
            # Random entry added in at time=5
            [0.155251, 0.164444, 0.173517, 0.176138, 0.169979, 0.160671]
        ],
        dtype=np.float32)
    # Add arbitrary offset - this is fine
    input_prob_matrix_0 = input_prob_matrix_0 + 2.0

    # len max_time_steps array of batch_size x depth matrices
    inputs = ([
        input_prob_matrix_0[t, :][np.newaxis, :] for t in range(seq_len_0)
    ]  # Pad to max_time_steps = 8
              + 2 * [np.zeros(
                  (1, depth), dtype=np.float32)])

    # batch_size length vector of sequence_lengths
    seq_lens = np.array([seq_len_0], dtype=np.int32)

    # batch_size length vector of log probabilities
    log_prob_truth = np.array(
        [
            -5.811451,  # output beam 0
            -6.63339  # output beam 1
        ],
        np.float32)[np.newaxis, :]

    # decode_truth: two SparseTensors, (ix, values, shape)
    decode_truth = [
        # beam 0, batch 0, two outputs decoded
        (np.array(
            [[0, 0], [0, 1]], dtype=np.int64), np.array(
                [1, 0], dtype=np.int64), np.array(
                    [1, 2], dtype=np.int64)),
        # beam 1, batch 0, one output decoded
        (np.array(
            [[0, 0]], dtype=np.int64), np.array(
                [1], dtype=np.int64), np.array(
                    [1, 1], dtype=np.int64)),
    ]

    # Test correct decoding.
    self._testCTCDecoder(
        ctc_ops.ctc_beam_search_decoder,
        inputs,
        seq_lens,
        log_prob_truth,
        decode_truth,
        beam_width=2,
        top_paths=2)

    # Requesting more paths than the beam width allows.
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                (".*requested more paths than the beam "
                                 "width.*")):
      self._testCTCDecoder(
          ctc_ops.ctc_beam_search_decoder,
          inputs,
          seq_lens,
          log_prob_truth,
          decode_truth,
          beam_width=2,
          top_paths=3)


if __name__ == "__main__":
  test.main()
