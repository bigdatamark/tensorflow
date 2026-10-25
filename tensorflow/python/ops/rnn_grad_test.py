# [MODULE DOC]
# Tests for the block LSTM gradient and cell kernels. Defines one test class plus two small
# module-level helpers for random inputs and gate layout conversion. Runs under the standard
# TensorFlow test main when executed directly.
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
"""Tests for gradients of (block) LSTM/GRU operations."""

import functools

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_rnn_ops
from tensorflow.python.ops import gradients
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_grad  # pylint: disable=unused-import
from tensorflow.python.platform import test


class RNNGradTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the block LSTM ops and their gradients: consistency between the V1 and V2
  # ops, and CPU versus GPU agreement of the single-step LSTM block cell and its gradient. Has
  # one shared helper method that runs a block LSTM op with zero peephole weights.

  @test_util.deprecated_graph_mode_only
  def testBlockLSTMV1V2Consistency(self):
    # [FUNCTION DOC]
    # Checks that BlockLSTM (gate order input, cell, forget, output) and BlockLSTMV2 (order input,
    # forget, cell, output) agree. Builds a one-step, batch-one, hidden-size-eight problem with
    # deterministic random weights, runs V1 with forget bias and clip disabled, and V2 on the
    # reordered weights and bias. Computes gradients of the summed outputs with respect to
    # weights and bias for both, and asserts exact equality of cell states, hidden states and
    # both gradients. Graph mode only.
    num_steps = 1
    batch_size = 1
    input_size = 1
    hidden_size = 8
    w = deterministic_random_uniform(
        [input_size + hidden_size, 4 * hidden_size])
    b = deterministic_random_uniform([4 * hidden_size])
    x = deterministic_random_uniform([num_steps, batch_size, input_size])
    cs_prev = h_prev = deterministic_random_uniform([batch_size, hidden_size])

    all_cs, all_h = self._lstm_block(
        functools.partial(
            gen_rnn_ops.BlockLSTM,
            forget_bias=0.0,  # Disable to match V2 default.
            cell_clip=0.0),  # Disable to match V2 default.
        w, b, x, cs_prev, h_prev)
    w_grad, b_grad = gradients.gradients(all_cs + all_h, [w, b])

    w_ifco, b_ifco = icfo_to_ifco(w, b)
    all_cs_ifco, all_h_ifco = self._lstm_block(
        gen_rnn_ops.BlockLSTMV2, w_ifco, b_ifco, x, cs_prev, h_prev)
    w_ifco_grad, b_ifco_grad = gradients.gradients(
        all_cs_ifco + all_h_ifco, [w_ifco, b_ifco])

    self.assertAllEqual(all_cs, all_cs_ifco)
    self.assertAllEqual(all_h, all_h_ifco)
    self.assertAllEqual(w_grad, w_ifco_grad)
    self.assertAllEqual(b_grad, b_ifco_grad)

  @test_util.deprecated_graph_mode_only
  def testLSTMBlockCell(self):
    # [FUNCTION DOC]
    # Checks the LSTM block cell and its gradient give the same results on CPU and GPU. Uses
    # random sizes (1 to 31) for batch, input and hidden dimensions, random weights and upstream
    # gradients, zero peephole weights. For each device setting, runs the cell op and then its
    # gradient op, collecting outputs and gradients, and asserts both pairs are close. Graph mode
    # only. Without a GPU both runs use the same device so the check is trivial.
    batch_size = np.random.randint(1, 32)
    input_size = np.random.randint(1, 32)
    hidden_size = np.random.randint(1, 32)
    w = deterministic_random_uniform(
        [input_size + hidden_size, 4 * hidden_size])
    b = deterministic_random_uniform([4 * hidden_size])
    x = deterministic_random_uniform([batch_size, input_size])
    cs_prev = h_prev = deterministic_random_uniform([batch_size, hidden_size])
    w_peephole = array_ops.zeros(cs_prev.shape[1:], dtype=w.dtype)
    cs_grad = deterministic_random_uniform([batch_size, hidden_size])
    h_grad = deterministic_random_uniform([batch_size, hidden_size])

    outputs = []
    grads = []
    for use_gpu in [False, True]:
      with self.cached_session(use_gpu=use_gpu):
        output = gen_rnn_ops.lstm_block_cell(
            x=x,
            cs_prev=cs_prev,
            h_prev=h_prev,
            w=w,
            wci=w_peephole,
            wcf=w_peephole,
            wco=w_peephole,
            b=b,
            forget_bias=1.0,
            cell_clip=0.0,
            use_peephole=False)
        (i, cs, f, o, ci, co, _) = output
        grad = gen_rnn_ops.lstm_block_cell_grad(
            x=x,
            cs_prev=cs_prev,
            h_prev=h_prev,
            w=w,
            wci=w_peephole,
            wcf=w_peephole,
            wco=w_peephole,
            b=b,
            i=i,
            cs=cs,
            f=f,
            o=o,
            ci=ci,
            co=co,
            cs_grad=cs_grad,
            h_grad=h_grad,
            use_peephole=False)
        outputs.append(output)
        grads.append(grad)
    self.assertAllClose(outputs[0], outputs[1])
    self.assertAllClose(grads[0], grads[1])

  def _lstm_block(self, op, w, b, x, cs_prev, h_prev):
    # [FUNCTION DOC]
    # HELPER: shared by the consistency test to run a block LSTM op variant.
    # Inputs: op (callable for the BlockLSTM or V2 op), weights, bias, input sequence, previous
    # cell state and previous hidden state. Uses zero peephole weights, no peephole, and the
    # sequence length taken from the first dimension of the input. Returns a pair of the all-steps
    # cell state and all-steps hidden output tensors.
    w_peephole = array_ops.zeros(cs_prev.shape[1:], dtype=w.dtype)
    _, all_cs, _, _, _, _, all_h = op(
        seq_len_max=math_ops.cast(array_ops.shape(x)[0], dtypes.int64),
        x=x,
        cs_prev=cs_prev,
        h_prev=h_prev,
        w=w,
        wci=w_peephole,
        wcf=w_peephole,
        wco=w_peephole,
        b=b,
        use_peephole=False)
    return all_cs, all_h


def deterministic_random_uniform(shape):
  # [FUNCTION DOC]
  # HELPER: test utility making random float32 data. Input: shape. Returns a float32 tensor of
  # uniform random values in zero to one, drawn from the global numpy generator (so repeatable
  # only if the caller seeds it; the name suggests determinism but no seed is set here).
  return ops.convert_to_tensor(np.random.random(shape), dtype=dtypes.float32)


def icfo_to_ifco(w, b):
  # [FUNCTION DOC]
  # HELPER: converts block LSTM weights and bias between gate layouts.
  # Inputs: w (weights, four gates split along the second axis) and b (bias, four equal parts).
  # Swaps the second and third gate blocks, turning input-cell-forget-output order into
  # input-forget-cell-output. Returns the pair of reordered weights and bias.
  """Convert gates' weights and biases from ICFO to IFCO layout."""
  w_i, w_c, w_f, w_o = array_ops.split(w, num_or_size_splits=4, axis=1)
  b_i, b_c, b_f, b_o = array_ops.split(b, num_or_size_splits=4)
  w_ifco = array_ops.concat([w_i, w_f, w_c, w_o], axis=1)
  b_ifco = array_ops.concat([b_i, b_f, b_c, b_o], axis=0)
  return w_ifco, b_ifco


if __name__ == "__main__":
  test.main()
