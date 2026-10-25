# [MODULE DOC]
# Gradient registration for the block LSTM ops. Defines the gradient function for BlockLSTM and
# registers it, at import time, for both the BlockLSTM and BlockLSTMV2 ops with the gradient
# registry. The function delegates to the generated block LSTM gradient kernel.
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
"""Gradients for (block) GRU/LSTM operators."""
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_rnn_ops


def _block_lstm_grad(op, *grads):
  # [FUNCTION DOC]
  # HELPER: gradient function registered for the BlockLSTM and BlockLSTMV2 ops.
  # Inputs: op (the forward operation, whose nine inputs and seven outputs are unpacked) and the
  # upstream gradients, of which only the cell-state gradient (second) and hidden-output gradient
  # (last) are used; the rest are ignored. Reads the peephole attribute from the op and calls the
  # block LSTM gradient kernel. Returns a tuple of nine gradients matching the op inputs: null
  # for the sequence length, then gradients for input, previous cell state, previous hidden
  # state, weights, three peephole weights and bias.
  """Gradient for the BlockLSTM op."""
  seq_len_max, x, cs_prev, h_prev, w, wci, wcf, wco, b = op.inputs
  i, cs, f, o, ci, co, h = op.outputs
  _, cs_grad, _, _, _, _, h_grad = grads
  (x_grad, cs_prev_grad, h_prev_grad, w_grad, wci_grad, wcf_grad, wco_grad,
   b_grad) = gen_rnn_ops.block_lstm_grad(
       seq_len_max=seq_len_max,
       x=x,
       cs_prev=cs_prev,
       h_prev=h_prev,
       w=w,
       wci=wci,
       wcf=wcf,
       wco=wco,
       b=b,
       i=i,
       cs=cs,
       f=f,
       o=o,
       ci=ci,
       co=co,
       h=h,
       cs_grad=cs_grad,
       h_grad=h_grad,
       use_peephole=op.get_attr("use_peephole"))
  return (None, x_grad, cs_prev_grad, h_prev_grad, w_grad, wci_grad, wcf_grad,
          wco_grad, b_grad)


ops.RegisterGradient("BlockLSTM")(_block_lstm_grad)
ops.RegisterGradient("BlockLSTMV2")(_block_lstm_grad)
