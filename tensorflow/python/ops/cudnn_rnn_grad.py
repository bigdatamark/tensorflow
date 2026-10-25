# [MODULE DOC]
# HELPER: registers gradient functions so automatic differentiation works for the cuDNN RNN ops.
# Registers gradient functions for the fused cuDNN recurrent-network ops (versions 1, 2 and 3).
# Importing the module has the side effect of registering each gradient with the global gradient
# registry under the op type names. Each gradient requires the forward op to have been run in
# training mode and forwards the op inputs, outputs and incoming gradients to the matching
# generated backprop op. No public names are exposed.
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
"""Gradients for CuudnnRNN operators."""
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_cudnn_rnn_ops


@ops.RegisterGradient("CudnnRNN")
def _cudnn_rnn_backward(op: ops.Operation, *grads):
  # [FUNCTION DOC]
  # HELPER: gradient registered for the CudnnRNN op; called by the gradient machinery.
  # Computes gradients of the version-1 cuDNN RNN op. Takes the forward op and the incoming
  # gradients (output, hidden-state output, cell-state output; extra ones ignored). Returns the
  # tuple of gradients for the op inputs (input, initial hidden, initial cell, params) produced by
  # the generated backprop op, using the saved reserve space and the op's dropout, seed, mode and
  # direction attributes. Raises a value error if the op was not run with is_training set.
  """Gradients for the CudnnRNN op."""
  if not op.get_attr("is_training"):
    raise ValueError(
        "To use CudnnRNN in gradients, is_training must be set to True.")
  return gen_cudnn_rnn_ops.cudnn_rnn_backprop(
      input=op.inputs[0],
      input_h=op.inputs[1],
      input_c=op.inputs[2],
      params=op.inputs[3],
      output=op.outputs[0],
      output_h=op.outputs[1],
      output_c=op.outputs[2],
      output_backprop=grads[0],
      output_h_backprop=grads[1],
      output_c_backprop=grads[2],
      reserve_space=op.outputs[3],
      dropout=op.get_attr("dropout"),
      seed=op.get_attr("seed"),
      seed2=op.get_attr("seed2"),
      rnn_mode=op.get_attr("rnn_mode"),
      input_mode=op.get_attr("input_mode"),
      direction=op.get_attr("direction"))


@ops.RegisterGradient("CudnnRNNV2")
def _cudnn_rnn_backward_v2(op: ops.Operation, *grad):
  # [FUNCTION DOC]
  # HELPER: gradient registered for the CudnnRNNV2 op; called by the gradient machinery.
  # Same as the version-1 gradient but also passes the host-reserved buffer (fifth output) to the
  # version-2 backprop op. Takes the forward op and the incoming gradients for output, hidden and
  # cell outputs. Returns the gradients for the op inputs. Raises a value error if the forward op
  # was not run with is_training set.
  if not op.get_attr("is_training"):
    raise ValueError(
        "To use CudnnRNNV2 in gradients, is_training must be set to True.")
  return gen_cudnn_rnn_ops.cudnn_rnn_backprop_v2(
      input=op.inputs[0],
      input_h=op.inputs[1],
      input_c=op.inputs[2],
      params=op.inputs[3],
      output=op.outputs[0],
      output_h=op.outputs[1],
      output_c=op.outputs[2],
      output_backprop=grad[0],
      output_h_backprop=grad[1],
      output_c_backprop=grad[2],
      reserve_space=op.outputs[3],
      host_reserved=op.outputs[4],
      dropout=op.get_attr("dropout"),
      seed=op.get_attr("seed"),
      seed2=op.get_attr("seed2"),
      rnn_mode=op.get_attr("rnn_mode"),
      input_mode=op.get_attr("input_mode"),
      direction=op.get_attr("direction"))


@ops.RegisterGradient("CudnnRNNV3")
def _cudnn_rnn_backwardv3(op: ops.Operation, *grads):
  # [FUNCTION DOC]
  # HELPER: gradient registered for the CudnnRNNV3 op; called by the gradient machinery.
  # Computes gradients of the version-3 cuDNN RNN op, which adds per-batch sequence lengths, a
  # time-major flag and a projection size. Takes the forward op and the incoming gradients.
  # Returns the backprop results for the first four inputs with an extra nothing/null entry
  # appended for the non-differentiable sequence lengths input. Raises a value error if the
  # forward op was not run with is_training set.
  """Gradients for the CudnnRNNV3 op."""
  if not op.get_attr("is_training"):
    raise ValueError(
        "To use CudnnRNNV3 in gradients, is_training must be set to True.")
  return gen_cudnn_rnn_ops.cudnn_rnn_backprop_v3(
      input=op.inputs[0],
      input_h=op.inputs[1],
      input_c=op.inputs[2],
      params=op.inputs[3],
      sequence_lengths=op.inputs[4],
      output=op.outputs[0],
      output_h=op.outputs[1],
      output_c=op.outputs[2],
      output_backprop=grads[0],
      output_h_backprop=grads[1],
      output_c_backprop=grads[2],
      reserve_space=op.outputs[3],
      host_reserved=op.outputs[4],
      dropout=op.get_attr("dropout"),
      seed=op.get_attr("seed"),
      seed2=op.get_attr("seed2"),
      time_major=op.get_attr("time_major"),
      num_proj=op.get_attr("num_proj"),
      rnn_mode=op.get_attr("rnn_mode"),
      input_mode=op.get_attr("input_mode"),
      direction=op.get_attr("direction")) + (None,)
