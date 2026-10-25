# [MODULE DOC]
# Gradient definitions for the legacy TensorArray operations (all versions: plain, V2, V3).
# At import time it marks the TensorArray creation, grad, size and close ops (including the
# grad-with-shape op) as not differentiable, and registers gradient functions for the read, write,
# gather, scatter, concat and split ops. The gradient functions build a gradient TensorArray
# (keyed by the enclosing gradients call) and write, read, scatter, gather, split or concat the
# incoming gradient, returning flow tensors or gradient tensors for control-dependency ordering.
# Not meant to be called directly; it is used by the automatic differentiation machinery.
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
"""Gradients for operators defined in tensor_array_ops.py."""
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops

# TODO(b/31222613): These ops may be differentiable, and there may be
# latent bugs here.
ops.NotDifferentiable("TensorArray")
ops.NotDifferentiable("TensorArrayGrad")
ops.NotDifferentiable("TensorArraySize")
ops.NotDifferentiable("TensorArrayClose")

ops.NotDifferentiable("TensorArrayV2")
ops.NotDifferentiable("TensorArrayGradV2")
ops.NotDifferentiable("TensorArraySizeV2")
ops.NotDifferentiable("TensorArrayCloseV2")

ops.NotDifferentiable("TensorArrayV3")
ops.NotDifferentiable("TensorArrayGradV3")
ops.NotDifferentiable("TensorArrayGradWithShape")
ops.NotDifferentiable("TensorArraySizeV3")
ops.NotDifferentiable("TensorArrayCloseV3")


def _GetGradSource(op_or_tensor):
  # [FUNCTION DOC]
  # HELPER: lets the TensorArray gradient functions find which gradients call they belong to.
  # Input: an operation or tensor (an input to a TensorArray grad call) with a slash-separated
  # name. Splits the name, finds the last component starting with the gradients prefix, and returns
  # the name prefix up to and including it (the unique label for that gradients call).
  # Raises a value error with a descriptive message if no component starts with the prefix, which
  # happens when a gradients call was given a custom name not starting with it.
  """Identify which call to tf.gradients created this gradient op or tensor.

  TensorArray gradient calls use an accumulator TensorArray object.  If
  multiple gradients are calculated and run in the same session, the multiple
  gradient nodes may accidentally flow through the same accumulator TensorArray.
  This double counting breaks the TensorArray gradient flow.

  The solution is to identify which gradient call this particular
  TensorArray*Grad is being called in, by looking at the input gradient
  tensor's name, and create or lookup an accumulator gradient TensorArray
  associated with this specific call.  This solves any confusion and ensures
  different gradients from the same forward graph get their own accumulators.

  This function creates the unique label associated with the tf.gradients call
  that is used to create the gradient TensorArray.

  Args:
    op_or_tensor: `Tensor` or `Operation` which is an input to a
      TensorArray*Grad call.

  Returns:
    A python string, the unique label associated with this particular
    gradients calculation.

  Raises:
    ValueError: If not called within a gradients calculation.
  """
  name_tokens = op_or_tensor.name.split("/")
  grad_pos = [i for i, x in enumerate(name_tokens) if x.startswith("gradients")]
  if not grad_pos:
    raise ValueError(
        "Expected op/tensor name to start with gradients (excluding scope)"
        f", got: {op_or_tensor.name}. This means that a tf.gradients op with "
        "this op in its dependency path has a custom name that does not start "
        "with 'gradients'. Please make sure all calls to tf.gradients that "
        "have non-empty `name` arguments use names that start with "
        "'gradients'.")
  return "/".join(name_tokens[:grad_pos[-1] + 1])


@ops.RegisterGradient("TensorArrayRead")
@ops.RegisterGradient("TensorArrayReadV2")
@ops.RegisterGradient("TensorArrayReadV3")
def _TensorArrayReadGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Registered gradient for the TensorArray read op (three versions).
  # Inputs: the forward read op (inputs are handle, index, flow; attribute dtype) and the gradient
  # tensor of its output. Builds the gradient TensorArray for this gradients call (keyed by the
  # source label) and writes the gradient at the read index.
  # Returns a list: nothing for handle and index, and the written array's flow for the flow input.
  # Raises a value error (from the helper) if the gradient name lacks the gradients prefix.
  """Gradient for TensorArrayRead.

  Args:
    op: Forward TensorArrayRead op.
    grad: Gradient `Tensor` to TensorArrayRead.

  Returns:
    A flow `Tensor`, which can be used in control dependencies to
    force the write of `grad` to the gradient `TensorArray`.
  """
  # Note: the forward flow dependency in the call to grad() is necessary for
  # the case of dynamic sized TensorArrays.  When creating the gradient
  # TensorArray, the final size of the forward array must be known.
  # For this we need to wait until it has been created by depending on
  # the input flow of the original op.
  handle = op.inputs[0]
  index = op.inputs[1]
  flow = op.inputs[2]
  dtype = op.get_attr("dtype")
  grad_source = _GetGradSource(grad)
  g = (tensor_array_ops.TensorArray(dtype=dtype, handle=handle, flow=flow,
                                    colocate_with_first_write_call=False)
       .grad(source=grad_source, flow=flow))
  w_g = g.write(index, grad)
  return [None, None, w_g.flow]


@ops.RegisterGradient("TensorArrayWrite")
@ops.RegisterGradient("TensorArrayWriteV2")
@ops.RegisterGradient("TensorArrayWriteV3")
def _TensorArrayWriteGrad(op: ops.Operation, flow):
  # [FUNCTION DOC]
  # Registered gradient for the TensorArray write op (three versions).
  # Inputs: the forward write op (inputs handle, index, value, flow; attribute T) and the incoming
  # gradient flow tensor. Adds an identity of the forward output flow and a control dependency so
  # the gradient array is not created before the final write (avoids a race), then reads the
  # gradient at the written index from the gradient TensorArray.
  # Returns a list: nothing for handle and index, the read gradient for the value, and the
  # barrier-protected flow for the flow input. Raises a value error if the flow name lacks the
  # gradients prefix. Adds identity ops to the graph.
  """Gradient for TensorArrayWrite.

  Args:
    op: Forward TensorArrayWrite op.
    flow: Gradient `Tensor` flow to TensorArrayWrite.

  Returns:
    A grad `Tensor`, the gradient created in an upstream ReadGrad or PackGrad.
  """
  # handle is the output store_handle of TensorArrayReadGrad or
  # the handle output of TensorArrayWriteGrad.  we must use this one.
  handle = op.inputs[0]
  index = op.inputs[1]
  dtype = op.get_attr("T")
  grad_source = _GetGradSource(flow)
  flow_out = array_ops.identity(op.outputs[0], "flow_out")
  # Avoid a race condition where the TensorArrayGrad op is executed before the
  # final TensorArrayWrite by adding a control dependency on the output flow of
  # the write to the input flow to the TensorArrayGrad.
  with ops.control_dependencies([flow_out]):
    flow = array_ops.identity(flow, "write_barrier")
  g = (tensor_array_ops.TensorArray(dtype=dtype, handle=handle, flow=flow,
                                    colocate_with_first_write_call=False)
       .grad(source=grad_source, flow=flow))
  grad = g.read(index)
  return [None, None, grad, flow]


@ops.RegisterGradient("TensorArrayGather")
@ops.RegisterGradient("TensorArrayGatherV2")
@ops.RegisterGradient("TensorArrayGatherV3")
def _TensorArrayGatherGrad(op: ops.Operation, grad):
  # [FUNCTION DOC]
  # Registered gradient for the TensorArray gather op (three versions).
  # Inputs: the forward gather op (inputs handle, indices, flow; attribute dtype) and the gradient
  # of the gathered tensor. Obtains the gradient TensorArray for this gradients call and scatters
  # the gradient into it at the gathered indices.
  # Returns a list: nothing for handle and indices, and the scatter's flow for the flow input.
  # Raises a value error if the gradient name lacks the gradients prefix.
  """Gradient for TensorArrayGather.

  Args:
    op: Forward TensorArrayGather op.
    grad: Gradient `Tensor` to TensorArrayGather.

  Returns:
    A flow `Tensor`, which can be used in control dependencies to
    force the write of `grad` to the gradient `TensorArray`.
  """
  # Note: the forward flow dependency in the call to grad() is necessary for
  # the case of dynamic sized TensorArrays.  When creating the gradient
  # TensorArray, the final size of the forward array must be known.
  # For this we need to wait until it has been created by depending on
  # the input flow of the original op.
  handle = op.inputs[0]
  indices = op.inputs[1]
  flow = op.inputs[2]
  dtype = op.get_attr("dtype")
  grad_source = _GetGradSource(grad)
  g = (tensor_array_ops.TensorArray(dtype=dtype, handle=handle, flow=flow,
                                    colocate_with_first_write_call=False)
       .grad(source=grad_source, flow=flow))
  u_g = g.scatter(indices, grad)
  return [None, None, u_g.flow]


@ops.RegisterGradient("TensorArrayScatter")
@ops.RegisterGradient("TensorArrayScatterV2")
@ops.RegisterGradient("TensorArrayScatterV3")
def _TensorArrayScatterGrad(op: ops.Operation, flow):
  # [FUNCTION DOC]
  # Registered gradient for the TensorArray scatter op (three versions).
  # Inputs: the forward scatter op (inputs handle, indices, value, flow; attribute T) and the
  # incoming gradient flow. Adds an identity of the forward output flow and a control dependency
  # barrier to avoid a race with the gradient array creation, then gathers the gradient at the
  # scattered indices from the gradient TensorArray.
  # Returns a list: nothing for handle and indices, the gathered gradient for the value, and the
  # barrier-protected flow. Raises a value error if the flow name lacks the gradients prefix.
  """Gradient for TensorArrayScatter.

  Args:
    op: Forward TensorArrayScatter op.
    flow: Gradient `Tensor` flow to TensorArrayScatter.

  Returns:
    A grad `Tensor`, the gradient created in upstream ReadGrads or PackGrad.
  """
  handle = op.inputs[0]
  indices = op.inputs[1]
  dtype = op.get_attr("T")
  grad_source = _GetGradSource(flow)
  flow_out = array_ops.identity(op.outputs[0], "flow_out")
  # Avoid a race condition where the TensorArrayGrad op is executed before the
  # TensorArrayScatter by adding a control dependency on the output flow of
  # the scatter to the input flow to the TensorArrayGrad.
  with ops.control_dependencies([flow_out]):
    flow = array_ops.identity(flow, "write_barrier")
  g = (tensor_array_ops.TensorArray(dtype=dtype, handle=handle, flow=flow,
                                    colocate_with_first_write_call=False)
       .grad(source=grad_source, flow=flow))
  grad = g.gather(indices)
  return [None, None, grad, flow]


@ops.RegisterGradient("TensorArrayConcat")
@ops.RegisterGradient("TensorArrayConcatV2")
@ops.RegisterGradient("TensorArrayConcatV3")
def _TensorArrayConcatGrad(op: ops.Operation, grad, unused_lengths_grad):
  # [FUNCTION DOC]
  # Registered gradient for the TensorArray concat op (three versions).
  # Inputs: the forward concat op (inputs handle, flow; second output is the element lengths;
  # attribute dtype), the gradient of the concatenated value, and the gradient of the lengths output
  # (ignored). Obtains the gradient TensorArray for this gradients call and splits the gradient into
  # it using the forward lengths.
  # Returns a list of two entries: nothing for the handle and the split's flow for the flow input.
  # Raises a value error if the gradient name lacks the gradients prefix.
  """Gradient for TensorArrayConcat.

  Args:
    op: Forward TensorArrayConcat op.
    grad: Gradient `Tensor` to TensorArrayConcat.

  Returns:
    A flow `Tensor`, which can be used in control dependencies to
    force the write of `grad` to the gradient `TensorArray`.
  """
  # Note: the forward flow dependency in the call to grad() is necessary for
  # the case of dynamic sized TensorArrays.  When creating the gradient
  # TensorArray, the final size of the forward array must be known.
  # For this we need to wait until it has been created by depending on
  # the input flow of the original op.
  handle = op.inputs[0]
  flow = op.inputs[1]
  lengths = op.outputs[1]
  dtype = op.get_attr("dtype")
  grad_source = _GetGradSource(grad)
  g = (tensor_array_ops.TensorArray(dtype=dtype, handle=handle, flow=flow,
                                    colocate_with_first_write_call=False)
       .grad(source=grad_source, flow=flow))
  u_g = g.split(grad, lengths=lengths)
  # handle, flow_in
  return [None, u_g.flow]


@ops.RegisterGradient("TensorArraySplit")
@ops.RegisterGradient("TensorArraySplitV2")
@ops.RegisterGradient("TensorArraySplitV3")
def _TensorArraySplitGrad(op: ops.Operation, flow):
  # [FUNCTION DOC]
  # Registered gradient for the TensorArray split op (three versions).
  # Inputs: the forward split op (inputs handle, value, lengths, flow; attribute T) and the incoming
  # gradient flow. Adds an identity of the forward output flow and a control dependency barrier to
  # avoid a race with gradient array creation, then concatenates the gradient array's contents.
  # Returns a list: nothing for handle, the concatenated gradient for the value, nothing for the
  # lengths, and the barrier-protected flow. Raises a value error if the flow name lacks the
  # gradients prefix.
  """Gradient for TensorArraySplit.

  Args:
    op: Forward TensorArraySplit op.
    flow: Gradient `Tensor` flow to TensorArraySplit.

  Returns:
    A grad `Tensor`, the gradient created in upstream ReadGrads or PackGrad.
  """
  handle = op.inputs[0]
  dtype = op.get_attr("T")
  grad_source = _GetGradSource(flow)
  flow_out = array_ops.identity(op.outputs[0], "flow_out")
  # Avoid a race condition where the TensorArrayGrad op is executed before the
  # TensorArraySplit by adding a control dependency on the output flow of
  # the split to the input flow to the TensorArrayGrad.
  with ops.control_dependencies([flow_out]):
    flow = array_ops.identity(flow, "write_barrier")
  g = (tensor_array_ops.TensorArray(dtype=dtype, handle=handle, flow=flow,
                                    colocate_with_first_write_call=False)
       .grad(source=grad_source, flow=flow))
  grad = g.concat()
  # handle, value, lengths, flow_in
  return [None, grad, None, flow]
