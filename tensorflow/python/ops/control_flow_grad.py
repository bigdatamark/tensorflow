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
# [MODULE DOC]
# HELPER: gradient-registration module for the low-level control flow ops (Switch, Merge, Exit,
# NextIteration, Enter, LoopCond and their reference-typed variants). The gradient functions are
# used by automatic differentiation when backpropagating through conditionals and while loops
# built from these primitives. Mapping: Switch gradients use Merge, Merge gradients use Switch,
# Exit gradients use Enter, Enter gradients use Exit (or accumulators), NextIteration gradient is
# an identity, and LoopCond stops gradients. Import-time side effect: registers each gradient
# function with the global gradient registry. It star-imports everything from the control flow
# ops module to get merge, switch, exit, and the context classes.
"""Gradients for operators defined in control_flow_ops.py."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import math_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import,undefined-variable,redefined-builtin
from tensorflow.python.ops.control_flow_ops import *
# pylint: enable=wildcard-import


def _SwitchGrad(op, *grad):
  # [FUNCTION DOC]
  # HELPER: gradient function registered for the Switch and RefSwitch ops.
  # Inputs: op (the forward switch operation) and its two output gradients (false branch first,
  # true branch second; either may be null). Returns a pair: gradient for the data input and
  # null for the predicate input. Behavior depends on the control flow context of the forward op.
  # In a while loop: on the first visit, if the exit-branch gradient exists, it makes a merge of
  # that gradient with itself, records it in the loop's switch map and returns it; if that
  # gradient is absent returns nulls (not differentiable). On the second visit, it updates the
  # merge's second input with a next-iteration/back edge using the other branch gradient and
  # returns nulls. In a conditional: takes the gradient of the other branch to detect a missing
  # zero gradient; returns nulls for it (or a merge of the branch gradient duplicated for resource
  # variables) otherwise merges both gradients. Outside any context: switches each gradient on
  # the predicate and merges them. Side effects: adds graph ops and updates the gradient state's
  # switch map.
  """Gradients for a Switch op is calculated using a Merge op.

  If the switch is a loop switch, it will be visited twice. We create
  the merge on the first visit, and update the other input of the merge
  on the second visit. A next_iteration is also added on second visit.
  """
  graph = ops.get_default_graph()
  # pylint: disable=protected-access
  op_ctxt = op._get_control_flow_context()
  grad_ctxt = graph._get_control_flow_context()
  # pylint: enable=protected-access
  if isinstance(op_ctxt, WhileContext):
    merge_grad = grad_ctxt.grad_state.switch_map.get(op)
    if merge_grad is not None:
      # This is the second time this Switch is visited. It comes from
      # the non-exit branch of the Switch, so update the second input
      # to the Merge.
      # TODO(yuanbyu): Perform shape inference with this new input.
      if grad[1] is not None:
        # pylint: disable=protected-access
        control_flow_ops._AddNextAndBackEdge(merge_grad, grad[1],
                                             enforce_shape_invariant=False)
        # pylint: enable=protected-access
      return None, None
    elif grad[0] is not None:
      # This is the first time this Switch is visited. It comes from
      # the Exit branch, which is grad[0]. grad[1] is empty at this point.
      # Use grad[0] for both inputs to merge for now, but update the second
      # input of merge when we see this Switch the second time.
      merge_grad = merge([grad[0], grad[0]], name="b_switch")[0]
      grad_ctxt.grad_state.switch_map[op] = merge_grad
      return merge_grad, None
    else:
      # This is the first time this Switch is visited. It comes from the
      # Identity branch. Such a Switch has `None` gradient for the Exit branch,
      # meaning the output is not differentiable.
      return None, None
  elif isinstance(op_ctxt, CondContext):
    zero_grad = grad[1 - op_ctxt.branch]
    # At this point, we have created zero_grad guarded by the right switch.
    # Unfortunately, we may still get None here for not trainable data types.
    if zero_grad is None:
      # For resource variables we get None always on the other branch, so bypass
      # this.
      if op.inputs[0].dtype == dtypes.resource:
        return merge(
            [grad[op_ctxt.branch]] * 2, name="cond_resource_grad")[0], None
      return None, None
    return merge(grad, name="cond_grad")[0], None
  else:
    false_grad = switch(grad[0], op.inputs[1])[0]
    true_grad = switch(grad[1], op.inputs[1])[1]
    return merge([false_grad, true_grad])[0], None


ops.RegisterGradient("Switch")(_SwitchGrad)
ops.RegisterGradient("RefSwitch")(_SwitchGrad)


@ops.RegisterGradient("Merge")
def _MergeGrad(op, grad, _):
  # [FUNCTION DOC]
  # HELPER: gradient function registered for the Merge op.
  # Inputs: op (forward merge), grad (gradient of the merged value), and an ignored gradient for
  # the index output. Returns the gradient(s) for the merge inputs, built with switch ops. If the
  # merge's input lives in a while loop context, switches the gradient on the loop pivot. If it
  # lives in a conditional, switches on the conditional's predicate; when this occurs inside a
  # loop gradient, the predicate value of every iteration is accumulated in the forward pass
  # (cached in the history map) and the accumulated value is used. Otherwise it builds, for each
  # input, a switch on whether the merge's chosen index equals that input and keeps the
  # taken-branch output, returning a list. Side effects: adds graph ops and may add forward
  # accumulators and update the history map of the gradient state.
  """Gradients for a Merge op are calculated using a Switch op."""
  input_op = op.inputs[0].op
  graph = ops.get_default_graph()
  # pylint: disable=protected-access
  op_ctxt = control_flow_util.GetOutputContext(input_op)
  grad_ctxt = graph._get_control_flow_context()
  # pylint: enable=protected-access
  if isinstance(op_ctxt, WhileContext):
    # pylint: disable=protected-access
    return control_flow_ops._SwitchRefOrTensor(grad, grad_ctxt.pivot)
    # pylint: enable=protected-access
  elif isinstance(op_ctxt, CondContext):
    pred = op_ctxt.pred
    if grad_ctxt and grad_ctxt.grad_state:
      # This Merge node is part of a cond within a loop.
      # The backprop needs to have the value of this predicate for every
      # iteration. So we must have its values accumulated in the forward, and
      # use the accumulated values as the predicate for this backprop switch.
      grad_state = grad_ctxt.grad_state
      real_pred = grad_state.history_map.get(pred.name)
      if real_pred is None:
        # Remember the value of pred for every iteration.
        grad_ctxt = grad_state.grad_context
        grad_ctxt.Exit()
        history_pred = grad_state.AddForwardAccumulator(pred)
        grad_ctxt.Enter()

        # Add the stack pop op. If pred.op is in a (outer) CondContext,
        # the stack pop will be guarded with a switch.
        real_pred = grad_state.AddBackpropAccumulatedValue(history_pred, pred)
        grad_state.history_map[pred.name] = real_pred
      pred = real_pred
    # pylint: disable=protected-access
    return control_flow_ops._SwitchRefOrTensor(grad, pred, name="cond_grad")
    # pylint: enable=protected-access
  else:
    num_inputs = len(op.inputs)
    cond = [math_ops.equal(op.outputs[1], i) for i in range(num_inputs)]
    # pylint: disable=protected-access
    return [
        control_flow_ops._SwitchRefOrTensor(grad, cond[i])[1]
        for i in range(num_inputs)
    ]
    # pylint: enable=protected-access


@ops.RegisterGradient("RefMerge")
def _RefMergeGrad(op, grad, _):
  # [FUNCTION DOC]
  # HELPER: gradient for the reference-typed Merge op; simply forwards the op, the gradient and
  # the ignored index gradient to the regular merge gradient function and returns its result.
  return _MergeGrad(op, grad, _)


@ops.RegisterGradient("Exit")
def _ExitGrad(op, grad):
  # [FUNCTION DOC]
  # HELPER: gradient function registered for the Exit and RefExit ops.
  # KNOWN ISSUE: the type error message for unsupported gradient types is built from two string
  # pieces with no space between "either" and the following backtick text, so it reads run-together.
  # Inputs: op (forward exit) and grad (incoming gradient: a tensor, indexed slices or sparse
  # tensor). Returns null if the gradient loop context disables backprop. Raises a type error if
  # the forward loop already has gradient state (second-order gradients of while loops are
  # unsupported) or if the gradient is of an unsupported type. Registers the gradient's tensor
  # names with the gradient loop context, then, inside that context, creates an Enter op named
  # for the backward exit (non-constant, using the context's parallel iterations), records it in
  # the context's loop enters and returns it.
  """Gradients for an exit op are calculated using an Enter op."""
  graph = ops.get_default_graph()
  # pylint: disable=protected-access
  op_ctxt = op._get_control_flow_context()
  grad_ctxt = graph._get_control_flow_context()
  # pylint: enable=protected-access
  if not grad_ctxt.back_prop:
    # The flag `back_prop` is set by users to suppress gradient
    # computation for this loop. If the attribute `back_prop` is false,
    # no gradient computation.
    return None

  if op_ctxt.grad_state:
    raise TypeError("Second-order gradient for while loops not supported.")

  if isinstance(grad, tensor.Tensor):
    grad_ctxt.AddName(grad.name)
  else:
    if not isinstance(
        grad, (indexed_slices.IndexedSlices, sparse_tensor.SparseTensor)):
      raise TypeError(f"Type {type(grad)} not supported, must be either"
                      "`indexed_slices.IndexedSlices` or `SparseTensor`.")
    grad_ctxt.AddName(grad.values.name)
    grad_ctxt.AddName(grad.indices.name)
    dense_shape = grad.dense_shape
    if dense_shape is not None:
      grad_ctxt.AddName(dense_shape.name)
  grad_ctxt.Enter()
  # pylint: disable=protected-access
  result = control_flow_ops._Enter(
      grad, grad_ctxt.name, is_constant=False,
      parallel_iterations=grad_ctxt.parallel_iterations,
      name="b_exit")
  # pylint: enable=protected-access
  grad_ctxt.loop_enters.append(result)
  grad_ctxt.Exit()
  return result


ops.RegisterGradient("RefExit")(_ExitGrad)


@ops.RegisterGradient("NextIteration")
def _NextIterationGrad(_, grad):
  # [FUNCTION DOC]
  # HELPER: gradient for NextIteration; ignores the op and returns the incoming gradient
  # unchanged (identity). The backward next-iteration is added elsewhere, in the switch gradient.
  """A forward next_iteration is translated into a backprop identity.

  Note that the backprop next_iteration is added in switch grad.
  """
  return grad


@ops.RegisterGradient("RefNextIteration")
def _RefNextIterationGrad(_, grad):
  # [FUNCTION DOC]
  # HELPER: gradient for the reference-typed NextIteration op; delegates to the regular
  # next-iteration gradient and returns the incoming gradient unchanged.
  return _NextIterationGrad(_, grad)


@ops.RegisterGradient("Enter")
def _EnterGrad(op, grad):
  # [FUNCTION DOC]
  # HELPER: gradient function registered for the Enter op.
  # Inputs: op (forward enter) and grad (incoming gradient). Returns the gradient unchanged when
  # there is no gradient context, when backprop is disabled for the loop, or when not inside a
  # gradient while-loop. For loop invariants (constant enters) it adds a backprop accumulator: a
  # standard one for dense tensors, an indexed-slices one for indexed slices; any other gradient
  # type raises a type error (sparse tensors are unsupported). For ordinary loop variables it
  # applies an exit op to the gradient, records the exit in the context's loop exits and
  # registers it as an exit result. Returns the resulting tensor. Side effects: adds graph ops
  # and mutates the gradient loop context lists.
  """Gradients for an Enter are calculated using an Exit op.

  For loop variables, grad is the gradient so just add an exit.
  For loop invariants, we need to add an accumulator loop.
  """
  graph = ops.get_default_graph()
  # pylint: disable=protected-access
  grad_ctxt = graph._get_control_flow_context()
  # pylint: enable=protected-access
  if grad_ctxt is None:
    return grad
  if not grad_ctxt.back_prop:
    # Skip gradient computation, if the attribute `back_prop` is false.
    return grad
  if grad_ctxt.grad_state is None:
    # Pass the gradient through if we are not in a gradient while context.
    return grad
  if op.get_attr("is_constant"):
    # Add a gradient accumulator for each loop invariant.
    if isinstance(grad, tensor.Tensor):
      result = grad_ctxt.AddBackpropAccumulator(op, grad)
    elif isinstance(grad, indexed_slices.IndexedSlices):
      result = grad_ctxt.AddBackpropIndexedSlicesAccumulator(op, grad)
    else:
      # TODO(yuanbyu, lukasr): Add support for SparseTensor.
      raise TypeError(f"Type {type(grad)} not supported,"
                      "must be Tensor or Indexed Slices")
  else:
    result = exit(grad)
    grad_ctxt.loop_exits.append(result)
    grad_ctxt.ExitResult([result])
  return result


@ops.RegisterGradient("RefEnter")
def _RefEnterGrad(op, grad):
  # [FUNCTION DOC]
  # HELPER: gradient for the reference-typed Enter op; forwards the op and gradient to the
  # regular enter gradient and returns its result.
  return _EnterGrad(op, grad)


@ops.RegisterGradient("LoopCond")
def _LoopCondGrad(_):
  # [FUNCTION DOC]
  # HELPER: gradient for LoopCond. Takes the op (ignored) and returns null, stopping
  # backpropagation through a while loop's predicate.
  """Stop backprop for the predicate of a while loop."""
  return None
