# [MODULE DOC]
# Helper module of small predicates and context-walking utilities for graph-mode (version 1)
# control flow, plus the global switch that decides whether control flow v2 is used. Kept separate
# to avoid a dependency cycle between the graph definition module and the control flow ops module.
# Import-time side effect: computes the global control flow v2 flag from the TF2 behavior setting
# and several environment variables (explicit v2 enable, cond v2, while v2, tensor array v2).
# Contains helpers to classify switch/merge/enter/exit ops, find enclosing while, cond and XLA
# contexts, and validate that an op's input comes from a legal context.
# HELPER: shared utility module for the control flow implementation and the graph module.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Utility functions for control flow.

This file is necessary to avoid cyclic dependencies between ops.py and
control_flow_ops.py.
"""

import os
import traceback

from tensorflow.python import tf2
from tensorflow.python.platform import tf_logging as logging

ENABLE_CONTROL_FLOW_V2 = ((tf2.enabled() and
                           os.getenv("TF_ENABLE_CONTROL_FLOW_V2") != "0") or
                          os.getenv("TF_ENABLE_CONTROL_FLOW_V2", "0") != "0" or
                          os.getenv("TF_ENABLE_COND_V2", "0") != "0" or
                          os.getenv("TF_ENABLE_WHILE_V2", "0") != "0" or
                          os.getenv("TF_ENABLE_TENSOR_ARRAY_V2", "0") != "0")


# TODO(b/137793122): Remove this.
def enable_control_flow_v2():  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # KNOWN ISSUE: marked for removal (see the to-do above it) and the docstring says not to use it,
  # yet it remains a live way to flip the global switch.
  # Sets the module-level control flow v2 flag to true. Takes no arguments, returns nothing, never
  # raises. Side effect: mutates module global state, affecting all later graph construction.
  """Use control flow v2.

  Do not use this symbol. This will be removed.
  """
  global ENABLE_CONTROL_FLOW_V2
  ENABLE_CONTROL_FLOW_V2 = True


def EnableControlFlowV2(graph):
  # [FUNCTION DOC]
  # Decides whether control flow v2 should be used when building into the given graph. Input: a
  # graph object. Returns true if the global flag is set, or if the graph is building a function and
  # is not a legacy function graph (detected by the absence of a captured-values attribute).
  # No side effects.
  """Returns whether control flow v2 should be used in `graph`."""
  # Enable new control flow in FuncGraphs (but not legacy _FuncGraphs).
  # TODO(skyewm): do something better than hasattr without messing up imports.
  return ENABLE_CONTROL_FLOW_V2 or (
      graph.building_function and not hasattr(graph, "_captured"))


def IsInXLAContext(op):
  # [FUNCTION DOC]
  # Tells whether an operation is inside an XLA compilation scope. Input: an operation. Returns
  # true if the op has a true XLA-compile attribute, or if its control flow context chain contains
  # an XLA context; otherwise false. A missing attribute (value error) is silently ignored.
  # No side effects.
  try:
    xla_compile = op.get_attr("_XlaCompile")
    if xla_compile: return True
  except ValueError:
    pass
  ctxt = op._get_control_flow_context()  # pylint: disable=protected-access
  return GetContainingXLAContext(ctxt) is not None


def InXlaContext(graph):
  # [FUNCTION DOC]
  # Tells whether the graph's current control flow context is, or is nested inside, an XLA context.
  # Input: a graph object. Returns a boolean. Reads private graph state; no side effects.
  ctxt = graph._get_control_flow_context()  # pylint: disable=protected-access
  return GetContainingXLAContext(ctxt) is not None


def GraphOrParentsInXlaContext(graph):
  # [FUNCTION DOC]
  # Walks from the given graph outward through its chain of outer graphs and returns true as soon
  # as one of them is in an XLA context. Returns false when a graph has no outer graph attribute
  # (the walk reaches the outermost graph). Input: a graph object. No side effects.
  while True:
    if InXlaContext(graph): return True
    try:
      graph = graph.outer_graph
    except AttributeError:
      return False


def IsInWhileLoop(op):
  # [FUNCTION DOC]
  # Tells whether an operation is inside a while loop context (directly or nested). Input: an
  # operation. Returns a boolean. No side effects.
  ctxt = op._get_control_flow_context()  # pylint: disable=protected-access
  return GetContainingWhileContext(ctxt) is not None


def IsInCond(op):
  # [FUNCTION DOC]
  # Tells whether an operation is inside a cond context (directly or nested). Input: an operation.
  # Returns a boolean. No side effects.
  ctxt = op._get_control_flow_context()  # pylint: disable=protected-access
  return GetContainingCondContext(ctxt) is not None


def IsSwitch(op):
  # [FUNCTION DOC]
  # HELPER: op-type predicate used by the graph-mode control flow code.
  # Returns true if the operation's type is the switch op or its reference-typed variant.
  """Return true if `op` is a Switch."""
  return op.type == "Switch" or op.type == "RefSwitch"


def IsMerge(op):
  # [FUNCTION DOC]
  # HELPER: op-type predicate used by the graph-mode control flow code.
  # Returns true if the operation's type is the merge op or its reference-typed variant.
  """Return true if `op` is a Merge."""
  return op.type == "Merge" or op.type == "RefMerge"


def IsLoopEnter(op):
  # [FUNCTION DOC]
  # HELPER: op-type predicate used by the graph-mode control flow code.
  # Returns true if the operation's type is the loop enter op or its reference-typed variant.
  """Returns true if `op` is an Enter."""
  return op.type == "Enter" or op.type == "RefEnter"


def IsLoopExit(op):
  # [FUNCTION DOC]
  # HELPER: op-type predicate used by the graph-mode control flow code.
  # Returns true if the operation's type is the loop exit op or its reference-typed variant.
  """Return true if `op` is an Exit."""
  return op.type == "Exit" or op.type == "RefExit"


def IsCondSwitch(op):
  # [FUNCTION DOC]
  # Decides whether a switch op was created for a conditional. Input: an operation. Returns false
  # if it is not a switch or has no outputs. Otherwise inspects every consumer of every output
  # (for a loop-enter consumer, its outer context is used) and returns true only if all those
  # consumers live in cond contexts. Switch ops are not in the context they represent, hence the
  # consumer-based test. A consumer with no context makes the result false. No side effects.
  """Return true if `op` is the Switch for a conditional."""
  if not IsSwitch(op):
    return False
  if not op.outputs:
    return False
  # Switch nodes are not part of the cond control flow context that they
  # represent, so consider the consumers of its outputs to determine if it is
  # cond switch or not. A switch is a cond switch iff all its consumers are in
  # cond contexts.
  is_cond_switch = True
  for o in op.outputs:
    for c in o.consumers():
      ctxt = c._get_control_flow_context()  # pylint: disable=protected-access
      if IsLoopEnter(c):
        ctxt = ctxt.outer_context
      is_cond_switch = is_cond_switch and (ctxt is not None and
                                           ctxt.IsCondContext())
  return is_cond_switch


def IsCondMerge(op):
  # [FUNCTION DOC]
  # Decides whether a merge op was created for a conditional. Input: an operation. Returns false
  # if it is not a merge or has no inputs. Otherwise returns true only if the output context of
  # every input's producing op is a cond context (merge ops are not in the context they represent,
  # hence the input-based test). No side effects.
  """Return true if `op` is the Merge for a conditional."""
  if not IsMerge(op):
    return False
  if not op.inputs:
    return False
  # Merge nodes are not part of the cond control flow context that they
  # represent, so consider the inputs to the merge of to determine if it is
  # cond merge or not: A merge is a cond merge iff all its inputs are in
  # cond contexts.
  is_cond_merge = True
  for i in op.inputs:
    ctxt = GetOutputContext(i.op)
    is_cond_merge = is_cond_merge and ctxt is not None and ctxt.IsCondContext()
  return is_cond_merge


def IsLoopSwitch(op):
  # [FUNCTION DOC]
  # Decides whether a switch op belongs to a while loop. Input: an operation. Returns true if it is
  # a switch, its own control flow context is a while context, and it is not a cond switch;
  # otherwise false. No side effects.
  """Return true if `op` is the Switch for a while loop."""
  if IsSwitch(op):
    ctxt = op._get_control_flow_context()  # pylint: disable=protected-access
    return ctxt is not None and ctxt.IsWhileContext() and not IsCondSwitch(op)
  return False


def IsLoopMerge(op):
  # [FUNCTION DOC]
  # Decides whether a merge op belongs to a while loop. Input: an operation. Returns true if it is
  # a merge, its own control flow context is a while context, and it is not a cond merge;
  # otherwise false. No side effects.
  """Return true if `op` is the Merge for a while loop."""
  if IsMerge(op):
    ctxt = op._get_control_flow_context()  # pylint: disable=protected-access
    return ctxt is not None and ctxt.IsWhileContext() and not IsCondMerge(op)
  return False


def IsLoopConstantEnter(op):
  # [FUNCTION DOC]
  # Tells whether an op is a loop enter whose constant attribute is set (a loop invariant). Input:
  # an operation. Returns false for non-enter ops, otherwise the value of the constant attribute
  # (a boolean). No side effects.
  """Return true iff op is a loop invariant."""
  return IsLoopEnter(op) and op.get_attr("is_constant")


def GetLoopConstantEnter(value):
  # [FUNCTION DOC]
  # Finds the constant loop enter op behind a tensor. Input: a tensor. Follows the first input
  # backwards through switch, identity and their reference-typed variants, and returns the op
  # reached if it is a constant loop enter; otherwise returns nothing (null). No side effects.
  """Return the enter op if we can infer `value` to be a loop invariant."""
  id_ops = {"Switch", "RefSwitch", "Identity", "RefIdentity"}
  op = value.op
  while op.type in id_ops:
    op = op.inputs[0].op
  return op if IsLoopConstantEnter(op) else None


def GetOutputContext(op):
  # [FUNCTION DOC]
  # Returns the control flow context in which the op's outputs live. Input: an operation. That is
  # the op's own context, except for loop exit ops, whose outputs belong to the enclosing (outer)
  # context. Returns null if the op has no context (for example graphs imported from a graph
  # definition). No side effects.
  """Return the control flow context for the output of an op."""
  ctxt = op._get_control_flow_context()  # pylint: disable=protected-access
  # Exit nodes usually have a control flow context, except in the case where the
  # exit node was imported via import_graph_def (in which case no nodes have
  # control flow contexts).
  if ctxt is not None and IsLoopExit(op):
    ctxt = ctxt.outer_context
  return ctxt


def GetContainingWhileContext(ctxt, stop_ctxt=None):
  # [FUNCTION DOC]
  # Walks outward from a control flow context and returns the first context that is a while
  # context, or the first context equal to stop_ctxt if one is given and met earlier. Inputs: a
  # context (may be null) and an optional stop context. Returns that context, or null if none is
  # found. No side effects.
  """Returns the first ancestor WhileContext of `ctxt`.

  Returns `ctxt` if `ctxt` is a WhileContext, or None if `ctxt` is not in a
  while loop.

  Args:
    ctxt: ControlFlowContext
    stop_ctxt: ControlFlowContext, optional. If provided, the search will end
      if it sees stop_ctxt.

  Returns:
    `ctxt` if `ctxt` is a WhileContext, the most nested WhileContext containing
    `ctxt`, or None if `ctxt` is not in a while loop.  If `stop_ctxt` is not
    `None`, this returns `ctxt` if it matches `stop_ctxt` in its traversal.
  """
  while ctxt:
    if ctxt.IsWhileContext() or ctxt == stop_ctxt: return ctxt
    ctxt = ctxt.outer_context
  return None


def GetContainingXLAContext(ctxt):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says the result is null when the context is "not in a while loop",
  # but the code searches for an XLA context; the while-loop wording is a copy-paste error.
  # Walks outward from a control flow context and returns the first XLA context (the context
  # itself if it is one), or null if there is none. Input: a context (may be null). No side
  # effects.
  """Returns the first ancestor XLAContext of `ctxt`.

  Returns `ctxt` if `ctxt` is a XLAContext, or None if `ctxt` is not in a
  while loop.

  Args:
    ctxt: ControlFlowContext

  Returns:
    `ctxt` if `ctxt` is a XLAContext, the most nested XLAContext containing
    `ctxt`, or None if `ctxt` is not in a while loop.
  """
  while ctxt:
    if ctxt.IsXLAContext(): return ctxt
    ctxt = ctxt.outer_context
  return None


def GetContainingCondContext(ctxt):
  # [FUNCTION DOC]
  # Walks outward from a control flow context and returns the first cond context (the context
  # itself if it is one), or null if there is none. Input: a context (may be null). No side
  # effects.
  """Returns the first ancestor CondContext of `ctxt`.

  Returns `ctxt` if `ctxt` is a CondContext, or None if `ctxt` is not in a cond.

  Args:
    ctxt: ControlFlowContext

  Returns:
    `ctxt` if `ctxt` is a CondContext, the most nested CondContext containing
    `ctxt`, or None if `ctxt` is not in a cond.
  """
  while ctxt:
    if ctxt.IsCondContext(): return ctxt
    ctxt = ctxt.outer_context
  return None


def IsContainingContext(ctxt, maybe_containing_ctxt):
  # [FUNCTION DOC]
  # Checks whether the second context is the same as, or an ancestor of, the first. Inputs: a
  # context and a candidate container (either may be null). Walks outward from the first context;
  # returns true if the candidate is met, and false if the chain ends in null first. When the
  # candidate is itself null, the result is true (null is treated as the outermost level). No side
  # effects.
  """Returns true if `maybe_containing_ctxt` is or contains `ctxt`."""
  while ctxt is not maybe_containing_ctxt:
    if ctxt is None: return False
    ctxt = ctxt.outer_context
  return True


def OpInContext(op, ctxt):
  # [FUNCTION DOC]
  # Tells whether an operation lives in the given context or one nested inside it. Inputs: an
  # operation and a context. Returns a boolean from the containing-context check applied to the
  # op's control flow context. No side effects.
  return IsContainingContext(op._get_control_flow_context(), ctxt)  # pylint: disable=protected-access


def TensorInContext(tensor, ctxt):
  # [FUNCTION DOC]
  # Tells whether the op producing a tensor lives in the given context or one nested inside it.
  # Inputs: a tensor and a context. Returns a boolean. No side effects.
  return OpInContext(tensor.op, ctxt)


def CheckInputFromValidContext(op, input_op):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it "returns whether" the input is usable, but it returns
  # nothing; it either passes silently or raises a value error. In the last gradient-of-gradient
  # case the code tests the grad state of the input's while context but then dereferences the
  # grad state of the input's (possibly different) context, which looks like a copy-paste slip.
  # Validates that input_op may feed op with respect to while loop and cond contexts. Inputs: the
  # consuming operation and the producing operation. Valid cases: the input has no context; both
  # share a context; neither is in a while loop (conds are allowed); the op is a loop enter or
  # switch (their contexts are cleared); the input's while loop contains the op's while loop; and
  # several gradient-context cases (op in a gradient context of the input's forward context, its
  # parent, a child of it, or input in the gradient or gradient-of-gradient context of op's loop).
  # Otherwise builds an error message (different while loops, or input is in a while loop while op
  # is not), writes it plus both ops' while contexts and creation tracebacks to the info log, and
  # raises a value error pointing readers at the log.
  """Returns whether `input_op` can be used from `op`s context.

  Conceptually, only inputs from op's while context or any ancestor while
  context (including outside of any context) are valid. In practice, there are
  many other edge cases as well.

  Args:
    op: Operation
    input_op: Operation

  Raises:
    ValueError: if input_op is from an invalid context.
  """
  op_ctxt = op._get_control_flow_context()  # pylint: disable=protected-access
  input_ctxt = GetOutputContext(input_op)
  valid = False

  if not input_ctxt:
    # input_op isn't in a control flow context.
    valid = True
  elif op_ctxt is input_ctxt:
    # input_op is in the same context as op.
    valid = True
  else:
    while_ctxt = GetContainingWhileContext(op_ctxt)
    input_while_ctxt = GetContainingWhileContext(input_ctxt)

    if while_ctxt is None:
      if input_while_ctxt is None:
        # Neither op nor input_op is in a while loop, but one or both are in
        # conds. We allow this, although execution will fail if the branch
        # corresponding to input_op's cond context isn't taken.
        valid = True
      # Invalid if op isn't in a while loop and input_op is. Unless...
      if IsLoopEnter(op):
        # WhileContext._BuildLoop clears context for Enter nodes.
        valid = True
      if IsSwitch(op):
        # CondContext.AddValue clears context for Switch nodes.
        valid = True
    elif IsContainingContext(while_ctxt, input_while_ctxt):
      # input_op is in a while loop which contains op's while loop (or not in a
      # while loop at all).
      valid = True
    elif (while_ctxt.grad_state and
          IsContainingContext(while_ctxt.grad_state.forward_context,
                              input_while_ctxt)):
      # op is in a gradient context and input_op is in the associated forward
      # pass context or an ancestor thereof. This case is need to build while
      # loop gradients.
      # NOTE(skyewm): we theoretically also need this case for custom gradient
      # functions that close over tensors from ancestor contexts, but I haven't
      # verified this.
      valid = True
    elif (while_ctxt.grad_state and
          while_ctxt.grad_state.forward_context is
          input_while_ctxt._outer_context):  # pylint: disable=protected-access
      # op is in a gradient context and input_op is in a child of the associated
      # forward pass context. This case is needed for the gradients of while
      # loops with conds.
      valid = True
    elif (input_while_ctxt.grad_state and
          input_while_ctxt.grad_state.forward_context is while_ctxt):
      # input_op is in the gradient context of op's context. This case is needed
      # when the gradient of a while loop gradient is requested (this will
      # eventually fail unless there is a stop_gradient() or similar).
      valid = True
    elif (input_while_ctxt.grad_state and
          input_ctxt.grad_state.forward_context.grad_state and
          input_ctxt.grad_state.forward_context.grad_state.forward_context is
          while_ctxt):
      # input_op is in the grad grad context of op's context. This case is
      # needed when the gradient of a while loop gradient is requested (this
      # will eventually fail unless there is a stop_gradient() or similar).
      valid = True

  if not valid:
    if while_ctxt:
      error_msg = (
          f"Cannot use '{input_op.name}' as input to '{op.name}' because they "
          "are in different while loops.")
    else:
      error_msg = (
          f"Cannot use '{input_op.name}' as input to '{op.name}' because "
          f"'{input_op.name}' is in a while loop.")

    # Log the error message plus the relevant stack traces. The stacks may be
    # useful for debugging this error, but we don't want to raise an
    # unreadable exception.
    log_msg = error_msg
    log_msg += "\n\n%s while context: %s" % (op.name, while_ctxt)
    log_msg += "\n%s while context: %s" % (input_op.name, input_while_ctxt)
    log_msg += "\n\nTraceback for %s:\n%s\nTraceback for %s:\n%s\n" % (
        op.name, "".join(traceback.format_list(op.traceback)),
        input_op.name, "".join(traceback.format_list(input_op.traceback)))
    logging.info(log_msg)
    raise ValueError(error_msg + " See info log for more details.")


def GetWhileContext(op):
  # [FUNCTION DOC]
  # Returns the while context an operation belongs to by asking the op's control flow context for
  # its while context. Input: an operation. Returns null if the op has no context. No side effects.
  """Get the WhileContext to which this op belongs."""
  ctxt = op._get_control_flow_context()  # pylint: disable=protected-access
  if ctxt:
    ctxt = ctxt.GetWhileContext()
  return ctxt
