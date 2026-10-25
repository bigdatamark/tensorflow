# [MODULE DOC]
# Helper module holding the bookkeeping that graph-mode (version 1) while loops need when computing
# gradients. Defines a per-loop gradient state class that links a forward loop context to its
# backprop loop context and manages stack-based accumulators for forward values needed in the
# backward pass, a manager class that maps forward loops to those states and builds zero
# gradients, a factory creating the manager when loops are involved, and zeros-like helpers for
# both control flow versions. Used by the gradient computation code in the gradients utilities.
# No import-time side effects. HELPER: internal support module for gradient computation.
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
"""Utilities for managing state of v1 control flow for computing gradients."""

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import control_flow_v2_func_graphs
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops

# pylint: disable=protected-access


def _GetMaxSizeFromNestedMaximumIterations(value, while_ctxt):
  # [FUNCTION DOC]
  # HELPER: sizing helper for gradient accumulator stacks inside XLA while loops.
  # Computes the maximum stack size needed to record a loop value for backprop by multiplying the
  # maximum-iteration counts of all while contexts between the value's context and the context in
  # which gradients are being computed. Inputs: the forward-loop tensor (used for error messages)
  # and the innermost while context enclosing it. For each enclosing loop, a maximum-iterations
  # tensor that is visible from the current context is multiplied in directly; otherwise its
  # constant value must be statically evaluable and that constant is used. Returns an integer
  # tensor holding the product (starting from one). Raises a value error if a loop has no
  # maximum iterations, or if it is defined in a nested context and is not statically known.
  # Reads the default graph's current control flow context; creates a constant op.
  """Calculate a max_size for use by stack ops inside an XLA while_loop.

  Args:
    value: The value inside the while_loop forward context.  Used for printing
      error messages.
    while_ctxt: The forward context inside which value resides.  This does not
      always match the value's immediate context, as `value` may be inside e.g.
      a cond context inside the while_loop.

  Returns:
    A tensor containing the `max_size` to feed to a Stack initializer.

  Raises:
    ValueError: If `value` is nested inside a `while_loop` that either
      lacks a `maximum_iterations` parameter, or the `maximum_iterations`
      parameter:

        - is inside a `while_loop` that is a parent of the calling context, and
        - cannot be evaluated at graph build time to a constant.
  """
  value_name = value.name
  # curr_ctxt is the context that tf.gradients was called in.
  curr_ctxt = ops.get_default_graph()._get_control_flow_context()  # pylint: disable=protected-access

  curr_ctxt_name = curr_ctxt.name if curr_ctxt is not None else ""
  max_size = constant_op.constant(1)

  # Loop through all containing while contexts between value and the
  # current context, multiplying together each context's
  # max_iterations to get the maximum stack size.
  while while_ctxt not in (None, curr_ctxt):
    max_iter = while_ctxt.maximum_iterations
    if max_iter is None:
      raise ValueError(
          "Cannot create a gradient accumulator for tensor '%s' inside "
          "XLA while_loop because maximum_iterations was not passed to "
          "the tf.while_loop call ('%s')." % (value_name, while_ctxt.name))

    # pylint: disable=protected-access
    max_iter_ctxt = max_iter.op._get_control_flow_context()
    # pylint: enable=protected-access

    # If max_iter_ctxt (non-strictly) contains curr_ctxt, then it's OK to use.
    if util.IsContainingContext(curr_ctxt, max_iter_ctxt):
      max_size *= max_iter
    else:
      # We cannot use max_iter because it's defined in a nested while
      # or cond context, so will fail if we try to use it as input to
      # any ops in curr_ctxt (e.g. max_size or the final accumulator
      # stack). Attempt to get a constant value out to use instead.
      const_max_iter = tensor_util.constant_value(max_iter)
      if const_max_iter is None:
        raise ValueError(
            "Cannot create a gradient accumulator for tensor '%s' inside XLA "
            "while_loop. maximum_iterations tensor '%s' for while_loop context "
            "'%s' must be statically known (e.g. a constant value or known "
            "shape dimension), or be defined at or outside the while loop "
            "context '%s' (currently defined in '%s')." %
            (value_name, max_iter.name, while_ctxt.name, curr_ctxt_name,
             max_iter_ctxt.name))
      max_size *= const_max_iter

    # Find the next outer WhileContext (or stop if we reach the
    # tf.gradient's context).
    while_ctxt = util.GetContainingWhileContext(
        while_ctxt.outer_context, stop_ctxt=curr_ctxt)

  return max_size


class _GradLoopState:
  # [CLASS DOC]
  # HELPER: internal bookkeeping object for building the gradient of one graph-mode while loop.
  # One instance exists per forward while loop and links the forward loop context to its
  # corresponding backprop loop context. Holds: the enclosing loop's gradient state (for nested
  # loops), forward and gradient contexts, forward and gradient loop counters, lazily created
  # synchronization trigger ops, a history map from forward tensors to their accumulator stacks, a
  # switch map from loop switches to gradient merges, lists of unused and deferred exits, the list
  # of forward loop exits, and a count of exits still pending. Created by the control flow state
  # manager during gradient construction; its methods add stack-based accumulators on the forward
  # side and matching pops on the backprop side.
  """The state used for constructing the gradient graph for a while loop.

  We create a _GradLoopState for each while loop in forward and its
  corresponding while loop in backprop. This gives us access to both
  the forward and the backprop WhileContexts.

  During the construction of gradient graph, any time when we detect
  a forward value that is needed for backprop, we create a history
  accumulator and add it to `history_map`. Any time when we backprop
  a loop switch op (in _SwitchGrad), we add the grad merge op in
  `switch_map`.
  """

  def __init__(self, forward_ctxt, outer_grad_state):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the error message for the unserialized-graph case is built from two adjacent
    # string pieces without a separating space, so it reads "withoutproperly".
    # Constructor. Inputs: the forward while context and the gradient state of the enclosing loop
    # (null for an outermost loop). Initializes all bookkeeping fields, copies the forward loop's
    # exits and sets the pending exit count to their number. Determines the outer forward context
    # (from the outer gradient state or the forward context's outer context) and raises a value
    # error if the forward context lacks an outer context attribute (graph not serialized through a
    # meta graph definition). Adds the forward loop counter inside the outer context. For a nested
    # loop it also accumulates the iteration count in the outer loop's history, creates the
    # backprop while context inside the outer gradient context and a backprop counter fed by the
    # accumulated count; for an outermost loop it creates the backprop context and counter directly.
    # Side effects: adds ops to the graph and enters and exits control flow contexts.
    # The grad loop state for the outer while loop.
    self._outer_grad_state = None

    # The while loop context for forward.
    self._forward_context = None

    # The loop counter added by AddForwardLoopCounter. It is the value
    # of the loop counter for the next iteration.
    self._forward_index = None

    # A sync op for forward.
    self._forward_sync = None

    # The while loop context for backprop.
    self._grad_context = None

    # The loop counter added by AddBackpropLoopCounter. It is the value
    # of the loop counter for the current iteration.
    self._grad_index = None

    # A sync op for backprop.
    self._grad_sync = None

    # Information needed by backprop.
    self._history_map = {}
    self._switch_map = {}
    self._unused_exits = []
    self._deferred_exits = []
    self._forward_loop_exits = list(forward_ctxt.loop_exits)
    self._pending_exits_count = len(forward_ctxt.loop_exits)

    self._outer_grad_state = outer_grad_state
    if outer_grad_state:
      outer_forward_ctxt = outer_grad_state.forward_context
    else:
      if not hasattr(forward_ctxt, "outer_context"):
        raise ValueError("Failed to call gradients on a while loop without"
                         "properly serializing graph via MetaGraphDef")
      outer_forward_ctxt = forward_ctxt.outer_context

    # Add the forward loop counter.
    with forward_ctxt._graph.as_default():  # pylint: disable=protected-access
      if outer_forward_ctxt:
        outer_forward_ctxt.Enter()
      cnt, forward_index = forward_ctxt.AddForwardLoopCounter(outer_grad_state)
      if outer_forward_ctxt:
        outer_forward_ctxt.Exit()
    self._forward_context = forward_ctxt
    self._forward_index = forward_index

    # Add the backprop WhileContext, and the backprop loop counter.
    if outer_grad_state:
      # This is a nested loop. Remember the iteration counts for each
      # execution of this inner loop.
      outer_forward_ctxt.AddName(cnt.name)
      history_cnt = outer_grad_state.AddForwardAccumulator(cnt)

      outer_grad_ctxt = outer_grad_state.grad_context
      outer_grad_ctxt.Enter()
      self._grad_context = control_flow_ops.WhileContext(
          maximum_iterations=forward_ctxt.maximum_iterations,
          parallel_iterations=forward_ctxt.parallel_iterations,
          back_prop=forward_ctxt.back_prop,
          swap_memory=forward_ctxt.swap_memory,
          name=forward_ctxt.name,
          grad_state=self)
      real_cnt = outer_grad_state.AddBackpropAccumulatedValue(history_cnt, cnt)
      self._grad_index = self._grad_context.AddBackpropLoopCounter(
          real_cnt, outer_grad_state)
      outer_grad_ctxt.Exit()
    else:
      if outer_forward_ctxt:
        outer_forward_ctxt.Enter()
      self._grad_context = control_flow_ops.WhileContext(
          maximum_iterations=forward_ctxt.maximum_iterations,
          parallel_iterations=forward_ctxt.parallel_iterations,
          back_prop=forward_ctxt.back_prop,
          swap_memory=forward_ctxt.swap_memory,
          name=forward_ctxt.name,
          grad_state=self)
      self._grad_index = self._grad_context.AddBackpropLoopCounter(
          cnt, outer_grad_state)
      if outer_forward_ctxt:
        outer_forward_ctxt.Exit()

  @property
  def outer_grad_state(self):
    # [FUNCTION DOC]
    # Read-only property returning the gradient state of the enclosing loop, or null for an
    # outermost loop. No side effects.
    """The grad loop state for outer loop."""
    return self._outer_grad_state

  @property
  def forward_context(self):
    # [FUNCTION DOC]
    # Read-only property returning the forward while loop context. No side effects.
    """The while loop context for forward."""
    return self._forward_context

  @property
  def forward_index(self):
    # [FUNCTION DOC]
    # Read-only property returning the forward loop counter tensor (the next-iteration value).
    # No side effects.
    """The loop index of forward loop."""
    return self._forward_index

  @property
  def forward_sync(self):
    # [FUNCTION DOC]
    # Lazily created synchronization node for the forward loop, used to keep stack push ops in
    # iteration order. On first access creates a control trigger op named for forward sync with no
    # control dependencies, assigns it to the forward context and adds it as a control input of
    # the forward counter's op. Later accesses return the same node. Side effects: graph mutation
    # on first access.
    """A control trigger node for synchronization in the forward loop.

    One main use is to keep the push ops of a stack executed in the
    iteration order.
    """
    if self._forward_sync is None:
      with ops.control_dependencies(None):
        self._forward_sync = control_flow_ops.control_trigger(name="f_sync")
      self._forward_sync._set_control_flow_context(self._forward_context)
      self._forward_index.op._add_control_input(self._forward_sync)
    return self._forward_sync

  @property
  def grad_context(self):
    # [FUNCTION DOC]
    # Read-only property returning the backprop while loop context. No side effects.
    """The corresponding WhileContext for gradient."""
    return self._grad_context

  @property
  def grad_index(self):
    # [FUNCTION DOC]
    # Read-only property returning the backprop loop counter tensor (current iteration value).
    # No side effects.
    """The loop index of backprop loop."""
    return self._grad_index

  @property
  def grad_sync(self):
    # [FUNCTION DOC]
    # Lazily created synchronization node for the backprop loop, used to keep stack pop ops in
    # iteration order. On first access creates a control trigger op named for backward sync with no
    # control dependencies, assigns it to the gradient context, adds it as a control input of the
    # backprop counter's op, and registers it as an inner op of the outer context if one exists.
    # Later accesses return the same node. Side effects: graph mutation on first access.
    """A control trigger node for synchronization in the grad loop.

    One main use is to keep the pop ops of a stack executed in the
    iteration order.
    """
    if self._grad_sync is None:
      with ops.control_dependencies(None):
        self._grad_sync = control_flow_ops.control_trigger(name="b_sync")
      self._grad_sync._set_control_flow_context(self._grad_context)
      self._grad_index.op._add_control_input(self._grad_sync)
      if self._grad_context.outer_context:
        self._grad_context.outer_context.AddInnerOp(self._grad_sync)
    return self._grad_sync

  @property
  def history_map(self):
    # [FUNCTION DOC]
    # Read-only property returning the mutable mapping from forward tensor names to the history
    # accumulators recorded for backprop. No side effects.
    """The map that records all the tensors needed for backprop."""
    return self._history_map

  @property
  def switch_map(self):
    # [FUNCTION DOC]
    # Read-only property returning the mutable mapping from the loop's switch ops to the gradient
    # merge ops created for them. No side effects.
    """The map that records all the Switch ops for the while loop."""
    return self._switch_map

  @property
  def unused_exits(self):
    # [FUNCTION DOC]
    # Read-only property returning the mutable list of loop exits whose outputs are not used by
    # the gradient computation. No side effects.
    """The list of "unused" exits."""
    return self._unused_exits

  @property
  def deferred_exits(self):
    # [FUNCTION DOC]
    # Read-only property returning the mutable list of loop exits whose gradient handling is
    # deferred until all gradients for the loop are available. No side effects.
    """The list of "deferred" exits."""
    return self._deferred_exits

  @property
  def forward_loop_exits(self):
    # [FUNCTION DOC]
    # Read-only property returning the list of exit tensors of the forward loop, as copied when the
    # state was created. No side effects.
    """The list of exits of the forward loop."""
    return self._forward_loop_exits

  @property
  def pending_exits_count(self):
    # [FUNCTION DOC]
    # Read-only property returning how many forward loop exits have not yet been seen by gradient
    # construction. Starts at the number of forward loop exits. No side effects.
    """The number of exits we expect to see but haven't."""
    return self._pending_exits_count

  @pending_exits_count.setter
  def pending_exits_count(self, cnt):
    # [FUNCTION DOC]
    # Setter for the pending-exits counter. Input: the new integer count. Overwrites the stored
    # value without validation; returns nothing.
    """Set the pending count to cnt."""
    self._pending_exits_count = cnt

  def AddForwardAccumulator(self, value, dead_branch=False):
    # [FUNCTION DOC]
    # Creates a stack that records, at every forward iteration, the value of a forward tensor that
    # backprop needs. Inputs: the forward tensor, and a flag (default false) saying the tensor is on
    # a dead branch of a cond. In the context where gradients are being computed, creates a stack
    # (with maximum size minus one, or a computed bound when inside an XLA context, colocated with
    # the value), makes it visible in the forward context, and pushes the value onto it: directly in
    # the forward context if the value lives there (ordering the push before the forward counter),
    # or inside the value's cond context (outer cond context for the dead-branch case) ordering the
    # push before the forward sync node. Always orders the push after the counter's increment.
    # Returns the stack tensor. Raises a type error if the value's context is neither the forward
    # context nor a cond context, and a value error from the XLA size computation. Side effects:
    # adds ops and control edges and temporarily enters and exits contexts.
    """Add an accumulator for each forward tensor that is needed in backprop.

    This is added to the forward loop at the first time when a tensor
    in the forward loop is used by backprop gradient computation loop.
    We create an accumulator that accumulates the value of tensor at each
    iteration. Called in the control flow context where gradients() is called.

    The pseudocode is:
    ```
      acc = stack();
      while (_pivot) {
        acc = stack_push(acc, value);
      }
    ```

    We make sure that the stack push op in one iteration is executed before
    next iteration. This is achieved by adding a control edge from
    `forward_index.op.inputs[0].op` to the push op, and another control
    edge from the push op to either `forward_index.op` or `forward_sync`.

    Args:
      value: The source tensor in forward that is to be accumulated.
      dead_branch: True iff the tensor is on a dead branch of a cond.

    Returns:
      The stack that contains the accumulated history of the tensor.

    Raises:
      TypeError: For internal errors involving the value condition context.
      ValueError: If `value` is inside a XLA scope and a valid max size
        for the stack can't be found.
    """
    # curr_ctxt is the context that tf.gradients was called in.
    with self._forward_index.graph.as_default():
      curr_ctxt = ops.get_default_graph()._get_control_flow_context()  # pylint: disable=protected-access
      with ops.control_dependencies(None):
        if curr_ctxt:
          curr_ctxt.Enter()
        with ops.colocate_with(value):
          # We only need to pass maximum_iterations to the stack if
          # we're inside an XLA context.
          if not util.IsInXLAContext(value.op):
            max_size = constant_op.constant(-1, dtypes.int32)
          else:
            max_size = _GetMaxSizeFromNestedMaximumIterations(
                value, self.forward_context)
          acc = gen_data_flow_ops.stack_v2(
              max_size=max_size, elem_type=value.dtype.base_dtype, name="f_acc")
        if curr_ctxt:
          curr_ctxt.Exit()

        # Make acc available in the forward context.
        enter_acc = self.forward_context.AddValue(acc)

        # Add the stack_push op in the context of value.op.
        swap_enabled = self.forward_context.swap_memory
        value_ctxt = util.GetOutputContext(value.op)
        if value_ctxt == self.forward_context:
          # value is not nested in the forward context.
          self.forward_context.Enter()
          push = gen_data_flow_ops.stack_push_v2(
              enter_acc, value, swap_memory=swap_enabled)
          self.forward_context.Exit()
          # Protect stack push and order it before forward_index.
          self.forward_index.op._add_control_input(push.op)
        else:
          # value is in a cond context within the forward context.
          if not isinstance(value_ctxt, control_flow_ops.CondContext):
            raise TypeError("value_ctxt is not a CondContext: %s" % value_ctxt)
          if dead_branch:
            # The special case for creating a zero tensor for a dead
            # branch of a switch. See _ControlFlowState.ZerosLikeV1WhileLoop().
            value_ctxt.outer_context.Enter()
            push = gen_data_flow_ops.stack_push_v2(
                enter_acc, value, swap_memory=swap_enabled)
            value_ctxt.outer_context.Exit()
            push.op._set_control_flow_context(value_ctxt)
          else:
            value_ctxt.Enter()
            push = gen_data_flow_ops.stack_push_v2(
                enter_acc, value, swap_memory=swap_enabled)
            value_ctxt.Exit()
          # Protect stack push and order it before forward_sync.
          self.forward_sync._add_control_input(push.op)
        # Order stack push after the successor of forward_index
        add_op = self.forward_index.op.inputs[0].op
        push.op._add_control_input(add_op)
        return acc

  def AddBackpropAccumulatedValue(self, history_value, value,
                                  dead_branch=False):
    # [FUNCTION DOC]
    # Adds, inside the backprop loop, the pop that reads back an accumulated forward value.
    # Inputs: the history stack, the original forward tensor, and a dead-branch flag (default
    # false). If the value is controlled by a cond context, guards the stack with a switch on the
    # cond predicate (using an accumulated predicate from this or an outer loop's history map if
    # one exists, and the opposite branch for the dead-branch case). Pops from the stack with the
    # value's dtype and copies the value's static shape. When the loop's parallel iterations exceed
    # one, orders the pop before the gradient sync node. Returns the popped tensor (the value for
    # the current backprop iteration). Side effects: adds ops and control edges in the grad context.
    """Add the getter for an accumulated value in the grad context.

    This is added to the backprop loop. Called in the grad context to
    get the value of an accumulated value. The stack pop op must be guarded
    by the pred of the controlling cond.

    Args:
      history_value: The history (a stack) of a value.
      value: The value that is pushed onto the stack.
      dead_branch: True iff the tensor is on a dead branch of a cond.

    Returns:
      The current value (the top of the stack).
    """
    history_ctxt = history_value.op._get_control_flow_context()
    # Find the cond context that controls history_value if any.
    cond_ctxt = None
    value_ctxt = value.op._get_control_flow_context()
    while value_ctxt and value_ctxt != history_ctxt:
      if isinstance(value_ctxt, control_flow_ops.CondContext):
        cond_ctxt = value_ctxt
        break
      value_ctxt = value_ctxt.outer_context
    with ops.control_dependencies(None):
      self.grad_context.Enter()
      if cond_ctxt:
        # Guard stack pop with a switch if it is controlled by a cond.
        grad_state = self
        pred = None
        while pred is None and grad_state:
          pred = grad_state.history_map.get(cond_ctxt.pred.name)
          grad_state = grad_state.outer_grad_state
        if pred is None:
          pred = cond_ctxt.pred
        branch = (1 - cond_ctxt.branch) if dead_branch else cond_ctxt.branch
        history_value = control_flow_ops._SwitchRefOrTensor(
            history_value, pred)[branch]
      pop = gen_data_flow_ops.stack_pop_v2(history_value,
                                           value.dtype.base_dtype)
      pop.set_shape(value.get_shape())
      self.grad_context.Exit()
    parallel_iterations = self.grad_context.parallel_iterations
    if parallel_iterations > 1:
      # All pops are ordered after pivot_for_body and before grad_sync.
      self.grad_sync._add_control_input(pop.op)
    return pop

  def GetRealValue(self, value):
    # [FUNCTION DOC]
    # Returns the tensor that backprop should use in place of a forward-loop tensor. Must be called
    # in the gradient loop context; the value must be a forward tensor needed by backprop and must
    # not come from a variable op (asserted). Looks first in the history map. Otherwise: if the
    # value comes (possibly through switch/identity) from a constant loop enter, climbs outward
    # through outer gradient states, and if it exits all of them treats it as loop invariant and
    # just brings it into the gradient context; if the value is a constant, clones it as a new
    # constant; else creates a forward accumulator (temporarily leaving the gradient context) and
    # a matching pop in the gradient context, re-importing it when it came from an outer state.
    # The result is cached in the history map under the value's name. Returns that tensor.
    # Side effects: adds ops and updates the history map.
    """Get the real value of `value`.

    If backprop "uses" a value produced by forward inference, an accumulator
    is added in the forward loop to accumulate its values.  We use the
    accumulated value. This method must be called in the grad loop context.
    `value` must be in forward and needed for backprop.

    Args:
      value: A tensor to be captured.

    Returns:
      The same tensor obtained from the saved history.
    """
    assert value.op.type not in ["Variable", "VariableV2"]
    real_value = self._history_map.get(value.name)
    if real_value is None:
      cur_value = value
      cur_grad_state = self
      while True:
        enter_op = util.GetLoopConstantEnter(cur_value)
        if enter_op:
          # Special case: cur_value comes from a constant Enter node.
          cur_value = enter_op.inputs[0]
          cur_grad_state = cur_grad_state.outer_grad_state
          if cur_grad_state is None:
            # We are now outside all nested loops for this gradient(),
            # so `value` is a loop invariant and there is no need to
            # save the history of value. Just make cur_value to enter
            # the right control flow context.
            real_value = self._grad_context.AddValue(cur_value)
            break
        elif constant_op.is_constant(cur_value):
          # If the value to be forwarded is a constant, clone the constant in
          # the gradient loop rather than using a stack.
          # TODO(phawkins): consider hoisting the constant out of the loop
          # instead.
          real_value = constant_op.constant(
              tensor_util.constant_value(cur_value), dtype=cur_value.dtype)
          break
        else:
          # Record the history of this value in forward_ctxt.
          self._grad_context.Exit()
          history_value = cur_grad_state.AddForwardAccumulator(cur_value)
          self._grad_context.Enter()
          break

      if real_value is None:
        # Add the stack pop op in the grad context.
        real_value = cur_grad_state.AddBackpropAccumulatedValue(
            history_value, cur_value)
        if cur_grad_state != self:
          real_value = self._grad_context.AddValue(real_value)
      self._history_map[value.name] = real_value
    return real_value


class _ControlFlowState:
  # [CLASS DOC]
  # HELPER: internal manager used by the graph-mode gradient computation when while loops are
  # involved. Holds one mapping from each forward while context to its gradient loop state. Created
  # on demand by the helper that scans the ops between the outputs and inputs; offers methods to
  # look up the gradient state of an op, enter and exit gradient loop contexts, register newly
  # found loops, build zero gradients for exits and loop ops, and patch the finished gradient graph.
  """Maintain the mapping from the loops to their grad states."""

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor. Takes no arguments and starts with an empty mapping from forward loop contexts
    # to gradient loop states. Returns nothing.
    self._map = {}  # maps forward loop context to _GradLoopState

  def GetGradState(self, op: ops.Operation, before):
    # [FUNCTION DOC]
    # Finds the gradient loop state for the loop an operation belongs to. Inputs: the operation and
    # a flag "before". When the flag is true and the op is a loop exit, the loop is the one that
    # encloses the exit's outer context (the exit's own loop is skipped); otherwise it is the
    # while context of the op itself. Returns the registered gradient state, or null if the op is
    # not in a registered loop. No side effects.
    """Return the grad state for this op if it's in a forward loop context."""
    if before and util.IsLoopExit(op):
      forward_ctxt = op._get_control_flow_context()  # pylint: disable=protected-access
      forward_ctxt = forward_ctxt.outer_context
      if forward_ctxt:
        forward_ctxt = forward_ctxt.GetWhileContext()
    else:
      forward_ctxt = util.GetWhileContext(op)
    if forward_ctxt:
      return self._map.get(forward_ctxt)
    return None

  def ProcessUnusedLoopExits(self, pending_count, to_ops_set):
    # [FUNCTION DOC]
    # Finds loop exits that nothing depends on but that must still be backpropagated. Inputs: a
    # mapping from op to the count of pending backprop inputs, and the set of ops that produce the
    # requested outputs. For each loop and each forward exit whose pending count is zero, decrements
    # the loop's pending-exit counter and records the exit as unused when it is not a requested
    # output; once the counter reaches zero, the loop's unused exits are added to the result. Also
    # sets the pending count of loop enter ops with zero count to one (needed for higher-order
    # gradients). Returns the list of unused exits to backprop (may contain repeats if called for
    # several loops). Side effects: mutates the loop states and the pending-count mapping.
    """Process all the "unused" loop exits.

    The "unused" exits of the loops are added to `unused_exits`. An exit is
    unused if its pending_count is 0. If there is an exit with real gradient,
    all these deferred exits will enter the backprop loop with zero gradient.
    Otherwise, they will enter the backprop loop with None. As an example,
    people often write:

    ```python
    v1, _ = tf.while_loop(p, b, [x1, x2])
    result = gradients(v1, x1)
    ```

    The exit node for x2 is not included by the betweenness analysis. But we
    need to backprop x2 if x2 is involved in computing v1.

    Args:
      pending_count: The number of backprop inputs for every op.
      to_ops_set: The set of ops for ys in gradients(ys, xs)

    Returns:
      The set of unused loop exits that we know at this point we need
      to backprop.
    """
    loop_exits = []
    for grad_state in self._map.values():
      for y in grad_state.forward_loop_exits:
        if pending_count[y.op] == 0:
          grad_state.pending_exits_count -= 1
          if y.op not in to_ops_set:
            grad_state.unused_exits.append(y)
          if grad_state.pending_exits_count == 0:
            loop_exits.extend(grad_state.unused_exits)
      # Need to include Enters in backprop for higher-order gradients.
      for y in grad_state.forward_context.loop_enters:
        if pending_count[y.op] == 0:
          pending_count[y.op] = 1
    return loop_exits

  def EnterGradWhileContext(self, op, before):
    # [FUNCTION DOC]
    # Enters the backprop while context of the loop containing the op, if any. Inputs: the op and
    # the "before" flag as in the lookup method. Returns nothing. Side effect: changes the graph's
    # current control flow context; must be paired with the exit method.
    """Enter the WhileContext for gradient computation."""
    grad_state = self.GetGradState(op, before)
    if grad_state:
      grad_state.grad_context.Enter()

  def ExitGradWhileContext(self, op, before):
    # [FUNCTION DOC]
    # Leaves the backprop while context of the loop containing the op, if any. Inputs: the op and
    # the "before" flag as in the lookup method. Returns nothing. Side effect: restores the graph's
    # control flow context to the enclosing one.
    """Exit the WhileContext for gradient computation."""
    grad_state = self.GetGradState(op, before)
    if grad_state:
      grad_state.grad_context.Exit()

  def AddWhileContext(self, op, between_op_list, between_ops):
    # [FUNCTION DOC]
    # Registers gradient state for the while loop that a loop-exit op belongs to. Must be called in
    # the context where gradients are computed. Inputs: the exit op, the ordered list of ops
    # between outputs and inputs, and the set of those ops. If the loop is new, finds the
    # enclosing loop's gradient state (if any), creates a gradient loop state for it and stores
    # it, then appends every exit of that loop not yet present to both the list and the set.
    # Returns nothing. Side effects: mutates the two collections and the internal mapping, and
    # adds counter ops to the graph.
    """Add the grad state for the while loop that op belongs to.

    Note that op is an Exit, and this method must be called in
    the control flow context where gradients() is called.

    Note that this method modifies `between_op_list` and `between_ops`.
    """
    forward_ctxt = util.GetWhileContext(op)
    grad_state = self._map.get(forward_ctxt)
    if grad_state is None:
      # This is a new while loop so create a grad state for it.
      outer_forward_ctxt = forward_ctxt.outer_context
      if outer_forward_ctxt:
        outer_forward_ctxt = outer_forward_ctxt.GetWhileContext()
      outer_grad_state = None
      if outer_forward_ctxt:
        outer_grad_state = self._map.get(outer_forward_ctxt)
      grad_state = _GradLoopState(forward_ctxt, outer_grad_state)
      self._map[forward_ctxt] = grad_state

      # We need to include all exits of a loop for backprop.
      for loop_exit in grad_state.forward_loop_exits:
        if loop_exit.op not in between_ops:
          between_ops.add(loop_exit.op)
          between_op_list.append(loop_exit.op)

  def ZerosLikeForExit(self, val):
    # [FUNCTION DOC]
    # Creates a zero gradient for an unused loop exit. Input: the output tensor of an exit op; must
    # be called where gradients are computed. For a nested loop with fully known shape, creates
    # zeros of that shape in the outer gradient context; with unknown shape, records the runtime
    # shape through the outer loop's accumulator and recreates zeros from the popped shape in the
    # outer gradient context. For an outermost loop, creates zeros from the static shape if known,
    # else a zeros-like of the value without optimization. Returns the zero tensor. Side effects:
    # adds ops and may add accumulators; enters and exits contexts.
    """Create zeros_like gradient for a loop exit.

    If the result of a loop variable is not used but is involved in
    computing the result of some needed loop variable, we create a
    zero-valued tensor that is fed as gradient for the Exit node of that
    loop variable. Note that val.op is an Exit, and this method must be
    called in the control flow context where gradients() is called.

    Args:
      val: The output tensor of an Exit op.

    Returns:
      A zero tensor of the same shape of val.
    """
    val_shape = val.get_shape()
    forward_ctxt = val.op._get_control_flow_context()
    outer_forward_ctxt = forward_ctxt.outer_context
    if outer_forward_ctxt:
      outer_forward_ctxt = outer_forward_ctxt.GetWhileContext()
    outer_grad_state = None
    if outer_forward_ctxt:
      outer_grad_state = self._map.get(outer_forward_ctxt)
    if outer_grad_state:
      # This is a nested loop.
      if val_shape.is_fully_defined():
        # If the shape is known statically, just create a zero tensor
        # with the right shape in the right context.
        outer_grad_state.grad_context.Enter()
        result = array_ops.zeros(val_shape.dims, val.dtype)
        outer_grad_state.grad_context.Exit()
      else:
        # Only the shape of value is needed for backprop.
        forward_ctxt.outer_context.Enter()
        shape = array_ops.shape_internal(val, optimize=False)
        forward_ctxt.outer_context.Exit()
        # Save the shape to a stack.
        history_shape = outer_grad_state.AddForwardAccumulator(shape)
        # Get the shape back from the stack.
        outer_grad_ctxt = outer_grad_state.grad_context
        outer_grad_ctxt.Enter()
        real_shape = outer_grad_state.AddBackpropAccumulatedValue(
            history_shape, shape)
        result = array_ops.zeros(real_shape, val.dtype)
        outer_grad_ctxt.Exit()
    else:
      # This is not a nested loop.
      if val_shape.is_fully_defined():
        # If the shape is known statically, just create a zero tensor
        # with the right shape.
        result = array_ops.zeros(val_shape.dims, val.dtype)
      else:
        result = array_ops.zeros_like(val, optimize=False)
    return result

  def ZerosLikeV1WhileLoop(self, op, index):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring promises a zero tensor, but it returns nothing (null) when the op
    # is a loop switch.
    # Creates zeros for one output of an op that may be in a graph-mode while loop; inside a loop
    # taking part in gradients it must be called in that loop's gradient context. Inputs: the op
    # and the output index. Returns null for loop switches; plain zeros-like when the graph is
    # building a function; the generic zeros helper when the op is not in a registered loop.
    # Otherwise, with static shape builds a constant (or zeros from the variable shape for
    # resource dtypes), guarded by a switch on the recorded predicate when the op is a cond switch.
    # With unknown shape, records the shape at runtime in a forward accumulator (using a special
    # switch for cond switches), pops it in the gradient loop and builds zeros of that shape.
    # Side effects: adds ops and accumulators, enters and exits contexts.
    """Create zeros_like for the specified output of an op.

    If op is in a while loop that is part of gradients(), this method
    must be called in its grad loop context.

    Args:
      op: A tensorflow operation.
      index: the index for a specific output of the op.

    Returns:
      A zero tensor of the same shape of op.outputs[index].
    """
    if util.IsLoopSwitch(op):
      return None
    if op.graph.building_function:
      # The optimization here is tricky to apply to functions
      return array_ops.zeros_like(op.outputs[index])
    dead_branch = util.IsSwitch(op)
    forward_ctxt = util.GetWhileContext(op)
    grad_state = self._map.get(forward_ctxt)
    if grad_state is None:
      # op is not in a while loop that is part of gradients().
      return ZerosLike(op, index)
    op_ctxt = op._get_control_flow_context()
    val = ops.convert_to_tensor(op.outputs[index], name="tensor")
    shape = val.get_shape()
    if shape.is_fully_defined():
      # If the shape is known statically, just create a zero tensor with
      # the right shape in the grad loop context.
      if val.dtype == dtypes.resource:
        result = array_ops.zeros(
            resource_variable_ops.variable_shape(val),
            dtype=default_gradient.get_zeros_dtype(val))
      else:
        result = constant_op.constant(0, shape=shape.dims, dtype=val.dtype)
      if dead_branch:
        # op is a cond switch. Guard the zero tensor with a switch.
        pred = grad_state.history_map.get(op_ctxt.pred.name)
        branch = op_ctxt.branch
        result = control_flow_ops._SwitchRefOrTensor(result, pred)[1 - branch]
    else:
      # Unknown shape so keep a history of the shape at runtime.
      if dead_branch:
        # Need to add a special switch to guard the value.
        pred = op_ctxt.pred
        branch = op_ctxt.branch
        op_ctxt.outer_context.Enter()
        val = control_flow_ops._SwitchRefOrTensor(op.inputs[0],
                                                  pred)[1 - branch]
        zeros_shape = array_ops.shape_internal(val, optimize=False)
        op_ctxt.outer_context.Exit()
        val.op._set_control_flow_context(op_ctxt)
        zeros_shape.op._set_control_flow_context(op_ctxt)
      else:
        op_ctxt.Enter()
        zeros_shape = array_ops.shape_internal(val, optimize=False)
        op_ctxt.Exit()

      # Add forward accumulator for shape.
      grad_state.grad_context.Exit()
      history_zeros_shape = grad_state.AddForwardAccumulator(
          zeros_shape, dead_branch=dead_branch)
      grad_state.grad_context.Enter()

      # Create a zero tensor with the right shape.
      shape = grad_state.AddBackpropAccumulatedValue(history_zeros_shape,
                                                     zeros_shape, dead_branch)
      result = array_ops.zeros(shape, val.dtype)
    return result

  def PostProcessing(self):
    # [FUNCTION DOC]
    # Final patch-up pass run after the gradient graph has been built. For every loop's gradient
    # merge whose two inputs are the same tensor (a loop variable whose next value does not depend
    # on its previous value), replaces the second input with a next-iteration of zeros so
    # iterations after the first receive zero gradient. Zeros are a constant when the shape is
    # static, else built from the shape of the merge's entry gradient in the outer gradient
    # context. Takes no arguments, returns nothing. Side effects: adds ops, rewires merge inputs.
    """Perform postprocessing at the end of gradients().

    We have created the gradient graph at this point. So this function
    can be used to perform any postprocessing on the gradient graph.
    We currently perform the following postprocessing:
      1. Patch the gradient graph if the output of a loop variable
         doesn't depend on its input.
    """
    for _, grad_state in self._map.items():
      for _, b_merge in grad_state.switch_map.items():
        if b_merge.op.inputs[0] == b_merge.op.inputs[1]:
          # The value of this loop variable at iteration i+1 doesn't
          # depend on its value at iteration i. So use zeros as the
          # gradients for all iterations > 0.
          dtype = b_merge.op.inputs[0].dtype
          shape = b_merge.op.inputs[0].get_shape()
          # pylint: disable=protected-access
          if shape.is_fully_defined():
            grad_state.grad_context.Enter()
            # Create a zeros and use it for iterations > 0.
            grad_val = constant_op.constant(0, dtype=dtype, shape=shape)
            next_grad_val = control_flow_ops._NextIteration(grad_val)
            grad_state.grad_context.Exit()
          else:
            # Create a zeros in the outer grad context.
            outer_grad_ctxt = grad_state.grad_context.outer_context
            if outer_grad_ctxt:
              outer_grad_ctxt.Enter()
            enter_grad_op = b_merge.op.inputs[0].op
            enter_grad = enter_grad_op.inputs[0]
            grad_shape = array_ops.shape_internal(enter_grad, optimize=False)
            grad_val = array_ops.zeros(grad_shape)
            if outer_grad_ctxt:
              outer_grad_ctxt.Exit()
            # Use the zeros for iterations > 0.
            grad_state.grad_context.Enter()
            next_grad_val = control_flow_ops._NextIteration(grad_val)
            grad_state.grad_context.Exit()
          b_merge.op._update_input(1, next_grad_val)
          # pylint: enable=protected-access


def MaybeCreateControlFlowState(between_op_list, between_ops,
                                colocate_gradients_with_ops):
  # [FUNCTION DOC]
  # Creates the loop gradient bookkeeping for one gradient computation, if any while loop is
  # involved. Inputs: the ordered list of ops between the outputs and inputs, the set of the same
  # ops, and a flag saying whether gradient ops are colocated with their forward ops. For every
  # loop exit in the list, registers its loop with a new state manager (created on first need),
  # under a colocation scope when the flag is true. Returns the manager, or null if there are no
  # loop exits. Side effects: mutates the list and set (exits get added) and adds graph ops.
  """Create the state for all the while loops involved in one gradients().

  We create a _ControlFlowState when there are while loops involved in
  gradients(). In gradients(), control flow logic is only invoked when
  the _ControlFlowState is not None.

  Note that this method modifies `between_op_list` and `between_ops`.
  """
  loop_state = None
  for op in between_op_list:
    if util.IsLoopExit(op):
      if loop_state is None:
        loop_state = _ControlFlowState()
      if colocate_gradients_with_ops:
        with ops.colocate_with(op):
          loop_state.AddWhileContext(op, between_op_list, between_ops)
      else:
        loop_state.AddWhileContext(op, between_op_list, between_ops)
  return loop_state


def _ZerosLikeV1(op, index):
  # [FUNCTION DOC]
  # HELPER: graph-mode (version 1) branch of the zeros-like gradient helper, for switch ops.
  # Inputs: an op and an output index. If the op is inside a cond context, creates zeros only on
  # the branch actually taken: switches the op's first input on the predicate, takes the opposite
  # branch, makes an identity pivot, and builds zeros (from the variable shape for resource
  # dtypes, else the runtime shape) under a control dependency on the pivot. Outside a cond,
  # returns a plain zeros-like of the output without optimization. Returns the zero tensor.
  """Branch of ZerosLike for TF1."""
  val = op.outputs[index]
  op_ctxt = op._get_control_flow_context()  # pylint: disable=protected-access
  if op_ctxt:
    # We are in a cond context. Use a switch to create zeros only when needed.
    pred = op_ctxt.pred
    branch = op_ctxt.branch
    switch_val = control_flow_ops.switch(op.inputs[0], pred)[1 - branch]
    # A op is created along the branch taken as control dependencies are on
    # the whole op and not on the tensor output.
    pivot = array_ops.identity(switch_val)
    if val.dtype == dtypes.resource:
      with ops.control_dependencies([pivot]):
        return array_ops.zeros(
            gen_resource_variable_ops.variable_shape(switch_val),
            dtype=default_gradient.get_zeros_dtype(val))
    zeros_shape = array_ops.shape_internal(switch_val, optimize=False)
    # Ensure ops created within array_ops.zeros are dominated by switch in
    # cond context.
    with ops.control_dependencies([pivot]):
      return array_ops.zeros(zeros_shape, dtype=val.dtype)
  else:
    return array_ops.zeros_like(val, optimize=False)


@array_ops._tag_zeros_tensor  # pylint: disable=protected-access
def _ConstantZeros(shape, dtype):
  # [FUNCTION DOC]
  # HELPER: builds the constant zeros used by the version 2 zeros-like helper in while bodies.
  # Inputs: a shape (list of dimensions) and a data type. Returns a constant tensor of zeros; the
  # decorator tags the result as a zeros tensor for later recognition. Adds a constant op.
  """Create a constant zero tensor."""
  return constant_op.constant(0, shape=shape, dtype=dtype)


def _ZerosLikeV2(op, index):
  # [FUNCTION DOC]
  # HELPER: version 2 branch of the zeros-like gradient helper.
  # Inputs: an op and an output index. For resource outputs returns zeros built from the variable
  # shape with the proper zero dtype. For non-variant outputs inside a while-body function graph it
  # avoids a zeros-like op (which would trigger accumulation): returns constant zeros when the
  # shape is fully known, else zeros built from a shape op. In every other case returns a plain
  # zeros-like without optimization. Returns the zero tensor. Adds ops to the graph.
  """Branch of ZerosLike for TF2."""
  val = op.outputs[index]
  if val.dtype == dtypes.resource:
    return array_ops.zeros(
        gen_resource_variable_ops.variable_shape(val),
        dtype=default_gradient.get_zeros_dtype(val))
  if (isinstance(val.op.graph, control_flow_v2_func_graphs.WhileBodyFuncGraph)
      and val.dtype != dtypes.variant):
    # In while_v2 we do not want to add a `ZerosLike` op because that will
    # trigger accumulation of `val`. Normally `ZerosLike` is preferred because
    # it helps avoid creating extra nodes(possibly Consts) for the shape.
    # For variants, we must use ZerosLike.
    if val.shape.is_fully_defined():
      return _ConstantZeros(val.shape.dims, val.dtype)
    else:
      # Note: Even though we add `Shape` in the default graph, while_v2 is smart
      # enough to place it in the forward graph i.e. `val.graph`.
      zeros_shape = array_ops.shape_internal(val, optimize=False)
      return array_ops.zeros(zeros_shape, val.dtype)
  else:
    return array_ops.zeros_like(val, optimize=False)


def ZerosLike(op, index):
  # [FUNCTION DOC]
  # Creates zeros shaped like one output of an op, used as a placeholder gradient. Inputs: the op
  # and the output index. Switch ops (graph-mode cond/loop primitives) use the version 1 branch;
  # all other ops use the version 2 branch. Returns the zero tensor. Adds ops to the graph.
  """Create zeros_like for the specified output of an op."""
  if not util.IsSwitch(op):
    return _ZerosLikeV2(op, index)
  else:
    return _ZerosLikeV1(op, index)
