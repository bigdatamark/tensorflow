# [MODULE DOC]
# Builds the context information that tf.function mixes into its tracing cache key, so that a
# function is retraced when conditions other than the arguments change: enclosing graph, device
# placement, colocation, distribution strategy replica context, saving variable policy and XLA
# control flow context. Exposes the EagerContext record type and make_function_context.
# Not a pure helper module but used internally by the polymorphic function machinery.
# No import-time side effects beyond imports.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Context information for a tf.function."""

from typing import NamedTuple, Any

from tensorflow.core.function.polymorphism import function_cache
from tensorflow.python.eager import context
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.saved_model import save_context


# EagerContext is used by tf.function to identify cases where tracing
# needs to occur due to a change in conditions other than the arguments.
class EagerContext(NamedTuple):
  # [CLASS DOC]
  # Immutable record of ambient conditions that affect tracing: the parent graph (nothing when
  # executing eagerly), device functions, colocation stack, whether in a cross-replica
  # context, the saved-model variable policy, and an identifier of the enclosing XLA context
  # (zero if none). Compared by value inside the cache key so differing conditions force a
  # retrace. Created only by make_function_context.
  parent_graph: Any
  device_functions: Any
  colocation_stack: Any
  in_cross_replica_context: Any
  variable_policy: Any
  xla_context_id: Any


def make_function_context(scope_type=None) -> function_cache.FunctionContext:
  # [FUNCTION DOC]
  # Captures the current execution environment into a function-cache context object.
  # Input: scope_type, optional, an opaque scope marker stored alongside (default nothing).
  # Behavior: if not executing eagerly, records a unique id for an enclosing XLA context that
  # requires unique retracing and, inside an init scope, the default graph as parent graph
  # (nothing if the init scope is eager). Reads the distribution strategy stack to decide
  # whether device functions must be part of the key; in eager mode records the current device
  # name, in graph mode the colocation stack and outer-to-inner device functions when a
  # strategy or callable device is active. Detects cross-replica context (false if the
  # strategy stack is empty or lacks a replica context) and the variable policy when inside a
  # save context. Returns a function context wrapping an EagerContext and scope_type.
  # No exceptions expected; no mutation of global state.
  """Generates a FunctionContext based on current contextual info."""
  ctx = context.context()

  # Don't need to open an init_scope if the tf.function call is in eager mode
  # already.
  executing_eagerly = ctx.executing_eagerly()
  parent_graph = None
  xla_context_id = 0
  if not executing_eagerly:
    # We want to force function retracing for each different
    # XLAControlFlowContext, so add `xla_context_id` to the context.
    xla_context = _enclosing_xla_context()
    if xla_context is not None and xla_context.RequiresUniqueFunctionRetracing(
    ):
      xla_context_id = id(xla_context)

    with ops.init_scope():
      # The graph, or whether we're executing eagerly, should be a part of the
      # cache key so we don't improperly capture tensors such as variables.
      executing_eagerly = ctx.executing_eagerly()
      parent_graph = None if executing_eagerly else ops.get_default_graph()

  # pylint: disable=protected-access
  default_graph = ops.get_default_graph()
  # TODO(b/117617952): The current distribution strategy will affect graph
  # building (e.g. accessing different variables from different devices) and
  # so requires retracing for each device.
  strategy_stack = default_graph._distribution_strategy_stack
  uses_distribution_strategy = (
      strategy_stack and
      strategy_stack[-1].strategy.extended._retrace_functions_for_each_device)
  if executing_eagerly:
    colocation_stack = ()
    if uses_distribution_strategy:
      device_functions = (pydev.merge_device(ctx.device_name),)
    else:
      device_functions = ()
  else:
    colocation_stack = tuple(default_graph._colocation_stack.peek_objs())
    if (uses_distribution_strategy or
        func_graph_module.device_stack_has_callable(
            default_graph._device_function_stack)):
      # Putting the device in the cache key ensures that call-site device
      # annotations are respected.
      device_functions = tuple(default_graph._device_functions_outer_to_inner)
    else:
      device_functions = ()

  in_cross_replica_context = False
  try:
    in_cross_replica_context = (strategy_stack[-1].replica_context is None)  # pylint: disable=protected-access
  except (AttributeError, IndexError):
    pass

  if save_context.in_save_context():
    variable_policy = (
        save_context.get_save_options().experimental_variable_policy)
  else:
    variable_policy = None

  return function_cache.FunctionContext(
      EagerContext(
          parent_graph,
          device_functions,
          colocation_stack,
          in_cross_replica_context,
          variable_policy,
          xla_context_id,
      ),
      scope_type,
  )


def _enclosing_xla_context():
  # [FUNCTION DOC]
  # HELPER: used by make_function_context to detect XLA compilation scopes.
  # Walks from the default graph through its control flow contexts (and their outer contexts),
  # then to the outer graph for function graphs, and returns the first XLA control flow context
  # found, or nothing if there is none. Takes no inputs and has no side effects.
  """Returns the XLAControlFlowContext, which exists inside a tpu.rewrite()."""
  graph = ops.get_default_graph()
  while graph is not None:
    # pylint: disable=protected-access
    context_ = graph._get_control_flow_context()
    # pylint: enable=protected-access
    while context_ is not None:
      if isinstance(context_, control_flow_ops.XLAControlFlowContext):
        return context_
      context_ = context_.outer_context
    # This may be a FuncGraph due to defuns or v2 control flow. We need to
    # find the original graph with the XLAControlFlowContext.
    graph = getattr(graph, "outer_graph", None)
  return None
