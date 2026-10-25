# [MODULE DOC]
# Defines the function-graph subclasses used by version 2 control flow (functional cond and
# while_loop). A common base class marks graphs built for a control flow op, and three thin
# subclasses tag cond branches, while-loop conditions and while-loop bodies so other code can tell
# them apart from ordinary functions by type. No import-time side effects; not a helper module,
# but consumed by the v2 cond/while implementations.
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
"""FuncGraphs for V2 control flow."""

from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops


class ControlFlowFuncGraph(func_graph.FuncGraph):
  # [CLASS DOC]
  # Base function graph for control flow bodies. Extends the generic function graph by copying the
  # outer graph's device scope stack at construction (so outer device placement is baked into the
  # body), flagging itself as a control flow graph, and inheriting the outer name scope when eager
  # execution is active outside functions. Created once per control flow op by the v2 cond/while
  # builders.
  """Contains control flow-specific FuncGraph logic."""

  def __init__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # Constructor. Forwards all positional and keyword arguments to the function graph base
    # constructor, then copies the outer graph's device function stack into this graph, sets the
    # control-flow-graph marker to true, and, if eager execution is enabled outside functions,
    # overrides this graph's name scope with the outer graph's current name scope. Returns nothing.
    # Reads the outer graph, which must exist (the base constructor sets it).
    super(ControlFlowFuncGraph, self).__init__(*args, **kwargs)
    outer_graph = self.outer_graph
    # Unlike tf.function, control flow FuncGraphs are generally created one per
    # op. This means hard-coding any outer device scopes in the body (rather
    # than inspecting the call-time placement of the control flow op) makes
    # sense.
    self._device_function_stack = outer_graph._device_function_stack.copy()  # pylint: disable=protected-access
    self.is_control_flow_graph = True
    if ops.executing_eagerly_outside_functions():
      func_graph.override_func_graph_name_scope(
          self, self.outer_graph.get_name_scope())


class CondBranchFuncGraph(ControlFlowFuncGraph):
  # [CLASS DOC]
  # Marker subclass of the control flow function graph used for the true/false branches of a
  # version 2 cond. Adds no state or behavior; exists so branches can be recognized by type.
  """FuncGraph for branches of tf.cond().

  This is used to distinguish cond branches from other functions.
  """


class WhileCondFuncGraph(ControlFlowFuncGraph):
  # [CLASS DOC]
  # Marker subclass of the control flow function graph used for the loop condition of a version 2
  # while loop. Adds no state or behavior; exists so condition graphs can be recognized by type.
  """FuncGraph for the condition of tf.while_loop().

  This is used to distinguish while conditions from other functions.
  """


class WhileBodyFuncGraph(ControlFlowFuncGraph):
  # [CLASS DOC]
  # Marker subclass of the control flow function graph used for the loop body of a version 2
  # while loop. Adds no state or behavior; exists so body graphs can be recognized by type.
  """FuncGraph for the body of tf.while_loop().

  This is used to distinguish while bodies from other functions.
  """
