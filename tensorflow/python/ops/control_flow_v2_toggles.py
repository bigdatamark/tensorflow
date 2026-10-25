# [MODULE DOC]
# Public (v1 only) API for switching version 2 control flow on and off and for querying the current
# state. Exposes three exported functions: enable, disable and is-enabled. The enable and disable
# functions update a usage gauge and the module-level flag in the control flow utilities module.
# No import-time side effects other than export registration through the export decorator.
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

"""API for enabling v2 control flow."""

from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["enable_control_flow_v2"])
def enable_control_flow_v2():  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # Opts in to version 2 control flow globally. Takes no arguments and returns nothing. Side
  # effects: writes a verbose-level log message, sets the control flow API usage gauge to true, and
  # sets the module-level control flow v2 flag in the control flow utilities module to true. Never
  # raises. Graph code always uses v2 inside functions regardless of this flag.
  """Use control flow v2.

  control flow v2 (cfv2) is an improved version of control flow in TensorFlow
  with support for higher order derivatives. Enabling cfv2 will change the
  graph/function representation of control flow, e.g., `tf.while_loop` and
  `tf.cond` will generate functional `While` and `If` ops instead of low-level
  `Switch`, `Merge` etc. ops. Note: Importing and running graphs exported
  with old control flow will still be supported.

  Calling tf.enable_control_flow_v2() lets you opt-in to this TensorFlow 2.0
  feature.

  Note: v2 control flow is always enabled inside of tf.function. Calling this
  function is not required.
  """
  # pylint: disable=protected-access
  logging.vlog(1, "Enabling control flow v2")
  ops._control_flow_api_gauge.get_cell().set(True)
  control_flow_util.ENABLE_CONTROL_FLOW_V2 = True


@tf_export(v1=["disable_control_flow_v2"])
def disable_control_flow_v2():  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # Opts out of version 2 control flow globally. Takes no arguments and returns nothing. Side
  # effects: writes a verbose-level log message, sets the control flow API usage gauge to false,
  # and sets the module-level control flow v2 flag in the control flow utilities module to false.
  # Has no effect on code built inside functions, which always uses v2. Never raises.
  """Opts out of control flow v2.

  Note: v2 control flow is always enabled inside of tf.function. Calling this
  function has no effect in that case.

  If your code needs tf.disable_control_flow_v2() to be called to work
  properly please file a bug.
  """
  # pylint: disable=protected-access
  logging.vlog(1, "Disabling control flow v2")
  ops._control_flow_api_gauge.get_cell().set(False)
  control_flow_util.ENABLE_CONTROL_FLOW_V2 = False


@tf_export(v1=["control_flow_v2_enabled"])
def control_flow_v2_enabled():  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # Reports whether version 2 control flow is in effect for the current default graph. Takes no
  # arguments; returns a boolean obtained by asking the control flow utilities whether v2 is
  # enabled for the default graph (true inside function graphs or when the global flag is on).
  # No side effects.
  """Returns `True` if v2 control flow is enabled.

  Note: v2 control flow is always enabled inside of tf.function.
  """
  return control_flow_util.EnableControlFlowV2(ops.get_default_graph())
