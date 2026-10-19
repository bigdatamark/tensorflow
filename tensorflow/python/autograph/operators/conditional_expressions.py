# [MODULE DOC]
# AutoGraph operator for conditional (ternary) expressions. Exposes if_exp, which stages a graph
# conditional when the condition is a dense tensor and otherwise evaluates as normal Python.
# Depends on the control flow operators module for branch type verification. No import-time side
# effects.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Conditional expressions (e.g. the ternary if statement)."""


from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.ops import cond as tf_cond


def if_exp(cond, if_true, if_false, expr_repr):
  # [FUNCTION DOC]
  # Evaluates a conditional expression. Inputs: cond (condition; a dense tensor or a Python
  # value), if_true and if_false (callbacks with no arguments producing each branch value),
  # expr_repr (text name of the expression used in error messages). A dense tensor condition
  # dispatches to the staged tensor version; anything else to the plain version. Returns the
  # selected branch value.
  if tensors.is_dense_tensor(cond):
    return _tf_if_exp(cond, if_true, if_false, expr_repr)
  else:
    return _py_if_exp(cond, if_true, if_false)


def _tf_if_exp(cond, if_true, if_false, expr_repr):
  # [FUNCTION DOC]
  # HELPER: tensor-condition implementation behind the conditional expression operator.
  # Stages a graph conditional whose branches call the given callbacks, remembering each branch's
  # result; once both branches have run, checks that they agree in type via the control flow
  # verification helper (which raises on mismatch, naming expr_repr). Returns the conditional's
  # output tensor(s).
  """Overload of if_exp that stages a TF cond."""
  # TODO(mdan): Use nonlocal once we no longer need to support py2.
  true_val = []
  false_val = []

  def true_fn():
    # [FUNCTION DOC]
    # HELPER: true branch of the staged conditional. Runs the true callback, records the result,
    # verifies it against the false result if already available, and returns the recorded value.
    true_val.append(if_true())
    if true_val and false_val:
      control_flow.verify_single_cond_var(expr_repr, true_val[0], false_val[0])
    return true_val[0]

  def false_fn():
    # [FUNCTION DOC]
    # HELPER: false branch of the staged conditional. Runs the false callback, records the result,
    # verifies it against the true result if already available, and returns the recorded value.
    false_val.append(if_false())
    if true_val and false_val:
      control_flow.verify_single_cond_var(expr_repr, true_val[0], false_val[0])
    return false_val[0]

  return tf_cond.cond(cond, true_fn, false_fn)


def _py_if_exp(cond, if_true, if_false):
  # [FUNCTION DOC]
  # HELPER: plain-Python implementation of the conditional expression operator. Calls and returns
  # the result of only the true callback if cond is truthy, otherwise only the false callback.
  return if_true() if cond else if_false()
