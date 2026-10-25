# [MODULE DOC]
# HELPER: gradient-registration module supporting the roll op in manip_ops.
# Registers the gradient function for the Roll operation, whose forward op is defined in
# manip_ops. Importing this module registers the gradient globally; it exports nothing else.
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
"""Gradients for operators defined in manip_ops.py."""

from tensorflow.python.framework import ops
from tensorflow.python.ops import manip_ops


@ops.RegisterGradient("Roll")
def _RollGrad(op, grad):
  # [FUNCTION DOC]
  # HELPER: gradient registered for the Roll op, used by automatic differentiation only.
  # Computes the gradient of a roll by rolling the incoming gradient by the negated shift along
  # the same axis. Inputs: op, the forward Roll operation (its second and third inputs are the
  # shift and axis); grad, the gradient tensor flowing back. Returns a tuple of the rolled
  # gradient for the data input and nothing (no gradient) for shift and axis. No side effects
  # other than adding roll ops to the current graph.
  # The gradient is just the roll reversed
  shift = op.inputs[1]
  axis = op.inputs[2]
  roll_grad = manip_ops.roll(grad, -shift, axis)
  return roll_grad, None, None
