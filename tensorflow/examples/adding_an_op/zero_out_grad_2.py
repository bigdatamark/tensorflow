# [MODULE DOC]
# Registers the gradient function for the tutorial ZeroOut custom op so that TensorFlow can
# differentiate through it. Importing the module has the side effect of registering the gradient
# under the op name ZeroOut in the global gradient registry. Part of the adding-an-op tutorial.
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

"""The gradient of the tutorial zero_out op."""

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import sparse_ops


@ops.RegisterGradient("ZeroOut")
def _zero_out_grad(op, grad):
  # [FUNCTION DOC]
  # HELPER: gradient registration glue for the ZeroOut op, called by TensorFlow autodiff.
  # Computes the gradient of zero_out with respect to its input. Parameters: op, the original
  # ZeroOut operation (its first input is used to get the shape); grad, the gradient tensor with
  # respect to the op output. Builds a dense tensor of the input shape that is zero everywhere
  # except at the all-zeros (first) index, which receives the first element of the flattened
  # incoming gradient. Returns a list holding that single tensor (one per op input). No side
  # effects other than adding graph operations.
  """The gradients for `zero_out`.

  Args:
    op: The `zero_out` `Operation` that we are differentiating, which we can use
      to find the inputs and outputs of the original op.
    grad: Gradient with respect to the output of the `zero_out` op.

  Returns:
    Gradients with respect to the input of `zero_out`.
  """
  to_zero = op.inputs[0]
  shape = array_ops.shape(to_zero)
  index = array_ops.zeros_like(shape)
  first_grad = array_ops.reshape(grad, [-1])[0]
  to_zero_grad = sparse_ops.sparse_to_dense([index], shape, first_grad, 0)
  return [to_zero_grad]  # List of one Tensor, since we have one input
