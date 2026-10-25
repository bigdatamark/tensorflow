# [MODULE DOC]
# Registers gradient functions for the optional-value ops (creating an optional from values and
# reading values from an optional). Import-time side effect: registration in the global gradient
# registry. Exposes nothing else.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Gradient functions for optional ops."""

from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_optional_ops


@ops.RegisterGradient("OptionalFromValue")
def _OptionalFromValueGrad(op, grad):
  # [FUNCTION DOC]
  # HELPER: registered gradient for the optional-from-value op.
  # Inputs: op, the forward op (its inputs give the dtypes and shapes); grad, the incoming
  # gradient, an optional value. Output: the values unpacked from the gradient optional, typed and
  # shaped like the op's inputs. Delegates to the get-value op.
  return gen_optional_ops.optional_get_value(
      grad, [t.dtype for t in op.inputs], [t.shape for t in op.inputs]
  )


@ops.RegisterGradient("OptionalGetValue")
def _OptionalGetValueGrad(unused_op, *grads):
  # [FUNCTION DOC]
  # HELPER: registered gradient for the optional-get-value op.
  # Inputs: the forward op (unused) and one gradient per output value. Output: a single optional
  # value wrapping all the gradients, as the gradient with respect to the optional input.
  return gen_optional_ops.optional_from_value(grads)
