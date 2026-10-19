# [MODULE DOC]
# Experimental, test-only Python wrappers for a few math operations (add, matrix multiply,
# negate, subtract, multiply, log of one plus x, division returning zero when dividing by zero)
# built on the native unified-API bindings. Each wrapper takes the thread's default execution
# context and forwards the call to the native implementation, so the operation is run eagerly or
# traced depending on the context. Mirrors the generated math ops module.
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
"""Experimental impl for gen_math_ops.py using unified APIs, for testing."""

from tensorflow.python.framework.experimental import _math_ops
from tensorflow.python.framework.experimental import context_stack as context


def add(a, b, name=None):
  # [FUNCTION DOC]
  # Element-wise addition. Inputs: a and b, unified-API tensor handles; name, optional operation
  # name (default none). Returns the handle of the sum, created in the default context.
  ctx = context.get_default()
  return _math_ops.add(ctx, a, b, name)


def mat_mul(a, b, name=None):
  # [FUNCTION DOC]
  # Matrix multiplication. Inputs: a and b, unified-API tensor handles; name, optional operation
  # name (default none). Returns the handle of the product, created in the default context.
  ctx = context.get_default()
  return _math_ops.mat_mul(ctx, a, b, name)


def neg(a, name=None):
  # [FUNCTION DOC]
  # Element-wise negation. Inputs: a, a unified-API tensor handle; name, optional operation name
  # (default none). Returns the handle of the negated tensor, created in the default context.
  ctx = context.get_default()
  return _math_ops.neg(ctx, a, name)


def sub(a, b, name=None):
  # [FUNCTION DOC]
  # Element-wise subtraction (a minus b). Inputs: a and b, unified-API tensor handles; name,
  # optional operation name (default none). Returns the handle of the difference, created in the
  # default context.
  ctx = context.get_default()
  return _math_ops.sub(ctx, a, b, name)


def mul(a, b, name=None):
  # [FUNCTION DOC]
  # Element-wise multiplication. Inputs: a and b, unified-API tensor handles; name, optional
  # operation name (default none). Returns the handle of the product, created in the default
  # context.
  ctx = context.get_default()
  return _math_ops.mul(ctx, a, b, name)


def log1p(a, name=None):
  # [FUNCTION DOC]
  # Element-wise natural logarithm of one plus the input. Inputs: a, a unified-API tensor handle;
  # name, optional operation name (default none). Returns the result handle, created in the
  # default context.
  ctx = context.get_default()
  return _math_ops.log1p(ctx, a, name)


def div_no_nan(a, b, name=None):
  # [FUNCTION DOC]
  # Element-wise division that yields zero where the divisor is zero. Inputs: a (numerator) and b
  # (denominator), unified-API tensor handles; name, optional operation name (default none).
  # Returns the quotient handle, created in the default context.
  ctx = context.get_default()
  return _math_ops.div_no_nan(ctx, a, b, name)
