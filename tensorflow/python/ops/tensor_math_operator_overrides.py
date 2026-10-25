# [MODULE DOC]
# Installs the arithmetic, logical and comparison operator overloads on the Tensor class. It
# defines small factory functions that lazily import the math ops module (to avoid circular
# imports) and forward to the matching math function, then registers them through the
# binary-operator override helper for add, and, div, floordiv, matmul, mod, mul, or, pow, sub,
# truediv and xor. It also overrides invert, negation, absolute value, the ordering comparisons
# (with automatic dtype promotion) and equality/inequality.
# Import-time side effect: mutates the Tensor class by registering all of these operators.
# Helper/glue module; nothing here is meant to be called directly except by tests.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Overrides for Tensor operators."""


from tensorflow.python.framework import override_binary_operator
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.util import tf_decorator


# pylint: disable=g-import-not-at-top
def _add_dispatch_factory(x, y, name=None):
  # [FUNCTION DOC]
  # HELPER: glue for the Tensor addition operator override.
  # Lazily imports the math ops module and returns the result of its addition dispatcher applied
  # to operands x and y with optional op name. Errors propagate from the dispatcher.
  from tensorflow.python.ops import math_ops

  return math_ops._add_dispatch(x, y, name=name)  # pylint: disable=protected-access


def _and_factory(x, y, name=None):
  # [FUNCTION DOC]
  # HELPER: glue for the Tensor bitwise/logical-and operator override.
  # Lazily imports the math ops module and returns the logical-and of x and y with optional op
  # name. Errors propagate from that function.
  from tensorflow.python.ops import math_ops

  return math_ops.and_(x, y, name=name)


def _div_factory(x, y, name=None):
  # [FUNCTION DOC]
  # HELPER: glue for the Tensor division operator override (legacy python-2 style division).
  # Lazily imports the math ops module and returns its div result for x and y with optional op
  # name. Errors propagate.
  from tensorflow.python.ops import math_ops

  return math_ops.div(x, y, name=name)


def _floordiv_factory(x, y, name=None):
  # [FUNCTION DOC]
  # HELPER: glue for the Tensor floor-division operator override.
  # Lazily imports the math ops module and returns floor division of x by y with optional op
  # name. Errors propagate.
  from tensorflow.python.ops import math_ops

  return math_ops.floordiv(x, y, name=name)


def _matmul_factory(a, b, name=None):
  # [FUNCTION DOC]
  # HELPER: glue for the Tensor matrix-multiplication operator override.
  # Lazily imports the math ops module and returns the result of its matmul wrapper for the
  # operands a and b with optional op name. Errors propagate.
  from tensorflow.python.ops import math_ops

  return math_ops.matmul_wrapper(a, b, name=name)


def _mod_factory(x, y, name=None):
  # [FUNCTION DOC]
  # HELPER: glue for the Tensor modulo operator override.
  # Lazily imports the math ops module and returns x modulo y with optional op name. Errors
  # propagate.
  from tensorflow.python.ops import math_ops

  return math_ops.mod(x, y, name=name)


def _mul_dispatch_factory(x, y, name=None):
  # [FUNCTION DOC]
  # HELPER: glue for the Tensor multiplication operator override.
  # Lazily imports the math ops and dtype modules. If either operand is a Tensor of boolean
  # dtype, both operands are cast to 32-bit integers, multiplied by the multiplication
  # dispatcher, and the product is cast back to boolean (so true times true is true, i.e.
  # logical and). Otherwise returns the multiplication dispatcher result for x and y directly.
  # Optional name is passed to the op. Errors propagate from the casts or dispatcher.
  from tensorflow.python.ops import math_ops
  from tensorflow.python.framework import dtypes

  if (isinstance(x, tensor_lib.Tensor) and x.dtype == dtypes.bool) or (
      isinstance(y, tensor_lib.Tensor) and y.dtype == dtypes.bool
  ):
    return gen_math_ops.cast(
        math_ops._mul_dispatch(
            gen_math_ops.cast(x, dtypes.int32),
            gen_math_ops.cast(y, dtypes.int32),
            name=name,
        ),
        dtypes.bool,
    )  # pylint: disable=protected-access
  return math_ops._mul_dispatch(x, y, name=name)  # pylint: disable=protected-access


def _or_factory(x, y, name=None):
  # [FUNCTION DOC]
  # HELPER: glue for the Tensor or operator override.
  # Lazily imports the math ops module and returns the logical-or of x and y with optional op
  # name. Errors propagate.
  from tensorflow.python.ops import math_ops

  return math_ops.or_(x, y, name=name)


def _pow_factory(x, y, name=None):
  # [FUNCTION DOC]
  # HELPER: glue for the Tensor power operator override.
  # Lazily imports the math ops module and returns x raised to y with optional op name. Errors
  # propagate.
  from tensorflow.python.ops import math_ops

  return math_ops.pow(x, y, name=name)


def _subtract_factory(x, y, name=None):
  # [FUNCTION DOC]
  # HELPER: glue for the Tensor subtraction operator override.
  # Lazily imports the math ops module and returns x minus y with optional op name. Errors
  # propagate.
  from tensorflow.python.ops import math_ops

  return math_ops.subtract(x, y, name=name)


def _truediv_factory(x, y, name=None):
  # [FUNCTION DOC]
  # HELPER: glue for the Tensor true-division operator override.
  # Lazily imports the math ops module and returns the true (floating point) quotient of x by y
  # with optional op name. Errors propagate.
  from tensorflow.python.ops import math_ops

  return math_ops.truediv(x, y, name=name)


def _xor_factory(x, y, name=None):
  # [FUNCTION DOC]
  # HELPER: glue for the Tensor exclusive-or operator override.
  # Lazily imports the math ops module and returns the logical xor of x and y with optional op
  # name. Errors propagate.
  from tensorflow.python.ops import math_ops

  return math_ops.xor_(x, y, name=name)


override_binary_operator.override_binary_operator_helper(
    _add_dispatch_factory, "add"
)
override_binary_operator.override_binary_operator_helper(_and_factory, "and")
override_binary_operator.override_binary_operator_helper(_div_factory, "div")
override_binary_operator.override_binary_operator_helper(
    _floordiv_factory, "floordiv"
)
override_binary_operator.override_binary_operator_helper(
    _matmul_factory, "matmul"
)
override_binary_operator.override_binary_operator_helper(_mod_factory, "mod")
override_binary_operator.override_binary_operator_helper(
    _mul_dispatch_factory, "mul"
)
override_binary_operator.override_binary_operator_helper(_or_factory, "or")
override_binary_operator.override_binary_operator_helper(_pow_factory, "pow")
override_binary_operator.override_binary_operator_helper(
    _subtract_factory, "sub"
)
override_binary_operator.override_binary_operator_helper(
    _truediv_factory, "truediv"
)
override_binary_operator.override_binary_operator_helper(_xor_factory, "xor")


def _invert_factory(x, name=None):
  # [FUNCTION DOC]
  # HELPER: glue for the Tensor invert (bitwise/logical not) operator override.
  # Lazily imports the math ops module and returns the inversion of x with optional op name.
  from tensorflow.python.ops import math_ops

  return math_ops.invert_(x, name=name)


def _abs_factory(x, name=None):
  # [FUNCTION DOC]
  # HELPER: glue for the Tensor absolute-value operator override.
  # Lazily imports the math ops module and returns the absolute value of x with optional name.
  from tensorflow.python.ops import math_ops

  return math_ops.abs(x, name=name)


def _tensor_equals_factory(self, other):
  # [FUNCTION DOC]
  # HELPER: glue for the Tensor equality operator override.
  # Lazily imports the math ops module and returns its tensor equality result between this
  # tensor (self) and other.
  from tensorflow.python.ops import math_ops

  return math_ops.tensor_equals(self, other)


def _tensor_not_equals_factory(self, other):
  # [FUNCTION DOC]
  # HELPER: glue for the Tensor inequality operator override.
  # Lazily imports the math ops module and returns its tensor inequality result between this
  # tensor (self) and other.
  from tensorflow.python.ops import math_ops

  return math_ops.tensor_not_equals(self, other)


def _promote_dtypes_decorator(fn):
  # [FUNCTION DOC]
  # HELPER: wraps comparison ops for the Tensor ordering operator overrides.
  # Takes a binary function and returns a wrapper (preserving the original's metadata through
  # the decorator utility) that first applies implicit dtype promotion to the two operands, then
  # calls the function with the promoted operands and any extra arguments.
  def wrapper(x, y, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: inner wrapper created by the promotion decorator.
    # Promotes tensors x and y to a common dtype where allowed, then returns the wrapped
    # function's result called with the promoted operands and the extra positional and keyword
    # arguments. Errors from promotion or the function propagate.
    x, y = override_binary_operator.maybe_promote_tensors(x, y)
    return fn(x, y, *args, **kwargs)

  return tf_decorator.make_decorator(fn, wrapper)


# pylint: disable=protected-access
tensor_lib.Tensor._override_operator("__invert__", _invert_factory)
tensor_lib.Tensor._override_operator("__neg__", gen_math_ops.neg)
tensor_lib.Tensor._override_operator("__abs__", _abs_factory)
tensor_lib.Tensor._override_operator("__lt__", _promote_dtypes_decorator(
    gen_math_ops.less))
tensor_lib.Tensor._override_operator("__le__", _promote_dtypes_decorator(
    gen_math_ops.less_equal))
tensor_lib.Tensor._override_operator("__gt__", _promote_dtypes_decorator(
    gen_math_ops.greater))
tensor_lib.Tensor._override_operator("__ge__", _promote_dtypes_decorator(
    gen_math_ops.greater_equal))
tensor_lib.Tensor._override_operator("__eq__", _tensor_equals_factory)
tensor_lib.Tensor._override_operator("__ne__", _tensor_not_equals_factory)
