# [MODULE DOC]
# AutoGraph logical operators as functions: not_, and_, or_, eq and not_eq. Each dispatches on
# whether operands are tensors (graph ops, with lazy conditionals for and/or) or plain Python
# values (ordinary Python semantics). The and/or operators take callables so the right operand
# is evaluated lazily. No import-time side effects.
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
"""Logical boolean operators: not, and, or."""

from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import gen_math_ops


def not_(a):
  # [FUNCTION DOC]
  # Functional logical negation. Returns a logical-not tensor if a is a tensor, otherwise the
  # Python truthiness negation of a.
  """Functional form of "not"."""
  if tensor_util.is_tf_type(a):
    return _tf_not(a)
  return _py_not(a)


def _tf_not(a):
  # [FUNCTION DOC]
  # HELPER: tensor branch of not_. Returns the element-wise logical negation tensor.
  """Implementation of the "not_" operator for TensorFlow."""
  return gen_math_ops.logical_not(a)


def _py_not(a):
  # [FUNCTION DOC]
  # HELPER: Python branch of not_. Returns the boolean negation of a.
  """Default Python implementation of the "not_" operator."""
  return not a


def and_(a, b):
  # [FUNCTION DOC]
  # Functional lazy "and". a and b are callables with no arguments. Calls a first; if the result
  # is a tensor, stages a conditional (b is called only on the taken branch), otherwise applies
  # Python short-circuit semantics. Returns the combined value.
  """Functional form of "and". Uses lazy evaluation semantics."""
  a_val = a()
  if tensor_util.is_tf_type(a_val):
    return _tf_lazy_and(a_val, b)
  return _py_lazy_and(a_val, b)


def _tf_lazy_and(cond, b):
  # [FUNCTION DOC]
  # HELPER: tensor branch of and_. Stages a graph conditional on the tensor cond: when true it
  # evaluates and returns b(); when false it returns cond. A TODO notes cond is not checked to
  # be scalar.
  """Lazy-eval equivalent of "and" for Tensors."""
  # TODO(mdan): Enforce cond is scalar here?
  return tf_cond.cond(cond, b, lambda: cond)


def _py_lazy_and(cond, b):
  # [FUNCTION DOC]
  # HELPER: Python branch of and_. Returns cond if falsy, otherwise the result of calling b.
  """Lazy-eval equivalent of "and" in Python."""
  return cond and b()


def or_(a, b):
  # [FUNCTION DOC]
  # Functional lazy "or". a and b are callables with no arguments. Calls a first; if the result
  # is a tensor, stages a conditional (b is called only on the taken branch), otherwise applies
  # Python short-circuit semantics. Returns the combined value.
  """Functional form of "or". Uses lazy evaluation semantics."""
  a_val = a()
  if tensor_util.is_tf_type(a_val):
    return _tf_lazy_or(a_val, b)
  return _py_lazy_or(a_val, b)


def _tf_lazy_or(cond, b):
  # [FUNCTION DOC]
  # HELPER: tensor branch of or_. Stages a graph conditional on the tensor cond: when true it
  # returns cond; when false it evaluates and returns b(). A TODO notes cond is not checked to
  # be scalar.
  """Lazy-eval equivalent of "or" for Tensors."""
  # TODO(mdan): Enforce cond is scalar here?
  return tf_cond.cond(cond, lambda: cond, b)


def _py_lazy_or(cond, b):
  # [FUNCTION DOC]
  # HELPER: Python branch of or_. Returns cond if truthy, otherwise the result of calling b.
  """Lazy-eval equivalent of "or" in Python."""
  return cond or b()


def eq(a, b):
  # [FUNCTION DOC]
  # Functional equality. If either operand is a tensor returns an element-wise equal tensor,
  # otherwise the Python equality result.
  """Functional form of "equal"."""
  if tensor_util.is_tf_type(a) or tensor_util.is_tf_type(b):
    return _tf_equal(a, b)
  return _py_equal(a, b)


def _tf_equal(a, b):
  # [FUNCTION DOC]
  # HELPER: tensor branch of eq. Returns the element-wise equality tensor of a and b.
  """Overload of "equal" for Tensors."""
  return gen_math_ops.equal(a, b)


def _py_equal(a, b):
  # [FUNCTION DOC]
  # HELPER: Python branch of eq. Returns the result of Python's equality comparison.
  """Overload of "equal" that falls back to Python's default implementation."""
  return a == b


def not_eq(a, b):
  # [FUNCTION DOC]
  # Functional inequality, built as the logical negation of eq. Returns a tensor if either
  # operand is a tensor, otherwise a Python boolean.
  """Functional form of "not-equal"."""
  return not_(eq(a, b))
