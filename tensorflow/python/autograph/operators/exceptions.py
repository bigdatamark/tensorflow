# [MODULE DOC]
# AutoGraph operator for exception-handling statements; currently only the functional form of
# the assert statement. assert_stmt validates the message callable and dispatches to a
# TensorFlow graph assert when the condition is a tensor, or to a Python assert otherwise.
# No import-time side effects.
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
"""Exception handling statements: assert, etc."""

from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.util import tf_inspect


def assert_stmt(expression1, expression2):
  # [FUNCTION DOC]
  # Functional form of an assert. expression1 is the condition (tensor or any Python value);
  # expression2 must be a callable taking no arguments that returns the message value(s).
  # Raises a value error if expression2 is not callable or declares parameters (positional or
  # keyword). A tensor condition is staged as a graph assert, anything else runs a Python
  # assert. Returns the graph operation for tensors, or nothing for Python values.
  """Functional form of an assert statement.

  This follows the semantics of the Python assert statement, however the
  concrete implementations may deviate from it. See the respective
  implementation for details.

  In general, the assert statement should not be used for control flow.
  Furthermore, it is encouraged that the assertion expressions should not have
  side effects.

  Args:
    expression1: Any
    expression2: Callable[[], Any], returns the expression to include in the
        error message when expression1 evaluates to False. When expression1 is
        True, the result of expression2 will not be evaluated, however,
        expression2 itself may be evaluated in some implementations.

  Returns:
    Any, implementation-dependent.

  Raises:
    ValueError: if any arguments are illegal.
  """
  if not callable(expression2):
    raise ValueError('{} must be a callable'.format(expression2))
  args, _, keywords, _ = tf_inspect.getargspec(expression2)
  if args or keywords:
    raise ValueError('{} may not have any arguments'.format(expression2))

  if tensor_util.is_tf_type(expression1):
    return _tf_assert_stmt(expression1, expression2)
  else:
    return _py_assert_stmt(expression1, expression2)


def _tf_assert_stmt(expression1, expression2):
  # [FUNCTION DOC]
  # HELPER: tensor branch of assert_stmt.
  # Calls expression2 to get the message tensor(s), wraps a single value into a list, and
  # creates a graph assert op on expression1 with those values as data. The check runs
  # regardless of Python's debug mode; failure surfaces at graph run time as an
  # invalid-argument error (the docstring says a value error, which is imprecise). Returns the
  # assert operation.
  """Overload of assert_stmt that stages a TF Assert.

  This implementation deviates from Python semantics as follows:
    (1) the assertion is verified regardless of the state of __debug__
    (2) on assertion failure, the graph execution will fail with
        tensorflow.errors.ValueError, rather than AssertionError.

  Args:
    expression1: tensorflow.Tensor, must evaluate to a tf.bool scalar
    expression2: Callable[[], Union[tensorflow.Tensor, List[tensorflow.Tensor]]]

  Returns:
    tensorflow.Operation
  """
  expression2_tensors = expression2()
  if not isinstance(expression2_tensors, list):
    expression2_tensors = [expression2_tensors]
  return control_flow_assert.Assert(expression1, expression2_tensors)


def _py_assert_stmt(expression1, expression2):
  # [FUNCTION DOC]
  # HELPER: Python branch of assert_stmt. Executes a normal assert on expression1, evaluating
  # expression2 only on failure to build the message, so it raises an assertion error when the
  # condition is false (and does nothing under optimized mode). Returns nothing.
  """Overload of assert_stmt that executes a Python assert statement."""
  assert expression1, expression2()
  return None
