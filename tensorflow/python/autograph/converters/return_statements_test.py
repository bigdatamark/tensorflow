# [MODULE DOC]
# Unit tests for the AutoGraph return_statements converter. Each test defines a small function
# with a particular return pattern (straight line, superfluous, conditional, nested, in context
# managers, loops, try blocks, no return, null return), transforms it with the functions and
# return_statements converters, and asserts that the transformed function gives the same result
# as the original for several inputs. Runs the TensorFlow test main when executed as a script.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for return_statements module."""

from tensorflow.python.autograph.converters import functions
from tensorflow.python.autograph.converters import return_statements
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


class SingleReturnTest(converter_testing.TestCase):
  # [CLASS DOC]
  # Test case for the return_statements converter. All tests compare a function's behavior before
  # and after transformation using the shared equivalence assertion helper.

  def assertTransformedEquivalent(self, f, *inputs):
    # [FUNCTION DOC]
    # HELPER: shared assertion for all tests in this class.
    # Transforms the given function with the functions and return_statements converters, calls
    # both the original and the transformed version with the given positional inputs, and
    # asserts the two results are equal. Inputs: the function and any number of arguments.
    tr = self.transform(f, (functions, return_statements))
    self.assertEqual(f(*inputs), tr(*inputs))

  def test_straightline(self):
    # [FUNCTION DOC]
    # Scenario: a function with a single return at the end. Expects equivalence for input two.

    def f(x):
      # [FUNCTION DOC]
      # Test subject: returns the square of the argument.
      return x * x

    self.assertTransformedEquivalent(f, 2)

  def test_superfluous_returns(self):
    # [FUNCTION DOC]
    # Scenario: a function with unreachable statements and a second return after the first
    # return, separated by other statements. Expects equivalence (the first return wins).

    def f():
      # [FUNCTION DOC]
      # Test subject: returns one; contains an unreachable reassignment and second return.
      retval = 1
      return retval
      retval = 2  # pylint:disable=unreachable
      return retval

    self.assertTransformedEquivalent(f)

  def test_superfluous_returns_adjacent(self):
    # [FUNCTION DOC]
    # Scenario: two return statements directly one after another. Expects equivalence.

    def f():
      # [FUNCTION DOC]
      # Test subject: returns one; a second return right after it is unreachable.
      return 1
      return 2  # pylint:disable=unreachable

    self.assertTransformedEquivalent(f)

  def test_conditional(self):
    # [FUNCTION DOC]
    # Scenario: a conditional that returns in both branches. Expects equivalence for a positive
    # and a negative input.

    def f(x):
      # [FUNCTION DOC]
      # Test subject: returns the argument if positive, otherwise its square.
      if x > 0:
        return x
      else:
        return x * x

    self.assertTransformedEquivalent(f, 2)
    self.assertTransformedEquivalent(f, -2)

  def test_conditional_missing_else(self):
    # [FUNCTION DOC]
    # Scenario: a conditional that returns only in its if branch, with no else and no final
    # return (falls off the end otherwise). Expects equivalence for positive and negative input.

    def f(x):
      # [FUNCTION DOC]
      # Test subject: returns the argument if positive, otherwise returns nothing.
      if x > 0:
        return x

    self.assertTransformedEquivalent(f, 2)
    self.assertTransformedEquivalent(f, -2)

  def test_conditional_missing_else_then_default(self):
    # [FUNCTION DOC]
    # Scenario: a conditional that returns in the if branch followed by a default return after it.
    # Expects equivalence for positive and negative input.

    def f(x):
      # [FUNCTION DOC]
      # Test subject: returns the argument if positive, otherwise its square.
      if x > 0:
        return x
      return x * x

    self.assertTransformedEquivalent(f, 2)
    self.assertTransformedEquivalent(f, -2)

  def test_conditional_else_only_then_default(self):
    # [FUNCTION DOC]
    # Scenario: a conditional where only the else branch returns and the if branch falls through
    # to a final return. Expects equivalence for positive and negative input.

    def f(x):
      # [FUNCTION DOC]
      # Test subject: squares a negative argument and returns it; returns a non-negative one as is.
      if x < 0:
        x *= x
      else:
        return x
      return x

    self.assertTransformedEquivalent(f, 2)
    self.assertTransformedEquivalent(f, -2)

  def test_conditional_nested(self):
    # [FUNCTION DOC]
    # Scenario: nested conditionals, each branch returning a different value. Expects equivalence
    # for inputs two, minus two and five (covering each path).

    def f(x):
      # [FUNCTION DOC]
      # Test subject: returns the argument, its square or its cube depending on nested tests.
      if x > 0:
        if x < 5:
          return x
        else:
          return x * x
      else:
        return x * x * x

    self.assertTransformedEquivalent(f, 2)
    self.assertTransformedEquivalent(f, -2)
    self.assertTransformedEquivalent(f, 5)

  def test_context_manager(self):
    # [FUNCTION DOC]
    # Scenario: a return inside a name-scope context manager. Expects equivalence for inputs two
    # and minus two.

    def f(x):
      # [FUNCTION DOC]
      # Test subject: returns the square of the argument from inside a name scope.
      with ops.name_scope(''):
        return x * x

    self.assertTransformedEquivalent(f, 2)
    self.assertTransformedEquivalent(f, -2)

  def test_context_manager_in_conditional(self):
    # [FUNCTION DOC]
    # Scenario: a context manager containing a return, itself inside one branch of a conditional.
    # Expects equivalence for positive and negative input.

    def f(x):
      # [FUNCTION DOC]
      # Test subject: squares a positive argument inside a name scope, else returns it unchanged.
      if x > 0:
        with ops.name_scope(''):
          return x * x
      else:
        return x

    self.assertTransformedEquivalent(f, 2)
    self.assertTransformedEquivalent(f, -2)

  def test_conditional_in_context_manager(self):
    # [FUNCTION DOC]
    # Scenario: a conditional with returns in both branches, inside a context manager. Expects
    # equivalence for positive and negative input.

    def f(x):
      # [FUNCTION DOC]
      # Test subject: inside a name scope, squares a positive argument, else returns it unchanged.
      with ops.name_scope(''):
        if x > 0:
          return x * x
        else:
          return x

    self.assertTransformedEquivalent(f, 2)
    self.assertTransformedEquivalent(f, -2)

  def test_no_return(self):
    # [FUNCTION DOC]
    # Scenario: a function with no return statement at all. Expects equivalence (both give
    # nothing) for input two.

    def f(x):
      # [FUNCTION DOC]
      # Test subject: modifies a local variable and returns nothing.
      x *= x

    self.assertTransformedEquivalent(f, 2)

  def test_nested_function(self):
    # [FUNCTION DOC]
    # Scenario: a function containing an inner function that has its own multiple returns, and
    # the outer function returns the inner result. Expects equivalence for positive and negative
    # input.

    def f(x):
      # [FUNCTION DOC]
      # Test subject: defines an inner function and returns its result for the argument.

      def inner_fn(y):
        # [FUNCTION DOC]
        # Inner helper of the test subject: returns the square of a positive value, else the
        # value unchanged.
        if y > 0:
          return y * y
        else:
          return y

      return inner_fn(x)

    self.assertTransformedEquivalent(f, 2)
    self.assertTransformedEquivalent(f, -2)

  def test_nested_function_in_control_flow(self):
    # [FUNCTION DOC]
    # Scenario: an inner function defined and called inside a conditional of the outer function,
    # where only the inner function has a return. Expects equivalence for inputs two and minus two.

    def f(x):
      # [FUNCTION DOC]
      # Test subject: conditionally defines and calls an inner function; returns nothing itself.

      if x:
        def inner_fn(y):
          # [FUNCTION DOC]
          # Inner helper of the test subject: returns its argument.
          return y
        inner_fn(x)

    self.assertTransformedEquivalent(f, 2)
    self.assertTransformedEquivalent(f, -2)

  def test_for_loop(self):
    # [FUNCTION DOC]
    # Scenario: a return inside a for loop body. Expects equivalence for loop counts two and zero
    # (the latter never executes the return).

    def f(n):
      # [FUNCTION DOC]
      # Test subject: returns one from inside a loop over a range; returns nothing if empty.
      for _ in range(n):
        return 1

    self.assertTransformedEquivalent(f, 2)
    self.assertTransformedEquivalent(f, 0)

  def test_while_loop(self):
    # [FUNCTION DOC]
    # Scenario: a conditional return inside a while loop plus a default return after the loop.
    # Expects equivalence for inputs zero, two and four.

    def f(n):
      # [FUNCTION DOC]
      # Test subject: accumulates a running sum and returns it once it exceeds four, else minus one.
      i = 0
      s = 0
      while i < n:
        i += 1
        s += i
        if s > 4:
          return s
      return -1

    self.assertTransformedEquivalent(f, 0)
    self.assertTransformedEquivalent(f, 2)
    self.assertTransformedEquivalent(f, 4)

  def test_null_return(self):
    # [FUNCTION DOC]
    # Scenario: bare return statements without a value. Expects equivalence for inputs four and
    # five.

    def f(n):
      # [FUNCTION DOC]
      # Test subject: returns nothing on every path, one of them guarded by a condition.
      if n > 4:
        return
      return

    self.assertTransformedEquivalent(f, 4)
    self.assertTransformedEquivalent(f, 5)

  def test_nested_multiple_withs(self):
    # [FUNCTION DOC]
    # Scenario: a while loop with two consecutive context managers, one containing a conditional
    # return, followed by further statements. Checks guards are placed correctly. Expects
    # equivalence for inputs zero, one, three and four.

    def f(x):
      # [FUNCTION DOC]
      # Test subject: counts down, returning the list built so far when the counter is even.
      v = []
      while x > 0:
        x -= 1
        with ops.name_scope(''):
          if x % 2 == 0:
            return v
        with ops.name_scope(''):
          v.append(x)
        v.append(x)
      return v

    self.assertTransformedEquivalent(f, 0)
    self.assertTransformedEquivalent(f, 1)
    self.assertTransformedEquivalent(f, 3)
    self.assertTransformedEquivalent(f, 4)

  def test_multiple_returns_in_nested_scope(self):
    # [FUNCTION DOC]
    # Scenario: a for loop with several returns, including one inside an exception handler after
    # a raised error. Expects equivalence for an empty list and several non-empty lists.

    def f(a):
      # [FUNCTION DOC]
      # Test subject: iterates over the list and returns an accumulator from a conditional or
      # from the exception handler; the statement after the try block is unreachable.
      v = []
      for x in a:
        x -= 1
        if x > 100:
          return v
        try:
          raise ValueError('intentional')
        except ValueError:  # pylint:disable=bare-except
          return v
        v.append(x)
      return v

    self.assertTransformedEquivalent(f, [])
    self.assertTransformedEquivalent(f, [1])
    self.assertTransformedEquivalent(f, [2])
    self.assertTransformedEquivalent(f, [1, 2, 3])

if __name__ == '__main__':
  test.main()
