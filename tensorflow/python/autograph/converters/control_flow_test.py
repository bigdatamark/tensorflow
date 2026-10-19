# [MODULE DOC]
# Unit tests for the autograph control_flow converter, which turns if, while and for statements
# into calls to autograph runtime operators. Tests are grouped by statement kind (nested, while,
# if, for, and combinations with else/break/continue). Each test defines a small function,
# converts it with the converter and runs it with tensor or Python inputs, asserting on the
# computed values or on raised errors. Defines a few module level globals (unaffected, mixed and
# local globals) that some tests read or modify. Runs the TensorFlow test main when executed.
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
"""Tests for control_flow module."""

import collections

import numpy as np

from tensorflow.python.autograph.converters import break_statements
from tensorflow.python.autograph.converters import continue_statements
from tensorflow.python.autograph.converters import control_flow
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import test
from tensorflow.python.util import nest


for_unaffected_global = None
for_mixed_globals_nonglobals = None
for_test_global_local = None


class ControlFlowTestBase(converter_testing.TestCase):
  # [CLASS DOC]
  # HELPER: shared base class for the control flow converter tests. Provides assertions that
  # compare (possibly tensor valued) results and that convert-and-run a function.

  def assertValuesEqual(self, actual, expected):
    # [FUNCTION DOC]
    # HELPER: asserts that a nested structure of results equals the expected structure. Tensors in
    # actual are evaluated to concrete values first; other values are used as they are. Returns
    # nothing; fails the test on mismatch.
    values = nest.map_structure(
        lambda x: self.evaluate(x) if tensor_util.is_tf_type(x) else x,
        actual)
    self.assertAllEqual(values, expected)

  def assertTransformedResult(self, f, inputs, expected):
    # [FUNCTION DOC]
    # HELPER: converts function f with the control flow converter, calls it with inputs (a single
    # value is wrapped into a one-element tuple, a tuple is spread as arguments) and asserts the
    # result equals expected using the value comparison helper. Returns nothing.
    if not isinstance(inputs, tuple):
      inputs = (inputs,)
    tr = self.transform(f, control_flow)
    returns = tr(*inputs)
    self.assertValuesEqual(returns, expected)


class NestedControlFlowTest(ControlFlowTestBase):
  # [CLASS DOC]
  # Tests for control flow statements nested inside each other (loops in loops, conditionals in
  # loops), including global variables and composite (attribute) state.

  def test_basic(self):
    # [FUNCTION DOC]
    # Scenario: a while loop containing an inner while loop and a variable first defined after the
    # inner loop. Input is a tensor 5; expects the tuple (25, 5, 0, 5).

    def f(n):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      i = 0
      j = 0
      s = 0
      while i < n:
        while j < i:
          j += 3
        u = i + j  # 'u' is not defined within the inner loop
        s += u
        i += 1
        j = 0
      return s, i, j, n

    self.assertTransformedResult(f, constant_op.constant(5),
                                 (25, 5, 0, 5))

  def test_mixed_globals_nonglobals(self):
    # [FUNCTION DOC]
    # Scenario: nested while loops where the outer loop updates a module level global together
    # with local variables. Input tensor 5; expects (25, 5, 0, 5).

    def f(n):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      global for_mixed_globals_nonglobals
      i = 0
      j = 0
      for_mixed_globals_nonglobals = 0
      while i < n:
        while j < i:
          j += 3
        u = i + j  # 'u' is not defined within the inner loop
        for_mixed_globals_nonglobals += u
        i += 1
        j = 0
      return for_mixed_globals_nonglobals, i, j, n

    self.assertTransformedResult(f, constant_op.constant(5),
                                 (25, 5, 0, 5))

  def test_composite_state_complex(self):
    # [FUNCTION DOC]
    # Scenario: nested objects with a dictionary attribute, mutated deep inside an if within a
    # while loop. Input tensor 5; expects the counter 0 and the deep attribute equal to 6.

    class TestClassX(object):
      # [CLASS DOC]
      # Local helper class holding one attribute x, set by the constructor.

      def __init__(self, x):
        # [FUNCTION DOC]
        # Constructor of a local test helper class; stores the given arguments (or fixed initial
        # values) as attributes.
        self.x = x

    class TestClassY(object):
      # [CLASS DOC]
      # Local helper class holding one attribute y, set by the constructor.

      def __init__(self, y):
        # [FUNCTION DOC]
        # Constructor of a local test helper class; stores the given arguments (or fixed initial
        # values) as attributes.
        self.y = y

    def f(n):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      tc = TestClassX(TestClassY({'z': TestClassX(n)}))
      if n > 0:
        while n > 0:
          if n < 2:
            tc.x.y['z'].x += 1
          n -= 1
      return n, tc

    tr = self.transform(f, control_flow)

    n, tc = tr(constant_op.constant(5))
    self.assertValuesEqual((n, tc.x.y['z'].x), (0, 6))


class WhileStatementTest(ControlFlowTestBase):
  # [CLASS DOC]
  # Tests for conversion of while loops: basic state, composite state (attributes, subscripts),
  # variables defined inside the loop, non-tensor state and dispatch decided by the condition.

  def test_basic(self):
    # [FUNCTION DOC]
    # Scenario: a while loop accumulating a sum and a counter. Input is a tensor 5; expects the
    # tuple (10, 5, 5).

    def f(n):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      i = 0
      s = 0
      while i < n:
        s += i
        i += 1
      return s, i, n

    self.assertTransformedResult(f, constant_op.constant(5), (10, 5, 5))

  def test_single_output(self):
    # [FUNCTION DOC]
    # Scenario: a while loop that modifies only one variable (the argument). Input is a tensor 5;
    # expects 0.

    def f(n):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      while n > 0:
        n -= 1
      return n

    self.assertTransformedResult(f, constant_op.constant(5), 0)

  def test_composite_state_attr(self):
    # [FUNCTION DOC]
    # Scenario: a while loop that increments an attribute of an object created before the loop.
    # Input tensor 5; expects the returned counter 0 (conversion succeeds).

    class TestClass(object):
      # [CLASS DOC]
      # Local helper class used as a mutable object whose attributes the loop or branch modifies.

      def __init__(self):
        # [FUNCTION DOC]
        # Constructor of a local test helper class; stores the given arguments (or fixed initial
        # values) as attributes.
        self.x = constant_op.constant(3)

    def f(n):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      tc = TestClass()
      while n > 0:
        tc.x += 1
        n -= 1
      return n

    self.assertTransformedResult(f, constant_op.constant(5), 0)

  def test_composite_state_slice(self):
    # [FUNCTION DOC]
    # Scenario: a while loop that increments a dictionary entry addressed by a variable key.
    # Input tensor 5; expects (10, 0).

    def f(n):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      d = {'a': n}
      k = 'a'
      while n > 0:
        d[k] += 1
        n -= 1
      return d[k], n

    self.assertTransformedResult(f, constant_op.constant(5), (10, 0))

  def test_composite_state_literal_slice(self):
    # [FUNCTION DOC]
    # Scenario: like the variable key case but the dictionary key is a literal. Expects (10, 0).

    def f(n):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      d = {'a': n}
      while n > 0:
        d['a'] += 1
        n -= 1
      return d['a'], n

    self.assertTransformedResult(f, constant_op.constant(5), (10, 0))

  def test_composite_state_attr_initialized_in_loop(self):
    # [FUNCTION DOC]
    # Scenario: an attribute is first assigned inside the loop. Expects 14 when the loop runs as a
    # tensor loop with a tensor initial value, and a value error saying the attribute must be
    # defined before the loop when the loop condition is a tensor and the other input is a plain
    # number.

    class TestClass(object):
      # [CLASS DOC]
      # Local helper class used as a mutable object whose attributes the loop or branch modifies.
      pass

    def f(n, x):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      tc = TestClass()
      while n < 5:
        if n == 0:
          tc.subattr = x
        else:
          tc.subattr = tc.subattr + 1
        n += 1
      return tc.subattr

    self.assertTransformedResult(f, (0, constant_op.constant(10)), 14)
    tr = self.transform(f, control_flow)
    with self.assertRaisesRegex(
        ValueError, "'tc.subattr' must be defined before the loop"):
      tr(constant_op.constant(0), 0)

  def test_composite_state_slice_initialized_in_loop(self):
    # [FUNCTION DOC]
    # Scenario: a dictionary entry with a variable key is first set inside the loop. Expects the
    # entry equal to 14 in the supported case and a value error (entry must be defined before the
    # loop) in the unsupported case.

    def f(n, x):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      d = {}
      k = 'subkey'
      while n < 5:
        if n == 0:
          d[k] = x
        else:
          d[k] = d[k] + 1
        n += 1
      return d

    self.assertTransformedResult(f, (0, constant_op.constant(10)),
                                 {'subkey': 14})
    tr = self.transform(f, control_flow)
    with self.assertRaisesRegex(
        ValueError, r"'d\[k\]' must be defined before the loop"):
      tr(constant_op.constant(0), 0)

  def test_composite_state_literal_slice_initialized_in_loop(self):
    # [FUNCTION DOC]
    # Scenario: a dictionary entry with a literal key is first set inside the loop. Expects the
    # entry equal to 14 in the supported case and a value error (entry must be defined before the
    # loop) in the unsupported case.

    def f(n, x):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      d = {}
      while n < 5:
        if n == 0:
          d['subkey'] = x
        else:
          d['subkey'] = d['subkey'] + 1
        n += 1
      return d

    self.assertTransformedResult(f, (0, constant_op.constant(10)),
                                 {'subkey': 14})
    tr = self.transform(f, control_flow)
    with self.assertRaisesRegex(
        ValueError, r"'d\['subkey'\]' must be defined before the loop"):
      tr(constant_op.constant(0), 0)

  def test_composite_state_slice_aliased_to_local(self):
    # [FUNCTION DOC]
    # Scenario: the dictionary key is a variable defined inside the loop. Eager execution yields
    # the entry 11; running the converted function as a graph function without further conversion
    # is expected to raise an inaccessible tensor error (a better message is a pending TODO).

    def f(n, x):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      d = {}
      while n < 5:
        k = 'subkey'
        d[k] = x + 1
        n += 1
      return d

    self.assertTransformedResult(f, (0, constant_op.constant(10)),
                                 {'subkey': 11})
    tr = self.transform(f, control_flow)
    # TODO(b/136999953): Better error message.
    # Note that this error happens at execution time.
    with self.assertRaises(errors.InaccessibleTensorError):
      graph_fn = def_function.function(tr, autograph=False)
      self.evaluate(
          graph_fn(constant_op.constant(0), constant_op.constant(5)))

  def test_local_composite_attr(self):
    # [FUNCTION DOC]
    # Scenario: an object created and mutated entirely inside the loop body must not become loop
    # state. Input tensor 5; expects 0.

    class TestClass(object):
      # [CLASS DOC]
      # Local helper class used as a mutable object whose attributes the loop or branch modifies.

      def __init__(self):
        # [FUNCTION DOC]
        # Constructor of a local test helper class; stores the given arguments (or fixed initial
        # values) as attributes.
        self.x = constant_op.constant(3)

    def f(n):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      while n > 0:
        tc = TestClass()
        tc.x = tc.x
        n -= 1
      return n

    self.assertTransformedResult(f, constant_op.constant(5), 0)

  def test_local_composite_slice(self):
    # [FUNCTION DOC]
    # Scenario: a dictionary created inside the loop and updated via a variable key stays local.
    # Input tensor 5; expects 0.

    def f(n):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      while n > 0:
        d = {'x': n}
        k = 'x'
        d[k] = d[k]
        n -= 1
      return n

    self.assertTransformedResult(f, constant_op.constant(5), 0)

  def test_local_composite_literal_slice(self):
    # [FUNCTION DOC]
    # Scenario: a dictionary created inside the loop and updated via a literal key stays local.
    # Input tensor 5; expects 0.

    def f(n):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      while n > 0:
        d = {'x': n}
        d['x'] = d['x']
        n -= 1
      return n

    self.assertTransformedResult(f, constant_op.constant(5), 0)

  def test_non_tensor_state(self):
    # [FUNCTION DOC]
    # Scenario: a loop variable that is a named tuple holding a list of tensors, which is allowed
    # as loop state. Input tensor 5; expects 4.

    # This class is ok to be in a tf.while's state.
    class TestClass(collections.namedtuple('TestClass', ('x'))):
      # [CLASS DOC]
      # Local named tuple subclass with one field x, used as legal non-tensor loop state.
      pass

    def f(n):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      tc = TestClass([constant_op.constant(0)])
      while n > 0:
        tc = TestClass([constant_op.constant(3)])
        tc.x[0] = tc.x[0] + 1
        n -= 1
      return tc.x[0]

    self.assertTransformedResult(f, constant_op.constant(5), 4)

  def test_non_tensor_state_illegal_type(self):
    # [FUNCTION DOC]
    # Scenario: a plain class instance created inside the loop and used after it would need to be
    # loop state, which is unsupported. Expects a value error saying the variable must be defined
    # before the loop.

    class TestClass(object):
      # [CLASS DOC]
      # Local helper class used as a mutable object whose attributes the loop or branch modifies.

      def __init__(self):
        # [FUNCTION DOC]
        # Constructor of a local test helper class; stores the given arguments (or fixed initial
        # values) as attributes.
        self.x = [constant_op.constant(3)]

    def f(n):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      while n > 0:
        tc = TestClass()
        tc.x[0] = tc.x[0] + 1
        n -= 1
      return tc.x[0]

    tr = self.transform(f, control_flow)

    # The tested function would require `tc` to become part of the while loop
    # state, but TensorFlow doesn't support classes at the moment.
    with self.assertRaisesRegex(
        ValueError, 'tc.*must be defined before the loop'):
      tr(constant_op.constant(5))

  def test_dispatches_by_cond_only(self):
    # [FUNCTION DOC]
    # Scenario: the choice between Python and tensor looping depends only on the loop condition.
    # With a tensor condition and integer state expects 10; with a Python condition and a custom
    # numeric object expects the object value 10; with a tensor condition and the custom object
    # expects a type error.

    class TensorIncompatibleNumeric(object):
      # [CLASS DOC]
      # Local helper class wrapping a value; supports addition in plain Python but not in tensor ops.
      """Works in arithmetic expression, but errors out with TF ops."""

      def __init__(self, val):
        # [FUNCTION DOC]
        # Constructor of a local test helper class; stores the given arguments (or fixed initial
        # values) as attributes.
        self.val = val

      def __add__(self, other):
        # [FUNCTION DOC]
        # Addition operator: returns a new wrapper whose value is the sum of this value and other.
        return TensorIncompatibleNumeric(self.val + other)

    def f(n, s):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      while n > 0:
        n -= 1
        s += n
      return s

    self.assertTransformedResult(f, (constant_op.constant(5), 0), 10)
    tr = self.transform(f, control_flow)
    # n alone controls the staging. When the loop is not staged, Python
    # knows how to add the two objects. But when staged, tf.while will
    # not know how to deal with the TensorIncompatibleNumeric object.
    self.assertEqual(tr(5, TensorIncompatibleNumeric(0)).val, 10)
    with self.assertRaises(TypeError):
      tr(constant_op.constant(5), TensorIncompatibleNumeric(0))


class IfStatementTest(ControlFlowTestBase):
  # [CLASS DOC]
  # Tests for conversion of if statements: balanced and unbalanced branches, local and global
  # variables, composite outputs, raising branches and variables created in only one branch.

  def test_basic(self):
    # [FUNCTION DOC]
    # Scenario: an if/else where each branch assigns a different variable. Expects (-1, 0) for a
    # positive tensor input and (0, -2) for a negative one.

    def f(n):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      a = 0
      b = 0
      if n > 0:
        a = -n
      else:
        b = 2 * n
      return a, b

    self.assertTransformedResult(f, constant_op.constant(1), (-1, 0))
    self.assertTransformedResult(f, constant_op.constant(-1), (0, -2))

  def test_sparse_tensor(self):
    # [FUNCTION DOC]
    # Scenario: an if whose condition is a sparse tensor (staged) or nothing/null (not staged).
    # Expects -1 for the sparse tensor condition and 1 for a null condition.

    def f(cond, a):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      if cond:
        a = -a
      return a

    st = sparse_tensor.SparseTensor(
        indices=((0,),), values=(0,), dense_shape=(1,))
    self.assertTransformedResult(f, (st, constant_op.constant(1)), -1)
    self.assertTransformedResult(f, (None, constant_op.constant(1)), 1)

  def test_complex_outputs(self):
    # [FUNCTION DOC]
    # Scenario: branches that assign attributes of an object argument. Expects (-1, 0) for a
    # positive tensor input and (0, -2) for a negative one.

    class TestClass(object):
      # [CLASS DOC]
      # Local helper class used as a mutable object whose attributes the loop or branch modifies.

      def __init__(self, a, b):
        # [FUNCTION DOC]
        # Constructor of a local test helper class; stores the given arguments (or fixed initial
        # values) as attributes.
        self.a = a
        self.b = b

    def f(n, obj):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      obj.a = 0
      obj.b = 0
      if n > 0:
        obj.a = -n
      else:
        obj.b = 2 * n
      return obj

    tr = self.transform(f, control_flow)

    res_obj = tr(constant_op.constant(1), TestClass(0, 0))
    self.assertValuesEqual((res_obj.a, res_obj.b), (-1, 0))
    res_obj = tr(constant_op.constant(-1), TestClass(0, 0))
    self.assertValuesEqual((res_obj.a, res_obj.b), (0, -2))

  def test_single_output(self):
    # [FUNCTION DOC]
    # Scenario: an if without else that modifies one variable. Input tensor 1 expects -1.

    def f(n):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      if n > 0:
        n = -n
      return n

    self.assertTransformedResult(f, constant_op.constant(1), -1)

  def test_unbalanced(self):
    # [FUNCTION DOC]
    # Scenario: an if without else that reassigns the argument. Expects 3 for tensor 2 and -3 for
    # tensor -3.

    def f(n):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      if n > 0:
        n = 3
      return n

    self.assertTransformedResult(f, constant_op.constant(2), 3)
    self.assertTransformedResult(f, constant_op.constant(-3), -3)

  def test_unbalanced_raising(self):
    # [FUNCTION DOC]
    # Scenario: an if branch that modifies a variable and then raises. Plain input -3 returns -3;
    # input 1 raises a value error.

    def f(n):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      if n > 0:
        n = n + 1
        raise ValueError()
      return n

    self.assertTransformedResult(f, -3, -3)

    tr = self.transform(f, control_flow)

    with self.assertRaises(ValueError):
      tr(1)

  def test_local_var(self):
    # [FUNCTION DOC]
    # Scenario: a variable created only inside the if branch and used there. Expects 5 for tensor
    # 1 and -1 for tensor -1.

    def f(n):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      if n > 0:
        b = 4
        n = b + 1
      return n

    self.assertTransformedResult(f, constant_op.constant(1), 5)
    self.assertTransformedResult(f, constant_op.constant(-1), -1)

  def test_local_remains_local(self):
    # [FUNCTION DOC]
    # Scenario: same as the local variable case; a branch-local variable must not leak into the
    # state. Expects 5 for tensor 1 and -1 for tensor -1.

    def f(n):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      if n > 0:
        b = 4
        n = b + 1
      return n

    self.assertTransformedResult(f, constant_op.constant(1), 5)
    self.assertTransformedResult(f, constant_op.constant(-1), -1)

  def test_global_local(self):
    # [FUNCTION DOC]
    # Scenario: a module level global declared and updated inside an if branch. Expects the call
    # with 1 to return 2 and the global to be set to 1 afterwards; asserts it started as nothing.

    def f(n):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      if n > 0:
        global for_test_global_local
        if for_test_global_local is None:
          for_test_global_local = 1
        else:
          for_test_global_local += 1
        n += for_test_global_local
      return n

    tr = self.transform(f, control_flow)
    assert for_test_global_local is None
    self.assertEqual(tr(1), 2)
    self.assertEqual(for_test_global_local, 1)

  def test_no_outputs(self):
    # [FUNCTION DOC]
    # Scenario: an if branch that only assigns an unused local variable. Expects the input back
    # unchanged (1 and -1).

    def f(n):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      if n > 0:
        b = 4  # pylint:disable=unused-variable
      return n

    self.assertTransformedResult(f, constant_op.constant(1), 1)
    self.assertTransformedResult(f, constant_op.constant(-1), -1)

  def test_created_outputs(self):
    # [FUNCTION DOC]
    # Scenario: both branches assign a variable not defined before the if. Expects -1 for input 0
    # and 2 for input 1.

    def f(i):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      if i == 0:
        result = i - 1
      else:
        result = i + 1
      return result

    self.assertTransformedResult(f, 0, -1)
    self.assertTransformedResult(f, 1, 2)

  def test_created_loop_local_outputs(self):
    # [FUNCTION DOC]
    # Scenario: inside a for loop, ifs create a loop-local variable then a second if updates the
    # outer state. Expects 14 for a range of 5 with initial 10.

    def f(n, x):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      for i in n:
        if i == 0:
          result = i - 1
        else:
          result = i + 1
        if result > 0:
          x += 1
      return x

    self.assertTransformedResult(f, (range(5), 10), 14)

  def test_created_loop_variable(self):
    # [FUNCTION DOC]
    # Scenario: a variable created in an if on the first iteration is read in later iterations.
    # Expects 14 for a range of 5 with initial 10.

    def f(n, x):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      for i in n:
        if i == 0:
          result = i - 1
        if i > 0:  # Using the result from previous iteration.
          if result < 0:
            x += 1
      return x

    self.assertTransformedResult(f, (range(5), 10), 14)

  def test_unaffected_global(self):
    # [FUNCTION DOC]
    # Scenario: a global is set to 3 then conditionally changed in the function. Expects 3 when
    # the branch is not taken, -1 when taken, and the global equal to -1 at the end. Modifies the
    # module level global.

    global for_unaffected_global
    for_unaffected_global = 3

    def f(i):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      global for_unaffected_global
      if i == 0:
        for_unaffected_global = i - 1
      return for_unaffected_global

    self.assertTransformedResult(f, 1, 3)
    self.assertTransformedResult(f, 0, -1)
    self.assertEqual(for_unaffected_global, -1)

  def test_unaffected_nonlocal(self):
    # [FUNCTION DOC]
    # Scenario: a nested function modifies a nonlocal variable inside an if. Expects 3 when the
    # branch is not taken and -1 when it is.

    def f(i):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      def inner_fn():
        # [FUNCTION DOC]
        # Nested function that, when the outer argument is 0, assigns a nonlocal variable. Returns nothing.
        nonlocal n
        if i == 0:
          n = i - 1

      n = 3
      inner_fn()
      return n

    self.assertTransformedResult(f, 1, 3)
    self.assertTransformedResult(f, 0, -1)

  def test_output_defined_in_prior_except(self):
    # [FUNCTION DOC]
    # Scenario: a variable defined in an exception handler is later conditionally reassigned.
    # Expects 1 for input 1 and -1 for input 0.

    def f(i):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      try:
        raise ValueError()
      except ValueError:
        x = 1
      if i == 0:
        x = i - 1
      return x

    self.assertTransformedResult(f, 1, 1)
    self.assertTransformedResult(f, 0, -1)

  def test_unbalanced_multiple_composites(self):
    # [FUNCTION DOC]
    # Scenario: an if without else that sets two attributes of an object and a plain variable.
    # Expects (7, 11, 13) for a true tensor condition and (2, 3, 5) for a false one.

    class Foo(object):
      # [CLASS DOC]
      # Local helper class holding attributes that the tested branch may overwrite.

      def __init__(self):
        # [FUNCTION DOC]
        # Constructor of a local test helper class; stores the given arguments (or fixed initial
        # values) as attributes.
        self.b = 2
        self.c = 3

    def f(x, condition):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.

      z = 5
      if condition:
        x.b = 7
        x.c = 11
        z = 13

      return x.b, x.c, z

    self.assertTransformedResult(f, (Foo(), constant_op.constant(True)),
                                 (7, 11, 13))
    self.assertTransformedResult(f, (Foo(), constant_op.constant(False)),
                                 (2, 3, 5))

  def test_unbalanced_composite(self):
    # [FUNCTION DOC]
    # Scenario: an if without else that sets one attribute and a plain variable. Expects (7, 13)
    # for a true tensor condition and (2, 5) for a false one.

    class Foo(object):
      # [CLASS DOC]
      # Local helper class holding attributes that the tested branch may overwrite.

      def __init__(self):
        # [FUNCTION DOC]
        # Constructor of a local test helper class; stores the given arguments (or fixed initial
        # values) as attributes.
        self.b = 2

    def f(x, condition):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.

      z = 5
      if condition:
        x.b = 7
        z = 13

      return x.b, z

    self.assertTransformedResult(f, (Foo(), constant_op.constant(True)),
                                 (7, 13))
    self.assertTransformedResult(f, (Foo(), constant_op.constant(False)),
                                 (2, 5))


class ForStatementTest(ControlFlowTestBase):
  # [CLASS DOC]
  # Tests for conversion of for loops over tensors and Python iterables: state, iterated
  # expression evaluation, composite state, tuple unpacking and comprehensions in the body.

  def test_basic(self):
    # [FUNCTION DOC]
    # Scenario: a for loop over a tensor accumulating a sum and a sum of squares. Expects (4, 10)
    # for the vector of 1 and 3, and (0, 0) for an empty int vector.

    def f(l):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      s1 = 0
      s2 = 0
      for e in l:
        s1 += e
        s2 += e * e
      return s1, s2

    self.assertTransformedResult(f, constant_op.constant([1, 3]), (4, 10))
    empty_vector = constant_op.constant([], shape=(0,), dtype=dtypes.int32)
    self.assertTransformedResult(f, empty_vector, (0, 0))

  def test_single_output(self):
    # [FUNCTION DOC]
    # Scenario: a for loop over a tensor accumulating a single sum. Expects 4 for the vector of 1
    # and 3, and 0 for an empty vector.

    def f(l):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      s = 0
      for e in l:
        s += e
      return s

    self.assertTransformedResult(f, constant_op.constant([1, 3]), 4)
    empty_vector = constant_op.constant([], shape=(0,), dtype=dtypes.int32)
    self.assertTransformedResult(f, empty_vector, 0)

  def test_iterated_expression(self):
    # [FUNCTION DOC]
    # Scenario: the iterated expression has a side effect (a counter). Verifies the result 10 for
    # input 5 and that the expression is evaluated exactly once.

    eval_count = [0]

    def count_evals(x):
      # [FUNCTION DOC]
      # HELPER: increments the shared evaluation counter and returns its argument unchanged.
      eval_count[0] += 1
      return x

    def f(n):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      s = 0
      for e in count_evals(range(n)):
        s += e
      return s

    tr = self.transform(f, control_flow)

    self.assertEqual(tr(5), 10)
    self.assertEqual(eval_count[0], 1)

  def test_composite_state_initialized_in_loop(self):
    # [FUNCTION DOC]
    # Scenario: an attribute first assigned inside a for loop. Expects 20 for a Python range with
    # a tensor initial value, and a value error (must be defined before the loop) for a tensor
    # iterable with a plain number.

    class TestClass(object):
      # [CLASS DOC]
      # Local helper class used as a mutable object whose attributes the loop or branch modifies.
      pass

    def f(n, x):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      tc = TestClass()
      for i in n:
        if i == 0:
          tc.x = x
        else:
          tc.x = tc.x + i
      return tc.x

    self.assertTransformedResult(f, (range(5), constant_op.constant(10)), 20)
    tr = self.transform(f, control_flow)

    with self.assertRaisesRegex(
        ValueError, "'tc.x' must be defined before the loop"):
      tr(constant_op.constant(list(range(5))), 0)

  def test_tuple_unpacking(self):
    # [FUNCTION DOC]
    # Scenario: a for loop over an enumerate pair with tuple unpacking into the target. Input list
    # of 3 and 3; expects 7.

    def f(x_list):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      z = constant_op.constant(0)  # pylint:disable=undefined-variable
      for i, x in enumerate(x_list):
        z = z + x + i
      return z

    self.assertTransformedResult(f, [3, 3], 7)

  def test_with_comprehension_in_body(self):
    # [FUNCTION DOC]
    # Scenario: a list comprehension inside a for loop body. Expects the vector 0 to 4 multiplied
    # by 4 for a three element tensor loop.

    def f(l, n):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      s = constant_op.constant(list(range(n)))
      for _ in l:
        s += constant_op.constant([a for a in range(n)])
      return s

    self.assertTransformedResult(f, (constant_op.constant([1, 2, 3]), 5),
                                 np.array(range(5)) * 4)


class AdvancedControlFlowTest(ControlFlowTestBase):
  # [CLASS DOC]
  # Tests for loops with else clauses and break statements, converting with the break, continue
  # and control flow converters together and comparing against plain Python behavior.

  def assertTransformedEquivalent(self, f, *inputs):
    # [FUNCTION DOC]
    # HELPER: converts f with the break, continue and control flow converters, calls the original
    # and converted functions with the given inputs and asserts the results are equal.
    tr = self.transform(
        f, (break_statements, continue_statements, control_flow))
    self.assertEqual(f(*inputs), tr(*inputs))

  def test_while_with_else(self):
    # [FUNCTION DOC]
    # Scenario: a while loop with an else clause. Expects converted and original to agree for 4 and 2.

    def f(x):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      while x > 2:
        x /= 2
      else:
        x += 1
      return x

    self.assertTransformedEquivalent(f, 4)
    self.assertTransformedEquivalent(f, 2)

  def test_while_with_else_and_break(self):
    # [FUNCTION DOC]
    # Scenario: a while loop with a conditional break and an else clause (runs only without break).
    # Expects agreement for true and false flags.

    def f(cond1):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      x = 8
      while x > 2:
        x /= 2
        if cond1:
          break
      else:
        x += 1
      return x

    self.assertTransformedEquivalent(f, True)
    self.assertTransformedEquivalent(f, False)

  def test_for_with_else(self):
    # [FUNCTION DOC]
    # Scenario: a for loop with an else clause. Expects agreement for an empty list and a list of 1, 2.

    def f(l):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      res = 0
      for x in l:
        res += x
      else:
        res += 1
      return res

    self.assertTransformedEquivalent(f, [])
    self.assertTransformedEquivalent(f, [1, 2])

  def test_for_with_else_and_break(self):
    # [FUNCTION DOC]
    # Scenario: a for loop with a conditional break and an else clause. Expects agreement for true
    # and false flags.

    def f(flag):
      # [FUNCTION DOC]
      # Function under test, defined locally in the enclosing test. It is converted by the control
      # flow converter and called by the test; it returns the values the test asserts on.
      l = [1, 2, 3]
      res = 0
      for x in l:
        res += x
        if flag:
          break
      else:
        res += 1
      return res

    self.assertTransformedEquivalent(f, True)
    self.assertTransformedEquivalent(f, False)


if __name__ == '__main__':
  test.main()
