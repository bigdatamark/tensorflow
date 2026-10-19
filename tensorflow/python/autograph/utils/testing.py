# [MODULE DOC]
# HELPER: shared test infrastructure for AutoGraph tests.
# Defines AutoGraphTestCase, a test base class that runs each test method inside a TensorFlow
# function and defers assertions until the function's values are computed. No import-time side
# effects.
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
"""Testing utilities."""

import re
import sys
import types
import unittest

from tensorflow.python.eager import def_function
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class AutoGraphTestCase(test.TestCase):
  # [CLASS DOC]
  # HELPER: base class for AutoGraph tests.
  # Wraps every test method (names with the test prefix defined directly on the subclass) so it
  # runs inside a TensorFlow function with autograph disabled. Overridden assertion methods
  # record their arguments during tracing and are executed afterwards on the evaluated values.
  # State per instance: variables (mapping from names to variables), trace_log (list of op types
  # created, collected through an op callback), assertions (deferred assertion records),
  # graph_assertions (checks run against the traced graph) and raises_cm (pending expected-error
  # context or nothing).
  """Tests specialized for AutoGraph, which run as tf.functions.

  These tests use a staged programming-like approach: most of the test code runs
  as-is inside a tf.function, but the assertions are lifted outside the
  function, and run with the corresponding function values instead.

  For example, the test:

      def test_foo(self):
        baz = bar();
        self.assertEqual(baz, value)

  is equivalent to writing:

      def test_foo(self):
        @tf.function
        def test_fn():
          baz = bar();
          return baz, value

        baz_actual, value_actual = test_fn()
        self.assertEqual(baz_actual, value_actual)

  Only assertions that require evaluation outside the function are lifted
  outside the function scope. The rest execute inline, at function creation
  time.
  """

  def __new__(cls, *args):
    # [FUNCTION DOC]
    # Creates the instance and replaces each test method defined directly in the class's own
    # namespace (inherited ones are skipped) with a wrapper that runs it as a TensorFlow
    # function. Input: the class and constructor arguments (ignored when creating the object).
    # Returns the new instance. Side effect: sets instance attributes for the wrapped methods.
    obj = super().__new__(cls)

    for name in cls.__dict__:
      if not name.startswith(unittest.TestLoader.testMethodPrefix):
        continue
      m = getattr(obj, name)
      if callable(m):
        wrapper = obj._run_as_tf_function(m)
        setattr(obj, name, types.MethodType(wrapper, obj))

    return obj

  def _op_callback(
      self, op_type, inputs, attrs, outputs, op_name=None, graph=None):
    # [FUNCTION DOC]
    # HELPER: op callback registered during setUp. Appends the type of every created op to the
    # trace log. Other arguments (inputs, attrs, outputs, op name, graph) are ignored.
    # Returns nothing.
    self.trace_log.append(op_type)

  def _run_as_tf_function(self, fn):
    # [FUNCTION DOC]
    # HELPER: builds the wrapper used by the constructor. Input fn: a bound test method.
    # Returns a function taking the test instance; it traces fn inside a TensorFlow function
    # (autograph off), runs graph assertions, evaluates the collected assertion arguments, and
    # then runs the deferred assertions on the concrete values. If an expected-error context was
    # opened and an error occurs, the error is passed to that context and the test ends;
    # otherwise the error propagates.

    def wrapper(self):
      # [FUNCTION DOC]
      # HELPER: the replacement test method. Resets per-test lists, traces and evaluates the
      # test, then runs deferred assertions. Returns nothing.
      @def_function.function(autograph=False)  # Testing autograph itself.
      def fn_wrapper():
        # [FUNCTION DOC]
        # HELPER: the traced function. Resets assertions, expected-error context, graph
        # assertions and trace log, calls the test body, and returns the list of argument lists
        # of the recorded assertions so they are evaluated as tensors.
        self.assertions = []
        self.raises_cm = None
        self.graph_assertions = []
        self.trace_log = []
        fn()
        targets = [args for _, args in self.assertions]
        return targets

      try:
        tensors = fn_wrapper()

        for assertion in self.graph_assertions:
          assertion(fn_wrapper.get_concrete_function().graph)

        actuals = self.evaluate(tensors)

      except:  # pylint:disable=bare-except
        if self.raises_cm is not None:
          # Note: Yes, the Raises and function contexts cross.
          self.raises_cm.__exit__(*sys.exc_info())
          return
        else:
          raise

      for (assertion, _), values in zip(self.assertions, actuals):
        assertion(*values)

    return wrapper

  def variable(self, name, value, dtype):
    # [FUNCTION DOC]
    # HELPER: returns a variable cached by name, creating it outside the function being traced
    # on first use (initial value and data type given) and running its initializer. Later calls
    # with the same name return the same variable and ignore value and dtype.
    with ops.init_scope():
      if name not in self.variables:
        self.variables[name] = variables.Variable(value, dtype=dtype)
        self.evaluate(self.variables[name].initializer)
    return self.variables[name]

  def setUp(self):
    # [FUNCTION DOC]
    # Runs the parent setup, then initializes the variable cache, trace log and expected-error
    # context, and registers the op callback so created ops are logged.
    super().setUp()
    self.variables = {}
    self.trace_log = []
    self.raises_cm = None
    op_callbacks.add_op_callback(self._op_callback)

  def tearDown(self):
    # [FUNCTION DOC]
    # Unregisters the op callback, clears the trace log and variable cache, and runs the parent
    # teardown.
    op_callbacks.remove_op_callback(self._op_callback)
    self.trace_log = None
    self.variables = None
    super().tearDown()

  def assertGraphContains(self, op_regex, n):
    # [FUNCTION DOC]
    # Registers a deferred check that the traced graph (top-level nodes plus nodes of library
    # functions) contains exactly n nodes whose names match the regular expression op_regex
    # (matched from the start of the name). Runs after tracing; returns nothing.
    def assertion(graph):
      # [FUNCTION DOC]
      # HELPER: the deferred check. Counts matching nodes in the graph and asserts the count
      # equals n.
      matches = []
      for node in graph.as_graph_def().node:
        if re.match(op_regex, node.name):
          matches.append(node)
      for fn in graph.as_graph_def().library.function:
        for node_def in fn.node_def:
          if re.match(op_regex, node_def.name):
            matches.append(node_def)
      self.assertLen(matches, n)

    self.graph_assertions.append(assertion)

  def assertOpCreated(self, op_type):
    # [FUNCTION DOC]
    # Asserts immediately (during tracing) that an op of the given type name was logged.
    self.assertIn(op_type, self.trace_log)

  def assertOpsNotCreated(self, op_types):
    # [FUNCTION DOC]
    # Asserts immediately that none of the given op type names (a collection) were logged.
    self.assertEmpty(set(op_types) & set(self.trace_log))

  def assertNoOpsCreated(self):
    # [FUNCTION DOC]
    # Asserts immediately that the trace log of created ops is empty.
    self.assertEmpty(self.trace_log)

  def assertEqual(self, *args):
    # [FUNCTION DOC]
    # Overrides equality assertion: does not assert now; records the parent's equality assertion
    # with the given arguments, to run later on evaluated values. Returns nothing.
    self.assertions.append((super().assertEqual, list(args)))

  def assertLess(self, *args):
    # [FUNCTION DOC]
    # Overrides less-than assertion: records the parent's assertion with the given arguments for
    # deferred execution on evaluated values. Returns nothing.
    self.assertions.append((super().assertLess, list(args)))

  def assertGreaterEqual(self, *args):
    # [FUNCTION DOC]
    # Overrides greater-or-equal assertion: records the parent's assertion with the given
    # arguments for deferred execution on evaluated values. Returns nothing.
    self.assertions.append((super().assertGreaterEqual, list(args)))

  def assertDictEqual(self, *args):
    # [FUNCTION DOC]
    # Overrides dictionary equality assertion: records the parent's assertion with the given
    # arguments for deferred execution on evaluated values. Returns nothing.
    self.assertions.append((super().assertDictEqual, list(args)))

  def assertRaisesRuntime(self, *args):
    # [FUNCTION DOC]
    # Declares that the test is expected to raise an error when the function runs. Arguments are
    # forwarded to the regex-matching raises assertion (error type and message pattern). Opens
    # that context immediately and leaves it open; the wrapper closes it with the caught error.
    # Raises a value error if called more than once in the same test. Returns nothing.
    if self.raises_cm is not None:
      raise ValueError('cannot use more than one assertRaisesRuntime in a test')
    self.raises_cm = self.assertRaisesRegex(*args)
    self.raises_cm.__enter__()
