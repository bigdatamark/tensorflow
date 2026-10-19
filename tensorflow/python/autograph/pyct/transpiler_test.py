# [MODULE DOC]
# Unit tests for the transpiler module (PyToPy). Defines a test transpiler that flips additions
# into subtractions, and tests covering plain functions, closures, globals, defaults, call trees,
# lambdas, nested functions, thread-safe caching, re-entrant transformation and avoidance of
# namespace collisions. Also defines two module-level globals used by the tests.
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
"""Tests for transpiler module."""

import threading

import gast

from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct import transpiler
from tensorflow.python.platform import test


class FlipSignTransformer(transformer.Base):
  # [CLASS DOC]
  # HELPER: test transformer used by the transpiler tests.
  # AST transformer that rewrites every addition into a subtraction, so tests can tell whether
  # a function was transformed.

  def visit_BinOp(self, node):
    # [FUNCTION DOC]
    # HELPER: visitor for the sign-flipping test transformer.
    # If the binary operation is an addition, replaces its operator with subtraction, then
    # visits children generically. Returns the (mutated) node.
    if isinstance(node.op, gast.Add):
      node.op = gast.Sub()
    return self.generic_visit(node)


class TestTranspiler(transpiler.PyToPy):
  # [CLASS DOC]
  # HELPER: minimal concrete transpiler for the tests. Uses a constant caching key, no extra
  # locals, and the sign-flipping transformer as its transformation.

  def get_caching_key(self, ctx):
    # [FUNCTION DOC]
    # HELPER: test override. Ignores the context and always returns the constant zero.
    del ctx
    return 0

  def get_extra_locals(self):
    # [FUNCTION DOC]
    # HELPER: test override. Returns an empty mapping (no extra locals).
    return {}

  def transform_ast(self, node, ctx):
    # [FUNCTION DOC]
    # HELPER: test override. Runs the sign-flipping transformer over the node using the given
    # context and returns the transformed node.
    return FlipSignTransformer(ctx).visit(node)


global_var_for_test_global = 1
global_var_for_test_namespace_collisions = object()


class PyToPyTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering PyToPy: function, closure, global, default, lambda, nested, concurrency,
  # re-entrance and namespace-collision behavior, using the sign-flipping test transpiler.

  def test_basic(self):
    # [FUNCTION DOC]
    # Transforms a simple function that adds one to its argument and asserts that, with the
    # addition flipped to subtraction, calling it with one returns zero.
    def f(a):
      # [FUNCTION DOC]
      # HELPER: test subject; returns the argument plus one.
      return a + 1

    tr = TestTranspiler()
    f, _, _ = tr.transform(f, None)

    self.assertEqual(f(1), 0)

  def test_closure(self):
    # [FUNCTION DOC]
    # Verifies that the transformed function keeps the original closure cell: after the captured
    # variable is rebound, the result reflects the new value (zero, then minus one).
    b = 1

    def f(a):
      # [FUNCTION DOC]
      # HELPER: test subject; returns the argument plus a captured variable.
      return a + b

    tr = TestTranspiler()
    f, _, _ = tr.transform(f, None)

    self.assertEqual(f(1), 0)
    b = 2
    self.assertEqual(f(1), -1)

  def test_global(self):
    # [FUNCTION DOC]
    # Verifies that the transformed function reads the module-level global at call time: after
    # the global is reassigned, results change from zero to minus one.
    def f(a):
      # [FUNCTION DOC]
      # HELPER: test subject; returns the argument plus a module-level global.
      return a + global_var_for_test_global

    tr = TestTranspiler()
    f, _, _ = tr.transform(f, None)

    global global_var_for_test_global
    global_var_for_test_global = 1
    self.assertEqual(f(1), 0)
    global_var_for_test_global = 2
    self.assertEqual(f(1), -1)

  def test_defaults(self):
    # [FUNCTION DOC]
    # Verifies default argument handling: the default is evaluated once at definition, so
    # changing the source variable afterwards has no effect, while the captured closure variable
    # still updates.
    b = 2
    c = 1

    def f(a, d=c + 1):
      # [FUNCTION DOC]
      # HELPER: test subject; returns the sum of the argument, a captured variable and a default.
      return a + b + d

    tr = TestTranspiler()
    f, _, _ = tr.transform(f, None)

    self.assertEqual(f(1), 1 - 2 - 2)
    c = 0
    self.assertEqual(f(1), 1 - 2 - 2)  # Defaults are evaluated at definition.
    b = 1
    self.assertEqual(f(1), 1 - 2 - 1)

  def test_call_tree(self):
    # [FUNCTION DOC]
    # Verifies that only the transformed function itself is rewritten, not the functions it
    # calls: the callee still adds, the caller's addition is flipped.

    def g(a):
      # [FUNCTION DOC]
      # HELPER: test callee; returns the argument plus one (not transformed).
      return a + 1

    def f(a):
      # [FUNCTION DOC]
      # HELPER: test subject; calls the callee and adds one.
      return g(a) + 1

    tr = TestTranspiler()
    f, _, _ = tr.transform(f, None)

    self.assertEqual(f(1), 1 - 1 + 1)  # Only f is converted.

  def test_lambda(self):
    # [FUNCTION DOC]
    # Transforms a lambda with a captured variable and a conditional expression, and asserts the
    # results for positive and negative input, including after the captured variable changes.
    b = 2
    f = lambda x: (b + (x if x > 0 else -x))

    tr = TestTranspiler()
    f, _, _ = tr.transform(f, None)

    self.assertEqual(f(1), 2 - 1)
    self.assertEqual(f(-1), 2 - 1)

    b = 3

    self.assertEqual(f(1), 3 - 1)
    self.assertEqual(f(-1), 3 - 1)

  def test_multiple_lambdas(self):
    # [FUNCTION DOC]
    # Defines two lambdas on one line and transforms the first; asserts the correct one is picked
    # (disambiguated by argument names) and its addition is flipped.
    a, b = 1, 2
    # This can be disambiguated by the argument names.
    f, _ = (lambda x: a + x, lambda y: b * y)

    tr = TestTranspiler()
    f, _, _ = tr.transform(f, None)

    self.assertEqual(f(1), 1 - 1)

  def test_nested_functions(self):
    # [FUNCTION DOC]
    # Transforms a function containing an inner function that uses a captured variable; asserts
    # the inner function's addition is also flipped.
    b = 2

    def f(x):
      # [FUNCTION DOC]
      # HELPER: test subject; defines and calls an inner function.

      def g(x):
        # [FUNCTION DOC]
        # HELPER: inner test function; returns a captured variable plus the argument.
        return b + x

      return g(x)

    tr = TestTranspiler()
    f, _, _ = tr.transform(f, None)

    self.assertEqual(f(1), 2 - 1)

  def test_nested_lambda(self):
    # [FUNCTION DOC]
    # Transforms a function that defines and calls an inner lambda using a captured variable;
    # asserts the lambda's addition is flipped.
    b = 2

    def f(x):
      # [FUNCTION DOC]
      # HELPER: test subject; defines and calls an inner lambda.
      g = lambda x: b + x
      return g(x)

    tr = TestTranspiler()
    f, _, _ = tr.transform(f, None)

    self.assertEqual(f(1), 2 - 1)

  def test_concurrency(self):
    # [FUNCTION DOC]
    # Starts ten threads that each transform the same function and record the generated module
    # name; asserts all threads got the same module, i.e. the cache prevented duplicate work.

    def f():
      # [FUNCTION DOC]
      # HELPER: test subject; does nothing.
      pass

    outputs = []

    tr = TestTranspiler()
    # Note: this is not a test, it's a required invariant.
    assert tr.get_caching_key(None) == tr.get_caching_key(None)

    def conversion_thread():
      # [FUNCTION DOC]
      # HELPER: thread body; transforms the function and appends the generated module's name to
      # the shared output list.
      _, mod, _ = tr.transform(f, None)
      outputs.append(mod.__name__)

    threads = tuple(
        threading.Thread(target=conversion_thread) for _ in range(10))
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    # Races would potentially create multiple functions / modules
    # (non-deterministically, but with high likelihood).
    self.assertEqual(len(set(outputs)), 1)

  def test_reentrance(self):
    # [FUNCTION DOC]
    # Uses a transpiler whose transformation re-enters the transpiler once for the same function;
    # asserts this does not deadlock (the lock is re-entrant) and the result is still flipped.

    def test_fn():
      # [FUNCTION DOC]
      # HELPER: test subject; returns one plus one.
      return 1 + 1

    class ReentrantTranspiler(transpiler.PyToPy):
      # [CLASS DOC]
      # HELPER: test transpiler that tracks recursion depth and, on the first transformation,
      # triggers a nested transform of the same function.

      def __init__(self):
        # [FUNCTION DOC]
        # Constructor. Calls the parent constructor and sets the recursion depth to zero.
        super(ReentrantTranspiler, self).__init__()
        self._recursion_depth = 0

      def get_caching_key(self, ctx):
        # [FUNCTION DOC]
        # HELPER: test override; ignores the context and returns the constant zero.
        del ctx
        return 0

      def get_extra_locals(self):
        # [FUNCTION DOC]
        # HELPER: test override; returns an empty mapping.
        return {}

      def transform_ast(self, node, ctx):
        # [FUNCTION DOC]
        # HELPER: test override. Increments the recursion depth; if below two, re-enters the
        # transform for the test function. Then flips signs in the node and returns it.
        self._recursion_depth += 1
        if self._recursion_depth < 2:
          self.transform(test_fn, None)
        return FlipSignTransformer(ctx).visit(node)

    tr = ReentrantTranspiler()

    f, _, _ = tr.transform(test_fn, None)
    self.assertEqual(f(), 0)

  def test_namespace_collisions_avoided(self):
    # [FUNCTION DOC]
    # Transforms a method whose name equals a module-level global that it returns; asserts the
    # transformed function still returns the global object (the generated name did not shadow it).

    class TestClass(object):
      # [CLASS DOC]
      # HELPER: test class holding a method named like a module-level global.

      def global_var_for_test_namespace_collisions(self):
        # [FUNCTION DOC]
        # HELPER: test subject; returns the module-level global of the same name.
        return global_var_for_test_namespace_collisions

    tr = TestTranspiler()
    obj = TestClass()

    f, _, _ = tr.transform(
        obj.global_var_for_test_namespace_collisions, None)
    self.assertIs(f(obj), global_var_for_test_namespace_collisions)


if __name__ == '__main__':
  test.main()
