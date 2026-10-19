# [MODULE DOC]
# Tests for the liveness analysis module. A shared base class parses a function and runs qualified
# names, activity, control flow graph, reaching function definitions and liveness analysis, with
# helpers asserting live-in and live-out annotations. The test class checks live-in and live-out
# symbol sets for conditionals, loops, try/except, nested functions, lambdas, deletion,
# comprehensions, global and nonlocal symbols. Defines module-level globals for the global test.
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
"""Tests for liveness module."""

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import naming
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import liveness
from tensorflow.python.autograph.pyct.static_analysis import reaching_fndefs
from tensorflow.python.platform import test


global_a = 7
global_b = 17


class LivenessAnalyzerTestBase(test.TestCase):
  # [CLASS DOC]
  # HELPER: shared base for liveness tests (also reused by the Python-3-only test module);
  # provides the full analysis pipeline helper and live-in/live-out assertions. Defines no tests.

  def _parse_and_analyze(self, test_fn):
    # [FUNCTION DOC]
    # HELPER: parses the source of a function, resolves qualified names, runs activity analysis,
    # builds control flow graphs, runs reaching function definitions and liveness analysis.
    # Returns the annotated function tree.
    # TODO(mdan): Use a custom FunctionTransformer here.
    node, source = parser.parse_entity(test_fn, future_features=())
    entity_info = transformer.EntityInfo(
        name=test_fn.__name__,
        source_code=source,
        source_file=None,
        future_features=(),
        namespace={})
    node = qual_names.resolve(node)
    namer = naming.Namer({})
    ctx = transformer.Context(entity_info, namer, None)
    node = activity.resolve(node, ctx)
    graphs = cfg.build(node)
    node = reaching_fndefs.resolve(node, ctx, graphs)
    node = liveness.resolve(node, ctx, graphs)
    return node

  def assertHasLiveOut(self, node, expected):
    # [FUNCTION DOC]
    # HELPER: asserts the node's live-out annotation, as a set of text names, equals the expected
    # value: a single name, a tuple of names, or empty for none.
    live_out = anno.getanno(node, anno.Static.LIVE_VARS_OUT)
    live_out_strs = set(str(v) for v in live_out)
    if not expected:
      expected = ()
    if not isinstance(expected, tuple):
      expected = (expected,)
    self.assertSetEqual(live_out_strs, set(expected))

  def assertHasLiveIn(self, node, expected):
    # [FUNCTION DOC]
    # HELPER: asserts the node's live-in annotation, as a set of text names, equals the expected
    # value: a single name, a tuple of names, or empty for none.
    live_in = anno.getanno(node, anno.Static.LIVE_VARS_IN)
    live_in_strs = set(str(v) for v in live_in)
    if not expected:
      expected = ()
    if not isinstance(expected, tuple):
      expected = (expected,)
    self.assertSetEqual(live_in_strs, set(expected))


class LivenessAnalyzerTest(LivenessAnalyzerTestBase):
  # [CLASS DOC]
  # Tests of liveness analysis: each test parses a small function and checks live-in or live-out
  # symbol sets on selected statements of its body.

  def test_live_out_try_block(self):
    # [FUNCTION DOC]
    # Scenario: if containing a try with bare handler, followed by a return of x. Expects x to be
    # live out of both the if and the try.

    def test_fn(x, a, b, c):  # pylint:disable=unused-argument
      # [FUNCTION DOC]
      # Test subject: conditional try/except that does nothing, then returns x.
      if a > 0:
        try:
          pass
        except:  # pylint:disable=bare-except
          pass
      return x

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveOut(fn_body[0], 'x')
    self.assertHasLiveOut(fn_body[0].body[0], 'x')

  def test_live_out_if_inside_except(self):
    # [FUNCTION DOC]
    # Scenario: an if inside the except handler of a try inside an if, then return x. Expects x
    # live out of the outer if, the try, and the first statement of the handler.

    def test_fn(x, a, b, c):  # pylint:disable=unused-argument
      # [FUNCTION DOC]
      # Test subject: nested conditional, try/except and conditional assignment of x, return x.
      if a > 0:
        try:
          pass
        except:  # pylint:disable=bare-except
          if b > 0:
            x = b
      return x

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveOut(fn_body[0], 'x')
    self.assertHasLiveOut(fn_body[0].body[0], 'x')
    self.assertHasLiveOut(fn_body[0].body[0].handlers[0].body[0], 'x')

  def test_live_out_stacked_if(self):
    # [FUNCTION DOC]
    # Scenario: two consecutive ifs assigning x, then return x. Expects a and x live out of the
    # first if (a is needed by the second test) and x live out of the second.

    def test_fn(x, a):
      # [FUNCTION DOC]
      # Test subject: two sequential conditional assignments then a return.
      if a > 0:
        x = 0
      if a > 1:
        x = 1
      return x

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveOut(fn_body[0], ('a', 'x'))
    self.assertHasLiveOut(fn_body[1], 'x')

  def test_live_out_stacked_if_else(self):
    # [FUNCTION DOC]
    # Scenario: an if followed by an if/else that always assigns x, then return x. Expects only a
    # live out of the first if (x is overwritten on both branches) and x live out of the second.

    def test_fn(x, a):
      # [FUNCTION DOC]
      # Test subject: conditional assignment followed by an exhaustive if/else assignment.
      if a > 0:
        x = 0
      if a > 1:
        x = 1
      else:
        x = 2
      return x

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveOut(fn_body[0], 'a')
    self.assertHasLiveOut(fn_body[1], 'x')

  def test_live_out_for_basic(self):
    # [FUNCTION DOC]
    # Scenario: for loop accumulating into x, then return x. Expects x live out of the loop.

    def test_fn(x, a):
      # [FUNCTION DOC]
      # Test subject: loop adding the loop variable into x, then returns x.
      for i in range(a):
        x += i
      return x

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveOut(fn_body[0], 'x')

  def test_live_out_for_iterate(self):
    # [FUNCTION DOC]
    # Scenario: loop whose iteration variable is used after the loop. Expects both x and the
    # loop variable live out of the loop.

    def test_fn(x, a):
      # [FUNCTION DOC]
      # Test subject: accumulating loop followed by a return of x and the loop variable.
      for i in range(a):
        x += i
      return x, i  # pylint:disable=undefined-loop-variable

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveOut(fn_body[0], ('x', 'i'))

  def test_live_out_attributes(self):
    # [FUNCTION DOC]
    # Scenario: conditional attribute write followed by a read of the attribute. Expects the
    # attribute and its parent object both live out of the if.

    def test_fn(x, a):
      # [FUNCTION DOC]
      # Test subject: conditionally sets an attribute of x, then returns that attribute.
      if a > 0:
        x.y = 0
      return x.y

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveOut(fn_body[0], ('x.y', 'x'))

  def test_live_out_nested_functions(self):
    # [FUNCTION DOC]
    # Scenario: variable assigned in an if and later captured by an inner function that is
    # defined and called after it. Expects a live out of the if because of the closure.

    def test_fn(a, b):
      # [FUNCTION DOC]
      # Test subject: conditional assignment, inner function reading it, and a call of it.
      if b:
        a = []

      def foo():
        # [FUNCTION DOC]
        # Inner function: returns the captured variable.
        return a

      foo()

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveOut(fn_body[0], 'a')

  def test_live_out_nested_functions_defined_ahead(self):
    # [FUNCTION DOC]
    # Scenario: inner function defined before a conditional assignment of the variable it
    # captures, then returned. Expects a and the function name live out of the if.

    def test_fn(a, b):
      # [FUNCTION DOC]
      # Test subject: defines an inner function, conditionally assigns a captured variable and
      # returns the inner function.
      def foo():
        # [FUNCTION DOC]
        # Inner function: returns the captured variable.
        return a

      if b:
        a = []

      return foo

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveOut(fn_body[1], ('a', 'foo'))

  def test_live_out_nested_functions_defined_after(self):
    # [FUNCTION DOC]
    # Scenario: conditional assignment followed by the definition and return of an inner function
    # that reads the variable. Expects a live out of the if.

    def test_fn(a, b):
      # [FUNCTION DOC]
      # Test subject: conditional assignment, then an inner function capturing it is returned.
      if b:
        a = []

      def foo():
        # [FUNCTION DOC]
        # Inner function: returns the captured variable.
        return a

      return foo

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveOut(fn_body[0], ('a',))

  def test_live_out_lambda(self):
    # [FUNCTION DOC]
    # Scenario: variable captured by an assigned lambda. Expects a and b live out of the first if,
    # and, per a TODO about a known deprecation issue, only the lambda variable live out of the
    # later if (the closure of the lambda is not counted there; the alternative expectation is
    # left commented out).

    def test_fn(a, b):
      # [FUNCTION DOC]
      # Test subject: conditional assignment, assigns a lambda reading it, a second conditional,
      # returns the lambda.
      if b:
        a = []

      foo = lambda: a

      if b:
        pass

      return foo

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveOut(fn_body[0], ('a', 'b'))
    #TODO(@bhack): replace this after deprecation
    # https://github.com/tensorflow/tensorflow/issues/56089
    self.assertHasLiveOut(fn_body[2], ('foo',))
    #self.assertHasLiveOut(fn_body[2], ('a', 'foo'))

  def test_live_out_nested_functions_hidden_by_argument(self):
    # [FUNCTION DOC]
    # Scenario: inner function whose parameter shadows an outer variable assigned later. Expects
    # only the function name live out of the if (the shadowed name is not a free variable).

    def test_fn(b):
      # [FUNCTION DOC]
      # Test subject: defines an inner function with a parameter, conditionally assigns an
      # unrelated same-named local, and returns the function.
      def foo(a):
        # [FUNCTION DOC]
        # Inner function: returns its parameter.
        return a

      if b:
        a = []  # pylint:disable=unused-variable

      return foo

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveOut(fn_body[1], ('foo'))

  def test_live_out_nested_functions_isolation(self):
    # [FUNCTION DOC]
    # Scenario: inner function that uses and then assigns a local of the same name as an outer
    # variable. Expects the inner function's local not to leak: only the called builtin name is
    # live out of the outer if.

    def test_fn(b):
      # [FUNCTION DOC]
      # Test subject: conditional assignment, then an inner function with its own same-named local
      # that is called.
      if b:
        a = 0  # pylint:disable=unused-variable

      def child():
        # [FUNCTION DOC]
        # Inner function: reads then assigns a local variable and returns it.
        max(a)  # pylint:disable=used-before-assignment
        a = 1
        return a

      child()

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveOut(fn_body[0], 'max')

  def test_live_out_deletion(self):
    # [FUNCTION DOC]
    # Scenario: loop in which y is deleted in one branch and assigned in the other. Expects
    # nothing live out of the loop.

    def test_fn(x, y, a):
      # [FUNCTION DOC]
      # Test subject: loop over a with a branch that deletes y and another that assigns it.
      for _ in a:
        if x:
          del y
        else:
          y = 0

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveOut(fn_body[0], ())

  def test_live_in_pass(self):
    # [FUNCTION DOC]
    # Scenario: if with a pass body then return x. Expects live-in sets: a and x for the if,
    # x for the pass, x for the return.

    def test_fn(x, a, b, c):  # pylint:disable=unused-argument
      # [FUNCTION DOC]
      # Test subject: conditional pass followed by a return of x.
      if a > 0:
        pass
      return x

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveIn(fn_body[0], ('a', 'x'))
    self.assertHasLiveIn(fn_body[0].body[0], ('x',))
    self.assertHasLiveIn(fn_body[1], ('x',))

  def test_live_in_raise(self):
    # [FUNCTION DOC]
    # Scenario: if whose body updates b and raises c. Expects live-in of the if to include a, b,
    # c and x; of the first body statement b and c; and of the final return x.

    def test_fn(x, a, b, c):
      # [FUNCTION DOC]
      # Test subject: conditional increment of b followed by a raise, then a return of x.
      if a > 0:
        b = b + 1
        raise c
      return x

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveIn(fn_body[0], ('a', 'b', 'c', 'x'))
    self.assertHasLiveIn(fn_body[0].body[0], ('b', 'c'))
    self.assertHasLiveIn(fn_body[1], ('x',))

  def test_live_out_except_variable(self):
    # [FUNCTION DOC]
    # Scenario: try with a handler binding an exception variable and re-raising it. Expects the
    # try's live-in to be the exception variable name and x; the exception type is not live
    # because nothing in the try body can raise (as noted in the test comment). The test name says
    # live-out but it asserts live-in.

    def test_fn(x, a):
      # [FUNCTION DOC]
      # Test subject: try/except binding the exception to a name and raising it, then return x.
      try:
        pass
      except a as b:
        raise b
      return x

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    # Note: 'a' is not live because there is no raise statement inside the
    # try, and we discount the possibility of other code in the try block
    # raising an error.
    self.assertHasLiveIn(fn_body[0], ('b', 'x'))

  def test_live_in_return_statement(self):
    # [FUNCTION DOC]
    # Scenario: conditional early return of x followed by another return of x. Expects live-in
    # a and x for the if, x for the inner return and x for the final return.

    def test_fn(x, a, b, c):  # pylint:disable=unused-argument
      # [FUNCTION DOC]
      # Test subject: returns x conditionally and then unconditionally.
      if a > 0:
        return x
      return x

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveIn(fn_body[0], ('a', 'x'))
    self.assertHasLiveIn(fn_body[0].body[0], ('x',))
    self.assertHasLiveIn(fn_body[1], ('x',))

  def test_live_in_try_block(self):
    # [FUNCTION DOC]
    # Scenario: if containing a try with a bare handler, then return x. Expects live-in a and x
    # for the if, x for the try, x for the return.

    def test_fn(x, a, b, c):  # pylint:disable=unused-argument
      # [FUNCTION DOC]
      # Test subject: conditional try/except that does nothing, then returns x.
      if a > 0:
        try:
          pass
        except:  # pylint:disable=bare-except
          pass
      return x

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveIn(fn_body[0], ('a', 'x'))
    self.assertHasLiveIn(fn_body[0].body[0], ('x',))
    self.assertHasLiveIn(fn_body[1], ('x',))

  def test_live_in_try_orelse(self):
    # [FUNCTION DOC]
    # Scenario: try with a handler and an else clause assigning x from b. Expects live-in a, b, x
    # for the if; b and x for the try; x for the return.

    def test_fn(x, a, b, c):  # pylint:disable=unused-argument
      # [FUNCTION DOC]
      # Test subject: conditional try/except/else where the else assigns x, then returns x.
      if a > 0:
        try:
          pass
        except:  # pylint:disable=bare-except
          pass
        else:
          x = b
      return x

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveIn(fn_body[0], ('a', 'b', 'x'))
    self.assertHasLiveIn(fn_body[0].body[0], ('b', 'x'))
    self.assertHasLiveIn(fn_body[1], ('x',))

  def test_live_in_if_inside_except(self):
    # [FUNCTION DOC]
    # Scenario: if inside the handler of a try inside an if. Expects live-in a, b, x for the
    # outer if; b and x for the try and for the inner if in the handler; x for the return.

    def test_fn(x, a, b, c):  # pylint:disable=unused-argument
      # [FUNCTION DOC]
      # Test subject: nested conditional/try/handler with a conditional assignment, return x.
      if a > 0:
        try:
          pass
        except:  # pylint:disable=bare-except
          if b > 0:
            x = b
      return x

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveIn(fn_body[0], ('a', 'b', 'x'))
    self.assertHasLiveIn(fn_body[0].body[0], ('b', 'x'))
    self.assertHasLiveIn(fn_body[0].body[0].handlers[0].body[0], ('b', 'x'))
    self.assertHasLiveIn(fn_body[1], ('x',))

  def test_live_in_stacked_if(self):
    # [FUNCTION DOC]
    # Scenario: two consecutive conditional assignments of x. Expects live-in a, b, c, x for the
    # first if and c, x for the second.

    def test_fn(x, a, b, c):
      # [FUNCTION DOC]
      # Test subject: two sequential conditional assignments to x, then returns x.
      if a > 0:
        x = b
      if c > 1:
        x = 0
      return x

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveIn(fn_body[0], ('a', 'b', 'c', 'x'))
    self.assertHasLiveIn(fn_body[1], ('c', 'x'))

  def test_live_in_stacked_if_else(self):
    # [FUNCTION DOC]
    # Scenario: an exhaustive if/else assigning x followed by a conditional assignment of x.
    # Expects live-in a, b, c, d for the first statement (x is not live in) and d, x for the
    # second.

    def test_fn(x, a, b, c, d):
      # [FUNCTION DOC]
      # Test subject: if/else assigning x, then a conditional assignment, then returns x.
      if a > 1:
        x = b
      else:
        x = c
      if d > 0:
        x = 0
      return x

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveIn(fn_body[0], ('a', 'b', 'c', 'd'))
    self.assertHasLiveIn(fn_body[1], ('d', 'x'))

  def test_live_in_for_basic(self):
    # [FUNCTION DOC]
    # Scenario: loop assigning x, accumulating into y and assigning z, then return y and z.
    # Expects loop live-in a, y, z (z is live because the loop may not execute).

    def test_fn(x, y, a):
      # [FUNCTION DOC]
      # Test subject: loop over a with assignments and an accumulation, then returns y and z.
      for i in a:
        x = i
        y += x
        z = 0
      return y, z

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveIn(fn_body[0], ('a', 'y', 'z'))

  def test_live_in_for_nested(self):
    # [FUNCTION DOC]
    # Scenario: nested loops with assignments and accumulation. Expects outer loop live-in
    # a, y, z.

    def test_fn(x, y, a):
      # [FUNCTION DOC]
      # Test subject: nested loops assigning and accumulating, then returns y and z.
      for i in a:
        for j in i:
          x = i
          y += x
          z = j
      return y, z

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveIn(fn_body[0], ('a', 'y', 'z'))

  def test_live_in_deletion(self):
    # [FUNCTION DOC]
    # Scenario: loop with a branch deleting y and another assigning it. Expects loop live-in
    # a, x, y (deleting a symbol counts as a use).

    def test_fn(x, y, a):
      # [FUNCTION DOC]
      # Test subject: loop over a with a branch that deletes y and another that assigns it.
      for _ in a:
        if x:
          del y
        else:
          y = 0

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveIn(fn_body[0], ('a', 'x', 'y'))

  def test_live_in_generator_comprehension(self):
    # [FUNCTION DOC]
    # Scenario: if whose test calls a function on a generator expression. Expects live-in to be
    # the called function name and the iterable, not the generator target.

    def test_fn(y):
      # [FUNCTION DOC]
      # Test subject: conditional on a call over a generator expression.
      if all(x for x in y):
        return

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveIn(fn_body[0], ('all', 'y'))

  def test_live_in_list_comprehension(self):
    # [FUNCTION DOC]
    # Scenario: if whose test is a list comprehension. Expects live-in to be only the iterable.

    def test_fn(y):
      # [FUNCTION DOC]
      # Test subject: conditional on a list comprehension.
      if [x for x in y]:
        return

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveIn(fn_body[0], ('y',))

  def test_live_in_list_comprehension_expression(self):
    # [FUNCTION DOC]
    # Scenario: augmented assignment adding the result of a call on a list comprehension. Expects
    # live-in to contain the iterable, the called function name and the augmented target.

    def test_fn(y, s):
      # [FUNCTION DOC]
      # Test subject: adds the result of an undefined function applied to a list comprehension.
      s += foo([x for x in y])  # pylint:disable=undefined-variable

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveIn(fn_body[0], ('y', 'foo', 's'))

  def test_live_in_set_comprehension(self):
    # [FUNCTION DOC]
    # Scenario: if whose test is a set comprehension. Expects live-in to be only the iterable.

    def test_fn(y):
      # [FUNCTION DOC]
      # Test subject: conditional on a set comprehension.
      if {x for x in y}:
        return

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveIn(fn_body[0], ('y',))

  def test_live_in_dict_comprehension(self):
    # [FUNCTION DOC]
    # Scenario: if whose test is a dictionary comprehension. Expects live-in to be only the
    # iterable.

    def test_fn(y):
      # [FUNCTION DOC]
      # Test subject: conditional on a dictionary comprehension.
      if {k: v for k, v in y}:
        return

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasLiveIn(fn_body[0], ('y',))

  def test_global_symbol(self):
    # [FUNCTION DOC]
    # Scenario: function declaring two module globals and conditionally assigning one, then
    # returning it. Checks the conditional (third statement): global_b live out, and global_a and
    # the parameter c live in.

    def test_fn(c):
      # [FUNCTION DOC]
      # Test subject: declares globals, assigns one in both branches of a conditional on the
      # other, returns it.
      global global_a
      global global_b
      if global_a:
        global_b = c
      else:
        global_b = c
      return global_b

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body
    self.assertHasLiveOut(fn_body[2], ('global_b',))
    self.assertHasLiveIn(fn_body[2], ('global_a', 'c'))

  def test_nonlocal_symbol(self):
    # [FUNCTION DOC]
    # Scenario: same as the global test but using enclosing-scope variables declared nonlocal.
    # Expects the conditional to have nonlocal_b live out and nonlocal_a plus c live in.

    nonlocal_a = 3
    nonlocal_b = 13

    def test_fn(c):
      # [FUNCTION DOC]
      # Test subject: declares nonlocals, assigns one in both branches of a conditional on the
      # other, returns it.
      nonlocal nonlocal_a
      nonlocal nonlocal_b
      if nonlocal_a:
        nonlocal_b = c
      else:
        nonlocal_b = c
      return nonlocal_b

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body
    self.assertHasLiveOut(fn_body[2], ('nonlocal_b',))
    self.assertHasLiveIn(fn_body[2], ('nonlocal_a', 'c'))


if __name__ == '__main__':
  test.main()
