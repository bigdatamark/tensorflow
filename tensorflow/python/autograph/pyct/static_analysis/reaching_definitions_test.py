# [MODULE DOC]
# Tests for the reaching definitions analysis. Defines a shared base test class that parses a
# function, runs qualified-name, activity, CFG and reaching-definitions passes, and offers
# assertions on definition counts, identity and defined-on-entry symbols; plus the concrete test
# case covering conditionals, loops, try/except, nested functions, deletion, global and nonlocal.
# Also defines two module-level globals used by the global-variable test. The base class is reused
# by the Python-3-only test module.
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
"""Tests for reaching_definitions module."""

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import naming
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import reaching_definitions
from tensorflow.python.platform import test


global_a = 7
global_b = 17


class ReachingDefinitionsAnalyzerTestBase(test.TestCase):
  # [CLASS DOC]
  # HELPER: shared base class for reaching definitions tests.
  # Test case base providing the analysis pipeline helper and assertion helpers for definitions
  # annotations. Has no tests of its own.

  def _parse_and_analyze(self, test_fn):
    # [FUNCTION DOC]
    # HELPER: runs the full analysis pipeline for tests.
    # Parses the source of the given function object into a syntax tree, resolves qualified
    # names, runs activity analysis, builds control flow graphs, then runs reaching definitions
    # with the default definition class. Returns the annotated function tree.
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
    node = reaching_definitions.resolve(node, ctx, graphs,
                                        reaching_definitions.Definition)
    return node

  def assertHasDefs(self, node, num):
    # [FUNCTION DOC]
    # HELPER: assertion for tests.
    # Asserts the node's definitions annotation has exactly num entries, each an instance of the
    # Definition class. Fails the test otherwise.
    defs = anno.getanno(node, anno.Static.DEFINITIONS)
    self.assertEqual(len(defs), num)
    for r in defs:
      self.assertIsInstance(r, reaching_definitions.Definition)

  def assertHasDefinedIn(self, node, expected):
    # [FUNCTION DOC]
    # HELPER: assertion for tests.
    # Asserts the set of names (as strings) in the node's defined-on-entry annotation equals
    # expected, which may be nothing/empty, a single name, or a tuple of names.
    defined_in = anno.getanno(node, anno.Static.DEFINED_VARS_IN)
    defined_in_str = set(str(v) for v in defined_in)
    if not expected:
      expected = ()
    if not isinstance(expected, tuple):
      expected = (expected,)
    self.assertSetEqual(defined_in_str, set(expected))

  def assertSameDef(self, first, second):
    # [FUNCTION DOC]
    # HELPER: assertion for tests.
    # Asserts both nodes have exactly one definition and that it is the very same object.
    self.assertHasDefs(first, 1)
    self.assertHasDefs(second, 1)
    self.assertIs(
        anno.getanno(first, anno.Static.DEFINITIONS)[0],
        anno.getanno(second, anno.Static.DEFINITIONS)[0])

  def assertNotSameDef(self, first, second):
    # [FUNCTION DOC]
    # HELPER: assertion for tests.
    # Asserts both nodes have exactly one definition each and that they are different objects.
    self.assertHasDefs(first, 1)
    self.assertHasDefs(second, 1)
    self.assertIsNot(
        anno.getanno(first, anno.Static.DEFINITIONS)[0],
        anno.getanno(second, anno.Static.DEFINITIONS)[0])


class ReachingDefinitionsAnalyzerTest(ReachingDefinitionsAnalyzerTestBase):
  # [CLASS DOC]
  # Concrete tests of reaching definitions: each test defines a small function, analyzes it and
  # asserts the number and identity of definitions reaching names and the defined-on-entry sets.

  def test_conditional(self):
    # [FUNCTION DOC]
    # Scenario: parameter reassigned, then conditionally reassigned, then returned. Asserts one
    # definition at the assignments and condition, two at the return, and both names defined on
    # entry to the if.

    def test_fn(a, b):
      # [FUNCTION DOC]
      # Analyzed function: assigns a, conditionally reassigns a on b, returns a.
      a = []
      if b:
        a = []
      return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasDefs(fn_body[0].targets[0], 1)
    self.assertHasDefs(fn_body[1].test, 1)
    self.assertHasDefs(fn_body[1].body[0].targets[0], 1)
    self.assertHasDefs(fn_body[2].value, 2)

    self.assertHasDefinedIn(fn_body[1], ('a', 'b'))

  def test_try_in_conditional(self):
    # [FUNCTION DOC]
    # Scenario: a try/except nested in an if. Asserts both the if and the try statement have
    # both names defined on entry.

    def test_fn(a, b):  # pylint:disable=unused-argument
      # [FUNCTION DOC]
      # Analyzed function: assigns a, then an if on b containing a try with an empty handler.
      a = []
      if b:
        try:
          pass
        except:  # pylint:disable=bare-except
          pass
      return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasDefinedIn(fn_body[1], ('a', 'b'))
    self.assertHasDefinedIn(fn_body[1].body[0], ('a', 'b'))

  def test_conditional_in_try_in_conditional(self):
    # [FUNCTION DOC]
    # Scenario: an if inside the try body of a try inside an if. Asserts the outer if, the try
    # and the inner if all have both names defined on entry (the exception type is not tracked).

    def test_fn(a, b):
      # [FUNCTION DOC]
      # Analyzed function: nested if, try, if with an assignment, and a handler that does nothing.
      a = []
      if b:
        try:
          if b:
            a = []
        except TestException:  # pylint:disable=undefined-variable,unused-variable
          pass
      return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasDefinedIn(fn_body[1], ('a', 'b'))
    self.assertHasDefinedIn(fn_body[1].body[0], ('a', 'b'))
    # Note: `TestException` and `e` are not tracked.
    self.assertHasDefinedIn(fn_body[1].body[0].body[0], ('a', 'b'))

  def test_conditional_in_except_in_conditional(self):
    # [FUNCTION DOC]
    # Scenario: an if inside an except handler of a try inside an if. Asserts the outer if, the
    # try and the if inside the handler body have both names defined on entry (the exception
    # type and bound name are not tracked).

    def test_fn(a, b):
      # [FUNCTION DOC]
      # Analyzed function: nested if, try with empty body, handler containing an if assignment.
      a = []
      if b:
        try:
          pass
        except TestException as e:  # pylint:disable=undefined-variable,unused-variable
          if b:
            a = []
      return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasDefinedIn(fn_body[1], ('a', 'b'))
    self.assertHasDefinedIn(fn_body[1].body[0], ('a', 'b'))
    # Note: `TestException` and `e` are not tracked.
    self.assertHasDefinedIn(fn_body[1].body[0].handlers[0].body[0], ('a', 'b'))

  def test_while(self):
    # [FUNCTION DOC]
    # Scenario: an infinite while loop reassigning a parameter. Asserts one definition before
    # the loop and for the loop body targets, and two for the first use inside the loop and the
    # final return, since the analysis does not know the loop condition is constant.

    def test_fn(a):
      # [FUNCTION DOC]
      # Analyzed function: calls max on a, loops forever assigning a to itself twice, returns a.
      max(a)
      while True:
        a = a
        a = a
      return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasDefs(fn_body[0].value.args[0], 1)
    self.assertHasDefs(fn_body[1].body[0].targets[0], 1)
    self.assertHasDefs(fn_body[1].body[1].targets[0], 1)
    self.assertHasDefs(fn_body[1].body[1].value, 1)
    # The loop does have an invariant test, but the CFG doesn't know that.
    self.assertHasDefs(fn_body[1].body[0].value, 2)
    self.assertHasDefs(fn_body[2].value, 2)

  def test_while_else(self):
    # [FUNCTION DOC]
    # Scenario: while loop with an augmented assignment, a conditional break, and an else clause
    # assigning y. Asserts definition counts for the initial assignment, loop test (two), loop
    # body, else target and the returned x and y (two each).

    def test_fn(x, i):
      # [FUNCTION DOC]
      # Analyzed function: sets y, loops on x adding i with a conditional break, else sets y.
      y = 0
      while x:
        x += i
        if i:
          break
      else:
        y = 1
      return x, y

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasDefs(fn_body[0].targets[0], 1)
    self.assertHasDefs(fn_body[1].test, 2)
    self.assertHasDefs(fn_body[1].body[0].target, 1)
    self.assertHasDefs(fn_body[1].body[1].test, 1)
    self.assertHasDefs(fn_body[1].orelse[0].targets[0], 1)
    self.assertHasDefs(fn_body[2].value.elts[0], 2)
    self.assertHasDefs(fn_body[2].value.elts[1], 2)

  def test_for_else(self):
    # [FUNCTION DOC]
    # Scenario: for loop over x with break/continue and an else clause assigning y. Asserts one
    # definition for the loop target, body and else targets, and two for the returned x and y.

    def test_fn(x, i):
      # [FUNCTION DOC]
      # Analyzed function: sets y, loops over x adding to x with break/continue, else sets y.
      y = 0
      for i in x:
        x += i
        if i:
          break
        else:
          continue
      else:
        y = 1
      return x, y

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasDefs(fn_body[0].targets[0], 1)
    self.assertHasDefs(fn_body[1].target, 1)
    self.assertHasDefs(fn_body[1].body[0].target, 1)
    self.assertHasDefs(fn_body[1].body[1].test, 1)
    self.assertHasDefs(fn_body[1].orelse[0].targets[0], 1)
    self.assertHasDefs(fn_body[2].value.elts[0], 2)
    self.assertHasDefs(fn_body[2].value.elts[1], 2)

  def test_nested_functions(self):
    # [FUNCTION DOC]
    # Scenario: a nested function defined in an if branch reads a variable assigned outside it.
    # Asserts normal counts in the outer function and zero definitions for the variable read
    # inside the inner function, because of late binding of closures.

    def test_fn(a, b):
      # [FUNCTION DOC]
      # Analyzed function: assigns a, conditionally reassigns a and defines and calls an inner
      # function returning a, then returns a.
      a = []
      if b:
        a = []

        def foo():
          # [FUNCTION DOC]
          # Inner function returning the enclosing variable a.
          return a

        foo()

      return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body
    def_of_a_in_if = fn_body[1].body[0].targets[0]

    self.assertHasDefs(fn_body[0].targets[0], 1)
    self.assertHasDefs(fn_body[1].test, 1)
    self.assertHasDefs(def_of_a_in_if, 1)
    self.assertHasDefs(fn_body[2].value, 2)

    inner_fn_body = fn_body[1].body[1].body
    def_of_a_in_foo = inner_fn_body[0].value
    # Even though `a` is visible in the inner function above, the late binding
    # makes it impossible to assume that the same value will be visible at
    # call time.
    self.assertHasDefs(def_of_a_in_foo, 0)

  def test_nested_functions_isolation(self):
    # [FUNCTION DOC]
    # Scenario: an inner function assigns a variable of the same name as its parent's. Asserts
    # the return values of parent and child use different definitions (the name is local to the
    # child).

    def test_fn(a):
      # [FUNCTION DOC]
      # Analyzed function: assigns a, defines and calls an inner function, returns a.
      a = 0

      def child():
        # [FUNCTION DOC]
        # Inner function: assigns its own local a and returns it.
        a = 1
        return a

      child()
      return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    parent_return = fn_body[3]
    child_return = fn_body[1].body[1]
    # The assignment `a = 1` makes `a` local to `child`.
    self.assertNotSameDef(parent_return.value, child_return.value)

  def test_function_call_in_with(self):
    # [FUNCTION DOC]
    # Scenario: a with statement whose context expression calls a locally defined function on a
    # parameter. Asserts the called function name has no definitions and the argument has one.

    def foo(_):
      # [FUNCTION DOC]
      # Helper function that does nothing, used as the context expression callee.
      pass

    def test_fn(a):
      # [FUNCTION DOC]
      # Analyzed function: with foo(a) returning a inside the block.
      with foo(a):
        return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasDefs(fn_body[0].items[0].context_expr.func, 0)
    self.assertHasDefs(fn_body[0].items[0].context_expr.args[0], 1)

  def test_mutation_subscript(self):
    # [FUNCTION DOC]
    # Scenario: a list is created then mutated by item assignment, then returned. Asserts the
    # creation, the mutated name and the later use share the same single definition.

    def test_fn(a):
      # [FUNCTION DOC]
      # Analyzed function: creates a list, assigns an item of it, returns it.
      l = []
      l[0] = a
      return l

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    creation = fn_body[0].targets[0]
    mutation = fn_body[1].targets[0].value
    use = fn_body[2].value
    self.assertSameDef(creation, mutation)
    self.assertSameDef(creation, use)

  def test_deletion_partial(self):
    # [FUNCTION DOC]
    # Scenario: a variable is deleted in one branch and reassigned in the other. Asserts the use
    # after the branches reaches only the else-branch definition, not the original.

    def test_fn(a):
      # [FUNCTION DOC]
      # Analyzed function: assigns a, deletes a in the if branch, reassigns in else, returns a.
      a = 0
      if a:
        del a
      else:
        a = 1
      return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    first_def = fn_body[0].targets[0]
    second_def = fn_body[1].orelse[0].targets[0]
    use = fn_body[2].value
    self.assertNotSameDef(use, first_def)
    self.assertSameDef(use, second_def)

  def test_deletion_total(self):
    # [FUNCTION DOC]
    # Scenario: a variable is assigned in both branches and then deleted. Asserts the later use
    # has no reaching definitions.

    def test_fn(a):
      # [FUNCTION DOC]
      # Analyzed function: assigns a in both branches, deletes a, then returns a.
      if a:
        a = 0
      else:
        a = 1
      del a
      return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    use = fn_body[2].value
    self.assertHasDefs(use, 0)

  def test_replacement(self):
    # [FUNCTION DOC]
    # Scenario: a parameter is replaced by the result of a call using it. Asserts the argument
    # use shares the parameter's definition, the assignment target has a different one, and the
    # return shares the target's definition.

    def foo(a):
      # [FUNCTION DOC]
      # Helper function returning its argument.
      return a

    def test_fn(a):
      # [FUNCTION DOC]
      # Analyzed function: reassigns a to foo applied to a, returns a.
      a = foo(a)
      return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    param = node.args.args[0]
    source = fn_body[0].value.args[0]
    target = fn_body[0].targets[0]
    retval = fn_body[1].value
    self.assertSameDef(param, source)
    self.assertNotSameDef(source, target)
    self.assertSameDef(target, retval)

  def test_comprehension_leaking(self):
    # [FUNCTION DOC]
    # Scenario: a list comprehension variable is used after the comprehension. Asserts the use
    # has no definitions, since in Python 3 comprehension variables do not leak.

    def test_fn(a):
      # [FUNCTION DOC]
      # Analyzed function: builds a list comprehension over a, then returns the loop variable.
      _ = [x for x in a]
      return x  # pylint:disable=undefined-loop-variable

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    listcomp_target = fn_body[0].value.generators[0].target
    retval = fn_body[1].value

    # Python2 leaks list comprehension symbols. Python3 doesn't.
    # For details, see:
    # https://stackoverflow.com/questions/4198906/list-comprehension-rebinds-names-even-after-scope-of-comprehension-is-this-righ
    self.assertHasDefs(retval, 0)

  def test_function_definition(self):
    # [FUNCTION DOC]
    # Scenario: a name bound by a nested function definition is tested and conditionally
    # reassigned. Asserts one definition at the test and assignment, two at the return, and the
    # name defined on entry to the if.

    def test_fn():
      # [FUNCTION DOC]
      # Analyzed function: defines inner function a, conditionally rebinds a to nothing, returns a.
      def a():
        # [FUNCTION DOC]
        # Empty inner function whose name is the tracked variable.
        pass
      if a:  # pylint:disable=using-constant-test
        a = None
      return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasDefs(fn_body[1].test, 1)
    self.assertHasDefs(fn_body[1].body[0].targets[0], 1)
    self.assertHasDefs(fn_body[2].value, 2)

    self.assertHasDefinedIn(fn_body[1], ('a',))

  def test_definitions_in_except_block(self):
    # [FUNCTION DOC]
    # Scenario: a variable is assigned only in an except handler of a try with an empty body,
    # then tested and reassigned. Asserts one definition at the test, one at the assignment, two
    # at the return, and the name defined on entry to the if.

    def test_fn():
      # [FUNCTION DOC]
      # Analyzed function: try/except assigning a in the handler, then conditional use of a.
      try:
        pass
      except ValueError:
        a = None
      if a:  # pylint:disable=using-constant-test
        a = None
      return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasDefs(fn_body[1].test, 1)
    self.assertHasDefs(fn_body[1].body[0].targets[0], 1)
    self.assertHasDefs(fn_body[2].value, 2)

    self.assertHasDefinedIn(fn_body[1], ('a',))

  def test_definitions_in_except_block_of_raising_try(self):
    # [FUNCTION DOC]
    # Same as the except-block test but the try body raises an error. Asserts the same counts:
    # one at the test and assignment, two at the return, name defined on entry to the if.

    def test_fn():
      # [FUNCTION DOC]
      # Analyzed function: try raising an error, handler assigns a, then conditional use of a.
      try:
        raise ValueError()
      except ValueError:
        a = None
      if a:  # pylint:disable=using-constant-test
        a = None
      return a

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasDefs(fn_body[1].test, 1)
    self.assertHasDefs(fn_body[1].body[0].targets[0], 1)
    self.assertHasDefs(fn_body[2].value, 2)

    self.assertHasDefinedIn(fn_body[1], ('a',))

  def test_global(self):
    # [FUNCTION DOC]
    # Scenario: a function declares two module globals, tests one, conditionally assigns the
    # other and returns both. Asserts one definition at the test and assignment, one for the
    # first returned name (same as the test), two for the second, and both names defined on
    # entry to the if.

    def test_fn():
      # [FUNCTION DOC]
      # Analyzed function: declares both globals, conditionally assigns the second, returns both.
      global global_a
      global global_b
      if global_a:
        global_b = []
      return global_a, global_b

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasDefs(fn_body[2].test, 1)
    self.assertHasDefs(fn_body[2].body[0].targets[0], 1)
    self.assertHasDefs(fn_body[3].value.elts[0], 1)
    self.assertHasDefs(fn_body[3].value.elts[1], 2)

    self.assertSameDef(fn_body[2].test, fn_body[3].value.elts[0])

    self.assertHasDefinedIn(fn_body[2], ('global_a', 'global_b'))

  def test_nonlocal(self):
    # [FUNCTION DOC]
    # Scenario: same as the global test but with enclosing-scope (nonlocal) variables. Asserts
    # the same counts and identity relations and that both names are defined on entry to the if.

    a = 3
    b = 13

    def test_fn():
      # [FUNCTION DOC]
      # Analyzed function: declares both outer names nonlocal, conditionally assigns the second,
      # returns both.
      nonlocal a
      nonlocal b
      if a:
        b = []
      return a, b

    node = self._parse_and_analyze(test_fn)
    fn_body = node.body

    self.assertHasDefs(fn_body[2].test, 1)
    self.assertHasDefs(fn_body[2].body[0].targets[0], 1)
    self.assertHasDefs(fn_body[3].value.elts[0], 1)
    self.assertHasDefs(fn_body[3].value.elts[1], 2)

    self.assertSameDef(fn_body[2].test, fn_body[3].value.elts[0])

    self.assertHasDefinedIn(fn_body[2], ('a', 'b'))

  def test_nonlocal_in_nested_function(self):
    # [FUNCTION DOC]
    # Scenario: an inner function declares the outer function's locals nonlocal. Asserts, on the
    # inner body, counts (one, one, one, two), identity of the tested and returned first name,
    # and both names defined on entry to the inner if.

    a = 3
    b = 13

    def test_fn():
      # [FUNCTION DOC]
      # Analyzed outer function: defines two locals and an inner function, returns its call.
      a = 3
      b = 13

      def local_fn():
        # [FUNCTION DOC]
        # Inner function: nonlocal declarations, conditional assignment, returns both names.
        nonlocal a, b
        if a:
          b = []
        return a, b

      return local_fn()

    node = self._parse_and_analyze(test_fn)
    local_body = node.body[2].body

    self.assertHasDefs(local_body[1].test, 1)
    self.assertHasDefs(local_body[1].body[0].targets[0], 1)
    self.assertHasDefs(local_body[2].value.elts[0], 1)
    self.assertHasDefs(local_body[2].value.elts[1], 2)

    self.assertSameDef(local_body[1].test, local_body[2].value.elts[0])

    # Note: the function name is visible inside the function body. But it's
    # a closure variable, not a local.
    #
    # Example:
    #
    #   >>> def f():
    #   ...  print(f)
    #   >>> g = f
    #   >>> f = 'something else'
    #   >>> g()
    #   something else
    #
    self.assertHasDefinedIn(local_body[1], ('a', 'b'))


if __name__ == '__main__':
  test.main()
