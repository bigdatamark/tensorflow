# [MODULE DOC]
# Unit tests for the AST-based source upgrader in the sibling ast_edits module, which underlies the
# TensorFlow upgrade tools. It defines several small "spec" classes (derived from the no-update
# specification) that describe hypothetical API changes to functions f, g, g2 and h: keyword
# renames, argument reorders, removal of deprecated keyword aliases, module deprecations and
# import renames. A single test case class feeds snippets of source text through the upgrader and
# checks the rewritten text, the generated report and the collected errors. It also covers
# symlink handling when upgrading directory trees in place or by copying.
# Nothing is registered at import time; the file ends by running the test runner when executed
# directly. Not a helper module: it is a test module.
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
"""Tests for ast_edits which is used in tf upgraders.

All of the tests assume that we want to change from an API containing

    import foo as f

    def f(a, b, kw1, kw2): ...
    def g(a, b, kw1, c, kw1_alias): ...
    def g2(a, b, kw1, c, d, kw1_alias): ...
    def h(a, kw1, kw2, kw1_alias, kw2_alias): ...

and the changes to the API consist of renaming, reordering, and/or removing
arguments. Thus, we want to be able to generate changes to produce each of the
following new APIs:

    import bar as f

    def f(a, b, kw1, kw3): ...
    def f(a, b, kw2, kw1): ...
    def f(a, b, kw3, kw1): ...
    def g(a, b, kw1, c): ...
    def g(a, b, c, kw1): ...
    def g2(a, b, kw1, c, d): ...
    def g2(a, b, c, d, kw1): ...
    def h(a, kw1, kw2): ...

"""

import ast
import io
import os

from tensorflow.python.framework import test_util
from tensorflow.python.platform import test as test_lib
from tensorflow.tools.compatibility import ast_edits


class ModuleDeprecationSpec(ast_edits.NoUpdateSpec):
  # [CLASS DOC]
  # Test specification that marks the module path "a.b" as deprecated with an error-level message
  # saying it is evil. Starts from the no-update specification (no renames, reorders or warnings)
  # and only adds one entry to the module deprecation mapping. Used by the module deprecation test.
  """A specification which deprecates 'a.b'."""

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor. Initializes the base no-update specification and then adds the entry for "a.b" to
    # the module deprecation mapping, with error severity and the message "a.b is evil.".
    # Takes no arguments, returns nothing; mutates only the new instance.
    ast_edits.NoUpdateSpec.__init__(self)
    self.module_deprecations.update({"a.b": (ast_edits.ERROR, "a.b is evil.")})


class RenameKeywordSpec(ast_edits.NoUpdateSpec):
  # [CLASS DOC]
  # Test specification describing an API change where keyword argument kw2 of function f is renamed
  # to kw3 (new signature has a, b, kw1, kw3). Built on the no-update specification; the rename is
  # installed by a separate method so subclasses can re-apply it. Base class of the combined
  # reorder-and-rename specification.
  """A specification where kw2 gets renamed to kw3.

  The new API is

    def f(a, b, kw1, kw3): ...

  """

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor. Initializes the base no-update specification, then calls the method that installs
    # the keyword rename. Takes no arguments, returns nothing.
    ast_edits.NoUpdateSpec.__init__(self)
    self.update_renames()

  def update_renames(self):
    # [FUNCTION DOC]
    # Registers, in the per-function keyword rename mapping, that for function f the keyword kw2 is
    # renamed to kw3. Overwrites any existing entry for f. Takes no arguments, returns nothing; mutates
    # the specification instance.
    self.function_keyword_renames["f"] = {"kw2": "kw3"}


class ReorderKeywordSpec(ast_edits.NoUpdateSpec):
  # [CLASS DOC]
  # Test specification describing an API change where kw2 of function f moves in front of kw1 (new
  # signature has a, b, kw2, kw1). Built on the no-update specification; the reorder is installed by
  # a separate method so subclasses can re-apply it. Base class of the combined reorder-and-rename
  # specification.
  """A specification where kw2 gets moved in front of kw1.

  The new API is

    def f(a, b, kw2, kw1): ...

  """

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor. Initializes the base no-update specification, then calls the method that installs
    # the argument reorder for f. Takes no arguments, returns nothing.
    ast_edits.NoUpdateSpec.__init__(self)
    self.update_reorders()

  def update_reorders(self):
    # [FUNCTION DOC]
    # Registers, in the per-function reorder mapping, the OLD parameter order for function f: a, b, kw1,
    # kw2 (the upgrader derives the new order from the keyword names it sees). Overwrites any existing
    # entry for f. Takes no arguments, returns nothing; mutates the specification instance.
    # Note that these should be in the old order.
    self.function_reorders["f"] = ["a", "b", "kw1", "kw2"]


class ReorderAndRenameKeywordSpec(ReorderKeywordSpec, RenameKeywordSpec):
  # [CLASS DOC]
  # Test specification combining the reorder and rename specifications: kw2 of function f is both
  # renamed to kw3 and moved in front of kw1 (new signature a, b, kw3, kw1). Inherits from both
  # the reorder and the rename specification classes.
  """A specification where kw2 gets moved in front of kw1 and is changed to kw3.

  The new API is

    def f(a, b, kw3, kw1): ...

  """

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor. Calls the constructors of both parent specifications explicitly and then calls the
    # rename and reorder installation methods again, so both mappings for f are populated.
    # Takes no arguments, returns nothing.
    ReorderKeywordSpec.__init__(self)
    RenameKeywordSpec.__init__(self)
    self.update_renames()
    self.update_reorders()


class RemoveDeprecatedAliasKeyword(ast_edits.NoUpdateSpec):
  # [CLASS DOC]
  # Test specification where the deprecated keyword alias kw1_alias is replaced by kw1 in functions g
  # and g2 (new signatures: g with a, b, kw1, c and g2 with a, b, kw1, c, d). Built on the no-update
  # specification; parent of the alias-removal-plus-reorder specification.
  """A specification where kw1_alias is removed in g.

  The new API is

    def g(a, b, kw1, c): ...
    def g2(a, b, kw1, c, d): ...

  """

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor. Initializes the base no-update specification and registers a keyword rename of
    # kw1_alias to kw1 for both g and g2 in the per-function keyword rename mapping.
    # Takes no arguments, returns nothing.
    ast_edits.NoUpdateSpec.__init__(self)
    self.function_keyword_renames["g"] = {"kw1_alias": "kw1"}
    self.function_keyword_renames["g2"] = {"kw1_alias": "kw1"}


class RemoveDeprecatedAliasAndReorderRest(RemoveDeprecatedAliasKeyword):
  # [CLASS DOC]
  # Test specification that extends the alias-removal specification by also moving kw1 to the end of
  # the parameters of g and g2 (new signatures: g with a, b, c, kw1 and g2 with a, b, c, d, kw1).
  # KNOWN ISSUE: its docstring repeats the one of the parent ("kw1_alias is removed in g") and does
  # not mention the reordering or g2.
  """A specification where kw1_alias is removed in g.

  The new API is

    def g(a, b, c, kw1): ...
    def g2(a, b, c, d, kw1): ...

  """

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor. Runs the parent constructor (which installs the kw1_alias to kw1 renames) and then
    # registers the old parameter orders in the reorder mapping: a, b, kw1, c for g and a, b, kw1, c,
    # d for g2. Takes no arguments, returns nothing.
    RemoveDeprecatedAliasKeyword.__init__(self)
    # Note that these should be in the old order.
    self.function_reorders["g"] = ["a", "b", "kw1", "c"]
    self.function_reorders["g2"] = ["a", "b", "kw1", "c", "d"]


class RemoveMultipleKeywordArguments(ast_edits.NoUpdateSpec):
  # [CLASS DOC]
  # Test specification where both deprecated aliases kw1_alias and kw2_alias of function h are
  # renamed to kw1 and kw2 respectively (new signature a, kw1, kw2). Built on the no-update
  # specification.
  """A specification where both keyword aliases are removed from h.

  The new API is

    def h(a, kw1, kw2): ...

  """

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor. Initializes the base no-update specification and registers a mapping with two
    # renames (kw1_alias to kw1, kw2_alias to kw2) for function h. Takes no arguments, returns nothing.
    ast_edits.NoUpdateSpec.__init__(self)
    self.function_keyword_renames["h"] = {
        "kw1_alias": "kw1",
        "kw2_alias": "kw2",
    }


class RenameImports(ast_edits.NoUpdateSpec):
  # [CLASS DOC]
  # Test specification for import renaming: the module foo is renamed to bar, except for anything
  # under the prefix foo.baz which is excluded. Built on the no-update specification.
  """Specification for renaming imports."""

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor. Initializes the base no-update specification and then sets the import rename
    # mapping with a single entry: foo maps to an import-rename record with new name bar and the
    # excluded prefix list containing foo.baz. Takes no arguments, returns nothing.
    ast_edits.NoUpdateSpec.__init__(self)
    self.import_renames = {
        "foo": ast_edits.ImportRename(
            "bar",
            excluded_prefixes=["foo.baz"])
    }


class TestAstEdits(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case (on the TensorFlow test base class) covering the AST code upgrader: keyword renames,
  # reordering, deprecated alias removal, warnings, module deprecation errors, import and
  # from-import renaming (including exclusions and imports inside functions), and handling of
  # symbolic links when upgrading trees in place or into an output directory. Shared fixture is the
  # upgrade helper method, which runs a specification over a text snippet in memory.

  def _upgrade(self, spec, old_file_text):
    # [FUNCTION DOC]
    # HELPER: shared test utility that runs the upgrader on in-memory text for all tests in this class.
    # Runs the code upgrader built from the given specification on the given source text, using
    # in-memory text streams and the fake file names test.py and test_out.py.
    # Inputs: spec is a specification object; old_file_text is the source code as a string.
    # Returns a pair: first a triple of (count, report, errors) as produced by the upgrader, second the
    # rewritten source text as a string. No files are touched; exceptions from the upgrader propagate.
    in_file = io.StringIO(old_file_text)
    out_file = io.StringIO()
    upgrader = ast_edits.ASTCodeUpgrader(spec)
    count, report, errors = (
        upgrader.process_opened_file("test.py", in_file,
                                     "test_out.py", out_file))
    return (count, report, errors), out_file.getvalue()

  def testModuleDeprecation(self):
    # [FUNCTION DOC]
    # Scenario: source text uses the deprecated module path a.b twice (a call a.b.c with argument a.b.x)
    # under the module deprecation specification. Asserts the text is unchanged, and the errors list
    # mentions "Using member a.b.c" with position 1:0 for the first error and position 1:6 for the
    # second error.
    # KNOWN ISSUE: the third assertion repeats the "Using member a.b.c" check on the first error
    # instead of checking the second error's message (probably meant to be the member a.b.x).
    text = "a.b.c(a.b.x)"
    (_, _, errors), new_text = self._upgrade(ModuleDeprecationSpec(), text)
    self.assertEqual(text, new_text)
    self.assertIn("Using member a.b.c", errors[0])
    self.assertIn("1:0", errors[0])
    self.assertIn("Using member a.b.c", errors[0])
    self.assertIn("1:6", errors[1])

  def testNoTransformIfNothingIsSupplied(self):
    # [FUNCTION DOC]
    # Scenario: the no-update specification is applied to two calls of f (keyword and positional
    # arguments). Asserts that the output text equals the input in both cases.
    text = "f(a, b, kw1=c, kw2=d)\n"
    _, new_text = self._upgrade(ast_edits.NoUpdateSpec(), text)
    self.assertEqual(new_text, text)

    text = "f(a, b, c, d)\n"
    _, new_text = self._upgrade(ast_edits.NoUpdateSpec(), text)
    self.assertEqual(new_text, text)

  def testKeywordRename(self):
    # [FUNCTION DOC]
    # Scenario: rename specification (kw2 to kw3). Asserts that a keyword call has kw2 renamed to kw3,
    # that a purely positional call is unchanged, that positional star-args produce no manual-check
    # warning in the report, and that a call using double-star keyword arguments produces the
    # "Manual check required" warning.
    # KNOWN ISSUE: a comment says star-args "should warn" but the test asserts that no warning appears.
    """Test that we get the expected result if renaming kw2 to kw3."""
    text = "f(a, b, kw1=c, kw2=d)\n"
    expected = "f(a, b, kw1=c, kw3=d)\n"
    (_, report, _), new_text = self._upgrade(RenameKeywordSpec(), text)
    self.assertEqual(new_text, expected)
    self.assertNotIn("Manual check required", report)

    # No keywords specified, no reordering, so we should get input as output
    text = "f(a, b, c, d)\n"
    (_, report, _), new_text = self._upgrade(RenameKeywordSpec(), text)
    self.assertEqual(new_text, text)
    self.assertNotIn("Manual check required", report)

    # Positional *args passed in that we cannot inspect, should warn
    text = "f(a, *args)\n"
    (_, report, _), _ = self._upgrade(RenameKeywordSpec(), text)
    self.assertNotIn("Manual check required", report)

    # **kwargs passed in that we cannot inspect, should warn
    text = "f(a, b, kw1=c, **kwargs)\n"
    (_, report, _), _ = self._upgrade(RenameKeywordSpec(), text)
    self.assertIn("Manual check required", report)

  def testKeywordReorderWithParens(self):
    # [FUNCTION DOC]
    # Scenario: reorder specification applied to a call whose positional arguments are wrapped in
    # redundant parentheses. Asserts the output is one of several acceptable forms: unchanged text,
    # all arguments turned into keywords with parentheses preserved, or with canonical parentheses.
    """Test that we get the expected result if there are parens around args."""
    text = "f((a), ( ( b ) ))\n"
    acceptable_outputs = [
        # No change is a valid output
        text,
        # Also cases where all arguments are fully specified are allowed
        "f(a=(a), b=( ( b ) ))\n",
        # Making the parens canonical is ok
        "f(a=(a), b=((b)))\n",
    ]
    _, new_text = self._upgrade(ReorderKeywordSpec(), text)
    self.assertIn(new_text, acceptable_outputs)

  def testKeywordReorder(self):
    # [FUNCTION DOC]
    # Scenario: reorder specification (kw2 now before kw1). Asserts that keyword calls come out as one
    # of the acceptable forms (unchanged, reordered or fully keyworded) without a manual-check
    # warning; positional calls are reordered or converted to keywords; positional star-args produce
    # the manual-check warning; double-star keyword arguments produce no warning.
    # KNOWN ISSUE: comments on the last two cases say the opposite of what the assertions check
    # (they name the wrong kind of unknown arguments as the one that should warn).
    """Test that we get the expected result if kw2 is now before kw1."""
    text = "f(a, b, kw1=c, kw2=d)\n"
    acceptable_outputs = [
        # No change is a valid output
        text,
        # Just reordering the kw.. args is also ok
        "f(a, b, kw2=d, kw1=c)\n",
        # Also cases where all arguments are fully specified are allowed
        "f(a=a, b=b, kw1=c, kw2=d)\n",
        "f(a=a, b=b, kw2=d, kw1=c)\n",
    ]
    (_, report, _), new_text = self._upgrade(ReorderKeywordSpec(), text)
    self.assertIn(new_text, acceptable_outputs)
    self.assertNotIn("Manual check required", report)

    # Keywords are reordered, so we should reorder arguments too
    text = "f(a, b, c, d)\n"
    acceptable_outputs = [
        "f(a, b, d, c)\n",
        "f(a=a, b=b, kw1=c, kw2=d)\n",
        "f(a=a, b=b, kw2=d, kw1=c)\n",
    ]
    (_, report, _), new_text = self._upgrade(ReorderKeywordSpec(), text)
    self.assertIn(new_text, acceptable_outputs)
    self.assertNotIn("Manual check required", report)

    # Positional *args passed in that we cannot inspect, should warn
    text = "f(a, b, *args)\n"
    (_, report, _), _ = self._upgrade(ReorderKeywordSpec(), text)
    self.assertIn("Manual check required", report)

    # **kwargs passed in that we cannot inspect, should warn
    text = "f(a, b, kw1=c, **kwargs)\n"
    (_, report, _), _ = self._upgrade(ReorderKeywordSpec(), text)
    self.assertNotIn("Manual check required", report)

  def testKeywordReorderAndRename(self):
    # [FUNCTION DOC]
    # Scenario: combined reorder-and-rename specification. Asserts keyword calls and positional calls
    # are rewritten to one of the acceptable forms using kw3 and reordering, with no manual-check
    # warning, and that both star-args and double-star keyword arguments produce the manual-check
    # warning.
    """Test that we get the expected result if kw2 is renamed and moved."""
    text = "f(a, b, kw1=c, kw2=d)\n"
    acceptable_outputs = [
        "f(a, b, kw3=d, kw1=c)\n",
        "f(a=a, b=b, kw1=c, kw3=d)\n",
        "f(a=a, b=b, kw3=d, kw1=c)\n",
    ]
    (_, report, _), new_text = self._upgrade(
        ReorderAndRenameKeywordSpec(), text)
    self.assertIn(new_text, acceptable_outputs)
    self.assertNotIn("Manual check required", report)

    # Keywords are reordered, so we should reorder arguments too
    text = "f(a, b, c, d)\n"
    acceptable_outputs = [
        "f(a, b, d, c)\n",
        "f(a=a, b=b, kw1=c, kw3=d)\n",
        "f(a=a, b=b, kw3=d, kw1=c)\n",
    ]
    (_, report, _), new_text = self._upgrade(
        ReorderAndRenameKeywordSpec(), text)
    self.assertIn(new_text, acceptable_outputs)
    self.assertNotIn("Manual check required", report)

    # Positional *args passed in that we cannot inspect, should warn
    text = "f(a, *args, kw1=c)\n"
    (_, report, _), _ = self._upgrade(ReorderAndRenameKeywordSpec(), text)
    self.assertIn("Manual check required", report)

    # **kwargs passed in that we cannot inspect, should warn
    text = "f(a, b, kw1=c, **kwargs)\n"
    (_, report, _), _ = self._upgrade(ReorderAndRenameKeywordSpec(), text)
    self.assertIn("Manual check required", report)

  def testRemoveDeprecatedKeywordAlias(self):
    # [FUNCTION DOC]
    # Scenario: alias-removal specification for g. Asserts that calls not using the alias are unchanged
    # (or fully keyworded), that purely positional calls are unchanged, and that kw1_alias is renamed to
    # kw1 whether it appears in the middle or last, with any acceptable ordering of keywords.
    """Test that we get the expected result if a keyword alias is removed."""
    text = "g(a, b, kw1=x, c=c)\n"
    acceptable_outputs = [
        # Not using deprecated alias, so original is ok
        text,
        "g(a=a, b=b, kw1=x, c=c)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasKeyword(), text)
    self.assertIn(new_text, acceptable_outputs)

    # No keyword used, should be no change
    text = "g(a, b, x, c)\n"
    _, new_text = self._upgrade(RemoveDeprecatedAliasKeyword(), text)
    self.assertEqual(new_text, text)

    # If we used the alias, it should get renamed
    text = "g(a, b, kw1_alias=x, c=c)\n"
    acceptable_outputs = [
        "g(a, b, kw1=x, c=c)\n",
        "g(a, b, c=c, kw1=x)\n",
        "g(a=a, b=b, kw1=x, c=c)\n",
        "g(a=a, b=b, c=c, kw1=x)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasKeyword(), text)
    self.assertIn(new_text, acceptable_outputs)

    # It should get renamed even if it's last
    text = "g(a, b, c=c, kw1_alias=x)\n"
    acceptable_outputs = [
        "g(a, b, kw1=x, c=c)\n",
        "g(a, b, c=c, kw1=x)\n",
        "g(a=a, b=b, kw1=x, c=c)\n",
        "g(a=a, b=b, c=c, kw1=x)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasKeyword(), text)
    self.assertIn(new_text, acceptable_outputs)

  def testRemoveDeprecatedKeywordAndReorder(self):
    # [FUNCTION DOC]
    # Scenario: alias-removal-plus-reorder specification for g. Asserts that keyword calls and
    # positional calls are reordered so kw1 comes last (or fully keyworded).
    # KNOWN ISSUE: the last two cases use the plain alias-removal specification rather than the
    # reorder one, so they do not test reordering; a comment about "c and d" refers to g2.
    """Test for when a keyword alias is removed and args are reordered."""
    text = "g(a, b, kw1=x, c=c)\n"
    acceptable_outputs = [
        "g(a, b, c=c, kw1=x)\n",
        "g(a=a, b=b, kw1=x, c=c)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasAndReorderRest(), text)
    self.assertIn(new_text, acceptable_outputs)

    # Keywords are reordered, so we should reorder arguments too
    text = "g(a, b, x, c)\n"
    # Don't accept an output which doesn't reorder c and d
    acceptable_outputs = [
        "g(a, b, c, x)\n",
        "g(a=a, b=b, kw1=x, c=c)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasAndReorderRest(), text)
    self.assertIn(new_text, acceptable_outputs)

    # If we used the alias, it should get renamed
    text = "g(a, b, kw1_alias=x, c=c)\n"
    acceptable_outputs = [
        "g(a, b, kw1=x, c=c)\n",
        "g(a, b, c=c, kw1=x)\n",
        "g(a=a, b=b, kw1=x, c=c)\n",
        "g(a=a, b=b, c=c, kw1=x)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasKeyword(), text)
    self.assertIn(new_text, acceptable_outputs)

    # It should get renamed and reordered even if it's last
    text = "g(a, b, c=c, kw1_alias=x)\n"
    acceptable_outputs = [
        "g(a, b, kw1=x, c=c)\n",
        "g(a, b, c=c, kw1=x)\n",
        "g(a=a, b=b, kw1=x, c=c)\n",
        "g(a=a, b=b, c=c, kw1=x)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasKeyword(), text)
    self.assertIn(new_text, acceptable_outputs)

  def testRemoveDeprecatedKeywordAndReorder2(self):
    # [FUNCTION DOC]
    # Scenario: same as the previous test but for function g2, which has more arguments. Asserts the
    # acceptable outputs for keyword calls, positional calls, calls using the alias, and calls with
    # keyword arguments in scrambled order.
    # KNOWN ISSUE: the alias cases use the plain alias-removal specification, not the reorder one.
    """Same as testRemoveDeprecatedKeywordAndReorder but on g2 (more args)."""
    text = "g2(a, b, kw1=x, c=c, d=d)\n"
    acceptable_outputs = [
        "g2(a, b, c=c, d=d, kw1=x)\n",
        "g2(a=a, b=b, kw1=x, c=c, d=d)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasAndReorderRest(), text)
    self.assertIn(new_text, acceptable_outputs)

    # Keywords are reordered, so we should reorder arguments too
    text = "g2(a, b, x, c, d)\n"
    # Don't accept an output which doesn't reorder c and d
    acceptable_outputs = [
        "g2(a, b, c, d, x)\n",
        "g2(a=a, b=b, kw1=x, c=c, d=d)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasAndReorderRest(), text)
    self.assertIn(new_text, acceptable_outputs)

    # If we used the alias, it should get renamed
    text = "g2(a, b, kw1_alias=x, c=c, d=d)\n"
    acceptable_outputs = [
        "g2(a, b, kw1=x, c=c, d=d)\n",
        "g2(a, b, c=c, d=d, kw1=x)\n",
        "g2(a=a, b=b, kw1=x, c=c, d=d)\n",
        "g2(a=a, b=b, c=c, d=d, kw1=x)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasKeyword(), text)
    self.assertIn(new_text, acceptable_outputs)

    # It should get renamed and reordered even if it's not in order
    text = "g2(a, b, d=d, c=c, kw1_alias=x)\n"
    acceptable_outputs = [
        "g2(a, b, kw1=x, c=c, d=d)\n",
        "g2(a, b, c=c, d=d, kw1=x)\n",
        "g2(a, b, d=d, c=c, kw1=x)\n",
        "g2(a=a, b=b, kw1=x, c=c, d=d)\n",
        "g2(a=a, b=b, c=c, d=d, kw1=x)\n",
        "g2(a=a, b=b, d=d, c=c, kw1=x)\n",
    ]
    _, new_text = self._upgrade(RemoveDeprecatedAliasKeyword(), text)
    self.assertIn(new_text, acceptable_outputs)

  def testRemoveMultipleKeywords(self):
    # [FUNCTION DOC]
    # Scenario: specification renaming two aliases of h at once. Asserts that calls with new names or
    # positional arguments are unchanged, and that calls using old names (in order, reversed, or mixed
    # with new names) produce one of the acceptable outputs.
    """Remove multiple keywords at once."""
    # Not using deprecated keywords -> no rename
    text = "h(a, kw1=x, kw2=y)\n"
    _, new_text = self._upgrade(RemoveMultipleKeywordArguments(), text)
    self.assertEqual(new_text, text)

    # Using positional arguments (in proper order) -> no change
    text = "h(a, x, y)\n"
    _, new_text = self._upgrade(RemoveMultipleKeywordArguments(), text)
    self.assertEqual(new_text, text)

    # Use only the old names, in order
    text = "h(a, kw1_alias=x, kw2_alias=y)\n"
    acceptable_outputs = [
        "h(a, x, y)\n",
        "h(a, kw1=x, kw2=y)\n",
        "h(a=a, kw1=x, kw2=y)\n",
        "h(a, kw2=y, kw1=x)\n",
        "h(a=a, kw2=y, kw1=x)\n",
    ]
    _, new_text = self._upgrade(RemoveMultipleKeywordArguments(), text)
    self.assertIn(new_text, acceptable_outputs)

    # Use only the old names, in reverse order, should give one of same outputs
    text = "h(a, kw2_alias=y, kw1_alias=x)\n"
    _, new_text = self._upgrade(RemoveMultipleKeywordArguments(), text)
    self.assertIn(new_text, acceptable_outputs)

    # Mix old and new names
    text = "h(a, kw1=x, kw2_alias=y)\n"
    _, new_text = self._upgrade(RemoveMultipleKeywordArguments(), text)
    self.assertIn(new_text, acceptable_outputs)

  def testUnrestrictedFunctionWarnings(self):
    # [FUNCTION DOC]
    # Scenario: a locally defined specification warns on any attribute call named foo (wildcard
    # function name). Asserts that the warning text "not good" appears in the report for calls such as
    # on an object, on a call result, or in a chain, and does not appear for a bare name, a plain call
    # of foo, a nested attribute call of another name, a differently named method, or attribute access
    # without a call.
    class FooWarningSpec(ast_edits.NoUpdateSpec):
      # [CLASS DOC]
      # Locally defined test specification (inside the unrestricted-warnings test) that registers a
      # warning for any attribute call named foo, regardless of the receiver. Built on the no-update
      # specification.
      """Usages of function attribute foo() prints out a warning."""

      def __init__(self):
        # [FUNCTION DOC]
        # Constructor. Initializes the base no-update specification and sets the function warnings mapping
        # with the wildcard entry for a method named foo, with warning severity and the text "not good".
        # Takes no arguments, returns nothing.
        ast_edits.NoUpdateSpec.__init__(self)
        self.function_warnings = {"*.foo": (ast_edits.WARNING, "not good")}

    texts = ["object.foo()", "get_object().foo()",
             "get_object().foo()", "object.foo().bar()"]
    for text in texts:
      (_, report, _), _ = self._upgrade(FooWarningSpec(), text)
      self.assertIn("not good", report)

    # Note that foo() won't result in a warning, because in this case foo is
    # not an attribute, but a name.
    false_alarms = ["foo", "foo()", "foo.bar()", "obj.run_foo()", "obj.foo"]
    for text in false_alarms:
      (_, report, _), _ = self._upgrade(FooWarningSpec(), text)
      self.assertNotIn("not good", report)

  def testFullNameNode(self):
    # [FUNCTION DOC]
    # Scenario: builds a syntax tree node for the dotted name a.b.c using the upgrader's helper.
    # Asserts the textual dump of the tree equals nested attribute nodes over a name node for a with
    # load context.
    t = ast_edits.full_name_node("a.b.c")
    self.assertEqual(
        ast.dump(t),
        "Attribute(value=Attribute(value=Name(id='a', ctx=Load()), attr='b', "
        "ctx=Load()), attr='c', ctx=Load())")

  def testImport(self):
    # [FUNCTION DOC]
    # Scenario: import rename specification (foo to bar). Asserts plain imports are rewritten: with an
    # alias, without an alias (an alias of foo is added), of a submodule, of a submodule with alias,
    # and in a list with another import.
    # foo should be renamed to bar.
    text = "import foo as f"
    expected_text = "import bar as f"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)

    text = "import foo"
    expected_text = "import bar as foo"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)

    text = "import foo.test"
    expected_text = "import bar.test"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)

    text = "import foo.test as t"
    expected_text = "import bar.test as t"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)

    text = "import foo as f, a as b"
    expected_text = "import bar as f, a as b"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)

  def testFromImport(self):
    # [FUNCTION DOC]
    # Scenario: import rename specification. Asserts from-imports of foo are rewritten to bar, for a
    # module, a submodule, a star import and a list of names.
    # foo should be renamed to bar.
    text = "from foo import a"
    expected_text = "from bar import a"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)

    text = "from foo.a import b"
    expected_text = "from bar.a import b"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)

    text = "from foo import *"
    expected_text = "from bar import *"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)

    text = "from foo import a, b"
    expected_text = "from bar import a, b"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)

  def testImport_NoChangeNeeded(self):
    # [FUNCTION DOC]
    # Scenario: an import of the already-renamed module bar with an alias. Asserts the text is
    # unchanged.
    text = "import bar as b"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(text, new_text)

  def testFromImport_NoChangeNeeded(self):
    # [FUNCTION DOC]
    # Scenario: a from-import of the already-renamed module bar. Asserts the text is unchanged.
    text = "from bar import a as b"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(text, new_text)

  def testExcludedImport(self):
    # [FUNCTION DOC]
    # Scenario: imports touching the excluded prefix foo.baz (plain, aliased, from-import of baz from
    # foo, and from-import out of foo.baz). Asserts the text is left unchanged in every case.
    # foo.baz module is excluded from changes.
    text = "import foo.baz"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(text, new_text)

    text = "import foo.baz as a"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(text, new_text)

    text = "from foo import baz as a"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(text, new_text)

    text = "from foo.baz import a"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(text, new_text)

  def testMultipleImports(self):
    # [FUNCTION DOC]
    # Scenario: one import statement mixing renamed and excluded modules, and a from-import mixing the
    # excluded name baz with other names. Asserts only non-excluded modules are renamed, and the
    # from-import is split into two statements: one keeping foo for baz, one from bar for the rest.
    text = "import foo.bar as a, foo.baz as b, foo.baz.c, foo.d"
    expected_text = "import bar.bar as a, foo.baz as b, foo.baz.c, bar.d"
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)

    text = "from foo import baz, a, c"
    expected_text = """from foo import baz
from bar import a, c"""
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)

  def testImportInsideFunction(self):
    # [FUNCTION DOC]
    # Scenario: a from-import of an excluded and a renamed name placed inside a function body.
    # Asserts the statement is split into two statements in place, leaving neighbouring imports and
    # the function structure intact.
    text = """
def t():
  from c import d
  from foo import baz, a
  from e import y
"""
    expected_text = """
def t():
  from c import d
  from foo import baz
  from bar import a
  from e import y
"""
    _, new_text = self._upgrade(RenameImports(), text)
    self.assertEqual(expected_text, new_text)

  def testUpgradeInplaceWithSymlink(self):
    # [FUNCTION DOC]
    # Scenario: a directory in the temporary directory contains a file with an import of foo and a
    # symbolic link to it in the same directory; the tree is upgraded in place. Asserts the link is
    # still a link pointing to the original file and the target file now imports bar. Skipped on
    # Windows because symbolic links do not work uniformly. Side effects: creates files and a link
    # in the test temporary directory.
    if os.name == "nt":
      self.skipTest("os.symlink doesn't work uniformly on Windows.")

    upgrade_dir = os.path.join(self.get_temp_dir(), "foo")
    os.mkdir(upgrade_dir)
    file_a = os.path.join(upgrade_dir, "a.py")
    file_b = os.path.join(upgrade_dir, "b.py")

    with open(file_a, "a") as f:
      f.write("import foo as f")
    os.symlink(file_a, file_b)

    upgrader = ast_edits.ASTCodeUpgrader(RenameImports())
    upgrader.process_tree_inplace(upgrade_dir)

    self.assertTrue(os.path.islink(file_b))
    self.assertEqual(file_a, os.readlink(file_b))
    with open(file_a, "r") as f:
      self.assertEqual("import bar as f", f.read())

  def testUpgradeInPlaceWithSymlinkInDifferentDir(self):
    # [FUNCTION DOC]
    # Scenario: the directory being upgraded contains a link to a file located in a different
    # directory; upgraded in place. Asserts the link remains and the target file is NOT upgraded
    # (still imports foo). Skipped on Windows. Side effects: creates directories, a file and a link
    # in the temporary directory.
    if os.name == "nt":
      self.skipTest("os.symlink doesn't work uniformly on Windows.")

    upgrade_dir = os.path.join(self.get_temp_dir(), "foo")
    other_dir = os.path.join(self.get_temp_dir(), "bar")
    os.mkdir(upgrade_dir)
    os.mkdir(other_dir)
    file_c = os.path.join(other_dir, "c.py")
    file_d = os.path.join(upgrade_dir, "d.py")

    with open(file_c, "a") as f:
      f.write("import foo as f")
    os.symlink(file_c, file_d)

    upgrader = ast_edits.ASTCodeUpgrader(RenameImports())
    upgrader.process_tree_inplace(upgrade_dir)

    self.assertTrue(os.path.islink(file_d))
    self.assertEqual(file_c, os.readlink(file_d))
    # File pointed to by symlink is in a different directory.
    # Therefore, it should not be upgraded.
    with open(file_c, "r") as f:
      self.assertEqual("import foo as f", f.read())

  def testUpgradeCopyWithSymlink(self):
    # [FUNCTION DOC]
    # Scenario: a directory with a file and a link to it in the same directory is upgraded into a
    # separate output directory with copying of other files enabled. Asserts the output has a link
    # pointing to the output copy of the file, and the copy contains the renamed import. Skipped on
    # Windows. Side effects: creates files, links and directories in the temporary directory.
    if os.name == "nt":
      self.skipTest("os.symlink doesn't work uniformly on Windows.")

    upgrade_dir = os.path.join(self.get_temp_dir(), "foo")
    output_dir = os.path.join(self.get_temp_dir(), "bar")
    os.mkdir(upgrade_dir)
    file_a = os.path.join(upgrade_dir, "a.py")
    file_b = os.path.join(upgrade_dir, "b.py")

    with open(file_a, "a") as f:
      f.write("import foo as f")
    os.symlink(file_a, file_b)

    upgrader = ast_edits.ASTCodeUpgrader(RenameImports())
    upgrader.process_tree(upgrade_dir, output_dir, copy_other_files=True)

    new_file_a = os.path.join(output_dir, "a.py")
    new_file_b = os.path.join(output_dir, "b.py")
    self.assertTrue(os.path.islink(new_file_b))
    self.assertEqual(new_file_a, os.readlink(new_file_b))
    with open(new_file_a, "r") as f:
      self.assertEqual("import bar as f", f.read())

  def testUpgradeCopyWithSymlinkInDifferentDir(self):
    # [FUNCTION DOC]
    # Scenario: the input directory has a link to a file in a different directory; tree is upgraded
    # into an output directory with copying of other files. Asserts the output contains a link, that
    # it still points to the original external file, and that this file is unchanged (not upgraded).
    # Skipped on Windows. Side effects: creates files, links and directories in the temporary directory.
    if os.name == "nt":
      self.skipTest("os.symlink doesn't work uniformly on Windows.")

    upgrade_dir = os.path.join(self.get_temp_dir(), "foo")
    other_dir = os.path.join(self.get_temp_dir(), "bar")
    output_dir = os.path.join(self.get_temp_dir(), "baz")
    os.mkdir(upgrade_dir)
    os.mkdir(other_dir)
    file_a = os.path.join(other_dir, "a.py")
    file_b = os.path.join(upgrade_dir, "b.py")

    with open(file_a, "a") as f:
      f.write("import foo as f")
    os.symlink(file_a, file_b)

    upgrader = ast_edits.ASTCodeUpgrader(RenameImports())
    upgrader.process_tree(upgrade_dir, output_dir, copy_other_files=True)

    new_file_b = os.path.join(output_dir, "b.py")
    self.assertTrue(os.path.islink(new_file_b))
    self.assertEqual(file_a, os.readlink(new_file_b))
    with open(file_a, "r") as f:
      self.assertEqual("import foo as f", f.read())


if __name__ == "__main__":
  test_lib.main()
