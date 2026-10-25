# [MODULE DOC]
# Unit tests for the decorator_utils module: qualified-name lookup, adding deprecation-style
# notices to docstrings, validating callables, and the cached class property decorator.
# Contains four test classes built on the TensorFlow test case base class plus one trivial
# module-level function used as a sample callable. Running the file as a script starts the
# TensorFlow test runner. No import-time side effects.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""decorator_utils tests."""

# pylint: disable=unused-import
import functools

from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import decorator_utils


def _test_function(unused_arg=0):
  # [FUNCTION DOC]
  # HELPER: sample callable shared by the tests in this file.
  # Accepts one optional, ignored argument (default zero), does nothing and returns nothing.
  pass


class GetQualifiedNameTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the qualified-name helper of decorator_utils for a method and for a
  # plain module-level function. No shared fixtures.

  def test_method(self):
    # [FUNCTION DOC]
    # Scenario: look up the qualified name of a method of this test class. Expects the class
    # name and method name joined by a dot.
    self.assertEqual(
        "GetQualifiedNameTest.test_method",
        decorator_utils.get_qualified_name(GetQualifiedNameTest.test_method))

  def test_function(self):
    # [FUNCTION DOC]
    # Scenario: look up the qualified name of the module-level sample function. Expects just
    # the function name, since it has no enclosing class.
    self.assertEqual("_test_function",
                     decorator_utils.get_qualified_name(_test_function))


class AddNoticeToDocstringTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the helper that inserts a warning notice and instructions into a
  # docstring. All tests go through the shared check method, which fixes the instructions,
  # the placeholder for missing docs, the brief-line suffix and a one-line notice.

  def _check(self, doc, expected):
    # [FUNCTION DOC]
    # HELPER: shared assertion for the tests in this class.
    # Inputs: doc, the original docstring text (may be nothing or empty); expected, the
    # docstring that should result. Calls the notice-adding helper with fixed instructions
    # "Instructions", placeholder "Nothing here", suffix "(suffix)" and notice "Go away", then
    # asserts that the result equals expected. Returns nothing; fails the test on mismatch.
    self.assertEqual(
        decorator_utils.add_notice_to_docstring(
            doc=doc,
            instructions="Instructions",
            no_doc_str="Nothing here",
            suffix_str="(suffix)",
            notice=["Go away"]),
        expected)

  def test_regular(self):
    # [FUNCTION DOC]
    # Scenario: a docstring with a brief line, a body and an Args section. Tries five layouts
    # (no indentation, two-space indentation with empty or whitespace-only blank lines, and a
    # leading blank line). Expects identical output each time: brief line with suffix, the
    # warning and instructions, then the de-indented body and Args section.
    expected = (
        "Brief (suffix)\n\nWarning: Go away\nInstructions\n\nDocstring\n\n"
        "Args:\n  arg1: desc")
    # No indent for main docstring
    self._check("Brief\n\nDocstring\n\nArgs:\n  arg1: desc", expected)
    # 2 space indent for main docstring, blank lines not indented
    self._check("Brief\n\n  Docstring\n\n  Args:\n    arg1: desc", expected)
    # 2 space indent for main docstring, blank lines indented as well.
    self._check("Brief\n  \n  Docstring\n  \n  Args:\n    arg1: desc", expected)
    # No indent for main docstring, first line blank.
    self._check("\n  Brief\n  \n  Docstring\n  \n  Args:\n    arg1: desc",
                expected)
    # 2 space indent, first line blank.
    self._check("\n  Brief\n  \n  Docstring\n  \n  Args:\n    arg1: desc",
                expected)

  def test_brief_only(self):
    # [FUNCTION DOC]
    # Scenario: docstrings that contain only a brief line, with various trailing newlines,
    # trailing spaces and leading blank lines. Expects the brief line with suffix followed by
    # the warning and instructions, and no body section.
    expected = "Brief (suffix)\n\nWarning: Go away\nInstructions"
    self._check("Brief", expected)
    self._check("Brief\n", expected)
    self._check("Brief\n  ", expected)
    self._check("\nBrief\n  ", expected)
    self._check("\n  Brief\n  ", expected)

  def test_no_docstring(self):
    # [FUNCTION DOC]
    # Scenario: the docstring is missing (nothing) or empty. Expects the placeholder text
    # followed by the warning and instructions.
    expected = "Nothing here\n\nWarning: Go away\nInstructions"
    self._check(None, expected)
    self._check("", expected)

  def test_no_empty_line(self):
    # [FUNCTION DOC]
    # Scenario: a brief line immediately followed by a body line with no blank line between,
    # with and without indentation and with an optional leading blank line. Expects a blank
    # line to be inserted so the output has the brief line with suffix, the notice and
    # instructions, then the de-indented body line.
    expected = "Brief (suffix)\n\nWarning: Go away\nInstructions\n\nDocstring"
    # No second line indent
    self._check("Brief\nDocstring", expected)
    # 2 space second line indent
    self._check("Brief\n  Docstring", expected)
    # No second line indent, first line blank
    self._check("\nBrief\nDocstring", expected)
    # 2 space second line indent, first line blank
    self._check("\n  Brief\n  Docstring", expected)


class ValidateCallableTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the callable-validation helper of decorator_utils: it should accept
  # functions, bound methods, callable objects and partial objects, and reject non-callables.

  def test_function(self):
    # [FUNCTION DOC]
    # Scenario: validate a plain function. Expects no error.
    decorator_utils.validate_callable(_test_function, "test")

  def test_method(self):
    # [FUNCTION DOC]
    # Scenario: validate a bound method of this test instance. Expects no error.
    decorator_utils.validate_callable(self.test_method, "test")

  def test_callable(self):
    # [FUNCTION DOC]
    # Scenario: validate an instance of a local class that defines a call method. Expects no
    # error because instances with a call method count as callable.

    class TestClass(object):
      # [CLASS DOC]
      # HELPER: local class whose instances are callable, used only as validation input.

      def __call__(self):
        # [FUNCTION DOC]
        # HELPER: makes instances callable. Takes no arguments, does nothing, returns nothing.
        pass

    decorator_utils.validate_callable(TestClass(), "test")

  def test_partial(self):
    # [FUNCTION DOC]
    # Scenario: validate a partial object wrapping the sample function with a bound keyword
    # argument. Expects no error.
    partial = functools.partial(_test_function, unused_arg=7)
    decorator_utils.validate_callable(partial, "test")

  def test_fail_non_callable(self):
    # [FUNCTION DOC]
    # Scenario: validate the integer zero, which is not callable. Expects a value error.
    x = 0
    self.assertRaises(ValueError, decorator_utils.validate_callable, x, "test")


class CachedClassPropertyTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the cached class property decorator: lazy computation, per-class
  # caching, subclass behavior, instance access and read-only enforcement.

  def testCachedClassProperty(self):
    # [FUNCTION DOC]
    # Scenario: define a class whose cached class property records each computing call in a
    # shared list and returns the class name, plus an empty subclass. Expects: nothing is
    # computed before first access; the first access computes once and later accesses reuse
    # the cached value; each subclass computes its own value once; access through instances
    # works and does not recompute; assigning to or deleting the property through an instance
    # raises an attribute error.
    log = []  # log all calls to `MyClass.value`.

    class MyClass(object):
      # [CLASS DOC]
      # HELPER: local class holding the cached class property under test.

      @decorator_utils.cached_classproperty
      def value(cls):  # pylint: disable=no-self-argument
        # [FUNCTION DOC]
        # HELPER: computation wrapped by the cached class property. Appends the class it was
        # invoked on to the enclosing log (side effect) and returns that class's name.
        log.append(cls)
        return cls.__name__

    class MySubclass(MyClass):
      # [CLASS DOC]
      # HELPER: empty subclass used to check that each subclass computes its own cached value.
      pass

    # Property is computed first time it is accessed.
    self.assertLen(log, 0)
    self.assertEqual(MyClass.value, "MyClass")
    self.assertEqual(log, [MyClass])

    # Cached values are used on subsequent accesses.
    self.assertEqual(MyClass.value, "MyClass")
    self.assertEqual(MyClass.value, "MyClass")
    self.assertEqual(log, [MyClass])

    # The wrapped method is called for each subclass.
    self.assertEqual(MySubclass.value, "MySubclass")
    self.assertEqual(log, [MyClass, MySubclass])
    self.assertEqual(MySubclass.value, "MySubclass")
    self.assertEqual(MySubclass.value, "MySubclass")
    self.assertEqual(log, [MyClass, MySubclass])

    # The property can also be accessed via an instance.
    self.assertEqual(MyClass().value, "MyClass")
    self.assertEqual(MySubclass().value, "MySubclass")
    self.assertEqual(log, [MyClass, MySubclass])

    # Attempts to modify the property via an instance will fail.
    with self.assertRaises(AttributeError):
      MyClass().value = 12
    with self.assertRaises(AttributeError):
      del MyClass().value


if __name__ == "__main__":
  test.main()
