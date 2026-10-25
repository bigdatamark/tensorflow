# [MODULE DOC]
# Unit tests for the deprecation module. Test classes cover: deprecated aliases for classes and
# functions; the deprecated decorator on functions, methods, properties, classes (plain, with a
# custom new method, enum and named tuple), warn-once and silencing behavior and strict mode;
# the deprecated-arguments decorator (positional, keyword-only, variadic, ok values); the
# deprecated-argument-values decorator; the argument lookup and docstring rewrite helpers; the
# deprecated-endpoints decorator; and the moved-module hook (using the two small fixture modules
# deprecated_module and deprecated_module_new). Most tests patch the TensorFlow logging warning
# function with a mock and inspect its call count and arguments. Running the file as a script
# starts the TensorFlow test runner.
# KNOWN ISSUE: the strict-mode test enables strict mode globally and never turns it off, so
# later tests in the same process may be affected.
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
"""Deprecation tests."""

# pylint: disable=unused-import

import collections
import enum

import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import strict_mode
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import deprecation
from tensorflow.python.util import tf_inspect


class DeprecatedAliasTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the deprecated-alias helper applied to a function and to a class. The warning
  # logger is mocked in each test to count and inspect warnings.

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_function_alias(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: alias the error logging function under a deprecated name. Expects that calling
    # the original logs no deprecation warning, the first call through the alias logs exactly
    # one warning whose location text mentions this test file, and a second alias call logs no
    # additional warning (warn-once default).
    deprecated_func = deprecation.deprecated_alias("deprecated.func",
                                                   "real.func",
                                                   logging.error)

    logging.error("fake error logged")
    self.assertEqual(0, mock_warning.call_count)
    deprecated_func("FAKE ERROR!")
    self.assertEqual(1, mock_warning.call_count)
    # Make sure the error points to the right file.
    self.assertRegex(mock_warning.call_args[0][1], r"deprecation_test\.py:")
    deprecated_func("ANOTHER FAKE ERROR!")
    self.assertEqual(1, mock_warning.call_count)

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_class_alias(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: alias a small class that records constructor arguments in a class-level list.
    # Prints the alias name, module and docstring (no assertions on them). Expects: creating the
    # original logs nothing; the first alias instantiation logs one warning pointing to this file;
    # a second one logs none; all three constructor arguments reached the original constructor in
    # order; and the alias constructor has the same argument specification as the original.
    class MyClass(object):
      # [CLASS DOC]
      # HELPER: sample class being aliased. Records each constructor argument in a class-level
      # list so the test can verify the alias forwards arguments.
      """My docstring."""

      init_args = []

      def __init__(self, arg):
        # [FUNCTION DOC]
        # HELPER: constructor of the sample class. Input: arg, any value; appends it to the
        # class-level list. Returns nothing.
        MyClass.init_args.append(arg)

    deprecated_cls = deprecation.deprecated_alias("deprecated.cls",
                                                  "real.cls",
                                                  MyClass)

    print(deprecated_cls.__name__)
    print(deprecated_cls.__module__)
    print(deprecated_cls.__doc__)

    MyClass("test")
    self.assertEqual(0, mock_warning.call_count)
    deprecated_cls("deprecated")
    self.assertEqual(1, mock_warning.call_count)
    # Make sure the error points to the right file.
    self.assertRegex(mock_warning.call_args[0][1], r"deprecation_test\.py:")
    deprecated_cls("deprecated again")
    self.assertEqual(1, mock_warning.call_count)

    self.assertEqual(["test", "deprecated", "deprecated again"],
                     MyClass.init_args)

    # Check __init__ signature matches for doc generation.
    self.assertEqual(
        tf_inspect.getfullargspec(MyClass.__init__),
        tf_inspect.getfullargspec(deprecated_cls.__init__))


class DeprecationTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the deprecated decorator on functions, instance methods, properties and classes.
  # Checks warning counts (with a mocked warning logger), docstring rewriting, warn-once and
  # silencing, strict mode, and argument validation. Provides a shared subset-assertion helper.

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_deprecated_once(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: decorate a no-op function with warn-once enabled and call it twice. Expects
    # exactly one warning in total.
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated(date, instructions, warn_once=True)
    def _fn():
      # [FUNCTION DOC]
      # HELPER: no-op function under test. Takes no arguments, returns nothing.
      pass

    _fn()
    self.assertEqual(1, mock_warning.call_count)
    _fn()
    self.assertEqual(1, mock_warning.call_count)

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_deprecated_init_class(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: decorate a class that defines its own init method and instantiate it twice.
    # Expects one warning in total (warn-once) and that the class docstring now contains the
    # "IS DEPRECATED" notice.
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated(date, instructions, warn_once=True)
    class MyClass():
      # [CLASS DOC]
      # HELPER: sample class decorated as deprecated; its constructor takes one ignored value.
      """A test class."""

      def __init__(self, a):
        # [FUNCTION DOC]
        # HELPER: constructor accepting one ignored argument; does nothing.
        pass

    MyClass("")
    self.assertEqual(1, mock_warning.call_count)
    MyClass("")
    self.assertEqual(1, mock_warning.call_count)
    self.assertIn("IS DEPRECATED", MyClass.__doc__)

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_deprecated_new_class(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: decorate a string subclass that customizes the new method and create two
    # instances. Expects the new method to be the wrapped entry point: one warning in total and
    # the "IS DEPRECATED" notice in the class docstring.
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated(date, instructions, warn_once=True)
    class MyStr(str):
      # [CLASS DOC]
      # HELPER: sample string subclass with a custom new method, decorated as deprecated.

      def __new__(cls, value):
        # [FUNCTION DOC]
        # HELPER: creation hook. Input: cls, the class; value, text. Returns a new string
        # instance of cls built by the base string type.
        return str.__new__(cls, value)

    MyStr("abc")
    self.assertEqual(1, mock_warning.call_count)
    MyStr("abc")
    self.assertEqual(1, mock_warning.call_count)
    self.assertIn("IS DEPRECATED", MyStr.__doc__)

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_deprecated_enum(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: decorate an enumeration class with two members and look members up by value.
    # Expects the lookups to return the right members, one warning in total (the first lookup
    # only) and the "IS DEPRECATED" notice in the class docstring.
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated(date, instructions, warn_once=True)
    class MyEnum(enum.Enum):
      # [CLASS DOC]
      # HELPER: sample enumeration with members a (1) and b (2), decorated as deprecated.
      a = 1
      b = 2

    self.assertIs(MyEnum(1), MyEnum.a)
    self.assertEqual(1, mock_warning.call_count)
    self.assertIs(MyEnum(2), MyEnum.b)
    self.assertEqual(1, mock_warning.call_count)
    self.assertIn("IS DEPRECATED", MyEnum.__doc__)

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_deprecated_namedtuple(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: apply the decorator directly (not with decorator syntax) to a named tuple type
    # with two fields, then create two instances. Expects one warning in total and the
    # "IS DEPRECATED" notice in the type docstring.
    date = "2016-07-04"
    instructions = "This is how you update..."

    mytuple = deprecation.deprecated(
        date, instructions, warn_once=True)(
            collections.namedtuple("my_tuple", ["field1", "field2"]))

    mytuple(1, 2)
    self.assertEqual(1, mock_warning.call_count)
    mytuple(3, 4)
    self.assertEqual(1, mock_warning.call_count)
    self.assertIn("IS DEPRECATED", mytuple.__doc__)

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_silence(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: decorated no-op function with warn-once disabled. Calls it normally (one
    # warning), then inside the silence context (no new warning), then again normally (second
    # warning). Verifies the silence context suppresses and then restores warnings.
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated(date, instructions, warn_once=False)
    def _fn():
      # [FUNCTION DOC]
      # HELPER: no-op function under test. Takes no arguments, returns nothing.
      pass

    _fn()
    self.assertEqual(1, mock_warning.call_count)

    with deprecation.silence():
      _fn()
    self.assertEqual(1, mock_warning.call_count)

    _fn()
    self.assertEqual(2, mock_warning.call_count)

  def test_strict_mode_deprecation(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: enables strict mode and never disables it, leaking global state into later
    # tests in the same process.
    # Scenario: decorate a no-op function, enable strict mode, then call it. Expects a runtime
    # error because strict mode turns the deprecation warning into an error.
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated(date, instructions, warn_once=True)
    def _fn():
      # [FUNCTION DOC]
      # HELPER: no-op function under test. Takes no arguments, returns nothing.
      pass

    strict_mode.enable_strict_mode()
    with self.assertRaises(RuntimeError):
      _fn()

  def _assert_subset(self, expected_subset, actual_set):
    # [FUNCTION DOC]
    # HELPER: shared assertion for the tests in this class. Inputs: expected_subset, a
    # collection of values; actual_set, a set. Fails the test, with a message naming both, unless
    # actual_set contains every element of expected_subset. Returns nothing.
    self.assertTrue(
        actual_set.issuperset(expected_subset),
        msg="%s is not a superset of %s." % (actual_set, expected_subset))

  def test_deprecated_illegal_args(self):
    # [FUNCTION DOC]
    # Scenario: build the deprecated decorator with an empty date, a wrongly ordered date, then
    # with missing or empty instructions. Expects a value error mentioning the required date
    # format for the bad dates and one mentioning instructions for the bad instructions.
    instructions = "This is how you update..."
    with self.assertRaisesRegex(ValueError, "YYYY-MM-DD"):
      deprecation.deprecated("", instructions)
    with self.assertRaisesRegex(ValueError, "YYYY-MM-DD"):
      deprecation.deprecated("07-04-2016", instructions)
    date = "2016-07-04"
    with self.assertRaisesRegex(ValueError, "instructions"):
      deprecation.deprecated(date, None)
    with self.assertRaisesRegex(ValueError, "instructions"):
      deprecation.deprecated(date, "")

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_no_date(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: decorate a documented two-argument adder with no removal date. Expects the
    # rewritten docstring to say "(deprecated)" on the first line, include the notice that it
    # will be removed in a future version plus the instructions, and keep the Args and Returns
    # sections. Calling it returns the sum, logs one warning mentioning "deprecated and will be
    # removed", and the warning arguments include "in a future version" and the instructions.
    date = None
    instructions = "This is how you update..."

    @deprecation.deprecated(date, instructions)
    def _fn(arg0, arg1):
      # [FUNCTION DOC]
      # HELPER: function under test. Inputs: arg0 and arg1, numbers. Returns their sum.
      """fn doc.

      Args:
        arg0: Arg 0.
        arg1: Arg 1.

      Returns:
        Sum of args.
      """
      return arg0 + arg1

    self.assertEqual(
        "fn doc. (deprecated)"
        "\n"
        "\nDeprecated: THIS FUNCTION IS DEPRECATED. "
        "It will be removed in a future version."
        "\nInstructions for updating:\n%s"
        "\n"
        "\nArgs:"
        "\n  arg0: Arg 0."
        "\n  arg1: Arg 1."
        "\n"
        "\nReturns:"
        "\n  Sum of args." % instructions, _fn.__doc__)

    # Assert calling new fn issues log warning.
    self.assertEqual(3, _fn(1, 2))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegex(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["in a future version", instructions]),
                        set(args[1:]))

  @test.mock.patch.object(logging, "warning", autospec=True)
  @test_util.run_deprecated_v1
  def test_static_fn_with_doc(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: decorate a documented two-argument adder with a removal date. Expects the name
    # to be preserved, the docstring rewritten with "(deprecated)" on the first line, the
    # "removed after <date>" notice, the instructions and the original Args and Returns sections.
    # Calling it returns the sum and logs one warning whose arguments include "after <date>" and
    # the instructions.
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated(date, instructions)
    def _fn(arg0, arg1):
      # [FUNCTION DOC]
      # HELPER: function under test. Inputs: arg0 and arg1, numbers. Returns their sum.
      """fn doc.

      Args:
        arg0: Arg 0.
        arg1: Arg 1.

      Returns:
        Sum of args.
      """
      return arg0 + arg1

    # Assert function docs are properly updated.
    self.assertEqual("_fn", _fn.__name__)
    self.assertEqual(
        "fn doc. (deprecated)"
        "\n"
        "\nDeprecated: THIS FUNCTION IS DEPRECATED. It will be removed after %s."
        "\nInstructions for updating:\n%s"
        "\n"
        "\nArgs:"
        "\n  arg0: Arg 0."
        "\n  arg1: Arg 1."
        "\n"
        "\nReturns:"
        "\n  Sum of args." % (date, instructions), _fn.__doc__)

    # Assert calling new fn issues log warning.
    self.assertEqual(3, _fn(1, 2))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegex(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

  @test.mock.patch.object(logging, "warning", autospec=True)
  @test_util.run_deprecated_v1
  def test_static_fn_with_one_line_doc(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: decorate an adder whose docstring is a single line. Expects the rewritten
    # docstring to be the line with "(deprecated)", the removal notice and the instructions (no
    # other sections); calling it returns the sum and logs one warning with the date text and
    # instructions among its arguments.
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated(date, instructions)
    def _fn(arg0, arg1):
      # [FUNCTION DOC]
      # HELPER: function under test. Inputs: arg0 and arg1, numbers. Returns their sum.
      """fn doc."""
      return arg0 + arg1

    # Assert function docs are properly updated.
    self.assertEqual("_fn", _fn.__name__)
    self.assertEqual(
        "fn doc. (deprecated)"
        "\n"
        "\nDeprecated: THIS FUNCTION IS DEPRECATED. It will be removed after %s."
        "\nInstructions for updating:\n%s" % (date, instructions), _fn.__doc__)

    # Assert calling new fn issues log warning.
    self.assertEqual(3, _fn(1, 2))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegex(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

  @test.mock.patch.object(logging, "warning", autospec=True)
  @test_util.run_deprecated_v1
  def test_static_fn_no_doc(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: decorate an adder that has no docstring. Expects the docstring to become the
    # placeholder "DEPRECATED FUNCTION" followed by the removal notice and instructions; calling
    # it returns the sum and logs one warning with the date text and instructions.
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated(date, instructions)
    def _fn(arg0, arg1):
      # [FUNCTION DOC]
      # HELPER: function under test. Inputs: arg0 and arg1, numbers. Returns their sum.
      return arg0 + arg1

    # Assert function docs are properly updated.
    self.assertEqual("_fn", _fn.__name__)
    self.assertEqual(
        "DEPRECATED FUNCTION"
        "\n"
        "\nDeprecated: THIS FUNCTION IS DEPRECATED. It will be removed after %s."
        "\nInstructions for updating:"
        "\n%s" % (date, instructions), _fn.__doc__)

    # Assert calling new fn issues log warning.
    self.assertEqual(3, _fn(1, 2))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegex(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_instance_fn_with_doc(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: decorate a documented adder method of a local class. Expects the class attribute's
    # docstring to be rewritten with "(deprecated)", the removal notice, instructions and the
    # original Args and Returns sections; calling the method on an instance returns the sum and
    # logs one warning containing the date text and instructions.
    date = "2016-07-04"
    instructions = "This is how you update..."

    class _Object(object):
      # [CLASS DOC]
      # HELPER: local class holding the deprecated method under test.

      def __init(self):
        # [FUNCTION DOC]
        # HELPER: placeholder method (its name is mangled, so it is never used as a constructor).
        # Does nothing.
        pass

      @deprecation.deprecated(date, instructions)
      def _fn(self, arg0, arg1):
        # [FUNCTION DOC]
        # HELPER: deprecated method under test. Inputs: arg0 and arg1, numbers. Returns their sum.
        """fn doc.

        Args:
          arg0: Arg 0.
          arg1: Arg 1.

        Returns:
          Sum of args.
        """
        return arg0 + arg1

    # Assert function docs are properly updated.
    self.assertEqual(
        "fn doc. (deprecated)"
        "\n"
        "\nDeprecated: THIS FUNCTION IS DEPRECATED. It will be removed after %s."
        "\nInstructions for updating:\n%s"
        "\n"
        "\nArgs:"
        "\n  arg0: Arg 0."
        "\n  arg1: Arg 1."
        "\n"
        "\nReturns:"
        "\n  Sum of args." % (date, instructions),
        getattr(_Object, "_fn").__doc__)

    # Assert calling new fn issues log warning.
    self.assertEqual(3, _Object()._fn(1, 2))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegex(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_instance_fn_with_one_line_doc(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: decorate an adder method whose docstring is one line. Expects the rewritten
    # docstring to be that line with "(deprecated)" plus the removal notice and instructions;
    # calling it returns the sum and logs one warning with the date text and instructions.
    date = "2016-07-04"
    instructions = "This is how you update..."

    class _Object(object):
      # [CLASS DOC]
      # HELPER: local class holding the deprecated method under test.

      def __init(self):
        # [FUNCTION DOC]
        # HELPER: placeholder method (its name is mangled, so it is never used as a constructor).
        # Does nothing.
        pass

      @deprecation.deprecated(date, instructions)
      def _fn(self, arg0, arg1):
        # [FUNCTION DOC]
        # HELPER: deprecated method under test. Inputs: arg0 and arg1, numbers. Returns their sum.
        """fn doc."""
        return arg0 + arg1

    # Assert function docs are properly updated.
    self.assertEqual(
        "fn doc. (deprecated)"
        "\n"
        "\nDeprecated: THIS FUNCTION IS DEPRECATED. It will be removed after %s."
        "\nInstructions for updating:\n%s" % (date, instructions),
        getattr(_Object, "_fn").__doc__)

    # Assert calling new fn issues log warning.
    self.assertEqual(3, _Object()._fn(1, 2))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegex(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_instance_fn_no_doc(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: decorate an undocumented adder method. Expects the docstring to be the
    # placeholder "DEPRECATED FUNCTION" plus the removal notice and instructions; calling it
    # returns the sum and logs one warning with the date text and instructions.
    date = "2016-07-04"
    instructions = "This is how you update..."

    class _Object(object):
      # [CLASS DOC]
      # HELPER: local class holding the deprecated method under test.

      def __init(self):
        # [FUNCTION DOC]
        # HELPER: placeholder method (its name is mangled, so it is never used as a constructor).
        # Does nothing.
        pass

      @deprecation.deprecated(date, instructions)
      def _fn(self, arg0, arg1):
        # [FUNCTION DOC]
        # HELPER: deprecated method under test. Inputs: arg0 and arg1, numbers. Returns their sum.
        return arg0 + arg1

    # Assert function docs are properly updated.
    self.assertEqual(
        "DEPRECATED FUNCTION"
        "\n"
        "\nDeprecated: THIS FUNCTION IS DEPRECATED. It will be removed after %s."
        "\nInstructions for updating:"
        "\n%s" % (date, instructions),
        getattr(_Object, "_fn").__doc__)

    # Assert calling new fn issues log warning.
    self.assertEqual(3, _Object()._fn(1, 2))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegex(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

  def test_prop_wrong_order(self):
    # [FUNCTION DOC]
    # Scenario: define a class where the deprecated decorator is applied on top of a property
    # (wrong order). Expects a value error telling the developer to put the property decorator
    # before the deprecated decorator; the error is raised while the class body is evaluated.
    with self.assertRaisesRegex(
        ValueError,
        "make sure @property appears before @deprecated in your source code"):
      # pylint: disable=unused-variable

      class _Object(object):
        # [CLASS DOC]
        # HELPER: local class whose definition is expected to fail with a value error.

        def __init(self):
          # [FUNCTION DOC]
          # HELPER: placeholder method that does nothing.
          pass

        @deprecation.deprecated("2016-07-04", "Instructions.")
        @property
        def _prop(self):
          # [FUNCTION DOC]
          # HELPER: property getter (wrongly decorated). Returns a fixed text.
          return "prop_wrong_order"

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_prop_with_doc(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: a documented property whose getter is decorated as deprecated (property outermost,
    # the correct order). Expects the property docstring to be rewritten with "(deprecated)", the
    # removal notice, instructions and the Returns section; reading the property returns its
    # text and logs one warning with the date text and instructions.
    date = "2016-07-04"
    instructions = "This is how you update..."

    class _Object(object):
      # [CLASS DOC]
      # HELPER: local class holding the deprecated property under test.

      def __init(self):
        # [FUNCTION DOC]
        # HELPER: placeholder method that does nothing.
        pass

      @property
      @deprecation.deprecated(date, instructions)
      def _prop(self):
        # [FUNCTION DOC]
        # HELPER: deprecated property getter under test. Returns a fixed text.
        """prop doc.

        Returns:
          String.
        """
        return "prop_with_doc"

    # Assert function docs are properly updated.
    self.assertEqual(
        "prop doc. (deprecated)"
        "\n"
        "\nDeprecated: THIS FUNCTION IS DEPRECATED. It will be removed after %s."
        "\nInstructions for updating:"
        "\n%s"
        "\n"
        "\nReturns:"
        "\n  String." % (date, instructions),
        getattr(_Object, "_prop").__doc__)

    # Assert calling new fn issues log warning.
    self.assertEqual("prop_with_doc", _Object()._prop)
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegex(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_prop_no_doc(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: an undocumented property with a deprecated getter. Expects the docstring to be
    # the placeholder "DEPRECATED FUNCTION" plus the removal notice and instructions; reading the
    # property returns its text and logs one warning with the date text and instructions.
    date = "2016-07-04"
    instructions = "This is how you update..."

    class _Object(object):
      # [CLASS DOC]
      # HELPER: local class holding the deprecated property under test.

      def __init(self):
        # [FUNCTION DOC]
        # HELPER: placeholder method that does nothing.
        pass

      @property
      @deprecation.deprecated(date, instructions)
      def _prop(self):
        # [FUNCTION DOC]
        # HELPER: deprecated property getter under test. Returns a fixed text.
        return "prop_no_doc"

    # Assert function docs are properly updated.
    self.assertEqual(
        "DEPRECATED FUNCTION"
        "\n"
        "\nDeprecated: THIS FUNCTION IS DEPRECATED. It will be removed after %s."
        "\nInstructions for updating:"
        "\n%s" % (date, instructions),
        getattr(_Object, "_prop").__doc__)

    # Assert calling new fn issues log warning.
    self.assertEqual("prop_no_doc", _Object()._prop)
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegex(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))


class DeprecatedArgsTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the deprecated-arguments decorator: argument validation, docstring rewriting,
  # warnings for positional, keyword, variadic and keyword-only arguments, ok values, and
  # warn-once behavior. Warnings are observed through a mocked warning logger.

  def _assert_subset(self, expected_subset, actual_set):
    # [FUNCTION DOC]
    # HELPER: shared assertion for the tests in this class. Inputs: expected_subset, a collection
    # of values; actual_set, a set. Fails the test, with a message naming both, unless actual_set
    # contains every element of expected_subset. Returns nothing.
    self.assertTrue(
        actual_set.issuperset(expected_subset),
        msg="%s is not a superset of %s." % (actual_set, expected_subset))

  def test_deprecated_illegal_args(self):
    # [FUNCTION DOC]
    # Scenario: build the deprecated-arguments decorator with an empty date, a wrongly ordered
    # date, missing or empty instructions, and with no argument names. Expects a value error
    # mentioning the date format, the instructions, or the word argument respectively.
    instructions = "This is how you update..."
    date = "2016-07-04"
    with self.assertRaisesRegex(ValueError, "YYYY-MM-DD"):
      deprecation.deprecated_args("", instructions, "deprecated")
    with self.assertRaisesRegex(ValueError, "YYYY-MM-DD"):
      deprecation.deprecated_args("07-04-2016", instructions, "deprecated")
    with self.assertRaisesRegex(ValueError, "instructions"):
      deprecation.deprecated_args(date, None, "deprecated")
    with self.assertRaisesRegex(ValueError, "instructions"):
      deprecation.deprecated_args(date, "", "deprecated")
    with self.assertRaisesRegex(ValueError, "argument"):
      deprecation.deprecated_args(date, instructions)

  def test_deprecated_missing_args(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: a comment inside the test says calls log nothing, but the test only checks
    # that decoration fails; no call is made.
    # Scenario: ask to deprecate an argument name that is not in the function signature. Expects
    # a value error saying the argument is not present and listing the missing name.
    date = "2016-07-04"
    instructions = "This is how you update..."

    def _fn(arg0, arg1, deprecated=None):
      # [FUNCTION DOC]
      # HELPER: sample function with an optional third argument; never called by the test.
      # Inputs: arg0, arg1 and deprecated. Returns the sum of the first two.
      return arg0 + arg1 if deprecated else arg1 + arg0

    # Assert calls without the deprecated argument log nothing.
    with self.assertRaisesRegex(ValueError, "not present.*\\['missing'\\]"):
      deprecation.deprecated_args(date, instructions, "missing")(_fn)

  @test.mock.patch.object(logging, "warning", autospec=True)
  @test_util.run_deprecated_v1
  def test_static_fn_with_doc(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: mark the argument named deprecated as deprecated on a documented function.
    # Expects the name to be preserved and the docstring to say "(deprecated arguments)" with the
    # "some arguments are deprecated" notice, removal date, instructions and the original
    # sections. Calls without the argument log nothing and return the sum; a call passing it
    # positionally logs one warning with the date text and instructions.
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_args(date, instructions, "deprecated")
    def _fn(arg0, arg1, deprecated=True):
      # [FUNCTION DOC]
      # HELPER: function under test. Inputs: arg0 and arg1, numbers; deprecated, a flag
      # (default true) choosing the order of addition. Returns the sum.
      """fn doc.

      Args:
        arg0: Arg 0.
        arg1: Arg 1.
        deprecated: Deprecated!

      Returns:
        Sum of args.
      """
      return arg0 + arg1 if deprecated else arg1 + arg0

    # Assert function docs are properly updated.
    self.assertEqual("_fn", _fn.__name__)
    self.assertEqual(
        "fn doc. (deprecated arguments)"
        "\n"
        "\nDeprecated: SOME ARGUMENTS ARE DEPRECATED: `(deprecated)`. "
        "They will be removed after %s."
        "\nInstructions for updating:\n%s"
        "\n"
        "\nArgs:"
        "\n  arg0: Arg 0."
        "\n  arg1: Arg 1."
        "\n  deprecated: Deprecated!"
        "\n"
        "\nReturns:"
        "\n  Sum of args." % (date, instructions), _fn.__doc__)

    # Assert calls without the deprecated argument log nothing.
    self.assertEqual(3, _fn(1, 2))
    self.assertEqual(0, mock_warning.call_count)

    # Assert calls with the deprecated argument log a warning.
    self.assertEqual(3, _fn(1, 2, True))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegex(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

  @test.mock.patch.object(logging, "warning", autospec=True)
  @test_util.run_deprecated_v1
  def test_static_fn_with_one_line_doc(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: same as the documented case but the function has a one-line docstring. Expects
    # the docstring to be that line with "(deprecated arguments)" plus the notice and
    # instructions; calls without the deprecated argument log nothing and a call passing it logs
    # one warning.
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_args(date, instructions, "deprecated")
    def _fn(arg0, arg1, deprecated=True):
      # [FUNCTION DOC]
      # HELPER: function under test. Inputs: arg0 and arg1, numbers; deprecated, a flag
      # (default true). Returns the sum.
      """fn doc."""
      return arg0 + arg1 if deprecated else arg1 + arg0

    # Assert function docs are properly updated.
    self.assertEqual("_fn", _fn.__name__)
    self.assertEqual(
        "fn doc. (deprecated arguments)"
        "\n"
        "\nDeprecated: SOME ARGUMENTS ARE DEPRECATED: `(deprecated)`. "
        "They will be removed after %s."
        "\nInstructions for updating:\n%s" % (date, instructions), _fn.__doc__)

    # Assert calls without the deprecated argument log nothing.
    self.assertEqual(3, _fn(1, 2))
    self.assertEqual(0, mock_warning.call_count)

    # Assert calls with the deprecated argument log a warning.
    self.assertEqual(3, _fn(1, 2, True))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegex(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

  @test.mock.patch.object(logging, "warning", autospec=True)
  @test_util.run_deprecated_v1
  def test_static_fn_no_doc(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: deprecated argument on an undocumented function. Expects the docstring to be the
    # placeholder "DEPRECATED FUNCTION ARGUMENTS" with the notice and instructions; calls without
    # the argument log nothing and a call passing it logs one warning.
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_args(date, instructions, "deprecated")
    def _fn(arg0, arg1, deprecated=True):
      # [FUNCTION DOC]
      # HELPER: function under test. Inputs: arg0 and arg1, numbers; deprecated, a flag
      # (default true). Returns the sum.
      return arg0 + arg1 if deprecated else arg1 + arg0

    # Assert function docs are properly updated.
    self.assertEqual("_fn", _fn.__name__)
    self.assertEqual(
        "DEPRECATED FUNCTION ARGUMENTS"
        "\n"
        "\nDeprecated: SOME ARGUMENTS ARE DEPRECATED: `(deprecated)`. "
        "They will be removed after %s."
        "\nInstructions for updating:"
        "\n%s" % (date, instructions), _fn.__doc__)

    # Assert calls without the deprecated argument log nothing.
    self.assertEqual(3, _fn(1, 2))
    self.assertEqual(0, mock_warning.call_count)

    # Assert calls with the deprecated argument log a warning.
    self.assertEqual(3, _fn(1, 2, True))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegex(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

  @test.mock.patch.object(logging, "warning", autospec=True)
  @test_util.run_deprecated_v1
  def test_varargs(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: the deprecated name is the variadic positional parameter. Expects no warning when
    # only two arguments are passed and one warning when extra positional arguments are passed.
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_args(date, instructions, "deprecated")
    def _fn(arg0, arg1, *deprecated):
      # [FUNCTION DOC]
      # HELPER: function under test. Inputs: arg0 and arg1, numbers; extra positional values.
      # Returns the sum of the first two.
      return arg0 + arg1 if deprecated else arg1 + arg0

    # Assert calls without the deprecated argument log nothing.
    self.assertEqual(3, _fn(1, 2))
    self.assertEqual(0, mock_warning.call_count)

    # Assert calls with the deprecated argument log a warning.
    self.assertEqual(3, _fn(1, 2, True, False))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegex(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

  @test.mock.patch.object(logging, "warning", autospec=True)
  @test_util.run_deprecated_v1
  def test_kwargs(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: the deprecated name is the variadic keyword parameter. Expects no warning when no
    # extra keyword arguments are passed and one warning when some are passed.
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_args(date, instructions, "deprecated")
    def _fn(arg0, arg1, **deprecated):
      # [FUNCTION DOC]
      # HELPER: function under test. Inputs: arg0 and arg1, numbers; extra keyword values.
      # Returns the sum of the first two.
      return arg0 + arg1 if deprecated else arg1 + arg0

    # Assert calls without the deprecated argument log nothing.
    self.assertEqual(3, _fn(1, 2))
    self.assertEqual(0, mock_warning.call_count)

    # Assert calls with the deprecated argument log a warning.
    self.assertEqual(3, _fn(1, 2, a=True, b=False))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegex(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

  @test.mock.patch.object(logging, "warning", autospec=True)
  @test_util.run_deprecated_v1
  def test_positional_and_named(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: two deprecated arguments, one supplied by position and one by keyword. Expects
    # no warning when neither is supplied and two separate warnings (one naming d1, one naming
    # d2, each with the date text and instructions) when both are supplied.
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_args(date, instructions, "d1", "d2")
    def _fn(arg0, d1=None, arg1=2, d2=None):
      # [FUNCTION DOC]
      # HELPER: function under test with two optional deprecated arguments. Returns a number
      # derived from the arguments depending on which optional ones are truthy.
      return arg0 + arg1 if d1 else arg1 + arg0 if d2 else arg0 * arg1

    # Assert calls without the deprecated arguments log nothing.
    self.assertEqual(2, _fn(1, arg1=2))
    self.assertEqual(0, mock_warning.call_count)

    # Assert calls with the deprecated arguments log warnings.
    self.assertEqual(2, _fn(1, None, 2, d2=False))
    self.assertEqual(2, mock_warning.call_count)
    (args1, _) = mock_warning.call_args_list[0]
    self.assertRegex(args1[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions, "d1"]),
                        set(args1[1:]))
    (args2, _) = mock_warning.call_args_list[1]
    self.assertRegex(args2[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions, "d2"]),
                        set(args2[1:]))

  @test.mock.patch.object(logging, "warning", autospec=True)
  @test_util.run_deprecated_v1
  def test_positional_and_named_with_ok_vals(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: two deprecated arguments each with an "ok" value (null for d1, a sample text for
    # d2). Expects no warning with neither supplied; two warnings (d1, d2) when supplied with
    # non-ok values; and no warning when d1 is null and d2 equals its ok value.
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_args(date, instructions, ("d1", None),
                                 ("d2", "my_ok_val"))
    def _fn(arg0, d1=None, arg1=2, d2=None):
      # [FUNCTION DOC]
      # HELPER: function under test with two optional deprecated arguments. Returns a number
      # derived from the arguments depending on which optional ones are truthy.
      return arg0 + arg1 if d1 else arg1 + arg0 if d2 else arg0 * arg1

    # Assert calls without the deprecated arguments log nothing.
    self.assertEqual(2, _fn(1, arg1=2))
    self.assertEqual(0, mock_warning.call_count)

    # Assert calls with the deprecated arguments log warnings.
    self.assertEqual(2, _fn(1, False, 2, d2=False))
    self.assertEqual(2, mock_warning.call_count)
    (args1, _) = mock_warning.call_args_list[0]
    self.assertRegex(args1[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions, "d1"]),
                        set(args1[1:]))
    (args2, _) = mock_warning.call_args_list[1]
    self.assertRegex(args2[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions, "d2"]),
                        set(args2[1:]))

    # Assert calls with the deprecated arguments don't log warnings if
    # the value matches the 'ok_val'.
    mock_warning.reset_mock()
    self.assertEqual(3, _fn(1, None, 2, d2="my_ok_val"))
    self.assertEqual(0, mock_warning.call_count)

  @test.mock.patch.object(logging, "warning", autospec=True)
  @test_util.run_deprecated_v1
  def test_kwonlyargs(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: the deprecated argument is keyword-only. Expects no warning when it is omitted
    # and one warning when it is supplied.
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_args(date, instructions, "deprecated")
    def _fn(*, arg0, arg1, deprecated=None):
      # [FUNCTION DOC]
      # HELPER: function under test taking only keyword arguments. Returns the sum of arg0 and
      # arg1.
      return arg0 + arg1 if deprecated is not None else arg1 + arg0

    # Assert calls without the deprecated argument log nothing.
    self.assertEqual(3, _fn(arg0=1, arg1=2))
    self.assertEqual(0, mock_warning.call_count)

    # Assert calls with the deprecated argument log a warning.
    self.assertEqual(3, _fn(arg0=1, arg1=2, deprecated=2))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegex(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

  @test.mock.patch.object(logging, "warning", autospec=True)
  @test_util.run_deprecated_v1
  def test_kwonlyargs_and_args(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: a function with positional, a required keyword-only and two deprecated
    # keyword-only arguments, the deprecated ones given as a single tuple specification. Expects
    # no warning without them and a warning when either is supplied (the mock call count is one
    # after each of the last two checks because the mock is not reset between them).
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_args(date, instructions,
                                 ("deprecated_arg1", "deprecated_arg2"))
    def _fn(arg0, arg1, *, kw1,
            deprecated_arg1=None,
            deprecated_arg2=None):
      # [FUNCTION DOC]
      # HELPER: function under test. Inputs: arg0 and arg1, numbers; keyword-only kw1 (required)
      # and two optional deprecated numbers. Returns the sum of all supplied numbers.
      res = arg0 + arg1 + kw1
      if deprecated_arg1 is not None:
        res += deprecated_arg1
      if deprecated_arg2 is not None:
        res += deprecated_arg2
      return res

    # Assert calls without the deprecated argument log nothing.
    self.assertEqual(6, _fn(1, 2, kw1=3))
    self.assertEqual(0, mock_warning.call_count)

    # Assert calls with the deprecated_arg1 argument log a warning.
    self.assertEqual(8, _fn(1, 2, kw1=3, deprecated_arg1=2))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegex(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

    # Assert calls with the deprecated arguments log a warning.
    self.assertEqual(12, _fn(1, 2, kw1=3, deprecated_arg1=2, deprecated_arg2=4))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegex(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

  @test.mock.patch.object(logging, "warning", autospec=True)
  @test_util.run_deprecated_v1
  def test_deprecated_args_once(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: one deprecated argument with warn-once enabled. Expects no warning when omitted
    # and one warning in total across repeated calls that supply it.
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_args(date, instructions, "arg", warn_once=True)
    def _fn(arg=0):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # HELPER: no-op function with one optional ignored argument; returns nothing.
      pass

    _fn()
    self.assertEqual(0, mock_warning.call_count)
    _fn(arg=0)
    self.assertEqual(1, mock_warning.call_count)
    _fn(arg=1)
    self.assertEqual(1, mock_warning.call_count)

  @test.mock.patch.object(logging, "warning", autospec=True)
  @test_util.run_deprecated_v1
  def test_deprecated_multiple_args_once_each(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: two deprecated arguments with warn-once enabled. Expects one warning for the
    # first use of each argument (two in total) and no more on repeated uses.
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_args(date, instructions, "arg0", "arg1",
                                 warn_once=True)
    def _fn(arg0=0, arg1=0):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # HELPER: no-op function with two optional ignored arguments; returns nothing.
      pass

    _fn(arg0=0)
    self.assertEqual(1, mock_warning.call_count)
    _fn(arg0=0)
    self.assertEqual(1, mock_warning.call_count)
    _fn(arg1=0)
    self.assertEqual(2, mock_warning.call_count)
    _fn(arg0=0)
    self.assertEqual(2, mock_warning.call_count)
    _fn(arg1=0)
    self.assertEqual(2, mock_warning.call_count)


class DeprecatedArgValuesTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the deprecated-argument-values decorator: validation, docstring rewriting, warnings
  # only for the deprecated values (including defaults), warn-once behavior and null values.

  def _assert_subset(self, expected_subset, actual_set):
    # [FUNCTION DOC]
    # HELPER: shared assertion for the tests in this class. Inputs: expected_subset, a collection
    # of values; actual_set, a set. Fails the test, with a message naming both, unless actual_set
    # contains every element of expected_subset. Returns nothing.
    self.assertTrue(
        actual_set.issuperset(expected_subset),
        msg="%s is not a superset of %s." % (actual_set, expected_subset))

  def test_deprecated_illegal_args(self):
    # [FUNCTION DOC]
    # Scenario: build the deprecated-argument-values decorator with an empty date, a wrongly
    # ordered date, missing or empty instructions, and with no deprecated values. Expects a value
    # error mentioning the date format, the instructions, or the word argument respectively.
    instructions = "This is how you update..."
    with self.assertRaisesRegex(ValueError, "YYYY-MM-DD"):
      deprecation.deprecated_arg_values("", instructions, deprecated=True)
    with self.assertRaisesRegex(ValueError, "YYYY-MM-DD"):
      deprecation.deprecated_arg_values(
          "07-04-2016", instructions, deprecated=True)
    date = "2016-07-04"
    with self.assertRaisesRegex(ValueError, "instructions"):
      deprecation.deprecated_arg_values(date, None, deprecated=True)
    with self.assertRaisesRegex(ValueError, "instructions"):
      deprecation.deprecated_arg_values(date, "", deprecated=True)
    with self.assertRaisesRegex(ValueError, "argument"):
      deprecation.deprecated_arg_values(date, instructions)

  @test.mock.patch.object(logging, "warning", autospec=True)
  @test_util.run_deprecated_v1
  def test_static_fn_with_doc(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: mark the value true of the argument named deprecated as deprecated (warn-once
    # off) on a documented function. Expects the docstring to say "(deprecated argument values)"
    # with the "some argument values are deprecated" notice naming the argument and value, the
    # removal date, instructions and original sections. Passing the other value logs nothing;
    # passing the deprecated value, or relying on the default of true, logs a warning each time.
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_arg_values(date, instructions, warn_once=False,
                                       deprecated=True)
    def _fn(arg0, arg1, deprecated=True):
      # [FUNCTION DOC]
      # HELPER: function under test. Inputs: arg0 and arg1, numbers; deprecated, a flag
      # (default true). Returns the sum.
      """fn doc.

      Args:
        arg0: Arg 0.
        arg1: Arg 1.
        deprecated: Deprecated!

      Returns:
        Sum of args.
      """
      return arg0 + arg1 if deprecated else arg1 + arg0

    # Assert function docs are properly updated.
    self.assertEqual("_fn", _fn.__name__)
    self.assertEqual(
        "fn doc. (deprecated argument values)"
        "\n"
        "\nDeprecated: SOME ARGUMENT VALUES ARE DEPRECATED: `(deprecated=True)`. "
        "They will be removed after %s."
        "\nInstructions for updating:\n%s"
        "\n"
        "\nArgs:"
        "\n  arg0: Arg 0."
        "\n  arg1: Arg 1."
        "\n  deprecated: Deprecated!"
        "\n"
        "\nReturns:"
        "\n  Sum of args." % (date, instructions), _fn.__doc__)

    # Assert calling new fn with non-deprecated value logs nothing.
    self.assertEqual(3, _fn(1, 2, deprecated=False))
    self.assertEqual(0, mock_warning.call_count)

    # Assert calling new fn with deprecated value issues log warning.
    self.assertEqual(3, _fn(1, 2, deprecated=True))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegex(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

    # Assert calling new fn with default deprecated value issues log warning.
    self.assertEqual(3, _fn(1, 2))
    self.assertEqual(2, mock_warning.call_count)

  @test.mock.patch.object(logging, "warning", autospec=True)
  @test_util.run_deprecated_v1
  def test_static_fn_with_one_line_doc(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: same as the documented case but with a one-line docstring. Expects the line with
    # "(deprecated argument values)" plus the notice and instructions, no warning for the
    # non-deprecated value, and a warning for the deprecated value and for the default.
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_arg_values(date, instructions, warn_once=False,
                                       deprecated=True)
    def _fn(arg0, arg1, deprecated=True):
      # [FUNCTION DOC]
      # HELPER: function under test. Inputs: arg0 and arg1, numbers; deprecated, a flag
      # (default true). Returns the sum.
      """fn doc."""
      return arg0 + arg1 if deprecated else arg1 + arg0

    # Assert function docs are properly updated.
    self.assertEqual("_fn", _fn.__name__)
    self.assertEqual(
        "fn doc. (deprecated argument values)"
        "\n"
        "\nDeprecated: SOME ARGUMENT VALUES ARE DEPRECATED: `(deprecated=True)`. "
        "They will be removed after %s."
        "\nInstructions for updating:\n%s" % (date, instructions), _fn.__doc__)

    # Assert calling new fn with non-deprecated value logs nothing.
    self.assertEqual(3, _fn(1, 2, deprecated=False))
    self.assertEqual(0, mock_warning.call_count)

    # Assert calling new fn with deprecated value issues log warning.
    self.assertEqual(3, _fn(1, 2, deprecated=True))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegex(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

    # Assert calling new fn with default deprecated value issues log warning.
    self.assertEqual(3, _fn(1, 2))
    self.assertEqual(2, mock_warning.call_count)

  @test.mock.patch.object(logging, "warning", autospec=True)
  @test_util.run_deprecated_v1
  def test_static_fn_no_doc(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: deprecated argument value on an undocumented function. Expects the docstring to
    # be the placeholder "DEPRECATED FUNCTION ARGUMENT VALUES" with the notice and instructions,
    # no warning for the other value, and warnings for the deprecated value and the default.
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_arg_values(date, instructions, warn_once=False,
                                       deprecated=True)
    def _fn(arg0, arg1, deprecated=True):
      # [FUNCTION DOC]
      # HELPER: function under test. Inputs: arg0 and arg1, numbers; deprecated, a flag
      # (default true). Returns the sum.
      return arg0 + arg1 if deprecated else arg1 + arg0

    # Assert function docs are properly updated.
    self.assertEqual("_fn", _fn.__name__)
    self.assertEqual(
        "DEPRECATED FUNCTION ARGUMENT VALUES"
        "\n"
        "\nDeprecated: SOME ARGUMENT VALUES ARE DEPRECATED: `(deprecated=True)`. "
        "They will be removed after %s."
        "\nInstructions for updating:"
        "\n%s" % (date, instructions), _fn.__doc__)

    # Assert calling new fn with non-deprecated value logs nothing.
    self.assertEqual(3, _fn(1, 2, deprecated=False))
    self.assertEqual(0, mock_warning.call_count)

    # Assert calling new fn issues log warning.
    self.assertEqual(3, _fn(1, 2, deprecated=True))
    self.assertEqual(1, mock_warning.call_count)
    (args, _) = mock_warning.call_args
    self.assertRegex(args[0], r"deprecated and will be removed")
    self._assert_subset(set(["after " + date, instructions]), set(args[1:]))

    # Assert calling new fn with default deprecated value issues log warning.
    self.assertEqual(3, _fn(1, 2))
    self.assertEqual(2, mock_warning.call_count)

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_deprecated_arg_values_once(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: one deprecated value with warn-once enabled. Expects no warning for the other
    # value and a single warning in total across repeated uses of the deprecated value.
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_arg_values(date, instructions, warn_once=True,
                                       deprecated=True)
    def _fn(deprecated):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # HELPER: no-op function with one ignored argument; returns nothing.
      pass

    _fn(deprecated=False)
    self.assertEqual(0, mock_warning.call_count)
    _fn(deprecated=True)
    self.assertEqual(1, mock_warning.call_count)
    _fn(deprecated=True)
    self.assertEqual(1, mock_warning.call_count)

  @test.mock.patch.object(logging, "warning", autospec=True)
  def test_deprecated_multiple_arg_values_once_each(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: two arguments each with a deprecated text value, warn-once enabled. Expects no
    # warning for allowed values, two warnings when both deprecated values are first used, and no
    # further warnings for repeats (including when only one is deprecated again).
    date = "2016-07-04"
    instructions = "This is how you update..."

    @deprecation.deprecated_arg_values(date, instructions, warn_once=True,
                                       arg0="forbidden", arg1="disallowed")
    def _fn(arg0, arg1):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # HELPER: no-op function with two ignored arguments; returns nothing.
      pass

    _fn(arg0="allowed", arg1="also allowed")
    self.assertEqual(0, mock_warning.call_count)
    _fn(arg0="forbidden", arg1="disallowed")
    self.assertEqual(2, mock_warning.call_count)
    _fn(arg0="forbidden", arg1="allowed")
    self.assertEqual(2, mock_warning.call_count)
    _fn(arg0="forbidden", arg1="disallowed")
    self.assertEqual(2, mock_warning.call_count)

  @test.mock.patch.object(logging, "warning", autospec=True)
  @test_util.run_in_graph_and_eager_modes
  def test_deprecated_arg_values_when_value_is_none(self, mock_warning):
    # [FUNCTION DOC]
    # KNOWN ISSUE: tensor equality is switched off only at the end of the test, so a failing
    # assertion leaves it enabled for later tests.
    # Scenario: the deprecated value is the null value, with tensor equality enabled and the test
    # run in both graph and eager modes. Passing a variable must not trigger an error from an
    # explicit equality check and must log no warning; passing null logs exactly one warning.

    @deprecation.deprecated_arg_values("2016-07-04",
                                       "This is how you update...",
                                       warn_once=True,
                                       arg0=None)
    def _fn(arg0):  # pylint: disable=unused-argument
      # [FUNCTION DOC]
      # HELPER: no-op function with one ignored argument; returns nothing.
      pass

    tensor.enable_tensor_equality()
    initial_count = mock_warning.call_count
    # Check that we avoid error from explicit `var == None` check.
    _fn(arg0=variables.Variable(0))
    self.assertEqual(initial_count, mock_warning.call_count)
    _fn(arg0=None)
    self.assertEqual(initial_count + 1, mock_warning.call_count)
    tensor.disable_tensor_equality()


class DeprecationArgumentsTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the argument lookup and docstring rewrite helpers of the deprecation module.

  def testDeprecatedArgumentLookup(self):
    # [FUNCTION DOC]
    # Scenario: call the argument lookup helper with only the new value, only the old value, and
    # both. Expects the supplied value to be returned in the first two cases and a value error
    # saying both names cannot be specified in the third.
    good_value = 3
    self.assertEqual(
        deprecation.deprecated_argument_lookup("val_new", good_value, "val_old",
                                               None), good_value)
    self.assertEqual(
        deprecation.deprecated_argument_lookup("val_new", None, "val_old",
                                               good_value), good_value)
    with self.assertRaisesRegex(ValueError,
                                "Cannot specify both 'val_old' and 'val_new'"):

      deprecation.deprecated_argument_lookup("val_new", good_value,
                                             "val_old", good_value)

  def testRewriteArgumentDocstring(self):
    # [FUNCTION DOC]
    # Scenario: rewrite a docstring that mentions arguments a and b (once in backticks and once
    # as section entries), renaming a to left and b to right in two successive calls. Expects
    # the result to equal the reference docstring using the new names in both places.
    docs = """Add `a` and `b`

    Args:
      a: first arg
      b: second arg
    """
    new_docs = deprecation.rewrite_argument_docstring(
        deprecation.rewrite_argument_docstring(docs, "a", "left"), "b", "right")
    new_docs_ref = """Add `left` and `right`

    Args:
      left: first arg
      right: second arg
    """
    self.assertEqual(new_docs, new_docs_ref)


class DeprecatedEndpointsTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the deprecated-endpoints decorator, which tags a symbol with a tuple of
  # deprecated API names.

  def testSingleDeprecatedEndpoint(self):
    # [FUNCTION DOC]
    # Scenario: tag a function with one endpoint name. Expects the symbol's deprecated-names
    # attribute to be a tuple containing just that name.
    @deprecation.deprecated_endpoints("foo1")
    def foo():
      # [FUNCTION DOC]
      # HELPER: no-op function being tagged; returns nothing.
      pass
    self.assertEqual(("foo1",), foo._tf_deprecated_api_names)

  def testMultipleDeprecatedEndpoint(self):
    # [FUNCTION DOC]
    # Scenario: tag a function with two endpoint names. Expects a tuple with both names in order.
    @deprecation.deprecated_endpoints("foo1", "foo2")
    def foo():
      # [FUNCTION DOC]
      # HELPER: no-op function being tagged; returns nothing.
      pass
    self.assertEqual(("foo1", "foo2"), foo._tf_deprecated_api_names)

  def testCannotSetDeprecatedEndpointsTwice(self):
    # [FUNCTION DOC]
    # Scenario: apply the deprecated-endpoints decorator twice to the same function. Expects the
    # "names already set" error to be raised during decoration.
    with self.assertRaises(deprecation.DeprecatedNamesAlreadySetError):
      @deprecation.deprecated_endpoints("foo1")
      @deprecation.deprecated_endpoints("foo2")
      def foo():  # pylint: disable=unused-variable
        # [FUNCTION DOC]
        # HELPER: no-op function being tagged; returns nothing.
        pass


class DeprecateMovedModuleTest(test.TestCase):
  # [CLASS DOC]
  # Test for the moved-module deprecation hook using the fixture modules deprecated_module and
  # deprecated_module_new. The warning logger is mocked.

  @test.mock.patch.object(logging, "warning", autospec=True)
  def testCallDeprecatedModule(self, mock_warning):
    # [FUNCTION DOC]
    # Scenario: import the deprecated fixture module (inside the test so the import itself does
    # not count) and call a function that only exists in its replacement module. Expects no
    # warning before the call, one warning after the first call, the replacement function's
    # result (one), and no additional warning on a second call.
    from tensorflow.python.util import deprecated_module  # pylint: disable=g-import-not-at-top
    self.assertEqual(0, mock_warning.call_count)
    result = deprecated_module.a()
    self.assertEqual(1, mock_warning.call_count)
    self.assertEqual(1, result)

    deprecated_module.a()
    self.assertEqual(1, mock_warning.call_count)


if __name__ == "__main__":
  test.main()
