# [MODULE DOC]
# Test module for the fast module type class (a module type with replaceable attribute-access
# callbacks and a fast dictionary cache). Defines small fixture subclasses and tests early
# attribute access, constructor argument errors, callbacks and the fast-dictionary helpers.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.python.util.fast_module_type."""

from tensorflow.python.platform import test
from tensorflow.python.util import fast_module_type
FastModuleType = fast_module_type.get_fast_module_type_class()


class ChildFastModule(FastModuleType):
  # [CLASS DOC]
  # HELPER: test fixture subclass of the fast module type with three replaceable attribute
  # lookup callbacks (two for the attribute-access hook, one for the fallback hook) that the
  # tests install on instances.

  def _getattribute1(self, name):  # pylint: disable=unused-argument
    # [FUNCTION DOC]
    # HELPER: test callback for the attribute-access hook; ignores the requested name and
    # always returns the number two.
    return 2

  def _getattribute2(self, name):  # pylint: disable=unused-argument
    # [FUNCTION DOC]
    # HELPER: test callback for the attribute-access hook; ignores the name and always raises an
    # attribute error so that lookup falls through to the fallback hook.
    raise AttributeError("Pass to getattr")

  def _getattr(self, name):  # pylint: disable=unused-argument
    # [FUNCTION DOC]
    # HELPER: test callback for the fallback attribute hook; ignores the name and always
    # returns the number three.
    return 3


class EarlyAttrAccessModule(FastModuleType):
  # [CLASS DOC]
  # HELPER: test fixture module type whose constructor sets an attribute before calling the
  # parent constructor, to verify that early attribute access does not crash.

  def __init__(self, name):
    # [FUNCTION DOC]
    # HELPER: constructor for the test fixture. Input: name, the module name string. Sets an
    # attribute with value one before delegating to the parent constructor with the name.
    self.some_attr = 1
    super().__init__(name)  # pytype: disable=too-many-function-args


class FastModuleTypeTest(test.TestCase):
  # [CLASS DOC]
  # Tests for the fast module type class: early attribute access, constructor argument
  # validation, default lookup, replaceable lookup callbacks, and the fast-dictionary helpers.

  def testAttributeAccessBeforeSuperInitDoesNotCrash(self):
    # [FUNCTION DOC]
    # Scenario: create the fixture that sets an attribute before the parent constructor runs.
    # Asserts that the attribute is readable afterwards with its value (one) and nothing crashes.
    # Tests that the attribute access before super().__init__() does not crash.
    module = EarlyAttrAccessModule("early_attr")
    self.assertEqual(1, module.some_attr)

  def testMissingModuleNameCallDoesNotCrash(self):
    # [FUNCTION DOC]
    # Scenario: construct the child module without the required name argument.
    # Asserts a type error is raised (rather than a crash).
    with self.assertRaises(TypeError):
      ChildFastModule()

  def testBaseGetattribute(self):
    # [FUNCTION DOC]
    # Scenario: create a child module, set an ordinary attribute and read it back.
    # Asserts the default lookup returns the stored value (one).
    # Tests that the default attribute lookup works.
    module = ChildFastModule("test")
    module.foo = 1
    self.assertEqual(1, module.foo)

  def testGetattributeCallback(self):
    # [FUNCTION DOC]
    # Scenario: install the first test callback as the attribute-access hook of a module.
    # Asserts that reading any undefined attribute returns the callback result (two).
    # Tests that functionality of __getattribute__ can be set as a callback.
    module = ChildFastModule("test")
    FastModuleType.set_getattribute_callback(module,
                                             ChildFastModule._getattribute1)
    self.assertEqual(2, module.foo)

  def testGetattrCallback(self):
    # [FUNCTION DOC]
    # Scenario: install an attribute-access hook that raises an attribute error and a fallback
    # hook that returns three.
    # Asserts that reading an undefined attribute yields the fallback value (three).
    # Tests that functionality of __getattr__ can be set as a callback.
    module = ChildFastModule("test")
    FastModuleType.set_getattribute_callback(module,
                                             ChildFastModule._getattribute2)
    FastModuleType.set_getattr_callback(module, ChildFastModule._getattr)
    self.assertEqual(3, module.foo)

  def testFastdictApis(self):
    # [FUNCTION DOC]
    # Scenario: exercise the fast-dictionary helpers on a fresh module.
    # Asserts that a missing key is reported absent, that fetching it raises a key error with a
    # "module has no attribute" message, and that after inserting the key it is reported present
    # and readable as a normal attribute with the inserted value (one).
    module = ChildFastModule("test")
    # At first "bar" does not exist in the module's attributes
    self.assertFalse(module._fastdict_key_in("bar"))
    with self.assertRaisesRegex(KeyError, "module has no attribute 'bar'"):
      module._fastdict_get("bar")

    module._fastdict_insert("bar", 1)
    # After _fastdict_insert() the attribute is added.
    self.assertTrue(module._fastdict_key_in("bar"))
    self.assertEqual(1, module.bar)


if __name__ == "__main__":
  test.main()
