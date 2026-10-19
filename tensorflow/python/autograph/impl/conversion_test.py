# [MODULE DOC]
# Unit tests for the AutoGraph conversion module, focused on the is_allowlisted check that decides
# whether a callable should be left unconverted. Covers plain functions, framework modules,
# module look-alikes, callable objects, bound methods wrapped by the eager function machinery,
# and natively bound (pybind) methods. Defines one test class and a main entry point; some tests
# modify the global conversion rules and the module registry without restoring them.
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
"""Tests for conversion module."""

import sys
import types
import weakref

from tensorflow.python.autograph import utils
from tensorflow.python.autograph.core import config
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.impl import api
from tensorflow.python.autograph.impl import conversion
from tensorflow.python.autograph.impl.testing import pybind_for_testing
from tensorflow.python.eager import function
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test


class ConversionTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the allowlisting logic of the conversion module. It has no shared fixtures;
  # each test builds its own functions, classes and fake modules.

  def _simple_program_ctx(self):
    # [FUNCTION DOC]
    # HELPER: builds a program context for tests (currently unused by any test in this file).
    # Takes no inputs; returns a program context with recursive conversion enabled and the
    # public autograph api module as the autograph module. No side effects.
    return converter.ProgramContext(
        options=converter.ConversionOptions(recursive=True),
        autograph_module=api)

  def test_is_allowlisted(self):
    # [FUNCTION DOC]
    # Checks the basic allowlist answers: a user-defined function is not allowlisted, while the
    # autograph utils module and a framework op function are. Returns nothing.

    def test_fn():
      # [FUNCTION DOC]
      # Sample user function that returns a constant tensor holding one; only inspected, not
      # called.
      return constant_op.constant(1)

    self.assertFalse(conversion.is_allowlisted(test_fn))
    self.assertTrue(conversion.is_allowlisted(utils))
    self.assertTrue(conversion.is_allowlisted(constant_op.constant))

  def test_is_allowlisted_tensorflow_like(self):
    # [FUNCTION DOC]
    # Scenario: a function lives in a module whose name merely starts like the framework name.
    # Expects it not to be allowlisted. Returns nothing.
    # KNOWN ISSUE: the function's module attribute is set to the module object rather than to its
    # name string, so the test may not exercise name-prefix matching as intended.

    tf_like = types.ModuleType('tensorflow_foo')
    def test_fn():
      # [FUNCTION DOC]
      # Empty sample function attached to the look-alike module; does nothing.
      pass
    tf_like.test_fn = test_fn
    test_fn.__module__ = tf_like

    self.assertFalse(conversion.is_allowlisted(tf_like.test_fn))

  def test_is_allowlisted_callable_allowlisted_call(self):
    # [FUNCTION DOC]
    # Scenario: a fake module is registered and a do-not-convert rule for it is prepended to the
    # global conversion rules. A class in that module has a call method and another method; a
    # subclass defined here adds a method. Expects the call method, instances, bound call and the
    # inherited method to be allowlisted, but not the subclass itself or its own new method.
    # Side effects: registers a module in the module registry and permanently changes the global
    # conversion rules (not restored). Returns nothing.

    allowlisted_mod = types.ModuleType('test_allowlisted_call')
    sys.modules['test_allowlisted_call'] = allowlisted_mod
    config.CONVERSION_RULES = ((config.DoNotConvert('test_allowlisted_call'),) +
                               config.CONVERSION_RULES)

    class TestClass:
      # [CLASS DOC]
      # Sample callable class pretending to belong to the allowlisted fake module.

      def __call__(self):
        # [FUNCTION DOC]
        # Empty call operator; does nothing.
        pass

      def allowlisted_method(self):
        # [FUNCTION DOC]
        # Empty method expected to count as allowlisted; does nothing.
        pass

    TestClass.__module__ = 'test_allowlisted_call'
    TestClass.__call__.__module__ = 'test_allowlisted_call'

    class Subclass(TestClass):
      # [CLASS DOC]
      # Subclass of the sample class, defined in the test module so it is not allowlisted.

      def converted_method(self):
        # [FUNCTION DOC]
        # Empty method expected not to be allowlisted; does nothing.
        pass

    tc = Subclass()

    self.assertTrue(conversion.is_allowlisted(TestClass.__call__))
    self.assertTrue(conversion.is_allowlisted(tc))
    self.assertTrue(conversion.is_allowlisted(tc.__call__))
    self.assertTrue(conversion.is_allowlisted(tc.allowlisted_method))
    self.assertFalse(conversion.is_allowlisted(Subclass))
    self.assertFalse(conversion.is_allowlisted(tc.converted_method))

  def test_is_allowlisted_tfmethodwrapper(self):
    # [FUNCTION DOC]
    # Scenario: a function is bound as a method to a target wrapper (holding a weak reference to
    # an object of an allowlisted fake class plus that object's method). Expects the bound method
    # to be treated as allowlisted. Side effects: registers a fake module and prepends a
    # do-not-convert rule to the global conversion rules (not restored). Returns nothing.
    allowlisted_mod = types.ModuleType('test_allowlisted_call')
    sys.modules['test_allowlisted_call'] = allowlisted_mod
    config.CONVERSION_RULES = ((config.DoNotConvert('test_allowlisted_call'),) +
                               config.CONVERSION_RULES)

    class TestClass:
      # [CLASS DOC]
      # Sample class assigned to the allowlisted fake module; provides a method to wrap.

      def member_function(self):
        # [FUNCTION DOC]
        # Empty method used as the wrapped target; does nothing.
        pass

    TestClass.__module__ = 'test_allowlisted_call'
    test_obj = TestClass()

    def test_fn(self):
      # [FUNCTION DOC]
      # Sample function to be bound as a method; ignores its instance argument and returns nothing.
      del self

    bound_method = types.MethodType(
        test_fn,
        function.TfMethodTarget(
            weakref.ref(test_obj), test_obj.member_function))

    self.assertTrue(conversion.is_allowlisted(bound_method))

  def test_is_allowlisted_pybind(self):
    # [FUNCTION DOC]
    # Scenario: with the conversion rules temporarily emptied, a method of a natively bound test
    # class is checked. Currently expects it NOT to be allowlisted (a TODO says it should be);
    # native bindings are handled by a separate check. Rules are restored by the patch. Returns
    # nothing.
    test_object = pybind_for_testing.TestClassDef()
    with test.mock.patch.object(config, 'CONVERSION_RULES', ()):
      # TODO(mdan): This should return True for functions and methods.
      # Note: currently, native bindings are allowlisted by a separate check.
      self.assertFalse(conversion.is_allowlisted(test_object.method))


if __name__ == '__main__':
  test.main()
