# [MODULE DOC]
# Unit tests for the AutoGraph function_wrappers module. Checks the function scope context
# manager with the name scopes feature, with the automatic control dependencies feature, and with
# everything disabled. Runs the TensorFlow test main when executed as a script.
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
"""Tests for function_wrappers module."""

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.core import function_wrappers
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class FunctionWrappersTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the function scope context manager. No shared fixtures.

  def test_name_scope(self):
    # [FUNCTION DOC]
    # Scenario: a function scope with only the name scopes feature enabled, named test_name.
    # A constant is created inside the scope. Asserts the constant's name contains the scope name.
    # Skipped in eager mode since tensor names are disabled there.
    if context.executing_eagerly():
      self.skipTest('Tensor names are disabled in eager')

    with function_wrappers.FunctionScope(
        'test_name', None,
        converter.ConversionOptions(
            optional_features=converter.Feature.NAME_SCOPES)):
      t = constant_op.constant(1)
    self.assertIn('test_name', t.name)

  def test_auto_control_deps(self):
    # [FUNCTION DOC]
    # Scenario: a function scope with only the automatic control dependencies feature. Inside it,
    # a variable initialized to one is assigned two and a constant is passed through the scope's
    # return method. Evaluating the returned op must run the assignment first, so the variable's
    # value afterwards is asserted to be two.
    v = variables.Variable(1)
    with function_wrappers.FunctionScope(
        '_', None,
        converter.ConversionOptions(
            optional_features=converter.Feature.AUTO_CONTROL_DEPS)) as scope:
      v.assign(2)
      op = scope.ret(constant_op.constant(1), True)
    self.evaluate(op)
    self.assertEqual(self.evaluate(v.read_value()), 2)

  def test_all_disabled(self):
    # [FUNCTION DOC]
    # Scenario: a function scope created with no function name and the standard options (no
    # optional features). A constant is created inside it and asserted to evaluate to one,
    # i.e. the scope is a harmless no-op.
    with function_wrappers.FunctionScope(None, None,
                                         converter.STANDARD_OPTIONS):
      t = constant_op.constant(1)
    self.assertEqual(self.evaluate(t), 1)


if __name__ == '__main__':
  test.main()
