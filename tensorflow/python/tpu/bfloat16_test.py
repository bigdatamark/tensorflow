# [MODULE DOC]
# Unit tests for the bfloat16 scope helper (bfloat16.py): scope naming, operation naming inside
# nested scopes, and the dtype behavior of the custom getter. Run through the TensorFlow test
# runner; no import-time side effects.
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
# =============================================================================

"""Tests for bfloat16 helper."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import test
from tensorflow.python.tpu import bfloat16
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables


class BFloat16ScopeTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the bfloat16 scope context manager. Tests are independent and need no
  # shared fixtures.

  def testDefaultScopeName(self):
    # [FUNCTION DOC]
    # Opens the scope without a name and asserts the yielded variable scope has an empty name.
    """Test if name for the variable scope is propagated correctly."""
    with bfloat16.bfloat16_scope() as bf:
      self.assertEqual(bf.name, "")

  def testCustomScopeName(self):
    # [FUNCTION DOC]
    # Opens the scope with the name bfloat16 and asserts the yielded variable scope carries
    # exactly that name.
    """Test if custom name for the variable scope is propagated correctly."""
    name = 'bfloat16'
    with bfloat16.bfloat16_scope('bfloat16') as bf:
      self.assertEqual(bf.name, name)

  def testVariableName(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring is copied from the custom-name test; this test actually checks
    # operation names produced inside nested name scopes, not variable scope names.
    # Scenario: in a fresh graph, three variables are created, then inside a graph name scope a
    # named bfloat16 scope casts two of them to bfloat16 and adds them, and a second unnamed
    # bfloat16 scope casts the third and adds it to the earlier sum. Asserts that the graph
    # contains operations named for the add and cast both under the named scope and directly
    # under the outer scope.
    """Test if custom name for the variable scope is propagated correctly."""
    g = ops.Graph()
    with g.as_default():
      a = variables.Variable(2.2, name='var_a')
      b = variables.Variable(3.3, name='var_b')
      d = variables.Variable(4.4, name='var_b')
      with g.name_scope('scope1'):
        with bfloat16.bfloat16_scope('bf16'):
          a = math_ops.cast(a, dtypes.bfloat16)
          b = math_ops.cast(b, dtypes.bfloat16)
          c = math_ops.add(a, b, name='addition')
        with bfloat16.bfloat16_scope():
          d = math_ops.cast(d, dtypes.bfloat16)
          math_ops.add(c, d, name='addition')

    g_ops = g.get_operations()
    ops_name = []
    for op in g_ops:
      ops_name.append(str(op.name))

    self.assertIn('scope1/bf16/addition', ops_name)
    self.assertIn('scope1/bf16/Cast', ops_name)
    self.assertIn('scope1/addition', ops_name)
    self.assertIn('scope1/Cast', ops_name)

  @test_util.run_deprecated_v1
  def testRequestedDType(self):
    # [FUNCTION DOC]
    # Graph-mode test (deprecated v1 decorator). Inside a bfloat16 scope, gets a variable with
    # default dtype and one with an explicit bfloat16 dtype. Asserts the first is float32, the
    # second reads as bfloat16, and that both underlying global variables in the scope are
    # stored as float32.
    """Test if requested dtype is honored in the getter.
    """
    with bfloat16.bfloat16_scope() as scope:
      v1 = variable_scope.get_variable("v1", [])
      self.assertEqual(v1.dtype.base_dtype, dtypes.float32)
      v2 = variable_scope.get_variable("v2", [], dtype=dtypes.bfloat16)
      self.assertEqual(v2.dtype.base_dtype, dtypes.bfloat16)
      self.assertEqual([dtypes.float32, dtypes.float32],
                       [v.dtype.base_dtype for v in scope.global_variables()])


if __name__ == "__main__":
  test.main()
