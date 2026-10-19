# [MODULE DOC]
# XLA-device tests for the mode where an eager op is wrapped in a call op at runtime. Defines one
# test case, built on the XLA test base class, that creates and reads variables inside the XLA test
# scope. The class decorator forces the only-as-function variant of the mode. Runs the test runner
# when executed as a script.
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
"""Tests for wrapping an eager op in a call op at runtime."""
from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import test_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables


@test_util.with_eager_op_as_function(only_as_function=True)
class RunEagerOpAsFunctionXlaTest(xla_test.XLATestCase):
  # [CLASS DOC]
  # XLA test case run only with the eager-op-as-function mode (decorator). Covers variable
  # creation, reading and destruction on the XLA test device.

  def testVarCreateReadDestroy(self):
    # [FUNCTION DOC]
    # Scenario: inside the XLA test scope, create a float variable with value one, read its
    # value, and assert it equals one. Variable destruction happens implicitly at scope exit.
    with self.test_scope():
      var = variables.Variable(1.0)
      value = var.read_value()
      self.assertAllEqual(value, 1.0)

  def testReadVariableInFunction(self):
    # [FUNCTION DOC]
    # Scenario: inside the XLA test scope, create a resource variable of value one and call a
    # traced function that reads it. Asserts the returned value converted to numpy equals one.
    with self.test_scope():
      v = resource_variable_ops.ResourceVariable(1.0)

      @def_function.function
      def f():
        # [FUNCTION DOC]
        # Traced function body: returns the current value of the captured resource variable.
        return v.read_value()

      var = f()
      self.assertEqual(1.0, var.numpy())


if __name__ == "__main__":
  test.main()
