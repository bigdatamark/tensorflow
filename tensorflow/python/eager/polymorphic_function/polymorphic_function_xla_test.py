# [MODULE DOC]
# Small test module for traced functions on the XLA test device: verifies that variables created
# inside a traced function (including one assigned in an init scope) work. Defines one XLA test
# case class; running as a script enables eager execution and starts the test runner. No
# import-time side effects.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
from tensorflow.compiler.tests import xla_test
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class FunctionTests(xla_test.XLATestCase):
  # [CLASS DOC]
  # Test case built on the XLA test base class, which provides a device scope for running ops on
  # the XLA test device. Covers variable handling inside traced functions under XLA. No shared
  # fixtures beyond the base class.

  def testVarInitializedInFunction(self):
    # [FUNCTION DOC]
    # Scenario: inside the XLA test device scope, a traced function lazily creates two variables
    # on its first trace (a vector of two floats, and a scalar that is then assigned a new value
    # from within an init scope) and stores them in an outer list. It returns the sum of both
    # variables and the input. The test calls it with a constant 2.0 and asserts the result is
    # approximately 13 and 14, showing that variables created during tracing are initialized
    # correctly and that the init-scope assignment takes effect. Takes no inputs, returns nothing.
    with self.test_scope():
      v_holder = []

      @polymorphic_function.function
      def add_var(x):
        # [FUNCTION DOC]
        # Traced function. On its first trace (when the outer holder list is empty) it creates a
        # two-element float variable and a scalar variable, assigns the scalar a new value inside
        # an init scope, and stores both in the holder. Returns the sum of both variables and the
        # input x (a tensor). Later traces reuse the stored variables.
        if not v_holder:
          v = variables.Variable([1., 2.])
          v_holder.append(v)
          already_initialized = variables.Variable(3.)
          with ops.init_scope():
            already_initialized.assign(10.)
          v_holder.append(already_initialized)
        return v_holder[0] + v_holder[1] + x

      self.assertAllClose([13., 14.], add_var(constant_op.constant(2.)))


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
