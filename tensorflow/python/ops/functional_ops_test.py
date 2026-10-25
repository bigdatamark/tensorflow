# [MODULE DOC]
# Unit tests for the functional If operation in the functional ops module. It checks that the
# conditional op works when its branches are legacy graph-mode Defun functions, polymorphic
# tf.function objects turned into concrete functions, and concrete functions returning a composite
# (sparse) tensor. Test module only; no import-time side effects beyond imports.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for functional operations."""

from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops import functional_ops
from tensorflow.python.platform import test


class FunctionalOpsTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the conditional (If) op of the functional ops module with different kinds of
  # branch functions. It has no shared fixtures; each test builds its own branch functions and inputs.

  def testIfWithDefun(self):
    # [FUNCTION DOC]
    # Builds a fresh graph with two Defun branch functions (add one and subtract one on a float
    # scalar), runs the If op with a false condition and the input ten, and asserts the result is
    # the list containing nine, i.e. the else branch was taken. Defun is graph-mode only.
    # Defun should only be used in graph mode
    with ops.Graph().as_default():
      @function.Defun(dtypes.float32)
      def Then(x):
        # [FUNCTION DOC]
        # Then-branch for the test: takes a float scalar and returns it plus one.
        return x + 1

      @function.Defun(dtypes.float32)
      def Else(x):
        # [FUNCTION DOC]
        # Else-branch for the test: takes a float scalar and returns it minus one.
        return x - 1

      inputs = [10.]
      result = self.evaluate(functional_ops.If(False, inputs, Then, Else))
      self.assertEqual([9.0], result)

  def testIfWithFunction(self):
    # [FUNCTION DOC]
    # Defines two tf.function branches with a scalar float input signature, obtains their concrete
    # functions, runs the If op with a false condition and input ten, and asserts the result is the
    # list containing nine (else branch taken).

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec((), dtypes.float32)])
    def Then(x):
      # [FUNCTION DOC]
      # Then-branch: returns the scalar float input plus one.
      return x + 1

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec((), dtypes.float32)])
    def Else(x):
      # [FUNCTION DOC]
      # Else-branch: returns the scalar float input minus one.
      return x - 1

    inputs = [10.]
    then_cf = Then.get_concrete_function()
    else_cf = Else.get_concrete_function()
    result = self.evaluate(functional_ops.If(False, inputs, then_cf, else_cf))
    self.assertEqual([9.0], result)

  def testIfWithFunctionComposite(self):
    # [FUNCTION DOC]
    # Defines two tf.function branches that return a one-element sparse tensor (value input plus
    # one, or input minus one), runs the If op with a false condition and input ten, and asserts the
    # result is a sparse tensor whose values equal the list containing nine. Verifies the If op
    # supports composite (non-dense) branch outputs.

    signature = [tensor_spec.TensorSpec([], dtypes.float32)]
    @def_function.function(input_signature=signature)
    def Then(x):
      # [FUNCTION DOC]
      # Then-branch: returns a sparse tensor of shape one holding the input plus one at index zero.
      return sparse_tensor.SparseTensor([[0]], [x + 1], [1])

    @def_function.function(input_signature=signature)
    def Else(x):
      # [FUNCTION DOC]
      # Else-branch: returns a sparse tensor of shape one holding the input minus one at index zero.
      return sparse_tensor.SparseTensor([[0]], [x - 1], [1])

    inputs = [10.]
    then_cf = Then.get_concrete_function()
    else_cf = Else.get_concrete_function()
    result = functional_ops.If(False, inputs, then_cf, else_cf)
    self.assertIsInstance(result, sparse_tensor.SparseTensor)
    self.assertAllEqual([9.0], result.values)


if __name__ == '__main__':
  test.main()
