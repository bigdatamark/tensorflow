# [MODULE DOC]
# Unit tests for variable_utils. Defines two minimal composite tensor test doubles (one that
# converts its inner variable to a tensor when asked and one that does not) and checks that
# resource variables inside plain structures and composite tensors are converted to tensors,
# and that replacing variables with atoms leaves other leaves untouched and yields structures
# that flatten to the expected values. Tests run in both graph and eager modes.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for variable_utils."""

from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils


class CT(composite_tensor.CompositeTensor):
  # [CLASS DOC]
  # HELPER: minimal composite tensor test double with no components and no conversion override.
  # Used to check that composite tensors without variable conversion pass through unchanged.
  """A generic CompositeTensor, used for constructing tests."""

  @property
  def _type_spec(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: returns the null value instead of a real type specification, so this test
    # double is not a fully valid composite tensor (acceptable here as it is never specialised).
    # Property required by the composite tensor interface; does nothing.
    pass


class CT2(composite_tensor.CompositeTensor):
  # [CLASS DOC]
  # HELPER: composite tensor test double holding a single component and overriding the
  # variable-conversion hook to return a new instance whose component is a tensor.
  """Another CompositeTensor, used for constructing tests."""

  def __init__(self, component):
    # [FUNCTION DOC]
    # Constructor. Input: component, any object (typically a resource variable); stored as the
    # component attribute.
    self.component = component

  @property
  def _type_spec(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: returns the null value instead of a real type specification, so this test
    # double is not a fully valid composite tensor.
    # Property required by the composite tensor interface; does nothing.
    pass

  def _convert_variables_to_tensors(self):
    # [FUNCTION DOC]
    # Conversion hook used by the variable utilities. Returns a new instance of the same class
    # whose component is the tensor obtained by converting the current component. Takes no
    # inputs; does not modify this instance.
    return CT2(ops.convert_to_tensor(self.component))


@test_util.run_all_in_graph_and_eager_modes
class VariableUtilsTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the variable utilities, run in both graph and eager modes via the class
  # decorator. Uses the two composite tensor test doubles defined above as fixtures.

  def test_convert_variables_to_tensors(self):
    # [FUNCTION DOC]
    # Builds a list of two resource variables, a constant, a nested list, an integer and a
    # plain composite tensor (initializing variables first in graph mode). Converts it and
    # asserts the first three entries are tensors, the list and integer keep their types, the
    # composite tensor is the identical object, and after evaluating the tensors the whole
    # result equals the expected values.
    ct = CT()
    data = [resource_variable_ops.ResourceVariable(1),
            resource_variable_ops.ResourceVariable(2),
            constant_op.constant(3),
            [4],
            5,
            ct]
    if not context.executing_eagerly():
      self.evaluate(variables.global_variables_initializer())

    results = variable_utils.convert_variables_to_tensors(data)
    expected_results = [1, 2, 3, [4], 5, ct]
    # Only ResourceVariables are converted to Tensors.
    self.assertIsInstance(results[0], tensor.Tensor)
    self.assertIsInstance(results[1], tensor.Tensor)
    self.assertIsInstance(results[2], tensor.Tensor)
    self.assertIsInstance(results[3], list)
    self.assertIsInstance(results[4], int)
    self.assertIs(results[5], ct)
    results[:3] = self.evaluate(results[:3])
    self.assertAllEqual(results, expected_results)

  def test_convert_variables_in_composite_tensor(self):
    # [FUNCTION DOC]
    # Wraps a resource variable in the converting composite tensor double, asserts its component
    # is a variable, converts it, and asserts the result's component is a tensor equal to one.
    ct2 = CT2(resource_variable_ops.ResourceVariable(1))
    if not context.executing_eagerly():
      self.evaluate(variables.global_variables_initializer())

    self.assertIsInstance(ct2.component,
                          resource_variable_ops.ResourceVariable)
    result = variable_utils.convert_variables_to_tensors(ct2)
    self.assertIsInstance(result.component, tensor.Tensor)
    self.assertAllEqual(result.component, 1)

  def test_replace_variables_with_atoms(self):
    # [FUNCTION DOC]
    # Builds a list of two resource variables, a constant, a nested list and an integer, replaces
    # variables with atoms, and asserts the first two entries become integers while the rest keep
    # their types and values. Finally flattens the result expanding composites and asserts the
    # zero atoms are kept as single leaves (flat values zero, zero, three, four, five).
    data = [resource_variable_ops.ResourceVariable(1),
            resource_variable_ops.ResourceVariable(2),
            constant_op.constant(3),
            [4],
            5]
    if not context.executing_eagerly():
      self.evaluate(variables.global_variables_initializer())

    results = variable_utils.replace_variables_with_atoms(data)
    expected_results = [0, 0, 3, [4], 5]
    # Only ResourceVariables are replaced with int 0s.
    self.assertIsInstance(results[0], int)
    self.assertIsInstance(results[1], int)
    self.assertIsInstance(results[2], tensor.Tensor)
    self.assertIsInstance(results[3], list)
    self.assertIsInstance(results[4], int)
    results[2] = self.evaluate(results[2])
    self.assertAllEqual(results, expected_results)

    # Make sure 0 is a tf.nest atom with expand_composites=True.
    flat_results = nest.flatten(results, expand_composites=True)
    expected_flat_results = [0, 0, 3, 4, 5]
    self.assertAllEqual(flat_results, expected_flat_results)


if __name__ == "__main__":
  test.main()
