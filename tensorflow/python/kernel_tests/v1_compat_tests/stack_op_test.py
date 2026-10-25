# [MODULE DOC]
# Test module for automatic stacking, where nested Python lists containing symbolic tensors are
# converted into a single tensor, covering V1 graph-only behavior (variables, placeholders and
# shape errors). Defines one test class; no import-time side effects.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""V1 tests for Stack and ParallelStack Ops."""

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class AutomaticStackingTest(test.TestCase):
  # [CLASS DOC]
  # Test case for converting nested lists that contain symbolic tensors into stacked tensors.
  # Tests run in deprecated V1 graph mode; no shared fixtures.

  @test_util.run_deprecated_v1
  # Tests symbolic tensor semantics
  def testVariable(self):
    # [FUNCTION DOC]
    # Scenario: a three by three nested list of zeros with a variable (value 17) in the center is
    # converted to a tensor. After initializing, the result is expected to show 17 in the center;
    # after assigning 38 to the variable, evaluating the same result tensor shows 38, proving the
    # conversion keeps a symbolic dependency on the variable.
    with self.session():
      v = variables.Variable(17)
      result = ops.convert_to_tensor([[0, 0, 0], [0, v, 0], [0, 0, 0]])
      self.evaluate(v.initializer)
      self.assertAllEqual([[0, 0, 0], [0, 17, 0], [0, 0, 0]],
                          self.evaluate(result))

      v.assign(38).op.run()
      self.assertAllEqual([[0, 0, 0], [0, 38, 0], [0, 0, 0]],
                          self.evaluate(result))

  @test_util.run_deprecated_v1
  # Placeholders are V1 only.
  def testPlaceholder(self):
    # [FUNCTION DOC]
    # Scenario: the same three by three list with a scalar int32 placeholder in the center is
    # converted, once with a defined scalar shape and once with an undefined shape. Evaluating with
    # fed values 1 and 2 must give a matrix whose center equals the fed value.
    with self.session():
      # Test using placeholder with a defined shape.
      ph_0 = array_ops.placeholder(dtypes.int32, shape=[])
      result_0 = ops.convert_to_tensor([[0, 0, 0], [0, ph_0, 0], [0, 0, 0]])
      self.assertAllEqual([[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                          result_0.eval(feed_dict={ph_0: 1}))
      self.assertAllEqual([[0, 0, 0], [0, 2, 0], [0, 0, 0]],
                          result_0.eval(feed_dict={ph_0: 2}))

      # Test using placeholder with an undefined shape.
      ph_1 = array_ops.placeholder(dtypes.int32)
      result_1 = ops.convert_to_tensor([[0, 0, 0], [0, ph_1, 0], [0, 0, 0]])
      self.assertAllEqual([[0, 0, 0], [0, 1, 0], [0, 0, 0]],
                          result_1.eval(feed_dict={ph_1: 1}))
      self.assertAllEqual([[0, 0, 0], [0, 2, 0], [0, 0, 0]],
                          result_1.eval(feed_dict={ph_1: 2}))

  @test_util.run_deprecated_v1
  # Placeholders and shape inference are only applicable in Graph mode.
  def testShapeErrors(self):
    # [FUNCTION DOC]
    # Scenario: (1) a placeholder with static shape of one element placed in the nested list must
    # make conversion raise a value error immediately. (2) a placeholder with unknown shape is
    # accepted at graph build time, but evaluating the result with a fed one-element list must
    # raise an invalid-argument error at run time.
    # Static shape error.
    ph_0 = array_ops.placeholder(dtypes.int32, shape=[1])
    with self.assertRaises(ValueError):
      ops.convert_to_tensor([[0, 0, 0], [0, ph_0, 0], [0, 0, 0]])

    # Dynamic shape error.
    ph_1 = array_ops.placeholder(dtypes.int32)
    result_1 = ops.convert_to_tensor([[0, 0, 0], [0, ph_1, 0], [0, 0, 0]])
    with self.session():
      with self.assertRaises(errors_impl.InvalidArgumentError):
        result_1.eval(feed_dict={ph_1: [1]})


if __name__ == "__main__":
  test.main()
