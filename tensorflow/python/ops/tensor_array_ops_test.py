# [MODULE DOC]
# Unit tests for the TensorArray implementation: concatenation of string elements with unknown
# shapes in graph mode, eager mode and inside a compiled function, construction with a numpy
# element shape, and static shape inference for stack and concat. Test module only; running it
# as a script launches the test runner.
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
"""Tests for tensor_array_ops."""

import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.platform import test


class TensorArrayOpsTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering TensorArray behaviors: concat of string arrays with unknown element
  # shapes (graph, eager, in a function), numpy element shapes, and shape inference. No shared
  # fixtures.

  @test_util.run_v1_only('Testing placeholders specifically.')
  def test_concat_graph(self):
    # [FUNCTION DOC]
    # Graph-mode test (v1 only). Builds a string TensorArray of size 4 with unknown element
    # shape and shape inference disabled, writes two placeholder vectors at indices 0 and 2 and
    # empty vectors at 1 and 3, then runs concat in a session feeding two string lists.
    # Asserts the result is the six strings joined in order.
    values = tensor_array_ops.TensorArray(
        size=4, dtype=dtypes.string, element_shape=[None], infer_shape=False)
    a = array_ops.placeholder(dtypes.string, [
        None,
    ])
    b = array_ops.placeholder(dtypes.string, [
        None,
    ])
    values = (values.write(0, a).write(
        1, constant_op.constant([], dtypes.string))).write(2, b).write(
            3, constant_op.constant([], dtypes.string))

    with self.session() as s:
      result = s.run(values.concat(), {a: ['a', 'b', 'c'], b: ['c', 'd', 'e']})
    self.assertAllEqual(result, [b'a', b'b', b'c', b'c', b'd', b'e'])

  @test_util.run_v2_only
  def test_concat(self):
    # [FUNCTION DOC]
    # Eager-mode (v2 only) version of the concat test: writes two constant string vectors and
    # two empty vectors into a size-4 string TensorArray with unknown element shape, then asserts
    # concat returns the six strings in order.
    values = tensor_array_ops.TensorArray(
        size=4, dtype=dtypes.string, element_shape=[None], infer_shape=False)
    a = constant_op.constant(['a', 'b', 'c'], dtypes.string)
    b = constant_op.constant(['c', 'd', 'e'], dtypes.string)
    values = (values.write(0, a).write(
        1, constant_op.constant([], dtypes.string))).write(2, b).write(
            3, constant_op.constant([], dtypes.string))
    self.assertAllEqual(values.concat(), [b'a', b'b', b'c', b'c', b'd', b'e'])

  @test_util.run_v2_only
  def test_concat_in_function(self):
    # [FUNCTION DOC]
    # Same scenario as the eager concat test but the array construction, writes and concat run
    # inside a compiled function taking two string vectors. Asserts the compiled call returns
    # the six concatenated strings in order.
    @def_function.function
    def fn(a, b):
      # [FUNCTION DOC]
      # Nested function under test: builds a size-4 string TensorArray with unknown element
      # shape, writes a, an empty vector, b and an empty vector, and returns the concatenation
      # of all elements. Inputs are two string vectors.
      values = tensor_array_ops.TensorArray(
          size=4, dtype=dtypes.string, element_shape=[None], infer_shape=False)
      values = (values.write(0, a).write(
          1, constant_op.constant([], dtypes.string))).write(2, b).write(
              3, constant_op.constant([], dtypes.string))
      return values.concat()

    self.assertAllEqual(fn(['a', 'b', 'c'], ['c', 'd', 'e']),
                        [b'a', b'b', b'c', b'c', b'd', b'e'])

  def test_init_numpy_shape(self):
    # [FUNCTION DOC]
    # Checks that a TensorArray can be created with a numpy array as its element shape inside a
    # compiled function. Asserts that writing a 2x3 array of ones and concatenating yields the
    # 2x3 matrix of ones.
    @def_function.function
    def fn():
      # [FUNCTION DOC]
      # Nested function: creates a fixed-size (1, non-dynamic) float32 TensorArray with a numpy
      # element shape of 2 by 3, writes a 2x3 array of ones at index 0, and returns concat.
      values = tensor_array_ops.TensorArray(
          np.float32,
          size=1,
          dynamic_size=False,
          element_shape=np.array((2, 3)))
      values = values.write(0, np.ones((2, 3)))
      return values.concat()
    self.assertAllEqual(fn(), [[1., 1., 1.], [1., 1., 1.]])

  def test_shape_inference_stack_concat(self):
    # [FUNCTION DOC]
    # Creates a float32 TensorArray of size 4 with no declared element shape, writes a 2x3
    # array at index 0, and asserts that the static shape of stack is 4 by 2 by 3 and that of
    # concat is 8 by 3, i.e. shape is inferred from the written element.
    arr = tensor_array_ops.TensorArray(size=4, dtype=dtypes.float32)
    new_arr = arr.write(0, np.ones((2, 3)))
    self.assertEqual(new_arr.stack().shape, (4, 2, 3))
    self.assertEqual(new_arr.concat().shape, (8, 3))


if __name__ == '__main__':
  test.main()
