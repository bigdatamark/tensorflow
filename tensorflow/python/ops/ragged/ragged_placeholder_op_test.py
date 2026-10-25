# [MODULE DOC]
# Unit tests for the ragged placeholder factory function: graph-mode string form of the created
# placeholders for several dtypes, ragged ranks and value shapes, the error raised in eager
# mode, and the absence of validation ops in the graph. Runs the test main as a script.
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
"""Tests for ragged_placeholder op."""
from absl.testing import parameterized

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import googletest


@test_util.run_all_in_graph_and_eager_modes
class RaggedPlaceholderOpTest(test_util.TensorFlowTestCase,
                              parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the ragged placeholder factory. Run in both graph and eager modes and
  # parameterized for the string-form test. No shared fixtures.

  @parameterized.parameters([
      # dtype, ragged_rank, value_shape, name -> expected
      (dtypes.int32, 0, [5], None,
       'Tensor("Placeholder:0", shape=(5,), dtype=int32)'),
      (dtypes.int32, 1, [], 'ph', 'tf.RaggedTensor('
       'values=Tensor("ph/flat_values:0", shape=(None,), dtype=int32), '
       'row_splits=Tensor("ph/row_splits_0:0", shape=(None,), dtype=int64))'),
      (dtypes.string, 1, [5], 'ph', 'tf.RaggedTensor('
       'values=Tensor("ph/flat_values:0", shape=(None, 5), dtype=string), '
       'row_splits=Tensor("ph/row_splits_0:0", shape=(None,), dtype=int64))'),
      (dtypes.float32, 2, [], 'ph', 'tf.RaggedTensor(values=tf.RaggedTensor('
       'values=Tensor("ph/flat_values:0", shape=(None,), dtype=float32), '
       'row_splits=Tensor("ph/row_splits_1:0", shape=(None,), dtype=int64)), '
       'row_splits=Tensor("ph/row_splits_0:0", shape=(None,), dtype=int64))'),
      (dtypes.int32, 2, [3, 5], 'ph', 'tf.RaggedTensor(values=tf.RaggedTensor('
       'values=Tensor("ph/flat_values:0", shape=(None, 3, 5), dtype=int32), '
       'row_splits=Tensor("ph/row_splits_1:0", shape=(None,), dtype=int64)), '
       'row_splits=Tensor("ph/row_splits_0:0", shape=(None,), dtype=int64))'),
  ])
  def testRaggedPlaceholder(self, dtype, ragged_rank, value_shape, name,
                            expected):
    # [FUNCTION DOC]
    # KNOWN ISSUE: in eager mode the body does nothing, so the test passes without asserting
    # anything; the checks only run in graph mode.
    # Parameterized by element dtype, ragged rank, shape of each flat value, optional placeholder
    # name and the expected text. In graph mode creates a ragged placeholder, converts it to a
    # string with unknown-dimension markers replaced by the word None, and asserts it equals the
    # expected text (a plain tensor when ragged rank is zero, nested ragged tensors otherwise).
    if not context.executing_eagerly():
      placeholder = ragged_factory_ops.placeholder(
          dtype, ragged_rank, value_shape, name)
      result = str(placeholder).replace('?', 'None')
      self.assertEqual(result, expected)

  def testRaggedPlaceholderRaisesExceptionInEagerMode(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: in graph mode the body does nothing, so the test is vacuous there.
    # In eager mode, asserts that creating a ragged placeholder raises a runtime error.
    if context.executing_eagerly():
      with self.assertRaises(RuntimeError):
        ragged_factory_ops.placeholder(dtypes.int32, 1, [])

  def testRaggedPlaceholderDoesNotIncludeValidationOps(self):
    # [FUNCTION DOC]
    # Graph mode only (returns immediately in eager mode). Creates a rank-one float ragged
    # placeholder in a fresh graph and asserts the graph contains exactly two placeholder
    # operations (values and row splits) and no validation operations.
    if context.executing_eagerly():
      return
    graph = ops.Graph()
    with graph.as_default():
      ragged_factory_ops.placeholder(
          dtypes.float32, ragged_rank=1, value_shape=[])
      self.assertEqual([op.type for op in graph.get_operations()],
                       ['Placeholder', 'Placeholder'])


if __name__ == '__main__':
  googletest.main()
