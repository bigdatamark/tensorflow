# [MODULE DOC]
# Tests for version 4 of the multiplex custom op example: single and list forms of the op,
# round-tripping a model through a SavedModel, mixing both forms in one traced function, and the
# error for inconsistent inputs. Importing the op modules loads compiled shared libraries. Run
# directly, it starts the TensorFlow test runner.
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
"""Tests for multiplex_4."""

import os

import numpy as np
import tensorflow as tf

from tensorflow.examples.custom_ops_doc.multiplex_4 import model_using_multiplex
from tensorflow.examples.custom_ops_doc.multiplex_4 import multiplex_4_op
# This pylint disable is only needed for internal google users
from tensorflow.python.framework import errors_impl  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import test_util  # pylint: disable=g-direct-tensorflow-import


@test_util.with_eager_op_as_function
class MultiplexOpTest(tf.test.TestCase):
  # [CLASS DOC]
  # Test case for the multiplex op wrapper of version 4. Most tests run in both graph and eager
  # modes. Has a private traced helper method shared by one test; no other fixtures.

  @test_util.run_in_graph_and_eager_modes
  def test_multiplex_int(self):
    # [FUNCTION DOC]
    # Uses the old single-tensor calling form on 64-bit integer vectors and asserts the result
    # equals a numpy where reference. Returns nothing.
    a = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
    b = tf.constant([10, 20, 30, 40, 50], dtype=tf.int64)
    cond = tf.constant([True, False, True, False, True], dtype=bool)
    expect = np.where(self.evaluate(cond), self.evaluate(a), self.evaluate(b))
    # expected result is [1, 20, 3, 40, 5]
    result = multiplex_4_op.multiplex(cond, a, b)
    self.assertAllEqual(result, expect)

  @test_util.run_in_graph_and_eager_modes
  def test_multiplex_select(self):
    # [FUNCTION DOC]
    # Uses the list calling form with three conditions and three value tensors plus a fallback,
    # and asserts the result equals the numpy select reference (first true condition wins).
    # Returns nothing.
    a1 = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
    a2 = tf.constant([6, 7, 8, 9, 10], dtype=tf.int64)
    a3 = tf.constant([11, 12, 13, 14, 15], dtype=tf.int64)
    a = [a1, a2, a3]
    b = tf.constant([101, 102, 103, 104, 105], dtype=tf.int64)
    cond1 = tf.constant([False, False, True, False, False], dtype=bool)
    cond2 = tf.constant([False, False, False, False, True], dtype=bool)
    cond3 = tf.constant([True, False, True, False, True], dtype=bool)
    cond = [cond1, cond2, cond3]
    expect = np.select([self.evaluate(i) for i in cond],
                       [self.evaluate(i) for i in a], self.evaluate(b))
    # expected result is [11, 102, 3, 104, 10]
    result = multiplex_4_op.multiplex(cond, a, b)
    self.assertAllEqual(result, expect)

  def test_multiplex_saved_model(self):
    # [FUNCTION DOC]
    # Saves a model using the version 4 op into a temporary directory, loads and runs it, and
    # asserts the result is the expected vector (1, 20, 3, 40, 5). Writes and reads temp files.
    # Returns nothing.
    path = os.path.join(self.create_tempdir(), 'model')
    model_using_multiplex.save(multiplex_4_op.multiplex, path)
    result = model_using_multiplex.load_and_use(path)
    self.assertAllEqual(result, tf.constant([1, 20, 3, 40, 5], dtype=tf.int64))

  # One tf.function that uses both multiplex with single tensors for `cond`
  # and `a` and with lists of tensors for `cond` and `a`, i.e. a graph
  # with two example_multiplex_dense kernels that have different numbers
  # of inputs.
  @tf.function
  def _both(self):
    # [FUNCTION DOC]
    # HELPER: traced function for test_both_single_and_list. Builds a three-way list-form
    # multiplex, then feeds its result to a single-tensor-form multiplex with another condition,
    # so one graph contains kernels with different numbers of inputs. Returns the final tensor.
    a1 = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
    a2 = tf.constant([6, 7, 8, 9, 10], dtype=tf.int64)
    a3 = tf.constant([11, 12, 13, 14, 15], dtype=tf.int64)
    a_123 = [a1, a2, a3]
    b_123 = tf.constant([101, 102, 103, 104, 105], dtype=tf.int64)
    cond1 = tf.constant([False, False, True, False, False], dtype=bool)
    cond2 = tf.constant([False, False, False, False, True], dtype=bool)
    cond3 = tf.constant([True, False, True, False, True], dtype=bool)
    cond_123 = [cond1, cond2, cond3]
    mux_123 = multiplex_4_op.multiplex(cond_123, a_123, b_123)
    b4 = tf.constant([201, 202, 203, 204, 205], dtype=tf.int64)
    cond4 = tf.constant([True, True, True, False, False], dtype=bool)
    result = multiplex_4_op.multiplex(cond4, mux_123, b4)
    return result

  def test_both_single_and_list(self):
    # [FUNCTION DOC]
    # Runs the traced helper and asserts the result is the vector (11, 102, 3, 204, 205).
    # Returns nothing.
    result = self._both()
    self.assertAllEqual(result,
                        tf.constant([11, 102, 3, 204, 205], dtype=tf.int64))

  @test_util.run_in_graph_and_eager_modes
  def test_inconsistent_inputs_error(self):
    # [FUNCTION DOC]
    # Passes a list of two value tensors with a single condition tensor and asserts evaluation
    # raises an invalid-argument error (eager) or value error (graph) about mismatched shape or
    # rank. Returns nothing.
    a1 = tf.constant([1, 2, 3, 4, 5], dtype=tf.int64)
    a2 = tf.constant([6, 7, 8, 9, 10], dtype=tf.int64)
    a = [a1, a2]
    b = tf.constant([101, 102, 103, 104, 105], dtype=tf.int64)
    cond = tf.constant([False, False, True, False, False], dtype=bool)
    with self.assertRaisesRegex(
        (errors_impl.InvalidArgumentError, ValueError),
        # Eager mode raises InvalidArgumentError with the following message
        r'(a_values\[0\] and b_values must have the same shape'
        r')|('
        # Graph mode raises ValueError with the following message
        r'Shapes must be equal rank, but are 2 and 1)'):
      self.evaluate(multiplex_4_op.multiplex(cond, a, b))


if __name__ == '__main__':
  tf.test.main()
