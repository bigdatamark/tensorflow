# [MODULE DOC]
# FileCheck test for importing a TF2 SavedModel into MLIR with structured return values. Verifies
# the index paths attached to function results for single values, multiple values (tuple or list
# forms), nested lists, mappings, and functions with multiple return statements. Uses the common
# test harness.
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

# RUN: %p/structured_output | FileCheck %s

# pylint: disable=missing-docstring,line-too-long
import tensorflow.compat.v2 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common


class TestModule(tf.Module):
  # [CLASS DOC]
  # Test module with several exported functions with different return structures. Function names
  # carry a numeric prefix so the imported output order matches source order.
  # The fNNNN name prefixes in this file are such that the sorted order of the
  # functions in the resulting MLIR output match the order in the source file,
  # allowing us to conveniently co-locate the CHECK's with the code they are
  # checking.
  #
  # Note: CHECK-DAG doesn't work with CHECK-SAME/CHECK-NEXT.

  # Check index paths for results.
  #
  # CHECK:      func {{@[a-zA-Z_0-9]+}}() -> (
  # CHECK-SAME:   tensor<1xf32> {tf_saved_model.index_path = []})
  # CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["f0000_single_return"]
  @tf.function(input_signature=[])
  def f0000_single_return(self):
    # [FUNCTION DOC]
    # No-argument function returning one constant float tensor of shape 1. Expects an empty
    # result index path.
    return tf.constant(1.0, shape=[1])

  # Check index paths for results with multiple return values.
  # Note that semantically in Python, multiple return values are equivalent
  # to returning a tuple/list.
  #
  # CHECK:      func {{@[a-zA-Z_0-9]+}}() -> (
  # CHECK-SAME:   tensor<1xf32> {tf_saved_model.index_path = [0]},
  # CHECK-SAME:   tensor<2xf32> {tf_saved_model.index_path = [1]})
  # CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["f0001_multiple_results_no_punctuation"]
  @tf.function(input_signature=[])
  def f0001_multiple_results_no_punctuation(self):
    # [FUNCTION DOC]
    # No-argument function returning two constant float tensors (shapes 1 and 2) as a bare tuple.
    # Expects result index paths 0 and 1.
    return tf.constant(1.0, shape=[1]), tf.constant(1.0, shape=[2])

  # Check index paths for results written explicitly with parentheses.
  # This is semantically equivalent to the earlier test without parentheses,
  # but this test serves as documentation of this behavior for the purposes
  # of tf_saved_model users.
  #
  # CHECK:      func {{@[a-zA-Z_0-9]+}}() -> (
  # CHECK-SAME:   tensor<1xf32> {tf_saved_model.index_path = [0]},
  # CHECK-SAME:   tensor<2xf32> {tf_saved_model.index_path = [1]})
  # CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["f0002_multiple_results_parentheses"]
  @tf.function(input_signature=[])
  def f0002_multiple_results_parentheses(self):
    # [FUNCTION DOC]
    # Same as the bare tuple case but the tuple is written with parentheses; expects identical
    # result index paths 0 and 1.
    return (tf.constant(1.0, shape=[1]), tf.constant(1.0, shape=[2]))

  # Check index paths for results written explicitly with brackets.
  # This is semantically equivalent to the earlier test without parentheses,
  # but this test serves as documentation of this behavior for the purposes
  # of tf_saved_model users.
  #
  # CHECK:      func {{@[a-zA-Z_0-9]+}}() -> (
  # CHECK-SAME:   tensor<1xf32> {tf_saved_model.index_path = [0]},
  # CHECK-SAME:   tensor<2xf32> {tf_saved_model.index_path = [1]})
  # CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["f0003_multiple_results_brackets"]
  @tf.function(input_signature=[])
  def f0003_multiple_results_brackets(self):
    # [FUNCTION DOC]
    # Same two tensors returned as a list; expects result index paths 0 and 1, the same as for a
    # tuple.
    return [tf.constant(1.0, shape=[1]), tf.constant(1.0, shape=[2])]

  # Check index paths for lists.
  #
  # CHECK:      func {{@[a-zA-Z_0-9]+}}() -> (
  # CHECK-SAME:   tensor<1xf32> {tf_saved_model.index_path = [0, 0]},
  # CHECK-SAME:   tensor<2xf32> {tf_saved_model.index_path = [0, 1]})
  # CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["f0004_list_2_elements"]
  @tf.function(input_signature=[])
  def f0004_list_2_elements(self):
    # [FUNCTION DOC]
    # Returns a single value that is a list of two tensors (a list nested in the result). Expects
    # result index paths 0,0 and 0,1.
    return [[tf.constant(1.0, shape=[1]), tf.constant(1.0, shape=[2])]]

  # Check index paths for dicts.
  # Keys are linearized in sorted order, matching `tf.nest.flatten`.
  # More thorough testing of this is in structured_input.py. The underlying code
  # path for linearization is shared, so no need to replicate that testing here.
  #
  # CHECK:      func {{@[a-zA-Z_0-9]+}}() -> (
  # CHECK-SAME:   tensor<1xf32> {tf_saved_model.index_path = ["x"]},
  # CHECK-SAME:   tensor<2xf32> {tf_saved_model.index_path = ["y"]})
  # CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["f0005_dict_2_keys"]
  @tf.function(input_signature=[])
  def f0005_dict_2_keys(self):
    # [FUNCTION DOC]
    # Returns a mapping with keys x (tensor of shape 1) and y (tensor of shape 2). Expects result
    # index paths using the key names in sorted order.
    return {
        'x': tf.constant(1.0, shape=[1]),
        'y': tf.constant(1.0, shape=[2]),
    }

  # Check index paths for outputs are correctly handled in the presence of
  # multiple return statements.
  #
  # CHECK:      func {{@[a-zA-Z_0-9]+}}(
  # CHECK-SAME:   %arg0: tensor<f32> {tf._user_specified_name = "x", tf_saved_model.index_path = [0]}
  # CHECK-SAME: ) -> (
  # CHECK-SAME:   tensor<1xf32> {tf_saved_model.index_path = ["x"]})
  # CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["f0006_multiple_return_statements"]
  @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
  def f0006_multiple_return_statements(self, x):
    # [FUNCTION DOC]
    # Takes a scalar float x and returns, in both branches of a conditional on x being greater than
    # three, a mapping with key x holding a constant tensor of shape 1. Checks that result index
    # paths are correct when the function has multiple return statements.
    if x > 3.:
      return {'x': tf.constant(1.0, shape=[1])}
    else:
      return {'x': tf.constant(1.0, shape=[1])}


if __name__ == '__main__':
  common.do_test(TestModule)
