# [MODULE DOC]
# FileCheck test for importing a TF2 SavedModel into MLIR with structured (nested) input
# signatures. Verifies the index paths attached to function arguments for plain argument lists,
# lists, dictionaries (sorted key order regardless of declaration order) and nested combinations.
# Uses the common test harness.
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

# RUN: %p/structured_input | FileCheck %s

# pylint: disable=missing-docstring,line-too-long
import tensorflow.compat.v2 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common


class TestModule(tf.Module):
  # [CLASS DOC]
  # Test module with several exported do-nothing functions, each with a differently structured
  # input signature. Function names carry a numeric prefix so the imported output order matches
  # source order. Only the signatures matter; all bodies return nothing.
  # The fNNNN name prefixes in this file are such that the sorted order of the
  # functions in the resulting MLIR output match the order in the source file,
  # allowing us to conveniently co-locate the CHECK's with the code they are
  # checking.
  #
  # Note: CHECK-DAG doesn't work with CHECK-SAME/CHECK-NEXT.

  # Check index paths for arguments.
  # The outer layer of the index path indexes into the arguments.
  #
  # CHECK:      func {{@[a-zA-Z_0-9]+}}(
  # CHECK-SAME:   %arg0: tensor<1xf32> {tf._user_specified_name = "x", tf_saved_model.index_path = [0]},
  # CHECK-SAME:   %arg1: tensor<2xf32> {tf._user_specified_name = "y", tf_saved_model.index_path = [1]})
  # CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["f0000_function_arity"]
  @tf.function(input_signature=[
      tf.TensorSpec([1], tf.float32),
      tf.TensorSpec([2], tf.float32)
  ])
  def f0000_function_arity(self, x, y):
    # [FUNCTION DOC]
    # Function with two separate tensor arguments: x of shape 1 and y of shape 2 (floats). The
    # test expects top level index paths 0 and 1. Returns nothing.
    return

  # Check index paths for lists.
  #
  # CHECK:      func {{@[a-zA-Z_0-9]+}}(
  # CHECK-SAME:   %arg0: tensor<f32> {tf._user_specified_name = "l", tf_saved_model.index_path = [0, 0]},
  # CHECK-SAME:   %arg1: tensor<f32> {tf._user_specified_name = "l", tf_saved_model.index_path = [0, 1]})
  # CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["f0001_list_2_elements"]
  @tf.function(input_signature=[[
      tf.TensorSpec([], tf.float32),
      tf.TensorSpec([], tf.float32),
  ]])
  def f0001_list_2_elements(self, l):
    # [FUNCTION DOC]
    # Function with one argument l that is a list of two scalar float tensors. Expected index
    # paths are the argument position followed by the list position. Returns nothing.
    return

  # Check index paths for dicts.
  # Keys are linearized in sorted order, matching `tf.nest.flatten`.
  #
  # CHECK:      func {{@[a-zA-Z_0-9]+}}(
  # CHECK-SAME:   %arg0: tensor<1xf32> {tf._user_specified_name = "d", tf_saved_model.index_path = [0, "x"]},
  # CHECK-SAME:   %arg1: tensor<2xf32> {tf._user_specified_name = "d", tf_saved_model.index_path = [0, "y"]})
  # CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["f0002_dict_2_keys"]
  @tf.function(input_signature=[{
      'x': tf.TensorSpec([1], tf.float32),
      'y': tf.TensorSpec([2], tf.float32),
  }])
  def f0002_dict_2_keys(self, d):
    # [FUNCTION DOC]
    # Function with one argument d that is a mapping with keys x (shape 1) and y (shape 2). The
    # index paths are the argument position followed by the key. Returns nothing.
    return

  # Check index paths for dicts, where the keys are not in sorted order.
  # The index path should be insensitive to the key order.
  #
  # CHECK:      func {{@[a-zA-Z_0-9]+}}(
  # CHECK-SAME:   %arg0: tensor<1xf32> {tf._user_specified_name = "d", tf_saved_model.index_path = [0, "x"]},
  # CHECK-SAME:   %arg1: tensor<2xf32> {tf._user_specified_name = "d", tf_saved_model.index_path = [0, "y"]})
  # CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["f0003_dict_2_keys_out_of_order"]
  @tf.function(input_signature=[{
      'y': tf.TensorSpec([2], tf.float32),
      'x': tf.TensorSpec([1], tf.float32),
  }])
  def f0003_dict_2_keys_out_of_order(self, d):
    # [FUNCTION DOC]
    # Same as the two key mapping case but the keys are declared in reverse order (y before x).
    # The test expects identical sorted-key ordering and index paths. Returns nothing.
    return

  # Slightly stronger stress test of multiple dict keys.
  #
  # CHECK:      func {{@[a-zA-Z_0-9]+}}(
  # CHECK-SAME:   %arg0: tensor<1xf32> {tf._user_specified_name = "d", tf_saved_model.index_path = [0, "a"]},
  # CHECK-SAME:   %arg1: tensor<2xf32> {tf._user_specified_name = "d", tf_saved_model.index_path = [0, "b"]},
  # CHECK-SAME:   %arg2: tensor<3xf32> {tf._user_specified_name = "d", tf_saved_model.index_path = [0, "c"]},
  # CHECK-SAME:   %arg3: tensor<4xf32> {tf._user_specified_name = "d", tf_saved_model.index_path = [0, "x"]},
  # CHECK-SAME:   %arg4: tensor<5xf32> {tf._user_specified_name = "d", tf_saved_model.index_path = [0, "y"]},
  # CHECK-SAME:   %arg5: tensor<6xf32> {tf._user_specified_name = "d", tf_saved_model.index_path = [0, "z"]})
  # CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["f0004_dict_many_keys"]
  @tf.function(input_signature=[{
      'x': tf.TensorSpec([4], tf.float32),
      'y': tf.TensorSpec([5], tf.float32),
      'z': tf.TensorSpec([6], tf.float32),
      'a': tf.TensorSpec([1], tf.float32),
      'b': tf.TensorSpec([2], tf.float32),
      'c': tf.TensorSpec([3], tf.float32),
  }])
  def f0004_dict_many_keys(self, d):
    # [FUNCTION DOC]
    # Function with a mapping argument d with six keys (a, b, c, x, y, z) declared in unsorted
    # order, with tensor shapes of sizes one through six. Expects arguments in sorted key order.
    # Returns nothing.
    return

  # Check a slightly more complex recursive structure.
  # Note that list elements can have heterogenous types.
  #
  # CHECK:      func {{@[a-zA-Z_0-9]+}}(
  # CHECK-SAME:   %arg0: tensor<1xf32> {tf._user_specified_name = "d", tf_saved_model.index_path = [0, "x", 0]},
  # CHECK-SAME:   %arg1: tensor<2xf32> {tf._user_specified_name = "d", tf_saved_model.index_path = [0, "x", 1]},
  # CHECK-SAME:   %arg2: tensor<3xf32> {tf._user_specified_name = "d", tf_saved_model.index_path = [0, "y"]})
  # CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["f0005_more_complex_recursive_structure"]
  @tf.function(input_signature=[{
      'x': [tf.TensorSpec([1], tf.float32),
            tf.TensorSpec([2], tf.float32)],
      'y': tf.TensorSpec([3], tf.float32),
  }])
  def f0005_more_complex_recursive_structure(self, d):
    # [FUNCTION DOC]
    # Function with a mapping argument d whose key x holds a list of two tensors (shapes 1 and 2)
    # and whose key y holds a single tensor (shape 3). Expects index paths that nest argument
    # position, key and list position. Returns nothing.
    return


if __name__ == '__main__':
  common.do_test(TestModule)
