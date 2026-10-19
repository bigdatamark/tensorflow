# [MODULE DOC]
# FileCheck-driven test for the TF saved model to MLIR import (basic case). Defines a module with
# one variable (42.0), one constant (43.0) and one exported function adding both to a scalar input.
# When run as a script it hands the module to the common test driver, which converts the saved
# model to MLIR and prints it; the embedded check directives verify the version attributes, a
# global tensor for the constant and for the mutable variable, and the exported function's
# argument and result annotations. No import-time side effects besides importing TensorFlow.
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

# RUN: %p/basic | FileCheck %s

# pylint: disable=missing-docstring,line-too-long
import tensorflow.compat.v2 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common


# Verify that the tf.versions attribute exists. It is difficult to enforce
# contents, since the version numbers change over time. The conversion logic
# itself is verified in the common graphdef converter, so here just assert
# it is being invoked.
# CHECK: module
# CHECK-SAME: tf.versions
# CHECK-SAME: bad_consumers
# CHECK-SAME: min_consumer
# CHECK-SAME: producer


class TestModule(tf.Module):
  # [CLASS DOC]
  # Test module under conversion. Holds a trainable variable with value 42 and a constant tensor
  # with value 43, and exports one function. Instantiated by the common driver.

  def __init__(self):
    # [FUNCTION DOC]
    # Calls the parent constructor, then creates the variable v42 (42.0) and constant c43 (43.0).
    # Takes no arguments and returns nothing.
    super(TestModule, self).__init__()
    self.v42 = tf.Variable(42.0)
    self.c43 = tf.constant(43.0)

  # During serialization, the constants are given internal (non-user-accessible, non-semantically-load-bearing) exported names.
  # CHECK: "tf_saved_model.global_tensor"() <{sym_name = "[[CONST:[a-zA-Z_0-9.]+]]", type = tensor<f32>, value = dense<4.300000e+01> : tensor<f32>}> {tf_saved_model.exported_names = [{{.*}}]} : () -> ()

  # CHECK: "tf_saved_model.global_tensor"() <{is_mutable, sym_name = "[[VAR:[a-zA-Z_0-9]+]]", type = tensor<f32>, value = dense<4.200000e+01> : tensor<f32>}> {tf_saved_model.exported_names = ["v42"]} : () -> ()
  # CHECK:      func {{@[a-zA-Z_0-9]+}}(
  # CHECK-SAME:   %arg0: tensor<f32> {tf._user_specified_name = "x", tf_saved_model.index_path = [0]},
  # CHECK-SAME:   %arg1: tensor<!tf_type.resource<tensor<f32>>>
  # CHECK-SAME:   %arg2: tensor<!tf_type.resource<tensor<f32>>>
  # CHECK-SAME:   tensor<f32> {tf_saved_model.index_path = []})
  # CHECK-SAME: attributes {{.*}} tf_saved_model.exported_names = ["some_function"]
  @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
  def some_function(self, x):
    # [FUNCTION DOC]
    # Exported function with a fixed scalar float input signature. Returns x plus the variable
    # plus the constant. The check lines above describe the expected MLIR: one argument for x plus
    # two resource arguments (variable and constant binding) and one scalar float result.
    return x + self.v42 + self.c43


if __name__ == '__main__':
  common.do_test(TestModule)
