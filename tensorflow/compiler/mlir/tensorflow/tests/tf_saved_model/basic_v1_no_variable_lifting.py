# [MODULE DOC]
# FileCheck-driven test for importing a TF1 saved model into MLIR with variable lifting disabled.
# Same model as the basic TF1 test (constant column times a trainable variable, one signature),
# but the driver is called with lifting turned off; the check directives verify that a module is
# produced and that no global tensor is emitted. No import-time side effects besides importing
# TensorFlow.
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

# RUN: %p/basic_v1_no_variable_lifting | FileCheck %s

# pylint: disable=missing-docstring,line-too-long
import tensorflow.compat.v1 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common_v1

# Verify that the tf.versions attribute exists. It is difficult to enforce
# contents, since the version numbers change over time. The conversion logic
# itself is verified in the common graphdef converter, so here just assert
# it is being invoked.
# CHECK: module
# CHECK-NOT: tf_saved_model.global_tensor


def Test():
  # [FUNCTION DOC]
  # Model builder for the common TF1 driver. In the default graph creates a 3x1 constant of ones
  # (x), a trainable 1x3 variable y with random normal initializer, and their matrix product r.
  # Returns a three-item result: a mapping with the key "key" to a signature definition (input
  # "x", output "r", method name "some_function"), then nothing and nothing.
  x = tf.constant([[1.0], [1.0], [1.0]])
  y = tf.compat.v1.get_variable(
      name='y',
      shape=(1, 3),
      initializer=tf.random_normal_initializer(),
      trainable=True)
  r = tf.matmul(x, y)

  tensor_info_x = tf.compat.v1.saved_model.utils.build_tensor_info(x)
  tensor_info_r = tf.compat.v1.saved_model.utils.build_tensor_info(r)

  return {
      'key': (tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
          inputs={'x': tensor_info_x},
          outputs={'r': tensor_info_r},
          method_name='some_function'))
  }, None, None


if __name__ == '__main__':
  common_v1.set_tf_options()
  common_v1.do_test(Test, lift_variables=False)
