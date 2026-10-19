# [MODULE DOC]
# FileCheck-driven test for importing a TF1 saved model that uses a function made with the legacy
# Defun decorator. The checks verify the exported function's argument order and index paths and
# that the call to the Defun-built function becomes a legacy call operation. Runs through the TF1
# common driver when executed as a script. No import-time side effects other than defining the
# Defun function object at import.
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

# RUN: %p/defun_export | FileCheck %s

# pylint: disable=missing-docstring,line-too-long
import tensorflow.compat.v1 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common_v1
from tensorflow.python.framework import function


@function.Defun(tf.float32, tf.float32)
def plus(a, b):
  # [FUNCTION DOC]
  # HELPER: Legacy graph function (two float inputs) that the test model calls.
  # Returns the sum of its two inputs. Decorated so that it is registered as a graph function
  # which is imported as a separate function and called through a legacy call operation.
  return a + b


def test_defun():
  # [FUNCTION DOC]
  # Model builder for the common TF1 driver (the name looks like a test but it is only passed to
  # the driver and not collected as a unit test). Creates two 3x1 float constants x (ones) and y
  # (twos), computes their sum with the Defun function, builds tensor infos and a signature
  # definition (inputs "x" and "y", output "z", method name "test_function"). Returns a three-item
  # result: a mapping from "key" to the signature, then nothing and nothing.
  x = tf.constant([[1.0], [1.0], [1.0]])
  y = tf.constant([[2.0], [2.0], [2.0]])

  # Verify that the function defined using function.Defun
  # has a corresponding tf.LegacyCall op.
  # CHECK:      func {{@[a-zA-Z_0-9]+}}(
  # CHECK-SAME: [[ARG0:%.*]]: tensor<3x1xf32> {tf_saved_model.index_path = ["y"]},
  # CHECK-SAME: [[ARG1:%.*]]: tensor<3x1xf32> {tf_saved_model.index_path = ["x"]}
  #
  # CHECK-NEXT: [[R0:%.*]] = "tf.LegacyCall"([[ARG1]], [[ARG0]])
  z = plus(x, y)

  tensor_info_x = tf.compat.v1.saved_model.utils.build_tensor_info(x)
  tensor_info_y = tf.compat.v1.saved_model.utils.build_tensor_info(y)
  tensor_info_z = tf.compat.v1.saved_model.utils.build_tensor_info(z)

  return {
      'key': (tf.compat.v1.saved_model.signature_def_utils.build_signature_def(
          inputs={
              'x': tensor_info_x,
              'y': tensor_info_y
          },
          outputs={'z': tensor_info_z},
          method_name='test_function'))
  }, None, None


if __name__ == '__main__':
  common_v1.set_tf_options()
  common_v1.do_test(test_defun)
