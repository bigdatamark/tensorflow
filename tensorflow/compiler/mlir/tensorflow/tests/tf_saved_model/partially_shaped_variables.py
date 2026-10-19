# [MODULE DOC]
# FileCheck test for importing a TF2 SavedModel with variables whose shapes are only partially
# known (fully unknown rank, and unknown dimension) into MLIR. Verifies the global tensors keep
# the unranked and dynamic shape types while their stored values have concrete shapes. No
# functions are exported. Uses the common test harness.
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

# RUN: %p/partially_shaped_variables | FileCheck %s

# pylint: disable=missing-docstring,line-too-long
import tensorflow.compat.v2 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common


class TestModule(tf.Module):
  # [CLASS DOC]
  # Test module holding two variables with partially specified shapes: v0 with completely
  # unknown shape and v1 with one unknown dimension. Exports no functions.

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor. Initializes the base module, then creates variable v0 (initial value a single
    # zero, shape declared as entirely unknown) and variable v1 (initial values zero and one, shape
    # declared as one dimension of unknown size). No parameters.
    super(TestModule, self).__init__()
    # CHECK: "tf_saved_model.global_tensor"() <{is_mutable, {{.*}} type = tensor<*xf32>, value = dense<0.000000e+00> : tensor<1xf32>}> {tf_saved_model.exported_names = ["v0"]} : () -> ()
    # CHECK: "tf_saved_model.global_tensor"() <{is_mutable, {{.*}} type = tensor<?xf32>, value = dense<[0.000000e+00, 1.000000e+00]> : tensor<2xf32>}> {tf_saved_model.exported_names = ["v1"]} : () -> ()
    self.v0 = tf.Variable([0.], shape=tf.TensorShape(None))
    self.v1 = tf.Variable([0., 1.], shape=[None])


if __name__ == '__main__':
  common.do_test(TestModule, exported_names=[])
