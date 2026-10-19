# [MODULE DOC]
# FileCheck-driven test for the TF saved model to MLIR import with debug information enabled.
# Defines a module with one exported addition function; the driver is called with debug info on,
# and the checks verify that the add and identity operations carry source locations, including
# one pointing at this file and a call-site location. No import-time side effects besides
# importing TensorFlow.
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

# RUN: %p/debug_info | FileCheck %s

# pylint: disable=missing-docstring,line-too-long
import tensorflow.compat.v2 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common


class TestModule(tf.Module):
  # [CLASS DOC]
  # Test module under conversion; has no state and exports one function of two scalar inputs.

  @tf.function(input_signature=[
      tf.TensorSpec([], tf.float32),
      tf.TensorSpec([], tf.float32)
  ])
  def some_function(self, x, y):
    # [FUNCTION DOC]
    # Exported function with two scalar float inputs (fixed signature). Returns their sum. The
    # check lines after the return describe the expected locations in the printed MLIR.
    return x + y
    # Basic check that the debug info file is being correctly saved and loaded.
    #
    # CHECK: "tf.AddV2"{{.*}}loc(#loc{{[0-9]+}})
    # CHECK: "tf.Identity"{{.*}}loc(#loc{{[0-9]+}})
    # CHECK: #loc{{[0-9]+}} = loc("{{.*}}debug_info.py":{{[0-9]+}}:{{[0-9]+}})
    # CHECK: #loc{{[0-9]+}} = loc(callsite(#loc{{[0-9]+}} at #loc{{[0-9]+}}))


if __name__ == '__main__':
  common.do_test(TestModule, show_debug_info=True)
