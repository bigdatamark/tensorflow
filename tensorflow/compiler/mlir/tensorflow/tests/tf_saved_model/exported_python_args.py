# [MODULE DOC]
# Negative FileCheck test for the TensorFlow SavedModel to MLIR importer. It builds a module whose
# traced function has a Python default argument that is a mapping (unsupported structured value) and
# checks that the import fails with an error naming the function and the index path of the argument.
# The expected output is run through a negated command, so the script is supposed to fail.
# Defines no public API; run via the test harness and the common helper module.
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

# RUN: (! %p/exported_python_args 2>&1) | FileCheck %s

# pylint: disable=missing-docstring,line-too-long,dangerous-default-value
import tensorflow.compat.v2 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common


class TestModule(tf.Module):
  # [CLASS DOC]
  # Test module that is exported as a SavedModel. It has no state. It exposes one function with a
  # scalar float input signature which calls a second function that has an unsupported default
  # argument value (a mapping), which the importer is expected to reject.

  @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
  def some_function(self, x):
    # [FUNCTION DOC]
    # Exported entry point taking a scalar float tensor x. Simply forwards x to the callee method and
    # returns its result (the same tensor). Its tracing pulls the callee into the saved model.
    return self.callee(x)

  # CHECK: While importing SavedModel function 'callee': in input signature:
  # CHECK-SAME: Unhandled structured value kind {{.*}} at index path: <value>.1.foo
  @tf.function
  def callee(self, x, n={'foo': 42}):
    # [FUNCTION DOC]
    # Traced helper function with a tensor argument x and an optional argument n whose default is a
    # mapping from a string key to an integer. The mapping default cannot be represented in the
    # input signature, so the SavedModel import fails (the checked error). Returns x unchanged; n is
    # unused.
    return x


if __name__ == '__main__':
  common.do_test(TestModule)
