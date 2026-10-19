# [MODULE DOC]
# FileCheck-driven test for the TF saved model to MLIR import of a module whose object graph has a
# cycle (a child module holds a reference back to its parent). The check verifies the variable in
# the child is exported once under the path child.my_variable and conversion does not loop
# forever. Runs through the TF2 common driver when executed as a script. No import-time side
# effects besides importing TensorFlow.
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

# RUN: %p/cyclic_object_graph | FileCheck %s

# pylint: disable=missing-docstring,line-too-long
import tensorflow.compat.v2 as tf
from tensorflow.compiler.mlir.tensorflow.tests.tf_saved_model import common


class ReferencesParent(tf.Module):
  # [CLASS DOC]
  # Child module that stores a reference to its parent (creating the cycle) and owns a variable
  # with initial value 3.

  def __init__(self, parent):
    # [FUNCTION DOC]
    # Input: parent (the module that owns this child). Calls the parent constructor, stores the
    # parent reference and creates the variable my_variable (3.0). Returns nothing.
    super(ReferencesParent, self).__init__()
    self.parent = parent
    # CHECK: tf_saved_model.global_tensor
    # CHECK-SAME: tf_saved_model.exported_names = ["child.my_variable"]
    self.my_variable = tf.Variable(3.)


# Creates a cyclic object graph.
class TestModule(tf.Module):
  # [CLASS DOC]
  # Root test module. Owns one child module that points back at it, forming a cyclic object graph.

  def __init__(self):
    # [FUNCTION DOC]
    # Calls the parent constructor and creates the child, passing itself as the child's parent.
    # Takes no arguments and returns nothing.
    super(TestModule, self).__init__()
    self.child = ReferencesParent(self)


if __name__ == '__main__':
  common.do_test(TestModule)
