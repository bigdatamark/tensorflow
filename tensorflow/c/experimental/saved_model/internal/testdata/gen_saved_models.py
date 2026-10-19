# [MODULE DOC]
# HELPER: Test-data generator script for the C API saved model tests.
# Executable that writes two small saved models (one with an uninitialized variable, one with a
# while loop) into subdirectories of a base directory given as the single command-line argument.
# Run as a main program it enables TF2 behavior and runs the application main. Nothing is
# exported for import; there are no import-time side effects beyond imports.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Creates saved models used for testing.

This executable should be run with an argument pointing to the testdata/ folder
in this directory. It will re-generate the saved models that are used for
testing.
"""

import os
from absl import app

from tensorflow.python.compat import v2_compat

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.module import module
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops import while_loop
from tensorflow.python.saved_model import saved_model


def _gen_uninitialized_variable(base_dir):
  # [FUNCTION DOC]
  # HELPER: Builds one test fixture for the saved model generator.
  # base_dir: directory under which a subdirectory for this model is created. Defines a module
  # holding an initialized float variable, a sub-module, an uninitialized float variable that
  # shares its name with an uninitialized integer variable in the sub-module, and a compute
  # function adding a scalar float input to the initialized variable. Saves it to the
  # subdirectory named for the uninitialized-variable model. Returns nothing; writes files.
  """Generates a saved model with an uninitialized variable."""

  class SubModule(module.Module):
    # [CLASS DOC]
    # Inner module holding a single uninitialized 64-bit integer variable. Used as a child of the
    # outer module to test name collisions between uninitialized variables.
    """A module with an UninitializedVariable."""

    def __init__(self):
      # [FUNCTION DOC]
      # Constructor. Creates the uninitialized integer variable attribute. No parameters.
      self.uninitialized_variable = resource_variable_ops.UninitializedVariable(
          name="uninitialized_variable", dtype=dtypes.int64)

  class Module(module.Module):
    # [CLASS DOC]
    # Outer module saved by the generator: an initialized float variable, an uninitialized float
    # variable, a sub-module, and a traced compute function.
    """A module with an UninitializedVariable."""

    def __init__(self):
      # [FUNCTION DOC]
      # Constructor. Calls the parent constructor, then creates the sub-module, an initialized
      # float variable (value one) and an uninitialized float variable named like the one in the
      # sub-module but of a different type. No parameters.
      super(Module, self).__init__()
      self.sub_module = SubModule()
      self.initialized_variable = variables.Variable(
          1.0, name="initialized_variable")
      # An UninitializedVariable with the same name as the variable in the
      # SubModule, but with a different type.
      self.uninitialized_variable = resource_variable_ops.UninitializedVariable(
          name="uninitialized_variable", dtype=dtypes.float32)

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec((), dtypes.float32)])
    def compute(self, value):
      # [FUNCTION DOC]
      # Exported traced function with a fixed scalar float input signature. value: scalar float.
      # Returns the initialized variable plus value.
      return self.initialized_variable + value

  to_save = Module()
  saved_model.save(
      to_save, export_dir=os.path.join(base_dir, "UninitializedVariable"))


def _gen_simple_while_loop(base_dir):
  # [FUNCTION DOC]
  # HELPER: Builds one test fixture for the saved model generator.
  # base_dir: directory under which the subdirectory for the simple-while-loop model is created.
  # Defines a module whose compute function runs a while loop and saves it there. Returns
  # nothing; writes files.
  """Generates a saved model with a while loop."""

  class Module(module.Module):
    # [CLASS DOC]
    # Module exposing one traced function that contains a while loop.
    """A module with a while loop."""

    @def_function.function(
        input_signature=[tensor_spec.TensorSpec((), dtypes.float32)])
    def compute(self, value):
      # [FUNCTION DOC]
      # Exported traced function with a scalar float input signature. value: scalar float.
      # Runs a loop with an accumulator starting at zero and a counter starting at value; while the
      # counter is positive, adds the counter to the accumulator and decrements the counter by
      # one. Returns the accumulator.
      acc, _ = while_loop.while_loop(
          cond=lambda acc, i: i > 0,
          body=lambda acc, i: (acc + i, i - 1),
          loop_vars=(constant_op.constant(0.0), value))
      return acc

  to_save = Module()
  saved_model.save(
      to_save, export_dir=os.path.join(base_dir, "SimpleWhileLoop"))


def main(args):
  # [FUNCTION DOC]
  # Program entry point. args: command-line list whose first item is the program name and second
  # is the base directory. Raises a usage error if the list does not have exactly two items.
  # Generates both saved models under the base directory. Returns nothing.
  if len(args) != 2:
    raise app.UsageError("Expected one argument (base_dir).")
  _, base_dir = args
  _gen_uninitialized_variable(base_dir)
  _gen_simple_while_loop(base_dir)


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  app.run(main)
