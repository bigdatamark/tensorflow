# [MODULE DOC]
# Standalone command-line utility that writes a small checkpoint to a path, used to produce test
# data. Takes an export path and the name of a module kind (a table module holding a dense-hash
# lookup table, or a variable module holding two variables), builds it and writes a checkpoint.
# Not imported by library code. Running as main enables v2 behavior and starts the app runner.

# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Standalone utility to generate some test saved models."""

from absl import app

from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.compat import v2_compat
from tensorflow.python.framework import dtypes
from tensorflow.python.module import module
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables


class TableModule(module.Module):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring talks about three variables and a compute method, but the class
  # holds only a dense-hash lookup table and has no compute method.
  # Module holding one dense-hash lookup table (integer 64 keys and values) with two entries.
  """Three vars (one in a sub-module) and compute method."""

  def __init__(self):
    # [FUNCTION DOC]
    # Creates the lookup table named t1 (default value -1, empty key 0, deleted key -1, 32 initial
    # buckets) and inserts the pairs 1 to 1 and 2 to 4. Takes no arguments.
    default_value = -1
    empty_key = 0
    deleted_key = -1
    self.lookup_table = lookup_ops.DenseHashTable(
        dtypes.int64,
        dtypes.int64,
        default_value=default_value,
        empty_key=empty_key,
        deleted_key=deleted_key,
        name="t1",
        initial_num_buckets=32)
    self.lookup_table.insert(1, 1)
    self.lookup_table.insert(2, 4)


class VariableModule(module.Module):
  # [CLASS DOC]
  # Module holding two float variables, one with three elements and one with two.

  def __init__(self):
    # [FUNCTION DOC]
    # Creates the two variables (values 1,2,3 and 4,5) as attributes. Takes no arguments.
    self.v = variables.Variable([1., 2., 3.])
    self.w = variables.Variable([4., 5.])

MODULE_CTORS = {
    "TableModule": TableModule,
    "VariableModule": VariableModule,
}


def main(args):
  # [FUNCTION DOC]
  # Entry point. args is the command-line list: program name, export path, module name. If the
  # count is not three, prints usage and the allowed names and returns 1; if the name is unknown,
  # prints the allowed names and returns 2. Otherwise builds the chosen module, writes a
  # checkpoint of it to the export path (file side effect) and returns nothing.
  if len(args) != 3:
    print("Expected: {export_path} {ModuleName}")
    print("Allowed ModuleNames:", MODULE_CTORS.keys())
    return 1

  _, export_path, module_name = args
  module_ctor = MODULE_CTORS.get(module_name)
  if not module_ctor:
    print("Expected ModuleName to be one of:", MODULE_CTORS.keys())
    return 2

  tf_module = module_ctor()
  ckpt = checkpoint.Checkpoint(tf_module)
  ckpt.write(export_path)


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  app.run(main)
