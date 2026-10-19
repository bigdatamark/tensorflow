# [MODULE DOC]
# HELPER: Test-data generator script for the C++ saved model loader tests.
# Standalone program that creates one of several named test saved models (module-based version 2
# models with variables, a cyclic object graph, assets, a static hash table, or a version 1 model
# built from a session) in an export directory. Defines the module classes, a session helper and
# a table mapping names to constructors and format versions. Run as main it enables TF2 behavior.
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
"""Standalone utility to generate some test saved models."""

import os

from absl import app

from tensorflow.python.client import session as session_lib
from tensorflow.python.compat import v2_compat
from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_spec
from tensorflow.python.module import module
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import save_options
from tensorflow.python.saved_model import saved_model
from tensorflow.python.trackable import asset


class VarsAndArithmeticObjectGraph(module.Module):
  # [CLASS DOC]
  # Test module with float variables x and y, a child module holding a variable z and a constant
  # tensor, and a traced compute function. Instantiated without arguments and saved.
  """Three vars (one in a sub-module) and compute method."""

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor, no parameters. Creates variables x (1.0) and y (2.0), a child module with
    # variable z (3.0) and constant c (5.0). Does not call the parent constructor.
    self.x = variables.Variable(1.0, name="variable_x")
    self.y = variables.Variable(2.0, name="variable_y")
    self.child = module.Module()
    self.child.z = variables.Variable(3.0, name="child_variable")
    self.child.c = ops.convert_to_tensor(5.0)

  @def_function.function(input_signature=[
      tensor_spec.TensorSpec((), dtypes.float32),
      tensor_spec.TensorSpec((), dtypes.float32)
  ])
  def compute(self, a, b):
    # [FUNCTION DOC]
    # Traced function taking two scalar floats a and b. Returns (a + x) times (b + y) divided by
    # the child's z, plus the child's constant.
    return (a + self.x) * (b + self.y) / (self.child.z) + self.child.c


class ReferencesParent(module.Module):
  # [CLASS DOC]
  # HELPER: Child half of the cyclic test module.
  # Module that keeps a reference to its parent object and owns a variable; used to create a
  # cycle in the object graph.

  def __init__(self, parent):
    # [FUNCTION DOC]
    # Constructor. parent: the owning module object. Calls the parent constructor, stores the
    # reference and creates a variable named MyVariable with value 3.
    super(ReferencesParent, self).__init__()
    self.parent = parent
    self.my_variable = variables.Variable(3., name="MyVariable")


# Creates a cyclic object graph.
class CyclicModule(module.Module):
  # [CLASS DOC]
  # Test module whose child refers back to it, giving a cyclic object graph for saving/loading.

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor, no parameters. Calls the parent constructor and creates a child that holds a
    # reference to this module.
    super(CyclicModule, self).__init__()
    self.child = ReferencesParent(self)


class AssetModule(module.Module):
  # [CLASS DOC]
  # Test module that holds a text file asset (from the saved model test data directory) and a
  # function that reads the asset's file contents.

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor, no parameters. Creates an asset from the test asset text file located via the
    # test source directory helper. Does not call the parent constructor.
    self.asset = asset.Asset(
        test.test_src_dir_path("cc/saved_model/testdata/test_asset.txt"))

  @def_function.function(input_signature=[])
  def read_file(self):
    # [FUNCTION DOC]
    # Traced function with no inputs. Returns the contents of the asset file as a string tensor.
    return io_ops.read_file(self.asset)


class StaticHashTableModule(module.Module):
  # [CLASS DOC]
  # Test module with an asset file listing words, a static hash table initialized from it (word
  # to line number, default value negative one), and a lookup function.
  """A module with an Asset, StaticHashTable, and a lookup function."""

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor, no parameters. Creates the asset from the hash table text file and a static
    # hash table mapping each whole line (string) to its line number (64-bit integer), with
    # negative one for missing keys. Does not call the parent constructor.
    self.asset = asset.Asset(
        test.test_src_dir_path(
            "cc/saved_model/testdata/static_hashtable_asset.txt"))
    self.table = lookup_ops.StaticHashTable(
        lookup_ops.TextFileInitializer(self.asset, dtypes.string,
                                       lookup_ops.TextFileIndex.WHOLE_LINE,
                                       dtypes.int64,
                                       lookup_ops.TextFileIndex.LINE_NUMBER),
        -1)

  @def_function.function(
      input_signature=[tensor_spec.TensorSpec(shape=None, dtype=dtypes.string)])
  def lookup(self, word):
    # [FUNCTION DOC]
    # Traced function taking a string tensor of any shape. Returns the table values for word,
    # negative one for unknown words.
    return self.table.lookup(word)


def get_simple_session():
  # [FUNCTION DOC]
  # Builds the version 1 test model. Disables eager execution globally (side effect), creates a
  # session, adds a float variable with value one to the default graph and runs the global
  # variable initializer. Takes no parameters. Returns the session.
  ops.disable_eager_execution()
  sess = session_lib.Session()
  variables.Variable(1.)
  sess.run(variables.global_variables_initializer())
  return sess


MODULE_CTORS = {
    "VarsAndArithmeticObjectGraph": (VarsAndArithmeticObjectGraph, 2),
    "CyclicModule": (CyclicModule, 2),
    "AssetModule": (AssetModule, 2),
    "StaticHashTableModule": (StaticHashTableModule, 2),
    "SimpleV1Model": (get_simple_session, 1),
}


def main(args):
  # [FUNCTION DOC]
  # KNOWN ISSUE: The lookup of the module name uses a mapping get without a default, so an
  # unknown name yields a null value and the tuple unpacking raises a type error; the intended
  # "Expected ModuleName" message and return code 2 are never reached for unknown names.
  # Program entry point. args: list of program name, export directory, module name. If the count
  # is wrong, prints usage and allowed names and returns 1. Otherwise creates the export
  # directory (fails if it exists), builds the chosen module or session, and saves it: for
  # version 2 models via the saved model save function with debug info enabled; for version 1 via
  # a builder with the serve tag. Returns nothing on success.
  if len(args) != 3:
    print("Expected: {export_path} {ModuleName}")
    print("Allowed ModuleNames:", MODULE_CTORS.keys())
    return 1

  _, export_path, module_name = args
  module_ctor, version = MODULE_CTORS.get(module_name)
  if not module_ctor:
    print("Expected ModuleName to be one of:", MODULE_CTORS.keys())
    return 2
  os.makedirs(export_path)

  tf_module = module_ctor()
  if version == 2:
    options = save_options.SaveOptions(save_debug_info=True)
    saved_model.save(tf_module, export_path, options=options)
  else:
    builder = saved_model.builder.SavedModelBuilder(export_path)
    builder.add_meta_graph_and_variables(tf_module, ["serve"])
    builder.save()


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  app.run(main)
