# [MODULE DOC]
# Test module checking that a user-defined wrapper around a variable, built only on the Trackable
# API, still works with checkpointing and saved-model export/load. A similar wrapper is used by
# Keras variables, so this guards that use case. Defines the wrapper class and one test case
# class; no import-time side effects beyond definitions. Run as a test binary.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests serialization of Variable wrapper class using the Trackable API.

A similar implementation is used in Keras Variable, and this test exists to
ensure the use case continues to work.
"""

import os

from tensorflow.python.checkpoint import checkpoint
from tensorflow.python.eager import test
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.trackable import autotrackable
from tensorflow.python.trackable import base


class VariableWrapper(base.Trackable):
  # [CLASS DOC]
  # HELPER: test fixture for VariableWrapperTest.
  # Minimal trackable object that wraps a real variable in an attribute named value and forwards
  # checkpoint serialization, restoration, saved-model export and object-proto writing to it. The
  # class flag marking it as acting like a resource variable lets the saved-model machinery treat
  # it as one. Constructed with an initial value.
  _should_act_as_resource_variable = True

  def __init__(self, value):
    # [FUNCTION DOC]
    # Constructor: creates the wrapped variable from value (any value accepted by the variable
    # constructor, such as a number) and stores it in the value attribute. Returns nothing.
    self.value = variables.Variable(value)

  @property
  def _shared_name(self):
    # [FUNCTION DOC]
    # HELPER: forwarding glue for the saved-model machinery.
    # Property returning the shared name of the wrapped variable.
    return self.value._shared_name

  def _serialize_to_tensors(self):
    # [FUNCTION DOC]
    # HELPER: forwarding glue for checkpoint saving.
    # Returns whatever the wrapped variable returns for serialization: a mapping from checkpoint
    # keys to tensors. No side effects.
    return self.value._serialize_to_tensors()

  def _restore_from_tensors(self, restored_tensors):
    # [FUNCTION DOC]
    # HELPER: forwarding glue for checkpoint restoring.
    # Delegates to the wrapped variable, passing restored_tensors (mapping from checkpoint keys
    # to restored tensors). Returns the wrapped variable's result (the restore operation);
    # restoring assigns the value into the wrapped variable.
    return self.value._restore_from_tensors(restored_tensors)

  def _export_to_saved_model_graph(
      self, object_map, tensor_map, options, **kwargs
  ):
    # [FUNCTION DOC]
    # HELPER: forwarding glue for saved-model export.
    # Asks the wrapped variable to export itself into the copy graph, passing object_map (mapping
    # from original objects to their exported copies, mutated), tensor_map (mapping from tensors
    # to copies), options (save options) and any extra keyword arguments. Afterwards registers in
    # object_map a new wrapper around the exported copy of the variable, keyed by this wrapper.
    # Returns the list of resources produced by the wrapped variable's export.
    resource_list = self.value._export_to_saved_model_graph(  # pylint:disable=protected-access
        object_map, tensor_map, options, **kwargs
    )
    object_map[self] = VariableWrapper(object_map[self.value])
    return resource_list

  def _write_object_proto(self, proto, options):
    # [FUNCTION DOC]
    # HELPER: forwarding glue for saved-model export.
    # Delegates to the wrapped variable to fill the saved-object description proto (mutated) using
    # the given save options. Returns the wrapped variable's result.
    return self.value._write_object_proto(proto, options)


class VariableWrapperTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the variable wrapper fixture with checkpoint save/restore and saved-model
  # save/load. Uses the per-test temporary directory; no shared fixtures.

  def test_checkpoint(self):
    # [FUNCTION DOC]
    # Test: wraps the number five, saves a checkpoint of it into the temp directory, changes the
    # wrapped value to one hundred, restores the checkpoint, and asserts the value is five again.
    v = VariableWrapper(5.0)
    root = autotrackable.AutoTrackable()
    root.v = v

    save_prefix = os.path.join(self.get_temp_dir(), "checkpoint")
    ckpt = checkpoint.Checkpoint(v=v)
    save_path = ckpt.save(save_prefix)

    v.value.assign(100)
    ckpt.restore(save_path)
    self.assertEqual(5, v.value.numpy())

  def test_export(self):
    # [FUNCTION DOC]
    # Test: wraps fifteen in a trackable root object, saves it as a saved model to the temp
    # directory, loads it back, and asserts the loaded attribute is a resource variable whose
    # value equals fifteen.
    v = VariableWrapper(15.0)
    root = autotrackable.AutoTrackable()
    root.v = v

    save_dir = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(root, save_dir)
    loaded = load.load(save_dir)
    self.assertTrue(resource_variable_ops.is_resource_variable(loaded.v))
    self.assertEqual(15, loaded.v.numpy())


if __name__ == "__main__":
  test.main()
