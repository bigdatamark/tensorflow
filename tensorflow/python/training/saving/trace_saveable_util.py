# [MODULE DOC]
# Helper module that traces the save and restore logic of legacy SaveableObject factories into
# concrete (graph) functions, used when exporting trackables to a SavedModel. Exposes one public
# function that traces every factory of a trackable and one private function that traces a single
# factory. No import-time side effects.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for tracing save and restore functions for SaveableObjects."""

from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import type_spec

from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.training.saving import saveable_object
from tensorflow.python.training.saving import saveable_object_util
from tensorflow.python.util import nest


def trace_save_restore_function_map(obj, factory_data_list):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring has a typo ("atttribute") and the returned dictionary omits
  # factories that are skipped (resource variables, non-callable factories, or factories whose
  # saveables validate to empty), which the docstring does not mention.
  # Traces save and restore concrete functions for each saveable factory. Inputs: obj, the
  # trackable that owns the factories; factory_data_list, a list of records each with a factory
  # and an attribute name. Factories are skipped when obj or the factory is a resource variable
  # or the factory is not callable, and when tracing yields no save function. Returns a mapping
  # from attribute name to a pair of (concrete save function, concrete restore function).
  # Exceptions from tracing propagate. Side effect: traces functions (may build graphs).
  """Traces all save and restore functions in the provided factory list.

  Args:
    obj: `Trackable` object.
    factory_data_list: List of `_CheckpointFactoryData`.

  Returns:
    Dict mapping atttribute names to tuples of concrete save/restore functions.
  """
  saveable_fns = {}

  for factory_data in factory_data_list:
    saveable_factory = factory_data.factory
    attribute_name = factory_data.name

    # If object revives as a resource (or TPU/Mirrored) variable,
    # there is no need to trace the save and restore functions.
    if (resource_variable_ops.is_resource_variable(obj) or
        resource_variable_ops.is_resource_variable(saveable_factory) or
        not callable(saveable_factory)):
      continue

    concrete_save, concrete_restore = (
        _trace_save_restore_functions(saveable_factory, obj))
    if not concrete_save:
      continue
    saveable_fns[attribute_name] = (concrete_save, concrete_restore)
  return saveable_fns


def _trace_save_restore_functions(saveable_factory, obj):
  # [FUNCTION DOC]
  # HELPER: per-factory tracing step for trace_save_restore_function_map.
  # If the factory was produced for an already-restored saveable (a partial carrying stored
  # save and restore functions), returns those two stored functions directly. Otherwise traces
  # a save function taking one scalar string checkpoint key, which calls the factory with that
  # key, normalizes a single saveable to a list, records the saveables in a shared list, and
  # returns a list of mappings with name, tensor and slice spec for every save spec. The
  # saveables are then validated for SavedModel use against obj; if none remain, returns a pair
  # of null values. Otherwise a restore function is traced whose input signature is the type
  # specs of all spec tensors; it repacks the flat restored tensors into per-saveable groups,
  # calls each saveable's restore, and returns a dummy constant 1. Returns the pair of concrete
  # save and restore functions. Validation errors propagate.
  """Traces save and restore functions."""
  if saveable_object_util.is_factory_for_restored_saveable_object(
      saveable_factory):
    return (
        saveable_factory.keywords["save_function"],
        saveable_factory.keywords["restore_function"],
    )

  saveables = []  # Store the saveables in a data structure accessible to both
  # the save and restore functions.

  @def_function.function(
      input_signature=[tensor_spec.TensorSpec([], dtypes.string)]
  )
  def save_fn(checkpoint_key):
    # [FUNCTION DOC]
    # HELPER: traced save function. Input: a scalar string tensor, the checkpoint key. Calls the
    # factory with that key, wraps a lone saveable in a list, stores the saveables into the
    # enclosing shared list (side effect), and returns a list with one mapping per save spec
    # holding its name, tensor and slice spec.
    maybe_saveable = saveable_factory(name=checkpoint_key)
    if isinstance(maybe_saveable, saveable_object.SaveableObject):
      maybe_saveable = [maybe_saveable]
    saveables[:] = maybe_saveable

    # Return list of all SaveSpecs created by the factory.
    ret = []
    for saveable in saveables:
      for spec in saveable.specs:
        ret.append({"name": spec.name, "tensor": spec.tensor,
                    "slice_spec": spec.slice_spec})
    return ret

  concrete_save = save_fn.get_concrete_function()

  # The SaveableObjects are produced when `save_fn` is traced.
  saveables = saveable_object_util.validate_saveables_for_saved_model(
      saveables, obj)
  if not saveables:
    return None, None

  # Use the SaveSpecs to define the input signature of the restore function.
  restored_type_specs = []
  tensor_structure = []
  for saveable in saveables:
    saveable_tensor_structure = []
    tensor_structure.append(saveable_tensor_structure)
    for spec in saveable.specs:
      restored_type_specs.append(type_spec.type_spec_from_value(spec.tensor))
      saveable_tensor_structure.append(spec.name)

  @def_function.function(input_signature=restored_type_specs)
  def restore_fn(*restored_tensors):
    # [FUNCTION DOC]
    # HELPER: traced restore function. Input: the flat restored tensors, one per save spec.
    # KNOWN ISSUE: the loop variable reuses the name of the function argument, shadowing it;
    # harmless here since the argument is no longer needed after repacking.
    # Repacks the flat tensors into the nested per-saveable structure, calls each saveable's
    # restore with its group of tensors (shapes unspecified), and returns the integer 1 as a
    # dummy output. Side effect: restores variables or resources owned by the saveables.
    structured_restored_tensors = nest.pack_sequence_as(
        tensor_structure, restored_tensors)
    for saveable, restored_tensors in zip(saveables,
                                          structured_restored_tensors):
      saveable.restore(restored_tensors, restored_shapes=None)
    return 1  # Return dummy tensor

  concrete_restore = restore_fn.get_concrete_function()
  return concrete_save, concrete_restore
