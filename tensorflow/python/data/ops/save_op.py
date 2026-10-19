# [MODULE DOC]
# Implementation of dataset saving: a function that writes a dataset to disk (optionally with
# checkpointing in eager mode), a dataset class that performs the save as it is iterated, and a
# helper preparing the save op parameters and writing the element spec file. Used by the main
# dataset module for the public save method. No import-time side effects beyond imports.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Implementation of SaveDataset in Python."""
import os

from tensorflow.python.checkpoint import checkpoint as checkpoint_lib
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import structured_function
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_experimental_dataset_ops as ged_ops
from tensorflow.python.platform import gfile
# TODO(b/238903802): Use TypeSpec serialization methods directly.
from tensorflow.python.saved_model import nested_structure_coder


def _save(input_dataset,
          path,
          compression=None,
          shard_func=None,
          checkpoint_args=None):
  # [FUNCTION DOC]
  # HELPER: implementation behind the public dataset save method.
  # Inputs: input_dataset, path (directory to write), compression (optional string), shard_func
  # (optional callback mapping an element to a shard index), checkpoint_args (optional mapping of
  # arguments for a checkpoint manager; mutated by adding a checkpoint entry).
  # In eager mode with checkpoint arguments: builds a save dataset and iterator, raises a value
  # error if the arguments already contain a checkpoint entry, restores the latest checkpoint,
  # then consumes the iterator fully, incrementing the step counter if provided and saving
  # checkpoints at the manager interval. Returns nothing in that case. Otherwise prepares the
  # attributes and returns the result of the save dataset op. Side effects: writes files.
  """Implements the save function and checkpoint functionality."""
  if context.executing_eagerly() and checkpoint_args:
    save_dataset = _SaveDataset(input_dataset, path, shard_func, compression)
    save_iterator = iter(save_dataset)

    if "checkpoint" in checkpoint_args:
      raise ValueError(
          "'Invalid `checkpoint_args`. `checkpoint_args` are not allowed "
          "to include 'checkpoint'."
      )
    checkpoint = checkpoint_lib.Checkpoint(iterator=save_iterator)
    checkpoint_args["checkpoint"] = checkpoint
    manager = checkpoint_management.CheckpointManager(**checkpoint_args)
    checkpoint.restore(manager.latest_checkpoint)

    for _ in enumerate(save_iterator):
      if "step_counter" in checkpoint_args:
        checkpoint_args["step_counter"].assign_add(delta=1)
      manager.save(check_interval=True)
  else:
    dataset, shard_func, use_shard_func, path = set_save_dataset_attributes(
        input_dataset, shard_func, path)
    return ged_ops.save_dataset(
        dataset._variant_tensor,   # pylint: disable=protected-access
        path=path,
        shard_func_other_args=shard_func.captured_inputs,
        compression=compression,
        shard_func=shard_func,
        use_shard_func=use_shard_func)


class _SaveDataset(dataset_ops.UnaryDataset):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring (which also has a stray extra quote) says it loads a previously
  # saved dataset, but this dataset writes its input to disk while being iterated.
  # Unary dataset that saves its input elements to a path as they are iterated, used for the
  # checkpointable eager save. Holds the element spec and the shard function.
  """"A dataset that loads previously saved dataset."""

  def __init__(self, dataset, path, shard_func, compression):
    # [FUNCTION DOC]
    # Inputs: dataset (input), path (output directory), shard_func (optional shard callback),
    # compression (optional string). Prepares save attributes (writes the spec file, creates the
    # directory), creates the second-version save op with flat output types and shapes, and calls
    # the parent constructor.
    self._element_spec = dataset.element_spec
    self._shard_func = shard_func
    dataset, shard_func, use_shard_func, path = set_save_dataset_attributes(
        dataset, shard_func, path)
    variant_tensor = ged_ops.save_dataset_v2(
        dataset._variant_tensor,  # pylint: disable=protected-access
        path=path,
        shard_func_other_args=shard_func.captured_inputs,
        shard_func=shard_func,
        use_shard_func=use_shard_func,
        compression=compression,
        output_types=structure.get_flat_tensor_types(dataset.element_spec),
        output_shapes=structure.get_flat_tensor_shapes(dataset.element_spec),
    )
    super().__init__(dataset, variant_tensor)

  def _functions(self):
    # [FUNCTION DOC]
    # Returns a list holding the shard function, used for function tracking.
    return [self._shard_func]

  @property
  def element_spec(self):
    # [FUNCTION DOC]
    # Property returning the element spec of the input dataset (unchanged).
    return self._element_spec


def set_save_dataset_attributes(dataset, shard_func, path):
  # [FUNCTION DOC]
  # HELPER: prepares parameters for the two save ops.
  # Inputs: dataset, shard_func (optional callback; nothing means a dummy that is flagged unused),
  # path (output directory string). Wraps the shard function with the dataset's element spec,
  # encodes the element spec and writes it to the spec file in the directory (creating the
  # directory), converts the path to a string tensor, adds the shard function to the default
  # graph and applies debug options to the dataset.
  # Returns a tuple of the dataset, the shard function, a flag saying whether it is used, and the
  # path tensor. Side effects: file system writes and graph mutation.
  """Sets parameters for SaveDatasetOp and SaveDatasetV2Op."""
  if shard_func is None:
    use_shard_func = False
    shard_func = lambda *x: None  # a dummy function that will not be used
  else:
    use_shard_func = True
  wrapped_func = structured_function.StructuredFunctionWrapper(
      shard_func,
      "save()",
      input_structure=dataset.element_spec,
      add_to_graph=False)
  encoded = nested_structure_coder.encode_structure(dataset.element_spec)
  gfile.MakeDirs(path)
  with gfile.GFile(os.path.join(path, dataset_ops.DATASET_SPEC_FILENAME),
                   "wb") as f:
    f.write(encoded.SerializeToString())
  path = ops.convert_to_tensor(path, dtype=dtypes.string, name="path")
  shard_func = wrapped_func.function
  shard_func.add_to_graph(ops.get_default_graph())
  # pylint: disable=protected-access
  dataset._apply_debug_options()
  return dataset, shard_func, use_shard_func, path
