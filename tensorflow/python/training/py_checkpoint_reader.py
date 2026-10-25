# [MODULE DOC]
# Extends the compiled checkpoint reader class with extra Python-level methods and provides the
# legacy factory function for creating a reader. At import time it attaches three methods
# (variable-to-dtype mapping, tensor-existence check, and tensor retrieval with error translation)
# onto the compiled reader class, mutating that shared class globally. It also exports the factory
# under the v1 training namespace. Native runtime errors, which arrive as generic runtime errors
# carrying only a message, are converted into typed TensorFlow errors by a message-matching helper.
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
"""Extending CheckpointReader for TensorFlow."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.util import compat
from tensorflow.python.util._pywrap_checkpoint_reader import CheckpointReader
from tensorflow.python.util.tf_export import tf_export


def error_translator(e):
  # [FUNCTION DOC]
  # HELPER: converts native checkpoint-reader runtime errors into typed TensorFlow errors for the
  # reader methods and the factory function in this module.
  # Always raises; never returns normally. Input e is the caught runtime error; only its text is
  # inspected. Mapping by message substring: "not found in checkpoint" or "failed to find any
  # matching files" gives a not-found error; "sliced checkpoints are not supported" or "data type
  # not supported" gives an unimplemented error; "failed to get matching files on" gives an
  # invalid-argument error; "unable to open table file" gives a data-loss error; "failed to find
  # the saved tensor slices" or "not convertible to numpy dtype" gives an internal error. Any other
  # message gives a generic operation error with the unknown code. The original message text is
  # preserved in the raised error. No side effects.
  """Translate the tensor_slice_reader.cc errors."""
  # TODO(b/143319754): Remove the RuntimeError casting logic once we resolve the
  # issue with throwing python exceptions from C++.
  error_message = str(e)
  if 'not found in checkpoint' in error_message or (
      'Failed to find any '
      'matching files for') in error_message:
    raise errors_impl.NotFoundError(None, None, error_message)
  elif 'Sliced checkpoints are not supported' in error_message or (
      'Data type '
      'not '
      'supported') in error_message:
    raise errors_impl.UnimplementedError(None, None, error_message)
  elif 'Failed to get matching files on' in error_message:
    raise errors_impl.InvalidArgumentError(None, None, error_message)
  elif 'Unable to open table file' in error_message:
    raise errors_impl.DataLossError(None, None, error_message)
  elif 'Failed to find the saved tensor slices' in error_message or (
      'not convertible to numpy dtype' in error_message):
    raise errors_impl.InternalError(None, None, error_message)
  else:
    raise errors_impl.OpError(None, None, error_message, errors_impl.UNKNOWN)


def get_variable_to_dtype_map(self):
  # [FUNCTION DOC]
  # Method patched onto the compiled reader class (self is the reader instance).
  # Returns a mapping from each variable name stored in the checkpoint to its TensorFlow data type
  # object, built by converting the native type enumeration values reported by the reader.
  # Takes no other inputs, raises nothing itself and has no side effects.
  return {
      name: dtypes.DType(type_enum)
      for name, type_enum in self._GetVariableToDataTypeMap().items()  # pylint: disable=protected-access
  }

CheckpointReader.get_variable_to_dtype_map = get_variable_to_dtype_map


def has_tensor(self, tensor_str):
  # [FUNCTION DOC]
  # Method patched onto the compiled reader class (self is the reader instance).
  # Input tensor_str is the tensor name as text; it is converted to bytes before the native lookup.
  # Returns true if the checkpoint contains a tensor with that name, otherwise false. Native errors
  # are not translated here. No side effects.
  return self._HasTensor(compat.as_bytes(tensor_str))  # pylint: disable=protected-access

CheckpointReader.has_tensor = has_tensor


def get_tensor(self, tensor_str):
  # [FUNCTION DOC]
  # Method patched onto the compiled reader class (self is the reader instance).
  # Input tensor_str is the tensor name as text (converted to bytes). Returns the tensor value
  # from the checkpoint as a numpy array. If the native reader raises a runtime error, it is passed
  # through the error translator, which raises a typed TensorFlow error (for example not-found when
  # the name is absent). No side effects.
  """Get the tensor from the Checkpoint object."""
  try:
    return CheckpointReader.CheckpointReader_GetTensor(
        self, compat.as_bytes(tensor_str))
  # TODO(b/143319754): Remove the RuntimeError casting logic once we resolve the
  # issue with throwing python exceptions from C++.
  except RuntimeError as e:
    error_translator(e)


CheckpointReader.get_tensor = get_tensor


# Disable invalid name to keep backwards compatibility with that function.
# It was previously exported from py_checkpoint_reader.i which did not conform
# to pylint checks.
# pylint: disable=invalid-name
@tf_export(v1=['train.NewCheckpointReader'])
def NewCheckpointReader(filepattern):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring calls the argument a filename and says it returns a "CheckPointReader"
  # (misspelled); the argument is actually a file prefix or pattern.
  # Factory for a checkpoint reader, exported as the v1 training API for backward compatibility.
  # Input filepattern is the checkpoint file prefix or pattern as text (converted to bytes).
  # Returns a new reader object. Native runtime errors raised while opening it (missing files,
  # unreadable table, unsupported sliced checkpoint) are translated into typed TensorFlow errors
  # such as not-found or data-loss.
  """A function that returns a CheckPointReader.

  Args:
    filepattern: The filename.

  Returns:
    A CheckpointReader object.
  """
  try:
    return CheckpointReader(compat.as_bytes(filepattern))
  # TODO(b/143319754): Remove the RuntimeError casting logic once we resolve the
  # issue with throwing python exceptions from C++.
  except RuntimeError as e:
    error_translator(e)
