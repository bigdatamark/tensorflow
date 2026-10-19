# [MODULE DOC]
# Thin Python utilities over the native gradient tape machinery: a scope that tracks accessed
# trainable variables, a context manager that pauses all gradient recording, queries on whether
# recording is needed or possible, and functions that record an operation on all tapes, only
# backward tapes, or only forward accumulators. All work is delegated to the eager C extension.
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
"""Gradient record utilities."""

import contextlib

from tensorflow.python import pywrap_tfe


class VariableWatcher(object):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring sample compares the watched variables attribute to a list, but it
  # is a method (must be called) and returns a tuple, not a list.
  # Context manager that creates a native variable watcher on entry and removes it on exit. While
  # active, accesses to trainable variables are recorded; the watched variables can be queried
  # at any time. Holds a single slot with the native watcher handle (none before entering).
  """A scope that tracks all trainable variable accesses within it.

  This explicitly ignores variables that are not marked as trainable.

  Sample usage:

  var = tf.Variable(0.0)
  with VariableWatcher() as variable_watcher:
    var.assign_add(1.0)

  assert variable_watcher.watched_variables == [var]
  """

  __slots__ = ["_variable_watcher"]

  def __init__(self):
    # [FUNCTION DOC]
    # Initializes the watcher handle to nothing; the native watcher is created on entering the
    # scope. Takes no arguments, returns nothing.
    self._variable_watcher = None

  def __enter__(self):
    # [FUNCTION DOC]
    # Creates a new native variable watcher, stores it, and returns this object so it can be
    # bound by the with statement. Side effect: registers the watcher with the runtime.
    self._variable_watcher = pywrap_tfe.TFE_Py_VariableWatcherNew()
    return self

  def __exit__(self, typ, value, traceback):
    # [FUNCTION DOC]
    # Removes the native watcher from the runtime. Exception information arguments are ignored and
    # exceptions are not suppressed. Returns nothing. The stored handle is kept, so the watched
    # variables can still be queried afterwards.
    pywrap_tfe.TFE_Py_VariableWatcherRemove(self._variable_watcher)

  def watched_variables(self):
    # [FUNCTION DOC]
    # Returns the tuple of trainable variables accessed while the watcher was active, as reported
    # by the native layer. Requires the scope to have been entered at least once.
    """Returns a tuple of variables accessed under this scope."""
    return pywrap_tfe.TFE_Py_VariableWatcherWatchedVariables(
        self._variable_watcher)


@contextlib.contextmanager
def stop_recording():
  # [FUNCTION DOC]
  # Context manager that pauses all gradient recording (backward tapes and forward accumulators)
  # on the current thread inside the with block. If recording was already stopped on entry it does
  # nothing; otherwise it stops recording and, in a finally clause, restarts it on exit. Yields
  # nothing.
  """Stop all gradient recording (backprop and forwardprop)."""
  is_stopped = pywrap_tfe.TFE_Py_TapeSetIsStopped()
  try:
    if not is_stopped:
      pywrap_tfe.TFE_Py_TapeSetStopOnThread()
    yield
  finally:
    if not is_stopped:
      pywrap_tfe.TFE_Py_TapeSetRestartOnThread()


def should_record_backprop(tensors):
  # [FUNCTION DOC]
  # Asks the native layer whether any active gradient tape watches any of the given tensors
  # (typically op inputs). Forward accumulators are not considered. Returns a boolean.
  """Returns true if any tape in the stack watches any of these tensors.

  Only takes GradientTapes into account, not forward accumulators.

  Args:
    tensors: Tensors to check, typically inputs to an operation.

  Returns:
    Boolean, whether any tape watches any of `tensors`.
  """
  return pywrap_tfe.TFE_Py_TapeSetShouldRecordBackprop(tensors)


def record_operation(op_type, output_tensors, input_tensors, backward_function,
                     forward_function=None):
  # [FUNCTION DOC]
  # Records an executed operation on all active tapes (backward tapes and forward accumulators).
  # op_type: operation type name. output_tensors and input_tensors: lists of the operation's
  # outputs and inputs. backward_function: callback computing input gradients from output
  # gradients. forward_function: optional callback (default none) for forward-mode use. Returns
  # nothing; side effect is updating tape state.
  """Records the operation on all tapes in the stack."""
  pywrap_tfe.TFE_Py_TapeSetRecordOperation(op_type, output_tensors,
                                           input_tensors, backward_function,
                                           forward_function)


def record_operation_backprop_only(op_type, output_tensors, input_tensors,
                                   backward_function):
  # [FUNCTION DOC]
  # Records an executed operation only on backward-mode tapes (gradient tapes), not on forward
  # accumulators. Arguments: operation type name, output tensors list, input tensors list, and
  # the backward callback. Returns nothing.
  """Records the operation on all backward tapes in the stack."""
  pywrap_tfe.TFE_Py_TapeSetRecordOperationBackprop(op_type, output_tensors,
                                                   input_tensors,
                                                   backward_function)


def record_operation_forwardprop_only(op_type, output_tensors, input_tensors,
                                      backward_function,
                                      forwardprop_output_indices):
  # [FUNCTION DOC]
  # Records an executed operation only on forward accumulators. Arguments: operation type name,
  # output tensors list, input tensors list, the backward callback (automatically transposed to
  # produce output gradients from input gradients), and forwardprop_output_indices, which marks
  # outputs that already hold forward-mode derivatives (may be none or empty). Returns nothing.
  """Records the operation on all forward accumulators in the stack.

  Args:
    op_type: a string for the operation type, used in the backprop code
    output_tensors: a list of Python Tensor objects output by the operation
    input_tensors: a list of input Tensors to the recorded operation
    backward_function: the function to be called to, given the gradients of the
      output tensors, produce the gradients of the input tensors. This function
      is automatically transposed to produce output gradients given input
      gradients.
    forwardprop_output_indices: indicates any output_tensors which contain JVPs.
      Typically these will have come from TFE_Py_PackForwardGradients. May be
      None or an empty sequence if there are no JVP outputs from the operation.
  """
  pywrap_tfe.TFE_Py_TapeSetRecordOperationForwardprop(
      op_type, output_tensors, input_tensors, backward_function,
      forwardprop_output_indices)


def could_possibly_record():
  # [FUNCTION DOC]
  # Returns true if at least one tape or accumulator is active (the tape set is not empty).
  """Returns True if any tape is active."""
  return not pywrap_tfe.TFE_Py_TapeSetIsEmpty()
