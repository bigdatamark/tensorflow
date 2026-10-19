# [MODULE DOC]
# HELPER: helper module for the Keras export code.
# KNOWN ISSUE: the lint marker at the end of the file says it must change together with a file of
# the very same path, so the intended mirror file is probably mis-named.
# Builds SavedModel signature definitions for supervised training and evaluation, for the
# Keras export code. Exposes two public builders (train and eval) that delegate to one private
# builder, using tensor-info conversion and the signature builder of the saved_model package and the
# method-name constants from the unexported constants module. No import-time side effects.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""SignatureDef utility functions implementation."""

from tensorflow.python.keras.saving.utils_v1 import unexported_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import utils_impl as utils


# LINT.IfChange
def supervised_train_signature_def(
    inputs, loss, predictions=None, metrics=None):
  # [FUNCTION DOC]
  # Builds a training signature definition. Inputs: inputs, mapping from names to tensors (must be
  # non-empty); loss, mapping of loss tensors; predictions and metrics, optional mappings of
  # prediction and metric tensors. Delegates to the private builder using the supervised-training
  # method name. Returns the signature definition; raises a value error if inputs is null or empty.
  return _supervised_signature_def(
      unexported_constants.SUPERVISED_TRAIN_METHOD_NAME, inputs, loss=loss,
      predictions=predictions, metrics=metrics)


def supervised_eval_signature_def(
    inputs, loss, predictions=None, metrics=None):
  # [FUNCTION DOC]
  # Builds an evaluation signature definition. Same inputs and errors as the training variant, but
  # uses the supervised-eval method name. Returns the signature definition.
  return _supervised_signature_def(
      unexported_constants.SUPERVISED_EVAL_METHOD_NAME, inputs, loss=loss,
      predictions=predictions, metrics=metrics)


def _supervised_signature_def(
    method_name, inputs, loss=None, predictions=None,
    metrics=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it raises a value error if "inputs or outputs" is none, but
  # outputs are never checked: loss, predictions and metrics may all be null and an empty output
  # set is accepted. It also says inputs only need to be non-null, yet empty inputs are rejected.
  # HELPER: shared implementation behind the train and eval signature builders.
  # Inputs: method_name, string; inputs, non-empty mapping from names to tensors; loss,
  # predictions, metrics, optional mappings of tensors. Converts every tensor to tensor info,
  # merges the three output mappings into one (later mappings overwrite equal names) and builds the
  # signature definition. Returns it; raises a value error when inputs is null or empty.
  """Creates a signature for training and eval data.

  This function produces signatures that describe the inputs and outputs
  of a supervised process, such as training or evaluation, that
  results in loss, metrics, and the like. Note that this function only requires
  inputs to be not None.

  Args:
    method_name: Method name of the SignatureDef as a string.
    inputs: dict of string to `Tensor`.
    loss: dict of string to `Tensor` representing computed loss.
    predictions: dict of string to `Tensor` representing the output predictions.
    metrics: dict of string to `Tensor` representing metric ops.

  Returns:
    A train- or eval-flavored signature_def.

  Raises:
    ValueError: If inputs or outputs is `None`.
  """
  if inputs is None or not inputs:
    raise ValueError('{} inputs cannot be None or empty.'.format(method_name))

  signature_inputs = {key: utils.build_tensor_info(tensor)
                      for key, tensor in inputs.items()}

  signature_outputs = {}
  for output_set in (loss, predictions, metrics):
    if output_set is not None:
      sig_out = {key: utils.build_tensor_info(tensor)
                 for key, tensor in output_set.items()}
      signature_outputs.update(sig_out)

  signature_def = signature_def_utils.build_signature_def(
      signature_inputs, signature_outputs, method_name)

  return signature_def
# LINT.ThenChange(//tensorflow/python/keras/saving/utils_v1/signature_def_utils.py)
