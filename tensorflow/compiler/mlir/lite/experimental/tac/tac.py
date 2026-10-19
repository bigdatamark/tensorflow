# [MODULE DOC]
# Python entry point for target aware conversion (TAC) of TFLite models. Exposes one function
# that validates its arguments and delegates to the native wrapper extension module imported
# at load time.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Target aware conversion for TFLite model."""

from tensorflow.compiler.mlir.lite.experimental.tac.py_wrapper import _pywrap_tac_wrapper


def run_tac(model_path, targets, output_path):
  # [FUNCTION DOC]
  # Runs target aware conversion on a TFLite model file. model_path: path to the model file
  # (must be non-empty); targets: non-empty list of target name strings such as GPU or CPU;
  # output_path: path for the result (must be non-empty). Raises a value error if any of the
  # three is empty or null. Otherwise calls the native wrapper and returns its result (a
  # success indicator). Side effect: native code writes the converted model to output_path.
  """Run target aware conversion for the given tflite model file.

  Args:
    model_path: Path to the tflite model file.
    targets: A list of string of the desired targets. E.g., ['GPU', 'CPU'].
    output_path: The output path.

  Returns:
    Whether the optimization succeeded.

  Raises:
    ValueError:
      Invalid model_path.
      Targets are not specified.
      Invalid output_path.
  """
  if not model_path:
    raise ValueError("Invalid model_path.")

  if not targets:
    raise ValueError("Targets are not specified.")

  if not output_path:
    raise ValueError("Invalid output_path.")

  return _pywrap_tac_wrapper.run_tac(model_path, targets, output_path)
