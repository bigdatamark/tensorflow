# [MODULE DOC]
# HELPER: thin helper module that gives the TFLite converter access to the native toco binding.
# Imports the TensorFlow native wrapper first (required ordering) and then the native toco API
# extension module, and exposes a single function that forwards serialized conversion requests to
# the native converter. No registrations or flags are defined at import time beyond those native
# imports.
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
"""Wraps toco interface with python lazy loader."""
# We need to import pywrap_tensorflow prior to the toco wrapper.
# pylint: disable=invalid-import-order,g-bad-import-order
from tensorflow.python import pywrap_tensorflow  # pylint: disable=unused-import
from tensorflow.python import _pywrap_toco_api


def wrapped_toco_convert(
    model_flags_str,
    toco_flags_str,
    input_data_str,
):
  # [FUNCTION DOC]
  # HELPER: glue between the Python TFLite converter and the native toco conversion entry point.
  # Calls the native conversion routine with the three serialized inputs and with the extended
  # return option turned off, and returns whatever the native call returns (the converted model
  # as serialized bytes). Parameters: model_flags_str is the serialized model flags message,
  # toco_flags_str is the serialized converter flags message, input_data_str is the serialized
  # input graph definition. All are required. Errors from the native layer propagate unchanged.
  """Wraps TocoConvert with lazy loader."""
  return _pywrap_toco_api.TocoConvert(
      model_flags_str,
      toco_flags_str,
      input_data_str,
      False,  # extended_return
  )
