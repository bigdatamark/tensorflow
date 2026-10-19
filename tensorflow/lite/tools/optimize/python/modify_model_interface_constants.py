# [MODULE DOC]
# HELPER: constants module for the modify_model_interface tool and library.
# Defines the supported integer interface types as a mapping from names (8-bit signed, 8-bit
# unsigned, 16-bit signed) to TensorFlow data types, the reverse mapping, collections of the
# names and the types, and the default (8-bit signed) name and type. No functions or side effects.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Constants for modify_model_interface."""

from tensorflow.python.framework import dtypes

STR_TO_TFLITE_TYPES = {
    'INT8': dtypes.int8,
    'UINT8': dtypes.uint8,
    'INT16': dtypes.int16,
}
TFLITE_TO_STR_TYPES = {v: k for k, v in STR_TO_TFLITE_TYPES.items()}

STR_TYPES = STR_TO_TFLITE_TYPES.keys()
TFLITE_TYPES = STR_TO_TFLITE_TYPES.values()

DEFAULT_STR_TYPE = 'INT8'
DEFAULT_TFLITE_TYPE = STR_TO_TFLITE_TYPES[DEFAULT_STR_TYPE]
