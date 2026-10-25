# [MODULE DOC]
# HELPER: public facade module for the signature-definition helpers implemented in the _impl module.
# Re-exports the public SignatureDef utilities from signature_def_utils_impl: building generic,
# classification, regression, prediction, supervised train and eval signatures, validity checking
# for serving, and creating/loading op-based signatures. It defines no logic of its own and has no
# import-time side effects beyond importing the implementation module (which registers API exports).
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
"""SignatureDef utility functions.

Utility functions for building and inspecting SignatureDef protos.
"""
# pylint: disable=unused-import
from tensorflow.python.saved_model.signature_def_utils_impl import build_signature_def
from tensorflow.python.saved_model.signature_def_utils_impl import classification_signature_def
from tensorflow.python.saved_model.signature_def_utils_impl import is_valid_signature
from tensorflow.python.saved_model.signature_def_utils_impl import load_op_from_signature_def
from tensorflow.python.saved_model.signature_def_utils_impl import op_signature_def
from tensorflow.python.saved_model.signature_def_utils_impl import predict_signature_def
from tensorflow.python.saved_model.signature_def_utils_impl import regression_signature_def
from tensorflow.python.saved_model.signature_def_utils_impl import supervised_eval_signature_def
from tensorflow.python.saved_model.signature_def_utils_impl import supervised_train_signature_def
# pylint: enable=unused-import
