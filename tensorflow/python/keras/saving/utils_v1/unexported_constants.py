# [MODULE DOC]
# HELPER: holds private string constants for SavedModel export: default signature keys for train
# and eval, the method names for supervised training and eval signatures, and the eval tag. They
# are kept here because they are not exported in the public saved_model constants modules, and each
# group mirrors a constant in the saved_model signature or tag constants. Used by the signature
# utilities and export utilities. No import-time side effects.
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
"""Signature constants for SavedModel save and restore operations.

These are the private constants that have not been exported.
"""

# LINT.IfChange
DEFAULT_TRAIN_SIGNATURE_DEF_KEY = "train"

DEFAULT_EVAL_SIGNATURE_DEF_KEY = "eval"

SUPERVISED_TRAIN_METHOD_NAME = "tensorflow/supervised/training"

SUPERVISED_EVAL_METHOD_NAME = "tensorflow/supervised/eval"
# LINT.ThenChange(//tensorflow/python/saved_model/signature_constants.py)

# LINT.IfChange
EVAL = "eval"
# LINT.ThenChange(//tensorflow/python/saved_model/tag_constants.py)
