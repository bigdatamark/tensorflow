# [MODULE DOC]
# HELPER: import-aggregator module for the structured package.
# Exists only to import the structured modules that define exported symbols (the array ops
# dispatch implementations and the StructuredTensor class) so they get registered, without
# importing them from the package initializer, which would cause circular dependencies.
# Exposes no definitions of its own; the imports are the import-time side effect.
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
"""Import all modules in the `structured` package that define exported symbols.

We don't import these modules from structured/__init__.py, since we want to
avoid circular dependencies.
"""


# pylint: disable=unused-import
from tensorflow.python.ops.structured import structured_array_ops
from tensorflow.python.ops.structured import structured_tensor
