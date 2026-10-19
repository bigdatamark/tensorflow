# [PACKAGE DOC]
# AutoGraph utility package: small helpers that generated (converted) code or the AutoGraph
# runtime can call. Re-exports control_dependency_on_returns (context manager creating control
# dependencies on function return values), alias_tensors (wraps tensors in identity ops) and
# dynamic_list_append (append that works on Python lists, tensor lists and tensor arrays).
# Other submodules: ag_logging (verbosity-controlled logging), misc, context_managers and
# tensor_list. Import has no side effects beyond importing those submodules.
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
"""Utility module that contains APIs usable in the generated code."""

from tensorflow.python.autograph.utils.context_managers import control_dependency_on_returns
from tensorflow.python.autograph.utils.misc import alias_tensors
from tensorflow.python.autograph.utils.tensor_list import dynamic_list_append
