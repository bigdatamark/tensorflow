# [MODULE DOC]
# HELPER: backward-compatibility shim. The checkpoint options code moved to the checkpoint
# package; this module only forwards attribute lookups to the new module through a module-level
# attribute hook created by the deprecation utility, which logs a deprecation message (naming
# version 2.11) when a name is accessed through the old path. It defines nothing itself and is
# meant to be deleted once all imports use the new location.
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
"""Options for saving Checkpoints."""

# TODO(kathywu): Delete this file after all imports have been moved to the path
# below.
from tensorflow.python.checkpoint import checkpoint_options
from tensorflow.python.util import deprecation

__getattr__ = deprecation.deprecate_moved_module(
    __name__, checkpoint_options, "2.11")
