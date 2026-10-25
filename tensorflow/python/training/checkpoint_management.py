# [MODULE DOC]
# HELPER: backward-compatibility shim that forwards the old training checkpoint_management path.
# Deprecated alias module. The real implementation now lives in the checkpoint package's
# checkpoint_management module. This file only installs a module-level attribute-lookup hook,
# created by the deprecation utility, that resolves any attribute requested from this module
# on the new module and emits a deprecation warning naming release 2.9 as the move version.
# It defines no functions or classes of its own. Import-time side effects: imports the new
# module and the deprecation utility, and sets the module-level attribute hook.
# A TODO in the file says it should be deleted once all imports use the new location.
#
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

# pylint: disable=invalid-name
"""Save and restore variables."""


# TODO(kathywu): Delete this file after all imports have been moved to the path
# below.
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.util import deprecation

__getattr__ = deprecation.deprecate_moved_module(
    __name__, checkpoint_management, "2.9")
