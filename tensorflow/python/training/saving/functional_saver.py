# [MODULE DOC]
# HELPER: backward-compatibility shim. The functional saver (saving and restoring variables
# inside traced functions) moved to the checkpoint package; this module only forwards attribute
# lookups to the new module through a module-level attribute hook created by the deprecation
# utility, which logs a deprecation message (naming version 2.11) when a name is accessed through
# the old path. It defines nothing itself and is meant to be deleted once imports are updated.
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
"""Saves and restore variables inside traced @tf.functions."""

# TODO(kathywu): Delete this file after all imports have been moved to the path
# below.
from tensorflow.python.checkpoint import functional_saver
from tensorflow.python.util import deprecation

__getattr__ = deprecation.deprecate_moved_module(
    __name__, functional_saver, "2.11")
