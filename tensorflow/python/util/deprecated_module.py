# [MODULE DOC]
# HELPER: test fixture module for the deprecation tests (deprecation_test).
# A deliberately "deprecated" module used to test the moved-module deprecation helper. At import
# time it imports its replacement module (deprecated_module_new) and installs a module-level
# attribute lookup hook created by the moved-module deprecation factory, passing this module's
# own name, the new module and the version string "2.9". After that, accessing any attribute that
# is not defined here is forwarded to the new module and logs a deprecation warning that points
# to the new location. The module itself defines no other public names.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""A deprecated module.

For testing `deprecation.deprecate_moved_module`.
"""

from tensorflow.python.util import deprecated_module_new
from tensorflow.python.util import deprecation

__getattr__ = deprecation.deprecate_moved_module(
    __name__, deprecated_module_new, "2.9")
