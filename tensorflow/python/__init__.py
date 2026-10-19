# [PACKAGE DOC]
# Root of the internal TensorFlow Python implementation package. Intentionally contains no imports
# or logic so that importing TensorFlow stays fast. The only thing it does is build the list of
# public names for star-imports: every name currently defined without a leading underscore, plus
# a small allowlist of version/build-info dunders (version, git version, compiler version, C++11
# ABI flag, monolithic build flag). Real functionality lives in submodules.
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
"""TensorFlow Python init file."""

# Do not add code to //third_party/tensorflow/python/__init__.py.
# This file is imported whenever TensorFlow is imported.
# Additional imports in this file could cause the internal
# import time of TensorFlow to increase by multiple seconds.


# Special dunders that we choose to export:
_exported_dunders = set([
    '__version__',
    '__git_version__',
    '__compiler_version__',
    '__cxx11_abi_flag__',
    '__monolithic_build__',
])

# Expose symbols minus dunders, unless they are allowlisted above.
# This is necessary to export our dunders.
__all__ = [s for s in dir() if s in _exported_dunders or not s.startswith('_')]
