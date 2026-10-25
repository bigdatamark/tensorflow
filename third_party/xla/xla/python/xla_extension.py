# [MODULE DOC]
# HELPER: Compatibility shim that keeps the old XLA extension module name importable.
# KNOWN ISSUE: The docstring says this is an XLA client in Python, but the module contains no client
# code of its own; it only re-exports every public name from the JAX library's private native module.
# It therefore fails to import when the jax package is not installed.
# Importing this module performs a wildcard import of everything public from the jaxlib native
# extension module and exposes those names (client, device, buffer and executable types etc.) as
# its own attributes. It defines no functions or classes and has no other import-time side effects
# beyond those of importing the JAX native module.
# Copyright 2025 The OpenXLA Authors.
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
"""An XLA client in Python."""

from jax.jaxlib._jax import *  # pylint: disable=wildcard-import
