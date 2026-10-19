# [MODULE DOC]
# HELPER: internal helper module for the experimental unified-API gradient (tape) code.
# Holds the single process-wide gradient registry object (created from the native tape bindings
# at import time) and exposes an accessor for it. Import-time side effect: instantiates the
# global registry.
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
"""Global GradientRegistry."""

from tensorflow.python.framework.experimental import _tape

_GRADIENT_REGISTRY_GLOBAL = _tape.GradientRegistry()


def get_global_registry():
  # [FUNCTION DOC]
  # HELPER: accessor for the module's global gradient registry. Takes no inputs and returns the
  # shared registry object (the same one on every call). No side effects.
  return _GRADIENT_REGISTRY_GLOBAL
