# [MODULE DOC]
# HELPER: tiny helper module used by tests and utilities taking part in the TFRT (TensorFlow
# runtime) migration.
# Exposes a single feature-switch function that reports whether the TFRT runtime should be
# enabled. In this version it is hard-wired to report that TFRT is disabled. The module has no
# import-time side effects and keeps no state.
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
"""Utilities for TFRT migration."""


def enabled():
  # [FUNCTION DOC]
  # HELPER: feature switch consulted during the TFRT migration.
  # Takes no inputs and returns a boolean. It always returns false, so TFRT is never enabled
  # through this switch. No exceptions and no side effects.
  """Returns true if TFRT should be enabled."""
  return False
