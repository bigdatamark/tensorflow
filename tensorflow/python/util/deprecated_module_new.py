# [MODULE DOC]
# HELPER: test fixture module for the deprecation tests (deprecation_test).
# Replacement target for the deprecated test module (deprecated_module). It only defines one
# trivial function so tests can verify that attribute access on the deprecated module is
# forwarded here. It has no import-time side effects.
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
"""A module to replace deprecated_module.

For testing `deprecation.deprecate_moved_module`.
"""


def a():
  # [FUNCTION DOC]
  # HELPER: sentinel function for the moved-module deprecation test.
  # Takes no arguments, has no side effects and always returns the integer one. Tests call it
  # through the deprecated module to confirm the forwarding reaches this module.
  return 1
