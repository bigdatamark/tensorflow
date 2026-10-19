# [MODULE DOC]
# HELPER: optional override module consumed by test_util to decide whether tests use the MLIR bridge.
# Provides a single function that reports the MLIR bridge as disabled. It is only effective when the
# build rule includes this module as a dependency; test_util then picks it up instead of its default.
# No import-time side effects.
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
"""Including this as a dependency will result in tests NOT using MLIR bridge.

This function is defined by default in test_util.py to None. The test_util then
attempts to import this module. If this file is made available through the BUILD
rule, then this function is overridden and will instead cause Tensorflow graphs
to be always NOT be compiled with MLIR bridge.
"""


def is_mlir_bridge_enabled():
  # [FUNCTION DOC]
  # HELPER: override hook read by test_util to configure test execution.
  # Takes no arguments and always returns the boolean false, meaning tests must not use the MLIR
  # bridge. No side effects.
  """Returns false if the MLIR bridge should be not be enabled for tests."""
  return False
