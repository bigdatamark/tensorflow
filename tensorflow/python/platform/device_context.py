# [MODULE DOC]
# HELPER: stub module providing device-context hooks for the open-source build.
# Defines a single placeholder for obtaining the enclosing TPU context; in this build it does
# nothing. Presumably replaced by an internal implementation elsewhere. No import side effects.

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Helpers to get device context."""

def enclosing_tpu_context():
  # [FUNCTION DOC]
  # HELPER: hook used by code that needs to know whether it runs inside a TPU context.
  # KNOWN ISSUE: the name suggests it returns the enclosing TPU context, but this stub has no
  # body beyond a no-op and always returns the null value.
  # Takes no inputs, has no side effects, returns nothing/null.
  pass
