# [MODULE DOC]
# HELPER: internal switch for TensorFlow 2 behavior.
# Thin wrappers over the native platform flag: enable, disable, and query whether v2 behavior is
# on. The query is exported under the internal public API namespace for v2 only.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Tools to help with the TensorFlow 2.0 transition.

This module is meant for TensorFlow internal implementation, not for users of
the TensorFlow library. For that see tf.compat instead.
"""

from tensorflow.python.platform import _pywrap_tf2
from tensorflow.python.util.tf_export import tf_export


def enable():
  # [FUNCTION DOC]
  # Turns on TensorFlow 2 behavior by setting the native global flag. No inputs, returns nothing.
  # Enables v2 behaviors.
  _pywrap_tf2.enable(True)


def disable():
  # [FUNCTION DOC]
  # Turns off TensorFlow 2 behavior by clearing the native global flag. No inputs, returns
  # nothing.
  # Disables v2 behaviors.
  _pywrap_tf2.enable(False)


@tf_export("__internal__.tf2.enabled", v1=[])
def enabled():
  # [FUNCTION DOC]
  # Reports whether TensorFlow 2 behavior is enabled. No inputs. Returns a boolean read from the
  # native global flag (which may consult the environment variable controlling v2 behavior).
  # Returns True iff TensorFlow 2.0 behavior should be enabled.
  return _pywrap_tf2.is_enabled()
