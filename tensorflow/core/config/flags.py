# [MODULE DOC]
# Gives Python access to the global TensorFlow feature-flag registry implemented in C++ and
# exposed through a native binding module. Exposes a module-level variable holding the lazily
# created flags object and an accessor function. Importing it loads the native TensorFlow
# extension and the flags binding as side effects.
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
"""Code to enable/disable feature flags."""

# pylint: disable=invalid-import-order, g-bad-import-order, unused-import
from tensorflow.python import pywrap_tensorflow
from tensorflow.python import flags_pybind


FLAGS = None


def config():
  # [FUNCTION DOC]
  # Returns the process-wide feature-flags object. Takes no inputs. On first call it creates the
  # native flags object and caches it in the module-level variable; later calls return the same
  # cached object. Side effect: sets that module-level variable. Raises nothing itself.
  global FLAGS
  if FLAGS is None:
    FLAGS = flags_pybind.Flags()
  return FLAGS
