# [MODULE DOC]
# HELPER: Tiny script used to time how long importing TensorFlow takes (for example by running
# it under an interpreter import-time profiling option).
# The module only imports the main tensorflow package, unused on purpose, and defines nothing
# else. The import itself is the whole side effect.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Script to measure the internal tensorflow import time."""

import tensorflow as tf  # pylint: disable=unused-import
