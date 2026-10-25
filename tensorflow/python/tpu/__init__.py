# [PACKAGE DOC]
# The TensorFlow Python TPU (Tensor Processing Unit) package: holds the Python-side TPU support
# such as TPU context and embedding utilities, bfloat16 helpers, TPU datasets and checkpoint
# hooks. This init file re-exports nothing; submodules are imported explicitly by users (the api
# submodule lists modules that must be scanned for API export decorations).
# Import-time side effects: sets two process environment variables identifying the ML platform
# as TensorFlow and recording the TensorFlow version string, overwriting any existing values.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
# =============================================================================

"""Ops related to Tensor Processing Units."""

import os
from tensorflow.python.framework import versions

os.environ['TPU_ML_PLATFORM'] = 'Tensorflow'
os.environ['TPU_ML_PLATFORM_VERSION'] = versions.__version__
