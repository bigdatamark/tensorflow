# [MODULE DOC]
# HELPER: thin loader module for the compiled op library used by the adding-an-op tutorial tests.
# Python wrapper for version 3 of the tutorial ZeroOut op library. At import time it loads the
# compiled shared library for the third kernel version (found via the TensorFlow resource loader
# data path) and re-exports the zero_out op as a module attribute.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""ZeroOut op Python library."""
import os.path

import tensorflow as tf

_zero_out_module = tf.load_op_library(
    os.path.join(tf.compat.v1.resource_loader.get_data_files_path(),
                 'zero_out_op_kernel_3.so'))
zero_out = _zero_out_module.zero_out
