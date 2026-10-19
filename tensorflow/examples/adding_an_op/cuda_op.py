# [MODULE DOC]
# Example Python wrapper for a custom CUDA kernel op. At import time, if TensorFlow was built
# with CUDA support, it loads the compiled op library file (cuda_op_kernel shared object) located
# in the data files directory and exposes the add-one op as a module-level callable named
# add_one (adds one to every element of an input tensor). Without CUDA the name is not defined.
# Part of the adding-an-op example; imports the full tensorflow package.
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
"""Cuda op Python library."""
import os.path

import tensorflow as tf

if tf.test.is_built_with_cuda():
  _cuda_op_module = tf.load_op_library(os.path.join(
      tf.compat.v1.resource_loader.get_data_files_path(), 'cuda_op_kernel.so'))
  add_one = _cuda_op_module.add_one
