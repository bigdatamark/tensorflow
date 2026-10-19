# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
# [PACKAGE DOC]
# Package for controlling the TensorFlow XLA JIT compiler from Python. It re-exports two
# submodules: jit (a scope manager that marks operations for JIT compilation) and xla (helpers for
# explicitly compiling computations with XLA). Importing the package imports both submodules,
# which registers their public API names; the re-exports exist only for that purpose.
"""A module for controlling the Tensorflow/XLA JIT compiler."""

# pylint: disable=unused-import
from tensorflow.python.compiler.xla import jit
from tensorflow.python.compiler.xla import xla
# pylint: enable=unused-import
