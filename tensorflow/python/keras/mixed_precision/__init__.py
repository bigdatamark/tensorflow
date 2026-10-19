# [PACKAGE DOC]
# Package for the legacy (tf.python.keras) mixed precision API. The package itself contains no
# code and re-exports nothing; it only carries a docstring pointing at the mixed precision guide.
# Its submodules provide the dtype Policy, the autocasting variable wrapper, the loss scale
# optimizer, loss scale serialization helpers, a GPU compatibility logger and test utilities.
# Importing the package has no side effects.
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
"""Keras mixed precision API.

See [the mixed precision guide](
  https://www.tensorflow.org/guide/keras/mixed_precision) to learn how to
use the API.
"""
