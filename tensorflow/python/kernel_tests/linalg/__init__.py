# [PACKAGE DOC]
# Package marker for the kernel tests of the linear algebra operations (tf.linalg and the
# related linalg_ops kernels such as Cholesky, determinants, eigendecomposition and einsum).
# It contains only test modules, each exercising one family of kernels by comparing results with
# NumPy, checking gradients and shapes, and benchmarking some ops. The package exports nothing
# and has no import-time side effects.
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
"""Kernel tests for tf.linalg."""
