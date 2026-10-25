# [PACKAGE DOC]
# Package for the parallel-for (pfor) machinery: vectorized loop conversion, a for_loop helper
# and Jacobian/batch-Jacobian computation, implemented in the submodules (control_flow_ops
# holds the public entry points; pfor, gradients and helpers hold the converter and the
# gradient utilities). This initializer is intentionally empty apart from its docstring: it
# re-exports nothing and has no import-time side effects, so users import the submodules
# directly.
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
"""Ops for pfor, for_loop, jacobian."""
