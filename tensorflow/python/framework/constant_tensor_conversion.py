# [MODULE DOC]
# HELPER: helper module that wires builtin Python values into the tensor conversion registry.
# Defines a conversion function that turns builtin values into constant tensors and, at import
# time, registers it with the tensor conversion registry: first as an internal conversion for the
# registry's set of constant-convertible types (priority 0), then for lists and tuples (priority
# 100) and finally for any object (priority 200, the lowest-precedence fallback). Importing this
# module therefore mutates global registry state. The constant module is imported lazily inside
# the conversion function so that this module stays cheap to import at framework load time.
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
"""Tensor conversion factory functions for builtins to constant Tensors."""

from tensorflow.python.framework import tensor_conversion_registry


# Factory function for tensor conversion for builtins. Import constant_op.py
# in-line so that it is only imported when it is needed. This file is imported
# at TF import time, thus that helps reduce import slowness.
def _constant_tensor_conversion_function(
    v, dtype=None, name=None, as_ref=False
):
  # [FUNCTION DOC]
  # HELPER: conversion callback registered with the tensor conversion registry for builtin values.
  # Converts an arbitrary value v (list, tuple, scalar or other convertible object) into a constant
  # tensor by delegating to the constant creation function. Optional dtype requests a data type and
  # optional name names the created operation. The as_ref flag is accepted only to satisfy the
  # registry callback signature and is ignored. Returns the resulting constant tensor. Errors from
  # the constant function (for example type or value errors for unconvertible input) propagate.
  # The constant module is imported lazily inside the call to keep package import time low.
  from tensorflow.python.framework import constant_op  # pylint: disable=g-import-not-at-top

  _ = as_ref
  return constant_op.constant(v, dtype=dtype, name=name)


# Register the conversion function for the "unconvertible" types
# as a conversion to a constant.
tensor_conversion_registry.register_tensor_conversion_function_internal(
    tensor_conversion_registry._CONSTANT_OP_CONVERTIBLES,  # pylint: disable=protected-access
    _constant_tensor_conversion_function,
    0,
)

tensor_conversion_registry.register_tensor_conversion_function(
    (list, tuple), _constant_tensor_conversion_function, 100
)
tensor_conversion_registry.register_tensor_conversion_function(
    object, _constant_tensor_conversion_function, 200
)
