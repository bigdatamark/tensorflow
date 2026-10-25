# [MODULE DOC]
# Python wrappers for the experimental tensor map (hash map of tensors) operations, delegating
# to the generated map ops, which are also re-exported through a wildcard import. Also registers
# gradient functions for lookup, insert and erase. Import-time side effects: marks the
# empty-map op as not differentiable and registers three gradients.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Ops to manipulate hashmap of tensors."""

# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_map_ops
from tensorflow.python.ops.gen_map_ops import *

ops.NotDifferentiable("EmptyTensorMap")

def empty_tensor_map():
  # [FUNCTION DOC]
  # Creates and returns a handle to a new empty tensor map (variant tensor). Takes no inputs.
  return gen_map_ops.empty_tensor_map()

def tensor_map_size(input_handle):
  # [FUNCTION DOC]
  # Returns a scalar integer tensor with the number of entries in the map given by input_handle.
  return gen_map_ops.tensor_map_size(input_handle)

def tensor_map_insert(input_handle, key, value):
  # [FUNCTION DOC]
  # Returns a new map handle equal to the map input_handle with the entry from key to value
  # added (replacing an existing entry for that key). The input map is not modified.
  return gen_map_ops.tensor_map_insert(input_handle, key, value)

def tensor_map_lookup(input_handle, key, value_dtype):
  # [FUNCTION DOC]
  # Returns the tensor stored under key in the map input_handle, with data type value_dtype.
  # The op fails at run time if the key is absent.
  return gen_map_ops.tensor_map_lookup(input_handle, key, value_dtype)

def tensor_map_erase(input_handle, key, value_dtype):
  # [FUNCTION DOC]
  # Returns a new map handle equal to input_handle without the entry for key. The value_dtype
  # gives the data type of the stored values. The op fails at run time if the key is absent.
  return gen_map_ops.tensor_map_erase(input_handle, key, value_dtype)

def tensor_map_has_key(input_handle, key):
  # [FUNCTION DOC]
  # Returns a boolean scalar tensor telling whether key is present in the map input_handle.
  return gen_map_ops.tensor_map_has_key(input_handle, key)


def tensor_map_stack_keys(input_handle, key_dtype):
  # [FUNCTION DOC]
  # Returns a tensor stacking all keys of the map input_handle, with data type key_dtype.
  return gen_map_ops.tensor_map_stack_keys(input_handle, key_dtype)


@ops.RegisterGradient("TensorMapLookup")
def LookupGrad(op, dval):
  # [FUNCTION DOC]
  # HELPER: gradient registered for the map lookup op, used by automatic differentiation.
  # Builds a map gradient as an empty map holding the incoming value gradient dval under the
  # looked-up key. Inputs: op, the forward lookup op (inputs are the map and key); dval, the
  # gradient of the looked-up value. Returns the map gradient and nothing for the key.
  _, k = op.inputs
  map_grad = empty_tensor_map()
  map_grad = tensor_map_insert(map_grad, k, dval)
  key_grad = None
  return map_grad, key_grad

@ops.RegisterGradient("TensorMapInsert")
def InsertGrad(op, dmap):
  # [FUNCTION DOC]
  # HELPER: gradient registered for the map insert op, used by automatic differentiation.
  # If the incoming map gradient dmap contains the inserted key, the value gradient is the
  # entry for that key and the map gradient is dmap with that entry erased; otherwise the value
  # gradient is zeros shaped like the value and the map gradient is dmap unchanged. Returns the
  # map gradient, nothing for the key, and the value gradient.
  _, k, v = op.inputs
  key_grad = None
  (value_grad, map_grad) = cond.cond(
      tensor_map_has_key(dmap, k), lambda:
      (tensor_map_lookup(dmap, k, v.dtype), tensor_map_erase(dmap, k, v.dtype)),
      lambda: (array_ops.zeros_like(v), dmap))
  return map_grad, key_grad, value_grad

@ops.RegisterGradient("TensorMapErase")
def EraseGrad(op, dmap):
  # [FUNCTION DOC]
  # HELPER: gradient registered for the map erase op, used by automatic differentiation.
  # KNOWN ISSUE: op is unused, and the key's gradient is nothing, so only the map gradient
  # passes through; the erased entry gets no gradient.
  # Passes the incoming map gradient dmap through unchanged. Returns the map gradient and
  # nothing for the key.
  key_grad = None
  map_grad = dmap
  return map_grad, key_grad
