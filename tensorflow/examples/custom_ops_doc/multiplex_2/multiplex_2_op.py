# [MODULE DOC]
# Python wrapper for the second version of the multiplex custom op example (dense elementwise
# select). At import time it loads the compiled kernel shared library from the data files next to
# this module and exposes the raw generated op plus a public multiplex function with a docstring.
# Differs from version 1 in that the op inputs are named a and b and the tests use 64-bit integers.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""A wrapper for gen_multiplex_2_op.py.

This defines a public API and provides a docstring for the C++ Op defined by
multiplex_2_kernel.cc
"""

import tensorflow as tf
from tensorflow.python.platform import resource_loader

_multiplex_2_module = tf.load_op_library(
    resource_loader.get_path_to_datafile("multiplex_2_kernel.so"))

examples_multiplex_dense = _multiplex_2_module.examples_multiplex_dense


def multiplex(cond, a, b, name=None):
  # [FUNCTION DOC]
  # Public wrapper around the generated dense multiplex op. Parameters: cond, a boolean tensor;
  # a and b, tensors of identical type and shape (same shape as cond); name, optional op name
  # (defaults to nothing). Returns a tensor taking elements from a where cond is true and from b
  # elsewhere. Type or shape mismatches are reported by the op (invalid-argument error in eager
  # mode, type or value error in graph mode). No side effects other than adding an op.
  """Return elements chosen from `a` or `b` depending on `cond`.

  This is similar to `np.where` and `tf.where`, but simplified to only handle
  the case of dense tensors, no optional parameters, no broadcasting, etc..

  >>> multiplex([True, False, False, True], [1,2,3,4], [100,200,300,400])
  <tf.Tensor: shape=(4,), dtype=int32, numpy=array([  1, 200, 300,   4], ...)>

  Args:
    cond: tf.Tensor of type bool. Where True, yield `a`, otherwise yield `b`.
    a: tf.Tensor with the same type and shape as `b`.
    b: tf.Tensor with the same type and shape as `a`.
    name: An optional name for the op.

  Returns:
    A tf.Tensor with elements from `a` where `cond` is True, and elements
    from `b` elsewhere.
  """
  return examples_multiplex_dense(
      cond=cond, a=a, b=b, name=name)
