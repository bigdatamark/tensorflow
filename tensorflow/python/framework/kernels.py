# [MODULE DOC]
# Thin Python wrappers over the C API for querying the kernels registered in the running process.
# Exposes one function returning all registered kernels and one returning the kernels of a named op;
# both return a KernelList protocol buffer message. Not a helper module; no import-time side effects.
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
"""Functions for querying registered kernels."""

from tensorflow.core.framework import kernel_def_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.util import compat


def get_all_registered_kernels():
  # [FUNCTION DOC]
  # Queries the C API for every kernel registered in the process, copies the returned buffer into
  # bytes, and parses it into a new KernelList message, which is returned. Takes no arguments.
  # Raises whatever the underlying C API or message parsing raises on failure. No other side effects.
  """Returns a KernelList proto of all registered kernels.
  """
  buf = c_api.TF_GetAllRegisteredKernels()
  data = c_api.TF_GetBuffer(buf)
  kernel_list = kernel_def_pb2.KernelList()
  kernel_list.ParseFromString(compat.as_bytes(data))
  return kernel_list


def get_registered_kernels_for_op(name):
  # [FUNCTION DOC]
  # Queries the C API for kernels registered for the op called name (a string), reads the returned
  # buffer, and parses it into a new KernelList message, which is returned. An unknown op name is
  # not an error here; the result is simply expected to hold no kernels. Failures from the C API or
  # parsing propagate. No other side effects.
  """Returns a KernelList proto of registered kernels for a given op.

  Args:
    name: A string representing the name of the op whose kernels to retrieve.
  """
  buf = c_api.TF_GetRegisteredKernelsForOp(name)
  data = c_api.TF_GetBuffer(buf)
  kernel_list = kernel_def_pb2.KernelList()
  kernel_list.ParseFromString(compat.as_bytes(data))
  return kernel_list
