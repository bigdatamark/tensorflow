# [MODULE DOC]
# Python wrapper around the native (pybind) low-level TensorFlow runtime client. Re-exports the
# global eager context factory functions and defines a Runtime subclass that converts between
# function definition protobuf messages and the serialized strings the native API uses.
# Import-time side effect: importing the core wrapper module loads TensorFlow's dynamic
# libraries (the import is kept only for that effect).
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Low level TF runtime client."""

# TF oddity: this import loads TF-specific dynamic libraries.
from tensorflow.python import pywrap_tensorflow  # pylint:disable=g-bad-import-order,unused-import

from tensorflow.core.framework import function_pb2
from tensorflow.core.function.runtime_client import runtime_client_pybind

GlobalEagerContext = runtime_client_pybind.GlobalEagerContext
GlobalPythonEagerContext = runtime_client_pybind.GlobalPythonEagerContext


# TODO(mdan): Map without adapters once pybind11_protobuf available
class Runtime(runtime_client_pybind.Runtime):
  # [CLASS DOC]
  # Runtime client that extends the native Runtime binding with protobuf-typed helpers.
  # Constructed from an eager context object (see the context factory functions in this
  # module); inherits the native operations such as transforming a function by a named pass.

  def GetFunctionProto(self, name: str) -> function_pb2.FunctionDef:
    # [FUNCTION DOC]
    # Looks up the function with the given name in the runtime (via the native serialized
    # accessor) and returns it parsed as a function definition message. Errors from the native
    # call (such as an unknown function) propagate; parsing failures propagate too.
    return function_pb2.FunctionDef.FromString(
        self.GetFunctionProtoString(name))

  def CreateFunction(self, function_def: function_pb2.FunctionDef):
    # [FUNCTION DOC]
    # Serializes the given function definition message and registers it in the runtime through
    # the native string-based creator, replacing/defining the function there. Returns nothing.
    self.CreateFunctionFromString(function_def.SerializeToString())
