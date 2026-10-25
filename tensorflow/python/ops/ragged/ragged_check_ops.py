# [MODULE DOC]
# Registers a RaggedTensor-specific implementation of the type assertion check
# (the public assert_type API) using the API dispatch mechanism. Importing this
# module has the side effect of registering that dispatch handler; the module
# exports only the single function assert_type. It is a small adapter module
# that lets the generic check ops accept ragged inputs by validating the flat
# values of the ragged tensor.
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
"""Asserts and Boolean Checks for RaggedTensors."""

from tensorflow.python.ops import check_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import dispatch


@dispatch.dispatch_for_api(check_ops.assert_type)
def assert_type(tensor: ragged_tensor.Ragged, tf_type, message=None, name=None):
  # [FUNCTION DOC]
  # Dispatch handler that makes the generic type assertion accept a ragged tensor. It is
  # registered for the public assert_type API and only runs when the first argument is
  # a ragged tensor. Inputs: tensor (a ragged tensor), tf_type (the expected data type),
  # message (optional text prepended to the error message) and name (optional op name).
  # It applies the standard assertion to the flat values of the ragged tensor, so the
  # check is on the element dtype. Returns whatever the underlying assertion returns
  # (an op, or nothing in eager mode). Raises a type error from the underlying assertion
  # if the dtype does not match the expected type.
  return check_ops.assert_type(tensor.flat_values, tf_type,
                               message=message, name=name)


