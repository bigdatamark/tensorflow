# [MODULE DOC]
# HELPER: internal-only marker types used to break circular imports. Each class is an empty
# interface that framework modules subclass so that other modules can run isinstance checks
# without importing the heavy implementation. These types must not be exported or relied on by
# external code. No import-time side effects.
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
"""Types internal to TensorFlow.

These types should not be exported. External code should not rely on these.
"""


# TODO(mdan): Is this strictly needed? Only ops.py really uses it.
class NativeObject(object):
  # [CLASS DOC]
  # HELPER: empty marker base for types natively supported by TensorFlow operations (notably
  # tensors); lets ops code test for them without circular imports.
  """Types natively supported by various TF operations.

  The most notable example of NativeObject is Tensor.
  """


class TypeSpec(object):
  # [CLASS DOC]
  # HELPER: empty marker interface standing for the type specification class in the framework
  # type spec module, used for isinstance checks without circular imports.
  """Interface for internal isinstance checks to framework/type_spec.py.

  This helps to avoid circular dependencies.
  """


class TensorSpec(object):
  # [CLASS DOC]
  # HELPER: empty marker interface standing for the tensor specification class, used for
  # isinstance checks without circular imports.
  """Interface for internal isinstance checks to framework/tensor_spec.py.

  This helps to avoid circular dependencies.
  """


class IndexedSlices(object):
  # [CLASS DOC]
  # HELPER: empty marker interface standing for the indexed slices class, used for isinstance
  # checks without circular imports.
  """Interface for internal isinstance checks to framework/indexed_slices.py.

  This helps to avoid circular dependencies.
  """


class RaggedTensor(object):
  # [CLASS DOC]
  # HELPER: empty marker interface standing for the ragged tensor class, used for isinstance
  # checks without circular imports.
  """Interface for internal isinstance checks to ops/ragged/ragged_tensor.py.

  This helps to avoid circular dependencies.
  """


class RaggedTensorSpec(object):
  # [CLASS DOC]
  # HELPER: empty marker interface standing for the ragged tensor specification class, used for
  # isinstance checks without circular imports. (Its docstring names the ragged tensor module.)
  """Interface for internal isinstance checks to ops/ragged/ragged_tensor.py.

  This helps to avoid circular dependencies.
  """
