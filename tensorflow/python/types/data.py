# [MODULE DOC]
# Declares the abstract marker types for tf.data datasets (one for TensorFlow 2 and one for
# TensorFlow 1), used for type checks without importing the heavy dataset implementation. Both
# are exported under internal public API names at import time via the export decorator.
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
"""Dataset types."""

import abc

from tensorflow.python.util.tf_export import tf_export


@tf_export("__internal__.types.data.Dataset", v1=[])
class DatasetV2(abc.ABC):
  # [CLASS DOC]
  # Abstract marker base class standing for the TensorFlow 2 dataset type. Has no members;
  # concrete dataset classes are expected to register or derive from it. Exported as an
  # internal API type for TensorFlow 2 only.
  """Represents the TensorFlow 2 type `tf.data.Dataset`."""


@tf_export(v1=["__internal__.types.data.Dataset"])
class DatasetV1(DatasetV2, abc.ABC):
  # [CLASS DOC]
  # Abstract marker base class standing for the TensorFlow 1 dataset type; a subclass of the
  # TensorFlow 2 marker. Has no members. Exported as an internal API type for TensorFlow 1 only.
  """Represents the TensorFlow 1 type `tf.data.Dataset`."""
