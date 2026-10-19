# [MODULE DOC]
# Unit tests for the HashableElementSpec wrapper and the memoized uncompression-function builder
# in the data service operations module. Covers equality/hash semantics of the wrapper and the
# caching behavior (including ragged tensor specs written in equivalent forms). Runs through the
# standard TensorFlow test main entry point; no import-time side effects.
#
# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for HashableElementSpec and related functionality."""

from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test


class HashableElementSpecTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering HashableElementSpec (equality and hashing) and the cache of the
  # uncompression-function builder. No shared fixtures; each test builds its own specs.

  def testEqual(self):
    # [FUNCTION DOC]
    # Scenario: two wrappers around separately built but identical scalar 32-bit integer specs.
    # Asserts they compare equal and have equal hashes.
    spec1 = data_service_ops.HashableElementSpec(
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.int32)
    )
    spec2 = data_service_ops.HashableElementSpec(
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.int32)
    )
    self.assertEqual(spec1, spec2)
    self.assertEqual(hash(spec1), hash(spec2))

  def testNotEqual(self):
    # [FUNCTION DOC]
    # Scenario: wrappers around scalar specs with different dtypes (32-bit vs 64-bit integer).
    # Asserts they are unequal and their hashes differ.
    spec1 = data_service_ops.HashableElementSpec(
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.int32)
    )
    spec2 = data_service_ops.HashableElementSpec(
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.int64)
    )
    self.assertNotEqual(spec1, spec2)
    self.assertNotEqual(hash(spec1), hash(spec2))

  def testNotEqualOtherType(self):
    # [FUNCTION DOC]
    # Scenario: compare a wrapper with an unrelated integer. Asserts they are not equal.
    spec1 = data_service_ops.HashableElementSpec(
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.int32)
    )
    self.assertNotEqual(spec1, 123)

  def testGetUncompressFuncCache(self):
    # [FUNCTION DOC]
    # Scenario: clear the cache, then request the uncompression function twice for equal specs
    # (32-bit) and once for a different spec (64-bit). Asserts the first two calls return the
    # very same cached object and the third returns a different object.
    spec1 = tensor_spec.TensorSpec(shape=(), dtype=dtypes.int32)
    spec2 = tensor_spec.TensorSpec(shape=(), dtype=dtypes.int64)
    data_service_ops._get_uncompress_func.cache_clear()
    func1 = data_service_ops._get_uncompress_func(
        data_service_ops.HashableElementSpec(spec1)
    )
    func2 = data_service_ops._get_uncompress_func(
        data_service_ops.HashableElementSpec(spec1)
    )
    func3 = data_service_ops._get_uncompress_func(
        data_service_ops.HashableElementSpec(spec2)
    )
    self.assertIs(func1, func2)
    self.assertIsNot(func1, func3)

  def testRaggedTensorUncompressFuncCache(self):
    # [FUNCTION DOC]
    # Scenario: three ragged tensor specs: one with a shape given as a short list, one with the
    # same shape written as an explicit shape object of dimensions, and one with a different
    # value dtype. After clearing the cache, asserts the first two (equivalent) specs yield the
    # same cached function object while the differing spec yields a different one.
    spec1 = ragged_tensor.RaggedTensorSpec(
        [3, None], dtypes.int32, 1, dtypes.int64
    )
    spec1_long_format = ragged_tensor.RaggedTensorSpec(
        tensor_shape.TensorShape(
            [tensor_shape.Dimension(3), tensor_shape.Dimension(None)]
        ),
        dtypes.int32,
        1,
        dtypes.int64,
    )
    spec2 = ragged_tensor.RaggedTensorSpec(
        ([3, None]), dtypes.int64, 1, dtypes.int64
    )
    data_service_ops._get_uncompress_func.cache_clear()
    func1a = data_service_ops._get_uncompress_func(
        data_service_ops.HashableElementSpec(spec1)
    )
    func1b = data_service_ops._get_uncompress_func(
        data_service_ops.HashableElementSpec(spec1_long_format)
    )
    func2 = data_service_ops._get_uncompress_func(
        data_service_ops.HashableElementSpec(spec2)
    )
    self.assertIs(func1a, func1b)
    self.assertIsNot(func1a, func2)


if __name__ == "__main__":
  test.main()
