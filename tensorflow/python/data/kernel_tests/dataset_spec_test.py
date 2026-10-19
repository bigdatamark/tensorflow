# [MODULE DOC]
# Tests for the dataset specification type of the tf.data API (the type describing a dataset's
# element spec and trace information). Covers use as a function input signature, access to the
# inner element spec, equality/hash of trace types, and the subtype and most specific common
# supertype relations, including dictionary element specs. No import-time side effects; running
# as a script starts the TensorFlow test runner.
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
"""Tests for `tf.data.DatasetSpec`."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.platform import test


class DatasetSpecTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the dataset spec type: signatures, element spec access, trace type equality and
  # subtype relations. No shared fixtures.

  @combinations.generate(test_base.default_test_combinations())
  def testInputSignature(self):
    # [FUNCTION DOC]
    # Scenario: builds a batched dataset of ten 32-bit integers (batch size 5) and defines a
    # graph function whose input signature is a dataset spec with an unknown-length 32-bit integer
    # tensor element and a scalar-shaped second argument. The function body does nothing. Calls it
    # with the dataset and expects no error (the dataset matches the signature). Returns nothing.
    dataset = dataset_ops.Dataset.from_tensor_slices(
        np.arange(10).astype(np.int32)).batch(5)

    @def_function.function(input_signature=[
        dataset_ops.DatasetSpec(
            tensor_spec.TensorSpec(
                shape=(None,), dtype=dtypes.int32, name=None),
            tensor_shape.TensorShape([]))
    ])
    def fn(_):
      # [FUNCTION DOC]
      # Nested no-op graph function used only to exercise the input signature check; takes one
      # ignored argument (a dataset) and returns nothing.
      pass

    fn(dataset)

  @combinations.generate(test_base.default_test_combinations())
  def testDatasetSpecInnerSpec(self):
    # [FUNCTION DOC]
    # Scenario: wraps a scalar 32-bit integer tensor spec in a dataset spec and asserts that the
    # element spec property returns that same inner spec. Returns nothing.
    inner_spec = tensor_spec.TensorSpec(shape=(), dtype=dtypes.int32)
    ds_spec = dataset_ops.DatasetSpec(inner_spec)
    self.assertEqual(ds_spec.element_spec, inner_spec)

  @combinations.generate(test_base.default_test_combinations())
  def testDatasetSpecTraceType(self):
    # [FUNCTION DOC]
    # Scenario: two dataset specs with identical scalar integer element spec and the same second
    # argument are asserted equal, to have equal hashes, and to be subtypes of each other. A third
    # spec differing only in the second argument is asserted unequal and not a subtype in either
    # direction. Returns nothing.
    trace_type_1 = dataset_ops.DatasetSpec(
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.int32),
        [5])
    trace_type_2 = dataset_ops.DatasetSpec(
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.int32),
        [5])

    self.assertEqual(trace_type_1, trace_type_2)
    self.assertEqual(hash(trace_type_1), hash(trace_type_2))
    self.assertTrue(trace_type_1.is_subtype_of(trace_type_2))
    self.assertTrue(trace_type_2.is_subtype_of(trace_type_1))

    trace_type_3 = dataset_ops.DatasetSpec(
        tensor_spec.TensorSpec(shape=(), dtype=dtypes.int32),
        [6])
    self.assertNotEqual(trace_type_1, trace_type_3)
    self.assertFalse(trace_type_1.is_subtype_of(trace_type_3))
    self.assertFalse(trace_type_3.is_subtype_of(trace_type_1))

  @combinations.generate(test_base.default_test_combinations())
  def testDatasetSpecHierarchical(self):
    # [FUNCTION DOC]
    # Scenario: four dataset specs with varying fully/partially unknown element shapes and second
    # arguments. Asserts a spec is a subtype of itself, that the three more specific specs are
    # subtypes of the fully general one but not vice versa, and that the most specific common
    # supertype of a spec with an empty list is itself, and with other specs is the fully general
    # spec. Returns nothing.
    spec_1 = dataset_ops.DatasetSpec(
        tensor_spec.TensorSpec(shape=(1, None), dtype=dtypes.int32),
        [5, None, 2])
    spec_2 = dataset_ops.DatasetSpec(
        tensor_spec.TensorSpec(shape=(None, None), dtype=dtypes.int32),
        [None, None, None])
    spec_3 = dataset_ops.DatasetSpec(
        tensor_spec.TensorSpec(shape=(1, 2), dtype=dtypes.int32),
        [5, 3, 2])
    spec_4 = dataset_ops.DatasetSpec(
        tensor_spec.TensorSpec(shape=(None, 2), dtype=dtypes.int32),
        [None, 1, None])

    self.assertTrue(spec_1.is_subtype_of(spec_1))

    self.assertTrue(spec_1.is_subtype_of(spec_2))
    self.assertTrue(spec_3.is_subtype_of(spec_2))
    self.assertTrue(spec_4.is_subtype_of(spec_2))

    self.assertFalse(spec_2.is_subtype_of(spec_1))
    self.assertFalse(spec_2.is_subtype_of(spec_3))
    self.assertFalse(spec_2.is_subtype_of(spec_4))

    self.assertEqual(spec_1.most_specific_common_supertype([]), spec_1)
    self.assertEqual(spec_1.most_specific_common_supertype([spec_4]), spec_2)
    self.assertEqual(
        spec_1.most_specific_common_supertype([spec_3, spec_4]), spec_2)
    self.assertEqual(
        spec_1.most_specific_common_supertype([spec_2, spec_3, spec_4]), spec_2)

  # TODO(b/220385675): element_spec should always be a TypeSpec.
  @combinations.generate(test_base.default_test_combinations())
  def testDatasetSpecHierarchicalDict(self):
    # [FUNCTION DOC]
    # Scenario: dataset specs whose element spec is a mapping from names to tensor specs (one
    # mapping maps to nothing). Asserts subtype relations within the same key (specific is a
    # subtype of general, not reverse), no subtype relation across different keys, reflexivity for
    # the null-valued spec, and that the most specific common supertype of that spec with an empty
    # list or itself is itself. Returns nothing.
    spec_1 = dataset_ops.DatasetSpec(
        {"a": tensor_spec.TensorSpec(shape=(1, None), dtype=dtypes.int32)},
        [])
    spec_2 = dataset_ops.DatasetSpec(
        {"a": tensor_spec.TensorSpec(shape=(None, None), dtype=dtypes.int32)},
        [])
    spec_3 = dataset_ops.DatasetSpec(
        {"b": tensor_spec.TensorSpec(shape=(1, None), dtype=dtypes.int32)},
        [])
    spec_4 = dataset_ops.DatasetSpec({"b": None}, [])

    self.assertTrue(spec_1.is_subtype_of(spec_1))
    self.assertTrue(spec_1.is_subtype_of(spec_2))
    self.assertFalse(spec_2.is_subtype_of(spec_1))

    self.assertFalse(spec_1.is_subtype_of(spec_3))
    self.assertFalse(spec_3.is_subtype_of(spec_1))
    self.assertFalse(spec_2.is_subtype_of(spec_3))
    self.assertFalse(spec_3.is_subtype_of(spec_2))

    self.assertTrue(spec_4.is_subtype_of(spec_4))
    self.assertEqual(spec_4.most_specific_common_supertype([]), spec_4)
    self.assertEqual(spec_4.most_specific_common_supertype([spec_4]), spec_4)


if __name__ == "__main__":
  test.main()
