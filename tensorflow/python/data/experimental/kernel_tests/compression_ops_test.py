# [MODULE DOC]
# Kernel tests for the experimental dataset compression ops (compress and uncompress of
# structured elements). Defines helper factories producing named sample elements (scalars,
# tuples, named tuples, unicode, sparse and ragged structures) and one test class covering
# round trips, dataset mapping, and error cases for mismatched dtype, shape and variant inputs.
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
"""Tests for compression ops."""
from collections import namedtuple
from absl.testing import parameterized

from tensorflow.python.data.experimental.ops import compression_ops
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.framework import combinations
from tensorflow.python.framework import errors
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test


def _test_objects():
  # [FUNCTION DOC]
  # HELPER: supplies parameter values for the compression tests.
  # Returns a list of named test objects covering int, string, tuple, nested tuple, named tuple,
  # unicode string, nested named tuples, mixed int/string tuple, a sparse tensor value and a
  # mapping containing sparse values and a tuple. Takes no inputs.

  Item = namedtuple("Item", "id name")

  return [
      combinations.NamedObject("int", 1),
      combinations.NamedObject("string", "dog"),
      combinations.NamedObject("tuple", (1, 1)),
      combinations.NamedObject("nested_tuple", ((1, 1), (2, 2))),
      combinations.NamedObject("named_tuple", Item(id=1, name="item1")),
      combinations.NamedObject("unicode", "アヒル"),
      combinations.NamedObject(
          "nested_named_tuple",
          (Item(id=1, name="item1"), Item(id=2, name="item2"))),
      combinations.NamedObject("int_string_tuple", (1, "dog")),
      combinations.NamedObject(
          "sparse",
          sparse_tensor.SparseTensorValue(
              indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])),
      combinations.NamedObject(
          "sparse_structured", {
              "a":
                  sparse_tensor.SparseTensorValue(
                      indices=[[0, 0], [1, 2]],
                      values=[1, 2],
                      dense_shape=[3, 4]),
              "b": (1, 2, "dog")
          })
  ]


def _test_v2_eager_only_objects():
  # [FUNCTION DOC]
  # HELPER: supplies parameter values for the compression tests that only work in eager mode.
  # Returns a list of named objects: a ragged tensor, and a mapping holding a sparse value and a
  # ragged tensor. Takes no inputs.
  return [
      combinations.NamedObject(
          "ragged",
          ragged_factory_ops.constant([[0, 1, 2, 3], [4, 5], [6, 7, 8], [9]])),
      combinations.NamedObject(
          "sparse_ragged_structured", {
              "sparse":
                  sparse_tensor.SparseTensorValue(
                      indices=[[0, 0], [1, 2]],
                      values=[1, 2],
                      dense_shape=[3, 4]),
              "ragged":
                  ragged_factory_ops.constant([[0, 1, 2, 3], [9]])
          })
  ]


class CompressionOpsTest(test_base.DatasetTestBase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the dataset compression ops: compress/uncompress round trips on many element
  # structures and the errors raised for mismatched inputs.

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(element=_test_objects())) +
      combinations.times(
          test_base.v2_eager_only_combinations(),
          combinations.combine(element=_test_v2_eager_only_objects())))
  def testCompression(self, element):
    # [FUNCTION DOC]
    # Test, parameterized by a named element. Unwraps the element, compresses it, uncompresses
    # it using the type spec derived from the original value, and asserts the evaluated result
    # equals the original element.
    element = element._obj

    compressed = compression_ops.compress(element)
    uncompressed = compression_ops.uncompress(
        compressed, structure.type_spec_from_value(element))
    self.assertValuesEqual(element, self.evaluate(uncompressed))

  @combinations.generate(
      combinations.times(test_base.default_test_combinations(),
                         combinations.combine(element=_test_objects())) +
      combinations.times(
          test_base.v2_eager_only_combinations(),
          combinations.combine(element=_test_v2_eager_only_objects())))
  def testDatasetCompression(self, element):
    # [FUNCTION DOC]
    # Test, parameterized by a named element. Builds a one-element dataset, maps it through
    # compress and then uncompress with the dataset's element spec, and asserts the dataset
    # produces the original element.
    element = element._obj

    dataset = dataset_ops.Dataset.from_tensors(element)
    element_spec = dataset.element_spec

    dataset = dataset.map(lambda *x: compression_ops.compress(x))
    dataset = dataset.map(lambda x: compression_ops.uncompress(x, element_spec))
    self.assertDatasetProduces(dataset, [element])

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testCompressionOutputDTypeMismatch(self):
    # [FUNCTION DOC]
    # Test: compresses a string, then uncompresses it with an integer type spec. Expects a
    # failed-precondition error mentioning that a tensor of type string was received.
    element = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    compressed = compression_ops.compress(element)
    with self.assertRaisesRegex(errors.FailedPreconditionError,
                                "but got a tensor of type string"):
      uncompressed = compression_ops.uncompress(
          compressed, structure.type_spec_from_value(0))
      self.evaluate(uncompressed)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testCompressionInputShapeMismatch(self):
    # [FUNCTION DOC]
    # Test: passes a list of two compressed values (wrong shape) to uncompress. Expects an
    # invalid-argument error in eager mode and a value error in graph mode.
    element = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    compressed = compression_ops.compress(element)
    compressed = [compressed, compressed]
    error = (
        errors.InvalidArgumentError
        if context.executing_eagerly() else ValueError)
    with self.assertRaises(error):
      uncompressed = compression_ops.uncompress(
          compressed, structure.type_spec_from_value(0))
      self.evaluate(uncompressed)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testCompressionInputDTypeMismatch(self):
    # [FUNCTION DOC]
    # Test: passes a plain list of integers (not a variant tensor) to uncompress and expects a
    # type error.
    uncompressed = list(range(10))
    with self.assertRaises(TypeError):
      uncompressed = compression_ops.uncompress(
          uncompressed, structure.type_spec_from_value(uncompressed))
      self.evaluate(uncompressed)

  @combinations.generate(
      combinations.times(test_base.default_test_combinations()))
  def testCompressionVariantMismatch(self):
    # [FUNCTION DOC]
    # Test: uses a dataset's variant tensor (not a compressed element) as input to uncompress
    # and expects an invalid-argument error.
    # Use a dataset as a variant.
    dataset = dataset_ops.Dataset.range(10)
    variant = dataset._variant_tensor
    with self.assertRaises(errors.InvalidArgumentError):
      uncompressed = compression_ops.uncompress(variant, dataset.element_spec)
      self.evaluate(uncompressed)

  # Only test eager mode since nested datasets are not allowed in graph mode.
  @combinations.generate(
      combinations.times(test_base.eager_only_combinations()))
  def testDatasetVariantMismatch(self):
    # [FUNCTION DOC]
    # Test (eager only): a dataset whose element is a nested dataset is mapped with uncompress,
    # which must raise a type error when the output is read.
    # Use a nested dataset as an example of a variant.
    dataset = dataset_ops.Dataset.from_tensors(dataset_ops.Dataset.range(10))
    with self.assertRaises(TypeError):
      dataset = dataset.map(
          lambda x: compression_ops.uncompress(x, dataset.element_spec))
      self.getDatasetOutput(dataset)


if __name__ == "__main__":
  test.main()
