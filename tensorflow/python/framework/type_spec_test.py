# [MODULE DOC]
# Tests for the TypeSpec base class and its batchable subclass. Defines helper value types and
# matching test spec classes (two tensors, two composites, nest of tensors), registers them in the
# spec registry and as value converters at import time, then defines two test case classes covering
# equality, subtyping, joins, compatibility, casting, tensor-list encoding, full types, registry
# errors and serialization. Several registry tests expect the main-program module name in messages.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for the TypeSpec base class."""

import collections

from absl.testing import parameterized

import numpy as np

from tensorflow.core.framework import full_type_pb2
from tensorflow.core.function import trace_type
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework import type_spec_registry
from tensorflow.python.framework.type_utils import fulltypes_for_flat_tensors
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import googletest
from tensorflow.python.util import nest
from tensorflow.python.util.compat import collections_abc


class TwoTensors:
  # [CLASS DOC]
  # HELPER: test value type used by the type spec tests.
  # Simple value holding two tensors (x and y) and a color string standing in for extra type
  # metadata. Described by the two-tensors spec class below.
  """A simple value type to test TypeSpec.

  Contains two tensors (x, y) and a string (color).  The color value is a
  stand-in for any extra type metadata we might need to store.
  """

  def __init__(self, x, y, color="red"):
    # [FUNCTION DOC]
    # HELPER: constructor of the test value type.
    # Inputs: x and y, anything convertible to a tensor (both are converted); color, a string
    # defaulting to red. Raises an assertion error if color is not a string. Stores the two
    # tensors and the color as attributes.
    assert isinstance(color, str)
    self.x = ops.convert_to_tensor(x)
    self.y = ops.convert_to_tensor(y)
    self.color = color


@type_spec_registry.register("tf.TwoTensorsSpec")
class TwoTensorsSpec(type_spec.TypeSpec):
  # [CLASS DOC]
  # HELPER: test spec class for the two-tensors value type, registered in the spec registry under
  # a test name at import time. Holds the shape and dtype of each tensor plus the color metadata.
  # Component specs are two tensor specs; serialization is the five attributes in order.
  """A TypeSpec for the TwoTensors value type."""

  def __init__(self, x_shape, x_dtype, y_shape, y_dtype, color="red"):
    # [FUNCTION DOC]
    # HELPER: constructor of the test spec. Inputs: shapes (anything convertible to a shape,
    # including nothing for unknown rank) and dtypes (anything convertible to a dtype) for the x and
    # y tensors, and color metadata (default red, any comparable object). Stores them as attributes.
    self.x_shape = tensor_shape.as_shape(x_shape)
    self.x_dtype = dtypes.as_dtype(x_dtype)
    self.y_shape = tensor_shape.as_shape(y_shape)
    self.y_dtype = dtypes.as_dtype(y_dtype)
    self.color = color

  value_type = property(lambda self: TwoTensors)

  @property
  def _component_specs(self):
    # [FUNCTION DOC]
    # HELPER: property giving the pair of tensor specs (x then y) built from the stored shapes and
    # dtypes.
    return (tensor_spec.TensorSpec(self.x_shape, self.x_dtype),
            tensor_spec.TensorSpec(self.y_shape, self.y_dtype))

  def _to_components(self, value):
    # [FUNCTION DOC]
    # HELPER: decomposes a two-tensors value into the pair of its x and y tensors.
    return (value.x, value.y)

  def _from_components(self, components):
    # [FUNCTION DOC]
    # HELPER: rebuilds a two-tensors value from a pair of tensors, using this spec's color.
    # Raises a value error (from unpacking) if components is not a pair.
    x, y = components
    return TwoTensors(x, y, self.color)

  def _serialize(self):
    # [FUNCTION DOC]
    # HELPER: returns the tuple of x shape, x dtype, y shape, y dtype and color.
    return (self.x_shape, self.x_dtype, self.y_shape, self.y_dtype, self.color)

  @classmethod
  def from_value(cls, value):
    # [FUNCTION DOC]
    # HELPER: class method building a spec from a two-tensors value by copying the shapes and dtypes
    # of its tensors and its color. Registered as the value converter for the value type.
    return cls(value.x.shape, value.x.dtype, value.y.shape, value.y.dtype,
               value.color)


@type_spec_registry.register("tf.TwoTensorsSpecTwin")
class TwoTensorsSpecTwin(TwoTensorsSpec):
  # [CLASS DOC]
  # HELPER: empty subclass of the two-tensors spec, registered under its own name. Used to check
  # that specs of different classes with identical content are unequal and not subtypes.
  pass


@type_spec_registry.register("tf.TwoTensorsSpecVariableSerialize")
class TwoTensorsSpecVariableSerialize(TwoTensorsSpec):
  # [CLASS DOC]
  # HELPER: two-tensors spec variant whose serialization changes structure depending on the color
  # metadata, used to test subtype and join behavior on mismatched serialization structures.

  def _serialize(self):
    # [FUNCTION DOC]
    # HELPER: returns a shorter four-item tuple when the color is the smaller-tuple marker, a
    # reordered five-item tuple (y shape, x shape, y dtype, color, x dtype) for the different-order
    # marker, and the normal five-item tuple otherwise.
    if self.color == "smaller_tuple":
      return (self.x_shape, self.x_dtype, self.y_shape, self.y_dtype)
    elif self.color == "different_order":
      return (self.y_shape, self.x_shape, self.y_dtype, self.color,
              self.x_dtype)

    return (self.x_shape, self.x_dtype, self.y_shape, self.y_dtype, self.color)


type_spec.register_type_spec_from_value_converter(
    TwoTensors, TwoTensorsSpec.from_value)


class TwoComposites:
  # [CLASS DOC]
  # HELPER: test value type holding two tensors or composite tensors (x and y) and a color string.
  # Described by the two-composites batchable spec class below.
  """A simple value type to test TypeSpec.

  Contains two composite tensorstensors (x, y) and a string (color).
  """

  def __init__(self, x, y, color="red"):
    # [FUNCTION DOC]
    # HELPER: constructor of the test value type. Inputs: x and y, tensors, composite tensors or
    # values convertible to them (converted accordingly); color, string defaulting to red. Raises
    # an assertion error if color is not a string. Stores all three as attributes.
    assert isinstance(color, str)
    self.x = ops.convert_to_tensor_or_composite(x)
    self.y = ops.convert_to_tensor_or_composite(y)
    self.color = color


@type_spec_registry.register("tf.TwoCompositesSpec")
class TwoCompositesSpec(type_spec.BatchableTypeSpec):
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring says it is the spec for the two-tensors value type, but it actually
  # describes the two-composites value type.
  # HELPER: batchable test spec for the two-composites value, registered in the spec registry at
  # import time. Holds a spec for each of the two components plus color metadata; component specs
  # are those two specs, and batching/unbatching delegates to each component.
  """A TypeSpec for the TwoTensors value type."""

  def __init__(self, x_spec, y_spec, color="red"):
    # [FUNCTION DOC]
    # HELPER: constructor. Inputs: x_spec and y_spec, specs of the two components; color, metadata
    # defaulting to red. Stores them as attributes.
    self.x_spec = x_spec
    self.y_spec = y_spec
    self.color = color

  value_type = property(lambda self: TwoComposites)

  @property
  def _component_specs(self):
    # [FUNCTION DOC]
    # HELPER: property returning the pair of the x spec and the y spec.
    return (self.x_spec, self.y_spec)

  def _to_components(self, value):
    # [FUNCTION DOC]
    # HELPER: decomposes a two-composites value into the pair of its x and y components.
    return (value.x, value.y)

  def _from_components(self, components):
    # [FUNCTION DOC]
    # HELPER: rebuilds a two-composites value from a pair of components with this spec's color.
    # Raises a value error (from unpacking) if components is not a pair.
    x, y = components
    return TwoComposites(x, y, self.color)

  def _serialize(self):
    # [FUNCTION DOC]
    # HELPER: returns the tuple of x spec, y spec and color.
    return (self.x_spec, self.y_spec, self.color)

  @classmethod
  def from_value(cls, value):
    # [FUNCTION DOC]
    # HELPER: class method building a spec from a two-composites value by inferring the spec of
    # each component and copying the color. Registered as the value converter for the value type.
    return cls(type_spec.type_spec_from_value(value.x),
               type_spec.type_spec_from_value(value.y),
               value.color)

  def _batch(self, batch_size):
    # [FUNCTION DOC]
    # HELPER: returns a new spec whose two component specs are each batched with batch_size (an
    # integer or nothing for unknown); color is preserved.
    return TwoCompositesSpec(
        self.x_spec._batch(batch_size), self.y_spec._batch(batch_size),
        self.color)

  def _unbatch(self):
    # [FUNCTION DOC]
    # HELPER: returns a new spec whose two component specs are each unbatched; color is preserved.
    return TwoCompositesSpec(self.x_spec._unbatch(), self.y_spec._unbatch(),
                             self.color)


type_spec.register_type_spec_from_value_converter(
    TwoComposites, TwoCompositesSpec.from_value)


class NestOfTensors:
  # [CLASS DOC]
  # KNOWN ISSUE: the docstring calls it a composite tensor, but it is a plain class that does not
  # derive from the composite tensor base.
  # HELPER: test value type wrapping an arbitrary nested structure (stored in the nest attribute).
  """CompositeTensor containing a nest of tensors."""

  def __init__(self, x):
    # [FUNCTION DOC]
    # HELPER: constructor storing the given nested structure x as the nest attribute.
    self.nest = x


@type_spec_registry.register("tf.NestOfTensorsSpec")
class NestOfTensorsSpec(type_spec.TypeSpec):
  # [CLASS DOC]
  # HELPER: test spec for the nest-of-tensors value type, registered in the spec registry at
  # import time. Stores one nested structure of specs (possibly with named tuples, dictionaries or
  # plain values); that structure is both the component specs and the serialization.
  """A TypeSpec for the NestOfTensors value type."""

  def __init__(self, spec):
    # [FUNCTION DOC]
    # HELPER: constructor storing the nested structure of specs as the spec attribute.
    self.spec = spec

  value_type = property(lambda self: NestOfTensors)
  _component_specs = property(lambda self: self.spec)

  def _to_components(self, value):
    # [FUNCTION DOC]
    # HELPER: flattens the given nested value into a flat list.
    return nest.flatten(value)

  def _from_components(self, components):
    # [FUNCTION DOC]
    # HELPER: packs the flat components back into the nested structure of this spec.
    return nest.pack_sequence_as(self.spec, components)

  def _serialize(self):
    # [FUNCTION DOC]
    # HELPER: returns the stored nested structure itself as the serialization.
    return self.spec

  def __repr__(self):
    # [FUNCTION DOC]
    # HELPER: string form. When the stored structure is a named tuple (has a sequence of string
    # field names) it formats the class name with the repr of the serialization directly;
    # otherwise falls back to the base class representation.
    if hasattr(self.spec, "_fields") and isinstance(
        self.spec._fields, collections_abc.Sequence) and all(
            isinstance(f, str) for f in self.spec._fields):
      return "%s(%r)" % (type(self).__name__, self._serialize())
    return super().__repr__()

  @classmethod
  def from_value(cls, value):
    # [FUNCTION DOC]
    # HELPER: class method building a spec from a nest-of-tensors value by inferring the spec of
    # every leaf of its nested structure. Registered as the value converter for the value type.
    return cls(nest.map_structure(type_spec.type_spec_from_value, value.nest))

  @classmethod
  def _deserialize(cls, spec):
    # [FUNCTION DOC]
    # HELPER: class method rebuilding a spec from its serialization, which is the nested
    # structure itself, passed as the single constructor argument.
    return cls(spec)


type_spec.register_type_spec_from_value_converter(
    NestOfTensors, NestOfTensorsSpec.from_value)

_TestNamedTuple = collections.namedtuple("NamedTuple", ["a", "b"])
_TestNamedTuple2 = collections.namedtuple("NamedTuple", ["a", "b"])
_TestNamedTupleSingleField = collections.namedtuple("SingleField", ["a"])
_TestNamedTupleDifferentField = collections.namedtuple("DifferentField",
                                                       ["a", "c"])


class TypeSpecTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the TypeSpec base class: equality and hashing, subtype and supertype checks,
  # compatibility, tensor decomposition and casting, tensor-list conversion, representation,
  # from-value inference, registry behavior and serialization. Uses the module-level two-tensors,
  # two-composites and nest-of-tensors test specs, and parameterized test generation.

  @parameterized.named_parameters(
      ("FullySpecified",
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool)),
      ("UnknownDim",
       TwoTensorsSpec([5, None], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, None], dtypes.int32, [None], dtypes.bool)),
      ("UnknownRank",
       TwoTensorsSpec(None, dtypes.int32, None, dtypes.bool),
       TwoTensorsSpec(None, dtypes.int32, None, dtypes.bool)),
      ("Metadata",
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool, "blue"),
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool, "blue")),
      ("NumpyMetadata",
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool,
                      (np.int32(1), np.float32(1.),
                       np.array([[1, 2], [3, 4]]))),
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool,
                      (np.int32(1), np.float32(1.),
                       np.array([[1, 2], [3, 4]])))),
      )
  def testEquality(self, v1, v2):
    # [FUNCTION DOC]
    # Parameterized over pairs of structurally identical specs (fully specified, unknown dims,
    # unknown rank, string metadata, numpy metadata). Asserts equality in both directions,
    # inequality operator false in both directions, and equal hashes.
    # pylint: disable=g-generic-assert
    self.assertEqual(v1, v2)
    self.assertEqual(v2, v1)
    self.assertFalse(v1 != v2)
    self.assertFalse(v2 != v1)
    self.assertEqual(hash(v1), hash(v2))

  @parameterized.named_parameters(
      ("UnknownDim",
       TwoTensorsSpec([5, None], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [8], dtypes.bool)),
      ("UnknownRank",
       TwoTensorsSpec(None, dtypes.int32, None, dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [8], dtypes.bool)),
      ("IncompatibleDtype",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.float32)),
      ("IncompatibleRank",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None, None], dtypes.bool)),
      ("IncompatibleDimSize",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 8], dtypes.int32, [None], dtypes.bool)),
      ("IncompatibleMetadata",
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool, "red"),
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool, "blue")),
      ("SwappedValues",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([None], dtypes.bool, [5, 3], dtypes.int32)),
      ("DiffMetadataNumpy",
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool,
                      np.array([[1, 2], [3, 4]])),
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool,
                      np.array([[1, 2], [3, 8]]))),
      ("DiffMetadataTensorSpecName",
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool,
                      tensor_spec.TensorSpec([4], name="a")),
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool,
                      tensor_spec.TensorSpec([4], name="b"))),
      ("Non-TypeSpec",
       TwoTensorsSpec([5, 3], dtypes.int32, [8], dtypes.bool), 5),
      )
  def testInequality(self, v1, v2):
    # [FUNCTION DOC]
    # Parameterized over pairs that differ (shape, rank, dtype, metadata, swapped values, numpy
    # metadata, tensor spec names, or a non-spec object). Asserts not-equal in both directions and
    # that the equality operator is false in both directions.
    # pylint: disable=g-generic-assert
    self.assertNotEqual(v1, v2)
    self.assertNotEqual(v2, v1)
    self.assertFalse(v1 == v2)
    self.assertFalse(v2 == v1)

  @parameterized.named_parameters(
      ("SameValue", TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool)),
      ("UnknownDim", TwoTensorsSpec([5, 3], dtypes.int32, [8], dtypes.bool),
       TwoTensorsSpec([5, None], dtypes.int32, [None], dtypes.bool)),
      ("UnknownRank", TwoTensorsSpec([5, 3], dtypes.int32, [8], dtypes.bool),
       TwoTensorsSpec(None, dtypes.int32, None, dtypes.bool)),
      ("NamedTuple",
       NestOfTensorsSpec(
           _TestNamedTuple(
               a=tensor_spec.TensorSpec([8, 5], dtypes.int32),
               b=tensor_spec.TensorSpec([8, 12], dtypes.int32))),
       NestOfTensorsSpec(
           _TestNamedTuple(
               a=tensor_spec.TensorSpec([None, 5], dtypes.int32),
               b=tensor_spec.TensorSpec([None, None], dtypes.int32)))),
      (
          "NamedTupleRedefined",
          NestOfTensorsSpec(
              _TestNamedTuple2(  # Separate but equivalent type.
                  a=tensor_spec.TensorSpec([8, 5], dtypes.int32),
                  b=tensor_spec.TensorSpec([8, 12], dtypes.int32))),
          NestOfTensorsSpec(
              _TestNamedTuple(
                  a=tensor_spec.TensorSpec([None, 5], dtypes.int32),
                  b=tensor_spec.TensorSpec([None, None], dtypes.int32)))),
  )
  def testIsSubtypeOf(self, v1, v2):
    # [FUNCTION DOC]
    # Parameterized over pairs where the first spec is more specific than (or equal to) the
    # second, including named tuples and an equivalent redefined named tuple type. Asserts that
    # the first is a subtype of the second.
    self.assertTrue(v1.is_subtype_of(v2))

  @parameterized.named_parameters(
      ("DifferentType",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpecTwin([5, 3], dtypes.int32, [None], dtypes.bool),
      ),
      ("DifferentDtype",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.float32)),
      ("DifferentRank", TwoTensorsSpec([5, 3], dtypes.int32, [None],
                                       dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None, None], dtypes.bool)),
      ("DifferentDimSize",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 8], dtypes.int32, [None], dtypes.bool)),
      ("DifferentMetadata",
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool, "red"),
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool, "blue")),
      ("SwappedValues", TwoTensorsSpec([5, 3], dtypes.int32, [None],
                                       dtypes.bool),
       TwoTensorsSpec([None], dtypes.bool, [5, 3], dtypes.int32)),
      ("SwappedDimensions",
       TwoTensorsSpec([3, 5], dtypes.int32, [None], dtypes.int32),
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.int32)),
      ("Supertype", TwoTensorsSpec([5, None], dtypes.int32, [None],
                                   dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [8], dtypes.bool)),
      ("SerializeDifferentStructure",
       TwoTensorsSpecVariableSerialize([5, None], dtypes.int32, [None],
                                       dtypes.bool),
       TwoTensorsSpecVariableSerialize([5, None], dtypes.int32, [None],
                                       dtypes.bool, "smaller_tuple")),
      ("SerializeDifferentOrder",
       TwoTensorsSpecVariableSerialize([5, None], dtypes.int32, [None],
                                       dtypes.bool),
       TwoTensorsSpecVariableSerialize([5, None], dtypes.int32, [None],
                                       dtypes.bool, "different_order")),
  )
  def testIsNotSubtypeOf(self, v1, v2):
    # [FUNCTION DOC]
    # Parameterized over pairs that must not be subtypes: different spec class, dtype, rank, dim
    # size, metadata, swapped values or dimensions, a supertype versus its subtype, and specs whose
    # serializations differ in structure or order. Asserts the first is not a subtype of the
    # second.
    self.assertFalse(v1.is_subtype_of(v2))

  @parameterized.named_parameters(
      ("SameValue", TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool)),
      ("DifferentValue",
       TwoTensorsSpec([2, 1], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([None, None], dtypes.int32, [None], dtypes.bool)),
      ("DifferentRank",
       TwoTensorsSpec([3, 2, 1], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec(None, dtypes.int32, [None], dtypes.bool)),
      (
          "NamedTupleRedefined",
          NestOfTensorsSpec(
              _TestNamedTuple2(  # Separate but equivalent type.
                  a=tensor_spec.TensorSpec([8, 3], dtypes.int32),
                  b=tensor_spec.TensorSpec([8, 12], dtypes.int32))),
          NestOfTensorsSpec(
              _TestNamedTuple(
                  a=tensor_spec.TensorSpec([None, 5], dtypes.int32),
                  b=tensor_spec.TensorSpec([7, None], dtypes.int32))),
          NestOfTensorsSpec(
              _TestNamedTuple(
                  a=tensor_spec.TensorSpec([None, None], dtypes.int32),
                  b=tensor_spec.TensorSpec([None, None], dtypes.int32)))),
  )
  def testMostSpecificCommonSupertype(self, v1, v2, result):
    # [FUNCTION DOC]
    # Parameterized over two specs and the expected join (same value, differing dimensions, ranks,
    # and redefined named tuples). Asserts that the join computed from either side equals the
    # expected spec, so the operation is symmetric.
    self.assertEqual(v1.most_specific_common_supertype([v2]), result)
    self.assertEqual(v2.most_specific_common_supertype([v1]), result)

  @parameterized.named_parameters(
      ("DifferentType",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpecTwin([5, 3], dtypes.int32, [None], dtypes.bool),
      ),
      ("DifferentDtype",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.float32)),
      ("DifferentMetadata",
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool, "red"),
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool, "blue")),
      ("SerializeDifferentStructure",
       TwoTensorsSpecVariableSerialize([5, None], dtypes.int32, [None],
                                       dtypes.bool),
       TwoTensorsSpecVariableSerialize([5, None], dtypes.int32, [None],
                                       dtypes.bool, "smaller_tuple")),
      ("SerializeDifferentOrder",
       TwoTensorsSpecVariableSerialize([5, None], dtypes.int32, [None],
                                       dtypes.bool),
       TwoTensorsSpecVariableSerialize([5, None], dtypes.int32, [None],
                                       dtypes.bool, "different_order")),
  )
  def testNoCommonSupertype(self, v1, v2):
    # [FUNCTION DOC]
    # Parameterized over pairs with no join (different class, dtype, metadata, or mismatched
    # serialization structure). Asserts that the join is nothing/null from either side.
    self.assertIsNone(v1.most_specific_common_supertype([v2]))
    self.assertIsNone(v2.most_specific_common_supertype([v1]))

  def testTensorDecomposition(self):
    # [FUNCTION DOC]
    # Builds a two-composites value of two ragged tensors (integer and float) and infers its spec.
    # Asserts the flattened spec list has four tensor specs (values and row splits per ragged
    # tensor), that decomposing the value into tensors yields four tensors with the expected
    # shapes and dtypes, and that rebuilding a value from an iterator over the tensors and
    # decomposing it again yields the very same tensor objects.
    value = TwoComposites(
        ragged_factory_ops.constant([[1, 2], [3]], dtypes.int32),
        ragged_factory_ops.constant([[5], [6, 7, 8]], dtypes.float32),
    )
    spec = type_spec.type_spec_from_value(value)

    self.assertEqual(
        spec.flatten(),
        [
            tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.int32),
            tensor_spec.TensorSpec(shape=(3,), dtype=dtypes.int64),
            tensor_spec.TensorSpec(shape=(None,), dtype=dtypes.float32),
            tensor_spec.TensorSpec(shape=(3,), dtype=dtypes.int64),
        ],
    )
    self.assertEqual(
        [trace_type.from_value(t) for t in spec.to_tensors(value)],
        [
            tensor_spec.TensorSpec(shape=(3,), dtype=dtypes.int32),
            tensor_spec.TensorSpec(shape=(3,), dtype=dtypes.int64),
            tensor_spec.TensorSpec(shape=(4,), dtype=dtypes.float32),
            tensor_spec.TensorSpec(shape=(3,), dtype=dtypes.int64),
        ],
    )

    flat_original = spec.to_tensors(value)
    reconstructed = spec.from_tensors(iter(flat_original))
    flat_reconstructed = spec.to_tensors(reconstructed)

    for original, reconstructed in zip(flat_original, flat_reconstructed):
      self.assertIs(original, reconstructed)

  def testCastDoesntRecreateCompositeTensor(self):
    # [FUNCTION DOC]
    # Builds a two-composites value of ragged tensors and its spec, casts the value with the
    # spec under a default internal casting context, and asserts the result is the identical
    # original object (no new value is created when nothing needs casting).
    value = TwoComposites(
        ragged_factory_ops.constant([[1, 2], [3]], dtypes.int32),
        ragged_factory_ops.constant([[5], [6, 7, 8]], dtypes.float32),
    )
    spec = type_spec.type_spec_from_value(value)

    casted_value = spec.cast(value, trace_type.InternalCastContext())

    self.assertIs(value, casted_value)

  @parameterized.named_parameters(
      ("SameValue",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool)),
      ("UnknownDim",
       TwoTensorsSpec([5, None], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [8], dtypes.bool)),
      ("UnknownRank",
       TwoTensorsSpec(None, dtypes.int32, None, dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [8], dtypes.bool)),
      ("NamedTuple",
       NestOfTensorsSpec(_TestNamedTuple(
           a=tensor_spec.TensorSpec([None, 5], dtypes.int32),
           b=tensor_spec.TensorSpec([None, None], dtypes.int32))),
       NestOfTensorsSpec(_TestNamedTuple(
           a=tensor_spec.TensorSpec([8, 5], dtypes.int32),
           b=tensor_spec.TensorSpec([8, 12], dtypes.int32)))),
      ("NamedTupleRedefined",
       NestOfTensorsSpec(_TestNamedTuple(
           a=tensor_spec.TensorSpec([None, 5], dtypes.int32),
           b=tensor_spec.TensorSpec([None, None], dtypes.int32))),
       NestOfTensorsSpec(_TestNamedTuple2(  # Separate but equivalent type.
           a=tensor_spec.TensorSpec([8, 5], dtypes.int32),
           b=tensor_spec.TensorSpec([8, 12], dtypes.int32)))),
      )
  def testIsCompatibleWith(self, v1, v2):
    # [FUNCTION DOC]
    # Parameterized over compatible pairs (same, unknown dim or rank, named tuples including a
    # redefined equivalent type). Asserts compatibility in both directions.
    self.assertTrue(v1.is_compatible_with(v2))
    self.assertTrue(v2.is_compatible_with(v1))

  @parameterized.named_parameters(
      ("IncompatibleDtype",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.float32)),
      ("IncompatibleRank",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None, None], dtypes.bool)),
      ("IncompatibleDimSize",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 8], dtypes.int32, [None], dtypes.bool)),
      ("IncompatibleMetadata",
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool, "red"),
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool, "blue")),
      ("SwappedValues",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([None], dtypes.bool, [5, 3], dtypes.int32)),
      )
  def testIsNotCompatibleWith(self, v1, v2):
    # [FUNCTION DOC]
    # Parameterized over incompatible pairs (dtype, rank, dimension size, metadata, swapped
    # values). Asserts incompatibility in both directions.
    self.assertFalse(v1.is_compatible_with(v2))
    self.assertFalse(v2.is_compatible_with(v1))

  @parameterized.named_parameters(
      ("EqualTypes",
       TwoTensorsSpec([5, 3], dtypes.int32, None, dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, None, dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, None, dtypes.bool)),
      ("UnknownDim",
       TwoTensorsSpec([5, None], dtypes.int32, [8], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, None], dtypes.int32, [None], dtypes.bool)),
      ("UnknownRank",
       TwoTensorsSpec(None, dtypes.int32, None, dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [8], dtypes.bool),
       TwoTensorsSpec(None, dtypes.int32, None, dtypes.bool)),
      ("DiffRank",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None, None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, None, dtypes.bool)),
      ("DiffDimSize",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 8], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, None], dtypes.int32, [None], dtypes.bool)),
      ("NamedTuple",
       NestOfTensorsSpec(_TestNamedTuple(
           a=tensor_spec.TensorSpec((), dtypes.int32),
           b=tensor_spec.TensorSpec((), dtypes.int32))),
       NestOfTensorsSpec(_TestNamedTuple(
           a=tensor_spec.TensorSpec((), dtypes.int32),
           b=tensor_spec.TensorSpec((), dtypes.int32))),
       NestOfTensorsSpec(_TestNamedTuple(
           a=tensor_spec.TensorSpec((), dtypes.int32),
           b=tensor_spec.TensorSpec((), dtypes.int32)))),
      ("NamedTupleRedefined",
       NestOfTensorsSpec(_TestNamedTuple(
           a=tensor_spec.TensorSpec((), dtypes.int32),
           b=tensor_spec.TensorSpec((), dtypes.int32))),
       NestOfTensorsSpec(_TestNamedTuple2(  # Separate but equivalent type.
           a=tensor_spec.TensorSpec((), dtypes.int32),
           b=tensor_spec.TensorSpec((), dtypes.int32))),
       NestOfTensorsSpec(_TestNamedTuple(
           a=tensor_spec.TensorSpec((), dtypes.int32),
           b=tensor_spec.TensorSpec((), dtypes.int32)))),
      )
  def testMostSpecificCompatibleType(self, v1, v2, expected):
    # [FUNCTION DOC]
    # Parameterized over two specs and the expected result of the deprecated most-specific
    # compatible type operation (equal types, unknown dims, ranks, named tuples). Asserts the
    # expected spec comes back from either side.
    self.assertEqual(v1.most_specific_compatible_type(v2), expected)
    self.assertEqual(v2.most_specific_compatible_type(v1), expected)

  @parameterized.named_parameters(
      ("IncompatibleDtype",
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool),
       TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.float32)),
      ("IncompatibleMetadata",
       TwoTensorsSpec([5, 3], dtypes.int32, None, dtypes.bool, "red"),
       TwoTensorsSpec([5, 3], dtypes.int32, None, dtypes.bool, "blue")),
      ("IncompatibleTensorSpecName",
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool,
                      tensor_spec.TensorSpec([4], name="a")),
       TwoTensorsSpec([5, 3], dtypes.int32, [3], dtypes.bool,
                      tensor_spec.TensorSpec([4], name="b"))),
      ("IncompatibleNestType",
       NestOfTensorsSpec(_TestNamedTuple(
           a=tensor_spec.TensorSpec((), dtypes.int32),
           b=tensor_spec.TensorSpec((), dtypes.int32))),
       NestOfTensorsSpec(dict(
           a=tensor_spec.TensorSpec((), dtypes.int32),
           b=tensor_spec.TensorSpec((), dtypes.int32)))),
      )
  def testMostSpecificCompatibleTypeException(self, v1, v2):
    # [FUNCTION DOC]
    # Parameterized over pairs with no compatible type (dtype, metadata, tensor spec name, nest
    # container type). Asserts the deprecated operation raises a value error from either side.
    with self.assertRaises(ValueError):
      v1.most_specific_compatible_type(v2)
    with self.assertRaises(ValueError):
      v2.most_specific_compatible_type(v1)

  def testMostSpecificCompatibleTypeNamedTupleIsNotTuple(self):
    # [FUNCTION DOC]
    # Builds nest specs from named tuples holding plain numbers and strings, and from a plain tuple.
    # Asserts that the join of two named-tuple specs (in either order) prints the same as the
    # third named-tuple spec, and differs in printed form from the plain-tuple spec.
    named_tuple_spec_a = NestOfTensorsSpec.from_value(NestOfTensors(
        _TestNamedTuple(a=1, b="aaa")))
    named_tuple_spec_b = NestOfTensorsSpec.from_value(NestOfTensors(
        _TestNamedTuple(a=2, b="bbb")))
    named_tuple_spec_c = NestOfTensorsSpec.from_value(NestOfTensors(
        _TestNamedTuple(a=3, b="ccc")))
    normal_tuple_spec = NestOfTensorsSpec.from_value(NestOfTensors((2, "bbb")))
    result_a_b = named_tuple_spec_a.most_specific_compatible_type(
        named_tuple_spec_b)
    result_b_a = named_tuple_spec_b.most_specific_compatible_type(
        named_tuple_spec_a)
    self.assertEqual(repr(result_a_b), repr(named_tuple_spec_c))
    self.assertEqual(repr(result_b_a), repr(named_tuple_spec_c))
    # Test that spec of named tuple is not equal to spec of normal tuple.
    self.assertNotEqual(repr(result_a_b), repr(normal_tuple_spec))

  @parameterized.named_parameters(
      ("IncompatibleDtype",
       NestOfTensorsSpec(_TestNamedTuple(
           a=tensor_spec.TensorSpec((), dtypes.int32),
           b=tensor_spec.TensorSpec((), dtypes.bool))),
       NestOfTensorsSpec(_TestNamedTuple(
           a=tensor_spec.TensorSpec((), dtypes.int32),
           b=tensor_spec.TensorSpec((), dtypes.int32)))),
      ("DifferentTupleSize",
       NestOfTensorsSpec(_TestNamedTuple(
           a=tensor_spec.TensorSpec((), dtypes.int32),
           b=tensor_spec.TensorSpec((), dtypes.bool))),
       NestOfTensorsSpec(_TestNamedTupleSingleField(
           a=tensor_spec.TensorSpec((), dtypes.int32)))),
      ("DifferentFieldName",
       NestOfTensorsSpec(_TestNamedTuple(
           a=tensor_spec.TensorSpec((), dtypes.int32),
           b=tensor_spec.TensorSpec((), dtypes.int32))),
       NestOfTensorsSpec(_TestNamedTupleDifferentField(
           a=tensor_spec.TensorSpec((), dtypes.int32),
           c=tensor_spec.TensorSpec((), dtypes.int32)))),
      ("NamedTupleAndTuple",
       NestOfTensorsSpec(_TestNamedTuple(
           a=tensor_spec.TensorSpec((), dtypes.int32),
           b=tensor_spec.TensorSpec((), dtypes.int32))),
       NestOfTensorsSpec((
           tensor_spec.TensorSpec((), dtypes.int32),
           tensor_spec.TensorSpec((), dtypes.int32)))),
      )
  def testMostSpecificCompatibleTypeForNamedTuplesException(self, v1, v2):
    # [FUNCTION DOC]
    # Parameterized over named-tuple nest specs that cannot be joined (dtype mismatch, different
    # tuple size, different field name, named tuple versus plain tuple). Asserts a value error
    # from the deprecated operation in both directions.
    with self.assertRaises(ValueError):
      v1.most_specific_compatible_type(v2)
    with self.assertRaises(ValueError):
      v2.most_specific_compatible_type(v1)

  def toTensorList(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name does not start with the test prefix, so the test runner never
    # collects or runs this method; its assertions are dead code.
    # Intended check: a spec from a two-tensors value converts the value to a tensor list of
    # length two whose items are the identical x and y tensors.
    value = TwoTensors([1, 2, 3], [1.0, 2.0], "red")
    spec = TwoTensorsSpec.from_value(value)
    tensor_list = spec._to_tensor_list(value)
    self.assertLen(tensor_list, 2)
    self.assertIs(tensor_list[0], value.x)
    self.assertIs(tensor_list[1], value.y)

  def fromTensorList(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name does not start with the test prefix, so the test runner never
    # collects or runs this method; its assertions are dead code.
    # Intended check: a spec with a chosen color rebuilds a value from a list of an integer tensor
    # and a float tensor, keeping the identical tensors and the color.
    x = ops.convert_to_tensor([1, 2, 3])
    y = ops.convert_to_tensor([1.0, 2.0])
    color = "green"
    spec = TwoTensorsSpec(x.shape, x.dtype, y.shape, y.dtype, color)
    value = spec._from_tensor_list([x, y])
    self.assertIs(value.x, x)
    self.assertIs(value.y, y)
    self.assertEqual(value.color, color)

  def fromIncompatibleTensorList(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name does not start with the test prefix, so the test runner never
    # collects or runs this method; its assertions are dead code.
    # Intended check: converting a tensor list back to a value raises a value error when the
    # shape of the first tensor mismatches the spec, and when the dtype of the second mismatches.
    x = ops.convert_to_tensor([1, 2, 3])
    y = ops.convert_to_tensor([1.0, 2.0])
    spec1 = TwoTensorsSpec([100], x.dtype, y.shape, y.dtype, "green")
    spec2 = TwoTensorsSpec(x.shape, x.dtype, y.shape, dtypes.bool, "green")
    with self.assertRaises(ValueError):
      spec1._from_tensor_list([x, y])  # shape mismatch
    with self.assertRaises(ValueError):
      spec2._from_tensor_list([x, y])  # dtype mismatch

  def testFlatTensorSpecs(self):
    # [FUNCTION DOC]
    # For a two-tensors spec with different shapes and dtypes, asserts the flat tensor specs are
    # the two tensor specs in order (x first, then y).
    spec = TwoTensorsSpec([5], dtypes.int32, [5, 8], dtypes.float32, "red")
    self.assertEqual(spec._flat_tensor_specs,
                     [tensor_spec.TensorSpec([5], dtypes.int32),
                      tensor_spec.TensorSpec([5, 8], dtypes.float32)])

  def testFullTypesForFlatTensors(self):
    # [FUNCTION DOC]
    # For a two-tensors spec (not ragged), computes full types for the flat tensors and asserts
    # the list has one "unset" full type per flat tensor spec, equal to the expected list.
    spec = TwoTensorsSpec([5], dtypes.int32, [5, 8], dtypes.float32, "red")
    full_type_list = fulltypes_for_flat_tensors(spec)
    expect = [
        full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_UNSET),
        full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_UNSET)
    ]
    self.assertEqual(len(spec._flat_tensor_specs), len(full_type_list))
    self.assertEqual(expect, full_type_list)

  def testRepr(self):
    # [FUNCTION DOC]
    # Asserts the printed form of a two-tensors spec is the class name followed by the repr of
    # its five serialized items (known and unknown-rank shapes, dtypes, color).
    spec = TwoTensorsSpec([5, 3], dtypes.int32, None, dtypes.bool)
    self.assertEqual(
        repr(spec),
        "TwoTensorsSpec(%r, %r, %r, %r, %r)" %
        (tensor_shape.TensorShape([5, 3]), dtypes.int32,
         tensor_shape.TensorShape(None), dtypes.bool, "red"))

  def testFromValue(self):
    # [FUNCTION DOC]
    # Asserts that inferring a spec from a two-tensors value through the public function uses the
    # registered converter and equals the spec built directly from the value.
    value = TwoTensors([1, 2, 3], [1.0, 2.0], "red")
    spec = type_spec.type_spec_from_value(value)
    self.assertEqual(spec, TwoTensorsSpec.from_value(value))

  def testCast(self):
    # [FUNCTION DOC]
    # Casts a two-tensors value built from plain numbers with an integer/float spec and asserts
    # both components become tensors of the expected dtypes. Then casts a two-composites value of
    # ragged tensors with its own spec and asserts the component dtypes are unchanged.
    spec = TwoTensorsSpec([], dtypes.int32, [], dtypes.float32)
    foo = spec._from_components([1, 2.3])
    ctx = trace_type.InternalCastContext()
    value = spec.cast(foo, ctx)
    tensor_type = type(ops.convert_to_tensor([1, 2, 3]))
    self.assertIsInstance(value.x, tensor_type)
    self.assertIsInstance(value.y, tensor_type)
    self.assertEqual(value.x.dtype, dtypes.int32)
    self.assertEqual(value.y.dtype, dtypes.float32)

    bar = TwoComposites(
        ragged_factory_ops.constant([[1, 2], [3]], dtypes.int32),
        ragged_factory_ops.constant([[5], [6, 7, 8]], dtypes.float32))
    bar_spec = type_spec.type_spec_from_value(bar)
    value = bar_spec.cast(bar, ctx)
    self.assertEqual(value.x.dtype, dtypes.int32)
    self.assertEqual(value.y.dtype, dtypes.float32)

  def testNestedRagged(self):
    # [FUNCTION DOC]
    # Checks compatibility between two-composites specs: one with a rank-zero ragged spec equals
    # (is compatible with) one using the matching plain tensor spec, but is not compatible with
    # one whose tensor spec has a different size.
    # Check that TwoCompositeSpecs are compatible if one has a nested
    # RaggedTensorSpec w/ ragged_rank=0 and the other has a corresponding
    # nested TensorSpec.
    spec1 = TwoCompositesSpec(
        ragged_tensor.RaggedTensorSpec([10], dtypes.int32, ragged_rank=0),
        tensor_spec.TensorSpec(None, dtypes.int32))
    spec2 = TwoCompositesSpec(
        tensor_spec.TensorSpec([10], dtypes.int32),
        tensor_spec.TensorSpec(None, dtypes.int32))
    spec3 = TwoCompositesSpec(
        tensor_spec.TensorSpec([12], dtypes.int32),
        tensor_spec.TensorSpec(None, dtypes.int32))
    self.assertTrue(spec1.is_compatible_with(spec2))
    self.assertFalse(spec1.is_compatible_with(spec3))

  def testRegistry(self):
    # [FUNCTION DOC]
    # Asserts that the registry returns the registered names for the two test spec classes and
    # that looking those names up returns the classes.
    self.assertEqual("tf.TwoCompositesSpec",
                     type_spec_registry.get_name(TwoCompositesSpec))
    self.assertEqual("tf.TwoTensorsSpec",
                     type_spec_registry.get_name(TwoTensorsSpec))
    self.assertEqual(TwoCompositesSpec,
                     type_spec_registry.lookup("tf.TwoCompositesSpec"))
    self.assertEqual(TwoTensorsSpec,
                     type_spec_registry.lookup("tf.TwoTensorsSpec"))

  def testRegistryTypeErrors(self):
    # [FUNCTION DOC]
    # Asserts the registration decorator factory raises a type error when the name is nothing or a
    # class instead of a string, and that applying the decorator to nothing or to a non-spec class
    # raises a type error saying a spec was expected.
    with self.assertRaisesRegex(TypeError, "Expected `name` to be a string"):
      type_spec_registry.register(None)

    with self.assertRaisesRegex(TypeError, "Expected `name` to be a string"):
      type_spec_registry.register(TwoTensorsSpec)

    with self.assertRaisesRegex(TypeError, "Expected `cls` to be a TypeSpec"):
      type_spec_registry.register("tf.foo")(None)

    with self.assertRaisesRegex(TypeError, "Expected `cls` to be a TypeSpec"):
      type_spec_registry.register("tf.foo")(ragged_tensor.RaggedTensor)

  def testRegistryDuplicateErrors(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the expected messages hard-code the main-program module name, so the test only
    # passes when the file is run as a script, not when imported under its module path.
    # Asserts that registering a new subclass under an already used name raises a value error
    # mentioning the name and existing class, and that registering an already registered class
    # under a new name raises a value error mentioning the existing name.
    with self.assertRaisesRegex(
        ValueError, "Name tf.TwoCompositesSpec has already been registered "
        "for class __main__.TwoCompositesSpec."):

      @type_spec_registry.register("tf.TwoCompositesSpec")  # pylint: disable=unused-variable
      class NewTypeSpec(TwoCompositesSpec):
        # [CLASS DOC]
        # HELPER: throwaway subclass whose registration under a duplicate name is expected to
        # fail; never successfully registered.
        pass

    with self.assertRaisesRegex(
        ValueError, "Class __main__.TwoCompositesSpec has already been "
        "registered with name tf.TwoCompositesSpec"):
      type_spec_registry.register("tf.NewName")(TwoCompositesSpec)

  def testRegistryNameErrors(self):
    # [FUNCTION DOC]
    # Asserts that names without the project-dot-type form (no dot, empty, containing a space) make
    # the registration decorator factory raise a value error.
    for bad_name in ["foo", "", "hello world"]:
      with self.assertRaises(ValueError):
        type_spec_registry.register(bad_name)

  def testRegistryLookupErrors(self):
    # [FUNCTION DOC]
    # Asserts that looking up a non-string name raises a type error and that looking up an
    # unregistered name raises a value error saying no spec was registered with that name.
    with self.assertRaises(TypeError):
      type_spec_registry.lookup(None)
    with self.assertRaisesRegex(
        ValueError, "No TypeSpec has been registered with name 'foo.bar'"):
      type_spec_registry.lookup("foo.bar")

  def testRegistryGetNameErrors(self):
    # [FUNCTION DOC]
    # Asserts that asking for the registered name of a non-class raises a type error, and that
    # asking for the name of an unregistered subclass raises a value error saying it has not been
    # registered (the message hard-codes the main-program module name).
    with self.assertRaises(TypeError):
      type_spec_registry.get_name(None)

    class Foo(TwoCompositesSpec):
      # [CLASS DOC]
      # HELPER: local unregistered subclass used to trigger the not-registered error.
      pass

    with self.assertRaisesRegex(
        ValueError, "TypeSpec __main__.Foo has not been registered."):
      type_spec_registry.get_name(Foo)

  def testSerialization(self):
    # [FUNCTION DOC]
    # Asserts that serializing a two-tensors spec with the trace-type serializer and
    # deserializing the result gives an equal spec (round trip).
    spec = TwoTensorsSpec([5, 3], dtypes.int32, [None], dtypes.bool)
    self.assertEqual(spec, trace_type.deserialize(trace_type.serialize(spec)))


class BatchableTypeSpecTest(test_util.TensorFlowTestCase,
                            parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the batchable spec base class using the two-composites test spec: batching and
  # unbatching of specs, flat tensor specs, full types, and tensor-list encoding/decoding of
  # values built from ragged tensors.

  @parameterized.parameters([
      {
          "unbatched":
              TwoCompositesSpec(
                  tensor_spec.TensorSpec([]), tensor_spec.TensorSpec([8])),
          "batch_size":
              5,
          "batched":
              TwoCompositesSpec(
                  tensor_spec.TensorSpec([5]), tensor_spec.TensorSpec([5, 8]))
      },
      {
          "unbatched":
              TwoCompositesSpec(
                  tensor_spec.TensorSpec(None), tensor_spec.TensorSpec([8])),
          "batch_size":
              None,
          "batched":
              TwoCompositesSpec(
                  tensor_spec.TensorSpec(None),
                  tensor_spec.TensorSpec([None, 8]))
      },
      {
          "unbatched":
              TwoCompositesSpec(
                  tensor_spec.TensorSpec([3, None]),
                  tensor_spec.TensorSpec([8])),
          "batch_size":
              12,
          "batched":
              TwoCompositesSpec(
                  tensor_spec.TensorSpec([12, 3, None]),
                  tensor_spec.TensorSpec([12, 8]))
      },
      {
          "unbatched":
              TwoCompositesSpec(
                  ragged_tensor.RaggedTensorSpec([3, None]),
                  tensor_spec.TensorSpec([8])),
          "batch_size":
              12,
          "batched":
              TwoCompositesSpec(
                  ragged_tensor.RaggedTensorSpec([12, 3, None]),
                  tensor_spec.TensorSpec([12, 8]))
      },
  ])
  def testBatch(self, unbatched, batch_size, batched):
    # [FUNCTION DOC]
    # Parameterized over an unbatched spec, a batch size (integer or unknown) and the expected
    # batched spec. Asserts batching gives the expected spec and unbatching it returns the
    # original.
    self.assertEqual(unbatched._batch(batch_size), batched)
    self.assertEqual(batched._unbatch(), unbatched)

  def testFlatTensorSpecs(self):
    # [FUNCTION DOC]
    # For a two-composites value of two ragged tensors, asserts the flat tensor specs (batchable
    # encoding) are two unknown-shape variant tensor specs, one per ragged component.
    a = TwoComposites(
        ragged_factory_ops.constant([[1, 2], [3]]),
        ragged_factory_ops.constant([[5], [6, 7, 8]]))
    a_spec = type_spec.type_spec_from_value(a)
    flat_specs = a_spec._flat_tensor_specs
    self.assertEqual(flat_specs, [
        tensor_spec.TensorSpec(None, dtypes.variant),
        tensor_spec.TensorSpec(None, dtypes.variant)
    ])

  def testFullTypesForFlatTensors(self):
    # [FUNCTION DOC]
    # For a two-composites value of two integer ragged tensors, computes full types for the flat
    # tensors and asserts there is one per flat spec, each a ragged type with a 32-bit integer
    # element argument.
    a = TwoComposites(
        ragged_factory_ops.constant([[1, 2], [3]]),
        ragged_factory_ops.constant([[5], [6, 7, 8]]))
    a_spec = type_spec.type_spec_from_value(a)
    full_type_list = fulltypes_for_flat_tensors(a_spec)
    expect = [
        full_type_pb2.FullTypeDef(
            type_id=full_type_pb2.TFT_RAGGED,
            args=[full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_INT32)]),
        full_type_pb2.FullTypeDef(
            type_id=full_type_pb2.TFT_RAGGED,
            args=[full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_INT32)]),
    ]
    self.assertEqual(len(a_spec._flat_tensor_specs), len(full_type_list))
    self.assertEqual(expect, full_type_list)

  def testToTensorList(self):
    # [FUNCTION DOC]
    # Encodes a two-composites ragged value to a tensor list and asserts there are two variant
    # scalar tensors. Rebuilds the value with the checked and with the unchecked conversion and
    # asserts both components and the color match the original.
    a = TwoComposites(
        ragged_factory_ops.constant([[1, 2], [3]]),
        ragged_factory_ops.constant([[5], [6, 7, 8]]))
    a_spec = type_spec.type_spec_from_value(a)
    tensor_list = a_spec._to_tensor_list(a)
    self.assertLen(tensor_list, 2)
    self.assertEqual(tensor_list[0].dtype, dtypes.variant)
    self.assertEqual(tensor_list[1].dtype, dtypes.variant)
    self.assertEqual(tensor_list[0].shape.rank, 0)
    self.assertEqual(tensor_list[1].shape.rank, 0)

    b = a_spec._from_tensor_list(tensor_list)
    self.assertAllEqual(a.x, b.x)
    self.assertAllEqual(a.y, b.y)
    self.assertEqual(a.color, b.color)

    c = a_spec._from_compatible_tensor_list(tensor_list)
    self.assertAllEqual(a.x, c.x)
    self.assertAllEqual(a.y, c.y)
    self.assertEqual(a.color, c.color)

  def testToBatchedTensorList(self):
    # [FUNCTION DOC]
    # Encodes a two-composites ragged value with the batched encoding and asserts there are two
    # variant tensors of rank one (batchable). Rebuilds the value with the checked and unchecked
    # conversions and asserts the components and color match the original.
    a = TwoComposites(
        ragged_factory_ops.constant([[1, 2], [3]]),
        ragged_factory_ops.constant([[5], [6, 7, 8]]))
    a_spec = type_spec.type_spec_from_value(a)
    tensor_list = a_spec._to_batched_tensor_list(a)
    self.assertLen(tensor_list, 2)
    self.assertEqual(tensor_list[0].dtype, dtypes.variant)
    self.assertEqual(tensor_list[1].dtype, dtypes.variant)
    self.assertEqual(tensor_list[0].shape.rank, 1)
    self.assertEqual(tensor_list[1].shape.rank, 1)

    b = a_spec._from_tensor_list(tensor_list)
    self.assertAllEqual(a.x, b.x)
    self.assertAllEqual(a.y, b.y)
    self.assertEqual(a.color, b.color)

    c = a_spec._from_compatible_tensor_list(tensor_list)
    self.assertAllEqual(a.x, c.x)
    self.assertAllEqual(a.y, c.y)
    self.assertEqual(a.color, c.color)


if __name__ == "__main__":
  googletest.main()
