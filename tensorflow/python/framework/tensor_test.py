# [MODULE DOC]
# Unit tests for the tensor module: name sanitization for specs, the TensorSpec type (construction,
# compatibility, representation, conversion from tensors, pickling, tracing types, placeholders,
# casting, flattening, serialization) and BoundedTensorSpec (bounds validation, read-only bounds,
# equality, conversion, pickling, serialization). Runs under the TensorFlow test framework via
# the standard test main entry point. No import-time side effects beyond imports.
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
"""Tests for Tensor and TensorSpec."""

import pickle

from absl.testing import parameterized
import numpy as np

from tensorflow.core.framework import full_type_pb2
from tensorflow.core.function import trace_type
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.framework import type_spec
from tensorflow.python.framework.type_utils import fulltypes_for_flat_tensors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest


class SpecNameSanizationTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case covering the spec-name sanitization helper in the tensor module. No fixtures.

  def test_sanitization(self):
    # [FUNCTION DOC]
    # Checks that sanitizing a spec name lower-cases it, replaces characters that are invalid in
    # names with underscores, prefixes names that start with a digit or underscore with a fixed
    # tensor prefix, and maps the empty string to the word unknown. Asserts exact outputs for each.
    self.assertEqual(tensor.sanitize_spec_name("normal"), "normal")
    self.assertEqual(
        tensor.sanitize_spec_name("123nums"), "tensor_123nums")
    self.assertEqual(tensor.sanitize_spec_name("____"), "tensor_____")
    self.assertEqual(tensor.sanitize_spec_name("_abc"), "tensor__abc")
    self.assertEqual(tensor.sanitize_spec_name("AdEfG"), "adefg")
    self.assertEqual(
        tensor.sanitize_spec_name("1_2_3_a"), "tensor_1_2_3_a")
    self.assertEqual(tensor.sanitize_spec_name("a_1-2"), "a_1_2")
    self.assertEqual(tensor.sanitize_spec_name("f6%hj"), "f6_hj")
    self.assertEqual(tensor.sanitize_spec_name("45&%"), "tensor_45__")
    self.assertEqual(tensor.sanitize_spec_name(""), "unknown")


class TensorSpecTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case (also parameterized) covering the TensorSpec type: construction defaults, shape and
  # dtype handling, compatibility checks, repr, factories, pickling, tracing-type relations,
  # placeholder creation, casting and flattening. Uses the default graph or eager mode of the
  # TensorFlow test framework.

  def testDefaultDType(self):
    # [FUNCTION DOC]
    # Verifies a spec built with only a shape gets the 32-bit float dtype by default.
    desc = tensor.TensorSpec([1])
    self.assertEqual(desc.dtype, dtypes.float32)

  def testAcceptsNumpyDType(self):
    # [FUNCTION DOC]
    # Verifies a numpy dtype passed to the constructor is converted to the matching TensorFlow dtype.
    desc = tensor.TensorSpec([1], np.float32)
    self.assertEqual(desc.dtype, dtypes.float32)

  def testAcceptsTensorShape(self):
    # [FUNCTION DOC]
    # Verifies a shape object (not just a list) is accepted and preserved as the spec shape.
    desc = tensor.TensorSpec(tensor_shape.TensorShape([1]), dtypes.float32)
    self.assertEqual(desc.shape, tensor_shape.TensorShape([1]))

  def testUnknownShape(self):
    # [FUNCTION DOC]
    # Verifies a nothing/null shape yields a spec whose shape is the unknown-rank shape.
    desc = tensor.TensorSpec(shape=None, dtype=dtypes.float32)
    self.assertEqual(desc.shape, tensor_shape.TensorShape(None))

  def testShapeCompatibility(self):
    # [FUNCTION DOC]
    # Builds, inside a fresh graph, placeholders of unknown, partial, full and rank-3 shapes and
    # checks the compatibility relation of specs with unknown, partial, full and rank-3 shapes
    # against each: unknown shapes match everything, partial matches same-rank matching dims, full
    # matches only equal or unknown shapes, and differing ranks are incompatible.
    # This test needs a placeholder which means we need to construct a graph.
    with ops.Graph().as_default():
      unknown = array_ops.placeholder(dtypes.int64)
      partial = array_ops.placeholder(dtypes.int64, shape=[None, 1])
      full = array_ops.placeholder(dtypes.int64, shape=[2, 3])
      rank3 = array_ops.placeholder(dtypes.int64, shape=[4, 5, 6])

      desc_unknown = tensor.TensorSpec(None, dtypes.int64)
      self.assertTrue(desc_unknown.is_compatible_with(unknown))
      self.assertTrue(desc_unknown.is_compatible_with(partial))
      self.assertTrue(desc_unknown.is_compatible_with(full))
      self.assertTrue(desc_unknown.is_compatible_with(rank3))

      desc_partial = tensor.TensorSpec([2, None], dtypes.int64)
      self.assertTrue(desc_partial.is_compatible_with(unknown))
      self.assertTrue(desc_partial.is_compatible_with(partial))
      self.assertTrue(desc_partial.is_compatible_with(full))
      self.assertFalse(desc_partial.is_compatible_with(rank3))

      desc_full = tensor.TensorSpec([2, 3], dtypes.int64)
      self.assertTrue(desc_full.is_compatible_with(unknown))
      self.assertFalse(desc_full.is_compatible_with(partial))
      self.assertTrue(desc_full.is_compatible_with(full))
      self.assertFalse(desc_full.is_compatible_with(rank3))

      desc_rank3 = tensor.TensorSpec([4, 5, 6], dtypes.int64)
      self.assertTrue(desc_rank3.is_compatible_with(unknown))
      self.assertFalse(desc_rank3.is_compatible_with(partial))
      self.assertFalse(desc_rank3.is_compatible_with(full))
      self.assertTrue(desc_rank3.is_compatible_with(rank3))

  def testTypeCompatibility(self):
    # [FUNCTION DOC]
    # Verifies a float spec is compatible with a float tensor of the same shape but not with an
    # integer tensor of that shape.
    floats = constant_op.constant(1, dtype=dtypes.float32, shape=[10, 10])
    ints = constant_op.constant(1, dtype=dtypes.int32, shape=[10, 10])
    desc = tensor.TensorSpec(shape=(10, 10), dtype=dtypes.float32)
    self.assertTrue(desc.is_compatible_with(floats))
    self.assertFalse(desc.is_compatible_with(ints))

  def testName(self):
    # [FUNCTION DOC]
    # Verifies the spec keeps arbitrary name strings (including ones not valid as tensor names,
    # such as a lone underscore) unchanged.
    # Note: "_" isn't a valid tensor name, but it is a valid python symbol
    # name; and tf.function constructs TensorSpecs using function argument
    # names.
    for name in ["beep", "foo/bar:0", "a-b_c/d", "_"]:
      desc = tensor.TensorSpec([1], dtypes.float32, name=name)
      self.assertEqual(desc.name, name)

  def testRepr(self):
    # [FUNCTION DOC]
    # Checks the textual representation of named and unnamed specs, including that unknown
    # dimensions print as a none marker under v2 shape behavior and as a question mark otherwise.
    desc1 = tensor.TensorSpec([1], dtypes.float32, name="beep")
    self.assertEqual(
        repr(desc1), "TensorSpec(shape=(1,), dtype=tf.float32, name='beep')")
    desc2 = tensor.TensorSpec([1, None], dtypes.int32)
    if desc2.shape._v2_behavior:
      self.assertEqual(
          repr(desc2), "TensorSpec(shape=(1, None), dtype=tf.int32, name=None)")
    else:
      self.assertEqual(
          repr(desc2), "TensorSpec(shape=(1, ?), dtype=tf.int32, name=None)")

  def testFromTensorSpec(self):
    # [FUNCTION DOC]
    # Verifies copying a spec with the from-spec factory yields an equal spec.
    spec_1 = tensor.TensorSpec((1, 2), dtypes.int32)
    spec_2 = tensor.TensorSpec.from_spec(spec_1)
    self.assertEqual(spec_1, spec_2)

  def testFromTensor(self):
    # [FUNCTION DOC]
    # Builds a spec from a scalar integer constant and checks dtype and empty shape; the name is
    # only asserted (as the constant op name) when not executing eagerly.
    zero = constant_op.constant(0)
    spec = tensor.TensorSpec.from_tensor(zero)
    self.assertEqual(spec.dtype, dtypes.int32)
    self.assertEqual(spec.shape, [])
    # Tensor.name is meaningless when eager execution is enabled.
    if not context.executing_eagerly():
      self.assertEqual(spec.name, "Const")

  def testFromPlaceholder(self):
    # [FUNCTION DOC]
    # In a fresh graph, builds specs from an unknown-shape and a partially known placeholder and
    # checks dtype, shape (unknown and unknown-by-one) and that the placeholder names are kept.
    # This test needs a placeholder which means we need to construct a graph.
    with ops.Graph().as_default():
      unknown = array_ops.placeholder(dtypes.int64, name="unknown")
      partial = array_ops.placeholder(
          dtypes.float32, shape=[None, 1], name="partial")

      spec_1 = tensor.TensorSpec.from_tensor(unknown)
      self.assertEqual(spec_1.dtype, dtypes.int64)
      self.assertEqual(spec_1.shape, None)
      self.assertEqual(spec_1.name, "unknown")
      spec_2 = tensor.TensorSpec.from_tensor(partial)
      self.assertEqual(spec_2.dtype, dtypes.float32)
      self.assertEqual(spec_2.shape.as_list(), [None, 1])
      self.assertEqual(spec_2.name, "partial")

  def testFromBoundedTensorSpec(self):
    # [FUNCTION DOC]
    # Converts a bounded spec to a plain spec with the from-spec factory and checks shape, dtype
    # and name are carried over.
    bounded_spec = tensor.BoundedTensorSpec((1, 2), dtypes.int32, 0, 1)
    spec = tensor.TensorSpec.from_spec(bounded_spec)
    self.assertEqual(bounded_spec.shape, spec.shape)
    self.assertEqual(bounded_spec.dtype, spec.dtype)
    self.assertEqual(bounded_spec.name, spec.name)

  def testPickleSerialization(self):
    # [FUNCTION DOC]
    # Pickles and unpickles a named spec and asserts the result equals the original.
    desc = tensor.TensorSpec([1, 5], dtypes.float32, "test")
    self.assertEqual(pickle.loads(pickle.dumps(desc)), desc)

  @test_util.deprecated_graph_mode_only
  def testTypeSpecFromValue(self):
    # [FUNCTION DOC]
    # Graph-mode-only test. Checks the generic type-spec-from-value helper returns the expected
    # int32 vector spec for both a numpy array and a constant tensor, and that doing so adds no
    # operations to the graph.
    g = ops.Graph()
    with g.as_default():
      v1 = np.array([1, 2, 3], np.int32)
      t1 = constant_op.constant(v1)

      ops_before = g.get_operations()

      expected = tensor.TensorSpec([3], dtypes.int32)
      self.assertEqual(expected, type_spec.type_spec_from_value(v1))
      self.assertEqual(expected, type_spec.type_spec_from_value(t1))

      # Check that creating TypeSpecs did not require building new Tensors.
      self.assertLen(g.get_operations(), len(ops_before))

  def testEqualTypes(self):
    # [FUNCTION DOC]
    # Creates two identical specs' tracing types and asserts they are equal and subtypes of each
    # other.
    signature_context = trace_type.InternalTracingContext()
    type_1 = tensor.TensorSpec(
        tensor_shape.TensorShape([1, 2, 3]), dtypes.float32,
        None).__tf_tracing_type__(signature_context)
    type_2 = tensor.TensorSpec(
        tensor_shape.TensorShape([1, 2, 3]), dtypes.float32,
        None).__tf_tracing_type__(signature_context)
    self.assertEqual(type_1, type_1)
    self.assertEqual(type_1, type_2)
    self.assertTrue(type_1.is_subtype_of(type_1))
    self.assertTrue(type_2.is_subtype_of(type_1))
    self.assertTrue(type_1.is_subtype_of(type_2))

  def testDtypeMismatch(self):
    # [FUNCTION DOC]
    # Asserts tracing types of specs differing only in dtype are unequal and neither is a subtype
    # of the other.
    signature_context = trace_type.InternalTracingContext()
    type_1 = tensor.TensorSpec(
        tensor_shape.TensorShape([1, 2, 3]), dtypes.float32,
        None).__tf_tracing_type__(signature_context)
    type_2 = tensor.TensorSpec(
        tensor_shape.TensorShape([1, 2, 3]), dtypes.int32,
        None).__tf_tracing_type__(signature_context)
    self.assertNotEqual(type_1, type_2)
    self.assertFalse(type_2.is_subtype_of(type_1))
    self.assertFalse(type_1.is_subtype_of(type_2))

  def testSubtypeOfShapeless(self):
    # [FUNCTION DOC]
    # Asserts a fully known shape tracing type is a subtype of the unknown-rank one, not vice versa.
    signature_context = trace_type.InternalTracingContext()
    type_1 = tensor.TensorSpec(
        tensor_shape.TensorShape(None), dtypes.float32,
        None).__tf_tracing_type__(signature_context)
    type_2 = tensor.TensorSpec(
        tensor_shape.TensorShape([1, 2, 3]), dtypes.float32,
        None).__tf_tracing_type__(signature_context)
    self.assertNotEqual(type_1, type_2)
    self.assertFalse(type_1.is_subtype_of(type_2))
    self.assertTrue(type_2.is_subtype_of(type_1))

  def testSubtypeOfDimlessShape(self):
    # [FUNCTION DOC]
    # Asserts a fully known shape tracing type is a subtype of the same-rank all-unknown-dimension
    # one, not vice versa.
    signature_context = trace_type.InternalTracingContext()
    type_1 = tensor.TensorSpec(
        tensor_shape.TensorShape([None, None, None]), dtypes.float32,
        None).__tf_tracing_type__(signature_context)
    type_2 = tensor.TensorSpec(
        tensor_shape.TensorShape([1, 2, 3]), dtypes.float32,
        None).__tf_tracing_type__(signature_context)
    self.assertNotEqual(type_1, type_2)
    self.assertFalse(type_1.is_subtype_of(type_2))
    self.assertTrue(type_2.is_subtype_of(type_1))

  def testSubtypeOfNamedSpec(self):
    # [FUNCTION DOC]
    # Asserts a spec without a name is a subtype of a named spec with the same shape and dtype,
    # while the named one is not a subtype of the unnamed one and the two are unequal.
    named = tensor.TensorSpec(
        tensor_shape.TensorShape([1, 2, 3]), dtypes.float32,
        "a_name")
    not_named = tensor.TensorSpec(
        tensor_shape.TensorShape([1, 2, 3]), dtypes.float32,
        None)
    self.assertNotEqual(named, not_named)
    self.assertFalse(named.is_subtype_of(not_named))
    self.assertTrue(not_named.is_subtype_of(named))

  def testFlatTensorSpecs(self):
    # [FUNCTION DOC]
    # Checks the internal flat tensor spec list of a spec is a list holding just that spec.
    spec = tensor.TensorSpec([1], np.float32)
    self.assertEqual(spec._flat_tensor_specs, [spec])

  def testFullTypesForFlatTensors(self):
    # [FUNCTION DOC]
    # Checks the full-type helper returns one unset full-type entry per flat tensor spec.
    spec = tensor.TensorSpec([1], np.float32)
    full_type_list = fulltypes_for_flat_tensors(spec)
    expect = [full_type_pb2.FullTypeDef(type_id=full_type_pb2.TFT_UNSET)]
    self.assertEqual(len(spec._flat_tensor_specs), len(full_type_list))
    self.assertEqual(expect, full_type_list)

  def testSerialization(self):
    # [FUNCTION DOC]
    # Serializes and deserializes named and nameless specs through the trace-type utilities and
    # asserts the round trip returns equal specs.
    nameless = tensor.TensorSpec([1], np.float32)
    named = tensor.TensorSpec([1, 2, 3], np.float32, name="some_name")
    self.assertEqual(nameless,
                     trace_type.deserialize(trace_type.serialize(nameless)))
    self.assertEqual(named, trace_type.deserialize(trace_type.serialize(named)))

  def testPlaceholderWithName(self):
    # [FUNCTION DOC]
    # Creates a placeholder from a named spec using a placeholder context on the default graph and
    # checks its name gets the zero output suffix and its dtype and shape match the spec.
    placeholder_context = trace_type.InternalPlaceholderContext(
        ops.get_default_graph())
    spec = tensor.TensorSpec([1], np.float32, name="test")
    placeholder = spec.placeholder_value(placeholder_context)
    self.assertEqual(placeholder.name, f"{spec.name}:0")
    self.assertEqual(placeholder.dtype, spec.dtype)
    self.assertEqual(placeholder.shape, spec.shape)

  def testMultiplePlaceholdersWithNames(self):
    # [FUNCTION DOC]
    # Creates placeholders from three named specs (including an unknown-shape one) in one context
    # and checks each keeps its own name with suffix, dtype and shape.
    placeholder_context = trace_type.InternalPlaceholderContext(
        ops.get_default_graph())
    spec1 = tensor.TensorSpec([1, 2], np.float32, name="test1")
    spec2 = tensor.TensorSpec([3], np.int32, name="test2")
    spec3 = tensor.TensorSpec(None, np.float32, name="test3")
    placeholder1 = spec1.placeholder_value(placeholder_context)
    placeholder2 = spec2.placeholder_value(placeholder_context)
    placeholder3 = spec3.placeholder_value(placeholder_context)
    self.assertEqual(placeholder1.name, f"{spec1.name}:0")
    self.assertEqual(placeholder2.name, f"{spec2.name}:0")
    self.assertEqual(placeholder3.name, f"{spec3.name}:0")
    self.assertEqual(placeholder1.dtype, spec1.dtype)
    self.assertEqual(placeholder2.dtype, spec2.dtype)
    self.assertEqual(placeholder3.dtype, spec3.dtype)
    self.assertEqual(placeholder1.shape, spec1.shape)
    self.assertEqual(placeholder2.shape, spec2.shape)
    self.assertEqual(placeholder3.shape, spec3.shape)

  def testPlaceholderWithoutName(self):
    # [FUNCTION DOC]
    # Creates a placeholder from an unnamed spec and checks it gets the default placeholder name,
    # with matching dtype and shape.
    placeholder_context = trace_type.InternalPlaceholderContext(
        ops.get_default_graph())
    spec = tensor.TensorSpec([1], np.float32)
    placeholder = spec.placeholder_value(placeholder_context)
    self.assertEqual(placeholder.name, "Placeholder:0")
    self.assertEqual(placeholder.dtype, spec.dtype)
    self.assertEqual(placeholder.shape, spec.shape)

  def testMultiplePlaceholdersWithoutNames(self):
    # [FUNCTION DOC]
    # Creates placeholders from three unnamed specs in one context and checks default names are
    # uniquified with increasing numeric suffixes, and dtypes and shapes match.
    placeholder_context = trace_type.InternalPlaceholderContext(
        ops.get_default_graph())
    spec1 = tensor.TensorSpec([1, 2], np.float32)
    spec2 = tensor.TensorSpec([3], np.int32)
    spec3 = tensor.TensorSpec(None, np.float32)
    placeholder1 = spec1.placeholder_value(placeholder_context)
    placeholder2 = spec2.placeholder_value(placeholder_context)
    placeholder3 = spec3.placeholder_value(placeholder_context)
    self.assertEqual(placeholder1.name, "Placeholder:0")
    self.assertEqual(placeholder2.name, "Placeholder_1:0")
    self.assertEqual(placeholder3.name, "Placeholder_2:0")
    self.assertEqual(placeholder1.dtype, spec1.dtype)
    self.assertEqual(placeholder2.dtype, spec2.dtype)
    self.assertEqual(placeholder3.dtype, spec3.dtype)
    self.assertEqual(placeholder1.shape, spec1.shape)
    self.assertEqual(placeholder2.shape, spec2.shape)
    self.assertEqual(placeholder3.shape, spec3.shape)

  def testGraphPlaceholderWithValidName(self):
    # [FUNCTION DOC]
    # Calls the internal graph placeholder builder with a valid name and checks the resulting
    # placeholder uses that name and matches the spec dtype and shape.
    spec = tensor.TensorSpec((2, 3), dtypes.float32, name="test")
    placeholder = spec._graph_placeholder(ops.get_default_graph(), spec.name)
    self.assertEqual(placeholder.name, f"{spec.name}:0")
    self.assertEqual(placeholder.dtype, spec.dtype)
    self.assertEqual(placeholder.shape, spec.shape)

  def testGraphPlaceholderWithInvalidName(self):
    # [FUNCTION DOC]
    # Calls the internal graph placeholder builder with an invalid name and checks it falls back
    # to the default placeholder name while matching dtype and shape.
    spec = tensor.TensorSpec((1, 2), dtypes.int32, name="a%!")
    placeholder = spec._graph_placeholder(ops.get_default_graph(), spec.name)
    self.assertEqual(placeholder.name, "Placeholder:0")
    self.assertEqual(placeholder.dtype, spec.dtype)
    self.assertEqual(placeholder.shape, spec.shape)

  def testCastPythinPrimitives(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the method name misspells Python as Pythin.
    # Casts a Python integer with a float scalar spec and asserts the result has the spec dtype.
    spec = tensor.TensorSpec([], dtypes.float32)
    ctx = trace_type.InternalCastContext()
    value = spec.cast(1, ctx)
    self.assertEqual(value.dtype, spec.dtype)

  def testCastTensor(self):
    # [FUNCTION DOC]
    # Asserts casting an int32 tensor with a float32 spec raises a type error (no implicit tensor
    # dtype conversion).
    spec = tensor.TensorSpec([], dtypes.float32)
    ctx = trace_type.InternalCastContext()
    # cast does not support cast int tensor to float tensor
    with self.assertRaises(TypeError):
      _ = spec.cast(constant_op.constant(1, dtype=dtypes.int32), ctx)

  def testCastAssert(self):
    # [FUNCTION DOC]
    # Asserts casting a list value with a scalar float spec raises a type error.
    spec = tensor.TensorSpec([], dtypes.float32)
    ctx = trace_type.InternalCastContext()
    with self.assertRaises(TypeError):
      _ = spec.cast([1, 2, 3], ctx)

  @parameterized.named_parameters(
      (
          "list",
          [[
              [tensor.TensorSpec(None, name="a")],
              [
                  tensor.TensorSpec(None, name="b"),
                  tensor.TensorSpec(None, name="c"),
              ],
          ]],
      ),
      (
          "tuple",
          ((
              (tensor.TensorSpec(None, name="a"),),
              (
                  tensor.TensorSpec(None, name="b"),
                  tensor.TensorSpec(None, name="c"),
              ),
          )),
      ),
      (
          "dict",
          {
              "a": {"key": tensor.TensorSpec(None, name="a")},
              "b": [tensor.TensorSpec(None, name="b")],
              "c": (tensor.TensorSpec(None, name="c"),),
          },
      ),
  )
  def testFlatten(self, structure):
    # [FUNCTION DOC]
    # Parameterized over a nested list, tuple and dict of named specs. Builds a legacy-signature
    # trace type from the structure, flattens it and asserts the result is the three specs a, b, c
    # in order.
    generated_type = trace_type.from_value(
        structure, trace_type.InternalTracingContext(is_legacy_signature=True)
    )
    flattened = generated_type.flatten()
    self.assertEqual(
        flattened,
        [
            tensor.TensorSpec(None, name="a"),
            tensor.TensorSpec(None, name="b"),
            tensor.TensorSpec(None, name="c"),
        ],
    )


class BoundedTensorSpecTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case covering BoundedTensorSpec: bound validation, numpy bound attributes, equality,
  # conversion, pickling and serialization.

  def testInvalidMinimum(self):
    # [FUNCTION DOC]
    # Asserts a minimum whose shape is incompatible with the spec shape raises a value error
    # mentioning incompatibility.
    with self.assertRaisesRegex(ValueError, "not compatible"):
      tensor.BoundedTensorSpec((3, 5), dtypes.uint8, (0, 0, 0), (1, 1))

  def testInvalidMaximum(self):
    # [FUNCTION DOC]
    # Asserts a maximum whose shape is incompatible with the spec shape raises a value error
    # mentioning incompatibility.
    with self.assertRaisesRegex(ValueError, "not compatible"):
      tensor.BoundedTensorSpec((3, 5), dtypes.uint8, 0, (1, 1, 1))

  def testMinimumMaximumAttributes(self):
    # [FUNCTION DOC]
    # Checks minimum and maximum are numpy arrays converted to the spec dtype, with the scalar
    # minimum kept scalar and the maximum a vector.
    spec = tensor.BoundedTensorSpec((1, 2, 3), dtypes.float32, 0,
                                    (5, 5, 5))
    self.assertEqual(type(spec.minimum), np.ndarray)
    self.assertEqual(type(spec.maximum), np.ndarray)
    self.assertAllEqual(spec.minimum, np.array(0, dtype=np.float32))
    self.assertAllEqual(spec.maximum, np.array([5, 5, 5], dtype=np.float32))

  def testNotWriteableNP(self):
    # [FUNCTION DOC]
    # Asserts writing into the minimum or maximum arrays raises a value error about read-only data.
    spec = tensor.BoundedTensorSpec((1, 2, 3), dtypes.float32, 0,
                                    (5, 5, 5))
    with self.assertRaisesRegex(ValueError, "read-only"):
      spec.minimum[0] = -1
    with self.assertRaisesRegex(ValueError, "read-only"):
      spec.maximum[0] = 100

  def testReuseSpec(self):
    # [FUNCTION DOC]
    # Rebuilds a spec from another's shape, dtype and bounds and asserts equality.
    spec_1 = tensor.BoundedTensorSpec((1, 2),
                                      dtypes.int32,
                                      minimum=0,
                                      maximum=1)
    spec_2 = tensor.BoundedTensorSpec(spec_1.shape, spec_1.dtype,
                                      spec_1.minimum, spec_1.maximum)
    self.assertEqual(spec_1, spec_2)

  def testScalarBounds(self):
    # [FUNCTION DOC]
    # With scalar shape and scalar bounds, checks bounds are numpy arrays comparing equal to the
    # given numbers and that they can be fed back into the constructor without error.
    spec = tensor.BoundedTensorSpec((),
                                    dtypes.float32,
                                    minimum=0.0,
                                    maximum=1.0)

    self.assertIsInstance(spec.minimum, np.ndarray)
    self.assertIsInstance(spec.maximum, np.ndarray)

    # Sanity check that numpy compares correctly to a scalar for an empty shape.
    self.assertEqual(0.0, spec.minimum)
    self.assertEqual(1.0, spec.maximum)

    # Check that the spec doesn't fail its own input validation.
    _ = tensor.BoundedTensorSpec(spec.shape, spec.dtype, spec.minimum,
                                 spec.maximum)

  def testFromBoundedTensorSpec(self):
    # [FUNCTION DOC]
    # Copies a bounded spec with the bounded from-spec factory and asserts the copy is equal.
    spec_1 = tensor.BoundedTensorSpec((1, 2),
                                      dtypes.int32,
                                      minimum=0,
                                      maximum=1)
    spec_2 = tensor.BoundedTensorSpec.from_spec(spec_1)
    self.assertEqual(spec_1, spec_2)

  def testEquality(self):
    # [FUNCTION DOC]
    # Checks equality semantics: bounds equal after rounding to dtype or after broadcasting a
    # scalar to a vector are equal, while differing bounds make specs unequal; symmetric checks.
    spec_1_1 = tensor.BoundedTensorSpec((1, 2, 3), dtypes.float32, 0,
                                        (5, 5, 5))
    spec_1_2 = tensor.BoundedTensorSpec(
        (1, 2, 3), dtypes.float32, 0.00000001, (5, 5, 5.00000000000000001))
    spec_2_1 = tensor.BoundedTensorSpec((1, 2, 3), dtypes.float32, 1,
                                        (5, 5, 5))
    spec_2_2 = tensor.BoundedTensorSpec((1, 2, 3), dtypes.float32,
                                        (1, 1, 1), (5, 5, 5))
    spec_2_3 = tensor.BoundedTensorSpec((1, 2, 3), dtypes.float32,
                                        (1, 1, 1), 5)
    spec_3_1 = tensor.BoundedTensorSpec((1, 2, 3), dtypes.float32,
                                        (2, 1, 1), (5, 5, 5))

    self.assertEqual(spec_1_1, spec_1_2)
    self.assertEqual(spec_1_2, spec_1_1)

    self.assertNotEqual(spec_1_1, spec_2_2)
    self.assertNotEqual(spec_1_1, spec_2_1)
    self.assertNotEqual(spec_2_2, spec_1_1)
    self.assertNotEqual(spec_2_1, spec_1_1)

    self.assertEqual(spec_2_1, spec_2_2)
    self.assertEqual(spec_2_2, spec_2_1)
    self.assertEqual(spec_2_2, spec_2_3)

    self.assertNotEqual(spec_1_1, spec_3_1)
    self.assertNotEqual(spec_2_1, spec_3_1)
    self.assertNotEqual(spec_2_2, spec_3_1)

  def testFromTensorSpec(self):
    # [FUNCTION DOC]
    # Converts a plain spec to a bounded spec and checks shape, dtype, name carry over and bounds
    # equal the dtype minimum and maximum.
    spec = tensor.TensorSpec((1, 2), dtypes.int32)
    bounded_spec = tensor.BoundedTensorSpec.from_spec(spec)
    self.assertEqual(spec.shape, bounded_spec.shape)
    self.assertEqual(spec.dtype, bounded_spec.dtype)
    self.assertEqual(spec.dtype.min, bounded_spec.minimum)
    self.assertEqual(spec.dtype.max, bounded_spec.maximum)
    self.assertEqual(spec.name, bounded_spec.name)

  def testPickleSerialization(self):
    # [FUNCTION DOC]
    # Pickles and unpickles a named bounded spec and asserts the result equals the original.
    desc = tensor.BoundedTensorSpec([1, 5], dtypes.float32, -1, 1, "test")
    self.assertEqual(pickle.loads(pickle.dumps(desc)), desc)

  def testSerialization(self):
    # [FUNCTION DOC]
    # Serializes and deserializes named and nameless bounded specs through the trace-type
    # utilities and asserts the round trip returns equal specs.
    nameless = tensor.BoundedTensorSpec([1], np.float32, 0, 1)
    named = tensor.BoundedTensorSpec([1, 2, 3],
                                     np.float32,
                                     0,
                                     1,
                                     name="some_name")
    self.assertEqual(nameless,
                     trace_type.deserialize(trace_type.serialize(nameless)))
    self.assertEqual(named, trace_type.deserialize(trace_type.serialize(named)))


if __name__ == "__main__":
  googletest.main()
