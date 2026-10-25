# [MODULE DOC]
# Unit tests for indexing and slicing (the item-access operation) of StructuredTensor.
# Defines example nested structures (a scalar struct, a second one, and a vector of six), a
# matching StructuredTensorSpec, small helpers to build slice specs and to compare structured
# tensors, and a test case checking field selection, multi-dimensional indexing, slicing and
# the errors raised for unsupported keys. Each check runs in graph and eager modes. Running the
# file as a script launches the test runner.
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
"""Tests for StructuredTensor."""

from absl.testing import parameterized

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.structured import structured_tensor
from tensorflow.python.platform import googletest


# TODO(edloper): Move this to a common util package (forked from ragged).
class _SliceBuilder:
  # [CLASS DOC]
  # HELPER: Test utility that lets tests write slice expressions readably.
  # Stateless object whose item-access operation just returns the key it receives, so indexing
  # an instance yields the raw slice spec (a key, slice, tuple of them, or a list) that Python
  # would pass to an item-access call. A module-level instance is used throughout the tests.
  """Helper to construct arguments for __getitem__.

  Usage: _SliceBuilder()[<expr>] slice_spec Python generates for <expr>.
  """

  def __getitem__(self, slice_spec):
    # [FUNCTION DOC]
    # HELPER: Lets tests construct slice specs with bracket syntax.
    # Takes slice_spec (any key, slice, ellipsis, tuple or list given inside the brackets) and
    # returns it unchanged. No exceptions and no side effects.
    return slice_spec


# TODO(edloper): Move this to a common util package (forked from ragged).
SLICE_BUILDER = _SliceBuilder()


# TODO(edloper): Move this to a common util package (forked from ragged).
def _make_tensor_slice_spec(slice_spec, use_constant=True):
  # [FUNCTION DOC]
  # HELPER: Test utility that converts integer slice specs into tensor-valued ones.
  # Inputs: slice_spec (a single key or slice, or a tuple of them) and use_constant (optional,
  # default true). Every integer, including the start, stop and step of slices, is replaced by
  # a scalar tensor: a constant when use_constant is true, otherwise a placeholder with that
  # default value (unknown at graph construction time). Non-integer pieces such as strings are
  # kept as they are. Returns a new spec of the same shape (a tuple if a tuple was given). The
  # input is not mutated; tensors are created in the current graph or eager context.
  """Wraps all integers in an extended slice spec w/ a tensor.

  This function is used to help test slicing when the slice spec contains
  tensors, rather than integers.

  Args:
    slice_spec: The extended slice spec.
    use_constant: If true, then wrap each integer with a tf.constant.  If false,
      then wrap each integer with a tf.placeholder.

  Returns:
    A copy of slice_spec, but with each integer i replaced with tf.constant(i).
  """

  def make_piece_scalar(piece):
    # [FUNCTION DOC]
    # HELPER: Inner worker of the tensor slice spec builder.
    # Converts one piece: an integer becomes a scalar constant or a placeholder-with-default
    # (depending on the enclosing use_constant flag); a slice is rebuilt with each of its start,
    # stop and step converted recursively; any other value is returned unchanged.
    if isinstance(piece, int):
      scalar = constant_op.constant(piece)
      if use_constant:
        return scalar
      else:
        return array_ops.placeholder_with_default(scalar, [])
    elif isinstance(piece, slice):
      return slice(
          make_piece_scalar(piece.start), make_piece_scalar(piece.stop),
          make_piece_scalar(piece.step))
    else:
      return piece

  if isinstance(slice_spec, tuple):
    return tuple(make_piece_scalar(piece) for piece in slice_spec)
  else:
    return make_piece_scalar(slice_spec)


EXAMPLE_STRUCT = {
    # f1: scalar value field
    "f1": 1,
    # f2: matrix field
    "f2": [[1, 2], [3, 4]],
    # f3: scalar structure field
    "f3": {
        "f3_1": 1
    },
    # f4: vector structure field
    "f4": [{
        "f4_1": 1,
        "f4_2": b"a"
    }, {
        "f4_1": 2,
        "f4_2": b"b"
    }],
    # f5: matrix structure field
    "f5": [[{
        "f5_1": 1
    }, {
        "f5_1": 2
    }], [{
        "f5_1": 3
    }, {
        "f5_1": 4
    }]],
}

EXAMPLE_STRUCT_2 = {
    # f1: scalar value field
    "f1": 5,
    # f2: matrix field
    "f2": [[6, 7], [8, 9]],
    # f3: scalar structure field
    "f3": {
        "f3_1": 9
    },
    # f4: vector structure field
    "f4": [{
        "f4_1": 5,
        "f4_2": b"A"
    }, {
        "f4_1": 6,
        "f4_2": b"B"
    }],
    # f5: matrix structure field
    "f5": [[{
        "f5_1": 6
    }, {
        "f5_1": 7
    }], [{
        "f5_1": 8
    }, {
        "f5_1": 9
    }]],
}

EXAMPLE_STRUCT_VECTOR = [EXAMPLE_STRUCT] * 5 + [EXAMPLE_STRUCT_2]

EXAMPLE_STRUCT_SPEC1 = structured_tensor.StructuredTensorSpec(
    [], {
        "f1":
            tensor_spec.TensorSpec([], dtypes.int32),
        "f2":
            tensor_spec.TensorSpec([2, 2], dtypes.int32),
        "f3":
            structured_tensor.StructuredTensorSpec(
                [], {"f3_1": tensor_spec.TensorSpec([], dtypes.int32)}),
        "f4":
            structured_tensor.StructuredTensorSpec(
                [2], {
                    "f4_1": tensor_spec.TensorSpec([2], dtypes.int32),
                    "f4_2": tensor_spec.TensorSpec([2], dtypes.string)
                }),
        "f5":
            structured_tensor.StructuredTensorSpec(
                [2, 2], {"f5_1": tensor_spec.TensorSpec([2, 2], dtypes.int32)}),
    })


@test_util.run_all_in_graph_and_eager_modes
class StructuredTensorSliceTest(test_util.TensorFlowTestCase,
                                parameterized.TestCase):
  # [CLASS DOC]
  # Test case covering item access (slicing) on StructuredTensor, for both scalar and vector
  # structures, with and without an explicit type spec, plus the error cases. The whole class
  # runs in both graph and eager modes. It overrides the equality assertion so structured
  # tensors can be compared with each other and with plain Python lists and dicts.

  def assertAllEqual(self, a, b, msg=None):
    # [FUNCTION DOC]
    # Equality assertion extended to structured tensors. Inputs: a and b (values to compare;
    # structured tensors, dicts, lists, tuples or ordinary array-likes) and msg (optional failure
    # message). With no structured tensor involved it defers to the base implementation. Two
    # structured tensors must have compatible shapes, the same field names, and equal field
    # values (compared recursively). A structured tensor versus a python value: a scalar one is
    # compared to a mapping field by field; otherwise the python value must be a list or tuple
    # whose length is compatible with the first dimension, and elements are compared pairwise.
    # Fails the test (assertion error) on any mismatch; returns nothing.
    if not (isinstance(a, structured_tensor.StructuredTensor) or
            isinstance(b, structured_tensor.StructuredTensor)):
      super(StructuredTensorSliceTest, self).assertAllEqual(a, b, msg)
    elif (isinstance(a, structured_tensor.StructuredTensor) and
          isinstance(b, structured_tensor.StructuredTensor)):
      a_shape = tensor_shape.as_shape(a.shape)
      b_shape = tensor_shape.as_shape(b.shape)
      a_shape.assert_is_compatible_with(b_shape)
      self.assertEqual(set(a.field_names()), set(b.field_names()))
      for field in a.field_names():
        self.assertAllEqual(a.field_value(field), b.field_value(field))
    elif isinstance(b, structured_tensor.StructuredTensor):
      self.assertAllEqual(b, a, msg)
    else:
      if a.rank == 0:
        self.assertIsInstance(b, dict)
        self.assertEqual(set(a.field_names()), set(b))
        for (key, b_val) in b.items():
          a_val = a.field_value(key)
          self.assertAllEqual(a_val, b_val)
      else:
        self.assertIsInstance(b, (list, tuple))
        a.shape[:1].assert_is_compatible_with([len(b)])
        for i in range(len(b)):
          self.assertAllEqual(a[i], b[i])

  def _TestGetItem(self, struct, slice_spec, expected):
    # [FUNCTION DOC]
    # HELPER: Shared check used by the slicing tests.
    # Applies the item-access operation on the given structured tensor with the slice spec in
    # three forms (plain integers, integers wrapped as constants, integers wrapped as placeholders
    # with unknown values) and asserts each result equals expected (a nested python value).
    # Failure messages include the slice spec. Returns nothing.
    """Helper function for testing StructuredTensor.__getitem__.

    Checks that calling `struct.__getitem__(slice_spec) returns the expected
    value.  Checks three different configurations for each slice spec:

      * Call __getitem__ with the slice spec as-is (with int values)
      * Call __getitem__ with int values in the slice spec wrapped in
        `tf.constant()`.
      * Call __getitem__ with int values in the slice spec wrapped in
        `tf.compat.v1.placeholder()` (so value is not known at graph
        construction time).

    Args:
      struct: The StructuredTensor to test.
      slice_spec: The slice spec.
      expected: The expected value of struct.__getitem__(slice_spec), as a
        python list.
    """
    tensor_slice_spec1 = _make_tensor_slice_spec(slice_spec, True)
    tensor_slice_spec2 = _make_tensor_slice_spec(slice_spec, False)
    value1 = struct.__getitem__(slice_spec)
    value2 = struct.__getitem__(tensor_slice_spec1)
    value3 = struct.__getitem__(tensor_slice_spec2)
    self.assertAllEqual(value1, expected, "slice_spec=%s" % (slice_spec,))
    self.assertAllEqual(value2, expected, "slice_spec=%s" % (slice_spec,))
    self.assertAllEqual(value3, expected, "slice_spec=%s" % (slice_spec,))

  @parameterized.parameters([
      # Simple indexing
      (SLICE_BUILDER["f1"], EXAMPLE_STRUCT["f1"]),
      (SLICE_BUILDER["f2"], EXAMPLE_STRUCT["f2"]),
      (SLICE_BUILDER["f3"], EXAMPLE_STRUCT["f3"]),
      (SLICE_BUILDER["f4"], EXAMPLE_STRUCT["f4"]),
      (SLICE_BUILDER["f5"], EXAMPLE_STRUCT["f5"]),
      # Multidimensional indexing
      (SLICE_BUILDER["f2", 1], EXAMPLE_STRUCT["f2"][1]),
      (SLICE_BUILDER["f3", "f3_1"], EXAMPLE_STRUCT["f3"]["f3_1"]),
      (SLICE_BUILDER["f4", 1], EXAMPLE_STRUCT["f4"][1]),
      (SLICE_BUILDER["f4", 1, "f4_2"], EXAMPLE_STRUCT["f4"][1]["f4_2"]),
      (SLICE_BUILDER["f5", 0, 1], EXAMPLE_STRUCT["f5"][0][1]),
      (SLICE_BUILDER["f5", 0, 1, "f5_1"], EXAMPLE_STRUCT["f5"][0][1]["f5_1"]),
      # Multidimensional slicing
      (SLICE_BUILDER["f2", 1:], EXAMPLE_STRUCT["f2"][1:]),
      (SLICE_BUILDER["f4", :1], EXAMPLE_STRUCT["f4"][:1]),
      (SLICE_BUILDER["f4", 1:, "f4_2"], [b"b"]),
      (SLICE_BUILDER["f4", :, "f4_2"], [b"a", b"b"]),
      (SLICE_BUILDER["f5", :, :, "f5_1"], [[1, 2], [3, 4]]),
      # Slicing over multiple keys
      (SLICE_BUILDER[:], EXAMPLE_STRUCT),
      # List-valued key.
      (["f2", 1], EXAMPLE_STRUCT["f2"][1]),
  ])
  def testGetitemFromScalarStruct(self, slice_spec, expected):
    # [FUNCTION DOC]
    # Parameterized test: slicing a scalar structured tensor built from the example nested
    # struct. Cases cover simple field lookup, multi-dimensional indexing into fields and nested
    # structures, slices over inner dimensions, the full slice and a list-valued key. The
    # tensor is built twice, once with inferred ragged types and once with an explicit type spec
    # that makes dense tensors, and in both cases the result must equal the expected value.
    # By default, lists are converted to RaggedTensors.
    struct = structured_tensor.StructuredTensor.from_pyval(EXAMPLE_STRUCT)
    self._TestGetItem(struct, slice_spec, expected)

    # Using an explicit TypeSpec, we can convert them to Tensors instead.
    struct2 = structured_tensor.StructuredTensor.from_pyval(
        EXAMPLE_STRUCT, EXAMPLE_STRUCT_SPEC1)
    self._TestGetItem(struct2, slice_spec, expected)

  @parameterized.parameters([
      (SLICE_BUILDER[2], EXAMPLE_STRUCT_VECTOR[2]),
      (SLICE_BUILDER[5], EXAMPLE_STRUCT_VECTOR[5]),
      (SLICE_BUILDER[-2], EXAMPLE_STRUCT_VECTOR[-2]),
      (SLICE_BUILDER[-1], EXAMPLE_STRUCT_VECTOR[-1]),
      (SLICE_BUILDER[2, "f1"], EXAMPLE_STRUCT_VECTOR[2]["f1"]),
      (SLICE_BUILDER[-1, "f1"], EXAMPLE_STRUCT_VECTOR[-1]["f1"]),
      (SLICE_BUILDER[5:], EXAMPLE_STRUCT_VECTOR[5:]),
      (SLICE_BUILDER[3:, "f1"], [1, 1, 5]),
      (SLICE_BUILDER[::2, "f1"], [1, 1, 1]),
      (SLICE_BUILDER[1::2, "f1"], [1, 1, 5]),
      (SLICE_BUILDER[4:, "f5", 0, 1, "f5_1"], [2, 7], True),
      (SLICE_BUILDER[4:, "f5", :, :, "f5_1"],
       [[[1, 2], [3, 4]], [[6, 7], [8, 9]]]),
  ])  # pyformat: disable
  def testGetitemFromVectorStruct(self,
                                  slice_spec,
                                  expected,
                                  test_requires_typespec=False):
    # [FUNCTION DOC]
    # Parameterized test: slicing a vector of six example structs. Cases cover integer indices
    # (including negative), field selection after an index, slices with start or step, and slices
    # reaching into the nested matrix field. When test_requires_typespec (optional, default false)
    # is true the inferred-type variant is skipped, since that case only works with explicit
    # specs; otherwise both the inferred variant and the variant built with a batched type spec
    # are checked against the expected value.
    # By default, lists are converted to RaggedTensors.
    if not test_requires_typespec:
      struct_vector = structured_tensor.StructuredTensor.from_pyval(
          EXAMPLE_STRUCT_VECTOR)
      self._TestGetItem(struct_vector, slice_spec, expected)

    # Using an explicit TypeSpec, we can convert them to Tensors instead.
    struct_vector2 = structured_tensor.StructuredTensor.from_pyval(
        EXAMPLE_STRUCT_VECTOR, EXAMPLE_STRUCT_SPEC1._batch(6))
    self._TestGetItem(struct_vector2, slice_spec, expected)

  # TODO(edloper): Add tests for slicing from matrix StructuredTensors.

  @parameterized.parameters([
      (SLICE_BUILDER[:2], r"Key for indexing a StructuredTensor must be "
       r"a string or a full slice \(':'\)"),
      (SLICE_BUILDER["f4", ...], r"Slicing not supported for Ellipsis"),
      (SLICE_BUILDER["f4", None], r"Slicing not supported for tf.newaxis"),
      (SLICE_BUILDER["f4", :, 0],
       r"Key for indexing a StructuredTensor must be a string"),
  ])
  def testGetItemError(self, slice_spec, error, exception=ValueError):
    # [FUNCTION DOC]
    # Parameterized test: invalid keys on a scalar structured tensor must raise errors. Cases:
    # a non-full slice, an ellipsis, a new-axis marker, and a non-string key after a slice. The
    # expected error message pattern is error, and exception (default value error) is the type.
    # Asserts the item-access operation raises that exception with a matching message.
    struct = structured_tensor.StructuredTensor.from_pyval(EXAMPLE_STRUCT)
    with self.assertRaisesRegex(exception, error):
      struct.__getitem__(slice_spec)

  @parameterized.parameters([
      (SLICE_BUILDER[:, 1],
       r"Key for indexing a StructuredTensor must be a string"),
  ])
  def testGetItemFromVectorError(self, slice_spec, error, exception=ValueError):
    # [FUNCTION DOC]
    # Parameterized test: invalid keys on a vector structured tensor. The case slices all rows
    # and then uses an integer for the field key, which must raise the exception type given
    # (default value error) with a message matching error (key must be a string). The tensor is
    # built from the example vector of structs. Returns nothing.
    struct = structured_tensor.StructuredTensor.from_pyval(
        EXAMPLE_STRUCT_VECTOR)
    with self.assertRaisesRegex(exception, error):
      struct.__getitem__(slice_spec)


if __name__ == "__main__":
  googletest.main()
