# [MODULE DOC]
# Tests that RaggedTensor and RaggedTensorSpec accept custom "supported value types". Defines a
# test-only extension type (WrappedTensor) and a global op dispatcher that unwraps it, then tests
# construction, with_values, elementwise ops, slicing, spec serialization, component specs,
# compatibility, conversion to list and spec inference. Import-time side effects: registers the
# dispatcher globally and registers WrappedTensor as a supported ragged value type.
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
"""Tests for RaggedTensor supported value types."""

from absl.testing import parameterized

from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.ops.ragged import ragged_tensor_test_ops as test_ops
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensorSpec
from tensorflow.python.platform import googletest
from tensorflow.python.util import dispatch


class WrappedTensor(extension_type.BatchableExtensionType):
  # [CLASS DOC]
  # HELPER: test-only batchable extension type wrapping a single tensor field named value, used
  # to verify that ragged tensors can hold non-Tensor value types. Exposes shape, dtype,
  # indexing and set_shape by delegating to the wrapped tensor. Has a nested Spec type.
  value: tensor.Tensor

  @property
  def shape(self):
    # [FUNCTION DOC]
    # HELPER: delegation for the wrapped value. Returns the static shape of the wrapped tensor.
    return self.value.shape

  @property
  def dtype(self):
    # [FUNCTION DOC]
    # HELPER: delegation for the wrapped value. Returns the dtype of the wrapped tensor.
    return self.value.dtype

  def __getitem__(self, idx):
    # [FUNCTION DOC]
    # HELPER: delegation for the wrapped value. Indexes or slices the wrapped tensor with the
    # given index and returns the result wrapped in a new WrappedTensor.
    return WrappedTensor(self.value.__getitem__(idx))

  def set_shape(self, shape):
    # [FUNCTION DOC]
    # HELPER: delegation for the wrapped value. Calls set_shape on the wrapped tensor with the
    # given shape, which updates static shape information, and returns what that call returns
    # (nothing).
    return self.value.set_shape(shape)

  class Spec(type_spec.TypeSpec):
    # [CLASS DOC]
    # HELPER: nested type specification for WrappedTensor, used as the flat-values spec in the
    # spec tests. KNOWN ISSUE: its shape and dtype properties read an attribute named value that
    # a plain type spec does not define, so accessing them would fail unless provided elsewhere.

    @property
    def shape(self):
      # [FUNCTION DOC]
      # KNOWN ISSUE: reads the value attribute, which this spec class does not define, so it
      # likely fails if accessed (it appears to be copied from the wrapper class).
      # Intended to return the shape of the wrapped value.
      return self.value.shape

    @property
    def dtype(self):
      # [FUNCTION DOC]
      # KNOWN ISSUE: reads the value attribute, which this spec class does not define, so it
      # likely fails if accessed (it appears to be copied from the wrapper class).
      # Intended to return the dtype of the wrapped value.
      return self.value.dtype


class WrappedTensorOpDispatcher(dispatch.GlobalOpDispatcher):
  # [CLASS DOC]
  # HELPER: global op dispatcher that lets any TensorFlow op accept WrappedTensor arguments. It
  # unwraps them, runs the op on plain tensors and rewraps the result, except for ops listed in
  # OPS_THAT_RETURN_UNTRACED_RESULTS (shape queries and rank assertions) whose results stay
  # plain. Registered globally at import time.
  """Global op dispatcher for WrappedTensor."""

  # For these ops, just return plain Tensors (not WrappedTensors).
  OPS_THAT_RETURN_UNTRACED_RESULTS = (array_ops.shape, array_ops.shape_v2,
                                      check_ops.assert_rank_at_least)

  def call_op(self, op, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: invokes the given op with the given positional and keyword arguments and returns
    # its result; a hook point separated from handle.
    return op(*args, **kwargs)

  def handle(self, op, args, kwargs):
    # [FUNCTION DOC]
    # Dispatcher entry point. Inputs: the op, its positional argument list and keyword mapping.
    # Returns the not-supported marker unless some argument is (or directly contains) a
    # WrappedTensor. Otherwise unwraps all arguments, calls the op, and returns the plain result
    # for the untraced-result ops, else the result wrapped in a WrappedTensor.
    # Dispatcher only applies if at least one arg is a WrappedTensor.
    if not (any(self.is_wrapped_tensor_arg(x) for x in args) or
            any(self.is_wrapped_tensor_arg(x) for x in kwargs.values())):
      return self.NOT_SUPPORTED

    args = [self.unwrap(v) for v in args]
    kwargs = dict([(k, self.unwrap(v)) for (k, v) in kwargs.items()])
    value = self.call_op(op, *args, **kwargs)
    if op in self.OPS_THAT_RETURN_UNTRACED_RESULTS:
      return value
    else:
      return WrappedTensor(value)

  def unwrap(self, value):
    # [FUNCTION DOC]
    # HELPER: returns the wrapped tensor for a WrappedTensor, a same-typed list or tuple with
    # each element unwrapped recursively, or the value unchanged otherwise.
    if isinstance(value, WrappedTensor):
      return value.value
    elif isinstance(value, (list, tuple)):
      return type(value)([self.unwrap(v) for v in value])
    else:
      return value

  def is_wrapped_tensor_arg(self, value):
    # [FUNCTION DOC]
    # HELPER: returns true if the value is a WrappedTensor or a list or tuple directly
    # containing one (only one level deep); false otherwise.
    if isinstance(value, WrappedTensor):
      return True
    if isinstance(value, (list, tuple)):
      if any(isinstance(x, WrappedTensor) for x in value):
        return True
    return False


WrappedTensorOpDispatcher().register()
ragged_tensor._add_supported_value_type(WrappedTensor)


# pylint: disable=g-complex-comprehension
@test_util.run_all_in_graph_and_eager_modes
class RaggedTensorSupportedValuesTest(test_util.TensorFlowTestCase,
                                      parameterized.TestCase):
  # [CLASS DOC]
  # Tests that RaggedTensor works with WrappedTensor values: construction, replacing values,
  # elementwise unary and binary ops, and slicing. Run in graph and eager modes.

  def assertAllTensorsEqual(self, list1, list2):
    # [FUNCTION DOC]
    # HELPER: asserts two sequences of tensors have the same length and are equal pairwise.
    self.assertLen(list1, len(list2))
    for (t1, t2) in zip(list1, list2):
      self.assertAllEqual(t1, t2)

  def testConstruction(self):
    # [FUNCTION DOC]
    # Scenario: a WrappedTensor of eight strings. Action: build ragged tensors with each
    # row-partitioning factory (splits, starts, limits, lengths, uniform row length). Expected:
    # values stay a WrappedTensor holding the original tensor, and the partitioning accessor
    # returns the supplied values.
    tensor_values = constant_op.constant(
        ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    values = WrappedTensor(tensor_values)

    row_splits = constant_op.constant([0, 2, 2, 5, 6, 8], dtypes.int64)
    rt = RaggedTensor.from_row_splits(values, row_splits)
    self.assertIsInstance(rt.values, WrappedTensor)
    self.assertAllEqual(rt.values.value, tensor_values)
    self.assertAllEqual(rt.row_splits, row_splits)

    row_starts = constant_op.constant([0, 2, 2, 5, 6], dtypes.int64)
    rt = RaggedTensor.from_row_starts(values, row_starts)
    self.assertIsInstance(rt.values, WrappedTensor)
    self.assertAllEqual(rt.values.value, tensor_values)
    self.assertAllEqual(rt.row_starts(), row_starts)

    row_limits = constant_op.constant([2, 2, 5, 6, 8], dtypes.int64)
    rt = RaggedTensor.from_row_limits(values, row_limits)
    self.assertIsInstance(rt.values, WrappedTensor)
    self.assertAllEqual(rt.values.value, tensor_values)
    self.assertAllEqual(rt.row_limits(), row_limits)

    row_lengths = constant_op.constant([2, 0, 3, 1, 2], dtypes.int64)
    rt = RaggedTensor.from_row_lengths(values, row_lengths)
    self.assertIsInstance(rt.values, WrappedTensor)
    self.assertAllEqual(rt.values.value, tensor_values)
    self.assertAllEqual(rt.row_lengths(), row_lengths)

    rt = RaggedTensor.from_uniform_row_length(values, 4)
    self.assertIsInstance(rt.values, WrappedTensor)
    self.assertAllEqual(rt.values.value, tensor_values)
    self.assertAllEqual(rt.uniform_row_length, 4)

  def testWithValues(self):
    # [FUNCTION DOC]
    # Scenario: a two-level ragged tensor over wrapped strings. Action: replace its values with a
    # plain integer tensor and with a wrapped integer tensor. Expected: the new values equal the
    # given ones and the wrapped case remains a WrappedTensor.
    tensor_values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    values = WrappedTensor(tensor_values)
    nested_row_splits = [[0, 2, 5], [0, 2, 2, 5, 6, 7]]
    rt = RaggedTensor.from_nested_row_splits(values, nested_row_splits)

    tensor_int = constant_op.constant([1, 2, 3, 4, 5])
    rt_int = rt.with_values(tensor_int)
    self.assertAllEqual(rt_int.values, tensor_int)

    rt_wrapped_int = rt.with_values(WrappedTensor(tensor_int))
    self.assertIsInstance(rt_wrapped_int.values, WrappedTensor)
    self.assertAllEqual(rt_wrapped_int.values.value, tensor_int)

  def testWithFlatValues(self):
    # [FUNCTION DOC]
    # Scenario: a two-level ragged tensor over wrapped strings. Action: replace the flat values
    # with a plain integer tensor and with a wrapped one. Expected: flat values equal the given
    # tensors and the wrapped case remains a WrappedTensor.
    tensor_values = constant_op.constant(['a', 'b', 'c', 'd', 'e', 'f', 'g'])
    values = WrappedTensor(tensor_values)
    nested_row_splits = [[0, 2, 5], [0, 2, 2, 5, 6, 7]]
    rt = RaggedTensor.from_nested_row_splits(values, nested_row_splits)

    tensor_int = constant_op.constant([1, 2, 3, 4, 5, 6, 7])
    rt_int = rt.with_flat_values(tensor_int)
    self.assertAllEqual(rt_int.flat_values, tensor_int)

    rt_wrapped_int = rt.with_flat_values(WrappedTensor(tensor_int))
    self.assertIsInstance(rt_wrapped_int.flat_values, WrappedTensor)
    self.assertAllEqual(rt_wrapped_int.flat_values.value, tensor_int)

  @parameterized.parameters(
      #=========================================================================
      # Test each unary op.
      #=========================================================================
      [{'x': ([[-2.0, 3.0], [-3.0]]), 'op': op}
       for op in test_ops.UNARY_FLOAT_OPS] +
      [{'x': ([[True, False], [True]]),
        'op': op}
       for op in test_ops.UNARY_BOOL_OPS] +
      [{'x': [[18, 512], [12412]],
        'x_dtype': dtypes.int32,
        'op': op}
       for op in test_ops.UNARY_INT_OPS] +
      [{'x': ([['abcd', 'efgh'], ['aabbccdd']]),
        'op': op}
       for op in test_ops.UNARY_STRING_OPS] +
      [
          {'op': clip_ops.clip_by_value,
           'x': ([[-2.0, 3.0], [-3.0]]),
           'clip_value_min': 0.1, 'clip_value_max': 4.0},
          {'op': math_ops.cast,
           'x': ([[-2.0, 3.0], [-3.0]]),
           'dtype': dtypes.int32},
          {'op': math_ops.saturate_cast,
           'x': ([[-2.0, 3.0], [-3.0]]),
           'dtype': dtypes.int32},
          {'op': string_ops.string_to_hash_bucket,
           'x': (
               [['abcd', 'efgh'], ['aabbccdd']]),
           'num_buckets': 1000},
          {'op': string_ops.string_to_hash_bucket_fast,
           'x': (
               [['abcd', 'efgh'], ['aabbccdd']]),
           'num_buckets': 1000},
          {'op': string_ops.string_to_hash_bucket_strong,
           'x': (
               [['abcd', 'efgh'], ['aabbccdd']]),
           'num_buckets': 1000,
           'key': [1231, 12512]},
          {'op': string_ops.string_to_number,
           'x': ([['-2.0', '3.0'], ['-3.0']])},
          {'op': string_ops.regex_full_match,
           'x': ([['hello', '123'], ['1+1']]),
           'pattern': r'\w+'},
          {'op': string_ops.regex_replace,
           'x': ([['hello', '123'], ['1+1']]),
           'pattern': r'\d',
           'rewrite': '#'},
          {'op': string_ops.substr,
           'x': ([['hello', '123'], ['1+1']]),
           'pos': 2, 'len': 3},
          {'op': array_ops.check_numerics,
           'x': ([[-2.0, 3.0], [-3.0]]),
           'message': 'check-numerics'},
          {'op': nn_ops.dropout,
           'x': ([[-2.0, 3.0], [-3.0]]),
           'rate': 0.5,
           'seed': 1},
          {'op': array_ops.expand_dims_v2,
           'x': ([[-2.0, 3.0], [-3.0]]),
           'axis': -1},
      ])  # pyformat: disable
  def testUnaryElementwiseOp(self,
                             x,
                             x_dtype=None,
                             op=math_ops.abs,
                             **extra_args):
    # [FUNCTION DOC]
    # Scenario: each unary op (float, bool, int, string ops and several ops needing extra
    # arguments) applied to a ragged tensor and to the same data with wrapped flat values.
    # Inputs: nested list x, optional dtype, the op (default absolute value), extra keyword
    # arguments for the op. Action: seed randomness identically, apply the op to both versions.
    # Expected: the wrapped result has WrappedTensor flat values equal to the plain result's, and
    # equal row splits. Side effect: sets the global random seed.
    x = ragged_factory_ops.constant(x, x_dtype)
    wrapped_x = ragged_tensor.RaggedTensor.from_nested_row_splits(
        WrappedTensor(x.flat_values), x.nested_row_splits)
    test_util.random_seed.set_seed(1234)
    res = op(x, **extra_args)
    test_util.random_seed.set_seed(1234)
    wrapped_res = op(wrapped_x, **extra_args)
    self.assertIsInstance(wrapped_res.flat_values, WrappedTensor)
    self.assertAllEqual(wrapped_res.flat_values.value, res.flat_values)
    self.assertAllTensorsEqual(wrapped_res.nested_row_splits,
                               res.nested_row_splits)

  @parameterized.parameters(
      #=========================================================================
      # Test each binary op.
      #=========================================================================
      [{'x': [[-2.0, 3.0], [-3.0]],
        'y': [[5.0, 1.0], [12.0]],
        'op': op}
       for op in test_ops.BINARY_FLOAT_OPS] +
      [{'x': [[-2, 3], [-3]],
        'y': [[5, 1], [12]],
        'op': op}
       for op in test_ops.BINARY_INT_OPS] +
      [{'x': [[True, True], [False]],
        'y': [[False, True], [False]],
        'op': op}
       for op in test_ops.BINARY_BOOL_OPS]
      )  # pyformat: disable
  def testBinaryElementwiseOp(self, x, y, op=math_ops.add):
    # [FUNCTION DOC]
    # Scenario: each binary op (float, int, bool) on two ragged operands, once plain and once
    # with wrapped flat values. Inputs: nested lists x and y and the op (default addition).
    # Expected: the wrapped result has WrappedTensor flat values equal to the plain result's and
    # equal row splits.
    x = ragged_factory_ops.constant(x)
    y = ragged_factory_ops.constant(y)
    wrapped_x = ragged_tensor.RaggedTensor.from_nested_row_splits(
        WrappedTensor(x.flat_values), x.nested_row_splits)
    wrapped_y = ragged_tensor.RaggedTensor.from_nested_row_splits(
        WrappedTensor(y.flat_values), y.nested_row_splits)
    res = op(x, y)
    wrapped_res = op(wrapped_x, wrapped_y)
    self.assertIsInstance(wrapped_res.flat_values, WrappedTensor)
    self.assertAllEqual(wrapped_res.flat_values.value, res.flat_values)
    self.assertAllTensorsEqual(wrapped_res.nested_row_splits,
                               res.nested_row_splits)

  @parameterized.parameters(
      (lambda x: x[1:], True),
      (lambda x: x[0], False),
      (lambda x: x[:], True),
      (lambda x: x[0, :], False),
      (lambda x: x[...], True),
      (lambda x: x[1:2, ...], True),
      (lambda x: x[..., 1:2], True),
      (lambda x: x[0:2, ::2], True),
  )
  def testSlicing(self, slice_fn, is_ragged_output):
    # [FUNCTION DOC]
    # Scenario: a ragged tensor with wrapped values and an identical plain one. Inputs: a slicing
    # function and whether it yields a ragged result. Expected: for ragged output the sliced
    # wrapped tensor is ragged with WrappedTensor flat values and the same values and row splits
    # as the plain slice; otherwise the result is a WrappedTensor equal to the plain slice.
    tensor_values = constant_op.constant([[1.0, 2], [3, 4], [5, 6], [7, 8]])
    row_splits = constant_op.constant([0, 2, 3, 4], dtypes.int32)
    raw_rt = RaggedTensor.from_row_splits(tensor_values, row_splits)

    values = WrappedTensor(tensor_values)
    rt = RaggedTensor.from_row_splits(values, row_splits)

    res = slice_fn(rt)
    raw_res = slice_fn(raw_rt)
    if is_ragged_output:
      self.assertIsInstance(res, RaggedTensor)
      self.assertIsInstance(res.flat_values, WrappedTensor)
      self.assertAllEqual(res.flat_values.value, raw_res.flat_values)
      self.assertAllTensorsEqual(res.nested_row_splits,
                                 raw_res.nested_row_splits)
    else:
      self.assertIsInstance(res, WrappedTensor)
      self.assertAllEqual(res.value, raw_res)


@test_util.run_all_in_graph_and_eager_modes
class RaggedTensorSpecSupportedValuesTest(test_util.TensorFlowTestCase,
                                          parameterized.TestCase):
  # [CLASS DOC]
  # Tests RaggedTensorSpec with a WrappedTensor flat-values spec: construction, serialization,
  # component specs, component conversion, compatibility, list conversion and spec inference.

  def assertAllTensorsEqual(self, list1, list2):
    # [FUNCTION DOC]
    # HELPER: asserts two sequences of tensors have the same length and are equal pairwise.
    self.assertLen(list1, len(list2))
    for (t1, t2) in zip(list1, list2):
      self.assertAllEqual(t1, t2)

  def testConstruction(self):
    # [FUNCTION DOC]
    # Scenario: build a spec with a wrapped flat-values spec. Expected: private and public
    # accessors report the given shape rank, dtype, row-splits dtype, ragged rank and flat-values
    # spec; building one with a dtype that differs from the flat-values spec dtype raises a
    # value error about the dtype mismatch.
    flat_values_spec = WrappedTensor.Spec(
        tensor.TensorSpec(shape=(None, 5), dtype=dtypes.float32))
    spec1 = RaggedTensorSpec(
        shape=None,
        dtype=dtypes.float32,
        ragged_rank=1,
        row_splits_dtype=dtypes.int64,
        flat_values_spec=flat_values_spec)
    self.assertIsNone(spec1._shape.rank)
    self.assertEqual(spec1._dtype, dtypes.float32)
    self.assertEqual(spec1._row_splits_dtype, dtypes.int64)
    self.assertEqual(spec1._ragged_rank, 1)
    self.assertEqual(spec1._flat_values_spec, flat_values_spec)

    self.assertIsNone(spec1.shape.rank)
    self.assertEqual(spec1.dtype, dtypes.float32)
    self.assertEqual(spec1.row_splits_dtype, dtypes.int64)
    self.assertEqual(spec1.ragged_rank, 1)
    self.assertEqual(spec1.flat_values_spec, flat_values_spec)

    with self.assertRaisesRegex(
        ValueError, 'dtype must be the same as flat_values_spec.dtype'
    ):
      spec1 = RaggedTensorSpec(
          shape=None,
          dtype=dtypes.float64,
          ragged_rank=1,
          row_splits_dtype=dtypes.int64,
          flat_values_spec=flat_values_spec,
      )

  @parameterized.parameters([
      (
          RaggedTensorSpec(
              ragged_rank=1,
              flat_values_spec=tensor.TensorSpec(None, dtypes.float32),
          ),
          (
              tensor_shape.TensorShape(None),
              dtypes.float32,
              1,
              dtypes.int64,
              tensor.TensorSpec(None, dtypes.float32),
          ),
      ),
      (
          RaggedTensorSpec(
              shape=(5, None, 5),
              ragged_rank=1,
              dtype=dtypes.float64,
              flat_values_spec=tensor.TensorSpec((5,), dtypes.float64),
          ),
          (
              tensor_shape.TensorShape((5, None, 5)),
              dtypes.float64,
              1,
              dtypes.int64,
              tensor.TensorSpec((5,), dtypes.float64),
          ),
      ),
  ])
  def testSerialize(self, rt_spec, expected):
    # [FUNCTION DOC]
    # Scenario: specs with a plain tensor flat-values spec. Action: serialize the spec.
    # Expected: the text representations of the result and the expected tuple (shape, dtype,
    # ragged rank, row-splits dtype, flat-values spec) are equal, since shape equality is unusual.
    serialization = rt_spec._serialize()
    # TensorShape has an unconventional definition of equality, so we can't use
    # assertEqual directly here.  But repr() is deterministic and lossless for
    # the expected values, so we can use that instead.
    self.assertEqual(repr(serialization), repr(expected))

  @parameterized.parameters([
      (RaggedTensorSpec(
          ragged_rank=0,
          shape=[5, 3],
          flat_values_spec=WrappedTensor.Spec(
              tensor.TensorSpec([5, 3], dtypes.float32))),
       [WrappedTensor.Spec(tensor.TensorSpec([5, 3], dtypes.float32))]),
      (RaggedTensorSpec(
          ragged_rank=1,
          flat_values_spec=WrappedTensor.Spec(
              tensor.TensorSpec([None, 3], dtypes.float32))),
       [
           WrappedTensor.Spec(
               tensor.TensorSpec([None, 3], dtypes.float32)),
           tensor.TensorSpec([None], dtypes.int64),
       ]),
      (RaggedTensorSpec(
          ragged_rank=2,
          dtype=dtypes.float64,
          flat_values_spec=WrappedTensor.Spec(
              tensor.TensorSpec([None, 3], dtypes.float64))),
       [
           WrappedTensor.Spec(
               tensor.TensorSpec([None, 3], dtypes.float64)),
           tensor.TensorSpec([None], dtypes.int64),
           tensor.TensorSpec([None], dtypes.int64),
       ]),
      (RaggedTensorSpec(
          shape=[5, None, None],
          dtype=dtypes.string,
          flat_values_spec=WrappedTensor.Spec(
              tensor.TensorSpec([None, 3], dtypes.string))),
       [
           WrappedTensor.Spec(tensor.TensorSpec([None, 3], dtypes.string)),
           tensor.TensorSpec([6], dtypes.int64),
           tensor.TensorSpec([None], dtypes.int64),
       ]),
  ])
  def testComponentSpecs(self, rt_spec, expected):
    # [FUNCTION DOC]
    # Scenario: specs with differing ragged ranks, dtypes and shapes. Expected: the component
    # specs are the wrapped flat-values spec followed by one int64 row-splits spec per ragged
    # level (with a static length when the outer dimension is known).
    self.assertEqual(rt_spec._component_specs, expected)

  @parameterized.parameters([
      {
          'rt_spec':
              RaggedTensorSpec(
                  shape=[3, None, None],
                  ragged_rank=1,
                  flat_values_spec=WrappedTensor.Spec(
                      tensor.TensorSpec(None, dtype=dtypes.float32))),
          'flat_values': [[1.0, 2.0], [3.0, 4.0]],
          'nested_row_splits': [[0, 1, 1, 2]],
      },
      {
          'rt_spec':
              RaggedTensorSpec(
                  shape=[2, None, None],
                  flat_values_spec=WrappedTensor.Spec(
                      tensor.TensorSpec(None, dtype=dtypes.float32))),
          'flat_values': [1.0, 2.0, 3.0, 4.0],
          'nested_row_splits': [[0, 2, 4], [0, 2, 3, 3, 4]],
      },
  ])
  def testToFromComponents(self, rt_spec, flat_values, nested_row_splits):
    # [FUNCTION DOC]
    # Scenario: a ragged tensor over wrapped float values with given nested row splits.
    # Action: convert to components and rebuild from them. Expected: the first component is the
    # wrapped values, the rest equal the row splits, and the rebuilt tensor has wrapped flat
    # values, equal row splits and the same dtype.
    wrapped_tensor = WrappedTensor(constant_op.constant(flat_values))
    rt = RaggedTensor.from_nested_row_splits(wrapped_tensor, nested_row_splits)
    components = rt_spec._to_components(rt)
    self.assertIsInstance(components[0], WrappedTensor)
    self.assertAllEqual(components[0].value, wrapped_tensor.value)
    self.assertAllTensorsEqual(components[1:], nested_row_splits)
    rt_reconstructed = rt_spec._from_components(components)
    self.assertIsInstance(rt_reconstructed.flat_values, WrappedTensor)
    self.assertAllEqual(
        rt_reconstructed.flat_values.value, wrapped_tensor.value
    )
    self.assertAllTensorsEqual(
        rt_reconstructed.nested_row_splits, rt.nested_row_splits
    )
    self.assertEqual(rt_reconstructed.dtype, rt.dtype)

  def testIsCompatibleWith(self):
    # [FUNCTION DOC]
    # Scenario: five specs differing in shape, dtype, ragged rank and flat-values spec.
    # Expected: only the first is compatible with the second (less specific shape); all other
    # pairs are incompatible; a plain tensor is incompatible with a wrapped-value spec while a
    # WrappedTensor value is compatible.
    spec1 = RaggedTensorSpec(
        [32, None, None],
        dtypes.float32,
        2,
        flat_values_spec=WrappedTensor.Spec(
            tensor.TensorSpec([None, None], dtypes.float32)
        ),
    )
    spec2 = RaggedTensorSpec(
        None,
        dtypes.float32,
        2,
        flat_values_spec=WrappedTensor.Spec(
            tensor.TensorSpec(None, dtypes.float32)
        ),
    )
    spec3 = RaggedTensorSpec(
        None,
        dtypes.int32,
        1,
        flat_values_spec=WrappedTensor.Spec(
            tensor.TensorSpec(None, dtypes.int32)
        ),
    )
    spec4 = RaggedTensorSpec(
        [None],
        dtypes.int32,
        0,
        flat_values_spec=WrappedTensor.Spec(
            tensor.TensorSpec(None, dtypes.int32)
        ),
    )
    spec5 = RaggedTensorSpec([None], dtypes.int32, 0)

    self.assertTrue(spec1.is_compatible_with(spec2))
    self.assertFalse(spec1.is_compatible_with(spec3))
    self.assertFalse(spec1.is_compatible_with(spec4))
    self.assertFalse(spec2.is_compatible_with(spec3))
    self.assertFalse(spec2.is_compatible_with(spec4))
    self.assertFalse(spec3.is_compatible_with(spec4))
    self.assertFalse(spec4.is_compatible_with(spec5))
    value = constant_op.constant([1, 2, 3])
    self.assertFalse(spec4.is_compatible_with(value))
    self.assertTrue(spec4.is_compatible_with(WrappedTensor(value)))

  def testToList(self):
    # [FUNCTION DOC]
    # Scenario (eager mode only): ragged tensor over wrapped strings. Expected: converting to a
    # list raises a value error when the wrapped type cannot be converted; it succeeds and equals
    # the expected nested list when the wrapped type provides a numpy method or a to_list method.
    with context.eager_mode():
      tensor_values = constant_op.constant(
          ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
      row_splits = constant_op.constant([0, 2, 2, 5, 6, 8], dtypes.int64)
      values = WrappedTensor(tensor_values)
      rt = RaggedTensor.from_row_splits(values, row_splits)
      expected = ragged_factory_ops.constant([['a', 'b'], [], ['c', 'd', 'e'],
                                              ['f'], ['g', 'h']]).to_list()

      with self.subTest('Raise on unsupported'):
        with self.assertRaisesRegex(
            ValueError,
            'values must be convertible to a list',
        ):
          _ = rt.to_list()

      with self.subTest('Value with numpy method'):

        class WrappedTensorWithNumpy(WrappedTensor):
          # [CLASS DOC]
          # HELPER: WrappedTensor subclass adding a numpy conversion method, used to show
          # list conversion works via numpy.

          def numpy(self):
            # [FUNCTION DOC]
            # HELPER: returns the wrapped tensor as a numpy array.
            return self.value.numpy()

        values = WrappedTensorWithNumpy(tensor_values)
        rt = RaggedTensor.from_row_splits(values, row_splits)
        self.assertEqual(rt.to_list(), expected)

      with self.subTest('Value with to_list method'):

        class WrappedTensorWithToList(WrappedTensor):
          # [CLASS DOC]
          # HELPER: WrappedTensor subclass adding a to_list method, used to show list
          # conversion works via that method.

          def to_list(self):
            # [FUNCTION DOC]
            # HELPER: returns the wrapped tensor's contents as a nested Python list.
            return self.value.numpy().tolist()

        values = WrappedTensorWithToList(tensor_values)
        rt = RaggedTensor.from_row_splits(values, row_splits)
        self.assertEqual(rt.to_list(), expected)

  def testFromValue(self):
    # [FUNCTION DOC]
    # Scenario: a ragged tensor with wrapped float values and int32 row splits. Action: infer a
    # type spec from the value. Expected: it equals the spec with shape four by unknown by two,
    # ragged rank one, int32 row-splits and a wrapped flat-values spec, and the trailing shape
    # of the spec matches the flat-values spec shape.
    tensor_values = constant_op.constant([[1.0, 2], [4, 5], [7, 8]])
    values = WrappedTensor(tensor_values)

    row_splits = constant_op.constant([0, 2, 3, 3, 3], dtypes.int32)
    rt = RaggedTensor.from_row_splits(values, row_splits)

    rt_spec = type_spec.type_spec_from_value(rt)
    self.assertEqual(
        rt_spec,
        RaggedTensorSpec(
            shape=[4, None, 2],
            dtype=dtypes.float32,
            ragged_rank=1,
            row_splits_dtype=dtypes.int32,
            flat_values_spec=WrappedTensor.Spec(
                tensor.TensorSpec([None, 2], dtypes.float32))))
    # Ensure the shape of flat_values_spec being consistent with the shape
    # of the RaggedTensor.
    self.assertEqual(rt_spec.shape[rt_spec.ragged_rank:],
                     rt_spec.flat_values_spec.shape)


if __name__ == '__main__':
  googletest.main()
