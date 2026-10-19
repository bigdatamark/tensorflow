# [MODULE DOC]
# Tests for running operations in eager execution mode: executing ops with various attribute
# kinds and single/multiple/list outputs, operator overloads, slicing, device placement and copy
# policies (GPU-only tests are skipped without a GPU), dtype matching for eager arguments, shape
# setting, conversion to primitive types, weak references and garbage collection of eager
# tensors, and per-context tensor caches. Runs the test main entry point when executed directly.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for operations in eager execution."""
import gc
import threading
import weakref

from absl.testing import parameterized
import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.eager import execute
from tensorflow.python.eager import test
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops


class OpsTest(test_util.TensorFlowTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case (with parameterized support) for eager op execution behaviors. No shared fixtures;
  # tests that need a GPU are decorated to run only when one is available.

  def testExecuteBasic(self):
    # [FUNCTION DOC]
    # Multiplies two scalar constants (3 and 5) and asserts the result equals 15.
    three = constant_op.constant(3)
    five = constant_op.constant(5)
    product = three * five
    self.assertAllEqual(15, product)

  @test_util.run_gpu_only
  def testMatMulGPU(self):
    # [FUNCTION DOC]
    # GPU-only. Places two 1x1 matrices on the GPU, multiplies them and asserts the result is 15.
    three = constant_op.constant([[3.]]).gpu()
    five = constant_op.constant([[5.]]).gpu()
    product = math_ops.matmul(three, five)
    self.assertEqual([[15.0]], product.numpy())

  def testExecuteStringAttr(self):
    # [FUNCTION DOC]
    # Runs a numeric check op with a string message attribute on 3.0 and asserts the output
    # equals the input value.
    three = constant_op.constant(3.0)
    checked_three = array_ops.check_numerics(three,
                                             message='just checking')
    self.assertEqual([[3]], checked_three.numpy())

  def testExecuteFloatAttr(self):
    # [FUNCTION DOC]
    # Runs approximate-equal on 3.0 and 2.8 with float tolerance 0.3 and asserts the result is true.
    three = constant_op.constant(3.0)
    almost_three = constant_op.constant(2.8)
    almost_equal = math_ops.approximate_equal(
        three, almost_three, tolerance=0.3)
    self.assertTrue(almost_equal)

  def testExecuteIntAttr(self):
    # [FUNCTION DOC]
    # Runs add-n (integer count attribute) on 3 and 4 and asserts the result equals 7.
    three = constant_op.constant(3)
    four = constant_op.constant(4)
    total = math_ops.add_n([three, four])
    self.assertAllEqual(7, total)

  def testExecuteBoolAttr(self):
    # [FUNCTION DOC]
    # Runs matrix multiply with the first-operand transpose boolean attribute on 1x1 matrices and
    # asserts the product is 15.
    three = constant_op.constant([[3]])
    five = constant_op.constant([[5]])
    product = math_ops.matmul(three, five, transpose_a=True)
    self.assertAllEqual([[15]], product)

  def testExecuteOneListOutput(self):
    # [FUNCTION DOC]
    # Splits a 2x3 matrix into three column slices along axis 1 and asserts each slice's content.
    split_dim = constant_op.constant(1)
    value = constant_op.constant([[0, 1, 2], [3, 4, 5]])
    x1, x2, x3 = array_ops.split(value, 3, axis=split_dim)
    self.assertAllEqual([[0], [3]], x1)
    self.assertAllEqual([[1], [4]], x2)
    self.assertAllEqual([[2], [5]], x3)

  def testGraphMode(self):
    # [FUNCTION DOC]
    # Inside a fresh graph in graph mode, creates a placeholder and asserts the graph holds
    # exactly one operation.
    graph = ops.Graph()
    with graph.as_default(), context.graph_mode():
      array_ops.placeholder(dtypes.int32)
    self.assertLen(graph.get_operations(), 1)

  # See comments on handling of int32 tensors on GPU in
  # EagerTensor.__init__.
  @test_util.run_gpu_only
  def testInt32CPUDefault(self):
    # [FUNCTION DOC]
    # GPU-only. Adds two int32 constants under a GPU device scope (int32 tensors stay on host)
    # and asserts the sum equals 3.
    with context.device('/gpu:0'):
      r = constant_op.constant(1) + constant_op.constant(2)
    self.assertAllEqual(r, 3)

  def testExecuteListOutputLen1(self):
    # [FUNCTION DOC]
    # Splits a matrix into one piece and asserts the result is a list of length one holding the
    # whole matrix.
    split_dim = constant_op.constant(1)
    value = constant_op.constant([[0, 1, 2], [3, 4, 5]])
    result = array_ops.split(value, 1, axis=split_dim)
    self.assertIsInstance(result, list)
    self.assertLen(result, 1)
    self.assertAllEqual([[0, 1, 2], [3, 4, 5]], result[0])

  def testExecuteListOutputLen0(self):
    # [FUNCTION DOC]
    # Unstacks an empty tensor into zero pieces and asserts the result is an empty list.
    empty = constant_op.constant([], dtype=dtypes.int32)
    result = array_ops_stack.unstack(empty, 0)
    self.assertIsInstance(result, list)
    self.assertEmpty(result)

  def testExecuteMultipleNonListOutput(self):
    # [FUNCTION DOC]
    # Runs list-difference, which has two non-list outputs, and asserts the result unpacks into
    # output and index values that are also accessible by field name, with expected contents.
    x = constant_op.constant([1, 2, 3, 4, 5, 6])
    y = constant_op.constant([1, 3, 5])
    result = array_ops.listdiff(x, y)
    out, idx = result
    self.assertIs(out, result.out)
    self.assertIs(idx, result.idx)
    self.assertAllEqual([2, 4, 6], out)
    self.assertAllEqual([1, 3, 5], idx)

  def testExecuteMultipleListOutput(self):
    # [FUNCTION DOC]
    # Runs the sparse split op (three list outputs) on a small sparse tensor split into two, and
    # asserts each output list has two entries, matches the named fields, and has the expected
    # indices, values and shapes per piece.
    split_dim = constant_op.constant(1, dtype=dtypes.int64)
    indices = constant_op.constant([[0, 2], [0, 4], [0, 5], [1, 0], [1, 1]],
                                   dtype=dtypes.int64)
    values = constant_op.constant([2, 3, 5, 7, 11])
    shape = constant_op.constant([2, 7], dtype=dtypes.int64)
    result = sparse_ops.gen_sparse_ops.sparse_split(
        split_dim,
        indices,
        values,
        shape,
        num_split=2)
    output_indices, output_values, output_shape = result
    self.assertLen(output_indices, 2)
    self.assertLen(output_values, 2)
    self.assertLen(output_shape, 2)
    self.assertEqual(output_indices, result.output_indices)
    self.assertEqual(output_values, result.output_values)
    self.assertEqual(output_shape, result.output_shape)
    self.assertAllEqual([[0, 2], [1, 0], [1, 1]], output_indices[0])
    self.assertAllEqual([[0, 0], [0, 1]], output_indices[1])
    self.assertAllEqual([2, 7, 11], output_values[0])
    self.assertAllEqual([3, 5], output_values[1])
    self.assertAllEqual([2, 4], output_shape[0])
    self.assertAllEqual([2, 3], output_shape[1])

  # TODO(josh11b): Test an op that has multiple outputs, some but not
  # all of which are lists. Examples: barrier_take_many (currently
  # unsupported since it uses a type list) or sdca_optimizer (I don't
  # have an example of legal inputs & outputs).

  def testComposition(self):
    # [FUNCTION DOC]
    # Adds a tensor to itself twice and asserts the dtype stays int32 and the result is 3.
    x = constant_op.constant(1, dtype=dtypes.int32)
    three_x = x + x + x
    self.assertEqual(dtypes.int32, three_x.dtype)
    self.assertAllEqual(3, three_x)

  def testOperatorOverrides(self):
    # [FUNCTION DOC]
    # Defines a helper that checks tensor operator overloads against NumPy results and runs it on
    # two pairs of integer lists.

    def ops_test(v1, v2):
      # [FUNCTION DOC]
      # HELPER: checks operator overloads for testOperatorOverrides.
      # Builds tensors from two lists and asserts negation, absolute value, arithmetic (power only
      # if all exponents are non-negative), comparisons, equality and inequality (shape handling
      # depends on whether equality semantics are enabled) and scalar-tensor indexing all match
      # the equivalent NumPy operations.
      a = constant_op.constant(v1)
      b = constant_op.constant(v2)

      self.assertAllEqual((-a), np.negative(v1))
      self.assertAllEqual(abs(b), np.absolute(v2))

      self.assertAllEqual((a + b), np.add(v1, v2))
      self.assertAllEqual((a - b), np.subtract(v1, v2))
      self.assertAllEqual((a * b), np.multiply(v1, v2))
      self.assertAllEqual((a * a), np.multiply(v1, v1))

      if all(x >= 0 for x in v2):
        self.assertAllEqual((a**b), np.power(v1, v2))
      self.assertAllEqual((a / b), np.true_divide(v1, v2))

      self.assertAllEqual((a / a), np.true_divide(v1, v1))
      self.assertAllEqual((a % b), np.mod(v1, v2))

      self.assertAllEqual((a < b), np.less(v1, v2))
      self.assertAllEqual((a <= b), np.less_equal(v1, v2))
      self.assertAllEqual((a > b), np.greater(v1, v2))
      self.assertAllEqual((a >= b), np.greater_equal(v1, v2))

      # TODO(b/120678848): Remove the else branch once we enable
      # tensor.Tensor._USE_EQUALITY by default.
      if tensor.Tensor._USE_EQUALITY:
        self.assertAllEqual((a == b), np.equal(v1, v2))
        self.assertAllEqual((a != b), np.not_equal(v1, v2))
      else:
        self.assertAllEqual((a == b), np.equal(v1, v2)[0])
        self.assertAllEqual((a != b), np.not_equal(v1, v2)[0])

      self.assertAllEqual(v1[0], a[constant_op.constant(0)])

    ops_test([1, 4, 8], [2, 3, 5])
    ops_test([1, -4, -5], [-2, 3, -6])

  def test_basic_slice(self):
    # [FUNCTION DOC]
    # Slices a 3x2x3 tensor with a variety of start/stop/stride combinations (including negative
    # strides) and asserts every result equals the NumPy slice.
    npt = np.arange(1, 19, dtype=np.float32).reshape(3, 2, 3)
    t = constant_op.constant(npt)

    self.assertAllEqual(npt[:, :, :], t[:, :, :])
    self.assertAllEqual(npt[::, ::, ::], t[::, ::, ::])
    self.assertAllEqual(npt[::1, ::1, ::1], t[::1, ::1, ::1])
    self.assertAllEqual(npt[::1, ::5, ::2], t[::1, ::5, ::2])
    self.assertAllEqual(npt[::-1, :, :], t[::-1, :, :])
    self.assertAllEqual(npt[:, ::-1, :], t[:, ::-1, :])
    self.assertAllEqual(npt[:, :, ::-1], t[:, :, ::-1])
    self.assertAllEqual(npt[-2::-1, :, ::1], t[-2::-1, :, ::1])
    self.assertAllEqual(npt[-2::-1, :, ::2], t[-2::-1, :, ::2])

  def testDegenerateSlices(self):
    # [FUNCTION DOC]
    # Slices with empty or contradictory intervals and strides and asserts results match NumPy.
    npt = np.arange(1, 19, dtype=np.float32).reshape(3, 2, 3)
    t = constant_op.constant(npt)
    # degenerate by offering a forward interval with a negative stride
    self.assertAllEqual(npt[0:-1:-1, :, :], t[0:-1:-1, :, :])
    # degenerate with a reverse interval with a positive stride
    self.assertAllEqual(npt[-1:0, :, :], t[-1:0, :, :])
    # empty interval in every dimension
    self.assertAllEqual(npt[-1:0, 2:2, 2:3:-1], t[-1:0, 2:2, 2:3:-1])

  def testEllipsis(self):
    # [FUNCTION DOC]
    # Slices a five-dimensional tensor using ellipsis at the start, middle, end, alone and
    # implicit, asserting results match NumPy.
    npt = np.array(
        [[[[[1, 2], [3, 4], [5, 6]]], [[[7, 8], [9, 10], [11, 12]]]]])
    t = constant_op.constant(npt)

    self.assertAllEqual(npt[0:], t[0:])
    # implicit ellipsis
    self.assertAllEqual(npt[0:, ...], t[0:, ...])
    # ellipsis alone
    self.assertAllEqual(npt[...], t[...])
    # ellipsis at end
    self.assertAllEqual(npt[0:1, ...], t[0:1, ...])
    # ellipsis at begin
    self.assertAllEqual(npt[..., 0:1], t[..., 0:1])
    # ellipsis at middle
    self.assertAllEqual(npt[0:1, ..., 0:1], t[0:1, ..., 0:1])

  def testShrink(self):
    # [FUNCTION DOC]
    # Indexes a five-dimensional tensor with integer indices (shrinking dimensions), with and
    # without ellipsis, and asserts results match NumPy.
    npt = np.array([[[[[1, 2, 4, 5], [5, 6, 7, 8], [9, 10, 11, 12]]],
                     [[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]]])
    t = constant_op.constant(npt)
    self.assertAllEqual(npt[:, :, :, :, 3], t[:, :, :, :, 3])
    self.assertAllEqual(npt[..., 3], t[..., 3])
    self.assertAllEqual(npt[:, 0], t[:, 0])
    self.assertAllEqual(npt[:, :, 0], t[:, :, 0])

  @test_util.run_gpu_only
  def testOpWithInputsOnDifferentDevices(self):
    # [FUNCTION DOC]
    # GPU-only. Reshapes a GPU tensor using a CPU shape input (the kernel needs the shape on host)
    # and asserts the result, copied to CPU, equals the expected column.
    # The GPU kernel for the Reshape op requires that the
    # shape input be on CPU.
    value = constant_op.constant([1., 2.]).gpu()
    shape = constant_op.constant([2, 1])
    reshaped = array_ops.reshape(value, shape)
    self.assertAllEqual([[1], [2]], reshaped.cpu())

  def testInt64(self):
    # [FUNCTION DOC]
    # Calls fill with a 64-bit integer shape tensor and asserts it produces two ones.
    # Fill requires the first input to be an int32 tensor.
    self.assertAllEqual(
        [1.0, 1.0],
        array_ops.fill(constant_op.constant([2], dtype=dtypes.int64),
                       constant_op.constant(1)))

  @test_util.run_gpu_only
  def testOutputOnHostMemory(self):
    # [FUNCTION DOC]
    # GPU-only. Takes the shape of a GPU tensor (output placed in host memory) and asserts it is
    # [1].
    # The Shape op kernel on GPU places the output in host memory.
    value = constant_op.constant([1.]).gpu()
    shape = array_ops.shape(value)
    self.assertEqual([1], shape.numpy())

  @test_util.run_gpu_only
  def testSilentCopy(self):
    # [FUNCTION DOC]
    # GPU-only. Temporarily swaps in a fresh eager context, sets the device policy to silent,
    # adds a CPU tensor and a GPU tensor, and asserts the sum is 2.0. Restores the original
    # context in a finally clause. Side effect: changes global context and device policy.
    # Temporarily replace the context
    # pylint: disable=protected-access
    old_context = context.context()
    context._set_context(context.Context())
    try:
      config.set_device_policy('silent')
      cpu_tensor = constant_op.constant(1.0)
      gpu_tensor = cpu_tensor.gpu()
      self.assertAllEqual(cpu_tensor + gpu_tensor, 2.0)
    finally:
      context._set_context(old_context)
    # pylint: enable=protected-access

  @test_util.run_gpu_only
  def testSoftPlacement(self):
    # [FUNCTION DOC]
    # GPU-only. Swaps in a fresh context with silent device policy and soft placement on, adds
    # two CPU tensors and asserts the result was placed on the first GPU. Restores the original
    # context afterwards.
    # Temporarily replace the context
    # pylint: disable=protected-access
    old_context = context.context()
    context._set_context(context.Context())
    try:
      config.set_device_policy('silent')
      config.set_soft_device_placement(True)
      cpu_tensor = constant_op.constant(1.0)
      result = cpu_tensor + cpu_tensor
      self.assertEqual(result.device,
                       '/job:localhost/replica:0/task:0/device:GPU:0')
    finally:
      context._set_context(old_context)
    # pylint: enable=protected-access

  def testRandomUniform(self):
    # [FUNCTION DOC]
    # Draws a scalar from a uniform random op with an empty tensor shape and asserts it is a
    # float32 scalar; then draws with tensor bounds 5 and 6 and asserts the value lies in range.
    scalar_shape = constant_op.constant([], dtype=dtypes.int32)

    x = random_ops.random_uniform(scalar_shape)
    self.assertEqual(0, x.shape.ndims)
    self.assertEqual(dtypes.float32, x.dtype)

    x = random_ops.random_uniform(
        scalar_shape, minval=constant_op.constant(5.),
        maxval=constant_op.constant(6.))
    self.assertLess(x, 6)
    self.assertGreaterEqual(x, 5)

  def testArgsToMatchingEagerDefault(self):
    # [FUNCTION DOC]
    # Exercises the eager helper that picks a common dtype for arguments from an allowed list
    # and a default. Asserts: ordinary integers follow standard conversion regardless of
    # default; a value too large for 32 bits picks 64-bit; when ordinary conversion fails the
    # default is used as a hint (unsigned types); an empty input returns the default; and string
    # arguments choose the string type instead of the default.
    # Uses default
    ctx = context.context()
    allowed_dtypes = [dtypes.int32, dtypes.int64]

    # Follows standard int conversion rules
    t, r = execute.args_to_matching_eager([[3, 4]], ctx, allowed_dtypes,
                                          dtypes.int32)
    self.assertEqual(t, dtypes.int32)
    self.assertEqual(r[0].dtype, dtypes.int32)
    t, r = execute.args_to_matching_eager([[3, 4]], ctx, allowed_dtypes,
                                          dtypes.int64)
    self.assertEqual(t, dtypes.int32)
    self.assertEqual(r[0].dtype, dtypes.int32)
    # Use int64 since it is a better fit
    t, r = execute.args_to_matching_eager([[2**48]], ctx, allowed_dtypes,
                                          dtypes.int32)
    self.assertEqual(t, dtypes.int64)
    self.assertEqual(r[0].dtype, dtypes.int64)

    # When the regular tensor conversion fails, then use the default type as a
    # hint.
    allowed_dtypes = [dtypes.uint32, dtypes.uint32]
    t, r = execute.args_to_matching_eager([[3, 4]], ctx, allowed_dtypes,
                                          dtypes.uint32)
    self.assertEqual(t, dtypes.uint32)
    self.assertEqual(r[0].dtype, dtypes.uint32)
    t, r = execute.args_to_matching_eager([[3, 4]], ctx, allowed_dtypes,
                                          dtypes.uint64)
    self.assertEqual(t, dtypes.uint64)
    self.assertEqual(r[0].dtype, dtypes.uint64)

    t, r = execute.args_to_matching_eager([], ctx, allowed_dtypes, dtypes.int64)
    self.assertEqual(t, dtypes.int64)

    # Doesn't use default
    allowed_dtypes = [dtypes.int32, dtypes.string]
    t, r = execute.args_to_matching_eager([['string', 'arg']], ctx,
                                          allowed_dtypes, dtypes.int32)
    self.assertEqual(t, dtypes.string)
    self.assertEqual(r[0].dtype, dtypes.string)

  def testIdentity(self):
    # [FUNCTION DOC]
    # Applies identity to the Python number 2 and asserts the result equals 2.
    self.assertAllEqual(2, array_ops.identity(2))

  @test_util.run_gpu_only
  def testIdentityOnVariable(self):
    # [FUNCTION DOC]
    # GPU-only. Creates a boolean resource variable on the GPU and asserts identity of it yields
    # true.
    with context.device('/gpu:0'):
      v = resource_variable_ops.ResourceVariable(True)
    self.assertAllEqual(True, array_ops.identity(v))

  def testIncompatibleSetShape(self):
    # [FUNCTION DOC]
    # Asserts that setting an incompatible static shape on a scalar eager tensor raises a value
    # error.
    x = constant_op.constant(1)
    with self.assertRaises(ValueError):
      x.set_shape((1, 2))

  def testCompatibleSetShape(self):
    # [FUNCTION DOC]
    # Sets a partially unknown but compatible shape on a 1x2 tensor and asserts the shape stays
    # 1 by 2.
    x = constant_op.constant([[1, 2]])
    x.set_shape(tensor_shape.TensorShape([None, 2]))
    self.assertEqual(x.get_shape(), (1, 2))

  @parameterized.named_parameters(
      ('Tensor', lambda: constant_op.constant(1.3+1j)),
      ('Variable', lambda: resource_variable_ops.ResourceVariable(1.3+1j)))
  def testCastToPrimitiveTypesFrom(self, value_fn):
    # [FUNCTION DOC]
    # Parameterized over a complex constant tensor and a complex resource variable (value_fn is a
    # factory creating the value). Asserts conversion to integer gives 1 (type int), to float
    # gives about 1.3, and to complex gives about 1.3 plus 1i.
    x = value_fn()
    self.assertIsInstance(int(x), int)
    self.assertEqual(int(x), 1)
    self.assertIsInstance(float(x), float)
    self.assertAllClose(float(x), 1.3)
    self.assertIsInstance(complex(x), complex)
    self.assertAllClose(complex(x), 1.3+1j)

  def testCastNonScalarToPrimitiveTypesFails(self):
    # [FUNCTION DOC]
    # Asserts converting a two-element tensor to integer or float raises a type error.
    x = constant_op.constant([1.3, 2])
    with self.assertRaises(TypeError):
      int(x)
    with self.assertRaises(TypeError):
      float(x)

  def testRange(self):
    # [FUNCTION DOC]
    # Uses a scalar integer tensor as the upper bound of a Python range and asserts it yields
    # 0 and 1.
    x = constant_op.constant(2)
    self.assertEqual([0, 1], list(range(x)))

  def testFormatString(self):
    # [FUNCTION DOC]
    # Formats a float tensor with two decimals using string formatting and asserts "3.14".
    x = constant_op.constant(3.1415)
    self.assertEqual('3.14', '{:.2f}'.format(x))

  def testNoOpIsNone(self):
    # [FUNCTION DOC]
    # Asserts that running the no-op in eager mode returns nothing (null value).
    self.assertIsNone(control_flow_ops.no_op())

  def testEagerContextPreservedAcrossThreads(self):
    # [FUNCTION DOC]
    # Asserts eager execution is active, then runs a function in a new thread and waits for it.
    # Inside that thread the function asserts eager execution is active, including within an
    # init scope, and that the context-switch stack has one entry that is not building a function
    # and whose enter function is eager mode.
    def init_fn():
      # [FUNCTION DOC]
      # HELPER: thread body for testEagerContextPreservedAcrossThreads; contains the assertions
      # about eager state and context switch stack described there. Returns nothing.
      self.assertTrue(context.executing_eagerly())
      with ops.init_scope():
        self.assertTrue(context.executing_eagerly())
        context_switches = context.context().context_switches
        self.assertLen(context_switches.stack, 1)
        self.assertFalse(context_switches.stack[0].is_building_function)
        self.assertEqual(context_switches.stack[0].enter_context_fn,
                         context.eager_mode)

    self.assertTrue(context.executing_eagerly())
    t1 = threading.Thread(target=init_fn)
    t1.start()
    t1.join()

  def testWeakrefEagerTensor(self):
    # [FUNCTION DOC]
    # Creates an eager tensor with extra Python attributes, keeps weak references to it and to an
    # attribute, deletes the tensor, and asserts both weak references are cleared.
    x = constant_op.constant([[1.]])
    x.at1 = constant_op.constant([[2.]])
    x.at2 = 3.
    weak_x = weakref.ref(x)
    weak_xat1 = weakref.ref(x.at1)
    del x
    self.assertIs(weak_x(), None)
    self.assertIs(weak_xat1(), None)

  def testWeakKeyDictionaryTensor(self):
    # [FUNCTION DOC]
    # Stores values in a weak-key dictionary keyed by tensor references. After deleting one
    # tensor and its reference, asserts it is gone from the dictionary and the other remains;
    # after deleting the second, asserts the dictionary is empty.
    weak_key_dict = weakref.WeakKeyDictionary()

    strong_x = constant_op.constant([[1.]])
    strong_y = constant_op.constant([[2.]])
    strong_x_ref = strong_x.ref()
    strong_y_ref = strong_y.ref()
    weak_key_dict[strong_x_ref] = constant_op.constant([[3.]])
    weak_key_dict[strong_y_ref] = constant_op.constant([[4.]])
    strong_y.a = constant_op.constant([[5.]])
    weak_x_ref = weakref.ref(strong_x)

    del strong_x, strong_x_ref
    self.assertIs(weak_x_ref(), None)
    self.assertEqual([strong_y_ref], list(weak_key_dict))
    self.assertLen(list(weak_key_dict), 1)
    self.assertLen(weak_key_dict, 1)

    del strong_y, strong_y_ref
    self.assertEqual([], list(weak_key_dict))

  def testEagerTensorsCanBeGarbageCollected(self):
    # [FUNCTION DOC]
    # Builds a reference cycle between two eager tensors via attributes, deletes the names, runs
    # garbage collection several times and asserts both weak references are cleared.
    x = constant_op.constant([[1.]])
    y = constant_op.constant([[2.]])
    x.y = y
    y.x = x
    weak_x = weakref.ref(x)
    weak_y = weakref.ref(y)
    del x
    del y
    # Run a gc a few times to ensure cycles are resolved.
    gc.collect()
    gc.collect()
    gc.collect()
    gc.collect()
    self.assertIs(weak_x(), None)
    self.assertIs(weak_y(), None)

  @test_util.disable_tfrt(
      'b/153697193: tfrt cannot decode python stacktrace yet')
  # TODO(b/234153596): Disabled because it invalidates stack traces on other
  # tests (due to partial migration to absl::Status).
  def DISABLED_testAsyncExceptionStackTrace(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name starts with DISABLED_ so the test runner never collects or runs it.
    # Intended scenario: switch to asynchronous execution, run a matmul with invalid shapes inside
    # a nested function so the error is deferred, then assert that reading the result raises an
    # invalid-argument error whose message mentions the originating function name, and finally
    # clear the async error and restore synchronous execution.
    config.set_synchronous_execution(False)

    def exception_originated_from_here():
      # [FUNCTION DOC]
      # Runs a matrix multiply with incompatible shapes and returns the result; used to produce
      # an error whose stack trace names this function.
      # Invalid shapes for matmul.
      return math_ops.matmul([[1]], [[2], [3]])

    # In sync mode, an exception would have been raised here but since this is
    # in async, the exception will be raised next.
    x = exception_originated_from_here()

    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                'in exception_originated_from_here'):
      x.numpy()

    context.async_clear_error()
    config.set_synchronous_execution(True)

  def testCrossContextTensorCache(self):
    # [FUNCTION DOC]
    # Creates a constant in the current context, swaps in a fresh context, creates an equal
    # constant there and asserts its value, restores the original context, and asserts the
    # first constant still has its value. Side effect: temporarily replaces the global context.
    old_context = context.context()
    old_x = constant_op.constant(9.5)
    context._set_context(context.Context())

    try:
      new_x = constant_op.constant(9.5)
      self.assertEqual(new_x.numpy(), 9.5)
    finally:
      context._set_context(old_context)

    self.assertEqual(old_x.numpy(), 9.5)

if __name__ == '__main__':
  test.main()
