# [MODULE DOC]
# Unit tests for the eager tensor class: construction from Python and numpy values (dtype and
# device arguments, downcasting rules, error cases), conversion to and from numpy (copies, views,
# memoryview access, dimension limits), string and repr formatting of ordinary, resource and
# variant tensors, iteration, indexing, boolean conversion, and the native tensor-shape-slice
# utility. Contains a module-private tensor-creation helper and two test classes. Runs the test
# runner when executed as a script.
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
"""Unit tests for TensorFlow "Eager" Mode's Tensor class."""

import copy
import re
import sys

import numpy as np

from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import context
from tensorflow.python.eager import core
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables


def _create_tensor(value, device=None, dtype=None):
  # [FUNCTION DOC]
  # HELPER: builds an eager tensor directly through the low-level tensor constructor for the tests.
  # Inputs: value (a Python or numpy value), device (optional device name; defaults to the eager
  # context's current device name), dtype (optional data type object; converted to its enum value
  # before use, default none meaning inferred). Ensures the eager context is initialized first.
  # Returns the new eager tensor. A native not-ok status raised by the constructor is converted to
  # the matching TensorFlow exception type and raised.
  context.ensure_initialized()
  ctx = context.context()
  if device is None:
    device = ctx.device_name
  if dtype is not None:
    dtype = dtype.as_datatype_enum
  try:
    return ops.EagerTensor(value, device=device, dtype=dtype)
  except core._NotOkStatusException as e:  # pylint: disable=protected-access
    raise core._status_to_exception(e)


class TFETensorTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the behavior of eager tensors: creation, dtype inference, numpy interop,
  # printing, iteration, memory views, and conversion errors. Most tests create tensors with the
  # module helper that calls the low-level constructor.

  def testScalarTensor(self):
    # [FUNCTION DOC]
    # Scenario: create a 32-bit integer scalar tensor with value three. Asserts it equals the
    # tensor made from a zero-dimensional numpy array, has the 32-bit integer dtype, rank zero and
    # an empty shape list, and that its string and repr forms contain the tensor type name.
    t = _create_tensor(3, dtype=dtypes.int32)
    self.assertAllEqual(t, _create_tensor(np.array(3)))
    self.assertEqual(dtypes.int32, t.dtype)
    self.assertEqual(0, t.shape.ndims)
    self.assertAllEqual([], t.shape.as_list())
    self.assertIn("tf.Tensor", str(t))
    self.assertIn("tf.Tensor", repr(t))

  def testBadConstructorArgs(self):
    # [FUNCTION DOC]
    # Scenario: call the low-level tensor constructor with invalid arguments. Asserts a type error
    # naming the missing device argument when no device is given, and a type error saying a data
    # type value is expected when the dtype is a string. If no GPU exists the test is skipped
    # afterwards; with a GPU it also asserts a type error about device parsing for a numeric
    # device argument.
    context.ensure_initialized()
    ctx = context.context()
    device = ctx.device_name
    # Missing device.
    with self.assertRaisesRegex(TypeError, r".*argument 'device' \(pos 2\).*"):
      ops.EagerTensor(1)
    # Bad dtype type.
    with self.assertRaisesRegex(TypeError,
                                "Expecting a DataType value for dtype. Got"):
      ops.EagerTensor(1, device=device, dtype="1")

    # Following errors happen when trying to copy to GPU.
    if not test_util.is_gpu_available():
      self.skipTest("No GPUs found")

    with ops.device("/device:GPU:0"):
      # Bad device.
      with self.assertRaisesRegex(TypeError, "Error parsing device argument"):
        ops.EagerTensor(1.0, device=1)

  def testNumpyValue(self):
    # [FUNCTION DOC]
    # Scenario: build a tensor from a one-element float numpy array and assert it equals the array.
    values = np.array([3.0])
    t = _create_tensor(values)
    self.assertAllEqual(values, t)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testNumpyDtypeSurvivesThroughTensorConversion(self):
    # [FUNCTION DOC]
    # Scenario (also checks that no new Python objects leak while running eagerly): convert numpy
    # scalars of 32 and 64 bit integer and float types with both the generic tensor conversion and
    # the constant function. Asserts the resulting numpy value keeps the original dtype and value.
    scalar_creators = [np.int32, np.int64, np.float32, np.float64]
    conversion_functions = [ops.convert_to_tensor, constant_op.constant]

    for scalar_creator in scalar_creators:
      for conversion_function in conversion_functions:
        np_val = scalar_creator(3)
        tensor_val = conversion_function(np_val)
        self.assertEqual(tensor_val.numpy().dtype, np_val.dtype)
        self.assertEqual(tensor_val.numpy(), np_val)

  def testNumpyValueWithCast(self):
    # [FUNCTION DOC]
    # Scenario: build a 64-bit float tensor from a 32-bit float numpy array and assert values are
    # equal. Then asserts a type error about an invalid dtype value when the dtype enum given is
    # out of range (thirty-four, above the largest valid data type code).
    values = np.array([3.0], dtype=np.float32)
    t = _create_tensor(values, dtype=dtypes.float64)
    self.assertAllEqual(values, t)
    ctx = context.context()
    # Bad dtype value.
    with self.assertRaisesRegex(TypeError, "Invalid dtype argument value"):
      # The max value of TF_DataType is 33, so using 34 for the dtype fails.
      ops.EagerTensor(values, device=ctx.device_name, dtype=34)

  def testNumpyOrderHandling(self):
    # [FUNCTION DOC]
    # Scenario: build a tensor from a two by two numpy array stored in column-major order and
    # assert the logical values match a row-major list.
    n = np.array([[1, 2], [3, 4]], order="F")
    t = _create_tensor(n)
    self.assertAllEqual([[1, 2], [3, 4]], t)

  def testNumpyArrayDtype(self):
    # [FUNCTION DOC]
    # Scenario: convert a float tensor to a numpy array requesting the 32-bit integer dtype and
    # assert the result is the truncated integer list.
    tensor = constant_op.constant([1.0, 2.0, 3.0])
    numpy_tensor = np.asarray(tensor, dtype=np.int32)
    self.assertAllEqual(numpy_tensor, [1, 2, 3])

  def testNdimsAgreesWithNumpy(self):
    # [FUNCTION DOC]
    # Scenario: for a scalar, a vector and a matrix, build a tensor from the numpy array and assert
    # the tensor's rank attribute equals the array's number of dimensions.
    numpy_tensor = np.asarray(1.0)
    tensor = constant_op.constant(numpy_tensor)
    self.assertAllEqual(numpy_tensor.ndim, tensor.ndim)

    numpy_tensor = np.asarray([1.0, 2.0, 3.0])
    tensor = constant_op.constant(numpy_tensor)
    self.assertAllEqual(numpy_tensor.ndim, tensor.ndim)

    numpy_tensor = np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    tensor = constant_op.constant(numpy_tensor)
    self.assertAllEqual(numpy_tensor.ndim, tensor.ndim)

  def testLenAgreesWithNumpy(self):
    # [FUNCTION DOC]
    # Scenario: for a scalar, the numpy array raises a type error for length, and the tensor
    # raises a type error whose message says a scalar tensor has no length. For a vector and a
    # matrix, asserts the tensor length equals the numpy length.
    numpy_tensor = np.asarray(1.0)
    tensor = constant_op.constant(numpy_tensor)
    with self.assertRaises(TypeError):
      len(numpy_tensor)
    with self.assertRaisesRegex(TypeError, r"Scalar tensor has no `len[(][)]`"):
      len(tensor)

    numpy_tensor = np.asarray([1.0, 2.0, 3.0])
    tensor = constant_op.constant(numpy_tensor)
    self.assertAllEqual(len(numpy_tensor), len(tensor))

    numpy_tensor = np.asarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    tensor = constant_op.constant(numpy_tensor)
    self.assertAllEqual(len(numpy_tensor), len(tensor))

  def testCopy(self):
    # [FUNCTION DOC]
    # Scenario: make a shallow copy and a deep copy of a scalar tensor, deleting each copy after
    # checking it equals one. Finally asserts the original is still intact (equal to one).
    t = constant_op.constant(1.0)
    tt = copy.copy(t)
    self.assertAllEqual(tt, 1.0)
    del tt
    tt = copy.deepcopy(t)
    self.assertAllEqual(tt, 1.0)
    del tt
    self.assertAllEqual(t, 1.0)

  def testConstantDtype(self):
    # [FUNCTION DOC]
    # Scenario: create a constant with a numpy 64-bit integer type as dtype and assert the tensor's
    # dtype is the 64-bit integer data type.
    self.assertEqual(
        constant_op.constant(1, dtype=np.int64).dtype, dtypes.int64)

  def testTensorAndNumpyMatrix(self):
    # [FUNCTION DOC]
    # Scenario: build a tensor from a nested list of floats. Asserts values equal the numpy matrix,
    # the dtype equals both numpy and TensorFlow 32-bit float, and the shape is two by two.
    expected = np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)
    actual = _create_tensor([[1.0, 2.0], [3.0, 4.0]])
    self.assertAllEqual(expected, actual)
    self.assertEqual(np.float32, actual.dtype)
    self.assertEqual(dtypes.float32, actual.dtype)
    self.assertAllEqual([2, 2], actual.shape.as_list())

  def testNumpyArrayInterface(self):
    # [FUNCTION DOC]
    # Scenario: wrap a float numpy matrix in an object that only exposes the numpy array interface
    # attribute, build a tensor from that object, and assert its values equal the original matrix.

    class ArrayAsArrayInterface:
      # [CLASS DOC]
      # HELPER: tiny test double that holds a numpy array and exposes only its array-interface
      # description, to check tensors can be built from array-interface objects.
      """Simple class that wraps an np.array as an __array_interface__."""

      def __init__(self, array):
        # [FUNCTION DOC]
        # Constructor. Input: a numpy array, stored as an attribute. Returns nothing.
        self.array = array

      @property
      def __array_interface__(self):
        # [FUNCTION DOC]
        # Property returning the wrapped array's array-interface mapping, which numpy and tensor
        # conversion use to read the data. No arguments, no side effects.
        return self.array.__array_interface__

    expected = np.array([[1.0, 2.0], [3.0, 4.0]], np.float32)
    array_interface = ArrayAsArrayInterface(expected)
    actual = _create_tensor(array_interface)
    self.assertAllEqual(expected, actual)

  def testFloatDowncast(self):
    # [FUNCTION DOC]
    # Scenario: a Python float becomes a 32-bit float tensor by default, but a 64-bit float tensor
    # when that dtype is explicitly requested. Asserts both dtypes.
    # Unless explicitly specified, float64->float32
    t = _create_tensor(3.0)
    self.assertEqual(dtypes.float32, t.dtype)
    t = _create_tensor(3.0, dtype=dtypes.float64)
    self.assertEqual(dtypes.float64, t.dtype)

  def testBool(self):
    # [FUNCTION DOC]
    # Scenario: truth value of one-element tensors of any rank. Asserts false for false, zero and
    # nested false values, and true for one (integer or float).
    self.assertFalse(bool(_create_tensor(False)))
    self.assertFalse(bool(_create_tensor([False])))
    self.assertFalse(bool(_create_tensor([[False]])))
    self.assertFalse(bool(_create_tensor([0])))
    self.assertFalse(bool(_create_tensor([0.])))
    self.assertTrue(bool(_create_tensor([1])))
    self.assertTrue(bool(_create_tensor([1.])))

  def testIndex(self):
    # [FUNCTION DOC]
    # Scenario: use a scalar integer tensor as a list index and assert the right element comes
    # back; using a one-element vector tensor as an index must raise a type error.
    self.assertEqual([42][_create_tensor(0)], 42)

    with self.assertRaises(TypeError):
      _ = [42][_create_tensor([0])]

  def testIntDowncast(self):
    # [FUNCTION DOC]
    # Scenario: a small Python integer becomes a 32-bit integer tensor, an explicit 64-bit request
    # gives 64-bit, and a value too big for 32 bits (two to the thirty-third) is inferred as
    # 64-bit. Asserts each dtype.
    t = _create_tensor(3)
    self.assertEqual(dtypes.int32, t.dtype)
    t = _create_tensor(3, dtype=dtypes.int64)
    self.assertEqual(dtypes.int64, t.dtype)
    t = _create_tensor(2**33)
    self.assertEqual(dtypes.int64, t.dtype)

  def testTensorCreationFailure(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the equality assertion inside the error-expecting block can never run its
    # comparison because creation is expected to raise first; it is dead code.
    # Scenario: creating a tensor from a ragged nested list (rows of different length) must raise
    # a value error.
    with self.assertRaises(ValueError):
      # Should fail because the each row of the Python object has a different
      # number of columns.
      self.assertEqual(None, _create_tensor([[1], [1, 2]]))

  def testMultiLineTensorStr(self):
    # [FUNCTION DOC]
    # Scenario: string form of a three by three identity tensor. Asserts it contains the shape and
    # dtype name text, and that it contains the string form of itself.
    t = _create_tensor(np.eye(3))
    tensor_str = str(t)
    self.assertIn("shape=%s, dtype=%s" % (t.shape, t.dtype.name), tensor_str)
    self.assertIn(str(t), tensor_str)

  def testMultiLineTensorRepr(self):
    # [FUNCTION DOC]
    # Scenario: repr of a three by three identity tensor. Asserts it starts with an opening angle
    # bracket, ends with a closing one, and contains the shape, dtype name and the numpy repr of
    # the values on a new line.
    t = _create_tensor(np.eye(3))
    tensor_repr = repr(t)
    self.assertTrue(tensor_repr.startswith("<"))
    self.assertTrue(tensor_repr.endswith(">"))
    self.assertIn(
        "shape=%s, dtype=%s, numpy=\n%r" % (t.shape, t.dtype.name, t.numpy()),
        tensor_repr)

  def testTensorStrReprObeyNumpyPrintOptions(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the global numpy print options are restored at the end of the test body without
    # a finally block, so a failing assertion leaves the altered options in place for later tests.
    # Scenario: set the numpy summarization threshold to two and edge items to one, create a tensor
    # of ten integers, and assert both the string and repr forms are summarized with an ellipsis
    # between the first and last values. Side effect: temporarily changes global numpy settings.
    orig_threshold = np.get_printoptions()["threshold"]
    orig_edgeitems = np.get_printoptions()["edgeitems"]
    np.set_printoptions(threshold=2, edgeitems=1)

    t = _create_tensor(np.arange(10, dtype=np.int32))
    self.assertTrue(re.match(r".*\[.*0.*\.\.\..*9.*\]", str(t)))
    self.assertTrue(re.match(r".*\[.*0.*\.\.\..*9.*\]", repr(t)))

    # Clean up: reset to previous printoptions.
    np.set_printoptions(threshold=orig_threshold, edgeitems=orig_edgeitems)

  def testZeroDimTensorStr(self):
    # [FUNCTION DOC]
    # Scenario: string form of the scalar 42 contains the value, an empty shape and the 32-bit
    # integer dtype name.
    t = _create_tensor(42)
    self.assertIn("42, shape=(), dtype=int32", str(t))

  def testZeroDimTensorRepr(self):
    # [FUNCTION DOC]
    # Scenario: repr of the scalar 42 starts with an opening angle bracket, ends with a closing
    # one, and contains the empty shape, the integer dtype name and the numpy value.
    t = _create_tensor(42)
    self.assertTrue(repr(t).startswith("<"))
    self.assertTrue(repr(t).endswith(">"))
    self.assertIn("shape=(), dtype=int32, numpy=42", repr(t))

  def testZeroSizeTensorStr(self):
    # [FUNCTION DOC]
    # Scenario: string form of an empty float tensor shows empty brackets, a shape of zero and the
    # 32-bit float dtype name.
    t = _create_tensor(np.zeros(0, dtype=np.float32))
    self.assertIn("[], shape=(0,), dtype=float32", str(t))

  def testZeroSizeTensorRepr(self):
    # [FUNCTION DOC]
    # Scenario: repr of an empty float tensor is wrapped in angle brackets and contains the zero
    # shape, the dtype name and the numpy repr of its (empty) values.
    t = _create_tensor(np.zeros(0, dtype=np.float32))
    self.assertTrue(repr(t).startswith("<"))
    self.assertTrue(repr(t).endswith(">"))
    self.assertIn("shape=(0,), dtype=float32, numpy=%r" % t.numpy(), repr(t))

  def testStringTensor(self):
    # [FUNCTION DOC]
    # Scenario: build a tensor from a two by two numpy array of byte strings, convert back to
    # numpy, and assert every element is equal to the original.
    t_np_orig = np.array([[b"a", b"ab"], [b"abc", b"abcd"]])
    t = _create_tensor(t_np_orig)
    t_np = t.numpy()
    self.assertTrue(np.all(t_np == t_np_orig), "%s vs %s" % (t_np, t_np_orig))

  def testIterateOverTensor(self):
    # [FUNCTION DOC]
    # Scenario: iterate a two by two tensor in lockstep with the source nested list; each row of
    # the tensor, as numpy, must equal the corresponding list row.
    l = [[1, 2], [3, 4]]
    t = _create_tensor(l)
    for list_element, tensor_element in zip(l, t):
      self.assertAllEqual(list_element, tensor_element.numpy())

  def testIterateOverScalarTensorRaises(self):
    # [FUNCTION DOC]
    # Scenario: asking for an iterator over a scalar tensor must raise a type error saying a scalar
    # tensor cannot be iterated.
    t = _create_tensor(1)
    with self.assertRaisesRegex(TypeError,
                                "Cannot iterate over a scalar tensor"):
      iter(t)

  @test_util.run_gpu_only
  def testStringTensorOnGPU(self):
    # [FUNCTION DOC]
    # Scenario (GPU only): create a string tensor under the first GPU device scope and assert its
    # device name mentions the GPU.
    with ops.device("/device:GPU:0"):
      t = _create_tensor("test string")
      self.assertIn("GPU", t.device)

  def testInvalidUTF8ProducesReasonableError(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the Python-2 skip branch is dead code on any supported interpreter.
    # Scenario: reading a file whose name is a byte string that is not valid text encoding must
    # raise a Unicode decode error rather than a native crash or opaque failure.
    if sys.version_info[0] < 3:
      self.skipTest("Test is only valid in python3.")
    with self.assertRaises(UnicodeDecodeError):
      io_ops.read_file(b"\xff")

  @test_util.run_in_graph_and_eager_modes
  def testConvertToTensorPreferredDtypeIsRespected(self):
    # [FUNCTION DOC]
    # Scenario (runs in graph and eager modes): converting 0.5 with an integer preferred dtype
    # still gives a 32-bit float (preference not applicable), while a 64-bit float preferred dtype
    # is honored. Asserts the resulting dtypes.
    self.assertEqual(
        ops.convert_to_tensor(0.5, preferred_dtype=dtypes.int32).dtype,
        dtypes.float32)
    self.assertEqual(
        ops.convert_to_tensor(0.5, preferred_dtype=dtypes.float64).dtype,
        dtypes.float64)

  @test_util.run_in_graph_and_eager_modes
  def testCompatibility(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager modes): checks constant creation compatibility rules. A fractional
    # float with any integer dtype raises a type error; integer five evaluates to five for half,
    # single, double and brain floating types; nested int or float values give complex128 tensors;
    # a fractional float with any quantized dtype raises a type error. A TODO notes quantized types
    # are broken in eager mode.
    integer_types = [
        dtypes.int8, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.uint8,
        dtypes.uint16, dtypes.uint32, dtypes.uint64
    ]

    # Floats are not compatible with ints
    for t in integer_types:
      with self.assertRaises(TypeError):
        constant_op.constant(0.5, dtype=t)

    # Ints compatible with floats
    self.assertEqual(
        self.evaluate(constant_op.constant(5, dtype=dtypes.float16)), 5.0)
    self.assertEqual(
        self.evaluate(constant_op.constant(5, dtype=dtypes.float32)), 5.0)
    self.assertEqual(
        self.evaluate(constant_op.constant(5, dtype=dtypes.float64)), 5.0)
    self.assertEqual(
        self.evaluate(constant_op.constant(5, dtype=dtypes.bfloat16)), 5.0)

    # Ints and floats are compatible with complex types
    self.assertEqual(
        constant_op.constant([[1.0]], dtype=dtypes.complex128).dtype,
        dtypes.complex128)
    self.assertEqual(
        constant_op.constant([[1]], dtype=dtypes.complex128).dtype,
        dtypes.complex128)

    # Quantized types are not compatible with floats
    quantized_types = [
        dtypes.qint16, dtypes.qint32, dtypes.qint8, dtypes.quint16,
        dtypes.quint8
    ]

    for t in quantized_types:
      with self.assertRaises(TypeError):
        constant_op.constant(0.5, dtype=t)

    # TODO(b/118402529): quantized types are broken in eager.

  @test_util.run_in_graph_and_eager_modes
  def testCConvertToTensor(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager modes): comparing an integer tensor with a float using less-than
    # must raise a type error since the float cannot be converted to the integer dtype.
    with self.assertRaises(TypeError):
      _ = constant_op.constant(0) < 0.5

  @test_util.run_in_graph_and_eager_modes
  def testConvertToTensorAllowsOverflow(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager modes): converting a value too large for the 8-bit unsigned
    # integer dtype must not raise. No assertion beyond not failing.
    _ = ops.convert_to_tensor(123456789, dtype=dtypes.uint8)

  @test_util.run_in_graph_and_eager_modes
  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testConvertToTensorNumpyZeroDim(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager modes, no leaked Python objects): convert a list of two
    # zero-dimensional numpy arrays of 32-bit integer, half and 32-bit float type. Asserts the
    # resulting tensor has the matching dtype and values sixty-five and sixteen.
    for np_type, dtype in [(np.int32, dtypes.int32), (np.half, dtypes.half),
                           (np.float32, dtypes.float32)]:
      x = ops.convert_to_tensor(
          [np.array(65, dtype=np_type),
           np.array(16, dtype=np_type)])
      self.assertEqual(x.dtype, dtype)
      self.assertAllEqual(x, [65, 16])

  @test_util.run_in_graph_and_eager_modes
  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testConvertToTensorNumpyScalar(self):
    # [FUNCTION DOC]
    # Scenario (graph and eager modes, no leaked Python objects): convert a list of two Python
    # integers obtained from numpy scalars and assert the values three hundred twenty-one and
    # sixteen.
    x = ops.convert_to_tensor([
        np.array(321, dtype=np.int64).item(),
        np.array(16, dtype=np.int64).item()
    ])
    self.assertAllEqual(x, [321, 16])

  def testEagerTensorError(self):
    # [FUNCTION DOC]
    # Scenario: converting a float to an integer dtype must raise a type error whose message says
    # the value cannot be converted to an eager tensor of that dtype.
    with self.assertRaisesRegex(TypeError,
                                "Cannot convert .* to EagerTensor of dtype .*"):
      _ = ops.convert_to_tensor(1., dtype=dtypes.int32)

  def testEagerLargeConstant(self):
    # [FUNCTION DOC]
    # Scenario: for 64 and 32 bit unsigned and signed integer dtypes, create constants holding the
    # dtype's maximum and minimum values and assert they read back exactly.
    for t in [dtypes.uint64, dtypes.uint32, dtypes.int32, dtypes.int64]:
      self.assertEqual(constant_op.constant(t.max, dtype=t).numpy(), t.max)
      self.assertEqual(constant_op.constant(t.min, dtype=t).numpy(), t.min)

  def test_numpyIsView(self):
    # [FUNCTION DOC]
    # Scenario: on the CPU, take the private numpy view of a one-element tensor, write forty-two
    # into it, and assert the tensor itself now holds forty-two (the view shares memory).
    with ops.device("CPU"):
      t = constant_op.constant([0.0])
      t._numpy()[0] = 42.0
      self.assertAllClose(t, constant_op.constant([42.0]))

  def test_numpyFailsForResource(self):
    # [FUNCTION DOC]
    # Scenario: asking for the private numpy view of a variable's resource handle must raise an
    # invalid-argument error saying a resource cannot be converted.
    v = variables.Variable(42)
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Cannot convert .+ resource"):
      v._handle._numpy()

  def test_numpyFailsForVariant(self):
    # [FUNCTION DOC]
    # Scenario: asking for the private numpy view of a variant-typed tensor (a reserved tensor
    # list) must raise an invalid-argument error saying a variant cannot be converted.
    variant_t = list_ops.tensor_list_reserve(
        element_shape=[], num_elements=1, element_dtype=dtypes.float32)
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                "Cannot convert .+ variant"):
      variant_t._numpy()

  def testMemoryviewFailsForResource(self):
    # [FUNCTION DOC]
    # Scenario: creating a numpy array from a memory view of a variable's resource handle must
    # raise a buffer error saying a resource cannot be converted.
    v = variables.Variable(42)
    with self.assertRaisesRegex(BufferError, "Cannot convert .+ resource"):
      np.asarray(memoryview(v._handle))

  def testMemoryviewFailsForVariant(self):
    # [FUNCTION DOC]
    # Scenario: creating a numpy array from a memory view of a variant tensor must raise a buffer
    # error saying a variant cannot be converted.
    variant_t = list_ops.tensor_list_reserve(
        element_shape=[], num_elements=1, element_dtype=dtypes.float32)
    with self.assertRaisesRegex(BufferError, "Cannot convert .+ variant"):
      np.asarray(memoryview(variant_t))

  def testMemoryviewIsReadonly(self):
    # [FUNCTION DOC]
    # Scenario: a memory view of a float tensor must report itself as read-only.
    t = constant_op.constant([0.0])
    self.assertTrue(memoryview(t).readonly)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testMemoryviewScalar(self):
    # [FUNCTION DOC]
    # Scenario (no leaked Python objects): a numpy array made from a memory view of a scalar float
    # tensor must equal the 32-bit float scalar forty-two.
    t = constant_op.constant(42.0)
    self.assertAllEqual(
        np.array(memoryview(t)), np.array(42.0, dtype=np.float32))

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testMemoryviewEmpty(self):
    # [FUNCTION DOC]
    # Scenario (no leaked Python objects): a numpy array made from a memory view of an empty float
    # tensor must equal an empty array.
    t = constant_op.constant([], dtype=np.float32)
    self.assertAllEqual(np.array(memoryview(t)), np.array([]))

  @test_util.run_gpu_only
  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testMemoryviewCopyToCPU(self):
    # [FUNCTION DOC]
    # Scenario (GPU only, no leaked Python objects): create a tensor on the GPU and assert that a
    # numpy array built from its memory view (which copies to host memory) has the expected value.
    with ops.device("/device:GPU:0"):
      t = constant_op.constant([0.0])
    self.assertAllEqual(
        np.array(memoryview(t)), np.array([0.0], dtype=np.float32))

  @test_util.disable_tfrt("b/169877776: ResourceVariable is not initialized "
                          "properly in TFRT")
  def testResourceTensorCopy(self):
    # [FUNCTION DOC]
    # Scenario (skipped without a GPU; disabled under TFRT): create a variable on the GPU, read it
    # through its handle, copy the handle to the CPU, read through that copy, and assert both reads
    # are equal.
    if not test_util.is_gpu_available():
      self.skipTest("GPU only")

    with ops.device("GPU:0"):
      v = resource_variable_ops.ResourceVariable(1.)

    read_handle_on_gpu = resource_variable_ops.read_variable_op(
        v.handle, dtypes.float32)
    handle_on_cpu = v.handle.cpu()
    read_handle_on_cpu = resource_variable_ops.read_variable_op(
        handle_on_cpu, dtypes.float32)

    self.assertAllEqual(read_handle_on_cpu, read_handle_on_gpu)

  def testEagerTensorFormat(self):
    # [FUNCTION DOC]
    # Scenario: formatting a scalar integer one via plain format, string conversion and repr
    # conversion gives respectively the bare value, the type-name form with shape and dtype, and
    # the angle-bracket repr form with numpy value. Asserts each exact text.
    t = array_ops.constant(1)
    self.assertEqual(f"{t}", "1")
    self.assertEqual(str(t), "tf.Tensor(1, shape=(), dtype=int32)")
    self.assertEqual(f"{t!s}", "tf.Tensor(1, shape=(), dtype=int32)")
    self.assertEqual(repr(t), "<tf.Tensor: shape=(), dtype=int32, numpy=1>")
    self.assertEqual(f"{t!r}", "<tf.Tensor: shape=(), dtype=int32, numpy=1>")

  def testEagerTensorFormatForResource(self):
    # [FUNCTION DOC]
    # Scenario: create a resource handle tensor and assert that plain, string and repr formats
    # show the resource handle description (name, device, container and a compiler-dependent
    # type name matched by wildcard), with dtype resource and a "value" label in the repr form.
    t = resource_variable_ops.VarHandleOp(shape=[], dtype=dtypes.float32)

    # type is compiler-dependent, as it comes from demangling.
    handle_str = (f"<ResourceHandle("
                  f"name=\"\", "
                  f"device=\"{t.device}\", "
                  f"container=\"localhost\", "
                  f"type=\"@@tensorflow@@Var@@\")>")

    def make_regex(s):
      # [FUNCTION DOC]
      # HELPER: turns an expected text into a matching pattern by escaping every special
      # character and then replacing the placeholder marker with a wildcard (for the
      # compiler-dependent type name). Input: expected text; returns the pattern string.
      return re.escape(s).replace("@@", ".*")

    self.assertRegex(f"{t}", make_regex(handle_str))
    self.assertRegex(
        str(t),
        make_regex(f"tf.Tensor({handle_str}, shape=(), dtype=resource)"))
    self.assertRegex(
        f"{t!s}",
        make_regex(f"tf.Tensor({handle_str}, shape=(), dtype=resource)"))
    self.assertRegex(
        repr(t),
        make_regex(
            f"<tf.Tensor: shape=(), dtype=resource, value={handle_str}>"))
    self.assertRegex(
        f"{t!r}",
        make_regex(
            f"<tf.Tensor: shape=(), dtype=resource, value={handle_str}>"))

  def testEagerTensorFormatForVariant(self):
    # [FUNCTION DOC]
    # Scenario: for a variant tensor holding a reserved tensor list, assert plain, string and repr
    # formats show the tensor-list placeholder text, with dtype variant in the string and repr
    # forms.
    t = list_ops.tensor_list_reserve(
        element_shape=[1], num_elements=1, element_dtype=dtypes.float32)
    self.assertEqual(f"{t}", "<TensorList>")
    self.assertEqual(str(t), "tf.Tensor(<TensorList>, shape=(), dtype=variant)")
    self.assertEqual(f"{t!s}",
                     "tf.Tensor(<TensorList>, shape=(), dtype=variant)")
    self.assertEqual(
        repr(t), "<tf.Tensor: shape=(), dtype=variant, value=<TensorList>>")
    self.assertEqual(
        f"{t!r}", "<tf.Tensor: shape=(), dtype=variant, value=<TensorList>>")

  def testNumpyTooManyDimensions(self):
    # [FUNCTION DOC]
    # Scenario: the maximum numpy dimension count is sixty-four for numpy 2 and later, else
    # thirty-two. Create a constant with one more dimension than that and assert that converting to
    # numpy raises an invalid-argument error stating the dimension count and the numpy maximum.
    max_dims = 64 if np.lib.NumpyVersion(np.__version__) >= "2.0.0.dev0" else 32
    t = constant_op.constant(1., shape=[1] * (max_dims + 1))
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "Cannot convert tensor with %d dimensions to NumPy array. NumPy arrays "
        "can have at most %d dimensions"% (max_dims + 1, max_dims)):
      t.numpy()

  def testNumpyDimsTooBig(self):
    # [FUNCTION DOC]
    # Scenario: build a tensor of ones with shape zero by two-to-the-31 by two-to-the-31 (no
    # elements, but huge dimension product). Asserts converting to numpy raises an invalid-argument
    # error saying numpy array creation failed for that shape because the array is too big.
    # Creating a Numpy array fails in some cases if the product of non-zero
    # dimensions is very big, even if the shape also has a zero in it.
    t = array_ops.ones((0, 2**31, 2**31))
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        r"Failed to create numpy array from tensor of shape "
        r"\[0, 2147483648, 2147483648\]. Numpy error.*array is too big"):
      t.numpy()


class TFETensorUtilTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the native tensor-shape-slice utility (takes a list of tensors and a dimension
  # index, returns a vector of those dimension sizes) plus a few misc tensor checks. Fixture: the
  # eager context is initialized in setUp.

  def setUp(self):
    # [FUNCTION DOC]
    # Test fixture: calls the base setUp and makes sure the eager context is initialized.
    super(TFETensorUtilTest, self).setUp()
    context.ensure_initialized()

  def testListOfThree(self):
    # [FUNCTION DOC]
    # Scenario: three integer tensors of different shapes. Slicing dimension zero must give sizes
    # three, two, four; slicing dimension one must give two, three, one. Asserts both vectors.
    t1 = _create_tensor([[1, 2], [3, 4], [5, 6]], dtype=dtypes.int32)
    t2 = _create_tensor([[1, 2, 5], [3, 4, 5]], dtype=dtypes.int32)
    t3 = _create_tensor([[1], [3], [5], [6]], dtype=dtypes.int32)

    r = pywrap_tfe.TFE_Py_TensorShapeSlice([t1, t2, t3], 0)
    self.assertAllEqual(np.array([3, 2, 4]), r.numpy())

    r = pywrap_tfe.TFE_Py_TensorShapeSlice([t1, t2, t3], 1)
    self.assertAllEqual(np.array([2, 3, 1]), r.numpy())

  def testEmptyTensorList(self):
    # [FUNCTION DOC]
    # Scenario: slicing an empty list of tensors returns an eager tensor with zero elements.
    a = pywrap_tfe.TFE_Py_TensorShapeSlice([], 0)
    self.assertTrue(isinstance(a, ops.EagerTensor))
    self.assertEqual(0, a.numpy().size)

  def testTensorListContainsNonTensors(self):
    # [FUNCTION DOC]
    # Scenario: a list holding a non-tensor (a string at index one, then an integer at index zero)
    # makes the shape-slice utility raise a type error naming the bad element index and type.
    t1 = _create_tensor([1, 2], dtype=dtypes.int32)

    with self.assertRaisesRegex(
        TypeError,
        r"Expected a list of EagerTensors but element 1 has type \"str\""):
      pywrap_tfe.TFE_Py_TensorShapeSlice([t1, "abc"], 0)

    with self.assertRaisesRegex(
        TypeError,
        r"Expected a list of EagerTensors but element 0 has type \"int\""):
      pywrap_tfe.TFE_Py_TensorShapeSlice([2, t1], 0)

  def testTensorListNotList(self):
    # [FUNCTION DOC]
    # Scenario: passing a single tensor instead of a list or tuple to the shape-slice utility
    # raises a type error saying the argument must be a list or tuple.
    t1 = _create_tensor([1, 2], dtype=dtypes.int32)

    with self.assertRaisesRegex(
        TypeError,
        r"tensors argument must be a list or a tuple. Got.*EagerTensor"):
      pywrap_tfe.TFE_Py_TensorShapeSlice(t1, -2)

  def testNegativeSliceDim(self):
    # [FUNCTION DOC]
    # Scenario: a negative slice dimension passed to the shape-slice utility raises a value error
    # saying the dimension must be non-negative.
    t1 = _create_tensor([1, 2], dtype=dtypes.int32)

    with self.assertRaisesRegex(
        ValueError, r"Slice dimension must be non-negative. Got -2"):
      pywrap_tfe.TFE_Py_TensorShapeSlice([t1], -2)

  def testUnicode(self):
    # [FUNCTION DOC]
    # Scenario: a constant made from a Unicode text value reads back as the equivalent byte string.
    self.assertEqual(constant_op.constant(u"asdf").numpy(), b"asdf")

  def testFloatTensor(self):
    # [FUNCTION DOC]
    # Scenario: tensors made from numpy 64, 32 and 16 bit float scalars keep those dtypes, while a
    # plain Python float gives a 32-bit float. Asserts each dtype.
    self.assertEqual(dtypes.float64, _create_tensor(np.float64()).dtype)  # pylint: disable=no-value-for-parameter
    self.assertEqual(dtypes.float32, _create_tensor(np.float32()).dtype)  # pylint: disable=no-value-for-parameter
    self.assertEqual(dtypes.float16, _create_tensor(np.float16()).dtype)  # pylint: disable=no-value-for-parameter
    self.assertEqual(dtypes.float32, _create_tensor(0.0).dtype)

  def testSliceDimOutOfRange(self):
    # [FUNCTION DOC]
    # Scenario: with a matrix, a vector and a scalar, slicing a dimension not smaller than some
    # tensor's rank raises an index error naming the dimension, the offending tensor's index and
    # its rank. Five combinations are asserted, including mixed lists where a later tensor fails.
    t1 = _create_tensor([[1, 2], [3, 4], [5, 6]], dtype=dtypes.int32)
    t2 = _create_tensor([1, 2], dtype=dtypes.int32)
    t3 = _create_tensor(2, dtype=dtypes.int32)

    with self.assertRaisesRegex(
        IndexError,
        r"Slice dimension \(2\) must be smaller than rank of all tensors, "
        "but tensor at index 0 has rank 2"):
      pywrap_tfe.TFE_Py_TensorShapeSlice([t1], 2)

    with self.assertRaisesRegex(
        IndexError,
        r"Slice dimension \(1\) must be smaller than rank of all tensors, "
        "but tensor at index 0 has rank 1"):
      pywrap_tfe.TFE_Py_TensorShapeSlice([t2], 1)

    with self.assertRaisesRegex(
        IndexError,
        r"Slice dimension \(1\) must be smaller than rank of all tensors, "
        "but tensor at index 1 has rank 1"):
      pywrap_tfe.TFE_Py_TensorShapeSlice([t1, t2], 1)

    with self.assertRaisesRegex(
        IndexError,
        r"Slice dimension \(0\) must be smaller than rank of all tensors, "
        "but tensor at index 0 has rank 0"):
      pywrap_tfe.TFE_Py_TensorShapeSlice([t3], 0)

    with self.assertRaisesRegex(
        IndexError,
        r"Slice dimension \(0\) must be smaller than rank of all tensors, "
        "but tensor at index 2 has rank 0"):
      pywrap_tfe.TFE_Py_TensorShapeSlice([t2, t1, t3], 0)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testTensorDir(self):
    # [FUNCTION DOC]
    # Scenario (no leaked Python objects): attach a custom attribute to a tensor instance. The
    # directory listing of the instance must contain it, and after removing it must equal the
    # directory listing of the eager tensor class.
    t = array_ops.ones(1)
    t.test_attr = "Test"

    instance_dir = dir(t)
    type_dir = dir(ops.EagerTensor)

    # Monkey patched attributes should show up in dir(t)
    self.assertIn("test_attr", instance_dir)
    instance_dir.remove("test_attr")
    self.assertEqual(instance_dir, type_dir)

  def testNonRectangularPackAsConstant(self):
    # [FUNCTION DOC]
    # Scenario: creating a constant from a list of two numpy arrays of incompatible shapes raises
    # a value error mentioning a non-rectangular Python sequence.
    l = [array_ops.zeros((10, 1)).numpy(), array_ops.zeros(1).numpy()]

    with self.assertRaisesRegex(ValueError, "non-rectangular Python sequence"):
      constant_op.constant(l)

  @test_util.assert_no_new_pyobjects_executing_eagerly()
  def testFloatAndIntAreConvertibleToComplex(self):
    # [FUNCTION DOC]
    # Scenario (no leaked Python objects): converting a nested list mixing floats, integers and
    # complex numbers to a 128-bit complex tensor gives values equal to the numpy complex array.
    a = [[1., 1], [1j, 2j]]
    np_value = np.array(a, dtype=np.complex128)
    tf_value = ops.convert_to_tensor(a, dtype=dtypes.complex128)
    self.assertAllEqual(tf_value.numpy(), np_value)


if __name__ == "__main__":
  test.main()
