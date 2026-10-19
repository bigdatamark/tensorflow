# [MODULE DOC]
# Tests for the DLPack interop functions (converting tensors to and from DLPack capsules).
# Builds a parameterized matrix over many dtypes (integers, floats, bfloat16, complex, bool) and
# several shapes (including empty and scalar shapes), then checks round trips, single-consumption
# of capsules, context reset behavior and error handling for unsupported inputs.
# When run as a script it enables eager execution before starting the test runner.
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
"""Tests for DLPack functions."""

from absl.testing import parameterized
import numpy as np

from tensorflow.python.dlpack import dlpack
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


int_dtypes = [
    np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32,
    np.uint64
]
float_dtypes = [np.float16, np.float32, np.float64]
complex_dtypes = [np.complex64, np.complex128]
dlpack_dtypes = (
    int_dtypes + float_dtypes + [dtypes.bfloat16] + complex_dtypes + [np.bool_]
)

testcase_shapes = [(), (1,), (2, 3), (2, 0), (0, 7), (4, 1, 2)]


def FormatShapeAndDtype(shape, dtype):
  # [FUNCTION DOC]
  # HELPER: builds test case names for the parameterized DLPack tests.
  # Takes a shape (sequence of dimension sizes) and a dtype; returns a string made of an underscore,
  # the dtype text and the comma-separated dimensions in square brackets.
  return "_{}[{}]".format(str(dtype), ",".join(map(str, shape)))


def GetNamedTestParameters():
  # [FUNCTION DOC]
  # HELPER: generates the parameter list for the round-trip test.
  # Takes no inputs. Returns a list of mappings, one per combination of dtype and shape, each with
  # a test case name, the dtype and the shape.
  result = []
  for dtype in dlpack_dtypes:
    for shape in testcase_shapes:
      result.append({
          "testcase_name": FormatShapeAndDtype(shape, dtype),
          "dtype": dtype,
          "shape": shape
      })
  return result


class DLPackTest(parameterized.TestCase, test.TestCase):
  # [CLASS DOC]
  # Parameterized test case covering the DLPack conversion functions to_dlpack and from_dlpack.
  # No shared fixtures; each test builds its own tensors.

  @parameterized.named_parameters(GetNamedTestParameters())
  def testRoundTrip(self, dtype, shape):
    # [FUNCTION DOC]
    # Scenario: for each dtype and shape, create a seeded random array (zeros for bool), make a
    # tensor from it (copied to an accelerator if available), export it as a DLPack capsule, delete
    # the original tensor, and import the capsule back into a tensor.
    # Asserts the values match the original array, and that the device is the CPU for 32-bit
    # integer tensors (which always live on CPU) and otherwise the same as the original device.
    np.random.seed(42)
    if dtype == np.bool_:
      np_array = np.random.randint(0, 1, shape, np.bool_)
    else:
      np_array = np.random.randint(0, 10, shape)
    # copy to gpu if available
    tf_tensor = array_ops.identity(constant_op.constant(np_array, dtype=dtype))
    tf_tensor_device = tf_tensor.device
    tf_tensor_dtype = tf_tensor.dtype
    dlcapsule = dlpack.to_dlpack(tf_tensor)
    del tf_tensor  # should still work
    tf_tensor2 = dlpack.from_dlpack(dlcapsule)
    self.assertAllClose(np_array, tf_tensor2)
    if tf_tensor_dtype == dtypes.int32:
      # int32 tensor is always on cpu for now
      self.assertEqual(tf_tensor2.device,
                       "/job:localhost/replica:0/task:0/device:CPU:0")
    else:
      self.assertEqual(tf_tensor_device, tf_tensor2.device)

  def testRoundTripWithoutToDlpack(self):
    # [FUNCTION DOC]
    # Scenario: a CPU tensor of random integers is passed directly to numpy's from-DLPack
    # conversion, without calling to_dlpack first. Asserts the resulting array equals the source.
    np_array = np.random.randint(0, 10, [42])
    self.assertAllEqual(
        np.from_dlpack(constant_op.constant(np_array).cpu()), np_array
    )

  def testTensorsCanBeConsumedOnceOnly(self):
    # [FUNCTION DOC]
    # Scenario: export a float tensor to a capsule and import it once successfully. A second import
    # of the same capsule must raise an error whose message says a DLPack tensor may be consumed at
    # most once.
    np.random.seed(42)
    np_array = np.random.randint(0, 10, (2, 3, 4))
    tf_tensor = constant_op.constant(np_array, dtype=np.float32)
    dlcapsule = dlpack.to_dlpack(tf_tensor)
    del tf_tensor  # should still work
    _ = dlpack.from_dlpack(dlcapsule)

    def ConsumeDLPackTensor():
      # [FUNCTION DOC]
      # HELPER: callable passed to the error assertion; imports the already consumed capsule again
      # and returns nothing (it is expected to raise).
      dlpack.from_dlpack(dlcapsule)  # Should can be consumed only once

    self.assertRaisesRegex(Exception,
                           ".*a DLPack tensor may be consumed at most once.*",
                           ConsumeDLPackTensor)

  def testDLPackFromWithoutContextInitialization(self):
    # [FUNCTION DOC]
    # Scenario: export a scalar tensor to a capsule, reset the eager context, then import the
    # capsule. Asserts only that no error is raised.
    tf_tensor = constant_op.constant(1)
    dlcapsule = dlpack.to_dlpack(tf_tensor)
    # Resetting the context doesn't cause an error.
    context._reset_context()
    _ = dlpack.from_dlpack(dlcapsule)

  def testUnsupportedTypeToDLPack(self):
    # [FUNCTION DOC]
    # Scenario: exporting a tensor of the unsupported 16-bit quantized integer dtype must raise an
    # error whose message says the type is not supported by dlpack.

    def UnsupportedQint16():
      # [FUNCTION DOC]
      # HELPER: callable passed to the error assertion; builds a 16-bit quantized integer tensor
      # and tries to export it to DLPack. Returns nothing (it is expected to raise).
      tf_tensor = constant_op.constant([[1, 4], [5, 2]], dtype=dtypes.qint16)
      _ = dlpack.to_dlpack(tf_tensor)

    self.assertRaisesRegex(Exception, ".* is not supported by dlpack",
                           UnsupportedQint16)

  def testMustPassTensorArgumentToDLPack(self):
    # [FUNCTION DOC]
    # Scenario: exporting a plain Python list instead of a tensor must raise an invalid-argument
    # error saying the argument must be a TF tensor, not a Python object.
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        "The argument to `to_dlpack` must be a TF tensor, not Python object"):
      dlpack.to_dlpack([1])


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
