# [MODULE DOC]
# Tests for the DTensor device as seen from Python: layout checking, async option, type-based
# dispatch, implicit copies of variables and integers, concurrent execution, string
# representation, cancellation, layout fetching, single-device meshes, unsupported ops, and
# pack/unpack on one-dimensional and two-dimensional meshes, including sparse tensors.
# Defines four test classes (general device behavior, pack/unpack on 1-D mesh, pack/unpack on
# 2-D mesh with 8 devices, and sparse pack/unpack) plus a few module constants and aliases for
# layout types. Mostly relies on the private device accessors of the DTensor api module.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for DTensorDevice in python."""
import os
import threading
from unittest import mock

from absl.testing import parameterized
import numpy as np

from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import dtensor_device
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_collective_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

# Convenient constants to use for tests.
_BATCH_DIM = "batch"
_MESH_DIM_X = "x"

# Shorter notation
Layout = layout_lib.Layout
Mesh = layout_lib.Mesh
UNSHARDED = layout_lib.UNSHARDED


class DTensorDeviceTest(test_util.DTensorBaseTest, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for general DTensor device behavior on a one-dimensional mesh of two devices with
  # a single batch dimension. The setup stores the mesh for the current device type as an
  # attribute used by all tests. Also supports parameterized tests.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup. Builds a two-device mesh (dimension named batch) for each of CPU, GPU and
    # TPU, lets the base class choose the matching one and stores it as mesh. Returns nothing.
    super(DTensorDeviceTest, self).setUp()
    device_ids = test_util.create_device_ids_array((2,))
    local_device_ids = np.ravel(device_ids).tolist()
    mesh_dict = {  # pylint: disable=g-complex-comprehension
        device: Mesh(
            [_BATCH_DIM],
            device_ids,
            local_device_ids,
            test_util.create_device_list((2,), device),
        )
        for device in ("CPU", "GPU", "TPU")
    }
    self.mesh = self.configTestMesh(mesh_dict)

  def testInvalidLayout(self):
    # [FUNCTION DOC]
    # Copies a replicated tensor to the mesh, applies an identity, and asserts that checking it
    # against a batch-sharded layout raises a value error.
    a = api.copy_to_mesh(
        constant_op.constant([1.0]), Layout.replicated(self.mesh, rank=1)
    )
    b = array_ops.identity(a)
    with self.assertRaises(ValueError):
      api.check_layout(b, Layout.batch_sharded(self.mesh, _BATCH_DIM, rank=1))

  @parameterized.parameters(True, False)
  def testAsyncOption(self, is_async):
    # [FUNCTION DOC]
    # Parameterized over is_async (true or false). Resets the DTensor device with the given
    # async option, runs an identity on a replicated tensor under the default mesh, and asserts
    # the value is unchanged. It only exercises the option and cannot verify that execution was
    # really asynchronous. Always resets global DTensor state afterwards.
    try:
      # There isn't a great way to test whether something actually executed
      # synchronously; this test just exercises the option.
      api.reset_dtensor_device(is_async=is_async)
      with api._dtensor_device()._experimental_default_mesh(self.mesh):
        with ops.device_v2(api.device_name()):
          a = api.copy_to_mesh(
              constant_op.constant([1.0]), Layout.replicated(self.mesh, rank=1)
          )
          b = array_ops.identity(a)
      self.assertEqual([1.0], b.numpy())
    finally:
      api._reset()  # pylint: disable=protected-access

  def testBasicTypeBasedDispatch(self):
    # [FUNCTION DOC]
    # Copies a vector to the mesh replicated, then slices it and multiplies it by two using
    # normal operators. Asserts the slice has a replicated layout and the original and
    # product values are as expected.
    # KNOWN ISSUE: after the multiplication the layout check is applied to the slice again
    # instead of the product, so the product's layout is never verified.
    # Tests for b = Op(a).
    a = constant_op.constant([1.0, 2.0, 3.0, 4.0])
    a = api.copy_to_mesh(a, Layout.replicated(self.mesh, rank=1))

    # __getitem__
    b = a[2:-2]
    api.check_layout(b, Layout.replicated(self.mesh, rank=1))

    c = a * 2
    api.check_layout(b, Layout.replicated(self.mesh, rank=1))

    self.assertAllEqual(a.numpy(), [1., 2., 3., 4.])
    self.assertAllEqual(c.numpy(), [2., 4., 6., 8.])

  @mock.patch.dict(os.environ, {"DTENSOR_ENABLE_MULTI_DEVICE_EXPANSION": "1"})
  def testMultiDeviceExpansion(self):
    # [FUNCTION DOC]
    # Same scenario as the basic dispatch test but with the multi-device expansion environment
    # flag patched on for this test only: slice and double a replicated vector, and assert the
    # values and the slice's replicated layout.
    # KNOWN ISSUE: the second layout check re-checks the slice rather than the product.
    # Tests for b = Op(a).
    a = constant_op.constant([1.0, 2.0, 3.0, 4.0])
    a = api.copy_to_mesh(a, Layout.replicated(self.mesh, rank=1))

    # __getitem__
    b = a[2:-2]
    api.check_layout(b, Layout.replicated(self.mesh, rank=1))

    c = a * 2
    api.check_layout(b, Layout.replicated(self.mesh, rank=1))

    self.assertAllEqual(a.numpy(), [1.0, 2.0, 3.0, 4.0])
    self.assertAllEqual(c.numpy(), [2.0, 4.0, 6.0, 8.0])

  def testNoImplicitCopyOnForLargeIntegerTensors(self):
    # [FUNCTION DOC]
    # Adds a mesh-resident int32 matrix to an ordinary 10x10 int32 tensor and expects an
    # unimplemented error (large integer tensors are not implicitly copied), while adding a
    # small 10-element int32 tensor succeeds.
    a = array_ops.ones([10, 10], dtype=dtypes.int32)
    a = api.copy_to_mesh(a, Layout.replicated(self.mesh, rank=2))
    big = array_ops.ones([10, 10], dtype=dtypes.int32)
    small = array_ops.ones([10], dtype=dtypes.int32)
    with self.assertRaises(errors_impl.UnimplementedError):
      a + big  # pylint:disable=pointless-statement
    a + small  # pylint:disable=pointless-statement

  def testConcurrentExecute(self):
    # [FUNCTION DOC]
    # Starts ten threads, each running a worker that ten times generates random values, copies
    # them to the mesh and runs a compiled function on them. Waits for all threads and asserts
    # each stored a non-null result. Checks that concurrent use of the device does not fail.
    results = {}

    def func(thread_id):
      # [FUNCTION DOC]
      # HELPER: thread worker for testConcurrentExecute. Takes a thread id; loops ten times
      # under the default mesh, creating a random vector, replicating it on the mesh, applying a
      # compiled update function and storing the latest result in the shared results mapping
      # under the thread id.
      @polymorphic_function.function
      def update_variable(initial_value, num_round):
        # [FUNCTION DOC]
        # Compiled helper returning the initial value plus initial value times the round number.
        y = math_ops.multiply(initial_value, num_round)
        return math_ops.add(initial_value, y)

      for n in range(10):
        with api._dtensor_device()._experimental_default_mesh(self.mesh):
          x = stateless_random_ops.stateless_random_uniform(
              [10], seed=(1, 2), minval=0, maxval=255
          )
          y = api.copy_to_mesh(x, Layout.replicated(self.mesh, rank=1))
          y = update_variable(y, n + 1)
          results[thread_id] = y

    threads = {}
    for a in range(10):
      t = threading.Thread(target=func, args=(a,))
      threads[a] = t
      t.start()

    for thrad_id, thread in threads.items():
      thread.join()
      self.assertIsNotNone(results[thrad_id])

  def testNoImplicitCopyOnForScalarVariableOnNonCPUMesh(self):
    # [FUNCTION DOC]
    # On non-CPU meshes, copying an ordinary variable to the mesh must raise an invalid-argument
    # error whose message says non-DTensor variables are only supported in limited cases and
    # mentions the shape and this test file. Skipped on the TFRT runtime and on CPU.
    self.skipForTfrt("b/235088250")
    self.skipForDeviceType(["CPU"], "CPU mesh implicit copy is allowed.")
    init_value = api.call_with_layout(
        array_ops.ones, shape=(1), layout=Layout.replicated(self.mesh, rank=1)
    )
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError,
        r"Using a non-DTensor variable with DTensor is only supported for..*\n"
        r".*Shape: \[1\].*\n"
        ".*device_test.py.*",
    ):
      api.copy_to_mesh(
          variables.Variable(init_value), Layout.replicated(self.mesh, rank=1)
      )

  @parameterized.named_parameters(
      test_util.product(
          [
              ("Int32", dtypes.int32),
              ("Float32", dtypes.float32),
              ("Int64", dtypes.int64),
              ("Float64", dtypes.float64),
          ],
          [
              (
                  "Scalar",
                  [],
              ),
              (
                  "RankOne",
                  [1],
              ),
              ("RankTwo", [2, 2]),
          ],
      )
  )
  def testImplicitCopyVariableOnCPUMesh(self, dtype, shape):
    # [FUNCTION DOC]
    # Parameterized over dtype (int32, float32, int64, float64) and shape (scalar, rank one,
    # 2x2). Creates an ordinary variable of ones and a zeros value, then under a default mesh
    # runs a compiled assignment of a packed replicated zeros DTensor to the variable (implicit
    # copy of the variable). Asserts the variable reads back as zeros with the replicated layout.
    # Skipped on TFRT, and on GPU/TPU where implicit variable copy is not allowed.
    self.skipForTfrt("b/235088250")
    self.skipForDeviceType(
        ["GPU", "TPU"], "Variable implicit copy is only allowed for CPU mesh.")

    variable = variables.Variable(array_ops.ones(shape=shape, dtype=dtype))
    new_value = array_ops.zeros(shape=shape, dtype=dtype)

    @polymorphic_function.function
    def assign_function(v, new_value):
      # [FUNCTION DOC]
      # Compiled helper: assigns new_value to variable v and returns the assignment result.
      return v.assign(new_value)

    layout = Layout.replicated(self.mesh, rank=len(shape))
    # Run explicitly on the dtensor device with a default mesh since
    # we do not have any registered mesh to broadcast the inputs to.
    with api.default_mesh(self.mesh):
      assign_function(variable, api.pack([new_value] * self.mesh.size, layout))
      read_value = variable.read_value()
    self.assertDTensorEqual(new_value, layout, read_value)

  def testNumpyCallWithReplicatedInput(self):
    # [FUNCTION DOC]
    # Replicates a vector on the mesh and asserts converting it to a numpy array returns the
    # original values.
    a = constant_op.constant([1.0, 2.0, 3.0, 4.0])
    a = api.copy_to_mesh(a, Layout.replicated(self.mesh, rank=1))
    b = a.numpy()
    self.assertAllEqual(b, [1., 2., 3., 4.])

  def testTensorIteration(self):
    # [FUNCTION DOC]
    # Replicates a vector on the mesh, iterates over it and asserts the first element is 1.
    a = constant_op.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    a = api.copy_to_mesh(a, Layout.replicated(self.mesh, rank=1))
    iterator = iter(a)
    self.assertAllClose(1., next(iterator))

  def testCopyToMeshWithSameLayout(self):
    # [FUNCTION DOC]
    # Copies a tensor to the mesh twice with the same replicated layout (the second copy is a
    # no-op or equivalent) and asserts the result still has that layout.
    a = constant_op.constant([1.0, 2.0, 3.0, 4.0])
    a = api.copy_to_mesh(a, Layout.replicated(self.mesh, rank=1))
    a = api.copy_to_mesh(a, Layout.replicated(self.mesh, rank=1))
    api.check_layout(a, Layout.replicated(self.mesh, rank=1))

  def testSetDefaultLayoutEager(self):
    # [FUNCTION DOC]
    # Replicates a 2x1 tensor, then reshapes it to a vector inside a scope setting the default
    # output layout to replicated rank one. Asserts the result has that layout and values 1, 1.
    tensor = constant_op.constant([[1.0], [1.0]])
    tensor = api.copy_to_mesh(tensor, Layout.replicated(self.mesh, rank=2))
    with api._dtensor_device()._default_layout(
        Layout.replicated(self.mesh, rank=1)):
      tensor = array_ops.reshape(tensor, [-1])
    api.check_layout(tensor, Layout.replicated(self.mesh, rank=1))
    self.assertAllClose([1., 1.], tensor.numpy())

  def testSetDefaultLayoutFunction(self):
    # [FUNCTION DOC]
    # Calls a compiled function that returns two reshaped constants under a scope that sets the
    # default layout to batch-sharded rank one. Asserts the first output has the batch-sharded
    # layout while the second is replicated, then relayouts the first to replicated and checks
    # both values are 1, 1.

    @polymorphic_function.function
    def func():
      # [FUNCTION DOC]
      # Compiled helper: reshapes a 2x1 constant of ones to a vector and returns it twice.
      tensor = constant_op.constant([[1.0], [1.0]])
      return array_ops.reshape(tensor, [-1]), array_ops.reshape(tensor, [-1])

    with api._dtensor_device()._default_layout(
        Layout.batch_sharded(self.mesh, batch_dim=_BATCH_DIM, rank=1)
    ):
      tensor1, tensor2 = func()

    api.check_layout(
        tensor1, Layout.batch_sharded(self.mesh, batch_dim=_BATCH_DIM, rank=1)
    )
    api.check_layout(tensor2, Layout.replicated(self.mesh, rank=1))
    tensor1 = api.relayout(tensor1, Layout.replicated(self.mesh, rank=1))

    self.assertAllClose([1.0, 1.0], tensor1.numpy())
    self.assertAllClose([1.0, 1.0], tensor2.numpy())

  @parameterized.named_parameters(
      # pylint: disable=unnecessary-lambda
      # Needed for the DVariable monkey patch to work.
      ("Variable", lambda x: d_variable.DVariable(x)),
      # pylint: enable=unnecessary-lambda
      ("Tensor", lambda x: x),
  )
  def testStringRepresentation(self, transform):
    # [FUNCTION DOC]
    # Parameterized with transform: either wrap the value in a DTensor variable or leave it as a
    # tensor. Builds a replicated scalar 8 and a batch-sharded vector from values 8 and 9,
    # applies the transform, and asserts the string form contains the values and the word layout.
    replicated = api.copy_to_mesh(
        constant_op.constant(8.0), Layout.replicated(self.mesh, rank=0)
    )
    replicated = transform(replicated)
    replicated_str = str(replicated)
    self.assertIn("8", replicated_str)
    self.assertIn("layout", replicated_str)

    sharded = api.pack(
        [constant_op.constant([8.0]), constant_op.constant([9.0])],
        layout=Layout([_BATCH_DIM], self.mesh),
    )
    sharded = transform(sharded)
    sharded_str = str(sharded)
    self.assertIn("8", sharded_str)
    self.assertIn("9", sharded_str)
    self.assertIn("layout", sharded_str)

  @parameterized.named_parameters(("Async", True), ("Sync", False))
  def testCancellation(self, is_async):
    # [FUNCTION DOC]
    # Parameterized over async and sync device modes. Creates a new DTensor device, packs one
    # value per mesh device where one device divides by zero, and runs a compiled function with
    # a collective reduction. Asserts an invalid-argument error about integer division by zero
    # rather than a deadlock, and that the device is still usable afterwards (identity and
    # unpack return the original pieces). Skipped on TFRT and TPU.
    self.skipForTfrt("b/181368626: support cancellation in tfrt.")
    self.skipForDeviceType(["TPU"], "b/195552283: Fix cancellation on TPU.")
    device = dtensor_device.DTensorDevice(meshes=[self.mesh], is_async=is_async)

    @polymorphic_function.function
    def f(x):
      # [FUNCTION DOC]
      # Compiled helper: divides x by itself as integers (fails on a zero), casts to float and
      # sums over axis 0, which needs a collective.
      # Integer division by 0 on one device, which returns a bad status.
      x = math_ops.cast(gen_math_ops.div(x=x, y=x), dtypes.float32)
      # A reduction requiring a collective, which would normally deadlock with
      # one of its participants missing.
      return math_ops.reduce_sum(x, axis=0)

    a = constant_op.constant([[1, 2]])
    b = constant_op.constant([[0, 1]])
    x = device.pack([a, b], layout=Layout([_BATCH_DIM, UNSHARDED], self.mesh))
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError, "Integer division by zero"
    ):
      y = f(x)
      y.numpy()
    z = array_ops.identity(x)
    self.assertAllClose([[[1, 2]], [[0, 1]]], device.unpack(z))

  def testCopyToMeshShapeFn(self):
    # [FUNCTION DOC]
    # Traces a function that copies a constant to the mesh and asserts the traced output has the
    # static shape of two elements (the shape function of the copy op works).

    @polymorphic_function.function
    def f():
      # [FUNCTION DOC]
      # Compiled helper: makes a two-element constant, replicates it on the mesh and returns it.
      c = constant_op.constant([1.0, 2.0])
      on_mesh = api.copy_to_mesh(c, Layout.replicated(self.mesh, rank=1))
      return on_mesh

    output, = f.get_concrete_function().outputs
    self.assertEqual([2], output.shape)

  def testUnpackInvalidInput(self):
    # [FUNCTION DOC]
    # Regression test: calls unpack with a nested list of floats given as a keyword argument and
    # expects a type error saying a tensor was expected.
    # Test for b/255629824
    with self.assertRaisesRegex(TypeError, "Expecting a Tensor"):
      api.unpack(
          **{
              "tensor": [[
                  41.8684053521925,
                  731.610023060566,
                  356.0701500440248,
                  9.62928117100512,
                  185.0041559439026,
                  225.87663065861508,
                  450.2403652750002,
                  268.7273627027147,
              ]]
          }
      )

  def testIsDTensorInvalidInput(self):
    # [FUNCTION DOC]
    # Regression test for invalid input. Passes an integer as the tensor keyword argument and
    # asserts the result is falsy.
    # KNOWN ISSUE: despite the name, it calls the layout-fetching function, not the is-DTensor
    # check, so it duplicates the fetch-layout test below with a weaker assertion.
    # Test for b/272381211
    self.assertFalse(api.fetch_layout(**{"tensor": -1024}))

  def testFetchLayoutInvalidInput(self):
    # [FUNCTION DOC]
    # Regression test: fetching the layout of an integer (given as keyword argument) returns the
    # null value rather than raising.
    # Test for b/272381211
    self.assertIsNone(api.fetch_layout(**{"tensor": -1024}))

  def testFetchLayoutForDVariablesReturnsCorrectLayout(self):
    # [FUNCTION DOC]
    # Creates a DTensor variable from a replicated rank-2 ones tensor and asserts fetching its
    # layout returns the same replicated layout.
    layout = Layout.replicated(self.mesh, 2)
    with api._dtensor_device()._experimental_default_mesh(self.mesh):
      dvariable = d_variable.DVariable(
          api.call_with_layout(
              array_ops.ones, shape=[2, 3], dtype=dtypes.float32, layout=layout
          )
      )
    self.assertEqual(layout, api.fetch_layout(dvariable))

  def testFetchLayoutForDTensorReturnsCorrectLayout(self):
    # [FUNCTION DOC]
    # Creates a replicated rank-2 ones DTensor and asserts fetching its layout returns it.
    layout = Layout.replicated(self.mesh, 2)
    tensor = api.call_with_layout(
        array_ops.ones, shape=[2, 3], dtype=dtypes.float32, layout=layout
    )
    self.assertEqual(layout, api.fetch_layout(tensor))

  def testFetchLayoutForRegularTensorsThrowsError(self):
    # [FUNCTION DOC]
    # Fetching the layout of an ordinary tensor raises an invalid-argument error saying the
    # tensor must be placed on the layout device.
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError,
        "FetchLayout expects a tensor placed on the layout device.",
    ):
      api.fetch_layout(constant_op.constant([2, 3]))

  def testFetchLayoutNotEagerlyRaisesRuntimeError(self):
    # [FUNCTION DOC]
    # Calling the layout-fetching function inside a compiled function must raise a runtime error
    # saying it has to be called eagerly.

    @polymorphic_function.function
    def f(dtensor_input):
      # [FUNCTION DOC]
      # Compiled helper that fetches the layout of its input (illegal in graph mode).
      api.fetch_layout(dtensor_input)

    with self.assertRaisesRegex(RuntimeError,
                                "`fetch_layout` must be called eagerly."):
      f(
          api.copy_to_mesh(
              constant_op.constant(1.0), Layout.replicated(self.mesh, rank=0)
          )
      )

  def testIsDTensor(self):
    # [FUNCTION DOC]
    # Asserts the is-DTensor predicate is false for an ordinary tensor, list, mapping, integer
    # and arbitrary object, and true for a DTensor and a DTensor variable.
    normal_tensor = array_ops.zeros(shape=[10, 10])
    self.assertFalse(api.is_dtensor(normal_tensor))

    layout = Layout.replicated(self.mesh, rank=1)
    d_tensor = api.call_with_layout(array_ops.zeros, layout=layout, shape=[10])
    self.assertTrue(api.is_dtensor(d_tensor))

    var = d_variable.DVariable(d_tensor)
    self.assertTrue(api.is_dtensor(var))

    self.assertFalse(api.is_dtensor([0, 1]))
    self.assertFalse(api.is_dtensor({False: True}))

    self.assertFalse(api.is_dtensor(1))

    class C:
      # [CLASS DOC]
      # HELPER: empty placeholder class used as an arbitrary non-tensor object for the
      # is-DTensor check.
      pass

    self.assertFalse(api.is_dtensor(C()))

  def testIsDTensorNotEagerlyRaisesRuntimeError(self):
    # [FUNCTION DOC]
    # Calling the is-DTensor predicate inside a compiled function must raise a runtime error
    # saying it has to be called eagerly.

    @polymorphic_function.function
    def f(dtensor_input):
      # [FUNCTION DOC]
      # Compiled helper that calls the is-DTensor predicate on its input (illegal in graph mode).
      api.is_dtensor(dtensor_input)

    with self.assertRaisesRegex(
        RuntimeError, "`is_dtensor` must be called eagerly."):
      f(
          api.copy_to_mesh(
              constant_op.constant(1.0), Layout.replicated(self.mesh, 0)
          )
      )

  def testSingleDeviceMesh(self):
    # [FUNCTION DOC]
    # Builds a mesh from the single device CPU:0. Under it as default mesh, a scalar constant is
    # an ordinary tensor placed on CPU:0, while a ones tensor is a DTensor whose layout mesh
    # equals that single-device mesh.
    # FIXME(b/274647196): Add a mesh_util API that takes CPU:0.
    cpu0_mesh = Mesh.from_device("/job:localhost/replica:0/task:0/device:CPU:0")
    with api.default_mesh(cpu0_mesh):
      a = constant_op.constant(1.0)

    self.assertFalse(api.is_dtensor(a))
    self.assertIn("CPU:0", a.device)

    with api.default_mesh(cpu0_mesh):
      b = array_ops.ones(shape=(3, 3))

    self.assertTrue(api.is_dtensor(b))
    self.assertEqual(api.fetch_layout(b).mesh, cpu0_mesh)

  def testUnsupportedOpReplicatedInput(self):
    # [FUNCTION DOC]
    # Under the default mesh, runs a single-member collective reduce op (no SPMD expander) on
    # unsharded input. Asserts the result is not a DTensor and equals the input values.
    with api.default_mesh(self.mesh):
      t = array_ops.ones(shape=(8, 3))
      a = gen_collective_ops.collective_reduce_v2(
          t,
          group_size=1,
          group_key=1030,
          instance_key=1,
          merge_op="Add",
          final_op="Id",
          ordering_token=[],
      )

    self.assertFalse(api.is_dtensor(a))
    self.assertAllClose(a.numpy(), t.numpy())

  def testUnsupportedOpShardedInput(self):
    # [FUNCTION DOC]
    # Same collective reduce op as above but on a batch-sharded input: expects an unimplemented
    # error containing "not supported" because the op has no SPMD expander.
    with api.default_mesh(self.mesh):
      t = array_ops.ones(shape=(8, 3))
      t = api.relayout(
          t, Layout.batch_sharded(mesh=self.mesh, batch_dim=_BATCH_DIM, rank=2)
      )
      with self.assertRaisesRegex(
          errors_impl.UnimplementedError, "not supported"
      ):
        # This is an Op that we don't have a SPMD expander.
        gen_collective_ops.collective_reduce_v2(
            t,
            group_size=1,
            group_key=1030,
            instance_key=1,
            merge_op="Add",
            final_op="Id",
            ordering_token=[],
        )


class DTensorPackUnpackOnOneDMeshTest(test_util.DTensorBaseTest):
  # [CLASS DOC]
  # Test case for packing and unpacking DTensors on a one-dimensional two-device mesh (batch
  # dimension), including error cases for variables, regular tensors and non-eager calls.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup. Builds a two-device mesh with a batch dimension for CPU, GPU and TPU, lets
    # the base class pick the right one and stores it as mesh. Returns nothing.
    super(DTensorPackUnpackOnOneDMeshTest, self).setUp()
    global_ids = test_util.create_device_ids_array((2,))
    local_device_ids = np.ravel(global_ids).tolist()
    mesh_dict = {  # pylint: disable=g-complex-comprehension
        device: Mesh(
            [_BATCH_DIM],
            global_ids,
            local_device_ids,
            test_util.create_device_list((2,), device),
        )
        for device in ("CPU", "GPU", "TPU")
    }
    self.mesh = self.configTestMesh(mesh_dict)

  def testUnpack(self):
    # [FUNCTION DOC]
    # Replicates a scalar on the mesh and asserts unpacking yields one copy per device, [1, 1].
    with api.default_mesh(self.mesh):
      v = constant_op.constant(1.0)
      v = api.copy_to_mesh(v, Layout.replicated(self.mesh, rank=0))
      self.assertAllClose([1.0, 1.0], api.unpack(v))

  def testUnpackVariables(self):
    # [FUNCTION DOC]
    # Unpacking a DTensor variable directly through the device raises a type error saying a
    # tensor was expected.
    v0 = d_variable.DVariable(
        api.call_with_layout(
            array_ops.ones,
            shape=[2, 3],
            dtype=dtypes.float32,
            layout=Layout.replicated(self.mesh, 2),
        )
    )
    with self.assertRaisesRegex(TypeError, "Expecting a Tensor"):
      api._dtensor_device().unpack(v0)

  def testUnpackingRegularTensorRaisesInvalidArgumentError(self):
    # [FUNCTION DOC]
    # Unpacking an ordinary tensor raises an invalid-argument error saying the tensor must be
    # placed on the DTensor device.
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError,
        "DTensorUnpack expects a tensor placed on the DTensor device",
    ):
      api._dtensor_device().unpack(constant_op.constant([1.0, 2.0]))

  def testUnpackingNotEagerlyRaisesRuntimeError(self):
    # [FUNCTION DOC]
    # Calling unpack inside a compiled function raises a runtime error saying it must be called
    # eagerly.

    @polymorphic_function.function
    def f(dtensor_input):
      # [FUNCTION DOC]
      # Compiled helper that unpacks its input through the device (illegal in graph mode).
      api._dtensor_device().unpack(dtensor_input)

    with self.assertRaisesRegex(
        RuntimeError, "`unpack` must be called eagerly."):
      f(
          api.copy_to_mesh(
              constant_op.constant(1.0), Layout.replicated(self.mesh, rank=0)
          )
      )

  def testPack(self):
    # [FUNCTION DOC]
    # Packs two vectors into a batch-sharded DTensor. Asserts the layout, the packed shape of
    # four elements, and that unpacking returns the original two vectors in order.
    a = constant_op.constant([1.0, 2.0])
    b = constant_op.constant([3.0, 4.0])
    with ops.device_v2(api.device_name()):
      packed_tensor = api.pack(
          [a, b], layout=Layout.batch_sharded(self.mesh, _BATCH_DIM, rank=1)
      )
      api.check_layout(
          packed_tensor, Layout.batch_sharded(self.mesh, _BATCH_DIM, rank=1)
      )
      self.assertAllEqual([
          4,
      ], packed_tensor.shape)
      unpacked_tensor = api.unpack(packed_tensor)
      self.assertAllClose([1., 2.], unpacked_tensor[0])
      self.assertAllClose([3., 4.], unpacked_tensor[1])

  def testPackingNotEagerlyRaisesRuntimeError(self):
    # [FUNCTION DOC]
    # Calling pack inside a compiled function raises a runtime error saying it must be called
    # eagerly.

    @polymorphic_function.function
    def f(a):
      # [FUNCTION DOC]
      # Compiled helper that packs two copies of its input with a replicated layout (illegal in
      # graph mode).
      api.pack([a, a], layout=Layout.replicated(self.mesh, rank=1))

    with self.assertRaisesRegex(RuntimeError, "`pack` must be called eagerly."):
      f(constant_op.constant([1.0]))

  def testPackingVariablesRaisesError(self):
    # [FUNCTION DOC]
    # Packing ordinary variables through the device raises an invalid-argument error saying
    # variable input is not supported.
    with self.assertRaisesRegex(
        errors_impl.InvalidArgumentError, "Variable input is not supported."
    ):
      api._dtensor_device().pack(
          [
              variables.Variable(array_ops.ones([2, 3])),
              variables.Variable(array_ops.ones([2, 3])),
          ],
          Layout.replicated(self.mesh, rank=2),
      )

  def testPackDevice(self):
    # [FUNCTION DOC]
    # Packs two vectors with a batch-sharded layout, unpacks, and asserts each unpacked piece is
    # placed on the corresponding local device of the mesh in order.
    a = constant_op.constant([1.0, 2.0])
    b = constant_op.constant([3.0, 4.0])
    with ops.device_v2(api.device_name()):
      packed_tensor = api.pack(
          [a, b], layout=Layout.batch_sharded(self.mesh, _BATCH_DIM, rank=1)
      )
      unpacked_tensor = api.unpack(packed_tensor)
      self.assertAllEqual(self.mesh.local_devices(),
                          [t.device for t in unpacked_tensor])

  def testPackScalar(self):
    # [FUNCTION DOC]
    # Packs two scalars with a rank-zero layout. Asserts the layout, an empty (scalar) shape,
    # and that unpacking returns both scalars.
    a = constant_op.constant(1.0)
    with ops.device_v2(api.device_name()):
      packed_layout = Layout([], self.mesh)
      packed_tensor = api.pack([a, a], layout=packed_layout)
      api.check_layout(packed_tensor, packed_layout)
      self.assertAllEqual([], packed_tensor.shape)
      unpacked_tensor = api.unpack(packed_tensor)
      self.assertAllClose([a, a], unpacked_tensor)

  def testPackHigherRankValue(self):
    # [FUNCTION DOC]
    # Packs two 2x2x3 integer tensors onto the one-dimensional mesh with the first dimension
    # sharded over batch. Asserts the layout and that the packed global shape is 4x2x3.
    # Pack a rank 3 matrix into a 1d mesh.
    a = constant_op.constant(
        [[[1, 2, 3], [4, 5, 6]], [[2, 3, 4], [5, 6, 7]]]
    )  # 2x2x3
    b = constant_op.constant(
        [[[3, 2, 1], [6, 5, 4]], [[4, 3, 2], [7, 6, 5]]]
    )  # 2x2x3
    pack_layout = Layout([_BATCH_DIM, UNSHARDED, UNSHARDED], self.mesh)
    with ops.device_v2(api.device_name()):
      # pack to 4x2x3
      packed_tensor = api.pack([a, b], layout=pack_layout)
      api.check_layout(packed_tensor, pack_layout)
      self.assertAllEqual([4, 2, 3], packed_tensor.shape)


class DTensorPackUnpackOnTwoDMeshTest(test_util.DTensorBaseTest):
  # [CLASS DOC]
  # Test case for packing and unpacking on a 2x4 mesh with dimensions batch and x (eight
  # devices); on TPU it requires exactly 8 cores. Stores the mesh as an attribute.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup. Skips on TPU unless there are exactly 8 devices, builds a 2x4 mesh for CPU,
    # GPU and TPU, and stores the one selected for the current device type as mesh.
    super().setUp()

    self.skipForDeviceType(["TPU"],
                           "all tests require 8 TPU cores.",
                           unless_device_count_equals_to=8)

    global_ids = test_util.create_device_ids_array((2, 4))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {  # pylint: disable=g-complex-comprehension
        device: Mesh(
            [_BATCH_DIM, _MESH_DIM_X],
            global_ids,
            local_ids,
            test_util.create_device_list((2, 4), device),
        )
        for device in ("CPU", "GPU", "TPU")
    }
    self.mesh = self.configTestMesh(mesh_dict)

  def testPackWithScalars(self):
    # [FUNCTION DOC]
    # Packs eight copies of a scalar with a rank-zero layout. Asserts the layout, scalar shape,
    # and that unpacking returns all eight scalars.
    a = constant_op.constant(1.23)
    with ops.device_v2(api.device_name()):
      packed_layout = Layout([], self.mesh)
      packed_tensor = api.pack([a, a, a, a, a, a, a, a], layout=packed_layout)

      api.check_layout(packed_tensor, packed_layout)

      self.assertAllEqual([], packed_tensor.shape)
      unpacked_tensor = api.unpack(packed_tensor)
      self.assertAllClose([a, a, a, a, a, a, a, a], unpacked_tensor)

  def testPackWithScalarsWithInvalidRank(self):
    # [FUNCTION DOC]
    # Packing eight scalars with a rank-one layout raises an invalid-argument error saying the
    # packed layout should have the same rank as the inputs.
    a = constant_op.constant(1.23)
    with ops.device_v2(api.device_name()):
      invalid_packed_layout = Layout([UNSHARDED], self.mesh)
      with self.assertRaisesRegex(
          errors_impl.InvalidArgumentError,
          "Packed layout should have the same rank",
      ):
        _ = api.pack([a, a, a, a, a, a, a, a], layout=invalid_packed_layout)

  def testPackWithBatchSharding(self):
    # [FUNCTION DOC]
    # Packs eight 2x1 tensors (four copies of one, four of another) with a batch-sharded rank-2
    # layout. Asserts the layout, a packed shape of 4x1, and that unpacking returns the eight.
    a = constant_op.constant([[1.0], [2.0]])
    b = constant_op.constant([[3.0], [4.0]])
    with ops.device_v2(api.device_name()):
      packed_tensor = api.pack(
          [a, a, a, a, b, b, b, b],
          layout=Layout.batch_sharded(self.mesh, _BATCH_DIM, rank=2),
      )
      api.check_layout(
          packed_tensor, Layout.batch_sharded(self.mesh, _BATCH_DIM, rank=2)
      )
      self.assertAllEqual([4, 1], packed_tensor.shape)
      unpacked_tensor = api.unpack(packed_tensor)
    self.assertAllClose([a, a, a, a, b, b, b, b], unpacked_tensor)

  def testPackWithFullReplicated(self):
    # [FUNCTION DOC]
    # Packs eight copies of a 2x1 tensor with a fully unsharded layout. Asserts the layout, the
    # packed shape 2x1 (unchanged), and that unpacking returns the eight copies.
    a = constant_op.constant([[1.0], [2.0]])
    with ops.device_v2(api.device_name()):
      packed_layout = Layout([UNSHARDED, UNSHARDED], self.mesh)
      packed_tensor = api.pack([a, a, a, a, a, a, a, a], layout=packed_layout)
      api.check_layout(packed_tensor, packed_layout)
      self.assertAllEqual([2, 1], packed_tensor.shape)
      unpacked_tensor = api.unpack(packed_tensor)
    self.assertAllClose([a, a, a, a, a, a, a, a], unpacked_tensor)

  def testFillUsesSpecifiedLayout(self):
    # [FUNCTION DOC]
    # Under the default mesh and a scope setting the default layout to batch-sharded rank 2,
    # fills a 4x1 tensor with ones. Asserts the output layout, the global shape [4, 1], and that
    # unpacking returns eight 2x1 pieces of ones.
    with api.default_mesh(self.mesh):
      # TODO(allenl): Figure out why the embedded constant (triggered for
      # dtypes.int32) gets a sharded layout by default.
      dims = constant_op.constant([4, 1], dtype=dtypes.int64)
      value = constant_op.constant(1.0)
      with api._dtensor_device()._default_layout(
          Layout.batch_sharded(self.mesh, _BATCH_DIM, rank=2)):
        filled = array_ops.fill(value=value, dims=dims)
      api.check_layout(
          filled, Layout.batch_sharded(self.mesh, _BATCH_DIM, rank=2)
      )
      unpacked_tensor = api.unpack(filled)
    self.assertAllClose(8 * [[[1.], [1.]]], unpacked_tensor)
    self.assertEqual([4, 1], filled.shape)


class DTensorSparse(test_util.DTensorBaseTest):
  # [CLASS DOC]
  # Test case for packing and unpacking sparse tensors on a 2x4 mesh (batch and x dimensions,
  # eight devices; TPU requires exactly 8 cores). Stores the mesh as an attribute.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup. Skips on TPU unless there are exactly 8 devices, builds a 2x4 mesh for CPU,
    # GPU and TPU, and stores the one selected for the current device type as mesh.
    super().setUp()

    self.skipForDeviceType(["TPU"],
                           "all tests require 8 TPU cores.",
                           unless_device_count_equals_to=8)

    global_ids = test_util.create_device_ids_array((2, 4))
    local_ids = np.ravel(global_ids).tolist()
    mesh_dict = {  # pylint: disable=g-complex-comprehension
        device: Mesh(
            [_BATCH_DIM, _MESH_DIM_X],
            global_ids,
            local_ids,
            test_util.create_device_list((2, 4), device),
        )
        for device in ("CPU", "GPU", "TPU")
    }
    self.mesh = self.configTestMesh(mesh_dict)

  @parameterized.named_parameters(
      test_util.product([("Replicated", "replicated"), ("Sharded", "batch")], [(
          "RankTwo",
          [2, 4],
      ), (
          "RankThree",
          [2, 2, 3],
      )]))
  def testPackUnpackReturnsCorrectValuesAndDevices(self, sharding, shape):
    # [FUNCTION DOC]
    # Parameterized over sharding kind (replicated or batch) and a shape (rank two or three).
    # Builds two sparse tensors from random dense values, packs eight (all the first, or four of
    # each) with the matching layout, unpacks, and asserts the global shape (doubled on the
    # first dimension for batch sharding), the dense values, and that indices and values of each
    # piece are on the mesh's local devices in order.
    a = sparse_ops.from_dense(
        stateless_random_ops.stateless_random_uniform(shape, seed=[0, 1])
    )
    b = sparse_ops.from_dense(
        stateless_random_ops.stateless_random_uniform(shape, seed=[0, 1])
    )

    if sharding == "replicated":
      layout = Layout(["unsharded"] * len(shape), self.mesh)
      input_tensor = 8 * [a]
      expected = 8 * [sparse_ops.sparse_tensor_to_dense(a)]
      expected_shape = shape
    else:
      layout = Layout([_BATCH_DIM] + ["unsharded"] * (len(shape) - 1),
                      self.mesh)
      input_tensor = 4 * [a] + 4 * [b]
      expected = 4 * [sparse_ops.sparse_tensor_to_dense(a)] + 4 * [
          sparse_ops.sparse_tensor_to_dense(b)
      ]
      expected_shape = [shape[0] * 2] + shape[1:]

    with ops.device_v2(api._dtensor_device().name):
      packed_tensor = api.pack(input_tensor, layout)
      api.check_layout(packed_tensor, layout)
      unpacked_tensor = api.unpack(packed_tensor)

    got = [sparse_ops.sparse_tensor_to_dense(t) for t in unpacked_tensor]

    # Check shape of packed tensor.
    self.assertAllEqual(expected_shape, packed_tensor.shape)
    # Check values.
    self.assertAllClose(expected, got)
    # Check devices.
    self.assertAllEqual(self.mesh.local_devices(),
                        [t.indices.device for t in unpacked_tensor])
    self.assertAllEqual(self.mesh.local_devices(),
                        [t.values.device for t in unpacked_tensor])

  def testPackingMixedTensorTypesRaisesTypeError(self):
    # [FUNCTION DOC]
    # Packing four dense and four sparse tensors together raises a type error saying sparse
    # tensors cannot be packed with ordinary tensors.
    tensor = stateless_random_ops.stateless_random_uniform([2, 4], seed=[0, 1])
    sparse_tensor = sparse_ops.from_dense(tensor)
    with ops.device_v2(api.device_name()):
      with self.assertRaisesRegex(TypeError,
                                  "Cannot Pack SparseTensors with Tensors."):
        api.pack(
            4 * [tensor] + 4 * [sparse_tensor],
            Layout.replicated(self.mesh, rank=2),
        )

  def testPackingTensorsWithDifferentShapesRaisesTypeError(self):
    # [FUNCTION DOC]
    # Packing four 2x2 sparse tensors with four 4x4 sparse tensors raises a type error saying all
    # input sparse tensors must have the same shape.
    a = sparse_ops.from_dense(
        stateless_random_ops.stateless_random_uniform([2, 2], seed=[0, 1])
    )
    b = sparse_ops.from_dense(
        stateless_random_ops.stateless_random_uniform([4, 4], seed=[0, 1])
    )
    with ops.device_v2(api.device_name()):
      with self.assertRaisesRegex(
          TypeError, "All input SparseTensors to Pack must be same shape."):
        api.pack(4 * [a] + 4 * [b], Layout.replicated(self.mesh, rank=2))

  def testPackingSparseTensorsReturnsCorrectLayout(self):
    # [FUNCTION DOC]
    # Packs eight copies of a 16x16 sparse tensor with a replicated rank-2 layout and asserts
    # the packed result has that layout.
    layout = Layout.replicated(self.mesh, 2)
    a = sparse_ops.from_dense(
        stateless_random_ops.stateless_random_uniform([16, 16], seed=[0, 1])
    )
    with ops.device_v2(api.device_name()):
      api.check_layout(api.pack(8 * [a], layout), layout)


if __name__ == "__main__":
  test.main()
