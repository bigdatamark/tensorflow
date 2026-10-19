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
# [MODULE DOC]
# Tests for DTensor programs that span multiple meshes. Defines module-level meshes (1D CPU,
# GPU, TPU, a CPU mesh with dimension y, and a one-device CPU mesh) and one test class,
# MultiMeshTest. Import-time side effects: builds the mesh objects. Run via the test main.
"""Tests for multiple meshes in DTensor."""

from absl.testing import parameterized
import numpy as np

from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import api
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import d_variable
from tensorflow.dtensor.python import layout as layout_lib
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

Layout = layout_lib.Layout
Mesh = layout_lib.Mesh
UNSHARDED = layout_lib.UNSHARDED

# Makes a 1D mesh with dimension X(2).
_MESH_DIM_X = 'x'
_MESH_DIM_Y = 'y'
_DEVICES_IDS = test_util.create_device_ids_array((2,))
_ONE_D_CPU_MESH = Mesh(
    [_MESH_DIM_X],
    _DEVICES_IDS,
    np.ravel(_DEVICES_IDS).tolist(),
    test_util.create_device_list((2,), 'CPU'),
)
_ONE_D_GPU_MESH = Mesh(
    [_MESH_DIM_X],
    _DEVICES_IDS,
    np.ravel(_DEVICES_IDS).tolist(),
    test_util.create_device_list((2,), 'GPU'),
)
_ONE_D_TPU_MESH = Mesh(
    [_MESH_DIM_X],
    _DEVICES_IDS,
    np.ravel(_DEVICES_IDS).tolist(),
    test_util.create_device_list((2,), 'TPU'),
)
_ONE_D_CPU_MESH_Y = Mesh(
    [_MESH_DIM_Y],
    _DEVICES_IDS,
    np.ravel(_DEVICES_IDS).tolist(),
    test_util.create_device_list((2,), 'CPU'),
)

_OTHER_CPU_MESH = Mesh(
    [_MESH_DIM_X],
    _DEVICES_IDS[:1],
    np.ravel(_DEVICES_IDS[:1]).tolist(),
    test_util.create_device_list((1,), 'CPU'),
)


class MultiMeshTest(test_util.DTensorBaseTest):
  # [CLASS DOC]
  # Test case covering DTensor programs that use more than one mesh at once: copying tensors
  # between meshes (CPU, GPU, TPU, host meshes), default-mesh nesting, cross-mesh functions,
  # cancellation on failure, resource variables on another mesh and gradients across meshes.
  # Shared fixture: setUp defines first_mesh (a 2-device CPU mesh) and second_mesh (TPU or GPU
  # mesh if present, else a second CPU mesh with a different dimension name).

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture run before every test. Calls the base setUp, sets first_mesh to the 2-device CPU
    # mesh, and sets second_mesh to the TPU mesh when TPU is present, else the GPU mesh when GPU
    # is present, else a CPU mesh with dimension y. For non-TPU device types it resets the
    # logical devices to two of the preferred type, then initializes the accelerator system.
    # Returns nothing; side effects are on global device and DTensor runtime state.
    super(MultiMeshTest, self).setUp()
    self.first_mesh = _ONE_D_CPU_MESH
    if test_util.is_tpu_present():
      self.second_mesh = _ONE_D_TPU_MESH
    elif test_util.is_gpu_present():
      self.second_mesh = _ONE_D_GPU_MESH
    else:
      self.second_mesh = _ONE_D_CPU_MESH_Y

    device_type = config.preferred_device_type()
    if device_type != 'TPU':
      test_util.reset_logical_devices(device_type, 2)
    accelerator_util.initialize_accelerator_system(device_type)

  def testBasicCopyToMesh(self):
    # [FUNCTION DOC]
    # Test: copies a numpy integer vector of zeros onto the first mesh with a replicated
    # layout and checks it equals the source. Then copies a vector of ones to the second mesh,
    # copies that DTensor over to the first mesh layout, and asserts the values and layout match.
    target_layout = Layout.replicated(self.first_mesh, rank=1)
    numpy_value = np.zeros([3], dtype=np.int32)
    dtensor_copy_from_numpy = api.copy_to_mesh(numpy_value, target_layout)
    self.assertDTensorEqual(numpy_value, target_layout, dtensor_copy_from_numpy)

    numpy_value = np.ones([3], dtype=np.int32)
    src_mesh = api.copy_to_mesh(
        numpy_value, Layout.replicated(self.second_mesh, rank=1)
    )
    dtensor_from_another_mesh = api.copy_to_mesh(src_mesh, target_layout)
    self.assertDTensorEqual(
        numpy_value, target_layout, dtensor_from_another_mesh
    )

  @parameterized.named_parameters(
      dict(testcase_name='Graph', is_eager=False),
      dict(testcase_name='Eager', is_eager=True),
  )
  def testCopyToMeshOneToOneSharded(self, is_eager):
    # [FUNCTION DOC]
    # Parameterized test (graph vs eager execution). Skipped on CPU-only setups. Copies a
    # replicated vector to the first mesh, then in a function (traced unless eager) relayouts it
    # to a layout sharded along x on the first mesh and copies that to a matching sharded
    # layout on the second mesh. Asserts both the source and target DTensors have the
    # expected values and layouts.
    if not test_util.is_tpu_present():
      self.skipForDeviceType(
          ['CPU'], 'Need at least one device mesh for this test.'
      )

    replicated_layout = Layout.replicated(self.first_mesh, rank=1)
    first_layout = Layout([_MESH_DIM_X], self.first_mesh)
    second_layout = Layout([_MESH_DIM_X], self.second_mesh)

    numpy_value = np.zeros([8], dtype=np.int32)
    dt_value = api.copy_to_mesh(numpy_value, replicated_layout)
    self.assertDTensorEqual(numpy_value, replicated_layout, dt_value)

    def fn(val):
      # [FUNCTION DOC]
      # Nested function under test: relayouts the input to the first-mesh sharded layout, copies
      # that result to the second-mesh sharded layout, and returns both DTensors as a pair.
      dt_source = api.relayout(val, first_layout)
      dt_target = api.copy_to_mesh(dt_source, second_layout)
      return dt_source, dt_target

    if not is_eager:
      fn = polymorphic_function.function(fn)

    dt_source, dt_target = fn(dt_value)
    self.assertDTensorEqual(numpy_value, first_layout, dt_source)
    self.assertDTensorEqual(numpy_value, second_layout, dt_target)

  @parameterized.named_parameters(
      dict(testcase_name='Graph', is_eager=False),
      dict(testcase_name='Eager', is_eager=True),
  )
  def testCopyToMeshToShardedLayout(self, is_eager):
    # [FUNCTION DOC]
    # Parameterized test (graph vs eager). Under a default-mesh scope of the first mesh, copies
    # a plain zero tensor into a layout sharded along x and asserts the resulting DTensor has
    # the expected values and layout.
    target_layout = Layout([_MESH_DIM_X], self.first_mesh)
    a_np = array_ops.zeros([8], dtype=dtypes.int32)

    def fn(val):
      # [FUNCTION DOC]
      # Nested function under test: copies the input to the sharded target layout and returns it.
      return api.copy_to_mesh(val, target_layout)

    if not is_eager:
      fn = polymorphic_function.function(fn)

    with api.default_mesh(self.first_mesh):
      dt_value = fn(a_np)

    self.assertDTensorEqual(a_np, target_layout, dt_value)

  def testNestedDefaultMesh(self):
    # [FUNCTION DOC]
    # Test: calls a traced function that adds a constant inside nested default-mesh scopes
    # (first, second, first) and asserts the result's mesh is the innermost scope's mesh each
    # time, and that it reverts to the outer mesh after each scope exits.

    @polymorphic_function.function
    def func(a):
      # [FUNCTION DOC]
      # Traced function under test: returns the input plus three.
      return a + 3.0

    with api.default_mesh(self.first_mesh):
      with api.default_mesh(self.second_mesh):
        with api.default_mesh(self.first_mesh):
          result = func(array_ops.ones(shape=()))
          self.assertEqual(api.fetch_layout(result).mesh, self.first_mesh)
        result = func(array_ops.ones(shape=()))
        self.assertEqual(api.fetch_layout(result).mesh, self.second_mesh)
      result = func(array_ops.ones(shape=()))
      self.assertEqual(api.fetch_layout(result).mesh, self.first_mesh)

  def testImplicitCopyToCPUMeshForStrings(self):
    # [FUNCTION DOC]
    # Test (skipped on CPU-only): a traced function receives a plain string tensor plus two
    # DTensors, one on a single-device host CPU mesh and one on the second (accelerator) mesh.
    # Asserts the string tensor is implicitly placed on the CPU mesh (replicated layout) rather
    # than the accelerator mesh.
    self.skipForDeviceType(
        ['CPU'],
        'Skipping test as only CPU mesh is available for multi-meshtest.',
    )

    host_device_id = test_util.create_device_ids_array((1,))
    host_cpu_mesh = Mesh(
        [_MESH_DIM_X],
        host_device_id,
        np.ravel(host_device_id).tolist(),
        test_util.create_device_list((1,), 'CPU'),
    )
    replicated_layout_on_cpu = Layout.replicated(host_cpu_mesh, rank=0)
    replicated_layout_on_tpu = Layout.replicated(self.second_mesh, rank=0)

    string_tensor = constant_op.constant('hello')

    @polymorphic_function.function
    def f(tensor, dtensor_a, dtensor_b):
      # [FUNCTION DOC]
      # Traced identity function: returns its three inputs unchanged (only so none are unused).
      # Return an identity op for all three inputs so that linter does not
      # complain about unused variables.
      return tensor, dtensor_a, dtensor_b

    cpu_dtensor = api.copy_to_mesh(
        constant_op.constant(1), replicated_layout_on_cpu
    )
    tpu_dtensor = api.copy_to_mesh(
        constant_op.constant(1), replicated_layout_on_tpu
    )

    string_dtensor, _, _ = f(string_tensor, cpu_dtensor, tpu_dtensor)

    # Regular string tensor should be implicitly copied onto the CPU mesh,
    # not the TPU mesh.
    self.assertEqual(api.fetch_layout(string_dtensor), replicated_layout_on_cpu)

  def testMultiMeshBroadcast(self):
    # [FUNCTION DOC]
    # Test: places a zero vector on the first mesh and a ones vector on the second, adds the
    # scalar one to each, and asserts each result stays on its own tensor's mesh (the constant
    # is broadcast to that mesh) with values one and two respectively.
    first_mesh_a = api.copy_to_mesh(
        np.zeros([3], dtype=np.int32),
        Layout.replicated(self.first_mesh, rank=1),
    )
    second_mesh_a = api.copy_to_mesh(
        np.ones([3], dtype=np.int32),
        Layout.replicated(self.second_mesh, rank=1),
    )
    self.assertDTensorEqual(
        np.asarray([1, 1, 1], dtype=np.int32),
        Layout.replicated(self.first_mesh, rank=1), first_mesh_a + 1)
    # Run an add with small constant - the constant should be broadcasted
    # onto the second mesh rather than the first.
    self.assertDTensorEqual(
        np.asarray([2, 2, 2], dtype=np.int32),
        Layout.replicated(self.second_mesh, rank=1), second_mesh_a + 1)

  def testMultiMeshAdd(self):
    # [FUNCTION DOC]
    # Test: adds two scalar DTensors on the first mesh and two vector DTensors on the second
    # mesh, independently, and asserts each sum has the expected value and stays on its mesh.
    a = constant_op.constant(1, dtype=dtypes.int32)
    b = constant_op.constant(2, dtype=dtypes.int32)
    with ops.device_v2(api.device_name()):
      first_mesh_a = api.copy_to_mesh(
          a, Layout.replicated(self.first_mesh, rank=0)
      )
      first_mesh_b = api.copy_to_mesh(
          b, Layout.replicated(self.first_mesh, rank=0)
      )
    # Copy-to-mesh doesn't work with multi-mesh as we always broadcast to
    # default mesh.
    # TODO(hthu): Use copy-to-mesh after the generic Relayout CL is in.
    second_mesh_a = api.copy_to_mesh(
        np.ones([3], dtype=np.int32),
        Layout.replicated(self.second_mesh, rank=1),
    )
    second_mesh_b = api.copy_to_mesh(
        np.zeros([3], dtype=np.int32),
        Layout.replicated(self.second_mesh, rank=1),
    )

    first_mesh_result = first_mesh_a + first_mesh_b
    second_mesh_result = second_mesh_a + second_mesh_b
    self.assertDTensorEqual(
        np.asarray(3, dtype=np.int32),
        Layout.replicated(self.first_mesh, rank=0), first_mesh_result)
    self.assertDTensorEqual(
        np.ones([3], dtype=np.int32),
        Layout.replicated(self.second_mesh, rank=1), second_mesh_result)

  def testMultiMeshFunc(self):
    # [FUNCTION DOC]
    # Test: adding a DTensor on the first mesh to one on the second mesh must fail mesh
    # propagation; asserts an unknown error is raised.
    a = constant_op.constant([1, 2, 3, 4], dtype=dtypes.int32)
    with ops.device_v2(api.device_name()):
      first_mesh_a = api.copy_to_mesh(
          a, Layout.replicated(self.first_mesh, rank=1)
      )
    second_mesh_a = api.copy_to_mesh(
        np.ones([4], dtype=np.int32),
        Layout.replicated(self.second_mesh, rank=1),
    )
    with self.assertRaises(errors_impl.UnknownError):
      # fails mesh propagation as it requires all inputs to be on the same
      # mesh.
      # pylint: disable=pointless-statement
      first_mesh_a + second_mesh_a
      # pylint: enable=pointless-statement

  def testMultiMeshInSideFunctionLayoutV2(self):
    # [FUNCTION DOC]
    # Test (skipped on CPU-only): a traced accelerator function casts and sums an input,
    # copies the scalar to a host CPU mesh inside the function and calls a CPU function that
    # takes the square root. Asserts the output equals the eagerly computed value with the
    # replicated host-CPU layout.
    self.skipForDeviceType(
        ['CPU'],
        'Skipping test as only CPU mesh is available for multi-meshtest.',
    )

    replicated_layout_on_tpu = Layout.replicated(self.second_mesh, rank=1)
    host_device_id = test_util.create_device_ids_array((1,))
    host_cpu_mesh = Mesh(
        [_MESH_DIM_X],
        host_device_id,
        np.ravel(host_device_id).tolist(),
        test_util.create_device_list((1,), 'CPU'),
    )
    replicated_layout_on_cpu = Layout.replicated(host_cpu_mesh, rank=0)

    a = constant_op.constant([1, 2, 3, 4], dtype=dtypes.int32)

    def func(t):
      # [FUNCTION DOC]
      # Reference (golden) computation: casts to float, sums all elements, returns the square root.
      t = math_ops.cast(t, dtypes.float32)
      t = math_ops.reduce_sum(t)
      return math_ops.sqrt(t)

    golden_result = func(a)

    a = api.copy_to_mesh(a, replicated_layout_on_tpu)

    @polymorphic_function.function
    def cpu_func(t):
      # [FUNCTION DOC]
      # Traced function meant to run on the CPU mesh: returns the square root of its input.
      return math_ops.sqrt(t)

    @polymorphic_function.function
    def tpu_func(t):
      # [FUNCTION DOC]
      # Traced function on the accelerator mesh: casts to float, sums, copies the scalar to the
      # host CPU mesh and returns the result of the CPU function applied to it.
      t = math_ops.cast(t, dtypes.float32)
      t = math_ops.reduce_sum(t)
      cpu_tensor = api.copy_to_mesh(t, replicated_layout_on_cpu)
      return cpu_func(cpu_tensor)

    with ops.device_v2(api.device_name()):
      output = tpu_func(a)
      self.assertDTensorEqual(golden_result, replicated_layout_on_cpu, output)

  def testMultiMeshCancellation(self):
    # [FUNCTION DOC]
    # Test (skipped on CPU-only): a CPU-mesh function divides integers by themselves with a
    # zero element (division by zero error) and an accelerator-mesh function copies that result
    # over and reduces it. Runs inside an async scope and asserts an exception mentioning
    # integer division by zero is raised at scope end, i.e. the accelerator work is cancelled
    # on the CPU failure.
    self.skipForDeviceType(
        ['CPU'],
        'Skipping test as only CPU mesh is available for multi-meshtest.',
    )

    host_device_id = test_util.create_device_ids_array((1,))
    host_cpu_mesh = Mesh(
        [_MESH_DIM_X],
        host_device_id,
        np.ravel(host_device_id).tolist(),
        test_util.create_device_list((1,), 'CPU'),
    )
    replicated_layout_on_cpu = Layout([UNSHARDED], host_cpu_mesh)
    replicated_layout_on_tpu = Layout([UNSHARDED], self.second_mesh)

    @polymorphic_function.function
    def cpu_func(x):
      # [FUNCTION DOC]
      # Traced CPU function: integer-divides the input by itself (errors on zero entries) and
      # casts the result to float.
      # Integer division by 0, which returns a bad status.
      x = math_ops.cast(gen_math_ops.div(x=x, y=x), dtypes.float32)
      return math_ops.cast(x, dtypes.float32)

    @polymorphic_function.function
    def tpu_func(cpu_tensor):
      # [FUNCTION DOC]
      # Traced accelerator function: runs the CPU function, copies its result to the accelerator
      # layout and returns the sum of all elements (which should be cancelled on CPU failure).
      cpu_result = cpu_func(cpu_tensor)
      tpu_tensor = api.copy_to_mesh(cpu_result, replicated_layout_on_tpu)
      # A reduction on the TPU mesh which must be cancelled in response to the
      # CPU mesh's failure.
      return math_ops.reduce_sum(tpu_tensor)

    cpu_tensor = api.copy_to_mesh(
        constant_op.constant([0, 1]), replicated_layout_on_cpu
    )

    with self.assertRaisesRegex(Exception, 'Integer division by zero'):
      # Ensure any errors are raised at end of scope.
      with context.async_scope():
        with ops.device_v2(api.device_name()):
          tpu_func(cpu_tensor)

  def testMultiMeshCPUToTPUTransfer(self):
    # [FUNCTION DOC]
    # Test (skipped on CPU-only): a CPU-mesh function casts a replicated 2-device CPU tensor
    # to float, copies it to the accelerator mesh and calls an accelerator function that
    # relayouts it to a layout sharded along x. Asserts the output layout is that sharded layout.
    self.skipForDeviceType(
        ['CPU'],
        'Skipping test as only CPU mesh is available for multi-meshtest.',
    )

    multiple_host_device_id = test_util.create_device_ids_array((2,))
    host_multi_cpu_mesh = Mesh(
        [_MESH_DIM_X],
        multiple_host_device_id,
        np.ravel(multiple_host_device_id).tolist(),
        test_util.create_device_list((2,), 'CPU'),
    )

    replicated_layout_on_cpu = Layout.replicated(host_multi_cpu_mesh, rank=1)
    sharded_layout_on_tpu_r1 = Layout([_MESH_DIM_X], self.second_mesh)
    replicated_layout_on_tpu_r1 = Layout.replicated(self.second_mesh, rank=1)

    a = constant_op.constant([1, 2, 3, 4], dtype=dtypes.int32)
    a = api.copy_to_mesh(a, replicated_layout_on_cpu)

    @polymorphic_function.function
    def tpu_func(t):
      # [FUNCTION DOC]
      # Traced accelerator function: relayouts the input to the x-sharded accelerator layout.
      return api.relayout(t, sharded_layout_on_tpu_r1)

    @polymorphic_function.function
    def cpu_func(t):
      # [FUNCTION DOC]
      # Traced CPU function: casts to float, copies to the replicated accelerator layout and
      # returns the accelerator function's result.
      t = math_ops.cast(t, dtypes.float32)
      tpu_tensor = api.copy_to_mesh(t, replicated_layout_on_tpu_r1)
      return tpu_func(tpu_tensor)

    with ops.device_v2(api.device_name()):
      output = cpu_func(a)

    api.check_layout(output, sharded_layout_on_tpu_r1)

  def testMultiMeshUnsupportedTypes(self):
    # [FUNCTION DOC]
    # Test (skipped on CPU-only): string tensors cannot be transferred between meshes. A CPU
    # function copies a string DTensor to the accelerator mesh; asserts an unknown error is
    # raised whose message mentions an unsupported output type.
    self.skipForDeviceType(
        ['CPU'],
        'Skipping test as only CPU mesh is available for multi-meshtest.',
    )

    host_device_id = test_util.create_device_ids_array((1,))
    host_cpu_mesh = Mesh(
        [_MESH_DIM_X],
        host_device_id,
        np.ravel(host_device_id).tolist(),
        test_util.create_device_list((1,), 'CPU'),
    )

    replicated_layout_on_cpu = Layout.replicated(host_cpu_mesh, rank=1)
    replicated_layout_on_tpu_r1 = Layout.replicated(self.second_mesh, rank=1)

    s = constant_op.constant(['a', 'b', 'c'], dtype=dtypes.string)
    s = api.copy_to_mesh(s, replicated_layout_on_cpu)

    @polymorphic_function.function
    def tpu_func(t):
      # [FUNCTION DOC]
      # Traced accelerator function: returns an identity of the input.
      return array_ops.identity(t)

    @polymorphic_function.function
    def cpu_func(t):
      # [FUNCTION DOC]
      # Traced CPU function: identity, then copies the string tensor to the accelerator mesh
      # and returns the accelerator function's result.
      t = array_ops.identity(t)
      tpu_tensor = api.copy_to_mesh(t, replicated_layout_on_tpu_r1)
      return tpu_func(tpu_tensor)

    with self.assertRaises(errors_impl.UnknownError) as ex:
      with ops.device_v2(api.device_name()):
        _ = str(cpu_func(s))

    self.assertIn('unsupported output type', ex.exception.message)

  def testMultiMeshCPUToCPUTransfer(self):
    # [FUNCTION DOC]
    # Test: transfers data between two single-device CPU meshes (CPU:0 and CPU:1). A sender
    # function casts the input and copies it to the receiver mesh, where a receiver function
    # sums and takes the square root. Asserts the result matches the eager golden value with a
    # replicated scalar layout on the receiver mesh.
    send_device_id = test_util.create_device_ids_array((1,))
    send_cpu_mesh = Mesh(
        [_MESH_DIM_X],
        send_device_id,
        np.ravel(send_device_id).tolist(),
        test_util.create_device_list((1,), 'CPU'),
    )
    recv_cpu_mesh = Mesh.from_string(
        '|x=1|0|0|/job:localhost/replica:0/task:0/device:CPU:1'
    )

    replicated_layout_on_cpu_send = Layout.replicated(send_cpu_mesh, rank=1)
    replicated_layout_on_cpu_recv = Layout.replicated(recv_cpu_mesh, rank=1)
    replicated_layout_on_cpu_r0 = Layout.replicated(recv_cpu_mesh, rank=0)

    def func(t):
      # [FUNCTION DOC]
      # Reference (golden) computation: casts to float, sums all elements, returns the square root.
      t = math_ops.cast(t, dtypes.float32)
      t = math_ops.reduce_sum(t)
      return math_ops.sqrt(t)

    @polymorphic_function.function
    def cpu_recv_func(t):
      # [FUNCTION DOC]
      # Traced function on the receiving CPU mesh: sums all elements and returns the square root.
      t = math_ops.reduce_sum(t)
      t = math_ops.sqrt(t)
      return t

    @polymorphic_function.function
    def cpu_send_func(t):
      # [FUNCTION DOC]
      # Traced function on the sending CPU mesh: casts to float, copies to the receiving mesh and
      # returns the receiver function's result.
      t = math_ops.cast(t, dtypes.float32)
      cpu_recv_tensor = api.copy_to_mesh(t, replicated_layout_on_cpu_recv)
      t = cpu_recv_func(cpu_recv_tensor)
      return t

    a = constant_op.constant([1, 2, 3, 4], dtype=dtypes.int32)
    golden_result = func(a)
    a = api.copy_to_mesh(a, replicated_layout_on_cpu_send)

    with ops.device_v2(api.device_name()):
      output = cpu_send_func(a)
      self.assertDTensorEqual(golden_result, replicated_layout_on_cpu_r0,
                              output)

  def testMultiMeshCPUTest(self):
    # [FUNCTION DOC]
    # Test: two CPU meshes (dimension names x and y) each hold a replicated vector. One traced
    # function takes one input from each mesh, squares the first and square-roots the second.
    # Asserts each output keeps the layout (mesh) of the corresponding input.
    device_ids = test_util.create_device_ids_array((2,))
    cpu_mesh_a = Mesh(
        ['x'],
        device_ids,
        np.ravel(device_ids).tolist(),
        test_util.create_device_list((2,), 'CPU'),
    )
    cpu_mesh_b = Mesh(
        ['y'],
        device_ids,
        np.ravel(device_ids).tolist(),
        test_util.create_device_list((2,), 'CPU'),
    )
    replicated_layout_on_a = Layout.replicated(cpu_mesh_a, rank=1)
    replicated_layout_on_b = Layout.replicated(cpu_mesh_b, rank=1)

    x = constant_op.constant([1, 2, 3, 4], dtype=dtypes.int32)
    y = constant_op.constant([1, 2, 3, 4], dtype=dtypes.int32)

    a = api.copy_to_mesh(x, replicated_layout_on_a)
    b = api.copy_to_mesh(y, replicated_layout_on_b)

    @polymorphic_function.function
    def func2(t1, t2):
      # [FUNCTION DOC]
      # Traced function: returns the float-cast first input squared and the square root of the
      # float-cast second input (each input lives on a different mesh).
      t1 = math_ops.cast(t1, dtypes.float32)
      t1 = t1 * t1

      t2 = math_ops.cast(t2, dtypes.float32)
      t2 = math_ops.sqrt(t2)
      return t1, t2

    with ops.device_v2(api.device_name()):
      output1, output2 = func2(a, b)

    api.check_layout(output1, replicated_layout_on_a)
    api.check_layout(output2, replicated_layout_on_b)

  def testFunctionWithMultiMeshInputOutputs(self):
    # [FUNCTION DOC]
    # Test (skipped on CPU-only): one traced function takes inputs from the accelerator and CPU
    # meshes and returns outputs on both (a CPU result via a nested CPU function and an
    # accelerator negated-sum result). Asserts both outputs equal the golden values and keep
    # their respective mesh layouts.
    self.skipForDeviceType(
        ['CPU'],
        'Skipping test as only CPU mesh is available for multi-meshtest.',
    )

    host_device_id = test_util.create_device_ids_array((1,))
    host_cpu_mesh = Mesh(
        [_MESH_DIM_X],
        host_device_id,
        np.ravel(host_device_id).tolist(),
        test_util.create_device_list((1,), 'CPU'),
    )
    replicated_layout_on_cpu = Layout.replicated(host_cpu_mesh, rank=1)
    replicated_layout_on_cpu_r0 = Layout.replicated(host_cpu_mesh, rank=0)
    replicated_layout_on_tpu_r0 = Layout.replicated(self.second_mesh, rank=0)
    replicated_layout_on_tpu = Layout.replicated(self.second_mesh, rank=1)

    a = constant_op.constant([1, 2, 3, 4], dtype=dtypes.int32)
    b = constant_op.constant([1, 2, 3, 4], dtype=dtypes.int32)

    def golden_func(t1, t2):
      # [FUNCTION DOC]
      # Reference computation on ordinary tensors: returns the square root of (sum of second
      # input plus squared first input) and the negated sum of the second input.
      t1 = math_ops.cast(t1, dtypes.float32)
      t1 = t1 * t1
      t2 = math_ops.cast(t2, dtypes.float32)
      t2 = math_ops.reduce_sum(t2)
      out1 = gen_math_ops.neg(t2)

      t2 = t2 + t1
      out0 = math_ops.sqrt(t2)
      return out0, out1

    golden_result0, golden_result1 = golden_func(a, b)

    cpu_dtensor = api.copy_to_mesh(a, replicated_layout_on_cpu)
    tpu_dtensor = api.copy_to_mesh(b, replicated_layout_on_tpu)

    @polymorphic_function.function
    def cpu_func(t1, t2):
      # [FUNCTION DOC]
      # Traced CPU function: adds the two inputs and returns the square root.
      t2 = t2 + t1
      return math_ops.sqrt(t2)

    @polymorphic_function.function
    def func(tpu_input, cpu_input):
      # [FUNCTION DOC]
      # Traced function with inputs on two meshes: squares the CPU input, sums and negates the
      # accelerator input, copies the sum to the CPU mesh, runs the CPU function, and returns
      # the pair (CPU output, accelerator output).
      cpu_input = math_ops.cast(cpu_input, dtypes.float32)
      cpu_input = cpu_input * cpu_input
      tpu_input = math_ops.cast(tpu_input, dtypes.float32)
      tpu_input = math_ops.reduce_sum(tpu_input)
      tpu_output = gen_math_ops.neg(tpu_input)

      cpu_tensor = api.copy_to_mesh(tpu_input, replicated_layout_on_cpu_r0)
      cpu_output = cpu_func(cpu_tensor, cpu_input)
      return cpu_output, tpu_output

    with ops.device_v2(api.device_name()):
      output0, output1 = func(tpu_dtensor, cpu_dtensor)

    self.assertDTensorEqual(golden_result0, replicated_layout_on_cpu, output0)
    self.assertDTensorEqual(golden_result1, replicated_layout_on_tpu_r0,
                            output1)

  def testMultiMeshWithResourceOps(self):
    # [FUNCTION DOC]
    # Test (skipped on CPU-only): creates a distributed variable on the accelerator mesh from a
    # 64-bit integer vector and passes it to a traced accelerator function that sums, takes the
    # square root on the host CPU mesh via a nested CPU function. Asserts the output equals the
    # golden value with the replicated CPU layout.
    self.skipForDeviceType(
        ['CPU'],
        'Skipping test as only CPU mesh is available for multi-meshtest.',
    )

    host_device_id = test_util.create_device_ids_array((1,))
    host_cpu_mesh = Mesh(
        [_MESH_DIM_X],
        host_device_id,
        np.ravel(host_device_id).tolist(),
        test_util.create_device_list((1,), 'CPU'),
    )

    replicated_layout_on_cpu = Layout.replicated(host_cpu_mesh, rank=0)
    replicated_layout_on_tpu = Layout.replicated(self.second_mesh, rank=1)
    a = constant_op.constant(
        [1, 2, 3, 4], dtype=dtypes.int64
    )  # NOTE(b/274627284): Variable of int32 type on GPU doesn't work.

    def func(t):
      # [FUNCTION DOC]
      # Reference (golden) computation: casts to float, sums all elements, returns the square root.
      t = math_ops.cast(t, dtypes.float32)
      t = math_ops.reduce_sum(t)
      return math_ops.sqrt(t)

    golden_result = func(a)

    @polymorphic_function.function
    def cpu_func(t):
      # [FUNCTION DOC]
      # Traced CPU function: returns the square root of its input.
      return math_ops.sqrt(t)

    @polymorphic_function.function
    def tpu_func(t):
      # [FUNCTION DOC]
      # Traced accelerator function: casts to float, sums, copies the scalar to the CPU mesh and
      # returns the CPU function's result.
      t = math_ops.cast(t, dtypes.float32)
      t = math_ops.reduce_sum(t)
      cpu_tensor = api.copy_to_mesh(t, replicated_layout_on_cpu)
      return cpu_func(cpu_tensor)

    with ops.device_v2(api.device_name()):
      v = api.copy_to_mesh(a, replicated_layout_on_tpu)
      w = d_variable.DVariable(v)
      output = tpu_func(w)

    self.assertDTensorEqual(golden_result, replicated_layout_on_cpu, output)

  @parameterized.named_parameters(
      ('_host_to_dev_sharded_i32', True, True, dtypes.int32),
      ('_dev_to_host_sharded_i32', False, True, dtypes.int32),
      ('_host_to_dev_replicated_i32', True, False, dtypes.int32),
      ('_dev_to_host_replicated_i32', False, False, dtypes.int32),
      ('_host_to_dev_sharded_bf16', True, True, dtypes.bfloat16),
      ('_dev_to_host_sharded_bf16', False, True, dtypes.bfloat16),
      ('_host_to_dev_replicated_bf16', True, False, dtypes.bfloat16),
      ('_dev_to_host_replicated_bf16', False, False, dtypes.bfloat16),
      ('_host_to_dev_sharded_f32', True, True, dtypes.float32),
      ('_dev_to_host_sharded_f32', False, True, dtypes.float32),
      ('_host_to_dev_replicated_f32', True, False, dtypes.float32),
      ('_dev_to_host_replicated_f32', False, False, dtypes.float32),
      ('_host_to_dev_sharded_f64', True, True, dtypes.float64),
      ('_dev_to_host_sharded_f64', False, True, dtypes.float64),
      ('_host_to_dev_replicated_f64', True, False, dtypes.float64),
      ('_dev_to_host_replicated_f64', False, False, dtypes.float64),
  )
  def testMultiMeshHostDeviceTransfer(self, host_to_dev, sharded, dtype):
    # [FUNCTION DOC]
    # Parameterized test over direction (host to device or device to host), sharded or
    # replicated layouts, and data type (int32, bfloat16, float32, float64). Skipped on
    # CPU-only. Builds an 8x8 ones tensor, places it on the source mesh with the source layout,
    # copies it to the destination mesh layout inside a traced function, and asserts both
    # source and destination DTensors match the data and layouts. Note the first mesh is
    # treated as the device mesh and the second mesh as the host (CPU) mesh.
    self.skipForDeviceType(
        ['CPU'],
        'Skipping test as only CPU mesh is available for multi-meshtest.',
    )

    def run_copy_to_mesh(data, src_layout, dst_layout):
      # [FUNCTION DOC]
      # Local helper: places the data on the source layout (replicating first, then relayouting
      # when the layout is sharded), copies it to the destination layout within a traced
      # function, and returns the pair (source DTensor, destination DTensor).

      @polymorphic_function.function
      def func(x):
        # [FUNCTION DOC]
        # Traced function: copies the input to the destination layout and returns it.
        return api.copy_to_mesh(x, dst_layout)

      if src_layout.is_fully_replicated():
        src_data = api.copy_to_mesh(data, src_layout)
      else:
        src_data = api.copy_to_mesh(
            data, Layout.replicated(src_layout.mesh, rank=len(data.shape))
        )
        src_data = api.relayout(src_data, src_layout)
      dst_data = func(src_data)
      return (src_data, dst_data)

    dev_mesh = self.first_mesh
    cpu_mesh = self.second_mesh

    if host_to_dev:
      src_mesh, dst_mesh = cpu_mesh, dev_mesh
    else:
      src_mesh, dst_mesh = dev_mesh, cpu_mesh

    if sharded:
      src_layout = Layout.batch_sharded(src_mesh, src_mesh.dim_names[0], rank=2)
      dst_layout = Layout.batch_sharded(dst_mesh, dst_mesh.dim_names[0], rank=2)
    else:
      src_layout = Layout.replicated(src_mesh, rank=2)
      dst_layout = Layout.replicated(dst_mesh, rank=2)

    data = array_ops.ones([8, 8], dtype=dtype)
    src, dst = run_copy_to_mesh(data, src_layout, dst_layout)
    self.assertDTensorEqual(data, src_layout, src)
    self.assertDTensorEqual(data, dst_layout, dst)

  @parameterized.named_parameters(('_host_to_tpu', True),
                                  ('_tpu_to_host', False))
  def testMultiMeshWithHostMesh(self, host_to_tpu):
    # [FUNCTION DOC]
    # Parameterized test (host to accelerator, or accelerator to host). Skipped on CPU-only.
    # Places an integer vector on the source layout (x-sharded accelerator layout or its host
    # mesh equivalent), copies it to the target layout in a traced function followed by an
    # identity, and asserts the output equals the original with the target layout.
    self.skipForDeviceType(
        ['CPU'],
        'Skipping test as only CPU mesh is available for multi-meshtest.',
    )

    sharded_layout_on_tpu = Layout([_MESH_DIM_X], self.second_mesh)
    host_layout = Layout(sharded_layout_on_tpu.sharding_specs,
                         sharded_layout_on_tpu.mesh.host_mesh())

    if host_to_tpu:
      source_layout = host_layout
      target_layout = sharded_layout_on_tpu
    else:
      source_layout = sharded_layout_on_tpu
      target_layout = host_layout

    numpy_a = constant_op.constant([1, 2, 3, 4], dtype=dtypes.int32)

    # TODO(b/193443769): switch to a single copy_to_mesh when this is supported.
    replicated_layout = Layout.replicated(source_layout.mesh,
                                          source_layout.rank)
    a = api.copy_to_mesh(numpy_a, replicated_layout)
    a = api.relayout(a, source_layout)

    @polymorphic_function.function
    def func(t):
      # [FUNCTION DOC]
      # Traced function: copies the input to the target layout and returns an identity of it.
      target_tensor = api.copy_to_mesh(t, target_layout)
      return array_ops.identity(target_tensor)

    with ops.device_v2(api.device_name()):
      dtensor_output = func(a)

    self.assertDTensorEqual(numpy_a, target_layout, dtensor_output)

  def testMultiMeshBackward(self):
    # [FUNCTION DOC]
    # Test (skipped on CPU-only): gradients through a host-to-accelerator copy. A traced function
    # scales a host-mesh input by four, copies it to the accelerator mesh, takes the square root
    # and returns the gradient with respect to the input. A second traced function takes the
    # gradient of that gradient. Asserts the first and second derivatives match the analytic
    # values, with the host layout.
    self.skipForDeviceType(
        ['CPU'],
        'Skipping test as only CPU mesh is available for multi-meshtest.',
    )

    replicated_layout_on_tpu = Layout.replicated(self.second_mesh, rank=1)
    host_layout = Layout.replicated(self.second_mesh.host_mesh(), rank=1)

    source_layout = host_layout
    target_layout = replicated_layout_on_tpu

    @polymorphic_function.function
    def func(x):
      # [FUNCTION DOC]
      # Traced function: scales the input by four, copies to the target layout, takes the
      # square root and returns the gradient of that with respect to the watched input.
      with backprop.GradientTape() as tape:
        tape.watch(x)
        x = x * 4.0
        t = api.copy_to_mesh(x, target_layout)
        sqrt = math_ops.sqrt(t)
        sqrt_grad = tape.gradient(sqrt, x)
        return sqrt_grad

    @polymorphic_function.function
    def second(x):
      # [FUNCTION DOC]
      # Traced function: returns the gradient of the first-order gradient function's output with
      # respect to the watched input (second derivative).
      with backprop.GradientTape() as tape:
        tape.watch(x)
        sqrt_grad = func(x)
        sqrt_grad_grad = tape.gradient(sqrt_grad, x)
        return sqrt_grad_grad

    numpy_a = constant_op.constant([1, 4, 16, 64], dtype=dtypes.float32)
    a = api.copy_to_mesh(numpy_a, source_layout)

    with ops.device_v2(api.device_name()):
      a_grad = func(a)

    self.assertDTensorEqual(0.5 * 0.5 * (1 / numpy_a)**0.5, host_layout, a_grad)
    with ops.device_v2(api.device_name()):
      a_grad_grad = second(a)
    self.assertDTensorEqual(-0.5 * 0.5 * 0.5 * (1 / numpy_a)**1.5, host_layout,
                            a_grad_grad)

  def testMultiMeshMultipleCopyToMesh(self):
    # [FUNCTION DOC]
    # Test (skipped on CPU-only): two host-mesh tensors are both copied to the x-sharded
    # accelerator layout inside one traced function and returned through identities. Asserts
    # both outputs equal their inputs with the target layout.
    self.skipForDeviceType(
        ['CPU'],
        'Skipping test as only CPU mesh is available for multi-meshtest.',
    )

    sharded_layout_on_tpu = Layout([_MESH_DIM_X], self.second_mesh)
    host_layout = Layout(
        sharded_layout_on_tpu.sharding_specs,
        sharded_layout_on_tpu.mesh.host_mesh(),
    )

    source_layout = host_layout
    target_layout = sharded_layout_on_tpu

    numpy_a = constant_op.constant([1, 2, 3, 4], dtype=dtypes.int32)
    numpy_b = constant_op.constant([2, 2, 3, 4], dtype=dtypes.int32)

    # TODO(b/193443769): switch to a single copy_to_mesh when this is supported.
    replicated_layout = Layout.replicated(
        source_layout.mesh, source_layout.rank
    )
    a = api.copy_to_mesh(numpy_a, replicated_layout)
    b = api.copy_to_mesh(numpy_b, replicated_layout)
    a = api.relayout(a, source_layout)
    b = api.relayout(b, source_layout)

    @polymorphic_function.function
    def func(a, b):
      # [FUNCTION DOC]
      # Traced function: copies both inputs to the target layout and returns identities of both.
      a = api.copy_to_mesh(a, target_layout)
      b = api.copy_to_mesh(b, target_layout)
      return array_ops.identity(a), array_ops.identity(b)

    with ops.device_v2(api.device_name()):
      dtensor_a, dtensor_b = func(a, b)

    self.assertDTensorEqual(numpy_a, target_layout, dtensor_a)
    self.assertDTensorEqual(numpy_b, target_layout, dtensor_b)

  def testDVariableDefaultMesh(self):
    # [FUNCTION DOC]
    # Test: copies a scalar to a different one-device mesh first, then to the first mesh, and
    # builds a distributed variable from the latter. Asserts nothing explicitly; it checks that
    # variable creation succeeds without being confused by the earlier use of another mesh.
    other_layout = Layout.replicated(_OTHER_CPU_MESH, rank=0)
    first_layout = Layout.replicated(_ONE_D_CPU_MESH, rank=0)

    _ = api.copy_to_mesh(1.0, other_layout)
    init_value = api.copy_to_mesh(1.0, first_layout)
    _ = d_variable.DVariable(init_value)


if __name__ == '__main__':
  test.main()
