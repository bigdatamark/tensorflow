# [MODULE DOC]
# Tests for the DTensor mesh utility module (mesh creation helpers) and for accelerator system
# initialization and shutdown. The first test class covers creating plain and distributed meshes
# (dictionary or list of dimensions, explicit device lists or specs, XLA SPMD, single and multiple
# clients simulated with mocks) without initializing accelerators for real. The second class
# initializes the accelerator system in setUp and covers initialization errors, logical CPU
# device setup, shutdown, TPU distributed meshes and the mesh barrier. Many tests are skipped
# depending on the device type of the test environment.
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
"""Tests for mesh_util."""
import os
from unittest import mock

from absl.testing import parameterized

from tensorflow.dtensor.python import accelerator_util
from tensorflow.dtensor.python import config
from tensorflow.dtensor.python import mesh_util
from tensorflow.dtensor.python.tests import test_util
from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config
from tensorflow.python.framework import device as tf_device
from tensorflow.python.platform import test


class MeshUtilTest(test_util.DTensorBaseTest):
  # [CLASS DOC]
  # Tests for mesh creation functions that do not need a real accelerator initialization.
  # Accelerator initialization state and client configuration are replaced with mocks where needed.
  """Tests for mesh_util that do not require accelerator initialization."""

  def test_mesh_creation(self):
    # [FUNCTION DOC]
    # Test (CPU/GPU only): creating a default mesh yields one dimension covering all local logical
    # devices of its device type; asserts the number of local devices and mesh size match.
    self.skipForDeviceType(
        ['TPU'], reason='Test is intended for CPUs and GPUs.'
    )
    mesh = mesh_util.create_mesh()
    num_devices = len(test_util.list_local_logical_devices(mesh.device_type()))
    self.assertEqual(mesh.num_local_devices(), num_devices)
    self.assertEqual(mesh.size, num_devices)

  def test_mesh_dict_creation(self):
    # [FUNCTION DOC]
    # Test (CPU/GPU only): creating a CPU mesh from a mapping of dimension names to sizes (x gets
    # all local CPU devices, y gets one). Asserts the dimension names, local device count and size.
    self.skipForDeviceType(
        ['TPU'], reason='Test is intended for CPUs and GPUs.'
    )
    num_devices = len(test_util.list_local_logical_devices('CPU'))
    mesh = mesh_util.create_mesh({'x': num_devices, 'y': 1}, device_type='CPU')
    num_devices = len(test_util.list_local_logical_devices(mesh.device_type()))
    self.assertEqual(mesh.num_local_devices(), num_devices)
    self.assertEqual(mesh.dim_names, ['x', 'y'])
    self.assertEqual(mesh.size, num_devices)

  def test_tpu_mesh_creation(self):
    # [FUNCTION DOC]
    # Test (TPU only): a named default TPU mesh spans all local logical TPU devices; asserts the
    # local device count and mesh size equal the number of devices.
    self.skipForDeviceType(['CPU', 'GPU'], reason='Test is intended for TPUs.')
    mesh = mesh_util.create_mesh(mesh_name='1d_mesh', device_type='TPU')
    num_devices = len(test_util.list_local_logical_devices('TPU'))
    self.assertEqual(mesh.num_local_devices(), num_devices)
    self.assertEqual(mesh.size, num_devices)

  @parameterized.named_parameters(('use_xla_spmd', True),
                                  ('do_not_use_xla_spmd', False))
  def test_tpu_2d_mesh_creation(self, use_xla_spmd):
    # [FUNCTION DOC]
    # Parameterized test (TPU with exactly two cores only): creates a 2x1 TPU mesh with dimensions
    # x and y and the given XLA SPMD flag. Asserts two local devices, size two, the dimension
    # names, and that the mesh reports the requested XLA SPMD setting.
    self.skipForDeviceType(['CPU', 'GPU'], reason='Test is intended for TPUs.')
    self.skipForDeviceType(['TPU'],
                           reason='Test requires exactly 2 cores',
                           unless_device_count_equals_to=2)
    devices = test_util.list_local_logical_devices('TPU')
    self.assertLen(devices, 2)
    mesh = mesh_util.create_mesh([('x', 2), ('y', 1)],
                                 device_type='TPU',
                                 use_xla_spmd=use_xla_spmd)
    self.assertEqual(mesh.num_local_devices(), 2)
    self.assertEqual(mesh.size, 2)
    self.assertAllEqual(mesh.dim_names, ['x', 'y'])
    self.assertEqual(mesh.use_xla_spmd(), use_xla_spmd)

  def test_tpu_2d_mesh_creation_with_devices(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the skip reason says "at least 2 cores" but the condition only runs the test
    # when exactly two TPU cores exist.
    # Test (TPU only): creates a 2x1 mesh from an explicit list of two device name strings.
    # Asserts two local devices, size two and dimension names x and y.
    self.skipForDeviceType(['CPU', 'GPU'], reason='Test is intended for TPUs.')
    self.skipForDeviceType(['TPU'],
                           reason='Test requires at least 2 cores',
                           unless_device_count_equals_to=2)
    devices = test_util.list_local_logical_devices('TPU')
    self.assertLen(devices, 2)
    mesh = mesh_util.create_mesh([('x', 2), ('y', 1)],
                                 devices=['/device:tpu:0', '/device:tpu:1'])
    self.assertEqual(mesh.num_local_devices(), 2)
    self.assertEqual(mesh.size, 2)
    self.assertAllEqual(mesh.dim_names, ['x', 'y'])

  def test_tpu_2d_mesh_creation_with_device_specs(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the skip reason says "at least 2 cores" but the condition only runs the test
    # when exactly two TPU cores exist.
    # Test (TPU only): creates a 2x1 mesh from a list of two device specification objects.
    # Asserts two local devices, size two and dimension names x and y.
    self.skipForDeviceType(['CPU', 'GPU'], reason='Test is intended for TPUs.')
    self.skipForDeviceType(['TPU'],
                           reason='Test requires at least 2 cores',
                           unless_device_count_equals_to=2)
    devices = test_util.list_local_logical_devices('TPU')
    self.assertLen(devices, 2)
    mesh = mesh_util.create_mesh(
        [('x', 2), ('y', 1)],
        devices=[
            tf_device.DeviceSpec.from_string('/tpu:0'),
            tf_device.DeviceSpec.from_string('/tpu:1'),
        ],
    )
    self.assertEqual(mesh.num_local_devices(), 2)
    self.assertEqual(mesh.size, 2)
    self.assertAllEqual(mesh.dim_names, ['x', 'y'])

  def test_single_client_mesh_creation(self):
    # [FUNCTION DOC]
    # Test (CPU only): with the accelerator-initialized check mocked to true, creating a
    # distributed mesh from a list of one dimension spanning all local CPUs yields a mesh whose
    # local device count and size equal the number of CPU devices.
    self.skipForDeviceType(['GPU', 'TPU'], reason='Test is intended for CPUs')
    num_devices = len(test_util.list_local_logical_devices('CPU'))
    with mock.patch.object(accelerator_util,
                           'is_initialized') as is_initialized:
      is_initialized.return_value = True
      mesh = mesh_util.create_distributed_mesh(
          mesh_name='single_client_1d_mesh', mesh_dims=[('x', num_devices)])
      self.assertEqual(mesh.num_local_devices(), num_devices)
      self.assertEqual(mesh.size, num_devices)

  def test_single_client_mesh_dict_creation(self):
    # [FUNCTION DOC]
    # Test (CPU only): like the single-client list test, but the dimensions are given as a mapping
    # (x with all CPUs, y with one). Initialization is mocked as done. Asserts local device count,
    # dimension names and size.
    self.skipForDeviceType(['GPU', 'TPU'], reason='Test is intended for CPUs')
    num_devices = len(test_util.list_local_logical_devices('CPU'))
    with mock.patch.object(
        accelerator_util, 'is_initialized'
    ) as is_initialized:
      is_initialized.return_value = True
      mesh = mesh_util.create_distributed_mesh(
          mesh_name='single_client_1d_mesh',
          mesh_dims={'x': num_devices, 'y': 1},
      )
      self.assertEqual(mesh.num_local_devices(), num_devices)
      self.assertEqual(mesh.dim_names, ['x', 'y'])
      self.assertEqual(mesh.size, num_devices)

  def test_single_client_mesh_with_local_devices(self):
    # [FUNCTION DOC]
    # Test (CPU only): a distributed mesh of one dimension of size one is created with an explicit
    # local device list containing the first CPU (initialization mocked as done). Asserts one local
    # device and size one.
    self.skipForDeviceType(['GPU', 'TPU'], reason='Test is intended for CPUs')
    with mock.patch.object(accelerator_util,
                           'is_initialized') as is_initialized:
      is_initialized.return_value = True
      mesh = mesh_util.create_distributed_mesh(
          mesh_name='single_client_1d_mesh',
          mesh_dims=[('x', 1)],
          local_devices=['CPU:0'])
      self.assertEqual(mesh.num_local_devices(), 1)
      self.assertEqual(mesh.size, 1)

  def test_create_distributed_mesh_requires_initialize(self):
    # [FUNCTION DOC]
    # Test (CPU only): with the accelerator-initialized check mocked to false, creating a
    # distributed mesh raises a value error saying accelerators are uninitialized.
    self.skipForDeviceType(['GPU', 'TPU'], reason='Test is intended for CPUs')

    with mock.patch.object(accelerator_util,
                           'is_initialized') as is_initialized:
      is_initialized.return_value = False
      with self.assertRaisesRegex(ValueError, 'Accelerators are uninitialized'):
        _ = mesh_util.create_distributed_mesh(
            mesh_name='single_client_1d_mesh',
            mesh_dims=[('x', 1)],
            local_devices=['CPU:0'])

  def test_single_client_mesh_creation_wrong_shape(self):
    # [FUNCTION DOC]
    # Test (CPU only): requesting a mesh dimension twice as large as the number of local CPUs
    # raises a value error saying the local device count must equal the total mesh size.
    self.skipForDeviceType(['GPU', 'TPU'], reason='Test is intended for CPUs')
    num_devices = len(test_util.list_local_logical_devices('CPU'))
    with mock.patch.object(accelerator_util,
                           'is_initialized') as is_initialized:
      is_initialized.return_value = True
      with self.assertRaisesRegex(ValueError,
                                  'must be equal to total size of the mesh'):
        mesh_util.create_distributed_mesh(
            mesh_name='single_client_1d_mesh',
            mesh_dims=[('x', num_devices * 2)])

  def test_single_client_mesh_creation_using_fewer_devices(self):
    # [FUNCTION DOC]
    # Test (CPU only): with four logical CPUs configured, a mesh of size two is created twice using
    # an explicit list of two local CPU devices (initialization mocked as done). Each time asserts
    # two local devices and size two (repeat creation must also work).
    self.skipForDeviceType(['GPU', 'TPU'], reason='Test is intended for CPUs')
    test_util.reset_logical_devices('CPU', 4)
    with mock.patch.object(accelerator_util,
                           'is_initialized') as is_initialized:
      is_initialized.return_value = True
      mesh = mesh_util.create_distributed_mesh(
          mesh_name='single_client_1d_mesh',
          mesh_dims=[('x', 2)],
          local_devices=['CPU:0', 'CPU:1'])
      self.assertEqual(mesh.num_local_devices(), 2)
      self.assertEqual(mesh.size, 2)

      mesh = mesh_util.create_distributed_mesh(
          mesh_name='single_client_1d_mesh',
          mesh_dims=[('x', 2)],
          local_devices=['CPU:0', 'CPU:1'])
      self.assertEqual(mesh.num_local_devices(), 2)
      self.assertEqual(mesh.size, 2)

  def test_single_client_mesh_creation_with_xla_spmd_raises_error(self):
    # [FUNCTION DOC]
    # Test (non-TPU only): requesting XLA SPMD when creating a CPU distributed mesh raises a value
    # error saying XLA SPMD is not supported for that device type.
    self.skipForDeviceType(['TPU'],
                           reason='Test is intended for non TPU devices')
    test_util.reset_logical_devices('CPU', 4)
    with mock.patch.object(accelerator_util,
                           'is_initialized') as is_initialized:
      is_initialized.return_value = True
      with self.assertRaisesRegex(
          ValueError, 'XLA SPMD is not currently not supported for'):
        mesh_util.create_distributed_mesh(
            mesh_name='single_client_mesh',
            mesh_dims=[('x', 2)],
            local_devices=['CPU:0', 'CPU:1'],
            use_xla_spmd=True)

  @mock.patch.object(config, 'num_clients')
  @mock.patch.object(accelerator_util, 'is_initialized')
  def test_multi_client_mesh_creation(self, num_clients, is_initialized):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the two mock decorators inject mocks (applied bottom-up, so the first argument
    # is really the initialization mock), so the argument names are swapped; the body ignores
    # both injected arguments and re-patches the same objects with new local mocks.
    # Test (CPU only): simulates two clients. The client count is mocked as two and initialization
    # as done, the context is reset and four logical CPUs are configured. For client zero a mesh of
    # size four is created with local devices CPU 0 and 1; for client one with CPU 2 and 3. Each
    # mesh must report two local devices and total size four.
    self.skipForDeviceType(['GPU', 'TPU'], reason='Test is intended for CPUs')
    with mock.patch.object(accelerator_util,
                           'is_initialized') as is_initialized:
      with mock.patch.object(config, 'num_clients') as num_clients:
        num_clients.return_value = 2
        is_initialized.return_value = True
        test_util.reset_context()
        cpus = tf_config.list_physical_devices('CPU')
        tf_config.set_logical_device_configuration(
            cpus[0], [context.LogicalDeviceConfiguration()] * 4
        )
        with mock.patch.object(config, 'client_id', return_value=0):
          mesh_1 = mesh_util.create_distributed_mesh(
              mesh_name='multi_client_1d_mesh_1',
              mesh_dims=[('x', 4)],
              local_devices=['CPU:0', 'CPU:1'])
        self.assertEqual(mesh_1.num_local_devices(), 2)
        self.assertEqual(mesh_1.size, 4)
        with mock.patch.object(config, 'client_id', return_value=1):
          mesh_2 = mesh_util.create_distributed_mesh(
              mesh_name='multi_client_1d_mesh_2',
              mesh_dims=[('x', 4)],
              local_devices=['CPU:2', 'CPU:3'])
        self.assertEqual(mesh_2.num_local_devices(), 2)
        self.assertEqual(mesh_2.size, 4)


class InitializedMeshUtilTest(test_util.DTensorBaseTest):
  # [CLASS DOC]
  # Tests that need a really initialized accelerator system: initialization and shutdown
  # behavior, error cases, logical CPU device configuration, distributed TPU meshes and the
  # mesh barrier. setUp initializes the system for the preferred device type; tearDown resets
  # the TensorFlow context.
  """Tests for mesh_util that require accelerator initialization."""

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture setup: calls the base setUp, then initializes the accelerator system for the
    # preferred device type (side effect: global DTensor initialization state).
    super().setUp()
    device_type = config.preferred_device_type()
    accelerator_util.initialize_accelerator_system(device_type)

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture cleanup: calls the base tearDown, then resets the global TensorFlow eager context
    # (using a private function) so later tests start from a clean state.
    super().tearDown()
    context._reset_context()  # pylint: disable=protected-access

  def test_is_initialized(self):
    # [FUNCTION DOC]
    # Test: after setUp, the accelerator system reports that it is initialized.
    self.assertTrue(accelerator_util.is_initialized())

  def test_initialize_accelerator_system(self):
    # [FUNCTION DOC]
    # Test: after shutting down, initializing with an explicit CPU device type returns CPU; after
    # another shutdown, initializing with no argument returns the preferred device type.
    accelerator_util.shutdown_accelerator_system()
    device_type = accelerator_util.initialize_accelerator_system('CPU')
    self.assertEqual(device_type, 'CPU')

    # Default uses preferred_device_type.
    accelerator_util.shutdown_accelerator_system()
    device_type = accelerator_util.initialize_accelerator_system()
    self.assertEqual(device_type, config.preferred_device_type())

  @mock.patch.dict(os.environ, {'DTENSOR_GPU_USE_NCCL_COMMUNICATION': '1'})
  def test_initialize_error_vgpu_with_nccl(self):
    # [FUNCTION DOC]
    # Test (GPU only, environment variable for NCCL communication patched to 1): after shutdown,
    # configuring twice as many logical GPUs as physical ones and initializing raises a value error
    # mentioning the NCCL environment variable (virtual GPUs are incompatible with NCCL).
    self.skipForDeviceType(['CPU', 'TPU'], reason='Test is intended for GPUs')

    accelerator_util.shutdown_accelerator_system()
    num_physical_devices = config.num_local_devices('GPU')
    test_util.reset_logical_devices('GPU', 2 * num_physical_devices)
    with self.assertRaisesRegex(ValueError,
                                'DTENSOR_GPU_USE_NCCL_COMMUNICATION'):
      accelerator_util.initialize_accelerator_system('GPU')

  @mock.patch.dict(os.environ, {'DTENSOR_GPU_USE_NCCL_COMMUNICATION': '1'})
  def test_initialize_with_nccl(self):
    # [FUNCTION DOC]
    # Test (GPU only, NCCL environment variable patched to 1): reinitializes the accelerator
    # system for GPU, creates a one-dimension mesh over all GPUs and runs a barrier on it. Only
    # checks the calls succeed; it cannot verify NCCL is actually used.
    self.skipForDeviceType(['CPU', 'TPU'], reason='Test is intended for GPUs')

    accelerator_util.shutdown_accelerator_system()
    accelerator_util.initialize_accelerator_system('GPU')

    num_devices = len(test_util.list_local_logical_devices('GPU'))
    mesh = mesh_util.create_mesh([('dim', num_devices)], device_type='GPU')
    # The following shall run, but there is no clear way to check if it uses
    # Collectives backed by NCCL.
    mesh_util.barrier(mesh)

  def test_initialize_after_tensorflow(self):
    # [FUNCTION DOC]
    # Test: after shutdown, initializing the TensorFlow context first and then the accelerator
    # system raises a value error saying TensorFlow has already been initialized.
    accelerator_util.shutdown_accelerator_system()
    context.ensure_initialized()

    with self.assertRaisesRegex(ValueError,
                                'TensorFlow has already been initialized'):
      accelerator_util.initialize_accelerator_system('CPU')

  def test_initialize_after_tensorflow_with_reset(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the log assertion indexes the captured log output as a nested sequence and reads a
    # message attribute from the first element, which looks like a mismatch with the structure
    # of the captured log (possibly an incorrect access).
    # Test: after shutdown, 32 logical CPUs are configured and TensorFlow is initialized; then
    # initializing the accelerator system for CPU with the reset-context option logs a warning
    # mentioning that option and preserves the 32 logical CPU devices.
    accelerator_util.shutdown_accelerator_system()
    test_util.reset_logical_devices('CPU', 32)
    context.ensure_initialized()

    with self.assertLogs(level='WARNING') as log:
      accelerator_util.initialize_accelerator_system(
          'CPU', experimental_reset_context=True
      )

    self.assertIn('experimental_reset_context', log[0][0].message)
    # Preserves the original logical device setting.
    self.assertLen(test_util.list_local_logical_devices('CPU'), 32)

  @parameterized.parameters(
      dict(
          device_type='CPU', skip_for=[]
      ),  # We can create CPU meshes on TPU and GPU platforms!
      dict(device_type='GPU', skip_for=['CPU', 'TPU']),
      dict(device_type='TPU', skip_for=['CPU', 'GPU']),
  )
  def test_initialize_with_manual_logical_cpu_devices(
      self, device_type: str, skip_for: list[str]
  ):
    # [FUNCTION DOC]
    # Parameterized test. Inputs: device_type (CPU, GPU or TPU) to initialize for, and skip_for, a
    # list of device types of the test environment on which the test is skipped. After shutdown and
    # resetting to one logical CPU, initializes the system requesting 32 logical CPU devices and
    # asserts exactly 32 local logical CPUs exist.
    self.skipForDeviceType(
        skip_for,
        reason=f'Test is not intended for {skip_for}',
    )
    accelerator_util.shutdown_accelerator_system()
    test_util.reset_logical_devices('CPU', 1)
    accelerator_util.initialize_accelerator_system(
        device_type, num_logical_cpu_devices=32
    )
    self.assertLen(test_util.list_local_logical_devices('CPU'), 32)

  def test_shutdown_accelerator_system(self):
    # [FUNCTION DOC]
    # Test: shutting down turns the initialized state off, and shutting down a second time raises
    # a value error saying the system is not initialized.
    self.assertTrue(accelerator_util.is_initialized())
    accelerator_util.shutdown_accelerator_system()
    self.assertFalse(accelerator_util.is_initialized())

    with self.assertRaisesRegex(ValueError, 'not initialized'):
      accelerator_util.shutdown_accelerator_system()

  def test_distributed_tpu_mesh_creation(self):
    # [FUNCTION DOC]
    # Test (TPU with exactly eight cores only): creates a distributed one-dimension TPU mesh
    # across all local TPU devices and asserts eight local devices and size eight.
    self.skipForDeviceType(['CPU', 'GPU'], reason='Test is intended for TPUs')
    self.skipForDeviceType(['TPU'],
                           reason='Test requires exactly 8 cores',
                           unless_device_count_equals_to=8)

    num_devices = len(test_util.list_local_logical_devices('TPU'))
    mesh = mesh_util.create_distributed_mesh(
        mesh_name='distributed_1d_mesh',
        mesh_dims=[('x', num_devices)],
        device_type='TPU')
    self.assertEqual(mesh.num_local_devices(), 8)
    self.assertEqual(mesh.size, 8)

  def test_mesh_barrier(self):
    # [FUNCTION DOC]
    # Smoke test: builds a one-dimension mesh over all local devices of the preferred type and calls
    # the mesh barrier with and without a name. Only checks that both calls run; barrier
    # semantics are not verified.
    device_type = config.preferred_device_type()

    num_devices = len(test_util.list_local_logical_devices(device_type))
    mesh = mesh_util.create_mesh([('dim', num_devices)],
                                 device_type=device_type)

    # FIXME(b/235416015): To really test this we'll need a new eager async
    # API.  The following shall run, but the barrier semantics is not tested.
    mesh_util.barrier(mesh, 'Name')
    mesh_util.barrier(mesh)


if __name__ == '__main__':
  test.main()
