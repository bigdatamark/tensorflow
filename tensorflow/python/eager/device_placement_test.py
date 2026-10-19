# [MODULE DOC]
# Tests for where eager ops and functions are placed. Covers soft placement (unsupported or
# unknown devices fall back to CPU or the first GPU), hard placement (explicit devices honored,
# with two virtual CPUs configured), and placement in a local cluster with remote workers. Many
# tests require a GPU and are decorated to run only when one exists. No import-time side effects;
# running as a script starts the test main.

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
"""Tests for device placement."""

from absl.testing import parameterized

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops


class SoftDevicePlacementTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for eager placement with soft device placement enabled. The fixture resets the eager
  # context, initializes it, turns soft placement on and enables device placement logging.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: calls the base setup, resets and initializes the global eager context, enables
    # soft device placement and turns on device placement logging. Returns nothing.
    super(SoftDevicePlacementTest, self).setUp()
    context._reset_context()
    context.ensure_initialized()
    config.set_soft_device_placement(enabled=True)
    context.context().log_device_placement = True

  @test_util.run_gpu_only
  def testDefaultPlacement(self):
    # [FUNCTION DOC]
    # GPU-only test. Adds two integer constants with no device scope and again under a CPU
    # scope. Asserts the unscoped result lives on a GPU and the scoped one on a CPU.
    a = constant_op.constant(1)
    b = constant_op.constant(2)
    c = a + b
    with ops.device('CPU'):
      d = a + b
    self.assertIn('GPU', c.device)
    self.assertIn('CPU', d.device)

  @test_util.run_gpu_only
  def testUnsupportedDevice(self):
    # [FUNCTION DOC]
    # GPU-only test. Under a first-GPU scope, adds two integers and indexes a string tensor (an
    # op without a GPU kernel). Asserts the addition stays on the first GPU while the string
    # indexing is soft-placed on the CPU.
    a = constant_op.constant(1)
    b = constant_op.constant(2)
    s = constant_op.constant(list('hello world'))
    with ops.device('GPU:0'):
      c = a + b
      t = s[a]
    self.assertIn('GPU:0', c.device)
    self.assertIn('CPU', t.device)

  @test_util.run_gpu_only
  def testUnknownDevice(self):
    # [FUNCTION DOC]
    # GPU-only test. Adds two integers under a scope naming a nonexistent GPU index 42. Asserts
    # soft placement puts the result on the first GPU.
    a = constant_op.constant(1)
    b = constant_op.constant(2)
    with ops.device('GPU:42'):
      c = a + b
    self.assertIn('GPU:0', c.device)

  def testNoGpu(self):
    # [FUNCTION DOC]
    # Test meaningful only on machines without a GPU (returns immediately if one is available).
    # Adds two integers with no scope and under a GPU scope. Asserts both results land on a CPU
    # because soft placement falls back.
    if test_util.is_gpu_available():
      # CPU only test.
      return
    a = constant_op.constant(1)
    b = constant_op.constant(2)
    c = a + b
    with ops.device('GPU'):
      d = a + b
    self.assertIn('CPU', c.device)
    self.assertIn('CPU', d.device)

  @test_util.run_gpu_only
  def testSoftPlacedGPU(self):
    # [FUNCTION DOC]
    # GPU-only test. Adds two integers under a scope naming GPU index 110, which does not exist.
    # Asserts the result is soft-placed on the first GPU.
    a = constant_op.constant(1)
    b = constant_op.constant(2)
    with ops.device('GPU:110'):
      c = a + b
    self.assertIn('GPU:0', c.device)

  @test_util.run_gpu_only
  def testNestedDeviceScope(self):
    # [FUNCTION DOC]
    # GPU-only test. Adds two integers inside a CPU scope nested with an inner scope naming the
    # nonexistent GPU index 42. Asserts the result ends on the first GPU (the inner scope wins
    # and is soft-placed; the test comment notes nested placement is not otherwise supported).
    a = constant_op.constant(1)
    b = constant_op.constant(2)
    with ops.device('CPU:0'):
      with ops.device('GPU:42'):
        c = a + b
    # We don't support nested device placement right now.
    self.assertIn('GPU:0', c.device)

  @parameterized.named_parameters(('float', 1.0, None),
                                  ('int32', [1], dtypes.int32),
                                  ('string', ['a'], None))
  def testSoftPlacedCPUConstant(self, value, dtype):
    # [FUNCTION DOC]
    # Parameterized over a float scalar, an int32 list and a string list (value and dtype).
    # Skipped when a GPU is available. Creates a constant under a first-GPU scope; since no GPU
    # exists it is soft-placed. Asserts both the device and the backing device are the first CPU.
    if test_util.is_gpu_available():
      self.skipTest('CPU only test')
    with ops.device('GPU:0'):
      a = constant_op.constant(value, dtype=dtype)
    self.assertIn('CPU:0', a.device)
    self.assertIn('CPU:0', a.backing_device)

  @parameterized.named_parameters(
      ('float', [1.0, 2.0], None, 'GPU:0'),
      ## TODO(b/179035075) enable when int32 numeric tensors are not confused
      ## with shape tensors
      # ('int32', [1, 2], dtypes.int32, 'GPU:0'),
      ('int64', [1, 2], dtypes.int64, 'GPU:0'))
  @test_util.run_gpu_only
  def testSoftPlacedNumericTensors(self, value, dtype, expect):
    # [FUNCTION DOC]
    # GPU-only parameterized test (float list and int64 list cases; the int32 case is disabled
    # pending a bug about confusion with shape tensors). Inputs: value, dtype, expect (device
    # name fragment). Adds two constants under a first-GPU scope and asserts the result's
    # backing device contains the expected GPU name.
    with ops.device('GPU:0'):
      a = math_ops.add(constant_op.constant(value, dtype=dtype),
                       constant_op.constant(value, dtype=dtype))
    self.assertIn(expect, a.backing_device)

  @test_util.run_gpu_only
  def testSoftPlacedShapeTensor(self):
    # [FUNCTION DOC]
    # GPU-only test. Under a first-GPU scope, takes the shape of a two-by-two integer constant
    # and adds an int32 vector to it. Asserts the result's backing device is the CPU because
    # shape computations stay on the host.
    with ops.device('GPU:0'):
      t = constant_op.constant([[1, 2], [3, 4]])
      a = math_ops.add(array_ops.shape(t),
                       constant_op.constant([10, 20], dtype=dtypes.int32))
    # Shape computations remain on CPU
    self.assertIn('CPU:0', a.backing_device)

  def testPlacedToDeviceInFunction(self):
    # [FUNCTION DOC]
    # Defines a traced function that builds a random 32 by 32 matrix and multiplies it by itself,
    # calls it, and asserts the result is on the first GPU if any physical GPU exists, otherwise
    # on the first CPU.

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Nested traced function: returns the product of a random uniform 32 by 32 matrix with
      # itself. No inputs.
      a = random_ops.random_uniform([32, 32])
      return math_ops.matmul(a, a)

    gpus = config.list_physical_devices('GPU')
    if not gpus:
      self.assertIn('CPU:0', f().device)
    else:
      self.assertIn('GPU:0', f().device)

  @test_util.disable_tfrt('b/173726713: Support properly inserting device at '
                          'tf_to_corert lowering.')
  def testUnknownDeviceInFunction(self):
    # [FUNCTION DOC]
    # Defines a traced function that adds constants under a scope naming nonexistent GPU 42, then
    # adds another constant outside it. Calls it and asserts the result is on the first GPU if a
    # physical GPU exists, otherwise on the first CPU (the placer replaces the unknown device).
    # Skipped under the alternative runtime because of a lowering limitation.

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Nested traced function: adds two constants under a scope for nonexistent GPU 42, then
      # adds a third constant to the sum and returns it. No inputs.
      with ops.device('GPU:42'):
        # With placer, the unknown GPU:42 will be replaced with GPU:0.
        a = constant_op.constant(1) + constant_op.constant(2)
      return a + constant_op.constant(2)

    gpus = config.list_physical_devices('GPU')
    if not gpus:
      self.assertIn('CPU:0', f().device)
    else:
      self.assertIn('GPU:0', f().device)


class HardDevicePlacementTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for eager placement with soft placement disabled, so requested devices are honored.
  # The fixture configures two logical CPU devices.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: calls the base setup, resets the eager context, disables soft placement, enables
    # placement logging, splits the first physical CPU into two logical CPUs, and asserts that
    # soft placement reads back as disabled through both the config API and the context.
    super(HardDevicePlacementTest, self).setUp()
    context._reset_context()
    config.set_soft_device_placement(enabled=False)
    context.context().log_device_placement = True
    cpus = context.context().list_physical_devices('CPU')
    # Set 2 virtual CPUs
    context.context().set_logical_device_configuration(cpus[0], [
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration()
    ])
    self.assertEqual(config.get_soft_device_placement(), False)
    self.assertEqual(context.context().soft_device_placement, False)

  @test_util.run_gpu_only
  def testIdentityCanCopy(self):
    # [FUNCTION DOC]
    # GPU-only test. Sets the device policy to explicit, creates a float constant on the first
    # CPU, then applies identity under a first-GPU scope. Asserts the input is on the CPU and the
    # copied result's device and backing device are the first GPU.
    config.set_device_policy('explicit')
    with ops.device('CPU:0'):
      x = constant_op.constant(1.0)
      self.assertIn('CPU:0', x.device)
      self.assertIn('CPU:0', x.backing_device)
    with ops.device('GPU:0'):
      y = array_ops.identity(x)
      self.assertIn('GPU:0', y.device)
      self.assertIn('GPU:0', y.backing_device)

  @test_util.run_gpu_only
  def testSimpleConstantsExplicitGPU(self):
    # [FUNCTION DOC]
    # GPU-only test. With explicit device policy and a first-GPU scope, checks that ones, zeros
    # and fill of a scalar or one-element shape produce the expected values.
    config.set_device_policy('explicit')
    with ops.device('GPU:0'):
      self.assertAllClose(1., array_ops.ones([]))
      self.assertAllClose(0., array_ops.zeros([]))
      self.assertAllClose([1.], array_ops.fill([1], 1.))

  def testSimpleConstantsExplicitCPU(self):
    # [FUNCTION DOC]
    # With explicit device policy and the second logical CPU scope, checks that ones, zeros, fill
    # and multiplying a constant by a scalar produce the expected values.
    config.set_device_policy('explicit')
    with ops.device('CPU:1'):
      self.assertAllClose(1., array_ops.ones([]))
      self.assertAllClose(0., array_ops.zeros([]))
      self.assertAllClose([1.], array_ops.fill([1], 1.))
      self.assertAllClose(2., constant_op.constant(1.) * 2.)

  @parameterized.named_parameters(
      ('float_cpu0', 'CPU:0', 1.0, None),
      ('int32_cpu0', 'CPU:0', [1], dtypes.int32),
      ('string_cpu0', 'CPU:0', ['a'], None),
      ('float_cpu1', 'CPU:1', 1.0, None),
      ('int32_cpu1', 'CPU:1', [1], dtypes.int32),
      ('string_cpu1', 'CPU:1', ['a'], None),
  )
  def testHardPlacedCPUConstant(self, device, value, dtype):
    # [FUNCTION DOC]
    # Parameterized over float, int32 and string values on each of the two logical CPUs.
    # Inputs: device name, value, dtype. Creates a constant under that device scope and asserts
    # its device name contains the requested device.
    with ops.device(device):
      a = constant_op.constant(value, dtype=dtype)
      self.assertIn(device, a.device)

  @parameterized.named_parameters(
      ('float', 'GPU:0', 1.0, None),
      ('int32', 'GPU:0', [1], dtypes.int32),
      ('string', 'GPU:0', ['a'], None),
  )
  def testHardPlacedGPUConstant(self, device, value, dtype):
    # [FUNCTION DOC]
    # Parameterized over float, int32 and string values on the first GPU; skipped without a GPU.
    # Creates a constant under the device scope and asserts its device contains the requested
    # name; for float32 values also asserts the backing device matches.
    if not test_util.is_gpu_available():
      self.skipTest('Test requires a GPU')
    with ops.device(device):
      a = constant_op.constant(value, dtype=dtype)
      self.assertIn(device, a.device)
      if a.dtype == dtypes.float32:
        self.assertIn(device, a.backing_device)


class ClusterPlacementTest(test.TestCase):
  # [CLASS DOC]
  # Tests for placement in a cluster: the fixture starts a local cluster of two workers and
  # connects the eager context to them, with soft placement enabled. Tests are skipped under
  # the alternative runtime because remote hosts are unsupported there.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: calls the base setup, resets the eager context, enables soft placement and
    # placement logging, creates a local cluster of two workers and no parameter servers, and
    # connects the context to both worker targets. Returns nothing.
    super(ClusterPlacementTest, self).setUp()
    context._reset_context()
    config.set_soft_device_placement(enabled=True)
    context.context().log_device_placement = True
    workers, _ = test_util.create_local_cluster(2, 0)
    remote.connect_to_remote_host([workers[0].target, workers[1].target])

  @test_util.disable_tfrt('remote host not supported yet.')
  def testNotFullySpecifiedTask(self):
    # [FUNCTION DOC]
    # Adds two integers under a scope naming only the worker job. Asserts the result is placed
    # on replica zero, task zero of that job.
    a = constant_op.constant(1)
    b = constant_op.constant(2)
    with ops.device('/job:worker'):
      c = a + b
    self.assertIn('/job:worker/replica:0/task:0', c.device)

  @test_util.disable_tfrt('remote host not supported yet.')
  def testRemoteUnknownDevice(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the assertion on the message text sits inside the assertRaises block after the
    # statement that raises, so it is never executed; only the raised error type is verified.
    # Scenario: adds two integers under a fully specified remote scope with nonexistent GPU 42 and
    # expects an invalid-argument error, since soft placement is unsupported on remote workers.
    a = constant_op.constant(1)
    b = constant_op.constant(2)
    # Right now we don't support soft device place on remote worker.
    with self.assertRaises(errors.InvalidArgumentError) as cm:
      with ops.device('/job:worker/replica:0/task:0/device:GPU:42'):
        c = a + b
        del c
      self.assertIn('unknown device', cm.exception.message)

  @test_util.disable_tfrt('remote host not supported yet.')
  def testUnknownDeviceInFunctionReturnUnknowDevice(self):
    # [FUNCTION DOC]
    # Defines a traced function that returns the sum of two constants computed under a scope for
    # nonexistent GPU 42. Calls it and asserts the result is on the first GPU if any physical GPU
    # exists, otherwise on the first CPU.

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Nested traced function: returns the sum of two integer constants computed under a scope
      # naming nonexistent GPU 42. No inputs.
      with ops.device('GPU:42'):
        return constant_op.constant(1) + constant_op.constant(2)

    gpus = config.list_physical_devices('GPU')
    if not gpus:
      self.assertIn('CPU:0', f().device)
    else:
      self.assertIn('GPU:0', f().device)

  @test_util.disable_tfrt('remote host not supported yet.')
  def testUnknownDeviceInFunction(self):
    # [FUNCTION DOC]
    # Same scenario as the identically named soft placement test but in the cluster fixture:
    # a traced function adds constants under a scope for nonexistent GPU 42 and then another
    # constant outside it. Asserts the result is on the first GPU if a physical GPU exists,
    # otherwise on the first CPU.

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Nested traced function: adds two constants under a scope naming nonexistent GPU 42, then
      # adds a third constant outside the scope and returns the sum. No inputs.
      with ops.device('GPU:42'):
        a = constant_op.constant(1) + constant_op.constant(2)
      return a + constant_op.constant(2)

    gpus = config.list_physical_devices('GPU')
    if not gpus:
      self.assertIn('CPU:0', f().device)
    else:
      self.assertIn('GPU:0', f().device)


if __name__ == '__main__':
  test.main()
