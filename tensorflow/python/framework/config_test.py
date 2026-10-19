# [MODULE DOC]
# Tests for the system configuration API in the framework config module: device policy, execution
# mode, thread pools, soft placement, placement logging, MLIR bridge, JIT and optimizer options,
# physical and logical device listing and configuration, visibility, memory growth and statistics,
# legacy configuration compatibility, collective ops, op determinism and TensorFloat-32. Many tests
# need a GPU and skip otherwise; most reset the eager context afterwards through a helper decorator.
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
"""Tests that the system configuration methods work properly."""

from absl.testing import parameterized

from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.core.protobuf import tensorflow_server_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
from tensorflow.python.util import compat


def reset_eager(fn):
  # [FUNCTION DOC]
  # HELPER: decorator for the tests in this file that restores a clean eager context afterwards.
  # Takes a test function and returns a wrapper that calls it with the same arguments and, even
  # on failure, resets the JIT compiler flags and the global context, re-enables eager execution
  # and asserts a context exists. Side effect: replaces the global runtime context.

  def wrapper(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: wrapper produced by the reset decorator. Forwards all positional and keyword
    # arguments to the wrapped test and returns its result; in a final step always resets the
    # JIT flags and global context and re-enables eager execution. Raises an assertion error if
    # no context exists afterwards; exceptions from the wrapped test propagate.
    try:
      return fn(*args, **kwargs)
    finally:
      # Reset the context.
      context._reset_jit_compiler_flags()
      context._reset_context()
      ops.enable_eager_execution_internal()
      assert context._context is not None

  return wrapper


@test_util.with_eager_op_as_function
class ConfigTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case (with parameterized support, run with eager op-as-function) for global runtime
  # configuration: device policy, execution mode, thread counts, soft placement, device placement
  # logging, MLIR bridge flags, JIT and graph optimizer options. Each test is wrapped to reset the
  # eager context afterwards.

  @test_util.disable_eager_op_as_function('b/204320409')
  @test_util.run_gpu_only
  @reset_eager
  def testDevicePolicy(self):
    # [FUNCTION DOC]
    # GPU-only. Scenario: the default device policy is silent; thread settings and policy may be
    # changed before initialization. After initialization, cycles the policy through silent,
    # silent-for-int32, warn and explicit, checking the getter and the context each time and
    # copying a tensor from CPU to GPU. Expected: silent-for-int32 raises an invalid-argument error
    # (conflicting devices) for float32 but not int32; explicit raises it for int32 as well;
    # setting the policy to nothing restores silent.
    self.assertEqual(context.DEVICE_PLACEMENT_SILENT,
                     context.context().device_policy)

    # If no op has been executed we should be able to set the device policy as
    # well as any init-time configs.
    config.set_intra_op_parallelism_threads(1)
    config.set_device_policy('silent')
    config.set_intra_op_parallelism_threads(2)

    context.ensure_initialized()

    def copy_tensor(dtype=dtypes.int32):
      # [FUNCTION DOC]
      # HELPER: used by the device policy test. Creates a constant one of the given dtype
      # (default 32-bit integer) on the CPU, copies it to the GPU and asserts that adding the two
      # tensors gives two. Raises whatever the current device policy raises for the copy.
      with ops.device('CPU:0'):
        cpu_tensor = constant_op.constant(1, dtype=dtype)
      gpu_tensor = cpu_tensor.gpu()
      self.assertAllEqual(cpu_tensor + gpu_tensor, 2.0)

    config.set_device_policy('silent')
    self.assertEqual(config.get_device_policy(), 'silent')
    self.assertEqual(context.DEVICE_PLACEMENT_SILENT,
                     context.context().device_policy)
    copy_tensor()

    config.set_device_policy('silent_for_int32')
    self.assertEqual(config.get_device_policy(), 'silent_for_int32')
    self.assertEqual(context.DEVICE_PLACEMENT_SILENT_FOR_INT32,
                     context.context().device_policy)
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'Tensors on conflicting devices'):
      copy_tensor(dtypes.float32)
    copy_tensor()

    config.set_device_policy('warn')
    self.assertEqual(config.get_device_policy(), 'warn')
    self.assertEqual(context.DEVICE_PLACEMENT_WARN,
                     context.context().device_policy)
    copy_tensor()

    config.set_device_policy('explicit')
    self.assertEqual(config.get_device_policy(), 'explicit')
    self.assertEqual(context.DEVICE_PLACEMENT_EXPLICIT,
                     context.context().device_policy)
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'Tensors on conflicting devices'):
      copy_tensor()

    config.set_device_policy(None)
    self.assertEqual(config.get_device_policy(), 'silent')

  @reset_eager
  def testExecutionMode(self):
    # [FUNCTION DOC]
    # Scenario: execution is synchronous by default. Before any op runs, thread counts and the
    # synchronous flag may be set. Then toggles synchronous execution on and off. Expected: the
    # getter and the context's execution mode (sync or async) follow each change.
    self.assertTrue(config.get_synchronous_execution())
    self.assertEqual(context.SYNC, context.context().execution_mode)

    # If no op has been executed we should be able to set the execution mode as
    # well as any init-time configs.
    config.set_intra_op_parallelism_threads(1)
    config.set_synchronous_execution(False)
    config.set_intra_op_parallelism_threads(2)

    config.set_synchronous_execution(True)
    self.assertTrue(config.get_synchronous_execution())
    self.assertEqual(context.SYNC, context.context().execution_mode)
    config.set_synchronous_execution(False)
    self.assertFalse(config.get_synchronous_execution())
    self.assertEqual(context.ASYNC, context.context().execution_mode)

  @reset_eager
  def testIntraOpParallelismThreads(self):
    # [FUNCTION DOC]
    # Scenario: set intra-op threads to 10 and check the getter matches the context; initialize the
    # context. Expected: setting a different value (1) afterwards raises a runtime error, while
    # re-setting the same value (10) is allowed.
    config.set_intra_op_parallelism_threads(10)
    self.assertEqual(config.get_intra_op_parallelism_threads(),
                     context.context().intra_op_parallelism_threads)

    context.ensure_initialized()

    with self.assertRaises(RuntimeError):
      config.set_intra_op_parallelism_threads(1)

    config.set_intra_op_parallelism_threads(10)

  @reset_eager
  def testInterOpParallelismThreads(self):
    # [FUNCTION DOC]
    # Scenario: set inter-op threads to 10 and check the getter matches the context; initialize the
    # context. Expected: setting a different value (1) afterwards raises a runtime error, while
    # re-setting the same value (10) is allowed.
    config.set_inter_op_parallelism_threads(10)
    self.assertEqual(config.get_inter_op_parallelism_threads(),
                     context.context().inter_op_parallelism_threads)

    context.ensure_initialized()

    with self.assertRaises(RuntimeError):
      config.set_inter_op_parallelism_threads(1)

    config.set_inter_op_parallelism_threads(10)

  @test_util.run_gpu_only
  @reset_eager
  def testSoftPlacement(self):
    # [FUNCTION DOC]
    # GPU-only. Scenario: soft device placement defaults to enabled when eager. With it enabled,
    # an op pinned to the GPU that lacks a GPU kernel runs (falls back to CPU) both eagerly and
    # inside a function. After disabling it, expected: the same op raises an invalid-argument error
    # in both modes. The getter is checked against the context at each step.
    if context.executing_eagerly():
      self.assertTrue(config.get_soft_device_placement())
    else:
      self.assertFalse(config.get_soft_device_placement())

    def test_attr():
      # [FUNCTION DOC]
      # HELPER: runs the test attribute op with float32 type pinned to the first GPU device and
      # returns its result; used to probe soft placement behavior.
      with ops.device('/device:GPU:0'):
        return test_ops.test_attr(T=dtypes.float32, name='test_attr')

    config.set_soft_device_placement(True)
    self.assertEqual(config.get_soft_device_placement(), True)
    self.assertEqual(config.get_soft_device_placement(),
                     context.context().soft_device_placement)

    # Since soft placement is enabled, the test_attr operation should fallback
    # to CPU with pure eager execution as well as functions
    test_attr()
    def_function.function(test_attr)()

    config.set_soft_device_placement(False)
    self.assertEqual(config.get_soft_device_placement(), False)
    self.assertEqual(config.get_soft_device_placement(),
                     context.context().soft_device_placement)

    # Since soft placement is disabled, the test_attr operation should fail on
    # GPU with pure eager execution as well as functions
    with self.assertRaises(errors.InvalidArgumentError):
      test_attr()
    with self.assertRaises(errors.InvalidArgumentError):
      def_function.function(test_attr)()

  @reset_eager
  def testLogDevicePlacement(self):
    # [FUNCTION DOC]
    # Scenario: device placement logging defaults to off. Toggles it on and off, checking the
    # getter against the context each time, initializes the context, then enables it again.
    # Expected: values follow each change and changing it after initialization raises no error.
    self.assertFalse(context.get_log_device_placement())

    context.set_log_device_placement(True)
    self.assertEqual(context.get_log_device_placement(), True)
    self.assertEqual(context.get_log_device_placement(),
                     context.context().log_device_placement)

    context.set_log_device_placement(False)
    self.assertEqual(context.get_log_device_placement(), False)
    self.assertEqual(context.get_log_device_placement(),
                     context.context().log_device_placement)

    context.ensure_initialized()

    # Changing the device placement should not throw an exception
    context.set_log_device_placement(True)

  @reset_eager
  def testEnableMlirBridge(self):
    # [FUNCTION DOC]
    # Scenario: by default the MLIR bridge flag is false and its rollout is unspecified. Enabling
    # it sets the flag true and rollout enabled; disabling it sets the flag false and rollout
    # disabled. All checked on the context's configuration.
    # Default value of enable_mlir_bridge is false.
    self.assertFalse(context.context().config.experimental.enable_mlir_bridge)
    self.assertEqual(
        context.context().config.experimental.mlir_bridge_rollout,
        config_pb2.ConfigProto.Experimental.MLIR_BRIDGE_ROLLOUT_UNSPECIFIED)

    # Tests enabling mlir bridge.
    config.enable_mlir_bridge()
    self.assertTrue(context.context().config.experimental.enable_mlir_bridge)
    self.assertEqual(
        context.context().config.experimental.mlir_bridge_rollout,
        config_pb2.ConfigProto.Experimental.MLIR_BRIDGE_ROLLOUT_ENABLED)

    # Tests disabling mlir bridge.
    config.disable_mlir_bridge()
    self.assertFalse(context.context().config.experimental.enable_mlir_bridge)
    self.assertEqual(
        context.context().config.experimental.mlir_bridge_rollout,
        config_pb2.ConfigProto.Experimental.MLIR_BRIDGE_ROLLOUT_DISABLED)

  @reset_eager
  def testResetMlirFlags(self):
    # [FUNCTION DOC]
    # Scenario: after the reset decorator restored the context from the preceding test, checks the
    # MLIR bridge flag is false and the rollout unspecified, i.e. flags do not leak between tests.
    # Changes nothing itself.
    # Default value of enable_mlir_bridge is false.
    self.assertFalse(context.context().config.experimental.enable_mlir_bridge)
    self.assertEqual(
        context.context().config.experimental.mlir_bridge_rollout,
        config_pb2.ConfigProto.Experimental.MLIR_BRIDGE_ROLLOUT_UNSPECIFIED)

  @test_util.run_gpu_only
  @reset_eager
  def testJit(self):
    # [FUNCTION DOC]
    # GPU-only. Scenario: the optimizer JIT setting defaults to empty. Runs a small multiply-add
    # function before, during and after setting the JIT mode to auto-clustering and back to empty.
    # Expected: the getter reflects each setting and the function evaluates without error. The
    # test cannot verify fusion actually happened.
    self.assertEqual(config.get_optimizer_jit(), '')

    # the following function should cause Op fusion to occur. However, there is
    # unfortunately no straightforward way to ensure this. We will just have to
    # settle for creating a test that can trigger JIT.
    @def_function.function
    def fun(a, b):
      # [FUNCTION DOC]
      # Traced function for the JIT test: multiplies a and b, adds a to the product and returns
      # the sum (an elementwise computation that can be fused).
      c = a * b
      d = c + a
      return d

    a = constant_op.constant([2., 2.])
    b = constant_op.constant([2., 2.])

    self.evaluate(fun(a, b))

    config.set_optimizer_jit('autoclustering')
    self.assertEqual(config.get_optimizer_jit(), 'autoclustering')

    self.evaluate(fun(a, b))

    config.set_optimizer_jit('')
    self.assertEqual(config.get_optimizer_jit(), '')

    self.evaluate(fun(a, b))

  @parameterized.named_parameters(
      ('LayoutOptimizer', 'layout_optimizer'),
      ('ConstantFolding', 'constant_folding'),
      ('ShapeOptimization', 'shape_optimization'), ('Remapping', 'remapping'),
      ('ArithmeticOptimization', 'arithmetic_optimization'),
      ('DependencyOptimization', 'dependency_optimization'),
      ('LoopOptimization', 'loop_optimization'),
      ('FunctionOptimization', 'function_optimization'),
      ('DebugStripper', 'debug_stripper'),
      ('ScopedAllocatorOptimization', 'scoped_allocator_optimization'),
      ('ImplementationSelector', 'implementation_selector'),
      ('AutoMixedPrecision', 'auto_mixed_precision'))
  @reset_eager
  def testOptimizerToggleOption(self, field):
    # [FUNCTION DOC]
    # Parameterized over names of toggle-style optimizer options (layout, constant folding, shape,
    # remapping, arithmetic, dependency, loop, function, debug stripper, scoped allocator,
    # implementation selector, auto mixed precision). Scenario: the option is initially unset;
    # set it true, then false through the config API. Expected: both the config getter and the
    # context's getter return the full options mapping with the option updated each time.
    # TODO(b/128531235): Improve testing of option
    options = config.get_optimizer_experimental_options()
    self.assertIsNone(options.get(field))

    config.set_optimizer_experimental_options({field: True})
    options[field] = True
    self.assertDictEqual(config.get_optimizer_experimental_options(), options)
    self.assertDictEqual(context.context().get_optimizer_experimental_options(),
                         options)

    config.set_optimizer_experimental_options({field: False})
    options[field] = False
    self.assertDictEqual(config.get_optimizer_experimental_options(), options)
    self.assertDictEqual(context.context().get_optimizer_experimental_options(),
                         options)

  @parameterized.named_parameters(
      ('DisableModelPruning', 'disable_model_pruning'),
      ('DisableMetaOptimizer', 'disable_meta_optimizer'))
  @reset_eager
  def testOptimizerBoolOption(self, field):
    # [FUNCTION DOC]
    # Parameterized over the boolean optimizer options disable-model-pruning and
    # disable-meta-optimizer. Scenario: the option is initially falsy; set it true, then false.
    # Expected: both the config getter and the context's getter return the full options mapping
    # with the option updated each time.
    # TODO(b/128531235): Improve testing of option
    options = config.get_optimizer_experimental_options()
    self.assertFalse(options.get(field))

    config.set_optimizer_experimental_options({field: True})
    options[field] = True
    self.assertDictEqual(config.get_optimizer_experimental_options(), options)
    self.assertDictEqual(context.context().get_optimizer_experimental_options(),
                         options)

    config.set_optimizer_experimental_options({field: False})
    options[field] = False
    self.assertDictEqual(config.get_optimizer_experimental_options(), options)
    self.assertDictEqual(context.context().get_optimizer_experimental_options(),
                         options)

  @test_util.run_gpu_only
  @reset_eager
  def testOptimizerToggleOptionPinToHost(self):
    # [FUNCTION DOC]
    # GPU-only. Scenario: force the optimizer to run on every graph (minimum node count of minus
    # one) and evaluate a function returning the device placement op. Expected: with pin-to-host
    # off the result names a GPU; after enabling it the result names a CPU; after disabling it
    # again it names a GPU. The options mapping is checked against both getters after each change.
    options = config.get_optimizer_experimental_options()
    self.assertIsNone(options.get('pin_to_host_optimization'))

    @def_function.function
    def fun():
      # [FUNCTION DOC]
      # Traced function that returns the output of the test device placement op, which reports
      # the device it ran on.
      op = test_ops.device_placement_op()
      return op

    # Force optimizer to run for all graphs
    config.set_optimizer_experimental_options({'min_graph_nodes': -1})
    options['min_graph_nodes'] = -1

    # Since pin to host is disabled, the operation should go on GPU
    gpu = self.evaluate(fun())
    self.assertIn(compat.as_bytes('GPU'), gpu)

    config.set_optimizer_experimental_options(
        {'pin_to_host_optimization': True})
    options['pin_to_host_optimization'] = True
    self.assertDictEqual(config.get_optimizer_experimental_options(), options)
    self.assertDictEqual(context.context().get_optimizer_experimental_options(),
                         options)

    # Since pin to host is enabled, the operation should go on CPU
    cpu = self.evaluate(fun())
    self.assertIn(compat.as_bytes('CPU'), cpu)

    config.set_optimizer_experimental_options(
        {'pin_to_host_optimization': False})
    options['pin_to_host_optimization'] = False
    self.assertDictEqual(config.get_optimizer_experimental_options(), options)
    self.assertDictEqual(context.context().get_optimizer_experimental_options(),
                         options)

    # Since pin to host is disabled again, the operation should go on GPU
    gpu2 = self.evaluate(fun())
    self.assertIn(compat.as_bytes('GPU'), gpu2)


class DeviceTest(test.TestCase):
  # [CLASS DOC]
  # Test case for device listing and configuration: physical and logical devices, visible-device
  # selection, memory growth, memory limits and related configuration. Tests are wrapped to reset
  # the eager context afterwards.

  @reset_eager
  def testPhysicalDevices(self):
    # [FUNCTION DOC]
    # Scenario: list physical CPU devices; expected at least one. If a GPU is available, also list
    # physical GPU devices and expect at least one.
    cpus = config.list_physical_devices('CPU')
    self.assertGreater(len(cpus), 0)
    if test_util.is_gpu_available():
      gpus = config.list_physical_devices('GPU')
      self.assertGreater(len(gpus), 0)

  @reset_eager
  def testCpuMultiple(self):
    # [FUNCTION DOC]
    # Scenario: the single physical CPU is split into two logical CPUs. Expected: two logical CPUs
    # are listed; constants can be placed on CPU 0 and 1, and CPU 2 (nonexistent) falls back to
    # GPU 0 if a GPU exists else CPU 0; ops can be placed on each listed logical device name.
    # Changing to a three-way split raises a runtime error saying it cannot be modified; repeating
    # the same two-way split is accepted.
    cpus = config.list_physical_devices('CPU')
    self.assertEqual(len(cpus), 1)

    config.set_logical_device_configuration(cpus[0], [
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration()
    ])

    context.ensure_initialized()

    vcpus = config.list_logical_devices('CPU')
    self.assertEqual(len(vcpus), 2)

    with ops.device('/device:CPU:0'):
      a = constant_op.constant(1.0)
      self.evaluate(a)
    with ops.device('/device:CPU:1'):
      b = constant_op.constant(1.0)
      self.evaluate(b)
    with ops.device('/device:CPU:2'):
      c = constant_op.constant(1.0)
      self.evaluate(c)
    if test_util.is_gpu_available():
      self.assertIn('GPU:0', c.device)
    else:
      self.assertIn('CPU:0', c.device)

    # Ensure we can place ops on each of the device names
    for vcpu in vcpus:
      with ops.device(vcpu.name):
        d = constant_op.constant(1.0)
        self.evaluate(d)

    # Modifying the CPU configuration is not supported
    with self.assertRaisesRegex(RuntimeError, 'cannot be modified'):
      config.set_logical_device_configuration(cpus[0], [
          context.LogicalDeviceConfiguration(),
          context.LogicalDeviceConfiguration(),
          context.LogicalDeviceConfiguration()
      ])

    # Setting the same CPU configuration is fine
    config.set_logical_device_configuration(cpus[0], [
        context.LogicalDeviceConfiguration(),
        context.LogicalDeviceConfiguration()
    ])

  @test_util.run_gpu_only
  @reset_eager
  def testGpuNone(self):
    # [FUNCTION DOC]
    # GPU-only. Scenario: disable soft placement, then make only the CPU visible. Expected: before,
    # one visible CPU, at least one visible GPU, and no XLA GPU devices; after, no visible GPUs and
    # no logical XLA GPU devices; placing an op on GPU 0 raises an invalid-argument error (could
    # not satisfy). Changing visible devices to the GPUs raises a runtime error (cannot be
    # modified); setting the same visibility again is fine.
    config.set_soft_device_placement(False)
    gpus = config.list_physical_devices('GPU')
    self.assertGreater(len(gpus), 0)

    cpus = config.list_physical_devices('CPU')
    self.assertEqual(len(cpus), 1)

    self.assertEqual(len(config.get_visible_devices('CPU')), 1)
    self.assertGreater(len(config.get_visible_devices('GPU')), 0)

    self.assertEqual(len(config.get_visible_devices('XLA_GPU')), 0)

    config.set_visible_devices(cpus[0])
    self.assertEqual(len(config.get_visible_devices('CPU')), 1)
    self.assertEqual(len(config.get_visible_devices('GPU')), 0)
    self.assertEqual(len(config.list_logical_devices('XLA_GPU')), 0)

    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'Could not satisfy'):
      with ops.device('/device:GPU:0'):
        a = array_ops.identity(1.0)
        self.evaluate(a)

    # Modifying the visible devices is not supported
    with self.assertRaisesRegex(RuntimeError, 'cannot be modified'):
      config.set_visible_devices(gpus)

    # Setting the same visible devices is fine
    config.set_visible_devices(cpus[0])

  @reset_eager
  def testGpuMultiple(self):
    # [FUNCTION DOC]
    # Scenario: with soft placement disabled and at least two physical GPUs (else skipped),
    # initialize the context and create a constant on each GPU by index. Expected: every placement
    # succeeds and evaluates.
    config.set_soft_device_placement(False)
    gpus = config.list_physical_devices('GPU')

    if len(gpus) < 2:
      self.skipTest('Need at least 2 GPUs')

    context.ensure_initialized()

    for i in range(0, len(gpus)):
      with ops.device('/device:GPU:' + str(i)):
        a = constant_op.constant(1.0)
        self.evaluate(a)

    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'Could not satisfy device specification'):
      with ops.device('/device:GPU:' + str(len(gpus))):
        a = constant_op.constant(1.0)
        self.evaluate(a)

  @reset_eager
  def testDeviceDetails(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the restricted-visibility branch only runs with more than two GPUs although
    # it selects the second GPU, so with exactly two GPUs it is silently skipped.
    # Scenario: details of the single physical CPU are an empty mapping. If a GPU is available, its
    # details must contain a non-empty string device name and, except on ROCm builds (which must
    # lack it), a compute capability pair with positive major and non-negative minor version.
    # Also checks details for the visible GPU after restricting visibility to the second GPU.
    (cpu,) = config.list_physical_devices('CPU')
    details = config.get_device_details(cpu)
    self.assertEqual(details, {})

    if not test_util.is_gpu_available():
      return

    gpus = config.list_physical_devices('GPU')
    details = config.get_device_details(gpus[0])
    self.assertIsInstance(details['device_name'], str)
    self.assertNotEmpty(details['device_name'])
    if test.is_built_with_rocm():
      # AMD GPUs do not have a compute capability
      self.assertNotIn('compute_capability', details)
    else:
      cc = details['compute_capability']
      self.assertIsInstance(cc, tuple)
      major, minor = cc
      self.assertGreater(major, 0)
      self.assertGreaterEqual(minor, 0)

    # Test GPU returned from get_visible_devices
    if len(gpus) > 2:
      config.set_visible_devices(gpus[1], 'GPU')
      (visible_gpu,) = config.get_visible_devices('GPU')
      details = config.get_device_details(visible_gpu)
      self.assertIsInstance(details['device_name'], str)

  @reset_eager
  def testDeviceDetailsErrors(self):
    # [FUNCTION DOC]
    # Scenario: ask for device details of a logical device, then of a fabricated physical device
    # not obtained from listing. Expected: a value error saying the argument must be a physical
    # device in the first case, and a value error saying it must come from listing physical
    # devices in the second.
    logical_devices = config.list_logical_devices()
    with self.assertRaisesRegex(ValueError,
                                'must be a tf.config.PhysicalDevice'):
      config.get_device_details(logical_devices[0])

    phys_dev = context.PhysicalDevice('/physical_device:CPU:100', 'CPU')
    with self.assertRaisesRegex(
        ValueError, 'The PhysicalDevice must be one obtained from '
        'calling `tf.config.list_physical_devices`'):
      config.get_device_details(phys_dev)

  @test_util.run_gpu_only
  @reset_eager
  def testVirtualGpu(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the check on the logical GPU count uses a truthiness assertion with the
    # expected count as its message, so it passes for any nonzero count and does not verify
    # the count.
    # GPU-only. Scenario: disable soft placement and split the last physical GPU into two virtual
    # devices of memory limit 10. Expected: the configuration getter was initially nothing and
    # then reports two; ops can be placed on every logical GPU, while placing on a nonexistent
    # index raises an invalid-argument error (could not satisfy). Changing the memory limits or
    # the number of virtual devices raises a runtime error (cannot be modified); re-applying the
    # same configuration is accepted.
    config.set_soft_device_placement(False)
    gpus = config.list_physical_devices('GPU')
    self.assertNotEqual(len(gpus), 0)

    self.assertIsNone(config.get_logical_device_configuration(gpus[-1]))
    config.set_logical_device_configuration(gpus[-1], [
        context.LogicalDeviceConfiguration(memory_limit=10),
        context.LogicalDeviceConfiguration(memory_limit=10)
    ])
    self.assertEqual(len(config.get_logical_device_configuration(gpus[-1])), 2)

    logical_gpus = config.list_logical_devices('GPU')
    self.assertTrue(len(logical_gpus), len(gpus) + 1)
    for i in range(0, len(logical_gpus)):
      with ops.device('/device:GPU:' + str(i)):
        a = array_ops.identity(1.0)
        self.evaluate(a)

    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'Could not satisfy'):
      with ops.device('/device:GPU:' + str(len(logical_gpus))):
        a = array_ops.identity(1.0)
        self.evaluate(a)

    # Modifying the GPU configuration is not supported
    with self.assertRaisesRegex(RuntimeError, 'cannot be modified'):
      config.set_logical_device_configuration(gpus[-1], [
          context.LogicalDeviceConfiguration(memory_limit=20),
          context.LogicalDeviceConfiguration(memory_limit=20)
      ])

    with self.assertRaisesRegex(RuntimeError, 'cannot be modified'):
      config.set_logical_device_configuration(gpus[-1], [
          context.LogicalDeviceConfiguration(memory_limit=10),
          context.LogicalDeviceConfiguration(memory_limit=10),
          context.LogicalDeviceConfiguration(memory_limit=10)
      ])

    # Setting the same GPU configuration is fine
    config.set_logical_device_configuration(gpus[-1], [
        context.LogicalDeviceConfiguration(memory_limit=10),
        context.LogicalDeviceConfiguration(memory_limit=10)
    ])

  @test_util.run_gpu_only
  @reset_eager
  def testGpuGrowth(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the logical GPU count check uses a truthiness assertion with the expected count
    # as its message, so it does not actually compare the counts.
    # GPU-only. Scenario: memory growth is initially unset for the last GPU; enable growth on every
    # GPU. Expected: the context configuration allows growth; changing growth to false afterwards
    # raises a runtime error (cannot be modified); setting true again is accepted.
    gpus = config.list_physical_devices('GPU')
    self.assertNotEqual(len(gpus), 0)

    self.assertIsNone(config.get_memory_growth(gpus[-1]))
    for gpu in gpus:
      config.set_memory_growth(gpu, True)

    c = context.context().config
    self.assertTrue(c.gpu_options.allow_growth)

    logical_gpus = config.list_logical_devices('GPU')
    self.assertTrue(len(logical_gpus), len(gpus))

    # Modifying the GPU configuration is not supported
    with self.assertRaisesRegex(RuntimeError, 'cannot be modified'):
      for gpu in gpus:
        config.set_memory_growth(gpu, False)

    # Setting the same GPU configuration is fine
    for gpu in gpus:
      config.set_memory_growth(gpu, True)

  @test_util.run_gpu_or_tpu
  @reset_eager
  def testGetMemoryInfoBasic(self, device_type):
    # [FUNCTION DOC]
    # GPU-or-TPU only; device_type is supplied by the decorator (device kind name). Scenario:
    # allocate a scalar on device zero and query memory info using its backing device name.
    # Expected: current and peak usage are positive, the result has exactly the keys current and
    # peak, and the memory usage query returns the same value as the current figure.
    with ops.device(f'{device_type}:0'):
      device = array_ops.zeros([]).backing_device
    info = config.get_memory_info(device)
    self.assertGreater(info['current'], 0)
    self.assertGreater(info['peak'], 0)
    self.assertEqual(info.keys(), {'current', 'peak'})
    self.assertEqual(config.get_memory_usage(device), info['current'])

  @test_util.run_gpu_or_tpu
  @reset_eager
  def testGetMemoryUsageSubstring(self, device_type):
    # [FUNCTION DOC]
    # GPU-or-TPU only; device_type is supplied by the decorator. Scenario: query memory info with
    # just the short device name (type and index) instead of a full name. Expected: the current
    # usage is positive, showing partial names are matched.
    info = config.get_memory_info(f'{device_type}:0')
    self.assertGreater(info['current'], 0)

  @reset_eager
  def testGetMemoryInfoCPU(self):
    # [FUNCTION DOC]
    # Scenario: query memory info and memory usage for the CPU. Expected: a value error saying
    # allocator statistics are not available for both. Skipped when the MKL build is enabled
    # because its CPU allocator does not raise.
    if test_util.IsMklEnabled():
      # TODO(gzmkl) work with Google team to address design issue in allocator.h
      self.skipTest('MklCPUAllocator does not throw exception. So skip test.')

    with self.assertRaisesRegex(ValueError, 'Allocator stats not available'):
      config.get_memory_info('CPU:0')
    with self.assertRaisesRegex(ValueError, 'Allocator stats not available'):
      config.get_memory_usage('CPU:0')

  @reset_eager
  def testGetMemoryInfoUnknownDevice(self):
    # [FUNCTION DOC]
    # Scenario: query memory info and memory usage for a nonexistent device type. Expected: a
    # value error saying no matching devices were found, for both queries.
    with self.assertRaisesRegex(ValueError, 'No matching devices found'):
      config.get_memory_info('unknown_device:0')
    with self.assertRaisesRegex(ValueError, 'No matching devices found'):
      config.get_memory_usage('unknown_device:0')

  @reset_eager
  def testGetMemoryInfoInvalidDeviceString(self):
    # [FUNCTION DOC]
    # Scenario: call the context memory info query with malformed device names (no index, empty
    # index, non-numeric index). Expected: each raises a value error about failing to parse the
    # device name.
    with self.assertRaisesRegex(ValueError, 'Failed parsing device name'):
      context.context().get_memory_info('GPU')
    with self.assertRaisesRegex(ValueError, 'Failed parsing device name'):
      context.context().get_memory_info('GPU:')
    with self.assertRaisesRegex(ValueError, 'Failed parsing device name'):
      context.context().get_memory_info('GPU:CPU')

  @test_util.run_gpu_or_tpu
  @reset_eager
  def testPeakMemoryUsage(self, device_type):
    # [FUNCTION DOC]
    # GPU-or-TPU only; device_type is supplied by the decorator. Scenario: allocate a million-float
    # zeros tensor, then a ones tensor, on device zero, then delete both. Expected: peak usage is
    # at least four million bytes after the first, grows by at least that after the second, and
    # does not decrease after freeing; peak is never below current.
    device = f'{device_type}:0'
    with ops.device(device):
      x1 = array_ops.zeros((1000, 1000))
    peak1 = config.get_memory_info(device)['peak']
    self.assertGreaterEqual(peak1, 4 * 1000 * 1000)
    with ops.device(device):
      x2 = array_ops.ones((1000, 1000))
    peak2 = config.get_memory_info(device)['peak']
    self.assertGreaterEqual(peak2, peak1 + 4 * 1000 * 1000)
    del x1, x2  # With CPython, causes tensor memory to be immediately freed
    peak3 = config.get_memory_info(device)['peak']
    self.assertGreaterEqual(peak3, peak2)
    self.assertGreaterEqual(peak3, config.get_memory_info(device)['current'])

  @test_util.run_gpu_or_tpu
  @reset_eager
  def testResetMemoryStats(self, device_type):
    # [FUNCTION DOC]
    # GPU-or-TPU only; device_type is supplied by the decorator. Scenario: allocate a million-float
    # tensor, reset memory stats, read info; then free the tensor, reset again and re-read.
    # Expected: first peak is at least four million bytes, at least current, and current is
    # positive; after the second reset the peak is strictly lower than the first peak.
    device = f'{device_type}:0'
    with ops.device(device):
      x = array_ops.zeros((1000, 1000), dtype=dtypes.float32)
    config.reset_memory_stats(device)
    info1 = config.get_memory_info(device)
    self.assertGreaterEqual(info1['peak'], 4 * 1000 * 1000)
    self.assertGreaterEqual(info1['peak'], info1['current'])
    self.assertGreater(info1['current'], 0)

    del x  # With CPython, causes tensor memory to be immediately freed
    config.reset_memory_stats(device)
    info2 = config.get_memory_info(device)
    self.assertLess(info2['peak'], info1['peak'])

  @reset_eager
  def testResetMemoryStatsCPU(self):
    # [FUNCTION DOC]
    # Scenario: reset memory stats on the CPU. Expected: a value error saying stats cannot be
    # reset. Skipped when the MKL build is enabled because its CPU allocator does not raise.
    if test_util.IsMklEnabled():
      # TODO(gzmkl) work with Google team to address design issue in allocator.h
      self.skipTest('MklCPUAllocator does not throw exception. So skip test.')

    with self.assertRaisesRegex(ValueError, 'Cannot reset memory stats'):
      config.reset_memory_stats('CPU:0')

  @reset_eager
  def testResetMemoryStatsUnknownDevice(self):
    # [FUNCTION DOC]
    # Scenario: reset memory stats for a nonexistent device type. Expected: a value error saying
    # no matching devices were found.
    with self.assertRaisesRegex(ValueError, 'No matching devices found'):
      config.reset_memory_stats('unknown_device:0')

  @reset_eager
  def testResetMemoryStatsInvalidDeviceString(self):
    # [FUNCTION DOC]
    # Scenario: call the context reset-memory-stats method with malformed device names (no index,
    # empty index, non-numeric index). Expected: each raises a value error about failing to parse
    # the device name.
    with self.assertRaisesRegex(ValueError, 'Failed parsing device name'):
      context.context().reset_memory_stats('GPU')
    with self.assertRaisesRegex(ValueError, 'Failed parsing device name'):
      context.context().reset_memory_stats('GPU:')
    with self.assertRaisesRegex(ValueError, 'Failed parsing device name'):
      context.context().reset_memory_stats('GPU:CPU')

  @test_util.run_gpu_only
  @reset_eager
  def testGpuInvalidConfig(self):
    # [FUNCTION DOC]
    # GPU-only. Scenario with several GPUs: enabling growth on only the first makes reading the
    # context configuration raise a value error (settings cannot differ); limiting visibility to
    # one GPU makes it succeed, with growth on for the first and off for the second; making all
    # visible again raises again. Then enabling growth on all GPUs works; splitting the last GPU
    # without memory limits raises a value error (memory limit); splitting it with limits turns
    # growth off in the configuration, and enabling growth on a GPU with virtual devices raises a
    # value error (virtual devices).
    gpus = config.list_physical_devices('GPU')
    self.assertNotEqual(len(gpus), 0)

    if len(gpus) > 1:
      # Assert if other GPUs were not configured
      config.set_memory_growth(gpus[0], True)
      with self.assertRaisesRegex(ValueError, 'cannot differ'):
        c = context.context().config

      # If we limit visibility to GPU 0, growth is fine
      config.set_visible_devices(gpus[0], 'GPU')
      c = context.context().config
      self.assertTrue(c.gpu_options.allow_growth)

      # Default setting for second GPU is False and works if we set visibility
      config.set_visible_devices(gpus[1], 'GPU')
      c = context.context().config
      self.assertFalse(c.gpu_options.allow_growth)

      # Growth now fails because all the GPUs are visible and not the same
      config.set_visible_devices(gpus, 'GPU')
      with self.assertRaisesRegex(ValueError, 'cannot differ'):
        c = context.context().config

    for gpu in gpus:
      config.set_memory_growth(gpu, True)

    c = context.context().config
    self.assertTrue(c.gpu_options.allow_growth)

    with self.assertRaisesRegex(ValueError, 'memory limit'):
      config.set_logical_device_configuration(gpus[-1], [
          context.LogicalDeviceConfiguration(),
          context.LogicalDeviceConfiguration()
      ])

    self.assertIsNone(config.get_logical_device_configuration(gpus[-1]))
    config.set_logical_device_configuration(gpus[-1], [
        context.LogicalDeviceConfiguration(memory_limit=10),
        context.LogicalDeviceConfiguration(memory_limit=10)
    ])

    c = context.context().config
    self.assertFalse(c.gpu_options.allow_growth)

    with self.assertRaisesRegex(ValueError, 'virtual devices'):
      config.set_memory_growth(gpus[-1], False)

  @test_util.run_gpu_only
  @reset_eager
  def testRemote(self):
    # [FUNCTION DOC]
    # GPU-only. Scenario: list logical GPUs (nonempty, names set), initialize, then define a
    # one-task local cluster server definition with the gRPC protocol and install it in the
    # context. Expected: logical GPUs are still listed with names afterwards. Side effect: sets the
    # server definition on the global context (the decorator resets it).
    gpus = config.list_logical_devices('GPU')
    self.assertNotEqual(len(gpus), 0)

    context.ensure_initialized()

    gpus = config.list_logical_devices('GPU')
    self.assertNotEqual(len(gpus), 0)
    for gpu in gpus:
      self.assertIsNotNone(gpu.name)

    context.ensure_initialized()

    job_name = 'test'
    cluster_def = cluster_pb2.ClusterDef()
    job_def = cluster_def.job.add()
    job_def.name = job_name
    job_def.tasks[0] = 'localhost:0'

    server_def = tensorflow_server_pb2.ServerDef(
        cluster=cluster_def, job_name=job_name, task_index=0, protocol='grpc')

    context.set_server_def(server_def)

    gpus = config.list_logical_devices('GPU')
    for gpu in gpus:
      self.assertIsNotNone(gpu.name)

  @reset_eager
  def testV1CompatibilityDummyInvisibleDeviceList(self):
    # [FUNCTION DOC]
    # Scenario for CPU-only machines (skipped if any GPU exists): clear cached physical devices and
    # set a legacy configuration whose GPU visible-device list is zero. Expected: the configuration
    # read back leaves that list untouched.
    gpus = config.list_physical_devices('GPU')
    if gpus:
      self.skipTest('Test requires no GPUs')

    # Ensure GPU options left untouched on CPU only environments
    context.context()._physical_devices = None
    context.context()._config = config_pb2.ConfigProto(
        gpu_options=config_pb2.GPUOptions(visible_device_list='0'))
    new_config = context.context().config
    self.assertEqual(new_config.gpu_options.visible_device_list, '0')

  @test_util.run_gpu_only
  @reset_eager
  def testV1Compatibility(self):
    # [FUNCTION DOC]
    # GPU-only. Scenario: inject legacy configuration protos into the context, clearing the
    # cached physical devices between cases, and read the effective configuration. Expected: an
    # empty config yields one CPU; a request for two CPUs is kept; an empty visible-device list
    # becomes all GPU indices joined by commas; a visible index equal to the GPU count raises a
    # value error (invalid visible device index); a single valid index is preserved.
    # Ensure we set 1 CPU by default
    context.context()._config = config_pb2.ConfigProto()
    new_config = context.context().config
    self.assertEqual(new_config.device_count['CPU'], 1)
    context.context()._physical_devices = None

    # Ensure CPU is split
    context.context()._config = config_pb2.ConfigProto(device_count={'CPU': 2})
    new_config = context.context().config
    self.assertEqual(new_config.device_count['CPU'], 2)
    context.context()._physical_devices = None

    # Handle empty visible device list
    context.context()._config = config_pb2.ConfigProto(
        gpu_options=config_pb2.GPUOptions(visible_device_list=''))
    gpus = config.list_physical_devices('GPU')
    gpu_count = len(gpus)
    new_config = context.context().config
    self.assertEqual(new_config.gpu_options.visible_device_list,
                     ','.join(str(i) for i in range(len(gpus))))
    context.context()._physical_devices = None

    # Handle invalid visible device list
    context.context()._config = config_pb2.ConfigProto(
        gpu_options=config_pb2.GPUOptions(visible_device_list=str(gpu_count)))
    with self.assertRaisesRegex(ValueError, 'Invalid visible device index'):
      gpus = config.list_physical_devices('GPU')
      new_config = context.context().config
    context.context()._physical_devices = None

    # Handle single visible device list
    context.context()._config = config_pb2.ConfigProto(
        gpu_options=config_pb2.GPUOptions(
            visible_device_list=str(gpu_count - 1)))
    gpus = config.list_physical_devices('GPU')
    new_config = context.context().config
    self.assertEqual(new_config.gpu_options.visible_device_list,
                     str(gpu_count - 1))
    context.context()._physical_devices = None

  def testConfigureCollectiveOps(self):
    # [FUNCTION DOC]
    # Scenario: configure collective ops with a worker leader, scoped allocator for collective
    # reduce, no NCCL, and a device filter for task one. Expected: the resulting configuration
    # holds that group leader and device filters, scoped allocator optimization turned on, and the
    # collective reduce op listed as enabled. Not wrapped to reset the context afterwards.
    context.context().configure_collective_ops(
        collective_leader='/job:worker/replica:0/task:0',
        scoped_allocator_enabled_ops=('CollectiveReduce',),
        use_nccl_communication=False,
        device_filters=['/job:worker/task:1'])
    new_config = context.context().config

    # Verify group leader
    self.assertEqual('/job:worker/replica:0/task:0',
                     new_config.experimental.collective_group_leader)

    # Verify device filters.
    self.assertEqual(['/job:worker/task:1'], new_config.device_filters)

    # Verify rewrite options.
    new_rewrite_options = new_config.graph_options.rewrite_options
    self.assertEqual(rewriter_config_pb2.RewriterConfig.ON,
                     new_rewrite_options.scoped_allocator_optimization)
    self.assertEqual(['CollectiveReduce'],
                     new_rewrite_options.scoped_allocator_opts.enable_op)

  def testDeterminism(self):
    # [FUNCTION DOC]
    # Scenario: disable then enable op determinism, checking the query after each. Expected: false
    # after disabling and true after enabling. Always disables determinism at the end to avoid
    # leaking global state. Does not test that ops are actually deterministic.
    # This does not test any ops are deterministic, because that is tested by
    # many kernel tests.
    try:
      config.disable_op_determinism()
      self.assertFalse(config.is_op_determinism_enabled())
      config.enable_op_determinism()
      self.assertTrue(config.is_op_determinism_enabled())
    finally:
      config.disable_op_determinism()


class TensorFloat32Test(test.TestCase):
  # [CLASS DOC]
  # Test case for the global TensorFloat-32 execution switch: the flag itself and its effect on
  # large matrix multiplication on supported NVIDIA GPUs. Uses a class constant for the matrix
  # dimension (large enough that the library uses TF32).

  def tearDown(self):
    # [FUNCTION DOC]
    # Fixture run after each test: calls the parent teardown, then re-enables TensorFloat-32
    # execution so that tests do not leak the global setting.
    super(TensorFloat32Test, self).tearDown()
    config.enable_tensor_float_32_execution(True)

  def test_tensor_float_32_global_variable(self):
    # [FUNCTION DOC]
    # Scenario: the flag defaults to enabled both in the config API and in the test op that reads
    # the native setting. Disable then re-enable it. Expected: both views agree at every step.
    self.assertTrue(config.tensor_float_32_execution_enabled())
    self.assertTrue(test_ops.is_tensor_float32_enabled())
    config.enable_tensor_float_32_execution(False)
    self.assertFalse(config.tensor_float_32_execution_enabled())
    self.assertFalse(test_ops.is_tensor_float32_enabled())
    config.enable_tensor_float_32_execution(True)
    self.assertTrue(config.tensor_float_32_execution_enabled())
    self.assertTrue(test_ops.is_tensor_float32_enabled())

  def _skip_if_tensor_float_32_unsupported(self):
    # [FUNCTION DOC]
    # HELPER: guard for the matrix multiplication tests below. Skips the current test unless an
    # NVIDIA GPU with compute capability of at least 8.0 is available. Returns nothing.
    if not test_util.is_gpu_available(
        cuda_only=True, min_cuda_compute_capability=(8, 0)):
      self.skipTest('TensorFloat-32 requires an NVIDIA GPU with compute '
                    'capability of at least 8.0')

  # Size of each dimension of matrices to test. cuBLAS does not use TF32 for
  # small matrices, so we must choose a large enough size to cause TF32 to be
  # used.
  DIM = 2 ** 10

  def test_tensor_float_32_enabled(self):
    # [FUNCTION DOC]
    # Skipped without a TF32-capable GPU. Scenario: multiply a large matrix filled with a value
    # slightly above one by a matrix of ones while TF32 is enabled. Expected: elements are rounded
    # to one, so every output equals the matrix dimension exactly.
    self._skip_if_tensor_float_32_unsupported()
    self.assertTrue(config.tensor_float_32_execution_enabled())

    x = array_ops.fill((self.DIM, self.DIM), 1 + 2**-12)
    y = array_ops.ones((self.DIM, self.DIM))
    out = math_ops.matmul(x, y)
    # In TensorFloat-32, each element of x is rounded to 1, so each output
    # element should be self.DIM.
    expected = array_ops.fill((self.DIM, self.DIM), float(self.DIM))
    self.assertAllEqual(out, expected)

  def test_tensor_float_32_disabled(self):
    # [FUNCTION DOC]
    # Skipped without a TF32-capable GPU. Scenario: disable TF32, then multiply a large matrix
    # filled with a value slightly above one by a matrix of ones. Expected: results are close to
    # the dimension times that value, within a tight relative tolerance, i.e. no TF32 rounding.
    self._skip_if_tensor_float_32_unsupported()
    self.assertTrue(config.tensor_float_32_execution_enabled())
    config.enable_tensor_float_32_execution(False)
    self.assertFalse(config.tensor_float_32_execution_enabled())

    x = array_ops.fill((self.DIM, self.DIM), 1 + 2**-12)
    y = array_ops.ones((self.DIM, self.DIM))
    out = math_ops.matmul(x, y)
    expected = array_ops.fill((self.DIM, self.DIM), self.DIM * (1 + 2**-12))
    self.assertAllClose(out, expected, rtol=2**-13, atol=0)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
