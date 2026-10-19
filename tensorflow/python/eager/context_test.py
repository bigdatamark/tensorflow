# [MODULE DOC]
# Tests for the eager context object: global seed setting, context lifetime relative to tensors,
# graph collection, function definition lookup, memory info queries, function name listing,
# logical CPU device configuration, and compiler IR retrieval from XLA-compiled functions.
# Defines a single test class. Running as a script enables eager execution and runs the test
# main. No import-time side effects.

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

import weakref

from absl.testing import parameterized
import numpy as np

from xla.service import hlo_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class ContextTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Test case for the eager context class and its module-level helpers, with parameterized test
  # support. Most tests build their own fresh context or use the global one; no shared fixtures.

  def testSetGlobalSeed(self):
    # [FUNCTION DOC]
    # Creates a fresh context and sets its global seed using a plain integer and then, for each
    # of four integer widths, using a numpy scalar, a numpy zero-dimensional array and a tensor.
    # Asserts nothing beyond all those seed forms being accepted without error.
    c = context.Context()
    c._set_global_seed(123)
    for t in [np.int32, np.int64, np.uint32, np.uint64]:
      c._set_global_seed(t(123))
      c._set_global_seed(np.array(123, dtype=t))
      c._set_global_seed(ops.convert_to_tensor(123, dtype=t))

  def testContextIsDestroyedAfterTensors(self):
    # [FUNCTION DOC]
    # Creates and initializes a new context while keeping only a weak reference, temporarily makes
    # it the global context to create a two-dimensional constant and an op output tensor (restoring
    # the original context even on error), then drops references one at a time. Asserts the
    # context stays alive after its own reference is deleted and after the first tensor is
    # deleted, and is destroyed only once the last tensor is deleted.
    # Create a new context
    new_context = context.Context()
    weak_c = weakref.ref(new_context)
    new_context.ensure_initialized()

    # Create a tensor with the new context as default.
    # Make sure to restore the original context.
    original_context = context.context()
    try:
      context._set_context(new_context)
      # Use a 2D tensor so that it is not cached.
      tensor1 = constant_op.constant([[3.]])
      # Produce a tensor as an operation output. This uses a different code path
      # from tensors created from Python.
      tensor2 = tensor1 * tensor1
      context._set_context(original_context)
    except:
      context._set_context(original_context)
      raise

    # Deleting our context reference should not delete the underlying object.
    del new_context
    self.assertIsNot(weak_c(), None)

    # Deleting the first tensor should not delete the context since there is
    # another tensor.
    del tensor1
    self.assertIsNot(weak_c(), None)

    # Deleting the last tensor should result in deleting its context.
    del tensor2
    self.assertIs(weak_c(), None)

  def testSimpleGraphCollection(self):
    # [FUNCTION DOC]
    # Defines a traced function that adds one under a CPU scope, runs it inside the graph
    # collection scope with a CPU constant, and asserts exactly one graph was collected and its
    # second node is placed on CPU zero.

    @def_function.function
    def f(x):
      # [FUNCTION DOC]
      # Traced function: under a CPU scope returns the input plus the float one.
      with ops.device('CPU:0'):
        return x + constant_op.constant(1.)

    with context.collect_graphs() as graphs:
      with ops.device('CPU:0'):
        x = constant_op.constant(1.)
      f(x)

    self.assertLen(graphs, 1)
    graph, = graphs
    self.assertIn('CPU:0', graph.node[1].device)

  @test_util.disable_tfrt(
      'b/171600738: tfrt does not support exporting post-optimization graph')
  def testGraphCollectionAfterDevicePlacement(self):
    # [FUNCTION DOC]
    # Defines a traced function adding one to its input, runs it under a CPU scope inside the
    # graph collection scope, and asserts exactly one graph was collected and its first node is
    # placed on CPU zero (the collected graph is the post-placement graph). Skipped under the
    # alternative runtime, which cannot export the optimized graph.

    @def_function.function
    def f(x):
      # [FUNCTION DOC]
      # Traced function: returns the input plus the float one.
      return x + constant_op.constant(1.)

    with context.collect_graphs() as graphs:
      with ops.device('CPU:0'):
        f(constant_op.constant(1.))

    self.assertLen(graphs, 1)
    graph, = graphs
    self.assertIn('CPU:0', graph.node[0].device)

  def testGetFunctionDef(self):
    # [FUNCTION DOC]
    # Traces a function returning a constant, fetches its function definition from the context by
    # the concrete function name, asserts it exists and contains a constant node, and asserts that
    # looking up an unknown function name raises a not-found error.

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Traced function: returns the float constant one. No inputs.
      return constant_op.constant(1.)

    concrete = f.get_concrete_function()
    function_def = context.get_function_def(concrete.name)

    self.assertIsNot(function_def, None)

    found_const_node = False
    for node_def in function_def.node_def:
      if node_def.op == 'Const':
        found_const_node = True
        break
    self.assertTrue(found_const_node)

    with self.assertRaises(errors.NotFoundError):
      _ = context.get_function_def('this_should_not_be_found')

  @test_util.run_gpu_only
  @test_util.disable_tfrt('b/169293680: TFE_GetTotalMemoryUsage is unsupported')
  def testGetMemoryInfo(self):
    # [FUNCTION DOC]
    # GPU-only test, skipped under the alternative runtime. Allocates a small tensor then asserts
    # the first GPU's memory info reports a current usage greater than zero.
    array_ops.zeros([10])  # Allocate some memory on the GPU.
    self.assertGreater(context.context().get_memory_info('GPU:0')['current'], 0)

  @test_util.disable_tfrt('b/169293680: TFE_GetTotalMemoryUsage is unsupported')
  def testGetMemoryInfoCPU(self):
    # [FUNCTION DOC]
    # Skipped when the MKL CPU allocator is enabled (it does not raise). Otherwise asserts that
    # asking for memory info of CPU zero raises a value error saying allocator stats are not
    # available. Also skipped under the alternative runtime.
    if test_util.IsMklEnabled():
      # TODO(gzmkl) work with Google team to address design issue in allocator.h
      self.skipTest('MklCPUAllocator does not throw exception. So skip test.')

    with self.assertRaisesRegex(ValueError, 'Allocator stats not available'):
      context.context().get_memory_info('CPU:0')

  @test_util.disable_tfrt('b/169293680: TFE_GetTotalMemoryUsage is unsupported')
  def testGetMemoryInfoUnknownDevice(self):
    # [FUNCTION DOC]
    # Asserts that memory info for a device type that does not exist raises a value error saying
    # no matching devices were found.
    with self.assertRaisesRegex(ValueError, 'No matching devices found'):
      context.context().get_memory_info('unknown_device:0')

  @test_util.disable_tfrt('b/169293680: TFE_GetTotalMemoryUsage is unsupported')
  def testGetMemoryInfoUnparsableDevice(self):
    # [FUNCTION DOC]
    # Asserts that memory info requests with malformed device names (no index, empty index,
    # non-numeric index) each raise a value error saying parsing of the device name failed.
    with self.assertRaisesRegex(ValueError, 'Failed parsing device name'):
      context.context().get_memory_info('GPU')
    with self.assertRaisesRegex(ValueError, 'Failed parsing device name'):
      context.context().get_memory_info('GPU:')
    with self.assertRaisesRegex(ValueError, 'Failed parsing device name'):
      context.context().get_memory_info('GPU:CPU')

  def testListFunctionNames(self):
    # [FUNCTION DOC]
    # Traces a function returning a constant and asserts its decoded concrete function name
    # appears in the global context's list of registered function names.

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Traced function: returns the float constant one. No inputs.
      return constant_op.constant(1.)

    concrete = f.get_concrete_function()
    self.assertIn(concrete.name.decode(),
                  context.context().list_function_names())

  def testSetLogicalDeviceAfterContextInitialization(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the name says "after context initialization" but the fresh context is not
    # explicitly initialized here, and the length check uses an identity comparison on integers
    # instead of equality (works only through small-integer caching).
    # Scenario: sets four logical CPU devices on a new context, asserts four are listed, then
    # asserts a second attempt to set eight raises a runtime error saying virtual CPUs are
    # already set.
    ctx = context.Context()
    ctx.set_logical_cpu_devices(4)
    self.assertIs(len(ctx.list_logical_devices('CPU')), 4)

    # Cannot set logical device twice.
    with self.assertRaisesRegex(RuntimeError, 'Virtual CPUs already set'):
      ctx.set_logical_cpu_devices(8)

  def testSetLogicalLocalCpuDevice(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the length checks use an identity comparison on integers rather than equality.
    # Scenario: replaces a context's logical device list with a single remote CPU device (not
    # local) and asserts one CPU is listed. Setting four logical CPUs must then succeed because
    # the existing device is not local, and the list is overwritten, giving four CPUs.
    ctx = context.Context()

    # Manually add a remote CPU device into logical device list.
    ctx._logical_devices = []  # pylint: disable=protected-access
    dev = context.LogicalDevice(name='/job:worker/replica:0/task:1',
                                device_type='CPU')
    ctx._logical_devices.append(dev)  # pylint: disable=protected-access
    self.assertIs(len(ctx.list_logical_devices('CPU')), 1)

    # This would pass the check since the previously added device is not local.
    ctx.set_logical_cpu_devices(4)
    # Logical device list would be overwritten after initialization.
    self.assertIs(len(ctx.list_logical_devices('CPU')), 4)

  @parameterized.named_parameters([(f'_{stage}', stage) for stage in [
      'hlo', 'hlo_serialized', 'optimized_hlo', 'optimized_hlo_serialized',
      'optimized_hlo_proto_serialized', 'optimized_hlo_dot'
  ]])
  def testGetCompilerIr(self, stage):
    # [FUNCTION DOC]
    # Parameterized over six stages: unoptimized and optimized HLO, their serialized forms, the
    # optimized serialized proto form and the optimized dot graph. Defines a jit-compiled function
    # doubling its input, calls it with a one-million element float tensor, and requests the
    # compiler IR for the stage; asserts it is non-empty. For the proto stage, parses the result
    # as an HLO proto, sums the buffer allocation sizes and asserts the total is at least twice and
    # less than four times the four-million-byte tensor size (input and output buffers).

    @def_function.function(jit_compile=True)
    def test_func(x):
      # [FUNCTION DOC]
      # Jit-compiled traced function: returns its input multiplied by two.
      return 2 * x

    a = array_ops.ones((1000, 1000))  # 4 * 1000 * 1000 in bytes
    result = test_func.experimental_get_compiler_ir(a)(stage=stage)
    self.assertNotEmpty(result)
    if stage == 'optimized_hlo_proto_serialized':
      hlo_proto = hlo_pb2.HloProto.FromString(result)
      allocations = hlo_proto.buffer_assignment.buffer_allocations
      buffer_size = sum(
          getattr(allocation, 'size') for allocation in allocations)
      # The sizes of input and output are both 4 * 1000 * 1000 in bytes.
      self.assertGreaterEqual(buffer_size, 2 * 4 * 1000 * 1000)
      self.assertLess(buffer_size, 4 * 4 * 1000 * 1000)


if __name__ == '__main__':
  ops.enable_eager_execution()
  test.main()
