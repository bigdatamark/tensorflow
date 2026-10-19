# [MODULE DOC]
# Unit tests for the parallel device (ParallelDevice): a virtual device running each eager
# operation on several component devices. Provides test helpers (collective reduction helpers
# with a unique-key counter, a lazily built dense layer) and a base test case that configures two
# logical devices (TPU, GPU or CPU, whichever exists) and builds a parallel device over them.
# Test classes cover registration, packing/unpacking, collectives, checkpointing and saved
# models (some skipped), variables, and layer training. Import-time state: a global lock and
# counter used to generate collective keys. Run as a script it enables eager execution first.
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

import os
import threading

from absl.testing import parameterized
from tensorflow.python.checkpoint import checkpoint as tracking
from tensorflow.python.checkpoint import checkpoint_management
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.distribute.parallel_device import parallel_device
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.module import module
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import collective_ops
from tensorflow.python.ops import control_flow_switch_case
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateful_random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import save
from tensorflow.python.util import nest

# When running collectives asynchronously, we need to give each parallel device
# execution a unique ID so the collectives don't interfere. Since the op is
# replicated with group/instance key intact, the replicated nodes will
# communicate.
# TODO(allenl): Switch to using a collective manager.
_COUNTER_LOCK = threading.Lock()
_COUNTER = 100


def _collective_reduce(inputs, operation, num_replicas):
  # [FUNCTION DOC]
  # HELPER: test utility that all-reduces every tensor in a nested structure across replicas.
  # Inputs: inputs (tensor or nested structure), operation (name of the merge operation, e.g.
  # an addition), num_replicas (group size). Each tensor gets a fresh unique group and instance
  # key from the module counter so concurrent collectives do not interfere. Returns a structure
  # of the same shape holding the reduced tensors.

  def _reduce_tensor(tensor):
    # [FUNCTION DOC]
    # HELPER: reduces a single tensor. Under the module lock reads and increments the global
    # counter to obtain a key (used as both group and instance key), then calls the collective
    # reduction with the enclosing group size and merge operation. Returns the reduced tensor.
    with _COUNTER_LOCK:
      global _COUNTER
      keys = _COUNTER
      _COUNTER += 1
    return collective_ops.all_reduce_v2(
        t=tensor,
        group_size=num_replicas,
        merge_op=operation,
        group_key=keys,
        instance_key=keys)

  return nest.map_structure(_reduce_tensor, inputs)


def _collective_sum(inputs, num_replicas):
  # [FUNCTION DOC]
  # HELPER: test utility that sums a nested structure of tensors across replicas by calling the
  # collective reduce helper with the addition operation. Returns the structure of sums.
  return _collective_reduce(
      inputs=inputs, operation="Add", num_replicas=num_replicas)


class _Dense(module.Module):
  # [CLASS DOC]
  # HELPER: minimal dense (fully connected) layer used by the layer tests. Variables are created
  # lazily on first call, with all-ones kernel and bias, so they are created under whatever
  # device scope is active. Holds output size, kernel and bias (the latter two start unset).

  def __init__(self, output_size):
    # [FUNCTION DOC]
    # Stores the number of output units and leaves kernel and bias unset until first call.
    self.output_size = output_size
    self.kernel = None
    self.bias = None

  def __call__(self, x):
    # [FUNCTION DOC]
    # Applies the layer to input x (a matrix of batch by features). On the first call creates the
    # kernel (all ones, shape output size by input feature count) and bias (all ones) variables.
    # Returns the matrix product of x with the transposed kernel plus the bias.
    if self.kernel is None:
      self.kernel = variables.Variable(
          array_ops.ones(
              array_ops_stack.stack([self.output_size,
                                     array_ops.shape(x)[-1]])))
      self.bias = variables.Variable(array_ops.ones([self.output_size]))
    return math_ops.matmul(x, self.kernel, transpose_b=True) + self.bias


class _VirtualDeviceTestCase(test.TestCase):
  # [CLASS DOC]
  # HELPER: base test case for parallel device tests. Its setUp picks the available accelerator
  # type (TPU, else GPU, else CPU), splits one physical device into two logical devices and
  # exposes a parallel device over devices zero and one as the device attribute, plus the
  # device type string. Subclasses hold the actual tests.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: detects TPU (initializes the TPU system), else GPU (two logical devices with a
    # 100 memory limit each), else CPU (two logical devices). Stores the device type, creates the
    # parallel device over the first two logical devices and asserts the component names contain
    # the expected type and indices. Side effects: modifies the global eager context's logical
    # device configuration.
    super(_VirtualDeviceTestCase, self).setUp()
    ctx = context.context()
    if ctx.list_physical_devices("TPU"):
      self.device_type = "TPU"
      tpu_cluster_resolver.initialize_tpu_system()
    elif ctx.list_physical_devices("GPU"):
      self.device_type = "GPU"
      gpus = ctx.list_physical_devices(self.device_type)
      ctx.set_logical_device_configuration(gpus[0], [
          context.LogicalDeviceConfiguration(memory_limit=100),
          context.LogicalDeviceConfiguration(memory_limit=100),
      ])
    else:
      self.device_type = "CPU"
      cpus = ctx.list_physical_devices("CPU")
      ctx.set_logical_device_configuration(cpus[0], [
          context.LogicalDeviceConfiguration(),
          context.LogicalDeviceConfiguration(),
      ])

    self.device = parallel_device.ParallelDevice(components=[
        "/job:localhost/device:{}:0".format(self.device_type),
        self.device_type + ":1"
    ])
    self.assertIn(self.device_type + ":0", self.device.components[0])
    self.assertIn(self.device_type + ":1", self.device.components[1])


class ParallelDeviceTests(_VirtualDeviceTestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests for the core parallel device behavior: registration, implicit copy errors, control flow,
  # string forms, device ids, generators, collectives (sync, async, in functions), checkpointing
  # and saved models (skipped), packing/unpacking of tensors, structures, composites and variables,
  # capturing, gradients, variable creation in functions and mismatched shapes. Uses the shared
  # parallel device fixture from the base class.

  def test_register_parallel_device(self):
    # [FUNCTION DOC]
    # Runs an addition of two constants under the parallel device, unpacks the result and expects
    # three on both components, each backed by the matching component device.
    with self.device:
      c = constant_op.constant(1.)
      d = constant_op.constant(2.)
      e = c + d
      outputs = self.device.unpack(e)
    self.assertAllClose([3., 3.], outputs)

    self.assertIn(self.device.components[0], outputs[0].backing_device)
    self.assertIn(self.device.components[1], outputs[1].backing_device)

  def test_no_implicit_copyon(self):
    # [FUNCTION DOC]
    # Adds two ordinary (non-parallel) tensors while the parallel device is active and expects an
    # invalid-argument error telling the user to first pack non-parallel tensors for each device.
    a1 = constant_op.constant(1.)
    a2 = constant_op.constant(2.)

    with self.device:
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          "First pack non-parallel tensors for each device"):
        a1 + a2  # pylint:disable=pointless-statement

  def test_error_message_length(self):
    # [FUNCTION DOC]
    # Passes a large (six-dimensional) ordinary tensor into an identity operation on the parallel
    # device and expects an invalid-argument error whose tensor-handle text is truncated (at most
    # about 150 characters followed by an ellipsis before the shape).
    x = array_ops.ones([3, 3, 3, 3, 3, 3])

    with self.device:
      with self.assertRaisesRegex(
          errors.InvalidArgumentError,
          r"TensorHandle\((.|\n){1,150}\[...\], shape="):
        array_ops.identity(x)

  def test_one_replica_eager_control_flow(self):
    # [FUNCTION DOC]
    # Creates a parallel device with a single component, packs a vector into it and uses a
    # reduced any-equals-four comparison as an eager if condition under the device. Expects the
    # true branch to run, so the unpacked result is the single-element list containing one.
    device = parallel_device.ParallelDevice(components=[
        "/job:localhost/device:{}:0".format(self.device_type),
    ])
    x = constant_op.constant([2, 3, 4])
    with device:
      x = device.pack([x])
      if math_ops.reduce_any(math_ops.equal(x, constant_op.constant(4))):
        y = constant_op.constant(1)
      else:
        y = constant_op.constant(2)
    self.assertAllEqual([1], device.unpack(y))

  @parameterized.named_parameters(
      ("variable", variables.Variable),
      ("tensor", lambda x: x))
  def test_string_representation(self, transform):
    # [FUNCTION DOC]
    # Parameterized by a transform (identity on the tensor, or conversion into a variable). Packs
    # two vectors, applies the transform under the parallel device and checks both the string and
    # the representation text contain the values (5 and 7) and both component device names.
    x = self.device.pack(
        [constant_op.constant([5., 6.]),
         constant_op.constant([6., 7.])])
    with self.device:
      x = transform(x)
    parallel_str = str(x)
    self.assertIn("5", parallel_str)
    self.assertIn("7", parallel_str)
    self.assertIn(self.device_type + ":0", parallel_str)
    self.assertIn(self.device_type + ":1", parallel_str)
    parallel_repr = repr(x)
    self.assertIn("5", parallel_repr)
    self.assertIn("7", parallel_repr)
    self.assertIn(self.device_type + ":0", parallel_repr)
    self.assertIn(self.device_type + ":1", parallel_repr)

  def test_device_id(self):
    # [FUNCTION DOC]
    # Unpacks the device ids property and expects the values zero and one, each living on the
    # corresponding component device.
    device_ids = self.device.unpack(self.device.device_ids)
    self.assertAllClose([0, 1], device_ids)
    # TODO(allenl): Should device IDs be int64 so they can be placed on GPUs?
    # Currently backing_device is CPU.
    self.assertIn(self.device.components[0], device_ids[0].device)
    self.assertIn(self.device.components[1], device_ids[1].device)

  def test_zeros(self):
    # [FUNCTION DOC]
    # Creates a zeros tensor under the parallel device whose size comes from a runtime value (ten)
    # and expects every unpacked component to be ten zeros.
    with self.device:
      x = array_ops.zeros([array_ops.identity(constant_op.constant(10))])
    for component in self.device.unpack(x):
      self.assertAllClose([0.] * 10, component)

  def test_generator(self):
    # [FUNCTION DOC]
    # Under the parallel device creates two stateful random generators: one seeded with zero on
    # all devices and one seeded with the per-device ids. Draws ten normal values from each and
    # expects the same-seed samples to match across components while the different-seed samples
    # differ from the first component.
    with self.device:
      g_same = stateful_random_ops.Generator.from_seed(0)
      g_different = stateful_random_ops.Generator.from_seed(
          self.device.device_ids)
      same = g_same.normal([10])
      different = g_different.normal([10])
    same_unpacked = self.device.unpack(same)
    different_unpacked = self.device.unpack(different)
    for same_component, different_component in zip(same_unpacked[1:],
                                                   different_unpacked[1:]):
      self.assertAllClose(same_component, same_unpacked[0])
      self.assertNotAllClose(different_component, different_unpacked[0])

  def test_collective_reduce(self):
    # [FUNCTION DOC]
    # Packs minus one and a half and three and a half, sums them with a collective across two
    # replicas under the device and expects two on each component, on the correct devices.
    x = self.device.pack(
        [constant_op.constant(-1.5),
         constant_op.constant(3.5)])
    with self.device:
      reduced = _collective_sum(x, num_replicas=2)
      outputs = self.device.unpack(reduced)
    self.assertAllClose([2., 2.], outputs)
    self.assertIn(self.device.components[0], outputs[0].backing_device)
    self.assertIn(self.device.components[1], outputs[1].backing_device)

  def test_collective_reduce_in_function(self):
    # [FUNCTION DOC]
    # Same as the basic collective sum test but the reduction runs inside a compiled function
    # called under the parallel device. Expects two on each component on the correct devices.
    x = self.device.pack(
        [constant_op.constant(-1.5),
         constant_op.constant(3.5)])
    with self.device:

      @def_function.function
      def reduce(t):
        # [FUNCTION DOC]
        # Compiled function that returns the two-replica collective sum of its input.
        return _collective_sum(t, num_replicas=2)

      reduced = reduce(x)
      outputs = self.device.unpack(reduced)
    self.assertAllClose([2., 2.], outputs)
    self.assertIn(self.device.components[0], outputs[0].backing_device)
    self.assertIn(self.device.components[1], outputs[1].backing_device)

  def test_collective_reduce_async_scope(self):
    # [FUNCTION DOC]
    # Runs the collective sum inside an asynchronous execution scope together with the parallel
    # device. The main goal is to avoid deadlocks; also expects two on each component on the
    # correct devices.
    # Note that ops on the parallel device currently don't execute
    # asynchronously. The test is just that we don't get deadlocks.
    x = self.device.pack(
        [constant_op.constant(-1.5),
         constant_op.constant(3.5)])
    with context.async_scope(), self.device:
      reduced = _collective_sum(x, num_replicas=2)
      outputs = self.device.unpack(reduced)
    self.assertAllClose([2., 2.], outputs)
    self.assertIn(self.device.components[0], outputs[0].backing_device)
    self.assertIn(self.device.components[1], outputs[1].backing_device)

  def test_collective_reduce_async_context(self):
    # [FUNCTION DOC]
    # Saves the current synchronous-execution setting, resets the eager context, turns
    # synchronous execution off and re-runs the fixture setup, then performs the collective sum
    # and expects two on each component. Always (in a finally block) resets the context again and
    # restores the original setting. Goal: no deadlocks with asynchronous execution.
    previous = config.get_synchronous_execution()
    try:
      context._reset_context()
      config.set_synchronous_execution(False)
      self.setUp()
      # Note that ops on the parallel device currently don't execute
      # asynchronously. The test is just that we don't get deadlocks.
      x = self.device.pack(
          [constant_op.constant(-1.5),
           constant_op.constant(3.5)])
      with self.device:
        reduced = _collective_sum(x, num_replicas=2)
        outputs = self.device.unpack(reduced)
      self.assertAllClose([2., 2.], outputs)
      self.assertIn(self.device.components[0], outputs[0].backing_device)
      self.assertIn(self.device.components[1], outputs[1].backing_device)
    finally:
      context._reset_context()
      config.set_synchronous_execution(previous)

  def test_collective_broadcast_in_function(self):
    # [FUNCTION DOC]
    # Skipped on TPU. Defines a compiled function that, depending on the device id, either sends
    # a broadcast of three times a constant two (device zero) or receives it (device one) through
    # a switch-case. Runs it under the parallel device with the device ids and expects the
    # unpacked results to be two on the first device and six on the second.
    if self.device_type == "TPU":
      self.skipTest("ParallelDevice broadcast collectives on TPUs need work")

    @def_function.function
    def broadcast_send_recv(device_id):
      # [FUNCTION DOC]
      # HELPER: compiled function choosing send or receive branch by device id (zero: send,
      # one: receive) and returning the branch result.
      c = constant_op.constant([2])

      @def_function.function
      def send():
        # [FUNCTION DOC]
        # HELPER: broadcast sender branch. Sends three times the constant over a two-member
        # group and, after the send finishes, returns the constant unchanged.
        s0 = collective_ops.broadcast_send(
            c * 3, c.shape, c.dtype, group_size=2, group_key=1, instance_key=1)
        with ops.control_dependencies([s0.op]):
          return array_ops.identity(c)

      @def_function.function
      def recv():
        # [FUNCTION DOC]
        # HELPER: broadcast receiver branch. Receives a value with the constant's shape and type
        # from the same group and key as the sender and returns it.
        r0 = collective_ops.broadcast_recv(
            c.shape, c.dtype, group_size=2, group_key=1, instance_key=1)
        return r0

      return control_flow_switch_case.switch_case(
          device_id, branch_fns={
              0: send,
              1: recv
          })

    with self.device:
      result = broadcast_send_recv(self.device.device_ids)
    self.assertAllClose([[2], [6]], self.device.unpack(result))

  def test_use_in_graph_error_is_informative(self):
    # [FUNCTION DOC]
    # Calls a compiled function that enters the parallel device inside its body and expects a
    # not-implemented error whose message mentions that it is unsupported inside a function.
    @def_function.function
    def uses_parallel():
      # [FUNCTION DOC]
      # HELPER: compiled function that enters the parallel device scope and unpacks a ones tensor,
      # which is expected to fail during tracing.
      with self.device:
        return self.device.unpack(array_ops.ones([]))

    with self.assertRaisesRegex(NotImplementedError, "inside `tf.function`"):
      uses_parallel()

  def test_checkpointing(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: skipped unconditionally at the start (tracking bug about revisiting parallel
    # device and checkpointing), so the rest of the body never runs.
    # Intended scenario: save a variable holding different values per device, overwrite it,
    # restore and expect the original values; restore on creation; then restore into a
    # single-device copy (the first component's value) and expect that asserting full consumption
    # fails because of the unmatched second component.
    self.skipTest("b/216201668: revisit parallel device and checkpointing.")

    prefix = os.path.join(self.get_temp_dir(), "ckpt")
    different_values = self.device.pack(
        [constant_op.constant(-1.),
         constant_op.constant(3.)])
    with self.device:
      v = variables.Variable(different_values)
      checkpoint = tracking.Checkpoint(v=v)
    save_path = checkpoint.save(prefix)
    with self.device:
      v.assign(constant_op.constant(0.))
    checkpoint.restore(save_path).assert_consumed()
    with self.device:
      outputs = self.device.unpack(v)
    self.assertAllClose([-1., 3.], outputs)

    with self.device:
      restore_on_create = tracking.Checkpoint()
      restore_on_create.restore(save_path)
      restore_on_create.v = variables.Variable(0.)
      outputs = self.device.unpack(restore_on_create.v)
    self.assertAllClose([-1., 3.], outputs)

    # Changing the number of devices / restoring into a single-device copy is OK
    single_device = tracking.Checkpoint(v=variables.Variable(0.))
    status = single_device.restore(save_path)
    status.assert_existing_objects_matched()
    self.assertAllClose(-1., single_device.v)
    with self.assertRaisesRegex(AssertionError, "parallel_component_1"):
      # There are parts of the variable that aren't restored into a
      # single-device copy.
      status.assert_consumed()

  def test_pack_composite(self):
    # [FUNCTION DOC]
    # Skipped unless running on CPU. Packs one small dataset per component into a parallel
    # dataset, creates an iterator and takes a sample under the device, unpacks the iterator and
    # sample, and checks the component iterators produce the expected next values (2 and 4) and
    # the sample components are 1 and 2.
    if self.device_type != "CPU":
      self.skipTest("Iterator GetNext doesn't work on accelerators.")
    datasets = [
        dataset_ops.Dataset.from_tensor_slices(
            [i + 1, (i + 1) * 2, (i + 1) * 3])
        for i in range(len(self.device.components))]
    parallel_dataset = self.device.pack(datasets)
    with self.device:
      iterator = iter(parallel_dataset)
      parallel_sample = next(iterator)
    component_iterators = self.device.unpack(iterator)
    self.assertEqual(2, next(component_iterators[0]).numpy())
    self.assertEqual(1, self.device.unpack(parallel_sample)[0].numpy())
    self.assertEqual(4, next(component_iterators[1]).numpy())
    self.assertEqual(2, self.device.unpack(parallel_sample)[1].numpy())

  def test_pack_structure(self):
    # [FUNCTION DOC]
    # Packs a per-device mapping (one entry holding the device index as a float), unpacks it and
    # expects the structure back with values zero and one.
    x_parts = [{"a": constant_op.constant(float(i))}
               for i in range(len(self.device.components))]
    x = self.device.pack(x_parts)
    self.assertAllClose([{"a": 0.}, {"a": 1.}], self.device.unpack(x))

  def test_pack_variable_value(self):
    # [FUNCTION DOC]
    # Packs variables (once outside and once under the device), then assigns a different value to
    # each variable. Expects both packed results to still hold the earlier values zero and one,
    # showing packing reads values rather than sharing the variable.
    x_parts = [variables.Variable(i)
               for i in range(len(self.device.components))]
    x = self.device.pack(x_parts)
    with self.device:
      x1 = self.device.pack(x_parts)
    for v in x_parts:
      v.assign(-10)  # Mutating the variable does not affect previous reads.
    self.assertAllClose([0, 1], self.device.unpack(x))
    self.assertAllClose([0, 1], self.device.unpack(x1))

  def test_unpack_variable_value(self):
    # [FUNCTION DOC]
    # Creates a variable from a packed tensor under the device, unpacks it, then assigns a new
    # value to the variable. Expects the unpacked result to keep the original values zero and one.
    x_parts = [constant_op.constant(i)
               for i in range(len(self.device.components))]
    x = self.device.pack(x_parts)
    with self.device:
      v = variables.Variable(x)
      v_unpacked = self.device.unpack(v)
      v.assign(-10)  # Mutating the variable does not affect previous reads.
    self.assertAllClose([0, 1], v_unpacked)

  def test_saved_model(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: skipped unconditionally at the start (tracking bug about revisiting parallel
    # device and saved models), so the rest of the body never runs.
    # Intended scenario: build a module with a per-device variable and a function doubling it,
    # save it, reset the context, load it without a device (expecting a single value) and under
    # the parallel device (expecting per-device values), then assign new values and re-check.
    self.skipTest("b/216201668: revisit parallel device and saved model")

    different_values = self.device.pack(
        [constant_op.constant(-1.),
         constant_op.constant(3.)])
    with self.device:
      m = module.Module()
      m.v = variables.Variable(different_values)
      m.f = def_function.function(lambda: m.v * 2.)
      self.assertAllClose([-2., 6.], self.device.unpack(m.f()))
    saved_model_path = os.path.join(self.get_temp_dir(), "saved_model")
    save.save(m, saved_model_path)

    context._reset_context()
    self.setUp()

    single_device_loaded = load.load(saved_model_path)
    self.assertAllClose(-2., single_device_loaded.f())
    assign_value = self.device.pack(
        [constant_op.constant(.1), constant_op.constant(.2)])
    with self.device:
      parallel_loaded = load.load(saved_model_path)
      self.assertAllClose([-2., 6.], self.device.unpack(parallel_loaded.f()))
      self.assertAllClose([-1., 3.], self.device.unpack(parallel_loaded.v))
      parallel_loaded.v.assign(assign_value)
      self.assertAllClose([.2, .4], self.device.unpack(parallel_loaded.f()))

  def _assert_close_to_non_parallel(self, computation):
    # [FUNCTION DOC]
    # HELPER: shared check for tests comparing parallel and ordinary execution. Input: a
    # callable taking no arguments and returning a structure of tensors. Runs it under the
    # parallel device and again normally, asserts both results have the same structure and a
    # non-zero number of tensors, that parallel results live on the parallel device and the
    # others do not, and that every unpacked component is close to the ordinary result.
    """Asserts that replication of `computation` works and is equivalent."""
    with self.device:
      parallel_result = computation()
    non_parallel_result = computation()
    # The computations should have the same number and structure of Tensor
    # objects, even though the tensors themselves will be on different devices
    # and represent different numbers of values.
    nest.assert_same_structure(parallel_result, non_parallel_result)
    non_parallel_flat = nest.flatten(non_parallel_result)
    parallel_flat = nest.flatten(parallel_result)
    self.assertGreater(len(parallel_flat), 0)
    for non_parallel, parallel in zip(non_parallel_flat, parallel_flat):
      self.assertEqual(self.device._name, parallel.device)
      self.assertNotEqual(self.device._name, non_parallel.device)
      for parallel_component in self.device.unpack(parallel):
        self.assertAllClose(non_parallel, parallel_component)

  def test_capturing(self):
    # [FUNCTION DOC]
    # Under the parallel device creates a tensor, then defines and calls a compiled function that
    # captures it and adds its argument (a ones vector). Expects both components to equal two and
    # three.
    with self.device:
      x = constant_op.constant([1., 2.])
      x = array_ops.identity(x)

      @def_function.function
      def f(y):
        # [FUNCTION DOC]
        # Compiled function adding its argument to the captured parallel tensor.
        return x + y

      y = array_ops.ones([2])
      parallel_result = f(y)
    self.assertAllClose([[2., 3.]] * 2, self.device.unpack(parallel_result))

  def test_euclidean_norm(self):
    # [FUNCTION DOC]
    # Checks that the Euclidean norm over rows of a ones matrix, and its gradient, computed under
    # the parallel device match ordinary execution (via the shared comparison helper).
    def _test_fn():
      # [FUNCTION DOC]
      # HELPER: computation under comparison. Watches a five by five ones matrix, takes the
      # row-wise Euclidean norm and returns it with its gradient with respect to the matrix.
      with backprop.GradientTape() as tape:
        x = array_ops.ones([5, 5])
        tape.watch(x)
        y = math_ops.reduce_euclidean_norm(x, axis=constant_op.constant(1))
      return y, tape.gradient(y, x)
    self._assert_close_to_non_parallel(_test_fn)

  def test_reduce_sum(self):
    # [FUNCTION DOC]
    # Checks that the row-wise sum of a ones matrix, and its gradient, computed under the
    # parallel device match ordinary execution (via the shared comparison helper).
    def _test_fn():
      # [FUNCTION DOC]
      # HELPER: computation under comparison. Watches a five by five ones matrix, sums each row
      # and returns the sums with their gradient with respect to the matrix.
      with backprop.GradientTape() as tape:
        x = array_ops.ones([5, 5])
        tape.watch(x)
        y = math_ops.reduce_sum(x, axis=constant_op.constant(1))
      return y, tape.gradient(y, x)
    self._assert_close_to_non_parallel(_test_fn)

  def test_variable_created_in_function(self):
    # [FUNCTION DOC]
    # Defines a module whose compiled call (autograph off) lazily creates four variables (one from
    # an initialization-scope value, one from a captured tensor, one from an initializer callable
    # and one from a constant) and multiplies the input by all of them. Under the parallel device
    # calls it with a scalar one and expects sixteen on both components.
    captured_value = constant_op.constant(2.)

    class M(module.Module):
      # [CLASS DOC]
      # Local helper module for the test. Variables are created lazily inside its compiled call.

      def __init__(self):
        # [FUNCTION DOC]
        # Initializes four variable attributes to unset.
        self.v = None
        self.w = None
        self.x = None
        self.z = None

      @def_function.function(autograph=False)
      def __call__(self, x):
        # [FUNCTION DOC]
        # On first call creates the four variables (one in the initialization scope, others from
        # the captured tensor, an initializer callable and a constant). Returns the input times
        # the product of all four variables.
        if self.v is None:
          with ops.init_scope():
            initial_value = constant_op.constant(2.)
            self.z = variables.Variable(initial_value)
          self.x = variables.Variable(captured_value)
          self.w = variables.Variable(lambda: constant_op.constant(2.))
          self.v = variables.Variable(constant_op.constant(2.))
        return x * self.v * self.w * self.x * self.z

    with self.device:
      m = M()
      packed_outputs = m(array_ops.ones([]))
      outputs = self.device.unpack(packed_outputs)
    self.assertAllClose([16., 16.], outputs)

  def test_different_shapes(self):
    # [FUNCTION DOC]
    # Packs vectors of different lengths, doubles them under the device and expects an unknown
    # static shape and the correct per-device values. Then packs tensors of different ranks and
    # expects reading the shape to raise an error saying the components do not all have the same
    # rank.
    x = self.device.pack(
        [constant_op.constant([1., 2.]),
         constant_op.constant([5.])])
    with self.device:
      y = x * 2.
    self.assertEqual([None], y.shape.as_list())
    self.assertAllClose([[2., 4.], [10.]], self.device.unpack(y))

    different_axes = self.device.pack(
        [constant_op.constant([1., 2.]),
         constant_op.constant([[5.]])])
    with self.assertRaisesRegex(Exception,
                                "components do not all have the same rank"):
      different_axes.shape  # pylint: disable=pointless-statement


class LayerTests(_VirtualDeviceTestCase):
  # [CLASS DOC]
  # Tests that a simple dense layer works under the parallel device: forward pass, synchronized
  # training using collective gradient sums, divergent training without reduction, and a skipped
  # checkpoint-based training loop.

  def test_layer_forward(self):
    # [FUNCTION DOC]
    # Applies the dense layer to a scalar-input batch under the device and expects three in every
    # output unit on both components, on the correct devices. Then packs different inputs per
    # device and expects the outputs of the two components to differ noticeably.
    with self.device:
      layer = _Dense(5)
      x = constant_op.constant([[2.]])
      y = layer(x)
      outputs = self.device.unpack(y)
    self.assertAllClose([[3.] * 5], outputs[0])
    self.assertAllClose([[3.] * 5], outputs[1])
    self.assertIn(self.device.components[0], outputs[0].backing_device)
    self.assertIn(self.device.components[1], outputs[1].backing_device)

    # With different Layer inputs we get different outputs
    x = self.device.pack(
        [constant_op.constant([[-0.5]]),
         constant_op.constant([[0.5]])])
    with self.device:
      y = layer(x)
      outputs = self.device.unpack(y)
    self.assertGreater(
        math_ops.reduce_max(math_ops.abs(outputs[0] - outputs[1])), 1e-5)
    self.assertIn(self.device.components[0], outputs[0].backing_device)
    self.assertIn(self.device.components[1], outputs[1].backing_device)

  def test_layer_sync_training(self):
    # [FUNCTION DOC]
    # Packs different inputs per device, runs one gradient step on a dense layer with gradients
    # summed across both replicas by a collective and a small learning rate. Expects both kernel
    # components to remain equal and both bias components to equal the analytically computed
    # value (within tolerance), on the correct devices.
    x = self.device.pack(
        [constant_op.constant([[-0.5]]),
         constant_op.constant([[0.5]])])
    with self.device:
      layer = _Dense(5)

      with backprop.GradientTape() as tape:
        y = layer(x)
        loss = (y - math_ops.range(5.))**2.
      parameters = layer.trainable_variables
      unreduced_gradients = tape.gradient(loss, parameters)
      reduced_gradients = _collective_sum(unreduced_gradients, num_replicas=2)
      for grad, param in zip(reduced_gradients, parameters):
        param.assign_sub(0.01 * grad)
    final_kernels = self.device.unpack(layer.kernel)
    self.assertAllClose(final_kernels[0], final_kernels[1])
    final_bias = self.device.unpack(layer.bias)
    expected_bias = (1. - 0.01 * 2. * (1. + .5 - math_ops.range(5.)) -
                     0.01 * 2. * (1. - .5 - math_ops.range(5.)))
    self.assertAllClose(expected_bias, final_bias[0], rtol=1e-4, atol=1e-4)
    self.assertAllClose(expected_bias, final_bias[1], rtol=1e-4, atol=1e-4)
    self.assertIn(self.device.components[0], final_kernels[0].backing_device)
    self.assertIn(self.device.components[1], final_kernels[1].backing_device)

  def test_layer_divergent_buffer_training(self):
    # [FUNCTION DOC]
    # Same as the synchronized training test but without reducing gradients, so each device
    # updates with its own gradient. Expects the kernel components to differ and each bias
    # component to equal its own analytically computed value, on the correct devices.
    x = self.device.pack(
        [constant_op.constant([[-0.5]]),
         constant_op.constant([[0.5]])])
    with self.device:
      layer = _Dense(5)

      with backprop.GradientTape() as tape:
        y = layer(x)
        loss = (y - math_ops.range(5.))**2.
      parameters = layer.trainable_variables
      unreduced_gradients = tape.gradient(loss, parameters)
      for grad, param in zip(unreduced_gradients, parameters):
        param.assign_sub(0.01 * grad)
    final_kernels = self.device.unpack(layer.kernel)
    self.assertNotAllClose(final_kernels[0], final_kernels[1])
    final_bias = self.device.unpack(layer.bias)
    self.assertAllClose(1. - 0.01 * 2. * (1. - .5 - math_ops.range(5.)),
                        final_bias[0])
    self.assertAllClose(1. - 0.01 * 2. * (1. + .5 - math_ops.range(5.)),
                        final_bias[1])
    self.assertIn(self.device.components[0], final_kernels[0].backing_device)
    self.assertIn(self.device.components[1], final_kernels[1].backing_device)

  def test_training_loop(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: skipped unconditionally at the start (tracking bug about revisiting parallel
    # device and checkpointing), so the rest of the body never runs.
    # Intended scenario: five times, create a layer with a checkpoint manager (restore or
    # initialize), then ten steps of synchronized training with collective gradient sums, saving a
    # checkpoint after each step.
    self.skipTest("b/216201668: revisit parallel device and checkpointing")
    for _ in range(5):
      layer = _Dense(5)
      checkpoint = tracking.Checkpoint(layer=layer)
      manager = checkpoint_management.CheckpointManager(
          checkpoint, directory=self.get_temp_dir(), max_to_keep=5)
      manager.restore_or_initialize()

      for _ in range(10):
        x = self.device.pack(
            [constant_op.constant([[-0.5]]),
             constant_op.constant([[0.5]])])
        with self.device:
          with backprop.GradientTape() as tape:
            y = layer(x)
            loss = (y - math_ops.range(5.))**2.
          parameters = layer.trainable_variables
          unreduced_gradients = tape.gradient(loss, parameters)
          reduced_gradients = _collective_sum(
              unreduced_gradients, num_replicas=len(self.device.components))
          for grad, param in zip(reduced_gradients, parameters):
            param.assign_sub(0.01 * grad)

        manager.save()


if __name__ == "__main__":
  ops.enable_eager_execution()
  test.main()
