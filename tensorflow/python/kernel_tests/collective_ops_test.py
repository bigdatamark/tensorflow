# [MODULE DOC]
# Test module for the version 1 and version 2 collective operations (all-reduce, all-gather,
# broadcast, all-to-all, group assignment) of the ops library. Tests run in eager mode on
# several logical CPU devices (or GPUs when available) in a single process, using one thread or
# function per device. Covers correctness of results, group and instance key handling, XLA
# compilation, aborting collectives, cancellation, timeouts, scheduling order and a few
# error cases. Defines two adapter classes that give the v1 and v2 ops one uniform calling
# interface, a helper to create ordering tokens, and module-level parameter combinations
# (device and communication settings; collective op choices). Import has no side effects beyond
# building those combinations; running as a script sets the NCCL debug environment variable to
# the info level, enables version 2 behavior and then runs the test main.
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for V2 Collective Operations."""

import os
import threading
import time
from absl.testing import parameterized

from tensorflow.python.compat import v2_compat
from tensorflow.python.data.experimental.ops import testing as dataset_testing
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import test_util
from tensorflow.python.eager import cancellation
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import collective_ops as _collective_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test


def create_ordering_token():
  # [FUNCTION DOC]
  # HELPER: shared by the tests to give v2 collectives an ordering token.
  # Creates a new scalar float resource variable (initial value one) on the current device
  # and returns its resource handle, used to order collective ops on that device.
  return resource_variable_ops.ResourceVariable(1.0).handle


class CollectiveOpsV1(object):
  # [CLASS DOC]
  # HELPER: adapter so tests can call the v1 collective ops through the same interface as the v2
  # adapter. A namespace of static methods (never instantiated). The all-reduce and all-gather
  # wrappers drop any ordering token argument because v1 ops do not accept one; broadcast send
  # and receive are the v1 ops directly.

  @staticmethod
  def all_reduce(t, group_size, group_key, instance_key, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: adapter around the v1 all-reduce. Removes the optional ordering token keyword
    # argument if present, then forwards the tensor, group size, group key, instance key and
    # all other positional and keyword arguments. Returns the reduced tensor.
    kwargs.pop('ordering_token', None)
    return _collective_ops.all_reduce(t, group_size, group_key, instance_key,
                                      *args, **kwargs)

  @staticmethod
  def all_gather(t, group_size, group_key, instance_key, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: adapter around the v1 all-gather. Removes the optional ordering token keyword
    # argument if present, then forwards the tensor, keys and remaining arguments. Returns the
    # gathered tensor.
    kwargs.pop('ordering_token', None)
    return _collective_ops.all_gather(t, group_size, group_key, instance_key,
                                      *args, **kwargs)

  broadcast_send = _collective_ops.broadcast_send
  broadcast_recv = _collective_ops.broadcast_recv


class CollectiveOpsV2(object):
  # [CLASS DOC]
  # HELPER: adapter so tests can call the v2 collective ops with plain Python numbers for the
  # group size, group key and instance key. A namespace of static methods (never instantiated).
  # Each wrapper first converts these keys into tensors via the identity op and then calls the
  # v2 op. Offers all-reduce, all-gather, broadcast send, broadcast receive and all-to-all.

  @staticmethod
  def all_reduce(t, group_size, group_key, instance_key, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: adapter around the v2 all-reduce. Converts group size, group key and instance key
    # to tensors, forwards them with the input tensor and the remaining arguments (such as the
    # ordering token or communication hint). Returns the reduced tensor.
    group_size = array_ops.identity(group_size)
    group_key = array_ops.identity(group_key)
    instance_key = array_ops.identity(instance_key)
    return _collective_ops.all_reduce_v2(t, group_size, group_key, instance_key,
                                         *args, **kwargs)

  @staticmethod
  def all_gather(t, group_size, group_key, instance_key, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: adapter around the v2 all-gather. Converts the three keys to tensors and forwards
    # them with the input tensor and remaining arguments. Returns the gathered tensor.
    group_size = array_ops.identity(group_size)
    group_key = array_ops.identity(group_key)
    instance_key = array_ops.identity(instance_key)
    return _collective_ops.all_gather_v2(t, group_size, group_key, instance_key,
                                         *args, **kwargs)

  @staticmethod
  def broadcast_send(t, shape, dtype, group_size, group_key, instance_key,
                     *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: adapter around the v2 broadcast send. Accepts the shape and dtype arguments only
    # to match the v1 calling interface and ignores them. Converts the three keys to tensors and
    # forwards the input tensor, keys and remaining arguments. Returns the broadcast tensor.
    group_size = array_ops.identity(group_size)
    group_key = array_ops.identity(group_key)
    instance_key = array_ops.identity(instance_key)
    return _collective_ops.broadcast_send_v2(t, group_size, group_key,
                                             instance_key, *args, **kwargs)

  @staticmethod
  def broadcast_recv(shape, dtype, group_size, group_key, instance_key, *args,
                     **kwargs):
    # [FUNCTION DOC]
    # HELPER: adapter around the v2 broadcast receive. Converts the three keys and the shape to
    # tensors, then forwards shape, dtype, keys and remaining arguments. Returns the received
    # tensor.
    group_size = array_ops.identity(group_size)
    group_key = array_ops.identity(group_key)
    instance_key = array_ops.identity(instance_key)
    shape = array_ops.identity(shape)
    return _collective_ops.broadcast_recv_v2(shape, dtype, group_size,
                                             group_key, instance_key, *args,
                                             **kwargs)

  @staticmethod
  def all_to_all(t, group_size, group_key, instance_key, *args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: adapter around the v2 all-to-all. Converts the three keys to tensors, forwards them
    # with the input tensor and remaining arguments. Returns the exchanged tensor.
    group_size = array_ops.identity(group_size)
    group_key = array_ops.identity(group_key)
    instance_key = array_ops.identity(instance_key)
    return _collective_ops.all_to_all_v2(
        t, group_size, group_key, instance_key, *args, **kwargs
    )


device_combination = (
    combinations.combine(device='CPU', communication='RING', required_gpus=0) +
    combinations.combine(
        device='GPU', communication=['RING', 'NCCL'], required_gpus=2))

collective_op_combinations = combinations.combine(collective_op=[
    combinations.NamedObject('all_reduce', CollectiveOpsV1.all_reduce),
    combinations.NamedObject('all_gather', CollectiveOpsV1.all_gather),
    combinations.NamedObject('all_reduce_v2', CollectiveOpsV2.all_reduce),
    combinations.NamedObject('all_gather_v2', CollectiveOpsV2.all_gather)
])


@combinations.generate(
    combinations.times(
        combinations.combine(
            collective_ops=[
                combinations.NamedObject('v1', CollectiveOpsV1),
                combinations.NamedObject('v2', CollectiveOpsV2)
            ],
            mode='eager'), device_combination))
class CollectiveOpsTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized eager-mode tests of basic collectives (reduce, gather, broadcast, all-to-all,
  # key scoping, group size one, invalid keys, multiple groups) for both the v1 and v2 adapters,
  # over each device and communication combination. Uses sixteen logical devices per test.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: resets the global eager context and sets up sixteen logical CPU devices (and
    # optionally GPUs) via the module setup helper, then runs the base class setup.
    _setup_context(num_devices=16)
    super().setUp()

  def testReduce(self, collective_ops, device, communication):
    # [FUNCTION DOC]
    # Scenario: all-reduce within compiled functions, first on a group of one device (result
    # equals the input, one) and then on a group of two devices (each gets the sum, two), with
    # ordering tokens per device and the parameterized communication hint. Asserts the values
    # are close.
    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device

    tokens = {}
    for dev in [dev0, dev1]:
      with ops.device(dev):
        tokens[dev] = create_ordering_token()

    @def_function.function
    def run_all_reduce_1device():
      # [FUNCTION DOC]
      # Compiled function: all-reduce of the value one on device zero with a group of one, using
      # that device's ordering token. Returns the reduced tensor (equal to the input).
      with ops.device(dev0):
        in_value = constant_op.constant([1.])
        group_size = 1
        group_key = 1
        instance_key = 1
        return collective_ops.all_reduce(
            in_value,
            group_size,
            group_key,
            instance_key,
            communication_hint=communication,
            ordering_token=tokens[dev0])

    @def_function.function
    def run_all_reduce_2devices():
      # [FUNCTION DOC]
      # Compiled function: all-reduce of the value one across a group of two devices (device
      # zero and one) sharing one group key and instance key, each with its own ordering token.
      # Returns the list of two results, each expected to equal two.
      in_value = constant_op.constant([1.])
      group_size = 2
      group_key = 2
      instance_key = 2
      collectives = []
      with ops.device(dev0):
        collectives.append(
            collective_ops.all_reduce(
                in_value,
                group_size,
                group_key,
                instance_key,
                ordering_token=tokens[dev0],
                communication_hint=communication))
      with ops.device(dev1):
        collectives.append(
            collective_ops.all_reduce(
                in_value,
                group_size,
                group_key,
                instance_key,
                ordering_token=tokens[dev1],
                communication_hint=communication))
      return collectives

    self.assertAllClose(run_all_reduce_1device(), [1.], rtol=1e-5, atol=1e-5)
    for result in run_all_reduce_2devices():
      self.assertAllClose(result, [2.], rtol=1e-5, atol=1e-5)

  def testGather(self, collective_ops, device, communication):
    # [FUNCTION DOC]
    # Scenario: all-gather in compiled functions over groups of one, two and sixteen devices
    # (the sixteen use logical CPU devices with their own tokens). Asserts the single value is
    # unchanged, the two-device result is the pair of ones, and the sixteen-device result is the
    # numbers zero through fifteen concatenated.
    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device

    tokens = {}
    for dev in [dev0, dev1]:
      with ops.device(dev):
        tokens[dev] = create_ordering_token()

    @def_function.function
    def run_all_gather_1device():
      # [FUNCTION DOC]
      # Compiled function: all-gather of one value on device zero with a group of one. Returns
      # the gathered tensor (equal to the input).
      with ops.device(dev0):
        in_value = constant_op.constant([1.])
        group_size = 1
        group_key = 1
        instance_key = 1
        return collective_ops.all_gather(
            in_value,
            group_size,
            group_key,
            instance_key,
            ordering_token=tokens[dev0],
            communication_hint=communication)

    @def_function.function
    def run_all_gather_2devices():
      # [FUNCTION DOC]
      # Compiled function: all-gather of the value one across devices zero and one in a group of
      # two. Returns a list of two results, each the pair of ones.
      in_value = constant_op.constant([1.])
      group_size = 2
      group_key = 2
      instance_key = 2
      collectives = []
      with ops.device(dev0):
        collectives.append(
            collective_ops.all_gather(
                in_value,
                group_size,
                group_key,
                instance_key,
                ordering_token=tokens[dev0],
                communication_hint=communication))
      with ops.device(dev1):
        collectives.append(
            collective_ops.all_gather(
                in_value,
                group_size,
                group_key,
                instance_key,
                ordering_token=tokens[dev1],
                communication_hint=communication))
      return collectives

    cpu_tokens = {}
    for i in range(16):
      with ops.device('/device:CPU:%d' % i):
        cpu_tokens[i] = create_ordering_token()

    @def_function.function
    def run_all_gather_16devices():
      # [FUNCTION DOC]
      # Compiled function: all-gather across sixteen logical CPU devices, device i contributing
      # the number i. Returns the list of sixteen results, each expected to be zero..fifteen.
      group_size = 16
      group_key = 3
      instance_key = 1
      collectives = []
      for i in range(16):
        with ops.device('/device:CPU:%d' % i):
          collectives.append(
              collective_ops.all_gather(
                  constant_op.constant([i]),
                  group_size,
                  group_key,
                  instance_key,
                  ordering_token=cpu_tokens[i],
                  communication_hint=communication))
      return collectives

    self.assertAllClose(run_all_gather_1device(), [1.], rtol=1e-5, atol=1e-5)
    for result in run_all_gather_2devices():
      self.assertAllClose(result, [1., 1.], rtol=1e-5, atol=1e-5)

    for result in run_all_gather_16devices():
      self.assertAllClose(
          result, list(range(16)), rtol=1e-5, atol=1e-5)

  def testBroadcast(self, collective_ops, device, communication):
    # [FUNCTION DOC]
    # Scenario: device zero broadcasts a three-element tensor and device one receives it in the
    # same compiled function. Asserts both results equal the sent values.
    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device

    @def_function.function
    def run_broadcast_2devices():
      # [FUNCTION DOC]
      # Compiled function: broadcast send of a three-element tensor from device zero and
      # receive on device one, group of two. Returns the list of the two resulting tensors.
      shape = [3]
      in_value = constant_op.constant([1., 2., 3.], shape=shape)
      group_size = 2
      group_key = 2
      instance_key = 2
      collectives = []
      with ops.device(dev0):
        collectives.append(
            collective_ops.broadcast_send(
                in_value,
                shape,
                in_value.dtype,
                group_size,
                group_key,
                instance_key,
                communication_hint=communication))
      with ops.device(dev1):
        collectives.append(
            collective_ops.broadcast_recv(
                shape,
                in_value.dtype,
                group_size,
                group_key,
                instance_key,
                communication_hint=communication))
      return collectives

    for result in run_broadcast_2devices():
      self.assertAllClose(result, [1., 2., 3.], rtol=1e-5, atol=1e-5)

  def testAllToAll(self, collective_ops, device, communication):
    # [FUNCTION DOC]
    # Scenario: skipped for the v1 adapter (all-to-all not implemented there). Runs all-to-all
    # on a group of one device (result unchanged, one) and on a group of two devices where
    # device i contributes the pair i, i. Asserts the one-device result is one and each
    # two-device result is zero then one.
    if str(collective_ops) == 'v1':
      self.skipTest('CollectiveAllToAllV1 is not implemented.')
    devices = ['/device:%s:0' % device, '/device:%s:1' % device]

    tokens = {}
    for dev in devices:
      with ops.device(dev):
        tokens[dev] = create_ordering_token()

    @def_function.function
    def run_all_to_all_1device():
      # [FUNCTION DOC]
      # Compiled function: all-to-all of one value on the first device with a group of one.
      # Returns the exchanged tensor.
      with ops.device(devices[0]):
        in_value = constant_op.constant([1.0])
        group_size = 1
        group_key = 1
        instance_key = 1
        return collective_ops.all_to_all(
            in_value,
            group_size,
            group_key,
            instance_key,
            communication_hint=communication,
            ordering_token=tokens[devices[0]],
        )

    @def_function.function
    def run_all_to_all_2devices():
      # [FUNCTION DOC]
      # Compiled function: all-to-all across two devices where device i contributes a
      # two-element tensor of the number i. Returns the list of per-device results.
      group_size = 2
      group_key = 2
      instance_key = 2
      collectives = []
      for i in range(2):
        with ops.device(devices[i]):
          collectives.append(
              collective_ops.all_to_all(
                  constant_op.constant([i, i]),
                  group_size,
                  group_key,
                  instance_key,
                  ordering_token=tokens[devices[i]],
                  communication_hint=communication,
              )
          )
      return collectives

    self.assertAllClose(run_all_to_all_1device(), [1.0])
    for result in run_all_to_all_2devices():
      self.assertAllClose(result, [0.0, 1.0])

  def testInstanceKeyScopedUnderGroupKey(self, collective_ops, device,
                                         communication):
    # [FUNCTION DOC]
    # Scenario: skipped on GPU when fewer than four GPUs exist. Four devices form two groups of
    # two (distinct group keys) that reuse the same instance key, proving instance keys are
    # scoped under group keys. Inputs one,two,three,four. Asserts results three, three, seven,
    # seven.
    if device == 'GPU' and context.num_gpus() < 4:
      self.skipTest('not enough GPU')

    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device
    dev2 = '/device:%s:2' % device
    dev3 = '/device:%s:3' % device

    tokens = {}
    for dev in [dev0, dev1, dev2, dev3]:
      with ops.device(dev):
        tokens[dev] = create_ordering_token()

    @def_function.function
    def run_all_reduce_4devices_same_instance_key():
      # [FUNCTION DOC]
      # Compiled function: all-reduce on four devices split into two groups of two (devices 0,1
      # and devices 2,3), all using one instance key and ordering tokens. Device inputs are
      # one to four. Returns the list of four results.
      # Use a common instance key for both groups.
      instance_key = 0
      # We will create 2 groups each with 2 devices.
      group_size = 2
      # Group 0 comprises dev0 and dev1.
      group0_key = 0
      # Group 1 comprises dev2 and dev3.
      group1_key = 1
      collectives = []
      with ops.device(dev0):
        collectives.append(
            collective_ops.all_reduce(
                constant_op.constant(1.),
                group_size,
                group0_key,
                instance_key,
                ordering_token=tokens[dev0],
            ))
      with ops.device(dev1):
        collectives.append(
            collective_ops.all_reduce(
                constant_op.constant(2.),
                group_size,
                group0_key,
                instance_key,
                ordering_token=tokens[dev1],
            ))
      with ops.device(dev2):
        collectives.append(
            collective_ops.all_reduce(
                constant_op.constant(3.),
                group_size,
                group1_key,
                instance_key,
                ordering_token=tokens[dev2],
            ))
      with ops.device(dev3):
        collectives.append(
            collective_ops.all_reduce(
                constant_op.constant(4.),
                group_size,
                group1_key,
                instance_key,
                ordering_token=tokens[dev3],
            ))
      return collectives

    results = run_all_reduce_4devices_same_instance_key()
    self.assertAllClose(results[0], 3., rtol=1e-5, atol=1e-5)
    self.assertAllClose(results[1], 3., rtol=1e-5, atol=1e-5)
    self.assertAllClose(results[2], 7., rtol=1e-5, atol=1e-5)
    self.assertAllClose(results[3], 7., rtol=1e-5, atol=1e-5)

  def testCollectiveGroupSizeOne(self, collective_ops, device, communication):
    # [FUNCTION DOC]
    # Scenario: eagerly runs all-reduce and then all-gather on one device with a group of one.
    # Asserts each result equals the input values unchanged.
    dev0 = '/device:%s:0' % device

    group_size = 1
    group_key = 100
    in_value = [1., 2., 3., 4.]
    in_tensor = constant_op.constant(in_value)

    tokens = {}
    for dev in [dev0]:
      with ops.device(dev):
        tokens[dev] = create_ordering_token()

    with ops.device(dev0):
      reduced_tensor = collective_ops.all_reduce(
          in_tensor,
          group_size,
          group_key,
          instance_key=100,
          ordering_token=tokens[dev0],
          communication_hint=communication)
    self.assertAllEqual(in_value, reduced_tensor.numpy())

    with ops.device(dev0):
      gathered_tensor = collective_ops.all_gather(
          in_tensor,
          group_size,
          group_key,
          instance_key=200,
          ordering_token=tokens[dev0],
          communication_hint=communication)
    self.assertAllEqual(in_value, gathered_tensor.numpy())

  def testCollectiveInvalidKey(self, collective_ops, device, communication):
    # [FUNCTION DOC]
    # Scenario: an all-reduce is run with a group and instance key, then an all-gather is run
    # reusing the same instance key. Expected: first result equals the input; second raises an
    # internal error saying the instance expected a different collective type.
    dev0 = '/device:%s:0' % device

    group_size = 1
    group_key = 100
    instance_key = 100
    in_value = [1., 2., 3., 4.]
    in_tensor = constant_op.constant(in_value)

    tokens = {}
    for dev in [dev0]:
      with ops.device(dev):
        tokens[dev] = create_ordering_token()

    with ops.device(dev0):
      reduced_tensor = collective_ops.all_reduce(
          in_tensor,
          group_size,
          group_key,
          instance_key,
          ordering_token=tokens[dev0],
          communication_hint=communication)
    self.assertAllEqual(in_value, reduced_tensor.numpy())

    with self.assertRaisesRegex(
        errors.InternalError, 'instance 100 expected type 0 and data_type 1 but'
        ' got type 2 and data_type 1'):
      with ops.device(dev0):
        collective_ops.all_gather(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=tokens[dev0],
            communication_hint=communication)

  def testMultipleGroups(self, collective_ops, device, communication):
    # [FUNCTION DOC]
    # Scenario: skipped on GPU with fewer than four GPUs. All-reduce is run in groups of two
    # (group key one) and then three (group key two) among four available devices. Asserts each
    # member of a group receives the group-key value times the group size in every element.
    if device == 'GPU' and context.num_gpus() < 4:
      self.skipTest('not enough GPU')

    num_elements = 4
    tokens = {}
    for device_idx in range(num_elements):
      dev = '/{}:{}'.format(device, device_idx)
      with ops.device(dev):
        tokens[dev] = create_ordering_token()

    @def_function.function
    def run_all_reduce(group_size, group_key):
      # [FUNCTION DOC]
      # Compiled function: all-reduce within a group of the given size and key, using the first
      # group-size devices; every device contributes the group key repeated. The key also serves
      # as instance key. Returns the list of per-device results.
      instance_key = group_key
      input_value = [float(group_key) for i in range(num_elements)]
      collectives = []
      for device_idx in range(group_size):
        dev = '/{}:{}'.format(device, device_idx)
        with ops.device(dev):
          input_tensor = constant_op.constant(input_value)
          collectives.append(
              collective_ops.all_reduce(
                  input_tensor,
                  group_size,
                  group_key,
                  instance_key,
                  ordering_token=tokens[dev],
                  communication_hint=communication))
      return collectives

    def run_and_assert(group_size, group_key):
      # [FUNCTION DOC]
      # Helper: runs the grouped all-reduce for the given group size and key and asserts every
      # result element equals the key times the group size. Returns nothing.
      for reduced_tensor in run_all_reduce(group_size, group_key):
        self.assertAllEqual(
            [float(group_key) * group_size for i in range(num_elements)],
            reduced_tensor.numpy())

    run_and_assert(group_size=2, group_key=1)
    run_and_assert(group_size=3, group_key=2)


@combinations.generate(
    combinations.times(
        combinations.combine(
            collective_ops=[combinations.NamedObject('v2', CollectiveOpsV2)],
            mode='eager',
            max_subdivs_per_device=[-1, 0, 16]), device_combination))
class AllReduceWithSubdivisionsTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Eager-mode tests of the v2 all-reduce with the maximum-subdivisions-per-device option set to
  # unset (negative one), zero or sixteen, across device and communication combinations.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: resets the global eager context via the module setup helper with the default
    # number of devices, then runs the base class setup.
    _setup_context()
    super().setUp()

  def testReduce(self, collective_ops, device, communication,
                 max_subdivs_per_device):
    # [FUNCTION DOC]
    # Scenario: all-reduce on a group of one and a group of two devices, passing the
    # subdivision option only when it is not negative one. Asserts the single result is one and
    # each two-device result is two.
    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device

    tokens = {}
    for dev in [dev0, dev1]:
      with ops.device(dev):
        tokens[dev] = create_ordering_token()

    @def_function.function
    def run_all_reduce_1device():
      # [FUNCTION DOC]
      # Compiled function: all-reduce of one on device zero with a group of one. When the
      # subdivision parameter is negative one it is not passed; otherwise it is passed through.
      # Returns the reduced tensor.
      with ops.device(dev0):
        in_value = constant_op.constant([1.])
        group_size = 1
        group_key = 1
        instance_key = 1
        if max_subdivs_per_device == -1:
          return collective_ops.all_reduce(
              in_value,
              group_size,
              group_key,
              instance_key,
              ordering_token=tokens[dev0],
              communication_hint=communication)
        else:
          return collective_ops.all_reduce(
              in_value,
              group_size,
              group_key,
              instance_key,
              ordering_token=tokens[dev0],
              communication_hint=communication,
              max_subdivs_per_device=max_subdivs_per_device)

    @def_function.function
    def run_all_reduce_2devices():
      # [FUNCTION DOC]
      # Compiled function: all-reduce of the value one across a group of two devices (device
      # zero and one) sharing one group key and instance key, each with its own ordering token.
      # Returns the list of two results, each expected to equal two.
      in_value = constant_op.constant([1.])
      group_size = 2
      group_key = 2
      instance_key = 2
      collectives = []
      with ops.device(dev0):
        collectives.append(
            collective_ops.all_reduce(
                in_value,
                group_size,
                group_key,
                instance_key,
                ordering_token=tokens[dev0],
                communication_hint=communication))
      with ops.device(dev1):
        collectives.append(
            collective_ops.all_reduce(
                in_value,
                group_size,
                group_key,
                instance_key,
                ordering_token=tokens[dev1],
                communication_hint=communication))
      return collectives

    self.assertAllClose(run_all_reduce_1device(), [1.], rtol=1e-5, atol=1e-5)
    for result in run_all_reduce_2devices():
      self.assertAllClose(result, [2.], rtol=1e-5, atol=1e-5)


@combinations.generate(
    combinations.combine(required_physical_gpus=2, mode='eager'))
class XlaTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Eager-mode tests (needing two physical GPUs) of the v2 all-reduce inside XLA-compiled
  # functions.

  def testReduce(self):
    # [FUNCTION DOC]
    # Scenario: two threads, one per GPU, each run an XLA-compiled all-reduce of one with a
    # group of two. Asserts both collected results are the value two.
    device0 = '/device:GPU:0'
    device1 = '/device:GPU:1'
    group_size = 2
    group_key = 100
    instance_key = 100
    results = []

    def all_reduce(device):
      # [FUNCTION DOC]
      # Thread body: creates an ordering token on the given device, runs a compiled all-reduce
      # on it and appends the result to the shared results list. Returns nothing.

      with ops.device(device):
        token = create_ordering_token()

      @def_function.function(jit_compile=True)
      def f():
        # [FUNCTION DOC]
        # XLA-compiled function: v2 all-reduce of one with the shared group size, group key,
        # instance key and the ordering token. Returns the reduced tensor.
        return _collective_ops.all_reduce_v2([1.],
                                             group_size,
                                             group_key,
                                             instance_key,
                                             ordering_token=token)

      with ops.device(device):
        results.append(f())

    t0 = threading.Thread(target=all_reduce, args=(device0,))
    t1 = threading.Thread(target=all_reduce, args=(device1,))
    t0.start()
    t1.start()
    t0.join()
    t1.join()

    self.assertAllEqual(results, [[2.], [2.]])

  def testReduceSameGraph(self):
    # [FUNCTION DOC]
    # Scenario: a single XLA-compiled function launches all-reduce on both GPUs. Expected
    # (currently): an invalid-argument error about accessing a resource, since multi-device
    # support in compiled functions is not available (see the bug note in the test).
    device0 = '/device:GPU:0'
    device1 = '/device:GPU:1'
    group_size = 2
    group_key = 100
    instance_key = 100
    results = []

    @def_function.function(jit_compile=True)
    def func():
      # [FUNCTION DOC]
      # XLA-compiled function that appends to the shared results list the all-reduce results for
      # both GPUs and returns that list. Expected to fail to compile or run.

      def all_reduce(device):
        # [FUNCTION DOC]
        # Creates an ordering token on the device and returns a v2 all-reduce of one using it.

        with ops.device(device):
          token = create_ordering_token()

          return _collective_ops.all_reduce_v2([1.],
                                               group_size,
                                               group_key,
                                               instance_key,
                                               ordering_token=token)

      results.append(all_reduce(device0))
      results.append(all_reduce(device1))
      return results

    # FIXME(b/204228837): the error shall no longer be about resources
    # after multi-device support in jit_compile lands. This will likely
    # becomes a deadlock near ResolveDeviceAssignment, or an error in the MLIR
    # bridge on resetting CollectiveInfo.
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'Trying to access resource'):
      func()


@combinations.generate(
    combinations.combine(
        required_physical_gpus=2, mode='eager', jit_compile=[True, False]))
class GroupAssignmentTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Eager-mode tests (needing two physical GPUs, with and without XLA compilation) of the
  # group assignment op that derives group size and key from a device index at run time.

  def testGroupAssignmentBeforeAllReduce(self, jit_compile):
    # [FUNCTION DOC]
    # Scenario: two threads (one per GPU) run a function that first assigns groups, using
    # assignment of two single-device groups, then all-reduces one. Asserts each result is
    # one (each device is alone in its group).
    device0 = '/device:GPU:0'
    device1 = '/device:GPU:1'
    instance_key = 100
    results = []

    group_assignment = [[0], [1]]

    def all_reduce(device, device_index):
      # [FUNCTION DOC]
      # Thread body: creates an ordering token on the device, runs a function for the given
      # device index and appends its result to the shared results list. Returns nothing.

      with ops.device(device):
        token = create_ordering_token()

      @def_function.function(jit_compile=jit_compile)
      def f(device_index):
        # [FUNCTION DOC]
        # Function: assigns a group (size and key) for the device index from the single-device
        # group assignment, then all-reduces one in it. Returns the reduced tensor.
        group_size, group_key = _collective_ops.assign_group_v2(
            group_assignment=group_assignment,
            device_index=device_index,
            base_key=1)
        return _collective_ops.all_reduce_v2([1.],
                                             group_size,
                                             group_key,
                                             instance_key,
                                             ordering_token=token)

      with ops.device(device):
        results.append(f(device_index))

    t0 = threading.Thread(target=all_reduce, args=(device0, 0))
    t1 = threading.Thread(target=all_reduce, args=(device1, 1))
    t0.start()
    t1.start()
    t0.join()
    t1.join()

    self.assertAllEqual(results, [[1.], [1.]])

  def testTwoGroupAssignmentBeforeAllReduce(self, jit_compile):
    # [FUNCTION DOC]
    # Scenario: two threads (one per GPU) each run a function doing two assignments and
    # all-reduces: first with single-device groups, then with one group of both devices (a
    # different base key). Asserts each device gets one from the first and two from the second.
    device0 = '/device:GPU:0'
    device1 = '/device:GPU:1'
    instance_key = 100
    results = []

    group_assignment1 = [[0], [1]]
    group_assignment2 = [[0, 1]]

    def all_reduce(device, device_index):
      # [FUNCTION DOC]
      # Thread body: creates an ordering token on the device, runs the two-stage function for
      # the device index and appends its pair of results to the shared list. Returns nothing.

      with ops.device(device):
        token = create_ordering_token()

      @def_function.function(jit_compile=jit_compile)
      def f(device_index):
        # [FUNCTION DOC]
        # Function: assigns groups with the first assignment, all-reduces one, assigns groups
        # again with the second assignment and another base key, all-reduces one again. Returns
        # both reduced tensors.
        group_size, group_key = _collective_ops.assign_group_v2(
            group_assignment=group_assignment1,
            device_index=device_index,
            base_key=1)
        r1 = _collective_ops.all_reduce_v2([1.],
                                           group_size,
                                           group_key,
                                           instance_key,
                                           ordering_token=token)

        group_size, group_key = _collective_ops.assign_group_v2(
            group_assignment=group_assignment2,
            device_index=device_index,
            base_key=10000)
        r2 = _collective_ops.all_reduce_v2([1.],
                                           group_size,
                                           group_key,
                                           instance_key,
                                           ordering_token=token)
        return r1, r2

      with ops.device(device):
        results.append(f(device_index))

    t0 = threading.Thread(target=all_reduce, args=(device0, 0))
    t1 = threading.Thread(target=all_reduce, args=(device1, 1))
    t0.start()
    t1.start()
    t0.join()
    t1.join()

    self.assertAllEqual(results, [[[1.], [2.]], [[1.], [2.]]])


@combinations.generate(
    combinations.times(collective_op_combinations, device_combination))
class AbortCollectiveOpsTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Eager-mode tests of aborting collective ops in the context, parameterized over the v1 and v2
  # all-reduce and all-gather variants and device and communication combinations. Each test hangs
  # a collective, aborts after a delay, checks the error, resets the context and verifies that
  # collectives work again.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: resets the global eager context via the module setup helper, then runs the base
    # class setup.
    _setup_context()
    super().setUp()

  def testAbortGroupParamsResolution(self, collective_op, device,
                                     communication):
    # [FUNCTION DOC]
    # Scenario: a single collective for a group of two hangs in group resolution. A thread aborts
    # collectives with unavailable "peer down" after two seconds. Expected: that collective and a
    # following one raise unavailable errors with that text. The context is then reset and a
    # function running the collective on both devices must complete.
    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device
    group_size = 2
    group_key = 100
    instance_key = 100
    in_tensor = constant_op.constant([1.])

    tokens = {}
    for device in [dev0, dev1]:
      with ops.device(device):
        tokens[device] = create_ordering_token()

    def abort_fn():
      # [FUNCTION DOC]
      # Thread body: sleeps two seconds then aborts all collective ops in the context with an
      # unavailable error with message "peer down". Returns nothing.
      time.sleep(2)
      context.context().abort_collective_ops(errors.UNAVAILABLE, 'peer down')

    t = threading.Thread(target=abort_fn)
    t.start()

    with self.assertRaisesRegex(errors.UnavailableError, 'peer down'):
      # This hangs on params resolution since we're only launching one
      # collective for a group size of 2.
      with ops.device(dev0):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=tokens[dev0],
            communication_hint=communication)

    # After abortion, subsequent collectives should fail immediately.
    with self.assertRaisesRegex(errors.UnavailableError, 'peer down'):
      with ops.device(dev0):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=tokens[dev0],
            communication_hint=communication)

    t.join()
    # Reset the context in order to reset the collective executor.
    _setup_context()

    # After reset non-NCCL collectives should work.
    def collective_fn():
      # [FUNCTION DOC]
      # Runs the parameterized collective on both devices in turn with the shared tensor, keys,
      # ordering tokens and communication hint. Returns nothing.
      for device in [dev0, dev1]:
        with ops.device(device):
          collective_op(
              in_tensor,
              group_size,
              group_key,
              instance_key,
              ordering_token=tokens[device],
              communication_hint=communication)

    def_function.function(collective_fn)()

  def testAbortInstanceParamsResolution(self, collective_op, device,
                                        communication):
    # [FUNCTION DOC]
    # Scenario: a normal collective resolves the group first. Then a thread aborts collectives
    # after two seconds, while a collective with a new instance key (group of two, one
    # participant) hangs in instance resolution. Expected: it and a later one raise unavailable
    # "peer down". The context is reset (twice: once directly and once by the setup helper) and
    # the original collective function must succeed again.
    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device
    group_size = 2
    group_key = 100
    instance_key = 100
    in_tensor = constant_op.constant([1.])

    tokens = {}
    for device in [dev0, dev1]:
      with ops.device(device):
        tokens[device] = create_ordering_token()

    def collective_fn():
      # [FUNCTION DOC]
      # Runs the parameterized collective on both devices in turn with the shared tensor, keys,
      # ordering tokens and communication hint. Returns nothing.
      for device in [dev0, dev1]:
        with ops.device(device):
          collective_op(
              in_tensor,
              group_size,
              group_key,
              instance_key,
              ordering_token=tokens[device],
              communication_hint=communication)

    # First perform a normal all-reduce to complete the group resolution.
    def_function.function(collective_fn)()

    def abort_fn():
      # [FUNCTION DOC]
      # Thread body: sleeps two seconds then aborts all collective ops in the context with an
      # unavailable error with message "peer down". Returns nothing.
      time.sleep(2)
      context.context().abort_collective_ops(errors.UNAVAILABLE, 'peer down')

    t = threading.Thread(target=abort_fn)
    t.start()

    # Use a different instance key to trigger another instance resolution.
    instance_key = 101
    with self.assertRaisesRegex(errors.UnavailableError, 'peer down'):
      # This hangs on params resolution since we're only launching one
      # collective for a group size of 2.
      with ops.device(dev0):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=tokens[dev0],
            communication_hint=communication)

    # After abortion, subsequent collectives should fail immediately.
    with self.assertRaisesRegex(errors.UnavailableError, 'peer down'):
      with ops.device(dev0):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=tokens[dev0],
            communication_hint=communication)

    context._reset_context()  # pylint: disable=protected-access
    t.join()
    # Reset the context in order to reset the collective executor.
    _setup_context()

    # After reset non-NCCL collectives should work.
    def_function.function(collective_fn)()

  def testAbortCommunication(self, collective_op, device, communication):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the hanging calls on device zero use the ordering token looked up by the loop
    # variable left over from the earlier token-creation loop (the last device, device one), not
    # the device zero token; this looks unintended though the test still passes.
    # Scenario: a normal collective completes resolution. A thread then aborts collectives after
    # two seconds while a single-participant collective on device zero hangs. Expected: it and a
    # following collective raise unavailable "peer down". The context is reset and the first
    # function runs successfully again.
    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device
    group_size = 2
    group_key = 100
    instance_key = 100
    in_tensor = constant_op.constant([1.])

    tokens = {}
    for device in [dev0, dev1]:
      with ops.device(device):
        tokens[device] = create_ordering_token()

    # First perform a normal collective to finish resolution.
    def collective_fn():
      # [FUNCTION DOC]
      # Runs the parameterized collective on both devices in turn with the shared tensor, keys,
      # ordering tokens and communication hint. Returns nothing.
      for device in [dev0, dev1]:
        with ops.device(device):
          collective_op(
              in_tensor,
              group_size,
              group_key,
              instance_key,
              ordering_token=tokens[device],
              communication_hint=communication)

    def_function.function(collective_fn)()

    # Launch a collective that hangs, and abort the collective executor after
    # the launch.
    def abort_fn():
      # [FUNCTION DOC]
      # Thread body: sleeps two seconds then aborts all collective ops in the context with an
      # unavailable error with message "peer down". Returns nothing.
      time.sleep(2)
      context.context().abort_collective_ops(errors.UNAVAILABLE, 'peer down')

    t = threading.Thread(target=abort_fn)
    t.start()

    with self.assertRaisesRegex(errors.UnavailableError, 'peer down'):
      with ops.device(dev0):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=tokens[device],
            communication_hint=communication)

    # After abortion, subsequent collectives should fail immediately.
    with self.assertRaisesRegex(errors.UnavailableError, 'peer down'):
      with ops.device(dev0):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=tokens[device],
            communication_hint=communication)

    # Reset the context in order to reset the collective executor.
    t.join()
    _setup_context()
    def_function.function(collective_fn)()


class OpCancellationTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Eager-mode tests of how collective ops react to other op errors (such as end-of-data) and to
  # cancellation: v1 collectives are aborted when an op error occurs while a collective is
  # active, v2 collectives are not and rely on cancellation instead.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: resets the global eager context via the module setup helper, then runs the base
    # class setup.
    _setup_context()
    super().setUp()

  @combinations.generate(
      combinations.times(
          combinations.combine(
              collective_op=[
                  combinations.NamedObject('all_reduce',
                                           CollectiveOpsV1.all_reduce),
                  combinations.NamedObject('all_gather',
                                           CollectiveOpsV1.all_gather),
              ],
              mode='eager'), device_combination))
  def testOpErrorNotAbortIfNoCollective(self, collective_op, device,
                                        communication):
    # [FUNCTION DOC]
    # Scenario (v1 all-reduce and all-gather): a function reads two items from a one-element
    # dataset, running a collective on each, so the second read raises end-of-data while no
    # collective is active. Expected: an out-of-range error, and collectives still work
    # afterwards, i.e. the context is not aborted.
    # Do not abort if there's no active collective ops. There could be
    # exceptions like EOF which we expect users to catch, aborting collective
    # ops on all op errors intervenes with this workflow.
    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device
    group_size = 2
    group_key = 100
    instance_key = 100
    dataset = dataset_ops.Dataset.from_tensors([1.])

    tokens = {}
    for device in [dev0, dev1]:
      with ops.device(device):
        tokens[device] = create_ordering_token()

    @def_function.function
    def collective_fn(in_tensor):
      # [FUNCTION DOC]
      # Compiled function: runs the parameterized collective on both devices in turn for the
      # given input tensor. Returns nothing.
      for device in [dev0, dev1]:
        with ops.device(device):
          collective_op(
              in_tensor,
              group_size,
              group_key,
              instance_key,
              ordering_token=tokens[device],
              communication_hint=communication)

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Compiled function: creates an iterator over the dataset, runs the collective on the first
      # item, then reads again, which raises end-of-data (out-of-range). Returns nothing.
      iterator = iter(dataset)
      collective_fn(next(iterator))
      # This next(iterator) should raise EOF.
      collective_fn(next(iterator))

    collective_fn(constant_op.constant([1.]))
    with self.assertRaises(errors.OutOfRangeError):
      f()
    collective_fn(constant_op.constant([1.]))

  @combinations.generate(
      combinations.times(
          combinations.combine(
              collective_op=[
                  combinations.NamedObject('all_reduce_v2',
                                           CollectiveOpsV2.all_reduce),
                  combinations.NamedObject('all_gather_v2',
                                           CollectiveOpsV2.all_gather),
              ],
              mode='eager'), device_combination))
  def testOpErrorNotAbortIfNoCollectiveV2(self, collective_op, device,
                                          communication):
    # [FUNCTION DOC]
    # Same scenario as the v1 variant but for the v2 all-reduce and all-gather: end-of-data raised
    # while no collective is active yields an out-of-range error, and collectives keep working
    # afterwards.
    # Do not abort if there's no active collective ops. There could be
    # exceptions like EOF which we expect users to catch, aborting collective
    # ops on all op errors intervenes with this workflow.
    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device
    group_size = 2
    group_key = 100
    instance_key = 100
    dataset = dataset_ops.Dataset.from_tensors([1.])
    tokens = {}
    for device in [dev0, dev1]:
      with ops.device(device):
        tokens[device] = create_ordering_token()

    @def_function.function
    def collective_fn(in_tensor):
      # [FUNCTION DOC]
      # Compiled function: runs the parameterized collective on both devices in turn for the
      # given input tensor. Returns nothing.
      for device in [dev0, dev1]:
        with ops.device(device):
          collective_op(
              in_tensor,
              group_size,
              group_key,
              instance_key,
              communication_hint=communication,
              ordering_token=tokens[device])

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Compiled function: creates an iterator over the dataset, runs the collective on the first
      # item, then reads again, which raises end-of-data (out-of-range). Returns nothing.
      iterator = iter(dataset)
      collective_fn(next(iterator))
      # This next(iterator) should raise EOF.
      collective_fn(next(iterator))

    collective_fn(constant_op.constant([1.]))
    with self.assertRaises(errors.OutOfRangeError):
      f()
    collective_fn(constant_op.constant([1.]))

  @combinations.generate(
      combinations.times(
          combinations.combine(
              collective_op=[
                  combinations.NamedObject('all_reduce',
                                           CollectiveOpsV1.all_reduce),
                  combinations.NamedObject('all_gather',
                                           CollectiveOpsV1.all_gather),
              ],
              mode='eager'), device_combination))
  def testOpErrorAbortWithCollective(self, collective_op, device,
                                     communication):
    # [FUNCTION DOC]
    # Scenario (v1 ops): a function launches a collective that cannot finish (group of two, one
    # participant) and then triggers end-of-data on a slowed dataset. Expected: an out-of-range
    # error, after which collectives are aborted so a new collective raises a cancelled error.
    # Abort v1 collective ops if there're active collective ops at the time of
    # an op error. This is due to the inability to cancel collective ops, and op
    # errors may cause running collective ops to hang.
    dev0 = '/device:%s:0' % device
    group_size = 2
    group_key = 100
    instance_key = 100
    in_tensor = constant_op.constant([1.])
    # Make the dataset sleep a while so that the collective is being executed
    # when the EOF happens.
    dataset = dataset_ops.Dataset.from_tensors([1.]).apply(
        dataset_testing.sleep(sleep_microseconds=200))

    tokens = {}
    for device in [dev0]:
      with ops.device(device):
        tokens[device] = create_ordering_token()

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Compiled function: starts an unfinishable collective on device zero, then reads two items
      # from the one-element slowed dataset so the second read raises end-of-data. Returns the
      # collective result if it ever got that far.
      # Launch a collective op that won't be able to finish to test abortion
      # when other ops error.
      with ops.device(dev0):
        ret = collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=tokens[dev0],
            communication_hint=communication)
      iterator = iter(dataset)
      next(iterator)
      # This should raise EOF.
      next(iterator)
      return ret

    with self.assertRaises(errors.OutOfRangeError):
      f()
    # Now collective ops is aborted, subsequent collective ops should fail with
    # the previous error.
    with self.assertRaises(errors.CancelledError):
      with ops.device(dev0):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=tokens[dev0],
            communication_hint=communication)

  @combinations.generate(
      combinations.times(
          combinations.combine(
              collective_op=[
                  combinations.NamedObject('all_reduce_v2',
                                           CollectiveOpsV2.all_reduce),
                  combinations.NamedObject('all_gather_v2',
                                           CollectiveOpsV2.all_gather),
              ],
              mode='eager'), device_combination))
  def testOpErrorNotAbortWithCollectiveV2(self, collective_op, device,
                                          communication):
    # [FUNCTION DOC]
    # Scenario (v2 ops): after a normal collective resolves the group, a function launches an
    # unfinishable collective and then triggers end-of-data on a slowed dataset. Expected: an
    # out-of-range error, no abort, and a new collective completes normally afterwards.
    # Do not abort v2 collective ops even if there're active collective ops at
    # the time of an op error. We rely cancellation to terminate active
    # collective ops.
    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device
    group_size = 2
    group_key = 100
    instance_key = 100
    in_tensor = constant_op.constant([1.])

    tokens = {}
    for device in [dev0, dev1]:
      with ops.device(device):
        tokens[device] = create_ordering_token()

    @def_function.function
    def collective_fn():
      # [FUNCTION DOC]
      # Compiled function: runs the parameterized collective on both devices in turn with the
      # shared tensor, keys and tokens. Returns nothing.
      for device in [dev0, dev1]:
        with ops.device(device):
          collective_op(
              in_tensor,
              group_size,
              group_key,
              instance_key,
              ordering_token=tokens[device],
              communication_hint=communication)

    # Local params resolution cannot be cancelled yet, so we perform a normal
    # collective so that the group is resolved.
    collective_fn()

    # Make the dataset sleep a while so that the collective is being executed
    # when the EOF happens.
    dataset = dataset_ops.Dataset.from_tensors([1.]).apply(
        dataset_testing.sleep(sleep_microseconds=200))

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Compiled function: starts an unfinishable collective on device zero, then reads two items
      # from the one-element slowed dataset so the second read raises end-of-data. Returns the
      # collective result if it ever got that far.
      # Launch a collective op that won't be able to finish to test cancellation
      # when other ops error.
      with ops.device(dev0):
        ret = collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=tokens[dev0],
            communication_hint=communication)
      iterator = iter(dataset)
      next(iterator)
      # This should raise EOF.
      next(iterator)
      return ret

    with self.assertRaises(errors.OutOfRangeError):
      f()
    # Collective ops shouldn't be aborted and new collectives should be able to
    # proceed.
    collective_fn()

  @combinations.generate(
      combinations.times(
          combinations.combine(
              collective_op=[
                  combinations.NamedObject('all_reduce_v2',
                                           CollectiveOpsV2.all_reduce),
                  combinations.NamedObject('all_gather_v2',
                                           CollectiveOpsV2.all_gather),
              ],
              mode='eager'), device_combination))
  def testCancelDuringParamResolutionV2(self, collective_op, device,
                                        communication):
    # [FUNCTION DOC]
    # Scenario (v2 ops): two threads run the same traced function on two devices, each under its
    # own cancellation manager; one is fed a mismatching value so its assertion fails and it
    # cancels the other (which would otherwise deadlock in parameter resolution). Expected: both
    # threads finish, as counted by a lock-protected counter equal to two.
    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device
    group_size = 2
    group_key = 100
    instance_key = 100
    in_tensor = constant_op.constant([1.])
    t1_cancellation_manager = cancellation.CancellationManager()
    t2_cancellation_manager = cancellation.CancellationManager()

    @def_function.function
    def _collective_fn(x):
      # [FUNCTION DOC]
      # Compiled function: asserts the input equals the shared input tensor (control dependency),
      # then runs the collective without an ordering token (placement is outside the function).
      # Returns the collective result; the assertion fails with an invalid-argument error for
      # mismatching input.
      # Run an assertion to crash one of the two function executions running
      # collectives. We explicitly cancel the other in response.
      assert_op = check_ops.assert_equal(x, in_tensor)
      with ops.control_dependencies([assert_op]):
        return collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            # This test cannot use ordering_token because the placement
            # occurs outside of tf.function and we cannot relocate the token
            # after concrete function is created.
            # since there is only 1 collective Op in the graph there is no
            # need to use a token for ordering.
            communication_hint=communication)

    collective_concrete = _collective_fn.get_concrete_function(in_tensor)

    finish_mu = threading.Lock()
    finishes = 0

    def _placement_wrapper(device, x, my_cancellation, other_cancellation):
      # [FUNCTION DOC]
      # Thread body: runs the concrete collective function on the given device through its own
      # cancellation manager. On an invalid-argument error it cancels the other manager; a
      # cancelled error is ignored. Otherwise returns the result; after an error it increments
      # the lock-protected finish counter (shared via closure) and returns nothing.
      try:
        with ops.device(device):
          cancelable_collective = my_cancellation.get_cancelable_function(
              collective_concrete)
          return cancelable_collective(x)
      except errors.InvalidArgumentError:
        # `assert_equal` failed for this execution of the function. The other
        # function would deadlock without cancellation.
        other_cancellation.start_cancel()
      except errors.CancelledError:
        pass
      nonlocal finishes
      with finish_mu:
        finishes += 1

    t1 = threading.Thread(
        target=_placement_wrapper,
        args=(dev0, constant_op.constant([1.]), t1_cancellation_manager,
              t2_cancellation_manager))
    t2 = threading.Thread(
        target=_placement_wrapper,
        # Will cause the assertion to fail
        args=(dev1, constant_op.constant([2.]), t2_cancellation_manager,
              t1_cancellation_manager))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    self.assertEqual(finishes, 2)


@combinations.generate(
    combinations.times(collective_op_combinations, device_combination))
class TimeoutTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Parameterized tests of the collective timeout option for v1 and v2 all-reduce and all-gather,
  # over device and communication combinations.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: resets the global eager context via the module setup helper, then runs the base
    # class setup.
    _setup_context()
    super().setUp()

  def testTimeout(self, collective_op, device, communication):
    # [FUNCTION DOC]
    # Scenario: with a one and a half second timeout, a full two-device collective succeeds; then
    # a run where only one device participates but a group of two is declared times out.
    # Expected: a deadline-exceeded error saying the collective timed out during execution, and
    # at least the timeout duration elapsed.
    timeout = 1.5

    tokens = {}
    for i in range(2):
      dev = '/{}:{}'.format(device, i)
      with ops.device(dev):
        tokens[dev] = create_ordering_token()

    @def_function.function
    def run(group_size, reported_group_size=None):
      # [FUNCTION DOC]
      # Compiled function: launches the collective with the timeout on the first group-size
      # devices, declaring reported-group-size (defaults to group-size) as the group size, with a
      # fixed four-element input. Returns the list of results.
      group_key = 20
      instance_key = 30
      tensor = [1., 2., 3., 4.]
      results = []
      if reported_group_size is None:
        reported_group_size = group_size
      for i in range(group_size):
        dev = '/{}:{}'.format(device, i)
        with ops.device(dev):
          input_data = constant_op.constant(tensor)
          result = collective_op(
              input_data,
              group_size=reported_group_size,
              group_key=group_key,
              instance_key=instance_key,
              ordering_token=tokens[dev],
              communication_hint=communication,
              timeout=timeout)
          results.append(result)
      return results

    run(2, 2)

    start_time = time.time()
    with self.assertRaisesRegex(errors.DeadlineExceededError,
                                'Collective has timed out during execution'):
      run(1, 2)
    elapsed = time.time() - start_time
    self.assertAllGreaterEqual(elapsed, timeout)

  def testParamResolutionAfterTimeout(self, collective_op, device,
                                      communication):
    # [FUNCTION DOC]
    # Scenario: the first device launches a collective for a group of two with a timeout and
    # times out waiting for the other worker. The second device, launched afterwards without a
    # timeout, must also fail immediately. Expected: deadline-exceeded errors saying the
    # collective timed out waiting for other workers, both times.
    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device
    timeout = 1.5
    group_key = 20
    instance_key = 30
    input_data = constant_op.constant([1., 2., 3., 4.])

    tokens = {}
    for device in [dev0, dev1]:
      with ops.device(device):
        tokens[device] = create_ordering_token()

    # This timeout comes from param solution.
    with self.assertRaisesRegex(
        errors.DeadlineExceededError,
        'Collective has timed out waiting for other workers'):
      with ops.device(dev0):
        collective_op(
            input_data,
            group_size=2,
            group_key=group_key,
            instance_key=instance_key,
            ordering_token=tokens[dev0],
            communication_hint=communication,
            timeout=timeout)

    # We launch the second device after the first device times out. This is to
    # simulate the situation when other workers are slow and the timeout is
    # short. It should error immediately.
    with self.assertRaisesRegex(
        errors.DeadlineExceededError,
        'Collective has timed out waiting for other workers'):
      with ops.device(dev1):
        collective_op(
            input_data,
            group_size=2,
            group_key=group_key,
            instance_key=instance_key,
            ordering_token=tokens[dev1],
            communication_hint=communication)

  def testExecutionAfterTimeout(self, collective_op, device, communication):
    # [FUNCTION DOC]
    # Scenario: a normal collective first completes parameter resolution. Then the first device
    # launches a collective with a timeout and times out during execution; the second device,
    # launched afterwards without a timeout, must also fail immediately. Expected:
    # deadline-exceeded errors saying the collective timed out during execution, both times.
    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device
    timeout = 1.5
    group_key = 20
    instance_key = 30
    input_data = constant_op.constant([1., 2., 3., 4.])

    tokens = {}
    for device in [dev0, dev1]:
      with ops.device(device):
        tokens[device] = create_ordering_token()

    @def_function.function
    def run():
      # [FUNCTION DOC]
      # Compiled function: runs the collective with the timeout on both devices in turn, group
      # of two. Returns nothing.
      for device in [dev0, dev1]:
        with ops.device(device):
          collective_op(
              input_data,
              group_size=2,
              group_key=group_key,
              instance_key=instance_key,
              ordering_token=tokens[device],
              communication_hint=communication,
              timeout=timeout)

    # Run a normal all-reduce to complete param resolution.
    run()

    with self.assertRaisesRegex(errors.DeadlineExceededError,
                                'Collective has timed out during execution'):
      with ops.device(dev0):
        collective_op(
            input_data,
            group_size=2,
            group_key=group_key,
            instance_key=instance_key,
            ordering_token=tokens[dev0],
            communication_hint=communication,
            timeout=timeout)

    # We launch the second device after the first device times out. This is to
    # simulate the situation when other workers are slow and the timeout is
    # short. It should error immediately.
    with self.assertRaisesRegex(errors.DeadlineExceededError,
                                'Collective has timed out during execution'):
      with ops.device(dev1):
        # No timeout.
        collective_op(
            input_data,
            group_size=2,
            group_key=group_key,
            instance_key=instance_key,
            ordering_token=tokens[dev1],
            communication_hint=communication)


class CommunicationHintTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests of the communication hint option of collectives (currently the NCCL fallback on CPU).

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: resets the global eager context via the module setup helper, then runs the base
    # class setup.
    _setup_context()
    super().setUp()

  @combinations.generate(
      combinations.times(collective_op_combinations,
                         combinations.combine(required_gpus=[0, 1])))
  def testNCCLFallbackOnCPU(self, collective_op):
    # [FUNCTION DOC]
    # Scenario: runs a collective across two logical CPU devices with the communication hint set
    # to NCCL. Expected: it succeeds by falling back to the ring implementation (no error). The
    # parameterization requests zero or one GPUs only so the test also runs in GPU builds.
    # communication_hint=NCCL should work for CPU by falling back to RING. The
    # test doesn't actually require GPU, only GPU builds. We specify
    # required_gpus=1 so that it's tested with GPU builds.
    dev0 = '/device:CPU:0'
    dev1 = '/device:CPU:1'
    group_key = 20
    instance_key = 30
    input_data = constant_op.constant([1., 2., 3., 4.])

    tokens = {}
    for device in [dev0, dev1]:
      with ops.device(device):
        tokens[device] = create_ordering_token()

    @def_function.function
    def run():
      # [FUNCTION DOC]
      # Compiled function: runs the collective on both CPU devices in turn with the NCCL hint.
      # Returns nothing.
      for device in [dev0, dev1]:
        with ops.device(device):
          collective_op(
              input_data,
              group_size=2,
              group_key=group_key,
              instance_key=instance_key,
              ordering_token=tokens[device],
              communication_hint='NCCL')

    run()


@combinations.generate(
    combinations.times(
        combinations.combine(
            collective_op=[
                combinations.NamedObject('all_reduce_v2',
                                         CollectiveOpsV2.all_reduce),
                combinations.NamedObject('all_gather_v2',
                                         CollectiveOpsV2.all_gather),
            ],
            mode='eager'), device_combination))
class OrderingTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Eager-mode tests that v2 collectives with the same ordering token are chained by control
  # dependencies in the traced graph, while collectives with fresh tokens are not.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: resets the global eager context via the module setup helper, then runs the base
    # class setup.
    _setup_context()
    super().setUp()

  def testOrdering(self, collective_op, device, communication):
    # [FUNCTION DOC]
    # Scenario: a traced function launches, per device, a first collective with a shared token,
    # a second with a fresh token and a third (next instance key) with the shared token. The test
    # inspects the graph: every operation exists and sits on its device; the third has exactly
    # one control input, the first collective; the second and first have no control inputs.
    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device
    group_size = 2
    group_key = 100
    instance_key = 100
    in_tensor = constant_op.constant([1.])

    with ops.device(dev0):
      token0 = create_ordering_token()
    with ops.device(dev1):
      token1 = create_ordering_token()

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Traced function: on each device launches three named collectives: first and third with
      # the device's shared token (the third on the next instance key) and the second with a
      # freshly created token. Returns nothing.
      # Launch the first collective with token.
      with ops.device(dev0):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=token0,
            name='FirstChainedDev0')
      with ops.device(dev1):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=token1,
            name='FirstChainedDev1')
      # Launch the second collective without token.
      with ops.device(dev0):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=create_ordering_token(),
            name='UnchainedDev0')
      with ops.device(dev1):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=create_ordering_token(),
            name='UnchainedDev1')
      # Launch the third collective with token.
      with ops.device(dev0):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key + 1,
            ordering_token=token0,
            name='SecondChainedDev0')
      with ops.device(dev1):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key + 1,
            ordering_token=token1,
            name='SecondChainedDev1')

    graph = f.get_concrete_function().graph
    for device, suffix in [(dev0, 'Dev0'), (dev1, 'Dev1')]:

      first = graph.get_operation_by_name('FirstChained' + suffix)
      second = graph.get_operation_by_name('Unchained' + suffix)
      third = graph.get_operation_by_name('SecondChained' + suffix)
      self.assertIsNotNone(first)
      self.assertTrue(first.device.endswith(device))
      self.assertIsNotNone(second)
      self.assertTrue(second.device.endswith(device))
      self.assertIsNotNone(third)
      self.assertTrue(third.device.endswith(device))

      # Try to find the third collective, which should have the first collective
      # as a control input.
      self.assertLen(third.control_inputs, 1)
      self.assertEqual(third.control_inputs[0].name, 'FirstChained' + suffix)

      self.assertEmpty(second.control_inputs)
      self.assertEmpty(first.control_inputs)


class InputPipelineTest(test.TestCase):
  # [CLASS DOC]
  # Tests of using a v2 collective inside a dataset input pipeline.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: runs the base class setup first, then resets the global eager context via the
    # module setup helper.
    super().setUp()
    _setup_context()

  def testMap(self):
    # [FUNCTION DOC]
    # Scenario: on two CPU devices, a function builds a one-element dataset and maps an
    # all-reduce (group of two) over it, fetching the first element. Inputs one and two.
    # Asserts both devices return three.
    group_size = 2
    group_key = 100
    instance_key = 100

    def create_dataset_and_fetch_one(t):
      # [FUNCTION DOC]
      # Helper: builds a one-element dataset from the input, maps the all-reduce function
      # over it and returns the first element. Input: a tensor value.
      dataset = dataset_ops.Dataset.from_tensor_slices([t])

      def reduce_fn(t):
        # [FUNCTION DOC]
        # Dataset map function: creates a fresh ordering token and returns the v2 all-reduce of
        # the element within the shared group and instance keys.
        # A token is created for each device.
        token = create_ordering_token()
        return CollectiveOpsV2.all_reduce(
            t,
            group_size=group_size,
            group_key=group_key,
            instance_key=instance_key,
            ordering_token=token)

      dataset = dataset.map(reduce_fn)
      return next(iter(dataset))

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Compiled function: runs the dataset helper on the first CPU with one and on the second
      # CPU with two. Returns the pair of fetched values.
      with ops.device('CPU:0'):
        value0 = create_dataset_and_fetch_one([1.])
      with ops.device('CPU:1'):
        value1 = create_dataset_and_fetch_one([2.])
      return value0, value1

    self.assertAllEqual(self.evaluate(f()), [[3.], [3.]])


@combinations.generate(
    combinations.times(
        combinations.combine(collective_op=[
            combinations.NamedObject('all_reduce_v2',
                                     CollectiveOpsV2.all_reduce),
            combinations.NamedObject('all_gather_v2',
                                     CollectiveOpsV2.all_gather)
        ]), device_combination))
class InvalidInputTest(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests that v2 all-reduce and all-gather reject invalid group size, group key and instance key
  # values with an invalid-argument error, over each device and communication combination.

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: resets the global eager context via the module setup helper, then runs the base
    # class setup.
    _setup_context()
    super().setUp()

  def testInvalidGroupKey(self, collective_op, device, communication):
    # [FUNCTION DOC]
    # Scenario: the collective is called with a group key given as a one-element list instead of
    # a scalar. Expected: an invalid-argument error.
    dev0 = '/device:%s:0' % device
    group_size = 2
    group_key = [100]
    instance_key = 100
    in_tensor = constant_op.constant([1.])

    with self.assertRaises(errors.InvalidArgumentError):
      with ops.device(dev0):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=create_ordering_token(),
            communication_hint=communication)

  def testInvalidGroupSize(self, collective_op, device, communication):
    # [FUNCTION DOC]
    # Scenario: the collective is called with a negative group size. Expected: an
    # invalid-argument error.
    dev0 = '/device:%s:0' % device
    group_size = -2
    group_key = 100
    instance_key = 100
    in_tensor = constant_op.constant([1.])

    with self.assertRaises(errors.InvalidArgumentError):
      with ops.device(dev0):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=create_ordering_token(),
            communication_hint=communication)

  def testInvalidInstanceKey(self, collective_op, device, communication):
    # [FUNCTION DOC]
    # Scenario: the collective is called with an instance key given as a one-element list instead
    # of a scalar. Expected: an invalid-argument error.
    dev0 = '/device:%s:0' % device
    group_size = 2
    group_key = 100
    instance_key = [100]
    in_tensor = constant_op.constant([1.])

    with self.assertRaises(errors.InvalidArgumentError):
      with ops.device(dev0):
        collective_op(
            in_tensor,
            group_size,
            group_key,
            instance_key,
            ordering_token=create_ordering_token(),
            communication_hint=communication)


class CollectiveOpsV3Test(test.TestCase, parameterized.TestCase):
  # [CLASS DOC]
  # Tests of the version 3 collective ops, which use explicit communicator handles created per
  # device with a rank (communicator initialization, all-reduce, all-to-all).

  def setUp(self):
    # [FUNCTION DOC]
    # Fixture: runs the base class setup first, then resets the global eager context via the
    # module setup helper.
    super().setUp()
    _setup_context()

  def testGroupInitialization(self):
    # [FUNCTION DOC]
    # Scenario: two communicators (ranks zero and one of a group of two) are initialized on two
    # CPU devices inside a function which is then evaluated. Expected: it runs without error
    # (a reduction check is still a to-do in the source).
    group_size = 2
    group_key = 100

    @def_function.function
    def f():
      # [FUNCTION DOC]
      # Compiled function: initializes a communicator with rank zero on the first CPU and rank
      # one on the second CPU for the shared group key and size. Returns nothing.
      with ops.device('CPU:0'):
        _collective_ops.initialize_communicator(
            group_key=group_key, rank=0, group_size=group_size)
      with ops.device('CPU:1'):
        _collective_ops.initialize_communicator(
            group_key=group_key, rank=1, group_size=group_size)

      # TODO(b/193864859): Add validation with reduction op.

    self.evaluate(f())

  @combinations.generate(device_combination)
  def testAllReduceV3(self, device, communication):
    # [FUNCTION DOC]
    # Scenario: two devices each initialize a communicator (ranks zero and one) and run the v3
    # all-reduce with add on inputs one and two. Asserts both results equal three.
    group_size = 2
    group_key = 101

    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device

    @def_function.function
    def run_all_reduce_2devices():
      # [FUNCTION DOC]
      # Compiled function: on each of two devices initializes a communicator (ranks zero and one,
      # shared group key and size, the communication hint) and runs the v3 all-reduce with the
      # add reduction on inputs one and two. Returns the list of two results, each expected three.
      collectives = []
      with ops.device(dev0):
        group_handle0 = _collective_ops.initialize_communicator(
            group_key=group_key,
            rank=0,
            group_size=group_size,
            communication_hint=communication)
        collectives.append(
            _collective_ops.all_reduce_v3(
                group_handle0, [1.0], reduction='Add'))
      with ops.device(dev1):
        group_handle1 = _collective_ops.initialize_communicator(
            group_key=group_key,
            rank=1,
            group_size=group_size,
            communication_hint=communication)
        collectives.append(
            _collective_ops.all_reduce_v3(
                group_handle1, [2.0], reduction='Add'))
      return collectives

    for result in run_all_reduce_2devices():
      self.assertAllClose(result, [3.], rtol=1e-5, atol=1e-5)

  @combinations.generate(device_combination)
  def testAllToAllV3(self, device, communication):
    # [FUNCTION DOC]
    # Scenario: ranks zero and one on two devices run the v3 all-to-all with inputs one,three and
    # two,four. Asserts the first result is one,two and the second is three,four.
    group_size = 2
    group_key = 104

    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device

    @def_function.function
    def run_all_to_all_2devices():
      # [FUNCTION DOC]
      # Compiled function: initializes communicators on two devices (ranks zero and one in order)
      # and runs the v3 all-to-all with inputs one,three and two,four. Returns the two results.
      collectives = []
      with ops.device(dev0):
        group_handle0 = _collective_ops.initialize_communicator(
            group_key=group_key,
            rank=0,
            group_size=group_size,
            communication_hint=communication)
        collectives.append(
            _collective_ops.all_to_all_v3(group_handle0, [1.0, 3.0]))
      with ops.device(dev1):
        group_handle1 = _collective_ops.initialize_communicator(
            group_key=group_key,
            rank=1,
            group_size=group_size,
            communication_hint=communication)
        collectives.append(
            _collective_ops.all_to_all_v3(group_handle1, [2.0, 4.0]))
      return collectives

    result = run_all_to_all_2devices()
    self.assertAllClose(result[0], [1.0, 2.0], rtol=1e-5, atol=1e-5)
    self.assertAllClose(result[1], [3.0, 4.0], rtol=1e-5, atol=1e-5)

  @combinations.generate(device_combination)
  def testAllToAllV3DifferentUserRank(self, device, communication):
    # [FUNCTION DOC]
    # Scenario: as the plain all-to-all test but the first device has rank one and the second rank
    # zero. Asserts the first result is two,one and the second is four,three.
    group_size = 2
    group_key = 105

    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device

    @def_function.function
    def run_all_to_all_2devices():
      # [FUNCTION DOC]
      # Compiled function: like the plain variant but the first device takes rank one and the
      # second device rank zero. Runs the v3 all-to-all with inputs one,three and two,four.
      # Returns the two results.
      collectives = []
      with ops.device(dev0):
        group_handle0 = _collective_ops.initialize_communicator(
            group_key=group_key,
            rank=1,
            group_size=group_size,
            communication_hint=communication)
        collectives.append(
            _collective_ops.all_to_all_v3(group_handle0, [1.0, 3.0]))
      with ops.device(dev1):
        group_handle1 = _collective_ops.initialize_communicator(
            group_key=group_key,
            rank=0,
            group_size=group_size,
            communication_hint=communication)
        collectives.append(
            _collective_ops.all_to_all_v3(group_handle1, [2.0, 4.0]))
      return collectives

    result = run_all_to_all_2devices()
    self.assertAllClose(result[0], [2.0, 1.0], rtol=1e-5, atol=1e-5)
    self.assertAllClose(result[1], [4.0, 3.0], rtol=1e-5, atol=1e-5)

  @combinations.generate(device_combination)
  def testAllToAllV3DifferentUserRankWithTensorInput(self, device,
                                                     communication):
    # [FUNCTION DOC]
    # KNOWN ISSUE: a bug-reference note in the code says the rank-one result (the first
    # device's) should be four,two, yet the assertions expect three,one for it and four,two for
    # the rank-zero device; the note says "this is correct" so expectations and note disagree.
    # Scenario: first device rank one, second device rank zero, inputs given as tensors
    # (one,two and three,four). Asserts the first result is three,one and the second four,two.

    group_size = 2
    group_key = 106

    dev0 = '/device:%s:0' % device
    dev1 = '/device:%s:1' % device

    @def_function.function
    def run_all_to_all_2devices():
      # [FUNCTION DOC]
      # Compiled function: first device has rank one, second device rank zero; the v3 all-to-all
      # inputs are tensors (one,two and three,four) rather than lists. Returns the two results.
      collectives = []
      with ops.device(dev0):
        group_handle0 = _collective_ops.initialize_communicator(
            group_key=group_key,
            rank=1,
            group_size=group_size,
            communication_hint=communication)
        collectives.append(
            _collective_ops.all_to_all_v3(group_handle0,
                                          constant_op.constant([1.0, 2.0])))
      with ops.device(dev1):
        group_handle1 = _collective_ops.initialize_communicator(
            group_key=group_key,
            rank=0,
            group_size=group_size,
            communication_hint=communication)
        collectives.append(
            _collective_ops.all_to_all_v3(group_handle1,
                                          constant_op.constant([3.0, 4.0])))

      return collectives

    result = run_all_to_all_2devices()
    # FIXME(b/214407359): This is correct.
    # result[0] is rank 1 and shall have 4, 2.
    self.assertAllClose(result[1], [4.0, 2.0], rtol=1e-5, atol=1e-5)
    self.assertAllClose(result[0], [3.0, 1.0], rtol=1e-5, atol=1e-5)


def _setup_context(num_devices=4):
  # [FUNCTION DOC]
  # HELPER: shared fixture for the tests in this module. Resets the global eager context,
  # makes sure at least the given number of logical CPU devices exist (default four),
  # initializes the context and turns on logging of device placement. Returns nothing.
  # Side effects: replaces the global context state, so earlier tensors or devices are invalid.
  context._reset_context()
  test_util.set_logical_devices_to_at_least('CPU', num_devices)
  context.ensure_initialized()
  context.set_log_device_placement(True)


if __name__ == '__main__':
  os.environ['NCCL_DEBUG'] = 'INFO'
  v2_compat.enable_v2_behavior()
  test.main()
