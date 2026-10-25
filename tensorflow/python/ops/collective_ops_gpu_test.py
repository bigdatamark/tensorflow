# [MODULE DOC]
# GPU-only tests for TensorFlow collective operations (all-reduce, broadcast send/receive,
# all-gather) using the NCCL communication backend. It defines one test case class that
# uses two GPUs as a collective group, checks numeric results, and checks the error messages
# for unsupported data types, mismatched group membership and aborted collectives.
# Importing the module has no side effects beyond imports; running it as a script starts the
# TensorFlow test runner.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Collective Operations that require GPU."""

import os
import threading
import time

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import collective_ops
from tensorflow.python.platform import test


class CollectiveOpGPUTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering NCCL-backed collective ops on two GPUs. Class-level state holds the
  # group size (two) and the two GPU device names, set in the class setup. Each test resets
  # the eager context via a helper and skips itself when too few GPUs are available.

  @classmethod
  def setUpClass(cls):
    # [FUNCTION DOC]
    # Class-level fixture run once before all tests. Takes the class; returns nothing.
    # Stores a group size of two and the names of GPU devices 0 and 1 on the class, and sets
    # the process environment variables that turn on informational NCCL logging and parallel
    # NCCL launch mode (a global side effect on the process environment).
    """Set group_size = num_gpus = 2 for all tests in this class."""
    super(CollectiveOpGPUTest, cls).setUpClass()
    # Group size is the number of devices in a group communicating collectively.
    # This will be passed into the collective ops in the tests below.
    cls._group_size = 2
    cls._devices = ['/device:GPU:{}'.format(i) for i in range(2)]
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['NCCL_LAUNCH_MODE'] = 'PARALLEL'

  def _setup_context(self, num_gpus=2):
    # [FUNCTION DOC]
    # HELPER: per-test setup shared by the tests in this class.
    # Resets the global eager context (discarding earlier collective state), counts the
    # physical GPUs, and skips the running test when fewer than num_gpus (default two) exist.
    # Otherwise it initializes the new context. Returns nothing.
    context._reset_context()
    gpus = config.list_physical_devices('GPU')
    if len(gpus) < num_gpus:
      self.skipTest(
          'Expected at least {} GPUs but found {} GPUs'.format(
              num_gpus, len(gpus)
          )
      )
    context.ensure_initialized()

  def testBasicNcclAllReduce(self):
    # [FUNCTION DOC]
    # Runs a sum all-reduce with a divide final step over two GPUs inside a compiled function,
    # one float input vector per GPU. Expects every returned result to equal the element-wise
    # average of the two inputs within a small tolerance.
    self._setup_context()

    inputs = [
        [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
        [0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3],
    ]
    expected = [0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2]
    group_key = 1
    instance_key = 1

    @def_function.function
    def run_basic_all_reduce():
      # [FUNCTION DOC]
      # HELPER: body of the compiled function for the basic all-reduce test.
      # Takes no arguments. On each GPU builds a constant from that device's input and issues
      # an all-reduce (add, then divide) in the shared group. Returns the list of per-device
      # results.
      collectives = []
      for i in range(self._group_size):
        with ops.device(self._devices[i]):
          t = constant_op.constant(inputs[i])
          collectives.append(
              collective_ops.all_reduce(
                  t, self._group_size, group_key, instance_key, 'Add', 'Div'
              )
          )
      return collectives

    for result in run_basic_all_reduce():
      self.assertAllClose(result, expected, rtol=1e-5, atol=1e-5)

  def testInt32Error(self):
    # [FUNCTION DOC]
    # Issues an all-reduce on 32-bit integer tensors on two GPUs and expects an internal error
    # stating that the integer type is not supported on the GPU device.
    self._setup_context()

    inputs = [[0, 1], [2, 3]]
    group_key = 1
    instance_key = 50

    @def_function.function
    def run_int32_error():
      # [FUNCTION DOC]
      # HELPER: compiled function for the integer-type error test. Takes no arguments; on each
      # GPU creates a 32-bit integer constant and starts an all-reduce on it. Returns nothing;
      # execution is expected to raise.
      for i in range(self._group_size):
        with ops.device(self._devices[i]):
          t = constant_op.constant(inputs[i], dtype=dtypes.int32)
          collective_ops.all_reduce(
              t, self._group_size, group_key, instance_key, 'Add', 'Div'
          )

    with self.assertRaisesRegex(
        errors.InternalError, 'does not support datatype DT_INT32 on DEVICE_GPU'
    ):
      run_int32_error()

  def testFp16Reduce(self):
    # [FUNCTION DOC]
    # Same as the basic all-reduce test but with half-precision float inputs. Expects the
    # element-wise average on both GPUs within a looser tolerance suited to half precision.
    self._setup_context()

    inputs = [
        [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
        [0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3],
    ]
    expected = [0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2]
    group_key = 1
    instance_key = 100

    @def_function.function
    def run_fp16_reduce():
      # [FUNCTION DOC]
      # HELPER: compiled function for the half-precision test. Takes no arguments; on each GPU
      # builds a half-precision constant and all-reduces it (add, then divide). Returns the
      # list of per-device results.
      collectives = []
      for i in range(self._group_size):
        with ops.device(self._devices[i]):
          t = constant_op.constant(inputs[i], dtype=dtypes.float16)
          collectives.append(
              collective_ops.all_reduce(
                  t, self._group_size, group_key, instance_key, 'Add', 'Div'
              )
          )
      return collectives

    for result in run_fp16_reduce():
      self.assertAllClose(result, expected, rtol=1e-3, atol=1e-3)

  def testNcclHintAllReduce(self):
    # [FUNCTION DOC]
    # Like the basic all-reduce test but explicitly passes the communication hint requesting
    # the NCCL backend. Expects the element-wise average from both GPUs.
    self._setup_context()

    inputs = [
        [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
        [0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3],
    ]
    expected = [0.2, 1.2, 2.2, 3.2, 4.2, 5.2, 6.2, 7.2]
    group_key = 1
    instance_key = 1

    @def_function.function
    def run_nccl_hint_all_reduce():
      # [FUNCTION DOC]
      # HELPER: compiled function for the NCCL-hint test. Takes no arguments; all-reduces one
      # constant per GPU with the NCCL communication hint. Returns the list of results.
      collectives = []
      for i in range(self._group_size):
        with ops.device(self._devices[i]):
          t = constant_op.constant(inputs[i])
          collectives.append(
              collective_ops.all_reduce(
                  t,
                  self._group_size,
                  group_key,
                  instance_key,
                  'Add',
                  'Div',
                  communication_hint='nccl',
              )
          )
      return collectives

    for result in run_nccl_hint_all_reduce():
      self.assertAllClose(result, expected, rtol=1e-5, atol=1e-5)

  def testBasicNcclBroadcast(self):
    # [FUNCTION DOC]
    # Broadcasts a float vector from GPU 0 (sender) to GPU 1 (receiver) in a compiled function.
    # Expects both returned values to equal the original vector.
    self._setup_context()

    tensor_value = [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1]
    group_key = 1
    instance_key = 1

    @def_function.function
    def run_basic_nccl_broadcast():
      # [FUNCTION DOC]
      # HELPER: compiled function for the broadcast test. Takes no arguments; GPU 0 issues a
      # broadcast send of the vector and GPU 1 issues the matching receive using the same
      # shape and type. Returns the list of the send and receive results.
      collectives = []
      with ops.device(self._devices[0]):
        t = constant_op.constant(tensor_value)
        collectives.append(
            collective_ops.broadcast_send(
                t, t.shape, t.dtype, self._group_size, group_key, instance_key
            )
        )
      with ops.device(self._devices[1]):
        t = constant_op.constant(tensor_value)
        collectives.append(
            collective_ops.broadcast_recv(
                t.shape, t.dtype, self._group_size, group_key, instance_key
            )
        )
      return collectives

    for result in run_basic_nccl_broadcast():
      self.assertAllClose(result, tensor_value, rtol=1e-5, atol=1e-5)

  def testNcclBroadcastDoubleRecv(self):
    # [FUNCTION DOC]
    # Issues a broadcast receive on both GPUs with no sender and expects an internal error
    # saying that no source was found.
    self._setup_context()

    tensor_value = [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1]
    group_key = 1
    instance_key = 1

    @def_function.function
    def run_nccl_broadcast_double_recv():
      # [FUNCTION DOC]
      # HELPER: compiled function that starts a broadcast receive on every test GPU with the
      # same keys and no send. Takes no arguments, returns nothing; running it is expected to
      # fail.
      for device in self._devices:
        with ops.device(device):
          t = constant_op.constant(tensor_value)
          collective_ops.broadcast_recv(
              t.shape, t.dtype, self._group_size, group_key, instance_key
          )

    with self.assertRaisesRegex(errors.InternalError, 'found no source'):
      run_nccl_broadcast_double_recv()

  def testNcclBroadcastDoubleSend(self):
    # [FUNCTION DOC]
    # Issues a broadcast send on both GPUs with the same keys and expects an internal error
    # saying the instance already has a source.
    self._setup_context()

    tensor_value = [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1]
    group_key = 1
    instance_key = 1

    @def_function.function
    def run_nccl_broadcast_double_send():
      # [FUNCTION DOC]
      # HELPER: compiled function that starts a broadcast send on every test GPU with the same
      # keys. Takes no arguments, returns nothing; running it is expected to fail.
      for device in self._devices:
        with ops.device(device):
          t = constant_op.constant(tensor_value)
          collective_ops.broadcast_send(
              t, t.shape, t.dtype, self._group_size, group_key, instance_key
          )

    with self.assertRaisesRegex(errors.InternalError, 'already has source'):
      run_nccl_broadcast_double_send()

  def testBasicNcclAllGather(self):
    # [FUNCTION DOC]
    # All-gathers one float vector per GPU in a compiled function. Expects each GPU's result
    # to be the two input vectors concatenated in device order.
    self._setup_context()

    inputs = [
        [0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1],
        [0.3, 1.3, 2.3, 3.3, 4.3, 5.3, 6.3, 7.3],
    ]
    expected = [
        0.1,
        1.1,
        2.1,
        3.1,
        4.1,
        5.1,
        6.1,
        7.1,
        0.3,
        1.3,
        2.3,
        3.3,
        4.3,
        5.3,
        6.3,
        7.3,
    ]
    group_key = 1
    instance_key = 1

    @def_function.function
    def run_basic_nccl_all_gather():
      # [FUNCTION DOC]
      # HELPER: compiled function for the all-gather test. Takes no arguments; each GPU
      # contributes its constant to an all-gather in the shared group. Returns the list of
      # per-device gathered tensors.
      collectives = []
      for i in range(self._group_size):
        with ops.device(self._devices[i]):
          t = constant_op.constant(inputs[i])
          collectives.append(
              collective_ops.all_gather(
                  t, self._group_size, group_key, instance_key
              )
          )
      return collectives

    for result in run_basic_nccl_all_gather():
      self.assertAllClose(result, expected, rtol=1e-5, atol=1e-5)

  def testCollectiveDeviceMismatch(self):
    # [FUNCTION DOC]
    # Joins one all-reduce on the CPU and one on a GPU to the same group and instance and
    # expects an internal error mentioning that the group has a different device type.
    self._setup_context()

    group_key = 10
    instance_key = 20
    t0 = [1, 2, 3, 4]
    t1 = [5, 6, 7, 8]

    @def_function.function
    def run_collective_device_mismatch():
      # [FUNCTION DOC]
      # HELPER: compiled function placing one all-reduce on the CPU and another on GPU 0, with
      # the same group and instance keys. Takes no arguments, returns nothing; running it is
      # expected to fail.
      with ops.device('/CPU:0'):
        in0 = constant_op.constant(t0)
        collective_ops.all_reduce(
            in0, self._group_size, group_key, instance_key, 'Add', 'Id'
        )
      with ops.device('/GPU:0'):
        in1 = constant_op.constant(t1)
        collective_ops.all_reduce(
            in1, self._group_size, group_key, instance_key, 'Add', 'Id'
        )

    with self.assertRaisesRegex(
        errors.InternalError, 'but that group has type'
    ):
      run_collective_device_mismatch()

  def testCollectiveReduceMinMax(self):
    # [FUNCTION DOC]
    # Runs NCCL all-reduce with the maximum and then the minimum merge operation over two GPUs
    # and checks that both results equal the element-wise maximum or minimum of the inputs.
    self._setup_context()

    @def_function.function
    def run_all_reduce(group_key, instance_key, merge_op):
      # [FUNCTION DOC]
      # HELPER: compiled function for the min/max test. Takes the group key, instance key and
      # merge operation name. Builds fixed vectors on GPU 0 and GPU 1 and all-reduces them with
      # the given merge operation, identity final step and NCCL hint. Returns both results.
      t0 = [1.0, 20.0, 3.0, 40.0, 5.0]
      t1 = [10.0, 2.0, 30.0, 4.0, 50.0]
      with ops.device('/GPU:0'):
        in0 = constant_op.constant(t0)
        c0 = collective_ops.all_reduce(
            in0,
            self._group_size,
            group_key,
            instance_key,
            merge_op,
            final_op='Id',
            communication_hint='nccl',
        )
      with ops.device('/GPU:1'):
        in1 = constant_op.constant(t1)
        c1 = collective_ops.all_reduce(
            in1,
            self._group_size,
            group_key,
            instance_key,
            merge_op,
            final_op='Id',
            communication_hint='nccl',
        )
      return c0, c1

    for combination in [
        ('Max', [10.0, 20.0, 30.0, 40.0, 50.0]),
        ('Min', [1.0, 2.0, 3.0, 4.0, 5.0]),
    ]:
      merge_op = combination[0]
      results = run_all_reduce(group_key=10, instance_key=20, merge_op=merge_op)
      expected = combination[1]
      for result in results:
        self.assertAllClose(result, expected, rtol=1e-5, atol=1e-5)

  def testNcclStress(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test skips itself unconditionally on its first statement (referencing a
    # bug about a driver/image change), so the stress loop below never runs.
    # Intended scenario: on a single GPU, issue a thousand single-member NCCL all-reduces.
    self.skipTest(
        'b/435404154: As we moved from NVIDIA CUDA base image to Ubuntu 22.04'
        ' with NVIDIA Driver 580 installed for RBE, this test is failing and'
        ' needs to be addressed as part of the bug.'
    )
    self._setup_context(num_gpus=1)

    num_iters = 1000
    for _ in range(num_iters):
      with ops.device('/device:GPU:0'):
        collective_ops.all_reduce(
            [1.0],
            group_size=1,
            group_key=0,
            instance_key=0,
            merge_op='Add',
            final_op='Id',
            communication_hint='NCCL',
        )

  @test_util.run_v2_only
  def testAbortNccl(self):
    # [FUNCTION DOC]
    # Runs only in v2 mode. After one successful two-GPU all-reduce completes group resolution,
    # a background thread aborts collective ops after two seconds with an unavailable error
    # ("peer down") while a second all-reduce hangs. Expects that call and a following one to
    # raise an unavailable error with that message. Then joins the thread, resets the context
    # and checks that the collective function works again.
    self._setup_context(num_gpus=2)

    group_size = 2
    group_key = 100
    instance_key = 100
    in_tensor = constant_op.constant(1.0)

    # First perform a normal collective to finish resolution.
    def collective_fn():
      # [FUNCTION DOC]
      # HELPER: issues an NCCL all-reduce of the shared input tensor on GPU 0 and GPU 1 with
      # the test's group and instance keys. Takes no arguments, returns nothing.
      for device in ['GPU:0', 'GPU:1']:
        with ops.device(device):
          collective_ops.all_reduce(
              in_tensor,
              group_size,
              group_key,
              instance_key,
              'Add',
              'Id',
              communication_hint='nccl',
          )

    def_function.function(collective_fn)()

    # Launch a collective that hangs, and abort the collective executor after
    # the launch.
    def abort_fn():
      # [FUNCTION DOC]
      # HELPER: thread target. Sleeps two seconds then aborts the collective executor in the
      # current context with an unavailable error and message "peer down". Returns nothing.
      time.sleep(2)
      context.context().abort_collective_ops(errors.UNAVAILABLE, 'peer down')

    t = threading.Thread(target=abort_fn)
    t.start()

    with self.assertRaisesRegex(errors.UnavailableError, 'peer down'):
      collective_ops.all_reduce(
          in_tensor,
          group_size,
          group_key,
          instance_key,
          'Add',
          'Id',
          communication_hint='nccl',
      )

    # After abortion, subsequent collectives should fail immediately.
    with self.assertRaisesRegex(errors.UnavailableError, 'peer down'):
      collective_ops.all_reduce(
          in_tensor,
          group_size,
          group_key,
          instance_key,
          'Add',
          'Id',
          communication_hint='nccl',
      )

    t.join()
    # Reset the context in order to reset the collective executor.
    context._reset_context()  # pylint: disable=protected-access
    def_function.function(collective_fn)()


if __name__ == '__main__':
  test.main()
