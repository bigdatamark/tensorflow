# [MODULE DOC]
# Tests for virtual GPU support (several logical GPUs on one physical GPU). Contains a helper
# class that builds session config and random multi-device add graphs, and a test case that
# checks device names in step stats and that random graphs match CPU-only results. GPU tests
# skip when no CUDA GPU is available.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for multiple virtual GPU support."""

import random

import numpy as np

from google.protobuf import text_format
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging


class VirtualGpuTestUtil(object):
  # [CLASS DOC]
  # HELPER: test utility for the virtual GPU tests. Holds matrix size, op count, virtual devices
  # per physical GPU, device name list (CPU first, then GPUs), a session config with virtual
  # devices of 2GB each, and cumulative per-device probabilities used to place random add ops.

  def __init__(self,
               dim=1000,
               num_ops=100,
               virtual_devices_per_gpu=None,
               device_probabilities=None):
    # [FUNCTION DOC]
    # Configures the utility. dim: matrix side length (default 1000). num_ops: number of random
    # add ops (default 100). virtual_devices_per_gpu: list with virtual device count per physical
    # GPU (default one GPU with three). device_probabilities: optional per-device weights
    # (default uniform), converted to cumulative values with the last forced above one. Builds the
    # session config and logs all settings. Returns nothing.
    self._dim = dim
    self._num_ops = num_ops
    if virtual_devices_per_gpu is None:
      self._virtual_devices_per_gpu = [3]
    else:
      self._virtual_devices_per_gpu = virtual_devices_per_gpu
    self._visible_device_list = [
        i for i in range(len(self._virtual_devices_per_gpu))
    ]
    gpu_devices = [
        ('/gpu:' + str(i)) for i in range(sum(self._virtual_devices_per_gpu))
    ]
    self.devices = ['/cpu:0'] + gpu_devices
    self._num_devices = len(self.devices)
    # Each virtual device gets 2GB memory.
    self._mem_limits_mb = [
        ([1 << 11] * i) for i in self._virtual_devices_per_gpu
    ]
    self.config = self._GetSessionConfig()

    if device_probabilities is not None:
      self._device_probabilities = list(device_probabilities)  # Deep copy
      for i in range(1, self._num_devices):
        self._device_probabilities[i] += self._device_probabilities[i - 1]
    else:
      # Each device gets same probability to be assigned an operation.
      step = 1.0 / self._num_devices
      self._device_probabilities = [
          (x + 1) * step for x in range(self._num_devices)
      ]
    # To prevent rounding error causing problems.
    self._device_probabilities[self._num_devices - 1] = 1.1

    logging.info('dim: %d', self._dim)
    logging.info('num_ops: %d', self._num_ops)
    logging.info('visible_device_list: %s', str(self._visible_device_list))
    logging.info('virtual_devices_per_gpu: %s',
                 str(self._virtual_devices_per_gpu))
    logging.info('mem_limits: %s', str(self._mem_limits_mb))
    logging.info('devices: %s', str(self.devices))
    logging.info('config: %s', text_format.MessageToString(self.config))
    logging.info('device_probabilities: %s', str(self._device_probabilities))

  # Creates virtual GPU devices
  def _GetSessionConfig(self):
    # [FUNCTION DOC]
    # HELPER: returns a session configuration whose GPU options make the listed physical GPUs
    # visible and split each into virtual devices with the configured memory limits.
    # KNOWN ISSUE: passes each per-GPU list of limits as a single memory limit value where a
    # number is expected; only works as intended if the field accepts it.
    virtual_device_gpu_options = config_pb2.GPUOptions(
        visible_device_list=','.join(str(d) for d in self._visible_device_list),
        experimental=config_pb2.GPUOptions.Experimental(virtual_devices=[
            config_pb2.GPUOptions.Experimental.VirtualDevices(
                memory_limit_mb=i) for i in self._mem_limits_mb
        ]))
    return config_pb2.ConfigProto(gpu_options=virtual_device_gpu_options)

  # Generates a list of 3-tuples, each tuple contains the source and destination
  # device index for a binary operation like 'add', like:
  # (src_device_1, src_device_2, dst_device)
  def _GenerateOperationPlacement(self):
    # [FUNCTION DOC]
    # HELPER: randomly picks, for each of the configured number of ops, three device indices
    # (two sources and a destination) by sampling the cumulative probabilities. Returns a list of
    # index tuples.
    result = []
    for unused_i in range(self._num_ops):
      op_device = ()
      for unused_j in range(3):
        random_num = random.random()
        for device_index in range(self._num_devices):
          if self._device_probabilities[device_index] > random_num:
            op_device += (device_index,)
            break
      result.append(op_device)
    return result

  # Logs part of the matrix for debugging purposes.
  def _LogMatrix(self, mat, dim):
    # [FUNCTION DOC]
    # HELPER: logs the top-left block (at most 10 by 10) of a matrix for debugging. Returns
    # nothing.
    logging.info('---- printing the first 10*10 submatrix ----')
    for i in range(min(10, dim)):
      row = ''
      for j in range(min(10, dim)):
        row += ' ' + str(mat[i][j])
      logging.info(row)

  # Runs a list of 'add' operations where each operation satisfies the device
  # placement constraints in `op_placement`, and returns the result.
  def _TestRandomGraphWithDevices(self,
                                  sess,
                                  seed,
                                  op_placement,
                                  devices,
                                  debug_mode=False):
    # [FUNCTION DOC]
    # HELPER: builds random placeholder matrices (seeded by seed plus device index) on each given
    # device, applies the add ops from the placement list on their destination devices, sums all
    # matrices on the CPU, optionally logs the graph (debug_mode, default false), runs it in the
    # given session, logs the result block and returns the resulting matrix.
    data = []
    shape = (self._dim, self._dim)
    feed_dict = {}
    # Initialize the matrices
    for i in range(len(devices)):
      with ops.device(devices[i]):
        var = array_ops.placeholder(dtypes.float32, shape=shape)
        np.random.seed(seed + i)
        feed_dict[var] = np.random.uniform(
            low=0, high=0.1, size=shape).astype(np.float32)
        data.append(var)
    # Run the 'add' operations on those matrices
    for op in op_placement:
      with ops.device(devices[op[2]]):
        data[op[2]] = math_ops.add(data[op[0]], data[op[1]])
    with ops.device('/cpu:0'):
      s = data[0]
      for i in range(1, len(data)):
        s = math_ops.add(s, data[i])
    if debug_mode:
      logging.info(ops.get_default_graph().as_graph_def())
    result = sess.run(s, feed_dict=feed_dict)
    self._LogMatrix(result, self._dim)
    return result

  # Generates a random graph with `self._num_ops` 'add' operations with each
  # operation placed on different virtual device, test that the result is
  # identical to the result obtained by running the same graph on cpu only.
  def TestRandomGraph(self, sess, op_placement=None, random_seed=None):
    # [FUNCTION DOC]
    # Runs the same random add graph once across virtual GPU devices and once on CPU only, and
    # compares results elementwise. sess: session to use. op_placement: optional list of device
    # index triples (default random). random_seed: optional seed (default random). Supplying
    # either enables debug mode. Logs mismatch details and returns false on the first difference,
    # otherwise returns true.
    debug_mode = False
    if op_placement is None:
      op_placement = self._GenerateOperationPlacement()
    else:
      debug_mode = True
    if random_seed is None:
      random_seed = random.randint(0, 1 << 31)
    else:
      debug_mode = True
    logging.info('Virtual gpu functional test for random graph...')
    logging.info('operation placement: %s', str(op_placement))
    logging.info('random seed: %d', random_seed)

    # Run with multiple virtual gpus.
    result_vgd = self._TestRandomGraphWithDevices(
        sess, random_seed, op_placement, self.devices, debug_mode=debug_mode)
    # Run with single cpu.
    result_cpu = self._TestRandomGraphWithDevices(
        sess,
        random_seed,
        op_placement, ['/cpu:0'] * self._num_devices,
        debug_mode=debug_mode)
    # Test the result
    for i in range(self._dim):
      for j in range(self._dim):
        if result_vgd[i][j] != result_cpu[i][j]:
          logging.error(
              'Result mismatch at row %d column %d: expected %f, actual %f', i,
              j, result_cpu[i][j], result_vgd[i][j])
          logging.error('Devices: %s', self.devices)
          logging.error('Memory limits (in MB): %s', self._mem_limits_mb)
          return False
    return True


class VirtualGpuTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for virtual GPU devices; holds one VirtualGpuTestUtil instance as shared fixture.

  def __init__(self, method_name):
    # [FUNCTION DOC]
    # Passes the test method name to the base test case and creates the default utility helper
    # (three virtual devices on one GPU). Returns nothing.
    super(VirtualGpuTest, self).__init__(method_name)
    self._util = VirtualGpuTestUtil()

  @test_util.deprecated_graph_mode_only
  def testStatsContainAllDeviceNames(self):
    # [FUNCTION DOC]
    # Graph-mode test, skipped without a CUDA GPU. Inside a session with virtual devices, creates
    # a variable on each device, sums them, runs with full tracing, and asserts the step stats
    # list the CPU and all three virtual GPU devices.
    with self.session(config=self._util.config) as sess:
      # TODO(laigd): b/70811538. The is_gpu_available() call will invoke
      # DeviceFactory::AddDevices() with a default SessionOption, which prevents
      # adding virtual devices in the future, thus must be called within a
      # context of a session within which virtual devices are created. Same in
      # the following test case.
      if not test.is_gpu_available(cuda_only=True):
        self.skipTest('No GPU available')
      run_options = config_pb2.RunOptions(
          trace_level=config_pb2.RunOptions.FULL_TRACE)
      run_metadata = config_pb2.RunMetadata()

      mat_shape = [10, 10]
      data = []
      for d in self._util.devices:
        with ops.device(d):
          var = variables.Variable(random_ops.random_uniform(mat_shape))
          self.evaluate(var.initializer)
          data.append(var)
      s = data[0]
      for i in range(1, len(data)):
        s = math_ops.add(s, data[i])
      sess.run(s, options=run_options, run_metadata=run_metadata)

    self.assertTrue(run_metadata.HasField('step_stats'))
    step_stats = run_metadata.step_stats
    devices = [d.device for d in step_stats.dev_stats]
    self.assertTrue('/job:localhost/replica:0/task:0/device:CPU:0' in devices)
    self.assertTrue('/job:localhost/replica:0/task:0/device:GPU:0' in devices)
    self.assertTrue('/job:localhost/replica:0/task:0/device:GPU:1' in devices)
    self.assertTrue('/job:localhost/replica:0/task:0/device:GPU:2' in devices)

  @test_util.deprecated_graph_mode_only
  def testLargeRandomGraph(self):
    # [FUNCTION DOC]
    # Graph-mode test, skipped without a CUDA GPU. Runs up to five random-graph comparisons and
    # stops at the first mismatch.
    # KNOWN ISSUE: on mismatch it just returns instead of failing, so the test cannot fail.
    with self.session(config=self._util.config) as sess:
      if not test.is_gpu_available(cuda_only=True):
        self.skipTest('No GPU available')
      for _ in range(5):
        if not self._util.TestRandomGraph(sess):
          return


if __name__ == '__main__':
  test.main()
