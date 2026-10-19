# [MODULE DOC]
# Unit tests for the v1 all-reduce subgraph builders. Covers flatten shape validation, ring
# permutation tables, the ring gather phase structure, and numeric correctness of ring, shuffle and
# recursive halving-doubling all-reduce (all on the local CPU, compared to a plain sum).
# Runs the standard test main when executed directly.
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
"""Tests for all_reduce."""

import time

import numpy as np

from tensorflow.core.framework import types_pb2
from tensorflow.python.distribute.v1 import all_reduce as ar
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging


class AllReduceTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Tests for the all-reduce builders. Holds helper methods that create input variables, build the
  # ring, shuffle and recursive halving-doubling reduction functions, and check that reduced
  # results (with a divide-by-device-count unary op) sum to the plain sum of inputs.

  @test_util.run_deprecated_v1
  def testFlattenTensorsShapesDefined(self):
    # [FUNCTION DOC]
    # Scenario: two references to a placeholder with an unknown dimension are passed to the
    # flatten helper. Expects a value error saying tensors must have statically known shape.
    x = array_ops.placeholder(types_pb2.DT_FLOAT, [None])
    with self.assertRaisesRegex(ValueError, "must have statically known shape"):
      ar._flatten_tensors([x, x])

  def testRingPermutations(self):
    # [FUNCTION DOC]
    # Checks the predecessor and rank tables from the ring permutation helper for many
    # combinations: zero devices (both empty), one or several workers, one or several subchunks,
    # and different GPU orderings. Asserts exact expected tables for each case.
    # 0 devices
    pred_by_c_d, rank_by_c_d = ar._ring_permutations(1, 0, [])
    self.assertEqual(pred_by_c_d, [])
    self.assertEqual(rank_by_c_d, [])
    # 1 worker, 1 subchunk cases
    pred_by_c_d, rank_by_c_d = ar._ring_permutations(1, 1, [0])
    self.assertEqual(pred_by_c_d, [[0]])
    self.assertEqual(rank_by_c_d, [[0]])
    pred_by_c_d, rank_by_c_d = ar._ring_permutations(1, 1, [0, 1, 2])
    self.assertEqual(pred_by_c_d, [[2, 0, 1]])
    self.assertEqual(rank_by_c_d, [[0, 1, 2]])
    # multiple workers, 1 subchunk cases
    pred_by_c_d, rank_by_c_d = ar._ring_permutations(2, 1, [0, 1, 2])
    self.assertEqual(pred_by_c_d, [[5, 0, 1, 2, 3, 4]])
    self.assertEqual(rank_by_c_d, [[0, 1, 2, 3, 4, 5]])
    pred_by_c_d, rank_by_c_d = ar._ring_permutations(3, 1, [0, 1, 2])
    self.assertEqual(pred_by_c_d, [[8, 0, 1, 2, 3, 4, 5, 6, 7]])
    self.assertEqual(rank_by_c_d, [[0, 1, 2, 3, 4, 5, 6, 7, 8]])
    pred_by_c_d, rank_by_c_d = ar._ring_permutations(2, 1, [2, 1, 0])
    self.assertEqual(pred_by_c_d, [[1, 2, 3, 4, 5, 0]])
    self.assertEqual(rank_by_c_d, [[2, 1, 0, 5, 4, 3]])
    # 1 worker, multiple subchunk cases
    pred_by_c_d, rank_by_c_d = ar._ring_permutations(1, 2, [0, 1, 2, 3])
    self.assertEqual(pred_by_c_d, [[3, 0, 1, 2], [3, 0, 1, 2]])
    self.assertEqual(rank_by_c_d, [[0, 1, 2, 3], [2, 3, 0, 1]])
    pred_by_c_d, rank_by_c_d = ar._ring_permutations(1, 4, [0, 1, 2, 3])
    self.assertEqual(pred_by_c_d, [[3, 0, 1, 2], [3, 0, 1, 2],
                                   [3, 0, 1, 2], [3, 0, 1, 2]])
    self.assertEqual(rank_by_c_d, [[0, 1, 2, 3], [3, 0, 1, 2],
                                   [2, 3, 0, 1], [1, 2, 3, 0]])
    # multiple worker, multiple subchunk cases
    pred_by_c_d, rank_by_c_d = ar._ring_permutations(2, 2, [0, 1, 2, 3])
    self.assertEqual(pred_by_c_d, [[7, 0, 1, 2, 3, 4, 5, 6],
                                   [3, 0, 5, 2, 7, 4, 1, 6]])
    self.assertEqual(rank_by_c_d, [[0, 1, 2, 3, 4, 5, 6, 7],
                                   [2, 3, 0, 1, 6, 7, 4, 5]])
    pred_by_c_d, rank_by_c_d = ar._ring_permutations(2, 2, [0, 3, 2, 1])
    self.assertEqual(pred_by_c_d, [[5, 2, 3, 0, 1, 6, 7, 4],
                                   [1, 2, 7, 0, 5, 6, 3, 4]])
    self.assertEqual(rank_by_c_d, [[0, 3, 2, 1, 4, 7, 6, 5],
                                   [2, 1, 0, 3, 6, 5, 4, 7]])

  def _buildInput(self, num_workers, num_gpus):
    # [FUNCTION DOC]
    # HELPER: creates test inputs for the ring gather test.
    # For each worker and GPU, makes an identity copy of a constant 16-element float vector placed
    # on a device named by task and GPU index. Returns the list of tensors and the list of device
    # name strings.
    t8 = constant_op.constant(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
        types_pb2.DT_FLOAT)
    input_tensors = []
    device_names = []
    for w in range(0, num_workers):
      for d in range(0, num_gpus):
        dn = "/replica:0/task:%d/device:GPU:%d" % (w, d % num_gpus)
        device_names.append(dn)
        with ops.device(dn):
          input_tensors.append(array_ops.identity(t8))
    return input_tensors, device_names

  @test_util.run_deprecated_v1
  def testBuildRingGatherPassStructure(self):
    # [FUNCTION DOC]
    # Scenario 1: one worker with one device; asserts the gather helper returns the inputs
    # unchanged. Scenario 2: one worker, four devices, two subchunks; asserts padding length is
    # zero, one output list per input, each with eight chunks, each chunk of length equal to the
    # tensor length divided by the chunk count.
    # 1 worker, 1 device
    input_tensors, device_names = self._buildInput(1, 1)
    pred_by_c_d, rank_by_c_d = ar._ring_permutations(1, 1, [0])
    output_tensors = ar._build_ring_gather(input_tensors, device_names, 1,
                                           pred_by_c_d, rank_by_c_d,
                                           math_ops.add)
    self.assertEqual(output_tensors, input_tensors)
    # 1 worker, 4 devices, 2 subchunks
    input_tensors, device_names = self._buildInput(1, 4)
    pred_by_c_d, rank_by_c_d = ar._ring_permutations(1, 2, [0, 1, 2, 3])
    output_tensors, pad_len = ar._build_ring_gather(
        input_tensors, device_names, 2, pred_by_c_d, rank_by_c_d, math_ops.add)
    self.assertEqual(0, pad_len)
    # same number outputs as inputs
    self.assertEqual(len(output_tensors), len(input_tensors))
    num_chunks = 2 * len(input_tensors)
    tlen = tensor_shape.dimension_value(input_tensors[0].shape[0])
    for otl in output_tensors:
      self.assertEqual(len(otl), num_chunks)
      for ot in otl:
        self.assertEqual(ot.shape, [tlen//num_chunks])

  def _buildInitialVars(self, shape, dev_list):
    # [FUNCTION DOC]
    # HELPER: creates one float variable of the given shape per device name and initializes it so
    # element i on device d holds i plus 0.01 times d. Runs the initialization ops immediately
    # (needs a default session). Returns the list of variables.
    values = []
    num_devices = len(dev_list)
    dim = np.prod(shape, dtype=int) if shape else 1
    for d in range(0, num_devices):
      with ops.device(dev_list[d]):
        npt = np.zeros(shape).astype(np.float32)
        alias = np.frombuffer(npt.data, dtype=np.float32)
        for i in range(0, dim):
          alias[i] = i + 0.01 * d
        var = state_ops.variable_op(shape, types_pb2.DT_FLOAT)
        state_ops.init_variable(var, npt).op.run()
        values.append(var)
    return values

  # pylint: disable=g-long-lambda

  def _buildRing(self, num_workers, num_gpus, subdiv):
    # [FUNCTION DOC]
    # HELPER: returns a function (tensors, unary op) that builds a ring all-reduce with addition
    # over the given worker count, subdivisions and the identity GPU ordering.
    gpu_perm = range(0, num_gpus)
    return lambda x, un_op: ar.build_ring_all_reduce(
        x, num_workers, subdiv, gpu_perm, math_ops.add, un_op)

  def _testAllReduce(self, num_workers, num_gpus, shape, build_f):
    # [FUNCTION DOC]
    # HELPER: shared correctness check for all algorithms.
    # Creates one variable per simulated device (all on the local CPU), builds the reduction with a
    # unary op dividing by the device count, then asserts the sum of the outputs is close to the
    # plain sum of the inputs.
    # Use local CPU as device for all inputs.
    num_devices = num_workers * num_gpus
    dev_list = ["/replica:0/task:0/device:CPU:0"
                for _ in range(num_devices)]
    with self.cached_session():
      input_tensors = self._buildInitialVars(shape, dev_list)
      un_op = lambda x: math_ops.div(
          x, constant_op.constant(num_devices, dtype=types_pb2.DT_FLOAT))
      simple_sum = math_ops.add_n(input_tensors)
      simple_sum.op.run()
      output_tensors = build_f(input_tensors, un_op)
      sum_reduced = math_ops.add_n(output_tensors)
      sum_reduced.op.run()
      self.assertAllClose(sum_reduced, self.evaluate(simple_sum))

  def _testRingAllReduce(self, num_workers, num_gpus, shape, subdiv):
    # [FUNCTION DOC]
    # HELPER: runs the correctness check for one ring configuration and logs the elapsed time.
    start_time = time.time()
    build_f = self._buildRing(num_workers, num_gpus, subdiv)
    self._testAllReduce(num_workers, num_gpus, shape, build_f)
    elapsed = time.time() - start_time
    tf_logging.info("RingAllReduce num_workers=%d num_gpus=%d shape=%s "
                    "subdiv=%d elapsed=%f" %
                    (num_workers, num_gpus, shape, subdiv, elapsed))

  @test_util.run_deprecated_v1
  def testRingAllReduce(self):
    # [FUNCTION DOC]
    # Runs the ring correctness check over scalar, vector, matrix and uneven shapes with various
    # worker, GPU and subdivision counts. One large case is commented out as too slow.
    self._testRingAllReduce(1, 2, [], 1)
    self._testRingAllReduce(1, 2, [8], 1)
    self._testRingAllReduce(1, 2, [4, 4], 1)
    self._testRingAllReduce(6, 1, [8], 1)
    self._testRingAllReduce(1, 8, [32], 1)
    self._testRingAllReduce(1, 8, [120], 1)
    self._testRingAllReduce(2, 8, [7, 13], 1)
    self._testRingAllReduce(2, 8, [8, 8], 2)
    self._testRingAllReduce(2, 8, [8, 8], 4)
    # TODO(tucker): The following test is surprisingly slow.
    # Diagnose and fix before re-enabling.
    # self._testRingAllReduce(4, 8, [8, 8, 2], 4)

  def _buildShuffle(self, num_workers, num_gpus, num_shards):
    # [FUNCTION DOC]
    # HELPER: returns a function (tensors, unary op) that builds a shuffle all-reduce using
    # n-ary addition with the given number of gather shards, all placed on the local CPU.
    # Use local CPU for all shuffle shards
    gather_devices = ["/replica:0/task:0/device:CPU:0"
                      for _ in range(num_shards)]
    return lambda x, un_op: ar.build_shuffle_all_reduce(
        x, gather_devices, math_ops.add_n, un_op)

  def _testShuffleAllReduce(self, num_workers, num_gpus, shape, num_shards):
    # [FUNCTION DOC]
    # HELPER: runs the correctness check for one shuffle configuration and logs elapsed time.
    start_time = time.time()
    build_f = self._buildShuffle(num_workers, num_gpus, num_shards)
    self._testAllReduce(num_workers, num_gpus, shape, build_f)
    elapsed = time.time() - start_time
    tf_logging.info("ShuffleAllReduce num_workers=%d num_gpus=%d shape=%s "
                    "elapsed=%f" % (num_workers, num_gpus, shape, elapsed))

  @test_util.run_deprecated_v1
  def testShuffleAllReduce(self):
    # [FUNCTION DOC]
    # Runs the shuffle correctness check over several shapes, device counts and shard counts.
    self._testShuffleAllReduce(1, 2, [], 1)
    self._testShuffleAllReduce(1, 2, [8], 1)
    self._testShuffleAllReduce(1, 2, [4, 4], 1)
    self._testShuffleAllReduce(1, 8, [32], 1)
    self._testShuffleAllReduce(1, 8, [120], 1)
    self._testShuffleAllReduce(2, 8, [7, 13], 3)
    self._testShuffleAllReduce(2, 8, [8, 8], 2)
    self._testShuffleAllReduce(2, 8, [8, 8], 4)
    self._testShuffleAllReduce(4, 8, [8, 8, 2], 4)

  def _buildRecursiveHD(self, num_workers, num_gpus):
    # [FUNCTION DOC]
    # HELPER: returns a function (tensors, unary op) that builds a recursive halving-doubling
    # all-reduce with addition. The worker and GPU counts are unused.
    return lambda x, un_op: ar.build_recursive_hd_all_reduce(
        x, math_ops.add, un_op)

  # pylint: enable=g-long-lambda

  def _testRecursiveHDAllReduce(self, num_workers, num_gpus, shape):
    # [FUNCTION DOC]
    # HELPER: runs the correctness check for one recursive halving-doubling configuration and
    # logs elapsed time.
    start_time = time.time()
    build_f = self._buildRecursiveHD(num_workers, num_gpus)
    self._testAllReduce(num_workers, num_gpus, shape, build_f)
    elapsed = time.time() - start_time
    tf_logging.info("RecursiveHDAllReduce num_workers=%d num_gpus=%d "
                    "shape=%s elapsed=%f" %
                    (num_workers, num_gpus, shape, elapsed))

  @test_util.run_deprecated_v1
  def testRecursiveHDAllReduce(self):
    # [FUNCTION DOC]
    # Runs the recursive halving-doubling correctness check over several shapes and device counts
    # (all powers of two).
    self._testRecursiveHDAllReduce(1, 2, [8])
    self._testRecursiveHDAllReduce(1, 2, [4, 4])
    self._testRecursiveHDAllReduce(1, 8, [32])
    self._testRecursiveHDAllReduce(1, 8, [120])
    self._testRecursiveHDAllReduce(2, 8, [8, 8])
    self._testRecursiveHDAllReduce(4, 8, [8, 8, 2])


if __name__ == "__main__":
  test.main()
