# [MODULE DOC]
# Kernel tests for the graph-mode StagingArea data structure (data_flow_ops.StagingArea).
# Covers put/get of single values, multiple values and named dictionaries, device colocation of
# the put and get ops, peeking at an element by index, size and clear, and blocking behavior when
# the capacity (element count) or memory limit is reached. Defines a module-level TIMEOUT of one
# second used when waiting on a producer thread. Running the file as a script starts the test
# runner. No import-time side effects beyond imports.
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
import queue
import threading

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

TIMEOUT = 1


class StageTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering data_flow_ops.StagingArea: staging and retrieving tensors, dictionaries of
  # named tensors, colocation, peek, size/clear, and capacity and memory-limit blocking. Most tests
  # build their own graph, finalize it and run it in a session; no shared fixtures.

  @test_util.run_deprecated_v1
  def testSimple(self):
    # [FUNCTION DOC]
    # Scenario: a staging area with one float tensor. A scaled 128x128 value is computed on the CPU,
    # put into the area and read back on the GPU device (or default device) where its square matrix
    # product maximum is computed. The graph is finalized, the area is primed with an input of -1,
    # then for ten steps put and get run together. Asserts each result matches the closed-form
    # expected value within a relative tolerance of 1e-4 (the get returns the previous step's put).
    with ops.Graph().as_default() as G:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32)
        v = 2. * (array_ops.zeros([128, 128]) + x)
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.StagingArea([dtypes.float32])
        stage = stager.put([v])
        y = stager.get()
        y = math_ops.reduce_max(math_ops.matmul(y, y))

    G.finalize()

    with self.session(graph=G) as sess:
      sess.run(stage, feed_dict={x: -1})
      for i in range(10):
        _, yval = sess.run([stage, y], feed_dict={x: i})
        self.assertAllClose(4 * (i - 1) * (i - 1) * 128, yval, rtol=1e-4)

  @test_util.run_deprecated_v1
  def testMultiple(self):
    # [FUNCTION DOC]
    # Scenario: a staging area with two float tensors (a scalar and a 128x128 matrix) staged as a
    # list. After priming with an input of -1, ten steps of put plus get run; the result combines
    # the scalar with the maximum of the matrix product. Asserts each value matches the closed-form
    # expectation within a relative tolerance of 1e-4.
    with ops.Graph().as_default() as G:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32)
        v = 2. * (array_ops.zeros([128, 128]) + x)
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.StagingArea([dtypes.float32, dtypes.float32])
        stage = stager.put([x, v])
        z, y = stager.get()
        y = math_ops.reduce_max(z * math_ops.matmul(y, y))

    G.finalize()

    with self.session(graph=G) as sess:
      sess.run(stage, feed_dict={x: -1})
      for i in range(10):
        _, yval = sess.run([stage, y], feed_dict={x: i})
        self.assertAllClose(
            4 * (i - 1) * (i - 1) * (i - 1) * 128, yval, rtol=1e-4)

  @test_util.run_deprecated_v1
  def testDictionary(self):
    # [FUNCTION DOC]
    # Scenario: same as the multiple-value test but the staging area is created with explicit
    # shapes and names, and values are put as a mapping from names to tensors; get returns a
    # mapping which is unpacked by name. Asserts the computed result over ten steps matches the
    # closed-form expectation within a relative tolerance of 1e-4.
    with ops.Graph().as_default() as G:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32)
        v = 2. * (array_ops.zeros([128, 128]) + x)
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.StagingArea(
            [dtypes.float32, dtypes.float32],
            shapes=[[], [128, 128]],
            names=['x', 'v'])
        stage = stager.put({'x': x, 'v': v})
        ret = stager.get()
        z = ret['x']
        y = ret['v']
        y = math_ops.reduce_max(z * math_ops.matmul(y, y))

    G.finalize()

    with self.session(graph=G) as sess:
      sess.run(stage, feed_dict={x: -1})
      for i in range(10):
        _, yval = sess.run([stage, y], feed_dict={x: i})
        self.assertAllClose(
            4 * (i - 1) * (i - 1) * (i - 1) * 128, yval, rtol=1e-4)

  def testColocation(self):
    # [FUNCTION DOC]
    # Scenario: builds a graph (no session run) where put is created under the GPU device scope
    # and get under the CPU scope. Asserts the put op's device equals the GPU device name (or the
    # canonical GPU device string when a GPU is present) and that the get op is on the CPU device.
    gpu_dev = test.gpu_device_name()

    with ops.Graph().as_default() as G:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32)
        v = 2. * (array_ops.zeros([128, 128]) + x)
      with ops.device(gpu_dev):
        stager = data_flow_ops.StagingArea([dtypes.float32])
        y = stager.put([v])
        expected_name = gpu_dev if 'gpu' not in gpu_dev else '/device:GPU:0'
        self.assertEqual(y.device, expected_name)
      with ops.device('/cpu:0'):
        x = stager.get()[0]
        self.assertEqual(x.device, '/device:CPU:0')

    G.finalize()

  @test_util.run_deprecated_v1
  def testPeek(self):
    # [FUNCTION DOC]
    # Scenario: a staging area of scalar int32 values; ten values are staged one per run. Then
    # peek is run with each index from zero to nine. Asserts the peeked element at index i equals
    # the value i that was staged. (The get op is built but never run.)
    with ops.Graph().as_default() as G:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.int32, name='x')
        p = array_ops.placeholder(dtypes.int32, name='p')
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.StagingArea(
            [
                dtypes.int32,
            ], shapes=[[]])
        stage = stager.put([x])
        peek = stager.peek(p)
        ret = stager.get()

    G.finalize()

    with self.session(graph=G) as sess:
      for i in range(10):
        sess.run(stage, feed_dict={x: i})

      for i in range(10):
        self.assertTrue(sess.run(peek, feed_dict={p: i}) == [i])

  def testPeekBadIndex(self):
    # [FUNCTION DOC]
    # Scenario: a staging area holding int32 vectors of length ten with one element staged.
    # Peeking with an empty (non-scalar) index must fail. Asserts a value error or an
    # invalid-argument error whose message says the index must be scalar.
    stager = data_flow_ops.StagingArea([
        dtypes.int32,
    ], shapes=[[10]])
    stager.put([array_ops.zeros([10], dtype=dtypes.int32)])

    with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError),
                                'must be scalar'):
      self.evaluate(stager.peek([]))

  @test_util.run_deprecated_v1
  def testSizeAndClear(self):
    # [FUNCTION DOC]
    # Scenario: a named two-tensor staging area. Stages one element and checks size is one, stages
    # another and checks size is two, runs clear and checks size is zero.
    with ops.Graph().as_default() as G:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32, name='x')
        v = 2. * (array_ops.zeros([128, 128]) + x)
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.StagingArea(
            [dtypes.float32, dtypes.float32],
            shapes=[[], [128, 128]],
            names=['x', 'v'])
        stage = stager.put({'x': x, 'v': v})
        ret = stager.get()
        size = stager.size()
        clear = stager.clear()

    G.finalize()

    with self.session(graph=G) as sess:
      sess.run(stage, feed_dict={x: -1})
      self.assertEqual(sess.run(size), 1)
      sess.run(stage, feed_dict={x: -1})
      self.assertEqual(sess.run(size), 2)
      sess.run(clear)
      self.assertEqual(sess.run(size), 0)

  @test_util.run_deprecated_v1
  def testCapacity(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test immediately skips itself (flaky on GPU, bug reference in the skip
    # message), so everything after the skip call is dead code and never runs.
    # Scenario: a scalar int32 staging area with capacity three. A background thread stages eight
    # values, blocking once full; the main thread drains a token queue until a one-second timeout.
    # Expected: the timeout occurs at iteration equal to the capacity, size equals the capacity,
    # and then eight gets return values in order, leaving the area empty.
    self.skipTest('b/123423516 this test is flaky on gpu.')
    capacity = 3

    with ops.Graph().as_default() as G:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.int32, name='x')
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.StagingArea(
            [
                dtypes.int32,
            ], capacity=capacity, shapes=[[]])
        stage = stager.put([x])
        ret = stager.get()
        size = stager.size()

    G.finalize()

    value_queue = queue.Queue()
    n = 8

    with self.session(graph=G) as sess:
      # Stage data in a separate thread which will block when it hits the
      # staging area's capacity and thus not fill the value_queue with n tokens
      def thread_run():
        # [FUNCTION DOC]
        # HELPER: producer body for the capacity test's background thread.
        # Stages the integers zero to n-1 one at a time (feeding each) and after each successful
        # stage puts a token on the shared queue. Blocks inside the run call once the staging area
        # is full. Takes no arguments, returns nothing.
        for i in range(n):
          sess.run(stage, feed_dict={x: i})
          value_queue.put(0)

      t = threading.Thread(target=thread_run)
      t.daemon = True
      t.start()

      # Get tokens from the value_queue until a timeout occurs
      try:
        for i in range(n):
          value_queue.get(timeout=TIMEOUT)
      except queue.Empty:
        pass

      # Should've timed out on the iteration 'capacity'
      if not i == capacity:
        self.fail("Expected to timeout on iteration '{}' "
                  "but instead timed out on iteration '{}' "
                  "Staging Area size is '{}' and configured "
                  "capacity is '{}'.".format(capacity, i, sess.run(size),
                                             capacity))

      # Should have capacity elements in the staging area
      self.assertTrue(sess.run(size) == capacity)

      # Clear the staging area completely
      for i in range(n):
        self.assertTrue(sess.run(ret) == [i])

      # It should now be empty
      self.assertTrue(sess.run(size) == 0)

  @test_util.run_deprecated_v1
  def testMemoryLimit(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: a comment on the chunk size says 256K but the value is 200 KiB; the derived
    # capacity is therefore two chunks (512 KiB limit divided by 200 KiB, rounded down).
    # Scenario: a uint8 staging area with a 512 KiB memory limit. A background thread stages eight
    # 200 KiB arrays, blocking when the limit is reached; the main thread drains a token queue until
    # a one-second timeout. Expected: timeout at the iteration equal to the derived capacity, size
    # equals that capacity, then eight gets return arrays filled with the staged index, leaving
    # the area empty.
    memory_limit = 512 * 1024  # 512K
    chunk = 200 * 1024  # 256K
    capacity = memory_limit // chunk

    with ops.Graph().as_default() as G:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.uint8, name='x')
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.StagingArea(
            [
                dtypes.uint8,
            ], memory_limit=memory_limit, shapes=[[]])
        stage = stager.put([x])
        ret = stager.get()
        size = stager.size()

    G.finalize()

    value_queue = queue.Queue()
    n = 8

    with self.session(graph=G) as sess:
      # Stage data in a separate thread which will block when it hits the
      # staging area's capacity and thus not fill the value_queue with n tokens
      def thread_run():
        # [FUNCTION DOC]
        # HELPER: producer body for the memory-limit test's background thread.
        # Stages n uint8 arrays, each filled with its index and of the chunk size, and after each
        # successful stage puts a token on the shared queue. Blocks once the memory limit is hit.
        # Takes no arguments, returns nothing.
        for i in range(n):
          sess.run(stage, feed_dict={x: np.full(chunk, i, dtype=np.uint8)})
          value_queue.put(0)

      t = threading.Thread(target=thread_run)
      t.daemon = True
      t.start()

      # Get tokens from the value_queue until a timeout occurs
      try:
        for i in range(n):
          value_queue.get(timeout=TIMEOUT)
      except queue.Empty:
        pass

      # Should've timed out on the iteration 'capacity'
      if not i == capacity:
        self.fail("Expected to timeout on iteration '{}' "
                  "but instead timed out on iteration '{}' "
                  "Staging Area size is '{}' and configured "
                  "capacity is '{}'.".format(capacity, i, sess.run(size),
                                             capacity))

      # Should have capacity elements in the staging area
      self.assertTrue(sess.run(size) == capacity)

      # Clear the staging area completely
      for i in range(n):
        self.assertTrue(np.all(sess.run(ret)[0] == i))

      self.assertTrue(sess.run(size) == 0)


if __name__ == '__main__':
  test.main()
