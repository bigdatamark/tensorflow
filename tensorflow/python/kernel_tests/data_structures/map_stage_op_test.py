# [MODULE DOC]
# Test module for the map staging area (MapStagingArea in the data flow ops) and the underlying
# map stage, map peek and map unstage ops, including the ordered variant.
# It covers put/get/peek of tuples and named dictionaries, size and clear, capacity and memory-limit
# blocking, key ordering, partial (incremental) insertion and partial gets, device colocation, and
# validation that keys must be scalar 64-bit integers.
# Defines a module-level timeout constant (seconds) and one test class; running the file as a script
# starts the standard test runner. No import-time side effects beyond imports.
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

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

TIMEOUT = 1


class MapStageTest(test.TestCase):
  # [CLASS DOC]
  # Test case for the map staging area. Most tests build a fresh graph pinned to the CPU for inputs and
  # to the GPU device name (or the default device if none) for the staging area, finalize it, and then
  # drive it in a session with fed placeholders. Several tests are limited to deprecated graph-mode
  # execution; the colocation and the non-scalar-key peek and unstage tests run without that decorator.
  # Holds no instance state; shares the module timeout constant for the blocking tests.

  @test_util.run_deprecated_v1
  def testSimple(self):
    # [FUNCTION DOC]
    # Test: basic put and get on a staging area holding one float tensor. A value derived from a
    # placeholder is staged under a key, then the key is fetched while staging the next one. Over ten
    # iterations the result of squaring the matrix and taking its maximum must match the expected number
    # (close within a relative tolerance), showing values pass through in order.
    with ops.Graph().as_default() as g:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32)
        pi = array_ops.placeholder(dtypes.int64)
        gi = array_ops.placeholder(dtypes.int64)
        v = 2. * (array_ops.zeros([128, 128]) + x)
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.MapStagingArea([dtypes.float32])
        stage = stager.put(pi, [v], [0])
        k, y = stager.get(gi)
        y = math_ops.reduce_max(math_ops.matmul(y, y))

    g.finalize()

    with self.session(graph=g) as sess:
      sess.run(stage, feed_dict={x: -1, pi: 0})
      for i in range(10):
        _, yval = sess.run([stage, y], feed_dict={x: i, pi: i + 1, gi: i})
        self.assertAllClose(4 * (i - 1) * (i - 1) * 128, yval, rtol=1e-4)

  @test_util.run_deprecated_v1
  def testMultiple(self):
    # [FUNCTION DOC]
    # Test: staging two float tensors per entry by index. The scalar and matrix components are put under
    # one key and fetched as a tuple; across ten iterations the combined computed result must match the
    # expected value within tolerance.
    with ops.Graph().as_default() as g:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32)
        pi = array_ops.placeholder(dtypes.int64)
        gi = array_ops.placeholder(dtypes.int64)
        v = 2. * (array_ops.zeros([128, 128]) + x)
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.MapStagingArea([dtypes.float32, dtypes.float32])
        stage = stager.put(pi, [x, v], [0, 1])
        k, (z, y) = stager.get(gi)
        y = math_ops.reduce_max(z * math_ops.matmul(y, y))

    g.finalize()

    with self.session(graph=g) as sess:
      sess.run(stage, feed_dict={x: -1, pi: 0})
      for i in range(10):
        _, yval = sess.run([stage, y], feed_dict={x: i, pi: i + 1, gi: i})
        self.assertAllClose(
            4 * (i - 1) * (i - 1) * (i - 1) * 128, yval, rtol=1e-4)

  @test_util.run_deprecated_v1
  def testDictionary(self):
    # [FUNCTION DOC]
    # Test: a staging area configured with names and shapes, so entries are put and returned as mappings
    # from names to tensors. Fetched components are combined and over ten iterations the result must match
    # the expected value within tolerance.
    with ops.Graph().as_default() as g:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32)
        pi = array_ops.placeholder(dtypes.int64)
        gi = array_ops.placeholder(dtypes.int64)
        v = 2. * (array_ops.zeros([128, 128]) + x)
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.MapStagingArea([dtypes.float32, dtypes.float32],
                                              shapes=[[], [128, 128]],
                                              names=['x', 'v'])
        stage = stager.put(pi, {'x': x, 'v': v})
        key, ret = stager.get(gi)
        z = ret['x']
        y = ret['v']
        y = math_ops.reduce_max(z * math_ops.matmul(y, y))

    g.finalize()

    with self.session(graph=g) as sess:
      sess.run(stage, feed_dict={x: -1, pi: 0})
      for i in range(10):
        _, yval = sess.run([stage, y], feed_dict={x: i, pi: i + 1, gi: i})
        self.assertAllClose(
            4 * (i - 1) * (i - 1) * (i - 1) * 128, yval, rtol=1e-4)

  def testColocation(self):
    # [FUNCTION DOC]
    # Test: device placement. The put op created under the GPU (or default) device must carry that device
    # name; the get, peek and keyless get ops created under the CPU device must all be on the CPU device.
    # No session is run; the graph is only built and finalized.
    gpu_dev = test.gpu_device_name()

    with ops.Graph().as_default() as g:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32)
        v = 2. * (array_ops.zeros([128, 128]) + x)
      with ops.device(gpu_dev):
        stager = data_flow_ops.MapStagingArea([dtypes.float32])
        y = stager.put(1, [v], [0])
        expected_name = gpu_dev if 'gpu' not in gpu_dev else '/device:GPU:0'
        self.assertEqual(y.device, expected_name)
      with ops.device('/cpu:0'):
        _, x = stager.get(1)
        y = stager.peek(1)[0]
        _, z = stager.get()
        self.assertEqual(x[0].device, '/device:CPU:0')
        self.assertEqual(y.device, '/device:CPU:0')
        self.assertEqual(z[0].device, '/device:CPU:0')

    g.finalize()

  @test_util.run_deprecated_v1
  def testPeek(self):
    # [FUNCTION DOC]
    # Test: peek does not remove entries. Ten integer entries are staged under keys zero to nine; peeking
    # each key returns its value, and the staging area size remains ten afterwards.
    with ops.Graph().as_default() as g:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.int32, name='x')
        pi = array_ops.placeholder(dtypes.int64)
        gi = array_ops.placeholder(dtypes.int64)
        p = array_ops.placeholder(dtypes.int32, name='p')
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.MapStagingArea([
            dtypes.int32,
        ], shapes=[[]])
        stage = stager.put(pi, [x], [0])
        peek = stager.peek(gi)
        size = stager.size()

    g.finalize()

    n = 10

    with self.session(graph=g) as sess:
      for i in range(n):
        sess.run(stage, feed_dict={x: i, pi: i})

      for i in range(n):
        self.assertEqual(sess.run(peek, feed_dict={gi: i})[0], i)

      self.assertEqual(sess.run(size), 10)

  @test_util.run_deprecated_v1
  def testSizeAndClear(self):
    # [FUNCTION DOC]
    # Test: size and clear on a dictionary-style staging area. Staging two entries gives sizes one and two;
    # running the clear op empties the area so the size is zero.
    with ops.Graph().as_default() as g:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32, name='x')
        pi = array_ops.placeholder(dtypes.int64)
        gi = array_ops.placeholder(dtypes.int64)
        v = 2. * (array_ops.zeros([128, 128]) + x)
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.MapStagingArea([dtypes.float32, dtypes.float32],
                                              shapes=[[], [128, 128]],
                                              names=['x', 'v'])
        stage = stager.put(pi, {'x': x, 'v': v})
        size = stager.size()
        clear = stager.clear()

    g.finalize()

    with self.session(graph=g) as sess:
      sess.run(stage, feed_dict={x: -1, pi: 3})
      self.assertEqual(sess.run(size), 1)
      sess.run(stage, feed_dict={x: -1, pi: 1})
      self.assertEqual(sess.run(size), 2)
      sess.run(clear)
      self.assertEqual(sess.run(size), 0)

  @test_util.run_deprecated_v1
  def testCapacity(self):
    # [FUNCTION DOC]
    # Test: capacity limit of a staging area (three entries). A background daemon thread stages eight
    # entries and reports each success through a queue; the main thread reads tokens until a timeout. The
    # timeout must occur exactly when the capacity is reached (otherwise the test fails with a message
    # giving the iteration, size and capacity), the size must equal the capacity, and after getting all
    # entries the size must be zero.
    capacity = 3

    with ops.Graph().as_default() as g:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.int32, name='x')
        pi = array_ops.placeholder(dtypes.int64, name='pi')
        gi = array_ops.placeholder(dtypes.int64, name='gi')
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.MapStagingArea([
            dtypes.int32,
        ],
                                              capacity=capacity,
                                              shapes=[[]])

      stage = stager.put(pi, [x], [0])
      get = stager.get()
      size = stager.size()

    g.finalize()

    value_queue = queue.Queue()
    n = 8

    with self.session(graph=g) as sess:
      # Stage data in a separate thread which will block when it hits the
      # staging area's capacity and thus not fill the value_queue with n tokens
      def thread_run():
        # [FUNCTION DOC]
        # HELPER: background worker for the capacity test, which stages entries without blocking the test.
        # Nested function with no arguments and no return value. Runs the stage op once for each of the eight
        # entries, feeding the loop number as value and key, and after each success puts a token on the shared
        # queue. It blocks inside the session run once the staging area is full. Side effects: session runs
        # and queue writes.
        for i in range(n):
          sess.run(stage, feed_dict={x: i, pi: i})
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
      self.assertEqual(sess.run(size), capacity)

      # Clear the staging area completely
      for i in range(n):
        sess.run(get)

      self.assertEqual(sess.run(size), 0)

  @test_util.run_deprecated_v1
  def testMemoryLimit(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the comment on the chunk size says 256K but the chunk is 200 times 1024 bytes
    # (200K); the derived capacity of two entries comes from the actual 200K value.
    # Test: memory limit of a staging area (512K limit, 200K entries so room for two). A background thread
    # stages eight byte-array entries; the main thread reads queue tokens until a timeout, which must occur
    # at the entry equal to the computed capacity (else the test fails with a descriptive message). The size
    # must then equal that capacity, and after getting all entries the size must be zero.
    memory_limit = 512 * 1024  # 512K
    chunk = 200 * 1024  # 256K
    capacity = memory_limit // chunk

    with ops.Graph().as_default() as g:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.uint8, name='x')
        pi = array_ops.placeholder(dtypes.int64, name='pi')
        gi = array_ops.placeholder(dtypes.int64, name='gi')
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.MapStagingArea([dtypes.uint8],
                                              memory_limit=memory_limit,
                                              shapes=[[]])
        stage = stager.put(pi, [x], [0])
        get = stager.get()
        size = stager.size()

    g.finalize()

    value_queue = queue.Queue()
    n = 8

    with self.session(graph=g) as sess:
      # Stage data in a separate thread which will block when it hits the
      # staging area's capacity and thus not fill the value_queue with n tokens
      def thread_run():
        # [FUNCTION DOC]
        # HELPER: background worker for the memory-limit test, which stages entries without blocking the test.
        # Nested function with no arguments and no return value. For each of eight iterations it builds an
        # array of the chunk size filled with the iteration number (unsigned 8-bit), stages it under that
        # key via the session, then puts a token on the shared queue. It blocks once the memory limit is hit.
        # Side effects: session runs and queue writes.
        for i in range(n):
          data = np.full(chunk, i, dtype=np.uint8)
          sess.run(stage, feed_dict={x: data, pi: i})
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
      self.assertEqual(sess.run(size), capacity)

      # Clear the staging area completely
      for i in range(n):
        sess.run(get)

      self.assertEqual(sess.run(size), 0)

  @test_util.run_deprecated_v1
  def testOrdering(self):
    # [FUNCTION DOC]
    # Test: an ordered staging area returns entries in ascending key order. Ten entries are staged with
    # keys in descending order; the size is ten; each keyless get returns the next smallest key whose
    # value equals the key; finally the size is zero. Imports the random module locally but does not use it.
    import random

    with ops.Graph().as_default() as g:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.int32, name='x')
        pi = array_ops.placeholder(dtypes.int64, name='pi')
        gi = array_ops.placeholder(dtypes.int64, name='gi')
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.MapStagingArea([
            dtypes.int32,
        ],
                                              shapes=[[]],
                                              ordered=True)
        stage = stager.put(pi, [x], [0])
        get = stager.get()
        size = stager.size()

    g.finalize()

    n = 10

    with self.session(graph=g) as sess:
      # Keys n-1..0
      keys = list(reversed(range(n)))

      for i in keys:
        sess.run(stage, feed_dict={pi: i, x: i})

      self.assertEqual(sess.run(size), n)

      # Check that key, values come out in ascending order
      for i, k in enumerate(reversed(keys)):
        get_key, values = sess.run(get)
        self.assertTrue(i == k == get_key == values)

      self.assertEqual(sess.run(size), 0)

  @test_util.run_deprecated_v1
  def testPartialDictInsert(self):
    # [FUNCTION DOC]
    # Test: incremental insertion by name. Part of the named fields (x and f) is staged under keys zero
    # and one, then the remaining field (v) is staged separately. Entries count as incomplete until all
    # fields are present: sizes (complete, incomplete) go from (0,0) to (0,1) to (0,2); completing key zero
    # gives (1,1) and a get returns the full mapping for it, leaving (0,1); completing key one then lets
    # its full mapping be fetched too.
    with ops.Graph().as_default() as g:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32)
        f = array_ops.placeholder(dtypes.float32)
        v = array_ops.placeholder(dtypes.float32)
        pi = array_ops.placeholder(dtypes.int64)
        gi = array_ops.placeholder(dtypes.int64)
      with ops.device(test.gpu_device_name()):
        # Test barrier with dictionary
        stager = data_flow_ops.MapStagingArea(
            [dtypes.float32, dtypes.float32, dtypes.float32],
            names=['x', 'v', 'f'])
        stage_xf = stager.put(pi, {'x': x, 'f': f})
        stage_v = stager.put(pi, {'v': v})
        key, ret = stager.get(gi)
        size = stager.size()
        isize = stager.incomplete_size()

    g.finalize()

    with self.session(graph=g) as sess:
      # 0 complete and incomplete entries
      self.assertEqual(sess.run([size, isize]), [0, 0])
      # Stage key 0, x and f tuple entries
      sess.run(stage_xf, feed_dict={pi: 0, x: 1, f: 2})
      self.assertEqual(sess.run([size, isize]), [0, 1])
      # Stage key 1, x and f tuple entries
      sess.run(stage_xf, feed_dict={pi: 1, x: 1, f: 2})
      self.assertEqual(sess.run([size, isize]), [0, 2])

      # Now complete key 0 with tuple entry v
      sess.run(stage_v, feed_dict={pi: 0, v: 1})
      # 1 complete and 1 incomplete entry
      self.assertEqual(sess.run([size, isize]), [1, 1])
      # We can now obtain tuple associated with key 0
      self.assertEqual(
          sess.run([key, ret], feed_dict={gi: 0}),
          [0, {
              'x': 1,
              'f': 2,
              'v': 1
          }])

      # 0 complete and 1 incomplete entry
      self.assertEqual(sess.run([size, isize]), [0, 1])
      # Now complete key 1 with tuple entry v
      sess.run(stage_v, feed_dict={pi: 1, v: 3})
      # We can now obtain tuple associated with key 1
      self.assertEqual(
          sess.run([key, ret], feed_dict={gi: 1}),
          [1, {
              'x': 1,
              'f': 2,
              'v': 3
          }])

  @test_util.run_deprecated_v1
  def testPartialIndexInsert(self):
    # [FUNCTION DOC]
    # Test: the same incremental scenario as the named version, but fields are addressed by index
    # positions (zero and two first, then one). Verifies complete and incomplete sizes after each step and
    # that fetched lists have all three components in index order once complete.
    with ops.Graph().as_default() as g:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32)
        f = array_ops.placeholder(dtypes.float32)
        v = array_ops.placeholder(dtypes.float32)
        pi = array_ops.placeholder(dtypes.int64)
        gi = array_ops.placeholder(dtypes.int64)
      with ops.device(test.gpu_device_name()):
        stager = data_flow_ops.MapStagingArea(
            [dtypes.float32, dtypes.float32, dtypes.float32])
        stage_xf = stager.put(pi, [x, f], [0, 2])
        stage_v = stager.put(pi, [v], [1])
        key, ret = stager.get(gi)
        size = stager.size()
        isize = stager.incomplete_size()

    g.finalize()

    with self.session(graph=g) as sess:
      # 0 complete and incomplete entries
      self.assertEqual(sess.run([size, isize]), [0, 0])
      # Stage key 0, x and f tuple entries
      sess.run(stage_xf, feed_dict={pi: 0, x: 1, f: 2})
      self.assertEqual(sess.run([size, isize]), [0, 1])
      # Stage key 1, x and f tuple entries
      sess.run(stage_xf, feed_dict={pi: 1, x: 1, f: 2})
      self.assertEqual(sess.run([size, isize]), [0, 2])

      # Now complete key 0 with tuple entry v
      sess.run(stage_v, feed_dict={pi: 0, v: 1})
      # 1 complete and 1 incomplete entry
      self.assertEqual(sess.run([size, isize]), [1, 1])
      # We can now obtain tuple associated with key 0
      self.assertEqual(sess.run([key, ret], feed_dict={gi: 0}), [0, [1, 1, 2]])

      # 0 complete and 1 incomplete entry
      self.assertEqual(sess.run([size, isize]), [0, 1])
      # Now complete key 1 with tuple entry v
      sess.run(stage_v, feed_dict={pi: 1, v: 3})
      # We can now obtain tuple associated with key 1
      self.assertEqual(sess.run([key, ret], feed_dict={gi: 1}), [1, [1, 3, 2]])

  @test_util.run_deprecated_v1
  def testPartialDictGetsAndPeeks(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: several comments disagree with what is asserted. After key one is completed the sizes
    # asserted are one complete and zero incomplete, but the comment says one complete and one incomplete;
    # the next comment says "still 1 complete and 1 incomplete" yet zero incomplete is asserted; and the
    # comment about the last fetch says x and f for key 1 although it fetches the v field.
    # Test: partial peeks and gets by name. After staging fields for keys zero and one, peek returns subsets
    # (x and f; v) without removing anything. Getting x and f for key zero removes only those; getting them
    # again raises an invalid-argument error saying the tensor at index zero for key zero has already been
    # removed. Getting v removes the rest of key zero. A keyless get returns the lowest remaining key's
    # x and f, and a keyed get returns its v; finally nothing is left.
    with ops.Graph().as_default() as g:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32)
        f = array_ops.placeholder(dtypes.float32)
        v = array_ops.placeholder(dtypes.float32)
        pi = array_ops.placeholder(dtypes.int64)
        pei = array_ops.placeholder(dtypes.int64)
        gi = array_ops.placeholder(dtypes.int64)
      with ops.device(test.gpu_device_name()):
        # Test barrier with dictionary
        stager = data_flow_ops.MapStagingArea(
            [dtypes.float32, dtypes.float32, dtypes.float32],
            names=['x', 'v', 'f'])
        stage_xf = stager.put(pi, {'x': x, 'f': f})
        stage_v = stager.put(pi, {'v': v})
        peek_xf = stager.peek(pei, ['x', 'f'])
        peek_v = stager.peek(pei, ['v'])
        key_xf, get_xf = stager.get(gi, ['x', 'f'])
        key_v, get_v = stager.get(gi, ['v'])
        pop_key_xf, pop_xf = stager.get(indices=['x', 'f'])
        pop_key_v, pop_v = stager.get(pi, ['v'])
        size = stager.size()
        isize = stager.incomplete_size()

    g.finalize()

    with self.session(graph=g) as sess:
      # 0 complete and incomplete entries
      self.assertEqual(sess.run([size, isize]), [0, 0])
      # Stage key 0, x and f tuple entries
      sess.run(stage_xf, feed_dict={pi: 0, x: 1, f: 2})
      self.assertEqual(sess.run([size, isize]), [0, 1])
      # Stage key 1, x and f tuple entries
      sess.run(stage_xf, feed_dict={pi: 1, x: 1, f: 2})
      self.assertEqual(sess.run([size, isize]), [0, 2])

      # Now complete key 0 with tuple entry v
      sess.run(stage_v, feed_dict={pi: 0, v: 1})
      # 1 complete and 1 incomplete entry
      self.assertEqual(sess.run([size, isize]), [1, 1])

      # We can now peek at 'x' and 'f' values associated with key 0
      self.assertEqual(sess.run(peek_xf, feed_dict={pei: 0}), {'x': 1, 'f': 2})
      # Peek at 'v' value associated with key 0
      self.assertEqual(sess.run(peek_v, feed_dict={pei: 0}), {'v': 1})
      # 1 complete and 1 incomplete entry
      self.assertEqual(sess.run([size, isize]), [1, 1])

      # We can now obtain 'x' and 'f' values associated with key 0
      self.assertEqual(
          sess.run([key_xf, get_xf], feed_dict={gi: 0}), [0, {
              'x': 1,
              'f': 2
          }])
      # Still have 1 complete and 1 incomplete entry
      self.assertEqual(sess.run([size, isize]), [1, 1])

      # We can no longer get 'x' and 'f' from key 0
      with self.assertRaises(errors.InvalidArgumentError) as cm:
        sess.run([key_xf, get_xf], feed_dict={gi: 0})

      exc_str = ("Tensor at index '0' for key '0' " 'has already been removed.')

      self.assertIn(exc_str, cm.exception.message)

      # Obtain 'v' value associated with key 0
      self.assertEqual(
          sess.run([key_v, get_v], feed_dict={gi: 0}), [0, {
              'v': 1
          }])
      # 0 complete and 1 incomplete entry
      self.assertEqual(sess.run([size, isize]), [0, 1])

      # Now complete key 1 with tuple entry v
      sess.run(stage_v, feed_dict={pi: 1, v: 1})
      # 1 complete and 1 incomplete entry
      self.assertEqual(sess.run([size, isize]), [1, 0])

      # Pop without key to obtain 'x' and 'f' values associated with key 1
      self.assertEqual(sess.run([pop_key_xf, pop_xf]), [1, {'x': 1, 'f': 2}])
      # still 1 complete and 1 incomplete entry
      self.assertEqual(sess.run([size, isize]), [1, 0])
      # We can now obtain 'x' and 'f' values associated with key 1
      self.assertEqual(
          sess.run([pop_key_v, pop_v], feed_dict={pi: 1}), [1, {
              'v': 1
          }])
      # Nothing is left
      self.assertEqual(sess.run([size, isize]), [0, 0])

  @test_util.run_deprecated_v1
  def testPartialIndexGets(self):
    # [FUNCTION DOC]
    # Test: partial gets by index. A complete three-field entry is staged under key zero (sizes one,
    # zero); getting indices zero and two returns those values and leaves the entry in place; getting
    # index one returns the last value and leaves the staging area empty (sizes zero, zero).
    with ops.Graph().as_default() as g:
      with ops.device('/cpu:0'):
        x = array_ops.placeholder(dtypes.float32)
        f = array_ops.placeholder(dtypes.float32)
        v = array_ops.placeholder(dtypes.float32)
        pi = array_ops.placeholder(dtypes.int64)
        pei = array_ops.placeholder(dtypes.int64)
        gi = array_ops.placeholder(dtypes.int64)
      with ops.device(test.gpu_device_name()):
        # Test again with partial index gets
        stager = data_flow_ops.MapStagingArea(
            [dtypes.float32, dtypes.float32, dtypes.float32])
        stage_xvf = stager.put(pi, [x, v, f], [0, 1, 2])
        key_xf, get_xf = stager.get(gi, [0, 2])
        key_v, get_v = stager.get(gi, [1])
        size = stager.size()
        isize = stager.incomplete_size()

    g.finalize()

    with self.session(graph=g) as sess:
      # Stage complete tuple
      sess.run(stage_xvf, feed_dict={pi: 0, x: 1, f: 2, v: 3})

      self.assertEqual(sess.run([size, isize]), [1, 0])

      # Partial get using indices
      self.assertEqual(
          sess.run([key_xf, get_xf], feed_dict={gi: 0}), [0, [1, 2]])

      # Still some of key 0 left
      self.assertEqual(sess.run([size, isize]), [1, 0])

      # Partial get of remaining index
      self.assertEqual(sess.run([key_v, get_v], feed_dict={gi: 0}), [0, [3]])

      # All gone
      self.assertEqual(sess.run([size, isize]), [0, 0])

  @test_util.run_deprecated_v1
  def testNonScalarKeyOrderedMap(self):
    # [FUNCTION DOC]
    # Test: the ordered map stage op rejects a non-scalar key. Calls the raw op with a key of shape one by
    # three; running it must raise an invalid-argument error saying the key must be an int64 scalar.
    with ops.Graph().as_default() as g:
      x = array_ops.placeholder(dtypes.float32)
      v = 2. * (array_ops.zeros([128, 128]) + x)
      t = data_flow_ops.gen_data_flow_ops.ordered_map_stage(
          key=constant_op.constant(value=[1], shape=(1, 3), dtype=dtypes.int64),
          indices=np.array([[6]]),
          values=[x, v],
          dtypes=[dtypes.int64],
          capacity=0,
          memory_limit=0,
          container='container1',
          shared_name='',
          name=None)

    g.finalize()

    with self.session(graph=g) as sess:
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  'key must be an int64 scalar'):
        sess.run(t, feed_dict={x: 1})

  @test_util.run_deprecated_v1
  def testNonScalarKeyUnorderedMap(self):
    # [FUNCTION DOC]
    # Test: the unordered map stage op rejects a non-scalar key. Calls the raw op with a key of shape one
    # by three; running it must raise an invalid-argument error saying the key must be an int64 scalar.
    with ops.Graph().as_default() as g:
      x = array_ops.placeholder(dtypes.float32)
      v = 2. * (array_ops.zeros([128, 128]) + x)
      t = data_flow_ops.gen_data_flow_ops.map_stage(
          key=constant_op.constant(value=[1], shape=(1, 3), dtype=dtypes.int64),
          indices=np.array([[6]]),
          values=[x, v],
          dtypes=[dtypes.int64],
          capacity=0,
          memory_limit=0,
          container='container1',
          shared_name='',
          name=None)

    g.finalize()

    with self.session(graph=g) as sess:
      with self.assertRaisesRegex(errors.InvalidArgumentError,
                                  'key must be an int64 scalar'):
        sess.run(t, feed_dict={x: 1})

  def testNonScalarKeyMapPeek(self):
    # [FUNCTION DOC]
    # Test: the map peek op rejects a non-scalar key. Calls the raw op with a key of shape one by three
    # and evaluates it; an invalid-argument error saying the key must be an int64 scalar is expected.
    with self.assertRaisesRegex(
        errors.InvalidArgumentError, 'key must be an int64 scalar'
    ):
      v = data_flow_ops.gen_data_flow_ops.map_peek(
          key=constant_op.constant(value=[1], shape=(1, 3), dtype=dtypes.int64),
          indices=np.array([[6]]),
          dtypes=[dtypes.int64],
          capacity=0,
          memory_limit=0,
          container='container1',
          shared_name='',
          name=None,
      )
      self.evaluate(v)

  def testNonScalarKeyMapUnStage(self):
    # [FUNCTION DOC]
    # Test: the map unstage op rejects a non-scalar key. Calls the raw op with a key of shape one by
    # three and evaluates it; an invalid-argument error saying the key must be an int64 scalar is expected.
    with self.assertRaisesRegex(
        errors.InvalidArgumentError, 'key must be an int64 scalar'
    ):
      v = data_flow_ops.gen_data_flow_ops.map_unstage(
          key=constant_op.constant(value=[1], shape=(1, 3), dtype=dtypes.int64),
          indices=np.array([[6]]),
          dtypes=[dtypes.int64],
          capacity=0,
          memory_limit=0,
          container='container1',
          shared_name='',
          name=None,
      )
      self.evaluate(v)


if __name__ == '__main__':
  test.main()
