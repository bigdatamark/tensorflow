# [MODULE DOC]
# Unit tests for the Barrier data-flow op (data_flow_ops.Barrier) in graph mode.
# Covers construction attributes, insert_many / take_many behavior, shape validation, small
# batches, close and cancel semantics, blocking and multi-threaded use, and errors when two
# barriers share a name but have incompatible component types or shapes.
# Not a helper module; it only defines one test class and a main entry point. No import-time
# side effects beyond imports.
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for barrier ops."""

import time

import numpy as np

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import test


class BarrierTest(test.TestCase):
  # [CLASS DOC]
  # Test case covering the Barrier op. Each test builds barriers inside a cached session (most are
  # restricted to graph mode with the deprecated-v1 decorator), enqueues keyed components with
  # insert_many, and reads them back with take_many while checking sizes, ordering-independent
  # contents, and error messages. Two private methods are shared scenario helpers parameterized
  # by whether pending enqueues are cancelled. Holds no instance state of its own.

  def testConstructorWithShapes(self):
    # [FUNCTION DOC]
    # Test: builds a Barrier with two float components of explicit shapes, a shared name and a name,
    # inside a fresh graph. Asserts the barrier reference is a tensor and that the underlying node
    # definition equals the expected text proto (capacity -1, component types, shapes, empty
    # container, shared name).
    with ops.Graph().as_default():
      b = data_flow_ops.Barrier(
          (dtypes.float32, dtypes.float32),
          shapes=((1, 2, 3), (8,)),
          shared_name="B",
          name="B")
    self.assertTrue(isinstance(b.barrier_ref, tensor.Tensor))
    self.assertProtoEquals("""
      name:'B' op:'Barrier'
      attr {
        key: "capacity"
        value {
          i: -1
        }
      }
      attr { key: 'component_types'
             value { list { type: DT_FLOAT type: DT_FLOAT } } }
      attr {
        key: 'shapes'
        value {
          list {
            shape {
              dim { size: 1 } dim { size: 2 } dim { size: 3 }
            }
            shape {
              dim { size: 8 }
            }
          }
        }
      }
      attr { key: 'container' value { s: "" } }
      attr { key: 'shared_name' value: { s: 'B' } }
      """, b.barrier_ref.op.node_def)

  @test_util.run_deprecated_v1
  def testInsertMany(self):
    # [FUNCTION DOC]
    # Test: with a two-component scalar barrier, checks the ready size has scalar-vector shape, then
    # inserts component 0 for three keys (ready size stays 0) and component 1 (ready size becomes 3).
    # Graph mode only.
    with self.cached_session():
      b = data_flow_ops.Barrier(
          (dtypes.float32, dtypes.float32), shapes=((), ()), name="B")
      size_t = b.ready_size()
      self.assertEqual([], size_t.get_shape())
      keys = [b"a", b"b", b"c"]
      insert_0_op = b.insert_many(0, keys, [10.0, 20.0, 30.0])
      insert_1_op = b.insert_many(1, keys, [100.0, 200.0, 300.0])

      self.assertEqual(self.evaluate(size_t), [0])
      insert_0_op.run()
      self.assertEqual(self.evaluate(size_t), [0])
      insert_1_op.run()
      self.assertEqual(self.evaluate(size_t), [3])

  def testInsertManyEmptyTensor(self):
    # [FUNCTION DOC]
    # Test: constructing a Barrier whose declared shapes include an empty (zero-sized) shape must
    # raise a value error whose message says empty tensors are not supported and names index 1.
    with self.cached_session():
      error_message = ("Empty tensors are not supported, but received shape "
                       r"\'\(0,\)\' at index 1")
      with self.assertRaisesRegex(ValueError, error_message):
        data_flow_ops.Barrier(
            (dtypes.float32, dtypes.float32), shapes=((1,), (0,)), name="B")

  @test_util.run_deprecated_v1
  def testInsertManyEmptyTensorUnknown(self):
    # [FUNCTION DOC]
    # Test: barrier without declared shapes; inserting a zero-element array of values for component
    # 0 builds fine and ready size is 0, but running the insert raises an op error stating that
    # tensors with no elements are not supported. Graph mode only.
    with self.cached_session():
      b = data_flow_ops.Barrier((dtypes.float32, dtypes.float32), name="B")
      size_t = b.ready_size()
      self.assertEqual([], size_t.get_shape())
      keys = [b"a", b"b", b"c"]
      insert_0_op = b.insert_many(0, keys, np.array([[], [], []], np.float32))
      self.assertEqual(self.evaluate(size_t), [0])
      with self.assertRaisesOpError(
          ".*Tensors with no elements are not supported.*"):
        insert_0_op.run()

  @test_util.run_deprecated_v1
  def testTakeMany(self):
    # [FUNCTION DOC]
    # Test: inserts both components for three keys, checks ready size is 3, takes 3 elements and
    # verifies every returned index equals the minimum 64-bit integer, and that each key maps to
    # the matching values of both components (order of keys is not assumed).
    with self.cached_session() as sess:
      b = data_flow_ops.Barrier(
          (dtypes.float32, dtypes.float32), shapes=((), ()), name="B")
      size_t = b.ready_size()
      keys = [b"a", b"b", b"c"]
      values_0 = [10.0, 20.0, 30.0]
      values_1 = [100.0, 200.0, 300.0]
      insert_0_op = b.insert_many(0, keys, values_0)
      insert_1_op = b.insert_many(1, keys, values_1)
      take_t = b.take_many(3)

      insert_0_op.run()
      insert_1_op.run()
      self.assertEqual(self.evaluate(size_t), [3])

      indices_val, keys_val, values_0_val, values_1_val = sess.run(
          [take_t[0], take_t[1], take_t[2][0], take_t[2][1]])

    self.assertAllEqual(indices_val, [-2**63] * 3)
    for k, v0, v1 in zip(keys, values_0, values_1):
      idx = keys_val.tolist().index(k)
      self.assertEqual(values_0_val[idx], v0)
      self.assertEqual(values_1_val[idx], v1)

  @test_util.run_deprecated_v1
  def testTakeManySmallBatch(self):
    # [FUNCTION DOC]
    # Test: exercises take_many with the allow-small-batch option on a barrier with four keys, where
    # only some elements are complete. After closing, checks incomplete and ready sizes, that
    # successive takes return reduced batches of 2 then 1 elements with correct values, that an empty
    # insert still works on a closed barrier, that inserting existing keys fails with a "is closed"
    # error, and that after a closing with cancel-pending-enqueues all further inserts fail.
    with self.cached_session() as sess:
      b = data_flow_ops.Barrier(
          (dtypes.float32, dtypes.float32), shapes=((), ()), name="B")
      size_t = b.ready_size()
      size_i = b.incomplete_size()
      keys = [b"a", b"b", b"c", b"d"]
      values_0 = [10.0, 20.0, 30.0, 40.0]
      values_1 = [100.0, 200.0, 300.0, 400.0]
      insert_0_op = b.insert_many(0, keys, values_0)
      # Split adding of the second component into two independent operations.
      # After insert_1_1_op, we'll have two ready elements in the barrier,
      # 2 will still be incomplete.
      insert_1_1_op = b.insert_many(1, keys[0:2], values_1[0:2])  # add "a", "b"
      insert_1_2_op = b.insert_many(1, keys[2:3], values_1[2:3])  # add "c"
      insert_1_3_op = b.insert_many(1, keys[3:], values_1[3:])  # add "d"
      insert_empty_op = b.insert_many(0, [], [])
      close_op = b.close()
      close_op_final = b.close(cancel_pending_enqueues=True)
      index_t, key_t, value_list_t = b.take_many(3, allow_small_batch=True)
      insert_0_op.run()
      insert_1_1_op.run()
      close_op.run()
      # Now we have a closed barrier with 2 ready elements. Running take_t
      # should return a reduced batch with 2 elements only.
      self.assertEqual(self.evaluate(size_i),
                       [2])  # assert that incomplete size = 2
      self.assertEqual(self.evaluate(size_t), [2])  # assert that ready size = 2
      _, keys_val, values_0_val, values_1_val = sess.run(
          [index_t, key_t, value_list_t[0], value_list_t[1]])
      # Check that correct values have been returned.
      for k, v0, v1 in zip(keys[0:2], values_0[0:2], values_1[0:2]):
        idx = keys_val.tolist().index(k)
        self.assertEqual(values_0_val[idx], v0)
        self.assertEqual(values_1_val[idx], v1)

      # The next insert completes the element with key "c". The next take_t
      # should return a batch with just 1 element.
      insert_1_2_op.run()
      self.assertEqual(self.evaluate(size_i),
                       [1])  # assert that incomplete size = 1
      self.assertEqual(self.evaluate(size_t), [1])  # assert that ready size = 1
      _, keys_val, values_0_val, values_1_val = sess.run(
          [index_t, key_t, value_list_t[0], value_list_t[1]])
      # Check that correct values have been returned.
      for k, v0, v1 in zip(keys[2:3], values_0[2:3], values_1[2:3]):
        idx = keys_val.tolist().index(k)
        self.assertEqual(values_0_val[idx], v0)
        self.assertEqual(values_1_val[idx], v1)

      # Adding nothing ought to work, even if the barrier is closed.
      insert_empty_op.run()

      # currently keys "a" and "b" are not in the barrier, adding them
      # again after it has been closed, ought to cause failure.
      with self.assertRaisesOpError("is closed"):
        insert_1_1_op.run()
      close_op_final.run()

      # These ops should fail because the barrier has now been closed with
      # cancel_pending_enqueues = True.
      with self.assertRaisesOpError("is closed"):
        insert_empty_op.run()
      with self.assertRaisesOpError("is closed"):
        insert_1_3_op.run()

  @test_util.run_deprecated_v1
  def testUseBarrierWithShape(self):
    # [FUNCTION DOC]
    # Test: barrier with fixed component shapes (2x2 and length 8). Inserts three keys, takes them
    # and checks indices, that returned arrays match the static shapes of the take outputs, and that
    # values per key match the inserted arrays.
    with self.cached_session() as sess:
      b = data_flow_ops.Barrier(
          (dtypes.float32, dtypes.float32), shapes=((2, 2), (8,)), name="B")
      size_t = b.ready_size()
      keys = [b"a", b"b", b"c"]
      values_0 = np.array(
          [[[10.0] * 2] * 2, [[20.0] * 2] * 2, [[30.0] * 2] * 2], np.float32)
      values_1 = np.array([[100.0] * 8, [200.0] * 8, [300.0] * 8], np.float32)
      insert_0_op = b.insert_many(0, keys, values_0)
      insert_1_op = b.insert_many(1, keys, values_1)
      take_t = b.take_many(3)

      insert_0_op.run()
      insert_1_op.run()
      self.assertEqual(self.evaluate(size_t), [3])

      indices_val, keys_val, values_0_val, values_1_val = sess.run(
          [take_t[0], take_t[1], take_t[2][0], take_t[2][1]])
      self.assertAllEqual(indices_val, [-2**63] * 3)
      self.assertShapeEqual(keys_val, take_t[1])
      self.assertShapeEqual(values_0_val, take_t[2][0])
      self.assertShapeEqual(values_1_val, take_t[2][1])

    for k, v0, v1 in zip(keys, values_0, values_1):
      idx = keys_val.tolist().index(k)
      self.assertAllEqual(values_0_val[idx], v0)
      self.assertAllEqual(values_1_val[idx], v1)

  @test_util.run_deprecated_v1
  def testParallelInsertMany(self):
    # [FUNCTION DOC]
    # Test: single-component scalar barrier; ten separate insert ops (one key each) are evaluated
    # together, ready size must be 10, and take_many(10) must return indices 2**63-offset style
    # sequence (minimum integer plus 0..9) with every key mapping to its value.
    with self.cached_session() as sess:
      b = data_flow_ops.Barrier(dtypes.float32, shapes=())
      size_t = b.ready_size()
      keys = [str(x).encode("ascii") for x in range(10)]
      values = [float(x) for x in range(10)]
      insert_ops = [b.insert_many(0, [k], [v]) for k, v in zip(keys, values)]
      take_t = b.take_many(10)

      self.evaluate(insert_ops)
      self.assertEqual(self.evaluate(size_t), [10])

      indices_val, keys_val, values_val = sess.run(
          [take_t[0], take_t[1], take_t[2][0]])

    self.assertAllEqual(indices_val, [-2**63 + x for x in range(10)])
    for k, v in zip(keys, values):
      idx = keys_val.tolist().index(k)
      self.assertEqual(values_val[idx], v)

  @test_util.run_deprecated_v1
  def testParallelTakeMany(self):
    # [FUNCTION DOC]
    # Test: inserts ten keys at once and then runs ten separate take_many(1) ops in one run call.
    # Verifies all indices equal the minimum 64-bit integer and that the set of (key, value) pairs
    # taken equals the set inserted.
    with self.cached_session() as sess:
      b = data_flow_ops.Barrier(dtypes.float32, shapes=())
      size_t = b.ready_size()
      keys = [str(x).encode("ascii") for x in range(10)]
      values = [float(x) for x in range(10)]
      insert_op = b.insert_many(0, keys, values)
      take_t = [b.take_many(1) for _ in keys]

      insert_op.run()
      self.assertEqual(self.evaluate(size_t), [10])

      index_fetches = []
      key_fetches = []
      value_fetches = []
      for ix_t, k_t, v_t in take_t:
        index_fetches.append(ix_t)
        key_fetches.append(k_t)
        value_fetches.append(v_t[0])
      vals = sess.run(index_fetches + key_fetches + value_fetches)

    index_vals = vals[:len(keys)]
    key_vals = vals[len(keys):2 * len(keys)]
    value_vals = vals[2 * len(keys):]

    taken_elems = []
    for k, v in zip(key_vals, value_vals):
      taken_elems.append((k[0], v[0]))

    self.assertAllEqual(np.hstack(index_vals), [-2**63] * 10)

    self.assertItemsEqual(
        zip(keys, values), [(k[0], v[0]) for k, v in zip(key_vals, value_vals)])

  @test_util.run_deprecated_v1
  def testBlockingTakeMany(self):
    # [FUNCTION DOC]
    # Test: a worker thread runs take_many(10) which blocks; the main thread sleeps briefly and then
    # runs ten single-key insert ops. The thread must receive indices consistent with the keys and
    # the full set of key/value pairs once all inserts finish.
    with self.cached_session() as sess:
      b = data_flow_ops.Barrier(dtypes.float32, shapes=())
      keys = [str(x).encode("ascii") for x in range(10)]
      values = [float(x) for x in range(10)]
      insert_ops = [b.insert_many(0, [k], [v]) for k, v in zip(keys, values)]
      take_t = b.take_many(10)

      def take():
        # [FUNCTION DOC]
        # Nested thread target: runs the blocking take op in the enclosing session and asserts that
        # returned indices equal each numeric key minus 2**63 and that key/value pairs match the inserted
        # ones. Takes no arguments, returns nothing; failures are surfaced via the checked thread.
        indices_val, keys_val, values_val = sess.run(
            [take_t[0], take_t[1], take_t[2][0]])
        self.assertAllEqual(indices_val,
                            [int(x.decode("ascii")) - 2**63 for x in keys_val])
        self.assertItemsEqual(zip(keys, values), zip(keys_val, values_val))

      t = self.checkedThread(target=take)
      t.start()
      time.sleep(0.1)
      for insert_op in insert_ops:
        insert_op.run()
      t.join()

  @test_util.run_deprecated_v1
  def testParallelInsertManyTakeMany(self):
    # [FUNCTION DOC]
    # Test: stress test with 100 iterations; each iteration inserts two components for ten keys
    # (key text prefixed by the iteration number) and a take_many(10). Starts all take threads, then
    # all insert threads, joins everything and verifies that exactly 100 batches were taken, that
    # all indices and keys match the expected sorted lists, and that each batch's values are
    # consistent with the numbers encoded in its keys.
    with self.cached_session() as sess:
      b = data_flow_ops.Barrier(
          (dtypes.float32, dtypes.int64), shapes=((), (2,)))
      num_iterations = 100
      keys = [str(x) for x in range(10)]
      values_0 = np.asarray(range(10), dtype=np.float32)
      values_1 = np.asarray([[x + 1, x + 2] for x in range(10)], dtype=np.int64)
      keys_i = lambda i: [("%d:%s" % (i, k)).encode("ascii") for k in keys]
      insert_0_ops = [
          b.insert_many(0, keys_i(i), values_0 + i)
          for i in range(num_iterations)
      ]
      insert_1_ops = [
          b.insert_many(1, keys_i(i), values_1 + i)
          for i in range(num_iterations)
      ]
      take_ops = [b.take_many(10) for _ in range(num_iterations)]

      def take(sess, i, taken):
        # [FUNCTION DOC]
        # Nested thread target: runs the take ops for iteration i in the given session and appends a
        # mapping with indices, keys and both value arrays to the shared list. Inputs: session,
        # iteration number, list to append to. Returns nothing.
        indices_val, keys_val, values_0_val, values_1_val = sess.run([
            take_ops[i][0], take_ops[i][1], take_ops[i][2][0], take_ops[i][2][1]
        ])
        taken.append({
            "indices": indices_val,
            "keys": keys_val,
            "values_0": values_0_val,
            "values_1": values_1_val
        })

      def insert(sess, i):
        # [FUNCTION DOC]
        # Nested thread target: runs both insert ops for iteration i in the given session. Inputs:
        # session and iteration number. Returns nothing.
        sess.run([insert_0_ops[i], insert_1_ops[i]])

      taken = []

      take_threads = [
          self.checkedThread(
              target=take, args=(sess, i, taken)) for i in range(num_iterations)
      ]
      insert_threads = [
          self.checkedThread(
              target=insert, args=(sess, i)) for i in range(num_iterations)
      ]

      for t in take_threads:
        t.start()
      time.sleep(0.1)
      for t in insert_threads:
        t.start()
      for t in take_threads:
        t.join()
      for t in insert_threads:
        t.join()

      self.assertEqual(len(taken), num_iterations)
      flatten = lambda l: [item for sublist in l for item in sublist]
      all_indices = sorted(flatten([t_i["indices"] for t_i in taken]))
      all_keys = sorted(flatten([t_i["keys"] for t_i in taken]))

      expected_keys = sorted(
          flatten([keys_i(i) for i in range(num_iterations)]))
      expected_indices = sorted(
          flatten([-2**63 + j] * 10 for j in range(num_iterations)))

      self.assertAllEqual(all_indices, expected_indices)
      self.assertAllEqual(all_keys, expected_keys)

      for taken_i in taken:
        outer_indices_from_keys = np.array(
            [int(k.decode("ascii").split(":")[0]) for k in taken_i["keys"]])
        inner_indices_from_keys = np.array(
            [int(k.decode("ascii").split(":")[1]) for k in taken_i["keys"]])
        self.assertAllEqual(taken_i["values_0"],
                            outer_indices_from_keys + inner_indices_from_keys)
        expected_values_1 = np.vstack(
            (1 + outer_indices_from_keys + inner_indices_from_keys,
             2 + outer_indices_from_keys + inner_indices_from_keys)).T
        self.assertAllEqual(taken_i["values_1"], expected_values_1)

  @test_util.run_deprecated_v1
  def testClose(self):
    # [FUNCTION DOC]
    # Test: closing a barrier without cancelling pending enqueues. After a partial insert and close,
    # new keys fail with "is closed", the pending component can still be inserted, taking more
    # elements than exist fails with an insufficient-elements message naming requested and total
    # size, taking the 3 available succeeds with correct values, and a further take fails.
    with self.cached_session() as sess:
      b = data_flow_ops.Barrier(
          (dtypes.float32, dtypes.float32), shapes=((), ()), name="B")
      size_t = b.ready_size()
      incomplete_t = b.incomplete_size()
      keys = [b"a", b"b", b"c"]
      values_0 = [10.0, 20.0, 30.0]
      values_1 = [100.0, 200.0, 300.0]
      insert_0_op = b.insert_many(0, keys, values_0)
      insert_1_op = b.insert_many(1, keys, values_1)
      close_op = b.close()
      fail_insert_op = b.insert_many(0, ["f"], [60.0])
      take_t = b.take_many(3)
      take_too_many_t = b.take_many(4)

      self.assertEqual(self.evaluate(size_t), [0])
      self.assertEqual(self.evaluate(incomplete_t), [0])
      insert_0_op.run()
      self.assertEqual(self.evaluate(size_t), [0])
      self.assertEqual(self.evaluate(incomplete_t), [3])
      close_op.run()

      # This op should fail because the barrier is closed.
      with self.assertRaisesOpError("is closed"):
        fail_insert_op.run()

      # This op should succeed because the barrier has not canceled
      # pending enqueues
      insert_1_op.run()
      self.assertEqual(self.evaluate(size_t), [3])
      self.assertEqual(self.evaluate(incomplete_t), [0])

      # This op should fail because the barrier is closed.
      with self.assertRaisesOpError("is closed"):
        fail_insert_op.run()

      # This op should fail because we requested more elements than are
      # available in incomplete + ready queue.
      with self.assertRaisesOpError(r"is closed and has insufficient elements "
                                    r"\(requested 4, total size 3\)"):
        sess.run(take_too_many_t[0])  # Sufficient to request just the indices

      # This op should succeed because there are still completed elements
      # to process.
      indices_val, keys_val, values_0_val, values_1_val = sess.run(
          [take_t[0], take_t[1], take_t[2][0], take_t[2][1]])
      self.assertAllEqual(indices_val, [-2**63] * 3)
      for k, v0, v1 in zip(keys, values_0, values_1):
        idx = keys_val.tolist().index(k)
        self.assertEqual(values_0_val[idx], v0)
        self.assertEqual(values_1_val[idx], v1)

      # This op should fail because there are no more completed elements and
      # the queue is closed.
      with self.assertRaisesOpError("is closed and has insufficient elements"):
        sess.run(take_t[0])

  @test_util.run_deprecated_v1
  def testCancel(self):
    # [FUNCTION DOC]
    # Test: closing with cancel-pending-enqueues. Two of three elements are completed before the
    # cancel; afterwards new inserts and the completion of the third element fail with "is closed",
    # taking 3 fails with an insufficient-elements message (requested 3, total 2), taking 2 succeeds
    # with the right values, and a subsequent take fails.
    with self.cached_session() as sess:
      b = data_flow_ops.Barrier(
          (dtypes.float32, dtypes.float32), shapes=((), ()), name="B")
      size_t = b.ready_size()
      incomplete_t = b.incomplete_size()
      keys = [b"a", b"b", b"c"]
      values_0 = [10.0, 20.0, 30.0]
      values_1 = [100.0, 200.0, 300.0]
      insert_0_op = b.insert_many(0, keys, values_0)
      insert_1_op = b.insert_many(1, keys[0:2], values_1[0:2])
      insert_2_op = b.insert_many(1, keys[2:], values_1[2:])
      cancel_op = b.close(cancel_pending_enqueues=True)
      fail_insert_op = b.insert_many(0, ["f"], [60.0])
      take_t = b.take_many(2)
      take_too_many_t = b.take_many(3)

      self.assertEqual(self.evaluate(size_t), [0])
      insert_0_op.run()
      insert_1_op.run()
      self.assertEqual(self.evaluate(size_t), [2])
      self.assertEqual(self.evaluate(incomplete_t), [1])
      cancel_op.run()

      # This op should fail because the queue is closed.
      with self.assertRaisesOpError("is closed"):
        fail_insert_op.run()

      # This op should fail because the queue is canceled.
      with self.assertRaisesOpError("is closed"):
        insert_2_op.run()

      # This op should fail because we requested more elements than are
      # available in incomplete + ready queue.
      with self.assertRaisesOpError(r"is closed and has insufficient elements "
                                    r"\(requested 3, total size 2\)"):
        sess.run(take_too_many_t[0])  # Sufficient to request just the indices

      # This op should succeed because there are still completed elements
      # to process.
      indices_val, keys_val, values_0_val, values_1_val = sess.run(
          [take_t[0], take_t[1], take_t[2][0], take_t[2][1]])
      self.assertAllEqual(indices_val, [-2**63] * 2)
      for k, v0, v1 in zip(keys[0:2], values_0[0:2], values_1[0:2]):
        idx = keys_val.tolist().index(k)
        self.assertEqual(values_0_val[idx], v0)
        self.assertEqual(values_1_val[idx], v1)

      # This op should fail because there are no more completed elements and
      # the queue is closed.
      with self.assertRaisesOpError("is closed and has insufficient elements"):
        sess.run(take_t[0])

  def _testClosedEmptyBarrierTakeManyAllowSmallBatchRaises(self, cancel):
    # [FUNCTION DOC]
    # HELPER: shared scenario for testClosedEmptyBarrierTakeManyAllowSmallBatchRaises.
    # Scenario helper: builds an empty barrier, creates a small-batch take, closes the barrier
    # (cancelling pending enqueues if the cancel argument is true) and asserts that running the take
    # raises an op error saying the barrier is closed with insufficient elements. Returns nothing.
    with self.cached_session() as sess:
      b = data_flow_ops.Barrier(
          (dtypes.float32, dtypes.float32), shapes=((), ()), name="B")
      take_t = b.take_many(1, allow_small_batch=True)
      self.evaluate(b.close(cancel))
      with self.assertRaisesOpError("is closed and has insufficient elements"):
        self.evaluate(take_t)

  @test_util.run_deprecated_v1
  def testClosedEmptyBarrierTakeManyAllowSmallBatchRaises(self):
    # [FUNCTION DOC]
    # Test: runs the closed-empty-barrier scenario helper twice, without and with cancel.
    self._testClosedEmptyBarrierTakeManyAllowSmallBatchRaises(cancel=False)
    self._testClosedEmptyBarrierTakeManyAllowSmallBatchRaises(cancel=True)

  def _testParallelInsertManyTakeManyCloseHalfwayThrough(self, cancel):
    # [FUNCTION DOC]
    # HELPER: shared scenario for the CloseHalfwayThrough and CancelHalfwayThrough tests.
    # Scenario helper taking a cancel flag. Starts 50 take threads and the first half of 50 insert
    # threads, closes the barrier (cancelling pending enqueues if requested), then starts the second
    # half. Take threads record batch size, or 0 when out-of-range is raised; insert threads ignore
    # cancelled errors. Asserts the sorted batch sizes are half zeros and half tens.
    with self.cached_session() as sess:
      b = data_flow_ops.Barrier(
          (dtypes.float32, dtypes.int64), shapes=((), (2,)))
      num_iterations = 50
      keys = [str(x) for x in range(10)]
      values_0 = np.asarray(range(10), dtype=np.float32)
      values_1 = np.asarray([[x + 1, x + 2] for x in range(10)], dtype=np.int64)
      keys_i = lambda i: [("%d:%s" % (i, k)).encode("ascii") for k in keys]
      insert_0_ops = [
          b.insert_many(0, keys_i(i), values_0 + i)
          for i in range(num_iterations)
      ]
      insert_1_ops = [
          b.insert_many(1, keys_i(i), values_1 + i)
          for i in range(num_iterations)
      ]
      take_ops = [b.take_many(10) for _ in range(num_iterations)]
      close_op = b.close(cancel_pending_enqueues=cancel)

      def take(sess, i, taken):
        # [FUNCTION DOC]
        # Nested thread target: runs take ops for iteration i; appends the number of taken elements to
        # the shared list, or zero if an out-of-range error occurs. Inputs: session, iteration, list.
        try:
          indices_val, unused_keys_val, unused_val_0, unused_val_1 = sess.run([
              take_ops[i][0], take_ops[i][1], take_ops[i][2][0],
              take_ops[i][2][1]
          ])
          taken.append(len(indices_val))
        except errors_impl.OutOfRangeError:
          taken.append(0)

      def insert(sess, i):
        # [FUNCTION DOC]
        # Nested thread target: runs both insert ops for iteration i, silently ignoring a cancelled
        # error. Inputs: session and iteration. Returns nothing.
        try:
          sess.run([insert_0_ops[i], insert_1_ops[i]])
        except errors_impl.CancelledError:
          pass

      taken = []

      take_threads = [
          self.checkedThread(
              target=take, args=(sess, i, taken)) for i in range(num_iterations)
      ]
      insert_threads = [
          self.checkedThread(
              target=insert, args=(sess, i)) for i in range(num_iterations)
      ]

      first_half_insert_threads = insert_threads[:num_iterations // 2]
      second_half_insert_threads = insert_threads[num_iterations // 2:]

      for t in take_threads:
        t.start()
      for t in first_half_insert_threads:
        t.start()
      for t in first_half_insert_threads:
        t.join()

      close_op.run()

      for t in second_half_insert_threads:
        t.start()
      for t in take_threads:
        t.join()
      for t in second_half_insert_threads:
        t.join()

      self.assertEqual(
          sorted(taken),
          [0] * (num_iterations // 2) + [10] * (num_iterations // 2))

  @test_util.run_deprecated_v1
  def testParallelInsertManyTakeManyCloseHalfwayThrough(self):
    # [FUNCTION DOC]
    # Test: runs the close-halfway scenario with cancel disabled.
    self._testParallelInsertManyTakeManyCloseHalfwayThrough(cancel=False)

  @test_util.run_deprecated_v1
  def testParallelInsertManyTakeManyCancelHalfwayThrough(self):
    # [FUNCTION DOC]
    # Test: runs the close-halfway scenario with cancel enabled.
    self._testParallelInsertManyTakeManyCloseHalfwayThrough(cancel=True)

  def _testParallelPartialInsertManyTakeManyCloseHalfwayThrough(self, cancel):
    # [FUNCTION DOC]
    # HELPER: shared scenario for the partial-insert Close/Cancel tests.
    # Scenario helper taking a cancel flag. Component 0 is inserted for 100 iterations first, then
    # take threads start, the barrier is closed (with or without cancelling), and only then the
    # component-1 inserts run. Without cancel, every take must return 10 elements; with cancel every
    # take must yield nothing (out-of-range recorded as 0) and insert errors of cancelled type are
    # ignored.
    with self.cached_session() as sess:
      b = data_flow_ops.Barrier(
          (dtypes.float32, dtypes.int64), shapes=((), (2,)))
      num_iterations = 100
      keys = [str(x) for x in range(10)]
      values_0 = np.asarray(range(10), dtype=np.float32)
      values_1 = np.asarray([[x + 1, x + 2] for x in range(10)], dtype=np.int64)
      keys_i = lambda i: [("%d:%s" % (i, k)).encode("ascii") for k in keys]
      insert_0_ops = [
          b.insert_many(
              0, keys_i(i), values_0 + i, name="insert_0_%d" % i)
          for i in range(num_iterations)
      ]

      close_op = b.close(cancel_pending_enqueues=cancel)

      take_ops = [
          b.take_many(
              10, name="take_%d" % i) for i in range(num_iterations)
      ]
      # insert_1_ops will only run after closure
      insert_1_ops = [
          b.insert_many(
              1, keys_i(i), values_1 + i, name="insert_1_%d" % i)
          for i in range(num_iterations)
      ]

      def take(sess, i, taken):
        # [FUNCTION DOC]
        # Nested thread target: runs take ops for iteration i and appends the batch size. When cancel is
        # set an out-of-range error is recorded as 0; otherwise errors propagate. Inputs: session,
        # iteration, list to append to.
        if cancel:
          try:
            indices_val, unused_keys_val, unused_val_0, unused_val_1 = sess.run(
                [
                    take_ops[i][0], take_ops[i][1], take_ops[i][2][0],
                    take_ops[i][2][1]
                ])
            taken.append(len(indices_val))
          except errors_impl.OutOfRangeError:
            taken.append(0)
        else:
          indices_val, unused_keys_val, unused_val_0, unused_val_1 = sess.run([
              take_ops[i][0], take_ops[i][1], take_ops[i][2][0],
              take_ops[i][2][1]
          ])
          taken.append(len(indices_val))

      def insert_0(sess, i):
        # [FUNCTION DOC]
        # Nested thread target: runs the component-0 insert op for iteration i using the given session.
        insert_0_ops[i].run(session=sess)

      def insert_1(sess, i):
        # [FUNCTION DOC]
        # Nested thread target: runs the component-1 insert op for iteration i in the given session;
        # when cancel is set, a cancelled error is swallowed, otherwise errors propagate.
        if cancel:
          try:
            insert_1_ops[i].run(session=sess)
          except errors_impl.CancelledError:
            pass
        else:
          insert_1_ops[i].run(session=sess)

      taken = []

      take_threads = [
          self.checkedThread(
              target=take, args=(sess, i, taken)) for i in range(num_iterations)
      ]
      insert_0_threads = [
          self.checkedThread(
              target=insert_0, args=(sess, i)) for i in range(num_iterations)
      ]
      insert_1_threads = [
          self.checkedThread(
              target=insert_1, args=(sess, i)) for i in range(num_iterations)
      ]

      for t in insert_0_threads:
        t.start()
      for t in insert_0_threads:
        t.join()
      for t in take_threads:
        t.start()

      close_op.run()

      for t in insert_1_threads:
        t.start()
      for t in take_threads:
        t.join()
      for t in insert_1_threads:
        t.join()

      if cancel:
        self.assertEqual(taken, [0] * num_iterations)
      else:
        self.assertEqual(taken, [10] * num_iterations)

  @test_util.run_deprecated_v1
  def testParallelPartialInsertManyTakeManyCloseHalfwayThrough(self):
    # [FUNCTION DOC]
    # Test: runs the partial-insert scenario with cancel disabled.
    self._testParallelPartialInsertManyTakeManyCloseHalfwayThrough(cancel=False)

  @test_util.run_deprecated_v1
  def testParallelPartialInsertManyTakeManyCancelHalfwayThrough(self):
    # [FUNCTION DOC]
    # Test: runs the partial-insert scenario with cancel enabled.
    self._testParallelPartialInsertManyTakeManyCloseHalfwayThrough(cancel=True)

  @test_util.run_deprecated_v1
  def testIncompatibleSharedBarrierErrors(self):
    # [FUNCTION DOC]
    # Test: creates pairs of barriers with the same shared name but different component types
    # (different dtype, different count) or different shapes (known vs unknown, different ranks or
    # sizes). Evaluating the first succeeds; evaluating the second must raise an op error mentioning
    # "component types" or "component shapes" as appropriate.
    with self.cached_session():
      # Do component types and shapes.
      b_a_1 = data_flow_ops.Barrier(
          (dtypes.float32,), shapes=(()), shared_name="b_a")
      b_a_2 = data_flow_ops.Barrier(
          (dtypes.int32,), shapes=(()), shared_name="b_a")
      self.evaluate(b_a_1.barrier_ref)
      with self.assertRaisesOpError("component types"):
        self.evaluate(b_a_2.barrier_ref)

      b_b_1 = data_flow_ops.Barrier(
          (dtypes.float32,), shapes=(()), shared_name="b_b")
      b_b_2 = data_flow_ops.Barrier(
          (dtypes.float32, dtypes.int32), shapes=((), ()), shared_name="b_b")
      self.evaluate(b_b_1.barrier_ref)
      with self.assertRaisesOpError("component types"):
        self.evaluate(b_b_2.barrier_ref)

      b_c_1 = data_flow_ops.Barrier(
          (dtypes.float32, dtypes.float32),
          shapes=((2, 2), (8,)),
          shared_name="b_c")
      b_c_2 = data_flow_ops.Barrier(
          (dtypes.float32, dtypes.float32), shared_name="b_c")
      self.evaluate(b_c_1.barrier_ref)
      with self.assertRaisesOpError("component shapes"):
        self.evaluate(b_c_2.barrier_ref)

      b_d_1 = data_flow_ops.Barrier(
          (dtypes.float32, dtypes.float32), shapes=((), ()), shared_name="b_d")
      b_d_2 = data_flow_ops.Barrier(
          (dtypes.float32, dtypes.float32),
          shapes=((2, 2), (8,)),
          shared_name="b_d")
      self.evaluate(b_d_1.barrier_ref)
      with self.assertRaisesOpError("component shapes"):
        self.evaluate(b_d_2.barrier_ref)

      b_e_1 = data_flow_ops.Barrier(
          (dtypes.float32, dtypes.float32),
          shapes=((2, 2), (8,)),
          shared_name="b_e")
      b_e_2 = data_flow_ops.Barrier(
          (dtypes.float32, dtypes.float32),
          shapes=((2, 5), (8,)),
          shared_name="b_e")
      self.evaluate(b_e_1.barrier_ref)
      with self.assertRaisesOpError("component shapes"):
        self.evaluate(b_e_2.barrier_ref)


if __name__ == "__main__":
  test.main()
