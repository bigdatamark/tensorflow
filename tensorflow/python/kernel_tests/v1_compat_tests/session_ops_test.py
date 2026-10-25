# [MODULE DOC]
# Test module for session tensor handles (get session handle, get session tensor, delete session
# tensor), run in deprecated V1 graph mode. Exercises storing tensors in the session, feeding
# handles back, evaluating them, deleting them, placing them on different devices, and feeding
# handles directly to tensors and variables. Defines one test class; no import-time side effects.
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
"""Tests for tensorflow.ops.session_ops."""
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import session_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class SessionOpsTest(test.TestCase):
  # [CLASS DOC]
  # Test case for session tensor handle operations. Every test uses a cached session and is
  # decorated to run in deprecated V1 graph mode; there are no shared fixtures.

  @test_util.run_deprecated_v1
  def testHandleBasic(self):
    # [FUNCTION DOC]
    # Scenario: multiply constants 10 and 5, store the product as a session handle, then create a
    # session tensor placeholder for that handle, multiply it by 10 and run with the handle fed.
    # Expects 500.
    with self.cached_session() as sess:
      # Return a handle.
      a = constant_op.constant(10)
      b = constant_op.constant(5)
      c = math_ops.multiply(a, b)
      h = session_ops.get_session_handle(c)
      h = self.evaluate(h)

      # Feed a tensor handle.
      f, x = session_ops.get_session_tensor(h.handle, dtypes.int32)
      y = math_ops.multiply(x, 10)
      self.assertEqual(500, sess.run(y, feed_dict={f: h.handle}))

  @test_util.run_deprecated_v1
  def testHandleEval(self):
    # [FUNCTION DOC]
    # Scenario: store the product of 10 and 5 as a session handle and evaluate the handle object
    # directly to read back the stored tensor. Expects 50.
    with self.cached_session() as sess:
      # Return a handle.
      a = constant_op.constant(10)
      b = constant_op.constant(5)
      c = math_ops.multiply(a, b)
      h = session_ops.get_session_handle(c)
      h = self.evaluate(h)

      # Get the tensor from its handle.
      self.assertEqual(50, h.eval())

  @test_util.run_deprecated_v1
  def testHandleAndValue(self):
    # [FUNCTION DOC]
    # Scenario: in one evaluation fetch both a handle to the product of 10 and 5 and a plain value
    # computed as 10 times that product. Expects the handle to read back 50 and the value to be 500.
    with self.cached_session() as sess:
      # Return a handle and a value.
      a = constant_op.constant(10)
      b = constant_op.constant(5)
      c = math_ops.multiply(a, b)
      h = session_ops.get_session_handle(c)
      v = math_ops.multiply(a, c)
      h, v = self.evaluate([h, v])

      self.assertEqual(50, h.eval())
      self.assertEqual(500, v)

  @test_util.run_deprecated_v1
  def testHandleCond(self):
    # [FUNCTION DOC]
    # Scenario: evaluate a Python-level condition (10 less than 5, false) and a handle to the
    # product 50 together, then choose in Python which multiplier (10 or 100) to apply to the
    # handle's fed tensor. Expects the result 5000 (the false branch).
    with self.cached_session() as sess:
      # Return a handle and a value
      a = constant_op.constant(10)
      b = constant_op.constant(5)
      p = math_ops.less(a, b)
      c = math_ops.multiply(a, b)
      h = session_ops.get_session_handle(c)
      p, h = self.evaluate([p, h])

      # Run by feeding a tensor handle.
      f, x = session_ops.get_session_tensor(h.handle, dtypes.int32)
      if p:
        y = math_ops.multiply(x, 10)
      else:
        y = math_ops.multiply(x, 100)
      result = sess.run(y, feed_dict={f: h.handle})

      self.assertEqual(5000, result)

  @test_util.run_deprecated_v1
  def testHandleForLoop(self):
    # [FUNCTION DOC]
    # Scenario: start with a handle to zero and, in a loop of 100 iterations, repeatedly feed the
    # current handle into a graph that adds one and returns a new handle (exercising handle
    # garbage collection). Expects the final handle to read back 100.
    with self.cached_session() as sess:
      # Initialize a handle.
      a = constant_op.constant(0)
      h = session_ops.get_session_handle(a)
      h = self.evaluate(h)

      # Do some computation.
      f, x = session_ops.get_session_tensor(h.handle, dtypes.int32)
      # Must define the loop body outside the loop.
      h_x = session_ops.get_session_handle(math_ops.add(x, 1))
      for _ in range(100):
        # This exercises garbage collection.
        h = sess.run(h_x, feed_dict={f: h.handle})

      self.assertEqual(100, h.eval())

  @test_util.run_deprecated_v1
  def testHandleWhileLoop(self):
    # [FUNCTION DOC]
    # Scenario: start with a handle to zero and loop, each time fetching both the predicate
    # "value less than 100" and a new handle holding the value plus one, until the predicate is
    # false. Because the predicate is computed before the increment, expects the final handle to
    # read back 101.
    with self.cached_session() as sess:
      # Initialize a handle.
      a = constant_op.constant(0)
      h = session_ops.get_session_handle(a)
      h = self.evaluate(h)

      # Do some computation.
      f, x = session_ops.get_session_tensor(h.handle, dtypes.int32)
      b = constant_op.constant(100)
      p = math_ops.less(x, b)
      # Must define the loop body outside the loop.
      h_x = session_ops.get_session_handle(math_ops.add(x, 1))
      while True:
        rp, h = sess.run([p, h_x], feed_dict={f: h.handle})
        if not rp:
          break

      self.assertEqual(101, h.eval())

  @test_util.run_deprecated_v1
  def testHandleMover(self):
    # [FUNCTION DOC]
    # Scenario: feed a handle holding 50 to a graph multiplying by 10 (expects 500), then create
    # on the GPU device (or default device name) a new handle holding 10 and feed it to the same
    # graph. Expects 100, showing a handle can be moved between devices.
    with self.cached_session() as sess:
      # Return a handle.
      a = constant_op.constant(10)
      b = constant_op.constant(5)
      c = math_ops.multiply(a, b)
      h = session_ops.get_session_handle(c)
      h = self.evaluate(h)

      # Feed a tensor handle.
      f, x = session_ops.get_session_tensor(h.handle, dtypes.int32)
      y = math_ops.multiply(x, 10)
      self.assertEqual(500, sess.run(y, feed_dict={f: h.handle}))

      # Feed another tensor handle.
      with ops.device(test.gpu_device_name()):
        a = constant_op.constant(10)
        h = session_ops.get_session_handle(a)
        h = self.evaluate(h)
        self.assertEqual(100, sess.run(y, feed_dict={f: h.handle}))

  @test_util.run_deprecated_v1
  def testHandleDelete(self):
    # [FUNCTION DOC]
    # Scenario: create a handle to the product of 10 and 5, evaluate it and call delete on the
    # resulting handle object. Expects no error; there are no explicit assertions.
    with self.cached_session() as sess:
      # Return a handle.
      a = constant_op.constant(10)
      b = constant_op.constant(5)
      c = math_ops.multiply(a, b)
      h = session_ops.get_session_handle(c)
      self.evaluate(h).delete()

  @test_util.run_deprecated_v1
  def testHandleDeleteRaw(self):
    # [FUNCTION DOC]
    # Scenario: create and evaluate a handle, take its raw handle string, build the delete-session
    # tensor op for it and run that op feeding the raw handle. Expects the run to succeed without
    # error; there are no explicit assertions.
    with self.cached_session() as sess:
      # Return a handle.
      a = constant_op.constant(10)
      b = constant_op.constant(5)
      c = math_ops.multiply(a, b)
      h = session_ops.get_session_handle(c)
      h = self.evaluate(h)

      # Delete using a raw tensor handle.
      raw_h = h.get_raw_handle()
      f, x = session_ops.delete_session_tensor(raw_h)
      sess.run(x, feed_dict={f: raw_h})

  @test_util.run_deprecated_v1
  def testMultiDevices(self):
    # [FUNCTION DOC]
    # Scenario: store the float 1.0 as a handle on the GPU device name and 2.0 as a handle on the
    # CPU, then add the two tensors recovered from their handles and store the sum as a handle.
    # Expects the sum handle to read back 3.0.
    with self.cached_session() as sess:
      with ops.device(test.gpu_device_name()):
        a = constant_op.constant(1.0)
        a_handle = self.evaluate(session_ops.get_session_handle(a))
      with ops.device("/cpu:0"):
        b = constant_op.constant(2.0)
        b_handle = self.evaluate(session_ops.get_session_handle(b))

      a_p, a_t = session_ops.get_session_tensor(a_handle.handle, dtypes.float32)
      b_p, b_t = session_ops.get_session_tensor(b_handle.handle, dtypes.float32)
      c = math_ops.add(a_t, b_t)
      c_handle = sess.run(
          session_ops.get_session_handle(c),
          feed_dict={a_p: a_handle.handle,
                     b_p: b_handle.handle})
      self.assertEqual(3.0, c_handle.eval())

  @test_util.run_deprecated_v1
  def testHandleGC(self):
    # [FUNCTION DOC]
    # Scenario: create two handles holding 1.0 on the CPU, then repeat 20 times an add operation
    # placed on the GPU device name that adds the constant handle to the running handle and
    # returns a new handle. Checks that repeated cross-device handle creation and replacement
    # works (garbage collection of old handles); no explicit assertions.
    with self.cached_session() as sess:
      # initial values live on CPU
      with ops.device("/cpu:0"):
        one = constant_op.constant(1, dtype=dtypes.float32)
        one_handle = self.evaluate(session_ops.get_session_handle(one))
        x_handle = self.evaluate(session_ops.get_session_handle(one))

      # addition lives on GPU
      with ops.device(test.gpu_device_name()):
        add_h1, add_t1 = session_ops.get_session_tensor(one_handle.handle,
                                                        dtypes.float32)
        add_h2, add_t2 = session_ops.get_session_tensor(x_handle.handle,
                                                        dtypes.float32)
        add_op = math_ops.add(add_t1, add_t2)
        add_output = session_ops.get_session_handle(add_op)

      # add 1 to tensor 20 times
      for _ in range(20):
        x_handle = sess.run(
            add_output,
            feed_dict={add_h1: one_handle.handle,
                       add_h2: x_handle.handle})

  @test_util.run_deprecated_v1
  def testHandlePlacement(self):
    # [FUNCTION DOC]
    # Scenario: create handles for 1.0 and 2.0 without explicit device scopes, recover the tensors
    # from the handles, add them, and store the sum as a handle. Expects the sum handle to read
    # back 3.0, so automatic placement works.
    with self.cached_session() as sess:
      a = constant_op.constant(1.0)
      a_handle_op = session_ops.get_session_handle(a)
      b = constant_op.constant(2.0)
      b_handle_op = session_ops.get_session_handle(b)

      a_handle = self.evaluate(a_handle_op)
      b_handle = self.evaluate(b_handle_op)

      a_p, a_t = session_ops.get_session_tensor(a_handle.handle, dtypes.float32)
      b_p, b_t = session_ops.get_session_tensor(b_handle.handle, dtypes.float32)

      c = math_ops.add(a_t, b_t)
      c_handle = sess.run(
          session_ops.get_session_handle(c),
          feed_dict={a_p: a_handle.handle,
                     b_p: b_handle.handle})
      self.assertEqual(3.0, c_handle.eval())

  @test_util.run_deprecated_v1
  def testFeedOneHandleDirectly(self):
    # [FUNCTION DOC]
    # Scenario: c is 10 times 5 and d is c squared; a handle to c is evaluated and then fed
    # directly in place of the tensor c when running d. Expects 2500.
    with self.cached_session() as sess:
      a = constant_op.constant(10.0)
      b = constant_op.constant(5.0)
      c = math_ops.multiply(a, b)
      d = math_ops.multiply(c, c)

      h_c = self.evaluate(session_ops.get_session_handle(c))

      self.assertAllClose(2500.0, sess.run(d, feed_dict={c: h_c}))

  @test_util.run_deprecated_v1
  def testDirectHandleFeedOverlappingWithFetches(self):
    # [FUNCTION DOC]
    # Scenario: c is 50 and d is the identity of c; a handle to c is fed in place of c while
    # fetching c, d, or both, sometimes also feeding d with a plain value 60 or the handle.
    # Expects: feeding the handle for c yields 50 for both c and d; feeding plain 60 for d yields
    # d equal to 60 with c 50; feeding 60 for c and the handle for d yields c 60 and d 50; feeding
    # the handle for both yields 50 for both. Fed values override computation of fetched tensors.
    with self.cached_session() as sess:
      a = constant_op.constant(10.0)
      b = constant_op.constant(5.0)
      c = math_ops.multiply(a, b)
      h_c = self.evaluate(session_ops.get_session_handle(c))
      d = array_ops.identity(c)

      c_val = sess.run(c, feed_dict={c: h_c})
      self.assertAllClose(50.0, c_val)

      d_val = sess.run(d, feed_dict={c: h_c})
      self.assertAllClose(50.0, d_val)

      c_val, d_val = sess.run([c, d], feed_dict={c: h_c, d: 60.0})
      self.assertAllClose(50.0, c_val)
      self.assertAllClose(60.0, d_val)

      c_val, d_val = sess.run([c, d], feed_dict={c: 60.0, d: h_c})
      self.assertAllClose(60.0, c_val)
      self.assertAllClose(50.0, d_val)

      c_val, d_val = sess.run([c, d], feed_dict={c: h_c, d: h_c})
      self.assertAllClose(50.0, c_val)
      self.assertAllClose(50.0, d_val)

  @test_util.run_deprecated_v1
  def testFeedTwoHandlesDirectly(self):
    # [FUNCTION DOC]
    # Scenario: c is 50, d is 2, e is c minus d; handles for c and d are fed directly in place of
    # those tensors. Expects 48 with the handles matched to their tensors and minus 48 when the
    # handles are swapped.
    with self.cached_session() as sess:
      a = constant_op.constant(10.0)
      b = constant_op.constant(5.0)
      c = math_ops.multiply(a, b)
      d = math_ops.div(a, b)
      e = math_ops.subtract(c, d)

      h_c = self.evaluate(session_ops.get_session_handle(c))
      h_d = self.evaluate(session_ops.get_session_handle(d))

      self.assertAllClose(48.0, sess.run(e, feed_dict={c: h_c, d: h_d}))
      self.assertAllClose(-48.0, sess.run(e, feed_dict={c: h_d, d: h_c}))

  @test_util.run_deprecated_v1
  def testFeedHandleToVariableDirectly(self):
    # [FUNCTION DOC]
    # Scenario: a variable starts at 12, with an increment-by-2 op and a tensor b equal to the
    # variable plus 5. A handle to the variable's read value is taken after initialization.
    # Asserts the variable reads 12, that b is 17 when the handle is fed for the variable, and,
    # after running the increment, that b is 19 when the same handle is fed again (the second
    # value reflects the variable's updated state, so the feed is expected to be read from the
    # variable's current value in this test's scenario).
    with self.cached_session() as sess:
      a = variables.Variable(12.0)
      inc_a = state_ops.assign_add(a, 2.0)
      b = math_ops.add(a, 5.0)
      self.evaluate(a.initializer)

      h_a_read = sess.run(session_ops.get_session_handle(a.read_value()))
      self.assertAllClose(12.0, self.evaluate(a))

      self.assertAllClose(17.0, sess.run(b, feed_dict={a: h_a_read}))
      self.evaluate(inc_a)
      self.assertAllClose(19.0, sess.run(b, feed_dict={a: h_a_read}))


if __name__ == "__main__":
  test.main()
