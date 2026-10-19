# [MODULE DOC]
# Test module for the session partial-run API (setup, then incremental fetches with feeds).
# A test case holds shared scenario helpers taking a session, and test methods run each scenario
# against a direct local session and a session on a local distributed server. Exposes nothing;
# the main block runs the tests. No import-time side effects.
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for tensorflow.python.client.session.Session's partial run APIs."""


from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import googletest
from tensorflow.python.training import server_lib

class PartialRunTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for partial runs. Methods named RunTest... are scenario helpers that take a
  # session; test... methods call them with a direct session ("Direct") or a session connected
  # to a local server ("Dist").

  def RunTestPartialRun(self, sess):
    # [FUNCTION DOC]
    # HELPER: scenario shared by the Direct and Dist partial-run tests.
    # Given a session, builds add and multiply on three placeholders, sets up a partial run,
    # fetches the sum (3), then the product with a fed value (153). Repeats with a second
    # handle on the same graph (3, then 162). Asserts the values.
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)
    r2 = math_ops.multiply(r1, c)

    h = sess.partial_run_setup([r1, r2], [a, b, c])
    res = sess.partial_run(h, r1, feed_dict={a: 1, b: 2})
    self.assertEqual(3, res)
    temp = res * 17
    res = sess.partial_run(h, r2, feed_dict={c: temp})
    self.assertEqual(153, res)

    # Call again on the same graph.
    h2 = sess.partial_run_setup([r1, r2], [a, b, c])
    res = sess.partial_run(h2, r1, feed_dict={a: 1, b: 2})
    self.assertEqual(3, res)
    temp = res * 18
    res = sess.partial_run(h2, r2, feed_dict={c: temp})
    self.assertEqual(162, res)

  def RunTestPartialRunIncomplete(self, sess):
    # [FUNCTION DOC]
    # HELPER: scenario shared by the Direct and Dist tests.
    # Sets up a partial run with two fetches but only performs the first step (asserts 3),
    # leaving the run incomplete; checks nothing breaks when it is abandoned.
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)
    r2 = math_ops.multiply(r1, c)

    h = sess.partial_run_setup([r1, r2], [a, b, c])
    res = sess.partial_run(h, r1, feed_dict={a: 1, b: 2})
    self.assertEqual(3, res)

  def RunTestConcurrentPartialRun(self, sess):
    # [FUNCTION DOC]
    # HELPER: scenario shared by the Direct and Dist tests.
    # Opens two partial-run handles at once on the same session, interleaves steps, and asserts
    # results 3, 66 and 462, showing handles are independent.
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)
    r2 = math_ops.multiply(r1, c)

    h1 = sess.partial_run_setup([r1], [a, b, c])
    h2 = sess.partial_run_setup([r1, r2], [a, b, c])
    res = sess.partial_run(h1, r1, feed_dict={a: 1, b: 2})
    self.assertEqual(3, res)
    temp = res * 19
    res = sess.partial_run(h2, r1, feed_dict={a: temp, b: 9})
    self.assertEqual(66, res)
    res = sess.partial_run(h2, r2, feed_dict={c: 7})
    self.assertEqual(462, res)

  def RunTestManyPartialRun(self, sess):
    # [FUNCTION DOC]
    # HELPER: scenario shared by the Direct and Dist tests.
    # Builds a chain of 200 multiplications by placeholders, fetches each stage in a separate
    # partial-run step feeding 1.0, and asserts the last is 2.0. Then runs all outputs in one
    # ordinary run and asserts 200 results ending in 2.0.
    steps = 200
    inputs = []
    outputs = []
    a = constant_op.constant(2.0, dtypes.float32)
    for i in range(steps):
      inputs.append(array_ops.placeholder(dtypes.float32, shape=[]))
      a = math_ops.multiply(a, inputs[i])
      outputs.append(a)

    h = sess.partial_run_setup(outputs, inputs)
    for i in range(steps):
      res = sess.partial_run(h, outputs[i], feed_dict={inputs[i]: 1.0})
    self.assertEqual(2.0, res)

    feed_dict = {}
    for i in range(steps):
      feed_dict[inputs[i]] = 1.0
    res = sess.run(outputs, feed_dict)
    self.assertEqual(steps, len(res))
    self.assertEqual(2.0, res[-1])

  def RunTestRunAndPartialRun(self, sess):
    # [FUNCTION DOC]
    # HELPER: scenario shared by the Direct and Dist tests.
    # Evaluates two derived constants with an ordinary evaluate call, then fetches the same
    # tensors via a partial run with no feeds, and asserts both results are equal.
    a = constant_op.constant(2.0, dtypes.float32)
    b = a * 2
    c = b * 3
    r1 = self.evaluate([b, c])
    h = sess.partial_run_setup([b, c], [])
    r2 = sess.partial_run(h, [b, c])
    self.assertEqual(r1, r2)

  def RunTestPartialRunMissingPlaceholderFeedException(self, sess):
    # [FUNCTION DOC]
    # HELPER: scenario shared by the Direct and Dist tests.
    # Sets up a partial run whose fetches depend on a placeholder that is never listed as a feed,
    # then runs a step without feeding it. Asserts an invalid-argument error about needing a
    # value for the placeholder.
    x = array_ops.placeholder(dtypes.float32, shape=())
    fetches = [x * 2, x * 3]
    handle = sess.partial_run_setup(fetches=fetches, feeds=[])
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'You must feed a value for placeholder'):
      sess.partial_run(handle, fetches[0])

  def RunTestPartialRunUnspecifiedFeed(self, sess):
    # [FUNCTION DOC]
    # HELPER: scenario shared by the Direct and Dist tests.
    # Sets up with two feeds, then feeds an extra placeholder not declared at setup. Asserts an
    # invalid-argument error saying it was not specified in partial-run setup.
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)

    h = sess.partial_run_setup([r1], [a, b])
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'was not specified in partial_run_setup.$'):
      sess.partial_run(h, r1, feed_dict={a: 1, b: 2, c: 3})

  def RunTestPartialRunUnspecifiedFetch(self, sess):
    # [FUNCTION DOC]
    # HELPER: scenario shared by the Direct and Dist tests.
    # Sets up with a single fetch, then requests a different tensor in a step. Asserts an
    # invalid-argument error saying it was not specified in partial-run setup.
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)
    r2 = math_ops.multiply(a, c)

    h = sess.partial_run_setup([r1], [a, b, c])
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'was not specified in partial_run_setup.$'):
      sess.partial_run(h, r2, feed_dict={a: 1, c: 3})

  def RunTestPartialRunAlreadyFed(self, sess):
    # [FUNCTION DOC]
    # HELPER: scenario shared by the Direct and Dist tests.
    # Feeds a placeholder in a first step, then feeds it again in a second step. Asserts an
    # invalid-argument error saying it has already been fed.
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)
    r2 = math_ops.multiply(a, c)

    h = sess.partial_run_setup([r1, r2], [a, b, c])
    sess.partial_run(h, r1, feed_dict={a: 1, b: 2})
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'has already been fed.$'):
      sess.partial_run(h, r2, feed_dict={a: 1, c: 3})

  def RunTestPartialRunAlreadyFetched(self, sess):
    # [FUNCTION DOC]
    # HELPER: scenario shared by the Direct and Dist tests.
    # Fetches a tensor in a first step, then fetches it again in a second step. Asserts an
    # invalid-argument error saying it has already been fetched.
    a = array_ops.placeholder(dtypes.float32, shape=[])
    b = array_ops.placeholder(dtypes.float32, shape=[])
    c = array_ops.placeholder(dtypes.float32, shape=[])
    r1 = math_ops.add(a, b)
    r2 = math_ops.multiply(a, c)

    h = sess.partial_run_setup([r1, r2], [a, b, c])
    sess.partial_run(h, r1, feed_dict={a: 1, b: 2})
    with self.assertRaisesRegex(errors.InvalidArgumentError,
                                'has already been fetched.$'):
      sess.partial_run(h, r1, feed_dict={c: 3})

  def RunTestPartialRunEmptyFetches(self, sess):
    # [FUNCTION DOC]
    # HELPER: scenario shared by the Direct and Dist tests.
    # Performs a first step with an empty fetch list that only feeds the placeholder, then a
    # second step fetching the doubled value. Asserts the result is a list holding 6.0.
    a = array_ops.placeholder(dtypes.float32)
    b = a * 2.0

    h = sess.partial_run_setup(fetches=[b], feeds=[a])
    sess.partial_run(h, [], {a: 3.0})
    r = sess.partial_run(h, [b], {})
    self.assertEqual([6.0], r)

  @test_util.run_deprecated_v1
  def testInvalidPartialRunSetup(self):
    # [FUNCTION DOC]
    # Calls partial-run setup with an empty fetch list and a feed; asserts an invalid-argument
    # error saying at least one target to fetch or execute is required. Legacy graph mode.
    sess = session.Session()
    x = array_ops.placeholder(dtypes.float32, shape=[])
    with self.assertRaisesRegex(
        errors.InvalidArgumentError,
        'specify at least one target to fetch or execute.'):
      sess.partial_run_setup(fetches=[], feeds=[x])

  @test_util.run_deprecated_v1
  def testPartialRunSetupNoFeedsPassed(self):
    # [FUNCTION DOC]
    # Sets up a partial run with only fetches (feeds omitted) on a constant, runs it, and asserts
    # the result is a list holding 6.0. Legacy graph mode.
    sess = session.Session()
    r1 = constant_op.constant([6.0])

    h = sess.partial_run_setup([r1])
    result1 = sess.partial_run(h, r1)
    self.assertEqual([6.0], result1)

  @test_util.run_deprecated_v1
  def testPartialRunDirect(self):
    # [FUNCTION DOC]
    # Runs the basic partial-run scenario on a direct local session.
    self.RunTestPartialRun(session.Session())

  @test_util.run_deprecated_v1
  def testPartialRunIncompleteDirect(self):
    # [FUNCTION DOC]
    # Runs the incomplete partial-run scenario on a direct local session.
    self.RunTestPartialRunIncomplete(session.Session())

  @test_util.run_deprecated_v1
  def testConcurrentPartialRunDirect(self):
    # [FUNCTION DOC]
    # Runs the concurrent-handles scenario on a direct local session.
    self.RunTestConcurrentPartialRun(session.Session())

  @test_util.run_deprecated_v1
  def testManyPartialRunDirect(self):
    # [FUNCTION DOC]
    # Runs the 200-step chain scenario on a direct local session.
    self.RunTestManyPartialRun(session.Session())

  @test_util.run_deprecated_v1
  def testRunAndPartialRunDirect(self):
    # [FUNCTION DOC]
    # Runs the ordinary-run versus partial-run comparison scenario on a direct local session.
    self.RunTestRunAndPartialRun(session.Session())

  @test_util.run_deprecated_v1
  def testPartialRunMissingPlaceholderFeedExceptionDirect(self):
    # [FUNCTION DOC]
    # Runs the missing-placeholder-feed error scenario on a direct local session.
    self.RunTestPartialRunMissingPlaceholderFeedException(session.Session())

  @test_util.run_deprecated_v1
  def testPartialRunUnspecifiedFeedDirect(self):
    # [FUNCTION DOC]
    # Runs the unspecified-feed error scenario on a direct local session.
    self.RunTestPartialRunUnspecifiedFeed(session.Session())

  @test_util.run_deprecated_v1
  def testPartialRunUnspecifiedFetchDirect(self):
    # [FUNCTION DOC]
    # Runs the unspecified-fetch error scenario on a direct local session.
    self.RunTestPartialRunUnspecifiedFetch(session.Session())

  @test_util.run_deprecated_v1
  def testPartialRunAlreadyFedDirect(self):
    # [FUNCTION DOC]
    # Runs the already-fed error scenario on a direct local session.
    self.RunTestPartialRunAlreadyFed(session.Session())

  @test_util.run_deprecated_v1
  def testPartialRunAlreadyFetchedDirect(self):
    # [FUNCTION DOC]
    # Runs the already-fetched error scenario on a direct local session.
    self.RunTestPartialRunAlreadyFetched(session.Session())

  @test_util.run_deprecated_v1
  def testPartialRunEmptyFetchesDirect(self):
    # [FUNCTION DOC]
    # Runs the empty-fetches scenario on a direct local session.
    self.RunTestPartialRunEmptyFetches(session.Session())

  @test_util.run_deprecated_v1
  def testPartialRunDist(self):
    # [FUNCTION DOC]
    # Starts a local server and runs the basic partial-run scenario on a session connected to it.
    server = server_lib.Server.create_local_server()
    self.RunTestPartialRun(session.Session(server.target))

  @test_util.run_deprecated_v1
  def testPartialRunIncompleteDist(self):
    # [FUNCTION DOC]
    # Starts a local server and runs the incomplete partial-run scenario on a session to it.
    server = server_lib.Server.create_local_server()
    self.RunTestPartialRunIncomplete(session.Session(server.target))

  @test_util.run_deprecated_v1
  def testConcurrentPartialRunDist(self):
    # [FUNCTION DOC]
    # Starts a local server and runs the concurrent-handles scenario on a session to it.
    server = server_lib.Server.create_local_server()
    self.RunTestConcurrentPartialRun(session.Session(server.target))

  @test_util.run_deprecated_v1
  def testManyPartialRunDist(self):
    # [FUNCTION DOC]
    # Starts a local server and runs the 200-step chain scenario on a session to it.
    server = server_lib.Server.create_local_server()
    self.RunTestManyPartialRun(session.Session(server.target))

  @test_util.run_deprecated_v1
  def testRunAndPartialRunDist(self):
    # [FUNCTION DOC]
    # Starts a local server and runs the ordinary-run versus partial-run scenario on a session
    # to it.
    server = server_lib.Server.create_local_server()
    self.RunTestRunAndPartialRun(session.Session(server.target))

  @test_util.run_deprecated_v1
  def testPartialRunMissingPlaceholderFeedExceptionDist(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the test is disabled; it skips immediately as flaky, so the server start and
    # the scenario call after the skip never execute.
    # Intended to run the missing-placeholder-feed error scenario on a session to a local server.
    self.skipTest('Flaky test. Short term b/278768411, long term b/280102873')
    server = server_lib.Server.create_local_server()
    self.RunTestPartialRunMissingPlaceholderFeedException(
        session.Session(server.target))

  @test_util.run_deprecated_v1
  def testPartialRunUnspecifiedFeedDist(self):
    # [FUNCTION DOC]
    # Starts a local server and runs the unspecified-feed error scenario on a session to it.
    server = server_lib.Server.create_local_server()
    self.RunTestPartialRunUnspecifiedFeed(session.Session(server.target))

  @test_util.run_deprecated_v1
  def testPartialRunUnspecifiedFetchDist(self):
    # [FUNCTION DOC]
    # Starts a local server and runs the unspecified-fetch error scenario on a session to it.
    server = server_lib.Server.create_local_server()
    self.RunTestPartialRunUnspecifiedFetch(session.Session(server.target))

  @test_util.run_deprecated_v1
  def testPartialRunAlreadyFedDist(self):
    # [FUNCTION DOC]
    # Starts a local server and runs the already-fed error scenario on a session to it.
    server = server_lib.Server.create_local_server()
    self.RunTestPartialRunAlreadyFed(session.Session(server.target))

  @test_util.run_deprecated_v1
  def testPartialRunAlreadyFetchedDist(self):
    # [FUNCTION DOC]
    # Starts a local server and runs the already-fetched error scenario on a session to it.
    server = server_lib.Server.create_local_server()
    self.RunTestPartialRunAlreadyFetched(session.Session(server.target))

  @test_util.run_deprecated_v1
  def testPartialRunEmptyFetchesDist(self):
    # [FUNCTION DOC]
    # Starts a local server and runs the empty-fetches scenario on a session to it.
    server = server_lib.Server.create_local_server()
    self.RunTestPartialRunEmptyFetches(session.Session(server.target))


if __name__ == '__main__':
  googletest.main()
