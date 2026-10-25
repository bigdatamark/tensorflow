# [MODULE DOC]
# Unit tests for basic_loops.basic_train_loop using a real Supervisor with a temporary log
# directory. Covers argument passing and stopping, propagation of ordinary exceptions and the
# automatic retry on aborted errors.
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
"""Tests for basic_loops.py."""

import os
import shutil

from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.training import basic_loops
from tensorflow.python.training import supervisor


def _test_dir(test_name):
  # [FUNCTION DOC]
  # HELPER: creates a clean log directory path for the loop tests.
  # Input test_name is a directory name placed under the test temp directory. If that directory
  # already exists it is deleted recursively (side effect). Returns the path; it is not created.
  test_dir = os.path.join(test.get_temp_dir(), test_name)
  if os.path.exists(test_dir):
    shutil.rmtree(test_dir)
  return test_dir


class BasicTrainLoopTest(test.TestCase):
  # [CLASS DOC]
  # Test case for basic_train_loop. Stateless; each test creates a Supervisor on a fresh log
  # directory inside its own default graph.

  def testBasicTrainLoop(self):
    # [FUNCTION DOC]
    # Runs the loop with a step function that counts calls, checks it receives the extra positional
    # and keyword arguments, and asks the supervisor to stop on the third call. Asserts exactly
    # three calls happened.
    logdir = _test_dir("basic_train_loop")
    # Counts the number of calls.
    num_calls = [0]

    def train_fn(unused_sess, sv, y, a):
      # [FUNCTION DOC]
      # Step function: increments the shared call counter, asserts the forwarded arguments equal
      # the expected values, and requests the supervisor to stop on the third call. Returns nothing.
      num_calls[0] += 1
      self.assertEqual("y", y)
      self.assertEqual("A", a)
      if num_calls[0] == 3:
        sv.request_stop()

    with ops.Graph().as_default():
      sv = supervisor.Supervisor(logdir=logdir)
      basic_loops.basic_train_loop(
          sv, train_fn, args=(sv, "y"), kwargs={"a": "A"})
      self.assertEqual(3, num_calls[0])

  def testBasicTrainLoopExceptionAborts(self):
    # [FUNCTION DOC]
    # Uses a step function that raises a runtime error on its third call and asserts the loop
    # propagates that error to the caller.
    logdir = _test_dir("basic_train_loop_exception_aborts")

    def train_fn(unused_sess):
      # [FUNCTION DOC]
      # Step function: increments a counter stored as a function attribute and raises a runtime
      # error with the message "Failed" when the counter reaches three.
      train_fn.counter += 1
      if train_fn.counter == 3:
        raise RuntimeError("Failed")

    # Function attribute use to count the number of calls.
    train_fn.counter = 0

    with ops.Graph().as_default():
      sv = supervisor.Supervisor(logdir=logdir)
      with self.assertRaisesRegex(RuntimeError, "Failed"):
        basic_loops.basic_train_loop(sv, train_fn)

  def testBasicTrainLoopRetryOnAborted(self):
    # [FUNCTION DOC]
    # Uses a step function that raises aborted errors until its retry budget is used up, then a
    # runtime error. Asserts the loop retries on aborted errors, finally propagates the runtime
    # error, and that no retries remain.
    # KNOWN ISSUE: reuses the same log directory name as the exception-aborts test, so state could
    # be shared between the two tests (the directory is cleared by the helper beforehand).
    logdir = _test_dir("basic_train_loop_exception_aborts")

    class AbortAndRetry:
      # [CLASS DOC]
      # Fake training step provider. Holds a call counter and a number of retries left; its step
      # method raises aborted errors while retries remain.

      def __init__(self):
        # [FUNCTION DOC]
        # Initializes the call counter to zero and the retries left to two.
        self.num_calls = 0
        self.retries_left = 2

      def train_fn(self, unused_sess):
        # [FUNCTION DOC]
        # Step function: counts the call, decrements retries left on every call number that is
        # two modulo three, then raises an aborted error while retries remain, otherwise a runtime
        # error with the message "Failed Again". Never returns normally.
        self.num_calls += 1
        if self.num_calls % 3 == 2:
          self.retries_left -= 1
        if self.retries_left > 0:
          raise errors_impl.AbortedError(None, None, "Aborted here")
        else:
          raise RuntimeError("Failed Again")

    with ops.Graph().as_default():
      sv = supervisor.Supervisor(logdir=logdir)
      aar = AbortAndRetry()
      with self.assertRaisesRegex(RuntimeError, "Failed Again"):
        basic_loops.basic_train_loop(sv, aar.train_fn)
      self.assertEqual(0, aar.retries_left)


if __name__ == "__main__":
  test.main()
