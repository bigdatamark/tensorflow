# [MODULE DOC]
# Tests for the sleep custom op example: both the synchronous and asynchronous ops run for
# roughly the requested time, and the asynchronous op reports an error for a negative delay.
# Importing the op module loads a compiled shared library. Run directly, it starts the test runner.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for sleep."""

import time

import tensorflow as tf

from tensorflow.examples.custom_ops_doc.sleep import sleep_op
# This pylint disable is only needed for internal google users
from tensorflow.python.framework import errors_impl  # pylint: disable=g-direct-tensorflow-import


class SleepTest(tf.test.TestCase):
  # [CLASS DOC]
  # Test case for the sleep op wrappers. Shares one private check helper; no other fixtures.

  def _check_sleep(self, op):
    # [FUNCTION DOC]
    # HELPER: shared check for the sleep tests. Parameter: op, a sleep wrapper. Runs the op with a
    # delay of 0.3 seconds inside a traced function, and asserts the result is a scalar and that
    # the elapsed wall-clock time exceeds ninety percent of the delay. Returns nothing.
    """Check that one sleep op works in isolation.

    See sleep_bin.py for an example of how the synchronous and asynchronous
    sleep ops differ in behavior.

    Args:
      op: The sleep op, either sleep_op.SyncSleep or sleep_op.AsyncSleep.
    """
    delay = 0.3  # delay in seconds
    start_t = time.time()
    func = tf.function(lambda: op(delay))
    results = self.evaluate(func())
    end_t = time.time()
    delta_t = end_t - start_t
    self.assertEqual(results.shape, tuple())
    self.assertGreater(delta_t, 0.9 * delay)

  def test_sync_sleep(self):
    # [FUNCTION DOC]
    # Runs the shared check with the synchronous sleep op. Returns nothing.
    self._check_sleep(sleep_op.SyncSleep)

  def test_async_sleep(self):
    # [FUNCTION DOC]
    # Runs the shared check with the asynchronous sleep op. Returns nothing.
    self._check_sleep(sleep_op.AsyncSleep)

  def test_async_sleep_error(self):
    # [FUNCTION DOC]
    # Runs the asynchronous op with a delay of -1 in a traced function and asserts evaluation
    # raises an invalid-argument error saying the delay must be non-negative (the kernel must still
    # invoke its completion callback on early error). Returns nothing.
    # It is import that ComputeAsync() calls its done() callback if it returns
    # early due to an error.
    func = tf.function(lambda: sleep_op.AsyncSleep(-1.0))
    with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                'Input `delay` must be non-negative.'):
      self.evaluate(func())


if __name__ == '__main__':
  tf.test.main()
