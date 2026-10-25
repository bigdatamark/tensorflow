# [MODULE DOC]
# Test module for the summary iterator: writes events with a file writer and checks that the
# iterator yields them in order and can resume reading after reaching end of file when new
# events are appended. Run as a test binary.
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
"""Tests for tensorflow.python.summary.summary_iterator."""

import glob
import os.path

from tensorflow.core.util import event_pb2
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.summary import summary_iterator
from tensorflow.python.summary.writer import writer


class SummaryIteratorTestCase(test.TestCase):
  # [CLASS DOC]
  # Test case for the summary iterator reading events produced by a version 1 file writer into a
  # temporary directory. No shared fixtures.

  @test_util.run_deprecated_v1
  def testSummaryIteratorEventsAddedAfterEndOfFile(self):
    # [FUNCTION DOC]
    # Test (runs in version 1 mode): writes a session-start log at step one, flushes, and opens an
    # iterator on the event file. Asserts the first event carries the file version marker, the
    # second is the step one start event, and then end of file raises stop-iteration. Appends a
    # step two start log, flushes, and asserts the same iterator now yields it, then hits end of
    # file again.
    test_dir = os.path.join(self.get_temp_dir(), "events")
    with writer.FileWriter(test_dir) as w:
      session_log_start = event_pb2.SessionLog.START
      w.add_session_log(event_pb2.SessionLog(status=session_log_start), 1)
      w.flush()
      path = glob.glob(os.path.join(test_dir, "event*"))[0]
      rr = summary_iterator.summary_iterator(path)
      # The first event should list the file_version.
      ev = next(rr)
      self.assertEqual("brain.Event:2", ev.file_version)
      # The next event should be the START message.
      ev = next(rr)
      self.assertEqual(1, ev.step)
      self.assertEqual(session_log_start, ev.session_log.status)
      # Reached EOF.
      self.assertRaises(StopIteration, lambda: next(rr))
      w.add_session_log(event_pb2.SessionLog(status=session_log_start), 2)
      w.flush()
      # The new event is read, after previously seeing EOF.
      ev = next(rr)
      self.assertEqual(2, ev.step)
      self.assertEqual(session_log_start, ev.session_log.status)
      # Get EOF again.
      self.assertRaises(StopIteration, lambda: next(rr))

if __name__ == "__main__":
  test.main()
