# [MODULE DOC]
# Tests for the native events writer wrapper: writing events to a file and reading them back, and
# rejecting objects that are not events. Imports the native wrapper modules; running as main
# starts the test runner.

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
"""Tests for the SWIG-wrapped events writer."""

import os.path

from tensorflow.core.framework import summary_pb2
from tensorflow.core.util import event_pb2
# pylint: disable=invalid-import-order, g-bad-import-order, wildcard-import, unused-import, undefined-variable
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import _pywrap_events_writer
from tensorflow.python.framework import errors
from tensorflow.python.framework import test_util
from tensorflow.python.lib.io import tf_record
from tensorflow.python.platform import googletest
from tensorflow.python.util import compat


class PywrapeventsWriterTest(test_util.TensorFlowTestCase):
  # [CLASS DOC]
  # Test case for the events writer wrapper.

  def testWriteEvents(self):
    # [FUNCTION DOC]
    # Creates a writer with a file prefix in the temp directory, writes one summary event, flushes
    # and closes. Asserts reading a nonexistent file raises a not-found error; that the first
    # record read is the file-version event; that the second equals the written event (time, step,
    # summary value); and that reading further ends the iteration.
    file_prefix = os.path.join(self.get_temp_dir(), "events")
    writer = _pywrap_events_writer.EventsWriter(compat.as_bytes(file_prefix))
    filename = compat.as_text(writer.FileName())
    event_written = event_pb2.Event(
        wall_time=123.45,
        step=67,
        summary=summary_pb2.Summary(
            value=[summary_pb2.Summary.Value(
                tag="foo", simple_value=89.0)]))
    writer.WriteEvent(event_written)
    writer.Flush()
    writer.Close()

    with self.assertRaises(errors.NotFoundError):
      for r in tf_record.tf_record_iterator(filename + "DOES_NOT_EXIST"):
        self.assertTrue(False)

    reader = tf_record.tf_record_iterator(filename)
    event_read = event_pb2.Event()

    event_read.ParseFromString(next(reader))
    self.assertTrue(event_read.HasField("file_version"))

    event_read.ParseFromString(next(reader))
    # Second event
    self.assertProtoEquals("""
    wall_time: 123.45 step: 67
    summary { value { tag: 'foo' simple_value: 89.0 } }
    """, event_read)

    with self.assertRaises(StopIteration):
      next(reader)

  def testWriteEventInvalidType(self):
    # [FUNCTION DOC]
    # Passes an object that is not an event to the writer's write operation and asserts a type
    # error whose message mentions the object's text form is raised.

    class _Invalid(object):
      # [CLASS DOC]
      # Test helper: plain object that is not an event; its text form is the word Invalid.

      def __str__(self):
        # [FUNCTION DOC]
        # Returns the fixed text Invalid.
        return "Invalid"

    with self.assertRaisesRegex(TypeError, "Invalid"):
      _pywrap_events_writer.EventsWriter(b"foo").WriteEvent(_Invalid())


if __name__ == "__main__":
  googletest.main()
