# [MODULE DOC]
# HELPER: Helper module for tests that inspect summary output.
# Provides two functions to read back summary event protos written by summary writers: one
# reading a single event file, one reading the only event file inside a log directory. No
# import-time side effects.
# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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

"""Utilities to test summaries."""

import os

from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record
from tensorflow.python.platform import gfile


def events_from_file(filepath):
  # [FUNCTION DOC]
  # HELPER: Test utility for reading summary event files.
  # Input: filepath is the path of a record-format event file. Reads every record, parses each as
  # an event protocol message and returns the list of events in file order. Parse or file errors
  # from the record reader propagate. Read-only; no other side effects.
  """Returns all events in a single event file.

  Args:
    filepath: Path to the event file.

  Returns:
    A list of all tf.compat.v1.Event protos in the event file.
  """
  records = list(tf_record.tf_record_iterator(filepath))
  result = []
  for r in records:
    event = event_pb2.Event()
    event.ParseFromString(r)
    result.append(event)
  return result


def events_from_logdir(logdir):
  # [FUNCTION DOC]
  # HELPER: Test utility for reading summary output from a log directory.
  # KNOWN ISSUE: the docstring mentions only the one-file requirement, but the code also
  # asserts that the directory exists, and it uses plain assertions (skipped when Python runs
  # with optimizations), so failures are assertion errors rather than proper argument errors.
  # Input: logdir is a directory expected to hold exactly one event file. Returns the events
  # from that file in order. Raises an assertion error if the directory is missing or does not
  # contain exactly one entry.
  """Returns all events in the single eventfile in logdir.

  Args:
    logdir: The directory in which the single event file is sought.

  Returns:
    A list of all tf.compat.v1.Event protos from the single event file.

  Raises:
    AssertionError: If logdir does not contain exactly one file.
  """
  assert gfile.Exists(logdir)
  files = gfile.ListDirectory(logdir)
  assert len(files) == 1, 'Found not exactly one file in logdir: %s' % files
  return events_from_file(os.path.join(logdir, files[0]))
