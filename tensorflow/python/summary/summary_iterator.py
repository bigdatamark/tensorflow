# [MODULE DOC]
# Provides the version 1 helper for reading event records back from an event file. Exposes the
# summary_iterator function (exported to the public API under the training namespace for
# version 1) and a private iterator class that decodes each record of the file as an event
# message. Depends on the record-file reader. No import-time side effects besides the export
# registration.
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

"""Provides a method for reading events from an event file via an iterator."""

from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record
from tensorflow.python.util.tf_export import tf_export


class _SummaryIterator(object):
  # [CLASS DOC]
  # HELPER: implementation detail of summary_iterator.
  # Iterator over the event messages stored in one event file. Holds a record iterator over the
  # file and decodes each raw record into an event message. It is its own iterator. The alias
  # named next exists for old-style iteration compatibility.
  """Yields `Event` protocol buffers from a given path."""

  def __init__(self, path):
    # [FUNCTION DOC]
    # Constructor: opens the event file at path (a string file path) for record reading by
    # creating the underlying record iterator, which stays open for the iterator's lifetime.
    # Errors for missing or unreadable files come from the record reader.
    self._tf_record_iterator = tf_record.tf_record_iterator(path)

  def __iter__(self):
    # [FUNCTION DOC]
    # Iterator protocol: returns this same object.
    return self

  def __next__(self):
    # [FUNCTION DOC]
    # Iterator protocol: reads the next raw record from the file and returns it parsed as an event
    # message. Raises a stop-iteration error at end of file (propagated from the record reader);
    # the record reader allows later calls to succeed if more data has been appended meanwhile.
    r = next(self._tf_record_iterator)
    return event_pb2.Event.FromString(r)

  next = __next__


@tf_export(v1=['train.summary_iterator'])
def summary_iterator(path):
  # [FUNCTION DOC]
  # Public version 1 entry point for reading an event file. Input: path, the string path of an
  # event file written by a summary writer. Returns an iterator object that yields one event
  # message per record, in file order. Opening happens immediately; nothing is read until
  # iteration. Does not close or wait; at end of file the iterator raises stop-iteration but can
  # be advanced again to pick up newly appended events.
  # pylint: disable=line-too-long
  """Returns a iterator for reading `Event` protocol buffers from an event file.

  You can use this function to read events written to an event file. It returns
  a Python iterator that yields `Event` protocol buffers.

  Example: Print the contents of an events file.

  ```python
  for e in tf.compat.v1.train.summary_iterator(path to events file):
      print(e)
  ```

  Example: Print selected summary values.

  ```python
  # This example supposes that the events file contains summaries with a
  # summary value tag 'loss'.  These could have been added by calling
  # `add_summary()`, passing the output of a scalar summary op created with
  # with: `tf.compat.v1.summary.scalar('loss', loss_tensor)`.
  for e in tf.compat.v1.train.summary_iterator(path to events file):
      for v in e.summary.value:
          if v.tag == 'loss':
              print(tf.make_ndarray(v.tensor))
  ```
  Example: Continuously check for new summary values.

  ```python
  summaries = tf.compat.v1.train.summary_iterator(path to events file)
  while True:
    for e in summaries:
        for v in e.summary.value:
            if v.tag == 'loss':
                print(tf.make_ndarray(v.tensor))
    # Wait for a bit before checking the file for any new events
    time.sleep(wait time)
  ```

  See the protocol buffer definitions of
  [Event](https://www.tensorflow.org/code/tensorflow/core/util/event.proto)
  and
  [Summary](https://www.tensorflow.org/code/tensorflow/core/framework/summary.proto)
  for more information about their attributes.

  Args:
    path: The path to an event file created by a `SummaryWriter`.

  Returns:
    A iterator that yields `Event` protocol buffers
  """
  return _SummaryIterator(path)
