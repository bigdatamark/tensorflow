# [MODULE DOC]
# Process-wide cache of FileWriter objects keyed by log directory. Defines FileWriterCache, exported
# in the public v1 summary API, which creates at most one writer per directory under a re-entrant
# lock. Used by components such as monitored sessions and hooks that need a shared writer.
# Import-time side effects: registers the cache class in the API export table.
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
"""A cache for FileWriters."""

import threading

from tensorflow.python.framework import ops
from tensorflow.python.summary.writer.writer import FileWriter
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=['summary.FileWriterCache'])
class FileWriterCache(object):
  # [CLASS DOC]
  # Static registry (never instantiated) holding a class-level mapping from log directory to
  # FileWriter and a class-level re-entrant lock guarding it. Offers static get and clear methods.
  # KNOWN ISSUE: the comment next to the lock refers to a stale name (a file writers variable);
  # the lock actually protects the cache mapping.
  """Cache for file writers.

  This class caches file writers, one per directory.
  """
  # Cache, keyed by directory.
  _cache = {}

  # Lock protecting _FILE_WRITERS.
  _lock = threading.RLock()

  @staticmethod
  def clear():
    # [FUNCTION DOC]
    # Static method. Under the lock, closes every cached writer (to release file handles) and then
    # replaces the cache with an empty mapping. Takes no inputs, returns nothing. Errors from
    # closing a writer propagate and leave the cache unchanged.
    """Clear cached summary writers. Currently only used for unit tests."""
    with FileWriterCache._lock:
      # Make sure all the writers are closed now (otherwise open file handles
      # may hang around, blocking deletions on Windows).
      for item in FileWriterCache._cache.values():
        item.close()
      FileWriterCache._cache = {}

  @staticmethod
  def get(logdir):
    # [FUNCTION DOC]
    # Static method. Input: logdir, the directory name (string). Under the lock, returns the cached
    # FileWriter for that directory, creating it on first use with the current default graph (so
    # the graph is written to the new event file). Side effect: creates the directory and events
    # file for a new writer. Writers created earlier are returned even if closed or if the default
    # graph differs.
    """Returns the FileWriter for the specified directory.

    Args:
      logdir: str, name of the directory.

    Returns:
      A `FileWriter`.
    """
    with FileWriterCache._lock:
      if logdir not in FileWriterCache._cache:
        FileWriterCache._cache[logdir] = FileWriter(
            logdir, graph=ops.get_default_graph())
      return FileWriterCache._cache[logdir]
