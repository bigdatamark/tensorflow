# [MODULE DOC]
# HELPER: tiny helper module that holds the per-thread worker context for the distribute
# coordinator; kept separate so the coordinator and its users avoid circular imports.
# Stores a thread-local holder (module-level state) in which the coordinator publishes the
# worker context of the task currently running on this thread, and exposes a single accessor
# to read it. Import has no side effects besides creating the thread-local holder.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""The context retrieval method for distribute coordinator."""

import threading

_worker_context = threading.local()


def get_current_worker_context():
  # [FUNCTION DOC]
  # Returns the worker context object that the coordinator installed for the calling thread,
  # or a null value when no worker context is active on this thread (including when the
  # thread never had one). Takes no inputs, has no side effects, raises nothing.
  """Returns the current task context."""
  try:
    return _worker_context.current
  except AttributeError:
    return None
