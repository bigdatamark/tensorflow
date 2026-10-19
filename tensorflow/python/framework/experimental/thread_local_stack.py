# [MODULE DOC]
# HELPER: internal helper module for the experimental unified-API framework code.
# Defines a minimal per-thread stack class used to hold implicit default objects (such as
# execution contexts). Has no import-time side effects.
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
"""Thread-local stack."""

import threading


# TODO(srbs): Move this to C++.
class ThreadLocalStack(threading.local):
  # [CLASS DOC]
  # HELPER: stack of objects whose contents are separate for every thread (it derives from the
  # thread-local base class, so the constructor runs again in each thread on first use). Holds a
  # list used as a stack and offers peek, push and pop. Used to provide implicit defaults.
  """A thread-local stack of objects for providing implicit defaults."""

  def __init__(self):
    # [FUNCTION DOC]
    # HELPER: constructor, run once per thread. Creates the empty list that backs the stack.
    # Returns nothing.
    super(ThreadLocalStack, self).__init__()
    self._stack = []

  def peek(self):
    # [FUNCTION DOC]
    # HELPER: returns the top element of the calling thread's stack without removing it, or
    # nothing (a null value) if the stack is empty.
    return self._stack[-1] if self._stack else None

  def push(self, ctx):
    # [FUNCTION DOC]
    # HELPER: puts ctx (any object) on top of the calling thread's stack. Returns nothing (the
    # list append returns nothing, so the returned value is always null).
    return self._stack.append(ctx)

  def pop(self):
    # [FUNCTION DOC]
    # HELPER: removes the top element of the calling thread's stack. Returns nothing (the removed
    # element is discarded). Raises an index error if the stack is empty.
    self._stack.pop()
