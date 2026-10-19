# [MODULE DOC]
# Thread-local stack helpers used to track implicit defaults (for example default graphs and
# default sessions). Provides the generic DefaultStack class plus a module-level default-session
# stack with functions to install a session as default (a context manager) and to query the
# current default session. Import-time side effects: creates the global default-session stack
# and registers the query function for public export (version 1 API only).
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Classes used to handle thread-local stacks."""

from collections.abc import Iterator
import threading
from typing import Generic, Optional, TypeVar

from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export

T = TypeVar("T")


class DefaultStack(threading.local, Generic[T]):
  # [CLASS DOC]
  # A per-thread stack of default objects. Each thread sees its own independent stack (the
  # constructor runs once per thread on first access). Holds the stack list and a flag saying
  # whether pushes and pops must be strictly nested. The top element is the current default.
  """A thread-local stack of objects for providing implicit defaults."""

  def __init__(self):
    # [FUNCTION DOC]
    # Initializes the per-thread state: nesting enforcement turned on and an empty stack.
    # Called automatically for each thread that first touches the instance.
    super().__init__()
    self._enforce_nesting = True
    self.stack: list[T] = []

  def get_default(self) -> Optional[T]:
    # [FUNCTION DOC]
    # Returns the top (innermost) default for the calling thread, or nothing if the stack is
    # empty. Does not modify the stack.
    return self.stack[-1] if self.stack else None

  def reset(self) -> None:
    # [FUNCTION DOC]
    # Discards all entries of the calling thread's stack by replacing it with an empty list.
    # Returns nothing. Controllers still active will find the stack empty when they exit.
    self.stack = []

  def is_cleared(self) -> bool:
    # [FUNCTION DOC]
    # Returns true if the calling thread's stack has no entries, false otherwise.
    return not self.stack

  @property
  def enforce_nesting(self) -> bool:
    # [FUNCTION DOC]
    # Property returning whether exits from controllers must occur in strict last-in
    # first-out order for the calling thread.
    return self._enforce_nesting

  @enforce_nesting.setter
  def enforce_nesting(self, value: bool):
    # [FUNCTION DOC]
    # Setter for the nesting flag. value is a boolean: true requires strict nesting, false allows
    # out-of-order removal. Affects only the calling thread's state. Returns nothing.
    self._enforce_nesting = value

  @tf_contextlib.contextmanager
  def get_controller(self, default: T) -> Iterator[T]:
    # [FUNCTION DOC]
    # KNOWN ISSUE: when nesting is enforced and the top of the stack is not the object being
    # exited, an assertion error is raised before anything is removed, so the entry stays on the
    # stack and the stack is left corrupted for subsequent users.
    # Context manager that pushes default on the thread's stack, yields it, and on exit removes
    # it. With nesting enforced the entry must be on top (else raises an assertion error); with
    # nesting off the first matching entry is removed wherever it is. If the stack was reset
    # during the block, nothing is removed. Input: default is any object to install.
    """A context manager for manipulating a default stack."""
    self.stack.append(default)
    try:
      yield default
    finally:
      # stack may be empty if reset() was called
      if self.stack:
        if self._enforce_nesting:
          if self.stack[-1] is not default:
            raise AssertionError(
                "Nesting violated for default stack of %s objects" %
                type(default))
          self.stack.pop()
        else:
          self.stack.remove(default)


_default_session_stack = DefaultStack()


def default_session(session):
  # [FUNCTION DOC]
  # Returns a context manager that installs the given session object as the current thread's
  # default session for the duration of the with block, so tensor and operation evaluation
  # without an explicit session uses it. Input: session is any object offering the run interface.
  # The default applies to the calling thread only. Side effect: pushes onto the global
  # default-session stack on entry and removes on exit (assertion error on improper nesting).
  """Python "with" handler for defining a default session.

  This function provides a means of registering a session for handling
  Tensor.eval() and Operation.run() calls. It is primarily intended for use
  by session.Session, but can be used with any object that implements
  the Session.run() interface.

  Use with the "with" keyword to specify that Tensor.eval() and Operation.run()
  invocations within the scope of a block should be executed by a particular
  session.

  The default session applies to the current thread only, so it is always
  possible to inspect the call stack and determine the scope of a default
  session. If you create a new thread, and wish to use the default session
  in that thread, you must explicitly add a "with ops.default_session(sess):"
  block in that thread's function.

  Example:
    The following code examples are equivalent:

    # 1. Using the Session object directly:
    sess = ...
    c = tf.constant(5.0)
    sess.run(c)

    # 2. Using default_session():
    sess = ...
    with ops.default_session(sess):
      c = tf.constant(5.0)
      result = c.eval()

    # 3. Overriding default_session():
    sess = ...
    with ops.default_session(sess):
      c = tf.constant(5.0)
      with ops.default_session(...):
        c.eval(session=sess)

  Args:
    session: The session to be installed as the default session.

  Returns:
    A context manager for the default session.
  """
  return _default_session_stack.get_controller(session)


@tf_export(v1=["get_default_session"])
def get_default_session():
  # [FUNCTION DOC]
  # Returns the innermost default session of the calling thread, or nothing if none is active.
  # Takes no inputs; does not modify any state.
  """Returns the default session for the current thread.

  The returned `Session` will be the innermost session on which a
  `Session` or `Session.as_default()` context has been entered.

  NOTE: The default session is a property of the current thread. If you
  create a new thread, and wish to use the default session in that
  thread, you must explicitly add a `with sess.as_default():` in that
  thread's function.

  Returns:
    The default `Session` being used in the current thread.
  """
  return _default_session_stack.get_default()
