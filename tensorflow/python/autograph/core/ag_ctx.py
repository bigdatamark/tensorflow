# [MODULE DOC]
# Thread-local control status context for AutoGraph. Maintains, per thread, a stack of control
# status objects recording whether the user has enabled, disabled or left unspecified AutoGraph
# conversion (plus optional conversion options). Exposes the public-internal accessor for the
# current context, a status enumeration, a context-manager class that pushes and pops entries,
# and a no-op context manager. Import-time side effects: creates the module-level thread-local
# storage, and probes whether source code can be retrieved for functions (setting a module flag);
# if source is unavailable (e.g. interactive shells) it logs a warning that AutoGraph is not
# available in this environment.
# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Thread-local context managers for AutoGraph."""

import enum
import inspect
import threading

from tensorflow.python.autograph.utils import ag_logging
from tensorflow.python.util.tf_export import tf_export


stacks = threading.local()


def _control_ctx():
  # [FUNCTION DOC]
  # HELPER: accessor for the per-thread control status stack used by the context classes.
  # Returns the list (stack) of control status contexts for the current thread, lazily creating
  # it on first use with a single default entry whose status is unspecified. Mutates the
  # thread-local storage on first call per thread.
  if not hasattr(stacks, 'control_status'):
    stacks.control_status = [_default_control_status_ctx()]
  return stacks.control_status


@tf_export('__internal__.autograph.control_status_ctx', v1=[])
def control_status_ctx():
  # [FUNCTION DOC]
  # Returns the innermost (top of stack) control status context of the current thread. Takes no
  # arguments. May create the thread's default stack as a side effect. Exported under the
  # internal autograph API namespace for use with the tf convert helper.
  """Returns the current control context for autograph.

  This method is useful when calling `tf.__internal__.autograph.tf_convert`,
  The context will be used by tf_convert to determine whether it should convert
  the input function. See the sample usage like below:

  ```
  def foo(func):
    return tf.__internal__.autograph.tf_convert(
       input_fn, ctx=tf.__internal__.autograph.control_status_ctx())()
  ```

  Returns:
    The current control context of autograph.
  """
  ret = _control_ctx()[-1]
  return ret


class Status(enum.Enum):
  # [CLASS DOC]
  # Enumeration of the user-controlled AutoGraph state: unspecified (zero), enabled (one) and
  # disabled (two).
  UNSPECIFIED = 0
  ENABLED = 1
  DISABLED = 2


class ControlStatusCtx(object):
  # [CLASS DOC]
  # Context manager recording whether AutoGraph is enabled by the user, with optional conversion
  # options. Attributes: status (a Status value) and options (arbitrary options object or
  # nothing). Entering pushes the instance on the current thread's stack; exiting pops it, so
  # nested uses form a stack that control_status_ctx reads.
  """A context that tracks whether autograph is enabled by the user."""

  def __init__(self, status, options=None):
    # [FUNCTION DOC]
    # Constructor. Inputs: status (a Status value) and optional options (default nothing).
    # Stores both. Returns nothing.
    self.status = status
    self.options = options

  def __enter__(self):
    # [FUNCTION DOC]
    # Context entry: pushes this instance on the current thread's control stack and returns
    # itself.
    _control_ctx().append(self)
    return self

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a readable string with the class name, status and options.
    return '{}[status={}, options={}]'.format(
        self.__class__.__name__, self.status, self.options)

  def __exit__(self, unused_type, unused_value, unused_traceback):
    # [FUNCTION DOC]
    # Context exit: asserts that this instance is on top of the thread's control stack (an
    # assertion error means contexts were exited out of order) and pops it. Exception
    # information arguments are ignored; exceptions are not suppressed.
    assert _control_ctx()[-1] is self
    _control_ctx().pop()


class NullCtx(object):
  # [CLASS DOC]
  # HELPER: stand-in for the standard library null context, usable where a context manager is
  # required but nothing should happen. Stateless; entering and exiting do nothing.
  """Helper substitute for contextlib.nullcontext."""

  def __enter__(self):
    # [FUNCTION DOC]
    # Context entry: does nothing and returns nothing (unlike the standard null context it does
    # not return a value).
    pass

  def __exit__(self, unused_type, unused_value, unused_traceback):
    # [FUNCTION DOC]
    # Context exit: does nothing, ignores exception information and does not suppress errors.
    pass


def _default_control_status_ctx():
  # [FUNCTION DOC]
  # HELPER: factory for the initial bottom-of-stack entry of the control stack.
  # Returns a new control status context with unspecified status and no options.
  return ControlStatusCtx(status=Status.UNSPECIFIED)


INSPECT_SOURCE_SUPPORTED = True
try:
  inspect.getsource(ag_logging.log)
except OSError:
  INSPECT_SOURCE_SUPPORTED = False
  ag_logging.warning(
      'AutoGraph is not available in this environment: functions lack code'
      ' information. This is typical of some environments like the interactive'
      ' Python shell. See'
      ' https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/limitations.md#access-to-source-code'
      ' for more information.')
