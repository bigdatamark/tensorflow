# [MODULE DOC]
# Provides a small stack data structure whose entries remember the source file name and line
# number at which each object was pushed. Exposes a generic wrapper class that pairs an object
# with its definition location and a generic stack class built from such wrappers. Used by the
# framework (for example to report where graph constructs were created) and relies on frame
# inspection from the standard library. Defines a module-level type variable but has no other
# import-time side effects.
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
"""A simple stack that associates filename and line numbers with each object."""

from collections.abc import Iterator
import inspect
import types

from typing import cast, Generic, Optional, TypeVar

T = TypeVar("T")


class TraceableObject(Generic[T]):
  # [CLASS DOC]
  # Wraps one arbitrary object together with the file name and line number where it was defined
  # or stored. Attributes: the wrapped object, the file name and the line number (the last two
  # are nothing/null until set). Class constants give the three result codes returned by the
  # frame-capturing method: success, heuristic used, and failure. Created directly with an
  # object (and optionally a location) or by the stack class when pushing.
  """Wrap an object together with its the code definition location."""

  # Return codes for the set_filename_and_line_from_caller() method.
  SUCCESS, HEURISTIC_USED, FAILURE = (0, 1, 2)

  def __init__(
      self,
      obj: T,
      filename: Optional[str] = None,
      lineno: Optional[int] = None,
  ):
    # [FUNCTION DOC]
    # Constructor. Inputs: obj, the object to wrap (any type); filename, optional text naming the
    # source file (default nothing); lineno, optional integer line number (default nothing).
    # Stores all three as attributes. Returns nothing, raises nothing.
    self.obj = obj
    self.filename = filename
    self.lineno = lineno

  def set_filename_and_line_from_caller(self, offset: int = 0) -> int:
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring refers to "self.HEURISTIC USED" (with a space) which is not an
    # attribute name; the real constant is the one with an underscore. The docstring also
    # suggests the heuristic can "fail" separately; in the code the failure code is returned only
    # when the current frame cannot be obtained at all.
    # Walks up the call stack from this method by offset plus one frames (so offset zero means
    # the direct caller of this method). If the stack runs out early, it stops at the outermost
    # frame and the heuristic-used code is returned. Then records that frame's file name and
    # current line number into the object's attributes (overwriting earlier values).
    # Input: offset, a non-negative integer, default zero. Returns an integer result code: success,
    # heuristic used, or failure (frame inspection unavailable; attributes left unchanged).
    # Side effect: mutates the filename and line number attributes. Raises nothing.
    """Set filename and line using the caller's stack frame.

    If the requested stack information is not available, a heuristic may
    be applied and self.HEURISTIC USED will be returned.  If the heuristic
    fails then no change will be made to the filename and lineno members
    (None by default) and self.FAILURE will be returned.

    Args:
      offset: Integer.  If 0, the caller's stack frame is used.  If 1,
          the caller's caller's stack frame is used.  Larger values are
          permissible but if out-of-range (larger than the number of stack
          frames available) the outermost stack frame will be used.

    Returns:
      TraceableObject.SUCCESS if appropriate stack information was found,
      TraceableObject.HEURISTIC_USED if the offset was larger than the stack,
      and TraceableObject.FAILURE if the stack was empty.
    """
    retcode = self.SUCCESS
    frame = inspect.currentframe()
    if not frame:
      return self.FAILURE
    frame = cast(types.FrameType, frame)
    # Offset is defined in "Args" as relative to the caller. We are one frame
    # beyond the caller.
    for _ in range(offset + 1):
      parent = frame.f_back
      if parent is None:
        # If the offset is too large then we use the largest offset possible.
        retcode = self.HEURISTIC_USED
        break
      parent = cast(types.FrameType, parent)
      frame = parent
    self.filename = frame.f_code.co_filename
    self.lineno = cast(int, frame.f_lineno)
    return retcode

  def copy_metadata(self):
    # [FUNCTION DOC]
    # Takes no inputs. Returns a new wrapper of the same class (so subclasses are preserved)
    # holding nothing/null as the object but carrying over this instance's file name and line
    # number. Useful for keeping location info without retaining a reference to the object.
    # No side effects on this instance.
    """Return a TraceableObject like this one, but without the object."""
    return self.__class__(None, filename=self.filename, lineno=self.lineno)


class TraceableStack(Generic[T]):
  # [CLASS DOC]
  # Last-in first-out stack of wrapped objects, each recording the file and line where it was
  # pushed. Holds a single private list of wrapper objects (oldest first). Supports pushing,
  # popping, peeking at the newest object, iterating newest to oldest, length and truth testing,
  # and shallow copying (intended for thread-local stacks).
  """A stack of TraceableObjects."""

  def __init__(
      self, existing_stack: Optional[list[TraceableObject[T]]] = None,
  ):
    # [FUNCTION DOC]
    # Constructor. Input: existing_stack, an optional list of wrapper objects (default nothing).
    # If given and non-empty, the new stack starts with a shallow copy of that list (the wrapper
    # objects are shared, the list is not); otherwise it starts empty. Returns nothing.
    """Constructor.

    Args:
      existing_stack: [TraceableObject, ...] If provided, this object will
        set its new stack to a SHALLOW COPY of existing_stack.
    """
    self._stack: list[TraceableObject[T]] = (existing_stack[:] if existing_stack
                                             else [])

  def push_obj(self, obj: T, offset: int = 0):
    # [FUNCTION DOC]
    # Pushes an object on top of the stack, wrapped so that it records the caller's file and
    # line. Inputs: obj, any object; offset, non-negative integer, default zero (zero records the
    # direct caller of this method, one records the caller's caller). Returns the integer result
    # code from the location capture: success, heuristic used, or failure. Side effect: appends
    # to the internal list (the object is added even if capture fails). Raises nothing.
    """Add object to the stack and record its filename and line information.

    Args:
      obj: An object to store on the stack.
      offset: Integer.  If 0, the caller's stack frame is used.  If 1,
          the caller's caller's stack frame is used.

    Returns:
      TraceableObject.SUCCESS if appropriate stack information was found,
      TraceableObject.HEURISTIC_USED if the stack was smaller than expected,
      and TraceableObject.FAILURE if the stack was empty.
    """
    traceable_obj = TraceableObject(obj)
    self._stack.append(traceable_obj)
    # Offset is defined in "Args" as relative to the caller.  We are 1 frame
    # beyond the caller and need to compensate.
    return traceable_obj.set_filename_and_line_from_caller(offset + 1)

  def pop_obj(self) -> T:
    # [FUNCTION DOC]
    # Removes the most recently pushed entry and returns only the original object, discarding
    # its file and line info. Raises an index error if the stack is empty. Mutates the stack.
    """Remove last-inserted object and return it, without filename/line info."""
    return self._stack.pop().obj

  def peek_top_obj(self) -> T:
    # [FUNCTION DOC]
    # Returns the most recently pushed object without removing it. Raises an index error if the
    # stack is empty. No side effects.
    """Return the most recent stored object."""
    return self._stack[-1].obj

  def peek_objs(self) -> Iterator[T]:
    # [FUNCTION DOC]
    # Returns a lazy iterator over the stored objects (not the wrappers), newest first. Does not
    # modify the stack; the iterator reflects the list as it is consumed, so changing the stack
    # while iterating is unsafe.
    """Return iterator over stored objects ordered newest to oldest."""
    return (t_obj.obj for t_obj in reversed(self._stack))

  def peek_traceable_objs(self) -> Iterator[TraceableObject[T]]:
    # [FUNCTION DOC]
    # Returns a lazy iterator over the wrapper objects (including file and line info), newest
    # first. Does not modify the stack; changing it while iterating is unsafe.
    """Return iterator over stored TraceableObjects ordered newest to oldest."""
    return reversed(self._stack)

  def __len__(self) -> int:
    # [FUNCTION DOC]
    # Returns the number of entries on the stack as an integer. Because of this, an empty stack
    # is treated as false in truth-value tests and a non-empty one as true. No side effects.
    """Return number of items on the stack, and used for truth-value testing."""
    return len(self._stack)

  def copy(self) -> "TraceableStack[T]":
    # [FUNCTION DOC]
    # Returns a new stack whose list is a shallow copy of this one's: same wrapper objects, new
    # list, so later pushes or pops on either stack do not affect the other. Intended for
    # thread-local stack inheritance. Note an empty stack copies to a fresh empty stack.
    """Return a copy of self referencing the same objects but in a new list.

    This method is implemented to support thread-local stacks.

    Returns:
      TraceableStack with a new list that holds existing objects.
    """
    return TraceableStack(self._stack)
