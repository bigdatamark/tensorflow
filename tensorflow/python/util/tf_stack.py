# [MODULE DOC]
# Python front end for fast stack-trace extraction backed by the native stack module. Defines
# per-thread stacks of source mappers (which remap file and line to other locations, e.g. for
# generated code) and of filters (sets of file names whose frames are hidden), context-manager
# base classes to push and pop them, a stack extraction function that applies the innermost
# mapper and filter of the calling thread, helpers to load traces from graph debug info and to
# build debug info, and aliases for the native stack and frame summary types.
# Import-time side effects: creates the module-level per-thread stack tables (each thread's
# stack starts with a no-op sentinel mapper or filter, created lazily). Part of the traceback
# machinery used by graph construction and error reporting.
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
"""Functions used to extract and analyze stacks.  Faster than Python libs."""
# pylint: disable=g-bad-name
import collections
import inspect
import threading

from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.python.util import _tf_stack

# Generally such lookups should be done using `threading.local()`. See
# https://blogs.gnome.org/jamesh/2008/06/11/tls-python/ for a detailed
# explanation of why. However the transform stacks are expected to be empty
# when a thread is joined, so reusing the key does not introduce a correctness
# issue. Moreover, get_ident is faster than storing and retrieving a unique
# key in a thread local store.
_get_thread_key = threading.get_ident


# TODO(mdan): Move these to C++ as well.
# Moving to C++ can further avoid extra copies made by get_effective_map.
_source_mapper_stacks = collections.defaultdict(lambda: [SentinelMapper()])
_source_filter_stacks = collections.defaultdict(lambda: [SentinelFilter()])


class StackTraceTransform(object):
  # [CLASS DOC]
  # Abstract context-manager base class for objects that transform how stack traces are
  # produced (mappers and filters). Subclasses provide the per-thread stack table through a
  # class attribute and implement the update method. Entering pushes the instance on the
  # current thread's stack (remembering the previous top as the parent) and refreshes the native
  # structure; exiting pops it. Holds the owning thread key (set on first entry; an instance must
  # not be shared between threads) and the parent transform.
  """Base class for stack trace transformation functions."""

  _stack_dict = None  # Subclasses should override
  _thread_key = None

  def __enter__(self):
    # [FUNCTION DOC]
    # Context-manager entry. Records the calling thread's key on first use (asserting on later
    # uses that the same thread is entering, otherwise an assertion error "Shared across
    # threads?"), looks up that thread's stack in the subclass-provided table, saves the current
    # top as the parent, pushes this instance, calls the update method and returns the instance.
    # Side effect: mutates the shared per-thread stack and the native structure.
    # Any given instance is assumed to be used by a single thread, which reduces
    # expensive thread local lookups.
    if self._thread_key is None:
      self._thread_key = _get_thread_key()
    else:
      assert self._thread_key == _get_thread_key(), 'Shared across threads?'

    stack = self._stack_dict[self._thread_key]
    self.parent = stack[-1]
    stack.append(self)
    self.update()
    return self

  def __exit__(self, unused_type, unused_value, unused_traceback):
    # [FUNCTION DOC]
    # Context-manager exit. Ignores the exception arguments (does not suppress exceptions),
    # pops the top of this thread's stack and asserts it is this instance (assertion error
    # "Concurrent access?" otherwise). Returns nothing. Note the native structure is not
    # refreshed to the parent on exit.
    top = self._stack_dict[self._thread_key].pop()
    assert top is self, 'Concurrent access?'

  def update(self):
    # [FUNCTION DOC]
    # Abstract hook called on entry to push the transform's data into the native structure.
    # Always raises a not-implemented error; subclasses must override.
    raise NotImplementedError('subclasses need to override this')


class StackTraceMapper(StackTraceTransform):
  # [CLASS DOC]
  # Abstract transform that remaps traceback locations to other source code. Uses the shared
  # per-thread table of mapper stacks and owns a native source-map object (internal map) that
  # extraction consults. Subclasses implement the method returning the effective mapping.
  """Allows remapping traceback information to different source code."""
  _stack_dict = _source_mapper_stacks

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor. Takes no inputs; creates an empty native source map held in the internal map
    # attribute. Does not call the base constructor (the base has none of its own).
    self.internal_map = _tf_stack.PyBindSourceMap()

  def update(self):
    # [FUNCTION DOC]
    # Refreshes the native source map from this mapper: converts the effective mapping into a
    # tuple of key and value pairs and loads it into the native map. Returns nothing.
    self.internal_map.update_to(tuple(self.get_effective_source_map().items()))

  def get_effective_source_map(self):
    # [FUNCTION DOC]
    # Abstract method: subclasses return a mapping from (file name, line number) pairs to
    # (file name, line number, function name) triples. Always raises a not-implemented error here.
    """Returns a map (filename, lineno) -> (filename, lineno, function_name)."""
    raise NotImplementedError('subclasses need to override this')


EMPTY_DICT = {}


class SentinelMapper(StackTraceMapper):
  # [CLASS DOC]
  # Do-nothing mapper placed at the bottom of every thread's mapper stack so the stack is never
  # empty. Its effective mapping is always the shared empty mapping.

  def get_effective_source_map(self):
    # [FUNCTION DOC]
    # Returns the shared empty mapping (no locations are remapped). Takes no inputs.
    return EMPTY_DICT


class StackTraceFilter(StackTraceTransform):
  # [CLASS DOC]
  # Abstract transform that hides stack frames belonging to certain files. Uses the per-thread
  # table of filter stacks and owns a native file set (internal set) consulted by extraction.
  # Subclasses implement the method returning the file names to hide.
  """Allows filtering traceback information by removing superfluous frames."""
  _stack_dict = _source_filter_stacks

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor. Takes no inputs; creates an empty native file set held in the internal set
    # attribute.
    self.internal_set = _tf_stack.PyBindFileSet()

  def update(self):
    # [FUNCTION DOC]
    # Refreshes the native file set from this filter: loads the set of filtered file names
    # returned by the subclass into the native set. Returns nothing.
    self.internal_set.update_to(set(self.get_filtered_filenames()))

  def get_filtered_filenames(self):
    # [FUNCTION DOC]
    # Abstract method: subclasses return a collection of file names whose frames are hidden.
    # Always raises a not-implemented error here.
    raise NotImplementedError('subclasses need to override this')


EMPTY_SET = frozenset()


class SentinelFilter(StackTraceFilter):
  # [CLASS DOC]
  # Do-nothing filter placed at the bottom of every thread's filter stack so the stack is never
  # empty. It hides no files.

  def get_filtered_filenames(self):
    # [FUNCTION DOC]
    # Returns the shared empty frozen set (no files are filtered). Takes no inputs.
    return EMPTY_SET


class CurrentModuleFilter(StackTraceFilter):
  # [CLASS DOC]
  # Filter that hides the frames of the file that created it (best effort: the file is found
  # from the caller's frame and may be unknown, in which case the null name is filtered). When
  # nested inside other filters it also includes the filtered files of its parent. Caches its
  # computed set after the first computation.
  """Filters stack frames from the module where this is used (best effort)."""

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor. Takes no inputs. Calls the base constructor, then inspects the interpreter
    # stack to find the file of the immediate caller and stores it as the file to filter (stays
    # the null value if the frame or file is unavailable); clears the cached set. Deletes local
    # frame references in a final step to avoid reference cycles.
    super().__init__()
    filter_filename = None
    outer_f = None
    f = inspect.currentframe()
    try:
      if f is not None:
        # The current frame is __init__. The first outer frame should be the
        # caller.
        outer_f = f.f_back
        if outer_f is not None:
          filter_filename = inspect.getfile(outer_f)
      self._filename = filter_filename
      # This may be called repeatedly: once on entry by the superclass, then by
      # each child context manager.
      self._cached_set = None
    finally:
      # Avoid reference cycles, see:
      # https://docs.python.org/3.7/library/inspect.html#the-interpreter-stack
      del f
      del outer_f

  def get_filtered_filenames(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: relies on the parent attribute, which exists only after the context manager
    # has been entered; calling this before entering would raise an attribute error. Also the
    # cached result is never invalidated.
    # Returns the cached frozen set if present. Otherwise builds a frozen set holding this
    # filter's file name, merges in the parent's filtered file names when a parent exists, caches
    # and returns it.
    if self._cached_set is not None:
      return self._cached_set

    filtered_filenames = frozenset((self._filename,))
    if self.parent is not None:
      filtered_filenames |= self.parent.get_filtered_filenames()
    self._cached_set = filtered_filenames
    return filtered_filenames


def extract_stack(stacklevel=1):
  # [FUNCTION DOC]
  # Captures the current Python call stack using the native extractor. Input: stacklevel
  # (integer, default 1), number of innermost frames to skip. Uses the calling thread's current
  # (innermost) source mapper and file filter. Returns a list-like summary of frame objects with
  # file name, line number, function name and source line. No side effects.
  """An eager-friendly alternative to traceback.extract_stack.

  Args:
    stacklevel: number of initial frames to skip when producing the stack.

  Returns:
    A list-like FrameSummary containing StackFrame-like objects, which are
    namedtuple-like objects with the following fields: filename, lineno, name,
    line, meant to masquerade as traceback.FrameSummary objects.
  """
  thread_key = _get_thread_key()
  return _tf_stack.extract_stack(
      _source_mapper_stacks[thread_key][-1].internal_map,
      _source_filter_stacks[thread_key][-1].internal_set,
      stacklevel,
  )


def LoadTracesFromDebugInfo(debug_info):
  # [FUNCTION DOC]
  # Input: a graph debug info protocol message. Serializes it and asks the native module to load
  # the stack traces it contains; returns whatever the native loader returns (a collection of
  # traces keyed by node). Raises whatever the native layer raises on malformed data.
  return _tf_stack.LoadTracesFromDebugInfo(debug_info.SerializeToString())


class GraphDebugInfoBuilder(_tf_stack.GraphDebugInfoBuilder):
  # [CLASS DOC]
  # Thin Python subclass of the native graph debug info builder that accepts and returns
  # protocol message objects instead of serialized strings. Accumulates debug info for several
  # functions and builds one combined message.

  def AppendGraphDebugInfo(self, fn_name, fn_debug_info):
    # [FUNCTION DOC]
    # Adds debug info for one function. Inputs: fn_name (string name of the function) and
    # fn_debug_info (graph debug info message). Serializes the message and passes it to the
    # native base implementation. Returns nothing; mutates the builder.
    debug_info_str = fn_debug_info.SerializeToString()
    super().AppendGraphDebugInfo(fn_name, debug_info_str)

  def Build(self):
    # [FUNCTION DOC]
    # Builds the combined debug info. Calls the native builder to get a serialized message,
    # parses it into a new graph debug info message and returns that message. Takes no inputs.
    debug_info_str = super().Build()
    debug_info = graph_debug_info_pb2.GraphDebugInfo()
    debug_info.ParseFromString(debug_info_str)
    return debug_info


StackSummary = _tf_stack.StackTrace
FrameSummary = _tf_stack.StackFrame
