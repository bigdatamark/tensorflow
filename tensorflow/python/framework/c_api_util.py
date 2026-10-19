# [MODULE DOC]
# HELPER: helper module wrapping native handles of the TensorFlow C API.
# Provides scoped wrapper classes that own C API objects (status, import options and results,
# functions, buffers, API definition maps) and release them when garbage collected, a
# single-ownership pointer wrapper with a safe access context manager, an error type for access
# after deletion, a context manager for temporary buffers, and a helper building output
# descriptors. No import-time side effects beyond importing the native session module.
# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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

"""Utilities for using the TensorFlow C API."""

import contextlib
from tensorflow.core.framework import api_def_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib


class AlreadyGarbageCollectedError(Exception):
  # [CLASS DOC]
  # Exception raised when code tries to use a wrapped native object after it has been released.

  def __init__(self, name, obj_type):
    # [FUNCTION DOC]
    # Constructor: inputs are the name of the wrapper and the type name of the object; builds the
    # message saying that the named object of that type was already garbage collected and cannot
    # be called, and passes it to the base exception.
    super(AlreadyGarbageCollectedError,
          self).__init__(f"{name} of type {obj_type} has already been garbage "
                         f"collected and cannot be called.")


# FIXME(b/235488206): Convert all Scoped objects to the context manager
# to protect against deletion during use when the object is attached to
# an attribute.
class UniquePtr(object):
  # [CLASS DOC]
  # Single-owner wrapper around a native object plus a deleter callback. The object is deleted
  # exactly once when the wrapper is collected. Access goes through a context manager that keeps
  # the wrapper (and thus the object) alive and raises if already deleted. Slots: the object, the
  # deleter, a name and the object's type name. Base class of the scoped function wrapper.
  """Wrapper around single-ownership C-API objects that handles deletion."""

  __slots__ = ["_obj", "deleter", "name", "type_name"]

  def __init__(self, name, obj, deleter):
    # [FUNCTION DOC]
    # Constructor. Inputs: name (string used in error messages), obj (the native object to own),
    # and deleter (callback taking the object and freeing it; captured now so it stays available
    # at interpreter shutdown). Also records the string form of the object's type. Returns
    # nothing.
    # '_' prefix marks _obj private, but unclear if it is required also to
    # maintain a special CPython destruction order.
    self._obj = obj
    self.name = name
    # Note: when we're destructing the global context (i.e when the process is
    # terminating) we may have already deleted other modules. By capturing the
    # DeleteGraph function here, we retain the ability to cleanly destroy the
    # graph at shutdown, which satisfies leak checkers.
    self.deleter = deleter
    self.type_name = str(type(obj))

  @contextlib.contextmanager
  def get(self):
    # [FUNCTION DOC]
    # Context manager (generator) yielding the wrapped native object. Raises the
    # already-garbage-collected error (naming the wrapper and the object's type) if the object has
    # already been released. While the context is active the wrapper itself stays referenced, so
    # the object cannot be deleted by collection.
    """Yields the managed C-API Object, guaranteeing aliveness.

    This is a context manager. Inside the context the C-API object is
    guaranteed to be alive.

    Raises:
      AlreadyGarbageCollectedError: if the object is already deleted.
    """
    # Thread-safety: self.__del__ never runs during the call of this function
    # because there is a reference to self from the argument list.
    if self._obj is None:
      raise AlreadyGarbageCollectedError(self.name, self.type_name)
    yield self._obj

  def __del__(self):
    # [FUNCTION DOC]
    # Finalizer: if the object has not been released yet, clears the reference first and then
    # calls the stored deleter on it, so deletion happens at most once. Returns nothing. Also
    # callable explicitly (tests do so) to simulate deletion.
    obj = self._obj
    if obj is not None:
      self._obj = None
      self.deleter(obj)


class ScopedTFStatus(object):
  # [CLASS DOC]
  # Owns a native status object created at construction (used to receive error information from
  # native calls) and deletes it on collection. Exposes it through the status attribute.
  """Wrapper around TF_Status that handles deletion."""

  __slots__ = ["status"]

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor with no arguments: creates a new native status object and stores it in the
    # status attribute.
    self.status = c_api.TF_NewStatus()

  def __del__(self):
    # [FUNCTION DOC]
    # Finalizer: deletes the native status object, guarded against interpreter shutdown when the
    # native module may be gone. Returns nothing.
    # Note: when we're destructing the global context (i.e when the process is
    # terminating) we can have already deleted other modules.
    if c_api is not None and c_api.TF_DeleteStatus is not None:
      c_api.TF_DeleteStatus(self.status)


class ScopedTFImportGraphDefOptions(object):
  # [CLASS DOC]
  # Owns a native graph-import options object created at construction and deletes it on
  # collection. Exposes it through the options attribute.
  """Wrapper around TF_ImportGraphDefOptions that handles deletion."""

  __slots__ = ["options"]

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor with no arguments: creates native graph-import options and stores them in the
    # options attribute.
    self.options = c_api.TF_NewImportGraphDefOptions()

  def __del__(self):
    # [FUNCTION DOC]
    # Finalizer: deletes the native import options, guarded against interpreter shutdown when the
    # native module may be gone. Returns nothing.
    # Note: when we're destructing the global context (i.e when the process is
    # terminating) we can have already deleted other modules.
    if c_api is not None and c_api.TF_DeleteImportGraphDefOptions is not None:
      c_api.TF_DeleteImportGraphDefOptions(self.options)


class ScopedTFImportGraphDefResults(object):
  # [CLASS DOC]
  # Owns a native graph-import results object passed at construction and deletes it on collection.
  """Wrapper around TF_ImportGraphDefResults that handles deletion."""

  __slots__ = ["results"]

  def __init__(self, results):
    # [FUNCTION DOC]
    # Constructor: stores the given native import results object, taking ownership of it.
    self.results = results

  def __del__(self):
    # [FUNCTION DOC]
    # Finalizer: deletes the native import results, guarded against interpreter shutdown when the
    # native module may be gone. Returns nothing.
    # Note: when we're destructing the global context (i.e when the process is
    # terminating) we can have already deleted other modules.
    if c_api is not None and c_api.TF_DeleteImportGraphDefResults is not None:
      c_api.TF_DeleteImportGraphDefResults(self.results)


class ScopedTFFunction(UniquePtr):
  # [CLASS DOC]
  # Single-ownership wrapper specialised for native function objects; deletion uses the native
  # function delete routine. Access it through the inherited context manager.
  """Wrapper around TF_Function that handles deletion."""

  def __init__(self, func, name):
    # [FUNCTION DOC]
    # Constructor: inputs are the native function object and a name used in error messages.
    # Delegates to the single-ownership pointer constructor with the native function deleter.
    super(ScopedTFFunction, self).__init__(
        name=name, obj=func, deleter=c_api.TF_DeleteFunction)


class ScopedTFBuffer(object):
  # [CLASS DOC]
  # HELPER: lifetime manager for a native buffer created from a string. Holds the buffer in a
  # single slot and deletes it on collection.
  """An internal class to help manage the TF_Buffer lifetime."""

  __slots__ = ["buffer"]

  def __init__(self, buf_string):
    # [FUNCTION DOC]
    # Constructor: input buf_string is text or bytes (text converted to bytes); creates a native
    # buffer holding a copy and stores it in the buffer attribute.
    self.buffer = c_api.TF_NewBufferFromString(compat.as_bytes(buf_string))

  def __del__(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: unlike the sibling wrappers, this finalizer does not guard against the native
    # module being already torn down at interpreter shutdown, so it may fail then.
    # Finalizer: deletes the native buffer held by this wrapper. Returns nothing.
    c_api.TF_DeleteBuffer(self.buffer)


class ApiDefMap(object):
  # [CLASS DOC]
  # Owns a native API definition map together with a cached name-to-op-definition mapping for all
  # registered ops. Created with no arguments; supports adding definitions from text, querying
  # API and op definitions by op name, and listing op names. Releases the native map when
  # collected. Uses fixed slots for its two attributes.
  """Wrapper around TF_ApiDefMap that handles querying and deletion.

  The OpDef protos are also stored in this class so that they could
  be queried by op name.
  """

  __slots__ = ["_api_def_map", "_op_per_name"]

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor with no arguments. Fetches the list of all registered op definitions from the
    # native layer, parses it into a message, creates the native API definition map from the same
    # buffer (the buffer is always freed afterwards), and builds a mapping from op name to op
    # definition. Native errors propagate.
    op_def_proto = op_def_pb2.OpList()
    buf = c_api.TF_GetAllOpList()
    try:
      op_def_proto.ParseFromString(c_api.TF_GetBuffer(buf))
      self._api_def_map = c_api.TF_NewApiDefMap(buf)
    finally:
      c_api.TF_DeleteBuffer(buf)

    self._op_per_name = {}
    for op in op_def_proto.op:
      self._op_per_name[op.name] = op

  def __del__(self):
    # [FUNCTION DOC]
    # Finalizer: releases the native API definition map, guarded against the native module or
    # delete function having been torn down at interpreter shutdown. Returns nothing.
    # Note: when we're destructing the global context (i.e when the process is
    # terminating) we can have already deleted other modules.
    if c_api is not None and c_api.TF_DeleteApiDefMap is not None:
      c_api.TF_DeleteApiDefMap(self._api_def_map)

  def put_api_def(self, text):
    # [FUNCTION DOC]
    # Adds or overrides API definitions in the native map from text-format content. Input text is
    # the serialized API definition text; its Python length is passed along. Returns nothing.
    # Side effect: mutates the native map, affecting later get_api_def calls.
    c_api.TF_ApiDefMapPut(self._api_def_map, text, len(text))

  def get_api_def(self, op_name):
    # [FUNCTION DOC]
    # Fetches the API definition for an op name from the native map, parsing the returned buffer
    # into a new API definition message which is returned. The temporary native buffer is always
    # freed. Errors from the native layer propagate (for unknown ops). The length passed is that
    # of the Python string.
    api_def_proto = api_def_pb2.ApiDef()
    buf = c_api.TF_ApiDefMapGet(self._api_def_map, op_name, len(op_name))
    try:
      api_def_proto.ParseFromString(c_api.TF_GetBuffer(buf))
    finally:
      c_api.TF_DeleteBuffer(buf)
    return api_def_proto

  def get_op_def(self, op_name):
    # [FUNCTION DOC]
    # Looks up the op definition message for an op name (string) from the cached list loaded at
    # construction. Returns it; raises a value error if the name is unknown. No side effects.
    if op_name in self._op_per_name:
      return self._op_per_name[op_name]
    raise ValueError(f"No op_def found for op name {op_name}.")

  def op_names(self):
    # [FUNCTION DOC]
    # Returns a view of the names of all ops known to this map (the keys of the internal mapping
    # from op name to op definition). No side effects.
    return self._op_per_name.keys()


@tf_contextlib.contextmanager
def tf_buffer(data=None):
  # [FUNCTION DOC]
  # Context manager (generator) yielding a native buffer. Input data is optional bytes or text;
  # when provided and non-empty the buffer is created holding a copy of it (text converted to
  # bytes), otherwise (including empty data) an empty buffer is created. The buffer is always
  # deleted when the context exits, even if the body raises.
  """Context manager that creates and deletes TF_Buffer.

  Example usage:
    with tf_buffer() as buf:
      # get serialized graph def into buf
      ...
      proto_data = c_api.TF_GetBuffer(buf)
      graph_def.ParseFromString(compat.as_bytes(proto_data))
    # buf has been deleted

    with tf_buffer(some_string) as buf:
      c_api.TF_SomeFunction(buf)
    # buf has been deleted

  Args:
    data: An optional `bytes`, `str`, or `unicode` object. If not None, the
      yielded buffer will contain this data.

  Yields:
    Created TF_Buffer
  """
  if data:
    buf = c_api.TF_NewBufferFromString(compat.as_bytes(data))
  else:
    buf = c_api.TF_NewBuffer()
  try:
    yield buf
  finally:
    c_api.TF_DeleteBuffer(buf)


def tf_output(c_op, index):
  # [FUNCTION DOC]
  # Builds a native output descriptor naming one output of an operation. Inputs: the wrapped
  # native operation and an integer output index (not range checked). Returns the new native
  # output object with its operation and index fields set.
  """Returns a wrapped TF_Output with specified operation and index.

  Args:
    c_op: wrapped TF_Operation
    index: integer

  Returns:
    Wrapped TF_Output
  """
  ret = c_api.TF_Output()
  ret.oper = c_op
  ret.index = index
  return ret
