# [MODULE DOC]
# Session tensor handle operations (v1 graph/session API, experimental). Lets a tensor produced in
# one session run stay alive inside the session runtime and be fetched, fed or deleted later
# by a string handle. Exposes the TensorHandle class and the exported get_session_handle,
# get_session_tensor and delete_session_tensor functions, plus private helpers that lazily
# build and cache reader, mover and deleter subgraphs in per-graph dictionaries. No import-time
# side effects.
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

"""Tensor Handle Operations."""

# pylint: disable=g-bad-name

from tensorflow.core.framework import resource_handle_pb2
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.util import compat
from tensorflow.python.util import numpy_compat
from tensorflow.python.util.tf_export import tf_export


def encode_resource_handle(resource_handle):
  # [FUNCTION DOC]
  # Serializes a resource handle protocol buffer message and returns it as a numpy array of the
  # custom resource struct dtype built from the serialized bytes, suitable as a feed value.
  # No side effects.
  """Encode a ResourceHandle proto as custom numpy struct type."""
  return numpy_compat.np_asarray(
      bytearray(resource_handle.SerializeToString()), dtype=dtypes.np_resource
  )


class TensorHandle:
  # [CLASS DOC]
  # Client side handle for a tensor that lives in a session's state. Holds the handle string
  # (name, unique id and device separated by semicolons), a lazily built resource handle message,
  # the dtype, the owning session and an automatic garbage collection flag. Created by the
  # session when a handle tensor is fetched. When collected while auto garbage collection is
  # on, it tells the session to delete the stored tensor. Supports reading the value, explicit
  # deletion, conversion to a feedable numpy value, and releasing control of its lifetime.
  """Represents a handle for a live tensor in a session."""

  def __init__(self, handle, dtype, session):
    # [FUNCTION DOC]
    # Constructor. Inputs: the handle (string or bytes, converted to text), the dtype of the
    # tensor, and the session that produced it. Initializes the resource handle cache to
    # nothing and enables automatic garbage collection. Returns nothing.
    """Constructs a new tensor handle.

    A tensor handle for a persistent tensor is a python string
    that has the form of "tensor_name;unique_id;device_name".

    Args:
      handle: A tensor handle.
      dtype: The data type of the tensor represented by `handle`.
      session: The session in which the tensor is produced.
    """
    self._handle = compat.as_str_any(handle)
    self._resource_handle = None
    self._dtype = dtype
    self._session = session
    self._auto_gc_enabled = True

  def __del__(self):
    # [FUNCTION DOC]
    # Finalizer. If automatic garbage collection is still enabled, registers this handle with the
    # session as dead so the stored tensor is freed on a later run. Otherwise does nothing.
    # KNOWN ISSUE: if the constructor failed before setting its attributes, this would raise
    # during finalization (an attribute error that Python only reports and ignores).
    if self._auto_gc_enabled:
      self._session._register_dead_handle(self.handle)

  def __str__(self):
    # [FUNCTION DOC]
    # Returns the handle string.
    return self._handle

  def _get_resource_handle(self):
    # [FUNCTION DOC]
    # HELPER: lazily builds and caches the resource handle message for this handle. On first use
    # sets its device to the last semicolon separated part of the handle, its container to the
    # session tensor handle key, and its name to the whole handle string. Returns the cached
    # message (mutating internal state on first call).
    """The ResourceHandle representation of this handle."""
    if not self._resource_handle:
      self._resource_handle = resource_handle_pb2.ResourceHandleProto()
      self._resource_handle.device = self._handle.split(";")[-1]
      self._resource_handle.container = (pywrap_tf_session.TENSOR_HANDLE_KEY)
      self._resource_handle.name = self._handle
    return self._resource_handle

  def to_numpy_array(self):
    # [FUNCTION DOC]
    # Returns the numpy struct array encoding this handle's resource handle message, usable as a
    # feed value in a session run call. Builds and caches the message if needed.
    """Convert a TensorHandle object to a feedable numpy value.

    Returns:
      A numpy array of a custom struct type that can be used as a feed value
      to run().
    """
    return encode_resource_handle(self._get_resource_handle())

  @property
  def handle(self):
    # [FUNCTION DOC]
    # Read-only property returning the handle string.
    """The string representation of this handle."""
    return self._handle

  def eval(self):
    # [FUNCTION DOC]
    # Fetches the stored tensor's value. Raises a type error if the handle was already deleted or
    # released. Gets (or creates and caches) a reader subgraph in the session graph, then runs it
    # in the session, feeding the handle string. Returns the fetched value. Side effect: may add
    # ops to the graph.
    """Return the value of the tensor represented by this handle."""
    if not self._auto_gc_enabled:
      raise TypeError("Persistent tensor %s may have already been deleted."
                      % self.handle)
    holder, reader = _get_handle_reader(self._session.graph, self._handle,
                                        self._dtype)
    return self._session.run(reader, feed_dict={holder: self._handle})

  def delete(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the deleter subgraph is always requested with the constant key zero, so the
    # cache returns the first deleter ever built in that graph, which was created for the first
    # handle's device; deleting a handle on another device would use the wrong device placement.
    # Also, automatic collection is disabled before the run, so a failing run leaks the tensor.
    # Forces deletion of the stored tensor. Raises a type error if already deleted or released.
    # Disables automatic collection, builds or reuses a deleter subgraph, and runs it feeding this
    # handle. Returns nothing.
    """Force the deletion of this persistent tensor."""
    if not self._auto_gc_enabled:
      raise TypeError("Persistent tensor %s may have already been deleted."
                      % self.handle)
    self._auto_gc_enabled = False
    holder, deleter = _get_handle_deleter(self._session.graph, 0, self._handle)
    self._session.run(deleter, feed_dict={holder: self.handle})

  def get_raw_handle(self):
    # [FUNCTION DOC]
    # Returns the handle string and, as a side effect, permanently disables automatic garbage
    # collection for this object so the caller must delete the stored tensor itself.
    """Return the raw handle of the tensor.

    Note that the method disables the automatic garbage collection of this
    persistent tensor. The caller is now responsible for managing the life
    time of the tensor.
    """
    self._auto_gc_enabled = False
    return self._handle

  @staticmethod
  def _get_device_name(handle):
    # [FUNCTION DOC]
    # HELPER: static; extracts the device part (after the last semicolon) of a handle string or
    # bytes and returns its canonical device name.
    """The device name encoded in the handle."""
    handle_str = compat.as_str_any(handle)
    return pydev.canonical_name(handle_str.split(";")[-1])

  @staticmethod
  def _get_reader_key(handle):
    # [FUNCTION DOC]
    # HELPER: static; builds the reader cache key for a handle, made of its first and last
    # semicolon separated parts (tensor name and device), joined by a semicolon.
    """The graph key for reader."""
    handle_parts = str(handle).split(";")
    return handle_parts[0] + ";" + handle_parts[-1]

  @staticmethod
  def _get_mover_key(feeder, handle):
    # [FUNCTION DOC]
    # HELPER: static; builds the mover cache key from the feeder tensor's op name, a semicolon
    # and the reader key of the handle.
    """The graph key for mover."""
    return feeder.op.name + ";" + TensorHandle._get_reader_key(handle)


@tf_export(v1=["get_session_handle"])
def get_session_handle(data, name=None):
  # [FUNCTION DOC]
  # Exported v1 experimental op builder. Inputs: a tensor to keep in the session and an optional
  # name prefix. Raises a type error if data is not a tensor. Adds a get-session-handle op
  # colocated with the data and returns a scalar string tensor holding the handle.
  """Return the handle of `data`.

  This is EXPERIMENTAL and subject to change.

  Keep `data` "in-place" in the runtime and create a handle that can be
  used to retrieve `data` in a subsequent run().

  Combined with `get_session_tensor`, we can keep a tensor produced in
  one run call in place, and use it as the input in a future run call.

  Args:
    data: A tensor to be stored in the session.
    name: Optional name prefix for the return tensor.

  Returns:
    A scalar string tensor representing a unique handle for `data`.

  Raises:
    TypeError: if `data` is not a Tensor.

  Example:

  ```python
  c = tf.multiply(a, b)
  h = tf.compat.v1.get_session_handle(c)
  h = sess.run(h)

  p, a = tf.compat.v1.get_session_tensor(h.handle, tf.float32)
  b = tf.multiply(a, 10)
  c = sess.run(b, feed_dict={p: h.handle})
  ```

  """
  if not isinstance(data, tensor_lib.Tensor):
    raise TypeError("`data` must be of type Tensor.")

  # Colocate this operation with data.
  with ops.colocate_with(data):
    return gen_data_flow_ops.get_session_handle(data, name=name)


@tf_export(v1=["get_session_tensor"])
def get_session_tensor(handle, dtype, name=None):
  # [FUNCTION DOC]
  # Exported v1 experimental op builder. Inputs: a handle string (used only to pick the device),
  # the output dtype and an optional name prefix. On the handle's device, creates a string
  # placeholder, registers it as a handle feeder with the dtype in the graph, and adds a
  # get-session-tensor op reading it. Returns a pair of the placeholder and the resulting tensor.
  """Get the tensor of type `dtype` by feeding a tensor handle.

  This is EXPERIMENTAL and subject to change.

  Get the value of the tensor from a tensor handle. The tensor
  is produced in a previous run() and stored in the state of the
  session.

  Args:
    handle: The string representation of a persistent tensor handle.
    dtype: The type of the output tensor.
    name: Optional name prefix for the return tensor.

  Returns:
    A pair of tensors. The first is a placeholder for feeding a
    tensor handle and the second is the tensor in the session state
    keyed by the tensor handle.

  Example:

  ```python
  c = tf.multiply(a, b)
  h = tf.compat.v1.get_session_handle(c)
  h = sess.run(h)

  p, a = tf.compat.v1.get_session_tensor(h.handle, tf.float32)
  b = tf.multiply(a, 10)
  c = sess.run(b, feed_dict={p: h.handle})
  ```

  """
  handle_device = TensorHandle._get_device_name(handle)
  with ops.device(handle_device):
    holder = array_ops.placeholder(dtypes.string)
    _register_handle_feeder(holder.graph, holder, dtype)
    tensor = gen_data_flow_ops.get_session_tensor(holder, dtype, name=name)
  return (holder, tensor)


@tf_export(v1=["delete_session_tensor"])
def delete_session_tensor(handle, name=None):
  # [FUNCTION DOC]
  # Exported v1 experimental op builder. Inputs: a handle string (used for device placement) and
  # an optional name prefix. On the handle's device creates a string placeholder and a delete op
  # consuming it. Returns a pair of the placeholder to feed and the deletion operation.
  """Delete the tensor for the given tensor handle.

  This is EXPERIMENTAL and subject to change.

  Delete the tensor of a given tensor handle. The tensor is produced
  in a previous run() and stored in the state of the session.

  Args:
    handle: The string representation of a persistent tensor handle.
    name: Optional name prefix for the return tensor.

  Returns:
    A pair of graph elements. The first is a placeholder for feeding a
    tensor handle and the second is a deletion operation.
  """
  handle_device = TensorHandle._get_device_name(handle)
  with ops.device(handle_device):
    holder = array_ops.placeholder(dtypes.string)
    deleter = gen_data_flow_ops.delete_session_tensor(holder, name=name)
  return (holder, deleter)


def _register_handle_feeder(graph, feeder, dtype):
  # [FUNCTION DOC]
  # HELPER: records in the graph's feeder mapping that the placeholder with the given op name
  # feeds tensors of the given dtype. Mutates the graph; returns nothing.
  graph._handle_feeders[feeder.op.name] = dtype


def _get_handle_feeder(graph, feeder):
  # [FUNCTION DOC]
  # HELPER: looks up the dtype registered for a feeder placeholder; returns it or nothing if the
  # feeder is not a registered handle feeder.
  return graph._handle_feeders.get(feeder.op.name)


def _get_handle_reader(graph, handle, dtype):
  # [FUNCTION DOC]
  # HELPER: returns the cached (placeholder, read tensor) pair for a handle's reader key in the
  # graph, creating it on the handle's device (and registering the placeholder as a feeder)
  # and caching it in the graph's reader mapping if absent.
  """Return a read subgraph for this handle."""
  graph_key = TensorHandle._get_reader_key(handle)
  result = graph._handle_readers.get(graph_key)
  if result is None:
    # Create reader if we haven't done it.
    handle_device = TensorHandle._get_device_name(handle)
    with graph.as_default(), graph.device(handle_device):
      holder = array_ops.placeholder(dtypes.string)
      _register_handle_feeder(holder.graph, holder, dtype)
      reader = gen_data_flow_ops.get_session_tensor(holder, dtype)
    result = (holder, reader)
    graph._handle_readers[graph_key] = result
  return result


def _get_handle_mover(graph, feeder, handle):
  # [FUNCTION DOC]
  # HELPER: returns the (placeholder, mover) subgraph that moves a stored tensor to the feeder's
  # device. Returns nothing if the feeder is not a registered handle feeder or already is on the
  # handle's device. Otherwise reuses or builds, on the feeder's device, a get-handle op over the
  # handle's reader, and caches it under the mover key in the graph.
  """Return a move subgraph for this pair of feeder and handle."""
  dtype = _get_handle_feeder(graph, feeder)
  if dtype is None:
    return None
  handle_device = TensorHandle._get_device_name(handle)
  if feeder.op.device == handle_device:
    return None
  # Now we know we have to move the tensor.
  graph_key = TensorHandle._get_mover_key(feeder, handle)
  result = graph._handle_movers.get(graph_key)
  if result is None:
    # Create mover if we haven't done it.
    holder, reader = _get_handle_reader(graph, handle, dtype)
    with graph.as_default(), graph.device(feeder.op.device):
      mover = gen_data_flow_ops.get_session_handle(reader)
    result = (holder, mover)
    graph._handle_movers[graph_key] = result
  return result


def _get_handle_deleter(graph, deleter_key, handle):
  # [FUNCTION DOC]
  # HELPER: returns the cached (placeholder, delete op) pair for the given deleter key in the
  # graph, building it on the handle's device and caching it if absent. Note the cache is keyed
  # only by the passed key, not by device (the handle's delete method passes a constant key).
  """Return a deletion subgraph for this handle."""
  result = graph._handle_deleters.get(deleter_key)
  if result is None:
    # Create deleter if we haven't done it.
    handle_device = TensorHandle._get_device_name(handle)
    with graph.as_default(), graph.device(handle_device):
      holder = array_ops.placeholder(dtypes.string)
      deleter = gen_data_flow_ops.delete_session_tensor(holder)
    result = (holder, deleter)
    graph._handle_deleters[deleter_key] = result
  return result
