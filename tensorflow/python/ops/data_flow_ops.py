# [MODULE DOC]
# Data flow operations for TensorFlow: the Python queue classes (QueueBase and its concrete
# subclasses for first-in-first-out, GPU-compatible FIFO, random shuffle, priority and padding
# queues), Barrier, the dense and sparse conditional gradient accumulators, the staging areas
# (StagingArea and MapStagingArea, optionally ordered) and RecordInput, plus small private
# helpers that normalise dtype, shape and name arguments.
# Each class wraps a generated op from the gen_data_flow_ops module and adds argument validation,
# static shape propagation and graph/eager handling. Public classes are exported to the public
# API through the tf_export decorator, with some deprecated v1 aliases.
# Import-time side effects: the wildcard import re-exports every generated data flow op from this
# module, and the decorators register the exported symbols and deprecated endpoints.
# This is a core library module (not a pure helper module).
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
"""Data Flow Operations."""
# pylint: disable=g-bad-name
import functools
import hashlib
import threading

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.lib.io import python_io
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import resource_variable_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.gen_data_flow_ops import *
from tensorflow.python.util import deprecation
from tensorflow.python.util.compat import collections_abc
from tensorflow.python.util.tf_export import tf_export

# pylint: enable=wildcard-import


def _as_type_list(dtypes):
  # [FUNCTION DOC]
  # HELPER: normalises dtype arguments for the queue and container constructors in this module.
  # Takes a single dtype or a list/tuple of dtypes (must not be null; asserts otherwise) and
  # returns a new list: a single dtype is wrapped in a one-element list, a list or tuple is
  # copied into a list. No side effects.
  """Convert dtypes to a list of types."""
  assert dtypes is not None
  if not (isinstance(dtypes, list) or isinstance(dtypes, tuple)):
    # We have a single type.
    return [dtypes]
  else:
    # We have a list or tuple of types.
    return list(dtypes)


def _as_shape_list(shapes,
                   dtypes,
                   unknown_dim_allowed=False,
                   unknown_rank_allowed=False):
  # [FUNCTION DOC]
  # HELPER: normalises shape arguments for the queue and container constructors in this module.
  # KNOWN ISSUE: the dtypes parameter is accepted but immediately deleted and never used.
  # KNOWN ISSUE: the docstring says it returns a list of tuples of ints, but it actually returns
  # a list of TensorShape objects (or null when shapes is null).
  # Inputs: shapes is null, a single TensorShape, a single shape written as a sequence of
  # integers/nulls, or a list/tuple of shapes; unknown_dim_allowed (default false) permits
  # partially defined shapes; unknown_rank_allowed (default false) permits shapes of unknown rank.
  # Returns null for null input, otherwise a list of TensorShapes.
  # Raises ValueError if partial shapes are allowed but the input is not a non-empty sequence of
  # shapes (a bare list of integers/nulls), if a shape is not fully defined when unknown
  # dimensions are disallowed, or if a shape has unknown rank when that is disallowed; raises
  # TypeError if shapes is not a TensorShape, list or tuple. No side effects.
  """Convert shapes to a list of tuples of int (or None)."""
  del dtypes
  if unknown_dim_allowed:
    if (not isinstance(shapes, collections_abc.Sequence) or not shapes or
        any(shape is None or isinstance(shape, int) for shape in shapes)):
      raise ValueError(
          "When providing partial shapes, a list of shapes must be provided.")
  if shapes is None:
    return None
  if isinstance(shapes, tensor_shape.TensorShape):
    shapes = [shapes]
  if not isinstance(shapes, (tuple, list)):
    raise TypeError(
        "Shapes must be a TensorShape or a list or tuple of TensorShapes, "
        f"got {type(shapes)} instead.")
  if all(shape is None or isinstance(shape, int) for shape in shapes):
    # We have a single shape.
    shapes = [shapes]
  shapes = [tensor_shape.as_shape(shape) for shape in shapes]
  if not unknown_dim_allowed:
    if any(not shape.is_fully_defined() for shape in shapes):
      raise ValueError(f"All shapes must be fully defined: {shapes}")
  if not unknown_rank_allowed:
    if any(shape.dims is None for shape in shapes):
      raise ValueError(f"All shapes must have a defined rank: {shapes}")

  return shapes


def _as_name_list(names, dtypes):
  # [FUNCTION DOC]
  # HELPER: normalises the optional component names for queue and container constructors.
  # Takes names (null, a single name, or a list/tuple of names) and the list of dtypes. Returns
  # null when names is null, otherwise a new list of names (a single name is wrapped).
  # Raises ValueError when the number of names differs from the number of dtypes.
  if names is None:
    return None
  if not isinstance(names, (list, tuple)):
    names = [names]
  if len(names) != len(dtypes):
    raise ValueError("List of names must have the same length as the list "
                     f"of dtypes, received len(names)={len(names)},"
                     f"len(dtypes)={len(dtypes)}")
  return list(names)


def _shape_common(s1, s2):
  # [FUNCTION DOC]
  # HELPER: used by QueueBase.from_list to merge the shapes of several queues.
  # Takes two shapes (anything convertible to a TensorShape) and returns the most specific shape
  # compatible with both: unknown shape if either rank is unknown or the ranks differ, otherwise
  # a shape keeping each dimension only where both agree and is known, with unknown elsewhere.
  # No side effects.
  """The greatest lower bound (ordered by specificity) TensorShape."""
  s1 = tensor_shape.TensorShape(s1)
  s2 = tensor_shape.TensorShape(s2)
  if s1.ndims is None or s2.ndims is None or s1.ndims != s2.ndims:
    return tensor_shape.unknown_shape()
  d = [
      d1 if d1 is not None and d1 == d2 else None
      for (d1, d2) in zip(s1.as_list(), s2.as_list())
  ]
  return tensor_shape.TensorShape(d)


# pylint: disable=protected-access
@tf_export("queue.QueueBase",
           v1=["queue.QueueBase", "io.QueueBase", "QueueBase"])
@deprecation.deprecated_endpoints(["io.QueueBase", "QueueBase"])
class QueueBase:
  # [CLASS DOC]
  # Common base class for all graph/eager queue wrappers. It holds the per-component dtypes,
  # shapes and optional names, the queue reference tensor (a resource handle in eager mode or a
  # resource/ref tensor in graph mode) and a display name. It provides enqueue, enqueue_many,
  # dequeue, dequeue_many, dequeue_up_to, close, is_closed and size, each of which builds the
  # matching generated op (the v2 resource variants when the reference is a resource) and sets
  # static output shapes in graph mode. Concrete subclasses create the queue op and call the
  # constructor with the resulting reference. The class is exported as the public queue base
  # with deprecated v1 aliases.
  """Base class for queue implementations.

  A queue is a TensorFlow data structure that stores tensors across
  multiple steps, and exposes operations that enqueue and dequeue
  tensors.

  Each queue element is a tuple of one or more tensors, where each
  tuple component has a static dtype, and may have a static shape. The
  queue implementations support versions of enqueue and dequeue that
  handle single elements, versions that support enqueuing and
  dequeuing a batch of elements at once.

  See `tf.queue.FIFOQueue` and
  `tf.queue.RandomShuffleQueue` for concrete
  implementations of this class, and instructions on how to create
  them.
  """

  def __init__(self, dtypes, shapes, names, queue_ref):
    # [FUNCTION DOC]
    # Stores queue metadata. dtypes: list of component types; shapes: null or list of shapes of
    # the same length (null gives unknown shapes for all components); names: null or list of the
    # same length; queue_ref: the queue handle tensor produced by the queue creation op.
    # Raises ValueError if shapes or names lengths differ from dtypes. In eager mode the name is
    # taken from the current scope name (or "Empty") and a resource deleter is attached so the
    # queue resource is freed with the object; in graph mode the name is the last component of
    # the queue op name.
    """Constructs a queue object from a queue reference.

    The two optional lists, `shapes` and `names`, must be of the same length
    as `dtypes` if provided.  The values at a given index `i` indicate the
    shape and name to use for the corresponding queue component in `dtypes`.

    Args:
      dtypes:  A list of types.  The length of dtypes must equal the number
        of tensors in each element.
      shapes: Constraints on the shapes of tensors in an element:
        A list of shape tuples or None. This list is the same length
        as dtypes.  If the shape of any tensors in the element are constrained,
        all must be; shapes can be None if the shapes should not be constrained.
      names: Optional list of names.  If provided, the `enqueue()` and
        `dequeue()` methods will use dictionaries with these names as keys.
        Must be None or a list or tuple of the same length as `dtypes`.
      queue_ref: The queue reference, i.e. the output of the queue op.

    Raises:
      ValueError: If one of the arguments is invalid.
    """
    self._dtypes = dtypes
    if shapes is not None:
      if len(shapes) != len(dtypes):
        raise ValueError("Queue shapes must have the same length as dtypes, "
                         f"received len(shapes)={len(shapes)}, "
                         f"len(dtypes)={len(dtypes)}")
      self._shapes = [tensor_shape.TensorShape(s) for s in shapes]
    else:
      self._shapes = [tensor_shape.unknown_shape() for _ in self._dtypes]
    if names is not None:
      if len(names) != len(dtypes):
        raise ValueError("Queue names must have the same length as dtypes,"
                         f"received len(names)={len(names)},"
                         f"len {len(dtypes)}")
      self._names = names
    else:
      self._names = None
    self._queue_ref = queue_ref
    if isinstance(queue_ref, ops.EagerTensor):
      if context.context().scope_name:
        self._name = context.context().scope_name
      else:
        self._name = "Empty"
      self._resource_deleter = resource_variable_ops.EagerResourceDeleter(
          queue_ref, None)
    else:
      self._name = self._queue_ref.op.name.split("/")[-1]

  @staticmethod
  def from_list(index, queues):
    # [FUNCTION DOC]
    # Static factory that builds a queue object whose reference is selected at run time from
    # several queues. index: integer scalar tensor choosing which queue; queues: non-empty list
    # of QueueBase objects with identical dtypes and names. The shapes are merged component-wise
    # to the most specific common shape. Stacks the queue references and gathers the chosen one.
    # Returns a new QueueBase. Raises TypeError if queues is empty, not a list, contains
    # non-queues, or dtypes or names differ. Adds stack/gather ops to the graph.
    """Create a queue using the queue reference from `queues[index]`.

    Args:
      index: An integer scalar tensor that determines the input that gets
        selected.
      queues: A list of `QueueBase` objects.

    Returns:
      A `QueueBase` object.

    Raises:
      TypeError: When `queues` is not a list of `QueueBase` objects,
        or when the data types of `queues` are not all the same.
    """
    if ((not queues) or (not isinstance(queues, list)) or
        (not all(isinstance(x, QueueBase) for x in queues))):
      raise TypeError("A list of queues expected")

    dtypes = queues[0].dtypes
    if not all(dtypes == q.dtypes for q in queues[1:]):
      raise TypeError("Queues do not have matching component dtypes.")

    names = queues[0].names
    if not all(names == q.names for q in queues[1:]):
      raise TypeError("Queues do not have matching component names.")

    queue_shapes = [q.shapes for q in queues]
    reduced_shapes = [
        functools.reduce(_shape_common, s) for s in zip(*queue_shapes)
    ]

    queue_refs = array_ops_stack.stack([x.queue_ref for x in queues])
    selected_queue = array_ops.gather(queue_refs, index)
    return QueueBase(
        dtypes=dtypes,
        shapes=reduced_shapes,
        names=names,
        queue_ref=selected_queue)

  @property
  def queue_ref(self):
    # [FUNCTION DOC]
    # Read-only property returning the queue handle tensor given at construction.
    """The underlying queue reference."""
    return self._queue_ref

  @property
  def name(self):
    # [FUNCTION DOC]
    # Read-only property returning the queue name: the cached name when executing eagerly,
    # otherwise the full name of the op that created the queue reference.
    """The name of the underlying queue."""
    if context.executing_eagerly():
      return self._name
    return self._queue_ref.op.name

  @property
  def dtypes(self):
    # [FUNCTION DOC]
    # Read-only property returning the list of component dtypes.
    """The list of dtypes for each component of a queue element."""
    return self._dtypes

  @property
  def shapes(self):
    # [FUNCTION DOC]
    # Read-only property returning the list of component TensorShapes (unknown shapes if none
    # were given).
    """The list of shapes for each component of a queue element."""
    return self._shapes

  @property
  def names(self):
    # [FUNCTION DOC]
    # Read-only property returning the list of component names, or null if unnamed.
    """The list of names for each component of a queue element."""
    return self._names

  def _check_enqueue_dtypes(self, vals):
    # [FUNCTION DOC]
    # HELPER: argument conversion shared by enqueue and enqueue_many.
    # Takes vals (a tensor, list/tuple of tensors, or a mapping from names to tensors). For a
    # mapping the queue must have names and the keys must match them; values are ordered by the
    # queue names. Unnamed queues reject mappings in reverse: a named queue requires a mapping.
    # Each value is converted to a tensor of the matching component dtype, named by position.
    # Returns a list of tensors. Raises ValueError for any of the mismatches above.
    # KNOWN ISSUE: the dtype loop uses zip, so surplus values beyond the number of dtypes are
    # silently dropped rather than reported.
    """Validate and convert `vals` to a list of `Tensor`s.

    The `vals` argument can be a Tensor, a list or tuple of tensors, or a
    dictionary with tensor values.

    If it is a dictionary, the queue must have been constructed with a
    `names` attribute and the dictionary keys must match the queue names.
    If the queue was constructed with a `names` attribute, `vals` must
    be a dictionary.

    Args:
      vals: A tensor, a list or tuple of tensors, or a dictionary..

    Returns:
      A list of `Tensor` objects.

    Raises:
      ValueError: If `vals` is invalid.
    """
    if isinstance(vals, dict):
      if not self._names:
        raise ValueError("Queue must have names to enqueue a dictionary")
      if sorted(self._names, key=str) != sorted(vals.keys(), key=str):
        raise ValueError("Keys in dictionary to enqueue do not match "
                         f"names of Queue.  Dictionary: {sorted(vals.keys())},"
                         f"Queue: {sorted(self._names)}")
      # The order of values in `self._names` indicates the order in which the
      # tensors in the dictionary `vals` must be listed.
      vals = [vals[k] for k in self._names]
    else:
      if self._names:
        raise ValueError("You must enqueue a dictionary in a Queue with names")
      if not isinstance(vals, (list, tuple)):
        vals = [vals]

    tensors = []
    for i, (val, dtype) in enumerate(zip(vals, self._dtypes)):
      tensors.append(
          ops.convert_to_tensor(val, dtype=dtype, name="component_%d" % i))

    return tensors

  def _scope_vals(self, vals):
    # [FUNCTION DOC]
    # HELPER: supplies the values list for the name scope of enqueue operations.
    # KNOWN ISSUE: the docstring says the result is a list, but a mapping input yields a
    # dictionary values view, and a list or tuple is returned unchanged (a tuple stays a tuple).
    # Returns the sequence itself for list/tuple, the values of a mapping, or a one-element list
    # wrapping a single tensor. No side effects.
    """Return a list of values to pass to `name_scope()`.

    Args:
      vals: A tensor, a list or tuple of tensors, or a dictionary.

    Returns:
      The values in vals as a list.
    """
    if isinstance(vals, (list, tuple)):
      return vals
    elif isinstance(vals, dict):
      return vals.values()
    else:
      return [vals]

  def enqueue(self, vals, name=None):
    # [FUNCTION DOC]
    # Adds (or in eager mode runs) an op that enqueues one element. vals: tensor, list/tuple or
    # mapping of component values (see the argument checker); name: optional op name.
    # Each value is converted to the component dtype and its static shape must be compatible with
    # the declared component shape (otherwise a shape-compatibility ValueError). Uses the v2
    # resource op when the queue reference is a resource, else the reference op. Returns the
    # enqueue operation. Blocks at run time when the queue is full; may raise a cancelled error
    # if the queue is closed.
    """Enqueues one element to this queue.

    If the queue is full when this operation executes, it will block
    until the element has been enqueued.

    At runtime, this operation may raise an error if the queue is
    `tf.QueueBase.close` before or during its execution. If the
    queue is closed before this operation runs,
    `tf.errors.CancelledError` will be raised. If this operation is
    blocked, and either (i) the queue is closed by a close operation
    with `cancel_pending_enqueues=True`, or (ii) the session is
    `tf.Session.close`,
    `tf.errors.CancelledError` will be raised.

    >>> q = tf.queue.FIFOQueue(capacity=3, dtypes=tf.int32)
    >>> q.enqueue(1)
    >>> q.enqueue(2)
    >>> q.size()
    <tf.Tensor: shape=(), dtype=int32, numpy=2>

    >>> q = tf.queue.FIFOQueue(2, tf.int32, shapes=tf.TensorShape(4))
    >>> q.enqueue(tf.constant([1, 2, 3, 4], dtype=tf.int32))
    >>> q.size()
    <tf.Tensor: shape=(), dtype=int32, numpy=1>

    Args:
      vals: A tensor, a list or tuple of tensors, or a dictionary containing
        the values to enqueue.
      name: A name for the operation (optional).

    Returns:
      The operation that enqueues a new tuple of tensors to the queue.
    """
    with ops.name_scope(name, "%s_enqueue" % self._name,
                        self._scope_vals(vals)) as scope:
      vals = self._check_enqueue_dtypes(vals)

      # NOTE(mrry): Not using a shape function because we need access to
      # the `QueueBase` object.
      for val, shape in zip(vals, self._shapes):
        val.get_shape().assert_is_compatible_with(shape)

      if self._queue_ref.dtype == _dtypes.resource:
        return gen_data_flow_ops.queue_enqueue_v2(
            self._queue_ref, vals, name=scope)
      else:
        return gen_data_flow_ops.queue_enqueue(
            self._queue_ref, vals, name=scope)

  def enqueue_many(self, vals, name=None):
    # [FUNCTION DOC]
    # Enqueues a batch of elements, slicing every component along dimension zero. vals: tensor,
    # list/tuple or mapping of batched component values; name: optional op name.
    # Statically merges the batch dimensions of all components (raises ValueError if they
    # conflict, or if a component has rank below one) and checks the remaining dimensions are
    # compatible with the declared shapes. Always uses the v2 resource op. Returns the enqueue
    # operation. Blocks when the queue is full; may raise a cancelled error if closed.
    """Enqueues zero or more elements to this queue.

    This operation slices each component tensor along the 0th dimension to
    make multiple queue elements. All of the tensors in `vals` must have the
    same size in the 0th dimension.

    If the queue is full when this operation executes, it will block
    until all of the elements have been enqueued.

    At runtime, this operation may raise an error if the queue is
    `tf.QueueBase.close` before or during its execution. If the
    queue is closed before this operation runs,
    `tf.errors.CancelledError` will be raised. If this operation is
    blocked, and either (i) the queue is closed by a close operation
    with `cancel_pending_enqueues=True`, or (ii) the session is
    `tf.Session.close`,
    `tf.errors.CancelledError` will be raised.

    >>> q = tf.queue.FIFOQueue(capacity=10, dtypes=tf.int32)
    >>> q.enqueue_many(tf.constant([1, 2, 3, 4, 5], dtype=tf.int32))
    >>> q.size()
    <tf.Tensor: shape=(), dtype=int32, numpy=5>

    Args:
      vals: A tensor, a list or tuple of tensors, or a dictionary
        from which the queue elements are taken.
      name: A name for the operation (optional).

    Returns:
      The operation that enqueues a batch of tuples of tensors to the queue.
    """
    with ops.name_scope(name, "%s_EnqueueMany" % self._name,
                        self._scope_vals(vals)) as scope:
      vals = self._check_enqueue_dtypes(vals)

      # NOTE(mrry): Not using a shape function because we need access to
      # the `QueueBase` object.
      # NOTE(fchollet): the code that follow is verbose because it needs to be
      # compatible with both TF v1 TensorShape behavior and TF v2 behavior.
      batch_dim = tensor_shape.dimension_value(
          vals[0].get_shape().with_rank_at_least(1)[0])
      batch_dim = tensor_shape.Dimension(batch_dim)
      for val, shape in zip(vals, self._shapes):
        val_batch_dim = tensor_shape.dimension_value(
            val.get_shape().with_rank_at_least(1)[0])
        val_batch_dim = tensor_shape.Dimension(val_batch_dim)
        batch_dim = batch_dim.merge_with(val_batch_dim)
        val.get_shape()[1:].assert_is_compatible_with(shape)

      return gen_data_flow_ops.queue_enqueue_many_v2(
          self._queue_ref, vals, name=scope)

  def _dequeue_return_value(self, tensors):
    # [FUNCTION DOC]
    # HELPER: shapes the result of the dequeue family of methods.
    # Takes the list of dequeued tensors and returns a mapping from component names to tensors if
    # the queue has names, a single tensor if there is exactly one component, else the list.
    """Return the value to return from a dequeue op.

    If the queue has names, return a dictionary with the
    names as keys.  Otherwise return either a single tensor
    or a list of tensors depending on the length of `tensors`.

    Args:
      tensors: List of tensors from the dequeue op.

    Returns:
      A single tensor, a list of tensors, or a dictionary
      of tensors.
    """
    if self._names:
      # The returned values in `tensors` are in the same order as
      # the names in `self._names`.
      return {n: tensors[i] for i, n in enumerate(self._names)}
    elif len(tensors) == 1:
      return tensors[0]
    else:
      return tensors

  def dequeue(self, name=None):
    # [FUNCTION DOC]
    # Dequeues one element. name: optional op name (default derived from the queue name).
    # Uses the v2 resource op when the reference is a resource. In graph mode each output's
    # static shape is set to the declared component shape. Returns a tensor, list of tensors or
    # mapping (see the return-value helper). Blocks when the queue is empty at run time; raises an
    # out-of-range error when closed and empty.
    """Dequeues one element from this queue.

    If the queue is empty when this operation executes, it will block
    until there is an element to dequeue.

    At runtime, this operation may raise an error if the queue is
    `tf.QueueBase.close` before or during its execution. If the
    queue is closed, the queue is empty, and there are no pending
    enqueue operations that can fulfill this request,
    `tf.errors.OutOfRangeError` will be raised. If the session is
    `tf.Session.close`,
    `tf.errors.CancelledError` will be raised.

    >>> q = tf.queue.FIFOQueue(capacity=2, dtypes=tf.int32)
    >>> q.enqueue(1)
    >>> q.enqueue(2)
    >>> q.dequeue()
    <tf.Tensor: shape=(), dtype=int32, numpy=1>
    >>> q.dequeue()
    <tf.Tensor: shape=(), dtype=int32, numpy=2>

    Args:
      name: A name for the operation (optional).

    Returns:
      The tuple of tensors that was dequeued.
    """
    if name is None:
      name = "%s_Dequeue" % self._name
    if self._queue_ref.dtype == _dtypes.resource:
      ret = gen_data_flow_ops.queue_dequeue_v2(
          self._queue_ref, self._dtypes, name=name)
    else:
      ret = gen_data_flow_ops.queue_dequeue(
          self._queue_ref, self._dtypes, name=name)

    # NOTE(mrry): Not using a shape function because we need access to
    # the `QueueBase` object.
    if not context.executing_eagerly():
      op = ret[0].op
      for output, shape in zip(op.values(), self._shapes):
        output.set_shape(shape)

    return self._dequeue_return_value(ret)

  def dequeue_many(self, n, name=None):
    # [FUNCTION DOC]
    # Dequeues n elements and concatenates each component along a new leading dimension.
    # n: scalar integer tensor (or constant); name: optional op name.
    # In graph mode, sets output static shapes to the batch size (if n is a constant, else
    # unknown) followed by the component shape. Returns a tensor, list or mapping like dequeue.
    # KNOWN ISSUE: the docstring says it returns a list, but it may return a single tensor or a
    # mapping depending on the queue. Raises out-of-range at run time if closed with fewer than n.
    """Dequeues and concatenates `n` elements from this queue.

    This operation concatenates queue-element component tensors along
    the 0th dimension to make a single component tensor.  All of the
    components in the dequeued tuple will have size `n` in the 0th dimension.

    If the queue is closed and there are less than `n` elements left, then an
    `OutOfRange` exception is raised.

    At runtime, this operation may raise an error if the queue is
    `tf.QueueBase.close` before or during its execution. If the
    queue is closed, the queue contains fewer than `n` elements, and
    there are no pending enqueue operations that can fulfill this
    request, `tf.errors.OutOfRangeError` will be raised. If the
    session is `tf.Session.close`,
    `tf.errors.CancelledError` will be raised.

    >>> q = tf.queue.FIFOQueue(10, tf.int32, shapes=tf.TensorShape(2))
    >>> q.enqueue(tf.constant([1, 2], dtype=tf.int32, shape=(2)))
    >>> q.enqueue(tf.constant([3, 4], dtype=tf.int32, shape=(2)))
    >>> q.enqueue(tf.constant([5, 6], dtype=tf.int32, shape=(2)))
    >>> q.enqueue(tf.constant([7, 8], dtype=tf.int32, shape=(2)))
    >>> q.dequeue_many(3)
    <tf.Tensor: shape=(3, 2), dtype=int32, numpy=
    array([[1, 2],
       [3, 4],
       [5, 6]], dtype=int32)>

    Args:
      n: A scalar `Tensor` containing the number of elements to dequeue.
      name: A name for the operation (optional).

    Returns:
      The list of concatenated tensors that was dequeued.
    """
    if name is None:
      name = "%s_DequeueMany" % self._name

    ret = gen_data_flow_ops.queue_dequeue_many_v2(
        self._queue_ref, n=n, component_types=self._dtypes, name=name)

    # NOTE(mrry): Not using a shape function because we need access to
    # the Queue object.
    if not context.executing_eagerly():
      op = ret[0].op
      batch_dim = tensor_shape.Dimension(
          tensor_util.constant_value(op.inputs[1]))
      for output, shape in zip(op.values(), self._shapes):
        output.set_shape(
            tensor_shape.TensorShape([batch_dim]).concatenate(shape))

    return self._dequeue_return_value(ret)

  def dequeue_up_to(self, n, name=None):
    # [FUNCTION DOC]
    # Like dequeue_many but returns fewer than n elements if the queue is closed with some left.
    # n: scalar integer tensor; name: optional op name. In graph mode output static shapes get an
    # unknown leading dimension followed by the component shape. Returns a tensor, list or
    # mapping. Not supported by every queue type (unimplemented error at run time); raises
    # out-of-range if closed and empty.
    """Dequeues and concatenates `n` elements from this queue.

    **Note** This operation is not supported by all queues.  If a queue does not
    support DequeueUpTo, then a `tf.errors.UnimplementedError` is raised.

    This operation concatenates queue-element component tensors along
    the 0th dimension to make a single component tensor. If the queue
    has not been closed, all of the components in the dequeued tuple
    will have size `n` in the 0th dimension.

    If the queue is closed and there are more than `0` but fewer than
    `n` elements remaining, then instead of raising a
    `tf.errors.OutOfRangeError` like `tf.QueueBase.dequeue_many`,
    less than `n` elements are returned immediately.  If the queue is
    closed and there are `0` elements left in the queue, then a
    `tf.errors.OutOfRangeError` is raised just like in `dequeue_many`.
    Otherwise the behavior is identical to `dequeue_many`.

    >>> q = tf.queue.FIFOQueue(10, tf.int32, shapes=tf.TensorShape(2))
    >>> q.enqueue(tf.constant([1, 2], dtype=tf.int32, shape=(2)))
    >>> q.enqueue(tf.constant([3, 4], dtype=tf.int32, shape=(2)))
    >>> q.close()
    >>> q.dequeue_up_to(5)
    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[1, 2],
       [3, 4]], dtype=int32)>

    Args:
      n: A scalar `Tensor` containing the number of elements to dequeue.
      name: A name for the operation (optional).

    Returns:
      The tuple of concatenated tensors that was dequeued.
    """
    if name is None:
      name = "%s_DequeueUpTo" % self._name

    ret = gen_data_flow_ops.queue_dequeue_up_to_v2(
        self._queue_ref, n=n, component_types=self._dtypes, name=name)

    # NOTE(mrry): Not using a shape function because we need access to
    # the Queue object.
    if not context.executing_eagerly():
      op = ret[0].op
      for output, shape in zip(op.values(), self._shapes):
        output.set_shape(tensor_shape.TensorShape([None]).concatenate(shape))

    return self._dequeue_return_value(ret)

  def close(self, cancel_pending_enqueues=False, name=None):
    # [FUNCTION DOC]
    # Closes the queue so no more elements can be enqueued. cancel_pending_enqueues: boolean,
    # default false; if true pending enqueue requests are cancelled. name: optional op name.
    # Uses the v2 resource op when the reference is a resource. Returns the close operation;
    # the queue is closed when it runs.
    """Closes this queue.

    This operation signals that no more elements will be enqueued in
    the given queue. Subsequent `enqueue` and `enqueue_many`
    operations will fail. Subsequent `dequeue` and `dequeue_many`
    operations will continue to succeed if sufficient elements remain
    in the queue. Subsequently, dequeue and dequeue_many operations
    that would otherwise block waiting for more elements (if close
    hadn't been called) will now fail immediately.

    If `cancel_pending_enqueues` is `True`, all pending requests will also
    be canceled.

    >>> q = tf.queue.FIFOQueue(capacity=3, dtypes=tf.int32)
    >>> q.is_closed()
    <tf.Tensor: shape=(), dtype=bool, numpy=False>
    >>> q.close()
    >>> q.is_closed()
    <tf.Tensor: shape=(), dtype=bool, numpy=True>

    Args:
      cancel_pending_enqueues: (Optional.) A boolean, defaulting to `False`
        (described above).
      name: A name for the operation (optional).

    Returns:
      The operation that closes the queue.
    """
    if name is None:
      name = "%s_Close" % self._name
    if self._queue_ref.dtype == _dtypes.resource:
      return gen_data_flow_ops.queue_close_v2(
          self._queue_ref,
          cancel_pending_enqueues=cancel_pending_enqueues,
          name=name)
    else:
      return gen_data_flow_ops.queue_close(
          self._queue_ref,
          cancel_pending_enqueues=cancel_pending_enqueues,
          name=name)

  def is_closed(self, name=None):
    # [FUNCTION DOC]
    # Builds an op reporting whether the queue is closed. name: optional op name. Uses the v2
    # resource op for resource references, else the reference op. Returns a scalar boolean
    # tensor.
    """Returns true if queue is closed.

    This operation returns true if the queue is closed and false if the queue
    is open.

    >>> q = tf.queue.FIFOQueue(capacity=3, dtypes=tf.int32)
    >>> q.is_closed()
    <tf.Tensor: shape=(), dtype=bool, numpy=False>

    Args:
      name: A name for the operation (optional).

    Returns:
      True if the queue is closed and false if the queue is open.
    """
    if name is None:
      name = "%s_Is_Closed" % self._name
    if self._queue_ref.dtype == _dtypes.resource:
      return gen_data_flow_ops.queue_is_closed_v2(self._queue_ref, name=name)
    else:
      return gen_data_flow_ops.queue_is_closed_(self._queue_ref, name=name)

  def size(self, name=None):
    # [FUNCTION DOC]
    # Builds an op returning the current number of elements. name: optional op name. Uses the v2
    # resource op for resource references. Returns a scalar int32 tensor.
    """Compute the number of elements in this queue.

    >>> q = tf.queue.FIFOQueue(capacity=10, dtypes=tf.int32)
    >>> q.enqueue_many(tf.constant([1, 2, 3, 4], dtype=tf.int32))
    >>> q.size()
    <tf.Tensor: shape=(), dtype=int32, numpy=4>

    Args:
      name: A name for the operation (optional).

    Returns:
      A scalar tensor containing the number of elements in this queue.
    """
    if name is None:
      name = "%s_Size" % self._name
    if self._queue_ref.dtype == _dtypes.resource:
      return gen_data_flow_ops.queue_size_v2(self._queue_ref, name=name)
    else:
      return gen_data_flow_ops.queue_size(self._queue_ref, name=name)

def _shared_name(shared_name):
  # [FUNCTION DOC]
  # HELPER: picks the resource sharing name for queues and containers.
  # When executing eagerly returns a fresh unique identifier string (so eager resources are never
  # shared accidentally, and the argument is ignored); otherwise returns shared_name unchanged
  # (possibly null).
  if context.executing_eagerly():
    return str(ops.uid())
  return shared_name


@tf_export(
    "queue.RandomShuffleQueue",
    v1=["queue.RandomShuffleQueue",
        "io.RandomShuffleQueue", "RandomShuffleQueue"])
@deprecation.deprecated_endpoints(
    ["io.RandomShuffleQueue", "RandomShuffleQueue"])
class RandomShuffleQueue(QueueBase):
  # [CLASS DOC]
  # Bounded queue that returns elements in random order. Creating an instance builds a random
  # shuffle queue resource op and registers its handle with the QueueBase constructor; all
  # enqueue/dequeue/close/size methods are inherited. Exported as a public queue class with
  # deprecated v1 aliases.
  """A queue implementation that dequeues elements in a random order.

  See `tf.queue.QueueBase` for a description of the methods on
  this class.
  """

  def __init__(self,
               capacity,
               min_after_dequeue,
               dtypes,
               shapes=None,
               names=None,
               seed=None,
               shared_name=None,
               name="random_shuffle_queue"):
    # [FUNCTION DOC]
    # Creates the random shuffle queue op. capacity: maximum number of elements; min_after_dequeue:
    # minimum number of elements left after a dequeue (for mixing; ignored once closed); dtypes:
    # component dtype or list; shapes: optional fully defined shapes; names: optional component
    # names; seed: optional integer for random seeding; shared_name: optional name for sharing
    # across sessions (ignored in eager mode, where a unique name is used); name: op name.
    # Seed handling: derives two seeds from the global/op seeds; if neither exists both are zero;
    # if only the graph seed exists and a shared name is given, the second seed is a hash of the
    # graph seed and shared name so that sharing workers agree. Raises ValueError/TypeError via
    # the shape and name helpers for invalid arguments. Side effect: adds a queue op to the graph
    # (or creates the eager resource).
    """Create a queue that dequeues elements in a random order.

    A `RandomShuffleQueue` has bounded capacity; supports multiple
    concurrent producers and consumers; and provides exactly-once
    delivery.

    A `RandomShuffleQueue` holds a list of up to `capacity`
    elements. Each element is a fixed-length tuple of tensors whose
    dtypes are described by `dtypes`, and whose shapes are optionally
    described by the `shapes` argument.

    If the `shapes` argument is specified, each component of a queue
    element must have the respective fixed shape. If it is
    unspecified, different queue elements may have different shapes,
    but the use of `dequeue_many` is disallowed.

    The `min_after_dequeue` argument allows the caller to specify a
    minimum number of elements that will remain in the queue after a
    `dequeue` or `dequeue_many` operation completes, to ensure a
    minimum level of mixing of elements. This invariant is maintained
    by blocking those operations until sufficient elements have been
    enqueued. The `min_after_dequeue` argument is ignored after the
    queue has been closed.

    Args:
      capacity: An integer. The upper bound on the number of elements
        that may be stored in this queue.
      min_after_dequeue: An integer (described above).
      dtypes:  A list of `DType` objects. The length of `dtypes` must equal
        the number of tensors in each queue element.
      shapes: (Optional.) A list of fully-defined `TensorShape` objects
        with the same length as `dtypes`, or `None`.
      names: (Optional.) A list of string naming the components in the queue
        with the same length as `dtypes`, or `None`.  If specified the dequeue
        methods return a dictionary with the names as keys.
      seed: A Python integer. Used to create a random seed. See
        `tf.compat.v1.set_random_seed`
        for behavior.
      shared_name: (Optional.) If non-empty, this queue will be shared under
        the given name across multiple sessions.
      name: Optional name for the queue operation.
    """
    dtypes = _as_type_list(dtypes)
    shapes = _as_shape_list(shapes, dtypes)
    names = _as_name_list(names, dtypes)
    seed1, seed2 = random_seed.get_seed(seed)
    if seed1 is None and seed2 is None:
      seed1, seed2 = 0, 0
    elif seed is None and shared_name is not None:
      # This means that graph seed is provided but op seed is not provided.
      # If shared_name is also provided, make seed2 depend only on the graph
      # seed and shared_name. (seed2 from get_seed() is generally dependent on
      # the id of the last op created.)
      string = (str(seed1) + shared_name).encode("utf-8")
      seed2 = int(hashlib.md5(string).hexdigest()[:8], 16) & 0x7FFFFFFF
    queue_ref = gen_data_flow_ops.random_shuffle_queue_v2(
        component_types=dtypes,
        shapes=shapes,
        capacity=capacity,
        min_after_dequeue=min_after_dequeue,
        seed=seed1,
        seed2=seed2,
        shared_name=_shared_name(shared_name),
        name=name)

    super(RandomShuffleQueue, self).__init__(dtypes, shapes, names, queue_ref)


@tf_export("queue.FIFOQueue", v1=["queue.FIFOQueue", "FIFOQueue"])
@deprecation.deprecated_endpoints("FIFOQueue")
class FIFOQueue(QueueBase):
  # [CLASS DOC]
  # Bounded first-in first-out queue. Instances create the FIFO queue resource on the CPU and
  # hand its handle to QueueBase, inheriting all queue methods. Exported publicly with
  # deprecated v1 aliases.
  """A queue implementation that dequeues elements in first-in first-out order.

  See `tf.queue.QueueBase` for a description of the methods on
  this class.
  """

  def __init__(self,
               capacity,
               dtypes,
               shapes=None,
               names=None,
               shared_name=None,
               name="fifo_queue"):
    # [FUNCTION DOC]
    # Creates the FIFO queue resource. capacity: maximum number of elements; dtypes: component
    # dtype or list; shapes: optional fully defined shapes; names: optional component names;
    # shared_name: optional sharing name (a unique name is used when eager); name: op name.
    # The op is created in the init scope on the CPU device. Argument errors surface as
    # ValueError/TypeError from the normalisation helpers.
    """Creates a queue that dequeues elements in a first-in first-out order.

    A `FIFOQueue` has bounded capacity; supports multiple concurrent
    producers and consumers; and provides exactly-once delivery.

    A `FIFOQueue` holds a list of up to `capacity` elements. Each
    element is a fixed-length tuple of tensors whose dtypes are
    described by `dtypes`, and whose shapes are optionally described
    by the `shapes` argument.

    If the `shapes` argument is specified, each component of a queue
    element must have the respective fixed shape. If it is
    unspecified, different queue elements may have different shapes,
    but the use of `dequeue_many` is disallowed.

    Args:
      capacity: An integer. The upper bound on the number of elements
        that may be stored in this queue.
      dtypes:  A list of `DType` objects. The length of `dtypes` must equal
        the number of tensors in each queue element.
      shapes: (Optional.) A list of fully-defined `TensorShape` objects
        with the same length as `dtypes`, or `None`.
      names: (Optional.) A list of string naming the components in the queue
        with the same length as `dtypes`, or `None`.  If specified the dequeue
        methods return a dictionary with the names as keys.
      shared_name: (Optional.) If non-empty, this queue will be shared under
        the given name across multiple sessions.
      name: Optional name for the queue operation.

    >>> q = tf.queue.FIFOQueue(capacity=10, dtypes=tf.int32)
    >>> q.size()
    <tf.Tensor: shape=(), dtype=int32, numpy=0>
    """
    dtypes = _as_type_list(dtypes)
    shapes = _as_shape_list(shapes, dtypes)
    names = _as_name_list(names, dtypes)
    with ops.init_scope(), ops.device("CPU"):
      queue_ref = gen_data_flow_ops.fifo_queue_v2(
          component_types=dtypes,
          shapes=shapes,
          capacity=capacity,
          shared_name=_shared_name(shared_name),
          name=name)

    super(FIFOQueue, self).__init__(dtypes, shapes, names, queue_ref)


# TODO(allenl): If GPU-compatible queues turn out to be useful, we should
# implement GPU kernels for EnqueueMany and DequeueMany so we can make the
# public FIFOQueue GPU-compatible and remove this internal version.
class GPUCompatibleFIFOQueue(QueueBase):
  # [CLASS DOC]
  # Internal (not exported) FIFO queue whose resource may live on CPU or GPU. Only single-element
  # enqueue and dequeue are supported; the batched variants are overridden to always fail.
  """A queue implementation that dequeues elements in first-in first-out order.

  GPUCompatibleFIFOQueue is like FIFOQueue, but the queue resource may be placed
  either on a CPU or on a GPU. It is not cross-device: enqueues and dequeues
  will be colocated with the queue resource. GPUCompatibleFIFOQueue only
  supports enqueue and dequeue at the moment, not enqueue_many or dequeue_many.

  See `tf.queue.QueueBase` for a description of the methods in this class.
  """

  def __init__(self,
               capacity,
               dtypes,
               shapes=None,
               names=None,
               shared_name=None,
               name="fifo_queue"):
    # [FUNCTION DOC]
    # Same arguments as the FIFO queue constructor (capacity, dtypes, optional shapes, names,
    # shared_name, name) but the resource is created in the init scope without forcing the CPU
    # device, so it can be placed on a GPU. Hands the handle to the QueueBase constructor.
    """Creates a queue that dequeues elements in a first-in first-out order.

    A `FIFOQueue` has bounded capacity; supports multiple concurrent
    producers and consumers; and provides exactly-once delivery.

    A `FIFOQueue` holds a list of up to `capacity` elements. Each
    element is a fixed-length tuple of tensors whose dtypes are
    described by `dtypes`, and whose shapes are optionally described
    by the `shapes` argument.

    If the `shapes` argument is specified, each component of a queue
    element must have the respective fixed shape. If it is
    unspecified, different queue elements may have different shapes,
    but the use of `dequeue_many` is disallowed.

    Args:
      capacity: An integer. The upper bound on the number of elements that may
        be stored in this queue.
      dtypes:  A list of `DType` objects. The length of `dtypes` must equal the
        number of tensors in each queue element.
      shapes: (Optional.) A list of fully-defined `TensorShape` objects with the
        same length as `dtypes`, or `None`.
      names: (Optional.) A list of strings naming the components in the queue
        with the same length as `dtypes`, or `None`.  If specified the dequeue
        methods return a dictionary with the names as keys.
      shared_name: (Optional.) If non-empty, this queue will be shared under the
        given name across multiple sessions.
      name: Optional name for the queue operation.
    """
    dtypes = _as_type_list(dtypes)
    shapes = _as_shape_list(shapes, dtypes)
    names = _as_name_list(names, dtypes)
    with ops.init_scope():
      queue_ref = gen_data_flow_ops.fifo_queue_v2(
          component_types=dtypes,
          shapes=shapes,
          capacity=capacity,
          shared_name=_shared_name(shared_name),
          name=name)

    super(GPUCompatibleFIFOQueue, self).__init__(
        dtypes, shapes, names, queue_ref)

  def enqueue_many(self, vals, name=None):
    # [FUNCTION DOC]
    # Override that always raises a not-implemented error; arguments are ignored.
    """enqueue_many is not supported on GPUCompatibleFIFOQueue."""
    raise NotImplementedError(
        "GPUCompatibleFIFOQueue does not support enqueue_many or dequeue_many, "
        "only enqueue and dequeue.")

  def dequeue_many(self, n, name=None):
    # [FUNCTION DOC]
    # Override that always raises a not-implemented error; arguments are ignored.
    """dequeue_many is not supported on GPUCompatibleFIFOQueue."""
    raise NotImplementedError(
        "GPUCompatibleFIFOQueue does not support enqueue_many or dequeue_many, "
        "only enqueue and dequeue.")


@tf_export(
    "queue.PaddingFIFOQueue",
    v1=["queue.PaddingFIFOQueue", "io.PaddingFIFOQueue", "PaddingFIFOQueue"])
@deprecation.deprecated_endpoints(["io.PaddingFIFOQueue", "PaddingFIFOQueue"])
class PaddingFIFOQueue(QueueBase):
  # [CLASS DOC]
  # FIFO queue whose component shapes may have unknown dimensions; dequeue_many pads those
  # dimensions with zeros up to the largest size in the batch. Inherits all queue methods.
  """A FIFOQueue that supports batching variable-sized tensors by padding.

  A `PaddingFIFOQueue` may contain components with dynamic shape, while also
  supporting `dequeue_many`.  See the constructor for more details.

  See `tf.queue.QueueBase` for a description of the methods on
  this class.
  """

  def __init__(self,
               capacity,
               dtypes,
               shapes,
               names=None,
               shared_name=None,
               name="padding_fifo_queue"):
    # [FUNCTION DOC]
    # Creates the padding FIFO queue resource. capacity: maximum number of elements; dtypes:
    # component dtype or list; shapes: required list of shapes (dimensions may be unknown);
    # names: optional component names; shared_name: optional sharing name; name: op name.
    # Raises ValueError if the number of shapes differs from the number of dtypes, or if shapes
    # are not a proper list of shapes (from the shape helper). Side effect: adds the queue op.
    """Creates a queue that dequeues elements in a first-in first-out order.

    A `PaddingFIFOQueue` has bounded capacity; supports multiple concurrent
    producers and consumers; and provides exactly-once delivery.

    A `PaddingFIFOQueue` holds a list of up to `capacity` elements. Each
    element is a fixed-length tuple of tensors whose dtypes are
    described by `dtypes`, and whose shapes are described by the `shapes`
    argument.

    The `shapes` argument must be specified; each component of a queue
    element must have the respective shape.  Shapes of fixed
    rank but variable size are allowed by setting any shape dimension to None.
    In this case, the inputs' shape may vary along the given dimension, and
    `dequeue_many` will pad the given dimension with zeros up to the maximum
    shape of all elements in the given batch.

    Args:
      capacity: An integer. The upper bound on the number of elements
        that may be stored in this queue.
      dtypes:  A list of `DType` objects. The length of `dtypes` must equal
        the number of tensors in each queue element.
      shapes: A list of `TensorShape` objects, with the same length as
        `dtypes`.  Any dimension in the `TensorShape` containing value
        `None` is dynamic and allows values to be enqueued with
         variable size in that dimension.
      names: (Optional.) A list of string naming the components in the queue
        with the same length as `dtypes`, or `None`.  If specified the dequeue
        methods return a dictionary with the names as keys.
      shared_name: (Optional.) If non-empty, this queue will be shared under
        the given name across multiple sessions.
      name: Optional name for the queue operation.

    Raises:
      ValueError: If shapes is not a list of shapes, or the lengths of dtypes
        and shapes do not match, or if names is specified and the lengths of
        dtypes and names do not match.
    """
    dtypes = _as_type_list(dtypes)
    shapes = _as_shape_list(shapes, dtypes, unknown_dim_allowed=True)
    names = _as_name_list(names, dtypes)
    if len(dtypes) != len(shapes):
      raise ValueError("Shapes must be provided for all components, "
                       f"but received {len(dtypes)} dtypes and "
                       f"{len(shapes)} shapes.")
    queue_ref = gen_data_flow_ops.padding_fifo_queue_v2(
        component_types=dtypes,
        shapes=shapes,
        capacity=capacity,
        shared_name=_shared_name(shared_name),
        name=name)

    super(PaddingFIFOQueue, self).__init__(dtypes, shapes, names, queue_ref)


@tf_export("queue.PriorityQueue",
           v1=["queue.PriorityQueue", "io.PriorityQueue", "PriorityQueue"])
@deprecation.deprecated_endpoints(["io.PriorityQueue", "PriorityQueue"])
class PriorityQueue(QueueBase):
  # [CLASS DOC]
  # Queue that dequeues elements in priority order (smallest int64 priority first). Elements carry
  # an extra leading int64 priority component which the constructor prepends to the dtypes and
  # shapes. Inherits all queue methods.
  """A queue implementation that dequeues elements in prioritized order.

  See `tf.queue.QueueBase` for a description of the methods on
  this class.
  """

  def __init__(self,
               capacity,
               types,
               shapes=None,
               names=None,
               shared_name=None,
               name="priority_queue"):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring headline says first-in first-out order, but the queue dequeues
    # by priority. Also, names is not passed through the name-list helper, so its length is not
    # validated against the types here; and since names are passed with the priority-prefixed
    # dtypes, a names list matching types alone has the wrong length for the base constructor.
    # Creates the priority queue resource. capacity: maximum elements; types: component dtype or
    # list excluding the priority; shapes: optional fully defined shapes excluding priority;
    # names: optional names; shared_name: optional sharing name; name: op name.
    # The base constructor receives int64 plus the types, and an empty-shape entry plus the
    # shapes (when given).
    """Creates a queue that dequeues elements in a first-in first-out order.

    A `PriorityQueue` has bounded capacity; supports multiple concurrent
    producers and consumers; and provides exactly-once delivery.

    A `PriorityQueue` holds a list of up to `capacity` elements. Each
    element is a fixed-length tuple of tensors whose dtypes are
    described by `types`, and whose shapes are optionally described
    by the `shapes` argument.

    If the `shapes` argument is specified, each component of a queue
    element must have the respective fixed shape. If it is
    unspecified, different queue elements may have different shapes,
    but the use of `dequeue_many` is disallowed.

    Enqueues and Dequeues to the `PriorityQueue` must include an additional
    tuple entry at the beginning: the `priority`.  The priority must be
    an int64 scalar (for `enqueue`) or an int64 vector (for `enqueue_many`).

    Args:
      capacity: An integer. The upper bound on the number of elements
        that may be stored in this queue.
      types:  A list of `DType` objects. The length of `types` must equal
        the number of tensors in each queue element, except the first priority
        element.  The first tensor in each element is the priority,
        which must be type int64.
      shapes: (Optional.) A list of fully-defined `TensorShape` objects,
        with the same length as `types`, or `None`.
      names: (Optional.) A list of strings naming the components in the queue
        with the same length as `dtypes`, or `None`.  If specified, the dequeue
        methods return a dictionary with the names as keys.
      shared_name: (Optional.) If non-empty, this queue will be shared under
        the given name across multiple sessions.
      name: Optional name for the queue operation.
    """
    types = _as_type_list(types)
    shapes = _as_shape_list(shapes, types)

    queue_ref = gen_data_flow_ops.priority_queue_v2(
        component_types=types,
        shapes=shapes,
        capacity=capacity,
        shared_name=_shared_name(shared_name),
        name=name)

    priority_dtypes = [_dtypes.int64] + types
    priority_shapes = [()] + shapes if shapes else shapes

    super(PriorityQueue, self).__init__(priority_dtypes, priority_shapes, names,
                                        queue_ref)


# TODO(josh11b): class BatchQueue(QueueBase):


class Barrier:
  # [CLASS DOC]
  # Wrapper around a barrier resource: a key-value map (string keys, tuples of tensors as values)
  # that persists across graph executions. Elements become complete when all components are
  # inserted. Holds the component types, shapes, barrier reference and a display name.
  # Provides insert_many, take_many, close, ready_size and incomplete_size.
  """Represents a key-value map that persists across graph executions."""

  def __init__(self, types, shapes=None, shared_name=None, name="barrier"):
    # [FUNCTION DOC]
    # Creates the barrier op. types: dtype or list of dtypes of the value components; shapes:
    # optional shape constraints (unknown shapes if omitted); shared_name: optional sharing name;
    # name: op name. Raises ValueError if any given shape has zero elements. Stores the types,
    # shapes, barrier reference and a name (scope name in eager mode, else the op's last name
    # component). Side effect: adds the barrier op.
    """Creates a barrier that persists across different graph executions.

    A barrier represents a key-value map, where each key is a string, and
    each value is a tuple of tensors.

    At runtime, the barrier contains 'complete' and 'incomplete'
    elements. A complete element has defined tensors for all
    components of its value tuple, and may be accessed using
    take_many. An incomplete element has some undefined components in
    its value tuple, and may be updated using insert_many.

    The barrier call `take_many` outputs values in a particular order.
    First, it only outputs completed values.  Second, the order in which
    completed values are returned matches the order in which their very
    first component was inserted into the barrier.  So, for example, for this
    sequence of insertions and removals:

      barrier = Barrier((tf.string, tf.int32), shapes=((), ()))
      barrier.insert_many(0, keys=["k1", "k2"], values=["a", "b"]).run()
      barrier.insert_many(1, keys=["k1"], values=[1]).run()
      barrier.insert_many(0, keys=["k3"], values=["c"]).run()
      barrier.insert_many(1, keys=["k3"], values=[3]).run()
      barrier.insert_many(1, keys=["k2"], values=[2]).run()

      (indices, keys, values) = barrier.take_many(2)
      (indices_val, keys_val, values0_val, values1_val) =
         session.run([indices, keys, values[0], values[1]])

    The output will be (up to permutation of "k1" and "k2"):

      indices_val == (-2**63, -2**63)
      keys_val == ("k1", "k2")
      values0_val == ("a", "b")
      values1_val == (1, 2)

    Note the key "k2" was inserted into the barrier before "k3".  Even though
    "k3" was completed first, both are complete by the time
    take_many is called.  As a result, "k2" is prioritized and "k1" and "k2"
    are returned first.  "k3" remains in the barrier until the next execution
    of `take_many`.  Since "k1" and "k2" had their first insertions into
    the barrier together, their indices are the same (-2**63).  The index
    of "k3" will be -2**63 + 1, because it was the next new inserted key.

    Args:
      types: A single dtype or a tuple of dtypes, corresponding to the
        dtypes of the tensor elements that comprise a value in this barrier.
      shapes: Optional. Constraints on the shapes of tensors in the values:
        a single tensor shape tuple; a tuple of tensor shape tuples
        for each barrier-element tuple component; or None if the shape should
        not be constrained.
      shared_name: Optional. If non-empty, this barrier will be shared under
        the given name across multiple sessions.
      name: Optional name for the barrier op.

    Raises:
      ValueError: If one of the `shapes` indicate no elements.
    """
    self._types = _as_type_list(types)

    if shapes is not None:
      shapes = _as_shape_list(shapes, self._types)
      self._shapes = [tensor_shape.TensorShape(s) for s in shapes]
      for i, shape in enumerate(self._shapes):
        if shape.num_elements() == 0:
          raise ValueError("Empty tensors are not supported, but received "
                           f"shape '{shape}' at index {i}")
    else:
      self._shapes = [tensor_shape.unknown_shape() for _ in self._types]

    self._barrier_ref = gen_data_flow_ops.barrier(
        component_types=self._types,
        shapes=self._shapes,
        shared_name=shared_name,
        name=name)
    if context.executing_eagerly():
      self._name = context.context().scope_name
    else:
      self._name = self._barrier_ref.op.name.split("/")[-1]

  @property
  def barrier_ref(self):
    # [FUNCTION DOC]
    # Read-only property returning the underlying barrier handle.
    """Get the underlying barrier reference."""
    return self._barrier_ref

  @property
  def name(self):
    # [FUNCTION DOC]
    # Read-only property: cached name when eager, otherwise the full op name of the barrier.
    """The name of the underlying barrier."""
    if context.executing_eagerly():
      return self._name
    return self._barrier_ref.op.name

  def insert_many(self, component_index, keys, values, name=None):
    # [FUNCTION DOC]
    # Builds an op that assigns values to one component of the elements identified by keys.
    # component_index: which component of the value tuple; keys: vector of n string keys;
    # values: tensor whose first dimension is n; name: optional op name. Returns the insert
    # operation. At run time raises an invalid-argument error for keys/values without elements.
    """For each key, assigns the respective value to the specified component.

    This operation updates each element at component_index.

    Args:
      component_index: The component of the value that is being assigned.
      keys: A vector of keys, with length n.
      values: An any-dimensional tensor of values, which are associated with the
        respective keys. The first dimension must have length n.
      name: Optional name for the op.

    Returns:
      The operation that performs the insertion.
    Raises:
      InvalidArgumentsError: If inserting keys and values without elements.
    """
    if name is None:
      name = "%s_BarrierInsertMany" % self._name
    return gen_data_flow_ops.barrier_insert_many(
        self._barrier_ref, keys, values, component_index, name=name)

  def take_many(self,
                num_elements,
                allow_small_batch=False,
                timeout=None,
                name=None):
    # [FUNCTION DOC]
    # Builds an op that removes completed elements. num_elements: how many to take;
    # allow_small_batch: if true and the barrier is closed, return what is available;
    # timeout: milliseconds to wait (documented as unsupported); name: optional op name.
    # Returns the op outputs: indices, keys and the list of value tensors. In graph mode sets
    # static shapes: batch dimension is the constant num_elements, or unknown when small batches
    # are allowed, followed by each component's shape.
    """Takes the given number of completed elements from this barrier.

    This operation concatenates completed-element component tensors along
    the 0th dimension to make a single component tensor.

    If barrier has no completed elements, this operation will block
    until there are 'num_elements' elements to take.

    TODO(b/25743580): the semantics of `allow_small_batch` are experimental
    and may be extended to other cases in the future.

    TODO(ebrevdo): If a take_many(allow_small_batch=True) is blocking
    already when the barrier is closed, it will block for ever. Fix this
    by using asynchronous operations.

    Args:
      num_elements: The number of elements to take.
      allow_small_batch: If the barrier is closed, don't block if there are less
        completed elements than requested, but instead return all available
        completed elements.
      timeout: This specifies the number of milliseconds to block
        before returning with DEADLINE_EXCEEDED. (This option is not
        supported yet.)
      name: A name for the operation (optional).

    Returns:
      A tuple of (index, key, value_list).
      "index" is a int64 tensor of length num_elements containing the
        index of the insert_many call for which the very first component of
        the given element was inserted into the Barrier, starting with
        the value -2**63.  Note, this value is different from the
        index of the insert_many call for which the element was completed.
      "key" is a string tensor of length num_elements containing the keys.
      "value_list" is a tuple of tensors, each one with size num_elements
        in the 0th dimension for each component in the barrier's values.

    """
    if name is None:
      name = "%s_BarrierTakeMany" % self._name
    ret = gen_data_flow_ops.barrier_take_many(
        self._barrier_ref,
        num_elements,
        self._types,
        allow_small_batch,
        timeout,
        name=name)

    # NOTE(mrry): Not using a shape function because we need access to
    # the Barrier object.
    if not context.executing_eagerly():
      op = ret[0].op
      if allow_small_batch:
        batch_dim = None
      else:
        batch_dim = tensor_shape.Dimension(
            tensor_util.constant_value(op.inputs[1]))
      op.outputs[0].set_shape(tensor_shape.TensorShape([batch_dim]))  # indices
      op.outputs[1].set_shape(tensor_shape.TensorShape([batch_dim]))  # keys
      for output, shape in zip(op.outputs[2:], self._shapes):  # value_list
        output.set_shape(
            tensor_shape.TensorShape([batch_dim]).concatenate(shape))

    return ret

  def close(self, cancel_pending_enqueues=False, name=None):
    # [FUNCTION DOC]
    # Builds an op closing the barrier: new keys can no longer be inserted. cancel_pending_enqueues:
    # boolean default false, cancels pending requests and further completion; name: optional.
    # Returns the close operation.
    """Closes this barrier.

    This operation signals that no more new key values will be inserted in the
    given barrier. Subsequent InsertMany operations with new keys will fail.
    InsertMany operations that just complement already existing keys with other
    components, will continue to succeed. Subsequent TakeMany operations will
    continue to succeed if sufficient elements remain in the barrier. Subsequent
    TakeMany operations that would block will fail immediately.

    If `cancel_pending_enqueues` is `True`, all pending requests to the
    underlying queue will also be canceled, and completing of already
    started values is also not acceptable anymore.

    Args:
      cancel_pending_enqueues: (Optional.) A boolean, defaulting to
        `False` (described above).
      name: Optional name for the op.

    Returns:
      The operation that closes the barrier.
    """
    if name is None:
      name = "%s_BarrierClose" % self._name
    return gen_data_flow_ops.barrier_close(
        self._barrier_ref,
        cancel_pending_enqueues=cancel_pending_enqueues,
        name=name)

  def ready_size(self, name=None):
    # [FUNCTION DOC]
    # Builds an op returning the number of complete elements. name: optional op name. Returns a
    # scalar int32 tensor.
    """Compute the number of complete elements in the given barrier.

    Args:
      name: A name for the operation (optional).

    Returns:
      A single-element tensor containing the number of complete elements in the
      given barrier.
    """
    if name is None:
      name = "%s_BarrierReadySize" % self._name
    return gen_data_flow_ops.barrier_ready_size(self._barrier_ref, name=name)

  def incomplete_size(self, name=None):
    # [FUNCTION DOC]
    # Builds an op returning the number of incomplete elements. name: optional op name. Returns a
    # scalar int32 tensor.
    """Compute the number of incomplete elements in the given barrier.

    Args:
      name: A name for the operation (optional).

    Returns:
      A single-element tensor containing the number of incomplete elements in
      the given barrier.
    """
    if name is None:
      name = "%s_BarrierIncompleteSize" % self._name
    return gen_data_flow_ops.barrier_incomplete_size(
        self._barrier_ref, name=name)


@tf_export(v1=["ConditionalAccumulatorBase"])
class ConditionalAccumulatorBase:
  # [CLASS DOC]
  # Base for accumulators that aggregate gradients only when they are up to date with the
  # accumulator's time step. Holds dtype, shape, accumulator handle and name; subclasses create
  # the handle. Provides num_accumulated and set_global_step (resource kernels).
  """A conditional accumulator for aggregating gradients.

  Up-to-date gradients (i.e., time step at which gradient was computed is
  equal to the accumulator's time step) are added to the accumulator.

  Extraction of the average gradient is blocked until the required number of
  gradients has been accumulated.
  """

  def __init__(self, dtype, shape, accumulator_ref):
    # [FUNCTION DOC]
    # Stores dtype, shape (unknown if null) and accumulator handle created by a subclass.
    # Name is the eager scope name, or the last component of the handle op name in graph mode.
    """Creates a new ConditionalAccumulator.

    Args:
      dtype: Datatype of the accumulated gradients.
      shape: Shape of the accumulated gradients.
      accumulator_ref: A handle to the conditional accumulator, created by sub-
        classes
    """
    self._dtype = dtype
    if shape is not None:
      self._shape = tensor_shape.TensorShape(shape)
    else:
      self._shape = tensor_shape.unknown_shape()
    self._accumulator_ref = accumulator_ref
    if context.executing_eagerly():
      self._name = context.context().scope_name
    else:
      self._name = self._accumulator_ref.op.name.split("/")[-1]

  @property
  def accumulator_ref(self):
    # [FUNCTION DOC]
    # Read-only property returning the accumulator handle.
    """The underlying accumulator reference."""
    return self._accumulator_ref

  @property
  def name(self):
    # [FUNCTION DOC]
    # Read-only property returning the cached accumulator name.
    """The name of the underlying accumulator."""
    return self._name

  @property
  def dtype(self):
    # [FUNCTION DOC]
    # Read-only property returning the dtype of accumulated gradients.
    """The datatype of the gradients accumulated by this accumulator."""
    return self._dtype

  def num_accumulated(self, name=None):
    # [FUNCTION DOC]
    # Builds an op returning how many gradients are currently aggregated (resource-based kernel).
    # name: optional op name, default derived from the accumulator name. Returns an int32 scalar
    # tensor.
    """Number of gradients that have currently been aggregated in accumulator.

    Args:
      name: Optional name for the operation.

    Returns:
      Number of accumulated gradients currently in accumulator.
    """
    if name is None:
      name = "%s_NumAccumulated" % self._name

    return gen_data_flow_ops.resource_accumulator_num_accumulated(
        self._accumulator_ref, name=name)

  def set_global_step(self, new_global_step, name=None):
    # [FUNCTION DOC]
    # Builds an op that sets the accumulator's time step (resource-based kernel). new_global_step:
    # variable or constant, converted to a tensor and cast to 64-bit integer; name: optional.
    # Returns the op. The kernel logs a warning if the new step is lower than the current one.
    """Sets the global time step of the accumulator.

    The operation logs a warning if we attempt to set to a time step that is
    lower than the accumulator's own time step.

    Args:
      new_global_step: Value of new time step. Can be a variable or a constant
      name: Optional name for the operation.

    Returns:
      Operation that sets the accumulator's time step.
    """
    return gen_data_flow_ops.resource_accumulator_set_global_step(
        self._accumulator_ref,
        math_ops.cast(ops.convert_to_tensor(new_global_step), _dtypes.int64),
        name=name)


@tf_export(v1=["ConditionalAccumulator"])
class ConditionalAccumulator(ConditionalAccumulatorBase):
  # [CLASS DOC]
  # Dense gradient accumulator backed by a resource conditional-accumulator op. Gradients whose
  # local step is older than the accumulator's step are dropped; take_grad blocks until enough
  # gradients are aggregated and then returns their average (or sum, per reduction type).
  # Exported for the v1 API only.
  """A conditional accumulator for aggregating gradients.

  Up-to-date gradients (i.e., time step at which gradient was computed is
  equal to the accumulator's time step) are added to the accumulator.

  Extraction of the average gradient is blocked until the required number of
  gradients has been accumulated.
  """

  def __init__(self,
               dtype,
               shape=None,
               shared_name=None,
               name="conditional_accumulator",
               reduction_type="MEAN"):
    # [FUNCTION DOC]
    # Creates the accumulator resource. dtype: dtype of gradients; shape: optional gradient shape
    # (unknown if null); shared_name: optional sharing name; name: op name; reduction_type: how
    # gradients are combined when taken, default mean. In eager mode attaches a resource deleter
    # to free the resource with the object. Then initialises the base class with the handle.
    """Creates a new ConditionalAccumulator.

    Args:
      dtype: Datatype of the accumulated gradients.
      shape: Shape of the accumulated gradients.
      shared_name: Optional. If non-empty, this accumulator will be shared under
        the given name across multiple sessions.
      name: Optional name for the accumulator.
      reduction_type: Reduction type to use when taking the gradient.
    """
    accumulator_ref = gen_data_flow_ops.resource_conditional_accumulator(
        dtype=dtype,
        shape=shape,
        shared_name=shared_name,
        name=name,
        reduction_type=reduction_type)
    if context.executing_eagerly():
      self._resource_deleter = resource_variable_ops.EagerResourceDeleter(
          handle=accumulator_ref, handle_device=context.context().device_name)

    super(ConditionalAccumulator, self).__init__(dtype, shape, accumulator_ref)

  def apply_grad(self, grad, local_step=0, name=None):
    # [FUNCTION DOC]
    # Builds an op that adds a dense gradient. grad: gradient convertible to a tensor of the
    # accumulator dtype; local_step: step at which it was computed (default 0, cast to 64-bit
    # integer); name: optional. Raises ValueError if the gradient shape is incompatible with the
    # accumulator shape. The op silently drops stale gradients. Returns the op.
    """Attempts to apply a gradient to the accumulator.

    The attempt is silently dropped if the gradient is stale, i.e., local_step
    is less than the accumulator's global time step.

    Args:
      grad: The gradient tensor to be applied.
      local_step: Time step at which the gradient was computed.
      name: Optional name for the operation.

    Returns:
      The operation that (conditionally) applies a gradient to the accumulator.

    Raises:
      ValueError: If grad is of the wrong shape
    """
    grad = ops.convert_to_tensor(grad, self._dtype)
    grad.get_shape().assert_is_compatible_with(self._shape)
    local_step = math_ops.cast(ops.convert_to_tensor(local_step), _dtypes.int64)

    return gen_data_flow_ops.resource_accumulator_apply_gradient(
        self._accumulator_ref, local_step=local_step, gradient=grad, name=name)

  def take_grad(self, num_required, name=None):
    # [FUNCTION DOC]
    # Builds an op that extracts the aggregated gradient once num_required gradients have been
    # applied (blocks until then), resets the counter and sum, and increments the accumulator's
    # time step. name: optional. Returns a tensor with the accumulator dtype whose static shape is
    # set to the accumulator shape. Run-time invalid-argument error if num_required is below one.
    """Attempts to extract the average gradient from the accumulator.

    The operation blocks until sufficient number of gradients have been
    successfully applied to the accumulator.

    Once successful, the following actions are also triggered:

    - Counter of accumulated gradients is reset to 0.
    - Aggregated gradient is reset to 0 tensor.
    - Accumulator's internal time step is incremented by 1.

    Args:
      num_required: Number of gradients that needs to have been aggregated
      name: Optional name for the operation

    Returns:
      A tensor holding the value of the average gradient.

    Raises:
      InvalidArgumentError: If num_required < 1
    """
    out = gen_data_flow_ops.resource_accumulator_take_gradient(
        self._accumulator_ref, num_required, dtype=self._dtype, name=name)
    out.set_shape(self._shape)
    return out


@tf_export(
    v1=["sparse.SparseConditionalAccumulator", "SparseConditionalAccumulator"])
class SparseConditionalAccumulator(ConditionalAccumulatorBase):
  # [CLASS DOC]
  # Accumulator for sparse gradients (indices, values, optional dense shape; or IndexedSlices).
  # Same stale-gradient dropping and blocking take semantics as the dense accumulator but uses the
  # older non-resource kernels, so no eager resource deleter is attached.
  # Exported for the v1 API only.
  """A conditional accumulator for aggregating sparse gradients.

  Sparse gradients are represented by `IndexedSlices`.

  Up-to-date gradients (i.e., time step at which gradient was computed is
  equal to the accumulator's time step) are added to the accumulator.

  Extraction of the average gradient is blocked until the required number of
  gradients has been accumulated.

  Args:
    dtype: Datatype of the accumulated gradients.
    shape: Shape of the accumulated gradients.
    shared_name: Optional. If non-empty, this accumulator will be shared under
      the given name across multiple sessions.
    name: Optional name for the accumulator.
    reduction_type: Reduction type to use when taking the gradient.
  """

  def __init__(self,
               dtype,
               shape=None,
               shared_name=None,
               name="sparse_conditional_accumulator",
               reduction_type="MEAN"):
    # [FUNCTION DOC]
    # Creates the sparse accumulator op and initialises the base class. dtype: gradient dtype;
    # shape: optional dense shape; shared_name: optional sharing name; name: op name;
    # reduction_type: how gradients are combined, default mean. Adds an op to the graph.
    accumulator_ref = gen_data_flow_ops.sparse_conditional_accumulator(
        dtype=dtype,
        shape=shape,
        shared_name=shared_name,
        name=name,
        reduction_type=reduction_type)
    super(SparseConditionalAccumulator, self).__init__(dtype, shape,
                                                       accumulator_ref)

  def apply_indexed_slices_grad(self, grad, local_step=0, name=None):
    # [FUNCTION DOC]
    # Convenience wrapper that applies an IndexedSlices gradient by splitting it into indices,
    # values and dense shape and delegating to apply_grad. local_step defaults to 0; name optional.
    # Returns the apply operation.
    """Attempts to apply a gradient to the accumulator.

    The attempt is silently dropped if the gradient is stale, i.e., `local_step`
    is less than the accumulator's global time step.

    Args:
      grad: The gradient `IndexedSlices` to be applied.
      local_step: Time step at which the gradient was computed.
      name: Optional name for the operation.

    Returns:
      The operation that (conditionally) applies a gradient to the accumulator.

    Raises:
      InvalidArgumentError: If grad is of the wrong shape
    """
    return self.apply_grad(
        grad_indices=grad.indices,
        grad_values=grad.values,
        grad_shape=grad.dense_shape,
        local_step=local_step,
        name=name)

  def apply_grad(self,
                 grad_indices,
                 grad_values,
                 grad_shape=None,
                 local_step=0,
                 name=None):
    # [FUNCTION DOC]
    # Builds an op applying a sparse gradient. grad_indices: vector of locations of nonzero slices
    # (cast to 64-bit integer); grad_values: the slices, same leading size as indices; grad_shape:
    # optional dense shape (null means unknown shape, passed as empty with a known-shape flag of
    # false); local_step: step of the gradient (default 0); name optional. Stale gradients are
    # silently dropped; invalid-argument error at run time for inconsistent shapes. Returns the op.
    """Attempts to apply a sparse gradient to the accumulator.

    The attempt is silently dropped if the gradient is stale, i.e., `local_step`
    is less than the accumulator's global time step.

    A sparse gradient is represented by its indices, values and possibly empty
    or None shape. Indices must be a vector representing the locations of
    non-zero entries in the tensor. Values are the non-zero slices of the
    gradient, and must have the same first dimension as indices, i.e., the nnz
    represented by indices and values must be consistent. Shape, if not empty or
    None, must be consistent with the accumulator's shape (if also provided).

    Example:
      A tensor [[0, 0], [0, 1], [2, 3]] can be represented
        indices: [1,2]
        values: [[0,1],[2,3]]
        shape: [3, 2]

    Args:
      grad_indices: Indices of the sparse gradient to be applied.
      grad_values: Values of the sparse gradient to be applied.
      grad_shape: Shape of the sparse gradient to be applied.
      local_step: Time step at which the gradient was computed.
      name: Optional name for the operation.

    Returns:
      The operation that (conditionally) applies a gradient to the accumulator.

    Raises:
      InvalidArgumentError: If grad is of the wrong shape
    """
    local_step = math_ops.cast(ops.convert_to_tensor(local_step), _dtypes.int64)
    return gen_data_flow_ops.sparse_accumulator_apply_gradient(
        self._accumulator_ref,
        local_step=local_step,
        gradient_indices=math_ops.cast(grad_indices, _dtypes.int64),
        gradient_values=grad_values,
        gradient_shape=math_ops.cast(
            [] if grad_shape is None else grad_shape, _dtypes.int64),
        has_known_shape=(grad_shape is not None),
        name=name)

  def take_grad(self, num_required, name=None):
    # [FUNCTION DOC]
    # Builds an op that blocks until num_required sparse gradients have been applied, then
    # returns the aggregate and resets the accumulator, incrementing its time step.
    # Returns the raw op outputs: indices, values and dense shape. name: optional. Run-time
    # invalid-argument error if num_required is below one.
    """Attempts to extract the average gradient from the accumulator.

    The operation blocks until sufficient number of gradients have been
    successfully applied to the accumulator.

    Once successful, the following actions are also triggered:
    - Counter of accumulated gradients is reset to 0.
    - Aggregated gradient is reset to 0 tensor.
    - Accumulator's internal time step is incremented by 1.

    Args:
      num_required: Number of gradients that needs to have been aggregated
      name: Optional name for the operation

    Returns:
      A tuple of indices, values, and shape representing the average gradient.

    Raises:
      InvalidArgumentError: If `num_required` < 1
    """
    return gen_data_flow_ops.sparse_accumulator_take_gradient(
        self._accumulator_ref, num_required, dtype=self._dtype, name=name)

  def take_indexed_slices_grad(self, num_required, name=None):
    # [FUNCTION DOC]
    # Same as take_grad but wraps the result in an IndexedSlices (indices, values, dense shape).
    # num_required: gradients needed; name optional. Run-time error if num_required is below one.
    """Attempts to extract the average gradient from the accumulator.

    The operation blocks until sufficient number of gradients have been
    successfully applied to the accumulator.

    Once successful, the following actions are also triggered:
    - Counter of accumulated gradients is reset to 0.
    - Aggregated gradient is reset to 0 tensor.
    - Accumulator's internal time step is incremented by 1.

    Args:
      num_required: Number of gradients that needs to have been aggregated
      name: Optional name for the operation

    Returns:
      An `IndexedSlices` holding the value of the average gradient.

    Raises:
      InvalidArgumentError: If `num_required` < 1
    """
    return_val = gen_data_flow_ops.sparse_accumulator_take_gradient(
        self._accumulator_ref, num_required, dtype=self._dtype, name=name)
    return indexed_slices.IndexedSlices(
        indices=return_val.indices,
        values=return_val.values,
        dense_shape=return_val.shape)

  # SparseConditionalAccumulator is not switched to resource. Use old kernels.
  def num_accumulated(self, name=None):
    # [FUNCTION DOC]
    # Override using the older non-resource kernel: builds an op returning the number of
    # gradients currently aggregated. name optional. Returns an int32 scalar tensor.
    """Number of gradients that have currently been aggregated in accumulator.

    Args:
      name: Optional name for the operation.

    Returns:
      Number of accumulated gradients currently in accumulator.
    """
    if name is None:
      name = "%s_NumAccumulated" % self._name

    return gen_data_flow_ops.accumulator_num_accumulated(
        self._accumulator_ref, name=name)

  def set_global_step(self, new_global_step, name=None):
    # [FUNCTION DOC]
    # Override using the older non-resource kernel: builds an op setting the accumulator's time
    # step to new_global_step (variable or constant, cast to 64-bit integer). name optional.
    # Returns the op; the kernel warns if the new step is lower than the current one.
    """Sets the global time step of the accumulator.

    The operation logs a warning if we attempt to set to a time step that is
    lower than the accumulator's own time step.

    Args:
      new_global_step: Value of new time step. Can be a variable or a constant
      name: Optional name for the operation.

    Returns:
      Operation that sets the accumulator's time step.
    """
    return gen_data_flow_ops.accumulator_set_global_step(
        self._accumulator_ref,
        math_ops.cast(ops.convert_to_tensor(new_global_step), _dtypes.int64),
        name=name)


class BaseStagingArea:
  # [CLASS DOC]
  # Shared implementation of staging areas (buffers of tensor tuples carried across steps, often
  # used to overlap host-to-device copies). Holds dtypes, shapes, optional names, a unique or
  # user-given shared name, capacity and memory limit, plus a no-op used to colocate every put
  # and get with the staging area's device. Provides validation and result-formatting helpers
  # for the concrete subclasses (StagingArea, MapStagingArea, OrderedMapStagingArea).
  # KNOWN ISSUE: the class attributes for an identifier counter and a lock are declared but not
  # used anywhere in this class.
  """Base class for Staging Areas."""
  _identifier = 0
  _lock = threading.Lock()

  def __init__(self,
               dtypes,
               shapes=None,
               names=None,
               shared_name=None,
               capacity=0,
               memory_limit=0):
    # [FUNCTION DOC]
    # Initialises common state. dtypes: list of component types; shapes: optional list of shapes
    # (same length; unknown if null); names: optional names (same length); shared_name: optional
    # string, otherwise a unique graph name derived from the class name is generated; capacity:
    # max elements, zero means unbounded; memory_limit: max bytes, zero means unbounded.
    # Raises ValueError if shared_name is not a string or lengths of shapes/names differ from
    # dtypes. Side effect: creates a no-op in the default graph under a root name scope, used for
    # colocation.
    if shared_name is None:
      self._name = (
          ops.get_default_graph().unique_name(self.__class__.__name__))
    elif isinstance(shared_name, str):
      self._name = shared_name
    else:
      raise ValueError(f"shared_name must be a string, got {shared_name}")

    self._dtypes = dtypes

    if shapes is not None:
      if len(shapes) != len(dtypes):
        raise ValueError("StagingArea shapes must be the same length as dtypes")
      self._shapes = [tensor_shape.TensorShape(s) for s in shapes]
    else:
      self._shapes = [tensor_shape.unknown_shape() for _ in self._dtypes]

    if names is not None:
      if len(names) != len(dtypes):
        raise ValueError("StagingArea names must be the same length as dtypes")
      self._names = names
    else:
      self._names = None

    self._capacity = capacity
    self._memory_limit = memory_limit

    # all get and put ops must colocate with this op
    with ops.name_scope("%s_root" % self._name):
      self._coloc_op = control_flow_ops.no_op()

  @property
  def name(self):
    # [FUNCTION DOC]
    # Read-only property returning the staging area name (also the shared resource name).
    """The name of the staging area."""
    return self._name

  @property
  def dtypes(self):
    # [FUNCTION DOC]
    # Read-only property returning the list of component dtypes.
    """The list of dtypes for each component of a staging area element."""
    return self._dtypes

  @property
  def shapes(self):
    # [FUNCTION DOC]
    # Read-only property returning the list of component shapes.
    """The list of shapes for each component of a staging area element."""
    return self._shapes

  @property
  def names(self):
    # [FUNCTION DOC]
    # Read-only property returning the component names, or null if unnamed.
    """The list of names for each component of a staging area element."""
    return self._names

  @property
  def capacity(self):
    # [FUNCTION DOC]
    # Read-only property returning the element capacity (zero means unbounded).
    """The maximum number of elements of this staging area."""
    return self._capacity

  @property
  def memory_limit(self):
    # [FUNCTION DOC]
    # Read-only property returning the byte limit (zero means unbounded).
    """The maximum number of bytes of this staging area."""
    return self._memory_limit

  def _check_put_dtypes(self, vals, indices=None):
    # [FUNCTION DOC]
    # HELPER: argument validation for put/insert style methods of the staging areas.
    # vals: a tensor, a list/tuple of tensors, or a mapping from component names to tensors;
    # indices: component positions for a list of values (required for lists, inferred from keys
    # for mappings). For mappings the keys must be a subset of the area's names; values are
    # ordered by the position of their names. Checks dtype equality and shape compatibility of
    # every value, then converts each to a tensor. Returns a pair: list of tensors and indices.
    # Raises ValueError for: mapping without names, unknown keys, list given to a named area,
    # missing indices, length mismatch between indices and values, too many values, or dtype
    # mismatch; shape incompatibility raises from the shape check.
    # KNOWN ISSUE: a single non-sequence value is compared by length against indices before being
    # wrapped in a list, so passing a bare tensor fails the length check or relies on tensor
    # length. An empty mapping makes the internal unpacking of zero results fail with a generic
    # error. The unknown-key message lacks a separator before the queue listing.
    """Validate and convert `vals` to a list of `Tensor`s.

    The `vals` argument can be a Tensor, a list or tuple of tensors, or a
    dictionary with tensor values.

    If `vals` is a list, then the appropriate indices associated with the
    values must be provided.

    If it is a dictionary, the staging area must have been constructed with a
    `names` attribute and the dictionary keys must match the staging area names.
    `indices` will be inferred from the dictionary keys.
    If the staging area was constructed with a `names` attribute, `vals` must
    be a dictionary.

    Checks that the dtype and shape of each value matches that
    of the staging area.

    Args:
      vals: A tensor, a list or tuple of tensors, or a dictionary.

    Returns:
      A (tensors, indices) tuple where `tensors` is a list of `Tensor` objects
      and `indices` is a list of indices associated with the tensors.

    Raises:
      ValueError: If `vals` or `indices` is invalid.
    """
    if isinstance(vals, dict):
      if not self._names:
        raise ValueError(
            "Staging areas must have names to enqueue a dictionary")
      if not set(vals.keys()).issubset(self._names):
        raise ValueError("Keys in dictionary to put do not match names "
                         f"of staging area. Dictionary: {sorted(vals.keys())}"
                         f"Queue: {sorted(self._names)}")
      # The order of values in `self._names` indicates the order in which the
      # tensors in the dictionary `vals` must be listed.
      vals, indices, _ = zip(*[(vals[k], i, k)
                               for i, k in enumerate(self._names)
                               if k in vals])
    else:
      if self._names:
        raise ValueError("You must enqueue a dictionary in a staging area "
                         "with names")

      if indices is None:
        raise ValueError("Indices must be supplied when inserting a list "
                         "of tensors")

      if len(indices) != len(vals):
        raise ValueError(f"Number of indices {len(indices)} doesn't match "
                         f"number of values {len(vals)}")

      if not isinstance(vals, (list, tuple)):
        vals = [vals]
        indices = [0]

    # Sanity check number of values
    if not len(vals) <= len(self._dtypes):
      raise ValueError(f"Unexpected number of inputs {len(vals)} vs "
                       f"{len(self._dtypes)}")

    tensors = []

    for val, i in zip(vals, indices):
      dtype, shape = self._dtypes[i], self._shapes[i]
      # Check dtype
      if val.dtype != dtype:
        raise ValueError(f"Datatypes do not match. "
                         f"Received val.dtype {str(val.dtype)} and "
                         f"dtype {str(dtype)}")
      # Check shape
      val.get_shape().assert_is_compatible_with(shape)

      tensors.append(
          ops.convert_to_tensor(val, dtype=dtype, name="component_%d" % i))

    return tensors, indices

  def _create_device_transfers(self, tensors):
    # [FUNCTION DOC]
    # HELPER: ensures returned tensors are placed per the caller's device scope.
    # Takes a tensor or list/tuple of tensors; if the current device scope differs from the
    # staging area's colocation op device, wraps each tensor in an identity op (forcing a
    # transfer). Returns a list of tensors (the original ones if devices match). Adds ops to the
    # graph.
    """Encode inter-device transfers if the current device
    is not the same as the Staging Area's device.
    """

    if not isinstance(tensors, (tuple, list)):
      tensors = [tensors]

    curr_device_scope = control_flow_ops.no_op().device

    if curr_device_scope != self._coloc_op.device:
      tensors = [array_ops.identity(t) for t in tensors]

    return tensors

  def _get_return_value(self, tensors, indices):
    # [FUNCTION DOC]
    # HELPER: formats the results of get-style ops. tensors: outputs of the op; indices: component
    # positions they correspond to. Applies device transfers, sets each output's static shape from
    # the declared shape of its component, then returns a mapping from names to tensors if the area
    # has names, otherwise the list of tensors.
    # KNOWN ISSUE: the docstring says a single tensor may be returned when there is one component,
    # but the code always returns a list (or a mapping).
    """Return the value to return from a get op.

    If the staging area has names, return a dictionary with the
    names as keys.  Otherwise return either a single tensor
    or a list of tensors depending on the length of `tensors`.

    Args:
      tensors: List of tensors from the get op.
      indices: Indices of associated names and shapes

    Returns:
      A single tensor, a list of tensors, or a dictionary
      of tensors.
    """

    tensors = self._create_device_transfers(tensors)

    # Sets shape
    for output, i in zip(tensors, indices):
      output.set_shape(self._shapes[i])

    if self._names:
      # The returned values in `tensors` are in the same order as
      # the names in `self._names`.
      return {self._names[i]: t for t, i in zip(tensors, indices)}
    return tensors

  def _scope_vals(self, vals):
    # [FUNCTION DOC]
    # HELPER: supplies the values for name scopes of put operations.
    # KNOWN ISSUE: docstring promises a list, but a mapping yields a dictionary values view and a
    # list or tuple is returned unchanged.
    # Returns the sequence itself, the mapping values, or a one-element list for a lone tensor.
    """Return a list of values to pass to `name_scope()`.

    Args:
      vals: A tensor, a list or tuple of tensors, or a dictionary.

    Returns:
      The values in vals as a list.
    """
    if isinstance(vals, (list, tuple)):
      return vals
    elif isinstance(vals, dict):
      return vals.values()
    else:
      return [vals]


class StagingArea(BaseStagingArea):
  # [CLASS DOC]
  # Unordered staging area: a buffer of tensor tuples with put (stage) and get/peek/size/clear
  # operations and no ordering guarantees. Storage lives on the device in effect at construction;
  # all ops are colocated with the area's root op. Elements are addressed positionally.
  """Class for staging inputs. No ordering guarantees.

  A `StagingArea` is a TensorFlow data structure that stores tensors across
  multiple steps, and exposes operations that can put and get tensors.

  Each `StagingArea` element is a tuple of one or more tensors, where each
  tuple component has a static dtype, and may have a static shape.

  The capacity of a `StagingArea` may be bounded or unbounded.
  It supports multiple concurrent producers and consumers; and
  provides exactly-once delivery.

  Each element of a `StagingArea` is a fixed-length tuple of tensors whose
  dtypes are described by `dtypes`, and whose shapes are optionally described
  by the `shapes` argument.

  If the `shapes` argument is specified, each component of a staging area
  element must have the respective fixed shape. If it is
  unspecified, different elements may have different shapes,

  It can be configured with a capacity in which case
  put(values) will block until space becomes available.

  Similarly, it can be configured with a memory limit which
  will block put(values) until space is available.
  This is mostly useful for limiting the number of tensors on
  devices such as GPUs.

  All get() and peek() commands block if the requested data
  is not present in the Staging Area.

  """

  def __init__(self,
               dtypes,
               shapes=None,
               names=None,
               shared_name=None,
               capacity=0,
               memory_limit=0):
    # [FUNCTION DOC]
    # Delegates to the base constructor. dtypes: list of component types; shapes, names:
    # optional lists of the same length; shared_name: optional string for sharing between Python
    # objects; capacity and memory_limit: optional bounds, zero meaning unbounded.
    # Raises ValueError for invalid arguments (from the base class).
    """Constructs a staging area object.

    The two optional lists, `shapes` and `names`, must be of the same length
    as `dtypes` if provided.  The values at a given index `i` indicate the
    shape and name to use for the corresponding queue component in `dtypes`.

    The device scope at the time of object creation determines where the
    storage for the `StagingArea` will reside.  Calls to `put` will incur a copy
    to this memory space, if necessary.  Tensors returned by `get` will be
    placed according to the device scope when `get` is called.

    Args:
      dtypes:  A list of types.  The length of dtypes must equal the number
        of tensors in each element.
      shapes: (Optional.) Constraints on the shapes of tensors in an element.
        A list of shape tuples or None. This list is the same length
        as dtypes.  If the shape of any tensors in the element are constrained,
        all must be; shapes can be None if the shapes should not be constrained.
      names: (Optional.) If provided, the `get()` and
        `put()` methods will use dictionaries with these names as keys.
        Must be None or a list or tuple of the same length as `dtypes`.
      shared_name: (Optional.) A name to be used for the shared object. By
        passing the same name to two different python objects they will share
        the underlying staging area. Must be a string.
      capacity: (Optional.) Maximum number of elements.
        An integer. If zero, the Staging Area is unbounded
      memory_limit: (Optional.) Maximum number of bytes of all tensors
        in the Staging Area.
        An integer. If zero, the Staging Area is unbounded

    Raises:
      ValueError: If one of the arguments is invalid.
    """

    super(StagingArea, self).__init__(dtypes, shapes, names, shared_name,
                                      capacity, memory_limit)

  def put(self, values, name=None):
    # [FUNCTION DOC]
    # Builds a stage op that puts one element into the area. values: tensor, list/tuple or mapping
    # of component tensors (count must match the dtypes; a bare tensor is wrapped); name optional.
    # Indices are assumed to be positions zero to n-1. Validates dtypes and shapes (ValueError on
    # mismatch). The op is colocated with the area and blocks at run time when capacity or the
    # memory limit is reached. Returns the op.
    """Create an op that places a value into the staging area.

    This operation will block if the `StagingArea` has reached
    its capacity.

    Args:
      values: A single tensor, a list or tuple of tensors, or a dictionary with
        tensor values. The number of elements must match the length of the
        list provided to the dtypes argument when creating the StagingArea.
      name: A name for the operation (optional).

    Returns:
        The created op.

    Raises:
      ValueError: If the number or type of inputs don't match the staging area.
    """
    with ops.name_scope(name, "%s_put" % self._name,
                        self._scope_vals(values)) as scope:

      if not isinstance(values, (list, tuple, dict)):
        values = [values]

      # Hard-code indices for this staging area
      indices = list(range(len(values)))
      vals, _ = self._check_put_dtypes(values, indices)

      with ops.colocate_with(self._coloc_op):
        op = gen_data_flow_ops.stage(
            values=vals,
            shared_name=self._name,
            name=scope,
            capacity=self._capacity,
            memory_limit=self._memory_limit)

      return op

  def __internal_get(self, get_fn, name):
    # [FUNCTION DOC]
    # HELPER: shared body of get and peek. get_fn is a callback with no arguments that builds the
    # actual op; it is invoked colocated with the area. The result is formatted with all component
    # indices.
    # KNOWN ISSUE: the name parameter is never used.
    with ops.colocate_with(self._coloc_op):
      ret = get_fn()

    indices = list(range(len(self._dtypes)))  # Hard coded
    return self._get_return_value(ret, indices)

  def get(self, name=None):
    # [FUNCTION DOC]
    # Builds an unstage op that removes and returns one element. name: optional op name (default
    # derived from the area name). Built colocated with the area; blocks at run time while empty.
    # Returns the list of tensors (or a mapping by component name for named areas), with static
    # shapes set and a device transfer inserted if the caller's device differs.
    """Gets one element from this staging area.

    If the staging area is empty when this operation executes, it will block
    until there is an element to dequeue.

    Note that unlike others ops that can block, like the queue Dequeue
    operations, this can stop other work from happening.  To avoid this, the
    intended use is for this to be called only when there will be an element
    already available.  One method for doing this in a training loop would be to
    run a `put()` call during a warmup session.run call, and then call both
    `get()` and `put()` in each subsequent step.

    The placement of the returned tensor will be determined by the current
    device scope when this function is called.

    Args:
      name: A name for the operation (optional).

    Returns:
      The tuple of tensors that was gotten.
    """
    if name is None:
      name = "%s_get" % self._name

    # pylint: disable=bad-continuation
    fn = lambda: gen_data_flow_ops.unstage(dtypes=self._dtypes,
                    shared_name=self._name, name=name,
                    capacity=self._capacity,
                    memory_limit=self._memory_limit)
    # pylint: enable=bad-continuation

    return self.__internal_get(fn, name)

  def peek(self, index, name=None):
    # [FUNCTION DOC]
    # Builds an op that reads (without removing) the element at the given position. index: integer
    # position within the area; name: optional op name. Blocks at run time until enough elements
    # exist. Returns the tensors formatted like get.
    """Peeks at an element in the staging area.

    If the staging area is too small to contain the element at
    the specified index, it will block until enough elements
    are inserted to complete the operation.

    The placement of the returned tensor will be determined by
    the current device scope when this function is called.

    Args:
      index: The index of the tensor within the staging area
              to look up.
      name: A name for the operation (optional).

    Returns:
      The tuple of tensors that was gotten.
    """
    if name is None:
      name = "%s_peek" % self._name

    # pylint: disable=bad-continuation
    fn = lambda: gen_data_flow_ops.stage_peek(index,
                    dtypes=self._dtypes, shared_name=self._name,
                    name=name, capacity=self._capacity,
                    memory_limit=self._memory_limit)
    # pylint: enable=bad-continuation

    return self.__internal_get(fn, name)

  def size(self, name=None):
    # [FUNCTION DOC]
    # Builds an op returning the number of elements currently held. name: optional op name.
    # Returns the op (a scalar tensor result).
    """Returns the number of elements in the staging area.

    Args:
        name: A name for the operation (optional)

    Returns:
        The created op
    """
    if name is None:
      name = "%s_size" % self._name

    return gen_data_flow_ops.stage_size(
        name=name,
        shared_name=self._name,
        dtypes=self._dtypes,
        capacity=self._capacity,
        memory_limit=self._memory_limit)

  def clear(self, name=None):
    # [FUNCTION DOC]
    # Builds an op that removes all elements. name: optional op name. Returns the op.
    """Clears the staging area.

    Args:
        name: A name for the operation (optional)

    Returns:
        The created op
    """
    if name is None:
      name = "%s_clear" % self._name

    return gen_data_flow_ops.stage_clear(
        name=name,
        shared_name=self._name,
        dtypes=self._dtypes,
        capacity=self._capacity,
        memory_limit=self._memory_limit)


class MapStagingArea(BaseStagingArea):
  # [CLASS DOC]
  # Staging area acting as an associative container keyed by int64 with tuple-of-tensor values.
  # Supports partial puts (kept in an incomplete map), partial gets, peek, pop by key, pop of
  # an arbitrary or smallest item, size, incomplete size and clear. In addition to base state it
  # holds an ordered flag and a set of function handles that point to either the hashtable or the
  # ordered (tree) generated ops, chosen once at construction.
  """A `MapStagingArea` is a TensorFlow data structure that stores tensors
  across multiple steps, and exposes operations that can put and get tensors.

  Each `MapStagingArea` element is a (key, value) pair.
  Only int64 keys are supported, other types should be
  hashed to produce a key.
  Values are a tuple of one or more tensors.
  Each tuple component has a static dtype,
  and may have a static shape.

  The capacity of a `MapStagingArea` may be bounded or unbounded.
  It supports multiple concurrent producers and consumers; and
  provides exactly-once delivery.

  Each value tuple of a `MapStagingArea` is a fixed-length tuple of tensors
  whose
  dtypes are described by `dtypes`, and whose shapes are optionally described
  by the `shapes` argument.

  If the `shapes` argument is specified, each component of a staging area
  element must have the respective fixed shape. If it is
  unspecified, different elements may have different shapes,

  It behaves like an associative container with support for:

   - put(key, values)
   - peek(key)         like dict.get(key)
   - get(key)          like dict.pop(key)
   - get(key=None)     like dict.popitem()
   - size()
   - clear()

  If ordered a tree structure ordered by key will be used and
  get(key=None) will remove (key, value) pairs in increasing key order.
  Otherwise a hashtable

  It can be configured with a capacity in which case
  put(key, values) will block until space becomes available.

  Similarly, it can be configured with a memory limit which
  will block put(key, values) until space is available.
  This is mostly useful for limiting the number of tensors on
  devices such as GPUs.

  All get() and peek() commands block if the requested
  (key, value) pair is not present in the staging area.

  Partial puts are supported and will be placed in an incomplete
  map until such time as all values associated with the key have
  been inserted. Once completed, this (key, value) pair will be
  inserted into the map. Data in the incomplete map
  counts towards the memory limit, but not towards capacity limit.

  Partial gets from the map are also supported.
  This removes the partially requested tensors from the entry,
  but the entry is only removed from the map once all tensors
  associated with it are removed.
  """

  def __init__(self,
               dtypes,
               shapes=None,
               names=None,
               shared_name=None,
               ordered=False,
               capacity=0,
               memory_limit=0):
    # [FUNCTION DOC]
    # Initialises the base state and selects the generated ops. dtypes: component types; shapes,
    # names: optional lists of equal length; shared_name: optional string; ordered: if true use
    # the key-ordered tree variants of the ops, otherwise hashtable variants (default false);
    # capacity and memory_limit: bounds, zero meaning unbounded. Raises ValueError for invalid
    # arguments (from the base class). Stores the ordered flag and seven op callables
    # (put, pop, pop-any, peek, size, incomplete size, clear).
    """Args:

      dtypes:  A list of types.  The length of dtypes must equal the number
        of tensors in each element.
      capacity: (Optional.) Maximum number of elements.
        An integer. If zero, the Staging Area is unbounded
      memory_limit: (Optional.) Maximum number of bytes of all tensors
        in the Staging Area (excluding keys).
        An integer. If zero, the Staging Area is unbounded
      ordered: (Optional.) If True the underlying data structure
        is a tree ordered on key. Otherwise assume a hashtable.
      shapes: (Optional.) Constraints on the shapes of tensors in an element.
        A list of shape tuples or None. This list is the same length
        as dtypes.  If the shape of any tensors in the element are constrained,
        all must be; shapes can be None if the shapes should not be constrained.
      names: (Optional.) If provided, the `get()` and
        `put()` methods will use dictionaries with these names as keys.
        Must be None or a list or tuple of the same length as `dtypes`.
      shared_name: (Optional.) A name to be used for the shared object. By
        passing the same name to two different python objects they will share
        the underlying staging area. Must be a string.

    Raises:
      ValueError: If one of the arguments is invalid.

    """

    super(MapStagingArea, self).__init__(dtypes, shapes, names, shared_name,
                                         capacity, memory_limit)

    # Defer to different methods depending if the map is ordered
    self._ordered = ordered

    if ordered:
      self._put_fn = gen_data_flow_ops.ordered_map_stage
      self._pop_fn = gen_data_flow_ops.ordered_map_unstage
      self._popitem_fn = gen_data_flow_ops.ordered_map_unstage_no_key
      self._peek_fn = gen_data_flow_ops.ordered_map_peek
      self._size_fn = gen_data_flow_ops.ordered_map_size
      self._incomplete_size_fn = gen_data_flow_ops.ordered_map_incomplete_size
      self._clear_fn = gen_data_flow_ops.ordered_map_clear
    else:
      self._put_fn = gen_data_flow_ops.map_stage
      self._pop_fn = gen_data_flow_ops.map_unstage
      self._popitem_fn = gen_data_flow_ops.map_unstage_no_key
      self._peek_fn = gen_data_flow_ops.map_peek
      self._size_fn = gen_data_flow_ops.map_size
      self._incomplete_size_fn = gen_data_flow_ops.map_incomplete_size
      self._clear_fn = gen_data_flow_ops.map_clear

  def put(self, key, vals, indices=None, name=None):
    # [FUNCTION DOC]
    # Builds an op storing (key, vals), possibly only some components. key: int64 key tensor;
    # vals: tensor, list/tuple or mapping of component tensors; indices: component positions,
    # required for lists, inferred for mappings; name: optional. Validates via the base put
    # checker (ValueError on mismatch), colocates with the area and blocks at run time when a
    # limit is reached. Returns the op.
    """Create an op that stores the (key, vals) pair in the staging area.

    Incomplete puts are possible, preferably using a dictionary for vals
    as the appropriate dtypes and shapes can be inferred from the value names
    dictionary key values. If vals is a list or tuple, indices must
    also be specified so that the op knows at which element position
    to perform the insert.

    This operation will block if the capacity or memory limit of this
    container is reached.

    Args:
        key: Key associated with the data
        vals: Tensor (or a dict/tuple of Tensors) to place
                into the staging area.
        indices: (Optional) if vals is a tuple/list, this is required.
        name: A name for the operation (optional)

    Returns:
        The created op

    Raises:
        ValueError: If the number or type of inputs don't match the staging
        area.
    """

    with ops.name_scope(name, "%s_put" % self._name,
                        self._scope_vals(vals)) as scope:

      vals, indices = self._check_put_dtypes(vals, indices)

      with ops.colocate_with(self._coloc_op):
        op = self._put_fn(
            key,
            indices,
            vals,
            dtypes=self._dtypes,
            shared_name=self._name,
            name=scope,
            capacity=self._capacity,
            memory_limit=self._memory_limit)
    return op

  def _get_indices_and_dtypes(self, indices=None):
    # [FUNCTION DOC]
    # HELPER: normalises the partial-retrieval indices for peek, get and pop.
    # indices: null (meaning all components), or a non-empty list/tuple of all integers or all
    # strings (component names, translated to positions via the area's names). Returns a pair of
    # integer positions and the matching dtypes. Raises TypeError for a non-sequence or mixed
    # types; ValueError for an empty list, names used on an unnamed area, or an unknown name.
    if indices is None:
      indices = list(range(len(self._dtypes)))

    if not isinstance(indices, (tuple, list)):
      raise TypeError(f"Invalid indices type {type(indices)}")

    if len(indices) == 0:
      raise ValueError("Empty indices")

    if all(isinstance(i, str) for i in indices):
      if self._names is None:
        raise ValueError(f"String indices provided {indices}, but "
                         "this Staging Area was not created with names.")

      try:
        indices = [self._names.index(n) for n in indices]
      except ValueError:
        raise ValueError(f"Named index not in "
                         f"Staging Area names {self._names}")
    elif all(isinstance(i, int) for i in indices):
      pass
    else:
      raise TypeError(f"Mixed types in indices {indices}. "
                      "May only be str or int")

    dtypes = [self._dtypes[i] for i in indices]

    return indices, dtypes

  def peek(self, key, indices=None, name=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: when no name is given the op is named with a pop suffix instead of a peek
    # suffix, and the docstring says it returns the created op while the code returns the
    # formatted tensors (list or mapping).
    # Builds an op reading without removing the entry for key. key: int64 key; indices: optional
    # partial list of component positions or names; name: optional. Blocks at run time until the
    # key exists. Returns the retrieved tensors with static shapes set.
    """Peeks at staging area data associated with the key.

    If the key is not in the staging area, it will block
    until the associated (key, value) is inserted.

    Args:
        key: Key associated with the required data
        indices: Partial list of tensors to retrieve (optional).
                A list of integer or string indices.
                String indices are only valid if the Staging Area
                has names associated with it.
        name: A name for the operation (optional)

    Returns:
        The created op
    """

    if name is None:
      name = "%s_pop" % self._name

    indices, dtypes = self._get_indices_and_dtypes(indices)

    with ops.colocate_with(self._coloc_op):
      result = self._peek_fn(
          key,
          shared_name=self._name,
          indices=indices,
          dtypes=dtypes,
          name=name,
          capacity=self._capacity,
          memory_limit=self._memory_limit)

    return self._get_return_value(result, indices)

  def get(self, key=None, indices=None, name=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it returns the created op, but it returns a pair of key and
    # formatted tensors.
    # Removes and returns an entry. key: int64 key, or null to take the smallest key (ordered
    # area) or an arbitrary entry; indices: optional partial component selection; name: optional.
    # Dispatches to the keyed pop or the no-key pop helper. Blocks at run time if absent/empty.
    """If the key is provided, the associated (key, value) is returned from the staging area.

    If the key is not in the staging area, this method will block until
    the associated (key, value) is inserted.
    If no key is provided and the staging area is ordered,
    the (key, value) with the smallest key will be returned.
    Otherwise, a random (key, value) will be returned.

    If the staging area is empty when this operation executes,
    it will block until there is an element to dequeue.

    Args:
        key: Key associated with the required data (Optional)
        indices: Partial list of tensors to retrieve (optional).
                A list of integer or string indices.
                String indices are only valid if the Staging Area
                has names associated with it.
        name: A name for the operation (optional)

    Returns:
        The created op
    """
    if key is None:
      return self._popitem(indices=indices, name=name)
    else:
      return self._pop(key, indices=indices, name=name)

  def _pop(self, key, indices=None, name=None):
    # [FUNCTION DOC]
    # HELPER: keyed removal used by get. key: int64 key; indices: optional partial component
    # selection; name: optional (default uses a get suffix). Builds the unstage op colocated with
    # the area. Returns a pair of the key and the formatted tensors.
    # KNOWN ISSUE: docstring claims it returns the created op, but a key/tensors pair is returned.
    """Remove and return the associated (key, value) is returned from the staging area.

    If the key is not in the staging area, this method will block until
    the associated (key, value) is inserted.
    Args:
        key: Key associated with the required data
        indices: Partial list of tensors to retrieve (optional).
                A list of integer or string indices.
                String indices are only valid if the Staging Area
                has names associated with it.
        name: A name for the operation (optional)

    Returns:
        The created op
    """
    if name is None:
      name = "%s_get" % self._name

    indices, dtypes = self._get_indices_and_dtypes(indices)

    with ops.colocate_with(self._coloc_op):
      result = self._pop_fn(
          key,
          shared_name=self._name,
          indices=indices,
          dtypes=dtypes,
          name=name,
          capacity=self._capacity,
          memory_limit=self._memory_limit)

    return key, self._get_return_value(result, indices)

  def _popitem(self, indices=None, name=None):
    # [FUNCTION DOC]
    # HELPER: no-key removal used by get. indices: optional partial component selection; name:
    # optional. Removes the smallest-key entry for an ordered area, otherwise an arbitrary one.
    # Returns a pair of the key (with a device transfer if needed) and the formatted tensors.
    # KNOWN ISSUE: the docstring lists a key argument that the method does not accept and claims
    # an op is returned.
    """If the staging area is ordered, the (key, value) with the smallest key will be returned.

    Otherwise, a random (key, value) will be returned.
    If the staging area is empty when this operation executes,
    it will block until there is an element to dequeue.

    Args:
        key: Key associated with the required data
        indices: Partial list of tensors to retrieve (optional).
                A list of integer or string indices.
                String indices are only valid if the Staging Area
                has names associated with it.
        name: A name for the operation (optional)

    Returns:
        The created op
    """
    if name is None:
      name = "%s_get_nokey" % self._name

    indices, dtypes = self._get_indices_and_dtypes(indices)

    with ops.colocate_with(self._coloc_op):
      key, result = self._popitem_fn(
          shared_name=self._name,
          indices=indices,
          dtypes=dtypes,
          name=name,
          capacity=self._capacity,
          memory_limit=self._memory_limit)

    # Separate keys and results out from
    # underlying namedtuple
    key = self._create_device_transfers(key)[0]
    result = self._get_return_value(result, indices)

    return key, result

  def size(self, name=None):
    # [FUNCTION DOC]
    # Builds an op returning the number of complete entries. name: optional. Returns the op.
    """Returns the number of elements in the staging area.

    Args:
        name: A name for the operation (optional)

    Returns:
        The created op
    """
    if name is None:
      name = "%s_size" % self._name

    return self._size_fn(
        shared_name=self._name,
        name=name,
        dtypes=self._dtypes,
        capacity=self._capacity,
        memory_limit=self._memory_limit)

  def incomplete_size(self, name=None):
    # [FUNCTION DOC]
    # Builds an op returning the number of partially filled entries. name: optional. Returns it.
    """Returns the number of incomplete elements in the staging area.

    Args:
        name: A name for the operation (optional)

    Returns:
        The created op
    """
    if name is None:
      name = "%s_incomplete_size" % self._name

    return self._incomplete_size_fn(
        shared_name=self._name,
        name=name,
        dtypes=self._dtypes,
        capacity=self._capacity,
        memory_limit=self._memory_limit)

  def clear(self, name=None):
    # [FUNCTION DOC]
    # Builds an op removing all entries, complete and incomplete. name: optional. Returns the op.
    """Clears the staging area.

    Args:
        name: A name for the operation (optional)

    Returns:
        The created op
    """
    if name is None:
      name = "%s_clear" % self._name

    return self._clear_fn(
        shared_name=self._name,
        name=name,
        dtypes=self._dtypes,
        capacity=self._capacity,
        memory_limit=self._memory_limit)


class RecordInput:
  # [CLASS DOC]
  # Configuration holder for the asynchronous record input op, which reads TFRecord files matching
  # a pattern into a shuffling buffer and yields random batches. Stores the settings given at
  # construction; the op itself is only created by get_yield_op.
  """RecordInput asynchronously reads and randomly yields TFRecords.

  A RecordInput Op will continuously read a batch of records asynchronously
  into a buffer of some fixed capacity. It can also asynchronously yield
  random records from this buffer.

  It will not start yielding until at least `buffer_size / 2` elements have been
  placed into the buffer so that sufficient randomization can take place.

  The order the files are read will be shifted each epoch by `shift_amount` so
  that the data is presented in a different order every epoch.
  """

  def __init__(self,
               file_pattern,
               batch_size=1,
               buffer_size=1,
               parallelism=1,
               shift_ratio=0,
               seed=0,
               name=None,
               batches=None,
               compression_type=None):
    # [FUNCTION DOC]
    # Stores settings; creates no op yet. file_pattern: path with wildcards; batch_size: records
    # per batch; buffer_size: buffer capacity; parallelism: reader threads; shift_ratio: fraction
    # of files by which the starting file moves each epoch; seed: random seed; name: optional op
    # name; batches: optional count of batches, in which case the stored batch size is multiplied
    # by it and the output is later split; compression_type: compression of the files, default
    # none. No validation is performed here despite the documented ValueError.
    """Constructs a RecordInput Op.

    Args:
      file_pattern: File path to the dataset, possibly containing wildcards.
        All matching files will be iterated over each epoch.
      batch_size: How many records to return at a time.
      buffer_size: The maximum number of records the buffer will contain.
      parallelism: How many reader threads to use for reading from files.
      shift_ratio: What percentage of the total number files to move the start
        file forward by each epoch.
      seed: Specify the random number seed used by generator that randomizes
        records.
      name: Optional name for the operation.
      batches: None by default, creating a single batch op. Otherwise specifies
        how many batches to create, which are returned as a list when
        `get_yield_op()` is called. An example use case is to split processing
        between devices on one computer.
      compression_type: The type of compression for the file. Currently ZLIB and
        GZIP are supported. Defaults to none.

    Raises:
      ValueError: If one of the arguments is invalid.
    """
    self._batch_size = batch_size
    if batches is not None:
      self._batch_size *= batches
    self._batches = batches
    self._file_pattern = file_pattern
    self._buffer_size = buffer_size
    self._parallelism = parallelism
    self._shift_ratio = shift_ratio
    self._seed = seed
    self._name = name
    self._compression_type = python_io.TFRecordCompressionType.NONE
    if compression_type is not None:
      self._compression_type = compression_type

  def get_yield_op(self):
    # [FUNCTION DOC]
    # Adds the record input op to the graph using the stored settings and returns its output: a
    # tensor of serialized records when batches is null; otherwise a list with one list of
    # scalar record tensors per batch, built by splitting the records and distributing them
    # round robin across batches (inside a name scope of the given name). Each execution of the op
    # yields a fresh group of records.
    """Adds a node that yields a group of records every time it is executed.
    If RecordInput `batches` parameter is not None, it yields a list of
    record batches with the specified `batch_size`.
    """
    compression_type = python_io.TFRecordOptions.get_compression_type_string(
        python_io.TFRecordOptions(self._compression_type))
    records = gen_data_flow_ops.record_input(
        file_pattern=self._file_pattern,
        file_buffer_size=self._buffer_size,
        file_parallelism=self._parallelism,
        file_shuffle_shift_ratio=self._shift_ratio,
        batch_size=self._batch_size,
        file_random_seed=self._seed,
        compression_type=compression_type,
        name=self._name)
    if self._batches is None:
      return records
    else:
      with ops.name_scope(self._name):
        batch_list = [[] for _ in range(self._batches)]
        records = array_ops.split(records, self._batch_size, 0)
        for index, protobuf in enumerate(records):
          batch_index = index % self._batches
          batch_list[batch_index].append(array_ops.reshape(protobuf, []))
        return batch_list
