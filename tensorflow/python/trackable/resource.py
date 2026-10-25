# [MODULE DOC]
# Defines the trackable "resource" object classes used by checkpointing and SavedModel export.
# A resource is an object that owns a tensor handle (for example a hash table or variable
# handle) that tf.function code can capture. The module provides a tracker and a scope that
# collect resources created in a block of code, a metaclass that lets creator hooks intercept
# resource construction, the base class CapturableResource, the user-facing TrackableResource
# (exported as saved_model.experimental.TrackableResource) and RestoredResource used when
# loading a SavedModel. Import-time side effect: creates the module-level global stack of
# resource trackers (initially empty) and registers the public export of TrackableResource.
#
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
"""Definitions for resource-type trackable object classes."""

import contextlib
import copy
import weakref

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.trackable import base
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export

# global _RESOURCE_TRACKER_STACK
_RESOURCE_TRACKER_STACK = []


class ResourceTracker:
  # [CLASS DOC]
  # Simple collector that records resource objects created while it is active (see
  # resource_tracker_scope). Holds a single private list of resources, in creation order.
  # Uses slots so instances carry no other attributes. Created with no arguments.
  """An object that tracks a list of resources."""

  __slots__ = ["_resources"]

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor. Takes no arguments and starts with an empty list of tracked resources.
    # Returns nothing and has no side effects beyond setting the private list.
    self._resources = []

  @property
  def resources(self):
    # [FUNCTION DOC]
    # Property returning the live (not copied) list of resources recorded so far, in the
    # order they were added. Callers mutating it affect the tracker.
    return self._resources

  def add_resource(self, resource):
    # [FUNCTION DOC]
    # Appends the given resource object to the tracked list. No deduplication is done.
    # Returns nothing.
    self._resources.append(resource)


@tf_contextlib.contextmanager
def resource_tracker_scope(resource_tracker):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring example code block is never closed (missing closing fence),
  # and the docstring claims it yields a scope although it yields nothing.
  # Context manager (generator based) that makes the given ResourceTracker active for the
  # duration of the with-block, so that every TrackableResource constructed inside registers
  # itself with it. Input: resource_tracker, a ResourceTracker instance. Yields nothing
  # useful (control to the block). Side effects: pushes the tracker onto the module-global
  # tracker stack on entry and, on exit (even when the block raises), replaces the stack with
  # the copy saved on entry. No exceptions of its own; block exceptions propagate.
  """A context to manage resource trackers.

  Use this in order to collect up all resources created within a block of code.
  Example usage:

  ```python
  resource_tracker = ResourceTracker()
  with resource_tracker_scope(resource_tracker):
    resource = TrackableResource()

  assert resource_tracker.resources == [resource]

  Args:
    resource_tracker: The passed in ResourceTracker object

  Yields:
    A scope in which the resource_tracker is active.
  """
  global _RESOURCE_TRACKER_STACK
  old = list(_RESOURCE_TRACKER_STACK)
  _RESOURCE_TRACKER_STACK.append(resource_tracker)
  try:
    yield
  finally:
    _RESOURCE_TRACKER_STACK = old


def _make_getter(captured_getter, captured_previous):
  # [FUNCTION DOC]
  # HELPER: used by the resource metaclass to chain resource-creator hooks.
  # Builds a closure that calls captured_getter with captured_previous as its first argument
  # followed by whatever arguments the closure receives, and returns that result. Exists so
  # each loop iteration binds its own values rather than sharing loop variables.
  """To avoid capturing loop variables."""

  def getter(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: the closure returned by the enclosing factory.
    # Forwards all positional and keyword arguments to the captured creator, passing the
    # captured next creator as first argument, and returns whatever it returns.
    return captured_getter(captured_previous, *args, **kwargs)

  return getter


class _ResourceMetaclass(type):
  # [CLASS DOC]
  # HELPER: metaclass behind CapturableResource.
  # Overrides instance creation so that creator functions registered on the default graph
  # (keyed by the resource type name) wrap construction like a middleware chain.
  """Metaclass for CapturableResource."""

  def __call__(cls, *args, **kwargs):
    # [FUNCTION DOC]
    # Intercepts construction of a resource class. Builds a chain: the innermost creator
    # allocates the object and runs its constructor with the given arguments; each creator
    # registered in the default graph's resource creator stack for this class's resource type
    # wraps it, in order. Calls the outermost one with the constructor arguments and returns
    # the new instance. Reads the current default graph; creators may raise anything.

    def default_resource_creator(next_creator, *a, **kw):
      # [FUNCTION DOC]
      # HELPER: innermost creator in the chain. Asserts no further creator follows (raises an
      # assertion error otherwise), creates a bare instance of the class, runs its constructor
      # with the given arguments and returns the instance.
      assert next_creator is None
      obj = cls.__new__(cls, *a, **kw)
      obj.__init__(*a, **kw)
      return obj

    previous_getter = lambda *a, **kw: default_resource_creator(None, *a, **kw)
    resource_creator_stack = ops.get_default_graph()._resource_creator_stack
    for getter in resource_creator_stack[cls._resource_type()]:
      previous_getter = _make_getter(getter, previous_getter)

    return previous_getter(*args, **kwargs)


class CapturableResource(base.Trackable, metaclass=_ResourceMetaclass):
  # [CLASS DOC]
  # Base class for trackable objects that own a lazily created resource handle tensor.
  # State: the cached handle, the required placement device string and the context factory
  # used to enter a graph or eager mode when the object is destroyed. Subclasses override the
  # create, initialize and destroy hooks. Construction goes through the resource metaclass so
  # creator hooks can intercept it. Handles are exported into SavedModel graphs and the hooks
  # are saved as functions when saving as a SavedModel.
  """Holds a Tensor which a tf.function can capture.

  `CapturableResource`s are discovered by traversing the graph of object
  attributes, e.g. during `tf.saved_model.save`. They are excluded from the
  scope-based tracking of `TrackableResource`; generally things that require
  initialization should inherit from `TrackableResource` instead of
  `CapturableResource` directly.
  """

  def __init__(self, device=""):
    # [FUNCTION DOC]
    # Constructor. Input: device, optional string naming a required placement (default empty
    # meaning no constraint). Sets the cached handle to nothing, stores the device, and records
    # the context in which destruction should run: eager mode if currently executing eagerly,
    # otherwise the current default graph. Returns nothing.
    """Initialize the `CapturableResource`.

    Args:
      device: A string indicating a required placement for this resource,
        e.g. "CPU" if this resource must be created on a CPU device. A blank
        device allows the user to place resource creation, so generally this
        should be blank unless the resource only makes sense on one device.
    """
    self._resource_handle_value = None
    self._resource_device = device
    self._self_destruction_context = (
        context.eager_mode if context.executing_eagerly()
        else ops.get_default_graph().as_default)

  @classmethod
  def _resource_type(cls):
    # [FUNCTION DOC]
    # HELPER: class method giving the key under which creator hooks are looked up.
    # Returns the class name as a string.
    return cls.__name__

  @property
  def _destruction_context(self):
    # [FUNCTION DOC]
    # HELPER: property returning the callable that makes a context manager used while
    # destroying the resource. Falls back to a context that does nothing if the stored one is
    # missing (for example when the object was not fully constructed).
    return getattr(self, "_self_destruction_context",
                   # no-op context
                   contextlib.suppress)

  @_destruction_context.setter
  def _destruction_context(self, destruction_context):
    # [FUNCTION DOC]
    # HELPER: setter storing the callable that produces the destruction context manager.
    # Input: destruction_context, a callable. Returns nothing.
    self._self_destruction_context = destruction_context

  def _create_resource(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the error message names TrackableResource although this is defined on
    # CapturableResource.
    # Hook that subclasses must override to create and return the resource handle tensor.
    # This default always raises a not-implemented error.
    """A function that creates a resource handle."""
    raise NotImplementedError("TrackableResource._create_resource not "
                              "implemented.")

  @property
  def _resource_handle(self):
    # [FUNCTION DOC]
    # HELPER: property returning the cached handle value, or nothing if not yet created.
    return self._resource_handle_value

  @_resource_handle.setter
  def _resource_handle(self, value):
    # [FUNCTION DOC]
    # HELPER: setter caching the handle. If the value is a graph or eager tensor, it gets a
    # weak back-reference to this object stored on it as its parent trackable. Returns nothing.
    if isinstance(value, (tensor.Tensor, ops.EagerTensor)):
      value._parent_trackable = weakref.ref(self)  # pylint: disable=protected-access
    self._resource_handle_value = value

  def _initialize(self):
    # [FUNCTION DOC]
    # Optional hook for subclasses to initialize the resource. Default does nothing.
    """A function that initializes the resource. Optional."""
    pass

  def _destroy_resource(self):
    # [FUNCTION DOC]
    # Optional hook for subclasses to release the resource. Default does nothing. Called from
    # the destructor inside the destruction context.
    """A function that destroys the resource. Optional."""
    pass

  @property
  def resource_handle(self):
    # [FUNCTION DOC]
    # Public property returning the resource handle tensor. On first access, creates it by
    # calling the create hook under the configured device placement and caches it; later
    # accesses return the cached handle. Exceptions from the create hook propagate.
    """Returns the resource handle associated with this Resource."""
    if self._resource_handle is None:
      with ops.device(self._resource_device):
        self._resource_handle = self._create_resource()
    return self._resource_handle

  def _export_to_saved_model_graph(
      self, object_map, tensor_map, **unused_kwargs):
    # [FUNCTION DOC]
    # Part of the Trackable interface for SavedModel export. Inputs: object_map, a mapping
    # from original objects to their exported copies (mutated); tensor_map, a mapping from
    # original tensors to exported tensors (mutated); extra keyword arguments are ignored.
    # Makes a shallow copy of this object, creates a fresh resource on it (under the device
    # placement), records the copy and the handle correspondence in the maps, and returns a
    # list holding this object's original resource handle (which is created if needed).
    """For implementing `Trackable`."""
    new_obj = copy.copy(self)
    # pylint: disable=protected-access
    with ops.device(self._resource_device):
      new_resource = new_obj._create_resource()
    new_obj._resource_handle = new_resource
    # pylint: enable=protected-access
    object_map[self] = new_obj
    tensor_map[self.resource_handle] = new_resource
    return [self.resource_handle]

  def _trackable_children(self, save_type=base.SaveType.CHECKPOINT, **kwargs):
    # [FUNCTION DOC]
    # Returns the mapping of child name to trackable for saving. Inputs: save_type, which kind
    # of save is happening (default checkpoint); extra keyword arguments passed to the parent
    # implementation. Starts from the parent's children; when saving as a SavedModel it adds
    # three zero-argument tf.function wrappers (without autograph) named for the create,
    # initialize and destroy hooks, so the loader can recreate the resource. Returns the
    # updated mapping.
    children = super()._trackable_children(save_type, **kwargs)
    if save_type == "savedmodel":
      @def_function.function(input_signature=[], autograph=False)
      def _creator():
        # [FUNCTION DOC]
        # HELPER: traced function that calls the owner's create hook and returns the new
        # resource handle.
        resource = self._create_resource()
        return resource

      @def_function.function(input_signature=[], autograph=False)
      def _initializer():
        # [FUNCTION DOC]
        # HELPER: traced function that calls the owner's initialize hook and returns the
        # constant one as a dummy output.
        self._initialize()
        return 1  # Dummy return

      @def_function.function(input_signature=[], autograph=False)
      def _destroyer():
        # [FUNCTION DOC]
        # HELPER: traced function that calls the owner's destroy hook and returns the
        # constant one as a dummy output.
        self._destroy_resource()
        return 1  # Dummy return

      children.update({
          "_create_resource": _creator,
          "_initialize": _initializer,
          "_destroy_resource": _destroyer,
      })
    return children

  def __del__(self):
    # [FUNCTION DOC]
    # Destructor. Enters the destruction context and calls the destroy hook. Any exception
    # (for example from interpreter shutdown races or an already collected related object) is
    # swallowed silently so no error is logged. Returns nothing.
    try:
      # Outer race condition: on program exit, the destruction context may be
      # deleted before this __del__ is called. At this point we can safely
      # exit without calling _destroy_resource() and let Python handle things.
      with self._destruction_context():
        # Inner race condition: possible between this and `ScopedTFFunction`
        # whereby if an entire garbage collection chain containing both
        # objects is moved to unreachable during the same garbage collection
        # cycle, the __del__ for `ScopedTFFunction` can be collected before
        # this method is called. In that case, we can't do much but
        # continue.
        self._destroy_resource()
    except Exception:  # pylint: disable=broad-except
      # Silence all error logs that occur when attempting to destroy this
      # resource.
      pass


@tf_export("saved_model.experimental.TrackableResource")
class TrackableResource(CapturableResource):
  # [CLASS DOC]
  # Public resource base class (exported as saved_model.experimental.TrackableResource) for
  # stateful tensors needing initialization. Differs from CapturableResource in that each new
  # instance registers itself with every active resource tracker (see resource_tracker_scope).
  # Subclasses override the create, initialize and destroy hooks.
  """Holds a Tensor which a tf.function can capture.

  A TrackableResource is most useful for stateful Tensors that require
  initialization, such as `tf.lookup.StaticHashTable`. `TrackableResource`s
  are discovered by traversing the graph of object attributes, e.g. during
  `tf.saved_model.save`.

  A TrackableResource has three methods to override:

  * `_create_resource` should create the resource tensor handle.
  * `_initialize` should initialize the resource held at `self.resource_handle`.
  * `_destroy_resource` is called upon a `TrackableResource`'s destruction
    and should decrement the resource's ref count. For most resources, this
    should be done with a call to `tf.raw_ops.DestroyResourceOp`.

  Example usage:

  >>> class DemoResource(tf.saved_model.experimental.TrackableResource):
  ...   def __init__(self):
  ...     super().__init__()
  ...     self._initialize()
  ...   def _create_resource(self):
  ...     return tf.raw_ops.VarHandleOp(dtype=tf.float32, shape=[2])
  ...   def _initialize(self):
  ...     tf.raw_ops.AssignVariableOp(
  ...         resource=self.resource_handle, value=tf.ones([2]))
  ...   def _destroy_resource(self):
  ...     tf.raw_ops.DestroyResourceOp(resource=self.resource_handle)
  >>> class DemoModule(tf.Module):
  ...   def __init__(self):
  ...     self.resource = DemoResource()
  ...   def increment(self, tensor):
  ...     return tensor + tf.raw_ops.ReadVariableOp(
  ...         resource=self.resource.resource_handle, dtype=tf.float32)
  >>> demo = DemoModule()
  >>> demo.increment([5, 1])
  <tf.Tensor: shape=(2,), dtype=float32, numpy=array([6., 2.], dtype=float32)>
  """

  def __init__(self, device=""):
    # [FUNCTION DOC]
    # Constructor. Input: device, optional placement string (default empty). Adds this object
    # to every tracker currently on the global tracker stack, then runs the parent constructor
    # with the device. Returns nothing. Side effect: mutates the active trackers.
    """Initialize the `TrackableResource`.

    Args:
      device: A string indicating a required placement for this resource,
        e.g. "CPU" if this resource must be created on a CPU device. A blank
        device allows the user to place resource creation, so generally this
        should be blank unless the resource only makes sense on one device.
    """
    global _RESOURCE_TRACKER_STACK
    for resource_tracker in _RESOURCE_TRACKER_STACK:
      resource_tracker.add_resource(self)
    super().__init__(device=device)


# TODO(b/124205571,b/124092991): Solve destruction of resources.
class RestoredResource(TrackableResource):
  # [CLASS DOC]
  # Resource object produced when loading a SavedModel. Behaves like TrackableResource, but
  # its create hook can be replaced by a restored function and it accepts restored children as
  # plain attributes. Instances are built through the proto deserialization class method.
  """Restored SavedResource."""

  def __init__(self, device=""):
    # [FUNCTION DOC]
    # Constructor. Input: device, optional placement string (default empty). Simply forwards
    # to the parent constructor. Returns nothing.
    super().__init__(device=device)

  @classmethod
  def _deserialize_from_proto(cls, object_proto, dependencies, **unused_kwargs):
    # [FUNCTION DOC]
    # Class method that rebuilds a restored resource from its saved proto. Inputs:
    # object_proto, the saved object message (its resource device field gives the placement);
    # dependencies, mapping of child names to restored objects; extra keyword arguments are
    # ignored. Creates an instance with that device and, if a dependency named for the create
    # hook exists, installs it as the instance's create hook. Returns the new object.
    obj = cls(device=object_proto.resource.device)
    resource_creator = dependencies.get("_create_resource")
    if resource_creator is not None:
      obj._create_resource = resource_creator  # pylint: disable=protected-access
    return obj

  def _add_trackable_child(self, name, value):
    # [FUNCTION DOC]
    # Attaches a restored child during loading. Inputs: name, attribute name string; value,
    # the restored object. Sets it as an attribute, and if it is a trackable that is not a
    # tf.function wrapper, also registers it as a tracked dependency under that name. Returns
    # nothing.
    setattr(self, name, value)
    if (isinstance(value, base.Trackable) and
        not isinstance(value, def_function.Function)):
      self._track_trackable(value, name)
