# [MODULE DOC]
# HELPER: utility module supporting the TPU distribution strategies and TPU variables.
# Small helpers for TPU distribution code: finding the enclosing TPU replicate context and its
# graph, temporarily leaving that context, wrapping raw variable assign and scatter operations so
# they run in the right graph and device scope (including packed variables), and tracking lazily
# initialized TPU variables until they are needed. Also defines an uninitialized-variable subclass
# that forces initialization on first read or assignment. No import-time side effects.
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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
"""Utility functions for TPU."""

import contextlib

from tensorflow.python.distribute import packed_distributed_variable as packed
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.tpu import tpu_replication


def enclosing_tpu_context():
  # [FUNCTION DOC]
  # Returns the TPU replicate context enclosing the current default graph, or nothing (null value)
  # when not inside a TPU rewrite. Takes no inputs, no side effects; delegates to the
  # context-and-graph lookup and drops the graph.
  """Returns the TPUReplicateContext, which exists inside a tpu.rewrite()."""
  return enclosing_tpu_context_and_graph()[0]


def enclosing_tpu_context_and_graph():
  # [FUNCTION DOC]
  # Walks from the current default graph outward (through outer graphs of function graphs) and,
  # in each graph, through the chain of control-flow contexts, looking for a TPU replicate
  # context. Returns a pair of that context and the graph it was found in, or a pair of two null
  # values if none exists. Takes no inputs; no side effects.
  """Returns the TPUReplicateContext which exists inside a tpu.rewrite(), and its associated graph."""
  graph = ops.get_default_graph()
  while graph is not None:
    ctx = graph._get_control_flow_context()  # pylint: disable=protected-access
    while ctx is not None:
      if isinstance(ctx, tpu_replication.TPUReplicateContext):
        return ctx, graph
      ctx = ctx.outer_context
    # This may be a FuncGraph due to defuns or v2 control flow. We need to
    # find the original graph with the XLAControlFlowContext.
    graph = getattr(graph, "outer_graph", None)
  return None, None


@contextlib.contextmanager
def outside_or_skip_tpu_context():
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says it returns a context manager, but it is itself the generator
  # behind a context manager decorator. Also the graph's control-flow context is restored only
  # on normal exit; if the body raises, the original context is not restored (no try/finally).
  # Context manager that, if inside a TPU replicate context, temporarily sets the graph's
  # control-flow context to the TPU context's outer context, then restores the saved one after the
  # body. If not inside a TPU context, does nothing. Takes no inputs; yields nothing.
  """Returns a context manager that skips current enclosing context if there is any."""
  ctx, graph = enclosing_tpu_context_and_graph()
  if ctx is None:
    yield
  else:
    saved_context = graph._get_control_flow_context()  # pylint: disable=protected-access
    graph._set_control_flow_context(ctx.outer_context)  # pylint: disable=protected-access
    yield
    graph._set_control_flow_context(saved_context)  # pylint: disable=protected-access


@contextlib.contextmanager
def _maybe_enter_graph(tensor):
  # [FUNCTION DOC]
  # HELPER: used by the raw assign and scatter wrappers to build ops in the right graph.
  # Context manager taking a tensor (such as a variable handle). If executing eagerly, the tensor
  # is an eager tensor, or a default graph exists, it simply yields; otherwise it enters the
  # tensor's own graph as default for the duration of the body. Yields nothing.
  # Note: might have an eager tensor but not be executing eagerly when
  # building functions.
  if (context.executing_eagerly() or isinstance(tensor, ops.EagerTensor) or
      ops.has_default_graph()):
    yield
  else:
    with tensor.graph.as_default():
      yield


@contextlib.contextmanager
def _maybe_on_device(var):
  # [FUNCTION DOC]
  # HELPER: used by the raw assign and scatter wrappers to pin ops for packed variables.
  # Context manager taking a variable. If it is a packed variable-and-device object, the body runs
  # under a device scope for that variable's device; otherwise no device scope is added.
  # Yields nothing.
  # Add a device scope for packed variables.
  if isinstance(var, packed.PackedVarAndDevice):
    with ops.device(var.device):
      yield
  else:
    yield


def make_raw_assign_fn(raw_assign_fn, use_handle=True):
  # [FUNCTION DOC]
  # HELPER: builds assign functions for TPU (possibly packed) variables.
  # Takes a raw assign function and a flag use_handle (default true: apply it to the variable's
  # handle; false: apply it to the variable itself). Returns a new assign function taking a
  # variable, value, ignored use_locking, optional name and read_value flag (default true). It
  # enters the right graph and device scope, converts the value to the variable's dtype, runs the
  # raw assign, and then returns a read of the variable (after the assign) or, when read_value is
  # false, the assign operation.
  """Wrap `raw_assign_fn` with the proper graph context and device scope.

  Args:
    raw_assign_fn: the function to be wrapped.
    use_handle: if True, the `raw_assign_fn` will be applied to the handle of a
      variable; otherwise it will be applied to the variable itself.

  Returns:
    The wrapped function.
  """

  def assign_fn(var, value, use_locking=False, name=None, read_value=True):
    # [FUNCTION DOC]
    # Assigns value to var through the wrapped raw function. use_locking is ignored. Returns the
    # variable's value read after the assignment, or the assign operation when read_value is
    # false. Side effect: adds assign ops in the variable's graph and device.
    del use_locking  # Unused.

    handle = var.handle if use_handle else var
    with _maybe_enter_graph(handle), _maybe_on_device(var):
      op = raw_assign_fn(
          handle, ops.convert_to_tensor(value, dtype=var.dtype), name=name)
      with ops.control_dependencies([op]):
        if read_value:
          return var._read_variable_op() if use_handle else var.read_value()  # pylint: disable=protected-access
        else:
          return op

  return assign_fn


def make_raw_scatter_xxx_fn(raw_scatter_xxx_fn):
  # [FUNCTION DOC]
  # HELPER: builds scatter-update functions for TPU (possibly packed) variables.
  # Takes a raw scatter function (taking handle, indices, values and name) and returns a wrapper
  # that works with or without a packed handle by entering the right graph and device scope.
  """Wrap `raw_scatter_xxx_fn` so that it can be called w/ and w/o packed handle."""

  def scatter_xxx_fn(var, sparse_delta, use_locking=False, name=None):  # pylint: disable=missing-docstring
    # [FUNCTION DOC]
    # Applies the raw scatter operation to the variable's handle using the indices and values of
    # the sparse delta (values converted to the variable's dtype). use_locking is ignored.
    # Returns a read of the variable taken after the scatter op. Adds ops to the graph.
    del use_locking  # Unused.

    handle = var.handle
    with _maybe_enter_graph(handle), _maybe_on_device(var):
      op = raw_scatter_xxx_fn(
          handle,
          sparse_delta.indices,
          ops.convert_to_tensor(sparse_delta.values, var.dtype),
          name=name)
      with ops.control_dependencies([op]):
        return var._read_variable_op()  # pylint: disable=protected-access

  return scatter_xxx_fn


class LazyVariableTracker(object):
  # [CLASS DOC]
  # Keeps a list of lazily created (uninitialized) TPU variables and initializes them all at once
  # on demand. State: a private list of pending variables. Created with no arguments; variables
  # are registered with the add method and flushed with the initialize-all method.
  """Class to track uninitialized lazy variables."""

  def __init__(self):
    # [FUNCTION DOC]
    # Creates the tracker with an empty list of pending uninitialized variables.
    self._uninitialized_var_list = []

  def initialize_all(self):
    # [FUNCTION DOC]
    # Assigns each pending variable its stored initial value, on the variable's (possibly packed)
    # handle device, then empties the pending list. Runs inside the init scope. When more than one
    # variable is pending the assignments are wrapped in a compiled-off, non-autograph function
    # for a single call; otherwise they run directly. Takes no inputs, returns nothing.
    """Initialize all uninitialized lazy variables stored in scope."""

    def assign_function(uninitialized_var_list):
      # [FUNCTION DOC]
      # For each variable in the given list, issues an assign-variable op with its initial value on
      # its handle's device (using the packed handle if there is one). Returns an empty constant
      # tensor so the function has an output.
      for var in uninitialized_var_list:
        val = var._initial_value  # pylint: disable=protected-access
        packed_var = getattr(var, "_packed_var", None)
        handle = getattr(packed_var, "packed_handle", var.handle)

        with ops.device(handle.device):
          resource_variable_ops.AssignVariableOp(resource=handle, value=val)
      return constant_op.constant([])

    assign_tf_function = def_function.function(
        assign_function, autograph=False, jit_compile=False,)

    with ops.init_scope():
      if len(self._uninitialized_var_list) > 1:
        assign_tf_function(self._uninitialized_var_list)
      else:
        assign_function(self._uninitialized_var_list)

    self._uninitialized_var_list = []

  def add_uninitialized_var(self, var):
    # [FUNCTION DOC]
    # Registers a variable to be initialized later by the initialize-all method. Returns nothing.
    self._uninitialized_var_list.append(var)


class TPUUninitializedVariable(resource_variable_ops.UninitializedVariable):
  # [CLASS DOC]
  # Uninitialized resource variable for TPU. Every read or assign first triggers initialization of
  # all lazy variables via the variable's lazy scope (a tracker set elsewhere), then defers to the
  # parent class behavior.
  """UninitializedVariable component for TPU.

  Sometimes user might assign (different values) to a single component of a
  mirrored TPU variable. Thus we need to initialize_all when the assign* or read
  is invoked on a single component.
  """

  def read_value(self):
    # [FUNCTION DOC]
    # Initializes all lazy variables in the lazy scope, then returns the parent's read value.
    self._lazy_scope.initialize_all()
    return super().read_value()

  def assign_sub(self, delta, use_locking=None, name=None, read_value=True):
    # [FUNCTION DOC]
    # Initializes all lazy variables, then subtracts delta via the parent implementation, passing
    # through use_locking, name and read_value. Returns the parent's result.
    self._lazy_scope.initialize_all()
    return super().assign_sub(
        delta, use_locking=use_locking, name=name, read_value=read_value
    )

  def assign(self, value, use_locking=None, name=None, read_value=True):
    # [FUNCTION DOC]
    # Initializes all lazy variables, then assigns value via the parent implementation, passing
    # through use_locking, name and read_value. Returns the parent's result.
    self._lazy_scope.initialize_all()
    return super().assign(
        value, use_locking=use_locking, name=name, read_value=read_value
    )

  def assign_add(self, delta, use_locking=None, name=None, read_value=True):
    # [FUNCTION DOC]
    # Initializes all lazy variables, then adds delta via the parent implementation, passing
    # through use_locking, name and read_value. Returns the parent's result.
    self._lazy_scope.initialize_all()
    return super().assign_add(
        delta, use_locking=use_locking, name=name, read_value=read_value
    )
