# [MODULE DOC]
# Implements the automatic control dependency context manager used when tracing functions: it
# makes stateful ops run and orders ops that use the same resource. Exposes op-type sets
# (asynchronous, legacy random, order-insensitive and skipped stateful ops, stateless allowlist),
# the stateful predicate, resource-type enum, collective manager id extraction, the context
# manager class, a registry and decorator for resource resolvers (with a default identity
# resolver registered at import time), and a function decorator. Import-time side effect:
# creates the resolver registry and registers the identity resolver.
# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""AutomaticControlDependencies and related functionality."""

import collections
import enum

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import auto_control_deps_utils as utils
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import registry
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_decorator

# LINT.IfChange
# Op types that should not run in program order, e.g. because they need to run
# asynchronously to avoid deadlock.

ASYNC_STATEFUL_OPS = frozenset((
    "CollectiveGather",
    "CollectiveReduce",
    "CollectiveBcastSend",
    "CollectiveBcastSendV2",
    "CollectiveBcastRecv",
    "CollectiveBcastRecvV2",
    "NcclAllReduce",
    # We do not add "Send" here since we want it to be added as a control output
    # in order to avoid being pruned.
    "Recv",
    "CollectiveInitializeCommunicator",
    "CollectiveAssignGroupV2",
))

LEGACY_RANDOM_OPS = frozenset((
    # These may be used in variable initializers -- thus their execution should
    # not be dependent on other stateful operations.  This is because although
    # according to program order, tf.Variables may be created in sequence,
    # their initialization happens outside of the program order (specifically,
    # in graph mode their initialization happens by calling a grouped
    # initializer operation or in eager mode, where initialization is lifted
    # out of the tf.function and executed the first time the function is
    # executed).
    #
    # Unless there is a specific dependency between the initializers
    # themselves (e.g. one initializer depends on a Variable whose value depends
    # on another initializer), the initialization can happen in any order so
    # long as it's before the associated Variable read operations.
    #
    # Note that in general the randomness of legacy random operations is only
    # guaranteed by providing a graph-level and op-level seed (and ordering of
    # the same op across multiple iterations of a while_loop is specifically not
    # guaranteed; see the discussion below).
    #
    # There is a possible race condition inside while_loop where the same
    # random OpKernel instantiation is reused across multiple steps
    # of the loop.  Since legacy Random OpKernels have an internal rng state,
    # automatic dependency tracking across loop steps would likely
    # fix this race; and for that case this denylist is problematic.
    # However, since automatic dependency tracking inside while loops is not
    # currently supported, and there are no other examples of OpKernel reuse
    # (each OpKernel is associated with a unique op in graph mode),
    # this denylist has no effect on the aforementioned behavior.
    #
    # TODO(ebrevdo,skyewm): Modify the check against this denylist to
    # only occur when the op is inside a "variable initialization scope"; and
    # add proper autodeps inside while_loops that respects this updated check.
    "RandomUniform",
    "RandomUniformInt",
    "RandomStandardNormal",
    "ParameterizedTruncatedNormal",
    "TruncatedNormal",
    "RandomShuffle",
    "Multinomial",
    "RandomGamma",
    "RandomGammaGrad",
    "RandomPoisson",
    "RandomPoissonV2",
))

MUST_RUN_ORDER_INSENSITIVE_STATEFUL_OPS = frozenset((
    "InfeedEnqueue",
    "InfeedEnqueueTuple",
    "EnqueueTPUEmbeddingSparseBatch",
    "EnqueueTPUEmbeddingIntegerBatch",
    "EnqueueTPUEmbeddingSparseTensorBatch",
    "EnqueueTPUEmbeddingRaggedTensorBatch",
    "EnqueueTPUEmbeddingArbitraryTensorBatch",
    "DynamicEnqueueTPUEmbeddingArbitraryTensorBatch",
))

# These ops are order-insensitive ans should in theory run, but at the moment
# they either always have the necessary data dependencies, or have workarounds
# in existing code that would break when adding new control deps. This
# inconsistency should be eventually fixed, but it would be more effective to
# retire the list instead.
SKIPPED_ORDER_INSENSITIVE_STATEFUL_OPS = frozenset((
    "CudnnRNN",
    "CudnnRNNBackprop",
    "CudnnRNNV2",
    "CudnnRNNV3",
    "CudnnRNNBackpropV2",
    "CudnnRNNBackpropV3",
    "RestoreV2",
    "SaveV2",
))
# LINT.ThenChange(//tensorflow/core/grappler/optimizers/function_optimizer.cc)

# Op types that are marked as stateless, but should be allowlisted to add auto
# control dependencies.
_ALLOWLIST_STATELESS_OPS = [
    # As TPU collective ops are blocking, if there are more than one collective
    # op in the function, we need to make sure different collectives ops are
    # scheduled in certain orders. Otherwise if at the same time all the
    # replicas are launching different collective ops/programs, it may cause
    # deadlock.
    "AllToAll",
    "CrossReplicaSum",
    "CollectivePermute",
]


def op_is_stateful(op):
  # [FUNCTION DOC]
  # HELPER: predicate used by automatic control dependencies to pick ops needing ordering.
  # Returns true if the operation is flagged stateful and its type is not in the asynchronous,
  # legacy random or skipped order-insensitive exclusion sets, or if its type is in the allowlist
  # of stateless ops that still need control dependencies. Input is a graph operation; no side
  # effects.
  # pylint: disable=protected-access
  ret = ((op._is_stateful and
          ((op.type not in ASYNC_STATEFUL_OPS) and
           (op.type not in LEGACY_RANDOM_OPS) and
           (op.type not in SKIPPED_ORDER_INSENSITIVE_STATEFUL_OPS))) or
         (op.type in _ALLOWLIST_STATELESS_OPS))
  return ret


class ResourceType(enum.Enum):
  # [CLASS DOC]
  # Enumeration of how an op uses a resource handle: read-only or read-write. Used by the
  # automatic control dependency logic to decide whether uses must be serialized.
  READ_ONLY = "read-only"
  READ_WRITE = "read-write"


def collective_manager_ids_from_op(op):
  # [FUNCTION DOC]
  # Extracts collective manager ids from an operation. For a collective reduce op returns a
  # one-element list with its manager id attribute; for a stateful partitioned call returns the
  # list stored in the collective manager ids attribute; in either case, if the attribute is
  # missing (value error), or for any other op type, returns an empty list. No side effects.
  """Returns CollectiveManager ID from the op if one exists, else None.

  CollectiveManager adds collective and no_op operations tagged with an ID,
  unique to the manager object. This function extracts that ID, or None, if the
  node was not generated by a CollectiveManager.

  Args:
    op: `Operation` to get the collective manager ID from.

  Returns:
    List of CollectiveManager IDs used by the op.
  """
  if op.type == "CollectiveReduce":
    try:
      return [op.get_attr("_collective_manager_id")]
    except ValueError:
      pass
  elif op.type == "StatefulPartitionedCall":
    try:
      return op.get_attr(utils.COLLECTIVE_MANAGER_IDS)
    except ValueError:
      pass
  return []


class AutomaticControlDependencies(object):
  # [CLASS DOC]
  # Context manager used while building graphs or functions that, on exit, rewrites the ops
  # created inside it so that stateful ops all run and ops touching the same resource execute in
  # program order. State: identity set of tensors marked as returns, the public set of ops that
  # must run, a list of independent ops, and (in graph mode) the graph and starting operation
  # count; after exit also the collective manager scopes used. Created with no arguments and used
  # in a with statement; call mark_as_return on results and run_independently for special ops.
  # In eager mode it is a no-op. Not thread safe.
  """Context manager to automatically add control dependencies.

  Code under this context manager will act as if a sensible set of control
  dependencies were present. More specifically:
    1. All stateful ops in the scope will execute (with the exception of ops in
       ASYNC_STATEFUL_OPS and LEGACY_RANDOM_OPS)
    2. Stateful ops which modify the same resource will execute in program order

  Note: creating variables in an automatic control dependencies context is not
  supported (the value of the variables will never change as they will keep
  getting reinitialized).

  NOT THREAD SAFE
  """

  def __init__(self):
    # [FUNCTION DOC]
    # Constructor with no arguments. Initializes an identity-based set of returned tensors, the
    # public set of ops that must run (filled in on exit), and an empty list of independent ops.
    self._returned_tensors = object_identity.ObjectIdentitySet()
    self.ops_which_must_run = set()
    self._independent_ops = []

  def mark_as_return(self, tensor):
    # [FUNCTION DOC]
    # Wraps a value in identity ops and records the new tensors as returned tensors so that, on
    # exit, they receive control dependencies on all must-run ops. Input tensor may be a dense
    # tensor, indexed slices, sparse tensor or tensor array (identity applied to values and
    # indices, or to the flow, respectively; dense shape is kept as is). Returns a new object of
    # the same kind built from the identity copies. Side effects: adds identity ops to the graph
    # and updates the internal set of returned tensors.
    """Acts like identity but marks the `Tensor` as a return value.

    This will possibly return a copy of the `Tensor`. Usage:

    ```
      with AutomaticControlDependencies() as a:
       ...
       t = a.mark_as_return(t)
      _ = ...(t...)  # i.e. it's safe to use t here
    ```

    Args:
      tensor: the `Tensor` to be marked

    Returns:
      a copy of the `Tensor`.
    """
    if isinstance(tensor, indexed_slices.IndexedSlices):
      values = array_ops.identity(tensor.values)
      indices = array_ops.identity(tensor.indices)
      self._returned_tensors.add(indices)
      self._returned_tensors.add(values)
      return indexed_slices.IndexedSlices(
          values, indices, dense_shape=tensor.dense_shape)
    elif isinstance(tensor, sparse_tensor.SparseTensor):
      values = array_ops.identity(tensor.values)
      indices = array_ops.identity(tensor.indices)
      self._returned_tensors.add(indices)
      self._returned_tensors.add(values)
      return sparse_tensor.SparseTensor(
          indices, values, dense_shape=tensor.dense_shape)
    elif isinstance(tensor, tensor_array_ops.TensorArray):
      flow = array_ops.identity(tensor.flow)
      self._returned_tensors.add(flow)
      return tensor_array_ops.build_ta_with_new_flow(tensor, flow)
    # We want to make the return values depend on the stateful operations, but
    # we don't want to introduce a cycle, so we make the return value the result
    # of a new identity operation that the stateful operations definitely don't
    # depend on.
    tensor = array_ops.identity(tensor)
    self._returned_tensors.add(tensor)
    return tensor

  def run_independently(self, op):
    # [FUNCTION DOC]
    # Marks an operation as independent: it is appended to the list of independent ops and tagged
    # with an attribute stating it has independent side effects. On exit such ops are forced to
    # run before the returned values but get no ordering edges relative to other ops. Input op is
    # the graph operation. Returns nothing. Mutates the op's attributes.
    """Marks the given op as independent.

    Overrides any other rule for the op.

    Independent ops are guaranteed to execute before the return values, but
    are allowed to run in parallel with everything else. Use in programs which
    can guarantee that an op has side effects that don't affect any other op.

    Args:
      op: An operation
    """
    self._independent_ops.append(op)
    op._set_attr("_independent_side_effects", attr_value_pb2.AttrValue(b=True))  # pylint: disable=protected-access

  def __enter__(self):
    # [FUNCTION DOC]
    # Enters the context. In eager mode simply returns the manager without touching anything.
    # In graph mode: remembers the default graph, turns on the graph's flag that enables
    # automatic control dependencies, registers this manager on the graph, and records how many
    # operations exist so only later ops are processed on exit. Returns the manager itself.
    # Not thread safe: assumes no other thread adds ops to the graph meanwhile.
    if context.executing_eagerly():
      return self
    # This code assumes no other thread is adding ops to the graph while
    # we're adding ops to the graph.
    # TODO(apassos): Fix this by locking the graph or using a temporary
    # graph (but that would mess up devices and collections at least,
    # probably other things as well).
    g = ops.get_default_graph()
    self._graph = g
    g._add_control_dependencies = True  # pylint: disable=protected-access
    g.experimental_acd_manager = self
    self._n_operations = g.num_operations()
    return self

  def _process_switch(self, switch_op, ops_which_must_run,
                      last_write_to_resource, merge_for_resource):
    # [FUNCTION DOC]
    # HELPER: internal step of the context exit logic that handles conditional branches.
    # Inputs: the switch op whose first input is a resource, the set of ops that must run
    # (mutated), a mapping from resource tensor ids to the last op that wrote them (mutated), and
    # a mapping from resource tensor ids to the merge op that must follow all their uses
    # (mutated). Recursively processes an upstream switch feeding this one first, returns early if
    # this switch was already handled, otherwise creates an artificial merge over the switch
    # outputs placed in the outer control flow context, marks it must-run, orders the switch after
    # the previous user of the resource, records the merge as the new last writer, and maps every
    # switch output to the merge. Returns nothing.
    """Processes a switch node for a resource input.

    When tensorflow creates a cond, it creates a control flow context for each
    branch of the cond. Each external tensor accessed by that branch is routed
    through a switch op, which gets created in the graph _after_ the op which
    uses that tensor get created.

    If the resource comes from another switch op we process that one first.

    _process_switch creates a corresponding merge node for the switch node. This
    merge node is added to the outer control flow context of the switch
    node. We also ensure that:

      1. The switch node executes after the previous op which used the resource
         tensor

      2. Any op which uses a resource output of the switch node executes before
         the merge for the switch node.

      3. The next op which uses the input resource to the switch node (which
         might be another switch node for the other branch of the conditional)
         will execute after the merge node is done.

      4. The merge node is marked as must_run so it will run even if no
         subsequent operation uses the resource.

    Args:
      switch_op: the switch op to be processed
      ops_which_must_run: the set of ops which must run
      last_write_to_resource: map from resource tensor to last op updating
        it
      merge_for_resource: map from resource tensor to merge which must follow
        all usages of it.
    """
    # pylint: disable=protected-access
    inp = switch_op.inputs[0]
    input_id = ops.tensor_id(inp)
    if inp.dtype == dtypes_module.resource and inp.op.type == "Switch":
      self._process_switch(inp.op, ops_which_must_run, last_write_to_resource,
                           merge_for_resource)
    output = switch_op.outputs[0]
    output_id = ops.tensor_id(output)
    if output_id in merge_for_resource:
      return
    new_merge = control_flow_ops.merge(
        switch_op.outputs, name="artificial_merge")
    new_merge[0].op._control_flow_context = (
        switch_op._control_flow_context.outer_context)
    # Ensures the merge always runs
    ops_which_must_run.add(new_merge[0].op)
    if input_id in last_write_to_resource:
      # Ensures the switch executes after the previous op using the resource.
      switch_op._add_control_input(last_write_to_resource[input_id])
    # Ensure the next op outside the cond happens after the merge.
    last_write_to_resource[input_id] = new_merge[0].op
    if input_id in merge_for_resource:
      merge_for_resource[input_id]._add_control_input(new_merge[0].op)
    for o in switch_op.outputs:
      # Ensures the merge will execute after all ops inside the cond
      merge_for_resource[ops.tensor_id(o)] = new_merge[0].op

  def __exit__(self, unused_type, unused_value, unused_traceback):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the local flag for building-function mode is only assigned inside the loop over
    # resource inputs, yet it is read afterwards when filtering control inputs; for an op with no
    # resource inputs it holds a stale value from an earlier op (it is only read when there are
    # control inputs, which implies it was set for this op, so this is latent rather than active).
    # KNOWN ISSUE: exceptions raised inside the with-block are not specially handled; the
    # dependency rewriting still runs, and the exception arguments are ignored.
    # Leaves the context and rewrites the graph. In eager mode does nothing. In graph mode:
    # raises a runtime error if the default graph changed since entry; restores the graph's
    # add-control-dependencies flag from the outer graph (or false) and clears the manager hook.
    # Then walks all operations created inside the context in creation order (skipping ops in
    # while loops) and: marks stateful ops (and unregistered ops, and consumed read-only
    # resource reads) as must-run; runs order-insensitive enqueue ops independently; makes
    # merge ops depend on all pending must-run ops; for each resource input, serializes ops by
    # adding control edges from the last writer (and from reads since the last write when the op
    # writes), handling switch ops through artificial merges; chains stateful ops without resource
    # inputs outside control flow in program order; chains collective manager scopes; and
    # drops control inputs from different control flow contexts when not building a function.
    # Finally records must-run ops, and makes each marked return tensor depend on them (through a
    # single no-op when building a function, else per control flow context), and stores the used
    # collective manager scopes in the collective manager ids attribute. Returns nothing.
    # pylint: disable=protected-access
    if context.executing_eagerly():
      return

    if self._graph is not ops.get_default_graph():
      raise RuntimeError(
          "Within the automatic control dependency context, the default graph"
          f" cannot change. Upon entry it was {self._graph}, but on exit it"
          f" changed to {ops.get_default_graph()}")

    outer_graph = getattr(self._graph, "outer_graph", None)
    if outer_graph is not None:
      self._graph._add_control_dependencies = outer_graph._add_control_dependencies
    else:
      self._graph._add_control_dependencies = False
    self._graph.experimental_acd_manager = None

    # map from resource tensor to the last op which wrote to it
    last_write_to_resource = {}
    # map from resource tensor to the list of reads from it since the last
    # write or since the beginning of the function.
    reads_since_last_write_to_resource = collections.defaultdict(list)
    # CollectiveManager manager_ids within a particular function call should not
    # be needed outside of that function call. So we keep them separate (though
    # the general idea of the maps is the same, in the future, we'll need to
    # correctly thread the control output outside).
    # Map from collective manager scope to the last op which used it
    collective_manager_scopes_opened = {}
    collective_manager_scopes_used = {}
    # set of conditional and loop exits
    ops_which_must_run = set()
    # merge which must depend on ops which use this resource
    merge_for_resource = {}

    new_operations = self._graph.get_operations()[self._n_operations:]

    # Ensures that uses of resource tensors get serialized properly and all
    # execute. This is done by keeping a map from resource tensor to the last op
    # in graph-construction order which used it (last_write_to_resource).
    #
    # Conditionals are written in TensorFlow such that every external tensor
    # accessed in the conditional goes through a switch op and every return
    # tensor (it's guaranteed that there will be at least one) goes through a
    # merge op.
    #
    # To handle conditionals, switches are handled in a special way (see
    # comments for _process_switch). Merge nodes created by TF's conditional
    # logic (as opposed to by _process_switch) are forced to run and also get a
    # control dependency added to them to ensure all stateful ops inside their
    # control flow context run.
    #
    # We also ensure that if an op is using a resource output by a switch node
    # (that is, a resource tensor for which there's a value in
    # merge_for_resource) this op will run before the merge for that resource.
    #
    # We try to add control inputs to nodes respecting their control flow
    # contexts to avoid dead nodes propagating everywhere and leading to
    # "retval[0] doesn't have value" errors. If a node gets a control dependency
    # on a dead node (i.e. a note from an untaken control flow branch) that node
    # will be marked as dead unless it's a merge node.
    #
    # TODO(apassos): serialize non-resource-taking stateful ops as well, and
    # test that it works. Support while loops. Support init_scope escaping from
    # this.
    for op in new_operations:
      # TODO(apassos) make this code safely support while loops.
      if control_flow_util.IsInWhileLoop(op):
        continue
      control_inputs = set()

      if op.type in MUST_RUN_ORDER_INSENSITIVE_STATEFUL_OPS:
        # This will add it to self._independent_ops, but also mark it with an
        # attribute.
        self.run_independently(op)

      if op in self._independent_ops:
        ops_which_must_run.add(op)
        continue

      # Ensure stateful ops run.
      # Read-only ops are added to control outputs if the read value is
      # consumed. This covers the case when the read value is returned from
      # the function since that goes through a tf.identity in mark_as_return.
      if ((op_def_registry.get(op.type) is None) or
          (op_is_stateful(op) and
           (op.type not in utils.RESOURCE_READ_OPS or
            any(output.consumers() for output in op.outputs)))):
        ops_which_must_run.add(op)

      # Make a note of all opened manager_ids.
      if op.type == "NoOp":
        try:
          collective_manager_scopes_opened[op.get_attr(
              "_collective_manager_id")] = op
        except ValueError:
          pass
      # Ignore switches (they're handled separately)
      if op.type == "Switch" and op.inputs[0].dtype == dtypes_module.resource:
        continue
      # Make merges trigger all other computation which must run
      # TODO(mdan): Don't do this. Write a transform to chains instead.
      # See core/common_runtime/control_flow_deps_to_chains.cc.
      if op.type == "Merge":
        for o in ops_which_must_run:
          op._add_control_input(o)
          for inp in o.inputs:
            input_id = ops.tensor_id(inp)
            if input_id in last_write_to_resource:
              last_write_to_resource[input_id] = op
        ops_which_must_run = set([op])
        continue

      resource_inputs = set()
      # Check for any resource inputs. If we find any, we update control_inputs
      # and last_write_to_resource.
      for inp, resource_type in _get_resource_inputs(op):
        is_read = resource_type == ResourceType.READ_ONLY
        input_id = ops.tensor_id(inp)

        # If the op receives the same resource tensor twice as an input, we skip
        # to avoid the op getting a control dependency on itself.
        if input_id in resource_inputs:
          continue

        resource_inputs.add(input_id)
        # Deal with switches, finally.
        if inp.op.type == "Switch":
          self._process_switch(inp.op, ops_which_must_run,
                               last_write_to_resource, merge_for_resource)
        is_building_function = op.graph.building_function
        # Ensure uses of resources are serialized
        if input_id in last_write_to_resource:
          if is_building_function or (
              last_write_to_resource[input_id]._control_flow_context
              is op._control_flow_context):
            control_inputs.add(last_write_to_resource[input_id])
        # Ensure merges happen after the closing of a cond block
        if input_id in merge_for_resource:
          merge_for_resource[input_id]._add_control_input(op)
        if is_read:
          reads_since_last_write_to_resource[input_id].append(op)
        else:
          control_inputs.update(reads_since_last_write_to_resource[input_id])
          reads_since_last_write_to_resource[input_id] = []
          last_write_to_resource[input_id] = op

      if (op_is_stateful(op) and not resource_inputs
          and op._control_flow_context is None):
        if None in last_write_to_resource:
          op._add_control_input(last_write_to_resource[None])
        last_write_to_resource[None] = op

      # Ensure ordering of collective ops
      manager_ids = collective_manager_ids_from_op(op)
      for manager_id in manager_ids:
        if manager_id in collective_manager_scopes_opened:
          # Chain this function call if the scope was opened.
          op._add_control_input(collective_manager_scopes_opened[manager_id])
          collective_manager_scopes_opened[manager_id] = op
        else:
          # If this op is in a scope not created here, create a chain starting
          # at this op.
          if manager_id in collective_manager_scopes_used:
            op._add_control_input(collective_manager_scopes_used[manager_id])
          collective_manager_scopes_used[manager_id] = op

      if control_inputs and not is_building_function:
        control_inputs = [
            c for c in control_inputs
            if c._control_flow_context is op._control_flow_context
        ]

      op._add_control_inputs(control_inputs)

    # Ensure all ops which must run do run
    self.ops_which_must_run.update(ops_which_must_run)

    control_output_op = None
    for idx, r in enumerate(
        nest.flatten(list(self._returned_tensors), expand_composites=True)):
      if self.ops_which_must_run:
        updated_ops_which_must_run = []
        if r.graph.building_function:
          # There may be many stateful ops in the graph. Adding them as
          # control inputs to each function output could create excessive
          # control edges in the graph. Thus we create an intermediate No-op to
          # chain the control dependencies between stateful ops and function
          # outputs.
          if idx == 0:
            control_output_op = control_flow_ops.no_op()
            control_output_op._add_control_inputs(self.ops_which_must_run)
          updated_ops_which_must_run = [control_output_op]
        else:
          updated_ops_which_must_run = [
              o for o in self.ops_which_must_run
              if o._control_flow_context is r.op._control_flow_context
          ]
        r.op._add_control_inputs(updated_ops_which_must_run)

    self.collective_manager_ids_used = collective_manager_scopes_used


_acd_resource_resolvers_registry = registry.Registry("acd_resource_resolvers")


def register_acd_resource_resolver(f):
  # [FUNCTION DOC]
  # Registers a resource resolver callback in the module-level resolver registry, so that it is
  # consulted for every operation processed by automatic control dependencies. Input f is the
  # callback taking the operation and two sets (reads, writes). Returns f unchanged so it can be
  # used as a decorator. Side effect: mutates the global registry; the registry may raise an
  # error if a resolver with the same name is already registered.
  """Register a function for resolving resources touched by an op.

  `f` is called for every Operation added in the ACD context with the op's
  original resource reads and writes. `f` is expected to update the sets of
  resource reads and writes in-place and return True if it updated either of the
  sets, False otherwise.

  Example:
  @register_acd_resource_resolver
  def identity_resolver(op, resource_reads, resource_writes):
    # op: The `Operation` being processed by ACD currently.
    # resource_reads: An `ObjectIdentitySet` of read-only resources.
    # resource_writes: An `ObjectIdentitySet` of read-write resources.
    def update(resource_inputs):
      to_remove = []
      to_add = []
      for resource in resource_inputs:
        if resource.op.type == "Identity":
          to_remove.append(resource)
          to_add.extend(resource.op.inputs)
      for t in to_remove:
        resource_inputs.discard(t)
      resource_inputs.update(to_add)
      return to_add or to_remove
    return update(resource_reads) or update(resource_writes)

  Args:
    f: Python function with signature
    (Operation, ObjectIdentitySet, ObjectIdentitySet) -> bool

  Returns:
    The function `f` after adding it to the registry.
  """
  _acd_resource_resolvers_registry.register(f)
  return f


@register_acd_resource_resolver
def _identity_resolver(op, resource_reads, resource_writes):
  # [FUNCTION DOC]
  # HELPER: default resource resolver registered with the automatic control dependency machinery.
  # Inputs: the operation being processed (ignored), the set of read-only resource tensors and
  # the set of read-write resource tensors. For each set, replaces every resource produced by an
  # Identity op with the inputs of that Identity op, mutating the sets in place. Returns a
  # truthy value (non-empty list of additions or removals) if either set changed, otherwise a
  # falsy value. Note the return is a list rather than a strict boolean.
  """Replaces Identity output with its input in resource_inputs."""
  del op
  def update(resource_inputs):
    # [FUNCTION DOC]
    # HELPER: inner helper of the identity resolver. Takes one mutable set of resource tensors,
    # swaps each Identity-produced resource for the inputs of its Identity op in place, and
    # returns the list of added resources, or the list of removed ones if nothing was added
    # (empty, so falsy, when nothing changed).
    to_remove = []
    to_add = []
    for resource in resource_inputs:
      if resource.op.type == "Identity":
        to_remove.append(resource)
        to_add.extend(resource.op.inputs)
    for t in to_remove:
      resource_inputs.discard(t)
    resource_inputs.update(to_add)
    return to_add or to_remove

  return update(resource_reads) or update(resource_writes)


def _get_resource_inputs(op):
  # [FUNCTION DOC]
  # HELPER: used by AutomaticControlDependencies.__exit__ to find which resources an op touches.
  # Generator. Starts from the read and write resource inputs reported by the utils module for
  # the op, then repeatedly applies every registered resource resolver (which may rewrite the
  # sets in place) until a full pass makes no updates; after any update, resources that are also
  # writes are removed from the read set. Yields pairs of (resource tensor, resource type): first
  # all remaining reads as read-only, then all writes as read-write. A handle that is never
  # written is treated as read-only. Side effects: resolvers may mutate the sets they receive.
  """Returns an iterable of resources touched by this `op`."""
  reads, writes = utils.get_read_write_resource_inputs(op)
  saturated = False
  while not saturated:
    saturated = True
    for key in _acd_resource_resolvers_registry.list():
      # Resolvers should return true if they are updating the list of
      # resource_inputs.
      # TODO(srbs): An alternate would be to just compare the old and new set
      # but that may not be as fast.
      updated = _acd_resource_resolvers_registry.lookup(key)(op, reads, writes)
      if updated:
        # Conservatively remove any resources from `reads` that are also writes.
        reads = reads.difference(writes)
      saturated = saturated and not updated

  # Note: A resource handle that is not written to is treated as read-only. We
  # don't have a special way of denoting an unused resource.
  for t in reads:
    yield (t, ResourceType.READ_ONLY)
  for t in writes:
    yield (t, ResourceType.READ_WRITE)


def automatic_control_dependencies(f):
  # [FUNCTION DOC]
  # Decorator factory-style helper: takes a callable f and returns a wrapper (preserving f's
  # decorator metadata via the tf_decorator machinery) that runs f under an automatic control
  # dependency context and marks all of its flattened results as return values. The wrapper
  # returns the result structure of f with each element replaced by its marked copy. No side
  # effects at wrapping time; the graph is only modified when the wrapper is called.
  """Wraps f to automatically insert control dependencies.

  The inserted dependencies ensure that:
    1. All stateful ops in f run when the result of f runs
    2. Updates to the same resources happen in order.

  Args:
    f: the function to be wrapped.

  Returns:
    The wrapped function.
  """

  def wrapper(*args, **kwargs):
    # [FUNCTION DOC]
    # HELPER: inner wrapper created by the decorator automatic_control_dependencies.
    # Calls the wrapped function inside a fresh automatic control dependency context, passes all
    # positional and keyword arguments through unchanged, flattens the result structure, marks
    # every flattened element as a return value (so each becomes dependent on all stateful ops),
    # and repacks the marked values into the same structure as the original result. Returns that
    # repacked structure. Exceptions from the wrapped function propagate; the context's exit
    # logic may raise a runtime error if the default graph changed. Side effects: adds identity
    # and control-dependency edges to the current graph (graph mode only).
    with AutomaticControlDependencies() as a:
      result = f(*args, **kwargs)
      result_flat = [a.mark_as_return(t) for t in nest.flatten(result)]
      return nest.pack_sequence_as(result, result_flat)

  return tf_decorator.make_decorator(f, wrapper)
