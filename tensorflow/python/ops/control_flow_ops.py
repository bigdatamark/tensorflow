# [MODULE DOC]
# Core control flow operations for TensorFlow graph construction: low-level frame primitives
# (identity, next-iteration, enter, exit, switch, merge), the control flow context classes that
# track which ops belong to a conditional, while loop or XLA scope (base ControlFlowContext,
# CondContext, WhileContext, XLAControlFlowContext) together with their protocol-buffer
# (de)serialization, and user-facing helpers for control dependencies (with_dependencies, group,
# tuple, and the v2 tuple). It also re-exports every generated raw control flow op via a wildcard
# import of the generated ops module, and keeps the builtin tuple type under a private alias
# because this module redefines the name tuple. Import-time side effects: registers the exported
# public API symbols (group, tuple, an internal enclosing-XLA-context getter) through the export
# decorators and the dispatch machinery. Many functions here are internal building blocks used by
# the cond and while_loop implementations in sibling modules.
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
"""Control Flow Operations.

See the [autograph](https://www.tensorflow.org/guide/autograph) guide.
"""
# pylint: disable=g-bad-name
import abc

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.protobuf import control_flow_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util as util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
# go/tf-wildcard-import
# pylint: disable=wildcard-import,undefined-variable
from tensorflow.python.ops.gen_control_flow_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.util import compat
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export


# We override the 'tuple' for a control flow op, so we keep python's
# existing 'tuple' for later use in this module.
_basetuple = tuple


# pylint: disable=protected-access


def _Identity(tensor, name=None):
  # [FUNCTION DOC]
  # HELPER: internal building block for control flow (loop variables and switch outputs).
  # Returns an identity of the given tensor. Inputs: tensor (a tensor or composite tensor; variables
  # are first converted to tensors) and an optional op name. Reference-typed tensors go through the
  # reference identity op, normal tensors through the ordinary identity op, and composite tensors
  # are handled recursively component by component. Raises a type error for any other input.
  # Creates identity ops in the current graph.
  """Return a tensor with the same shape and contents as the input tensor.

  Args:
    tensor: A Tensor.
    name: A name for this operation (optional).

  Returns:
    A Tensor with the same type and value as the input Tensor.
  """
  tensor = ops.internal_convert_to_tensor_or_composite(tensor, as_ref=True)
  # TODO(b/246438937): Remove this when we expand ResourceVariables into
  # dt_resource tensors.
  tensor = variable_utils.convert_variables_to_tensors(tensor)
  if isinstance(tensor, tensor_lib.Tensor):
    if tensor.dtype._is_ref_dtype:  # pylint: disable=protected-access
      return gen_array_ops.ref_identity(tensor, name=name)
    else:
      return array_ops.identity(tensor, name=name)
  elif isinstance(tensor, composite_tensor.CompositeTensor):
    return nest.map_structure(_Identity, tensor, expand_composites=True)
  else:
    raise TypeError("'tensor' must be a Tensor or CompositeTensor. "
                    f"Received: {type(tensor)}.")


def _NextIteration(tensor, name=None):
  # [FUNCTION DOC]
  # HELPER: internal building block for while loops (builds the back edge source).
  # Inputs: tensor (a tensor or composite tensor) and an optional op name. Returns the output of a
  # next-iteration op (the reference variant for reference-typed tensors); composite tensors are
  # mapped recursively over their components. Raises a type error for any other input type.
  # Adds next-iteration ops to the current graph.
  tensor = ops.internal_convert_to_tensor_or_composite(tensor, as_ref=True)
  if isinstance(tensor, tensor_lib.Tensor):
    if tensor.dtype._is_ref_dtype:  # pylint: disable=protected-access
      return ref_next_iteration(tensor, name=name)
    else:
      return next_iteration(tensor, name=name)
  elif isinstance(tensor, composite_tensor.CompositeTensor):
    return nest.map_structure(_NextIteration, tensor, expand_composites=True)
  else:
    raise TypeError("'tensor' must be a Tensor or CompositeTensor. "
                    f"Received: {type(tensor)}.")


def _Enter(tensor,
           frame_name,
           is_constant=False,
           parallel_iterations=10,
           use_ref=True,
           use_input_shape=True,
           name=None):
  # [FUNCTION DOC]
  # HELPER: internal building block that moves values into a loop frame.
  # KNOWN ISSUE: the docstring's Raises section describes a value error about a shape invariant,
  # but this function has no shape invariant parameter and never raises a value error itself; the
  # only error it raises is a type error when the input is neither a tensor nor a composite tensor.
  # Creates or finds a child frame and makes the tensor available inside it. Inputs: tensor (tensor
  # or composite tensor), frame_name (unique frame name), is_constant (whether the value is constant
  # in the frame, default false), parallel_iterations (max parallel iterations, default 10), use_ref
  # (use the reference enter op for reference-typed tensors, default true), use_input_shape (copy the
  # input static shape onto the result, default true), and an optional op name. Returns the entered
  # tensor, or a structure of entered components for composite tensors. Adds enter ops to the graph.
  """Creates or finds a child frame, and makes `tensor` available to it.

  The unique `frame_name` is used by the `Executor` to identify frames. If
  `is_constant` is true, `tensor` is a constant in the child frame; otherwise
  it may be changed in the child frame. At most `parallel_iterations`
  iterations are run in parallel in the child frame.

  Args:
    tensor: The tensor to be made available to the child frame.
    frame_name: The name of the child frame.
    is_constant: If true, the output is constant within the child frame.
    parallel_iterations: The number of iterations allowed to run in parallel.
    use_ref: If true, use ref_enter if tensor is of ref type.
    use_input_shape: If true, set the result's shape based on tensor's shape.
    name: A name for this operation (optional).

  Returns:
    The same tensor as `tensor`.

  Raises:
    ValueError: If any tensor in `tensor` has a less specific shape
      than its corresponding shape in `shape_invariant`.
  """
  tensor = ops.internal_convert_to_tensor_or_composite(tensor, as_ref=True)
  if isinstance(tensor, tensor_lib.Tensor):
    if tensor.dtype._is_ref_dtype and use_ref:  # pylint: disable=protected-access
      result = gen_control_flow_ops.ref_enter(
          tensor, frame_name, is_constant, parallel_iterations, name=name)
    else:
      result = gen_control_flow_ops.enter(
          tensor, frame_name, is_constant, parallel_iterations, name=name)
    if use_input_shape:
      result.set_shape(tensor.get_shape())
    return result
  elif isinstance(tensor, composite_tensor.CompositeTensor):

    def enter_component(t):
      # [FUNCTION DOC]
      # HELPER: per-component callback used when entering a composite tensor. Takes one component
      # tensor and returns it entered into the same frame with the same settings as the outer call.
      return _Enter(t, frame_name, is_constant, parallel_iterations, use_ref,
                    use_input_shape)

    return nest.map_structure(enter_component, tensor, expand_composites=True)
  else:
    raise TypeError("'tensor' must be a Tensor or CompositeTensor. "
                    f"Received: {type(tensor)}.")


def exit(tensor, name=None):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Exits the current frame to its parent frame, making the tensor available there. Inputs: tensor
  # (tensor or composite tensor) and an optional op name. Returns the exit op output (reference exit
  # for reference-typed tensors), or a structure of exited components for composite tensors. Raises a
  # type error for any other input type. Adds exit ops to the current graph. Shadows the builtin name.
  """Exits the current frame to its parent frame.

  Exit makes its input `tensor` available to the parent frame.

  Args:
    tensor: The tensor to be made available to the parent frame.
    name: A name for this operation (optional).

  Returns:
    The same tensor as `tensor`.
  """
  tensor = ops.internal_convert_to_tensor_or_composite(tensor, as_ref=True)
  if isinstance(tensor, tensor_lib.Tensor):
    if tensor.dtype._is_ref_dtype:  # pylint: disable=protected-access
      return gen_control_flow_ops.ref_exit(tensor, name)
    else:
      return gen_control_flow_ops._exit(tensor, name)
  elif isinstance(tensor, composite_tensor.CompositeTensor):
    return nest.map_structure(exit, tensor, expand_composites=True)
  else:
    raise TypeError("'tensor' must be a Tensor or CompositeTensor. "
                    f"Received: {type(tensor)}.")


def switch(data, pred, dtype=None, name=None):
  # [FUNCTION DOC]
  # Routes data to one of two outputs depending on a boolean predicate. Inputs: data (tensor or
  # composite tensor), pred (scalar boolean, converted to a tensor), dtype (optional element type
  # used when converting data), name (optional op name scope). Returns a pair (false output, true
  # output); for composite tensors each component is switched separately and the pair is repacked
  # into the original structure. Raises a type error if data is neither a tensor nor composite tensor.
  # Adds switch ops to the graph.
  """Forwards `data` to an output determined by `pred`.

  If `pred` is false, the `data` input is forwarded to the first output.
  Otherwise, the data goes to the second output.

  This op handles `Tensor`s and `IndexedSlices`.

  Args:
    data: The tensor to be forwarded to the appropriate output.
    pred: A scalar that specifies which output port will receive data.
    dtype: Optional element type for the returned tensor. If missing, the type
      is inferred from the type of `value`.
    name: A name for this operation (optional).

  Returns:
    `(output_false, output_true)`: If `pred` is true, data will be forwarded
    to `output_true`, otherwise it goes to `output_false`.
  """
  with ops.name_scope(name, "Switch", [data, pred]) as name:
    data = ops.internal_convert_to_tensor_or_composite(
        data, dtype=dtype, name="data", as_ref=True)
    pred = ops.convert_to_tensor(pred, name="pred")
    if isinstance(data, tensor_lib.Tensor):
      return gen_control_flow_ops.switch(data, pred, name=name)
    else:
      if not isinstance(data, composite_tensor.CompositeTensor):
        raise TypeError(
            "'data' must be a Tensor or CompositeTensor. "
            f"Received: {type(data)}.")
      tensors = nest.flatten(data, expand_composites=True)
      mapped = [gen_control_flow_ops.switch(tensor, pred) for tensor in tensors]
      mapped_f, mapped_t = zip(*mapped)
      return (nest.pack_sequence_as(data, mapped_f, expand_composites=True),
              nest.pack_sequence_as(data, mapped_t, expand_composites=True))


def _SwitchRefOrTensor(data, pred, name="Switch"):
  # [FUNCTION DOC]
  # HELPER: internal switch wrapper used by cond and while loop construction.
  # Like the public switch but picks the reference-switch op for reference-typed tensors. Inputs:
  # data (tensor or composite tensor), pred (scalar boolean), name (op name, default Switch).
  # Returns the (false output, true output) pair. The ops are created with colocation constraints
  # ignoring any existing colocation stack, so the switch is placed with the data and not with a
  # variable being updated in an enclosing colocation scope. Raises a type error via switch for
  # unsupported input types.
  """Forwards `data` to an output determined by `pred`.

  If `pred` is false, the `data` input is forwarded to the first output.
  Otherwise, the data goes to the second output.

  This op handles `Tensor`s and `IndexedSlices`.

  Args:
    data: The tensor to be forwarded to the appropriate output.
    pred: A scalar that specifies which output port will receive data.
    name: A name for this operation (optional).

  Returns:
    `(output_false, output_true)`: If `pred` is true, data will be forwarded to
    `output_true`, otherwise it goes to `output_false`.

  Raises:
    TypeError: if data is not a Tensor or IndexedSlices
  """
  data = ops.convert_to_tensor_or_composite(data, name="data")
  # NOTE(vrv): ops.colocate_with(data, ignore_existing=True) below
  # addresses the following scenario.
  #
  # Assume you execute Optimizer.apply_gradients() in a branch of a cond().
  #
  # 1. The update op is created inside a `with ops.colocate(var):` block
  #
  # 2. Some tensor `data` is captured and a switch is created in a
  #    `with ops.colocate_with(data):` block.
  #
  # with ops.colocate_with(var):
  #  with ops.colocate_with(data):
  #    op = ...
  #
  # var and data may be pinned to different devices, so we want to ops
  # created within ops.colocate_with(data) to ignore the existing stack.
  with ops.colocate_with(data, ignore_existing=True):
    if isinstance(data, tensor_lib.Tensor):
      if data.dtype._is_ref_dtype:  # pylint: disable=protected-access
        return ref_switch(data, pred, name=name)
    return switch(data, pred, name=name)


def merge(inputs, name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says value errors come from indexed slices lacking a dense shape,
  # but this function does not check that itself and also raises type errors that are undocumented.
  # Returns the one available element among the inputs together with its index. Inputs: inputs (a
  # list of tensors or composite tensors, at most one of which is available at run time) and an
  # optional op name. Returns a pair (chosen value, index). Mixed tensors and indexed slices are
  # all converted to indexed slices; all-reference inputs use the reference merge op; other
  # composites are merged component by component and repacked, requiring identical structures.
  # Raises a value error if any input is nothing/null, a type error if an input is not a tensor or
  # composite tensor, and a structure-mismatch error from the structure check.
  """Returns the value of an available element of `inputs`.

  This op tests each of the tensors in `inputs` in turn to determine if any of
  them is available. If it finds an available tensor, it returns it and its
  index in `inputs`.

  It is an error if more than one tensor in `inputs` is available. If no tensor
  in `inputs` is available, the returned tensor and index are not set.

  This op handles both `Tensor`s and `IndexedSlices`. If inputs has a mix of
  `Tensor`s and `IndexedSlices`, all inputs are converted to IndexedSlices
  before merging.

  Args:
    inputs: The input tensors, at most one of which is available.
    name: A name for this operation (optional).

  Returns:
    A tuple containing the chosen input tensor and its index in `inputs`.

  Raises:
    ValueError: If any of the inputs is None, or inputs are IndexedSlices and
      some but not all have a dense_shape property.
  """
  if any(inp is None for inp in inputs):
    raise ValueError("At least one of the merge inputs is None: %s" % inputs)
  with ops.name_scope(name, "Merge", inputs) as name:
    inputs = [
        ops.internal_convert_to_tensor_or_composite(inp, as_ref=True)
        for inp in inputs
    ]
    if all(isinstance(v, tensor_lib.Tensor) for v in inputs):
      if all(v.dtype._is_ref_dtype for v in inputs):  # pylint: disable=protected-access
        return gen_control_flow_ops.ref_merge(inputs, name)
      else:
        return gen_control_flow_ops.merge(inputs, name)
    else:
      # If there is a mix of tensors and indexed slices, then convert the
      # tensors to indexed slices.
      if all(
          isinstance(v, (indexed_slices.IndexedSlices, tensor_lib.Tensor))
          for v in inputs):
        inputs = math_ops._as_indexed_slices_list(inputs, optimize=False)

      for v in inputs:
        if not isinstance(v, composite_tensor.CompositeTensor):
          raise TypeError("Type %s not supported" % type(v))

      for v in inputs[1:]:
        nest.assert_same_structure(inputs[0], v, expand_composites=True)

      flat_inputs = [nest.flatten(v, expand_composites=True) for v in inputs]
      merged_results = [
          gen_control_flow_ops.merge(component)
          for component in zip(*flat_inputs)
      ]
      flat_merged = [tensor for (tensor, _) in merged_results]
      chosen_index = merged_results[0][1]
      merged_inputs = nest.pack_sequence_as(
          inputs[0], flat_merged, expand_composites=True)
      return (merged_inputs, chosen_index)


def _convert_tensorarray_to_flow(tensor_or_tensor_array):
  # [FUNCTION DOC]
  # HELPER: lets loop machinery treat tensor arrays as plain tensors. If the argument is a tensor
  # array, returns its flow scalar tensor; otherwise returns the argument unchanged.
  if isinstance(tensor_or_tensor_array, tensor_array_ops.TensorArray):
    return tensor_or_tensor_array.flow
  else:
    return tensor_or_tensor_array


def _convert_flow_to_tensorarray(tensor_or_tensor_array, tensor_or_flow):
  # [FUNCTION DOC]
  # HELPER: inverse of the tensor-array-to-flow conversion. If the first argument is a tensor array,
  # returns a new tensor array built from it with the given flow tensor; otherwise returns the second
  # argument (the flow/tensor) unchanged.
  if isinstance(tensor_or_tensor_array, tensor_array_ops.TensorArray):
    return tensor_array_ops.build_ta_with_new_flow(tensor_or_tensor_array,
                                                   tensor_or_flow)
  else:
    return tensor_or_flow


def _convert_to_tensor_or_composite_or_tensorarray(var):
  # [FUNCTION DOC]
  # HELPER: converts a loop variable to a tensor form. Tensor arrays are returned unchanged; any
  # other value is converted to a tensor or composite tensor (conversion errors propagate).
  if isinstance(var, tensor_array_ops.TensorArray):
    return var
  return ops.convert_to_tensor_or_composite(var)


# TODO(xjun): replace this with is_subtype_of after it is landed.
def _ShapeLessThanOrEqual(shape1, shape2):
  # [FUNCTION DOC]
  # HELPER: shape comparison for loop shape invariants. Returns true if shape1 is at least as
  # specific as shape2: always true when shape2 has unknown rank; false if ranks differ; otherwise
  # every dimension known in shape2 must equal the matching dimension in shape1. Inputs are two
  # tensor shapes; returns a boolean; no side effects.
  if shape2.dims is None:
    return True
  if shape1.ndims != shape2.ndims:
    return False
  for dim1, dim2 in zip(shape1.dims, shape2.dims):
    if dim2.value is not None and dim1.value != dim2.value:
      return False
  return True


def _shape_invariant_to_type_spec(var, shape=None):
  # [FUNCTION DOC]
  # HELPER: normalizes a user supplied shape invariant for loop variables into a type spec.
  # Inputs: var (tensor, tensor array or composite tensor; tensor arrays are first replaced by their
  # flow) and shape (nothing/null, a type spec, or a tensor shape). With no shape, returns the type
  # spec derived from var; a type spec is returned as is after checking compatibility with var; a
  # tensor shape yields a tensor spec for plain tensors, or is delegated to the composite tensor's
  # own method. Raises a type error for an incompatible type spec, an unsupported shape type, or a
  # composite tensor that cannot be described by a tensor shape.
  """Converts a shape invariant to a TypeSpec.

  If `var` is a TensorArray, it will first be converted to its flow.

  Args:
    var: The tensor, tensor array or composite tensor whose shape is described
      by the shape invariant.
    shape: A `TypeSpec` or `TensorShape`.  If `shape` is already a `TypeSpec`,
      then it is simply returned as-is.

  Returns:
    A `TypeSpec` for `var`, consistent with the given shape.

  Raises:
    TypeError: If `shape` is a TypeSpec and not compatible with `var`.
    TypeError: If `shape` is not None, a TypeSpec, or a TensorShape.
    TypeError: If `shape` is a TensorShape, `var` is a CompositeTensor, and
      `var` doesn't implement the `_shape_invariant_to_type_spec` method.
  """
  var = _convert_tensorarray_to_flow(var)
  if shape is None:
    return type_spec.type_spec_from_value(var)
  elif isinstance(shape, type_spec.TypeSpec):
    if not shape.is_compatible_with(var):
      raise TypeError("TypeSpec %r is not compatible with %r" % (shape, var))
    return shape
  elif not isinstance(shape, tensor_shape.TensorShape):
    raise TypeError(
        "'shape' must be one of TypeSpec, TensorShape or None. "
        f"Received: {type(shape)}")

  if isinstance(var, tensor_lib.Tensor):
    return tensor_lib.TensorSpec(shape, var.dtype)
  else:
    try:
      return var._shape_invariant_to_type_spec(shape)  # pylint: disable=protected-access
    except NotImplementedError as e:
      raise TypeError(
          f"To describe or constrain a {type(var).__name__}, use a "
          f"{type(var._type_spec).__name__} instead of a TensorShape.") from e  # pylint: disable=protected-access


def _EnforceShapeInvariant(merge_var, next_var):
  # [FUNCTION DOC]
  # HELPER: validates loop variable shape invariants during while loop construction.
  # Inputs: merge_var (the merge output holding the initial loop value) and next_var (the value after
  # one iteration). Compares static shapes and raises a value error with advice about the shape
  # invariants argument when the new shape is less specific than the entering shape. Raises a type
  # error if merge_var is not a plain tensor (this is not documented in the docstring). Asserts that
  # the merge input comes from a loop enter op. Returns nothing.
  """Check if the shapes of the loops variables are invariants.

  Args:
    merge_var: The tensor representing the initial values of the loop
      variables.
    next_var: The tensor representing the values of the loop variables
      after one loop iteration.

  Raises:
    ValueError: If any tensor in `merge_var` has a more specific shape than
      its corresponding tensor in `next_var`.
  """
  if isinstance(merge_var, tensor_lib.Tensor):
    m_shape = merge_var.get_shape()
    n_shape = next_var.get_shape()
    if not _ShapeLessThanOrEqual(n_shape, m_shape):
      enter = merge_var.op.inputs[0].op
      assert util.IsLoopEnter(enter)
      input_t = enter.inputs[0]
      raise ValueError(
          "Input tensor '%s' enters the loop with shape %s, but has shape %s "
          "after one iteration. To allow the shape to vary across iterations, "
          "use the `shape_invariants` argument of tf.while_loop to specify a "
          "less-specific shape." % (input_t.name, input_t.shape, n_shape))
  else:
    raise TypeError("'merge_var' must be a Tensor. "
                    f"Received: {type(merge_var)}.")


def _AddNextAndBackEdge(m, v, enforce_shape_invariant=True):
  # [FUNCTION DOC]
  # HELPER: closes a while loop by connecting the loop body result back to the merge node.
  # KNOWN ISSUE: for composite tensors it returns the result of mapping an update callback that
  # returns nothing, so the return value is a structure of null values rather than the new value,
  # inconsistent with the plain tensor case. Raises a type error for other input types.
  # Inputs: m (merge output, tensor or composite tensor), v (new value), enforce_shape_invariant
  # (check shapes first, default true; applies to plain tensors only). Wraps v in a next-iteration op
  # and rewires the second input of the merge op(s) to it. Returns the next-iteration value for plain
  # tensors.
  """Add NextIteration and back edge from v to m."""
  if isinstance(m, tensor_lib.Tensor):
    v = ops.convert_to_tensor(v)
    v = _NextIteration(v)
    if enforce_shape_invariant:
      # Make sure the shapes of loop outputs are correct. We do this before
      # calling _update_input, which will raise a less-helpful error message if
      # the types don't match.
      # TODO(skyewm): call this for other cases below (needs testing)
      _EnforceShapeInvariant(m, v)
    m.op._update_input(1, v)  # pylint: disable=protected-access
  elif isinstance(m, composite_tensor.CompositeTensor):
    # pylint: disable=protected-access
    def update_component(m_component, v_component):
      # [FUNCTION DOC]
      # HELPER: per-component callback. Rewires the second input of the merge op of the given
      # component to the matching new component. Returns nothing (mutates the graph).
      m_component.op._update_input(1, v_component)

    if isinstance(m, indexed_slices.IndexedSlices):
      v = math_ops._as_indexed_slices(v, optimize=False)
    # pylint: enable=protected-access
    v = _NextIteration(v)
    return nest.map_structure(update_component, m, v, expand_composites=True)
  else:
    raise TypeError("'m' must be a Tensor or CompositeTensor. "
                    f"Received: {type(m)}.")
  return v


class ControlFlowContext(metaclass=abc.ABCMeta):
  # [CLASS DOC]
  # Abstract base class for control flow contexts (conditional, while loop, XLA), used during graph
  # construction to record which values and ops belong to a scope. State: the list of nested
  # contexts, the enclosing (outer) context captured from the default graph at creation (and this
  # context registers itself as nested in it), a stack of previously active contexts used by enter
  # and exit, the set of tensor names seen in the context, and a mapping from external tensor names
  # to the tensors this context uses in their place. Subclasses must supply the name, gradient
  # state, back-prop flag and serialization; the base class provides entering/exiting the context
  # on the default graph, protocol-buffer value (de)serialization and default no-op hooks.
  """The base class for control flow context.

  The usage pattern is a sequence of (Enter, Exit) followed by a final
  ExitResult.

  We maintain the following state for control flow contexts during graph
  construction:
   1. graph has _control_flow_context: the current context used to
      construct new nodes. Changed by ctxt.Enter() and ctxt.Exit()
   2. op has _control_flow_context: the context to which the op belongs.
      Set at the time the op is created. Immutable.
   3. A ControlFlowContext has _outer_context: the context in which this
      context is created. Set at the time a context is created. Immutable.
   4. A ControlFlowContext has _context_stack.
      Pushed and popped by ctxt.Enter() and ctxt.Exit()
  """

  def __init__(self, values_def=None, import_scope=None):
    # [FUNCTION DOC]
    # Creates the base context state. Inputs: values_def (optional serialized values definition) and
    # import_scope (optional name scope prefix, used only when restoring from values_def). Reads the
    # current context of the default graph as the outer context and, if one exists, appends this
    # context to its nested contexts (side effect on the outer context). With values_def the seen and
    # external value sets are rebuilt from it; otherwise they start empty.
    self._nested_contexts = []
    self._outer_context = ops.get_default_graph()._get_control_flow_context()
    if self._outer_context:
      self._outer_context._nested_contexts.append(self)  # pylint: disable=protected-access
    self._context_stack = []
    if values_def:
      self._init_values_from_proto(values_def, import_scope=import_scope)
    else:
      # The names of tensors that have been already seen in this context.
      self._values = set()
      # The keys are the names of tensors referenced by but external to this
      # context. Each value is the Tensor that should be used by this context to
      # access the key value (e.g. a switch output guarding a cond input value).
      self._external_values = {}

  def _init_values_from_proto(self, values_def, import_scope=None):
    # [FUNCTION DOC]
    # HELPER: deserialization step for the base context. Restores the set of seen tensor names and
    # the mapping of external tensors from a values definition, prepending the import scope to every
    # name and resolving them in the default graph. Also sets this context as the control flow
    # context of every internal op named in the values. Asserts the argument has the right message
    # type. Returns nothing.
    """Initializes values and external_values from `ValuesDef` protocol buffer.

    Args:
      values_def: `ValuesDef` protocol buffer.
      import_scope: Optional `string`. Name scope to add.
    """
    assert isinstance(values_def, control_flow_pb2.ValuesDef)
    self._values = set(
        ops.prepend_name_scope(value, import_scope)
        for value in values_def.values)
    g = ops.get_default_graph()
    self._external_values = {}
    for k, v in values_def.external_values.items():
      k = ops.prepend_name_scope(k, import_scope)
      self._external_values[k] = g.as_graph_element(
          ops.prepend_name_scope(v, import_scope))
    op_names = set([
        op.split(":")[0]
        for op in self._values - set(self._external_values.keys())
    ])
    for op in op_names:
      # pylint: disable=protected-access
      g.as_graph_element(op)._set_control_flow_context(self)
      # pylint: enable=protected-access

  @property
  def name(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the base class never sets the name attribute; it relies on subclasses to assign it
    # in their constructors, so accessing it on a bare base instance raises an attribute error.
    # Property returning the unique name of this context.
    return self._name

  @property
  def outer_context(self):
    # [FUNCTION DOC]
    # Property returning the context that was active in the graph when this one was created, or
    # nothing/null if this is outermost.
    """Return the context containing this context."""
    return self._outer_context

  @property
  def grad_state(self):
    # [FUNCTION DOC]
    # Abstract property: the gradient loop state. Always raises a not-implemented error here;
    # subclasses override it.
    raise NotImplementedError("Abstract method")

  @property
  def back_prop(self):
    # [FUNCTION DOC]
    # Abstract property: whether backpropagation is enabled. Always raises a not-implemented error
    # here; subclasses override it.
    raise NotImplementedError("Abstract method")

  @abc.abstractmethod
  def to_control_flow_context_def(self, context_def, export_scope=None):
    # [FUNCTION DOC]
    # Abstract method: serializes this context into the given context definition message. Inputs:
    # context_def (message to fill) and export_scope (optional name scope to strip). Raises a
    # not-implemented error in the base class; subclasses override it.
    """Serializes this into `context_def`.

    Args:
      context_def: a `ControlFlowContextDef` protocol buffer.
      export_scope: Optional `string`. Name scope to remove.
    """
    raise NotImplementedError("Abstract method")

  def _to_values_def(self, export_scope=None):
    # [FUNCTION DOC]
    # HELPER: serialization helper for subclasses. Builds a values definition message holding the
    # sorted names of seen tensors and the external-value mapping, stripping the export scope from
    # all names. Input: export_scope (optional). Returns the message.
    """Converts the values to a `ValuesDef` protocol buffer.

    Args:
      export_scope: Optional `string`. Name scope to remove.

    Returns:
      A `ValuesDef` protocol buffer.
    """
    values_def = control_flow_pb2.ValuesDef()
    values_def.values.extend(
        [ops.strip_name_scope(v, export_scope) for v in sorted(self._values)])
    for k, v in self._external_values.items():
      k = ops.strip_name_scope(k, export_scope)
      values_def.external_values[k] = ops.strip_name_scope(v.name, export_scope)
    return values_def

  def AddName(self, name):
    # [FUNCTION DOC]
    # Records a tensor name as seen in this context (adds it to the set of values). Returns nothing.
    self._values.add(name)

  # pylint: disable=protected-access
  def Enter(self):
    # [FUNCTION DOC]
    # Makes this the active control flow context of the default graph, pushing the previously active
    # context onto this context's stack. Side effect on the graph; returns nothing.
    """Enter this control flow context."""
    graph = ops.get_default_graph()
    self._context_stack.append(graph._get_control_flow_context())
    graph._set_control_flow_context(self)

  def Exit(self):
    # [FUNCTION DOC]
    # Leaves this context: pops the previously saved context from the stack and makes it the active
    # context of the default graph. Must be paired with a prior enter, otherwise popping an empty
    # stack raises an error. Returns nothing.
    """Exit this control flow context."""
    graph = ops.get_default_graph()
    last_context = self._context_stack.pop()
    graph._set_control_flow_context(last_context)

  def EnterGradientColocation(self, op: ops.Operation, gradient_uid):
    # [FUNCTION DOC]
    # Hook called when gradient construction starts colocated with an op. Inputs: op and the
    # gradient unique id. The base behavior forwards the call to the outer context if there is one;
    # otherwise does nothing. Subclasses may override. Returns nothing.
    """Start building a gradient colocated with an op."""
    if self._outer_context:
      self._outer_context.EnterGradientColocation(op, gradient_uid)

  def ExitGradientColocation(self, op: ops.Operation, gradient_uid):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring is copied from the enter variant and says "Start building" although
    # this method is the exit counterpart.
    # Hook called when gradient construction colocated with an op ends. Inputs: op and the gradient
    # unique id. Forwards to the outer context if any; otherwise does nothing. Returns nothing.
    """Start building a gradient colocated with an op."""
    if self._outer_context:
      self._outer_context.ExitGradientColocation(op, gradient_uid)

  def ExitResult(self, result):
    # [FUNCTION DOC]
    # Makes result tensors known to the outer context. Input: result, a (possibly nested) structure
    # of tensors. If an outer context exists, every tensor name is added to its seen-values set.
    # Returns nothing.
    """Make a list of tensors available in the outer context."""
    if self._outer_context:
      def fn(x):
        # [FUNCTION DOC]
        # HELPER: callback applied to each tensor; registers its name in the outer context and
        # returns the tensor unchanged.
        self._outer_context.AddName(x.name)
        return x
      nest.map_structure(fn, result, expand_composites=True)

  def GetWhileContext(self):
    # [FUNCTION DOC]
    # Returns the nearest enclosing while-loop context (searching outward through outer contexts),
    # or nothing/null if there is none.
    """Return the while context containing this context."""
    if self._outer_context:
      return self._outer_context.GetWhileContext()
    return None

  def _RemoveExternalControlEdges(self, op: ops.Operation):
    # [FUNCTION DOC]
    # HELPER: control dependency cleanup used when adding ops to a context. Input: op. If there is no\
    """Remove any external control dependency on this op."""
    while_ctxt = self.GetWhileContext()
    # A control input of `op` is internal if it is in the same while
    # loop context as the enclosing while loop context of self.
    if while_ctxt is None:
      internal_control_inputs, external_control_inputs = op.control_inputs, []
    else:
      internal_control_inputs, external_control_inputs = [], []
      for x in op.control_inputs:
        ctxt = util.GetOutputContext(x)
        if ctxt is not None and ctxt.GetWhileContext() == while_ctxt:
          internal_control_inputs.append(x)
        else:
          external_control_inputs.append(x)
    if len(internal_control_inputs) != len(op.control_inputs):
      # TODO(mdan): perhaps there should be a replace_control_inputs()
      op._remove_all_control_inputs()
      op._add_control_inputs(internal_control_inputs)
    return internal_control_inputs, external_control_inputs

  # pylint: enable=protected-access

  def AddInnerOp(self, op: ops.Operation):
    # [FUNCTION DOC]
    # Notifies this context that an op was added in a nested scope; the base behavior propagates the
    # notification outward to the outer context if present. Input: op. Returns nothing.
    """Notifies a scope about an operator added to an inner scope."""
    if self._outer_context:
      self._outer_context.AddInnerOp(op)

  def GetControlPivot(self):
    # [FUNCTION DOC]
    # Returns the pivot op/tensor used to gate ops in this context; the base class has none and
    # returns nothing/null.
    """Returns the pivot node for this context, or None."""
    return None

  def IsWhileContext(self):
    # [FUNCTION DOC]
    # Returns false; overridden by while contexts to return true.
    return False

  def IsCondContext(self):
    # [FUNCTION DOC]
    # Returns false; overridden by conditional contexts to return true.
    return False

  def IsXLAContext(self):
    # [FUNCTION DOC]
    # Returns false; overridden by XLA contexts to return true.
    return False

  def __str__(self):
    # [FUNCTION DOC]
    # String form of the context: simply its name.
    return self.name


class CondContext(ControlFlowContext):
  # [CLASS DOC]
  # Control flow context for one branch of a conditional (cond) in a v1 graph. Holds the predicate
  # tensor, the pivot tensor for this branch (the predicate switched to this branch), and the branch
  # index (0 for false, 1 for true). Values from outside the branch that are used inside are routed
  # through switch ops, ops without inputs get a control edge from the pivot, and branch outputs are
  # built through switches. It can be serialized to and restored from a protocol buffer.
  """The context for the conditional construct."""

  def __init__(self,
               pred=None,
               pivot=None,
               branch=None,
               name="cond_text",
               context_def=None,
               import_scope=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: pred and pivot default to nothing/null, but without a context definition both are
    # dereferenced to read names, so omitting them raises an attribute error.
    # Creates a conditional context. Inputs: pred (boolean predicate tensor), pivot (predicate
    # tensor in this branch), branch (0 or 1), name (base name made unique in the default graph,
    # default cond_text), context_def (optional serialized definition to restore from) and
    # import_scope (optional name scope, only used when restoring). Without context_def, registers
    # the predicate as an external value, the pivot as a seen value, and sets this context on the
    # pivot op.
    """Creates a `CondContext`.

    Args:
      pred: The `boolean` tensor for the conditional predicate.
      pivot: The predicate tensor in this branch.
      branch: 0 or 1 representing this branch.
      name: Name of the `CondContext` python object.
      context_def: Optional `ContextDef` protocol buffer to initialize the
        `CondContext` object from.
      import_scope: Optional `string`. Name scope to add. Only used when
        initialing from protocol buffer.
    """
    self._name = ops.get_default_graph().unique_name(name)

    if context_def:
      self._init_from_proto(context_def, import_scope=import_scope)
    else:
      # Initializes the default fields.
      ControlFlowContext.__init__(self)
      self._pred = pred  # The boolean tensor for the cond predicate
      self._pivot = pivot  # The predicate tensor in this branch
      self._branch = branch  # 0 or 1 representing this branch

      # Values considered to have been already seen in this context. pred is not
      # included in this context.
      self._values.add(pred.name)
      self._external_values[pred.name] = pred
      self._values.add(pivot.name)
      pivot.op._set_control_flow_context(self)  # pylint: disable=protected-access

  def _init_from_proto(self, context_def, import_scope=None):
    # [FUNCTION DOC]
    # HELPER: restores state from a conditional context definition. Inputs: context_def and
    # optional import_scope. Sets name, predicate, pivot and branch by resolving the named graph
    # elements (with scope prefixed), then runs the base initialization with the saved values.
    # Asserts the message type. Returns nothing.
    """Creates a new `CondContext` from protocol buffer.

    Args:
      context_def: `CondContextDef` protocol buffer.
      import_scope: Optional `string`. Name scope to add.
    """
    assert isinstance(context_def, control_flow_pb2.CondContextDef)
    # Create from context_def.
    g = ops.get_default_graph()
    self._name = ops.prepend_name_scope(context_def.context_name, import_scope)
    self._pred = g.as_graph_element(
        ops.prepend_name_scope(context_def.pred_name, import_scope))
    self._pivot = g.as_graph_element(
        ops.prepend_name_scope(context_def.pivot_name, import_scope))
    self._branch = context_def.branch
    super(CondContext, self).__init__(
        values_def=context_def.values_def, import_scope=import_scope)

  @property
  def pred(self):
    # [FUNCTION DOC]
    # Property: the boolean predicate tensor of the conditional.
    return self._pred

  @property
  def pivot(self):
    # [FUNCTION DOC]
    # Property: the predicate tensor switched to this branch, used as the control pivot.
    return self._pivot

  @property
  def branch(self):
    # [FUNCTION DOC]
    # Property: which branch this context represents (0 for false, 1 for true).
    return self._branch

  @property
  def grad_state(self):
    # [FUNCTION DOC]
    # Property: the gradient loop state of the enclosing while context, or nothing/null if this
    # conditional is not inside a while loop.
    if self.GetWhileContext():
      return self.GetWhileContext().grad_state
    return None

  @property
  def back_prop(self):
    # [FUNCTION DOC]
    # Property: the back-prop setting of the enclosing while context, or false when not inside one.
    if self.GetWhileContext():
      return self.GetWhileContext().back_prop
    return False

  def GetControlPivot(self):
    # [FUNCTION DOC]
    # Returns the pivot tensor for this branch (overrides the base version that returns nothing).
    return self._pivot

  def to_proto(self, export_scope=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: nested contexts are serialized without passing the export scope along, so nested
    # names are not stripped.
    # Serializes this context into a conditional context definition message. Input: export_scope
    # (optional; only contexts whose name starts with it are exported, and the scope is stripped from
    # names). Records name, predicate, pivot, branch, values and nested contexts. Returns the
    # message, or nothing/null if the name does not match the export scope.
    """Converts a `CondContext` to a `CondContextDef` protocol buffer.

    Args:
      export_scope: Optional `string`. Name scope to remove.

    Returns:
      A `CondContextDef` protocol buffer.
    """
    if (export_scope is None or self.name.startswith(export_scope)):
      context_def = control_flow_pb2.CondContextDef()
      context_def.context_name = ops.strip_name_scope(self.name, export_scope)
      context_def.pred_name = ops.strip_name_scope(self._pred.name,
                                                   export_scope)
      context_def.pivot_name = ops.strip_name_scope(self._pivot.name,
                                                    export_scope)
      context_def.branch = self._branch
      context_def.values_def.MergeFrom(
          super(CondContext, self)._to_values_def(export_scope))
      for nested in self._nested_contexts:
        nested_def = context_def.nested_contexts.add()
        nested.to_control_flow_context_def(nested_def)

      return context_def
    else:
      return None

  @staticmethod
  def from_proto(context_def, import_scope=None):
    # [FUNCTION DOC]
    # Static factory. Inputs: context_def (conditional context definition) and optional
    # import_scope. Builds the context, enters it while recreating nested contexts from their
    # definitions, exits it, and returns the new context.
    """Returns a `CondContext` object created from `context_def`."""
    ret = CondContext(context_def=context_def, import_scope=import_scope)

    ret.Enter()
    for nested_def in context_def.nested_contexts:
      from_control_flow_context_def(nested_def, import_scope=import_scope)
    ret.Exit()
    return ret

  def to_control_flow_context_def(self, context_def, export_scope=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: if serialization returns nothing/null (name outside export scope), copying that
    # into the message fails instead of being skipped.
    # Fills the conditional-context field of the given generic context definition with this context
    # serialized (export_scope optional). Returns nothing.
    context_def.cond_ctxt.CopyFrom(self.to_proto(export_scope=export_scope))

  def AddValue(self, val):
    # [FUNCTION DOC]
    # Makes an outside tensor usable in this branch. Input: val (tensor). If already seen, returns
    # its recorded external replacement (or val itself). Otherwise records it, recursively asks the
    # outer context, then creates a switch on the predicate (without inheriting control
    # dependencies) and takes the output for this branch, marks the switch op as non-fetchable and
    # as belonging to this context, adds names to all enclosing contexts and remembers the mapping.
    # Returns the tensor to use inside the branch.
    """Add `val` to the current context and its outer context recursively."""
    if val.name in self._values:
      # Use the real value if it comes from outer context. This is needed in
      # particular for nested conds.
      result = self._external_values.get(val.name)
      result = val if result is None else result
    else:
      result = val
      self._values.add(val.name)
      if self._outer_context:
        result = self._outer_context.AddValue(val)
        self._values.add(result.name)
        self._external_values[result.name] = result
      with ops.control_dependencies(None):
        result = _SwitchRefOrTensor(result, self._pred)[self._branch]
        if self._outer_context:
          self._outer_context.AddInnerOp(result.op)

      result.op.graph.prevent_fetching(result.op)
      # pylint: disable=protected-access
      result.op._set_control_flow_context(self)
      # pylint: enable=protected-access

      # Mark Switch output as seen by this context and any outer contexts,
      # just like what we do for normal op outputs in _AddOpInternal() below.
      ctxt = self
      while ctxt is not None:
        # pylint: disable=protected-access
        ctxt._values.add(result.name)
        ctxt = ctxt._outer_context
        # pylint: enable=protected-access

      self._external_values[val.name] = result
    return result

  def AddOp(self, op: ops.Operation):
    # [FUNCTION DOC]
    # Registers an op in this context; simply delegates to the internal adder. Input: op.
    self._AddOpInternal(op)

  def _AddOpInternal(self, op: ops.Operation):
    # [FUNCTION DOC]
    # HELPER: core of adding an op to a conditional branch. Input: op. Ops without inputs lose
    # external control edges and, unless a control input is already in this context, get a control
    # edge from the pivot. For ops with inputs, each input is routed through the value-adding logic
    # (with an exception for merge inputs coming from next-iteration ops) and rewired if replaced;
    # function-call and symbolic-gradient ops also get a control edge from the pivot. Marks output
    # names as seen here and in all outer contexts, prevents fetching of the op (except an outermost
    # loop exit) and notifies the outer context. Mutates the op and graph; returns nothing.
    """Add `op` to the current context."""
    if not op.inputs:
      # If we're in a while loop, remove any control inputs from outside the
      # loop.
      self._RemoveExternalControlEdges(op)

      if not any(
          util.OpInContext(input_op, self) for input_op in op.control_inputs):
        # pylint: disable=protected-access
        op._add_control_input(self._pivot.op)
        # pylint: enable=protected-access
    else:
      # Make each input to 'op' available in this CondContext. If an input is
      # already part of this context there's nothing to do, but if it's
      # external, AddValue() will handle adding the appropriate Switch node and
      # other bookkeeping.
      for index in range(len(op.inputs)):
        x = op.inputs[index]
        if op.type == "Merge" and x.op.type == "NextIteration":
          # Edge case: if we're importing a while loop inside this CondContext,
          # AddValue() will not correctly handle the NextIteration inputs to
          # Merge node. The problem is that the NextIteration should also be
          # part of this context, but if we're importing it won't have been
          # processed and added to the context yet, so AddValue() will try to
          # add a Switch which results in an invalid graph. Instead, we use the
          # NextIteration input as-is here, and it will eventually be added to
          # the context via AddOp().
          real_x = x
        else:
          real_x = self.AddValue(x)
        if real_x != x:
          # pylint: disable=protected-access
          op._update_input(index, real_x)
          # pylint: enable=protected-access
      # Remove any external control dependency on this op.
      self._RemoveExternalControlEdges(op)
      # pylint: disable=protected-access
      if op.graph._is_function(op.type) or op.type == "SymbolicGradient":
        op._add_control_input(self._pivot.op)
      # pylint: enable=protected-access

    # Mark op's outputs as seen by this context and any outer contexts.
    output_names = [x.name for x in op.outputs]
    ctxt = self
    while ctxt is not None:
      # pylint: disable=protected-access
      ctxt._values.update(output_names)
      ctxt = ctxt._outer_context
      # pylint: enable=protected-access

    if self._outer_context or not util.IsLoopExit(op):
      op.graph.prevent_fetching(op)

    if self._outer_context:
      self._outer_context.AddInnerOp(op)

  def _ProcessOutputTensor(self, val):
    # [FUNCTION DOC]
    # HELPER: handles a tensor returned from a branch function. Input: val. If val was not created
    # in this context (for example a function returning its argument), it is registered, routed
    # through the outer context and then through a switch for this branch. Otherwise returns its
    # external replacement if one exists. Returns the tensor to use as the branch output.
    """Process an output tensor of a conditional branch."""
    real_val = val
    if val.name not in self._values:
      # Handle the special case of lambda: x
      self._values.add(val.name)
      if self._outer_context:
        real_val = self._outer_context.AddValue(val)
        self._values.add(real_val.name)
        self._external_values[real_val.name] = real_val
      real_val = _SwitchRefOrTensor(real_val, self._pred)[self._branch]
      self._external_values[val.name] = real_val
    else:
      external_val = self._external_values.get(val.name)
      if external_val is not None:
        real_val = external_val
    return real_val

  def _BuildCondTensor(self, v):
    # [FUNCTION DOC]
    # HELPER: converts one branch result into an output tensor. Input: v, an operation or a value.
    # Operations are represented by the pivot with a control dependency on the op. Other values have
    # tensor arrays replaced by their flow, are converted to a tensor and processed as branch
    # outputs. Returns the resulting tensor.
    if isinstance(v, ops.Operation):
      # Use pivot as the proxy for this op.
      return with_dependencies([v], self._pivot)
    else:
      v = nest.map_structure(
          _convert_tensorarray_to_flow, v, expand_composites=True)
      return self._ProcessOutputTensor(ops.convert_to_tensor(v))

  def BuildCondBranch(self, fn):
    # [FUNCTION DOC]
    # Builds the subgraph for this branch. Input: fn, a callable with no arguments returning
    # tensors, structures of tensors, an operation, or nothing. Summary ops created by fn are removed
    # from the summary collection and attached as control dependencies of the result (a bare no-op
    # when fn returned nothing). Variables are converted to tensors and every leaf is converted via
    # the branch tensor builder. Returns a pair: the original result and the list of branch output
    # tensors (the result is wrapped in a list if it is not a list or tuple); both are nothing/null
    # when fn returns nothing and no summaries were added.
    """Add the subgraph defined by fn() to the graph."""
    pre_summaries = ops.get_collection(ops.GraphKeys._SUMMARY_COLLECTION)  # pylint: disable=protected-access
    original_result = fn()
    post_summaries = ops.get_collection(ops.GraphKeys._SUMMARY_COLLECTION)  # pylint: disable=protected-access
    if len(post_summaries) > len(pre_summaries):
      new_summaries = post_summaries[len(pre_summaries):]
      summary_ref = ops.get_collection_ref(ops.GraphKeys._SUMMARY_COLLECTION)  # pylint: disable=protected-access
      summary_ref[:] = pre_summaries
      with ops.control_dependencies(new_summaries):
        if original_result is None:
          return no_op(), None
        elif not isinstance(original_result, ops.Operation):
          original_result = variable_utils.convert_variables_to_tensors(
              original_result)
          original_result = nest.map_structure(
              array_ops.identity, original_result, expand_composites=True)
    if original_result is None:
      return None, None

    original_result = variable_utils.convert_variables_to_tensors(
        original_result)
    result = nest.map_structure(
        self._BuildCondTensor, original_result, expand_composites=True)
    if not isinstance(result, (list, _basetuple)):
      result = [result]
    return original_result, result

  def IsCondContext(self):
    # [FUNCTION DOC]
    # Returns true; marks this as a conditional context.
    return True


# pylint: enable=g-doc-args
# pylint: enable=redefined-outer-name


def _resource_safe_shape(t):
  # [FUNCTION DOC]
  # HELPER: shape lookup for loop accumulators. Input: t, a tensor. For resource tensors, follows
  # the first-input chain back to the originating op and returns the static shape stored in its
  # shape attribute (the shape of the variable pointed to); for other tensors returns a dynamic
  # shape tensor created without optimization.
  """Returns the shape of t or the variable it points to."""
  if t.dtype == dtypes.resource:
    while t.op.inputs:
      t = t.op.inputs[0]
    return tensor_shape.TensorShape(t.op.get_attr("shape"))
  return array_ops.shape_internal(t, optimize=False)


# TODO(yuanbyu): Consider having a unified notion of context for
# not only conditionals and loops but also control dependency and
# subgraphs.
class WhileContext(ControlFlowContext):
  # [CLASS DOC]
  # Control flow context for a v1 while loop. Holds the maximum iterations, parallel iterations,
  # back-prop and swap-memory settings, the gradient loop state, the pivot (loop termination
  # condition), separate pivots for the predicate and body functions (used to control constants
  # created there), and lists of enter and exit tensors for loop variables. Provides building of the
  # loop (enter, merge, switch, next-iteration, exit), loop counters and gradient accumulators,
  # and protocol buffer serialization.
  """The context for the loop construct."""

  def __init__(self,
               maximum_iterations=None,
               parallel_iterations=10,
               back_prop=True,
               swap_memory=False,
               name="while_context",
               grad_state=None,
               context_def=None,
               import_scope=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring begins with four quote characters, so its text starts with a stray
    # quote mark.
    # Creates a while context. Inputs: maximum_iterations (optional loop bound tensor),
    # parallel_iterations (positive integer, default 10), back_prop (default true), swap_memory
    # (default false), name (base name, default while_context), grad_state (gradient loop state),
    # context_def (optional serialized definition to restore from) and import_scope. Initializes
    # from either the definition or the arguments and stores the gradient state.
    """"Creates a `WhileContext`.

    Args:
      maximum_iterations: Optional upper bound on number of loop iterations.
      parallel_iterations: The number of iterations allowed to run in parallel.
      back_prop: Whether backprop is enabled for this while loop.
      swap_memory: Whether GPU-CPU memory swap is enabled for this loop.
      name: Optional name prefix for the returned tensors.
      grad_state: The gradient loop state.
      context_def: Optional `WhileContextDef` protocol buffer to initialize the
        `Whilecontext` python object from.
      import_scope: Optional `string`. Name scope to add. Only used when
        initialing from protocol buffer.
    """
    if context_def:
      self._init_from_proto(context_def, import_scope=import_scope)
    else:
      ControlFlowContext.__init__(self)
      self._init_from_args(maximum_iterations, parallel_iterations, back_prop,
                           swap_memory, name)
    # The gradient loop state.
    self._grad_state = grad_state

  def _init_from_args(self, maximum_iterations, parallel_iterations, back_prop,
                      swap_memory, name):
    # [FUNCTION DOC]
    # HELPER: argument initialization. Inputs as for the constructor (without definition or grad
    # state). Raises a value error if parallel_iterations is not a positive integer. Sets the unique
    # name, settings, empty enter/exit lists, no pivots, and remembers the default graph.
    """Creates a new `WhileContext` from arguments.

    Args:
      maximum_iterations: Optional upper bound on number of loop iterations.
      parallel_iterations: The number of iterations allowed to run in parallel.
      back_prop: Whether backprop is enabled for this while loop.
      swap_memory: Whether GPU-CPU memory swap is enabled for this loop.
      name: Optional name prefix for the returned tensors.

    Raises:
      ValueError: If `parallel_iterations` has invalid value.
    """
    if not isinstance(parallel_iterations, int) or (parallel_iterations <= 0):
      raise ValueError("'parallel_iterations' must be a positive integer: "
                       "%s" % parallel_iterations)
    self._name = ops.get_default_graph().unique_name(name)
    self._maximum_iterations = maximum_iterations
    self._parallel_iterations = parallel_iterations
    self._back_prop = back_prop
    self._swap_memory = swap_memory
    # We use this node to control constants created by the pred lambda.
    self._pivot_for_pred = None
    # We use this node to control constants created by the body lambda.
    self._pivot_for_body = None
    # The boolean tensor for loop termination condition. Used in code
    # generation for gradient computation
    self._pivot = None
    # The list of exit tensors for loop variables.
    self._loop_exits = []
    # The list of enter tensors for loop variables.
    self._loop_enters = []
    self._graph = ops.get_default_graph()

  def _init_from_proto(self, context_def, import_scope=None):
    # [FUNCTION DOC]
    # HELPER: restores a while context from a definition. Inputs: context_def and optional
    # import_scope. Resolves the maximum iterations, pivots, and enter/exit tensors by name in the
    # default graph, runs base initialization, and when an import scope is used rewrites the frame
    # name attribute of every loop-enter op to the new name. Asserts the message type. Returns nothing.
    """Creates a new `WhileContext` from protocol buffer.

    Args:
      context_def: `WhileContextDef` protocol buffer.
      import_scope: Optional `string`. Name scope to add.
    """
    assert isinstance(context_def, control_flow_pb2.WhileContextDef)
    # Create from context_def.
    g = ops.get_default_graph()
    self._name = ops.prepend_name_scope(context_def.context_name, import_scope)
    if context_def.maximum_iterations_name:
      self._maximum_iterations = g.as_graph_element(
          ops.prepend_name_scope(context_def.maximum_iterations_name,
                                 import_scope))
    else:
      self._maximum_iterations = None
    self._parallel_iterations = context_def.parallel_iterations
    self._back_prop = context_def.back_prop
    self._swap_memory = context_def.swap_memory
    self._pivot_for_pred = g.as_graph_element(
        ops.prepend_name_scope(context_def.pivot_for_pred_name, import_scope))
    # We use this node to control constants created by the body lambda.
    self._pivot_for_body = g.as_graph_element(
        ops.prepend_name_scope(context_def.pivot_for_body_name, import_scope))
    # The boolean tensor for loop termination condition. Used in code
    # generation for gradient computation.
    self._pivot = g.as_graph_element(
        ops.prepend_name_scope(context_def.pivot_name, import_scope))
    # The list of exit tensors for loop variables.
    self._loop_exits = [
        g.as_graph_element(ops.prepend_name_scope(exit_name, import_scope))
        for exit_name in context_def.loop_exit_names
    ]
    # The list of enter tensors for loop variables.
    self._loop_enters = [
        g.as_graph_element(ops.prepend_name_scope(enter_name, import_scope))
        for enter_name in context_def.loop_enter_names
    ]
    super(WhileContext, self).__init__(
        values_def=context_def.values_def, import_scope=import_scope)

    # import_scope causes self.name to be different from the original serialized
    # context's name. Rewrite "frame_name" attrs with the new name.
    if import_scope:
      for tensor_name in self._values:
        op = g.as_graph_element(tensor_name).op
        if util.IsLoopEnter(op):
          # pylint: disable=protected-access
          op._set_attr("frame_name",
                       attr_value_pb2.AttrValue(s=compat.as_bytes(self.name)))
          # pylint: enable=protected-access
    self._graph = ops.get_default_graph()

  @property
  def maximum_iterations(self):
    # [FUNCTION DOC]
    # Property: the maximum number of iterations (tensor) or nothing/null if unbounded.
    """The maximum number of iterations that will be executed."""
    return self._maximum_iterations

  @property
  def parallel_iterations(self):
    # [FUNCTION DOC]
    # Property: number of loop iterations allowed to run in parallel.
    """The number of iterations allowed to run in parallel."""
    return self._parallel_iterations

  @property
  def back_prop(self):
    # [FUNCTION DOC]
    # Property: whether backpropagation is enabled for this loop.
    """True iff backprop is enabled for this while loop."""
    return self._back_prop

  @property
  def swap_memory(self):
    # [FUNCTION DOC]
    # Property: whether GPU-to-CPU memory swapping is enabled for this loop.
    """True iff GPU-CPU memory swap is enabled for this while loop."""
    return self._swap_memory

  @property
  def pivot(self):
    # [FUNCTION DOC]
    # Property: boolean tensor representing the loop termination condition (unset until built).
    """The boolean tensor representing the loop termination condition."""
    return self._pivot

  @property
  def loop_enters(self):
    # [FUNCTION DOC]
    # Property: list of enter tensors for the loop variables.
    """The list of enter tensors for loop variables."""
    return self._loop_enters

  @property
  def loop_exits(self):
    # [FUNCTION DOC]
    # Property: list of exit tensors for the loop variables.
    """The list of exit tensors for loop variables."""
    return self._loop_exits

  @property
  def grad_state(self):
    # [FUNCTION DOC]
    # Property: the gradient loop state given at construction (nothing/null for forward loops).
    """The gradient loop state."""
    return self._grad_state

  def to_proto(self, export_scope=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: nested contexts are serialized without the export scope being passed along.
    # Serializes this loop into a while context definition. Input: export_scope (optional; only
    # contexts whose name starts with it are exported, scope stripped from names). Records name,
    # settings, optional maximum iterations, pivots, enter/exit names, values and nested contexts.
    # Returns the message, or nothing/null if the name is outside the export scope.
    """Converts a `WhileContext` to a `WhileContextDef` protocol buffer.

    Args:
      export_scope: Optional `string`. Name scope to remove.

    Returns:
      A `WhileContextDef` protocol buffer.
    """
    if (export_scope is None or self.name.startswith(export_scope)):
      context_def = control_flow_pb2.WhileContextDef()
      context_def.context_name = ops.strip_name_scope(self.name, export_scope)
      context_def.parallel_iterations = self._parallel_iterations
      if self._maximum_iterations is not None:
        context_def.maximum_iterations_name = ops.strip_name_scope(
            self._maximum_iterations.name, export_scope)
      context_def.back_prop = self._back_prop
      context_def.swap_memory = self._swap_memory
      context_def.pivot_for_pred_name = ops.strip_name_scope(
          self._pivot_for_pred.name, export_scope)
      context_def.pivot_for_body_name = ops.strip_name_scope(
          self._pivot_for_body.name, export_scope)
      context_def.pivot_name = ops.strip_name_scope(self._pivot.name,
                                                    export_scope)
      context_def.loop_exit_names.extend([
          ops.strip_name_scope(l.name, export_scope) for l in self._loop_exits
      ])
      context_def.loop_enter_names.extend([
          ops.strip_name_scope(l.name, export_scope) for l in self._loop_enters
      ])
      context_def.values_def.MergeFrom(
          super(WhileContext, self)._to_values_def(export_scope=export_scope))
      for nested in self._nested_contexts:
        nested_def = context_def.nested_contexts.add()
        nested.to_control_flow_context_def(nested_def)

      return context_def
    else:
      return None

  def to_control_flow_context_def(self, context_def, export_scope=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: if serialization yields nothing/null (name outside the export scope) the copy
    # fails instead of being skipped.
    # Fills the while-context field of the given generic context definition with this context
    # serialized (export_scope optional). Returns nothing.
    context_def.while_ctxt.CopyFrom(self.to_proto(export_scope=export_scope))

  @staticmethod
  def from_proto(context_def, import_scope=None):
    # [FUNCTION DOC]
    # Static factory. Inputs: context_def (while context definition), import_scope (optional).
    # Builds the context, enters it while recreating nested contexts, exits it and returns it.
    """Returns a `WhileContext` object created from `context_def`.

    Args:
      context_def: A `WhileContextDef` protocol buffer.
      import_scope: Optional `string`. Name scope to add.

    Returns:
      A `WhileContext` Python object.
    """
    ret = WhileContext(context_def=context_def, import_scope=import_scope)
    ret.Enter()
    for nested_def in context_def.nested_contexts:
      from_control_flow_context_def(nested_def, import_scope=import_scope)
    ret.Exit()
    return ret

  def GetWhileContext(self):
    # [FUNCTION DOC]
    # Returns this context itself (the nearest enclosing while context).
    return self

  def GetControlPivot(self):
    # [FUNCTION DOC]
    # Returns the body pivot if set, otherwise the predicate pivot (which may also be unset).
    if self._pivot_for_body is not None:
      return self._pivot_for_body
    return self._pivot_for_pred

  def AddValue(self, val):
    # [FUNCTION DOC]
    # Makes an outside tensor available inside the loop. Input: val (tensor). If new to this
    # context (and not produced by it), it is recorded; when building a gradient loop and val comes
    # from the forward loop, the forward history value is returned instead. Otherwise the value is
    # first obtained from the outer context, then a constant enter op is created (no inherited
    # control dependencies, feeding disabled), fixed up, recorded as an external value and returned.
    # Already known values return their recorded replacement. Returns the tensor to use in the loop.
    """Add `val` to the current context and its outer context recursively."""
    result = val
    new_value = val.name not in self._values
    # Don't treat ops in this context as new values. Usually all known values
    # are in self._values, except when we're importing a while loop inside this
    # WhileContext. Since there's a cycle in this case, `val` may be part of the
    # imported while loop but not yet processed by this context and added to
    # self._values in _AddOpInternal. We only want to process external input
    # tensors to the while loop here.
    new_value &= val.op._control_flow_context is not self  # pylint: disable=protected-access
    if new_value:
      self._values.add(val.name)

      # If we are in a grad context and val is from its forward context,
      # use GetRealValue(), which adds the logic to save the history of
      # val in forward.
      grad_ctxt = ops.get_default_graph()._get_control_flow_context()
      if grad_ctxt:
        grad_ctxt = grad_ctxt.GetWhileContext()
        if grad_ctxt.grad_state:
          forward_ctxt = util.GetWhileContext(val.op)
          if util.IsLoopExit(val.op):
            forward_ctxt = forward_ctxt.outer_context
            if forward_ctxt:
              forward_ctxt = forward_ctxt.GetWhileContext()
          if forward_ctxt == grad_ctxt.grad_state.forward_context:
            real_val = grad_ctxt.grad_state.GetRealValue(val)
            self._external_values[val.name] = real_val
            return real_val

      if self._outer_context is not None:
        result = self._outer_context.AddValue(val)
      # Create an Enter to make `result` known to this loop context.
      with ops.control_dependencies(None):
        enter = _Enter(
            result,
            self._name,
            is_constant=True,
            parallel_iterations=self._parallel_iterations)
        enter.graph.prevent_feeding(enter)
        if self._outer_context:
          self._outer_context.AddInnerOp(enter.op)
      # Fix the control inputs and control flow context of these enter ops.
      self._FixControlInputsAndContext([enter])

      # Add `enter` in this context.
      self._values.add(enter.name)
      self._external_values[val.name] = enter
      result = enter
    else:
      actual_val = self._external_values.get(val.name)
      if actual_val is not None:
        result = actual_val
    return result

  def AddOp(self, op: ops.Operation):
    # [FUNCTION DOC]
    # Registers an op in the loop. Input: op. Shape, size and rank ops in a gradient loop whose
    # input comes from the forward loop (and not in an XLA context) are moved into the input's own
    # context to save memory; all others go to the internal adder.
    """Add `op` to the current context."""
    # For a reduction op, if op is in a grad context and its input is from
    # its forward context, moving op to the forward context means we would
    # store the tensor after the reduction as opposed to the tensor before
    # reduction, and therefore could significantly reduce memory consumption.
    # For now, we do this only for a few ops.
    #
    # If in XLA context, do not move constant ops to forward pass as pushing to
    # and popping from a stack removes the constant property of an op and breaks
    # XLA compilation, which requires certain inputs to be constant for certain
    # ops.
    if not util.IsInXLAContext(op) and op.type in {"Shape", "Size", "Rank"}:
      grad_ctxt = ops.get_default_graph()._get_control_flow_context()
      if grad_ctxt:
        grad_ctxt = grad_ctxt.GetWhileContext()
        if grad_ctxt.grad_state:
          op_input_forward_ctxt = util.GetWhileContext(op.inputs[0].op)
          if op_input_forward_ctxt == grad_ctxt.grad_state.forward_context:
            op_input_ctxt = op.inputs[0].op._get_control_flow_context()
            op._set_control_flow_context(op_input_ctxt)
            op_input_ctxt._AddOpInternal(op)
            return
    self._AddOpInternal(op)

  #  pylint: disable=g-doc-args
  def _AddOpInternal(self, op: ops.Operation):
    # [FUNCTION DOC]
    # HELPER: core of adding an op to the loop. Input: op. Partitioned-call ops get a control edge
    # from the control pivot. Ops with no inputs lose external control edges and, if no internal
    # control inputs remain, get a control edge from the pivot; their outputs are recorded. Ops with
    # inputs have each input routed through the value-adding logic and rewired, then get a loop-invariant
    # control dependency guard. External control inputs are replaced by identity ops created inside the
    # loop. Output names are recorded, fetching/feeding blocked and the outer context notified.
    # Mutates graph and context; returns nothing.
    """Add `op` to the current context.

    We move any external control dependencies of the op to the loop pivot, to
    ensure they get executed.
    """
    # This is needed to prevent frame mismatch errors where there are Const
    # nodes inside tf.function in v1 while_loop and inlining is turned on.
    if op.type in ["PartitionedCall", "StatefulPartitionedCall"]:
      op._add_control_input(self.GetControlPivot().op)  # pylint: disable=protected-access
    if not op.inputs:
      # Remove any external control dependency on this op
      control_inputs, external_inputs = self._RemoveExternalControlEdges(op)
      # Add a control edge from the control pivot to this op.
      if not control_inputs:
        # pylint: disable=protected-access
        op._add_control_input(self.GetControlPivot().op)
        # pylint: enable=protected-access
      for x in op.outputs:
        self._values.add(x.name)
    else:
      for index in range(len(op.inputs)):
        x = op.inputs[index]
        real_x = self.AddValue(x)
        if real_x != x:
          op._update_input(index, real_x)  # pylint: disable=protected-access
      # Remove any external control dependency on this op.
      _, external_inputs = self._RemoveExternalControlEdges(op)
      # Add a control dependency to prevent loop invariants from
      # enabling ops that should not be executed.
      self._MaybeAddControlDependency(op)
      for x in op.outputs:
        self._values.add(x.name)
    if external_inputs:
      # Use an identity to pull control inputs as data inputs. Note that we
      # ignore ops which don't have outputs. TODO(apassos): fix that
      with ops.control_dependencies(None):
        self.Enter()
        external_inputs = [
            array_ops.identity(x.outputs[0]).op
            for x in external_inputs
            if x.outputs
        ]
        self.Exit()
      op._add_control_inputs(external_inputs)  # pylint: disable=protected-access
    if self._outer_context or not util.IsLoopExit(op):
      op.graph.prevent_fetching(op)
      for x in op.outputs:
        op.graph.prevent_feeding(x)

    if self._outer_context:
      self._outer_context.AddInnerOp(op)

  def _MaybeAddControlDependency(self, op: ops.Operation):
    # [FUNCTION DOC]
    # HELPER: guards ops that depend only on loop invariants. Input: op. If the op has no control
    # inputs and either is a function call / symbolic gradient op or all its inputs come from
    # constant (loop-invariant) enter ops, adds a control edge from the control pivot so the op runs
    # only when the loop iterates. Mutates the op; returns nothing.
    """Add a control input to the op if it only depends on loop invariants."""

    def _IsOpFree(op):
      # [FUNCTION DOC]
      # HELPER: predicate for the guard above. Input: op. Returns false if it already has control
      # inputs; true if it is a function-call or symbolic-gradient op; otherwise true only when every
      # input is produced by a loop-constant enter op.
      """Determines if `op` needs a control dependency."""
      if op.control_inputs:
        return False
      # pylint: disable=protected-access
      if op.graph._is_function(op.type) or op.type == "SymbolicGradient":
        return True
      # pylint: enable=protected-access
      for x in op.inputs:
        if not util.IsLoopConstantEnter(x.op):
          return False
      return True

    if _IsOpFree(op):
      # pylint: disable=protected-access
      op._add_control_input(self.GetControlPivot().op)
      # pylint: enable=protected-access

  def AddForwardLoopCounter(self, outer_grad_state):
    # [FUNCTION DOC]
    # Adds a counting loop to the forward loop, used when starting to build the gradient loop; must
    # be called in the outer context of this forward context. Input: outer_grad_state (the outer
    # gradient state, or nothing/null if not nested). Creates a zero-initialized counter (ordered
    # after the outer index add op when nested), enters it into this loop, increments it on every
    # iteration using the loop pivot, and exits the final count. Registers the new enter and exit
    # tensors in the loop lists. Returns a pair: the total iteration count tensor and the
    # next-iteration tensor of the counter.
    """Adds a loop that counts the number of iterations.

    This is added to the forward loop at the time when we start to
    create the loop for backprop gradient computation. Called in
    the outer context of this forward context.

    The pseudocode is:
      `n = 0; while (_pivot) { n++; }`

    Note that a control dependency is added to `n` to ensure the correct
    execution order of stack push ops.

    Args:
      outer_grad_state: The outer grad state. None if not nested.

    Returns:
      The number of iterations taken by the forward loop and the loop index.
    """
    n = constant_op.constant(0, name="f_count")
    if outer_grad_state is not None:
      # Force the stack pushes of i-th execution of an inner loop to be ordered
      # before the pushes of (i+1)-th execution of the same inner loop.
      outer_add_op = outer_grad_state.forward_index.op.inputs[0].op
      n.op._add_control_input(outer_add_op)  # pylint: disable=protected-access

    self.Enter()
    self.AddName(n.name)
    enter_n = _Enter(
        n,
        self._name,
        is_constant=False,
        parallel_iterations=self._parallel_iterations,
        name="f_count")
    self.loop_enters.append(enter_n)

    merge_n = merge([enter_n, enter_n])[0]
    switch_n = switch(merge_n, self._pivot)

    index = math_ops.add(switch_n[1], 1)
    next_n = _NextIteration(index)
    merge_n.op._update_input(1, next_n)

    total_iterations = exit(switch_n[0], name="f_count")
    self.loop_exits.append(total_iterations)
    self.ExitResult([total_iterations])
    self.Exit()
    return total_iterations, next_n

  def AddBackpropLoopCounter(self, count, outer_grad_state):
    # [FUNCTION DOC]
    # Adds the counter loop that controls termination of the backprop loop. Inputs: count (number
    # of iterations to run backward) and outer_grad_state (outer gradient state or nothing/null).
    # If the count lives in another graph it is brought in via an identity. Builds enter/merge,
    # a greater-or-equal-one test as the loop condition (sets both the predicate pivot and the
    # pivot), switch, subtraction (set as the body pivot), next-iteration and exit ops, records the
    # enter and exit tensors, and, when nested, adds the exit as a control input of the outer
    # gradient sync op. Returns the next-iteration tensor of the counter (the loop index).
    """Add the backprop loop that controls the iterations.

    This is added to the backprop loop. It is used to control the loop
    termination of the backprop loop. Called in the outer context of
    this grad context.

    The pseudocode is:
      `n = count; while (n >= 1) { n--; }`

    Note that a control dependency is added to `final_zero` to ensure the
    correct execution order of stack pop ops.

    Args:
      count: The number of iterations for backprop.
      outer_grad_state: The outer grad state. None if not nested.

    Returns:
      The loop index.
    """
    in_separate_functions = count.graph is not ops.get_default_graph()
    if in_separate_functions:
      # Brings the count into this graph
      count = array_ops.identity(count)
    else:
      # TODO(apassos) XLA expects this constant to be created outside the loop,
      # so doing that for now.
      one = constant_op.constant(1, name="b_count")

    self.Enter()
    self.AddName(count.name)
    enter_count = _Enter(
        count,
        self._name,
        is_constant=False,
        parallel_iterations=self._parallel_iterations,
        name="b_count")
    self.loop_enters.append(enter_count)

    merge_count = merge([enter_count, enter_count])[0]
    self._pivot_for_pred = merge_count

    if in_separate_functions:
      one = constant_op.constant(1, name="b_count")
    pred = math_ops.greater_equal(merge_count, one)
    self._pivot = loop_cond(pred, name="b_count")
    switch_count = switch(merge_count, self._pivot)

    index = math_ops.subtract(switch_count[1], one)
    self._pivot_for_body = index
    next_count = _NextIteration(index)
    merge_count.op._update_input(1, next_count)

    final_zero = exit(switch_count[0], name="b_count")
    self.loop_exits.append(final_zero)
    if outer_grad_state is not None:
      # Force the stack pops of i-th execution of an inner loop to be ordered
      # before the pops of (i+1)-th execution of the same inner loop.
      # pylint: disable=protected-access
      outer_grad_state.grad_sync._add_control_input(final_zero.op)
      # pylint: enable=protected-access

    self.ExitResult([final_zero])
    self.Exit()
    return next_count

  def AddBackpropAccumulator(self, op: ops.Operation, grad):
    # [FUNCTION DOC]
    # Adds an accumulation loop that sums the per-iteration partial gradient of a loop invariant.
    # Called while in the gradient while context. Inputs: op (the enter op of the loop invariant) and
    # grad (partial gradient tensor of one iteration). Temporarily leaves this context and builds a
    # zero accumulator in the outer context, with the shape taken statically from grad when fully
    # known, else computed dynamically from the forward value (using forward history when nested
    # in another gradient loop). Re-enters, then builds enter, merge, switch, add, next-iteration and
    # exit ops. Registers enter/exit tensors and exposes the exit through the outer context.
    # Returns the accumulated gradient tensor.
    """Add an accumulation loop for every loop invariant.

    This is added to the backprop loop. It is used to accumulate partial
    gradients within each loop iteration. Called when in the gradient while
    context.

    The pseudocode is:
      ```
      acc = 0.0;
      while (_pivot) {
        acc += grad;
      }
      ```

    Args:
      op: The Enter op for a loop invariant.
      grad: The partial gradient of an iteration for a loop invariant.

    Returns:
      The gradient for a loop invariant.
    """
    self.Exit()
    # Create a zeros tensor with the right shape for acc. If we don't
    # know the full shape statically, we will have to get the shape
    # dynamically from the forward inference. Getting the shape right
    # for the zeros is only needed for the base case when the loop exits
    # without running any iterations.
    shape = grad.get_shape()
    if shape.is_fully_defined():
      if self.outer_context:
        self.outer_context.Enter()
      acc = constant_op.constant(0, grad.dtype, shape=shape, name="b_acc")
      if self.outer_context:
        self.outer_context.Exit()
    else:
      value = op.inputs[0]
      if (isinstance(self.outer_context, WhileContext) and
          self.outer_context.grad_state is not None):
        # We are in a nested while loop.
        forward_ctxt = self.grad_state.forward_context
        forward_ctxt.outer_context.Enter()
        zeros_shape = array_ops.shape_internal(value, optimize=False)
        forward_ctxt.outer_context.Exit()
        outer_grad_state = self.grad_state.outer_grad_state
        history_zeros_shape = outer_grad_state.AddForwardAccumulator(
            zeros_shape)
        self.outer_context.Enter()
        real_shape = outer_grad_state.AddBackpropAccumulatedValue(
            history_zeros_shape, zeros_shape)
        acc = array_ops.zeros(real_shape, grad.dtype)
        self.outer_context.Exit()
      else:
        if self.outer_context:
          self.outer_context.Enter()
        zeros_shape = array_ops.shape_internal(value, optimize=False)
        acc = array_ops.zeros(zeros_shape, grad.dtype)
        if self.outer_context:
          self.outer_context.Exit()

    self.Enter()
    self.AddName(acc.name)
    enter_acc = _Enter(
        acc,
        self._name,
        is_constant=False,
        parallel_iterations=self._parallel_iterations,
        name="b_acc")
    self.loop_enters.append(enter_acc)

    merge_acc = merge([enter_acc, enter_acc], name="b_acc")[0]
    switch_acc_false, switch_acc_true = switch(merge_acc, self._pivot)

    add_acc = math_ops.add(switch_acc_true, grad)
    next_acc = _NextIteration(add_acc)
    merge_acc.op._update_input(1, next_acc)  # pylint: disable=protected-access

    result_acc = exit(switch_acc_false, name="b_acc")
    self.loop_exits.append(result_acc)
    self.ExitResult([result_acc])
    return result_acc

  def AddBackpropIndexedSlicesAccumulator(self, op: ops.Operation, grad):
    # [FUNCTION DOC]
    # Variant of the accumulator for gradients that are indexed slices (efficient for embedding
    # updates). Inputs: op (the enter op of the loop invariant) and grad (indexed slices with values,
    # indices and optional dense shape). Leaves this context to build initial accumulators in the
    # outer context (one row of zeros for values, a single zero index, and a zero dense shape when
    # present), re-enters, and builds per-component enter/merge/switch loops that concatenate new
    # indices and values each iteration, and keep the elementwise maximum for the dense shape. The
    # enters do not copy input shapes since accumulators grow. Registers enter/exit tensors.
    # Returns indexed slices made from the exit tensors (dense shape omitted if the gradient had none).
    """This is used for accumulating gradients that are IndexedSlices.

    This is essentially the equivalent of AddBackpropAccumulator but optimized
    for things like updating embeddings from within a while loop.

    Args:
      op: The Enter op for a loop invariant.
      grad: The partial gradients represented as an IndexedSlices.

    Returns:
      The accumulated IndexedSlices gradient of the loop invariant.
    """
    values = grad.values
    indices = grad.indices
    dense_shape = grad.dense_shape

    self.Exit()
    if self.outer_context:
      self.outer_context.Enter()
    if values.get_shape().is_fully_defined():
      values_shape = tensor_shape.TensorShape([tensor_shape.Dimension(1)] +
                                              values.get_shape().dims[1:])
      if self.outer_context:
        self.outer_context.Enter()
      values_acc = constant_op.constant(
          0, values.dtype, shape=values_shape, name="b_acc")
      if self.outer_context:
        self.outer_context.Exit()
    else:
      values_shape = _resource_safe_shape(op.inputs[0])[1:]
      values_shape = array_ops.concat([[1], values_shape], 0)
      values_acc = array_ops.zeros(values_shape, dtype=values.dtype)
    indices_acc = constant_op.constant([0], indices.dtype)
    shape_acc = None
    if dense_shape is not None:
      if dense_shape.get_shape().is_fully_defined():
        if self.outer_context:
          self.outer_context.Enter()
        shape_acc = constant_op.constant(
            0, dense_shape.dtype, shape=dense_shape.get_shape())
        if self.outer_context:
          self.outer_context.Exit()
      else:
        shape_acc = array_ops.zeros_like(
            array_ops.shape_internal(
                op.inputs[0], optimize=False, out_type=dense_shape.dtype),
            optimize=False)

    if self.outer_context:
      self.outer_context.Exit()

    self.Enter()
    self.AddName(values_acc.name)
    self.AddName(indices_acc.name)
    init_acc = [indices_acc, values_acc]
    if shape_acc is not None:
      self.AddName(shape_acc.name)
      init_acc.append(shape_acc)

    # Set use_input_shape=False since the accumulator tensors will grow in
    # size. If use_input_shape=True, the _update_input call below will result in
    # incompatible shapes.
    enter_acc = [
        _Enter(
            x,
            self._name,
            is_constant=False,
            parallel_iterations=self._parallel_iterations,
            use_input_shape=False,
            name="b_acc") for x in init_acc
    ]
    # Manually set appropriate partial shapes.
    enter_acc[0].set_shape([None])
    if values_acc.shape.dims is not None:
      enter_acc[1].set_shape([None] + values_acc.shape.as_list()[1:])
    self.loop_enters.extend(enter_acc)

    merge_acc = [merge([x, x], name="b_acc")[0] for x in enter_acc]
    switch_acc = [switch(x, self._pivot) for x in merge_acc]

    # The actual accumulation.
    acc_indexed_slices = [
        array_ops.concat([xa[1], xv], 0)
        for xa, xv in zip(switch_acc[:2], [indices, values])
    ]
    if shape_acc is not None:
      # For the shape we just keep the maximum
      acc_indexed_slices.append(math_ops.maximum(dense_shape, switch_acc[2][1]))

    next_acc = [_NextIteration(x) for x in acc_indexed_slices]
    for xm, xn in zip(merge_acc, next_acc):
      xm.op._update_input(1, xn)  # pylint: disable=protected-access

    exit_acc = [exit(x[0], name="b_acc") for x in switch_acc]
    self.loop_exits.extend(exit_acc)

    self.ExitResult(exit_acc)
    return indexed_slices.IndexedSlices(
        indices=exit_acc[0],
        values=exit_acc[1],
        dense_shape=exit_acc[2] if shape_acc is not None else None)

  def _InitializeValues(self, values):
    # [FUNCTION DOC]
    # HELPER: resets the set of names seen in this context to the names of the given tensors.
    # Input: values (list of tensors). Raises a type error if any element is not a tensor. Replaces
    # the previous contents of the seen-values set. Returns nothing.
    """Makes the values known to this context."""
    self._values = set()
    for x in values:
      if isinstance(x, tensor_lib.Tensor):
        self._values.add(x.name)
      else:
        raise TypeError("'values' must be a list of Tensors. "
                        f"Received: {type(x)}.")

  def _BuildLoop(self, pred, body, flat_orig_loop_vars, flat_loop_vars,
                 loop_vars_signature):
    # [FUNCTION DOC]
    # HELPER: core of loop construction (called with this context already entered). Inputs: pred
    # (callable producing the continue condition from loop variables), body (callable producing new
    # loop variables), flat_orig_loop_vars (flat original variables, used to detect tensor arrays),
    # flat_loop_vars (flat tensors, tensor arrays replaced by flow), and loop_vars_signature
    # (structure of type specs carrying shape invariants). Makes variables known to outer contexts,
    # creates enter ops (checking each initial shape against its invariant), fixes control inputs,
    # builds merge, the loop condition from pred, switches, and the body on identities of the true
    # outputs; summaries created in the body are attached as control dependencies; checks that body
    # output structure matches the inputs; closes the loop via next-iteration/back edges; adds exit
    # ops and exposes them to the outer context. Raises a value error if a shape invariant is
    # incompatible with the initial shape or if the body returns a different number of values, and
    # structure errors on mismatches. Returns a pair: the original body result structure and the
    # list of exit tensors. Mutates graph, context lists and pivots.
    """Core: Add the loop termination condition and body to the graph."""
    flat_shape_invariants = nest.map_structure(
        lambda spec: spec.shape,
        nest.flatten(loop_vars_signature, expand_composites=True))

    # Let the context know the loop variables so the loop variables
    # would be added in the outer contexts properly.
    self._InitializeValues(flat_loop_vars)
    if self._outer_context:
      real_vars = [self._outer_context.AddValue(x) for x in flat_loop_vars]
    else:
      real_vars = flat_loop_vars

    enter_vars = []
    with ops.control_dependencies(None):
      for real_var, shape_invariant in zip(real_vars, flat_shape_invariants):
        enter_var = _Enter(
            real_var,
            self._name,
            is_constant=False,
            parallel_iterations=self._parallel_iterations,
            use_input_shape=False)

        if _ShapeLessThanOrEqual(real_var.get_shape(), shape_invariant):
          enter_var.set_shape(shape_invariant)
        else:
          raise ValueError(
              f"The shape invariant specified for {real_var.name} is not "
              "compatible with the initial shape of the loop variable. It "
              f"enters the loop with shape {real_var.get_shape()}, but the "
              f"specified shape invariant is {shape_invariant}.")

        enter_var.graph.prevent_feeding(enter_var)
        if self._outer_context:
          self._outer_context.AddInnerOp(enter_var.op)
        enter_vars.append(enter_var)

    # Finds the closest enclosing non-None control pivot.
    outer_context = self._outer_context
    control_pivot = None
    while outer_context is not None and control_pivot is None:
      control_pivot = outer_context.GetControlPivot()
      # pylint: disable=protected-access
      outer_context = outer_context._outer_context
      # pylint: enable=protected-access

    if control_pivot is not None:
      for var in enter_vars:
        if util.IsLoopConstantEnter(var.op.inputs[0].op):
          # pylint: disable=protected-access
          var.op._add_control_input(control_pivot.op)
          # pylint: enable=protected-access

    # Fix the control inputs and control flow context of these enter ops.
    self._FixControlInputsAndContext(enter_vars)
    self._InitializeValues(enter_vars)
    self._loop_enters = enter_vars

    merge_vars = [merge([x, x])[0] for x in enter_vars]
    self._pivot_for_pred = merge_vars[0]

    merge_vars_with_tensorarrays = nest.map_structure(
        _convert_flow_to_tensorarray, flat_orig_loop_vars, merge_vars)
    # Build the graph for pred.
    packed_vars = nest.pack_sequence_as(
        structure=loop_vars_signature,
        flat_sequence=merge_vars_with_tensorarrays,
        expand_composites=True)
    c = ops.convert_to_tensor(pred(*packed_vars))
    self._pivot = loop_cond(c, name="LoopCond")
    switch_vars = [_SwitchRefOrTensor(x, self._pivot) for x in merge_vars]

    # Build the graph for body.
    vars_for_body = [_Identity(x[1]) for x in switch_vars]
    self._pivot_for_body = vars_for_body[0]
    # Convert TensorArray flow variables inside the context back into
    # their associated TensorArrays for calling the body.
    vars_for_body_with_tensorarrays = nest.map_structure(
        _convert_flow_to_tensorarray, flat_orig_loop_vars, vars_for_body)
    packed_vars_for_body = nest.pack_sequence_as(
        structure=loop_vars_signature,
        flat_sequence=vars_for_body_with_tensorarrays,
        expand_composites=True)
    pre_summaries = ops.get_collection(ops.GraphKeys._SUMMARY_COLLECTION)  # pylint: disable=protected-access
    body_result = body(*packed_vars_for_body)
    post_summaries = ops.get_collection(ops.GraphKeys._SUMMARY_COLLECTION)  # pylint: disable=protected-access
    if not nest.is_nested(body_result):
      body_result = [body_result]
    if len(post_summaries) > len(pre_summaries):
      new_summaries = post_summaries[len(pre_summaries):]
      summary_ref = ops.get_collection_ref(ops.GraphKeys._SUMMARY_COLLECTION)  # pylint: disable=protected-access
      summary_ref[:] = pre_summaries
      with ops.control_dependencies(new_summaries):

        def map_fn(x):
          # [FUNCTION DOC]
          # HELPER: callback applied to each body result when summaries were created. Tensor arrays
          # pass through unchanged; every other value is replaced by an identity so the summary
          # control dependencies apply to it.
          # TODO(apassos) figure out how to trigger with tensor arrays as well
          if isinstance(x, tensor_array_ops.TensorArray):
            return x
          return array_ops.identity(x)

        body_result = nest.map_structure(
            map_fn, body_result, expand_composites=True)

    body_result = variable_utils.convert_variables_to_tensors(body_result)
    # Compare the structure types of input and output of body.
    # For backwards compatibility, the first layer is forced to a list
    # during this comparison, because inputs are typically lists and
    # outputs of the body are typically tuples.
    nest.assert_same_structure(
        list(packed_vars_for_body), list(body_result), expand_composites=True)

    # Store body_result to keep track of TensorArrays returned by body
    original_body_result = body_result
    # Convert TensorArrays returned by body into their flow variables
    result = nest.map_structure(
        _convert_tensorarray_to_flow,
        nest.flatten(body_result, expand_composites=True),
        expand_composites=True)
    result = ops.convert_n_to_tensor_or_composite(result)

    # Add NextIteration and the back edges to complete the loop.
    if len(merge_vars) != len(result):
      raise ValueError("Number of inputs and outputs of 'body' must match "
                       f"'loop_vars'. Got {len(merge_vars)} for the number of "
                       f"inputs/outputs, and {len(result)} for 'loop_vars'.")
    next_vars = []
    for m, v in zip(merge_vars, result):
      next_vars.append(_AddNextAndBackEdge(m, v))

    # Add the exit ops.
    exit_vars = [exit(x[0]) for x in switch_vars]
    self._loop_exits = exit_vars

    # Exit the loop.
    self.ExitResult(exit_vars)

    return original_body_result, exit_vars

  def BuildLoop(self, pred, body, loop_vars, shape_invariants,
                return_same_structure):
    # [FUNCTION DOC]
    # Public entry for building the loop. Inputs: pred (condition callable), body (body callable),
    # loop_vars (structure of tensors, composite tensors or tensor arrays), shape_invariants
    # (optional structure matching loop_vars with shapes or type specs), and return_same_structure
    # (keep the loop_vars structure even for a single variable). Converts variables (tensor arrays to
    # flow tensors), derives type specs for shape invariants, enters this context, builds the loop
    # while holding the graph mutation lock, always exits the context, and converts flow outputs back
    # to tensor arrays. Returns the exit values packed like the body result; when not preserving
    # structure and there is a single variable, returns just that element.
    """Add the loop termination condition and body to the graph."""

    # Keep flat_orig_loop_vars to identify which are TensorArrays
    flat_orig_loop_vars = nest.flatten(loop_vars, expand_composites=True)

    loop_vars = nest.map_structure(
        _convert_to_tensor_or_composite_or_tensorarray, loop_vars)
    # Convert TensorArrays to their flow variables
    flat_loop_vars = nest.map_structure(
        _convert_tensorarray_to_flow,
        nest.flatten(loop_vars, expand_composites=True))

    if shape_invariants is not None:
      loop_vars_signature = nest.map_structure(
          _shape_invariant_to_type_spec, loop_vars, shape_invariants)
    else:
      loop_vars_signature = nest.map_structure(
          _shape_invariant_to_type_spec, loop_vars)

    try:
      self.Enter()
      # _BuildLoop calls _update_input in several places. _mutation_lock()
      # ensures a Session.run call cannot occur between creating and mutating
      # new ops.
      with ops.get_default_graph()._mutation_lock():  # pylint: disable=protected-access
        original_body_result, exit_vars = self._BuildLoop(
            pred, body, flat_orig_loop_vars, flat_loop_vars,
            loop_vars_signature)
    finally:
      self.Exit()

    flat_result = nest.flatten(original_body_result, expand_composites=True)
    # Convert TensorArray flow variables outside the context back into
    # their associated TensorArrays for returning to caller.
    exit_vars_with_tensorarrays = nest.map_structure(
        _convert_flow_to_tensorarray, flat_result, exit_vars)

    packed_exit_vars = nest.pack_sequence_as(
        structure=original_body_result,
        flat_sequence=exit_vars_with_tensorarrays,
        expand_composites=True)

    if return_same_structure:
      return packed_exit_vars
    else:
      return packed_exit_vars[0] if len(exit_vars) == 1 else packed_exit_vars

  def _FixControlInputsAndContext(self, enters):
    # [FUNCTION DOC]
    # HELPER: post-processes enter ops created for loop variables. Input: enters (list of tensors,
    # otherwise a type error is raised). For each, sets the control flow context to this loop and
    # adds only those control dependencies of the original input that belong to an ancestor context
    # inside the outer while context; other control inputs are dropped. Records the op as seen by the
    # graph's control dependency tracking. Mutates ops; returns nothing.
    graph = ops.get_default_graph()
    # pylint: disable=protected-access
    for e in enters:
      if isinstance(e, tensor_lib.Tensor):
        xs = [e]
      else:
        raise TypeError("'enters' must be a list of Tensors. "
                        f"Received: {type(e)}.")
      for x in xs:
        inp_op = x.op.inputs[0].op
        control_inputs = graph._control_dependencies_for_inputs([inp_op])
        outer_control_inputs = []
        for op in control_inputs:
          # We need to keep control inputs that are in any ancestor
          # ControlFlowContext, and within outer WhileContext.
          keep_as_control_input = True
          op_ctxt = util.GetOutputContext(op)
          outer_ctxt = self.outer_context
          outer_while_context = (None if outer_ctxt is None else
                                 outer_ctxt.GetWhileContext())
          while outer_ctxt != op_ctxt:
            if outer_ctxt is None or outer_ctxt == outer_while_context:
              keep_as_control_input = False
              break
            outer_ctxt = outer_ctxt.outer_context
          if keep_as_control_input:
            outer_control_inputs.append(op)
        x.op._set_control_flow_context(self)
        x.op._add_control_inputs(outer_control_inputs)
        graph._record_op_seen_by_control_dependencies(x.op)
    # pylint: enable=protected-access

  def IsWhileContext(self):
    # [FUNCTION DOC]
    # Returns true; marks this as a while loop context.
    return True


# pylint: enable=redefined-outer-name


def _AsTensorList(x, p):
  # [FUNCTION DOC]
  # HELPER: normalizes the outputs of cond-like branches to a list of tensors. Inputs: x (a tensor,
  # KNOWN ISSUE: for indexed slices only values and indices are copied, so any dense shape is
  # dropped from the result.
  # indexed slices, operation, or list/tuple of them) and p (tensor used as the stand-in for
  # operations). Operations become an identity of p with a dependency on the operation; every entry
  # is converted and wrapped in an identity. Returns a list.
  """Return x as a list of Tensors or IndexedSlices.

  For entries of `x` that are Operations, this returns an Identity of `p`
  with a dependency on the operation.

  Args:
    x: A Tensor/IndexedSlices/Operation or a list or tuple of them.
    p: A Tensor to return for entries in `x` that are Operations.

  Returns:
    A list of Tensors or IndexedSlices.
  """
  if not isinstance(x, (list, _basetuple)):
    x = [x]

  l = []
  for v in x:
    if isinstance(v, ops.Operation):
      v = with_dependencies([v], p)
    v = ops.convert_to_tensor_or_composite(v)
    if isinstance(v, tensor_lib.Tensor):
      l.append(array_ops.identity(v))
    else:
      l.append(
          indexed_slices.IndexedSlices(
              array_ops.identity(v.values), array_ops.identity(v.indices)))
  return l


def _CheckResults(a, b):
  # [FUNCTION DOC]
  # HELPER: sanity check for two branch result lists. Inputs: a and b, lists of tensors. Asserts
  # equal length and equal dtypes pairwise. Raises an assertion error with a message on mismatch
  # (not a regular value error, and skipped if assertions are disabled). Returns nothing.
  assert len(a) == len(b), (
      "Values returned by a() and b() must have the same length.")
  for x, y in zip(a, b):
    assert x.dtype == y.dtype, (
        "Values returned by a() [%s] and b() [%s] must have "
        "the same type: %s, %s." % (x.name, y.name, x.dtype.name, y.dtype.name))


def with_dependencies(dependencies, output_tensor, name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring says a type error is raised for other inputs, but arbitrary values
  # are converted to tensors instead and a type error arises only from the identity helper for
  # unsupported types.
  # Returns the content of output_tensor so that it is available only after the dependencies have
  # run. Inputs: dependencies (iterable of operations), output_tensor (tensor or indexed slices) and
  # optional name. In eager mode returns output_tensor unchanged. In graph mode, creates identity
  # ops under the control dependencies, colocated with the tensor; for indexed slices only the
  # values pass through the identity and indices and dense shape are reused.
  """Produces the content of `output_tensor` only after `dependencies`.

  In some cases, a user may want the output of an operation to be
  consumed externally only after some other dependencies have run
  first. This function ensures returns `output_tensor`, but only after all
  operations in `dependencies` have run. Note that this means that there is
  no guarantee that `output_tensor` will be evaluated after any `dependencies`
  have run.

  See also `tf.tuple` and `tf.group`.

  Args:
    dependencies: Iterable of operations to run before this op finishes.
    output_tensor: A `Tensor` or `IndexedSlices` that will be returned.
    name: (Optional) A name for this operation.

  Returns:
    Same as `output_tensor`.

  Raises:
    TypeError: if `output_tensor` is not a `Tensor` or `IndexedSlices`.
  """
  if context.executing_eagerly():
    return output_tensor
  with ops.name_scope(name, "control_dependency",
                      list(dependencies) + [output_tensor]) as name:
    with ops.colocate_with(output_tensor):
      with ops.control_dependencies(dependencies):
        output_tensor = ops.convert_to_tensor_or_composite(output_tensor)
        if isinstance(output_tensor, indexed_slices.IndexedSlices):
          return indexed_slices.IndexedSlices(
              _Identity(output_tensor.values, name=name), output_tensor.indices,
              output_tensor.dense_shape)
        else:
          return _Identity(output_tensor, name=name)


def _GroupControlDeps(dev, deps, name=None):
  # [FUNCTION DOC]
  # HELPER for the group function. Inputs: dev (device string or nothing/null), deps (operations or
  # tensors to depend on) and optional name. Creates a no-op that has control dependencies on all
  # deps, placed on the device when one is given. Returns the no-op operation.
  with ops.control_dependencies(deps):
    if dev is None:
      return no_op(name=name)
    else:
      with ops.device(dev):
        return no_op(name=name)


# TODO(touts): Accept "inputs" as a list.
@tf_export("group")
def group(*inputs, **kwargs):
  # [FUNCTION DOC]
  # KNOWN ISSUE: in eager mode it returns early before checking keyword arguments, so unknown
  # keywords are silently accepted there; also the docstring says it returns an operation but eager
  # mode returns nothing.
  # Creates an operation that finishes only when all given inputs have finished (no output). Inputs:
  # any number of tensors/operations (nested structures are flattened) and the optional keyword
  # name; any other keyword raises a value error. Inputs lacking a device attribute raise a type
  # error. Returns nothing/null in eager mode; with no inputs returns a no-op; with inputs on a
  # single device returns one no-op depending on them; across several devices builds one grouped
  # no-op per device (sorted by device name) and a root no-op depending on those.
  """Create an op that groups multiple operations.

  When this op finishes, all ops in `inputs` have finished. This op has no
  output.

  Note: *In TensorFlow 2 with eager and/or Autograph, you should not require
  this method, as ops execute in the expected order thanks to automatic control
  dependencies.* Only use `tf.group` when working with v1
  `tf.Graph` code.

  When operating in a v1-style graph context, ops are not executed in the same
  order as specified in the code; TensorFlow will attempt to execute ops in
  parallel or in an order convenient to the result it is computing.  `tf.group`
  allows you to request that one or more results finish before execution
  continues.

  `tf.group` creates a single op (of type `NoOp`), and then adds appropriate
  control dependencies.  Thus, `c = tf.group(a, b)` will compute the same graph
  as this:

      with tf.control_dependencies([a, b]):
          c = tf.no_op()

  See also `tf.tuple` and
  `tf.control_dependencies`.

  Args:
    *inputs: Zero or more tensors to group.
    name: A name for this operation (optional).

  Returns:
    An Operation that executes all its inputs.

  Raises:
    ValueError: If an unknown keyword argument is provided.
  """
  if context.executing_eagerly():
    return None
  name = kwargs.pop("name", None)
  if kwargs:
    raise ValueError("Unknown keyword arguments: " + ", ".join(kwargs.keys()))
  with ops.name_scope(name, "group_deps", inputs) as name:
    # Grouping no inputs means do nothing
    if not inputs:
      return no_op(name=name)

    # Sorts *inputs according to their devices.
    ops_on_device = {}  # device -> operations specified on the device.
    for inp in nest.flatten(inputs, expand_composites=True):
      if not hasattr(inp, "device"):
        raise TypeError("'inputs' should be zero or more (nested) Tensors. "
                        f"Received '{inp}' with type '{type(inp)}'.")
      dev = inp.device
      if dev in ops_on_device:
        ops_on_device[dev].append(inp)
      else:
        ops_on_device[dev] = [inp]
    if len(ops_on_device) == 1:
      # 1-level tree. The root node is the returned NoOp node.
      (dev, deps), = ops_on_device.items()
      return _GroupControlDeps(dev, deps, name=name)

    # 2-level tree. The root node is the returned NoOp node.
    # deps contains 1 NoOp node for each device.
    deps = []

    def device_key(dev):
      # [FUNCTION DOC]
      # HELPER: sort key for device names; maps a missing device (nothing/null) to the empty
      # string so that it can be compared with strings.
      """A sort key that allows None to be compared to strings."""
      return "" if dev is None else dev

    for dev in sorted(ops_on_device, key=device_key):
      deps.append(_GroupControlDeps(dev, ops_on_device[dev]))

    with ops.control_dependencies(deps):
      return no_op(name=name)


@tf_export("tuple", v1=[])
@dispatch.add_dispatch_support
def tuple_v2(tensors, control_inputs=None, name=None):
  # [FUNCTION DOC]
  # Public v2 wrapper of the tuple function (exported as tuple). Inputs: tensors (list of tensors,
  # indexed slices, or null entries), control_inputs (optional additional operations to finish
  # first) and name. Simply forwards to the module's v1 tuple implementation with the arguments
  # reordered and returns its result; errors propagate from there.
  """Groups tensors together.

  The returned tensors have the same value as the input tensors, but they
  are computed only after all the input tensors have been computed.

  Note: *In TensorFlow 2 with eager and/or Autograph, you should not require
  this method, as ops execute in the expected order thanks to automatic control
  dependencies.* Only use `tf.tuple` when working with v1 `tf.Graph` code.

  See also `tf.group` and `tf.control_dependencies`.

  Example:
  >>> with tf.Graph().as_default():
  ...   with tf.compat.v1.Session() as sess:
  ...     v = tf.Variable(0.0)
  ...     a = tf.constant(1.0)
  ...     sess.run(tf.compat.v1.global_variables_initializer())
  ...     for i in range(5):
  ...       update_op = v.assign_add(1.0)
  ...       b = a + v
  ...       res_b = sess.run(b)
  ...       res_v = sess.run(v)
  ...       print(res_v)
  0.0
  0.0
  0.0
  0.0
  0.0

  >>> with tf.Graph().as_default():
  ...   with tf.compat.v1.Session() as sess:
  ...     v = tf.Variable(0.0)
  ...     a = tf.constant(1.0)
  ...     sess.run(tf.compat.v1.global_variables_initializer())
  ...     for i in range(5):
  ...       update_op = v.assign_add(1.0)
  ...       calc = [a + v]
  ...       # `tf.tuple` ensures `update_op` is run before `b`
  ...       b = tf.tuple(calc, [tf.group(update_op)])
  ...       res_b = sess.run(b)
  ...       res_v = sess.run(v)
  ...       print(res_v)
  1.0
  2.0
  3.0
  4.0
  5.0


  Args:
    tensors: A list of `Tensor`s or `IndexedSlices`, some entries can be `None`.
    control_inputs: List of additional ops to finish before returning.
    name: (optional) A name to use as a `name_scope` for the operation.

  Returns:
    Same as `tensors`.

  Raises:
    ValueError: If `tensors` does not contain any `Tensor` or `IndexedSlices`.
    TypeError: If `control_inputs` is not a list of `Operation` or `Tensor`
      objects.

  """
  return tuple(tensors=tensors, name=name, control_inputs=control_inputs)  # pylint: disable=redefined-builtin


@tf_export(v1=["tuple"])
@dispatch.add_dispatch_support
def tuple(tensors, name=None, control_inputs=None):  # pylint: disable=redefined-builtin
  # [FUNCTION DOC]
  # Returns tensors with the same values as the inputs, but each available only after all inputs and
  # extra control inputs have been computed (a join point). Inputs: tensors (list of tensors,
  # indexed slices, operations or null entries), name, control_inputs (additional operations or
  # tensors that must finish). Eager mode returns the input unchanged. In graph mode, non-tensor
  # values are converted, the underlying operations are collected and de-duplicated in creation order,
  # grouped into one gate, and each tensor is re-emitted with a dependency on the gate (operations
  # are regrouped under the gate, null entries stay null). Raises a type error for control inputs that
  # are neither tensors nor operations and a value error if there is nothing to gate. Returns a list.
  # Shadows the builtin tuple type in this module.
  """Group tensors together.

  This creates a tuple of tensors with the same values as the `tensors`
  argument, except that the value of each tensor is only returned after the
  values of all tensors have been computed.

  `control_inputs` contains additional ops that have to finish before this op
  finishes, but whose outputs are not returned.

  This can be used as a "join" mechanism for parallel computations: all the
  argument tensors can be computed in parallel, but the values of any tensor
  returned by `tuple` are only available after all the parallel computations
  are done.

  See also `tf.group` and
  `tf.control_dependencies`.

  Args:
    tensors: A list of `Tensor`s or `IndexedSlices`, some entries can be `None`.
    name: (optional) A name to use as a `name_scope` for the operation.
    control_inputs: List of additional ops to finish before returning.

  Returns:
    Same as `tensors`.

  Raises:
    ValueError: If `tensors` does not contain any `Tensor` or `IndexedSlices`.
    TypeError: If `control_inputs` is not a list of `Operation` or `Tensor`
      objects.

  """
  if context.executing_eagerly():
    return tensors
  with ops.name_scope(name, "tuple", tensors) as name:
    tensors = [
        t if (isinstance(t, ops.Operation) or tensor_util.is_tf_type(t) or
              t is None) else ops.convert_to_tensor(t) for t in tensors
    ]
    gating_ops = [
        t if isinstance(t, ops.Operation) else t.op
        for t in tensors
        if t is not None
    ]
    if control_inputs:
      for c in control_inputs:
        if isinstance(c, tensor_lib.Tensor):
          c = c.op
        elif not isinstance(c, ops.Operation):
          raise TypeError(
              "'control_inputs' must only contain Operation or Tensor. "
              f"Received: {type(c)}")
        gating_ops.append(c)
    # Note that in order to ensure ordering in the pbtxt, we must take care to
    # ensure the order here.
    gating_ops = sorted(set(gating_ops), key=lambda op: op._id)  # Uniquify ops.
    if not gating_ops:
      raise ValueError("'tensors' must have at least one Tensor. "
                       f"Received: {tensors}.")
    gate = group(*gating_ops)
    tpl = []
    for t in tensors:
      if tensor_util.is_tf_type(t):
        tpl.append(with_dependencies([gate], t))
      elif isinstance(t, ops.Operation):
        with ops.control_dependencies([gate]):
          tpl.append(group(t))
      else:
        tpl.append(None)
    return tpl


class XLAControlFlowContext(ControlFlowContext):
  # [CLASS DOC]
  # Base class for XLA and TPU control flow contexts. It marks a region compiled by XLA: it reports
  # that it is an XLA context, performs no per-op or per-value bookkeeping, and is looked up by
  # the enclosing-XLA-context getter. Its name is fixed.
  """Base class for XLA and TPU control flow contexts."""

  def __init__(self):
    # [FUNCTION DOC]
    # Creates the context via the base constructor (registering with any current outer context)
    # and sets the name to a fixed string. No arguments.
    super(XLAControlFlowContext, self).__init__()
    self._name = "XLAControlFlowContext"

  def to_control_flow_context_def(self, context_def, export_scope=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the base version always raises a not-implemented error, so serializing an XLA
    # context always fails; inputs are the context definition message and optional export scope.
    # pylint: disable=useless-super-delegation
    # NOTE(slebedev): the method is required by `ControlFlowContext`.
    # Required override that just delegates to the base class.
    super(XLAControlFlowContext,
          self).to_control_flow_context_def(context_def, export_scope)

  def IsXLAContext(self):
    # [FUNCTION DOC]
    # Returns true; identifies this as an XLA context.
    return True

  def AddOp(self, _):
    # [FUNCTION DOC]
    # Does nothing: XLA contexts do not track ops. The op argument is ignored.
    pass

  def AddValue(self, x):
    # [FUNCTION DOC]
    # Returns the given value unchanged: no switching or entering is done in XLA contexts.
    return x

  def RequiresUniqueFunctionRetracing(self):
    # [FUNCTION DOC]
    # Returns false: a function does not need retracing when this context changes. Subclasses may
    # override.
    """Returns whether the tf.function should be retraced if the context changes.
    """
    return False


@tf_export("__internal__.get_enclosing_xla_context", v1=[])
def get_enclosing_xla_context():
  # [FUNCTION DOC]
  # Finds the XLA control flow context that encloses the current default graph. Walks the current
  # control flow context and its outer contexts; if none is an XLA context, moves to the outer
  # graph (for function graphs and v2 control flow) and repeats. Returns the XLA context, or
  # nothing/null if there is none. Exported as an internal API. No side effects.
  """Recursively find and return the XLAControlFlowContext."""
  graph = ops.get_default_graph()
  while graph is not None:
    # pylint: disable=protected-access
    context_ = graph._get_control_flow_context()
    # pylint: enable=protected-access
    while context_ is not None:
      if isinstance(context_, XLAControlFlowContext):
        return context_
      context_ = context_.outer_context
    # This may be a FuncGraph due to defuns or v2 control flow. We need to
    # find the original graph with the XLAControlFlowContext.
    graph = getattr(graph, "outer_graph", None)
  return None


def from_control_flow_context_def(context_def, import_scope=None):
  # [FUNCTION DOC]
  # Deserializes a generic control flow context definition into the right context object. Inputs:
  # context_def (message with either a conditional or a while field set) and optional import_scope.
  # Returns a new conditional or while context (recreating nested contexts). Raises a
  # not-implemented error if neither field is set.
  """Deserializes `context_def` into the appropriate ControlFlowContext.

  Args:
    context_def: ControlFlowContextDef proto
    import_scope: Optional `string`. Name scope to add.

  Returns:
    A ControlFlowContext subclass
  """
  if context_def.HasField("cond_ctxt"):
    return CondContext.from_proto(
        context_def.cond_ctxt, import_scope=import_scope)
  if context_def.HasField("while_ctxt"):
    return WhileContext.from_proto(
        context_def.while_ctxt, import_scope=import_scope)
  raise NotImplementedError("Unknown ControlFlowContextDef field: %s" %
                            context_def.WhichOneof("ctxt"))


ops.register_proto_function(
    ops.GraphKeys.COND_CONTEXT,
    proto_type=control_flow_pb2.CondContextDef,
    to_proto=CondContext.to_proto,
    from_proto=CondContext.from_proto)

ops.register_proto_function(
    ops.GraphKeys.WHILE_CONTEXT,
    proto_type=control_flow_pb2.WhileContextDef,
    to_proto=WhileContext.to_proto,
    from_proto=WhileContext.from_proto)
