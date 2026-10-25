# [MODULE DOC]
# Core of the vectorized parallel-for ("pfor") machinery. Given the graph of a loop body (a set of
# operations that depends on a scalar loop-iteration placeholder), this module adds new operations
# to the graph that compute, in one shot, the same results as running the body once per iteration.
# Every converted value is carried in a WrappedTensor that records whether it is "stacked" (has an
# extra leading dimension holding one slice per iteration), loop invariant, or logically stacked
# sparse components.
# Contents: helpers for variant (TensorList / Optional) handling and stacking; the WhileOp converter
# for V1 control-flow while loops; the _PforInput object handed to converters; the registry plus the
# RegisterPFor and RegisterPForWithArgs decorators; PForConfig (reductions across iterations); the
# PFor class that walks the graph and drives conversion; WhileV2 for functional while ops; and a
# large collection of per-operation converters (image, nn, array, math, random, linalg, logging,
# TensorArray, TensorList, Stack, parsing, functional, spectral ops).
# Import-time side effects: defines the (deprecated, ignored) boolean flag for falling back to a
# while loop, creates the global converter registry and fills it by registering every converter,
# and creates module-level caches for stack and channel-flatten bookkeeping.
# Public pieces are consumed by the sibling control-flow loop-building module; underscore names are
# internal implementation details.
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
"""Compiled parallel-for loop."""
# pylint: disable=missing-docstring,g-direct-tensorflow-import

import collections
import functools
import string
import sys
import traceback
from typing import List

from tensorflow.compiler.tf2xla.python import xla
from tensorflow.core.framework import full_type_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import execute
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import smart_cond
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_switch_case
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gen_linalg_ops
from tensorflow.python.ops import gen_list_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_nn_ops
from tensorflow.python.ops import gen_optional_ops
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import gen_sparse_ops
from tensorflow.python.ops import gen_spectral_ops
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import linalg_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import manip_ops
from tensorflow.python.ops import map_fn
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import special_math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.platform import flags
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import numpy_compat
from tensorflow.python.util import object_identity


# TODO(agarwal): remove flag.
flags.DEFINE_bool(
    "op_conversion_fallback_to_while_loop", True,
    "DEPRECATED: Flag is ignored.")


def _variant_handle_data(t):
  # [FUNCTION DOC]
  # HELPER: reads handle data for the variant-handling logic in this module.
  # Fetches the shape-and-type handle data attached to a variant tensor (eager-safe lookup).
  # Input: a tensor. Returns the list of shape-and-type entries, or nothing/null if the handle data
  # was never set. No side effects, no exceptions of its own.
  """Fetches handle data for a variant tensor `t`, or None if unavailable."""
  handle_data = resource_variable_ops.get_eager_safe_handle_data(t)
  if not handle_data.is_set:
    return None
  return handle_data.shape_and_type


def _variant_type_id(t):
  # [FUNCTION DOC]
  # HELPER: classifies variant tensors for the stacking logic.
  # Input: a tensor. Returns the full-type identifier of the first handle-data entry when the tensor
  # has the variant dtype and handle data is present; returns nothing/null for non-variant tensors
  # or when handle data is missing or empty (such tensors are presumed not to be lists/optionals).
  """Returns the full_type_pb2 type of `t`, or None if it is not available."""
  if t.dtype != dtypes.variant:
    return None
  shapes_and_types = _variant_handle_data(t)
  if shapes_and_types is None or not shapes_and_types:
    # TODO(b/169968286): Identify all variant tensors (e.g. maps) and we can
    # make this an error instead of assuming TensorLists have handle data.
    return None  # Presumed not a TensorList/Optional
  return shapes_and_types[0].type.type_id


_INTERNAL_STACKING_TYPE_IDS = (
    full_type_pb2.TFT_ARRAY,
    full_type_pb2.TFT_OPTIONAL)


def _is_variant_with_internal_stacking(t):
  # [FUNCTION DOC]
  # HELPER: decides whether a variant tensor is kept as a scalar by pfor.
  # KNOWN ISSUE: the docstring says it returns nothing/null when the answer is unknown, but the code
  # always returns a real boolean; a missing type identifier is simply "not in" the stacking set and
  # yields false.
  # Input: a tensor. Returns true when its variant type is array (TensorList) or optional, false
  # otherwise. For such tensors pfor keeps the variant scalar and stacks its contents instead.
  """Identifies variant tensors which pfor always maintains as scalars.

  For these, the pfor tensor is recorded as "stacked" if the content of the
  variant tensor (e.g. the elements of a TensorList) are all stacked.

  Args:
    t: A tensor to identify.
  Returns:
    True if `t` is a TensorList/Optional, False not, None if unknown.
  """
  type_id = _variant_type_id(t)
  return type_id in _INTERNAL_STACKING_TYPE_IDS


def _parse_variant_shapes_and_types(t):
  # [FUNCTION DOC]
  # HELPER: extracts handle data for variant tensors that need to be stacked.
  # Input: a variant tensor. Returns its shape-and-type list when the first entry is an array type or
  # has an unset type. Raises a value error if no handle data is present, or if the first entry has
  # any other type (the variant cannot be stacked). No side effects.
  """Extracts shape and dtype information from a variant tensor `t`."""
  shapes_and_types = _variant_handle_data(t)
  if shapes_and_types is None or not shapes_and_types:
    raise ValueError("Required handle data not set for {!r}".format(t))
  if shapes_and_types[0].type.type_id == full_type_pb2.TFT_ARRAY:
    return shapes_and_types
  else:
    if shapes_and_types[0].type.type_id == full_type_pb2.TFT_UNSET:
      return shapes_and_types
    else:
      raise ValueError(
          "Attempted to stack a variant-dtype tensor with no type set ({!r})"
          .format(t))


def _rank(t):
  # [FUNCTION DOC]
  # HELPER: static-or-dynamic rank utility used by broadcasting code.
  # Input: a tensor (or anything). Returns the statically known rank as a plain integer when
  # available, otherwise a rank tensor computed in the graph (adds an operation to the graph).
  """Returns rank as an integer (when statically known) or as a tensor."""
  rank = t.get_shape().rank if isinstance(t, tensor_lib.Tensor) else None
  return array_ops.rank(t) if rank is None else rank


def _size(t, dtype=None):
  # [FUNCTION DOC]
  # HELPER: static-or-dynamic element-count utility.
  # Inputs: a tensor and an optional output integer type (only used for the dynamic case).
  # Returns the statically known number of elements as a plain integer when available, otherwise a
  # size tensor created in the graph.
  """Returns size as an integer (when statically known) or as a tensor."""
  size = (
      t.get_shape().num_elements() if isinstance(t, tensor_lib.Tensor) else None
  )
  return array_ops.size(t, out_type=dtype) if size is None else size


def _expand_dims(t, axis, num_axes=1):
  # [FUNCTION DOC]
  # HELPER: multi-axis variant of expand-dims used by broadcasting helpers.
  # Inputs: tensor, the axis at which to insert, and the number of axes (default one). When the
  # number of axes is a plain integer it inserts that many size-one dimensions one at a time. When it
  # is a tensor, it builds a new shape (keeping the first dimension, then that many ones, then the
  # rest of the shape) and reshapes. Returns the expanded tensor.
  """Similar to `expand_dims` but supports insertion of multiple axes."""
  if isinstance(num_axes, int):
    for _ in range(num_axes):
      t = array_ops.expand_dims(t, axis)
  else:
    shape = array_ops.shape(t)
    ones = array_ops.fill(
        array_ops.reshape(num_axes, [1]), constant_op.constant(1, shape.dtype)
    )
    new_shape = array_ops.concat([shape[:1], ones, shape[1:]], axis=0)
    t = array_ops.reshape(t, new_shape)
  return t


def _stack(t, length):
  # [FUNCTION DOC]
  # HELPER: turns a loop-invariant value into a stacked one.
  # Inputs: a tensor and the loop-length vector (one-element integer vector or tensor). For ordinary
  # dtypes it tiles the tensor along a new leading axis and returns it wrapped as stacked. For
  # variants that are TensorLists with array handle data it builds a stacked TensorList (each element
  # tiled) via the list helper; other variant kinds, or handle data of unexpected length, raise a
  # value error. Adds operations to the graph.
  """stacks `t` `length` times."""
  # Note that this stacking may currently be triggered, for example, when a
  # loop invariant tensor with dtype variant is input to a while_loop which then
  # produces a loop dependent output. Simply stacking the variants may not be
  # suitable since operations on stacked handles may expect a vectorized version
  # of the variant.
  if t.dtype == dtypes.variant:
    shapes_and_types = _parse_variant_shapes_and_types(t)
    if shapes_and_types[0].type.type_id == full_type_pb2.TFT_ARRAY:
      if len(shapes_and_types) != 1:
        raise ValueError(
            f"Expected handle data of length 1, got {shapes_and_types!r} of "
            f"length {len(shapes_and_types)}.")
      return wrap(
          _stack_tensor_list(t, shapes_and_types[0].dtype, length),
          True)
    else:
      raise ValueError(
          "Attempted to stack an unhandled variant-dtype tensor of "
          f"type {shapes_and_types[0].type!r} ({t!r}).")
  shape = array_ops.shape(t)
  ones = array_ops.ones_like(shape)
  ones = array_ops.reshape(ones, [-1])
  length = array_ops.reshape(length, [-1])
  length = math_ops.cast(length, shape.dtype)
  multiples = array_ops.concat([length, ones], 0)
  t = array_ops.tile(array_ops.expand_dims(t, 0), multiples)
  return wrap(t, True)


# The following stateful ops can be safely called once, and with the same
# signature as the unconverted version, if their inputs are loop invariant.
# TODO(agarwal): implement a strategy for converting Variable reads/writes. The
# plan is to map each read/write in the loop_fn to a corresponding merged
# read/write in the converted graph. Writes need to be mergeable (e.g.
# AssignAdd) to be used in `pfor`. Given a certain read/write order in the
# loop_fn, doing a one-to-one conversion will simulate executing such
# instructions in lock-step across all iterations.
passthrough_stateful_ops = set([
    "VariableV2",
    "VarHandleOp",
    "VariableShape",
    "ReadVariableOp",
    "StackV2",
    "TensorArrayWriteV3",
    "TensorArrayReadV3",
    "TensorArraySizeV3",
])


# Ops which we will treat like stateful for the purpose of vectorization.
# Typically this is used to force pfor converters to run for these ops.
force_stateful_ops = set([
    # We vectorize this since we need to change the element shape set on the
    # list.
    "TensorListReserve",
])


def _is_stateful_pfor_op(op):
  # [FUNCTION DOC]
  # HELPER: statefulness test used when deciding whether an op must be converted.
  # Input: a graph operation or a WhileOp. A WhileOp reports its own statefulness. Constants and ops
  # in the passthrough set are never stateful; ops in the forced set always are; otherwise the op
  # definition's flag is used (an assertion fails if the op has no definition). Returns a boolean.
  if isinstance(op, WhileOp):
    return op.is_stateful
  if op.type == "Const":
    # Const didn't have an op_def.
    return False
  if op.type in passthrough_stateful_ops:
    return False
  if op.type in force_stateful_ops:
    return True
  assert hasattr(op, "op_def") and op.op_def is not None, op
  return op.op_def.is_stateful


# pylint: disable=protected-access
class WhileOp:
  # [CLASS DOC]
  # Converter state for a V1-style (control-flow primitive) while loop whose pieces appear in the
  # pfor loop body. Constructed from one Exit tensor of the loop, it reverse engineers the standard
  # Enter/Merge/Switch/NextIteration/Exit pattern and records: exit switches, body outputs,
  # NextIteration control inputs, Enter merges, loop outputs, the Enter tensors that are loop
  # variables, the "direct" Enter tensors for resources/variants that the body captures, whether the
  # loop was created inside the pfor body, and whether it contains stateful ops. An instance acts as
  # its own pfor converter (see pfor_converter) and, when called with a pfor input, produces a new
  # vectorized while loop that tracks which iterations are still active.
  """Object for storing state for converting the outputs of a while_loop."""

  def __init__(
      self,
      exit_node: tensor_lib.Tensor,
      pfor_ops: List[ops.Operation],
      fallback_to_while_loop: bool,
      pfor_config: "PForConfig",
  ):
    # [FUNCTION DOC]
    # Builds the converter for one while loop from its Exit tensor.
    # Inputs: exit_node (a tensor produced by an Exit op of the loop), pfor_ops (list of operations
    # that belong to the pfor loop body), fallback_to_while_loop (flag stored for sub-converters), and
    # pfor_config (the PForConfig in use, may be null).
    # It locates the while context, condition tensor and context name; decides whether the loop was
    # created inside the pfor body; walks every loop exit back through Switch, Merge, Enter and
    # NextIteration, collecting them in aligned lists; then scans all graph operations in the same
    # control-flow context to gather extra Enter tensors (resource and variant ones go to the direct
    # list) and to compute the stateful flag.
    # Raises assertion errors if the graph does not match the expected pattern or the argument is not a
    # tensor. Reads the default graph; changes nothing in it.
    """Initializer.

    Args:
      exit_node: A tensor output from the while_loop.
      pfor_ops: list of ops inside the current pfor loop.
      fallback_to_while_loop: If True, fallback to while loop when conversion of
        an op is not supported
      pfor_config: PForConfig object used while constructing loop body.
    """
    self._fallback_to_while_loop = fallback_to_while_loop
    self._pfor_config = pfor_config
    self._pfor_ops = set(pfor_ops)
    self._pfor_op_ids = set(x._id for x in pfor_ops)
    assert isinstance(exit_node, tensor_lib.Tensor)
    self._while_context = exit_node.op._get_control_flow_context()
    assert isinstance(self._while_context, control_flow_ops.WhileContext)
    self._context_name = self._while_context.name
    self._condition = self._while_context.pivot.op.inputs[0]
    # Parts of an external while_loop could be created inside a pfor loop.
    # However for the purpose here, we declare such loops to be external. Also
    # note that we check if the condition was created inside or outside to
    # determine if the while_loop was first created inside or outside.
    # TODO(agarwal): check that the Enter and Exit of this loop are unstacked.
    self._is_inside_loop = self.op_is_inside_loop(self._condition.op)
    if self._is_inside_loop:
      for e in self._while_context.loop_exits:
        assert self.op_is_inside_loop(e.op)

    # Note the code below tries to reverse engineer an existing while_loop graph
    # by assuming the following pattern of nodes.
    #
    #          NextIteration <---- Body <--- Enter
    #              |                ^
    #              V             ___| Y
    #    Enter -> Merge -> Switch___
    #                       ^       | N
    #                       |       V
    #                  LoopCond    Exit

    # Node that elements in the list below correspond one-to-one with each
    # other. i.e. these lists are the same size, and the i_th entry corresponds
    # to different Operations/Tensors of a single cycle as illustrated above.
    # List of Switch ops (ops.Operation) that feed into an Exit Node.
    self._exit_switches = []
    # List of inputs (tensor_lib.Tensor) to NextIteration.
    self._body_outputs = []
    # List of list of control inputs of the NextIteration nodes.
    self._next_iter_control_inputs = []
    # List of Merge ops (ops.Operation).
    self._enter_merges = []
    # List of output (tensor_lib.Tensor) of Exit nodes.
    self._outputs = []

    # List of Enter Tensors.
    # There are two types of Enter nodes:
    # - The Enter nodes that are used in the `loop_vars` argument to
    # `while_loop` (see
    # https://www.tensorflow.org/api_docs/python/tf/while_loop). We collect
    # these Enter nodes immediately below by tracing backwards from the Exit
    # nodes via Exit <- Switch <- Merge <- Enter. You can see this chain in the
    # diagram above. This allows us to have a 1:1 correspondence between the
    # self._outputs and the first elements in self._enters.
    # - The Enter nodes that are used only by the body. They don't appear in the
    # `loop_vars` and are not returned from the `while_loop`. In Python code,
    # they are usually captured by the body lambda. We collect them below by
    # iterating over all the ops in the graph. They are appended to the end of
    # self._enters or self._direct_enters, and don't correspond to any outputs
    # in self._outputs. Note that we keep the resource/variant Enter nodes in
    # self._direct_enters and the constructed while_loop's body uses them
    # directly as opposed to passing them as loop variables. This is done
    # because the while_body cannot partition the resource/variant Tensors, so
    # it has to leave them unchanged.
    self._enters = []
    self._direct_enters = []

    for e in self._while_context.loop_exits:
      self._outputs.append(e.op.outputs[0])
      switch = e.op.inputs[0].op
      assert switch.type == "Switch", switch
      self._exit_switches.append(switch)
      merge = switch.inputs[0].op
      assert merge.type == "Merge", merge
      self._enter_merges.append(merge)
      enter = merge.inputs[0].op
      assert enter.type == "Enter", enter
      self._enters.append(enter.outputs[0])
      next_iter = merge.inputs[1].op
      assert next_iter.type == "NextIteration", next_iter
      self._body_outputs.append(next_iter.inputs[0])
      self._next_iter_control_inputs.append(next_iter.control_inputs)

    # Collect all the Enter nodes that are not part of `loop_vars`, the second
    # category described above.
    # Also track whether the loop body has any stateful ops.
    self._is_stateful = False
    for op in ops.get_default_graph().get_operations():
      # TODO(agarwal): make sure this works with nested case.
      control_flow_context = op._get_control_flow_context()
      if control_flow_context is None:
        continue
      if control_flow_context.name == self._context_name:
        self._is_stateful |= _is_stateful_pfor_op(op)
        if op.type == "Enter":
          output = op.outputs[0]
          if output not in self._enters:
            if output.dtype in (dtypes.resource, dtypes.variant):
              if output not in self._direct_enters:
                self._direct_enters.append(output)
            else:
              self._enters.append(output)

  def __str__(self) -> str:
    # [FUNCTION DOC]
    # Returns a short text of the form while-loop followed by the loop's context name. Used for logging
    # and error messages only.
    """String representation."""
    return "while_loop(%s)" % self.name

  @property
  def inputs(self):
    # [FUNCTION DOC]
    # Property: returns the list of the inputs feeding every Enter node (loop variables first, then the
    # direct enters), i.e. the external values the while loop consumes.
    """Input to all the Enter nodes."""
    return [x.op.inputs[0] for x in self._enters + self._direct_enters]

  @property
  def control_inputs(self):
    # [FUNCTION DOC]
    # Property: returns a flat list of all control-dependency operations of all Enter nodes (loop
    # variable enters first, then direct enters).
    """Control input to all the Enter nodes."""
    control_inputs = []
    for x in self._enters + self._direct_enters:
      control_inputs.extend(x.op.control_inputs)
    return control_inputs

  @property
  def outputs(self) -> List[tensor_lib.Tensor]:
    # [FUNCTION DOC]
    # Property: returns the list of tensors produced by the Exit nodes of the loop.
    """Outputs of all the Exit nodes."""
    return self._outputs

  @property
  def name(self) -> str:
    # [FUNCTION DOC]
    # Property: returns the name of the while-loop's control-flow context.
    """Context name for the while loop."""
    return self._context_name

  @property
  def is_inside_loop(self) -> bool:
    # [FUNCTION DOC]
    # Property: true if the while loop was first created inside the pfor loop body (decided from the
    # location of the loop condition), false if it is treated as an external loop.
    """Returns true if the while_loop was created inside the pfor."""
    return self._is_inside_loop

  def op_is_inside_loop(self, op: ops.Operation) -> bool:
    # [FUNCTION DOC]
    # Tells whether an operation was created inside the pfor loop body.
    # Input: a graph operation (assertion error otherwise). Returns true if its id is among the ids of
    # the pfor body operations. Ids are compared rather than objects because the framework may hand
    # back different Python wrappers for the same node.
    """True if op was created inside the pfor loop body."""
    assert isinstance(op, ops.Operation)
    # Note that we use self._pfor_op_ids for the check and not self._pfor_ops
    # since it appears there tensorflow API could return different python
    # objects representing the same Operation node.
    return op._id in self._pfor_op_ids

  @property
  def is_stateful(self) -> bool:
    # [FUNCTION DOC]
    # Property: true if any operation inside this loop's control-flow context is stateful (computed
    # once at construction).
    return self._is_stateful

  @property
  def pfor_converter(self) -> "WhileOp":
    # [FUNCTION DOC]
    # Property used by the PFor driver: returns this object itself so that, being callable, it serves
    # as the converter for the whole while loop.
    """Return a converter for the while loop."""
    return self

  def _init_pfor(self, parent_pfor, indices, cond_stacked, inputs,
                 inputs_stacked):
    # [FUNCTION DOC]
    # HELPER: creates the nested PFor converter used for the condition and body of the new loop.
    # Inputs: parent_pfor (the PFor converting this loop), indices (int32 tensor of still-active
    # iteration ids), cond_stacked (whether the loop condition varies by iteration), inputs (list of
    # tensors matching the loop-variable Enter nodes) and inputs_stacked (matching list of booleans).
    # Creates a PFor over the same body ops, with its length equal to the number of active indices, and
    # pre-seeds its conversion map: direct Enter inputs map to their converted unstacked values (they
    # must be unstacked), Enter nodes map to the given inputs, and Merge/Switch outputs map to the same
    # inputs. The second Merge output gets a dummy constant of the wrong type on purpose, so that using
    # it as a real input fails at graph-construction time. Returns the new PFor.
    """Create a PFor object for converting parts of the while_loop.

    Args:
      parent_pfor: PFor object being used for converting the while_loop.
      indices: int32 Tensor of ids for the iterations that are still active
        (i.e. did not exit the while_loop).
      cond_stacked: True if the while_loop condition is stacked.
      inputs: list of input Tensors corresponding 1-to-1 with self._enters. Note
        that these Tensors are a subset of the loop variables for the generated
        while_loop.
      inputs_stacked: List of booleans corresponding 1-to-1 with `inputs`,
        indicating if the value is stacked or not.

    Returns:
      A PFor instance. The instance is initialized by adding conversion mappings
        of nodes that will be external to the conversion that the returned
        instance will be used for. e.g. Enter nodes as well as Merge and Switch
        outputs are mapped to converted values.
    """
    num_outputs = len(self._outputs)
    assert len(inputs) == len(self._enters)
    assert len(inputs_stacked) == len(self._enters)
    loop_var = parent_pfor.loop_var
    loop_len = array_ops.size(indices)
    pfor = PFor(
        loop_var,
        loop_len,
        pfor_ops=self._pfor_ops,
        all_indices=indices,
        all_indices_partitioned=cond_stacked,
        fallback_to_while_loop=self._fallback_to_while_loop,
        pfor_config=self._pfor_config)
    # Map all inputs of Enter nodes in self._direct_enters to their converted
    # values.
    for enter in self._direct_enters:
      enter_input = enter.op.inputs[0]
      converted_enter, stacked, is_sparse_stacked = parent_pfor._convert_helper(
          enter_input)
      # Since these are resources / variants, they should be unstacked.
      assert not stacked and not is_sparse_stacked, (enter, converted_enter)
      pfor._add_conversion(enter, wrap(converted_enter, False))

    # Map all Enter nodes to the inputs.
    for enter, inp, stacked in zip(self._enters, inputs, inputs_stacked):
      pfor._add_conversion(enter, wrap(inp, stacked))
    # Map outputs of Switch and Merge.
    for i in range(num_outputs):
      wrapped_inp = wrap(inputs[i], inputs_stacked[i])
      merge = self._enter_merges[i]
      pfor._add_conversion(merge.outputs[0], wrapped_inp)
      # Note that second output of Merge is typically not used, except possibly
      # as a control dependency. To avoid trying to output the correct value, we
      # employ a hack here. We output a dummy invalid value with an incorrect
      # dtype. This will allow control dependency to work but if using it as an
      # input, it should typically lead to errors during graph construction due
      # to dtype mismatch.
      # TODO(agarwal): Check in the original graph to see if there are any
      # consumers of this Tensor that use it as an input.
      pfor._add_conversion(merge.outputs[1],
                           wrap(constant_op.constant(-1.0), False))
      switch = self._exit_switches[i]
      # Don't need to worry about switch.output[0] which will feed to Exit node.
      pfor._add_conversion(switch.outputs[1], wrapped_inp)
    return pfor

  def _convert_enter(self, parent_pfor: "PFor", enter):
    # [FUNCTION DOC]
    # HELPER: converts the outside input of one Enter node.
    # Inputs: parent_pfor (the PFor in charge) and an Enter tensor. Converts its input and its control
    # inputs through the parent. If there are control inputs it passes the value through an identity
    # operation under those dependencies. Returns a pair of the converted tensor and whether it is
    # stacked.
    """Converts an Enter node."""
    inp, stacked, _ = parent_pfor._convert_helper(enter.op.inputs[0])
    control_inputs = []
    for x in enter.op.control_inputs:
      converted = parent_pfor._convert_helper(x)
      if not isinstance(converted, ops.Operation):
        converted = converted.t
      control_inputs.append(converted)
    if control_inputs:
      with ops.control_dependencies(control_inputs):
        inp = array_ops.identity(inp)
    return inp, stacked

  def _maybe_stacked(self, cache, inp):
    # [FUNCTION DOC]
    # HELPER: conservative, cached guess of whether converting a tensor will produce a stacked value.
    # Inputs: cache (mapping from tensors to booleans, updated in place) and inp (a tensor).
    # Returns true if the tensor may end up stacked; a false answer guarantees it is unstacked. Tensors
    # produced outside the loop, and certain shape/rank/zeros/ones/TensorArray-creation ops, are
    # unstacked. Stateful ops and Exit ops are assumed stacked. Otherwise it is stacked if any input
    # may be stacked (recursive).
    """Heuristic to figure out if the converting inp leads to a stacked value.


    Args:
      cache: map from Tensor to boolean indicating stacked/unstacked.
      inp: input Tensor.

    Returns:
      True if `inp` could get stacked. If the function returns False, the
      converted value should be guaranteed to be unstacked. If returning True,
      it may or may not be stacked.
    """
    if inp in cache:
      return cache[inp]
    if not self.op_is_inside_loop(inp.op):
      return False
    op = inp.op
    output = False
    if op.type in [
        "OnesLike",
        "Shape",
        "Rank",
        "ShapeN",
        "ZerosLike",
        "TensorArrayV3",
        "TensorArraySizeV3",
    ]:
      output = False
    elif _is_stateful_pfor_op(op):
      # This may be fairly aggressive.
      output = True
    elif op.type == "Exit":
      # This may be fairly aggressive.
      output = True
    else:
      for t in op.inputs:
        if self._maybe_stacked(cache, t):
          output = True
          break
    cache[inp] = output
    return output

  def _create_init_values(self, pfor_input: "_PforInput"):
    # [FUNCTION DOC]
    # HELPER: builds the initial argument list for the converted while loop.
    # Input: the pfor input for the loop. Under a name scope for loop initialisation, converts all
    # Enter nodes, seeds a stackedness cache (loop-variable Switch outputs get their Enter's
    # stackedness), then decides for each input whether it must be stacked, stacking loop-invariant
    # inputs whose body output could be stacked. Creates one TensorArray per loop output.
    # Returns a triple: the initial values (a true flag, all pfor indices, the inputs, the
    # TensorArrays), a list of booleans telling which inputs are stacked, and the shape invariants.
    """Create arguments passed to converted while_loop."""
    with ops.name_scope("while_init"):
      loop_len_vector = pfor_input.pfor.loop_len_vector
      loop_len = loop_len_vector[0]
      num_outputs = len(self._outputs)

      inputs = []
      maybe_stacked_cache = {}
      # Convert all the Enters. Need to do this before checking for stacking
      # below.
      for i, enter in enumerate(self._enters):
        inp, stacked = self._convert_enter(pfor_input.pfor, enter)
        inputs.append(inp)
        maybe_stacked_cache[enter] = stacked
        # Since this enter node is part of the `loop_vars`, it corresponds to an
        # output and its preceding switch. We mark this switch's output the same
        # stackness, to act at the base case for the logic below. Below, we will
        # be going through the body figuring out which inputs might need to be
        # stacked and which inputs can safely remain unstacked.
        if i < num_outputs:
          maybe_stacked_cache[self._exit_switches[i].outputs[1]] = stacked

      # Shape invariants for init_values corresponding to self._enters.
      input_shape_invariants = []
      # TensorArrays for outputs of converted while loop
      output_tas = []
      # Shape invariants for output TensorArrays.
      ta_shape_invariants = []
      # List of booleans indicating stackness of inputs, i.e. tensors
      # corresponding to self._enters.
      inputs_stacked = []
      for i, inp in enumerate(inputs):
        enter = self._enters[i]
        inp_stacked = self._maybe_stacked(maybe_stacked_cache, enter)
        # Note that even when an input is unstacked, the body could make it
        # stacked. we use a heuristic below to figure out if body may be making
        # it stacked.
        if i < num_outputs:
          body_output = self._body_outputs[i]
          if enter.op in self._pfor_ops:
            body_output_stacked = self._maybe_stacked(maybe_stacked_cache,
                                                      body_output)
          else:
            # If constructed outside of pfor loop, then the output would not be
            # stacked.
            body_output_stacked = False
          if body_output_stacked and not inp_stacked:
            inp = _stack(inp, loop_len_vector).t
            inputs[i] = inp
            inp_stacked = True
          # TODO(agarwal): other attributes for the TensorArray ?
          output_tas.append(tensor_array_ops.TensorArray(inp.dtype, loop_len))
          ta_shape_invariants.append(tensor_shape.TensorShape(None))

        inputs_stacked.append(inp_stacked)
        input_shape_invariants.append(tensor_shape.TensorShape(None))

      # See documentation for __call__ for the structure of init_values.
      init_values = [True, pfor_input.pfor.all_indices] + inputs + output_tas
      # TODO(agarwal): try stricter shape invariants
      shape_invariants = (
          [tensor_shape.TensorShape(None),
           tensor_shape.TensorShape(None)] + input_shape_invariants +
          ta_shape_invariants)

      return init_values, inputs_stacked, shape_invariants

  def _process_cond_unstacked(self, conditions, indices, inputs, output_tas):
    # [FUNCTION DOC]
    # HELPER: handles an iteration step when the loop condition is the same for all iterations.
    # Inputs: conditions, indices, inputs, output_tas. Since all iterations end together, nothing is
    # partitioned; the loop-variable values are written to index zero of each output TensorArray only
    # when the loop is finished (a conditional picks between the old array and the written one).
    # Returns a tuple of: the scalar "not all done" flag, the indices, the inputs and the new arrays.
    """Handles case when condition is unstacked.

    Note that all iterations end together. So we don't need to partition the
    inputs. When all iterations are done, we write the inputs to the
    TensorArrays. Note that we only write to index 0 of output_tas. Since all
    iterations end together, they can all be output together.
    """
    not_all_done = array_ops.reshape(conditions, [])
    new_output_tas = []
    # pylint: disable=cell-var-from-loop
    for i, out_ta in enumerate(output_tas):
      inp = inputs[i]
      new_output_tas.append(
          tf_cond.cond(not_all_done, lambda: out_ta,
                       lambda: out_ta.write(0, inp)))
    # pylint: enable=cell-var-from-loop
    return not_all_done, indices, inputs, new_output_tas

  def _process_cond_stacked(self, conditions, indices, inputs, inputs_stacked,
                            output_tas):
    # [FUNCTION DOC]
    # HELPER: handles an iteration step when the condition differs between iterations.
    # Inputs: conditions, indices, inputs, inputs_stacked, output_tas. Computes whether any iteration
    # continues, partitions the indices and each stacked input into finished and continuing parts
    # (unstacked inputs are stacked for the finished part), and scatters the finished values into the
    # corresponding output TensorArrays by iteration id.
    # Returns a tuple of the "not all done" flag, the new indices, the new inputs and the new arrays.
    num_outputs = len(self._outputs)
    # Compute if all iterations are done.
    not_all_done = math_ops.reduce_any(conditions)
    conditions_int = math_ops.cast(conditions, dtypes.int32)
    # Partition the indices.
    done_indices, new_indices = data_flow_ops.dynamic_partition(
        indices, conditions_int, 2)

    new_inputs = []
    new_output_tas = []
    for i, (inp, stacked) in enumerate(zip(inputs, inputs_stacked)):
      # Partition the inputs.
      if stacked:
        done_inp, new_inp = data_flow_ops.dynamic_partition(
            inp, conditions_int, 2)
      else:
        # TODO(agarwal): avoid this stacking. See TODO earlier in
        # _process_cond_unstacked.
        done_inp = _stack(inp, [array_ops.size(done_indices)]).t
        new_inp = inp
      new_inputs.append(new_inp)
      # For iterations that are done, write them to TensorArrays.
      if i < num_outputs:
        out_ta = output_tas[i]
        # Note that done_indices can be empty. done_inp should also be empty in
        # that case.
        new_output_tas.append(out_ta.scatter(done_indices, done_inp))
    return not_all_done, new_indices, new_inputs, new_output_tas

  def _process_body(
      self,
      pfor_input: "_PforInput",
      inputs_stacked,
      new_indices,
      cond_stacked,
      new_inputs,
      not_all_done,
  ):
    # [FUNCTION DOC]
    # HELPER: converts the body of the while loop for the still-active iterations.
    # Inputs: the pfor input, inputs_stacked, new_indices, cond_stacked, new_inputs, not_all_done.
    # Creates a body PFor, then for every body output builds a conditional: if not all iterations are
    # done, it converts the output (together with its NextIteration control dependencies, wrapped in an
    # identity); otherwise it returns an empty constant of the right dtype, since that value is unused.
    # Returns the list of new loop-variable values.
    """Convert the body function."""

    def true_fn(control_inputs, body_pfor, body_output, stacked):
      # [FUNCTION DOC]
      # Nested helper of the body processing: converts one body output when the loop continues.
      # Inputs: control_inputs (operations the NextIteration node depends on), body_pfor (the PFor for
      # the body), body_output (tensor to convert) and stacked (expected stackedness). Converts the
      # control-input outputs, converts the body output (forcing stacking if stacked is true; otherwise
      # asserting that the conversion stayed unstacked), and returns an identity of the result created
      # under the converted control dependencies.
      """Converts the body function for all but last iteration.

      This essentially converts body_output. Additionally, it needs to handle
      any control dependencies on the NextIteration node. So it creates another
      Identity node with the converted dependencies.
      """
      converted_control_inp = []
      for x in control_inputs:
        for t in x.outputs:
          converted_control_inp.append(body_pfor._convert_helper(t).t)
      if stacked:
        # Note convert always does the stacking.
        output = body_pfor.convert(body_output)
      else:
        output, convert_stacked, _ = body_pfor._convert_helper(body_output)
        assert convert_stacked == stacked, body_output
      with ops.control_dependencies(converted_control_inp):
        return array_ops.identity(output)

    body_pfor = self._init_pfor(pfor_input.pfor, new_indices, cond_stacked,
                                new_inputs, inputs_stacked)
    new_outputs = []

    for i, (body_output,
            stacked) in enumerate(zip(self._body_outputs, inputs_stacked)):
      control_inp = self._next_iter_control_inputs[i]
      out_dtype = body_output.dtype
      # Note that we want to run the body only if not all pfor iterations are
      # done. If all are done, we return empty tensors since these values will
      # not be used. Notice that the value returned by the loop is based on
      # TensorArrays and not directly on these returned values.
      # pylint: disable=cell-var-from-loop
      new_output = tf_cond.cond(
          not_all_done,
          lambda: true_fn(control_inp, body_pfor, body_output, stacked),
          lambda: constant_op.constant([], dtype=out_dtype))
      # pylint: enable=cell-var-from-loop
      new_outputs.append(new_output)
    return new_outputs

  def __call__(self, pfor_input: "_PforInput"):
    # [FUNCTION DOC]
    # Converts the whole while loop into a vectorized while loop (the converter entry point).
    # Input: the pfor input for any Exit node of the loop. Builds initial values, then runs a new
    # while loop whose arguments are: a "not all done" flag, the active iteration indices, the
    # loop-variable and direct-enter values, and one TensorArray per original output. Each iteration
    # reconverts the condition for active iterations, writes finished iterations to the TensorArrays,
    # partitions inputs if the condition is stacked, and converts the body. After the loop, outputs are
    # the stacked TensorArrays (condition stacked) or the element at index zero (condition unstacked,
    # stackedness following the corresponding input).
    # Returns a list of wrapped output tensors. Adds control-flow operations to the graph; asserts that
    # the condition's stackedness was determined.
    """Converter for the while_loop.

    The conversion of a while_loop is another while_loop.

    The arguments to this converted while_loop are as follows:
    not_all_done: Boolean scalar Tensor indicating if all the pfor iterations
      are done.
    indices: int32 1-D Tensor storing the id of the iterations that are not
      done.
    args: Remaining arguments. These can be divided into 3 categories:
      - First set of arguments are the tensors that correspond to the initial
        elements of self._enters. The elements that appear in original while
        loop's `loop_vars`.
      - The second set of arguments are the tensors that correspond to the
        remaining elements of self._enters. These are the tensors that directly
        enter the original while loop body.
       - Finally, the last set of arguments are TensorArrays. These TensorArrays
         correspond to the outputs of the original while_loop, i.e. to the
         elements in self._outputs. Each TensorArray has `PFor.loop_len`
         elements, i.e. the number of pfor iterations. At the end, the i'th
         element of each TensorArray will contain the output computed by the
         i'th iteration of pfor. Note that elements can be written into these
         tensors arrays in any order, depending on when the corresponding pfor
         iteration is done.
      If the original while_loop had `k` tensors in its `loop_vars` and its body
      directly captured `m` tensors, the `args` will contain `2 * k + m` values.

    In each iteration, the while_loop body recomputes the condition for all
    active pfor iterations to see which of them are now done. It then partitions
    all the inputs and passes them along to the converted body. Values for all
    the iterations that are done are written to TensorArrays indexed by the pfor
    iteration number. When all iterations are done, the TensorArrays are stacked
    to get the final value.

    Args:
      pfor_input: A PForInput object corresponding to the output of any Exit
        node from this while loop.

    Returns:
      List of converted outputs.
    """
    # Create init_values that will be passed to the while_loop.
    init_values, inputs_stacked, shape_invariants = self._create_init_values(
        pfor_input)
    # Note that we use a list as a hack since we need the nested function body
    # to set the value of cond_is_stacked. python2.x doesn't support nonlocal
    # variables.
    cond_is_stacked = [None]

    def cond(not_all_done, *_):
      # [FUNCTION DOC]
      # Nested loop condition for the generated while loop: returns its first argument, the "not all
      # done" flag; remaining arguments are ignored.
      return not_all_done

    def body(not_all_done, indices, *args):
      # [FUNCTION DOC]
      # Nested loop body for the generated while loop. Inputs: the not-all-done flag, the active indices
      # and the remaining arguments (inputs followed by output TensorArrays). Converts the loop
      # condition for the active iterations (assuming stacked, recording the real stackedness in an
      # enclosing list), processes finished iterations through the stacked or unstacked handler, converts
      # the loop body, and returns a tuple with the updated flag, indices, new loop-variable values, the
      # remaining direct-enter values and the updated TensorArrays.
      # See documentation for __call__ for the structure of *args.
      num_enters = len(self._enters)
      inputs = args[:num_enters]
      output_tas = args[num_enters:]
      # TODO(agarwal): see which outputs have consumers and only populate the
      # TensorArrays corresponding to those. Or do those paths get trimmed out
      # from inside the while_loop body?
      assert len(inputs) >= len(output_tas)
      assert len(inputs) == len(inputs_stacked)

      # Convert condition
      with ops.name_scope("while_cond"):
        # Note that we set cond_stacked to True here. At this point we don't
        # know if it could be loop invariant, hence the conservative value is
        # to assume stacked.
        cond_pfor = self._init_pfor(
            pfor_input.pfor,
            indices,
            cond_stacked=True,
            inputs=inputs,
            inputs_stacked=inputs_stacked)
        conditions, cond_stacked, _ = cond_pfor._convert_helper(self._condition)
        cond_is_stacked[0] = cond_stacked

      # Recompute the new condition, write outputs of done iterations, and
      # partition the inputs if needed.
      if not cond_stacked:
        (not_all_done, new_indices, new_inputs,
         new_output_tas) = self._process_cond_unstacked(conditions, indices,
                                                        inputs, output_tas)
      else:
        (not_all_done, new_indices, new_inputs,
         new_output_tas) = self._process_cond_stacked(conditions, indices,
                                                      inputs, inputs_stacked,
                                                      output_tas)

      # Convert body
      with ops.name_scope("while_body"):
        #  Compute the outputs from the body.
        new_outputs = self._process_body(pfor_input, inputs_stacked,
                                         new_indices, cond_stacked, new_inputs,
                                         not_all_done)

      # Note that the first num_outputs new values of inputs are computed using
      # the body. Rest of them were direct Enters into the condition/body and
      # the partitioning done earlier is sufficient to give the new value.
      num_outputs = len(self._outputs)
      new_args = ([not_all_done, new_indices] + new_outputs +
                  list(new_inputs[num_outputs:]) + new_output_tas)
      return tuple(new_args)

    while_outputs = while_loop.while_loop(
        cond, body, init_values, shape_invariants=shape_invariants)
    output_tas = while_outputs[-len(self._outputs):]
    outputs = []
    assert cond_is_stacked[0] is not None
    for inp_stacked, ta in zip(inputs_stacked, output_tas):
      if cond_is_stacked[0]:
        outputs.append(wrap(ta.stack(), True))
      else:
        # Note that if while_loop condition is unstacked, all iterations exit at
        # the same time and we wrote those outputs in index 0 of the tensor
        # array.
        outputs.append(wrap(ta.read(0), inp_stacked))
    return outputs


class ConversionNotImplementedError(Exception):
  # [CLASS DOC]
  # Exception raised by converter helper methods (stacked_input and unstacked_input) when an
  # operation's inputs do not have the loop-invariance pattern a converter supports. The driver
  # catches it and, when allowed, falls back to a sequential while loop; otherwise it re-raises it
  # as a value error.
  pass


class _PforInput:
  # [CLASS DOC]
  # Argument object passed to every registered pfor converter. It stores the PFor driver (public
  # attribute pfor), the operation being converted and the list of already converted inputs
  # (wrapped tensors). It offers accessors for inputs, outputs and attributes, assertions about
  # stackedness, and helpers to stack inputs or prepare them for broadcasting. Slots are used to keep
  # instances small.
  """Input object passed to registered pfor converters."""

  __slots__ = ["pfor", "_op", "_inputs"]

  def __init__(self, pfor: "PFor", op: ops.Operation, inputs):
    # [FUNCTION DOC]
    # Creates the converter argument object. Inputs: the PFor driver, the operation being converted and
    # the list of wrapped converted inputs. Only stores them; no graph changes.
    """Creates a _PforInput object.

    Args:
      pfor: PFor converter object.
      op: the Operation object that is being converted.
      inputs: list of WrappedTensor objects representing converted values of the
        inputs of `op`.
    """
    self.pfor = pfor
    self._op = op
    self._inputs = inputs

  def stack_inputs(self, stack_indices=None, tile_variants=False):
    # [FUNCTION DOC]
    # Stacks, in place, the loop-invariant inputs at the given positions.
    # Inputs: stack_indices (positions to process; null means all inputs) and tile_variants (default
    # false). Unstacked inputs are stacked using the loop length; if tile_variants is true, variant
    # inputs that were stacked are additionally tiled to a vectorized shape. If tile_variants is false
    # and an input is an already stacked variant it is un-tiled instead. Mutates the stored inputs
    # list; adds graph operations.
    """Stacks unstacked inputs at `stack_indices`.

    Args:
      stack_indices: indices of inputs at which stacking is done. If None,
        stacking is done at all indices.
      tile_variants: If True, affected indices which have a variant dtype will
        be tiled after this operation to match the expected shape of a
        vectorized tensor. Variants generally need to be un-tiled when they are
        inputs to operations and tiled when returned.
    """
    if stack_indices is None:
      stack_indices = range(len(self._inputs))
    length = self.pfor.loop_len_vector
    for i in stack_indices:
      inp = self._inputs[i]
      is_variant = inp.t.dtype == dtypes.variant
      if not inp.is_stacked:
        self._inputs[i] = _stack(inp.t, length)
        if tile_variants and is_variant:
          self._inputs[i] = wrap(
              _tile_variant_with_length(self._inputs[i].t, length), True)
      elif not tile_variants and is_variant:
        self._inputs[i] = wrap(_untile_variant(self._inputs[i].t), True)

  def expanddim_inputs_for_broadcast(self):
    # [FUNCTION DOC]
    # Reshapes stacked inputs, in place, so ordinary broadcasting works with the extra leading axis.
    # Does nothing for fewer than two inputs. Computes each input's rank counting the stack dimension
    # (unstacked inputs get one added), takes the maximum rank, and inserts size-one axes right after
    # the leading dimension of each stacked input of lower rank. Heavily tuned for statically known
    # ranks (plain integers), falling back to dynamic ranks only when needed.
    """Reshapes stacked inputs to prepare them for broadcast.

    Since stacked inputs have an extra leading dimension, automatic broadcasting
    rules could incorrectly try to expand dimensions before that leading
    dimension. To avoid that, we reshape these stacked inputs to the maximum
    rank they will need to be broadcasted to.

    IMPORTANT: This function is heavily optimized for statically known ranks
    because it's on the critical path of some huge training graphs.
    """
    if len(self._inputs) < 2:
      return

    ranks = [
        _rank(inp.t) if inp.is_stacked else (_rank(inp.t) + 1)
        for inp in self._inputs
    ]
    if all(isinstance(rank, int) for rank in ranks):
      max_rank = max(ranks)
    else:
      max_rank = functools.reduce(math_ops.maximum, ranks)

    for i, inp in enumerate(self._inputs):
      if not inp.is_stacked:
        continue
      if isinstance(max_rank, int) and ranks[i] == max_rank:
        continue
      self._inputs[i] = wrap(_expand_dims(inp.t, 1, max_rank - ranks[i]), True)

  @property
  def inputs(self):
    # [FUNCTION DOC]
    # Property: returns the list of wrapped converted inputs (the live internal list, not a copy).
    return self._inputs

  @property
  def num_inputs(self):
    # [FUNCTION DOC]
    # Property: number of inputs of the operation being converted.
    return len(self._inputs)

  def input(self, index):
    # [FUNCTION DOC]
    # Returns the wrapped converted input at the given index. Fails an assertion if the index is out of
    # range. Result is a wrapped tensor (tensor, stacked flag, sparse-stacked flag).
    assert len(self._inputs) > index, (index, self._inputs)
    return self._inputs[index]

  def stacked_input(self, index):
    # [FUNCTION DOC]
    # Returns the tensor of the input at an index, requiring it to be stacked.
    # Raises a conversion-not-implemented error naming the input (by definition name when known, else
    # by position) and the op type when the input is loop invariant.
    t, is_stacked, _ = self.input(index)
    if not is_stacked:
      op_type = self.op_type
      op_def = getattr(self._op, "op_def", None)
      if op_def is None:
        input_name = "at index %d" % index
      else:
        input_name = "\"%s\"" % op_def.input_arg[index].name
      raise ConversionNotImplementedError(
          f"Input {input_name} of op '{op_type}' expected to be not loop "
          "invariant.")
    return t

  def unstacked_input(self, index):
    # [FUNCTION DOC]
    # Returns the tensor of the input at an index, requiring it to be loop invariant.
    # Raises a conversion-not-implemented error naming the input and the op type when the input is
    # stacked.
    t, is_stacked, _ = self.input(index)
    if is_stacked:
      op_type = self.op_type
      op_def = getattr(self._op, "op_def", None)
      if op_def is None:
        input_name = "at index %d" % index
      else:
        input_name = "\"%s\"" % op_def.input_arg[index].name
      raise ConversionNotImplementedError(
          f"Input {input_name} of op '{op_type}' expected to be loop "
          "invariant.")
    return t

  @property
  def op(self) -> ops.Operation:
    # [FUNCTION DOC]
    # Property: the original (unconverted) operation being converted.
    return self._op

  @property
  def op_type(self):
    # [FUNCTION DOC]
    # Property: the type name of the operation being converted.
    return self._op.type

  def get_attr(self, attr):
    # [FUNCTION DOC]
    # Returns the value of the named attribute of the operation being converted.
    return self._op.get_attr(attr)

  @property
  def outputs(self):
    # [FUNCTION DOC]
    # Property: the output tensors of the original operation.
    return self._op.outputs

  def output(self, index):
    # [FUNCTION DOC]
    # Returns the original operation's output tensor at an index; assertion error if out of range.
    assert index < len(self._op.outputs)
    return self._op.outputs[index]


_pfor_converter_registry = {}


class RegisterPFor:
  # [CLASS DOC]
  # Decorator class that registers a converter function for one operation type in the global
  # converter registry. It is instantiated with an op type name and applied to a function taking a
  # _PforInput and returning wrapped tensors (a list matching the op outputs, or an operation for
  # zero-output ops). Several decorators can be stacked on one converter. The detailed contract is
  # described in the docstring below.
  """Utility to register converters for pfor.

  Usage:
  @RegisterPFor(foo_op_type)
  def _foo_converter(pfor_input: _PforInput):
    ...

  The above will register conversion function `_foo_converter` for handling
  conversion of `foo_op_type`. These converters are called during vectorization
  of a `pfor` loop body. For each operation node in this loop body,
  the vectorization process will call the converter corresponding to the
  operation type of the node.

  During conversion, the registered function will be called with a single
  argument `pfor_input`, of type `PForInput`, which will contain state needed
  for the conversion.  When the converter is called for a node, all its inputs
  should already have been converted and these converted values are stored in
  `pfor_input.inputs`.  This registered function should output a list of
  WrappedTensor objects with the same length as the number of outputs of the
  node being converted. If the node had zero outputs, then it should return an
  ops.Operation object.  These new sets of nodes should implement the
  functionality of running that operation for the number of iterations specified
  by `pfor_input.pfor.loop_len_vector[0]` where the inputs of the node for each
  iteration are picked from `pfor_inputs.inputs()`.

  One tricky aspect of the conversion process is keeping track of, and
  leveraging loop invariance of computation. Each converted input is a
  WrappedTensor which indicates whether the input was loop invariant or not. If
  the converted value is loop invariant, its rank should match the rank of the
  corresponding tensor in the loop body, else its rank is larger by 1. The
  converter should look at the loop invariance of the inputs and generate new
  nodes based on that. Note that the converter will not be called if all inputs
  are loop invariant and the operation is not stateful. The converter should
  determine if its own output is loop invariant and `wrap` its output
  accordingly.

  Example:

  Here, the converter is trying to convert a Reshape node in the loop body. This
  node will have two inputs: the tensor to reshape, and the new shape.  The
  example here only handles the case where the shape is loop invariant.

  @RegisterPFor("Reshape")
  def _convert_reshape(pfor_input: _PforInput):
    # We assume that input is not loop invariant. Call to `stacked_input`
    # asserts that and returns the converted value. This value will have a rank
    # larger by 1 compared to the rank of the input in the loop body.
    t = pfor_input.stacked_input(0)

    # We assume that shape input is loop invariant. Call to `unstacked_input`
    # asserts that and returns the converted value.
    shape = pfor_input.unstacked_input(1)

    # We compute `new_shape` by prepending the number of iterations to the
    # original shape.
    new_shape = array_ops.concat([pfor_input.pfor.loop_len_vector, shape],
                                 axis=0)

    # The vectorized output involves reshaping the converted input `t` using
    # `new_shape`.
    new_output = array_ops.reshape(t, new_shape)

    # The converted output is marked as not loop invariant using the call to
    # wrap.
    return wrap(new_output, True)
  """

  def __init__(self, op_type):
    # [FUNCTION DOC]
    # Stores the operation type name for which a converter will be registered.
    """Creates an object to register a converter for op with type `op_type`."""
    self.op_type = op_type

  def __call__(self, converter):
    # [FUNCTION DOC]
    # Registers the decorated converter under the stored op type in the global registry and returns the
    # converter unchanged. Raises an assertion error if that op type is already registered.
    # Side effect: mutates the module-level registry.
    name = self.op_type
    assert name not in _pfor_converter_registry, "Re-registering %s " % name
    _pfor_converter_registry[name] = converter
    return converter


class RegisterPForWithArgs(RegisterPFor):
  # [CLASS DOC]
  # Variant of RegisterPFor that binds extra positional and keyword arguments to the converter at
  # registration time. The converter receives the pfor input, then the op type name, then the bound
  # arguments. Used to share one converter body among several ops, parameterised by an op function.
  """Utility to register converters for pfor.

  Usage:
  @RegisteRPFor(foo_op_type, foo=value, ....)
  def _foo_converter(pfor_input, foo=None, ....):
    ...

  See RegisterPFor for details on the conversion function.
  `RegisterPForWithArgs` allows binding extra arguments to the
  conversion function at registration time.
  """

  def __init__(self, op_type, *args, **kw_args):
    # [FUNCTION DOC]
    # Stores the op type (through the parent constructor) and the extra positional and keyword
    # arguments that will be passed to the converter when it is invoked.
    super(RegisterPForWithArgs, self).__init__(op_type)
    self._args = args
    self._kw_args = kw_args

  def __call__(self, converter):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the class docstring example spells the decorator name with odd capitalisation
    # (a typo); the registration itself works through this method.
    # Registers a small wrapper in the registry (via the parent) that calls the converter with the pfor
    # input, the op type and the bound arguments. Returns the original converter, not the wrapper.

    def _f(pfor_input: _PforInput):
      # [FUNCTION DOC]
      # Nested wrapper stored in the registry: takes a pfor input and calls the user converter with it,
      # the op type and the bound extra arguments; returns whatever the converter returns.
      return converter(pfor_input, self.op_type, *self._args, **self._kw_args)

    super(RegisterPForWithArgs, self).__call__(_f)
    return converter


# TODO(agarwal): call raw_ops instead of calling these low level routines.
def _create_op(op_type, inputs, op_dtypes, attrs=None):
  # [FUNCTION DOC]
  # HELPER: low-level op creation used by many converters.
  # KNOWN ISSUE: the attrs parameter defaults to null, but the body iterates over it unconditionally,
  # so calling without attrs raises a type error.
  # Inputs: op type name, input tensors, output dtypes, and attrs (mapping of attribute names to
  # attribute values). Creates the op in the default graph on the computing device, then records it
  # on the gradient tape using a flat alternating list of attribute names and values. Returns the new
  # operation.
  """Utility to create an op."""
  op = ops.get_default_graph().create_op(
      op_type, inputs, op_dtypes, attrs=attrs, compute_device=True)
  flat_attrs = []
  # The tape expects an alternating flat list of names and attribute values.
  for a in attrs:
    flat_attrs.append(str(a))
    flat_attrs.append(op.get_attr(str(a)))
  execute.record_gradient(op_type, op.inputs, tuple(flat_attrs), op.outputs[:])
  return op


WrappedTensor = collections.namedtuple("WrappedTensor",
                                       ["t", "is_stacked", "is_sparse_stacked"])
"""Wrapper around the result of a Tensor conversion.

The additional fields are useful for keeping track of the conversion state as
data flows through the ops in the loop body. For every op whose output is a
Tensor, its converter should return either a WrappedTensor or a list of
WrappedTensors.

Args:
  t: The converted tensor
  is_stacked: True if the tensor is stacked, i.e. represents the results of all
    the iterations of the loop, where each row i of the tensor corresponds to
    that op's output on iteration i of the loop. False if the tensor is not
    stacked, i.e. represents the result of the op on of a single iteration of
    the loop, where the result does not vary between iterations.
  is_sparse_stacked: True if the tensor corresponds to a component tensor
    (indices, values, or dense_shape) of a sparse tensor, and has been logically
    stacked via a sparse conversion.
"""


def wrap(tensor, is_stacked=True, is_sparse_stacked=False):
  # [FUNCTION DOC]
  # HELPER: constructor for WrappedTensor values returned by converters.
  # Inputs: a tensor, is_stacked (default true) and is_sparse_stacked (default false). Asserts the
  # flags are booleans, the value is a tensor, and that sparse-stacked implies stacked. Returns a
  # WrappedTensor tuple.
  """Helper to create a WrappedTensor object."""
  assert isinstance(is_stacked, bool)
  assert isinstance(is_sparse_stacked, bool)
  assert isinstance(tensor, tensor_lib.Tensor), type(tensor)
  assert not is_sparse_stacked or is_stacked, ("If the wrapped tensor is "
                                               "stacked via a sparse "
                                               "conversion, it must also be "
                                               "stacked.")
  return WrappedTensor(tensor, is_stacked, is_sparse_stacked)


def _wrap_and_tile_variants(tensor, length):
  # [FUNCTION DOC]
  # HELPER: wraps a result as stacked, first tiling it to the loop length if it is a variant.
  # Inputs: a tensor and the loop-length vector. Returns a stacked wrapped tensor.
  if tensor.dtype == dtypes.variant:
    tensor = _tile_variant_with_length(tensor, length)
  return wrap(tensor)


def _fallback_converter(pfor_input: _PforInput, root_cause="", warn=False):
  # [FUNCTION DOC]
  # Generic fallback converter: runs the original op sequentially in a while loop, one iteration at a
  # time, and stacks the results.
  # Inputs: pfor_input, root_cause (text explaining why vectorization was not possible, default
  # empty) and warn (if true log a warning, otherwise a debug message). Determines the iteration
  # count (statically if known), indexes stacked inputs per iteration, recreates the op with the
  # original attributes inside a while loop and writes each output to a TensorArray.
  # Returns a tuple of stacked wrapped outputs. Side effects: logging, graph operations.
  msg = ("Using a while_loop for converting "
         f"{pfor_input.op_type} cause {root_cause}")
  if warn:
    logging.warning(msg)
  else:
    logging.debug(msg)
  output_dtypes = [x.dtype for x in pfor_input.outputs]
  iter_vec = pfor_input.pfor.loop_len_vector
  # Use constant value if available, so that output shapes are static.
  iter_vec_value = tensor_util.constant_value(iter_vec)
  if iter_vec_value is not None:
    iters = iter_vec_value[0].item()
  else:
    iters = iter_vec[0]

  def while_body(i, *ta_list):
    # [FUNCTION DOC]
    # Nested loop body of the fallback converter. Inputs: iteration counter and one TensorArray per
    # output. Slices the stacked inputs at the counter, creates the original op on them, writes every
    # output to its array at the counter and returns the incremented counter with the arrays.
    """Body of while loop."""
    inputs = [
        x[i, ...] if stacked else x for x, stacked, _ in pfor_input.inputs
    ]
    op_outputs = _create_op(
        pfor_input.op_type,
        inputs,
        output_dtypes,
        attrs=pfor_input.op.node_def.attr).outputs

    outputs = []
    # TODO(agarwal): Add tf.debugging asserts to check that the shapes across
    # the different iterations are the same.
    for out, ta in zip(op_outputs, ta_list):
      assert isinstance(out, tensor_lib.Tensor)
      outputs.append(ta.write(i, out))
    return tuple([i + 1] + outputs)

  ta_list = while_loop.while_loop(
      lambda i, *ta: i < iters, while_body, [0] +
      [tensor_array_ops.TensorArray(dtype, iters) for dtype in output_dtypes
      ])[1:]
  return tuple([wrap(ta.stack(), True) for ta in ta_list])


class PForConfig:
  # [CLASS DOC]
  # Configuration object shared between the user's loop function and the pfor converter. It holds the
  # number of iterations when statically known and a map from reduction placeholder operations to
  # the pair of (traced reduction function, original arguments). Loop bodies call reduce and its
  # convenience variants to request computations over all iterations; the PFor driver later looks
  # them up while converting.
  """A configuration object used to communicate with loop body function."""

  def __init__(self):
    # [FUNCTION DOC]
    # Initialises the configuration with an unknown iteration count and an empty reduction map.
    # This may be set to the number of iterations.
    self._maybe_iters = None
    # Map from reduction node, created by `reduce`, to the bundle of reduction
    # function and arguments.
    self._reduce_map = {}

  def _has_reductions(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: documented as returning true/false but actually returns the number of recorded
    # reductions (an integer), which is merely truthy or falsy.
    # Reports whether the loop body registered any reductions.
    """True if some reductions where performed by loop body."""
    return len(self._reduce_map)

  def _set_iters(self, iters):
    # [FUNCTION DOC]
    # Records the number of pfor iterations. Input: an integer or a scalar tensor (converted to a
    # constant value if possible, else null). Mutates the stored iteration count.
    """Set number of pfor iterations."""
    if isinstance(iters, tensor_lib.Tensor):
      iters = tensor_util.constant_value(iters)
    self._maybe_iters = iters

  def reduce(self, fn, *args):
    # [FUNCTION DOC]
    # Registers a reduction over all pfor iterations and returns a placeholder-like result.
    # Inputs: fn (a function of tensors with one extra leading iteration dimension, returning tensors)
    # and any number of unvectorized argument tensors. Asserts graph mode. Each argument must be a
    # tensor (value error otherwise); fn is traced once into a concrete function using specs with the
    # iteration count prepended to each argument shape. Placeholder (with default, or plain placeholder
    # when the shape is not fully defined) outputs are created under control dependencies on the
    # arguments and joined by an identity-n operation; that operation is recorded in the reduction map
    # together with the function and arguments. Raises a value error if fn returns non-tensors.
    # Returns the single output tensor or a tuple of them; the results are loop invariant. May not work
    # inside control flow constructs.
    """Performs reduction `fn` on `args` vectorized across pfor iterations.

    Note that `fn` is traced once inside the loop function context. Hence any
    captures or side-effects will happen in that context. Call to the traced
    version of `fn` happens during the construction of the vectorized code.

    Note that this currently may not work inside a control flow construct.
    Args:
      fn: a reduction function. It will be called with arguments that have the
        same structure as *args but with individual values whose rank may be
        higher by 1 since they represent loop invariant vectorized versions of
        the corresponding Tensors in *args.
      *args: unvectorized Tensors.

    Returns:
      The result of running `fn` on the vectorized versions of `*args`. These
      outputs will be available as loop invariant values to all the iterations.
    """
    assert not context.executing_eagerly()
    # Creates a concrete function that will be used for reduction.
    tensor_specs = []
    for arg in args:
      if not isinstance(arg, tensor_lib.Tensor):
        raise ValueError(f"Got a non-Tensor argument {arg} in reduce.")
      batched_shape = tensor_shape.TensorShape([self._maybe_iters
                                               ]).concatenate(arg.shape)
      tensor_specs.append(
          tensor_lib.TensorSpec(shape=batched_shape, dtype=arg.dtype))
    concrete_function = def_function.function(fn).get_concrete_function(
        *tensor_specs)

    # Creates PlaceholderWithDefault and IdentityN nodes corresponding the
    # reduction.
    pl_outputs = []
    with ops.control_dependencies(args):
      for output in concrete_function.outputs:
        if not isinstance(output, tensor_lib.Tensor):
          raise ValueError(f"Got a non-Tensor output {output} while running "
                           "reduce.")
        # Note that we use placeholder_with_default just to make XLA happy since
        # it does not like placeholder ops.
        if output.shape.is_fully_defined():
          dummy = array_ops.zeros(output.shape.as_list(), dtype=output.dtype)
          pl_outputs.append(
              array_ops.placeholder_with_default(dummy, shape=output.shape))
        else:
          # TODO(agarwal): support case when under XLA and output.shape is not
          # fully defined.
          pl_outputs.append(
              array_ops.placeholder(output.dtype, shape=output.shape))

      reduction_op = array_ops.identity_n(pl_outputs)[0].op
    self._reduce_map[reduction_op] = (concrete_function, args)
    if len(reduction_op.outputs) == 1:
      return reduction_op.outputs[0]
    else:
      return tuple(reduction_op.outputs)

  # TODO(agarwal): handle reductions inside control flow constructs.
  def reduce_concat(self, x):
    # [FUNCTION DOC]
    # Reduction that stacks the value of the argument across all iterations; returns a tensor with one
    # more dimension. Implemented through reduce with an identity function. Argument: unvectorized
    # tensor.
    """Performs a concat reduction on `x` across pfor iterations.

    Note that this currently may not work inside a control flow construct.
    Args:
      x: an unvectorized Tensor.

    Returns:
      A Tensor that has rank one higher than `x`. The value is the vectorized
      version of `x`, i.e. stacking the value of `x` across different pfor
      iterations.
    """
    return self.reduce(lambda y: y, x)

  def reduce_mean(self, x):
    # [FUNCTION DOC]
    # Reduction that averages the argument over the iteration axis; result has the same rank as the
    # argument. Implemented through reduce with a mean along the first axis.
    """Performs a mean reduction on `x` across pfor iterations.

    Note that this currently may not work inside a control flow construct.
    Args:
      x: an unvectorized Tensor.

    Returns:
      A Tensor that has same rank as `x`. The value is the mean of the values
      of `x` across the pfor iterations.
    """
    return self.reduce(lambda y: math_ops.reduce_mean(y, axis=0), x)

  def reduce_sum(self, x):
    # [FUNCTION DOC]
    # Reduction that sums the argument over the iteration axis; result has the same rank as the
    # argument. Implemented through reduce with a sum along the first axis.
    """Performs a sum reduction on `x` across pfor iterations.

    Note that this currently may not work inside a control flow construct.
    Args:
      x: an unvectorized Tensor.

    Returns:
      A Tensor that has same rank as `x`. The value is the sum of the values
      of `x` across the pfor iterations.
    """
    return self.reduce(lambda y: math_ops.reduce_sum(y, axis=0), x)

  def _lookup_reduction(self, t):
    # [FUNCTION DOC]
    # HELPER: finds the reduction registered for a tensor. Input: a tensor (assertion error otherwise).
    # Returns the stored pair of function and arguments for the tensor's producing op, or null.
    """Lookups Tensor `t` in the reduction maps."""
    assert isinstance(t, tensor_lib.Tensor), t
    return self._reduce_map.get(t.op)


class PFor:
  # [CLASS DOC]
  # Driver that rewrites a loop body into vectorized operations. It keeps the loop variable
  # placeholder, the loop length vector, the set (and ids) of body ops, the conversion map from
  # original tensors/ops to wrapped converted values, the active iteration ids (all_indices, possibly
  # partitioned inside control flow), the while-loop fallback flag, a warning flag and the optional
  # PForConfig. The convert method returns the vectorized equivalent of a tensor; internally a
  # greedy, iterative traversal converts each op by reusing it as is, cloning it with converted
  # control dependencies, or invoking its registered converter. Existing operations are never
  # modified; only new ones are added.
  """Implementation of rewrite of parallel-for loops.

  This class takes a DAG or a set of DAGs representing the body of a
  parallel-for loop, and adds new operations to the graph that implements
  functionality equivalent to running that loop body for a specified number of
  iterations. This new set of nodes may or may not use a tensorflow loop
  construct.

  The process of conversion does not delete or change any existing operations.
  It only adds operations that efficiently implement the equivalent
  functionality. We refer to the added ops as "converted ops".

  The conversion process uses a simple greedy heuristic. It walks the loop body
  and tries to express the functionality of running each node in a loop with a
  new set of nodes. When converting an op several cases are possible:
  - The op is not inside the loop body. Hence it can be used as is.
  - The op does not depend on the iteration number and is stateless. In this
    case, it can be used as is.
  - The op is not stateful, and depends on iteration number only through control
    dependencies. In this case, we can create a single op with same inputs and
    attributes, but with "converted" control dependencies.
  - The op is not stateful, and all its inputs are loop invariant. In this
    case, similar to above, we can create a single op with same inputs and
    attributes, but with "converted" control dependencies.
  - The op is stateful or at least one of the inputs is not loop invariant. In
    this case, we run the registered converter for that op to create a set of
    converted ops. All nodes in the set will have converted control dependencies
    corresponding to control dependencies of the original op. If the op returned
    multiple outputs, "converted outputs" could be produced by different ops in
    this set.
  """

  def __init__(self,
               loop_var,
               loop_len,
               pfor_ops,
               fallback_to_while_loop,
               all_indices=None,
               all_indices_partitioned=False,
               pfor_config=None,
               warn=False):
    # [FUNCTION DOC]
    # Creates the converter for one pfor loop.
    # Inputs: loop_var (output of the placeholder holding the int32 iteration number), loop_len
    # (scalar or scalar tensor), pfor_ops (all ops in the body), fallback_to_while_loop (use sequential
    # loops when no vectorization exists), all_indices (optional sorted unique int32 ids of active
    # iterations; defaults to a range over the loop length), all_indices_partitioned (true when only
    # some iterations may be active; requires all_indices), pfor_config and warn (warn on while
    # fallbacks). Asserts that loop_var comes from a placeholder-with-default op. Initialises the
    # conversion map with loop_var mapped to the stacked indices.
    """Creates an object to rewrite a parallel-for loop.

    Args:
      loop_var: Tensor output of a Placeholder operation. The value should
        be an int32 scalar representing the loop iteration number.
      loop_len: A scalar or scalar Tensor representing the number of iterations
        the loop is run for.
      pfor_ops: List of all ops inside the loop body.
      fallback_to_while_loop: If True, on failure to vectorize an op, a while
        loop is used to sequentially execute that op.
      all_indices: If not None, an int32 vector with size `loop_len`
        representing the iteration ids that are still active. These values
        should be unique and sorted. However they may not be contiguous. This is
        typically the case when inside a control flow construct which has
        partitioned the indices of the iterations that are being converted.
      all_indices_partitioned: If True, this object is being constructed from a
        control flow construct where not all the pfor iterations are guaranteed
        to be active.
      pfor_config: PForConfig object used while constructing the loop body.
      warn: Whether or not to warn on while loop conversions.
    """
    assert isinstance(loop_var, tensor_lib.Tensor)
    assert loop_var.op.type == "PlaceholderWithDefault"
    self._loop_var = loop_var
    loop_len_value = tensor_util.constant_value(loop_len)
    if loop_len_value is not None:
      loop_len = loop_len_value
      self._loop_len_vector = ops.convert_to_tensor([loop_len])
    else:
      self._loop_len_vector = array_ops.reshape(loop_len, [1])
    self._all_indices_partitioned = all_indices_partitioned
    if all_indices_partitioned:
      assert all_indices is not None
    if all_indices is None:
      self.all_indices = math_ops.range(
          loop_len, dtype=dtypes.int32, name="all_indices"
      )
    else:
      self.all_indices = all_indices

    self._conversion_map = object_identity.ObjectIdentityDictionary()
    self._conversion_map[loop_var] = wrap(self.all_indices, True)
    self._pfor_ops = set(pfor_ops)
    self._pfor_op_ids = set(x._id for x in pfor_ops)
    self._fallback_to_while_loop = fallback_to_while_loop
    self._warn = warn
    self._pfor_config = pfor_config

  def op_is_inside_loop(self, op):
    # [FUNCTION DOC]
    # Returns true if the given operation (assertion error otherwise) belongs to the pfor loop body,
    # comparing operation ids because wrappers may differ for the same node.
    """True if op was created inside the pfor loop body."""
    assert isinstance(op, ops.Operation)
    # Note that we use self._pfor_op_ids for the check and not self._pfor_ops
    # since it appears there tensorflow API could return different python
    # objects representing the same Operation node.
    return op._id in self._pfor_op_ids

  def _convert_sparse(self, y):
    # [FUNCTION DOC]
    # HELPER: converts a sparse tensor by logically stacking its components.
    # Input: a sparse tensor from the loop body. Converts indices, values and dense shape. If all are
    # stacked via sparse conversion it rebuilds a sparse tensor from them directly; if none is, it
    # tiles/unwraps each component and rebatches through serialization (see the restack helper).
    # Mixed cases fail an assertion. Returns a sparse tensor whose dense shape has the loop length
    # prepended and whose other dimensions are the maxima over iterations.
    """Returns the converted value corresponding to SparseTensor y.

    For SparseTensors, instead of stacking the component tensors separately,
    resulting in component tensors with shapes (N, m, rank), (N, m), and (N,
    rank) respectively for indices, values, and dense_shape (where N is the loop
    length and m is the number of sparse tensor values per loop iter), we want
    to logically stack the SparseTensors, to create a SparseTensor whose
    components are size (N * m, rank + 1), (N * m, ), and (rank + 1,)
    respectively.

    Here, we try to get the conversion of each component tensor.
    If the tensors are stacked via a sparse conversion, return the resulting
    SparseTensor composed of the converted components. Otherwise, the component
    tensors are either unstacked or stacked naively. In the latter case, we
    unstack the component tensors to reform loop_len SparseTensor elements,
    then correctly batch them.

    The unstacked tensors must have the same rank. Each dimension of each
    SparseTensor will expand to be the largest among all SparseTensor elements
    for that dimension. For example, if there are N SparseTensors of rank 3
    being stacked, with N dense shapes, where the i_th shape is (x_i, y_i, z_i),
    the new dense shape will be (N, max_i(x_i), max_i(y_i), max_i(z_i)).

    Args:
      y: A tf.sparse.SparseTensor.

    Returns:
      A tf.sparse.SparseTensor that is the converted value corresponding to y.
    """
    outputs = [
        self._convert_helper(t) for t in (y.indices, y.values, y.dense_shape)
    ]
    assert all(isinstance(o, WrappedTensor) for o in outputs)

    if all(w.is_sparse_stacked for w in outputs):
      return sparse_tensor.SparseTensor(*[w.t for w in outputs])

    assert not any(w.is_sparse_stacked for w in outputs), (
        "Error converting SparseTensor. All components should be logically "
        "stacked, or none.")

    # If component tensors were not sparsely stacked, they are either unstacked
    # or stacked without knowledge that they are components of sparse tensors.
    # In this case, we have to restack them.
    return self._restack_sparse_tensor_logically(
        *[self._unwrap_or_tile(w) for w in outputs])

  def _restack_sparse_tensor_logically(self, indices, values, shape):
    # [FUNCTION DOC]
    # HELPER: rebuilds one batched sparse tensor from per-iteration components.
    # Inputs: indices, values and dense shape, each with the leading iteration dimension. Serializes
    # each iteration's sparse tensor to a variant with a map function, then deserializes the whole
    # batch with a rank one higher than the original (when known). Returns a sparse tensor.
    sparse_tensor_rank = indices.get_shape().dims[-1].value
    if sparse_tensor_rank is not None:
      sparse_tensor_rank += 1

    def fn(args):
      # [FUNCTION DOC]
      # Nested map function: serializes the sparse tensor given by three components (indices, values,
      # dense shape) into a variant and returns it.
      res = gen_sparse_ops.serialize_sparse(
          args[0], args[1], args[2], out_type=dtypes.variant)
      return res

    # Applies a map function to the component tensors to serialize each
    # sparse tensor element and batch them all, then deserializes the batch.
    # TODO(rachelim): Try to do this without map_fn -- add the right offsets
    # to shape and indices tensors instead.
    result = map_fn.map_fn(fn, [indices, values, shape], dtype=dtypes.variant)
    return sparse_ops.deserialize_sparse(
        result, dtype=values.dtype, rank=sparse_tensor_rank)

  def _unwrap_or_tile(self, wrapped_tensor):
    # [FUNCTION DOC]
    # HELPER: returns the raw tensor of a wrapped value, stacking it to the loop length first if it
    # is not already stacked.
    """Given a wrapped tensor, unwrap if stacked. Otherwise, tiles it."""
    output, is_stacked = wrapped_tensor.t, wrapped_tensor.is_stacked
    if is_stacked:
      return output
    else:
      return _stack(output, self._loop_len_vector).t

  def convert(self, y):
    # [FUNCTION DOC]
    # Public entry point: returns the converted (vectorized) value of a tensor, sparse tensor or
    # operation.
    # Input: null (returns null), a sparse tensor (converted logically), a tensor or an operation
    # without outputs (anything else fails an assertion). Tensors always come back stacked (loop
    # invariants are tiled); operations come back as the converted operation.
    """Returns the converted value corresponding to y.

    Args:
      y: A Tensor or a ops.Operation object. If latter, y should not have
        any outputs.

    Returns:
      If y does not need to be converted, it returns y as is. Else it returns
      the "converted value" corresponding to y.
    """
    if y is None:
      return None
    if isinstance(y, sparse_tensor.SparseTensor):
      return self._convert_sparse(y)
    assert isinstance(y, (tensor_lib.Tensor, ops.Operation)), y
    output = self._convert_helper(y)
    if isinstance(output, WrappedTensor):
      assert isinstance(y, tensor_lib.Tensor)
      return self._unwrap_or_tile(output)
    else:
      assert isinstance(y, ops.Operation)
      assert not y.outputs
      assert isinstance(output, ops.Operation)
    return output

  def _was_converted(self, t):
    # [FUNCTION DOC]
    # HELPER: true if the conversion recorded for a tensor is a different tensor than the original.
    """True if t is not a conversion of itself."""
    converted_t = self._conversion_map[t]
    return converted_t.t is not t

  def _add_conversion(self, old_output, new_output):
    # [FUNCTION DOC]
    # HELPER: records a conversion in the map. Inputs: an old tensor or operation and its new wrapped
    # tensor or operation (types asserted). Mutates the conversion map.
    assert isinstance(
        old_output, (tensor_lib.Tensor, ops.Operation)), old_output
    assert isinstance(new_output, (WrappedTensor, ops.Operation)), new_output
    self._conversion_map[old_output] = new_output

  def _convert_reduction(self, y):
    # [FUNCTION DOC]
    # HELPER: converts a tensor that is a placeholder for a reduction registered in the config.
    # Input: a tensor or operation. Returns null when there is no config, the argument is an
    # operation, or no reduction is registered. Otherwise looks up the (function, arguments) pair,
    # requires every argument to be already converted (assertion with a message about control flow
    # constructs otherwise), unwraps or tiles them, calls the function and returns a list of wrapped,
    # loop-invariant results.
    # Handle reductions.
    if self._pfor_config is None or isinstance(y, ops.Operation):
      return None
    reduction = self._pfor_config._lookup_reduction(y)
    if reduction is None:
      return None
    (reduction_fn, reduction_args) = reduction
    batched_args = []
    for reduction_arg in reduction_args:
      assert isinstance(reduction_arg, tensor_lib.Tensor), reduction_arg
      # Tensor being reduced should already be converted due to a control
      # dependency on the created placeholder.
      # Note that in cases where reduction_arg is in an outer context, one
      # needs to locate the corresponding Enter node and use that to lookup
      # the conversion.
      # TODO(agarwal): handle reductions inside control flow constructs.
      assert reduction_arg in self._conversion_map, (
          "Unable to handle reduction of %s, possibly as it was used "
          "inside a control flow construct. Note that reductions across "
          "pfor iterations are currently not supported inside control flow "
          "constructs." % reduction_arg)
      batched_arg = self._conversion_map[reduction_arg]
      batched_args.append(self._unwrap_or_tile(batched_arg))
    outputs = reduction_fn(*batched_args)
    return [wrap(output, False) for output in nest.flatten(outputs)]

  def _convert_helper(self, op_or_tensor):
    # [FUNCTION DOC]
    # Core iterative conversion of a tensor or operation and everything it depends on.
    # Input: a tensor or an operation without outputs. Uses an explicit stack. For each item: skip if
    # already converted; handle Exit tensors by wrapping their while loop in a WhileOp when created
    # inside the body; push unconverted inputs and control inputs; then decide how to convert: reduction
    # placeholder; reuse as is (outside the body, or stateless with nothing converted); recreate the op
    # with converted control inputs (nothing stacked); or call the registered converter (the op's own
    # converter, or the global registry, or the sequential fallback). Raises a value error when no
    # converter exists and fallback is not allowed or not possible (variant outputs or vectorized
    # variant inputs), and converts conversion-not-implemented errors into fallbacks or value errors.
    # Other converter errors are logged with op and creation stack traces and re-raised. Output
    # shapes get the loop length prepended when stacked, with special handling for variants.
    # Returns the converted wrapped tensor(s) or operation, which are also cached in the map.
    stack = collections.deque([op_or_tensor])
    while stack:
      y = stack[0]
      if y in self._conversion_map:
        assert isinstance(self._conversion_map[y],
                          (WrappedTensor, ops.Operation))
        stack.popleft()
        continue
      if isinstance(y, ops.Operation):
        assert not y.outputs, (
            "We only support converting Operation objects with no outputs. "
            "Got %s", y)
        y_op = y
      else:
        assert isinstance(y, tensor_lib.Tensor), y
        y_op = y.op

      is_while_loop = y_op.type == "Exit"
      if is_while_loop:
        while_op = WhileOp(
            y, pfor_ops=self._pfor_ops,
            fallback_to_while_loop=self.fallback_to_while_loop,
            pfor_config=self._pfor_config)
        is_inside_loop = while_op.is_inside_loop
        # If all nodes in the while_loop graph were created inside the pfor, we
        # treat the whole loop subgraph as a single op (y_op) and try to convert
        # it. For while_loops that are created completely or partially outside,
        # we treat them as external and should be able to simply return the Exit
        # node output as is without needing any conversion. Note that for
        # while_loops that are partially constructed inside, we assume they will
        # be loop invariant. If that is not the case, it will create runtime
        # errors since the converted graph would depend on the self._loop_var
        # placeholder.
        if is_inside_loop:
          y_op = while_op
      else:
        is_inside_loop = self.op_is_inside_loop(y_op)

      # If this op was not created inside the loop body, we will return as is.
      # 1. Convert inputs and control inputs.

      def _add_to_stack(x):
        # [FUNCTION DOC]
        # Nested helper: pushes a tensor/operation to the front of the work stack if it is not yet
        # converted. Returns true if it was pushed, false if it was already converted.
        if x not in self._conversion_map:
          stack.appendleft(x)
          return True
        else:
          return False

      if is_inside_loop:
        added_to_stack = False
        for inp in y_op.inputs:
          added_to_stack |= _add_to_stack(inp)
        for cinp in y_op.control_inputs:
          if cinp.outputs:
            for t in cinp.outputs:
              added_to_stack |= _add_to_stack(t)
          else:
            added_to_stack |= _add_to_stack(cinp)
        if added_to_stack:
          continue

        converted_inputs = [self._conversion_map[inp] for inp in y_op.inputs]
        some_input_converted = any(self._was_converted(x) for x in y_op.inputs)
        some_input_stacked = any(x.is_stacked for x in converted_inputs)

        converted_control_ops = set()
        some_control_input_converted = False
        for cinp in y_op.control_inputs:
          if cinp.outputs:
            for t in cinp.outputs:
              converted_t = self._conversion_map[t]
              if self._was_converted(t):
                some_control_input_converted = True
              converted_control_ops.add(converted_t.t.op)
          else:
            converted_cinp = self._conversion_map[cinp]
            assert isinstance(converted_cinp, ops.Operation)
            if converted_cinp != cinp:
              some_control_input_converted = True
            converted_control_ops.add(converted_cinp)
        converted_control_ops = list(converted_control_ops)
        is_stateful = _is_stateful_pfor_op(y_op)
      else:
        converted_inputs = []
        converted_control_ops = []
      logging.vlog(3, "converting op:%s\ninputs:%s\ncontrol_inputs:%s", y_op,
                   converted_inputs, converted_control_ops)

      # 2. Convert y_op
      # If converting a while_loop, we let the while_loop convertor deal with
      # putting the control dependencies appropriately.
      control_dependencies = [] if is_while_loop else converted_control_ops
      with ops.control_dependencies(control_dependencies), ops.name_scope(
          y_op.name + "/pfor/"), ops.get_default_graph()._original_op(y_op):
        # Op is a placeholder for a reduction.
        reduce_output = self._convert_reduction(y)
        if reduce_output is not None:
          new_outputs = reduce_output
        # None of the inputs and control inputs were converted.
        elif ((not is_inside_loop or
               (not is_stateful and not some_input_converted and
                not some_control_input_converted)) and
              y.graph == ops.get_default_graph()):
          if y is y_op:
            assert not isinstance(y_op, WhileOp)
            new_outputs = y_op
          else:
            new_outputs = [wrap(x, False) for x in y_op.outputs]
        elif not (is_stateful or is_while_loop or some_input_stacked):
          # All inputs are unstacked or unconverted but some control inputs are
          # converted.
          # TODO(rachelim): Handle the case where some inputs are sparsely
          # stacked (i.e. any(x.is_sparse_stacked for x in converted_inputs))
          new_op = _create_op(y_op.type, [x.t for x in converted_inputs],
                              [x.dtype for x in y_op.outputs],
                              y_op.node_def.attr)
          if y is y_op:
            new_outputs = new_op
          else:
            new_outputs = []
            for old_output, new_output in zip(y_op.outputs, new_op.outputs):
              handle_data_util.copy_handle_data(old_output, new_output)
              new_outputs.append(wrap(new_output, False))
        else:
          # Either some inputs are not loop invariant or op is stateful.
          if hasattr(y_op, "pfor_converter"):
            converter = y_op.pfor_converter
          else:
            converter = _pfor_converter_registry.get(y_op.type, None)
          if converter is None:
            root_cause = "there is no registered converter for this op."
            has_variant_outputs = any(x.dtype == dtypes.variant for x in
                                      y_op.outputs)
            has_vectorized_variant_inputs = any(
                _is_variant_with_internal_stacking(x) for x in
                y_op.inputs)
            if (self._fallback_to_while_loop and not has_variant_outputs
                and not has_vectorized_variant_inputs):
              converter = functools.partial(
                  _fallback_converter, root_cause=root_cause, warn=self._warn)
            else:
              message = (f"No pfor vectorization defined for {y_op.type}\n"
                         f"{y_op}\n inputs: {converted_inputs}.")
              if not self._fallback_to_while_loop:
                message += ("Consider enabling the fallback_to_while_loop "
                            "option to pfor, which may run slower.")
              raise ValueError(message)
          # TODO(rachelim): Handle the case where some inputs are sparsely
          # stacked. We should only call the converter if it supports handling
          # those inputs.
          pfor_inputs = _PforInput(self, y_op, converted_inputs)
          try:
            try:
              new_outputs = converter(pfor_inputs)
            except ConversionNotImplementedError as e:
              has_vectorized_variant_inputs = any(
                  _is_variant_with_internal_stacking(x) for x in
                  y_op.inputs)
              if (self._fallback_to_while_loop
                  and not has_vectorized_variant_inputs):
                new_outputs = _fallback_converter(
                    pfor_inputs, root_cause=str(e))
              else:
                raise ValueError(str(e)).with_traceback(sys.exc_info()[2])
          except Exception as e:  # pylint: disable=broad-except
            logging.error(
                f"Got error while pfor was converting op {y_op} with inputs "
                f"{y_op.inputs[:]}\n, converted inputs {pfor_inputs.inputs}\n"
                f"Here are the pfor conversion stack traces: {e}")
            original_op = y_op
            while isinstance(original_op, ops.Operation):
              logging.error(
                  "%s\ncreated at:\n  %s", original_op,
                  "  ".join(traceback.format_list(original_op.traceback)))
              original_op = original_op._original_op
            raise

          if isinstance(new_outputs, WrappedTensor):
            new_outputs = [new_outputs]
          assert isinstance(new_outputs,
                            (list, tuple, ops.Operation)), new_outputs
        logging.vlog(2, f"converted {y_op} {new_outputs}")

        # Insert into self._conversion_map
        if y is y_op:
          assert isinstance(new_outputs, ops.Operation)
          self._add_conversion(y_op, new_outputs)
        else:
          assert len(y_op.outputs) == len(new_outputs), (y_op, y_op.outputs,
                                                         new_outputs)
          for old_output, new_output in zip(y_op.outputs, new_outputs):
            assert isinstance(new_output, WrappedTensor), (new_output, y, y_op)
            assert old_output.dtype == new_output.t.dtype, (new_output, y, y_op)
            # Set shape for converted output.
            output_shape = old_output.shape
            if not new_output.is_sparse_stacked:
              if new_output.is_stacked:
                loop_len = tensor_util.constant_value(self.loop_len_vector)
                if loop_len is None:
                  batch_dim = tensor_shape.TensorShape([None])
                else:
                  batch_dim = tensor_shape.TensorShape(loop_len)
                output_shape = batch_dim.concatenate(output_shape)
              if _is_variant_with_internal_stacking(new_output.t):
                new_output.t.set_shape([])
              else:
                new_output.t.set_shape(output_shape)
            self._add_conversion(old_output, new_output)
        stack.popleft()

    return self._conversion_map[op_or_tensor]

  @property
  def loop_len_vector(self):
    # [FUNCTION DOC]
    # Property: a one-element vector holding the number of iterations (constant when known).
    """Returns a single element vector whose value is number of iterations."""
    return self._loop_len_vector

  @property
  def loop_var(self):
    # [FUNCTION DOC]
    # Property: the placeholder tensor holding the loop iteration number.
    """Returns placeholder loop variable."""
    return self._loop_var

  @property
  def pfor_ops(self):
    # [FUNCTION DOC]
    # Property: the set of operations that form the loop body.
    return self._pfor_ops

  @property
  def pfor_config(self):
    # [FUNCTION DOC]
    # Property: the PForConfig in use (may be null).
    return self._pfor_config

  @property
  def all_indices_partitioned(self):
    # [FUNCTION DOC]
    # Property: true when inside a control flow construct where not all pfor iterations may be active.
    """all_indices_partitioned property.

    Returns:
      True if we are inside a control flow construct and not all pfor iterations
      may be active.
    """
    return self._all_indices_partitioned

  @property
  def fallback_to_while_loop(self):
    # [FUNCTION DOC]
    # Property: whether unconvertible ops may be run sequentially in a while loop.
    return self._fallback_to_while_loop


# The code below defines converters for different operations. Please see comment
# for RegisterPFor to see how converters should be defined.


# image_ops


@RegisterPFor("AdjustContrastv2")
def _convert_adjust_contrastv2(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for the AdjustContrastv2 op. Requires stacked images (input 0) and a
  # loop-invariant contrast factor (input 1); applies the image op to the stacked batch (the extra
  # leading axis acts like a batch axis). Returns one stacked wrapped tensor.
  images = pfor_input.stacked_input(0)
  contrast_factor = pfor_input.unstacked_input(1)
  return wrap(gen_image_ops.adjust_contrastv2(images, contrast_factor), True)


@RegisterPFor("AdjustHue")
def _convert_adjust_hue(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for AdjustHue. Requires stacked images (input 0) and a loop-invariant
  # delta (input 1); applies the op directly to the stacked images. Returns one stacked tensor.
  images = pfor_input.stacked_input(0)
  delta = pfor_input.unstacked_input(1)
  return wrap(gen_image_ops.adjust_hue(images, delta), True)


@RegisterPFor("AdjustSaturation")
def _convert_adjust_saturation(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for AdjustSaturation. Requires stacked images (input 0) and a
  # loop-invariant scale (input 1); applies the op directly. Returns one stacked tensor.
  images = pfor_input.stacked_input(0)
  scale = pfor_input.unstacked_input(1)
  return wrap(gen_image_ops.adjust_saturation(images, scale), True)


# nn_ops


def _flatten_first_two_dims(x):
  # [FUNCTION DOC]
  # HELPER: merges the first two dimensions of a tensor into one (unknown-size dimension inferred),
  # keeping the rest of the shape. Used to fold the iteration axis into a batch axis. Returns the
  # reshaped tensor.
  """Merges first two dimensions."""
  old_shape = array_ops.shape(x)
  first_dim = constant_op.constant([-1], dtype=old_shape.dtype)
  new_shape = array_ops.concat([first_dim, old_shape[2:]], axis=0)
  return array_ops.reshape(x, new_shape)


def _unflatten_first_dim(x, first_dim):
  # [FUNCTION DOC]
  # HELPER: inverse of merging the leading dimensions. Inputs: a tensor and a one-element vector with
  # the size of the new first dimension (the loop length). Reshapes so the first dimension becomes
  # that size followed by an inferred dimension, keeping the remaining dimensions.
  """Splits first dimension into [first_dim, -1]."""
  old_shape = array_ops.shape(x)
  first_dim = math_ops.cast(first_dim, old_shape.dtype)
  second_dim = constant_op.constant([-1], dtype=old_shape.dtype)
  new_shape = array_ops.concat([first_dim, second_dim, old_shape[1:]], axis=0)
  return array_ops.reshape(x, new_shape)


def _inputs_with_flattening(pfor_input: _PforInput, input_indices):
  # [FUNCTION DOC]
  # HELPER: prepares inputs for ops that treat the first axis as batch.
  # Inputs: pfor_input and input_indices (positions to stack and flatten; null means none). Stacks
  # those inputs (mutating the pfor input), flattens their first two dims, and takes every other
  # input as loop invariant (so it raises a conversion-not-implemented error if such an input is
  # stacked). Returns a list of tensors, one per input.
  """Stacks and flattens first dim of inputs at indices `input_indices`."""
  if input_indices is None:
    input_indices = []
  pfor_input.stack_inputs(stack_indices=input_indices)
  inputs = []
  for i in range(pfor_input.num_inputs):
    if i in input_indices:
      inp = pfor_input.stacked_input(i)
      inp = _flatten_first_two_dims(inp)
    else:
      inp = pfor_input.unstacked_input(i)
    inputs.append(inp)
  return inputs


@RegisterPForWithArgs("Conv2D", dims=[0])
@RegisterPForWithArgs("DepthToSpace", dims=[0])
@RegisterPForWithArgs("AvgPool", dims=[0])
@RegisterPForWithArgs("AvgPool3D", dims=[0])
@RegisterPForWithArgs("MaxPool", dims=[0])
@RegisterPForWithArgs("MaxPoolV2", dims=[0])
@RegisterPForWithArgs("MaxPool3D", dims=[0])
@RegisterPForWithArgs("MaxPool3DGrad", dims=[0, 1, 2])
@RegisterPForWithArgs("MaxPoolGrad", dims=[0, 1, 2])
@RegisterPForWithArgs("MaxPoolGradV2", dims=[0, 1, 2])
@RegisterPForWithArgs("MaxPool3DGradGrad", dims=[0, 1, 2])
@RegisterPForWithArgs("MaxPoolGradGrad", dims=[0, 1, 2])
@RegisterPForWithArgs("MaxPoolGradGradV2", dims=[0, 1, 2])
@RegisterPForWithArgs("SoftmaxCrossEntropyWithLogits", dims=[0, 1])
@RegisterPForWithArgs("SparseSoftmaxCrossEntropyWithLogits", dims=[0, 1])
@RegisterPForWithArgs("SpaceToDepth", dims=[0])
def _convert_flatten_batch(pfor_input: _PforInput, op_type, dims):
  # [FUNCTION DOC]
  # Registered pfor converter shared by many batch-style ops (convolution, pooling and their
  # gradients, depth/space conversions, softmax cross entropy). The registration supplies which input
  # positions are flattened.
  # Flattens the iteration axis into the batch axis for the selected inputs, recreates the op with the
  # original attributes, then splits the first dimension of every output back to the loop length.
  # Returns a list of stacked tensors. The op type argument is unused.
  del op_type
  inputs = _inputs_with_flattening(pfor_input, dims)
  outputs = _create_op(
      pfor_input.op_type,
      inputs, [x.dtype for x in pfor_input.outputs],
      attrs=pfor_input.op.node_def.attr).outputs
  n = pfor_input.pfor.loop_len_vector
  outputs = [_unflatten_first_dim(x, n) for x in outputs]
  return [wrap(x, True) for x in outputs]


_channel_flatten_input_cache = {}


@RegisterPFor("BatchToSpaceND")
def _convert_batch_to_space_nd(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for BatchToSpaceND. Requires a stacked input and loop-invariant block
  # shape and crops. Reshapes and transposes so the iteration axis sits inside the axes that move
  # to space, merges it with the batch axis, runs the op, and splits the resulting batch axis again.
  # Returns one stacked tensor.
  inp = pfor_input.stacked_input(0)
  block_shape = pfor_input.unstacked_input(1)
  crops = pfor_input.unstacked_input(2)

  inp_shape = array_ops.shape(inp)
  n = math_ops.cast(pfor_input.pfor.loop_len_vector, inp_shape.dtype)
  block_shape = math_ops.cast(block_shape, inp_shape.dtype)

  # Reshape and transpose to move the vectorization axis inside the axes that
  # will move to space.
  # Reshape to 4D and transpose
  block_size = math_ops.reduce_prod(block_shape)
  neg_one = constant_op.constant(-1, dtype=inp_shape.dtype)
  new_shape = [n[0], block_size, inp_shape[1] // block_size, neg_one]
  inp = array_ops.reshape(inp, new_shape)

  inp = array_ops.transpose(inp, [1, 0, 2, 3])
  # Reshape back to merge the block, vectorization and batch dimension, and
  # restore the other dimensions.
  new_shape = array_ops.concat([n * inp_shape[1], inp_shape[2:]], axis=0)
  inp = array_ops.reshape(inp, new_shape)

  # Call batch_to_space and then split the new batch axis.
  output = gen_array_ops.batch_to_space_nd(inp, block_shape, crops)
  output = _unflatten_first_dim(output, n)
  return wrap(output, True)


@RegisterPFor("SpaceToBatchND")
def _convert_space_to_batch_nd(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for SpaceToBatchND. Requires a stacked input and loop-invariant block
  # shape and paddings. Folds the iteration axis into batch, runs the op, then reshapes and transposes
  # so the iteration axis comes first again. Returns one stacked tensor.
  inp = pfor_input.stacked_input(0)
  block_shape = pfor_input.unstacked_input(1)
  paddings = pfor_input.unstacked_input(2)

  inp_shape = array_ops.shape(inp)
  n = math_ops.cast(pfor_input.pfor.loop_len_vector, inp_shape.dtype)
  block_shape = math_ops.cast(block_shape, inp_shape.dtype)

  inp = _flatten_first_two_dims(inp)
  output = gen_array_ops.space_to_batch_nd(inp, block_shape, paddings)
  output_shape = array_ops.shape(output)

  block_size = math_ops.reduce_prod(block_shape)
  neg_one = constant_op.constant(-1, dtype=inp_shape.dtype)
  new_shape = [block_size, n[0], neg_one]
  output = array_ops.reshape(output, new_shape)

  output = array_ops.transpose(output, [1, 0, 2])
  new_shape = array_ops.concat(
      [n, block_size * inp_shape[1:2], output_shape[1:]], axis=0)
  output = array_ops.reshape(output, new_shape)
  return wrap(output, True)


def _channel_flatten_input(x, data_format):
  # [FUNCTION DOC]
  # HELPER: merges the stack (iteration) dimension into the channel dimension of a 5-d tensor.
  # Inputs: a tensor and the data format (NCHW or NHWC, compared as bytes). Transposes the stack axis
  # next to the channel axis and reshapes to merge them. Results are cached in a module-level
  # dictionary keyed by graph, tensor reference and format (the cache is never cleared). Returns a
  # triple: transformed tensor, the permutation and the shape needed to undo it.
  """Merge the stack dimension with the channel dimension.

  If S is pfor's stacking dimension, then,
    - for SNCHW, we transpose to NSCHW. If N dimension has size 1, the transpose
      should be cheap.
    - for SNHWC, we transpose to NHWSC.
  We then merge the S and C dimension.

  Args:
    x: tensor_lib.Tensor to transform.
    data_format: "NCHW" or "NHWC".

  Returns:
    A 3-element tuple with the transformed value, along with the shape for
    reshape and order for transpose required to transform back.
  """

  graph = ops.get_default_graph()
  cache_key = (graph, x.ref(), data_format)
  if cache_key not in _channel_flatten_input_cache:
    x_shape = array_ops.shape(x)
    neg_ones = constant_op.constant([-1], dtype=x_shape.dtype)
    if data_format == b"NCHW":
      order = [1, 0, 2, 3, 4]
      shape = array_ops.concat([x_shape[1:2], neg_ones, x_shape[3:]], axis=0)
      reverse_order = order
    else:
      order = [1, 2, 3, 0, 4]
      shape = array_ops.concat([x_shape[1:4], neg_ones], axis=0)
      reverse_order = [3, 0, 1, 2, 4]
    # Move S dimension next to C dimension.
    x = array_ops.transpose(x, order)
    reverse_shape = array_ops.shape(x)
    # Reshape to merge the S and C dimension.
    x = array_ops.reshape(x, shape)
    outputs = x, reverse_order, reverse_shape
    _channel_flatten_input_cache[cache_key] = outputs
  else:
    outputs = _channel_flatten_input_cache[cache_key]
  return outputs


# Note that with training=True, running FusedBatchNormV3 on individual examples
# is very different from running FusedBatchNormV3 on a batch of those examples.
# This is because, for the latter case, the operation can be considered as first
# computing the mean and variance over all the examples and then using these
# to scale all those examples. This creates a data dependency between these
# different "iterations" since the inputs to the scaling step depends on the
# statistics coming from all these inputs.
# As with other kernels, the conversion here effectively runs the kernel
# independently for each iteration, and returns outputs by stacking outputs from
# each of those iterations.
@RegisterPFor("FusedBatchNormV3")
def _convert_fused_batch_norm(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for FusedBatchNormV3.
  # Inference mode: iteration and batch axes are merged like a normal batch op; the extra statistics
  # outputs are returned as unstacked zeros. Training mode: statistics must be computed per iteration,
  # so the iteration axis is merged into the channel axis (and the per-channel parameters are
  # flattened to match), the op is run once, and the output is reshaped/transposed back; the other
  # outputs are split back to the loop length. Returns the list of six wrapped outputs.
  is_training = pfor_input.get_attr("is_training")
  # When BatchNorm is used with training=False, mean and variance are provided
  # externally and used as is by the op. Thus, we can merge the S and N
  # dimensions as we do for regular operations.
  # When BatchNorm is used with training=True, mean and variance are computed
  # for each channel across the batch dimension (first one). If we merge S and N
  # dimensions, mean and variances will be computed over a larger set. So, we
  # merge the S and C dimensions instead.
  if not is_training:
    # We return zeros for batch_mean and batch_variance output. Note that CPU
    # and GPU seem to have different behavior for those two outputs. CPU outputs
    # zero because these values are not used during inference. GPU outputs
    # something, probably real means and variances.
    inputs = _inputs_with_flattening(pfor_input, [0])
    outputs = _create_op(
        pfor_input.op_type,
        inputs, [x.dtype for x in pfor_input.outputs],
        attrs=pfor_input.op.node_def.attr).outputs
    y = outputs[0]
    n = pfor_input.pfor.loop_len_vector
    y = _unflatten_first_dim(y, n)
    mean = pfor_input.unstacked_input(3)
    zeros = array_ops.zeros_like(mean)
    return [wrap(y, True)] + [wrap(zeros, False)] * 5

  pfor_input.stack_inputs()
  data_format = pfor_input.get_attr("data_format")
  # We merge the first dimension with the "C" dimension, run FusedBatchNormV3,
  # and then transpose back.
  x = pfor_input.stacked_input(0)
  x, reverse_order, reverse_shape = _channel_flatten_input(x, data_format)
  # Note that we stack all the other inputs as well so that they are the same
  # size as the new size of the channel dimension.
  inputs = [x] + [
      array_ops.reshape(pfor_input.stacked_input(i), [-1])
      for i in range(1, pfor_input.num_inputs)
  ]
  outputs = _create_op(
      pfor_input.op_type,
      inputs, [x.dtype for x in pfor_input.outputs],
      attrs=pfor_input.op.node_def.attr).outputs
  y = outputs[0]
  y = array_ops.reshape(y, reverse_shape)
  y = array_ops.transpose(y, reverse_order)
  n = pfor_input.pfor.loop_len_vector
  outputs = [_unflatten_first_dim(x, n) for x in outputs[1:]]
  outputs = [y] + outputs
  return [wrap(x, True) for x in outputs]


@RegisterPFor("FusedBatchNormGradV3")
def _convert_fused_batch_norm_grad(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for FusedBatchNormGradV3. Stacks all inputs, merges the iteration axis
  # into the channel axis for the backpropagated gradient and the activations, flattens the
  # per-channel inputs, runs the op once and restores the layout of the first output; the other
  # outputs are split along the first dimension. Returns a list of stacked outputs.
  pfor_input.stack_inputs()
  data_format = pfor_input.get_attr("data_format")
  y_backprop = pfor_input.stacked_input(0)
  y_backprop, _, _ = _channel_flatten_input(y_backprop, data_format)
  x = pfor_input.stacked_input(1)
  x, x_reverse_order, x_reverse_shape = _channel_flatten_input(x, data_format)
  inputs = [y_backprop, x] + [
      array_ops.reshape(pfor_input.stacked_input(i), [-1])
      for i in range(2, pfor_input.num_inputs)
  ]
  outputs = _create_op(
      pfor_input.op_type,
      inputs, [x.dtype for x in pfor_input.outputs],
      attrs=pfor_input.op.node_def.attr).outputs
  x_backprop = outputs[0]
  x_backprop = array_ops.reshape(x_backprop, x_reverse_shape)
  x_backprop = array_ops.transpose(x_backprop, x_reverse_order)
  n = pfor_input.pfor.loop_len_vector
  outputs = [_unflatten_first_dim(x, n) for x in outputs[1:]]
  outputs = [x_backprop] + outputs
  return [wrap(output, True) for output in outputs]


@RegisterPForWithArgs("Conv2DBackpropInput", flatten_dims=[2], shape_dim=0)
@RegisterPForWithArgs("AvgPoolGrad", flatten_dims=[1], shape_dim=0)
@RegisterPForWithArgs("AvgPool3DGrad", flatten_dims=[1], shape_dim=0)
def _convert_flatten_batch_shape_input(
    pfor_input: _PforInput, op_type, flatten_dims, shape_dim):
  # [FUNCTION DOC]
  # Registered pfor converter for ops whose first input is a size/shape tensor (convolution input
  # gradient, average-pool gradients). The registration says which inputs to flatten and where the
  # shape input is. Flattens the batch-like inputs, multiplies the size entry for the batch dimension
  # by the loop length, recreates the op, and splits the first dimension of the outputs.
  # Returns a list of stacked outputs. The op type argument is unused.
  del op_type
  inputs = _inputs_with_flattening(pfor_input, flatten_dims)
  n = pfor_input.pfor.loop_len_vector
  # Adjust the `input_sizes` input.
  ones = array_ops.ones([array_ops.shape(inputs[shape_dim])[0] - 1],
                        dtype=n.dtype)
  inputs[shape_dim] *= array_ops.concat([n, ones], axis=0)
  outputs = _create_op(
      pfor_input.op_type,
      inputs, [x.dtype for x in pfor_input.outputs],
      attrs=pfor_input.op.node_def.attr).outputs
  outputs = [_unflatten_first_dim(x, n) for x in outputs]
  return [wrap(x, True) for x in outputs]


@RegisterPFor("Conv2DBackpropFilter")
def _convert_conv2d_backprop_filter(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for Conv2DBackpropFilter. Stacks the gradient input.
  # If the forward input is stacked it falls back to a sequential while loop (with a logged warning
  # that this is inefficient), writing each result to a TensorArray. Otherwise it merges the
  # iteration axis with the filter's output-channel axis by enlarging the filter sizes, calls the
  # op once, and reshapes/transposes the result back. Returns a stacked tensor.
  pfor_input.stack_inputs(stack_indices=[2])
  inputs, inputs_stacked, _ = pfor_input.input(0)
  filter_sizes = pfor_input.unstacked_input(1)
  grads = pfor_input.stacked_input(2)
  strides = pfor_input.get_attr("strides")
  padding = pfor_input.get_attr("padding")
  use_cudnn_on_gpu = pfor_input.get_attr("use_cudnn_on_gpu")
  data_format = pfor_input.get_attr("data_format")
  dilations = pfor_input.get_attr("dilations")
  if inputs_stacked:
    # TODO(agarwal): Implement this efficiently.
    logging.warning("Conv2DBackpropFilter uses a while_loop. Fix that!")

    def while_body(i, ta):
      # [FUNCTION DOC]
      # Nested loop body used by the sequential path of the filter-gradient converter: given the counter
      # and a TensorArray, computes the filter gradient for slice i of the stacked inputs and gradients
      # with the captured strides, padding, data format and dilations, writes it at i, and returns the
      # incremented counter and the array.
      inp_i = inputs[i, ...]
      grad_i = grads[i, ...]
      output = nn_ops.conv2d_backprop_filter(
          inp_i,
          filter_sizes,
          grad_i,
          strides=strides,
          padding=padding,
          use_cudnn_on_gpu=use_cudnn_on_gpu,
          data_format=data_format,
          dilations=dilations)
      return i + 1, ta.write(i, output)

    n = array_ops.reshape(pfor_input.pfor.loop_len_vector, [])
    _, ta = while_loop.while_loop(
        lambda i, ta: i < n, while_body,
        (0, tensor_array_ops.TensorArray(inputs.dtype, n)))
    output = ta.stack()
    return wrap(output, True)
  else:
    # We merge the stack dimension with the channel dimension of the gradients
    # and pretend we had a larger filter (see change to filter_sizes below).
    # Once the filter backprop is computed, we reshape and transpose back
    # appropriately.
    grads, _, _ = _channel_flatten_input(grads, data_format)
    n = pfor_input.pfor.loop_len_vector
    old_filter_sizes = filter_sizes
    filter_sizes *= array_ops.concat([[1, 1, 1], n], axis=0)
    output = nn_ops.conv2d_backprop_filter(
        inputs,
        filter_sizes,
        grads,
        strides=strides,
        padding=padding,
        use_cudnn_on_gpu=use_cudnn_on_gpu,
        data_format=data_format,
        dilations=dilations)
    new_filter_shape = array_ops.concat([old_filter_sizes[:3], n, [-1]], axis=0)
    output = array_ops.reshape(output, new_filter_shape)
    output = array_ops.transpose(output, [3, 0, 1, 2, 4])
    return wrap(output, True)


def _flatten_with_inner_dim(x, dim, x_rank):
  # [FUNCTION DOC]
  # HELPER: moves the leading (iteration) axis next to a chosen dimension and merges them.
  # Inputs: tensor, the dimension index to merge with, and the tensor's rank. Transposes the first
  # axis to sit before that dimension and reshapes (merged dimension inferred). Returns the result.
  """Merges the first dim with the specified dim."""
  shape = array_ops.shape(x)
  x = array_ops.transpose(x,
                          list(range(1, dim)) + [0] + list(range(dim, x_rank)))

  if dim < x_rank - 1:
    new_shape_pieces = [shape[1:dim], [-1], shape[dim + 1:]]
  else:
    new_shape_pieces = [shape[1:dim], [-1]]
  new_shape = array_ops.concat(new_shape_pieces, axis=0)
  return array_ops.reshape(x, new_shape)


def _unflatten_with_inner_dim(x, dim, x_rank, stack_size):
  # [FUNCTION DOC]
  # HELPER: undoes the flattening above. Inputs: tensor, merged dimension index, original rank and
  # the stack size. Splits the merged dimension into the stack size and an inferred size, then
  # transposes the stack axis to the front. Returns the tensor.
  """Undoes _flatten_with_inner_dim."""
  shape = array_ops.shape(x)
  if dim < x_rank - 1:
    new_shape_pieces = [shape[:dim], [stack_size], [-1], shape[dim + 1:]]
  else:
    new_shape_pieces = [shape[:dim], [stack_size], [-1]]
  new_shape = array_ops.concat(new_shape_pieces, axis=0)
  x = array_ops.reshape(x, new_shape)
  dims_permutation = [dim] + list(range(dim)) + list(range(dim + 1, x_rank + 1))
  return array_ops.transpose(x, dims_permutation)


@RegisterPFor("DepthwiseConv2dNative")
def _convert_depthwise_conv2d_native(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for DepthwiseConv2dNative. Both input and kernel must be stacked. Since
  # kernels may differ per iteration, the iteration axis is folded into the channel dimension of the
  # input and into the kernel's input-channel dimension; the op runs once and the result is unfolded.
  # Returns one stacked tensor.
  # Kernel can be vectorized, so folding to batch dimension does not work. We
  # instead fold into the channel dimension because it is parallel.
  stack_size = pfor_input.pfor.loop_len_vector[0]
  data_format = pfor_input.get_attr("data_format")
  c_dim = 1 if data_format == b"NCHW" else 3
  t = _flatten_with_inner_dim(pfor_input.stacked_input(0), c_dim + 1, 5)
  kernel = _flatten_with_inner_dim(pfor_input.stacked_input(1), 3, 5)
  conv = _create_op(
      "DepthwiseConv2dNative", [t, kernel],
      [x.dtype for x in pfor_input.outputs],
      attrs=pfor_input.op.node_def.attr).outputs[0]
  return wrap(_unflatten_with_inner_dim(conv, c_dim, 4, stack_size), True)


@RegisterPFor("DepthwiseConv2dNativeBackpropInput")
def _convert_depthwise_conv2d_native_backprop_input(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for DepthwiseConv2dNativeBackpropInput. Takes a loop-invariant input
  # size vector (scaled in the channel entry by the loop length) plus stacked kernel and output
  # gradient, folds the iteration axis into channels, runs the op once and unfolds. Returns one
  # stacked tensor.
  stack_size = pfor_input.pfor.loop_len_vector[0]
  input_sizes = pfor_input.unstacked_input(0)
  data_format = pfor_input.get_attr("data_format")
  c_dim = 1 if data_format == b"NCHW" else 3
  input_sizes_mutipliers = [
      constant_op.constant([1] * c_dim, dtype=dtypes.int32), [stack_size]
  ]
  if c_dim < 3:
    input_sizes_mutipliers += [
        constant_op.constant([1] * (3 - c_dim), dtype=dtypes.int32)
    ]
  input_sizes *= array_ops.concat(input_sizes_mutipliers, axis=0)
  kernel = _flatten_with_inner_dim(pfor_input.stacked_input(1), 3, 5)
  out_backprop = _flatten_with_inner_dim(
      pfor_input.stacked_input(2), c_dim + 1, 5)
  result = _create_op(
      "DepthwiseConv2dNativeBackpropInput", [input_sizes, kernel, out_backprop],
      [x.dtype for x in pfor_input.outputs],
      attrs=pfor_input.op.node_def.attr).outputs[0]
  return wrap(_unflatten_with_inner_dim(result, c_dim, 4, stack_size), True)


@RegisterPFor("DepthwiseConv2dNativeBackpropFilter")
def _convert_depthwise_conv2d_native_backprop_filter(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for DepthwiseConv2dNativeBackpropFilter. Takes stacked input and output
  # gradient and a loop-invariant filter size (whose channel entry is scaled by the loop length),
  # folds the iteration axis into channels, runs the op once and unfolds. Returns one stacked tensor.
  stack_size = pfor_input.pfor.loop_len_vector[0]
  data_format = pfor_input.get_attr("data_format")
  c_dim = 1 if data_format == b"NCHW" else 3
  inputs = _flatten_with_inner_dim(pfor_input.stacked_input(0), c_dim + 1, 5)
  filter_sizes = pfor_input.unstacked_input(1)
  filter_sizes_multipliers = [
      constant_op.constant([1, 1], dtype=dtypes.int32), [stack_size],
      constant_op.constant([1], dtype=dtypes.int32)
  ]
  filter_sizes *= array_ops.concat(filter_sizes_multipliers, axis=0)
  out_backprop = _flatten_with_inner_dim(
      pfor_input.stacked_input(2), c_dim + 1, 5)
  result = _create_op(
      "DepthwiseConv2dNativeBackpropFilter",
      [inputs, filter_sizes, out_backprop],
      [x.dtype for x in pfor_input.outputs],
      attrs=pfor_input.op.node_def.attr).outputs[0]
  return wrap(_unflatten_with_inner_dim(result, 2, 4, stack_size), True)


@RegisterPForWithArgs("LogSoftmax", gen_nn_ops.log_softmax)
@RegisterPForWithArgs("Softmax", gen_nn_ops.softmax)
def _convert_softmax(pfor_input: _PforInput, op_type, op_func):
  # [FUNCTION DOC]
  # Registered pfor converter for Softmax and LogSoftmax (the op function is supplied by the
  # registration). Applies the function to the stacked input; the op works over the last axis so no
  # adjustment is needed. Returns one stacked tensor. Op type argument unused.
  del op_type
  return wrap(op_func(pfor_input.stacked_input(0)), True)


# array_ops


@RegisterPForWithArgs("Identity", array_ops.identity)
@RegisterPForWithArgs("StopGradient", array_ops.stop_gradient)
@RegisterPForWithArgs("MatrixDiag", array_ops.matrix_diag)
@RegisterPForWithArgs("MatrixDiagPart", array_ops.matrix_diag_part)
@RegisterPForWithArgs("_EagerConst", array_ops.identity)
def _convert_identity(pfor_input: _PforInput, op_type, op_func):
  # [FUNCTION DOC]
  # Registered pfor converter for ops that act elementwise on the leading axes (Identity,
  # StopGradient, MatrixDiag, MatrixDiagPart, EagerConst). Calls the supplied op function on all raw
  # input tensors and returns one stacked tensor. Op type argument unused.
  del op_type
  return wrap(op_func(*[x.t for x in pfor_input.inputs]), True)


@RegisterPFor("IdentityN")
def _convert_identity_n(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for IdentityN. Applies identity-n to the raw tensors and returns a list
  # of wrapped outputs that keep each input's stackedness.
  outputs = array_ops.identity_n([x.t for x in pfor_input.inputs])
  return [
      wrap(out, inp.is_stacked) for out, inp in zip(outputs, pfor_input.inputs)
  ]


@RegisterPFor("Reshape")
def _convert_reshape(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for Reshape. Requires a stacked tensor and a loop-invariant target
  # shape; prepends the loop length to the shape and reshapes. Returns one stacked tensor.
  t = pfor_input.stacked_input(0)
  shape = pfor_input.unstacked_input(1)
  n = math_ops.cast(pfor_input.pfor.loop_len_vector, shape.dtype)
  new_shape = array_ops.concat([n, shape], axis=0)
  return wrap(array_ops.reshape(t, new_shape), True)


@RegisterPFor("TopK")
@RegisterPFor("TopKV2")
def _convert_top_k(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for TopK and TopKV2. Recreates the op directly on the raw inputs
  # (top-k works on the last axis) with the original attributes and returns all outputs as stacked.
  outputs = _create_op(
      op_type=pfor_input.op_type,
      inputs=[x.t for x in pfor_input.inputs],
      op_dtypes=[x.dtype for x in pfor_input.outputs],
      attrs=pfor_input.op.node_def.attr).outputs
  return [wrap(x, True) for x in outputs]


@RegisterPFor("Fill")
def _convert_fill(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for Fill. Requires loop-invariant dims and a stacked value. Reshapes
  # the value to add trailing singleton axes and broadcasts it to the loop length followed by the
  # dims. Returns one stacked tensor.
  dims = pfor_input.unstacked_input(0)
  value = pfor_input.stacked_input(1)
  # Expand the rank of `value`
  new_shape = array_ops.concat(
      [[-1], array_ops.ones([array_ops.size(dims)], dtype=dtypes.int32)],
      axis=0)
  value = array_ops.reshape(value, new_shape)
  # Compute the new output shape
  new_dims = array_ops.concat([pfor_input.pfor.loop_len_vector, dims], axis=0)
  # Broadcast
  return wrap(array_ops.broadcast_to(value, new_dims), True)


@RegisterPFor("BroadcastTo")
def _convert_broadcast_to(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for BroadcastTo. Requires a stacked tensor and a loop-invariant target
  # shape. Expands the tensor with singleton axes after the stack axis so ranks match, then
  # broadcasts to the loop length followed by the shape. Returns one stacked tensor.
  shape = pfor_input.unstacked_input(1)
  n = math_ops.cast(pfor_input.pfor.loop_len_vector, shape.dtype)
  new_shape = array_ops.concat([n, shape], axis=0)
  new_rank = _size(new_shape, dtypes.int32)

  t = pfor_input.stacked_input(0)
  t = _expand_dims(t, 1, new_rank - _rank(t))
  return wrap(array_ops.broadcast_to(t, new_shape), True)


@RegisterPFor("ExpandDims")
def _convert_expanddims(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for ExpandDims. Requires a stacked tensor and loop-invariant axis;
  # non-negative axes are shifted by one for the extra leading axis. Returns one stacked tensor.
  t = pfor_input.stacked_input(0)
  dim = pfor_input.unstacked_input(1)
  dim += math_ops.cast(dim >= 0, dim.dtype)
  return wrap(array_ops.expand_dims(t, axis=dim), True)


@RegisterPForWithArgs("LowerBound", gen_array_ops.lower_bound)
@RegisterPForWithArgs("UpperBound", gen_array_ops.upper_bound)
def _convert_searchsorted(pfor_input: _PforInput, _, op_func):
  # [FUNCTION DOC]
  # Registered pfor converter for LowerBound and UpperBound (op function supplied by registration).
  # Stacks both inputs, flattens the iteration axis into the batch axis, calls the function with the
  # output type attribute, and splits the result back. Returns one stacked tensor.
  pfor_input.stack_inputs()
  sorted_inputs = _flatten_first_two_dims(pfor_input.stacked_input(0))
  values = _flatten_first_two_dims(pfor_input.stacked_input(1))
  out_type = pfor_input.get_attr("out_type")
  output = op_func(sorted_inputs, values, out_type)
  return wrap(
      _unflatten_first_dim(output, pfor_input.pfor.loop_len_vector), True)


@RegisterPFor("MatrixBandPart")
def _convert_matrix_band_part(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for MatrixBandPart. Requires a stacked matrix and loop-invariant lower
  # and upper band counts; applies the op to the batch. Returns one stacked tensor.
  t = pfor_input.stacked_input(0)
  num_lower = pfor_input.unstacked_input(1)
  num_upper = pfor_input.unstacked_input(2)
  return wrap(
      array_ops.matrix_band_part(t, num_lower=num_lower, num_upper=num_upper),
      True)


@RegisterPFor("MatrixSetDiag")
def _convert_matrix_set_diag(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for MatrixSetDiag. Stacks both inputs and applies the op on the
  # batches. Returns one stacked tensor.
  pfor_input.stack_inputs()
  t = pfor_input.stacked_input(0)
  diag = pfor_input.stacked_input(1)
  return wrap(array_ops.matrix_set_diag(t, diag), True)


# Registrations for Matrix{Diag,DiagPart,SetDiag}V2-3.
# The input orders defined in the OpKernel and the actual python API are
# different (for compatibility with V1), so we cannot use _convert_identity.
# v2 is not compatible with v3 and is never exposed on the public API.
@RegisterPFor("MatrixDiagV2")
@RegisterPFor("MatrixDiagV3")
def _convert_matrix_diag_v2(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for MatrixDiagV2 and MatrixDiagV3. Takes a stacked diagonal and
  # loop-invariant k, row count, column count and padding value; the V3 version also passes the
  # alignment attribute. The input order of the kernel differs from the Python API, hence this
  # dedicated converter. Returns one stacked tensor.
  params = {
      "diagonal": pfor_input.stacked_input(0),
      "k": pfor_input.unstacked_input(1),
      "num_rows": pfor_input.unstacked_input(2),
      "num_cols": pfor_input.unstacked_input(3),
      "padding_value": pfor_input.unstacked_input(4)
  }
  if pfor_input.op_type == "MatrixDiagV2":
    return wrap(array_ops.matrix_diag_v2(**params), True)
  params["align"] = pfor_input.get_attr("align")
  return wrap(array_ops.matrix_diag(**params), True)


@RegisterPFor("Diag")
def _convert_diag(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for Diag. When the stacked input has rank two (a vector per iteration)
  # it uses the matrix-diagonal function; otherwise it falls back to a sequential while loop without
  # a warning. Returns stacked output(s).
  diag = pfor_input.stacked_input(0)
  if diag.shape.ndims == 2:
    # We can use matrix_diag.
    return wrap(array_ops.matrix_diag(diag), True)
  else:
    # It is not clear if we can do better than a while loop here with existing
    # kernels.
    return _fallback_converter(pfor_input, warn=False)


# See notes for MatrixDiagV2
@RegisterPFor("MatrixDiagPartV2")
@RegisterPFor("MatrixDiagPartV3")
def _convert_matrix_diag_part_v2(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for MatrixDiagPartV2 and V3. Takes a stacked matrix and loop-invariant
  # k and padding value; V3 also passes the alignment attribute. Returns one stacked tensor.
  params = {
      "input": pfor_input.stacked_input(0),
      "k": pfor_input.unstacked_input(1),
      "padding_value": pfor_input.unstacked_input(2)
  }
  if pfor_input.op_type == "MatrixDiagPartV2":
    return wrap(array_ops.matrix_diag_part_v2(**params), True)
  params["align"] = pfor_input.get_attr("align")
  return wrap(array_ops.matrix_diag_part(**params), True)


# See notes for MatrixDiagV2
@RegisterPFor("MatrixSetDiagV2")
@RegisterPFor("MatrixSetDiagV3")
def _convert_matrix_set_diag_v2(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for MatrixSetDiagV2 and V3. Stacks the matrix and diagonal, takes a
  # loop-invariant k, and (for V3) the alignment attribute. Returns one stacked tensor.
  pfor_input.stack_inputs([0, 1])
  params = {
      "input": pfor_input.stacked_input(0),
      "diagonal": pfor_input.stacked_input(1),
      "k": pfor_input.unstacked_input(2)
  }
  if pfor_input.op_type == "MatrixSetDiagV2":
    return wrap(array_ops.matrix_set_diag_v2(**params), True)
  params["align"] = pfor_input.get_attr("align")
  return wrap(array_ops.matrix_set_diag(**params), True)


@RegisterPFor("DiagPart")
def _convert_diag_part(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for DiagPart. If the stacked input has rank three (a matrix per
  # iteration) it uses the matrix-diagonal-part function; otherwise it falls back to a sequential
  # while loop without a warning.
  inp = pfor_input.stacked_input(0)
  if inp.shape.ndims == 3:
    # We can use matrix_diag_part.
    return wrap(array_ops.matrix_diag_part(inp), True)
  else:
    # It is not clear if we can do better than a while loop here with existing
    # kernels.
    return _fallback_converter(pfor_input, warn=False)


@RegisterPFor("OneHot")
def _convert_one_hot(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for OneHot. Stacked indices; loop-invariant depth, on-value and
  # off-value; a non-negative axis attribute is shifted by one. Returns one stacked tensor.
  indices = pfor_input.stacked_input(0)
  depth = pfor_input.unstacked_input(1)
  on_value = pfor_input.unstacked_input(2)
  off_value = pfor_input.unstacked_input(3)
  axis = pfor_input.get_attr("axis")
  if axis >= 0:
    axis += 1
  return wrap(
      array_ops.one_hot(indices, depth, on_value, off_value, axis), True)


@RegisterPFor("Slice")
def _convert_slice(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for Slice. The input must be stacked and sizes loop invariant.
  # If the begin vector is loop invariant, prepends zero/minus-one entries and slices directly.
  # Otherwise (per-iteration begins) negative sizes are resolved against the original shape, the
  # largest resulting shape over iterations is taken (ragged outputs unsupported), and the slice is
  # done by computing coordinates for each output element, adding the per-iteration begin offsets and
  # gathering with batch dimensions. Returns one stacked tensor.
  t = pfor_input.stacked_input(0)
  begin, begin_stacked, _ = pfor_input.input(1)
  size = pfor_input.unstacked_input(2)
  if not begin_stacked:
    begin = array_ops.concat([[0], begin], axis=0)
    size = array_ops.concat([[-1], size], axis=0)
    return wrap(array_ops.slice(t, begin, size), True)
  else:
    # Handle negative sizes.
    #
    # If the `begin` entry corresponding to a negative `size` is loop-variant,
    # the output would be ragged. This case is not supported. But `size` having
    # some negative values and some loop-variant `begin`s is OK (and it's hard
    # to tell the difference statically).
    t_shape = array_ops.shape(t)
    size = math_ops.cast(size, t_shape.dtype)
    begin = math_ops.cast(begin, t_shape.dtype)
    n = math_ops.cast(pfor_input.pfor.loop_len_vector, t_shape.dtype)
    original_unstacked_shape = _stack(t_shape[1:], n).t
    broadcast_size = _stack(size, n).t
    result_shape = array_ops.where(
        math_ops.less(broadcast_size, 0),
        original_unstacked_shape - begin + broadcast_size + 1, broadcast_size)
    result_shape = math_ops.cast(math_ops.reduce_max(result_shape, axis=0),
                                 dtypes.int64)

    # Now we enumerate points in the sliced region for each pfor iteration and
    # gather them.
    cumsize = math_ops.cumprod(result_shape, exclusive=True, reverse=True)
    result_num_elements = math_ops.reduce_prod(result_shape)
    # Offsets are loop-variant. We first compute loop-invariant gather
    # coordinates, then broadcast-add the loop-variant `begin` offsets.
    result_base_coordinates = (
        math_ops.range(result_num_elements, dtype=dtypes.int64)[:, None]
        // cumsize[None, :]) % result_shape[None, :]
    result_coordinates = (
        begin[:, None, :]
        + math_ops.cast(result_base_coordinates, begin.dtype)[None, :, :])
    result_flat = array_ops.gather_nd(params=t, indices=result_coordinates,
                                      batch_dims=1)
    result_stacked_shape = array_ops.concat(
        [math_ops.cast(pfor_input.pfor.loop_len_vector, result_shape.dtype),
         result_shape],
        axis=0)
    return wrap(array_ops.reshape(result_flat, result_stacked_shape), True)


@RegisterPFor("Tile")
def _convert_tile(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for Tile. Stacked tensor, loop-invariant multiples with a leading one
  # added for the stack axis. Returns one stacked tensor.
  t = pfor_input.stacked_input(0)
  multiples = pfor_input.unstacked_input(1)
  multiples = array_ops.concat([[1], multiples], 0)
  return wrap(array_ops.tile(t, multiples), True)


@RegisterPFor("Pack")
def _convert_pack(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for Pack (stack). Stacks every input, shifts a non-negative axis by one
  # and stacks them. Returns one stacked tensor.
  pfor_input.stack_inputs()
  axis = pfor_input.get_attr("axis")
  if axis >= 0:
    axis += 1
  return wrap(
      array_ops_stack.stack([x.t for x in pfor_input.inputs], axis=axis), True)


@RegisterPFor("Unpack")
def _convert_unpack(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for Unpack (unstack). Stacked input; axis shifted by one when
  # non-negative; number of outputs from the attribute. Returns a list of stacked tensors.
  value = pfor_input.stacked_input(0)
  axis = pfor_input.get_attr("axis")
  if axis >= 0:
    axis += 1
  num = pfor_input.get_attr("num")
  return [wrap(x, True) for x
          in array_ops_stack.unstack(value, axis=axis, num=num)]


@RegisterPFor("Pad")
def _convert_pad(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for Pad. Stacked tensor and loop-invariant paddings with a zero row
  # added for the stack axis; constant mode. Returns one stacked tensor.
  t = pfor_input.stacked_input(0)
  paddings = pfor_input.unstacked_input(1)
  paddings = array_ops.concat([[[0, 0]], paddings], 0)
  return wrap(array_ops.pad(t, paddings, mode="CONSTANT"), True)


@RegisterPFor("PadV2")
def _convert_pad_v2(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for PadV2. Same as the Pad converter, using the version of the pad
  # function that takes a constant value. Returns one stacked tensor.
  t = pfor_input.stacked_input(0)
  paddings = pfor_input.unstacked_input(1)
  paddings = array_ops.concat([[[0, 0]], paddings], 0)
  return wrap(array_ops.pad_v2(t, paddings, mode="CONSTANT"), True)


@RegisterPFor("Split")
def _convert_split(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for Split. Loop-invariant split dimension (shifted by one when
  # non-negative) and stacked value; number of splits from the attribute. Returns a list of stacked
  # tensors.
  split_dim = pfor_input.unstacked_input(0)
  t = pfor_input.stacked_input(1)
  num_split = pfor_input.get_attr("num_split")
  split_dim += math_ops.cast(split_dim >= 0, dtypes.int32)
  return [wrap(x, True) for x in array_ops.split(t, num_split, axis=split_dim)]


@RegisterPFor("SplitV")
def _convert_split_v(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for SplitV. Stacked value, loop-invariant sizes and split dimension
  # (shifted by one when non-negative). Returns a list of stacked tensors.
  t = pfor_input.stacked_input(0)
  splits = pfor_input.unstacked_input(1)
  split_dim = pfor_input.unstacked_input(2)
  split_dim += math_ops.cast(split_dim >= 0, dtypes.int32)
  return [wrap(x, True) for x in array_ops.split(t, splits, axis=split_dim)]


@RegisterPFor("Squeeze")
def _convert_squeeze(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for Squeeze. Reads the squeeze_dims attribute, shifting non-negative
  # entries by one. Returns one stacked tensor.
  t = pfor_input.stacked_input(0)
  squeeze_dims = pfor_input.get_attr("squeeze_dims")
  squeeze_dims = [i + 1 if i >= 0 else i for i in squeeze_dims]
  return wrap(array_ops.squeeze(t, axis=squeeze_dims), True)


@RegisterPFor("ReverseV2")
def _convert_reverse(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for ReverseV2. Stacked value, loop-invariant axes; non-negative axes
  # are shifted by one. Returns one stacked tensor.
  value = pfor_input.stacked_input(0)
  axis = pfor_input.unstacked_input(1)
  new_axis = array_ops.where_v2(axis >= 0, axis + 1, axis)
  return wrap(gen_array_ops.reverse_v2(value, axis=new_axis), True)


@RegisterPForWithArgs("Transpose", gen_array_ops.transpose)
@RegisterPForWithArgs("ConjugateTranspose", gen_array_ops.conjugate_transpose)
def _convert_transpose(pfor_input: _PforInput, _, op_func):
  # [FUNCTION DOC]
  # Registered pfor converter for Transpose and ConjugateTranspose (op function supplied by
  # registration). Stacked tensor and loop-invariant permutation; the new permutation keeps the stack
  # axis first and shifts the others by one. Returns one stacked tensor.
  t = pfor_input.stacked_input(0)
  perm = pfor_input.unstacked_input(1)
  new_perm = array_ops.concat([[0], perm + 1], axis=0)
  return wrap(op_func(t, new_perm), True)


@RegisterPFor("ZerosLike")
def _convert_zeros_like(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for ZerosLike. Builds zeros with the per-iteration shape (stack
  # dimension removed) and the input's dtype. The result is loop invariant (not stacked).
  t = pfor_input.stacked_input(0)
  shape = array_ops.shape(t)[1:]
  return wrap(array_ops.zeros(shape, dtype=t.dtype), False)


@RegisterPFor("OnesLike")
def _convert_ones_like(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for OnesLike. Builds ones with the per-iteration shape and the input's
  # dtype. The result is loop invariant (not stacked).
  t = pfor_input.stacked_input(0)
  shape = array_ops.shape(t)[1:]
  return wrap(array_ops.ones(shape, dtype=t.dtype), False)


@RegisterPFor("Gather")
@RegisterPFor("GatherV2")
def _convert_gather(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # KNOWN ISSUE: if neither params nor indices is stacked the function falls off the end and returns
  # nothing; this is only safe because the driver never calls converters in that case.
  # Registered pfor converter for Gather and GatherV2. Reads batch_dims and, for V2, the loop-invariant axis (constant if possible). Case 1: stacked
  # indices with unstacked params: if the indices are exactly the full iteration index list and the
  # params size matches, it returns params directly; otherwise it moves the stack axis of the indices
  # to the batch_dims position if needed, gathers, and transposes the gathered axis to the front when
  # the axis is non-zero. Case 2: stacked params: stacks the indices too, shifts axis and batch_dims,
  # and gathers. Returns one stacked tensor.
  param, param_stacked, _ = pfor_input.input(0)
  indices, indices_stacked, _ = pfor_input.input(1)
  batch_dims = pfor_input.get_attr("batch_dims")

  op_type = pfor_input.op_type
  if op_type == "Gather":
    validate_indices = pfor_input.get_attr("validate_indices")
    axis = 0
  else:
    validate_indices = None
    # Assume we will never have a Tensor with rank > 2**32.
    axis = math_ops.cast(pfor_input.unstacked_input(2), dtypes.int32)
    axis_value = tensor_util.constant_value(axis)
    if axis_value is not None:
      axis = axis_value
  if indices_stacked and not param_stacked:
    if indices is pfor_input.pfor.all_indices and axis == 0:
      param_shape0 = tensor_shape.dimension_value(param.shape[0])
      indices_shape0 = tensor_shape.dimension_value(indices.shape[0])
      if param_shape0 is not None and indices_shape0 == param_shape0:
        # Note that with loops and conditionals, indices may not be contiguous.
        # However they will be sorted and unique. So if the shape matches, then
        # it must be picking up all the rows of param.
        return wrap(param, True)

    if batch_dims != 0:
      # Convert `batch_dims` to its positive equivalent if necessary.
      batch_dims_pos = batch_dims
      if batch_dims < 0:
        batch_dims_pos += array_ops.rank(indices)
      # In order to maintain
      #   indices.shape[:batch_dims] == params.shape[:batch_dims]
      # with stacked indices, we move the first dimension of `indices` to the
      # `batch_dims + 1`th position. The (non-batch) index dimensions will be
      # inserted into the shape of `output` at the `axis` dimension, which is
      # then transposed to the front (below).
      order = array_ops.concat([
          math_ops.range(1, batch_dims_pos + 1),
          [0],
          math_ops.range(batch_dims_pos + 1, array_ops.rank(indices))], axis=0)
      indices = array_ops.transpose(indices, order)

    output = array_ops.gather(
        param, indices, validate_indices=validate_indices, axis=axis,
        batch_dims=batch_dims)
    if axis != 0:
      axis = smart_cond.smart_cond(axis < 0,
                                   lambda: axis + array_ops.rank(param),
                                   lambda: ops.convert_to_tensor(axis))
      order = array_ops.concat(
          [[axis],
           math_ops.range(axis),
           math_ops.range(axis + 1, array_ops.rank(output))],
          axis=0)
      output = smart_cond.smart_cond(
          math_ops.equal(axis, 0), lambda: output,
          lambda: array_ops.transpose(output, order))
    return wrap(output, True)
  if param_stacked:
    pfor_input.stack_inputs(stack_indices=[1])
    indices = pfor_input.stacked_input(1)
    if isinstance(axis, tensor_lib.Tensor):
      axis = array_ops.where(axis >= 0, axis + 1, axis)
    else:
      axis = axis + 1 if axis >= 0 else axis
    batch_dims = batch_dims + 1 if batch_dims >= 0 else batch_dims
    output = array_ops.gather(param, indices, axis=axis, batch_dims=batch_dims)
    return wrap(output, True)


@RegisterPFor("GatherNd")
def _convert_gather_nd(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for GatherNd. Stacks the indices and requires stacked params (unstacked
  # params are not yet supported); gathers with one batch dimension. Returns one stacked tensor.
  # TODO(jmenick): Add support for unstacked params.
  pfor_input.stack_inputs(stack_indices=[1])
  params = pfor_input.stacked_input(0)
  indices = pfor_input.stacked_input(1)
  stacked_result = array_ops.gather_nd(params, indices, batch_dims=1)
  return wrap(stacked_result, True)


@RegisterPFor("ConcatV2")
def _convert_concatv2(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for ConcatV2. Stacks all value inputs, takes the last input as the
  # loop-invariant axis (shifted by one when non-negative) and concatenates. Returns one stacked
  # tensor.
  n = pfor_input.num_inputs
  pfor_input.stack_inputs(stack_indices=range(n - 1))
  axis = pfor_input.unstacked_input(n - 1)
  axis += math_ops.cast(axis >= 0, axis.dtype)
  return wrap(
      array_ops.concat([x.t for x in pfor_input.inputs[:n - 1]], axis=axis),
      True)


@RegisterPFor("StridedSlice")
def _convert_strided_slice(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for StridedSlice. Stacked input; loop-invariant begin, end, strides.
  # Prepends entries for the stack axis (begin and end zero, stride one), shifts the begin/end masks
  # (with the new bit set) and the ellipsis/new-axis/shrink-axis masks left by one, then slices.
  # Returns one stacked tensor.
  inp = pfor_input.stacked_input(0)
  begin = pfor_input.unstacked_input(1)
  end = pfor_input.unstacked_input(2)
  strides = pfor_input.unstacked_input(3)
  begin_mask = pfor_input.get_attr("begin_mask")
  end_mask = pfor_input.get_attr("end_mask")
  ellipsis_mask = pfor_input.get_attr("ellipsis_mask")
  new_axis_mask = pfor_input.get_attr("new_axis_mask")
  shrink_axis_mask = pfor_input.get_attr("shrink_axis_mask")

  begin = array_ops.concat([[0], begin], axis=0)
  end = array_ops.concat([[0], end], axis=0)
  strides = array_ops.concat([[1], strides], axis=0)
  begin_mask = begin_mask << 1 | 1
  end_mask = end_mask << 1 | 1
  ellipsis_mask <<= 1
  new_axis_mask <<= 1
  shrink_axis_mask <<= 1
  return wrap(
      array_ops.strided_slice(
          inp,
          begin,
          end,
          strides,
          begin_mask=begin_mask,
          end_mask=end_mask,
          ellipsis_mask=ellipsis_mask,
          new_axis_mask=new_axis_mask,
          shrink_axis_mask=shrink_axis_mask), True)


@RegisterPFor("StridedSliceGrad")
def _convert_strided_slice_grad(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for StridedSliceGrad. Loop-invariant shape, begin, end, strides and
  # stacked gradient. Prepends the loop length to the shape, the stack-axis entries to the slice
  # parameters, shifts the masks as in the forward converter, and calls the gradient op.
  # Returns one stacked tensor.
  shape = pfor_input.unstacked_input(0)
  begin = pfor_input.unstacked_input(1)
  end = pfor_input.unstacked_input(2)
  strides = pfor_input.unstacked_input(3)
  dy = pfor_input.stacked_input(4)
  begin_mask = pfor_input.get_attr("begin_mask")
  end_mask = pfor_input.get_attr("end_mask")
  ellipsis_mask = pfor_input.get_attr("ellipsis_mask")
  new_axis_mask = pfor_input.get_attr("new_axis_mask")
  shrink_axis_mask = pfor_input.get_attr("shrink_axis_mask")

  shape = array_ops.concat(
      [math_ops.cast(pfor_input.pfor.loop_len_vector, shape.dtype), shape],
      axis=0)
  begin = array_ops.concat([[0], begin], axis=0)
  end = array_ops.concat([[0], end], axis=0)
  strides = array_ops.concat([[1], strides], axis=0)
  begin_mask = begin_mask << 1 | 1
  end_mask = end_mask << 1 | 1
  ellipsis_mask <<= 1
  new_axis_mask <<= 1
  shrink_axis_mask <<= 1
  return wrap(
      array_ops.strided_slice_grad(
          shape,
          begin,
          end,
          strides,
          dy,
          begin_mask=begin_mask,
          end_mask=end_mask,
          ellipsis_mask=ellipsis_mask,
          new_axis_mask=new_axis_mask,
          shrink_axis_mask=shrink_axis_mask), True)


@RegisterPFor("CheckNumerics")
def _convert_check_numerics(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for CheckNumerics. Applies the check, with the original message
  # attribute, to the stacked input. Returns one stacked tensor. Side effect: runtime check op.
  t = pfor_input.stacked_input(0)
  message = pfor_input.get_attr("message")
  return wrap(gen_array_ops.check_numerics(t, message), True)


@RegisterPFor("EnsureShape")
def _convert_ensure_shape(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for EnsureShape. Prepends an unknown dimension to the shape attribute
  # and applies the op to the stacked input. Returns one stacked tensor.
  t = pfor_input.stacked_input(0)
  shape = tensor_shape.TensorShape(pfor_input.get_attr("shape"))
  return wrap(gen_array_ops.ensure_shape(t, [None] + shape), True)


# manip_ops


@RegisterPFor("Roll")
def _convert_roll(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for Roll. Stacked tensor, loop-invariant axis, possibly per-iteration
  # shift. If the shift is loop invariant it rolls with the axis shifted by one. Otherwise it sums the
  # shifts per axis per iteration (segment sum), builds for every output element the source
  # coordinates with wrap-around modulo, and gathers with a batch dimension, then reshapes to the
  # input shape. Returns one stacked tensor.
  t = pfor_input.stacked_input(0)
  shift, shift_stacked, _ = pfor_input.input(1)
  axis = pfor_input.unstacked_input(2)
  if not shift_stacked:
    return wrap(manip_ops.roll(t, shift, axis + 1), True)
  else:
    # `axis` and `shift` may both be vectors, with repeated axes summing the
    # corresponding `shift`s. We scatter shifts into a dense array of shape
    # [loop_len, num_unstacked_axes] indicating the offset for each axis.
    num_unstacked_axes = math_ops.cast(array_ops.rank(t), dtypes.int64) - 1
    axis = math_ops.cast(array_ops.reshape(axis, [-1]), dtypes.int64)
    loop_len = math_ops.cast(pfor_input.pfor.loop_len_vector[0], dtypes.int64)
    shift = math_ops.cast(array_ops.reshape(shift, [loop_len, -1]),
                          dtypes.int64)
    axis_segment_ids = (
        math_ops.range(loop_len, dtype=dtypes.int64)[:, None]
        * num_unstacked_axes + axis[None, :])
    axis_offsets = array_ops.reshape(
        math_ops.unsorted_segment_sum(
            data=shift, segment_ids=axis_segment_ids,
            num_segments=loop_len * num_unstacked_axes),
        [loop_len, num_unstacked_axes])

    # Determine the coordinates in the input array of each result and gather
    # them.
    unstacked_shape = array_ops.shape(t, out_type=dtypes.int64)[1:]
    cumsize = math_ops.cumprod(unstacked_shape, exclusive=True, reverse=True)
    num_unstacked_elements = math_ops.reduce_prod(unstacked_shape)
    result_coordinates = (
        (math_ops.range(num_unstacked_elements,
                        dtype=dtypes.int64)[None, :, None]
         // cumsize[None, None, :] - axis_offsets[:, None, :])
        % unstacked_shape[None, None, :])
    result_flat = array_ops.gather_nd(params=t, indices=result_coordinates,
                                      batch_dims=1)
    return wrap(array_ops.reshape(result_flat, array_ops.shape(t)),
                True)

# math_ops


@RegisterPFor("MatMul")
def _convert_matmul(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for MatMul. Handles three cases: both operands stacked (batched
  # matmul); only the left stacked (fold the iteration axis into rows, multiply, unfold); only the
  # right stacked (move the stack axis to be a column block with a transpose or cheap reshape,
  # multiply, unfold and transpose). Honors both transpose attributes. Returns one stacked tensor.
  # TODO(agarwal): Check if tiling is faster than two transposes.
  a, a_stacked, _ = pfor_input.input(0)
  b, b_stacked, _ = pfor_input.input(1)
  tr_a = pfor_input.get_attr("transpose_a")
  tr_b = pfor_input.get_attr("transpose_b")
  if a_stacked and b_stacked:
    output = wrap(
        math_ops.matmul(a, b, transpose_a=tr_a, transpose_b=tr_b), True
    )
    return output
  elif a_stacked:
    if tr_a:
      a = array_ops.transpose(a, [0, 2, 1])
    if a.shape.is_fully_defined():
      x, y, z = a.shape
    else:
      x, y, z = [
          array_ops.reshape(i, [])
          for i in array_ops.split(array_ops.shape(a), 3)
      ]
    a = array_ops.reshape(a, [x * y, z])
    prod = math_ops.matmul(a, b, transpose_b=tr_b)
    return wrap(array_ops.reshape(prod, [x, y, -1]), True)
  else:
    assert b_stacked
    if tr_b:
      perm = [2, 0, 1]
      b = array_ops.transpose(b, perm)
    else:
      # As an optimization, if one of the first two dimensions is 1, then we can
      # reshape instead of transpose.
      # TODO(agarwal): This check can be done inside Transpose kernel.
      b_shape = array_ops.shape(b)
      min_dim = math_ops.minimum(b_shape[0], b_shape[1])
      perm = array_ops.where(
          math_ops.equal(min_dim, 1), [0, 1, 2], [1, 0, 2])
      new_shape = array_ops_stack.stack([b_shape[1], b_shape[0], b_shape[2]])
      b = array_ops.transpose(b, perm)
      b = array_ops.reshape(b, new_shape)

    if b.shape.is_fully_defined():
      x, y, z = b.shape
    else:
      x, y, z = [
          array_ops.reshape(i, [])
          for i in array_ops.split(array_ops.shape(b), 3)
      ]
    b = array_ops.reshape(b, [x, y * z])
    prod = math_ops.matmul(a, b, transpose_a=tr_a)
    prod = array_ops.reshape(prod, [-1, y, z])
    prod = array_ops.transpose(prod, [1, 0, 2])
    return wrap(prod, True)


# TODO(rmlarsen): Use the converter of BatchMatMulV2 once compatibility window
# is met.
@RegisterPFor("BatchMatMul")
def _convert_batch_mat_mul(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for BatchMatMul (v1, no broadcasting). Stacks both operands, flattens
  # the iteration and batch axes, multiplies with the adjoint attributes, and splits the leading
  # dimension back. Returns one stacked tensor.
  # TODO(agarwal): There may be a more efficient way to do this instead of
  # stacking the inputs.
  pfor_input.stack_inputs()
  x = pfor_input.stacked_input(0)
  y = pfor_input.stacked_input(1)
  adj_x = pfor_input.get_attr("adj_x")
  adj_y = pfor_input.get_attr("adj_y")

  x = _flatten_first_two_dims(x)
  y = _flatten_first_two_dims(y)
  output = math_ops.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
  output = _unflatten_first_dim(output, pfor_input.pfor.loop_len_vector)
  return wrap(output, True)


@RegisterPFor("BatchMatMulV2")
def _convert_batch_mat_mul_v2(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for BatchMatMulV2. Reshapes the inputs for broadcast, then multiplies
  # using the adjoint attributes. Returns one stacked tensor.
  pfor_input.expanddim_inputs_for_broadcast()
  x = pfor_input.input(0)[0]
  y = pfor_input.input(1)[0]
  adj_x = pfor_input.get_attr("adj_x")
  adj_y = pfor_input.get_attr("adj_y")

  output = math_ops.matmul(x, y, adjoint_a=adj_x, adjoint_b=adj_y)
  return wrap(output, True)


@RegisterPForWithArgs("Sum", math_ops.reduce_sum)
@RegisterPForWithArgs("Prod", math_ops.reduce_prod)
@RegisterPForWithArgs("Max", math_ops.reduce_max)
@RegisterPForWithArgs("Min", math_ops.reduce_min)
@RegisterPForWithArgs("Mean", math_ops.reduce_mean)
@RegisterPForWithArgs("All", math_ops.reduce_all)
@RegisterPForWithArgs("Any", math_ops.reduce_any)
def _convert_reduction(pfor_input: _PforInput, _, op_func):
  # [FUNCTION DOC]
  # Registered pfor converter for reduction ops Sum, Prod, Max, Min, Mean, All and Any (function
  # supplied by registration). Stacked tensor and loop-invariant reduction indices; non-negative
  # indices are shifted by one; keep_dims attribute preserved. Returns one stacked tensor.
  t = pfor_input.stacked_input(0)
  indices = pfor_input.unstacked_input(1)
  # Shift positive indices by one to account for the extra dimension.
  indices += math_ops.cast(indices >= 0, indices.dtype)
  keep_dims = pfor_input.get_attr("keep_dims")
  return wrap(op_func(t, indices, keepdims=keep_dims), True)


@RegisterPForWithArgs("ArgMax", math_ops.argmax)
@RegisterPForWithArgs("ArgMin", math_ops.argmin)
def _convert_argmax_argmin(pfor_input: _PforInput, _, op_func):
  # [FUNCTION DOC]
  # Registered pfor converter for ArgMax and ArgMin. Stacked tensor and loop-invariant dimension
  # (shifted by one when non-negative); output_type attribute preserved. Returns one stacked tensor.
  t = pfor_input.stacked_input(0)
  dimension = pfor_input.unstacked_input(1)
  dimension += math_ops.cast(dimension >= 0, dimension.dtype)
  output_type = pfor_input.get_attr("output_type")
  return wrap(op_func(t, axis=dimension, output_type=output_type), True)


@RegisterPFor("Bucketize")
def _convert_bucketize(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for Bucketize. Applies the elementwise op to the stacked input using
  # the boundaries attribute. Returns one stacked tensor.
  t = pfor_input.stacked_input(0)
  boundaries = pfor_input.get_attr("boundaries")
  return wrap(math_ops.bucketize(t, boundaries), True)


@RegisterPFor("ClipByValue")
def _convert_clip_by_value(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for ClipByValue. Stacked tensor and loop-invariant min and max; calls
  # the low-level op. Returns one stacked tensor.
  t = pfor_input.stacked_input(0)
  clip_value_min = pfor_input.unstacked_input(1)
  clip_value_max = pfor_input.unstacked_input(2)
  return wrap(gen_math_ops._clip_by_value(t, clip_value_min, clip_value_max),
              True)


@RegisterPForWithArgs("Cumsum", math_ops.cumsum)
@RegisterPForWithArgs("Cumprod", math_ops.cumprod)
def _convert_cumfoo(pfor_input: _PforInput, _, op_func):
  # [FUNCTION DOC]
  # Registered pfor converter for Cumsum and Cumprod (function supplied by registration). Stacked
  # tensor, loop-invariant axis (shifted by one when non-negative), exclusive and reverse attributes
  # preserved. Returns one stacked tensor.
  t = pfor_input.stacked_input(0)
  axis = pfor_input.unstacked_input(1)
  # Shift positive indices by one to account for the extra dimension.
  axis += math_ops.cast(axis >= 0, axis.dtype)
  exclusive = pfor_input.get_attr("exclusive")
  reverse = pfor_input.get_attr("reverse")
  return wrap(op_func(t, axis, exclusive=exclusive, reverse=reverse), True)


@RegisterPFor("BiasAdd")
def _convert_biasadd(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for BiasAdd. If the bias is stacked, ordinary addition is used after
  # broadcasting reshapes (for the channels-first format the bias is rearranged). If only the value
  # is stacked, the iteration axis is merged into batch for the channels-first format (otherwise the
  # bias-add works directly) and the result is reshaped back. Returns one stacked tensor.
  t, t_stacked, _ = pfor_input.input(0)
  bias, bias_stacked, _ = pfor_input.input(1)
  data_format = pfor_input.get_attr("data_format").decode()
  if bias_stacked:
    # BiasAdd only supports 1-D biases, so cast bias to match value and use Add.
    pfor_input.expanddim_inputs_for_broadcast()
    t, _, _ = pfor_input.input(0)
    bias = math_ops.cast(pfor_input.stacked_input(1), t.dtype)
    if compat.as_bytes(data_format) == b"NCHW":
      b_shape = array_ops.shape(bias)
      new_b_shape = array_ops.concat(
          [b_shape[:-3], b_shape[-1:], b_shape[-3:-1]], axis=0)
      bias = array_ops.reshape(bias, new_b_shape)
    return wrap(math_ops.add(t, bias), True)
  else:
    assert t_stacked, "At least one input to BiasAdd should be loop variant."
    if compat.as_bytes(data_format) == b"NCHW":
      shape = array_ops.shape(t)
      flattened_shape = array_ops.concat([[-1], shape[2:]], axis=0)
      t = array_ops.reshape(t, flattened_shape)
      t = nn_ops.bias_add(t, bias, data_format="NCHW")
      t = array_ops.reshape(t, shape)
      return wrap(t, True)
    return wrap(nn_ops.bias_add(t, bias, data_format=data_format), True)


@RegisterPForWithArgs("UnsortedSegmentSum", math_ops.unsorted_segment_sum)
@RegisterPForWithArgs("UnsortedSegmentMax", math_ops.unsorted_segment_max)
@RegisterPForWithArgs("UnsortedSegmentMin", math_ops.unsorted_segment_min)
@RegisterPForWithArgs("UnsortedSegmentProd", math_ops.unsorted_segment_prod)
def _convert_unsortedsegmentsum(pfor_input: _PforInput, _, op_func):
  # [FUNCTION DOC]
  # Registered pfor converter for UnsortedSegmentSum/Max/Min/Prod (function supplied by
  # registration). Stacks data and segment ids, requires loop-invariant segment count, offsets
  # each iteration's non-negative segment ids by iteration index times segment count so all
  # iterations share one reduction (casting types if they differ), then reshapes the output to
  # separate the iteration axis. Returns one stacked tensor.
  pfor_input.stack_inputs([0, 1])
  data = pfor_input.stacked_input(0)
  segment_ids = pfor_input.stacked_input(1)
  # TODO(agarwal): handle stacked?
  num_segments = pfor_input.unstacked_input(2)
  if segment_ids.dtype != num_segments.dtype:
    segment_ids = math_ops.cast(segment_ids, dtypes.int64)
    num_segments = math_ops.cast(num_segments, dtypes.int64)
  dtype = segment_ids.dtype
  segment_shape = array_ops.shape(segment_ids, out_type=dtype)
  n = segment_shape[0]
  ones = array_ops.ones_like(segment_shape, dtype=dtype)[1:]
  segment_offset = num_segments * math_ops.range(n, dtype=dtype)
  segment_offset = array_ops.reshape(segment_offset,
                                     array_ops.concat([[n], ones], axis=0))
  segment_ids = array_ops.where(
      segment_ids >= 0, segment_ids + segment_offset, segment_ids
  )
  num_segments = math_ops.cast(num_segments, dtypes.int64) * math_ops.cast(
      n, dtypes.int64)
  output = op_func(data, segment_ids, num_segments)
  new_output_shape = array_ops.concat(
      [[n, -1], array_ops.shape(output)[1:]], axis=0)
  output = array_ops.reshape(output, new_output_shape)
  return wrap(output, True)


def _flatten_array_with_offset(ids, offset_delta, num_rows):
  # [FUNCTION DOC]
  # HELPER: flattens a rank-2 (or broadcast rank-1) id tensor while adding a per-row offset.
  # Inputs: ids, offset_delta (offset step per row) and num_rows. Row i receives offset i times the
  # delta, then everything is flattened. Returns the flat id tensor.
  """Flattens a rank 2 tensor, adding an offset to each row."""
  # Note that if `ids` is rank 1, it is broadcast to rank 2.
  offset_delta = math_ops.cast(offset_delta, ids.dtype)
  n = math_ops.cast(num_rows, dtype=ids.dtype)
  offsets = math_ops.range(
      start=0, limit=n * offset_delta, delta=offset_delta, dtype=ids.dtype)
  offsets = array_ops.expand_dims(offsets, -1)
  ids += offsets
  return array_ops.reshape(ids, [-1])


@RegisterPForWithArgs("SparseSegmentSum", math_ops.sparse_segment_sum_v2)
@RegisterPForWithArgs("SparseSegmentMean", math_ops.sparse_segment_mean_v2)
@RegisterPForWithArgs("SparseSegmentSqrtN", math_ops.sparse_segment_sqrt_n_v2)
@RegisterPForWithArgs("SparseSegmentSumWithNumSegments",
                      math_ops.sparse_segment_sum_v2)
@RegisterPForWithArgs("SparseSegmentMeanWithNumSegments",
                      math_ops.sparse_segment_mean_v2)
@RegisterPForWithArgs("SparseSegmentSqrtNWithNumSegments",
                      math_ops.sparse_segment_sqrt_n_v2)
def _convert_sparse_segment(pfor_input: _PforInput, _, op_func):
  # [FUNCTION DOC]
  # Registered pfor converter for the SparseSegment{Sum,Mean,SqrtN} ops with or without a segment
  # count (function supplied by registration). Segment ids must be loop invariant when no count is
  # given (otherwise output sizes could differ); stacks indices when ids are stacked. Flattens
  # indices (offset by data size when data is stacked) and segment ids (offset by segment count),
  # calls the op with a multiplied segment count, and splits the output back. Returns one stacked
  # tensor.
  _, segment_ids_stacked, _ = pfor_input.input(2)
  if segment_ids_stacked:
    pfor_input.stack_inputs([1])
  data, data_stacked, _ = pfor_input.input(0)
  indices, _, _ = pfor_input.input(1)
  num_inputs = len(pfor_input.inputs)
  assert num_inputs in (3, 4)
  if num_inputs == 3:
    # `segment_ids` needs to be unstacked since otherwise output sizes could
    # differ across pfor iterations.
    segment_ids = pfor_input.unstacked_input(2)
    num_segments = nn_ops.relu(math_ops.reduce_max(segment_ids) + 1)
  else:
    segment_ids, _, _ = pfor_input.input(2)
    num_segments = pfor_input.unstacked_input(3)

  n = pfor_input.pfor.loop_len_vector[0]
  if data_stacked:
    indices = _flatten_array_with_offset(indices, array_ops.shape(data)[1], n)
    data = _flatten_first_two_dims(data)
  else:
    indices = array_ops.reshape(indices, [-1])
  segment_ids = _flatten_array_with_offset(segment_ids, num_segments, n)

  if num_inputs == 3:
    num_segments = None
  else:
    num_segments *= n
  output = op_func(data, indices, segment_ids, num_segments=num_segments)
  output = _unflatten_first_dim(output, [n])
  return wrap(output, True)


@RegisterPForWithArgs("SparseSegmentSumGrad", math_ops.sparse_segment_sum_grad)
@RegisterPForWithArgs("SparseSegmentMeanGrad",
                      math_ops.sparse_segment_mean_grad)
@RegisterPForWithArgs("SparseSegmentSqrtNGrad",
                      math_ops.sparse_segment_sqrt_n_grad)
def _convert_sparse_segment_grad(pfor_input: _PforInput, _, op_func):
  # [FUNCTION DOC]
  # Registered pfor converter for SparseSegment{Sum,Mean,SqrtN}Grad. Stacked gradient; loop-invariant
  # indices, segment ids and output dimension. Flattens indices and segment ids with offsets, merges
  # the gradient's first two dims, scales the output size by the loop length, calls the gradient
  # function and splits. Returns one stacked tensor.
  grad = pfor_input.stacked_input(0)
  indices = pfor_input.unstacked_input(1)
  segment_ids = pfor_input.unstacked_input(2)
  dim0 = pfor_input.unstacked_input(3)

  n = pfor_input.pfor.loop_len_vector[0]
  indices = _flatten_array_with_offset(indices, dim0, n)
  num_segments = nn_ops.relu(math_ops.reduce_max(segment_ids) + 1)
  segment_ids = _flatten_array_with_offset(segment_ids, num_segments, n)
  grad = _flatten_first_two_dims(grad)
  dim0 *= n
  output = op_func(grad, indices, segment_ids, dim0)
  output = _unflatten_first_dim(output, [n])
  return wrap(output, True)


@RegisterPFor("Cast")
def _convert_cast(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for Cast. Casts the stacked input to the DstT attribute type.
  inp = pfor_input.stacked_input(0)
  dtype = pfor_input.get_attr("DstT")
  return wrap(math_ops.cast(inp, dtype), True)


@RegisterPFor("Abs")
@RegisterPFor("Acos")
@RegisterPFor("Acosh")
@RegisterPFor("Add")
@RegisterPFor("AddV2")
@RegisterPFor("Angle")
@RegisterPFor("Asin")
@RegisterPFor("Asinh")
@RegisterPFor("Atan")
@RegisterPFor("Atan2")
@RegisterPFor("Atanh")
@RegisterPFor("BesselI0")
@RegisterPFor("BesselI1")
@RegisterPFor("BesselI0e")
@RegisterPFor("BesselI1e")
@RegisterPFor("BesselK0")
@RegisterPFor("BesselK1")
@RegisterPFor("BesselK0e")
@RegisterPFor("BesselK1e")
@RegisterPFor("BesselJ0")
@RegisterPFor("BesselJ1")
@RegisterPFor("BesselY0")
@RegisterPFor("BesselY1")
@RegisterPFor("BitwiseAnd")
@RegisterPFor("BitwiseOr")
@RegisterPFor("BitwiseXor")
@RegisterPFor("Ceil")
@RegisterPFor("Complex")
@RegisterPFor("ComplexAbs")
@RegisterPFor("Conj")
@RegisterPFor("Cos")
@RegisterPFor("Cosh")
@RegisterPFor("Dawsn")
@RegisterPFor("Digamma")
@RegisterPFor("Div")
@RegisterPFor("DivNoNan")
@RegisterPFor("Elu")
@RegisterPFor("Erf")
@RegisterPFor("Erfc")
@RegisterPFor("Erfinv")
@RegisterPFor("Exp")
@RegisterPFor("Expint")
@RegisterPFor("Expm1")
@RegisterPFor("Floor")
@RegisterPFor("FloorDiv")
@RegisterPFor("FloorMod")
@RegisterPFor("FresnelCos")
@RegisterPFor("FresnelSin")
@RegisterPFor("Greater")
@RegisterPFor("GreaterEqual")
@RegisterPFor("Igamma")
@RegisterPFor("IgammaGradA")
@RegisterPFor("Igammac")
@RegisterPFor("Imag")
@RegisterPFor("Inv")
@RegisterPFor("Invert")
@RegisterPFor("IsFinite")
@RegisterPFor("IsInf")
@RegisterPFor("IsNan")
@RegisterPFor("LeftShift")
@RegisterPFor("Less")
@RegisterPFor("LessEqual")
@RegisterPFor("Lgamma")
@RegisterPFor("Log")
@RegisterPFor("Log1p")
@RegisterPFor("LogicalAnd")
@RegisterPFor("LogicalNot")
@RegisterPFor("LogicalOr")
@RegisterPFor("LogicalXor")
@RegisterPFor("Maximum")
@RegisterPFor("Minimum")
@RegisterPFor("Mod")
@RegisterPFor("Mul")
@RegisterPFor("MulNoNan")
@RegisterPFor("Ndtri")
@RegisterPFor("Neg")
@RegisterPFor("Polygamma")
@RegisterPFor("Pow")
@RegisterPFor("Real")
@RegisterPFor("RealDiv")
@RegisterPFor("Reciprocal")
@RegisterPFor("Relu")
@RegisterPFor("Relu6")
@RegisterPFor("RightShift")
@RegisterPFor("Rint")
@RegisterPFor("Round")
@RegisterPFor("Rsqrt")
@RegisterPFor("Selu")
@RegisterPFor("Sigmoid")
@RegisterPFor("Sign")
@RegisterPFor("Sin")
@RegisterPFor("Sinh")
@RegisterPFor("Softplus")
@RegisterPFor("Softsign")
@RegisterPFor("Spence")
@RegisterPFor("Sqrt")
@RegisterPFor("Square")
@RegisterPFor("SquaredDifference")
@RegisterPFor("Sub")
@RegisterPFor("Tan")
@RegisterPFor("Tanh")
@RegisterPFor("TruncateDiv")
@RegisterPFor("TruncateMod")
@RegisterPFor("Xdivy")
@RegisterPFor("Xlogy")
@RegisterPFor("Xlog1py")
@RegisterPFor("Zeta")
def _convert_cwise(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for a long list of elementwise unary/binary ops (arithmetic,
  # comparison, logical, bitwise, special functions, activations). If there are several inputs it
  # prepares them for broadcasting; then recreates the op with its attributes directly on the raw
  # inputs and returns the single output as stacked (asserts exactly one output).
  if pfor_input.num_inputs > 1:
    pfor_input.expanddim_inputs_for_broadcast()

  out = _create_op(
      pfor_input.op_type, [x.t for x in pfor_input.inputs],
      [x.dtype for x in pfor_input.outputs],
      attrs=pfor_input.op.node_def.attr).outputs
  assert len(out) == 1
  out = out[0]

  op_output = wrap(out, True)
  return op_output


@RegisterPFor("XlaSharding")
def _convert_xla_sharding(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for XlaSharding. Applies the sharding annotation, with the original
  # sharding attribute, to the stacked input. Returns one stacked tensor.
  t = pfor_input.stacked_input(0)
  sharding = pfor_input.get_attr("sharding")
  return wrap(xla.sharding(t, sharding=sharding), True)


@RegisterPFor("LeakyRelu")
def _convert_leaky_relu(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for LeakyRelu. Applies the op with the alpha attribute to the stacked
  # input. Returns one stacked tensor.
  t = pfor_input.stacked_input(0)
  alpha = pfor_input.get_attr("alpha")
  return wrap(gen_nn_ops.leaky_relu(t, alpha=alpha), True)


@RegisterPFor("Equal")
def _convert_equal(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for Equal. Prepares inputs for broadcasting, then applies the op with
  # the incompatible-shape-error attribute preserved. Returns one stacked tensor.
  pfor_input.expanddim_inputs_for_broadcast()
  x = pfor_input.input(0)[0]
  y = pfor_input.input(1)[0]
  incompatible_shape_error = pfor_input.get_attr("incompatible_shape_error")
  return wrap(gen_math_ops.equal(
      x, y, incompatible_shape_error=incompatible_shape_error), True)


@RegisterPFor("NotEqual")
def _convert_not_equal(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for NotEqual. Same pattern as Equal. Returns one stacked tensor.
  pfor_input.expanddim_inputs_for_broadcast()
  x = pfor_input.input(0)[0]
  y = pfor_input.input(1)[0]
  incompatible_shape_error = pfor_input.get_attr("incompatible_shape_error")
  return wrap(gen_math_ops.not_equal(
      x, y, incompatible_shape_error=incompatible_shape_error), True)


@RegisterPFor("ApproximateEqual")
def _convert_approximate_equal(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for ApproximateEqual. Prepares inputs for broadcasting and applies the
  # op with the tolerance attribute. Returns one stacked tensor.
  pfor_input.expanddim_inputs_for_broadcast()
  x = pfor_input.input(0)[0]
  y = pfor_input.input(1)[0]
  tolerance = pfor_input.get_attr("tolerance")
  return wrap(math_ops.approximate_equal(x, y, tolerance=tolerance), True)


@RegisterPFor("Shape")
def _convert_shape(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for Shape. Computes the shape of the stacked input with the out_type
  # attribute and drops the stack dimension. The result is loop invariant.
  out_type = pfor_input.get_attr("out_type")
  return wrap(
      array_ops.shape(pfor_input.stacked_input(0), out_type=out_type)[1:],
      False)


@RegisterPFor("ShapeN")
def _convert_shape_n(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for ShapeN. For each input computes its shape, dropping the stack
  # dimension for stacked inputs. Returns a list of loop-invariant tensors.
  out_type = pfor_input.get_attr("out_type")
  shapes = [
      array_ops.shape(x, out_type=out_type)[1:] if stacked else array_ops.shape(
          x, out_type=out_type) for x, stacked, _ in pfor_input.inputs
  ]
  return [wrap(x, False) for x in shapes]


@RegisterPFor("Size")
def _convert_size(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for Size. Computes the element count of the stacked input and divides
  # by the loop length. Result is loop invariant.
  out_type = pfor_input.get_attr("out_type")
  n = math_ops.cast(pfor_input.pfor.loop_len_vector[0], out_type)
  return wrap(
      array_ops.size(pfor_input.stacked_input(0), out_type=out_type) // n,
      False)


@RegisterPFor("Rank")
def _convert_rank(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for Rank. Rank of the stacked input minus one; loop invariant.
  return wrap(array_ops.rank(pfor_input.stacked_input(0)) - 1, False)


@RegisterPFor("AddN")
def _convert_addn(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for AddN (no broadcasting support). Stacks all inputs without tiling
  # variants and adds them; variant results are tiled to the loop length. Returns one stacked tensor.
  # AddN does not support broadcasting.
  pfor_input.stack_inputs(tile_variants=False)
  return _wrap_and_tile_variants(
      math_ops.add_n([x.t for x in pfor_input.inputs]),
      pfor_input.pfor.loop_len_vector)


@RegisterPFor("Cross")
def _convert_cross(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for Cross. Stacks both inputs and applies the cross product. Returns one
  # stacked tensor.
  pfor_input.stack_inputs()
  a = pfor_input.stacked_input(0)
  b = pfor_input.stacked_input(1)
  return wrap(math_ops.cross(a, b), True)


@RegisterPFor("BiasAddGrad")
def _convert_biasaddgrad(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for BiasAddGrad. Stacked gradient. For the channels-first format sums
  # over axes 1, 3 and 4; otherwise reshapes to (iterations, -1, channels) and sums the middle axis.
  # Returns one stacked tensor.
  grad = pfor_input.stacked_input(0)
  fmt = pfor_input.get_attr("data_format")
  if fmt == b"NCHW":
    output = math_ops.reduce_sum(grad, axis=[1, 3, 4], keepdims=False)
  else:
    grad_shape = array_ops.shape(grad)
    last_dim_shape = grad_shape[-1]
    first_dim_shape = grad_shape[0]
    output = array_ops.reshape(grad, [first_dim_shape, -1, last_dim_shape])
    output = math_ops.reduce_sum(output, axis=[1], keepdims=False)
  return wrap(output, True)


# Some required ops are not exposed under the tf namespace. Hence relying on
# _create_op to create them.
@RegisterPForWithArgs("EluGrad")
@RegisterPForWithArgs("LeakyReluGrad")
@RegisterPForWithArgs("ReciprocalGrad")
@RegisterPForWithArgs("Relu6Grad")
@RegisterPForWithArgs("ReluGrad")
@RegisterPForWithArgs("RsqrtGrad")
@RegisterPForWithArgs("SeluGrad")
@RegisterPForWithArgs("SigmoidGrad")
@RegisterPForWithArgs("SoftplusGrad")
@RegisterPForWithArgs("SoftsignGrad")
@RegisterPForWithArgs("SqrtGrad")
@RegisterPForWithArgs("TanhGrad")
def _convert_grads(pfor_input: _PforInput, op_type, *args, **kw_args):
  # [FUNCTION DOC]
  # Registered pfor converter for activation gradient ops (Elu, LeakyRelu, Reciprocal, Relu6, Relu,
  # Rsqrt, Selu, Sigmoid, Softplus, Softsign, Sqrt, Tanh gradients). These ops do not broadcast, so
  # every input is stacked, and the op is recreated through the low-level creator (it is not exposed
  # in the public namespace). Extra registration arguments are ignored. Returns a list of stacked
  # tensors.
  del args
  del kw_args
  # TODO(agarwal): Looks like these ops don't support broadcasting. Hence we
  # have to use tiling here.
  pfor_input.stack_inputs()
  outputs = _create_op(
      op_type, [x.t for x in pfor_input.inputs],
      [x.dtype for x in pfor_input.outputs],
      attrs=pfor_input.op.node_def.attr).outputs
  return [wrap(x, True) for x in outputs]


@RegisterPFor("Select")
def _convert_select(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the final list comprehension iterates over the op outputs but wraps the same single
  # merged result for each, so it only behaves correctly because Select has exactly one output.
  # Registered pfor converter for Select (v1). Stacks all inputs. If the condition has rank greater
  # than one, the iteration axis is merged into the batch axis (decided at graph run time with a
  # conditional) before recreating the op, and split afterwards. Returns a list of stacked tensors.
  pfor_input.stack_inputs()
  cond = pfor_input.stacked_input(0)
  t = pfor_input.stacked_input(1)
  e = pfor_input.stacked_input(2)
  cond_rank = array_ops.rank(cond)
  cond, t, e = smart_cond.smart_cond(
      cond_rank > 1, lambda: _inputs_with_flattening(pfor_input, [0, 1, 2]),
      lambda: [cond, t, e])
  outputs = _create_op(
      pfor_input.op_type, [cond, t, e], [x.dtype for x in pfor_input.outputs],
      attrs=pfor_input.op.node_def.attr).outputs
  n = pfor_input.pfor.loop_len_vector
  out = smart_cond.smart_cond(cond_rank > 1,
                              lambda: _unflatten_first_dim(outputs[0], n),
                              lambda: outputs[0])
  return [wrap(out, True) for x in outputs]


@RegisterPFor("SelectV2")
def _convert_selectv2(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for SelectV2. Prepares inputs for broadcasting and applies the
  # broadcasting where function. Returns one stacked tensor.
  pfor_input.expanddim_inputs_for_broadcast()
  cond = pfor_input.input(0)[0]
  t = pfor_input.input(1)[0]
  e = pfor_input.input(2)[0]
  out = array_ops.where_v2(cond, t, e)
  return wrap(out, True)


# random_ops


def _transpose_dim_to_front(x, dim):
  # [FUNCTION DOC]
  # HELPER: moves one dimension of a tensor to the front. Inputs: tensor and dimension index (may be
  # a tensor). Transposes with a permutation built from the dimension followed by the remaining ones.
  rank = array_ops.rank(x)
  return array_ops.transpose(
      x,
      perm=array_ops.concat(
          [[dim], math_ops.range(0, dim),
           math_ops.range(dim + 1, rank)],
          axis=0))


@RegisterPForWithArgs("RandomUniform")
@RegisterPForWithArgs("RandomUniformInt")
@RegisterPForWithArgs("RandomStandardNormal")
@RegisterPForWithArgs("TruncatedNormal")
def _convert_random(pfor_input: _PforInput, op_type, *args, **kw_args):
  # [FUNCTION DOC]
  # Registered pfor converter for stateful RandomUniform, RandomUniformInt, RandomStandardNormal and
  # TruncatedNormal. All inputs must be loop invariant. Prepends the loop length to the shape input
  # and recreates the op, so the vectorized call draws independent samples per iteration (values may
  # differ from a sequential run; the warning about that is commented out). Extra args ignored.
  # Returns a list of stacked tensors.
  del args
  del kw_args
  inputs = [pfor_input.unstacked_input(i) for i in range(pfor_input.num_inputs)]
  # inputs[0] is "shape"
  n = math_ops.cast(pfor_input.pfor.loop_len_vector, inputs[0].dtype)
  inputs[0] = array_ops.concat([n, inputs[0]], axis=0)
  # TODO(b/222761732): Turn this warning back on when legacy RNGs are
  #   deprecated.
  # logging.warning(
  #     "Note that %s inside pfor op may not give same output as "
  #     "inside a sequential loop.", op_type)
  outputs = _create_op(
      op_type,
      inputs, [x.dtype for x in pfor_input.outputs],
      attrs=pfor_input.op.node_def.attr).outputs
  return [wrap(x, True) for x in outputs]


@RegisterPFor("RandomGamma")
@RegisterPFor("RandomPoissonV2")
def _convert_random_with_param(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for RandomGamma and RandomPoissonV2. Shape must be loop invariant. If
  # the parameter (alpha or rate) is stacked, sampling happens with the original shape and the
  # sample dimensions are moved so the iteration axis leads; otherwise the loop length is prepended
  # to the shape. Returns one stacked tensor.
  shape = pfor_input.unstacked_input(0)
  # param is lam (Poisson rate) or alpha (Gamma shape).
  param, param_stacked, _ = pfor_input.input(1)
  # TODO(b/222761732): Turn this warning back on when legacy RNGs are
  #   deprecated.
  # logging.warning(
  #     "Note that %s inside pfor op may not give same output as "
  #     "inside a sequential loop.", pfor_input.op_type)

  if param_stacked:
    samples = _create_op(
        pfor_input.op_type,
        inputs=[shape, param],
        op_dtypes=[x.dtype for x in pfor_input.outputs],
        attrs=pfor_input.op.node_def.attr).outputs[0]
    loop_dim = array_ops.shape(shape)[0]
    stacked_samples = _transpose_dim_to_front(samples, loop_dim)
  else:
    n = math_ops.cast(pfor_input.pfor.loop_len_vector, shape.dtype)
    shape = array_ops.concat([n, shape], axis=0)
    stacked_samples = _create_op(
        pfor_input.op_type,
        inputs=[shape, param],
        op_dtypes=[x.dtype for x in pfor_input.outputs],
        attrs=pfor_input.op.node_def.attr).outputs[0]

  return wrap(stacked_samples, True)


@RegisterPFor("Multinomial")
def _convert_multinomial(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for Multinomial. Loop-invariant sample count. If logits are stacked,
  # flattens the iteration axis into the batch, samples and unflattens. Otherwise draws loop-length
  # times as many samples and rearranges them so each iteration has its own set. Uses the seed
  # attributes and output dtype. Returns one stacked tensor.
  logits, logits_stacked, _ = pfor_input.input(0)
  num_samples = pfor_input.unstacked_input(1)
  seed = pfor_input.get_attr("seed")
  seed2 = pfor_input.get_attr("seed2")
  output_dtype = pfor_input.get_attr("output_dtype")
  # TODO(b/222761732): Turn this warning back on when legacy RNGs are
  #   deprecated.
  # logging.warning(
  #     "Note that Multinomial inside pfor op may not give same output as "
  #     "inside a sequential loop.")

  n = pfor_input.pfor.loop_len_vector[0]
  if logits_stacked:
    flattened_logits = _flatten_first_two_dims(logits)
    samples = gen_random_ops.multinomial(
        flattened_logits,
        num_samples,
        seed=seed,
        seed2=seed2,
        output_dtype=output_dtype)
    stacked_samples = _unflatten_first_dim(samples, [n])
  else:
    samples = gen_random_ops.multinomial(
        logits,
        num_samples * n,
        seed=seed,
        seed2=seed2,
        output_dtype=output_dtype)
    stacked_samples = array_ops.transpose(
        array_ops.reshape(samples, [-1, n, num_samples]), [1, 0, 2])

  return wrap(stacked_samples, True)


@RegisterPFor("StatelessMultinomial")
@RegisterPFor("StatelessParameterizedTruncatedNormal")
@RegisterPFor("StatelessRandomBinomial")
@RegisterPFor("StatelessRandomGammaV2")
@RegisterPFor("StatelessRandomNormal")
@RegisterPFor("StatelessRandomPoisson")
@RegisterPFor("StatelessRandomUniform")
@RegisterPFor("StatelessRandomUniformInt")
@RegisterPFor("StatelessRandomUniformFullInt")
@RegisterPFor("StatelessTruncatedNormal")
def _convert_stateless_multinomial(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the name says multinomial, but this converter is registered for ten stateless
  # random ops (binomial, gamma, normal, poisson, uniform variants, truncated normal and others).
  # Registered pfor converter that always uses the sequential fallback, so each iteration gets
  # reproducible results from its own seed; vectorizing would change the random numbers. No warning
  # is logged. Returns the stacked outputs from the fallback.
  # Unlike stateful random ops, for stateless ones we want better
  # reproducibility based on seed. Hence we don't want to use a similar strategy
  # as used for stateful ones where we generate a possibly different set of
  # random numbers under vectorization.
  # Unfortunately, the kernels currently are not necessarily setup to do this
  # efficiently and hence we fallback to a sequential loop for vectorization.
  return _fallback_converter(pfor_input, warn=False)


# linalg_ops


@RegisterPForWithArgs("XlaEinsum")
@RegisterPForWithArgs("Einsum")
def _convert_einsum(pfor_input: _PforInput, op_type):
  # [FUNCTION DOC]
  # Registered pfor converter for Einsum and XlaEinsum. Parses the equation attribute, picks an
  # unused letter as the new iteration axis symbol (value error if none is available), prefixes it to
  # every stacked input term and to the output, then runs the einsum (XLA version or the regular one)
  # with the new equation. At least one input must be stacked. Returns one stacked tensor.
  # Einsum may have either 1 or 2 inputs.
  inputs, input_stacked, _ = zip(*[
      pfor_input.input(i)
      for i in range(pfor_input.num_inputs)])

  # Parse the einsum equation.
  equation = pfor_input.get_attr("equation").decode("utf-8")
  input_expr, output_expr = equation.split("->")
  input_exprs = input_expr.split(",")

  # Pick a placeholder symbol to use for the new axis.
  chosen_symbol = None
  for s in string.ascii_letters:
    if s in equation:
      continue
    else:
      chosen_symbol = s
      break

  if chosen_symbol is None:
    raise ValueError("Could not figure out what symbol to use for new axis.")

  assert any(input_stacked)
  for i in range(len(inputs)):
    if input_stacked[i]:
      input_exprs[i] = "{}{}".format(chosen_symbol, input_exprs[i])
  output_expr = "{}{}".format(chosen_symbol, output_expr)

  new_equation = "{}->{}".format(",".join(input_exprs), output_expr)

  if op_type == "XlaEinsum":
    if len(inputs) == 1:
      result = xla.einsum(equation=new_equation, a=inputs[0])
    else:
      result = xla.einsum(equation=new_equation, a=inputs[0], b=inputs[1])
  else:
    assert op_type == "Einsum"
    result = special_math_ops.einsum(new_equation, *inputs)

  return wrap(result, True)


@RegisterPFor("Cholesky")
def _convert_cholesky(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for Cholesky. Applies it to the stacked matrices. Returns one stacked
  # tensor.
  t = pfor_input.stacked_input(0)
  return wrap(linalg_ops.cholesky(t), True)


@RegisterPFor("LogMatrixDeterminant")
def _convert_log_matrix_determinant(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for LogMatrixDeterminant. Applies it to the stacked matrices and
  # returns a list with both outputs (sign and log determinant) stacked.
  t = pfor_input.stacked_input(0)
  return [wrap(x, True) for x in linalg_ops.log_matrix_determinant(t)]


@RegisterPFor("MatrixInverse")
def _convert_matrix_inverse(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for MatrixInverse. Stacked matrices; adjoint attribute preserved.
  # Returns one stacked tensor.
  t = pfor_input.stacked_input(0)
  adjoint = pfor_input.get_attr("adjoint")
  return wrap(gen_linalg_ops.matrix_inverse(t, adjoint=adjoint), True)


@RegisterPFor("MatrixSolve")
def _convert_matrix_solve(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for MatrixSolve. Stacks both inputs, preserves the adjoint attribute.
  # Returns one stacked tensor.
  pfor_input.stack_inputs()
  matrix = pfor_input.stacked_input(0)
  rhs = pfor_input.stacked_input(1)
  adjoint = pfor_input.get_attr("adjoint")
  output = gen_linalg_ops.matrix_solve(
      matrix, rhs, adjoint=adjoint)
  return wrap(output, True)


@RegisterPFor("MatrixTriangularSolve")
def _convert_matrix_triangular_solve(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for MatrixTriangularSolve. Prepares inputs for broadcasting; preserves
  # the lower and adjoint attributes. Returns one stacked tensor.
  pfor_input.expanddim_inputs_for_broadcast()
  matrix = pfor_input.input(0)[0]
  rhs = pfor_input.input(1)[0]
  lower = pfor_input.get_attr("lower")
  adjoint = pfor_input.get_attr("adjoint")
  output = linalg_ops.matrix_triangular_solve(
      matrix, rhs, lower=lower, adjoint=adjoint)
  return wrap(output, True)


@RegisterPFor("SelfAdjointEigV2")
def _convert_self_adjoint_eig(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for SelfAdjointEigV2. Stacked matrices; compute_v attribute preserved.
  # Returns the eigenvalues as stacked, and the eigenvectors stacked only if compute_v is true (the
  # tensor has shape zero otherwise).
  t = pfor_input.stacked_input(0)
  compute_v = pfor_input.get_attr("compute_v")
  e, v = gen_linalg_ops.self_adjoint_eig_v2(t, compute_v=compute_v)
  # If compute_v is False, v will have shape [0].
  return wrap(e, True), wrap(v, compute_v)


# logging_ops


@RegisterPFor("Assert")
def _convert_assert(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for Assert. If the condition is stacked it is reduced with logical-and
  # over all iterations. Recreates the Assert op with the condition and the raw data inputs; the op has
  # no outputs, so the new operation is returned (not a wrapped tensor). Side effect: runtime check.
  cond, cond_stacked, _ = pfor_input.input(0)
  if cond_stacked:
    cond = math_ops.reduce_all(cond)

  data_list = [x.t for x in pfor_input.inputs][1:]
  return _create_op(
      "Assert", [cond] + data_list, [], attrs=pfor_input.op.node_def.attr)


@RegisterPFor("Print")
def _convert_print(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for Print. Only the first input is stacked; other inputs are printed
  # once (not per iteration). Recreates the op and returns a list of stacked outputs. Side effect:
  # prints at run time.
  # Note that we don't stack all the inputs. Hence unstacked values are printed
  # once here vs multiple times in a while_loop.
  pfor_input.stack_inputs([0])
  outputs = _create_op(
      "Print", [x.t for x in pfor_input.inputs],
      [x.dtype for x in pfor_input.outputs],
      attrs=pfor_input.op.node_def.attr).outputs
  return [wrap(x, True) for x in outputs]


@RegisterPFor("PrintV2")
def _convert_print_v2(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for PrintV2. Recreates the op on the raw inputs, printing the full
  # tensors including the batch dimension when stacked. Returns the new operation. Side effect:
  # prints at run time.
  # Print the full input Tensor(s), including the batch dimension if stacked.
  return _create_op(
      "PrintV2", [x.t for x in pfor_input.inputs],
      [x.dtype for x in pfor_input.outputs],
      attrs=pfor_input.op.node_def.attr)


@RegisterPFor("StringFormat")
def _convert_string_format(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for StringFormat. Recreates the op on the raw inputs, so the formatted
  # text includes the stacked dimension. Returns outputs as loop-invariant (unstacked) tensors.
  # Format using the full input Tensor(s), including the batch dimension if
  # stacked.
  op = _create_op(
      "StringFormat", [x.t for x in pfor_input.inputs],
      [x.dtype for x in pfor_input.outputs],
      attrs=pfor_input.op.node_def.attr)
  return [wrap(output, False) for output in op.outputs]


# data_flow_ops

# TensorArray conversion is tricky since we don't support arrays of
# TensorArrays. For converting them, we consider two distinct cases:
#
# 1. The array is constructed outside the pfor call, and read/written inside the
# loop.
# This is an easier case since we don't need to make an array of TensorArrays.
# A correctness requirement is that these parallel iterations shouldn't attempt
# to write to the same location. Hence at conversion time we disallow indices to
# be loop-invariant as that would guarantee a collision. Even if the indices are
# not loop-invariant, they could conflict and that shall trigger runtime errors.
#
# 2. The array is constructed and used entirely inside each pfor iteration.
# For simplicity, here we require that the indices used for write/scatter are
# "unstacked". Otherwise it becomes hard to merge the TensorArrays created in
# different pfor iterations. We consider two sub_cases:
#
# 2a Elements written to the array are "stacked"
# To simulate multiple TensorArrays, we may increase the dimension of each
# element of the array. i.e. the i_th row of the j_th entry of the converted
# TensorArray corresponds to the j_th entry of the TensorArray in the i_th
# pfor iteration.
#
# 2b Elements written to the array are "unstacked"
# In this case we don't increase the dimensions to avoid redundant tiling. Each
# iteration is trying to write the same value. So we convert that to a single
# write.
#
# Here are some tricks used to implement the above:
# - TensorArrayV3 constructor encodes the element shape as an attr. Instead of
# trying to trace whether future writes are stacked or unstacked in order to set
# this attr, we set it to correspond to unknown shape.
# - We use the "flow" output of the different ops to track whether the array
# elements are stacked or unstacked. If a stacked write/scatter is done, we make
# the flow stacked as well.
# - We use some heuristic traversal of the graph to track whether the
# TensorArray handle was created inside or outside the pfor loop.


@RegisterPFor("TensorArrayV3")
def _convert_tensor_array_v3(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for TensorArrayV3 creation. Requires a loop-invariant size. Creates a
  # new array with the same attributes but an unknown element shape (writes may be stacked or not).
  # Returns the handle and the flow, both unstacked for now.
  size = pfor_input.unstacked_input(0)
  dtype = pfor_input.get_attr("dtype")
  dynamic_size = pfor_input.get_attr("dynamic_size")
  clear_after_read = pfor_input.get_attr("clear_after_read")
  identical_element_shapes = pfor_input.get_attr("identical_element_shapes")
  tensor_array_name = pfor_input.get_attr("tensor_array_name")
  handle, flow = data_flow_ops.tensor_array_v3(
      size,
      dtype=dtype,
      # We don't set element shape since we don't know if writes are stacked or
      # not yet.
      element_shape=None,
      dynamic_size=dynamic_size,
      clear_after_read=clear_after_read,
      identical_element_shapes=identical_element_shapes,
      tensor_array_name=tensor_array_name)
  # Note we keep flow unstacked for now since we don't know if writes will be
  # stacked or not.
  return wrap(handle, False), wrap(flow, False)


@RegisterPFor("TensorArraySizeV3")
def _convert_tensor_array_size_v3(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for TensorArraySizeV3. Loop-invariant handle; unstacks the flow if it
  # is stacked; returns the (unstacked) size.
  handle = pfor_input.unstacked_input(0)
  flow, flow_stacked, _ = pfor_input.input(1)
  if flow_stacked:
    flow = _unstack_flow(flow)
  size = data_flow_ops.tensor_array_size_v3(handle, flow)
  return wrap(size, False)


def _handle_inside_pfor(pfor_input: _PforInput, handle):
  # [FUNCTION DOC]
  # HELPER: decides whether a TensorArray handle was created inside the pfor body.
  # Inputs: pfor_input and a handle tensor. Walks back through Enter and Identity ops to find the
  # creating op; raises a value error if the source is not a TensorArray creation or gradient op.
  # Heuristic: does not handle switch/merge or user manipulation of handles. Returns a boolean.
  """Returns True if handle was created inside the pfor loop."""
  # We use some heuristic to find the original TensorArray creation op.
  # The logic should handle the common cases (except cond based subgraphs).
  # In theory the user could perform different operations on the handle (like
  # Reshape, stack multiple handles, etc) which could break this logic.
  # TODO(agarwal): handle Switch/Merge.
  while handle.op.type in ("Enter", "Identity"):
    handle = handle.op.inputs[0]
  if handle.op.type not in [
      "TensorArrayV3", "TensorArrayGradV3", "TensorArrayGradWithShape"
  ]:
    raise ValueError(f"Unable to find source for handle {handle}.")
  else:
    return pfor_input.pfor.op_is_inside_loop(handle.op)


def _unstack_flow(value):
  # [FUNCTION DOC]
  # HELPER: takes a stacked TensorArray flow value and returns the unstacked flow by gathering index
  # zero (all entries are the same).
  # TODO(agarwal): consider looking if this is a Tile op then get its input.
  # This may avoid running the Tile operations.
  return array_ops.gather(value, 0)


@RegisterPFor("TensorArrayReadV3")
def _convert_tensor_array_read_v3(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for TensorArrayReadV3. Loop-invariant handle; flow unstacked if needed.
  # If the array was created inside the loop: stacked indices with stacked flow raise a value error
  # (unsupported); stacked indices with unstacked flow gather; otherwise a plain read (with a gather by
  # the active indices when the flow is stacked and the iteration set is partitioned). If created
  # outside, values are unstacked, so the result is stacked exactly when the index is. Returns one
  # wrapped tensor.
  handle = pfor_input.unstacked_input(0)
  index, index_stacked, _ = pfor_input.input(1)
  dtype = pfor_input.get_attr("dtype")
  flow, flow_stacked, _ = pfor_input.input(2)
  if flow_stacked:
    flow = _unstack_flow(flow)

  is_inside_pfor = _handle_inside_pfor(pfor_input, pfor_input.op.inputs[0])
  if is_inside_pfor:
    # Note that if we are inside a control flow construct inside the pfor, and
    # only some of the iterations are doing the read (i.e.
    # `all_indices_partitioned` is True), then the read operation should only
    # return values for the currently active pfor iterations (`all_indices`
    # below). Hence, whenever the returned value is stacked (i.e. `flow` is
    # stacked), we may need to do an extra gather after reading the values. Also
    # note that if `is_inside` is false, then values in the tensor array are
    # unstacked. So the check is only needed in this branch.
    all_indices = pfor_input.pfor.all_indices
    all_indices_partitioned = pfor_input.pfor.all_indices_partitioned
    # Note: flow_stacked indicates if values in the TensorArray are stacked or
    # not.
    if index_stacked:
      if flow_stacked:
        raise ValueError(
            "It looks like TensorArrayReadV3 was called on a TensorArray whose"
            " values are not loop-invariant, and the read indices were also"
            " not loop invariant. This is currently unsupported.")
      value = data_flow_ops.tensor_array_gather_v3(
          handle, index, flow, dtype=dtype)
      return wrap(value, True)
    value = data_flow_ops.tensor_array_read_v3(handle, index, flow, dtype=dtype)
    if flow_stacked and all_indices_partitioned:
      value = array_ops.gather(value, all_indices)
    return wrap(value, flow_stacked)
  # Values in the TensorArray should be unstacked (since different iterations
  # couldn't write to the same location). So whether output is stacked or not
  # depends on index_stacked.
  if index_stacked:
    value = data_flow_ops.tensor_array_gather_v3(
        handle, index, flow, dtype=dtype)
  else:
    value = data_flow_ops.tensor_array_read_v3(handle, index, flow, dtype=dtype)
  return wrap(value, index_stacked)


@RegisterPFor("TensorArrayWriteV3")
def _convert_tensor_array_write_v3(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for TensorArrayWriteV3. Raises a value error for stacked values inside
  # partitioned control flow. For arrays created inside the loop, indices must be loop invariant
  # (value error otherwise); unstacked write keeps flow unstacked, while a stacked value or flow
  # stacks the value and the resulting flow. For arrays created outside, indices must be stacked
  # (value error otherwise) and the write becomes a scatter. Returns the (possibly stacked) flow.
  handle = pfor_input.unstacked_input(0)
  index, index_stacked, _ = pfor_input.input(1)
  value, value_stacked, _ = pfor_input.input(2)
  flow, flow_stacked, _ = pfor_input.input(3)
  if value_stacked and pfor_input.pfor.all_indices_partitioned:
    # Looks like we are in a control flow in a pfor where not all iterations are
    # active now. We don't allow that since that could lead to different indices
    # having different shapes which will be hard to merge later.
    raise ValueError("Writing non loop invariant values to TensorArray from "
                     "inside a while_loop/cond not supported.")
  if flow_stacked:
    flow = _unstack_flow(flow)
  is_inside = _handle_inside_pfor(pfor_input, pfor_input.op.inputs[0])
  if is_inside:
    if index_stacked:
      raise ValueError(f"Need indices for {handle} to be loop invariant.")
    if not flow_stacked and not value_stacked:
      flow_out = data_flow_ops.tensor_array_write_v3(handle, index, value, flow)
      return wrap(flow_out, False)
    else:
      if not value_stacked:
        value = _stack(value, pfor_input.pfor.loop_len_vector).t
      # TODO(agarwal): Note that if flow is unstacked and value is stacked, then
      # this may or may not be a safe situation. flow is unstacked both for a
      # freshly created TensorArray, as well as after unstacked values are
      # written to it. If it is the latter, then we cannot write a stacked value
      # now since that may cause runtime errors due to different shapes in the
      # array. At the moment we are not able to handle this gracefully and
      # distinguish between the two cases. That would require some heuristic
      # traversal of the graph to figure out whether all the writes are
      # unstacked or not.
      flow_out = data_flow_ops.tensor_array_write_v3(handle, index, value, flow)
      return _stack(flow_out, pfor_input.pfor.loop_len_vector)
  else:
    if not index_stacked:
      raise ValueError(f"Need indices for {handle} to be not loop invariant.")
    # Note that even when index_stacked is true, actual values in index may
    # still not be unique. However that will cause runtime error when executing
    # the scatter operation below.
    if not value_stacked:
      value = _stack(value, pfor_input.pfor.loop_len_vector).t
    flow_out = data_flow_ops.tensor_array_scatter_v3(handle, index, value, flow)
    return _stack(flow_out, pfor_input.pfor.loop_len_vector)


def _transpose_first_two_dims(value):
  # [FUNCTION DOC]
  # HELPER: swaps the first two dimensions of a tensor (via a three-dimensional reshape, transpose and
  # reshape back). Returns the tensor.
  # TODO(agarwal): optimize if one of the dims == 1.
  value_shape = array_ops.shape(value)
  v0 = value_shape[0]
  v1 = value_shape[1]
  value = array_ops.reshape(value, [v0, v1, -1])
  value = array_ops.transpose(value, [1, 0, 2])
  new_shape = array_ops.concat([[v1, v0], value_shape[2:]], axis=0)
  return array_ops.reshape(value, new_shape)


@RegisterPFor("TensorArrayGatherV3")
def _convert_tensor_array_gather_v3(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for TensorArrayGatherV3. Flattens indices, unstacks flow. For arrays
  # created inside the loop: stacked indices with stacked flow raise a value error; stacked indices
  # unflatten the gathered values; unstacked indices keep values, transposing the first two dims when
  # the flow is stacked. For arrays created outside: unflatten when indices are stacked. Element
  # shape attribute support is a TODO. Returns one wrapped tensor.
  handle = pfor_input.unstacked_input(0)
  indices, indices_stacked, _ = pfor_input.input(1)
  indices = array_ops.reshape(indices, [-1])
  flow, flow_stacked, _ = pfor_input.input(2)
  if flow_stacked:
    flow = _unstack_flow(flow)
  dtype = pfor_input.get_attr("dtype")
  # TODO(agarwal): support element_shape attr?

  n = pfor_input.pfor.loop_len_vector
  value = data_flow_ops.tensor_array_gather_v3(
      handle, indices, flow, dtype=dtype)
  is_inside = _handle_inside_pfor(pfor_input, pfor_input.op.inputs[0])
  if is_inside:
    # flow_stacked indicates if values in the TensorArray are stacked or not.
    if indices_stacked:
      if flow_stacked:
        raise ValueError(
            "It looks like TensorArrayGatherV3 was called on a TensorArray "
            "whose values are not loop-invariant, and the indices were also "
            "not loop invariant. This is currently unsupported.")
      else:
        value = _unflatten_first_dim(value, n)
        return wrap(value, True)
    else:
      if flow_stacked:
        # Since elements in this array are stacked and `value` was produced by
        # gather, its first two dims are "gathered elements" and "stack
        # dimension". Our semantics require these two to be flipped.
        value = _transpose_first_two_dims(value)
      return wrap(value, flow_stacked)
  else:
    # Values in the TensorArray should be unstacked (since different iterations
    # couldn't write to the same location). So whether output is stacked or not
    # depends on indices_stacked.
    if indices_stacked:
      value = _unflatten_first_dim(value, n)
    return wrap(value, indices_stacked)


@RegisterPFor("TensorArrayScatterV3")
def _convert_tensor_array_scatter_v3(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for TensorArrayScatterV3. Flattens indices; unstacks flow. For arrays
  # created inside the loop: indices must be loop invariant (value error); an all-unstacked scatter
  # stays unstacked, otherwise the value is stacked, its first two dims transposed, scattered, and the
  # flow stacked. For arrays created outside: indices must be stacked (value error otherwise) and the
  # value is flattened before scattering. Returns the flow.
  handle = pfor_input.unstacked_input(0)
  indices, indices_stacked, _ = pfor_input.input(1)
  indices = array_ops.reshape(indices, [-1])
  value, value_stacked, _ = pfor_input.input(2)
  flow, flow_stacked, _ = pfor_input.input(3)

  if flow_stacked:
    flow = _unstack_flow(flow)

  is_inside = _handle_inside_pfor(pfor_input, pfor_input.op.inputs[0])
  if is_inside:
    if indices_stacked:
      raise ValueError(f"Need indices for {handle} to be loop invariant.")
    # Note that flow_stacked indicates if existing values in the array are
    # stacked or not.
    if not flow_stacked and not value_stacked:
      flow_out = data_flow_ops.tensor_array_scatter_v3(handle, indices, value,
                                                       flow)
      return wrap(flow_out, False)
    if not value_stacked:
      # TODO(agarwal): tile in the second dimension directly instead of
      # transposing below.
      value = _stack(value, pfor_input.pfor.loop_len_vector).t

    value = _transpose_first_two_dims(value)
    # TODO(agarwal): Note that if a previous write was unstacked, flow will be
    # unstacked, and a stacked value may be written here which may cause
    # runtime error due to different elements having different shape. We do
    # not try to prevent that.
    flow_out = data_flow_ops.tensor_array_scatter_v3(handle, indices, value,
                                                     flow)
    return _stack(flow_out, pfor_input.pfor.loop_len_vector)
  if not indices_stacked:
    raise ValueError(f"Need indices for {handle} to be not loop invariant.")
  if not value_stacked:
    value = _stack(value, pfor_input.pfor.loop_len_vector).t
  value = _flatten_first_two_dims(value)
  flow_out = data_flow_ops.tensor_array_scatter_v3(handle, indices, value, flow)
  return _stack(flow_out, pfor_input.pfor.loop_len_vector)


@RegisterPFor("TensorArrayGradV3")
def _convert_tensor_array_grad_v3(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for TensorArrayGradV3. Loop-invariant handle, unstacked flow, source
  # attribute. Assumes gradients will be stacked and so creates the gradient array with the loop
  # length prepended to the element shape. Returns the gradient handle (unstacked) and flow (stacked).
  handle = pfor_input.unstacked_input(0)
  flow, flow_stacked, _ = pfor_input.input(1)
  if flow_stacked:
    flow = _unstack_flow(flow)
  source = pfor_input.get_attr("source")
  # TODO(agarwal): For now, we assume that gradients are stacked if the
  # TensorArrayGradV3 call is being done inside the pfor. Getting that wrong
  # will give runtime error due to incorrect shape being written to the
  # accumulator. It is difficult to know in advance if gradients written will be
  # stacked or not. Note that flow being stacked is not indicative of the
  # gradient being stacked or not. Revisit this later.
  shape_to_prepend = pfor_input.pfor.loop_len_vector
  grad_handle, flow_out = data_flow_ops.tensor_array_grad_with_shape(
      handle=handle,
      flow_in=flow,
      shape_to_prepend=shape_to_prepend,
      source=source)
  flow_out = _stack(flow_out, pfor_input.pfor.loop_len_vector).t
  return [wrap(grad_handle, False), wrap(flow_out, True)]


def _stack_tensor_list_shape(shape, first_dim):
  # [FUNCTION DOC]
  # HELPER: computes the element shape of a stacked TensorList.
  # KNOWN ISSUE: when the shape is a constant that contains no negative entries and is not empty,
  # no branch returns, so the function returns nothing instead of the loop length followed by the
  # shape.
  # Inputs: element shape (tensor) and the first-dimension vector (loop length). If the constant shape
  # has a negative (unknown) entry, returns minus one; if empty, returns the first dimension; for a
  # non-constant shape, picks at run time between minus one and the concatenation.
  shape_value = tensor_util.constant_value(shape)
  # Note that negative values in the shape are used to signify unknown shapes
  # and are handled in a special way.
  if shape_value is not None:
    shape_value = numpy_compat.np_asarray(shape_value)
    if -1 in shape_value:
      return constant_op.constant(-1)
    elif not shape_value.size:
      return first_dim
  else:
    shape = array_ops.reshape(shape, [-1])
    return tf_cond.cond(
        math_ops.reduce_any(shape < 0),
        lambda: constant_op.constant(-1),
        lambda: array_ops.concat([first_dim, shape], axis=0))


def _tile_variant_with_length(t, length):
  # [FUNCTION DOC]
  # HELPER: tiles a scalar variant tensor to the loop length.
  # Inputs: variant tensor and loop-length vector. Variants that are TensorLists or optionals are
  # returned unchanged (their contents are vectorized). Otherwise sets the static shape to scalar
  # (mutating the tensor's metadata), reshapes to a vector and tiles on the CPU device, copying the
  # handle data to the result. Returns the tiled tensor.
  """stacks `t` `length` times."""
  if _is_variant_with_internal_stacking(t):
    # The content of TensorLists is vectorized, not the variant itself.
    return t
  original_tensor = t
  t.set_shape([])
  t = array_ops.reshape(t, [-1])
  with ops.device("CPU:0"):
    result = array_ops.tile(t, length)
    # TODO(b/169968286): Should regular shape functions do handle data
    # propagation here?
    handle_data_util.copy_handle_data(original_tensor, result)
    return result


def _tile_variant(t, pfor_input: _PforInput):
  # [FUNCTION DOC]
  # HELPER: tiles a variant by the loop length taken from the pfor input; wraps the length-based
  # variant tiling.
  """stacks `t` according to its loop context."""
  return _tile_variant_with_length(t, pfor_input.pfor.loop_len_vector)


def _untile_variant(t):
  # [FUNCTION DOC]
  # HELPER: inverse of tiling a variant. For TensorLists/optionals the tensor must be scalar-compatible
  # (raises an assertion error otherwise) and is returned unchanged; for other variants returns the
  # first element.
  if _is_variant_with_internal_stacking(t):
    # The content of TensorLists is vectorized, not the variant itself.
    if not t.shape.is_compatible_with([]):
      raise AssertionError(
          ("Unexpectedly saw a vectorized variant (e.g. TensorList) with "
           f"non-scalar shape: {t!r}"))
    return t
  return array_ops.gather(t, 0)


@RegisterPFor("OptionalFromValue")
def _convert_optional_from_value(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for OptionalFromValue. Stacks all inputs and creates a single optional
  # from them; the result is the scalar variant marked as stacked.
  pfor_input.stack_inputs()
  return wrap(
      gen_optional_ops.optional_from_value([x.t for x in pfor_input.inputs]),
      True,
  )


@RegisterPFor("OptionalGetValue")
def _convert_optional_get_value(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for OptionalGetValue. Stacked optional handle. Prepends the loop length
  # (when statically known, else unknown) to each output shape and calls the op with the output
  # types. Returns a list of stacked tensors.
  handle = pfor_input.stacked_input(0)
  output_types = pfor_input.get_attr("output_types")
  original_output_shapes = pfor_input.get_attr("output_shapes")
  output_shapes = []
  for shape in original_output_shapes:
    shape = tensor_shape.TensorShape(shape)
    loop_len_value = tensor_util.constant_value(pfor_input.pfor.loop_len_vector)
    loop_len_shape = tensor_shape.TensorShape(
        [loop_len_value[0] if loop_len_value is not None else None]
    )
    shape = loop_len_shape.concatenate(shape)
    output_shapes.append(shape.as_proto())
  results = gen_optional_ops.optional_get_value(
      handle, output_types, output_shapes
  )
  return [wrap(t, True) for t in results]


@RegisterPFor("TensorListReserve")
def _convert_tensor_list_reserve(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for TensorListReserve. Loop-invariant element shape and count. Prepends
  # the loop length to the element shape, reserves the list and tiles the handle. Returns one stacked
  # wrapped variant. (Forced to be treated as stateful so this runs.)
  element_shape = pfor_input.unstacked_input(0)
  num_elements = pfor_input.unstacked_input(1)
  element_dtype = pfor_input.get_attr("element_dtype")

  # Prepend a dimension to element_shape.
  element_shape = _stack_tensor_list_shape(element_shape,
                                           pfor_input.pfor.loop_len_vector)
  handle = list_ops.tensor_list_reserve(
      element_shape, num_elements, element_dtype=element_dtype)

  return wrap(_tile_variant(handle, pfor_input), True)


@RegisterPFor("TensorListElementShape")
def _convert_tensor_list_element_shape(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for TensorListElementShape. Un-tiles the handle, reads the element
  # shape, flattens it and drops the stack entry. Returns a loop-invariant tensor.
  handle = _untile_variant(pfor_input.stacked_input(0))
  shape_type = pfor_input.get_attr("shape_type")
  shape = list_ops.tensor_list_element_shape(handle, shape_type)
  shape = array_ops.reshape(shape, [-1])
  shape = shape[1:]
  return wrap(shape, False)


@RegisterPFor("TensorListLength")
def _convert_tensor_list_length(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for TensorListLength. Un-tiles the handle and returns the length as a
  # loop-invariant tensor.
  handle = _untile_variant(pfor_input.stacked_input(0))
  return wrap(list_ops.tensor_list_length(handle), False)


def _stack_tensor_list(handle, dtype, loop_len_vector, element_shape=None):
  # [FUNCTION DOC]
  # HELPER: converts a loop-invariant TensorList into one whose elements are stacked.
  # Inputs: handle, element dtype, loop-length vector and optional element shape (default read from
  # the list). Reserves a new list with stacked element shape, then loops over all elements, stacking
  # each and storing it. Returns the new list handle.
  if element_shape is None:
    element_shape = list_ops.tensor_list_element_shape(handle, dtypes.int32)
  length = list_ops.tensor_list_length(handle)
  new_handle = list_ops.tensor_list_reserve(
      _stack_tensor_list_shape(element_shape, loop_len_vector), length, dtype)

  def _body_fn(i, h):
    # [FUNCTION DOC]
    # Nested loop body for stacking a list: for index i and the new list, reads element i of the
    # original list, stacks it and sets it in the new list; returns the next index and list.
    elem = list_ops.tensor_list_get_item(handle, i, dtype, element_shape)
    elem = _stack(elem, loop_len_vector).t
    return i + 1, list_ops.tensor_list_set_item(h, i, elem)

  return while_loop.while_loop(lambda i, _: i < length, _body_fn,
                               [0, new_handle])[1]


@RegisterPFor("TensorListGetItem")
def _convert_tensor_list_get_item(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for TensorListGetItem. If the list is stacked: with stacked indices it
  # maps over iterations sequentially (cheaper than gather plus concat), each reading its own item
  # and slicing its iteration; with an unstacked index it reads directly. If the list is not stacked
  # the index must be stacked (assertion) and a list gather is used. Returns one stacked tensor.
  handle, handle_stacked, _ = pfor_input.input(0)
  index, index_stacked, _ = pfor_input.input(1)
  element_shape = pfor_input.unstacked_input(2)
  element_dtype = pfor_input.get_attr("element_dtype")

  if handle_stacked:
    handle = _untile_variant(handle)
    element_shape = _stack_tensor_list_shape(element_shape,
                                             pfor_input.pfor.loop_len_vector)
    if index_stacked:
      # We use a sequential loop since that may be more efficient than first
      # gathering and concatenating all the element corresponding to `index`,
      # and then doing a gather on it.
      def _map_fn(i):
        # [FUNCTION DOC]
        # Nested function mapped over iteration ids: gets the item at this iteration's index from the list
        # and gathers the slice for this iteration. Returns that slice.
        item_i = list_ops.tensor_list_get_item(
            handle,
            index[i],
            element_dtype=element_dtype)
        return array_ops.gather(item_i, i)

      output = map_fn.map_fn(_map_fn, pfor_input.pfor.all_indices)
      return wrap(output, True)
    else:
      output = list_ops.tensor_list_get_item(
          handle,
          index,
          element_shape=element_shape,
          element_dtype=element_dtype)
      return wrap(output, True)
  else:
    assert index_stacked
    return wrap(
        list_ops.tensor_list_gather(
            handle,
            index,
            element_shape=element_shape,
            element_dtype=element_dtype), True)


@RegisterPFor("TensorListSetItem")
def _convert_tensor_array_set_item(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the name mentions a tensor array, but this is the converter for TensorListSetItem.
  # Registered pfor converter for TensorListSetItem. If the list is unstacked and the index equals the
  # full iteration index list, items are scattered into the list (stacking the item if needed) and
  # the list stays unstacked. Otherwise the list is stacked (or un-tiled). Stacked (loop-variant)
  # indices raise a value error (unsupported). Otherwise the item is stacked, set, and the handle is
  # tiled. Returns one stacked wrapped variant (or an unstacked one in the scatter case).
  handle, handle_stacked, _ = pfor_input.input(0)
  index, index_stacked, _ = pfor_input.input(1)
  item, item_stacked, _ = pfor_input.input(2)

  if not handle_stacked:
    # Special case where we can statically guarantee that the indices are
    # disjoint.
    if index is pfor_input.pfor.all_indices:
      if not item_stacked:
        item = _stack(item, pfor_input.pfor.loop_len_vector).t
      return wrap(
          list_ops.tensor_list_scatter(item, index, input_handle=handle), False)
    else:
      handle = _stack_tensor_list(handle, item.dtype,
                                  pfor_input.pfor.loop_len_vector)
  else:
    handle = _untile_variant(handle)

  if index_stacked:
    # TODO(agarwal): handle this.
    raise ValueError("Vectorizing writes to a TensorList with loop "
                     "variant indices is currently unsupported.")

  else:
    if not item_stacked:
      item = _stack(item, pfor_input.pfor.loop_len_vector).t
    handle = list_ops.tensor_list_set_item(handle, index, item)
    return wrap(_tile_variant(handle, pfor_input), True)


@RegisterPFor("TensorListPushBack")
def _convert_tensor_list_push_back(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for TensorListPushBack. Un-tiles or stacks the list, stacks the pushed
  # tensor if needed, pushes and tiles the result. Returns one stacked variant.
  handle, handle_stacked, _ = pfor_input.input(0)
  tensor, tensor_stacked, _ = pfor_input.input(1)
  if handle_stacked:
    handle = _untile_variant(handle)
  else:
    handle = _stack_tensor_list(handle, tensor.dtype,
                                pfor_input.pfor.loop_len_vector)
  if not tensor_stacked:
    tensor = _stack(tensor, pfor_input.pfor.loop_len_vector).t
  handle = list_ops.tensor_list_push_back(handle, tensor)
  return wrap(_tile_variant(handle, pfor_input), True)


@RegisterPFor("TensorListPopBack")
def _convert_tensor_array_push_back(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the name says tensor array push back, but this converts TensorListPopBack.
  # Registered pfor converter for TensorListPopBack. Stacked list handle (un-tiled); loop-invariant
  # element shape; if the shape is a scalar (unspecified) uses minus one, else prepends the loop
  # length. Pops with the element dtype attribute. Returns the new handle and the popped tensor, both
  # stacked.
  handle = pfor_input.stacked_input(0)
  element_shape = pfor_input.unstacked_input(1)
  handle = _untile_variant(handle)

  if element_shape.shape.ndims == 0:
    # Default / unspecified
    vectorized_shape = -1
  else:
    # PopBack has an element shape set when it's the gradient of PushBack, only
    # used when the list is uninitialized.
    n = math_ops.cast(pfor_input.pfor.loop_len_vector, element_shape.dtype)
    vectorized_shape = array_ops.concat([n, element_shape], axis=0)

  output_handle, tensor = gen_list_ops.tensor_list_pop_back(
      input_handle=handle, element_dtype=pfor_input.get_attr("element_dtype"),
      element_shape=vectorized_shape)
  return wrap(output_handle, True), wrap(tensor, True)


@RegisterPFor("TensorListConcatV2")
def _convert_tensor_list_concat_v2(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for TensorListConcatV2. Stacked list; loop-invariant element shape and
  # leading dims. Finds the dynamic element shape from the first element (or a fixed fallback for an
  # empty list), builds a copy of the list with each element's first two dims transposed using a
  # while loop, concatenates, and transposes the result back. Returns the stacked concatenated tensor
  # and the unstacked lengths.
  input_handle = pfor_input.stacked_input(0)
  element_shape = pfor_input.unstacked_input(1)
  leading_dims = pfor_input.unstacked_input(2)
  element_dtype = pfor_input.get_attr("element_dtype")

  handle = _untile_variant(input_handle)
  length = list_ops.tensor_list_length(handle)
  # Note that element_shape attribute can have incomplete shapes. This doesn't
  # seem to work well when creating another list and then doing a concat on it.
  # Hence we try to find the dynamic shape here.
  element_shape = tf_cond.cond(
      length > 0, lambda: array_ops.shape(
          list_ops.tensor_list_get_item(handle, 0, element_dtype, None)),
      lambda: constant_op.constant([0, 0], dtype=dtypes.int32))
  # The code below creates a copy of the list with each elements' first two
  # dimensions transposed.
  new_element_shape = array_ops.concat(
      [element_shape[1:2], element_shape[0:1], element_shape[2:]], axis=0)

  # Create a new TensorList with elements transposed.
  def _transpose_elem(i, h):
    # [FUNCTION DOC]
    # Nested loop body: reads element i of the original list, swaps its first two dims, stores it in
    # the new list at i and returns the next index and the list.
    elem = list_ops.tensor_list_get_item(handle, i, element_dtype, None)
    elem = _transpose_first_two_dims(elem)
    return i + 1, list_ops.tensor_list_set_item(h, i, elem)

  new_handle = list_ops.tensor_list_reserve(new_element_shape, length,
                                            element_dtype)
  new_handle = while_loop.while_loop(lambda i, _: i < length, _transpose_elem,
                                     [0, new_handle])[1]
  output, lengths = gen_list_ops.tensor_list_concat_v2(
      input_handle=new_handle,
      element_dtype=element_dtype,
      element_shape=new_element_shape,
      leading_dims=leading_dims)
  output = _transpose_first_two_dims(output)
  return wrap(output, True), wrap(lengths, False)


@RegisterPFor("TensorListStack")
def _convert_tensor_list_stack(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for TensorListStack. Stacked list, loop-invariant element shape, with
  # element dtype and number-of-elements attributes. Stacks the list and swaps the first two dims so
  # the iteration axis comes first. Returns one stacked tensor.
  handle = pfor_input.stacked_input(0)
  input_shape = pfor_input.unstacked_input(1)
  element_dtype = pfor_input.get_attr("element_dtype")
  num_elements = pfor_input.get_attr("num_elements")

  handle = _untile_variant(handle)
  input_shape = _stack_tensor_list_shape(input_shape,
                                         pfor_input.pfor.loop_len_vector)
  output = list_ops.tensor_list_stack(
      handle,
      element_dtype,
      element_shape=input_shape,
      num_elements=num_elements)
  output = _transpose_first_two_dims(output)
  return wrap(output, True)


@RegisterPFor("TensorListGather")
def _convert_tensor_list_gather(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for TensorListGather. If the list is stacked: stacked indices map
  # sequentially over iterations (gather then slice along the last index axis); unstacked indices
  # gather directly. If the list is unstacked the indices must be stacked: flattens the indices,
  # gathers and reshapes to the index shape plus the element shape. Returns one stacked tensor.
  handle, handle_stacked, _ = pfor_input.input(0)
  index, index_stacked, _ = pfor_input.input(1)
  element_shape = pfor_input.unstacked_input(2)
  element_dtype = pfor_input.get_attr("element_dtype")

  if handle_stacked:
    handle = _untile_variant(handle)
    element_shape = _stack_tensor_list_shape(element_shape,
                                             pfor_input.pfor.loop_len_vector)
    if index_stacked:
      # We use a sequential loop since that may be more efficient than first
      # gathering and concatenating all the element corresponding to `index`,
      # and then doing a gather on it.
      def _map_fn(i):
        # [FUNCTION DOC]
        # Nested function mapped over iteration ids: gathers the list entries given by this iteration's
        # index row, then selects this iteration's slice along the last index axis. Returns the slice.
        item_i = list_ops.tensor_list_gather(
            handle,
            index[i],
            element_dtype=element_dtype)
        axis = array_ops.rank(index) - 1
        return array_ops.gather(item_i, i, axis=axis)

      output = map_fn.map_fn(_map_fn, pfor_input.pfor.all_indices)
      return wrap(output, True)
    else:
      output = list_ops.tensor_list_gather(
          handle,
          index,
          element_shape=element_shape,
          element_dtype=element_dtype)
      return wrap(output, True)
  else:
    assert index_stacked
    index_shape = array_ops.shape(index)
    index = array_ops.reshape(index, [-1])
    values = list_ops.tensor_list_gather(
        handle, index, element_shape=element_shape, element_dtype=element_dtype)
    final_shape = array_ops.concat(
        [index_shape, array_ops.shape(values)[1:]], axis=0)
    return wrap(array_ops.reshape(values, final_shape), True)


@RegisterPFor("TensorListScatterIntoExistingList")
def _convert_tensor_list_scatter(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for TensorListScatterIntoExistingList. Stacks the items; stacks or
  # un-tiles the list; swaps the first two dims of the items. Unstacked indices scatter directly.
  # Stacked indices: extracts the unique indices, gathers those list entries, updates them with a
  # scatter on (position, iteration) pairs, then scatters the chunks back (note: last update wins,
  # not guaranteed on GPU). Returns one stacked variant.
  pfor_input.stack_inputs([1])
  handle, handle_stacked, _ = pfor_input.input(0)
  item = pfor_input.stacked_input(1)
  indices, indices_stacked, _ = pfor_input.input(2)
  if handle_stacked:
    handle = _untile_variant(handle)
  else:
    handle = _stack_tensor_list(handle, item.dtype,
                                pfor_input.pfor.loop_len_vector)

  item = _transpose_first_two_dims(item)
  if indices_stacked:
    # Pretend the list is a dense tensor:
    #   list_as_dense: Tensor[list_len, loop_len, ...]
    # And indices are a tensor with shape (before transpose):
    #   indices: Tensor[loop_len, num_scatters]
    # The item to scatter has shape (before transpose):
    #   item: Tensor[loop_len, num_scatters, ...]
    #
    # We want list_as_dense[indices[i, j], i] = item[i, j]
    #
    # Since we're not just indexing along the first axis of `list_as_dense`, we
    # need to first extract the relevant list entries based on `indices`,
    # scatter into them according to the loop index, and re-scatter the chunks
    # we updated back into the list.
    indices = _transpose_first_two_dims(indices)
    indices_flat = array_ops.reshape(indices, [-1])
    # In many cases `indices` will be unique across pfor iterations, but this is
    # not guaranteed. If there are duplicates, we need to map multiple updates
    # to a single chunk extracted from the list. The last update should win.
    unique_indices = array_ops.unique(indices_flat)
    gathered_items = list_ops.tensor_list_gather(
        handle, unique_indices.y, element_dtype=item.dtype,
        element_shape=array_ops.shape(item)[1:])
    loop_idx = math_ops.range(pfor_input.pfor.loop_len_vector[0])
    scatters_per_op = array_ops.shape(indices)[0]

    unique_indices_loop_idx = array_ops.reshape(array_ops.tile(
        loop_idx[None, :], [scatters_per_op, 1]), [-1])
    scatter_indices = array_ops_stack.stack(
        [unique_indices.idx, unique_indices_loop_idx],
        axis=1)
    # This op does *not* guarantee last-update-wins on GPU, so semantics may not
    # be exactly preserved for duplicate updates there.
    scattered = array_ops.tensor_scatter_nd_update(
        tensor=gathered_items,
        indices=scatter_indices,
        updates=_flatten_first_two_dims(item))
    handle = list_ops.tensor_list_scatter(
        scattered, unique_indices.y, input_handle=handle)
  else:
    handle = list_ops.tensor_list_scatter(item, indices, input_handle=handle)
  return wrap(_tile_variant(handle, pfor_input), True)


@RegisterPFor("TensorListFromTensor")
def _convert_tensor_list_from_tensor(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for TensorListFromTensor. Stacked tensor and loop-invariant element
  # shape; swaps the first two dims, stacks the element shape, builds the list and tiles the handle.
  # Returns one stacked variant.
  tensor = pfor_input.stacked_input(0)
  element_shape = pfor_input.unstacked_input(1)
  tensor = _transpose_first_two_dims(tensor)
  element_shape = _stack_tensor_list_shape(element_shape,
                                           pfor_input.pfor.loop_len_vector)
  handle = list_ops.tensor_list_from_tensor(tensor, element_shape)
  return wrap(_tile_variant(handle, pfor_input), True)


@RegisterPFor("TensorScatterUpdate")
def _convert_tensor_scatter_update(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for TensorScatterUpdate. Stacks tensor, indices and updates, builds an
  # index grid of iteration ids matching the indices, prepends it as a new first coordinate and calls
  # the scatter update. Returns one stacked tensor.
  pfor_input.stack_inputs([0, 1, 2])
  tensor = pfor_input.stacked_input(0)
  indices = pfor_input.stacked_input(1)
  updates = pfor_input.stacked_input(2)

  indices_shape = array_ops.shape(indices)
  indices_rank = array_ops.rank(indices)
  loop_length = indices_shape[0]

  # Create a loop count range and extend its dimensions to match `indices`.
  loop_count_shape = array_ops.tensor_scatter_nd_update(
      array_ops.ones([indices_rank], dtype=dtypes.int32), [[0]], [loop_length])
  loop_count = array_ops.reshape(math_ops.range(loop_length), loop_count_shape)

  # Tile the loop count range for the batch dimensions (all except the first and
  # last dimensions of indices).
  # Rank(indices) >= 3 always for this function so we always have at least 1.
  tile_multiplier = array_ops.tensor_scatter_nd_update(
      indices_shape, [[0], [indices_rank - 1]], [1, 1])
  meta_index = array_ops.tile(loop_count, tile_multiplier)

  # Insert the loop-identifying index.
  indices = array_ops.concat([meta_index, indices], axis=-1)

  result = array_ops.tensor_scatter_nd_update(tensor, indices, updates)
  return wrap(result, True)

# StackV2 conversion is tricky since we don't have arrays of StackV2. So similar
# to TensorArrays, we convert them by changing the dimension of the elements
# inside the stack.
#
# We consider two cases:
#
# 1. StackV2 is constructed and used entirely inside the pfor loop.
# We keep a single Stack and perform the push/pop operations of all the
# iterations in lock-step. We also assume that all the iterations perform these
# operations. In case of dynamic control flow, if only some of the iterations
# try to perform a push/pop, then the conversion may not work correctly and may
# cause undefined behavior.
# TODO(agarwal): test StackV2 with dynamic control flow.
#
# 2. StackV2 is constructed outside the pfor loop.
# Performing stack push/pop in a parallel fashion is ill-defined. However given
# that reading stacks created externally is a common operation when computing
# jacobians, we provide some special semantics here as follows.
#  - disallow push operations to the stack
#  - pop operations are performed in lock step by all iterations, similar to the
#  case when the stack is created inside. A single value is popped during the
#  lock-step operation and broadcast to all the iterations. Values in the stack
#  are assumed to be loop-invariant.
#
# Some other implementation details:
# We use an ugly logic to find whether values in Stack data structure are
# loop invariant or not. When converting push/pop operations, we keep track of
# whether the last conversion used a stacked value or not (see _stack_cache
# below). As a result if an unstacked value is written first, subsequent stacked
# writes are disallowed when they could have been allowed in theory.

# Map from cache key based on StackV2 handle to a bool indicating whether values
# are stacked or not.
# TODO(agarwal): move _stack_cache inside pfor?
_stack_cache = {}


def _stack_cache_key(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # HELPER: builds the key into the module-level stack cache for a stack push/pop converter.
  # Input: pfor input (op type must be StackPushV2 or StackPopV2, asserted). Walks back from the
  # handle through Identity and Enter to the StackV2 creator (asserted). Returns a triple of the
  # default graph, the PFor object and the creating tensor.
  """Create cache key corresponding to a stack handle."""
  op_type = pfor_input.op_type
  assert op_type in ["StackPushV2", "StackPopV2"], op_type
  orig_handle = pfor_input.op.inputs[0]
  while orig_handle.op.type in ["Identity", "Enter"]:
    orig_handle = orig_handle.op.inputs[0]
  assert orig_handle.op.type == "StackV2", orig_handle.op
  return ops.get_default_graph(), pfor_input.pfor, orig_handle


def _stack_handle_inside_pfor(handle, pfor_input: _PforInput):
  # [FUNCTION DOC]
  # HELPER: true if the StackV2 behind a handle (found through Identity and Enter ops, asserted to be
  # a StackV2 creation) was created inside the pfor loop body.
  while handle.op.type in ["Identity", "Enter"]:
    handle = handle.op.inputs[0]
  assert handle.op.type == "StackV2", ("Unable to find StackV2 op. Got %s" %
                                       handle.op)
  return pfor_input.pfor.op_is_inside_loop(handle.op)


@RegisterPFor("StackPushV2")
def _convert_stack_push_v2(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for StackPushV2. Raises a value error if the stack was created outside
  # the pfor loop. Records in the global stack cache whether the stack elements are stacked (first
  # push decides). A later stacked push to a stack that was determined unstacked raises a value
  # error; an unstacked push to a stacked stack is stacked first. Returns the pushed value wrapped
  # with the cached stackedness. Side effect: mutates the module-level cache.
  handle = pfor_input.unstacked_input(0)
  elem, elem_stacked, _ = pfor_input.input(1)
  swap_memory = pfor_input.get_attr("swap_memory")

  if not _stack_handle_inside_pfor(pfor_input.op.inputs[0], pfor_input):
    raise ValueError("StackPushV2 not allowed on stacks created outside pfor.")
  stack_cache_key = _stack_cache_key(pfor_input)
  stacked = _stack_cache.get(stack_cache_key, None)
  if stacked is None:
    stacked = elem_stacked
    _stack_cache[stack_cache_key] = stacked
  else:
    # If we previously made it unstacked then we can't revert to being stacked.
    if not stacked and elem_stacked:
      raise ValueError(
          "It looks like the stack was previously determined to be loop "
          "invariant, but we are now trying to push a loop dependent value "
          "to it. This is currently unsupported.")
    if stacked and not elem_stacked:
      elem = _stack(elem, pfor_input.pfor.loop_len_vector).t
  out = data_flow_ops.stack_push_v2(handle, elem, swap_memory=swap_memory)
  return wrap(out, stacked)


# Note that inputs to this convertor will be unstacked. However it should get
# called since it is a stateful op.
@RegisterPFor("StackPopV2")
def _convert_stack_pop_v2(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for StackPopV2. Loop-invariant handle; stackedness taken from the
  # cache (default unstacked and recorded if no push was converted yet). Pops with the element type
  # attribute. Returns the popped value wrapped accordingly. Side effect: may write the global cache.
  handle = pfor_input.unstacked_input(0)
  stack_cache_key = _stack_cache_key(pfor_input)
  stacked = _stack_cache.get(stack_cache_key, None)
  # If a StackPushV2 has not been converted yet, we default to unstacked since
  # the push could be outside of pfor, or the convertor may not be called if the
  # inputs are unconverted.
  if stacked is None:
    stacked = False
    _stack_cache[stack_cache_key] = False
  elem_type = pfor_input.get_attr("elem_type")
  out = data_flow_ops.stack_pop_v2(handle, elem_type)
  return wrap(out, stacked)


# parsing_ops


@RegisterPFor("DecodeCSV")
def _convert_decode_csv(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for DecodeCSV. Stacked lines and loop-invariant record defaults; field
  # delimiter, quote and selected-column attributes preserved (empty selection becomes null). Returns
  # a list of stacked tensors.
  lines = pfor_input.stacked_input(0)
  record_defaults = [
      pfor_input.unstacked_input(i) for i in range(1, pfor_input.num_inputs)
  ]
  field_delim = pfor_input.get_attr("field_delim")
  use_quote_delim = pfor_input.get_attr("use_quote_delim")
  select_cols = pfor_input.get_attr("select_cols")
  if not select_cols:
    select_cols = None
  return [
      wrap(t, True) for t in gen_parsing_ops.decode_csv(
          lines,
          record_defaults,
          field_delim=field_delim,
          use_quote_delim=use_quote_delim,
          select_cols=select_cols)
  ]


@RegisterPFor("ParseSingleExample")
def _convert_parse_single_example(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for ParseSingleExample. Stacked serialized input; loop-invariant dense
  # defaults. Uses the batch parse-example op with the keys, types and shapes attributes. Returns the
  # flattened outputs, each marked stacked and sparse-stacked.
  serialized = pfor_input.stacked_input(0)
  dense_defaults = [
      pfor_input.unstacked_input(i) for i in range(1, pfor_input.num_inputs)
  ]
  sparse_keys = pfor_input.get_attr("sparse_keys")
  dense_keys = pfor_input.get_attr("dense_keys")
  sparse_types = pfor_input.get_attr("sparse_types")
  dense_shapes = pfor_input.get_attr("dense_shapes")
  output = gen_parsing_ops.parse_example(
      serialized=serialized,
      names=[],
      dense_defaults=dense_defaults,
      sparse_keys=sparse_keys,
      dense_keys=dense_keys,
      sparse_types=sparse_types,
      dense_shapes=dense_shapes)
  return [wrap(t, True, True) for t in nest.flatten(output)]


@RegisterPFor("ParseExampleV2")
def _convert_parse_example_v2(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for ParseExampleV2. Stacked serialized input (must be rank one or of
  # unknown rank, otherwise a value error); loop-invariant key lists and dense defaults; sparse
  # and ragged attributes preserved. Returns the flattened outputs, each marked stacked and
  # sparse-stacked.
  serialized = pfor_input.stacked_input(0)
  sparse_keys = pfor_input.unstacked_input(2)
  dense_keys = pfor_input.unstacked_input(3)
  ragged_keys = pfor_input.unstacked_input(4)
  dense_defaults = [
      pfor_input.unstacked_input(i) for i in range(5, pfor_input.num_inputs)
  ]
  num_sparse = pfor_input.get_attr("num_sparse")
  sparse_types = pfor_input.get_attr("sparse_types")
  ragged_value_types = pfor_input.get_attr("ragged_value_types")
  ragged_split_types = pfor_input.get_attr("ragged_split_types")
  dense_shapes = pfor_input.get_attr("dense_shapes")
  if serialized.shape.ndims not in (None, 1):
    raise ValueError("ParseExampleV2 can only be converted if `serialized` "
                     f"is scalar. Received shape: {serialized.shape}.")
  output = gen_parsing_ops.parse_example_v2(
      serialized=serialized,
      names=[],
      sparse_keys=sparse_keys,
      dense_keys=dense_keys,
      ragged_keys=ragged_keys,
      dense_defaults=dense_defaults,
      num_sparse=num_sparse,
      sparse_types=sparse_types,
      ragged_value_types=ragged_value_types,
      ragged_split_types=ragged_split_types,
      dense_shapes=dense_shapes)
  return [wrap(t, True, True) for t in nest.flatten(output)]


# functional_ops


def _convert_function_call(func, converter, inputs):
  # [FUNCTION DOC]
  # HELPER: converts a call to a function graph using a given PFor converter.
  # Inputs: func (a concrete function wrapping a function graph, asserted), converter (a PFor,
  # asserted) and inputs (list of wrapped tensors). Traces a helper function that maps the function's
  # graph inputs to the given wrapped inputs in the converter and converts all graph outputs; calls
  # it, and re-wraps each result with the stackedness decided during conversion. Returns a list of
  # wrapped outputs.
  assert isinstance(func.graph, func_graph.FuncGraph), func
  assert isinstance(converter, PFor)

  graph_outputs = func.graph.outputs[:len(func.function_type.flat_outputs)]
  # TODO(agarwal): consider caching this function definition.
  @def_function.function
  def f(*args):
    # [FUNCTION DOC]
    # Nested traced function used when converting a function call: asserts all arguments are wrapped
    # tensors matching the function inputs, registers each graph input's conversion, converts the graph
    # outputs and returns a tuple of their raw tensors.
    assert all(isinstance(arg, WrappedTensor) for arg in args), args
    assert len(args) == len(func.graph.inputs), (args, func.graph.inputs)
    #  Map inputs to function arguments.
    for inp, arg in zip(func.graph.inputs, args):
      converter._add_conversion(inp, arg)
    # Convert output tensors.
    return tuple([converter._convert_helper(x).t for x in graph_outputs])

  call_outputs = f(*inputs)
  assert len(call_outputs) == len(graph_outputs)
  outputs = []
  for call_output, output_tensor in zip(call_outputs, graph_outputs):
    func_output = converter._convert_helper(output_tensor)
    outputs.append(
        wrap(call_output, func_output.is_stacked, func_output.is_sparse_stacked)
    )
  return outputs


@RegisterPFor("StatefulPartitionedCall")
@RegisterPFor("PartitionedCall")
def _convert_partitioned_call(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for PartitionedCall and StatefulPartitionedCall. Looks up the called
  # function graph by name (assertion if it has no function graph), creates a child PFor over its ops
  # inheriting loop variable, length, indices and config, and converts the call. Returns the list of
  # wrapped outputs.
  func_name = pfor_input.get_attr("f").name
  func = pfor_input.op.graph._get_function(compat.as_bytes(func_name))
  assert isinstance(func.graph, func_graph.FuncGraph), (
      "Could not find FuncGraph object for %s. Got func %s" % (func_name, func))
  pfor = pfor_input.pfor
  converter = PFor(
      loop_var=pfor.loop_var,
      loop_len=pfor.loop_len_vector[0],
      pfor_ops=func.graph.get_operations(),
      fallback_to_while_loop=pfor.fallback_to_while_loop,
      all_indices=pfor.all_indices,
      all_indices_partitioned=pfor.all_indices_partitioned,
      pfor_config=pfor.pfor_config)
  return _convert_function_call(func, converter, pfor_input.inputs)


def _partition_inputs_for_indices(inputs, indices):
  # [FUNCTION DOC]
  # HELPER: selects, for a conditional branch, the rows of stacked inputs belonging to the given
  # iteration indices (unstacked inputs pass through). Returns a new list of wrapped inputs.
  new_inputs = []
  for inp in inputs:
    if inp.is_stacked:
      new_inputs.append(wrap(array_ops.gather(inp.t, indices), True))
    else:
      new_inputs.append(inp)
  return new_inputs


def _outputs_for_branch(func_name, indices, pfor_input: _PforInput, inputs):
  # [FUNCTION DOC]
  # HELPER: converts one branch function of a conditional.
  # Inputs: function name, indices of the iterations taking this branch (null means all current
  # ones), the pfor input and the branch inputs. Creates a child PFor over the branch graph with
  # those indices (marked partitioned when explicit), converts the call and stacks any unstacked
  # output to the branch's length. Returns a list of raw stacked tensors.
  if indices is None:
    indices = pfor_input.pfor.all_indices
    partitioned = pfor_input.pfor.all_indices_partitioned
  else:
    partitioned = True
  func = pfor_input.op.graph._get_function(func_name)
  converter = PFor(
      loop_var=pfor_input.pfor.loop_var,
      loop_len=array_ops.size(indices),
      pfor_ops=func.graph.get_operations(),
      fallback_to_while_loop=pfor_input.pfor.fallback_to_while_loop,
      all_indices=indices,
      all_indices_partitioned=partitioned,
      pfor_config=pfor_input.pfor.pfor_config)
  outputs = _convert_function_call(func, converter, inputs)
  stacked_outputs = []
  for out in outputs:
    if not out.is_stacked:
      stacked_outputs.append(_stack(out.t, [array_ops.size(indices)]).t)
    else:
      stacked_outputs.append(out.t)
  return stacked_outputs


# TODO(agarwal): Currently the converted code aggressively tiles loop variant
# outputs from the then/else branches. Instead, it could do so only if at least
# one of the branch outputs is loop variant.
@RegisterPFor("StatelessIf")
@RegisterPFor("If")
def _convert_if(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for If and StatelessIf. If the condition is stacked, partitions the
  # iteration indices (and inputs) into the true and false groups, converts each branch for its
  # group and merges the outputs with a dynamic stitch (note: ordering of side effects between
  # branches is undefined). If the condition is unstacked a normal conditional picks which branch's
  # conversion runs. All outputs are stacked (loop variant outputs are tiled eagerly, a noted TODO).
  # Returns a list of stacked tensors.
  cond, cond_stacked, _ = pfor_input.input(0)
  inputs = pfor_input.inputs[1:]
  then_branch = pfor_input.get_attr("then_branch")
  else_branch = pfor_input.get_attr("else_branch")

  if cond_stacked:
    cond_int = math_ops.cast(cond, dtypes.int32)
    # Compute loop indices for the different branches
    false_indices, true_indices = data_flow_ops.dynamic_partition(
        pfor_input.pfor.all_indices, cond_int, 2)
    # Compute indices for cond being True or False.
    if pfor_input.pfor.all_indices_partitioned:
      else_indices, then_indices = data_flow_ops.dynamic_partition(
          math_ops.range(pfor_input.pfor.loop_len_vector[0]),
          cond_int, 2)
    else:
      else_indices, then_indices = false_indices, true_indices
    # Partition inputs
    then_inputs = _partition_inputs_for_indices(inputs, then_indices)
    else_inputs = _partition_inputs_for_indices(inputs, else_indices)

    # Convert "then" branch.
    then_outputs = _outputs_for_branch(then_branch.name, true_indices,
                                       pfor_input, then_inputs)

    # Convert "else" branch.
    else_outputs = _outputs_for_branch(else_branch.name, false_indices,
                                       pfor_input, else_inputs)

    assert len(then_outputs) == len(else_outputs)
    # Note that if the "then" and "else" branches are updating the same state,
    # and possibly reading them as well, it could lead to undefined behavior
    # since the ordering of those operations is not well defined.
    # One possibility is to order all the "then" branches to execute before all
    # the "else" branches so that the side-effects in the former are visible to
    # the latter. For now, we leave that as undefined behavior.
    outputs = []
    # Merge outputs
    for then_output, else_output in zip(then_outputs, else_outputs):
      out = data_flow_ops.dynamic_stitch([then_indices, else_indices],
                                         [then_output, else_output])
      outputs.append(wrap(out, True))
    return outputs
  else:
    outputs = tf_cond.cond(
        cond,
        lambda: _outputs_for_branch(then_branch.name, None, pfor_input, inputs),
        lambda: _outputs_for_branch(else_branch.name, None, pfor_input, inputs))
    return [wrap(t, True) for t in outputs]


@RegisterPFor("Case")
@RegisterPFor("StatelessCase")
def _convert_stateless_case(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # KNOWN ISSUE: in the stacked-index branch all branch outputs are concatenated into a single list
  # and stitched into one result, and only one wrapped output is returned, so this is only correct for
  # cases with exactly one output.
  # Registered pfor converter for Case and StatelessCase. With a stacked branch index it partitions iteration indices and inputs per branch, converts each
  # branch for its group and stitches. With an unstacked index it builds one callable per branch and
  # uses a switch-case. Returns a list of stacked outputs.
  branch_idx, is_stacked, _ = pfor_input.input(0)
  branches = pfor_input.get_attr("branches")
  inputs = pfor_input.inputs[1:]

  if is_stacked:
    logging.info("Running stacked flow")

    # Compute loop indices for the different branches
    switch_indices = data_flow_ops.dynamic_partition(
        pfor_input.pfor.all_indices, branch_idx, len(branches))
    if pfor_input.pfor.all_indices_partitioned:
      partitioned_indices = data_flow_ops.dynamic_partition(
          math_ops.range(pfor_input.pfor.loop_len_vector[0]), branch_idx,
          len(branches))
    else:
      partitioned_indices = switch_indices
    # Partition inputs
    input_list = []
    for indices in partitioned_indices:
      input_list.append(_partition_inputs_for_indices(inputs, indices))

    outputs = []
    for (b, indices, inputs) in zip(branches, switch_indices, input_list):
      out = _outputs_for_branch(b.name, indices, pfor_input, inputs)
      outputs.extend(out)

    out = data_flow_ops.dynamic_stitch(partitioned_indices, outputs)
    return [wrap(out, True)]
  else:
    new_branches = []
    for b in branches:
      def new_function(func=b.name):
        # [FUNCTION DOC]
        # Nested per-branch callable for the unstacked case: takes the branch function name (bound at
        # creation) and returns the converted outputs for all current iterations using the inputs after the
        # branch index.
        return _outputs_for_branch(func, None, pfor_input,
                                   pfor_input.inputs[1:])

      new_branches.append(new_function)

    outputs = []
    outputs = control_flow_switch_case.switch_case(branch_idx, new_branches)
    return [wrap(t, True) for t in outputs]


class WhileV2:
  # [CLASS DOC]
  # Converter for functional while ops (While and StatelessWhile) that carry cond and body as
  # function graphs. Constructed from the pfor input, it finds the two function graphs (value error if
  # missing), determines which body inputs pass through unchanged, and reads the parallel-iterations
  # attribute. Calling it builds the vectorized while loop, iterating until the stackedness of body
  # inputs and outputs agree (restarting after stacking inputs).
  """Object for vectorizing V2 while_loop op."""

  def __init__(self, pfor_input: _PforInput):
    # [FUNCTION DOC]
    # Prepares the converter. Input: the pfor input of the while op. Looks up the cond and body function
    # graphs by name (raises a value error if either is not found), records which body input positions
    # are returned unchanged (pass-through), and stores the parallel-iterations attribute.
    self._pfor_input = pfor_input
    self._pfor = pfor_input.pfor
    cond_func_name = pfor_input.get_attr("cond").name
    self._cond_func = pfor_input.op.graph._get_function(compat.as_bytes(
        cond_func_name))
    body_func_name = pfor_input.get_attr("body").name
    self._body_func = pfor_input.op.graph._get_function(compat.as_bytes(
        body_func_name))
    if self._cond_func is None or self._body_func is None:
      raise ValueError("Error extracting cond and body functions for op "
                       f"{self._pfor_input.op}.")
    # Indices of inputs that are passed unchanged through the while loop body.
    # Typically these are tensors captured from outside the body context.
    self._body_pass_through_indices = set()
    for i, (inp, out) in enumerate(zip(self._body_func.graph.inputs,
                                       self._body_func.graph.outputs)):
      if id(inp) == id(out):
        self._body_pass_through_indices.add(i)
    self._parallel_iterations = self._pfor_input.get_attr("parallel_iterations")

  def _output_shapes(self):
    # [FUNCTION DOC]
    # HELPER: computes the shape invariants for the vectorized loop variables.
    # Merges inferred output shapes with the output_shapes attribute (unknown if absent) and prepends an
    # unknown leading dimension for stacked inputs (an empty prefix for TensorList/optional variants,
    # which stay scalar). Asserts that the number of outputs equals the number of inputs.
    # Returns a list of shapes.
    # Calculate output shape for vectorized loop. This will be used as
    # shape_invariant. Merges shape inference outputs with the `output_shapes`
    # attribute of the op.
    output_shapes = [out.shape for out in self._pfor_input.op.outputs]
    shapes = self._pfor_input.get_attr("output_shapes")
    if not shapes:
      shapes = [tensor_shape.TensorShape(None) for _ in output_shapes]
    else:
      shapes = [tensor_shape.TensorShape(shape) for shape in shapes]
    for i, shape in enumerate(shapes):
      shape = shape.merge_with(output_shapes[i])
      pfor_input = self._pfor_input.input(i)
      if pfor_input.is_stacked:
        if _is_variant_with_internal_stacking(pfor_input.t):
          shape = tensor_shape.TensorShape([]).concatenate(shape)
        else:
          shape = tensor_shape.TensorShape([None]).concatenate(shape)
      output_shapes[i] = shape
    assert len(output_shapes) == self._pfor_input.num_inputs
    return output_shapes

  def _init_values(self):
    # [FUNCTION DOC]
    # HELPER: builds the initial loop arguments (a true flag, the active indices, the inputs and one
    # output TensorArray per input).
    # For TensorList inputs it builds a nested accumulator: an array whose entries are lists of loop
    # length, because finished iterations are scattered per list element; optionals or other
    # variants raise a not-implemented error. Other inputs get a TensorArray of loop-length elements.
    # Returns the list of initial values.
    """Create arguments passed to converted while_loop."""
    loop_len = self._pfor.loop_len_vector[0]
    inputs = []
    # TensorArrays for outputs of converted while loop
    output_tas = []

    with ops.name_scope("while_init"):
      for inp in self._pfor_input.inputs:
        inputs.append(inp.t)
        variant_type_id = _variant_type_id(inp.t)
        if variant_type_id in _INTERNAL_STACKING_TYPE_IDS:
          if variant_type_id != full_type_pb2.TFT_ARRAY:
            raise NotImplementedError(
                "While loop conversion is only supported for TensorLists. Got "
                f"another variant {inp.t}, probably an optional. Please file "
                "a bug.")

          # For TensorLists, the input format is:
          #
          #   List[user_list_len, Tensor[loop_len, ...]]
          #
          # rather than the usual
          #
          #   Tensor[loop_len, ...]
          #
          # The body of the loop will take and return lists in this "internal
          # vectorization" format, so we want to keep it that way as much as
          # possible. We'll accumulate finished iterations (only relevant for
          # pfor-loop-variant while_loop conditions) in an accumulator with
          # type :
          #
          #   List[user_list_len, List[loop_len, Tensor[...]]]
          #
          # This means that each while_loop iteration, we'll iterate over the
          # length of the TensorList, dividing done/remaining pfor loop indices
          # and scattering the done indices into the inner nested list of the
          # accumulator.
          element_shape = list_ops.tensor_list_element_shape(
              inp.t, dtypes.int32)
          if inp.is_stacked:
            # Shapes may be tf.constant(-1) for fully dynamic, in which case
            # slicing is an error.
            element_shape = tf_cond.cond(
                math_ops.equal(array_ops.rank(element_shape), 0),
                lambda: element_shape,
                lambda: element_shape[1:])
          dtype = _parse_variant_shapes_and_types(inp.t)[0].dtype

          def _init_loop_body(index, output_ta):
            # [FUNCTION DOC]
            # Nested loop body used while building a nested TensorList accumulator: writes a freshly reserved
            # list of loop length at the current index of the array and returns the next index and the array.
            output_ta = output_ta.write(
                index,
                list_ops.tensor_list_reserve(element_shape, loop_len, dtype))
            return index + 1, output_ta

          length = list_ops.tensor_list_length(inp.t)
          output_ta = tensor_array_ops.TensorArray(
            inp.t.dtype,  # Variant; this is a nested TensorList
            size=length,
            dynamic_size=True,
            infer_shape=False)
          _, output_ta = while_loop.while_loop(lambda index, _: index < length,
                                               _init_loop_body, [0, output_ta])
        else:
          output_ta = tensor_array_ops.TensorArray(
            inp.t.dtype,
            size=loop_len,
            dynamic_size=False,
            infer_shape=True)
        output_tas.append(output_ta)
    # See documentation for __call__ for the structure of init_values.
    indices = (
        math_ops.range(self._pfor.loop_len_vector[0])
        if self._pfor.all_indices_partitioned else self._pfor.all_indices)
    return [True, indices] + inputs + output_tas

  def _process_cond_unstacked(self, conditions, indices, inputs, output_tas):
    # [FUNCTION DOC]
    # HELPER: step handler when the condition is the same for every iteration: reduces the condition to
    # a scalar flag and returns it with the unchanged indices, inputs and arrays.
    """Handles case when condition is pfor loop invariant."""
    # Note that all iterations end together. So we don't need to partition the
    # inputs.
    not_all_done = array_ops.reshape(conditions, [])
    return not_all_done, indices, inputs, output_tas

  def _process_cond_stacked(self, conditions, indices, inputs, inputs_stacked,
                            output_tas):
    # [FUNCTION DOC]
    # HELPER: step handler when the condition varies per iteration. Computes the any-continues flag,
    # partitions the indices, and for each input: pass-through inputs are kept; TensorList inputs are
    # partitioned element by element with a loop that scatters finished entries into the nested
    # accumulator; other inputs are partitioned (unstacked ones stacked for the finished part) and the
    # finished values scattered into the output array. Returns the flag, the new indices, the new inputs
    # and the new arrays.
    """Handles case when condition is pfor loop dependent."""
    # Compute if all iterations are done.
    not_all_done = math_ops.reduce_any(conditions)
    conditions_int = math_ops.cast(conditions, dtypes.int32)
    # Partition the indices.
    done_indices, new_indices = data_flow_ops.dynamic_partition(
        indices, conditions_int, 2)

    new_inputs = []
    new_output_tas = []
    for i, (inp, stacked) in enumerate(zip(inputs, inputs_stacked)):
      pass_through = i in self._body_pass_through_indices
      if not pass_through and  _variant_type_id(inp) == full_type_pb2.TFT_ARRAY:
        shape_and_type = _parse_variant_shapes_and_types(inp)[0]
        element_shape = list_ops.tensor_list_element_shape(inp, dtypes.int32)
        user_list_len = list_ops.tensor_list_length(inp)

        def _split_vectorized_ta_element(index, new_inp, new_out_ta):
          # [FUNCTION DOC]
          # Nested loop body used when partitioning a vectorized TensorList: for list position index, reads
          # the element, partitions it by the condition (or stacks it when unstacked), stores the continuing
          # part, scatters the finished part into the accumulator entry, and returns the next index, new list
          # and updated array.
          elem = list_ops.tensor_list_get_item(inp, index, shape_and_type.dtype,
                                               element_shape)
          if stacked:
            done_elem, new_elem = data_flow_ops.dynamic_partition(
                elem, conditions_int, 2)
            new_inp = list_ops.tensor_list_set_item(new_inp, index, new_elem)
          else:
            done_elem = _stack(elem, [array_ops.size(done_indices)]).t
          done_accum = new_out_ta.read(index)
          done_accum = list_ops.tensor_list_scatter(
              tensor=done_elem, indices=done_indices, input_handle=done_accum)
          new_out_ta = new_out_ta.write(index, done_accum)
          return index + 1, new_inp, new_out_ta

        length = list_ops.tensor_list_length(inp)
        new_inp = list_ops.tensor_list_reserve(
            tensor_shape.TensorShape([None])
            + tensor_shape.TensorShape(shape_and_type.shape)[1:],
            user_list_len, shape_and_type.dtype)
        _, new_inp, out_ta = while_loop.while_loop(
            lambda index, unused_new_inp, unused_new_out_ta: index < length,
            _split_vectorized_ta_element, [0, new_inp, output_tas[i]])
      else:
        # Partition the inputs.
        if stacked:
          done_inp, new_inp = data_flow_ops.dynamic_partition(
              inp, conditions_int, 2)
        else:
          if not pass_through:
            done_inp = _stack(inp, [array_ops.size(done_indices)]).t
          new_inp = inp

        out_ta = output_tas[i]
        if not pass_through:
          # Note that done_indices can be empty. done_inp should also be empty
          # in that case.
          out_ta = out_ta.scatter(done_indices, done_inp)
      new_inputs.append(new_inp)
      new_output_tas.append(out_ta)

    assert len(new_output_tas) == len(output_tas)
    assert len(new_inputs) == len(inputs)
    return not_all_done, new_indices, new_inputs, new_output_tas

  def _process_body(self, inputs_stacked, new_indices, cond_stacked,
                    new_inputs, not_all_done):
    # [FUNCTION DOC]
    # HELPER: converts the loop body for active iterations. Inputs: inputs_stacked, new_indices,
    # cond_stacked, new_inputs, not_all_done. Runs a fixed-point iteration (inside a nested function)
    # that converts the body, and whenever an output is stacked but its input is not, stacks that input
    # and retries. A conditional skips the work when all iterations are done. Returns the new
    # loop-variable values and the list of positions whose inputs had to be stacked (in that case the
    # generated code is to be abandoned).
    """Convert the body function."""
    # This is used to store the indices of inputs to the while op that need to
    # be stacked. This stacking may be needed in cases where the input to the
    # while_loop is loop_invariant but the corresponding output is not.
    mismatching_stacked_indices = []

    def true_fn():
      # [FUNCTION DOC]
      # Nested function that performs the fixed-point conversion of the body. Repeats: build a child
      # PFor for the active iterations, convert the body call, compare stackedness of outputs against
      # inputs, and stack and record mismatching inputs. When stable, returns the raw output tensors; if
      # inputs had to be stacked it instead returns identities of the old inputs guarded by an assertion
      # that fails if ever executed, since that graph is discarded.
      """Converts the body function for all but last iteration."""
      wrapped_inputs = [wrap(inp, stacked) for inp, stacked in
                        zip(new_inputs, inputs_stacked)]
      # Note the iterative process below to figure out loop invariance.
      # Here we iterate on vectorization process till a fixed point. The issue
      # is that the while body can take pfor loop invariant inputs but return
      # loop variant outputs. For any loop variant output, the corresponding
      # input has to be then made loop variant (since subsequent while
      # iterations will need to see loop variant values).
      # However once we make a new input loop variant, we might make other
      # outputs loop variant. Hence we need to iterate till we get fixed point.
      while True:
        if self._pfor.all_indices_partitioned:
          indices = array_ops.gather(self._pfor.all_indices, new_indices)
        else:
          indices = new_indices
        body_pfor = PFor(
            loop_var=self._pfor.loop_var,
            loop_len=array_ops.size(new_indices),
            pfor_ops=self._body_func.graph.get_operations(),
            fallback_to_while_loop=self._pfor.fallback_to_while_loop,
            all_indices=indices,
            all_indices_partitioned=(self._pfor.all_indices_partitioned or
                                     cond_stacked),
            pfor_config=self._pfor.pfor_config)
        stacking_mismatch = False
        outputs = _convert_function_call(self._body_func,
                                         body_pfor,
                                         wrapped_inputs)
        for i, (out, inp) in enumerate(zip(outputs, wrapped_inputs)):
          if out.is_stacked != inp.is_stacked:
            stacking_mismatch = True
            mismatching_stacked_indices.append(i)
            stacked = _stack(inp.t, [array_ops.size(new_indices)])
            if inp.t.dtype == dtypes.variant:
              stacked = wrap(
                  _tile_variant_with_length(stacked.t,
                                            [array_ops.size(new_indices)]))
            wrapped_inputs[i] = stacked
        if not stacking_mismatch:
          if mismatching_stacked_indices:
            # We needed to stack some inputs. This code will be abandoned and
            # should not get executed. Hence we simply return `new_inputs` to
            # make sure the graph construction code completes.
            with ops.control_dependencies([
                control_flow_assert.Assert(
                    False, ["pfor ERROR: this branch should never execute"])
            ]):
              return [array_ops.identity(x) for x in new_inputs]
          else:
            return [out.t for out in outputs]

    # If all are done, we simply return `new_inputs`. Else we need to run the
    # body function.
    return tf_cond.cond(
        not_all_done,
        true_fn,
        lambda: list(new_inputs)), mismatching_stacked_indices

  def __call__(self):
    # [FUNCTION DOC]
    # Builds the vectorized while loop for the functional While op (converter entry point).
    # Defines condition and body functions, wraps loop construction in a traced function used only to
    # obtain a concrete graph, and checks whether any input needed stacking. If so, stacks those
    # inputs (tiling variants), discards the work and calls itself again (at most one recursion).
    # Otherwise runs the traced function and wraps outputs: stacked if the condition is stacked and the
    # position is not pass-through, else with the input's stackedness. Arguments to the loop: a
    # not-all-done flag, active indices, the inputs and one TensorArray per output. Returns a list of
    # wrapped outputs.
    """Converter for the V2 while_loop.

    The conversion of a while_loop is another while_loop.

    The arguments to this converted while_loop are as follows:
    not_all_done: Boolean scalar Tensor indicating if all the pfor iterations
      are done.
    indices: int32 1-D Tensor storing the id of the pfor iterations that are not
      done.
    args: Remaining arguments. These can be divided into 2 categories:
      - The first set of arguments correspond one-to-one to the inputs to the
        unvectorized while_loop.
      - The second set are TensorArrays, corresponding one-to-one to each output
        of the unvectorized while_loop. Each TensorArray has `PFor.loop_len`
        elements, i.e. the number of pfor iterations. At the end, the i'th
        element of each TensorArray will contain the output computed by the i'th
        iteration of pfor. Note that elements can be written into these tensors
        arrays in any order, depending on when the corresponding pfor iteration
        is done.
    In each iteration, the while_loop body recomputes the condition for all
    active pfor iterations to see which of them are now done. It then partitions
    all the inputs and passes them along to the converted body. Values for all
    the iterations that are done are written to TensorArrays indexed by the pfor
    iteration number. When all iterations are done, the TensorArrays are stacked
    to get the final value.

    Returns:
      List of converted outputs.
    """
    output_shapes = self._output_shapes()
    # Note that we use these lists as a hack since we need the `body` to compute
    # these values during construction of the while_loop graph.
    cond_is_stacked = [None]
    indices_to_stack = []

    def cond(not_all_done, *_):
      # [FUNCTION DOC]
      # Nested loop condition: returns the first argument, the not-all-done flag; the rest are ignored.
      return not_all_done

    def body(not_all_done, indices, *args):
      # [FUNCTION DOC]
      # Nested loop body. Inputs: flag, active indices and the remaining arguments (inputs then arrays).
      # Converts the loop condition in a child PFor (assuming partitioned indices), records its
      # stackedness, handles finished iterations via the stacked or unstacked handler, converts the body,
      # records which inputs must be stacked, sets static shapes on new values from the shape invariants
      # and returns the new loop arguments as a tuple.
      # See documentation for __call__ for the structure of *args.
      num_inputs = self._pfor_input.num_inputs
      inputs = args[:num_inputs]
      output_tas = args[num_inputs:]
      inputs_stacked = [x.is_stacked for x in self._pfor_input.inputs]
      assert len(inputs) >= len(output_tas)
      assert len(inputs) == len(inputs_stacked)
      # Convert condition
      with ops.name_scope("while_cond"):
        # Note that we set all_indices_partitioned to True here. At this point
        # we don't know if indices will be partitioned. Hence we use the
        # conservative value.
        cond_pfor = PFor(
            loop_var=self._pfor.loop_var,
            loop_len=array_ops.size(indices),
            pfor_ops=self._cond_func.graph.get_operations(),
            fallback_to_while_loop=self._pfor.fallback_to_while_loop,
            all_indices=indices,
            all_indices_partitioned=True,
            pfor_config=self._pfor.pfor_config)

        wrapped_inputs = [wrap(inp, stacked) for inp, stacked
                          in zip(inputs, inputs_stacked)]
        conditions, cond_stacked, _ = _convert_function_call(
            self._cond_func,
            cond_pfor,
            wrapped_inputs)[0]
        cond_is_stacked[0] = cond_stacked

      # Recompute the new condition, write outputs of done iterations, and
      # partition the inputs if needed.
      if not cond_stacked:
        (not_all_done, new_indices, new_inputs,
         new_output_tas) = self._process_cond_unstacked(conditions, indices,
                                                        inputs, output_tas)
      else:
        (not_all_done, new_indices, new_inputs,
         new_output_tas) = self._process_cond_stacked(conditions, indices,
                                                      inputs, inputs_stacked,
                                                      output_tas)
      # Convert body
      with ops.name_scope("while_body"):
        #  Compute the outputs from the body.
        new_outputs, mismatching_stacked_indices = self._process_body(
            inputs_stacked, new_indices, cond_stacked, new_inputs, not_all_done)

      indices_to_stack[:] = mismatching_stacked_indices
      for i, new_output in enumerate(new_outputs):
        new_output.set_shape(output_shapes[i])
      new_args = ([not_all_done, new_indices] + new_outputs +
                  list(new_output_tas))
      return tuple(new_args)

    # Note that we run the code below in a function since we might abandon the
    # generated code in cases where the conversion dictates that some inputs be
    # further stacked. Hence we run the graph construction using
    # `get_concrete_function` and avoid calling the constructed function if not
    # needed.
    @def_function.function
    def while_fn():
      # [FUNCTION DOC]
      # Nested traced function that actually builds the loop. Creates initial values and shape
      # invariants, runs the while loop with the original parallel-iterations setting, and returns the raw
      # loop outputs if some input must be restacked (that function is abandoned). Otherwise it assembles
      # final outputs: for stacked conditions, pass-through values come from the initial values and
      # other outputs come from the stacked arrays (TensorList outputs are rebuilt by stacking each
      # nested list); unstacked conditions return the final loop values.
      # Create init_values that will be passed to the while_loop.
      init_values = self._init_values()
      ta_shape_invariants = [tensor_shape.TensorShape([]) for _ in
                             self._pfor_input.outputs]
      shape_invariants = (
          [tensor_shape.TensorShape([]), tensor_shape.TensorShape([None])]
          + output_shapes + ta_shape_invariants)

      while_outputs = while_loop.while_loop(
          cond,
          body,
          init_values,
          shape_invariants=shape_invariants,
          parallel_iterations=self._parallel_iterations)
      if indices_to_stack:
        # This function will be abandoned.
        return while_outputs
      else:
        num_inputs = self._pfor_input.num_inputs
        new_inputs = while_outputs[2:num_inputs+2]
        output_tas = while_outputs[num_inputs+2:]
        assert cond_is_stacked[0] is not None
        outputs = []
        for i, inp in enumerate(new_inputs):
          if cond_is_stacked[0]:
            if i in self._body_pass_through_indices:
              outputs.append(init_values[i + 2])
            else:
              ta = output_tas[i]
              if _variant_type_id(inp) == full_type_pb2.TFT_ARRAY:
                shape_and_type = _parse_variant_shapes_and_types(inp)[0]
                length = list_ops.tensor_list_length(inp)

                # We have been accumulating values in a:
                #
                #   List[user_list_len, List[loop_len, Tensor[...]]]
                #
                # We want to return an output in the same format as the input:
                #
                #   List[user_list_len, Tensor[loop_len, ...]]
                #
                # So we need to loop over the list and stack its contents.
                def _stack_loop_body(index, output_list):
                  # [FUNCTION DOC]
                  # Nested loop body used to rebuild the output TensorList: stacks the nested list at index i from
                  # the accumulator array and sets it in the output list; returns the next index and list.
                  current_value = ta.read(index)
                  output_list = list_ops.tensor_list_set_item(
                      output_list, index,
                      list_ops.tensor_list_stack(
                          current_value, shape_and_type.dtype))
                  return index + 1, output_list

                output_list = list_ops.tensor_list_reserve(
                    tensor_shape.TensorShape(shape_and_type.shape), length,
                    shape_and_type.dtype)
                _, output_list = while_loop.while_loop(
                    lambda index, _: index < length, _stack_loop_body,
                    [0, output_list])
                outputs.append(output_list)
              else:
                outputs.append(ta.stack())
          else:
            outputs.append(inp)
        return outputs

    _ = while_fn.get_concrete_function()
    if indices_to_stack:
      # Need to abandon the current conversion, stack some inputs and restart.
      self._pfor_input.stack_inputs(
          stack_indices=indices_to_stack, tile_variants=True)
      # Note that this call will recurse at most one time. The first call will
      # do the required stacking, based on the iterative procedure in
      # _process_body, and the next invocation to __call__ should not need to do
      # any more stacking.
      # We invoke `self()` here as a way to discard any corrupted state.
      return self()
    else:
      outputs = while_fn()
      wrapped_outputs = []
      for i, (out, inp) in enumerate(zip(outputs, self._pfor_input.inputs)):
        if i not in self._body_pass_through_indices and cond_is_stacked[0]:
          wrapped_outputs.append(wrap(out, True))
        else:
          wrapped_outputs.append(wrap(out, inp.is_stacked))
      return wrapped_outputs


@RegisterPFor("StatelessWhile")
@RegisterPFor("While")
def _convert_while(pfor_input: _PforInput):
  # [FUNCTION DOC]
  # Registered pfor converter for While and StatelessWhile: creates the functional while converter for
  # the pfor input and calls it. Returns its list of wrapped outputs.
  converter = WhileV2(pfor_input)
  return converter()


# spectral_ops


@RegisterPForWithArgs("FFT", gen_spectral_ops.fft)
@RegisterPForWithArgs("FFT2D", gen_spectral_ops.fft2d)
@RegisterPForWithArgs("FFT3D", gen_spectral_ops.fft3d)
@RegisterPForWithArgs("IFFT", gen_spectral_ops.ifft)
@RegisterPForWithArgs("IFFT2D", gen_spectral_ops.ifft2d)
@RegisterPForWithArgs("IFFT3D", gen_spectral_ops.ifft3d)
def _convert_fft(pfor_input: _PforInput, _, op_func):
  # [FUNCTION DOC]
  # Registered pfor converter for FFT, FFT2D, FFT3D and their inverses (function supplied by
  # registration). These ops act on the innermost dimensions, so the function is applied directly to
  # the stacked input. Returns one stacked tensor.
  return wrap(op_func(pfor_input.stacked_input(0)), True)


@RegisterPForWithArgs("RFFT", gen_spectral_ops.rfft, "Tcomplex")
@RegisterPForWithArgs("RFFT2D", gen_spectral_ops.rfft2d, "Tcomplex")
@RegisterPForWithArgs("RFFT3D", gen_spectral_ops.rfft3d, "Tcomplex")
@RegisterPForWithArgs("IRFFT", gen_spectral_ops.irfft, "Treal")
@RegisterPForWithArgs("IRFFT2D", gen_spectral_ops.irfft2d, "Treal")
@RegisterPForWithArgs("IRFFT3D", gen_spectral_ops.irfft3d, "Treal")
def _convert_rfft(pfor_input: _PforInput, _, op_func, attr_name):
  # [FUNCTION DOC]
  # Registered pfor converter for real-input FFTs and inverse real FFTs in 1-3 dimensions (function
  # and the name of the type attribute supplied by registration). Stacked input and loop-invariant
  # FFT length; reads the named type attribute and calls the function. Returns one stacked tensor.
  inp = pfor_input.stacked_input(0)
  fft_length = pfor_input.unstacked_input(1)
  attr = pfor_input.get_attr(attr_name)
  return wrap(op_func(inp, fft_length, attr), True)
