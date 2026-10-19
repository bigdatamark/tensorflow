# [MODULE DOC]
# Runtime helpers called by generated op wrapper code to run an op eagerly. Provides the
# execution entry points (plain, cancellable, and with op callbacks), placeholder gradient
# recording hooks that a later import of the backprop module replaces, argument converters that
# turn Python values into the attribute forms the native layer expects (float, int, string, bool,
# type, shape, tensor proto), and converters that turn lists of values into eager tensors with
# consistent dtypes. Defines the module-level alias for the default execution function. Other
# modules may monkey-patch the aliases. No import-time side effects beyond imports.

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
"""Functions called by the generated code to execute an eager-mode op."""

from google.protobuf import text_format
from tensorflow.core.framework import tensor_pb2
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import core
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_conversion_registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.types import core as core_types
from tensorflow.python.util import compat


def quick_execute(op_name, num_outputs, inputs, attrs, ctx, name=None):
  # [FUNCTION DOC]
  # Runs one registered op eagerly through the native layer and returns its outputs.
  # Inputs: op_name (string, registered op name); num_outputs (integer count of outputs to
  # fetch); inputs (list of tensors or tensor-convertible values); attrs (flat tuple alternating
  # attribute names and values); ctx (the eager context, used for its device name, lazy
  # initialization and native handle); name (optional string, appended to error messages).
  # Returns the list of output tensors (empty when there are none).
  # Raises: the mapped framework exception when the native layer reports a non-OK status (with
  # the name appended to the message if given, original chain suppressed); a symbolic-tensor
  # error if a type error occurred and any input is a Keras symbolic tensor; otherwise the
  # original type error is re-raised.
  # Side effects: ensures the context is initialized; executes the op on the context device.
  """Execute a TensorFlow operation.

  Args:
    op_name: Name of the TensorFlow operation (see REGISTER_OP in C++ code) to
      execute.
    num_outputs: The number of outputs of the operation to fetch. (Explicitly
      provided instead of being inferred for performance reasons).
    inputs: A list of inputs to the operation. Each entry should be a Tensor, or
      a value which can be passed to the Tensor constructor to create one.
    attrs: A tuple with alternating string attr names and attr values for this
      operation.
    ctx: The value of context.context().
    name: Customized name for the operation.

  Returns:
    List of output Tensor objects. The list is empty if there are no outputs

  Raises:
    An exception on error.
  """
  device_name = ctx.device_name
  # pylint: disable=protected-access
  try:
    ctx.ensure_initialized()
    tensors = pywrap_tfe.TFE_Py_Execute(ctx._handle, device_name, op_name,
                                        inputs, attrs, num_outputs)
  except core._NotOkStatusException as e:
    if name is not None:
      e.message += " name: " + name
    raise core._status_to_exception(e) from None
  except TypeError as e:
    keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
    if keras_symbolic_tensors:
      raise core._SymbolicException(
          "Inputs to eager execution function cannot be Keras symbolic "
          "tensors, but found {}".format(keras_symbolic_tensors))
    raise e
  # pylint: enable=protected-access
  return tensors


def execute_with_cancellation(op_name,
                              num_outputs,
                              inputs,
                              attrs,
                              ctx,
                              cancellation_manager,
                              name=None):
  # [FUNCTION DOC]
  # Same as the plain eager execution function but the op can be cancelled. Inputs: op_name,
  # num_outputs, inputs, attrs, ctx and name as for the plain version, plus cancellation_manager
  # (a cancellation manager object whose native implementation is passed to the native layer so
  # the op can be cancelled). Returns the list of output tensors (empty when none).
  # Raises: mapped framework exception on a non-OK native status (name appended to message if
  # given); a symbolic-tensor error for Keras symbolic inputs after a type error; otherwise the
  # original type error. Side effects: ensures the context is initialized and runs the op.
  """Execute a TensorFlow operation.

  Args:
    op_name: Name of the TensorFlow operation (see REGISTER_OP in C++ code) to
      execute.
    num_outputs: The number of outputs of the operation to fetch. (Explicitly
      provided instead of being inferred for performance reasons).
    inputs: A list of inputs to the operation. Each entry should be a Tensor, or
      a value which can be passed to the Tensor constructor to create one.
    attrs: A tuple with alternating string attr names and attr values for this
      operation.
    ctx: The value of context.context().
    cancellation_manager: a `CancellationManager` object that can be used to
      cancel the operation.
    name: Customized name for the operation.

  Returns:
    List of output Tensor objects. The list is empty if there are no outputs

  Raises:
    An exception on error.
  """
  device_name = ctx.device_name
  # pylint: disable=protected-access
  try:
    ctx.ensure_initialized()
    tensors = pywrap_tfe.TFE_Py_ExecuteCancelable(ctx._handle, device_name,
                                                  op_name, inputs, attrs,
                                                  cancellation_manager._impl,
                                                  num_outputs)
  except core._NotOkStatusException as e:
    if name is not None:
      e.message += " name: " + name
    raise core._status_to_exception(e) from None
  except TypeError as e:
    keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
    if keras_symbolic_tensors:
      raise core._SymbolicException(
          "Inputs to eager execution function cannot be Keras symbolic "
          "tensors, but found {}".format(keras_symbolic_tensors))
    raise e
  # pylint: enable=protected-access
  return tensors


def execute_with_callbacks(op_name, num_outputs, inputs, attrs, ctx, name=None):
  # [FUNCTION DOC]
  # Wrapper around the plain eager execution function that also notifies op callbacks. Runs the
  # op with the same arguments (op_name, num_outputs, inputs, attrs, ctx, optional name), then
  # calls every callback registered on the context with the op name, inputs as a tuple, attrs,
  # output tensors and name. Returns the output tensors. Errors from execution or callbacks
  # propagate. Intended to replace the default execution alias when callbacks are enabled.
  """Monkey-patch to execute to enable execution callbacks."""
  tensors = quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
  for callback in ctx.op_callbacks:
    callback(op_name, tuple(inputs), attrs, tensors, name)

  return tensors


execute = quick_execute


def must_record_gradient():
  # [FUNCTION DOC]
  # HELPER: placeholder hook for generated op wrappers, replaced once the backprop module loads.
  # Default implementation always says gradients need not be recorded. No inputs; returns false.
  """Import backprop if you want gradients recorded."""
  return False


def record_gradient(unused_op_name, unused_inputs, unused_attrs,
                    unused_outputs):
  # [FUNCTION DOC]
  # HELPER: placeholder hook for generated op wrappers, replaced once the backprop module loads.
  # Default implementation ignores its four arguments (op name, inputs, attributes, outputs) and
  # does nothing. Returns nothing.
  """Import backprop if you want gradients recorded."""
  pass


def make_float(v, arg_name):
  # [FUNCTION DOC]
  # HELPER: converts an op attribute value for generated op wrappers.
  # Inputs: v (a real number) and arg_name (string used in the error message).
  # Returns v as a floating point number. Raises a type error if v is not a real number type.
  if not isinstance(v, compat.real_types):
    raise TypeError("Expected float for argument '%s' not %s." %
                    (arg_name, repr(v)))
  return float(v)


def make_int(v, arg_name):
  # [FUNCTION DOC]
  # HELPER: converts an op attribute value for generated op wrappers.
  # Inputs: v (value convertible to an integer) and arg_name (string for the error message).
  # Returns v as an integer. Strings are rejected outright. Raises a type error if v is a string
  # or cannot be converted to an integer.
  if isinstance(v, str):
    raise TypeError("Expected int for argument '%s' not %s." %
                    (arg_name, repr(v)))
  try:
    return int(v)
  except (ValueError, TypeError):
    raise TypeError("Expected int for argument '%s' not %s." %
                    (arg_name, repr(v)))


def make_str(v, arg_name):
  # [FUNCTION DOC]
  # HELPER: converts an op attribute value for generated op wrappers.
  # Inputs: v (bytes or text string) and arg_name (string for the error message).
  # Returns v as bytes (text is encoded). Raises a type error if v is neither bytes nor text.
  if not isinstance(v, compat.bytes_or_text_types):
    raise TypeError("Expected string for argument '%s' not %s." %
                    (arg_name, repr(v)))
  return compat.as_bytes(v)  # Convert unicode strings to bytes.


def make_bool(v, arg_name):
  # [FUNCTION DOC]
  # HELPER: validates an op attribute value for generated op wrappers.
  # Inputs: v (expected boolean) and arg_name (string for the error message).
  # Returns v unchanged. Raises a type error if v is not a boolean.
  if not isinstance(v, bool):
    raise TypeError("Expected bool for argument '%s' not %s." %
                    (arg_name, repr(v)))
  return v


def make_type(v, arg_name):
  # [FUNCTION DOC]
  # HELPER: converts an op attribute value for generated op wrappers.
  # Inputs: v (anything convertible to a data type) and arg_name (string for the error message).
  # Returns the integer enum code of the base (non-reference) data type. Raises a type error if
  # v cannot be converted to a data type.
  try:
    v = dtypes.as_dtype(v).base_dtype
  except TypeError:
    raise TypeError("Expected DataType for argument '%s' not %s." %
                    (arg_name, repr(v)))
  i = v.as_datatype_enum
  return i


def make_shape(v, arg_name):
  # [FUNCTION DOC]
  # HELPER: converts an op attribute value for generated op wrappers.
  # KNOWN ISSUE: the docstring is just "Convert v into a list" while the real description of
  # arguments and return value sits in plain comments inside the body; it is not always a list.
  # Inputs: v (shape proto, list of integers, or tensor shape) and arg_name (string for errors).
  # Returns nothing/null value when the rank is unknown, otherwise a list of dimension sizes
  # (null entries for unknown dimensions). Raises a type error or value error, with the
  # argument name in the message, if v cannot be converted to a shape.
  """Convert v into a list."""
  # Args:
  #   v: A TensorShapeProto, a list of ints, or a tensor_shape.TensorShape.
  #   arg_name: String, for error messages.

  # Returns:
  #   None if the rank is unknown, otherwise a list of ints (or Nones in the
  #   position where the dimension is unknown).
  try:
    shape = tensor_shape.as_shape(v)
  except TypeError as e:
    raise TypeError("Error converting %s to a TensorShape: %s." % (arg_name, e))
  except ValueError as e:
    raise ValueError("Error converting %s to a TensorShape: %s." %
                     (arg_name, e))
  if shape.ndims is None:
    return None
  else:
    return shape.as_list()


def make_tensor(v, arg_name):
  # [FUNCTION DOC]
  # HELPER: converts an op attribute value for generated op wrappers.
  # Inputs: v (a tensor proto message or a string holding its text format) and arg_name (string
  # for the error message). Returns the tensor proto: v itself if already one, or a new message
  # parsed from the text. Raises a type error for any other kind of value; text that fails to
  # parse raises a parse error from the text format library.
  """Ensure v is a TensorProto."""
  if isinstance(v, tensor_pb2.TensorProto):
    return v
  elif isinstance(v, str):
    pb = tensor_pb2.TensorProto()
    text_format.Merge(v, pb)
    return pb
  raise TypeError(
      "Don't know how to convert %s to a TensorProto for argument '%s'." %
      (repr(v), arg_name))


def args_to_matching_eager(l, ctx, allowed_dtypes, default_dtype=None):
  # [FUNCTION DOC]
  # HELPER: converts the inputs of one type-constrained argument for generated op wrappers.
  # Inputs: l (sequence of values); ctx (eager context, unused); allowed_dtypes (collection of
  # data types the op accepts, may be empty); default_dtype (optional fallback data type).
  # Behavior: an empty sequence with a default type returns that type and an empty list; if all
  # entries are already eager tensors the list is returned as is with the first one's type;
  # otherwise the dtype comes from the first existing tensor, or is inferred from the first value
  # (trying the default conversion first when allowed types are given, falling back to the
  # default type), and every value is converted to that dtype.
  # Returns a pair: integer enum of the chosen data type and the list of tensors.
  # Raises a symbolic-tensor error if any converted result is a Keras symbolic tensor; conversion
  # errors propagate.
  """Convert sequence `l` to eager same-type Tensors."""
  del ctx  # Unused
  if (not l) and (default_dtype is not None):
    return default_dtype, []  # List is empty; assume default dtype.
  for x in l:
    if not isinstance(x, core_types.Value):
      break
  else:  # note: intentional for-else
    return l[0]._datatype_enum(), l  # pylint: disable=protected-access

  # Is some input already a Tensor with a dtype?
  dtype = None
  for t in l:
    if isinstance(t, core_types.Value):
      dtype = t.dtype
      break

  if dtype is None:
    # Infer a dtype based on the first value, and use that dtype for the
    # remaining values.

    ret = []
    for t in l:
      tensor = None
      # First see if we can get a valid dtype with the default conversion
      # and see if it matches an allowed dtypes. Some ops like ConcatV2 may
      # not list allowed dtypes, in which case we should skip this.
      if dtype is None and allowed_dtypes:
        tensor = tensor_conversion_registry.convert(t)
        # If we did not match an allowed dtype, try again with the default
        # dtype. This could be because we have an empty tensor and thus we
        # picked the wrong type.
        if tensor.dtype not in allowed_dtypes:
          tensor = None

      if tensor is None:
        tensor = tensor_conversion_registry.convert(
            t, dtype, preferred_dtype=default_dtype
        )

      ret.append(tensor)
      if dtype is None:
        dtype = tensor.dtype
  else:
    ret = [tensor_conversion_registry.convert(t, dtype) for t in l]

  # TODO(slebedev): consider removing this as it leaks a Keras concept.
  # pylint: disable=protected-access
  keras_symbolic_tensors = [x for x in ret if _is_keras_symbolic_tensor(x)]
  if keras_symbolic_tensors:
    raise core._SymbolicException(
        "Using symbolic output of a Keras layer during eager execution "
        "{}".format(keras_symbolic_tensors))
  # pylint: enable=protected-access
  return dtype.as_datatype_enum, ret


def convert_to_mixed_eager_tensors(values, ctx):
  # [FUNCTION DOC]
  # HELPER: converts the inputs of a mixed-type argument list for generated op wrappers.
  # Inputs: values (sequence of values, each converted independently) and ctx (unused).
  # Returns a pair: list of integer data type enums, one per value, and the list of tensors.
  # Conversion errors propagate.
  del ctx  # Unused
  v = [tensor_conversion_registry.convert(t) for t in values]
  types = [t._datatype_enum() for t in v]  # pylint: disable=protected-access
  return types, v


def args_to_mixed_eager_tensors(lists, ctx):
  # [FUNCTION DOC]
  # HELPER: converts several parallel lists of values for generated op wrappers.
  # Inputs: lists (a sequence of at least two lists of equal length); ctx (unused).
  # For each position, the dtype is taken from any existing tensor at that position across the
  # lists, otherwise from converting the first list's value; all lists' values at that position
  # are converted to it. Returns a pair: list of integer type enums (one per position) and a list
  # of lists of tensors mirroring the input shape.
  # Raises a value error if list lengths differ; fails an assertion if fewer than two lists.
  """Converts a list of same-length lists of values to eager tensors."""
  del ctx  # Unused
  assert len(lists) > 1

  # Generate an error if len(lists[i]) is not the same for all i.
  lists_ret = [[]]
  for l in lists[1:]:
    if len(l) != len(lists[0]):
      raise ValueError(
          "Expected list arguments to be the same length: %d != %d (%r vs. %r)."
          % (len(lists[0]), len(l), lists[0], l))
    lists_ret.append([])

  # Convert the first element of each list first, then the second element, etc.
  types = []
  for i in range(len(lists[0])):
    dtype = None
    # If any list has a Tensor, use that dtype
    for l in lists:
      if isinstance(l[i], core_types.Value):
        dtype = l[i].dtype
        break
    if dtype is None:
      # Convert the first one and use its dtype.
      lists_ret[0].append(tensor_conversion_registry.convert(lists[0][i]))
      dtype = lists_ret[0][i].dtype
      for j in range(1, len(lists)):
        lists_ret[j].append(
            tensor_conversion_registry.convert(lists[j][i], dtype=dtype)
        )
    else:
      # Convert everything to the found dtype.
      for j in range(len(lists)):
        lists_ret[j].append(
            tensor_conversion_registry.convert(lists[j][i], dtype=dtype)
        )
    types.append(dtype.as_datatype_enum)
  return types, lists_ret


def _is_keras_symbolic_tensor(x):
  # [FUNCTION DOC]
  # HELPER: lets the eager execution paths detect Keras symbolic tensors.
  # Input: x (any object). Returns true if x has a graph whose name is the Keras graph name.
  return hasattr(x, "graph") and getattr(x.graph, "name", None) == "keras_graph"
