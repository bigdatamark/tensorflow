# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
# [MODULE DOC]
# Public entry points for registering thread-local "op callbacks" that observe (and may override
# the outputs of) op execution in eager mode, FuncGraph execution in eager mode, and op creation
# during graph construction. Exposes add, remove, clear, a predicate telling whether callbacks
# should currently run, and the invoker used by the eager/graph machinery. State lives on the
# thread-local eager context. Adding the first callback in eager mode swaps the module-level
# eager execute function for a callback-aware variant; removing the last restores it.
# No import-time side effects beyond imports.
"""Unified callbacks op execution and creation under eager and graph modes."""

from tensorflow.python.eager import context
from tensorflow.python.eager import execute


def add_op_callback(callback_fn):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring's Args section is badly mangled by reflowing (the callback
  # signature description is run together); also the docstring says eager output overriding is
  # not implemented, so returned eager tensors are not actually applied.
  # Registers a callback in the current thread's eager context. Input: callback_fn, a callable
  # taking op type, inputs, attributes, outputs, op name and graph. Returns nothing. Raises a
  # value error if callback_fn is null or not callable. Side effect: when the context is
  # executing eagerly, replaces the module-level eager execute function with the variant that
  # invokes callbacks.
  r"""Add a thread-local callback that intercepts op execution and op creation.

  The `callback_fn` will be invoked immediately after any of the three types
  of events:
    - The execution of an TensorFlow operation ("op" for short hereafter)
      under eager mode,
    - The execution of a FuncGraph under eager mode,
    - The creation of an op during graph construction (e.g., in
      @tf.function-decorated Python functions).

  Known limitations:
    1. Under graph mode, overriding the output tensors of control-flow ops,
       including "If", "StatelessIf" and "While", may cause errors
       (b/139668453). Overriding other tensors in a graph consisting of such
       control-flow ops is okay.
    2. Under eager mode, calling eager ops from the callback function itself
       may lead to recursion stack overflow. This can be prevented by
       returning from the callback function immediately on encountering the
       op type involved (b/140334369).

  Args:
    callback_fn: A callback_fn that has the following signature: def
      callback_fn(op_type, inputs, attrs, outputs, op_name=None, graph=None): #
      op_type: The type of the op, as a string. E.g., "MatMul". #          For
        the special case of FuncGraph execution, op_type #          takes the
        name of the graph name, e.g., #          "__inference_my_func_24". #
        inputs: (`tuple` of `Tensor`s) Input tensors to the op or the #
        FuncGraph. #         - In eager execution, these are `EagerTensor`s. #
        - In graph construction, these are non-eager `Tensor`s #           that
        form the inputs to the just-created op. # attrs: The attributes of the
        op or FuncGraph of which the execution #        or creation caused the
        current invocation of the callback. #        This is applicable to both
        eager- and graph-based execution, #        as well as graph
        construction. # This is a tuple of alternating attribute keys and
        attribute # values. E.g., `('adjoint_a', False, 'adjoint_b', False)`. #
        outputs: (`tuple of `Tensor`s) Output tensors from the op or #
        FuncGraph. #          In eager execution, these are `EagerTensor`s. #
        In graph construction, these are non-eager `Tensor`s that #          are
        the outputs of the just-created op. # op_name: Name of the op. #
        - If the current invocation of the callback is due to the #
        eager execution of an op or FuncGraph, this will be #            `None`,
        as op names are meaningless in eager execution. #          - In graph
        construction, this is the name of the op, e.g., #            "MatMul_2".
        #
      graph: The graph that the op belongs to (if any). #        - In eager
        execution of an op or FuncGraph, this is `None`. #        - In graph
        construction, this is the op's enclosing graph #          as a
        `tf.Graph` object. # # Return values: #   This callback function is
        expected to return `None` or #   a `list` or `tuple` of `Tensor`s with
        its length matching #   `len(outputs)`, in the order that corresponds to
        that of the #   `outputs` argument. #   If the return value is `None`,
        downstream execution or graph #   construction will be unaffected. #
        However, if the return value is a `list` or `tuple` of `Tensor`s, #   -
        In eager execution, these returned `Tensor`s should be #
        `EagerTensor`s. Their values will replace the original values of #
        `outputs` for downstream eager execution. (*Not implemented yet*). #   -
        In graph construction, these returned `Tensor`s should be #
        non-eager `Tensor`s. Their values will replace the original #
        `outputs` for downstream graph construction.

  Raises:
    ValueError: If `callback_fn` is `None` or not callable.
  """
  # TODO(b/139668041): Implement support for overriding `EagerTensor`s from
  # callback.
  if callback_fn is None:
    raise ValueError("Passed callback function cannot be None.")
  if not callable(callback_fn):
    raise ValueError(
        "Callback function passed to op_callback() is expected to be callable, "
        f"but got {callback_fn} of type {type(callback_fn)}.")
  ctx = context.context()
  ctx.add_op_callback(callback_fn)
  if ctx.executing_eagerly():
    # Monkey-patch `execute.execute()`.
    execute.execute = execute.execute_with_callbacks


def should_invoke_op_callbacks():
  # [FUNCTION DOC]
  # KNOWN ISSUE: docstring claims a boolean is returned, but the expression returns the
  # callbacks collection itself (possibly an empty list) when no callbacks exist, or the
  # boolean negation otherwise, so the result is only truthy/falsy, not strictly boolean.
  # Takes no inputs. Reads the thread-local context and reports whether callbacks are registered
  # and none is currently being invoked (guards against recursion). No side effects.
  """Determine if op callbacks are present and should be invoked.

  Returns:
    A thread-local result (boolean) indicating whether any op callback(s) exist
    and should be invoked.
  """
  ctx = context.context()
  return ctx.op_callbacks and not ctx.invoking_op_callbacks


def remove_op_callback(op_callback):
  # [FUNCTION DOC]
  # Unregisters a previously added callback from the current thread's context. Input:
  # op_callback, the callable to remove. Returns nothing. Raises a key error (from the context)
  # if it was never registered. Side effect: if the context is executing eagerly and no
  # callbacks remain, restores the module-level eager execute function to the quick variant.
  """Remove an already-added op callback.

  Args:
    op_callback: The op callback to be removed.

  Raises:
    KeyError: If `op_callback` has not been registered using `add_op_callback()`
      before.
  """
  ctx = context.context()
  ctx.remove_op_callback(op_callback)
  if ctx.executing_eagerly() and not ctx.op_callbacks:
    # Undo monkey-patch of execute.execute if there are no more callbacks.
    execute.execute = execute.quick_execute


def clear_op_callbacks():
  # [FUNCTION DOC]
  # Removes every callback registered in the current thread by removing them one by one (which
  # also restores the eager execute function when the last one goes). No inputs, returns nothing.
  # KNOWN ISSUE: iterates directly over the context's callback collection while removal mutates
  # it, which may skip entries if the collection is the same list being modified.
  """Clear all op callbacks registered in the current thread."""
  for callback in context.context().op_callbacks:
    remove_op_callback(callback)


def invoke_op_callbacks(op_type,
                        inputs,
                        attrs,
                        outputs,
                        op_name=None,
                        graph=None):
  # [FUNCTION DOC]
  # Runs all registered callbacks, in order, for one op event. Inputs: op_type (string),
  # inputs (tensors), attrs (either a mapping of names to values, which is flattened to a tuple
  # of alternating keys and values, or already a tuple), outputs (tensors), op_name and graph
  # (both optional, default none). Each callback receives the previous callback's result as its
  # outputs. Returns the final outputs (possibly replaced by callbacks, or unchanged if
  # callbacks return nothing); if no callbacks exist, returns outputs unchanged. Raises a value
  # error if a callback returns a non-null result whose length differs from the original outputs.
  # Side effect: sets the context's invoking flag during the calls and always clears it afterward.
  r"""Invoke the callbacks that exist in the current scope (if any).

  If no callbacks are present in the current scope, this method returns
  immediately.

  Args:
    op_type: Type of the operation (e.g., "MatMul").
    inputs: Input tensors to the op. These are `EagerTensor`s in the case of
      eager execution of ops or `FuncGraph`s, and are non-eager `Tensor`s in the
      case of graph construction.
    attrs: Attributes of the op, as `tuple` of alternating keys and values.
    outputs: Output tensors from the op. These are `EagerTensor`s in the case of
      eager execution and are non-eager `Tensor`s in the case of graph
      construction.
    op_name: Name of the op. Applicable if and only if this method is invoked
      due to the graph construction of an op or the eager execution of a
      `FuncGraph`.
    graph: The graph involved (if any).
      - In the case if the eager execution of an op or FuncGraph, this is
        `None`.
      - In the case of the graph construction of an op, this is the `tf.Graph`
        object being built.

  Returns:
    `None`, or a `list` or `tuple` of output tenors that will override the
    original (input) `outputs`.
  """
  ctx = context.context()
  if ctx.op_callbacks:
    # Guards against stack overflow that can result from recursive invocation
    # due to op constructions inside client-supplied op callbacks.
    ctx.invoking_op_callbacks = True
    try:
      if isinstance(attrs, dict):
        attrs_list = []
        for key in attrs:
          attrs_list.append(key)
          attrs_list.append(attrs[key])
        attrs_tuple = tuple(attrs_list)
      else:
        attrs_tuple = attrs

      new_outputs = outputs
      for callback in ctx.op_callbacks:
        new_outputs = callback(
            op_type,
            inputs,
            attrs_tuple,
            new_outputs,
            op_name=op_name,
            graph=graph)
        if new_outputs is not None and len(new_outputs) != len(outputs):
          raise ValueError(
              f"The op callback returned {len(new_outputs)} tensors, which "
              f"does not match the original number of outputs of op {op_name} "
              f"({len(outputs)}).")
      return new_outputs
    finally:
      ctx.invoking_op_callbacks = False
  else:
    return outputs
