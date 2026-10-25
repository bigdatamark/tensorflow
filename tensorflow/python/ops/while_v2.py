# [MODULE DOC]
# Implementation of the "control flow v2" while loop: builds a single functional While (or
# StatelessWhile) operation whose condition and body are separate function graphs, and registers the
# gradient function for that operation. Exposes the loop builder (used by the public while loop
# wrapper module), the gradient function, an asynchronous no-op builder and many private helpers for
# accumulating forward-pass intermediates in tensor lists, capturing tensors, and validating shapes
# and types. Import-time side effects: registers the gradient for the While and StatelessWhile op
# types, and defines a namedtuple type used as a cache key. Works with the sibling indexed-slices
# rewriter module.
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
# =============================================================================
"""while_v2 and gradient.

This is a version of while_loop that emits a single While op, as well as the
gradient function for While ops produced by while_loop. This will eventually
replace the current tf.while_loop implementation once it reaches feature and
performance parity.
"""
import collections

from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import backprop_util
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util as util_v1
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_v2_indexed_slices_rewriter
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import variable_utils

# pylint: disable=protected-access


def while_loop(cond,
               body,
               loop_vars,
               shape_invariants=None,
               parallel_iterations=10,
               maximum_iterations=None,
               name=None,
               return_same_structure=True,
               back_prop=True):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the default for return_same_structure here is true, whereas the legacy v1 function it
  # mirrors defaults to false.
  # Builds a single While op for the loop. Inputs: cond and body are callables over the loop variables;
  # loop_vars is a possibly nested structure of tensors, indexed slices, tensor arrays or variables
  # (variables are converted to tensors); shape_invariants optionally gives allowed shapes; parallel
  # iterations (default 10) is forwarded to the op; maximum_iterations is an optional scalar bound;
  # name is an optional scope (default "while"); return_same_structure (default true) keeps the
  # structure even for a single variable; back_prop (default true) enables accumulating intermediates
  # for gradients when that is globally enabled and wraps outputs in stop-gradient when false.
  # Steps: tensor arrays become their flow tensors, a loop counter and a maximum-iterations variable
  # are prepended, cond and body are traced into function graphs (with wrapped versions that re-pack
  # arguments, squeeze a non-scalar predicate, check the output structure matches and increment the
  # counter), body captures are added as extra loop variables and duplicated in the cond graph, shape
  # compatibility and input/output counts and types are checked, optional intermediate accumulators
  # (tensor lists) are added, and the While op is built. In v1 graph mode outputs are wrapped in identity
  # ops so pruning works. Returns the final loop variables, packed like the input (a single value is
  # unpacked when return_same_structure is false and there is one output). Raises value errors for
  # shape-invariant violations and type errors for dtype changes (from the helper checks), and errors
  # from structure mismatch between body outputs and loop variables.
  """Like tf.while_loop, except emits a single While op."""
  loop_vars = variable_utils.convert_variables_to_tensors(loop_vars)
  # Keep the original loop_vars around to know which args were TensorArrays.
  orig_loop_vars = loop_vars
  flat_orig_loop_vars = nest.flatten(orig_loop_vars, expand_composites=True)
  # Cache its length since we use it at multiple places below.
  len_orig_loop_vars = len(orig_loop_vars)

  # Convert TensorArrays to their flow variables. These get converted back to
  # TensorArrays before calling `cond` and `body`. See `wrapped_cond` and
  # `wrapped_body` below.
  loop_vars = _tensor_array_to_flow(loop_vars)
  loop_vars = nest.map_structure(
      indexed_slices.internal_convert_to_tensor_or_indexed_slices,
      loop_vars,
      expand_composites=True)

  # `loop_vars_signature` is a structure of TypeSpecs and has the same
  # structure with the `orig_loop_vars`. If `shape_invariants` is not None, its
  # shape information comes from `shape_invariants` instead of `orig_loop_vars`.
  # It is used to pack flattened vars into structured vars.
  if shape_invariants is not None:
    loop_vars_signature = nest.map_structure(
        control_flow_ops._shape_invariant_to_type_spec,
        loop_vars, shape_invariants)
  else:
    loop_vars_signature = nest.map_structure(
        control_flow_ops._shape_invariant_to_type_spec, loop_vars)

  flat_shape_invariants = nest.map_structure(
      lambda spec: spec.shape,
      nest.flatten(loop_vars_signature, expand_composites=True))

  if not name:
    name = "while"

  with ops.name_scope(name) as scope:
    with ops.name_scope(None):
      cond_name = util.unique_fn_name(scope, "cond")
      body_name = util.unique_fn_name(scope, "body")
    maximum_iterations_loop_var = _build_maximum_iterations_loop_var(
        maximum_iterations)
    loop_counter = constant_op.constant(
        0,
        dtype=maximum_iterations_loop_var.dtype
        if maximum_iterations is not None else None,
        name="loop_counter")
    # Add loop counter needed for computing gradients.
    loop_vars = [loop_counter, maximum_iterations_loop_var] + list(loop_vars)

    func_graph_signature = (
        [tensor_spec.TensorSpec.from_tensor(loop_counter),
         tensor_spec.TensorSpec.from_tensor(maximum_iterations_loop_var)] +
        list(loop_vars_signature))

    # Automatic control dependencies are added in defuns, but not in v1
    # graphs. Propagate that behavior here.
    add_control_dependencies = ops.get_default_graph()._add_control_dependencies

    def wrapped_cond(loop_counter, maximum_iterations_arg, *args):
      # [FUNCTION DOC]
      # HELPER: Nested helper of the loop builder; this is the condition function traced into the cond graph.
      # Takes the loop counter, the maximum iterations argument and the flattened loop variables. Re-packs
      # the variables into the original structure (flows become tensor arrays), calls the user condition,
      # squeezes the predicate if it is not scalar, and returns it. When a maximum number of iterations was
      # given, returns the logical and of counter below the maximum and the predicate.
      """Extra `cond` wrapper that can handle the extra counter loop_var."""
      # Convert the flow variables in `args` to TensorArrays. `args` should
      # already have the same structure as `orig_loop_vars` but currently there
      # is no nest.zip so we call `_pack_sequence_as` which flattens `args`,
      # converts flows in `args` to TensorArrays and packs it into the
      # structure of `loop_vars_signature`.
      pred = cond(
          *_pack_sequence_as(loop_vars_signature, flat_orig_loop_vars, args))
      if (tensor_util.is_tf_type(pred) and
          (pred.shape.dims is None or pred.shape.dims)):
        pred = array_ops.squeeze_v2(pred)

      if maximum_iterations is None:
        return pred
      else:
        return math_ops.logical_and(
            loop_counter < maximum_iterations_arg, pred)

    cond_graph = func_graph_module.func_graph_from_py_func(
        cond_name,
        wrapped_cond,
        [],  # We provide signature instead of args.
        {},
        signature=func_graph_signature,
        func_graph=util.WhileCondFuncGraph(
            cond_name, collections=ops.get_default_graph()._collections),  # pylint: disable=protected-access
        add_control_dependencies=add_control_dependencies)

    def wrapped_body(loop_counter, maximum_iterations_arg, *args):
      # [FUNCTION DOC]
      # HELPER: Nested helper of the loop builder; this is the body function traced into the body graph.
      # Takes the loop counter, the maximum iterations argument and the flattened loop variables. Copies
      # handle data onto the placeholders, captures the cond graph captures first so ordering matches, calls
      # the user body on the re-packed variables, wraps a non-nested result in a list, asserts that the
      # output structure matches the original loop variables (retrying after converting variables to
      # tensors, and ignoring list versus tuple differences), converts tensor arrays to flows, and returns
      # the incremented counter, the maximum iterations argument and the outputs. Raises a value error on
      # structure mismatch.
      """Loop body augmented with counter update.

      Args:
        loop_counter: Loop counter which needs to be incremented in the body.
        maximum_iterations_arg: Maximum iterations of the loop.
        *args: List of args

      Returns:
        A list of tensors the same length as args.
      """
      # The function was created with a signature rather than tensors, so
      # internal placeholders were created without handle data.
      _copy_handle_data(nest.flatten(loop_vars[2:], expand_composites=True),
                        nest.flatten(args, expand_composites=True))
      # Capture the tensors already captured in cond_graph so that they appear
      # in the same order in body_graph.external_captures.
      for t in cond_graph.external_captures:
        ops.get_default_graph().capture(t)

      # Convert the flow variables in `args` to TensorArrays. `args` should
      # already have the same structure as `orig_loop_vars` but currently there
      # is no nest.zip so we call `_pack_sequence_as` which flattens `args`,
      # converts flows in `args` to TensorArrays and packs it into the
      # structure of `loop_vars_signature`.
      outputs = body(
          *_pack_sequence_as(loop_vars_signature, flat_orig_loop_vars, args))
      if not nest.is_nested(outputs):
        outputs = [outputs]
      try:
        # The legacy while_loop considers list and tuple to be the same
        # structure.
        nest.assert_same_structure(outputs, orig_loop_vars, check_types=False,
                                   expand_composites=True)
      except ValueError:
        # Traditionally we consider variables and tensors to be the same
        # structure.
        vars1 = variable_utils.convert_variables_to_tensors(outputs)
        vars2 = variable_utils.convert_variables_to_tensors(orig_loop_vars)
        nest.assert_same_structure(vars1, vars2, check_types=False,
                                   expand_composites=True)
      outputs = _tensor_array_to_flow(outputs)

      # TODO(srbs): Update lowering code to create _Enter nodes with
      # is_constant=True for inputs that are directly passed to outputs.
      return [loop_counter + 1, maximum_iterations_arg] + list(outputs)

    body_graph = func_graph_module.func_graph_from_py_func(
        body_name,
        wrapped_body,
        [],  # We provide signature instead of args.
        {},
        signature=func_graph_signature,
        func_graph=util.WhileBodyFuncGraph(
            body_name, collections=ops.get_default_graph()._collections),  # pylint: disable=protected-access
        add_control_dependencies=add_control_dependencies)
    # Add external captures of body to the list of loop vars.
    # Note that external tensors will be treated as loop invariants, i.e.,
    # the value of that tensor in each iteration is the same as it was at the
    # beginning of the loop execution.
    deferred_external_captures = nest.flatten(
        [c() for c in body_graph.deferred_external_captures],
        expand_composites=True)
    loop_vars = (
        loop_vars + body_graph.external_captures + deferred_external_captures)
    # TODO(srbs): Update lowering code to create _Enter nodes with
    # is_constant=True for inputs that are directly passed to outputs.
    body_graph.outputs.extend(body_graph.internal_captures)
    body_graph.outputs.extend(body_graph.deferred_internal_captures)

    # Capture the extra `external_captures` of `body_graph` in `cond_graph` so
    # that it expects to receive those as arguments.
    with cond_graph.as_default():
      num_cond_captures = len(cond_graph.external_captures)
      assert (cond_graph.external_captures ==
              body_graph.external_captures[:num_cond_captures])
      _duplicate_body_captures_in_cond(
          cond_graph, body_graph.external_captures[num_cond_captures:] +
          deferred_external_captures)

    # Make sure that the shapes of the loop outputs are compatible with the
    # shape invariants, or the shapes of the loop vars if the invariants are not
    # specified.
    num_flattened_outputs = len(nest.flatten(orig_loop_vars,
                                             expand_composites=True))
    # First var is loop counter and second var is maximum_iterations.
    first_loop_var_index = 2
    _check_shapes_compat(
        body_graph.outputs[first_loop_var_index:first_loop_var_index +
                           num_flattened_outputs],
        flat_shape_invariants,
        nest.flatten(loop_vars[first_loop_var_index:first_loop_var_index +
                               len_orig_loop_vars], expand_composites=True))

    num_original_outputs = len(body_graph.outputs)
    if back_prop and util.output_all_intermediates():
      # Export all tensors in the loop body that may be needed for gradient
      # computation. We do this by accumulating the intermediate values in
      # TensorLists.
      intermediate_tensors = _get_intermediates(body_graph)

      for intermediate_tensor in intermediate_tensors:
        tensor_list = list_ops.empty_tensor_list(
            element_dtype=intermediate_tensor.dtype,
            element_shape=intermediate_tensor.shape,
            max_num_elements=maximum_iterations)
        loop_vars.append(tensor_list)
        with cond_graph.as_default():
          # Add a placeholder to cond_graph's inputs corresponding to the
          # tensor_list.
          cond_graph.capture(tensor_list)
        with body_graph.as_default():
          # Push the intermediate tensor to the tensor list. This captures the
          # `tensor_list` as well.
          appended_tensor_list = list_ops.tensor_list_push_back(
              tensor_list, intermediate_tensor)
          # Add this modified tensor list to the list of outputs.
          body_graph.outputs.append(appended_tensor_list)

    flattened_loop_vars = nest.flatten(loop_vars, expand_composites=True)
    _check_num_inputs_outputs(cond_graph, body_graph,
                              len(flattened_loop_vars))
    _check_inputs_outputs_types_match(body_graph, flattened_loop_vars)

    with ops.control_dependencies(
        list(cond_graph.function_captures.control) + list(
            body_graph.function_captures.control)):
      output_shapes = [t.shape for t in body_graph.outputs]
      orig_loop_vars_range = slice(first_loop_var_index,
                                   first_loop_var_index + num_flattened_outputs)
      output_shapes[orig_loop_vars_range] = flat_shape_invariants

      outputs = _build_while_op(
          flattened_loop_vars,
          cond_graph,
          body_graph,
          output_shapes=output_shapes,
          parallel_iterations=parallel_iterations,
          name=scope,
          num_original_outputs=num_original_outputs)
    if not ops.get_default_graph().building_function:
      # In V1 graph mode, return identities for each output of the While op,
      # rather than the output of the While op directly. This makes pruning work
      # if the output of while_loop() is fetched: the lowering pass converts the
      # While outputs into IdentityN outputs, which if fetched will cause all
      # ops in the body to be run (since it takes all exit ops as input). After
      # lowering, each output identity op will end up with only the appropriate
      # exit op as input.
      outputs = tuple(array_ops.identity(t) for t in outputs)

  output_loop_vars = outputs[first_loop_var_index:first_loop_var_index +
                             num_flattened_outputs]
  if not back_prop:
    output_loop_vars = [array_ops.stop_gradient(t) for t in output_loop_vars]
  outputs = _pack_sequence_as(
      loop_vars_signature, flat_orig_loop_vars, output_loop_vars)

  if return_same_structure:
    return outputs

  flattened_outputs = nest.flatten(outputs, expand_composites=True)
  if len(flattened_outputs) == 1:
    return flattened_outputs[0]
  else:
    return outputs


@ops.RegisterGradient("StatelessWhile")
@ops.RegisterGradient("While")
def _WhileGrad(op: ops.Operation, *grads):  # pylint: disable=invalid-name
  # [FUNCTION DOC]
  # Gradient function registered for the While and StatelessWhile ops. Inputs: op, the While operation
  # (possibly not the real one, since the tape may forget it), and the incoming gradients, one per
  # output. Recovers the forward cond and body graphs, preprocesses incoming gradients, drops gradients
  # for skipped capture inputs, builds the body gradient graph (accumulating forward intermediates), and
  # if that needs extra forward outputs, rewrites the forward op in place (new function attributes,
  # type list, output shapes, extra inputs and outputs, handle data); this mutation may make active
  # sessions unrunnable (per an inline note). Then marks the number of original outputs, resolves
  # captured inputs, rewrites indexed-slices gradients, builds a cond graph that compares the counter
  # against the forward iteration count, builds the gradient While op and returns the structured
  # gradients (None for outputs without incoming gradient). Raises an assertion error if the rewritten
  # op's input and output lengths are inconsistent. Also renames the forward graphs when rewriting.
  """The gradient of a While op produced by while_loop."""
  # Note that op is not always the same as while_op because the gradient tape,
  # for eager mode compatibility, forgets information about the proper op. Since
  # the loop cannot run in eager mode, however, we can safely introspect into
  # the graph here.
  while_op = op.outputs[0].op
  cond_graph = _get_graph(while_op, "cond", "_cond_graph")
  body_graph = _get_graph(while_op, "body", "_body_graph")
  orig_num_params = len(body_graph.outputs)

  maximum_iterations = op.inputs[1]
  parallel_iterations = op.get_attr("parallel_iterations")

  try:
    num_original_outputs = while_op.get_attr("_num_original_outputs")
  except:  # pylint: disable=bare-except
    num_original_outputs = len(while_op.outputs)

  num_intermediates = len(while_op.outputs) - num_original_outputs
  grads = [
      _preprocess_grad(grad, body_out, while_in, while_out)  # pylint: disable=g-complex-comprehension
      for grad, body_out, while_in, while_out in zip(
          grads[:num_original_outputs],
          body_graph.outputs[:num_original_outputs],
          while_op.inputs[:num_original_outputs],
          while_op.outputs[:num_original_outputs])
  ] + [None] * num_intermediates

  # Skip gradients with respect to the captures whenever possible.
  if getattr(op, "skip_input_indices", None) is not None:
    captures_start_index = (
        len(body_graph.inputs) - len(body_graph.internal_captures))
    for i in op.skip_input_indices:
      if i >= captures_start_index:
        grads[i] = None

  # We compute the gradient for the sub-graph between trainable ys and xs
  # with non-None incoming gradients. We later pad the None's to the list of
  # outputs.
  ys, xs, non_none_grads = zip(*[(y, x, grad) for (y, x, grad) in zip(
      body_graph.outputs, body_graph.inputs, grads) if grad is not None])

  body_grad_graph, args = _create_grad_func(
      ys, xs, non_none_grads, cond_graph, body_graph,
      util.unique_grad_fn_name(body_graph.name), op, maximum_iterations)

  if body_grad_graph.while_op_needs_rewrite:
    # Modify 'op' to output the intermediate accumulators needed by the grad
    # function.
    # NOTE(skyewm): if there are any active sessions, this modification to `op`
    # may make them unrunnable!

    cond_graph.name += "_rewritten"
    body_graph.name += "_rewritten"

    # `body_grad_graph.extra_inputs` here is equivalent to skimming off the new
    # `body_graph.external_captures` added during `_create_grad_func`.
    new_inputs = body_grad_graph.extra_inputs
    new_outputs = body_graph.outputs[orig_num_params:]

    while_op._set_func_attr("cond", util.create_new_tf_function(cond_graph))
    while_op._set_func_attr("body", util.create_new_tf_function(body_graph))
    if len(body_graph.output_types) != len(while_op.inputs) + len(new_inputs):
      # Continuing leads to an invalid graph with disconnected inputs.
      raise AssertionError(
          "Inputs and outputs constructed for the forward op of a While "
          "gradient don't match with 'output_types' at  "
          f"{len(body_graph.output_types)},'inputs' at length "
          f"{len(while_op.inputs)}, and 'new_inputs' at length "
          f"{len(new_inputs)}. This doesn't make sense, please file a bug.")
    while_op._set_type_list_attr("T", body_graph.output_types)
    while_op._set_shape_list_attr("output_shapes", body_graph.output_shapes)
    while_op._add_while_inputs(new_inputs)
    while_op._add_outputs([t.dtype for t in new_outputs],
                          [t.shape for t in new_outputs])
    _copy_handle_data(new_outputs, while_op.outputs[orig_num_params:])

  # Do not ignore grads wrt extra outputs when computing higher order
  # derivatives.
  while_op._set_attr("_num_original_outputs",
                     attr_value_pb2.AttrValue(i=len(while_op.outputs)))

  captured_inputs = _resolve_grad_captures(body_graph, body_grad_graph,
                                           while_op)
  loop_vars = args + captured_inputs

  # This modifies body_grad_graph.
  loop_vars = while_v2_indexed_slices_rewriter.rewrite_grad_indexed_slices(
      grads, body_grad_graph, loop_vars, while_op.inputs)

  def grad_cond(counter, unused_maximum_iterations_arg, forward_loop_iters,
                *unused_args):
    # [FUNCTION DOC]
    # HELPER: Nested condition function for the gradient loop. Takes the counter, an unused maximum
    # iterations argument, the forward iteration count and unused remaining arguments; returns whether the
    # counter is still below the forward iteration count.
    return counter < forward_loop_iters

  grad_cond_name = util.unique_grad_fn_name(op.get_attr("cond").name)
  cond_grad_graph = func_graph_module.func_graph_from_py_func(
      grad_cond_name, grad_cond, loop_vars, {},
      func_graph=util.WhileCondFuncGraph(grad_cond_name))

  _check_num_inputs_outputs(cond_grad_graph, body_grad_graph, len(loop_vars))

  outputs = _build_while_op(
      loop_vars,
      cond_grad_graph,
      body_grad_graph,
      output_shapes=[t.shape for t in body_grad_graph.outputs],
      parallel_iterations=parallel_iterations,
      name="%s_grad" % while_op.name,
      num_original_outputs=len(body_grad_graph.outputs))

  # See comment in while_loop.
  outputs = [array_ops.identity(t) for t in outputs]
  return _get_structured_grad_output(outputs, grads, body_grad_graph)


def _build_while_op(loop_vars, cond_graph, body_graph, output_shapes,
                    parallel_iterations, name, num_original_outputs):
  # [FUNCTION DOC]
  # HELPER: Builds the functional While operation for a loop (used for forward and gradient loops).
  # Inputs: loop_vars (flattened inputs), cond_graph and body_graph (function graphs), output_shapes,
  # parallel_iterations, name, and the number of original outputs. Chooses the stateful While op if
  # either graph contains stateful ops, otherwise the stateless variant. Creates the op (via a tape
  # gradient wrapper), copies handle data, sets lowering and XLA compile-time-constant attributes, the
  # read-only resource inputs attribute and the original-output-count attribute, links the graphs to the
  # op, and sets their outer graph to the current default graph. Returns the op output tensors.
  """Builds the functional StatelessWhile/While op."""
  cond_stateful_ops = [
      op for op in cond_graph.get_operations() if op._is_stateful
  ]
  body_stateful_ops = [
      op for op in body_graph.get_operations() if op._is_stateful
  ]
  if (cond_stateful_ops or body_stateful_ops):
    op_fn = gen_functional_ops._while
  else:
    op_fn = gen_functional_ops.stateless_while

  def _make_op(inputs):
    # [FUNCTION DOC]
    # HELPER: Nested helper of the While op builder that actually creates the op from given inputs, sets
    # attributes and cached graphs on it, and returns its output tensors. Side effects: sets the outer
    # graph of cond and body graphs and attributes on the new op.
    while_op, tensors = util.get_op_and_outputs(op_fn(
        inputs,
        util.create_new_tf_function(cond_graph),
        util.create_new_tf_function(body_graph),
        output_shapes=output_shapes,
        parallel_iterations=parallel_iterations,
        name=name))
    _copy_handle_data(body_graph.outputs, tensors)
    util.maybe_set_lowering_attr(while_op)
    util.maybe_propagate_compile_time_consts_in_xla(while_op)
    _set_read_only_resource_inputs_attr(while_op, [cond_graph, body_graph])
    # This is needed so we do not compute derivative wrt these extra outputs.
    while_op._set_attr("_num_original_outputs",
                       attr_value_pb2.AttrValue(i=num_original_outputs))
    # The while op may be created inside a tf.function, in which case ops
    # needs to capture "through" it when taking gradients; outer_graph is used
    # as a sanity check that capturing only happens from parent to child.
    cond_graph.outer_graph = ops.get_default_graph()
    body_graph.outer_graph = ops.get_default_graph()
    while_op._cond_graph = cond_graph
    while_op._body_graph = body_graph
    return tensors
  return util.run_as_function_for_tape_gradients(_make_op, loop_vars)


def _get_intermediates(func_graph):
  # [FUNCTION DOC]
  # HELPER: Selects forward-body tensors to accumulate for gradient computation. Input: a function
  # graph. Returns a list of output tensors of its operations, skipping identity and mutex-lock ops,
  # the loop counter input, resource tensors, tensors that already have an accumulator and tensors that
  # are captured values (loop invariants). No side effects.
  """Returns all tensors in `func_graph` that should be accumulated."""
  # We currently accumulate output tensors of most ops in the function and rely
  # on the pruning pass to get rid of the unused accumulators at runtime.
  # However, this can bloat the GraphDef and make debugging harder so we perform
  # some optimizations.
  #
  # Optimization we currently perform:
  # 1. We do not accumulate tensors which already have an accumulator
  #    in the loop body.
  # 2. We do not accumulate outputs of Identity nodes. When building the
  #    FuncGraph, we add an Identity node for each output (see
  #    `AutomaticControlDependencies.mark_as_return`). Accumulating outputs
  #    of all these nodes bloats the GraphDef quite a bit so we remove those.
  #    Since the gradient of an Identity node does not rely on its forward op's
  #    input this is safe to do.
  #
  # Other possible optimizations:
  # 1. Only accumulate tensors that will be required by the backward pass.
  #    This will require running the gradient pass and hence would increase the
  #    graph building time for the forward pass.
  # 2. Do not accumulate Const nodes created inside the loop body.
  # 3. Do not accumulate loop vars that are returned as-is just like captured
  #    tensors.
  intermediates = []
  reverse_captures = dict((v.ref(), k) for k, v in func_graph.captures)

  for op in func_graph.get_operations():
    if op.type == "Identity":
      continue
    # Accumulating mutexes can cause deadlock.
    if op.type == "MutexLock":
      continue
    for o in op.outputs:
      if (o is not func_graph.inputs[0] and  # Loop counter.
          o.dtype != dtypes.resource and  # Do not accumulate resource tensors.
          _get_accumulator(o) is None and  # Has existing accumulator.
          o.ref() not in reverse_captures
         ):  # Captured value, hence loop invariant.
        intermediates.append(o)
  return intermediates


def _preprocess_grad(grad, body_graph_output, while_op_input, while_op_output):
  # [FUNCTION DOC]
  # HELPER: Computes the initial gradient for one While output. Inputs: the incoming gradient, the
  # matching body graph output, While op input and While op output. Returns nothing (null) when the body
  # output is not trainable; zeros for resource or variant outputs whose gradient is missing but
  # supports a default gradient; a dense tensor when the gradient is indexed slices; otherwise the
  # gradient unchanged.
  """Returns the initial gradient to be used for a given output tensor.

  Args:
    grad: the original gradient Tensor passed to the gradient function.
    body_graph_output: the corresponding Tensor in the body graph.
    while_op_input: the corresponding Tensor input of the While op.
    while_op_output: the corresponding Tensor output of the While op.

  Returns:
    A Tensor or None.
  """
  # Set the incoming gradient of non-trainable inputs to None. It is possible
  # that we receive non-None gradients for non-trainable types in nested while
  # loops because we accumulate outputs of the inner while as variant tensors
  # which are trainable and hence receive zeros_like tensors in the gradient
  # pass. The non-trainable tensors then receive the popped zeros tensor from
  # this zeros variant. The gradient for the loop vars corresponding to these
  # tensors is None or zeros (this happens only if the loop var is accumulated
  # as well) in _grad_fn so we reset these.
  # TODO(b/118712257): Remove once we can handle None output grads in _grad_fn.
  if not _is_trainable(body_graph_output):
    return None

  # GradientTape initializes resource and variant grads as None instead of
  # zeros. Set to zeros so _GradientsHelper computes the gradients instead of
  # returning None.
  # TODO(b/143286622): The supports_default_grad check is needed
  # because While op emits non-differentiable resource tensors
  # as outputs. Remove this check when that is not the case.
  # Note: We use `while_op_input` instead of `while_op_output` for the call
  # to `supports_default_grad` because `while_op_output` may be missing
  # handle_data if the While is in a restored saved model.
  if (while_op_output.dtype in (dtypes.resource, dtypes.variant) and
      default_gradient.supports_default_grad(while_op_input) and grad is None):
    return _zeros_like(while_op_input, while_op_output)

  # Convert IndexedSlices to dense tensors since it is unlikely that downstream
  # gradient functions with properly handle indexed slices. This is similar to
  # what we do in tf.function gradients.
  if isinstance(grad, indexed_slices.IndexedSlices):
    return ops.convert_to_tensor(grad)

  return grad


# TODO(skyewm): make this return constants if op_output's shape is fully
# defined (this can be done by checking the "shape" attr of resource vars).
def _zeros_like(op_input, op_output):
  # [FUNCTION DOC]
  # HELPER: Creates zeros matching a While output, also supporting resource variable handles. For a
  # resource output, builds zeros with the variable's shape and the dtype derived from the op input;
  # otherwise zeros shaped like the output. Inputs: op input and op output tensors. Returns a tensor.
  """Like array_ops.zeros_like() but also accepts resource var handles."""
  if op_output.dtype == dtypes.resource:
    # Note: We use `op_input` instead of `op_output` to get the zeros dtype
    # because `op_output` may be missing handle_data if the While is in a
    # restored saved model.
    return array_ops.zeros(
        gen_resource_variable_ops.variable_shape(op_output),
        dtype=default_gradient.get_zeros_dtype(op_input))
  return array_ops.zeros_like(op_output)


def _is_trainable(tensor):
  # [FUNCTION DOC]
  # HELPER: Decides whether a tensor can have gradients. Returns false if not trainable by the general
  # rule, or if it is the variant value popped from a tensor list whose element type is untrainable
  # (in which case the element type decides). Otherwise true. Asserts the popped value is variant.
  """Returns whether the given tensor is trainable."""
  if not backprop_util.IsTrainable(tensor):
    return False

  # Special case: untrainable accumulator output. The gradients algorithm
  # doesn't know about tensor lists of untrainable elements. In theory the
  # tensor list gradient functions should return None as appropriate, but
  # because we can't return None from the gradient function we filter out
  # untrainable accumulator output here to avoid computing the gradient at all.
  if tensor.op.type == "TensorListPopBack" and tensor.value_index == 0:
    assert tensor.dtype == dtypes.variant
    element_type = tensor.op.get_attr("element_dtype")
    return backprop_util.IsTrainable(element_type)

  return True


def _get_graph(while_op, func_attr_name, attr_graph_name):
  # [FUNCTION DOC]
  # HELPER: Returns the function graph for a While op's cond or body. Inputs: the op, the function
  # attribute name and the name of the cached graph attribute. Uses the cached graph if present;
  # otherwise rebuilds it from the op's output shapes and function name. Side effect: stores a
  # reference to the op on the graph. Returns the function graph.
  """Returns `FuncGraph` for the given function attribute.

  Args:
    while_op: The While Operation.
    func_attr_name: string
    attr_graph_name: cached forward graph name

  Returns:
    `FuncGraph`
  """
  func_graph = getattr(while_op, attr_graph_name, None)
  if func_graph is None:
    # TODO(srbs): Handle TensorShapeProto in function_def_to_graph.input_shapes.
    input_shapes = [
        tensor_shape.TensorShape(s) for s in while_op.get_attr("output_shapes")
    ]
    func_name = while_op.get_attr(func_attr_name).name
    func_graph = util.get_func_graph(while_op, input_shapes, func_name)
  func_graph._while = while_op
  return func_graph


def _create_grad_func(ys, xs, grads, cond_graph, body_graph, name, while_op,
                      maximum_iterations):
  # [FUNCTION DOC]
  # HELPER: Builds the body gradient function graph. Inputs: ys and xs tensors, incoming gradients,
  # the forward cond and body graphs, the new graph name, the forward While op and the maximum
  # iterations. Creates a counter, builds the gradient graph (via the gradient helper) inside a
  # specialized gradient function graph, then appends outputs for every captured tensor using the
  # mapping from internal captures to outputs. Returns a pair: the gradient graph and the argument list
  # (counter, maximum iterations, total iterations, gradients). Raises a value error if a captured
  # tensor has no recorded output; asserts equal lengths of ys and grads.
  """Builds and returns the gradient FuncGraph of `func_graph` and its args.

  The returned grad_func_graph must be called with the returned
  args + grad_func_graph.captures.

  Args:
    ys: A `Tensor` or list of tensors to be differentiated.
    xs: A `Tensor` or list of tensors to be used for differentiation.
    grads: The incoming grads for `ys`.
    cond_graph: FuncGraph for the forward cond function.
    body_graph: FuncGraph for the forward body function.
    name: Name of the returned gradient function.
    while_op: The forward While op.
    maximum_iterations: Tensor. The maximum number of iterations.

  Returns:
    2-tuple of (grad_func_graph, args).
  """
  assert len(ys) == len(grads)

  total_iters = while_op.outputs[0]
  counter = constant_op.constant(
      0, dtype=total_iters.dtype, name="grad_counter")

  # Build frozen sets so that we do not have linear time lookups in
  # `_is_loop_invariant`. Note: `body_graph.inputs` and `body_graph.outputs`
  # may get updated during gradient computation because we add accumulators to
  # the forward op. However, those are not loop invariants so wouldn't affect
  # the output of `_is_loop_invariant`. Also we would never attempt to capture
  # those accumulators so `_is_loop_invariant` should never receive those new
  # tensors as args.
  body_graph_inputs = object_identity.ObjectIdentitySet(body_graph.inputs)
  body_graph_outputs = object_identity.ObjectIdentitySet(body_graph.outputs)

  args = [counter, maximum_iterations, total_iters] + list(grads)
  # Note: The returned function does not have `args` in the list of
  # `external_captures`.
  grad_func_graph = func_graph_module.func_graph_from_py_func(
      name,
      lambda *args: _grad_fn(ys, xs, args, body_graph),
      args, {},
      func_graph=_WhileBodyGradFuncGraph(name, cond_graph, body_graph,
                                         maximum_iterations, while_op,
                                         body_graph_inputs, body_graph_outputs))

  # Update the list of outputs with tensors corresponding to the captured
  # tensors. We capture 3 types of tensors when building the grad fn:
  # 1. Accumulators for forward graph intermediates which are not loop
  #    invariants. The outputs corresponding to these are populated in
  #    `internal_capture_to_output` by `_WhileBodyGradFuncGraph`.
  # 2. Resources, which are output as is.
  # 3. Forward graph loop invariants, which are output as is.
  for external_capture, internal_capture in grad_func_graph.captures:
    if (ops.tensor_id(internal_capture)
        in grad_func_graph.internal_capture_to_output):
      new_output = grad_func_graph.internal_capture_to_output[ops.tensor_id(
          internal_capture)]
    else:
      raise ValueError(
          f"Tensor {str(internal_capture)} which captures "
          f"{str(external_capture)} is in list of "
          f"internal_captures but not in internal_capture_to_output.")
    grad_func_graph.outputs.append(new_output)
    grad_func_graph.structured_outputs.append(new_output)

  return grad_func_graph, args


def _grad_fn(ys, xs, args, func_graph):
  # [FUNCTION DOC]
  # HELPER: Body function of the gradient loop built by the gradient graph builder.
  # KNOWN ISSUE: the docstring describes argument 1 as total iterations and argument 2 as maximum
  # iterations, but the code (and the caller) use argument 1 for maximum iterations and argument 2 for
  # total iterations.
  # Computes gradients of the forward graph outputs with respect to its inputs using the incoming
  # gradients (arguments from index 3), requesting zeros for unconnected gradients, and asserts none is
  # null. Returns the incremented counter, maximum iterations, total iterations and the gradient tensors.
  """Computes the gradient of `func_graph` in the current graph.

  This function builds the gradient graph of the corresponding forward-pass
  `func_graph` by differentiating `func_graph`'s outputs w.r.t. its inputs.

  Args:
    ys: A `Tensor` or list of tensors to be differentiated.
    xs: A `Tensor` or list of tensors to be used for differentiation.
    args: The input arguments.
      args[0] - Loop counter
      args[1] - Total number of iterations.
      args[2] - maximum_iterations.
      args[3:] - Incoming gradients for `ys`.
    func_graph: function.FuncGraph. The corresponding forward-pass function.

  Returns:
    The output gradient Tensors.
  """
  grad_ys = args[3:]

  # Build the gradient graph. Note that this builds the gradient computation of
  # func_graph in the current graph, which requires capturing tensors from
  # func_graph. The captured func_graph tensors are resolved to external tensors
  # after the forward While op has been rewritten in _resolve_grad_captures.
  # TODO(srbs): Mark GradientsHelper as public?
  grad_outs = gradients_util._GradientsHelper(
      ys, xs, grad_ys=grad_ys, src_graph=func_graph,
      unconnected_gradients="zero")

  # TODO(b/118712257): Handle the case when grad_outs has None's e.g. when there
  # is a tf.StopGradient in the loop body.
  assert all(g is not None for g in grad_outs)
  counter = args[0]
  maximum_iterations = args[1]
  total_iters = args[2]
  return [counter + 1, maximum_iterations, total_iters] + grad_outs


def _resolve_grad_captures(body_graph, body_grad_graph, while_op):
  # [FUNCTION DOC]
  # HELPER: Maps external captures of the body gradient graph to tensors usable in the outer graph.
  # Captures from the forward body graph are replaced by the matching While op output (and asserted to
  # be in the outer graph); others are kept. Inputs: forward body graph, body gradient graph, While op.
  # Returns the list of captured inputs.
  """Returns the tensors to pass as captured inputs to `body_grad_graph`.

  `body_grad_graph` may have external references to:
  1. Its outer graph containing the input gradients. These are left as-is.
  2. Accumulators captured from the forward-pass graph. These should have been
     added as `while_op` outputs after the gradient graph was built. We replace
     these with the corresponding output of `while_op`, i.e. a tensor in
     `body_graph.outer_graph`. In the case of nested control flow or functions,
     the gradient logic handling `body_grad_graph.outer_graph` will make sure
     the tensor from `body_graph.outer_graph` is also correctly captured.

  Args:
    body_graph: FuncGraph. The forward-pass body function.
    body_grad_graph: FuncGraph. The body gradients function.
    while_op: The forward-pass While Operation calling `body_graph`.

  Returns:
    A list of input tensors to be passed as the captured inputs to
    `body_grad_graph`.
  """
  new_capture_inputs = []
  for t in body_grad_graph.external_captures:
    # Resolve tensors captured from the forward graph to the outputs of the
    # forward while_op.
    if t.graph == body_graph:
      # Captured accumulator or loop invariant.
      for i, output in enumerate(t.graph.outputs):
        if output is t:
          t = while_op.outputs[i]
          break

      # Note: We rely on the capturing logic of the gradient While op graph to
      # correctly capture the tensors in `body_graph.outer_graph`. Both cond_v2
      # and while_v2 handle this while building their gradient functions.
      assert t.graph == body_graph.outer_graph

    new_capture_inputs.append(t)
  return new_capture_inputs


def _get_structured_grad_output(outputs, grads, body_grad_graph):
  # [FUNCTION DOC]
  # HELPER: Converts raw gradient loop outputs into the gradient values to return. Inputs: raw outputs,
  # the incoming gradients and the body gradient graph. For each incoming gradient: null stays null;
  # indexed-slices outputs are reassembled from three consecutive raw outputs (values, indices, dense
  # shape); tensors take one raw output. The first three raw outputs (counter, maximum iterations, total
  # iterations) are skipped. Returns a list that may contain nulls.
  """Returns the values that should be returned from the while grad function.

  Args:
    outputs: the raw Tensor outputs of the grad While op.
    grads: the input gradients to the gradient function.
    body_grad_graph: _WhileBodyGradFuncGraph.

  Returns:
    A list of gradient values. May include Nones.
  """
  result = []
  # outputs[0] is the loop counter.
  # outputs[1] is maximum_iterations.
  # outputs[2] is the total number of loop iterations.
  outputs_idx = 3
  structured_outputs_idx = 3
  for g in grads:
    # Set None as the output gradient for tensors with None input gradient.
    if g is None:
      result.append(None)
      continue
    output = body_grad_graph.structured_outputs[structured_outputs_idx]
    structured_outputs_idx += 1
    if isinstance(output, indexed_slices.IndexedSlices):
      # TODO(skyewm): is there a more robust way to determine the order of
      # flattened IndexedSlices components?
      result.append(indexed_slices.IndexedSlices(
          values=outputs[outputs_idx],
          indices=outputs[outputs_idx + 1],
          dense_shape=outputs[outputs_idx + 2]))
      outputs_idx += 3
    else:
      assert isinstance(output, tensor_lib.Tensor)
      result.append(outputs[outputs_idx])
      outputs_idx += 1

  return result


def _get_accumulator(tensor):
  # [FUNCTION DOC]
  # HELPER: Finds existing accumulators for the gradient machinery.
  # KNOWN ISSUE: the docstring calls the graph output output_t in one place but the pattern names it
  # output_tl; also the nested lookup it uses has a shadowed variable (see that function).
  # Looks for an existing accumulator (tensor list) for a tensor in a function graph: a push-back
  # op consuming the tensor and a loop-input list whose output (or identity of it) is a graph output at
  # the same index as the input. Returns that output tensor or null. Asserts the tensor belongs to a
  # function graph.
  r"""Returns TensorList if any containing accumulated values of tensor.

  We try to find a pattern of the form:

     input_tl   tensor
        \        /
    (TensorListPushBack)
            |
        output_tl

  which satisfies the following conditions:

  1. input_tl must be in tensor.graph.inputs.
  2. output_tl or Identity(output_tl) must be in tensor.graph.outputs.
  3. tensor.graph.input_index(input_tl) == tensor.graph.output_index(output_t).

  output_tl or Identity(output_tl) (whichever is in tensor.graph.outputs) is
  returned if such a pattern is found else None is returned.

  Args:
    tensor: The Tensor to be accumulated.

  Returns:
    A variant tensor in the same graph as `tensor` or None if no accumulator is
    found.
  """
  assert isinstance(tensor.graph, func_graph_module.FuncGraph)

  def get_func_graph_output(t):
    # [FUNCTION DOC]
    # HELPER: Nested helper of the accumulator lookup.
    # KNOWN ISSUE: it indexes the first consumer of the tensor without checking that any consumer exists
    # (fails with an index error if none), and the inner generator reuses the name of the argument.
    # Returns the given tensor if it is a graph output, or the output of its identity consumer if that is
    # a graph output, otherwise null.
    """Returns t or Identity(t) whichever exists in graph outputs else None."""
    for output in tensor.graph.outputs:
      if output is t:
        return t
    # tf.defun adds an Identity for each output, check whether that is the case.
    identity_op = t.consumers()[0]
    if (identity_op.type == "Identity" and
        any(identity_op.outputs[0] is t for t in tensor.graph.outputs)):
      return identity_op.outputs[0]
    return None

  for consumer in tensor.consumers():
    # Find the consumer that is a TensorListPushBack node whose TensorList input
    # is in the list of function inputs.
    if consumer.type != "TensorListPushBack":
      continue

    accum_input_idx = -1
    for accum_input_idx, inp in enumerate(tensor.graph.inputs):
      if inp is consumer.inputs[0]:
        break
    else:
      continue

    output = get_func_graph_output(consumer.outputs[0])
    if output is None:
      # The TensorList output of `consumer` is not in the list of function
      # outputs.
      continue

    for accum_output_idx, out in enumerate(tensor.graph.outputs):
      if out is output:
        if accum_input_idx == accum_output_idx:
          return output
        break

  return None


OptimizedReductionOpsCacheKey = collections.namedtuple(
    "OptimizedReductionOpsCacheKey", [
        "op_type",
        "inputs",
        "dtypes",
        "input_types",
        "name",
        "attrs",
        "op_def",
        "compute_device",
    ])


class _WhileBodyGradFuncGraph(util.WhileBodyFuncGraph):
  # [CLASS DOC]
  # HELPER: Support class for building while-loop gradients.
  # Function graph for the body of the gradient While loop. Subclass of the while-body function graph.
  # It controls capturing of tensors from the forward body graph: resource tensors are captured
  # from the outer graph (must be loop invariants), loop invariants are captured directly, constants
  # are copied, and all other tensors get an accumulator (tensor list) pushed in the forward graph and
  # popped here. Attributes: extra_inputs (initial empty tensor lists for new accumulators; non-empty
  # means the forward op must be rewritten), internal_capture_to_output (mapping from captured
  # placeholder ids to tensors to append to outputs), plus private references to the forward graphs,
  # maximum iterations, forward op and the map of indirect captures. Also moves a few cheap reduction
  # ops (shape, size, rank, tensor-list shape/length) into the forward graph to save memory.
  """FuncGraph for the gradient function of the body of a While op.

  Contains the logic for capturing the tensors from the body of the forward
  While op which is as follows:
  1. If the tensor is of resource type (these are not accumulated):
     a. Ensure that the tensor is a loop invariant, i.e., it exists in both loop
        inputs and outputs at the same index.
     b. Lookup the corresponding resource tensor in the forward outer graph and
        try to capture that.
  2. If the tensor is not of resource type:
     a. Create an accumulator for that tensor and output it from the forward
        pass. Note this also requires adding it as an input to the forward pass.
     b. Capture the accumulator from the forward pass in this FuncGraph. This
        will later be resolved to the correct output of the forward While op.
     c. Pop a value from the captured placeholder and use it as the captured
        value for the forward pass tensor.

  This only allows capturing tensors in the forward graph. A ValueError is
  raised if an attempt is made to capture a tensor not in the forward graph.
  To manually capture a tensor that is not in the forward graph, call `capture`
  with `allowlisted=True`.

  Note: The `captures` dict does not contain the forward tensor since it is not
  directly captured. It contains the accumulator corresponding to this forward
  tensor.

  Attributes:
    while_op_needs_rewrite: True if any non-resource intermediates were
      captured, meaning the forward While op needs to be rewritten to output the
      corresponding accumulators.
    extra_inputs: list of EmptyTensorList tensors to be used as initial input to
    the new accumulators in the forward graph. It may also contain external
    captures of the custom gradient function.
    internal_capture_to_output: dict from a tensor_id(captured placeholder) to
      the corresponding tensor that needs to be added to the list of outputs.
      For instance, when capturing an accumulator TensorList this contains the
      TensorList obtained after popping a tensor from the list. Other entries
      in this dict are expected, though not enforced, to be identities.
      This dict is needed because these output tensors need to be added to
      FuncGraph.outputs "after" the tensors returned from the gradient function.
  """

  def __init__(self, name, forward_cond_graph, forward_body_graph,
               maximum_iterations, forward_while_op, body_graph_inputs,
               body_graph_outputs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the parameters for the sets of forward body inputs and outputs are accepted but never
    # used or stored.
    # Constructor. Inputs: graph name, forward cond graph, forward body graph, maximum iterations tensor,
    # forward While op, and the two unused identity sets. Calls the base constructor with the name,
    # initializes the extra inputs list, the capture-to-output mapping and the indirect captures mapping,
    # and stores the forward references.
    super(_WhileBodyGradFuncGraph, self).__init__(name)
    self.extra_inputs = []
    self.internal_capture_to_output = {}
    # FuncGraph for the body of the forward While op.
    self._forward_graph = forward_body_graph
    # FuncGraph for the cond of the forward While op.
    self._forward_cond_graph = forward_cond_graph
    self._maximum_iterations = maximum_iterations
    self._forward_while_op = forward_while_op
    # Dict from forward intermediate tensor to its indirectly captured tensor
    # in this graph. Indirect capturing happens in two ways:
    # 1. For non-resource tensors we capture their accumulators from the forward
    #    outer graph and pop values from that accumulator inside this graph
    #    using TensorListPopBack.
    # 2. For resource tensors we directly capture their corresponding tensor
    #    in the forward outer graph.
    self._indirect_captures = {}

  @property
  def while_op_needs_rewrite(self):
    # [FUNCTION DOC]
    # KNOWN ISSUE: despite the name suggesting a boolean, this returns the list of extra inputs itself
    # (empty means false, non-empty means true).
    # Property: tells whether the forward While op must be rewritten, i.e. whether accumulators were
    # created. Returns the extra inputs list.
    return self.extra_inputs

  def _create_op_internal(
      self,
      op_type,
      inputs,
      dtypes=None,  # pylint: disable=redefined-outer-name
      input_types=None,
      name=None,
      attrs=None,
      op_def=None,
      compute_device=True):
    # [FUNCTION DOC]
    # Overrides operation creation. For a few cheap reduction-like op types (shape, size, rank, tensor
    # list element shape and length) whose inputs all come from the forward graph and have no accumulator,
    # and when intermediates are not all output, not in an XLA context and not under higher-order tape
    # wrapping, creates the op in the forward graph instead (so the small result is stored rather than the
    # large tensor) via the move helper. Otherwise defers to the base class. Inputs mirror the base
    # class: op type, inputs, dtypes, input types, name, attributes, op definition and whether to compute
    # the device. Returns the created op (or its result).
    # For a reduction op, if op is in the gradient body graph and its input is
    # from the forward graph, moving op to the forward graph means we would
    # store the tensor after the reduction as opposed to the tensor before
    # reduction, and therefore could significantly reduce memory consumption.
    # For now, we do this only for a few ops.
    #
    # We don't do this if any input tensor has already been accumulated. This
    # can happen if we output all intermediates in the forward pass.
    #
    # If in XLA context, do not move constant ops to forward pass as pushing to
    # and popping from a TensorList removes the constant property of an op and
    # breaks XLA compilation, which requires certain inputs to be compile-time
    # constant for certain ops.
    #
    # This optimization is currently also disabled when under a persistent tape,
    # since it leads to an unbounded number of side outputs. With caching it may
    # be possible to re-enable it.
    optimized_reduction_ops = {
        "Shape", "Size", "Rank", "TensorListElementShape", "TensorListLength"
    }
    if (op_type in optimized_reduction_ops and
        not util.output_all_intermediates() and
        all(input.graph is self._forward_graph for input in inputs) and
        all(_get_accumulator(input) is None for input in inputs) and
        not util_v1.GraphOrParentsInXlaContext(self._forward_graph) and
        not util.graph_wrapped_for_higher_order_tape_gradients(
            self._forward_graph)):
      return self._move_op_to_forward_graph(
          op_type,
          inputs,
          dtypes=dtypes,
          input_types=input_types,
          name=name,
          attrs=attrs,
          op_def=op_def,
          compute_device=compute_device)

    return super(_WhileBodyGradFuncGraph, self)._create_op_internal(
        op_type,
        inputs,
        dtypes=dtypes,
        input_types=input_types,
        name=name,
        attrs=attrs,
        op_def=op_def,
        compute_device=compute_device)

  def _move_op_to_forward_graph(
      self,
      op_type,
      inputs,
      dtypes=None,  # pylint: disable=redefined-outer-name
      input_types=None,
      name=None,
      attrs=None,
      op_def=None,
      compute_device=True):
    # [FUNCTION DOC]
    # HELPER: Creates the given op in the forward graph, with caching. Same inputs as the creation
    # override. Looks up a cache stored on the forward graph (created on demand) keyed by the op
    # description; returns the cached result if present. Otherwise creates the op in the forward graph with
    # a name stripped of its trailing slash so it is made unique, stores it in the cache and returns it.
    # Side effects: creates the cache attribute and an op in the forward graph.
    # We have a cache of reduction ops that have already been moved to the
    # forward graph, and we will check it first to avoid moving an op twice.
    if not hasattr(self._forward_graph, "_optimized_reduction_ops_cache"):
      self._forward_graph._optimized_reduction_ops_cache = {}
    cache_key = self._get_optimized_reduction_ops_cache_key(
        op_type, inputs, dtypes, input_types, name, attrs, op_def,
        compute_device)
    cached_op = self._forward_graph._optimized_reduction_ops_cache.get(
        cache_key)
    if cached_op is not None:
      # This op has already been moved to the forward graph and we have it in
      # the cache.
      return cached_op

    with self._forward_graph.as_default():
      # `name` was built using name_scope stack of gradient graph and may not
      # be unique in the forward graph. `Graph.create_op` does not uniquify
      # names which are name scopes i.e. end in `/`. To ensure that the op
      # created gets a unique name in the forward graph we get rid of the
      # trailing slash.
      name = ops.name_from_scope_name(name)
      result = self._forward_graph._create_op_internal(
          op_type,
          inputs,
          dtypes=dtypes,
          input_types=input_types,
          name=name,
          attrs=attrs,
          op_def=op_def,
          compute_device=compute_device)

      # Store the op we just moved to the forward graph so that it does
      # not need to be added there again.
      self._forward_graph._optimized_reduction_ops_cache[cache_key] = result
      return result

  def _get_optimized_reduction_ops_cache_key(
      self,
      op_type,
      inputs,
      dtypes=None,  # pylint: disable=redefined-outer-name
      input_types=None,
      name=None,
      attrs=None,
      op_def=None,
      compute_device=True):
    # [FUNCTION DOC]
    # HELPER: Builds a hashable cache key for the moved-op cache. Inputs are the same as the op creation
    # override. Converts input tensors to references, dtypes and input types to tuples, attributes to a
    # sorted tuple of names and serialized values, and the op definition to serialized bytes. Returns the
    # named tuple key.
    # We need all elements of CacheKey to be hashable.
    inputs = tuple(map(lambda t: t.ref(), inputs))

    if dtypes is not None:
      dtypes = tuple(dtypes)

    if input_types is not None:
      input_types = tuple(input_types)

    if attrs is not None:
      hashable_attrs = []
      for attr_name, attr_value in sorted(attrs.items()):
        hashable_attrs.append((attr_name, attr_value.SerializeToString()))
      attrs = tuple(hashable_attrs)

    if op_def is not None:
      op_def = op_def.SerializeToString()

    return OptimizedReductionOpsCacheKey(op_type, inputs, dtypes, input_types,
                                         name, attrs, op_def, compute_device)

  def _capture_helper(self, tensor, name):
    # [FUNCTION DOC]
    # Overrides tensor capture for the gradient graph as described in the class docs. Inputs: a tensor and
    # a name. Returns an existing indirect capture if there is one. Tensors from other graphs are captured
    # by the base class (recording an identity mapping for new captures). For forward-graph tensors:
    # looks through identity ops, captures loop invariants directly, copies constants, delegates resource
    # tensors to the resource helper, otherwise finds or creates an accumulator in the forward graph (empty
    # tensor list built in the forward outer graph without control inputs, pushed to in the forward body,
    # added to the forward body outputs and captured in the forward cond graph), captures the accumulator
    # and pops a value from it. Side effects: appends to extra inputs, mutates forward graphs, updates
    # the indirect capture and capture-to-output mappings. Returns the captured tensor.
    """Implements the capturing described in the class docstring."""
    captured_tensor = self._indirect_captures.get(ops.tensor_id(tensor))
    if captured_tensor is not None:
      return captured_tensor

    if tensor.graph is not self._forward_graph:
      already_captured = id(tensor) in self.function_captures.by_val_internal
      captured_tensor = super(_WhileBodyGradFuncGraph, self)._capture_helper(
          tensor, name)
      if not already_captured:
        # Adds the captured tensor to the list of outputs so that the input
        # and output signatures match.
        self.internal_capture_to_output[ops.tensor_id(
            captured_tensor)] = captured_tensor
        self._indirect_captures[ops.tensor_id(tensor)] = captured_tensor
      return captured_tensor

    while tensor.op.type == "Identity":
      # We do not accumulate the output of identity nodes so we try to capture
      # the input of the Identity node instead.
      tensor = tensor.op.inputs[0]

    captured_tensor = self._indirect_captures.get(ops.tensor_id(tensor))
    if captured_tensor is not None:
      return captured_tensor

    # No need to accumulate loop invariants. Capture them directly.
    # The captured tensor gets resolved to the corresponding while output in
    # `_resolve_grad_captures`.
    if _is_loop_invariant(tensor, self._forward_graph.inputs,
                          self._forward_graph.outputs):
      captured_tensor = super(_WhileBodyGradFuncGraph,
                              self)._capture_helper(tensor, name)
      # Add to `internal_capture_to_output` so that this gets added to the list
      # of outputs.
      self.internal_capture_to_output[ops.tensor_id(
          captured_tensor)] = captured_tensor
      self._indirect_captures[ops.tensor_id(tensor)] = captured_tensor
      return captured_tensor

    # Do not accumulate Const nodes. Instead copy them directly in the backward
    # graph.
    # TODO(srbs): This just checks for `Const` nodes. Consider checking for
    # graph compile time consts in general.
    # TODO(srbs): Consider making this a loop input.
    if constant_op.is_constant(tensor):
      real_value = constant_op.constant(
          tensor_util.constant_value(tensor), dtype=tensor.dtype)
      self._indirect_captures[ops.tensor_id(tensor)] = real_value
      return real_value

    # Resource tensors are not accumulated and handled specially.
    if tensor.dtype == dtypes.resource:
      return self._resource_capture_helper(tensor)

    # Create or find an existing accumulator output for `tensor` in the forward
    # graph, and fetch from this accumulator in the gradient graph to get the
    # raw intermediate value.
    accumulator = _get_accumulator(tensor)
    if accumulator is None:
      # Create the initial empty tensor list.
      #
      # Note: We clear the control dependencies to avoid a cycle in case a
      # control tensor has an input path to an output of the  forward While.
      #
      # E.g.:
      # x = tf.while_loop(...)
      # y = f(x)
      # with tf.control_dependencies([y]):
      #   tf.gradients(y, x)
      #
      # Since the EmptyTensorList is fed back into the forward While, not
      # removing the control edge would cause a cycle.
      with self._forward_graph.outer_graph.as_default():
        with util.clear_control_inputs():
          tensor_list = list_ops.empty_tensor_list(
              element_dtype=tensor.dtype,
              element_shape=tensor.shape,
              max_num_elements=self._maximum_iterations,
              name=_build_accumulator_name(tensor))
      self.extra_inputs.append(tensor_list)

      # Push the intermediate tensor to the tensor list. This captures
      # `tensor_list`.
      with self._forward_graph.as_default():
        accumulator = list_ops.tensor_list_push_back(tensor_list, tensor)
      # Add the modified tensor list to the list of outputs. This output will be
      # all the accumulated values.
      self._forward_graph.outputs.append(accumulator)

      # Capture in the cond graph as well so the forward cond and body inputs
      # match.
      with self._forward_cond_graph.as_default():
        self._forward_cond_graph.capture(tensor_list)

    # Capture the accumulator tensor list in the gradient graph directly from
    # the forward graph -- we'll later modify this to capture the final list
    # output by the forward While op instead.
    captured_accumulator = super(_WhileBodyGradFuncGraph, self)._capture_helper(
        accumulator, name)

    # Pop the intermediate value from the tensor list in the gradient graph.
    new_tensor_list, captured_tensor = list_ops.tensor_list_pop_back(
        captured_accumulator, element_dtype=tensor.dtype)

    self._indirect_captures[ops.tensor_id(tensor)] = captured_tensor
    self.internal_capture_to_output[ops.tensor_id(
        captured_accumulator)] = new_tensor_list
    return captured_tensor

  def _resource_capture_helper(self, tensor):
    # [FUNCTION DOC]
    # HELPER: Captures a resource tensor, which is never accumulated. Input: the resource tensor from the
    # forward body graph. Finds its loop input index, the matching placeholder and the outer graph tensor,
    # asserts both are resources, requires the output at the same index to refer to the same input (loop
    # invariant), then captures the outer tensor in this graph, records and returns it. Raises an assertion
    # error if the resource is not a loop invariant.
    """Returns the captured resource tensor.

    Resource-type tensors are not accumulated. If a resource tensor exists in
    the loop body it must either be a loop input or an output of a nested While
    op inside the loop body which had captured the external resource.

    Args:
      tensor: the external resource Tensor to be captured.

    Returns:
      Tensor in this graph.
    """
    assert tensor.dtype == dtypes.resource

    forward_graph_input_names = [t.name for t in self._forward_graph.inputs]
    forward_graph_name_to_opdef = {
        op.name: op.node_def for op in self._forward_graph.get_operations()}
    index = util.resource_input_index(
        tensor.name, forward_graph_input_names,
        forward_graph_name_to_opdef,
        self._forward_graph._functions)

    input_placeholder = self._forward_graph.inputs[index]
    tensor_in_outer_graph = self._forward_graph._while.inputs[index]

    assert input_placeholder.dtype == dtypes.resource
    assert tensor_in_outer_graph.dtype == dtypes.resource
    # This must be a loop invariant. However, infrastructure
    # (e.g. tf.vectorized_map) may insert identity nodes, function calls, conds,
    # etc. which take and return the resource tensor unmodified; this means that
    # the Python objects may differ.
    if index != util.resource_input_index(
        self._forward_graph.outputs[index].name, forward_graph_input_names,
        forward_graph_name_to_opdef,
        self._forward_graph._functions):
      raise AssertionError(
          f"Resource tensors must be loop invariants {tensor_in_outer_graph}")

    self._indirect_captures[ops.tensor_id(tensor)] = self.capture(
        tensor_in_outer_graph)
    return self._indirect_captures[ops.tensor_id(tensor)]


def _check_shapes_compat(flat_output_tensors, flat_shape_invariants,
                         flat_input_tensors):
  # [FUNCTION DOC]
  # HELPER: Validation for the loop builder. For each output tensor, shape invariant and input tensor,
  # raises a value error if the output shape is not less than or equal to the invariant, telling the
  # user the variable changed shape across iterations and to use the shape invariants argument.
  for (t, shape, input_t) in zip(flat_output_tensors, flat_shape_invariants,
                                 flat_input_tensors):
    if not control_flow_ops._ShapeLessThanOrEqual(t.shape, shape):
      raise ValueError(
          f"Input tensor `{input_t.name}` enters the loop with shape {shape}, "
          f"but has shape {t.shape} after one iteration. To allow the shape to "
          "vary across iterations, use the `shape_invariants` argument of "
          "tf.while_loop to specify a less-specific shape.")


def _check_num_inputs_outputs(cond_graph, body_graph, num_flattened_loop_vars):
  # [FUNCTION DOC]
  # HELPER: Sanity check for the loop builder. Asserts that cond graph and body graph each have the
  # expected number of inputs, the cond graph has one output, and the body graph has the expected number
  # of outputs. Raises an assertion error with a message otherwise; returns nothing.
  """Checks the number of inputs/outputs of `cond_graph` and `body_graph`."""
  assert len(cond_graph.inputs) == num_flattened_loop_vars, (
      "cond_graph takes %d inputs; Expected: %d" % (len(cond_graph.inputs),
                                                    num_flattened_loop_vars))
  assert len(cond_graph.outputs) == 1, (
      "cond_graph has %d outputs; Expected: 1" % len(cond_graph.outputs))
  assert len(body_graph.inputs) == num_flattened_loop_vars, (
      "body_graph takes %d inputs; Expected: %d" % (len(body_graph.inputs),
                                                    num_flattened_loop_vars))
  assert len(body_graph.outputs) == num_flattened_loop_vars, (
      "body_graph has %d outputs; Expected: %d" % (len(body_graph.outputs),
                                                   num_flattened_loop_vars))


def _check_inputs_outputs_types_match(body_graph, flattened_loop_vars):
  # [FUNCTION DOC]
  # HELPER: Validation for the loop builder. Compares the dtype of each body input with the matching
  # output; raises a type error naming the loop variable if the type changed after one iteration.
  for inp, out, loop_var in zip(body_graph.inputs, body_graph.outputs,
                                flattened_loop_vars):
    if inp.dtype != out.dtype:
      raise TypeError(
          f"Loop var {loop_var.name} enters the loop with type {inp.dtype} "
          f"but has type {out.dtype} after 1 iteration. {loop_var.name} type "
          "should remain constant.")


def _build_cond_placeholders_name_prefix(cond_graph):
  # [FUNCTION DOC]
  # HELPER: Returns a unique name prefix, derived from the cond graph name, used for the placeholders
  # that duplicate body captures in the cond graph.
  return cond_graph.unique_name(cond_graph.name + "___redundant_placeholder")


def _duplicate_body_captures_in_cond(cond_graph, body_graph_captures):
  # [FUNCTION DOC]
  # HELPER: Makes the cond graph signature match the body graph. Input: the cond graph and the tensors
  # captured by the body. Creates one placeholder per body capture (using the low level graph API with a
  # unique name prefix), appends them to the cond graph captures and inputs. Side effects: mutates the
  # cond graph. Returns nothing.
  """Creates placeholders for body captures in cond_graph.

  This is needed to match signatures of cond and body graphs.

  Args:
    cond_graph: cond branch graph
    body_graph_captures: Tensors which were captured when building the
      `body_graph`.
  """
  types = [t.dtype.as_datatype_enum for t in body_graph_captures]
  # TODO(srbs): Providing a unique prefix does not ensure that there is no
  # conflict between the placeholder names and existing nodes in the graph.
  # However passing a list of strings may not be performant.
  # Ideally we should move `Graph.unique_name` to C++ or make
  # `Graph._names_in_use` a trie so that we can find a unique prefix.
  # TODO(b/143286622): This should not be required once captures are separated
  # from regular loop vars.
  with cond_graph._c_graph.get() as c_graph:
    placeholders = c_api.TF_CreatePlaceholders(
        c_graph, types,
        compat.as_str(_build_cond_placeholders_name_prefix(cond_graph)))
  placeholder_ops = [
      ops.Operation._from_c_op(ph.oper, cond_graph) for ph in placeholders
  ]

  tensors = []
  for op in placeholder_ops:
    tensors.append(op.outputs[0])

  # Update `cond_graph._captures` and `cond_graph.inputs` to contain the
  # newly created placeholders.
  tuples = zip(body_graph_captures, tensors)
  keys = [id(t) for t in body_graph_captures]
  for k, v in zip(keys, tuples):
    cond_graph._function_captures.add_or_replace(
        key=k,
        external=v[0],
        internal=v[1],
        is_by_ref=False)
  cond_graph.inputs.extend(tensors)


def _copy_handle_data(src_tensors, tgt_tensors):
  # [FUNCTION DOC]
  # HELPER: Copies resource/variant handle data from each source tensor to the paired target tensor
  # (pairs matched by position). Returns nothing.
  for src_t, tgt_t in zip(src_tensors, tgt_tensors):
    handle_data_util.copy_handle_data(src_t, tgt_t)


def _pack_sequence_as(loop_vars_signature, flat_orig_loop_vars, loop_vars):
  # [FUNCTION DOC]
  # HELPER: Like the structure packing utility but replaces flow tensors with tensor arrays. Inputs:
  # the signature structure, the flattened original loop variables, and the structured or flat loop
  # variables. Each flattened value is replaced by a tensor array with the new flow if the original was a
  # tensor array. Returns the repacked structure.
  """Like `nest.pack_sequence_as` but also replaces flows with TensorArrays."""

  def flow_to_tensor_array(flow, ta):  # pylint: disable=missing-docstring
    # [FUNCTION DOC]
    # HELPER: Nested helper: given a flow tensor and the original value, builds a tensor array with the
    # new flow if the original is a tensor array; otherwise returns the flow unchanged.
    return (tensor_array_ops.build_ta_with_new_flow(ta, flow) if isinstance(  # pylint: disable=g-long-ternary
        ta, tensor_array_ops.TensorArray) else flow)

  flattened_loop_vars = [
      flow_to_tensor_array(*z)
      for z in zip(nest.flatten(loop_vars, expand_composites=True),
                   flat_orig_loop_vars)
  ]
  return nest.pack_sequence_as(loop_vars_signature, flattened_loop_vars,
                               expand_composites=True)


def _tensor_array_to_flow(loop_vars):
  # [FUNCTION DOC]
  # HELPER: Replaces each tensor array in a structure with its flow tensor, leaving other values
  # unchanged. Returns the mapped structure.

  def f(maybe_ta):
    # [FUNCTION DOC]
    # HELPER: Nested mapping function; returns the flow of a tensor array, else the value itself.
    if isinstance(maybe_ta, tensor_array_ops.TensorArray):
      return maybe_ta.flow
    return maybe_ta

  return nest.map_structure(f, loop_vars, expand_composites=True)


def _build_maximum_iterations_loop_var(maximum_iterations):
  # [FUNCTION DOC]
  # HELPER: Converts the optional maximum iterations to an int32 scalar tensor named
  # maximum_iterations; null becomes minus one, meaning unbounded. Returns the tensor.
  if maximum_iterations is None:
    # Default value for max_num_elements to EmptyTensorList meaning that the
    # list size is unbounded.
    maximum_iterations = -1
  # EmptyTensorList expects `max_num_elements` to be of type int32.
  return ops.convert_to_tensor(
      maximum_iterations, dtype=dtypes.int32, name="maximum_iterations")


def _build_accumulator_name(tensor):
  # [FUNCTION DOC]
  # HELPER: Returns the accumulator name for a tensor: its name plus a suffix, with colons replaced by
  # underscores because name scopes disallow colons.
  # Tensor name may be of the form "pow/y:0". Name scope does not allow ":".
  return "{}/accumulator".format(tensor.name).replace(":", "_")


def _is_loop_invariant(tensor, inputs, outputs):
  # [FUNCTION DOC]
  # HELPER: Returns true if the tensor (compared by identity) appears in both the given inputs and
  # outputs, i.e. it is passed through unchanged.
  return (any(tensor is t for t in inputs) and
          any(tensor is t for t in outputs))


def _set_read_only_resource_inputs_attr(op: ops.Operation, branch_graphs):
  # [FUNCTION DOC]
  # HELPER: Sets the attribute listing read-only resource inputs on a While op. Computes the
  # intersection of read-only resource input indices over the branch graphs (starting from all inputs)
  # and stores the sorted indices on the op. Used for automatic control dependencies. Returns nothing.
  """Sets the list of resource inputs which are read-only.

  This is used by AutomaticControlDependencies.

  Args:
    op: While Operation.
    branch_graphs: List of branch FuncGraphs.
  """
  read_only_indices = set(range(len(op.inputs)))
  for branch_graph in branch_graphs:
    if not read_only_indices:
      break
    branch_read_only_indices = acd.get_read_only_resource_input_indices_graph(
        branch_graph)
    read_only_indices = read_only_indices.intersection(branch_read_only_indices)

  ops.set_int_list_attr(op, acd.READ_ONLY_RESOURCE_INPUTS_ATTR,
                        sorted(read_only_indices))


def async_noop(name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring has no Returns section, but it returns the While operation (not an output
  # tensor).
  # Builds an operation that is a no-op executed by an asynchronous kernel, to encourage aggressive
  # inter-op parallelism by scheduling downstream ops on other threads. Input: optional name (default
  # "async_noop"). Builds identity cond and body graphs over a single boolean, creates a stateful While
  # op without a parallel iterations argument, and disables lowering with switch/merge. Returns the
  # While operation.
  """Returns a no-op that is implemented as an async kernel.

  This operation may be useful to implement "aggressive inter-op parallelism"
  because it will cause any immediate downstream operations to be scheduled
  on different threads.

  Args:
    name: The name of the operation.
  """

  with ops.name_scope(name, "async_noop") as name:
    cond_init_value = constant_op.constant(False, name="cond_init_value")

    func_graph_signature = [tensor_spec.TensorSpec(shape=(), dtype=dtypes.bool)]

    cond_graph = func_graph_module.func_graph_from_py_func(
        "cond_graph",
        lambda x: x,
        [cond_init_value],
        {},
        signature=func_graph_signature,
        func_graph=util.WhileCondFuncGraph(
            "cond_graph", collections=ops.get_default_graph()._collections
        ),  # pylint: disable=protected-access
        add_control_dependencies=False,
    )

    body_graph = func_graph_module.func_graph_from_py_func(
        "body_graph",
        lambda x: x,
        [cond_init_value],
        {},
        signature=func_graph_signature,
        func_graph=util.WhileBodyFuncGraph(
            "body_graph", collections=ops.get_default_graph()._collections
        ),  # pylint: disable=protected-access
        add_control_dependencies=False,
    )

    while_op, _ = util.get_op_and_outputs(
        gen_functional_ops._while(
            [cond_init_value],
            util.create_new_tf_function(cond_graph),
            util.create_new_tf_function(body_graph),
            output_shapes=[[]],
            name=name,
        )
    )

    # Disable lowering using switch merge.
    util.maybe_set_lowering_attr(while_op, lower_using_switch_merge=False)

  return while_op

# pylint: enable=protected-access
