# [MODULE DOC]
# Public entry points of the parallel_for package: the sequential for_loop (a while-loop based
# map that stacks per-iteration outputs), the vectorizing pfor (which runs the loop body once
# on a symbolic iteration index and converts the resulting graph through the PFor converter),
# and vectorized_map (exported as the public vectorized_map API, a map over the first
# dimension of tensors built on pfor). Also contains private helpers for composite-tensor
# conversion, detecting pfor_config arguments, XLA context detection, tiling when a parallel
# iteration limit is set, and broadcasting gathers.
# Import-time side effects: only imports and the public-API export registration of
# vectorized_map; defines the module constant naming the pfor_config keyword.
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
"""for_loop and pfor ops."""
# pylint: disable=g-direct-tensorflow-import

import functools

from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.autograph.core import ag_ctx as autograph_ctx
from tensorflow.python.autograph.impl import api as autograph
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_loop
from tensorflow.python.ops.parallel_for.pfor import PFor
from tensorflow.python.ops.parallel_for.pfor import PForConfig
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util import variable_utils
from tensorflow.python.util.tf_export import tf_export


def for_loop(loop_fn, loop_fn_dtypes, iters, parallel_iterations=None):
  # [FUNCTION DOC]
  # Sequential reference implementation of the loop that pfor vectorizes.
  # Inputs: loop_fn, a callable taking a scalar int32 iteration index and returning a nested
  # structure of tensors (entries may be nothing); loop_fn_dtypes, nested structure of dtypes
  # matching the outputs; iters, the iteration count (may be a tensor); parallel_iterations,
  # optional limit passed to the underlying while loop. Builds a while loop whose body (after
  # autograph conversion) calls loop_fn and writes each output into per-output tensor arrays,
  # then stacks them. Entries that are nothing stay nothing. If the loop body never ran (zero
  # iterations) it re-calls loop_fn once with a placeholder index to infer output ranks and
  # returns empty tensors, falling back to a single empty vector if that fails.
  # Returns the stacked outputs packed like loop_fn_dtypes. Raises a value error if loop_fn
  # returns a different number of outputs than there are dtypes.
  """Runs `loop_fn` `iters` times and stacks the outputs.


  Runs `loop_fn` `iters` times, with input values from 0 to `iters - 1`, and
  stacks corresponding outputs of the different runs.

  Args:
    loop_fn: A function that takes an int32 scalar tf.Tensor object representing
      the iteration number, and returns a possibly nested structure of tensor
      objects. The shape of these outputs should not depend on the input.
    loop_fn_dtypes: dtypes for the outputs of `loop_fn`.
    iters: Number of iterations for which to run `loop_fn`.
    parallel_iterations: The number of iterations that can be dispatched in
      parallel. This knob can be used to control the total memory usage.

  Returns:
    Returns a nested structure of stacked output tensor objects with the same
    nested structure as the output of `loop_fn`.
  """

  flat_loop_fn_dtypes = nest.flatten(loop_fn_dtypes)
  is_none_list = []

  def while_body(i, *ta_list):
    # [FUNCTION DOC]
    # HELPER: Loop body used by for_loop. Inputs: the iteration index and one tensor array per
    # output. Calls the autograph-converted loop_fn, validates the output count (value error
    # on mismatch), records in the enclosing list which outputs were nothing (clearing and
    # refilling that list: side effect), writes non-nothing outputs into their arrays at the
    # index, and returns the incremented index followed by the arrays.
    """Body of while loop."""
    fn_conv = autograph.tf_convert(loop_fn, autograph_ctx.control_status_ctx())
    fn_output = nest.flatten(fn_conv(i))
    if len(fn_output) != len(flat_loop_fn_dtypes):
      raise ValueError(
          f"Number of expected outputs {len(flat_loop_fn_dtypes)}, does not "
          f"match the number of actual outputs {len(fn_output)} from loop_fn: "
          f"{loop_fn} with output {fn_output}.")
    outputs = []
    del is_none_list[:]
    is_none_list.extend(x is None for x in fn_output)
    for out, ta in zip(fn_output, ta_list):
      # TODO(agarwal): support returning Operation objects from loop_fn.
      if out is not None:
        # out may be a ref tensor, wrap it in identity to get a non-ref tensor.
        ta = ta.write(i, out)
      outputs.append(ta)
    return tuple([i + 1] + outputs)

  if parallel_iterations is not None:
    extra_args = {"parallel_iterations": parallel_iterations}
  else:
    extra_args = {}
  ta_list = while_loop.while_loop(lambda i, *ta: i < iters, while_body, [0] + [
      tensor_array_ops.TensorArray(dtype.base_dtype, iters)
      for dtype in flat_loop_fn_dtypes
  ], **extra_args)[1:]

  # TODO(rachelim): enable this for sparse tensors

  output = [
      None if is_none else ta.stack()
      for ta, is_none in zip(ta_list, is_none_list)
  ]
  assert len(output) in (0, len(flat_loop_fn_dtypes))
  if not output:
    # This may happen for the case where iters == 0.
    # Pack a list of empty tensors with the proper ranks to match pfor output on 0 iters
    loop_var = array_ops.placeholder_with_default(0, shape=[])
    try:
      loop_fn_out = loop_fn(loop_var)
      out_shapes = [
          [0] + ops.convert_to_tensor(x).shape
          for x in nest.flatten(loop_fn_out)
      ]
      output = [
          array_ops.zeros(out_shapes[i], dt)
          for i, dt in enumerate(flat_loop_fn_dtypes)
      ]
    except Exception:
      output = [array_ops.zeros([0])]
  return nest.pack_sequence_as(loop_fn_dtypes, output)


def _flatten_first_two_dims(x):
  # [FUNCTION DOC]
  # HELPER: Used by the tiled path of the pfor implementation to merge the tile axis and the
  # per-tile iteration axis. Input x: a tensor of rank at least two. Returns it reshaped so the
  # first two dimensions are multiplied into one (dynamic shapes are used).
  """Flattens the first two dimensions of x into a single dimension."""
  old_shape = array_ops.shape(x)
  new_shape = array_ops.concat([[old_shape[0] * old_shape[1]], old_shape[2:]],
                               axis=0)
  return array_ops.reshape(x, new_shape)


PFOR_CONFIG_ARG = "pfor_config"


def _is_under_xla_context():
  # [FUNCTION DOC]
  # HELPER: Lets pfor decide whether to wrap its work in a function. Walks the current default
  # graph and, through outer graphs of function graphs, every nested control-flow context and
  # returns true if any is an XLA compilation context, else false. Reads graph state only.
  """Check if we are currently inside an XLA compile context."""
  g = ops.get_default_graph()
  while g is not None:
    control_flow_context = g._get_control_flow_context()  # pylint: disable=protected-access
    while control_flow_context is not None:
      if control_flow_context.IsXLAContext():
        return True
      else:
        control_flow_context = control_flow_context.outer_context
    # If g is a FuncGraph, get its outer_graph.
    g = getattr(g, "outer_graph", None)
  return False


def pfor(loop_fn,
         iters,
         fallback_to_while_loop=True,
         parallel_iterations=None,
         warn=False):
  # [FUNCTION DOC]
  # Vectorized replacement for running a loop body for each index and stacking outputs.
  # KNOWN ISSUE: the docstring says a value error is raised when parallel_iterations is not an
  # integer greater than one, but the check actually happens in the implementation helper and
  # rejects values below one and equal to one with different messages (the latter pointing to
  # for_loop). Also, the global run-functions-eagerly setting is restored only on normal
  # completion; if building the graph raises, the setting stays overridden.
  # Inputs: loop_fn (callable of an index and optionally the pfor_config keyword), iters
  # (count), fallback_to_while_loop (default true), parallel_iterations (optional chunk size),
  # warn (default false). When executing eagerly or inside an XLA context, the work is wrapped
  # into a tf.function; if functions are globally forced to run eagerly it logs a warning,
  # temporarily turns that off and restores it afterwards. Returns the stacked outputs with the
  # same structure as loop_fn's output.
  """Equivalent to running `loop_fn` `iters` times and stacking the outputs.

  `pfor` has functionality similar to `for_loop`, i.e. running `loop_fn` `iters`
  times, with input from 0 to `iters - 1`, and stacking corresponding output of
  each iteration. However the implementation does not use a `tf.while_loop`.
  Instead it adds new operations to the graph that collectively compute the same
  value as what running `loop_fn` in a loop would compute.


  This is an experimental feature and currently has a lot of limitations:
    - There should be no data dependency between the different iterations. For
      example, a future iteration should not depend on a value or side-effect of
      a previous iteration.
    - Stateful kernels may mostly not be supported since these often imply a
      data dependency or ordering of the iterations. We do support a limited set
      of such stateful kernels though (like RandomFoo, Variable operations like
      reads, etc).
    - Conversion works only on a limited set of kernels for which a converter
      has been registered.
    - `loop_fn` has limited support for control flow operations. `tf.cond` in
      particular is not supported.
    - `loop_fn` should return nested structure of Tensors or Operations. However
      if an Operation is returned, it should have zero outputs.
    - The shape and dtype of `loop_fn` outputs should not depend on the input
      to loop_fn.

  Args:
    loop_fn: A function that takes an int32 scalar tf.Tensor object representing
      the iteration number, and optionally a keyword argument `pfor_config` set
      to a PForConfig object. It returns a possibly nested structure of Tensor
      or Operation objects. Note that if setting `parallel_iterations` argument
      to something other than None, `loop_fn` may be called more than once
      during graph construction. So it may need to avoid mutating global state.
    iters: Number of iterations for which to run `loop_fn`.
    fallback_to_while_loop: If true, on failing to vectorize an operation, pfor
      fallbacks to using a `tf.while_loop` to dispatch the iterations.
    parallel_iterations: A knob to control how many iterations are vectorized
      and dispatched in parallel. The default value of None corresponds to
      vectorizing all the iterations.  If `parallel_iterations` is smaller than
      `iters`, then chunks of at most that many iterations are dispatched in
      sequence. This knob can be used to control the total memory usage.
    warn: Whether or not to warn when falling back to while loops.

  Returns:
    Returns a nested structure of stacked tensor objects with the same nested
    structure as the output of `loop_fn`.
  Raises:
    ValueError: If parallel_iterations is not None and not an integer > 1.
  """
  def f():
    # [FUNCTION DOC]
    # HELPER: Zero-argument closure that calls the pfor implementation with the arguments
    # captured from pfor, so it can optionally be wrapped in a tf.function. Returns its result.
    return _pfor_impl(
        loop_fn,
        iters,
        fallback_to_while_loop=fallback_to_while_loop,
        parallel_iterations=parallel_iterations,
        warn=warn)
  # Note that we wrap into a tf.function if in eager execution mode or under
  # XLA compilation. The latter is so that we don't compile operations like
  # tf.placeholder that are created by the loop body.
  functions_run_eagerly = None
  if context.executing_eagerly() or _is_under_xla_context():
    functions_run_eagerly = def_function.functions_run_eagerly()
    if functions_run_eagerly:
      logging.warning(
          "It looks like tf.function behavior was disabled, perhaps using "
          "tf.config.run_functions_eagerly. Vectorization "
          "primitives (e.g. tf.vectorized_map) require tf.function to work. "
          "These primitives will override the disable.")
      def_function.run_functions_eagerly(False)
    f = def_function.function(f)

  outputs = f()
  if functions_run_eagerly is not None:
    def_function.run_functions_eagerly(functions_run_eagerly)
  return outputs


def _should_expand_composite(value):
  # [FUNCTION DOC]
  # HELPER: Predicate for the composite-tensor conversion helpers. Returns true if value is a
  # composite tensor other than a sparse tensor or indexed slices (those two are handled by
  # the converter itself), else false.
  return (isinstance(value, composite_tensor.CompositeTensor)
          # Leave sparse tensors to be converted by `PFor._convert_sparse`.
          and not isinstance(value, sparse_tensor.SparseTensor)
          and not isinstance(value, indexed_slices.IndexedSlices))


# pylint: disable=protected-access
def _composite_to_tensors(value, is_batched=False):
  # [FUNCTION DOC]
  # HELPER: Flattens loop-body outputs so they can be stacked. Inputs: value, any loop output;
  # is_batched, whether to use the batched tensor-list form (default false). Composite tensors
  # (see the expansion predicate) become a list of tensors via their type spec; others are
  # returned unchanged. Raises a value error if the composite's type spec is not batchable.
  """Converts a CompositeTensor into a list of stackable tensors."""
  if _should_expand_composite(value):
    spec = value._type_spec
    if not isinstance(spec, type_spec.BatchableTypeSpec):
      raise ValueError(f"CompositeTensor instance {value} returned from "
                       "parallel_for or vectorized_map loop body must provide "
                       f"a `BatchableTypeSpec` (saw: {spec}).")
    if is_batched:
      return spec._to_batched_tensor_list(value)
    return spec._to_tensor_list(value)
  return value
# pylint: enable=protected-access


# pylint: disable=protected-access
def _composite_from_tensors(stacked_tensors,
                            preconverted_value,
                            batch_size):
  # [FUNCTION DOC]
  # HELPER: Inverse of the composite flattening. Inputs: stacked_tensors, the stacked tensors
  # for one output; preconverted_value, the original (unflattened) per-iteration output used
  # to learn the structure; batch_size, the number of iterations. For expandable composites
  # rebuilds a composite whose type spec is batched by batch_size; otherwise returns the
  # tensors unchanged.
  """Converts a list of stacked tensors to a batch CompositeTensor."""
  if _should_expand_composite(preconverted_value):
    batch_type_spec = preconverted_value._type_spec._batch(batch_size)
    return batch_type_spec._from_compatible_tensor_list(stacked_tensors)
  return stacked_tensors
# pylint: enable=protected-access


def _loop_fn_has_config(loop_fn):
  # [FUNCTION DOC]
  # HELPER: Determines whether pfor must pass the configuration object to a loop body.
  # Input loop_fn: a plain function, a partial object, or a callable object. Returns true if
  # the argument names include the pfor_config keyword (for partial objects only if it was not
  # already bound by keyword). For callable objects inspects the call method after unwrapping
  # decorators; raises a value error if the object has no call method.
  """Test if `loop_fn` has a `pfor_config` argument."""
  if tf_inspect.isfunction(loop_fn):
    argspec = tf_inspect.getargspec(loop_fn)
    return PFOR_CONFIG_ARG in argspec.args
  elif isinstance(loop_fn, functools.partial):
    fn = loop_fn.func
    argspec = tf_inspect.getargspec(fn)
    return (PFOR_CONFIG_ARG in argspec.args and
            PFOR_CONFIG_ARG not in loop_fn.keywords)
  else:
    loop_class = tf_decorator.unwrap(loop_fn)[1]
    if not hasattr(loop_class, "__call__"):
      raise ValueError("`loop_fn` object did not have a __call__ method")
    argspec = tf_inspect.getargspec(loop_class.__call__)
    return PFOR_CONFIG_ARG in argspec.args


def _pfor_impl(loop_fn,
               iters,
               fallback_to_while_loop,
               parallel_iterations=None,
               pfor_config=None,
               warn=False):
  # [FUNCTION DOC]
  # HELPER: Graph-mode implementation behind pfor (and recursive tiling).
  # KNOWN ISSUE: the warn flag is honored only on the untiled path; when parallel_iterations
  # splits the work, the converter for the leftover iterations is built without it, so
  # warnings follow the converter default. Also, the branch for indexed-slices outputs does
  # the same dense conversion as the other branch; only the logged warning differs.
  # Inputs: loop_fn, iters, fallback_to_while_loop as in pfor; parallel_iterations (optional);
  # pfor_config (shared configuration object, created if the body wants it and none given);
  # warn. Asserts it is not executing eagerly. Runs the body once under a name scope with a
  # placeholder index, converts outputs to tensors, then converts the new ops with the PFor
  # converter. With parallel_iterations: validates it (value error if below one, or equal to
  # one, or if cross-iteration reductions are used), ignores it when iters is smaller, otherwise
  # vectorizes the remainder iterations and runs a sequential loop over tiles of that many
  # iterations, concatenating and restoring static shapes. Returns outputs in loop_fn's
  # structure, rebuilding composite tensors.
  """Implementation of pfor."""
  assert not context.executing_eagerly()
  loop_fn_has_config = _loop_fn_has_config(loop_fn)
  existing_ops = set(ops.get_default_graph().get_operations())
  iters_value = tensor_util.constant_value(iters)
  # Run the loop body
  with ops.name_scope("loop_body"):
    loop_var = array_ops.placeholder_with_default(0, shape=[])
    if loop_fn_has_config:
      if pfor_config is None:
        pfor_config = PForConfig()
        pfor_config._set_iters(iters)  # pylint: disable=protected-access
      loop_fn_outputs = loop_fn(loop_var, **{PFOR_CONFIG_ARG: pfor_config})
    else:
      assert pfor_config is None
      f = autograph.tf_convert(loop_fn, autograph_ctx.control_status_ctx())
      loop_fn_outputs = f(loop_var)
    loop_fn_output_tensors = nest.map_structure(_composite_to_tensors,
                                                loop_fn_outputs)

  # Convert outputs to Tensor if needed.
  tmp_loop_fn_outputs = []
  for loop_fn_output in nest.flatten(loop_fn_output_tensors):
    if (loop_fn_output is not None and not isinstance(
        loop_fn_output,
        (ops.Operation, tensor.Tensor, sparse_tensor.SparseTensor))):
      if isinstance(loop_fn_output, indexed_slices.IndexedSlices):
        logging.warn("Converting %s to a dense representation may make it slow."
                     " Alternatively, output the indices and values of the"
                     " IndexedSlices separately, and handle the vectorized"
                     " outputs directly." % loop_fn_output)
        loop_fn_output = ops.convert_to_tensor(loop_fn_output)
      else:
        loop_fn_output = ops.convert_to_tensor(loop_fn_output)
    tmp_loop_fn_outputs.append(loop_fn_output)
  loop_fn_output_tensors = nest.pack_sequence_as(loop_fn_output_tensors,
                                                 tmp_loop_fn_outputs)

  new_ops = set(ops.get_default_graph().get_operations()) - existing_ops
  iters = ops.convert_to_tensor(iters)
  if parallel_iterations is not None:
    if parallel_iterations < 1:
      raise ValueError(
          "Argument `parallel_iterations` must be None or a positive integer. "
          f"Received: {parallel_iterations}.")
    if parallel_iterations == 1:
      raise ValueError(
          "Found `parallel_iterations == 1`. Use `for_loop` instead.")
    if iters_value is not None and iters_value < parallel_iterations:
      parallel_iterations = None
  if parallel_iterations is None:
    with ops.name_scope("pfor"):
      converter = PFor(
          loop_var,
          iters,
          new_ops,
          fallback_to_while_loop=fallback_to_while_loop,
          pfor_config=pfor_config,
          warn=warn)
      flattened_output_tensors = []
      for loop_fn_output in nest.flatten(loop_fn_output_tensors):
        output = converter.convert(loop_fn_output)
        flattened_output_tensors.append(output)
  else:
    if pfor_config is not None and pfor_config._has_reductions():  # pylint: disable=protected-access
      raise ValueError("Setting `parallel_iterations` currently unsupported if "
                       "reductions across iterations are performed.")
    num_tiled_iterations = iters // parallel_iterations
    num_remaining_iterations = iters % parallel_iterations
    # TODO(agarwal): Avoid calling loop_fn twice. Generate the loop body inside
    # a tf.function and extract the graph from there to vectorize it.
    with ops.name_scope("pfor_untiled"):
      converter = PFor(loop_var, num_remaining_iterations, new_ops,
                       fallback_to_while_loop=fallback_to_while_loop,
                       pfor_config=pfor_config)
      remaining_output_tensors = []
      flattened_output_tensors = nest.flatten(loop_fn_output_tensors)
      for loop_fn_output in flattened_output_tensors:
        output = converter.convert(loop_fn_output)
        remaining_output_tensors.append(output)

    with ops.name_scope("pfor_tiled"):
      loop_fn_dtypes = [ops.convert_to_tensor(x).dtype
                        for x in flattened_output_tensors]

      def tiled_loop_body(j):
        # [FUNCTION DOC]
        # HELPER: Body for the sequential loop over tiles. Input j: the tile index. Computes
        # the starting offset past the remainder iterations and runs a nested pfor
        # conversion over one tile; returns the flattened stacked outputs for that tile.
        offset = j * parallel_iterations + num_remaining_iterations

        def tiled_loop_fn(i, pfor_config=None):
          # [FUNCTION DOC]
          # HELPER: Loop body for one tile. Inputs: the index within the tile and an optional
          # shared configuration. Calls the user loop body with the global index (tile offset
          # added), passing the configuration only if the body accepts it, and returns a flat
          # list of the outputs with composites expanded into tensors.
          if loop_fn_has_config:
            loop_fn_outputs = loop_fn(i + offset, pfor_config=pfor_config)
          else:
            loop_fn_outputs = loop_fn(i + offset)
          return nest.flatten(
              # Stacking across iterations requires explicit Tensors.
              nest.map_structure(_composite_to_tensors, loop_fn_outputs))

        return _pfor_impl(
            tiled_loop_fn,
            parallel_iterations,
            fallback_to_while_loop=fallback_to_while_loop,
            pfor_config=pfor_config)

      tiled_output_tensors = for_loop(
          tiled_loop_body, loop_fn_dtypes,
          num_tiled_iterations, parallel_iterations=1)
      tiled_output_tensors = [
          _flatten_first_two_dims(y) for y in tiled_output_tensors]

    with ops.name_scope("pfor"):
      if iters_value is None or iters_value % parallel_iterations:
        output_tensors = cond.cond(
            math_ops.equal(num_remaining_iterations, 0),
            lambda: tiled_output_tensors,
            lambda: [array_ops.concat([x, y], axis=0)  # pylint: disable=g-long-lambda
                     for x, y in zip(remaining_output_tensors,
                                     tiled_output_tensors)])
      else:
        output_tensors = tiled_output_tensors
      flattened_output_tensors = nest.flatten(output_tensors)

      for output, original_output in zip(flattened_output_tensors,
                                         nest.flatten(loop_fn_output_tensors)):
        # Restore any shape information lost from tiling.
        # TODO(b/174254748): this may not be correct for stacked `variant`s.
        output.set_shape(
            tensor_shape.TensorShape([iters_value]).concatenate(
                original_output.shape))
  return nest.map_structure_up_to(
      loop_fn_outputs,
      functools.partial(_composite_from_tensors, batch_size=iters_value),
      nest.pack_sequence_as(loop_fn_output_tensors,
                            flattened_output_tensors),
      loop_fn_outputs)


def _broadcasting_gather(x, i):
  # [FUNCTION DOC]
  # HELPER: Per-element selection used by vectorized_map. Inputs: x, a tensor whose first
  # dimension is either the batch size or one; i, the index (scalar tensor). If the first
  # dimension is statically one, the index becomes zero; if unknown, it becomes zero at run
  # time when the dimension is not greater than one. Returns the gathered slice.
  """Wrapper for gather that implicitly broadcasts unit dimensions."""
  static_first_dim = tensor_shape.dimension_value(x.shape[0])
  if static_first_dim == 1:
    i = 0
  elif static_first_dim is None:
    i = array_ops.where_v2(array_ops.shape(x)[0] > 1, i, 0)
  result = array_ops.gather(x, i)
  return result


# pylint: disable=protected-access
def _gather_from_tensor_or_composite(x, i):
  # [FUNCTION DOC]
  # HELPER: Gathers the i-th element of a tensor or composite tensor for vectorized_map.
  # Composites are split into batched component tensors, each gathered with unit-dimension
  # broadcasting, and rebuilt with the unbatched type spec; plain tensors are gathered directly.
  """Wrapper for gather that handles CompositeTensors."""
  if _should_expand_composite(x):
    spec = x._type_spec
    gathered_tensors = [_broadcasting_gather(t, i)
                        for t in spec._to_batched_tensor_list(x)]
    return spec._unbatch()._from_compatible_tensor_list(gathered_tensors)
  return _broadcasting_gather(x, i)
# pylint: enable=protected-access


@tf_export("vectorized_map")
def vectorized_map(fn, elems, fallback_to_while_loop=True, warn=True):
  # [FUNCTION DOC]
  # Public API: applies fn to every slice along the first dimension of elems, vectorized.
  # Inputs: fn (callable on one slice structure); elems (tensor or nested structure whose
  # first dimensions broadcast to a common batch size); fallback_to_while_loop (default true);
  # warn (default true, unlike pfor). Converts variables and inputs to tensors, defines a body
  # that gathers the index-th slice (size-one dimensions broadcast) and calls fn, computes the
  # batch size (static maximum, or dynamic maximum if any first dimension is unknown) and
  # delegates to pfor. Returns stacked outputs. Raises a value error if vectorization fails
  # and fallback is disabled.
  """Parallel map on the list of tensors unpacked from `elems` on dimension 0.

  This method works similar to `tf.map_fn` but is optimized to run much faster,
  possibly with a much larger memory footprint. The speedups are obtained by
  vectorization (see [Auto-Vectorizing TensorFlow Graphs: Jacobians,
  Auto-Batching and Beyond](https://arxiv.org/pdf/1903.04243.pdf)). The idea
  behind vectorization is to semantically launch all the invocations of `fn` in
  parallel and fuse corresponding operations across all these invocations. This
  fusion is done statically at graph generation time and the generated code is
  often similar in performance to a manually fused version.

  Because `tf.vectorized_map` fully parallelizes the batch, this method will
  generally be significantly faster than using `tf.map_fn`, especially in eager
  mode. However this is an experimental feature and currently has a lot of
  limitations:
    - There should be no data dependency between the different semantic
      invocations of `fn`, i.e. it should be safe to map the elements of the
      inputs in any order.
    - Stateful kernels may mostly not be supported since these often imply a
      data dependency. We do support a limited set of such stateful kernels
      though (like RandomFoo, Variable operations like reads, etc).
    - `fn` has limited support for control flow operations.
    - `fn` should return nested structure of Tensors or Operations. However
      if an Operation is returned, it should have zero outputs.
    - The shape and dtype of any intermediate or output tensors in the
      computation of `fn` should not depend on the input to `fn`.

  Examples:
  ```python
  def outer_product(a):
    return tf.tensordot(a, a, 0)

  batch_size = 100
  a = tf.ones((batch_size, 32, 32))
  c = tf.vectorized_map(outer_product, a)
  assert c.shape == (batch_size, 32, 32, 32, 32)
  ```

  ```python
  # Computing per-example gradients

  batch_size = 10
  num_features = 32
  layer = tf.keras.layers.Dense(1)

  def model_fn(arg):
    with tf.GradientTape() as g:
      inp, label = arg
      inp = tf.expand_dims(inp, 0)
      label = tf.expand_dims(label, 0)
      prediction = layer(inp)
      loss = tf.nn.l2_loss(label - prediction)
    return g.gradient(loss, (layer.kernel, layer.bias))

  inputs = tf.random.uniform([batch_size, num_features])
  labels = tf.random.uniform([batch_size, 1])
  per_example_gradients = tf.vectorized_map(model_fn, (inputs, labels))
  assert per_example_gradients[0].shape == (batch_size, num_features, 1)
  assert per_example_gradients[1].shape == (batch_size, 1)
  ```

  Args:
    fn: The callable to be performed. It accepts one argument, which will have
      the same (possibly nested) structure as `elems`, and returns a possibly
      nested structure of Tensors and Operations, which may be different than
      the structure of `elems`.
    elems: A tensor or (possibly nested) sequence of tensors, each of which will
      be unpacked along their first dimension. The nested sequence of the
      resulting slices will be mapped over by `fn`. The first dimensions of all
      elements must broadcast to a consistent value; equivalently, each
      element tensor must have first dimension of either `B` or `1`, for some
      common batch size `B >= 1`.
    fallback_to_while_loop: If true, on failing to vectorize an operation,
      the unsupported op is wrapped in a tf.while_loop to execute the map
      iterations. Note that this fallback only happens for unsupported ops and
      other parts of `fn` are still vectorized. If false, on encountering an
      unsupported op, a ValueError is thrown. Note that the fallbacks can result
      in slowdowns since vectorization often yields speedup of one to two orders
      of magnitude.
    warn: If set to `false`, this will supress any warnings due to operation
    conversions in the provided `fn` falling back to while loops.

  Returns:
    A tensor or (possibly nested) sequence of tensors. Each tensor packs the
    results of applying fn to tensors unpacked from elems along the first
    dimension, from first to last.

    Although they are less common as user-visible inputs and outputs, note that
    tensors of type `tf.variant` which represent tensor lists (for example from
    `tf.raw_ops.TensorListFromTensor`) are vectorized by stacking the list
    contents rather than the variant itself, and so the container tensor will
    have a scalar shape when returned rather than the usual stacked shape. This
    improves the performance of control flow gradient vectorization.

  Raises:
    ValueError: If vectorization fails and fallback_to_while_loop is False.
  """
  elems = variable_utils.convert_variables_to_tensors(elems)
  elems = nest.map_structure(ops.convert_to_tensor,
                             elems,
                             expand_composites=True)

  def loop_fn(i):
    # [FUNCTION DOC]
    # HELPER: Loop body handed to pfor by vectorized_map. Input i: the iteration index tensor.
    # Gathers the i-th slice of every element and returns the result of calling fn on that
    # structure.
    gathered_elems = nest.map_structure(
        lambda x: _gather_from_tensor_or_composite(x, i), elems)
    return fn(gathered_elems)

  # Extract batch size from the maximum first dimension of any element.
  flat_elems = nest.flatten(
      nest.map_structure(
          functools.partial(_composite_to_tensors,
                            is_batched=True),
          elems))
  def _get_shape(x):
    # [FUNCTION DOC]
    # HELPER: Static first-dimension probe for vectorized_map. Returns nothing if the tensor
    # has unknown rank; otherwise the first entry of its static shape (which may itself be
    # nothing when that dimension is unknown, and fails for rank-zero tensors).
    if x.shape.rank is None:
      return None
    return x.shape.as_list()[0]
  static_first_dims = [_get_shape(elem) for elem in flat_elems]
  if any(s is None for s in static_first_dims):
    batch_size = math_ops.reduce_max(
        [array_ops.shape(elem)[0] for elem in flat_elems])
  else:
    batch_size = max(static_first_dims)

  return pfor(
      loop_fn,
      batch_size,
      fallback_to_while_loop=fallback_to_while_loop,
      warn=warn)
