# [MODULE DOC]
# Implementation of ConcreteFunction, the callable produced by tracing a tf.function for one
# specific input signature, together with the machinery that supports its gradients.
# Contents: name helpers for generated forward/backward/inference functions; classes that build
# and cache forward/backward function pairs (a delayed-rewrite strategy for graph-mode symbolic
# gradients, and first-order / higher-order tape strategies including forward-mode JVP wrapping);
# a small call-record helper; the ConcreteFunction class itself (calling with flat or structured
# signatures, captures, saved-model export, printing); a garbage collector object that releases
# the function graph; and small utility helpers.
# Relationship: consumed by polymorphic_function and tracing code in the same package; builds on
# atomic_function, function type utilities and the framework function graph.
# Import-time side effects: only defines constants and classes; no registrations.
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
# pylint: disable=unidiomatic-typecheck
"""Implementation for ConcreteFunction."""

import collections

from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.eager import forwardprop_util
from tensorflow.python.eager import record
from tensorflow.python.eager.graph_only_ops import graph_placeholder
from tensorflow.python.eager.polymorphic_function import atomic_function
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.eager.polymorphic_function import saved_model_exported_concrete
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity


def _is_type_subset(a, b):
  # [FUNCTION DOC]
  # HELPER: type compatibility check.
  # KNOWN ISSUE: not referenced anywhere else in this module; appears to be dead code.
  # Inputs: a, an expected spec or any value; b, a candidate spec. If a is a type specification,
  # returns whether its most specific compatible type with b equals a itself (so b is a subset
  # of a); for any other value of a returns true. No side effects.
  """Returns true if `b` is a subset of type `a` (or if a is not a TypeSpec.)"""
  if isinstance(a, type_spec.TypeSpec):
    return a.most_specific_compatible_type(b) == a
  return True


_FORWARD_PREFIX = "__forward_"
_BACKWARD_PREFIX = "__backward_"
_INFERENCE_PREFIX = "__inference_"


def _forward_name(n):
  # [FUNCTION DOC]
  # HELPER: naming utility for generated gradient functions.
  # Input: n, a base function name. Returns a string made of the forward prefix, the base name
  # and a process-unique id, so each generated forward function gets a distinct name.
  """The name of a generated forward defun named n."""
  return "%s%s_%s" % (_FORWARD_PREFIX, n, ops.uid())


def _backward_name(n):
  # [FUNCTION DOC]
  # HELPER: naming utility for generated gradient functions.
  # Input: n, a base function name. Returns a string made of the backward prefix, the base name
  # and a process-unique id.
  """The name of a generated backward defun named n."""
  return "%s%s_%s" % (_BACKWARD_PREFIX, n, ops.uid())


def _inference_name(n):
  # [FUNCTION DOC]
  # HELPER: naming utility for inference (forward without gradient) functions.
  # Input: n, a base function name. Returns a string made of the inference prefix, the base name
  # and a process-unique id.
  """The name of a forward-but-no-gradient defun named n."""
  return "%s%s_%s" % (_INFERENCE_PREFIX, n, ops.uid())


def _create_forward_backward_with_graph(
    attrs, forward_graph, backwards_graph: func_graph_module.FuncGraph
):
  # [FUNCTION DOC]
  # HELPER: shared by the delayed-rewrite and tape gradient function classes.
  # Inputs: attrs, mapping of function attributes of the original function; forward_graph, the
  # function graph of the forward computation (with any side outputs already added);
  # backwards_graph, the function graph of the gradient computation.
  # Behavior: copies the attributes minus the implements marker (generated functions have
  # unstable signatures), links the two functions by attributes naming each other, wraps the
  # backward graph into a ConcreteFunction with an empty-input structured signature and
  # the backward outputs, and turns the forward graph into an atomic function.
  # Returns a pair (forward atomic function, backward concrete function). The forward graph
  # name is used for the forward function's generated name.
  """Creates forward and backward functions from the function graphs."""
  forward_function_name = _forward_name(forward_graph.name)
  common_attributes = dict(attrs)
  # NB: forward and backward function need to drop "_implements".
  # attribute, because their signature contains all the intermediate tensors
  # that they compute. Thus they don't have a stable signature which can
  # be directly optimized downstream.
  # See for more details:
  # https://github.com/tensorflow/community/blob/master/rfcs/20190610-standardizing-composite_ops.md#appendix-future-support-for-optimizing-gradient-functions
  common_attributes.pop(attributes_lib.IMPLEMENTS, None)
  backward_function_attr = attributes_lib.parse_func_attrs(
      {attributes_lib.FORWARD_FUNCTION: forward_function_name})
  backward_function_attr.update(common_attributes)
  # TODO(fmuham): Include inputs as well.
  function_type = function_type_lib.from_structured_signature(
      ((), {}),
      backwards_graph.structured_outputs,
      backwards_graph.function_captures.capture_types,
  )
  backward_function = ConcreteFunction.from_func_graph(
      backwards_graph, function_type, attrs=backward_function_attr
  )
  forward_function_attr = attributes_lib.parse_func_attrs(
      {attributes_lib.BACKWARD_FUNCTION: backward_function.name}
  )
  forward_function_attr.update(common_attributes)
  forward_function = atomic_function.from_func_graph(
      forward_function_name, forward_graph, forward_function_attr
  )
  return forward_function, backward_function


class _DelayedRewriteGradientFunctions(object):
  # [CLASS DOC]
  # HELPER: internal gradient support for ConcreteFunction.
  # Lazily builds and caches pairs of forward and backward functions for symbolic (non-tape)
  # gradients while graph building. The inference function is called as is; only if gradients
  # are requested is the call operation rewritten to a forward call with extra side outputs
  # that the backward function consumes. State: the function graph, the inference atomic
  # function, its attributes, the cached pairs keyed by number of output gradients accepted,
  # the number of user-visible outputs and a graph deleter keeping the graph alive.
  # Does not support forward-mode autodiff or higher-order tape gradients.
  """Caches forward/backward functions with a delayed forward rewrite."""

  def __init__(
      self, atomic_fn: atomic_function.AtomicFunction, func_graph_deleter
  ):
    # [FUNCTION DOC]
    # Stores the inference atomic function, its graph and attributes, creates an empty cache of
    # forward/backward pairs and records how many outputs the graph has right now (the graph
    # is mutated later when side outputs are appended). Inputs: atomic_fn, the inference
    # function; func_graph_deleter, object that keeps the graph alive. No exceptions.
    """Construct an inference function and initialize caches."""
    # A map from the number of forward function outputs with accepted gradients
    # to forward and backward functions, used to cache non-tape backward
    # function generation.
    self._cached_function_pairs = {}
    self._func_graph = atomic_fn.graph
    self._inference_function = atomic_fn
    self._attrs = atomic_fn.attributes
    self._gradient_name = None
    # Note that the FuncGraph is mutated later, so we need to inspect it now to
    # figure out the user-specified outputs of the inference function.
    self._num_inference_outputs = len(self._func_graph.outputs)
    self._func_graph_deleter = func_graph_deleter

  def forward_backward(self, num_doutputs=None):
    # [FUNCTION DOC]
    # Returns a cached (forward function, backward function) pair. Input: num_doutputs, optional
    # count of leading outputs for which gradients are accepted; defaults to the number of
    # user-visible inference outputs. On a cache miss builds the pair and stores it in the
    # cache keyed by that count (side effect: mutates the graph outputs through construction).
    """A possibly-cached pair of forward and backward functions."""
    if num_doutputs is None:
      num_doutputs = self._num_inference_outputs
    forward_backward = self._cached_function_pairs.get(num_doutputs)
    if forward_backward is not None:
      return forward_backward
    forward, backward = self._construct_forward_backward(num_doutputs)
    self._cached_function_pairs[num_doutputs] = (forward, backward)
    return forward, backward

  def _construct_forward_backward(self, num_doutputs):
    # [FUNCTION DOC]
    # HELPER: builds the pair used by forward_backward.
    # Input: num_doutputs, number of leading outputs of the graph for which the backward
    # function takes output gradients. Takes the trainable ones, builds a backward graph that
    # computes gradients of them with respect to the graph inputs, then appends to the forward
    # graph outputs any tensors of the forward graph that the backward graph captured (side
    # outputs) and creates the forward/backward functions. Returns (forward atomic function,
    # backward concrete function). Side effect: mutates the forward graph's outputs list.
    """Constructs a pair of forward and backward functions.

    Args:
      num_doutputs: The constructed backprop function will take output gradients
        for the first `num_doutputs` outputs of the forward function. Defaults
        to the number of outputs for the inference function, but when
        higher-order gradients are computed this will increase to include side
        outputs.

    Returns:
      A pair of (forward_function, backward_function):
        forward_function: A re-generated inference function (an
          AtomicFunction) to account for new side outputs, if any extra
          were required when building the backward pass.
        backward_function: A ConcreteFunction that Takes `num_doutputs`
          arguments and returns gradients with respect to inputs of the forward
          function.
    """
    trainable_outputs = [
        output for output in self._func_graph.outputs[:num_doutputs]
        if backprop_util.IsTrainable(output)]

    signature = []
    for t in trainable_outputs:
      signature.append(
          tensor_lib.TensorSpec(*default_gradient.shape_and_dtype(t)))

    def _backprop_function(*grad_ys):
      # [FUNCTION DOC]
      # HELPER: traced to build the backward graph. Takes the output gradients as positional
      # tensors and returns the gradients of the trainable outputs with respect to the forward
      # graph inputs, computed with no device constraint.
      with ops.device(None):
        return gradients_util._GradientsHelper(  # pylint: disable=protected-access
            trainable_outputs,
            self._func_graph.inputs,
            grad_ys=grad_ys,
            src_graph=self._func_graph)

    with self._func_graph.as_default():
      backwards_graph = func_graph_module.FuncGraph(
          _backward_name(self._func_graph.name))
      func_graph_module.func_graph_from_py_func(
          name=backwards_graph.name,
          python_func=_backprop_function,
          args=[], kwargs={},
          signature=signature,
          func_graph=backwards_graph)
      backwards_graph_captures = backwards_graph.external_captures
      captures_from_forward = [
          c for c in backwards_graph_captures if
          not isinstance(c, ops.EagerTensor) and c.graph is self._func_graph]

      existing_outputs = object_identity.ObjectIdentitySet(
          self._func_graph.outputs)
      for capture in captures_from_forward:
        if capture not in existing_outputs:
          existing_outputs.add(capture)
          self._func_graph.outputs.append(capture)

      forward_function, backward_function = _create_forward_backward_with_graph(
          self._attrs, self._func_graph, backwards_graph)
      return forward_function, backward_function

  def _rewrite_forward_and_call_backward(self, op: ops.Operation, *doutputs):
    # [FUNCTION DOC]
    # Gradient function registered for function call operations during symbolic gradient
    # computation. Inputs: op, the inference call operation; doutputs, gradients with respect
    # to the call's outputs (entries may be nothing). If the backward function has no outputs,
    # returns its structured outputs immediately. Otherwise registers the forward function in
    # the op's graph, rewrites the call op in place to call the forward function (updating the
    # function attribute, output types and extra outputs and copying resource handle data),
    # replaces missing gradients with zeros and sparse gradients with dense tensors, remaps the
    # backward captures to the op's new outputs, and returns the result of calling the backward
    # function. Side effects: mutates the operation and its graph.
    """Add outputs to the forward call and feed them to the grad function."""
    forward_function, backwards_function = self.forward_backward(len(doutputs))
    if not backwards_function.outputs:
      return backwards_function.structured_outputs

    op.graph._add_function_recursive(forward_function)  # pylint: disable=protected-access

    # pylint: disable=protected-access
    # Rewrite an inference call op to be a forward call op
    op._set_func_attr("f", forward_function.name)
    op._set_type_list_attr(
        "Tout",
        [
            o.dtype.as_datatype_enum
            for o in forward_function.function_type.flat_outputs
        ],
    )
    truncated_outputs = forward_function.function_type.flat_outputs[
        len(op.outputs) :
    ]
    op._add_outputs(
        [o.dtype.as_datatype_enum for o in truncated_outputs],
        [o.shape for o in truncated_outputs],
    )
    for i in range(len(op.outputs)):
      output_type = forward_function.function_type.flat_outputs[i]
      handle_data = output_type.dtype._handle_data
      if handle_data:
        handle_data_util.set_handle_data(
            op.outputs[i], handle_data.shape_inference
        )
    # pylint: enable=protected-access

    capture_mapping = dict(
        zip((ops.tensor_id(t) for t in self._func_graph.outputs), op.outputs))
    remapped_captures = [
        capture_mapping.get(ops.tensor_id(capture), capture)
        for capture in backwards_function.captured_inputs
    ]

    # Replace Nones with zeros since we're calling a graph function which
    # expects numeric inputs.
    cleaned_doutputs = []
    for doutput, placeholder in zip(doutputs, self._func_graph.outputs):
      if backprop_util.IsTrainable(placeholder):
        if isinstance(doutput, indexed_slices.IndexedSlices):
          # Gradient passed to a backward ConcreteFunction must be tf.Tensor,
          # so we convert tf.IndexedSlices to tf.Tensor.
          cleaned_doutputs.append(ops.convert_to_tensor(doutput))
        elif doutput is not None:
          cleaned_doutputs.append(doutput)
        else:
          cleaned_doutputs.append(default_gradient.zeros_like(placeholder))

    # Compute the gradients using the side outputs
    return backwards_function._call_flat(  # pylint: disable=protected-access
        cleaned_doutputs, remapped_captures)

  def get_gradient_function(self):
    # [FUNCTION DOC]
    # Takes no inputs; returns the bound gradient callback (the rewrite-and-call-backward method)
    # to be registered as the gradient of function call operations. No side effects.
    """Returns gradient function.

    The gradient rewrites an inference call op to a forward call op, but does
    not modify a pre-existing forward call op. It then computes the gradient
    from the output's gradients and the side outputs of the forward op.
    """
    return self._rewrite_forward_and_call_backward

  def forward(self, inference_args=None, input_tangents=None):
    # [FUNCTION DOC]
    # Returns the plain inference atomic function (only user-specified outputs). Inputs:
    # inference_args, ignored (kept for interface compatibility with the tape class);
    # input_tangents, optional list of tangents; must be empty or absent.
    # Raises an internal error if non-empty tangents are supplied because forward-mode autodiff
    # is unsupported by this class.
    """A forward function with only user-specified outputs.

    The call operation for the returned inference function can be rewritten into
    a forward function. This only happens if the backward function (from the
    `backward` method) ends up being used to compute gradients.

    This approach avoids constructing unnecessary graphs, but it only works if
    we are calling this function when not executing eagerly.

    Args:
      inference_args: A flat list of Tensors, arguments to the inference
        function. Unused, but taken for compatibility with
        _TapeGradientFunctions.
      input_tangents: A flat list of Tensors, jvps associated with
        `inference_args`. Unused; if required, tape functions must be used
        instead.

    Returns:
      An atomic_function.AtomicFunction.
    """
    del inference_args  # unused
    if input_tangents:
      # This class does not support special-cased forwardprop. The arguments are
      # here for compatibility with _TapeGradientFunctions.
      raise errors.InternalError("unexpectedly got forwardprop information in "
                                 "a class that does not support forwardprop.")
    return self._inference_function

  def _backward(self, outputs):
    # [FUNCTION DOC]
    # HELPER: used by record. Input: outputs, list of output tensors of the forward call.
    # Returns a pair of a backward callback and the outputs to record; the callback takes
    # gradients and applies the rewrite-and-call-backward logic to the operation that produced
    # the first output.
    """Fetch a backward function for `outputs` from the forward function."""
    def _backward_function(*args):
      # [FUNCTION DOC]
      # Backward callback for tape recording: takes gradient tensors and returns the input
      # gradients by rewriting the call op that produced the first forward output and calling
      # the backward function.
      call_op = outputs[0].op
      return self._rewrite_forward_and_call_backward(call_op, *args)
    return _backward_function, outputs

  def record(self, flat_outputs, inference_args, input_tangents):
    # [FUNCTION DOC]
    # Records the function call on the active gradient tapes. Inputs: flat_outputs, flat list of
    # call results; inference_args, flat list of input tensors; input_tangents, flat list of
    # input tangents. Returns nothing. Side effect: registers the operation under the inference
    # function's name with the tape, with inputs being arguments plus tangents and a backward
    # callback built from the outputs.
    """Record the function call operation.

    _DelayedRewriteGradientFunctions supports only first-order backprop tape
    gradients (and then only when graph building). It does not work with
    higher-order tape gradients or forward autodiff, but does work with
    higher-order symbolic gradients (tf.gradients).

    Args:
      flat_outputs: The result of running `forward`.
      inference_args: A flat list of Tensors with inference inputs to the
        operation.
      input_tangents: A flat list of Tensors with input tangents consumed by the
        operation.
    """
    backward_function, to_record = self._backward(flat_outputs)
    record.record_operation(
        self._inference_function.cached_definition.signature.name,
        to_record,
        inference_args + input_tangents,
        backward_function,
    )


# Contains information about a forward function wrapped to compute jvps.
_ForwardWrapper = collections.namedtuple(
    "_ForwardWrapper", (
        # The wrapper Graph.
        "graph",
        # A flat list of non-tangent Tensor outputs from the wrapped forward
        # function.
        "outputs",
        # Indices for output tangents, same format as
        # forwardprop_util.pack_tangents.
        "output_indices",
        # A flat list of tangents for `outputs`.
        "output_tangents"))


class _TapeGradientFunctions(object):
  # [CLASS DOC]
  # HELPER: internal gradient support for ConcreteFunction (abstract base class).
  # Builds and caches a forward function with side outputs plus a backward function with fixed
  # signature, suitable for recording on eager or graph-building gradient tapes, and optionally
  # wraps them to compute forward-mode JVPs inline. State: the function graph, attributes,
  # graph deleter, forwardprop input indices, the delayed-rewrite helper, output counts, the
  # cached forward/backward functions and forwardprop output indices. Subclasses implement
  # _forward_and_backward_functions and differ in how many side outputs get gradients.
  """Caches forward and backward functions compatible with eager gradients.

  In contrast to the delayed-rewrite approach in
  `_DelayedRewriteGradientFunctions` which only works with delayed execution,
  the forward function generated by this class has a fixed set of outputs which
  may be preserved by a tape in order to compute gradients later.

  This class is abstract; its child classes differ in how many side outputs of
  the forward function their backward function accepts gradients for, which
  determines whether higher-order tape gradients are possible.
  """

  def __init__(
      self,
      func_graph: func_graph_module.FuncGraph,
      attrs,
      func_graph_deleter,
      forwardprop_input_indices,
      delayed_rewrite_functions,
      need_gradients_for_jvps,
  ):
    # [FUNCTION DOC]
    # Stores the given function graph, attributes, graph deleter, forwardprop input indices
    # (list of lists of pairs of input index and jvp index), delayed-rewrite helper and
    # a flag saying whether gradients of JVPs are needed, and initializes empty caches for the
    # forward graph, forward and backward functions. Counts the graph's current outputs and how
    # many of them are trainable. No exceptions.
    self._func_graph = func_graph
    self._forward_graph = None
    self._attrs = attrs
    self._forward = None
    self._backward = None
    self._num_outputs = len(func_graph.outputs)
    self._func_graph_deleter = func_graph_deleter
    self._forwardprop_input_indices = forwardprop_input_indices
    self._forwardprop_output_indices = None
    self._num_forwardprop_outputs = 0
    self._num_inference_outputs = len(func_graph.outputs)
    self._num_trainable_inference_outputs = len(
        [t for t in func_graph.outputs if backprop_util.IsTrainable(t)])
    self._delayed_rewrite_functions = delayed_rewrite_functions
    self._need_gradients_for_jvps = need_gradients_for_jvps

  def _build_functions_for_outputs(
      self, outputs, inference_args, input_tangents):
    # [FUNCTION DOC]
    # HELPER: core builder used by subclasses. Inputs: outputs, tensors of the forward graph
    # whose gradients the backward function accepts; inference_args, flat input tensors;
    # input_tangents, flat tangent tensors (empty when no forwardprop). Builds a backward graph
    # with a placeholder per trainable output, differentiates to the forward graph inputs
    # (densifying sparse gradients when tangents exist), appends captured forward tensors to the
    # forward graph outputs, and creates the functions. Without tangents returns (forward
    # function, forward graph, backward function, nothing, zero). With tangents also wraps the
    # forward function with JVP computation and the backward function with JVP backprop,
    # reorders forward outputs, and returns the wrapped functions, the wrapper graph, output
    # indices and the number of output tangents. Raises an internal error if the wrapper
    # graph input count disagrees with inference args plus tangents. Mutates the forward graph.
    """Forward+backward functions where the backward function sees `outputs`."""
    # First figure out which of `outputs` are trainable. We'll accept gradients
    # for each of these in the backward function.
    trainable_outputs = []
    trainable_indices = []
    for index, output in enumerate(outputs):

      if backprop_util.IsTrainable(output):
        trainable_outputs.append(output)
        trainable_indices.append(index)

    backwards_graph = func_graph_module.FuncGraph(
        _backward_name(self._func_graph.name))
    with backwards_graph.as_default():
      gradients_wrt_outputs = []
      for output in trainable_outputs:
        gradient_shape, gradient_dtype = default_gradient.shape_and_dtype(
            output)
        gradient_placeholder = graph_placeholder(gradient_dtype, gradient_shape)
        handle_data_util.copy_handle_data(output, gradient_placeholder)
        gradients_wrt_outputs.append(gradient_placeholder)
      with ops.device(None):
        gradients_wrt_inputs = gradients_util._GradientsHelper(  # pylint: disable=protected-access
            trainable_outputs,
            self._func_graph.inputs,
            grad_ys=gradients_wrt_outputs,
            src_graph=self._func_graph)

      if input_tangents:
        # Convert IndexedSlices to dense tensors (as we do elsewhere for
        # function gradients). Our C++ bindings don't know how to handle them
        # currently.
        gradients_wrt_inputs = nest.map_structure(
            lambda x: ops.convert_to_tensor(x) if x is not None else None,
            gradients_wrt_inputs)
      captures_from_forward = [
          c for c in backwards_graph.external_captures
          if not isinstance(c, ops.EagerTensor) and c.graph is self._func_graph
      ]
      existing_outputs = object_identity.ObjectIdentitySet(
          self._func_graph.outputs)
      for capture in captures_from_forward:
        if capture not in existing_outputs:
          existing_outputs.add(capture)
          self._func_graph.outputs.append(capture)

    # The ordering of `backwards_graph.inputs` is important: inputs of
    # `backward_function` correspond to outputs (including
    # side outputs) of `self._tape_forward_function`.
    backwards_graph.inputs = (
        gradients_wrt_outputs + backwards_graph.internal_captures)
    backwards_graph.outputs.extend(
        grad
        for grad in nest.flatten(gradients_wrt_inputs, expand_composites=True)
        if grad is not None)
    backwards_graph.structured_outputs = gradients_wrt_inputs

    forward_function, backward_function = _create_forward_backward_with_graph(
        self._attrs, self._func_graph, backwards_graph)

    if not input_tangents:
      # There is no need to special-case forwardprop, so we can return the
      # forward+backward pair we've created without further wrapping.
      return (forward_function, self._func_graph, backward_function,
              # No forwardprop outputs.
              None, 0)
    forward_wrapper = self._wrap_forward_function_with_jvps(
        forward_function, backward_function, inference_args, input_tangents)
    (wrapped_backwards_graph,
     forward_wrapper) = self._wrap_backward_function_with_jvp_backprop(
         backward_function, gradients_wrt_outputs, forward_wrapper)
    # Now that we've added new captures, we need to make sure forward outputs
    # are in the same order the backward function expects them to be in:
    # [inference outputs] + [jvps] + [side outputs] + [captures].
    forward_wrapper = self._shuffle_forward_outputs(forward_wrapper)
    (wrapped_forward_function,
     wrapped_backward_function) = _create_forward_backward_with_graph(
         self._attrs, forward_wrapper.graph, wrapped_backwards_graph)
    if (len(inference_args) + len(input_tangents)
        != len(forward_wrapper.graph.inputs)):
      raise errors.InternalError(
          f"The forward graph had {len(forward_wrapper.graph.inputs)} inputs, "
          f"but we expected {len(inference_args) + len(input_tangents)} "
          f"({len(inference_args)} inference inputs and "
          f"{len(input_tangents)} input tangents).")
    return (wrapped_forward_function, forward_wrapper.graph,
            wrapped_backward_function, forward_wrapper.output_indices,
            len(forward_wrapper.output_tangents))

  def _wrap_forward_function_with_jvps(
      self, forward_function, backward_function,
      inference_args, input_tangents):
    # [FUNCTION DOC]
    # HELPER: part of the forwardprop support. Inputs: forward_function and backward_function,
    # the plain pair; inference_args and input_tangents, flat tensors. Builds a new graph with
    # placeholders for each original input (preserving captures) and for each tangent, records
    # them as identity-like operations, calls the forward function under forward accumulators
    # (with its gradient overridden by the delayed rewrite function), registers a forwardprop-only
    # record using the wrapped backward, and packs output JVPs. Returns a forward wrapper record
    # of graph, outputs, output indices and output tangents. Raises an internal error if jvp
    # indices do not line up with the graph inputs, and a shape compatibility error if a tangent
    # shape mismatches its placeholder. Side effects: pushes and pops forwardprop state and
    # records operations on tapes.
    """Adds inline JVP computation to a forward function."""
    forward_wrapper_graph = func_graph_module.FuncGraph(
        _forward_name(self._func_graph.name))
    with forward_wrapper_graph.as_default():
      # Tell forward accumulators to free up space for new JVP computations,
      # since one may be in the process of computing a JVP (if that computation
      # triggered this function building).
      #
      # We'll make symbolic versions of input JVPs, run the forward function
      # under forward accumulators to get symbolic output JVPs, then set those
      # as outputs of the new wrapped forward function.
      with forwardprop_util.push_forwardprop_state():
        forward_captures = {
            ops.tensor_id(internal): external
            for external, internal in self._func_graph.captures}
        for input_index, real_input in enumerate(self._func_graph.inputs):
          # This loop is more or less equivalent to running tf.identity on each
          # of self._func_graph.inputs. However, doing that also captures jvps
          # for resource handles, which confuses the jvp capturing code below
          # (since primal inputs are interwoven with jvp inputs).
          input_placeholder = array_ops.placeholder(
              dtype=real_input.dtype,
              shape=real_input.shape)
          capture = forward_captures.get(ops.tensor_id(real_input))
          if capture is not None:
            forward_wrapper_graph.add_capture(capture, input_placeholder)
            if capture.dtype == dtypes.resource:
              handle_data_util.copy_handle_data(capture, input_placeholder)
          else:
            forward_wrapper_graph.inputs.append(input_placeholder)
        for inp, arg in zip(forward_wrapper_graph.inputs, inference_args):
          record.record_operation(
              "captured_value", [inp], [arg],
              backward_function=lambda x: [x],
              forward_function=lambda x: [x])
        num_inference_inputs = len(inference_args)
        for tape_indices in self._forwardprop_input_indices:
          for input_index, jvp_index in tape_indices:
            input_placeholder = forward_wrapper_graph.inputs[input_index]
            if len(forward_wrapper_graph.inputs) != jvp_index:
              raise errors.InternalError(
                  f"Expected {jvp_index} forward graph inputs, "
                  f"got {len(forward_wrapper_graph.inputs)}.")
            gradient_shape, gradient_dtype = default_gradient.shape_and_dtype(
                input_placeholder)
            jvp_placeholder = graph_placeholder(gradient_dtype, gradient_shape)
            external_jvp = input_tangents[jvp_index - num_inference_inputs]
            forward_wrapper_graph.add_capture(external_jvp, jvp_placeholder)
            tensor_shape.TensorShape(
                external_jvp.shape).assert_is_compatible_with(
                    jvp_placeholder.shape)
            record.record_operation(
                "captured_value",
                [jvp_placeholder],
                [external_jvp],
                backward_function=lambda x: [x],
                forward_function=lambda x: [x])
        forward_inputs = forward_wrapper_graph.inputs[:num_inference_inputs]
        gradient_function = (
            self._delayed_rewrite_functions._rewrite_forward_and_call_backward)  # pylint: disable=protected-access
        with ops.get_default_graph()._override_gradient_function(  # pylint: disable=protected-access
            {"PartitionedCall": gradient_function,
             "StatefulPartitionedCall": gradient_function}):
          forward_outputs = forward_function.call_flat(*forward_inputs)
          if isinstance(forward_outputs, ops.Operation):
            # _wrapped_backward_function expects a list, but if the function has
            # no outputs its call() returns an Operation. We need to undo that
            # so we don't cause problems later.
            forward_outputs = []
        py_backward, _ = self._wrap_backward_function(
            self._func_graph, backward_function, forward_outputs)
      # We will never request backward tape gradients for this operation
      # directly since we're wrapping the call; forwardprop will call the
      # backward function (and nested forward accumulators may build
      # higher-order gradients), but any watching GradientTapes should ignore
      # it.
      #
      # TODO(allenl): It might be better to explicitly stop backward recording
      # so we don't use the second-order tape cases unnecessarily.
      record.record_operation_forwardprop_only(
          forward_function.cached_definition.signature.name,
          forward_outputs, forward_inputs, py_backward, None)
      output_indices, output_tangents = (
          pywrap_tfe.TFE_Py_PackJVPs(forward_outputs))
      output_tangents = [forward_wrapper_graph.capture(t)
                         for t in output_tangents]
    return _ForwardWrapper(
        graph=forward_wrapper_graph, outputs=forward_outputs,
        output_indices=output_indices, output_tangents=output_tangents)

  def _wrap_backward_function_with_jvp_backprop(
      self, backward_function, gradients_wrt_outputs, forward_wrapper):
    # [FUNCTION DOC]
    # HELPER: part of the forwardprop support. Inputs: backward_function, the plain backward
    # function; gradients_wrt_outputs, placeholders for trainable output gradients;
    # forward_wrapper, the record from wrapping the forward function. Builds a new backward graph
    # that backpropagates through the wrapped computation, and, if gradients for JVPs are
    # needed, loops differentiating output tangents with respect to the wrapper graph inputs and
    # aggregating with the first set of input gradients until no new captures appear (extending
    # forward outputs and tangents). Returns a pair: the wrapped backward graph and the updated
    # forward wrapper. Mutates the forward wrapper outputs list.
    """Wraps `backward_function` to include gradients for JVPs."""
    wrapped_backwards_graph = func_graph_module.FuncGraph(
        _backward_name(self._func_graph.name))
    with wrapped_backwards_graph.as_default():
      py_backward, recorded_outputs = self._wrap_backward_function(
          self._func_graph, backward_function, forward_wrapper.outputs)
      trainable_index = 0
      forward_doutputs = []
      doutput_args = []
      for output in recorded_outputs:
        if backprop_util.IsTrainable(output):
          doutput = gradients_wrt_outputs[trainable_index]
          doutput_placeholder = graph_placeholder(doutput.dtype, doutput.shape)
          doutput_args.append(doutput_placeholder)
          forward_doutputs.append(doutput_placeholder)
          trainable_index += 1
        else:
          doutput_args.append(None)

      dinputs = py_backward(*doutput_args)
      existing_outputs = object_identity.ObjectIdentitySet(
          forward_wrapper.outputs + forward_wrapper.output_tangents)
      num_processed_output_tangents = 0
      gradients_wrt_output_tangents = []
      tangent_doutputs = []
      output_tangents = forward_wrapper.output_tangents
      output_indices = forward_wrapper.output_indices
      if self._need_gradients_for_jvps:
        # TODO(allenl): Consider using a throwaway graph to avoid extra gradient
        # evaluations; gradients for jvps may have common subgraphs.
        while num_processed_output_tangents != len(output_tangents):
          for output in output_tangents[num_processed_output_tangents:]:
            gradient_shape, gradient_dtype = default_gradient.shape_and_dtype(
                output)
            placeholder = graph_placeholder(gradient_dtype, gradient_shape)
            gradients_wrt_output_tangents.append(placeholder)
            tangent_doutputs.append(placeholder)
          num_processed_output_tangents = len(output_tangents)
          with ops.device(None):
            gradients_wrt_inputs = gradients_util._GradientsHelper(  # pylint: disable=protected-access
                output_tangents,
                forward_wrapper.graph.inputs,
                grad_ys=gradients_wrt_output_tangents,
                src_graph=forward_wrapper.graph)
          dinputs = [
              backprop_util.AggregateIndexedSlicesGradients((existing, new))
              for existing, new in zip(dinputs, gradients_wrt_inputs)
              if existing is not None or new is not None]
          dinputs.extend(gradients_wrt_inputs[len(dinputs):])
          captures_from_forward = [
              c for c in wrapped_backwards_graph.external_captures
              if (not isinstance(c, ops.EagerTensor)
                  and c.graph is forward_wrapper.graph)]
          for capture in captures_from_forward:
            if capture not in existing_outputs:
              existing_outputs.add(capture)
              forward_wrapper.outputs.append(capture)
          output_indices, output_tangents = (
              forwardprop_util.pack_tangents(forward_wrapper.outputs))
          output_tangents = [forward_wrapper.graph.capture(t)
                             for t in output_tangents]
          for t in output_tangents:
            existing_outputs.add(t)
    wrapped_backwards_graph.inputs = (
        forward_doutputs[:self._num_trainable_inference_outputs]
        + tangent_doutputs
        + forward_doutputs[self._num_trainable_inference_outputs:]
        + wrapped_backwards_graph.internal_captures)
    wrapped_backwards_graph.structured_outputs = dinputs
    wrapped_backwards_graph.outputs = [t for t in dinputs if t is not None]
    return (wrapped_backwards_graph,
            forward_wrapper._replace(output_indices=output_indices,
                                     output_tangents=output_tangents))

  def _shuffle_forward_outputs(self, forward_wrapper):
    # [FUNCTION DOC]
    # HELPER: part of the forwardprop support. Input: forward_wrapper record. Sets the wrapper
    # graph outputs to the order inference outputs, then JVPs, then side outputs and captures,
    # and returns the wrapper record with output indices remapped to that order.
    # Side effect: assigns the graph's outputs.
    """Reorders function outputs so captures are last."""
    def _index_map(original):
      # [FUNCTION DOC]
      # HELPER: maps one original output index to its index after reordering. Indices of
      # inference outputs stay, indices beyond the recorded outputs shift to follow the
      # inference outputs, and other indices are offset by the number of output tangents.
      if original < self._num_inference_outputs:
        return original
      if original >= len(forward_wrapper.outputs):
        return (original - len(forward_wrapper.outputs)
                + self._num_inference_outputs)
      return original + len(forward_wrapper.output_tangents)
    output_indices = nest.map_structure(
        _index_map, forward_wrapper.output_indices)
    forward_wrapper.graph.outputs = (
        forward_wrapper.outputs[:self._num_inference_outputs]
        + forward_wrapper.output_tangents
        + forward_wrapper.outputs[self._num_inference_outputs:])
    return forward_wrapper._replace(output_indices=output_indices)

  def forward(self, inference_args, input_tangents):
    # [FUNCTION DOC]
    # Returns the forward atomic function with side outputs, building it on first use via the
    # subclass hook and caching the forward function, forward graph, backward function and
    # forwardprop output info. Inputs: inference_args and input_tangents, flat tensor lists.
    # Later calls ignore their arguments and return the cached function.
    """Construct or fetch a forward function with side-outputs.

    When graph building without a tape active, symbolic gradients rely on
    regenerating the backward function for higher-order gradients (to account
    for new side outputs of the rewritten forward function call). Thus there is
    no fixed backward function for this case. However, when a tape is active
    (eager or graph building), we generate fixed backward and forward functions
    at forward function call time.

    This difference between the tape and non-tape cases is to avoid building
    unneeded backward functions while graph building (where we may or may not
    eventually need gradients).

    Args:
      inference_args: A flat list of Tensors, arguments to the inference
        function.
      input_tangents: A flat list of Tensors, jvps associated with
        `inference_args`.

    Returns:
      A forward atomic_function.AtomicFunction.
    """
    if self._forward is None:
      (
          self._forward,
          self._forward_graph,
          self._backward,
          self._forwardprop_output_indices,
          self._num_forwardprop_outputs,
      ) = self._forward_and_backward_functions(inference_args, input_tangents)
    return self._forward

  def _wrap_backward_function(
      self, forward_graph: func_graph_module.FuncGraph, backward, outputs
  ):
    # [FUNCTION DOC]
    # HELPER: used by record and the forwardprop wrappers.
    # Inputs: forward_graph, the function graph of the forward function; backward, the backward
    # concrete function; outputs, the actual output tensors of a call of the forward function.
    # Remaps backward captures that are forward graph outputs to the real outputs; raises an
    # internal error if any capture is still a symbolic tensor of the forward graph. Chooses
    # which outputs are relevant (dropping forwardprop outputs when JVP gradients are not
    # needed), remembers non-trainable positions to skip, and precomputes zeros for variant
    # outputs. Returns a pair: a wrapper callback and the list of outputs to record. The
    # callback takes output gradients (nothing entries allowed), converts sparse gradients to
    # dense, substitutes zeros for missing ones, and calls the backward function; if the
    # backward function has no outputs it returns its structured outputs.
    """Create a backward function given `outputs` from the forward function."""
    capture_mapping = dict(
        zip((ops.tensor_id(t) for t in forward_graph.outputs), outputs)
    )
    captured_inputs = backward.captured_inputs
    remapped_captures = [
        capture_mapping.get(ops.tensor_id(capture), capture)
        for capture in captured_inputs
    ]
    if any(
        t.graph is forward_graph
        for t in remapped_captures
        if not isinstance(t, ops.EagerTensor)
    ):
      incorrect_mapping = [
          t
          for t in remapped_captures
          if (
              not isinstance(t, ops.EagerTensor)
              and t.graph is not forward_graph
          )
      ]
      raise errors.InternalError(
          "Failed to map all backward graph captures to "
          "the forward graph. Incorrectly mapped: "
          f"{incorrect_mapping}."
      )
    # We may need to use zeros_like to get a zero for variant Tensors with
    # unconnected gradients. We do that in advance so we don't have to hold on
    # to the outputs themselves, which may not be needed otherwise.
    variant_zeros_like = {}
    backward_function_inputs = len(backward.inputs) - len(captured_inputs)
    recorded_outputs = []
    trainable_recorded_outputs = 0
    skip_positions = []
    if self._num_forwardprop_outputs and not self._need_gradients_for_jvps:
      relevant_outputs = (
          outputs[: self._num_inference_outputs]
          + outputs[
              self._num_inference_outputs + self._num_forwardprop_outputs :
          ]
      )
    else:
      relevant_outputs = outputs
    for output_index, output in enumerate(relevant_outputs):
      if trainable_recorded_outputs < backward_function_inputs:
        recorded_outputs.append(output)
      if backprop_util.IsTrainable(output):
        trainable_recorded_outputs += 1
      else:
        skip_positions.append(output_index)
      if output.dtype == dtypes.variant:
        variant_zeros_like[output_index] = default_gradient.zeros_like(output)

    def _backward_function_wrapper(*args):
      # [FUNCTION DOC]
      # Callback returned by the enclosing method: takes positional output gradients (tensors,
      # sparse gradients or nothing), densifies sparse ones, skips non-trainable positions,
      # replaces missing gradients by zeros (variant zeros when the placeholder is variant),
      # stops once enough gradients are collected, and returns the result of calling the backward
      # function with the remapped captures. Returns the backward function's structured
      # outputs directly when it produces no outputs.
      """Process output gradients and call the backward function."""
      if not backward.outputs:
        return backward.structured_outputs

      processed_args = []
      input_index = 0
      for output_index, arg in enumerate(args):
        # Convert IndexedSlices to dense tensors. The IndexedSlices optimization
        # is only really effective when doing tf.gather(variable) as the
        # adjoint functions for most operations are unlikely to preserve the
        # sparsity in IndexedSlices.
        if isinstance(arg, indexed_slices.IndexedSlices):
          arg = ops.convert_to_tensor(arg)
        if output_index in skip_positions:
          continue
        if arg is None:
          # We're calling a (non-polymorphic) ConcreteFunction, so we need to
          # have a Tensor value for each Tensor we thought would be trainable
          # based on its dtype, even if it ended up being unconnected.
          input_placeholder = backward.inputs[
              input_index]
          if input_placeholder.dtype == dtypes.variant:
            arg = variant_zeros_like[output_index]
          else:
            arg = array_ops.zeros(
                *default_gradient.shape_and_dtype(input_placeholder))
        processed_args.append(arg)
        input_index += 1
        if input_index >= backward_function_inputs:
          break
      return backward._call_flat(  # pylint: disable=protected-access
          processed_args, remapped_captures)

    return _backward_function_wrapper, recorded_outputs

  def record(self, flat_outputs, inference_args, input_tangents):
    # [FUNCTION DOC]
    # Records a call of the forward function on the active tapes. Inputs: flat_outputs, results
    # of the forward call; inference_args and input_tangents, flat tensor lists. Builds the
    # backward callback and outputs to record. With forwardprop output indices, records two
    # operations: a backprop-only one on the arguments and a forwardprop-only one on arguments
    # plus tangents; otherwise records a single combined operation. Returns nothing.
    """Record the function call operation.

    For backprop, indicates the backward function to use and which new Tensors
    must be watched. For forwardprop from eager, the function call itself will
    have produced tangents which need to be recorded.

    Args:
      flat_outputs: The result of running `forward`.
      inference_args: A flat list of Tensors with inference inputs to the
        operation.
      input_tangents: A flat list of Tensors with input tangents consumed by the
        operation.
    """
    backward_function, to_record = self._wrap_backward_function(
        self._forward_graph, self._backward, flat_outputs
    )
    if self._forwardprop_output_indices:
      record.record_operation_backprop_only(
          self._forward.cached_definition.signature.name,
          to_record,
          inference_args,
          backward_function,
      )
      record.record_operation_forwardprop_only(
          self._forward.cached_definition.signature.name,
          flat_outputs,
          inference_args + input_tangents,
          backward_function,
          self._forwardprop_output_indices,
      )
    else:
      record.record_operation(
          self._forward.cached_definition.signature.name,
          to_record,
          inference_args + input_tangents,
          backward_function,
      )


class _FirstOrderTapeGradientFunctions(_TapeGradientFunctions):
  # [CLASS DOC]
  # HELPER: internal gradient support for ConcreteFunction.
  # Tape-gradient function cache for the common case where only first-order gradients are
  # needed: the backward function accepts gradients only for the real outputs, not side
  # outputs, which avoids extra graph building. Created by ConcreteFunction when a single
  # non-persistent tape is watching.
  """Caches tape-friendly functions for first-order gradients."""

  def __init__(
      self,
      func_graph: func_graph_module.FuncGraph,
      attrs,
      func_graph_deleter,
      forwardprop_input_indices,
      delayed_rewrite_functions,
      need_gradients_for_jvps,
  ):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the graph deleter and forwardprop input indices are assigned again after the
    # base constructor has already stored them, so the two assignments are redundant.
    # Passes all arguments (function graph, attributes, graph deleter, forwardprop input
    # indices, delayed-rewrite helper, JVP gradient flag) to the base constructor and then
    # re-stores the deleter and indices. Returns nothing.
    super().__init__(
        func_graph,
        attrs,
        func_graph_deleter,
        forwardprop_input_indices,
        delayed_rewrite_functions,
        need_gradients_for_jvps,
    )
    self._func_graph_deleter = func_graph_deleter
    self._forwardprop_input_indices = forwardprop_input_indices

  def _forward_and_backward_functions(self, inference_args, input_tangents):
    # [FUNCTION DOC]
    # Builds the forward and backward functions for first-order gradients. Inputs:
    # inference_args and input_tangents, flat tensor lists. Uses only the original inference
    # outputs of the graph as the outputs the backward function takes gradients for, and
    # delegates to the shared builder. Returns the same five-element tuple as the builder
    # (forward function, forward graph, backward function, forwardprop output indices, number
    # of output tangents).
    """Shortcut for when only first-order gradients are required.

    The returned backward function does not accept gradients with respect to
    side output of forward_function. This is fine as long as the user can't
    possibly request second order tape gradients, as when they've used a single
    non-persistent GradientTape. Since we don't need the backward function to
    take gradients with respect to side outputs, we can skip some potentially
    slow graph building.

    Args:
      inference_args: A flat list of Tensors, arguments to the inference
        function.
      input_tangents: A flat list of Tensors, jvps associated with
        `inference_args`.

    Returns:
      A tuple of (forward_function, backward_function):
        forward_function: Takes the same inputs as the inference function, but
          returns side outputs used by backward_function in addition to the
          inference function's outputs.
        backward_function: Takes side outputs from forward_function and
          gradients with respect to the "real" outputs of forward_function and
          returns gradients with respect to the inputs.
    """
    outputs = self._func_graph.outputs[:self._num_inference_outputs]
    return self._build_functions_for_outputs(
        outputs, inference_args, input_tangents)


class _HigherOrderTapeGradientFunctions(_TapeGradientFunctions):
  # [CLASS DOC]
  # HELPER: internal gradient support for ConcreteFunction.
  # Tape-gradient function cache for higher-order gradients: the backward function accepts
  # gradients for all outputs of the forward function including side outputs, so gradients of
  # gradients can be taken. Used when a persistent tape or nested tapes are involved.
  """Caches tape-friendly functions for higher-order gradients."""

  # TODO(b/136189779): Cond/while under a tape may need similar logic. Consider
  # generalizing if so.
  def _forward_and_backward_functions(self, inference_args, input_tangents):
    # [FUNCTION DOC]
    # Builds forward and backward functions suitable for higher-order gradients. Inputs:
    # inference_args and input_tangents, flat tensor lists. Repeatedly builds throwaway pairs
    # until the set of forward graph outputs (side outputs added by gradient construction)
    # stops growing with trainable tensors, logging a warning every fifth iteration after
    # twenty iterations that this is slow and may indicate a gradient registration bug. Then
    # builds the final pair. Returns the five-element tuple of the shared builder. Raises an
    # internal error if building the final pair added new trainable outputs. Side effect:
    # mutates the graph outputs and emits warnings.
    """Forward and backward functions suitable for higher-order gradients.

    Unlike in `_FirstOrderTapeGradientFunctions`, the backward function built by
    this method accepts gradients for all of the outputs of the returned forward
    function, including side outputs.

    Args:
      inference_args: A flat list of Tensors, arguments to the inference
        function.
      input_tangents: A flat list of Tensors, jvps associated with
        `inference_args`.

    Returns:
      A tuple of (forward_function, backward_function):
        forward_function: Takes the same inputs as the inference function, but
          returns side outputs used by backward_function in addition to the
          inference function's outputs.
        backward_function: Takes side outputs from forward_function and
          gradients with respect to all of its outputs, real and side. Returns
          gradients with respect to the inputs.
    """
    outputs = []
    iteration_count = 0
    # First we need to figure out how many side outputs from the forward pass
    # will be required. We do this in a temporary graph to avoid actually
    # running multiple copies of the backward pass (one per _GradientsHelper
    # call).
    #
    # While computing gradients, the backward function captures Tensors from
    # the forward function. We add these as side outputs of the original
    # function. However, we then need to accept output gradients with respect
    # to these side outputs for higher order gradients to work. Thus we loop
    # until the number of outputs of the function stabilizes. Note that this
    # is only required for tape gradients, where we need to declare in advance
    # all of the forward op's outputs: symbolic gradients with tf.gradients
    # instead rely on regenerating backward functions when higher-order
    # gradients are requested.
    while (len(outputs) < len(self._func_graph.outputs)
           # It's possible for gradient generation to add new ops to the forward
           # pass. If all of the new outputs are non-trainable, there's no
           # reason to continue.
           and any(backprop_util.IsTrainable(output)
                   for output in self._func_graph.outputs[len(outputs):])):
      iteration_count += 1
      if iteration_count >= 20 and iteration_count % 5 == 0:
        new_op_with_trainable_output = None
        num_new_trainable_outputs = 0
        for output in self._func_graph.outputs[len(outputs):]:
          if backprop_util.IsTrainable(output):
            num_new_trainable_outputs += 1
            new_op_with_trainable_output = output.op
        logging.warning(
            ("Determining side outputs for the function '{}' is taking longer "
             "than expected ({} iterations, typically this converges in 5 or "
             "so). This could indicate that a gradient registration is adding "
             "new ops to the forward pass every time gradients are generated. "
             "{} new trainable output(s) were added this iteration, one from "
             "the following op:\n {}\nThis may indicate a TensorFlow bug, or "
             "an issue in a tf.custom_gradient.")
            .format(
                self._func_graph.name, iteration_count,
                num_new_trainable_outputs, new_op_with_trainable_output))
      outputs = list(self._func_graph.outputs)
      self._build_functions_for_outputs(
          outputs, inference_args, input_tangents)

    (forward_function, forward_graph,
     backward_function, output_indices, num_output_tangents) = (
         self._build_functions_for_outputs(
             outputs, inference_args, input_tangents))
    if (len(self._func_graph.outputs) > len(outputs)
        and any(backprop_util.IsTrainable(output)
                for output in self._func_graph.outputs[len(outputs):])):
      raise errors.InternalError(
          "Unexpectedly added new outputs to the forward function when "
          "building the backward function: "
          f"{self._func_graph.outputs[len(outputs):]}.")
    return (forward_function, forward_graph, backward_function, output_indices,
            num_output_tangents)


class _ForwardBackwardCall(object):
  # [CLASS DOC]
  # HELPER: internal glue between ConcreteFunction calls and gradient support.
  # Holds, between executing a call and recording it on tapes, the gradient-functions object
  # that produces forward/backward functions, the flat inference arguments, the input tangents
  # and a flag saying whether any tape is watching. Uses slots to stay lightweight.
  """Holds the state of a function call between execution and recording."""

  __slots__ = [
      "_functions", "_inference_args", "_input_tangents", "_tape_watching"
  ]

  def __init__(self, functions, inference_args, input_tangents, tape_watching):
    # [FUNCTION DOC]
    # Stores the four inputs unchanged: the gradient-functions object, flat inference argument
    # list, flat input tangent list and the boolean tape-watching flag. Returns nothing.
    """Collects information about the function call.

    Args:
      functions: An object which produces forward and backward functions, either
        a _DelayedRewriteGradientFunctions or a _TapeGradientFunctions object.
      inference_args: A flat list of Tensors, arguments to the inference
        function.
      input_tangents: A flat list of Tensors, jvps associated with
        `inference_args`.
      tape_watching: Boolean, with True indicating that recording is necessary.
    """
    self._functions = functions
    self._inference_args = inference_args
    self._input_tangents = input_tangents
    self._tape_watching = tape_watching

  def forward(self):
    # [FUNCTION DOC]
    # Asks the gradient-functions object for the forward function for this call. Takes no
    # inputs. Returns a pair: the forward function and the combined list of inference arguments
    # followed by input tangents to call it with.
    """Builds or retrieves a forward function for this call."""
    forward_function = self._functions.forward(
        self._inference_args, self._input_tangents
    )
    return forward_function, self._inference_args + self._input_tangents

  def record(self, flat_outputs):
    # [FUNCTION DOC]
    # Records the executed call on tapes. Input: flat_outputs, results of running the forward
    # function (may be an operation or nothing for functions without outputs). Only when a tape
    # is watching and there are real outputs does it forward them with the stored arguments
    # and tangents to the gradient-functions object. Returns nothing.
    """Given outputs from the execution of `forward`, records the operation."""
    if (
        self._tape_watching
        and not isinstance(flat_outputs, ops.Operation)
        and flat_outputs is not None
    ):
      # We only record function calls which have outputs, and then only when a
      # tape is watching.
      self._functions.record(
          flat_outputs, self._inference_args, self._input_tangents
      )


class ConcreteFunction(core.ConcreteFunction, trackable.Trackable):
  # [CLASS DOC]
  # A callable graph function traced for one specific input signature, created from a
  # tf.function or reconstructed from an atomic function. Wraps an atomic inference function
  # and its function graph, the external captures, the function type (structured signature),
  # output shapes, attributes and caches of gradient function helpers (delayed-rewrite, first
  # and higher order tape). Supports calling with the original structured signature or the
  # legacy flat tensor signature, taking gradients under tapes, adding itself to graphs,
  # printing signatures, and exporting/serialization for saved models (it is trackable).
  # Optionally owns its function graph via a garbage collector that breaks reference cycles.
  """A `tf.types.experimental.ConcreteFunction` created from `tf.function`."""

  def __init__(
      self, atomic_fn: atomic_function.AtomicFunction, shared_func_graph=True
  ):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it raises a value error when the number of placeholders
    # differs from the number of inputs, but no such check exists in this constructor.
    # Inputs: atomic_fn, the inference atomic function that forms the forward pass;
    # shared_func_graph, optional boolean, default true. When false, creates a garbage collector
    # that owns the graph and breaks its reference cycles when this object is deleted.
    # Takes the graph, function type and attributes from the atomic function, gathers captured
    # inputs (external plus deferred), records output shapes, creates the delayed-rewrite
    # gradient helper and empty caches for tape helpers, and caches the inference function.
    # The flat-signature fields (argument keywords and positional count) start empty and are
    # assigned after construction by the caller.
    """Initialize a `ConcreteFunction`.

    Args:
     atomic_fn: Inference atomic function to form basis of forward pass.
     shared_func_graph: If False, the ConcreteFunction takes ownership of
       `func_graph` and will break reference cycles when it is deleted. This
       makes the FuncGraph inoperable.

    Raises:
      ValueError: If number of input_placeholders is not equal to the number
        of function inputs.
    """
    # _arg_keywords and _num_positional_args define the flat signature.  They
    # are assigned after construction.
    self._arg_keywords = None
    self._num_positional_args = None

    self._func_graph = atomic_fn.graph
    self._captured_inputs = (
        self._func_graph.external_captures
        + self._func_graph.deferred_external_captures
    )
    self._function_type = atomic_fn.function_type

    self._output_shapes = tuple(
        output.shape for output in self._func_graph.outputs)
    self._attrs = attributes_lib.parse_func_attrs(
        atomic_fn.attributes or {}
    )

    if shared_func_graph:
      self._garbage_collector = None
    else:
      self._garbage_collector = ConcreteFunctionGarbageCollector(
          atomic_fn.graph
      )

    # Pairs of forward and backward functions used for computing gradients.
    #
    # These each get a reference to the FuncGraph deleter since they use the
    # FuncGraph directly.
    self._delayed_rewrite_functions = _DelayedRewriteGradientFunctions(
        atomic_fn, self._garbage_collector)
    self._first_order_tape_functions = {}
    self._higher_order_tape_functions = {}
    # Cache the inference function to avoid a (Python) function call when not
    # building gradients.
    self._inference_function = self._delayed_rewrite_functions.forward()

  @classmethod
  def from_func_graph(cls, graph, function_type, attrs, shared_func_graph=True):
    # [FUNCTION DOC]
    # Alternate constructor. Inputs: graph, a function graph; function_type, the structured
    # signature (may be nothing); attrs, mapping of function attributes; shared_func_graph,
    # optional boolean default true (see the constructor). Builds an atomic inference function
    # named with the inference prefix and the graph's name, and returns a new concrete function
    # wrapping it.
    atomic_fn = atomic_function.from_func_graph(
        _inference_name(graph.name), graph, attrs, function_type
    )
    return ConcreteFunction(atomic_fn, shared_func_graph=shared_func_graph)

  @property
  def function_type(self):
    # [FUNCTION DOC]
    # Property returning the structured function type (signature) stored at construction; may
    # be nothing for legacy flat functions. No side effects.
    """Return the FunctionType associated with this ConcreteFunction."""
    return self._function_type

  @property
  def inference_fn(self):
    # [FUNCTION DOC]
    # Property returning the cached inference atomic function (the forward pass without
    # gradient side outputs). No side effects.
    """Return the inference function associated with this ConcreteFunction."""
    return self._inference_function

  # TODO(fmuham): Remove this property.
  @property
  def _function_spec(self):
    # [FUNCTION DOC]
    # HELPER: temporary compatibility property (marked for removal).
    # Returns nothing when there is no function type; otherwise builds a function spec from the
    # function type, using the defaults of the optional parameters, no input-signature
    # flag set, and this function's name. No side effects.
    if self.function_type is None:
      return None

    return function_type_utils.FunctionSpec(
        self.function_type,
        {
            p.default
            for p in self.function_type.parameters.values()
            if p.optional
        },
        False,
        name=self.name,
    )

  @property
  def variables(self):
    # [FUNCTION DOC]
    # Property returning an immutable sequence of the variables recorded on the function graph.
    """Sequence of variables for this function."""
    return tuple(self._func_graph.variables)

  def set_variables(self, variables):
    # [FUNCTION DOC]
    # Replaces the variables recorded on the function graph with the given collection.
    # Input: variables, a collection of variables. Returns nothing. Mutates the function graph.
    self._func_graph.variables = variables

  @property
  def trainable_variables(self):
    # [FUNCTION DOC]
    # Property returning an immutable sequence of the trainable variables of the function graph.
    """Sequence of trainable variables for this function."""
    return tuple(self._func_graph.trainable_variables)

  def __call__(self, *args, **kwargs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring lists an assertion error for functions not created through
    # get_concrete_function, but this method never raises one; it just delegates.
    # Calls the function with positional and keyword arguments matching either the original
    # structured signature or the flat tensor signature, and returns the function's result
    # (structured outputs). Raises a type error if arguments match neither signature. Delegates
    # entirely to the internal call implementation.
    """Executes the wrapped function.

    ConcreteFunctions have two signatures:

    * The signature of the original function wrapped by this ConcreteFunction.
    * A flat signature, where each argument accepts a single Tensor.

    The original function signature is generally preferred, but the flat input
    signature is supported for backward compatibility.

    ### Original Function Signature

    When calling a ConcreteFunction with the signature of the original function,
    each argument must match the type or value that was used when the
    ConcreteFunction's graph was traced.  In particular:

    * Tensor arguments (including CompositeTensors, such as RaggedTensor) must
      have matching `TypeSpec`s.
    * Non-Tensor arguments (such as booleans or ints) must have equal values.
    * Nested arguments (such as lists, tuples, or dictionaries) must have the
      same nesting structure; and each nested value must have a matching type
      or value.

    The default value for any arguments that were traced with non-Tensor values
    is the value that was used in the trace.  Arguments that were traced with
    tensor arguments do not have a default value (even if the original function
    had a default value for that argument).

    ### Flat Signature

    When calling a ConcreteFunction with the flat signature, the arguments
    correspond to the flattened component tensors of the arguments that were
    used to construct the ConcreteFunction.  Parameter names are assigned based
    on `TensorSpec.name` (when specified) or the original argument names (with
    suffixes automatically added for nested arguments or composite tensors with
    multiple components).

    Args:
      *args: Positional arguments to the concrete function.
      **kwargs: Keyword arguments to the concrete function.

    Returns:
      The result of applying the TF function on the given Tensors.

    Raises:
      AssertionError: If this `ConcreteFunction` was not created through
        `get_concrete_function`.
      TypeError: If the arguments do not match the function's signature.
    """
    return self._call_impl(args, kwargs)

  def _call_impl(self, args, kwargs):
    # [FUNCTION DOC]
    # Implements calling. Inputs: args, tuple of positional arguments; kwargs, mapping of
    # keyword arguments. Inside a profiler trace scope named after the graph, tries the
    # structured signature first when a function type exists; if that raises a type error, falls
    # back to the flat signature, and if both fail raises a type error whose message combines
    # both failure messages. Without a function type uses only the flat signature. Returns the
    # call result.
    """See `__call__` for details."""
    with trace.Trace(self._func_graph.name, tf_function_call="concrete"):
      # Construct the list of input tensors: check if the structured signature
      # applies first; and if not, then use the flat signature.
      if self.function_type is not None:
        try:
          return self._call_with_structured_signature(args, kwargs)
        except TypeError as structured_err:
          try:
            return self._call_with_flat_signature(args, kwargs)
          except (TypeError, ValueError) as flat_err:
            raise TypeError(  # pylint: disable=raise-missing-from
                str(structured_err)
                + "\nFallback to flat signature also failed due to: "
                + str(flat_err)
            )

      return self._call_with_flat_signature(args, kwargs)

  def _call_with_flat_signature(self, args, kwargs):
    # [FUNCTION DOC]
    # HELPER: legacy calling path used by the call implementation.
    # Inputs: args, positional arguments; kwargs, keyword arguments (names are sanitized).
    # Fills missing positional slots from the flat argument keywords, then verifies there are
    # no extra or duplicated keywords and that every argument is a tensor or resource variable.
    # Raises a type error for too many positional arguments, missing required arguments,
    # duplicate values, unexpected keywords, or non-tensor arguments. Returns the result of the
    # flat call with the captured inputs.
    """Executes the wrapped function with the flat signature.

    Args:
      args: Positional arguments to the concrete function.
      kwargs: Keyword arguments to the concrete function.

    Returns:
      The result of applying the function on the Tensors/Variables contained in
      `args` and `kwargs`.
    Raises:
      TypeError: if `args` and `kwargs` do not match the flat signature of this
        `ConcreteFunction`.
    """
    if len(args) > self._num_positional_args:
      raise TypeError(
          f"{self._flat_signature_summary()} takes {self._num_positional_args} "
          f"positional arguments, got {len(args)}.")
    args = list(args)
    kwargs = dict(kwargs)
    kwargs = {
        function_type_lib.sanitize_arg_name(k): v for k, v in kwargs.items()
    }
    for keyword in self._arg_keywords[len(args):]:
      try:
        args.append(
            kwargs.pop(
                function_type_lib.sanitize_arg_name(compat.as_str(keyword))))
      except KeyError:
        specified_keywords = (
            list(self._arg_keywords[:len(args)]) + list(kwargs.keys()))
        missing_required_args = sorted(
            set(self._arg_keywords) - set(specified_keywords))
        raise TypeError(f"{self._flat_signature_summary()} missing required "
                        f"arguments: {', '.join(missing_required_args)}.")
    if kwargs:
      positional_arg_keywords = set(self._arg_keywords[:len(args)])
      for unused_key in kwargs:
        if unused_key in positional_arg_keywords:
          raise TypeError(f"{self._flat_signature_summary()} got two values "
                          f"for '{unused_key}'.")
      raise TypeError(f"{self._flat_signature_summary()} got unexpected "
                      f"keyword arguments: {', '.join(sorted(kwargs))}.")

    for i, arg in enumerate(args):
      if not isinstance(
          arg, (tensor_lib.Tensor, resource_variable_ops.BaseResourceVariable)):
        raise TypeError(f"{self._flat_signature_summary()}: expected argument "
                        f"#{i}(zero-based) to be a Tensor; "
                        f"got {type(arg).__name__} ({arg}).")
    return self._call_flat(args, self.captured_inputs)

  def _call_with_structured_signature(self, args, kwargs):
    # [FUNCTION DOC]
    # HELPER: preferred calling path used by the call implementation.
    # Inputs: args and kwargs, the caller's positional and keyword arguments. Canonicalizes them
    # against the function type, flattens them to tensors and calls the flat implementation;
    # the result is packed into the structured outputs. Raises a type error if arguments do
    # not match the structured signature.
    """Executes the wrapped function with the structured signature.

    Args:
      args: Positional arguments to the concrete function.
      kwargs: Keyword arguments to the concrete function.

    Returns:
      The result of applying the function on the Tensors/Variables contained in
      `args` and `kwargs`.
    Raises:
      TypeError: if `args` and `kwargs` do not match the structured signature
        of this `ConcreteFunction`.
    """
    bound_args = (
        function_type_utils.canonicalize_function_inputs(
            args, kwargs, self.function_type)
    )
    filtered_flat_args = self.function_type.unpack_inputs(bound_args)
    return self._call_flat(
        filtered_flat_args,
        captured_inputs=self.captured_inputs)

  def _call_flat(self, tensor_inputs, captured_inputs):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it raises a value error for arguments that are not
    # tensors or variables and mentions an args name; the code never checks types. The value
    # error it actually raises is for a graph-building input whose shape is incompatible with
    # the traced input shape.
    # Core execution path. Inputs: tensor_inputs, list of tensors (or variables) for the
    # function's flat inputs; captured_inputs, list of captured tensors appended after them.
    # Behavior: propagates the unsaveable status of this function's graph to the current graph
    # being built, marks variables as accessed when tapes or variable watchers might record,
    # and (only when not executing eagerly) checks input shapes against the traced placeholders,
    # skipping resource and variant dtypes. If eager and no tape can record, runs the inference
    # function directly. Otherwise picks forward/backward functions for the gradient situation,
    # calls the forward function (in graph mode overriding the gradient of partitioned call
    # operations with the delayed rewrite gradient), records the call on tapes, and returns the
    # outputs packed into the function's structured output form.
    """Executes the wrapped function.

    Args:
      tensor_inputs: a list of only Tensors generated from args, kwargs.
      captured_inputs: the captured inputs that are also part of the input args
        to the actual execution. By default, it should be self._captured_inputs.
    Returns:
      The result of applying the TF function to `args`.

    Raises:
      ValueError: If `args` contains anything other than Tensors or Variables.
    """
    ctx = context.context()
    executing_eagerly = ctx.executing_eagerly()

    # Copy saveable status of function's graph to current FuncGraph.
    default_graph = ops.get_default_graph()
    if default_graph.building_function and not self._func_graph.saveable:
      default_graph.mark_as_unsaveable(self._func_graph.saving_errors)

    if (record.could_possibly_record() or
        hasattr(default_graph, "watch_variable")):
      for v in self._func_graph.variables:
        resource_variable_ops.variable_accessed(v)

    # TODO(fmuham): check in eager mode too.
    if not executing_eagerly:
      for i, tensor_input in enumerate(tensor_inputs):
        # Can not compare shapes in these cases
        # TODO(b/216506654): Consider moving this check elsewhere and making it
        # work for all types (e.g. by including shape for Variables).
        if (tensor_input.dtype == dtypes.resource or
            tensor_input.dtype == dtypes.variant):
          continue

        # If we're graph building, shape inference is on. We check for input
        # compatibility up front to avoid hard to debug incompatibilities
        # later.
        graph_input_shape = tensor_shape.TensorShape(
            self._func_graph.inputs[i].shape)
        if not graph_input_shape.is_compatible_with(tensor_input.shape):
          raise ValueError(
              f"Tensor {tensor_input} is not compatible with the shape this "
              f"function was traced with. Expected shape "
              f"{self._func_graph.inputs[i].shape}, but got shape "
              f"{tensor_input.shape}.\n\nIf you called get_concrete_function, "
              f"you may need to pass a tf.TensorSpec(..., shape=...) with a "
              f"less specific shape, having None on axes which can vary.")

    args = tensor_inputs + captured_inputs
    possible_gradient_type = gradients_util.PossibleTapeGradientTypes(args)
    if (possible_gradient_type == gradients_util.POSSIBLE_GRADIENT_TYPES_NONE
        and executing_eagerly):
      # No tape is watching; skip to running the function.
      return self._inference_function.call_preflattened(args)
    forward_backward = self._select_forward_and_backward_functions(
        args,
        possible_gradient_type,
        executing_eagerly)
    forward_function, args_with_tangents = forward_backward.forward()
    if executing_eagerly:
      flat_outputs = forward_function.call_flat(*args_with_tangents)
    else:
      with default_graph._override_gradient_function(  # pylint: disable=protected-access
          {"PartitionedCall": self._get_gradient_function(),
           "StatefulPartitionedCall": self._get_gradient_function()}):
        flat_outputs = forward_function.call_flat(*args_with_tangents)
    forward_backward.record(flat_outputs)
    return self.function_type.pack_output(flat_outputs)

  @property
  def name(self):
    # [FUNCTION DOC]
    # Property returning the name (a string) of the inference function's definition.
    """`ConcreteFunction` name."""
    return self._delayed_rewrite_functions.forward().name

  @property
  def graph(self):
    # [FUNCTION DOC]
    # Property returning the function graph this concrete function was built from.
    """Returns the graph from which this function was constructed."""
    return self._func_graph

  @property
  def inputs(self):
    # [FUNCTION DOC]
    # Property returning the list of placeholder tensors of the function graph that correspond
    # to the function arguments (including captures' placeholders).
    """Returns tensors in `self.graph` corresponding to arguments."""
    return self._func_graph.inputs

  @property
  def structured_input_signature(self):
    # [FUNCTION DOC]
    # Property returning the pair of positional and keyword argument specifications recorded
    # on the function graph when it was traced. No side effects.
    """Returns structured signature for this concrete function.

    Returns:
      A tuple `(args, kwargs)`, where:

        * `args` is a tuple that specifies the expected type or value each for
          positional argument.
        * `kwargs` is a dictionary that specifies the expected type or value
          for each keyword-only argument.

      The type or value for each argument is specified using one of the
      following:

        * A `tf.TypeSpec`, indicating that a Tensor or other TensorFlow-native
          value is expected.
        * A Python value, such as an integer, indicating that an equal value
          is expected.
        * A nested structure of `tf.TypeSpec`s and Python values, indicating
          that a corresponding nested structure is expected.
    """
    return self._func_graph.structured_input_signature

  @property
  def outputs(self):
    # [FUNCTION DOC]
    # Property returning the flat list of output tensors of the function graph.
    """Returns tensors in `self.graph` corresponding to returned tensors."""
    return self._func_graph.outputs

  @property
  def structured_outputs(self):
    # [FUNCTION DOC]
    # Property returning the outputs as structured by the original traced Python function.
    """Returns outputs in `self.graph` as returned by the original function."""
    return self._func_graph.structured_outputs

  def set_external_captures(self, captures):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring requires new values to be consistent with the original
    # captures, but nothing is validated (the code has a TODO for it) and the wrapped graph is
    # not updated; the list is simply stored.
    # Input: captures, list of tensors (value captures) or closures (call-time captures).
    # Replaces the stored captured inputs list. Returns nothing.
    """Updates the function capture values.

    The new values must have tensor types and shapes consistent with the
    original captures of the concrete function, but it is allowed to change a
    value captured with a deferred one and vice-versa.

    Args:
      captures: A list of tensors or closures. Tensors are value captures, and
        closures are call-time (deferred captures).
    """
    # TODO(wxinyi): 1. verify that the new captures' type spec is compatible
    # with the original's. However, doing so requires MirroredVariable captures
    # initialized. 2. replace the original/new captures/deferred
    # captures in the wrapped graph. Doing such for a capture-to-deferred
    # capture replacement requires more arguments than the deferred capture
    # itself, e.g. default value, spec.
    self._captured_inputs = captures

  def replace_capture_with_deferred_capture(self,
                                            tensor,
                                            closure,
                                            spec,
                                            placeholder=None,
                                            default_value=None):
    # [FUNCTION DOC]
    # Swaps an existing captured tensor for a deferred capture closure evaluated at call time.
    # Inputs: tensor, the currently captured tensor (found by identity in the captured inputs;
    # may be absent for functions restored from a saved model); closure, callable with no
    # arguments returning a structure of tensors; spec, type specification of the value;
    # placeholder, optional internal placeholder (inferred from the trailing graph inputs by
    # capture position when omitted); default_value, optional value for environments that
    # cannot evaluate the closure.
    # Raises a value error if no placeholder is given and the tensor is not among the captures,
    # or if the spec is compatible with neither the tensor nor the placeholder. Side effects:
    # updates the function graph's capture and, if the tensor was found, replaces its slot in
    # the captured inputs list with the closure (preserving order). Returns nothing.
    """Replaces existing capture `tensor` with a deferred capture `closure`.

    This API replaces the capture `tensor` from the concrete function's captured
    inputs list, and places the deferred capture `closure` in
    its spot so the order of captured inputs is preserved. This is important
    because the old `tensor` and the new `closure` will have the same internal
    placeholder, which can be passed through the `placeholder` argument, or
    skipped, in which case we find the placeholder from internal inputs by
    indexing `tensor` in the external captured inputs list. Thus, it is
    important that the new deferred capture has output spec (specified by the
    `spec` argument) compatible with the internal placeholder (`placeholder`)
    and the original capture (`tensor`).

    For example,

    ```python
    bool_captured_tensor = tf.constant(True)
    float_captured_tensor = tf.constant([3.], dtype=tf.float32)
    value = tf.constant([2.], dtype=tf.float32)

    @tf.function
    def fn():
      deferred_tensor = ops.get_default_graph().capture_call_time_value(
          lambda: value,
          tf.TensorSpec(shape=(1,), dtype=tf.float32))
      if bool_captured_tensor:
        return deferred_tensor
      else:
        return deferred_tensor + float_captured_tensor

    concrete_fn = fn.get_concrete_function()
    print(concrete_fn())  # tf.Tensor([2.], shape=(1,), dtype=float32)

    new_bool_captured_tensor = constant_op.constant(False)
    def bool_closure():
      return new_bool_captured_tensor

    concrete_fn.replace_capture_with_deferred_capture(
        bool_captured_tensor,
        bool_closure,
        spec=tensor_lib.TensorSpec(shape=(), dtype=dtypes.bool))

    print(concrete_fn())  # tf.Tensor([5.], shape=(1,), dtype=float32)
    ```

    Args:
      tensor: Tensor already captured. This `tensor` should be listed in
        concrete_function.captured_inputs except when it's empty such as when
        the concrete function is restored from SavedModel.
      closure: function which takes no arguments, to be evaluated at function
        call time, returning a nest of tensors compatible with `spec`.
      spec: nest of TypeSpec for the value to capture.
      placeholder: optional. The internal placeholder corresponding to the
        captured `tensor` and the new `closure`.
      default_value: optional value to use in environments that cannot safely
        evaluate closure.
    """
    capture_index = None
    for i, capture in enumerate(self._captured_inputs):
      if id(tensor) == id(capture):
        capture_index = i
        break

    if placeholder is None:
      if capture_index is None:
        raise ValueError(
            f"Did not find `tensor` argument {tensor} in the ConcreteFunction's"
            " captured inputs list, and did not receive a placeholder argument."
            " Thus we're unable to infer the internal placeholder. ")

      placeholder = self.inputs[-len(self._captured_inputs) + capture_index]

    if not (spec.is_compatible_with(tensor) or
            spec.is_compatible_with(placeholder)):
      raise ValueError(
          f"Attempting to substitute closure with spec {spec} that's "
          f"incompatible with the original capture {tensor} or the internal "
          f"placeholder {placeholder}.")

    self._func_graph.replace_capture_with_deferred_capture(
        tensor=tensor,
        closure=closure,
        spec=spec,
        placeholder=placeholder,
        default_value=default_value)

    if capture_index is not None:
      self._captured_inputs[capture_index] = closure

  @property
  def captured_inputs(self):
    # [FUNCTION DOC]
    # Property returning a flat list of captured tensors: each stored capture that is callable
    # (a deferred capture) is evaluated now, then everything is flattened including composite
    # tensors into components. Calling it may therefore run user closures.
    """Returns external Tensors captured by this function.

    self.__call__(*args) passes `args + self.captured_inputs` to the function.
    """
    return nest.flatten(
        [x() if callable(x) else x for x in self._captured_inputs],
        expand_composites=True)

  @property
  def function_def(self):
    # [FUNCTION DOC]
    # Property returning the cached function definition protocol buffer of the inference
    # function.
    """Returns a `FunctionDef` object representing this function."""
    return self._delayed_rewrite_functions.forward().cached_definition

  @property
  def output_shapes(self):
    # [FUNCTION DOC]
    # Property returning the structured outputs with each element replaced by its shape
    # (unknown-rank shape for elements without one); composite tensors are replaced by their
    # components first. Note the constructor also stores a flat output shapes tuple that this
    # property does not use.
    """The function's output shapes."""
    return nest.map_structure(
        lambda x: getattr(x, "shape", tensor_shape.TensorShape(None)),
        composite_tensor.replace_composites_with_components(
            self._func_graph.structured_outputs),
        expand_composites=False)

  @property
  def output_dtypes(self):
    # [FUNCTION DOC]
    # Property returning the structured outputs with each element replaced by its data type
    # (nothing for null outputs); composite tensors are replaced by components first. Marked
    # with a note that it may be removed. No docstring.
    # TODO(akshayka): Consider removing this.
    return nest.map_structure(
        lambda x: x.dtype if x is not None else None,
        composite_tensor.replace_composites_with_components(
            self._func_graph.structured_outputs),
        expand_composites=False)

  def add_to_graph(self, g=None, overwrite=False):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the overwrite parameter is documented but never used in the body.
    # Registers the inference function with a graph. Inputs: g, optional graph; when omitted
    # and not executing eagerly the default graph is used (in eager mode with no graph nothing
    # happens since the definition is added to the context at construction); overwrite, ignored.
    # Side effect: recursively adds the function and its dependencies to the graph. Returns
    # nothing.
    """Registers the function, adds it to the graph g or default graph.

    Args:
      g: If specified, registers the function with this graph. Defaults to the
        current context (either the default graph or the eager context).
      overwrite: A bool. If True, its forward function will overwrite
        any existing function of the same signature name in the graph `g`.
    """
    # If we are not executing eagerly, adds the function to default graph if no
    # graph is specified.
    # In case of eager execution, function definition gets added to context
    # during construction itself.

    if not context.executing_eagerly() and not g:
      g = ops.get_default_graph()

    if g is not None:
      g._add_function_recursive(self._delayed_rewrite_functions.forward())  # pylint: disable=protected-access

  def add_gradient_functions_to_graph(self, g=None):
    # [FUNCTION DOC]
    # KNOWN ISSUE: the docstring says it uses the current context when no graph is given, but
    # when executing eagerly with g omitted, g stays null and the next step fails with an
    # attribute error.
    # Input: g, optional graph (default graph is used when graph building). Adds the inference
    # function, the delayed-rewrite forward function and the backward function (building them if
    # needed) to the graph recursively. Returns nothing.
    """Add forward/backward functions to graph `g` or the current context."""
    if not context.executing_eagerly() and not g:
      g = ops.get_default_graph()
    g._add_function_recursive(self._delayed_rewrite_functions.forward())  # pylint: disable=protected-access
    forward_function, backward_function = (
        self._delayed_rewrite_functions.forward_backward())
    g._add_function_recursive(forward_function)  # pylint: disable=protected-access
    backward_function.add_to_graph(g)

  def _get_gradient_function(self):
    # [FUNCTION DOC]
    # HELPER: used by the flat call path when graph building.
    # KNOWN ISSUE: the docstring claims lazy creation at first call; it simply returns the
    # existing bound rewrite-and-call-backward method of the delayed-rewrite helper.
    # Returns that callback; no side effects.
    """Returns gradient function. It will be lazily created at first call."""
    return self._delayed_rewrite_functions._rewrite_forward_and_call_backward  # pylint: disable=protected-access

  def _select_forward_and_backward_functions(
      self, args, possible_gradient_type, executing_eagerly):
    # [FUNCTION DOC]
    # HELPER: used by the flat call path.
    # Inputs: args, flat list of all forward inputs (user plus captured); possible_gradient_type,
    # constant saying whether no tape, a single first-order tape, or higher-order situations
    # apply; executing_eagerly, boolean.
    # Computes input tangents from forward accumulators (eager only) and whether JVP gradients
    # are needed, builds a cache key from those, and returns a call-record object: for first
    # order with tangents or eager execution, a cached first-order tape helper; for first order
    # while graph building without tangents, the delayed-rewrite helper; for higher order, a
    # cached higher-order tape helper; with no tape, the delayed-rewrite helper with recording
    # off. Side effect: fills the per-function caches of tape helpers.
    """Selects forward and backward functions based on the calling context.

    The forward function computes the "real" function outputs, `self._outputs`,
    and any extra values needed by the corresponding backward function.

    Args:
      args: A flat list of Tensors with all of the inputs to the forward
        function (including user-specified and captured inputs).
      possible_gradient_type: One of gradients_util.POSSIBLE_GRADIENT_TYPES_*.
      executing_eagerly: Boolean, the value of context.executing_eagerly().

    Returns:
      An object with a `forward` method returning a tuple of (forward_function :
      AtomicFunction, augmented_arguments : List), and a corresponding
      `record` method which takes outputs from the forward function and records
      the operation. forward_function should be called with augmented_arguments.
    """
    if executing_eagerly:
      input_tangents = forwardprop_util.pack_tangents(args)
    else:
      input_tangents = forwardprop_util.TangentInfo()
    need_gradients_for_jvps = record.should_record_backprop(
        input_tangents.tangents)
    # Allows re-use of forward and backward function pairs depending on the
    # tapes and forward accumulators watching its inputs.
    cache_key = (need_gradients_for_jvps, input_tangents.indices)
    if (possible_gradient_type
        == gradients_util.POSSIBLE_GRADIENT_TYPES_FIRST_ORDER):
      if input_tangents.indices or executing_eagerly:
        # There is a single non-persistent tape active, so the user can only
        # request first-order gradients from a tape. We can spend less time
        # graph building since we know this.
        #
        # We may still end up computing higher-order gradients, but that'd be
        # through `tf.gradients`, which can re-write the forward pass and so
        # needs no preparation here.
        functions = self._first_order_tape_functions.get(cache_key, None)
        if functions is None:
          functions = _FirstOrderTapeGradientFunctions(
              self._func_graph, self._attrs, self._garbage_collector,
              forwardprop_input_indices=input_tangents.indices,
              delayed_rewrite_functions=self._delayed_rewrite_functions,
              need_gradients_for_jvps=need_gradients_for_jvps)
          self._first_order_tape_functions[cache_key] = functions
        return _ForwardBackwardCall(
            functions, args, input_tangents.tangents, tape_watching=True)
      else:
        # We can avoid computing second-order gradients in some cases by doing a
        # delayed rewrite when graph building. Since we know we'll only compute
        # first-order tape gradients, the delayed rewrite is safe: we won't need
        # to tell the tape about side outputs.
        #
        # TODO(allenl): This case is really dirty. It would be better if we
        # could temporarily pop all of the current tapes to avoid
        # accidentally taking second-order gradients.
        return _ForwardBackwardCall(
            self._delayed_rewrite_functions, args, input_tangents.tangents,
            tape_watching=True)
    elif (possible_gradient_type
          == gradients_util.POSSIBLE_GRADIENT_TYPES_HIGHER_ORDER):
      # Either there's a persistent tape watching, or there are multiple nested
      # tapes. Either way, the user may request higher-order gradients. We'll
      # spend a bit more time and make sure higher-order gradients are correct.
      functions = self._higher_order_tape_functions.get(
          cache_key, None)
      if functions is None:
        functions = _HigherOrderTapeGradientFunctions(
            self._func_graph, self._attrs, self._garbage_collector,
            forwardprop_input_indices=input_tangents.indices,
            delayed_rewrite_functions=self._delayed_rewrite_functions,
            need_gradients_for_jvps=need_gradients_for_jvps)
        self._higher_order_tape_functions[cache_key] = functions
      return _ForwardBackwardCall(functions, args, input_tangents.tangents,
                                  tape_watching=True)
    # else possible_gradient_type == POSSIBLE_GRADIENT_TYPES_NONE, meaning no
    # tape is recording.
    return _ForwardBackwardCall(
        self._delayed_rewrite_functions, args, input_tangents.tangents,
        tape_watching=False)

  @property
  def _as_name_attr_list(self):
    # [FUNCTION DOC]
    # Property returning a name-and-attributes protocol buffer for this function: its name plus
    # a copy of every function attribute. Used when referring to the function from operations.
    """Returns a `NameAttrList` representing this function."""
    ret = attr_value_pb2.NameAttrList(name=self.name)
    for name, value in self._attrs.items():
      ret.attr[name].CopyFrom(value)
    return ret

  def _flat_signature_summary(self):
    # [FUNCTION DOC]
    # HELPER: builds text for flat-signature error messages and repr.
    # KNOWN ISSUE: when there are more positional args than keyword names, it extends the stored
    # argument keywords list in place with placeholder names, mutating object state even though
    # it is documented as a pure summary.
    # Returns the graph name followed by the comma separated argument names in parentheses.
    # Asserts that the keywords and positional count were assigned; failing that raises an
    # assertion error.
    """Returns a string summarizing this function's flat signature."""
    assert self._arg_keywords is not None
    assert self._num_positional_args is not None
    arg_names = self._arg_keywords
    if self._num_positional_args > len(arg_names):
      arg_names.extend(
          "<arg{}>".format(i + 1)
          for i in range(len(arg_names), self._num_positional_args))
    return f"{self._func_graph.name}({', '.join(arg_names)})"

  def pretty_printed_signature(self, verbose=True):
    # [FUNCTION DOC]
    # Input: verbose, optional boolean, default true. Returns the representation text of the
    # function type when verbose, otherwise its shorter string form. Asserts that a function
    # type exists (assertion error otherwise).
    """Returns a string summarizing the signature of this concrete function."""
    assert self.function_type is not None
    if verbose:
      return repr(self.function_type)
    else:
      return str(self.function_type)

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns a developer-facing string: with a function type, the short signature and memory
    # address; otherwise, with a flat signature, its summary and address; otherwise the default
    # object representation.
    if self.function_type is not None:
      return "<ConcreteFunction {} at 0x{:X}>".format(
          self.pretty_printed_signature(verbose=False), id(self)
      )
    elif not (self._num_positional_args is None or self._arg_keywords is None):
      return "<ConcreteFunction {} at 0x{:X}>".format(
          self._flat_signature_summary(), id(self)
      )
    else:
      return object.__repr__(self)

  def __str__(self):
    # [FUNCTION DOC]
    # Returns a user-facing string: the verbose signature prefixed by the class name when a
    # function type exists, otherwise the same text as the developer representation.
    if self.function_type is not None:
      return "ConcreteFunction {}".format(
          self.pretty_printed_signature(verbose=True)
      )
    else:
      return self.__repr__()

  def _trackable_children(self, save_type="checkpoint", **kwargs):
    # [FUNCTION DOC]
    # Implements the trackable dependency listing. Inputs: save_type, string, default
    # checkpoint; extra keyword arguments ignored. For checkpoints returns an empty mapping
    # (functions are not part of checkpoint dependencies). Otherwise returns a mapping from
    # generated names (capture followed by position) to captured tensors, skipping variant and
    # resource dtypes and resource variables whose wrappers live elsewhere in the object graph.
    """Implements `Trackable`."""
    if save_type == "checkpoint":
      # Checkpoint dependencies do not include functions at all. Users
      # expect the checkpointed variables to be saved using the model
      # architecture, e.g. `model.layers[1].kernel` or `model.variables`.
      return {}

    captured_trackables = {}
    for n, (capture, _) in enumerate(self.graph.captures):
      if (capture.dtype not in (dtypes.variant, dtypes.resource) and
          not resource_variable_ops.is_resource_variable(capture)):
        # Variant/resource type tensors are skipped since we have no way of
        # getting the `Trackable` wrapper for these tensors. The wrappers are
        # expected to be elsewhere in the saved object graph.
        # TODO(b/223866972): Directly encode/decode tensor captures.

        # Resource variable captures are also skipped at this time, to maintain
        # existing behavior.
        # TODO(b/217979389): Return the non-constant captures as children.

        captured_trackables[f"capture_{n}"] = capture

    return captured_trackables

  def _deserialization_dependencies(self, children):
    # [FUNCTION DOC]
    # Trackable hook: input children, a mapping of dependencies; returns it unchanged, meaning
    # all children are needed before this object is deserialized.
    return children

  def _export_to_saved_model_graph(self, object_map, tensor_map,
                                   **unused_kwargs):
    # [FUNCTION DOC]
    # Saved-model export hook. Inputs: object_map, mapping from trackable objects to exported
    # objects (mutated); tensor_map, mapping of tensors passed to the exported wrapper; extra
    # keyword arguments ignored. Raises a value error listing the saving errors if the function
    # graph is not saveable. Otherwise adds the function to the current graph, stores an
    # exported concrete function wrapper for this object in object_map, and returns an empty
    # list of extra objects.
    if not self.graph.saveable:
      raise ValueError(
          (f"Unable to save function {self.name} for the following reason(s):\n"
           + "\n".join(self.graph.saving_errors)))
    self.add_to_graph()
    object_map[self] = saved_model_exported_concrete.ExportedConcreteFunction(
        self, tensor_map)
    return []


class ConcreteFunctionGarbageCollector:
  # [CLASS DOC]
  # HELPER: owner of a function graph on behalf of a concrete function. Holds a single
  # reference to the graph and, when garbage collected, dismantles the graph to break reference
  # cycles (making it unusable) unless release was called. Uses slots. Created by a concrete
  # function built with an unshared graph.
  """Cleans up reference cycles when a `ConcreteFunction` goes out of scope."""

  __slots__ = ["_func_graph"]

  def __init__(self, func_graph):
    # [FUNCTION DOC]
    # Stores the function graph to be dismantled later. Input: func_graph. Returns nothing.
    self._func_graph = func_graph

  def release(self):
    # [FUNCTION DOC]
    # Cancels the pending graph dismantling by dropping the reference to the graph. Takes no
    # inputs; returns nothing.
    """Call off the FuncGraph deletion."""
    self._func_graph = None

  def __del__(self):
    # [FUNCTION DOC]
    # Finalizer. If the module is still loaded (it may be nothing during interpreter shutdown)
    # and the graph was not released, dismantles the function graph to break reference cycles.
    # Any exception during dismantling is silently swallowed. Returns nothing.
    if func_graph_module is None or self._func_graph is None:
      return
    try:
      func_graph_module.dismantle_func_graph(self._func_graph)
    except:  # pylint: disable=bare-except
      pass


class _Marker(object):
  # [CLASS DOC]
  # HELPER: wrapper for a string so it prints without quotes when nested arguments are
  # pretty-printed in signatures.
  # KNOWN ISSUE: not referenced anywhere else in this module; appears to be leftover dead code.
  # Holds one string and uses slots.
  """Markers used to pretty-print nested args in function signatures."""

  __slots__ = ["_s"]

  def __init__(self, s):
    # [FUNCTION DOC]
    # Stores the string to display. Input: s, a string. Returns nothing.
    self._s = s

  def __repr__(self):
    # [FUNCTION DOC]
    # Returns the stored text as is, without quotes.
    return str(self._s)


def _contains_type_spec(value):
  # [FUNCTION DOC]
  # HELPER: type specification detector.
  # KNOWN ISSUE: not referenced anywhere else in this module; appears to be dead code.
  # Input: value, any nested structure. Flattens it and returns true if any leaf is a type
  # specification, else false. No side effects.
  return any(isinstance(x, type_spec.TypeSpec) for x in nest.flatten(value))
