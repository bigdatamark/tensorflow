# [MODULE DOC]
# Implements importing a serialized graph definition into the current default graph (the public
# import-graph-definition API) plus a variant for function bodies that skips colocation
# validation. Validates and canonicalises arguments, optionally strips producer default
# attributes, delegates the import to the native C API, converts native errors to value errors,
# post-processes new operations (devices, colocation) and registers imported functions.
# Many private helpers support this. Import-time side effects: registers the public API
# names via the export decorator; defines an OSS flag (set by the build tooling) that selects
# serialized or direct graph definition passing to the native layer.

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
"""A utility function for importing TensorFlow graphs."""
import contextlib

from tensorflow.core.framework import graph_pb2
from tensorflow.python import tf2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.framework import c_api_util
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import control_flow_util
from tensorflow.python.util import compat
from tensorflow.python.util.deprecation import deprecated_args
from tensorflow.python.util.tf_export import tf_export


# TODO(b/307794935): Remove after bug is fixed.
is_oss = True  # Updated by copybara.


def _IsControlInput(input_name):
  # [FUNCTION DOC]
  # HELPER: classifies input names while importing graphs.
  # Returns True if the input name string starts with a caret (a control-dependency input).
  # Note it appears unused in this module.
  # Expected format: '^operation_name' (control input).
  return input_name.startswith('^')


def _ParseTensorName(tensor_name):
  # [FUNCTION DOC]
  # HELPER: splits tensor names for the input-map and return-element handling during import.
  # Input: a tensor name string, either an operation name alone (output index zero) or an
  # operation name, a colon and an integer output index. Returns a pair of the operation name and
  # the integer index. Raises a value error if the part after the colon is not an integer or if
  # the name has more than one colon.
  """Parses a tensor name into an operation name and output index.

  This function will canonicalize tensor names as follows:

  * "foo:0"       -> ("foo", 0)
  * "foo:7"       -> ("foo", 7)
  * "foo"         -> ("foo", 0)
  * "foo:bar:baz" -> ValueError

  Args:
    tensor_name: The name of a tensor.

  Returns:
    A tuple containing the operation name, and the output index.

  Raises:
    ValueError: If `tensor_name' cannot be interpreted as the name of a tensor.
  """
  components = tensor_name.split(':')
  if len(components) == 2:
    # Expected format: 'operation_name:output_index'.
    try:
      output_index = int(components[1])
    except ValueError:
      raise ValueError(f'Cannot convert {tensor_name!r} to a tensor name. '
                       'Second component of the name following the `:` should '
                       f'be an int. Got {components[1]}.')
    return components[0], output_index
  elif len(components) == 1:
    # Expected format: 'operation_name' (implicit 0th output).
    return components[0], 0
  else:
    raise ValueError(f"Cannot convert '{tensor_name}' to a tensor name. Tensor "
                     'names should not contain more than 1 `:`. Obtained '
                     f'{len(components) - 1}')


@contextlib.contextmanager
def _MaybeDevice(device):
  # [FUNCTION DOC]
  # HELPER: context manager used when post-processing imported operations.
  # Input: a device specification (string or similar) or nothing/empty. Enters a device scope
  # for the given device for the duration of the managed block; if the device is empty or
  # nothing, enters no scope. Yields nothing.
  """Applies the given device only if device is not None or empty."""
  if device:
    with ops.device(device):
      yield
  else:
    yield


def _ProcessGraphDefParam(graph_def):
  # [FUNCTION DOC]
  # HELPER: argument validation for the graph import entry point.
  # Input: a graph definition protobuf or a duck-typed message. A non-standard message is merged
  # into a new graph definition (a type error is raised if that fails). A standard graph
  # definition is used as is, and its nodes are MODIFIED IN PLACE to fill in default attribute
  # values from registered op definitions (unknown ops, presumably functions, are skipped).
  # Output: the graph definition to import. Side effect: caller-visible mutation described above.
  """Type-checks and possibly canonicalizes `graph_def`."""
  if not isinstance(graph_def, graph_pb2.GraphDef):
    # `graph_def` could be a dynamically-created message, so try a duck-typed
    # approach
    try:
      old_graph_def = graph_def
      graph_def = graph_pb2.GraphDef()
      graph_def.MergeFrom(old_graph_def)
    except TypeError:
      raise TypeError('Argument `graph_def` must be a GraphDef proto.')
  else:
    # If we're using the graph_def provided by the caller, modify graph_def
    # in-place to add attr defaults to the NodeDefs (this is visible to the
    # caller).
    # NOTE(skyewm): this is undocumented behavior that at least meta_graph.py
    # depends on. It might make sense to move this to meta_graph.py and have
    # import_graph_def not modify the graph_def argument (we'd have to make sure
    # this doesn't break anything else.)
    for node in graph_def.node:
      op_def = op_def_registry.get(node.op)
      if op_def is None:
        # Assume unrecognized ops are functions for now. TF_ImportGraphDef will
        # report an error if the op is actually missing.
        continue
      _SetDefaultAttrValues(node, op_def)

  return graph_def


def _ProcessInputMapParam(input_map):
  # [FUNCTION DOC]
  # HELPER: argument validation for the graph import entry point.
  # Input: the input mapping or nothing. Returns an empty mapping for nothing, otherwise the same
  # mapping. Raises a type error if it is not a dictionary or if any key is not a string (text
  # or bytes). Does not validate the values here.
  """Type-checks and possibly canonicalizes `input_map`."""
  if input_map is None:
    input_map = {}
  else:
    if not isinstance(input_map, dict):
      raise TypeError('Argument `input_map` must be a dictionary. Obtained '
                      f'{type(input_map).__name__}')
    if not all(
        isinstance(k, compat.bytes_or_text_types) for k in input_map.keys()):
      raise TypeError('All keys for argument `input_map` must be strings. '
                      f'Obtained keys: {list(input_map.keys())}')
  return input_map


def _ProcessReturnElementsParam(return_elements):
  # [FUNCTION DOC]
  # HELPER: argument validation for the graph import entry point.
  # Input: a list of names (text or bytes) or nothing. Returns nothing for nothing, otherwise a
  # tuple of the names converted to text. Raises a type error if any element is not a string.
  """Type-checks and possibly canonicalizes `return_elements`."""
  if return_elements is None:
    return None
  if not all(
      isinstance(x, compat.bytes_or_text_types) for x in return_elements):
    raise TypeError('Argument `return_elements` must be a list of strings. '
                    f'Obtained {return_elements}.')
  return tuple(compat.as_str(x) for x in return_elements)


def _FindAttrInOpDef(attr_name, op_def):
  # [FUNCTION DOC]
  # HELPER: attribute lookup used when stripping producer default attributes.
  # Inputs: an attribute name and an op definition protobuf. Returns the matching attribute
  # definition, or nothing if the op definition has no attribute with that name.
  for attr_def in op_def.attr:
    if attr_name == attr_def.name:
      return attr_def
  return None


def _RemoveDefaultAttrs(producer_op_list, graph_def):
  # [FUNCTION DOC]
  # HELPER: lets graphs from newer producers be imported by older consumers.
  # Inputs: an op list protobuf describing the producer's ops, and a graph definition. For each
  # node whose op is in the producer list and registered locally, removes attributes that the
  # local op definition does not know but whose value equals the producer's default. The graph
  # definition is MODIFIED IN PLACE. Ops not registered locally are skipped. Returns nothing.
  """Removes unknown default attrs according to `producer_op_list`.

  Removes any unknown attrs in `graph_def` (i.e. attrs that do not appear in
  registered OpDefs) that have a default value in `producer_op_list`.

  Args:
    producer_op_list: OpList proto.
    graph_def: GraphDef proto
  """
  producer_op_dict = {op.name: op for op in producer_op_list.op}
  for node in graph_def.node:
    # Remove any default attr values that aren't in op_def.
    if node.op in producer_op_dict:
      op_def = op_def_registry.get(node.op)
      if op_def is None:
        # Some custom op registrations won't show up here. That's OK, attribute
        # stripping just won't be available.
        continue
      producer_op_def = producer_op_dict[node.op]
      # We make a copy of node.attr to iterate through since we may modify
      # node.attr inside the loop.
      for key in list(node.attr):
        if _FindAttrInOpDef(key, op_def) is None:
          # No attr_def in consumer, look in producer.
          attr_def = _FindAttrInOpDef(key, producer_op_def)
          if (attr_def and attr_def.HasField('default_value') and
              node.attr[key] == attr_def.default_value):
            # Unknown attr had default value in producer, delete it so it can be
            # understood by consumer.
            del node.attr[key]


def _ConvertInputMapValues(name, input_map):
  # [FUNCTION DOC]
  # HELPER: prepares the input mapping for the graph import entry point.
  # Inputs: the name prefix given to the import (string) and the input mapping. If every value is
  # already a tensor, returns the mapping unchanged. Otherwise converts all values to tensors
  # inside a name scope for inputs and returns a new mapping. Raises a value error if conversion
  # is needed but the name is the empty string. Should be called inside the import name scope.
  """Ensures all input map values are tensors.

  This should be called from inside the import name scope.

  Args:
    name: the `name` argument passed to import_graph_def
    input_map: the `input_map` argument passed to import_graph_def.

  Returns:
    An possibly-updated version of `input_map`.

  Raises:
    ValueError: if input map values cannot be converted due to empty name scope.
  """
  if not all(isinstance(v, tensor.Tensor) for v in input_map.values()):
    if name == '':  # pylint: disable=g-explicit-bool-comparison
      raise ValueError(
          'tf.import_graph_def() requires a non-empty `name` if `input_map` '
          'contains non-Tensor values. Try calling tf.convert_to_tensor() on '
          '`input_map` values before calling tf.import_graph_def().')
    with ops.name_scope('_inputs'):
      input_map = {k: ops.convert_to_tensor(v) for k, v in input_map.items()}
  return input_map


def _PopulateTFImportGraphDefOptions(options, prefix, input_map,
                                     return_elements,
                                     validate_colocation_constraints,
                                     propagate_device_spec=False):
  # [FUNCTION DOC]
  # HELPER: translates Python import arguments into native import options.
  # Inputs: the native options object (modified in place), name prefix, input mapping (from
  # source names to tensors; names starting with a caret remap control dependencies to
  # operations), list of return element names (names with a colon are outputs, others are
  # operations), a flag for validating colocation constraints, and a flag (default false) for
  # propagating device specs. Always enables name uniquification. Returns nothing. Raises a
  # value error from name parsing for malformed tensor names.
  """Populates the TF_ImportGraphDefOptions `options`."""
  c_api.TF_ImportGraphDefOptionsSetPrefix(options, prefix)
  c_api.TF_ImportGraphDefOptionsSetUniquifyNames(options, True)
  c_api.TF_ImportGraphDefOptionsSetPropagateDeviceSpec(options,
                                                       propagate_device_spec)

  for input_src, input_dst in input_map.items():
    input_src = compat.as_str(input_src)
    if input_src.startswith('^'):
      src_name = compat.as_str(input_src[1:])
      dst_op = input_dst._as_tf_output().oper  # pylint: disable=protected-access
      c_api.TF_ImportGraphDefOptionsRemapControlDependency(
          options, src_name, dst_op)
    else:
      src_name, src_idx = _ParseTensorName(input_src)
      src_name = compat.as_str(src_name)
      dst_output = input_dst._as_tf_output()  # pylint: disable=protected-access
      c_api.TF_ImportGraphDefOptionsAddInputMapping(options, src_name, src_idx,
                                                    dst_output)
  for name in return_elements or []:
    if ':' in name:
      op_name, index = _ParseTensorName(name)
      op_name = compat.as_str(op_name)
      c_api.TF_ImportGraphDefOptionsAddReturnOutput(options, op_name, index)
    else:
      c_api.TF_ImportGraphDefOptionsAddReturnOperation(options,
                                                       compat.as_str(name))

  c_api.TF_ImportGraphDefOptionsSetValidateColocationConstraints(
      options, validate_colocation_constraints)


def _ProcessNewOps(graph):
  # [FUNCTION DOC]
  # HELPER: post-import fix-up of newly created operations.
  # Input: the graph that just received imported operations. Registers the new operations,
  # clears their devices, then for ops without colocation applies the graph's device functions
  # (keeping the original device as a scope), and for ops with colocation derives the device
  # from the first colocated op that has one. Missing colocated ops raise a value error unless
  # TF2 behavior or control flow v2 is enabled, in which case they are ignored. Mutates the
  # graph's operations; returns nothing.
  # KNOWN ISSUE: a source comment questions why colocation overrides the original device; the
  # behavior is intentional but unexplained.
  """Processes the newly-added TF_Operations in `graph`."""
  # Maps from a node to the names of the ops it's colocated with, if colocation
  # is specified in the attributes.
  colocation_pairs = {}

  for new_op in graph._add_new_tf_operations(compute_devices=False):  # pylint: disable=protected-access
    original_device = new_op.device
    new_op._set_device('')  # pylint: disable=protected-access
    colocation_names = _GetColocationNames(new_op)
    if colocation_names:
      colocation_pairs[new_op] = colocation_names
      # Don't set a device for this op, since colocation constraints override
      # device functions and the original device. Note that this op's device may
      # still be set by the loop below.
      # TODO(skyewm): why does it override the original device?
    else:
      with _MaybeDevice(original_device):
        graph._apply_device_functions(new_op)  # pylint: disable=protected-access

  # The following loop populates the device field of ops that are colocated
  # with another op.  This is implied by the colocation attribute, but we
  # propagate the device field for completeness.
  for op, coloc_op_list in colocation_pairs.items():
    coloc_device = None
    # Find any device in the list of colocated ops that have a device, if it
    # exists.  We assume that if multiple ops have devices, they refer to the
    # same device.  Otherwise, a runtime error will occur since the colocation
    # property cannot be guaranteed.  Note in TF2 colocations have been removed
    # from the public API and will be considered a hint, so there is no runtime
    # error.
    #
    # One possible improvement is to try to check for compatibility of all
    # devices in this list at import time here, which would require
    # implementing a compatibility function for device specs in python.
    for coloc_op_name in coloc_op_list:
      try:
        coloc_op = graph._get_operation_by_name(coloc_op_name)  # pylint: disable=protected-access
      except KeyError:
        # Do not error in TF2 if the colocation cannot be guaranteed
        if tf2.enabled() or control_flow_util.EnableControlFlowV2(graph):
          continue

        raise ValueError(f'Specified colocation to an op: {coloc_op_name} that '
                         f'does not exist during import for op: {op.name}')
      if coloc_op.device:
        coloc_device = pydev.DeviceSpec.from_string(coloc_op.device)
        break
    if coloc_device:
      op._set_device(coloc_device)  # pylint: disable=protected-access


def _GetColocationNames(op):
  # [FUNCTION DOC]
  # HELPER: colocation lookup for the post-import operation fix-up.
  # Input: an operation. Returns a list of names of other ops it must be colocated with, taken
  # from its class attribute entries starting with the colocation prefix, excluding itself.
  # Returns nothing (not an empty list) if the op has no class attribute.
  """Returns names of the ops that `op` should be colocated with."""
  colocation_names = []
  try:
    class_values = op.get_attr('_class')
  except ValueError:
    # No _class attr
    return
  for val in class_values:
    val = compat.as_str(val)
    if val.startswith('loc:@'):
      colocation_node_name = val[len('loc:@'):]
      if colocation_node_name != op.name:
        colocation_names.append(colocation_node_name)
  return colocation_names


def _GatherReturnElements(requested_return_elements, graph, results):
  # [FUNCTION DOC]
  # HELPER: builds the return value of the graph import entry point.
  # Inputs: the requested names (strings; those with a colon are tensors, others operations), the
  # graph, and the native import results. Returns a list of operations and tensors in the order
  # requested, taking tensors and operations from the native results in order. No mutation.
  """Returns the requested return elements from results.

  Args:
    requested_return_elements: list of strings of operation and tensor names
    graph: Graph
    results: wrapped TF_ImportGraphDefResults

  Returns:
    list of `Operation` and/or `Tensor` objects
  """
  return_outputs = c_api.TF_ImportGraphDefResultsReturnOutputs(results)
  return_opers = c_api.TF_ImportGraphDefResultsReturnOperations(results)

  combined_return_elements = []
  outputs_idx = 0
  opers_idx = 0
  for name in requested_return_elements:
    if ':' in name:
      combined_return_elements.append(
          graph._get_tensor_by_tf_output(return_outputs[outputs_idx]))  # pylint: disable=protected-access
      outputs_idx += 1
    else:
      combined_return_elements.append(
          graph._get_operation_by_tf_operation(return_opers[opers_idx]))  # pylint: disable=protected-access
      opers_idx += 1
  return combined_return_elements


def _SetDefaultAttrValues(node_def, op_def):
  # [FUNCTION DOC]
  # HELPER: fills in missing attributes during graph definition preprocessing.
  # Inputs: a node definition (modified in place) and the matching op definition. For every
  # attribute with a default in the op definition that is absent or unset in the node, copies
  # the default into the node. Fails an assertion if the op names do not match. Returns nothing.
  """Set any default attr values in `node_def` that aren't present."""
  assert node_def.op == op_def.name
  for attr_def in op_def.attr:
    key = attr_def.name
    if attr_def.HasField('default_value'):
      value = node_def.attr[key]
      if value is None or value.WhichOneof('value') is None:
        node_def.attr[key].CopyFrom(attr_def.default_value)


@tf_export('graph_util.import_graph_def', 'import_graph_def')
@deprecated_args(None, 'Please file an issue at '
                 'https://github.com/tensorflow/tensorflow/issues if you depend'
                 ' on this feature.', 'op_dict')
def import_graph_def(graph_def,
                     input_map=None,
                     return_elements=None,
                     name=None,
                     op_dict=None,
                     producer_op_list=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring's Returns section mentions a "returns_elements" name that does not
  # exist (the parameter is return_elements), and op_dict is accepted but ignored (deleted).
  # Public (exported, deprecated-argument) entry point that imports a graph definition into the
  # current default graph. Inputs: graph definition, optional input mapping, optional list of
  # names to return, optional name prefix, ignored op_dict, optional producer op list.
  # Output: list of operations/tensors for the return names, or nothing if none were requested.
  # Raises type errors for bad argument types and value errors for unknown names or malformed
  # graphs. Side effects: adds operations (and functions) to the default graph and may modify
  # the supplied graph definition in place (defaults filled, producer defaults stripped).
  # Delegates to the internal import function.
  """Imports the graph from `graph_def` into the current default `Graph`.

  This function provides a way to import a serialized TensorFlow
  [`GraphDef`](https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto)
  protocol buffer, and extract individual objects in the `GraphDef` as
  `tf.Tensor` and `tf.Operation` objects. Once extracted,
  these objects are placed into the current default `Graph`. See
  `tf.Graph.as_graph_def` for a way to create a `GraphDef`
  proto.

  Args:
    graph_def: A `GraphDef` proto containing operations to be imported into
      the default graph.
    input_map: A dictionary mapping input names (as strings) in `graph_def`
      to `Tensor` objects. The values of the named input tensors in the
      imported graph will be re-mapped to the respective `Tensor` values.
    return_elements: A list of strings containing operation names in
      `graph_def` that will be returned as `Operation` objects; and/or
      tensor names in `graph_def` that will be returned as `Tensor` objects.
    name: (Optional.) A prefix that will be prepended to the names in
      `graph_def`. Note that this does not apply to imported function names.
      Defaults to `"import"`.
    op_dict: (Optional.) Deprecated, do not use.
    producer_op_list: (Optional.) An `OpList` proto with the (possibly stripped)
      list of `OpDef`s used by the producer of the graph. If provided,
      unrecognized attrs for ops in `graph_def` that have their default value
      according to `producer_op_list` will be removed. This will allow some more
      `GraphDef`s produced by later binaries to be accepted by earlier binaries.

  Returns:
    A list of `Operation` and/or `Tensor` objects from the imported graph,
    corresponding to the names in `return_elements`,
    and None if `returns_elements` is None.

  Raises:
    TypeError: If `graph_def` is not a `GraphDef` proto,
      `input_map` is not a dictionary mapping strings to `Tensor` objects,
      or `return_elements` is not a list of strings.
    ValueError: If `input_map`, or `return_elements` contains names that
      do not appear in `graph_def`, or `graph_def` is not well-formed (e.g.
      it refers to an unknown tensor).
  """
  del op_dict
  return _import_graph_def_internal(
      graph_def,
      input_map=input_map,
      return_elements=return_elements,
      name=name,
      producer_op_list=producer_op_list)


def import_graph_def_for_function(  # pylint: disable=invalid-name
    graph_def, name=None, propagate_device_spec=False):
  # [FUNCTION DOC]
  # Variant of the graph import entry point used for function bodies: imports a graph definition
  # into the default graph without validating colocation constraints and without an input map,
  # return elements or producer op list. Inputs: graph definition, optional name prefix, and
  # optional flag (default false) to propagate device specs. Returns nothing meaningful (the
  # internal call returns nothing when no return elements are requested, and it is returned).
  # Raises the same errors as the internal import.
  """Like import_graph_def but does not validate colocation constraints."""
  return _import_graph_def_internal(
      graph_def,
      validate_colocation_constraints=False,
      name=name,
      propagate_device_spec=propagate_device_spec)


def _import_graph_def_internal(  # pylint: disable=invalid-name
    graph_def,
    input_map=None,
    return_elements=None,
    validate_colocation_constraints=True,
    name=None,
    producer_op_list=None,
    propagate_device_spec=False):
  # [FUNCTION DOC]
  # Core implementation of graph import. Steps: validate and canonicalise the graph definition,
  # input mapping and return elements; optionally strip producer default attributes; open a name
  # scope to compute the import prefix; convert input map values to tensors; fill native import
  # options; under the graph's mutation lock, serialise (or pass directly, depending on the OSS
  # flag) and import via the native API, converting invalid-argument errors to value errors and
  # always freeing the native buffer; post-process new operations; register imported functions
  # with the graph; raise a value error if any input mapping key was unused (likely a typo).
  # Inputs: graph definition, input mapping, return element names, colocation validation flag
  # (default true), name prefix, producer op list, device spec propagation flag (default false).
  # Output: list of operations/tensors matching the return names, or nothing if none requested.
  # Side effects: mutates the default graph and possibly the supplied graph definition.
  """Imports the graph from `graph_def` into the current default `Graph`.

  This function provides a way to import a serialized TensorFlow
  [`GraphDef`](https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto)
  protocol buffer, and extract individual objects in the `GraphDef` as
  `tf.Tensor` and `tf.Operation` objects. Once extracted,
  these objects are placed into the current default `Graph`. See
  `tf.Graph.as_graph_def` for a way to create a `GraphDef`
  proto.

  Args:
    graph_def: A `GraphDef` proto containing operations to be imported into the
      default graph.
    input_map: A dictionary mapping input names (as strings) in `graph_def` to
      `Tensor` objects. The values of the named input tensors in the imported
      graph will be re-mapped to the respective `Tensor` values.
    return_elements: A list of strings containing operation names in `graph_def`
      that will be returned as `Operation` objects; and/or tensor names in
      `graph_def` that will be returned as `Tensor` objects.
    validate_colocation_constraints: Whether to validate colocation constraints.
    name: (Optional.) A prefix that will be prepended to the names in
      `graph_def`. Note that this does not apply to imported function names.
      Defaults to `"import"`.
    producer_op_list: (Optional.) An `OpList` proto with the (possibly stripped)
      list of `OpDef`s used by the producer of the graph. If provided,
      unrecognized attrs for ops in `graph_def` that have their default value
      according to `producer_op_list` will be removed. This will allow some more
      `GraphDef`s produced by later binaries to be accepted by earlier binaries.
    propagate_device_spec: Whether to propagate assigned device information
      when importing a graph from a GraphDef into the current default `Graph`.

  Returns:
    A list of `Operation` and/or `Tensor` objects from the imported graph,
    corresponding to the names in `return_elements`,
    and None if `returns_elements` is None.

  Raises:
    TypeError: If `graph_def` is not a `GraphDef` proto,
      `input_map` is not a dictionary mapping strings to `Tensor` objects,
      or `return_elements` is not a list of strings.
    ValueError: If `input_map`, or `return_elements` contains names that
      do not appear in `graph_def`, or `graph_def` is not well-formed (e.g.
      it refers to an unknown tensor).
  """
  graph_def = _ProcessGraphDefParam(graph_def)
  input_map = _ProcessInputMapParam(input_map)
  return_elements = _ProcessReturnElementsParam(return_elements)

  if producer_op_list is not None:
    # TODO(skyewm): make a copy of graph_def so we're not mutating the argument?
    _RemoveDefaultAttrs(producer_op_list, graph_def)

  graph = ops.get_default_graph()
  with ops.name_scope(name, 'import', input_map.values()) as scope:
    # Save unique prefix generated by name_scope
    if scope:
      assert scope.endswith('/')
      prefix = scope[:-1]
    else:
      prefix = ''

    # Generate any input map tensors inside name scope
    input_map = _ConvertInputMapValues(name, input_map)

  scoped_options = c_api_util.ScopedTFImportGraphDefOptions()
  options = scoped_options.options
  _PopulateTFImportGraphDefOptions(options, prefix, input_map, return_elements,
                                   validate_colocation_constraints,
                                   propagate_device_spec)

  # _ProcessNewOps mutates the new operations. _mutation_lock ensures a
  # Session.run call cannot occur between creating the TF_Operations in the
  # TF_GraphImportGraphDefWithResults call and mutating them in _ProcessNewOps.
  with graph._mutation_lock():  # pylint: disable=protected-access
    if is_oss:
      graph_def_input = c_api.TF_NewBufferFromString(
          compat.as_bytes(graph_def.SerializeToString())
      )
      graph_import_graphdef = c_api.TF_GraphImportGraphDefWithResults
    else:
      graph_def_input = graph_def
      graph_import_graphdef = (
          c_api.TF_GraphImportGraphDefWithResultsNoSerialization
      )
    try:
      with graph._c_graph.get() as c_graph:  # pylint: disable=protected-access
        results = graph_import_graphdef(c_graph, graph_def_input, options)
      results = c_api_util.ScopedTFImportGraphDefResults(results)
    except errors.InvalidArgumentError as e:
      # Convert to ValueError for backwards compatibility.
      raise ValueError(str(e))
    finally:
      if is_oss:
        c_api.TF_DeleteBuffer(graph_def_input)

    # Create _DefinedFunctions for any imported functions.
    #
    # We do this by creating _DefinedFunctions directly from `graph_def`, and
    # adding them to `graph`. Adding an existing function to a TF_Graph is a
    # no-op, so this only has the effect of updating the Python state (usually
    # _DefinedFunction.add_to_graph also adds the function to the TF_Graph).
    #
    # TODO(skyewm): fetch the TF_Functions directly from the TF_Graph
    # TODO(skyewm): avoid sending serialized FunctionDefs back to the TF_Graph

    _ProcessNewOps(graph)

  if graph_def.library and graph_def.library.function:
    functions = function.from_library(graph_def.library)
    for f in functions:
      f.add_to_graph(graph)

  # Treat input mappings that don't appear in the graph as an error, because
  # they are likely to be due to a typo.
  missing_unused_input_keys = (
      c_api.TF_ImportGraphDefResultsMissingUnusedInputMappings_wrapper(
          results.results))
  if missing_unused_input_keys:
    missing_unused_input_keys = [
        compat.as_str(s) for s in missing_unused_input_keys
    ]
    missing_keys = ', '.join(missing_unused_input_keys)
    raise ValueError(
        'Attempted to map inputs that were not found in graph_def: '
        f'[{missing_keys}]')

  if return_elements is None:
    return None
  else:
    return _GatherReturnElements(return_elements, graph, results.results)
