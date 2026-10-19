# [MODULE DOC]
# Shim layer used when deserializing saved models: it re-binds the tensors/variables/resources that
# a traced function originally captured back into its concrete function. Exposes
# get_tensor_from_node (resolve a restored object into a capturable tensor) and restore_captures
# (inject captures into a concrete function and refresh its function type). No import-time side
# effects beyond imports. Not a pure helper module; it is part of the saved-model load path.
# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""A shim layer for working with functions exported/restored from saved models.

This functionality should ultimately be moved into a first-class core API.
"""

import warnings

from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import resource


def get_tensor_from_node(node):
  # [FUNCTION DOC]
  # HELPER: converts restored saved-model objects into tensors for restore_captures.
  # KNOWN ISSUE: the docstring says it returns a list of tensors, but it returns a single object
  # (a tensor, or the distributed/sharded variable or table itself), not a list.
  # Input: node, a restored variable, distributed variable/table, sharded variable, asset,
  # tensor, or capturable resource. Runs under the init scope (outside any function graph).
  # Distributed variables, tables and sharded variables are returned unchanged; resource
  # variables yield their handle; assets yield their path; tensor-like values are returned
  # unchanged; capturable resources yield their resource handle (which may execute restored
  # functions as a side effect). Raises a value error if the node is none of these.
  """Resolves a saved model graph node into a tensor to be captured.

  Args:
    node: a tensor, variable, or resource to be resolved into a capturable
      tensor

  Returns:
    A list of tensors.
  Raises:
    ValueError: if the node cannot be converted into a tensor.
  """
  with ops.init_scope():
    # TODO(b/210144904): Use __tf_tensor__ instead of `is_[...]` checks
    if getattr(node, "is_distributed_variable", False):
      return node
    elif getattr(node, "is_distributed_table", False):
      return node
    elif getattr(node, "is_sharded_variable", False):
      return node
    elif resource_variable_ops.is_resource_variable(node):
      return node.handle
    elif isinstance(node, asset.Asset):
      return node.asset_path
    elif tensor_util.is_tf_type(node):
      return node
    elif isinstance(node, resource.CapturableResource):
      # Note: this executes restored functions in the CapturableResource.
      return node.resource_handle
    raise ValueError(f"Cannot convert node {node} to tensor.")


def restore_captures(concrete_function, inputs):
  # [FUNCTION DOC]
  # Re-attaches the originally captured tensors to a deserialized concrete function.
  # Inputs: concrete_function, the function being restored (mutated in place); inputs, a list of
  # tensors or objects (variables, resources, assets, distributed values) that were captured.
  # Each input is resolved to a tensor; variables are recorded on the function. The bound inputs
  # are matched with the trailing placeholder inputs of the function graph. Inputs that define
  # a restore-capture hook delegate to it; otherwise the graph capture is replaced with the real
  # tensor, resource handle data is copied for resource-typed placeholders, and the capture is
  # registered. A warning is emitted if any restored capture is null (sharded variables loaded
  # without a distribution strategy). Finally the external captures are set and, if the function
  # has a function type, it is rebuilt to include the new capture types. Returns nothing.
  """Restore captures for the concrete function.

  Used at deserialization time.  For functions that are being deserialized,
  saved model restores objects that tensors were captured from, but functions
  only know about their tensors -- object information is destroyed by tracing.
  This additional logic extracts the tensors which the function originally
  captured.

  Args:
    concrete_function: the concrete function for which to restore captures
    inputs: a list tensors or other Python objects (such as variables) which
      contain tensors that were originally captured by the function
  """
  bound_inputs = [get_tensor_from_node(obj) for obj in inputs]
  # pylint: disable=g-complex-comprehension
  bound_variables = [
      obj
      for obj in inputs
      if isinstance(
          obj,
          (variables_lib.Variable, resource_variable_ops.BaseResourceVariable),
      )
  ]
  # TODO(b/205010575): This is only injecting the captured inputs into the
  # concrete function, note that we did not modify the FuncGraph
  # itself.
  captured_inputs_list = []
  concrete_function.set_variables(bound_variables)
  if bound_inputs:
    for bound_input, internal_capture in zip(
        bound_inputs, concrete_function.inputs[-len(bound_inputs) :]
    ):
      # Distributed inputs have special logic for capturing, so we call their
      # custom restoration methods
      if hasattr(bound_input, "__tf_experimental_restore_capture__"):
        captured_inputs_list.append(
            bound_input.__tf_experimental_restore_capture__(
                concrete_function, internal_capture
            )
        )
      else:
        captured_inputs_list.append(bound_input)
        concrete_function.graph.replace_capture(bound_input, internal_capture)
        if internal_capture.dtype == dtypes.resource:
          if resource_variable_ops.is_resource_variable(bound_input):
            try:
              handle = bound_input.handle
            except ValueError:
              # For mirrored variables we'll copy handle data for components
              # as they get captured.
              pass
            else:
              handle_data_util.copy_handle_data(handle, internal_capture)
          else:
            # TODO(b/213451747): Remove need to call copy_handle_data
            handle_data_util.copy_handle_data(bound_input, internal_capture)
        # Setting "captures" first means "capture" won't create a new
        # placeholder for this input.
        concrete_function.graph.capture(bound_input)

  if any([inp is None for inp in captured_inputs_list]):
    warnings.warn(
        "Trying to load ShardedVariables using tf.saved_model.load. "
        "This won't work if using a tf.distribute.Strategy, and may "
        "use excess memory if not using a Strategy. Ignore this "
        "warning if using tf.keras.models.load_model."
    )
  concrete_function.set_external_captures(captured_inputs_list)

  # Update FunctionType with new captures.
  if concrete_function.function_type:
    concrete_function._function_type = function_type_lib.FunctionType(  # pylint: disable=protected-access
        concrete_function.function_type.parameters.values(),
        concrete_function.graph.function_captures.capture_types,
        return_annotation=concrete_function.function_type.output,
    )
