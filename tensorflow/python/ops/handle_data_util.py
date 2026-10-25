# [MODULE DOC]
# HELPER: internal helper module used by function/graph/eager code to move resource and variant
# handle metadata between tensors.
# KNOWN ISSUE: the module docstring says "Decorator to overrides the gradient for a function", which
# is a copy-paste leftover; the module contains no decorator and no gradient code.
# Utilities for reading, copying, setting and creating "handle data" (the shapes and dtypes of the
# elements held by resource and variant tensors). Handle data is kept either on the eager tensor
# itself or inside the C graph for symbolic tensors. Exposes get_resource_handle_data (also exported
# as an internal API), get_handle_data, copy_handle_data, set_handle_data and create_handle_data.
# Importing the module has no side effects other than the export registration of the one
# exported function.
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
"""Decorator to overrides the gradient for a function."""

from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.types import core
from tensorflow.python.util import compat
from tensorflow.python.util.tf_export import tf_export


@tf_export("__internal__.ops.get_resource_handle_data", v1=[])
def get_resource_handle_data(graph_op):
  # [FUNCTION DOC]
  # Reads the resource/variant handle data attached to a symbolic (graph) tensor from the C graph
  # and returns it as a parsed handle-data message. Input graph_op is a symbolic tensor; an
  # assertion error is raised if it is not a symbolic tensor or if it is an eager value. The
  # graph's C handle is locked while the shape-and-type bytes are fetched. Returns the parsed
  # handle data message (empty if nothing was recorded). No mutation of arguments.
  assert (isinstance(graph_op, core.Symbol)
          and not isinstance(graph_op, core.Value))

  with graph_op.graph._c_graph.get() as c_graph:  # pylint: disable=protected-access
    handle_data = pywrap_tf_session.GetHandleShapeAndType(
        c_graph, graph_op._as_tf_output())  # pylint: disable=protected-access

  return cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData.FromString(
      compat.as_bytes(handle_data))


def get_handle_data(source_t):
  # [FUNCTION DOC]
  # HELPER: dispatcher used by copy_handle_data to read handle data from either tensor flavor.
  # Takes a tensor. For an eager value it returns the handle data stored on the tensor object; for
  # a symbolic tensor it queries the graph via get_resource_handle_data. Returns the handle data
  # message. No side effects.
  """Obtains HandleData from a tensor."""
  if isinstance(source_t, core.Value):
    return source_t._handle_data  # pylint: disable=protected-access
  return get_resource_handle_data(source_t)


def copy_handle_data(source_t, target_t):
  # [FUNCTION DOC]
  # Copies handle data from source_t to target_t, but only when target_t has a resource or variant
  # dtype; for any other dtype it does nothing. The source's handle data is read with
  # get_handle_data and applied with set_handle_data (which itself ignores missing or unset data).
  # Returns nothing. Side effect: mutates target_t (its eager attribute or its graph metadata).
  """Copies HandleData for variant and resource type tensors if available.

  The CppShapeInferenceResult::HandleData proto contains information about the
  shapes and types of the element tensors of resource/variant type tensors.
  We need to copy this across function boundaries, i.e., when capturing a
  placeholder or when returning a function tensor as output. If we don't do this
  the element tensors will have unknown shapes, e.g., if a TensorList variant
  tensor is captured as a placeholder, elements popped from that list would have
  unknown shape.

  Args:
    source_t: The tensor to copy HandleData from.
    target_t: The tensor to copy HandleData to.
  """
  if (target_t.dtype == dtypes.resource or
      target_t.dtype == dtypes.variant):
    handle_data = get_handle_data(source_t)
    set_handle_data(target_t, handle_data)


def set_handle_data(target_t, handle_data):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the docstring has a typo ("giver" instead of "given"); behavior is unaffected.
  # Attaches handle data to target_t. If handle_data is null, not marked as set, or has no
  # shape-and-type entries, it returns silently without changing anything. For an eager value the
  # handle data object is stored directly on the tensor; for a symbolic tensor it is serialized
  # and written into the C graph for that tensor's output. Returns nothing; mutates target_t.
  """Sets handle data on the giver tensor."""
  if (
      handle_data is None
      or not handle_data.is_set
      or not handle_data.shape_and_type
  ):
    return

  # pylint: disable=protected-access
  if isinstance(target_t, core.Value):
    target_t._handle_data = handle_data
    return
  with target_t.graph._c_graph.get() as c_graph:
    pywrap_tf_session.SetHandleShapeAndType(c_graph, target_t._as_tf_output(),
                                            handle_data.SerializeToString())
  # pylint: enable=protected-access


def create_handle_data(shape, dtype):
  # [FUNCTION DOC]
  # Builds a new handle-data message that is marked as set and holds exactly one shape-and-type
  # entry. Inputs: shape is a tensor-shape object (converted to its proto form) and dtype is a
  # TensorFlow dtype (converted to its enum value). Returns the new message; no side effects.
  handle_data = cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData()
  handle_data.is_set = True
  handle_data.shape_and_type.append(
      cpp_shape_inference_pb2.CppShapeInferenceResult.HandleShapeAndType(
          shape=shape.as_proto(), dtype=dtype.as_datatype_enum))
  return handle_data
