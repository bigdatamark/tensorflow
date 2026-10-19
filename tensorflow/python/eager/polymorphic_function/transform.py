# [MODULE DOC]
# Holds two module-level registries used by the tf.function implementation: a list of
# transformations applied to every newly built function graph, and a list of callbacks invoked on
# every newly created concrete function. Both lists start empty and are meant to be appended to by
# other modules; this module exposes two small functions that run all registered entries in order.
# Import-time side effects: only creation of the two empty lists (global mutable state).
# Not a helper for end users; it is internal glue between the tracing code and its extensions.
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
"""tf.function transformations implementation."""

# TODO(fmuham): Move this logic to core/function when layered.
# TODO(fmuham): Deprecate and migrate these as AtomicFunction transformations.
FUNC_GRAPH_TRANSFORMS = []
CONCRETE_FUNCTION_CALLBACKS = []


def apply_func_graph_transforms(func_graph):
  # [FUNCTION DOC]
  # HELPER: internal hook used by the function tracing code to run registered graph transforms.
  # Takes a function graph and calls each registered transformation, in registration order, with
  # that graph. Transformations are expected to modify the graph in place; return values are
  # ignored and nothing is returned. Exceptions raised by a transformation propagate and stop the
  # remaining ones. Reads the module-level transformation list.
  """Applies registered transformations to FuncGraph."""
  for transform in FUNC_GRAPH_TRANSFORMS:
    transform(func_graph)


def call_concrete_function_callbacks(concrete_fn):
  # [FUNCTION DOC]
  # HELPER: internal hook used by the function tracing code to notify registered listeners.
  # Takes a newly created concrete function and calls each registered callback, in registration
  # order, with it. Return values are ignored and nothing is returned; exceptions raised by a
  # callback propagate and stop the remaining callbacks. Reads the module-level callback list.
  """Calls registered callbacks against new ConcreteFunctions."""
  for callback in CONCRETE_FUNCTION_CALLBACKS:
    callback(concrete_fn)
