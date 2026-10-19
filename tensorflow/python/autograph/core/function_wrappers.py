# [MODULE DOC]
# Runtime support for wrapping the body of AutoGraph-converted functions with auxiliary logic.
# Defines a function-scope context manager (optional TensorFlow name scope, optional automatic
# control dependencies, and tracking of the AutoGraph control status) plus a helper that runs a
# callback inside such a scope. Converted code uses it to open a scope on entry and route return
# values through its return method. No import-time side effects.
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
"""Support for wrapping converted functions bodies with auxiliary logic."""

from tensorflow.python.autograph.core import ag_ctx
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.operators import variables
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.util import nest


# TODO(mdan): Move this into operators - it represents a function definition.


class FunctionScope(object):
  # [CLASS DOC]
  # Context manager wrapped around the body of a converted function. Depending on the conversion
  # options it enters (a) an AutoGraph control status context marked enabled when the user
  # requested conversion, (b) a TensorFlow name scope named after the function when the name
  # scopes feature is on, and (c) an automatic control dependencies scope when that feature is on.
  # Attributes: name, options, callopts (options for nested calls), flags for which parts are in
  # use, and the inner scope objects. Offers a return method that marks returned tensors for
  # control dependencies. Entering and exiting enter and exit the inner scopes in the same order.
  """Context manager that wraps the body of a converted function.

  This context manager handles various operations related to the scope of a
  function:
    * optional TF name scopes - these name scopes match the name of the
        function, for easy visualization in tensorBoard;
    * optional automatic control dependencies - this adds the same mechanism
        for control dependencies that is used by `@tf.function`; it can be
        optionally enabled when using `tf.autograph.to_graph`;
    * tracking of autograph conversion state (whether it's enabled by the user,
        conversion options;
  """

  def __init__(self, function_name, scope_name, options):
    # [FUNCTION DOC]
    # Constructor. Inputs: function_name (string used to name the TensorFlow name scope after
    # sanitizing; may be nothing when name scopes are off), scope_name (stored as the name
    # attribute; not otherwise used) and options (conversion options). Stores options; if the
    # conversion was user requested, prepares an enabled control status context; computes the
    # options for nested calls; if the name scopes feature is used, creates a name scope
    # object; if the automatic control dependencies feature is used, creates that scope and
    # initializes the return-value-marked flag to false. Nothing is entered until the context
    # manager is used. Returns nothing.
    self.name = scope_name
    self.options = options

    if options.user_requested:
      self.autograph_ctx = ag_ctx.ControlStatusCtx(ag_ctx.Status.ENABLED,
                                                   options)
    self.callopts = options.call_options()

    use_name_scope = options.uses(converter.Feature.NAME_SCOPES)
    self.use_name_scope = use_name_scope
    if use_name_scope:
      self.name_scope = ops.name_scope(self._sanitize(function_name))

    use_auto_deps = self.options.uses(converter.Feature.AUTO_CONTROL_DEPS)
    self.use_auto_deps = use_auto_deps
    if use_auto_deps:
      self.autodeps_scope = auto_control_deps.AutomaticControlDependencies()
      self._return_value_marked = False

  def _sanitize(self, name):
    # [FUNCTION DOC]
    # HELPER: prepares a function name for use as a TensorFlow name scope.
    # Input: name, a string or nothing. If the name is non-empty and starts with an underscore,
    # returns it prefixed with the letters fn (TensorFlow rejects leading underscores at top
    # level); otherwise returns it unchanged. The docstring is only a link and does not describe
    # this behavior.
    """See https://www.tensorflow.org/api_docs/python/tf/Graph#name_scope."""
    # TensorFlow doesn't like leading underscores at the top level.
    if name and name.startswith('_'):
      name = 'fn' + name
    return name

  def __enter__(self):
    # [FUNCTION DOC]
    # Context entry. Enters, in order, the control status context (only if conversion was user
    # requested), the name scope (if used) and the automatic control dependencies scope (if used).
    # Returns this scope object.
    if self.options.user_requested:
      self.autograph_ctx.__enter__()
    if self.use_name_scope:
      self.name_scope.__enter__()
    if self.use_auto_deps:
      self.autodeps_scope.__enter__()
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    # [FUNCTION DOC]
    # Context exit. Inputs: the exception type, value and traceback (or nothing) which are
    # forwarded to each inner scope. Exits the control status context (if user requested), the
    # name scope (if used) and the control dependencies scope (if used), in the same order as
    # they were entered (not reversed). Returns nothing, so exceptions are not suppressed.
    if self.options.user_requested:
      self.autograph_ctx.__exit__(exc_type, exc_val, exc_tb)
    if self.use_name_scope:
      self.name_scope.__exit__(exc_type, exc_val, exc_tb)
    if self.use_auto_deps:
      self.autodeps_scope.__exit__(exc_type, exc_val, exc_tb)

  def ret(self, value, did_return):
    # [FUNCTION DOC]
    # Called by converted code on function exit to pass through the return value. Inputs: value
    # (the value being returned) and did_return (ignored). If the value is the special
    # undefined-return marker, returns nothing (null). If automatic control dependencies are in
    # use: records that a return value was marked; a null value is returned as is (no dummy
    # return is created); otherwise every tensor-like leaf in the (possibly nested) value is
    # passed through the control-dependencies scope's mark-as-return operation and the rebuilt
    # structure is returned. Without that feature the value is returned unchanged.
    """Marks a value as returned from the function guarded by the scope."""
    del did_return

    if isinstance(value, variables.UndefinedReturnValue):
      return None

    if self.use_auto_deps:
      self._return_value_marked = True
      if value is None:
        # We don't create dummy returns, to preserve Python semantics. The user
        # is responsible for adding a return value to the top-level function.
        return None

      def _mark_return_if_tensor(t):
        # [FUNCTION DOC]
        # HELPER: mapping function used by the return method.
        # Input: any leaf value. If it is a TensorFlow tensor type, returns the result of marking
        # it as a return in the control dependencies scope; otherwise returns it unchanged.
        if tensor_util.is_tf_type(t):
          return self.autodeps_scope.mark_as_return(t)
        return t

      value = nest.map_structure(_mark_return_if_tensor, value)
    return value


def with_function_scope(thunk, scope_name, options):
  # [FUNCTION DOC]
  # HELPER: inline form of the function scope context manager, for use in generated code.
  # Inputs: thunk (a callback taking the scope object), scope_name (stored name) and options
  # (conversion options). Opens a function scope using the fixed function name used for lambdas,
  # calls the callback with the scope inside it and returns the callback's result. Scope exit
  # happens afterwards.
  """Inline version of the FunctionScope context manager."""
  with FunctionScope('lambda_', scope_name, options) as scope:
    return thunk(scope)
