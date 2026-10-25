# [MODULE DOC]
# Numeric-safety utilities for graph code: asserting that a tensor holds only finite values and
# adding finiteness checks to every floating point tensor of the default graph.
# Exposes the legacy (v1) and current (v2) finite-value assertion functions plus a graph-wide
# check-numerics helper (v1 only). All of them build on the check-numerics op and control
# dependencies. Import-time side effects: only the public-API export and dispatch/deprecation
# decorator registrations of the functions below.
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

"""Connects all half, float and double tensors to CheckNumericsOp."""

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.util import deprecation
from tensorflow.python.util import dispatch
from tensorflow.python.util.tf_export import tf_export


@tf_export(v1=["debugging.assert_all_finite", "verify_tensor_all_finite"])
@dispatch.add_dispatch_support
@deprecation.deprecated_endpoints("verify_tensor_all_finite")
def verify_tensor_all_finite(t=None, msg=None, name=None, x=None, message=None):
  # [FUNCTION DOC]
  # Deprecated v1 wrapper for asserting that a tensor contains no NaN or infinite values.
  # Inputs: t (tensor to check, optional only because x may be used instead), msg (failure
  # message), name (optional op name), x (deprecated alias for t), message (deprecated alias for
  # msg). The alias lookup picks the alias value or the primary one and complains if both are given.
  # Output: the same tensor, made to depend on the finiteness check. Raises an invalid-argument
  # error at run time if non-finite values exist. Side effects: deprecation warnings only.
  """Assert that the tensor does not contain any NaN's or Inf's.

  Args:
    t: Tensor to check.
    msg: Message to log on failure.
    name: A name for this operation (optional).
    x: Alias for t.
    message: Alias for msg.

  Returns:
    Same tensor as `t`.
  """
  x = deprecation.deprecated_argument_lookup("x", x, "t", t)
  message = deprecation.deprecated_argument_lookup(
      "message", message, "msg", msg)
  return verify_tensor_all_finite_v2(x, message, name)


@tf_export("debugging.assert_all_finite", v1=[])
@dispatch.add_dispatch_support
def verify_tensor_all_finite_v2(x, message, name=None):
  # [FUNCTION DOC]
  # Asserts that a tensor has no NaN or infinite values.
  # Inputs: x (tensor or value convertible to one), message (text reported on failure), name
  # (optional name scope, default "VerifyFinite").
  # Output: a tensor with the same value as x, colocated with x and gated by a control dependency
  # on a check-numerics op, so any consumer triggers the check. A run-time invalid-argument error
  # is raised when non-finite values are found. No other side effects beyond adding graph ops.
  """Assert that the tensor does not contain any NaN's or Inf's.

  >>> @tf.function
  ... def f(x):
  ...   x = tf.debugging.assert_all_finite(x, 'Input x must be all finite')
  ...   return x + 1

  >>> f(tf.constant([np.inf, 1, 2]))
  Traceback (most recent call last):
     ...
  InvalidArgumentError: ...

  Args:
    x: Tensor to check.
    message: Message to log on failure.
    name: A name for this operation (optional).

  Returns:
    Same tensor as `x`.
  """
  with ops.name_scope(name, "VerifyFinite", [x]) as name:
    x = ops.convert_to_tensor(x, name="x")
    with ops.colocate_with(x):
      verify_input = array_ops.check_numerics(x, message=message)
      out = control_flow_ops.with_dependencies([verify_input], x)
  return out


@tf_export(v1=["add_check_numerics_ops"])
def add_check_numerics_ops():
  # [FUNCTION DOC]
  # Adds a check-numerics op for every half, float or double output of every op in the default
  # graph, chaining them with control dependencies so each check runs after the previous ones
  # (relying on the graph's operation list being in creation order). The message of each check is
  # the op name plus the output index.
  # Takes no inputs. Returns a grouped op that depends on the last check (and thereby all).
  # Raises a runtime error when eager execution is enabled, and a value error if any such op lives
  # inside a control flow context (conditional or loop). Side effect: mutates the default graph.
  """Connect a `tf.debugging.check_numerics` to every floating point tensor.

  `check_numerics` operations themselves are added for each `half`, `float`,
  or `double` tensor in the current default graph. For all ops in the graph, the
  `check_numerics` op for all of its (`half`, `float`, or `double`) inputs
  is guaranteed to run before the `check_numerics` op on any of its outputs.

  Note: This API is not compatible with the use of `tf.cond` or
  `tf.while_loop`, and will raise a `ValueError` if you attempt to call it
  in such a graph.

  Returns:
    A `group` op depending on all `check_numerics` ops added.

  Raises:
    ValueError: If the graph contains any numeric operations in a control flow
      structure.
    RuntimeError: If called with eager execution enabled.

  @compatibility(eager)
  Not compatible with eager execution. To check for `Inf`s and `NaN`s under
  eager execution, call `tf.debugging.enable_check_numerics()` once before
  executing the checked operations.
  @end_compatibility
  """
  if context.executing_eagerly():
    raise RuntimeError(
        "add_check_numerics_ops() is not compatible with eager execution. "
        "To check for Inf's and NaN's under eager execution, call "
        "tf.debugging.enable_check_numerics() once before executing the "
        "checked operations.")

  check_op = []
  # This code relies on the ordering of ops in get_operations().
  # The producer of a tensor always comes before that tensor's consumer in
  # this list. This is true because get_operations() returns ops in the order
  # added, and an op can only be added after its inputs are added.
  for op in ops.get_default_graph().get_operations():
    for output in op.outputs:
      if output.dtype in [dtypes.float16, dtypes.float32, dtypes.float64]:
        if op._get_control_flow_context() is not None:  # pylint: disable=protected-access
          raise ValueError("`tf.add_check_numerics_ops() is not compatible "
                           "with TensorFlow control flow operations such as "
                           "`tf.cond()` or `tf.while_loop()`.")

        message = op.name + ":" + str(output.value_index)
        with ops.control_dependencies(check_op):
          check_op = [array_ops.check_numerics(output, message=message)]
  return control_flow_ops.group(*check_op)
