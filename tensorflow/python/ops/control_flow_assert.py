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
# [MODULE DOC]
# Defines the public assertion op for control flow (the Assert function, exported as the
# debugging Assert endpoint and the top-level Assert endpoint) plus a private helper that
# renders eager tensors as short text for error messages. It works in both eager mode (raises
# immediately) and graph mode (builds an assert operation, guarded by a conditional when the
# data may live on a device). Imports have no side effects beyond API export registration and
# dispatch / should-use-result decorators applied to Assert.
"""Assert functions for Control Flow Operations."""

from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_logging_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.util import dispatch
from tensorflow.python.util import tf_should_use
from tensorflow.python.util.tf_export import tf_export


def _summarize_eager(tensor, summarize=None):
  # [FUNCTION DOC]
  # HELPER: formats eager tensor data for the error message raised by Assert in eager mode.
  # Takes an eager tensor and an optional count of leading elements to include (default 3
  # when nothing is given; a negative value means include everything, by using the tensor's
  # size). For tensors with rank above zero it flattens the values and joins the first
  # elements as text with commas, appending an ellipsis marker if elements were cut off. For
  # scalars it prints the single value unless the count is zero, in which case the result is
  # empty. Returns the resulting string; no side effects.
  """Returns a summarized string representation of eager `tensor`.

  Args:
    tensor: EagerTensor to summarize
    summarize: Include these many first elements of `array`
  """
  # Emulate the behavior of Tensor::SummarizeValue()
  if summarize is None:
    summarize = 3
  elif summarize < 0:
    summarize = array_ops.size(tensor)

  # reshape((-1,)) is the fastest way to get a flat array view
  if tensor._rank():  # pylint: disable=protected-access
    flat = tensor.numpy().reshape((-1,))
    lst = [str(x) for x in flat[:summarize]]
    if len(lst) < flat.size:
      lst.append("...")
  else:
    # tensor.numpy() returns a scalar for zero dimensional arrays
    if gen_math_ops.not_equal(summarize, 0):
      lst = [str(tensor.numpy())]
    else:
      lst = []

  return ", ".join(lst)


# Assert and Print are special symbols in python, so we must
# use an upper-case version of them.
@tf_export("debugging.Assert", "Assert")
@dispatch.add_dispatch_support
@tf_should_use.should_use_result
def Assert(condition, data, summarize=None, name=None):
  # [FUNCTION DOC]
  # KNOWN ISSUE: the Raises section of the docstring actually holds a TF1 usage note, not an
  # exception description; the real exception is an invalid-argument error raised only in eager
  # mode. Also, the second eager-mode check after the guarded conditional is unreachable
  # in practice because eager execution already returned earlier.
  # Asserts that a condition holds. Inputs: condition (a boolean or boolean tensor), data (a
  # list of tensors or values to show on failure), summarize (optional number of entries to
  # print per tensor), name (optional op name). In eager mode, if the condition is false it
  # converts data to tensors, summarizes them and raises an invalid-argument error whose
  # message includes the condition and the summarized data; otherwise it returns nothing. In
  # graph mode it opens a name scope and converts data to tensors. If every tensor is of string
  # or 32-bit integer type (assumed to be host resident) it directly creates the logging assert
  # op and returns it. Otherwise it converts the condition to a tensor and builds a conditional
  # named AssertGuard that does nothing when true and runs the assert op when false, returning
  # the guard's underlying operation. The result is wrapped so unused results trigger a warning.
  """Asserts that the given condition is true.

  If `condition` evaluates to false, print the list of tensors in `data`.
  `summarize` determines how many entries of the tensors to print.

  Args:
    condition: The condition to evaluate.
    data: The tensors to print out when condition is false.
    summarize: Print this many entries of each tensor.
    name: A name for this operation (optional).

  Returns:
    assert_op: An `Operation` that, when executed, raises a
    `tf.errors.InvalidArgumentError` if `condition` is not true.
    @compatibility(eager)
    returns None
    @end_compatibility

  Raises:
    @compatibility(TF1)
    When in TF V1 mode (that is, outside `tf.function`) Assert needs a control
    dependency on the output to ensure the assertion executes:

  ```python
  # Ensure maximum element of x is smaller or equal to 1
  assert_op = tf.Assert(tf.less_equal(tf.reduce_max(x), 1.), [x])
  with tf.control_dependencies([assert_op]):
    ... code using x ...
  ```

    @end_compatibility
  """
  if context.executing_eagerly():
    if not condition:
      xs = ops.convert_n_to_tensor(data)
      data_str = [_summarize_eager(x, summarize) for x in xs]
      raise errors.InvalidArgumentError(
          node_def=None,
          op=None,
          message="Expected '%s' to be true. Summarized data: %s" %
          (condition, "\n".join(data_str)))
    return

  with ops.name_scope(name, "Assert", [condition, data]) as name:
    xs = ops.convert_n_to_tensor(data)
    if all(x.dtype in {dtypes.string, dtypes.int32} for x in xs):
      # As a simple heuristic, we assume that string and int32 are
      # on host to avoid the need to use cond. If it is not case,
      # we will pay the price copying the tensor to host memory.
      return gen_logging_ops._assert(condition, data, summarize, name="Assert")  # pylint: disable=protected-access
    else:
      condition = ops.convert_to_tensor(condition, name="Condition")

      def true_assert():
        # [FUNCTION DOC]
        # HELPER: branch callback for the conditional guard inside Assert.
        # Takes no arguments; creates and returns the logging assert op using the enclosing
        # condition, data and summarize values, named Assert. Only executed when the guarded
        # condition is false.
        return gen_logging_ops._assert(  # pylint: disable=protected-access
            condition, data, summarize, name="Assert")

      guarded_assert = cond.cond(
          condition,
          gen_control_flow_ops.no_op,
          true_assert,
          name="AssertGuard")
      if context.executing_eagerly():
        return
      return guarded_assert.op
