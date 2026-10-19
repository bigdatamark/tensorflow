# [MODULE DOC]
# Utilities that choose between two branches, preferring static (graph-construction time)
# evaluation of the predicate over building a dynamic conditional operation.
# Exposes smart_cond (an if/else that calls only one Python callable when the predicate is
# statically known, otherwise delegates to the graph conditional), smart_constant_value (extracts
# a constant boolean from a predicate if possible), and smart_case (a multi-branch version built
# on the shared case helper from the control-flow case module).
# Import-time side effect: registers smart_cond for export under the internal public API
# namespace (version 2 only). Used by Keras and other framework code.
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
"""smart_cond and related utilities."""

from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_case
from tensorflow.python.util.tf_export import tf_export


@tf_export("__internal__.smart_cond.smart_cond", v1=[])
def smart_cond(pred, true_fn=None, false_fn=None, name=None):
  # [FUNCTION DOC]
  # Returns the result of calling the true callable or the false callable depending on pred.
  # Inputs: pred is a scalar (Python bool, 0/1 integer, or boolean tensor); true_fn and false_fn
  # are callables taking no arguments (both default to nothing but both are required in practice);
  # name is an optional name scope used only when a dynamic conditional is built.
  # If the predicate's value is statically known, only the chosen callable is invoked directly
  # and its result returned; otherwise the graph conditional is built and invoked with both.
  # Raises a type error if either callable argument is not callable (checked before evaluating
  # the predicate), and propagates the type error from smart_constant_value for bad predicates.
  # Side effects: may add conditional operations to the current graph.
  """Return either `true_fn()` if predicate `pred` is true else `false_fn()`.

  If `pred` is a bool or has a constant value, we return either `true_fn()`
  or `false_fn()`, otherwise we use `tf.cond` to dynamically route to both.

  Args:
    pred: A scalar determining whether to return the result of `true_fn` or
      `false_fn`.
    true_fn: The callable to be performed if pred is true.
    false_fn: The callable to be performed if pred is false.
    name: Optional name prefix when using `tf.cond`.

  Returns:
    Tensors returned by the call to either `true_fn` or `false_fn`.

  Raises:
    TypeError: If `true_fn` or `false_fn` is not callable.
  """
  if not callable(true_fn):
    raise TypeError(f"Argument `true_fn` must be callable. Received {true_fn}")
  if not callable(false_fn):
    raise TypeError(
        f"Argument `false_fn` must be callable. Received {false_fn}")

  pred_value = smart_constant_value(pred)
  if pred_value is not None:
    if pred_value:
      return true_fn()
    else:
      return false_fn()
  else:
    return cond.cond(pred, true_fn=true_fn, false_fn=false_fn,
                     name=name)


def smart_constant_value(pred):
  # [FUNCTION DOC]
  # HELPER: Used by smart_cond to decide whether a predicate can be resolved statically.
  # KNOWN ISSUE: the docstring promises True/False/None, but for a tensor predicate the value
  # returned is whatever constant extraction yields (for example a numpy boolean scalar, or a
  # non-boolean constant), not guaranteed to be a Python bool. Also, the 0/1 membership test
  # runs before the bool check and raises a different type error for unhashable inputs.
  # Input: pred is a tensor, Python bool, or the integers 0 or 1.
  # For tensors it first tries static constant-value inference, then falls back to attempting
  # to evaluate the tensor's subgraph as a constant. Integers 0/1 are converted to bool.
  # Returns the constant value, or nothing if the tensor's value is dynamic.
  # Raises a type error for any other kind of predicate. No side effects.
  """Return the bool value for `pred`, or None if `pred` had a dynamic value.

  Args:
    pred: A scalar, either a Python bool or tensor.

  Returns:
    True or False if `pred` has a constant boolean value, None otherwise.

  Raises:
    TypeError: If `pred` is not a Tensor or bool.
  """
  if isinstance(pred, tensor.Tensor):
    pred_value = tensor_util.constant_value(pred)
    # TODO(skyewm): consider folding this into tensor_util.constant_value.
    # pylint: disable=protected-access
    if pred_value is None:
      pred_value = tensor_util.try_evaluate_constant(pred)
    # pylint: enable=protected-access
  elif pred in {0, 1}:  # Accept 1/0 as valid boolean values
    pred_value = bool(pred)
  elif isinstance(pred, bool):
    pred_value = pred
  else:
    raise TypeError("Argument `pred` must be a Tensor, or a Python bool, or 1 "
                    f"or 0. Received: pred={pred} of type "
                    f"{type(pred).__name__}")

  return pred_value


def smart_case(pred_fn_pairs, default=None, exclusive=False, name="smart_case"):
  # [FUNCTION DOC]
  # Multi-branch conditional like the graph case operation, but predicates that are Python bools
  # or constants are resolved statically using smart_cond as the per-branch conditional.
  # Inputs: pred_fn_pairs is a mapping or list of (predicate, callable) pairs; default is an
  # optional callable used when no predicate holds; exclusive requests that at most one
  # predicate be true (checked dynamically for non-constant predicates); name is the name scope.
  # Returns the outputs of the first branch whose predicate is true, else of the default.
  # Raises type errors for malformed pairs, non-callable branches or a bad container type; these
  # are raised by the shared helper. Delegates entirely to that helper, allowing Python
  # predicates. Side effects: may add operations to the current graph.
  """Like tf.case, except attempts to statically evaluate predicates.

  If any predicate in `pred_fn_pairs` is a bool or has a constant value, the
  associated callable will be called or omitted depending on its value.
  Otherwise this functions like tf.case.

  Args:
    pred_fn_pairs: Dict or list of pairs of a boolean scalar tensor and a
                   callable which returns a list of tensors.
    default: Optional callable that returns a list of tensors.
    exclusive: True iff at most one predicate is allowed to evaluate to `True`.
    name: A name for this operation (optional).

  Returns:
    The tensors returned by the first pair whose predicate evaluated to True, or
    those returned by `default` if none does.

  Raises:
    TypeError: If `pred_fn_pairs` is not a list/dictionary.
    TypeError: If `pred_fn_pairs` is a list but does not contain 2-tuples.
    TypeError: If `fns[i]` is not callable for any i, or `default` is not
               callable.
  """
  return control_flow_case._case_helper(  # pylint: disable=protected-access
      smart_cond,
      pred_fn_pairs,
      default,
      exclusive,
      name,
      allow_python_preds=True)
