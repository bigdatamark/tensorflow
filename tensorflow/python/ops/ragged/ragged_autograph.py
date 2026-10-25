# [MODULE DOC]
# HELPER: Internal glue that lets autograph-converted for loops iterate over ragged tensors.
# Registers an overload of the autograph for-statement operator for RaggedTensor. A converted
# loop over a ragged tensor iterates over its outermost rows by index, implemented on top of the
# autograph while-statement operator. Importing this module has the side effect of registering
# the overload in the autograph control-flow loop registry; it exposes no public names.
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
"""Autograph-specific overrides for ragged_tensor."""
from tensorflow.python.autograph.operators import control_flow
from tensorflow.python.ops import cond as tf_cond
from tensorflow.python.ops.ragged import ragged_tensor


def _tf_ragged_for_stmt(
    iter_, extra_test, body, get_state, set_state, symbol_names, opts
):
  # [FUNCTION DOC]
  # HELPER: Autograph for-statement overload for ragged tensors, registered at module import.
  # Runs a converted for loop over the rows of a ragged tensor as a graph while loop. Inputs:
  # iter_ is the ragged tensor to iterate; extra_test is an optional callback returning a
  # boolean that can end the loop early (nothing if absent); body is a callback taking one row;
  # get_state and set_state read and write the loop variables; symbol_names names those
  # variables; opts is a mapping of loop options (a maximum-iterations hint is added to it).
  # The number of iterations is the static first dimension when known, otherwise the first
  # entry of the row lengths. Returns nothing; results flow through set_state. Raises whatever
  # the loop-variable initialization verifier raises if loop variables are undefined or have
  # inconsistent types. Side effects: mutates opts and invokes the user callbacks.
  """Overload of for_stmt that iterates over TF ragged tensors."""
  init_vars = get_state()
  control_flow.verify_loop_init_vars(init_vars, symbol_names)

  # TODO(mdan): Move this into len()? Requires eager support.
  if iter_.shape and iter_.shape[0] is not None:
    n = iter_.shape[0]
  else:
    n = iter_.row_lengths()[0]

  iterate_index = 0

  def aug_get_state():
    # [FUNCTION DOC]
    # HELPER: Augmented state getter for the row-iteration while loop.
    # Returns the loop variables of the user loop as a tuple preceded by the current iteration
    # index. Takes no inputs and has no side effects.
    return (iterate_index,) + get_state()

  def aug_set_state(aug_loop_vars):
    # [FUNCTION DOC]
    # HELPER: Augmented state setter for the row-iteration while loop.
    # Takes the augmented tuple of loop variables (iteration index first, then user variables),
    # stores the index in the enclosing function's iteration counter and forwards the remaining
    # user variables to the original state setter. Returns nothing. Side effect: rebinds the
    # enclosing iteration index and updates the user's loop state.
    nonlocal iterate_index
    # TODO(b/171479293): Drop the lint override.
    iterate_index, *loop_vars = aug_loop_vars  # pylint:disable=unused-variable
    # The iteration index is not "output" by the for loop. If the iteration
    # index is used outside the loop, it will appear
    # in the loop vars separately.
    set_state(loop_vars)

  def aug_body():
    # [FUNCTION DOC]
    # HELPER: Loop body wrapper for the row-iteration while loop.
    # Calls the user body with the ragged tensor row at the current iteration index, then
    # increments the index. Takes no inputs, returns nothing; side effect is advancing the
    # enclosing iteration counter and whatever the user body does.
    nonlocal iterate_index
    body(iter_[iterate_index])
    iterate_index += 1

  def aug_test():
    # [FUNCTION DOC]
    # HELPER: Loop condition for the row-iteration while loop.
    # Returns a boolean (scalar tensor or value) that is true while the iteration index is below
    # the row count; when an extra test callback was supplied the result is additionally
    # conditioned on it through a graph conditional. Takes no inputs and has no side effects
    # beyond evaluating the extra test.
    main_test = iterate_index < n
    if extra_test is not None:
      return tf_cond.cond(main_test, extra_test, lambda: False)
    return main_test

  control_flow._add_max_iterations_hint(opts, n)  # pylint: disable=protected-access

  control_flow._tf_while_stmt(  # pylint: disable=protected-access
      aug_test,
      aug_body,
      aug_get_state,
      aug_set_state,
      ('<internal iterate>',) + symbol_names,
      opts,
  )


control_flow.for_loop_registry.register(
    ragged_tensor.RaggedTensor, _tf_ragged_for_stmt
)
